"""CPU fake-quant for the Qwen2.5-0.5B critical-layer pre-reg.

Round-to-nearest, per-group asymmetric affine quant, group size 128,
dequantized back to fp32. Not InstinctRazor's symmetric/MSE-clip MoE grid.
See PREREG_qwen05b_critical_layer.md.
"""
from __future__ import annotations

import os
from typing import Iterable

import torch
import torch.nn as nn

GROUP_SIZE = 128


def _group_size(in_dim: int, group: int) -> int:
    if group and in_dim % group == 0:
        return group
    return in_dim


@torch.no_grad()
def fakequant_asymmetric(w: torch.Tensor, bits: int, group: int = GROUP_SIZE) -> torch.Tensor:
    """Snap `w` to a B-bit asymmetric per-group grid and dequant to the same dtype/shape.

    Grouping is along the last dimension. If that dim is not divisible by
    `group`, the entire last dim is one group (InstinctRazor fallback).
    Constant groups (max == min) are left untouched.
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    if bits >= 16:
        return w.clone() if not w.is_contiguous() else w

    orig_dtype = w.dtype
    shp = w.shape
    in_dim = shp[-1]
    g = _group_size(in_dim, group)
    wb = w.reshape(-1, g).float()
    wmin = wb.min(dim=-1, keepdim=True).values
    wmax = wb.max(dim=-1, keepdim=True).values
    span = wmax - wmin
    constant = span <= 0
    qmax = (1 << bits) - 1
    scale = torch.where(constant, torch.ones_like(span), span / qmax)
    zp = torch.round(-wmin / scale).clamp(0, qmax)
    q = torch.clamp(torch.round(wb / scale + zp), 0, qmax)
    deq = (q - zp) * scale
    deq = torch.where(constant, wmin.expand_as(wb), deq)
    return deq.reshape(shp).to(orig_dtype)


def iter_block_linears(model: nn.Module):
    """Yield (block_idx, local_name, linear) for every Linear in model.model.layers."""
    layers = model.model.layers
    for i, block in enumerate(layers):
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                yield i, name, mod


def block_linear_names(model: nn.Module, block_idx: int) -> list[str]:
    return [name for i, name, _ in iter_block_linears(model) if i == block_idx]


@torch.no_grad()
def snapshot_block_linears(model: nn.Module) -> dict[tuple[int, str], torch.Tensor]:
    """CPU clones of every block Linear weight. Biases are not snapshotted."""
    out: dict[tuple[int, str], torch.Tensor] = {}
    for i, name, mod in iter_block_linears(model):
        out[(i, name)] = mod.weight.detach().cpu().contiguous().clone()
    return out


@torch.no_grad()
def apply_block_quant(
    model: nn.Module,
    bits: int,
    originals: dict[tuple[int, str], torch.Tensor],
    protect: Iterable[int] = (),
    group: int = GROUP_SIZE,
) -> dict:
    """In-place fake-quant of all block Linears except those in `protect`.

    Protected blocks are restored from `originals` (so a second call can
    switch which block is protected without reloading the model).
    Embeddings, lm_head, norms, and biases are never written.
    """
    protect_set = set(int(x) for x in protect)
    n_quant = 0
    n_prot = 0
    for i, name, mod in iter_block_linears(model):
        key = (i, name)
        src = originals[key]
        if i in protect_set:
            mod.weight.copy_(src.to(device=mod.weight.device, dtype=mod.weight.dtype))
            n_prot += 1
        else:
            q = fakequant_asymmetric(src, bits, group)
            mod.weight.copy_(q.to(device=mod.weight.device, dtype=mod.weight.dtype))
            n_quant += 1
    return {"n_quantized_tensors": n_quant, "n_protected_tensors": n_prot, "protect": sorted(protect_set), "bits": bits}


@torch.no_grad()
def restore_all_block_linears(model: nn.Module, originals: dict[tuple[int, str], torch.Tensor]) -> None:
    for i, name, mod in iter_block_linears(model):
        src = originals[(i, name)]
        mod.weight.copy_(src.to(device=mod.weight.device, dtype=mod.weight.dtype))


class DiskWeightBank:
    """Per-block Linear weight snapshots on disk so the live model is the only RAM copy."""

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.n_layers = 0
        self.names_by_layer: dict[int, list[str]] = {}

    def _path(self, i: int) -> str:
        return os.path.join(self.directory, f"block_{i}.pt")

    @torch.no_grad()
    def capture(self, model: nn.Module) -> None:
        n = len(model.model.layers)
        for i in range(n):
            payload: dict[str, torch.Tensor] = {}
            for name, mod in model.model.layers[i].named_modules():
                if isinstance(mod, nn.Linear):
                    payload[name] = mod.weight.detach().cpu().contiguous().clone()
            torch.save(payload, self._path(i))
            self.names_by_layer[i] = list(payload.keys())
            del payload
        self.n_layers = n

    def load_block(self, i: int) -> dict[str, torch.Tensor]:
        return torch.load(self._path(i), map_location="cpu", weights_only=True)

    @torch.no_grad()
    def write_block(self, model: nn.Module, i: int, bits: int | None, group: int = GROUP_SIZE) -> None:
        """Write original (`bits is None`) or fake-quantized weights into block `i`."""
        payload = self.load_block(i)
        for name, mod in model.model.layers[i].named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            src = payload[name]
            if bits is None:
                tensor = src
            else:
                tensor = fakequant_asymmetric(src, bits, group)
            mod.weight.copy_(tensor.to(device=mod.weight.device, dtype=mod.weight.dtype))
        del payload

    @torch.no_grad()
    def apply(self, model: nn.Module, bits: int, protect: Iterable[int] = (), group: int = GROUP_SIZE) -> dict:
        protect_set = set(int(x) for x in protect)
        n_quant = n_prot = 0
        for i in range(self.n_layers):
            if i in protect_set:
                self.write_block(model, i, bits=None, group=group)
                n_prot += len(self.names_by_layer.get(i, []))
            else:
                self.write_block(model, i, bits=bits, group=group)
                n_quant += len(self.names_by_layer.get(i, []))
        return {"n_quantized_tensors": n_quant, "n_protected_tensors": n_prot, "protect": sorted(protect_set), "bits": bits}
