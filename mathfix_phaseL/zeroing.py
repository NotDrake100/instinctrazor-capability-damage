"""Zero a trained LoRA delta on chosen decoder layers without retraining.

For every PEFT LoraLayer whose name matches ``layers.{L}.`` with L in the
zero-set, both ``lora_A`` and ``lora_B`` weights are set to 0. Then
``dW = scaling * B @ A = 0`` for all 7 target modules in that layer.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

import torch
from peft.tuners.lora import LoraLayer

Snap = Dict[str, Dict[Tuple[str, str], torch.Tensor]]


def layer_index(module_name: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", module_name)
    return int(m.group(1)) if m else None


def snapshot_lora_weights(model) -> Snap:
    snaps: Snap = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, LoraLayer):
            continue
        snaps[name] = {}
        for a in mod.lora_A:
            snaps[name][("A", a)] = mod.lora_A[a].weight.detach().clone()
            snaps[name][("B", a)] = mod.lora_B[a].weight.detach().clone()
    return snaps


@torch.no_grad()
def restore_lora_weights(model, snaps: Snap) -> None:
    for name, mod in model.named_modules():
        if name not in snaps:
            continue
        for a in mod.lora_A:
            mod.lora_A[a].weight.copy_(snaps[name][("A", a)])
            mod.lora_B[a].weight.copy_(snaps[name][("B", a)])


@torch.no_grad()
def zero_lora_delta_layers(model, layers: Iterable[int]) -> int:
    """Zero LoRA A/B in the given decoder layers. Returns #modules touched."""
    want = set(int(x) for x in layers)
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, LoraLayer):
            continue
        li = layer_index(name)
        if li is None or li not in want:
            continue
        for a in mod.lora_A:
            mod.lora_A[a].weight.zero_()
            mod.lora_B[a].weight.zero_()
            n += 1
    return n


def lora_layer_ids(model) -> List[int]:
    ids = set()
    for name, mod in model.named_modules():
        if isinstance(mod, LoraLayer):
            li = layer_index(name)
            if li is not None:
                ids.add(li)
    return sorted(ids)
