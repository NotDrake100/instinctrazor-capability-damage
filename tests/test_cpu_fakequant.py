"""Unit tests for the CPU asymmetric fake-quant (PREREG_qwen05b_critical_layer.md)."""
from __future__ import annotations

import os
import sys
import unittest

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from cpu_fakequant import (  # noqa: E402
    DiskWeightBank,
    apply_block_quant,
    fakequant_asymmetric,
    snapshot_block_linears,
)


class Block(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=True)
        self.mlp = nn.Linear(d, 2 * d, bias=False)

    def forward(self, x):
        return self.mlp(self.q_proj(x))


class Inner(nn.Module):
    def __init__(self, n=3, d=128, vocab=16):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([Block(d) for _ in range(n)])
        self.norm = nn.LayerNorm(d)


class Tiny(nn.Module):
    def __init__(self, n=3, d=128, vocab=16):
        super().__init__()
        self.model = Inner(n, d, vocab)
        self.lm_head = nn.Linear(d, vocab, bias=False)


def _per_group_scale(w: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    g = group if w.shape[-1] % group == 0 else w.shape[-1]
    wb = w.reshape(-1, g)
    qmax = (1 << bits) - 1
    return (wb.max(dim=-1).values - wb.min(dim=-1).values) / qmax


class TestFakeQuantRoundTrip(unittest.TestCase):
    def test_error_within_one_lsb(self):
        torch.manual_seed(0)
        w = torch.randn(64, 256)
        for bits in (2, 3, 4, 8):
            wq = fakequant_asymmetric(w, bits, 128)
            scale = _per_group_scale(w, bits, 128)
            err = (w.reshape(-1, 128) - wq.reshape(-1, 128)).abs().max(dim=-1).values
            # RTN + integer zp: reconstruction error is at most 1 LSB (scale).
            ok = (scale == 0) | (err <= scale + 1e-5)
            self.assertTrue(bool(ok.all()), f"bits={bits} max_err={err.max()} max_scale={scale.max()}")

    def test_idempotent(self):
        torch.manual_seed(1)
        w = torch.randn(16, 256)
        wq = fakequant_asymmetric(w, 3, 128)
        wqq = fakequant_asymmetric(wq, 3, 128)
        self.assertTrue(torch.allclose(wq, wqq, atol=1e-6))

    def test_constant_group_untouched(self):
        w = torch.full((4, 128), 3.25)
        wq = fakequant_asymmetric(w, 2, 128)
        self.assertTrue(torch.allclose(w, wq))

    def test_nondivisible_last_dim_is_one_group(self):
        torch.manual_seed(2)
        w = torch.randn(8, 100)
        wq = fakequant_asymmetric(w, 4, 128)
        self.assertEqual(wq.shape, w.shape)
        scale = (w.max() - w.min()) / 15
        self.assertLessEqual(float((w - wq).abs().max()), float(scale) + 1e-5)

    def test_high_bits_noop(self):
        w = torch.randn(8, 128)
        wq = fakequant_asymmetric(w, 16, 128)
        self.assertTrue(torch.allclose(w, wq))


class TestProtectedBlockUntouched(unittest.TestCase):
    def test_protect_leaves_block_and_nonlinears(self):
        torch.manual_seed(3)
        model = Tiny()
        embed0 = model.model.embed_tokens.weight.detach().clone()
        head0 = model.lm_head.weight.detach().clone()
        bias0 = model.model.layers[1].q_proj.bias.detach().clone()
        orig = snapshot_block_linears(model)
        apply_block_quant(model, bits=2, originals=orig, protect=(1,))
        for name, mod in model.model.layers[1].named_modules():
            if isinstance(mod, nn.Linear):
                self.assertTrue(torch.equal(mod.weight.cpu(), orig[(1, name)]))
        # other blocks should move (overwhelmingly likely at 2 bits)
        moved = False
        for name, mod in model.model.layers[0].named_modules():
            if isinstance(mod, nn.Linear):
                if not torch.equal(mod.weight.cpu(), orig[(0, name)]):
                    moved = True
        self.assertTrue(moved, "unprotected block did not change; quantizer is a no-op")
        self.assertTrue(torch.equal(model.model.embed_tokens.weight, embed0))
        self.assertTrue(torch.equal(model.lm_head.weight, head0))
        self.assertTrue(torch.equal(model.model.layers[1].q_proj.bias, bias0))

    def test_disk_bank_protect(self):
        torch.manual_seed(4)
        model = Tiny()
        orig = snapshot_block_linears(model)
        bank = DiskWeightBank("/tmp/ir_qwen05b_test_bank")
        bank.capture(model)
        bank.apply(model, bits=3, protect=(0,))
        for name, mod in model.model.layers[0].named_modules():
            if isinstance(mod, nn.Linear):
                self.assertTrue(torch.equal(mod.weight.cpu(), orig[(0, name)]))


if __name__ == "__main__":
    unittest.main()
