"""Tiny smoke: random TinyLM, fake-quant, NLL, greedy generate. Not the registered test."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from qwen05b_critical_layer import (  # noqa: E402
    g_num,
    generate_splits,
    run_smoke,
    SHOT_PAIRS,
)


class TestScoringAndSplits(unittest.TestCase):
    def test_g_num(self):
        self.assertTrue(g_num("132", "132"))
        self.assertTrue(g_num("132\n", "132"))
        self.assertTrue(g_num("132 extra", "132"))
        self.assertFalse(g_num("1320", "132"))
        self.assertFalse(g_num("13", "132"))
        self.assertFalse(g_num(" 99", "132"))

    def test_splits_disjoint(self):
        d = generate_splits()
        sets = []
        for name in ("pilot", "calibration", "test"):
            pairs = {(it["x"], it["y"]) for it in d[name]}
            self.assertEqual(len(pairs), len(d[name]))
            self.assertTrue(pairs.isdisjoint(SHOT_PAIRS))
            sets.append(pairs)
        self.assertTrue(sets[0].isdisjoint(sets[1]))
        self.assertTrue(sets[0].isdisjoint(sets[2]))
        self.assertTrue(sets[1].isdisjoint(sets[2]))
        self.assertEqual(len(d["pilot"]), 50)
        self.assertEqual(len(d["calibration"]), 64)
        self.assertEqual(len(d["test"]), 300)


class TestSmokeRun(unittest.TestCase):
    def test_smoke(self):
        out = os.path.join(ROOT, "results_qwen05b", "smoke_unittest")
        r = run_smoke(out)
        self.assertTrue(r["ok"])
        self.assertGreaterEqual(r["n_used"], 1)
        self.assertIn("apply", r)


if __name__ == "__main__":
    unittest.main()
