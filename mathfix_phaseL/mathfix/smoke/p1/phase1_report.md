# Phase 1 report (auto-generated from results.json; numbers are mean ± std over seeds)

Base on its base-correct GSM8K items: lenient 0.0, mean tokens 16, reasoning steps present 0%

| night | recall | GSM strict | GSM lenient | tokens | % steps | % trunc | fail: no_steps / fact_style / reasoning_wrong / truncated | mult | arc |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 | 16 | 0 | 100 | 0 / 0 / 0 / 100 | 0.0 | 0.0 |

Prefill test at night 1: strict 0.0±0.0 (trained 0.0)

## Evidence vs pre-registered rules
```
{
 "H1": {
  "short_fail_share": 0.0,
  "token_ratio_vs_base": 1.0,
  "steps_drop_pts": 0.0,
  "prefill_gain_pts": 0.0,
  "F1": false,
  "F2": false,
  "F3": false,
  "F4_capability_intact": false,
  "supported": false,
  "pure_format": false,
  "restoration_pts": 0.0
 },
 "H2": {
  "best_block": "B0-1",
  "best_block_layers": [
   0,
   1
  ],
  "best_block_gain": 0.0,
  "best_block_recall_cost": 0.0,
  "best_block_top2_in_seeds": 1,
  "median_block_gain": 0.0,
  "best_single": "L0",
  "best_single_gain": 0.0,
  "all_off_gain": 0.0,
  "reasoning_wrong_share_night_last": 0.0,
  "L1": false,
  "L2": false,
  "L3": false,
  "supported": false,
  "restoration_pts": 0.0
 },
 "H3": {
  "best_edit": "minus_shared_top1",
  "edit_gain": 0.0,
  "random_gain": 0.0,
  "edit_recall_change": 0.0,
  "merge_sanity_abs_diff": [
   0.0
  ],
  "S1": false,
  "S2": false,
  "S3_valid": true,
  "supported": false,
  "restoration_pts": 0.0
 }
}
```

## Phase-2 choice
```
{
 "mechanism": "H1",
 "reason": "NO hypothesis met its pre-registered criteria; pre-registered fallback = reasoning replay vs token-matched control (mechanism-agnostic)",
 "fallback": true,
 "protect_layers": [
  0,
  1
 ],
 "edit": "minus_shared_top1"
}
```

## Layer swap (top 10 by GSM gain)
| config | layers off | GSM gain | recall cost |
|---|---|---|---|
| L0 | [0] | 0.0±0.0 | 0.0 |
| L1 | [1] | 0.0±0.0 | 0.0 |
| B0-1 | [0, 1] | 0.0±0.0 | 0.0 |
| all | [0, 1] | 0.0±0.0 | 0.0 |

## Shared-direction edits (change vs merged unedited night-7 model)
| edit | GSM gain | recall change | mult | arc |
|---|---|---|---|---|
| minus_shared_top1 | 0.0±0.0 | 0.0 | 0.0 | 0.0 |
| minus_own_top1 | 0.0±0.0 | 0.0 | 0.0 | 0.0 |
| minus_random_rank1 | 0.0±0.0 | 0.0 | 0.0 | 0.0 |
| base_no_adapter | 0.0±0.0 | 0.0 | 0.0 | 0.0 |

shared stats: {'cos_seed_pairs': nan, 'cos_own_vs_shared': 0.999999965940203, 'energy_shared': 0.5839781486504052, 'energy_own_top1': 0.583978234297448, 'note': 'means over all 168 matrices; cos = |cosine| of top-1 left singular vectors; energy = fraction of ||dW||_F^2 at night 7'}
