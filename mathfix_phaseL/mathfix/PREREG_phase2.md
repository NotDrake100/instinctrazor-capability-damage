# mathfix Phase 2: pre-registered design and PASS criteria (written before any Phase-2 run)

Written 2026-10-17 ~19:05 IST, before Phase 1 or Phase 2 had run on real data. Phase 2 starts only after Phase 1 has
produced `phase1_out/phase1_summary.json`. That file names the mechanism (rules in predictions.md).

Common to all arms (identical to sleep_shield.py arm b): Qwen2.5-0.5B-Instruct (config `phase2_0p5b`), LoRA r16 / α32 on
q, k, v, o, gate, up and down. Learning rate 2e-4, AdamW (fresh each night), 120 steps/night, batch 16, fp16 autocast,
grad-clip 1.0. 7 nights × 100 fictional facts, accumulating. Seeds 0, 1, 2. Same probe hash (62a39450…) and fact hash
(f43d375c…). Evaluated every night on base-correct items: recall on the reworded questions, GSM8K strict (v2 parser),
GSM8K lenient (last number), mean output tokens, % outputs with reasoning steps, failure categories, mult and ARC.
Adapter-off recall is checked at nights 1 and 7.

## Arms by mechanism
- **H1, format overwrite.** `b` facts only. `r` facts + GSM8K-train worked solutions. `c` facts + non-math filler.
  - K = 384 GSM8K *train* items, fixed: random.Random(2024) sample, excluding the 32 items sleep_shield.py used as KL
    texts and any question identical to a scored test probe (the probes are from the test split anyway).
  - Format: the user turn is the question plus the probe's own instruction suffix, in the model's chat template. The
    assistant turn is the GSM8K solution with `<<…>>` calculator tags removed and `#### N` replaced by
    `The answer is N.`
  - Mixing: every step has batch 16. The extra slots follow the pattern [3, 3, 3, 3, 4], which is exactly 20% of the
    1,920 training examples per night = 384, so each replay item is used exactly once per night (fresh permutation each
    night). The fact slots use the same RNG stream as arm b. Facts and extras go through separate micro-batches and the
    loss is normalized by the total number of supervised tokens. This equals one combined batch (checked in
    tests_cpu.py, |diff| 3e-6).
  - Control `c`: Databricks Dolly-15k items with no digit anywhere in instruction, context or response. Each is paired
    1:1 with a replay item and truncated to exactly the same number of supervised tokens (asserted), placed in the same
    slots. Prompt (unsupervised) tokens are not matched; both totals are logged.
  - Fix arm `r`, control `c`.
- **H2, localized layers.** `b`; `p` = no LoRA on the Phase-1 best-swap block; `rnd` = no LoRA on the same number of
  random other layers (Random(777+seed)); `bot` = no LoRA on the bottom k layers (top k if the block *is* the bottom).
  Fix `p`; controls `rnd` and `bot` (the better one is used).
- **H3, shared shift.** `b`; `x` = after each night's training, remove the top-1 singular component of every LoRA update
  (exact, re-factored at rank 16); `rr` = remove a random rank-1 matrix of the same norm σ1, drawn inside the update's
  own row and column span so it stays representable at rank 16. Fix `x`, control `rr`. (Phase 1 also tests the
  cross-seed *shared* direction. That cannot be computed online, so the online fix uses each run's own top-1, which
  Palit et al. use.)

## PASS criteria (night 7, mean over 3 seeds; all must hold)
- **P1:** fix-arm GSM8K strict retention ≥ 50% of base-correct items.
- **P2:** fix − best control > the combined seed std, sqrt(sd_fix² + sd_control²), sample std with ddof = 1.
- **P3:** |fix recall − b recall| ≤ 5 pts.
- **Validity:** adapter-off recall ≤ base recall + 5 in every run, at nights 1 and 7.

Reported separately, not part of PASS: b lenient − strict (≥ 20 pts = parser-level format loss), mult and ARC
retention, output length and % steps per arm.

## 1.5B confirmation (`--config phase2_1p5b`)
Same arms and criteria on Qwen2.5-1.5B-Instruct, with base-correct sets recomputed for that model. For H2, Phase-1 layer
indices (24 layers) are mapped to 28 layers by relative depth, round(l·27/23). This is a confirmation run. It uses the
0.5B-derived mechanism and is labelled as such.
