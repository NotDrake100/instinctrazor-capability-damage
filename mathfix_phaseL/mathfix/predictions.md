# mathfix Phase 1: predictions, written BEFORE any Phase-1 run

Written 2026-10-17 ~19:05 IST, before any Phase-1 run on real data. Only a CPU smoke test with a tiny random model and the
CPU unit checks in `tests_cpu.py` had been run at this point. Once Phase 1 has run, edit this file only by adding a dated
"post-hoc" section. Do not change the rules below.

## What we already know (sleep_shield.py, Vast 2080 Ti, arm b, 3 seeds)
Night 7 retention of base-correct items: mult 41.7, ARC 57.5, GSM8K **2.7** (base GSM8K 50% = 100 base-correct items).
Recall 87.1. The GSM8K drop is already ~96% after night 1 (4.0 / 4.0 / 0.0 per seed). It happens with the strict v2
parser, which already accepts `\boxed{}`, "the answer is \(N\)" and so on, and falls back to the last number when no
answer marker is present.

## Hypotheses
- **H1, format overwrite.** Training on 1–3-token answers ("Firkurulish.") teaches the model a global "answer
  immediately and briefly" policy. GSM8K fails because the model stops writing its chain of thought. The capability is
  still there but is not used.
- **H2, real reasoning loss localized to specific layers.** The update damages the computation of multi-step arithmetic
  in a few layers. The model still tries to reason but gets it wrong, or the damaged layers are where the math lives.
- **H3, shared removable shift (Palit et al., arXiv 2610.08718).** Most of the damage is one shared direction in each
  weight update, written by everything the new facts have in common (short-answer region, "answer now"). It hides the old
  behaviour rather than erasing it, and subtracting it brings the behaviour back.

These are not mutually exclusive. H1 and H3 can both hold if the "answer briefly" policy is the shared direction.

## Predictions per measurement
| measurement | H1 format overwrite | H2 localized reasoning loss | H3 shared removable shift |
|---|---|---|---|
| (b) failure categories | mostly `no_steps` / `fact_style`; mean output length falls to a small fraction of base; % outputs with reasoning steps collapses | mostly `reasoning_wrong` (steps present, wrong result); output length near base | no specific category signature; whatever category H1/H2 predict, it is restored by (f) |
| (b) strict vs lenient | lenient ≈ strict: a short answer has no marker, so strict already uses the last number. A big lenient > strict gap would instead mean the marker is present but points to the wrong number (parser-level format loss) | lenient ≈ strict | lenient ≈ strict |
| (g) prefill "Let's solve this step by step." | rescues a large part (≥ +25 pts) | little rescue (reasoning written but wrong) | maybe partial |
| (c) residual-stream drift | drift grows with depth; large final-position KL; **high shift coherence** (one common displacement across prompts) in late layers | drift peaks at specific (mid) layers; lower coherence | **high shift coherence** at most layers; drift largely shared across prompts |
| (d) logit lens, first answer token | base: P(reasoning-start tokens) dominates in late layers. Night 7: P(short-answer-start / digit tokens) takes over in the last layers, while early/mid layers look like base | start-token distribution near base (the model still begins reasoning) | short-answer start dominates, and the shift sits along one direction (pair with (f)) |
| (e) layer swap, night-7 adapter | no single layer restores much. Possibly late blocks restore some format, at a recall cost. Gains are diffuse across blocks | **one 3-layer block restores ≥ 25 pts**, consistently across seeds, with small gains elsewhere | diffuse: no single block. The shift is written in every layer |
| (f) remove top-1 shared direction | may restore if the "answer briefly" policy is low-rank | little effect beyond the random rank-1 control | **restores ≥ 20 pts and ≥ 10 pts more than random rank-1**, at a moderate recall cost |

## Decision rules (automatic; implemented verbatim in `phase1_summarize.py`, TH dict)
All rules use night 7, mean over seeds 0, 1, 2. Percentages are of base-correct items.

**H1 supported** if all three hold:
- F1: ≥ 60% of GSM8K failures are `no_steps` or `fact_style`.
- F2: mean output tokens ≤ 0.5 × base mean on the same items.
- F3: % outputs with reasoning steps (≥ 2 arithmetic operations/equations) drops by ≥ 40 pts vs base.

F4 (capability intact) adds: prefill strict retention ≥ trained strict + 25 pts. If H1 and F4 both hold, call it
"pure format".

**H2 supported** if all three hold:
- L1: the best disjoint 3-layer block swap (LoRA off on that block only) raises GSM8K strict on the fixed ≤ 100-item
  subset by ≥ 25 pts over the unswapped night-7 adapter.
- L2: that block is among the top-2 blocks in ≥ 2 of 3 seeds.
- L3: its gain is ≥ 2 × the median block gain (localized, not diffuse).

**H3 supported** if all three hold:
- S1: the better of {minus shared top-1, minus own top-1} raises GSM8K strict by ≥ 20 pts over the merged unedited
  night-7 model.
- S2: that gain beats the random rank-1 control (same norm as own top-1) by ≥ 10 pts.
- S3 (validity): the merged unedited model is within 10 pts of the PEFT-path night-7 score on the same subset.

**Parser-level format loss** in b: lenient − strict ≥ 20 pts.

**Phase-2 choice.** Among supported hypotheses, take the one with the largest GSM8K restoration: H1 uses the prefill
gain, H2 the best block gain, H3 the edit gain. If none is supported, the pre-registered fallback is H1's
reasoning-replay vs token-matched control. That test does not depend on the mechanism, and the report says "no
mechanism identified".

## Definitions (transparent rules, `common.py`)
- Reasoning steps present: ≥ 2 matches of `number op number` (+ − × ÷ * / x, \times, \cdot, \div) or `= number`. On
  the 60 base outputs saved in `../diag/gsm_raw_N60_max512.json`, 59/60 count as having steps.
- Failure category, first rule that applies: `truncated` (no end token within 512) → `fact_style` (≤ 12 words, 0
  arithmetic ops, and either no digit or a fact-stream word: …ish / …ium / River / a fictional name) → `no_steps`
  (< 2 ops) → `reasoning_wrong` (≥ 2 ops, finished, wrong final answer) → `other`.
- Logit-lens token sets come from the data. R = first tokens of ≥ 5% of base outputs that have reasoning steps.
  S = first tokens of ≥ 5% of night-7 `no_steps`/`fact_style` failures, excluding R. D = digit tokens 0–9. These sets
  are chosen after night 7 has been generated but before the lens is computed. This is a small researcher degree of
  freedom and is reported.
