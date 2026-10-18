# mathfix Phase L: is format damage localized in a few LoRA layers?

Pre-registered BEFORE any Phase-L training, calibration scan, or test-set evaluation.

Written 2026-10-18 by the Phase-L driver. After this file is committed, edit only by adding dated
appendices (timing-pilot / N choice, then results pointer). Do not change the claim, arms, metrics,
layer-selection rule, random-subset seed, kill rule, or N decision rule after any test-set number
is known.

Upstream `mathfix/` code is copied unchanged from the attached phase-3 bundle. New Phase-L files
live beside it; they import `mathfix/common.py` and do not edit it.

## Claim (verbatim; one falsifiable sentence)

On Qwen2.5-0.5B-Instruct after N nights of arm-`b` terse fact training (exactly the mathfix `b` recipe and fact/probe sets, 3 seeds 0/1/2), zeroing the LoRA delta in the k=4 layers whose single-layer zeroing most increases mean GSM8K output length on a 32-item calibration set (GSM8K train split, disjoint from the test probes, chosen per seed from calibration only) keeps GSM8K strict retention >= 50% of base-correct test items (mean over seeds) AND keeps recall >= full-b recall - 5 pts (mean over seeds) AND beats random 4-layer zeroing: mean selected retention - mean of the 24 random runs (8 seeded random 4-layer subsets x 3 seeds, seed 1234) > 2 x sample std of those 24 random runs.

## What is already known (not this experiment)

- Arm `b` (terse targets `"X."`) overwrites GSM8K output format: night-2 strict ~6%, outputs ~4–20 tokens, ~96% of failures `no_steps`. A "Let's solve this step by step." prefill restores ~+28 pts. Mixing long text / self-voiced targets / answer-only loss did not keep recall AND math together.
- Separately, quantization work suggested a skill can sit in a small number of layers. This run asks whether the *format* damage in the LoRA delta is similarly localized and separable from fact storage.

## Arms / conditions

| id | what |
|---|---|
| `base` | Qwen2.5-0.5B-Instruct, no adapter. |
| `full_b` | Night-N arm-`b` adapter, LoRA delta intact. |
| `selected_k4` | Same adapter; LoRA delta zeroed in the k=4 layers chosen by the calibration rule below. Verdict arm. |
| `random_k4` | Control. Same adapter; LoRA delta zeroed in a random 4-layer subset. 8 subsets × 3 seeds = 24 runs. |
| `selected_k2`, `selected_k8` | Same selection ranking, top-2 / top-8. **Post-hoc only. Not in the verdict.** |

Baselines: `base` and `full_b`. Control: `random_k4`.

## N and the decision rule (locked before test eval)

- Default **N = 2** nights = 240 steps (120 steps/night × 2).
- **Before any test-set eval**, run a timing pilot: a few arm-`b` training steps + a few greedy generations on the real 0.5B (CPU). Extrapolate wall time for the full plan:
  - 3 seeds × N nights × 120 training steps
  - base GSM8K on the 200 mathfix test probes (needed to mark base-correct items)
  - calibration scan: 3 seeds × 24 single-layer zeroings × 32 calib items
  - test evals of `full_b`, `selected_k4`, and 8 random subsets, on base-correct GSM items + recall
- If that prediction exceeds **~5 h wall**, set **N = 1** (120 steps).
- Record the measured pilot numbers, the predicted hours for N=2 and N=1, and the chosen N in a dated appendix of *this file*, and commit that appendix **before any test-set eval**.
- After that commit, N is frozen. No later retuning of k, N, calibration metric, sets, or seeds.

## Training recipe (arm `b`, verbatim from `mathfix/common.py`)

- Model: `Qwen/Qwen2.5-0.5B-Instruct`. Qwen2.5-0.5B-Instruct has 24 decoder layers.
- LoRA: r=16, α=32, dropout=0, bias=none, task_type=CAUSAL_LM.
- Target modules (all layers): `self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj` (7 modules).
- Optimizer: AdamW, lr=2e-4, weight_decay=0, grad clip 1.0. `common.Trainer.train_night` with `extra=None`.
- 120 steps/night, batch 16. Night n+1 continues from the night-n adapter (`PeftModel.from_pretrained(..., is_trainable=True)`).
- Facts: `common.make_facts` / `load_facts`, `fact_seed=12345`, 100 facts/night. Targets `train_a = "{a}."` (terse). For N<7 the stream is the prefix of the 7-night stream (same RNG). Hash asserted only when nights==7.
- Seeds: **0, 1, 2**. `torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)` at the start of each night, and `train_night` uses `Random(seed * 100 + night)` for batch sampling (unchanged).
- Device: this VM's CPU (`common.DEV` fallback). dtype float32. No GPU, no paid compute.
- Generation batch size may be lowered on CPU for memory; decoding is greedy and left-padded as in `common.Gen`, so outputs are not a design change.

## Test probes and GSM metric

- Probes: `common.load_probes` with `n_probe=200`, `n_calib=40`, `probe_seed=1234`. GSM8K items are from the **test** split. Non-smoke probe hash must equal `62a39450d3434e0dfc267e584c096a17d61fdf56cc39cc2f18a619942ad74ced`.
- GSM prompt suffix (verbatim): `"\nSolve step by step. End with a final line exactly of the form 'The answer is N.' where N is a number."`
- Parser: `common.parse_gsm` (v2 / "strict"). Lenient and failure categories are logged but are not verdict metrics.
- **Retention** (primary GSM number): on the base-correct subset of the 200 test probes, `100 * n_strict_ok / n_base_correct`. GSM may be generated only on those base-correct items after the base run has marked them.
- `max_new_tokens = 512` for every GSM config (`common.BASE_CFG["gsm_max_new"]`). Greedy (`do_sample=False`).
- Base run must score all 200 test probes once, to define the base-correct set. That set is shared across seeds (base has no adapter).

## Recall

- `common.eval_recall`: two reworded questions per fact, `max_new_tokens=24`, `ans_match`.
- Facts scored = nights 1..N of the fact stream (the facts that were trained).
- `full_b` recall and `selected_k4` recall are both measured this way. Condition 2 uses the mean over the three seeds.

## Calibration set (layer selection ONLY; never in the verdict)

- Source: `openai/gsm8k` `main` **train** split (already a different split from the test probes).
- Additionally drop any train row whose `question` text equals a test-probe question with the solve-suffix stripped.
- Draw **32** indices with `random.Random(20261018).sample(eligible, 32)`, then store them sorted. **Same 32 items for every seed.**
- Prompted with the same GSM suffix as the test probes.
- Used only to rank layers by mean generated token count. No test-probe item is used for selection.

## Exact layer-selection rule (per seed; calibration only)

Let `n_layers = 24`. For seed s in {0,1,2}:

1. Load that seed's night-N adapter on the frozen base.
2. For each layer L in `0 .. n_layers-1`:
   - Zero the LoRA delta of **all 7 modules** in decoder layer L (see Zeroing).
   - Greedy-generate the 32 calibration prompts, `max_new_tokens=512`.
   - Record `mean_tokens(L)` = mean of `common.Gen` token counts (tokens until EOS, else `max_new`).
   - Restore the adapter weights before the next L.
3. Rank layers by `mean_tokens` **descending**. Ties broken by **lower layer index first**.
4. `selected_k4(s)` = the first 4 layers in that ranking. (`selected_k2` / `selected_k8` = first 2 / first 8; post-hoc.)
5. Test eval of `selected_k4`: zero those 4 layers **together** (not one at a time) and score GSM retention + recall.

Selection sees only calibration mean length. It does not see test GSM accuracy, test length, or recall.

## Random 4-layer subsets (control)

```
rng = random.Random(1234)
subsets = []
while len(subsets) < 8:
    s = tuple(sorted(rng.sample(range(n_layers), 4)))
    if s not in subsets:
        subsets.append(list(s))
```

The same 8 subsets are applied to each of the 3 seed adapters → 24 runs. Each run reports retention on that seed's (shared) base-correct GSM items.

## Zeroing (no retraining)

For every PEFT `LoraLayer` whose module name matches `layers.{L}.` with L in the zero-set, set **both** `lora_A[*].weight` and `lora_B[*].weight` to 0 (in-place, after saving a clone for restore). Then `dW = scaling * B @ A = 0` for all 7 projections in that decoder layer. Other layers unchanged. This is an edit of a trained adapter, not a re-init and not a re-train.

Unit test (tiny model first): zeroing **all** layers reproduces base greedy outputs exactly; zeroing **none** reproduces full-adapter outputs exactly.

## Verdict metrics (means over seeds 0/1/2 unless stated)

Let `ret(c, s)` = GSM strict retention of condition c on seed s.
Let `rec(c, s)` = recall of condition c on seed s.

- **C1 (retention):** `mean_s ret(selected_k4, s) >= 50`.
- **C2 (recall):** `mean_s rec(selected_k4, s) >= mean_s rec(full_b, s) - 5`.
- **C3 (beats random):** `mean_s ret(selected_k4, s) - mean(24 random retentions) > 2 * sample_std(24 random retentions, ddof=1)`.
- **Validity:** `mean_s ret(full_b, s) < 50`. If this fails, report **INVALID**, not PASS, and do not treat C1–C3 as a confirmation.

**Kill rule:** any of C1 / C2 / C3 fails ⇒ the claim is **DEAD as stated**. Record the numbers. Do not retune k, N, the calibration metric, the sets, or the seeds after any test-set eval.

**PASS** iff Validity holds AND C1 AND C2 AND C3.
**FAIL** iff Validity holds AND at least one of C1–C3 fails.
**INVALID** iff Validity fails (full `b` was not damaged).

## Secondary numbers (labelled post-hoc; never the verdict)

- `selected_k2` and `selected_k8` retention / recall / mean tokens / % steps.
- Recall of the 24 random runs.
- Mean tokens and `% steps` (`common.has_steps`) for every condition.
- Per-layer calibration `mean_tokens` tables.
- Wall time.

## Logging

- Raw JSON: `results/phaseL/` (per-seed records, calibration ranks, random subsets, aggregates, timing, commit hashes).
- Human report: `RESULTS_phaseL.md` with the verdict line first, then every number in a table.

## Smoke / correctness (not evidence)

1. Tiny-model unit test of zeroing (all-zero == base, no-zero == full adapter).
2. Optional `MATHFIX_SMOKE=1` path of the driver (tiny-Qwen2, 1 night, tiny probes). Numbers are meaningless.

## Compute

One CPU VM, 4 cores, no GPU, no paid compute. Accuracy and honesty first. A clean FAIL is a valid outcome. Never fake or extrapolate a missing number into the verdict tables.

## Appendix A — timing pilot and N choice

Written 2026-10-18 08:14:37 +0000 after the timing pilot and **before any test-set eval**.
Prereg commit at first lock: see git history of this file before this appendix.

### Measured (real Qwen2.5-0.5B-Instruct, CPU, not test-set)

| quantity | value |
|---|---|
| device | cpu |
| n_layers | 24 |
| n_cpu | 4 |
| sec / train step (4 steps incl. first) | 5.9868 |
| gen tokens / sec (4 prompts, max_new=64) | 33.8043 |
| tokens produced in that gen | 256 |
| gen wall sec | 7.57 |
| second gen (2 prompts, max_new=128) tokens / sec | 20.2871 |
| pilot wall sec | 46.1 |

### Predicted full-plan wall (from those rates; not measured)

| N | predicted hours | train h | base GSM h | calib h | test GSM h | recall h |
|---|---|---|---|---|---|---|
| 1 | 3.66 | 0.60 | 0.36 | 0.80 | 1.26 | 0.47 |
| 2 | 4.73 | 1.20 | 0.36 | 0.80 | 1.26 | 0.95 |

Prediction assumptions (timing only, never used as experimental numbers): base mean 218 tokens;
calib 20 layers × 15 tok + 4 layers × 180 tok; test mix short/long as in `predict_hours`.

### N choice

**N = 2**

Rule applied: default N=2; if predicted wall for the full plan exceeds ~5 h, set N=1.
N=2 predicted 4.73 h, which is
at or below the 5 h cutoff.
