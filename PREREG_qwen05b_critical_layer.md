# Pre-registration: Qwen2.5-0.5B-Instruct single critical-layer protection

**Date locked:** 2026-10-18  
**Model:** `Qwen/Qwen2.5-0.5B-Instruct`  
**Device:** CPU, `torch.float32`. No GPU. No paid compute.  
**This file is committed before any test-set evaluation.**

This is one pre-registered replication of InstinctRazor's critical-layer
protection finding (Qwen3-30B-A3B layers 9+47; granite-1b layer 0), on a
third model family. Accuracy over excitement. No tuning after seeing
test results.

## Claim (verbatim)

On Qwen2.5-0.5B-Instruct quantized to bit-width B, keeping the single
transformer block L* (chosen only from a calibration scan) at full
precision recovers at least 50% of the 2-digit multiplication accuracy
lost to quantization on a fresh test set of n=300, AND beats the mean
of 8 random single-block protections by more than 2 standard deviations
of those 8.

## What is reused from this repo, and what is not

**Reused (comparability):**

- 2-digit multiplication prompt format from `bigrun/bigrun.py` (Qwen3 suite):
  three fixed shots then `"{x}*{y}="`.
- Scoring `g_num` from `scripts/run.py` / `bigrun/bigrun.py`: strip, require
  that the continuation starts with the gold integer and that the next
  character is not a digit. For this task (products of two 2-digit
  positives) this is exact-match on the integer answer.
- Greedy decoding (`do_sample=False`), `max_new_tokens=6` (same as Qwen3 `mul2`).
- 95% bootstrap CIs, 10_000 resamples, numpy Generator seed 0
  (same as `scripts/analyze.py`).

**Not reused — CPU fake-quant (declared here):**

InstinctRazor's `moe_quant.fakequant` is a *symmetric* per-block integer
grid with optional 16-step MSE clip search, written for MoE expert
tensors (`General-Instinct/InstinctRazor` @ `89bd8b8`). It is not
vendored in this repo. Qwen2.5-0.5B-Instruct is a *dense* decoder
(no routed experts). This pre-reg therefore uses a simple CPU
fake-quant specified below, not InstinctRazor's MoE recipe. Embeddings
and `lm_head` stay fp32 (the granite/Qwen3 flat recipe quantized those
to INT4; we do not).

## Quantizer (locked)

- Round-to-nearest, per-group *asymmetric* affine quant, group size 128,
  dequantized back to fp32 (weights stay fp32 tensors; only the values
  are snapped to the B-bit grid).
- Last dimension is the grouping axis. If `in_features % 128 != 0`, the
  entire last dimension is one group (same fallback as InstinctRazor).
- Integer grid `{0, …, 2^B − 1}`. Per group:
  `scale = (max − min) / (2^B − 1)`; if `max == min`, the group is a
  no-op (constant reconstructed exactly).
  `zero_point = clamp(round(−min / scale), 0, 2^B − 1)`.
  `q = clamp(round(w / scale + zero_point), 0, 2^B − 1)`.
  `dequant = (q − zero_point) * scale`.
- **What is quantized:** every `torch.nn.Linear` inside
  `model.model.layers[i]` (Qwen2 attention + MLP projections), all
  blocks, unless that block is listed as protected.
- **What stays fp32:** `embed_tokens`, `lm_head`, layer norms, biases
  (biases are not fake-quantized), any non-Linear parameter.
- Protecting block `L` means that block's Linear weights are left at
  their original fp32 values; every other block is fake-quantized.

## Data (locked)

2-digit × 2-digit multiplication. Factors independently uniform on
`{10, …, 99}`. Three disjoint sets, generated from the seeds below
*before* any model scoring. A pair `(x, y)` is unique across all three
sets and is never one of the three shot pairs `(12, 11)`, `(23, 14)`,
`(31, 25)`.

| split        | n   | seed | role                                      |
|--------------|-----|------|-------------------------------------------|
| pilot        | 50  | 11   | choose B only                             |
| calibration  | 64  | 22   | choose L* only (NLL; no generation)       |
| test         | 300 | 33   | scored once, after B and L* are frozen    |

Prompt (byte-identical to Qwen3 `mul2`):

```
12*11=132
23*14=322
31*25=775
{x}*{y}=
```

No chat template. No system prompt. The instruct checkpoint is used as
a completion model, matching the repo's base-model protocol.

Generation of the *problem lists* is not an evaluation. The test list
must not be scored until B and L* are written to disk.

## Choosing B (pilot set only)

1. Score fp32 on the pilot set.
2. Try bit-widths **4, then 3, then 2**, in that order. At each width,
   fake-quant *all* Linear weights in *all* blocks (no protection).
3. Pick the **highest** bit-width whose full-quant pilot accuracy is
   at least **15 percentage points** below fp32 pilot accuracy
   (`acc_fp32 − acc_quant >= 0.15`).
4. If none qualifies, **stop**. That is a valid result: the claim is
   not testable on this model/quantizer at these widths. Do not change
   B, the drop threshold, the prompt, or the metric. Write RESULTS.md
   and halt; do not run the test set.

## Choosing L* (calibration set only)

No generation. Starting from the fully quantized model at B, restore
each transformer block to fp32 one at a time and measure the **mean
negative log-likelihood of the gold answer tokens** on the 64
calibration items.

- Tokenize `prompt` and `prompt + gold` with the tokenizer's default
  special tokens. Gold tokens are the suffix of `prompt+gold` after
  the prompt token length. If the tokenizer is not prefix-stable,
  fall back to encoding `gold` with `add_special_tokens=False` as the
  suffix. If a suffix is empty, that item is skipped (must be logged).
- Per item: mean NLL over that item's gold tokens (teacher forcing,
  shift-by-one logits). Report the mean of those per-item means
  (items equally weighted).
- `L* = argmin_i NLL(restore i)`. Equivalently, the block whose
  restoration lowers NLL the most relative to full-quant.
- Tie-break: lowest block index.
- Save the full per-block table (index, NLL, drop vs full-quant).
  This table is secondary and is not a verdict input.

## Random control (locked before test scoring)

After L* is chosen, draw **8** distinct blocks uniformly without
replacement from `{0, …, n_layers−1} \ {L*}`, using
`random.Random(44).sample`. Evaluate each as a single-block protection
on the test set (same protocol as L*).

Sample standard deviation (`ddof=1`) of the 8 test accuracies.

## Test evaluation (n=300, run once)

Conditions, in this order: fp32; full-quant at B; protect L*; the 8
random single-block protections. Each condition is scored exactly
once. No re-runs, no bit-width changes, no prompt changes.

Report accuracy and a 95% bootstrap CI for each.

## Verdict (locked)

Let `acc(·)` be test accuracy.

- Recovery fraction:
  `R = (acc(L*) − acc(quant)) / (acc(fp32) − acc(quant))`.
- If `acc(fp32) == acc(quant)`, the denominator is zero: **FAIL**
  (nothing to recover). Do not replace the metric.
- `std_rand` = sample std (`ddof=1`) of the 8 random-protection
  accuracies. If `std_rand == 0`, the second inequality becomes
  `acc(L*) > mean(random)` (strict).

**PASS** requires both:

1. `R >= 0.5`
2. `acc(L*) > mean(random) + 2 * std_rand`

**Kill rule:** if either condition fails, the claim is dead for this
model / this quantizer / these seeds. No changing B, metric, sets,
prompt, scoring, or seeds afterwards.

## Secondary (reported only; not in the verdict)

- Whether L* is block 0 or the last block (granite-1b was 0; Qwen3-30B
  multiplication's worst single layer was 9, protection used 9 and 47).
- The full per-block calibration NLL table.

## Implementation constraints (locked)

- `torch.manual_seed(0)` at process start.
- `max_new_tokens = 6`, greedy, left padding, `pad_token = eos` if unset.
- Generation batch size 4 (does not change greedy token identity when
  attention masks are correct).
- Bootstrap: 10_000 resamples, `numpy.random.default_rng(0)`.
- Intermediate JSON is written after pilot and after calibration so
  the test phase cannot “forget” the locked B and L*.
- Unit tests cover fake-quant round-trip error and “protected block
  untouched”. A tiny randomly-initialized smoke model is used for a
  plumbing smoke run. The smoke model is **not** the registered test.

## What we will not do

- Look at test-set model outputs before B and L* are frozen on disk.
- Tune bit-width, group size, prompt, decoding, or scoring after
  seeing any test number.
- Switch to a chat template if few-shot completion is weak.
- Claim InstinctRazor's 3.05-bit MoE recipe was reproduced here.
