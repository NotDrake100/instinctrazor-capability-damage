# Qwen2.5-0.5B-Instruct critical-layer replication

**Verdict: FAIL.** The pre-registered claim is dead for this model / this CPU fake-quant / these seeds. No B, metric, set, prompt, or seed was changed after seeing the test numbers.

Pre-registration: `PREREG_qwen05b_critical_layer.md`, committed as `65c8414` before any test-set evaluation.  
Raw files: `results_qwen05b/{problems,pilot,calibration,test,results}.json`.  
Quantizer: CPU asymmetric RTN, group 128, dequantized to fp32. InstinctRazor’s symmetric/MSE-clip MoE `fakequant` was **not** used (dense model, no GPU; the pre-reg specified this grid). Embeddings and `lm_head` stayed fp32.

## Locked choices

| | |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, 24 blocks, fp32 CPU |
| B (pilot only) | **4** |
| L* (calibration NLL only) | **block 16** (not 0, not last) |
| Random blocks (seed 44, exclude L*) | 13, 17, 18, 3, 5, 12, 7, 9 |
| Wall time | **756.1 s** (~12.6 min): pilot 35.8 + calibrate 216.7 + test 499.6 |

## Pilot (n=50) — choose B

Highest bit-width whose full-quant accuracy is ≥15 points below fp32.

| condition | correct | acc | 95% bootstrap CI |
|---|---|---|---|
| fp32 | 30/50 | 0.600 | [0.460, 0.740] |
| 4-bit all blocks | 19/50 | 0.380 | [0.240, 0.520] |
| 3-bit all blocks | 3/50 | 0.060 | [0.000, 0.140] |
| 2-bit all blocks | 0/50 | 0.000 | [0.000, 0.000] |

4-bit drop = 0.220 ≥ 0.15, so **B = 4**. (3-bit and 2-bit also qualify; the rule takes the highest width.)

## Calibration NLL (n=64) — choose L*

No generation. Mean per-item NLL of gold answer tokens. Full-quant NLL = 0.666004.

| | block | NLL | drop vs full-quant |
|---|---|---|---|
| **L\*** | **16** | **0.608313** | **0.057691** |
| runner-up | 15 | 0.614403 | 0.051601 |
| third | 21 | 0.625105 | 0.040899 |

L* beats the runner-up by **0.00609 nats**. L* is **not** block 0 (granite) and **not** the last block (Qwen3 recall used 47 / last-ish). Several restorations *raised* NLL (blocks 1, 2, 4, 5, 7, 9, 13, 14, 19). Full 24-row table: `results_qwen05b/calibration.json`.

## Test (n=300, scored once)

| condition | correct | acc | 95% bootstrap CI |
|---|---|---|---|
| fp32 | 162/300 | 0.5400 | [0.4833, 0.5967] |
| full-quant @ 4-bit | 93/300 | 0.3100 | [0.2567, 0.3633] |
| **protect L\* = 16** | **104/300** | **0.3467** | **[0.2933, 0.4000]** |
| protect random 13 | 96/300 | 0.3200 | [0.2667, 0.3733] |
| protect random 17 | 100/300 | 0.3333 | [0.2800, 0.3867] |
| protect random 18 | 94/300 | 0.3133 | [0.2600, 0.3667] |
| protect random 3 | 87/300 | 0.2900 | [0.2367, 0.3400] |
| protect random 5 | 91/300 | 0.3033 | [0.2533, 0.3567] |
| protect random 12 | 97/300 | 0.3233 | [0.2700, 0.3767] |
| protect random 7 | 86/300 | 0.2867 | [0.2367, 0.3367] |
| protect random 9 | 93/300 | 0.3100 | [0.2600, 0.3633] |

mean(random) = 0.3100; sample std (ddof=1) = 0.01613; mean + 2 std = 0.3423.

## Verdict

```
R = (0.3467 − 0.3100) / (0.5400 − 0.3100) = 0.1594
```

| pre-registered condition | result |
|---|---|
| R ≥ 0.5 | **FAIL** (0.159 < 0.5) |
| acc(L*) > mean(random) + 2 std | **PASS** (0.3467 > 0.3423) |
| **both required** | **FAIL** |

Protecting L* recovered about **16%** of the accuracy lost to 4-bit fake-quant (~3.7 points of a 23-point drop). It did beat the 8 random single-block protections, but only just (the 95% CIs all overlap). The kill rule fires. Claim is dead here.

## What went wrong / caveats

- Nothing failed to run. Unit tests and the tiny random-model smoke passed before the registered eval.
- InstinctRazor was not applied: this is a different grid (asymmetric RTN, no MSE clip) and a dense 0.5B instruct checkpoint, not the granite/Qwen3 MoE recipe. A FAIL here does **not** refute the original GPU MoE numbers.
- The instruct model often continues after the integer with Chinese “请逐步…”. Scoring is the repo’s `g_num` (integer prefix, next char not a digit), so those extra tokens do not flip a correct integer to wrong.
- transformers 5.20 printed `torch_dtype` deprecation and ignored `top_p`/`top_k` under greedy decoding. Decoding was still `do_sample=False`.
- RAM was tight (~2.5 GiB available with the fp32 model resident). Original block weights were snapshotted to disk; no OOM.
- fp32 test acc (54%) is below fp32 pilot acc (60%). That is expected noise (n=50 vs n=300); B was not revisited.
- Secondary comparison: L* = 16, not layer 0 (granite) and not a last-block analogue of Qwen3-30B’s 47.

No numbers above were invented. Recompute from `results_qwen05b/test.json` `correct` vectors if you want to check the arithmetic.
