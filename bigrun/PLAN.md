# Big run: same experiment on Qwen3-30B-A3B-Base + OLMoE-1B-7B (one rented GPU)

## What it measures
- Suites (exact-match, greedy, few-shot): 3-digit addition, 2-digit multiplication, CounterFact recall, 8-pair in-context lookup, Python output prediction (code), and 2-step word problems (multi-step reasoning).
- A suite is dropped automatically if the base model gets fewer than 30 items right.
- For each model the script does the following, in order:
  1. Scores the original model and keeps up to 400 correct items per suite.
  2. Per-layer scan: for each layer, quantizes only that layer (INT3 experts / INT4 attention, InstinctRazor's own `fakequant` with 16-step clip search) and records which items survive.
  3. Applies the full flat recipe, plus `--protect-layers` variants: layer 0, the worst layer from the scan, and the worst 2 layers.
  4. Takes the best variant and compares it with flat on fresh held-out items (seed 1, a disjoint CounterFact slice, up to 800 items per suite).
- `analyze_big.py` then writes `analysis.json` (CIs, kill condition, per-layer drops, Spearman, held-out gains), `chart.png` and `fix_chart.png`.

## Requirements
| | OLMoE-1B-7B-0924 | Qwen3-30B-A3B-Base |
|---|---|---|
| weights (bf16) | 13.8 GB | 61.1 GB |
| min VRAM | 24 GB | **80 GB** (61 GB weights + per-layer backup + activations) |
| layers / experts | 16 / 64 (olmoe) | 48 / 128 (qwen3_moe) |
| est. runtime | ~0.5 h | ~2–3 h |

- **Recommended GPU:** 1× H100 80GB (or A100 80GB). One GPU of that size runs both models sequentially.
- **Host:** 100+ GB free disk (both models in the HF cache plus outputs), 64+ GB system RAM.
- **Time and cost:** ~3–4 h total including downloads. On marketplace H100 pricing (~$1.5–2.5/h) that is about $5–10. A100-80GB is cheaper per hour but maybe 1.5–2× slower. Nothing has been rented.

## Setup and run
```bash
# on the GPU box, inside this folder (bigrun.py, analyze_big.py, run.sh, the .patch)
bash run.sh          # venv, transformers 5.9 (InstinctRazor's pinned version, fused experts),
                     # clones InstinctRazor@89bd8b8 + applies the patch, downloads CounterFact, runs both models
# outputs: out/olmoe_1b7b/ and out/qwen3_30b_a3b/ (items.json, perlayer_items.json, variants.json,
#          heldout.json, analysis.json, chart.png, fix_chart.png, log.txt), then results_bigrun.tgz
```
- Results are saved after every layer and every variant, so a crash loses at most one step.
- Copy `results_bigrun.tgz` off the box before destroying the instance.

## Validation status and risks
- **Smoke test:** the whole pipeline ran end to end on CPU with granite-3.0-1b-a400m (2 layers, n≤40, transformers 4.46). The output is in `smoke_granite_cpu/` (plumbing only; n is too small to read accuracy from).
- **Untested path:** the transformers 5.9 / CUDA path has not been run. `bigrun.py` asserts that expert tensors were found, so a version that exposes per-expert Linears fails fast instead of silently skipping the experts.
- **Bit accounting:** the effective-bits number double-counts tied embeddings. Qwen3-30B does not tie, so it is unaffected.
- **Noise:** reloading and re-quantizing the same spec gave different scores at n=40 on CPU bf16 (smoke: 37.5% vs 45% on add3). Trust the paired comparisons at n≥200 and on held-out items, not single small-n numbers.

## Budget scaling (Qwen3 run, 2026-10-16) — what was actually run
- **Budget:** $4.39 Vast credit, hard cap on spend $3.51 (20% margin). OLMoE was dropped (Qwen3 only).
- **Machine:** 1x A100 80GB PCIe (Vast offer 48064008, Japan), $1.069/h incl. 150 GB disk, $0.004/GB download, 250 GB RAM. H100 offers were $2.6–3.5/h (plus ~$0.04/GB download on most), which would have bought ~1 h; the cheaper A100 SXM4/PCIe offers seen earlier disappeared before renting. This offer is marked "deverified" by Vast but has reliability 0.999.
- **Discovery items:** `--n 200` (was 400) correct items kept per suite, from the same 600-item pools (CounterFact pool 3000). All six suites kept 200.
- **Held-out items:** `--hpool 1000` (new flag) so the held-out pool is 1000 generated items per suite; kept up to 2n = 400 correct per suite (target ≥ 200 per suite).
- **Per-layer scan:** all 48 layers individually, ordered layer 0 first, then strided (0,4,8,…,1,5,…) so that if the new `--scan-budget-min 75` (live-adjustable via `out/<tag>/SCANMIN`) cut the scan, the scanned layers would still be spread evenly. Measured ~37 s/layer, so the full scan fits.
- **Unchanged:** quantization recipe (INT3 experts / INT4 attention, group 128, 16-step clip search), variants (flat, protect L0, protect worst-1, protect worst-2), `--bs 100`.
- **Loader:** `device_map="cuda"` (stream weights straight to GPU) instead of loading to CPU then `.to()`.
- **run.sh fix:** `ls` of a missing patch glob returned non-zero under `set -o pipefail` and killed the script silently; fixed with `|| true` and an idempotent apply.
