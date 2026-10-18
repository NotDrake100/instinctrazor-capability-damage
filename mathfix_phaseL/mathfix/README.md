# mathfix: is the GSM8K collapse in sleep-shield arm b format loss or real forgetting? (diagnose first, fix second)

The files below were built and CPU-smoke-tested on 2026-10-17. **No GPU run has happened, so there are no real
results yet.**

| file | what |
|---|---|
| `common.py` | model, probes, fact stream, v2 parser, chat format, LoRA config, `train_night`, all **verbatim from ../sleep_shield.py**. Probe/fact hashes are asserted (62a39450… / f43d375c…). Adds the lenient parser, reasoning-step and failure-category rules, and generation with token counts. |
| `predictions.md` | H1/H2/H3 predictions for (b)–(g) and the automatic decision rules. Written before any run. |
| `phase1.py` | Phase 1, arm b only, seeds 0–2, 7 nights. (a) raw GSM8K jsonl, (b) failure categories, strict vs lenient, review dump, (c) per-layer drift/KL/shift coherence, (d) logit lens, (e) layer and 3-block swap, (f) shared-direction removal + own-top-1 + random controls, (g) prefill test. |
| `phase1_summarize.py` | Tables, rule-based classification, `phase1_summary.json` (incl. `phase2_choice`), `phase1_report.md`, `failures_review.md` (50 random failures for nights 1/4/7). |
| `PREREG_phase2.md` | Phase-2 arms per mechanism and the PASS criteria. Written before any run. |
| `phase2.py` | Phase 2. Reads `phase1_summary.json`. H1: b/r/c (replay vs token-matched Dolly filler). H2: b/p/rnd/bot. H3: b/x/rr. `--config phase2_1p5b` = Qwen2.5-1.5B-Instruct confirmation. |
| `run_phase1.sh`, `run_phase2.sh`, `_deps.sh`, `requirements.txt` | `pip install transformers peft datasets accelerate`, then nohup to `log.txt`. |
| `tests_cpu.py` | Correctness checks on the real 0.5B (CPU): LoRA swap, merged-edit path, micro-batch loss, SVD edits. Output in `smoke/tests_cpu_output.txt`. |
| `smoke/` | Tiny-model CPU smoke runs of phase 1 and phase 2 (H1/H2/H3), all exit 0. Numbers are meaningless (random tiny model). |

## Run (Vast pytorch box)
```
bash run_phase1.sh            # ~1-1.5 h on a 2080 Ti (estimate, see below); re-run the same command to resume
# when log.txt says "PHASE 1 DONE": read phase1_out/phase1_report.md, hand-review phase1_out/failures_review.md
bash run_phase2.sh            # mechanism from phase1_summary.json; RUN_1P5B=1 adds the 1.5B confirmation
```

## Runtime estimate, Phase 1 (2080 Ti; not measured yet, built from the sleep_shield Vast log)
| part | estimate |
|---|---|
| 21 nights of training + nightly eval | ~20–25 min (Vast: ~21 s of training per night, ~30–60 s of eval) |
| base eval + prefill | ~3 min |
| drift + logit lens | ~2–3 min |
| layer swap: 3 seeds × 34 configs (none, 24 single layers, 8 blocks, all) × (GSM8K on 100 items + recall on 1,400 questions) | ~20 min if outputs stay short, up to ~50 min if swaps bring back 512-token reasoning |
| shared-direction edits: 3 seeds × 5 variants | ~8 min |
| **total** | **~1.0–1.5 h** (T4 maybe ~1.3–2 h) |

`phase1.py` logs a measured estimate after night 1. If it is far above 2 h, restart on a fresh `--out` with
`--swap_gsm_max_new 384`. That cap affects only the (e)/(f) re-scoring, never the training evaluation.

Phase 2 at 0.5B: roughly 1.5 h for H1, because the r and c steps carry ~250-token extra sequences. H2 and H3 take about
1–1.2 h. The 1.5B config is about 3× slower and should run on a GPU with 16 GB or more.
