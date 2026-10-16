#!/usr/bin/env bash
# One-shot runner for a single rented GPU (1x 80GB: H100 / A100-80GB). Everything lands in ./out/<tag>/.
set -euo pipefail; cd "$(dirname "$0")"
python3 -m venv .venv && source .venv/bin/activate
pip install -q "torch==2.11.0" "transformers==5.9.0" "accelerate==1.13.0" numpy scipy matplotlib sentencepiece hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
[ -d InstinctRazor ] || git clone -q https://github.com/General-Instinct/InstinctRazor
( cd InstinctRazor && git checkout -q 89bd8b8 && P=$(ls ../0001-*.patch ../../patch/0001-*.patch 2>/dev/null | head -1 || true); if git apply --check "$P" 2>/dev/null; then git apply "$P"; else git apply -R --check "$P" && echo "patch already applied"; fi )
[ -f counterfact.json ] || curl -sLo counterfact.json https://rome.baulab.info/data/dsets/counterfact.json
export IR_SRC="$PWD/InstinctRazor/src/quant" COUNTERFACT="$PWD/counterfact.json"
nvidia-smi | tee gpu.txt
# Qwen3-30B-A3B-Base only (budget run; see PLAN.md "Budget scaling"). Results are saved after every step in out/qwen3_30b_a3b/.
python3 bigrun.py --model Qwen/Qwen3-30B-A3B-Base --tag qwen3_30b_a3b --n ${N:-200} --hpool ${HPOOL:-1000} --bs ${BS:-100} --scan-budget-min ${SCANMIN:-75} 2>&1 | tee qwen3.log
python3 analyze_big.py out/qwen3_30b_a3b || true
tar czf results_bigrun.tgz out *.log gpu.txt && echo "ALL DONE -> results_bigrun.tgz"
