#!/usr/bin/env bash
# One-shot runner for a single rented GPU (1x 80GB: H100 / A100-80GB). Everything lands in ./out/<tag>/.
set -euo pipefail; cd "$(dirname "$0")"
python3 -m venv .venv && source .venv/bin/activate
pip install -q "torch==2.11.0" "transformers==5.9.0" "accelerate==1.13.0" numpy scipy matplotlib sentencepiece hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
[ -d InstinctRazor ] || git clone -q https://github.com/General-Instinct/InstinctRazor
( cd InstinctRazor && git checkout -q 89bd8b8 && git apply ../0001-quant_save-protect-layers-keeps-chosen-decoder-layer.patch )
[ -f counterfact.json ] || curl -sLo counterfact.json https://rome.baulab.info/data/dsets/counterfact.json
export IR_SRC="$PWD/InstinctRazor/src/quant" COUNTERFACT="$PWD/counterfact.json"
nvidia-smi | tee gpu.txt
# 1) small second MoE first (fast, catches setup bugs): OLMoE-1B-7B (olmoe, OPD-capable family)
python3 bigrun.py --model allenai/OLMoE-1B-7B-0924 --tag olmoe_1b7b --n 400 --bs 64 2>&1 | tee olmoe.log
python3 analyze_big.py out/olmoe_1b7b
# 2) main target: Qwen3-30B-A3B base (qwen3_moe, the family InstinctRazor validated OPD on)
python3 bigrun.py --model Qwen/Qwen3-30B-A3B-Base --tag qwen3_30b_a3b --n 400 --bs 48 2>&1 | tee qwen3.log
python3 analyze_big.py out/qwen3_30b_a3b
tar czf results_bigrun.tgz out *.log gpu.txt && echo "ALL DONE -> results_bigrun.tgz"
