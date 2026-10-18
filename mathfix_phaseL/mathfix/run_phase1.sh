#!/usr/bin/env bash
# PHASE 1 (diagnosis). Usage on a Vast pytorch box:   bash run_phase1.sh            (resumable: just run again)
#   optional: OUT=/workspace/p1 bash run_phase1.sh --swap_gsm_max_new 384   (only if the logged estimate is far over 2 h)
set -euo pipefail
cd "$(dirname "$0")"
source ./_deps.sh
OUT="${OUT:-$(pwd)/phase1_out}"
nohup python -u phase1.py --out "$OUT" "$@" >> log.txt 2>&1 &
echo "phase 1 started, PID $! | log: $(pwd)/log.txt (also $OUT/log.txt) | follow: tail -f $(pwd)/log.txt"
