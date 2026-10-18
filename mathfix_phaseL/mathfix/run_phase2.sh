#!/usr/bin/env bash
# PHASE 2 (fix). Run ONLY after phase 1 finished (needs phase1_out/phase1_summary.json).
#   bash run_phase2.sh                         # 0.5B, mechanism chosen by phase 1
#   RUN_1P5B=1 bash run_phase2.sh              # then the Qwen2.5-1.5B confirmation, sequentially in the same job
#   MECH=H1 bash run_phase2.sh                 # override the phase-1 choice (logged as override)
set -euo pipefail
cd "$(dirname "$0")"
P1="${P1:-$(pwd)/phase1_out}"
[ -f "$P1/phase1_summary.json" ] || { echo "no $P1/phase1_summary.json -- run phase 1 first"; exit 1; }
source ./_deps.sh
EXTRA=(); [ -n "${MECH:-}" ] && EXTRA=(--mechanism "$MECH")
CMD="python -u phase2.py --config phase2_0p5b --phase1 $P1 ${EXTRA[*]:-}"
[ "${RUN_1P5B:-0}" = "1" ] && CMD="$CMD && python -u phase2.py --config phase2_1p5b --phase1 $P1 ${EXTRA[*]:-}"
nohup bash -c "$CMD" >> log.txt 2>&1 &
echo "phase 2 started, PID $! | log: $(pwd)/log.txt | follow: tail -f $(pwd)/log.txt"
