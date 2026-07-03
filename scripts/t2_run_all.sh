#!/usr/bin/env bash
# Direct T2 chain (launcher's pgrep gate deadlocked on its own parent cmdline).
# Trainings verified complete; run the three grids back-to-back on GPU 0.
set -uo pipefail
cd /home/tersenov/software/tilt-audit
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
log() { echo "- $(date -u +%H:%M) $1" >> NIGHT_LOG.md; }
for cfg in "s_clean learned:clean" "s_mis_p03 learned:mis+0.3" "s_mis_m03 learned:mis-0.3"; do
  set -- $cfg
  log "[JOB] T2 grid starting for $2 (GPU 0)"
  if .venv/bin/python scripts/run_t2.py --ckpt checkpoints/$1.pkl --label "$2" \
       --out results/t2_learned.jsonl >> queue/logs/t2_$1.log 2>&1; then
    log "[JOB] T2 grid for $2 done"
  else
    log "[FAIL] T2 grid for $2 rc!=0 (log: queue/logs/t2_$1.log) — continuing"
  fi
done
log "[JOB] T2 ALL GRIDS COMPLETE"
touch queue/T2_DONE
