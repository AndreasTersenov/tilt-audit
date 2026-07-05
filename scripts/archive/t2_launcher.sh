#!/usr/bin/env bash
# Waits for each score-net checkpoint, then runs its T2 grid on GPU 0.
# Sequential: trainings finish together (~02:05), then the three grids run
# back-to-back. Confirmatory tier T2.
set -uo pipefail
TA=/home/tersenov/software/tilt-audit
cd "$TA"
log() { echo "- $(date -u +%H:%M) $1" >> NIGHT_LOG.md; }
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false

run_t2() {  # $1 ckpt basename, $2 label
  until [ -f "checkpoints/$1.pkl" ]; do sleep 60; done
  # wait until training is REALLY done (file written at completion has step=60000;
  # 30-min interim checkpoints also write the file — check the trainer exited)
  while pgrep -f "train_score.py.*$1" > /dev/null; do sleep 60; done
  log "[JOB] T2 grid starting for $2 (GPU 0)"
  if .venv/bin/python scripts/run_t2.py --ckpt "checkpoints/$1.pkl" --label "$2" \
       --out results/t2_learned.jsonl >> queue/logs/t2_$1.log 2>&1; then
    log "[JOB] T2 grid for $2 done"
  else
    log "[FAIL] T2 grid for $2 rc!=0 — one retry"
    if .venv/bin/python scripts/run_t2.py --ckpt "checkpoints/$1.pkl" --label "$2" \
         --out results/t2_learned.jsonl >> queue/logs/t2_$1.log 2>&1; then
      log "[JOB] T2 grid for $2 done (retry)"
    else
      log "[FAIL] T2 grid for $2 failed twice"
    fi
  fi
}

run_t2 s_clean "learned:clean"
run_t2 s_mis_p03 "learned:mis+0.3"
run_t2 s_mis_m03 "learned:mis-0.3"
log "[JOB] T2 ALL GRIDS COMPLETE"
touch queue/T2_DONE
