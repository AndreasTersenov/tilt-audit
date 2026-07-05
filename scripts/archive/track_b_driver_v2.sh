#!/usr/bin/env bash
# Track B v2 (23:45 restructure): measured pace 200 rows/h means the full
# pre-registered B1 (6 alpha x 3 seeds) alone would eat ~9 h. Reallocation,
# priority order: (1) full alpha sweep at seed 0 — the scientific core of
# E-20260702a is the alpha SHAPE; (2) B2 twin complete (E-20260702b);
# (3) B1 seeds 1,2 as filler. Two parallel streams double throughput:
# stream A keeps its PRM on GPU 2 (with the generation server), stream B
# puts its PRM on GPU 1 (queue workers there are no-prealloc now).
# Anchors (sap/iid) DROPPED: prior-night measurements on the identical Qwen
# setup already exist; B2 is new science, anchors were "if time".
set -uo pipefail

PR=/home/tersenov/software/particle-reasoners
TA=/home/tersenov/software/tilt-audit
NIGHT_LOG=$TA/NIGHT_LOG.md
PY=$PR/.venv/bin/python
WRAP=$TA/scripts/b1_alpha_wrapper.py

cd "$PR"
export HF_HOME="$PWD/hf_cache" VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1

log() { echo "- $(date -u +%H:%M) $1" >> "$NIGHT_LOG"; echo "$(date -u +%H:%M) $1"; }

run_alpha() {  # $1 = gpu for PRM, $2 = alpha, $3 = seeds
  CUDA_VISIBLE_DEVICES=$1 DEFENSIVE_ALPHA=$2 $PY "$WRAP" \
    --n 16 --num-problems 100 --seeds "$3" --methods defensive \
    --out "results/tables/reliability_alpha_$2.jsonl" \
    --plot "results/plots/reliability_alpha_$2.png" \
    >> "results/logs/b1_stream_gpu$1.log" 2>&1
}

# ---------------- Phase 1: finish the alpha sweep at seed 0 ----------------
log "[STEER] Track B restructured (pace 200 rows/h, ~3x plan estimate): two parallel PRM streams (GPU 2 + GPU 1), priority = full alpha sweep at seed 0, then complete B2 twin, then B1 seeds 1,2 as filler. Anchors dropped (prior-night sap/iid measurements exist). E-entry expectations unchanged."
(
  for A in 0.10 0.25 0.50; do run_alpha 2 "$A" 0; log "[JOB] B1 alpha=$A seed 0 done (stream A)"; done
  touch "$TA/queue/B1_PHASE1_A_DONE"
) &
(
  for A in 0.15 0.35; do run_alpha 1 "$A" 0; log "[JOB] B1 alpha=$A seed 0 done (stream B)"; done
  run_alpha 1 0.05 1   # finish the interrupted alpha=0.05 seed 1
  log "[JOB] B1 alpha=0.05 seed 1 done (stream B)"
  touch "$TA/queue/B1_PHASE1_B_DONE"
) &
wait
log "[JOB] B1 phase 1 complete: all 6 alphas at seed 0 (plus 0.05 seeds 0,1)"

# ---------------- Phase 2: server swap to R1, B2 twin in parallel ----------------
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 20
nohup experiments/serve_generator.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 8000 2 0.35 \
  > results/logs/vllm_b2.log 2>&1 &
deadline=$(( $(date +%s) + 600 ))
until curl -s --max-time 2 http://0.0.0.0:8000/v1/models | grep -q '"id"'; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "[FAIL] R1 server did not come up in 10 min; one retry"
    pkill -f "vllm.entrypoints.openai.api_server" || true; sleep 20
    nohup experiments/serve_generator.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 8000 2 0.35 \
      > results/logs/vllm_b2_retry.log 2>&1 &
    deadline=$(( $(date +%s) + 600 ))
    until curl -s --max-time 2 http://0.0.0.0:8000/v1/models | grep -q '"id"'; do
      if [ "$(date +%s)" -gt "$deadline" ]; then
        log "[FAIL] R1 server failed twice - abandoning B2 per playbook"
        exit 1
      fi
      sleep 10
    done
    break
  fi
  sleep 10
done
log "[JOB] R1 server up; B2 twin starting (sap,iid,defensive; N=16; 100 problems; parallel seeds on 2 streams)"

run_b2() {  # $1 = prm gpu, $2 = seed, $3 = nprob
  CUDA_VISIBLE_DEVICES=$1 $PY experiments/run_reliability.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --n 16 --num-problems "$3" --seed "$2" --methods sap,iid,defensive \
    --out "results/tables/reliability_r1_twin_s$2.jsonl" \
    --plot "results/plots/reliability_r1_twin_s$2.png" \
    >> "results/logs/b2_stream_gpu$1.log" 2>&1
}

t0=$(date +%s)
( run_b2 2 0 100; log "[JOB] B2 seed 0 done (stream A)" ) &
( run_b2 1 1 100; log "[JOB] B2 seed 1 done (stream B)" ) &
wait
mins=$(( ($(date +%s) - t0) / 60 ))
if [ "$mins" -gt 150 ]; then
  NPROB=60
  log "[STEER] B2 pace slow (${mins}m for seeds 0,1) -> seed 2 trimmed to 60 problems (trim pre-authorized in plan par.2)"
else
  NPROB=100
fi
run_b2 2 2 $NPROB
log "[JOB] B2 seed 2 done (${NPROB} problems) — E-20260702b data complete"

# ---------------- Phase 3: B1 extra seeds as filler ----------------
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 20
nohup experiments/serve_generator.sh Qwen/Qwen2.5-Math-1.5B-Instruct 8000 2 0.35 \
  > results/logs/vllm_b1_phase3.log 2>&1 &
deadline=$(( $(date +%s) + 600 ))
until curl -s --max-time 2 http://0.0.0.0:8000/v1/models | grep -q '"id"'; do
  [ "$(date +%s)" -gt "$deadline" ] && { log "[FAIL] Qwen server no restart; phase 3 skipped"; exit 0; }
  sleep 10
done
log "[JOB] B1 phase 3 (extra seeds) starting"
( for A in 0.10 0.25 0.50; do run_alpha 2 "$A" 1,2; log "[JOB] B1 alpha=$A seeds 1,2 done (A)"; done ) &
( run_alpha 1 0.05 2; for A in 0.15 0.35; do run_alpha 1 "$A" 1,2; log "[JOB] B1 alpha=$A seeds 1,2 done (B)"; done ) &
wait
log "[JOB] Track B COMPLETE (full pre-registered grid)"
touch "$TA/queue/TRACK_B_DONE"
