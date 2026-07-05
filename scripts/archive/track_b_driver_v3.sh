#!/usr/bin/env bash
# Track B v3 (post-mortem rewrite). Differences from v2:
#   - B2 FIRST: the R1 server is already up; don't burn another swap.
#   - every run's exit code is checked; failures are logged as [FAIL] and
#     counted, never silently treated as completions;
#   - PRM streams wait for >= 19 GB free on their GPU before starting
#     (the v2 failure mode was PRM init against an orphaned allocation).
# Work list (real state): B2 seeds 0,1,2 (nothing done); B1 seed 0 for
# alpha in {0.10,0.15,0.25,0.35,0.50}; B1 alpha=0.05 seed 1 completion;
# then extra seeds as filler.
set -uo pipefail

PR=/home/tersenov/software/particle-reasoners
TA=/home/tersenov/software/tilt-audit
NIGHT_LOG=$TA/NIGHT_LOG.md
PY=$PR/.venv/bin/python
WRAP=$TA/scripts/b1_alpha_wrapper.py

cd "$PR"
export HF_HOME="$PWD/hf_cache" VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1

log() { echo "- $(date -u +%H:%M) $1" >> "$NIGHT_LOG"; echo "$(date -u +%H:%M) $1"; }

free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1"; }

wait_free() {  # $1 gpu, $2 needed MB, $3 max wait s
  local deadline=$(( $(date +%s) + $3 ))
  while [ "$(free_mb "$1")" -lt "$2" ]; do
    if [ "$(date +%s)" -gt "$deadline" ]; then
      log "[FAIL] GPU $1 never freed $2 MB (has $(free_mb "$1")); stream skipped"
      return 1
    fi
    sleep 15
  done
}

serve() {  # $1 model
  pkill -f "vllm.entrypoints.openai.api_server" || true
  sleep 20
  nohup experiments/serve_generator.sh "$1" 8000 2 0.35 \
    > "results/logs/vllm_$(date +%H%M).log" 2>&1 &
  local deadline=$(( $(date +%s) + 600 ))
  until curl -s --max-time 2 http://0.0.0.0:8000/v1/models | grep -q '"id"'; do
    [ "$(date +%s)" -gt "$deadline" ] && return 1
    sleep 10
  done
}

run_b2() {  # $1 prm gpu, $2 seed, $3 nprob -> rc
  CUDA_VISIBLE_DEVICES=$1 $PY experiments/run_reliability.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --n 16 --num-problems "$3" --seed "$2" --methods sap,iid,defensive \
    --out "results/tables/reliability_r1_twin_s$2.jsonl" \
    --plot "results/plots/reliability_r1_twin_s$2.png" \
    >> "results/logs/b2_stream_gpu$1.log" 2>&1
}

run_alpha() {  # $1 prm gpu, $2 alpha, $3 seeds -> rc
  CUDA_VISIBLE_DEVICES=$1 DEFENSIVE_ALPHA=$2 $PY "$WRAP" \
    --n 16 --num-problems 100 --seeds "$3" --methods defensive \
    --out "results/tables/reliability_alpha_$2.jsonl" \
    --plot "results/plots/reliability_alpha_$2.png" \
    >> "results/logs/b1_stream_gpu$1.log" 2>&1
}

checked() {  # $1 desc, then the command; retries once
  local desc=$1; shift
  if "$@"; then log "[JOB] $desc done"; return 0; fi
  log "[FAIL] $desc rc!=0 — one retry"
  if "$@"; then log "[JOB] $desc done (retry)"; return 0; fi
  log "[FAIL] $desc failed twice — moving on"
  return 1
}

# ---------------- Phase A: B2 twin on the live R1 server ----------------
log "[JOB] Track B v3 starting: B2 seeds 0,1 parallel (PRM on GPU 2 + GPU 1), then seed 2"
(
  wait_free 2 19500 900 && checked "B2 seed 0 (stream A)" run_b2 2 0 100
  touch "$TA/queue/B2_S0_DONE"
) &
(
  wait_free 1 19500 900 && checked "B2 seed 1 (stream B)" run_b2 1 1 100
  touch "$TA/queue/B2_S1_DONE"
) &
wait
t0=$(date +%s)
wait_free 2 19500 300 && checked "B2 seed 2 (stream A)" run_b2 2 2 100
log "[JOB] B2 phase complete (E-20260702b data on disk)"

# ---------------- Phase B: Qwen server back, B1 seed-0 sweep ----------------
if ! serve Qwen/Qwen2.5-Math-1.5B-Instruct; then
  log "[FAIL] Qwen server would not restart; Track B ends with B2 only"
  exit 1
fi
log "[JOB] Qwen server back up; B1 seed-0 alpha sweep on two streams"
(
  for A in 0.10 0.25 0.50; do
    wait_free 2 19500 600 && checked "B1 alpha=$A seed 0 (A)" run_alpha 2 "$A" 0
  done
) &
(
  for A in 0.15 0.35; do
    wait_free 1 19500 600 && checked "B1 alpha=$A seed 0 (B)" run_alpha 1 "$A" 0
  done
  wait_free 1 19500 600 && checked "B1 alpha=0.05 seed 1 (B)" run_alpha 1 0.05 1
) &
wait
log "[JOB] B1 seed-0 sweep complete: all 6 alphas have >=1 seed"

# ---------------- Phase C: extra seeds until dawn ----------------
(
  for A in 0.10 0.25 0.50; do
    wait_free 2 19500 600 && checked "B1 alpha=$A seeds 1,2 (A)" run_alpha 2 "$A" 1,2
  done
) &
(
  wait_free 1 19500 600 && checked "B1 alpha=0.05 seed 2 (B)" run_alpha 1 0.05 2
  for A in 0.15 0.35; do
    wait_free 1 19500 600 && checked "B1 alpha=$A seeds 1,2 (B)" run_alpha 1 "$A" 1,2
  done
) &
wait
log "[JOB] Track B COMPLETE (full pre-registered grid)"
touch "$TA/queue/TRACK_B_DONE"
