#!/usr/bin/env bash
# GPU-1 Track B chain (dual-server plan, 23:52 steer):
#   Qwen server on GPU 1 :8001 + PRM on GPU 1 -> B1 alpha sweep seed 0
#   then alpha=0.05 seeds 1,2 -> then kill local Qwen, PRM stays on GPU 1,
#   B2 seed 1 against the R1 server on :8000 (GPU 2) until dawn.
# Sequential, rc-checked, no cross-GPU orchestration to race.
set -uo pipefail
PR=/home/tersenov/software/particle-reasoners
TA=/home/tersenov/software/tilt-audit
NIGHT_LOG=$TA/NIGHT_LOG.md
PY=$PR/.venv/bin/python
WRAP=$TA/scripts/b1_alpha_wrapper.py
cd "$PR"
export HF_HOME="$PWD/hf_cache" VLLM_USE_FLASHINFER_SAMPLER=0 PYTHONUNBUFFERED=1

log() { echo "- $(date -u +%H:%M) $1" >> "$NIGHT_LOG"; echo "$(date -u +%H:%M) $1"; }

# wait for the GPU-1 Qwen server
deadline=$(( $(date +%s) + 900 ))
until curl -s --max-time 2 http://0.0.0.0:8001/v1/models | grep -q '"id"'; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "[FAIL] Qwen-on-GPU1 server never came up; GPU-1 chain aborting"
    exit 1
  fi
  sleep 10
done
log "[JOB] Qwen server up on GPU 1 :8001; B1 seed-0 sweep starting (PRM co-resident on GPU 1)"

b1() {  # $1 alpha, $2 seeds
  CUDA_VISIBLE_DEVICES=1 DEFENSIVE_ALPHA=$1 $PY "$WRAP" \
    --endpoint http://0.0.0.0:8001/v1 \
    --n 16 --num-problems 100 --seeds "$2" --methods defensive \
    --out "results/tables/reliability_alpha_$1.jsonl" \
    --plot "results/plots/reliability_alpha_$1.png" \
    >> results/logs/b1_gpu1_chain.log 2>&1
}

for A in 0.10 0.15 0.25 0.35 0.50; do
  if b1 "$A" 0; then log "[JOB] B1 alpha=$A seed 0 done"; else
    log "[FAIL] B1 alpha=$A seed 0 rc!=0 — one retry"
    if b1 "$A" 0; then log "[JOB] B1 alpha=$A seed 0 done (retry)"; else
      log "[FAIL] B1 alpha=$A seed 0 failed twice — continuing sweep"; fi
  fi
done
log "[JOB] B1 seed-0 sweep COMPLETE (all 6 alphas have seed 0)"
touch "$TA/queue/B1_SEED0_DONE"

if b1 0.05 2; then log "[JOB] B1 alpha=0.05 seed 2 done (seeds 0,1,2 complete at 0.05)"; fi

# hand GPU 1 to B2 seed 1: kill only the :8001 server (match the port arg)
pkill -f "port 8001" 2>/dev/null || pkill -f "api_server --model Qwen" 2>/dev/null || true
sleep 15
log "[JOB] GPU-1 Qwen server down; B2 seed 1 starting against R1 server :8000"
CUDA_VISIBLE_DEVICES=1 $PY experiments/run_reliability.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --n 16 --num-problems 100 --seed 1 --methods sap,iid,defensive \
  --out results/tables/reliability_r1_twin_s1.jsonl \
  --plot results/plots/reliability_r1_twin_s1.png \
  >> results/logs/b2_gpu1_chain.log 2>&1
log "[JOB] B2 seed 1 finished rc=$? (or was running at dawn)"
touch "$TA/queue/TRACK_B_GPU1_DONE"
