#!/usr/bin/env bash
# Track B driver (GPU 2, one linear chain): B1 alpha-sweep -> sap/iid anchors -> B2 R1 twin.
# Appends [JOB] lines to tilt-audit/NIGHT_LOG.md as steps start/finish. NOT -e: one step
# failing must not kill the chain (per-run JSONL is incremental, partial night scoreable).
set -uo pipefail

PR=<predecessor-project>
TA=.
NIGHT_LOG=$TA/NIGHT_LOG.md
PY=$PR/.venv/bin/python
WRAP=$TA/scripts/b1_alpha_wrapper.py

cd "$PR"
export HF_HOME="$PWD/hf_cache" VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=2

log() { echo "- $(date -u +%H:%M) $1" >> "$NIGHT_LOG"; echo "$(date -u +%H:%M) $1"; }

wait_server() {  # wait up to $1 seconds for the vLLM server to answer
  local deadline=$(( $(date +%s) + $1 ))
  until curl -s --max-time 2 http://0.0.0.0:8000/v1/models | grep -q '"id"'; do
    [ "$(date +%s)" -gt "$deadline" ] && return 1
    sleep 10
  done
}

# ---------- B1: defensive-mixture alpha sweep (E-20260702a) ----------
log "[JOB] B1 alpha-sweep starting (defensive, N=16, 100 problems, seeds 0,1,2; alpha in 0.05..0.50)"
for ALPHA in 0.05 0.10 0.15 0.25 0.35 0.50; do
  t0=$(date +%s)
  DEFENSIVE_ALPHA=$ALPHA $PY "$WRAP" \
    --n 16 --num-problems 100 --seeds 0,1,2 --methods defensive \
    --out "results/tables/reliability_alpha_${ALPHA}.jsonl" \
    --plot "results/plots/reliability_alpha_${ALPHA}.png" \
    >> results/logs/b1_alpha_sweep.log 2>&1
  rc=$?
  mins=$(( ($(date +%s) - t0) / 60 ))
  log "[JOB] B1 alpha=$ALPHA finished rc=$rc (${mins}m) -> reliability_alpha_${ALPHA}.jsonl"
done

# ---------- B1 anchors: sap + iid at one seed (pre-registered 'if time') ----------
log "[JOB] B1 anchors starting (sap,iid, seed 0, 100 problems)"
$PY experiments/run_reliability.py \
  --n 16 --num-problems 100 --seed 0 --methods sap,iid \
  --out results/tables/reliability_anchors_s0.jsonl \
  --plot results/plots/reliability_anchors_s0.png \
  >> results/logs/b1_anchors.log 2>&1
log "[JOB] B1 anchors finished rc=$?"

# ---------- server swap: Qwen -> R1-distill ----------
log "[JOB] B1 done; swapping GPU-2 server to DeepSeek-R1-Distill-Qwen-1.5B for B2"
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 20
nohup experiments/serve_generator.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 8000 2 0.35 \
  > results/logs/vllm_b2.log 2>&1 &
if ! wait_server 600; then
  log "[FAIL] R1 vLLM server did not come up in 10 min; retrying once"
  pkill -f "vllm.entrypoints.openai.api_server" || true
  sleep 20
  nohup experiments/serve_generator.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 8000 2 0.35 \
    > results/logs/vllm_b2_retry.log 2>&1 &
  if ! wait_server 600; then
    log "[FAIL] R1 server failed twice - abandoning B2 per playbook; GPU 2 free for Track A"
    exit 1
  fi
fi
log "[JOB] R1 server up; B2 twin starting (sap,iid,defensive, N=16, seeds 0,1,2)"

# ---------- B2: R1-distill twin (E-20260702b), per-seed with pace guard ----------
NPROB=100
for SEED in 0 1 2; do
  t0=$(date +%s)
  $PY experiments/run_reliability.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --n 16 --num-problems $NPROB --seed $SEED --methods sap,iid,defensive \
    --out "results/tables/reliability_r1_twin_s${SEED}.jsonl" \
    --plot "results/plots/reliability_r1_twin_s${SEED}.png" \
    >> results/logs/b2_r1_twin.log 2>&1
  rc=$?
  mins=$(( ($(date +%s) - t0) / 60 ))
  log "[JOB] B2 seed=$SEED finished rc=$rc (${mins}m, ${NPROB} problems)"
  if [ "$mins" -gt 100 ]; then
    NPROB=60
    log "[STEER] B2 pace <25 problems/h at seed $SEED -> trimming remaining seeds to 60 problems (pre-authorized in plan par.2)"
  fi
done

pkill -f "vllm.entrypoints.openai.api_server" || true
log "[JOB] Track B chain complete; GPU 2 released to Track A (T3)"
touch "$TA/queue/TRACK_B_DONE"
