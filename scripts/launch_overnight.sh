#!/bin/bash
# Launch the 2026-07-05 constructive-diagnostics + hardening night.
# Idempotent-ish: safe to re-run; workers skip done/running ids. Detaches all
# long-lived processes with setsid so they survive this shell / the session.
#
#   bash scripts/launch_overnight.sh [HOURS]     # default 10
set -u
cd /mnt/home/tersenov/software/tilt-audit || exit 1

HOURS="${1:-10}"
BRANCH="overnight-2026-07-05"
UNTIL="$(date -u -d "+${HOURS} hours" +%H:%M)"
LOG="lab-notebook/NIGHT_LOG_2026-07-06.md"

mkdir -p queue/locks queue/done queue/failed queue/logs lab-notebook logs
rm -f queue/STOP
# launch minute-of-day, for the conductor's deadline midnight-wrap logic
.venv/bin/python -c "import datetime as d; n=d.datetime.utcnow(); print(n.hour*60+n.minute)" \
  > queue/_launch_min

# --- private night-log header (gitignored) ---------------------------------
if [ ! -f "$LOG" ]; then
  cat > "$LOG" <<EOF
# NIGHT LOG — 2026-07-05 → 07-06 · Constructive-diagnostics + hardening night

> One tagged entry per event, newest last. Tags: [JOB] [GATE] [RESULT] [STEER]
> [FAIL] [NOTE]. Times UTC. Deadline = ${UNTIL}Z. GPUs 0/1/2 (3 reserved).
> Plan: docs/plans/OVERNIGHT_2026-07-05_CONSTRUCTIVE_DIAGNOSTICS.md.
> Predictions P-20260705e/g/h/i FROZEN in RESEARCH_LOG.md, pushed on branch ${BRANCH}.
> Floor (guaranteed): gold-library expansion + net ensemble. Wildcard: mode-prober,
> SURE audit, likelihood-PPC, independent-ref KSD (piloted, promoted on escape).

EOF
fi

# --- public pre-registration + infra commit (best-effort) ------------------
git rev-parse --verify "$BRANCH" >/dev/null 2>&1 && git checkout "$BRANCH" \
  || git checkout -b "$BRANCH"
git add scripts/conductor.py scripts/night_generator.py scripts/night_monitor.sh \
        scripts/launch_overnight.sh RESEARCH_LOG.md >/dev/null 2>&1
git commit -m "overnight campaign: conductor + generator + frozen pre-registration (P-20260705e/g/h/i)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LRaU1o5Y8AszbJnef35Xq3" >/dev/null 2>&1 \
  && echo "[launch] pre-reg committed on $BRANCH"
git push -u origin "$BRANCH" >/dev/null 2>&1 \
  && echo "[launch] pushed $BRANCH" \
  || echo "[launch] push deferred (no remote/creds) — commits are local"

# --- start conductors, one per GPU, detached -------------------------------
for g in 0 1 2; do
  setsid nohup .venv/bin/python scripts/conductor.py \
      --gpu "$g" --generator --until "$UNTIL" \
      > "logs/conductor_gpu${g}.out" 2>&1 &
  echo $! > "queue/conductor_gpu${g}.pid"
done

# --- start the monitor (util + digest + commit/push) -----------------------
setsid nohup bash scripts/night_monitor.sh "$BRANCH" \
    > logs/night_monitor.out 2>&1 &
echo $! > queue/monitor.pid

sleep 2
echo "[launch] conductors on GPUs 0/1/2 + monitor up; deadline ${UNTIL}Z; branch ${BRANCH}"
echo "[launch] pids: $(cat queue/conductor_gpu0.pid) $(cat queue/conductor_gpu1.pid) $(cat queue/conductor_gpu2.pid) monitor=$(cat queue/monitor.pid)"
