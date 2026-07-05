#!/bin/bash
# Overnight monitor: samples GPU utilization every 15 min into queue/util.log,
# and every 45 min runs night_digest.py + commits/pushes results to the
# overnight branch. Runs until queue/STOP appears (set by the conductor at the
# deadline). Detached; its internal sleeps are fine (it is its own process).
cd /mnt/home/tersenov/software/tilt-audit || exit 1
BRANCH="${1:-overnight-2026-07-05}"

commit_push() {
  { echo "===== digest $(date -u +%H:%M)Z ====="; .venv/bin/python scripts/night_digest.py; } \
    >> logs/night_digest.log 2>&1
  git add -A results figures RESEARCH_LOG.md >/dev/null 2>&1
  if git commit -m "overnight $(date -u +%H:%M)Z: results checkpoint" >/dev/null 2>&1; then
    git pull --rebase --autostash origin "$BRANCH" >/dev/null 2>&1
    if git push origin "$BRANCH" >/dev/null 2>&1; then
      echo "$(date -u +%H:%M)Z committed+pushed" >> logs/night_monitor.out
    else
      echo "$(date -u +%H:%M)Z committed locally (push failed; will retry)" >> logs/night_monitor.out
    fi
  fi
}

i=0
while [ ! -f queue/STOP ]; do
  {
    echo -n "$(date -u +%H:%M)Z "
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader | tr '\n' '|'
    echo
  } >> queue/util.log 2>&1
  i=$((i + 1))
  [ $((i % 3)) -eq 0 ] && commit_push
  sleep 900
done
commit_push
echo "$(date -u +%H:%M)Z monitor stopped (STOP seen)" >> logs/night_monitor.out
