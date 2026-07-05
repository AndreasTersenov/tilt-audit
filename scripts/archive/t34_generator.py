#!/usr/bin/env python
"""T3/T4 filler generator: appends jobs when the queue runs low (plan section 5).

Phases, in priority order (state in queue/gen_state.json):
  1. T3 seed densification: seeds 3..9 over the full T1 grid, one job per seed
  2. T3 N=512 arm, seeds 0..2
  3. T4.4 unbounded densification: seeds 10+ (never exhausts)
Every generated job is tier T3/T4 and tag=exploratory unless it is the
pre-registered T3 densification (seeds 3-9: tag=confirmatory-densify).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "queue"
STATE = QUEUE / "gen_state.json"
PY = ROOT / ".venv" / "bin" / "python"


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else {"next_seed": 3}
    seed = state["next_seed"]
    jobs = []
    if seed <= 9:
        jobs.append({
            "id": f"t3_seed{seed}",
            "tier": "T3",
            "gpu": "any",
            "est_min": 15,
            "cmd": (f"{PY} scripts/run_t1.py --seeds {seed} "
                    f"--tag confirmatory-densify --out results/t3_seeds.jsonl"),
        })
    elif seed == 10:
        jobs.append({
            "id": "t3_n512",
            "tier": "T3",
            "gpu": "any",
            "est_min": 30,
            "cmd": (f"{PY} scripts/run_t1.py --Ns 512 --seeds 0,1,2 "
                    f"--tag exploratory --out results/t3_n512.jsonl"),
        })
    else:
        jobs.append({
            "id": f"t4_seed{seed}",
            "tier": "T4",
            "gpu": "any",
            "est_min": 15,
            "cmd": (f"{PY} scripts/run_t1.py --seeds {seed} "
                    f"--tag exploratory --out results/t4_densify.jsonl"),
        })
    state["next_seed"] = seed + 1
    with (QUEUE / "jobs.jsonl").open("a") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")
    STATE.write_text(json.dumps(state))
    print(f"[generator] appended {[j['id'] for j in jobs]}")


if __name__ == "__main__":
    main()
