#!/usr/bin/env python
"""45-min digest: compact state of tonight's JSONLs (run-plan section 7 cadence)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def rows(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.open()]


def main():
    ksd = rows("results/ksd_trial.jsonl")
    tra = rows("results/transfer.jsonl")

    by = defaultdict(list)
    for r in ksd:
        if r.get("arm") in ("power", "mixture", "wrongref"):
            by[(r["arm"], r["config"], r["budget"], r["kernel"])].append(r)
    if by:
        print("== KSD trial (detect rate | median ratio-to-null-q95) ==")
        arms = sorted({k[0] for k in by})
        for arm in arms:
            print(f"-- {arm} --")
            configs = sorted({k[1] for k in by if k[0] == arm})
            budgets = sorted({k[2] for k in by if k[0] == arm})
            for cfg in configs:
                line = f"{cfg:>14} "
                for bud in budgets:
                    rr = by.get((arm, cfg, bud, "imq_paper"), [])
                    if rr:
                        det = np.mean([r["detect"] for r in rr])
                        rat = np.median([r["ratio_q95"] for r in rr])
                        line += f"| N{bud}: {det:.2f} ({rat:.2f}x) "
                    else:
                        line += f"| N{bud}:   --      "
                print(line)

    if tra:
        print("== transfer vs gold (median mmd2 / swd2 / bp0 cov) ==")
        byt = defaultdict(list)
        for r in tra:
            byt[(r["n"], r["tilt"], r["sampler"])].append(r)
        for k in sorted(byt, key=str):
            rr = byt[k]
            mm = np.median([r["mmd2"] for r in rr])
            sw = np.median([r["swd2"] for r in rr])
            bp = np.median([r["bp0_cov68"] for r in rr])
            print(f"n{k[0]} {k[1]:>6} {k[2]:>13}: mmd2={mm:.3e} "
                  f"swd2={sw:.3e} bp0={bp:.2f} ({len(rr)} rows)")

    golds = sorted(Path("results/gold").glob("gold_*.json"))
    if golds:
        print(f"== gold library: {len(golds)} configs ==")
        bad = []
        for g in golds:
            m = json.loads(g.read_text())
            if m["rhat_max"] >= 1.01 or m["ess_min"] <= 400:
                bad.append((g.stem, m["rhat_max"], m["ess_min"]))
        print("T-L2: all pass" if not bad else f"T-L2 FAILURES: {bad}")


if __name__ == "__main__":
    main()
