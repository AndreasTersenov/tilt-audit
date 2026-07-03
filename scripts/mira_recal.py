#!/usr/bin/env python
"""MIRA recalibration pass (A2, exploratory-diagnostic).

Motivation (measured tonight): mira-score's truth-based min-max norm gives an
oracle false-positive rate of 0.65-0.80 at q=4096 under the analytic-z rule
(same asymmetric-normalization bug class as tarp's norm=True, found by the
T-N3-style battery null); and a 20-rep two-sided rank calibration cannot
reach alpha=0.05 (min p = 2/21). This pass:

 1. appends oracle null reps 20..59 for the truth-norm variant (test='mira'),
 2. appends a symmetric-wrap variant (test='mira_sym'): inputs standardized
    by pooled-SAMPLE mean/std, mapped to ~unit cube, norm=False — truths
    never enter the transform,
 3. appends corrected empirically-calibrated verdicts as test='mira_cal'
    (ONE-sided: score below null = the known failure direction) for both
    variants, same (test,config,budget,rep) keys — analysis is keep-last.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from mira_score import mira

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "a2_power.jsonl"
ARCH = ROOT / "results" / "archives"

CONFIGS = ["oracle", "dps", "sap", "twisted", "dps_em03", "dps_ep03"]
BUDGETS = [64, 256]
MIRA_RUNS = 64

torch.set_num_threads(16)
_cache = {}


def load(name):
    if name not in _cache:
        _cache[name] = dict(np.load(ARCH / name))
    return _cache[name]


def mira_score_for(config, budget, rep, sym):
    d = load(f"cond_{config}.npz")
    L, S, q = d["samples"].shape
    rng = np.random.default_rng(hash(("mira", config, budget, rep)) & 0xFFFFFFFF)
    li = rng.choice(L, L, replace=True)
    si = rng.choice(S, budget, replace=False)
    samples = d["samples"][li][:, si].astype(np.float64)
    truths = d["truths"][li].astype(np.float64)
    if sym:
        m = samples.mean((0, 1))
        s = samples.std((0, 1))
        samples = ((samples - m) / s + 4.0) / 8.0
        truths = ((truths - m) / s + 4.0) / 8.0
    torch.manual_seed(hash(("mira", config, budget, rep, sym)) & 0x7FFFFFFF)
    mean, _ = mira(torch.as_tensor(truths), torch.as_tensor(samples[None]),
                   num_runs=MIRA_RUNS, norm=not sym, disable_tqdm=True)
    return float(mean[0])


def main():
    scores = {}  # (variant, config, budget, rep) -> score
    # reuse existing truth-norm rows
    for line in OUT.read_text().splitlines():
        r = json.loads(line)
        if r.get("test") == "mira":
            scores[("mira", r["config"], r["budget"], r["rep"])] = r["stat"]

    new_rows = []
    for budget in BUDGETS:
        for rep in range(20, 60):  # extra oracle nulls, truth-norm variant
            if ("mira", "oracle", budget, rep) not in scores:
                s = mira_score_for("oracle", budget, rep, sym=False)
                scores[("mira", "oracle", budget, rep)] = s
                new_rows.append(dict(test="mira", config="oracle",
                                     budget=budget, rep=rep, stat=s,
                                     num_runs=MIRA_RUNS, tag="nullpad"))
        for config in CONFIGS:
            nreps = 60 if config == "oracle" else 20
            for rep in range(nreps):
                s = mira_score_for(config, budget, rep, sym=True)
                scores[("mira_sym", config, budget, rep)] = s
                new_rows.append(dict(test="mira_sym", config=config,
                                     budget=budget, rep=rep, stat=s,
                                     num_runs=MIRA_RUNS, tag="confirmatory"))
        print(f"budget {budget} scored", flush=True)

    # one-sided empirical calibration for both variants
    for variant, cal in (("mira", "mira_cal"), ("mira_sym", "mira_sym_cal")):
        for budget in BUDGETS:
            null = np.asarray([v for (t, c, b, _), v in scores.items()
                               if t == variant and c == "oracle" and b == budget])
            for config in CONFIGS:
                if config == "oracle":
                    continue
                for rep in range(20):
                    key = (variant, config, budget, rep)
                    if key not in scores:
                        continue
                    s = scores[key]
                    p = float((np.sum(null <= s) + 1) / (len(null) + 1))
                    new_rows.append(dict(test=cal, config=config,
                                         budget=budget, rep=rep, stat=s,
                                         pvalue=p, detected=bool(p < 0.05),
                                         n_null=int(len(null)),
                                         tag="confirmatory"))

    with OUT.open("a") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")
    print(f"appended {len(new_rows)} rows", flush=True)


if __name__ == "__main__":
    main()
