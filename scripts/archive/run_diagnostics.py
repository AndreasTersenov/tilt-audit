#!/usr/bin/env python
"""A2 power battery: PQMass / TARP / MIRA vs oracle-generated failures of
exactly known size. CPU-only (<=50 joblib workers); runs OUTSIDE the GPU
queue.

What each test consumes (the wiring matters and is easy to get wrong):

  PQMass  (unconditional, two-sample): budget-many UNWEIGHTED z-samples from
          a failure archive vs the same budget from the independent
          oracle_ref bank (single fixed y, shift-calibrated b). Output:
          p-value under the chi^2 null. oracle_null-vs-oracle_ref is the
          built-in same-vs-same control at every budget.

  TARP    (conditional, coverage): the cond_*.npz sets — L=128 observations
          y_l with generating truths z*_l and S=256 candidate posterior
          samples per y, all at b=s^2 (beta=1; the only b where truths are
          exchangeable target draws). Statistic: max_alpha |ecp - alpha| at a
          sample sub-budget. Detection is calibrated EMPIRICALLY: p-value =
          rank of the statistic within the cond_oracle null reps at matched
          budget.

  MIRA    (conditional, score): same cond sets; score with analytic null
          (2N+3)/(3(N+1)) (N = sub-budget - 1) and band sqrt(1/(18L));
          two-sided z-test p-value. num_runs=64 per rep (recorded).

Each rep bootstrap-resamples the L truths (conditional tests) or re-draws the
budget subsample (PQMass) — 20 reps per cell for error bars.

Output rows -> results/a2_power.jsonl (append-mode).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARCH = ROOT / "results" / "archives"

PQM_CONFIGS = ["oracle_null", "dps", "sap", "twisted", "dps_em03",
               "dps_ep03", "twisted_em03", "dps_s05", "dps_s2"]
COND_CONFIGS = ["oracle", "dps", "sap", "twisted", "dps_em03", "dps_ep03"]
PQM_BUDGETS = [64, 256, 1024, 4096]
COND_BUDGETS = [64, 256]
REPS = 20
MIRA_RUNS = 64

_cache = {}


def load(name):
    if name not in _cache:
        _cache[name] = dict(np.load(ARCH / name))
    return _cache[name]


def num_refs_for(budget):
    return max(16, min(100, budget // 8))


def pqmass_cell(config, budget, rep):
    from pqm import pqm_pvalue
    x = load(f"{config}.npz")["z"]
    ref = load("oracle_ref.npz")["z"]
    rng = np.random.default_rng(hash((config, budget, rep)) & 0xFFFFFFFF)
    xi = rng.choice(x.shape[0], budget, replace=False)
    ri = rng.choice(ref.shape[0], budget, replace=False)
    np.random.seed((hash(("pqm", config, budget, rep)) & 0x7FFFFFFF))
    t0 = time.time()
    p = float(pqm_pvalue(x[xi], ref[ri], num_refs=num_refs_for(budget)))
    return dict(test="pqmass", config=config, budget=budget, rep=rep,
                stat=p, pvalue=p, detected=bool(p < 0.05),
                num_refs=num_refs_for(budget), wall=round(time.time() - t0, 2))


def tarp_wrap(samples, truths):
    """Symmetric pre-standardization; tarp called with norm=False.

    tarp's norm=True min-max normalizes by the TRUTHS' empirical range: every
    truth is inside the box by construction while fresh samples fall outside
    in ~2q/L dims each, so sample distances are asymmetrically inflated —
    d-extensive null miscalibration (measured: max|ecp-alpha| = 0.20 at
    q=4096 on the EXACT posterior; 0.04-0.08 after this wrap). Standardize by
    pooled-SAMPLE mean/std (truths never enter the transform), map +-4sd to
    [0,1] so the uniform references land in the mass."""
    m = samples.mean((0, 1))
    s = samples.std((0, 1))
    return ((samples - m) / s + 4.0) / 8.0, ((truths - m) / s + 4.0) / 8.0


def tarp_cell(config, budget, rep):
    from tarp import get_tarp_coverage
    d = load(f"cond_{config}.npz")
    L, S, q = d["samples"].shape
    rng = np.random.default_rng(hash(("tarp", config, budget, rep)) & 0xFFFFFFFF)
    li = rng.choice(L, L, replace=True)   # bootstrap the truths
    si = rng.choice(S, budget, replace=False)
    samples = d["samples"][li][:, si].transpose(1, 0, 2).astype(np.float64)
    sn, tn = tarp_wrap(samples, d["truths"][li].astype(np.float64))
    t0 = time.time()
    ecp, alpha = get_tarp_coverage(sn, tn, norm=False, seed=rep)
    stat = float(np.max(np.abs(ecp - alpha)))
    return dict(test="tarp", config=config, budget=budget, rep=rep,
                stat=stat, wall=round(time.time() - t0, 2))


def mira_cell(config, budget, rep):
    import torch
    torch.set_num_threads(2)
    from mira_score import mira
    d = load(f"cond_{config}.npz")
    L, S, q = d["samples"].shape
    rng = np.random.default_rng(hash(("mira", config, budget, rep)) & 0xFFFFFFFF)
    li = rng.choice(L, L, replace=True)
    si = rng.choice(S, budget, replace=False)
    truths = torch.as_tensor(d["truths"][li])
    post = torch.as_tensor(d["samples"][li][:, si][None])
    torch.manual_seed(hash(("mira", config, budget, rep)) & 0x7FFFFFFF)
    t0 = time.time()
    mean, _ = mira(truths, post, num_runs=MIRA_RUNS, norm=True,
                   disable_tqdm=True)
    score = float(mean[0])
    N = budget - 1
    refv = (2 * N + 3) / (3 * (N + 1))
    se = float(np.sqrt(1.0 / (18 * L)))
    z = (score - refv) / se
    from scipy import stats
    p = float(2 * stats.norm.sf(abs(z)))
    return dict(test="mira", config=config, budget=budget, rep=rep,
                stat=score, zscore=z, pvalue=p, detected=bool(p < 0.05),
                null_ref=refv, num_runs=MIRA_RUNS,
                wall=round(time.time() - t0, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="pqmass,tarp,mira")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--tag", default="confirmatory")
    ap.add_argument("--out", default="results/a2_power.jsonl")
    args = ap.parse_args()

    tests = args.tests.split(",")
    cells = []
    if "pqmass" in tests:
        cells += [("pqmass", c, b, r) for c in PQM_CONFIGS
                  for b in PQM_BUDGETS for r in range(args.reps)]
    if "tarp" in tests:
        cells += [("tarp", c, b, r) for c in COND_CONFIGS
                  for b in COND_BUDGETS for r in range(args.reps)]
    if "mira" in tests:
        cells += [("mira", c, b, r) for c in COND_CONFIGS
                  for b in COND_BUDGETS for r in range(args.reps)]
    print(f"[a2-power] {len(cells)} cells, {args.workers} workers", flush=True)

    fns = {"pqmass": pqmass_cell, "tarp": tarp_cell, "mira": mira_cell}
    from joblib import Parallel, delayed
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # batched dispatch so rows appear (append-mode) as they finish
    BATCH = 200
    for i in range(0, len(cells), BATCH):
        chunk = cells[i:i + BATCH]
        rows = Parallel(n_jobs=args.workers, verbose=0)(
            delayed(fns[t])(c, b, r) for (t, c, b, r) in chunk)
        with out.open("a") as f:
            for row in rows:
                row["tag"] = args.tag
                f.write(json.dumps(row) + "\n")
        print(f"[a2-power] {min(i + BATCH, len(cells))}/{len(cells)} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # post-pass: empirical p-values for TARP from the oracle null at matched
    # budget (rank statistic; rewrites nothing — emits summary rows)
    if "tarp" in tests:
        import collections
        by = collections.defaultdict(list)
        for line in out.read_text().splitlines():
            row = json.loads(line)
            if row.get("test") == "tarp" and row.get("tag") == args.tag:
                by[(row["config"], row["budget"])].append(row["stat"])
        with out.open("a") as f:
            for (config, budget), stats_ in by.items():
                null = np.asarray(by.get(("oracle", budget), []))
                if len(null) == 0:
                    continue
                for rep, s in enumerate(stats_):
                    p = float((np.sum(null >= s) + 1) / (len(null) + 1))
                    f.write(json.dumps(dict(
                        test="tarp_cal", config=config, budget=budget,
                        rep=rep, stat=s, pvalue=p,
                        detected=bool(p < 0.05), tag=args.tag)) + "\n")
    print(f"[a2-power] done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
