#!/usr/bin/env python
"""Relative-certificate validation (P-20260704g; zero new compute).

The mean-side identity E[log w] = log Z - KL_path makes the DIFFERENCE of
mean log-weights between two samplers on the SAME reference an exact,
degeneracy-immune estimator of -ΔKL_path. Three questions:

 (a) estimator precision: sampled Δmean(logw) vs exact Δ(kl_path_exact),
     in units of its own standard error (exact grid, N=256, 8 seeds);
 (b) does path ordering agree with ENDPOINT ordering (the scientifically
     relevant one) across sampler pairs?
 (c) learned nets: does the relative certificate's ranking (dps vs unguided,
     same net) match the empirical endpoint ranking?
"""
import collections
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(name, keys):
    rows = [json.loads(l) for l in open(ROOT / "results" / name)]
    seen = {}
    for r in rows:
        seen[tuple(r.get(k) for k in keys)] = r
    return list(seen.values())


# ---------------------------------------------------------------- exact grid
rows = load("cert_killtest.jsonl",
            ("dim", "y_seed", "shift", "mode", "N", "seed", "eps"))
core = [r for r in rows if r["eps"] == 0 and r["y_seed"] == 999
        and r["N"] == 256]
# mean(logw) is not stored per row; reconstruct: kl_path_hat = log_z_est - mean
# => mean(logw) = log_z_est - kl_path_hat
cell = collections.defaultdict(list)
law = {}
for r in core:
    cell[(r["dim"], r["shift"], r["mode"])].append(
        r["log_z_est"] - r["kl_path_hat"])
    law[(r["dim"], r["shift"], r["mode"])] = (r["kl_path_exact"],
                                              r["kl_end_exact"])

print("=" * 96)
print("(a) exact grid: sampled Δmean(logw) vs exact ΔKL_path  [z = deviation/s.e.]")
zs, order_pairs = [], []
for dim in sorted({k[0] for k in cell}):
    for shift in (0.5, 1.0, 2.0, 4.0):
        modes = [m for m in ("exact_guidance", "dps", "unguided")
                 if (dim, shift, m) in cell]
        for m1, m2 in itertools.combinations(modes, 2):
            a = np.asarray(cell[(dim, shift, m1)])
            bvals = np.asarray(cell[(dim, shift, m2)])
            d_hat = a.mean() - bvals.mean()
            se = np.sqrt(a.var(ddof=1) / len(a) + bvals.var(ddof=1) / len(bvals))
            d_exact = -(law[(dim, shift, m1)][0] - law[(dim, shift, m2)][0])
            zs.append((d_hat - d_exact) / se)
            # (b) ordering agreement: path vs endpoint
            path_order = law[(dim, shift, m1)][0] < law[(dim, shift, m2)][0]
            end_order = law[(dim, shift, m1)][1] < law[(dim, shift, m2)][1]
            order_pairs.append(path_order == end_order)
zs = np.asarray(zs)
frac2 = float(np.mean(np.abs(zs) <= 2))
print(f"  {len(zs)} sampler pairs: |z|<=2 for {frac2:.1%} "
      f"(median |z| = {np.median(np.abs(zs)):.2f}, max = {np.abs(zs).max():.1f})")
print(f"  P-g(a) [within 2 s.e. everywhere]: "
      f"{'HIT' if frac2 == 1.0 else f'{frac2:.1%} -> ' + ('HIT-ish' if frac2 >= 0.95 else 'MISS')}")
agree = float(np.mean(order_pairs))
print(f"(b) path-order == endpoint-order: {agree:.1%} of {len(order_pairs)} pairs "
      f"-> {'HIT' if agree >= 0.9 else 'MISS'}")

# ---------------------------------------------------------------- learned nets
print("=" * 96)
print("(c) learned nets: relative certificate (dps - unguided, same net) vs "
      "empirical endpoint ranking")
lrows = load("cert_learned.jsonl", ("score", "shift", "mode", "seed"))
lcell = collections.defaultdict(list)
for r in lrows:
    lcell[(r["score"], r["shift"], r["mode"])].append(r)
ok_all, n_checks = True, 0
for score in sorted({r["score"] for r in lrows}):
    line = f"  {score:>18}: "
    for shift in (0.5, 1.0, 2.0, 4.0):
        d_ = lcell.get((score, shift, "dps"))
        u_ = lcell.get((score, shift, "unguided"))
        if not d_ or not u_:
            continue
        rel = (np.mean([r["log_z_est"] - r["kl_path_hat"] for r in d_])
               - np.mean([r["log_z_est"] - r["kl_path_hat"] for r in u_]))
        # empirical endpoint: raw W2 vs the TRUE target
        w2_d = np.median([r["raw_w2"] for r in d_])
        w2_u = np.median([r["raw_w2"] for r in u_])
        cert_says_dps = rel > 0        # higher mean logw = closer (path)
        truth_says_dps = w2_d < w2_u   # endpoint metric
        ok = cert_says_dps == truth_says_dps
        ok_all = ok_all and ok
        n_checks += 1
        line += f"{shift}σ:{'OK' if ok else 'FLIP'}(Δ={rel:+.3g}) "
    print(line)
print(f"  P-g(c): {'HIT' if ok_all else 'MISS'} ({n_checks} checks)")
