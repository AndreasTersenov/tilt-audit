#!/usr/bin/env python
"""T-K1: admission gate for the score-KSD implementation (plan §4.2).

1. Algebra: closed-form Stein H matches the autodiff Stein kernel to <1e-8
   (both kernels, random small case).
2. Null behavior: oracle samples vs the TRUE tilted-target score at budgets
   {256, 1024}, 100 reps each, three kernels (imq-paper, imq-c1, rbf-median):
   the null must be tight (a usable 95th percentile: CV of the statistic
   finite and the empirical quantile stable under half-splits) and the
   statistic must decrease with N (the paper's stated behavior).
3. Sanity power (not part of the frozen grid): dps@1sigma archive should
   exceed the null 95th percentile at budget 1024.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from tilt_audit import ksd, samplers, tilt
from tilt_audit.fields import grid_to_z, make_basis, make_pk, smoothing_operator

TF = 9.0
S_OBS = 0.5
Y_KEY = 999


def main():
    ok = True

    # -- 1. autodiff cross-check (small: N=8, d=6) --
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    X = jax.random.normal(k1, (8, 6))
    S = jax.random.normal(k2, (8, 6))
    for kernel, p1, p2 in [("imq", 0.7, -0.5), ("imq", 1.0, -0.5),
                           ("rbf", 1.3, 0.0)]:
        Hc = ksd.stein_H(X, S, kernel, p1, p2)
        Ha = ksd.stein_H_autodiff(X, S, kernel, p1, p2)
        err = float(jnp.max(jnp.abs(Hc - Ha)))
        tag = "PASS" if err < 1e-8 else "FAIL"
        ok &= err < 1e-8
        print(f"[T-K1.1] {kernel}(p1={p1}) closed-vs-autodiff "
              f"max|dH|={err:.2e} {tag}", flush=True)

    # -- 2. null behavior on the real 64^2 target --
    n = 64
    basis = make_basis(n)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    az = jnp.asarray(smoothing_operator(basis))
    y, _ = tilt.make_observation(jax.random.PRNGKey(Y_KEY), Pz, az, S_OBS)
    b = float(tilt.calibrate_b(Pz, az, y, target_shift=1.0))
    mu, Sig = tilt.posterior_params(Pz, az, y, b)
    munp, Signp = np.asarray(mu), np.asarray(Sig)

    def oracle_draw(seed, N):
        k = jax.random.fold_in(jax.random.PRNGKey(seed), 424242)
        return np.asarray(mu + jnp.sqrt(Sig)
                          * jax.random.normal(k, (N, Pz.shape[0])))

    kernels = {
        "imq_paper": ("imq", ksd.c_paper(az), -0.5),
        "imq_c1": ("imq", 1.0, -0.5),
    }
    print(f"[T-K1.2] c_paper = {ksd.c_paper(az):.4f}", flush=True)

    results = {}
    for budget in (256, 1024):
        # rbf: median heuristic pinned on ONE oracle draw per budget
        h = ksd.median_heuristic(oracle_draw(9000, budget))
        kernels_b = dict(kernels, rbf_med=("rbf", h, 0.0))
        for kname, (kern, p1, p2) in kernels_b.items():
            t0 = time.time()
            vals = []
            for rep in range(100):
                Xs = oracle_draw(10_000 + rep, budget)
                Ss = ksd.score_gaussian(Xs, munp, Signp)
                vals.append(ksd.ksd_stats(Xs, Ss, kern, p1, p2)["score_ksd"])
            vals = np.array(vals)
            q95a = float(np.quantile(vals[:50], 0.95))
            q95b = float(np.quantile(vals[50:], 0.95))
            spread = abs(q95a - q95b) / max(np.std(vals), 1e-30)
            tight = np.std(vals) / vals.mean() < 0.5 and spread < 2.0
            ok &= tight
            results[f"{kname}_N{budget}"] = dict(
                mean=float(vals.mean()), sd=float(vals.std()),
                q95=float(np.quantile(vals, 0.95)), h_or_c=p1)
            print(f"[T-K1.2] {kname} N={budget}: null mean={vals.mean():.5f} "
                  f"sd={vals.std():.2e} q95={np.quantile(vals, 0.95):.5f} "
                  f"halfsplit|dq95|/sd={spread:.2f} "
                  f"{'PASS' if tight else 'FAIL'} "
                  f"({time.time()-t0:.0f}s/100reps)", flush=True)

    for kname in list(kernels) + ["rbf_med"]:
        dec = results[f"{kname}_N1024"]["mean"] < results[f"{kname}_N256"]["mean"]
        ok &= dec
        print(f"[T-K1.2] {kname}: mean decreases 256->1024: "
              f"{'PASS' if dec else 'FAIL'}", flush=True)

    # -- 3. sanity power: dps archive at budget 1024 --
    arch = np.load("results/archives/dps.npz")
    zdps = np.asarray(arch["z"][:1024], dtype=np.float64)
    Sdps = ksd.score_gaussian(zdps, munp, Signp)
    for kname, (kern, p1, p2) in kernels.items():
        stat = ksd.ksd_stats(zdps, Sdps, kern, p1, p2)["score_ksd"]
        q95 = results[f"{kname}_N1024"]["q95"]
        det = stat > q95
        print(f"[T-K1.3] dps@1sigma {kname} N=1024: stat={stat:.5f} "
              f"null q95={q95:.5f} -> {'DETECTED' if det else 'MISSED'}",
              flush=True)

    Path("results").mkdir(exist_ok=True)
    with open("results/tk1_nulls.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"gate_ksd: {'ALL GREEN' if ok else 'FAILURES PRESENT'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
