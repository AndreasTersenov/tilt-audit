#!/usr/bin/env python
"""Gate T-N3: null calibration of the three community diagnostics.

A diagnostic that fails its null does not enter the A2 power study.

1. PQMass: 100 truly independent same-vs-same pairs (fresh closed-form oracle
   draws per rep, 64^2, budget 256/side, num_refs 32) -> p-values ~ U(0,1),
   KS-test p > 0.01. Fresh draws per rep, not archive subsamples: overlapping
   subsamples of one bank would correlate the p-values and invalidate the KS.
2. TARP: cond_oracle.npz (L=128 y-draws, S=256 exact-posterior samples at
   beta=1) -> max_alpha |ecp - alpha| < 0.10 (binomial se at L=128 is ~0.044
   mid-curve; 0.10 ~ 2.3 sigma against the sup-statistic).
3. MIRA: cond_oracle.npz, num_runs=128, norm on -> score within the exact
   finite-N null (2N+3)/(3(N+1)) (N = S-1) +- 2*sqrt(1/(18L)).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats

ARCH = Path(__file__).resolve().parent.parent / "results" / "archives"


def pqmass_null():
    import jax
    from pqm import pqm_pvalue
    from tilt_audit import tilt
    from tilt_audit.fields import grid_to_z, make_basis, make_pk, smoothing_operator
    import jax.numpy as jnp

    basis = make_basis(64)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    az = jnp.asarray(smoothing_operator(basis))
    y, _ = tilt.make_observation(jax.random.PRNGKey(999), Pz, az, 0.5)
    b = float(tilt.calibrate_b(Pz, az, y, target_shift=1.0))
    mu, Sig = tilt.posterior_params(Pz, az, y, b)
    mu = np.asarray(mu); sd = np.sqrt(np.asarray(Sig))

    rng = np.random.default_rng(20260704)
    pvals = []
    for rep in range(100):
        x = mu + sd * rng.standard_normal((256, mu.shape[0]))
        z = mu + sd * rng.standard_normal((256, mu.shape[0]))
        np.random.seed(rep)  # pqm uses global numpy randomness internally
        pvals.append(float(pqm_pvalue(x, z, num_refs=32)))
    ks = stats.kstest(pvals, "uniform")
    ok = ks.pvalue > 0.01
    print(f"T-N3 PQMass null: KS p={ks.pvalue:.4f} over 100 fresh same-vs-same"
          f" reps (need >0.01) -> {'PASS' if ok else 'FAIL'}")
    return ok


def tarp_null():
    """tarp is wrapped (see run_diagnostics.tarp_wrap): its norm=True
    truth-based min-max is asymmetric and d-extensively miscalibrates the
    null at q=4096 (max dev 0.20 on the exact posterior — itself an A2
    finding, logged 2026-07-04). Median over 5 reference seeds < 0.10."""
    from tarp import get_tarp_coverage
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_diagnostics import tarp_wrap
    d = np.load(ARCH / "cond_oracle.npz")
    samples = d["samples"].transpose(1, 0, 2).astype(np.float64)
    sn, tn = tarp_wrap(samples, d["truths"].astype(np.float64))
    devs = []
    for seed in range(5):
        ecp, alpha = get_tarp_coverage(sn, tn, norm=False, seed=seed)
        devs.append(float(np.max(np.abs(ecp - alpha))))
    dev = float(np.median(devs))
    ok = dev < 0.10
    print(f"T-N3 TARP null (wrapped): median max|ecp-alpha|={dev:.4f} over 5 "
          f"seeds {[round(v, 3) for v in devs]} (need <0.10) -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def mira_null():
    import torch
    from mira_score import mira
    d = np.load(ARCH / "cond_oracle.npz")
    truths = torch.as_tensor(d["truths"])
    post = torch.as_tensor(d["samples"])[None]  # (1, L, S, q)
    torch.manual_seed(20260704)
    mean, _ = mira(truths, post, num_runs=128, norm=True, disable_tqdm=True)
    score = float(mean[0])
    L, S = d["samples"].shape[0], d["samples"].shape[1]
    N = S - 1
    ref = (2 * N + 3) / (3 * (N + 1))
    band = 2 * np.sqrt(1.0 / (18 * L))
    ok = abs(score - ref) < band
    print(f"T-N3 MIRA null: score={score:.4f} vs analytic ref {ref:.4f} "
          f"+- {band:.4f} (L={L}, S={S}) -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    results = {}
    for name, fn in [("pqmass", pqmass_null), ("tarp", tarp_null),
                     ("mira", mira_null)]:
        try:
            results[name] = bool(fn())
        except Exception as e:  # a crashed diagnostic fails its null
            print(f"T-N3 {name} CRASHED: {e!r} -> FAIL")
            results[name] = False
    print("T-N3 summary:", json.dumps(results))
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
