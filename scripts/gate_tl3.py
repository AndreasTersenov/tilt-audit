#!/usr/bin/env python
"""T-L3: two independently-seeded gold runs of one 64^2 config must agree
(run-plan spec): per-mode z-tests between the two chains' means (using both ESS
values) + MMD between the two draw sets below the split-null threshold."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from tilt_audit import metrics_gold


def main():
    lam = "0.3143"
    a = f"results/gold/gold_n64_mid_y0_lam{lam}_s0"
    bpath = f"results/gold/gold_n64_mid_y0_lam{lam}_s1"
    da, db = np.load(a + ".npz"), np.load(bpath + ".npz")
    d = da["z"].shape[-1]
    za = da["z"].astype(np.float64).reshape(-1, d)
    zb = db["z"].astype(np.float64).reshape(-1, d)
    ea = da["ess"].astype(np.float64)
    eb = db["ess"].astype(np.float64)

    se = np.sqrt(za.var(0) / np.maximum(ea, 1) + zb.var(0) / np.maximum(eb, 1))
    z = (za.mean(0) - zb.mean(0)) / se
    fz = float(np.mean(np.abs(z) > 4))
    vr = za.var(0) / zb.var(0)
    fr = float(np.mean(np.abs(vr - 1) > 0.1))

    # MMD split-null: half-vs-half within run A (matched sizes) x 20 splits
    h = metrics_gold.gold_bandwidth(za)
    rng = np.random.default_rng(3)
    nulls = []
    m = za.shape[0] // 2
    for _ in range(20):
        p = rng.permutation(za.shape[0])
        nulls.append(metrics_gold.mmd2(za[p[:m]], za[p[m:2 * m]], h,
                                       max_gold=2048))
    q95 = float(np.quantile(nulls, 0.95))
    cross = metrics_gold.mmd2(za[rng.permutation(za.shape[0])[:m]],
                              zb[rng.permutation(zb.shape[0])[m:2 * m]], h,
                              max_gold=2048)
    ok = fz <= 1e-3 and fr <= 0.02 and cross < 2.0 * q95
    print(f"[T-L3] mean z: frac|z|>4={fz:.2e} max|z|={np.abs(z).max():.2f}; "
          f"var: frac|vr-1|>10%={fr:.4f}; "
          f"MMD cross={cross:.3e} vs split-null q95={q95:.3e} "
          f"=> {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
