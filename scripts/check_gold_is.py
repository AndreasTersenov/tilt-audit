#!/usr/bin/env python
"""Independent T-L1b verification: importance-sample the lognormal posterior
from the closed-form LINEAR posterior and compare per-mode means with NUTS.

Both densities are exact, so this needs no MCMC at all:
  target   pi_lam(g) prop p(g) exp(-||A kappa(g) - y||^2 / 2b)
  proposal pi_0(g)   prop p(g) exp(-||A g       - y||^2 / 2b)   (= Wiener)
  log w    = -(||A kappa(g) - y||^2 - ||A g - y||^2) / 2b.

At lam = 0.01 the two are close (ESS/M ~ 1), so the IS mean is exact to tiny
MC error. Checks: (1) NUTS mean vs IS mean per mode (z-scores ~ N(0,1) if
NUTS is exact); (2) the offset pattern (IS mean - Wiener mean) vs the NUTS
offset pattern (correlation ~ 1: the "T-L1 failure" is physics, not error).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from tilt_audit import lognormal
from tilt_audit.fields import grid_to_z, make_basis, make_pk, smoothing_operator

M = 200_000
CHUNK = 20_000


def main():
    n, lam = 32, 0.01
    gold = f"results/gold/gold_n{n}_mid_y0_lam{lam:.4g}_s0"
    meta = json.loads(Path(gold + ".json").read_text())
    dat = np.load(gold + ".npz")
    b = meta["b"]

    basis = make_basis(n)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    az = jnp.asarray(smoothing_operator(basis))
    y = jnp.asarray(dat["y"].astype(np.float64))
    mu, Sig = lognormal.linear_limit_params(Pz, az, y, lam, b)
    d = int(Pz.shape[0])

    @jax.jit
    def chunk_logw(g):
        r_lin = az * g - y
        r_non = lognormal.forward_z(g, basis, az, lam) - y
        return -(jnp.sum(r_non**2, axis=-1) - jnp.sum(r_lin**2, axis=-1)) / (2 * b)

    key = jax.random.PRNGKey(2718)
    logw, gsum, gwsum = [], 0.0, 0.0
    g_all_w, g_all = [], []
    for i in range(M // CHUNK):
        k = jax.random.fold_in(key, i)
        g = mu + jnp.sqrt(Sig) * jax.random.normal(k, (CHUNK, d))
        lw = chunk_logw(g)
        logw.append(np.asarray(lw))
        g_all.append(np.asarray(g))
    logw = np.concatenate(logw)
    g_all = np.concatenate(g_all)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    ess = 1.0 / np.sum(w**2)
    mean_is = (w[:, None] * g_all).sum(axis=0)
    var_is = (w[:, None] * (g_all - mean_is) ** 2).sum(axis=0)
    se_is = np.sqrt(var_is / ess)

    z = dat["z"].astype(np.float64).reshape(-1, d)
    ess_nuts = dat["ess"].astype(np.float64)
    mean_nuts = z.mean(axis=0)
    se_nuts = np.sqrt(z.var(axis=0) / np.maximum(ess_nuts, 1.0))

    zsc = (mean_nuts - mean_is) / np.sqrt(se_is**2 + se_nuts**2)
    off_is = mean_is - np.asarray(mu)
    off_nuts = mean_nuts - np.asarray(mu)
    corr = float(np.corrcoef(off_is, off_nuts)[0, 1])
    print(f"[T-L1b:IS] M={M} IS-ESS={ess:.0f} ({ess/M:.2%} of M)")
    print(f"[T-L1b:IS] NUTS-vs-IS mean z: max|z|={np.abs(zsc).max():.2f} "
          f"frac|z|>4={float(np.mean(np.abs(zsc) > 4)):.2e} "
          f"sd(z)={zsc.std():.2f}")
    print(f"[T-L1b:IS] offset pattern corr(IS, NUTS) = {corr:.4f}; "
          f"rms offset = {off_is.std():.3e} (physics, not sampler error, "
          f"if corr~1)")
    ok = (np.abs(zsc).max() < 6.0 and float(np.mean(np.abs(zsc) > 4)) < 1e-3
          and corr > 0.9)
    print(f"[T-L1b:IS] {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
