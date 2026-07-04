"""Sample-vs-gold metrics for the non-Gaussian substrate (plan §3).

The posterior is non-Gaussian, so the closed-form Gaussian metrics do NOT
apply; everything here is referenced to MCMC gold draws:

  mmd2        unbiased RBF MMD^2, bandwidth = median pairwise distance of the
              GOLD draws (pinned; computed once per gold set)
  swd2        sliced W2^2 over 100 FIXED random directions (seed 0), matched
              sample counts via sorted-quantile pairing
  mode_z      per-mode mean z-scores vs gold (se from gold's per-mode ESS and
              the sample count), reported as max and frac(|z| > 3)
  var_ratio   per-mode variance ratio vs gold, median |ratio - 1|
  bp_cov68    per-band coverage: fraction of sampler draws inside the gold
              16-84% band of each band-power functional B_j = mean_k(z_k^2)

The floor row is a disjoint gold subsample of matched size run through the
same functions (the MC floor every sampler is compared against).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def gold_bandwidth(gold, max_rows=2048):
    g = np.asarray(gold)
    if g.shape[0] > max_rows:
        g = g[np.random.default_rng(1).choice(g.shape[0], max_rows,
                                              replace=False)]
    sq = np.sum(g**2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (g @ g.T), 0.0)
    return float(np.sqrt(np.median(d2[np.triu_indices_from(d2, k=1)])))


@jax.jit
def _mmd2_terms(X, G, h):
    def k(a, bm):
        sq_a = jnp.sum(a**2, 1)
        sq_b = jnp.sum(bm**2, 1)
        d2 = jnp.maximum(sq_a[:, None] + sq_b[None, :] - 2 * (a @ bm.T), 0.0)
        return jnp.exp(-d2 / (2.0 * h**2))
    Kxx = k(X, X)
    Kgg = k(G, G)
    Kxg = k(X, G)
    n, m = X.shape[0], G.shape[0]
    xx = (Kxx.sum() - jnp.trace(Kxx)) / (n * (n - 1))
    gg = (Kgg.sum() - jnp.trace(Kgg)) / (m * (m - 1))
    return xx + gg - 2.0 * Kxg.mean()


def mmd2(X, gold, h, max_gold=2048):
    g = np.asarray(gold)
    if g.shape[0] > max_gold:
        g = g[np.random.default_rng(2).choice(g.shape[0], max_gold,
                                              replace=False)]
    return float(_mmd2_terms(jnp.asarray(X), jnp.asarray(g), h))


def sliced_w2(X, gold, n_dirs=100, seed=0):
    X = np.asarray(X)
    g = np.asarray(gold)
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_dirs, X.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    px = np.sort(X @ dirs.T, axis=0)              # (N, n_dirs)
    qlev = (np.arange(X.shape[0]) + 0.5) / X.shape[0]
    pg = np.quantile(g @ dirs.T, qlev, axis=0)    # matched quantiles
    return float(np.mean((px - pg) ** 2))


def mode_stats(X, gold, gold_ess):
    X = np.asarray(X)
    g = np.asarray(gold)
    gm, gv = g.mean(0), g.var(0)
    se = np.sqrt(gv / np.maximum(np.asarray(gold_ess), 1.0)
                 + X.var(0) / X.shape[0])
    z = (X.mean(0) - gm) / se
    vr = X.var(0) / np.maximum(gv, 1e-30)
    return dict(mode_z_max=float(np.abs(z).max()),
                mode_z_frac3=float(np.mean(np.abs(z) > 3)),
                var_ratio_med_dev=float(np.median(np.abs(vr - 1.0))),
                var_ratio_med=float(np.median(vr)))


def bp_coverage(X, gold, masks):
    X = np.asarray(X)
    g = np.asarray(gold)
    out = {}
    for j, m in enumerate(masks):
        bx = (X[:, m] ** 2).mean(axis=1)
        bg = (g[:, m] ** 2).mean(axis=1)
        lo, hi = np.quantile(bg, [0.16, 0.84])
        out[f"bp{j}_cov68"] = float(np.mean((bx >= lo) & (bx <= hi)))
    return out


def evaluate_vs_gold(X, gold, gold_ess, masks, h=None):
    """All metrics in one call; h precomputed via gold_bandwidth for reuse."""
    if h is None:
        h = gold_bandwidth(gold)
    out = dict(mmd2=mmd2(X, gold, h), swd2=sliced_w2(X, gold), mmd_h=h)
    out.update(mode_stats(X, gold, gold_ess))
    out.update(bp_coverage(X, gold, masks))
    return out
