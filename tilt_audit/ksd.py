"""Score-based kernelized Stein discrepancy, following arXiv:2602.04189 v2.

No reference code exists (hunt logged 2026-07-04); this implements the paper's
recipe exactly, cross-checked against an autodiff Stein kernel in gate T-K1:

- V-statistic core: H_ij = u_p(x_i, x_j) with the Langevin Stein kernel
      u_p(x,x') = s(x)^T k s(x') + s(x)^T grad_{x'} k + s(x')^T grad_x k
                  + tr(grad_x grad_{x'} k).
- Reported metric (their Algorithm 1, step 9), dimension-normalized:
      score_ksd = (1/N) sqrt( sum_ij H_ij / d ) = sqrt(mean(H) / d).
- Primary kernel: IMQ k = (c^2 + r^2)^beta, beta = -1/2, with the PAPER's
  scale rule c = 1/(median(singular values of A) + 1) — from the forward
  operator, not the samples.
- Secondary (pre-registered in the plan): IMQ c=1 and RBF with the median
  pairwise-distance heuristic.
- The paper has NO calibration (within-task ranking only). Our empirical
  60-null alpha=0.05 rank calibration lives in the battery scripts, not here.

All kernels are radial phi(r^2); the closed forms used below:
  H_ij = phi*<s_i,s_j> + 2 phi' * [<s_j - s_i, x_i - x_j>] - 2 d phi'
         - 4 r^2 phi''.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np


def c_paper(az) -> float:
    """The paper's IMQ scale: c = 1/(median(singvals(A)) + 1)."""
    return float(1.0 / (np.median(np.asarray(az)) + 1.0))


def median_heuristic(X) -> float:
    """Median pairwise distance (subsampled to <=1024 rows for cost)."""
    X = np.asarray(X)
    if X.shape[0] > 1024:
        X = X[np.random.default_rng(0).choice(X.shape[0], 1024, replace=False)]
    sq = np.sum(X**2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0)
    off = d2[np.triu_indices_from(d2, k=1)]
    return float(np.sqrt(np.median(off)))


def _phi_imq(r2, c, beta):
    base = c**2 + r2
    phi = base**beta
    phi1 = beta * base ** (beta - 1.0)
    phi2 = beta * (beta - 1.0) * base ** (beta - 2.0)
    return phi, phi1, phi2


def _phi_rbf(r2, h):
    phi = jnp.exp(-r2 / (2.0 * h**2))
    phi1 = -phi / (2.0 * h**2)
    phi2 = phi / (4.0 * h**4)
    return phi, phi1, phi2


@functools.partial(jax.jit, static_argnames=("kernel",))
def stein_H(X, S, kernel: str, p1: float, p2: float):
    """The N x N Stein-kernel matrix. kernel: 'imq' (p1=c, p2=beta) or
    'rbf' (p1=h, p2 unused)."""
    sq = jnp.sum(X**2, axis=1)
    r2 = jnp.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0)
    phi, phi1, phi2 = (_phi_imq(r2, p1, p2) if kernel == "imq"
                       else _phi_rbf(r2, p1))
    SS = S @ S.T
    SX = S @ X.T                       # <s_i, x_j>
    sx = jnp.sum(S * X, axis=1)        # <s_i, x_i>
    # <s_j - s_i, x_i - x_j> as an (i, j) matrix:
    cross = (SX.T - sx[None, :]) - (sx[:, None] - SX)
    d = X.shape[1]
    return phi * SS + 2.0 * phi1 * cross - 2.0 * d * phi1 - 4.0 * r2 * phi2


def ksd_stats(X, S, kernel: str, p1: float, p2: float = -0.5) -> dict:
    """Paper statistic + U-statistic companion from one H matrix."""
    H = stein_H(jnp.asarray(X), jnp.asarray(S), kernel, p1, p2)
    N, d = X.shape
    v_mean = float(jnp.mean(H))
    u_mean = float((jnp.sum(H) - jnp.trace(H)) / (N * (N - 1)))
    return dict(
        score_ksd=float(np.sqrt(max(v_mean, 0.0) / d)),
        v_mean=v_mean, u_mean=u_mean, N=int(N), d=int(d),
        kernel=kernel, p1=float(p1), p2=float(p2))


def stein_H_autodiff(X, S, kernel: str, p1: float, p2: float = -0.5):
    """Slow autodiff reference for the T-K1 cross-check (small N, d only)."""
    def k_fn(x, y):
        r2 = jnp.sum((x - y) ** 2)
        if kernel == "imq":
            return (p1**2 + r2) ** p2
        return jnp.exp(-r2 / (2.0 * p1**2))

    gx = jax.grad(k_fn, argnums=0)
    gy = jax.grad(k_fn, argnums=1)

    def u_pair(xi, si, xj, sj):
        tr = jnp.trace(jax.jacfwd(gx, argnums=1)(xi, xj))
        return (k_fn(xi, xj) * si @ sj + si @ gy(xi, xj)
                + sj @ gx(xi, xj) + tr)

    return jax.vmap(lambda xi, si: jax.vmap(
        lambda xj, sj: u_pair(xi, si, xj, sj))(X, S))(X, S)


# --- target scores for tonight's substrates ---

def score_gaussian(z, mu, Sig):
    """Tilted-GRF posterior score, exact per mode. z: (N, d)."""
    return -(z - mu[None, :]) / Sig[None, :]
