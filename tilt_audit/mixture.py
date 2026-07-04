"""Exact 2-component Gaussian mixture in the z-basis (missed-mode substrate).

pi(z) = w * N(z; +dmu, diag(Pz)) + (1 - w) * N(z; -dmu, diag(Pz))

Both components share the GRF covariance; the mean offset lives in the few
lowest-|k| modes only (default: the 4 modes of largest prior power, at
3 sigma_prior each => Mahalanobis separation ||2 dmu||_C = 12, essentially
disjoint components — the well-separated archetype score-based tests are
theoretically blind to). Everything is closed-form: log-density, score via
responsibilities, moments, and exact sampling.

Pinned: this substrate has NO observation/forward model — the mixture IS the
target. Failure modes under test are synthetic sampler pathologies:
single-component coverage (missed mode) and swapped weights.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .fields import Basis, grid_to_z


def make_offset(Pz, basis: Basis, n_modes: int = 4,
                scale: float = 3.0) -> np.ndarray:
    """dmu: `scale` prior-sigmas in each of the n_modes largest-power modes."""
    kz = grid_to_z(basis.kmag, basis)
    Pnp = np.asarray(Pz)
    idx = np.argsort(kz)[:n_modes]  # lowest |k| = largest prior power
    dmu = np.zeros_like(Pnp)
    dmu[idx] = scale * np.sqrt(Pnp[idx])
    return dmu


def _comp_logpdf(z, mu, Pz):
    return -0.5 * jnp.sum((z - mu) ** 2 / Pz + jnp.log(2.0 * jnp.pi * Pz),
                          axis=-1)


def log_prob(z, dmu, Pz, w):
    la = _comp_logpdf(z, +dmu, Pz) + jnp.log(w)
    lb = _comp_logpdf(z, -dmu, Pz) + jnp.log1p(-w)
    return jnp.logaddexp(la, lb)


def score(z, dmu, Pz, w):
    """d log pi / dz via responsibilities. z: (..., d)."""
    la = _comp_logpdf(z, +dmu, Pz) + jnp.log(w)
    lb = _comp_logpdf(z, -dmu, Pz) + jnp.log1p(-w)
    ra = jnp.exp(la - jnp.logaddexp(la, lb))[..., None]
    sa = -(z - dmu) / Pz
    sb = -(z + dmu) / Pz
    return ra * sa + (1.0 - ra) * sb


def sample(key, dmu, Pz, w, N: int, mode: str = "both"):
    """Exact draws. mode: 'both' (the oracle), 'plus'/'minus' (single-
    component pathologies), or 'swapped' (weights 1-w/w)."""
    kc, kg = jax.random.split(key)
    g = jnp.sqrt(Pz) * jax.random.normal(kg, (N, Pz.shape[0]))
    if mode == "plus":
        signs = jnp.ones((N, 1))
    elif mode == "minus":
        signs = -jnp.ones((N, 1))
    else:
        p_plus = w if mode == "both" else 1.0 - w
        signs = jnp.where(jax.random.uniform(kc, (N, 1)) < p_plus, 1.0, -1.0)
    return g + signs * dmu


def moments(dmu, Pz, w):
    """Exact mean and per-mode variance of the mixture."""
    mean = (2.0 * w - 1.0) * dmu
    var = Pz + 4.0 * w * (1.0 - w) * dmu**2
    return mean, var
