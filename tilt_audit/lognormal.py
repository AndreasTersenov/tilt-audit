"""Nonlinear-forward-model substrate: lognormal observation of the GRF prior.

The prior is UNCHANGED (g ~ N(0, C), diagonal Pz in the z-basis) so the whole
diffusion/sampler harness keeps its closed forms. Nonlinearity enters ONLY
through the observation:

    y = A . kappa(g) + n,   kappa(g) = (exp(lam*g - lam^2 sig_g^2 / 2) - 1)/lam,

with kappa applied pixelwise (sig_g^2 = 1 by the make_pk normalization, so
kappa is mean-zero exactly and kappa = g + O(lam)). The 1/lam normalization is
PINNED: it makes lam -> 0 recover y = A g + n — the *identical* linear model
of the Gaussian bench (same y scale, same SNR, same b semantics) — and keeps
the kappa field near unit pixel variance (sqrt(e^{lam^2}-1)/lam ~ 1.02 at the
default lam) at any nonlinearity. Skewness is scale-invariant, so the
skewness(kappa)=1 calibration of lam is unaffected. [Amended after the 16^2
smoke: the UNnormalized map at lam=0.01 + fixed noise drives the posterior to
g ~ 50 sigma where the nonlinearity is large — T-L1 failed by construction,
not by NUTS error; see NIGHT_LOG_2026-07-05.md 23:4x.]
A is the standard Fourier-diagonal smoothing operator; white noise of std s is
identical in pixel and z basis (the transform is orthonormal), so y lives in
the z-basis like every other observation in this repo.

Pinned conventions (do not change silently):
- lam -> 0 recovers y = A g + n exactly: gate T-L1 compares NUTS at lam=0.01
  against tilt.posterior_params(Pz, az, y, b) directly.
- The tilt strength b is calibrated on the LINEARIZED operator az (the lam->0
  map) via tilt.calibrate_b — the standard calibration of the whole bench,
  a fixed convention at finite lam (reported in every output row).
- Default lam solves skewness(kappa) = 1 for the pixel marginal:
  (w+2)*sqrt(w-1) = 1 with w = e^{lam^2}  =>  lam ~ 0.31.
- NUTS parameterization is the whitened u-space, g_z = sqrt(Pz) * u: the
  prior becomes N(0, I) and the posterior is near-diagonal, which is what
  diagonal mass-matrix adaptation wants.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import tilt
from .fields import Basis, pack, unpack


def default_lambda() -> float:
    """lam with pixel-marginal skewness(kappa) = 1 (lognormal closed form)."""
    # skew = (w + 2) sqrt(w - 1), w = e^{lam^2}; solve skew = 1 by bisection.
    lo, hi = 1.0 + 1e-9, 2.0
    for _ in range(200):
        w = 0.5 * (lo + hi)
        if (w + 2.0) * np.sqrt(w - 1.0) > 1.0:
            hi = w
        else:
            lo = w
    return float(np.sqrt(np.log(0.5 * (lo + hi))))


def kappa(g_pix: jnp.ndarray, lam: float) -> jnp.ndarray:
    """Pixelwise shifted-lognormal map, mean-zero, kappa = g + O(lam)."""
    return (jnp.exp(lam * g_pix - 0.5 * lam**2) - 1.0) / lam


def forward_z(g_z: jnp.ndarray, basis: Basis, az: jnp.ndarray,
              lam: float) -> jnp.ndarray:
    """g in z-basis -> A.kappa(g) in z-basis (the noiseless observation)."""
    return az * pack(kappa(unpack(g_z, basis), lam), basis)


def make_lognormal_observation(key, Pz, az, basis, lam, s):
    """Synthetic y = A.kappa(g_truth) + s*xi in the z-basis."""
    k1, k2 = jax.random.split(key)
    g_truth = jnp.sqrt(Pz) * jax.random.normal(k1, Pz.shape)
    y = forward_z(g_truth, basis, az, lam) + s * jax.random.normal(k2, Pz.shape)
    return y, g_truth


def calibrate_b_linearized(Pz, az, y, lam, target_shift):
    """Tilt strength via the linearized (lam->0) operator az — the standard
    bench calibration (pinned convention; lam kept in the signature so every
    call site names the substrate explicitly)."""
    del lam
    return tilt.calibrate_b(Pz, az, y, target_shift)


def log_posterior_g(g_z, Pz, az, basis, lam, y, b):
    """log pi(g_z) up to a constant: GRF prior + Gaussian likelihood of the
    nonlinear map with tilt exponent b (the r/beta convention of tilt.py)."""
    lp = -0.5 * jnp.sum(g_z**2 / Pz)
    resid = forward_z(g_z, basis, az, lam) - y
    return lp - jnp.sum(resid**2) / (2.0 * b)


def score_g(g_z, Pz, az, basis, lam, y, b):
    """d log pi / d g_z — the TRUE target score (KSD control mode; guidance)."""
    return jax.grad(log_posterior_g)(g_z, Pz, az, basis, lam, y, b)


def make_potential_u(Pz, az, basis, lam, y, b):
    """Whitened-space potential U(u) = -log pi(sqrt(Pz) u) for NUTS.

    The Jacobian of the fixed linear map is constant — dropped, as usual.
    """
    sqrtP = jnp.sqrt(Pz)

    def potential(u):
        return -log_posterior_g(sqrtP * u, Pz, az, basis, lam, y, b)

    return potential


def linear_limit_params(Pz, az, y, lam, b):
    """Closed-form posterior of the linearized model (T-L1 reference)."""
    del lam
    return tilt.posterior_params(Pz, az, y, b)
