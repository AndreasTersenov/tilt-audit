"""Reward tilt r(x) = -||Ax - y||^2 / (2 s^2) and its exact Gaussian target.

With A diagonal (a_k) in the z-basis and tilt exponent r/beta, the tilted
target sigma ~ p * exp(r/beta) is the Wiener posterior with effective noise
b = beta * s^2, per coordinate:

    Sig*_k = (1/P_k + a_k^2 / b)^-1,   mu*_k = Sig*_k a_k y_k / b.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def posterior_params(Pz, az, y, b):
    Sig = 1.0 / (1.0 / Pz + az**2 / b)
    mu = Sig * az * y / b
    return mu, Sig


def log_reward_over_beta(z0, az, y, b):
    """r(x0)/beta summed over coords; z0 shape (..., d)."""
    return -jnp.sum((az * z0 - y) ** 2, axis=-1) / (2.0 * b)


def log_z_analytic(Pz, az, y, b):
    """log E_p[exp(r/beta)] in closed form (per-mode Gaussian integral)."""
    denom = b + az**2 * Pz
    return jnp.sum(0.5 * jnp.log(b / denom) - y**2 / (2.0 * denom))


def rms_shift(Pz, az, y, b):
    """Tilt strength: RMS of the posterior mean shift in prior-sigma units."""
    mu, _ = posterior_params(Pz, az, y, b)
    return jnp.sqrt(jnp.mean(mu**2 / Pz))


def calibrate_b(Pz, az, y, target_shift, lo=1e-8, hi=1e8, iters=200):
    """Bisection for b such that rms_shift == target (shift is decreasing in b)."""
    def body(_, bounds):
        blo, bhi = bounds
        mid = jnp.sqrt(blo * bhi)  # geometric bisection: b spans many decades
        shift = rms_shift(Pz, az, y, mid)
        too_strong = shift > target_shift  # tilt too strong -> raise b
        return jnp.where(too_strong, mid, blo), jnp.where(too_strong, bhi, mid)
    blo, bhi = jax.lax.fori_loop(0, iters, body, (jnp.float64(lo), jnp.float64(hi)))
    return jnp.sqrt(blo * bhi)


def make_observation(key, Pz, az, s):
    """Synthetic y from a held-out prior draw + white noise of std s (z-basis)."""
    k1, k2 = jax.random.split(key)
    z_truth = jnp.sqrt(Pz) * jax.random.normal(k1, Pz.shape)
    y = az * z_truth + s * jax.random.normal(k2, Pz.shape)
    return y, z_truth
