"""VP-OU forward process, per-mode closed forms.

Forward SDE dx = -x dt + sqrt(2) dW  =>  alpha_t = e^-t, sig2_t = 1 - e^-2t,
stationary N(0, 1). For a Gaussian prior coordinate with variance P:

    marginal   V_t     = alpha_t^2 P + sig2_t
    Tweedie    E[x0|xt] = (alpha_t P / V_t) xt
    posterior  Var[x0|xt] = P sig2_t / V_t
"""
from __future__ import annotations

import jax.numpy as jnp


def alpha_t(t):
    return jnp.exp(-t)


def sig2_t(t):
    return 1.0 - jnp.exp(-2.0 * t)


def marginal_var(t, Pz):
    return alpha_t(t) ** 2 * Pz + sig2_t(t)


def x0hat_coef(t, Pz):
    """E[x0|xt] = coef * xt (per mode)."""
    return alpha_t(t) * Pz / marginal_var(t, Pz)


def var0_t(t, Pz):
    """Var[x0|xt] per mode."""
    return Pz * sig2_t(t) / marginal_var(t, Pz)
