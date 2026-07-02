"""Learned-score sampler pathway (T2): same five schemes on a trained net.

State is carried in the z-basis; each step unpacks to pixels for the net.
Base kernel = DDPM ancestral step with the Tweedie x0hat plug-in (the
practitioner-standard choice, shared by all learned arms):

    q(x_t' | x_t, x0hat) = N(A x_t + B x0hat, kv),
    A = e^-dt sig2_t' / sig2_t, B = alpha_t' (1 - e^-2dt) / sig2_t,
    kv = sig2_t' (1 - e^-2dt) / sig2_t.

Arms:
  dps          ancestral + true DPS guidance (VJP through the denoiser)
  sap          ancestral + resample every step on e^{r(x0hat)/beta}
  twisted      proper twisted SMC: analytic-shape per-mode twist
               psi_t(z) = N(y; a c0(t) z, b + a^2 var0(t)) used for BOTH the
               conjugate proposal and the weights, so every integral is
               closed-form and the FK target is the learned chain tilted by
               e^{r/beta}. Nonzero increments now measure score error.
  terminal_is  ancestral unguided + terminal e^{r/beta} reweight

The same pathway with score_fn='analytic' (exact prior score, same ancestral
kernel) is the pathway control: it isolates kernel-choice effects from score
error in the T2 decomposition.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import diffusion, tilt
from .fields import pack, unpack
from .samplers import _ess, _log_psi, systematic_resample


def make_score_x0hat(model, params, basis, Pz_analytic=None):
    """Returns x0hat_fn(z, t) -> z0hat (N, d), Tweedie from the net or analytic."""
    if model == "analytic":
        def x0hat_fn(z, t):
            return diffusion.x0hat_coef(t, Pz_analytic) * z
        return jax.jit(x0hat_fn)

    def x0hat_fn(z, t):
        x = unpack(z, basis)
        eps = model.apply(params, x, jnp.full(x.shape[0], t))
        sig = jnp.sqrt(diffusion.sig2_t(t))
        alpha = diffusion.alpha_t(t)
        x0h = (x - sig * eps) / alpha
        return pack(x0h, basis)
    return jax.jit(x0hat_fn)


def _ancestral_coefs(t, t_next):
    dt = t - t_next
    s2t = diffusion.sig2_t(t)
    s2n = diffusion.sig2_t(t_next)
    e = jnp.exp(-dt)
    A = e * s2n / s2t
    B = diffusion.alpha_t(t_next) * (1.0 - e**2) / s2t
    kv = s2n * (1.0 - e**2) / s2t
    return A, B, kv


def run_learned(name, key, x0hat_fn, basis, Pz, az, y, b, N, T, tf,
                ess_frac=0.5):
    """One learned-arm run; returns the same dict shape as the exact samplers."""
    d = Pz.shape[0]
    ts_full = diffusion.time_grid(T, tf)
    k_init, k_steps = jax.random.split(jax.random.PRNGKey(0) if key is None else key)
    z = jnp.sqrt(diffusion.marginal_var(tf, Pz)) * jax.random.normal(k_init, (N, d))
    logw = _log_psi(tf, z, Pz, az, y, b) if name == "twisted" else jnp.zeros(N)
    log_z_acc = 0.0
    ess_traj = []

    if name == "dps":
        def dps_loss(z_flat, t):
            z0h = x0hat_fn(z_flat, t)
            return jnp.sum((az * z0h - y) ** 2) / (2.0 * b)
        dps_grad = jax.jit(jax.grad(dps_loss))

    step_keys = jax.random.split(k_steps, T)
    ts_j = jnp.asarray(ts_full)  # pass t as traced scalars: no per-step retrace
    for i in range(T):
        t, t_next = ts_j[i], ts_j[i + 1]
        dt = t - t_next
        k_noise, k_res = jax.random.split(step_keys[i])
        A, B, kv = _ancestral_coefs(t, t_next)
        z0h = x0hat_fn(z, t)
        m = A * z + B * z0h

        if name == "dps":
            # Euler guidance blows up once b makes the term stiff (observed
            # W2 ~ 1e106 in the pathway control). Damp with the ANALYTIC
            # linearization as preconditioner: per mode the guidance is
            # ~ lam_g z with lam_g = -2 a^2 c0^2 / b, so use the exponential-
            # integrator step factor (e^{lam dt}-1)/lam instead of dt. Exact
            # same small-dt limit; bounded displacement for any b.
            tm = 0.5 * (t + t_next)
            c0m = diffusion.x0hat_coef(tm, Pz)
            lam_g = -2.0 * az**2 * c0m**2 / b
            g1 = (jnp.exp(lam_g * dt) - 1.0) / jnp.where(
                jnp.abs(lam_g) > 1e-12, lam_g, 1.0)
            g1 = jnp.where(jnp.abs(lam_g) > 1e-12, g1, dt)
            m = m + g1 * 2.0 * (-dps_grad(z, t))
            z = m + jnp.sqrt(kv) * jax.random.normal(k_noise, z.shape)

        elif name in ("terminal_is", "sap"):
            z = m + jnp.sqrt(kv) * jax.random.normal(k_noise, z.shape)
            if name == "sap":
                z0h_new = x0hat_fn(z, t_next) if t_next > 0 else z
                pot = tilt.log_reward_over_beta(z0h_new, az, y, b)
                ess_traj.append(float(_ess(pot)))
                idx = systematic_resample(k_res, pot, N)
                z = z[idx]

        elif name == "twisted":
            c0n = diffusion.x0hat_coef(t_next, Pz)
            Dn = b + az**2 * diffusion.var0_t(t_next, Pz)
            denom_int = Dn + (az * c0n) ** 2 * kv
            # conjugate proposal q* ~ p_theta(z'|z) psi_hat(z'), closed form
            var_q = kv * Dn / denom_int
            m_q = (m * Dn + kv * az * c0n * y) / denom_int
            z_new = m_q + jnp.sqrt(var_q) * jax.random.normal(k_noise, z.shape)
            # incr = log int p psi' - log psi_t  (both closed-form)
            log_int = jnp.sum(0.5 * jnp.log(b / denom_int)
                              - (az * c0n * m - y) ** 2 / (2.0 * denom_int),
                              axis=-1)
            incr = log_int - _log_psi(t, z, Pz, az, y, b)
            logw = logw + incr
            z = z_new
            ess = float(_ess(logw))
            ess_traj.append(ess)
            if ess < ess_frac * N:
                log_z_acc += float(jax.nn.logsumexp(logw) - jnp.log(N))
                idx = systematic_resample(k_res, logw, N)
                z = z[idx]
                logw = jnp.zeros(N)

    out = {"z": z, "logw": logw}
    if name == "terminal_is":
        lw = tilt.log_reward_over_beta(z, az, y, b)
        out["logw"] = lw
        out["log_z_est"] = float(jax.nn.logsumexp(lw) - jnp.log(N))
        out["ess_final"] = float(_ess(lw))
    if name == "twisted":
        out["log_z_est"] = log_z_acc + float(jax.nn.logsumexp(logw) - jnp.log(N))
        out["ess_final"] = float(_ess(logw))
    if ess_traj:
        out["ess_traj"] = jnp.asarray(ess_traj)
    return out
