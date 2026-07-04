"""Transfer-grid samplers: exact GRF prior dynamics + NONLINEAR likelihood
guidance through the lognormal observation (plan §3).

The prior is the unchanged GRF, so the unguided backward kernel stays exact
per mode; only the guidance changes — gradients flow through
y = A.kappa(g) + n via autodiff, exactly the way practitioners wire DPS/Remy
into a nonlinear forward model. State in the z-basis, (N, d).

Arms (twisted is EXCLUDED: no closed twist off-Gaussian — plan §3):
  dps           Tweedie x0hat plug-in, denominator b (the standard biased arm)
  dps_inflated  denominator b + a^2 * Var-proxy; the Var-proxy is PINNED as
                var0_t(t, Pz) per mode THROUGH THE LINEARIZED MAP — reduces to
                exact_guidance in the lam->0 limit
  remy          per-level preconditioned ULA, likelihood noise inflated by the
                diffusion level: b + sigma_t^2 a^2 (Remy et al. scheme), K
                steps per level
  terminal_is   exact unguided prior chain + terminal e^{r/beta} reweight

Guidance displacement uses the SAME damping + clip conventions as
samplers_learned.run_learned (analytic exponential-integrator preconditioner;
3x step-noise clip with clip_frac recorded) so transfer rows are comparable
with the Gaussian-bench learned rows.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import diffusion, lognormal, tilt
from .fields import Basis
from .samplers import _ess, backward_kernel, systematic_resample


def _resid_ll(z0, basis, az, y, b_or_D, lam):
    """-||A kappa(z0) - y||^2 / (2 D), summed over batch (for jax.grad)."""
    r = lognormal.forward_z(z0, basis, az, lam) - y
    return -jnp.sum(r**2 / (2.0 * b_or_D))


def run_transfer(name, key, basis: Basis, Pz, az, y, b, lam,
                 N=256, T=64, tf=9.0, K=30, eps0=0.1):
    d = Pz.shape[0]
    ts_full = jnp.asarray(diffusion.time_grid(T, tf))
    k_init, k_steps = jax.random.split(key)
    z = jnp.sqrt(diffusion.marginal_var(tf, Pz)) * jax.random.normal(
        k_init, (N, d))
    step_keys = jax.random.split(k_steps, T)

    if name == "terminal_is":
        def step(zc, inp):
            t, t_next, k = inp
            coef, kvar = backward_kernel(t, t - t_next, Pz)
            return coef * zc + jnp.sqrt(kvar) * jax.random.normal(k, zc.shape), None
        z, _ = jax.lax.scan(step, z, (ts_full[:-1], ts_full[1:], step_keys))
        lw = _resid_ll_batch(z, basis, az, y, b, lam)
        return {"z": z, "logw": lw,
                "log_z_est": float(jax.nn.logsumexp(lw) - jnp.log(N)),
                "ess_final": float(_ess(lw))}

    if name in ("dps", "dps_inflated"):
        @jax.jit
        def guid_grad(zc, t, D):
            def loss(zf):
                z0h = diffusion.x0hat_coef(t, Pz) * zf
                return -_resid_ll(z0h, basis, az, y, D, lam)
            return jax.grad(loss)(zc)

        clip_events = []
        for i in range(T):
            t, t_next = ts_full[i], ts_full[i + 1]
            dt = t - t_next
            k_noise = step_keys[i]
            coef, kvar = backward_kernel(t, dt, Pz)
            m = coef * z
            tm = 0.5 * (t + t_next)
            D = (b if name == "dps"
                 else b + az**2 * diffusion.var0_t(tm, Pz))
            # analytic linearized preconditioner + clip (run_learned pinned)
            c0m = diffusion.x0hat_coef(tm, Pz)
            lam_g = -2.0 * az**2 * c0m**2 / D
            g1 = (jnp.exp(lam_g * dt) - 1.0) / jnp.where(
                jnp.abs(lam_g) > 1e-12, lam_g, 1.0)
            g1 = jnp.where(jnp.abs(lam_g) > 1e-12, g1, dt)
            disp = g1 * 2.0 * (-guid_grad(z, t, D))
            # per-particle L2 cap at 3x the step's total noise scale
            # (run_learned's 3*sqrt(kv*d) with scalar kv == 3*sqrt(sum kv))
            cap = 3.0 * jnp.sqrt(jnp.sum(jnp.maximum(kvar, 1e-12)))
            norms = jnp.linalg.norm(disp, axis=-1, keepdims=True)
            disp = disp * jnp.minimum(1.0, cap / jnp.maximum(norms, 1e-30))
            clip_events.append(float(jnp.mean(norms[:, 0] > cap)))
            z = m + disp + jnp.sqrt(kvar) * jax.random.normal(k_noise, z.shape)
        return {"z": z, "logw": jnp.zeros(N),
                "clip_frac": float(sum(clip_events) / len(clip_events))}

    if name == "remy":
        # Mirrors samplers.remy (sigma2 inflation) EXACTLY — same levels
        # (ts_full[1:], ending at t=0), same linearized-target preconditioner
        # vtil, same ULA discretization z + eps/2 * score + sqrt(eps) * xi —
        # so the K-convergence comparison across substrates (P-20260704n) is
        # scheme-identical; only the likelihood score is nonlinear.
        @jax.jit
        def level(z, inp):
            t, keys_lvl = inp
            V = diffusion.marginal_var(t, Pz)
            denom = b + diffusion.sig2_t(t) * az**2
            vtil = 1.0 / (1.0 / V + az**2 / denom)  # linearized level target
            eps = eps0 * vtil

            def ula(zc, k):
                def logp(zf):
                    lp = -0.5 * jnp.sum(zf**2 / V)
                    return lp + _resid_ll(zf, basis, az, y, denom, lam)
                s = jax.grad(logp)(zc)
                return (zc + 0.5 * eps * s
                        + jnp.sqrt(eps) * jax.random.normal(k, zc.shape),
                        None)
            z, _ = jax.lax.scan(ula, z, keys_lvl)
            return z, None

        levels = ts_full[1:]  # T levels, ending exactly at t=0
        lvl_keys = jax.random.split(k_steps, T * K).reshape(T, K, 2)
        z, _ = jax.lax.scan(level, z, (levels, lvl_keys))
        return {"z": z, "logw": jnp.zeros(N)}

    raise ValueError(name)


def _resid_ll_batch(z0, basis, az, y, b, lam):
    r = lognormal.forward_z(z0, basis, az, lam) - y
    return -jnp.sum(r**2, axis=-1) / (2.0 * b)
