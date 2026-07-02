"""The five samplers (plus one control), one target.

All exact-score arms run in the real diagonal z-basis with state (N, d):
per-coordinate linear dynamics, global SMC weights/resampling. The unguided
dynamics use the EXACT per-mode backward OU kernel (closed form here), so the
prior chain hits N(0, P) at t=0 for any step count T and the log-Z estimator
targets the true analytic Z — guidance terms enter as Euler shifts
2*g*dt on top, so scheme bias is isolated from prior-discretization error.
Particles initialize from the exact prior marginal N(0, V_tf).

Samplers:
  oracle          direct per-mode draws from sigma (finite-N floor reference)
  dps             Tweedie plug-in guidance, no covariance inflation (the bias)
  exact_guidance  plug-in with the exact inflation term (discretization
                  control; exploratory addition, not in the frozen T1 four)
  sap             reward-as-potential SMC: resample every step on e^{r(x0hat)/beta}
  twisted         proper twisted SMC: closed-form optimal twist, ESS-gated
                  resampling, unbiased log-Z accumulator
  terminal_is     unguided reverse diffusion + terminal IS reweight (best-of-N)

Each returns a dict with z (N, d), logw (N,), and per-sampler diagnostics.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from . import diffusion, tilt


def _ess(logw):
    w = jax.nn.softmax(logw)
    return 1.0 / jnp.sum(w**2)


def systematic_resample(key, logw, N):
    w = jax.nn.softmax(logw)
    u = (jax.random.uniform(key) + jnp.arange(N)) / N
    idx = jnp.searchsorted(jnp.cumsum(w), u)
    return jnp.clip(idx, 0, N - 1)


def _log_psi(t, z, Pz, az, y, b):
    """log E[e^{r(x0)/beta} | x_t]: the closed-form optimal twist, summed over modes."""
    v = diffusion.var0_t(t, Pz)
    x0h = diffusion.x0hat_coef(t, Pz) * z
    denom = b + az**2 * v
    return jnp.sum(0.5 * jnp.log(b / denom) - (az * x0h - y) ** 2 / (2.0 * denom),
                   axis=-1)


def _guidance(t, z, Pz, az, y, b, inflate: bool):
    """Gradient of log N(y; a*x0hat, b [+ a^2 Var0]) wrt z (per mode)."""
    c0 = diffusion.x0hat_coef(t, Pz)
    x0h = c0 * z
    denom = b + (az**2 * diffusion.var0_t(t, Pz) if inflate else 0.0)
    return -(az * x0h - y) * az * c0 / denom


def backward_kernel(t, dt, Pz):
    """EXACT per-mode backward transition of the prior OU process.

    p(x_{t-dt} | x_t) = N(coef * x_t, var). Exact kernels compose exactly, so
    the unguided chain's t=0 marginal is the prior N(0, P) for ANY step count:
    no Euler bias enters the FK target or the log-Z estimator, and guidance
    scheme bias is isolated from time-discretization error by construction.
    """
    e = jnp.exp(-dt)
    V_prev = diffusion.marginal_var(t - dt, Pz)
    V_t = diffusion.marginal_var(t, Pz)
    coef = e * V_prev / V_t
    var = V_prev - e**2 * V_prev**2 / V_t
    return coef, var


def oracle(key, Pz, az, y, b, N, T, tf):
    mu, Sig = tilt.posterior_params(Pz, az, y, b)
    z = mu + jnp.sqrt(Sig) * jax.random.normal(key, (N, Pz.shape[0]))
    return {"z": z, "logw": jnp.zeros(N)}


def _run_pointwise(key, Pz, az, y, b, N, T, tf, mode):
    """Shared Euler loop for dps / exact_guidance / unguided prior dynamics."""
    ts_full = jnp.asarray(diffusion.time_grid(T, tf))
    ts, dts = ts_full[:-1], ts_full[:-1] - ts_full[1:]
    k_init, k_steps = jax.random.split(key)
    z0 = jnp.sqrt(diffusion.marginal_var(tf, Pz)) * jax.random.normal(
        k_init, (N, Pz.shape[0]))

    def step(z, inp):
        t, dt, k = inp
        noise = jax.random.normal(k, z.shape)
        if mode == "unguided":
            coef, kvar = backward_kernel(t, dt, Pz)
            return coef * z + jnp.sqrt(kvar) * noise, None
        # Guided arms: the reverse drift is linear per mode,
        # dz/dtau = Lam z + u + sqrt(2) dW, so each step integrates EXACTLY
        # with coefficients frozen at the step midpoint (exponential
        # integrator). Unconditionally stable — forward Euler blows up at
        # T=64 once the calibrated b makes the guidance stiff.
        tm = t - 0.5 * dt
        V = diffusion.marginal_var(tm, Pz)
        c0 = diffusion.x0hat_coef(tm, Pz)
        denom = b if mode == "dps" else b + az**2 * diffusion.var0_t(tm, Pz)
        lam = 1.0 - 2.0 / V - 2.0 * az**2 * c0**2 / denom
        u = 2.0 * az * c0 * y / denom
        phi = jnp.exp(lam * dt)
        # (phi - 1)/lam and (phi^2 - 1)/lam, safe as lam -> 0
        g1 = jnp.where(jnp.abs(lam) > 1e-12, (phi - 1.0) / lam, dt)
        g2 = jnp.where(jnp.abs(lam) > 1e-12, (phi**2 - 1.0) / lam, 2.0 * dt)
        return phi * z + g1 * u + jnp.sqrt(g2) * noise, None

    keys = jax.random.split(k_steps, T)
    z, _ = jax.lax.scan(step, z0, (ts, dts, keys))
    return z


def dps(key, Pz, az, y, b, N, T, tf):
    z = _run_pointwise(key, Pz, az, y, b, N, T, tf, "dps")
    return {"z": z, "logw": jnp.zeros(N)}


def exact_guidance(key, Pz, az, y, b, N, T, tf):
    z = _run_pointwise(key, Pz, az, y, b, N, T, tf, "exact_guidance")
    return {"z": z, "logw": jnp.zeros(N)}


def terminal_is(key, Pz, az, y, b, N, T, tf):
    z = _run_pointwise(key, Pz, az, y, b, N, T, tf, "unguided")
    logw = tilt.log_reward_over_beta(z, az, y, b)
    log_z_est = jax.nn.logsumexp(logw) - jnp.log(N)
    return {"z": z, "logw": logw, "log_z_est": log_z_est, "ess_final": _ess(logw)}


def sap(key, Pz, az, y, b, N, T, tf):
    """Reward-as-potential SMC: prior dynamics, resample EVERY step on e^{r(x0hat)/beta}.

    Deliberately improper (the SAP analog): potentials compound through
    resampling, so the effective tilt grows with depth T. That pathology is
    the thing being audited, not a bug.
    """
    ts_full = jnp.asarray(diffusion.time_grid(T, tf))
    ts, dts = ts_full[:-1], ts_full[:-1] - ts_full[1:]
    k_init, k_steps = jax.random.split(key)
    z0 = jnp.sqrt(diffusion.marginal_var(tf, Pz)) * jax.random.normal(
        k_init, (N, Pz.shape[0]))

    def step(z, inp):
        t, dt, k = inp
        k_noise, k_res = jax.random.split(k)
        coef, kvar = backward_kernel(t, dt, Pz)
        z = coef * z + jnp.sqrt(kvar) * jax.random.normal(k_noise, z.shape)
        t_next = t - dt
        x0h = diffusion.x0hat_coef(t_next, Pz) * z
        pot = tilt.log_reward_over_beta(x0h, az, y, b)
        ess = _ess(pot)
        idx = systematic_resample(k_res, pot, N)
        return z[idx], ess

    keys = jax.random.split(k_steps, T)
    z, ess_traj = jax.lax.scan(step, z0, (ts, dts, keys))
    return {"z": z, "logw": jnp.zeros(N), "ess_traj": ess_traj}


def twisted(key, Pz, az, y, b, N, T, tf, ess_frac=0.5, proposal="conjugate"):
    """Proper twisted SMC with the closed-form optimal twist.

    proposal='conjugate' (the T1 arm): q* ~ p(z'|z) psi_t'(z') per mode in
    closed form — the proper twisted sampler. With exact kernel and exact
    twist the tower property makes the incremental weights identically zero
    (fully-conjugate linear-Gaussian case); the general weight formula is
    still evaluated every step, so any error anywhere in the kernel / twist /
    posterior algebra shows up as nonzero weights (max_abs_incr diagnostic).
    With learned or misspecified scores the conjugacy breaks and the weights
    do real work.

    proposal='prior' (exploratory diagnostic): exact backward-prior proposals,
    twist enters only through the weights (pure psi_t/psi_t-1 potentials).
    Valid & unbiased, but weight-degenerates as d grows — the display of WHY
    twisted proposals matter; its ESS trajectory feeds the certificate story.
    """
    ts_full = jnp.asarray(diffusion.time_grid(T, tf))
    ts, dts = ts_full[:-1], ts_full[:-1] - ts_full[1:]
    k_init, k_steps = jax.random.split(key)
    z = jnp.sqrt(diffusion.marginal_var(tf, Pz)) * jax.random.normal(
        k_init, (N, Pz.shape[0]))
    logw = _log_psi(tf, z, Pz, az, y, b)

    def step(carry, inp):
        z, logw, log_z_acc, max_incr = carry
        t, dt, k = inp
        k_noise, k_res = jax.random.split(k)
        t_next = t - dt
        coef, kvar = backward_kernel(t, dt, Pz)
        m_p = coef * z
        if proposal == "prior":
            z_new = m_p + jnp.sqrt(kvar) * jax.random.normal(k_noise, z.shape)
            incr = (_log_psi(t_next, z_new, Pz, az, y, b)
                    - _log_psi(t, z, Pz, az, y, b))
        else:  # conjugate twisted proposal
            c0n = diffusion.x0hat_coef(t_next, Pz)
            Dn = b + az**2 * diffusion.var0_t(t_next, Pz)
            prec_q = 1.0 / kvar + (az * c0n) ** 2 / Dn
            var_q = 1.0 / prec_q
            m_q = var_q * (m_p / kvar + az * c0n * y / Dn)
            z_new = m_q + jnp.sqrt(var_q) * jax.random.normal(k_noise, z.shape)
            log_pq = jnp.sum(
                -0.5 * (z_new - m_p) ** 2 / kvar - 0.5 * jnp.log(kvar)
                + 0.5 * (z_new - m_q) ** 2 / var_q + 0.5 * jnp.log(var_q),
                axis=-1)
            incr = (log_pq + _log_psi(t_next, z_new, Pz, az, y, b)
                    - _log_psi(t, z, Pz, az, y, b))
        logw = logw + incr
        max_incr = jnp.maximum(max_incr, jnp.abs(incr).max())
        ess = _ess(logw)

        def do_resample(args):
            z_new, logw, log_z_acc = args
            log_z_acc = log_z_acc + jax.nn.logsumexp(logw) - jnp.log(N)
            idx = systematic_resample(k_res, logw, N)
            return z_new[idx], jnp.zeros(N), log_z_acc

        z_out, logw_out, log_z_out = jax.lax.cond(
            ess < ess_frac * N, do_resample, lambda a: a, (z_new, logw, log_z_acc))
        return (z_out, logw_out, log_z_out, max_incr), ess

    keys = jax.random.split(k_steps, T)
    (z, logw, log_z_acc, max_incr), ess_traj = jax.lax.scan(
        step, (z, logw, 0.0, 0.0), (ts, dts, keys))
    log_z_est = log_z_acc + jax.nn.logsumexp(logw) - jnp.log(N)
    return {"z": z, "logw": logw, "log_z_est": log_z_est, "ess_traj": ess_traj,
            "ess_final": _ess(logw), "max_abs_incr": max_incr}


SAMPLERS = {
    "oracle": oracle,
    "dps": dps,
    "exact_guidance": exact_guidance,
    "sap": sap,
    "twisted": twisted,
    "terminal_is": terminal_is,
}


def run_sampler(name, key, Pz, az, y, b, N, T, tf, **kw):
    fn = SAMPLERS[name]
    jitted = jax.jit(functools.partial(fn, **kw) if kw else fn,
                     static_argnames=("N", "T", "tf"))
    return jitted(key, Pz, az, y, b, N=N, T=T, tf=tf)
