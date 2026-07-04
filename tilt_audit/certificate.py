"""Residual path-space certificates for guided pointwise samplers.

Plan: docs/PLAN_CERT_KILLTEST.md (P-20260704a-c). For a sampler whose per-step
kernel is Gaussian per mode and applied without resampling, the residual
log-weight of each trajectory against the exact Feynman-Kac target
Q* ~ P_ref * e^{r(x0)/beta} (whose x0 marginal IS the tilted posterior):

    log w = sum_t [ log N(z_t'; m_ref, v_ref) - log N(z_t'; m_s, v_s) ]
            + r(z_0)/beta

Both kernels are closed-form here: the reference is samplers.backward_kernel
(exact backward prior chain), the sampler kernel is the exponential-integrator
guided step. The step math below MIRRORS samplers._run_pointwise line for line
(deliberate ~30-line duplication instead of refactoring the gate-frozen
module; gates G-C1..3 in scripts/gate_cert.py break loudly if they drift).

The certificate measures the chain's distance to its own intended target under
the model it was given — steering + discretization, NOT model error. With a
contaminated score, pass Pz_run as Pz and the true prior only enters the
caller's metrics.

Instruments per run (certify()):
    log_z_est      logsumexp(logw) - log N; E[Z_hat] = Z exactly (any N)
    ess_res        1 / sum(normalized w^2): repair affordability
    kl_path_hat    log_z_est - mean(logw): consistent estimator of the path
                   KL(P_sampler || Q*), an upper bound on the endpoint
                   KL(pi_sampler || pi) by data processing
    khat           PSIS generalized-Pareto tail index of the weights
                   (arviz.psislw): the "is ESS lying" diagnostic

Repair costs nothing: metrics.evaluate(out) with out["logw"] = the residual
weights IS the importance-repaired estimate.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from . import diffusion, tilt
from .samplers import backward_kernel


def run_pointwise_cert(key, Pz, az, y, b, N, T, tf, mode="dps",
                       modewise=False):
    """dps / exact_guidance / unguided trajectories + residual log-weights.

    Identical dynamics to samplers._run_pointwise (same integrator, same key
    layout within the step scan), plus per-step log-RN accumulation against
    the exact backward prior kernel. mode='unguided' is the plumbing anchor:
    kernels coincide, so the step-RN is identically zero and logw reduces to
    the terminal_is weights r(z0)/b.

    modewise=True additionally accumulates the per-mode decomposition of the
    log-RN (N, d) — exact here because modes are independent — the
    Rao-Blackwell/per-mode certificate used by the kill criterion's rescue
    clause. Memory: N*d fp64.
    """
    ts_full = jnp.asarray(diffusion.time_grid(T, tf))
    ts, dts = ts_full[:-1], ts_full[:-1] - ts_full[1:]
    k_init, k_steps = jax.random.split(key)
    z0 = jnp.sqrt(diffusion.marginal_var(tf, Pz)) * jax.random.normal(
        k_init, (N, Pz.shape[0]))
    d = Pz.shape[0]
    acc0 = jnp.zeros((N, d)) if modewise else jnp.zeros(N)

    def step(carry, inp):
        z, acc = carry
        t, dt, k = inp
        noise = jax.random.normal(k, z.shape)
        coef_r, kvar_r = backward_kernel(t, dt, Pz)
        m_ref = coef_r * z

        if mode == "unguided":
            z_new = m_ref + jnp.sqrt(kvar_r) * noise
            return (z_new, acc), None

        # guided kernel — mirrors samplers._run_pointwise exactly
        tm = t - 0.5 * dt
        V = diffusion.marginal_var(tm, Pz)
        c0 = diffusion.x0hat_coef(tm, Pz)
        denom = b if mode == "dps" else b + az**2 * diffusion.var0_t(tm, Pz)
        lam = 1.0 - 2.0 / V - 2.0 * az**2 * c0**2 / denom
        u = 2.0 * az * c0 * y / denom
        phi = jnp.exp(lam * dt)
        g1 = jnp.where(jnp.abs(lam) > 1e-12, (phi - 1.0) / lam, dt)
        g2 = jnp.where(jnp.abs(lam) > 1e-12, (phi**2 - 1.0) / lam, 2.0 * dt)
        m_s = phi * z + g1 * u
        z_new = m_s + jnp.sqrt(g2) * noise

        # log N(z'; m_ref, v_ref) - log N(z'; m_s, v_s) per mode;
        # (z' - m_s)^2 / v_s = noise^2 by construction
        incr = (0.5 * jnp.log(g2 / kvar_r)
                - 0.5 * (z_new - m_ref) ** 2 / kvar_r
                + 0.5 * noise**2)
        if not modewise:
            incr = jnp.sum(incr, axis=-1)
        return (z_new, acc + incr), None

    keys = jax.random.split(k_steps, T)
    (z, acc), _ = jax.lax.scan(step, (z0, acc0), (ts, dts, keys))
    r_modes = -(az * z - y) ** 2 / (2.0 * b)
    out = {"z": z}
    if modewise:
        out["logw_modes"] = acc + r_modes
        out["logw"] = jnp.sum(out["logw_modes"], axis=-1)
        out["step_rn"] = jnp.sum(acc, axis=-1)
    else:
        out["logw"] = acc + jnp.sum(r_modes, axis=-1)
        out["step_rn"] = acc
    out["log_z_est"] = jax.nn.logsumexp(out["logw"]) - jnp.log(N)
    return out


def run_cert(mode, key, Pz, az, y, b, N, T, tf, modewise=False):
    fn = functools.partial(run_pointwise_cert, mode=mode, modewise=modewise)
    return jax.jit(fn, static_argnames=("N", "T", "tf"))(
        key, Pz, az, y, b, N=N, T=T, tf=tf)


def certify_modewise(out):
    """Per-mode (Rao-Blackwell) certificate scalars: each mode's weights are
    a valid 1-D certificate on its own marginal; degeneracy does not compound
    across modes. Returns summaries over modes."""
    lwm = np.asarray(out["logw_modes"], dtype=np.float64)  # (N, d)
    N, d = lwm.shape
    lse = _logsumexp(lwm, axis=0) - np.log(N)              # (d,) logZ_k est
    kl_k = lse - lwm.mean(axis=0)                          # (d,) per-mode KLhat
    w = np.exp(lwm - lwm.max(axis=0, keepdims=True))
    w = w / w.sum(axis=0, keepdims=True)
    ess_k = 1.0 / np.sum(w**2, axis=0)                     # (d,)
    return {"kl_modes_sum": float(kl_k.sum()),
            "ess_modes_med": float(np.median(ess_k)),
            "ess_modes_min": float(ess_k.min()),
            "log_z_modes_sum": float(lse.sum())}


def _logsumexp(a, axis=None):
    amax = a.max(axis=axis, keepdims=True)
    return np.squeeze(amax, axis=axis) + np.log(
        np.exp(a - amax).sum(axis=axis))


def chain_law(mode, Pz, az, y, b, T, tf):
    """EXACT law of the discrete guided chain + exact certificate expectations.

    The chain is linear-Gaussian per mode, so its endpoint law N(m0, v0), the
    expected residual log-weight E[log w], and therefore the exact path KL
    (= log Z - E[log w]) and exact endpoint KL/W2 all have closed forms —
    a zero-noise ground-truth axis for the kill test, and an exact check of
    the data-processing inequality kl_path >= kl_end.
    """
    Pz = np.asarray(Pz, dtype=np.float64)
    az_ = np.asarray(az, dtype=np.float64)
    y_ = np.asarray(y, dtype=np.float64)
    ts_full = diffusion.time_grid(T, tf)
    m = np.zeros_like(Pz)
    v = np.asarray(diffusion.marginal_var(tf, Pz), dtype=np.float64)
    e_logw = 0.0
    for i in range(T):
        t, t_next = ts_full[i], ts_full[i + 1]
        dt = t - t_next
        e = np.exp(-dt)
        V_prev = np.asarray(diffusion.marginal_var(t_next, Pz))
        V_t = np.asarray(diffusion.marginal_var(t, Pz))
        coef_r = e * V_prev / V_t
        kvar_r = V_prev - e**2 * V_prev**2 / V_t
        if mode == "unguided":
            m, v = coef_r * m, coef_r**2 * v + kvar_r
            continue
        tm = t - 0.5 * dt
        Vm = np.asarray(diffusion.marginal_var(tm, Pz))
        c0 = np.asarray(diffusion.x0hat_coef(tm, Pz))
        denom = b if mode == "dps" else b + az_**2 * np.asarray(
            diffusion.var0_t(tm, Pz))
        lam = 1.0 - 2.0 / Vm - 2.0 * az_**2 * c0**2 / denom
        u = 2.0 * az_ * c0 * y_ / denom
        phi = np.exp(lam * dt)
        g1 = np.where(np.abs(lam) > 1e-12, (phi - 1.0) / lam, dt)
        g2 = np.where(np.abs(lam) > 1e-12, (phi**2 - 1.0) / lam, 2.0 * dt)
        # E over z_t ~ N(m, v) of the step increment:
        #   0.5 log(g2/kvar_r) - 0.5 E[(z' - m_ref)^2]/kvar_r + 0.5
        # with z' - m_ref = (phi - coef_r) z + g1 u + sqrt(g2) eps
        a_d = phi - coef_r
        mu_d = a_d * m + g1 * u
        var_d = a_d**2 * v + g2
        e_logw += float(np.sum(0.5 * np.log(g2 / kvar_r)
                               - 0.5 * (mu_d**2 + var_d) / kvar_r + 0.5))
        m, v = phi * m + g1 * u, phi**2 * v + g2
    # terminal reward expectation over z0 ~ N(m, v)
    e_logw += float(-np.sum((az_ * m - y_) ** 2 + az_**2 * v) / (2.0 * b))
    log_z = float(tilt.log_z_analytic(jnp.asarray(Pz), jnp.asarray(az_),
                                      jnp.asarray(y_), b))
    return {"m_end": m, "v_end": v, "e_logw": e_logw,
            "log_z_analytic": log_z, "kl_path_exact": log_z - e_logw}


def endpoint_errors(law, Pz_true, az, y, b):
    """Exact endpoint KL/W2 of the chain law vs the target built on Pz_true."""
    from . import metrics
    mu, Sig = tilt.posterior_params(jnp.asarray(Pz_true), jnp.asarray(az),
                                    jnp.asarray(y), b)
    m = jnp.asarray(law["m_end"])
    v = jnp.asarray(law["v_end"])
    return {"kl_end_exact": float(metrics.gaussian_kl(m, v, mu, Sig)),
            "w2_end_exact": float(jnp.sqrt(metrics.gaussian_w2sq(m, v, mu,
                                                                 Sig)))}


def run_learned_cert(name, key, x0hat_fn, basis, Pz, az, y, b, N, T, tf,
                     clip=True, gscale=1.0):
    """Certificate-instrumented learned-pathway sampler (Stage 2).

    Mirrors samplers_learned.run_learned's 'dps' and ancestral-unguided
    branches (kept in sync by eye; gates G-L1/2 break loudly on drift). The
    guided and reference kernels share the per-step variance kv and differ
    only by the guidance displacement (clip included), so the per-step
    per-mode log-RN is  -(disp^2 + 2 disp sqrt(kv) eps) / (2 kv)  — free to
    accumulate; no extra net evaluations. Certificate target: the LEARNED
    prior chain tilted by e^{r/beta} (steering error given the net).
    """
    from . import samplers_learned as sl
    d = Pz.shape[0]
    ts_full = diffusion.time_grid(T, tf)
    k_init, k_steps = jax.random.split(key)
    z = jnp.sqrt(diffusion.marginal_var(tf, Pz)) * jax.random.normal(
        k_init, (N, d))
    acc = jnp.zeros((N, d))
    clip_events = []

    if name == "dps":
        def dps_loss(z_flat, t):
            z0h = x0hat_fn(z_flat, t)
            return jnp.sum((az * z0h - y) ** 2) / (2.0 * b)
        dps_grad = jax.jit(jax.grad(dps_loss))

    step_keys = jax.random.split(k_steps, T)
    ts_j = jnp.asarray(ts_full)
    for i in range(T):
        t, t_next = ts_j[i], ts_j[i + 1]
        dt = t - t_next
        k_noise, _ = jax.random.split(step_keys[i])
        A, B, kv = sl._ancestral_coefs(t, t_next)
        z0h = x0hat_fn(z, t)
        m = A * z + B * z0h
        noise = jax.random.normal(k_noise, z.shape)

        if name == "dps":
            tm = 0.5 * (t + t_next)
            c0m = diffusion.x0hat_coef(tm, Pz)
            lam_g = -2.0 * az**2 * c0m**2 / b
            g1 = (jnp.exp(lam_g * dt) - 1.0) / jnp.where(
                jnp.abs(lam_g) > 1e-12, lam_g, 1.0)
            g1 = jnp.where(jnp.abs(lam_g) > 1e-12, g1, dt)
            disp = gscale * g1 * 2.0 * (-dps_grad(z, t))
            if clip:
                cap = 3.0 * jnp.sqrt(jnp.maximum(kv, 1e-6) * d)
                norms = jnp.linalg.norm(disp, axis=-1, keepdims=True)
                disp = disp * jnp.minimum(1.0, cap / jnp.maximum(norms, 1e-30))
                clip_events.append(float(jnp.mean(norms[:, 0] > cap)))
            if float(ts_full[i + 1]) > 0.0:  # stochastic step: RN well-defined
                acc = acc - (disp**2
                             + 2.0 * disp * jnp.sqrt(kv) * noise) / (2.0 * kv)
            # final step (kv=0) is deterministic: guided and reference kernels
            # are mutually singular; its O(dt) displacement is reported, not
            # certified (see docstring)
            z = m + disp + jnp.sqrt(kv) * noise
        else:  # unguided ancestral reference: disp = 0, RN increment = 0
            z = m + jnp.sqrt(kv) * noise

    logw_modes = acc + (-(az * z - y) ** 2 / (2.0 * b))
    logw = jnp.sum(logw_modes, axis=-1)
    out = {"z": z, "logw_modes": logw_modes, "logw": logw,
           "step_rn": jnp.sum(acc, axis=-1),
           "log_z_est": jax.nn.logsumexp(logw) - jnp.log(N)}
    if clip_events:
        out["clip_frac"] = float(sum(clip_events) / len(clip_events))
    return out


def chain_law_ancestral(mode, Pz, az, y, b, T, tf, gscale=1.0):
    """EXACT law + certificate expectations for the ANALYTIC-x0hat ancestral
    pathway (per mode linear; the pathway-control ground truth for Stage 2).
    Clip assumed inactive (verified empirically via clip_frac ~ 0)."""
    Pz_ = np.asarray(Pz, dtype=np.float64)
    az_ = np.asarray(az, dtype=np.float64)
    y_ = np.asarray(y, dtype=np.float64)
    ts_full = diffusion.time_grid(T, tf)
    m = np.zeros_like(Pz_)
    v = np.asarray(diffusion.marginal_var(tf, Pz_), dtype=np.float64)
    e_logw = 0.0
    for i in range(T):
        t, t_next = ts_full[i], ts_full[i + 1]
        dt = t - t_next
        s2t = 1.0 - np.exp(-2.0 * t)
        s2n = 1.0 - np.exp(-2.0 * t_next)
        e = np.exp(-dt)
        A = e * s2n / s2t
        B = np.exp(-t_next) * (1.0 - e**2) / s2t
        kv = s2n * (1.0 - e**2) / s2t
        c0 = np.asarray(diffusion.x0hat_coef(t, Pz_))
        lin = A + B * c0                        # unguided mean coefficient
        if mode == "dps":
            tm = 0.5 * (t + t_next)
            c0m = np.asarray(diffusion.x0hat_coef(tm, Pz_))
            lam_g = -2.0 * az_**2 * c0m**2 / b
            g1 = np.where(np.abs(lam_g) > 1e-12,
                          (np.exp(lam_g * dt) - 1.0) / np.where(
                              np.abs(lam_g) > 1e-12, lam_g, 1.0), dt)
            # disp = -2 g1 a c0 (a c0 z - y)/b  (linear in z)
            c1 = gscale * (-2.0) * g1 * az_**2 * c0**2 / b
            c2 = gscale * 2.0 * g1 * az_ * c0 * y_ / b
            mu_d = c1 * m + c2
            var_d = c1**2 * v
            if t_next > 0.0:  # the deterministic final step is uncertified
                e_logw += float(-np.sum((mu_d**2 + var_d) / (2.0 * kv)))
            m = lin * m + mu_d
            v = (lin + c1) ** 2 * v + kv
        else:
            m = lin * m
            v = lin**2 * v + kv
    e_logw += float(-np.sum((az_ * m - y_) ** 2 + az_**2 * v) / (2.0 * b))
    log_z = float(tilt.log_z_analytic(jnp.asarray(Pz_), jnp.asarray(az_),
                                      jnp.asarray(y_), b))
    return {"m_end": m, "v_end": v, "e_logw": e_logw,
            "log_z_analytic": log_z, "kl_path_exact": log_z - e_logw}


def remy_ais(key, Pz, az, y, b, N, T, tf, K=10, eps0=0.1):
    """EXPLORATORY: annealed-importance-sampling certificate for the Remy
    scheme. Mirrors samplers.remy's dynamics (kept in sync by eye; the gated
    sampler is untouched) and accumulates the standard AIS increments
    log pi_i(z) - log pi_{i-1}(z) at each cooling, with pi_i the level's
    sigma^2-inflated unnormalized target (level likelihood normalized in y).

    Validity caveat (documented in the plan): ULA does not leave each level
    target exactly invariant, so E[w] carries an O(eps0) bias — quantified
    against the analytic normalization log Z_ais = log_z_analytic
    - 0.5*sum(log 2 pi b).
    """
    ts_full = jnp.asarray(diffusion.time_grid(T, tf))
    levels, prevs = ts_full[1:], ts_full[:-1]
    k_init, k_steps = jax.random.split(key)
    z = jnp.sqrt(diffusion.marginal_var(tf, Pz)) * jax.random.normal(
        k_init, (N, Pz.shape[0]))

    def log_p(t, z):
        V = diffusion.marginal_var(t, Pz)
        return -0.5 * jnp.sum(z**2 / V + jnp.log(2 * jnp.pi * V), axis=-1)

    def log_lik(t, z):
        denom = b + diffusion.sig2_t(t) * az**2
        return -0.5 * jnp.sum((az * z - y) ** 2 / denom
                              + jnp.log(2 * jnp.pi * denom), axis=-1)

    def level_step(carry, inp):
        z, logw = carry
        t, t_prev, k_level = inp
        # AIS increment at the incoming state (level t_prev's sample)
        incr = log_p(t, z) - log_p(t_prev, z) + log_lik(t, z)
        # subtract previous level's likelihood except at the first level,
        # where pi_0 = p_{tf} alone: sig2_t(tf) ~ 1 and t_prev == tf marks it
        incr = incr - jnp.where(t_prev >= ts_full[0], 0.0,
                                log_lik(t_prev, z))
        logw = logw + incr
        keys_level = jax.random.split(k_level, K)
        V = diffusion.marginal_var(t, Pz)
        denom = b + diffusion.sig2_t(t) * az**2
        vtil = 1.0 / (1.0 / V + az**2 / denom)

        def langevin(z, kk):
            score = -z / V - (az * z - y) * az / denom
            return (z + 0.5 * eps0 * vtil * score
                    + jnp.sqrt(eps0 * vtil) * jax.random.normal(kk, z.shape)), None

        z, _ = jax.lax.scan(langevin, z, keys_level)
        return (z, logw), None

    keys = jax.random.split(k_steps, T)
    (z, logw), _ = jax.lax.scan(level_step, (z, jnp.zeros(N)),
                                (levels, prevs, keys))
    log_z_ais = (tilt.log_z_analytic(Pz, az, y, b)
                 - 0.5 * jnp.sum(jnp.log(2 * jnp.pi * b
                                         * jnp.ones_like(Pz))))
    return {"z": z, "logw": logw, "log_z_ais_analytic": log_z_ais,
            "log_z_est": jax.nn.logsumexp(logw) - jnp.log(N)}


def certify(out):
    """Certificate scalars from a run_pointwise_cert output dict."""
    from arviz import psislw
    logw = np.asarray(out["logw"], dtype=np.float64)
    lw = logw - logw.max()
    wbar = np.exp(lw) / np.exp(lw).sum()
    ess = 1.0 / np.sum(wbar**2)
    log_z_est = float(out["log_z_est"])
    kl_path = log_z_est - float(logw.mean())
    _, khat = psislw(logw - logw.max())
    return {"ess_res": float(ess), "khat": float(khat),
            "kl_path_hat": float(kl_path), "log_z_est": log_z_est,
            "logw_std": float(logw.std())}
