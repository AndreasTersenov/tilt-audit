"""Exact metrics vs the closed-form target sigma = N(mu*, Sig*), z-basis diagonal.

The sampler output is summarized by a weighted per-mode Gaussian fit
(m_k, v_k); W2 and KL are then Gaussian closed forms. The same fit applied to
the oracle's exact draws defines the finite-N floor, so fit bias cancels in
the scheme-vs-floor comparison (the kill criterion's 3x test).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import tilt


def weighted_moments(z, logw):
    w = jax.nn.softmax(logw)
    m = jnp.einsum("n,nd->d", w, z)
    v = jnp.einsum("n,nd->d", w, (z - m) ** 2)
    return m, v


def gaussian_w2sq(m, v, mu, Sig):
    """Squared W2 between diagonal Gaussians (sum over modes)."""
    return jnp.sum((m - mu) ** 2 + (jnp.sqrt(v) - jnp.sqrt(Sig)) ** 2)


def gaussian_kl(m, v, mu, Sig):
    """KL( N(m, v) || N(mu, Sig) ), diagonal, summed over modes."""
    return 0.5 * jnp.sum(v / Sig + (m - mu) ** 2 / Sig - 1.0 + jnp.log(Sig / v))


def ess(logw):
    w = jax.nn.softmax(logw)
    return 1.0 / jnp.sum(w**2)


def gamma_star(m, v, Pz, az, y, b, lo=-2.0, hi=3.0, num=2001):
    """Effective temperature: gamma minimizing KL(fit || sigma_gamma).

    sigma_gamma ~ p * e^{gamma r / beta} has per-mode
    Sig_g = (1/P + g a^2/b)^-1, mu_g = Sig_g g a y / b. gamma* > 1 = runs cold.
    Log-spaced scan + local parabolic refinement; scan bounds wide because SAP
    can run very cold.
    """
    gammas = jnp.logspace(lo, hi, num)

    def kl_of_gamma(g):
        Sig_g = 1.0 / (1.0 / Pz + g * az**2 / b)
        mu_g = Sig_g * g * az * y / b
        return gaussian_kl(m, v, mu_g, Sig_g)

    kls = jax.vmap(kl_of_gamma)(gammas)
    i = jnp.argmin(kls)
    # parabolic refinement in log-gamma on the winning triple
    i = jnp.clip(i, 1, num - 2)
    x0, x1, x2 = jnp.log(gammas[i - 1]), jnp.log(gammas[i]), jnp.log(gammas[i + 1])
    f0, f1, f2 = kls[i - 1], kls[i], kls[i + 1]
    denom = (f0 - 2.0 * f1 + f2)
    delta = jnp.where(jnp.abs(denom) > 1e-30,
                      0.5 * (f0 - f2) / denom * (x1 - x0), 0.0)
    return jnp.exp(x1 + delta), kls[i]


def functional_coverage(m, v, mu, Sig, c, level=0.68):
    """Coverage of the sampler's central `level` CI for linear functional c.z
    under the exact target. 0.68 in = calibrated; below = over-concentrated."""
    from jax.scipy.stats import norm
    mean_fit = jnp.dot(c, m)
    sd_fit = jnp.sqrt(jnp.dot(c**2, v))
    mean_tgt = jnp.dot(c, mu)
    sd_tgt = jnp.sqrt(jnp.dot(c**2, Sig))
    zq = norm.ppf(0.5 + level / 2.0)
    lo = mean_fit - zq * sd_fit
    hi = mean_fit + zq * sd_fit
    return norm.cdf((hi - mean_tgt) / sd_tgt) - norm.cdf((lo - mean_tgt) / sd_tgt)


def band_power_coverage(key, z, logw, mu, Sig, mask, level=0.68, n_mc=20000):
    """Coverage for the quadratic band-power functional B = mean_{k in band} z_k^2.

    CI from the weighted particle population (weighted quantiles); truth
    distribution by MC from the exact target (effectively exact at n_mc).
    """
    idx = jnp.where(mask, size=int(mask.sum()))[0]
    bp = jnp.mean(z[:, idx] ** 2, axis=1)
    w = jax.nn.softmax(logw)
    order = jnp.argsort(bp)
    cdf = jnp.cumsum(w[order])
    qlo, qhi = 0.5 - level / 2.0, 0.5 + level / 2.0
    lo = bp[order][jnp.searchsorted(cdf, qlo)]
    hi = bp[order][jnp.searchsorted(cdf, qhi)]
    draws = mu[idx] + jnp.sqrt(Sig[idx]) * jax.random.normal(key, (n_mc, idx.shape[0]))
    bp_true = jnp.mean(draws**2, axis=1)
    return jnp.mean((bp_true >= lo) & (bp_true <= hi))


def evaluate(key, out, Pz, az, y, b, basis, band_mask_list):
    """Full metric row for one sampler output dict."""
    mu, Sig = tilt.posterior_params(Pz, az, y, b)
    m, v = weighted_moments(out["z"], out["logw"])
    d = Pz.shape[0]
    n = basis.n
    # mean-field functional: pixel mean = z_DC / n; DC is the first self-conj coord
    c_mean = jnp.zeros(d).at[0].set(1.0 / n)
    row = {
        "w2": float(jnp.sqrt(gaussian_w2sq(m, v, mu, Sig))),
        "kl": float(gaussian_kl(m, v, mu, Sig)),
        "cov_mean_68": float(functional_coverage(m, v, mu, Sig, c_mean, 0.68)),
        "cov_mean_95": float(functional_coverage(m, v, mu, Sig, c_mean, 0.95)),
        "ess_final": float(ess(out["logw"])),
    }
    gs, gs_kl = gamma_star(m, v, Pz, az, y, b)
    row["gamma_star"] = float(gs)
    row["gamma_star_kl"] = float(gs_kl)
    for j, mask in enumerate(band_mask_list):
        key, sub = jax.random.split(key)
        row[f"cov_band{j}_68"] = float(
            band_power_coverage(sub, out["z"], out["logw"], mu, Sig,
                                jnp.asarray(mask), 0.68))
    if "log_z_est" in out:
        row["log_z_est"] = float(out["log_z_est"])
        row["log_z_analytic"] = float(tilt.log_z_analytic(Pz, az, y, b))
    if "ess_traj" in out:
        traj = jnp.asarray(out["ess_traj"])
        row["ess_traj_min"] = float(traj.min())
        row["ess_traj_mean"] = float(traj.mean())
    return row
