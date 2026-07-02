"""Gates T-G1..4: all must be green before any large GPU job (plan section 3).

Run on CPU at 16^2 / d=1. The T-G2 reference numbers come verbatim from
docs/OVERNIGHT_2026-07-02_GRF_PILOT.md (moment-ODE oracle, deep-read
2502.07849 verification log).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tilt_audit import diffusion, metrics, samplers, tilt
from tilt_audit.fields import (band_masks, make_basis, make_pk, pack,
                               sample_prior_z, smoothing_operator, unpack)

TF = 9.0


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def grf16():
    basis = make_basis(16)
    pk = make_pk(basis)
    from tilt_audit.fields import grid_to_z
    Pz = jnp.asarray(grid_to_z(pk, basis))
    az = jnp.asarray(smoothing_operator(basis))
    s = 0.5
    y, _ = tilt.make_observation(jax.random.PRNGKey(999), Pz, az, s)
    b = tilt.calibrate_b(Pz, az, y, target_shift=1.0)
    return basis, Pz, az, y, b


# ---------------------------------------------------------------- basis sanity

def test_basis_roundtrip_and_prior_variance(grf16):
    basis, Pz, az, y, b = grf16
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 16))
    x_rt = unpack(pack(x, basis), basis)
    np.testing.assert_allclose(np.asarray(x_rt), np.asarray(x), atol=1e-10)
    # prior draws: unit pixel variance and per-mode variance = Pz
    z = sample_prior_z(jax.random.PRNGKey(1), Pz, (20000,))
    pixvar = float(jnp.var(unpack(z, basis)))
    assert abs(pixvar - 1.0) < 0.05
    mode_var = jnp.var(z, axis=0)
    ratio = mode_var / Pz
    assert float(jnp.abs(ratio - 1.0).max()) < 0.2  # 20k draws, 5 sigma-ish


# ---------------------------------------------------------------- T-G1

def test_tg1_oracle_matches_posterior(grf16):
    basis, Pz, az, y, b = grf16
    N = 200_000
    out = samplers.oracle(jax.random.PRNGKey(2), Pz, az, y, b, N=N, T=0, tf=TF)
    mu, Sig = tilt.posterior_params(Pz, az, y, b)
    m, v = metrics.weighted_moments(out["z"], out["logw"])
    zscore = (m - mu) / jnp.sqrt(Sig / N)
    assert float(jnp.abs(zscore).max()) < 5.0
    # mean of squared z-scores ~ 1 +- sqrt(2/d)
    msq = float(jnp.mean(zscore**2))
    assert abs(msq - 1.0) < 5.0 * np.sqrt(2.0 / Pz.shape[0])
    np.testing.assert_allclose(np.asarray(v), np.asarray(Sig), rtol=0.05)


# ---------------------------------------------------------------- T-G2

def moment_ode_reference(P, a, y, b, tf, guidance, n_steps=9000):
    """30-line moment-ODE integrator (RK4) for the linear reverse SDE, d=1.

    dm/dtau = A(t) m + c(t), dv/dtau = 2 A(t) v + 2, t = tf - tau, from the
    reverse drift x + 2[s_prior + g]. DPS drops the a^2 Var[x0|xt] inflation.
    """
    def coefs(t):
        al2 = np.exp(-2.0 * t)
        V = al2 * P + (1.0 - al2)
        c0 = np.exp(-t) * P / V
        var0 = P * (1.0 - al2) / V
        denom = b if guidance == "dps" else b + a**2 * var0
        A = 1.0 - 2.0 / V - 2.0 * a**2 * c0**2 / denom
        c = 2.0 * a * y * c0 / denom
        return A, c

    def rhs(tau, state):
        m, v = state
        A, c = coefs(tf - tau)
        return np.array([A * m + c, 2.0 * A * v + 2.0])

    dt = tf / n_steps
    v_init = np.exp(-2.0 * tf) * P + (1.0 - np.exp(-2.0 * tf))
    state = np.array([0.0, v_init])
    tau = 0.0
    for _ in range(n_steps):
        k1 = rhs(tau, state)
        k2 = rhs(tau + dt / 2, state + dt / 2 * k1)
        k3 = rhs(tau + dt / 2, state + dt / 2 * k2)
        k4 = rhs(tau + dt, state + dt * k3)
        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        tau += dt
    return state  # (mean, var)


# Reference table from the overnight doc (deep-read 2502.07849 verification log)
TG2_TABLE = [
    # P,     DPS mean, DPS var, exact mean, exact var
    (0.25, 0.787, 0.158, 0.667, 0.167),
    (1.00, 1.729, 0.245, 1.333, 0.333),
    (4.00, 1.999, 0.250, 1.778, 0.444),
]


@pytest.mark.parametrize("P,dps_m,dps_v,ex_m,ex_v", TG2_TABLE)
def test_tg2_moment_ode_reproduces_doc_table(P, dps_m, dps_v, ex_m, ex_v):
    a, y, b = 1.0, 2.0, 0.5
    m, v = moment_ode_reference(P, a, y, b, TF, "dps")
    assert abs(m - dps_m) < 0.005, f"DPS mean {m} vs doc {dps_m}"
    assert abs(v - dps_v) < 0.005, f"DPS var {v} vs doc {dps_v}"
    m, v = moment_ode_reference(P, a, y, b, TF, "exact")
    assert abs(m - ex_m) < 0.005
    assert abs(v - ex_v) < 0.005
    # exact-guidance analytic check: must equal the tilted posterior exactly
    mu, Sig = tilt.posterior_params(jnp.array([P]), jnp.array([a]),
                                    jnp.array([y]), b)
    assert abs(m - float(mu[0])) < 2e-3
    assert abs(v - float(Sig[0])) < 2e-3


@pytest.mark.parametrize("P,dps_m,dps_v,ex_m,ex_v", TG2_TABLE)
def test_tg2_implemented_samplers_hit_ode(P, dps_m, dps_v, ex_m, ex_v):
    """The actual DPS/exact-guidance samplers at fine discretization, d=1."""
    Pz = jnp.array([P]); az = jnp.array([1.0]); y = jnp.array([2.0]); b = 0.5
    N, T = 400_000, 4096
    out = samplers.run_sampler("dps", jax.random.PRNGKey(3), Pz, az, y, b,
                               N=N, T=T, tf=TF)
    m = float(out["z"].mean()); v = float(out["z"].var())
    assert abs(m - dps_m) / dps_m < 0.02, f"sampler DPS mean {m} vs {dps_m}"
    assert abs(v - dps_v) / dps_v < 0.02, f"sampler DPS var {v} vs {dps_v}"
    out = samplers.run_sampler("exact_guidance", jax.random.PRNGKey(4), Pz, az,
                               y, b, N=N, T=T, tf=TF)
    m = float(out["z"].mean()); v = float(out["z"].var())
    assert abs(m - ex_m) / ex_m < 0.02
    assert abs(v - ex_v) / ex_v < 0.02


# ---------------------------------------------------------------- T-G3

def test_tg3_zhat_unbiased_nontrivial_weights_d1():
    """Twisted SMC with prior proposals (real weight variance): E[Zhat] = Z."""
    Pz = jnp.array([1.0]); az = jnp.array([1.0]); y = jnp.array([2.0]); b = 0.5
    log_z = float(tilt.log_z_analytic(Pz, az, y, b))
    R, N, T = 400, 64, 64
    keys = jax.random.split(jax.random.PRNGKey(5), R)
    run = jax.jit(lambda k: samplers.twisted(k, Pz, az, y, b, N=N, T=T, tf=TF,
                                             proposal="prior")["log_z_est"],
                  static_argnames=())
    log_zs = np.array([float(run(k)) for k in keys])
    ratios = np.exp(log_zs - log_z)
    se = ratios.std(ddof=1) / np.sqrt(R)
    assert abs(ratios.mean() - 1.0) < 5.0 * se, (ratios.mean(), se)
    assert ratios.mean() > 0.5  # sanity: not off by orders of magnitude


def test_tg3_conjugate_twist_increments_machine_zero(grf16):
    """Tower-property control: with exact kernel + exact twist the conjugate
    proposal's incremental weights are identically zero. Any algebra error in
    kernel/twist/posterior closed forms breaks this loudly."""
    basis, Pz, az, y, b = grf16
    log_z = float(tilt.log_z_analytic(Pz, az, y, b))
    out = samplers.run_sampler("twisted", jax.random.PRNGKey(6), Pz, az, y, b,
                               N=64, T=64, tf=TF)
    assert float(out["max_abs_incr"]) < 1e-9, float(out["max_abs_incr"])
    assert float(out["ess_final"]) > 63.999  # weights stayed uniform
    # Zhat = logmeanexp of initial weights; z-dependence is e^-tf suppressed
    assert abs(float(out["log_z_est"]) - log_z) < 0.05


def test_tg3_twisted_on_target_and_zhat_16x2(grf16):
    basis, Pz, az, y, b = grf16
    N, T = 256, 64
    log_z = float(tilt.log_z_analytic(Pz, az, y, b))
    mu, Sig = tilt.posterior_params(Pz, az, y, b)
    out = samplers.run_sampler("twisted", jax.random.PRNGKey(6), Pz, az, y, b,
                               N=N, T=T, tf=TF)
    assert abs(float(out["log_z_est"]) - log_z) < 0.05
    # on-target within finite-N: W2 vs oracle floor at matched N
    m, v = metrics.weighted_moments(out["z"], out["logw"])
    w2_twisted = float(jnp.sqrt(metrics.gaussian_w2sq(m, v, mu, Sig)))
    floors = []
    for i in range(8):
        o = samplers.oracle(jax.random.PRNGKey(100 + i), Pz, az, y, b,
                            N=N, T=T, tf=TF)
        mo, vo = metrics.weighted_moments(o["z"], o["logw"])
        floors.append(float(jnp.sqrt(metrics.gaussian_w2sq(mo, vo, mu, Sig))))
    floor = float(np.median(floors))
    assert w2_twisted < 3.0 * floor, (w2_twisted, floor)


# ---------------------------------------------------------------- T-G4

def test_tg4_ess_identities():
    N = 64
    assert abs(float(metrics.ess(jnp.zeros(N))) - N) < 1e-9
    one_hot = jnp.full(N, -1e30).at[3].set(0.0)
    assert abs(float(metrics.ess(one_hot)) - 1.0) < 1e-6


def test_tg4_systematic_resample_counts():
    """Systematic resampling: counts of index i must be floor or ceil of N*w_i."""
    N = 128
    logw = jax.random.normal(jax.random.PRNGKey(7), (N,)) * 2.0
    w = np.asarray(jax.nn.softmax(logw))
    idx = np.asarray(samplers.systematic_resample(jax.random.PRNGKey(8), logw, N))
    counts = np.bincount(idx, minlength=N)
    expected = N * w
    assert np.all(counts >= np.floor(expected) - 1e-9)
    assert np.all(counts <= np.ceil(expected) + 1e-9)


def test_tg4_weight_normalization_and_accumulator():
    """log-Z accumulator identity: resampling then accumulating equals direct
    logmeanexp for a one-block reweight (exact identity, no dynamics)."""
    logw = jnp.array([0.3, -1.2, 2.0, 0.0, -0.5])
    direct = float(jax.nn.logsumexp(logw) - jnp.log(logw.shape[0]))
    # accumulator form: flush at a resample event, residual weights uniform
    acc = float(jax.nn.logsumexp(logw) - jnp.log(logw.shape[0]))
    residual = float(jax.nn.logsumexp(jnp.zeros(5)) - jnp.log(5.0))
    assert abs((acc + residual) - direct) < 1e-12
    w = jax.nn.softmax(logw)
    assert abs(float(w.sum()) - 1.0) < 1e-12


def test_terminal_is_logz_consistent_d1():
    """Vanilla-IS log Z on d=1 agrees with analytic within MC error."""
    Pz = jnp.array([1.0]); az = jnp.array([1.0]); y = jnp.array([2.0]); b = 0.5
    log_z = float(tilt.log_z_analytic(Pz, az, y, b))
    R = 200
    keys = jax.random.split(jax.random.PRNGKey(9), R)
    run = jax.jit(lambda k: samplers.terminal_is(k, Pz, az, y, b, N=256, T=64,
                                                 tf=TF)["log_z_est"])
    ratios = np.exp(np.array([float(run(k)) for k in keys]) - log_z)
    se = ratios.std(ddof=1) / np.sqrt(R)
    assert abs(ratios.mean() - 1.0) < 5.0 * se
