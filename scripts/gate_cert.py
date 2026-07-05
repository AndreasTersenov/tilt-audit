#!/usr/bin/env python
"""Gates G-C1..3 for the certificate kill test (the run run-plan section 2.3).

G-C1  plumbing at machine zero: unguided mode's step-RN identically 0 and
      logw identically r(z0)/beta (the terminal_is identity).
G-C2  exactness anchors at 16^2/1sigma/N=256: (a) exact_guidance repaired W2
      within 1.5x oracle floor; (b) the data-processing inequality
      kl_path_exact >= kl_end_exact holds for dps AND exact_guidance at all
      four shifts (closed-form both sides); (c) the sampled estimator KL-hat
      lands within [0.2, 1.5]x of the exact path KL (median over 8 seeds).
G-C3  Z-hat unbiasedness: dps 16^2/0.5sigma, 32 seeds, mean(Zhat/Z) within
      5 s.e. of 1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from tilt_audit import certificate, metrics, samplers, tilt
from tilt_audit.fields import grid_to_z, make_basis, make_pk, smoothing_operator

TF = 9.0
S_OBS = 0.5
Y_KEY = 999
T = 64


def setup(n=16, shift=1.0):
    basis = make_basis(n)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    az = jnp.asarray(smoothing_operator(basis))
    y, _ = tilt.make_observation(jax.random.PRNGKey(Y_KEY), Pz, az, S_OBS)
    b = float(tilt.calibrate_b(Pz, az, y, target_shift=shift))
    return Pz, az, y, b


def gc1():
    Pz, az, y, b = setup()
    out = certificate.run_cert("unguided", jax.random.PRNGKey(0), Pz, az, y,
                               b, N=64, T=T, tf=TF)
    rn = float(jnp.abs(out["step_rn"]).max())
    resid = float(jnp.abs(out["logw"]
                          - tilt.log_reward_over_beta(out["z"], az, y, b)).max())
    ok = rn < 1e-8 and resid < 1e-8
    print(f"G-C1 {'PASS' if ok else 'FAIL'}: max|step_rn|={rn:.2e}, "
          f"max|logw - r/b|={resid:.2e} (need <1e-8)")
    return ok


def gc2():
    """Respecced per run-plan section 7: DPI everywhere + tightness anchor on the good
    sampler + estimator fidelity in a healthy-weights regime (the only regime
    where any estimator can have it)."""
    # (a) data-processing inequality, closed form both sides, all shifts
    ok_a = True
    for shift in (0.5, 1.0, 2.0, 4.0):
        Pz_, az_, y_, b_ = setup(shift=shift)
        for mode in ("dps", "exact_guidance"):
            law = certificate.chain_law(mode, Pz_, az_, y_, b_, T, TF)
            end = certificate.endpoint_errors(law, Pz_, az_, y_, b_)
            if law["kl_path_exact"] < end["kl_end_exact"] - 1e-6:
                ok_a = False
                print(f"  DPI VIOLATION {mode} shift={shift}: "
                      f"path {law['kl_path_exact']:.4g} < "
                      f"end {end['kl_end_exact']:.4g}")
    # (b) tightness anchor: the certificate prices the good sampler tightly
    Pz, az, y, b = setup(shift=1.0)
    law_eg = certificate.chain_law("exact_guidance", Pz, az, y, b, T, TF)
    end_eg = certificate.endpoint_errors(law_eg, Pz, az, y, b)
    tight = law_eg["kl_path_exact"] / end_eg["kl_end_exact"]
    ok_b = 1.0 - 1e-9 <= tight <= 1.3
    # (c) estimator fidelity where weights are healthy (eg at 0.25 sigma)
    Pz_, az_, y_, b_ = setup(shift=0.25)
    law = certificate.chain_law("exact_guidance", Pz_, az_, y_, b_, T, TF)
    ratios = []
    for s in range(8):
        o = certificate.run_cert("exact_guidance", jax.random.PRNGKey(200 + s),
                                 Pz_, az_, y_, b_, N=256, T=T, tf=TF)
        ratios.append(certificate.certify(o)["kl_path_hat"]
                      / law["kl_path_exact"])
    med = float(np.median(ratios))
    ok_c = 0.5 <= med <= 1.5
    ok = ok_a and ok_b and ok_c
    print(f"G-C2 {'PASS' if ok else 'FAIL'}: DPI "
          f"{'holds all shifts' if ok_a else 'VIOLATED'} | eg tightness "
          f"path/end = {tight:.3f} (need <=1.3) | KLhat/exact median {med:.3f}"
          f" at eg/0.25sigma (exact {law['kl_path_exact']:.3g} nats; need in "
          f"[0.5, 1.5])")
    return ok


def gc3():
    """Zhat unbiasedness, verifiable only where path-KL is a few nats:
    exact_guidance at 0.25 sigma."""
    Pz, az, y, b = setup(shift=0.25)
    log_z = float(tilt.log_z_analytic(Pz, az, y, b))
    law = certificate.chain_law("exact_guidance", Pz, az, y, b, T, TF)
    ratios = []
    for s in range(32):
        o = certificate.run_cert("exact_guidance", jax.random.PRNGKey(300 + s),
                                 Pz, az, y, b, N=256, T=T, tf=TF)
        ratios.append(np.exp(float(o["log_z_est"]) - log_z))
    ratios = np.asarray(ratios)
    se = ratios.std(ddof=1) / np.sqrt(len(ratios))
    ok = abs(ratios.mean() - 1.0) < 5.0 * se and ratios.mean() > 0.3
    print(f"G-C3 {'PASS' if ok else 'FAIL'}: mean(Zhat/Z) = "
          f"{ratios.mean():.3f} +- {se:.3f} at eg/0.25sigma (exact path KL "
          f"{law['kl_path_exact']:.3g}; need within 5 s.e. of 1)")
    return ok


if __name__ == "__main__":
    oks = [gc1(), gc2(), gc3()]
    print("gate_cert:", "ALL GREEN" if all(oks) else "FAILURES PRESENT")
    sys.exit(0 if all(oks) else 1)
