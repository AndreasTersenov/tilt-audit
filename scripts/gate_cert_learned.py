#!/usr/bin/env python
"""Gates G-L1/2 for Stage 2 (docs/PLAN_CERT_LEARNED.md §3)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from tilt_audit import certificate, samplers_learned, tilt
from tilt_audit.fields import grid_to_z, make_basis, make_pk, smoothing_operator

TF = 9.0
T = 256

basis = make_basis(64)
Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
az = jnp.asarray(smoothing_operator(basis))
y, _ = tilt.make_observation(jax.random.PRNGKey(999), Pz, az, 0.5)
b = float(tilt.calibrate_b(Pz, az, y, target_shift=1.0))
x0hat = samplers_learned.make_score_x0hat("analytic", None, basis,
                                          Pz_analytic=Pz)

# G-L1: unguided pathway — step-RN identically zero, logw == r/beta
res = certificate.run_learned_cert("unguided", jax.random.PRNGKey(0), x0hat,
                                   basis, Pz, az, y, b, N=64, T=T, tf=TF)
rn = float(jnp.abs(res["step_rn"]).max())
resid = float(jnp.abs(res["logw"]
                      - tilt.log_reward_over_beta(res["z"], az, y, b)).max())
ok1 = rn == 0.0 and resid < 1e-8
print(f"G-L1 {'PASS' if ok1 else 'FAIL'}: max|step_rn|={rn:.1e}, "
      f"max|logw - r/b|={resid:.1e}")

# G-L2: code-correctness via the one statistic that is unbiased regardless of
# weight degeneracy: sampled mean(logw) must match the exact chain-law E[logw]
# (tests every coefficient of both implementations, both shifts). Per-mode-sum
# ratios are operating characteristics, reported informationally (their
# saturation at ~logN nats/mode is a finding, not a bug — see plan notes).
ok2 = True
info = []
for shift in (0.125, 1.0):
    b_s = float(tilt.calibrate_b(Pz, az, y, target_shift=shift))
    law = certificate.chain_law_ancestral("dps", Pz, az, y, b_s, T, TF)
    means, ratios, clips = [], [], []
    for s_ in range(8):
        r = certificate.run_learned_cert("dps", jax.random.PRNGKey(100 + s_),
                                         x0hat, basis, Pz, az, y, b_s, N=256,
                                         T=T, tf=TF, clip=False)
        means.append(float(np.mean(np.asarray(r["logw"]))))
        ratios.append(certificate.certify_modewise(r)["kl_modes_sum"]
                      / law["kl_path_exact"])
        clips.append(r.get("clip_frac", 0.0))
    mu_hat = float(np.mean(means))
    se = float(np.std(means, ddof=1) / np.sqrt(len(means)))
    ok_mean = abs(mu_hat - law["e_logw"]) < 5 * se
    ok_clip = float(np.median(clips)) < 0.05
    ok2 = ok2 and ok_mean and ok_clip
    info.append(f"shift {shift}: mean(logw) {mu_hat:.4g} vs exact "
                f"{law['e_logw']:.4g} (se {se:.3g}) "
                f"{'OK' if ok_mean else 'MISMATCH'} | per-mode-sum/exact "
                f"{float(np.median(ratios)):.3f} (info) | clip "
                f"{float(np.median(clips)):.3f}")
print(f"G-L2 {'PASS' if ok2 else 'FAIL'}: " + " || ".join(info))
sys.exit(0 if (ok1 and ok2) else 1)