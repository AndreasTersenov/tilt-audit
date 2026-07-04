#!/usr/bin/env python
"""T-M1: mixture-oracle exactness (plan §4.3).

At 20k draws per weight: (1) empirical component weight within the 3-sigma
binomial band of w; (2) per-mode means/variances match the closed-form
mixture moments (z-tests with exact per-mode variances; the same fraction
criteria as T-L1); (3) the score at the exact mean +-dmu points responds to
the offset modes only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from tilt_audit import mixture, tilt
from tilt_audit.fields import grid_to_z, make_basis, make_pk, smoothing_operator

S_OBS = 0.5
Y_KEY = 999
NDRAWS = 20_000


def main():
    ok = True
    basis = make_basis(64)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    dmu = jnp.asarray(mixture.make_offset(Pz, basis))
    dmunp = np.asarray(dmu)
    off_idx = np.where(dmunp != 0)[0]
    sep = float(np.sqrt(np.sum((2 * dmunp) ** 2 / np.asarray(Pz))))
    print(f"[T-M1] offset modes: {off_idx.tolist()}, Mahalanobis "
          f"separation = {sep:.1f} sigma", flush=True)

    for w in (0.5, 0.8):
        zs = np.asarray(mixture.sample(jax.random.PRNGKey(11), dmu, Pz, w,
                                       NDRAWS, "both"), dtype=np.float64)
        # component assignment by sign of projection onto dmu
        proj = zs[:, off_idx] @ dmunp[off_idx]
        w_emp = float((proj > 0).mean())
        se_w = np.sqrt(w * (1 - w) / NDRAWS)
        okw = abs(w_emp - w) < 3 * se_w
        ok &= okw
        print(f"[T-M1] w={w}: empirical weight {w_emp:.4f} "
              f"(3sig band +-{3*se_w:.4f}) {'PASS' if okw else 'FAIL'}",
              flush=True)

        m_ex, v_ex = mixture.moments(dmu, Pz, w)
        m_ex, v_ex = np.asarray(m_ex), np.asarray(v_ex)
        zm = (zs.mean(0) - m_ex) / np.sqrt(v_ex / NDRAWS)
        fz = float(np.mean(np.abs(zm) > 4))
        vr = zs.var(0) / v_ex
        fr = float(np.mean(np.abs(vr - 1) > 0.05))
        okm = fz <= 1e-3 and fr <= 0.02
        ok &= okm
        print(f"[T-M1] w={w}: moments frac|z|>4={fz:.2e} "
              f"frac|vr-1|>5%={fr:.4f} max|z|={np.abs(zm).max():.2f} "
              f"{'PASS' if okm else 'FAIL'}", flush=True)

        # score check: at z=+dmu the responsibility ~1 (12-sigma separation),
        # so score ~ -(z - dmu)/Pz = 0 in offset modes; nonzero at z=0 only
        # through the mixture pull.
        s_plus = np.asarray(mixture.score(dmu[None, :], dmu, Pz, w))[0]
        oks = float(np.abs(s_plus[off_idx]).max()) < 1e-6
        ok &= oks
        print(f"[T-M1] w={w}: score at +dmu offset modes "
              f"max|s|={np.abs(s_plus[off_idx]).max():.2e} "
              f"{'PASS' if oks else 'FAIL'}", flush=True)

    print(f"gate_mixture: {'ALL GREEN' if ok else 'FAILURES PRESENT'}",
          flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
