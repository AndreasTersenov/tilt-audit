#!/usr/bin/env python
"""Gate T-N2: Remy-scheme reduction anchor at 16^2 / 1 sigma / N=256.

(1) inflation='exact' (each level targets exactly p(x_t|y)) + K=100 must land
    within 1.5x the oracle finite-N floor — validates the whole annealing/
    Langevin machinery against the closed form.
(2) With their sigma_t^2 inflation, W2 must decrease monotonically in K
    (median over 3 seeds per K) — the t->0 target is the true posterior, so
    more equilibration must help.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from tilt_audit import metrics, samplers, tilt
from tilt_audit.fields import grid_to_z, make_basis, make_pk, smoothing_operator

TF = 9.0
S_OBS = 0.5
Y_KEY = 999
N, T = 256, 64
KS = [1, 2, 5, 10, 30, 100]


def w2(res, mu, Sig):
    m, v = metrics.weighted_moments(res["z"], res["logw"])
    return float(jnp.sqrt(metrics.gaussian_w2sq(m, v, mu, Sig)))


def main():
    basis = make_basis(16)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    az = jnp.asarray(smoothing_operator(basis))
    y, _ = tilt.make_observation(jax.random.PRNGKey(Y_KEY), Pz, az, S_OBS)
    b = float(tilt.calibrate_b(Pz, az, y, target_shift=1.0))
    mu, Sig = tilt.posterior_params(Pz, az, y, b)

    floors = []
    for i in range(8):
        floors.append(w2(samplers.oracle(jax.random.PRNGKey(100 + i), Pz, az,
                                         y, b, N=N, T=T, tf=TF), mu, Sig))
    floor = float(np.median(floors))

    exact = w2(samplers.run_sampler("remy", jax.random.PRNGKey(1), Pz, az, y,
                                    b, N=N, T=T, tf=TF, K=100,
                                    inflation="exact"), mu, Sig)
    ok1 = exact < 1.5 * floor

    med = []
    for K in KS:
        vals = [w2(samplers.run_sampler("remy", jax.random.PRNGKey(10 + s),
                                        Pz, az, y, b, N=N, T=T, tf=TF, K=K,
                                        inflation="sigma2"), mu, Sig)
                for s in range(3)]
        med.append(float(np.median(vals)))
    ok2 = all(med[i + 1] <= med[i] for i in range(len(med) - 1))

    print(f"T-N2 floor={floor:.4g}; exact-inflation K=100: W2={exact:.4g} "
          f"({exact/floor:.2f}x floor, need <1.5x) -> "
          f"{'PASS' if ok1 else 'FAIL'}")
    print(f"T-N2 sigma2-inflation W2(K) medians over seeds "
          f"{dict(zip(KS, [round(v, 4) for v in med]))} monotone: "
          f"{'PASS' if ok2 else 'FAIL'}")
    sys.exit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
