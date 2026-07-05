#!/usr/bin/env python
"""Diagnostic B — the likelihood posterior-predictive residual (P-20260705h).

The one piece of ground truth available at RUNTIME is the observation y and the
known forward model (A, s): no score, no gold draws needed. For a correct
posterior sample x0, the standardized residual

    R(x0) = || A x0 - y ||^2 / s^2

is a sum of d_obs independent standard-normal squares  =>  R ~ chi^2(d_obs)
under the true posterior predictive (each observed mode k contributes
(a_k x0_k - y_k)^2 / s^2, and marginally a_k x0_k - y_k ~ N(0, s^2) when x0 ~ sigma).
So two ground-truth-free checks on a bank of sampler draws {x0^(i)}:

  chi2 test : is mean_i R(x0^i) consistent with d_obs (a sampler that
              over-fits y reads LOW, one that under-fits reads HIGH)?
  whiteness : are the per-mode standardized residuals r_k = (a_k x0_k - y_k)/s
              marginally N(0,1) with unit variance (over the bank)?

Barrier this attacks: reference confounding (barrier 3) — it uses the data, not
the (possibly wrong) score. Predicted operating envelope (P-h): catches
guidance-STRENGTH bias (dps over/under-fit), BLIND to failures living in the
null space of A (missed modes, prior-collapse in unconstrained directions),
because those fit y equally well.

Detection is calibrated against the oracle's own null: a two-sided z-test on
the bank-mean residual, |mean(R)/d_obs - 1| vs the oracle's spread at matched N.
Rows -> results/ppc.jsonl (append; tag pilot|grid).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tilt_audit import samplers, tilt
from tilt_audit.fields import grid_to_z, make_basis, make_pk, smoothing_operator

S_OBS = 0.5
Y_KEY = 999
SHIFTS = {"mid": 1.0, "strong": 2.0}
OUT_DEFAULT = "results/ppc.jsonl"


def residual_stats(z0, az, y, s):
    """R = ||A z0 - y||^2 / s^2 per draw; per-mode standardized residual r_k."""
    r = (az[None, :] * z0 - y[None, :]) / s          # (N, d)
    R = jnp.sum(r**2, axis=1)                          # (N,)
    return np.asarray(R), np.asarray(r)


def null_band(key, Pz, az, y, b, s, N, reps=200):
    """Oracle-null distribution of mean(R)/d over `reps` independent banks."""
    keys = jax.random.split(key, reps)
    d = Pz.shape[0]
    vals = []
    for k in keys:
        out = samplers.oracle(k, Pz, az, y, b, N=N, T=0, tf=9.0)
        R, _ = residual_stats(out["z"], az, y, s)
        vals.append(R.mean() / d)
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std(ddof=1))


def nullspace_collapse(key, Pz, az, y, b, N):
    """A constructed failure that fits y EXACTLY but collapses variance in the
    null space of A (the a_k = 0 modes): posterior mean on observed modes,
    near-zero draws on unobserved ones. PPC should be BLIND to this (P-h)."""
    mu, Sig = tilt.posterior_params(Pz, az, y, b)
    obs = np.asarray(az) > 1e-8
    z = np.asarray(mu)[None, :] + np.zeros((N, Pz.shape[0]))
    # correct posterior spread on observed modes, collapsed on null-space modes
    key = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31)))
    noise = key.standard_normal((N, Pz.shape[0])) * np.sqrt(np.asarray(Sig))
    z[:, obs] += noise[:, obs]
    return jnp.asarray(z)


def run(n, tilts, guidance_ladder, N, seeds, tag, out):
    basis = make_basis(n)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    az = jnp.asarray(smoothing_operator(basis))
    d = int(Pz.shape[0])
    d_obs = int(np.sum(np.asarray(az) > 1e-8))
    fout = open(out, "a")

    for tilt_name in tilts:
        y, _ = tilt.make_observation(jax.random.PRNGKey(Y_KEY), Pz, az, S_OBS)
        b = tilt.calibrate_b(Pz, az, y, target_shift=SHIFTS[tilt_name])
        nb_mean, nb_std = null_band(jax.random.PRNGKey(7), Pz, az, y, b, S_OBS, N)

        def emit(sampler_name, z0, extra):
            R, r = residual_stats(z0, az, y, S_OBS)
            stat = R.mean() / d
            z = (stat - nb_mean) / (nb_std + 1e-12)
            row = {"diag": "ppc", "tag": tag, "n": n, "d": d, "d_obs": d_obs,
                   "tilt": tilt_name, "sampler": sampler_name, "N": N,
                   "mean_R_over_d": float(stat), "null_mean": nb_mean,
                   "null_std": nb_std, "z": float(z),
                   "detect": bool(abs(z) > 2.0),
                   "whiteness_var": float(np.var(r)), **extra}
            fout.write(json.dumps(row) + "\n"); fout.flush()
            print(f"  {tilt_name:>6} {sampler_name:>16}: mean(R)/d={stat:.3f} "
                  f"z={z:+.1f} detect={row['detect']} whitevar={row['whiteness_var']:.2f}")

        # oracle control (should NOT detect)
        o = samplers.oracle(jax.random.PRNGKey(100), Pz, az, y, b, N=N, T=0, tf=9.0)
        emit("oracle", o["z"], {"true_fail": False})

        # guidance-strength ladder on dps (should detect the over/under-fit)
        for g in guidance_ladder:
            out_s = samplers.run_sampler("dps", jax.random.PRNGKey(200 + int(g * 10)),
                                         Pz, az, y, b * g, N=N, T=64, tf=9.0)
            emit(f"dps_g{g:g}", out_s["z"], {"true_fail": True, "gscale": g})

        # null-space collapse (fits y; PPC predicted BLIND)
        z_ns = nullspace_collapse(jax.random.PRNGKey(300), Pz, az, y, b, N)
        emit("nullspace_collapse", z_ns, {"true_fail": True, "class": "nullspace"})

    fout.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tilts", default="mid,strong")
    ap.add_argument("--guidance-ladder", default="0.5,1.0,2.0",
                    help="b-scale multipliers for dps (1.0 = calibrated)")
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    run(args.n, args.tilts.split(","),
        [float(x) for x in args.guidance_ladder.split(",")],
        args.N, [int(s) for s in args.seeds.split(",")], args.tag, args.out)


if __name__ == "__main__":
    main()
