#!/usr/bin/env python
"""A2 failure archives: z-space sample banks with exactly known damage.

Two modes:

--mode archive (unconditional, the PQMass food): at 64^2, pinned T1
  observation (Y_KEY=999), shift-calibrated b; 4096 samples per config built
  from 16 particle-seed batches of N=256. Weighted samplers (twisted) are
  systematically resampled to unweighted draws first — that is what a
  practitioner would feed a sample-based test; ess_final recorded in meta.
  Configs: oracle_null + oracle_ref (independent oracle banks: the
  same-vs-same null pair), dps, sap, twisted at 1sigma; dps at 0.5/2 sigma
  (strength axis); dps@eps=+-0.3 and twisted@eps=-0.3 (misspec axis;
  dps@eps=-0.3 is THE accidental-compensation config, W2 ~6x floor with
  gamma*~1).

--mode conditional (the TARP/MIRA food): L fresh observations y_l (y_seeds
  2000..2000+L-1) with their generating truths z*_l, S=N posterior samples
  per y. Run at b = S_OBS^2 exactly (beta=1, the Bayes point): that is the
  ONLY configuration where truths are exchangeable draws from the oracle's
  target, i.e. where TARP's nominal coverage and MIRA's 2/3 null are
  well-defined. (The plan suggested reusing A1's shift-calibrated draws;
  that would make even the oracle read as miscalibrated — deviation logged
  in NIGHT_LOG_2026-07-04.md.) Per-y rms shift recorded in meta.

Output: results/archives/{name}.npz (z fp32) + sidecar meta in the npz.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from tilt_audit import misspec, samplers, tilt
from tilt_audit.fields import grid_to_z, make_basis, make_pk, smoothing_operator
from tilt_audit.samplers import systematic_resample

TF = 9.0
S_OBS = 0.5
Y_KEY = 999

# name -> (sampler, shift, eps)
ARCHIVE_CONFIGS = {
    "oracle_null": ("oracle", 1.0, 0.0),
    "oracle_ref": ("oracle", 1.0, 0.0),
    "dps": ("dps", 1.0, 0.0),
    "sap": ("sap", 1.0, 0.0),
    "twisted": ("twisted", 1.0, 0.0),
    "dps_em03": ("dps", 1.0, -0.3),
    "dps_ep03": ("dps", 1.0, +0.3),
    "twisted_em03": ("twisted", 1.0, -0.3),
    "dps_s05": ("dps", 0.5, 0.0),
    "dps_s2": ("dps", 2.0, 0.0),
}

# conditional mode: name -> (sampler, eps); b = S_OBS^2 always
COND_CONFIGS = {
    "oracle": ("oracle", 0.0),
    "dps": ("dps", 0.0),
    "sap": ("sap", 0.0),
    "twisted": ("twisted", 0.0),
    "dps_em03": ("dps", -0.3),
    "dps_ep03": ("dps", +0.3),
}


def unweight(res, key, N):
    logw = np.asarray(res["logw"])
    if np.ptp(logw) > 1e-9:
        idx = np.asarray(systematic_resample(key, jnp.asarray(logw), N))
        return np.asarray(res["z"])[idx]
    return np.asarray(res["z"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["archive", "conditional"], required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--batches", type=int, default=16,
                   help="archive mode: particle-seed batches (16 x 256 = 4096)")
    p.add_argument("--L", type=int, default=128,
                   help="conditional mode: number of observations")
    p.add_argument("--configs", default="",
                   help="comma subset; empty = all for the mode")
    p.add_argument("--outdir", default="results/archives")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    basis = make_basis(args.n)
    pk = make_pk(basis)
    Pz = jnp.asarray(grid_to_z(pk, basis))
    az = jnp.asarray(smoothing_operator(basis))

    if args.mode == "archive":
        configs = {k: v for k, v in ARCHIVE_CONFIGS.items()
                   if not args.configs or k in args.configs.split(",")}
        y, _ = tilt.make_observation(jax.random.PRNGKey(Y_KEY), Pz, az, S_OBS)
        b_cache = {}
        for name, (sampler, shift, eps) in configs.items():
            fout = outdir / f"{name}.npz"
            if fout.exists():
                print(f"[a2-arch] {name} exists, skipping", flush=True)
                continue
            if shift not in b_cache:
                b_cache[shift] = float(tilt.calibrate_b(Pz, az, y,
                                                        target_shift=shift))
            b = b_cache[shift]
            Pz_run = (jnp.asarray(misspec.contaminated_pz(basis, pk, eps))
                      if eps != 0.0 else Pz)
            # oracle_ref gets a disjoint seed block from oracle_null
            seed0 = 1600 if name == "oracle_ref" else 0
            t0 = time.time()
            chunks, ess = [], []
            for s in range(seed0, seed0 + args.batches):
                key = jax.random.fold_in(
                    jax.random.PRNGKey(s),
                    hash((args.n, name, args.N, args.T)) & 0x7FFFFFFF)
                res = samplers.run_sampler(sampler, key, Pz_run, az, y, b,
                                           N=args.N, T=args.T, tf=TF)
                res = {k: jax.device_get(v) for k, v in res.items()}
                ess.append(float(res.get("ess_final",
                                          np.asarray(args.N, dtype=float))))
                chunks.append(unweight(res, jax.random.fold_in(key, 7),
                                       args.N).astype(np.float32))
            z = np.concatenate(chunks, 0)
            meta = dict(mode="archive", sampler=sampler, shift=shift, b=b,
                        eps=eps, n=args.n, N=args.N, T=args.T,
                        batches=args.batches, seed0=seed0, y_seed=Y_KEY,
                        s_obs=S_OBS, ess_final_mean=float(np.mean(ess)))
            np.savez(fout, z=z, meta=json.dumps(meta))
            print(f"[a2-arch] {name}: {z.shape} in {time.time()-t0:.0f}s "
                  f"(b={b:.4g})", flush=True)

    else:  # conditional
        configs = {k: v for k, v in COND_CONFIGS.items()
                   if not args.configs or k in args.configs.split(",")}
        b = S_OBS**2  # beta = 1: the Bayes point (see module docstring)
        for name, (sampler, eps) in configs.items():
            fout = outdir / f"cond_{name}.npz"
            if fout.exists():
                print(f"[a2-cond] {name} exists, skipping", flush=True)
                continue
            Pz_run = (jnp.asarray(misspec.contaminated_pz(basis, pk, eps))
                      if eps != 0.0 else Pz)
            t0 = time.time()
            truths, ys, samps, shifts = [], [], [], []
            for l in range(args.L):
                yk = jax.random.PRNGKey(2000 + l)
                y_l, z_star = tilt.make_observation(yk, Pz, az, S_OBS)
                key = jax.random.fold_in(
                    jax.random.PRNGKey(l),
                    hash((args.n, "cond", name, args.N, args.T)) & 0x7FFFFFFF)
                res = samplers.run_sampler(sampler, key, Pz_run, az, y_l, b,
                                           N=args.N, T=args.T, tf=TF)
                res = {k: jax.device_get(v) for k, v in res.items()}
                truths.append(np.asarray(z_star, dtype=np.float32))
                ys.append(np.asarray(y_l, dtype=np.float32))
                samps.append(unweight(res, jax.random.fold_in(key, 7),
                                      args.N).astype(np.float32))
                shifts.append(float(tilt.rms_shift(Pz, az, y_l, b)))
            meta = dict(mode="conditional", sampler=sampler, b=b, eps=eps,
                        n=args.n, N=args.N, T=args.T, L=args.L,
                        y_seed0=2000, s_obs=S_OBS,
                        rms_shift_mean=float(np.mean(shifts)))
            np.savez(fout, truths=np.stack(truths), ys=np.stack(ys),
                     samples=np.stack(samps), meta=json.dumps(meta))
            print(f"[a2-cond] {name}: samples {np.stack(samps).shape} in "
                  f"{time.time()-t0:.0f}s (shift~{np.mean(shifts):.2f})",
                  flush=True)


if __name__ == "__main__":
    main()
