#!/usr/bin/env python
"""T1 grid runner: samplers x tilt-strength x dimension x N x seeds, exact scores.

Each run appends one JSON line to --out immediately (crash-safe). The oracle
rows ARE the finite-N floor; analysis joins on (dim, shift, N).

The observation y is drawn once per dimension (held-out key), shared across
all samplers/seeds/N so every row answers the same inference problem.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

from tilt_audit import metrics, misspec, samplers, tilt
from tilt_audit.fields import (band_masks, grid_to_z, make_basis, make_pk,
                               smoothing_operator)

TF = 9.0
S_OBS = 0.5
Y_KEY = 999  # held-out observation key, fixed for the whole night


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dims", default="16,32,64")
    p.add_argument("--shifts", default="0.5,1,2,4")
    p.add_argument("--Ns", default="16,64,256")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--samplers",
                   default="oracle,dps,sap,twisted,terminal_is,exact_guidance")
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--score", default="exact",
                   help="'exact' or 'misspec:<eps>' (analytic contaminated prior)")
    p.add_argument("--tag", default="confirmatory")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dims = [int(v) for v in args.dims.split(",")]
    shifts = [float(v) for v in args.shifts.split(",")]
    Ns = [int(v) for v in args.Ns.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    sampler_names = args.samplers.split(",")
    eps = float(args.score.split(":")[1]) if args.score.startswith("misspec") else 0.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for n in dims:
        basis = make_basis(n)
        pk = make_pk(basis)
        Pz = jnp.asarray(grid_to_z(pk, basis))
        Pz_run = (jnp.asarray(misspec.contaminated_pz(basis, pk, eps))
                  if eps != 0.0 else Pz)
        az = jnp.asarray(smoothing_operator(basis))
        masks = band_masks(basis)
        y, _ = tilt.make_observation(jax.random.PRNGKey(Y_KEY), Pz, az, S_OBS)
        for shift in shifts:
            b = float(tilt.calibrate_b(Pz, az, y, target_shift=shift))
            for N in Ns:
                for seed in seeds:
                    for name in sampler_names:
                        cfg_id = hash((n, shift, N, seed, name)) & 0x7FFFFFFF
                        key = jax.random.fold_in(jax.random.PRNGKey(seed), cfg_id)
                        kw = {}
                        run_name = name
                        if name == "twisted_potential":
                            run_name, kw = "twisted", {"proposal": "prior"}
                        t0 = time.time()
                        # samplers draw with the (possibly contaminated) prior
                        # Pz_run; metrics always score against the TRUE target
                        res = samplers.run_sampler(
                            run_name, key, Pz_run, az, y, b, N=N, T=args.T,
                            tf=TF, **kw)
                        res = {k: jax.device_get(v) for k, v in res.items()}
                        wall = time.time() - t0
                        mkey = jax.random.fold_in(key, 1)
                        row = metrics.evaluate(mkey, res, Pz, az, y, b, basis,
                                               masks)
                        row.update(dim=n, d=n * n, shift=shift, b=b,
                                   beta=b / S_OBS**2, s=S_OBS, N=N, T=args.T,
                                   seed=seed, sampler=name, score=args.score,
                                   eps=eps, tag=args.tag, wall=round(wall, 3),
                                   ts=time.strftime("%H:%M:%S"))
                        with out.open("a") as f:
                            f.write(json.dumps(row) + "\n")
                        print(f"[t1] {n}x{n} shift={shift} N={N} seed={seed} "
                              f"{name}: W2={row['w2']:.4g} KL={row['kl']:.4g} "
                              f"g*={row['gamma_star']:.3g} ({wall:.1f}s)",
                              flush=True)


if __name__ == "__main__":
    main()
