#!/usr/bin/env python
"""T2: rerun the T1 64^2 rows with a learned (or pathway-control) score.

Joins with t1_core.jsonl oracle/exact rows on (dim, shift, N, seed) for the
scheme-bias / score-error / misspecification / finite-N decomposition.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

from tilt_audit import metrics, samplers_learned, tilt
from tilt_audit.fields import (band_masks, grid_to_z, make_basis, make_pk,
                               smoothing_operator)
from tilt_audit.scorenet import UNet

TF = 9.0
S_OBS = 0.5
Y_KEY = 999


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="path to a train_score.py pickle, or 'analytic'")
    p.add_argument("--label", required=True,
                   help="score label for rows, e.g. learned:clean")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--shifts", default="0.5,1,2,4")
    p.add_argument("--Ns", default="16,64,256")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--samplers", default="dps,sap,twisted,terminal_is")
    p.add_argument("--T", type=int, default=256)
    p.add_argument("--tag", default="confirmatory")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    n = args.n
    basis = make_basis(n)
    pk = make_pk(basis)
    Pz = jnp.asarray(grid_to_z(pk, basis))
    az = jnp.asarray(smoothing_operator(basis))
    masks = band_masks(basis)
    y, _ = tilt.make_observation(jax.random.PRNGKey(Y_KEY), Pz, az, S_OBS)

    if args.ckpt == "analytic":
        x0hat_fn = samplers_learned.make_score_x0hat("analytic", None, basis,
                                                     Pz_analytic=Pz)
        eps_train = 0.0
    else:
        with open(args.ckpt, "rb") as f:
            ckpt = pickle.load(f)
        model = UNet()
        x0hat_fn = samplers_learned.make_score_x0hat(model, ckpt["ema"], basis)
        eps_train = ckpt["eps"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for shift in [float(v) for v in args.shifts.split(",")]:
        b = float(tilt.calibrate_b(Pz, az, y, target_shift=shift))
        for N in [int(v) for v in args.Ns.split(",")]:
            for seed in [int(v) for v in args.seeds.split(",")]:
                for name in args.samplers.split(","):
                    cfg_id = hash((n, shift, N, seed, name, args.label)) & 0x7FFFFFFF
                    key = jax.random.fold_in(jax.random.PRNGKey(seed), cfg_id)
                    t0 = time.time()
                    res = samplers_learned.run_learned(
                        name, key, x0hat_fn, basis, Pz, az, y, b,
                        N=N, T=args.T, tf=TF)
                    wall = time.time() - t0
                    row = metrics.evaluate(jax.random.fold_in(key, 1), res,
                                           Pz, az, y, b, basis, masks)
                    if "clip_frac" in res:
                        row["clip_frac"] = res["clip_frac"]
                    row.update(dim=n, d=n * n, shift=shift, b=b,
                               beta=b / S_OBS**2, s=S_OBS, N=N, T=args.T,
                               seed=seed, sampler=name, score=args.label,
                               eps=eps_train, tag=args.tag,
                               wall=round(wall, 3),
                               ts=time.strftime("%H:%M:%S"))
                    with out.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"[t2:{args.label}] shift={shift} N={N} seed={seed} "
                          f"{name}: W2={row['w2']:.4g} KL={row['kl']:.4g} "
                          f"g*={row['gamma_star']:.3g} ({wall:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
