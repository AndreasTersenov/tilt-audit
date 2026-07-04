#!/usr/bin/env python
"""Track B transfer grids: samplers on the lognormal substrate vs MCMC gold.

Per gold config (matched by --n/--tilt/--yseed): runs the requested samplers
at N/T/seeds, computes metrics_gold vs the gold draws, appends rows to
results/transfer.jsonl. The floor row ('gold_floor') is a disjoint gold
subsample of matched size N. y and b are read FROM the gold sidecar so the
target is bit-identical to what NUTS sampled.
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

from tilt_audit import lognormal, metrics_gold, samplers_lognormal
from tilt_audit.fields import (band_masks, grid_to_z, make_basis, make_pk,
                               smoothing_operator)

TF = 9.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--tilt", required=True)
    p.add_argument("--yseed", type=int, required=True)
    p.add_argument("--samplers",
                   default="dps,dps_inflated,remy5,remy30,remy100,terminal_is")
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--gold-dir", default="results/gold")
    p.add_argument("--tag", default="confirmatory")
    p.add_argument("--out", default="results/transfer.jsonl")
    args = p.parse_args()

    lam = lognormal.default_lambda()
    stem = f"gold_n{args.n}_{args.tilt}_y{args.yseed}_lam{lam:.4g}_s0"
    meta = json.loads((Path(args.gold_dir) / f"{stem}.json").read_text())
    dat = np.load(Path(args.gold_dir) / f"{stem}.npz")
    gold = dat["z"].astype(np.float64).reshape(-1, meta["d"])
    gold_ess = dat["ess"].astype(np.float64)
    y = jnp.asarray(dat["y"].astype(np.float64))
    b = meta["b"]

    basis = make_basis(args.n)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    az = jnp.asarray(smoothing_operator(basis))
    masks = [np.asarray(m) for m in band_masks(basis)]
    h = metrics_gold.gold_bandwidth(gold)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = dict(n=args.n, d=meta["d"], tilt=args.tilt, shift=meta["shift"],
                yseed=args.yseed, lam=lam, b=b, N=args.N, T=args.T,
                gold_draws=gold.shape[0], tag=args.tag)

    def emit(row):
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    # floor rows: disjoint gold subsamples of matched size
    rng = np.random.default_rng(123)
    perm = rng.permutation(gold.shape[0])
    n_floor = min(len(args.seeds.split(",")), gold.shape[0] // args.N - 1)
    for s in range(n_floor):
        sub = gold[perm[s * args.N:(s + 1) * args.N]]
        rest = np.delete(gold, perm[s * args.N:(s + 1) * args.N], axis=0)
        m = metrics_gold.evaluate_vs_gold(sub, rest, gold_ess, masks, h)
        emit(dict(base, sampler="gold_floor", seed=s,
                  ts=time.strftime("%H:%M:%S"), **m))
    print(f"[transfer:{stem}] {n_floor} floor rows", flush=True)

    for name in args.samplers.split(","):
        K = 0
        sname = name
        if name.startswith("remy"):
            K = int(name[4:])
            sname = "remy"
        for seed in [int(v) for v in args.seeds.split(",")]:
            cfg = hash((args.n, args.tilt, args.yseed, name, args.N,
                        args.T, seed, "transfer")) & 0x7FFFFFFF
            key = jax.random.fold_in(jax.random.PRNGKey(seed), cfg)
            t0 = time.time()
            res = samplers_lognormal.run_transfer(
                sname, key, basis, Pz, az, y, b, lam,
                N=args.N, T=args.T, tf=TF, K=max(K, 1))
            wall = time.time() - t0
            z = np.asarray(res["z"], dtype=np.float64)
            logw = np.asarray(res["logw"])
            if np.ptp(logw) > 1e-9:  # weighted arm: resample to unweighted
                from tilt_audit.samplers import systematic_resample
                idx = np.asarray(systematic_resample(
                    jax.random.fold_in(key, 7), jnp.asarray(logw), args.N))
                z = z[idx]
            m = metrics_gold.evaluate_vs_gold(z, gold, gold_ess, masks, h)
            row = dict(base, sampler=name, K=K, seed=seed,
                       wall=round(wall, 2), ts=time.strftime("%H:%M:%S"),
                       clip_frac=res.get("clip_frac", 0.0),
                       ess_final=res.get("ess_final", float(args.N)), **m)
            emit(row)
            print(f"[transfer:{stem}] {name} s={seed}: "
                  f"mmd2={m['mmd2']:.3e} swd2={m['swd2']:.3e} "
                  f"bp_cov=({m['bp0_cov68']:.2f},{m['bp1_cov68']:.2f},"
                  f"{m['bp2_cov68']:.2f}) ({wall:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
