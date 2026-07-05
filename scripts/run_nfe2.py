#!/usr/bin/env python
"""NFE-vs-2NFE doubling check on the amortized-ODE (flow-matching) column —
the JADE-class experiment (P-20260705d).

Per beta=1 gold config: sample the conditional FM net by fixed-step Euler ODE
at NFE and 2*NFE (independent noise seeds), PQMass agreement between the two
sets, plus MMD-to-gold in kappa-z space (the truth column). Floor scale learned
from the k2k bug: report the 90th pct of |disjoint gold-split MMD| as
floor_scale alongside raw mmd2 (no ratio clamps).
Rows -> results/nfe2.jsonl.
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
import numpy as np

from tilt_audit import lognormal, metrics_gold
from tilt_audit.fields import (grid_to_z, make_basis, make_pk, pack,
                               smoothing_operator, unpack)
from tilt_audit.scorenet import UNetCond

N = 256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/fm_cond.pkl")
    p.add_argument("--yseeds", default="0,1,2,3")
    p.add_argument("--nfes", default="8,16,32,64,128,256")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--tag", default="confirmatory_nfe2")
    p.add_argument("--out", default="results/nfe2.jsonl")
    args = p.parse_args()

    from pqm import pqm_pvalue

    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    lam = ck["lam"]
    n = ck["n"]
    model = UNetCond(chs=tuple(ck["chs"]))
    basis = make_basis(n)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    az = jnp.asarray(smoothing_operator(basis))

    @jax.jit
    def ode_sample(key, y_map, nfe):
        x = jax.random.normal(key, (N, n, n), dtype=jnp.float32)
        dt = 1.0 / nfe
        ym = jnp.broadcast_to(y_map, (N, n, n)).astype(jnp.float32)

        def step(x, i):
            t = 1.0 - i * dt
            v = model.apply(ck["ema"], x, ym,
                            jnp.full(N, t, dtype=jnp.float32))
            return x - dt * v, None
        x, _ = jax.lax.scan(step, x, jnp.arange(nfe))
        return x

    out_path = Path(args.out)
    for yseed in [int(v) for v in args.yseeds.split(",")]:
        stem = f"gold_n{n}_bayes_y{yseed}_lam{lam:.4g}_s0"
        fjson = Path("results/gold") / f"{stem}.json"
        if not fjson.exists():
            print(f"[nfe2] {stem} missing, skipped", flush=True)
            continue
        dat = np.load(Path("results/gold") / f"{stem}.npz")
        meta = json.loads(fjson.read_text())
        gold_g = dat["z"].astype(np.float64).reshape(-1, meta["d"])
        y = jnp.asarray(dat["y"].astype(np.float64))
        y_map = unpack(y, basis)
        gold_k = np.asarray(pack(lognormal.kappa(
            unpack(jnp.asarray(gold_g), basis), lam), basis))
        h = metrics_gold.gold_bandwidth(gold_k)
        rng = np.random.default_rng(7)
        floors = []
        for i in range(10):
            perm = rng.permutation(gold_k.shape[0])
            floors.append(metrics_gold.mmd2(gold_k[perm[:N]],
                                            gold_k[perm[N:]], h))
        floor_scale = float(np.quantile(np.abs(floors), 0.9))

        for nfe in [int(v) for v in args.nfes.split(",")]:
            for rep in range(args.reps):
                ka = jax.random.fold_in(jax.random.PRNGKey(rep),
                                        hash((stem, "a", nfe)) & 0x7FFFFFFF)
                kb = jax.random.fold_in(jax.random.PRNGKey(900 + rep),
                                        hash((stem, "b", nfe)) & 0x7FFFFFFF)
                t0 = time.time()
                xa = ode_sample(ka, y_map, nfe)
                xb = ode_sample(kb, y_map, 2 * nfe)
                za = np.asarray(pack(xa.astype(jnp.float64), basis))
                zb = np.asarray(pack(xb.astype(jnp.float64), basis))
                np.random.seed(hash(("nfe2", stem, nfe, rep)) & 0x7FFFFFFF)
                pv = float(pqm_pvalue(za, zb, num_refs=32))
                m = metrics_gold.mmd2(za, gold_k, h)
                with out_path.open("a") as f:
                    f.write(json.dumps(dict(
                        arm="fm_ode", yseed=yseed, nfe=nfe, rep=rep,
                        pqm_p=pv, agree=bool(pv >= 0.05),
                        mmd2_to_gold=float(m), floor_scale=floor_scale,
                        lam=lam, tag=args.tag,
                        wall=round(time.time() - t0, 1),
                        ts=time.strftime("%H:%M:%S"))) + "\n")
            print(f"[nfe2] {stem} NFE={nfe} done", flush=True)


if __name__ == "__main__":
    main()
