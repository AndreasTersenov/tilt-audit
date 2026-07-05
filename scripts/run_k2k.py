#!/usr/bin/env python
"""The K-vs-2K self-consistency certificate: the one-day box (P-20260705b/c).

Protocol under test (truth-free, runtime): run the sampler at budget K and at
2K with independent seeds, compare the two sample sets with PQMass, declare
"converged" if the test reads null (pqm p-value >= 0.05 — PQMass's own null,
whose FP rates we validated across the arms-night battery).

Three arms, all against tonight's gold standards so every certificate verdict
gets a truth column:

  remy   K in {10,15,25,50,100}: remy@K vs remy@2K (indep seeds), plus
         MMD-to-gold of the K run — does the certificate's verdict track the
         true convergence? (P-20260705b)
  dps    T in {64,128,256,512}: dps@T vs dps@2T — the false-certification
         exhibit: agreement while MMD-to-gold stays >>floor (P-20260705c);
         valid-scope boundary of the certificate.
  stuck  two independent single-mode samplers on the 50/50 mixture — the
         inherited R-hat blind spot, demonstrated (construction, not
         prediction).

Rows -> results/k2k.jsonl (append-mode).
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

from tilt_audit import lognormal, metrics_gold, mixture, samplers_lognormal
from tilt_audit.fields import (band_masks, grid_to_z, make_basis, make_pk,
                               smoothing_operator)

TF = 9.0
N = 256


def num_refs_for(budget):
    return max(16, min(100, budget // 8))


def load_gold(n, tiltname, yseed, lam):
    stem = f"gold_n{n}_{tiltname}_y{yseed}_lam{lam:.4g}_s0"
    meta = json.loads((Path("results/gold") / f"{stem}.json").read_text())
    dat = np.load(Path("results/gold") / f"{stem}.npz")
    gold = dat["z"].astype(np.float64).reshape(-1, meta["d"])
    y = jnp.asarray(dat["y"].astype(np.float64))
    return gold, y, meta["b"], stem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="remy,dps,stuck")
    p.add_argument("--tilts", default="mid,strong")
    p.add_argument("--Ks", default="10,15,25,50,100")
    p.add_argument("--Ts", default="64,128,256,512")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--tag", default="confirmatory_k2k")
    p.add_argument("--out", default="results/k2k.jsonl")
    args = p.parse_args()

    from pqm import pqm_pvalue

    lam = lognormal.default_lambda()
    n = 64
    basis = make_basis(n)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    az = jnp.asarray(smoothing_operator(basis))
    masks = [np.asarray(m) for m in band_masks(basis)]
    out_path = Path(args.out)

    def emit(row):
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def run(sname, key, y, b, T, K):
        res = samplers_lognormal.run_transfer(
            sname, key, basis, Pz, az, y, b, lam, N=N, T=T, tf=TF,
            K=max(K, 1))
        return np.asarray(res["z"], dtype=np.float64)

    def pqm_p(xa, xb, seed):
        np.random.seed(seed & 0x7FFFFFFF)
        return float(pqm_pvalue(xa, xb, num_refs=num_refs_for(N)))

    arms = args.arms.split(",")

    if "remy" in arms or "dps" in arms:
        for tiltname in args.tilts.split(","):
            gold, y, b, stem = load_gold(n, tiltname, 0, lam)
            h = metrics_gold.gold_bandwidth(gold)
            floor = np.median([metrics_gold.mmd2(
                gold[i * N:(i + 1) * N], gold[(i + 1) * N:], h)
                for i in range(5)])

            if "remy" in arms:
                for K in [int(v) for v in args.Ks.split(",")]:
                    for rep in range(args.reps):
                        ka = jax.random.fold_in(jax.random.PRNGKey(rep),
                                                hash((stem, "a", K))
                                                & 0x7FFFFFFF)
                        kb = jax.random.fold_in(jax.random.PRNGKey(500 + rep),
                                                hash((stem, "b", K))
                                                & 0x7FFFFFFF)
                        t0 = time.time()
                        za = run("remy", ka, y, b, 64, K)
                        zb = run("remy", kb, y, b, 64, 2 * K)
                        pv = pqm_p(za, zb, hash(("k2k", stem, K, rep)))
                        m = metrics_gold.mmd2(za, gold, h)
                        emit(dict(arm="remy", tilt=tiltname, K=K, rep=rep,
                                  pqm_p=pv, agree=bool(pv >= 0.05),
                                  mmd2_to_gold=m, mmd2_floor=float(floor),
                                  ratio_floor=float(m / max(floor, 1e-12)),
                                  tag=args.tag,
                                  wall=round(time.time() - t0, 1),
                                  ts=time.strftime("%H:%M:%S")))
                    print(f"[k2k:remy] {tiltname} K={K} done", flush=True)

            if "dps" in arms:
                for T in [int(v) for v in args.Ts.split(",")]:
                    for rep in range(args.reps):
                        ka = jax.random.fold_in(jax.random.PRNGKey(rep),
                                                hash((stem, "da", T))
                                                & 0x7FFFFFFF)
                        kb = jax.random.fold_in(jax.random.PRNGKey(500 + rep),
                                                hash((stem, "db", T))
                                                & 0x7FFFFFFF)
                        t0 = time.time()
                        za = run("dps", ka, y, b, T, 1)
                        zb = run("dps", kb, y, b, 2 * T, 1)
                        pv = pqm_p(za, zb, hash(("k2kd", stem, T, rep)))
                        m = metrics_gold.mmd2(za, gold, h)
                        emit(dict(arm="dps", tilt=tiltname, T=T, rep=rep,
                                  pqm_p=pv, agree=bool(pv >= 0.05),
                                  mmd2_to_gold=m, mmd2_floor=float(floor),
                                  ratio_floor=float(m / max(floor, 1e-12)),
                                  tag=args.tag,
                                  wall=round(time.time() - t0, 1),
                                  ts=time.strftime("%H:%M:%S")))
                    print(f"[k2k:dps] {tiltname} T={T} done", flush=True)

    if "stuck" in arms:
        dmu = jnp.asarray(mixture.make_offset(Pz, basis))
        for rep in range(5):
            ka = jax.random.fold_in(jax.random.PRNGKey(rep), 41)
            kb = jax.random.fold_in(jax.random.PRNGKey(500 + rep), 42)
            za = np.asarray(mixture.sample(ka, dmu, Pz, 0.5, N, "plus"),
                            dtype=np.float64)
            zb = np.asarray(mixture.sample(kb, dmu, Pz, 0.5, N, "plus"),
                            dtype=np.float64)
            pv = pqm_p(za, zb, hash(("k2ks", rep)))
            emit(dict(arm="stuck", w=0.5, rep=rep, pqm_p=pv,
                      agree=bool(pv >= 0.05), true_state="half missing",
                      tag=args.tag, ts=time.strftime("%H:%M:%S")))
        print("[k2k:stuck] demo done", flush=True)


if __name__ == "__main__":
    main()
