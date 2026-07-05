#!/usr/bin/env python
"""A-lognormal: score-KSD (true score) on the nonlinear-substrate sampler
outputs, calibrated against gold-subsample nulls (run-plan section 2, last Track A arm).

Sampler outputs are REGENERATED with run_transfer's exact cfg-hash seeds (no
stored samples needed). The true score is lognormal.score_g (autodiff through
the forward map). Null: disjoint 256-draw gold subsamples vs the same score
(60 reps). Detection at the empirical one-sided alpha=0.05 as everywhere.
Rows -> results/ksd_trial.jsonl, arm='lognormal'; damage joins to
results/transfer.jsonl on (n, tilt, yseed, sampler, seed) at analysis time.
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

from tilt_audit import ksd, lognormal, samplers_lognormal
from tilt_audit.fields import (grid_to_z, make_basis, make_pk,
                               smoothing_operator)

TF = 9.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--tilts", default="mid,strong")
    p.add_argument("--yseeds", default="0,1,2,3")
    p.add_argument("--samplers",
                   default="dps,dps_inflated,remy5,remy30,remy100,terminal_is")
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--tag", default="confirmatory")
    p.add_argument("--out", default="results/ksd_trial.jsonl")
    args = p.parse_args()

    lam = lognormal.default_lambda()
    basis = make_basis(args.n)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    az = jnp.asarray(smoothing_operator(basis))
    out_path = Path(args.out)
    kernels = {"imq_paper": ("imq", ksd.c_paper(az), -0.5),
               "imq_c1": ("imq", 1.0, -0.5)}

    for tiltname in args.tilts.split(","):
        for yseed in [int(v) for v in args.yseeds.split(",")]:
            stem = f"gold_n{args.n}_{tiltname}_y{yseed}_lam{lam:.4g}_s0"
            fjson = Path("results/gold") / f"{stem}.json"
            if not fjson.exists():
                print(f"[ksd:ln] {stem} missing gold, skipped", flush=True)
                continue
            meta = json.loads(fjson.read_text())
            dat = np.load(Path("results/gold") / f"{stem}.npz")
            gold = dat["z"].astype(np.float64).reshape(-1, meta["d"])
            y = jnp.asarray(dat["y"].astype(np.float64))
            b = meta["b"]

            score_j = jax.jit(jax.vmap(
                lambda zz: lognormal.score_g(zz, Pz, az, basis, lam, y, b)))

            def score_fn(X):
                return np.asarray(score_j(jnp.asarray(X)))

            # nulls: disjoint gold subsamples
            rng = np.random.default_rng(17)
            perm = rng.permutation(gold.shape[0])
            n_null = min(60, gold.shape[0] // args.N - 1)
            nulls = {}
            for kname, (kern, p1, p2) in kernels.items():
                vals = []
                for r in range(n_null):
                    G = gold[perm[r * args.N:(r + 1) * args.N]]
                    vals.append(ksd.ksd_stats(G, score_fn(G), kern, p1, p2)
                                ["score_ksd"])
                nulls[kname] = dict(mean=float(np.mean(vals)),
                                    sd=float(np.std(vals)),
                                    q95=float(np.quantile(vals, 0.95)),
                                    n=n_null)
            print(f"[ksd:ln] {stem}: nulls done "
                  f"(imq_paper q95={nulls['imq_paper']['q95']:.4f})",
                  flush=True)

            for name in args.samplers.split(","):
                K = 0
                sname = name
                if name.startswith("remy"):
                    K = int(name[4:])
                    sname = "remy"
                for seed in [int(v) for v in args.seeds.split(",")]:
                    cfg = hash((args.n, tiltname, yseed, name, args.N,
                                args.T, seed, "transfer")) & 0x7FFFFFFF
                    key = jax.random.fold_in(jax.random.PRNGKey(seed), cfg)
                    res = samplers_lognormal.run_transfer(
                        sname, key, basis, Pz, az, y, b, lam,
                        N=args.N, T=args.T, tf=TF, K=max(K, 1))
                    z = np.asarray(res["z"], dtype=np.float64)
                    lw = np.asarray(res["logw"])
                    if np.ptp(lw) > 1e-9:
                        from tilt_audit.samplers import systematic_resample
                        idx = np.asarray(systematic_resample(
                            jax.random.fold_in(key, 7), jnp.asarray(lw),
                            args.N))
                        z = z[idx]
                    S = score_fn(z)
                    for kname, (kern, p1, p2) in kernels.items():
                        t0 = time.time()
                        st = ksd.ksd_stats(z, S, kern, p1, p2)
                        st = {k: v for k, v in st.items()
                              if k not in ("kernel", "p1", "p2")}
                        nul = nulls[kname]
                        row = dict(arm="lognormal", config=name,
                                   sampler=name, K=K, n=args.n,
                                   tilt=tiltname, yseed=yseed, seed=seed,
                                   lam=lam, b=b, budget=args.N,
                                   kernel=kname, score_mode="true",
                                   detect=bool(st["score_ksd"] > nul["q95"]),
                                   ratio_q95=st["score_ksd"] / nul["q95"],
                                   null_q95=nul["q95"],
                                   null_mean=nul["mean"], tag=args.tag,
                                   wall=round(time.time() - t0, 2),
                                   ts=time.strftime("%H:%M:%S"), **st)
                        with out_path.open("a") as f:
                            f.write(json.dumps(row) + "\n")
                print(f"[ksd:ln] {stem} {name}: done", flush=True)


if __name__ == "__main__":
    main()
