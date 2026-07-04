#!/usr/bin/env python
"""Learned-score transfer columns: ln_clean / ln_mis nets driving DPS on the
lognormal substrate (plan §3, after trainings land).

Construction: the nets model the KAPPA-field prior, so the sampler state is
the packed kappa field z_k and the observation is LINEAR there
(y = az * z_k + n) — the nonlinearity lives inside the learned prior. We
reuse samplers_learned.run_learned('dps') verbatim with:
  - x0hat from the net (make_score_x0hat),
  - Pz = EMPIRICAL per-mode variance of packed kappa draws (32k draws,
    <1% MC error) — used only for the init marginal and the analytic
    preconditioner, both approximation-tolerant,
  - y, b from the gold sidecar (identical target).
Comparison is in kappa-z space against pack(kappa(unpack(gold_g))). mode_z
uses the g-space median ESS as a scalar proxy (flagged mode_z_proxy) — the
ESS-free metrics (mmd2, swd2, bp_cov) are the headline columns.
Rows -> results/transfer.jsonl, sampler='dps_<net>', space='kappa_z'.
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

from tilt_audit import lognormal, metrics_gold, samplers_learned
from tilt_audit.fields import (band_masks, grid_to_z, make_basis, make_pk,
                               pack, smoothing_operator, unpack)
from tilt_audit.scorenet import UNet

TF = 9.0


def kappa_pz_empirical(key, Pz, basis, lam, n_draws=32768, chunk=4096):
    acc = None
    n = 0
    for i in range(n_draws // chunk):
        k = jax.random.fold_in(key, i)
        g = jnp.sqrt(Pz) * jax.random.normal(k, (chunk, Pz.shape[0]))
        zk = pack(lognormal.kappa(unpack(g, basis), lam), basis)
        s = jnp.sum(zk**2, axis=0)
        acc = s if acc is None else acc + s
        n += chunk
    return np.asarray(acc / n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--nets", default="ln_clean,ln_mis")
    p.add_argument("--tilts", default="mid,strong")
    p.add_argument("--yseeds", default="0,1,2,3")
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--tag", default="confirmatory")
    p.add_argument("--out", default="results/transfer.jsonl")
    args = p.parse_args()

    lam = lognormal.default_lambda()
    basis = make_basis(args.n)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    az = jnp.asarray(smoothing_operator(basis))
    masks = [np.asarray(m) for m in band_masks(basis)]
    Pkz = jnp.asarray(kappa_pz_empirical(jax.random.PRNGKey(5), Pz, basis,
                                         lam))
    print(f"[transferL] empirical kappa Pz: pixel var "
          f"{float(Pkz.mean()):.4f} (expect ~{(np.exp(lam**2)-1)/lam**2:.4f})",
          flush=True)

    out_path = Path(args.out)
    for netname in args.nets.split(","):
        with open(f"checkpoints/{netname}.pkl", "rb") as f:
            ck = pickle.load(f)
        x0hat_fn = samplers_learned.make_score_x0hat(UNet(), ck["ema"], basis)
        for tiltname in args.tilts.split(","):
            for yseed in [int(v) for v in args.yseeds.split(",")]:
                stem = (f"gold_n{args.n}_{tiltname}_y{yseed}"
                        f"_lam{lam:.4g}_s0")
                fjson = Path("results/gold") / f"{stem}.json"
                if not fjson.exists():
                    continue
                meta = json.loads(fjson.read_text())
                dat = np.load(Path("results/gold") / f"{stem}.npz")
                gold_g = dat["z"].astype(np.float64).reshape(-1, meta["d"])
                ess_proxy = np.full(meta["d"],
                                    float(np.median(dat["ess"])))
                y = jnp.asarray(dat["y"].astype(np.float64))
                b = meta["b"]
                gold_k = np.asarray(pack(lognormal.kappa(
                    unpack(jnp.asarray(gold_g), basis), lam), basis))
                h = metrics_gold.gold_bandwidth(gold_k)
                base = dict(n=args.n, d=meta["d"], tilt=tiltname,
                            shift=meta["shift"], yseed=yseed, lam=lam, b=b,
                            N=args.N, T=args.T, space="kappa_z",
                            mode_z_proxy=True, gold_draws=gold_k.shape[0],
                            tag=args.tag)
                for seed in [int(v) for v in args.seeds.split(",")]:
                    cfg = hash((args.n, tiltname, yseed, netname, args.N,
                                args.T, seed, "transferL")) & 0x7FFFFFFF
                    key = jax.random.fold_in(jax.random.PRNGKey(seed), cfg)
                    t0 = time.time()
                    res = samplers_learned.run_learned(
                        "dps", key, x0hat_fn, basis, Pkz, az, y, b,
                        N=args.N, T=args.T, tf=TF)
                    wall = time.time() - t0
                    zk = np.asarray(res["z"], dtype=np.float64)
                    m = metrics_gold.evaluate_vs_gold(zk, gold_k, ess_proxy,
                                                      masks, h)
                    row = dict(base, sampler=f"dps_{netname}", seed=seed,
                               clip_frac=res.get("clip_frac", 0.0),
                               wall=round(wall, 2),
                               ts=time.strftime("%H:%M:%S"), **m)
                    with out_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"[transferL:{stem}] dps_{netname} s={seed}: "
                          f"mmd2={m['mmd2']:.3e} swd2={m['swd2']:.3e} "
                          f"clip={row['clip_frac']:.2f} ({wall:.0f}s)",
                          flush=True)


if __name__ == "__main__":
    main()
