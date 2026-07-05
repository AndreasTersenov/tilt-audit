#!/usr/bin/env python
"""Checkpoint-#1 zoom (pre-registered in run-plan section 7, P-j HIT branch): the
PQMass/TARP contrast on the mixture pathologies + the weight-knob ladder.

- pqmass: two-sample vs an independent oracle-mixture reference bank
  (p < 0.05 detects; mix_both is the FP control).
- tarp: marginal wrap for an unconditional target — truths are fresh mixture
  draws, each truth's 'posterior samples' an independent bank subsample;
  exchangeable under the null. Detection threshold = q95 of 20 null reps of
  the SAME construction on the mix_both bank (empirical calibration, as
  everywhere tonight).
- ksd: same weight ladder for the blind-vs-seeing contrast row (GPU-friendly
  but fine on CPU at N<=1024).

Weight ladder: w in {0.5, 0.8, 0.95, 0.99} x pathology mix_plus (misses the
1-w mode) + mix_both control. Rows -> results/mixture_contrast.jsonl.
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

from tilt_audit import ksd, mixture
from tilt_audit.fields import grid_to_z, make_basis, make_pk, smoothing_operator

BANK = 16384
L_TRUTHS = 128


def tarp_wrap(samples, truths):
    m = samples.mean((0, 1))
    s = samples.std((0, 1))
    return ((samples - m) / s + 4.0) / 8.0, ((truths - m) / s + 4.0) / 8.0


def tarp_stat(bank, truths, budget, rng, seed):
    from tarp import get_tarp_coverage
    si = np.stack([rng.choice(bank.shape[0], budget, replace=False)
                   for _ in range(truths.shape[0])])
    samples = bank[si].transpose(1, 0, 2)  # (budget, L, d)
    sn, tn = tarp_wrap(samples.astype(np.float64),
                       truths.astype(np.float64))
    ecp, alpha = get_tarp_coverage(sn, tn, norm=False, seed=seed)
    return float(np.max(np.abs(ecp - alpha)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tests", default="pqmass,tarp,ksd")
    p.add_argument("--weights", default="0.5,0.8,0.95,0.99")
    p.add_argument("--budgets", default="256,1024")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--tag", default="exploratory_ckpt1")
    p.add_argument("--out", default="results/mixture_contrast.jsonl")
    args = p.parse_args()

    basis = make_basis(64)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    az = jnp.asarray(smoothing_operator(basis))
    dmu = jnp.asarray(mixture.make_offset(Pz, basis))
    out_path = Path(args.out)
    tests = args.tests.split(",")

    def emit(row):
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    for w in [float(v) for v in args.weights.split(",")]:
        banks = {}
        for mode, kseed in (("both", 1), ("plus", 2), ("ref", 3)):
            k = jax.random.fold_in(jax.random.PRNGKey(90210),
                                   hash((mode, w, kseed)) & 0x7FFFFFFF)
            banks[mode] = np.asarray(
                mixture.sample(k, dmu, Pz, w, BANK,
                               "both" if mode == "ref" else mode),
                dtype=np.float64)
        ktruth = jax.random.fold_in(jax.random.PRNGKey(777),
                                    hash(("truths", w)) & 0x7FFFFFFF)
        truths = np.asarray(mixture.sample(ktruth, dmu, Pz, w, L_TRUTHS,
                                           "both"), dtype=np.float64)

        for budget in [int(v) for v in args.budgets.split(",")]:
            if "pqmass" in tests:
                from pqm import pqm_pvalue
                for cfg in ("both", "plus"):
                    for rep in range(args.reps):
                        rng = np.random.default_rng(
                            hash(("pqm", w, cfg, budget, rep)) & 0xFFFFFFFF)
                        xi = rng.choice(BANK, budget, replace=False)
                        ri = rng.choice(BANK, budget, replace=False)
                        np.random.seed(hash(("pqs", w, cfg, budget, rep))
                                       & 0x7FFFFFFF)
                        t0 = time.time()
                        pv = float(pqm_pvalue(
                            banks[cfg][xi], banks["ref"][ri],
                            num_refs=max(16, min(100, budget // 8))))
                        emit(dict(test="pqmass", w=w, config=f"mix_{cfg}",
                                  budget=budget, rep=rep, stat=pv, pvalue=pv,
                                  detected=bool(pv < 0.05), tag=args.tag,
                                  wall=round(time.time() - t0, 2)))
                print(f"[mixc] pqmass w={w} N={budget} done", flush=True)

            if "tarp" in tests:
                rngn = np.random.default_rng(hash(("tarpn", w, budget))
                                             & 0xFFFFFFFF)
                nulls = [tarp_stat(banks["ref"], truths, budget, rngn, r)
                         for r in range(20)]
                q95 = float(np.quantile(nulls, 0.95))
                for cfg in ("both", "plus"):
                    for rep in range(args.reps):
                        rng = np.random.default_rng(
                            hash(("tarp", w, cfg, budget, rep)) & 0xFFFFFFFF)
                        t0 = time.time()
                        st = tarp_stat(banks[cfg], truths, budget, rng,
                                       1000 + rep)
                        emit(dict(test="tarp", w=w, config=f"mix_{cfg}",
                                  budget=budget, rep=rep, stat=st,
                                  null_q95=q95,
                                  detected=bool(st > q95), tag=args.tag,
                                  wall=round(time.time() - t0, 2)))
                print(f"[mixc] tarp w={w} N={budget} (null q95={q95:.4f}) "
                      f"done", flush=True)

            if "ksd" in tests:
                def score_fn(X):
                    return np.asarray(mixture.score(jnp.asarray(X), dmu,
                                                    Pz, w))
                kern, p1, p2 = "imq", ksd.c_paper(az), -0.5
                nulls = []
                for r in range(30):
                    k = jax.random.fold_in(jax.random.PRNGKey(555),
                                           hash((w, budget, r)) & 0x7FFFFFFF)
                    G = np.asarray(mixture.sample(k, dmu, Pz, w, budget,
                                                  "both"), dtype=np.float64)
                    nulls.append(ksd.ksd_stats(G, score_fn(G), kern, p1, p2)
                                 ["score_ksd"])
                q95 = float(np.quantile(nulls, 0.95))
                for cfg in ("both", "plus"):
                    for rep in range(min(args.reps, BANK // budget)):
                        X = banks[cfg][rep * budget:(rep + 1) * budget]
                        t0 = time.time()
                        st = ksd.ksd_stats(X, score_fn(X), kern, p1, p2)
                        emit(dict(test="ksd_imq_paper", w=w,
                                  config=f"mix_{cfg}", budget=budget,
                                  rep=rep, stat=st["score_ksd"],
                                  null_q95=q95,
                                  ratio_q95=st["score_ksd"] / q95,
                                  detected=bool(st["score_ksd"] > q95),
                                  tag=args.tag,
                                  wall=round(time.time() - t0, 2)))
                print(f"[mixc] ksd w={w} N={budget} done", flush=True)


if __name__ == "__main__":
    main()
