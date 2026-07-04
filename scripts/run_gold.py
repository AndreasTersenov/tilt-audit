#!/usr/bin/env python
"""NUTS gold standards on the lognormal-observation posterior (plan §3).

One invocation = one config (crash-safe at config granularity). Samples in the
whitened u-space (prior = N(0,I)); draws stored in the z-basis as float32 npz
with a JSON sidecar carrying diagnostics + provenance. With --linear-check
(lam=0.01), compares NUTS moments against the closed-form linearized posterior
and prints the T-L1 gate line.

T-L1 operationalization (pinned pre-data; the plan's "per-mode mean z <4,
variance ratios within 5%" made statistical at d modes x finite ESS):
  frac(|mean z-score| > 4) <= 1e-3, median |var ratio - 1| <= 0.02,
  frac(|var ratio - 1| > 0.05) <= 0.02, max R-hat < 1.01, min ESS > 400.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from tilt_audit import lognormal, tilt
from tilt_audit.fields import grid_to_z, make_basis, make_pk, smoothing_operator

S_OBS = 0.5
Y_KEY = 999
SHIFTS = {"mid": 1.0, "strong": 2.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--lam", type=float, default=None,
                   help="default: skewness(kappa)=1 calibration")
    p.add_argument("--tilt", choices=list(SHIFTS), required=True)
    p.add_argument("--yseed", type=int, default=0)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--draws", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--linear-check", action="store_true",
                   help="T-L1 mode: compare vs closed-form linearized posterior")
    p.add_argument("--tag", default="confirmatory")
    p.add_argument("--out-dir", default="results/gold")
    args = p.parse_args()

    from numpyro.infer import MCMC, NUTS

    lam = lognormal.default_lambda() if args.lam is None else args.lam
    n = args.n
    basis = make_basis(n)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    az = jnp.asarray(smoothing_operator(basis))

    ykey = jax.random.fold_in(jax.random.PRNGKey(Y_KEY), args.yseed)
    y, g_truth = lognormal.make_lognormal_observation(ykey, Pz, az, basis,
                                                      lam, S_OBS)
    b = float(lognormal.calibrate_b_linearized(Pz, az, y, lam,
                                               SHIFTS[args.tilt]))

    pot = lognormal.make_potential_u(Pz, az, basis, lam, y, b)
    d = int(Pz.shape[0])
    ikey, rkey = jax.random.split(jax.random.PRNGKey(args.seed))
    u0 = 0.1 * jax.random.normal(ikey, (args.chains, d))

    mcmc = MCMC(NUTS(potential_fn=pot), num_warmup=args.warmup,
                num_samples=args.draws, num_chains=args.chains,
                chain_method="vectorized", progress_bar=False)
    t0 = time.time()
    mcmc.run(rkey, init_params=u0)
    wall = time.time() - t0

    u = np.asarray(mcmc.get_samples(group_by_chain=True))  # (C, S, d)
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    ess = np.asarray(effective_sample_size(u))
    rhat = np.asarray(split_gelman_rubin(u))
    z = u * np.asarray(np.sqrt(Pz))  # z-basis draws

    name = f"gold_n{n}_{args.tilt}_y{args.yseed}_lam{lam:.4g}_s{args.seed}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    meta = dict(n=n, d=d, lam=lam, tilt=args.tilt, shift=SHIFTS[args.tilt],
                b=b, s_obs=S_OBS, yseed=args.yseed, y_key=Y_KEY,
                chains=args.chains, warmup=args.warmup, draws=args.draws,
                seed=args.seed, tag=args.tag, wall=round(wall, 1),
                rhat_max=float(rhat.max()), ess_min=float(ess.min()),
                ess_med=float(np.median(ess)), commit=commit,
                b_convention="calibrate_b on linearized operator lam*az")
    np.savez_compressed(out_dir / f"{name}.npz",
                        z=z.astype(np.float32), ess=ess.astype(np.float32),
                        y=np.asarray(y, dtype=np.float32),
                        g_truth=np.asarray(g_truth, dtype=np.float32))
    tl2 = (rhat.max() < 1.01) and (ess.min() > 400)
    print(f"[gold:{name}] wall={wall:.0f}s rhat_max={rhat.max():.4f} "
          f"ess_min={ess.min():.0f} ess_med={np.median(ess):.0f} "
          f"T-L2={'PASS' if tl2 else 'FAIL'}", flush=True)

    if args.linear_check:
        mu, Sig = lognormal.linear_limit_params(Pz, az, y, lam, b)
        mu, Sig = np.asarray(mu), np.asarray(Sig)
        zf = z.reshape(-1, d)
        zm, zv = zf.mean(axis=0), zf.var(axis=0)
        mean_z = (zm - mu) / np.sqrt(Sig / np.maximum(ess, 1.0))
        vratio = zv / Sig
        fz = float(np.mean(np.abs(mean_z) > 4.0))
        medr = float(np.median(np.abs(vratio - 1.0)))
        fr = float(np.mean(np.abs(vratio - 1.0) > 0.05))
        ok = (fz <= 1e-3) and (medr <= 0.02) and (fr <= 0.02) and tl2
        meta["tl1"] = dict(frac_meanz_gt4=fz, med_vratio_dev=medr,
                           frac_vratio_gt5pct=fr,
                           max_abs_meanz=float(np.abs(mean_z).max()),
                           max_vratio_dev=float(np.abs(vratio - 1.0).max()),
                           passed=bool(ok))
        print(f"[T-L1:n{n}] frac|z|>4={fz:.2e} med|vr-1|={medr:.4f} "
              f"frac|vr-1|>5%={fr:.4f} max|z|={np.abs(mean_z).max():.2f} "
              f"=> {'PASS' if ok else 'FAIL'}", flush=True)

    with (out_dir / f"{name}.json").open("w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
