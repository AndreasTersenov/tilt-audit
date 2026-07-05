#!/usr/bin/env python
"""A4 runner: the Remy-scheme K-sweep (+ misspec pair), exact scores.

Same row format as run_t1.py (joins on dim/shift/N/seed); extra columns
K, eps0, inflation. Oracle floor rows are re-emitted per (dim, shift, N,
seed) so the arm is self-contained.

Fidelity caveats vs Remy et al. 2023 (arXiv:2201.05561, CosmoStat/
jax-lensing) — what this arm audits is the pre-registered SCHEME
(P-20260703e: sigma_t^2-inflated annealed Langevin, K steps per level on our
log grid), not a line-by-line port. Differences from their production code,
recorded here after reading it: (1) they use adaptively-annealed MH-adjusted
HMC (geometric cooling 0.98, >=10 steps/level) rather than a fixed ladder —
our K plays the role of their steps-per-level knob; (2) their step size
scales as sqrt(sigma/sigma_max) globally, ours is per-mode preconditioned
(see samplers.remy docstring — a global step is stability-capped by UV modes
on this substrate); (3) they end with a probability-flow-ODE denoise, we end
with K Langevin steps at t=0 whose target IS the exact posterior. The
sigma^2 likelihood inflation — the mechanism P-20260703e bets on — is
verbatim theirs (sample_hmc.py: (sigma_gamma^2 + sigma^2) in the data term).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax

from tilt_audit import metrics, misspec, samplers, tilt
from tilt_audit.fields import (band_masks, grid_to_z, make_basis, make_pk,
                               smoothing_operator)
import jax.numpy as jnp

TF = 9.0
S_OBS = 0.5
Y_KEY = 999  # the pinned T1 observation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dims", default="64")
    p.add_argument("--shifts", default="0.5,1,2,4")
    p.add_argument("--Ks", default="1,2,5,10,30,100")
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--eps0", type=float, default=0.1)
    p.add_argument("--inflation", default="sigma2", choices=["sigma2", "exact"])
    p.add_argument("--score", default="exact",
                   help="'exact' or 'misspec:<eps>'")
    p.add_argument("--with-oracle", action="store_true",
                   help="emit an oracle floor row per (dim, shift, seed)")
    p.add_argument("--tag", default="confirmatory")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dims = [int(v) for v in args.dims.split(",")]
    shifts = [float(v) for v in args.shifts.split(",")]
    Ks = [int(v) for v in args.Ks.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
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
            for seed in seeds:
                names = (["oracle"] if args.with_oracle else []) + \
                        [("remy", K) for K in Ks]
                for name in names:
                    K = None
                    if isinstance(name, tuple):
                        name, K = name
                    cfg_id = hash((n, shift, args.N, seed, name, K,
                                   args.eps0, args.inflation)) & 0x7FFFFFFF
                    key = jax.random.fold_in(jax.random.PRNGKey(seed), cfg_id)
                    kw = ({} if name == "oracle"
                          else {"K": K, "eps0": args.eps0,
                                "inflation": args.inflation})
                    t0 = time.time()
                    res = samplers.run_sampler(name, key, Pz_run, az, y, b,
                                               N=args.N, T=args.T, tf=TF, **kw)
                    res = {k: jax.device_get(v) for k, v in res.items()}
                    wall = time.time() - t0
                    mkey = jax.random.fold_in(key, 1)
                    row = metrics.evaluate(mkey, res, Pz, az, y, b, basis,
                                           masks)
                    row.update(dim=n, d=n * n, shift=shift, b=b,
                               beta=b / S_OBS**2, s=S_OBS, N=args.N,
                               T=args.T, seed=seed, sampler=name,
                               K=(0 if K is None else K), eps0=args.eps0,
                               inflation=args.inflation, score=args.score,
                               eps=eps, tag=args.tag, y_seed=Y_KEY,
                               wall=round(wall, 3),
                               ts=time.strftime("%H:%M:%S"))
                    with out.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"[a4] {n}x{n} shift={shift} seed={seed} "
                          f"{name}{'' if K is None else f' K={K}'}: "
                          f"W2={row['w2']:.4g} g*={row['gamma_star']:.3g} "
                          f"({wall:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
