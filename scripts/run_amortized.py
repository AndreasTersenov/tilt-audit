#!/usr/bin/env python
"""A3 audit runner: sample the CONDITIONAL net's plain ancestral reverse
diffusion (no guidance — the conditioning is in the net) and score it against
the exact Wiener posterior at the trained Bayes point b = s^2.

Alongside the geometry metrics (W2/KL/gamma* from metrics.evaluate), each row
carries the SUMMARY-level checks the community would run (P-20260703d's
"summaries pass / geometry fails" contrast is computed within our own rows):
  rel_mean_err   |m - mu*| / |mu*|                (posterior-mean agreement)
  px_var_ratio   mean_k v / mean_k Sig*           (pixel-variance agreement;
                 unitary basis: pixel variance = mean over modes)
  bp{j}_ratio    band <m^2+v> / band <mu*^2+Sig*> (band-power agreement)

Oracle rows (--with-oracle) are emitted at the same (y, b, N) for the floor.
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

from tilt_audit import metrics, samplers, samplers_learned, tilt
from tilt_audit.fields import (band_masks, grid_to_z, make_basis, make_pk,
                               smoothing_operator, unpack)
from tilt_audit.scorenet import UNetCond

TF = 9.0
S_OBS = 0.5


def summary_checks(res, mu, Sig, masks):
    m, v = metrics.weighted_moments(res["z"], res["logw"])
    out = {
        "rel_mean_err": float(jnp.linalg.norm(m - mu) / jnp.linalg.norm(mu)),
        "px_var_ratio": float(jnp.mean(v) / jnp.mean(Sig)),
    }
    for j, mask in enumerate(masks):
        mk = jnp.asarray(mask)
        bp_s = jnp.mean((m**2 + v)[mk])
        bp_t = jnp.mean((mu**2 + Sig)[mk])
        out[f"bp{j}_ratio"] = float(bp_s / bp_t)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--label", required=True, help="e.g. amortized:default")
    p.add_argument("--y-seeds", default="999,1000,1001,1002,1003")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--T", type=int, default=256)
    p.add_argument("--with-oracle", action="store_true")
    p.add_argument("--tag", default="confirmatory")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.ckpt, "rb") as f:
        ck = pickle.load(f)
    n = ck["n"]
    model = UNetCond(chs=tuple(ck["chs"]))
    ema = ck["ema"]

    basis = make_basis(n)
    pk = make_pk(basis)
    Pz = jnp.asarray(grid_to_z(pk, basis))
    az = jnp.asarray(smoothing_operator(basis))
    masks = band_masks(basis)
    b = S_OBS**2  # the Bayes point the net was trained at

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for yseed in [int(v) for v in args.y_seeds.split(",")]:
        y, _ = tilt.make_observation(jax.random.PRNGKey(yseed), Pz, az, S_OBS)
        mu, Sig = tilt.posterior_params(Pz, az, y, b)
        shift = float(tilt.rms_shift(Pz, az, y, b))
        y_map32 = jnp.asarray(unpack(y, basis), dtype=jnp.float32)

        def x0hat_fn(z, t, _ym=y_map32):
            x = unpack(z, basis)
            B = x.shape[0]
            eps = model.apply(ema, x.astype(jnp.float32),
                              jnp.broadcast_to(_ym, x.shape).astype(jnp.float32),
                              jnp.full(B, t, dtype=jnp.float32))
            eps = eps.astype(z.dtype)
            sig = jnp.sqrt(1.0 - jnp.exp(-2.0 * t))
            alpha = jnp.exp(-t)
            from tilt_audit.fields import pack
            return pack((x - sig * eps) / alpha, basis)

        x0hat_jit = jax.jit(x0hat_fn)

        for seed in [int(v) for v in args.seeds.split(",")]:
            names = (["oracle"] if args.with_oracle else []) + ["ancestral"]
            for name in names:
                cfg_id = hash((n, yseed, args.N, seed, name,
                               args.label)) & 0x7FFFFFFF
                key = jax.random.fold_in(jax.random.PRNGKey(seed), cfg_id)
                t0 = time.time()
                if name == "oracle":
                    res = samplers.run_sampler("oracle", key, Pz, az, y, b,
                                               N=args.N, T=args.T, tf=TF)
                    res = {k: jax.device_get(v) for k, v in res.items()}
                else:
                    res = samplers_learned.run_learned(
                        "ancestral", key, x0hat_jit, basis, Pz, az, y, b,
                        N=args.N, T=args.T, tf=TF)
                wall = time.time() - t0
                row = metrics.evaluate(jax.random.fold_in(key, 1), res, Pz,
                                       az, y, b, basis, masks)
                row.update(summary_checks(res, mu, Sig, masks))
                row.update(dim=n, d=n * n, shift=round(shift, 4), b=b,
                           beta=1.0, s=S_OBS, N=args.N, T=args.T, seed=seed,
                           sampler=name, score=args.label, y_seed=yseed,
                           train_steps=ck.get("step"), tag=args.tag,
                           wall=round(wall, 3), ts=time.strftime("%H:%M:%S"))
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[a3:{args.label}] y{yseed} seed={seed} {name}: "
                      f"W2={row['w2']:.4g} g*={row['gamma_star']:.3g} "
                      f"meanerr={row['rel_mean_err']:.3g} "
                      f"pxvar={row['px_var_ratio']:.3g} ({wall:.1f}s)",
                      flush=True)


if __name__ == "__main__":
    main()
