#!/usr/bin/env python
"""EXPLORATORY: AIS certificate for the Remy scheme (plan §3, GPU-2 column)."""
import argparse
import functools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from tilt_audit import certificate, metrics, tilt
from tilt_audit.fields import (band_masks, grid_to_z, make_basis, make_pk,
                               smoothing_operator)

TF = 9.0
S_OBS = 0.5
Y_KEY = 999


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dims", default="64")
    p.add_argument("--shifts", default="1,4")
    p.add_argument("--Ks", default="5,30")
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for n in [int(v) for v in args.dims.split(",")]:
        basis = make_basis(n)
        Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
        az = jnp.asarray(smoothing_operator(basis))
        masks = band_masks(basis)
        y, _ = tilt.make_observation(jax.random.PRNGKey(Y_KEY), Pz, az, S_OBS)
        for shift in [float(v) for v in args.shifts.split(",")]:
            b = float(tilt.calibrate_b(Pz, az, y, target_shift=shift))
            for K in [int(v) for v in args.Ks.split(",")]:
                fn = jax.jit(functools.partial(certificate.remy_ais, K=K),
                             static_argnames=("N", "T", "tf"))
                for seed in [int(v) for v in args.seeds.split(",")]:
                    key = jax.random.fold_in(
                        jax.random.PRNGKey(seed),
                        hash((n, shift, K, seed, "remyais")) & 0x7FFFFFFF)
                    t0 = time.time()
                    res = fn(key, Pz, az, y, b, N=args.N, T=args.T, tf=TF)
                    res = {k: (jax.device_get(v) if hasattr(v, "shape")
                               else v) for k, v in res.items()}
                    wall = time.time() - t0
                    cert = certificate.certify(res)
                    raw = metrics.evaluate(
                        jax.random.fold_in(key, 1),
                        {"z": res["z"], "logw": np.zeros(args.N)},
                        Pz, az, y, b, basis, masks)
                    rep = metrics.evaluate(
                        jax.random.fold_in(key, 2),
                        {"z": res["z"], "logw": res["logw"]},
                        Pz, az, y, b, basis, masks)
                    row = dict(dim=n, d=n * n, shift=shift, b=b, N=args.N,
                               T=args.T, K=K, seed=seed, mode="remy_ais",
                               score="exact", eps=0.0, y_seed=Y_KEY,
                               tag="exploratory", wall=round(wall, 3),
                               log_z_ais_analytic=float(res["log_z_ais_analytic"]),
                               ts=time.strftime("%H:%M:%S"))
                    row.update(cert)
                    row.update({f"raw_{k}": v for k, v in raw.items()})
                    row.update({f"rep_{k}": v for k, v in rep.items()})
                    with out.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    zgap = cert["log_z_est"] - res["log_z_ais_analytic"]
                    print(f"[cert-remy] {n}x{n} shift={shift} K={K} s={seed}:"
                          f" ESS={cert['ess_res']:.1f} khat={cert['khat']:.2f}"
                          f" logZgap={zgap:.3g} rawW2={raw['w2']:.3g} "
                          f"repW2={rep['w2']:.3g} ({wall:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
