#!/usr/bin/env python
"""Stage-2 runner: block-wise certificates on the learned-score pathway
(docs/PLAN_CERT_LEARNED.md). Per row: joint + per-mode + per-|k|-band
certificate instruments, empirical raw metrics vs the TRUE target, per-band
empirical damage with matched oracle band floors, clip_frac, and (for the
analytic pathway) the exact chain-law columns."""
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

from tilt_audit import certificate, metrics, samplers, samplers_learned, tilt
from tilt_audit.fields import (band_masks, grid_to_z, make_basis, make_pk,
                               smoothing_operator)
from tilt_audit.scorenet import UNet

TF = 9.0
S_OBS = 0.5
Y_KEY = 999


def band_cert(logw_modes, mask):
    lw = np.asarray(logw_modes[:, mask], dtype=np.float64).sum(axis=1)
    m = lw.max()
    log_z = m + np.log(np.exp(lw - m).mean())
    w = np.exp(lw - m)
    w = w / w.sum()
    from arviz import psislw
    _, khat = psislw(lw - m)
    return {"ess": float(1.0 / np.sum(w**2)),
            "kl_hat": float(log_z - lw.mean()),
            "khat": float(khat)}


def band_truth(z, mask, mu, Sig, oracle_z):
    zb = jnp.asarray(z[:, mask])
    m = zb.mean(axis=0)
    v = jnp.maximum(zb.var(axis=0), 1e-12)
    kl = float(metrics.gaussian_kl(m, v, mu[mask], Sig[mask]))
    ob = jnp.asarray(oracle_z[:, mask])
    mo, vo = ob.mean(axis=0), jnp.maximum(ob.var(axis=0), 1e-12)
    floor = float(metrics.gaussian_kl(mo, vo, mu[mask], Sig[mask]))
    return {"kl_true": kl, "kl_floor": floor}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="path to a train_score pickle, or 'analytic'")
    p.add_argument("--label", required=True)
    p.add_argument("--shifts", default="0.5,1,2,4")
    p.add_argument("--modes", default="dps,unguided")
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--T", type=int, default=256)
    p.add_argument("--tag", default="confirmatory")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    n = 64
    basis = make_basis(n)
    pk = make_pk(basis)
    Pz = jnp.asarray(grid_to_z(pk, basis))
    az = jnp.asarray(smoothing_operator(basis))
    masks = [np.asarray(m) for m in band_masks(basis)]
    y, _ = tilt.make_observation(jax.random.PRNGKey(Y_KEY), Pz, az, S_OBS)

    if args.ckpt == "analytic":
        x0hat_fn = samplers_learned.make_score_x0hat("analytic", None, basis,
                                                     Pz_analytic=Pz)
    else:
        with open(args.ckpt, "rb") as f:
            ck = pickle.load(f)
        x0hat_fn = samplers_learned.make_score_x0hat(UNet(), ck["ema"], basis)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for shift in [float(v) for v in args.shifts.split(",")]:
        b = float(tilt.calibrate_b(Pz, az, y, target_shift=shift))
        mu, Sig = tilt.posterior_params(Pz, az, y, b)
        okey = jax.random.fold_in(jax.random.PRNGKey(9), hash((shift, "ofl"))
                                  & 0x7FFFFFFF)
        oracle_z = np.asarray(samplers.oracle(okey, Pz, az, y, b, N=args.N,
                                              T=1, tf=TF)["z"])
        law_cols = {}
        if args.ckpt == "analytic":
            for mode in args.modes.split(","):
                law = certificate.chain_law_ancestral(mode, Pz, az, y, b,
                                                      args.T, TF)
                ends = certificate.endpoint_errors(law, Pz, az, y, b)
                law_cols[mode] = dict(kl_path_exact=law["kl_path_exact"],
                                      log_z_model=law["log_z_analytic"],
                                      **ends)
        for mode in args.modes.split(","):
            for seed in [int(v) for v in args.seeds.split(",")]:
                cfg_id = hash((n, shift, mode, args.N, seed, args.label,
                               "certL")) & 0x7FFFFFFF
                key = jax.random.fold_in(jax.random.PRNGKey(seed), cfg_id)
                t0 = time.time()
                res = certificate.run_learned_cert(
                    mode, key, x0hat_fn, basis, Pz, az, y, b,
                    N=args.N, T=args.T, tf=TF)
                res = {k: (jax.device_get(v) if hasattr(v, "shape") else v)
                       for k, v in res.items()}
                wall = time.time() - t0
                cert = certificate.certify(res)
                cert.update(certificate.certify_modewise(res))
                row = dict(dim=n, d=n * n, shift=shift, b=b, N=args.N,
                           T=args.T, seed=seed, mode=mode, score=args.label,
                           y_seed=Y_KEY, tag=args.tag,
                           clip_frac=res.get("clip_frac", 0.0),
                           wall=round(wall, 3), ts=time.strftime("%H:%M:%S"))
                row.update(law_cols.get(mode, {}))
                row.update(cert)
                for j, mask in enumerate(masks):
                    bc = band_cert(res["logw_modes"], mask)
                    bt = band_truth(res["z"], mask, mu, Sig, oracle_z)
                    row.update({f"band{j}_{k}": v for k, v in bc.items()})
                    row.update({f"band{j}_{k}": v for k, v in bt.items()})
                raw = metrics.evaluate(jax.random.fold_in(key, 1),
                                       {"z": res["z"],
                                        "logw": np.zeros(args.N)},
                                       Pz, az, y, b, basis, masks)
                row.update({f"raw_{k}": v for k, v in raw.items()})
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[certL:{args.label}] shift={shift} {mode} s={seed}: "
                      f"KLhat={cert['kl_path_hat']:.3g} "
                      f"modeESS={cert['ess_modes_med']:.0f} "
                      f"b0ESS={row['band0_ess']:.0f} "
                      f"b2ESS={row['band2_ess']:.1f} "
                      f"clip={row['clip_frac']:.2f} ({wall:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
