#!/usr/bin/env python
"""Certificate kill-test grid runner (docs/PLAN_CERT_KILLTEST.md §3).

Per row: exact chain-law quantities (zero-noise ground truth: kl_path_exact,
kl_end_exact, w2_end_exact — vs the TRUE target; kl_end_model additionally for
misspec rows), sampled certificate instruments (log Ẑ, ESS_res, k̂, KL̂,
mode-wise Rao-Blackwell summaries where enabled), and empirical raw/repaired
metrics. The certificate and chain law live entirely in the sampler's OWN
model (Pz_run): with a contaminated score they price the steering step only —
the kl_end_exact column then carries the model pass-through the certificate
cannot see (the attribution demo).
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

from tilt_audit import certificate, metrics, misspec, tilt
from tilt_audit.fields import (band_masks, grid_to_z, make_basis, make_pk,
                               smoothing_operator)

TF = 9.0
S_OBS = 0.5
Y_KEY = 999


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dims", default="16,32,64")
    p.add_argument("--shifts", default="0.5,1,2,4")
    p.add_argument("--modes", default="dps,exact_guidance,unguided")
    p.add_argument("--Ns", default="256")
    p.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    p.add_argument("--T", type=int, default=64)
    p.add_argument("--score", default="exact")
    p.add_argument("--y-seeds", default="")
    p.add_argument("--modewise-dims", default="64",
                   help="dims that also emit per-mode (Rao-Blackwell) columns")
    p.add_argument("--tag", default="confirmatory")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dims = [int(v) for v in args.dims.split(",")]
    shifts = [float(v) for v in args.shifts.split(",")]
    modes = args.modes.split(",")
    Ns = [int(v) for v in args.Ns.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    y_seeds = ([int(v) for v in args.y_seeds.split(",")] if args.y_seeds
               else [Y_KEY])
    mw_dims = {int(v) for v in args.modewise_dims.split(",") if v}
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
        for yseed in y_seeds:
            y, _ = tilt.make_observation(jax.random.PRNGKey(yseed), Pz, az,
                                         S_OBS)
            for shift in shifts:
                b = float(tilt.calibrate_b(Pz, az, y, target_shift=shift))
                for mode in modes:
                    law = certificate.chain_law(mode, Pz_run, az, y, b,
                                                args.T, TF)
                    ends_true = certificate.endpoint_errors(law, Pz, az, y, b)
                    law_cols = dict(kl_path_exact=law["kl_path_exact"],
                                    e_logw_exact=law["e_logw"],
                                    log_z_model=law["log_z_analytic"],
                                    **ends_true)
                    if eps != 0.0:
                        em = certificate.endpoint_errors(law, Pz_run, az, y, b)
                        law_cols["kl_end_model"] = em["kl_end_exact"]
                        law_cols["w2_end_model"] = em["w2_end_exact"]
                    for N in Ns:
                        for seed in seeds:
                            cfg_id = hash((n, yseed, shift, mode, N, seed,
                                           eps, "cert")) & 0x7FFFFFFF
                            key = jax.random.fold_in(jax.random.PRNGKey(seed),
                                                     cfg_id)
                            mw = n in mw_dims
                            t0 = time.time()
                            res = certificate.run_cert(
                                mode, key, Pz_run, az, y, b, N=N, T=args.T,
                                tf=TF, modewise=mw)
                            res = {k: jax.device_get(v)
                                   for k, v in res.items()}
                            wall = time.time() - t0
                            cert = certificate.certify(res)
                            if mw:
                                cert.update(certificate.certify_modewise(res))
                            mkey = jax.random.fold_in(key, 1)
                            raw = metrics.evaluate(
                                mkey, {"z": res["z"],
                                       "logw": np.zeros(N)},
                                Pz, az, y, b, basis, masks)
                            rep = metrics.evaluate(
                                jax.random.fold_in(key, 2),
                                {"z": res["z"], "logw": res["logw"]},
                                Pz, az, y, b, basis, masks)
                            row = dict(dim=n, d=n * n, shift=shift, b=b,
                                       N=N, T=args.T, seed=seed, mode=mode,
                                       score=args.score, eps=eps,
                                       y_seed=yseed, tag=args.tag,
                                       wall=round(wall, 3),
                                       ts=time.strftime("%H:%M:%S"))
                            row.update(law_cols)
                            row.update(cert)
                            row.update({f"raw_{k}": v for k, v in raw.items()})
                            row.update({f"rep_{k}": v for k, v in rep.items()})
                            with out.open("a") as f:
                                f.write(json.dumps(row) + "\n")
                            print(f"[cert] {n}x{n} y{yseed} shift={shift} "
                                  f"{mode} N={N} s={seed}: "
                                  f"pathKL*={law['kl_path_exact']:.3g} "
                                  f"endKL*={ends_true['kl_end_exact']:.3g} "
                                  f"KLhat={cert['kl_path_hat']:.3g} "
                                  f"ESS={cert['ess_res']:.1f} "
                                  f"khat={cert['khat']:.2f} ({wall:.1f}s)",
                                  flush=True)


if __name__ == "__main__":
    main()
