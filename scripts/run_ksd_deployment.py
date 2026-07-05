#!/usr/bin/env python
"""KSD deployment columns: the score a practitioner actually has (plan §2).

Reference score per arXiv:2602.04189 Appendix A.2, mapped to our VP
convention: prior score from the trained net averaged over M=4 Gaussian
perturbations at the noise level with sigma_t = 0.3
(t* = -0.5 log(1 - 0.09) = 0.04716, alpha = e^{-t*}), plus the ANALYTIC
likelihood gradient a(y - a z)/b. Scores computed in pixel space and packed
(z = Ux orthonormal => s_z = U s_x).

Nulls: oracle draws judged by the SAME deployment score (the well-calibrated
practitioner), 60 reps, cached per (net, budget, kernel).
Rows -> results/ksd_trial.jsonl, arm='deployment'.
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

from tilt_audit import ksd, tilt
from tilt_audit.fields import (grid_to_z, make_basis, make_pk, pack,
                               smoothing_operator, unpack)
from tilt_audit.scorenet import UNet

S_OBS = 0.5
Y_KEY = 999
M_PERT = 4

ARCHIVES = ["oracle_null", "dps", "sap", "dps_em03", "twisted_em03"]


def make_deployment_score(ckpt_path, basis, az, y, b, sigma_score=0.3):
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    model = UNet()
    t_star = float(-0.5 * np.log(1.0 - sigma_score**2))
    alpha = float(np.exp(-t_star))

    @jax.jit
    def score_fn_j(z, key):
        x = unpack(z, basis)  # (N, n, n)
        s_acc = jnp.zeros_like(x)
        for m in range(M_PERT):
            xi = jax.random.normal(jax.random.fold_in(key, m), x.shape)
            xt = alpha * x + sigma_score * xi
            eps = model.apply(ck["ema"], xt.astype(jnp.float32),
                              jnp.full(x.shape[0], t_star,
                                       dtype=jnp.float32))
            s_acc = s_acc + alpha * (-eps.astype(x.dtype) / sigma_score)
        s_prior_z = pack(s_acc / M_PERT, basis)
        s_lik = az * (y - az * z) / b
        return s_prior_z + s_lik

    def score_fn(X, seed=0, chunk=256):
        # chunked: M=4 unrolled net calls at full batch OOM at N=1024
        # when the GPU is shared (paid 00:25); activations cap at chunk size
        outs = []
        for i in range(0, X.shape[0], chunk):
            outs.append(np.asarray(score_fn_j(
                jnp.asarray(X[i:i + chunk]),
                jax.random.fold_in(jax.random.PRNGKey(seed), i))))
        return np.concatenate(outs, axis=0)
    return score_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", default="s_clean,s_mis_m03")
    p.add_argument("--budgets", default="256,1024")
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--archives", default=",".join(ARCHIVES))
    p.add_argument("--tag", default="confirmatory")
    p.add_argument("--out", default="results/ksd_trial.jsonl")
    args = p.parse_args()

    basis = make_basis(64)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    az = jnp.asarray(smoothing_operator(basis))
    y, _ = tilt.make_observation(jax.random.PRNGKey(Y_KEY), Pz, az, S_OBS)
    b = float(tilt.calibrate_b(Pz, az, y, target_shift=1.0))
    mu, Sig = tilt.posterior_params(Pz, az, y, b)
    d = int(Pz.shape[0])

    def draw_oracle(seed, N):
        k = jax.random.fold_in(jax.random.PRNGKey(seed), 81)
        return np.asarray(mu + jnp.sqrt(Sig) * jax.random.normal(k, (N, d)))

    def load_bank(name):
        for dd in ("results/archives16k", "results/archives"):
            f = Path(dd) / f"{name}.npz"
            if f.exists():
                dat = np.load(f, allow_pickle=True)
                return (np.asarray(dat["z"], dtype=np.float64),
                        json.loads(str(dat["meta"])))
        raise FileNotFoundError(name)

    out_path = Path(args.out)
    cache = Path("results/ksd_nulls")
    kernels = {"imq_paper": ("imq", ksd.c_paper(az), -0.5),
               "imq_c1": ("imq", 1.0, -0.5)}
    for net in args.nets.split(","):
        score_fn = make_deployment_score(f"checkpoints/{net}.pkl", basis,
                                         az, y, b, sigma_score=args.sigma)
        for budget in [int(v) for v in args.budgets.split(",")]:
            for kname, (kern, p1, p2) in kernels.items():
                nul = ksd_null = None
                tagk = f"deploy_{net}_{kname}_N{budget}_sig{args.sigma}"
                fnul = cache / f"{tagk}.json"
                if fnul.exists():
                    nul = json.loads(fnul.read_text())
                else:
                    vals = []
                    for rep in range(60):
                        X = draw_oracle(60_000 + rep, budget)
                        S = score_fn(X, seed=rep)
                        vals.append(ksd.ksd_stats(X, S, kern, p1, p2)
                                    ["score_ksd"])
                    nul = dict(mean=float(np.mean(vals)),
                               sd=float(np.std(vals)),
                               q95=float(np.quantile(vals, 0.95)), n=60)
                    fnul.write_text(json.dumps(nul))
                for cname in args.archives.split(","):
                    bank, meta = load_bank(cname)
                    n_reps = min(8, bank.shape[0] // budget)
                    for rep in range(n_reps):
                        X = bank[rep * budget:(rep + 1) * budget]
                        t0 = time.time()
                        st = ksd.ksd_stats(X, score_fn(X, seed=1000 + rep),
                                           kern, p1, p2)
                        st = {k: v for k, v in st.items()
                              if k not in ("kernel", "p1", "p2")}
                        row = dict(arm="deployment", config=cname,
                                   net=net, sigma=args.sigma,
                                   budget=budget, rep=rep,
                                   n_reps=n_reps, kernel=kname,
                                   p1=float(p1), score_mode="net",
                                   sampler=meta["sampler"],
                                   eps_sampler=meta["eps"],
                                   detect=bool(st["score_ksd"] > nul["q95"]),
                                   ratio_q95=st["score_ksd"] / nul["q95"],
                                   null_q95=nul["q95"],
                                   null_mean=nul["mean"], tag=args.tag,
                                   wall=round(time.time() - t0, 2),
                                   ts=time.strftime("%H:%M:%S"), **st)
                        with out_path.open("a") as f:
                            f.write(json.dumps(row) + "\n")
                    print(f"[ksd:deploy] {net} {cname} N={budget} {kname}: "
                          f"{n_reps} reps", flush=True)


if __name__ == "__main__":
    main()
