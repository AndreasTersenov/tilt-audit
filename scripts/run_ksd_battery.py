#!/usr/bin/env python
"""Track A batteries: score-KSD trial grids -> results/ksd_trial.jsonl.

Arms (plan §2; all detections at empirically calibrated one-sided alpha=0.05,
rank vs 60 oracle nulls; nulls cached in results/ksd_nulls/):

  null     — write the T-K1 100-rep null curves as rows (bookkeeping).
  power    — GRF archives x budgets {64,256,1024,4096} x kernels; TRUE score.
  wrongref — contaminated reference score (analytic eps) on matched-wrong /
             clean-sampler / oracle banks; nulls generated from the WRONG
             model itself (the practitioner's own null).
  mixture  — exact 2-component mixture; sampler pathologies {both(control),
             plus(missed mode), swapped}; TRUE mixture score.

Rep policy (pinned): disjoint blocks, n_reps = min(20, bank // budget) — the
20-rep target is honored where the bank allows; actual n_reps in every row.
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

from tilt_audit import ksd, misspec, mixture, tilt
from tilt_audit.fields import grid_to_z, make_basis, make_pk, smoothing_operator

TF = 9.0
S_OBS = 0.5
Y_KEY = 999
BUDGETS = (64, 256, 1024, 4096)
N_NULL = 60

POWER_ARCHIVES = ["oracle_null", "dps", "sap", "twisted", "dps_em03",
                  "dps_ep03", "twisted_em03", "dps_s05", "dps_s2"]
ARCHIVE_SHIFT = {"dps_s05": 0.5, "dps_s2": 2.0}  # rest at 1.0


def setup(n=64):
    basis = make_basis(n)
    pk = make_pk(basis)
    Pz = jnp.asarray(grid_to_z(pk, basis))
    az = jnp.asarray(smoothing_operator(basis))
    y, _ = tilt.make_observation(jax.random.PRNGKey(Y_KEY), Pz, az, S_OBS)
    return basis, pk, Pz, az, y


def kernel_set(az):
    return {"imq_paper": ("imq", ksd.c_paper(az), -0.5),
            "imq_c1": ("imq", 1.0, -0.5),
            "rbf_med": ("rbf", None, 0.0)}  # h resolved per (target, budget)


def load_bank(name):
    for d in ("results/archives16k", "results/archives"):
        f = Path(d) / f"{name}.npz"
        if f.exists():
            dat = np.load(f, allow_pickle=True)
            return (np.asarray(dat["z"], dtype=np.float64),
                    json.loads(str(dat["meta"])))
    raise FileNotFoundError(name)


def null_stats(cache_dir, tag, draw_fn, score_fn, kern, p1, p2, budget):
    """60-rep null for one (target, budget, kernel); cached to JSON."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{tag}_N{budget}.json"
    if f.exists():
        return json.loads(f.read_text())
    vals = []
    for rep in range(N_NULL):
        X = draw_fn(50_000 + rep, budget)
        vals.append(ksd.ksd_stats(X, score_fn(X), kern, p1, p2)["score_ksd"])
    out = dict(mean=float(np.mean(vals)), sd=float(np.std(vals)),
               q95=float(np.quantile(vals, 0.95)),
               q99=float(np.quantile(vals, 0.99)), n=N_NULL, p1=float(p1))
    f.write_text(json.dumps(out))
    return out


def emit(out_path, row):
    with out_path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def run_target_battery(out_path, arm, tag_prefix, banks, draw_null, score_fn,
                       az, budgets, extra, tag):
    """Shared engine: banks = {config_name: (samples, meta_extra)};
    draw_null(seed, N) draws null samples; score_fn(X) the reference score."""
    kernels = kernel_set(az)
    cache = Path("results/ksd_nulls")
    for budget in budgets:
        h_med = ksd.median_heuristic(draw_null(49_999, min(budget, 1024)))
        for kname, (kern, p1, p2) in kernels.items():
            p1r = h_med if kname == "rbf_med" else p1
            nul = null_stats(cache, f"{tag_prefix}_{kname}", draw_null,
                             score_fn, kern, p1r, p2, budget)
            for cname, (bank, meta_extra) in banks.items():
                n_reps = min(20, bank.shape[0] // budget)
                for rep in range(n_reps):
                    X = bank[rep * budget:(rep + 1) * budget]
                    t0 = time.time()
                    st = ksd.ksd_stats(X, score_fn(X), kern, p1r, p2)
                    st = {k: v for k, v in st.items()
                          if k not in ("kernel", "p1", "p2")}
                    row = dict(arm=arm, config=cname, budget=budget,
                               rep=rep, n_reps=n_reps, kernel=kname,
                               p1=float(p1r),
                               detect=bool(st["score_ksd"] > nul["q95"]),
                               ratio_q95=st["score_ksd"] / nul["q95"],
                               null_mean=nul["mean"], null_sd=nul["sd"],
                               null_q95=nul["q95"], tag=tag,
                               wall=round(time.time() - t0, 2),
                               ts=time.strftime("%H:%M:%S"), **st,
                               **meta_extra, **extra)
                    emit(out_path, row)
                print(f"[ksd:{arm}] {cname} N={budget} {kname}: "
                      f"{n_reps} reps done", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True,
                   choices=["null", "power", "wrongref", "mixture"])
    p.add_argument("--budgets", default="64,256,1024,4096")
    p.add_argument("--eps", type=float, default=-0.3,
                   help="wrongref: reference-score contamination")
    p.add_argument("--weights", default="0.5,0.8", help="mixture weights")
    p.add_argument("--tag", default="confirmatory")
    p.add_argument("--out", default="results/ksd_trial.jsonl")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    budgets = tuple(int(v) for v in args.budgets.split(","))
    basis, pk, Pz, az, y = setup()
    d = int(Pz.shape[0])

    if args.arm == "null":
        nulls = json.loads(Path("results/tk1_nulls.json").read_text())
        for k, v in nulls.items():
            emit(out_path, dict(arm="null", config=k, tag="confirmatory",
                                ts=time.strftime("%H:%M:%S"), **v))
        print(f"[ksd:null] {len(nulls)} T-K1 null rows written", flush=True)
        return

    if args.arm == "power":
        by_shift = {}
        for name in POWER_ARCHIVES:
            by_shift.setdefault(ARCHIVE_SHIFT.get(name, 1.0), []).append(name)
        for shift, names in by_shift.items():
            b = float(tilt.calibrate_b(Pz, az, y, target_shift=shift))
            mu, Sig = tilt.posterior_params(Pz, az, y, b)
            munp, Signp = np.asarray(mu), np.asarray(Sig)

            def draw_null(seed, N, mu=mu, Sig=Sig):
                k = jax.random.fold_in(jax.random.PRNGKey(seed), 77)
                return np.asarray(mu + jnp.sqrt(Sig)
                                  * jax.random.normal(k, (N, d)))

            def score_fn(X, munp=munp, Signp=Signp):
                return ksd.score_gaussian(X, munp, Signp)

            banks = {}
            for nm in names:
                z, meta = load_bank(nm)
                banks[nm] = (z, dict(sampler=meta["sampler"],
                                     eps=meta["eps"], shift=shift,
                                     bank=z.shape[0]))
            run_target_battery(out_path, "power", f"grf_s{shift}", banks,
                               draw_null, score_fn, az, budgets,
                               dict(score_mode="true"), args.tag)

    elif args.arm == "wrongref":
        # reference score from the eps-contaminated prior, SAME y and b
        shift = 1.0
        b = float(tilt.calibrate_b(Pz, az, y, target_shift=shift))
        Pz_e = jnp.asarray(misspec.contaminated_pz(basis, pk, args.eps))
        mu_e, Sig_e = tilt.posterior_params(Pz_e, az, y, b)
        munp, Signp = np.asarray(mu_e), np.asarray(Sig_e)

        def draw_null(seed, N):
            # the practitioner's own null: draws from THEIR wrong posterior
            k = jax.random.fold_in(jax.random.PRNGKey(seed), 78)
            return np.asarray(mu_e + jnp.sqrt(Sig_e)
                              * jax.random.normal(k, (N, d)))

        def score_fn(X):
            return ksd.score_gaussian(X, munp, Signp)

        banks = {}
        for nm in ("dps_em03", "twisted_em03", "dps", "oracle_null"):
            z, meta = load_bank(nm)
            banks[nm] = (z, dict(sampler=meta["sampler"],
                                 eps_sampler=meta["eps"], shift=shift,
                                 bank=z.shape[0]))
        run_target_battery(out_path, "wrongref", f"wref_e{args.eps}", banks,
                           draw_null, score_fn, az, budgets,
                           dict(score_mode="wrongref", eps_ref=args.eps),
                           args.tag)

    elif args.arm == "mixture":
        dmu = jnp.asarray(mixture.make_offset(Pz, basis))
        for w in [float(v) for v in args.weights.split(",")]:
            def draw_null(seed, N, w=w):
                k = jax.random.fold_in(jax.random.PRNGKey(seed), 79)
                return np.asarray(mixture.sample(k, dmu, Pz, w, N, "both"))

            def score_fn(X, w=w):
                return np.asarray(mixture.score(jnp.asarray(X), dmu, Pz, w))

            banks = {}
            for mode in ("both", "plus", "swapped"):
                k = jax.random.fold_in(jax.random.PRNGKey(31337),
                                       hash((mode, w)) & 0x7FFFFFFF)
                bank = np.asarray(mixture.sample(k, dmu, Pz, w, 16384, mode),
                                  dtype=np.float64)
                banks[f"mix_{mode}"] = (bank, dict(sampler=mode, w=w,
                                                   bank=bank.shape[0]))
            run_target_battery(out_path, "mixture", f"mix_w{w}", banks,
                               draw_null, score_fn, az, budgets,
                               dict(score_mode="true", w=w), args.tag)


if __name__ == "__main__":
    main()
