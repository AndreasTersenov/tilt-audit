#!/usr/bin/env python
"""Diagnostic C — independent-reference KSD (P-20260705i). Attacks barrier 3
(reference confounding): score-KSD false-certifies a matched-wrong sampler
because the diagnostic reuses the SAME wrong score the sampler used (P-20260704k
— checking a model against itself). The fix: judge the sampler with a
DELIBERATELY INDEPENDENT reference score whose error is uncorrelated with the
sampler's.

Setup on the Gaussian bench (all exact):
  target       true tilted-GRF posterior N(mu, Sig).
  sampler      a PROPER sampler of an eps_s-contaminated target (draws from
               N(mu_s, Sig_s)); same-reference KSD reads it as null.
  references   the eps_r-contaminated posterior score, swept over eps_r
               INDEPENDENTLY of eps_s (eps_r = eps_s is the confounded baseline;
               eps_r = 0 is the true score). Detection is against that
               reference's OWN null (oracle draws from the eps_r target judged
               by the eps_r score — the practitioner's null), 95th pct over reps.

Claim (P-i): detection of the eps_s sampler recovers from ~FP to >=0.9 as the
independent reference's error falls below the sampler's true damage; a
reference no better than the sampler's own error buys nothing. Deliverable:
detect-power and KSD-ratio vs |eps_r|, with the sampler's true W2 damage marked.
Rows -> results/indepref.jsonl (append; tag pilot|grid).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tilt_audit import ksd, metrics, misspec, tilt
from tilt_audit.fields import grid_to_z, make_basis, make_pk, smoothing_operator

S_OBS = 0.5
Y_KEY = 999
SHIFTS = {"mid": 1.0, "strong": 2.0}
OUT_DEFAULT = "results/indepref.jsonl"


def gauss_draws(key, mu, Sig, N):
    return mu[None, :] + jnp.sqrt(Sig)[None, :] * jax.random.normal(
        key, (N, mu.shape[0]))


def ksd_of(block, mu_ref, Sig_ref, az, kernel):
    S = ksd.score_gaussian(block, mu_ref, Sig_ref)
    p1 = ksd.c_paper(az) if kernel == "imq" else ksd.median_heuristic(block)
    return ksd.ksd_stats(np.asarray(block), np.asarray(S), kernel, p1)["score_ksd"]


def null_q95(key, mu_ref, Sig_ref, az, kernel, N, reps):
    vals = [ksd_of(gauss_draws(k, mu_ref, Sig_ref, N), mu_ref, Sig_ref, az, kernel)
            for k in jax.random.split(key, reps)]
    return float(np.quantile(vals, 0.95)), float(np.median(vals))


def run(n, tilt_name, eps_s, eps_r_ladder, N, kernel, reps, tag, out):
    basis = make_basis(n)
    pk = make_pk(basis)
    Pz = jnp.asarray(grid_to_z(pk, basis))
    az = jnp.asarray(smoothing_operator(basis))
    y, _ = tilt.make_observation(jax.random.PRNGKey(Y_KEY), Pz, az, S_OBS)
    b = tilt.calibrate_b(Pz, az, y, target_shift=SHIFTS[tilt_name])
    mu_t, Sig_t = tilt.posterior_params(Pz, az, y, b)

    # the matched-wrong sampler: proper draws from the eps_s-contaminated target
    Pz_s = jnp.asarray(misspec.contaminated_pz(basis, pk, eps_s))
    mu_s, Sig_s = tilt.posterior_params(Pz_s, az, y, b)
    damage = float(jnp.sqrt(metrics.gaussian_w2sq(mu_s, Sig_s, mu_t, Sig_t)))
    floor = float(jnp.sqrt(metrics.gaussian_w2sq(mu_t, Sig_t, mu_t, Sig_t))) + 1e-12
    bank = gauss_draws(jax.random.PRNGKey(1), mu_s, Sig_s, 8192)

    fout = open(out, "a")
    print(f"[{tilt_name}] sampler eps_s={eps_s}  true W2 damage={damage:.3f} "
          f"(floor~0)  N={N} kernel={kernel}")
    for eps_r in eps_r_ladder:
        Pz_r = jnp.asarray(misspec.contaminated_pz(basis, pk, eps_r))
        mu_r, Sig_r = tilt.posterior_params(Pz_r, az, y, b)
        q95, med = null_q95(jax.random.PRNGKey(7), mu_r, Sig_r, az, kernel, N, reps)
        # detection POWER over disjoint N-blocks of the sampler bank
        B = max(1, bank.shape[0] // N)
        ratios = [ksd_of(bank[i * N:(i + 1) * N], mu_r, Sig_r, az, kernel) / (q95 + 1e-30)
                  for i in range(B)]
        power = float(np.mean([r > 1.0 for r in ratios]))
        ratio = float(np.median(ratios))
        # reference's own error (how far the reference target is from truth)
        ref_err = float(jnp.sqrt(metrics.gaussian_w2sq(mu_r, Sig_r, mu_t, Sig_t)))
        row = {"diag": "indepref", "tag": tag, "n": n, "d": int(Pz.shape[0]),
               "tilt": tilt_name, "eps_s": eps_s, "eps_r": eps_r,
               "kernel": kernel, "N": N, "n_blocks": B, "null_q95": q95,
               "ratio_q95_median": ratio, "power": power,
               "detect": bool(power >= 0.9),
               "sampler_damage_w2": damage, "ref_err_w2": ref_err,
               "confounded": bool(abs(eps_r - eps_s) < 1e-9)}
        fout.write(json.dumps(row) + "\n"); fout.flush()
        tags = " <-confounded" if row["confounded"] else (
            " <-true-score" if abs(eps_r) < 1e-9 else "")
        print(f"  eps_r={eps_r:+.2f} ref_err={ref_err:5.2f}: median ksd/q95={ratio:5.2f} "
              f"power={power:.2f}{tags}")
    fout.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tilt", default="mid")
    ap.add_argument("--eps-s", type=float, default=-0.3)
    ap.add_argument("--eps-r-ladder", default="-0.3,-0.2,-0.1,-0.05,0.0,0.1")
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--kernel", default="imq")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    run(args.n, args.tilt, args.eps_s,
        [float(x) for x in args.eps_r_ladder.split(",")],
        args.N, args.kernel, args.reps, args.tag, args.out)


if __name__ == "__main__":
    main()
