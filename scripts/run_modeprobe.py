#!/usr/bin/env python
"""Diagnostic A — mode-prober (P-20260705g). The constructive attack on the
project's flagship blind spot (barrier 1, locality: score-KSD is TOTALLY blind
to a missed mode).

Idea: a single-mode sampler's draws all sit in one basin. Launch short,
preconditioned Langevin EXCURSIONS from those draws, driven by the REFERENCE
score (exact mixture score here; learned-net arm later). If a runtime-cheap
excursion (K steps, K <= the sampler's own step count) climbs out and lands in
a basin the sampler never visited, we have a ground-truth-free missed-mode
alarm. This is the "sampling beyond high-density regions" move
(cf. arXiv:2507.05482) turned into a DETECTOR.

The physics sets the frontier. Between two components separated by Mahalanobis
Delta, the log-density barrier from mode to saddle is ~ Delta^2/8 (1-D along
the offset direction), so a Langevin walker crosses in ~ exp(Delta^2/(8 tau))
correlation times. Cheap excursions therefore find the mode below some
Delta*(K, tau) and fail above it. P-g: Delta* > 4 sigma, fails at 12 sigma.
Temperature tau > 1 targets pi^(1/tau) (flatter) and pushes Delta* out — the
tunable that trades detection reach against false crossings.

Excursion (preconditioned ULA, metric = prior covariance Pz, target pi^(1/tau)):
    z <- z + (eps/2) Pz (score(z)/tau) + sqrt(eps Pz) * noise

Detection: fraction of excursions ending in the OTHER basin (proj = z.dmu < 0
for a plus-only sampler). Crossed if that fraction exceeds a small threshold.
The plus-only sampler is the missed-mode failure; the 'both' sampler is the
false-crossing control (its draws already span both basins, so a crossing is
not a discovery). Rows -> results/modeprobe.jsonl (append; tag pilot|grid).
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tilt_audit import mixture
from tilt_audit.fields import grid_to_z, make_basis, make_pk

OUT_DEFAULT = "results/modeprobe.jsonl"
CROSS_FRAC = 0.01          # excursion "discovers" the mode if >1% land across


@functools.partial(jax.jit, static_argnums=(5,))
def excursions(key, z0, dmu, Pz, w, K, eps, tau):
    def step(z, k):
        s = mixture.score(z, dmu, Pz, w)
        noise = jax.random.normal(k, z.shape) * jnp.sqrt(eps * Pz)
        z = z + 0.5 * eps * Pz * (s / tau) + noise
        return z, None
    keys = jax.random.split(key, K)
    zK, _ = jax.lax.scan(step, z0, keys)
    return zK


def mahalanobis_sep(dmu, Pz):
    """Separation ||(+dmu)-(-dmu)||_C = 2 sqrt(sum dmu^2/Pz)."""
    return float(2.0 * np.sqrt(np.sum(np.asarray(dmu) ** 2 / np.asarray(Pz))))


def run(n, deltas, Ks, taus, eps, N, w, sampler_mode, tag, out):
    basis = make_basis(n)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    fout = open(out, "a")
    for delta in deltas:
        scale = delta / 4.0                       # dmu_k = scale*sqrt(Pz_k), 4 modes
        dmu = jnp.asarray(mixture.make_offset(Pz, basis, n_modes=4, scale=scale))
        sep = mahalanobis_sep(dmu, Pz)
        # sampler draws: 'plus' = missed-mode failure; 'both' = FP control
        z0 = mixture.sample(jax.random.PRNGKey(11), dmu, Pz, w, N, mode=sampler_mode)
        proj0 = np.asarray(jnp.sum(z0 * dmu, axis=1))
        base_other = float(np.mean(proj0 < 0))    # mass already across (control)
        for K in Ks:
            for tau in taus:
                zK = excursions(jax.random.PRNGKey(23), z0, dmu, Pz, w,
                                int(K), eps, float(tau))
                proj = np.asarray(jnp.sum(zK * dmu, axis=1))
                frac_other = float(np.mean(proj < 0))
                # discovery = crossing mass ABOVE what the sampler already had
                discovered = frac_other - base_other
                crossed = bool(discovered > CROSS_FRAC)
                row = {"diag": "modeprobe", "tag": tag, "n": n,
                       "sampler_mode": sampler_mode, "w": w,
                       "delta_nominal": delta, "sep": sep, "K": int(K),
                       "tau": float(tau), "eps": eps, "N": N,
                       "frac_other": frac_other, "base_other": base_other,
                       "discovered": discovered, "crossed": crossed}
                fout.write(json.dumps(row) + "\n"); fout.flush()
                print(f"  sep={sep:5.1f} K={int(K):>4} tau={tau:>3.0f}: "
                      f"frac_other={frac_other:.3f} (base {base_other:.3f}) "
                      f"discovered={discovered:+.3f} crossed={crossed}")
    fout.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seps", default="2,4,6,8,12",
                    help="target Mahalanobis separations (sigma)")
    ap.add_argument("--Ks", default="64,256")
    ap.add_argument("--taus", default="1,4")
    ap.add_argument("--eps", type=float, default=0.2)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--sampler-mode", default="plus",
                    choices=["plus", "both"])
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()
    run(args.n, [float(x) for x in args.seps.split(",")],
        [int(x) for x in args.Ks.split(",")],
        [float(x) for x in args.taus.split(",")],
        args.eps, args.N, args.w, args.sampler_mode, args.tag, args.out)


if __name__ == "__main__":
    main()
