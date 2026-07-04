#!/usr/bin/env python
"""Budget ladder to 16k (checkpoint-#1 zoom, rung 3): does KSD mode-blindness
persist at N=16384? Paired design: 3 independent (mix_plus, mix_both) bank
pairs at w=0.5; report plus/both statistic ratios (blind <=> ratio ~ 1).
No 60-null battery at this budget — the paired control IS the comparison
(logged design decision; a full null costs ~1 GPU-h for a foregone
conclusion at ratio 1.00)."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from tilt_audit import ksd, mixture
from tilt_audit.fields import grid_to_z, make_basis, make_pk, smoothing_operator


def main():
    basis = make_basis(64)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    az = jnp.asarray(smoothing_operator(basis))
    dmu = jnp.asarray(mixture.make_offset(Pz, basis))
    w = 0.5
    N = 16384

    def score_fn(X):
        return np.asarray(mixture.score(jnp.asarray(X), dmu, Pz, w))

    out = Path("results/mixture_contrast.jsonl")
    for rep in range(3):
        rows = {}
        for mode in ("plus", "both"):
            k = jax.random.fold_in(jax.random.PRNGKey(161616),
                                   hash((mode, rep)) & 0x7FFFFFFF)
            X = np.asarray(mixture.sample(k, dmu, Pz, w, N, mode),
                           dtype=np.float64)
            t0 = time.time()
            st = ksd.ksd_stats(X, score_fn(X), "imq", ksd.c_paper(az), -0.5)
            rows[mode] = st["score_ksd"]
            with out.open("a") as f:
                f.write(json.dumps(dict(
                    test="ksd_imq_paper", w=w, config=f"mix_{mode}",
                    budget=N, rep=rep, stat=st["score_ksd"],
                    tag="exploratory_ckpt1_16k",
                    wall=round(time.time() - t0, 1))) + "\n")
        print(f"[ksd16k] rep {rep}: plus/both = "
              f"{rows['plus']/rows['both']:.5f} "
              f"(plus={rows['plus']:.5f} both={rows['both']:.5f})",
              flush=True)


if __name__ == "__main__":
    main()
