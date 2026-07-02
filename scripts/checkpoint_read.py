#!/usr/bin/env python
"""Checkpoint read: digest results/*.jsonl into the steering table + quick plots.

Prints per (dim, shift): oracle floor (median W2 over seeds at each N), each
scheme's median W2 and its ratio to the floor, gamma*, and the kill-criterion
verdict (any scheme > 3x floor anywhere?). Saves figures/checkpoint_*.png.
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load(patterns):
    rows = []
    for pat in patterns:
        for path in glob.glob(str(ROOT / "results" / pat)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    df = load(sys.argv[1:] or ["t1_core.jsonl", "t1_controls.jsonl",
                               "t3_seeds.jsonl", "t4_densify.jsonl"])
    if df.empty:
        print("no rows yet")
        return
    med = (df.groupby(["dim", "shift", "N", "sampler"])
             .agg(w2=("w2", "median"), kl=("kl", "median"),
                  gs=("gamma_star", "median"), gm=("gamma_mean", "median"),
                  vr=("var_ratio_logmed", "median"),
                  cov68=("cov_mean_68", "median"), nrow=("w2", "size"))
             .reset_index())
    oracle = med[med.sampler == "oracle"][["dim", "shift", "N", "w2"]].rename(
        columns={"w2": "floor"})
    tab = med.merge(oracle, on=["dim", "shift", "N"])
    tab["ratio"] = tab.w2 / tab.floor

    print("\n=== W2 / oracle-floor ratios (median over seeds) ===")
    print(f"{'dim':>4} {'shift':>5} {'N':>4} | " + " | ".join(
        f"{s:>14}" for s in sorted(tab.sampler.unique()) if s != "oracle"))
    for (dim, shift, N), g in tab.groupby(["dim", "shift", "N"]):
        cells = []
        for s in sorted(tab.sampler.unique()):
            if s == "oracle":
                continue
            r = g[g.sampler == s]
            cells.append(f"{r.ratio.iloc[0]:>7.2f}x g{r.gs.iloc[0]:>6.2f}"
                         if len(r) else " " * 14)
        print(f"{dim:>4} {shift:>5} {N:>4} | " + " | ".join(cells))

    # kill criterion: schemes = the frozen four, max ratio anywhere
    frozen = tab[tab.sampler.isin(["dps", "sap", "twisted", "terminal_is"])]
    mx = frozen.loc[frozen.ratio.idxmax()]
    print(f"\nKILL CRITERION: max scheme W2/floor = {mx.ratio:.1f}x "
          f"({mx.sampler} @ dim={mx.dim} shift={mx['shift']} N={mx.N}) "
          f"-> {'NO-GO signal' if mx.ratio < 3 else 'bias EXISTS (GO-side)'}")

    # money plot: W2 vs shift per scheme, one panel per dim, N=256
    for N in sorted(tab.N.unique()):
        sub = tab[tab.N == N]
        dims = sorted(sub.dim.unique())
        fig, axes = plt.subplots(1, len(dims), figsize=(4 * len(dims), 3.4),
                                 squeeze=False)
        for ax, dim in zip(axes[0], dims):
            for s in sorted(sub.sampler.unique()):
                g = sub[(sub.dim == dim) & (sub.sampler == s)].sort_values("shift")
                if len(g):
                    ax.plot(g["shift"], g.w2, "o-", label=s)
            ax.set(xscale="log", yscale="log", xlabel="tilt strength (sigma)",
                   title=f"{dim}x{dim}, N={N}")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("W2 to sigma")
        axes[0][-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(ROOT / "figures" / f"checkpoint_w2_N{N}.png", dpi=110)
        plt.close(fig)
    print(f"\nfigures/checkpoint_w2_N*.png written; "
          f"{len(df)} rows total across {df.sampler.nunique()} samplers")


if __name__ == "__main__":
    main()
