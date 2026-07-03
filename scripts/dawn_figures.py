#!/usr/bin/env python
"""Dawn deliverable figures (working-quality; /polish later for anything public).

(i)   money plot: W2 vs tilt strength per scheme, panel per dim, floor band
(ii)  decomposition bars at 64^2: scheme bias / kernel choice / score error /
      misspecification, per scheme (needs t2 rows; skips gracefully if absent)
(iii) certificate panel: log Zhat - log Z per certificate-bearing arm + SAP/
      twisted-potential ESS summaries vs d
(iv)  B1 alpha sweep: AUROC + selected-accuracy vs alpha (defensive)
(v)   B2 twin: AUROC/acc per method (whatever rows exist)
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PR = Path("/home/tersenov/software/particle-reasoners")
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def load(patterns, root=ROOT / "results"):
    rows = []
    for pat in patterns:
        for path in glob.glob(str(root / pat)):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
    return pd.DataFrame(rows)


def auroc(conf, correct):
    conf, correct = np.asarray(conf, float), np.asarray(correct, int)
    pos, neg = conf[correct == 1], conf[correct == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq


def fig_money():
    df = load(["t1_core.jsonl", "t1_controls.jsonl", "t3_seeds.jsonl",
               "t1_weak.jsonl"])
    df = df[df["T"] == 64] if "T" in df.columns else df
    med = (df.groupby(["dim", "shift", "N", "sampler"]).w2.median()
             .reset_index())
    for N in (64, 256):
        sub = med[med.N == N]
        dims = sorted(sub.dim.unique())
        fig, axes = plt.subplots(1, len(dims), figsize=(4.2 * len(dims), 3.6),
                                 squeeze=False, sharey=True)
        order = ["dps", "sap", "terminal_is", "twisted", "twisted_potential",
                 "exact_guidance", "oracle"]
        colors = plt.cm.tab10(np.linspace(0, 1, len(order)))
        for ax, dim in zip(axes[0], dims):
            for s, c in zip(order, colors):
                g = sub[(sub.dim == dim) & (sub.sampler == s)].sort_values("shift")
                if len(g):
                    style = dict(color=c, marker="o", lw=1.6)
                    if s == "oracle":
                        style.update(ls="--", marker=None, color="k", lw=1.2)
                    ax.plot(g["shift"], g.w2, label=s, **style)
            ax.set(xscale="log", yscale="log",
                   xlabel="tilt strength (prior sigma units)",
                   title=f"d = {dim}x{dim}")
            ax.grid(alpha=0.3, which="both")
        axes[0][0].set_ylabel(f"W2 to exact tilt (N={N})")
        axes[0][-1].legend(fontsize=7, loc="upper left")
        fig.suptitle("Guided-sampler error vs oracle floor (exact scores, T=64)",
                     y=1.02, fontsize=11)
        fig.tight_layout()
        fig.savefig(FIG / f"fig_money_w2_N{N}.png", dpi=130,
                    bbox_inches="tight")
        plt.close(fig)
    print("money plots written")


def fig_decomposition():
    exact = load(["t1_core.jsonl"])
    t2 = load(["t2_learned.jsonl", "t2_pathway_ctrl.jsonl"])
    mis = load(["t1_misspec.jsonl"])
    if t2.empty:
        print("decomposition: no t2 rows yet — skipped")
        return
    frames = [exact.assign(arm="exact"), t2.assign(arm=t2.score)]
    if not mis.empty:
        frames.append(mis.assign(arm="exact-mis" + mis.eps.astype(str)))
    allr = pd.concat(frames, ignore_index=True)
    sel = allr[(allr.dim == 64) & (allr.N == 256) & (allr["shift"].isin([1.0, 4.0]))]
    samplers = ["dps", "sap", "twisted", "terminal_is"]
    arms = [a for a in ["exact", "pathway:analytic", "learned:clean",
                        "learned:mis+0.3", "learned:mis-0.3",
                        "exact-mis0.3", "exact-mis-0.3"]
            if a in set(sel.arm)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, shift in zip(axes, [1.0, 4.0]):
        ss = sel[sel["shift"] == shift]
        x = np.arange(len(samplers))
        w = 0.8 / max(len(arms), 1)
        for i, arm in enumerate(arms):
            vals = [ss[(ss.sampler == s) & (ss.arm == arm)].w2.median()
                    for s in samplers]
            ax.bar(x + i * w, vals, w, label=arm)
        floor = ss[ss.sampler == "oracle"].w2.median() if "oracle" in set(ss.sampler) else np.nan
        if np.isfinite(floor):
            ax.axhline(floor, color="k", ls="--", lw=1, label="oracle floor")
        ax.set(yscale="log", xticks=x + 0.4, xticklabels=samplers,
               title=f"shift = {shift} sigma")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("W2 to exact tilt (64x64, N=256)")
    axes[1].legend(fontsize=7)
    fig.suptitle("Error decomposition: scheme / kernel / score error / misspec")
    fig.tight_layout()
    fig.savefig(FIG / "fig_decomposition.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("decomposition written")


def fig_certificate():
    df = load(["t1_core.jsonl", "t1_controls.jsonl"])
    has_z = df.dropna(subset=["log_z_est"]) if "log_z_est" in df else pd.DataFrame()
    if has_z.empty:
        print("certificate: no logZ rows")
        return
    has_z = has_z.assign(zerr=has_z.log_z_est - has_z.log_z_analytic)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for s, m in [("twisted", "o"), ("terminal_is", "s"),
                 ("twisted_potential", "^")]:
        g = has_z[(has_z.sampler == s) & (has_z.N == 256)]
        if g.empty:
            continue
        med = g.groupby(["dim", "shift"]).zerr.median().reset_index()
        for shift, grp in med.groupby("shift"):
            axes[0].plot(grp.dim**2, np.abs(grp.zerr) + 1e-16, m + "-",
                         alpha=0.6, label=f"{s} shift={shift}" if shift in (1.0,) else None)
    axes[0].set(xscale="log", yscale="log", xlabel="d",
                ylabel="|log Zhat - log Z|", title="certificate error vs d (N=256)")
    axes[0].legend(fontsize=6)
    axes[0].grid(alpha=0.3, which="both")
    ess = df[df.sampler.isin(["sap", "twisted_potential", "twisted"])]
    if "ess_traj_min" in ess:
        med = (ess.groupby(["sampler", "dim"]).ess_traj_min.median()
                  .reset_index())
        for s, grp in med.groupby("sampler"):
            axes[1].plot(grp.dim**2, grp.ess_traj_min, "o-", label=s)
    axes[1].set(xscale="log", xlabel="d", ylabel="min ESS along trajectory",
                title="weight degeneracy vs d (N=256)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_certificate.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("certificate written")


def _b_metrics(df, conf_col="confidence", corr_col="modal_correct"):
    return pd.Series({
        "auroc": auroc(df[conf_col], df[corr_col]),
        "acc_selected": df["selected_correct"].mean(),
        "n": len(df),
    })


def fig_b1():
    rows = []
    for path in sorted(glob.glob(str(PR / "results/tables/reliability_alpha_*.jsonl"))):
        alpha = float(Path(path).stem.split("_")[-1])
        df = pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])
        if df.empty:
            continue
        m = _b_metrics(df)
        m["alpha"] = alpha
        rows.append(m)
    if not rows:
        print("b1: nothing")
        return
    t = pd.DataFrame(rows).sort_values("alpha")
    fig, ax1 = plt.subplots(figsize=(6, 3.8))
    ax1.plot(t.alpha, t.auroc, "o-", color="C0", label="AUROC (insurance conf.)")
    ax1.set(xlabel="defensive fraction alpha", ylabel="AUROC", ylim=(0.4, 1.0))
    ax1.axhline(0.65, color="C0", ls=":", lw=1)
    ax2 = ax1.twinx()
    ax2.plot(t.alpha, t.acc_selected, "s-", color="C3", label="selected accuracy")
    ax2.set_ylabel("accuracy")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="lower right")
    for _, r in t.iterrows():
        ax1.annotate(f"n={int(r['n'])}", (r.alpha, r.auroc), fontsize=6,
                     textcoords="offset points", xytext=(0, 6))
    ax1.grid(alpha=0.3)
    ax1.set_title("B1: defensive-mixture alpha sweep (E-20260702a)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_b1_alpha.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("b1 written:", t[["alpha", "auroc", "acc_selected", "n"]].to_string(index=False))


def fig_b2():
    dfs = []
    for path in sorted(glob.glob(str(PR / "results/tables/reliability_r1_twin_s*.jsonl"))):
        dfs.append(pd.DataFrame([json.loads(l) for l in open(path) if l.strip()]))
    if not dfs:
        print("b2: nothing")
        return
    df = pd.concat(dfs, ignore_index=True)
    t = df.groupby("method").apply(_b_metrics, include_groups=False).reset_index()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(t))
    ax.bar(x - 0.2, t.auroc, 0.38, label="AUROC")
    ax.bar(x + 0.2, t.acc_selected, 0.38, label="selected acc")
    ax.set_xticks(x, t.method)
    for xi, r in zip(x, t.itertuples()):
        ax.annotate(f"n={int(r.n)}", (xi, max(r.auroc, r.acc_selected) + 0.02),
                    ha="center", fontsize=7)
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set(ylim=(0, 1.05), title="B2: R1-distill twin (E-20260702b) — partial N")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig_b2_twin.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("b2 written:", t.to_string(index=False))


if __name__ == "__main__":
    fig_money()
    fig_decomposition()
    fig_certificate()
    fig_b1()
    fig_b2()
