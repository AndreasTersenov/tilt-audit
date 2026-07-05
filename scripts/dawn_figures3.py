#!/usr/bin/env python
"""Certifier-night dawn figures (plan §8). CPU only; regenerates from JSONLs;
skips gracefully if an input is missing.

  fig_ksd_power      (i)   KSD detection vs budget per archive, compensation
                           highlighted; PQMass/TARP power alongside (a2 data)
  fig_mixture        (ii)  the missed-mode confession: KSD vs PQMass vs TARP
                           on mix_plus across budgets + weight ladder
  fig_wrongref       (iii) true-ref vs wrong-ref detection matrix (the false
                           certification + false alarm structure)
  fig_transfer       (iv)  sampler damage vs gold on the lognormal substrate
                           (mmd2/floor), mid+strong, vs Gaussian W2/floor
  fig_remyK          (v)   Remy MMD(K) on the nonlinear substrate
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

BUDGETS = [64, 256, 1024, 4096]


def rows_of(path):
    p = RES / path
    if not p.exists():
        return []
    return [json.loads(x) for x in p.open()]


def det_rate(rows, **match):
    sel = [r for r in rows
           if all(r.get(k) == v for k, v in match.items())]
    if not sel:
        return np.nan
    return float(np.mean([r["detect"] if "detect" in r else r["detected"]
                          for r in sel]))


def fig_ksd_power():
    ksd = [r for r in rows_of("ksd_trial.jsonl") if r.get("arm") == "power"
           and r.get("kernel") == "imq_paper"]
    a2 = rows_of("a2_power.jsonl")
    if not ksd:
        return print("skip fig_ksd_power")
    configs = ["dps", "sap", "dps_s05", "dps_s2", "dps_ep03", "dps_em03",
               "twisted_em03", "twisted", "oracle_null"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    panels = [("score-KSD (true score)", ksd, "config", "detect"),
              ("PQMass", [r for r in a2 if r["test"] == "pqmass"],
               "config", "detected"),
              ("TARP (wrapped)", [r for r in a2 if r["test"] == "tarp_cal"],
               "config", "detected")]
    if not [r for r in a2 if r.get("test") == "tarp_cal"]:
        panels[2] = ("TARP", [r for r in a2 if r["test"] == "tarp"],
                     "config", "detected")
    for ax, (title, rr, ckey, dkey) in zip(axes, panels):
        for cfg in configs:
            xs, ys = [], []
            for b in BUDGETS:
                sel = [r for r in rr if r.get(ckey) == cfg
                       and r.get("budget") == b
                       and r.get(dkey) is not None]
                if sel:
                    xs.append(b)
                    ys.append(np.mean([r[dkey] for r in sel]))
            if not xs:
                continue
            lw = 3.0 if cfg == "dps_em03" else 1.4
            color = "#d62728" if cfg == "dps_em03" else None
            zorder = 5 if cfg == "dps_em03" else 2
            ls = ":" if cfg == "oracle_null" else "-"
            ax.plot(xs, ys, "o-", lw=lw, color=color, ls=ls, ms=4,
                    zorder=zorder, label=cfg)
        ax.set_xscale("log")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.05, color="gray", lw=0.6, ls="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("budget N")
    axes[0].set_ylabel("detection rate (empirical alpha=0.05)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Score-KSD with the true score is loud on scheme bias — "
                 "including the compensation config (red)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ksd_power.png", dpi=150)
    print("wrote fig_ksd_power.png")


def fig_mixture():
    mix = [r for r in rows_of("ksd_trial.jsonl") if r.get("arm") == "mixture"
           and r.get("kernel") == "imq_paper" and r.get("w") == 0.5]
    con = rows_of("mixture_contrast.jsonl")
    if not mix and not con:
        return print("skip fig_mixture")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for cfg, color in (("mix_plus", "#d62728"), ("mix_swapped", "#ff7f0e"),
                       ("mix_both", "#7f7f7f")):
        xs, ys = [], []
        for b in BUDGETS + [16384]:
            sel = [r for r in mix if r["config"] == cfg
                   and r["budget"] == b]
            sel += [r for r in con if r.get("test") == "ksd_imq_paper"
                    and r.get("w") == 0.5 and r["config"] == cfg
                    and r["budget"] == b]
            vals = [r.get("detect", r.get("detected")) for r in sel]
            vals = [v for v in vals if v is not None]
            if vals:
                xs.append(b)
                ys.append(np.mean(vals))
        ax.plot(xs, ys, "o-", color=color, label=f"KSD: {cfg}")
    for test, marker in (("pqmass", "s"), ("tarp", "^")):
        xs, ys = [], []
        for b in BUDGETS:
            sel = [r for r in con if r.get("test") == test
                   and r.get("w") == 0.5 and r["config"] == "mix_plus"
                   and r["budget"] == b]
            if sel:
                xs.append(b)
                ys.append(np.mean([r["detected"] for r in sel]))
        if xs:
            ax.plot(xs, ys, marker + "--", color="#2ca02c" if
                    test == "pqmass" else "#1f77b4",
                    label=f"{test}: mix_plus")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.05, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("budget N")
    ax.set_ylabel("detection rate")
    ax.set_title("w=0.5: half the posterior missing (12$\\sigma$ apart)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ws = sorted({r["w"] for r in con})
    for test, color in (("ksd_imq_paper", "#d62728"),
                        ("pqmass", "#2ca02c"), ("tarp", "#1f77b4")):
        xs, ys = [], []
        for w in ws:
            sel = [r for r in con if r.get("test") == test
                   and r.get("w") == w and r["config"] == "mix_plus"
                   and r["budget"] == 1024]
            if sel:
                xs.append(1 - w)
                ys.append(np.mean([r["detected"] for r in sel]))
        if xs:
            ax.plot(xs, ys, "o-", color=color, label=test)
    ax.set_xscale("log")
    ax.set_xlabel("missing-mode weight (1-w)")
    ax.set_ylabel("detection rate at N=1024")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("weight ladder: where sample-side detection dies")
    ax.legend(fontsize=8)
    fig.suptitle("The missed-mode confession: score-KSD cannot see a missing "
                 "mode; sample-space tests can", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig_mixture.png", dpi=150)
    print("wrote fig_mixture.png")


def fig_wrongref():
    ksd = rows_of("ksd_trial.jsonl")
    power = [r for r in ksd if r.get("arm") == "power"
             and r.get("kernel") == "imq_paper"]
    wref = [r for r in ksd if r.get("arm") == "wrongref"
            and r.get("kernel") == "imq_paper"]
    dep = [r for r in ksd if r.get("arm") == "deployment"
           and r.get("kernel") == "imq_paper"]
    if not wref:
        return print("skip fig_wrongref")
    configs = ["oracle_null", "dps", "dps_em03", "twisted_em03"]
    cols = [("true score\n(analytic)", power, {}),
            ("wrong ref\n(analytic eps=-0.3)", wref, {"eps_ref": -0.3}),
            ("net s_clean\n(deployment)", dep, {"net": "s_clean"}),
            ("net s_mis_m03\n(deployment)", dep, {"net": "s_mis_m03"})]
    cols = [(t, rr, m) for t, rr, m in cols if rr]
    M = np.full((len(configs), len(cols)), np.nan)
    for j, (_, rr, match) in enumerate(cols):
        for i, cfg in enumerate(configs):
            M[i, j] = det_rate(rr, config=cfg, budget=1024, **match)
    fig, ax = plt.subplots(figsize=(1.9 + 1.6 * len(cols), 4))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        fontsize=11,
                        color="white" if 0.25 < M[i, j] < 0.8 else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], fontsize=9)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=10)
    ax.set_title("Detection rate at N=1024 by reference score\n"
                 "(false certification = green where the row is damaged; "
                 "false alarm = red on oracle_null)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_wrongref.png", dpi=150)
    print("wrote fig_wrongref.png")


def fig_transfer():
    tra = [r for r in rows_of("transfer.jsonl") if r.get("n") == 64]
    if not tra:
        return print("skip fig_transfer")
    samplers = ["gold_floor", "dps", "dps_inflated", "remy5", "remy30",
                "remy100", "terminal_is", "dps_ln_clean", "dps_ln_mis"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, tiltname in zip(axes, ("mid", "strong")):
        floor = np.median([r["mmd2"] for r in tra
                           if r["tilt"] == tiltname
                           and r["sampler"] == "gold_floor"]) or 1e-12
        xs, ys, labels = [], [], []
        for i, s in enumerate(samplers):
            sel = [r["mmd2"] for r in tra if r["tilt"] == tiltname
                   and r["sampler"] == s]
            if sel:
                xs.append(i)
                ys.append(np.median(sel) / floor)
                labels.append(s)
        ax.bar(range(len(ys)), ys,
               color=["#7f7f7f" if l == "gold_floor" else "#1f77b4"
                      if l.startswith("remy") else "#d62728"
                      if l.startswith("dps") else "#9467bd" for l in labels])
        ax.set_yscale("log")
        ax.axhline(1.0, color="k", lw=0.7)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"tilt: {tiltname}")
    axes[0].set_ylabel("median MMD$^2$ / gold-floor MMD$^2$")
    fig.suptitle("Sampler damage vs MCMC gold on the lognormal substrate "
                 "(64$^2$)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig_transfer.png", dpi=150)
    print("wrote fig_transfer.png")


def fig_remyK():
    tra = [r for r in rows_of("transfer.jsonl")
           if r.get("n") == 64 and str(r.get("sampler", "")).startswith("remy")]
    if not tra:
        return print("skip fig_remyK")
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for tiltname, color in (("mid", "#1f77b4"), ("strong", "#d62728")):
        byK = defaultdict(list)
        for r in tra:
            if r["tilt"] == tiltname:
                byK[r["K"]].append(r["mmd2"])
        Ks = sorted(byK)
        med = [np.median(byK[k]) for k in Ks]
        lo = [np.quantile(byK[k], 0.25) for k in Ks]
        hi = [np.quantile(byK[k], 0.75) for k in Ks]
        ax.plot(Ks, med, "o-", color=color, label=f"tilt {tiltname}")
        ax.fill_between(Ks, lo, hi, color=color, alpha=0.2)
    for r in rows_of("transfer.jsonl"):
        pass
    fl = [r["mmd2"] for r in rows_of("transfer.jsonl")
          if r.get("n") == 64 and r.get("sampler") == "gold_floor"]
    if fl:
        ax.axhline(np.median(fl), color="gray", ls="--", lw=1,
                   label="gold floor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Langevin steps per level K")
    ax.set_ylabel("MMD$^2$ to gold")
    ax.set_title("Remy K-convergence on the nonlinear substrate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig_remyK.png", dpi=150)
    print("wrote fig_remyK.png")


if __name__ == "__main__":
    fig_ksd_power()
    fig_mixture()
    fig_wrongref()
    fig_transfer()
    fig_remyK()
