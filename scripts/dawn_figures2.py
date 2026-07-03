#!/usr/bin/env python
"""Arms-night dawn figures (plan section 8). CPU only.

  fig_a1_violins    (i)   multi-y W2/floor ratio distributions
  fig_a2_power      (ii)  detection rate vs budget per diagnostic,
                          compensation config highlighted
  fig_a3_contrast   (iii) amortized: summary checks vs geometry metrics
  fig_a4_remyK      (iv)  Remy W2(K) per shift + equal-NFE vs DPS
  fig_a2_scatter    (v)   diagnostic score vs true W2 across archives

Each figure regenerates from the JSONLs; per-archive true damage is computed
directly from the sample banks against the closed form (sidecar
figures/archive_damage.json). Skips gracefully if an input is missing.
"""
import collections
import json
import sys
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

SAMPLER_COLORS = {"dps": "#d62728", "sap": "#ff7f0e", "twisted": "#2ca02c",
                  "terminal_is": "#9467bd", "remy": "#1f77b4",
                  "ancestral": "#8c564b", "exact_guidance": "#7f7f7f"}


def rows_of(path):
    p = RES / path
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def dedupe(rows, keys):
    seen = {}
    for r in rows:
        seen[tuple(r.get(k) for k in keys)] = r
    return list(seen.values())


# ---------------------------------------------------------------- (i)
def fig_a1_violins():
    rows = rows_of("a1_multiy.jsonl")
    if not rows:
        return
    floor = collections.defaultdict(list)
    for r in rows:
        if r["sampler"] == "oracle":
            floor[(r["dim"], r["shift"], r["y_seed"])].append(r["w2"])
    floor = {k: np.median(v) for k, v in floor.items()}
    cell = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["sampler"] == "oracle" or r["dim"] != 64:
            continue
        f = floor.get((64, r["shift"], r["y_seed"]))
        if f:
            cell[r["sampler"]][r["shift"]].append(r["w2"] / f)
    shifts = [0.5, 1.0, 2.0, 4.0]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.19
    for i, (s, col) in enumerate([(s, SAMPLER_COLORS[s]) for s in
                                  ("twisted", "dps", "sap", "terminal_is")]):
        pos = np.arange(len(shifts)) + (i - 1.5) * width
        data = [cell[s][sh] for sh in shifts]
        vp = ax.violinplot(data, positions=pos, widths=width * 0.9,
                           showmedians=True)
        for body in vp["bodies"]:
            body.set_facecolor(col), body.set_alpha(0.6)
        for part in ("cmedians", "cbars", "cmins", "cmaxes"):
            vp[part].set_color(col)
        ax.plot([], [], color=col, label=s)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.axhline(3.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(shifts)), [f"{s}σ" for s in shifts])
    ax.set_xlabel("tilt strength (posterior-mean shift)")
    ax.set_ylabel("W2 / oracle floor")
    n_y = len({r["y_seed"] for r in rows if r["dim"] == 64})
    ax.set_title(f"Scheme bias is y-generic: ratios across {n_y} observation "
                 f"draws (64², N=256, T=64)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_a1_multiy_violins.png", dpi=180)
    plt.close(fig)
    print("fig (i) done")


# ---------------------------------------------------------------- (ii)
def fig_a2_power():
    rows = dedupe(rows_of("a2_power.jsonl"),
                  ("test", "config", "budget", "rep"))
    if not rows:
        return
    det = collections.defaultdict(list)
    for r in rows:
        if "detected" in r and r.get("tag") != "nullpad":
            det[(r["test"], r["config"], r["budget"])].append(r["detected"])
    panels = [("pqmass", "PQMass (p<0.05)"),
              ("tarp_cal", "TARP (empir. calibrated)"),
              ("mira_sym_cal", "MIRA (symmetric wrap, empir. calibrated)")]
    configs = ["dps", "sap", "twisted", "dps_em03", "dps_ep03",
               "twisted_em03", "dps_s05", "dps_s2", "oracle_null"]
    labels = {"dps_em03": "dps@ε=−0.3 (compensation)", "oracle_null": "oracle (null)"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (test, title) in zip(axes, panels):
        for c in configs:
            budgets = sorted({b for (t, cc, b) in det if t == test and cc == c})
            if not budgets:
                continue
            rates = [np.mean(det[(test, c, b)]) for b in budgets]
            emph = c == "dps_em03"
            ax.plot(budgets, rates, "o-", lw=2.5 if emph else 1.2,
                    ms=6 if emph else 4,
                    color="#d62728" if emph else None,
                    alpha=1.0 if emph else 0.65,
                    label=labels.get(c, c))
        ax.set_xscale("log")
        ax.axhline(0.05, color="k", ls=":", lw=0.8)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("sample budget")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("detection rate @ α=0.05")
    axes[-1].legend(fontsize=6.5, loc="center right")
    fig.suptitle("Certify the certifiers: power vs budget on failures of "
                 "exactly known size (64²)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig_a2_power_curves.png", dpi=180)
    plt.close(fig)
    print("fig (ii) done")


# ---------------------------------------------------------------- (iii)
def fig_a3_contrast():
    rows = rows_of("a3_amortized.jsonl")
    if not rows:
        return
    floor = collections.defaultdict(list)
    for r in rows:
        if r["sampler"] == "oracle":
            floor[r["y_seed"]].append(r["w2"])
    floor = {k: np.median(v) for k, v in floor.items()}
    by_ckpt = collections.defaultdict(list)
    for r in rows:
        if r["sampler"] == "ancestral" and r["y_seed"] in floor:
            by_ckpt[r["score"]].append(r)
    metrics_def = [("rel_mean_err", "posterior-mean rel. err", 1),
                   ("px_var_ratio", "pixel-variance ratio − 1", 2),
                   ("bp0_ratio", "band-power ratio − 1 (low k)", 2),
                   ("bp2_ratio", "band-power ratio − 1 (high k)", 2),
                   ("w2_floor", "W2 / oracle floor − 1", 3),
                   ("var_logmed", "|median log v/Σ*|", 3)]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    n_ck = len(by_ckpt)
    for i, (ck, rr) in enumerate(sorted(by_ckpt.items())):
        vals = []
        for key, _, _ in metrics_def:
            if key == "w2_floor":
                v = [r["w2"] / floor[r["y_seed"]] - 1 for r in rr]
            elif key == "var_logmed":
                v = [abs(r["var_ratio_logmed"]) for r in rr]
            elif key == "rel_mean_err":
                v = [r[key] for r in rr]
            else:
                v = [abs(r[key] - 1) for r in rr]
            vals.append(np.median(v))
        x = np.arange(len(metrics_def)) + (i - (n_ck - 1) / 2) * 0.8 / n_ck
        ax.bar(x, vals, width=0.75 / n_ck, label=ck)
    ax.set_yscale("log")
    ax.axhline(0.05, color="k", ls=":", lw=0.8, label="5% band")
    ax.set_xticks(range(len(metrics_def)),
                  [m[1] for m in metrics_def], rotation=20, ha="right",
                  fontsize=8)
    ax.set_ylabel("deviation (log scale)")
    ax.set_title("Amortized-conditional arm: summary checks vs geometry "
                 "(medians over y-draws × seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_a3_summary_vs_geometry.png", dpi=180)
    plt.close(fig)
    print("fig (iii) done")


# ---------------------------------------------------------------- (iv)
def fig_a4_remyK():
    rows = rows_of("a4_remy.jsonl")
    if not rows:
        return
    floor = collections.defaultdict(list)
    for r in rows:
        if r["sampler"] == "oracle":
            floor[(r["dim"], r["shift"])].append(r["w2"])
    floor = {k: np.median(v) for k, v in floor.items()}
    # DPS reference at matched config from the multi-y file (pinned-y T1
    # ratios are indistinguishable per checkpoint read #1)
    a1 = rows_of("a1_multiy.jsonl")
    dps_ref = collections.defaultdict(list)
    ofl = collections.defaultdict(list)
    for r in a1:
        if r["dim"] == 64:
            if r["sampler"] == "dps":
                dps_ref[r["shift"]].append(r["w2"])
            elif r["sampler"] == "oracle":
                ofl[(r["shift"], r["y_seed"])].append(r["w2"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    shifts = [0.5, 1.0, 2.0, 4.0]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(shifts)))
    for sh, col in zip(shifts, cmap):
        pts = collections.defaultdict(list)
        for r in rows:
            if (r["sampler"] == "remy" and r["dim"] == 64 and
                    r["shift"] == sh and r["eps"] == 0 and
                    r.get("eps0", 0.1) == 0.1):
                pts[r["K"]].append(r["w2"] / floor[(64, sh)])
        Ks = sorted(pts)
        med = [np.median(pts[k]) for k in Ks]
        ax.plot(Ks, med, "o-", color=col, label=f"{sh}σ")
    # DPS equal-NFE line: T-invariant single-pass, 64 evals ~ K=1 column
    for sh, col in zip(shifts, cmap):
        if dps_ref[sh]:
            fl = np.median([np.median(v) for k, v in ofl.items()
                            if k[0] == sh])
            ax.axhline(np.median(dps_ref[sh]) / fl, color=col, ls="--",
                       lw=0.9, alpha=0.7)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K (Langevin steps per level; NFE = 64·K)")
    ax.set_ylabel("W2 / oracle floor")
    ax.set_title("Remy-scheme error budget vs compute (64², N=256)\n"
                 "dashed: DPS single-pass (T-invariant, 64 NFE) at same tilt")
    ax.legend(title="tilt", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_a4_remy_K.png", dpi=180)
    plt.close(fig)
    print("fig (iv) done")


# ---------------------------------------------------------------- (v)
def archive_damage():
    """True W2 (and /floor) per unconditional archive, from the banks."""
    import jax
    import jax.numpy as jnp
    from tilt_audit import metrics as M, tilt
    from tilt_audit.fields import (grid_to_z, make_basis, make_pk,
                                   smoothing_operator)
    basis = make_basis(64)
    Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
    az = jnp.asarray(smoothing_operator(basis))
    y, _ = tilt.make_observation(jax.random.PRNGKey(999), Pz, az, 0.5)
    out = {}
    for f in sorted((RES / "archives").glob("*.npz")):
        if f.name.startswith("cond_"):
            continue
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        b = meta["b"]
        mu, Sig = tilt.posterior_params(Pz, az, y, b)
        z = jnp.asarray(d["z"], dtype=jnp.float64)
        m, v = M.weighted_moments(z, jnp.zeros(z.shape[0]))
        w2 = float(jnp.sqrt(M.gaussian_w2sq(m, v, mu, Sig)))
        # matched-size oracle floor
        fl = []
        for i in range(4):
            zo = mu + jnp.sqrt(Sig) * jax.random.normal(
                jax.random.PRNGKey(50 + i), z.shape)
            mo, vo = M.weighted_moments(zo, jnp.zeros(z.shape[0]))
            fl.append(float(jnp.sqrt(M.gaussian_w2sq(mo, vo, mu, Sig))))
        gs, _ = M.gamma_star(m, v, Pz, az, y, b)
        out[f.stem] = dict(w2=w2, floor=float(np.median(fl)),
                           ratio=w2 / float(np.median(fl)),
                           gamma_star=float(gs), **meta)
    (FIG / "archive_damage.json").write_text(json.dumps(out, indent=1))
    return out


def fig_a2_scatter():
    rows = dedupe(rows_of("a2_power.jsonl"),
                  ("test", "config", "budget", "rep"))
    if not rows:
        return
    dmg_path = FIG / "archive_damage.json"
    dmg = (json.loads(dmg_path.read_text()) if dmg_path.exists()
           else archive_damage())
    stat = collections.defaultdict(list)
    for r in rows:
        if r["test"] == "pqmass" and r["budget"] == 4096:
            stat[r["config"]].append(r["stat"])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for c, d in dmg.items():
        if c not in stat:
            continue
        p = np.maximum(np.asarray(stat[c]), 1e-300)
        x = d["ratio"]
        ylog = -np.log10(p)
        ax.errorbar(x, np.median(ylog),
                    yerr=[[np.median(ylog) - np.percentile(ylog, 25)],
                          [np.percentile(ylog, 75) - np.median(ylog)]],
                    fmt="o", ms=7,
                    color="#d62728" if c == "dps_em03" else "#1f77b4")
        ax.annotate(c, (x, np.median(ylog)), textcoords="offset points",
                    xytext=(6, 4), fontsize=7)
    ax.axhline(-np.log10(0.05), color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("true damage: W2 / oracle floor (4096-sample bank)")
    ax.set_ylabel("PQMass −log10 p @ budget 4096")
    ax.set_title("Diagnostic signal vs true posterior damage")
    fig.tight_layout()
    fig.savefig(FIG / "fig_a2_score_vs_damage.png", dpi=180)
    plt.close(fig)
    print("fig (v) done")


if __name__ == "__main__":
    which = sys.argv[1:] or ["i", "ii", "iii", "iv", "v"]
    if "i" in which:
        fig_a1_violins()
    if "ii" in which:
        fig_a2_power()
    if "iii" in which:
        fig_a3_contrast()
    if "iv" in which:
        fig_a4_remyK()
    if "v" in which:
        fig_a2_scatter()
