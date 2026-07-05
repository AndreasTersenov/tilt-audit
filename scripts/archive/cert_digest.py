#!/usr/bin/env python
"""Kill-test digest: verdict readouts for P-20260704a-c + kill criterion.

Reads results/cert_killtest.jsonl; prints the decision tables; writes
figures/cert_killtest.png (4 panels). Exact chain-law columns are the
ground-truth axes (zero MC noise); sampled instruments are judged against
them.
"""
import collections
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in open(ROOT / "results/cert_killtest.jsonl")]
seen = {}
for r in rows:
    seen[(r["dim"], r["y_seed"], r["shift"], r["mode"], r["N"], r["seed"],
          r["eps"])] = r
rows = list(seen.values())
core = [r for r in rows if r["eps"] == 0 and r["y_seed"] == 999]


def med(vals):
    return float(np.median(vals)) if vals else float("nan")


cell = collections.defaultdict(list)
for r in core:
    cell[(r["dim"], r["shift"], r["mode"], r["N"])].append(r)

# ---------------------------------------------------------------- table 1
print("=" * 100)
print("T1. Exact bound anatomy (zero-noise): path KL vs endpoint KL, per dim/shift/mode")
print(f"{'dim':>4} {'shift':>5} {'mode':>15} {'pathKL*':>10} {'endKL*':>10} "
      f"{'tightness':>9} {'w2_end*':>8}")
for (dim, shift, mode, N), rr in sorted(cell.items()):
    if N != 256 or mode == "unguided":
        continue
    r = rr[0]
    print(f"{dim:>4} {shift:>5} {mode:>15} {r['kl_path_exact']:>10.3g} "
          f"{r['kl_end_exact']:>10.3g} "
          f"{r['kl_path_exact']/max(r['kl_end_exact'],1e-12):>9.2f} "
          f"{r['w2_end_exact']:>8.3g}")

# ---------------------------------------------------------------- P-a
print("=" * 100)
print("T2. P-20260704a: ESS + khat at N=256; KLhat monotone in true damage?")
for dim in sorted({r["dim"] for r in core}):
    line = f"dim {dim:>3}: "
    ess_all, mono_pairs = [], []
    per_shift = []
    for shift in (0.5, 1.0, 2.0, 4.0):
        rr = cell.get((dim, shift, "dps", 256), [])
        if not rr:
            continue
        ess = med([r["ess_res"] for r in rr])
        klh = med([r["kl_path_hat"] for r in rr])
        kh = med([r["khat"] for r in rr])
        per_shift.append((shift, ess, klh, kh, rr[0]["kl_end_exact"]))
        ess_all.append(ess)
    mono = all(per_shift[i + 1][2] > per_shift[i][2]
               for i in range(len(per_shift) - 1))
    # contrast: eg must certify below dps at every shift
    contrast = all(
        med([r["kl_path_hat"] for r in cell.get((dim, s, "exact_guidance"),
            cell.get((dim, s, "exact_guidance", 256), []))]) <
        med([r["kl_path_hat"] for r in cell.get((dim, s, "dps", 256), [])])
        for s in (0.5, 1.0, 2.0, 4.0)
        if cell.get((dim, s, "dps", 256)))
    line += " | ".join(f"{s}σ: ESS={e:.1f} KLhat={k:.3g} khat={kh:.1f}"
                       for s, e, k, kh, _ in per_shift)
    print(line)
    print(f"        KLhat monotone in damage: {mono} | "
          f"eg<dps contrast at all shifts: {contrast}")

# ---------------------------------------------------------------- P-b
print("=" * 100)
print("T3. P-20260704b: repair at 16^2/1σ (dps) — raw vs repaired W2")
rr = cell.get((16, 1.0, "dps", 256), [])
if rr:
    raw = med([r["raw_w2"] for r in rr])
    rep = med([r["rep_w2"] for r in rr])
    print(f"raw W2 = {raw:.4g}, repaired W2 = {rep:.4g} "
          f"(need rep <= raw/2 = {raw/2:.4g}) -> "
          f"{'HIT' if rep <= raw / 2 else 'MISS'}")

# ---------------------------------------------------------------- P-c
print("=" * 100)
print("T4. P-20260704c: rank-corr(KLhat, true endpoint KL), exact-score grid")
xs = [r["kl_path_hat"] for r in core if r["mode"] != "unguided" and r["N"] == 256]
ys = [r["kl_end_exact"] for r in core if r["mode"] != "unguided" and r["N"] == 256]
rho = stats.spearmanr(xs, ys).statistic
print(f"pooled Spearman rho = {rho:.3f} over {len(xs)} rows "
      f"(need >= 0.9) -> {'HIT' if rho >= 0.9 else 'MISS'}")
for dim in sorted({r["dim"] for r in core}):
    sub = [(r["kl_path_hat"], r["kl_end_exact"]) for r in core
           if r["dim"] == dim and r["mode"] != "unguided" and r["N"] == 256]
    if len(sub) > 8:
        rho_d = stats.spearmanr(*zip(*sub)).statistic
        print(f"  dim {dim}: rho = {rho_d:.3f} (n={len(sub)})")

# ---------------------------------------------------------------- rescue
print("=" * 100)
print("T5. Kill-criterion rescue clause: per-mode (Rao-Blackwell) instruments at 64^2")
for shift in (0.5, 1.0, 2.0, 4.0):
    for mode in ("dps", "exact_guidance"):
        rr = [r for r in cell.get((64, shift, mode, 256), [])
              if "kl_modes_sum" in r]
        if rr:
            print(f"  {mode:>15} {shift}σ: sum per-mode KLhat = "
                  f"{med([r['kl_modes_sum'] for r in rr]):.4g} "
                  f"(exact path {rr[0]['kl_path_exact']:.4g}, end "
                  f"{rr[0]['kl_end_exact']:.4g}) | per-mode ESS med = "
                  f"{med([r['ess_modes_med'] for r in rr]):.0f}/256")

# ---------------------------------------------------------------- misspec + multi-y
mis = [r for r in rows if r["eps"] != 0]
if mis:
    print("=" * 100)
    print("T6. Attribution scope (misspec column): certificate sees steering, not model")
    for eps in sorted({r["eps"] for r in mis}):
        rr = [r for r in mis if r["eps"] == eps]
        print(f"  eps={eps:+.1f}: KLhat={med([r['kl_path_hat'] for r in rr]):.4g} "
              f"(model-side path* {rr[0]['kl_path_exact']:.4g}, model-side end* "
              f"{rr[0].get('kl_end_model', float('nan')):.4g}) vs TRUE end* "
              f"{rr[0]['kl_end_exact']:.4g}")
my = [r for r in rows if r["y_seed"] != 999]
if my:
    per_y = collections.defaultdict(list)
    for r in my:
        per_y[r["y_seed"]].append(r["kl_path_hat"])
    v = [med(vv) for vv in per_y.values()]
    print(f"T6b. multi-y stability of KLhat (dps 64^2/1σ): "
          f"{[round(x, 1) for x in v]}")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
dims = sorted({r["dim"] for r in core})
cmap = dict(zip(dims, plt.cm.viridis(np.linspace(0.1, 0.85, len(dims)))))
ax = axes[0]
for dim in dims:
    pts = [(r["kl_end_exact"], r["kl_path_hat"]) for r in core
           if r["dim"] == dim and r["mode"] != "unguided" and r["N"] == 256]
    if pts:
        x, yv = zip(*pts)
        ax.loglog(x, yv, "o", ms=3, alpha=0.5, color=cmap[dim],
                  label=f"{dim}²")
lims = [1e-1, 1e5]
ax.loglog(lims, lims, "k:", lw=0.8)
ax.set(xlabel="true endpoint KL (exact)", ylabel="KL̂ (sampled certificate)",
       title="instrument vs truth")
ax.legend(fontsize=7)
ax = axes[1]
for dim in dims:
    per = [(s, med([r["ess_res"] for r in cell.get((dim, s, "dps", 256), [])]))
           for s in (0.5, 1.0, 2.0, 4.0)]
    per = [(s, e) for s, e in per if not np.isnan(e)]
    ax.semilogy([s for s, _ in per], [e for _, e in per], "o-",
                color=cmap[dim], label=f"{dim}²")
ax.set(xlabel="tilt strength", ylabel="residual ESS / 256",
       title="repair affordability (dps)")
ax = axes[2]
rr = [r for r in rows if r["dim"] == 64 and r["shift"] in (1.0, 4.0)
      and r["mode"] == "dps" and r["eps"] == 0 and r["y_seed"] == 999]
per_n = collections.defaultdict(list)
for r in rr:
    per_n[(r["shift"], r["N"])].append(r["kl_path_hat"])
for shift, mk in ((1.0, "o-"), (4.0, "s--")):
    ns = sorted({n for s, n in per_n if s == shift})
    if ns:
        ax.loglog(ns, [med(per_n[(shift, n)]) for n in ns], mk,
                  label=f"{shift}σ (exact {rr[0]['kl_path_exact']:.0f})")
ax.set(xlabel="N (particles)", ylabel="KL̂",
       title="estimator vs N (64², dps)")
ax.legend(fontsize=7)
ax = axes[3]
for mode, col in (("dps", "#d62728"), ("exact_guidance", "#2ca02c")):
    pts = [(r["d"], r["kl_path_exact"] / max(r["kl_end_exact"], 1e-12))
           for (dim, s, m, N), rr2 in cell.items() if m == mode and N == 256
           for r in rr2[:1] if s == 1.0]
    if pts:
        x, yv = zip(*sorted(pts))
        ax.loglog(x, yv, "o-", color=col, label=mode)
ax.axhline(1, color="k", ls=":", lw=0.8)
ax.set(xlabel="d", ylabel="path KL / endpoint KL",
       title="bound tightness at 1σ: tight when green, loud when red")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(ROOT / "figures/cert_killtest.png", dpi=180)
print("figure -> figures/cert_killtest.png")
