#!/usr/bin/env python
"""Stage-2 digest: P-20260704d-f verdicts + operating characteristics
(docs/PLAN_CERT_LEARNED.md). Reads results/cert_learned.jsonl."""
import collections
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
rows = [json.loads(l) for l in open(ROOT / "results/cert_learned.jsonl")]
seen = {}
for r in rows:
    seen[(r["score"], r["shift"], r["mode"], r["seed"])] = r
rows = list(seen.values())
dps = [r for r in rows if r["mode"] == "dps"]

med = lambda v: float(np.median(v)) if len(v) else float("nan")
cell = collections.defaultdict(list)
for r in dps:
    cell[(r["score"], r["shift"])].append(r)

print("=" * 100)
print("T1. Certificate + block instruments per net (dps rows; medians over 5 seeds)")
print(f"{'net':>18} {'shift':>5} {'KLhat':>10} {'modeESS':>8} {'b0ESS':>6} "
      f"{'b0KLhat':>8} {'b0true-fl':>9} {'b2ESS':>6} {'clip':>5}")
for (score, shift), rr in sorted(cell.items()):
    print(f"{score:>18} {shift:>5} {med([r['kl_path_hat'] for r in rr]):>10.3g}"
          f" {med([r['ess_modes_med'] for r in rr]):>8.0f}"
          f" {med([r['band0_ess'] for r in rr]):>6.1f}"
          f" {med([r['band0_kl_hat'] for r in rr]):>8.3g}"
          f" {med([r['band0_kl_true'] - r['band0_kl_floor'] for r in rr]):>9.3g}"
          f" {med([r['band2_ess'] for r in rr]):>6.1f}"
          f" {med([r['clip_frac'] for r in rr]):>5.2f}")

print("=" * 100)
print("P-20260704d: clean net — per-mode health + band ordering")
cl = [r for r in dps if r["score"] == "learned:clean"]
e1 = med([r["ess_modes_med"] for r in cl if r["shift"] == 1.0])
ok_ess = e1 >= 100
pairs = [(r["band0_kl_hat"], r["band0_kl_true"] - r["band0_kl_floor"])
         for r in cl]
rho0 = stats.spearmanr(*zip(*pairs)).statistic if len(pairs) > 4 else float("nan")
b2 = med([r["band2_ess"] for r in cl])
print(f"  median per-mode ESS @1σ = {e1:.0f}/256 (need >=100) -> "
      f"{'OK' if ok_ess else 'FAIL'}")
print(f"  band0 KLhat vs band0 true-damage Spearman = {rho0:.3f} "
      f"(need >=0.8) -> {'OK' if rho0 >= 0.8 else 'FAIL'}")
print(f"  band2 (widest) ESS = {b2:.1f} (predicted degenerate ~1)")
print(f"  P-d verdict: {'HIT' if ok_ess and rho0 >= 0.8 else 'MISS/PARTIAL'}")

print("=" * 100)
print("P-20260704e: analytic pathway — per-mode-sum vs exact chain law")
an = [r for r in dps if r["score"] == "pathway:analytic"]
for shift in (0.5, 1.0, 2.0, 4.0):
    rr = [r for r in an if r["shift"] == shift and "kl_path_exact" in r]
    if rr:
        ratio = med([r["kl_modes_sum"] / r["kl_path_exact"] for r in rr])
        print(f"  shift {shift}: per-mode-sum/exact = {ratio:.3f}")
best = max(med([r["kl_modes_sum"] / r["kl_path_exact"] for r in an
                if r["shift"] == s]) for s in (0.5, 1.0, 2.0, 4.0))
print(f"  P-e verdict (within 20% anywhere on this grid): "
      f"{'HIT' if best >= 0.8 else 'MISS'} (best ratio {best:.3f})")

print("=" * 100)
print("P-20260704f: clean-net vs analytic-pathway dps readings")
ok_f = True
for shift in (0.5, 1.0, 2.0, 4.0):
    a = med([r["kl_path_hat"] for r in an if r["shift"] == shift])
    c = med([r["kl_path_hat"] for r in cl if r["shift"] == shift])
    ratio = c / a
    ok_f = ok_f and 0.5 <= ratio <= 2.0
    print(f"  shift {shift}: clean/analytic = {c:.3g}/{a:.3g} = {ratio:.2f}")
print(f"  P-f verdict (within 2x everywhere): {'HIT' if ok_f else 'MISS'}")

print("=" * 100)
print("Kill/reshape check: clean-net median per-mode ESS @1σ "
      f"= {e1:.0f} (<10 kills the block-wise arc) -> "
      f"{'SURVIVES' if e1 >= 10 else 'KILLED'}")
print("Scope note: contaminated nets read hotter through steering violence "
      "(clip activity + VJP through the wrong score), not through a model "
      "term — indirect sensitivity, to be reported as such.")
