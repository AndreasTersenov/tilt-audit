#!/usr/bin/env python
"""README hero figure: the measured operating envelope of every diagnostic.

One matrix, instruments x failure classes, every cell a measured verdict from
the JSONLs (values quoted in-cell). This is the project's one-glance summary.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).resolve().parent.parent / "figures"

GREEN, RED, AMBER, GRAY = "#2e7d32", "#b3261e", "#b8860b", "#9e9e9e"
BG = {"caught": "#e8f5e9", "fooled": "#fdecea", "partial": "#fff8e1",
      "na": "#f5f5f5"}
FG = {"caught": GREEN, "fooled": RED, "partial": AMBER, "na": GRAY}

ROWS = [
    ("temperature check\n(sample variance ratio)", [
        ("caught", "reads 1.33–1.43"),
        ("fooled", "reads exactly 1.00\ntrue error 6× floor"),
        ("na", "not applicable"),
        ("caught", "passes"),
    ]),
    ("two-sample test\n(PQMass)", [
        ("caught", "power 1.00 from N=64"),
        ("partial", "power 0.50 at N=64\n1.00 from N=256"),
        ("caught", "power 1.00\n(down to 5% modes)"),
        ("caught", "false alarms ~5%"),
    ]),
    ("coverage test\n(TARP, symmetrized)", [
        ("caught", "power 1.00"),
        ("caught", "power 1.00"),
        ("partial", "power 0.90 at 50/50\nblind at 80/20"),
        ("caught", "false alarms ~5%"),
    ]),
    ("score certificate,\ntrue score (score-KSD)", [
        ("caught", "power 1.00 from N=64"),
        ("caught", "power 1.00 from N=64"),
        ("fooled", "1.00× null\nat any budget to 16,384"),
        ("caught", "false alarms ~5%"),
    ]),
    ("score certificate,\nlearned score (deployed)", [
        ("fooled", "certifies DPS clean\n(true error 30× floor)"),
        ("partial", "signal collapses\n2.9× → 1.1× null"),
        ("fooled", "inherited"),
        ("partial", "5% score error →\n100% false alarms"),
    ]),
    ("budget-doubling check\n(K vs 2K, truth-free)", [
        ("fooled", "agrees at every T\nwhile 10³× off"),
        ("na", "not tested"),
        ("fooled", "stuck runs agree"),
        ("caught", "passes"),
    ]),
]
COLS = ["biased dynamics\n(DPS-class)", "compensating errors\n(score × scheme)",
        "missing posterior mode\n(12σ apart)", "honest sampler\n(control)"]

fig, ax = plt.subplots(figsize=(11.5, 7.2))
nr, nc = len(ROWS), len(COLS)
for i, (rname, cells) in enumerate(ROWS):
    for j, (verdict, note) in enumerate(cells):
        y = nr - 1 - i
        ax.add_patch(plt.Rectangle((j, y), 0.96, 0.92,
                                   facecolor=BG[verdict],
                                   edgecolor="#d0d0d0", lw=0.8))
        label = {"caught": "CAUGHT", "fooled": "FOOLED",
                 "partial": "PARTIAL", "na": "—"}[verdict]
        if j == 3 and verdict == "caught":
            label = "PASSES"
        ax.text(j + 0.48, y + 0.62, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=FG[verdict])
        ax.text(j + 0.48, y + 0.28, note, ha="center", va="center",
                fontsize=7.4, color="#444444")
for i, (rname, _) in enumerate(ROWS):
    ax.text(-0.06, nr - 1 - i + 0.46, rname, ha="right", va="center",
            fontsize=9.3)
for j, cname in enumerate(COLS):
    ax.text(j + 0.48, nr + 0.12, cname, ha="center", va="bottom",
            fontsize=9.3)
ax.set_xlim(-2.35, nc)
ax.set_ylim(-0.35, nr + 0.75)
ax.axis("off")
ax.set_title("What each diagnostic actually catches, measured on ground truth",
             fontsize=13, pad=26, x=0.43)
fig.text(0.5, 0.015,
         "Every cell is a measured detection rate on the tilt-audit bench, "
         "where the true posterior is known exactly. Data: results/*.jsonl.",
         ha="center", fontsize=8, color="#666666")
fig.tight_layout()
fig.savefig(FIG / "fig_envelope.png", dpi=160)
print("wrote fig_envelope.png")
