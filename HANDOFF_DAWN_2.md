# HANDOFF_DAWN_2 — 2026-07-04 · The four arms night

> FINAL — assembled at H8.1. Chronology: NIGHT_LOG_2026-07-04.md. Plan:
> docs/OVERNIGHT_2026-07-03_ARMS_NIGHT.md. Repo PUBLIC since H0.1 (before the
> first GPU job): https://github.com/AndreasTersenov/tilt-audit
> All four confirmatory arms completed; figures in `figures/`.

## 1. Verdict table — P-20260703b–e (PROPOSED; scoring happens WITH Andreas)

| Prediction | Proposed | One-line evidence |
|---|---|---|
| **P-b** (85%) multi-y: ratio structure y-generic, spread ≲10% | **HIT** | 24 y-draws, 64², N=256: DPS ratio monotone-in-β 24/24, ordering twisted<dps<min(sap,tIS) 24/24; rel IQR of W2/floor 0.8–8.5% (dps 2.0–2.7%, twisted ≈1%); medians 4.85/7.38/13.09/27.47 at 0.5/1/2/4σ, twisted pinned at 1.00. Persists at N=64 and 128² (fillers). |
| **P-c** (70%) diagnostics detect plain DPS but ALL miss the compensation config | **MISS on the blind-spot clause; HIT on the detection clause** | Plain DPS: detected by PQMass/TARP/MIRA at every budget ≥64 (clause 1 ✓). The ε=−0.3+DPS compensation config is DETECTED, not missed: PQMass 0.5@64→0.95@256→1.0@1024+ (and 1.0 at 16k), TARP 1.0@64, repaired MIRA 1.0@64. The trap fools exactly the γ* temperature scalar (γ*=1.001 at ε*=−0.28±0.01, W2 basin min ~5.8× floor) — not the distance-based tests. The REAL blind spots found: the tools' own preprocessing (below). |
| **P-d** (60%) amortized: summaries pass, geometry fails ≥3× floor | **MISS on the ≥3× geometry clause (direction: amortized is *better* than predicted)** | Default net (60k): W2/floor 2.56 (2.18–2.80 over 5 y × 3 seeds) — below the 3× bar; summaries: mean err 3.4% ✓, band powers −0.1%/−1.7% ✓, pixel variance −7.5% (a careful summary reader WOULD see it; γ*=1.07). Capacity/steps ladder saturates: 6k→3.0×, 15k/half/default ≈2.6–2.7×; failure onset between 2k (26×, variance collapse, γ*=0.07) and 6k. The note's message flips to: at this difficulty amortization is fine — steering is the problem. |
| **P-e** (55%) Rémy: warm at small K, K-converging, misspec sign-flipped | **HIT (K-convergence + sign-flip clean; warm-clause with a documented nuance)** | W2(K) strictly monotone ↓ at every tilt, 1.0–1.2× floor by K=100 (T-N2 anchor: exact-inflation K=100 = 1.00× floor); ε=−0.3 AGGRAVATES (K=30/4σ: 7.55× vs 3.42× clean; ε=+0.3 comparable) — sign-flipped vs DPS's cancellation ✓. Small-K state is warm in VARIANCE (var_log_med +1.1..+3.4) while γ_mean=4–7 shows simultaneous mean OVERSHOOT (mid-anneal V_t≈1 under-regularizes high-k modes); the scalar γ* (1.3–3) averages the two into nothing — same conflation lesson as P-20260702e. |

## 2. Per-arm results

### A1 — multi-y (1440 confirmatory rows; fillers: 24y, seeds×4, N=64 arm, exact_guidance column, 128²)
The single-y caveat is dead: orderings and β-monotonicity hold per-draw 24/24; spreads
0.8–8.5% rel IQR. exact_guidance ≈ floor across y (discretization control clean under
multi-y). 128² column (4 y): structure persists at d=16384.
Figure: `fig_a1_multiy_violins.png`.

### A2 — certify the certifiers (1200-cell battery + recal + two-sided cal + 16k rung)
- **Power curves** (`fig_a2_power_curves.png`): every real failure (dps, sap, dps@±0.3,
  dps@0.5σ/2σ, twisted@−0.3) detected at ~1.0 from budget ~256 (most from 64); near-exact
  twisted correctly reads as null in PQMass/TARP/repaired-MIRA; oracle nulls sit at 0.05.
  PQMass 16k rung: nulls clean, compensation configs 1.0.
- **Finding 1 — tarp-0.1.0 normalization bug (d-extensive):** norm=True min-maxes by the
  TRUTHS' range → truths always inside the box, samples spill out in ~2q/L dims →
  max|ecp−α| = 0.20 at q=4096 on the EXACT posterior (0.05@64d, 0.15@2048d). Symmetric
  pooled-sample standardization (truths outside the transform) restores the null
  (0.04–0.08) and keeps power (dps→0.98). Found BY the T-N3 null gate.
- **Finding 2 — mira-score-0.1.7, same bug class:** truth-based min-max → oracle
  false-positive 0.65–0.80 under its analytic-z null at q=4096; flags the near-exact
  twisted sampler. Symmetric wrap repairs it EXACTLY: null score 0.6677±0.0029 vs analytic
  (2N+3)/(3(N+1)) = 0.66797. Two-sided empirical calibration (60 nulls): dps/sap/±0.3 all
  ~1.0, twisted 0.00; the `side` column separates overconfident (low) from sap's
  collapse-reads-as-underconfident (high).
- **ε\* trap pinned:** γ\* crosses 1.000 at ε\* = −0.28 ± 0.01; W2/floor flat basin ~5.8×
  over ε∈[−0.32,−0.24]; wrong-model exact oracle (pass-through) spans 2.1→6.2× over the
  ladder; at ε=−0.44 DPS bias hides entirely inside misspec damage. Cross-check to last
  night: abs W2 2.915 vs 2.916 at ε=−0.3.
- Figures: `fig_a2_power_curves.png`, `fig_a2_score_vs_damage.png` (+`archive_damage.json`).

### A3 — amortized-conditional (3 trainings + 2 warmup ckpts as a free down-ladder; 5 y × 3 seeds each)
| ckpt | W2/floor | mean err | px var | bp low/high k | γ* |
|---|---|---|---|---|---|
| warmup2k | 25.8 | 34% | 3.8× | +24%/+270% | 0.07 (collapse) |
| warmup6k | 2.99 | 4.2% | −13% | +0.3%/−9% | 1.04 |
| quarter15k | 2.70 | 3.6% | −8% | −0.3%/−2% | 1.07 |
| halfcap | 2.60 | 3.4% | −8% | −0.2%/−2% | 1.08 |
| default60k | **2.56** | 3.4% | −7.5% | −0.1%/−1.7% | 1.07 |
Saturation by ~6–15k steps; capacity not binding. The amortized class lands ~2.6× floor
with NO steering machinery — cf. DPS at 4.9–27× on the same substrate.
Figure: `fig_a3_summary_vs_geometry.png`.

### A4 — Rémy scheme (K-sweep + 16² column + misspec pairs + densified K + eps0/2 + seeds×6 + 128²)
- Audited = the PRE-REGISTERED scheme (σ_t²-inflated annealed Langevin, K steps/level on
  our grid; per-mode preconditioned ULA — step size was unpinned, decision logged pre-data).
  Fidelity caveats vs their production code (adaptive MH-HMC, sqrt step scaling, ODE
  denoise) in run_a4.py's header after reading CosmoStat/jax-lensing; the σ² inflation is
  verbatim theirs.
- **Equal-NFE read** (`fig_a4_remy_K.png`): Rémy needs K≈7/10/15–20/25–30 (≈450–1900 NFE)
  to MATCH single-pass DPS (64 NFE) at 0.5/1/2/4σ — then keeps converging to the floor
  while DPS is stuck. "Conservative but expensive — and it actually gets there."
  eps0/2 control: no material change (ULA bias not driving conclusions). 128²: K-convergence
  persists, ratios shift up ~2× at K=30 (d-extensive equilibration cost).

## 3. Utilization report — honest version
All four confirmatory cores were COMPLETE by ~H1.5 and fillers through rung 7 by ~H2.2:
sampler throughput beat the plan's estimates ~10× (JIT reuse; solo-GPU trainings 3–5×).
Actual GPU burn ≈ 4–5 GPU·h of the 24 available. After ~H2, the executor session was
suspended externally (timers fired ~6 h late), so the queue drained and no further filler
rungs were launched; GPUs idled from ~H2.2 to dawn (util log: means 3–10% over 22:50→06:06,
concentrated in the first samples). What idle time did NOT cost: every planned confirmatory
cell, every checkpoint-branch add-on, and all five figures exist. What it did cost:
unbounded rungs (deeper seed/y densification, 128² Rémy misspec, ε* at other shifts).

## 4. Incidents
- **CUDA stack near-miss (H0.1):** `uv pip install pqm` pulled torch-2.12.1+triton,
  upgrading nvidia-cudnn-cu12 past driver 575's ceiling — all GPU JAX dead
  (CUDNN_STATUS_NOT_INITIALIZED). Purged torch's CUDA stack, pinned cudnn<9.14, verified;
  MIRA got CPU-only torch. Lesson: diagnostics install with --no-deps or a separate env.
- **Self-match deadlock, mild recurrence (H1.2):** a launch-time pgrep matched my own
  tool-call shell (pattern text in its cmdline — the exact class behind last night's T2
  launcher deadlock); the down-ladder launcher waited on itself, ~10 min GPU-1 idle.
  Killed by PID, no surviving children, relaunched directly. The lesson is now: no pgrep
  in launchers, period — resolve numeric PIDs first.
- **1 GB push hang (H1):** results/archives16k/ wasn't gitignored; a commit carried
  4×268 MB banks. Soft-reset, gitignored, recommitted clean. New-results-subdir ⇒
  gitignore line at creation.
- **Foreground-timeout kill (H0.8):** first A3-quarter launch lacked `&`; harness killed
  it at 2 min; 7 rows survived (append-mode). Relaunched backgrounded.
- **Self-inflicted statistics, both caught pre-contamination:** (a) truth-vs-samples χ²
  "exchangeability alarm" was in-sample bias (+2/S exactly); (b) first mira_cal used a
  20-null two-sided rank (floor p=0.095 — can never detect at 0.05). Corrected in-file;
  kept as worked examples.
- run_t1 "oracle" rows under --score misspec are the WRONG-MODEL oracle (pass-through),
  not the floor — caught in the ε* analysis; dawn analysis uses clean floors from t1_core.

## 5. Next actions (for the joint session)
- Score P-20260703b–e (table above is a proposal).
- Decide reporting for the TARP/MIRA findings: upstream issues (tarp repo; SammyS15/
  mira-score) + a "certify the certifiers, including their preprocessing" section — the
  strongest new material of the night; both bugs are invisible without an oracle.
- P-d MISS follow-through: reframe the note ("steering, not amortization, is the failure
  point at this difficulty") + the failure-onset curve (2k→6k) as the D&J-adjacent panel.
- Equal-NFE figure → the CosmoStat-facing artifact (Rémy/Starck orbit + tour notebook).
- Optional cheap adds if more GPU time: ε* ladders at other shifts; Rémy misspec at 128²;
  MIRA/TARP power at L>128 (tighter nulls).
- E-entry drafted in RESEARCH_LOG.md (Results + Updated beliefs; Outcomes/Lessons at scoring).
