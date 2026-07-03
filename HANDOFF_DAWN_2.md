# HANDOFF_DAWN_2 — 2026-07-04 · The four arms night

> ASSEMBLING — skeleton written mid-night (H1), finalized at dawn.
> Chronology: NIGHT_LOG_2026-07-04.md. Plan: docs/OVERNIGHT_2026-07-03_ARMS_NIGHT.md.
> Repo PUBLIC since 22:17 UTC (H0.1, before the first GPU job):
> https://github.com/AndreasTersenov/tilt-audit

## 1. Verdict table — P-20260703b–e (PROPOSED; scoring happens WITH Andreas)

| Prediction | Proposed | One-line evidence |
|---|---|---|
| **P-b** (85%) multi-y: ratio structure y-generic, spread ≲10% | **HIT** | 24 y-draws, 64²: DPS ratio monotone-in-β 24/24, ordering twisted<dps<min(sap,tIS) 24/24; rel IQR of W2/floor ratios 0.8–8.5% (dps 2.0–2.7%, twisted ≈1%); medians 4.85/7.38/13.09/27.47 at 0.5–4σ, twisted pinned at 1.00. |
| **P-c** (70%) diagnostics detect plain DPS but ALL miss the compensation config | **MISS on the blind-spot clause / HIT on the detection clause** (TBD final numbers) | Plain DPS detected by all three tests at every budget ≥64 (clause 1 ✓). But the ε=−0.3+DPS compensation config is DETECTED: PQMass 0.95@256, 1.0@1024+; TARP 1.0@64; the distance-based tests see per-mode geometry the γ* scalar misses. The trap fools the temperature diagnostic exactly (γ*=1.001 at ε*=−0.28) — not the sample-based tests. |
| **P-d** (60%) amortized: summaries pass, geometry fails ≥3× floor | **TBD at dawn** | Early (quarter-trained 15k ckpt, y999): meanerr 3.2%, pxvar 0.91, W2 2.25× floor — geometry BELOW the 3× bar already at quarter training; if default net confirms, propose MISS (the interesting direction: amortization at this substrate's difficulty is fine; steering is the problem). |
| **P-e** (55%) Rémy: warm at small K, K-converging, misspec sign-flipped | **HIT (all three clauses; γ* nuance)** (TBD final) | W2(K) monotone ↓ at every tilt, reaching 1.0–1.2× floor by K=100 (16²: T-N2 anchor 1.00× exact-inflation); misspec SIGN-FLIP confirmed: ε=−0.3 aggravates (7.55 vs 3.42 at K=30/4σ) where it cancels DPS. Small-K "warm" clause needs the γ_mean/var split (raw γ*>1 at strong tilt reflects mean-lag, not over-concentration — same γ*-conflation lesson as P-e last night). |

## 2. Per-arm results

### A1 — multi-y ensemble (confirmatory 1440 rows + fillers: 24y total, N=64 arm, exact_guidance column, 128² column)
- Headline: the single-y caveat is dead. Ratio distributions across 24 observation draws are tight (rel IQR ≤8.5% everywhere at 64², N=256); orderings and β-monotonicity hold per-draw 24/24.
- 128² column (4 y): structure persists at d=16384. (numbers at dawn)
- Figure: fig_a1_multiy_violins.png (done).

### A2 — diagnostic power ("certify the certifiers"; 1200-cell battery + recal + 16k rung)
- Power curves: PQMass/TARP/MIRA all detect dps, sap, dps@±0.3, dps@0.5σ/2σ at ~every budget tested; twisted (near-exact) reads as null in PQMass/TARP. (final table at dawn)
- **Finding 1 (tool bug, TARP):** tarp-0.1.0 norm=True normalizes by the TRUTHS' min-max → truths always inside the box, samples spill out in ~2q/L dims → d-extensive null miscalibration: max|ecp−α| = 0.20 at q=4096 on the EXACT posterior. Symmetric pooled-sample standardization restores the null (0.04–0.08) and keeps power (dps→0.98). Battery uses the wrapped version; upstream issue-worthy.
- **Finding 2 (tool bug, MIRA):** mira-score's truth-based min-max norm shows the same bug class: oracle false-positive 0.65–0.80 under its analytic-z null at q=4096; flags the near-exact twisted sampler. → empirical one-sided calibration (60 oracle nulls) + symmetric-wrap variant (mira_sym). (final rates at dawn)
- **ε\* trap pinned quantitatively:** γ\* crosses 1.000 at ε\* = −0.28 ± 0.01; W2/floor bottoms at ~5.8× on a flat basin ε∈[−0.32,−0.24]; wrong-model exact oracle (pass-through) spans 2.1→6.2× floor over the ladder — at ε=−0.44, DPS bias hides entirely inside misspec damage.
- Figures: fig_a2_power_curves.png, fig_a2_score_vs_damage.png (+archive_damage.json).

### A3 — amortized-conditional arm
- Trainings: default (60k), halfcap (60k), quarter (15k) + the 2k/6k warmups double as a training-budget DOWN-ladder for free.
- Audits at the trained Bayes point b=s² for 5 y-draws × 3 seeds; summary checks (mean/marginal/band-power) computed in-row next to geometry (W2/floor, γ*, var-spectrum).
- (results at dawn; early quarter numbers above)
- Figure: fig_a3_summary_vs_geometry.png.

### A4 — Rémy-scheme arm (confirmatory K-sweep + misspec pairs + K-densification + 128² + eps0/2 control + seeds ×6)
- The scheme audited is the PRE-REGISTERED one (σ_t²-inflated annealed Langevin, K steps/level); fidelity caveats vs Remy-et-al production code (adaptive MH-HMC, sqrt step scaling, ODE denoise) documented in run_a4.py header after reading CosmoStat/jax-lensing. The σ² inflation itself is verbatim theirs.
- T-N2 anchor: exact-inflation K=100 → 1.00× floor; σ²-inflation W2(K) strictly monotone.
- Equal-NFE read (fig iv): Rémy needs K≈7/10/15–20/25–30 (≈450–1900 NFE) to MATCH single-pass DPS (64 NFE) at 0.5/1/2/4σ — then keeps converging to the floor while DPS is stuck at 4.9–27×. "Conservative but expensive, and it actually gets there."
- Misspec sign-flip: ε<0 aggravates the conservative scheme (opposite of DPS's cancellation); ε=+0.3 (numbers at dawn).
- Figure: fig_a4_remy_K.png (done).

## 3. Utilization report
(TBD at dawn: queue/util2.log summary + per-GPU narrative. Note: all four arms' confirmatory cores completed by ~H1 — the night's estimates were ~10× conservative on sampler throughput; GPUs kept fed with the pre-listed filler ladder.)

## 4. Incidents
- **CUDA stack near-miss (H0.1):** `uv pip install pqm` silently pulled torch-2.12.1+triton, upgrading nvidia-cudnn-cu12 to 9.24 (needs driver ≥580; titan has 575) — every GPU JAX call died with CUDNN_STATUS_NOT_INITIALIZED. Fixed by purging torch's CUDA stack + pinning cudnn<9.14; MIRA got a CPU-only torch instead. Lesson: on a shared JAX venv, diagnostics get --no-deps or their own env; torch is a CUDA-stack trojan.
- **Foreground-timeout kill (H0.8):** first A3-quarter launch lacked `&` — the harness 2-min timeout killed it mid-run; 7 rows survived (append-mode). Relaunched backgrounded. Kill-by-PID discipline held all night; no pattern-matching kills.
- **Self-inflicted statistics (caught before contaminating results):** (a) my truth-vs-samples chi² exchangeability "alarm" was in-sample bias (+2/S exactly); (b) first mira_cal used a 20-null two-sided rank whose floor is p=0.095 — could never detect at α=0.05. Both corrected in-file (keep-last dedupe), both logged as worked examples of the Feynman clause.
- run_t1 "oracle" rows under --score misspec are the WRONG-MODEL oracle (pass-through), not the floor — caught in the ε* analysis; dawn analysis uses clean floors from t1_core.

## 5. Next actions (for the joint session)
- Score P-20260703b–e (this table is a proposal).
- Decide how to report the TARP/MIRA normalization findings (upstream issues + a "diagnostics need symmetric preprocessing" subsection; strengthens the certify-the-certifiers section from "power curves" to "we found real bugs in the field's tools with the oracle").
- P-d follow-ups depending on final verdict: if MISS (amortized fine at this difficulty) → the note's framing shifts to "steering is the failure point; amortization is not" + training-budget failure-onset curve from the 2k/6k/15k/60k ladder.
- Equal-NFE figure → the CosmoStat-facing artifact; consider sending to Rémy/Starck orbit with the tour notebook.
- Fold A1 spread numbers into the note draft; 128² columns close the d-extensivity referee question.
