# HANDOFF_DAWN — 2026-07-03 · GRF pilot overnight

> Status at last update: 00:30 UTC (H1.7). LIVE DOCUMENT during the night;
> final version at dawn. Chronology in NIGHT_LOG.md; per-event detail there.

## 1. Verdict table — P-20260702d–h (PROPOSED scores; final scoring with Andreas)

| Prediction | Proposed | One-line evidence |
|---|---|---|
| **P-d** (75%) DPS off-target, gap grows with β | **HIT** | Exact-score T1: W2/floor 1.4×→28× rising monotonically in β at every d and N (N=256, 64²: 5.0×@0.5σ → 27.7×@4σ); γ*=1.33–1.43 (over-concentrated); mean-CI coverage under-covers (cov68 → 0 at strong tilt). Discretization control (exact_guidance) ≈ floor. |
| **P-e** (70%) SAP runs cold, γ*>1 | **PARTIAL HIT (propose: HIT with revision note)** | γ*>1 (1.35–1.73, cold) exactly where selection can track — weak tilt/low d (16², 0.5σ, N≥64). Elsewhere the pathology is WORSE but different: per-step full-reward resampling collapses variance without moving the mean (γ_mean→0, var_ratio_logmed≈−20, W2 4–143× floor). The discrete substrate's scalar-γ* signature does not transfer globally; the over-concentration mechanism does. |
| **P-f** (85%) twisted SMC on-target, valid Ẑ | **HIT** | Conjugate twisted SMC = 0.96–1.06× oracle floor in ALL 36 (d,β,N) cells, γ*=1.00; incremental weights machine-zero (<1e-9; tower-property self-test) so log Ẑ = log Z exactly; T-G3 prior-proposal variant: Ẑ unbiased (mean ratio 1±5se, d=1, real weight variance). |
| **P-g** (55%) misspec propagates differently per scheme | *(pending — T2 learned nets ~02:05; analytic-contamination arm already in)* | Analytic ε-contaminated exact-score runs done (t1_misspec.jsonl); learned S-mis pending. |
| **P-h** (25%) KILL: bias negligible at realistic settings | **MISS (= good for the program)** | Kill criterion decisively NOT triggered: DPS alone ≥3× floor from 0.5σ upward at N=256 (all d); max frozen-scheme ratio 251× (terminal-IS). Bias is d-extensive as the deep-read predicted (KL/mode ~O(1)). |

## 2. GO / NO-GO recommendation (plan §3)

**GO** — recommend committing to Direction 1 as the UVa flagship.
- "≥2 schemes with nonzero, cleanly-scaling bias in ≥2 knobs at realistic
  settings": DPS scales in β, d, and N-relative floor; SAP scales in β and T
  (depth); terminal-IS scales in d. All at 0.5–1σ tilts — realistic strengths.
- Three-way decomposition separability: exact-score arms cleanly separate
  scheme bias from finite-N (oracle floor) and discretization
  (exact_guidance control ≈ floor); score-error and misspec axes land with T2
  (~03:00). *(Update at dawn.)*
- The certificate story has teeth already: potential-only twisting (the naive
  ψ-weights variant) weight-degenerates at d≥256 while the conjugate twisted
  sampler is exact — ESS/Ẑ trajectories separate correct from degenerate
  samplers on this oracle.

## 3. Utilization report

*(filled at dawn by scripts/dawn_report.py; util.log samples every 15 min)*

## 4. Track B — E-20260702a/b status

- **B1 (α-sweep):** α=0.05 complete at seeds 0,1 (n=124-ish/seed pair);
  AUROC 0.500 at α=0.05 — mechanically expected (n_iid=1 stratum carries no
  ranking signal at N=16; effective α quantization: round(α·16)/16 makes
  0.10 and 0.15 both n_iid=2). Sweep for α ∈ {0.10..0.50} seed 0 running on
  the dual-server config; threshold test (AUROC≥0.65 by α=0.25) lands ~02:00.
- **B2 (R1 twin):** running but SLOW (~3.5 problems/h/stream; R1 emits
  huge step-segmented traces through the SAP harness). Will be PARTIAL at
  dawn (~25–40 problems total across 1–2 seeds). Early n=11 read: SAP
  selected-accuracy 9% vs iid 45% (directional; R1's long traces + 4096
  max-model-len truncation may be breaking SAP's step segmentation — needs a
  daytime look before believing).

## 5. Exploratory results beyond the pre-registration (firewalled, all tagged)

- **T4.1 memoryless-schedule theorem (2409.08861) exactly quantified**: at
  tf=3, KL to true tilt: 0.09 (memoryless σ²=2) → 0.2–0.4 (σ²=1) → 14–87
  (σ²=0.5) → 5e3–4e4 (σ²=0.1). At tf=9 all σ²∈[1,2] sit inside the
  finite-horizon residual — the value-function bias is a short-horizon /
  low-noise phenomenon, catastrophic exactly in the σ→0 (FM/DDIM-style)
  fine-tuning limit. Files: results/t41_memoryless.jsonl.
- **Weak-tilt arm (0.125σ, 0.25σ)**: locates the 3×-floor crossing for the
  P-h evidence. (In t1_weak.jsonl; fold into the money plot at dawn.)
- **T-sensitivity (T∈{32,64,256})**: pins the exact_guidance control's
  residual as discretization; SAP's γ*(T) depth-dependence measured in the
  continuous substrate. (t1_tsens.jsonl; analysis at dawn.)
- **d=128² arm (T4.2)**: extends extensivity to d=16384. (Queued.)

## 6. Failures & incidents (full detail in NIGHT_LOG)

- Two numerical traps caught by smoke tests BEFORE any confirmatory row:
  forward-Euler guidance instability (fix: exponential integrator) and
  uniform-time-grid under-resolution (fix: log grid). Gates re-run green.
- fp64 leak into score-net training (5.6 h projected) — caught at step 5000,
  restarted fp32 (all three nets concurrent, ~2 h).
- Track B automation went through three drivers: v1 pace 3× optimistic;
  v2 instant-fail cascade (orphaned PRM held GPU memory + unchecked rc);
  v3 killed before its server-swap race could destroy the live R1 run.
  Stable end state = dual-server config with rc-checked sequential chains.
- Learned-pathway DPS Euler blowup caught by the pathway control (fix:
  analytic-preconditioned exponential guidance step).

## 7. Next-session actions

- Score P-d–h WITH Andreas (this table is a proposal).
- T2 decomposition figure + P-g verdict if trainings landed late.
- B2: rerun twin with a daytime budget or trimmed config; investigate R1
  step-segmentation under the 4096 cap before interpreting SAP numbers.
- Decide repo publication shape (explicitly deferred by the plan).
- E-20260702a/b/c Result+Updated-belief blocks: drafted at dawn in
  particle-reasoners/RESEARCH_LOG.md (do not edit predictions).
