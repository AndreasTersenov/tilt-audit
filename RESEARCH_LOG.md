# RESEARCH_LOG — tilt-audit

> Migrated from particle-reasoners RESEARCH_LOG — canonical pre-registration timestamp
> lives there until scoring. Predictions below are FROZEN as of 2026-07-02 (before the
> first GPU job); they are scored with the owner, never edited unilaterally.

## Predictions

### P-20260702d · plug-in guidance measurably off-target, gap grows with β · conf 75% · resolve-by 2026-07-05 · OPEN
- **Claim:** [GRF pilot P1] DPS-style plug-in guidance is off-target — over-concentrated,
  under-covering — with the gap growing with guidance strength.
- **Made:** 2026-07-02 · **Context:** GRF pilot overnight run (docs/OVERNIGHT_2026-07-02_GRF_PILOT.md)
- **Resolution criterion:** T1 grid exact W₂/KL + functional coverage vs β.
- **Outcome:** (pending)
- **Lesson:** (pending)

### P-20260702e · reward-as-potential SMC runs cold (γ* > 1) · conf 70% · resolve-by 2026-07-05 · OPEN
- **Claim:** [GRF pilot P2] the SAP analog reproduces the discrete substrate's over-tilt
  signature in ℝ^d.
- **Made:** 2026-07-02 · **Context:** GRF pilot overnight run
- **Resolution criterion:** γ* fits on T1 output.
- **Outcome:** (pending)
- **Lesson:** (pending)

### P-20260702f · proper twisted SMC on-target with valid Ẑ · conf 85% · resolve-by 2026-07-05 · OPEN
- **Claim:** [GRF pilot P3] proper twisted SMC is on-target within finite-N error, Ẑ bracketing
  the analytic Z.
- **Made:** 2026-07-02 · **Context:** GRF pilot overnight run
- **Resolution criterion:** T1 grid + T-G3 gate outputs.
- **Outcome:** (pending)
- **Lesson:** (pending)

### P-20260702g · misspecification propagates differently per scheme · conf 55% · resolve-by 2026-07-05 · OPEN
- **Claim:** [GRF pilot P4] a contaminated score is amplified by plug-in guidance and partially
  absorbed into weights by proper SMC — the decomposition is non-trivial.
- **Made:** 2026-07-02 · **Context:** GRF pilot overnight run (T2 tier)
- **Resolution criterion:** three-way decomposition at 64².
- **Outcome:** (pending)
- **Lesson:** (pending)

### P-20260702h · KILL BRANCH: bias negligible at realistic settings · conf 25% · resolve-by 2026-07-05 · OPEN
- **Claim:** [GRF pilot P5, REVISED from 40% after deep-read of 2502.07849 — owner's number]
  at realistic β and reward SNR, plug-in bias is negligible in absolute terms at d=4096.
- **Made:** 2026-07-02 · **Context:** the pre-registered NO-GO branch; plan §3 kill criterion.
- **Resolution criterion:** no scheme W₂ > 3× oracle finite-N floor at any (β,d) in the T1 box.
- **Outcome:** (pending)
- **Lesson:** (pending)

## Experiments

### E-20260702c · GRF pilot core (Track A, T1+T2)
- **Hypothesis:** the proper-vs-improper tilting phenomenology reproduces in continuous
  high-dimensional space with exactly known σ — see P-20260702d–h for the per-claim splits.
- **Setup:** JAX, this repo, GPUs 0,1 (titan); spec + gates in
  particle-reasoners/docs/OVERNIGHT_2026-07-02_GRF_PILOT.md §3–4; seeds 0,1,2; frozen before
  first GPU job.
- **Expectation:** encoded in P-20260702d–h (frozen).
- **Result:** (pending)
- **Updated belief:** (pending)
