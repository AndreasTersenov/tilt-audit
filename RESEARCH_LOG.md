# RESEARCH_LOG — tilt-audit

> Migrated from particle-reasoners RESEARCH_LOG — canonical pre-registration timestamp
> lives there until scoring. Predictions below are FROZEN as of 2026-07-02 (before the
> first GPU job); they are scored with the owner, never edited unilaterally.

## Predictions

### P-20260702d · plug-in guidance measurably off-target, gap grows with β · conf 75% · resolve-by 2026-07-05 · SCORED 2026-07-03: HIT
- **Claim:** [GRF pilot P1] DPS-style plug-in guidance is off-target — over-concentrated,
  under-covering — with the gap growing with guidance strength.
- **Made:** 2026-07-02 · **Context:** GRF pilot overnight run (docs/OVERNIGHT_2026-07-02_GRF_PILOT.md)
- **Resolution criterion:** T1 grid exact W₂/KL + functional coverage vs β.
- **Outcome:** SCORED WITH OWNER 2026-07-03: HIT. DPS bias measured (1.4-28x oracle floor, monotone in beta, absolute W2 also monotone) AND analytically confirmed: independent stiff-ODE (Radau) per-mode prediction matches the measured grid to 0.7-2.8%. gamma*=1.33-1.43; 68% CI coverage -> 0 at strong tilt; d-extensive to d=16384.
- **Lesson:** The analytic cross-check (predict the grid from closed forms with a second, independent implementation) is cheap and should be standard for every oracle-substrate claim. Caveat carried: all numbers conditional on one synthetic y (structure is y-generic per the analytics; multi-y ensemble is a cheap daytime add).

### P-20260702e · reward-as-potential SMC runs cold (γ* > 1) · conf 70% · resolve-by 2026-07-05 · SCORED 2026-07-03: HIT (depth-qualified)
- **Claim:** [GRF pilot P2] the SAP analog reproduces the discrete substrate's over-tilt
  signature in ℝ^d.
- **Made:** 2026-07-02 · **Context:** GRF pilot overnight run
- **Resolution criterion:** γ* fits on T1 output.
- **Outcome:** SCORED WITH OWNER 2026-07-03: HIT, depth-qualified. gamma*(T) rises monotonically with step count (0.21/0.52/1.72 at T=32/64/256, 32^2/0.5sigma/N=256) — the discrete substrate's depth law transfers to R^d and SAP is cold (gamma*>1) at practitioner depths. At strong tilt/high d the pathology saturates into variance collapse without mean tracking before coldness can express.
- **Lesson:** Two: (1) scalar gamma* misleads under population collapse — always report the mean-tracking and variance-ratio split alongside; (2) the step count T was left unpinned by the plan and materially affects the signature — pre-register T (or a T-ladder) explicitly next time.

### P-20260702f · proper twisted SMC on-target with valid Ẑ · conf 85% · resolve-by 2026-07-05 · SCORED 2026-07-03: HIT
- **Claim:** [GRF pilot P3] proper twisted SMC is on-target within finite-N error, Ẑ bracketing
  the analytic Z.
- **Made:** 2026-07-02 · **Context:** GRF pilot overnight run
- **Resolution criterion:** T1 grid + T-G3 gate outputs.
- **Outcome:** SCORED WITH OWNER 2026-07-03: HIT (conjugate-proposal arm, designated pre-data at 23:18 — the variant that satisfies frozen gate T-G3). On-target at 0.96-1.06x floor in all 36 cells, gamma*=1.00, log Zhat = log Z with machine-zero incremental weights; unbiasedness verified separately with real weight variance (d=1, prior proposals).
- **Lesson:** 'Proper twisted SMC' must mean twisted PROPOSALS, not just psi-ratio potentials: the potential-only variant is formally unbiased yet collapses at d>=256 (single-run log Zhat ~400 log-units low) — itself a certificate-relevant demonstration. Ambiguous sampler specs get resolved and logged BEFORE data; that discipline is what made this scoreable.

### P-20260702g · misspecification propagates differently per scheme · conf 55% · resolve-by 2026-07-05 · SCORED 2026-07-03: HIT (mechanism corrected)
- **Claim:** [GRF pilot P4] a contaminated score is amplified by plug-in guidance and partially
  absorbed into weights by proper SMC — the decomposition is non-trivial.
- **Made:** 2026-07-02 · **Context:** GRF pilot overnight run (T2 tier)
- **Resolution criterion:** three-way decomposition at 64².
- **Outcome:** SCORED WITH OWNER 2026-07-03: HIT on the headline (propagation differs per scheme; decomposition non-trivial; smooth eps-ladder, tight seeds), with BOTH named mechanisms corrected: proper SMC does NOT absorb misspecification — conjugate w.r.t. the wrong model, it exactly samples the wrong tilt (pass-through 1:1, W2 0.50->2.57 across eps); DPS interacts sign-dependently (eps=+0.3 amplifies 3.77->5.97; eps=-0.3 cancels to 2.92 with gamma*=0.98). SAP's collapse masks eps entirely.
- **Lesson:** The accidental-compensation trap is real: at eps=-0.3 the temperature diagnostic alone would pass a doubly-broken pipeline (W2 still 6x floor catches it) — audits need multiple, mechanistically-different metrics. The wrongness of the absorption intuition is worth more than the prediction: weights built from the wrong model cannot know it is wrong.

### P-20260702h · KILL BRANCH: bias negligible at realistic settings · conf 25% · resolve-by 2026-07-05 · SCORED 2026-07-03: MISS (kill branch did not fire — GO)
- **Claim:** [GRF pilot P5, REVISED from 40% after deep-read of 2502.07849 — owner's number]
  at realistic β and reward SNR, plug-in bias is negligible in absolute terms at d=4096.
- **Made:** 2026-07-02 · **Context:** the pre-registered NO-GO branch; plan §3 kill criterion.
- **Resolution criterion:** no scheme W₂ > 3× oracle finite-N floor at any (β,d) in the T1 box.
- **Outcome:** SCORED WITH OWNER 2026-07-03: MISS, the good direction. Kill criterion (no scheme >3x floor anywhere) decisively not triggered: DPS alone >=3x from 0.5sigma at N=256 at all d, never below ~3.3x even at 0.125sigma; max frozen-scheme ratio 251x. GO condition of plan par.3 met on both conjuncts.
- **Lesson:** The deep-read-driven revision (40%->25% against) was directionally right and the mechanism-level reasoning (CFG blessing non-transferable, bias d-extensive) held quantitatively. Reading the threat paper closely before the experiment was the highest-value hour of the prep.

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

## Predictions — arms night (2026-07-03 → 04; frozen with owner sign-off before the run; PUBLIC push precedes first GPU job)

### P-20260703b · multi-y: ratio structure is y-generic · conf 85% · resolve-by 2026-07-06 · OPEN
- **Claim:** T1's qualitative structure (scheme-to-floor orderings; DPS monotone growth in β) holds across ≥12 observation realizations, with relative spread of W2/floor ratios ≲10% at N=256.
- **Made:** 2026-07-03 · **Context:** referee-proofing the single-y caveat (docs/OVERNIGHT_2026-07-03_ARMS_NIGHT.md, arm A1).
- **Resolution criterion:** medians/IQR of ratios across y-draws, 64² focus.
- **Outcome:** (pending)
- **Lesson:** (pending)

### P-20260703c · diagnostics blind spot: compensation config fools the field's tests · conf 70% · resolve-by 2026-07-06 · OPEN
- **Claim:** MIRA and TARP detect plain-DPS over-concentration (64², 1σ, N=256) at budgets ≤4096 samples; NO standard sample-based diagnostic (PQMass, MIRA, TARP) flags the ε=−0.3 + DPS accidental-compensation configuration at those budgets (W2 still ~6× floor).
- **Made:** 2026-07-03 · **Context:** arm A2 ("certify the certifiers").
- **Resolution criterion:** power curves vs sample budget + same-vs-same null calibration of each test.
- **Outcome:** (pending)
- **Lesson:** (pending)

### P-20260703d · amortized-conditional: summaries pass, geometry fails · conf 60% · resolve-by 2026-07-06 · OPEN
- **Claim:** a conditional score model trained to convergence on (field, observation) pairs passes summary checks (posterior mean, marginals, band powers within a few %) while geometry metrics (W2 to exact posterior; variance-spectrum ratio) sit ≥3× the oracle floor — the Doeser & Jasche phenomenology reproduced against a closed-form reference.
- **Made:** 2026-07-03 · **Context:** arm A3 (Legin+/D&J class).
- **Resolution criterion:** amortized-arm metrics vs (μ*,Σ*) at matched N.
- **Outcome:** (pending)
- **Lesson:** (pending)

### P-20260703e · Rémy scheme: conservative, K-converging, misspec sign-flipped · conf 55% · resolve-by 2026-07-06 · OPEN
- **Claim:** σ_t²-inflated annealed-Langevin (Rémy-style) runs WARM (γ*<1) at small K (few Langevin steps/level), approaches the target monotonically as K grows; its misspecification interaction is sign-flipped vs DPS (ε<0 aggravates rather than cancels).
- **Made:** 2026-07-03 · **Context:** arm A4 (field's flagship mass-mapper).
- **Resolution criterion:** K-sweep W2/γ* + ε=±0.3 pair at fixed K.
- **Outcome:** (pending)
- **Lesson:** (pending)
