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

### E-20260703b · arms night (A1 multi-y, A2 diagnostic power, A3 amortized, A4 Remy)
- **Hypothesis:** per-claim splits in P-20260703b–e (frozen below; publicly timestamped
  at github.com/AndreasTersenov/tilt-audit before the first GPU job).
- **Setup:** this repo at the arms-night commits; GPUs 0–2 (titan); plan
  docs/OVERNIGHT_2026-07-03_ARMS_NIGHT.md; gates T-G1..15 + T-N1/N2/N3 green pre-burn;
  chronology NIGHT_LOG_2026-07-04.md; deliverables HANDOFF_DAWN_2.md.
- **Expectation:** encoded in P-20260703b–e.
- **Result (drafted at dawn; scoring pending):** A1: ratio structure y-generic across 24
  draws (rel IQR ≤8.5%, orderings 24/24) — P-b HIT-shaped. A2: every oracle-made failure
  including the ε=−0.3 compensation config is detected by PQMass/TARP/MIRA at budgets
  64–256 (P-c blind-spot clause MISS-shaped); the trap fools exactly the γ* scalar
  (γ*=1.001 at ε*=−0.28±0.01, W2 basin ~5.8× floor); two community tools (tarp-0.1.0,
  mira-score-0.1.7) have the same truth-based-normalization bug, d-extensive at q=4096,
  found by the null gates and repaired by symmetric pooled-sample standardization (MIRA
  null then hits its analytic value to 4 decimal places). A3: default conditional net
  lands at 2.56× floor with summaries at the few-% level (px variance −7.5% visible);
  ladder saturates by ~6–15k steps; failure onset 2k→6k (variance collapse) — P-d
  geometry clause MISS-shaped, the good-for-amortization direction. A4: W2(K) strictly
  monotone to 1.0–1.2× floor by K=100 with the exact-inflation anchor at 1.00×; misspec
  sign-flip confirmed (ε=−0.3 aggravates); small-K state = fat variance + mean overshoot
  simultaneously (scalar γ* uninformative there) — P-e HIT-shaped with the nuance.
- **Updated belief:** the audit's three-way separation works as a production tool, not
  just a demo: it found quantitative bugs in the field's own diagnostics (the certifiers
  need certifying, literally), located the exact ε where the temperature scalar lies, and
  priced the field's flagship conservative sampler in NFE against its aggressive default.
  The program's headline sharpens: sample-based diagnostics are fine (better than
  predicted) — scalar summaries and unaudited preprocessing are the failure surface;
  amortization is not the weak link at this difficulty, steering is.

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

## Brainstorm exit — 2026-07-04 · post-arms-night redirect ("what would actually matter")

- **Context:** owner's verdict on the arms night: internally solid, externally thin —
  the arena is too easy for the questions that matter; confirmatory diligence, not field
  news. Session ran choose→expand→stress on "what is the interesting program".
- **Decision:** PURSUE direction A — **runtime steering certificates for guided diffusion**
  (per-run, per-observation, reference-free error meter: residual path-space weights of the
  sampler against its own intended target; Ẑ + ESS + Jensen-gap KL bound + k-hat; optional
  reweight/repair). Audience: diffusion-methods ML community first (owner's call), evals
  world second. The Gaussian arena is reframed as the CALIBRATION BENCH for the instrument
  (measuring tightness, false-certification rate, informativeness region) — the toy is the
  NIST lab, not the product. Directions B (adversarial least-detectable failures), C
  (sampler-bias→parameter-bias transfer), D (Gaussian-mixture multimodal oracle) parked:
  B+C as one later field-facing paper; D folds in as the missed-mode case generator for
  the certificate's false-certification table.
- **Positioning risks named:** adjacency to twisted-SMC / importance-corrected guidance
  (Wu+ TDS) — claim is the VALIDATED INSTRUMENT with measured operating characteristics,
  not "reweighting exists"; "IS diagnostics lie" (missed modes, PSIS/k-hat literature) —
  import, don't reinvent, and measure the lying rate against the oracle; scope honestly as
  certifying the steering link only (model link = Doeser–Jasche; support link = Diao–Seljak).
- **Disconfirming evidence (Darwin lines):** (1) owner judges the arms-night results
  uninteresting to the field — logged as evidence about problem choice, not execution;
  (2) amortized arm's P-d MISS is partially an artifact of the substrate's easiness
  (linear posterior map) — weakens any "amortization is fine" claim beyond the toy;
  (3) known failure mode of the proposed instrument: high-ESS lies under missed modes.
- **Predictions (STAMPED BY OWNER 2026-07-04, adopted as proposed; FROZEN):**
  - P-20260704a · conf 60% · SCORED 2026-07-04 (owner authorized in-chat): **HIT,
    stronger than predicted** — residual ESS ≡ 1.0 (not merely <10) at every dim and
    N up to 65536, yet KLhat is strictly monotone in true damage at EVERY dim
    including 128², and exact_guidance certifies below dps at all shifts/dims. The
    meter is directional-and-honest all the way to d=16384.
  - P-20260704b · conf 65% · SCORED 2026-07-04: **MISS (instructive)** — repair by
    residual reweighting HURTS even at 16² (W2 1.36→3.57): exact chain-law shows dps
    path-KL = 2701 nats at 16²/1σ vs endpoint 28.5 — the path deviates ~95× more
    than the endpoint, so no feasible N buys a usable population. Lesson: the
    certify-or-repair framing dies; certify-and-rank survives.
  - P-20260704c · conf 75% · SCORED 2026-07-04: **HIT by the frozen criterion**
    (pooled Spearman 0.950 ≥ 0.9, n=256 rows) with the honest caveat that
    within-dim rho degrades with scale (0.95/0.87/0.81/0.74 at 16/32/64/128²).
- **Kill criterion:** directional signal lost at d≥1024 (certificate non-monotone in true
  damage) AND not recovered by per-mode Rao-Blackwellization ⇒ drop the flagship framing;
  fall back to "certificates and their limits" short paper; revisit B/C.
- **Cheapest next experiment (~half a day, this repo):** accumulate closed-form residual
  log-weights along DPS/sap/remy trajectories (both kernels known Gaussians per mode);
  emit per-run Ẑ vs analytic Z, ESS, k-hat, Jensen-gap KL estimate; compare against the
  true KL/W2 already in the grid; repair-by-resampling column. Gate: existing suite stays
  green; certificate validated first on twisted (must read ~exact) and terminal_is (its
  weights ARE the K=∞ case).
- **Idea ledger:** BORN: steering certificates (A); adversarial blind-spot minimax (B,
  parked); bias→parameter transfer (C, parked); mixture oracle (D, parked/absorbed into A's
  false-certification table). KILLED: none. The arms-night arms are COMPLETE, not dead —
  they become the bench's documentation.

### E-20260704a · certificate kill test (weight degeneracy at scale)
- **Hypothesis:** P-20260704a–c (stamped pre-run; plan docs/PLAN_CERT_KILLTEST.md, incl.
  §7 amendments logged before the grid ran).
- **Setup:** tilt_audit/certificate.py (exact chain-law + sampled instruments), gates
  G-C1..3 green pre-grid; 626 rows on GPUs 1–2 → results/cert_killtest*.jsonl; digest
  scripts/cert_digest.py; figure figures/cert_killtest.png.
- **Expectation:** encoded in P-20260704a–c.
- **Result:** kill criterion NOT triggered — the flagship survives, reshaped. Verdicts
  above (a HIT+, b MISS, c HIT-with-caveat). Beyond the frozen questions, four findings:
  (1) exact bound anatomy: the path-KL certificate is TIGHT for good samplers (ratio
  1.01–1.05 at every d incl. 128²) and loud-but-loose for bad ones (13–95×, because the
  guided path deviates far more than its endpoint) — "tight when green, loud when red";
  (2) per-mode Rao-Blackwellization is not just a rescue but near-exactification: per-mode
  ESS 110–253/256 where joint ESS ≡ 1, and the per-mode summed certificate reproduces the
  exact path-KL to 4 significant figures for exact_guidance; ordered and O(true) for dps;
  (3) attribution scope measured both ways: with a contaminated score the certificate
  prices steering only (blind to model error in BOTH directions: under-reads at ε=+0.3,
  over-reads at −0.3) — the composability demo the skeptic asks for; (4) Ẑ recovery and
  importance repair are dead at scale for any feasible N (ESS ≡ 1.0 at N=65536), incl.
  the AIS variant for the Remy scheme (log Ẑ gap −10⁵ nats).
- **Updated belief:** the runtime-certificate flagship is alive with a sharper product
  shape: certify-and-rank (detection, comparison, exact pricing of good samplers), with
  per-mode/blockwise decomposition as the tightening arc (in the wild: wavelet/scale
  blocks — the Starck-lineage connection), explicitly NOT repair. Next experiments in
  order: (i) block-wise certificates on a LEARNED score at 64² (does near-exactification
  survive off-diagonal correlations?), (ii) the false-certification table on Gaussian
  mixtures (missed modes), (iii) the wrapper demo on a public pretrained net.
