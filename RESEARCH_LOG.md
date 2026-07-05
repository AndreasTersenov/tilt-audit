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

### P-20260704d–f · Stage 2: learned-score block-wise certificates · STAMPED 2026-07-04 (adopted as proposed) · OPEN
- Full wording in docs/PLAN_CERT_LEARNED.md §4 (frozen there): (d, 60%) per-mode ESS
  ≥100/256 on the clean net at 1σ + low-|k| band Spearman ≥0.8 vs band truth, widest band
  degenerates; (e, 70%) analytic-pathway per-mode-summed reading within 20% of exact
  chain-law path KL in the healthy regime; (f, 60%) clean-net dps readings within 2× of
  analytic-pathway dps at every shift. Kill/reshape: median per-mode ESS <10 on the
  learned net ⇒ block-wise arc dies for real nets.
- Gate-time context (pre-data, logged): G-L1/2 green; final-step determinism, clip
  quantification, and the per-mode logN-saturation ceiling documented before the grid.

### P-20260704d–f SCORED 2026-07-04 (owner engaged in-session) — all three MISS; reshape criterion FIRED
- **P-d MISS:** clean-net median per-mode ESS = 3/256 at 1σ (needed ≥100); band0 ordering
  Spearman 0.275 (needed ≥0.8); band2 degenerate as predicted (the one sub-clause that hit).
- **P-e MISS:** analytic-pathway per-mode-sum/exact = 0.043–0.161 across the grid — the
  per-mode log N saturation ceiling binds everywhere at T=256; Part III's near-
  exactification was specific to small per-mode KLs, not to diagonality.
- **P-f MISS by 8–125×:** the learned net's guidance (VJP through net error; clip firing
  0.37–0.72 at the mis nets, present at clean too) dominates the reading. The certificate
  honestly reports that learned-DPS is far from ITS OWN intended target path-wise — but
  the meter's value on real nets is net-error-dominated: the net changes the meaning.

### E-20260704b · Stage 2: block-wise certificates on learned scores
- **Hypothesis/Setup/Expectation:** docs/PLAN_CERT_LEARNED.md (stamped); gates G-L1/2
  green pre-grid (gate work itself yielded: deterministic-final-step caveat with material
  displacement at strong guidance; clip = the exact 0.2% law gap; per-mode ceiling
  ≈ log N nats/mode). 160 rows → results/cert_learned.jsonl.
- **Result:** the naive per-trajectory certificate — joint AND block-wise — does not
  transfer to trained scores; the pre-registered reshape criterion fired (per-mode ESS 3).
  Diagnosis is clean and mechanistic, not mysterious: everything on the Ẑ/exponentiated
  side of the instrument degenerates; the MEAN side survives untouched — sampled
  E[log w] matched the exact chain law to <1 s.e. on every configuration including
  learned pathways (G-L2), and it is unbiased regardless of degeneracy.
- **Updated belief:** the deployable instrument is the mean-side statistic:
  (1) RELATIVE certificates — differences of E[log w] between steering schemes on the
  SAME net/reference cancel the unknown log Z: unbiased, low-variance sampler ranking at
  any d on real nets (the practitioner's actual question: "which guidance scheme is less
  wrong on MY model"); NOT valid across different nets (references differ). (2) The
  per-step rate profile (the E[log w] integrand) as a stability/violence monitor — what
  clip_frac crudely proxied. Absolute KL certificates remain bench-only methodology.
  Next disposable version: validate the relative instrument's operating characteristics
  on the bench (does ΔE[log w] rank schemes consistently with true ΔKL across the grid?
  — computable from EXISTING cert_killtest + cert_learned rows, zero new GPU) before
  any new framing. The negative results are the motivation section, not a failure to
  report: "runtime certificates for guided diffusion are harder than they look; here is
  what survives, measured against an exact oracle."

### P-20260704g · relative certificate validation (analysis of existing rows; predictions made pre-analysis, owner go-ahead in-session) · OPEN
- (a, 85%) sampled ΔE[log w] between samplers (same reference) recovers exact Δpath-KL
  within 2 s.e. across the exact-score grid; (b, 70%) path-KL ordering = endpoint-KL
  ordering in ≥90% of comparable pairs; (c, 60%) on learned nets, relative-certificate
  ranking (dps vs unguided, same net) matches empirical endpoint-W2 ranking at all
  shifts/nets. Resolution: scripts/cert_relative_digest.py on cert_killtest.jsonl +
  cert_learned.jsonl. No new compute.

### P-20260704g SCORED 2026-07-04 — (a) HIT, (b) HIT, (c) MISS ⇒ the deployable-instrument claim dies at endpoint-relevance
- (a) 40/40 pairs within 2 s.e. (median 0.69): ΔE[log w] is an exact, degeneracy-immune
  estimator of Δpath-KL — the estimator was never the problem.
- (b) 100% path/endpoint order agreement on the EXACT grid.
- (c) MISS 11/12: on every learned net the ranking flips — guided-DPS travels ~2e6 nats
  farther (true, exactly measured) while ENDING closer to the truth. Path divergence is
  not endpoint-relevant on real nets. The analytic-pathway control agrees 4/4, isolating
  the mechanism: learned-net path violence, the same quantity that killed Stage 2.

### E-20260704c · certificate line: full-arc close-out (three disposable rounds, one day)
- **Result of the arc:** every deployable variant of the runtime certificate is now dead
  by its own pre-registered test, with exact diagnoses: absolute certificates (loose
  ~100x for bad samplers; Ẑ side degenerate at any feasible N); importance repair (path
  >> endpoint, dead even at 16²); block-wise on real nets (per-mode ESS 3; net violence
  dominates); relative mean-side (exact estimator, but path order ≠ endpoint order on
  learned nets). What stands: bench-side absolute pricing (tight-when-green/loud-when-red
  vs exact references — a method-development tool, not a deployment monitor) and the
  per-step rate profile as an unvalidated engineering diagnostic.
- **Updated belief:** the pain point (no runtime correctness signal for guided diffusion)
  is real but not solvable by path-space importance machinery on trained nets — the
  path/endpoint gap is the fundamental obstruction, measured at 8–125x and
  order-breaking. Candidate next framings, NOT yet chosen: (1) write the negative arc as
  the paper — "runtime certificates for guided diffusion: exact-oracle post-mortem of the
  obvious constructions" (forecloses the paths anyone would try first; strongest material:
  the exact bound-anatomy and the three pre-registered kill rounds); (2) endpoint-space
  instruments instead of path-space (e.g. learned-twist consistency checks, sample-based
  self-diagnostics calibrated on the bench — overlaps direction B); (3) return to parked
  directions B (adversarial blind-spot minimax) / C (bias→parameter transfer) / D
  (mixture oracle), which the arena supports today. Decision with owner.

### P-20260704h · kill-verification pass (owner-requested adversarial check of the certificate kill) · OPEN
- (i, 95%) end-to-end code verification vs closed form through the net-callable path
  (analytic score with controlled contamination as a synthetic wrong net): mean(logw)
  matches its exact chain law within 2 s.e.
- (ii, 55%) guidance-scale ladder on the clean net (0.1/0.25/0.5/1.0): at 0.25-0.5 the
  certificate partially revives — KLhat drops 10-100x, median per-mode ESS > 50.
- (iii, 50%) THE decisive one: guided-vs-guided relative ranking (ΔE[log w] across the
  scale ladder, same net) agrees with empirical endpoint-W2 ordering at every shift.
  HIT un-kills the relative instrument for its actual use case; MISS confirms the kill
  at the use-case level. Also: T=64 column for accumulation-length sensitivity.

### P-20260704h SCORED 2026-07-04 — kill VERIFIED: (i) HIT, (ii) MISS, (iii) MISS
- (i) accumulation code verified end-to-end (synthetic wrong-net through the callable
  path vs closed form: z=-0.65 at 24 seeds; the earlier 2-4 s.e. "offset" was
  small-sample s.e. noise). No simple mistake.
- (ii) no practitioner-regime rescue: gscale 0.1-1.0 all keep per-mode ESS 2-8, clip
  0.24-0.59; T=64 is WORSE (clip 0.95, KLhat 2.7e9).
- (iii) no use-case rescue: guided-vs-guided ranking by mean(logw) is ANTI-correlated
  with endpoint W2 at mild tilts (rho=-0.80; cert always picks the weakest guidance,
  truth picks mid-scale). Path divergence does not track endpoint quality even within
  a guided family on one net. My own "relative instrument survives" claim from the
  P-g round is hereby corrected: it survives the exact grid only.
- **The certificate kill is confirmed against all three identified escape routes.**
  Negative-arc write-up is now paper-grade: verified mechanism, no rescue found at
  (implementation | guidance scale | horizon | comparison class).

### NOTE 2026-07-04 · literature sweep (agent-verified, ~25 searches): novelty + positioning of the certificate arc
- **Not scooped:** no published use of path-space residual IW as a standalone
  DIAGNOSTIC for guided diffusion (only as correctors: TDS 2306.17775, RNE 2506.05668);
  no published path-KL≈100×-endpoint-KL quantification; no published anatomy of weight
  degeneracy on trained nets (folklore acknowledgments only: 2506.05231, 2601.21951).
  The negative-result paper is alive.
- **The motivational niche is partially occupied — and the owner's skepticism about the
  Stein direction was RIGHT:** score-KSD (2602.04189, Feb 2026) already ships a
  ground-truth-free posterior-fidelity diagnostic for diffusion inverse solvers via
  kernel Stein discrepancy on the endpoint score field, incl. the accuracy≠fidelity
  finding. ⇒ Idea ledger: "build endpoint/Stein instrument" KILLED (superseded — done by
  others); REBORN as "bench-audit score-KSD": run THE existing working certificate on our
  archives where truth is exact — power curves, false-certification rate, blind spots
  (mixture oracle mode-blindness is the natural adversarial case). This is the A2
  certify-the-certifiers machinery pointed at the field's newest tool, and the natural
  experimental companion to the negative-result paper.
- **Paper framing (referee-proofing):** NOT "first ground-truth-free certificate"
  (rejection via 2602.04189). Instead: "the mechanism-obvious path-space certificate —
  the first thing an SBI person reaches for — fails on trained nets; anatomy (net-gradient
  noise; path-KL ~100× looser than endpoint-KL, the single most citable number); why
  endpoint-space instruments are the route that survives." Companions, not competitors:
  gold-standard benchmarks 2509.12821 / 2503.03007 (they show samplers are wrong; we show
  the cheap self-certificate can't detect it).
- **Validation-practices map confirmed** (8-paper sample, table in agent report): ML
  papers point-metrics only (DPS/DDRM/ΠGDM: PSNR/SSIM/LPIPS/FID); science papers
  distributional-but-offline on sims (Remy posterior-predictive on sims; newer astro:
  TARP/SBC on sims); distributional validation on REAL observations: none found —
  state as absence-of-evidence with the structural reason.

## Predictions — certifier trial + transfer night (2026-07-04 → 05; FROZEN at owner sign-off; public push precedes first GPU job)

### P-20260704i · score-KSD detects the standard failures · conf 70% · resolve-by 2026-07-07 · SCORED 2026-07-05: HIT
- **Claim:** with the TRUE target score, score-KSD passes its null gate and detects dps,
  sap, AND the ε=−0.3 compensation config on the GRF archives with power ≥0.9 at budgets
  ≤1024 (empirically calibrated α=0.05).
- **Resolution criterion:** A-null + A-power grids (docs/OVERNIGHT_2026-07-04_CERTIFIER_TRANSFER_NIGHT.md §2).
- **Outcome:** SCORED WITH OWNER 2026-07-05: HIT, with margin — power 1.00 at EVERY
  budget incl. 64 for all three configs (ratios 2.8–19× null); the compensation config
  that fooled γ*/TARP/MIRA is loud to the true-score instrument. Correct non-detection:
  twisted@1σ (its true damage IS the floor). Lesson: score-space sensitivity to SCHEME
  bias exceeds sample-space tests by an order of magnitude in budget.

### P-20260704j · missed-mode blindness · conf 70% · resolve-by 2026-07-07 · SCORED 2026-07-05: HIT
- **Claim:** on the exact 2-component mixture (50/50 weights), a sampler covering only
  one mode goes UNDETECTED by score-KSD at α=0.05 in ≥50% of reps at budgets ≤1024,
  while PQMass and TARP flag it at ≥0.9 power on the same sets.
- **Resolution criterion:** A-mixture grid + T-M1 gate.
- **Outcome:** SCORED WITH OWNER 2026-07-05: HIT, total. KSD undetected in 91–100% of
  reps at every budget AND at N=16,384 (paired control: plus/both = 1.0001–1.0007);
  weight-swap equally invisible. PQMass 1.00; TARP 0.90 (FP nominal after 40-null
  recalibration). Ladder: PQMass detection dies between 95/5 and 99/1 weights; KSD
  flat at FP across the whole ladder. Lesson: Stein-type mode-blindness is TOTAL on a
  4096-dim field posterior — "more samples" does not exist as a cure.

### P-20260704k · wrong-reference false-certification · conf 65% · resolve-by 2026-07-07 · SCORED 2026-07-05: HIT
- **Claim:** with a contaminated reference score (ε=−0.3, analytic or mis-trained net),
  score-KSD reads the MATCHED wrong sampler as null-consistent (≤1.5× the null's 95%
  quantile) while its true damage is ≥3× floor — the deployment-configuration trap.
- **Resolution criterion:** A-wrongref grid.
- **Outcome:** SCORED WITH OWNER 2026-07-05: HIT (owner call: criterion met as
  written; refinement logged, not penalized). Analytic: twisted_em03 (proper sampler
  of its wrong target, true damage 17.2× floor) reads 0.99–1.00× — and across a BAND
  of references (ε_ref −0.3…−0.2); dps_em03 ESCAPES analytically (scheme bias shows
  through, 3.8×). Net reference: BOTH matched samplers ≤1.2× — and beyond the claim,
  even the CLEAN net false-certifies plain dps (0.97×, null inflated 1.9×; σ-ladder:
  the paper's own σ=0.3 misses dps, σ=0.1 would catch it at 3×-degraded effect size).
  Plus the false-alarm side: ε_ref=−0.05 flags perfect samples at 1.00. Lesson: in
  deployment the reading is REFERENCE-error-dominated in both directions; the trap's
  precise home is proper-sampler + matched-wrong-score, and net-quality scores put
  everything in that regime.

### P-20260704l · gold standards are manufacturable at 64² · conf 75% · resolve-by 2026-07-07 · SCORED 2026-07-05: HIT
- **Claim:** NUTS on the nonlinear-forward-model posterior passes T-L1 (Gaussian-limit
  match), T-L2 (R-hat<1.01, ESS>400), T-L3 (seed independence) at 64² within the H4 box.
- **Resolution criterion:** gate outputs, logged with numbers.
- **Outcome:** SCORED WITH OWNER 2026-07-05: HIT — all gates green by ~H1 (box H4).
  T-L1 three legs (λ=1e-4 at both resolutions; closed-form-IS cross-check max|z|=3.29
  sd(z)=0.97; mean-offsets ∝ λ over two decades); T-L2 26/26 (R-hat ≤ 1.0006, ESS_min
  > 10k); T-L3 clean. 74 s per 64² config; 128² also passed (119 s). Lesson: gold
  standards on this class of nonlinear posterior are ~1 GPU-minute each — "too
  expensive to validate" is empirically false at bench scale; the whitened z-basis
  parameterization is what makes NUTS this efficient.

### P-20260704m · DPS overconfidence transfers to the nonlinear substrate · conf 65% · resolve-by 2026-07-07 · SCORED 2026-07-05: MISS (instructive)
- **Claim:** vs MCMC gold standards, DPS shows band-power 68%-coverage ≤0.5 at the strong
  tilt and the damage ordering dps > dps-inflated ≈ remy@K=100 is preserved at every
  gold-standard config (MMD and sliced-W2 agree on the ordering).
- **Resolution criterion:** transfer grid vs gold (§3).
- **Outcome:** SCORED WITH OWNER 2026-07-05: MISS by the letter (the conjunction fails),
  instructive by content. Coverage clause: loud HIT (0.00–0.08 at strong tilt; high-k
  bands 0.00 even at mid). dps > dps_inflated: holds everywhere. BUT dps_inflated ≈
  remy100 FAILED at every config — the linearized inflation's advantage decays
  11.6× → 2.8× → 1.1× at skewness 0.5 → 1 → 2 (λ decay law; same shape at strong
  tilt) while remy100 stays ≤15× floor throughout. Metric agreement: everywhere except
  one 10%-level wobble (dps vs dps_inflated, n64 mid). Lesson: exact_guidance's
  exactness was a GAUSSIAN ACCIDENT — I predicted the correction would stay
  remy-grade off-Gaussian and it does not; plug-in covariance corrections die with
  non-Gaussianity, Langevin refinement does not. The miss produced the transfer
  chapter's best figure.

### P-20260704n · Rémy K-convergence transfers · conf 75% · resolve-by 2026-07-07 · SCORED 2026-07-05: HIT
- **Claim:** remy MMD-to-gold decreases monotonically in K ∈ {5,30,100} at every
  gold-standard config.
- **Resolution criterion:** transfer grid vs gold (§3).
- **Outcome:** SCORED WITH OWNER 2026-07-05: HIT — monotone at every config, both
  resolutions, every λ and y-draw, with order-of-magnitude separation between K
  levels; K=100 sits at/near the gold floor (mmd2 ~0 at 32² mid). KSD-side bonus: the
  true-score instrument resolves even remy100's documented +2.6% ULA stationary excess
  (reads 1.008–1.015). Lesson: "pay more compute, get closer" is the property that
  transfers; it is the robust axis a practitioner should buy quality on.

### E-20260705a · certifier trial + transfer night (Results + Updated beliefs DRAFT; Outcomes/Lessons reserved for joint scoring)
- **Hypothesis/setup:** the six frozen predictions P-20260704i–n; plan
  docs/OVERNIGHT_2026-07-04_CERTIFIER_TRANSFER_NIGHT.md; all gates green before burn
  (T-K1, T-M1, T-L1 ×3 legs, T-L2 26/26, T-L3). No reference code exists for
  arXiv:2602.04189 — implemented their exact recipe in JAX (V-statistic,
  (1/N)√(ΣH/d), IMQ β=−1/2, c=1/(median(singvals A)+1)), autodiff-verified to 1e-13;
  our 60-null empirical α=0.05 calibration is strictly more than the paper does
  (it has none).
- **Results (headline numbers):**
  (i) TRUE-score KSD: dps/sap/dps_em03 detected at power 1.00 at ALL budgets
  (ratios 2.8–19× null) — including the ε-compensation config every sample-based
  test missed. twisted@1σ "missed" correctly (its true damage IS the floor).
  (ii) MODE-BLINDNESS: one mode of a 50/50, 12σ-separated mixture missing →
  KSD reads 1.00× null at every budget through N=16,384 (paired 16k control:
  plus/both = 1.0001–1.0007). PQMass flags at 1.00 (dies only at 99/1 weights);
  TARP 0.90 at 50/50 (FP-elevated cell noted). Weight-swap equally invisible to KSD.
  (iii) WRONG-REFERENCE: false-certification is a BAND (twisted_em03, true damage
  17.2× floor, reads ≤1.0× null for ε_ref ∈ {−0.3, −0.2}); false-alarm cliff:
  ε_ref=−0.05 flags perfect samples at 1.00. DPS's scheme bias shows through any
  analytic reference (3.8×).
  (iv) DEPLOYMENT (net score, the paper's own construction): null inflates 1.9×;
  dps (30× floor damage) reads 0.97× → detect 0.00 with BOTH nets — the deployment
  configuration false-certifies the diagnostic's textbook target; only sap (159×
  floor) clears the net-noise. ε=±0.3 net contamination is a rounding error next to
  generic net score error. On the lognormal substrate the within-task RANKING
  (the paper's claimed use) inverts: dps at 1.07 vs remy30 at 1.76 against a 14×
  true-damage gap the other way.
  (v) GOLD STANDARDS: 26+ NUTS configs at 64²/32² (74 s / 43 s per config,
  R-hat ≤ 1.0006, ESS_min > 10k), T-L1 via λ=1e-4 limit + closed-form-IS cross-check
  (max|z|=3.29, sd(z)=0.97) + λ-scaling law (offsets ∝ λ across two decades);
  T-L3 seed-independence clean. 128² stretch ran as filler.
  (vi) TRANSFER: dps overconfidence transfers band-structured (high-k coverage
  0.00 even at mid tilt; ≤0.08 everywhere at strong). remy K∈{5,30,100} monotone
  → floor at every config. THE BREAK: dps_inflated ≈ remy100 fails off-Gaussian —
  the linearized inflation's advantage decays 11.6× → 2.8× → 1.1× at skewness
  0.5 → 1 → 2 (the exact_guidance-was-exact property is a Gaussian accident).
  Learned-net DPS on the κ-substrate: clip 0.66–0.99, coverage collapse at strong
  tilt — the practitioner pathway degrades as on the Gaussian bench.
- **Updated beliefs (draft):** the "runtime certificate" niche is now empirically
  mapped from both directions: path-space certificates die of weight degeneracy
  (E-20260704c), and the surviving endpoint certificate (score-KSD) is (a) blind to
  the failure mode MCMC-free samplers are most suspected of (missed modes), (b) in
  deployment dominated by reference-score error in both directions (false certs +
  false alarms), and (c) unable to rank the practically relevant samplers. What
  actually works, measured: manufactured MCMC gold standards on a nonlinearized
  bench (cheap: ~1 GPU-min/config) + sample-space two-sample tests (PQMass) against
  them — and the Rémy-style Langevin route is the sampler whose quality is robust
  to non-Gaussianity while plug-in corrections are not. Paper structure: three acts
  (true-score power / structural blindness / deployment fragility) + the transfer
  chapter with the decay law.

## Brainstorm exit — 2026-07-05 · stress: K-vs-2K convergence certificate

- **Decision:** PURSUE, minimally scoped — as the constructive FINAL ACT of the
  certificate-audit story (blog + eventual paper), one-day box. Standalone-paper
  version PARKED (cause: outcompeted by the folded-in framing; its realistic ceiling
  is a validation contribution and the owner's visibility goal doesn't close).
  Ownership note, recorded honestly: the idea was Claude-proposed; owner authorized
  the small version without adopting the why ("I don't really see the value") —
  scope was cut to match partial ownership, per the absorbed-idea guard.
- **Predictions:**
  - P-20260705b · the calibrated K-vs-2K test tracks true convergence · conf 80% ·
    resolve-by 2026-07-09 · OPEN. Claim: PQMass(remy@K, remy@2K) with empirical null
    calibration reads null at K≥50 AND detects at K≤15 with power ≥0.9 on ≥2 gold
    configs (64² mid+strong) — i.e., the truth-free certificate agrees with the
    gold-referenced convergence measured tonight. Criterion: the one-day script's
    table.
  - P-20260705c · the false-certification boundary is demonstrable · conf 75% ·
    resolve-by 2026-07-09 · OPEN. Claim: there exists T* ≤ 512 where dps@T* vs
    dps@2T* passes the agreement test while dps MMD-to-gold stays >50× floor —
    the measured "R-hat lies outside the asymptotically-exact class" exhibit.
    Criterion: same script; plus the mixture stuck-mode agreement demo (construction,
    not prediction).
- **Kill criteria:** drop if (a) the test cannot achieve nominal FP at converged K
  AND ≥0.9 power at K/4 within the one-day box (blind or noisy in the regime that
  matters), or (b) the 30-min literature check finds this exact validated
  construction already published for annealed/diffusion samplers (then cite,
  don't build — cheaper win, same ending).
- **Cheapest next experiment:** one script (~150 lines, reuses run_transfer +
  pqm): remy K ∈ {10,15,25,50,100,200} paired-seed runs at 2 gold configs +
  dps T-ladder {64..512} + mixture stuck-mode demo; PQMass null via split-halves.
  ~1 GPU-hour total. Precondition: the 30-min literature check (SMC/annealed
  convergence diagnostics, R-hat lineage, diffusion-NFE plateau practices).
- **Disconfirming evidence surfaced (Darwin lines, idea survives WITH these):**
  - The trick is folklore-adjacent (Gelman–Rubin lineage; "increase NFE until
    plateau" is common practice) — the delta is the calibrated test + measured
    false-certification boundary, i.e., validation, not invention.
  - Inherits the stuck-run blind spot (agreement ≠ correctness; missed modes
    certify) — measurable, but caps every claim.
  - Naive cost is 3× the sampler being certified; the valid class (asymptotically
    exact) is used by the already-careful crowd. Subset-chain variants mitigate.
  - Owner's visibility motive does not close for the standalone version — hence
    the scoping.
- **Idea ledger:** born: K-vs-2K certificate (alive, scoped as story ending).
  Parked: standalone convergence-certificate paper (outcompeted by folded framing).

- **NOTE 2026-07-05 · K-vs-2K literature gate: BUILD.** Agent-verified: no published
  calibrated, ground-truth-free, two-budget distributional convergence test for
  annealed/diffusion samplers; kill criterion (b) of the 2026-07-05 brainstorm exit
  does NOT fire. Nearest neighbors, in order of danger for the novelty claim: AIDE
  (1705.07224 — budget-vs-gold KL, algorithm-intrusive, no calibration), R* (Lambert
  & Vehtari 2022 — classifier two-sample MCMC diagnostic, same-budget stationary
  chains, soft calibration), bidirectional MC/BREAD (1606.02275 — simulated-data
  divergence bounds for annealed samplers), L-lag couplings (1905.09971), wild-
  bootstrap MMD for dependent samples (1408.5404 — the calibration ingredient),
  adaptive-solver step-doubling (the intra-sampler ancestor), Richardson-Romberg
  (the old-literature ancestor, scalar expectations only). Rémy et al. 2023 confirmed
  to use NO formal convergence diagnostic (schedule chosen empirically). The measured
  false-certification boundary exists in NONE of the neighbors. Every qualifier in
  "first calibrated, ground-truth-free, non-intrusive two-budget distributional test
  with measured power and false-certification boundary" is load-bearing.
