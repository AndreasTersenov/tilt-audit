# HANDOFF — DAWN 3 (certifier trial + transfer night, 2026-07-04 → 05)

> Night log: NIGHT_LOG_2026-07-05.md (H0 = 23:18 UTC; compute phase closed ~02:15 —
> the entire program, ladder included, fit in H3 of the 10 h window).
> Plan: docs/OVERNIGHT_2026-07-04_CERTIFIER_TRANSFER_NIGHT.md. All gates green
> before burn. All scoring below is PROPOSED — final scoring happens with Andreas.
> Data: results/ksd_trial.jsonl (5,395 rows), transfer.jsonl (1,730),
> mixture_contrast.jsonl (486), results/gold/ (38 configs incl. 128²).
> Figures: figures/fig_{ksd_power,mixture,wrongref,transfer,remyK}.png.

## 1. Verdict table — P-20260704i–n (PROPOSED)

| P | Claim (short) | Proposed | One-line evidence |
|---|---|---|---|
| i | KSD detects standard failures incl. compensation | **HIT** | dps/sap/dps_em03 power 1.00 at ALL budgets incl. 64 (ratios 2.8–19× null); the compensation config that fooled γ*/TARP/MIRA is loud to the true-score KSD |
| j | missed-mode blindness | **HIT (total)** | mix_plus (50/50, 12σ apart, one mode gone) reads 1.00× null at every budget through N=16,384 (paired 16k: plus/both = 1.0001–1.0007); PQMass 1.00, TARP 0.90 on the same banks (frozen w=0.5 criterion fully met) |
| k | wrong-reference false-certification | **HIT (refined)** | twisted_em03 (true damage 17.2× floor) reads ≤1.0× null under the matched-wrong ref — and across a BAND (ε_ref −0.3…−0.2); refinement: dps_em03 escapes (scheme bias visible through any analytic ref, 3.8×); plus a false-alarm cliff (perfect samples flagged 1.00 at ε_ref=−0.05) |
| l | gold standards manufacturable at 64² | **HIT** | Full gate set green ~H1 (box was H4): T-L1 (λ=1e-4 both res; closed-form-IS cross-check max|z|=3.29, sd(z)=0.97; offsets ∝ λ over two decades), T-L2 26/26 (R-hat ≤ 1.0006, ESS_min > 10k), T-L3 clean; 74 s per 64² config; 128² stretch also passed (119 s) |
| m | DPS overconfidence transfers | **HIT, ordering clause partial** | Coverage clause loud: bp-coverage 0.00–0.08 at strong tilt (0.00 on high-k bands even at mid). dps > dps_inflated holds everywhere; **dps_inflated ≈ remy100 BREAKS** — see decay law below. mmd2/swd2 agree on all orderings except dps-vs-dps_inflated at n64 mid (10%-level; discuss at scoring) |
| n | Rémy K-convergence transfers | **HIT** | K 5→30→100 monotone at every config, both resolutions, every λ; K=100 sits at/near the gold floor (mmd2 ~0 at 32² mid; ≤15× floor at every λ) |

## 2. Beyond the frozen grid (checkpoint zooms, tagged exploratory)

- **Deployment columns (the paper's own score construction, σ=0.3, M=4):** the null
  inflates 1.9× (net score error dominates) and **dps — 30× floor damage, the
  diagnostic's textbook target — reads 0.97× → false-certified with BOTH nets.**
  Only sap (159× floor) clears the net-noise. ε=±0.3 net contamination is a rounding
  error next to generic net score error (s_clean ≈ s_mis readings).
- **σ-ladder (referee-proofing):** the false-certification is σ-dependent — σ=0.1
  catches dps (1.00 at ratio 1.11, still 3× degraded vs true score), the paper's
  σ=0.3 misses it, σ=0.6 worse. Headline stands as "at the paper's own recommended
  construction"; σ-tuning partially rescues but is a priori unknown.
- **ε-ladder:** false-certification is a band, not a point; false-alarm curve
  saturates at ε_ref=−0.05. Reference error dominates the reading in both directions.
- **Weight ladder:** PQMass detection of the missing mode dies between 95/5 and 99/1
  (N=1024) — the practitioner envelope; KSD flat at FP across the entire ladder.
  TARP FP-cell resolved: 40-null calibration gives nominal FP (0.30 was quantile
  noise); TARP 0.90 at 50/50 stands, blind at w≥0.8.
- **A-lognormal ranking inversion:** KSD ranks dps (1.07) below remy30 (1.76)
  against a 14× true-damage gap the other way — variance-collapse damage is
  near-invisible to score-KSD; the paper's within-task-ranking use misranks the two
  most practically relevant samplers. remy100 reads 1.008–1.015 — consistent with
  resolving the documented +2.6% ULA stationary-variance excess.
- **λ decay law (the ordering-break zoom):** dps_inflated's advantage over dps =
  **11.6× / 2.8× / 1.1×** at λ = 0.16 / 0.314 / 0.5 (skewness 0.5 / 1 / 2), same
  shape at strong tilt; remy100 stays ≤15× floor at every λ. The exactness of the
  inflated correction is a Gaussian accident; the Langevin route is
  nonlinearity-robust.
- **Learned-net transfer columns:** dps_ln_clean/mis at mid: mmd2 ~0.10-0.11 (clip
  0.66-0.71); at strong: 0.58-0.60 with clip 0.89-0.99 and coverage collapse —
  the practitioner pathway degrades with tilt exactly as on the Gaussian bench.
  (Metric note: MMD saturates for far-off samples; sliced-W2 keeps resolving —
  both reported.)

## 3. The night's story (draft for the paper's three acts + transfer chapter)

Score-KSD with the true score is genuinely powerful against scheme bias — it catches
the compensation config every sample-based test missed (act 1) — but it is
structurally blind to missing modes at any budget (act 2), and with the score a
practitioner actually has, the reading is dominated by reference error in both
directions: matched errors are false-certified across a band, small reference errors
false-alarm on perfect samples, and at the paper's own recommended construction the
classic biased sampler is certified clean (act 3). Meanwhile, MCMC gold standards on
a nonlinear-forward-model substrate cost ~1 GPU-minute per config with every gate
passable (chapter 4), and against them: DPS overconfidence transfers band-structured,
Rémy's K-convergence transfers cleanly, and the linearized covariance correction
that was exact on the Gaussian bench decays to worthless by skewness 2 — measured as
a three-point decay law.

## 4. Incidents (all recovered; NIGHT_LOG for detail)

- Typed-timestamp drift in the log (lesson 6, third payment) — corrected in place;
  all entries after the correction line use date -u. Sequence/content unaffected.
- transfer-32 fired before its golds existed (gate on GPU availability, not data
  availability) — 0 rows lost; refired with correct gates.
- The "CPU" mixture battery launched unpinned → JAX preallocated 30.7 GiB of GPU 0 →
  deployment + transferL crashed to 0 rows; requeued (deployment score now chunked).
  **NEW LESSON: intended-CPU jobs get explicit CUDA_VISIBLE_DEVICES=""** — an
  unpinned JAX process is a GPU squatter with default prealloc.
- Two dict-collision crashes in the battery runner (kernel; w) — caught at first
  row, fixed, no loss. One pgrep-liveness self-match hang (lesson 1, mild, no loss).
- XLA constant-folding warnings on the nonlinear guidance transpose — benign.

## 5. Utilization & pace

15-min instantaneous samples (queue/util3.log, 23:20–02:05): GPU0 mean 50%, GPU1
48%, GPU2 17% — understated by sampling between bursts, but the honest headline is
different: **the night was completion-bound, not compute-bound.** Estimates ran
5–10× conservative again (lesson 9): trainings 25 min not 60; 64² NUTS gold 74 s;
the full confirmatory core fit in H0.6, the whole program including every ladder
rung in H3. ~7 h of the window returned unused after the ladder exhausted — closure
justified under the expected-information gate (logged [STEER] 02:1x).

## 6. Next actions (for the joint session)

1. Score P-20260704i–n together (table above PROPOSED; the P-m ordering sub-clause
   and P-k's dps_em03-vs-twisted_em03 refinement deserve the discussion).
2. Paper skeleton: three acts + transfer chapter; the negative-result paper's
   positioning (companion to score-KSD, lead with 100× path/endpoint) now extends
   with tonight's endpoint-certificate audit — decide whether one paper or two.
3. Upstream bug reports (tarp norm=True, mira) still owed from the arms night.
4. Blog/explainer extension (Part 9: the certifier trial) if the public-artifact
   route is taken; all five figures are draft-quality for it.
5. Idea ledger: "certify the certifiers" line is now COMPLETE on both routes
   (path-space: dead by weight degeneracy; endpoint-space: blind + fragile);
   the constructive survivor is gold-standard manufacture + sample-space tests.
