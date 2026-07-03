# HANDOFF_DAWN — 2026-07-03 · GRF pilot overnight

> FINAL — assembled 05:30 UTC (H6.7); Track B streams run until ~06:30.
> Chronology in NIGHT_LOG.md. All confirmatory results final; T2 N=16 tail
> and B2 seeds 1/2 were still appending at assembly (counts below are as-of).

## 1. Verdict table — P-20260702d–h (SCORED with Andreas, 2026-07-03 morning: d HIT · e HIT depth-qualified · f HIT · g HIT mechanism-corrected · h MISS ⇒ GO)

| Prediction | Proposed | One-line evidence |
|---|---|---|
| **P-d** (75%) DPS off-target, gap grows with β | **HIT** | Exact-score T1: W2/floor 1.4×→28× rising monotonically in β at every d and N (N=256, 64²: 5.0×@0.5σ → 27.7×@4σ); γ*=1.33–1.43 (over-concentrated); mean-CI coverage under-covers (cov68 → 0 at strong tilt). Discretization control (exact_guidance) ≈ floor. |
| **P-e** (70%) SAP runs cold, γ*>1 | **HIT (depth-qualified)** | γ*(T) rises monotonically with step count — 32²/0.5σ/N=256: γ* = 0.21 / 0.52 / **1.72** at T = 32 / 64 / 256 — i.e. SAP runs cold at practitioner-typical depths, and the γ*(T) depth law transfers from the discrete substrate to ℝ^d. At strong tilts the pathology saturates into variance collapse without mean tracking (γ_mean→0, var_ratio_logmed≈−20, W2 4–143× floor) — over-concentration in an even more damaging form. T1's frozen grid (T=64) alone would have read as a partial miss; the T-sensitivity arm (exploratory) resolves it. |
| **P-f** (85%) twisted SMC on-target, valid Ẑ | **HIT** | Conjugate twisted SMC = 0.96–1.06× oracle floor in ALL 36 (d,β,N) cells, γ*=1.00; incremental weights machine-zero (<1e-9; tower-property self-test) so log Ẑ = log Z exactly; T-G3 prior-proposal variant: Ẑ unbiased (mean ratio 1±5se, d=1, real weight variance). |
| **P-g** (55%) misspec propagates differently per scheme | **HIT (mechanism corrected)** | Propagation differs dramatically per scheme (64², N=256, 1σ): conjugate twisted passes ε through 1:1 (0.50→2.57 W2 — with a wrong score it *exactly samples the wrong tilt*; weights stay conjugate-degenerate, so NO absorption, contra the prediction's mechanism); DPS is sign-dependent and non-additive (ε=+0.3 amplifies 3.77→5.97; ε=−0.3 *cancels* to 2.92 with γ*=0.98 — accidental compensation, the baryonic-feedback trap in miniature); SAP's collapse masks ε entirely. Learned-net axis: kernel choice dominates twisted degradation (pathway control 5.11 vs learned 5.21 vs exact 0.50); net score error adds ~1.3 W2 to DPS. ε-ladder densification (±0.05, ±0.2) queued for the cancellation curve. |
| **P-h** (25%) KILL: bias negligible at realistic settings | **MISS (= good for the program)** | Kill criterion decisively NOT triggered: DPS alone ≥3× floor from 0.5σ upward at N=256 (all d); max frozen-scheme ratio 251× (terminal-IS). Bias is d-extensive as the deep-read predicted (KL/mode ~O(1)). |

## 2. GO / NO-GO recommendation (plan §3)

**GO** — recommend committing to Direction 1 as the UVa flagship.
- "≥2 schemes with nonzero, cleanly-scaling bias in ≥2 knobs at realistic
  settings": DPS scales in β, d, and N-relative floor; SAP scales in β and T
  (depth); terminal-IS scales in d. All at 0.5–1σ tilts — realistic strengths.
- Three-way decomposition separability — CONFIRMED end-to-end: exact-score
  arms separate scheme bias from finite-N (oracle floor = d/2N in W2², d/N in
  KL — verified to d=16384) and discretization (exact_guidance ≈ floor;
  T-matched exact reference at T=256 in t1_exact_T256.jsonl); the pathway
  control separates kernel-choice from net score error; the analytic ε-arm +
  learned S-mis separate misspecification from both. This axis — the one the
  prior art lacks — worked as designed.
- The certificate story has teeth already: potential-only twisting (the naive
  ψ-weights variant) weight-degenerates at d≥256 while the conjugate twisted
  sampler is exact — ESS/Ẑ trajectories separate correct from degenerate
  samplers on this oracle.

## 3. Utilization report (scripts/dawn_report.py, 25 samples @ 15 min)

- **GPU 0**: mean util 71%, busy-share 76%, mean mem 19.5 GB. Idle windows:
  23:18–23:33 (implementation, pre-gates by design) and 01:48–02:33 (the T2
  launcher pgrep deadlock — the night's one real utilization loss, ~45 min).
- **GPU 1**: mean util 67%, busy-share 76%, mean mem 28.0 GB. Idle: 23:18–00:03
  (implementation + the GPU-1 server memory-settling incident), 04:18–05:18
  (B2-s1 PRM present but generation-bound; bursty scoring).
- **GPU 2**: mean util 84%, busy-share 96%, mean mem 33.7 GB — effectively
  saturated all night (one 23:48 dip during the server swap).
- Queue: 23 jobs done, 1 "failed" (the fp64 training ghost — real fp32 run
  succeeded outside the queue). GPU 3 untouched (other users).
- Track A row counts: t1_core 540 (frozen grid), controls 216, misspec+ε-ladder
  504, tsens+depth 576, weak 324, t3 seeds 1080, zoom 48, d128 216, T4.1 16,
  pathway 144, t2_learned 290+ (N=256/64 blocks complete for all three nets).

## 4. Track B — E-20260702a/b status

- **B1 (α-sweep) — E-a answerable.** Full sweep at seed 0 (100 problems/α),
  plus α=0.05 at seeds 1,2. AUROC: 0.500 / 0.673 / 0.645 / 0.729 / 0.725 /
  0.763 for α = 0.05 / 0.10 / 0.15 / 0.25 / 0.35 / 0.50; selected-accuracy
  0.68–0.73 flat through α=0.35, 0.64 at α=0.50. **Threshold expectation
  (AUROC ≥ 0.65 by α=0.25) HIT at 0.729.** Monotone rise holds modulo two
  mechanically-predicted artifacts: α=0.05 is pinned at 0.500 (n_iid=1 — a
  one-sample stratum has no ranking signal) and 0.10≈0.15 tie (both n_iid=2;
  effective α = round(16α)/16). Accuracy drop at α=0.50 (−0.05) is
  direction-consistent but ~1σ at one seed. *The operative variable is
  n_iid, not α — future sweeps should be designed in n_iid units.*
- **B2 (R1 twin) — E-b surprising, needs autopsy before scoring.** Final:
  seed 0 complete (100 problems), seed 1 near-complete (78), seed 2 LOST to a
  second bite of the open-"w" truncation bug at shutdown (a zombie driver-v3
  retry rewrote the file; a 19-row remnant plus the 05:30 metrics snapshot at
  n=51 survive — numbers below include the pooled data as of the last good
  read). Accuracy ordering is clean and expectation-consistent: SAP
  0.17–0.26 < defensive 0.26–0.32 < iid 0.39–0.47 — search damages R1
  accuracy even harder than Qwen's. But calibration INVERTED for every
  method: AUROC iid-majority 0.17, iid-weighted 0.21, defensive 0.36, SAP
  0.38 — all below chance, including the i.i.d. anchor that scored ≥0.7 on
  Qwen. Literal scoring would be "SAP≤0.6 HIT, iid≥0.7 MISS", but the joint
  all-inverted pattern (worst for iid!) points at a harness artifact:
  R1's very long traces + max-model-len 4096 → truncation-corrupted answer
  extraction plausibly correlates confidence with wrongness. Run the
  truncation-rate-by-correctness autopsy before proposing a score.

## 5. Exploratory results beyond the pre-registration (firewalled, all tagged)

- **T4.1 memoryless-schedule theorem (2409.08861) exactly quantified**: at
  tf=3, KL to true tilt: 0.09 (memoryless σ²=2) → 0.2–0.4 (σ²=1) → 14–87
  (σ²=0.5) → 5e3–4e4 (σ²=0.1). At tf=9 all σ²∈[1,2] sit inside the
  finite-horizon residual — the value-function bias is a short-horizon /
  low-noise phenomenon, catastrophic exactly in the σ→0 (FM/DDIM-style)
  fine-tuning limit. Files: results/t41_memoryless.jsonl.
- **Weak-tilt arm (0.125σ, 0.25σ)**: the 3×-floor crossing is N-dependent —
  at N=256 DPS NEVER passes (3.3–3.5× floor even at 0.125σ, every d); at
  N=64 the crossing sits at ~0.7σ. Even eighth-sigma tilts are auditable at
  modest particle counts: the strongest possible P-h evidence.
- **T-sensitivity / depth law (T∈{32..1024})**: DPS is T-invariant (pure
  continuum scheme bias — integrator validated); SAP's γ* rises monotonically
  with depth (0.21→0.52→1.72 for T=32→64→256 at 32²/0.5σ) — the
  particle-reasoners γ*(T) law transfers to ℝ^d.
- **d=128² arm (T4.2)**: DPS KL 29→200→1738→16372 nats for d=256→16384
  (254× the oracle floor at 128²; floor = d/N nats exactly). Extensivity
  holds; no high-dimensional rescue anywhere in reach.

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
  analytic-preconditioned exponential guidance step + per-particle
  displacement clip at 3× step noise, activation fraction recorded per row).
- T2 launcher deadlocked on its own pgrep pattern (heredoc text in the parent
  shell's cmdline) — 45 min of GPU-0 loss; two other self-matching
  pkill/pgrep incidents earlier. LESSON: never pattern-match process names
  with strings that appear in your own invocation; kill by PID.
- **Data loss (small, metrics preserved):** particle-reasoners'
  run_reliability opens --out in "w" mode — each invocation TRUNCATES the
  file. The α=0.05 seed-2 run overwrote the raw rows of seeds 0,1 (their
  AUROC/acc numbers survive in NIGHT_LOG snapshots: 0.500/0.710 at n=124).
  All other α files are single-invocation. The same bug bit HARDER at
  shutdown: a surviving driver-v3 subshell auto-retried the killed B2 seed-2
  run and its fresh invocation truncated that file (80 problems of raw rows →
  19-row remnant; n=51 metrics snapshot preserved). Fix run_reliability to
  append-or-unique-name BEFORE any future use, and never leave retry wrappers
  running past their driver.
- fp64→fp32 training restart cost ~35 min of GPU-0 time; three concurrent
  T2 grids OOM'd (VJP+metric peaks) → serialized to two lanes.

## 7. Next-session actions

- **Remy-method arm (designed 2026-07-03, owner-approved as next experiment; not yet
  launched).** Implement Remy et al. (2023) mass-mapping sampling exactly in the
  harness: guidance with the likelihood covariance inflated by the diffusion noise
  level (sigma_t^2 — an upper bound on Var[x0|xt], i.e. the CONSERVATIVE side of the
  guidance spectrum, vs DPS's deletion on the aggressive side) + K Langevin
  equilibration steps per noise level. Sweep K; measure W2/coverage vs the oracle and
  response to the misspecification knob (kappa-TNG prior <-> eps contamination; note
  the accidental-compensation trap may operate in REVERSE for a conservative scheme).
  Expected yield (per the expected-information gate): the exact error budget of the
  community's flagship mass-mapping method as a function of its compute budget —
  the program's first outward-facing result and the natural bridge artifact to
  CosmoStat. ~1 day of compute when green-lit.
- **Amortized-conditional arm (designed 2026-07-03, owner-approved for next steps; not
  launched).** Audit the amortized-posterior-sampler class (Legin+ 2304.03788 / SimBIG-style,
  and the class Doeser & Jasche 2606.10023 audited with HMC references): train a small
  CONDITIONAL score model on (field, observation) pairs from the Gaussian testbed and compare
  its samples to the exact Wiener posterior — amortization + score error measured with an
  exact reference at any d, zero steering confound (it is the amortized analog of the T2 arm;
  one new training script on the existing harness). Expected yield (gate): quantifies whether
  D&J's "summary checks pass, geometry wrong" failure reproduces against a closed-form
  reference, and prices the amortized class against the steered class on ONE substrate —
  the comparison neither literature currently has. ~half a day of compute when green-lit.
- Score P-d–h WITH Andreas (this table is a proposal).
- T2 decomposition figure + P-g verdict if trainings landed late.
- B2: rerun twin with a daytime budget or trimmed config; investigate R1
  step-segmentation under the 4096 cap before interpreting SAP numbers.
- Decide repo publication shape (explicitly deferred by the plan).
- E-20260702a/b/c Result+Updated-belief blocks: drafted at dawn in
  particle-reasoners/RESEARCH_LOG.md (do not edit predictions).
