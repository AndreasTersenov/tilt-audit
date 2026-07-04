# PLAN — Certificate kill test (weight degeneracy at scale)

> Status: SIGNED OFF by Andreas 2026-07-04 (predictions adopted as proposed). Predictions P-20260704a–c (RESEARCH_LOG.md, brainstorm
> exit 2026-07-04) are adopted-as-proposed on sign-off unless amended there first.
> GPUs 1+2 (owner allocation); GPU 0 untouched. Expected wall: implementation-bound
> (~2–3 h code+gates), compute ~1 h across both GPUs.

## 0. The question this answers, and what changes per outcome

Does the residual path-space certificate of a guided diffusion sampler stay
*informative* as dimension grows — or do the weights degenerate into a vacuous
instrument? If informative (even just directionally) at d=4096: the steering-certificates
flagship proceeds (theory → wrapper → bench paper). If vacuous at d≥1024 and not rescued
by per-mode Rao-Blackwellization: flagship dies as scoped; fall back per the logged kill
criterion. There is no third outcome that leaves the decision unchanged — the run earns
its GPUs.

## 1. The object being computed

For samplers with per-step Gaussian kernels (no resampling), along the sampler's OWN
trajectories, per particle:

    log w  =  Σ_t [ log N(x_{t'}; m_ref(t), v_ref(t)) − log N(x_{t'}; m_s(t), v_s(t)) ]
              + r(x_0)/β

where (m_ref, v_ref) is the exact backward prior kernel (samplers.backward_kernel) and
(m_s, v_s) the sampler's actual kernel (for dps / exact_guidance: the exponential-
integrator guided kernel — mean AND variance differ from ref; both closed-form per mode).
Initial distributions coincide (N(0, V_tf)) — no boundary term. The FK target
ℚ* ∝ ℙ_ref · e^{r(x_0)/β} has the tilted posterior as its x_0 marginal exactly, so:

- log Ẑ = logsumexp(log w) − log N   → checked against tilt.log_z_analytic (unbiasedness).
- ESS_res = 1/Σ w̄²                   → repair affordability.
- KL̂ = log Ẑ − mean(log w)           → consistent estimate of path KL(ℙ_s‖ℚ*), which
                                        upper-bounds endpoint KL(π_s‖π) (data processing).
- k̂ (Pareto tail index of w)         → the "is ESS lying" diagnostic (PSIS; use arviz).
- Repair = metrics.evaluate with logw=residual weights (the existing weighted-moment
  machinery IS the repaired estimator; W2_repaired vs W2_raw is the repair column).

Note the certificate deliberately includes discretization error (it measures the chain's
distance to the exact FK target) — exact_guidance at T=64 ≈ floor, so its certificate
must read small (gate G-C2).

**Scope pins.** Samplers WITH resampling (sap, twisted) are excluded from the new code:
twisted already carries its own machine-zero incremental weights (used as the "exact
certificate" anchor, zero new code); sap's path measure needs genealogy tracking — out of
scope for the kill test. Misspecified-score rows compute the certificate w.r.t. the
CONTAMINATED model (what a practitioner could do): the certificate then reads steering
error only, while true error includes model pass-through — quantifying the known scope
limit ("certified steering ≠ correct posterior") with data, pre-empting the obvious
attack. remy gets an AIS-style certificate (level-density increments, 5 lines in the
sampler) as an EXPLORATORY column only — ULA leaves each level target inexactly invariant,
so its AIS weights carry a documented O(eps0) bias; bench quantifies it.

## 2. Implementation (surgical; frozen harness untouched except one exploratory hook)

1. `tilt_audit/certificate.py` (~120 lines):
   - `run_pointwise_cert(mode, key, Pz, az, y, b, N, T, tf)` — mirrors
     samplers._run_pointwise's guided step (same coefficients, same keys structure) and
     accumulates per-step log-RN vs backward_kernel + terminal r/β. Returns
     {z, logw_res, log_z_est}. ~30 lines duplicated from samplers.py rather than
     refactoring the gate-frozen module; documented cross-reference both sides.
   - `certify(out)` → {ess_res, khat, kl_path_hat, log_z_est}.
   - khat via arviz.psislw (wheel reuse; CPU-only deps — verify JAX GPU alive post-install,
     tonight's paid lesson).
2. `scripts/run_cert.py` — grid runner (pattern of run_t1.py): per row emit BOTH raw and
   repaired metrics.evaluate outputs + certificate fields + true KL/W2 + log Z analytic.
   Append-mode JSONL → results/cert_killtest.jsonl.
3. `scripts/gate_cert.py` — three gates, green before the grid:
   - **G-C1 (plumbing, machine-zero):** for mode='unguided' the accumulated step-RN ≡ 0
     and log w ≡ r(x_0)/β (terminal_is identity), atol 1e-8.
   - **G-C2 (bound sanity):** exact_guidance 16²/1σ/N=256: repaired W2 within 1.5× floor;
     KL̂ ≥ true endpoint KL for ≥90% of 8 seeds (bound direction).
   - **G-C3 (unbiasedness):** dps 16²/0.5σ, 32 seeds: mean(Ẑ/Z) within 5 s.e. of 1.
4. `scripts/cert_digest.py` — verdict plots + kill-criterion evaluation printed:
   (i) KL̂ vs true KL scatter, rank-corr per dim (P-c resolution);
   (ii) ESS_res and k̂ vs dim × damage (P-a resolution);
   (iii) W2 raw vs repaired vs N (P-b resolution);
   (iv) Ẑ/Z calibration histogram.

## 3. Grid (T=64 pinned, y=Y_KEY=999 unless stated; 8 particle-seeds everywhere)

**GPU 1 — the core (confirmatory for P-20260704a–c):**
- dims {16, 32, 64} × shifts {0.5, 1, 2, 4} × samplers {dps, exact_guidance} × N=256.
- terminal_is anchor: same dims × shifts, N=256 (certificate = its known weights; the
  worst-proposal baseline).
- N-ladder at 64² × shifts {1, 4}: N ∈ {64, 1024, 4096, 16384} × {dps, exact_guidance}
  ("what N buys an informative certificate at this d" — the affordability curve).

**GPU 2 — scale, robustness, scope columns:**
- 128² (d=16384) × shifts {0.5, 1, 2, 4} × {dps, exact_guidance} × N ∈ {256, 4096}.
- Misspec scope column: dps @ ε=±0.3, 64²/1σ, N=256 (certificate vs true-error split —
  the attribution demo).
- Multi-y robustness: y-seeds {1000..1003}, 64²/1σ, dps, N=256.
- Exploratory: remy-AIS @ K ∈ {5, 30}, 64² × shifts {1, 4}, N=256.

Both GPUs: XLA_PYTHON_CLIENT_PREALLOCATE=false, MEM_FRACTION=0.95 (2-GPU allocation rule),
fp64 (default). Row throughput from tonight: ~0.1–2 s post-JIT; N=16384 at 64² and
N=4096 at 128² are the heavy cells (~GBs, fine at 0.95×40GB).

## 4. Decision mapping (pre-registered readouts)

- **P-20260704a** resolves on plot (ii): ESS_res at 64²/N=256 vs monotonicity of KL̂ in
  true damage across the β ladder.
- **P-20260704b** resolves on plot (iii) at 16²: W2_repaired ≤ W2_raw/2 at 1σ.
- **P-20260704c** resolves on plot (i): rank-corr(KL̂, true KL) ≥ 0.9 pooled over the
  exact-score grid.
- **Kill criterion** (logged): at d≥1024, KL̂ non-monotone in true damage AND k̂>1
  everywhere AND not recovered by per-mode Rao-Blackwellization → flagship dies as
  scoped. (Rao-Blackwellized variant — per-mode analytic marginalization of the weights —
  is a follow-up implementation IF the raw version fails; not built preemptively.)

## 5. Risks / honesty notes

- KL̂ at N=256 under heavy degeneracy underestimates path-KL (mean(log w) is fine, log Ẑ
  biased low ⇒ KL̂ biased LOW). The N-ladder measures this bias directly (KL̂ vs N).
  Directional monotonicity — the P-a clause — is the robust readout, by design.
- The certificate certifies the steering step GIVEN the model; the misspec column
  displays exactly that limit. This goes in every figure caption from day one.
- arviz install = the only new dependency; installed CPU-side, GPU stack re-verified after.

## 6. Sequence

1. Sign-off on this plan + prediction stamp (adopt/amend P-20260704a–c).
2. certificate.py + gate_cert.py → gates green (CPU + brief GPU-1 smoke).
3. Launch GPU-1 core grid; implement remy-AIS hook + launch GPU-2 columns.
4. cert_digest.py while grids run; verdict table vs P-20260704a–c; log + push.

## 7. Amendments (2026-07-04, during implementation — logged before the grid ran)

- **Finding at gate time (G-C1 green, code verified):** exact chain-law computation shows
  the path-KL bound is TIGHT for good samplers (exact_guidance 16²/1σ: path 9.8 vs
  endpoint 9.3 nats — it correctly prices the discretization bias) but LOOSE ~95× for bad
  ones (dps: path 2701 vs endpoint 28.5): DPS's path deviates far more than its endpoint.
  Consequences: Ẑ recovery and importance repair are unaffordable for badly-wrong samplers
  at ANY feasible N (weights e^-thousands); P-20260704b likely resolves MISS (frozen, will
  be scored as measured). Instrument profile to test: "tight when green, loud when red".
- **Gate respec (criteria were mis-calibrated to the object, not the code):** G-C2 keeps
  the DPI check (all shifts, both samplers, closed form) and replaces repair-on-eg with
  the tightness anchor (eg path/end ≤ 1.3 at 1σ) ; estimator-fidelity and Ẑ-unbiasedness
  (G-C3) re-anchored at a healthy-weights regime (exact_guidance at 0.25σ, exact path-KL
  ≤~3 nats) — the only regime where those criteria are achievable by any estimator.
- **Addition at zero grid cost:** exact phase diagram (path-KL, endpoint-KL, tightness
  ratio per sampler × dim × shift) from chain_law alone — no sampling. Mode-wise weight
  accumulation flag added (per-mode certificates = the Rao-Blackwell rescue; exact here
  because modes are independent in the bench) — emitted for 64² rows so the kill
  criterion's rescue clause resolves in the same run.
