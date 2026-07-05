# OVERNIGHT PLAN — 2026-07-04 → 07-05 · The certifier trial + transfer night

> **Status: SIGNED OFF** by Andreas 2026-07-04 (interview on record: window ~10 h; Track A
> wins any compute contention; wake for nothing; 2 lognormal nets). Predictions
> P-20260704i–n are FROZEN in RESEARCH_LOG.md at launch — never edited, scored with the
> owner. **Window:** ~10 h, GPUs 0,1,2 (titan A100-40GB, no scheduler; GPU 3 = other
> users, untouched). **Executor:** a fresh session. Self-contained; context chain §9;
> live steering §7. Night log: **NIGHT_LOG_2026-07-05.md**.

## 0. Mission and non-negotiables

One sentence: put the field's one surviving runtime certificate (score-KSD,
arXiv:2602.04189) on trial against exact and manufactured truth — including the two
attacks it has never faced (missed modes, wrong reference score) — while upgrading the
bench with MCMC gold standards on a NON-GAUSSIAN substrate (nonlinear forward model,
the mass-mapping-realistic case) and measuring which of our Gaussian-bench findings
transfer.

Non-negotiables:
- **PUBLIC push BEFORE the first GPU job** (repo already public: push this plan + the
  frozen predictions; push again at every checkpoint and at dawn).
- **Gates before burn (§4).** The existing suites green on current code
  (`pytest tests/test_gates.py`, `scripts/gate_cert.py`) plus the new gates T-K1 (KSD
  null), T-M1 (mixture oracle exactness), T-L1/2/3 (gold-standard validity). No large
  GPU run before the relevant gate is green. Per-arm gating as in the arms night: each
  arm's new-code gate blocks THAT arm only.
- **Frozen predictions:** P-20260704i–n and the confirmatory grids in §2–3. Adaptation
  adds exploratory jobs (tagged, one-line expected-yield note each); never edits these.
- **ALL the compute:** ~30 GPU·hours. The queue must never idle a GPU while §7's filler
  ladder has rungs — and the ladder is designed so every rung tightens a confirmatory
  number (no manufactured work). Utilization sampled every 15 min to `queue/util3.log`;
  <20% on any GPU for 15 min with rungs pending = investigate and fill.
- **Crash-safety:** append-mode JSONL everywhere; trainings checkpoint every 30 min;
  NUTS chains save incrementally per chain.
- **Etiquette:** GPUs 0,1,2 only; `XLA_PYTHON_CLIENT_PREALLOCATE=false`; 3-GPU nights cap
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75` per process (may raise to 0.95 when a GPU runs a
  single job); fp32 trainings (`TILT_AUDIT_X64=0`), fp64 samplers/metrics/NUTS; ≤50 CPU
  workers for batteries.
- **Process discipline — the FULL paid-lessons list (two nights of tuition; every line
  was bought):**
  1. Kill by numeric PID only. NEVER pattern-match process names — not for kills, not
     for liveness waits (two self-match deadlocks so far; launcher gates use PIDs
     resolved BEFORE constructing the command).
  2. After killing any parent, `ps` the tree; kill surviving children by PID.
  3. Background every long launch (`nohup ... &`); the harness kills foreground
     commands at 2 min (one A3 run died this way).
  4. `pip/uv install` can silently replace the CUDA stack (torch pulled cudnn-9.24 and
     killed all JAX GPU). After ANY install: run the 3-line GPU matmul check before
     proceeding. Install torch-dependent tools CPU-only (`--index-url .../cpu`) or
     `--no-deps`.
  5. Any NEW results subdirectory gets a `.gitignore` line at creation (a 1 GB commit
     hung the public push once). Big binaries never enter git.
  6. Append-mode output files only; timestamps via `$(date -u +%H:%M)`, never typed.
  7. Training scripts write intermediate checkpoints to the SAME path — "file exists"
     is the WRONG readiness gate; gate on the training PID exiting.
  8. Every result-triggered heavy job gets a two-sentence expected-yield note in the
     night log BEFORE launch (owner's expected-information gate).
  9. Estimates from the last two nights ran ~10× conservative for sampler grids
     (JIT reuse → ~1 s/row) and ~2–3× conservative for solo trainings. Re-estimate from
     the first measured rows, then refill the schedule — do not coast.
  10. jax.jit + `float()` on traced values = crash; keep host-side conversions outside
      jitted functions (cost one remy-AIS run).

## 1. GPU map & standing state

| GPU | H0–0.5 | H0.5–~H5 | ~H5–H8 | H8–H10 |
|---|---|---|---|---|
| 0 | gates, installs, pushes | mixture+archive generation → 2 lognormal trainings (sequential) | Track A learned/wrong-score columns; gold-standard support | fillers |
| 1 | gates | NUTS gold standards (64² configs) | Track B transfer grids vs gold | fillers |
| 2 | smoke tests | NUTS gold standards (32² + more 64²) | Track B transfer grids + learned columns | fillers |
| CPU | plan copy, code hunt | KSD/PQMass/TARP battery (≤50 workers) from ~H1 | battery round 2 (lognormal + mixture columns) | dawn assembly |

Standing state at H0: nothing of ours is running (verify `nvidia-smi` + `ps`); GPU 3
belongs to other users. The repo is public with all prior results pushed. Existing
assets to REUSE, not rebuild: `results/archives/` (10 GRF failure banks, 4096 samples
each + 6 conditional sets), `results/archives16k/`, checkpoints `s_clean.pkl`,
`s_mis_m03.pkl`, `s_mis_p03.pkl` (64² GRF score nets), the whole `tilt_audit/` harness,
`scripts/run_diagnostics.py` (battery pattern), `scripts/gate_cert.py` (green),
`certificate.py` (chain laws; the cert instruments are NOT tonight's subject — dead line,
see RESEARCH_LOG E-20260704c).

## 2. Track A (confirmatory; WINS any contention) — the score-KSD trial

**Wheel reuse (mandatory, ~30 min time-box):** hunt the reference implementation of
arXiv:2602.04189 ("Beyond Accuracy: Evaluating Posterior Fidelity of Diffusion Inverse
Solvers" — Qiu, Yang, Liu, Wang, Shen; Feb 2026): paper PDF code link, GitHub search,
PyPI. If found: use it, wrap it, follow THEIR estimator/normalization exactly (they are
the audit subject). If not found in 30 min: implement KSD ourselves — U-statistic
estimator, two kernels pre-registered (IMQ with c=1, β=−1/2; RBF with median heuristic),
following the standard formulas (Liu-Lee-Jordan / Chwialkowski et al.; look at existing
implementations e.g. the `kgof` package for conventions) — and say so in every output.
Either way the null gate decides admission.

**The target score (what KSD consumes):** ∇log π at the sample points.
- True-score mode (the control): posterior score = prior score + likelihood score —
  closed form on ALL tonight's substrates (GRF per mode; mixture via responsibilities;
  lognormal-in-g via chain rule). Zero nets needed.
- Deployment mode (the trial): the score a practitioner actually has — the learned
  net's score (evaluate at small t, or per the paper's own construction if they specify
  one — READ the paper's method section first) + likelihood gradient. Wrong-reference
  mode: same but with the contaminated prior score (analytic ε=±0.3, plus the mis-trained
  nets when they land).

**Confirmatory grids (CPU battery unless noted; append → results/ksd_trial.jsonl):**
- A-null: T-K1 gate then 100-rep null curves (oracle samples vs true score) at budgets
  {256, 1024}, both kernels.
- A-power: existing GRF archives {dps, sap, twisted, dps_em03, dps_ep03, twisted_em03,
  dps_s05, dps_s2, oracle_null} × budgets {64, 256, 1024, 4096} × 20 reps × 2 kernels —
  detection at empirically calibrated α=0.05 (rank vs 60 oracle nulls; one-sided unless
  the statistic is two-sided by construction — follow the paper). The compensation row
  (dps_em03) is the money cell.
- A-mixture (missed modes; GPU 0 generation first): minimal 2-component mixture oracle —
  components N(±Δμ, C) with the SAME GRF covariance, mean offset in the lowest few
  modes; posterior = exact 2-component mixture with evidence-ratio weights (closed
  form). New module `tilt_audit/mixture.py` + gate T-M1 (oracle sampling matches exact
  component weights and moments; 20k draws). Failure configs: single-component sampler
  at true weights {50/50, 80/20} (the missed-mode archetype), + weight-swapped sampler.
  Question: does score-KSD (true score, which KNOWS both modes) flag a sample set that
  sits entirely in one mode? Theory says score-based tests can be blind to missing
  well-separated components — measure it. PQMass/TARP run the same configs (they should
  catch it — the contrast IS the result).
- A-wrongref (the KSD compensation trap): KSD with contaminated reference score
  (analytic ε=−0.3) evaluated on the ε=−0.3-matched sampler archives vs on clean-sampler
  archives — does a matched-wrong pair read as null while true damage is ≥3× floor?
  Then repeat with the mis-trained net's score when trainings land (the full deployment
  configuration).
- A-lognormal (after T-L gates): KSD true-score readings on Track B's sampler outputs
  vs their MCMC-truth damage — does its power survive a skewed target?

## 3. Track B — gold standards + the nonlinear-forward-model transfer

**Substrate (the construction that keeps every exact tool):** stay in g-space. Prior:
our standard GRF g ~ N(0, C) — the diffusion prior and all its closed forms are
UNCHANGED. Nonlinearity enters ONLY through the observation:
y = A·κ(g) + n, with κ(g) = exp(λg − λ²σ_g²/2) − 1 (shifted lognormal; λ = nonlinearity
knob; λ→0 recovers y = λ·A·g + n, the exactly-solvable linear case — that limit is gate
T-L1). This is exactly how nonlinearity appears in practice (nonlinear forward model /
reduction), the posterior in g is non-Gaussian, and DPS/Rémy handle it the way
practitioners do (likelihood gradients through the nonlinear map via the chain rule).
Default λ: calibrate so that skewness of κ is ~1 (report it); λ-ladder is a filler rung.

**Gold standards (GPUs 1+2, the night's backbone):** NUTS on the g-posterior
(differentiable log-density = GRF prior + Gaussian likelihood of the nonlinear map).
Wheel reuse: `numpyro` NUTS first choice (install into .venv via uv; GPU-verify after —
lesson 4), `blackjax` fallback. Configs: 64² × tilt {mid, strong} × y-draws {0..3} and
32² × the same (cheap column), 4 chains × 1k warmup + 4k draws each, fp64;
thinned draws saved to `results/gold/` (**gitignore at creation** — lesson 5).
Gates:
- **T-L1 (Gaussian-limit, the crux):** at λ=0.01 the NUTS posterior must match the
  closed-form Wiener answer: per-mode mean z-scores <4, variance ratios within 5%,
  on 32² and 64².
- **T-L2 (convergence):** R-hat < 1.01 every coordinate, min bulk-ESS > 400 per config.
- **T-L3 (independence check):** two independently-seeded gold runs of one 64² config
  agree (per-mode z-tests + MMD below null threshold).
If T-L1/2 cannot pass at 64² within ~H4 (tuning time-boxed): drop to 32²-only gold
standards — the night stays scoreable (P-l scored MISS honestly; transfer grids run at
32²; freed GPU time → Track A ladders, per priority).

**Trainings (GPU 0, sequential, ~2 h total):** two 64² score nets on the κ-FIELD prior
(draw g, map to κ, train on κ maps — these are the nets a practitioner would own):
`ln_clean` and `ln_mis` (trained on contaminated spectrum, ε=−0.3 analog). fp32, 60k
steps, 30-min ckpts. They feed Track A's deployment columns and the learned-DPS
transfer rows. (Wheel: adapt `scripts/train_score.py` minimally — new sample-map line.)

**Confirmatory transfer grids (GPUs 1+2 after gold standards):** samplers in g-space
with exact prior score + nonlinear likelihood guidance: dps, dps-inflated (the
exact-guidance analog: denominator b + a²·Var-proxy — pin the proxy in code comments),
remy K ∈ {5, 30, 100}, terminal_is. N=256, T=64, seeds {0..4}, at every gold-standard
config. Twisted is EXCLUDED (no closed twist off-Gaussian — note in the doc, do not
improvise one at 2 a.m.). Metrics vs gold draws (posterior is non-Gaussian; the Gaussian
closed forms do NOT apply — new `tilt_audit/metrics_gold.py`): MMD (RBF, median
heuristic on gold), sliced-W2 (100 fixed random directions, seeded), per-mode mean/var
z-scores vs gold, 68%-coverage of the 3 band-power functionals vs gold quantiles.
Floor reference: a disjoint thinned gold subsample of matched size N — the "oracle"
row. Learned-score columns (ln_clean / ln_mis nets driving DPS) after trainings land.
→ results/transfer.jsonl.

## 4. Gate summary (all logged [GATE] with numbers)

1. Existing suites: `pytest tests/test_gates.py` (15) + `scripts/gate_cert.py` (G-C1..3)
   green on current code.
2. T-K1: KSD null — oracle-vs-true-score readings calibrated (empirical null well-formed,
   no drift with budget), both kernels. A kernel that fails its null is dropped (logged).
3. T-M1: mixture oracle exactness (weights + moments at 20k draws).
4. T-L1/2/3 as in §3.
5. Smoke rule: every new runner does a 16²/small-budget smoke row before its grid.

## 5. Timeline (estimates; re-estimate at first measured rows — lesson 9)

| Hours | GPU 0 | GPU 1 | GPU 2 | CPU |
|---|---|---|---|---|
| 0–0.5 | gates, installs+verify, push | gate smoke | gate smoke | code hunt (KSD impl) |
| 0.5–2.5 | mixture gen + archives; trainings start | NUTS: T-L1 λ→0 gates, then 64² golds | NUTS: 32² golds + 64² | T-K1 + A-null + A-power |
| 2.5–5 | trainings finish; A-wrongref (analytic) | 64² golds | 64² golds | A-mixture battery; checkpoint #1 read |
| 5–8 | A deployment columns (nets) | transfer grids vs gold | transfer grids | A-lognormal battery; checkpoint #2 |
| 8–10 | fillers | fillers | fillers | dawn assembly |

## 6. Failure playbook

- KSD reference code not found in 30 min → implement per §2 spec; T-K1 decides.
- KSD null fails for a kernel → drop that kernel; if BOTH fail, suspect wrapping (the
  TARP lesson: check normalization/preprocessing asymmetries) before concluding.
- NUTS diverges / T-L1 fails at 64² → mass-matrix adaptation + smaller step; time-box
  to H4; then 32² fallback (scoreable night preserved).
- numpyro install breaks JAX GPU → lesson 4 procedure; blackjax fallback.
- Training NaN → halve LR once; then smaller net; 30-min box.
- OOM → halve batch/N; never two trainings + a grid on one GPU.
- GitHub unreachable → log, continue local, retry at checkpoints.
- Another user's PIDs on 0–2 → downshift immediately.
- A track dies irrecoverably → ALL GPUs to the other track's ladder (owner priority:
  A > B). A partial night must still be scoreable.

## 7. Live adaptation protocol

**Firewall:** may reallocate, reorder, add exploratory jobs (tagged, with expected-yield
notes); may NOT edit P-20260704i–n or §2–3 grids. Confirmatory/exploratory split stays
clean in the JSONLs.

**Checkpoint read #1 (~H2.5) — after A-power + first golds:**
- KSD detecting everything incl. compensation (P-i shaped): proceed; the trial's teeth
  move to mixture + wrongref columns.
- KSD null broken out of the box (the TARP scenario): diagnose preprocessing FIRST; if
  it is a genuine tool bug, that is a headline finding — verify with the care of the
  TARP/MIRA episodes (symmetric-wrapping checks), then continue with the repaired
  version alongside the published one.
- T-L1 failing: reallocate one gold GPU to Track A ladders while tuning continues on
  the other (priority rule).
**Checkpoint read #2 (~H5) — after mixture + wrongref + first transfer rows:**
- P-j HIT (mode-blindness confirmed): zoom — weight-knob ladder {50/50→99/1}, budget
  ladder to 16k, and the PQMass/TARP contrast rows; this is the paper's second act.
- P-j MISS (KSD catches missed modes): equally important — measure its power curve on
  the weight knob; the "endpoint route survives" story strengthens.
- P-k HIT (wrong-reference false-certification): add the ε-ladder (±0.1, ±0.2) — where
  does false-certification switch on?
- Transfer rows P-m-shaped: fold into note; if orderings BREAK on lognormal, that is a
  finding — densify tilts and λ-ladder before dawn.
**Cadence:** scripted digest of fresh JSONLs every ~45 min; one [STEER] entry per read;
public push at each checkpoint. **Zoom rules (pre-noted):** any diagnostic with
non-monotone power vs budget; any NUTS config whose two seeds disagree post-T-L3;
any transfer ratio >3× its Gaussian analog.

**Filler ladder (unbounded, value order; every rung tightens a confirmatory number):**
1. More gold y-draws (toward 8 per config) — halves MC error on every transfer number.
2. Gold chain extensions (+4k draws) on the configs feeding the tightest claims.
3. Mixture weight-knob + budget ladders (P-j resolution sharpens).
4. Wrong-reference ε-ladder (P-k zero-crossing).
5. λ-ladder (nonlinearity dial: skewness {0.5, 1, 2}) — transfer-vs-non-Gaussianity curve.
6. 128² gold-standard stretch (one config; only if 64² golds finished early).
7. Third net (ln_mis ε=+0.3) for the wrongref sign axis.
8. Seed densification of anything above (never exhausts).

## 8. Deliverables at dawn

1. `results/ksd_trial.jsonl`, `results/transfer.jsonl`, `results/gold/` library,
   mixture module + archives; figures: (i) KSD power curves with compensation row
   highlighted vs PQMass/TARP; (ii) the mixture missed-mode confession plot (KSD vs
   PQMass/TARP vs weight knob); (iii) wrong-reference false-certification table/curve;
   (iv) transfer panel: sampler damage on lognormal vs Gaussian at matched tilts;
   (v) Rémy K-curve on the nonlinear substrate.
2. `NIGHT_LOG_2026-07-05.md` complete ([JOB]/[GATE]/[RESULT]/[STEER]/[FAIL]/[NOTE],
   written as things happen); utilization report from `queue/util3.log`.
3. `HANDOFF_DAWN_3.md`: P-20260704i–n verdict table (PROPOSED + one-line evidence —
   scoring WITH the owner), per-track results, incidents, next-actions.
4. RESEARCH_LOG: E-entry Results + Updated-beliefs drafted (Outcomes/Lessons reserved).
5. Public repo pushed through the final commit.

## 9. Context chain for the fresh session (read in order)

1. This file, IN FULL.
2. `RESEARCH_LOG.md` — P-20260704i–n (frozen) + the certificate-arc entries
   (P-20260704a–h, E-20260704a–c) + the literature-sweep NOTE (why this night exists).
3. `HANDOFF_DAWN_2.md` §4 (incidents) and `NIGHT_LOG_2026-07-04.md` [FAIL]/[STEER]
   lines — the paid-lessons provenance.
4. Code: `tilt_audit/` package (fields/diffusion/tilt/samplers/metrics/misspec/
   certificate), `scripts/run_t1.py`, `scripts/make_archives.py`,
   `scripts/run_diagnostics.py`, `scripts/gate_cert.py` — the patterns to extend.
5. `docs/explainer/certificate_explainer.html` — the plain-language framing (fast skim;
   the night's results extend its Part 6–8 story).
6. Skills: cluster-resources, coding-guidelines, research-principles §§5–7.

## 10. First prompt for the executing session (copy-paste)

```
Execute docs/OVERNIGHT_2026-07-04_CERTIFIER_TRANSFER_NIGHT.md in ~/software/tilt-audit.
Read it IN FULL first, then its §9 context chain in order.

Ground rules, non-negotiable:
- Predictions P-20260704i–n in RESEARCH_LOG.md are FROZEN; the §2–3 grids are the
  confirmatory core. Never edit them; result-triggered additions are tagged exploratory
  with a one-line expected-yield note each.
- Gates first (§4): existing suites green, then T-K1/T-M1 before Track A's batteries and
  T-L1/2/3 before Track B's grids. Push the plan+predictions publicly BEFORE the first
  GPU job; push again at every checkpoint and at dawn.
- GPUs 0,1,2 are mine for ~10 h, titan, no scheduler. Track A WINS any compute
  contention. Never idle a GPU while §7's filler ladder has rungs — and never
  manufacture work outside the ladder. ≤50 CPU workers. JAX no-prealloc; fp32
  trainings; fp64 samplers/metrics/NUTS.
- Don't reinvent wheels: hunt the score-KSD reference code (arXiv:2602.04189) first
  (30-min box; fallback spec in §2); numpyro NUTS (blackjax fallback); adapt
  train_score.py and run_diagnostics.py rather than rewriting; reuse ALL existing
  archives and nets.
- Process discipline: §0's ten paid lessons apply verbatim — PID-only kills, no pgrep
  patterns anywhere, background every launch, GPU-verify after ANY install, gitignore
  new results dirs at creation, append-mode, shell timestamps, PID-exit (not
  file-exists) readiness gates, re-estimate from first measured rows, expected-yield
  notes before heavy result-triggered jobs.
- Keep NIGHT_LOG_2026-07-05.md as you go ([JOB]/[GATE]/[RESULT]/[STEER]/[FAIL]/[NOTE],
  written when things happen). Steer live per §7: checkpoint reads ~H2.5 and ~H5,
  45-min digest cadence, pre-noted zoom rules, filler ladder in value order.
- Failures: §6 playbook. Time-box every debug; a partial night must still be scoreable.

At dawn deliver: HANDOFF_DAWN_3.md with the P-20260704i–n verdict table (PROPOSED
scores + one-line evidence — final scoring happens with me), the five §8 figures, the
utilization report, drafted E-entry Results, and the public repo pushed through the
final commit. Wake me for nothing — adapt, log, and keep the GPUs full. Note the time
difference: "dawn" means 10 hours from launch; do not stop before that.
```
