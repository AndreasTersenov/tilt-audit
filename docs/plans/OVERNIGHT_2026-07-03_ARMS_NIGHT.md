# OVERNIGHT PLAN — 2026-07-03 → 07-04 · The four arms night

> **Status: SIGNED OFF** by Andreas 2026-07-03 (interview on record: predictions adopted as
> proposed; priority order multi-y > diagnostic-power > amortized > Rémy; repo goes PUBLIC;
> 100% tilt-audit, no LLM filler). **Window:** ~8 h, GPUs 0,1,2 (titan A100-40GB, no
> scheduler). **Executor:** a fresh session. Self-contained; context chain §9; live
> steering §7. Predictions P-20260703b–e are FROZEN in RESEARCH_LOG.md — never edited.

## 0. Mission and non-negotiables

One sentence: convert last night's GO'd pilot into the *complete* paper dataset — audit the
amortized and Rémy inference classes on the same oracle, de-caveat the headline with a
multi-y ensemble, and calibrate the community's own diagnostics against failures of exactly
known size — using ALL 24 GPU-hours (the filler ladder in §7 is unbounded and every rung
sharpens the same paper).

Non-negotiables:
- **PUBLIC push BEFORE the first GPU job.** Create the public GitHub repo
  (`gh repo create tilt-audit --public --source . --push`, owner's account; confirm remote
  URL in NIGHT_LOG). This publicly timestamps the frozen predictions — tonight's
  pre-registration becomes externally verifiable. Push again at every steering checkpoint
  and at dawn. If GitHub is unreachable, log it, fall back to local-only, continue.
- **Gates before burn (§3):** the existing 15-gate suite green on current code, plus the
  three new arm gates. No large GPU run before green.
- **Frozen predictions:** P-20260703b–e and the four arm grids as specced here are the
  confirmatory core. Adaptation adds exploratory jobs (tagged); it never edits these.
- **All the compute:** the queue must never idle a GPU while any §7 filler rung remains.
  Utilization logged every 15 min; <20% for 10 min with work pending = investigate.
- **Crash-safety:** every runner appends per-row JSONL (append-mode files only — the "w"
  truncation bug class is dead; keep it dead). Trainings checkpoint every 30 min.
- **Etiquette:** GPUs 0,1,2 only; JAX no-prealloc everywhere (`XLA_PYTHON_CLIENT_PREALLOCATE=false`),
  fp32 for trainings (`TILT_AUDIT_X64=0`), fp64 for samplers/metrics (default). ≤50 CPU workers
  for the diagnostic batteries. Check nvidia-smi before claiming; downshift if другие users appear.
- **Process discipline (paid lessons, 2026-07-03):** kill by PID only, never pattern-match
  (three self-match incidents); after killing any parent, `ps` the tree for surviving
  children; no vLLM tonight → the orphan-EngineCore class shouldn't exist, but check
  compute-apps at every checkpoint anyway; NIGHT_LOG timestamps via `$(date -u +%H:%M)`,
  never hand-typed; every result-triggered job gets a one-line expected-yield note
  (owner's expected-information gate — it's in CLAUDE.md now).

## 1. GPU map & standing state

| GPU | Assignment |
|---|---|
| 0 | A3 amortized trainings → A2 archive generation → fillers |
| 1 | A1 multi-y → A4 Rémy K-sweep → fillers |
| 2 | A2 archives → A3 audit runs → fillers (absorb leftover R1-seed-2 PRM if still up) |

Standing state to check at H0: the E-20260703a seed-2 rerun (particle-reasoners; generator
on GPU 1, PRM on GPU 2) may still be finishing — if alive, let it run (it self-terminates;
~≤1 h), schedule around it, and afterwards verify its EngineCore children actually exited
(kill leftover PIDs). Its completion note goes to the OLD log (NIGHT_LOG.md); tonight's log
is **NIGHT_LOG_2026-07-04.md** (fresh file, same tag grammar: [JOB]/[GATE]/[RESULT]/[STEER]/
[FAIL]/[NOTE], one entry per event, written as things happen).

## 2. The four arms (confirmatory core; priority order = drop order reversed)

### A1 — multi-y ensemble (referee-proofing; ~2 GPU-h; GPU 1 first)
- Extend `scripts/run_t1.py` with `--y-seeds` (draw y per seed via
  `make_observation(PRNGKey(yseed), ...)`; b recalibrated per y; record `y_seed`, b per row).
- Grid: y_seeds 1000–1011 (12 draws) × dims {16,32,64} × shifts {0.5,1,2,4} × N=256 ×
  particle-seeds {0,1} × samplers {oracle,dps,sap,twisted,terminal_is}; T=64 (pinned).
  → `results/a1_multiy.jsonl`.
- Analysis target: per-(dim,shift) distribution of W2/floor ratios across y.
- Filler extension (§7): more y-seeds, N=64 arm, exact_guidance column.

### A2 — diagnostic-power ("certify the certifiers"; GPU for archives, then ≤50 CPUs)
- **Wheel reuse (mandatory checks before writing anything):** PQMass — `pip install pqm`
  (Ciela, github.com/Ciela-Institute/PQM); TARP — `pip install tarp` (Lemos+). MIRA
  (2605.02014) — search GitHub/paper for the reference implementation first (Ciela orbit;
  likely `Ciela-Institute/MIRA` or linked in the PDF); only if none exists, implement from
  the paper's analytic expressions in a standalone `tilt_audit/diagnostics_mira.py` with the
  paper's own toy examples as unit tests. Wrap all three behind one thin interface
  (`scripts/run_diagnostics.py`).
- **Null calibration gate first (§3, T-N3):** every test run same-vs-same (two oracle
  archives) must return its nominal null behavior before any power claim.
- Failure archives (GPU; `results/archives/*.npz`, z-space samples + configs): at 64², 1σ,
  T=64: oracle ×2 (null + reference), dps, sap, twisted, dps@ε=−0.3 (the compensation
  config), dps@ε=+0.3, twisted@ε=−0.3; sizes: 4096 samples each (build by batching N=256
  runs × 16 particle-seeds; record per-batch seeds). Plus 2σ and 0.5σ dps for the
  strength axis.
- Power curves (CPU battery, joblib ≤50 workers): each diagnostic × each archive-vs-oracle
  pair × sample budgets {64, 256, 1024, 4096} × 20 bootstrap resamples
  → `results/a2_power.jsonl` (test, config, budget, score/p-value, detected@α=0.05).
- MIRA/TARP need conditional/joint structure — for these, generate matched
  (y, posterior-samples) sets across the multi-y draws from A1 (reuse!); document exactly
  what each test consumes in the runner header.
- Analysis target: detection probability vs budget per (test × failure); the money cell is
  the compensation config's row (P-20260703c).

### A3 — amortized-conditional arm (GPU 0; trainings ~2–3 h total)
- New `scripts/train_conditional.py`: extend the existing UNet (`tilt_audit/scorenet.py`)
  to conditional — input channels (x_t, y_map) (y fixed-noise map via unpack(y)/unpack per
  sample), same DSM loss, fp32, EMA, 30-min ckpts; train on pairs (draw z0~prior, y=az0+sε
  per sample — infinite paired data). Three runs: default (60k steps), half-capacity, and
  quarter-steps (15k) — the "failure onset" ladder.
- New `scripts/run_amortized.py`: sample the conditional reverse diffusion (plain ancestral,
  NO guidance — conditioning is in the net) for the T1 y (Y_KEY=999) AND 4 of A1's y-draws;
  N=256, T=256, 3 seeds; evaluate with the standard metrics vs (μ*,Σ*) per y
  → `results/a3_amortized.jsonl`. Also run the SUMMARY-level checks the community would run
  (mean/marginal/band-power agreement) so P-20260703d's "summaries pass / geometry fails"
  contrast is computed within our own rows.
- Gate T-N1 (§3) before the big grid.

### A4 — Rémy-method arm (GPU 1 after A1; implementation ≤2 h)
- **Wheel reuse:** consult the local repos first — `~/software/PnPMass`, `~/software/PnP`,
  and Rémy's public `CosmoStat/jax-lensing` (clone if net available) — for the exact
  annealing schedule, per-level step counts, and step-size convention used in Remy et al.
  (A&A 2023, arXiv:2201.05561). Do NOT port their code wholesale: implement the SCHEME in
  our harness (`samplers.py::remy` or a small module) so metrics/gates apply unchanged:
  per noise level t_i on our log grid, K Langevin steps targeting
  score_prior(x,t_i) + ∇ log N(y; a·x, Σ_n + σ_{t_i}² a²) — i.e. likelihood inflated by the
  DIFFUSION noise level (their choice), then cool.
- Gate T-N2 (§3): with the inflation term replaced by the exact Var[x0|xt] and K large, the
  sampler must converge to the oracle (reduction anchor); and at K→large with their σ²
  inflation the t→0 target is the true posterior → W2 must decrease monotonically in K.
- Grid: K ∈ {1, 2, 5, 10, 30, 100} × shifts {0.5,1,2,4} × 64² × N=256 × seeds {0,1,2}
  (+16² column for cheap K→∞ checks) → `results/a4_remy.jsonl`. Then the misspec pair:
  ε=±0.3 × K∈{5,30} × shifts {1,4} (P-20260703e's sign-flip test).
- Filler extension: K ladder densification, 128², ε ladder.

## 3. Gates (all green before any big GPU job; ~30 min)

- **T-G1..15 (existing suite):** `pytest tests/test_gates.py` on current code. Green required.
- **T-N1 (conditional net sanity):** after 2k warmup steps, the conditional net's posterior
  mean at low noise beats the PRIOR mean baseline on held-out pairs by ≥5× in MSE toward
  μ* (i.e., it is actually using y). Cheap, catches wiring bugs before 60k steps.
- **T-N2 (Rémy reduction anchor):** as specced in A4 — exact-inflation + K=100 lands within
  1.5× oracle floor at 16²/1σ; W2(K) monotone decreasing on the same config with their
  σ_t²-inflation.
- **T-N3 (diagnostic null calibration):** PQMass p-values ~U(0,1) on same-vs-same (KS test
  p>0.01 over 100 reps); TARP nominal coverage on oracle-vs-oracle; MIRA hits its analytic
  reference value ±2σ on a known-matched toy. A diagnostic that fails its null does not
  enter the power study (log and proceed with the others).

## 4. Queue mechanics

Reuse `scripts/queue_worker.py` (one worker per GPU, atomic lock claim, retry once,
[JOB]/[FAIL] lines) — point NIGHT_LOG writes at NIGHT_LOG_2026-07-04.md. Jobs in
`queue/jobs2.jsonl` (fresh file; done/failed markers in queue2/). The generator refills from
the §7 filler ladder when depth < 3. CPU battery (A2 power curves) runs OUTSIDE the GPU
queue (its own nohup + log; it must not block GPU workers).

## 5. Timeline (estimates; adapt via §7, never by editing the frozen core)

| Hours | GPU 0 | GPU 1 | GPU 2 | CPU |
|---|---|---|---|---|
| 0–0.5 | gates + push public | gates | (R1 s2 tail) | plan copy, ledger check |
| 0.5–2.5 | A3 trainings ×3 | A1 multi-y | A2 archives | — |
| 2.5–4.5 | A3 trainings tail | A4 Rémy impl+gate+K-sweep | A2 archives → A3 audit prep | A2 power battery |
| 4.5–6.5 | A3 audit grid | A4 misspec pair + fillers | A3 audit grid | A2 battery + analysis |
| 6.5–8 | fillers | fillers | fillers | dawn assembly |

## 6. Failure playbook

- OOM → halve batch/N; JAX no-prealloc everywhere; never two trainings + audit on one GPU.
- Training NaN → diagnose-training checklist, 30-min box, fall back to smaller net.
- MIRA implementation unclear → time-box 45 min; if no trustworthy implementation or
  reproduction of the paper's toy numbers, drop MIRA (log), run PQMass+TARP; P-20260703c
  resolves on the tests that pass their null gate.
- A new sampler fails its gate → fix ≤30 min or drop the arm (priority order protects the
  rest); a partial night must still be scoreable.
- GitHub push fails → log, continue local-only, retry at checkpoints.
- Another user's PIDs appear → downshift immediately (their GPU wins).

## 7. Live adaptation protocol

**Firewall:** may reallocate compute, reorder, add exploratory jobs (tagged `exploratory`
with a one-line expected-yield note); may NOT edit P-20260703b–e or the four confirmatory
grids. Confirmatory/exploratory split stays clean in the JSONLs.

**Checkpoint read #1 (~H2.5) — after A1 done + first A2 nulls/power rows:**
- P-b-shaped (ratios stable): proceed; fold spread numbers into the note draft plan.
- Ratios NOT stable across y (>25% spread): this is a FINDING — densify y (24 draws), add
  the y-dependence diagnosis (which modes drive it), deprioritize A4 by one slot.
- Diagnostics all failing nulls: suspect harness/wrapping before science; re-derive on toy
  Gaussians; only then believe anything about power.

**Checkpoint read #2 (~H5) — after A3 audit rows + A4 K-sweep:**
- P-d-shaped (amortized passes summaries, fails geometry): add the "failure onset" analysis
  (which training budget first fails) — this is the D&J-reproduction headline; consider a
  4th conditional net at 2× capacity as exploratory.
- Amortized nails geometry too (P-d MISS): equally interesting — add training-budget
  DOWN-ladder (5k, 2k steps) to find where it breaks; the note's message becomes "amortized
  is fine at this substrate's difficulty; steering is the problem."
- Rémy K-curve converging as predicted: add the K-vs-compute-matched comparison (Rémy@K
  equal-NFE vs DPS single-pass — same budget, who's less wrong? THE practical figure).
- Compensation config FOOLING diagnostics (P-c HIT): zoom — budget ladder to 16k samples
  (does ANY budget catch it?), add twisted@ε=−0.3 control, and generate the "money figure"
  data (diagnostic score vs true W2 scatter across all archives).

**Zoom rules (pre-noted "strange"):** any diagnostic with non-monotone power vs budget;
any arm whose seed-spread exceeds 3× its A1-matched analog; Rémy W2 non-monotone in K
(gate-adjacent — treat as bug first).

**Filler ladder (unbounded, in value order; every rung feeds the note):**
1. More y-realizations (toward 24) at N=256, 64².
2. A2 bootstrap replicates ×3 (error bars on power curves) + budget 16384 rung.
3. Rémy K-ladder densification + equal-NFE comparison rows.
4. Amortized: extra particle-seeds; 4th net (capacity ×2) as exploratory.
5. 128² columns: multi-y (4 draws), dps/twisted/oracle, Rémy K∈{5,30}.
6. ε-ladder refinement around the compensation zero-crossing (find ε* where DPS bias
   exactly cancels; ±0.02 resolution) — nails the trap quantitatively.
7. Seed densification of anything above (never exhausts).

**Cadence:** re-read fresh JSONLs every ~45 min (scripted digest, not eyeballing); one
[STEER] entry per read; public push at each checkpoint.

## 8. Deliverables at dawn

1. `results/a1_multiy.jsonl`, `a2_power.jsonl` (+archives), `a3_amortized.jsonl`,
   `a4_remy.jsonl`, filler outputs; figures: (i) multi-y ratio violins; (ii) power curves
   per diagnostic with the compensation row highlighted; (iii) amortized summaries-vs-
   geometry contrast; (iv) Rémy W2(K) + equal-NFE comparison; (v) diagnostic-score vs
   true-W2 scatter (the certify-the-certifiers money figure).
2. `NIGHT_LOG_2026-07-04.md` complete; utilization report (same dawn_report pattern).
3. `HANDOFF_DAWN_2.md`: P-20260703b–e verdict table (PROPOSED scores + one-line evidence;
   scoring WITH Andreas), per-arm results, incidents, next-actions for the note draft.
4. RESEARCH_LOG: E-entry Results + Updated beliefs drafted (Outcomes/Lessons left for the
   joint scoring session).
5. Public repo pushed through the final commit; README updated with a 5-line "what this is"
   + link to the tour notebook (it's public now — make the landing page make sense).

## 9. Context chain for the fresh session (read in order)

1. This file, IN FULL.
2. `RESEARCH_LOG.md` — P-20260703b–e (frozen) + the scored P-20260702d–h block (context).
3. `HANDOFF_DAWN.md` — last night's results + the four arms' design paragraphs (§7).
4. `NIGHT_LOG.md` — skim the [FAIL]/[STEER] entries: every process-discipline lesson in §0
   was paid for there.
5. `notebooks/overnight_pilot_tour.ipynb` — §§2–4c for the science framing (fast skim).
6. Code: `tilt_audit/` package + `scripts/run_t1.py` + `tests/test_gates.py` (the harness
   tonight's arms extend — reuse everything).
7. Skills: cluster-resources, coding-guidelines, research-principles §§5–7.

## 10. First prompt for the executing session (copy-paste)

```
Execute docs/OVERNIGHT_2026-07-03_ARMS_NIGHT.md in ~/software/tilt-audit. Read it IN FULL
first, then its §9 context chain in order.

Ground rules, non-negotiable:
- Predictions P-20260703b–e in RESEARCH_LOG.md are FROZEN; the four arm grids in §2 are the
  confirmatory core. You never edit them; result-triggered additions are tagged exploratory
  with a one-line expected-yield note each.
- Gates first: the existing 15-gate suite green on current code, plus T-N1/N2/N3 (§3),
  before any large GPU job.
- Make the repo PUBLIC before the first GPU job (gh repo create, push) — the frozen
  predictions must be publicly timestamped BEFORE results exist. Push again at every
  checkpoint and at dawn.
- GPUs 0,1,2 are mine for ~8h, titan, no scheduler. Never idle a GPU while the §7 filler
  ladder has rungs. ≤50 CPU workers for the diagnostic battery. JAX no-prealloc; fp32
  trainings; fp64 samplers/metrics.
- Don't reinvent wheels: pqm and tarp are pip-installable; hunt for the MIRA reference
  implementation before writing one; consult ~/software/PnPMass and CosmoStat/jax-lensing
  for the Rémy schedule. Reuse the existing tilt_audit harness for everything.
- Process discipline (last night's paid lessons): kill by PID only — NEVER pattern-match
  process names; check for surviving children after any kill; append-mode outputs only;
  shell-generated timestamps; two-sentence expected-yield note before any heavy
  result-triggered run.
- Keep NIGHT_LOG_2026-07-04.md as you go ([JOB]/[GATE]/[RESULT]/[STEER]/[FAIL]/[NOTE],
  written when things happen). Steer live per §7: checkpoint reads at ~H2.5 and ~H5,
  45-min cadence, zoom rules, filler ladder.
- Failures: §6 playbook. Time-box debugging; a partial night must still be scoreable.

At dawn deliver: HANDOFF_DAWN_2.md with the P-20260703b–e verdict table (PROPOSED scores +
one-line evidence — final scoring happens with me), the five figures from §8, utilization
report, drafted E-entry results, and the public repo pushed through the final commit. Wake
me for nothing — adapt, log, and keep the GPUs full.
```
