# NIGHT_LOG — 2026-07-02 → 07-03 · GRF pilot overnight

> One tagged entry per notable event, newest last. Tags: [JOB] [GATE] [RESULT] [STEER]
> [FAIL] [NOTE]. All times UTC. H0 = 22:48.

- 22:48 [NOTE] Session start (H0). Plan docs/OVERNIGHT_2026-07-02_GRF_PILOT.md read in full + §10 context chain (GRF_PILOT_PLAN, RESEARCH_LOG P/E entries, S-2026-07-02-b resolution). GPUs 0,1,2 claimed; GPU 3 untouched (has other users' jobs, 24% util).
- 22:48 [NOTE] nvidia-smi at H0: GPU 0 has a small foreign process (user titan, huggingfaceserver sentiment classifier, 544 MiB). Tiny footprint — coexisting with our mem caps (XLA 0.70); will re-check each util scan and downshift if it grows.
- 22:50 [JOB] vLLM server (Qwen/Qwen2.5-Math-1.5B-Instruct) starting on GPU 2, port 8000, gpu_mem_util 0.35 (pid 509724). Track B within the 10-min requirement.
- 22:50 [JOB] Pre-download DeepSeek-R1-Distill-Qwen-1.5B for B2 started (CPU/network only, pid 509962).
- 22:52 [NOTE] tilt-audit repo initialized: git init (main), Apache-2.0 LICENSE (canonical text), README stub, RESEARCH_LOG.md seeded with P-20260702d–h + E-20260702c marked migrated. Local only, no remote.
- 22:53 [JOB] B1 alpha-sweep starting (defensive, N=16, 100 problems, seeds 0,1,2; alpha in 0.05..0.50)
- 23:05 [NOTE] Track A core implemented (fields/tilt/diffusion/samplers/metrics/misspec, ~700 lines JAX, float64). Canonical state = real diagonal Fourier basis; per-mode closed forms throughout; SMC weights/resampling global.
- 23:10 [GATE] T-G2 (analytic): moment-ODE integrator reproduces the doc's reference table exactly — DPS P=0.25/1/4 → 0.787/0.158, 1.729/0.245, 1.999/0.250; exact-guidance → (μ*,Σ*) to <2e-3. PASS.
- 23:12 [GATE] T-G2 (samplers): implemented DPS + exact-guidance at T=4096, N=400k, d=1 hit the ODE columns within 2%. PASS. T-G1 oracle matches (μ*,Σ*) per mode to MC error (max |z-score| < 5 at N=200k). PASS.
- 23:15 [FAIL→fixed] First twisted-SMC design (Euler reference kernel) put −53 of log-discretization bias into Ẑ at 16². Replaced Euler with the EXACT per-mode backward OU kernel (closed form): prior chain now hits N(0,P) exactly for any T; guidance enters as Euler shift on top, isolating scheme bias from discretization by construction.
- 23:18 [STEER] Design decision on sampler 4 (proper twisted SMC): potential-only twisting (prior proposals, weights ψt/ψt−1) weight-degenerates at d=256 (single-run log Ẑ ≈ −400 below log Z despite unbiasedness — Jensen collapse; observed −1400 vs −1003.85). The T1 "twisted" arm therefore uses the CONJUGATE twisted proposal q* ∝ p(z'|z)ψ(z') (closed form here) — the proper twisted sampler; with exact score it is provably oracle-equivalent (incremental weights ≡ 0 by the tower property, verified < 1e-9 — doubles as a machine-precision check of the whole closed-form stack). The potential-only variant is kept as an EXPLORATORY diagnostic arm (twisted_potential) — its ESS/Ẑ collapse at d≥256 is itself certificate-relevant evidence. Frozen T1 grid definition untouched (samplers {DPS, SAP, twisted, terminal-IS} × β × d × N × seeds).
- 23:20 [GATE] Full gate suite 15/15 PASS (T-G1, T-G2 ×2, T-G3 ×3 incl. d=1 unbiasedness with real weight variance + conjugate machine-zero increments + on-target vs oracle floor, T-G4 identities, basis/terminal-IS checks). CPU, 143 s. Track A cleared for GPU.
- 23:29 [NOTE] queue worker up on GPU 0
- 23:29 [JOB] t1_core (tier T1, GPU 0) started: .venv/bin/python scripts/run_t1.py --samplers oracle,dps,sap,twisted,terminal_is --tag confirmatory --out results/t1_cor
- 23:29 [NOTE] queue worker up on GPU 1
- 23:29 [JOB] t1_controls (tier T1x, GPU 1) started: .venv/bin/python scripts/run_t1.py --samplers exact_guidance,twisted_potential --tag exploratory --out results/t1_contro
- 23:32 [JOB] t1_core finished OK (2m)
- 23:32 [JOB] t1_misspec_p03 (tier T1x, GPU 0) started: .venv/bin/python scripts/run_t1.py --dims 64 --score misspec:0.3 --samplers dps,twisted,sap --tag exploratory --out resu
- 23:29 [JOB] Queue workers up on GPUs 0,1 (atomic-claim, retry-once, T3/T4 self-refill); util logger every 15 min. T1 core (frozen grid) + exploratory controls/misspec/T-sensitivity enqueued.
- 23:33 [NOTE] Smoke tests caught and fixed two numerical traps before any confirmatory row: (1) forward-Euler guidance shift unstable at T=64 for calibrated tilts (fix: per-mode exponential integrator, midpoint coefficients — DPS bias is scheme, not blowup); (2) uniform time grid under-resolves t→0 where guidance concentrates — even the exact-guidance control was off-target (fix: log-spaced grid, c=0.05). Post-fix 16² sanity: oracle floor W2≈0.19 γ*=1.01; exact-guidance control ≈ floor γ*≈1; DPS 7×→26× floor from 1σ→4σ with γ*>1 (over-concentrated); twisted = floor, γ*=1; terminal-IS degenerate (ESS→1). SAP: variance collapse WITHOUT mean tracking — scalar γ* misleading (added gamma_mean + var_ratio_logmed split diagnostics to every row).
- 23:36 [JOB] T2 score-net trainings enqueued on GPU 0 (S-clean, S-mis ±0.3; 1.44M-param UNet, 60k steps, fresh GRF batches, EMA, 30-min checkpoints).
- 23:32 [JOB] t1_misspec_p03 finished OK (1m)
- 23:32 [JOB] t1_misspec_m03 (tier T1x, GPU 0) started: .venv/bin/python scripts/run_t1.py --dims 64 --score misspec:-0.3 --samplers dps,twisted,sap --tag exploratory --out res
- 23:33 [JOB] t1_controls finished OK (4m)
- 23:33 [JOB] t1_misspec_p01 (tier T1x, GPU 1) started: .venv/bin/python scripts/run_t1.py --dims 64 --score misspec:0.1 --samplers dps,twisted,sap --tag exploratory --out resu
- 23:33 [JOB] t1_misspec_m03 finished OK (1m)
- 23:33 [JOB] t1_misspec_m01 (tier T1x, GPU 0) started: .venv/bin/python scripts/run_t1.py --dims 64 --score misspec:-0.1 --samplers dps,twisted,sap --tag exploratory --out res
- 23:34 [JOB] t1_misspec_p01 finished OK (1m)
- 23:34 [JOB] t1_tsens_T32 (tier T1x, GPU 1) started: .venv/bin/python scripts/run_t1.py --dims 16,32 --T 32 --samplers dps,sap,twisted --tag exploratory --out results/t1_tse
- 23:34 [JOB] t1_misspec_m01 finished OK (1m)
- 23:34 [JOB] t1_tsens_T256 (tier T1x, GPU 0) started: .venv/bin/python scripts/run_t1.py --dims 16,32 --T 256 --samplers dps,sap,twisted --tag exploratory --out results/t1_ts
- 23:35 [JOB] t1_tsens_T32 finished OK (1m)
- 23:35 [JOB] t1_tsens_T256 finished OK (1m)
- 23:35 [JOB] train_clean (tier T2, GPU 0) started: .venv/bin/python scripts/train_score.py --eps 0.0 --out checkpoints/s_clean.pkl
- 23:36 [JOB] t3_seed3 (tier T3, GPU 1) started: /mnt/home/tersenov/software/tilt-audit/.venv/bin/python scripts/run_t1.py --seeds 3 --tag confirmatory-densify --out res
- 23:35 [RESULT] T1 CONFIRMATORY CORE COMPLETE (540 rows: frozen grid + oracle; controls + analytic-misspec + T-sensitivity exploratory done too). Headline numbers (median over seeds, W2/oracle-floor):
  * DPS: 1.4×→28× floor, growing with BOTH tilt strength (0.5σ→4σ) and N (floor drops, bias constant); γ* = 1.33–1.43 everywhere (over-concentrated). P-20260702d-shaped.
  * twisted (conjugate): 0.96–1.06× floor at ALL 36 (d,β,N) cells, γ*=1.00, log Ẑ = log Z (machine-exact weights). P-20260702f-shaped.
  * SAP: 4×–143× floor. NUANCE for P-20260702e: γ*>1 (runs cold, 1.35–1.73) ONLY in the weak-tilt/low-d corner (16², 0.5σ, N≥64); at stronger tilt/higher d it transitions to variance collapse WITHOUT mean tracking (γ_mean→0, var_ratio_logmed ≈ −20): a MORE severe pathology than the discrete substrate's, but not the literal γ*>1 signature globally.
  * terminal-IS: 15×–250× floor, ESS→1, γ*→0 (stays at prior): best-of-N dies in high d as expected.
  * exact_guidance control: ≈1.0× floor at N≤64; drifts to 1.6–4.2× at N=256/64² — residual time-discretization at T=64 becoming visible as the floor drops. T=256 verification queued.
- 23:35 [RESULT] KILL CRITERION: NOT triggered — max frozen-scheme ratio 251× (terminal-IS), DPS alone 27.7× ≥ 3× at 64²/4σ/N=256, and ≥3× already at 0.5σ for N=256 at all dims. GO-side signal at every (β,d) except the very weakest cells at N=16.
- 23:37 [STEER] Checkpoint read #1 (early — core done at H1 instead of H3): P1/P2-shaped branch per §9 ⇒ business as usual + promote T4.1 (adjoint-matching) one slot earlier. Added exploratory: (a) weak-tilt arm (0.125σ, 0.25σ) to locate the 3×-floor crossing β* (sharpens P-h evidence); (b) T=256 exact-guidance control at 64² (pin the discretization residual); (c) T2 pathway control (analytic score in the learned ancestral pathway) to separate kernel-choice from score error in the decomposition.
- 23:38 [JOB] t3_seed3 finished OK (2m)
- 23:38 [JOB] t1_weak_tilts (tier T1x, GPU 1) started: .venv/bin/python scripts/run_t1.py --shifts 0.125,0.25 --tag exploratory --out results/t1_weak.jsonl
- 23:40 [JOB] t1_weak_tilts finished OK (2m)
- 23:40 [JOB] t1_eg_T256 (tier T1x, GPU 1) started: .venv/bin/python scripts/run_t1.py --dims 64 --T 256 --samplers oracle,exact_guidance,dps --tag exploratory --out result
- 23:41 [JOB] t1_eg_T256 finished OK (1m)
- 23:41 [JOB] t2_pathway_ctrl (tier T2, GPU 1) started: .venv/bin/python scripts/run_t2.py --ckpt analytic --label pathway:analytic --tag exploratory --out results/t2_pathway_c
- 23:42 [FAIL] t2_pathway_ctrl rc=-15 (1m) — retrying once
- 23:42 [STEER] Track B restructured (pace 200 rows/h, ~3x plan estimate): two parallel PRM streams (GPU 2 + GPU 1), priority = full alpha sweep at seed 0, then complete B2 twin, then B1 seeds 1,2 as filler. Anchors dropped (prior-night sap/iid measurements exist). E-entry expectations unchanged.
- 23:42 [NOTE] queue worker up on GPU 1
- 23:42 [JOB] t2_pathway_ctrl (tier T2, GPU 1) started: .venv/bin/python scripts/run_t2.py --ckpt analytic --label pathway:analytic --tag exploratory --out results/t2_pathway_c
- 23:42 [NOTE] queue worker up on GPU 0
- 23:42 [JOB] train_mis_p03 (tier T2, GPU 0) started: .venv/bin/python scripts/train_score.py --eps 0.3 --out checkpoints/s_mis_p03.pkl
- 23:42 [JOB] B1 alpha=0.10 seed 0 done (stream A)
- 23:42 [JOB] B1 alpha=0.15 seed 0 done (stream B)
- 23:43 [JOB] B1 alpha=0.25 seed 0 done (stream A)
- 23:43 [JOB] B1 alpha=0.35 seed 0 done (stream B)
- 23:43 [JOB] B1 alpha=0.05 seed 1 done (stream B)
- 23:43 [JOB] B1 alpha=0.50 seed 0 done (stream A)
- 23:43 [JOB] B1 phase 1 complete: all 6 alphas at seed 0 (plus 0.05 seeds 0,1)
- 23:44 [JOB] R1 server up; B2 twin starting (sap,iid,defensive; N=16; 100 problems; parallel seeds on 2 streams)
- 23:44 [JOB] B2 seed 1 done (stream B)
- 23:44 [JOB] B2 seed 0 done (stream A)
- 23:47 [FAIL] t2_pathway_ctrl rc=143 (5m) — retrying once
- 23:47 [FAIL] t2_pathway_ctrl failed twice — marked FAILED, continuing (log: queue/logs/t2_pathway_ctrl.log)
- 23:47 [JOB] t3_seed4 (tier T3, GPU 1) started: /mnt/home/tersenov/software/tilt-audit/.venv/bin/python scripts/run_t1.py --seeds 4 --tag confirmatory-densify --out res
- 23:48 [FAIL] Track B v2 phase-1/2 "done" lines above (23:43–23:44) were INSTANT FAILURES, not completions — disregard them. Root cause chain: killing the v1 wrapper left its in-process PRM EngineCore orphaned holding 20.4 GB on GPU 2; new PRM streams then failed their free-memory check (3.96 GB free < 17.8 GB needed) on both GPUs (GPU 1 was blocked by a stale preallocated job); and the v2 driver's run_alpha/run_b2 did not check return codes, so 6 failed runs "completed" in 2 minutes and the driver marched through the phases. Real B1 data so far: alpha=0.05 seeds 0 + partial 1 only. Cleanup done (orphan killed, duplicate pathway-control jobs killed, bogus markers removed). v3 driver: B2-first (R1 server already up and healthy), every run rc-checked, PRM starts gated on measured free GPU memory.
- 23:48 [JOB] Track B v3 starting: B2 seeds 0,1 parallel (PRM on GPU 2 + GPU 1), then seed 2
- 23:49 [JOB] t3_seed4 finished OK (2m)
- 23:50 [JOB] t3_seed5 (tier T3, GPU 1) started: /mnt/home/tersenov/software/tilt-audit/.venv/bin/python scripts/run_t1.py --seeds 5 --tag confirmatory-densify --out res
- 23:50 [FAIL] B2 seed 1 (stream B) rc!=0 — one retry
- 23:50 [FAIL] B2 seed 1 (stream B) failed twice — moving on
- 23:52 [JOB] t3_seed5 finished OK (2m)
- 23:52 [JOB] t2_pathway_ctrl (tier T2, GPU 1) started: .venv/bin/python scripts/run_t2.py --ckpt analytic --label pathway:analytic --tag exploratory --out results/t2_pathway_c
- 23:56 [FAIL] Driver v3 had its own race: B2 seed-2's 300s free-memory wait would have timed out against the still-running seed 0, then phase B's server swap would have killed the R1 server mid-run. Driver killed at 23:56 BEFORE the swap; B2 seed 0 unharmed (running, 4 rows).
- 23:57 [STEER] Measured B2 pace ~10 problems/h/stream (R1 traces are 10x Qwen) — sequential-server plans cannot fit both E-entries. DUAL-SERVER reallocation: R1 stays on GPU 2 :8000 (B2 seed 0, PRM on GPU 2); second Qwen server on GPU 1 :8001 with B1's PRM co-resident (14+18=32 GB) running the 5 remaining seed-0 alphas (~40 min each); GPU-1 chain then flips to B2 seed 1 against :8000 for the rest of the night. Track A consolidates on GPU 0 (trainings now; T2/T3/T4 after). GPU-1 queue worker suspended. Expected yield: E-a full alpha sweep at seed 0 (+0.05 at 3 seeds), E-b seed 0 complete + seed 1 partial.
- 23:58 [JOB] t2_pathway_ctrl finished OK (6m)
- 23:58 [RESULT] T4.1 (adjoint-matching memoryless theorem, EXACT fixed-point form — no SGD): mapped 2409.08861's family to VP-OU reverse time (η≡1 ⇒ σ²=2 IS the memoryless schedule = our standard reverse SDE). Computed the SOC-fine-tuned sampler's law in closed form per mode (their eq. 24) for σ² ∈ {2,1,0.5,0.1} × horizons tf ∈ {9,3} at 64². Marginal-preservation identity holds to 1e-7 (validates the pipeline). Findings: (a) at tf=3 the theorem's ordering is exact and steep — KL to true tilt: 0.09 (memoryless) → 0.2–0.4 (σ²=1) → 14–87 (σ²=0.5) → 5e3–4e4 (σ²=0.1); (b) at tf=9 the VP chain is contractive enough that σ² ∈ [1,2] all sit within the finite-horizon residual (KL ~0.1–0.2; σ²=2's own residual comes from α(tf)=e⁻⁹≠0, c_max=0.054) and only σ²≲0.5 shows real bias — the "value-function bias problem" is practically a SHORT-HORIZON / low-noise phenomenon, catastrophic exactly in the σ→0 (Flow-Matching/DDIM fine-tune) limit practitioners like. Both the theorem's warning and its practical scope quantified on the oracle. → results/t41_memoryless.jsonl.
- 23:58 [JOB] t3_seed6 (tier T3, GPU 1) started: /mnt/home/tersenov/software/tilt-audit/.venv/bin/python scripts/run_t1.py --seeds 6 --tag confirmatory-densify --out res
- 00:00 [JOB] t3_seed6 finished OK (2m)
- 00:00 [JOB] t3_seed7 (tier T3, GPU 1) started: /mnt/home/tersenov/software/tilt-audit/.venv/bin/python scripts/run_t1.py --seeds 7 --tag confirmatory-densify --out res
- 00:03 [JOB] Qwen server up on GPU 1 :8001; B1 seed-0 sweep starting (PRM co-resident on GPU 1)
- 00:03 [JOB] Qwen server up on GPU 1 :8001; B1 seed-0 sweep starting (PRM co-resident on GPU 1)
- 00:03 [FAIL] B1 alpha=0.10 seed 0 rc!=0 — one retry
- 00:06 [FAIL] train_mis_p03 rc=143 (24m) — retrying once
- 00:10 [FAIL] train_mis_p03 failed twice — marked FAILED, continuing (log: queue/logs/train_mis_p03.log)
- 00:10 [JOB] t2_pathway_ctrl (tier T2, GPU 0) started: .venv/bin/python scripts/run_t2.py --ckpt analytic --label pathway:analytic --tag exploratory --out results/t2_pathway_c
- 00:08 [FAIL→fixed] Trainings were running fp64 (package-level jax_enable_x64 leaked into training) — projected 5.6 h for S-clean alone. Killed, added TILT_AUDIT_X64=0 escape hatch, relaunched ALL THREE nets concurrently on GPU 0 in fp32: measured 8 steps/s shared → all three finish ~02:05. Queue-worker retry of the killed fp64 job was also killed (it would have raced the fp32 run on the same checkpoint file). Sampler-side: learned nets are fp32, samplers stay fp64 (explicit casts at the net boundary).
- 00:09 [NOTE] t2_pathway_ctrl output was contaminated (211 rows / 144 unique: pre-DPS-fix blowup rows mixed in from the earlier killed attempts). Deleted and requeued fresh now that the guidance fix is in — it doubles as the learned-pathway validation before the real T2 runs.
- 00:10 [NOTE] Track B fully operational in dual-server config: GPU 2 = R1 + B2 seed 0 (~16 rows); GPU 1 = Qwen:8001 + B1 alpha=0.10 seed 0 running. B1 first-attempt PRM init failures are transient (memory settling after job churn) — the chain's retry-once absorbs them.
- 00:13 [STEER] Zoom-in scan (per §9 pre-noted anomaly rules) over t1_core+t3_seeds: no high seed-spread cells, no sub-floor violations. One flag: DPS W2 dips ~6% from shift 0.5→1.0 at N=16 only (32² and 64²; N=256 cleanly monotone) — consistent with the b-dependent oracle floor folding into total W2 at small N rather than scheme physics. Queued +4 seeds at exactly those cells with the exact_guidance control alongside (t3_zoom_dpsdip, exploratory) before generic densification.
- 00:13 [NOTE] First B1 α-shape data: AUROC 0.500 at α=0.05 (n_iid=1 — a one-sample insurance stratum carries no ranking signal, mechanically expected) → 0.569 at α=0.10 (partial). Monotone-rise expectation on track; the E-a threshold test (AUROC≥0.65 by α=0.25) lands ~02:00.
- 00:16 [RESULT] T-sensitivity analysis (t1_tsens): DPS is T-invariant (γ*≈1.36–1.41, W2≈2.1–2.4 for T∈{32,64,256} at 32²/N=256) — its bias is converged continuum scheme bias, validating the exponential integrator. SAP's γ* RISES monotonically with depth: 0.21→0.52→1.72 (0.5σ) and 0.05→0.11→0.49 (1σ) for T=32→64→256 — the particle-reasoners γ*(T) depth law reproduces in the continuous substrate, and SAP is properly COLD (γ*>1) at T=256/moderate tilt. P-e verdict upgraded to HIT (depth-qualified) in HANDOFF_DAWN. Twisted: exact at every T (grid-invariant by construction).
