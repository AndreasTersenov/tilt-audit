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
