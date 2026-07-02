# NIGHT_LOG — 2026-07-02 → 07-03 · GRF pilot overnight

> One tagged entry per notable event, newest last. Tags: [JOB] [GATE] [RESULT] [STEER]
> [FAIL] [NOTE]. All times UTC. H0 = 22:48.

- 22:48 [NOTE] Session start (H0). Plan docs/OVERNIGHT_2026-07-02_GRF_PILOT.md read in full + §10 context chain (GRF_PILOT_PLAN, RESEARCH_LOG P/E entries, S-2026-07-02-b resolution). GPUs 0,1,2 claimed; GPU 3 untouched (has other users' jobs, 24% util).
- 22:48 [NOTE] nvidia-smi at H0: GPU 0 has a small foreign process (user titan, huggingfaceserver sentiment classifier, 544 MiB). Tiny footprint — coexisting with our mem caps (XLA 0.70); will re-check each util scan and downshift if it grows.
- 22:50 [JOB] vLLM server (Qwen/Qwen2.5-Math-1.5B-Instruct) starting on GPU 2, port 8000, gpu_mem_util 0.35 (pid 509724). Track B within the 10-min requirement.
- 22:50 [JOB] Pre-download DeepSeek-R1-Distill-Qwen-1.5B for B2 started (CPU/network only, pid 509962).
- 22:52 [NOTE] tilt-audit repo initialized: git init (main), Apache-2.0 LICENSE (canonical text), README stub, RESEARCH_LOG.md seeded with P-20260702d–h + E-20260702c marked migrated. Local only, no remote.
