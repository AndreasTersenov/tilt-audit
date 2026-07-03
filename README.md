# tilt-audit

**What this is.** An oracle-verified correctness audit of inference-time-steered diffusion
sampling: every sampler the field actually uses (plug-in guidance / DPS, reward-as-potential
SMC, twisted SMC, terminal importance reweighting, σ²-inflated annealed Langevin à la Remy
et al., amortized conditional score models), run against targets where the tilted posterior
is **exactly known** — so scheme bias, finite-N noise, discretization, score error, and
model misspecification separate cleanly instead of blurring into "the samples look fine".

The substrate: a Gaussian random field prior N(0, C) with a cosmology-flavored spectrum,
tilted by a quadratic reward r(x) = −‖Ax−y‖²/(2s²). The target σ ∝ p·e^{r/β} is the Wiener
posterior — per-Fourier-mode closed forms for the score, the diffusion marginals, the
optimal twist, W₂, KL, log Z, and every diagnostic's null. A null result can't be blamed on
estimator noise, and a diagnostic that fails its null calibration is caught by construction.

**Pre-registration discipline.** Predictions are frozen in `RESEARCH_LOG.md` *before* each
run night and scored afterwards, never edited. This repo went public before the
2026-07-03→04 run's first GPU job, so P-20260703b–e are externally timestamped.

Start here: [`notebooks/overnight_pilot_tour.ipynb`](notebooks/overnight_pilot_tour.ipynb)
— a guided tour of the pilot night (what was built, why, and what came out), assuming no
diffusion background. Then `HANDOFF_DAWN.md` / `HANDOFF_DAWN_2.md` for the results of the
two run nights, and `NIGHT_LOG*.md` for the as-it-happened chronology.

Layout: `tilt_audit/` (closed forms, samplers, metrics) · `scripts/` (grid runners, gates,
diagnostics battery, figures) · `tests/test_gates.py` (the gate suite that must be green
before any GPU burn) · `results/` (append-only JSONL rows with full provenance) ·
`docs/` (the frozen overnight plans).

License: Apache-2.0.
