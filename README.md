# tilt-audit

**Working name.** Oracle-verified correctness audit for inference-time-steered diffusion
sampling, on targets where the tilted distribution is *exactly* known.

The pilot substrate: a Gaussian random field prior N(0, C) with a cosmology-flavored power
spectrum, tilted by a quadratic reward r(x) = −‖Ax−y‖²/(2s²), so the target
σ ∝ p·e^{r/β} is exactly Gaussian (Wiener posterior) in any dimension — per-Fourier-mode
closed forms for the score, the diffusion marginals, the optimal twist, W₂, KL, and log Z.

Five samplers, one target:
1. **oracle** — direct per-mode Gaussian draws from σ (finite-N floor reference);
2. **DPS plug-in guidance** — Tweedie point-estimate likelihood guidance (the field's workhorse bias);
3. **reward-as-potential SMC** — resample every step on e^{r(x̂₀)/β};
4. **proper twisted SMC** — closed-form optimal twist, ESS-gated resampling, valid log Ẑ;
5. **unguided + terminal importance reweighting**.

Everything analytic: a null result can't be blamed on estimator noise.

Status: overnight GO/NO-GO pilot (2026-07-02). See `RESEARCH_LOG.md` for the frozen,
pre-registered predictions and `NIGHT_LOG.md` for the chronological run record.

License: Apache-2.0.
