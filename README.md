# tilt-audit

**The problem.** Generative models are increasingly used as priors inside
scientific inference: train a diffusion or flow model on simulations, then
*steer* it with data to draw samples from a posterior — over dark-matter maps,
medical images, molecules. The steering methods in actual use are
approximations without guarantees, and on real data their output cannot be
checked, because checking would require the true posterior — the very thing
being computed. Results from these pipelines carry error bars that no one has
verified, and the diagnostics offered as verification have rarely been tested
themselves.

**What this repository is.** A test bench where the true posterior is known
*exactly* — in closed form, to machine precision, at field scale (4,096 to
16,384 dimensions) — used for two audits, run with pre-registered predictions
and a public lab notebook:

1. **Anatomy of the samplers.** Every steering scheme in practical use —
   plug-in guidance, potential-based and properly twisted sequential Monte
   Carlo, terminal importance reweighting, inflated-noise annealed Langevin,
   amortized conditional score and flow-matching models — measured against
   exact targets, so that scheme bias, score error, discretization, finite-N
   noise, and model misspecification separate cleanly. Mechanisms are
   confirmed analytically where possible (the plug-in bias grid matches an
   independent closed-form ODE prediction to 1–3%).

2. **Audit of the diagnostics.** Everything that claims to *detect* sampler
   failure — coverage tests, two-sample tests, score-based certificates,
   budget-doubling convergence checks — is first forced through a null gate
   (does it pass on a provably perfect sampler?) and then through constructed
   failures with exactly known damage (biased dynamics, contaminated scores,
   compensating errors, missing posterior modes, wrong reference scores). The
   result is an operating envelope for each instrument: what it reliably
   catches, what it is structurally blind to, and where its silence should
   not be mistaken for safety.

**Why it might matter beyond this corner of cosmology.** The question "your
sampler passed the diagnostic — but would the diagnostic have caught the
failure you actually care about?" is the validity question for any evaluation
of a generative system, and it can only be answered where ground truth is
exact. This bench is a small, fully worked instance of that discipline:
null-gate the instrument, construct the failure, measure the blind spot,
pre-register the predictions, publish the misses.

**Read the story:** the full plain-language writeup — every number measured,
every plot element defined — is at
[andreastersenov.github.io/tilt-audit](https://andreastersenov.github.io/tilt-audit/)
(source: [`docs/explainer/`](docs/explainer/certificate_explainer.html)).

![A score-based certificate is blind to a missing mode; sample-space tests are not](figures/fig_mixture.png)

## Headline measurements

| Finding | Where |
|---|---|
| Plug-in guidance bias: 1.4–28× the oracle floor, monotone in steering strength, d-extensive — confirmed analytically to 1–3% by an independent stiff-ODE prediction | `results/t1_core.jsonl` |
| The compensation trap: at score contamination ε\*≈−0.28, the temperature diagnostic reads *exactly* clean (γ\*=1.00) while true error is ~6× floor — and this configuration also evades every sample-based test in the battery | `results/eps_star.jsonl`, `results/a2_power.jsonl` |
| Path-space certificates (importance-weight ledgers) die of weight degeneracy on trained networks — ESS ≡ 1.0 at every scale tested — while their exact-score version is tight | `results/cert_*.jsonl` |
| Score-based certification (reimplemented from arXiv:2602.04189 and given the calibration its authors omit): power 1.00 on dynamics bias including the compensation trap; **statistically indistinguishable from perfect with half the posterior missing** (12σ-separated mode, any budget to 16,384); with a learned score it certifies the most-used biased sampler as clean at the paper's own settings | `results/ksd_trial.jsonl`, `figures/fig_ksd_power.png`, `fig_mixture.png`, `fig_wrongref.png` |
| MCMC gold standards on a nonlinear (lognormal-observation) substrate cost ~74 s per 64² configuration with all correctness gates green, and scale to 128² — offline validation is far cheaper than its reputation | `scripts/run_gold.py`, gates T-L1/2/3 |
| Transfer decay law: the covariance correction that is *exact* on the Gaussian bench buys 11.6× / 2.8× / 1.1× at skewness 0.5 / 1 / 2 — a Gaussian accident — while annealed-Langevin refinement stays ≤15× floor at every nonlinearity | `results/transfer.jsonl`, `figures/fig_transfer.png`, `fig_remyK.png` |
| Budget-doubling convergence checks are one-directional: the alarm is trustworthy, the silence certifies nothing — slow convergence, biased samplers, and stuck modes all pass, and for deterministic-ODE samplers the alarm never fires at all | `results/k2k.jsonl`, `results/nfe2.jsonl` |

## The bench

The substrate is a Gaussian random field prior N(0, C) with a
cosmology-flavored spectrum, tilted by a quadratic reward
r(x) = −‖Ax−y‖²/(2s²); the target is the Wiener posterior, with
per-Fourier-mode closed forms for the score, the diffusion marginals, the
optimal twist, W₂, KL, log Z, and every diagnostic's null. Three extensions
keep the exactness while removing the politeness: an exact two-component
**mixture** arena (missed-mode pathologies), a **nonlinear lognormal
observation** with NUTS gold standards (gated by the λ→0 limit, an independent
closed-form importance-sampling cross-check, and seed-independence tests), and
trained **score / flow-matching networks** for the deployment-realistic
columns.

## Reproduce

```bash
git clone https://github.com/AndreasTersenov/tilt-audit && cd tilt-audit
uv venv && uv pip install -e . "jax[cuda12]" numpyro "arviz<1" pqm tarp
uv run pytest tests/test_gates.py        # the gate suite
uv run python scripts/gate_ksd.py        # null-gates the certificate instrument
uv run python scripts/run_gold.py --n 32 --tilt mid --yseed 0 --linear-check
```

Every results row is append-only JSONL with config and provenance; every
figure regenerates from the JSONLs (`scripts/dawn_figures3.py`). GPU runs were
on single A100s; the gates and small grids run on CPU (the full gate suite
passes on CPU in ~2.5 minutes — CI config in `ci/`).

## Process

Every experiment was **pre-registered**: predictions frozen with confidence
levels in [`RESEARCH_LOG.md`](RESEARCH_LOG.md) and pushed publicly *before*
each night's first GPU job, then scored afterwards — 15+ predictions across
four run nights, hits and misses alike, with corrections logged in the open
when we were the ones who were wrong. The as-it-happened record, including
every mistake, is in [`lab-notebook/`](lab-notebook/).

## Layout

`tilt_audit/` closed forms, samplers, metrics, certificates ·
`scripts/` grid runners, gates, batteries, figures ·
`tests/` the gate suite · `results/` JSONL data (sample banks and gold draws
are regenerable, not tracked) · `figures/` all regenerable ·
`docs/` frozen overnight plans and the writeup ·
`lab-notebook/` night logs and scored handoffs · `RESEARCH_LOG.md` the
prediction ledger.

## Citing

See [`CITATION.cff`](CITATION.cff). License: Apache-2.0.
