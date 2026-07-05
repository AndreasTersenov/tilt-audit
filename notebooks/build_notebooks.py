#!/usr/bin/env python
"""Generate the guided-tour notebook suite (01..04) from the tracked results.

Each notebook is self-contained, runs top to bottom on CPU in minutes, and
reads only tracked files (results/*.jsonl, figures/). Regenerate with:
    python notebooks/build_notebooks.py
    jupyter nbconvert --execute --inplace notebooks/0*.ipynb
"""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).resolve().parent

PREAMBLE = '''import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"

def rows(name):
    p = RES / name
    if not p.exists():
        print(f"({name} not found, cell skipped)")
        return []
    return [json.loads(l) for l in p.open()]
'''


def nb(cells, path):
    n = nbf.v4.new_notebook()
    n["cells"] = cells
    n["metadata"]["kernelspec"] = {"display_name": "Python 3",
                                   "language": "python", "name": "python3"}
    nbf.write(n, HERE / path)
    print("wrote", path)


md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

# ================================================================ 01
nb01 = [
md(r"""# 01 · The bench

**The problem this project attacks.** Generative models are increasingly used
as priors inside scientific inference. A diffusion or flow model is trained on
simulations, then steered with data so that its samples follow a posterior
distribution over maps, images, or molecules. The steering methods in actual
use are approximations without guarantees, and on real data their output
cannot be checked, because checking would require the true posterior. That is
the thing being computed. So results ship with error bars nobody has verified,
and the diagnostics offered as verification have rarely been tested themselves.

**The move.** Build a world where the true posterior is known exactly, at the
scale where these methods actually operate (fields with thousands of
dimensions). Use it for two audits. First, measure how wrong each sampler
actually is, and why. Second, test the diagnostics themselves: force each one
to pass on a provably perfect sampler before it is allowed an opinion, then
show it constructed failures with exactly known damage and record what it
catches.

This notebook builds the bench and demonstrates its exactness. Notebook 02
covers the sampler anatomy, 03 puts the proposed certificates on trial, and
04 extends everything to a nonlinear substrate with MCMC gold standards."""),
md(r"""## The construction, precisely

The prior is a Gaussian random field. In its Fourier basis it is a list of
independent Gaussians, one per spatial frequency $k$, with variances given by
a power spectrum:

$$z_k \sim \mathcal{N}\big(0,\, P(k)\big), \qquad
P(k) \propto \max(k,1)^{n_s-4}\, e^{-(k/k_c)^2}.$$

**In words:** each spatial scale fluctuates independently, large scales
(small $k$) fluctuate strongly and small scales weakly, the way cosmological
matter fields do. The observation applies a smoothing operator $A$ (diagonal
in the same basis, entries $a_k$) and adds white noise of standard deviation
$s$. Steering the prior toward data $y$ means targeting

$$\sigma(x) \;\propto\; p(x)\, e^{r(x)/\beta}, \qquad
r(x) = -\tfrac{1}{2 s^2}\|Ax - y\|^2 .$$

**In words:** weight every prior map by how well it explains the data, with
$\beta$ setting how hard the data pull. For this family the target is the
classical Wiener posterior, again independent per mode, with closed-form mean
and variance:

$$\Sigma^*_k = \Big(\tfrac{1}{P(k)} + \tfrac{a_k^2}{b}\Big)^{-1},
\qquad \mu^*_k = \Sigma^*_k\, \frac{a_k\, y_k}{b},
\qquad b = \beta s^2 .$$

**In words:** where the instrument sees well ($a_k$ large relative to noise)
the posterior follows the data and its variance shrinks far below the prior.
Where the instrument is blind, the posterior falls back to the prior. Every
quantity this project measures against (the score, the diffusion marginals,
the optimal twist, distances like $W_2$ and KL, the null of every diagnostic)
follows from these two formulas in closed form, at any size."""),
code(PREAMBLE + r'''
from tilt_audit import tilt
from tilt_audit.fields import (grid_to_z, make_basis, make_pk,
                               smoothing_operator, unpack)
import jax
import jax.numpy as jnp

n = 64
basis = make_basis(n)
Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
az = jnp.asarray(smoothing_operator(basis))
y, z_truth = tilt.make_observation(jax.random.PRNGKey(999), Pz, az, 0.5)
b = float(tilt.calibrate_b(Pz, az, y, target_shift=1.0))
mu, Sig = tilt.posterior_params(Pz, az, y, b)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, field, title in [
        (axes[0], unpack(z_truth, basis), "a prior draw (the 'truth')"),
        (axes[1], unpack(y, basis), "the observation y (smoothed + noisy)"),
        (axes[2], unpack(mu, basis), "exact posterior mean (closed form)")]:
    ax.imshow(np.asarray(field), cmap="RdBu_r")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.show()
print(f"dimensions: {int(Pz.shape[0])}, steering strength b = {b:.4g}")'''),
md(r"""**Reading the panels.** Left: one random map the prior considers
plausible. Middle: what the instrument records of it, blurred by $A$ and
buried in noise at small scales. Right: the exact posterior mean, which
recovers structure exactly where the data support it and smooths away what
the noise has destroyed.

The next figure shows the same statement in frequency space, which is where
this bench does all its accounting."""),
code(r'''kz = grid_to_z(basis.kmag, basis)
order = np.argsort(kz)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.loglog(kz[order][1:], np.asarray(Pz)[order][1:], lw=1.2,
          label="prior variance P(k)")
ax.loglog(kz[order][1:], np.asarray(az)[order][1:] ** 2, lw=1.2,
          label="instrument response a(k)$^2$")
ax.set_xlabel("spatial frequency |k|")
ax.set_title("the two ingredients")
ax.legend(fontsize=9)

ax = axes[1]
ratio = np.asarray(Sig) / np.asarray(Pz)
ax.semilogx(kz[order][1:], ratio[order][1:], ".", ms=2, color="#31688e")
ax.set_xlabel("spatial frequency |k|")
ax.set_ylabel("posterior variance / prior variance")
ax.set_ylim(0, 1.05)
ax.set_title("where the data inform")
plt.tight_layout()
plt.show()
print(f"most informed mode: posterior variance shrunk to "
      f"{ratio.min():.3f} of the prior")
print(f"fraction of modes where data cut the variance by half or more: "
      f"{np.mean(ratio < 0.5):.2f}")'''),
md(r"""**Reading the right panel.** Each dot is one of the 4,096 modes. A value
of 1 means the data taught the posterior nothing about that mode, a value near
0 means the data pinned it. The transition happens where the instrument
response falls below the noise. Sampler failures live in exactly this
geometry: a biased sampler typically gets the well-informed modes roughly
right and quietly mis-states the transition region, which is why this project
reports per-band numbers throughout.

## Steering strength, calibrated not guessed

Comparisons across settings need a physical scale for "how hard did the data
pull". We use the root-mean-square shift of the posterior mean away from the
prior mean, measured in units of the prior standard deviation, and we solve
for the $b$ that produces a requested shift:

$$\mathrm{shift}(b) = \sqrt{\tfrac{1}{d}\textstyle\sum_k
\mu^{*2}_k(b) / P(k)} .$$

**In words:** a shift of 1 means the data moved the answer by about one prior
sigma, a strong but realistic pull. The project's standard grid uses shifts
of 0.5, 1, 2, and 4."""),
code(r'''bs = np.geomspace(1e-4, 1e2, 60)
shifts = [float(tilt.rms_shift(Pz, az, y, float(bb))) for bb in bs]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.semilogx(bs, shifts, lw=1.5)
for target in (0.5, 1.0, 2.0, 4.0):
    bcal = float(tilt.calibrate_b(Pz, az, y, target_shift=target))
    ax.plot([bcal], [target], "o", color="#b3261e")
    ax.annotate(f"{target}$\sigma$", (bcal, target),
                textcoords="offset points", xytext=(8, -2), fontsize=9)
ax.set_xlabel("tilt parameter b (small = strong pull)")
ax.set_ylabel("posterior shift (prior sigmas)")
ax.set_title("calibrating steering strength")
plt.tight_layout()
plt.show()'''),
md(r"""## Error bars of zero

The payoff of the closed forms: the distance between any Gaussian sampler
output and the truth is itself a formula. For diagonal Gaussians the squared
Wasserstein-2 distance decomposes per mode:

$$W_2^2 = \sum_k \Big[(m_k - \mu^*_k)^2 +
\big(\sqrt{v_k} - \sqrt{\Sigma^*_k}\big)^2\Big].$$

**In words:** a mean-error term plus a width-error term, per mode, summed.
The demonstration below builds a deliberately broken sampler (all posterior
widths inflated by 30%), computes its exact damage from the formula, and
shows the empirical estimate converging to that exact value as the sample
budget grows. On this bench, "how wrong is the sampler" is not itself an
estimate with error bars. It is a number."""),
code(r'''infl = 1.3
w2_exact = float(np.sum((np.sqrt(infl) - 1.0) ** 2 * np.asarray(Sig)))
Ns = [64, 256, 1024, 4096, 16384]
est = []
for N in Ns:
    k = jax.random.PRNGKey(N)
    zz = np.asarray(mu)[None] + np.sqrt(infl * np.asarray(Sig))[None] \
         * np.asarray(jax.random.normal(k, (N, int(Pz.shape[0]))))
    m_hat, v_hat = zz.mean(0), zz.var(0)
    est.append(float(np.sum((m_hat - np.asarray(mu)) ** 2
               + (np.sqrt(v_hat) - np.sqrt(np.asarray(Sig))) ** 2)))
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.axhline(w2_exact, color="#b3261e", lw=1.2,
           label=f"exact damage (formula): {w2_exact:.3f}")
ax.plot(Ns, est, "o-", color="#31688e", label="empirical estimate")
ax.set_xscale("log")
ax.set_xlabel("sample budget N")
ax.set_ylabel("W$_2^2$ to the true posterior")
ax.set_title("a 30%-overdispersed sampler, damage known exactly")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()'''),
md(r"""## The null-gate discipline

The same exactness disciplines the diagnostics. Before any test is allowed to
judge a sampler in this project, it must pass a null gate: run it on the
oracle (exact draws from the closed-form posterior) and require the nominal
false-alarm behavior. A test that flags a provably perfect sampler is broken,
whatever else it claims. This rule found real problems in released tools
during the project, and every instrument in notebooks 02 and 03 passed
through it first. Here is the gate for the bench itself."""),
code(r'''from tilt_audit import samplers

res = samplers.oracle(jax.random.PRNGKey(0), Pz, az, y, b, N=4096, T=1, tf=9.0)
z = np.asarray(res["z"])
zscores = (z.mean(0) - np.asarray(mu)) / np.sqrt(np.asarray(Sig) / z.shape[0])
vratio = z.var(0) / np.asarray(Sig)
print(f"per-mode mean z-scores: max |z| = {np.abs(zscores).max():.2f} "
      f"(expected < ~4.3 for {z.shape[1]} modes under pure noise)")
print(f"per-mode variance ratios: median = {np.median(vratio):.4f} "
      f"(expected 1.000 +- {np.sqrt(2/z.shape[0]):.3f})")'''),
md(r"""Both numbers sit at their theoretical noise floors.

**Where the data live.** Every experiment appends rows to `results/*.jsonl`
with full configuration. Every figure regenerates from those files. The
prediction ledger (`RESEARCH_LOG.md`) holds the pre-registered expectations
for each experiment, frozen and pushed publicly before the runs, and scored
afterwards, misses included. Continue with notebook 02."""),
]

# ================================================================ 02
nb02 = [
md(r"""# 02 · Sampler anatomy

Six ways to steer a diffusion sampler, measured against exact targets.

* **oracle**: draws from the exact posterior (the finite-N floor reference).
* **plug-in guidance (DPS-class)**: nudges each denoising step with the data
  gradient through a point estimate. Cheap, ubiquitous, no guarantee.
* **reward-as-potential SMC**: a particle population resampled every step on
  the reward. Improper by construction (the tilt compounds with depth).
* **twisted SMC**: the properly weighted particle method, with the optimal
  twist available in closed form on this bench.
* **terminal reweighting**: unguided samples, importance-reweighted once at
  the end.
* **inflated-noise annealed Langevin**: a ladder of noise levels with K
  Langevin corrections per level, the data pull deliberately weakened while
  the map is still noisy.

The mechanism at the heart of the first result deserves its equation. At step
time $t$ the exact data pull on mode $k$ is

$$g^{\mathrm{exact}}_k \;=\; \nabla_z \log
\mathcal{N}\!\big(y_k;\; a_k \hat{x}_k(z),\; b + a_k^2\, v_k(t)\big),$$

while plug-in guidance uses the same expression **without** the
$a_k^2 v_k(t)$ term. **In words:** the exact pull discounts the data early in
generation, when the current estimate $\hat{x}$ is still uncertain by
$v_k(t)$. The plug-in shortcut trusts its point estimate completely at every
step, so it over-pulls early and locks in overconfident structure. Everything
in the first figure follows from that one missing term."""),
code(PREAMBLE + r'''
t1 = [r for r in rows("t1_core.jsonl") if r.get("dim") == 64
      and r.get("eps", 0) == 0]
by = defaultdict(list)
floors = defaultdict(list)
for r in t1:
    if r["sampler"] == "oracle":
        floors[r["shift"]].append(r["w2"])
for r in t1:
    if r["sampler"] != "oracle":
        by[(r["sampler"], r["shift"])].append(r["w2"])

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for s in sorted({k[0] for k in by}):
    shifts = sorted({k[1] for k in by if k[0] == s})
    ratios = [np.median(by[(s, sh)]) / np.median(floors[sh]) for sh in shifts]
    lo = [np.quantile(np.array(by[(s, sh)]) / np.median(floors[sh]), 0.25)
          for sh in shifts]
    hi = [np.quantile(np.array(by[(s, sh)]) / np.median(floors[sh]), 0.75)
          for sh in shifts]
    ax.plot(shifts, ratios, "o-", label=s)
    ax.fill_between(shifts, lo, hi, alpha=0.15)
ax.axhline(1, color="gray", lw=0.8, ls="--")
ax.set_yscale("log")
ax.set_xlabel("steering strength (posterior shift, in prior sigmas)")
ax.set_ylabel("W$_2^2$ error / oracle floor")
ax.set_title("Sampler error vs the exact posterior (64x64 fields, "
             "bands = interquartile over seeds)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()'''),
md(r"""**Reading the figure.** Each line is one sampler, each point a steering
strength, the shaded band the interquartile range over random seeds. The
properly weighted method (twisted) sits on the floor, as theory demands on
this conjugate bench, which makes it the control that validates the harness
rather than a finding. Plug-in guidance (dps) is substantially and
monotonically biased. The resampling shortcut (sap) and terminal reweighting
degrade faster. The dps curve was additionally confirmed by an independent
closed-form calculation (a stiff-ODE integration of the guided dynamics per
mode) that reproduces the measured grid to 1 to 3 percent, so the mechanism
is understood, not just observed.

## The anatomy of the bias

Two views of the same error, from the same rows. Left, the **width story**:
the median log-ratio of sampler width to true width per mode (0 means correct
widths, negative means overconfident). Right, **what that costs**: how often
the true map falls inside the sampler's central 68% interval. The honest
value is 0.68."""),
code(r'''fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
for s in sorted({k[0] for k in by}):
    shifts = sorted({k[1] for k in by if k[0] == s})
    vr = [np.median([r["var_ratio_logmed"] for r in t1
                     if r["sampler"] == s and r["shift"] == sh])
          for sh in shifts]
    cov = [np.median([r["cov_mean_68"] for r in t1
                      if r["sampler"] == s and r["shift"] == sh])
           for sh in shifts]
    axes[0].plot(shifts, vr, "o-", label=s)
    axes[1].plot(shifts, cov, "o-", label=s)
axes[0].axhline(0, color="gray", lw=0.8, ls="--")
axes[0].set_ylabel("median log (sampler width / true width)")
axes[0].set_title("width errors (negative = overconfident)")
axes[1].axhline(0.68, color="gray", lw=0.8, ls="--")
axes[1].set_ylabel("68% interval coverage of the truth")
axes[1].set_title("what it costs: coverage")
for ax in axes:
    ax.set_xlabel("steering strength (prior sigmas)")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()'''),
md(r"""**Reading the panels.** dps loses width steadily as steering grows and
its coverage collapses far below 0.68: error bars that look precise and are
simply wrong. sap collapses much harder. The twisted control holds both
dashed lines, which is what "the harness is sound" looks like in this view.

## Two scaling laws worth knowing

Left: the plug-in bias is **extensive in dimension**. The same experiment at
16x16, 32x32, and 64x64 (256 to 4,096 modes), at fixed steering strength: the
floor-relative error keeps growing with resolution, so bigger maps make the
problem worse, not better. Right: the resampling shortcut's pathology is
**depth-dependent**. Its effective tilt exponent $\gamma^*$ (the steering
strength the samples actually express, recovered by fitting the per-mode
widths, 1 means faithful) rises with the number of steps T, because its
potentials compound at every resampling."""),
code(r'''t1all = [r for r in rows("t1_core.jsonl") if r.get("eps", 0) == 0]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
for sh in (0.5, 1.0, 2.0):
    dims, vals = [], []
    for dim in (16, 32, 64):
        dps_w = [r["w2"] for r in t1all if r["sampler"] == "dps"
                 and r["dim"] == dim and r["shift"] == sh]
        orc_w = [r["w2"] for r in t1all if r["sampler"] == "oracle"
                 and r["dim"] == dim and r["shift"] == sh]
        if dps_w and orc_w:
            dims.append(dim * dim)
            vals.append(np.median(dps_w) / np.median(orc_w))
    ax.plot(dims, vals, "o-", label=f"shift {sh}$\sigma$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("number of modes d")
ax.set_ylabel("dps W$_2^2$ / oracle floor")
ax.set_title("plug-in bias grows with dimension")
ax.legend(fontsize=9)

ax = axes[1]
ts = rows("t1_tsens.jsonl")
byT = defaultdict(list)
for r in ts:
    if r["sampler"] == "sap":
        byT[r["T"]].append(r["gamma_star"])
Ts = sorted(byT)
ax.plot(Ts, [np.median(byT[T]) for T in Ts], "o-", color="#b3261e")
ax.axhline(1.0, color="gray", lw=0.8, ls="--",
           label="faithful ($\gamma^*$=1)")
ax.set_xscale("log")
ax.set_xlabel("number of steps T")
ax.set_ylabel("effective tilt exponent $\gamma^*$")
ax.set_title("resampling shortcut: pathology compounds with depth")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()'''),
md(r"""## Misspecification, and the trap

Real deployments never have a perfect score. The bench contaminates the score
in a controlled way, a spectral tilt of the prior:

$$P_\varepsilon(k) = P(k)\, (k / k_p)^{\varepsilon},$$

**in words** a smooth re-weighting of small against large scales pivoting at
$k_p$, the analog of baryonic-feedback systematics in cosmology. The figure
below shows how each scheme transmits the contamination. The proper method
(twisted) passes it through one-to-one: built from the wrong model, it
faithfully samples the wrong posterior, and nothing in its own bookkeeping
can know. Plug-in guidance interacts with the contamination sign-dependently,
because the spectral tilt partially cancels or reinforces its own bias."""),
code(r'''ms = [r for r in rows("t1_misspec.jsonl") if r.get("dim") == 64]
core = [r for r in rows("t1_core.jsonl") if r.get("dim") == 64
        and r.get("eps", 0) == 0 and r.get("shift") == 1.0]
fig, ax = plt.subplots(figsize=(7, 4.2))
for s, color in (("dps", "#b3261e"), ("twisted", "#2e7d32"),
                 ("sap", "#b8860b")):
    eps_vals = sorted({r["eps"] for r in ms if r["sampler"] == s})
    pts = []
    for e in eps_vals:
        w = [r["w2"] for r in ms if r["sampler"] == s and r["eps"] == e
             and r.get("shift") == 1.0]
        if w:
            pts.append((e, np.median(w)))
    e0 = [r["w2"] for r in core if r["sampler"] == s]
    if e0:
        pts.append((0.0, np.median(e0)))
    pts.sort()
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                color=color, label=s)
ax.set_xlabel("score contamination $\\varepsilon$")
ax.set_ylabel("true W$_2^2$ error at shift 1$\sigma$")
ax.set_title("how each scheme transmits a wrong score")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()'''),
md(r"""**The trap.** Because dps and the contamination can partially cancel,
there is a special contamination $\varepsilon^*$ where the first-look
diagnostic reads perfectly clean while the pipeline is doubly broken. The
temperature diagnostic below is $\gamma^*$ again: the steering strength the
samples express, so 1.00 reads as "faithful"."""),
code(r'''es = rows("eps_star.jsonl")
by_eps = defaultdict(lambda: defaultdict(list))
for r in es:
    if r["sampler"] == "dps":
        by_eps[r["eps"]]["g"].append(r["gamma_star"])
        by_eps[r["eps"]]["w"].append(r["w2"])
eps_grid = sorted(by_eps)
if eps_grid:
    g = [np.median(by_eps[e]["g"]) for e in eps_grid]
    w = [np.median(by_eps[e]["w"]) for e in eps_grid]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(eps_grid, g, "o-")
    axes[0].axhline(1.0, color="crimson", lw=1, ls="--",
                    label="reads 'perfect'")
    axes[0].set_xlabel("score contamination $\\varepsilon$")
    axes[0].set_ylabel("temperature diagnostic $\gamma^*$")
    axes[0].legend()
    axes[1].plot(eps_grid, w, "o-")
    axes[1].set_xlabel("score contamination $\\varepsilon$")
    axes[1].set_ylabel("true W$_2^2$ error")
    for ax in axes:
        ax.axvline(-0.28, color="gray", lw=0.8)
    fig.suptitle("The compensation trap: at $\\varepsilon^*$ = -0.28 the "
                 "temperature reads exactly clean while the true error "
                 "stays ~6x floor")
    plt.tight_layout()
    plt.show()'''),
md(r"""This doubly-wrong-but-clean-looking configuration is carried into
notebook 03 as the hardest test for the diagnostics. Sample-based tests do
catch it, though they need several times their usual budget. Whether the
proposed runtime certificates catch it is one of the questions the trial
answers."""),
]

# ================================================================ 03
nb03 = [
md(r"""# 03 · The certificates on trial

Can a running sampler certify its own output without ground truth? Two
mechanisms have been proposed. This notebook tests both.

## Route one: journey accounting

A steered sampler knows both its own nudged step rules and the unsteered
ones, so it can price its own trajectory. With $p_0$ the unsteered step
probabilities, $p_s$ the steered ones, and $L$ the data fit of the final map,

$$w(\mathrm{trajectory}) \;=\; \prod_t
\frac{p_0(\mathrm{step}_t)}{p_s(\mathrm{step}_t)} \;\cdot\; L(x_0),$$

$$\mathrm{ESS} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}, \qquad
\widehat{\mathrm{KL}} = \log\langle w \rangle - \langle \log w \rangle .$$

**In words:** every step pays a price for how much the nudge distorted it,
and the product over the trajectory is an exact accounting device. If the
sampler were perfect the weights would be identical across trajectories, so
their spread is a certificate reading. ESS near the sample count means
healthy, ESS near 1 means the books have collapsed onto a single trajectory.
$\widehat{\mathrm{KL}}$ turns the spread into nats of path-space damage.
All of it computable at runtime, no ground truth anywhere."""),
code(PREAMBLE + r'''
kt = rows("cert_killtest.jsonl")
readable = [r for r in kt if r.get("score") == "exact"
            and r.get("kl_path_exact") and r["kl_path_exact"] < 50]
fig, ax = plt.subplots(figsize=(6.2, 4.6))
xs = [r["kl_path_exact"] for r in readable]
ys = [r["kl_path_hat"] for r in readable]
ax.plot([0, 50], [0, 50], color="gray", lw=0.8, ls="--",
        label="perfect meter")
ax.plot(xs, ys, "o", ms=4, alpha=0.6, color="#31688e")
ax.set_xlabel("true path-space damage (exact, nats)")
ax.set_ylabel("certificate reading (nats)")
ax.set_title("route one with an EXACT score: a tight instrument")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
r = np.array(ys) / np.array(xs)
print(f"reading/truth over the readable regime: median {np.median(r):.2f}, "
      f"IQR {np.quantile(r,0.25):.2f} to {np.quantile(r,0.75):.2f}")
print("(when damage reaches thousands of nats the estimator saturates to a")
print(" lower bound, which is still unmissable)")'''),
md(r"""**Reading the scatter.** Each point is one run with the exact score.
Its true path-space damage is on the x axis (computable on this bench only)
and the certificate's runtime reading on the y axis. The points hug the
diagonal. As an instrument on perfect ingredients, route one works.

## Route one dies on trained networks

Swap the exact score for a trained network, everything else identical. The
weights collapse: the network's small, jagged gradient errors accumulate
multiplicatively along hundreds of steps."""),
code(r'''cl = rows("cert_learned.jsonl")
learned = [r for r in cl if r.get("score") not in ("exact", "analytic")]
by = defaultdict(list)
for r in learned:
    by[(r.get("score"), r.get("shift"))].append(r["ess_res"])
fig, ax = plt.subplots(figsize=(7.5, 3.8))
labels, vals = [], []
for k in sorted(by, key=str):
    labels.append(f"{k[0]}, shift {k[1]}")
    vals.append(np.median(by[k]))
ax.bar(range(len(vals)), vals, color="#b3261e")
ax.axhline(256, color="#2e7d32", lw=1, ls="--", label="healthy (N=256)")
ax.set_ylabel("residual ESS (log scale)")
ax.set_yscale("log")
ax.set_ylim(0.5, 400)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
ax.set_title("route one with trained networks: ESS = 1 everywhere")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
print(f"median residual ESS across all learned-net configurations: "
      f"{np.median([r['ess_res'] for r in learned]):.2f} out of 256")
print("one sample carries all the weight. the reading is noise.")'''),
md(r"""Every bar sits at 1: across networks and steering strengths, a single
trajectory carries the entire ledger, so the reading measures nothing. This
was established through a pre-registered kill test with adversarially checked
escape routes (gentler steering, shorter runs, block-wise books, rankings
instead of absolute readings). None survives. Route one is dead in
deployment.

## Route two: slope checking (score-KSD)

The proposed alternative examines the destination instead of the journey.
The **score** of the target,

$$s(x) = \nabla_x \log \pi(x),$$

**in words** the local uphill direction of the target's probability at the
point $x$, is compared with the arrangement of your samples through a Stein
discrepancy:

$$\mathrm{score\text{-}KSD} = \frac{1}{N}\sqrt{\frac{1}{d}\sum_{i,j}
u_\pi(x_i, x_j)}\,,$$

where $u_\pi$ couples the scores at pairs of samples with a kernel, and a
mathematical identity (due to Stein) guarantees the sum cancels in
expectation exactly when the samples come from $\pi$. **In words:** zero-ish
for a perfect sampler, growing when the sample arrangement disagrees with the
target's slopes. This is the certificate proposed for diffusion inverse
solvers in the recent literature (arXiv:2602.04189). No reference
implementation exists, so this repository contains one, verified against
automatic differentiation, and adds a calibrated detection threshold, which
the paper does not specify.

Before the full-scale trial, the structural weakness can be seen with the
naked eye in two dimensions."""),
code(r'''from tilt_audit import ksd as ksdmod

dmu = np.array([2.5, 0.0])
w = 0.5
rng = np.random.default_rng(1)

def sample2(N, mode):
    signs = {"both": rng.choice([-1, 1], N, p=[1 - w, w]),
             "plus": np.ones(N, dtype=int)}[mode]
    return rng.standard_normal((N, 2)) + signs[:, None] * dmu[None, :]

def score2(X):
    la = -0.5 * np.sum((X - dmu) ** 2, axis=1) + np.log(w)
    lb = -0.5 * np.sum((X + dmu) ** 2, axis=1) + np.log(1 - w)
    ra = 1.0 / (1.0 + np.exp(lb - la))
    return (ra[:, None] * (-(X - dmu)) + (1 - ra)[:, None] * (-(X + dmu)))

Xb, Xp = sample2(256, "both"), sample2(256, "plus")
gx, gy = np.meshgrid(np.linspace(-5, 5, 22), np.linspace(-3.5, 3.5, 16))
G = np.stack([gx.ravel(), gy.ravel()], axis=1)
S = score2(G)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
for ax, X, title in ((axes[0], Xb, "honest sampler (both islands)"),
                     (axes[1], Xp, "broken sampler (one island missing)")):
    ax.quiver(G[:, 0], G[:, 1], S[:, 0], S[:, 1], color="#bbbbbb",
              width=0.0025, scale=60)
    ax.plot(X[:, 0], X[:, 1], ".", ms=3, color="#31688e")
    st = ksdmod.ksd_stats(X.astype(np.float64),
                          score2(X).astype(np.float64), "imq", 1.0, -0.5)
    ax.set_title(f"{title}, score-KSD reading: {st['score_ksd']:.3f}",
                 fontsize=10)
    ax.set_xlim(-5, 5)
plt.tight_layout()
plt.show()'''),
md(r"""**Reading the panels.** Gray arrows are the target's score field, the
slopes the certificate checks against. Blue dots are the samples. On the
right, the entire left island of probability is missing, half the posterior,
and the reading barely moves: every sample sits in a region where the local
slopes are exactly what the mixture score says they should be. The score
field around the occupied island does not know the other island is empty.
More samples cannot help, because they all land in the same place.

## The full-scale trial

The same instrument at field scale, on failure archives with exactly known
damage. Detection is at an empirically calibrated 5% false-alarm level, rank
against 60 perfect-sampler readings."""),
code(r'''ksd = rows("ksd_trial.jsonl")
power = [r for r in ksd if r.get("arm") == "power"
         and r.get("kernel") == "imq_paper"]
byp = defaultdict(list)
for r in power:
    byp[(r["config"], r["budget"])].append(r["detect"])
budgets = [64, 256, 1024, 4096]
fig, ax = plt.subplots(figsize=(7, 4.2))
for cfg, label, color in (
        ("dps", "biased dynamics", "#31688e"),
        ("sap", "improper resampling", "#b8860b"),
        ("dps_em03", "compensating errors", "#b3261e"),
        ("oracle_null", "perfect sampler (control)", "#7f7f7f")):
    ys = [np.mean(byp.get((cfg, b), [np.nan])) for b in budgets]
    ax.plot(budgets, ys, "o-", color=color, label=label,
            ls=":" if cfg == "oracle_null" else "-")
ax.set_xscale("log")
ax.axhline(0.05, color="gray", lw=0.7, ls="--")
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("sample budget N")
ax.set_ylabel("detection rate")
ax.set_title("verdict one: with the TRUE score, strong on dynamics errors")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
print("even the compensating-errors configuration (which reads clean on the")
print("temperature diagnostic and needs 4x budget for sample-based tests)")
print("is caught with certainty from 64 samples.")'''),
code(r'''mix = [r for r in ksd if r.get("arm") == "mixture"
       and r.get("kernel") == "imq_paper" and r.get("w") == 0.5]
con = rows("mixture_contrast.jsonl")
bym = defaultdict(list)
for r in mix:
    bym[(r["config"], r["budget"])].append(r["detect"])

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
kb = [64, 256, 1024, 4096, 16384]
for cfg, color, label in (
        ("mix_plus", "#b3261e", "score-KSD, one island missing"),
        ("mix_both", "#7f7f7f", "score-KSD, honest control")):
    xs, ys = [], []
    for b in kb:
        v = list(bym.get((cfg, b), []))
        v += [r["detected"] for r in con
              if r.get("test") == "ksd_imq_paper" and r.get("w") == 0.5
              and r["config"] == cfg and r["budget"] == b
              and r.get("detected") is not None]
        if v:
            xs.append(b)
            ys.append(np.mean(v))
    ax.plot(xs, ys, "o-", color=color, label=label,
            ls=":" if cfg == "mix_both" else "-")
# at N=16,384 the paired plus/both statistic ratio is 1.0001 to 1.0007
# (results/mixture_contrast.jsonl, tag exploratory_ckpt1_16k)
for test, color in (("pqmass", "#2e7d32"), ("tarp", "#31688e")):
    xs, ys = [], []
    for b in (256, 1024):
        v = [r["detected"] for r in con if r.get("test") == test
             and r.get("w") == 0.5 and r["config"] == "mix_plus"
             and r["budget"] == b]
        if v:
            xs.append(b)
            ys.append(np.mean(v))
    ax.plot(xs, ys, "s--", color=color, label=f"{test}, one island missing")
ax.set_xscale("log")
ax.axhline(0.05, color="gray", lw=0.7, ls="--")
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("sample budget N")
ax.set_ylabel("detection rate")
ax.set_title("verdict two: blind to a missing mode at ANY budget")
ax.legend(fontsize=8)

ax = axes[1]
ws = sorted({r["w"] for r in con})
for test, color, label in (("ksd_imq_paper", "#b3261e", "score-KSD"),
                           ("pqmass", "#2e7d32", "PQMass"),
                           ("tarp", "#31688e", "TARP")):
    xs, ys = [], []
    for w_ in ws:
        v = [r["detected"] for r in con if r.get("test") == test
             and r.get("w") == w_ and r["config"] == "mix_plus"
             and r["budget"] == 1024]
        if v:
            xs.append(1 - w_)
            ys.append(np.mean(v))
    if xs:
        ax.plot(xs, ys, "o-", color=color, label=label)
ax.set_xscale("log")
ax.set_xlabel("weight of the missing island (1-w)")
ax.set_ylabel("detection rate at N=1024")
ax.set_ylim(-0.05, 1.05)
ax.set_title("the weight ladder: where sample-side detection dies")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()'''),
md(r"""**Reading the left panel.** The red line is the certificate judging a
sampler that misses half the posterior: it never leaves its own false-alarm
floor, out to 16,384 samples. The green and blue points are ordinary
sample-space tests on the same sample sets, at full power. **Right panel:**
turning the missing island rarer. The sample-space tests keep catching it
down to islands holding a few percent of the probability, then die for budget
reasons. The certificate is flat at the floor throughout. The blindness is
structural, not statistical.

## Verdict three: deployment

In practice nobody has the true score. The paper's own recipe builds the
reference from the trained network. Two things happen: the baseline reading
on perfect samples roughly doubles (the network's score error dominates the
statistic), and against that inflated baseline the most-used biased sampler
disappears."""),
code(r'''dep = [r for r in ksd if r.get("arm") == "deployment"
       and r.get("kernel") == "imq_paper" and r.get("budget") == 1024]
byd = defaultdict(list)
for r in dep:
    byd[(r["net"], r["config"])].append(r["ratio_q95"])
print("reference score from a trained network (the paper's own recipe),")
print("reading relative to the calibrated null at N=1024:")
print()
print("  judged sampler        net A     net B     true damage")
for cfg, label, dmg in (("oracle_null", "perfect sampler", "none"),
                        ("dps", "plug-in guidance", "30x floor"),
                        ("dps_em03", "compensating errors", "23x floor"),
                        ("sap", "improper resampling", "159x floor")):
    a = np.median(byd.get(("s_clean", cfg), [np.nan]))
    m = np.median(byd.get(("s_mis_m03", cfg), [np.nan]))
    print(f"  {label:20s}  {a:5.2f}x    {m:5.2f}x    {dmg}")
print()
print("plug-in guidance reads BELOW the null with both networks: the")
print("deployed certificate certifies the textbook target as clean. only")
print("catastrophic damage still clears the network-noise floor. a scan of")
print("the recipe's noise-level knob (sigma = 0.1 / 0.3 / 0.6) shows one")
print("setting partially recovers detection, but nothing tells a")
print("practitioner in advance which setting that is.")'''),
md(r"""## The envelope

The full picture in one matrix, including the coverage tests of the earlier
battery and the budget-doubling check of notebook 04. Strong where designed,
blind where deployment needs it most. That pattern is the project's central
finding."""),
code(r'''from IPython.display import Image, display
p = ROOT / "figures" / "fig_envelope.png"
if p.exists():
    display(Image(str(p), width=950))'''),
]

# ================================================================ 04
nb04 = [
md(r"""# 04 · Gold standards and the nonlinear transfer

Everything so far lived on a bench with closed forms. Real inference problems
are not this polite, so the last act removes the politeness while keeping the
control. The prior stays the same Gaussian field $g$. The observation becomes
nonlinear:

$$y = A\,\kappa(g) + \text{noise}, \qquad
\kappa(g) = \frac{e^{\lambda g - \lambda^2/2} - 1}{\lambda}.$$

**In words:** we observe a smoothed version of an exponentially warped field,
the kind of lopsided (lognormal) field that nonlinear structure formation
produces. The dial $\lambda$ sets the warp strength, and the normalization
keeps the warped field on the same scale at every $\lambda$. Two properties
make this the right substrate. First, at $\lambda \to 0$ it collapses back
to the exact bench, so correctness is testable at the flip of a dial. Second,
the posterior over $g$ is genuinely non-Gaussian, which is what everything
must now be measured against."""),
code(PREAMBLE + r'''
from tilt_audit import lognormal
from tilt_audit.fields import grid_to_z, make_basis, make_pk, unpack
import jax
import jax.numpy as jnp

n = 64
basis = make_basis(n)
Pz = jnp.asarray(grid_to_z(make_pk(basis), basis))
lam = lognormal.default_lambda()
g = jnp.sqrt(Pz) * jax.random.normal(jax.random.PRNGKey(3), Pz.shape)
gmap = np.asarray(unpack(g, basis))
kmap = np.asarray(lognormal.kappa(unpack(g, basis), lam))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
axes[0].imshow(gmap, cmap="RdBu_r")
axes[0].set_title("the Gaussian field g")
axes[1].imshow(kmap, cmap="RdBu_r")
axes[1].set_title(f"the warped field, lambda={lam:.2f}")
for ax in axes[:2]:
    ax.axis("off")
axes[2].hist(gmap.ravel(), bins=60, alpha=0.6, density=True, label="g")
axes[2].hist(kmap.ravel(), bins=60, alpha=0.6, density=True,
             label="warped")
axes[2].set_title("pixel histograms: the warp in one look")
axes[2].legend(fontsize=9)
plt.tight_layout()
plt.show()
skew = float(((kmap - kmap.mean()) ** 3).mean() / kmap.std() ** 3)
print(f"default lambda = {lam:.4f}, chosen so the warped field's skewness "
      f"is 1 (measured here: {skew:.2f})")'''),
md(r"""**Reading the panels.** Same underlying field, before and after the
warp. The histogram shows what the warp does: a heavy bright tail and a
compressed dark side, the signature of nonlinear structure. Real matter maps
look like the orange histogram, not the blue one.

## Gold standards, and their gates

On this substrate no formula gives the posterior, which is exactly the
situation practitioners face. The standard lament says proper validation is
too expensive. We measured the lament. A NUTS (Hamiltonian MCMC) gold
standard on the whitened parameterization of this posterior takes about
**74 seconds per 64x64 configuration** on one GPU, and it is not trusted
until it passes three gates.

1. **The dial test.** At $\lambda = 10^{-4}$ the sampler must reproduce the
   closed-form Wiener answer of notebook 01, per mode. It does, and the
   residual mean offsets scale linearly in $\lambda$ over two decades, which
   separates real nonlinear physics from sampler error.
2. **An independent referee.** At small warp both densities are known, so
   exact linear-posterior draws can be importance-reweighted into the
   nonlinear posterior with no MCMC at all. NUTS means agree with this
   closed-form construction with a z-score spread of 0.97, textbook
   agreement.
3. **Seed independence.** Two independently seeded runs agree per mode and in
   distribution (a two-sample test below its own split-null threshold).

The whole apparatus repeats at 128x128 (16,384 dimensions, 119 seconds). The
claim "offline validation is unaffordable" did not survive contact with the
measurement."""),
code(r'''tra = [r for r in rows("transfer.jsonl") if r.get("n") == 64
       and r.get("space") != "kappa_z"
       and abs(r.get("lam", 0) - 0.3143) < 1e-3]
by = defaultdict(list)
for r in tra:
    by[(r["tilt"], r["sampler"])].append(r["mmd2"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
order = ["gold_floor", "dps", "dps_inflated", "remy5", "remy30", "remy100",
         "terminal_is"]
disp = {"remy5": "langevin K=5", "remy30": "langevin K=30",
        "remy100": "langevin K=100", "dps_inflated": "dps + inflation",
        "gold_floor": "gold floor", "terminal_is": "terminal reweight"}
for ax, tiltname in zip(axes, ("mid", "strong")):
    floor = abs(np.median(by.get((tiltname, "gold_floor"), [1e-12])))
    labels, vals = [], []
    for s in order:
        v = by.get((tiltname, s))
        if v:
            labels.append(disp.get(s, s))
            vals.append(max(np.median(v), floor / 10) / floor)
    ax.bar(range(len(vals)), vals, color="#31688e")
    ax.set_yscale("log")
    ax.axhline(1, color="k", lw=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_title(f"steering: {tiltname}")
axes[0].set_ylabel("MMD$^2$ to gold / floor")
fig.suptitle("Sampler damage vs MCMC gold standards on the nonlinear "
             "substrate")
plt.tight_layout()
plt.show()'''),
md(r"""**Reading the bars.** The distance is now a kernel two-sample distance
(MMD) to the gold-standard draws, floor-referenced exactly as before (the
floor is a held-out gold subsample of the same size). Plug-in guidance stays
badly off. The annealed-Langevin ladder is the one method that walks down to
the floor as its per-level correction budget K grows, and the next figure
shows that convergence directly."""),
code(r'''fig, ax = plt.subplots(figsize=(6.8, 4.2))
for tiltname, color in (("mid", "#31688e"), ("strong", "#b3261e")):
    Ks, med, lo, hi = [], [], [], []
    for K, name in ((5, "remy5"), (30, "remy30"), (100, "remy100")):
        v = by.get((tiltname, name), [])
        if v:
            Ks.append(K)
            med.append(np.median(v))
            lo.append(np.quantile(v, 0.25))
            hi.append(np.quantile(v, 0.75))
    ax.plot(Ks, med, "o-", color=color, label=f"steering {tiltname}")
    ax.fill_between(Ks, lo, hi, color=color, alpha=0.18)
fl = [abs(v) for v in by.get(("mid", "gold_floor"), [])]
if fl:
    ax.axhline(np.median(fl), color="gray", ls="--", lw=1,
               label="gold floor")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Langevin corrections per noise level, K")
ax.set_ylabel("MMD$^2$ to gold")
ax.set_title("buy compute, approach the truth: the K-convergence law")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()'''),
md(r"""## The decay law: which Gaussian-bench conclusions survive

One conclusion from the exact bench breaks here, and it breaks in the most
useful way possible: measurably, as a function of the realism dial. On the
Gaussian bench, adding the predicted-uncertainty term to plug-in guidance
(the missing term from notebook 02) makes it exact. On the warped substrate
that correction is only a linearization, and its value decays as the warp
grows. At skewness 1 it is not even reliable across observations: for some
observed skies it helps by 3.6x, for others it makes things worse."""),
code(r'''lam_rows = [r for r in rows("transfer.jsonl") if r.get("n") == 64
            and r.get("tilt") == "mid" and r.get("yseed") == 0
            and r.get("space") != "kappa_z"]
byl = defaultdict(lambda: defaultdict(list))
for r in lam_rows:
    byl[round(r["lam"], 3)][r["sampler"]].append(r["mmd2"])
pts = []
for lam_, skew in ((0.08, 0.25), (0.16, 0.5), (0.314, 1.0), (0.5, 2.0)):
    d = byl.get(lam_, {})
    if "dps" in d and "dps_inflated" in d:
        pts.append((skew, np.median(d["dps"]) / np.median(d["dps_inflated"])))

allr = [r for r in rows("transfer.jsonl") if r.get("n") == 64
        and r.get("tilt") == "mid" and r.get("space") != "kappa_z"
        and abs(r.get("lam", 0) - 0.3143) < 1e-3]
byy = defaultdict(lambda: defaultdict(list))
for r in allr:
    byy[r["yseed"]][r["sampler"]].append(r["mmd2"])
advs = sorted(np.median(d["dps"]) / np.median(d["dps_inflated"])
              for d in byy.values() if "dps" in d and "dps_inflated" in d)

fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color="#31688e",
        label="matched observation")
ax.plot([1.0] * len(advs), advs, "x", color="#b3261e", ms=7,
        label="8 observations at skewness 1")
ax.axhline(1.0, color="gray", lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("non-Gaussianity of the substrate (skewness)")
ax.set_ylabel("advantage of the covariance fix (1 = none)")
ax.set_title("the decay law: an exact correction becomes a coin flip")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()
print("matched-observation ladder: "
      + " / ".join(f"{p[1]:.1f}x" for p in pts))
print(f"across observations at skewness 1: median {np.median(advs):.2f}x, "
      f"range {advs[0]:.2f}x to {advs[-1]:.2f}x (below 1x = the fix hurt)")'''),
md(r"""**Reading the figure.** Blue: the same observed sky, warped
progressively harder. The correction's advantage falls two orders of
magnitude. Red crosses: at skewness 1, eight different observed skies. The
spread crosses 1, so whether the fix helps at all depends on the sky you
happen to have. The correction's exactness on the Gaussian bench was an
accident of that bench. The Langevin route's convergence, by contrast,
transfers untouched (previous figure), which is the practically important
asymmetry: refinement survives non-Gaussianity, clever closed-form
corrections do not.

## The last certificate standing, and its honest scope

One truth-free runtime check survived the whole audit in any form: run the
sampler at budget K and at 2K with fresh seeds, compare the two outputs with
a calibrated two-sample test, and treat agreement as convergence. Because
every run here is also compared to a gold standard, the bench can state
exactly what that check certifies."""),
code(r'''k2k = rows("k2k.jsonl")
byk = defaultdict(list)
for r in k2k:
    if r.get("arm") == "remy" and r.get("tilt") == "strong":
        byk[r["K"]].append((r["agree"], r["mmd2_to_gold"]))
Ks = sorted(byk)
fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True,
                         height_ratios=[1, 1.4])
ax = axes[0]
ax.plot(Ks, [np.mean([v[0] for v in byk[K]]) for K in Ks], "o-",
        color="#31688e")
ax.set_ylabel("doubling check\nagreement rate")
ax.set_ylim(-0.05, 1.05)
ax.set_title("annealed Langevin, strong steering: the check tracks truth")
ax = axes[1]
ax.plot(Ks, [np.median([v[1] for v in byk[K]]) for K in Ks], "o-",
        color="#b3261e")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel("budget K")
ax.set_ylabel("true distance\nto gold (MMD$^2$)")
plt.tight_layout()
plt.show()

byd = defaultdict(list)
for r in k2k:
    if r.get("arm") == "dps" and r.get("tilt") == "mid":
        byd[r["T"]].append((r["agree"], r["mmd2_to_gold"]))
print("the boundary of validity, measured:")
print()
print("  plug-in guidance (converges to the WRONG answer):")
for T in sorted(byd):
    ag = np.mean([v[0] for v in byd[T]])
    print(f"    T={T:4d}: doubling check agrees at rate {ag:.2f} "
          f"while the truth says badly wrong")
stuck = [r for r in k2k if r.get("arm") == "stuck"]
if stuck:
    print(f"  two runs stuck on the same island: agreement rate "
          f"{np.mean([r['agree'] for r in stuck]):.2f} "
          f"(half the posterior missing)")
nfe = rows("nfe2.jsonl")
if nfe:
    ags = np.mean([r["agree"] for r in nfe])
    m = np.median([r["mmd2_to_gold"] for r in nfe])
    fs = np.median([r["floor_scale"] for r in nfe])
    print(f"  deterministic-ODE (flow matching) sampler: agreement {ags:.2f}")
    print(f"    at EVERY step count while sitting {m/fs:.0f}x above the "
          f"gold floor")'''),
md(r"""**Reading the two panels.** Top: how often the truth-free doubling
check declares "converged". Bottom: the true distance to the gold standard.
For this sampler class the two move together, the check flags every
unconverged budget and passes the converged one. The printed lines below the
figure are the measured boundary of that validity: a sampler that converges
confidently to the wrong answer gets co-signed, two runs missing the same
island agree with each other, and for deterministic-ODE samplers the alarm
never fires at all. The honest scope is one-directional. **Disagreement
proves non-convergence, cheaply and with no ground truth. Agreement proves
nothing.**

## What a practitioner should actually do

Offline, build a matched synthetic bench for your problem, manufacture MCMC
gold standards on it (minutes, not weeks, as measured above), and run your
actual sampler against them with sample-space tests. Measure how your
conclusions decay as the bench's realism dial turns, the way the covariance
correction was measured above. At runtime, use the doubling check for its
alarm and never for its silence.

The full story, told slowly with every construction explained, is the blog
post at [andreastersenov.github.io/tilt-audit](https://andreastersenov.github.io/tilt-audit/).
The prediction ledger with every pre-registered expectation and its score is
`RESEARCH_LOG.md`."""),
]

nb(nb01, "01_the_bench.ipynb")
nb(nb02, "02_sampler_anatomy.ipynb")
nb(nb03, "03_certificates_on_trial.ipynb")
nb(nb04, "04_gold_standards_and_transfer.ipynb")
