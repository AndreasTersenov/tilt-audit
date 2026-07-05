#!/usr/bin/env python
"""Generate the guided-tour notebook suite (01..04) from the tracked results.

Each notebook is self-contained, runs top to bottom on CPU in minutes, and
reads only tracked files (results/*.jsonl, figures/). Regenerate with:
    python notebooks/build_notebooks.py && jupyter nbconvert --execute --inplace notebooks/0*.ipynb
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

# ---------------------------------------------------------------- 01
nb01 = [
md("""# 01 · The bench

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

This notebook builds the bench and demonstrates its exactness. The next three
walk through the sampler anatomy (02), the trial of the certificates (03), and
the nonlinear extension with MCMC gold standards (04)."""),
md("""## The construction

The prior is a Gaussian random field, a structured random texture with a
cosmology-flavored power spectrum. Written in its Fourier basis it is a list
of independent Gaussians, one per spatial frequency. The data enter through a
quadratic reward. With observation y, smoothing operator A, and noise level s,
the steered target is

$$\\sigma(x) \\propto p(x)\\, e^{r(x)/\\beta}, \\qquad r(x) = -\\|Ax - y\\|^2 / (2 s^2).$$

In words: take the prior probability of a map, multiply by how well that map
explains the data, with the steering strength set by beta. For this family the
tilted target is the classical Wiener posterior. Its mean, its variance, the
score, the optimal twist, and the exact distance of any Gaussian approximation
from it are all available in closed form, per Fourier mode, at any size. A
sampler's error can therefore be measured with error bars of zero."""),
code(PREAMBLE + '''
from tilt_audit import tilt
from tilt_audit.fields import (grid_to_z, make_basis, make_pk,
                               smoothing_operator, unpack, sample_prior_z)
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
    im = ax.imshow(np.asarray(field), cmap="RdBu_r")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.show()
print(f"dimensions: {int(Pz.shape[0])}, steering strength b = {b:.4g}")'''),
md("""## Exactness, demonstrated

The oracle sampler draws directly from the closed-form posterior. If the bench
is what we claim, the oracle's empirical moments must match the formulas to
within Monte Carlo noise, mode by mode. This same pattern, called a null gate
in this project, is applied to every instrument before it is trusted: a test
that does not pass on the exact answer is not allowed to judge anything."""),
code('''from tilt_audit import samplers

res = samplers.oracle(jax.random.PRNGKey(0), Pz, az, y, b, N=4096, T=1, tf=9.0)
z = np.asarray(res["z"])
zscores = (z.mean(0) - np.asarray(mu)) / np.sqrt(np.asarray(Sig) / z.shape[0])
vratio = z.var(0) / np.asarray(Sig)
print(f"per-mode mean z-scores: max |z| = {np.abs(zscores).max():.2f} "
      f"(expected < ~4.3 for {z.shape[1]} modes under pure noise)")
print(f"per-mode variance ratios: median = {np.median(vratio):.4f} "
      f"(expected 1.000 +- {np.sqrt(2/z.shape[0]):.3f})")'''),
md("""Both numbers sit at their theoretical noise floors. The bench is exact,
and every result in the next notebooks inherits that property: when a sampler
reads as 10 times worse than the floor, that is a measurement, not an estimate.

**Where the data live.** Every experiment appends rows to `results/*.jsonl`
with full configuration. Every figure in the repository regenerates from those
files. The prediction ledger (`RESEARCH_LOG.md`) holds the pre-registered
expectations for each experiment, frozen and pushed publicly before the runs,
and scored afterwards. Continue with notebook 02 for what the samplers
actually do."""),
]

# ---------------------------------------------------------------- 02
nb02 = [
md("""# 02 · Sampler anatomy

Six ways to steer a diffusion sampler, measured against exact targets. The
zoo, each with its one-line mechanism:

* **oracle**: draws from the exact posterior (the finite-N floor reference).
* **plug-in guidance (DPS-class)**: nudges each denoising step with the
  data gradient through a point estimate. Cheap, ubiquitous, no guarantee.
* **reward-as-potential SMC**: a particle population resampled every step on
  the reward. Improper by construction (the tilt compounds with depth).
* **twisted SMC**: the properly weighted particle method, with the optimal
  twist available in closed form on this bench.
* **terminal reweighting**: unguided samples, importance-reweighted once at
  the end.
* **inflated-noise annealed Langevin**: a ladder of noise levels with K
  Langevin corrections per level, the data pull deliberately weakened while
  the map is still noisy.

The measurement below is the squared Wasserstein distance to the exact
posterior, divided by the oracle floor, so 1.0 means indistinguishable from
perfect and 10 means ten times the unavoidable error."""),
code(PREAMBLE + '''
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
    ax.plot(shifts, ratios, "o-", label=s)
ax.axhline(1, color="gray", lw=0.8, ls="--")
ax.set_yscale("log")
ax.set_xlabel("steering strength (posterior shift, in prior sigmas)")
ax.set_ylabel("W2 error / oracle floor")
ax.set_title("Sampler error vs exact posterior, 64x64 fields")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()'''),
md("""Reading the figure. The properly weighted method (twisted) sits on the
floor, as theory demands on this conjugate bench, which makes it the control
that validates the harness rather than a finding. Plug-in guidance is
substantially and monotonically biased, growing with steering strength. The
resampling shortcut and terminal reweighting degrade faster. The plug-in bias
curve was additionally confirmed by an independent closed-form calculation
(a stiff-ODE integration of the guided dynamics, matching the measured grid
to a few percent), so the mechanism is understood, not just observed.

## Misspecification, and the trap

Real deployments never have a perfect score. The bench can contaminate the
score in a controlled way (a spectral tilt of the prior, the analog of
baryonic-feedback systematics in cosmology) and watch how each scheme reacts.
One interaction matters most. Plug-in guidance is biased on its own, and the
score contamination can partially cancel that bias in the summary statistics
a practitioner would check first."""),
code('''es = rows("eps_star.jsonl")
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
    axes[0].set_xlabel("score contamination eps")
    axes[0].set_ylabel("temperature diagnostic (gamma*)")
    axes[0].legend()
    axes[1].plot(eps_grid, w, "o-")
    axes[1].set_xlabel("score contamination eps")
    axes[1].set_ylabel("true W2 error")
    for ax in axes:
        ax.axvline(-0.28, color="gray", lw=0.8)
    fig.suptitle("The compensation trap: at eps* = -0.28 the temperature "
                 "reads exactly clean while the true error stays ~6x floor")
    plt.tight_layout()
    plt.show()'''),
md("""This is the configuration used throughout notebook 03 as the hardest
test for the diagnostics: a doubly wrong pipeline (biased sampler, wrong
score) whose errors cancel in the statistic a person would check first. The
temperature diagnostic is fooled exactly. Sample-based tests do catch this
configuration, but they need several times more samples than usual. Whether
the proposed runtime certificates catch it is one of the questions the next
notebook answers."""),
]

# ---------------------------------------------------------------- 03
nb03 = [
md("""# 03 · The certificates on trial

Can a running sampler certify its own output without ground truth? Two
mechanisms have been proposed. This notebook tests both against the bench.

**Route one, journey accounting.** A steered sampler knows both its own nudged
step rules and the unsteered ones, so it can compute importance weights over
its own trajectories while it runs. Perfectly flat weights mean a perfect
sampler. The spread of the weights is the reading.

**Route two, slope checking (score-KSD).** Compare the arrangement of your
samples against the target's score (the local uphill direction of its
probability) using a Stein discrepancy. Zero-ish for a perfect sampler. This
is the certificate proposed for diffusion inverse solvers in the recent
literature (arXiv:2602.04189). No reference implementation exists, so this
repository contains one, verified against automatic differentiation, and adds
a calibrated detection threshold, which the paper does not specify."""),
code(PREAMBLE + '''
kt = rows("cert_killtest.jsonl")
cl = rows("cert_learned.jsonl")
if kt and cl:
    exact = [r for r in kt if r.get("score") == "exact"
             and r.get("kl_path_exact") and r["kl_path_exact"] < 50]
    learned = [r for r in cl if r.get("score") not in ("exact", "analytic")]
    print("route one, journey accounting:")
    if exact:
        tight = np.median([r["kl_path_hat"] / r["kl_path_exact"]
                           for r in exact])
        print(f"  exact score, readable regime (<50 nats true damage): "
              f"reading / truth = {tight:.2f} (tight instrument)")
        print(f"  (when damage is thousands of nats the estimator saturates "
              f"to a lower bound, still unmissable)")
    if learned:
        ess = np.median([r["ess_res"] for r in learned])
        print(f"  trained network score: residual ESS = {ess:.1f} "
              f"(weights collapse onto one sample, reading is noise)")
    print()
    print("the network's jagged gradients destroy the bookkeeping at every")
    print("scale tested. route one is dead in deployment. details: the blog,")
    print("parts 4 to 8.")'''),
md("""## Route two under trial

Three verdicts, each measured. The battery: archives of sampler failures with
exactly known damage, detection at an empirically calibrated 5% false-alarm
level (rank against 60 perfect-sampler readings)."""),
code('''ksd = rows("ksd_trial.jsonl")
power = [r for r in ksd if r.get("arm") == "power"
         and r.get("kernel") == "imq_paper"]
by = defaultdict(list)
for r in power:
    by[(r["config"], r["budget"])].append(r["detect"])
print("verdict one, with the TRUE score it is strong (detection rate at N=64):")
for cfg in ("dps", "sap", "dps_em03", "oracle_null"):
    v = by.get((cfg, 64), [])
    if v:
        label = {"dps": "biased dynamics", "sap": "improper resampling",
                 "dps_em03": "compensating errors",
                 "oracle_null": "perfect sampler (control)"}[cfg]
        print(f"  {label:32s} {np.mean(v):.2f}")'''),
code('''mix = [r for r in ksd if r.get("arm") == "mixture"
       and r.get("kernel") == "imq_paper" and r.get("w") == 0.5]
con = rows("mixture_contrast.jsonl")
bym = defaultdict(list)
for r in mix:
    bym[(r["config"], r["budget"])].append(r["detect"])
print("verdict two, structurally blind to a missing mode.")
print("target: two islands of probability, 12 sigma apart, 50/50 weights.")
print("sampler: visits ONE island. half the posterior is missing.")
print()
print("  budget   score-KSD detects   PQMass detects")
for b in (256, 1024):
    k = np.mean(bym.get(("mix_plus", b), [np.nan]))
    p = np.mean([r["detected"] for r in con
                 if r.get("test") == "pqmass" and r.get("w") == 0.5
                 and r["config"] == "mix_plus" and r["budget"] == b] or [np.nan])
    print(f"  {b:6d}   {k:^17.2f}   {p:^14.2f}")
print()
print("the score is only ever evaluated where samples already sit, and no")
print("sample ever stands on the empty island. more samples do not help")
print("(the paired ratio stays at 1.000 out to 16,384 samples).")'''),
code('''dep = [r for r in ksd if r.get("arm") == "deployment"
       and r.get("kernel") == "imq_paper" and r.get("budget") == 1024]
byd = defaultdict(list)
for r in dep:
    byd[(r["net"], r["config"])].append(r["ratio_q95"])
print("verdict three, gullible with the score a practitioner actually has.")
print("reference score from a trained network (the paper's own recipe):")
print()
print("  judged sampler        reading vs null (net A / net B)   true damage")
rows_ = [("oracle_null", "perfect sampler", "none"),
         ("dps", "plug-in guidance", "30x floor"),
         ("sap", "improper resampling", "159x floor")]
for cfg, label, dmg in rows_:
    a = np.median(byd.get(("s_clean", cfg), [np.nan]))
    m = np.median(byd.get(("s_mis_m03", cfg), [np.nan]))
    print(f"  {label:20s}  {a:.2f}x / {m:.2f}x{'':18s}{dmg}")
print()
print("the most-used biased sampler reads BELOW the null with both networks.")
print("the certificate, deployed as its authors specify, certifies it clean.")'''),
md("""## The envelope

The full picture, including the coverage tests and the budget-doubling check
of notebook 04, is one matrix (regenerated below from the tracked figure).
Strong where designed, blind where it matters most in deployment. That pattern
is the project's central finding."""),
code('''from IPython.display import Image, display
p = ROOT / "figures" / "fig_envelope.png"
if p.exists():
    display(Image(str(p), width=950))'''),
]

# ---------------------------------------------------------------- 04
nb04 = [
md("""# 04 · Gold standards and the nonlinear transfer

Everything so far lived on a bench with closed forms. Real inference problems
are not this polite, so the last act removes the politeness while keeping the
control. The prior stays the same Gaussian field g. The observation becomes
nonlinear:

$$y = A\\,\\kappa(g) + \\text{noise}, \\qquad
\\kappa(g) = \\left(e^{\\lambda g - \\lambda^2/2} - 1\\right)/\\lambda .$$

In words: we observe a smoothed version of an exponentially warped field, the
kind of lopsided (lognormal) field that nonlinear structure formation
produces. The dial lambda sets the warp strength. At lambda equal to zero the
problem collapses back to the exact bench, and that limit is the built-in
correctness test for everything here.

**Gold standards.** On this substrate no formula gives the posterior, which is
exactly the situation practitioners face. The claim that proper validation is
too expensive was tested directly: a NUTS (Hamiltonian MCMC) gold standard
takes about 74 seconds per 64x64 configuration on one GPU, passes a
Gaussian-limit gate, an independent closed-form importance-sampling
cross-check, and a seed-independence gate, and the whole apparatus repeats at
128x128."""),
code(PREAMBLE + '''
tra = [r for r in rows("transfer.jsonl") if r.get("n") == 64
       and r.get("space") != "kappa_z" and abs(r.get("lam", 0) - 0.3143) < 1e-3]
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
    ax.set_title(f"tilt: {tiltname}")
axes[0].set_ylabel("MMD$^2$ to gold / floor")
fig.suptitle("Sampler damage vs MCMC gold standards, nonlinear substrate")
plt.tight_layout()
plt.show()'''),
md("""Two findings transfer, one breaks informatively. Plug-in guidance stays
overconfident (its fine-scale error bars cover the truth zero percent of the
time at strong steering). Annealed-Langevin refinement keeps buying accuracy
smoothly with compute, all the way to the gold standard. But the cheap
covariance correction that was mathematically exact on the Gaussian bench
decays to worthless as the warp grows:"""),
code('''lam_rows = [r for r in rows("transfer.jsonl") if r.get("n") == 64
            and r.get("tilt") == "mid" and r.get("yseed") == 0
            and r.get("space") != "kappa_z"]
byl = defaultdict(lambda: defaultdict(list))
for r in lam_rows:
    byl[round(r["lam"], 3)][r["sampler"]].append(r["mmd2"])
print("advantage of the inflated-covariance fix over plain plug-in guidance")
print("(matched observation, yseed 0):")
print()
print("  warp (lam)   skewness   advantage")
for lam, skew in ((0.08, 0.25), (0.16, 0.5), (0.314, 1.0), (0.5, 2.0)):
    d = byl.get(lam, {})
    if "dps" in d and "dps_inflated" in d:
        adv = np.median(d["dps"]) / np.median(d["dps_inflated"])
        print(f"  {lam:8.3f}   {skew:8.1f}   {adv:6.1f}x")
print()
allr = [r for r in rows("transfer.jsonl") if r.get("n") == 64
        and r.get("tilt") == "mid" and r.get("space") != "kappa_z"
        and abs(r.get("lam", 0) - 0.3143) < 1e-3]
byy = defaultdict(lambda: defaultdict(list))
for r in allr:
    byy[r["yseed"]][r["sampler"]].append(r["mmd2"])
advs = sorted(np.median(d["dps"]) / np.median(d["dps_inflated"])
              for d in byy.values() if "dps" in d and "dps_inflated" in d)
print(f"and at skewness 1 the advantage is observation-dependent: across "
      f"{len(advs)} observations \\nthe median is "
      f"{np.median(advs):.2f}x with range {advs[0]:.2f}x to {advs[-1]:.2f}x "
      f"(below 1x = the fix hurt).")
print("the correction's exactness was a Gaussian accident.")'''),
md("""## The last certificate standing, and its honest scope

One truth-free runtime check survived the audit in any form: run the sampler
at budget K and at 2K with fresh seeds, compare the two outputs with a
calibrated two-sample test, and treat agreement as convergence. The bench can
measure exactly what that certifies, because every run also gets compared to
the gold standard."""),
code('''k2k = rows("k2k.jsonl")
byk = defaultdict(list)
for r in k2k:
    if r.get("arm") in ("remy", "dps"):
        key = (r["arm"], r["tilt"], r.get("K", r.get("T")))
        byk[key].append((r["agree"], r["mmd2_to_gold"]))
print("budget-doubling check vs the truth (strong tilt):")
print()
print("  sampler    budget   check says     truth says")
for (arm, tiltname, K), v in sorted(byk.items(), key=str):
    if tiltname != "strong":
        continue
    ag = np.mean([x[0] for x in v])
    m = np.median([x[1] for x in v])
    verdict = "converged" if ag >= 0.5 else "NOT converged"
    truth = "at gold floor" if m < 5e-3 else f"{m:.1e} from gold"
    print(f"  {arm:8s}   {K:6d}   {verdict:13s}  {truth}")
print()
nfe = rows("nfe2.jsonl")
if nfe:
    ags = np.mean([r["agree"] for r in nfe])
    m = np.median([r["mmd2_to_gold"] for r in nfe])
    fs = np.median([r["floor_scale"] for r in nfe])
    print(f"and for a deterministic-ODE (flow matching) sampler: agreement "
          f"{ags:.2f}\\nat EVERY step count while sitting {m/fs:.0f}x above "
          f"the gold floor.")'''),
md("""The pattern: for the annealed sampler at strong steering the check is
textbook, it flags every unconverged budget and passes the converged one. But
plug-in guidance converges confidently to the wrong answer and the check
co-signs it, two runs stuck on the same island agree, and for
deterministic-ODE samplers the alarm never fires at all. The honest scope is
one-directional. Disagreement proves non-convergence cheaply and with no
ground truth. Agreement proves nothing.

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
