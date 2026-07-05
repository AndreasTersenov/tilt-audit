# Title: `norm=True` normalizes by the truths' empirical range, which miscalibrates the test in high dimensions — an exact posterior reads max|ecp−α| ≈ 0.2 at d = 4096

Thanks for tarp — we use it as one of three posterior-accuracy diagnostics in a
calibration-audit study, and it's been genuinely useful. While running null
gates ("does the test pass on an exactly correct posterior?") we hit a subtle
statistical trap in the `norm=True` path that we think is worth reporting,
with measurements and a fix that worked for us.

**The issue.** In `src/tarp/drp.py`
([L88–L93](https://github.com/Ciela-Institute/tarp/blob/e05241c9e78ab821738e79e189ccdb63ebc6bb97/src/tarp/drp.py#L88-L93)),
`norm=True` min–max normalizes both `samples` and `theta` using the **truths'**
empirical per-dimension range. Every truth is inside the unit box by
construction, but a fresh posterior sample falls outside the empirical range of
L truths in roughly `2/(L+1)` of dimensions — about `2d/L` dimensions per
sample in total. Each out-of-box coordinate inflates that sample's distance to
the (in-box, `U[0,1]^d`) reference points, while truth-to-reference distances
are never inflated. The asymmetry is per-dimension, so the resulting null
miscalibration grows with dimension d.

**Measured impact** (L=128 truths, S=256 samples per truth, samples and truths
both exact draws from the same posterior, so the test should pass):

| d | max\|ecp − α\| with `norm=True` |
|---|---|
| 64 | ≈ 0.05 |
| 2048 | ≈ 0.15 |
| 4096 | ≈ 0.20 |

i.e. at d ≈ 4096 a perfectly calibrated posterior fails the coverage test
badly. Direction and d-scaling reproduce across seeds.

**Minimal reproduction sketch:**

```python
import numpy as np
from tarp import get_tarp_coverage

rng = np.random.default_rng(0)
L, S, d = 128, 256, 4096
mu = rng.standard_normal((L, d))                       # per-observation posterior mean
theta = mu + rng.standard_normal((L, d))               # truths ~ exact posterior
samples = mu[None] + rng.standard_normal((S, L, d))    # exact posterior draws

ecp, alpha = get_tarp_coverage(samples, theta, norm=True, seed=0)
print(np.max(np.abs(ecp - alpha)))   # ~0.2, should be ~0
```

**Fix that worked for us.** Standardize symmetrically before calling tarp, with
statistics that the truths never enter — pooled **sample** mean/std per
dimension, mapped so ±4 sd lands in [0,1] — then call with `norm=False`:

```python
m, s = samples.mean((0, 1)), samples.std((0, 1))
samples_n = ((samples - m) / s + 4.0) / 8.0
theta_n   = ((theta   - m) / s + 4.0) / 8.0
ecp, alpha = get_tarp_coverage(samples_n, theta_n, norm=False, seed=0)
```

This restores the null (max|ecp−α| ≈ 0.04–0.08 at d=4096 in our runs, n=20
reps, empirically calibrated false-positive rate 0.00 at α=0.05) and keeps
power (our injected-failure configs are still detected at rate ~1.0). Any
symmetric transform would presumably do; the load-bearing property is that the
normalization box is not defined by the truths.

**Suggested change.** Either (a) compute the normalization from the pooled
samples (or samples ∪ truths) rather than truths alone, or (b) keep the current
behavior but document that `norm=True` is unsafe as d grows (the effect is
already ~0.05 at d=64, within the tolerance of typical visual checks). Possibly
relevant neighbors: #7 discussed the `norm=False`-with-unnormalized-data
footgun, and PR #12 fixes the analogous asymmetry for user-supplied references.
The sbi port shares this normalization scheme (`sbi/diagnostics/tarp.py`,
`z_score_theta`); their #1832/#1837 fixed the references half but the
truth-based box remains, so a resolution here may be worth propagating.

Happy to provide the full battery scripts/numbers or turn the fix into a PR if
you'd like. All code and measurements:
https://github.com/AndreasTersenov/tilt-audit
