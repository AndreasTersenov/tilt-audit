# Title: `norm=True` min–max normalizes by the truths' range (docstring says pooled Z-score) — biases the score below the analytic null; an exact posterior is flagged as miscalibrated 65–80% of the time in high dimensions

Thanks for releasing mira-score — we adopted MIRA as one of three diagnostics
in a calibration-audit study precisely because its analytic finite-N null,
score = (2N+3)/(3(N+1)) with fluctuation band ±√(1/18L), makes it unusually
testable. Running it through a null gate ("does the test pass on an exactly
correct posterior?") we found a subtle normalization asymmetry — the same trap
we independently hit in the `tarp` package (Ciela-Institute/tarp) — with
measurements and a fix that worked.

**The issue.** In `src/mira_score/mira.py`
([L62–L68](https://github.com/SammyS15/mira-score/blob/c57487198ac30711783b78ac2af6a76758544483/src/mira_score/mira.py#L62-L68)),
`norm=True` min–max normalizes truth and posterior using the **truths'**
per-dimension empirical min/max. Two consequences:

1. It contradicts the docstring, which promises "Z-score (global mean/std
   across truth and posterior)" — the documented (symmetric) behavior is
   actually the safe one.
2. Every truth is inside the unit box by construction, while a fresh posterior
   sample falls outside it in ≈ 2q/(T+1) dimensions. Out-of-box coordinates
   asymmetrically inflate posterior-sample distances to the in-box `U[0,1]^q`
   centers relative to truth distances, which biases the score away from the
   analytic null — the bias is per-dimension, so it grows with q.

**Measured impact** (T=128 truths, q=4096, exact posterior samples,
num_runs=64, 20 bootstrap reps): mean score **0.6302** at S=64 (analytic null
0.671875) and **0.6256** at S=256 (null 0.667969) — about 2 standard errors
low with band √(1/(18·128)) ≈ 0.021. Under a two-sided z-test at α=0.05, an
**exactly correct** posterior is flagged as miscalibrated in **80%** (S=64)
and **65%** (S=256) of reps.

**Minimal reproduction sketch:**

```python
import torch
from mira_score import mira

torch.manual_seed(0)
T, S, q = 128, 64, 4096
mu = torch.randn(T, q)                                   # per-observation posterior mean
truth = mu + torch.randn(T, q)                           # truths ~ exact posterior
post = (mu[:, None, :] + torch.randn(T, S, q))[None]     # (M=1, T, S, q), exact draws

mean, std = mira(truth, post, num_runs=64, norm=True)
N = S - 1
print(mean.item(), (2*N + 3) / (3*(N + 1)))   # ~0.63 vs 0.6719 — many SE apart
```

**Fix that worked for us.** Make the transform match the docstring —
symmetric, with statistics the truths never enter. We standardized by pooled
posterior-**sample** mean/std per dimension, mapped ±4 sd into [0,1], and
called `mira(..., norm=False)`:

```python
m, s = post.mean(dim=(0, 1, 2)), post.std(dim=(0, 1, 2))
post_n  = ((post  - m) / s + 4.0) / 8.0
truth_n = ((truth - m) / s + 4.0) / 8.0
mean, std = mira(truth_n, post_n, num_runs=64, norm=False)
```

With this, the exact posterior's mean score lands on the analytic null to ~4
decimal places (0.6716 vs 0.671875 at S=64; 0.6677 vs 0.667969 at S=256 in our
runs), the oracle false-positive rate drops to 0.00, and detection power on
our injected-failure configs is preserved (~1.0 on the failure family the
score is sensitive to).

**Suggested change.** Replace the truth-based min–max with the pooled
standardization the docstring already describes (or normalize by pooled
min/max over truth ∪ posterior). If the current behavior is intentional, a
docs note that `norm=True` shifts the null away from (2N+3)/(3(N+1))
increasingly with q would save users from silently invalid z-tests. Happy to
share our full null-battery numbers or open a PR. All code and measurements:
https://github.com/AndreasTersenov/tilt-audit
