# Draft Slack message to Sammy (mira-score) — NOT SENT

Hey Sammy! I've been stress-testing calibration diagnostics on an exact-posterior
test bench (running MIRA alongside TARP and PQMass) and found a small but real bug
in mira-score that I figured you'd want to know about directly.

The issue: with `norm=True`, the code normalizes truth and posterior by the
**truths'** per-dimension min–max (src/mira_score/mira.py, L62–68) — while the
docstring says pooled Z-score across truth and posterior. The truth-only box is
the problem: every truth lands inside [0,1]^q by construction, but a perfectly
calibrated posterior's samples fall outside it in ~2/(T+1) of dimensions each
(order statistics). Since the random centers are drawn inside the box, sample
distances get inflated asymmetrically relative to truth distances — which biases
the score *below* the 2/3 null, i.e. it reads as fake overconfidence. The effect
is per-dimension, so it grows with q.

Measured on an exactly calibrated posterior (T=128, S=64, q=4096, 6 seeds):
0.6611 ± 0.0009 vs the analytic null 0.6719 — small per test, but it grows with
dimension, and it gets ~2.5× worse if the user bootstraps truths with replacement
for error bars (duplicated truths shrink the effective box — common practice, so
worth a docs warning too).

The fix is just making the code do what the docstring already says: standardize
both arrays symmetrically with pooled statistics (we used posterior-sample
mean/std per dimension, mapped ±4 sd into [0,1], then `norm=False`). With that,
the exact posterior lands on (2N+3)/(3(N+1)) to ~3 decimals in all our tests and
power is unchanged.

I filed the details with a self-contained repro as issue #1 on the repo (heads
up: I'm adding a comment there correcting the magnitude in my original report —
first version overstated it by including our own bootstrap protocol). Happy to
send a PR if useful. And MIRA's analytic null is honestly what made this
findable at all — most diagnostics don't give you anything this testable.
