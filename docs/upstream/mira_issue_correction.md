# Draft correction comment for mira-score#1 (NOT POSTED — awaiting Andreas)

**Correction on the magnitude (the mechanism stands; our headline numbers
overstated it).**

After filing, we re-verified everything on pure iid Gaussians, independently of
our own benchmark harness. Honest accounting:

1. **The package-only effect is real but smaller than we quoted.** Running the
   reproduction above (T=128, S=64, q=4096, 6 seeds): `norm=True` gives
   **0.6611 ± 0.0009** vs the analytic null 0.6719 — a deficit of **0.011**,
   highly significant and growing with q (≈0.674 / 0.671 / 0.661 at
   q = 16 / 256 / 4096), and the symmetric-normalization fix restores 0.668.
   But it is ~4× smaller than the 0.63 / "flagged 65–80%" we originally quoted.

2. **Most of our quoted magnitude came from our own evaluation protocol
   interacting with the bug.** Our battery bootstrap-resampled the truths *with
   replacement* before calling `mira`. Duplicated truths do not extend the
   min–max box, so the box is effectively built from ~63% as many distinct
   truths, which widens the out-of-box effect: with the same iid data,
   truth-bootstrap-with-replacement moves the score from 0.6611 to
   **0.6443 ± 0.0021** (deficit 0.028). Worth knowing since bootstrapping
   truths for error bars is common practice — it *amplifies* this normalization
   asymmetry — but it is our protocol, not your package.

3. **Unchanged:** the truth-range min–max at L62–L68 contradicting the
   docstring's promised pooled Z-score; the direction and q-growth of the bias;
   and the suggested fix (make the code match the docstring), which restores
   the analytic null in all our tests.

Apologies for the imprecise magnitude in the original report — the corrected
numbers above are from the self-contained script in the issue body (plus the
one-line bootstrap variant), no external harness involved.
