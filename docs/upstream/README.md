# Upstream bug reports (drafted 2026-07-05, verified against current main)

Two sibling normalization bugs found by the arms-night null gates (an exact
posterior must pass its own test), verified present in current upstream code by
agent inspection with permalinks. Full drafts below are ready to file.

- tarp_issue.md — Ciela-Institute/tarp: norm=True normalizes by the TRUTHS'
  range; d-extensive null miscalibration (exact posterior: max|ecp-a|=0.20 at
  d=4096). No release since 0.1.1; bug byte-identical in main. Related: their
  PR #12 fixes the sibling (references) asymmetry; sbi's port fixed only the
  references half (sbi#1832/#1837) — cross-reference may propagate the fix.
- mira_issue.md — SammyS15/mira-score: same bug class; code contradicts its
  own docstring (promises pooled Z-score, does truth-only min-max); exact
  posterior flagged 65-80% of the time at q=4096. Zero issues on their tracker.
