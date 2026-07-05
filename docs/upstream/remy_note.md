# Draft note to Benjamin Rémy (NOT SENT — Andreas's review and call)

Subject: JADE — a possible normalization subtlety in the joint MIRA score (+ your code link 404s)

Hi Benjamin,

Congratulations on JADE — I read it the day it appeared. The joint field +
cosmology construction is elegant, and it was oddly validating for me to see
your test bench: I've been running experiments on almost exactly the same
substrate (shifted-lognormal fields with NUTS gold standards on the
differentiable forward model) for a project auditing sampler diagnostics.

Two small things from that project that seem directly useful to you:

1. Your footnote's code link (github.com/b-remy/jade) currently returns a 404 —
   probably just a private repo waiting to be flipped public.

2. Last week we found a normalization trap in MIRA-style scores in high
   dimensions, and your joint number sits exactly on its signature. Short
   version: if truth/posterior are normalized using the *truths'* empirical
   min–max range (as the `mira-score` reference package does with `norm=True` —
   I filed the details with a fix at github.com/SammyS15/mira-score/issues/1),
   every fresh posterior sample lands outside the truths' box in ~2q/L
   dimensions, which asymmetrically inflates sample distances and biases the
   score *below* the null. The bias grows with dimension: on an exactly
   calibrated posterior at q ≈ 4096 we measure ~0.630 against the finite-N null
   (2N+3)/(3(N+1)) ≈ 0.667.

   What made me write: your joint score (0.635 ± 0.017, q ≈ 82k) is within a
   fraction of a σ of that signature, while your 6-dimensional marginal (0.659)
   is clean — which is exactly the dimensional fingerprint this effect
   produces. The paper doesn't say how the mixed-scale joint space was
   normalized, so I can't tell from outside. The two-line check: standardize
   by pooled posterior-sample mean/std per dimension (truths never entering
   the transform) and recompute. If the number moves up toward 0.667, JADE is
   *better* calibrated than the paper currently claims — the "mildly
   overconfident" reading would be an artifact of the normalization, not a
   property of your posterior.

No action needed on my side either way — I just figured you'd want to know
before the referee does. Happy to share our null-test batteries if useful
(everything is public: github.com/AndreasTersenov/tilt-audit).

Χαιρετίσματα,
Andreas
