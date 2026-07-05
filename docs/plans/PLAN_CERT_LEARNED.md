# PLAN — Stage 2: block-wise certificates on a LEARNED score

> Status: SIGNED OFF — P-20260704d–f STAMPED BY OWNER 2026-07-04 (adopted as proposed)
> (gates run first regardless). Follows docs/PLAN_CERT_KILLTEST.md; the question is its
> E-entry's next-step (i): does per-mode/block near-exactification survive the
> off-diagonal correlations of a real trained score network?

## 0. Decision this informs

If block-wise certificates stay healthy on a trained net → the tightening arc
(per-band instruments, "wavelet-band-wise" in the wild) is confirmed on realistic
score error and the wrapper design proceeds with block decomposition as a core
feature. If per-mode certificates degenerate under net coupling → the product is
joint-directional-only (weaker paper, wrapper still viable for ranking). Either way
the block-size resolution limit (largest block that stays certifiable at budget N)
is a new, quotable operating characteristic.

## 1. Object

Learned-pathway sampler (samplers_learned ancestral kernel, x0hat from a trained
UNet): the guided (dps) kernel and the unguided reference kernel share the SAME
per-step variance kv and differ only by the guidance displacement disp (clip
included), so the per-step per-mode log-RN is

    incr_k = -(disp_k^2 + 2 disp_k sqrt(kv) eps_k) / (2 kv)

— free to accumulate during sampling (no extra net evaluations). Terminal + r/beta
per mode. Certificate target: the LEARNED prior chain tilted by the reward (steering
error given the net — the honest scope). Blocks: single modes, the three band_masks
|k| bands, and the joint. Per-block ESS, KL-hat, khat.

Ground truth axes: (a) pathway-analytic control (x0hat = analytic score, same
ancestral kernel) is linear per mode → a chain_law variant gives its EXACT path/
endpoint KL (new: certificate.chain_law_ancestral); (b) empirical per-band damage of
each sampler vs the true target (fitted band moments vs (mu*, Sigma*), oracle band
floor alongside).

## 2. Grid (64^2, T=256 — the T2 convention; N=256; seeds 0-4; y=999)

- nets: s_clean (main), s_mis_m03 / s_mis_p03 (model-error scope columns),
  analytic (pathway control with exact chain law).
- modes: dps (biased) + unguided (plumbing anchor: certificate == r/beta).
- shifts: 0.5, 1, 2, 4.
GPU 1: analytic + s_clean. GPU 2: s_mis_m03 + s_mis_p03. ~160 rows, ~15-30 s/row.

## 3. Gates before the grid

- G-L1: unguided learned pathway: step-RN == 0 exactly, logw == r/beta (plumbing).
- G-L2: pathway-analytic dps at 64^2/1sigma: per-mode-summed certificate reading
  within [0.5, 1.2]x of its exact chain-law path KL (8 seeds median); DPI holds
  (exact path >= exact endpoint KL).

## 4. Predictions (PROPOSED, awaiting stamp)

- P-20260704d · conf 60%: on the learned (clean) net, per-mode certificates stay
  healthy — median per-mode ESS >= 100/256 at 1sigma — and the low-|k| band
  certificate ranks damage across shifts correctly (Spearman >= 0.8 vs band truth);
  the full-|k|-range band (thousands of modes) degenerates like the joint (its
  block KL exceeds log N).
- P-20260704e · conf 70%: the pathway-analytic control's per-mode-summed reading
  reproduces its exact chain-law path KL within 20% (median over seeds), i.e. the
  Part III near-exactification transfers to the ancestral kernel unchanged.
- P-20260704f · conf 60%: steering dominates net error in the certificate's view:
  clean-net dps readings are within 2x of analytic-pathway dps readings at every
  shift (the net changes the meter's value, not its meaning).
- Kill/reshape criterion: median per-mode ESS < 10 on the learned net at 1sigma =>
  the block-wise tightening arc dies for real nets; product reshapes to
  joint-directional-only.

## 5. Deliverables

results/cert_learned.jsonl; digest section appended to cert_digest.py (or a small
cert_learned_digest.py); verdict lines into RESEARCH_LOG; notebook Part III gets a
one-paragraph + one-panel update; push.
