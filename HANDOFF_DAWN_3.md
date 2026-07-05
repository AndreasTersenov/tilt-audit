# HANDOFF — DAWN 3 (certifier trial + transfer night, 2026-07-04 → 05)

> STATUS: DRAFT SKELETON (being filled as lanes complete; final assembly at dawn).
> Night log: NIGHT_LOG_2026-07-05.md. Plan: docs/OVERNIGHT_2026-07-04_CERTIFIER_TRANSFER_NIGHT.md.
> All scoring below is PROPOSED — final scoring happens with Andreas.

## 1. Verdict table — P-20260704i–n (PROPOSED)

| P | Claim (short) | Proposed | One-line evidence |
|---|---|---|---|
| i | KSD detects standard failures incl. compensation | **HIT** | dps/sap/dps_em03 power 1.00 at ALL budgets (ratios 2.8–19× null); the ε-compensation config that fooled γ*/TARP/MIRA is loud |
| j | missed-mode blindness | **HIT (total)** | mix_plus (half the posterior missing, 12σ) reads 1.00× null at every budget through N=16,384 (detect ≈ α); PQMass flags at 1.00; weight/16k ladders sharpen |
| k | wrong-reference false-certification | **HIT (refined)** | twisted_em03 (true damage 17.2× floor) reads ≤1.0× null under matched-wrong ref — and across a BAND of refs (−0.3…−0.2); dps_em03 escapes (scheme bias visible through any ref); oracle_null false-alarms at 1.00 from ε_ref=−0.05 |
| l | gold standards manufacturable at 64² | **HIT** | Full gate set green: T-L1 (λ=1e-4 both res + IS cross-check + λ-scaling), T-L2 26/26 (R-hat ≤ 1.0006, ESS_min > 10k), T-L3 seed-independence; 74 s/config |
| m | DPS overconfidence transfers | **HIT w/ finding** | bp coverage ≤0.5 clause: loud (0.00 on high-k bands even at mid tilt); ordering clause: dps > dps_inflated holds, but dps_inflated ≈ remy100 BREAKS (linearized inflation buys little off-Gaussian) — [metric-agreement clause: TO FINALIZE from full grid] |
| n | Rémy K-convergence transfers | **HIT (pending full grid)** | K=5→30→100 monotone at every config so far; K=100 sits at the gold floor (mmd2 ~0 at 32² mid; bp_cov ≈ 0.68) |

## 2. Beyond the frozen grid (checkpoint zooms, all tagged exploratory)

- **ε-ladder**: false-certification is a band, not a point; false-alarm curve saturates
  at ε_ref = −0.05 (perfect samples flagged 1.00) — reference error dominates the
  reading. [Deployment-net columns: PENDING rerun]
- **Weight ladder {0.5→0.99} + 16k budget**: KSD flat at 1.00× throughout;
  [where PQMass/TARP detection dies: FILL from mixture_contrast.jsonl]
- **A-lognormal ranking inversion**: KSD misranks dps (1.07) vs remy30 (1.76) against
  a 14× true-damage gap the other way — variance-collapse damage is near-invisible;
  the paper's within-task-ranking use fails on the realistic substrate.
- **λ-ladder (0.16 / 0.5)**: [FILL — transfer-vs-non-Gaussianity curve]

## 3. The night's story in three sentences (draft)

Score-KSD with the true score is genuinely powerful against scheme bias — it catches
the compensation config every sample-based test missed — but it is structurally blind
to missing modes (1.00× null with half the posterior gone, at any budget), and with
the score a practitioner actually has, its reading is dominated by reference error in
both directions (false certification across a band of matched errors; false alarms
from 5% reference error on perfect samples). Meanwhile, MCMC gold standards on a
nonlinear-forward-model substrate are cheap to manufacture (74 s/config, all gates
green), and against them the Gaussian-bench findings transfer with one instructive
break: DPS overconfidence persists band-structured, Rémy's K-convergence is clean,
but the linearized covariance correction that was exact on the Gaussian bench buys
almost nothing off-Gaussian.

## 4. Incidents (all recovered; see NIGHT_LOG for detail)

- Typed-timestamp drift in the night log (lesson 6 re-paid; corrected in place,
  all entries after the correction line use date -u).
- transfer-32 fired before its golds existed (gate was on GPU availability, not
  data) — 0 rows lost; refired correctly.
- "CPU" mixture battery launched unpinned → JAX preallocated 30.7 GiB on GPU 0 →
  deployment + transferL OOM'd to zero rows; both requeued (deployment score now
  chunked). NEW LESSON: intended-CPU jobs get explicit CUDA_VISIBLE_DEVICES="".
- Two dict-collision crashes in the battery runner (kernel kw; w kw) — caught at
  first row each, fixed, no data loss.
- XLA constant-folding warnings on the nonlinear guidance transpose — benign.

## 5. Utilization

[FILL from queue/util3.log at dawn — note the 23:5x GPU-0/2 idle gap (~10 min)
during the gate-race incident and the planned lull while golds finished.]

## 6. Next actions (for the joint session)

1. Score P-20260704i–n together (table above is PROPOSED).
2. Decide the paper skeleton: the three-act structure (true-score power /
   mode-blindness / deployment fragility) + the transfer chapter now has all its
   measured numbers.
3. [FILL: deployment-net columns verdict once rerun lands]
4. Upstream bug reports (tarp, mira) still owed from the arms night.
5. Blog/explainer Part 9 (tonight's arc) if the paper route is taken.
