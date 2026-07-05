#!/usr/bin/env python
"""Results-aware refill generator for the 2026-07-05 constructive-diagnostics
night. Called by conductor.py when the queue runs low; appends the highest-
priority not-yet-queued jobs until the pending count reaches --target.

Priority (per the brainstorm exit 2026-07-05: "harden + wildcard, then
escalate"):

  TIER 1  perpetual sinks, always meaningful, never exhaust:
    1a  gold-standard library expansion (NUTS golds, ~1-2 min each) — bayes
        golds first (needed downstream), then the lambda decay-law ladder,
        extra observations, 128^2; then UNBOUNDED yseed densification so the
        generator can always return a job (the never-idle guarantee).
    1b  score/flow net ensemble (train_*, ~25-40 min each) — kills the
        "your net is a toy" objection; bounded list, emitted if the checkpoint
        is missing.
  TIER 2  new-science full grids, GATED on the matching pilot's result
    (adaptation). Only emitted once the pilot JSONL shows the escape/descent
    criterion. Populated live as the pilot scripts land; see is_promoted().
  TIER 3  escalation (joint-inference arena pilot) — only when Tier 1b is
    complete AND the gold library is deep, i.e. late-night surplus.

A job is (id, tier, gpu, est_min, cmd). Existence is checked two ways: a
queue sentinel (done/failed/pending id already in jobs.jsonl) and the job's
output artifact on disk (gold .json / checkpoint .pkl), so reruns never
duplicate work. Training jobs set TILT_AUDIT_X64=0 in-cmd (fp32); gold/audit
jobs keep the fp64 default.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "queue"
STATE = QUEUE / "gen_state.json"
GOLD = ROOT / "results" / "gold"
CKPT = ROOT / "checkpoints"
PY = ".venv/bin/python"                       # cwd is ROOT when the cmd runs
NOX64 = "TILT_AUDIT_X64=0 "                    # training opt-out prefix


# ----------------------------------------------------------------- state I/O

def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"densify_y": 8}                     # unbounded gold fallback cursor


def save_state(state):
    STATE.write_text(json.dumps(state))


def queued_ids():
    ids = set()
    path = QUEUE / "jobs.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def pending_count():
    jobs, ids = [], []
    path = QUEUE / "jobs.jsonl"
    if not path.exists():
        return 0
    for line in path.read_text().splitlines():
        if line.strip():
            ids.append(json.loads(line)["id"])
    n = 0
    for jid in ids:
        if (QUEUE / "done" / jid).exists():
            continue
        if (QUEUE / "failed" / jid).exists():
            continue
        if (QUEUE / "locks" / jid).exists():
            continue
        n += 1
    return n


def gold_exists(n, tilt, y, lam=None, seed=0):
    """True if a matching gold sidecar already sits in results/gold/."""
    if lam is None:
        pat = f"gold_n{n}_{tilt}_y{y}_*_s{seed}.json"
    else:
        pat = f"gold_n{n}_{tilt}_y{y}_lam{lam:.4g}_s{seed}.json"
    return any(GOLD.glob(pat))


# ----------------------------------------------------------------- job lists

def gold_job(n, tilt, y, lam=None, seed=0):
    lam_tag = "def" if lam is None else f"{lam:.4g}"
    jid = f"gold_n{n}_{tilt}_y{y}_lam{lam_tag}_s{seed}"
    lam_flag = "" if lam is None else f" --lam {lam}"
    seed_flag = "" if seed == 0 else f" --seed {seed} --tag tl3"
    est = {32: 1, 64: 2, 128: 3}.get(n, 2)
    return {"id": jid, "tier": "T1a-gold", "gpu": "any", "est_min": est,
            "cmd": (f"{PY} scripts/run_gold.py --n {n} --tilt {tilt} "
                    f"--yseed {y}{lam_flag}{seed_flag}")}


def gold_candidates(state):
    """Ordered gold library, highest value first, then unbounded densify."""
    out = []
    # bayes golds (needed by nfe2 + the joint-arena escalation)
    for n in (32, 64):
        for y in (0, 1, 2, 3):
            if not gold_exists(n, "bayes", y):
                out.append(gold_job(n, "bayes", y))
    # lambda decay-law ladder at n=64 (densifies the transfer chapter figure)
    for lam in (0.08, 0.16, 0.63, 1.26):
        for tilt in ("mid", "strong"):
            if not gold_exists(64, tilt, 0, lam=lam):
                out.append(gold_job(64, tilt, 0, lam=lam))
    # more observations (cross-observation spread on the flagship configs)
    for tilt in ("mid", "strong"):
        for y in (4, 5, 6, 7):
            if not gold_exists(64, tilt, y):
                out.append(gold_job(64, tilt, y))
    # 128^2 stretch
    for tilt in ("mid", "strong"):
        for y in (0, 1, 2, 3):
            if not gold_exists(128, tilt, y):
                out.append(gold_job(128, tilt, y))
    # UNBOUNDED fallback: never lets the generator return empty
    if not out:
        y = state["densify_y"]
        out.append(gold_job(64, "mid", y))
        state["densify_y"] = y + 1
    return out


NET_ENSEMBLE = [
    # (id, out-checkpoint, cmd-without-py-prefix)  — emitted if ckpt missing
    ("train_s_p01",  "s_mis_p01.pkl",
     "scripts/train_score.py --eps 0.1 --out checkpoints/s_mis_p01.pkl"),
    ("train_s_m01",  "s_mis_m01.pkl",
     "scripts/train_score.py --eps -0.1 --out checkpoints/s_mis_m01.pkl"),
    ("train_s_long", "s_clean_long.pkl",
     "scripts/train_score.py --eps 0.0 --steps 120000 "
     "--out checkpoints/s_clean_long.pkl"),
    ("train_ln_m03", "ln_mis_m03.pkl",
     "scripts/train_lognormal.py --eps -0.3 --out checkpoints/ln_mis_m03.pkl"),
    ("train_ln_long", "ln_clean_long.pkl",
     "scripts/train_lognormal.py --eps 0.0 --steps 120000 "
     "--out checkpoints/ln_clean_long.pkl"),
    ("train_fm_big", "fm_big.pkl",
     "scripts/train_fm.py --chs 64,128,256 --out checkpoints/fm_big.pkl"),
]


def net_candidates():
    out = []
    for jid, ckpt, cmd in NET_ENSEMBLE:
        if (CKPT / ckpt).exists():
            continue
        prefix = "" if cmd.startswith("scripts/train_fm") else NOX64
        out.append({"id": jid, "tier": "T1b-net", "gpu": "any", "est_min": 35,
                    "cmd": f"{prefix}{PY} {cmd}"})
    return out


# --------------------------------------------- TIER 2 (science, gated) hooks

def rows(name):
    p = ROOT / "results" / name
    if not p.exists():
        return []
    return [json.loads(x) for x in p.open() if x.strip()]


def is_promoted(pilot_file, criterion):
    """True if the pilot's rows meet the promotion criterion (adaptation)."""
    rr = [r for r in rows(pilot_file) if r.get("tag") == "pilot"]
    return bool(rr) and criterion(rr)


def science_candidates():
    """Full grids for the constructive diagnostics, each gated on its pilot.
    Empty until a pilot lands + passes; conductor/steering fills these live.
    Kept here so the campaign runs the new science even if the session drops
    after the pilots are in."""
    out = []
    # SURE envelope — promoted if the exact-denoiser home pilot descended.
    if is_promoted("sure.jsonl",
                   lambda rr: any(r.get("kl_descends") for r in rr)):
        for sub in ("gauss", "mixture", "lognormal"):
            for den in ("exact", "net"):
                jid = f"sure_{sub}_{den}"
                if jid not in queued_ids():
                    out.append({"id": jid, "tier": "T2-sure", "gpu": "any",
                                "est_min": 20,
                                "cmd": (f"{PY} scripts/run_sure.py --substrate "
                                        f"{sub} --denoiser {den} --n 64 "
                                        f"--tag grid")})
    # mode-prober grid — promoted if excursions crossed at <=8 sigma.
    if is_promoted("modeprobe.jsonl",
                   lambda rr: any(r.get("crossed") and r.get("sep", 99) <= 8
                                  for r in rr)):
        for n in (32, 64):
            jid = f"modeprobe_grid_n{n}"
            if jid not in queued_ids():
                out.append({"id": jid, "tier": "T2-mode", "gpu": "any",
                            "est_min": 20,
                            "cmd": (f"{PY} scripts/run_modeprobe.py --n {n} "
                                    f"--seps 2,3,4,6,8 --tag grid")})
    return out


# ------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=9,
                    help="top the queue up to this many pending jobs")
    args = ap.parse_args()

    state = load_state()
    have = queued_ids()
    need = args.target - pending_count()
    if need <= 0:
        print("[generator] queue deep enough, nothing to add")
        return

    appended = []
    # priority: science-when-promoted > up to 2 net trainings > gold sinks.
    buckets = (science_candidates()
               + net_candidates()[:2]
               + gold_candidates(state))
    with (QUEUE / "jobs.jsonl").open("a") as f:
        for job in buckets:
            if need <= 0:
                break
            if job["id"] in have:
                continue
            f.write(json.dumps(job) + "\n")
            have.add(job["id"])
            appended.append(job["id"])
            need -= 1
    # guarantee progress even if everything above was already queued:
    # emit one unbounded densify gold so the generator is never a no-op.
    if not appended:
        y = state["densify_y"]
        job = gold_job(64, "mid", y)
        state["densify_y"] = y + 1
        with (QUEUE / "jobs.jsonl").open("a") as f:
            f.write(json.dumps(job) + "\n")
        appended.append(job["id"])

    save_state(state)
    print(f"[generator] appended {appended}")


if __name__ == "__main__":
    main()
