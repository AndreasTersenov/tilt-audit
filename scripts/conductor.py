#!/usr/bin/env python
"""Overnight conductor: one worker per GPU, self-refilling, deadline-aware.

Adapted from scripts/archive/queue_worker.py (the proven backbone) for the
2026-07-05 constructive-diagnostics night. Changes from the archived worker:
  - ROOT is the repo root (this file lives in scripts/, not scripts/archive/).
  - NIGHT_LOG -> lab-notebook/NIGHT_LOG_2026-07-06.md (the private night log).
  - --until HH:MM (UTC) hard deadline: stop CLAIMING new jobs past it, let the
    running job drain, then exit. Touches queue/STOP so peers also drain.
  - solo mem fraction (default 0.9; override via XLA_MEM_FRACTION) since GPUs
    0/1/2 are ours tonight; still prealloc=false so small jobs stay small.
  - refill via scripts/night_generator.py, which tops the queue up to a target
    depth (not one job) so 3 workers never starve.

State is pure filesystem (crash-safe, resumable): queue/{locks,done,failed,logs}
sentinels + append-only job outputs. A job is (id, tier, gpu, est_min, cmd);
cmd runs with shell=True so it can set its own env (e.g. TILT_AUDIT_X64=0 for
training). Idempotent: rerunning the worker skips done/failed/running ids.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "queue"
NIGHT_LOG = ROOT / "lab-notebook" / "NIGHT_LOG_2026-07-06.md"
GENERATOR = ROOT / "scripts" / "night_generator.py"
MEM_FRACTION = os.environ.get("XLA_MEM_FRACTION", "0.9")
REFILL_BELOW = 3        # trigger the generator when global pending < this
TARGET_DEPTH = 9        # generator tops the queue up toward this (3x GPUs)


def night_log(msg):
    stamp = datetime.now(timezone.utc).strftime("%H:%M")
    NIGHT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with NIGHT_LOG.open("a") as f:
        f.write(f"- {stamp} {msg}\n")


def load_jobs():
    jobs = []
    path = QUEUE / "jobs.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def job_state(job_id):
    if (QUEUE / "done" / job_id).exists():
        return "done"
    if (QUEUE / "failed" / job_id).exists():
        return "failed"
    if (QUEUE / "locks" / job_id).exists():
        return "running"
    return "pending"


def claim(job_id):
    try:
        fd = os.open(QUEUE / "locks" / job_id,
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def run_job(job, gpu):
    job_id = job["id"]
    logf = QUEUE / "logs" / f"{job_id}.log"
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=str(gpu),
               XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_MEM_FRACTION=MEM_FRACTION)
    night_log(f"[JOB] {job_id} (tier {job.get('tier','?')}, GPU {gpu}) started: "
              f"{job['cmd'][:120]}")
    for attempt in (1, 2):
        t0 = time.time()
        with logf.open("a") as lf:
            lf.write(f"\n===== attempt {attempt} @ "
                     f"{datetime.now(timezone.utc).isoformat()} =====\n")
            lf.flush()
            rc = subprocess.call(job["cmd"], shell=True, stdout=lf, stderr=lf,
                                 cwd=ROOT, env=env)
        mins = (time.time() - t0) / 60
        if rc == 0:
            (QUEUE / "done" / job_id).touch()
            night_log(f"[JOB] {job_id} finished OK ({mins:.0f}m)")
            return
        if attempt == 1:
            night_log(f"[FAIL] {job_id} rc={rc} ({mins:.0f}m) — retrying once")
    (QUEUE / "failed" / job_id).touch()
    night_log(f"[FAIL] {job_id} failed twice — marked FAILED, continuing "
              f"(log: queue/logs/{job_id}.log)")


def past_deadline(until):
    if not until:
        return False
    now = datetime.now(timezone.utc).strftime("%H:%M")
    # string compare is wrong across midnight; compare as minutes with wrap.
    def mins(hhmm):
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)
    now_m, dl_m = mins(now), mins(until)
    # deadline is "later tonight"; treat a deadline < launch as next-day.
    # We stored the launch minute in QUEUE/_launch_min for the wrap check.
    launch_f = QUEUE / "_launch_min"
    launch_m = int(launch_f.read_text()) if launch_f.exists() else now_m
    # unwrap: add 1440 to any time earlier than launch (i.e. after midnight)
    def unwrap(m):
        return m + 1440 if m < launch_m else m
    return unwrap(now_m) >= unwrap(dl_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--generator", action="store_true",
                    help="refill via night_generator.py when depth < threshold")
    ap.add_argument("--until", default=None,
                    help="UTC HH:MM hard deadline; stop claiming past it")
    args = ap.parse_args()

    for sub in ("locks", "done", "failed", "logs"):
        (QUEUE / sub).mkdir(parents=True, exist_ok=True)
    night_log(f"[NOTE] conductor up on GPU {args.gpu} "
              f"(mem_frac {MEM_FRACTION}, deadline {args.until or 'none'})")

    while True:
        if (QUEUE / "STOP").exists():
            night_log(f"[NOTE] conductor GPU {args.gpu} stopping (STOP file)")
            return
        if past_deadline(args.until):
            (QUEUE / "STOP").touch()
            night_log(f"[NOTE] conductor GPU {args.gpu} hit deadline "
                      f"{args.until}Z — draining, STOP set for peers")
            return
        jobs = load_jobs()
        pending = [j for j in jobs if job_state(j["id"]) == "pending"
                   and str(j.get("gpu", "any")) in (str(args.gpu), "any")]
        ran = False
        for job in pending:
            if claim(job["id"]):
                run_job(job, args.gpu)
                ran = True
                break
        if not ran:
            n_pending = len([j for j in jobs if job_state(j["id"]) == "pending"])
            if args.generator and n_pending < REFILL_BELOW:
                try:
                    fd = os.open(QUEUE / "locks" / "_generator",
                                 os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    try:
                        subprocess.call(
                            [sys.executable, str(GENERATOR),
                             "--target", str(TARGET_DEPTH)], cwd=ROOT)
                    finally:
                        os.unlink(QUEUE / "locks" / "_generator")
                except FileExistsError:
                    pass  # another worker is generating
            time.sleep(15)


if __name__ == "__main__":
    main()
