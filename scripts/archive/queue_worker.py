#!/usr/bin/env python
"""One worker per GPU. Claims jobs from queue/jobs.jsonl by atomic lock-file
creation, runs them with CUDA_VISIBLE_DEVICES pinned, retries once, appends
[JOB]/[FAIL] lines to NIGHT_LOG.md. When idle and the pending count is low,
invokes the T3/T4 generator to refill the queue.
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
NIGHT_LOG = ROOT / "NIGHT_LOG.md"


def night_log(msg):
    stamp = datetime.now(timezone.utc).strftime("%H:%M")
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
        fd = os.open(QUEUE / "locks" / job_id, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def run_job(job, gpu):
    job_id = job["id"]
    logf = QUEUE / "logs" / f"{job_id}.log"
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=str(gpu),
               # no-prealloc: Track A jobs are small and share GPU 1 with the
               # second B1 stream's PRM (18 GB); cap keeps us honest anyway
               XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_MEM_FRACTION="0.40")
    night_log(f"[JOB] {job_id} (tier {job.get('tier','?')}, GPU {gpu}) started: "
              f"{job['cmd'][:120]}")
    for attempt in (1, 2):
        t0 = time.time()
        with logf.open("a") as lf:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--generator", action="store_true",
                    help="refill queue via t34_generator.py when depth < 3")
    args = ap.parse_args()

    for sub in ("locks", "done", "failed", "logs"):
        (QUEUE / sub).mkdir(parents=True, exist_ok=True)
    night_log(f"[NOTE] queue worker up on GPU {args.gpu}")

    idle_since = None
    while True:
        if (QUEUE / "STOP").exists():
            night_log(f"[NOTE] queue worker GPU {args.gpu} stopping (STOP file)")
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
            if args.generator and n_pending < 3:
                try:
                    fd = os.open(QUEUE / "locks" / "_generator",
                                 os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                    subprocess.call([sys.executable,
                                     str(ROOT / "scripts" / "t34_generator.py")],
                                    cwd=ROOT)
                    os.unlink(QUEUE / "locks" / "_generator")
                except FileExistsError:
                    pass
            time.sleep(20)


if __name__ == "__main__":
    main()
