#!/usr/bin/env python
"""Utilization + job tally for HANDOFF_DAWN.md (plan section 5)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
H0 = "22:48"


def parse_util():
    text = (ROOT / "queue" / "util.log").read_text()
    blocks = re.split(r"=== (\d\d:\d\d)\n", text)[1:]
    samples = []  # (hhmm, [(idx, mem, util)])
    for stamp, body in zip(blocks[::2], blocks[1::2]):
        rows = []
        for line in body.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                rows.append((int(parts[0]), int(parts[1].split()[0]),
                             int(parts[2].split()[0])))
        samples.append((stamp, rows))
    return samples


def main():
    samples = parse_util()
    per_gpu = {i: [] for i in range(4)}
    for stamp, rows in samples:
        for idx, mem, util in rows:
            per_gpu[idx].append((stamp, mem, util))
    print(f"utilization samples: {len(samples)} (15-min cadence since {H0})")
    for i in (0, 1, 2):
        vals = per_gpu[i]
        if not vals:
            continue
        utils = [u for _, _, u in vals]
        mems = [m for _, m, u in vals]
        busy = sum(1 for u in utils if u >= 20) / len(utils)
        print(f"GPU {i}: mean util {sum(utils)/len(utils):.0f}%, "
              f"share of samples >=20% util: {busy:.0%}, "
              f"mean mem {sum(mems)/len(mems)/1024:.1f} GB")
        idle = [s for s, m, u in vals if u < 20]
        if idle:
            print(f"   idle(<20%) sample times: {' '.join(idle)}")
    done = sorted(p.name for p in (ROOT / "queue" / "done").iterdir())
    failed = sorted(p.name for p in (ROOT / "queue" / "failed").iterdir())
    print(f"\nqueue jobs done ({len(done)}): {' '.join(done)}")
    print(f"queue jobs failed ({len(failed)}): {' '.join(failed) or '-'}")
    results = sorted((ROOT / "results").glob("*.jsonl"))
    print("\nresult files:")
    for r in results:
        n = sum(1 for l in r.open() if l.strip())
        print(f"  {r.name}: {n} rows")


if __name__ == "__main__":
    main()
