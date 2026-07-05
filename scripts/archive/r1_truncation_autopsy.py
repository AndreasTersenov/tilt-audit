#!/usr/bin/env python
"""R1 twin truncation autopsy (E-20260702b follow-up).

Hypothesis under test: the all-inverted calibration on the R1 twin (iid AUROC
0.17, wrong rows at confidence 1.0 / n_distinct 1) is manufactured by
truncation: R1's reasoning exceeds the 2048-token completion budget, boxed-
answer extraction fails, and population_confidence sums every failure into a
single "<none>" bucket that becomes a high-confidence wrong modal answer.

Design: replicate the night's iid generation (raw problem as user message,
temperature 0.8, n=16) on the SAME problems at two budgets:
  A = 2048 tokens  (the night's bon-max-tokens default — the treatment)
  B = 12000 tokens (recovery arm; server now runs max-model-len 16384)
and measure, per sample: finish_reason, completion tokens, extraction success,
correctness; per population: modal answer, confidence, n_distinct.

Verdict criteria (pre-stated):
  ARTIFACT CONFIRMED if, at budget A, extraction failure is concentrated in
  truncated samples AND the "<none>" bucket is modal on most of the night's
  high-confidence-wrong problems AND budget B substantially removes both
  (truncation < ~20%, modal flips away from "<none>", accuracy of modal rises).
  Otherwise the inversion stands as a real (model-family) phenomenon.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import datasets
from openai import AsyncOpenAI

sys.path.insert(0, "<predecessor-project>/src")
from particle_reasoners.reproduction.grading import extract_boxed, grade
from particle_reasoners.verification.calibration import population_confidence

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
N_SAMPLES = 16
TEMP = 0.8  # harness default


async def gen_one(client, sem, problem, max_tokens):
    async with sem:
        r = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": problem}],
            max_tokens=max_tokens, temperature=TEMP)
    ch = r.choices[0]
    text = ch.message.content or ""
    return {
        "finish_reason": ch.finish_reason,
        "tokens": r.usage.completion_tokens,
        "answer": extract_boxed(text),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-problems", type=int, default=20)
    ap.add_argument("--budgets", default="2048,12000")
    ap.add_argument("--out", default="results/r1_autopsy.jsonl")
    args = ap.parse_args()

    # harness's exact problem set and order
    ds = datasets.load_dataset("HuggingFaceH4/MATH-500")["test"]
    ds = ds.add_column("orig_idx", list(range(len(ds))))
    ds = ds.filter(lambda ex: ex["level"] in {4, 5}).select(range(100))

    # night rows, seed 0: pick 16 high-conf-wrong + 4 right iid problems
    night = [json.loads(l) for l in open(
        "<predecessor-project>/results/tables/"
        "reliability_r1_twin_s0.jsonl") if l.strip()]
    iid = {r["problem_id"]: r for r in night if r["method"] == "iid-majority"}
    wrong = [p for p, r in iid.items()
             if r["modal_correct"] == 0 and r["confidence"] >= 0.9][:16]
    right = [p for p, r in iid.items() if r["modal_correct"] == 1][:4]
    targets = set(wrong + right)
    problems = [ex for ex in ds if f"math500_{ex['orig_idx']}" in targets]
    problems = problems[:args.n_problems]
    print(f"{len(problems)} problems ({len(wrong)} night-HCW, {len(right)} night-right)")

    client = AsyncOpenAI(base_url="http://0.0.0.0:8000/v1", api_key="x")
    sem = asyncio.Semaphore(48)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for budget in [int(b) for b in args.budgets.split(",")]:
        tasks = {}
        for ex in problems:
            pid = f"math500_{ex['orig_idx']}"
            tasks[pid] = [asyncio.create_task(
                gen_one(client, sem, ex["problem"], budget))
                for _ in range(N_SAMPLES)]
        for ex in problems:
            pid = f"math500_{ex['orig_idx']}"
            gold = ex["answer"]
            samples = await asyncio.gather(*tasks[pid])
            modal, conf, ent, ndist = population_confidence(
                [s["answer"] for s in samples], [0.0] * len(samples))
            n_trunc = sum(s["finish_reason"] == "length" for s in samples)
            n_fail = sum(s["answer"] is None for s in samples)
            fail_in_trunc = sum(s["answer"] is None for s in samples
                                if s["finish_reason"] == "length")
            row = {
                "budget": budget, "problem_id": pid,
                "night_modal_correct": iid[pid]["modal_correct"],
                "night_confidence": iid[pid]["confidence"],
                "trunc_rate": n_trunc / N_SAMPLES,
                "extract_fail_rate": n_fail / N_SAMPLES,
                "fail_given_trunc": (fail_in_trunc / n_trunc) if n_trunc else None,
                "fail_given_ok": ((n_fail - fail_in_trunc) / (N_SAMPLES - n_trunc))
                                 if n_trunc < N_SAMPLES else None,
                "modal_is_none": modal == "<none>",
                "modal_correct": int(grade(gold, None if modal == "<none>" else modal)),
                "confidence": conf, "n_distinct": ndist,
                "mean_tokens": sum(s["tokens"] for s in samples) / N_SAMPLES,
            }
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{budget}] {pid}: trunc={row['trunc_rate']:.2f} "
                  f"fail={row['extract_fail_rate']:.2f} modal_none={row['modal_is_none']} "
                  f"correct={row['modal_correct']} conf={conf:.2f} "
                  f"toks={row['mean_tokens']:.0f}", flush=True)

    print("\n=== POOLED ===")
    rows = [json.loads(l) for l in out.open() if l.strip()]
    import collections
    for budget in sorted({r["budget"] for r in rows}):
        rs = [r for r in rows if r["budget"] == budget]
        print(f"budget {budget}: trunc {sum(r['trunc_rate'] for r in rs)/len(rs):.2f}, "
              f"extract-fail {sum(r['extract_fail_rate'] for r in rs)/len(rs):.2f}, "
              f"modal=<none> on {sum(r['modal_is_none'] for r in rs)}/{len(rs)}, "
              f"modal-acc {sum(r['modal_correct'] for r in rs)/len(rs):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
