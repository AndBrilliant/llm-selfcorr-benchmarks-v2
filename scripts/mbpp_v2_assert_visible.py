#!/usr/bin/env python3
"""One-assert-visible MBPP control (Qwen only).

Tests whether the high false-acceptance rates in the production ledger are an
artifact of hidden interface conventions. The generator and both evaluator arms
see the first reference assert in addition to the prose and entry-point name.
Everything else (tasks, strict verdict grammar, no output cap, raw-response
preservation, ground-truth tests) matches the production protocol.

The output ledger is independent of production and resume-safe.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.mbpp_v2_pilot as pilot


def first_assert(task: dict) -> str:
    for line in task["test"].splitlines():
        if line.lstrip().startswith("assert "):
            return line.strip()
    raise ValueError(f"no assert in {task['task_id']}")


def augmented(task: dict) -> str:
    return (task["prompt"] + "\n\nFor example, the solution must satisfy this "
            "reference assertion:\n" + first_assert(task))


def generation_prompt(task: dict) -> str:
    return augmented(task)


def selfcheck_prompt(task: dict, code: str) -> str:
    return (
        "You are evaluating one immutable Python candidate solution.\n\n"
        f"[PROBLEM]\n{augmented(task)}\n\n"
        f"[CANDIDATE]\n{code}\n\n"
        "Analyze whether the candidate satisfies the specification and tests. "
        "Do not modify, repair, rewrite, or provide an alternative solution. "
        "End with exactly one final line containing only one of:\n"
        "VERDICT: ACCEPT\nVERDICT: REJECT"
    )


def steelman_prompt(task: dict, code: str) -> str:
    return (
        "You are a neutral arbiter judging a debate between two expert Python programmers.\n\n"
        f"[PROBLEM]\n{augmented(task)}\n\n"
        f"[CANDIDATE]\n{code}\n\n"
        "Do the following in order:\n"
        "1. STEELMAN AGAINST: give the strongest case that the candidate is wrong.\n"
        "2. STEELMAN FOR: give the strongest case that the candidate is correct.\n"
        "3. Decide which case survives scrutiny. Judge only the candidate as given; "
        "do not provide a corrected version.\n"
        "End with exactly one final line containing only one of:\n"
        "VERDICT: ACCEPT\nVERDICT: REJECT"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/mbpp_v2/assert_visible_qwen")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    pilot.SCHEMA_VERSION = "mbpp-v2-assert-visible/1"
    tasks = pilot.load_task_map()
    done = pilot.completed_pairs(out)
    work = [(tid, task) for tid, task in sorted(
        tasks.items(), key=lambda kv: int(kv[0].split("/")[1]))
        if ("qwen38-flash", tid) not in done]
    if args.limit:
        work = work[:args.limit]
    print(f"assert-visible Qwen: {len(work)} pairs to run, {len(done)} already done", flush=True)

    def run(item):
        tid, task = item
        pilot.run_pair(task, tid, "qwen38-flash", out, 3)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, work))
    pilot.write_manifest(out)
    print(json.dumps({"mode": "assert_visible_qwen", "completed": len(work),
                      "skipped": len(done)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
