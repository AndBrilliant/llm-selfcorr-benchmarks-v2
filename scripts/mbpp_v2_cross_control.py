#!/usr/bin/env python3
"""Cross-evaluation matrix + neutral verification control (review-driven arms).

Cross mode (--mode cross): each evaluator family judges the frozen production
candidates of every OTHER generator family (self-evaluation already exists in
the production ledger). Distinguishes a model-specific self-evaluation
disadvantage from general evaluator permissiveness while holding candidate
difficulty fixed. 5 x 4 x 947 pairs, two arms each.

Control mode (--mode control): one neutral, non-adversarial detailed-
verification call per production pair, same strict verdict grammar, no
output-length cap. Separates adversarial structure from added deliberation.

Separate immutable ledgers; production data never modified.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.mbpp_v2_pilot as pilot
from scripts.mbpp_v2_pilot import (
    append_jsonl, call_model, selfcheck_prompt, steelman_prompt, strict_verdict,
    utc_now, write_manifest,
)

MODELS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
WRITE_LOCK = threading.Lock()

CONTROL_PROMPT = (
    "You are performing a careful neutral verification of one immutable Python "
    "candidate solution.\n\n"
    "[PROBLEM]\n{prompt}\n\n[CANDIDATE]\n{code}\n\n"
    "Verify the candidate against the specification step by step: check the "
    "logic, trace the behavior on the requirements, and confirm whether it "
    "satisfies the specification. Do not modify, repair, rewrite, or provide "
    "an alternative solution, and do not presume the candidate is correct or "
    "incorrect; verify it.\n"
    "End with exactly one final line containing only one of:\n"
    "VERDICT: ACCEPT\nVERDICT: REJECT"
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_cross(task: dict, evaluator: str, generator: str, code: str, out: Path, retries: int) -> None:
    rec = {"schema_version": pilot.SCHEMA_VERSION, "ts": utc_now(), "evaluator": evaluator,
           "generator": generator, "task_id": task["task_id"]}
    for arm, prompt_fn in (("selfcheck", selfcheck_prompt), ("steelman", steelman_prompt)):
        event = call_model(evaluator, prompt_fn(task, code), f"cross_{arm}", task["task_id"], out, retries)
        rec[f"{arm}_event_id"] = event["event_id"]
        rec[f"{arm}_verdict"] = strict_verdict(event.get("raw_response"))
    with WRITE_LOCK:
        append_jsonl(out / "cross_records.jsonl", rec)


def run_control(task: dict, model: str, code: str, out: Path, retries: int) -> None:
    prompt = CONTROL_PROMPT.format(prompt=task["prompt"], code=code)
    event = call_model(model, prompt, "control", task["task_id"], out, retries)
    rec = {"schema_version": pilot.SCHEMA_VERSION, "ts": utc_now(), "model": model,
           "task_id": task["task_id"], "control_event_id": event["event_id"],
           "control_verdict": strict_verdict(event.get("raw_response"))}
    with WRITE_LOCK:
        append_jsonl(out / "control_records.jsonl", rec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cross", "control"], required=True)
    ap.add_argument("--schema", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--production-ledger", default="data/mbpp_v2/production")
    ap.add_argument("--transport-retries", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    args.schema = args.schema or ("mbpp-v2-cross/1" if args.mode == "cross" else "mbpp-v2-control/1")
    args.out = args.out or ("data/mbpp_v2/cross_eval" if args.mode == "cross" else "data/mbpp_v2/control")
    pilot.SCHEMA_VERSION = args.schema
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    retries = args.transport_retries + 1

    tasks = {t["task_id"]: t for t in read_jsonl(ROOT / "data/MBPP.jsonl")}
    cands = {(c["model"], c["task_id"]): c for c in read_jsonl(ROOT / args.production_ledger / "candidate_records.jsonl")}

    done = {(r["evaluator"], r["generator"], r["task_id"]) for r in read_jsonl(out / "cross_records.jsonl")} if args.mode == "cross" \
        else {(r["model"], r["task_id"]) for r in read_jsonl(out / "control_records.jsonl")}

    work = []
    if args.mode == "cross":
        for ev in MODELS:
            for gen in MODELS:
                if gen == ev:
                    continue
                for (gm, tid), c in cands.items():
                    if gm != gen or (ev, gen, tid) in done:
                        continue
                    work.append((tasks[tid], ev, gen, c["code"]))
    else:
        for (m, tid), c in cands.items():
            if (m, tid) not in done:
                work.append((tasks[tid], m, None, c["code"]))

    if args.limit:
        work = work[:args.limit]
    pilot.log(f"{args.mode}: {len(work)} calls to make, {len(done)} already done")

    partitions = [work[i::args.workers] for i in range(args.workers)]
    partitions = [p for p in partitions if p]

    def worker(partition):
        for task, a, b, code in partition:
            try:
                if args.mode == "cross":
                    run_cross(task, a, b, code, out, retries)
                else:
                    run_control(task, a, code, out, retries)
            except Exception as exc:
                pilot.log(f"ERROR {args.mode} {a} {b} {task['task_id']}: {type(exc).__name__}: {exc}")
                with WRITE_LOCK:
                    append_jsonl(out / "runner_errors.jsonl", {
                        "ts": utc_now(), "args": [a, b, task["task_id"]],
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    with ThreadPoolExecutor(max_workers=len(partitions)) as pool:
        list(pool.map(worker, partitions))

    write_manifest(out)
    print(json.dumps({"mode": args.mode, "completed": len(work), "skipped": len(done)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
