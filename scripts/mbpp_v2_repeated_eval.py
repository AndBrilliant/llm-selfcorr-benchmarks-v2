#!/usr/bin/env python3
"""Tier 1(i): repeated independent evaluator calls per arm (rounds 2-3).

For every frozen candidate in the production ledger, make two additional
independent temperature-0 calls per arm (self-check and steelman), producing
k=3 total verdicts per arm per candidate including the production round.
This instantiates the repeated-evaluation premise of Lemma lem:repeated:
it measures round-to-round verdict stability and whether repeated
acceptances sharpen discrimination of T beyond a single verdict.

Events go to a SEPARATE immutable ledger (data/mbpp_v2/repeated_eval/);
the production ledger is never modified. No max_tokens in any request.
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

WRITE_LOCK = threading.Lock()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def done_pairs(out: Path) -> set[tuple[str, str]]:
    return {(r["model"], r["task_id"]) for r in read_jsonl(out / "repeated_records.jsonl")}


def run_pair_rounds(task: dict, model: str, out: Path, max_transport_attempts: int) -> None:
    key = (model, task["task_id"])
    candidate = CANDIDATES.get(key)
    if candidate is None:
        return
    code = candidate["code"]
    prompts = {"selfcheck": selfcheck_prompt(task, code), "steelman": steelman_prompt(task, code)}
    record = {"schema_version": pilot.SCHEMA_VERSION, "ts": utc_now(), "model": model,
              "task_id": task["task_id"], "production_candidate": True}
    for round_no in (2, 3):
        for arm, prompt in prompts.items():
            event = call_model(model, prompt, f"{arm}_r{round_no}", task["task_id"], out,
                               max_transport_attempts)
            verdict = strict_verdict(event.get("raw_response"))
            record[f"{arm}_r{round_no}_event_id"] = event["event_id"]
            record[f"{arm}_r{round_no}_verdict"] = verdict
            record[f"{arm}_r{round_no}_extraction"] = "parsed" if verdict else "malformed_or_error"
            record[f"{arm}_r{round_no}_finish_reason"] = event.get("finish_reason")
    with WRITE_LOCK:
        append_jsonl(out / "repeated_records.jsonl", record)


CANDIDATES: dict[tuple[str, str], dict] = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="mbpp-v2-repeated/1")
    ap.add_argument("--out", default="data/mbpp_v2/repeated_eval")
    ap.add_argument("--production-ledger", default="data/mbpp_v2/production")
    ap.add_argument("--transport-retries", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test cap on pairs")
    args = ap.parse_args()

    pilot.SCHEMA_VERSION = args.schema
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    tasks = read_jsonl(ROOT / "data/MBPP.jsonl")
    global CANDIDATES
    CANDIDATES = {(c["model"], c["task_id"]): c
                  for c in read_jsonl(ROOT / args.production_ledger / "candidate_records.jsonl")}

    done = done_pairs(out)
    work = [(t, t["task_id"], m) for t in tasks for m in
            ("gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna")
            if (m, t["task_id"]) in CANDIDATES and (m, t["task_id"]) not in done]
    if args.limit:
        work = work[:args.limit]
    pilot.log(f"repeated-eval: {len(work)} pairs to run, {len(done)} already done")

    partitions = [work[i::args.workers] for i in range(args.workers)]
    partitions = [p for p in partitions if p]

    def worker(partition):
        for task, _task_id, model in partition:
            try:
                run_pair_rounds(task, model, out, args.transport_retries + 1)
            except Exception as exc:
                pilot.log(f"ERROR pair {model} {task['task_id']}: {type(exc).__name__}: {exc}")
                with WRITE_LOCK:
                    append_jsonl(out / "runner_errors.jsonl", {
                        "ts": utc_now(), "model": model, "task_id": task["task_id"],
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    with ThreadPoolExecutor(max_workers=len(partitions)) as pool:
        list(pool.map(worker, partitions))

    write_manifest(out)
    print(json.dumps({"completed_pairs": len(work), "skipped_pairs": len(done)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
