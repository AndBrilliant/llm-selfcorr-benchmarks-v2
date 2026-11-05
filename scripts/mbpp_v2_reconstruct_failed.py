#!/usr/bin/env python3
"""Reconstruct derived records for pairs whose final selected attempt failed.

The events ledger is complete and immutable; a pair whose generation or
evaluator call hit provider-side finish_reason=length (the model's own output
ceiling, deterministic at temperature 0) has no usable verdict on that arm.
Per protocol that failure stays VISIBLE: the pair is restored as a derived
record with extraction='malformed_or_error' (or evaluator arms 'not_run' when
generation failed), counted in fixed denominators, never silently dropped and
never retried.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mbpp_v2_pilot import append_jsonl, strict_verdict


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def failed(e: dict) -> bool:
    return e.get("http_status") != 200 or e.get("error") is not None or e.get("finish_reason") == "length"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/mbpp_v2/production")
    parser.add_argument("--schema", default="mbpp-v2-production/1")
    args = parser.parse_args()
    out = ROOT / args.out

    tasks = [json.loads(l)["task_id"] for l in (ROOT / "data/MBPP.jsonl").read_text().splitlines() if l.strip()]
    models = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]

    events = read_jsonl(out / "events.jsonl")
    candidates = {(c["model"], c["task_id"]): c for c in read_jsonl(out / "candidate_records.jsonl")}
    present = {(r["model"], r["task_id"]) for r in read_jsonl(out / "derived_records.jsonl")}

    by_pair_stage: dict[tuple, list[dict]] = defaultdict(list)
    for e in events:
        by_pair_stage[(e["model"], e["task_id"], e["stage"])].append(e)

    restored = 0
    for task_id in tasks:
        for model in models:
            key = (model, task_id)
            if key in present:
                continue
            gen_events = by_pair_stage.get((model, task_id, "generation"), [])
            if not gen_events:
                print(f"WARN no generation events for {key}; skipping")
                continue
            candidate = candidates.get(key)
            if candidate is not None:
                gen_event = next(
                    (e for e in gen_events if e["event_id"] == candidate["generation_event_id"]),
                    gen_events[-1],
                )
            else:
                gen_event = gen_events[-1]

            derived = {
                "schema_version": args.schema,
                "ts": gen_event["ts"],
                "task_id": task_id,
                "model": model,
                "generation_event_id": gen_event["event_id"],
                "generation_error": gen_event.get("error"),
                "generation_http_status": gen_event.get("http_status"),
                "generation_finish_reason": gen_event.get("finish_reason"),
                "code_sha256": candidate["code_sha256"] if candidate else None,
                "T": candidate["T"] if candidate else None,
                "test_reason": candidate["test_reason"] if candidate else None,
                "reconstructed_from_failed_attempt": True,
            }

            if candidate is not None:
                for stage in ("selfcheck", "steelman"):
                    evs = by_pair_stage.get((model, task_id, stage), [])
                    if not evs:
                        # pair was re-run after repair removal; final run aborted
                        # before evaluators: keep arms not_run only if generation
                        # itself failed, else treat as malformed (no event = no verdict)
                        derived[f"{stage}_event_id"] = None
                        derived[f"{stage}_verdict"] = None
                        derived[f"{stage}_extraction"] = "malformed_or_error"
                        continue
                    event = evs[-1]
                    verdict = strict_verdict(event.get("raw_response"))
                    if verdict is not None and not failed(event):
                        status = "parsed"
                    elif verdict is not None and failed(event):
                        # a verdict parsed from a failed event must never be used
                        verdict = None
                        status = "malformed_or_error"
                    else:
                        status = "malformed_or_error"
                    derived[f"{stage}_event_id"] = event["event_id"]
                    derived[f"{stage}_verdict"] = verdict
                    derived[f"{stage}_extraction"] = status
                    derived["ts"] = max(derived["ts"], event["ts"])
            else:
                for stage in ("selfcheck", "steelman"):
                    derived[f"{stage}_event_id"] = None
                    derived[f"{stage}_verdict"] = None
                    derived[f"{stage}_extraction"] = "not_run"

            append_jsonl(out / "derived_records.jsonl", derived)
            restored += 1

    append_jsonl(out / "repair_log.jsonl", {
        "ts": datetime.now(timezone.utc).isoformat(),
        "reason": "restored pairs whose final selected attempt failed with provider-side length truncation as visible malformed records; no regeneration (deterministic at temperature 0)",
        "restored_derived_records": restored,
    })
    print(f"restored {restored} derived records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
