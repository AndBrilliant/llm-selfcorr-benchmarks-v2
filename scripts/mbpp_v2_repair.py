#!/usr/bin/env python3
"""Repair pass for the fresh MBPP production ledger.

Two defects are repaired, with every action appended to repair_log.jsonl:

1. Records referencing failed API events (HTTP/transport error or provider
   truncation) are removed so the resume pass regenerates those pairs.
2. Duplicate (model, task_id) records — written by an orphaned runner process
   after its launcher was killed — are deduplicated, keeping the earliest
   record as canonical. Removed rows are preserved in duplicate_records.jsonl.

The events ledger is append-only and is never modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mbpp_v2_pilot import append_jsonl, log


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def event_failed(event: dict) -> bool:
    return (
        event.get("http_status") != 200
        or event.get("error") is not None
        or event.get("finish_reason") == "length"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/mbpp_v2/production")
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    out = ROOT / args.out

    events = read_jsonl(out / "events.jsonl")
    by_id = {e["event_id"]: e for e in events}

    stats = {
        "removed_failedref_candidate_records": 0,
        "removed_failedref_derived_records": 0,
        "duplicate_candidate_records": 0,
        "duplicate_derived_records": 0,
    }

    # --- pass 1: drop records referencing failed events ----------------------
    kept_candidates = []
    for row in read_jsonl(out / "candidate_records.jsonl"):
        gen = by_id.get(row.get("generation_event_id"))
        if gen is None or event_failed(gen):
            stats["removed_failedref_candidate_records"] += 1
        else:
            kept_candidates.append(row)

    kept_derived = []
    for row in read_jsonl(out / "derived_records.jsonl"):
        bad = False
        for key in ("generation_event_id", "selfcheck_event_id", "steelman_event_id"):
            eid = row.get(key)
            if eid is None:
                continue
            event = by_id.get(eid)
            if event is None or event_failed(event):
                bad = True
                break
        if bad:
            stats["removed_failedref_derived_records"] += 1
        else:
            kept_derived.append(row)

    # --- pass 2: dedupe by (model, task_id), earliest record wins ------------
    dupes = []
    canonical: dict[tuple[str, str], dict] = {}
    for row in kept_candidates:
        key = (row["model"], row["task_id"])
        if key in canonical:
            stats["duplicate_candidate_records"] += 1
            dupes.append({"record_type": "candidate", "record": row,
                          "canonical_ts": canonical[key]["ts"]})
        else:
            canonical[key] = row
    kept_candidates = list(canonical.values())

    canonical = {}
    for row in kept_derived:
        key = (row["model"], row["task_id"])
        if key in canonical:
            stats["duplicate_derived_records"] += 1
            dupes.append({"record_type": "derived", "record": row,
                          "canonical_ts": canonical[key]["ts"]})
        else:
            canonical[key] = row
    kept_derived = list(canonical.values())

    if dupes:
        with (out / "duplicate_records.jsonl").open("a", encoding="utf-8") as fh:
            for row in dupes:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    (out / "candidate_records.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in kept_candidates), encoding="utf-8")
    (out / "derived_records.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in kept_derived), encoding="utf-8")

    repair_record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "reason": args.reason,
        **stats,
        "removed_pairs": sorted({(r["model"], r["task_id"]) for r in kept_derived}),
        "kept_candidate_records": len(kept_candidates),
        "kept_derived_records": len(kept_derived),
        "events_ledger_untouched": True,
    }
    append_jsonl(out / "repair_log.jsonl", repair_record)
    log(json.dumps(repair_record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
