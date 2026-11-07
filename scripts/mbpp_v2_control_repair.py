#!/usr/bin/env python3
"""Repair control ledger: point 747 phantom records at real preserved error events.

Background: during the control run the OpenAI key was invalidated mid-run
(2026-11-06 ~07:16 UTC). For GPT-5.4, 747 tasks failed all 3 transport attempts
with HTTP 401. Those 3 attempts per task ARE preserved in events.jsonl (2,241
events), but call_model's exhausted-retry fallback returned a synthetic event_id
that was never persisted -- so control_records.jsonl referenced events that do
not exist. The verdict (null) and record visibility are correct and unchanged;
only control_event_id is repointed to the terminal persisted attempt (attempt 3)
for that (model, task).

The repair is logged to repair_log.jsonl (production-ledger precedent).
Idempotent: records whose event_id already exists in events.jsonl are untouched.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mbpp_v2/control"

recs = [json.loads(l) for l in (OUT / "control_records.jsonl").read_text().splitlines() if l.strip()]
events = [json.loads(l) for l in (OUT / "events.jsonl").read_text().splitlines() if l.strip()]
ev_ids = {e["event_id"] for e in events}

# terminal (highest transport_attempt) event per (model, task)
terminal = {}
for e in events:
    k = (e["model"], e["task_id"])
    if k not in terminal or e.get("transport_attempt", 0) >= terminal[k].get("transport_attempt", 0):
        terminal[k] = e

repaired = []
for r in recs:
    if r["control_event_id"] in ev_ids:
        continue
    key = (r["model"], r["task_id"])
    assert key in terminal, f"no persisted event for {key}"
    t = terminal[key]
    assert t.get("error") and "401" in str(t.get("http_status")), f"unexpected non-error terminal event for {key}"
    repaired.append({
        "repair": "repoint_event_id",
        "model": r["model"], "task_id": r["task_id"],
        "old_event_id": r["control_event_id"], "new_event_id": t["event_id"],
        "cause": "exhausted-retry fallback emitted an unpersisted synthetic event_id; "
                 "all transport attempts failed with HTTP 401 (key invalidated mid-run)",
        "verdict_unchanged": r["control_verdict"],
    })
    r["control_event_id"] = t["event_id"]

with (OUT / "control_records.jsonl").open("w") as fh:
    for r in recs:
        fh.write(json.dumps(r) + "\n")
with (OUT / "repair_log.jsonl").open("a") as fh:
    for r in repaired:
        fh.write(json.dumps(r) + "\n")

print(f"repaired {len(repaired)} records")
from collections import Counter
print(Counter(r["model"] for r in repaired))

# verify: every record id now resolves
ev_ids = {json.loads(l)["event_id"] for l in (OUT / "events.jsonl").read_text().splitlines() if l.strip()}
recs2 = [json.loads(l) for l in (OUT / "control_records.jsonl").read_text().splitlines() if l.strip()]
missing = [r for r in recs2 if r["control_event_id"] not in ev_ids]
print(f"records: {len(recs2)}, unresolved after repair: {len(missing)}")
