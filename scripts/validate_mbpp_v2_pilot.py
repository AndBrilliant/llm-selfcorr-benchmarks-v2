#!/usr/bin/env python3
"""Independent validation for the fresh MBPP pilot ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mbpp_v2_pilot import extract_code, run_task_tests, strict_verdict


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/mbpp_v2/pilot")
    args = parser.parse_args()
    out = ROOT / args.out

    events = read_jsonl(out / "events.jsonl")
    candidates = read_jsonl(out / "candidate_records.jsonl")
    derived = read_jsonl(out / "derived_records.jsonl")
    task_map = {
        row["task_id"]: row
        for row in read_jsonl(ROOT / "data/MBPP.jsonl")
    }

    assert events, "no events"
    assert derived, "no derived records"

    event_by_id = {row["event_id"]: row for row in events}
    assert len(event_by_id) == len(events), "duplicate event IDs"

    for name, rows in (("candidate", candidates), ("derived", derived)):
        seen = set()
        for row in rows:
            key = (row.get("model"), row.get("task_id"))
            assert key not in seen, f"duplicate {name} record for {key}"
            seen.add(key)

    # Candidate records must reference a clean, reproducible generation event
    # (they carry extracted code and test results). Derived records may
    # reference failed events ONLY as visible malformed records.
    candidate_gen_ids = {row["generation_event_id"] for row in candidates}

    success_events = 0
    malformed_events = 0
    length_events = 0
    for event in events:
        if event.get("http_status") != 200 or event.get("error"):
            malformed_events += 1
            continue
        success_events += 1
        raw = event.get("raw_response")
        assert isinstance(raw, str), f"{event['event_id']}: missing raw response"
        assert raw, f"{event['event_id']}: empty raw response"
        assert event.get("raw_response_sha256") == sha256_text(raw), (
            f"{event['event_id']}: raw response hash mismatch"
        )
        if event.get("finish_reason") == "length":
            length_events += 1

    for event_id in candidate_gen_ids:
        event = event_by_id[event_id]
        assert event.get("http_status") == 200 and not event.get("error"), (
            f"{event_id}: candidate references a failed API attempt"
        )
        assert event.get("finish_reason") != "length", (
            f"{event_id}: candidate references a provider-truncated event"
        )

    candidate_keys = set()
    for candidate in candidates:
        key = (candidate["model"], candidate["task_id"])
        assert key not in candidate_keys, f"duplicate candidate record: {key}"
        candidate_keys.add(key)
        generation = event_by_id[candidate["generation_event_id"]]
        recomputed_code = extract_code(generation.get("raw_response"), task_map[candidate["task_id"]])
        assert recomputed_code == candidate["code"], (
            f"candidate code does not reproduce from raw generation response: {key}"
        )
        passed, reason = run_task_tests(candidate["code"], task_map[candidate["task_id"]])
        assert int(passed) == int(candidate["T"]), (
            f"candidate T does not reproduce: {key}: {reason}"
        )
        # test_reason embeds random tempfile paths from the sandboxed run
        # (possibly mid-token at the truncation boundary); normalize the
        # whole /tmp/tmp* fragment before comparing
        normalize = lambda s: re.sub(r"/tmp/tmp\w*", "/tmp/TMPFILE", s)
        assert normalize(reason) == normalize(candidate["test_reason"]), (
            f"candidate test reason does not reproduce: {key}"
        )

    for row in derived:
        key = (row["model"], row["task_id"])
        generation = event_by_id[row["generation_event_id"]]
        assert generation["stage"] == "generation", f"bad generation ref: {key}"
        if row["code_sha256"] is not None:
            assert key in candidate_keys, f"missing candidate record: {key}"
        for stage in ("selfcheck", "steelman"):
            event_id = row[f"{stage}_event_id"]
            status = row[f"{stage}_extraction"]
            if status == "not_run":
                # arm never attempted: no event may be referenced, no verdict
                assert event_id is None, f"{stage} not_run but references event: {key}"
                assert row[f"{stage}_verdict"] is None, (
                    f"{stage} not_run but carries a verdict: {key}"
                )
                continue
            event = event_by_id[event_id]
            assert event["stage"] == stage, f"bad {stage} ref: {key}"
            verdict = strict_verdict(event.get("raw_response"))
            if status == "parsed":
                # parsed requires a clean event and a reproducible verdict
                assert event.get("http_status") == 200 and not event.get("error"), (
                    f"{stage} parsed from a failed API attempt: {key}"
                )
                assert event.get("finish_reason") != "length", (
                    f"{stage} parsed from a provider-truncated event: {key}"
                )
                assert verdict == row[f"{stage}_verdict"] and verdict is not None, (
                    f"{stage} verdict does not reproduce from raw response: {key}"
                )
            else:
                # malformed_or_error: visible failure, never a verdict.
                # Referenced event must be failed or its raw response
                # unparseable; a verdict parsed from a failed event is
                # discarded, so the recorded verdict must be None.
                assert status == "malformed_or_error", f"unknown {stage} status: {key}"
                assert row[f"{stage}_verdict"] is None, (
                    f"{stage} malformed record carries a verdict: {key}"
                )
                assert verdict is None or (
                    event.get("http_status") != 200
                    or event.get("error")
                    or event.get("finish_reason") == "length"
                ), f"{stage} malformed but event is clean and parseable: {key}"

    manifest = out / "manifest.sha256"
    assert manifest.exists(), "missing manifest"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = out / relative
        assert path.is_file(), f"manifest file missing: {relative}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, f"manifest hash mismatch: {relative}"

    parsed_evaluator = sum(
        1 for row in derived
        for stage in ("selfcheck", "steelman")
        if row[f"{stage}_extraction"] == "parsed"
    )
    malformed_evaluator = sum(
        1 for row in derived
        for stage in ("selfcheck", "steelman")
        if row[f"{stage}_extraction"] == "malformed_or_error"
    )
    notrun_evaluator = sum(
        1 for row in derived
        for stage in ("selfcheck", "steelman")
        if row[f"{stage}_extraction"] == "not_run"
    )
    total_evaluator = 2 * len(derived)
    # All consistency checks above are asserts; reaching here IS the pass.
    # Malformed records are expected and visible, not failures.
    report = {
        "events": len(events),
        "successful_events": success_events,
        "failed_transport_or_http_events": malformed_events,
        "length_limited_events": length_events,
        "candidate_records": len(candidates),
        "derived_records": len(derived),
        "evaluator_verdicts_reproduced": parsed_evaluator,
        "evaluator_verdicts_malformed_visible": malformed_evaluator,
        "evaluator_verdicts_not_run": notrun_evaluator,
        "evaluator_verdicts_total": total_evaluator,
        "manifest_files": len(manifest.read_text(encoding="utf-8").splitlines()),
        "status": "PASS",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
