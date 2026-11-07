#!/usr/bin/env python3
"""Full-population incorrect-candidate severity audit.

For every production candidate labeled T=0, execute it against each individual
reference assert and record how many fail. This replaces the earlier 200-candidate
sample audit with a deterministic full-population audit.

Test-string hygiene: the corpus `test` field carries a trailing v1-harness artifact
(`def check(candidate): pass`) that is stripped here; task MBPP/367 carries setup code
(Node construction) whose lines must execute before its asserts and are treated as
setup, not as asserts.

Output: data/mbpp_v2/analysis/error_severity_audit.jsonl (one row per incorrect
candidate, with per-assert booleans) plus a summary to stdout.
"""
import json
import multiprocessing as mp
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAFE = "import sys\nsys.setrecursionlimit(10000)\n"


def split_parts(raw: str) -> tuple[str, list[str]]:
    """Return (setup_src, [assert_srcs]) from a corpus test string."""
    lines = [l for l in raw.splitlines() if l.strip()]
    # drop trailing v1-harness artifact
    while lines and (lines[-1].strip() == "pass" or lines[-1].startswith("def check(")):
        lines.pop()
    asserts = [l for l in lines if l.lstrip().startswith("assert ")]
    setup = "\n".join(l for l in lines if not l.lstrip().startswith("assert "))
    return setup, asserts


def job(arg):
    code, setup, a = arg
    src = SAFE + code + "\n\n" + setup + ("\n\n" if setup else "") + a + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        path = fh.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=10)
        return r.returncode != 0
    except subprocess.TimeoutExpired:
        return True
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> None:
    tasks = {t["task_id"]: t for t in
             (json.loads(l) for l in (ROOT / "data/MBPP.jsonl").read_text().splitlines() if l.strip())}
    parts = {tid: split_parts(t["test"]) for tid, t in tasks.items()}
    recs = [json.loads(l) for l in
            (ROOT / "data/mbpp_v2/production/candidate_records.jsonl").read_text().splitlines() if l.strip()]
    bad = [r for r in recs if r.get("T") == 0]

    jobs, keys = [], []
    for r in bad:
        setup, asserts = parts[r["task_id"]]
        for a in asserts:
            jobs.append((r["code"], setup, a))
            keys.append((r["model"], r["task_id"]))
    with mp.Pool(16) as pool:
        results = pool.map(job, jobs, chunksize=8)

    agg: dict[tuple, list] = defaultdict(list)
    for k, failed in zip(keys, results):
        agg[k].append(failed)

    rows = []
    hist: dict[int, int] = defaultdict(int)
    n_all = 0
    for (m, tid), fails in agg.items():
        n = len(fails)
        f = sum(fails)
        rows.append({"model": m, "task_id": tid, "n_asserts": n, "n_failed": f,
                     "per_assert_failed": fails})
        hist[f] += 1
        if f == n:
            n_all += 1

    out = ROOT / "data/mbpp_v2/analysis/error_severity_audit.jsonl"
    with out.open("w") as fh:
        for row in sorted(rows, key=lambda r: (r["model"], r["task_id"])):
            fh.write(json.dumps(row) + "\n")

    print(f"incorrect candidates audited: {len(rows)}")
    print(f"fail every assert: {n_all} ({100*n_all/len(rows):.1f}%)")
    print("histogram of #failed asserts:", dict(sorted(hist.items())))


if __name__ == "__main__":
    main()
