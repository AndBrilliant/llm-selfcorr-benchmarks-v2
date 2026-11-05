#!/usr/bin/env python3
"""Pilot runner for the fresh MBPP benchmark.

This is a small production gate, not the full benchmark. It stores every complete
raw API response before parsing and uses only strict terminal verdicts. A malformed
verdict is preserved as an extraction failure; it is never resolved by fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "data/MBPP.jsonl"
PRINT_LOCK = threading.Lock()
LEDGER_LOCK = threading.Lock()

SCHEMA_VERSION = "mbpp-v2-pilot/1"

MODEL_SPECS = {
    "gpt-5.4": {
        "provider": "openai",
        "model_id": "gpt-5.4",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "key_file": "openai",
    },
    "ds-v4-flash": {
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-v4.1-flash",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "key_file": "openrouter",
    },
    "glm-5.3-flash": {
        "provider": "openrouter",
        "model_id": "z-ai/glm-5.3-flash",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "key_file": "openrouter",
    },
    "qwen38-flash": {
        "provider": "openrouter",
        "model_id": "qwen/qwen3.8-flash",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "key_file": "openrouter",
    },
    "gpt-6-luna": {
        "provider": "openrouter",
        "model_id": "openai/gpt-6-luna",
        "base_url": "https://openrouter.ai/api/v1/chat/completions",
        "key_file": "openrouter",
    },
}

DANGEROUS = {
    "os", "subprocess", "sys", "socket", "shutil", "pickle", "ctypes",
    "importlib", "builtins", "pathlib", "tempfile", "webbrowser", "requests",
    "urllib", "http", "smtplib", "ftplib", "telnetlib", "multiprocessing",
    "signal", "resource", "gc", "inspect",
}
DANGEROUS_CALLS = {
    "eval", "exec", "open", "__import__", "compile", "input", "breakpoint",
    "system", "popen", "getattr", "setattr", "delattr", "globals", "locals",
    "vars", "exit", "quit",
}
SAFE_HEADER = (
    "from typing import *\n"
    "import math, re, collections, itertools, heapq, bisect, string, "
    "functools, operator, random, statistics, cmath, fractions, decimal, "
    "datetime, copy\n"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    with LEDGER_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())


def load_task_map() -> dict[str, dict]:
    rows = {}
    with TASKS.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                rows[row["task_id"]] = row
    return rows


def generation_prompt(task: dict) -> str:
    return task["prompt"]


def selfcheck_prompt(task: dict, code: str) -> str:
    return (
        "You are evaluating one immutable Python candidate solution.\n\n"
        f"[PROBLEM]\n{task['prompt']}\n\n"
        f"[CANDIDATE]\n{code}\n\n"
        "Analyze whether the candidate satisfies the specification and tests. "
        "Do not modify, repair, rewrite, or provide an alternative solution.\n"
        "End with exactly one final line containing only one of:\n"
        "VERDICT: ACCEPT\nVERDICT: REJECT"
    )


def steelman_prompt(task: dict, code: str) -> str:
    return (
        "You are a neutral arbiter judging a debate between two expert Python programmers.\n\n"
        f"[PROBLEM]\n{task['prompt']}\n\n"
        f"[CANDIDATE]\n{code}\n\n"
        "Do the following in order:\n"
        "1. STEELMAN AGAINST: give the strongest case that the candidate is wrong.\n"
        "2. STEELMAN FOR: give the strongest case that the candidate is correct.\n"
        "3. Decide which case survives scrutiny. Judge only the candidate as given; "
        "do not provide a corrected version.\n"
        "End with exactly one final line containing only one of:\n"
        "VERDICT: ACCEPT\nVERDICT: REJECT"
    )


def strict_verdict(response: str | None) -> str | None:
    if response is None:
        return None
    lines = [line.strip() for line in response.rstrip().splitlines() if line.strip()]
    if not lines:
        return None
    matches = [line for line in lines if re.fullmatch(r"VERDICT\s*:\s*(ACCEPT|REJECT)", line, re.IGNORECASE)]
    if len(matches) != 1 or matches[0] != lines[-1]:
        return None
    return re.fullmatch(r"VERDICT\s*:\s*(ACCEPT|REJECT)", matches[0], re.IGNORECASE).group(1).upper()


def extract_code(text: str | None, task: dict) -> str | None:
    if not text:
        return None
    t = text.strip()
    match = re.search(r"```(?:python|py)?\s*\n?(.*?)```", t, re.DOTALL)
    if match:
        t = match.group(1).strip()
    lines = t.splitlines()
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("def ", "import ", "from ")):
            start = i
            break
    if start is not None:
        return "\n".join(lines[start:]).strip() or None
    combined = task["prompt"].rstrip() + "\n" + t
    return combined.strip() or None


def ast_guard(code: str) -> tuple[bool, str]:
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax error: {str(exc)[:120]}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in DANGEROUS:
                    return False, f"blocked import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in DANGEROUS:
                return False, f"blocked from-import {node.module}"
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else None)
            if name in DANGEROUS_CALLS:
                return False, f"blocked {name}()"
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in DANGEROUS:
                return False, f"blocked {func.value.id}.{func.attr}"
    return True, ""


def run_task_tests(code: str, task: dict) -> tuple[bool, str]:
    ok, why = ast_guard(code)
    if not ok:
        return False, why
    runner = (
        SAFE_HEADER + "\n" + code + "\n\n" + task["test"] +
        f"\n\ncheck({task['entry_point']})\nprint('PASS')\n"
    )
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir="/tmp") as fh:
            fh.write(runner)
            path = fh.name
        result = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and "PASS" in result.stdout:
            return True, "pass"
        err = (result.stderr or result.stdout or "").strip()
        if "AssertionError" in err:
            detail = err.splitlines()[-1][:200] if err.splitlines() else ""
            return False, "assertion failed: " + detail
        return False, (err or f"exit {result.returncode}")[:240]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def call_model(model: str, prompt: str, stage: str, task_id: str, out: Path,
               max_transport_attempts: int) -> dict:
    spec = MODEL_SPECS[model]
    key = (Path.home() / ".keys" / spec["key_file"]).read_text().strip()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    # NO max_tokens / max_completion_tokens: the API must never truncate a
    # response at our request. Responses generate until the model itself
    # finishes (finish_reason="stop") or the provider's own context ceiling
    # is reached (finish_reason="length", recorded visibly, never retried).
    body = {
        "model": spec["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    base = {
        "schema_version": SCHEMA_VERSION,
        "ts": utc_now(),
        "stage": stage,
        "task_id": task_id,
        "model": model,
        "provider": spec["provider"],
        "provider_model_id": spec["model_id"],
        "prompt_sha256": sha256_text(prompt),
        "max_output_tokens": None,
        "temperature": 0,
    }

    last_error = None
    for attempt in range(1, max_transport_attempts + 1):
        started = time.time()
        log(f"CALL {model} {stage} {task_id} attempt={attempt}")
        try:
            response = requests.post(
                spec["base_url"], headers=headers, json=body, timeout=1800
            )
            duration = round(time.time() - started, 3)
            try:
                payload = response.json()
            except ValueError:
                payload = None
            record = dict(base)
            record.update({
                "event_id": str(uuid.uuid4()),
                "transport_attempt": attempt,
                "http_status": response.status_code,
                "duration_s": duration,
                "raw_response": None,
                "raw_response_sha256": None,
                "finish_reason": None,
                "usage": None,
                "provider_response_id": None,
                "error": None,
            })
            if response.status_code >= 400 or payload is None:
                record["error"] = response.text[:1000] if payload is None else json.dumps(payload)[:1000]
                append_jsonl(out / "events.jsonl", record)
                last_error = record["error"]
                time.sleep(min(2**attempt, 8))
                continue
            choices = payload.get("choices") or []
            message = (choices[0].get("message") if choices else {}) or {}
            raw = message.get("content")
            if isinstance(raw, list):
                raw = "".join(part.get("text", "") for part in raw if isinstance(part, dict))
            finish_reason = choices[0].get("finish_reason") if choices else None
            if not isinstance(raw, str):
                provider_error = "provider returned no string content"
            elif finish_reason == "length":
                provider_error = "provider response truncated by max tokens"
            else:
                provider_error = None
            record.update({
                "raw_response": raw if isinstance(raw, str) else None,
                "raw_response_sha256": sha256_text(raw) if isinstance(raw, str) else None,
                "finish_reason": finish_reason,
                "usage": payload.get("usage"),
                "provider_response_id": payload.get("id"),
                "error": provider_error,
            })
            append_jsonl(out / "events.jsonl", record)
            if provider_error is None:
                return record
            # Max-token truncation at temperature 0 is deterministic: a retry
            # reproduces the same reasoning loop and burns the same budget.
            # Record it once as an immutable failed attempt; never retry.
            # Other provider errors are retried under the fixed policy.
            if finish_reason != "length":
                last_error = provider_error
                if attempt < max_transport_attempts:
                    time.sleep(min(2**attempt, 8))
                    continue
            return record
        except requests.RequestException as exc:
            duration = round(time.time() - started, 3)
            record = dict(base)
            record.update({
                "event_id": str(uuid.uuid4()),
                "transport_attempt": attempt,
                "http_status": None,
                "duration_s": duration,
                "raw_response": None,
                "raw_response_sha256": None,
                "finish_reason": None,
                "usage": None,
                "provider_response_id": None,
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            })
            append_jsonl(out / "events.jsonl", record)
            last_error = record["error"]
            time.sleep(min(2**attempt, 8))

    return dict(base, event_id=str(uuid.uuid4()), transport_attempt=max_transport_attempts, http_status=None,
                duration_s=None, raw_response=None, raw_response_sha256=None,
                finish_reason=None, usage=None, provider_response_id=None,
                error=f"call failed after {max_transport_attempts} attempts: {last_error}")


def completed_pairs(out: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    path = out / "derived_records.jsonl"
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((row.get("model"), row.get("task_id")))
    return done


def already_done(out: Path, model: str, task_id: str) -> bool:
    return (model, task_id) in completed_pairs(out)


def write_manifest(out: Path) -> None:
    files = sorted(
        path for path in out.rglob("*")
        if path.is_file() and path.name != "manifest.sha256"
    )
    lines = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(out).as_posix()}")
    (out / "manifest.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pair(task: dict, task_id: str, model: str, out: Path,
             max_transport_attempts: int) -> dict:
    """Run one (task, model) pair: generate, test, then both evaluator arms.

    Returns the pair's token usage. Every API attempt is appended to the
    immutable event ledger before any parsing happens.
    """
    log(f"RUN {model} {task_id}")

    gen_prompt = generation_prompt(task)
    gen_event = call_model(model, gen_prompt, "generation", task_id, out,
                           max_transport_attempts)
    code = extract_code(gen_event.get("raw_response"), task)
    candidate = None
    if code is not None:
        passed, test_reason = run_task_tests(code, task)
        candidate = {
            "schema_version": SCHEMA_VERSION,
            "ts": utc_now(),
            "task_id": task_id,
            "model": model,
            "generation_event_id": gen_event["event_id"],
            "code": code,
            "code_sha256": sha256_text(code),
            "T": int(passed),
            "test_reason": test_reason,
        }
        append_jsonl(out / "candidate_records.jsonl", candidate)

    derived = {
        "schema_version": SCHEMA_VERSION,
        "ts": utc_now(),
        "task_id": task_id,
        "model": model,
        "generation_event_id": gen_event["event_id"],
        "generation_error": gen_event.get("error"),
        "generation_http_status": gen_event.get("http_status"),
        "generation_finish_reason": gen_event.get("finish_reason"),
        "code_sha256": candidate["code_sha256"] if candidate else None,
        "T": candidate["T"] if candidate else None,
        "test_reason": candidate["test_reason"] if candidate else None,
        "selfcheck_event_id": None,
        "selfcheck_verdict": None,
        "selfcheck_extraction": "not_run",
        "steelman_event_id": None,
        "steelman_verdict": None,
        "steelman_extraction": "not_run",
    }

    usage = {"input_tokens": 0, "output_tokens": 0}

    if candidate is not None:
        # selfcheck and steelman are independent given the frozen candidate:
        # run them concurrently. Determinism is unaffected (temperature 0,
        # independent calls, both read the same frozen candidate).
        def eval_stage(stage: str, prompt_fn):
            event = call_model(model, prompt_fn(task, code), stage, task_id, out,
                               max_transport_attempts)
            verdict = strict_verdict(event.get("raw_response"))
            u = event.get("usage") or {}
            return {
                f"{stage}_event_id": event["event_id"],
                f"{stage}_verdict": verdict,
                f"{stage}_extraction": (
                    "parsed" if verdict is not None and event.get("error") is None
                    else "malformed_or_error"
                ),
                "_in": u.get("prompt_tokens", 0),
                "_out": u.get("completion_tokens", 0),
            }

        with ThreadPoolExecutor(max_workers=2) as eval_pool:
            for results in eval_pool.map(
                eval_stage,
                ("selfcheck", "steelman"),
                (selfcheck_prompt, steelman_prompt),
            ):
                usage["input_tokens"] += results.pop("_in")
                usage["output_tokens"] += results.pop("_out")
                derived.update(results)

    append_jsonl(out / "derived_records.jsonl", derived)
    return usage


def main() -> int:
    global SCHEMA_VERSION
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt-5.4", "ds-v4-flash"])
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="task IDs to run; defaults to every task in the corpus")
    parser.add_argument("--out", default="data/mbpp_v2/pilot")
    parser.add_argument("--transport-retries", type=int, default=2,
                        help="retries after the initial attempt for transport/HTTP errors; "
                             "deterministic max-token truncation is never retried")
    parser.add_argument("--workers", type=int, default=1,
                        help="concurrent (task, model) pairs; each pair is sequential internally")
    parser.add_argument("--schema", default=SCHEMA_VERSION,
                        help="schema version string written to all records")
    args = parser.parse_args()

    SCHEMA_VERSION = args.schema

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    task_map = load_task_map()
    if args.tasks is None:
        args.tasks = sorted(task_map, key=lambda tid: int(tid.split("/")[1]))
    missing = [task_id for task_id in args.tasks if task_id not in task_map]
    if missing:
        raise SystemExit(f"Unknown task IDs: {missing}")
    for model in args.models:
        if model not in MODEL_SPECS:
            raise SystemExit(f"Unknown model: {model}")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    max_transport_attempts = args.transport_retries + 1
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    # Full work list in deterministic order, minus pairs already present in the
    # immutable ledger. Disjoint round-robin partitions keep workers from ever
    # sharing a pair, so resume after interruption cannot duplicate work.
    done = completed_pairs(out)
    work = [
        (task_map[task_id], task_id, model)
        for task_id in args.tasks
        for model in args.models
        if (model, task_id) not in done
    ]
    skipped = len(args.tasks) * len(args.models) - len(work)
    if skipped:
        log(f"SKIP {skipped} already-recorded pairs")

    partitions = [work[i::args.workers] for i in range(args.workers)]
    partitions = [p for p in partitions if p]

    def worker(partition: list) -> dict:
        subtotal = {"input_tokens": 0, "output_tokens": 0}
        for task, task_id, model in partition:
            try:
                usage = run_pair(task, task_id, model, out, max_transport_attempts)
            except Exception as exc:  # one bad pair must not kill the pool
                log(f"ERROR pair {model} {task_id}: {type(exc).__name__}: {exc}")
                append_jsonl(out / "runner_errors.jsonl", {
                    "ts": utc_now(), "model": model, "task_id": task_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            subtotal["input_tokens"] += usage["input_tokens"]
            subtotal["output_tokens"] += usage["output_tokens"]
        return subtotal

    with ThreadPoolExecutor(max_workers=len(partitions)) as pool:
        for subtotal in pool.map(worker, partitions):
            total_usage["input_tokens"] += subtotal["input_tokens"]
            total_usage["output_tokens"] += subtotal["output_tokens"]

    write_manifest(out)
    print(json.dumps({"completed_pairs": len(work), "skipped_pairs": skipped,
                      "usage": total_usage}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
