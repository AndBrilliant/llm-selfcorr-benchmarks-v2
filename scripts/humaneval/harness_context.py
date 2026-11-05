#!/usr/bin/env python3
"""Context-separation 2x2 on frozen HumanEval candidates (reviewer Issue 5).

The published HumanEval arms are BOTH fresh-context: arm1 self-check and arm2
steelman each receive a standalone prompt. This harness supplies the two missing
cells of the 2x2 by replaying the ORIGINAL generation conversation and appending
the SAME evaluator instruction as the next user turn:

                     self-check        steelman
  fresh context      arm1 (exists)     arm2 (exists)
  same context       same_sc (here)    same_st (here)

For self-check, the appended instruction and parser are identical to the fresh
arm. For steelman, the core adversarial instruction, temperature, and 2,048-token
cap match, but this same-context variant omits the fresh arm's mandatory
CASE_AGAINST/CASE_FOR response template and scans backward for a FINAL_VERDICT
line instead of applying the fresh arm's structural parser. It therefore does
not isolate context independently of response-format and parsing differences.
Model: gpt-5.4-2026-03-05 (Stage-0 snapshot), temperature 0.

Record: task_id, accept_same_sc, accept_same_st, raw responses, usage.
Resume-safe.
"""
import json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/Drew/mcp-servers/paper-tools/check-citations")
import api_client  # loads API keys from ~/.keys
from openai import OpenAI

DATA = Path.home() / "selfcorr-bench" / "context_data"
OUT = DATA / "context_same.jsonl"
MODEL = "gpt-5.4-2026-03-05"

SC_TEMPLATE = (
    "You are checking a Python solution that you produced earlier for a "
    "programming problem.\n\n"
    "Determine whether the candidate solution is fully correct for all "
    "inputs allowed by the problem specification.\n\n"
    "Do not modify, repair, rewrite, or improve the solution.\n"
    "Do not propose an alternative solution.\n"
    "Do not execute the code.\n"
    "Do not assume access to unit tests, reference answers, execution "
    "results, or any information not shown below.\n\n"
    "Programming problem:\n---BEGIN PROBLEM---\n{task_prompt}\n---END PROBLEM---\n\n"
    "Your previously produced solution:\n---BEGIN CANDIDATE---\n{candidate_code}\n"
    "---END CANDIDATE---\n\n"
    "Is the candidate solution fully correct?\n\n"
    "Answer with exactly YES or NO.\nDo not output anything else."
)

ST_TEMPLATE = (
    "You are critically evaluating a Python solution that you produced "
    "earlier for a programming problem.\n\n"
    "Your task is to determine whether the candidate is fully correct for "
    "all inputs allowed by the specification.\n\n"
    "Perform a structured adversarial evaluation before giving your verdict.\n\n"
    "First, make the strongest specific case that the candidate is WRONG. "
    "Look actively for edge cases, specification mismatches, incorrect "
    "assumptions, arithmetic or logical errors, and cases where apparently "
    "plausible code would fail.\n\n"
    "Second, make the strongest specific case that the candidate is "
    "CORRECT. Check whether the apparent concerns actually apply and "
    "whether the implementation satisfies the complete specification.\n\n"
    "Finally, decide which case survives scrutiny.\n\n"
    "You are evaluating the existing candidate only.\n\n"
    "Do not modify, repair, rewrite, or improve the candidate.\n"
    "Do not provide corrected code or an alternative implementation.\n"
    "Do not execute the code.\n"
    "Do not assume access to unit tests, reference answers, execution "
    "results, or any information not shown below.\n\n"
    "Programming problem:\n---BEGIN PROBLEM---\n{task_prompt}\n---END PROBLEM---\n\n"
    "Your previously produced solution:\n---BEGIN CANDIDATE---\n{candidate_code}\n"
    "---END CANDIDATE---\n\n"
    "Is the candidate solution fully correct?\n\n"
    "End your response with the final verdict on its own line in exactly "
    "this format:\nFINAL_VERDICT: YES\nor\nFINAL_VERDICT: NO\n"
    "The final line must be exactly FINAL_VERDICT: YES or FINAL_VERDICT: NO."
)

def parse_st(raw):
    for line in reversed((raw or "").strip().splitlines()):
        t = line.strip().upper()
        if t.startswith("FINAL_VERDICT:"):
            v = t.split(":", 1)[1].strip()
            return 1 if v == "YES" else (0 if v == "NO" else None)
    return None

def main():
    api_client.load_keys()
    client = OpenAI()
    rows = [json.loads(l) for l in open(DATA / "baseline.jsonl") if l.strip()]
    done = {}
    if OUT.exists():
        for l in open(OUT):
            if l.strip():
                r = json.loads(l)
                if "error" not in r and r.get("accept_same_sc") in (0, 1) \
                        and r.get("accept_same_st") in (0, 1):
                    done[r["task_id"]] = r
    todo = [r for r in rows if r["task_id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    lock = threading.Lock()
    totals = {"sc_in": 0, "sc_out": 0, "st_in": 0, "st_out": 0}

    def call(messages, max_tok):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0,
            max_completion_tokens=max_tok, reasoning_effort="none")
        u = resp.usage
        return resp.choices[0].message.content, (u.prompt_tokens if u else 0), \
               (u.completion_tokens if u else 0)

    def one(row):
        tid = row["task_id"]
        rec = {"task_id": tid, "task_index": row["task_index"],
               "baseline_correct": row["baseline_correct"]}
        hist = [{"role": "user", "content": row["generation_prompt"]},
                {"role": "assistant", "content": row["raw_model_response"]}]
        try:
            p = SC_TEMPLATE.format(task_prompt=row["prompt"],
                                   candidate_code=row["candidate_code"])
            txt, i, o = call(hist + [{"role": "user", "content": p}], 16)
            v = (txt or "").strip().upper()
            if v not in ("YES", "NO"):
                rec["error"] = f"sc invalid verdict: {txt[:60]}"
            rec["accept_same_sc"] = 1 if v == "YES" else 0
            rec["same_sc_raw"] = txt
            with lock:
                totals["sc_in"] += i; totals["sc_out"] += o
        except Exception as e:
            rec["error"] = f"sc {type(e).__name__}: {e}"[:300]
        try:
            p = ST_TEMPLATE.format(task_prompt=row["prompt"],
                                   candidate_code=row["candidate_code"])
            txt, i, o = call(hist + [{"role": "user", "content": p}], 2048)
            v = parse_st(txt)
            if v is None:
                rec["error"] = (rec.get("error", "") + " st invalid verdict")[:300]
            else:
                rec["accept_same_st"] = v
            rec["same_st_raw"] = txt
            with lock:
                totals["st_in"] += i; totals["st_out"] += o
        except Exception as e:
            rec["error"] = (rec.get("error", "") + f" st {type(e).__name__}: {e}")[:300]
        with lock:
            with open(OUT, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done[tid] = rec
            n = len(done)
            if n % 20 == 0 or "error" in rec:
                print(f"[{n}] {tid} sc={rec.get('accept_same_sc')} "
                      f"st={rec.get('accept_same_st')}"
                      + (f" ERR {rec['error'][:80]}" if "error" in rec else ""),
                      flush=True)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(one, todo))
    print(f"TOKENS {json.dumps(totals)}", flush=True)
    print("DONE", OUT, flush=True)

if __name__ == "__main__":
    main()
