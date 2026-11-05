#!/usr/bin/env python3
"""Tier 0 moderator-stratified arm-association analysis (pre-registered).

Implements scripts/mbpp_v2_moderator_spec.md. Reads the immutable production
ledger; computes per-family phi between self-check and steelman verdicts on
the paired incorrect stratum, crude and stratified by each pre-registered
moderator, with per-stratum Fisher exact p-values and joint-acceptance rates.
Writes json/txt/tex plus a manifest. No model calls; fully deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
VERDICTS = {"ACCEPT": True, "REJECT": False}

MODERATORS = ("provenance", "prompt_words", "code_lines", "n_args", "concept")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def phi_and_fisher(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """2x2 [[a,b],[c,d]]: a=both accept, b=sc only, c=st only, d=both reject.

    Returns (phi, two-sided Fisher exact p by hypergeometric enumeration).
    """
    n = a + b + c + d
    if n == 0:
        return 0.0, 1.0
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    denom = math.sqrt(row1 * row2 * col1 * col2)
    phi = ((a * d - b * c) / denom) if denom else 0.0
    # Fisher exact two-sided (probability of tables with prob <= observed)
    def logp(x: int) -> float:
        return (math.lgamma(row1 + 1) + math.lgamma(row2 + 1) +
                math.lgamma(col1 + 1) + math.lgamma(col2 + 1) -
                math.lgamma(n + 1) - math.lgamma(x + 1) -
                math.lgamma(row1 - x + 1) - math.lgamma(col1 - x + 1) -
                math.lgamma(row2 - col1 + x + 1))
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    p_obs = logp(a)
    total = 0.0
    for x in range(lo, hi + 1):
        lp = logp(x)
        if lp <= p_obs + 1e-9:
            total += math.exp(lp)
    return phi, min(1.0, total)


def moderator_values(row: dict, task: dict) -> dict:
    code = row["code"]
    n = int(row["task_id"].split("/")[1])
    provenance = "intro" if n <= 10 else ("crowd" if n <= 510 else "sanitized")
    prompt_words = len(task["prompt"].split())
    code_lines = len([ln for ln in code.splitlines() if ln.strip()])
    sig = re.search(r"def\s+" + re.escape(task["entry_point"]) + r"\s*\(([^)]*)\)", code)
    args = sig.group(1).strip() if sig else ""
    n_args = 1 if args and "," not in args and args != "self" else 2
    if not args:
        n_args = 1
    text = bool(re.search(r"^\s*(import re\b|from re\b)", code, re.M)) or \
        bool(re.search(r"\.\s*(split|join|strip|lower|upper|replace|find|startswith|endswith|format|encode|decode|splitlines|count|index|isdigit|isalpha|isnumeric|zfill|ljust|rjust|center|title|capitalize|swapcase|partition|rpartition|removeprefix|removesuffix)\s*\(", code))
    return {
        "provenance": provenance,
        "prompt_words": prompt_words,
        "code_lines": code_lines,
        "n_args": n_args,
        "concept": "text" if text else "other",
    }


def tertiles(values: list[int]) -> list[int]:
    s = sorted(values)
    return [s[len(s) // 3], s[2 * len(s) // 3]]


def stratify(rows: list[dict], name: str, t1: int, t2: int, mv: dict) -> dict:
    out = {}
    for r in rows:
        v = mv[(r["model"], r["task_id"])][name]
        if name in ("prompt_words", "code_lines"):
            v = "lo" if v <= t1 else ("mid" if v <= t2 else "hi")
        out.setdefault(str(v), []).append(r)
    return out


def cells(rows: list[dict]) -> tuple[int, int, int, int]:
    a = b = c = d = 0
    for r in rows:
        sc, st = VERDICTS[r["selfcheck_verdict"]], VERDICTS[r["steelman_verdict"]]
        if sc and st:
            a += 1
        elif sc:
            b += 1
        elif st:
            c += 1
        else:
            d += 1
    return a, b, c, d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="data/mbpp_v2/production")
    ap.add_argument("--out", default="data/mbpp_v2/moderator_analysis")
    args = ap.parse_args()
    ledger = ROOT / args.ledger
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    tasks = {t["task_id"]: t for t in read_jsonl(ROOT / "data/MBPP.jsonl")}
    derived = read_jsonl(ledger / "derived_records.jsonl")
    candidates = {(c["model"], c["task_id"]): c for c in read_jsonl(ledger / "candidate_records.jsonl")}

    # rows: paired, with frozen code attached
    rows = []
    for r in derived:
        if r.get("T") in (0, 1) and r.get("selfcheck_verdict") in VERDICTS and r.get("steelman_verdict") in VERDICTS:
            c = candidates.get((r["model"], r["task_id"]))
            if c:
                rows.append({"model": r["model"], "task_id": r["task_id"], "T": r["T"],
                             "selfcheck_verdict": r["selfcheck_verdict"], "steelman_verdict": r["steelman_verdict"],
                             "code": c["code"]})

    mv = {(r["model"], r["task_id"]): moderator_values(r, tasks[r["task_id"]]) for r in rows}
    pw_t = tertiles([v["prompt_words"] for v in mv.values()])
    cl_t = tertiles([v["code_lines"] for v in mv.values()])

    result = {"spec": "scripts/mbpp_v2_moderator_spec.md", "registration_commit": "0e16327",
              "ledger": args.ledger, "families": []}
    txt = ["Tier 0 moderator-stratified arm association (pre-registered)", "=" * 62, ""]
    for model in MODEL_ORDER:
        mr = [r for r in rows if r["model"] == model]
        m0 = [r for r in mr if r["T"] == 0]
        fam = {"model": model, "paired_incorrect": len(m0)}
        a, b, c, d = cells(m0)
        crude_phi, crude_p = phi_and_fisher(a, b, c, d)
        fam["crude"] = {"a_joint_accept": a, "b_sc_only": b, "c_st_only": c, "d_joint_reject": d,
                        "phi": crude_phi, "fisher_p": crude_p,
                        "joint_accept_pct": 100.0 * a / len(m0) if m0 else None}
        fam["moderators"] = {}
        txt.append(f"== {model} (paired incorrect n={len(m0)})")
        txt.append(f"   crude: a={a} b={b} c={c} d={d} phi={crude_phi:.3f} fisher_p={crude_p:.3g} "
                   f"joint_accept={fam['crude']['joint_accept_pct']:.1f}%")
        for mod in MODERATORS:
            strata = stratify(m0, mod, pw_t[1], cl_t[1], mv)
            sres = {}
            weighted_phi_num = weighted_n = 0
            all_ns = True
            for sname, srows in sorted(strata.items()):
                sa, sb, sc_, sd = cells(srows)
                sphi, sp = phi_and_fisher(sa, sb, sc_, sd)
                sres[sname] = {"n": len(srows), "a": sa, "b": sb, "c": sc_, "d": sd,
                               "phi": sphi, "fisher_p": sp,
                               "joint_accept_pct": 100.0 * sa / len(srows) if srows else None}
                weighted_phi_num += sphi * len(srows)
                weighted_n += len(srows)
                if not (sp > 0.05):
                    all_ns = False
            phi_mh = weighted_phi_num / weighted_n if weighted_n else 0.0
            absorbed = abs(phi_mh) < 0.10 and all_ns
            persistent = abs(phi_mh - crude_phi) <= 0.05 and \
                all((s["joint_accept_pct"] or 0) >= 70 for s in sres.values())
            sres["_phi_stratified"] = phi_mh
            sres["_rule"] = "persistent" if persistent else ("absorbed" if absorbed else "mixed")
            fam["moderators"][mod] = sres
            txt.append(f"   {mod}: phi_stratified={phi_mh:.3f} rule={sres['_rule']} " +
                       " ".join(f"[{k} n={v['n']} phi={v['phi']:.2f} p={v['fisher_p']:.2g} jA={v['joint_accept_pct']:.0f}%]"
                                for k, v in sres.items() if not k.startswith("_")))
        result["families"].append(fam)
        txt.append("")

    # family-level rule summary (persistent if ANY moderator fires persistent
    # in >=3 of 5 families; absorbed only if some moderator is absorbed in >=3)
    verdict_counts = {"persistent": 0, "absorbed": 0, "mixed": 0}
    for fam in result["families"]:
        rules = [fam["moderators"][m]["_rule"] for m in MODERATORS]
        fam["_any_persistent"] = "persistent" in rules
        fam["_any_absorbed"] = "absorbed" in rules
        verdict_counts["persistent" if fam["_any_persistent"] else ("absorbed" if fam["_any_absorbed"] else "mixed")] += 1
    npersist = verdict_counts["persistent"]
    result["headline"] = ("persistent" if npersist >= 3 else
                          ("absorbed" if verdict_counts["absorbed"] >= 3 else "mixed"))
    result["family_rule_counts"] = verdict_counts
    txt.append(f"HEADLINE: {result['headline']} ({npersist}/5 families have at least one fully persistent moderator)")

    (out / "moderator_analysis.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out / "moderator_analysis_summary.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")

    entries = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "manifest.sha256":
            entries.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}")
    (out / "manifest.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")
    print("\n".join(txt[-2:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
