#!/usr/bin/env python3
"""Tier 1 analyses: repeated-evaluation sharpening and candidate replication.

Repeated evaluation (data/mbpp_v2/repeated_eval, rounds 2-3 + production
round 1): per arm and family, is verdict stability high at temperature 0,
and does accepting in ALL k rounds carry more information about T than a
single acceptance? If P(T=0 | all 3 ACCEPT) ~= P(T=0 | 1 ACCEPT), repeated
confident acceptance adds no evidence -- the shared-ceiling signature that
motivates the information bound.

Candidate replication (data/mbpp_v2/replication): two independent greedy
candidates per (task, model). Tests whether an arm's verdict encodes
cell-level information: does the verdict on candidate A predict the ground
truth of candidate B (Fisher exact), and does the verdict repeat across
independent candidates of the same task (verdict_A vs verdict_B)?
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
VERDICTS = {"ACCEPT": True, "REJECT": False}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    if n == 0:
        return 1.0
    r1, r2 = a + b, c + d
    k1, k2 = a + c, b + d

    def logp(x: int) -> float:
        return (math.lgamma(r1 + 1) + math.lgamma(r2 + 1) + math.lgamma(k1 + 1) +
                math.lgamma(k2 + 1) - math.lgamma(n + 1) - math.lgamma(x + 1) -
                math.lgamma(r1 - x + 1) - math.lgamma(k1 - x + 1) -
                math.lgamma(r2 - k1 + x + 1))

    lo, hi = max(0, k1 - r2), min(r1, k1)
    p_obs = logp(a)
    total = 0.0
    for x in range(lo, hi + 1):
        lp = logp(x)
        if lp <= p_obs + 1e-9:
            total += math.exp(lp)
    return min(1.0, total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--production", default="data/mbpp_v2/production")
    ap.add_argument("--repeated", default="data/mbpp_v2/repeated_eval")
    ap.add_argument("--replication", default="data/mbpp_v2/replication")
    ap.add_argument("--out", default="data/mbpp_v2/tier1_analysis")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    prod = {(r["model"], r["task_id"]): r for r in read_jsonl(ROOT / args.production / "derived_records.jsonl")}
    rep = {(r["model"], r["task_id"]): r for r in read_jsonl(ROOT / args.repeated / "repeated_records.jsonl")}
    repl = {(r["model"], r["task_id"]): r for r in read_jsonl(ROOT / args.replication / "derived_records.jsonl")}

    result = {"repeated_evaluation": [], "candidate_replication": []}
    lines = ["Tier 1 analyses: repeated evaluation and candidate replication", "=" * 64, ""]

    # ---- A. repeated evaluation ----
    lines.append("A. Repeated evaluation (k=3 rounds per arm, temperature 0)")
    for model in MODEL_ORDER:
        fam = {"model": model, "arms": {}}
        lines.append(f"== {model}")
        for arm in ("selfcheck", "steelman"):
            # per-candidate: (v1, v2, v3, T)
            rows = []
            for (m, t), r2 in rep.items():
                if m != model:
                    continue
                p = prod.get((m, t))
                if not p or p.get("T") not in (0, 1):
                    continue
                v1, v2, v3 = p.get(f"{arm}_verdict"), r2.get(f"{arm}_r2_verdict"), r2.get(f"{arm}_r3_verdict")
                vals = [VERDICTS.get(v) for v in (v1, v2, v3)]
                if any(v is None for v in vals):
                    continue
                rows.append((vals, p["T"]))
            n = len(rows)
            allsame = sum(v[0] == v[1] == v[2] for v, _ in rows)
            acc_all3 = [T for v, T in rows if all(v)]
            acc_1or2 = [T for v, T in rows if any(v) and not all(v)]
            rej_all = [T for v, T in rows if not any(v)]
            def far(ts):
                return None if not ts else 100.0 * sum(t == 0 for t in ts) / len(ts)
            r1only = [T for (v, T) in rows if v[0]]
            a = {
                "n": n,
                "all_rounds_agree": allsame,
                "all_rounds_agree_pct": 100.0 * allsame / n if n else None,
                "far_given_3_of_3_accept": far(acc_all3),
                "n_3_of_3": len(acc_all3),
                "far_given_partial_accept": far(acc_1or2),
                "n_partial": len(acc_1or2),
                "far_given_round1_accept": far(r1only),
                "n_round1": len(r1only),
                "far_given_all_reject": far(rej_all),
                "n_all_reject": len(rej_all),
            }
            fam["arms"][arm] = a
            lines.append(f"   {arm}: n={n} all3agree={allsame} ({100*allsame/n:.1f}%) "
                         f"P(T=0|3/3 ACC)={a['far_given_3_of_3_accept']:.1f}% (n={len(acc_all3)}) "
                         f"P(T=0|1-2 ACC)={a['far_given_partial_accept']:.1f}% (n={len(acc_1or2)}) "
                         f"P(T=0|all REJ)={a['far_given_all_reject']:.1f}% (n={len(rej_all)})")
        result["repeated_evaluation"].append(fam)
    lines.append("")

    # ---- B. candidate replication ----
    lines.append("B. Candidate replication (two independent greedy candidates per task)")
    for model in MODEL_ORDER:
        cross = []
        for (m, t), p in prod.items():
            if m != model:
                continue
            q = repl.get((m, t))
            if not q or p.get("T") not in (0, 1) or q.get("T") not in (0, 1):
                continue
            cross.append((p, q))
        fam = {"model": model, "cells": len(cross)}
        lines.append(f"== {model} (cells={len(cross)})")
        for arm in ("selfcheck", "steelman"):
            va, vb = [], []
            ta, tb = [], []
            for p, q in cross:
                if p.get(f"{arm}_verdict") in VERDICTS and q.get(f"{arm}_verdict") in VERDICTS:
                    va.append(VERDICTS[p[f"{arm}_verdict"]])
                    vb.append(VERDICTS[q[f"{arm}_verdict"]])
                    ta.append(p["T"])
                    tb.append(q["T"])
            n = len(va)
            # verdict repeat rate across independent candidates
            agree = sum(x == y for x, y in zip(va, vb))
            # cross-candidate information: verdict on A vs T of B
            ct = [((x, y2)) for x, y2 in zip(va, tb)]
            a10 = sum(1 for x, t in ct if x and t == 0)
            a11 = sum(1 for x, t in ct if x and t == 1)
            r10 = sum(1 for x, t in ct if not x and t == 0)
            r11 = sum(1 for x, t in ct if not x and t == 1)
            p_cross = fisher_two_sided(a10, a11, r10, r11)
            # task-level correlated failure: does the arm's ERROR on candidate A
            # predict that the independent candidate B is also incorrect?
            err_cells = [(VERDICTS[p[f"{arm}_verdict"]] != (p["T"] == 1), q["T"])
                         for p, q in cross
                         if p.get(f"{arm}_verdict") in VERDICTS and q.get("T") in (0, 1)]
            e1 = sum(1 for e, t in err_cells if e and t == 0)
            e0 = sum(1 for e, t in err_cells if e and t == 1)
            c1 = sum(1 for e, t in err_cells if not e and t == 0)
            c0 = sum(1 for e, t in err_cells if not e and t == 1)
            p_err = fisher_two_sided(e1, e0, c1, c0)
            # both-candidate joint FAR: cells where arm accepts BOTH candidates
            both = [tb[i] for i in range(n) if va[i] and vb[i]]
            either = [tb[i] for i in range(n) if va[i] or vb[i]]
            def far(ts):
                return None if not ts else 100.0 * sum(t == 0 for t in ts) / len(ts)
            fam[arm] = {
                "n": n,
                "verdict_repeat_pct": 100.0 * agree / n if n else None,
                "cross_candidate_fisher_p": p_cross,
                "accA_T0": a10, "accA_T1": a11, "rejA_T0": r10, "rejA_T1": r11,
                "errorA_predicts_T0": {"err_T0": e1, "err_T1": e0, "corr_T0": c1, "corr_T1": c0,
                                       "fisher_p": p_err},
                "far_given_both_accept": far(both),
                "n_both_accept": len(both),
                "far_given_either_accept": far(either),
                "n_either_accept": len(either),
            }
            lines.append(f"   {arm}: n={n} verdict-repeat={100*agree/n:.1f}% "
                         f"cross-cand Fisher p={p_cross:.3g} "
                         f"errA->T_B=0 Fisher p={p_err:.3g} "
                         f"P(T=0|accept both)={fam[arm]['far_given_both_accept']:.1f}% (n={len(both)}) "
                         f"P(T=0|accept either)={fam[arm]['far_given_either_accept']:.1f}% (n={len(either)})")
        result["candidate_replication"].append(fam)
    lines.append("")

    # ---- C. always-accept baseline for the replication association ----
    # The review objection: an always-accept evaluator has err_A = 1{T_A=0},
    # so "err_A predicts T_B=0" reduces to task-difficulty persistence
    # P(T_B=0 | T_A=0). Report baseline vs observed, and the incremental
    # signal within T_A strata (does the verdict predict B beyond T_A?).
    lines.append("C. Always-accept baseline for the cross-candidate association")
    result["always_accept_baseline"] = []
    for model in MODEL_ORDER:
        cells = []
        for (m, t), p in prod.items():
            if m != model:
                continue
            q = repl.get((m, t))
            if not q or p.get("T") not in (0, 1) or q.get("T") not in (0, 1):
                continue
            cells.append((p, q))
        fam = {"model": model, "cells": len(cells),
               "difficulty_baseline": None, "arms": {}}
        ta0 = sum(p["T"] == 0 for p, q in cells)
        ta0_tb0 = sum(p["T"] == 0 and q["T"] == 0 for p, q in cells)
        fam["difficulty_baseline"] = {
            "n_TA0": ta0, "P_TB0_given_TA0": (100.0 * ta0_tb0 / ta0) if ta0 else None}
        lines.append(f"== {model} (cells={len(cells)}): baseline P(T_B=0|T_A=0)="
                     f"{fam['difficulty_baseline']['P_TB0_given_TA0']:.1f}% (n={ta0})")
        for arm in ("selfcheck", "steelman"):
            n = err = err_tb0 = 0
            accA_ta0 = accA_ta0_tb0 = rejA_ta0 = rejA_ta0_tb0 = 0
            for p, q in cells:
                if p.get(f"{arm}_verdict") not in VERDICTS:
                    continue
                n += 1
                TA, TB = p["T"], q["T"]
                acc = VERDICTS[p[f"{arm}_verdict"]]
                e = acc != (TA == 1)
                if e:
                    err += 1
                    err_tb0 += (TB == 0)
                if TA == 0:
                    if acc:
                        accA_ta0 += 1
                        accA_ta0_tb0 += (TB == 0)
                    else:
                        rejA_ta0 += 1
                        rejA_ta0_tb0 += (TB == 0)
            p_inc = fisher_two_sided(accA_ta0_tb0, accA_ta0 - accA_ta0_tb0,
                                     rejA_ta0_tb0, rejA_ta0 - rejA_ta0_tb0)
            a = {"n": n,
                 "P_TB0_given_errA": (100.0 * err_tb0 / err) if err else None,
                 "n_errA": err,
                 "P_TB0_given_acceptA_TA0": (100.0 * accA_ta0_tb0 / accA_ta0) if accA_ta0 else None,
                 "n_acceptA_TA0": accA_ta0,
                 "P_TB0_given_rejectA_TA0": (100.0 * rejA_ta0_tb0 / rejA_ta0) if rejA_ta0 else None,
                 "n_rejectA_TA0": rejA_ta0,
                 "incremental_fisher_p": p_inc}
            fam["arms"][arm] = a
            lines.append(f"   {arm}: P(T_B=0|err_A)={a['P_TB0_given_errA']:.1f}% (n={err}) | "
                         f"within T_A=0: accept A -> {a['P_TB0_given_acceptA_TA0']:.1f}% (n={accA_ta0}), "
                         f"reject A -> {a['P_TB0_given_rejectA_TA0']:.1f}% (n={rejA_ta0}), p={p_inc:.3g}")
        result["always_accept_baseline"].append(fam)
    lines.append("")

    (out / "tier1_analysis.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out / "tier1_analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    entries = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "manifest.sha256":
            entries.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}")
    (out / "manifest.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
