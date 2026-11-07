#!/usr/bin/env python3
"""Cross-evaluation matrix analysis: does an evaluator's blind spot extend to
other families' candidates?

Design: every frozen production candidate (generator G, task t) is judged by
the four OTHER families (evaluator E != G) in both selfcheck and steelman
arms. G's own judgments of the same candidates live in the production ledger,
so self vs. cross comparisons are paired at the candidate level.

Denominators: candidates with ground truth assigned and a parsed terminal
verdict in both arms of the comparison being made. Malformed records are
reported as counts, never imputed.

Output: data/mbpp_v2/cross_eval/cross_analysis_summary.txt
"""
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V = {"ACCEPT": True, "REJECT": False}
MODELS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]
NAMES = {
    "gpt-5.4": "GPT-5.4",
    "ds-v4-flash": "DeepSeek",
    "glm-5.3-flash": "GLM",
    "qwen38-flash": "Qwen",
    "gpt-6-luna": "GPT-6-Luna",
}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    from math import comb
    x = min(b, c)
    p = sum(comb(n, i) for i in range(0, x + 1)) / 2**n
    return min(1.0, 2 * p)


def main():
    # production: generator's own judgments + ground truth
    prod = {}
    for l in (ROOT / "data/mbpp_v2/production/derived_records.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            prod[(r["model"], r["task_id"])] = r

    cross = [json.loads(l) for l in (ROOT / "data/mbpp_v2/cross_eval/cross_records.jsonl").read_text().splitlines() if l.strip()]
    malformed = defaultdict(int)
    rows = []
    for r in cross:
        e, g, t = r["evaluator"], r["generator"], r["task_id"]
        for arm in ("selfcheck", "steelman"):
            v = r.get(f"{arm}_verdict")
            if v not in V:
                malformed[(e, arm)] += 1
                continue
            p = prod.get((g, t))
            if not p or p.get("T") not in (0, 1):
                continue
            rows.append((e, g, t, arm, V[v], p["T"], p.get(f"{arm}_verdict")))

    out = []
    out.append("Cross-evaluation matrix analysis")
    out.append("================================")
    out.append("Each frozen production candidate is judged by the four other families in both")
    out.append("arms; the generator's own judgments come from the production ledger. Rates use")
    out.append("the paired stratum (ground truth assigned, parsed terminal verdict in both arms")
    out.append("of the comparison). Malformed cross records are reported, never imputed.")
    out.append("")

    for arm in ("selfcheck", "steelman"):
        out.append(f"--- arm: {arm} ---")
        out.append(f"{'evaluator':12} {'cand family':14} {'n_pair':>6} {'FAR% [95% CI]':>22} {'TAR% [95% CI]':>22}")
        # evaluator E on its OWN candidates (production)
        for e in MODELS:
            own = [p for (m, _), p in prod.items() if m == e and p.get("T") in (0, 1)
                   and p.get(f"{arm}_verdict") in V]
            k0 = sum(1 for p in own if p["T"] == 0 and V[p[f"{arm}_verdict"]]); n0 = sum(1 for p in own if p["T"] == 0)
            k1 = sum(1 for p in own if p["T"] == 1 and V[p[f"{arm}_verdict"]]); n1 = sum(1 for p in own if p["T"] == 1)
            w0, w1 = wilson(k0, n0), wilson(k1, n1)
            out.append(f"{NAMES[e]:12} {'(own)':14} {n0+n1:>6} {100*k0/n0:6.1f} [{100*w0[0]:.1f},{100*w0[1]:.1f}]   {100*k1/n1:6.1f} [{100*w1[0]:.1f},{100*w1[1]:.1f}]")
            # E on each other family (cross)
            for g in MODELS:
                if g == e:
                    continue
                sel = [r for r in rows if r[0] == e and r[1] == g and r[3] == arm]
                k0 = sum(1 for r in sel if r[5] == 0 and r[4]); n0 = sum(1 for r in sel if r[5] == 0)
                k1 = sum(1 for r in sel if r[5] == 1 and r[4]); n1 = sum(1 for r in sel if r[5] == 1)
                w0, w1 = wilson(k0, n0), wilson(k1, n1)
                out.append(f"{'':12} {NAMES[g]:14} {n0+n1:>6} {100*k0/n0:6.1f} [{100*w0[0]:.1f},{100*w0[1]:.1f}]   {100*k1/n1:6.1f} [{100*w1[0]:.1f},{100*w1[1]:.1f}]")
            # paired discordance: E-cross vs generator-self on the same incorrect candidates
            b = c = 0
            for r in rows:
                e_, g_, t_, arm_, ev, T, self_v = r
                if e_ == e and arm_ == arm and T == 0 and self_v in V:
                    b += ev and not V[self_v]
                    c += (not ev) and V[self_v]
            out.append(f"{'':12} discordance vs generator-self on T=0: E rejects & G accepts = {b}, E accepts & G rejects = {c}, exact p = {mcnemar(b,c):.6g}")
            out.append("")

    out.append("malformed cross records (reported, never imputed):")
    for k in sorted(malformed):
        out.append(f"  {NAMES[k[0]]} {k[1]}: {malformed[k]}")

    summary = "\n".join(out)
    (ROOT / "data/mbpp_v2/cross_eval/cross_analysis_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
