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
    out.append("arms; the generator's own judgments come from the production ledger. Displayed")
    out.append("arm rates use every record with ground truth assigned and a parsed terminal")
    out.append("verdict in that arm (arm-level denominators, shown per cell). The self-versus-")
    out.append("external discordance analysis is candidate-paired: it uses only incorrect")
    out.append("candidates with a valid verdict in both the cross arm and the generator's")
    out.append("production arm. Malformed cross records are reported, never imputed.")
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
            # paired discordance on the same incorrect candidates, both arms valid:
            #   ext_rej_g_acc: external evaluator REJECTS while the generating family's
            #                  own evaluator ACCEPTS (own family more permissive)
            #   ext_acc_g_rej: external evaluator ACCEPTS while the own evaluator REJECTS
            ext_rej_g_acc = ext_acc_g_rej = 0
            per_task = defaultdict(lambda: [0, 0])  # task -> [ext_rej_g_acc, ext_acc_g_rej]
            for r in rows:
                e_, g_, t_, arm_, ev, T, self_v = r
                if e_ == e and arm_ == arm and T == 0 and self_v in V:
                    if (not ev) and V[self_v]:
                        ext_rej_g_acc += 1
                        per_task[t_][0] += 1
                    elif ev and not V[self_v]:
                        ext_acc_g_rej += 1
                        per_task[t_][1] += 1
            p_disc = mcnemar(ext_rej_g_acc, ext_acc_g_rej)
            out.append(
                f"{'':12} discordance vs generator-self on T=0 (paired, both arms valid):"
            )
            out.append(
                f"{'':12}   external rejects & own accepts (own family more permissive) = {ext_rej_g_acc};"
            )
            out.append(
                f"{'':12}   external accepts & own rejects = {ext_acc_g_rej};"
                f" exact McNemar p = {p_disc:.6g}"
            )
            # task-clustered sensitivity: candidates from the same task share the benchmark
            # problem, so cluster by task with a paired cluster bootstrap on the net count
            import random
            random.seed(20261107)
            tasks = sorted(per_task)
            if tasks:
                diffs = {tt: per_task[tt][0] - per_task[tt][1] for tt in tasks}
                B = 4000
                boots = []
                for _ in range(B):
                    s = 0
                    for _ in range(len(tasks)):
                        s += diffs[tasks[random.randrange(len(tasks))]]
                    boots.append(s)
                boots.sort()
                lo, hi = boots[int(0.025 * B)], boots[int(0.975 * B)]
                same_dir = sum(1 for tt in tasks if diffs[tt] > 0)
                opp_dir = sum(1 for tt in tasks if diffs[tt] < 0)
                out.append(
                    f"{'':12}   task-clustered paired bootstrap on the net discordance count"
                    f" (n_tasks={len(tasks)}): 95% CI [{lo}, {hi}];"
                    f" tasks net toward own-family permissiveness: {same_dir},"
                    f" opposite: {opp_dir}"
                )
            out.append("")

    out.append("malformed cross records (reported, never imputed):")
    for k in sorted(malformed):
        out.append(f"  {NAMES[k[0]]} {k[1]}: {malformed[k]}")

    summary = "\n".join(out)
    (ROOT / "data/mbpp_v2/cross_eval/cross_analysis_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
