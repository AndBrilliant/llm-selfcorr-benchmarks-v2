#!/usr/bin/env python3
"""Neutral-verification control-arm analysis.

One neutral, non-adversarial step-by-step verification call per production
pair (same frozen candidates, same strict terminal verdict grammar, no output
cap; prompt in mbpp_v2_cross_control.CONTROL_PROMPT). Purpose: separate the
steelman arm's adversarial content from added deliberation/re-engagement
(matched-computation control for review item G).

Denominators: triple-paired stratum per family -- ground truth assigned and a
parsed terminal verdict from all three arms (selfcheck, steelman, control).
Malformed records are reported, never imputed.

Outputs a fixed-width summary to data/mbpp_v2/control/control_analysis_summary.txt.
"""
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V = {"ACCEPT": True, "REJECT": False}
MODELS = ["gpt-5.4", "ds-v4-flash", "glm-5.3-flash", "qwen38-flash", "gpt-6-luna"]


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
    x = min(b, c)
    from math import comb
    p = sum(comb(n, i) for i in range(0, x + 1)) / 2**n
    return min(1.0, 2 * p)


def main():
    prod = {}
    for l in (ROOT / "data/mbpp_v2/production/derived_records.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            prod[(r["model"], r["task_id"])] = r

    ctrl, ctrl_malformed = {}, defaultdict(int)
    for l in (ROOT / "data/mbpp_v2/control/control_records.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r["control_verdict"] in V:
            ctrl[(r["model"], r["task_id"])] = r["control_verdict"]
        else:
            ctrl_malformed[r["model"]] += 1

    steel_tokens = {}
    for l in (ROOT / "data/mbpp_v2/production/events.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        e = json.loads(l)
        if e.get("stage") == "steelman" and e.get("usage"):
            steel_tokens[e["event_id"]] = (
                e["usage"].get("completion_tokens") or e["usage"].get("output_tokens") or 0
            )
    ctrl_tokens = {}
    for l in (ROOT / "data/mbpp_v2/control/events.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        e = json.loads(l)
        if e.get("stage") == "control" and e.get("usage") and e.get("finish_reason") == "stop":
            ctrl_tokens[(e["model"], e["task_id"])] = (
                e["usage"].get("completion_tokens") or e["usage"].get("output_tokens") or 0
            )

    out = []
    out.append("Neutral-verification control-arm analysis")
    out.append("=========================================")
    out.append("Triple-paired stratum: ground truth assigned, parsed terminal verdict from")
    out.append("all three arms (selfcheck, steelman, control). Malformed control records are")
    out.append("reported as counts, never imputed. GPT-5.4's OpenAI key was invalidated")
    out.append("mid-run; its 747 auth-failed tasks stay visible as malformed records")
    out.append("(repair_log.jsonl), leaving n=200 triple-paired for that family.")
    out.append("")
    out.append(f"{'family':14} {'arm':10} {'n_pair':>6} {'FAR% [95% CI]':>22} {'TAR% [95% CI]':>22}")

    for m in MODELS:
        trip = sorted(
            t
            for (mm, t), v in ctrl.items()
            if mm == m
            and (m, t) in prod
            and prod[(m, t)].get("T") in (0, 1)
            and prod[(m, t)].get("selfcheck_verdict") in V
            and prod[(m, t)].get("steelman_verdict") in V
        )
        for arm in ("selfcheck", "steelman", "control"):
            if arm == "control":
                acc = {t: V[ctrl[(m, t)]] for t in trip}
            else:
                key = f"{arm}_verdict"
                acc = {t: V[prod[(m, t)][key]] for t in trip}
            k0 = sum(1 for t in trip if prod[(m, t)]["T"] == 0 and acc[t])
            n0 = sum(1 for t in trip if prod[(m, t)]["T"] == 0)
            k1 = sum(1 for t in trip if prod[(m, t)]["T"] == 1 and acc[t])
            n1 = sum(1 for t in trip if prod[(m, t)]["T"] == 1)
            w0, w1 = wilson(k0, n0), wilson(k1, n1)
            out.append(
                f"{m:14} {arm:10} {n0+n1:>6} "
                f"{100*k0/n0:6.1f} [{100*w0[0]:.1f},{100*w0[1]:.1f}]".rjust(28)
                + f"   {100*k1/n1:6.1f} [{100*w1[0]:.1f},{100*w1[1]:.1f}]"
            )
        for T in (0, 1):
            b = sum(
                1
                for t in trip
                if prod[(m, t)]["T"] == T and V[ctrl[(m, t)]] and not V[prod[(m, t)]["steelman_verdict"]]
            )
            c = sum(
                1
                for t in trip
                if prod[(m, t)]["T"] == T and not V[ctrl[(m, t)]] and V[prod[(m, t)]["steelman_verdict"]]
            )
            out.append(f"{'':14} control-vs-steelman T={T}: b/c = {b}/{c}  exact p = {mcnemar(b,c):.6g}")
        st = [steel_tokens[prod[(m, t)]["steelman_event_id"]] for t in trip
              if prod[(m, t)].get("steelman_event_id") in steel_tokens]
        ct = [ctrl_tokens[(m, t)] for t in trip if (m, t) in ctrl_tokens]
        if st and ct:
            out.append(
                f"{'':14} realized completion tokens: steelman med {statistics.median(st):.0f}/mean "
                f"{statistics.mean(st):.0f}; control med {statistics.median(ct):.0f}/mean "
                f"{statistics.mean(ct):.0f} (control/steelman mean ratio "
                f"{statistics.mean(ct)/statistics.mean(st):.2f}x, n={len(ct)})"
            )
        out.append(
            f"{'':14} malformed control records: {ctrl_malformed[m]} (reported, never imputed)"
        )
        out.append("")

    summary = "\n".join(out)
    (ROOT / "data/mbpp_v2/control/control_analysis_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
