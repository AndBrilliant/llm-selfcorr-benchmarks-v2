#!/usr/bin/env python3
"""Compare production Qwen with the one-assert-visible Qwen control.

Rates use each ledger's paired stratum (ground truth assigned and a parsed
terminal verdict from both evaluator arms). Malformed records are counted, never
imputed. Candidates are regenerated in the control, so comparisons are between
arms/ledgers, not candidate-paired effects.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V = {"ACCEPT", "REJECT"}


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return c-h, c+h


def load(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("model") == "qwen38-flash"]
    paired = [r for r in rows if r.get("T") in (0, 1)
              and r.get("selfcheck_verdict") in V and r.get("steelman_verdict") in V]
    malformed = sum(1 for r in rows if r.get("T") in (0, 1)
                    and (r.get("selfcheck_verdict") not in V
                         or r.get("steelman_verdict") not in V))
    return rows, paired, malformed


def rates(paired):
    out = {}
    for arm in ["selfcheck", "steelman"]:
        k0 = sum(1 for r in paired if r["T"] == 0 and r[f"{arm}_verdict"] == "ACCEPT")
        n0 = sum(1 for r in paired if r["T"] == 0)
        k1 = sum(1 for r in paired if r["T"] == 1 and r[f"{arm}_verdict"] == "ACCEPT")
        n1 = sum(1 for r in paired if r["T"] == 1)
        out[arm] = (k0, n0, k1, n1)
    return out


def main():
    datasets = [("production Qwen", ROOT / "data/mbpp_v2/production/derived_records.jsonl"),
                ("assert-visible Qwen", ROOT / "data/mbpp_v2/assert_visible_qwen/derived_records.jsonl")]
    lines = ["One-assert-visible Qwen control", "===============================", ""]
    for name, path in datasets:
        rows, paired, malformed = load(path)
        assigned = [r for r in rows if r.get("T") in (0, 1)]
        acc = sum(r["T"] for r in assigned) / len(assigned)
        lines.append(f"{name}: attempted={len(rows)}, ground-truth assigned={len(assigned)}, "
                     f"generation accuracy={100*acc:.1f}%, paired={len(paired)}, "
                     f"malformed-or-unpaired={malformed}")
        for arm, (k0, n0, k1, n1) in rates(paired).items():
            w0, w1 = wilson(k0, n0), wilson(k1, n1)
            lines.append(f"  {arm}: FAR={100*k0/n0:.1f}% [{100*w0[0]:.1f},{100*w0[1]:.1f}] "
                         f"({k0}/{n0}); TAR={100*k1/n1:.1f}% [{100*w1[0]:.1f},{100*w1[1]:.1f}] "
                         f"({k1}/{n1})")
        lines.append("")
    out = ROOT / "data/mbpp_v2/assert_visible_qwen/assert_visible_analysis_summary.txt"
    out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
