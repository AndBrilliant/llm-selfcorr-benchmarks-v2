#!/usr/bin/env python3
"""Deterministic analysis for the fresh complete-response MBPP benchmark."""
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


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def holm_adjust(pairs: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(pairs, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p_value))
        adjusted[name] = running
    return adjusted


def pct(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else 100.0 * numerator / denominator


def decision(verdict: str | None) -> bool | None:
    return VERDICTS.get(verdict)


def analyze(rows: list[dict], model_order: list[str] = MODEL_ORDER) -> dict:
    by_model: dict[str, list[dict]] = OrderedDict((model, []) for model in model_order)
    for row in rows:
        model = row.get("model")
        if model in by_model:
            by_model[model].append(row)

    families = []
    far_p_pairs = []
    for model, model_rows in by_model.items():
        incorrect_total = sum(row.get("T") == 0 for row in model_rows)
        correct_total = sum(row.get("T") == 1 for row in model_rows)
        no_candidate = sum(row.get("code_sha256") is None for row in model_rows)
        no_truth = sum(row.get("T") is None for row in model_rows)
        malformed_selfcheck = sum(row.get("selfcheck_extraction") != "parsed" for row in model_rows)
        malformed_steelman = sum(row.get("steelman_extraction") != "parsed" for row in model_rows)

        paired = [
            row for row in model_rows
            if row.get("T") in (0, 1)
            and decision(row.get("selfcheck_verdict")) is not None
            and decision(row.get("steelman_verdict")) is not None
        ]
        sc_incorrect = [row for row in paired if row["T"] == 0 and decision(row["selfcheck_verdict"])]
        st_incorrect = [row for row in paired if row["T"] == 0 and decision(row["steelman_verdict"])]
        sc_correct = [row for row in paired if row["T"] == 1 and decision(row["selfcheck_verdict"])]
        st_correct = [row for row in paired if row["T"] == 1 and decision(row["steelman_verdict"])]

        sc_only_wrong = sum(
            row["T"] == 0 and decision(row["selfcheck_verdict"]) and not decision(row["steelman_verdict"])
            for row in paired
        )
        st_only_wrong = sum(
            row["T"] == 0 and not decision(row["selfcheck_verdict"]) and decision(row["steelman_verdict"])
            for row in paired
        )
        far_b = sc_only_wrong
        far_c = st_only_wrong
        far_p = exact_mcnemar_p(far_b, far_c)

        sc_error_st_correct = sum(
            decision(row["selfcheck_verdict"]) != (row["T"] == 1)
            and decision(row["steelman_verdict"]) == (row["T"] == 1)
            for row in paired
        )
        st_error_sc_correct = sum(
            decision(row["steelman_verdict"]) != (row["T"] == 1)
            and decision(row["selfcheck_verdict"]) == (row["T"] == 1)
            for row in paired
        )
        overall_p = exact_mcnemar_p(sc_error_st_correct, st_error_sc_correct)
        far_p_pairs.append((model, far_p))

        families.append({
            "model": model,
            "tasks": len(model_rows),
            "no_candidate": no_candidate,
            "no_truth": no_truth,
            "malformed_selfcheck": malformed_selfcheck,
            "malformed_steelman": malformed_steelman,
            "paired_evaluated": len(paired),
            "incorrect_total": incorrect_total,
            "correct_total": correct_total,
            "selfcheck_false_accepts": len(sc_incorrect),
            "steelman_false_accepts": len(st_incorrect),
            "selfcheck_true_accepts": len(sc_correct),
            "steelman_true_accepts": len(st_correct),
            "selfcheck_far_pct": pct(len(sc_incorrect), incorrect_total),
            "steelman_far_pct": pct(len(st_incorrect), incorrect_total),
            "selfcheck_true_acceptance_pct": pct(len(sc_correct), correct_total),
            "steelman_true_acceptance_pct": pct(len(st_correct), correct_total),
            "far_discordant_selfcheck_only": far_b,
            "far_discordant_steelman_only": far_c,
            "far_mcnemar_exact_p": far_p,
            "overall_discordant_corrected_by_steelman": sc_error_st_correct,
            "overall_discordant_introduced_by_steelman": st_error_sc_correct,
            "overall_mcnemar_exact_p": overall_p,
            "far_delta_percentage_points": (
                None if incorrect_total == 0
                else pct(len(st_incorrect), incorrect_total) - pct(len(sc_incorrect), incorrect_total)
            ),
        })

    adjusted = holm_adjust(far_p_pairs)
    for family in families:
        family["far_mcnemar_exact_holm_p"] = adjusted[family["model"]]
    return {"families": families}


def write_outputs(result: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "mbpp_v2_analysis.json"
    summary_path = out / "mbpp_v2_analysis_summary.txt"
    table_path = out / "mbpp_v2_analysis_table.tex"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "Fresh complete-response MBPP analysis",
        "=====================================",
        "Exact two-sided McNemar p-values; Holm adjustment across the five model families.",
        "",
    ]
    for family in result["families"]:
        lines.append(
            f"{family['model']}: tasks={family['tasks']} paired={family['paired_evaluated']} "
            f"incorrect={family['incorrect_total']} correct={family['correct_total']} "
            f"SC FAR={family['selfcheck_false_accepts']}/{family['incorrect_total']} "
            f"ST FAR={family['steelman_false_accepts']}/{family['incorrect_total']} "
            f"discordant={family['far_discordant_selfcheck_only']}/{family['far_discordant_steelman_only']} "
            f"exact p={family['far_mcnemar_exact_p']:.12g} Holm p={family['far_mcnemar_exact_holm_p']:.12g} "
            f"overall={family['overall_discordant_corrected_by_steelman']}/{family['overall_discordant_introduced_by_steelman']} "
            f"p={family['overall_mcnemar_exact_p']:.12g}"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\hline",
        "Model & Tasks & Incorrect & SC FAR & ST FAR & Discordant & Exact $p$ & Holm $p$ \\\\",
        "\\hline",
    ]
    for family in result["families"]:
        tex.append(
            f"{family['model']} & {family['tasks']} & {family['incorrect_total']} & "
            f"{family['selfcheck_false_accepts']}/{family['incorrect_total']} & "
            f"{family['steelman_false_accepts']}/{family['incorrect_total']} & "
            f"{family['far_discordant_selfcheck_only']}/{family['far_discordant_steelman_only']} & "
            f"{family['far_mcnemar_exact_p']:.6g} & {family['far_mcnemar_exact_holm_p']:.6g} \\\\"
        )
    tex.extend(["\\hline", "\\end{tabular}", ""])
    table_path.write_text("\n".join(tex), encoding="utf-8")

    manifest = out / "manifest.sha256"
    entries = []
    for path in sorted((json_path, summary_path, table_path)):
        entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", default="data/mbpp_v2/production")
    parser.add_argument("--out", default="data/mbpp_v2/analysis")
    args = parser.parse_args()

    rows = read_jsonl(ROOT / args.ledger / "derived_records.jsonl")
    result = analyze(rows)
    write_outputs(result, ROOT / args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
