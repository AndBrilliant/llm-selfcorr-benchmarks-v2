#!/usr/bin/env python3
"""Reproduce HumanEval score intervals and paired between-arm contrasts.

The three frozen evaluator arms judge the same candidates. For each contrast, we
therefore resample task IDs, not independent arm-level outcomes. Incorrect and
correct tasks are resampled separately so that both denominators remain fixed at
9 and 155. Percentile intervals use 4,000 resamples and seed 20261104.

No pseudocount or continuity correction is added. In particular, a zero false-
acceptance cell makes the point estimate of E infinite. Such resamples are kept
and summarized explicitly rather than discarded.

Marginal E intervals are transformed two-sided 95% Koopman score intervals for
R = tau/kappa, with the Miettinen--Nurminen finite-sample correction. The score
interval treats the incorrect and correct strata as independent binomial samples;
it is a marginal interval, not a paired between-arm interval. The execution-arm
upper endpoint is infinite. Thus 1.68 bits is the lower endpoint of a two-sided
95% interval, not a separately constructed one-sided 95% lower confidence bound.

Run from any directory:
    python3 scripts/humaneval/paired_contrasts.py

Outputs, under data/humaneval/:
    paired_contrasts.json
    paired_contrasts_summary.txt

Requires NumPy and SciPy. No network access or model API calls are used.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "humaneval"
RAW = DATA / "data"
ARM_FILES = (
    ("selfcheck", "arm1_selfcheck.jsonl"),
    ("steelman", "arm2_steelman.jsonl"),
    ("execution", "arm3_execution.jsonl"),
)
BOOTSTRAPS = 4000
SEED = 20261104


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_task(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    result = {row["task_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate task_id in {path}")
    return result


def load_paired_rows() -> list[dict]:
    baseline = by_task(RAW / "baseline.jsonl")
    arms = {name: by_task(RAW / filename) for name, filename in ARM_FILES}
    ids = sorted(baseline, key=lambda task: baseline[task]["task_index"])
    if len(ids) != 164:
        raise ValueError(f"Expected 164 tasks, found {len(ids)}")

    paired = []
    for task_id in ids:
        base = baseline[task_id]
        acceptance = []
        for name, _ in ARM_FILES:
            row = arms[name].get(task_id)
            if row is None:
                raise ValueError(f"Missing {name} verdict for {task_id}")
            if row.get("candidate_sha256") != base.get("candidate_sha256"):
                raise ValueError(f"Candidate hash mismatch for {name}, {task_id}")
            if row.get("parse_status") != "valid" or row.get("acceptance") not in (0, 1):
                raise ValueError(f"Invalid {name} verdict for {task_id}")
            acceptance.append(int(row["acceptance"]))
        paired.append({
            "task_id": task_id,
            "T": int(base["baseline_correct"]),
            "acceptance": acceptance,
        })

    if sum(row["T"] for row in paired) != 155:
        raise ValueError("Expected 155 correct and 9 incorrect candidates")
    return paired


def rates(rows: list[dict], arm: int) -> tuple[float, float]:
    incorrect = [row for row in rows if row["T"] == 0]
    correct = [row for row in rows if row["T"] == 1]
    kappa = sum(row["acceptance"][arm] for row in incorrect) / len(incorrect)
    tau = sum(row["acceptance"][arm] for row in correct) / len(correct)
    return kappa, tau


def evidence(kappa: float, tau: float) -> float:
    if kappa == 0.0 and tau > 0.0:
        return math.inf
    if tau == 0.0 and kappa > 0.0:
        return -math.inf
    if kappa == 0.0 or tau == 0.0:
        return math.nan
    return math.log2(tau / kappa)


def koopman_ratio_interval(
    correct_accepted: int,
    n_correct: int,
    incorrect_accepted: int,
    n_incorrect: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Two-sided Koopman score interval for R = tau/kappa.

    This is the cubic-inversion implementation of the constrained score test.
    The z-squared multiplier N/(N-1) is the Miettinen--Nurminen correction.
    """
    if not (0 < correct_accepted <= n_correct and 0 <= incorrect_accepted <= n_incorrect):
        raise ValueError("Score-interval counts are outside their supported range")
    x = correct_accepted + incorrect_accepted
    n = n_correct + n_incorrect
    z2 = norm.isf(alpha / 2) ** 2 * n / (n - 1)
    coefficients = [
        n_incorrect * (n_incorrect * n * correct_accepted
                       + n_correct * (n_incorrect + correct_accepted) * z2),
        -n_incorrect * (n_incorrect * n_correct * x
                        + 2 * n * incorrect_accepted * correct_accepted
                        + n_correct * (n_incorrect + incorrect_accepted
                                       + 2 * correct_accepted) * z2),
        2 * n_incorrect * n_correct * incorrect_accepted * x
        + n * incorrect_accepted**2 * correct_accepted
        + n_incorrect * n_correct * x * z2,
        -n_correct * incorrect_accepted**2 * x,
    ]
    roots = np.roots(coefficients)
    real_roots = sorted(
        root.real for root in roots
        if abs(root.imag) < 1e-8 and -1e-9 <= root.real <= 1 + 1e-9
    )
    if len(real_roots) != 3:
        raise ArithmeticError(f"Could not identify three score roots: {roots}")
    # Koopman's inversion uses the two smallest real roots, in descending order.
    low_p, high_p = real_roots[1], real_roots[0]

    def ratio_at(p: float) -> float:
        denominator = incorrect_accepted + n_correct - n * p
        numerator = 1 - (n_correct - correct_accepted) * (1 - p) / denominator
        return numerator / p

    lower = ratio_at(low_p)
    upper = math.inf if incorrect_accepted == 0 else ratio_at(high_p)
    return lower, upper


def percentile_interval(values: list[float]) -> tuple[float, float]:
    """Inverse-empirical-CDF 95% percentiles over all supplied draws.

    Undefined contrasts are rejected rather than silently conditioning the
    interval on a subset. Infinite extended-real draws remain in the ordering.
    """
    if any(math.isnan(value) for value in values):
        raise ValueError("Cannot compute a full-draw percentile interval with undefined draws")
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile interval with no draws")
    lower_index = math.ceil(0.025 * len(ordered)) - 1
    upper_index = math.ceil(0.975 * len(ordered)) - 1
    return ordered[max(0, lower_index)], ordered[max(0, upper_index)]


def json_number(value: float):
    if math.isnan(value):
        return "undefined"
    if value == math.inf:
        return "Infinity"
    if value == -math.inf:
        return "-Infinity"
    return value


def json_interval(interval: tuple[float, float]) -> list:
    return [json_number(value) for value in interval]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    rows = load_paired_rows()
    names = [name for name, _ in ARM_FILES]
    marginal = {}
    for arm, name in enumerate(names):
        kappa, tau = rates(rows, arm)
        n0 = sum(row["T"] == 0 for row in rows)
        n1 = sum(row["T"] == 1 for row in rows)
        x0 = round(kappa * n0)
        x1 = round(tau * n1)
        ratio_interval = koopman_ratio_interval(x1, n1, x0, n0)
        e_interval = tuple(math.log2(value) for value in ratio_interval)
        marginal[name] = {
            "incorrect_accepted": x0,
            "n_incorrect": n0,
            "correct_accepted": x1,
            "n_correct": n1,
            "kappa": kappa,
            "tau": tau,
            "E_bits": json_number(evidence(kappa, tau)),
            "ratio_score_ci95": json_interval(ratio_interval),
            "E_score_ci95_bits": json_interval(e_interval),
        }

    # Guard the manuscript's rounded intervals and its zero-cell interpretation.
    for name, expected in {
        "selfcheck": (0.09130989, 1.41572566),
        "steelman": (0.57027432, 2.99993857),
        "execution": (1.67750251, math.inf),
    }.items():
        actual = marginal[name]["E_score_ci95_bits"]
        assert abs(actual[0] - expected[0]) < 1e-7, (name, actual)
        if math.isinf(expected[1]):
            assert actual[1] == "Infinity"
        else:
            assert abs(actual[1] - expected[1]) < 1e-7, (name, actual)

    rng = random.Random(SEED)
    incorrect = [row for row in rows if row["T"] == 0]
    correct = [row for row in rows if row["T"] == 1]
    contrasts = ((1, 0), (2, 0), (2, 1))
    draws = {(pair, metric): [] for pair in contrasts for metric in ("kappa", "tau", "E")}
    for _ in range(BOOTSTRAPS):
        sample = [incorrect[rng.randrange(len(incorrect))] for _ in incorrect]
        sample += [correct[rng.randrange(len(correct))] for _ in correct]
        arm_values = []
        for arm in range(3):
            kappa, tau = rates(sample, arm)
            arm_values.append((kappa, tau, evidence(kappa, tau)))
        for pair in contrasts:
            for metric_index, metric in enumerate(("kappa", "tau", "E")):
                later, earlier = pair
                draws[pair, metric].append(
                    arm_values[later][metric_index] - arm_values[earlier][metric_index]
                )

    paired = {}
    for later, earlier in contrasts:
        key = f"{names[later]}_minus_{names[earlier]}"
        k_later, t_later = rates(rows, later)
        k_earlier, t_earlier = rates(rows, earlier)
        points = (
            k_later - k_earlier,
            t_later - t_earlier,
            evidence(k_later, t_later) - evidence(k_earlier, t_earlier),
        )
        paired[key] = {}
        for metric, point in zip(("kappa", "tau", "E"), points):
            values = draws[(later, earlier), metric]
            undefined = sum(math.isnan(value) for value in values)
            degenerate_infinite = len(set(values)) == 1 and math.isinf(values[0])
            interval_available = undefined == 0 and not degenerate_infinite
            paired[key][metric] = {
                "point": json_number(point),
                "bootstrap_percentile_ci95": (
                    json_interval(percentile_interval(values)) if interval_available else None
                ),
                "interval_status": (
                    "inverse-ECDF percentile over all 4000 draws" if interval_available
                    else "unavailable: undefined contrasts or a degenerate infinite distribution"
                ),
                "positive_infinity_draws": sum(value == math.inf for value in values),
                "negative_infinity_draws": sum(value == -math.inf for value in values),
                "undefined_draws": undefined,
            }

    output = {
        "design": "Three arms paired on the same 164 frozen HumanEval candidates",
        "bootstrap": {
            "resamples": BOOTSTRAPS,
            "seed": SEED,
            "method": "task-level paired inverse-ECDF percentile bootstrap, stratified by T",
            "stratum_sizes": {"T0": len(incorrect), "T1": len(correct)},
            "zero_cells": "retained; no pseudocount or continuity correction",
            "coverage": "nominal 95%, unadjusted for multiplicity",
        },
        "marginal_interval_method": (
            "two-sided 95% Koopman score interval for tau/kappa with "
            "Miettinen-Nurminen N/(N-1) correction, transformed by log2"
        ),
        "marginal": marginal,
        "paired_contrasts": paired,
    }
    json_path = DATA / "paired_contrasts.json"
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")

    lines = [
        "HumanEval frozen-candidate uncertainty analysis",
        "===============================================",
        f"Paired stratified inverse-ECDF bootstrap: B={BOOTSTRAPS}, seed={SEED}; no pseudocounts.",
        "Marginal intervals: two-sided 95% Koopman score, Miettinen-Nurminen correction.",
        "",
        "Marginal E intervals (bits):",
    ]
    for name in names:
        interval = marginal[name]["E_score_ci95_bits"]
        lines.append(f"  {name}: {interval}")
    lines += ["", "Paired contrasts (point; 95% percentile interval):"]
    for key, metrics in paired.items():
        lines.append(f"  {key}")
        for metric in ("kappa", "tau", "E"):
            result = metrics[metric]
            lines.append(
                f"    {metric}: {result['point']}; "
                f"{result['bootstrap_percentile_ci95']}; "
                f"+inf draws={result['positive_infinity_draws']}, "
                f"undefined={result['undefined_draws']}"
            )
    lines += [
        "",
        "Execution-arm paired E intervals are unavailable when contrasts are undefined",
        "or the bootstrap distribution is degenerate at infinity; no draws are dropped.",
        "The execution-arm marginal E lower endpoint is 1.6775 bits (reported as 1.68).",
        "Its upper endpoint is infinite; this is not a separate one-sided 95% LCB.",
        "Nominal intervals describe representative/exchangeable task inference;",
        "they are not uncertainty about the realized fixed corpus itself.",
    ]
    summary_path = DATA / "paired_contrasts_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    for path in (json_path, summary_path):
        print(f"{sha256(path)}  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
