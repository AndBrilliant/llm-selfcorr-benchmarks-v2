#!/usr/bin/env python3
"""Offline regression checks for the consolidated HumanEval artifacts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pc = load_module("scripts/humaneval/paired_contrasts.py", "paired_contrasts")


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2.0 * tail)


class TestHumanEvalAnalysis(unittest.TestCase):
    def test_paired_corpus_and_counts(self):
        rows = pc.load_paired_rows()
        self.assertEqual(len(rows), 164)
        self.assertEqual(sum(row["T"] == 0 for row in rows), 9)
        expected = [(6, 146), (3, 149), (0, 149)]
        for arm, (wrong_yes, correct_yes) in enumerate(expected):
            self.assertEqual(
                sum(row["acceptance"][arm] for row in rows if row["T"] == 0), wrong_yes
            )
            self.assertEqual(
                sum(row["acceptance"][arm] for row in rows if row["T"] == 1), correct_yes
            )

    def test_mcnemar_stratification_matches_frozen_ledger(self):
        rows = pc.load_paired_rows()
        incorrect = [row for row in rows if row["T"] == 0]
        selfcheck_only = sum(
            row["acceptance"][0] == 1 and row["acceptance"][1] == 0
            for row in incorrect
        )
        steelman_only = sum(
            row["acceptance"][0] == 0 and row["acceptance"][1] == 1
            for row in incorrect
        )
        self.assertEqual((selfcheck_only, steelman_only), (5, 2))
        self.assertEqual(mcnemar_exact(selfcheck_only, steelman_only), 0.453125)

        corrected = sum(
            row["acceptance"][0] != row["T"] and row["acceptance"][1] == row["T"]
            for row in rows
        )
        introduced = sum(
            row["acceptance"][1] != row["T"] and row["acceptance"][0] == row["T"]
            for row in rows
        )
        self.assertEqual((corrected, introduced), (10, 4))
        self.assertEqual(mcnemar_exact(corrected, introduced), 0.1795654296875)

    def test_koopman_intervals_match_archived_reference(self):
        cases = (
            (146, 155, 6, 9, 1.0653370093631995, 2.6679389574121513),
            (149, 155, 3, 9, 1.4848058684339014, 7.99965934558666),
        )
        for x1, n1, x0, n0, low, high in cases:
            with self.subTest(counts=(x1, n1, x0, n0)):
                actual = pc.koopman_ratio_interval(x1, n1, x0, n0)
                self.assertAlmostEqual(actual[0], low, places=12)
                self.assertAlmostEqual(actual[1], high, places=12)

    def test_execution_zero_cell_has_infinite_upper_endpoint(self):
        low, high = pc.koopman_ratio_interval(149, 155, 0, 9)
        self.assertAlmostEqual(math.log2(low), 1.6775025088397648, places=12)
        self.assertTrue(math.isinf(high))

    def test_inverse_ecdf_percentiles_use_all_draws(self):
        values = list(range(3900)) + [math.inf] * 100
        self.assertEqual(pc.percentile_interval(values), (99, 3899))
        with self.assertRaises(ValueError):
            pc.percentile_interval([1.0, math.nan, 2.0])

    def test_archived_extended_value_counts(self):
        output = json.loads((pc.DATA / "paired_contrasts.json").read_text())
        steelman = output["paired_contrasts"]["steelman_minus_selfcheck"]["E"]
        self.assertEqual(steelman["positive_infinity_draws"], 100)
        self.assertEqual(steelman["undefined_draws"], 0)
        self.assertEqual(
            steelman["bootstrap_percentile_ci95"],
            [-0.7082066011847826, 3.2591923395394],
        )
        execution = output["paired_contrasts"]["execution_minus_steelman"]["E"]
        self.assertEqual(execution["undefined_draws"], 100)
        self.assertIsNone(execution["bootstrap_percentile_ci95"])

    def test_manifest_hashes(self):
        manifest = pc.DATA / "MANIFEST.sha256"
        for line in manifest.read_text().splitlines():
            expected, relative = line.split(maxsplit=1)
            path = pc.DATA / relative.strip()
            self.assertTrue(path.is_file(), relative)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_no_legacy_mbpp_outputs_or_analysis(self):
        for relative in ("data/mbpp", "data/mbpp_paired", "data/mbpp_control"):
            self.assertFalse((ROOT / relative).exists(), relative)
        legacy_scripts = {
            "harness_paired.py",
            "harness_control.py",
            "verdict_parsing.py",
            "mbpp_multiplicity.py",
            "mbpp_verdict_audit.py",
            "mbpp_control_semantic_audit.py",
            "mbpp_first_record_sensitivity.py",
            "association_analysis.py",
            "final_table_paired.py",
        }
        for name in legacy_scripts:
            self.assertFalse((ROOT / "scripts" / name).exists(), name)
        mbpp = json.loads((ROOT / "data/MBPP.jsonl").read_text().splitlines()[0])
        self.assertIn("task_id", mbpp)
        self.assertNotIn("accept_sc", mbpp)
        self.assertNotIn("accept_st", mbpp)


if __name__ == "__main__":
    unittest.main()
