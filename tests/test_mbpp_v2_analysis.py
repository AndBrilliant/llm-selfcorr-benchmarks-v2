#!/usr/bin/env python3
"""Regression tests for the fresh MBPP deterministic analysis."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mbpp_v2_analysis", ROOT / "scripts/mbpp_v2_analysis.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class TestFreshMBPPAnalysis(unittest.TestCase):
    def row(self, model, task_id, T, sc, st):
        return {
            "model": model,
            "task_id": task_id,
            "T": T,
            "selfcheck_verdict": sc,
            "selfcheck_extraction": "parsed" if sc in {"ACCEPT", "REJECT"} else "malformed_or_error",
            "steelman_verdict": st,
            "steelman_extraction": "parsed" if st in {"ACCEPT", "REJECT"} else "malformed_or_error",
        }

    def test_fixed_denominators_and_exact_discordance(self):
        rows = [
            self.row("m1", "t1", 0, "ACCEPT", "REJECT"),
            self.row("m1", "t2", 0, "ACCEPT", "ACCEPT"),
            self.row("m1", "t3", 0, "REJECT", "ACCEPT"),
            self.row("m1", "t4", 1, "ACCEPT", "REJECT"),
            self.row("m1", "t5", 1, "ACCEPT", "ACCEPT"),
            self.row("m1", "t6", None, "ACCEPT", "ACCEPT"),
        ]
        result = analysis.analyze(rows, ["m1"])
        family = result["families"][0]
        self.assertEqual(family["tasks"], 6)
        self.assertEqual(family["incorrect_total"], 3)
        self.assertEqual(family["correct_total"], 2)
        self.assertEqual(family["paired_evaluated"], 5)
        self.assertEqual(family["selfcheck_false_accepts"], 2)
        self.assertEqual(family["steelman_false_accepts"], 2)
        self.assertEqual(family["far_discordant_selfcheck_only"], 1)
        self.assertEqual(family["far_discordant_steelman_only"], 1)
        self.assertEqual(family["far_mcnemar_exact_p"], 1.0)
        self.assertEqual(family["overall_discordant_corrected_by_steelman"], 1)
        self.assertEqual(family["overall_discordant_introduced_by_steelman"], 2)
        self.assertEqual(family["overall_mcnemar_exact_p"], 1.0)

    def test_holm_adjustment_is_monotone(self):
        adjusted = analysis.holm_adjust([("a", 0.01), ("b", 0.02), ("c", 0.03)])
        self.assertEqual(adjusted["a"], 0.03)
        self.assertEqual(adjusted["b"], 0.04)
        self.assertEqual(adjusted["c"], 0.04)

    def test_exact_mcnemar_known_cells(self):
        self.assertEqual(analysis.exact_mcnemar_p(5, 2), 0.453125)
        self.assertEqual(analysis.exact_mcnemar_p(10, 4), 0.1795654296875)
        self.assertEqual(analysis.exact_mcnemar_p(0, 0), 1.0)

    def test_missing_verdict_is_visible_not_a_rejection(self):
        row = self.row("m1", "t1", 0, "ACCEPT", "not_a_verdict")
        result = analysis.analyze([row], ["m1"])
        family = result["families"][0]
        self.assertEqual(family["malformed_steelman"], 1)
        self.assertEqual(family["paired_evaluated"], 0)
        self.assertEqual(family["steelman_false_accepts"], 0)
        self.assertEqual(family["steelman_far_pct"], 0.0)

    def test_write_outputs_are_manifested(self):
        import tempfile

        result = analysis.analyze([self.row("m1", "t1", 0, "ACCEPT", "REJECT")], ["m1"])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            analysis.write_outputs(result, out)
            manifest = (out / "manifest.sha256").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(manifest), 3)
            for line in manifest:
                digest, name = line.split(maxsplit=1)
                actual = analysis.hashlib.sha256((out / name).read_bytes()).hexdigest()
                self.assertEqual(actual, digest)


if __name__ == "__main__":
    unittest.main()
