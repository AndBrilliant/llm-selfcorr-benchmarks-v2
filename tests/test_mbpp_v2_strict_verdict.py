#!/usr/bin/env python3
"""Regression tests for the strict fresh-MBPP verdict parser."""
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mbpp_v2_pilot", ROOT / "scripts/mbpp_v2_pilot.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
strict_steelman_verdict = module.strict_verdict


class TestStrictSteelmanVerdict(unittest.TestCase):
    def test_accepts_single_terminal_verdict(self):
        self.assertEqual(
            strict_steelman_verdict("Analysis text.\n\nVERDICT: ACCEPT\n"),
            "ACCEPT",
        )
        self.assertEqual(
            strict_steelman_verdict("Analysis text.\nVERDICT: REJECT"),
            "REJECT",
        )

    def test_accepts_case_insensitive_verdict(self):
        self.assertEqual(
            strict_steelman_verdict("Analysis.\nverdict: accept"),
            "ACCEPT",
        )

    def test_rejects_missing_verdict(self):
        self.assertIsNone(strict_steelman_verdict("Analysis only."))
        self.assertIsNone(strict_steelman_verdict(""))
        self.assertIsNone(strict_steelman_verdict(None))

    def test_rejects_nonterminal_verdict(self):
        self.assertIsNone(
            strict_steelman_verdict("VERDICT: ACCEPT\nTrailing explanation.")
        )

    def test_rejects_multiple_verdict_markers(self):
        self.assertIsNone(
            strict_steelman_verdict("VERDICT: ACCEPT\nAnalysis.\nVERDICT: REJECT")
        )

    def test_rejects_suffix_collision(self):
        self.assertIsNone(strict_steelman_verdict("Analysis.\nVERDICT: ACCEPTABLE"))

    def test_rejects_prefix_collision(self):
        self.assertIsNone(
            strict_steelman_verdict("Analysis.\nNOT VERDICT: ACCEPT")
        )


if __name__ == "__main__":
    unittest.main()
