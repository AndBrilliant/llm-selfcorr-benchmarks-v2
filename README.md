# llm-selfcorr-benchmarks-v2

Consolidated data and analysis repository for *Limits of Self-Correction in LLMs: An
Information-Theoretic Analysis of Correlated Errors* (A. M. Brilliant).

## Migration checkpoint

This repository carries forward the complete preserved HumanEval experiment and its offline
analysis. The earlier MBPP candidates, evaluator responses, verdicts, harnesses, parsers, and
analyses are **not included**. The prior repository remains intact separately as the historical
record.

`data/MBPP.jsonl` contains only the original 947 MBPP task definitions. It contains no generated
candidate, evaluator decision, or result. A fresh MBPP protocol and new experiment will be added
only after preservation and reconstruction tests pass on a pilot.

## Contents

| Artifact | Design | Location |
|---|---|---|
| HumanEval corpus | 164 public task definitions | `data/HumanEval.jsonl` |
| HumanEval frozen candidates and three evaluator arms | One immutable candidate per task, 155 correct and 9 incorrect | `data/humaneval/data/` |
| HumanEval context comparison | Preserved same-context records | `data/humaneval/context/` |
| HumanEval audit guide | Manifest, validation commands, and protocol | `data/humaneval/REPRODUCIBILITY.md` |
| HumanEval paired analysis | Koopman score intervals and task-paired bootstrap | `scripts/humaneval/paired_contrasts.py` |
| MBPP task definitions only | 947 tasks; no experimental outputs | `data/MBPP.jsonl` |

The companion [`human-eval-selfcorr`](https://github.com/AndBrilliant/human-eval-selfcorr)
repository contains the full HumanEval certifier and regression suite.

## Offline validation

From this repository's root:

```bash
(cd data/humaneval && sha256sum -c MANIFEST.sha256)
python3 tests/test_humaneval_analysis.py
python3 scripts/humaneval/paired_contrasts.py
python3 tests/test_humaneval_analysis.py
```

These commands make no model API calls. The paired analysis requires NumPy and SciPy. Running it
must regenerate the archived JSON and summary byte-identically.

## Scope

HumanEval measures evaluator acceptance of frozen candidates, not candidate repair. The execution
arm is an oracle-grounded positive control. The nine-incorrect-candidate stratum limits precision.
No fresh MBPP empirical claim is made at this migration checkpoint.
