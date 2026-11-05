# HumanEval reproduction guide

This guide audits the completed HumanEval experiment without making API calls. The preserved
corpus is the experiment; regeneration is a replication, not reproduction.

## Experimental invariants

- **N = 164** canonical HumanEval tasks.
- The same frozen candidate is judged by every evaluator arm. Arms never regenerate, repair, or
  revise candidates.
- Each Stage-0 candidate was extracted, hashed, and persisted before test execution.
- Stage-0 correctness was scored once, frozen, and never modified. There are 155 correct and nine
  incorrect candidates.
- Evaluator raw responses were persisted before parsing. A frozen response is never resampled.
- No correction or revision phase exists in the experiment.

## Preserved counts

| Arm | Incorrect accepted | Correct accepted |
|---|---:|---:|
| Pure self-check | 6/9 | 146/155 |
| Structured steelman | 3/9 | 149/155 |
| Execution-grounded positive control | 0/9 | 149/155 |

The execution evaluator receives a binary signal from the reference-test mechanism defining
correctness. It is an oracle-grounded positive control, not independent validation. The signal is
an input to a fallible evaluator, not a mechanically copied acceptance decision; six correct
candidates receive NO despite PASS.

## Hash verification

From the consolidated repository root:

```bash
(cd data/humaneval && sha256sum -c MANIFEST.sha256)
python3 tests/test_humaneval_analysis.py
```

The manifest covers the frozen candidates, Stage-0 labels, full evaluator responses, parsed
ledgers, execution signals, and execution-signal certification. The tests also verify the paired
counts, exact McNemar strata, score intervals, bootstrap behavior, and absence of legacy MBPP
outputs from this repository.

## Full certifier suite

The complete HumanEval certifier and regression suite lives in the companion
[`human-eval-selfcorr`](https://github.com/AndBrilliant/human-eval-selfcorr) repository. The
audited companion revision is commit `7194b02`. From that repository's root:

```bash
python3 validate_baseline.py
python3 arms/selfcheck/validate_selfcheck.py
python3 arms/steelman/validate_steelman.py
python3 arms/execution/validate_execution.py

python3 tests/test_stage0.py                    # 29 tests
python3 tests/test_arm1_selfcheck.py            # 19 tests
python3 tests/test_arm2_steelman.py             # 24 tests
python3 tests/test_arm3_execution.py            # 54 tests
```

`arms/execution/validate_execution_signals.py` is audit-safe but rewrites the certification file
with a fresh timestamp. Do not use it when byte-identical preservation is required.

## Paired uncertainty analysis

From this repository's root:

```bash
python3 scripts/humaneval/paired_contrasts.py
python3 tests/test_humaneval_analysis.py
```

The analysis joins the three verdict ledgers to `baseline.jsonl` by task ID and verifies the
candidate hash in every arm. It computes two-sided 95% Koopman score intervals for `tau/kappa`
with the Miettinen--Nurminen `N/(N-1)` correction and transforms them to acceptance evidence in
bits. No pseudocount is added. The execution arm's zero false-acceptance cell therefore gives an
infinite upper endpoint; 1.68 bits is the lower endpoint of the two-sided interval, not a separate
one-sided bound.

Between-arm contrasts use 4,000 task-level paired bootstrap resamples with seed `20261104`.
Incorrect and correct tasks are resampled separately, fixing the denominators at nine and 155.
Inverse-empirical-CDF percentiles retain infinite and undefined draws. Execution-arm paired
intervals are unavailable when the distribution is degenerate at infinity or contains undefined
contrasts. These intervals address task resampling, not repeated-call variability.

Outputs are `paired_contrasts.json` and `paired_contrasts_summary.txt` in this directory. Both
regenerate byte-identically. NumPy and SciPy are required; no network or model API is used.

## Context comparison

`context/context_same.jsonl` preserves the separate same-context comparison. It uses related but
not identical prompt and parsing conditions and does not isolate context alone. Its nine-error
stratum does not support a general claim about fresh-context effects.

## Replication commands

Generation and evaluator replication commands are preserved in the companion
`human-eval-selfcorr` repository. They are not needed for this audit and are not run by any command
above.
