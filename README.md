# llm-selfcorr-benchmarks-v2

Consolidated data and analysis repository for *Limits of Self-Correction in LLMs: An
Information-Theoretic Analysis of Correlated Errors* (A. M. Brilliant).

This repository is the manuscript's reproducibility anchor. It contains the complete
preserved HumanEval experiment, the fresh complete-response MBPP benchmark (production
run plus repeated-evaluation, candidate-replication, moderator, and baseline ledgers),
every deterministic analysis, validators, regression tests, and SHA-256 manifests over
all artifacts.

## Contents

| Artifact | Design | Location |
|---|---|---|
| HumanEval corpus | 164 public task definitions | `data/HumanEval.jsonl` |
| HumanEval frozen candidates and three evaluator arms | One immutable candidate per task, 155 correct and 9 incorrect | `data/humaneval/data/` |
| HumanEval context comparison | Preserved same-context records | `data/humaneval/context/` |
| HumanEval audit guide | Manifest, validation commands, and protocol | `data/humaneval/REPRODUCIBILITY.md` |
| HumanEval paired analysis | Koopman score intervals and task-paired bootstrap | `scripts/humaneval/paired_contrasts.py` |
| MBPP task definitions | 947 tasks | `data/MBPP.jsonl` |
| MBPP production ledger | 4,735 pairs x 5 families, complete-response protocol, immutable event ledger, 67 restored provider-ceiling malformed records | `data/mbpp_v2/production/` |
| MBPP paired analysis | Exact McNemar, five-family Holm, paired-stratum FAR/TAR rates, Wilson CIs | `data/mbpp_v2/analysis/` (`scripts/mbpp_v2_analysis.py`) |
| MBPP moderator analysis | Pre-registered stratification (spec: `scripts/mbpp_v2_moderator_spec.md`) | `data/mbpp_v2/moderator_analysis/` |
| MBPP repeated evaluation | k=3 rounds per arm, 4,718 pairs, 18,884 events | `data/mbpp_v2/repeated_eval/` |
| MBPP candidate replication | Second independent greedy candidate per (task, model), 4,735 pairs | `data/mbpp_v2/replication/` |
| Tier 1 analyses | Verdict stability, repeated-acceptance error rates, cross-candidate association, always-accept baseline | `data/mbpp_v2/tier1_analysis/` (`scripts/mbpp_v2_tier1_analysis.py`) |
| Ledger validator | Duplicates, hashes, verdict reproduction from raw text, manifest verification | `scripts/validate_mbpp_v2_pilot.py` |

The historical pre-v2 MBPP run (broken-storage era) is retained intact separately in the
prior repository and is not part of this benchmark; the manuscript reports it only as
superseded record.

The companion [`human-eval-selfcorr`](https://github.com/AndBrilliant/human-eval-selfcorr)
repository contains the full HumanEval certifier and regression suite.

## Protocol guarantees

- Every API attempt is appended verbatim to an immutable JSONL event ledger before any
  parsing; raw responses are SHA-256 hashed.
- No `max_tokens` / `max_completion_tokens` parameter appears in any request; provider-side
  `finish_reason=length` terminations stay visible as malformed records and are never
  retried into regeneration, never recoded.
- Verdicts extract only via a strict terminal `VERDICT: ACCEPT` / `VERDICT: REJECT` grammar;
  no substring fallback.
- FAR/TAR rates use paired-stratum denominators; malformed records are reported as counts,
  never imputed.

## Offline validation

From this repository's root:

```bash
python3 scripts/validate_mbpp_v2_pilot.py --out data/mbpp_v2/production
python3 scripts/validate_mbpp_v2_pilot.py --out data/mbpp_v2/replication
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/mbpp_v2_analysis.py --ledger data/mbpp_v2/production --out data/mbpp_v2/analysis
python3 scripts/mbpp_v2_moderator_analysis.py
python3 scripts/mbpp_v2_tier1_analysis.py
python3 scripts/humaneval/paired_contrasts.py
```
