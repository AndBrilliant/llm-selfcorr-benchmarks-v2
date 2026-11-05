# Pre-registered moderator specification — MBPP arm-association stratification (Tier 0)

Registration date: 2026-11-05.
Registered BEFORE any stratified outcome was computed. The production ledger
(commit 91603a6) and the arm-level association statistics were public; no
moderator-conditioned statistic existed at registration time.

## Question

Does conditioning on pre-specified, correctness-blind candidate/task features Z
absorb the observed association between the self-check and steelman verdicts
on incorrect candidates? If yes, arm agreement is explained by shared *visible*
surface features. If phi persists within every stratum, the association is
consistent with a shared latent factor (while remaining agnostic about whether
that factor is the framework's Z or an unobserved surface feature — the test
cannot distinguish those two).

## Estimand and population

Primary population: per family, the paired incorrect stratum (T=0, terminal
verdict from both arms) — the same population as the primary FAR contrast.
Secondary population: all paired records (both T strata) for robustness.

Primary statistic: phi coefficient between the two arms' binary verdicts.
Association measure is phi (not odds ratio) because discordant-off-diagonal
cells are small in some strata; phi is bounded and directly comparable across
strata of different sizes.

## Moderators (all computable from the task definition and frozen candidate
code before any evaluator outcome is known; none uses T, test outcomes,
verdicts, or evaluator text)

1. `provenance` — MBPP task-id range: intro (1–10), crowd (11–510),
   sanitized (511–974). Structural property of the benchmark.
2. `prompt_words` — word count of the task prompt, tertiles.
3. `code_lines` — line count of the frozen candidate code, tertiles
   (blank lines stripped).
4. `n_args` — number of formal parameters of the entry-point signature:
   1 vs >=2.
5. `concept` — `text` if the candidate code imports `re` or calls any str
   method; else `other`. Deterministic keyword scan of the frozen code.

## Pre-registered decision rule

Let phi_crude be the per-family phi on the primary population, and phi_MH the
Mantel–Haenszel common phi adjusted for the moderator strata (per moderator,
per family).

- **Absorbed** (mundane explanation supported): for a majority of families,
  |phi_MH| < 0.10 AND the stratified association is non-significant
  (Fisher exact p > 0.05) in every stratum of at least one moderator.
- **Persistent** (consistent with shared latent factor): phi_MH within 0.05
  of phi_crude in every family for at least one moderator, and joint
  acceptance within strata stays >= 70% in every stratum of that moderator.
- **Mixed**: neither rule fires; report per-stratum estimates without a
  headline claim.

Multiplicity: the decision rule is evaluated per moderator with the family as
the unit; no p-value is the sole arbiter. Per-stratum Fisher p-values are
reported descriptively. No post-hoc moderators will be added to this spec;
any new moderator requires a new registration commit before its outcomes are
examined.

## Data

- Ledger: data/mbpp_v2/production (immutable, manifest 91603a6)
- Candidates: candidate_records.jsonl (frozen code)
- Analysis script: scripts/mbpp_v2_moderator_analysis.py
- Outputs: data/mbpp_v2/moderator_analysis/ (json, txt, tex, manifest)
