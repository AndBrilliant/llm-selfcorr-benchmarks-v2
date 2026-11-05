# Fresh MBPP rebuild plan

Status: **not running**. No model API calls have been launched for the fresh benchmark.

## Non-negotiable rule

Every verdict must be reconstructible from the exact preserved raw text. A response that does
not contain exactly one terminal `VERDICT: ACCEPT` or `VERDICT: REJECT` line is malformed and
must be retained as an extraction failure. There is no substring fallback and no verdict is
copied from in-memory content unless that content is also persisted.

No request ever carries a `max_tokens` or `max_completion_tokens` parameter. The harness
never truncates a response; each response generates until the model itself ends it. A
`finish_reason=length` event can therefore only come from the provider's own context ceiling
and is recorded visibly, never retried (it is deterministic at temperature 0), and never
silently converted into a verdict.

## Fresh record schema

Each generation and evaluator attempt must persist, before parsing:

- run ID, model, task ID, prompt fingerprint, timestamp, attempt number, and retry count;
- complete raw response text exactly as returned;
- provider completion/finish metadata when available;
- HTTP/provider status and error class on failure;
- derived candidate code or parsed verdict only in a separate analysis step.

A transport-failed call may be retried under the pre-specified fixed retry
policy. Provider-side `finish_reason=length` truncation is deterministic at temperature 0,
so it is recorded once as an immutable failed attempt and never retried. Note the harness
itself never truncates: no request carries a token cap, so a `length` finish can only come
from the provider's own context ceiling. Every attempt remains in the immutable event
ledger, and only a complete selected attempt may feed a derived record.

A record is immutable after append. A manifest of SHA-256 hashes is committed after each stage.

## Pilot gate

Run a small fixed pilot first. It passes only if:

1. every complete raw response is present;
2. every stored verdict is reproduced from the stored raw text;
3. every malformed response remains an extraction failure;
4. no parser fallback fires;
5. manifest hashes verify;
6. restart/resume does not mutate or duplicate complete records.

Only after the pilot passes may production calls begin.

## Production sequence

1. Generate one new candidate per task and model under the new protocol.
2. Execute the task tests once on the frozen candidate and persist the result.
3. Evaluate each frozen candidate under both evaluator prompts.
4. Persist all complete responses and metadata before parsing.
5. Parse only from preserved text with the strict terminal parser.
6. Commit immutable ledgers, manifest, parser, analysis scripts, and tests.
7. Run the offline analysis and only then update the manuscript.

## Analysis requirements

- Fixed task and correctness denominators.
- Exact paired tests with Holm adjustment across the pre-specified five-family family.
- Bootstrap intervals must resample task IDs while keeping paired arms together.
- Malformed responses are reported separately; they are not silently recoded as rejections.
- Every table and manuscript claim must trace to a preserved ledger row and manifest hash.
