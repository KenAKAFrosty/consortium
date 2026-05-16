[README](../../README.md) | [Lab](../README.md) | [Journal](./README.md) | [Plans](../plans/README.md) | [Decisions](../decisions/README.md)

# 2026-05-15 — M4 Judge Layer Corrections

Status: validated
Date: 2026-05-15

## Question

The original M4 outline in the plan had `assign_blind_ids(candidates) -> (HashMap<BlindId, ProviderTag>, Vec<BlindCandidate>)` and `OrderedJudgementStructuredData { ordered_ids: Vec<&str> }`. Are those shapes good enough to ship M4 against?

## Hypothesis

No on both. The blind-id-to-provider mapping is architecturally wrong: M5 phase 1 (intra-model multi-judge ranking) generates multiple candidates per provider, so a `BlindId → ProviderTag` map can't tell those candidates apart. The parse contract was also too lax: with just `ordered_ids: Vec<&str>`, there was no plan to validate that the judge's ranking was actually a full permutation of the provided ids — no plan to reject duplicates, unknowns, partial rankings, or empty rankings.

## What We Tried

Pre-coded an alignment pass that pinned the corrected shape, then implemented M4 against the corrected design.

### Blind id mapping: id → candidate index, not id → provider

`assign_blind_ids(&[Candidate]) -> (Vec<BlindCandidate>, HashMap<BlindId, usize>)`. The `usize` is the candidate's index in the input slice. Callers retain the original `Vec<Candidate>` and use the returned map to look up the full `Candidate` (with provider, model, content) by `BlindId`. Two same-provider-same-model candidates with different content are distinguishable; the blind id corresponds to a specific candidate, not a category.

`Candidate { content: String, provider: ProviderKind, model: String }`. Provider and model are preserved on the candidate for post-judgment provenance but never sent to the judge — the judge gets only `BlindCandidate { id, content }`.

This matters because M5 phase 1 looks like: "sample N completions from model X, run a multi-judge ranking over those N." If `assign_blind_ids` collapsed to provider, all N candidates would share a single mapping entry and the layer would lose track of which was which.

### Typed parse contract

`OrderedJudgement { ordered_ids: Vec<BlindId>, reasoning: String, raw_response: String }`. The reasoning is kept so callers can audit the judge's thinking; the raw response is kept so the full transcript is recoverable for debug. The ordering is `Vec<BlindId>` (owned), not `Vec<&str>` (borrowed) — owned at the agnostic boundary per the existing decision 0001.

`JudgementParseError` is a typed enum with one variant per failure mode the parser can detect: `MissingReasoningTag`, `MissingRankingTag`, `EmptyRanking`, `UnknownId { id }`, `DuplicateId { id }`, `MissingIds { missing: Vec<String> }`. Each carries enough context to debug an offending response without re-fetching it.

`JudgementError` wraps `Provider(AgnosticCompletionError)` and `Parse(JudgementParseError)` so the public `judge_rank` API surfaces both failure categories distinctly.

### Strict ranking validation

The parser:
- Tolerates whitespace around blind ids and inside the `<reasoning>` block.
- Splits the `<ranking>` block on commas, trims, drops empties from the split.
- Rejects empty ranking → `EmptyRanking`.
- Rejects unknown blind ids → `UnknownId { id }` (first occurrence reported).
- Rejects duplicates → `DuplicateId { id }`.
- Rejects missing-from-expected → `MissingIds { missing }` (sorted, full list).

No `LengthMismatch` variant — that case is always reachable by one of the more specific variants (UnknownId for extras, MissingIds for shortages).

### Locked prompt contract

`JUDGE_SYSTEM_PROMPT` is a `pub const &'static str` with:
- The two-block response format (`<reasoning>` then `<ranking>`).
- "No ties. Every position is a strict preference."
- "Every candidate identifier must appear in the ranking exactly once."
- "Do not speculate about the source model or organization. Judge the content alone."
- "Do not write anything outside the two tagged blocks."

A test asserts `JUDGE_SYSTEM_PROMPT` does not contain `claude`, `openai`, `gemini`, `anthropic`, `cohere`, or `gpt` (case-insensitive). A parallel test asserts `build_judge_user_message` doesn't leak provider names into the user payload.

### Provider-agnostic invocation

`judge_rank(candidates, invoke_judge)` takes a closure `FnOnce(JudgeRequest) -> Future<Output = Result<String, AgnosticCompletionError>>`. `JudgeRequest { system_prompt: &'static str, user_message: String }` gives the caller both pieces — they wire them into whichever provider command shape they need. Tests pass canned responses through closures without touching real providers. M5 will provide the concrete provider-as-judge wiring; M4 keeps the boundary provider-agnostic.

### Deterministic aggregation

`aggregate_rankings(&[OrderedJudgement]) -> AggregatedRanking`. Borda count: i-th place earns `N - i` where N is the ranking length. Scores summed across judges. Final order: by score desc.

Tie-break: lexicographic `BlindId` string comparison. Documented as deterministic but not numeric — `c10` sorts before `c2`. Callers that need numeric ordering or a different rule can post-process `AggregatedRanking.scores` (exposed deliberately for this).

`AggregatedRanking` carries both `ordered_ids` (the final order) and `scores` (the full Borda totals) so the audit chain stays intact.

### Module placement

`src/judge/mod.rs`. The user explicitly called out that `src/lib.rs` is already large and M4 was the right point to split. M4 lives in its own module; `lib.rs` retains only `pub mod judge;` plus the re-exports at crate root.

The same slice removed five M0-era placeholders from `lib.rs`: `ORDERED_JUDGEMENT_SYSTEM_PROMPT` constant, `OrderedJudgementStructuredData`, `SortableJudgementProvider`, `AiCompletionCommand`, `make_sortable_judgement_command`. All superseded by `src/judge/`. Dead-code warnings on `ORDERED_JUDGEMENT_SYSTEM_PROMPT` and `ordered_ids` that had been lingering since M0 are finally cleared.

## Result

89 tests pass. Coverage of the parse contract is exhaustive: every `JudgementParseError` variant has a dedicated test with a hand-written response. The aggregation has tests for the single-ranking Borda math, unanimous case, disagreement case, tied case (verifying lex tie-break), and empty case.

The blind-id → candidate-index mapping was the right call. The test `assign_blind_ids_assigns_sequential_ids_preserving_provenance` explicitly constructs two OpenAI candidates with different content and verifies that the blind-id map distinguishes them — that test would have been impossible to write under the original `BlindId → ProviderTag` shape.

## Decision

Lock the M4 contract as shipped. Future judge layer work (additional aggregation methods like Copeland or mean-rank, dyn-dispatched judge providers, batched judge invocations, etc.) is M4b+ and stays additive to this surface.

## Next

- M5: two-phase consortium orchestrator. Phase 1 fans out N completions per model, judges them with the M4 layer, keeps a per-model winner. Phase 2 takes the per-model winners and runs cross-model judging. Streams intermediate results via `mpsc` or a callback trait. The M4 primitives (`Candidate`, `assign_blind_ids`, `judge_rank`, `aggregate_rankings`) plug in directly.
- Possible M4b: Copeland aggregation (head-to-head wins) and mean-rank as alternative aggregation strategies behind an enum. Useful once we have real-world judge transcripts to compare aggregation behaviors on.

## See Also

- [Implementation Plan — M4](../plans/2026-05-15-implementation-plan.md)
- [2026-05-15 — M3 Multi-Provider Embedding Direction](./2026-05-15-m3-multi-provider-embedding.md)
- [Journal Index](./README.md)
