[README](../../README.md) | [Lab](../README.md) | [Journal](./README.md) | [Plans](../plans/README.md) | [Decisions](../decisions/README.md)

# 2026-05-15 — M5a Two-Phase Consortium Orchestrator

Status: validated
Date: 2026-05-15

## Question

The M5 plan entry called for a single-prompt two-phase orchestrator and mentioned surfacing intermediate results "via an `mpsc` channel or a callback trait". Which surface should M5a actually ship — streaming-first, or a typed in-memory outcome — and how should the orchestrator route per-`(slot, sample)` provenance from M1's `input_index` mechanism through the M4 judge layer to a single winner?

## Hypothesis

Lead with a typed in-memory outcome and defer streaming. Reasons:

- The orchestration result shape is load-bearing for every downstream feature (M6 dataset writer, M9 bindings, M10 telemetry). Getting the shape right is the actual hard part. An `mpsc` or callback surface can be added later over the same orchestration without redesigning the result types.
- Streaming-first invites premature decisions about backpressure, channel ownership, cancellation semantics, and item granularity — none of which the consumer story currently requires.
- The brief for this slice explicitly says: streaming/hooks can come later once the orchestration result shape is correct.

For provenance routing, lean on the M1 contract (`ProviderAttempt::input_index` survives fan-out reordering) and the M4 contract (`assign_blind_ids` returns `BlindId → candidate-index`). Compose them: assign a fan-out `input_index` to every `(slot, sample)` pair, and at judge time compose the M4 index map with a per-slot `candidate-index → sample-index` map. A winner's `BlindId` then resolves all the way back to a specific `SampleAttempt`.

## What We Tried

Built `src/orchestrator/mod.rs` from scratch with these shape decisions:

### Result types

- `ConsortiumOutcome { phase_one: Vec<ModelPhaseOutcome>, phase_two: Option<CrossModelPhaseOutcome> }` is the top-level return value.
- `ModelPhaseOutcome { model_label, provider, samples, judge_outcomes, aggregated, winner }` keeps every sample's `ProviderAttempt` (success or failure) and every judge's `Result<OrderedJudgement, JudgementError>` (success or failure). Failures are first-class records, never dropped.
- `SampleAttempt { sample_index, attempt }` keeps the slot-local sample index alongside the underlying attempt. `PhaseOneWinner.sample_index` indexes back into `ModelPhaseOutcome.samples` so callers can recover the exact originating attempt (timing, retries, raw output) in one hop.
- `CrossModelPhaseOutcome { candidates, judge_outcomes, aggregated, winner }` and `PhaseTwoWinner.model_index` trace cross-model winners back into `phase_one` for the same one-hop provenance story.

### Single mega fan-out

For Phase 1 sampling, the orchestrator assembles one big `Vec<AiCompletionInputs<'a>>` with `slot.samples` copies of each slot's input, then runs one `multi_infer` call. A parallel `routing: Vec<(slot_index, sample_index)>` is keyed by the outer `input_index`, so each returned `ProviderAttempt` is binned back into its slot in O(1).

This is the use case the M1 plan called out when it added `input_index` to `ProviderAttempt`: duplicate same-provider requests are valid and need correlation back to their originating slot. Doing per-slot `multi_infer` calls would have wasted the existing concurrency, since `multi_infer` is `FuturesUnordered`-driven.

### `JudgeProvider` at the orchestrator boundary, not in `src/judge/`

The M4 layer stayed closure-based: `judge_rank` takes `FnOnce(JudgeRequest) -> Fut`. M5a invokes the same judge once per slot in Phase 1 and once in Phase 2, which is awkward to model with `FnOnce` alone.

The minimal addition is a tiny trait at the orchestrator level:

```rust
pub trait JudgeProvider: Send + Sync {
    fn label(&self) -> &str;
    fn invoke<'a>(
        &'a self,
        request: JudgeRequest,
    ) -> BoxFuture<'a, Result<String, AgnosticCompletionError>>;
}
```

Judges are passed as `&[&dyn JudgeProvider]`. Inside the orchestrator, each invocation site adapts the trait back into the M4 closure shape: `judge_rank(&blind, |req| judge.invoke(req))`. The judge module itself never grows a provider path — the trait lives where the multi-invocation requirement is, not where the parse/aggregate primitives live.

### Singleton short-circuits

Phase 1 with a single surviving sample picks that sample as winner without invoking any judge (it would be a 1-candidate ranking — wasteful). Same for Phase 2 with a single surviving slot. Both report `aggregated = None` and an empty `judge_outcomes`. The candidate list disambiguates "no judges ran because trivial" from "every judge failed": a singleton candidate list with `aggregated = None` is the trivial case; a multi-candidate list with `aggregated = None` is the all-judges-failed case.

### Failure preservation contract

- 0 successes in a slot → `winner = None`, empty `judge_outcomes`, empty `aggregated`. Every failed `ProviderAttempt` stays in `samples`.
- N≥2 successes with M judges, K of which fail → `judge_outcomes` carries all M outcomes (K of them `Err`). Aggregation skips the K failures. If K = M, `aggregated = None` and `winner = None`, but every judge error is still visible.
- The aggregator's "all rankings over the same universe" invariant is preserved naturally: `judge_rank` validates that each successful ranking is exactly equal to the expected blind-id set, so successful judgements from one session always share a universe.

### `AiCompletionInputs::provider()` helper

Added a small accessor on `AiCompletionInputs` so `ModelPhaseOutcome.provider` is populated even for slots that drew zero successful attempts (where there'd be no `ProviderAttempt.provider` to read from). The previous shape implicitly relied on at least one attempt existing per slot, which the failure path violates.

### Module placement and removed comment

`consortium_completion` and its types live in `src/orchestrator/mod.rs`. The trailing M5-placeholder comment in `src/lib.rs` is removed. `lib.rs` re-exports the public orchestrator surface at the crate root.

## Result

`cargo test --lib` is green at 99 passed / 0 failed / 5 ignored. The two new orchestrator tests cover:

- Happy path: 2 slots × 2 samples × 2 closure judges. Verifies every sample is recorded, every judge succeeds, Phase 1 winners trace back to `sample_index = 0` via the blind-id provenance map, Phase 2 picks the expected slot, and the aggregated cross-model ranking's first id matches the resolved winner.
- Partial failure: slot A returns 503 for every sample (transient retries exhausted), slot B succeeds. Judge `j1` succeeds; judge `j2` returns `AgnosticCompletionError::Auth`. The test asserts: slot A retains both failed `ServerError` attempts and produces `winner = None`; slot B records both judge outcomes, `j2`'s `Auth` error is preserved as `JudgementError::Provider(Auth)` (not collapsed into a parse-error variant), and `j1` still produces a winner. Phase 2 short-circuits to the lone survivor with empty `judge_outcomes`.

Provider tests use `mockito`. Judges use a test-only `FnJudge<F>` struct that implements `JudgeProvider` over a synchronous closure — exactly the "canned closures over HTTP mocks" pattern the brief asked for.

Clippy is clean on the new module. Pre-existing dead-code warnings on the four stub providers (Deepseek, Kimi, Qwen, Llama) and an unrelated `collapsible_if` in `src/diversification/mod.rs` are M7 / unrelated territory and were not touched.

## Revision (post-review): explicit blind-id provenance

A review caught a real contract gap in the first cut: `JudgeOutcome.result` and `AggregatedRanking.scores` both carry `BlindId`s through to the public outcome, but the orchestrator computed the `BlindId → sample_index` map (Phase 1) and `BlindId → model_index` map (Phase 2) internally, used them to resolve winners, then dropped them. A caller could see a `BlindId` in a preserved judge result — winning or non-winning — and have no way to resolve it back to a concrete candidate except by guessing the orchestrator's sequential assignment convention.

For a slice whose whole point is typed provenance and failure preservation, that was a contract gap, not a presentation gap. Fix:

- New struct `JudgedSample { blind_id, sample_index, content }`.
- `ModelPhaseOutcome.judged: Vec<JudgedSample>` — explicit blind-id mapping for every candidate the Phase 1 judges (would have) seen. Populated whenever at least one sample succeeded, including the singleton short-circuit case.
- `CrossModelCandidate` gained two fields: `blind_id: BlindId` and `content: String`. Populated for every Phase 1 winner that made it into Phase 2, including the singleton case.
- Both phases now run `assign_blind_ids` up front (before the singleton check) so the mapping is recorded uniformly. The singleton case still skips actual judge invocation; only the blind-id assignment runs.
- The happy-path test now pulls a non-winning `BlindId` from a successful judge result, resolves it through `judged` (Phase 1) and through `candidates` (Phase 2), and verifies the resulting `sample_index` / `model_index` points back to a concrete `SampleAttempt` / `ModelPhaseOutcome`. The contract is proven from the outside, not just from the orchestrator's perspective.

The fix is purely additive on the public surface: existing fields kept their names and meanings, two new fields appeared, one new struct showed up. Tests are green at 99 passed / 0 failed / 5 ignored after the revision.

## Decision

Lock the M5a result shape as shipped (post-revision). Streaming surfaces (`mpsc`, callback trait), per-phase concurrency for judge calls, configurable retry policy for judge invocations, and multi-prompt orchestration are all additive over this typed outcome. They are not bundled with M5a.

## Next

- M5b: parallelize judge invocation within a phase (Phase 1 across slots and across judges, Phase 2 across judges) once the in-memory shape is in real use and we know the latency budget.
- M5c: a streaming surface — likely `mpsc::Sender<PhaseEvent>` driven off the same orchestrator, where `PhaseEvent` is one event per finalized `ModelPhaseOutcome` plus the final `CrossModelPhaseOutcome`. Keep the typed outcome as the canonical surface; streaming is a view onto it.
- M6: multi-prompt orchestration / DatasetBuilder builds on top of `consortium_completion` per prompt.

## See Also

- [Implementation Plan — M5](../plans/2026-05-15-implementation-plan.md)
- [2026-05-15 — M4 Judge Layer Corrections](./2026-05-15-m4-judge-layer.md)
- [Decision 0001: Owned Types At Async Boundaries](../decisions/0001-owned-types-at-async-boundaries.md)
- [Journal Index](./README.md)
