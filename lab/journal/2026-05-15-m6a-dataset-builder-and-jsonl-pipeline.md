[README](../../README.md) | [Lab](../README.md) | [Journal](./README.md) | [Plans](../plans/README.md) | [Decisions](../decisions/README.md)

# 2026-05-15 — M6a Dataset Builder And JSONL Pipeline

Status: validated
Date: 2026-05-15

## Question

The M6 plan entry called for a `DatasetBuilder` that fans `consortium_completion` across many prompts, emits a stream of prompt outcomes, and writes JSONL per finalized row. The load-bearing design decision was the **prompt-to-slot planning boundary**: `ConsortiumSlot<'a>` is built on borrowed `AiCompletionInputs<'a>` (per-provider `&Client` + `&Command`), so a multi-prompt builder cannot just "store slot configs" — it has to plan a fresh, owned command per prompt and then build a slot that borrows from it for one orchestration call. What shape gets that boundary right without contorting the builder's storage, and what failure split surfaces the three real categories (fatal setup, fatal run, per-prompt)?

## Hypothesis

Lock the M6a slice around four shapes:

1. `SlotTemplate` owns a provider client *by value* and a `plan` closure `Fn(&str) -> Result<XxxCompletionCommand, Box<dyn Error + Send + Sync>>`. Per-prompt commands live only inside `process_prompt` — never in the builder.
2. `DatasetBuilder<E: Embedder>` is generic over the embedder (not boxed) because [`Embedder`](../../src/embeddings/mod.rs) uses native `async fn in trait` and is not dyn-safe.
3. `PromptOutcome` is a three-variant enum (`Completed` / `Skipped` / `Failed`), one per input prompt, in original prompt-index order — including in-place `Skipped` entries.
4. `DatasetRow` is a small Serialize-only projection (winner content + provider + model_label, skip/failure reason). The full [`ConsortiumOutcome`](../../src/orchestrator/mod.rs) is not serialized into JSONL.

Defer:

- Per-prompt parallelism (M6b-style fan-out over selected prompts).
- Auto-chunking inside `Embedder::embed` for very large prompt batches.
- Streaming `PhaseEvent`s during a single prompt's orchestration (M5c).
- Richer JSONL projection that captures intermediate phase data.

## What We Tried

Built `src/dataset/mod.rs` as a single module covering the surface plus tests.

### Prompt-to-slot planning boundary

`SlotTemplate` is a per-provider enum holding the client (`OpenAiClient` / `ClaudeClient` / `GeminiClient`, by value, all `Clone` and cheap-to-clone via inner `reqwest::Client`) and a planner closure stored as `Arc<dyn Fn(&str) -> Result<XxxCompletionCommand, Box<dyn Error + Send + Sync>> + Send + Sync>`. Constructors `SlotTemplate::{openai, claude, gemini}` accept `Fn(&str) -> Result<Cmd, E> where E: Into<Box<dyn Error + Send + Sync>>` — callers can return `String`, `&'static str`, `Infallible`, or any concrete `Error` impl without naming a crate-defined error alias. The `E -> Box<dyn Error...>` lift happens inside the constructor.

Per prompt, `process_prompt` walks the template slice, calls each planner, and collects an internal `Vec<SlotCommand>` (a private 3-variant enum mirroring the provider triple). It then zips templates × commands into `Vec<ConsortiumSlot<'_>>` borrowing both from the templates (for the client) and from the per-iteration command vec. Both lifetimes resolve to the function's stack frame — neither escapes. The orchestrator's `ConsortiumOutcome` is fully owned, so it survives the function's return.

The template ↔ command variant pairing is an internal invariant of `SlotTemplate::plan_for` (the variant of the produced `SlotCommand` always matches `self`'s variant). The zip uses a `match` over the pair with an `unreachable!()` arm for mismatch — a programmer-error path inside this module, not a runtime concern.

### Failure split

Three explicit categories matching the orchestrator pattern:

- `DatasetBuildError` — eagerly surfaced by `DatasetBuilder::build` before any prompt work. Variants: `NoSlots`, `NoJudges`, `ZeroSamples { slot_index, model_label }`.
- `DatasetRunError` — one-shot fatal error surfaced by `DatasetRun::execute` before the stream is constructed. Variants: `Embedding(AgnosticEmbeddingError)`, `EmbeddingCountMismatch { expected, got }`, and (post-review) `EmbeddingDimensionMismatch { row_index, expected, actual }`. (No `EmptyPrompts` — empty input yields an empty stream, consistent with the rest of the crate's "empty is a valid no-op" stance.)
- `PromptRunError` — per-prompt failure surfaced inside `PromptOutcome::Failed`. Only variant for M6a is `SlotPlanning { slot_index, model_label, #[source] source: Box<dyn Error + Send + Sync> }`. The outer envelope is typed and stable; the planner's underlying error stays opaque so callers can downcast or `to_string()` without the dataset module having to know the planner closure's error type.

The orchestrator itself is infallible at the function-call level — every per-slot / per-judge / per-sample failure already lives inside `ConsortiumOutcome`. M6a does not collapse that internal record into a new "failed prompt" outcome; a prompt whose `ConsortiumOutcome.phase_two.winner` is `None` is still a `PromptOutcome::Completed` with full provenance.

### Stream construction

`DatasetRun::execute` does eager work first (embedding + diversification) and then returns `impl Stream<Item = PromptOutcome>` built via `futures::stream::unfold`. Sequential per-prompt iteration matches the M6a brief ("shape first, throughput second"); per-prompt parallelism is a follow-up. The unfold state owns `slot_templates`, `judges` (as `Vec<Arc<dyn JudgeProvider>>`), `prompts`, the `selected` index set, and the next-position cursor — the embedder is consumed in `execute` and not stored in the stream state.

The stream is not declared `Send`. The M5a orchestrator captures `&dyn JudgeProvider` across `.await` points; because `JudgeProvider`'s trait-object form does not pick up `Sync` from the supertrait declaration, the orchestrator's future is intrinsically non-`Send`. Adopting a `Send` dataset stream would require modifying the M5a orchestrator surface (`&[&(dyn JudgeProvider + Send + Sync)]`), and the M6a brief explicitly said "keep `JudgeProvider` reuse as-is from M5a". The trade-off: callers consume the stream on a single task. Spawning a multi-task driver becomes its own slice once the cost is real.

### Skip-when-no-filter fast path

If `stop_condition.max_n.is_none() && stop_condition.similarity_tripwire.is_none()`, the builder cannot exclude any prompt regardless of embedding. The fast path selects all indices directly without calling `embedder.embed`. The happy-path JSONL flow can also skip embedding entirely by configuring this no-filter `StopCondition` (the default produced by `DatasetBuilder::new`), which is useful for "consortium inference SDK" use cases where the user already has the prompts they want and does not need diversity selection.

A `PanicEmbedder` test embedder asserts this short-circuit really avoids the embed call when an empty prompt vec is also handed in alongside a non-trivial `StopCondition`.

### Original-order preservation

The unfold iterates `next = 0..prompts.len()` regardless of selection. Skipped prompts appear in-place as `PromptOutcome::Skipped { prompt_index, prompt, reason: NotSelectedByDiversification }` rather than being filtered out or moved to the tail. Callers downstream (JSONL writer, future progress UIs) see one row per input prompt, in the order they were submitted.

### JSONL row + writer

`DatasetRow` is a deliberately small Serialize-only projection:

```jsonl
{"prompt_index":0,"prompt":"...","status":{"kind":"completed","winner":{"model_label":"slot-a","provider":"openai","content":"..."}}}
{"prompt_index":1,"prompt":"...","status":{"kind":"skipped","reason":"not_selected_by_diversification"}}
{"prompt_index":2,"prompt":"...","status":{"kind":"failed","error":"slot 0 (slot-a): planner failed: ..."}}
```

`write_jsonl<W: AsyncWrite, S: Stream<Item = PromptOutcome>>` writes one line, flushes after every line, and stops at the first I/O or serialization error. The flush-per-row contract is what lets a tail / reader see finalized prompts promptly and what lets a crash preserve everything up to the last finalized prompt. Serializing the full `ConsortiumOutcome` graph was deliberately not attempted — callers who need richer audit data should consume the `PromptOutcome` stream directly.

## Result

`cargo test --lib` is green at 107 passed / 0 failed / 5 ignored (+8 over M5a's 99 — 3 builder validation, 1 happy path, 1 failure continuation, 1 JSONL writer, 1 empty-prompts short-circuit, 1 jagged-embedding-dimension guard added post-review).

`cargo clippy --lib --tests --all-features` is clean on the new `src/dataset/` module. Pre-existing warnings on the four stub providers and one `collapsible_if` in `src/diversification/mod.rs` are unchanged from M5a (M7 / unrelated territory and not in scope for this slice).

`cargo fmt` was run.

### Tests of note

- `happy_path_emits_one_outcome_per_prompt_in_original_index_order` — 3 prompts × 2 OpenAi slots × 2 samples × 1 judge. `Centroid` + `max_n=2` selects indices {0, 2}; index 1 appears in-place as `Skipped`. Each completed prompt resolves a Phase 2 winner whose content matches one of the two mockito-backed slots.
- `failing_planner_yields_failed_then_continues_with_later_prompts` — 3 prompts, no diversification (so `PanicEmbedder` doubles as a guard that embedding really is skipped). The planner returns `Err(String)` for any prompt containing "fail". Prompt 0 yields `Failed { error: SlotPlanning { slot_index: 0, model_label: "slot-a", source: <contains "planner says no"> } }`; prompts 1 and 2 still produce `Completed`. The String-error path also exercises the `E: Into<Box<dyn Error + Send + Sync>>` constructor bound.
- `write_jsonl_emits_one_row_per_outcome_with_winner_projection` — drives `write_jsonl` directly against three canned `PromptOutcome`s (no orchestration), parses each line as JSON, and asserts the projection (kind tag, winner fields for `completed`, reason for `skipped`, error text for `failed`).
- `empty_prompts_yields_empty_stream_without_calling_embedder` — `PanicEmbedder` + `StopCondition::with_max_n(5)` would normally embed; with empty prompts the run short-circuits to the no-filter path and never calls `embed`.
- `execute_returns_typed_error_when_embedder_yields_mixed_dimension_vectors` — `JaggedEmbedder` returns row 0 with dim 3 and row 1 with dim 4. The dataset layer rejects the batch with `DatasetRunError::EmbeddingDimensionMismatch { row_index: 1, expected: 3, actual: 4 }` before `select_diverse` is reached.

## Revision (post-review): guard embedding dimensions before `select_diverse`

A review caught a real runtime hole in the first cut: `DatasetRun::execute` validated `batch.vectors.len() == prompts.len()` but not per-row dimensionality, then immediately handed the batch to [`select_diverse`](../../src/diversification/mod.rs) — which is documented to **panic** on mixed dimensions. Both real embedders today return `Vec<Vec<f32>>` without validating per-row consistency before yielding the batch (see `src/ai_client_apis/openai/embeddings.rs` and `src/ai_client_apis/cohere/embeddings.rs`), so a misbehaving or evolving provider parser could escape the public M6a surface as a panic instead of a typed runtime error.

Fix: a new `DatasetRunError::EmbeddingDimensionMismatch { row_index, expected, actual }` variant; `DatasetRun::execute` anchors `expected_dim` on row 0 and rejects the first divergent row before `select_diverse`. A new `JaggedEmbedder` test fixture and the `execute_returns_typed_error_when_embedder_yields_mixed_dimension_vectors` regression test prove the dimension-mismatch path returns `Err(EmbeddingDimensionMismatch { .. })` instead of unwinding through diversification.

Provider-side hardening (map malformed mixed-dimension provider responses to `AgnosticEmbeddingError::MalformedResponse` earlier, so the dataset layer never has to see them) is a related follow-up but out of scope for M6a — it touches the wire-level parsers and the embedder trait contract, not the dataset orchestration layer.

The fix is purely additive on the public surface: existing `DatasetRunError` variants kept their names and meanings, one new variant appeared. Tests are green at 107 passed / 0 failed / 5 ignored after the revision.

## Decision

Lock the M6a result shape as shipped (post-revision). Streaming-progress events (per-prompt phase transitions), per-prompt parallelism, auto-chunking inside `Embedder::embed`, and richer JSONL projections are all additive over this surface and not bundled with M6a.

## Next

- M6b: per-prompt parallelism via bounded `FuturesUnordered` over selected prompts. Stop-condition-aware: skipped prompts still need to surface in-order, so the stream wrapper would reorder completed work into prompt-index order before yielding (or accept out-of-order delivery as a separate `execute_unordered` surface).
- M6c: auto-chunking inside the per-provider embedder impls so callers don't have to hand-shard inputs that exceed `OPENAI: 2048` / `COHERE: 96` per-request limits.
- M5b is still upstream: parallelizing judge invocation inside `consortium_completion` would benefit M6a directly without any dataset-layer change.
- A future slice may revisit the JSONL row shape to capture richer per-prompt provenance (judge outcomes summary, sample count, total tokens). Today's shape is intentionally minimal; once dataset consumers exist in real use, the shape can grow informed by what they actually need.

## See Also

- [Implementation Plan — M6](../plans/2026-05-15-implementation-plan.md)
- [2026-05-15 — M5a Two-Phase Consortium Orchestrator](./2026-05-15-m5a-two-phase-orchestrator.md)
- [2026-05-15 — M4 Judge Layer Corrections](./2026-05-15-m4-judge-layer.md)
- [2026-05-15 — M3 Multi-Provider Embedding Direction](./2026-05-15-m3-multi-provider-embedding.md)
- [Decision 0001: Owned Types At Async Boundaries](../decisions/0001-owned-types-at-async-boundaries.md)
- [Journal Index](./README.md)
