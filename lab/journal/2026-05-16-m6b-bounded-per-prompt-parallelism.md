[README](../../README.md) | [Lab](../README.md) | [Journal](./README.md) | [Plans](../plans/README.md) | [Decisions](../decisions/README.md)

# 2026-05-16 — M6b Bounded Per-Prompt Parallelism

Status: validated
Date: 2026-05-16

## Question

[M6a](./2026-05-15-m6a-dataset-builder-and-jsonl-pipeline.md) shipped the `DatasetBuilder` / `DatasetRun` split with strictly sequential per-prompt execution. The implementation plan called out M6b as "per-prompt parallelism via bounded `FuturesUnordered` with original-order delivery." The load-bearing design question for this slice: can we add bounded concurrency under the existing `DatasetRun::execute(...)` signature — without growing a separate `execute_unordered` surface, without making the stream `Send`, and without giving up the M6a guarantee that emitted outcomes are in original `prompt_index` order with skipped prompts in-place?

## Hypothesis

Lock the M6b slice around four shapes:

1. A `DatasetBuilder::parallelism(n)` builder method with default `1` (preserves M6a sequential behaviour). Eager rejection of `0` via a new `DatasetBuildError::ZeroParallelism` variant.
2. `DatasetRun::execute` keeps its existing signature. Internally, the `stream::unfold` body is rewritten around a bounded `FuturesUnordered<Pin<Box<dyn Future<Output = (usize, PromptOutcome)>>>>`.
3. Original-order delivery is preserved by a `BTreeMap<usize, PromptOutcome>` reorder buffer plus a `next_to_emit` cursor. The reorder buffer is bounded by `parallelism` because only *completed selected* outcomes are buffered.
4. Skipped prompts are *not* pre-scheduled into the in-flight queue. They wait in `prompts[i]` and emit synchronously when the emit cursor reaches them. This avoids buffering an unbounded prefix of skipped outcomes when most of the input is filtered out by diversification.

Defer (still):

- A separate `execute_unordered` surface. The ordered surface is sufficient for M6.
- `tokio::spawn`-backed parallelism. The stream remains single-task / non-`Send` (the M5a orchestrator's future is non-`Send`, see [M6a journal](./2026-05-15-m6a-dataset-builder-and-jsonl-pipeline.md)). Cooperative concurrency under `FuturesUnordered` is enough for the per-prompt fan-out we care about; making the stream `Send` would require touching the M5a orchestrator's signature.
- Auto-chunking inside `Embedder::embed` (still M6c / M3b territory).

## What We Tried

Single-module change in [`src/dataset/mod.rs`](../../src/dataset/mod.rs). No changes to the orchestrator, judge, embedder, or diversification modules.

### Builder knob and eager validation

Added a `parallelism: usize` field to `DatasetBuilder` defaulting to `1`, a `.parallelism(n)` setter, and a `DatasetBuildError::ZeroParallelism` variant. The validation sits next to the existing `NoSlots` / `NoJudges` / `ZeroSamples` checks, so `0` never reaches `execute`. `1` keeps M6a's exact behaviour.

### Per-run state and the in-flight queue

`StreamState` now wraps the shared per-run handles in `Arc`:

- `slot_templates: Arc<Vec<SlotTemplate>>`
- `judges: Arc<Vec<Arc<dyn JudgeProvider>>>`

Each in-flight future clones the two `Arc`s and owns the moved-out prompt `String`. The captured set is `'static`, so the future is box-pinnable as `Pin<Box<dyn Future<Output = (usize, PromptOutcome)>>>` (no explicit `Send` bound — the stream stays non-`Send`).

`StreamState` also carries:

- `next_to_schedule: usize` (advances through every prompt slot)
- `next_to_emit: usize` (advances only on emit)
- `parallelism: usize`
- `in_flight: FuturesUnordered<PromptFuture>`
- `reorder_buffer: BTreeMap<usize, PromptOutcome>`

### Fill → emit loop

Each iteration of the unfold body does, in order:

1. **Fill.** While `in_flight.len() < parallelism && next_to_schedule < prompts.len()`: if `selected.contains(&i)`, take the prompt, build the boxed `process_prompt` future returning `(i, PromptOutcome)`, push it into `in_flight`. If skipped, just advance `next_to_schedule` — the prompt stays in `prompts[i]` for emit time. Skipped slots never count against the in-flight cap.
2. **Emit.**
   - If `reorder_buffer` holds `next_to_emit` → pop, bump cursor, yield.
   - Else if `next_to_emit` is skipped → take the prompt, yield `PromptOutcome::Skipped` in-place, bump cursor.
   - Else (`next_to_emit` is selected, not yet ready) → `in_flight.next().await`, insert into the reorder buffer, loop back through fill+emit. If `in_flight` is unexpectedly empty here, return `None` (invariant: the fill phase guarantees the future for `next_to_emit` is either in flight or already buffered).
3. `next_to_emit >= prompts.len()` → return `None`. Debug-asserts that both the reorder buffer and the in-flight queue are empty.

Skipped prompts therefore flow through the same ordered emit path as completed/failed ones — they are not silently routed around the buffer. They just never enter the in-flight queue.

### Failure preservation

Untouched. Each future's body is the existing `match process_prompt(&templates, &judges, &prompt).await { Ok => Completed, Err => Failed }`, just lifted into a boxed future. `PromptRunError::SlotPlanning` continues to surface inline as `PromptOutcome::Failed`. A failing prompt never terminates the stream.

## Tests

`cargo test --lib` is 111 passed / 0 failed / 5 ignored (+4 over M6a's 107). All 8 M6a tests stay green at the default `parallelism = 1`.

Four new tests in `src/dataset/mod.rs#tests`:

1. **Builder rejects `parallelism = 0`.** Synchronous, asserts `Err(DatasetBuildError::ZeroParallelism)`.
2. **In-order delivery when later prompts finish first.** Three prompts, `parallelism = 3`. Three mockito mocks differentiated by `Matcher::Regex(r#""pN""#)` on the request body return distinct content markers `R0` / `R1` / `R2`. A custom `SleepJudge` sleeps `200 ms` real-time only when judging candidates containing `R0`; prompts 1 and 2 sleep `0 ms`. Despite prompt 0 finishing last in wall-clock time, the emitted stream is `[prompt_index = 0, 1, 2]` and every outcome is `Completed`.
3. **Bounded concurrency at `parallelism = 3` over six prompts.** Each judge invocation does `active.fetch_add → max_seen.fetch_max → batch_barrier.wait().await → active.fetch_sub`. The barrier is sized to `parallelism`, so it only releases once exactly `parallelism` judge calls are concurrently suspended — direct evidence that the run keeps that many prompts in flight at once. The outer `tokio::time::timeout(10s, ...)` turns a too-low-concurrency bug (barrier deadlock) into a clean test failure. Asserts `max_seen == parallelism`, `active == 0` at the end, and order preservation across both batches.
4. **Partial failure under concurrency.** Five prompts at `parallelism = 3` where odd-indexed prompts contain "fail" and the planner rejects them. Asserts the stream emits 5 outcomes in original index order with `Completed` at 0/2/4 and `Failed { PromptRunError::SlotPlanning { source } }` at 1/3, with `source.to_string()` carrying the originating "planner says no for fail-1" / "fail-3" payload.

## Verified Properties

- Public surface unchanged: `DatasetRun::execute(prompts) -> Result<impl Stream<Item = PromptOutcome>, DatasetRunError>` is identical to M6a.
- Default behaviour unchanged: `parallelism = 1` keeps M6a's exact emission shape — verified by all 8 M6a tests staying green.
- Order preservation: completed outcomes that finish out-of-order are still emitted in original `prompt_index` order; skipped prompts emit in-place at their slot.
- Bounded in-flight work: at most `parallelism` `process_prompt` futures are alive concurrently — verified by a barrier-pinned judge.
- Failure preservation: per-prompt planner errors still produce `PromptOutcome::Failed` without terminating the stream.
- No `Send` regression: the stream stays non-`Send` (no orchestrator surgery), which is the M6a constraint.
- No spawned tasks: `FuturesUnordered` runs cooperatively inside the unfold body's single task.

## What's Next

- **M6c.** Auto-chunking inside `Embedder::embed` (OpenAI: 2048 / Cohere v3: 96). Closely related to [M3b](./2026-05-15-m3-multi-provider-embedding.md).
- **M5b.** Parallelise judge invocation across slots and across judges within a single prompt's phase — currently each prompt's judges still run sequentially in M5a. Independent from M6b's per-prompt parallelism.
- **M5c.** Streaming `PhaseEvent` surface over the same orchestrator. Likely consumes the per-prompt outcome stream as one of its inputs.

## See Also

- [M6a Dataset Builder And JSONL Pipeline](./2026-05-15-m6a-dataset-builder-and-jsonl-pipeline.md)
- [M5a Two-Phase Consortium Orchestrator](./2026-05-15-m5a-two-phase-orchestrator.md)
- [Current Implementation Plan](../plans/2026-05-15-implementation-plan.md)
