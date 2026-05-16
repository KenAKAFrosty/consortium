[README](../../README.md) | [Lab](../README.md) | [Journal](./README.md) | [Plans](../plans/README.md) | [Decisions](../decisions/README.md)

# 2026-05-16 — M5c Streaming PhaseEvent Surface

Status: validated
Date: 2026-05-16

## Question

[M5a](./2026-05-15-m5a-two-phase-orchestrator.md) settled the typed in-memory outcome shape and [M5b](./2026-05-16-m5b-parallel-judge-fanout.md) made the orchestrator's per-slot and per-judge work concurrent inside a single task. M5c's load-bearing question: how does a real-time progress consumer observe orchestration without losing any of the typed-outcome / failure-preservation invariants the previous slices locked in? Specifically — where does the event surface attach, what does an event carry, and how do streaming and non-streaming code paths stay impossible to diverge?

## Hypothesis

Three observations frame the shape:

1. **The canonical [`ConsortiumOutcome`] already exists and is authoritative.** Streaming should not duplicate provenance — it should be a *view* of the canonical state at well-defined checkpoints. Anything an event carries that's also in the canonical outcome should agree with the canonical outcome, by construction.
2. **The natural emission points are already in the code.** Phase 1 slots resolve at the [`futures::stream::FuturesUnordered::next`] site M5b introduced — that is the moment a [`ModelPhaseOutcome`] is finalized and is exactly the moment a progress consumer wants notified of. Phase 2 finishes at the bottom of [`consortium_completion`]; one more emission there closes the run.
3. **Events cannot trivially carry full provenance.** [`ProviderAttempt::result`] embeds [`AgnosticCompletionError::Transport { source: reqwest::Error }`], and [`reqwest::Error`] does not implement [`Clone`]. Same for [`JudgementError`]. Sending a full `ModelPhaseOutcome` through a channel requires either Arc-ifying the canonical outcome (changes public types) or moving the outcome twice (impossible). The natural answer: events carry *compact summaries*, and callers read full provenance from the returned canonical outcome.

Lock the slice around four shapes:

1. **Two public entry points, one body.** `consortium_completion(slots, judges) -> ConsortiumOutcome` keeps its M5a/b signature. New `consortium_completion_streaming(slots, judges, events: tokio::sync::mpsc::UnboundedSender<PhaseEvent>) -> ConsortiumOutcome` adds the event sender. Both delegate to a private `consortium_completion_impl(slots, judges, events: Option<&UnboundedSender<PhaseEvent>>)` — the *single* orchestration body. There is no separate "streaming orchestrator" code path; the diff is two `if let Some(tx) = events { let _ = tx.send(...); }` blocks at the emission sites.
2. **Compact event payloads.** `PhaseOneSlotEvent` carries `slot_index`, `model_label`, `provider`, the sample / judge counts (`total_samples`, `successful_samples`, `failed_samples`, `judges_run`, `judges_succeeded`, `judges_failed`), and the already-`Clone` `Option<PhaseOneWinner>`. `PhaseTwoFinishedEvent` carries the analogous Phase 2 counts plus `Option<PhaseTwoWinner>`. These payloads side-step every non-`Clone` type in the canonical outcome without losing the information a progress consumer actually needs.
3. **Unbounded sender, drop-on-receiver-loss semantics.** [`tokio::sync::mpsc::UnboundedSender::send`] is synchronous from the orchestrator's perspective — it never awaits, never blocks, never imposes backpressure on the orchestration. If the receiver is dropped, sends return `Err(SendError)` which the orchestrator discards (`let _ = tx.send(...)`). The canonical outcome is still returned in full; streaming is purely opportunistic.
4. **Always one terminal event.** `PhaseTwoFinished` fires exactly once at the end, even when `phase_two` is `None` (no slot produced a Phase 1 winner). All counts are zero in that case and `winner` is `None`. The event's existence signals overall completion; its payload describes what happened.

Defer (still):

- Per-event timing / latency metrics. Useful, but not load-bearing for the surface shape — additive over the existing `PhaseOneSlotEvent` and `PhaseTwoFinishedEvent` later.
- A bounded-sender variant. Today's downstream consumers (CLI, future bindings, tests) all run in the same task as the orchestrator and an unbounded channel cannot cause unbounded memory growth (the orchestration has a fixed number of slots and one Phase 2 event). If a future streaming dataset path ever needs backpressure, that's its own slice.
- An "intermediate" event between Phase 1 and Phase 2 (e.g., "all Phase 1 slots done"). Today's consumers can compute that by counting `PhaseOneSlotFinished` events against `slots.len()`. Adding a synthetic event for it would add API surface without new information.
- Cancellation semantics ("kill the orchestration when the receiver is dropped"). The dropped-receiver path silently drops events but the orchestration still runs to completion and returns the canonical outcome. If a caller wants early termination, they wrap the call in [`tokio::time::timeout`] or drop the surrounding task entirely — that's a caller-side concern.

## What We Tried

Single-module change in [`src/orchestrator/mod.rs`](../../src/orchestrator/mod.rs) plus re-exports in [`src/lib.rs`](../../src/lib.rs). No changes to `src/judge/`, `src/dataset/`, or any provider client. No new dependencies (the `tokio` `full` feature already includes `sync`).

### Public surface

```rust
pub enum PhaseEvent {
    PhaseOneSlotFinished(PhaseOneSlotEvent),
    PhaseTwoFinished(PhaseTwoFinishedEvent),
}

pub struct PhaseOneSlotEvent {
    pub slot_index: usize,
    pub model_label: String,
    pub provider: ProviderKind,
    pub total_samples: usize,
    pub successful_samples: usize,
    pub failed_samples: usize,
    pub judges_run: usize,
    pub judges_succeeded: usize,
    pub judges_failed: usize,
    pub winner: Option<PhaseOneWinner>,
}

pub struct PhaseTwoFinishedEvent {
    pub winner: Option<PhaseTwoWinner>,
    pub candidates: usize,
    pub judges_run: usize,
    pub judges_succeeded: usize,
    pub judges_failed: usize,
}

pub async fn consortium_completion_streaming<'a>(
    slots: &'a [ConsortiumSlot<'a>],
    judges: &'a [&'a dyn JudgeProvider],
    events: tokio::sync::mpsc::UnboundedSender<PhaseEvent>,
) -> ConsortiumOutcome;
```

`PhaseOneSlotEvent::from_outcome(slot_index, &outcome)` and `PhaseTwoFinishedEvent::from_outcome(&outcome)` are crate-private constructors; they compute the counts from references and clone only the already-`Clone` winner field.

### Shared body

The original `consortium_completion` body is now the body of `consortium_completion_impl(slots, judges, events: Option<&UnboundedSender<PhaseEvent>>)`. The two public entry points are one-line delegations:

```rust
pub async fn consortium_completion<'a>(
    slots: &'a [ConsortiumSlot<'a>],
    judges: &'a [&'a dyn JudgeProvider],
) -> ConsortiumOutcome {
    consortium_completion_impl(slots, judges, None).await
}

pub async fn consortium_completion_streaming<'a>(
    slots: &'a [ConsortiumSlot<'a>],
    judges: &'a [&'a dyn JudgeProvider],
    events: tokio::sync::mpsc::UnboundedSender<PhaseEvent>,
) -> ConsortiumOutcome {
    consortium_completion_impl(slots, judges, Some(&events)).await
}
```

This is the load-bearing convergence point: there is no streaming-specific orchestration code, so any semantic change in one path is by construction also a change to the other path.

### Emission points

Two `if let Some(tx) = events { let _ = tx.send(...); }` blocks inside `consortium_completion_impl`:

1. Inside the `while let Some((slot_index, outcome)) = slot_fanout.next().await` loop, *before* `phase_one_buf[slot_index] = Some(outcome);`. The summary is built from an immutable borrow of `outcome`; only after the send does ownership move into the reorder buffer. Real completion order is naturally preserved because `slot_fanout.next().await` yields in completion order.
2. Right after `let phase_two = phase_two_outcome(&phase_one, judges).await;` and before constructing the final `ConsortiumOutcome`. Always one send, regardless of whether `phase_two` is `Some` or `None`.

The canonical outcome path is unchanged — `phase_one_buf` still reorders by `slot_index`, the final `phase_one` is still in slot-index order, every M5a / M5b invariant is preserved.

### Why unbounded

Bounded `mpsc::Sender::send` returns a `Future` that suspends when the channel is full. Using it would mean: a slow receiver can pause the orchestrator mid-fan-out, which (a) defeats the M5b "concurrent fan-out" property in practice and (b) couples the orchestrator's progress to a streaming consumer's responsiveness. Unbounded keeps the canonical outcome path the authoritative path; streaming is a best-effort progress view.

Memory bound is trivially `slots.len() + 1` events: a fixed cap baked into the configuration.

### Module doc + lib re-exports

The orchestrator module doc grew a new "Streaming (M5c)" section spelling out the event semantics, the unbounded / silent-fail backpressure stance, and the shared-body guarantee. [`src/lib.rs`](../../src/lib.rs) re-exports `PhaseEvent`, `PhaseOneSlotEvent`, `PhaseTwoFinishedEvent`, and `consortium_completion_streaming` at the crate root alongside the existing M5a/b types.

## Tests

`cargo test --lib` is 123 passed / 0 failed / 6 ignored (+4 + 1 ignored over M5b's 119 / 5). All pre-existing tests stay green; the M5b parallel-fan-out tests in particular continue to pass without modification, demonstrating the streaming refactor didn't break the M5a/b shape.

Four new orchestrator tests in `src/orchestrator/mod.rs#tests`, plus one `#[ignore]`-gated real-API test:

1. **`streaming_emits_phase_one_events_in_real_completion_order_with_canonical_outcome_in_slot_order`.** Three slots × two samples each, single judge that sleeps 60s virtual time only when judging slot 0's candidates (matched by the `S0-content` mockito body). Under `#[tokio::test(start_paused = true)]`, slot 0's judging completes last. The test asserts: the first three events are all `PhaseOneSlotFinished`, their `slot_index` set is `{0, 1, 2}`, the first event's `slot_index` is *not* 0, the last Phase 1 event's `slot_index` *is* 0, and the canonical `outcome.phase_one[i].model_label` is `slot-i` regardless. The fourth event is `PhaseTwoFinished` with `candidates == 3` and a `Some(winner)`. This is the central "events arrive in real completion order while the canonical outcome stays deterministic" proof.

2. **`streaming_phase_two_finished_event_matches_canonical_outcome`.** Two slots × two samples × two judges, happy path. Asserts the `PhaseTwoFinishedEvent`'s `candidates`, `judges_run`, `judges_succeeded`, `judges_failed`, and `winner.{model_index, content, model_label, provider}` all agree with the canonical `outcome.phase_two`'s respective fields. Also verifies the two Phase 1 events have `total_samples == 2`, `successful_samples == 2`, `judges_run == 2`, `judges_succeeded == 2`. This is the "events do not lie about what happened" invariant.

3. **`streaming_preserves_failed_attempts_and_failed_judges_in_canonical_outcome`.** Slot A returns 503 for every sample (transient retries exhausted under `start_paused = true`); slot B succeeds with one passing judge and one failing judge. Events: slot A's event has `successful_samples == 0`, `failed_samples == 2`, `judges_run == 0`, `winner = None`; slot B's event has `judges_run == 2`, `judges_succeeded == 1`, `judges_failed == 1`, `winner = Some(_)`. Phase 2 event has `candidates == 1`, `judges_run == 0` (singleton short-circuit). Canonical outcome separately asserted: slot A's `samples` still carries both `Err(ServerError { status: 503, .. })` attempts; slot B's `judge_outcomes` still carries the failing `j2` as `Err(JudgementError::Provider(Auth { .. }))`. The streaming view does not replace canonical provenance.

4. **`streaming_singleton_short_circuits_emit_clean_events`.** One slot × one sample × one judge that *panics* when invoked. Singleton short-circuits skip the judge call entirely, so the test would panic if the singleton-skip were broken. Events: one `PhaseOneSlotFinished` with `total_samples == 1`, `successful_samples == 1`, `judges_run == 0`, `winner = Some(_)`; one `PhaseTwoFinished` with `candidates == 1`, `judges_run == 0`, `winner = Some(_)`. Two events total. Canonical outcome's `phase_one[0].judge_outcomes` and `phase_two.judge_outcomes` are both empty.

5. **`real_api_consortium_completion_emits_events_and_returns_canonical_outcome`** (`#[ignore]`). Reads `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` from the environment. **Skips cleanly with an `eprintln!` and an early return** if any key is missing — the test is safe to leave in place in environments without keys. When all three keys are present, builds one OpenAI / one Claude / one Gemini slot with `samples = 1` against the same fixture prompt (`"In one short sentence, explain why airplanes can fly."`), wires two live-OpenAI-backed judges (so Phase 2 is a real ranking, not a singleton), and runs `consortium_completion_streaming`. Asserts: exactly three `PhaseOneSlotFinished` events plus one `PhaseTwoFinished`; the canonical `outcome.phase_two.winner` is `Some(_)`; each Phase 1 event's `total_samples` / `judges_run` agree with the canonical outcome's `samples.len()` / `judge_outcomes.len()` at the same `slot_index`. Run with `cargo test -- --ignored real_api_consortium_completion`. The plan's post-M5 verification step ("`--ignored` consortium test produces a coherent best-of-best completion for a fixture prompt") is now satisfied at this slice.

A small `collect_events` helper drains the receiver until the sender is dropped (the sender is moved into `consortium_completion_streaming` and dropped at function return, closing the channel cleanly).

## Verified Properties

- **Canonical surface unchanged.** `consortium_completion(slots, judges) -> ConsortiumOutcome` and every M5a/b public type keep their meaning. The pre-existing M5a / M5b tests pass without modification, including the "real completion order with preserved slot-index order in canonical outcome" test from M5b which still demonstrates the slot-index reorder buffer works.
- **Events arrive in real completion order.** Slot 0, when made the slowest, produces the last Phase 1 event. Slots 1 and 2 (both fast) produce the first two Phase 1 events in cooperative-scheduling order.
- **Canonical `phase_one` is still in slot-index order** regardless of which slot's future yielded first. The M5b reorder buffer is the source of truth for canonical order; the event channel is the source of truth for real completion order.
- **One terminal event always.** `PhaseTwoFinished` is emitted exactly once at the end of every run, including when `phase_two` is `None`.
- **Events agree with the canonical outcome.** Per-slot `total_samples` / `judges_run` etc. match `canonical.samples.len()` / `canonical.judge_outcomes.len()` by construction (they're computed from the same outcome reference) and verified explicitly in the real-API test.
- **Failure preservation unchanged.** Failed `ProviderAttempt`s stay in `ModelPhaseOutcome.samples`; failed `JudgeOutcome.result`s stay in `judge_outcomes`. Events count them via `failed_samples` / `judges_failed` but the canonical outcome retains the typed payloads.
- **No `Send` regression.** The orchestrator's future is still non-`Send` (the `&dyn JudgeProvider` borrow contract from M5a is unchanged). No `tokio::spawn`. The `UnboundedSender<PhaseEvent>` is `Send + Sync` for any `T: Send`, but the orchestrator's future capturing it does not become `Send` because the trait-object borrow remains the limiting factor — consistent with the M6b dataset stream contract.
- **Both paths share one body.** `consortium_completion` and `consortium_completion_streaming` are both one-line delegations to `consortium_completion_impl`. Any future orchestration change automatically applies to both.

## Decision

Lock the M5c surface as shipped. Future streaming work (per-event timing, callback-trait variant, bounded-sender backpressure, cancellation hooks) is additive over this surface, not a redesign of it. The shared `consortium_completion_impl` is the natural extension point if more emission sites become useful (e.g., per-sample completion events would emit inside `phase_one_for_slot` alongside the existing `slot_fanout.next().await` site).

## What's Next

- **M7.** Replicate the M2 provider pattern for Deepseek / Kimi K2 / Qwen / Llama. Mechanical breadth expansion; no architectural questions remaining at the orchestration layer after M5c.
- **Provider-side mixed-dimension embedding hardening** (M6a / M6c post-review follow-up). Map malformed mixed-dimension provider responses to `AgnosticEmbeddingError::MalformedResponse` earlier inside each `*_embed_chunk` so the dataset layer's `EmbeddingDimensionMismatch` guard becomes a defensive backstop rather than a primary path.
- **Dataset-layer streaming surface.** `DatasetRun::execute` already returns a `Stream<Item = PromptOutcome>`; a future slice could thread per-prompt `PhaseEvent`s through that stream (e.g., as a `PromptOutcome::Progress(PhaseEvent)` variant) for callers that want orchestration progress alongside dataset-row completion. Out of scope for M5c; mentioned because the surfaces compose naturally.

## See Also

- [M5a Two-Phase Consortium Orchestrator](./2026-05-15-m5a-two-phase-orchestrator.md)
- [M5b Parallel Judge Fan-Out](./2026-05-16-m5b-parallel-judge-fanout.md)
- [M6a Dataset Builder And JSONL Pipeline](./2026-05-15-m6a-dataset-builder-and-jsonl-pipeline.md)
- [M6b Bounded Per-Prompt Parallelism](./2026-05-16-m6b-bounded-per-prompt-parallelism.md)
- [Current Implementation Plan](../plans/2026-05-15-implementation-plan.md)
