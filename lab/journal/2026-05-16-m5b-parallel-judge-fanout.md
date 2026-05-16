[README](../../README.md) | [Lab](../README.md) | [Journal](./README.md) | [Plans](../plans/README.md) | [Decisions](../decisions/README.md)

# 2026-05-16 — M5b Parallel Judge Fan-Out

Status: validated
Date: 2026-05-16

## Question

[M5a](./2026-05-15-m5a-two-phase-orchestrator.md) settled the two-phase orchestrator's typed-outcome shape but ran judges strictly sequentially: per slot in Phase 1, then per judge inside that slot. Phase 1 sampling was already concurrent because `multi_infer` fans out across every `(slot, sample)` pair, but everything after the fan-out — judging the slot, picking a slot winner, running the next slot, judging Phase 2 — happened one `.await` at a time. The load-bearing question for M5b: can we parallelise the slot-level Phase 1 work, the per-slot judges, and the Phase 2 judges *inside* `consortium_completion` without changing the public surface, without spawning tasks, and without weakening any of the M5a determinism invariants (`phase_one` in slot order, `judge_outcomes` in input order, failed judges preserved at their original index, aggregation deterministic)?

## Hypothesis

Yes. The shape locks around three observations:

1. **Each judge invocation is independent of every other judge invocation in the same ranking session.** `judge_rank` is a pure-function-from-blind-candidates pipeline — only the per-judge provider call has side effects, and those are bounded inside the judge impl. Parallel invocation cannot change the set of `OrderedJudgement`s the orchestrator collects; only the order they arrive in. Order can be restored after the fact with a reorder buffer.
2. **Each `phase_one_for_slot(...)` call is independent of every other slot.** Each owns its `Vec<SampleAttempt>` (taken from `by_slot[slot_index]`), its own blind id assignment (sequential `c1`, `c2`, ... per slot), and its own judge invocations. Slots share `judges: &'a [&'a dyn JudgeProvider]` but only read from it.
3. **The fan-out factor is bounded by user configuration.** Number of slots and number of judges are both configured up front and stay in the single digits in practice. The orchestrator can hold all of them in flight at once with no explicit concurrency cap — it does not need an M6b-style `parallelism` knob.

Lock the slice around four shapes:

1. **Public surface unchanged.** `consortium_completion(slots, judges) -> ConsortiumOutcome` keeps its signature. Every public type (`ModelPhaseOutcome`, `CrossModelPhaseOutcome`, `JudgeOutcome`, `JudgedSample`, `CrossModelCandidate`, ...) keeps its meaning. Field-order invariants are preserved by reorder buffers rather than by sequential execution.
2. **In-task `FuturesUnordered` + reorder buffer at every fan-out site.** Three sites: (a) per-slot judges inside `phase_one_for_slot`, (b) per-slot `phase_one_for_slot` calls inside `consortium_completion`, (c) per-judge calls inside `phase_two_outcome`. Each `FuturesUnordered` holds `LocalBoxFuture<'_, (usize, T)>` so the index travels with the result. After the inner stream drains, a `Vec<Option<T>>` indexed by the original input position is unwrapped into the deterministic output vector.
3. **No `tokio::spawn`; the orchestrator's future stays non-`Send`.** The M5a-era reason — `&dyn JudgeProvider` is non-`Sync` even when the trait itself declares `Send + Sync` as supertraits, because supertrait bounds do not lift onto the trait object type — still applies. M5b uses `LocalBoxFuture` / `.boxed_local()` so the in-flight futures match that non-`Send` reality. The M6b dataset stream's non-`Send` contract is therefore untouched.
4. **Shared helper for the two judge loops.** Phase 1 per-slot judges and Phase 2 cross-model judges had byte-for-byte identical loop bodies. M5b extracts them into a crate-private `invoke_judges_in_parallel(blind, judges) -> (Vec<JudgeOutcome>, Vec<OrderedJudgement>)` and reuses it from both sites.

Defer (still):

- M5c streaming `PhaseEvent` surface — an additive view over the same orchestrator, not a refactor of it.
- M5d multi-prompt orchestration — already lives one layer up in the dataset module (M6a/M6b).
- A `parallelism` knob on `consortium_completion`. Slot and judge counts are configured-up-front and small; the existing M6b per-prompt cap is the right place to limit total concurrency, not here.
- Making the orchestrator's future `Send`. That would require either pinning each in-flight subtree to its own `tokio::spawn` (rejected by the slice brief) or changing the `JudgeProvider` trait surface (out of M5b scope and not load-bearing for current use cases).

## What We Tried

Single-module change in [`src/orchestrator/mod.rs`](../../src/orchestrator/mod.rs). No changes to `src/judge/`, `src/dataset/`, or any provider client. No new dependencies.

### Per-slot judges in Phase 1 (and Phase 2 cross-model judges)

Old shape (Phase 1; Phase 2 was identical):

```rust
let mut judge_outcomes: Vec<JudgeOutcome> = Vec::with_capacity(judges.len());
let mut successful: Vec<OrderedJudgement> = Vec::new();
for judge in judges {
    let label = judge.label().to_string();
    let result = judge_rank(&blind, |req| judge.invoke(req)).await;
    if let Ok(judgement) = &result {
        successful.push(judgement.clone());
    }
    judge_outcomes.push(JudgeOutcome { judge_label: label, result });
}
```

New shape — collapsed onto a shared helper called from both sites:

```rust
async fn invoke_judges_in_parallel<'a>(
    blind: &[BlindCandidate],
    judges: &'a [&'a dyn JudgeProvider],
) -> (Vec<JudgeOutcome>, Vec<OrderedJudgement>) {
    let mut in_flight: FuturesUnordered<LocalBoxFuture<'_, (usize, JudgeOutcome)>> =
        FuturesUnordered::new();
    for (judge_index, judge) in judges.iter().enumerate() {
        let label = judge.label().to_string();
        in_flight.push(async move {
            let result = judge_rank(blind, |req| judge.invoke(req)).await;
            (judge_index, JudgeOutcome { judge_label: label, result })
        }.boxed_local());
    }

    let mut buf: Vec<Option<JudgeOutcome>> =
        (0..judges.len()).map(|_| None).collect();
    while let Some((idx, outcome)) = in_flight.next().await {
        buf[idx] = Some(outcome);
    }
    let judge_outcomes: Vec<JudgeOutcome> = buf
        .into_iter()
        .map(|o| o.expect("every judge future writes its judge_index slot exactly once"))
        .collect();
    let successful: Vec<OrderedJudgement> = judge_outcomes
        .iter()
        .filter_map(|jo| jo.result.as_ref().ok().cloned())
        .collect();
    (judge_outcomes, successful)
}
```

Both call sites become a single line: `let (judge_outcomes, successful) = invoke_judges_in_parallel(&blind, judges).await;`. The `OrderedJudgement` clone count is the same as before — M5a was already cloning inside the success branch.

### Slot-level Phase 1 fan-out

Old shape:

```rust
let mut phase_one: Vec<ModelPhaseOutcome> = Vec::with_capacity(slots.len());
for (slot_index, slot) in slots.iter().enumerate() {
    let slot_samples = std::mem::take(&mut by_slot[slot_index]);
    let provider = slot.input.provider();
    let outcome = phase_one_for_slot(slot, provider, slot_samples, judges).await;
    phase_one.push(outcome);
}
```

New shape:

```rust
let mut slot_fanout: FuturesUnordered<LocalBoxFuture<'_, (usize, ModelPhaseOutcome)>> =
    FuturesUnordered::new();
for (slot_index, slot) in slots.iter().enumerate() {
    let slot_samples = std::mem::take(&mut by_slot[slot_index]);
    let provider = slot.input.provider();
    slot_fanout.push(async move {
        let outcome = phase_one_for_slot(slot, provider, slot_samples, judges).await;
        (slot_index, outcome)
    }.boxed_local());
}

let mut phase_one_buf: Vec<Option<ModelPhaseOutcome>> =
    (0..slots.len()).map(|_| None).collect();
while let Some((slot_index, outcome)) = slot_fanout.next().await {
    phase_one_buf[slot_index] = Some(outcome);
}
let phase_one: Vec<ModelPhaseOutcome> = phase_one_buf
    .into_iter()
    .map(|o| o.expect("every slot future writes its slot_index slot exactly once"))
    .collect();
```

`by_slot[slot_index]` is consumed by `std::mem::take` *before* the future is pushed, so each future owns its `Vec<SampleAttempt>` and the in-flight set never shares mutable state. `slot: &ConsortiumSlot<'a>` and `judges: &'a [...]` stay borrowed for the function body's lifetime — the futures' lifetimes are anchored on `'_` in the `FuturesUnordered` type parameter, which the compiler resolves to the relevant inner scope.

### Why `LocalBoxFuture` and not `BoxFuture`

`JudgeProvider: Send + Sync` makes any concrete impl `Send + Sync`, but it does *not* propagate to the trait object. `dyn JudgeProvider` does not pick up `Sync` from its supertrait, so `&dyn JudgeProvider` is neither `Sync` nor `Send`. An async block that captures `&dyn JudgeProvider` across `.await` is therefore non-`Send` — which is why the M6b dataset stream is non-`Send` already (the in-task `FuturesUnordered` there is fine; spawning it onto a Send runtime executor would not be).

The M5b in-flight futures inherit this non-`Send`-ness, so they're stored as `LocalBoxFuture<'_, T> = Pin<Box<dyn Future<Output = T> + 'a>>` via `.boxed_local()`. This was a deliberate choice: keep the same constraint that already exists, do not impose new bounds on `JudgeProvider` or its callers.

### Module doc update

The orchestrator module doc grew a new "Concurrency (M5b)" section spelling out which fan-outs are concurrent, the non-`Send` rationale, and the public-surface guarantee that order and provenance are preserved regardless of internal completion order.

## Tests

`cargo test --lib` is 119 passed / 0 failed / 5 ignored (+4 over M6c's 115). All pre-existing tests stay green at the M5a outcome shape.

Four new tests in `src/orchestrator/mod.rs#tests`:

1. **`phase_one_judges_run_in_parallel_within_a_slot`.** Three `BarrierJudge`s each call `tokio::sync::Barrier::new(3).wait().await`. The barrier only releases once exactly three judges are concurrently suspended — direct evidence of parallel fan-out. An outer `tokio::time::timeout(10s, ...)` turns a sequential-execution regression into a clean test failure instead of a hang. After the orchestrator returns, the test asserts `max_seen == parallelism = 3` (high-water mark of concurrent judges), `active == 0` (all judges released their slot), and `judge_outcomes` labels are exactly `[j1, j2, j3]` regardless of which judge crossed the barrier first.

2. **`phase_two_judges_run_in_parallel_with_preserved_order`.** Two slots × one sample each, so Phase 1 singleton short-circuits skip judging entirely and the only judge invocations happen in Phase 2. Three judges pinned at a `Barrier(3)` prove cross-model fan-out is concurrent; `phase_two.judge_outcomes` is asserted `[j1, j2, j3]` after reorder. The singleton-Phase-1 setup is deliberate: it isolates Phase 2 concurrency from Phase 1 concurrency.

3. **`phase_one_slots_run_concurrently_with_preserved_slot_order`.** Three slots with a single shared `OrderingJudge`. Slot 0's judge sleeps 60s (virtual under `#[tokio::test(start_paused = true)]`); slots 1 and 2 are fast. The judge flips `slot_1_finished_before_slot_0` when slot 1's judging completes before `slot_0_finished` is set — which is only possible if slot futures run concurrently. The test asserts that flag is `true`, that all three slots produced winners, and that `phase_one`'s `model_label`s are `["slot-0", "slot-1", "slot-2"]` even though slot 0 finished judging last in virtual time.

4. **`parallel_phase_one_preserves_failed_judges_at_their_input_index`.** Four judges interleaved `[Ok, Err, Ok, Err]` (using the same `FnJudge` test double from M5a's tests). Under parallel execution and arbitrary completion order, the reorder buffer must restore the original index alignment. The test asserts each `judge_outcomes[i]` has the matching label *and* the matching success state, and that the failed judges still carry their typed `JudgementError::Provider(AgnosticCompletionError::Auth { .. })` payload (i.e., the reorder did not collapse error variants). Aggregation still runs across the two successful judges and produces a winner.

A new `BarrierJudge` test double (parallel to M6b's `ConcurrencyJudge`) lives in the orchestrator test module: it owns an `Arc<tokio::sync::Barrier>` plus shared `AtomicUsize` active / max-seen counters and a `Mutex<Vec<String>>` completion-order log, and pushes its label onto the log every time its invocation crosses the barrier. The completion-order log isn't asserted on (completion order is unpredictable under cooperative scheduling and that's the point), but it stays in the test double so a future debug session can inspect what actually happened.

## Verified Properties

- **Public surface unchanged.** `consortium_completion(slots, judges) -> ConsortiumOutcome` and every public type keep their M5a meaning. The pre-existing M5a tests (`happy_path_two_phase_picks_a_winner_traceable_to_originating_sample`, `partial_failure_preserves_failed_attempts_and_failed_judges`) stay green without any modification.
- **`phase_one` is in slot order.** Even when slot 1 finishes judging before slot 0 in real (virtual) time, `phase_one[0]` is the slot-0 outcome and `phase_one[1]` is the slot-1 outcome.
- **`judge_outcomes` is in original judge order.** Verified at both fan-out sites (Phase 1 per slot and Phase 2 cross-model), with both all-success and interleaved-success/failure inputs.
- **Failed judges are preserved at their original index.** A failing `j2` does not migrate to `judge_outcomes[3]` just because it completed before `j3`; its `JudgementError::Provider(Auth)` payload is identical to the M5a behaviour.
- **Aggregation is deterministic.** `aggregate_rankings` consumes successful judgements as a set with a documented tie-break; parallel fan-out cannot change which judgements are successful, so the aggregation output is bit-for-bit identical to the M5a behaviour. Verified by the happy-path M5a test continuing to pass.
- **Concurrency really happened.** Both per-judge and per-slot concurrency are proven by mechanisms that would deadlock or fail under sequential execution: a `tokio::sync::Barrier` sized to the expected concurrency; an "out-of-order completion" flag that would never trip if slot judging were serialised.
- **No `Send` regression.** The orchestrator's future is still non-`Send` (the M6b dataset stream's non-`Send`-ness is unchanged). No `tokio::spawn`. All in-task cooperative concurrency.

## Decision

Lock the M5b internals as shipped. Future concurrency work in this layer (per-judge timeout policy, per-slot timeout policy, cancellation on first winner) is additive over this surface, not a redesign of it. The shared `invoke_judges_in_parallel` helper is the natural extension point if a `parallelism` knob ever becomes load-bearing.

## What's Next

- **M5c.** Streaming `PhaseEvent` surface over the same orchestrator. Likely consumes the per-slot Phase 1 outcomes plus the Phase 2 outcome as discrete events. Now that fan-out is concurrent, "first slot finished" events arrive in real (not artificial sequential) order, which is the natural granularity for the streaming surface.
- **M7.** Replicate the M2 provider pattern for Deepseek / Kimi K2 / Qwen / Llama. Mechanical breadth expansion.
- **Provider-side mixed-dimension hardening** (M6a / M6c post-review follow-up). Map malformed mixed-dimension provider responses to `AgnosticEmbeddingError::MalformedResponse` earlier inside each `*_embed_chunk`.

## See Also

- [M5a Two-Phase Consortium Orchestrator](./2026-05-15-m5a-two-phase-orchestrator.md)
- [M6b Bounded Per-Prompt Parallelism](./2026-05-16-m6b-bounded-per-prompt-parallelism.md)
- [M6c Embedder Auto-Chunking](./2026-05-16-m6c-embedder-auto-chunking.md)
- [Current Implementation Plan](../plans/2026-05-15-implementation-plan.md)
