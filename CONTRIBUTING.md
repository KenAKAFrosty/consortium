[README](./README.md) | [Lab](./lab/README.md) | [Plans](./lab/plans/README.md) | [Journal](./lab/journal/README.md) | [Decisions](./lab/decisions/README.md)

# Contributing

This project is building a Rust-native consortium inference and dataset-generation crate. Contributions should preserve a high bar for correctness, robustness, and performance as the codebase grows.

This document applies to both human contributors and automated contributors.

## Design Priorities

In general, optimize in this order:

1. Correctness
2. Robustness
3. Performance
4. Ergonomics

API ergonomics matter, but not at the expense of correctness, debuggability, or throughput.

## Core Expectations

- Make invalid states unrepresentable.
- Prefer explicit types over conventions.
- Preserve partial failure information; do not silently drop it.
- Favor designs that compose well under concurrency and retries.
- Optimize obvious hot-path waste early; benchmark before complex optimization.

## API Design

- Prefer enums, structs, and newtypes over booleans and stringly-typed APIs.
- Avoid stringly-typed errors. Do not use raw `String` or ad hoc text as the primary error contract when a typed error enum or structured error value is appropriate.
- Prefer config structs over long parameter lists.
- Public APIs should feel Rust-native. Do not shape the API surface around OpenAI compatibility or generic OpenAPI-style schemas unless there is a concrete integration reason.
- Use builder patterns when construction has multiple optional knobs or invariants to enforce.
- If a type has invariants, encode them in the type system where practical instead of relying on comments or call-order assumptions.

## Type Design

- Prefer small, explicit domain types over passing around loosely related primitives.
- Use enums to model real state transitions and provider/judge/result variants.
- Use `Option` only when absence is semantically valid. Do not use `Option` to defer making a better type.
- Use `Result` for fallible operations; do not encode error states in sentinel values.
- Use owned types at async, network, thread, and persistence boundaries.
- Borrow internally when it reduces allocations without complicating the design.

## Pattern Matching

- Prefer exhaustive `match` statements.
- Avoid wildcard arms unless there is a documented reason.
- If a wildcard or fallback arm is necessary, leave a brief comment explaining why.
- When an enum is expected to grow, make that intention explicit with type design and comments rather than relying on vague fallback behavior.

## Error Handling

- Library code should not `panic!` for expected failure modes.
- Errors should be typed, structured, and actionable.
- Attach enough context to make failures diagnosable without forcing callers to reproduce them blindly.
- Preserve provider-attempt failures, judge failures, and prompt-level failures as first-class outcomes where applicable.
- Do not collapse multiple distinct failure modes into generic text if they need different caller behavior.

## Robustness And Retries

- Retries should be policy-driven, not ad hoc loops scattered through the codebase.
- Retry only transient failures such as rate limits, timeouts, and transport instability.
- Do not retry non-idempotent operations blindly.
- Keep retry ownership tied to the operation being retried so in-flight concurrent work remains understandable.
- When this project standardizes on a retry primitive, use it consistently. `backon` is the likely default unless a better project-wide choice emerges.
- Surface retry counts and terminal failure reasons where useful.

## Async And Concurrency

- Do not block in async contexts.
- Avoid hidden runtime ownership in library code; callers should manage the runtime.
- Use bounded concurrency when fan-out can grow with input size.
- Treat cancellation, timeout behavior, and backpressure as part of the design, not afterthoughts.
- Do not spawn detached tasks without a clear ownership and lifecycle story.
- Prefer structured concurrency patterns over fire-and-forget work.

## Performance And Allocation Discipline

- Avoid unnecessary allocations, clones, buffering, and JSON reserialization.
- Reuse clients and other reusable state where practical.
- Prefer streaming or incremental processing when full materialization is unnecessary.
- Be deliberate about hot-path `String`, `Vec`, and map creation.
- If an optimization makes the code materially harder to reason about, justify it with evidence.

## Provider And Protocol Work

- Keep provider-specific code isolated behind clear boundaries.
- Normalize provider outputs into shared internal types without erasing important semantics.
- Preserve provider metadata that may matter for retries, metrics, or debugging.
- Do not let one provider's quirks dictate the shape of the entire crate unless the tradeoff is explicit and justified.

## Testing

- Add unit tests for nontrivial logic.
- Add regression tests for bugs that were fixed.
- Use mocked tests for provider parsing and transport behavior.
- Gate real network/API tests behind `#[ignore]` or equivalent opt-in mechanisms.
- Prefer deterministic tests. If randomness is required, make it reproducible.
- Use property-style tests where they add value, especially for ranking, aggregation, selection, and retry behavior.

## Dependencies

- New dependencies require a reason.
- Prefer focused, well-maintained crates over large abstraction-heavy additions.
- Avoid adding dependencies solely to save a small amount of local code when the dependency meaningfully expands compile time, API surface, or maintenance burden.
- If a dependency becomes foundational, use it consistently rather than creating multiple competing patterns.

## Documentation

- Keep docs concrete.
- Document invariants, error semantics, retry behavior, and performance-sensitive behavior.
- When adding a new public abstraction, document why it exists and what problem it solves.
- Keep examples realistic and aligned with the intended architecture.

## Lab Workflow

- Put scoped, forward-looking work plans in `lab/plans/`.
- Put meaningful experiments, findings, dead ends, and course corrections in `lab/journal/`.
- Put durable project conclusions in `lab/decisions/`.
- Every markdown file should include a simple top navigation block.
- Every leaf document should link back up and link sideways to relevant related notes.
- Add each new lab note to its parent `README.md` index so docs do not become orphaned.
- If a note is superseded, mark it clearly and link the replacement.
- If a change materially alters direction, invalidates an assumption, or settles a recurring design question, update the relevant lab note as part of the same contribution.

## Pull Requests

- Keep changes scoped.
- Do not mix unrelated refactors with behavior changes.
- Update tests and docs alongside behavior changes.
- Avoid unrelated formatting churn.
- If a public API changes, call it out explicitly in the PR description.

## Review Checklist

Before considering a contribution ready, check:

- Are invalid states modeled out of the design where practical?
- Are enums used where they clarify behavior and state?
- Are matches exhaustive unless there is a documented reason otherwise?
- Are errors typed and actionable rather than stringly-typed?
- Are retries applied only where appropriate?
- Is async code non-blocking and concurrency-bounded?
- Are obvious extra allocations and clones avoided?
- Are tests covering the new behavior and likely regressions?
- Are relevant lab notes updated for meaningful plan, experiment, or decision changes?
- Does the change preserve or improve clarity instead of adding abstraction for its own sake?

## Tooling Expectations

Contributions should be compatible with:

- `cargo fmt`
- `cargo clippy --all-targets --all-features -- -D warnings`
- `cargo test`

As CI is added, treat these checks as the minimum quality gate rather than a suggestion.

## See Also

- [README](./README.md)
- [Lab Home](./lab/README.md)
- [Current Implementation Plan](./lab/plans/2026-05-15-implementation-plan.md)
