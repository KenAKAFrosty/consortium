[README](../../README.md) | [Lab](../README.md) | [Decisions](./README.md) | [Plans](../plans/README.md) | [Journal](../journal/README.md)

# 0001 — Owned Types At Async Boundaries

Status: accepted
Date: 2026-05-15

## Decision

Cross-provider boundary types should use owned payloads rather than borrowed references. In practice, this means public agnostic output types should prefer `String` and `bytes::Bytes` or `Vec<u8>` at async, network, and persistence boundaries.

## Why

Provider responses arrive from async HTTP calls and deserialized payloads. Borrow-heavy output contracts make those boundaries harder to model, complicate lifetimes across futures, and are a poor fit for reusable results that may be retried, buffered, streamed, or written out.

## Consequences

- Early `src/lib.rs` scaffolding is allowed to change shape before provider implementations land.
- Borrowing is still fine internally when it reduces allocations without complicating the design.
- Reviewers should push back on new borrowed output contracts that cross async or provider boundaries without a strong reason.

## See Also

- [Implementation Plan](../plans/2026-05-15-implementation-plan.md)
- [Initial Repo Read And Planning Journal](../journal/2026-05-15-initial-repo-read-and-planning.md)
