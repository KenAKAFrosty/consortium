[README](../../README.md) | [Lab](../README.md) | [Journal](./README.md) | [Plans](../plans/README.md) | [Decisions](../decisions/README.md)

# 2026-05-15 — Stabilization: Secrets, Backon, And Constructor Shape

Status: validated
Date: 2026-05-15

## Question

Before M3 (embeddings + diversification) gets layered on top of the provider clients, what cleanup debt should be paid down?

## Hypothesis

Three concrete items had accumulated since M2: provider API keys were stored as `String` (no redaction at rest or in Debug output); the retry implementation had its own hand-rolled splitmix-jitter machinery that should be a known crate (`backon`); and the constructor shape forced tests through `new(...).with_base_url(...)`, an awkward "allocate the default, then overwrite" pattern. None blocked correctness; each was the kind of debt that compounds when more layers are added on top.

## What We Tried

### Secrets

- Added `secrecy = "0.10"`. Switched each client's `api_key: String` to `api_key: SecretString`.
- Constructors accept `impl Into<SecretString>` so callers can still pass `String` directly (the most common ergonomic).
- Wire calls now use `.expose_secret()` at the single bearer-auth / `x-api-key` / `x-goog-api-key` site per provider. The exposure surface is one line per provider.
- Debug output: `SecretString`'s `Debug` impl emits `Secret([REDACTED])`, so the `#[derive(Debug)]` on each client struct now redacts automatically. Added a sanity test `debug_output_redacts_api_key` in OpenAI tests to guard against accidental future regression (e.g., a custom Debug impl that prints fields directly).

### Retry → backon

- Added `backon = "1"`.
- Removed: `run_with_retry`, `RetryOutcome`, `delay_for_attempt`, `jitter_within`, `JITTER_COUNTER`, `splitmix` hash code. About 70 lines of bespoke retry/jitter machinery gone.
- Retry orchestration moved directly into `build_attempt`. The loop is now:
  - Build a `backon::ExponentialBuilder::default().with_min_delay(...).with_max_times(MAX_RETRIES).with_jitter().build()` iterator at the top.
  - For each non-transient error, break immediately.
  - For each transient error, take `backoff.next()`; if `None`, retry budget is exhausted → break with the error.
  - If `next()` returned `Some(backoff_delay)`, choose `retry_after_override(&err).unwrap_or(backoff_delay)` and sleep that long. Increment `retries`. Repeat.
- Key subtlety: `backon::Backoff` is just `Iterator<Item = Duration>` and does not see the error. To honor `RateLimited::retry_after` we kept a tiny `retry_after_override(&AgnosticCompletionError)` helper and the override path. Importantly, we still call `backoff.next()` and consume the iterator slot even when the override fires, so the retry budget bounds total attempts regardless of which delay was actually used. Without that, a stream of `RateLimited` errors with `Retry-After` set could loop forever bypassing `MAX_RETRIES`.
- Constants: `MAX_RETRIES = 2` (1 initial + 2 retries, matching the previous `DEFAULT_MAX_ATTEMPTS = 3`) and `BASE_RETRY_DELAY = 100ms`. The existing `multi_infer_openai_transient_503_drives_retry_then_surfaces_failure` test (asserts `retries == 2` after exhaustion) still passes unchanged — confirms behavior preserved.

### Constructor shape

Replaced three constructors (`new`, `from_env`, `with_base_url` builder) with four explicit constructors per provider:

- `new(api_key) -> Self` — explicit key, default base URL.
- `new_with_base_url(api_key, base_url) -> Self` — explicit key, explicit URL. The test path now uses this consistently — `OpenAiClient::new_with_base_url("test-key", server.url())` is one call instead of two.
- `from_env() -> Result<Self, *ClientError>` — env key, default base URL.
- `from_env_with_base_url(base_url) -> Result<Self, *ClientError>` — env key, explicit URL. Covers the prod use case where the env key is the normal path but the URL needs override (e.g., a custom inference gateway).

Dropped: `with_base_url(self, ...) -> Self` builder method. All three providers updated.

### Role enum serialization

Claude and Gemini role enums (`{User, Assistant}` / `{User, Model}`) mapped 1:1 to lowercase wire strings, with a hand-written `as_wire_str(self) -> &'static str` helper used at the single wire-construction site. Replaced with `#[derive(Serialize)] #[serde(rename_all = "lowercase")]` and let the wire message struct hold the role enum directly (the enum is `Copy`). Net: about 10 lines removed per provider, no behavior change.

OpenAI stays on the `&'static str` + `as_wire_str` shape because its wire path also needs a "system" role that the public `OpenAiRole` enum does not expose. Replacing `as_wire_str` would require either:
- A separate internal `WireRole` enum with a `From<OpenAiRole>` impl plus a `WireRole::System` variant — more code than what we have.
- Broadening the public `OpenAiRole` to include `System` — but `system_prompt` is exposed as its own command field by design; adding `System` to the role enum would give callers two ways to set system content and invite divergence.

Neither is "cleanly replace manual impl noise with serde derives," so OpenAI keeps the existing shape.

### Documentation: Deserialize semantics

Added doc comments on `AgnosticCompletionError::Deserialize` and `MalformedResponse` clarifying that both variants describe provider-wire-protocol violations (JSON didn't decode against our schema, or decoded but failed a semantic invariant like empty `choices`). Explicitly not for malformed structured text produced by the LLM itself — if that ever becomes a modeled concern (e.g., JSON-mode schema violations), it gets its own variant rather than overloading these.

## Result

All 35 tests pass (was 34 before the new redaction sanity test). No behavior change observable from the test surface; the `multi_infer_openai_transient_503_drives_retry_then_surfaces_failure` test still asserts `retries == 2` and a typed `ServerError` with the original status, proving the backon-based retry preserves the previous contract.

Code shape:

- `src/lib.rs` retry path is now ~50 lines smaller. The bespoke jitter machinery (about 30 lines of splitmix hashing + atomic counter) is gone.
- Each provider client's constructor block is ~15 lines (was ~12); the gain in explicit constructors offsets the loss of `with_base_url`. Test call sites are net shorter.
- `Cargo.lock` grew by `backon` + `secrecy` + their transitive deps, but the source tree is net smaller.

## Decision

Keep `backon` for retry backoff computation. The `Backoff` trait's lack of error-awareness is a real limitation for `Retry-After` honoring, but the workaround (a 6-line `retry_after_override` helper plus consuming `backoff.next()` for the budget) is small enough to live with. If `backon` later exposes an error-aware delay hook, fold that in. Otherwise, this stays.

Keep `secrecy::SecretString` for at-rest key storage. The `expose_secret()` call site count is exactly 1 per provider (the request builder), and the redaction is automatic via `Debug`. The cost is the `impl Into<SecretString>` ergonomic shim, which is cheap.

The four-constructor shape (`new`, `new_with_base_url`, `from_env`, `from_env_with_base_url`) is more discoverable than `new(...).with_base_url(...)` chaining and removes the "allocate default URL then overwrite" awkwardness. Worth the extra method per provider.

## Next

- M3: embeddings + diversification per the existing plan. `Embedder` trait, `OpenAiEmbedder` impl reusing the OpenAI `reqwest::Client`, `select_diverse` with `Centroid` / `FarthestPointSampling` strategies.

## See Also

- [Implementation Plan — M2, post-M2c stabilization](../plans/2026-05-15-implementation-plan.md)
- [2026-05-15 — Gemini Seed And Extraction Checkpoint](./2026-05-15-gemini-seed-and-extraction-checkpoint.md)
- [Journal Index](./README.md)
