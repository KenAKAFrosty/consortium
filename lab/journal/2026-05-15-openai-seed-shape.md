[README](../../README.md) | [Lab](../README.md) | [Journal](./README.md) | [Plans](../plans/README.md) | [Decisions](../decisions/README.md)

# 2026-05-15 — OpenAI Seed Shape For M2

Status: validated
Date: 2026-05-15

## Question

What concrete shape should the real provider clients take, and what is reusable vs. provider-specific? Validate on one provider before fanning out to three.

## Hypothesis

A typed per-provider failure enum that mirrors `AgnosticCompletionError` variants, plus a thin per-provider conversion helper into the agnostic error, would land cleanly without forcing premature shared abstractions.

## What We Tried

Implemented `src/ai_client_apis/openai/mod.rs` as the M2 seed:

- `OpenAiClient { http: reqwest::Client, base_url: String, api_key: String }` with `new`, `from_env`, and `with_base_url(self, ...) -> Self` (builder-style override so `mockito` tests redirect the base URL without hacks).
- `OpenAiCompletionCommand { model, system_prompt: Option<String>, messages: Vec<OpenAiMessage>, max_tokens: Option<u32>, temperature: Option<f32> }` with `Default`.
- `OpenAiMessage { role: OpenAiRole, content: String }` and `OpenAiRole { User, Assistant }`. System role is exposed via `system_prompt` and prepended at wire-time.
- `OpenAiCompletionFailure` (thiserror): `Transport(reqwest::Error)`, `Deserialize(serde_json::Error)`, `Auth { message }`, `RateLimited { retry_after, message }`, `InvalidRequest { message }`, `ServerError { status, message }`. Variants intentionally mirror `AgnosticCompletionError` so the lib-side `openai_failure_to_agnostic` conversion is a straight match without information loss.
- Private wire types for the request body, response body, and OpenAI error body, all `serde`-derived.
- HTTP path uses `response.bytes().await` + `serde_json::from_slice` so transport vs deserialize failures stay distinguishable (unlike `response.json()` which collapses both into `reqwest::Error`).
- `parse_retry_after` currently parses only the `Retry-After: <seconds>` form. HTTP-date form is deferred — would require `chrono` or hand-parsing, and OpenAI emits seconds in practice.
- 7 `mockito` tests cover success / 401 / 403 / 429-with-`Retry-After` / 400 / 503 / malformed JSON. One `#[ignore]` live test gated on `OPENAI_API_KEY`. Plus an end-to-end test in `lib.rs` that runs `multi_infer` against a mockito-backed `OpenAiClient` and asserts real text + token counts come through.

## Result

The shape held. The per-provider failure enum mapping cleanly to the agnostic error keeps both layers honest: provider-specific tests can assert against typed variants without going through the agnostic layer, and the agnostic conversion is a single small function that's easy to keep in sync. `with_base_url` taking `self` and returning `Self` is the right ergonomics for the rare "I need to point this somewhere else" case (tests, internal proxies) without polluting the prod constructor surface.

What is reusable: the field shape (`http`, `base_url`, `api_key`), the constructor pattern (`new`/`from_env`/`with_base_url`), the per-provider failure enum mirroring agnostic variants, the bytes-then-serde split. What is not reusable yet: nothing has been extracted. Per the user's "minimal shared extraction" rule, helpers (e.g., common `parse_retry_after`, common status mapping) should only be pulled up once the second provider proves the same shape applies.

## Decision

Use this shape as the template for Claude (M2b) and Gemini (M2c). The plan was updated to call out the seed-first sequencing and the journal reference. Do not extract shared helpers until after the second provider lands.

## Next

- M2b: Claude client mirroring this shape, accounting for Anthropic's `x-api-key` header instead of bearer auth, the `anthropic-version` header, and the messages/system body shape.
- M2c: Gemini client mirroring this shape, accounting for Google's `?key=` query auth and the very different request/response body.
- After M2b: revisit whether `parse_retry_after`, the 4xx/5xx-to-failure mapping, or the bytes-then-serde glue should move to a shared helper. Do not extract before that data point.
- Eventually: drop `AgnosticCompletionError::ProviderStub` once Claude and Gemini land, and tighten the agnostic error enum.

## See Also

- [Implementation Plan — M2](../plans/2026-05-15-implementation-plan.md)
- [Decision 0001: Owned Types At Async Boundaries](../decisions/0001-owned-types-at-async-boundaries.md)
- [Journal Index](./README.md)
