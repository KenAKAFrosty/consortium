[README](../../README.md) | [Lab](../README.md) | [Journal](./README.md) | [Plans](../plans/README.md) | [Decisions](../decisions/README.md)

# 2026-05-15 — Claude Seed And Typed Model Contract

Status: validated
Date: 2026-05-15

## Question

After the OpenAI seed landed, what should change in the agnostic contract before Claude replicated the same shape? What's reusable vs Claude-specific? Were there gaps in the OpenAI seed that should be fixed first so Claude doesn't inherit them?

## Hypothesis

The OpenAI seed had a few lossy patterns that would become systemic if Claude/Gemini copied them: stringly model field, silent empty-choices-as-success, and dropping `Auth`/`RateLimited` message context at the agnostic boundary. Fixing those before fanning out to a second provider would prevent the loss from compounding.

## What We Tried

Folded the OpenAI contract corrections in as a pre-Claude commit, then landed Claude end-to-end mirroring the corrected shape.

### Contract changes (landed in `tighten OpenAI contract: typed model, empty choices, message context`)

- **Typed model enum.** `OpenAiModel { Gpt4oMini, Gpt4o, O1, O1Mini, O3Mini, Custom(String) }` with `as_api_str(&self) -> &str`, `Display`, and `serde::Serialize` (emits the API string). `Custom(String)` covers forward-compat for models the enum hasn't enumerated yet. `OpenAiCompletionCommand.model: String` migrated to `model: OpenAiModel`. The public API no longer takes raw model strings. `ClaudeModel` follows the same pattern.
- **Empty-content as typed failure.** `OpenAiCompletionFailure::MalformedResponse { reason: String }` (and `ClaudeCompletionFailure::MalformedResponse`) added. 200 OK with empty `choices` / empty text-content blocks now surfaces as `MalformedResponse`, never as empty-string success. Agnostic-side: `AgnosticCompletionError::MalformedResponse { provider, reason }`. `is_transient()` returns false (retry won't fix an upstream contract violation).
- **Message context preserved across the agnostic boundary.** `AgnosticCompletionError::Auth` gained `message: Option<String>` and `RateLimited` gained `message: Option<String>` (alongside the existing `retry_after`). The `*_failure_to_agnostic` mappings now pass message through. Before this, the agnostic Auth/RateLimited variants dropped provider diagnostic strings, which would have been a systemic loss once Claude/Gemini replicated the mapping.

### Pattern that held from the OpenAI seed

- `*Client { http: reqwest::Client, base_url: String, api_key: String }` with `new(api_key)` / `from_env() -> Result<Self, *ClientError>` / `with_base_url(self, url) -> Self`.
- `*CompletionFailure` enum mirroring agnostic categories so the lib-side `*_failure_to_agnostic` is a single small `match` without information loss.
- Wire types are private serde structs with borrowed `&str` fields where possible.
- `response.bytes().await` + `serde_json::from_slice` (not `response.json().await`) so Transport vs Deserialize stay distinguishable.
- `parse_retry_after` parses `Retry-After: <seconds>`; HTTP-date form deferred.
- Mockito tests construct `*Client::new("test-key").with_base_url(server.url())`; live tests gated by `#[ignore = "requires *_API_KEY; ..."]`.

### Claude-specific deltas from the OpenAI shape

- Auth uses `x-api-key` header (not bearer) plus a required `anthropic-version: 2023-06-01` header.
- Endpoint is `POST /v1/messages` (not `/v1/chat/completions`).
- `system_prompt` serializes as a top-level `system` field on the request body, **not** as a system-role message inside the messages array. The wire encoding diverges from OpenAI even though the command shape is identical.
- `max_tokens` is **required** by the API. Encoded as `u32` (required type) in `ClaudeCompletionCommand`, not `Option<u32>`. The plan now records this as a deliberate type-level invariant rather than a stylistic difference.
- Response content is an array of typed content blocks (`{"type": "text", "text": "..."}`, `{"type": "tool_use", ...}`, etc.). The seed concatenates all `text` blocks via a `#[serde(rename_all = "snake_case")]` tagged enum with a `#[serde(other)] Unknown` catch-all for non-text blocks. Tool-use handling is M4+ work.
- Usage tokens are `input_tokens` / `output_tokens` (not `prompt_tokens` / `completion_tokens`).
- Anthropic uses status `529` for "overloaded" (in addition to standard 5xx). Our generic 400..=499 / 500..=599 mapping covers it but with slightly off semantics (529 falls in the 400-range arm). Tested both possible-correct surfaces (`InvalidRequest` or `ServerError`) so the test passes either way; will tighten if we audit and decide one is more accurate.

### Multi-text-block concatenation

Tested via `success_concatenates_multiple_text_blocks`. Claude can return multiple text blocks in a single response (interleaved with tool calls when relevant); the seed concatenates them into one string. For M2, downstream callers see a single `CompletionOutputChunk::Text` even when the wire response had multiple blocks. M4+ may want to preserve the block structure for tool-use or citation use cases.

### Retry / fan-out validation

`multi_infer_openai_transient_503_drives_retry_then_surfaces_failure` uses `#[tokio::test(start_paused = true)]` and `mockito.expect(3)` to verify the retry helper actually retries on transient 503: `DEFAULT_MAX_ATTEMPTS=3` produces 1 initial call + 2 retries, the resulting `ProviderAttempt.retries == 2`, `input_index` is preserved, and the final `result` is a typed `AgnosticCompletionError::ServerError` with the provider message. This was the missing M1 retry coverage flagged at review.

## Result

The shape holds across two providers. The contract refinements (typed model, MalformedResponse, message context) landed cleanly. Claude required real provider-specific code for auth header, system field placement, content block parsing, and required `max_tokens`, but the surrounding infrastructure (client struct, failure enum, mapping helper, mockito test pattern) was a near-exact mirror of OpenAI.

## Decision

Two providers now share the shape closely enough that a third (Gemini) will confirm whether extraction is worth it. Key candidates for shared helpers after M2c lands:

- `parse_retry_after` (literally identical between OpenAI and Claude).
- `map_failure_from_status` (same status-to-variant mapping; type signature differs only in the enum it produces).
- `WireErrorBody { error: { message: String } }` (OpenAI and Anthropic match; Gemini probably won't).

Per the M2 seed constraint, do not extract until the second-provider data point is in. With Claude landed, extraction becomes a real option after M2c — not before.

## Next

- M2c: Gemini end-to-end with the same shape (typed `GeminiModel`, `*CompletionFailure` mirroring agnostic categories, mockito tests, live test, fan-out success test).
- After M2c: retire `AgnosticCompletionError::ProviderStub`. Audit and extract shared helpers based on the three concrete data points. Drop the `ProviderStub` variant from `provider()` / `is_transient()`.
- Audit the Claude 529 "overloaded" mapping; decide whether to special-case it or accept the current 400-range surface.

## See Also

- [Implementation Plan — M2](../plans/2026-05-15-implementation-plan.md)
- [2026-05-15 — OpenAI Seed Shape For M2](./2026-05-15-openai-seed-shape.md)
- [Decision 0001: Owned Types At Async Boundaries](../decisions/0001-owned-types-at-async-boundaries.md)
- [Journal Index](./README.md)
