[README](../../README.md) | [Lab](../README.md) | [Journal](./README.md) | [Plans](../plans/README.md) | [Decisions](../decisions/README.md)

# 2026-05-15 — Gemini Seed And Extraction Checkpoint

Status: validated
Date: 2026-05-15

## Question

With OpenAI and Claude landed, does Gemini exhibit the same surface shape closely enough that the shared candidates (`parse_retry_after`, `map_failure_from_status`, `WireErrorBody`, the failure-enum variant list) actually deserve extraction — or does the third data point reveal divergence we'd have papered over with a premature abstraction?

## Hypothesis

The pattern would hold. Gemini's wire shape is the most divergent of the three (URL-embedded model, different role naming, different request body structure), so if the per-provider Failure enum / status mapping / retry-after parsing still apply cleanly here, they apply everywhere.

## What We Tried

Implemented `src/ai_client_apis/gemini/mod.rs` end-to-end mirroring the OpenAI/Claude seed shape, kept the per-provider failure enum, and observed which parts of the file are now triplicated.

### What held across all three providers (extraction candidates)

- **`parse_retry_after`** is byte-identical. Same `headers.get(RETRY_AFTER)?.to_str().ok()?.parse::<u64>().ok().map(Duration::from_secs)` in every provider. Cleanest extraction candidate.
- **`map_failure_from_status`** is the same status-to-variant skeleton in every provider: 401/403 → Auth, 429 → RateLimited with `parse_retry_after`, 400..=499 → InvalidRequest, 500..=599 → ServerError, fallback → ServerError. Diverges only in the failure enum it constructs. Lifts as either a small generic helper taking constructor closures per category, or via a `trait ProviderFailureBuilder` with one method per agnostic category. The closure form is shorter; the trait form is named and more discoverable. Either works.
- **`WireErrorBody { error: { message: String } }`** — surprisingly, OpenAI, Anthropic, and Google all share `{"error":{"message":"..."}}` as the error body shape. Google's body also carries `code` / `status` fields but we never read them. Lifts cleanly.
- **The `*CompletionFailure` variant set** is now identical across providers: `Transport(reqwest::Error)`, `Deserialize(serde_json::Error)`, `Auth { message: Option<String> }`, `RateLimited { retry_after: Option<Duration>, message: Option<String> }`, `InvalidRequest { message: String }`, `ServerError { status: u16, message: Option<String> }`, `MalformedResponse { reason: String }`. The actual `*_failure_to_agnostic` conversion is mechanical at this point — same match arms, different namespace.
- **The constructor / config shape** (`new` / `from_env` / `with_base_url`) is identical across providers. A `trait Provider` or builder could lift it, but the M8 plan already covers genericification — not extracting it now.
- **`response.bytes().await` + `serde_json::from_slice` (not `response.json().await`)** to keep Transport / Deserialize distinguishable.

### What's provider-specific (will not extract)

- **URL construction.** OpenAI hits a single fixed endpoint (`/v1/chat/completions`). Claude hits a single fixed endpoint (`/v1/messages`). Gemini's URL contains the model id in the path (`/v1beta/models/{model}:generateContent`). Generalizing this is more work than it's worth.
- **Auth scheme.** OpenAI uses bearer (`Authorization: Bearer ...`). Claude uses `x-api-key` + a required `anthropic-version` header. Gemini uses `x-goog-api-key`. Three different patterns — not worth lifting.
- **Request body shape.** All three diverge in field names and nesting (`messages` vs `messages` vs `contents`, top-level `system` vs `system_prompt` vs `systemInstruction`, flat `max_tokens` vs flat `max_tokens` vs nested `generationConfig.maxOutputTokens`). The wire types stay provider-private.
- **Response content extraction.** OpenAI: `choices[0].message.content`. Claude: `content[].text` (with non-text blocks). Gemini: `candidates[0].content.parts[].text` (with non-text parts). Each provider parses and concatenates differently. Provider-specific.
- **Token usage field names.** `prompt_tokens` / `completion_tokens` (OpenAI), `input_tokens` / `output_tokens` (Claude), `promptTokenCount` / `candidatesTokenCount` (Gemini). All map to the same two-number agnostic shape but each provider's wire type stays separate.

### Gemini-specific deltas worth noting

- The model id is part of the URL path, not the request body. `command.model.as_api_str()` is interpolated into the path at request time.
- `GeminiRole { User, Model }` exposes the API-native role names. We considered `User, Assistant` for cross-provider consistency and rejected it: the user constraint is "do not genericify providers," and exposing the wire-native name avoids a translation that would surprise anyone reading Google's docs.
- `system_prompt` serializes as a top-level `systemInstruction: { parts: [{ text }] }` object — different from both OpenAI (system-role message in the messages array) and Claude (top-level `system` string).
- `max_tokens` is `Option<u32>` (not required, unlike Claude). The wire field is `maxOutputTokens` inside a nested `generationConfig` object; the agnostic command shape uses `max_tokens` for consistency with OpenAI / Claude.
- Gemini's `parts` can carry `text`, `inlineData`, `functionCall`, etc. The wire type uses `text: Option<String>` (defaulting via `#[serde(default)]`) and we filter for `Some`. Non-text parts deserialize successfully but contribute nothing to the agnostic text output.

### M1 stub-mode tests retired

With Gemini real, no provider stubs remain. The original M1 stub-mode contract test (`multi_infer_returns_one_attempt_per_stub_input_with_failures_preserved`) was deleted — it had nothing to test against. The duplicate-same-provider input-correlation test (`multi_infer_preserves_input_index_with_duplicate_same_provider_inputs`) was rewritten to use three OpenAI inputs against a single mockito mock with `.expect(3)`, preserving the original contract validation (each duplicate input keeps a distinct `input_index`) without depending on stubs.

`AgnosticCompletionError::ProviderStub` is now unreachable from `multi_infer`. The variant stays in the enum until the extraction-checkpoint slice retires it cleanly along with the shared-helper extraction.

## Result

The hypothesis held. Across three concrete providers, the extraction candidates are crisp:

- `parse_retry_after` — trivial three-line helper. Lift to a shared module.
- `map_failure_from_status` — identical skeleton. Lift via a closure-based or trait-based generic helper.
- `WireErrorBody` — three providers share `{"error":{"message"}}`. Lift cleanly.
- `*CompletionFailure` variant set — identical across providers. Either keep per-provider (current shape) and extract a shared helper for the conversion, or unify under a single shared `ProviderFailure` enum and have provider-specific `*Failure` types only when they need extra fields. Unifying is simpler; the per-provider Failure type was useful as a per-provider boundary marker but it currently carries no provider-specific fields. Decision deferred to the extraction slice.

The trio of "per-provider Failure enum mirrors agnostic categories, then converts via a 1-arm-per-variant `*_failure_to_agnostic` helper" is verbose but is what the contract demands today. The extraction slice can either keep it and lift the conversion, or collapse to one shared `ProviderFailure` enum + provider-tagging at the agnostic-conversion boundary.

## Decision

Land the Gemini seed as its own commit with no extraction. Defer the extraction work to a follow-up slice ("M2d — Extraction checkpoint" or similar) so the diff for that slice is a pure refactor with no behavior changes. Retire `ProviderStub` in that same slice — it's now unreachable.

## Next

- **Extraction slice.** Land `parse_retry_after`, `map_failure_from_status`, and `WireErrorBody` as shared helpers. Decide on per-provider `*CompletionFailure` vs single shared `ProviderFailure`. Retire `AgnosticCompletionError::ProviderStub`. No behavior change.
- **M3 — Embeddings + diversification.** Per the existing plan: `Embedder` trait, `OpenAiEmbedder` impl, `select_diverse` with `Centroid` / `FarthestPointSampling` strategies. OpenAI HTTP client reused.
- **Audit 529 mapping in Claude.** Anthropic's `529 overloaded` status currently falls into the 400-range InvalidRequest arm in our generic status mapping; the test accepts either `InvalidRequest` or `ServerError`. Tighten in the extraction pass if we decide the mapping is wrong.

## See Also

- [Implementation Plan — M2](../plans/2026-05-15-implementation-plan.md)
- [2026-05-15 — OpenAI Seed Shape For M2](./2026-05-15-openai-seed-shape.md)
- [2026-05-15 — Claude Seed And Typed Model Contract](./2026-05-15-claude-seed-and-typed-models.md)
- [Decision 0001: Owned Types At Async Boundaries](../decisions/0001-owned-types-at-async-boundaries.md)
- [Journal Index](./README.md)
