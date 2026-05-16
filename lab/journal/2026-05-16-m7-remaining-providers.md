[README](../../README.md) | [Lab](../README.md) | [Journal](./README.md) | [Plans](../plans/README.md) | [Decisions](../decisions/README.md)

# 2026-05-16 — M7 Remaining Provider Clients

Status: validated
Date: 2026-05-16

## Question

[M1](../plans/2026-05-15-implementation-plan.md#m1--async-runtime--error-types) reserved seven `ProviderKind` slots; [M2a/b/c](../plans/2026-05-15-implementation-plan.md#m2--three-real-provider-clients) filled three of them (OpenAI / Claude / Gemini) and locked in the per-provider client shape and the [`FailureFromStatus`](../../src/ai_client_apis/shared.rs) extraction. The remaining four — Deepseek, Kimi K2, Qwen, Llama — sat as commented-out arms in [`AiCompletionInputs`](../../src/lib.rs) / [`RawAiCompletionResult`](../../src/lib.rs) / [`multi_infer`](../../src/lib.rs) and as eight-line stub structs returning `Err(*CompletionFailure {})` in their respective `src/ai_client_apis/<name>/mod.rs` files. With the post-M6c provider-side [embedder mixed-dimension hardening](./2026-05-16-embedder-mixed-dimension-hardening.md) slice landed, the path was clear to backfill the four completion providers without touching the embedding / dataset / orchestrator surfaces.

The constraint was to **replicate the M2 pattern verbatim** — not to broaden into [M8](../plans/2026-05-15-implementation-plan.md#m8--genericify) generification. The four data points from M2 (three real providers + the extraction checkpoint) already proved the shape; M7 is mechanical fan-out, not a redesign.

## Hypothesis

All four targets ship OpenAI-compatible chat-completions surfaces (bearer auth, `POST /v1/chat/completions`, `{choices: [{message: {content}}], usage: {prompt_tokens, completion_tokens}}`), so each new module would be a near-byte-for-byte mirror of [`src/ai_client_apis/openai/mod.rs`](../../src/ai_client_apis/openai/mod.rs) with the URL, env-var name, and `*Model` enum swapped out. Wiring into `multi_infer` would compose to one `ProviderKind` variant + one `AiCompletionInputs` variant + one `RawAiCompletionResult` variant + one `convert_raw_result_to_agnostic_output` match arm + one fan-out match arm + one `*_failure_to_agnostic` helper per provider, all parallel to the OpenAI shape.

## What We Tried

Four new modules, identical structure per provider. Each module ships:

- `*Client { http: reqwest::Client, base_url: String, api_key: SecretString }` with the canonical four constructors — `new` / `new_with_base_url` / `from_env` / `from_env_with_base_url` (matching the [stabilization-pass shape](./2026-05-15-stabilization-secrets-and-backon.md#constructor-shape)).
- Typed `*Model` enum implementing `as_api_str(&self) -> &str` / `Display` / `Serialize` (via hand-written `serializer.serialize_str(self.as_api_str())`, not `#[derive]`, so `Custom(String)` carries through to the wire id unchanged).
- `*Role { User, Assistant }` with a private `as_wire_str(self) -> &'static str` helper (mirrors [`OpenAiRole`](../../src/ai_client_apis/openai/mod.rs) — these providers all need a `"system"` wire role that is *not* exposed in the public role enum, since `system_prompt` is its own command field).
- `*Message`, `*CompletionCommand` with `Default`, `*CompletionSuccess`, `*CompletionFailure` (the M2 seven-variant set — `Transport / Deserialize / Auth / RateLimited / InvalidRequest / ServerError / MalformedResponse`), `*Result` type alias.
- `impl super::shared::FailureFromStatus for *CompletionFailure` so the four 401/403/429/4xx/5xx status arms are one `super::shared::map_status_to_failure(...)` call at the request site.
- `*_get_completion(&*Client, &*CompletionCommand) -> *Result` using `response.bytes().await` + `serde_json::from_slice` (so `Transport` vs `Deserialize` stay distinguishable) and surfacing 200 OK with empty `choices` as `*CompletionFailure::MalformedResponse { reason: "response contained no choices" }`.

Per-provider deltas:

| Provider | Base URL (default) | Env var | Model enum |
| -------- | ------------------ | ------- | ---------- |
| **Deepseek** | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | `{ Chat ("deepseek-chat"), Reasoner ("deepseek-reasoner"), Custom }` |
| **Kimi K2** | `https://api.moonshot.ai` | `MOONSHOT_API_KEY` | `{ KimiK20905Preview, MoonshotV1_8k / _32k / _128k, Custom }` |
| **Qwen** | `https://dashscope-intl.aliyuncs.com/compatible-mode` | `DASHSCOPE_API_KEY` | `{ QwenTurbo, QwenPlus, QwenMax, Qwen3_32B, Qwen3_235BA22B, Custom }` |
| **Llama** | `https://api.llama.com` | `LLAMA_API_KEY` | `{ Llama4Maverick17B, Llama4Scout17B, Llama3_3_70B, Custom }` |

The env-var name choice for Kimi K2 deliberately targets the **vendor** (Moonshot) rather than the **model** (Kimi K2), matching the precedent set by [`OPENAI_API_KEY`](../../src/ai_client_apis/openai/mod.rs) (vendor, not "GPT") and [`ANTHROPIC_API_KEY`](../../src/ai_client_apis/claude/mod.rs) (vendor, not "Claude"). The Qwen default base URL picks the **international** DashScope endpoint (`dashscope-intl.aliyuncs.com`) so the default-constructed client is usable globally; the China endpoint (`dashscope.aliyuncs.com`) is reachable through `new_with_base_url` / `from_env_with_base_url`.

`src/lib.rs` was extended along five seams without restructuring:

- `use crate::ai_client_apis::{claude::*, deepseek::*, gemini::*, kimik2::*, llama::*, openai::*, qwen::*};` and four new `pub use` blocks at the crate root for `*Client / *ClientError / *CompletionCommand / *Message / *Model / *Role` per provider.
- `AiCompletionInputs` and `RawAiCompletionResult` picked up four new variants each (uncommented in the existing slot order `KimiK2 / Deepseek / Qwen / Llama`).
- `ProviderKind` picked up `KimiK2 / Deepseek / Qwen / Llama` variants. `provider()` and `is_transient()` on `AgnosticCompletionError` continue to be exhaustive via the `..` capture patterns, so no further match arms needed updating there. `AiCompletionInputs::provider` was extended with four matching arms.
- Four `*_failure_to_agnostic` helpers were added next to the existing three. Every arm is mechanical — variant for variant, same shape as the M2 trio.
- `convert_raw_result_to_agnostic_output` and `multi_infer` each grew four match arms.

`src/dataset/mod.rs::provider_str` (`src/dataset/mod.rs:804`) was the only out-of-`src/lib.rs` exhaustive-match site flagged by the compiler — it gained four arms (`kimik2 / deepseek / qwen / llama`).

## Result

`cargo test --lib` is **171 passed / 0 failed / 10 ignored** (+44 over the post-hardening 127 / 6).

The +44 breaks down as:

- 10 per-provider mockito tests × 4 providers = **40 module-local tests** covering `success` / `auth_failure_on_401` / `auth_failure_on_403` / `rate_limit_carries_retry_after` / `invalid_request_on_400` / `server_error_on_503` / `malformed_json_maps_to_deserialize` / `empty_choices_maps_to_malformed_response` / `*_model_serializes_to_expected_wire_value` / `debug_output_redacts_api_key`. The `empty_choices_maps_to_malformed_response` test is the per-provider malformed-response invariant — these providers share OpenAI's `choices: []` shape, so the invariant is the same as OpenAI's.
- **4 new `multi_infer` fan-out tests** in `src/lib.rs#tests` (`multi_infer_deepseek_success_path_emits_real_text_and_tokens` and the three siblings), each asserting `provider`, `input_index`, the text chunk content, and the input/output token counts. These match the existing M2 fan-out coverage that proves each provider arm is wired correctly into the agnostic boundary.

The +4 ignored = one `#[ignore]`-gated live test per new provider (`live_deepseek_completion_returns_real_response` etc.), each gated on its own env var with a `#[ignore = "requires *_API_KEY; run with cargo test -- --ignored"]` message identical to the M2 pattern.

`cargo clippy --lib --all-features` drops from **33 warnings** to **1 warning**. The 32 `is never used` warnings on the four stub modules' types and functions are gone. The one remaining warning is the pre-existing `collapsible_if` in [`src/diversification/mod.rs:93`](../../src/diversification/mod.rs) — unchanged, deliberately out of scope.

## Verified Properties

- **Public surface conventions held.** All four new provider clients ship the canonical four constructors. All four `*Model` enums implement `as_api_str` / `Display` / `Serialize` with a `Custom(String)` escape hatch. All four `*Role` enums are `Copy`. All four `*CompletionCommand`s have `Default`.
- **Secret handling preserved.** Every new client stores its key as `SecretString` and exposes it exactly once (`bearer_auth(client.api_key.expose_secret())` in the single request builder per provider). `debug_output_redacts_api_key` per provider asserts `format!("{client:?}")` does not contain the supplied key string, guarding the auto-redacting `Debug` derive against future regressions (e.g., a hand-written `Debug` impl that fields-out the secret).
- **Typed provider-specific failures land at the typed agnostic boundary.** Every `*CompletionFailure` variant maps one-for-one through `*_failure_to_agnostic` into the matching `AgnosticCompletionError` variant, tagged with the right `provider: ProviderKind::*`. Non-transient failures (`Auth`, `InvalidRequest`, `Deserialize`, `MalformedResponse`) stay non-transient through `is_transient()`; `Transport`, `RateLimited`, `ServerError` stay transient. The retry budget in `build_attempt` honors `RateLimited::retry_after` via the existing `retry_after_override` helper.
- **Shared status-mapping helper reused.** Each new `*CompletionFailure` impls `FailureFromStatus` in four trivial methods, and the request site is one `super::shared::map_status_to_failure(status.as_u16(), &headers, &bytes)` call — no per-provider 4xx/5xx mapping was reintroduced.
- **Wire-protocol distinguishability preserved.** Each new `*_get_completion` uses `response.bytes().await` then `serde_json::from_slice` (not `response.json().await`), so a `reqwest::Error` from a dropped connection stays `Transport(reqwest::Error)` rather than collapsing into a Deserialize-shaped path.
- **Empty-content surfaces as `MalformedResponse`, not as empty-string success.** Each new provider's `*_get_completion` returns `*CompletionFailure::MalformedResponse { reason: "response contained no choices" }` for 200 OK with `choices: []`. Per-provider `empty_choices_maps_to_malformed_response` asserts the reason substring `"no choices"`.
- **No embedding / dataset / orchestrator behavior touched.** The only non-`lib.rs` extension was `src/dataset/mod.rs::provider_str`, which is a pure exhaustive-match completion — JSONL emission gains four new `provider:` string values (`kimik2 / deepseek / qwen / llama`) but no other change.

## Decision

Lock the M7 slice as shipped. The four new providers fill the seven `ProviderKind` slots reserved at M1 and clear the 32 `is never used` clippy warnings that the stub files were accumulating. The M2 pattern transferred mechanically; the per-provider `*CompletionFailure` enums kept the per-provider boundary marker even though every one of them now carries the same variant set, consistent with the [extraction-checkpoint decision](./2026-05-15-gemini-seed-and-extraction-checkpoint.md#decision) to defer collapsing them to a single shared `ProviderFailure` enum until [M8](../plans/2026-05-15-implementation-plan.md#m8--genericify).

The four shipped `*Model` enums are intentionally narrow — they list the model ids most likely to be exercised today and lean on `Custom(String)` for everything else, mirroring the [post-M2c narrowing of `OpenAiModel`](../plans/2026-05-15-implementation-plan.md#m2--three-real-provider-clients) (where the o-series variants were retired because the chat-completions wire shape can't emit `max_completion_tokens` / `developer` role). Adding more named variants is cheap when usage data warrants it.

## Next

- **M8.** Generic-first `Provider` / `Embedder` traits with associated types. With seven concrete provider impls (three from M2, four from M7), the data point is sufficient to design the trait without over-fitting to OpenAI's shape — Claude's required `max_tokens: u32`, Gemini's URL-embedded model id, and the Llama / Kimi K2 / Qwen / Deepseek bearer-auth variants all need to compose under the same generic surface.
- **Collapse the seven per-provider `*CompletionFailure` enums to a shared `ProviderFailure`?** Still deferred. Every one of the seven now carries the same variant set with no provider-specific fields. The post-M2c [extraction journal](./2026-05-15-gemini-seed-and-extraction-checkpoint.md#decision) flagged this as either "keep per-provider and lift the conversion" or "collapse to one shared `ProviderFailure` enum"; M8 is the natural moment to revisit.
- **Live-test gating audit.** With four new `#[ignore]`-gated live tests, `cargo test -- --ignored` now needs seven completion env vars + two embedding env vars to fully exercise. Worth a small `lab/decisions/` note documenting the matrix if the live-test set grows further.

## See Also

- [Implementation Plan — M7](../plans/2026-05-15-implementation-plan.md#m7--remaining-four-providers)
- [2026-05-15 — OpenAI Seed Shape For M2](./2026-05-15-openai-seed-shape.md)
- [2026-05-15 — Claude Seed And Typed Model Contract](./2026-05-15-claude-seed-and-typed-models.md)
- [2026-05-15 — Gemini Seed And Extraction Checkpoint](./2026-05-15-gemini-seed-and-extraction-checkpoint.md)
- [2026-05-15 — Stabilization: Secrets, Backon, And Constructor Shape](./2026-05-15-stabilization-secrets-and-backon.md)
- [2026-05-16 — Embedder Mixed-Dimension Hardening](./2026-05-16-embedder-mixed-dimension-hardening.md)
- [Decision 0001: Owned Types At Async Boundaries](../decisions/0001-owned-types-at-async-boundaries.md)
- [Journal Index](./README.md)
