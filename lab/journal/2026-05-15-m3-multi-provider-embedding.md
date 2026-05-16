[README](../../README.md) | [Lab](../README.md) | [Journal](./README.md) | [Plans](../plans/README.md) | [Decisions](../decisions/README.md)

# 2026-05-15 — M3 Multi-Provider Embedding Direction

Status: validated
Date: 2026-05-15

## Question

The original M3 plan called for one embedding provider (OpenAI) reusing the OpenAI HTTP client. Is that the right shape to ship?

## Hypothesis

A single-provider embedding seed would cause the `Embedder` abstraction to calcify around OpenAI's request / response shape. Embedding APIs diverge in ways that don't surface until you have two providers side by side: input field naming (`input` vs `texts`), required parameters (Cohere's `input_type`), response shape (flat array vs `embeddings.float`), token-usage reporting (`prompt_tokens` vs `billed_units.input_tokens`), and model defaults (1536 dims vs 1024). If we only had OpenAI, the trait would silently adopt OpenAI's idioms — and we'd find out the abstraction was wrong at M3c when a second provider tried to fit through it.

## What We Tried

Reframed M3a from "embedder + diversification, OpenAI first" to "embedder + diversification, Cohere AND OpenAI from day one." Both providers land in the same slice. The abstraction has to hold against both before it ships.

Concrete design constraints this forces:

- **Trait must not assume an `input` field name.** Both providers take `&[String]` agnostically; provider-specific renaming happens inside the wire types.
- **Trait must not assume a single required parameter set.** Cohere requires `input_type` per call; OpenAI does not. Both are configured on the `*Embedder` struct itself (via `with_input_type` builder for Cohere), not passed through the trait method. The trait's only per-call parameter is the inputs.
- **Trait must not assume a single response shape.** Cohere returns `embeddings: {float: [[...]]}` (nested); OpenAI returns `data: [{embedding}]` (per-item objects). Both normalize to `EmbeddingBatch { vectors: Vec<Vec<f32>>, usage }`.
- **Usage tracking must not assume a single token field.** Cohere bills via `meta.billed_units.input_tokens`; OpenAI reports `usage.prompt_tokens`. Both normalize to `EmbeddingUsage { input_tokens }`.
- **Model selection stays provider-specific.** No cross-provider `EmbeddingModel` enum. Each provider has its own typed model enum (`CohereEmbeddingModel`, `OpenAiEmbeddingModel`) with a `Custom(String)` escape hatch and a manual `Serialize` impl that emits the wire string.
- **Error mapping stays typed and provider-aware.** `AgnosticEmbeddingError` mirrors the completion-side variant set (`Transport`, `Deserialize`, `Auth`, `RateLimited`, `InvalidRequest`, `ServerError`, `MalformedResponse`) and tags each variant with `provider: EmbeddingProvider`. The shared `FailureFromStatus` trait in `src/ai_client_apis/shared.rs` is reused across both embedders — same 401/403/429/4xx/5xx mapping, different failure enum.

## Result

The reframing landed in the M3 plan section before any code. The agnostic `Embedder` trait is defined once, both providers land their embedders together, and the diversification module operates on `&[Vec<f32>]` without any provider awareness. Test matrix exercises both wire paths via `mockito` plus the synthetic 2D corner-cluster selection.

What is reusable from the completions path:

- The `*Client` constructor shape (`new` / `new_with_base_url` / `from_env` / `from_env_with_base_url`) carries over directly to `*Embedder`. SecretString for the API key; `with_base_url` overrides for test mockito setup.
- `FailureFromStatus` trait and `map_status_to_failure` helper from `ai_client_apis/shared.rs` — extended to embedding failures with no changes.
- Typed model enum pattern (`as_api_str` + `Display` + manual `Serialize` for `Custom(String)` support) — repeated for `CohereEmbeddingModel` and `OpenAiEmbeddingModel`.
- `response.bytes() + serde_json::from_slice` split (keeps Transport / Deserialize distinguishable) — same on embeddings.

What is new / different:

- The `Embedder` struct itself holds configuration that was on the `*CompletionCommand` for chat (model + input_type). The trait's per-call parameter is just `&[String]` inputs. This is cleaner for embeddings because model/input_type rarely change call-to-call within a corpus.
- Native `async fn` in trait (Rust 2024 edition) — no `async_trait` crate. Static-dispatch use only; if dyn-Embedder becomes needed, that's its own slice.
- `CohereEmbeddingInputType` enum (`SearchDocument` / `SearchQuery` / `Classification` / `Clustering`) — Cohere's v3+ API requires this. Defaulting to `SearchDocument` since it's the most general.

## Decision

Two embedders ship together in M3a. No single-provider seed. Diversification logic stays provider-agnostic over `&[Vec<f32>]`. Auto-chunking and multimodal support are M3b-or-later.

## Next

- M3a implementation slice: agnostic `Embedder` trait + `CohereEmbedder` + `OpenAiEmbedder` + `select_diverse` with both strategies + synthetic-2D tests.
- After M3a: M4 (judge phase) per existing plan. M3b items (auto-chunking, multimodal) deferred until concrete callers need them.

## See Also

- [Implementation Plan — M3](../plans/2026-05-15-implementation-plan.md)
- [2026-05-15 — Stabilization: Secrets, Backon, And Constructor Shape](./2026-05-15-stabilization-secrets-and-backon.md)
- [Journal Index](./README.md)
