[README](../../README.md) | [Lab](../README.md) | [Plans](./README.md) | [Journal](../journal/README.md) | [Decisions](../decisions/README.md)

# Consortium — Implementation Plan

Status: active
Date: 2026-05-15

## Context

`consortium` is the early scaffolding of a Rust crate whose **primary** product is *evergreen, best-of-best `prompt → completion` datasets*. The pipeline:

1. Embed user-supplied prompts and select a maximally diverse subset (latent-space fencing) so the dataset covers the input distribution without redundancy.
2. For each selected prompt, generate completions across many LLM providers (multi-model fan-out).
3. Have multiple LLM judges source-blind rank the candidate completions; aggregate the rankings.
4. Emit the final `prompt → best-completion` corpus.

Secondary use case: the same machinery doubles as a "consortium" inference SDK for runtime calls (slower, costlier, higher quality). The crate also incidentally yields a normalized set of provider clients.

Provider modules: Claude / OpenAI / Gemini are real (M2a/b/c landed); Deepseek / Kimi / Qwen / Llama remain stubs commented out of `multi_infer` until M7. M0 added the dependency baseline (`tokio` full, `futures`, `reqwest` rustls-tls + json, `serde` derive, `serde_json`, `thiserror`, `bytes`; dev-deps `tokio-test`, `mockito`) and moved agnostic output types to owned payloads. M1 landed the async fan-out: `multi_infer` is `pub async fn(&MultiAiCompletionInputs<'_>) -> Vec<ProviderAttempt>` with typed `AgnosticCompletionError`, `ProviderKind`, and a local retry helper; each `ProviderAttempt` carries `input_index` so callers can correlate results back to specific input slots even when the same provider repeats. The public construction path is `MultiAiCompletionInputs::new(&[AiCompletionInputs::…])`, with active provider client / command / model / message / role types re-exported at the crate root. M2 landed real clients for all three target providers: each provider's `*Client` (with `new` / `from_env` / `with_base_url`), typed `*Model` enum, `*CompletionCommand`, mockito-tested wire path, and per-provider failure mapping into `AgnosticCompletionError`. The post-M2c extraction-checkpoint pass lifted `parse_retry_after`, the shared `WireErrorBody`, and the 401/403/429/4xx/5xx status-to-failure mapping into `src/ai_client_apis/shared.rs` via a small `FailureFromStatus` trait that each provider impls in four trivial methods. `AgnosticCompletionError::ProviderStub` was retired in that same commit. A subsequent stabilization pass swapped `api_key: String` for `secrecy::SecretString` across all three clients (Debug output now redacts automatically), replaced the bespoke exponential-backoff-with-splitmix-jitter retry helper with `backon`-driven backoff (`run_with_retry` / `RetryOutcome` / `delay_for_attempt` / `jitter_within` / `JITTER_COUNTER` all gone), and reshaped constructors to `new` / `new_with_base_url` / `from_env` / `from_env_with_base_url` so tests no longer rely on the awkward "allocate default base URL then overwrite" pattern. The boundary the next phase picks up at is M3 (embeddings + diversification).

**Decisions locked in via Q&A + plan review:**
- Initial provider depth: Claude, OpenAI, Gemini only. Deepseek/Kimi/Qwen/Llama stay as stubs until M7.
- Non-streaming completions first; streaming becomes an additive future capability.
- Embedding source is **pluggable from day one** via an `Embedder` trait (with at least one shipped impl).
- Diversification selection algorithm is configurable via a `SelectionStrategy` enum: `Centroid` (default, matches README) and `FarthestPointSampling`.
- Cross-provider boundary types are **owned**, not borrowed. The current lifetime-heavy `AgnosticCompletionOutput` scaffolding is allowed to change early; real output payloads should use `String` and `bytes::Bytes`/`Vec<u8>` so they survive async/network boundaries cleanly.
- Failure handling is explicit at three levels: provider-attempt failures are preserved in fan-out results, prompt-level failures are preserved in dataset builds, and only configuration / unrecoverable orchestration failures bubble as top-level `Result` errors. No silent dropping of failed attempts.
- The crate becomes **async-native** in M1. Public orchestration APIs become `async fn`s and require a caller-managed runtime; the library should not try to hide or own a Tokio runtime internally.
- M8 genericification is **generic-first**. Introduce `Provider` / `Embedder` traits for static dispatch first; add boxed/dynamic wrappers only later if runtime heterogeneity proves necessary.

## Phased Plan

### M0 — Foundation
- Keep `README.md` aligned with current project direction while preserving the lightweight navigation pattern used across repo docs.
- Keep `CONTRIBUTING.md` and the `lab/` indexes current as implementation decisions, experiments, and workflow expectations evolve.
- Add `LICENSE-MIT` and `LICENSE-APACHE` text files.
- Update `Cargo.toml` `[dependencies]`: `tokio` (full), `futures`, `reqwest` (rustls-tls, json), `serde` (derive), `serde_json`, `thiserror`, `bytes`. Dev-deps: `tokio-test`, `mockito`.
- Replace the borrowed/lifetime-parameterized agnostic output scaffolding in `src/lib.rs` with owned payload shapes before real provider work starts. The conceptual types stay, but `&str` / `&Vec<u8>` output fields should not survive into M2.

### M1 — Async Runtime + Error Types

Status: landed.

- `multi_infer` is `pub async fn` returning `Vec<ProviderAttempt>`. The library does not own a Tokio runtime; callers provide one (`#[tokio::test]`, `#[tokio::main]`, or a manually-built runtime).
- `ProviderKind { Claude, OpenAi, Gemini }` names the three M2-tier providers. Deepseek/Kimi/Qwen/Llama remain stubs through M7.
- `AgnosticCompletionError` (thiserror), every variant carrying `provider: ProviderKind`. Current contract (after M2 refinements and the post-M2c extraction pass):
  - `Transport { source: reqwest::Error }`
  - `Deserialize { source: serde_json::Error }`
  - `RateLimited { retry_after: Option<Duration>, message: Option<String> }`
  - `Auth { message: Option<String> }`
  - `InvalidRequest { message: String }`
  - `ServerError { status: u16, message: Option<String> }`
  - `MalformedResponse { reason: String }`

  Accessors: `provider()` and `is_transient()`. Transient = Transport / RateLimited / ServerError. (M1 introduced the variant set originally; M2 added `Auth.message`, `RateLimited.message`, and `MalformedResponse`. The M1-era `ProviderStub` placeholder was retired in the extraction-checkpoint commit once all three real providers were live.)
- `ProviderAttempt { provider, result, retries, latency, input_index }` with `result: Result<AgnosticCompletionOutput, AgnosticCompletionError>`. Per-provider failures stay inside `result` — never collapsed to a top-level fan-out error. `input_index` (added post-M1) lets callers correlate completion-order attempts back to original input slots.
- Fan-out is `FuturesUnordered<BoxFuture<'a, ProviderAttempt>>` across Claude/OpenAI/Gemini so the three concrete branch futures share one in-flight queue.
- Retry orchestration is folded directly into `build_attempt` and driven by `backon::ExponentialBuilder` for backoff computation. The hand-rolled `run_with_retry` / `delay_for_attempt` / `JITTER_COUNTER` / splitmix jitter machinery was removed during the stabilization pass after M2c. `MAX_RETRIES = 2` (1 initial + 2 retries) and `BASE_RETRY_DELAY = 100ms` are kept as named constants. The thin loop still special-cases `AgnosticCompletionError::RateLimited::retry_after` via a small `retry_after_override` helper (backon's `Backoff` trait does not expose the error to its delay computation, so this stayed as a hand-rolled branch). The `backoff.next()` iterator drives the retry budget regardless of which branch (override vs exponential) ends up sleeping, so the bounded-attempts contract holds.
- `convert_raw_result_to_agnostic_output` is crate-private and returns `Result<AgnosticCompletionOutput, AgnosticCompletionError>`. M2 filled in real per-provider Ok arms (text chunks + token counts); the seed-era empty-vec stubs are gone.
- **Verified during M1 (pre-extraction tests since superseded by M2 fan-out tests):** an `#[tokio::test]` constructed three stub inputs and asserted each produced exactly one `ProviderAttempt` with the matching provider id and a typed error. After M2c retired all stubs, that test was replaced by the M2-era fan-out success / duplicate-index / transient-retry tests in `src/lib.rs#tests`.

### M2 — Three Real Provider Clients

Sequenced as one provider at a time, validating the shape on the first before replicating it. Each provider lives under `src/ai_client_apis/<name>/mod.rs` and follows the same contract.

**Provider-wide conventions (landed during M2a/M2b).** Apply to every provider in M2 and beyond:

- Typed model enum (`OpenAiModel`, `ClaudeModel`, later `GeminiModel`) with `as_api_str`, `Display`, and `serde::Serialize`. `Custom(String)` variant for forward-compat. Public command structs must not expose raw model strings.
- Per-provider `*CompletionFailure` enum with: `Transport(reqwest::Error)`, `Deserialize(serde_json::Error)`, `Auth { message: Option<String> }`, `RateLimited { retry_after: Option<Duration>, message: Option<String> }`, `InvalidRequest { message: String }`, `ServerError { status: u16, message: Option<String> }`, `MalformedResponse { reason: String }`.
- 200-OK responses with missing/empty content blocks must surface as `MalformedResponse`, never empty-string success.
- Agnostic `AgnosticCompletionError` carries the same shape including `Auth.message`, `RateLimited.message`, and a top-level `MalformedResponse { provider, reason }` variant.

**M2a — OpenAI (seed): landed.** See [2026-05-15-openai-seed-shape](../journal/2026-05-15-openai-seed-shape.md) for the original shape and [2026-05-15-claude-seed-and-typed-models](../journal/2026-05-15-claude-seed-and-typed-models.md) for the contract refinements (typed model, MalformedResponse, message-context preservation) that landed with the Claude seed.

- `OpenAiClient` (`new` / `from_env` reading `OPENAI_API_KEY` / `with_base_url` builder).
- `OpenAiCompletionCommand { model: OpenAiModel, system_prompt: Option<String>, messages, max_tokens: Option<u32>, temperature: Option<f32> }` with `Default`.
- `OpenAiCompletionFailure` typed enum (full variant list above). Converted to `AgnosticCompletionError` via `openai_failure_to_agnostic` in `lib.rs`.
- `convert_raw_result_to_agnostic_output` OpenAI arm populates a text chunk and real token counts.
- Tests: success / 401 / 403 / 429-with-`Retry-After` / 400 / 503 / malformed-JSON / empty-choices-`MalformedResponse` / model-serialization / `multi_infer` fan-out success / `multi_infer` 503-driven retry verifying `ProviderAttempt.retries == 2`. Plus 1 `#[ignore]` live test on `OPENAI_API_KEY`.

`OpenAiModel` was subsequently narrowed (commit `b15a099`) to `{ Gpt4oMini, Gpt4o, Gpt4Turbo, Custom(String) }`. The o-series variants (`O1`, `O1Mini`, `O3Mini`) were dropped because OpenAI reasoning models require `max_completion_tokens` (not `max_tokens`) and `developer` role (not `system`); the current chat-completions wire path does not emit that shape. Adding o-series support is its own future slice with a branched request builder. `Custom` is documented as a chat-completions-shape-only escape hatch.

**M2b — Claude: landed.**

- `ClaudeClient` (`new` / `from_env` reading `ANTHROPIC_API_KEY` / `with_base_url`). Auth via `x-api-key` header and required `anthropic-version: 2023-06-01`.
- `ClaudeCompletionCommand { model: ClaudeModel, system_prompt: Option<String>, messages: Vec<ClaudeMessage>, max_tokens: u32, temperature: Option<f32> }` with `Default`. `max_tokens` is **required** at the type level because Anthropic's API rejects requests without it.
- `ClaudeModel { Opus47, Sonnet46, Haiku45, Custom(String) }`.
- Endpoint: `POST /v1/messages`. Response content is an array of typed content blocks; the seed parses and concatenates `text` blocks and surfaces `MalformedResponse` if no text is produced. Other block types (e.g., `tool_use`) are deserialized to a catch-all `Unknown` variant and dropped — handling them is M4+ work.
- `ClaudeCompletionFailure` typed enum (full variant list above). Converted via `claude_failure_to_agnostic`.
- Tests: success / multi-text-block concatenation / 401 / 429-with-`Retry-After` / 400 / 503 / 529 (Anthropic's "overloaded") / malformed-JSON / empty-content-`MalformedResponse` / model-serialization / `multi_infer` fan-out success. Plus 1 `#[ignore]` live test on `ANTHROPIC_API_KEY`.

**M2c — Gemini: landed.**

- `GeminiClient` (`new` / `from_env` reading `GEMINI_API_KEY` / `with_base_url`). Auth via `x-goog-api-key` header (the URL `?key=` query form is also supported by the API but the header path is cleaner for mocking and avoids the key surfacing in logs / URLs).
- `GeminiCompletionCommand { model: GeminiModel, system_prompt: Option<String>, messages: Vec<GeminiMessage>, max_tokens: Option<u32>, temperature: Option<f32> }` with `Default`. `max_tokens` stays `Option<u32>` because Google's API does not require it.
- `GeminiModel { Gemini20Flash, Gemini15Pro, Gemini15Flash, Custom(String) }`.
- `GeminiRole { User, Model }` — Gemini's API uses `model` rather than `assistant` for prior-turn replies, so the role enum exposes the API-native name.
- Endpoint: `POST {base_url}/v1beta/models/{model}:generateContent`. The model id is part of the URL path, not the request body. The request body uses `contents` (Gemini's name for messages), a top-level `systemInstruction` field for the system prompt, and a nested `generationConfig` object for `maxOutputTokens` / `temperature`.
- Response: `candidates[].content.parts[]` where each part may carry `text`. The seed concatenates all text parts and surfaces `MalformedResponse` if candidates is empty or no text parts are present. Non-text parts (e.g., `inlineData`, `functionCall`) deserialize but their text field is `None` and they're filtered out.
- `GeminiCompletionFailure` typed enum (same variant list as OpenAI / Claude). Converted via `gemini_failure_to_agnostic`.
- Tests: success / multi-text-part concatenation / 401 / 429-with-`Retry-After` / 400 / 503 / malformed-JSON / empty-candidates-`MalformedResponse` / empty-text-parts-`MalformedResponse` / model-serialization / `multi_infer` fan-out success. Plus 1 `#[ignore]` live test on `GEMINI_API_KEY`.

**Extraction checkpoint: landed.** Shared utilities moved into `src/ai_client_apis/shared.rs` (`pub(crate)`):

- `parse_retry_after(&HeaderMap) -> Option<Duration>` — one definition, used by all three providers.
- `WireErrorBody { error: { message } }` — one definition, plus `parse_error_message(&Bytes) -> Option<String>` wrapping the parse.
- `FailureFromStatus` trait with four constructor methods (`auth`, `rate_limited`, `invalid_request`, `server_error`) and `map_status_to_failure<F: FailureFromStatus>(status, headers, bytes) -> F` doing the 401/403 → Auth, 429 → RateLimited (with `parse_retry_after`), 400..=499 → InvalidRequest, 500..=599 → ServerError, fallback → ServerError mapping. Each provider impls the trait in four trivial methods and the call site becomes a single `super::shared::map_status_to_failure(...)` line.
- `AgnosticCompletionError::ProviderStub` retired. `provider()` and `is_transient()` exhaustive matches updated.

Per-provider `*CompletionFailure` enums were kept as-is — the extraction was a pure refactor with no semantic redesign. Collapsing to a shared `ProviderFailure` enum was considered and deferred since it'd be a behavior-shape change, not a refactor.

Provider behavior, public API surface, and wire shape did not change. Test matrix stayed green at 34 passed + 3 ignored.

- **Verify per provider:** `mockito` parsing tests covering success, each typed failure variant, and the `MalformedResponse` empty-content edge; `#[ignore]`-gated integration tests that hit real APIs when keys are present (`cargo test -- --ignored`).

### M3 — Embedding + Diversification

**Direction change (recorded 2026-05-15):** the original M3 framing was "one `Embedder` trait with an `OpenAiEmbedder` reusing the OpenAI HTTP client." That single-provider seed is too forgiving — embedding abstractions calcify around the first provider's shape fast. M3 is reframed to land Cohere and OpenAI as first-class embedders together, so the agnostic boundary has to survive two providers with different model families, dimensions, and request shapes from day one. See [2026-05-15 — M3 Multi-Provider Embedding Direction](../journal/2026-05-15-m3-multi-provider-embedding.md).

**M3a — Embedding abstraction + Cohere seed + OpenAI seed + selection core.**

New modules:

- `src/embeddings/mod.rs` — agnostic boundary:
  - `pub trait Embedder { async fn embed(&self, inputs: &[String]) -> Result<EmbeddingBatch, AgnosticEmbeddingError>; }` (native async fn in trait; static dispatch only).
  - `pub enum EmbeddingProvider { Cohere, OpenAi }`.
  - `pub enum AgnosticEmbeddingError` (thiserror) — same variant set as `AgnosticCompletionError`: `Transport`, `Deserialize`, `Auth { message }`, `RateLimited { retry_after, message }`, `InvalidRequest { message }`, `ServerError { status, message }`, `MalformedResponse { reason }`. All variants carry `provider: EmbeddingProvider`.
  - `pub struct EmbeddingBatch { pub vectors: Vec<Vec<f32>>, pub usage: EmbeddingUsage }`.
  - `pub struct EmbeddingUsage { pub input_tokens: u64 }` (room to add `model_used`, `request_id`, etc. without breaking `EmbeddingBatch`).
  - Crate-private `cohere_embedding_failure_to_agnostic` and `openai_embedding_failure_to_agnostic` mappers (mirrors the completions pattern).
- `src/ai_client_apis/cohere/mod.rs` + `embeddings.rs` — `CohereEmbedder` with `new` / `new_with_base_url` / `from_env` (reads `COHERE_API_KEY`) / `from_env_with_base_url`, builder methods `with_model` / `with_input_type`. Typed `CohereEmbeddingModel { EmbedEnglishV3, EmbedMultilingualV3, EmbedEnglishLightV3, Custom(String) }` with `as_api_str` / `Display` / `Serialize`. Typed `CohereEmbeddingInputType { SearchDocument, SearchQuery, Classification, Clustering }` with `#[serde(rename_all = "snake_case")] Serialize`. `POST /v2/embed` with `Authorization: Bearer`. Request body `{model, input_type, texts, embedding_types: ["float"]}`. Response `{embeddings: {float: [[...]]}, meta: {billed_units: {input_tokens}}}`. `CohereEmbeddingFailure` typed enum impls `FailureFromStatus` so the shared status mapping in `src/ai_client_apis/shared.rs` is reused.
- `src/ai_client_apis/openai/embeddings.rs` — `OpenAiEmbedder` with the same four-constructor shape and `with_model` builder. Typed `OpenAiEmbeddingModel { TextEmbedding3Small, TextEmbedding3Large, TextEmbeddingAda002, Custom(String) }`. `POST /v1/embeddings` reusing the existing OpenAI bearer-auth pattern. Request `{model, input}`. Response `{data: [{embedding}], usage: {prompt_tokens}}`. `OpenAiEmbeddingFailure` impls `FailureFromStatus`.
- `src/diversification/mod.rs` — provider-agnostic selection over `&[Vec<f32>]`:
  - `pub enum SelectionStrategy { Centroid, FarthestPointSampling }`.
  - `pub struct StopCondition { pub max_n: Option<usize>, pub similarity_tripwire: Option<f32> }`.
  - `pub fn select_diverse(embeddings: &[Vec<f32>], strategy: SelectionStrategy, stop: StopCondition) -> Vec<usize>` returning indices into the input. Centroid: running-mean-then-pick-least-similar-to-mean. FPS: pick-farthest-from-nearest-already-selected.
  - Private `cosine_similarity` helper.

Defaults:

- Cohere: `embed-english-v3.0` + `search_document` input type. **Not** `embed-v4.0` (multimodal) — text embeddings only in M3a per direction change.
- OpenAI: `text-embedding-3-small`.

Batching: each `embed` call makes one HTTP request. Callers must chunk before the per-provider per-request input limit (OpenAI: 2048, Cohere: 96 for v3). Auto-chunking is a future M3b candidate.

Tests:

- Provider-local `mockito` tests per embedder: success / auth / rate-limit-with-Retry-After / invalid-request / server-error / malformed-JSON / empty-embeddings-`MalformedResponse` / model-serialization. Plus one `#[ignore]` live test per provider gated on the respective env var.
- Provider-agnostic synthetic-2D selection tests: hand-built clusters at corners of a square; both `Centroid` and `FarthestPointSampling` strategies pick one representative per cluster when `max_n` matches the cluster count. Plus `max_n` and `similarity_tripwire` each tested in isolation.

**M3b (deferred, not in this slice).** Auto-chunking inside `Embedder::embed` with backpressure. Multimodal embedding support (Cohere `embed-v4.0` / OpenAI image-embedding endpoints if they emerge). Additional providers (Voyage, Mixedbread, etc.) as motivated. Embedding-side retry helper analogous to the completions path if real-world rate limits make it necessary.

### M4 — Judge Phase

**Direction corrections (recorded 2026-05-15 before coding):** the original M4 outline had a real architectural bug — `HashMap<BlindId, ProviderTag>` collapses candidates to provider identity, which doesn't survive M5 phase-1 where multiple candidates from the same provider get judged together. The plan is corrected to map blind ids to **candidate identity** (specifically: candidate index in the original slice), and to make the parse contract typed and strict from day one. See [2026-05-15 — M4 Judge Layer Corrections](../journal/2026-05-15-m4-judge-layer.md).

**M4a — Judge primitives (landed):** new module `src/judge/`.

- `Candidate { content, provider: ProviderKind, model: String }` — what callers assemble after fan-out. Provider/model stay with the candidate but are never sent to the judge.
- `BlindId(String)` — opaque, neutral, displays as `c1`, `c2`, etc.
- `BlindCandidate { id, content }` — what the judge actually sees.
- `assign_blind_ids(&[Candidate]) -> (Vec<BlindCandidate>, HashMap<BlindId, usize>)`. The `usize` is the candidate's index in the original slice, so the caller recovers full provenance — including which of two same-provider candidates was which — without leaking anything to the judge.
- `JUDGE_SYSTEM_PROMPT` (`pub const`) locks the response shape: `<reasoning>...</reasoning>` then `<ranking>id1,id2,...</ranking>`. No ties. Every candidate exactly once. No text outside the blocks. Explicitly tells the judge it does not know the source model/provider. A test asserts the prompt does not mention any known provider name.
- `build_judge_user_message(&[BlindCandidate]) -> String` formats candidates using their blind ids only.
- `parse_ordered_judgement(raw, &HashSet<BlindId>) -> Result<OrderedJudgement, JudgementParseError>`. Tolerates whitespace around ids and inside the reasoning block. Strict on semantics: rejects missing tags, empty ranking, unknown ids, duplicate ids, and missing-from-expected ids. `OrderedJudgement` carries `ordered_ids`, `reasoning`, and `raw_response` (for audit).
- `judge_rank(candidates, invoke_judge)` — `invoke_judge: FnOnce(JudgeRequest) -> Future<Output = Result<String, AgnosticCompletionError>>`. The closure-injected provider call keeps the judge layer provider-agnostic; the caller wires whichever real provider (or mock) as the judge. Provider errors propagate as `JudgementError::Provider`; parse failures as `JudgementError::Parse`.
- `aggregate_rankings(&[OrderedJudgement]) -> AggregatedRanking { ordered_ids, scores }`. Borda count: i-th-place candidate in a length-N ranking earns `N - i` points; summed across judges; ordered by total desc. Tie-break: lexicographic on `BlindId` (deterministic; documented as `c10 < c2`).

Removed from `src/lib.rs` as part of this slice: the M0-era `ORDERED_JUDGEMENT_SYSTEM_PROMPT` placeholder, `OrderedJudgementStructuredData`, `SortableJudgementProvider`, `AiCompletionCommand`, and `make_sortable_judgement_command`. All superseded by `src/judge/`.

- **Verified:** 19 unit tests covering blind-id assignment + provenance, prompt + user-message provider-neutrality, parse success / whitespace tolerance / each typed parse error, `judge_rank` happy path + provider-error + parse-error propagation, Borda aggregation single / unanimous / disagreeing / tied / empty.

### M5 — Two-Phase Consortium

The empty `consortium_completion` (`src/lib.rs:157`) becomes the orchestrator:

- **Phase 1 (intra-model):** for each model, sample N completions; run multi-judge ranking within that model; keep the winner.
- **Phase 2 (inter-model):** feed per-model winners into cross-model multi-judge ranking; emit the overall best.
- Surface intermediate results via an `mpsc` channel or a callback trait (the comment at `src/lib.rs:161` mentions this hook intent) so callers can stream phase-1 winners as they finalize.
- Preserve per-provider and per-judge failures in the orchestration result instead of flattening them away; if a prompt fails end-to-end, that becomes a first-class prompt outcome rather than an omitted row.
- **Verify:** one `--ignored` end-to-end test that exercises both phases against real APIs.

### M6 — Dataset Pipeline + JSONL Writer

- `pub struct DatasetBuilder` with config (providers, samples-per-model N, judges, embedder, selection strategy, stop condition).
- Validate config eagerly and fail fast on setup issues.
- `build_dataset(prompts)` returns a stream of prompt outcomes, not bare rows, so completed rows and prompt-level failures are both surfaced. Concrete shape can be `Result<impl Stream<Item = PromptOutcome>, DatasetBuildError>` or equivalent builder/start split, but the contract is: fatal setup errors short-circuit, per-prompt failures continue through the stream.
- JSONL writer that flushes per row.
- **Verify:** run end-to-end with ~20 synthetic prompts against the three real providers; inspect the JSONL.

### M7 — Remaining Four Providers

Replicate M2 for Deepseek, Kimi K2, Qwen, Llama. Uncomment the corresponding arms in `AiCompletionInputs`/`RawAiCompletionResult` (`src/lib.rs:13-16`, `:28-31`) and the match arms in `multi_infer` (`src/lib.rs:103-114`).

### M8 — Genericify

- Lift the seven concrete `*Client` / `*CompletionCommand` / `*Result` types behind a generic-first `Provider` trait with associated types. Prefer static dispatch in the first pass; if callers later need runtime-pluggable heterogeneous providers, add boxed adapters as a follow-up instead of forcing trait-object design here.
- Parameterize the judge system prompt.
- Add a second `Embedder` impl (e.g., `fastembed` local) to validate the trait abstraction under a real second implementer.

### M9 — Language Bindings

- New crates in a workspace layout: `bindings/napi/` (TS via napi-rs) and `bindings/pyo3/` (Python).
- Expose a minimal surface: `build_dataset(prompts, config) -> Stream<Row>` and `consortium_complete(prompt, config) -> Completion`.

### M10 — Polish

- Progress events on a dedicated channel (per-prompt status: embedded, sampling, judging, finalized).
- Telemetry / metrics hooks (timings, retry counts, token totals); refine `CompletionOutputTokensUsed` per `src/lib.rs:46` TODO once provider reporting is well-understood.
- `examples/` directory: one dataset-generation example, one runtime-inference example.
- CI: build, `clippy -D warnings`, test, fmt check.
- Perf benchmarks against a fixture prompt set.

## Critical Files To Modify

- `src/lib.rs` — primary orchestrator; current `multi_infer`, `consortium_completion`, judge scaffolding all live here. Will be split into submodules as it grows past ~300 lines.
- `src/ai_client_apis/{claude,openai,gemini}/mod.rs` — filled in M2.
- `src/ai_client_apis/{deepseek,kimik2,qwen,llama}/mod.rs` — filled in M7.
- `Cargo.toml` — dependencies added in M0.
- `README.md`, `CONTRIBUTING.md` — root documentation and contributor contract.
- `lab/README.md`, `lab/plans/`, `lab/journal/`, `lab/decisions/` — working-note system and indexes.
- New: `src/diversification/`, `src/judge/`, `src/dataset/`.
- New: `LICENSE-MIT`, `LICENSE-APACHE`.

## Reuse Opportunities

The current `src/lib.rs` scaffolding is valuable as a concept map, but **not every concrete type shape is frozen**:

- `AgnosticCompletionOutput`, `CompletionOutputChunk`, `CompletionOutputImage`, `CompletionOutputTokensUsed` (`src/lib.rs:35-56`) stay conceptually, but their fields should become owned and their lifetime parameters are expected to disappear.
- `RawAiCompletionResult`, `AiCompletionInputs`, `MultiAiCompletionInputs` (`src/lib.rs:9-32`) are useful early fan-out scaffolding through M2/M7, but may be split or replaced once M8 introduces trait-based provider abstractions.
- `OrderedJudgementStructuredData`, `SortableJudgementProvider`, `AiCompletionCommand`, `make_sortable_judgement_command` (`src/lib.rs:175-205`) capture the right judge-layer concepts and should be filled in, even if they move into `src/judge/`.
- Per-provider stub modules remain good starting points because they already impose a uniform shape; M2 establishes the fill-in pattern, M7 replicates it mechanically.

## Verification

- **After M2:** `cargo test --lib` passes. With keys in env, `cargo test -- --ignored` round-trips all three flagship providers.
- **After M3:** synthetic-2D selection tests pass; visually plot picked indices to confirm latent-space fencing.
- **After M5:** `--ignored` consortium test produces a coherent best-of-best completion for a fixture prompt.
- **After M6:** `cargo run --example build_dataset` with a ~20-prompt fixture produces a JSONL corpus where diverse-prompt subset visibly spans the input set.
- **After M9:** example TS and Python scripts import their respective bindings and produce a row.

## Open Follow-Ups

- The README currently describes the centroid-style selection idea more directly than the `SelectionStrategy` surface the crate will eventually expose. Once FPS and related abstractions settle, update the README without losing its concise top-level framing.
- `CompletionOutputTokensUsed` breakdown (`src/lib.rs:46` TODO): refine to capture reasoning/cache/system splits once we know what each provider actually reports.

## See Also

- [Plans Index](./README.md)
- [Initial Repo Read And Planning Journal](../journal/2026-05-15-initial-repo-read-and-planning.md)
- [Decision 0001: Owned Types At Async Boundaries](../decisions/0001-owned-types-at-async-boundaries.md)
