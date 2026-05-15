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

Provider modules: Claude / OpenAI / Gemini are real (M2a/b/c landed); Deepseek / Kimi / Qwen / Llama remain stubs commented out of `multi_infer` until M7. M0 added the dependency baseline (`tokio` full, `futures`, `reqwest` rustls-tls + json, `serde` derive, `serde_json`, `thiserror`, `bytes`; dev-deps `tokio-test`, `mockito`) and moved agnostic output types to owned payloads. M1 landed the async fan-out: `multi_infer` is `pub async fn(&MultiAiCompletionInputs<'_>) -> Vec<ProviderAttempt>` with typed `AgnosticCompletionError`, `ProviderKind`, and a local retry helper; each `ProviderAttempt` carries `input_index` so callers can correlate results back to specific input slots even when the same provider repeats. The public construction path is `MultiAiCompletionInputs::new(&[AiCompletionInputs::…])`, with active provider client / command / model / message / role types re-exported at the crate root. M2 landed real clients for all three target providers: each provider's `*Client` (with `new` / `from_env` / `with_base_url`), typed `*Model` enum, `*CompletionCommand`, mockito-tested wire path, and per-provider failure mapping into `AgnosticCompletionError`. The post-M2c extraction-checkpoint pass lifted `parse_retry_after`, the shared `WireErrorBody`, and the 401/403/429/4xx/5xx status-to-failure mapping into `src/ai_client_apis/shared.rs` via a small `FailureFromStatus` trait that each provider impls in four trivial methods. `AgnosticCompletionError::ProviderStub` was retired in that same commit. The boundary the next phase picks up at is M3 (embeddings + diversification).

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
- `run_with_retry` is local and concrete (not generic over error type). Exponential backoff with bounded jitter, honors `RateLimited::retry_after` when set. Defaults: `DEFAULT_MAX_ATTEMPTS = 3`, `DEFAULT_BASE_DELAY = 100ms`. Expected to be replaced by `backon` once the project standardizes per `CONTRIBUTING.md`.
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

New module `src/diversification/`:

- `trait Embedder { async fn embed(&self, prompts: &[String]) -> Result<Vec<Vec<f32>>, _> }` — pluggable per the locked-in decision.
- One initial impl: `OpenAiEmbedder` (text-embedding-3-small) reusing the OpenAI HTTP client.
- Cosine similarity + running-centroid helpers.
- `enum SelectionStrategy { Centroid, FarthestPointSampling }`.
- `struct StopCondition { max_n: Option<usize>, similarity_tripwire: Option<f32> }`.
- `fn select_diverse(embeddings, strategy, stop) -> Vec<usize>` returning indices into the input set.
- **Verify:** synthetic-2D-embedding unit tests (clusters at the corners of a square + noise) confirm corner-picking for both strategies; tripwire and `max_n` each tested in isolation.

### M4 — Judge Phase

- Write the actual `ORDERED_JUDGEMENT_SYSTEM_PROMPT` (currently WIP at `src/lib.rs:164`): reasoning-first inside `<reasoning>` tags, then `<ranking>id1,id2,…</ranking>`.
- `fn assign_blind_ids(candidates) -> (HashMap<BlindId, ProviderTag>, Vec<BlindCandidate>)` so judges never see source attribution.
- `async fn judge_rank(candidates, judge_provider) -> OrderedJudgementStructuredData` — the struct is already declared at `src/lib.rs:175`. Implement an XML-tag parser for judge output.
- `fn aggregate_rankings(rankings: &[…]) -> Vec<BlindId>` — start with Borda count, leave the door open for Copeland/mean-rank later.
- **Verify:** unit tests parse hand-written judge XML strings and aggregate to known winners.

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
