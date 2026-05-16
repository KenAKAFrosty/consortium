[README](../../README.md) | [Lab](../README.md) | [Journal](./README.md) | [Plans](../plans/README.md) | [Decisions](../decisions/README.md)

# 2026-05-16 — M6c Embedder Auto-Chunking

Status: validated
Date: 2026-05-16

## Question

[M3a](./2026-05-15-m3-multi-provider-embedding.md) shipped the agnostic `Embedder` boundary with one HTTP request per `embed` call. Callers were responsible for hand-sharding inputs that exceeded the provider per-request limits documented in `src/embeddings/mod.rs` (OpenAI: 2048, Cohere v3: 96). [M6a](./2026-05-15-m6a-dataset-builder-and-jsonl-pipeline.md) inherited that constraint — [`DatasetRun::execute`](../../src/dataset/mod.rs) calls `embedder.embed(&prompts)` once over the full prompt batch and would fail at the embedding boundary for any batch above the provider limit. The load-bearing design question for M6c: where does auto-chunking live — at the dataset layer (the only current caller), at the agnostic `Embedder` trait (forces every implementer to handle it), or inside each shipped concrete embedder?

## Hypothesis

Auto-chunking belongs **inside the concrete provider embedders only**. The agnostic boundary expresses a contract: "give me inputs, get back same-order vectors and aggregated usage." How an implementation satisfies that — one HTTP request or many — is an implementation detail it should be free to choose. Folding chunking into the trait forces every implementer to either reimplement it or import a helper; folding it into the dataset layer leaks provider-specific limits into provider-agnostic code. Per-provider implementation is the right boundary.

Lock the M6c slice around four shapes:

1. **Public `Embedder` boundary unchanged.** `Embedder::embed(&[String]) -> Result<EmbeddingBatch, AgnosticEmbeddingError>` keeps its signature. The trait doc updates to note that shipped impls auto-chunk while custom impls remain free to issue a single request.
2. **Per-provider `MAX_INPUTS_PER_REQUEST` constant** kept module-private. Today's documented limits: OpenAI = 2048, Cohere v3 = 96. Hidden from callers — they don't need to size their pipelines around it.
3. **Tiny chunking shell + unchanged chunk helper** in each provider. The existing `*_embed_raw` body becomes `*_embed_chunk`; a new `*_embed_raw` is the chunking shell. Single-chunk fast path calls the chunk helper directly with the original slice — no extra allocation in the common case. Multi-chunk path pre-allocates `Vec::with_capacity(inputs.len())` once.
4. **First-failure short-circuit, no partial-success surface.** Per-chunk errors propagate through the existing `*_failure_to_agnostic` shims. Earlier chunks' vectors are not returned partially. Sequential per-chunk execution because (a) short-circuit on first error wastes no in-flight work and (b) per-prompt parallelism (M6b) is already where the dataset pipeline gets its concurrency — the embedder is called once per `DatasetRun`.

Defer:

- Multimodal Cohere `embed-v4.0` — separate per-request limit, separate code path.
- Embedding-side retry helper analogous to the completions path. Today's `Embedder` impls have no retry; if real-world embedding rate limits force retries, that's its own slice.
- Provider-side hardening to map mixed-dimension responses as `MalformedResponse` earlier (the M6a post-review follow-up). Still touches the wire-level parsers, not the chunking surface.

## What We Tried

Two files changed: [`src/ai_client_apis/openai/embeddings.rs`](../../src/ai_client_apis/openai/embeddings.rs) and [`src/ai_client_apis/cohere/embeddings.rs`](../../src/ai_client_apis/cohere/embeddings.rs). Identical shape per provider.

### Chunking shell

```rust
async fn provider_embed_raw(embedder, inputs) -> Result<EmbeddingBatch, ProviderEmbeddingFailure> {
    if inputs.len() <= MAX_INPUTS_PER_REQUEST {
        return provider_embed_chunk(embedder, inputs).await;
    }

    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
    let mut input_tokens: u64 = 0;
    for chunk in inputs.chunks(MAX_INPUTS_PER_REQUEST) {
        let batch = provider_embed_chunk(embedder, chunk).await?;
        vectors.extend(batch.vectors);
        input_tokens = input_tokens.saturating_add(batch.usage.input_tokens);
    }
    Ok(EmbeddingBatch { vectors, usage: EmbeddingUsage { input_tokens } })
}
```

Three properties this gives for free:

- **Empty inputs.** `inputs.len() == 0` falls into the fast path and calls `provider_embed_chunk` once with the empty slice. The provider rejects it with whatever `InvalidRequest`/`MalformedResponse` it normally would — same behavior as before M6c.
- **Single-chunk allocation.** No extra `Vec` materialization. The chunk helper returns its own `EmbeddingBatch`, which is the function result.
- **Multi-chunk allocation.** Exactly one `Vec::with_capacity(inputs.len())`. `extend` from each chunk's owned `Vec<Vec<f32>>` reuses the inner allocations (no per-vector clone).

`saturating_add` rather than wrapping or panicking: token counts will not realistically overflow `u64`, but silent wrap-around would be worse than a saturated sentinel.

### Chunk helper unchanged

Renamed `*_embed_raw` → `*_embed_chunk` with no body changes. The OpenAI helper's existing index-based slot rebuild, both providers' `MalformedResponse` guards, and the shared `FailureFromStatus` status mapping all stay exactly as they were. Pure rename + extract.

### Trait doc update

`Embedder` trait doc in [`src/embeddings/mod.rs`](../../src/embeddings/mod.rs) updated to:

- Describe what the trait *guarantees*: input-position-preserving vectors, aggregated `EmbeddingUsage`, no partial-success surface on failure.
- Describe what the shipped impls *do*: auto-chunk at documented per-request limits, concatenate in input order, sum usage.
- Make clear that custom impls remain free to issue a single request and reject over-limit inputs themselves.

### Dataset module comment update

[`src/dataset/mod.rs`](../../src/dataset/mod.rs) M6b "What deliberately is not" parenthetical updated to note that auto-chunking is now transparent inside the provider embedders — the dataset layer still issues one logical `embedder.embed(&prompts)` call from its perspective.

## Tests

`cargo test --lib` is 115 passed / 0 failed / 5 ignored (+4 over M6b's 111). All pre-existing embedder tests stay green at the single-chunk fast path.

Four new tests (two per provider) in `src/ai_client_apis/openai/embeddings.rs#tests` and `src/ai_client_apis/cohere/embeddings.rs#tests`:

1. **Over-limit auto-chunking with exact order preservation and aggregated usage.** Per provider, drive `MAX_INPUTS_PER_REQUEST + N` inputs (N = 2 for OpenAI, N = 4 for Cohere) through `embed`. The first input of each expected chunk is a unique marker substring (`provider-auto-chunk-0-marker` / `provider-auto-chunk-1-marker`); two mockito mocks are registered with `Matcher::Regex` body matchers so each chunk's request routes to its own mock without false positives. Each chunk's mock response carries vectors built so that the i-th-globally vector equals `[i as f32, 0.0]`. The aggregated `EmbeddingBatch` is asserted to have `vectors[i] == [i as f32, 0.0]` for every input position — direct evidence chunk concatenation preserved order. Aggregated `usage.input_tokens` is asserted to equal `chunk0_tokens + chunk1_tokens`.

2. **Later-chunk failure surfacing without silent truncation.** Per provider, same chunked input shape, but chunk 1's mock returns a typed error (OpenAI: 401 → `AgnosticEmbeddingError::Auth`; Cohere: 503 → `AgnosticEmbeddingError::ServerError { status: 503, .. }`). The assertion proves `embed` returns the typed error from chunk 1 verbatim, not a partial `EmbeddingBatch` carrying chunk 0's vectors with a missing tail. This is the load-bearing guarantee of "no partial-success surface."

Both per-provider tests use a private `build_*_chunk_response_body(starting_global, count, tokens)` helper to programmatically construct chunk-relative response payloads, so test fixtures stay readable even at OpenAI's 2048-input chunk size.

## Verified Properties

- Public surface unchanged: `Embedder::embed` signature is identical to M3a; the agnostic boundary still expresses one logical call per `embed`.
- Single-chunk behavior unchanged: existing per-provider embedder tests (success / each typed failure / `MalformedResponse` edges) stay green at default `inputs.len() < MAX_INPUTS_PER_REQUEST`.
- Output order preserved: per-provider over-limit tests assert `vectors[i] == [i as f32, 0.0]` across the chunk boundary.
- Aggregated usage: per-provider over-limit tests assert summed `input_tokens` across chunks.
- No partial-success surface: per-provider later-chunk-failure tests assert the typed error surfaces and no partial `EmbeddingBatch` is returned.
- Allocation discipline: single-chunk path allocates nothing extra; multi-chunk path pre-allocates one `Vec::with_capacity(inputs.len())` and extends from owned per-chunk vectors (no inner-vector clones).
- No new clippy warnings introduced: pre-existing 33 warnings (M7 stubs + one `collapsible_if` in `diversification`) unchanged.

## Decision

Auto-chunking lives inside the concrete provider embedders only. The agnostic `Embedder` boundary stays a "one logical call" contract; the trait doc documents what shipped impls do without forcing custom impls to follow suit. Per-provider `MAX_INPUTS_PER_REQUEST` constants stay module-private — callers do not need to know.

## What's Next

- **M5b.** Parallelise judge invocation across slots and across judges within a single prompt's phase — currently each prompt's judges still run sequentially in M5a. Doable without a `Send` refactor by keeping the per-prompt orchestrator state on a single task and using an in-task `FuturesUnordered`.
- **M7.** Replicate the M2 provider pattern for Deepseek / Kimi K2 / Qwen / Llama. Mechanical breadth expansion; does not unblock any currently-failing path.
- **Provider-side mixed-dimension hardening** (M6a post-review follow-up). Map malformed mixed-dimension provider responses to `AgnosticEmbeddingError::MalformedResponse` earlier inside each `*_embed_chunk` so the dataset layer's `EmbeddingDimensionMismatch` guard becomes a defensive backstop rather than a primary path.
- **Embedding-side retry primitive** if real-world embedding rate limits ever force one. Not currently a load-bearing gap.

## See Also

- [M6b Bounded Per-Prompt Parallelism](./2026-05-16-m6b-bounded-per-prompt-parallelism.md)
- [M6a Dataset Builder And JSONL Pipeline](./2026-05-15-m6a-dataset-builder-and-jsonl-pipeline.md)
- [M3 Multi-Provider Embedding Direction](./2026-05-15-m3-multi-provider-embedding.md)
- [Current Implementation Plan](../plans/2026-05-15-implementation-plan.md)
