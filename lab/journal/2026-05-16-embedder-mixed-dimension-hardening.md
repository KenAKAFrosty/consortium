[README](../../README.md) | [Lab](../README.md) | [Journal](./README.md) | [Plans](../plans/README.md) | [Decisions](../decisions/README.md)

# 2026-05-16 — Embedder Mixed-Dimension Hardening

Status: validated
Date: 2026-05-16

## Question

[M6a post-review](./2026-05-15-m6a-dataset-builder-and-jsonl-pipeline.md#revision-post-review-guard-embedding-dimensions-before-select_diverse) added [`DatasetRunError::EmbeddingDimensionMismatch`](../../src/dataset/mod.rs) because [`select_diverse`](../../src/diversification/mod.rs) is documented to **panic** on jagged inputs. The dataset layer caught that hole — but the shipped embedders ([`OpenAiEmbedder`](../../src/ai_client_apis/openai/embeddings.rs), [`CohereEmbedder`](../../src/ai_client_apis/cohere/embeddings.rs)) still happily forwarded mixed-dimension provider responses out across the agnostic [`Embedder`] boundary. The [M6c follow-up list](./2026-05-16-m6c-embedder-auto-chunking.md#whats-next) flagged this as a known gap: provider-side hardening should make `EmbeddingDimensionMismatch` a defensive backstop rather than a primary path.

Two real malformed shapes can surface today:

1. **Intra-chunk mismatch.** A single provider response with vectors of differing dimensions inside one `/v1/embeddings` or `/v2/embed` request. The current parsers accept it.
2. **Cross-chunk drift.** Auto-chunking ([M6c](./2026-05-16-m6c-embedder-auto-chunking.md)) issues multiple HTTP requests for over-limit batches. A later chunk whose vectors are a different dimension than the first chunk's would slip through too, because each chunk's response was being validated in isolation.

Both cases would have escaped the agnostic boundary as an apparently-valid [`EmbeddingBatch`] and only been caught at the dataset layer's guard — losing typed provider provenance (`provider: EmbeddingProvider`) along the way.

## Hypothesis

Lock the hardening slice around four constraints:

1. **Public `Embedder` surface unchanged.** The trait, the agnostic [`AgnosticEmbeddingError`] variant set, and every public constructor stay exactly as they are.
2. **Dataset-layer `EmbeddingDimensionMismatch` guard stays in place.** Custom `Embedder` impls are free to skip the per-chunk hardening; the dataset layer must keep its backstop.
3. **Intra-chunk check inside each `*_embed_chunk`.** Anchor on `vectors[0].len()` after the response is parsed; reject the first divergent index as `MalformedResponse`. Cohere's existing empty-vector guard ensures `vectors[0]` exists; OpenAI's existing missing-index guard plays the same role.
4. **Cross-chunk check inside each `*_embed_raw`.** Anchor `expected_dim` on the first non-empty chunk. Each chunk is already intra-chunk-uniform, so the cross-chunk loop only needs to compare one vector per chunk. Mismatch surfaces as `MalformedResponse` with the typed provider provenance and short-circuits the chunking loop (no partial-success surface).

## What We Tried

Two files changed: [`src/ai_client_apis/openai/embeddings.rs`](../../src/ai_client_apis/openai/embeddings.rs) and [`src/ai_client_apis/cohere/embeddings.rs`](../../src/ai_client_apis/cohere/embeddings.rs). Identical shape per provider.

### Intra-chunk check

OpenAI: added after the slot-rebuild loop, immediately before the `Ok(EmbeddingBatch { .. })` return. Anchors on `vectors[0].len()` and iterates `vectors.iter().enumerate().skip(1)` — first mismatch returns
`OpenAiEmbeddingFailure::MalformedResponse { reason: "vector at index {i} has dimension {actual} but vector at index 0 has dimension {expected}" }`.
Empty `vectors` is a no-op (the `if let Some(first) = vectors.first()` guard) — the existing missing-index path already rejects non-empty inputs that returned no data.

Cohere: same shape, after the existing `embeddings.float.is_empty()` and count-mismatch guards. Those guards already ensure `parsed.embeddings.float[0]` exists, so the dimension anchor is direct (no `if let`). Same reason-string shape.

### Cross-chunk check

Both providers' `*_embed_raw` chunking shells gained:

```rust
let mut expected_dim: Option<usize> = None;
for chunk in inputs.chunks(MAX_INPUTS_PER_REQUEST) {
    let batch = *_embed_chunk(embedder, chunk).await?;
    if let Some(first) = batch.vectors.first() {
        let chunk_dim = first.len();
        match expected_dim {
            None => expected_dim = Some(chunk_dim),
            Some(prev) if prev != chunk_dim => {
                return Err(*EmbeddingFailure::MalformedResponse {
                    reason: format!(
                        "chunk vector dimension {chunk_dim} differs from earlier chunk dimension {prev}"
                    ),
                });
            }
            Some(_) => {}
        }
    }
    vectors.extend(batch.vectors);
    input_tokens = input_tokens.saturating_add(batch.usage.input_tokens);
}
```

Each chunk is already intra-chunk-uniform thanks to the per-chunk check; cross-chunk only needs to compare `batch.vectors[0].len()` against `expected_dim`. Empty-vector chunks (theoretical only — empty inputs route to the single-chunk fast path, not into this loop) are skipped without anchoring. Mismatch short-circuits with the typed `MalformedResponse`; earlier chunks' vectors are dropped — consistent with the [M6c](./2026-05-16-m6c-embedder-auto-chunking.md) "no partial-success surface" guarantee.

### What deliberately is not in scope

- M7 provider expansion (Deepseek / Kimi K2 / Qwen / Llama).
- M8 genericification (`Provider` / `Embedder` traits with associated types).
- Multimodal Cohere `embed-v4.0` — flagged in [M6c](./2026-05-16-m6c-embedder-auto-chunking.md) as needing its own per-request limit and a separate code path; adding it would also need its own dimension-hardening pass.
- Embedding-side retry primitive analogous to the completions path — not load-bearing today.
- Dataset-layer change. `DatasetRunError::EmbeddingDimensionMismatch` stays in place; the dataset-level regression test (`execute_returns_typed_error_when_embedder_yields_mixed_dimension_vectors`) stays green as a defensive backstop.

## Result

`cargo test --lib` is **127 passed / 0 failed / 6 ignored** (+4 over M5c's 123 / 6). All pre-existing tests stay green, including the M6c auto-chunking happy-path and later-chunk-failure tests and the M6a dataset-layer jagged-dimension regression.

Four new tests:

1. **`openai::embeddings::tests::intra_chunk_mixed_dimension_response_maps_to_malformed_response`.** Single `/v1/embeddings` response with `data[0].embedding = [0.1, 0.2, 0.3]` and `data[1].embedding = [0.4, 0.5]`. Asserts the returned `AgnosticEmbeddingError::MalformedResponse` carries `provider = OpenAi` and a reason mentioning `vector at index 1`, `dimension 2`, and `dimension 3`.

2. **`openai::embeddings::tests::cross_chunk_dimension_drift_maps_to_malformed_response`.** `MAX_INPUTS_PER_REQUEST + 2` inputs routed to two mocks via unique-marker substrings (same routing pattern as the M6c chunking tests). Chunk 0 returns vectors of dim 2 (via `build_openai_chunk_response_body`); chunk 1 returns hand-built dim-3 vectors. Asserts the returned `AgnosticEmbeddingError::MalformedResponse` carries `provider = OpenAi` and a reason mentioning `chunk vector dimension 3` and `earlier chunk dimension 2`.

3. **`cohere::embeddings::tests::intra_chunk_mixed_dimension_response_maps_to_malformed_response`.** Single `/v2/embed` response with `embeddings.float = [[0.1, 0.2, 0.3], [0.4, 0.5]]`. Symmetric assertions to the OpenAI case with `provider = Cohere`.

4. **`cohere::embeddings::tests::cross_chunk_dimension_drift_maps_to_malformed_response`.** `MAX_INPUTS_PER_REQUEST + 4` inputs routed to two mocks via unique markers. Chunk 0 returns dim-2 vectors; chunk 1 returns hand-built dim-3 vectors. Symmetric assertions with `provider = Cohere`.

## Verified Properties

- **Public surface unchanged.** [`Embedder::embed`] signature, [`AgnosticEmbeddingError`] variant set, all public constructors, and [`EmbeddingBatch`] / [`EmbeddingUsage`] shapes are identical to M6c.
- **Provider provenance preserved.** Both new failure paths carry `EmbeddingProvider::{OpenAi, Cohere}` on the typed `MalformedResponse`, so callers can attribute the error.
- **No partial-success surface.** Cross-chunk mismatch short-circuits and discards earlier chunks' vectors, consistent with the [M6c](./2026-05-16-m6c-embedder-auto-chunking.md) "first failure wins" contract. The later-chunk-failure tests from M6c stay green.
- **M6c chunking semantics preserved.** Same intra-chunk routing under `mockito` with regex-marker matchers; same input-order vector concatenation in the happy paths; same `saturating_add` aggregation of `EmbeddingUsage.input_tokens`.
- **Dataset-layer backstop intact.** `DatasetRunError::EmbeddingDimensionMismatch` and its `execute_returns_typed_error_when_embedder_yields_mixed_dimension_vectors` regression test were not touched.
- **No new clippy warnings.** Pre-existing warnings (M7 stubs + one `collapsible_if` in `diversification`) unchanged.

## Decision

Lock the hardening slice as shipped. The provider-side `MalformedResponse` is now the primary path for mixed-dimension responses; the dataset layer's `EmbeddingDimensionMismatch` becomes a defensive backstop for custom `Embedder` impls that choose not to validate, exactly as the M6a post-review note anticipated.

The agnostic `Embedder` boundary still expresses one logical call per `embed` with uniform-dimension guarantees from the shipped impls; custom impls remain free to issue a single request and skip the per-chunk check. The trait doc on `Embedder` already covers this stance (M6c) and does not need a new clause for mixed-dimension hardening — uniform per-row dimensionality is implicit in "a batch of input-position-preserving vectors".

## Next

- **M7.** Replicate the M2 provider pattern for Deepseek / Kimi K2 / Qwen / Llama. The same intra-chunk dimension check applies to any new embedder providers as they land.
- **M8.** Generic-first `Provider` / `Embedder` traits with associated types. The hardening pattern (per-chunk anchor + cross-chunk anchor) is small enough to inline per provider; if a generic helper becomes warranted under M8, it would live in `src/embeddings/` or a new `embeddings/shared.rs`, parallel to the completions-side `src/ai_client_apis/shared.rs::FailureFromStatus`.
- **Multimodal Cohere `embed-v4.0`.** Its dimension assumptions differ (multimodal embeddings may diverge across modality); the hardening shape may need to be relaxed or made input-type-aware before that lands.

## See Also

- [M6c Embedder Auto-Chunking](./2026-05-16-m6c-embedder-auto-chunking.md)
- [M6a Dataset Builder And JSONL Pipeline](./2026-05-15-m6a-dataset-builder-and-jsonl-pipeline.md)
- [M3 Multi-Provider Embedding Direction](./2026-05-15-m3-multi-provider-embedding.md)
- [Current Implementation Plan](../plans/2026-05-15-implementation-plan.md)
