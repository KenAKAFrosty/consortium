use std::time::Duration;

use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use crate::embeddings::{
    AgnosticEmbeddingError, Embedder, EmbeddingBatch, EmbeddingUsage,
    openai_failure_to_agnostic,
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// Maximum number of inputs accepted by OpenAI's `/v1/embeddings` endpoint in a
/// single request. Used to auto-chunk over-limit batches inside
/// [`OpenAiEmbedder::embed`] so callers never need to hand-shard.
const MAX_INPUTS_PER_REQUEST: usize = 2048;

#[derive(Debug, Clone)]
pub struct OpenAiEmbedder {
    http: reqwest::Client,
    base_url: String,
    api_key: SecretString,
    model: OpenAiEmbeddingModel,
}

impl OpenAiEmbedder {
    pub fn new(api_key: impl Into<SecretString>) -> Self {
        Self::new_with_base_url(api_key, DEFAULT_BASE_URL)
    }

    pub fn new_with_base_url(
        api_key: impl Into<SecretString>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into(),
            api_key: api_key.into(),
            model: OpenAiEmbeddingModel::TextEmbedding3Small,
        }
    }

    pub fn from_env() -> Result<Self, OpenAiEmbedderError> {
        Self::from_env_with_base_url(DEFAULT_BASE_URL)
    }

    pub fn from_env_with_base_url(
        base_url: impl Into<String>,
    ) -> Result<Self, OpenAiEmbedderError> {
        let key =
            std::env::var("OPENAI_API_KEY").map_err(|_| OpenAiEmbedderError::MissingApiKey)?;
        Ok(Self::new_with_base_url(key, base_url))
    }

    pub fn with_model(mut self, model: OpenAiEmbeddingModel) -> Self {
        self.model = model;
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OpenAiEmbedderError {
    #[error("OPENAI_API_KEY is not set in the environment")]
    MissingApiKey,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenAiEmbeddingModel {
    TextEmbedding3Small,
    TextEmbedding3Large,
    TextEmbeddingAda002,
    /// Escape hatch for OpenAI text embedding models not yet enumerated.
    Custom(String),
}

impl OpenAiEmbeddingModel {
    pub fn as_api_str(&self) -> &str {
        match self {
            Self::TextEmbedding3Small => "text-embedding-3-small",
            Self::TextEmbedding3Large => "text-embedding-3-large",
            Self::TextEmbeddingAda002 => "text-embedding-ada-002",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for OpenAiEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_api_str())
    }
}

impl Serialize for OpenAiEmbeddingModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_api_str())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OpenAiEmbeddingFailure {
    #[error("openai-embedding transport failure: {0}")]
    Transport(#[source] reqwest::Error),
    #[error("openai-embedding response deserialization failed: {0}")]
    Deserialize(#[source] serde_json::Error),
    #[error("openai-embedding authentication failed")]
    Auth { message: Option<String> },
    #[error("openai-embedding rate limited")]
    RateLimited {
        retry_after: Option<Duration>,
        message: Option<String>,
    },
    #[error("openai-embedding invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("openai-embedding server error (status {status})")]
    ServerError {
        status: u16,
        message: Option<String>,
    },
    #[error("openai-embedding response malformed: {reason}")]
    MalformedResponse { reason: String },
}

impl crate::ai_client_apis::shared::FailureFromStatus for OpenAiEmbeddingFailure {
    fn auth(message: Option<String>) -> Self {
        Self::Auth { message }
    }
    fn rate_limited(retry_after: Option<Duration>, message: Option<String>) -> Self {
        Self::RateLimited {
            retry_after,
            message,
        }
    }
    fn invalid_request(message: String) -> Self {
        Self::InvalidRequest { message }
    }
    fn server_error(status: u16, message: Option<String>) -> Self {
        Self::ServerError { status, message }
    }
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a OpenAiEmbeddingModel,
    input: &'a [String],
}

#[derive(Deserialize)]
struct WireResponse {
    data: Vec<WireDataEntry>,
    usage: WireUsage,
}

#[derive(Deserialize)]
struct WireDataEntry {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct WireUsage {
    prompt_tokens: u64,
}

impl Embedder for OpenAiEmbedder {
    async fn embed(
        &self,
        inputs: &[String],
    ) -> Result<EmbeddingBatch, AgnosticEmbeddingError> {
        openai_embed_raw(self, inputs)
            .await
            .map_err(openai_failure_to_agnostic)
    }
}

/// Top-level entry point used by [`Embedder::embed`]. Auto-chunks
/// `inputs.chunks(MAX_INPUTS_PER_REQUEST)` and concatenates per-chunk results
/// in input order. The single-chunk path (including the empty-input case)
/// calls [`openai_embed_chunk`] directly with the original slice, so no extra
/// allocation happens in the common case. A failing chunk short-circuits with
/// its typed [`OpenAiEmbeddingFailure`]; vectors from earlier chunks are not
/// returned partially.
///
/// Cross-chunk vector dimensions are also verified: each chunk is already
/// intra-chunk-uniform (enforced inside [`openai_embed_chunk`]), so this loop
/// only needs to anchor on the first non-empty chunk's dimension and reject
/// any later chunk whose first vector dimension differs.
async fn openai_embed_raw(
    embedder: &OpenAiEmbedder,
    inputs: &[String],
) -> Result<EmbeddingBatch, OpenAiEmbeddingFailure> {
    if inputs.len() <= MAX_INPUTS_PER_REQUEST {
        return openai_embed_chunk(embedder, inputs).await;
    }

    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
    let mut input_tokens: u64 = 0;
    let mut expected_dim: Option<usize> = None;
    for chunk in inputs.chunks(MAX_INPUTS_PER_REQUEST) {
        let batch = openai_embed_chunk(embedder, chunk).await?;
        if let Some(first) = batch.vectors.first() {
            let chunk_dim = first.len();
            match expected_dim {
                None => expected_dim = Some(chunk_dim),
                Some(prev) if prev != chunk_dim => {
                    return Err(OpenAiEmbeddingFailure::MalformedResponse {
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
    Ok(EmbeddingBatch {
        vectors,
        usage: EmbeddingUsage { input_tokens },
    })
}

async fn openai_embed_chunk(
    embedder: &OpenAiEmbedder,
    inputs: &[String],
) -> Result<EmbeddingBatch, OpenAiEmbeddingFailure> {
    let body = WireRequest {
        model: &embedder.model,
        input: inputs,
    };

    let url = format!("{}/v1/embeddings", embedder.base_url);
    let response = embedder
        .http
        .post(&url)
        .bearer_auth(embedder.api_key.expose_secret())
        .json(&body)
        .send()
        .await
        .map_err(OpenAiEmbeddingFailure::Transport)?;

    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response
        .bytes()
        .await
        .map_err(OpenAiEmbeddingFailure::Transport)?;

    if !status.is_success() {
        return Err(crate::ai_client_apis::shared::map_status_to_failure(
            status.as_u16(),
            &headers,
            &bytes,
        ));
    }

    let parsed: WireResponse =
        serde_json::from_slice(&bytes).map_err(OpenAiEmbeddingFailure::Deserialize)?;

    // OpenAI's response is documented as same-order-as-inputs, but the API still
    // returns an explicit `index` per entry. Honor it: reconstruct vectors by
    // index so the agnostic contract `inputs[i] -> vectors[i]` holds even if a
    // future API change reorders the response.
    let n = inputs.len();
    let mut slots: Vec<Option<Vec<f32>>> = (0..n).map(|_| None).collect();
    for entry in parsed.data {
        if entry.index >= n {
            return Err(OpenAiEmbeddingFailure::MalformedResponse {
                reason: format!(
                    "response contained index {} but only {} inputs were sent",
                    entry.index, n
                ),
            });
        }
        if slots[entry.index].is_some() {
            return Err(OpenAiEmbeddingFailure::MalformedResponse {
                reason: format!("response contained duplicate index {}", entry.index),
            });
        }
        slots[entry.index] = Some(entry.embedding);
    }

    let mut vectors = Vec::with_capacity(n);
    for (i, slot) in slots.into_iter().enumerate() {
        match slot {
            Some(v) => vectors.push(v),
            None => {
                return Err(OpenAiEmbeddingFailure::MalformedResponse {
                    reason: format!("response missing index {}", i),
                });
            }
        }
    }

    // Reject intra-chunk mixed-dimension responses early. The dataset layer's
    // EmbeddingDimensionMismatch guard (src/dataset/mod.rs) is now a
    // defensive backstop — the typed MalformedResponse surfaces here, with
    // full provider provenance, before the agnostic Embedder boundary
    // returns.
    if let Some(first) = vectors.first() {
        let expected = first.len();
        for (i, v) in vectors.iter().enumerate().skip(1) {
            if v.len() != expected {
                return Err(OpenAiEmbeddingFailure::MalformedResponse {
                    reason: format!(
                        "vector at index {i} has dimension {} but vector at index 0 has dimension {expected}",
                        v.len()
                    ),
                });
            }
        }
    }

    Ok(EmbeddingBatch {
        vectors,
        usage: EmbeddingUsage {
            input_tokens: parsed.usage.prompt_tokens,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_inputs() -> Vec<String> {
        vec!["alpha".to_string(), "beta".to_string()]
    }

    #[test]
    fn openai_embedding_model_serializes_to_expected_wire_value() {
        assert_eq!(
            OpenAiEmbeddingModel::TextEmbedding3Small.as_api_str(),
            "text-embedding-3-small"
        );
        assert_eq!(
            OpenAiEmbeddingModel::TextEmbedding3Large.as_api_str(),
            "text-embedding-3-large"
        );
        assert_eq!(
            OpenAiEmbeddingModel::TextEmbeddingAda002.as_api_str(),
            "text-embedding-ada-002"
        );
        assert_eq!(
            OpenAiEmbeddingModel::Custom("text-embedding-foo".to_string()).as_api_str(),
            "text-embedding-foo"
        );

        let json = serde_json::to_string(&OpenAiEmbeddingModel::TextEmbedding3Small).unwrap();
        assert_eq!(json, r#""text-embedding-3-small""#);
    }

    #[test]
    fn debug_output_redacts_api_key() {
        let embedder = OpenAiEmbedder::new_with_base_url(
            "super-secret-openai-embedder-key".to_string(),
            "https://example.test",
        );
        let debug = format!("{embedder:?}");
        assert!(
            !debug.contains("super-secret-openai-embedder-key"),
            "Debug output must not leak api_key: {debug}"
        );
    }

    #[tokio::test]
    async fn success_returns_vectors_and_tokens() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/embeddings")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                        {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]}
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 4, "total_tokens": 4}
                }"#,
            )
            .create_async()
            .await;

        let embedder =
            OpenAiEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let batch = embedder.embed(&sample_inputs()).await.expect("embed");

        assert_eq!(batch.vectors.len(), 2);
        assert_eq!(batch.vectors[0], vec![0.1, 0.2]);
        assert_eq!(batch.vectors[1], vec![0.3, 0.4]);
        assert_eq!(batch.usage.input_tokens, 4);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn auth_failure_on_401() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(401)
            .with_body(r#"{"error":{"message":"Invalid API key"}}"#)
            .create_async()
            .await;

        let embedder =
            OpenAiEmbedder::new_with_base_url("bad-key".to_string(), server.url());
        let err = embedder.embed(&sample_inputs()).await.expect_err("auth");
        match err {
            AgnosticEmbeddingError::Auth { provider, message } => {
                assert_eq!(provider, crate::EmbeddingProvider::OpenAi);
                assert_eq!(message.as_deref(), Some("Invalid API key"));
            }
            other => panic!("expected Auth, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rate_limit_carries_retry_after() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(429)
            .with_header("retry-after", "20")
            .with_body(r#"{"error":{"message":"rate limited"}}"#)
            .create_async()
            .await;

        let embedder =
            OpenAiEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder.embed(&sample_inputs()).await.expect_err("rate");
        match err {
            AgnosticEmbeddingError::RateLimited {
                retry_after,
                message,
                ..
            } => {
                assert_eq!(retry_after, Some(Duration::from_secs(20)));
                assert_eq!(message.as_deref(), Some("rate limited"));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn invalid_request_on_400() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(400)
            .with_body(r#"{"error":{"message":"input cannot be empty"}}"#)
            .create_async()
            .await;

        let embedder =
            OpenAiEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder.embed(&sample_inputs()).await.expect_err("invalid");
        match err {
            AgnosticEmbeddingError::InvalidRequest { message, .. } => {
                assert_eq!(message, "input cannot be empty");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn server_error_on_503() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(503)
            .with_body(r#"{"error":{"message":"unavailable"}}"#)
            .create_async()
            .await;

        let embedder =
            OpenAiEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder.embed(&sample_inputs()).await.expect_err("server");
        match err {
            AgnosticEmbeddingError::ServerError {
                status, message, ..
            } => {
                assert_eq!(status, 503);
                assert_eq!(message.as_deref(), Some("unavailable"));
            }
            other => panic!("expected ServerError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn malformed_json_maps_to_deserialize() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(200)
            .with_body("not valid json")
            .create_async()
            .await;

        let embedder =
            OpenAiEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder.embed(&sample_inputs()).await.expect_err("deser");
        assert!(matches!(err, AgnosticEmbeddingError::Deserialize { .. }));
    }

    #[tokio::test]
    async fn empty_data_maps_to_malformed_response() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(200)
            .with_body(
                r#"{
                    "object": "list",
                    "data": [],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 0, "total_tokens": 0}
                }"#,
            )
            .create_async()
            .await;

        let embedder =
            OpenAiEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder.embed(&sample_inputs()).await.expect_err("malformed");
        match err {
            AgnosticEmbeddingError::MalformedResponse { reason, .. } => {
                assert!(reason.contains("missing index 0"), "got: {reason}");
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn out_of_order_data_is_reconstructed_by_index() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(200)
            .with_body(
                r#"{
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": 2, "embedding": [0.5, 0.6]},
                        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                        {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]}
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 6, "total_tokens": 6}
                }"#,
            )
            .create_async()
            .await;

        let embedder =
            OpenAiEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let inputs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let batch = embedder.embed(&inputs).await.expect("embed");

        assert_eq!(batch.vectors.len(), 3);
        assert_eq!(batch.vectors[0], vec![0.1, 0.2]);
        assert_eq!(batch.vectors[1], vec![0.3, 0.4]);
        assert_eq!(batch.vectors[2], vec![0.5, 0.6]);
    }

    #[tokio::test]
    async fn out_of_range_index_maps_to_malformed_response() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(200)
            .with_body(
                r#"{
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                        {"object": "embedding", "index": 5, "embedding": [0.3, 0.4]}
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 4, "total_tokens": 4}
                }"#,
            )
            .create_async()
            .await;

        let embedder =
            OpenAiEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder.embed(&sample_inputs()).await.expect_err("out of range");
        match err {
            AgnosticEmbeddingError::MalformedResponse { reason, .. } => {
                assert!(
                    reason.contains("index 5") && reason.contains("2 inputs"),
                    "got: {reason}"
                );
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn duplicate_index_maps_to_malformed_response() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(200)
            .with_body(
                r#"{
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                        {"object": "embedding", "index": 0, "embedding": [0.3, 0.4]}
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 4, "total_tokens": 4}
                }"#,
            )
            .create_async()
            .await;

        let embedder =
            OpenAiEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder.embed(&sample_inputs()).await.expect_err("duplicate");
        match err {
            AgnosticEmbeddingError::MalformedResponse { reason, .. } => {
                assert!(reason.contains("duplicate index 0"), "got: {reason}");
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn missing_non_zero_index_maps_to_malformed_response() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(200)
            .with_body(
                r#"{
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 2, "total_tokens": 2}
                }"#,
            )
            .create_async()
            .await;

        let embedder =
            OpenAiEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder.embed(&sample_inputs()).await.expect_err("missing");
        match err {
            AgnosticEmbeddingError::MalformedResponse { reason, .. } => {
                assert!(reason.contains("missing index 1"), "got: {reason}");
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    /// Build a synthetic OpenAI embeddings response covering `count` entries.
    /// Each entry's `embedding` is `[(starting_global + chunk_local_index) as
    /// f32, 0.0]`, so callers can recover the exact input-position from the
    /// returned vector's first element and assert order is preserved across
    /// chunks.
    fn build_openai_chunk_response_body(
        starting_global: usize,
        count: usize,
        tokens: u64,
    ) -> String {
        let data: Vec<serde_json::Value> = (0..count)
            .map(|i| {
                serde_json::json!({
                    "object": "embedding",
                    "index": i,
                    "embedding": [(starting_global + i) as f32, 0.0_f32],
                })
            })
            .collect();
        serde_json::json!({
            "object": "list",
            "data": data,
            "model": "text-embedding-3-small",
            "usage": { "prompt_tokens": tokens, "total_tokens": tokens },
        })
        .to_string()
    }

    /// Build `total` input strings where the first input of each expected
    /// chunk carries a unique `marker` substring so mockito body matchers can
    /// route each chunk's request to its own mock without false positives.
    fn marker_inputs_for_two_chunks(
        total: usize,
        chunk_size: usize,
        marker_a: &str,
        marker_b: &str,
    ) -> Vec<String> {
        let mut inputs = Vec::with_capacity(total);
        for i in 0..total {
            let s = if i == 0 {
                marker_a.to_string()
            } else if i == chunk_size {
                marker_b.to_string()
            } else {
                format!("input-{i}")
            };
            inputs.push(s);
        }
        inputs
    }

    /// Over-limit batch (`MAX_INPUTS_PER_REQUEST + 2` inputs) must auto-chunk
    /// into two `/v1/embeddings` requests, concatenate vectors in input order,
    /// and sum the per-chunk `usage.input_tokens` into one aggregated
    /// [`EmbeddingUsage`].
    #[tokio::test]
    async fn auto_chunks_over_limit_inputs_and_preserves_order_with_aggregated_usage() {
        let total = MAX_INPUTS_PER_REQUEST + 2;
        let inputs = marker_inputs_for_two_chunks(
            total,
            MAX_INPUTS_PER_REQUEST,
            "openai-auto-chunk-0-marker",
            "openai-auto-chunk-1-marker",
        );

        let chunk_0_body =
            build_openai_chunk_response_body(0, MAX_INPUTS_PER_REQUEST, 100);
        let chunk_1_body =
            build_openai_chunk_response_body(MAX_INPUTS_PER_REQUEST, 2, 50);

        let mut server = mockito::Server::new_async().await;
        let m0 = server
            .mock("POST", "/v1/embeddings")
            .match_body(mockito::Matcher::Regex(
                "openai-auto-chunk-0-marker".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(chunk_0_body)
            .expect(1)
            .create_async()
            .await;
        let m1 = server
            .mock("POST", "/v1/embeddings")
            .match_body(mockito::Matcher::Regex(
                "openai-auto-chunk-1-marker".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(chunk_1_body)
            .expect(1)
            .create_async()
            .await;

        let embedder =
            OpenAiEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let batch = embedder.embed(&inputs).await.expect("multi-chunk embed");

        assert_eq!(batch.vectors.len(), total, "one vector per input");
        for (i, v) in batch.vectors.iter().enumerate() {
            assert_eq!(
                v,
                &vec![i as f32, 0.0_f32],
                "vector at global index {i} mismatched — chunk concatenation broke ordering"
            );
        }
        assert_eq!(
            batch.usage.input_tokens, 150,
            "aggregated usage must sum across chunks (100 + 50)"
        );

        m0.assert_async().await;
        m1.assert_async().await;
    }

    /// A later-chunk failure must surface as the correct typed
    /// [`AgnosticEmbeddingError`] from that chunk. Vectors from prior
    /// successful chunks must not leak out as a partial-success
    /// [`EmbeddingBatch`].
    #[tokio::test]
    async fn auto_chunking_propagates_later_chunk_failure_without_silent_truncation() {
        let total = MAX_INPUTS_PER_REQUEST + 2;
        let inputs = marker_inputs_for_two_chunks(
            total,
            MAX_INPUTS_PER_REQUEST,
            "openai-fail-chunk-0-marker",
            "openai-fail-chunk-1-marker",
        );

        let chunk_0_body =
            build_openai_chunk_response_body(0, MAX_INPUTS_PER_REQUEST, 100);

        let mut server = mockito::Server::new_async().await;
        let _m0 = server
            .mock("POST", "/v1/embeddings")
            .match_body(mockito::Matcher::Regex(
                "openai-fail-chunk-0-marker".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(chunk_0_body)
            .expect(1)
            .create_async()
            .await;
        let _m1 = server
            .mock("POST", "/v1/embeddings")
            .match_body(mockito::Matcher::Regex(
                "openai-fail-chunk-1-marker".to_string(),
            ))
            .with_status(401)
            .with_body(r#"{"error":{"message":"Invalid API key on chunk 1"}}"#)
            .expect(1)
            .create_async()
            .await;

        let embedder =
            OpenAiEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder
            .embed(&inputs)
            .await
            .expect_err("later-chunk failure must surface, not be silently truncated");
        match err {
            AgnosticEmbeddingError::Auth { provider, message } => {
                assert_eq!(provider, crate::EmbeddingProvider::OpenAi);
                assert_eq!(message.as_deref(), Some("Invalid API key on chunk 1"));
            }
            other => panic!("expected Auth from second chunk, got {other:?}"),
        }
    }

    /// A single chunk response with vectors of differing dimensions must
    /// surface as [`AgnosticEmbeddingError::MalformedResponse`] inside the
    /// embedder itself — *before* the dataset layer's
    /// `EmbeddingDimensionMismatch` guard runs. This proves the dataset
    /// guard is a defensive backstop rather than the primary path.
    #[tokio::test]
    async fn intra_chunk_mixed_dimension_response_maps_to_malformed_response() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/embeddings")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "object": "list",
                    "data": [
                        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
                        {"object": "embedding", "index": 1, "embedding": [0.4, 0.5]}
                    ],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 4, "total_tokens": 4}
                }"#,
            )
            .create_async()
            .await;

        let embedder =
            OpenAiEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder
            .embed(&sample_inputs())
            .await
            .expect_err("mixed-dimension response must surface as MalformedResponse");
        match err {
            AgnosticEmbeddingError::MalformedResponse { provider, reason } => {
                assert_eq!(provider, crate::EmbeddingProvider::OpenAi);
                assert!(
                    reason.contains("vector at index 1")
                        && reason.contains("dimension 2")
                        && reason.contains("dimension 3"),
                    "got: {reason}"
                );
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    /// An auto-chunked batch whose later chunk returns vectors of a
    /// different dimension than the first chunk must surface as
    /// [`AgnosticEmbeddingError::MalformedResponse`] from the cross-chunk
    /// check in [`openai_embed_raw`]. Earlier vectors must not leak out as
    /// a partial-success batch.
    #[tokio::test]
    async fn cross_chunk_dimension_drift_maps_to_malformed_response() {
        let total = MAX_INPUTS_PER_REQUEST + 2;
        let inputs = marker_inputs_for_two_chunks(
            total,
            MAX_INPUTS_PER_REQUEST,
            "openai-dim-drift-chunk-0-marker",
            "openai-dim-drift-chunk-1-marker",
        );

        // Chunk 0 returns vectors of dimension 2 (the standard build_*_chunk
        // helper output). Chunk 1 returns a hand-built body whose two
        // vectors are dimension 3 — uniform within the chunk, but
        // inconsistent across chunks.
        let chunk_0_body =
            build_openai_chunk_response_body(0, MAX_INPUTS_PER_REQUEST, 100);
        let chunk_1_body = serde_json::json!({
            "object": "list",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [9.0_f32, 9.0_f32, 9.0_f32]},
                {"object": "embedding", "index": 1, "embedding": [9.0_f32, 9.0_f32, 9.0_f32]},
            ],
            "model": "text-embedding-3-small",
            "usage": { "prompt_tokens": 50, "total_tokens": 50 },
        })
        .to_string();

        let mut server = mockito::Server::new_async().await;
        let _m0 = server
            .mock("POST", "/v1/embeddings")
            .match_body(mockito::Matcher::Regex(
                "openai-dim-drift-chunk-0-marker".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(chunk_0_body)
            .expect(1)
            .create_async()
            .await;
        let _m1 = server
            .mock("POST", "/v1/embeddings")
            .match_body(mockito::Matcher::Regex(
                "openai-dim-drift-chunk-1-marker".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(chunk_1_body)
            .expect(1)
            .create_async()
            .await;

        let embedder =
            OpenAiEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder
            .embed(&inputs)
            .await
            .expect_err("cross-chunk dimension drift must surface as MalformedResponse");
        match err {
            AgnosticEmbeddingError::MalformedResponse { provider, reason } => {
                assert_eq!(provider, crate::EmbeddingProvider::OpenAi);
                assert!(
                    reason.contains("chunk vector dimension 3")
                        && reason.contains("earlier chunk dimension 2"),
                    "got: {reason}"
                );
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY; run with `cargo test -- --ignored`"]
    async fn live_openai_embedding_returns_real_vectors() {
        let embedder = OpenAiEmbedder::from_env().expect("OPENAI_API_KEY must be set");
        let batch = embedder
            .embed(&["hello world".to_string()])
            .await
            .expect("live openai embed should succeed");
        assert_eq!(batch.vectors.len(), 1);
        assert!(!batch.vectors[0].is_empty(), "vector should be non-empty");
        assert!(batch.usage.input_tokens > 0);
    }
}
