use std::time::Duration;

use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use crate::embeddings::{
    AgnosticEmbeddingError, Embedder, EmbeddingBatch, EmbeddingUsage,
    cohere_failure_to_agnostic,
};

const DEFAULT_BASE_URL: &str = "https://api.cohere.com";
const EMBEDDING_TYPES_FLOAT: &[&str] = &["float"];

/// Maximum number of inputs accepted by Cohere's `/v2/embed` endpoint in a
/// single request for v3-family text embedding models. Used to auto-chunk
/// over-limit batches inside [`CohereEmbedder::embed`] so callers never need
/// to hand-shard. A v4 multimodal path would need its own constant.
const MAX_INPUTS_PER_REQUEST: usize = 96;

#[derive(Debug, Clone)]
pub struct CohereEmbedder {
    http: reqwest::Client,
    base_url: String,
    api_key: SecretString,
    model: CohereEmbeddingModel,
    input_type: CohereEmbeddingInputType,
}

impl CohereEmbedder {
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
            model: CohereEmbeddingModel::EmbedEnglishV3,
            input_type: CohereEmbeddingInputType::SearchDocument,
        }
    }

    pub fn from_env() -> Result<Self, CohereClientError> {
        Self::from_env_with_base_url(DEFAULT_BASE_URL)
    }

    pub fn from_env_with_base_url(
        base_url: impl Into<String>,
    ) -> Result<Self, CohereClientError> {
        let key = std::env::var("COHERE_API_KEY").map_err(|_| CohereClientError::MissingApiKey)?;
        Ok(Self::new_with_base_url(key, base_url))
    }

    pub fn with_model(mut self, model: CohereEmbeddingModel) -> Self {
        self.model = model;
        self
    }

    pub fn with_input_type(mut self, input_type: CohereEmbeddingInputType) -> Self {
        self.input_type = input_type;
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CohereClientError {
    #[error("COHERE_API_KEY is not set in the environment")]
    MissingApiKey,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CohereEmbeddingModel {
    EmbedEnglishV3,
    EmbedMultilingualV3,
    EmbedEnglishLightV3,
    /// Escape hatch for Cohere text embedding models not yet enumerated. Not
    /// appropriate for multimodal models (`embed-v4.0`) — this seed only emits
    /// text-embedding-shaped requests.
    Custom(String),
}

impl CohereEmbeddingModel {
    pub fn as_api_str(&self) -> &str {
        match self {
            Self::EmbedEnglishV3 => "embed-english-v3.0",
            Self::EmbedMultilingualV3 => "embed-multilingual-v3.0",
            Self::EmbedEnglishLightV3 => "embed-english-light-v3.0",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for CohereEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_api_str())
    }
}

impl Serialize for CohereEmbeddingModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_api_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CohereEmbeddingInputType {
    SearchDocument,
    SearchQuery,
    Classification,
    Clustering,
}

#[derive(Debug, thiserror::Error)]
pub enum CohereEmbeddingFailure {
    #[error("cohere transport failure: {0}")]
    Transport(#[source] reqwest::Error),
    #[error("cohere response deserialization failed: {0}")]
    Deserialize(#[source] serde_json::Error),
    #[error("cohere authentication failed")]
    Auth { message: Option<String> },
    #[error("cohere rate limited")]
    RateLimited {
        retry_after: Option<Duration>,
        message: Option<String>,
    },
    #[error("cohere invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("cohere server error (status {status})")]
    ServerError {
        status: u16,
        message: Option<String>,
    },
    #[error("cohere response malformed: {reason}")]
    MalformedResponse { reason: String },
}

impl crate::ai_client_apis::shared::FailureFromStatus for CohereEmbeddingFailure {
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
    model: &'a CohereEmbeddingModel,
    input_type: CohereEmbeddingInputType,
    texts: &'a [String],
    embedding_types: &'static [&'static str],
}

#[derive(Deserialize)]
struct WireResponse {
    embeddings: WireEmbeddingsField,
    #[serde(default)]
    meta: Option<WireMeta>,
}

#[derive(Deserialize)]
struct WireEmbeddingsField {
    float: Vec<Vec<f32>>,
}

#[derive(Deserialize, Default)]
struct WireMeta {
    #[serde(default)]
    billed_units: Option<WireBilledUnits>,
}

#[derive(Deserialize, Default)]
struct WireBilledUnits {
    #[serde(default)]
    input_tokens: u64,
}

impl Embedder for CohereEmbedder {
    async fn embed(
        &self,
        inputs: &[String],
    ) -> Result<EmbeddingBatch, AgnosticEmbeddingError> {
        let failure = cohere_embed_raw(self, inputs).await;
        failure.map_err(cohere_failure_to_agnostic)
    }
}

/// Top-level entry point used by [`Embedder::embed`]. Auto-chunks
/// `inputs.chunks(MAX_INPUTS_PER_REQUEST)` and concatenates per-chunk results
/// in input order. The single-chunk path (including the empty-input case)
/// calls [`cohere_embed_chunk`] directly with the original slice, so no extra
/// allocation happens in the common case. A failing chunk short-circuits with
/// its typed [`CohereEmbeddingFailure`]; vectors from earlier chunks are not
/// returned partially.
///
/// Cross-chunk vector dimensions are also verified: each chunk is already
/// intra-chunk-uniform (enforced inside [`cohere_embed_chunk`]), so this loop
/// only needs to anchor on the first non-empty chunk's dimension and reject
/// any later chunk whose first vector dimension differs.
async fn cohere_embed_raw(
    embedder: &CohereEmbedder,
    inputs: &[String],
) -> Result<EmbeddingBatch, CohereEmbeddingFailure> {
    if inputs.len() <= MAX_INPUTS_PER_REQUEST {
        return cohere_embed_chunk(embedder, inputs).await;
    }

    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
    let mut input_tokens: u64 = 0;
    let mut expected_dim: Option<usize> = None;
    for chunk in inputs.chunks(MAX_INPUTS_PER_REQUEST) {
        let batch = cohere_embed_chunk(embedder, chunk).await?;
        if let Some(first) = batch.vectors.first() {
            let chunk_dim = first.len();
            match expected_dim {
                None => expected_dim = Some(chunk_dim),
                Some(prev) if prev != chunk_dim => {
                    return Err(CohereEmbeddingFailure::MalformedResponse {
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

async fn cohere_embed_chunk(
    embedder: &CohereEmbedder,
    inputs: &[String],
) -> Result<EmbeddingBatch, CohereEmbeddingFailure> {
    let body = WireRequest {
        model: &embedder.model,
        input_type: embedder.input_type,
        texts: inputs,
        embedding_types: EMBEDDING_TYPES_FLOAT,
    };

    let url = format!("{}/v2/embed", embedder.base_url);
    let response = embedder
        .http
        .post(&url)
        .bearer_auth(embedder.api_key.expose_secret())
        .json(&body)
        .send()
        .await
        .map_err(CohereEmbeddingFailure::Transport)?;

    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response
        .bytes()
        .await
        .map_err(CohereEmbeddingFailure::Transport)?;

    if !status.is_success() {
        return Err(crate::ai_client_apis::shared::map_status_to_failure(
            status.as_u16(),
            &headers,
            &bytes,
        ));
    }

    let parsed: WireResponse =
        serde_json::from_slice(&bytes).map_err(CohereEmbeddingFailure::Deserialize)?;

    if parsed.embeddings.float.is_empty() {
        return Err(CohereEmbeddingFailure::MalformedResponse {
            reason: "response contained no embeddings".to_string(),
        });
    }
    if parsed.embeddings.float.len() != inputs.len() {
        return Err(CohereEmbeddingFailure::MalformedResponse {
            reason: format!(
                "embedding count mismatch: requested {}, received {}",
                inputs.len(),
                parsed.embeddings.float.len()
            ),
        });
    }

    // Reject intra-chunk mixed-dimension responses early. The dataset layer's
    // EmbeddingDimensionMismatch guard (src/dataset/mod.rs) is now a
    // defensive backstop — the typed MalformedResponse surfaces here, with
    // full provider provenance, before the agnostic Embedder boundary
    // returns. The empty-vector path is guarded above, so vectors[0] exists.
    let expected = parsed.embeddings.float[0].len();
    for (i, v) in parsed.embeddings.float.iter().enumerate().skip(1) {
        if v.len() != expected {
            return Err(CohereEmbeddingFailure::MalformedResponse {
                reason: format!(
                    "vector at index {i} has dimension {} but vector at index 0 has dimension {expected}",
                    v.len()
                ),
            });
        }
    }

    let input_tokens = parsed
        .meta
        .and_then(|m| m.billed_units)
        .map(|b| b.input_tokens)
        .unwrap_or(0);

    Ok(EmbeddingBatch {
        vectors: parsed.embeddings.float,
        usage: EmbeddingUsage { input_tokens },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_inputs() -> Vec<String> {
        vec!["alpha".to_string(), "beta".to_string()]
    }

    #[test]
    fn cohere_embedding_model_serializes_to_expected_wire_value() {
        assert_eq!(
            CohereEmbeddingModel::EmbedEnglishV3.as_api_str(),
            "embed-english-v3.0"
        );
        assert_eq!(
            CohereEmbeddingModel::EmbedMultilingualV3.as_api_str(),
            "embed-multilingual-v3.0"
        );
        assert_eq!(
            CohereEmbeddingModel::EmbedEnglishLightV3.as_api_str(),
            "embed-english-light-v3.0"
        );
        assert_eq!(
            CohereEmbeddingModel::Custom("embed-foo".to_string()).as_api_str(),
            "embed-foo"
        );

        let json =
            serde_json::to_string(&CohereEmbeddingModel::EmbedEnglishV3).expect("serialize");
        assert_eq!(json, r#""embed-english-v3.0""#);
    }

    #[test]
    fn cohere_input_type_serializes_as_snake_case() {
        let json = serde_json::to_string(&CohereEmbeddingInputType::SearchDocument).unwrap();
        assert_eq!(json, r#""search_document""#);
        let json = serde_json::to_string(&CohereEmbeddingInputType::Clustering).unwrap();
        assert_eq!(json, r#""clustering""#);
    }

    #[test]
    fn debug_output_redacts_api_key() {
        let embedder = CohereEmbedder::new_with_base_url(
            "super-secret-cohere-key".to_string(),
            "https://example.test",
        );
        let debug = format!("{embedder:?}");
        assert!(
            !debug.contains("super-secret-cohere-key"),
            "Debug output must not leak api_key: {debug}"
        );
    }

    #[tokio::test]
    async fn success_returns_vectors_and_tokens() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v2/embed")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "id": "abc",
                    "embeddings": {"float": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]},
                    "texts": ["alpha", "beta"],
                    "meta": {"billed_units": {"input_tokens": 5}}
                }"#,
            )
            .create_async()
            .await;

        let embedder =
            CohereEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let batch = embedder.embed(&sample_inputs()).await.expect("embed");

        assert_eq!(batch.vectors.len(), 2);
        assert_eq!(batch.vectors[0], vec![0.1, 0.2, 0.3]);
        assert_eq!(batch.vectors[1], vec![0.4, 0.5, 0.6]);
        assert_eq!(batch.usage.input_tokens, 5);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn auth_failure_on_401() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v2/embed")
            .with_status(401)
            .with_body(r#"{"error":{"message":"invalid api token"}}"#)
            .create_async()
            .await;

        let embedder =
            CohereEmbedder::new_with_base_url("bad-key".to_string(), server.url());
        let err = embedder.embed(&sample_inputs()).await.expect_err("auth");
        match err {
            AgnosticEmbeddingError::Auth { provider, message } => {
                assert_eq!(provider, crate::EmbeddingProvider::Cohere);
                assert_eq!(message.as_deref(), Some("invalid api token"));
            }
            other => panic!("expected Auth, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rate_limit_carries_retry_after() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v2/embed")
            .with_status(429)
            .with_header("retry-after", "12")
            .with_body(r#"{"error":{"message":"throttled"}}"#)
            .create_async()
            .await;

        let embedder =
            CohereEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder.embed(&sample_inputs()).await.expect_err("rate");
        match err {
            AgnosticEmbeddingError::RateLimited {
                retry_after,
                message,
                ..
            } => {
                assert_eq!(retry_after, Some(Duration::from_secs(12)));
                assert_eq!(message.as_deref(), Some("throttled"));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn invalid_request_on_400() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v2/embed")
            .with_status(400)
            .with_body(r#"{"error":{"message":"texts cannot be empty"}}"#)
            .create_async()
            .await;

        let embedder =
            CohereEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder.embed(&sample_inputs()).await.expect_err("invalid");
        match err {
            AgnosticEmbeddingError::InvalidRequest { message, .. } => {
                assert_eq!(message, "texts cannot be empty");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn server_error_on_503() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v2/embed")
            .with_status(503)
            .with_body(r#"{"error":{"message":"unavailable"}}"#)
            .create_async()
            .await;

        let embedder =
            CohereEmbedder::new_with_base_url("test-key".to_string(), server.url());
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
            .mock("POST", "/v2/embed")
            .with_status(200)
            .with_body("not valid json")
            .create_async()
            .await;

        let embedder =
            CohereEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder.embed(&sample_inputs()).await.expect_err("deser");
        assert!(matches!(err, AgnosticEmbeddingError::Deserialize { .. }));
    }

    #[tokio::test]
    async fn empty_embeddings_maps_to_malformed_response() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v2/embed")
            .with_status(200)
            .with_body(
                r#"{
                    "id": "abc",
                    "embeddings": {"float": []},
                    "texts": [],
                    "meta": {"billed_units": {"input_tokens": 0}}
                }"#,
            )
            .create_async()
            .await;

        let embedder =
            CohereEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder.embed(&sample_inputs()).await.expect_err("malformed");
        match err {
            AgnosticEmbeddingError::MalformedResponse { reason, .. } => {
                assert!(reason.contains("no embeddings"), "got: {reason}");
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn embedding_count_mismatch_maps_to_malformed_response() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v2/embed")
            .with_status(200)
            .with_body(
                r#"{
                    "embeddings": {"float": [[0.1, 0.2]]},
                    "meta": {"billed_units": {"input_tokens": 1}}
                }"#,
            )
            .create_async()
            .await;

        let embedder =
            CohereEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder.embed(&sample_inputs()).await.expect_err("mismatch");
        match err {
            AgnosticEmbeddingError::MalformedResponse { reason, .. } => {
                assert!(reason.contains("count mismatch"), "got: {reason}");
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    /// Build a synthetic Cohere embeddings response covering `count` entries.
    /// The i-th vector is `[(starting_global + i) as f32, 0.0]`, so callers
    /// can recover the original input position from the returned vector and
    /// assert order preservation across chunks.
    fn build_cohere_chunk_response_body(
        starting_global: usize,
        count: usize,
        tokens: u64,
    ) -> String {
        let float: Vec<Vec<f32>> = (0..count)
            .map(|i| vec![(starting_global + i) as f32, 0.0_f32])
            .collect();
        serde_json::json!({
            "id": "test",
            "embeddings": { "float": float },
            "meta": { "billed_units": { "input_tokens": tokens } },
        })
        .to_string()
    }

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

    /// Over-limit batch (`MAX_INPUTS_PER_REQUEST + 4` inputs) must auto-chunk
    /// into two `/v2/embed` requests, concatenate vectors in input order, and
    /// sum the per-chunk `meta.billed_units.input_tokens` into one aggregated
    /// [`EmbeddingUsage`].
    #[tokio::test]
    async fn auto_chunks_over_limit_inputs_and_preserves_order_with_aggregated_usage() {
        let total = MAX_INPUTS_PER_REQUEST + 4;
        let inputs = marker_inputs_for_two_chunks(
            total,
            MAX_INPUTS_PER_REQUEST,
            "cohere-auto-chunk-0-marker",
            "cohere-auto-chunk-1-marker",
        );

        let chunk_0_body =
            build_cohere_chunk_response_body(0, MAX_INPUTS_PER_REQUEST, 60);
        let chunk_1_body =
            build_cohere_chunk_response_body(MAX_INPUTS_PER_REQUEST, 4, 10);

        let mut server = mockito::Server::new_async().await;
        let m0 = server
            .mock("POST", "/v2/embed")
            .match_body(mockito::Matcher::Regex(
                "cohere-auto-chunk-0-marker".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(chunk_0_body)
            .expect(1)
            .create_async()
            .await;
        let m1 = server
            .mock("POST", "/v2/embed")
            .match_body(mockito::Matcher::Regex(
                "cohere-auto-chunk-1-marker".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(chunk_1_body)
            .expect(1)
            .create_async()
            .await;

        let embedder =
            CohereEmbedder::new_with_base_url("test-key".to_string(), server.url());
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
            batch.usage.input_tokens, 70,
            "aggregated usage must sum across chunks (60 + 10)"
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
        let total = MAX_INPUTS_PER_REQUEST + 4;
        let inputs = marker_inputs_for_two_chunks(
            total,
            MAX_INPUTS_PER_REQUEST,
            "cohere-fail-chunk-0-marker",
            "cohere-fail-chunk-1-marker",
        );

        let chunk_0_body =
            build_cohere_chunk_response_body(0, MAX_INPUTS_PER_REQUEST, 60);

        let mut server = mockito::Server::new_async().await;
        let _m0 = server
            .mock("POST", "/v2/embed")
            .match_body(mockito::Matcher::Regex(
                "cohere-fail-chunk-0-marker".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(chunk_0_body)
            .expect(1)
            .create_async()
            .await;
        let _m1 = server
            .mock("POST", "/v2/embed")
            .match_body(mockito::Matcher::Regex(
                "cohere-fail-chunk-1-marker".to_string(),
            ))
            .with_status(503)
            .with_body(r#"{"error":{"message":"upstream unavailable on chunk 1"}}"#)
            .expect(1)
            .create_async()
            .await;

        let embedder =
            CohereEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder
            .embed(&inputs)
            .await
            .expect_err("later-chunk failure must surface, not be silently truncated");
        match err {
            AgnosticEmbeddingError::ServerError {
                provider,
                status,
                message,
            } => {
                assert_eq!(provider, crate::EmbeddingProvider::Cohere);
                assert_eq!(status, 503);
                assert_eq!(
                    message.as_deref(),
                    Some("upstream unavailable on chunk 1")
                );
            }
            other => panic!("expected ServerError from second chunk, got {other:?}"),
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
            .mock("POST", "/v2/embed")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "id": "abc",
                    "embeddings": {"float": [[0.1, 0.2, 0.3], [0.4, 0.5]]},
                    "texts": ["alpha", "beta"],
                    "meta": {"billed_units": {"input_tokens": 5}}
                }"#,
            )
            .create_async()
            .await;

        let embedder =
            CohereEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder
            .embed(&sample_inputs())
            .await
            .expect_err("mixed-dimension response must surface as MalformedResponse");
        match err {
            AgnosticEmbeddingError::MalformedResponse { provider, reason } => {
                assert_eq!(provider, crate::EmbeddingProvider::Cohere);
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
    /// check in [`cohere_embed_raw`]. Earlier vectors must not leak out as
    /// a partial-success batch.
    #[tokio::test]
    async fn cross_chunk_dimension_drift_maps_to_malformed_response() {
        let total = MAX_INPUTS_PER_REQUEST + 4;
        let inputs = marker_inputs_for_two_chunks(
            total,
            MAX_INPUTS_PER_REQUEST,
            "cohere-dim-drift-chunk-0-marker",
            "cohere-dim-drift-chunk-1-marker",
        );

        // Chunk 0 returns vectors of dimension 2 (the standard build_*_chunk
        // helper output). Chunk 1 returns a hand-built body whose four
        // vectors are dimension 3 — uniform within the chunk, but
        // inconsistent across chunks.
        let chunk_0_body =
            build_cohere_chunk_response_body(0, MAX_INPUTS_PER_REQUEST, 60);
        let chunk_1_float: Vec<Vec<f32>> =
            (0..4).map(|_| vec![9.0_f32, 9.0_f32, 9.0_f32]).collect();
        let chunk_1_body = serde_json::json!({
            "id": "test",
            "embeddings": { "float": chunk_1_float },
            "meta": { "billed_units": { "input_tokens": 10 } },
        })
        .to_string();

        let mut server = mockito::Server::new_async().await;
        let _m0 = server
            .mock("POST", "/v2/embed")
            .match_body(mockito::Matcher::Regex(
                "cohere-dim-drift-chunk-0-marker".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(chunk_0_body)
            .expect(1)
            .create_async()
            .await;
        let _m1 = server
            .mock("POST", "/v2/embed")
            .match_body(mockito::Matcher::Regex(
                "cohere-dim-drift-chunk-1-marker".to_string(),
            ))
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(chunk_1_body)
            .expect(1)
            .create_async()
            .await;

        let embedder =
            CohereEmbedder::new_with_base_url("test-key".to_string(), server.url());
        let err = embedder
            .embed(&inputs)
            .await
            .expect_err("cross-chunk dimension drift must surface as MalformedResponse");
        match err {
            AgnosticEmbeddingError::MalformedResponse { provider, reason } => {
                assert_eq!(provider, crate::EmbeddingProvider::Cohere);
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
    #[ignore = "requires COHERE_API_KEY; run with `cargo test -- --ignored`"]
    async fn live_cohere_embedding_returns_real_vectors() {
        let embedder = CohereEmbedder::from_env().expect("COHERE_API_KEY must be set");
        let batch = embedder
            .embed(&["hello world".to_string()])
            .await
            .expect("live cohere embed should succeed");
        assert_eq!(batch.vectors.len(), 1);
        assert!(!batch.vectors[0].is_empty(), "vector should be non-empty");
    }
}
