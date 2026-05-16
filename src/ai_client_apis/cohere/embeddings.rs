use std::time::Duration;

use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

use crate::embeddings::{
    AgnosticEmbeddingError, Embedder, EmbeddingBatch, EmbeddingUsage,
    cohere_failure_to_agnostic,
};

const DEFAULT_BASE_URL: &str = "https://api.cohere.com";
const EMBEDDING_TYPES_FLOAT: &[&str] = &["float"];

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

async fn cohere_embed_raw(
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
