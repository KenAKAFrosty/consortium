use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EmbeddingProvider {
    Cohere,
    OpenAi,
}

#[derive(Debug, thiserror::Error)]
pub enum AgnosticEmbeddingError {
    #[error("{provider:?}: transport failure: {source}")]
    Transport {
        provider: EmbeddingProvider,
        #[source]
        source: reqwest::Error,
    },
    /// Provider HTTP response JSON could not be decoded against the wire schema.
    /// Non-transient: re-issuing the same request would yield the same payload.
    /// Not for malformed structured text produced by the LLM — embedders do not
    /// generate text.
    #[error("{provider:?}: response deserialization failed: {source}")]
    Deserialize {
        provider: EmbeddingProvider,
        #[source]
        source: serde_json::Error,
    },
    #[error("{provider:?}: rate limited")]
    RateLimited {
        provider: EmbeddingProvider,
        retry_after: Option<Duration>,
        message: Option<String>,
    },
    #[error("{provider:?}: authentication failed")]
    Auth {
        provider: EmbeddingProvider,
        message: Option<String>,
    },
    #[error("{provider:?}: invalid request: {message}")]
    InvalidRequest {
        provider: EmbeddingProvider,
        message: String,
    },
    #[error("{provider:?}: server error (status {status})")]
    ServerError {
        provider: EmbeddingProvider,
        status: u16,
        message: Option<String>,
    },
    /// Provider returned valid JSON that parsed against the wire schema but failed
    /// a semantic invariant — e.g., the response was 200 OK but contained no
    /// embedding vectors, or the vector count did not match the input count.
    /// Non-transient.
    #[error("{provider:?}: response malformed: {reason}")]
    MalformedResponse {
        provider: EmbeddingProvider,
        reason: String,
    },
}

impl AgnosticEmbeddingError {
    pub fn provider(&self) -> EmbeddingProvider {
        match self {
            Self::Transport { provider, .. }
            | Self::Deserialize { provider, .. }
            | Self::RateLimited { provider, .. }
            | Self::Auth { provider, .. }
            | Self::InvalidRequest { provider, .. }
            | Self::ServerError { provider, .. }
            | Self::MalformedResponse { provider, .. } => *provider,
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingUsage {
    pub input_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct EmbeddingBatch {
    pub vectors: Vec<Vec<f32>>,
    pub usage: EmbeddingUsage,
}

/// Provider-agnostic embedding boundary.
///
/// `inputs[i]` corresponds to `vectors[i]` in the returned batch — order is
/// preserved. An empty `inputs` slice is forwarded to the provider as-is;
/// implementations do not pre-validate.
///
/// Implementations may issue one or more HTTP requests. The shipped
/// [`crate::OpenAiEmbedder`] and [`crate::CohereEmbedder`] auto-chunk inputs
/// at the provider's documented per-request limit (OpenAI: 2048, Cohere v3:
/// 96), concatenate per-chunk results in input order, and sum per-chunk
/// [`EmbeddingUsage`] into one aggregated total. If any chunk fails, the
/// typed [`AgnosticEmbeddingError`] for that chunk is returned — there is no
/// partial-success surface. Custom [`Embedder`] impls are free to issue a
/// single request and reject over-limit inputs themselves.
///
/// Native `async fn` in trait (Rust 2024). Static dispatch only; `dyn Embedder`
/// is not supported.
pub trait Embedder {
    fn embed(
        &self,
        inputs: &[String],
    ) -> impl std::future::Future<Output = Result<EmbeddingBatch, AgnosticEmbeddingError>> + Send;
}

pub(crate) fn cohere_failure_to_agnostic(
    failure: crate::ai_client_apis::cohere::embeddings::CohereEmbeddingFailure,
) -> AgnosticEmbeddingError {
    use crate::ai_client_apis::cohere::embeddings::CohereEmbeddingFailure as F;
    let provider = EmbeddingProvider::Cohere;
    match failure {
        F::Transport(source) => AgnosticEmbeddingError::Transport { provider, source },
        F::Deserialize(source) => AgnosticEmbeddingError::Deserialize { provider, source },
        F::Auth { message } => AgnosticEmbeddingError::Auth { provider, message },
        F::RateLimited {
            retry_after,
            message,
        } => AgnosticEmbeddingError::RateLimited {
            provider,
            retry_after,
            message,
        },
        F::InvalidRequest { message } => AgnosticEmbeddingError::InvalidRequest {
            provider,
            message,
        },
        F::ServerError { status, message } => AgnosticEmbeddingError::ServerError {
            provider,
            status,
            message,
        },
        F::MalformedResponse { reason } => {
            AgnosticEmbeddingError::MalformedResponse { provider, reason }
        }
    }
}

pub(crate) fn openai_failure_to_agnostic(
    failure: crate::ai_client_apis::openai::embeddings::OpenAiEmbeddingFailure,
) -> AgnosticEmbeddingError {
    use crate::ai_client_apis::openai::embeddings::OpenAiEmbeddingFailure as F;
    let provider = EmbeddingProvider::OpenAi;
    match failure {
        F::Transport(source) => AgnosticEmbeddingError::Transport { provider, source },
        F::Deserialize(source) => AgnosticEmbeddingError::Deserialize { provider, source },
        F::Auth { message } => AgnosticEmbeddingError::Auth { provider, message },
        F::RateLimited {
            retry_after,
            message,
        } => AgnosticEmbeddingError::RateLimited {
            provider,
            retry_after,
            message,
        },
        F::InvalidRequest { message } => AgnosticEmbeddingError::InvalidRequest {
            provider,
            message,
        },
        F::ServerError { status, message } => AgnosticEmbeddingError::ServerError {
            provider,
            status,
            message,
        },
        F::MalformedResponse { reason } => {
            AgnosticEmbeddingError::MalformedResponse { provider, reason }
        }
    }
}
