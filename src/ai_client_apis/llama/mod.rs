use std::time::Duration;

use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://api.llama.com";

#[derive(Debug, Clone)]
pub struct LlamaClient {
    http: reqwest::Client,
    base_url: String,
    api_key: SecretString,
}

impl LlamaClient {
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
        }
    }

    pub fn from_env() -> Result<Self, LlamaClientError> {
        Self::from_env_with_base_url(DEFAULT_BASE_URL)
    }

    pub fn from_env_with_base_url(
        base_url: impl Into<String>,
    ) -> Result<Self, LlamaClientError> {
        let key = std::env::var("LLAMA_API_KEY").map_err(|_| LlamaClientError::MissingApiKey)?;
        Ok(Self::new_with_base_url(key, base_url))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LlamaClientError {
    #[error("LLAMA_API_KEY is not set in the environment")]
    MissingApiKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaRole {
    User,
    Assistant,
}

impl LlamaRole {
    fn as_wire_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlamaModel {
    /// Llama 4 Maverick 17B (128-expert MoE, FP8). Maps to wire id
    /// `Llama-4-Maverick-17B-128E-Instruct-FP8`.
    Llama4Maverick17B,
    /// Llama 4 Scout 17B (16-expert MoE, FP8). Maps to wire id
    /// `Llama-4-Scout-17B-16E-Instruct-FP8`.
    Llama4Scout17B,
    /// Llama 3.3 70B Instruct. Maps to wire id `Llama-3.3-70B-Instruct`.
    Llama3_3_70B,
    /// Escape hatch for Meta-hosted OpenAI-compatible chat models that have not been
    /// added to this enum yet.
    Custom(String),
}

impl LlamaModel {
    pub fn as_api_str(&self) -> &str {
        match self {
            Self::Llama4Maverick17B => "Llama-4-Maverick-17B-128E-Instruct-FP8",
            Self::Llama4Scout17B => "Llama-4-Scout-17B-16E-Instruct-FP8",
            Self::Llama3_3_70B => "Llama-3.3-70B-Instruct",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for LlamaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_api_str())
    }
}

impl Serialize for LlamaModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_api_str())
    }
}

#[derive(Debug, Clone)]
pub struct LlamaMessage {
    pub role: LlamaRole,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct LlamaCompletionCommand {
    pub model: LlamaModel,
    pub system_prompt: Option<String>,
    pub messages: Vec<LlamaMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

impl Default for LlamaCompletionCommand {
    fn default() -> Self {
        Self {
            model: LlamaModel::Llama3_3_70B,
            system_prompt: None,
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
        }
    }
}

#[derive(Debug)]
pub struct LlamaCompletionSuccess {
    pub content: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum LlamaCompletionFailure {
    #[error("llama transport failure: {0}")]
    Transport(#[source] reqwest::Error),
    #[error("llama response deserialization failed: {0}")]
    Deserialize(#[source] serde_json::Error),
    #[error("llama authentication failed")]
    Auth { message: Option<String> },
    #[error("llama rate limited")]
    RateLimited {
        retry_after: Option<Duration>,
        message: Option<String>,
    },
    #[error("llama invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("llama server error (status {status})")]
    ServerError {
        status: u16,
        message: Option<String>,
    },
    #[error("llama response malformed: {reason}")]
    MalformedResponse { reason: String },
}

impl super::shared::FailureFromStatus for LlamaCompletionFailure {
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

pub type LlamaResult = Result<LlamaCompletionSuccess, LlamaCompletionFailure>;

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a LlamaModel,
    messages: &'a [WireMessage<'a>],
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct WireMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct WireResponse {
    choices: Vec<WireResponseChoice>,
    usage: WireUsage,
}

#[derive(Deserialize)]
struct WireResponseChoice {
    message: WireResponseMessage,
}

#[derive(Deserialize)]
struct WireResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct WireUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

pub async fn llama_get_completion(
    client: &LlamaClient,
    command: &LlamaCompletionCommand,
) -> LlamaResult {
    let mut wire_messages: Vec<WireMessage<'_>> =
        Vec::with_capacity(command.messages.len() + usize::from(command.system_prompt.is_some()));
    if let Some(system) = command.system_prompt.as_deref() {
        wire_messages.push(WireMessage {
            role: "system",
            content: system,
        });
    }
    for msg in &command.messages {
        wire_messages.push(WireMessage {
            role: msg.role.as_wire_str(),
            content: &msg.content,
        });
    }

    let body = WireRequest {
        model: &command.model,
        messages: &wire_messages,
        max_tokens: command.max_tokens,
        temperature: command.temperature,
    };

    let url = format!("{}/v1/chat/completions", client.base_url);
    let response = client
        .http
        .post(&url)
        .bearer_auth(client.api_key.expose_secret())
        .json(&body)
        .send()
        .await
        .map_err(LlamaCompletionFailure::Transport)?;

    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response
        .bytes()
        .await
        .map_err(LlamaCompletionFailure::Transport)?;

    if status.is_success() {
        let parsed: WireResponse =
            serde_json::from_slice(&bytes).map_err(LlamaCompletionFailure::Deserialize)?;
        let first_choice = parsed.choices.into_iter().next().ok_or_else(|| {
            LlamaCompletionFailure::MalformedResponse {
                reason: "response contained no choices".to_string(),
            }
        })?;
        Ok(LlamaCompletionSuccess {
            content: first_choice.message.content,
            prompt_tokens: parsed.usage.prompt_tokens,
            completion_tokens: parsed.usage.completion_tokens,
        })
    } else {
        Err(super::shared::map_status_to_failure(
            status.as_u16(),
            &headers,
            &bytes,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_command() -> LlamaCompletionCommand {
        LlamaCompletionCommand {
            model: LlamaModel::Llama3_3_70B,
            system_prompt: Some("You are a helpful assistant.".to_string()),
            messages: vec![LlamaMessage {
                role: LlamaRole::User,
                content: "Hi".to_string(),
            }],
            max_tokens: Some(64),
            temperature: Some(0.0),
        }
    }

    #[test]
    fn llama_model_serializes_to_expected_wire_value() {
        assert_eq!(
            LlamaModel::Llama4Maverick17B.as_api_str(),
            "Llama-4-Maverick-17B-128E-Instruct-FP8"
        );
        assert_eq!(
            LlamaModel::Llama4Scout17B.as_api_str(),
            "Llama-4-Scout-17B-16E-Instruct-FP8"
        );
        assert_eq!(
            LlamaModel::Llama3_3_70B.as_api_str(),
            "Llama-3.3-70B-Instruct"
        );
        assert_eq!(
            LlamaModel::Custom("Llama-3.1-8B-Instruct".to_string()).as_api_str(),
            "Llama-3.1-8B-Instruct"
        );

        assert_eq!(
            LlamaModel::Llama3_3_70B.to_string(),
            "Llama-3.3-70B-Instruct"
        );

        let json = serde_json::to_string(&LlamaModel::Llama4Scout17B).expect("serialize");
        assert_eq!(json, r#""Llama-4-Scout-17B-16E-Instruct-FP8""#);
    }

    #[tokio::test]
    async fn success_returns_content_and_tokens() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .match_header("authorization", "Bearer test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3}
                }"#,
            )
            .create_async()
            .await;

        let client = LlamaClient::new_with_base_url("test-key".to_string(), server.url());
        let success = llama_get_completion(&client, &sample_command())
            .await
            .expect("expected success");
        assert_eq!(success.content, "Hello!");
        assert_eq!(success.prompt_tokens, 7);
        assert_eq!(success.completion_tokens, 3);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn auth_failure_on_401() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(401)
            .with_body(r#"{"error":{"message":"Invalid API key"}}"#)
            .create_async()
            .await;

        let client = LlamaClient::new_with_base_url("bad-key".to_string(), server.url());
        let err = llama_get_completion(&client, &sample_command())
            .await
            .expect_err("expected auth failure");
        match err {
            LlamaCompletionFailure::Auth { message } => {
                assert_eq!(message.as_deref(), Some("Invalid API key"));
            }
            other => panic!("expected Auth failure, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn auth_failure_on_403() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(403)
            .with_body(r#"{"error":{"message":"Forbidden"}}"#)
            .create_async()
            .await;

        let client = LlamaClient::new_with_base_url("test-key".to_string(), server.url());
        let err = llama_get_completion(&client, &sample_command())
            .await
            .expect_err("expected auth failure");
        assert!(matches!(err, LlamaCompletionFailure::Auth { .. }));
    }

    #[tokio::test]
    async fn rate_limit_carries_retry_after() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(429)
            .with_header("retry-after", "15")
            .with_body(r#"{"error":{"message":"Slow down"}}"#)
            .create_async()
            .await;

        let client = LlamaClient::new_with_base_url("test-key".to_string(), server.url());
        let err = llama_get_completion(&client, &sample_command())
            .await
            .expect_err("expected rate limit");
        match err {
            LlamaCompletionFailure::RateLimited {
                retry_after,
                message,
            } => {
                assert_eq!(retry_after, Some(Duration::from_secs(15)));
                assert_eq!(message.as_deref(), Some("Slow down"));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn invalid_request_on_400() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(400)
            .with_body(r#"{"error":{"message":"Unsupported model"}}"#)
            .create_async()
            .await;

        let client = LlamaClient::new_with_base_url("test-key".to_string(), server.url());
        let err = llama_get_completion(&client, &sample_command())
            .await
            .expect_err("expected invalid request");
        match err {
            LlamaCompletionFailure::InvalidRequest { message } => {
                assert_eq!(message, "Unsupported model");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn server_error_on_503() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(503)
            .with_body(r#"{"error":{"message":"Service unavailable"}}"#)
            .create_async()
            .await;

        let client = LlamaClient::new_with_base_url("test-key".to_string(), server.url());
        let err = llama_get_completion(&client, &sample_command())
            .await
            .expect_err("expected server error");
        match err {
            LlamaCompletionFailure::ServerError { status, message } => {
                assert_eq!(status, 503);
                assert_eq!(message.as_deref(), Some("Service unavailable"));
            }
            other => panic!("expected ServerError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn malformed_json_maps_to_deserialize() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_body("not valid json")
            .create_async()
            .await;

        let client = LlamaClient::new_with_base_url("test-key".to_string(), server.url());
        let err = llama_get_completion(&client, &sample_command())
            .await
            .expect_err("expected deserialize failure");
        assert!(matches!(err, LlamaCompletionFailure::Deserialize(_)));
    }

    #[tokio::test]
    async fn empty_choices_maps_to_malformed_response() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_body(r#"{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0}}"#)
            .create_async()
            .await;

        let client = LlamaClient::new_with_base_url("test-key".to_string(), server.url());
        let err = llama_get_completion(&client, &sample_command())
            .await
            .expect_err("empty choices must surface as a typed failure");
        match err {
            LlamaCompletionFailure::MalformedResponse { reason } => {
                assert!(reason.contains("no choices"), "unexpected reason: {reason}");
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    #[test]
    fn debug_output_redacts_api_key() {
        let client = LlamaClient::new_with_base_url(
            "super-secret-llama-key".to_string(),
            "https://example.test",
        );
        let debug = format!("{client:?}");
        assert!(
            !debug.contains("super-secret-llama-key"),
            "Debug output must not leak api_key: {debug}"
        );
    }

    #[tokio::test]
    #[ignore = "requires LLAMA_API_KEY; run with `cargo test -- --ignored`"]
    async fn live_llama_completion_returns_real_response() {
        let client = LlamaClient::from_env().expect("LLAMA_API_KEY must be set");
        let command = LlamaCompletionCommand {
            model: LlamaModel::Llama3_3_70B,
            system_prompt: Some("Reply with exactly the word 'pong'.".to_string()),
            messages: vec![LlamaMessage {
                role: LlamaRole::User,
                content: "ping".to_string(),
            }],
            max_tokens: Some(8),
            temperature: Some(0.0),
        };
        let success = llama_get_completion(&client, &command)
            .await
            .expect("live Llama request should succeed");
        assert!(!success.content.is_empty(), "content should be non-empty");
        assert!(success.prompt_tokens > 0, "prompt_tokens should be > 0");
        assert!(
            success.completion_tokens > 0,
            "completion_tokens should be > 0"
        );
    }
}
