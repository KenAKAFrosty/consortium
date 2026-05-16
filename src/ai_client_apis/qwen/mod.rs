use std::time::Duration;

use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

/// International (global) DashScope endpoint. The China-region endpoint
/// (`https://dashscope.aliyuncs.com/compatible-mode`) can be substituted via
/// [`QwenClient::new_with_base_url`] / [`QwenClient::from_env_with_base_url`].
const DEFAULT_BASE_URL: &str = "https://dashscope-intl.aliyuncs.com/compatible-mode";

#[derive(Debug, Clone)]
pub struct QwenClient {
    http: reqwest::Client,
    base_url: String,
    api_key: SecretString,
}

impl QwenClient {
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

    pub fn from_env() -> Result<Self, QwenClientError> {
        Self::from_env_with_base_url(DEFAULT_BASE_URL)
    }

    pub fn from_env_with_base_url(
        base_url: impl Into<String>,
    ) -> Result<Self, QwenClientError> {
        let key =
            std::env::var("DASHSCOPE_API_KEY").map_err(|_| QwenClientError::MissingApiKey)?;
        Ok(Self::new_with_base_url(key, base_url))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum QwenClientError {
    #[error("DASHSCOPE_API_KEY is not set in the environment")]
    MissingApiKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenRole {
    User,
    Assistant,
}

impl QwenRole {
    fn as_wire_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QwenModel {
    /// Latency-optimized general chat model. Maps to wire id `qwen-turbo`.
    QwenTurbo,
    /// Balanced general chat model. Maps to wire id `qwen-plus`.
    QwenPlus,
    /// Capability-optimized general chat model. Maps to wire id `qwen-max`.
    QwenMax,
    /// Qwen3 32B dense chat model. Maps to wire id `qwen3-32b`.
    Qwen3_32B,
    /// Qwen3 235B MoE chat model. Maps to wire id `qwen3-235b-a22b`.
    Qwen3_235BA22B,
    /// Escape hatch for DashScope-hosted OpenAI-compatible chat models that have not
    /// been added to this enum yet.
    Custom(String),
}

impl QwenModel {
    pub fn as_api_str(&self) -> &str {
        match self {
            Self::QwenTurbo => "qwen-turbo",
            Self::QwenPlus => "qwen-plus",
            Self::QwenMax => "qwen-max",
            Self::Qwen3_32B => "qwen3-32b",
            Self::Qwen3_235BA22B => "qwen3-235b-a22b",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for QwenModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_api_str())
    }
}

impl Serialize for QwenModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_api_str())
    }
}

#[derive(Debug, Clone)]
pub struct QwenMessage {
    pub role: QwenRole,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct QwenCompletionCommand {
    pub model: QwenModel,
    pub system_prompt: Option<String>,
    pub messages: Vec<QwenMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

impl Default for QwenCompletionCommand {
    fn default() -> Self {
        Self {
            model: QwenModel::QwenPlus,
            system_prompt: None,
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
        }
    }
}

#[derive(Debug)]
pub struct QwenCompletionSuccess {
    pub content: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum QwenCompletionFailure {
    #[error("qwen transport failure: {0}")]
    Transport(#[source] reqwest::Error),
    #[error("qwen response deserialization failed: {0}")]
    Deserialize(#[source] serde_json::Error),
    #[error("qwen authentication failed")]
    Auth { message: Option<String> },
    #[error("qwen rate limited")]
    RateLimited {
        retry_after: Option<Duration>,
        message: Option<String>,
    },
    #[error("qwen invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("qwen server error (status {status})")]
    ServerError {
        status: u16,
        message: Option<String>,
    },
    #[error("qwen response malformed: {reason}")]
    MalformedResponse { reason: String },
}

impl super::shared::FailureFromStatus for QwenCompletionFailure {
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

pub type QwenResult = Result<QwenCompletionSuccess, QwenCompletionFailure>;

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a QwenModel,
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

pub async fn qwen_get_completion(
    client: &QwenClient,
    command: &QwenCompletionCommand,
) -> QwenResult {
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
        .map_err(QwenCompletionFailure::Transport)?;

    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response
        .bytes()
        .await
        .map_err(QwenCompletionFailure::Transport)?;

    if status.is_success() {
        let parsed: WireResponse =
            serde_json::from_slice(&bytes).map_err(QwenCompletionFailure::Deserialize)?;
        let first_choice = parsed.choices.into_iter().next().ok_or_else(|| {
            QwenCompletionFailure::MalformedResponse {
                reason: "response contained no choices".to_string(),
            }
        })?;
        Ok(QwenCompletionSuccess {
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

    fn sample_command() -> QwenCompletionCommand {
        QwenCompletionCommand {
            model: QwenModel::QwenPlus,
            system_prompt: Some("You are a helpful assistant.".to_string()),
            messages: vec![QwenMessage {
                role: QwenRole::User,
                content: "Hi".to_string(),
            }],
            max_tokens: Some(64),
            temperature: Some(0.0),
        }
    }

    #[test]
    fn qwen_model_serializes_to_expected_wire_value() {
        assert_eq!(QwenModel::QwenTurbo.as_api_str(), "qwen-turbo");
        assert_eq!(QwenModel::QwenPlus.as_api_str(), "qwen-plus");
        assert_eq!(QwenModel::QwenMax.as_api_str(), "qwen-max");
        assert_eq!(QwenModel::Qwen3_32B.as_api_str(), "qwen3-32b");
        assert_eq!(QwenModel::Qwen3_235BA22B.as_api_str(), "qwen3-235b-a22b");
        assert_eq!(
            QwenModel::Custom("qwen2.5-coder-32b-instruct".to_string()).as_api_str(),
            "qwen2.5-coder-32b-instruct"
        );

        assert_eq!(QwenModel::QwenMax.to_string(), "qwen-max");

        let json = serde_json::to_string(&QwenModel::QwenTurbo).expect("serialize");
        assert_eq!(json, r#""qwen-turbo""#);
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

        let client = QwenClient::new_with_base_url("test-key".to_string(), server.url());
        let success = qwen_get_completion(&client, &sample_command())
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

        let client = QwenClient::new_with_base_url("bad-key".to_string(), server.url());
        let err = qwen_get_completion(&client, &sample_command())
            .await
            .expect_err("expected auth failure");
        match err {
            QwenCompletionFailure::Auth { message } => {
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

        let client = QwenClient::new_with_base_url("test-key".to_string(), server.url());
        let err = qwen_get_completion(&client, &sample_command())
            .await
            .expect_err("expected auth failure");
        assert!(matches!(err, QwenCompletionFailure::Auth { .. }));
    }

    #[tokio::test]
    async fn rate_limit_carries_retry_after() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(429)
            .with_header("retry-after", "20")
            .with_body(r#"{"error":{"message":"Throttled"}}"#)
            .create_async()
            .await;

        let client = QwenClient::new_with_base_url("test-key".to_string(), server.url());
        let err = qwen_get_completion(&client, &sample_command())
            .await
            .expect_err("expected rate limit");
        match err {
            QwenCompletionFailure::RateLimited {
                retry_after,
                message,
            } => {
                assert_eq!(retry_after, Some(Duration::from_secs(20)));
                assert_eq!(message.as_deref(), Some("Throttled"));
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

        let client = QwenClient::new_with_base_url("test-key".to_string(), server.url());
        let err = qwen_get_completion(&client, &sample_command())
            .await
            .expect_err("expected invalid request");
        match err {
            QwenCompletionFailure::InvalidRequest { message } => {
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

        let client = QwenClient::new_with_base_url("test-key".to_string(), server.url());
        let err = qwen_get_completion(&client, &sample_command())
            .await
            .expect_err("expected server error");
        match err {
            QwenCompletionFailure::ServerError { status, message } => {
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

        let client = QwenClient::new_with_base_url("test-key".to_string(), server.url());
        let err = qwen_get_completion(&client, &sample_command())
            .await
            .expect_err("expected deserialize failure");
        assert!(matches!(err, QwenCompletionFailure::Deserialize(_)));
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

        let client = QwenClient::new_with_base_url("test-key".to_string(), server.url());
        let err = qwen_get_completion(&client, &sample_command())
            .await
            .expect_err("empty choices must surface as a typed failure");
        match err {
            QwenCompletionFailure::MalformedResponse { reason } => {
                assert!(reason.contains("no choices"), "unexpected reason: {reason}");
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    #[test]
    fn debug_output_redacts_api_key() {
        let client = QwenClient::new_with_base_url(
            "super-secret-dashscope-key".to_string(),
            "https://example.test",
        );
        let debug = format!("{client:?}");
        assert!(
            !debug.contains("super-secret-dashscope-key"),
            "Debug output must not leak api_key: {debug}"
        );
    }

    #[tokio::test]
    #[ignore = "requires DASHSCOPE_API_KEY; run with `cargo test -- --ignored`"]
    async fn live_qwen_completion_returns_real_response() {
        let client = QwenClient::from_env().expect("DASHSCOPE_API_KEY must be set");
        let command = QwenCompletionCommand {
            model: QwenModel::QwenTurbo,
            system_prompt: Some("Reply with exactly the word 'pong'.".to_string()),
            messages: vec![QwenMessage {
                role: QwenRole::User,
                content: "ping".to_string(),
            }],
            max_tokens: Some(8),
            temperature: Some(0.0),
        };
        let success = qwen_get_completion(&client, &command)
            .await
            .expect("live Qwen request should succeed");
        assert!(!success.content.is_empty(), "content should be non-empty");
        assert!(success.prompt_tokens > 0, "prompt_tokens should be > 0");
        assert!(
            success.completion_tokens > 0,
            "completion_tokens should be > 0"
        );
    }
}
