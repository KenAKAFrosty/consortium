use std::time::Duration;

use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";

#[derive(Debug, Clone)]
pub struct DeepseekClient {
    http: reqwest::Client,
    base_url: String,
    api_key: SecretString,
}

impl DeepseekClient {
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

    pub fn from_env() -> Result<Self, DeepseekClientError> {
        Self::from_env_with_base_url(DEFAULT_BASE_URL)
    }

    pub fn from_env_with_base_url(
        base_url: impl Into<String>,
    ) -> Result<Self, DeepseekClientError> {
        let key =
            std::env::var("DEEPSEEK_API_KEY").map_err(|_| DeepseekClientError::MissingApiKey)?;
        Ok(Self::new_with_base_url(key, base_url))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DeepseekClientError {
    #[error("DEEPSEEK_API_KEY is not set in the environment")]
    MissingApiKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepseekRole {
    User,
    Assistant,
}

impl DeepseekRole {
    fn as_wire_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeepseekModel {
    /// DeepSeek-V3 chat model. Maps to wire id `deepseek-chat`.
    Chat,
    /// DeepSeek-R1 reasoning model. Maps to wire id `deepseek-reasoner`.
    Reasoner,
    /// Escape hatch for OpenAI-compatible-chat-completions models that have not been
    /// added to this enum yet.
    Custom(String),
}

impl DeepseekModel {
    pub fn as_api_str(&self) -> &str {
        match self {
            Self::Chat => "deepseek-chat",
            Self::Reasoner => "deepseek-reasoner",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for DeepseekModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_api_str())
    }
}

impl Serialize for DeepseekModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_api_str())
    }
}

#[derive(Debug, Clone)]
pub struct DeepseekMessage {
    pub role: DeepseekRole,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct DeepseekCompletionCommand {
    pub model: DeepseekModel,
    pub system_prompt: Option<String>,
    pub messages: Vec<DeepseekMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

impl Default for DeepseekCompletionCommand {
    fn default() -> Self {
        Self {
            model: DeepseekModel::Chat,
            system_prompt: None,
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
        }
    }
}

#[derive(Debug)]
pub struct DeepseekCompletionSuccess {
    pub content: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum DeepseekCompletionFailure {
    #[error("deepseek transport failure: {0}")]
    Transport(#[source] reqwest::Error),
    #[error("deepseek response deserialization failed: {0}")]
    Deserialize(#[source] serde_json::Error),
    #[error("deepseek authentication failed")]
    Auth { message: Option<String> },
    #[error("deepseek rate limited")]
    RateLimited {
        retry_after: Option<Duration>,
        message: Option<String>,
    },
    #[error("deepseek invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("deepseek server error (status {status})")]
    ServerError {
        status: u16,
        message: Option<String>,
    },
    #[error("deepseek response malformed: {reason}")]
    MalformedResponse { reason: String },
}

impl super::shared::FailureFromStatus for DeepseekCompletionFailure {
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

pub type DeepseekResult = Result<DeepseekCompletionSuccess, DeepseekCompletionFailure>;

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a DeepseekModel,
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

pub async fn deepseek_get_completion(
    client: &DeepseekClient,
    command: &DeepseekCompletionCommand,
) -> DeepseekResult {
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
        .map_err(DeepseekCompletionFailure::Transport)?;

    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response
        .bytes()
        .await
        .map_err(DeepseekCompletionFailure::Transport)?;

    if status.is_success() {
        let parsed: WireResponse =
            serde_json::from_slice(&bytes).map_err(DeepseekCompletionFailure::Deserialize)?;
        let first_choice = parsed.choices.into_iter().next().ok_or_else(|| {
            DeepseekCompletionFailure::MalformedResponse {
                reason: "response contained no choices".to_string(),
            }
        })?;
        Ok(DeepseekCompletionSuccess {
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

    fn sample_command() -> DeepseekCompletionCommand {
        DeepseekCompletionCommand {
            model: DeepseekModel::Chat,
            system_prompt: Some("You are a helpful assistant.".to_string()),
            messages: vec![DeepseekMessage {
                role: DeepseekRole::User,
                content: "Hi".to_string(),
            }],
            max_tokens: Some(64),
            temperature: Some(0.0),
        }
    }

    #[test]
    fn deepseek_model_serializes_to_expected_wire_value() {
        assert_eq!(DeepseekModel::Chat.as_api_str(), "deepseek-chat");
        assert_eq!(DeepseekModel::Reasoner.as_api_str(), "deepseek-reasoner");
        assert_eq!(
            DeepseekModel::Custom("deepseek-coder".to_string()).as_api_str(),
            "deepseek-coder"
        );

        assert_eq!(DeepseekModel::Chat.to_string(), "deepseek-chat");

        let json = serde_json::to_string(&DeepseekModel::Reasoner).expect("serialize");
        assert_eq!(json, r#""deepseek-reasoner""#);
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

        let client = DeepseekClient::new_with_base_url("test-key".to_string(), server.url());
        let success = deepseek_get_completion(&client, &sample_command())
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

        let client = DeepseekClient::new_with_base_url("bad-key".to_string(), server.url());
        let err = deepseek_get_completion(&client, &sample_command())
            .await
            .expect_err("expected auth failure");
        match err {
            DeepseekCompletionFailure::Auth { message } => {
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

        let client = DeepseekClient::new_with_base_url("test-key".to_string(), server.url());
        let err = deepseek_get_completion(&client, &sample_command())
            .await
            .expect_err("expected auth failure");
        assert!(matches!(err, DeepseekCompletionFailure::Auth { .. }));
    }

    #[tokio::test]
    async fn rate_limit_carries_retry_after() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(429)
            .with_header("retry-after", "30")
            .with_body(r#"{"error":{"message":"Too many requests"}}"#)
            .create_async()
            .await;

        let client = DeepseekClient::new_with_base_url("test-key".to_string(), server.url());
        let err = deepseek_get_completion(&client, &sample_command())
            .await
            .expect_err("expected rate limit");
        match err {
            DeepseekCompletionFailure::RateLimited {
                retry_after,
                message,
            } => {
                assert_eq!(retry_after, Some(Duration::from_secs(30)));
                assert_eq!(message.as_deref(), Some("Too many requests"));
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

        let client = DeepseekClient::new_with_base_url("test-key".to_string(), server.url());
        let err = deepseek_get_completion(&client, &sample_command())
            .await
            .expect_err("expected invalid request");
        match err {
            DeepseekCompletionFailure::InvalidRequest { message } => {
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

        let client = DeepseekClient::new_with_base_url("test-key".to_string(), server.url());
        let err = deepseek_get_completion(&client, &sample_command())
            .await
            .expect_err("expected server error");
        match err {
            DeepseekCompletionFailure::ServerError { status, message } => {
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

        let client = DeepseekClient::new_with_base_url("test-key".to_string(), server.url());
        let err = deepseek_get_completion(&client, &sample_command())
            .await
            .expect_err("expected deserialize failure");
        assert!(matches!(err, DeepseekCompletionFailure::Deserialize(_)));
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

        let client = DeepseekClient::new_with_base_url("test-key".to_string(), server.url());
        let err = deepseek_get_completion(&client, &sample_command())
            .await
            .expect_err("empty choices must surface as a typed failure");
        match err {
            DeepseekCompletionFailure::MalformedResponse { reason } => {
                assert!(reason.contains("no choices"), "unexpected reason: {reason}");
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    #[test]
    fn debug_output_redacts_api_key() {
        let client = DeepseekClient::new_with_base_url(
            "super-secret-deepseek-key".to_string(),
            "https://example.test",
        );
        let debug = format!("{client:?}");
        assert!(
            !debug.contains("super-secret-deepseek-key"),
            "Debug output must not leak api_key: {debug}"
        );
    }

    #[tokio::test]
    #[ignore = "requires DEEPSEEK_API_KEY; run with `cargo test -- --ignored`"]
    async fn live_deepseek_completion_returns_real_response() {
        let client = DeepseekClient::from_env().expect("DEEPSEEK_API_KEY must be set");
        let command = DeepseekCompletionCommand {
            model: DeepseekModel::Chat,
            system_prompt: Some("Reply with exactly the word 'pong'.".to_string()),
            messages: vec![DeepseekMessage {
                role: DeepseekRole::User,
                content: "ping".to_string(),
            }],
            max_tokens: Some(8),
            temperature: Some(0.0),
        };
        let success = deepseek_get_completion(&client, &command)
            .await
            .expect("live Deepseek request should succeed");
        assert!(!success.content.is_empty(), "content should be non-empty");
        assert!(success.prompt_tokens > 0, "prompt_tokens should be > 0");
        assert!(
            success.completion_tokens > 0,
            "completion_tokens should be > 0"
        );
    }
}
