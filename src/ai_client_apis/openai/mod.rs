use std::time::Duration;

use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://api.openai.com";

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    http: reqwest::Client,
    base_url: String,
    api_key: SecretString,
}

impl OpenAiClient {
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

    pub fn from_env() -> Result<Self, OpenAiClientError> {
        Self::from_env_with_base_url(DEFAULT_BASE_URL)
    }

    pub fn from_env_with_base_url(
        base_url: impl Into<String>,
    ) -> Result<Self, OpenAiClientError> {
        let key = std::env::var("OPENAI_API_KEY").map_err(|_| OpenAiClientError::MissingApiKey)?;
        Ok(Self::new_with_base_url(key, base_url))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OpenAiClientError {
    #[error("OPENAI_API_KEY is not set in the environment")]
    MissingApiKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenAiRole {
    User,
    Assistant,
}

impl OpenAiRole {
    fn as_wire_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenAiModel {
    Gpt4oMini,
    Gpt4o,
    Gpt4Turbo,
    /// Escape hatch for chat-completions-compatible models that have not been added
    /// to this enum yet. Not appropriate for reasoning models (o1, o3, etc.) — those
    /// require `max_completion_tokens` instead of `max_tokens` and `developer` role
    /// in place of `system`, neither of which this seed emits.
    Custom(String),
}

impl OpenAiModel {
    pub fn as_api_str(&self) -> &str {
        match self {
            Self::Gpt4oMini => "gpt-4o-mini",
            Self::Gpt4o => "gpt-4o",
            Self::Gpt4Turbo => "gpt-4-turbo",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for OpenAiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_api_str())
    }
}

impl Serialize for OpenAiModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_api_str())
    }
}

#[derive(Debug, Clone)]
pub struct OpenAiMessage {
    pub role: OpenAiRole,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct OpenAiCompletionCommand {
    pub model: OpenAiModel,
    pub system_prompt: Option<String>,
    pub messages: Vec<OpenAiMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

impl Default for OpenAiCompletionCommand {
    fn default() -> Self {
        Self {
            model: OpenAiModel::Gpt4oMini,
            system_prompt: None,
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
        }
    }
}

#[derive(Debug)]
pub struct OpenAiCompletionSuccess {
    pub content: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum OpenAiCompletionFailure {
    #[error("openai transport failure: {0}")]
    Transport(#[source] reqwest::Error),
    #[error("openai response deserialization failed: {0}")]
    Deserialize(#[source] serde_json::Error),
    #[error("openai authentication failed")]
    Auth { message: Option<String> },
    #[error("openai rate limited")]
    RateLimited {
        retry_after: Option<Duration>,
        message: Option<String>,
    },
    #[error("openai invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("openai server error (status {status})")]
    ServerError {
        status: u16,
        message: Option<String>,
    },
    #[error("openai response malformed: {reason}")]
    MalformedResponse { reason: String },
}

impl super::shared::FailureFromStatus for OpenAiCompletionFailure {
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

pub type OpenAiResult = Result<OpenAiCompletionSuccess, OpenAiCompletionFailure>;

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a OpenAiModel,
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

pub async fn openai_get_completion(
    client: &OpenAiClient,
    command: &OpenAiCompletionCommand,
) -> OpenAiResult {
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
        .map_err(OpenAiCompletionFailure::Transport)?;

    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response
        .bytes()
        .await
        .map_err(OpenAiCompletionFailure::Transport)?;

    if status.is_success() {
        let parsed: WireResponse =
            serde_json::from_slice(&bytes).map_err(OpenAiCompletionFailure::Deserialize)?;
        let first_choice = parsed.choices.into_iter().next().ok_or_else(|| {
            OpenAiCompletionFailure::MalformedResponse {
                reason: "response contained no choices".to_string(),
            }
        })?;
        Ok(OpenAiCompletionSuccess {
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

    fn sample_command() -> OpenAiCompletionCommand {
        OpenAiCompletionCommand {
            model: OpenAiModel::Gpt4oMini,
            system_prompt: Some("You are a helpful assistant.".to_string()),
            messages: vec![OpenAiMessage {
                role: OpenAiRole::User,
                content: "Hi".to_string(),
            }],
            max_tokens: Some(64),
            temperature: Some(0.0),
        }
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

        let client = OpenAiClient::new_with_base_url("test-key".to_string(), server.url());
        let success = openai_get_completion(&client, &sample_command())
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

        let client = OpenAiClient::new_with_base_url("bad-key".to_string(), server.url());
        let err = openai_get_completion(&client, &sample_command())
            .await
            .expect_err("expected auth failure");
        match err {
            OpenAiCompletionFailure::Auth { message } => {
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

        let client = OpenAiClient::new_with_base_url("test-key".to_string(), server.url());
        let err = openai_get_completion(&client, &sample_command())
            .await
            .expect_err("expected auth failure");
        assert!(matches!(err, OpenAiCompletionFailure::Auth { .. }));
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

        let client = OpenAiClient::new_with_base_url("test-key".to_string(), server.url());
        let err = openai_get_completion(&client, &sample_command())
            .await
            .expect_err("expected rate limit");
        match err {
            OpenAiCompletionFailure::RateLimited { retry_after, .. } => {
                assert_eq!(retry_after, Some(Duration::from_secs(30)));
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

        let client = OpenAiClient::new_with_base_url("test-key".to_string(), server.url());
        let err = openai_get_completion(&client, &sample_command())
            .await
            .expect_err("expected invalid request");
        match err {
            OpenAiCompletionFailure::InvalidRequest { message } => {
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

        let client = OpenAiClient::new_with_base_url("test-key".to_string(), server.url());
        let err = openai_get_completion(&client, &sample_command())
            .await
            .expect_err("expected server error");
        match err {
            OpenAiCompletionFailure::ServerError { status, message } => {
                assert_eq!(status, 503);
                assert_eq!(message.as_deref(), Some("Service unavailable"));
            }
            other => panic!("expected ServerError, got {other:?}"),
        }
    }

    #[test]
    fn debug_output_redacts_api_key() {
        let client = OpenAiClient::new_with_base_url(
            "super-secret-openai-key".to_string(),
            "https://example.test",
        );
        let debug = format!("{client:?}");
        assert!(
            !debug.contains("super-secret-openai-key"),
            "Debug output must not leak api_key: {debug}"
        );
    }

    #[test]
    fn openai_model_serializes_to_expected_wire_value() {
        assert_eq!(OpenAiModel::Gpt4oMini.as_api_str(), "gpt-4o-mini");
        assert_eq!(OpenAiModel::Gpt4o.as_api_str(), "gpt-4o");
        assert_eq!(OpenAiModel::Gpt4Turbo.as_api_str(), "gpt-4-turbo");
        assert_eq!(
            OpenAiModel::Custom("gpt-3.5-turbo".to_string()).as_api_str(),
            "gpt-3.5-turbo"
        );

        assert_eq!(OpenAiModel::Gpt4o.to_string(), "gpt-4o");

        let json = serde_json::to_string(&OpenAiModel::Gpt4oMini).expect("serialize");
        assert_eq!(json, r#""gpt-4o-mini""#);
        let json_custom = serde_json::to_string(&OpenAiModel::Custom("gpt-4-turbo".to_string()))
            .expect("serialize");
        assert_eq!(json_custom, r#""gpt-4-turbo""#);
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

        let client = OpenAiClient::new_with_base_url("test-key".to_string(), server.url());
        let err = openai_get_completion(&client, &sample_command())
            .await
            .expect_err("empty choices must surface as a typed failure");
        match err {
            OpenAiCompletionFailure::MalformedResponse { reason } => {
                assert!(reason.contains("no choices"), "unexpected reason: {reason}");
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
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

        let client = OpenAiClient::new_with_base_url("test-key".to_string(), server.url());
        let err = openai_get_completion(&client, &sample_command())
            .await
            .expect_err("expected deserialize failure");
        assert!(matches!(err, OpenAiCompletionFailure::Deserialize(_)));
    }

    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY; run with `cargo test -- --ignored`"]
    async fn live_openai_completion_returns_real_response() {
        let client = OpenAiClient::from_env().expect("OPENAI_API_KEY must be set");
        let command = OpenAiCompletionCommand {
            model: OpenAiModel::Gpt4oMini,
            system_prompt: Some("Reply with exactly the word 'pong'.".to_string()),
            messages: vec![OpenAiMessage {
                role: OpenAiRole::User,
                content: "ping".to_string(),
            }],
            max_tokens: Some(8),
            temperature: Some(0.0),
        };
        let success = openai_get_completion(&client, &command)
            .await
            .expect("live OpenAI request should succeed");
        assert!(!success.content.is_empty(), "content should be non-empty");
        assert!(success.prompt_tokens > 0, "prompt_tokens should be > 0");
        assert!(
            success.completion_tokens > 0,
            "completion_tokens should be > 0"
        );
    }
}
