use std::time::Duration;

use bytes::Bytes;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://api.openai.com";

#[derive(Debug, Clone)]
pub struct OpenAiClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl OpenAiClient {
    pub fn new(api_key: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key,
        }
    }

    pub fn from_env() -> Result<Self, OpenAiClientError> {
        let key = std::env::var("OPENAI_API_KEY").map_err(|_| OpenAiClientError::MissingApiKey)?;
        Ok(Self::new(key))
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
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
    O1,
    O1Mini,
    O3Mini,
    Custom(String),
}

impl OpenAiModel {
    pub fn as_api_str(&self) -> &str {
        match self {
            Self::Gpt4oMini => "gpt-4o-mini",
            Self::Gpt4o => "gpt-4o",
            Self::O1 => "o1",
            Self::O1Mini => "o1-mini",
            Self::O3Mini => "o3-mini",
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

#[derive(Deserialize)]
struct WireErrorBody {
    error: WireErrorPayload,
}

#[derive(Deserialize)]
struct WireErrorPayload {
    message: String,
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
        .bearer_auth(&client.api_key)
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
        Err(map_failure_from_status(status.as_u16(), &headers, &bytes))
    }
}

fn map_failure_from_status(
    status: u16,
    headers: &HeaderMap,
    bytes: &Bytes,
) -> OpenAiCompletionFailure {
    let message = serde_json::from_slice::<WireErrorBody>(bytes)
        .ok()
        .map(|b| b.error.message);

    match status {
        401 | 403 => OpenAiCompletionFailure::Auth { message },
        429 => OpenAiCompletionFailure::RateLimited {
            retry_after: parse_retry_after(headers),
            message,
        },
        400..=499 => OpenAiCompletionFailure::InvalidRequest {
            message: message.unwrap_or_else(|| format!("HTTP {status}")),
        },
        500..=599 => OpenAiCompletionFailure::ServerError { status, message },
        // 1xx/3xx or any other non-success status surfaces as a typed ServerError so the
        // caller still gets a concrete status code instead of a silent collapse.
        _ => OpenAiCompletionFailure::ServerError { status, message },
    }
}

fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let raw = headers.get(RETRY_AFTER)?.to_str().ok()?;
    raw.parse::<u64>().ok().map(Duration::from_secs)
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

        let client = OpenAiClient::new("test-key".to_string()).with_base_url(server.url());
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

        let client = OpenAiClient::new("bad-key".to_string()).with_base_url(server.url());
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

        let client = OpenAiClient::new("test-key".to_string()).with_base_url(server.url());
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

        let client = OpenAiClient::new("test-key".to_string()).with_base_url(server.url());
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

        let client = OpenAiClient::new("test-key".to_string()).with_base_url(server.url());
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

        let client = OpenAiClient::new("test-key".to_string()).with_base_url(server.url());
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
    fn openai_model_serializes_to_expected_wire_value() {
        assert_eq!(OpenAiModel::Gpt4oMini.as_api_str(), "gpt-4o-mini");
        assert_eq!(OpenAiModel::Gpt4o.as_api_str(), "gpt-4o");
        assert_eq!(OpenAiModel::O1.as_api_str(), "o1");
        assert_eq!(OpenAiModel::O1Mini.as_api_str(), "o1-mini");
        assert_eq!(OpenAiModel::O3Mini.as_api_str(), "o3-mini");
        assert_eq!(
            OpenAiModel::Custom("gpt-4-turbo".to_string()).as_api_str(),
            "gpt-4-turbo"
        );

        assert_eq!(OpenAiModel::Gpt4o.to_string(), "gpt-4o");

        let json = serde_json::to_string(&OpenAiModel::Gpt4oMini).expect("serialize");
        assert_eq!(json, r#""gpt-4o-mini""#);
        let json_custom =
            serde_json::to_string(&OpenAiModel::Custom("o3".to_string())).expect("serialize");
        assert_eq!(json_custom, r#""o3""#);
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

        let client = OpenAiClient::new("test-key".to_string()).with_base_url(server.url());
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

        let client = OpenAiClient::new("test-key".to_string()).with_base_url(server.url());
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
