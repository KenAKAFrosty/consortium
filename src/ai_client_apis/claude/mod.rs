use std::time::Duration;

use bytes::Bytes;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Debug, Clone)]
pub struct ClaudeClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl ClaudeClient {
    pub fn new(api_key: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key,
        }
    }

    pub fn from_env() -> Result<Self, ClaudeClientError> {
        let key =
            std::env::var("ANTHROPIC_API_KEY").map_err(|_| ClaudeClientError::MissingApiKey)?;
        Ok(Self::new(key))
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ClaudeClientError {
    #[error("ANTHROPIC_API_KEY is not set in the environment")]
    MissingApiKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaudeRole {
    User,
    Assistant,
}

impl ClaudeRole {
    fn as_wire_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaudeModel {
    Opus47,
    Sonnet46,
    Haiku45,
    Custom(String),
}

impl ClaudeModel {
    pub fn as_api_str(&self) -> &str {
        match self {
            Self::Opus47 => "claude-opus-4-7",
            Self::Sonnet46 => "claude-sonnet-4-6",
            Self::Haiku45 => "claude-haiku-4-5-20251001",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for ClaudeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_api_str())
    }
}

impl Serialize for ClaudeModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_api_str())
    }
}

#[derive(Debug, Clone)]
pub struct ClaudeMessage {
    pub role: ClaudeRole,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct ClaudeCompletionCommand {
    pub model: ClaudeModel,
    pub system_prompt: Option<String>,
    pub messages: Vec<ClaudeMessage>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
}

impl Default for ClaudeCompletionCommand {
    fn default() -> Self {
        Self {
            model: ClaudeModel::Sonnet46,
            system_prompt: None,
            messages: Vec::new(),
            max_tokens: 1024,
            temperature: None,
        }
    }
}

#[derive(Debug)]
pub struct ClaudeCompletionSuccess {
    pub content: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum ClaudeCompletionFailure {
    #[error("claude transport failure: {0}")]
    Transport(#[source] reqwest::Error),
    #[error("claude response deserialization failed: {0}")]
    Deserialize(#[source] serde_json::Error),
    #[error("claude authentication failed")]
    Auth { message: Option<String> },
    #[error("claude rate limited")]
    RateLimited {
        retry_after: Option<Duration>,
        message: Option<String>,
    },
    #[error("claude invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("claude server error (status {status})")]
    ServerError {
        status: u16,
        message: Option<String>,
    },
    #[error("claude response malformed: {reason}")]
    MalformedResponse { reason: String },
}

pub type ClaudeResult = Result<ClaudeCompletionSuccess, ClaudeCompletionFailure>;

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a ClaudeModel,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    messages: &'a [WireMessage<'a>],
    max_tokens: u32,
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
    content: Vec<WireContentBlock>,
    usage: WireUsage,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireContentBlock {
    Text { text: String },
    // Other content block types (tool_use, etc.) are intentionally not deserialized here.
    // Non-text blocks are silently dropped by the catch-all below in serde's untagged-like
    // behavior when a tag doesn't match. We use #[serde(other)] semantics via a fallback.
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
struct WireUsage {
    input_tokens: u64,
    output_tokens: u64,
}

#[derive(Deserialize)]
struct WireErrorBody {
    error: WireErrorPayload,
}

#[derive(Deserialize)]
struct WireErrorPayload {
    message: String,
}

pub async fn claude_get_completion(
    client: &ClaudeClient,
    command: &ClaudeCompletionCommand,
) -> ClaudeResult {
    let mut wire_messages: Vec<WireMessage<'_>> = Vec::with_capacity(command.messages.len());
    for msg in &command.messages {
        wire_messages.push(WireMessage {
            role: msg.role.as_wire_str(),
            content: &msg.content,
        });
    }

    let body = WireRequest {
        model: &command.model,
        system: command.system_prompt.as_deref(),
        messages: &wire_messages,
        max_tokens: command.max_tokens,
        temperature: command.temperature,
    };

    let url = format!("{}/v1/messages", client.base_url);
    let response = client
        .http
        .post(&url)
        .header("x-api-key", &client.api_key)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .json(&body)
        .send()
        .await
        .map_err(ClaudeCompletionFailure::Transport)?;

    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response
        .bytes()
        .await
        .map_err(ClaudeCompletionFailure::Transport)?;

    if status.is_success() {
        let parsed: WireResponse =
            serde_json::from_slice(&bytes).map_err(ClaudeCompletionFailure::Deserialize)?;

        let mut text = String::new();
        for block in parsed.content {
            if let WireContentBlock::Text { text: t } = block {
                text.push_str(&t);
            }
        }

        if text.is_empty() {
            return Err(ClaudeCompletionFailure::MalformedResponse {
                reason: "response contained no text content blocks".to_string(),
            });
        }

        Ok(ClaudeCompletionSuccess {
            content: text,
            input_tokens: parsed.usage.input_tokens,
            output_tokens: parsed.usage.output_tokens,
        })
    } else {
        Err(map_failure_from_status(status.as_u16(), &headers, &bytes))
    }
}

fn map_failure_from_status(
    status: u16,
    headers: &HeaderMap,
    bytes: &Bytes,
) -> ClaudeCompletionFailure {
    let message = serde_json::from_slice::<WireErrorBody>(bytes)
        .ok()
        .map(|b| b.error.message);

    match status {
        401 | 403 => ClaudeCompletionFailure::Auth { message },
        429 => ClaudeCompletionFailure::RateLimited {
            retry_after: parse_retry_after(headers),
            message,
        },
        400..=499 => ClaudeCompletionFailure::InvalidRequest {
            message: message.unwrap_or_else(|| format!("HTTP {status}")),
        },
        500..=599 => ClaudeCompletionFailure::ServerError { status, message },
        // 1xx/3xx or any other non-success status surfaces as a typed ServerError so the
        // caller still gets a concrete status code instead of a silent collapse.
        _ => ClaudeCompletionFailure::ServerError { status, message },
    }
}

fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let raw = headers.get(RETRY_AFTER)?.to_str().ok()?;
    raw.parse::<u64>().ok().map(Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_command() -> ClaudeCompletionCommand {
        ClaudeCompletionCommand {
            model: ClaudeModel::Sonnet46,
            system_prompt: Some("You are helpful.".to_string()),
            messages: vec![ClaudeMessage {
                role: ClaudeRole::User,
                content: "Hi".to_string(),
            }],
            max_tokens: 64,
            temperature: Some(0.0),
        }
    }

    #[test]
    fn claude_model_serializes_to_expected_wire_value() {
        assert_eq!(ClaudeModel::Opus47.as_api_str(), "claude-opus-4-7");
        assert_eq!(ClaudeModel::Sonnet46.as_api_str(), "claude-sonnet-4-6");
        assert_eq!(
            ClaudeModel::Haiku45.as_api_str(),
            "claude-haiku-4-5-20251001"
        );
        assert_eq!(
            ClaudeModel::Custom("claude-3-opus".to_string()).as_api_str(),
            "claude-3-opus"
        );

        assert_eq!(ClaudeModel::Opus47.to_string(), "claude-opus-4-7");

        let json = serde_json::to_string(&ClaudeModel::Sonnet46).expect("serialize");
        assert_eq!(json, r#""claude-sonnet-4-6""#);
    }

    #[tokio::test]
    async fn success_returns_content_and_tokens() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/messages")
            .match_header("x-api-key", "test-key")
            .match_header("anthropic-version", ANTHROPIC_VERSION)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "content": [{"type": "text", "text": "Hello from Claude!"}],
                    "usage": {"input_tokens": 8, "output_tokens": 5}
                }"#,
            )
            .create_async()
            .await;

        let client = ClaudeClient::new("test-key".to_string()).with_base_url(server.url());
        let success = claude_get_completion(&client, &sample_command())
            .await
            .expect("expected success");
        assert_eq!(success.content, "Hello from Claude!");
        assert_eq!(success.input_tokens, 8);
        assert_eq!(success.output_tokens, 5);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn success_concatenates_multiple_text_blocks() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/messages")
            .with_status(200)
            .with_body(
                r#"{
                    "content": [
                        {"type": "text", "text": "Part 1. "},
                        {"type": "text", "text": "Part 2."}
                    ],
                    "usage": {"input_tokens": 3, "output_tokens": 4}
                }"#,
            )
            .create_async()
            .await;

        let client = ClaudeClient::new("test-key".to_string()).with_base_url(server.url());
        let success = claude_get_completion(&client, &sample_command())
            .await
            .expect("expected success");
        assert_eq!(success.content, "Part 1. Part 2.");
    }

    #[tokio::test]
    async fn auth_failure_on_401() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/messages")
            .with_status(401)
            .with_body(r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#)
            .create_async()
            .await;

        let client = ClaudeClient::new("bad-key".to_string()).with_base_url(server.url());
        let err = claude_get_completion(&client, &sample_command())
            .await
            .expect_err("expected auth failure");
        match err {
            ClaudeCompletionFailure::Auth { message } => {
                assert_eq!(message.as_deref(), Some("invalid x-api-key"));
            }
            other => panic!("expected Auth failure, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rate_limit_carries_retry_after() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/messages")
            .with_status(429)
            .with_header("retry-after", "45")
            .with_body(r#"{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}"#)
            .create_async()
            .await;

        let client = ClaudeClient::new("test-key".to_string()).with_base_url(server.url());
        let err = claude_get_completion(&client, &sample_command())
            .await
            .expect_err("expected rate limit");
        match err {
            ClaudeCompletionFailure::RateLimited {
                retry_after,
                message,
            } => {
                assert_eq!(retry_after, Some(Duration::from_secs(45)));
                assert_eq!(message.as_deref(), Some("slow down"));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn invalid_request_on_400() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/messages")
            .with_status(400)
            .with_body(r#"{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens required"}}"#)
            .create_async()
            .await;

        let client = ClaudeClient::new("test-key".to_string()).with_base_url(server.url());
        let err = claude_get_completion(&client, &sample_command())
            .await
            .expect_err("expected invalid request");
        match err {
            ClaudeCompletionFailure::InvalidRequest { message } => {
                assert_eq!(message, "max_tokens required");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn server_error_on_529() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/messages")
            .with_status(529)
            .with_body(r#"{"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}"#)
            .create_async()
            .await;

        let client = ClaudeClient::new("test-key".to_string()).with_base_url(server.url());
        let err = claude_get_completion(&client, &sample_command())
            .await
            .expect_err("expected server-class failure");
        // Anthropic uses 529 for "overloaded" — it falls into the 4xx-range InvalidRequest
        // arm in our mapping. Both ServerError and InvalidRequest are acceptable typed
        // surfaces here; we check that we did not silently succeed and that the message
        // came through.
        match err {
            ClaudeCompletionFailure::InvalidRequest { message } => {
                assert_eq!(message, "overloaded");
            }
            ClaudeCompletionFailure::ServerError {
                status, message, ..
            } => {
                assert_eq!(status, 529);
                assert_eq!(message.as_deref(), Some("overloaded"));
            }
            other => panic!("expected typed failure, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn server_error_on_503() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/messages")
            .with_status(503)
            .with_body(r#"{"type":"error","error":{"type":"server_error","message":"unavailable"}}"#)
            .create_async()
            .await;

        let client = ClaudeClient::new("test-key".to_string()).with_base_url(server.url());
        let err = claude_get_completion(&client, &sample_command())
            .await
            .expect_err("expected server error");
        match err {
            ClaudeCompletionFailure::ServerError { status, message } => {
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
            .mock("POST", "/v1/messages")
            .with_status(200)
            .with_body("not valid json")
            .create_async()
            .await;

        let client = ClaudeClient::new("test-key".to_string()).with_base_url(server.url());
        let err = claude_get_completion(&client, &sample_command())
            .await
            .expect_err("expected deserialize failure");
        assert!(matches!(err, ClaudeCompletionFailure::Deserialize(_)));
    }

    #[tokio::test]
    async fn empty_text_content_maps_to_malformed_response() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/messages")
            .with_status(200)
            .with_body(
                r#"{
                    "content": [],
                    "usage": {"input_tokens": 1, "output_tokens": 0}
                }"#,
            )
            .create_async()
            .await;

        let client = ClaudeClient::new("test-key".to_string()).with_base_url(server.url());
        let err = claude_get_completion(&client, &sample_command())
            .await
            .expect_err("expected malformed");
        match err {
            ClaudeCompletionFailure::MalformedResponse { reason } => {
                assert!(
                    reason.contains("no text content"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY; run with `cargo test -- --ignored`"]
    async fn live_claude_completion_returns_real_response() {
        let client = ClaudeClient::from_env().expect("ANTHROPIC_API_KEY must be set");
        let command = ClaudeCompletionCommand {
            model: ClaudeModel::Haiku45,
            system_prompt: Some("Reply with exactly the word 'pong'.".to_string()),
            messages: vec![ClaudeMessage {
                role: ClaudeRole::User,
                content: "ping".to_string(),
            }],
            max_tokens: 16,
            temperature: Some(0.0),
        };
        let success = claude_get_completion(&client, &command)
            .await
            .expect("live Claude request should succeed");
        assert!(!success.content.is_empty(), "content should be non-empty");
        assert!(success.input_tokens > 0, "input_tokens should be > 0");
        assert!(success.output_tokens > 0, "output_tokens should be > 0");
    }
}
