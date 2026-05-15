use std::time::Duration;

use bytes::Bytes;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";

#[derive(Debug, Clone)]
pub struct GeminiClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl GeminiClient {
    pub fn new(api_key: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key,
        }
    }

    pub fn from_env() -> Result<Self, GeminiClientError> {
        let key = std::env::var("GEMINI_API_KEY").map_err(|_| GeminiClientError::MissingApiKey)?;
        Ok(Self::new(key))
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GeminiClientError {
    #[error("GEMINI_API_KEY is not set in the environment")]
    MissingApiKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeminiRole {
    User,
    Model,
}

impl GeminiRole {
    fn as_wire_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Model => "model",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeminiModel {
    Gemini20Flash,
    Gemini15Pro,
    Gemini15Flash,
    Custom(String),
}

impl GeminiModel {
    pub fn as_api_str(&self) -> &str {
        match self {
            Self::Gemini20Flash => "gemini-2.0-flash",
            Self::Gemini15Pro => "gemini-1.5-pro",
            Self::Gemini15Flash => "gemini-1.5-flash",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for GeminiModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_api_str())
    }
}

impl Serialize for GeminiModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_api_str())
    }
}

#[derive(Debug, Clone)]
pub struct GeminiMessage {
    pub role: GeminiRole,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct GeminiCompletionCommand {
    pub model: GeminiModel,
    pub system_prompt: Option<String>,
    pub messages: Vec<GeminiMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

impl Default for GeminiCompletionCommand {
    fn default() -> Self {
        Self {
            model: GeminiModel::Gemini15Flash,
            system_prompt: None,
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
        }
    }
}

#[derive(Debug)]
pub struct GeminiCompletionSuccess {
    pub content: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum GeminiCompletionFailure {
    #[error("gemini transport failure: {0}")]
    Transport(#[source] reqwest::Error),
    #[error("gemini response deserialization failed: {0}")]
    Deserialize(#[source] serde_json::Error),
    #[error("gemini authentication failed")]
    Auth { message: Option<String> },
    #[error("gemini rate limited")]
    RateLimited {
        retry_after: Option<Duration>,
        message: Option<String>,
    },
    #[error("gemini invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("gemini server error (status {status})")]
    ServerError {
        status: u16,
        message: Option<String>,
    },
    #[error("gemini response malformed: {reason}")]
    MalformedResponse { reason: String },
}

pub type GeminiResult = Result<GeminiCompletionSuccess, GeminiCompletionFailure>;

#[derive(Serialize)]
struct WireRequest<'a> {
    contents: Vec<WireContentReq<'a>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "systemInstruction")]
    system_instruction: Option<WireSystemInstruction<'a>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "generationConfig")]
    generation_config: Option<WireGenerationConfig>,
}

#[derive(Serialize)]
struct WireContentReq<'a> {
    role: &'a str,
    parts: Vec<WirePartReq<'a>>,
}

#[derive(Serialize)]
struct WirePartReq<'a> {
    text: &'a str,
}

#[derive(Serialize)]
struct WireSystemInstruction<'a> {
    parts: Vec<WirePartReq<'a>>,
}

#[derive(Serialize)]
struct WireGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxOutputTokens")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Deserialize)]
struct WireResponse {
    #[serde(default)]
    candidates: Vec<WireCandidate>,
    #[serde(rename = "usageMetadata", default)]
    usage_metadata: WireUsageMetadata,
}

#[derive(Deserialize)]
struct WireCandidate {
    content: WireContentResp,
}

#[derive(Deserialize)]
struct WireContentResp {
    #[serde(default)]
    parts: Vec<WirePartResp>,
}

#[derive(Deserialize)]
struct WirePartResp {
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize, Default)]
struct WireUsageMetadata {
    #[serde(rename = "promptTokenCount", default)]
    prompt_token_count: u64,
    #[serde(rename = "candidatesTokenCount", default)]
    candidates_token_count: u64,
}

#[derive(Deserialize)]
struct WireErrorBody {
    error: WireErrorPayload,
}

#[derive(Deserialize)]
struct WireErrorPayload {
    message: String,
}

pub async fn gemini_get_completion(
    client: &GeminiClient,
    command: &GeminiCompletionCommand,
) -> GeminiResult {
    let contents: Vec<WireContentReq<'_>> = command
        .messages
        .iter()
        .map(|m| WireContentReq {
            role: m.role.as_wire_str(),
            parts: vec![WirePartReq { text: &m.content }],
        })
        .collect();

    let system_instruction = command
        .system_prompt
        .as_deref()
        .map(|s| WireSystemInstruction {
            parts: vec![WirePartReq { text: s }],
        });

    let generation_config = if command.max_tokens.is_some() || command.temperature.is_some() {
        Some(WireGenerationConfig {
            max_output_tokens: command.max_tokens,
            temperature: command.temperature,
        })
    } else {
        None
    };

    let body = WireRequest {
        contents,
        system_instruction,
        generation_config,
    };

    let url = format!(
        "{}/v1beta/models/{}:generateContent",
        client.base_url,
        command.model.as_api_str()
    );

    let response = client
        .http
        .post(&url)
        .header("x-goog-api-key", &client.api_key)
        .json(&body)
        .send()
        .await
        .map_err(GeminiCompletionFailure::Transport)?;

    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response
        .bytes()
        .await
        .map_err(GeminiCompletionFailure::Transport)?;

    if status.is_success() {
        let parsed: WireResponse =
            serde_json::from_slice(&bytes).map_err(GeminiCompletionFailure::Deserialize)?;

        let candidate = parsed.candidates.into_iter().next().ok_or_else(|| {
            GeminiCompletionFailure::MalformedResponse {
                reason: "response contained no candidates".to_string(),
            }
        })?;

        let text: String = candidate
            .content
            .parts
            .into_iter()
            .filter_map(|p| p.text)
            .collect::<Vec<_>>()
            .concat();

        if text.is_empty() {
            return Err(GeminiCompletionFailure::MalformedResponse {
                reason: "candidate contained no text parts".to_string(),
            });
        }

        Ok(GeminiCompletionSuccess {
            content: text,
            input_tokens: parsed.usage_metadata.prompt_token_count,
            output_tokens: parsed.usage_metadata.candidates_token_count,
        })
    } else {
        Err(map_failure_from_status(status.as_u16(), &headers, &bytes))
    }
}

fn map_failure_from_status(
    status: u16,
    headers: &HeaderMap,
    bytes: &Bytes,
) -> GeminiCompletionFailure {
    let message = serde_json::from_slice::<WireErrorBody>(bytes)
        .ok()
        .map(|b| b.error.message);

    match status {
        401 | 403 => GeminiCompletionFailure::Auth { message },
        429 => GeminiCompletionFailure::RateLimited {
            retry_after: parse_retry_after(headers),
            message,
        },
        400..=499 => GeminiCompletionFailure::InvalidRequest {
            message: message.unwrap_or_else(|| format!("HTTP {status}")),
        },
        500..=599 => GeminiCompletionFailure::ServerError { status, message },
        // 1xx/3xx or any other non-success status surfaces as a typed ServerError so the
        // caller still gets a concrete status code instead of a silent collapse.
        _ => GeminiCompletionFailure::ServerError { status, message },
    }
}

fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let raw = headers.get(RETRY_AFTER)?.to_str().ok()?;
    raw.parse::<u64>().ok().map(Duration::from_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_command() -> GeminiCompletionCommand {
        GeminiCompletionCommand {
            model: GeminiModel::Gemini15Flash,
            system_prompt: Some("You are helpful.".to_string()),
            messages: vec![GeminiMessage {
                role: GeminiRole::User,
                content: "Hi".to_string(),
            }],
            max_tokens: Some(64),
            temperature: Some(0.0),
        }
    }

    #[test]
    fn gemini_model_serializes_to_expected_wire_value() {
        assert_eq!(GeminiModel::Gemini20Flash.as_api_str(), "gemini-2.0-flash");
        assert_eq!(GeminiModel::Gemini15Pro.as_api_str(), "gemini-1.5-pro");
        assert_eq!(GeminiModel::Gemini15Flash.as_api_str(), "gemini-1.5-flash");
        assert_eq!(
            GeminiModel::Custom("gemini-1.5-flash-8b".to_string()).as_api_str(),
            "gemini-1.5-flash-8b"
        );

        assert_eq!(GeminiModel::Gemini15Pro.to_string(), "gemini-1.5-pro");

        let json = serde_json::to_string(&GeminiModel::Gemini15Flash).expect("serialize");
        assert_eq!(json, r#""gemini-1.5-flash""#);
    }

    #[tokio::test]
    async fn success_returns_content_and_tokens() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1beta/models/gemini-1.5-flash:generateContent")
            .match_header("x-goog-api-key", "test-key")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "candidates": [{
                        "content": {
                            "role": "model",
                            "parts": [{"text": "Hello from Gemini!"}]
                        }
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 9,
                        "candidatesTokenCount": 6
                    }
                }"#,
            )
            .create_async()
            .await;

        let client = GeminiClient::new("test-key".to_string()).with_base_url(server.url());
        let success = gemini_get_completion(&client, &sample_command())
            .await
            .expect("expected success");
        assert_eq!(success.content, "Hello from Gemini!");
        assert_eq!(success.input_tokens, 9);
        assert_eq!(success.output_tokens, 6);
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn success_concatenates_multiple_text_parts() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1beta/models/gemini-1.5-flash:generateContent")
            .with_status(200)
            .with_body(
                r#"{
                    "candidates": [{
                        "content": {
                            "parts": [
                                {"text": "Part 1. "},
                                {"text": "Part 2."}
                            ]
                        }
                    }],
                    "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3}
                }"#,
            )
            .create_async()
            .await;

        let client = GeminiClient::new("test-key".to_string()).with_base_url(server.url());
        let success = gemini_get_completion(&client, &sample_command())
            .await
            .expect("expected success");
        assert_eq!(success.content, "Part 1. Part 2.");
    }

    #[tokio::test]
    async fn auth_failure_on_401() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1beta/models/gemini-1.5-flash:generateContent")
            .with_status(401)
            .with_body(r#"{"error":{"code":401,"message":"API key invalid","status":"UNAUTHENTICATED"}}"#)
            .create_async()
            .await;

        let client = GeminiClient::new("bad-key".to_string()).with_base_url(server.url());
        let err = gemini_get_completion(&client, &sample_command())
            .await
            .expect_err("expected auth failure");
        match err {
            GeminiCompletionFailure::Auth { message } => {
                assert_eq!(message.as_deref(), Some("API key invalid"));
            }
            other => panic!("expected Auth failure, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn rate_limit_carries_retry_after() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1beta/models/gemini-1.5-flash:generateContent")
            .with_status(429)
            .with_header("retry-after", "60")
            .with_body(r#"{"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED"}}"#)
            .create_async()
            .await;

        let client = GeminiClient::new("test-key".to_string()).with_base_url(server.url());
        let err = gemini_get_completion(&client, &sample_command())
            .await
            .expect_err("expected rate limit");
        match err {
            GeminiCompletionFailure::RateLimited {
                retry_after,
                message,
            } => {
                assert_eq!(retry_after, Some(Duration::from_secs(60)));
                assert_eq!(message.as_deref(), Some("Quota exceeded"));
            }
            other => panic!("expected RateLimited, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn invalid_request_on_400() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1beta/models/gemini-1.5-flash:generateContent")
            .with_status(400)
            .with_body(r#"{"error":{"code":400,"message":"Invalid argument","status":"INVALID_ARGUMENT"}}"#)
            .create_async()
            .await;

        let client = GeminiClient::new("test-key".to_string()).with_base_url(server.url());
        let err = gemini_get_completion(&client, &sample_command())
            .await
            .expect_err("expected invalid request");
        match err {
            GeminiCompletionFailure::InvalidRequest { message } => {
                assert_eq!(message, "Invalid argument");
            }
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn server_error_on_503() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1beta/models/gemini-1.5-flash:generateContent")
            .with_status(503)
            .with_body(r#"{"error":{"code":503,"message":"backend overloaded","status":"UNAVAILABLE"}}"#)
            .create_async()
            .await;

        let client = GeminiClient::new("test-key".to_string()).with_base_url(server.url());
        let err = gemini_get_completion(&client, &sample_command())
            .await
            .expect_err("expected server error");
        match err {
            GeminiCompletionFailure::ServerError { status, message } => {
                assert_eq!(status, 503);
                assert_eq!(message.as_deref(), Some("backend overloaded"));
            }
            other => panic!("expected ServerError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn malformed_json_maps_to_deserialize() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1beta/models/gemini-1.5-flash:generateContent")
            .with_status(200)
            .with_body("not valid json")
            .create_async()
            .await;

        let client = GeminiClient::new("test-key".to_string()).with_base_url(server.url());
        let err = gemini_get_completion(&client, &sample_command())
            .await
            .expect_err("expected deserialize failure");
        assert!(matches!(err, GeminiCompletionFailure::Deserialize(_)));
    }

    #[tokio::test]
    async fn empty_candidates_maps_to_malformed_response() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1beta/models/gemini-1.5-flash:generateContent")
            .with_status(200)
            .with_body(
                r#"{"candidates":[],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":0}}"#,
            )
            .create_async()
            .await;

        let client = GeminiClient::new("test-key".to_string()).with_base_url(server.url());
        let err = gemini_get_completion(&client, &sample_command())
            .await
            .expect_err("expected malformed");
        match err {
            GeminiCompletionFailure::MalformedResponse { reason } => {
                assert!(
                    reason.contains("no candidates"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_text_parts_maps_to_malformed_response() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1beta/models/gemini-1.5-flash:generateContent")
            .with_status(200)
            .with_body(
                r#"{
                    "candidates": [{"content": {"parts": []}}],
                    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 0}
                }"#,
            )
            .create_async()
            .await;

        let client = GeminiClient::new("test-key".to_string()).with_base_url(server.url());
        let err = gemini_get_completion(&client, &sample_command())
            .await
            .expect_err("expected malformed");
        match err {
            GeminiCompletionFailure::MalformedResponse { reason } => {
                assert!(
                    reason.contains("no text parts"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY; run with `cargo test -- --ignored`"]
    async fn live_gemini_completion_returns_real_response() {
        let client = GeminiClient::from_env().expect("GEMINI_API_KEY must be set");
        let command = GeminiCompletionCommand {
            model: GeminiModel::Gemini15Flash,
            system_prompt: Some("Reply with exactly the word 'pong'.".to_string()),
            messages: vec![GeminiMessage {
                role: GeminiRole::User,
                content: "ping".to_string(),
            }],
            max_tokens: Some(16),
            temperature: Some(0.0),
        };
        let success = gemini_get_completion(&client, &command)
            .await
            .expect("live Gemini request should succeed");
        assert!(!success.content.is_empty(), "content should be non-empty");
        assert!(success.input_tokens > 0, "input_tokens should be > 0");
        assert!(success.output_tokens > 0, "output_tokens should be > 0");
    }
}
