use std::time::Duration;

use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};

const DEFAULT_BASE_URL: &str = "https://api.moonshot.ai";

#[derive(Debug, Clone)]
pub struct KimiK2Client {
    http: reqwest::Client,
    base_url: String,
    api_key: SecretString,
}

impl KimiK2Client {
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

    pub fn from_env() -> Result<Self, KimiK2ClientError> {
        Self::from_env_with_base_url(DEFAULT_BASE_URL)
    }

    pub fn from_env_with_base_url(
        base_url: impl Into<String>,
    ) -> Result<Self, KimiK2ClientError> {
        let key =
            std::env::var("MOONSHOT_API_KEY").map_err(|_| KimiK2ClientError::MissingApiKey)?;
        Ok(Self::new_with_base_url(key, base_url))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum KimiK2ClientError {
    #[error("MOONSHOT_API_KEY is not set in the environment")]
    MissingApiKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KimiK2Role {
    User,
    Assistant,
}

impl KimiK2Role {
    fn as_wire_str(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KimiK2Model {
    /// Kimi K2 preview snapshot. Maps to wire id `kimi-k2-0905-preview`.
    KimiK20905Preview,
    /// Moonshot v1 8k-context chat model. Maps to wire id `moonshot-v1-8k`.
    MoonshotV1_8k,
    /// Moonshot v1 32k-context chat model. Maps to wire id `moonshot-v1-32k`.
    MoonshotV1_32k,
    /// Moonshot v1 128k-context chat model. Maps to wire id `moonshot-v1-128k`.
    MoonshotV1_128k,
    /// Escape hatch for Moonshot-hosted OpenAI-compatible chat models that have not
    /// been added to this enum yet.
    Custom(String),
}

impl KimiK2Model {
    pub fn as_api_str(&self) -> &str {
        match self {
            Self::KimiK20905Preview => "kimi-k2-0905-preview",
            Self::MoonshotV1_8k => "moonshot-v1-8k",
            Self::MoonshotV1_32k => "moonshot-v1-32k",
            Self::MoonshotV1_128k => "moonshot-v1-128k",
            Self::Custom(s) => s.as_str(),
        }
    }
}

impl std::fmt::Display for KimiK2Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_api_str())
    }
}

impl Serialize for KimiK2Model {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.as_api_str())
    }
}

#[derive(Debug, Clone)]
pub struct KimiK2Message {
    pub role: KimiK2Role,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct KimiK2CompletionCommand {
    pub model: KimiK2Model,
    pub system_prompt: Option<String>,
    pub messages: Vec<KimiK2Message>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

impl Default for KimiK2CompletionCommand {
    fn default() -> Self {
        Self {
            model: KimiK2Model::KimiK20905Preview,
            system_prompt: None,
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
        }
    }
}

#[derive(Debug)]
pub struct KimiK2CompletionSuccess {
    pub content: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum KimiK2CompletionFailure {
    #[error("kimi-k2 transport failure: {0}")]
    Transport(#[source] reqwest::Error),
    #[error("kimi-k2 response deserialization failed: {0}")]
    Deserialize(#[source] serde_json::Error),
    #[error("kimi-k2 authentication failed")]
    Auth { message: Option<String> },
    #[error("kimi-k2 rate limited")]
    RateLimited {
        retry_after: Option<Duration>,
        message: Option<String>,
    },
    #[error("kimi-k2 invalid request: {message}")]
    InvalidRequest { message: String },
    #[error("kimi-k2 server error (status {status})")]
    ServerError {
        status: u16,
        message: Option<String>,
    },
    #[error("kimi-k2 response malformed: {reason}")]
    MalformedResponse { reason: String },
}

impl super::shared::FailureFromStatus for KimiK2CompletionFailure {
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

pub type KimiK2Result = Result<KimiK2CompletionSuccess, KimiK2CompletionFailure>;

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a KimiK2Model,
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

pub async fn kimik2_get_completion(
    client: &KimiK2Client,
    command: &KimiK2CompletionCommand,
) -> KimiK2Result {
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
        .map_err(KimiK2CompletionFailure::Transport)?;

    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response
        .bytes()
        .await
        .map_err(KimiK2CompletionFailure::Transport)?;

    if status.is_success() {
        let parsed: WireResponse =
            serde_json::from_slice(&bytes).map_err(KimiK2CompletionFailure::Deserialize)?;
        let first_choice = parsed.choices.into_iter().next().ok_or_else(|| {
            KimiK2CompletionFailure::MalformedResponse {
                reason: "response contained no choices".to_string(),
            }
        })?;
        Ok(KimiK2CompletionSuccess {
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

    fn sample_command() -> KimiK2CompletionCommand {
        KimiK2CompletionCommand {
            model: KimiK2Model::KimiK20905Preview,
            system_prompt: Some("You are a helpful assistant.".to_string()),
            messages: vec![KimiK2Message {
                role: KimiK2Role::User,
                content: "Hi".to_string(),
            }],
            max_tokens: Some(64),
            temperature: Some(0.0),
        }
    }

    #[test]
    fn kimik2_model_serializes_to_expected_wire_value() {
        assert_eq!(
            KimiK2Model::KimiK20905Preview.as_api_str(),
            "kimi-k2-0905-preview"
        );
        assert_eq!(KimiK2Model::MoonshotV1_8k.as_api_str(), "moonshot-v1-8k");
        assert_eq!(KimiK2Model::MoonshotV1_32k.as_api_str(), "moonshot-v1-32k");
        assert_eq!(
            KimiK2Model::MoonshotV1_128k.as_api_str(),
            "moonshot-v1-128k"
        );
        assert_eq!(
            KimiK2Model::Custom("kimi-k2-turbo".to_string()).as_api_str(),
            "kimi-k2-turbo"
        );

        assert_eq!(KimiK2Model::MoonshotV1_8k.to_string(), "moonshot-v1-8k");

        let json = serde_json::to_string(&KimiK2Model::KimiK20905Preview).expect("serialize");
        assert_eq!(json, r#""kimi-k2-0905-preview""#);
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

        let client = KimiK2Client::new_with_base_url("test-key".to_string(), server.url());
        let success = kimik2_get_completion(&client, &sample_command())
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

        let client = KimiK2Client::new_with_base_url("bad-key".to_string(), server.url());
        let err = kimik2_get_completion(&client, &sample_command())
            .await
            .expect_err("expected auth failure");
        match err {
            KimiK2CompletionFailure::Auth { message } => {
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

        let client = KimiK2Client::new_with_base_url("test-key".to_string(), server.url());
        let err = kimik2_get_completion(&client, &sample_command())
            .await
            .expect_err("expected auth failure");
        assert!(matches!(err, KimiK2CompletionFailure::Auth { .. }));
    }

    #[tokio::test]
    async fn rate_limit_carries_retry_after() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(429)
            .with_header("retry-after", "45")
            .with_body(r#"{"error":{"message":"Too many requests"}}"#)
            .create_async()
            .await;

        let client = KimiK2Client::new_with_base_url("test-key".to_string(), server.url());
        let err = kimik2_get_completion(&client, &sample_command())
            .await
            .expect_err("expected rate limit");
        match err {
            KimiK2CompletionFailure::RateLimited {
                retry_after,
                message,
            } => {
                assert_eq!(retry_after, Some(Duration::from_secs(45)));
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

        let client = KimiK2Client::new_with_base_url("test-key".to_string(), server.url());
        let err = kimik2_get_completion(&client, &sample_command())
            .await
            .expect_err("expected invalid request");
        match err {
            KimiK2CompletionFailure::InvalidRequest { message } => {
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

        let client = KimiK2Client::new_with_base_url("test-key".to_string(), server.url());
        let err = kimik2_get_completion(&client, &sample_command())
            .await
            .expect_err("expected server error");
        match err {
            KimiK2CompletionFailure::ServerError { status, message } => {
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

        let client = KimiK2Client::new_with_base_url("test-key".to_string(), server.url());
        let err = kimik2_get_completion(&client, &sample_command())
            .await
            .expect_err("expected deserialize failure");
        assert!(matches!(err, KimiK2CompletionFailure::Deserialize(_)));
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

        let client = KimiK2Client::new_with_base_url("test-key".to_string(), server.url());
        let err = kimik2_get_completion(&client, &sample_command())
            .await
            .expect_err("empty choices must surface as a typed failure");
        match err {
            KimiK2CompletionFailure::MalformedResponse { reason } => {
                assert!(reason.contains("no choices"), "unexpected reason: {reason}");
            }
            other => panic!("expected MalformedResponse, got {other:?}"),
        }
    }

    #[test]
    fn debug_output_redacts_api_key() {
        let client = KimiK2Client::new_with_base_url(
            "super-secret-moonshot-key".to_string(),
            "https://example.test",
        );
        let debug = format!("{client:?}");
        assert!(
            !debug.contains("super-secret-moonshot-key"),
            "Debug output must not leak api_key: {debug}"
        );
    }

    #[tokio::test]
    #[ignore = "requires MOONSHOT_API_KEY; run with `cargo test -- --ignored`"]
    async fn live_kimik2_completion_returns_real_response() {
        let client = KimiK2Client::from_env().expect("MOONSHOT_API_KEY must be set");
        let command = KimiK2CompletionCommand {
            model: KimiK2Model::MoonshotV1_8k,
            system_prompt: Some("Reply with exactly the word 'pong'.".to_string()),
            messages: vec![KimiK2Message {
                role: KimiK2Role::User,
                content: "ping".to_string(),
            }],
            max_tokens: Some(8),
            temperature: Some(0.0),
        };
        let success = kimik2_get_completion(&client, &command)
            .await
            .expect("live Kimi K2 request should succeed");
        assert!(!success.content.is_empty(), "content should be non-empty");
        assert!(success.prompt_tokens > 0, "prompt_tokens should be > 0");
        assert!(
            success.completion_tokens > 0,
            "completion_tokens should be > 0"
        );
    }
}
