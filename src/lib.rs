mod ai_client_apis;

use std::time::{Duration, Instant};

use backon::{BackoffBuilder, ExponentialBuilder};
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use futures::{FutureExt, StreamExt};

use crate::ai_client_apis::{claude::*, gemini::*, openai::*};

pub use crate::ai_client_apis::claude::{
    ClaudeClient, ClaudeClientError, ClaudeCompletionCommand, ClaudeMessage, ClaudeModel,
    ClaudeRole,
};
pub use crate::ai_client_apis::gemini::{
    GeminiClient, GeminiClientError, GeminiCompletionCommand, GeminiMessage, GeminiModel,
    GeminiRole,
};
pub use crate::ai_client_apis::openai::{
    OpenAiClient, OpenAiClientError, OpenAiCompletionCommand, OpenAiMessage, OpenAiModel,
    OpenAiRole,
};

#[derive(Clone, Copy)]
pub enum AiCompletionInputs<'a> {
    Gemini(&'a GeminiClient, &'a GeminiCompletionCommand),
    OpenAi(&'a OpenAiClient, &'a OpenAiCompletionCommand),
    Claude(&'a ClaudeClient, &'a ClaudeCompletionCommand),
    // KimiK2(&'a KimiK2Client, &'a KimiK2CompletionCommand),
    // Deepseek(&'a DeepseekClient, &'a DeepseekCompletionCommand),
    // Qwen(&'a QwenClient, &'a QwenCompletionCommand),
    // Llama(&'a LlamaClient, &'a LlamaCompletionCommand),
}

pub struct MultiAiCompletionInputs<'a> {
    completion_inputs: &'a [AiCompletionInputs<'a>],
}

impl<'a> MultiAiCompletionInputs<'a> {
    pub fn new(completion_inputs: &'a [AiCompletionInputs<'a>]) -> Self {
        Self { completion_inputs }
    }
}

#[derive(Debug)]
enum RawAiCompletionResult {
    Gemini(GeminiResult),
    OpenAi(OpenAiResult),
    Claude(ClaudeResult),
    // KimiK2(KimiK2Result),
    // Deepseek(DeepseekResult),
    // Qwen(QwenResult),
    // Llama(LlamaResult),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderKind {
    Claude,
    OpenAi,
    Gemini,
}

#[derive(Debug, thiserror::Error)]
pub enum AgnosticCompletionError {
    #[error("{provider:?}: transport failure: {source}")]
    Transport {
        provider: ProviderKind,
        #[source]
        source: reqwest::Error,
    },
    /// Provider HTTP response JSON could not be decoded against the wire schema.
    /// Non-transient: re-issuing the same request would yield the same payload.
    ///
    /// This is **not** the right variant for malformed structured text produced by
    /// the LLM itself (e.g., a model that violated a requested JSON schema inside
    /// its content). When that becomes a modeled concern, add a separate variant
    /// (e.g., `ModelOutputSchemaViolation`) rather than overloading this one.
    #[error("{provider:?}: response deserialization failed: {source}")]
    Deserialize {
        provider: ProviderKind,
        #[source]
        source: serde_json::Error,
    },
    #[error("{provider:?}: rate limited")]
    RateLimited {
        provider: ProviderKind,
        retry_after: Option<Duration>,
        message: Option<String>,
    },
    #[error("{provider:?}: authentication failed")]
    Auth {
        provider: ProviderKind,
        message: Option<String>,
    },
    #[error("{provider:?}: invalid request: {message}")]
    InvalidRequest {
        provider: ProviderKind,
        message: String,
    },
    #[error("{provider:?}: server error (status {status})")]
    ServerError {
        provider: ProviderKind,
        status: u16,
        message: Option<String>,
    },
    /// Provider returned a valid-JSON response that parsed against the wire schema
    /// but failed a semantic invariant — e.g., 200 OK with `choices: []`, or a
    /// candidate with no text parts. Non-transient.
    ///
    /// Like `Deserialize`, this is about provider-wire-protocol violations, not
    /// about whether the LLM's text content satisfied a downstream contract.
    #[error("{provider:?}: response malformed: {reason}")]
    MalformedResponse {
        provider: ProviderKind,
        reason: String,
    },
}

impl AgnosticCompletionError {
    pub fn provider(&self) -> ProviderKind {
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

    pub fn is_transient(&self) -> bool {
        match self {
            Self::Transport { .. } | Self::RateLimited { .. } | Self::ServerError { .. } => true,
            Self::Deserialize { .. }
            | Self::Auth { .. }
            | Self::InvalidRequest { .. }
            | Self::MalformedResponse { .. } => false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum CompletionOutputImage {
    Base64(String),
    Raw(bytes::Bytes),
}
#[derive(Debug, Clone)]
pub enum CompletionOutputChunk {
    Text(String),
    Image(CompletionOutputImage),
}

#[derive(Debug, Clone)]
//TOOD: Make this more detailed with breakdowns like # of reasoning tokens, or system tokens vs other input tokens, etc.
pub struct CompletionOutputTokensUsed {
    pub input: u64,
    pub output: u64,
}

#[derive(Debug, Clone)]
pub struct AgnosticCompletionOutput {
    pub chunks: Vec<CompletionOutputChunk>,
    pub tokens_used: CompletionOutputTokensUsed,
}

#[derive(Debug)]
pub struct ProviderAttempt {
    pub provider: ProviderKind,
    pub input_index: usize,
    pub result: Result<AgnosticCompletionOutput, AgnosticCompletionError>,
    pub retries: u32,
    pub latency: Duration,
}

fn gemini_failure_to_agnostic(
    failure: crate::ai_client_apis::gemini::GeminiCompletionFailure,
) -> AgnosticCompletionError {
    use crate::ai_client_apis::gemini::GeminiCompletionFailure as F;
    let provider = ProviderKind::Gemini;
    match failure {
        F::Transport(source) => AgnosticCompletionError::Transport { provider, source },
        F::Deserialize(source) => AgnosticCompletionError::Deserialize { provider, source },
        F::Auth { message } => AgnosticCompletionError::Auth { provider, message },
        F::RateLimited {
            retry_after,
            message,
        } => AgnosticCompletionError::RateLimited {
            provider,
            retry_after,
            message,
        },
        F::InvalidRequest { message } => AgnosticCompletionError::InvalidRequest {
            provider,
            message,
        },
        F::ServerError { status, message } => AgnosticCompletionError::ServerError {
            provider,
            status,
            message,
        },
        F::MalformedResponse { reason } => {
            AgnosticCompletionError::MalformedResponse { provider, reason }
        }
    }
}

fn claude_failure_to_agnostic(
    failure: crate::ai_client_apis::claude::ClaudeCompletionFailure,
) -> AgnosticCompletionError {
    use crate::ai_client_apis::claude::ClaudeCompletionFailure as F;
    let provider = ProviderKind::Claude;
    match failure {
        F::Transport(source) => AgnosticCompletionError::Transport { provider, source },
        F::Deserialize(source) => AgnosticCompletionError::Deserialize { provider, source },
        F::Auth { message } => AgnosticCompletionError::Auth { provider, message },
        F::RateLimited {
            retry_after,
            message,
        } => AgnosticCompletionError::RateLimited {
            provider,
            retry_after,
            message,
        },
        F::InvalidRequest { message } => AgnosticCompletionError::InvalidRequest {
            provider,
            message,
        },
        F::ServerError { status, message } => AgnosticCompletionError::ServerError {
            provider,
            status,
            message,
        },
        F::MalformedResponse { reason } => {
            AgnosticCompletionError::MalformedResponse { provider, reason }
        }
    }
}

fn openai_failure_to_agnostic(
    failure: crate::ai_client_apis::openai::OpenAiCompletionFailure,
) -> AgnosticCompletionError {
    use crate::ai_client_apis::openai::OpenAiCompletionFailure as F;
    let provider = ProviderKind::OpenAi;
    match failure {
        F::Transport(source) => AgnosticCompletionError::Transport { provider, source },
        F::Deserialize(source) => AgnosticCompletionError::Deserialize { provider, source },
        F::Auth { message } => AgnosticCompletionError::Auth { provider, message },
        F::RateLimited {
            retry_after,
            message,
        } => AgnosticCompletionError::RateLimited {
            provider,
            retry_after,
            message,
        },
        F::InvalidRequest { message } => AgnosticCompletionError::InvalidRequest {
            provider,
            message,
        },
        F::ServerError { status, message } => AgnosticCompletionError::ServerError {
            provider,
            status,
            message,
        },
        F::MalformedResponse { reason } => {
            AgnosticCompletionError::MalformedResponse { provider, reason }
        }
    }
}

fn convert_raw_result_to_agnostic_output(
    raw_result: RawAiCompletionResult,
) -> Result<AgnosticCompletionOutput, AgnosticCompletionError> {
    match raw_result {
        RawAiCompletionResult::OpenAi(result) => match result {
            Ok(success) => Ok(AgnosticCompletionOutput {
                chunks: vec![CompletionOutputChunk::Text(success.content)],
                tokens_used: CompletionOutputTokensUsed {
                    input: success.prompt_tokens,
                    output: success.completion_tokens,
                },
            }),
            Err(failure) => Err(openai_failure_to_agnostic(failure)),
        },
        RawAiCompletionResult::Claude(result) => match result {
            Ok(success) => Ok(AgnosticCompletionOutput {
                chunks: vec![CompletionOutputChunk::Text(success.content)],
                tokens_used: CompletionOutputTokensUsed {
                    input: success.input_tokens,
                    output: success.output_tokens,
                },
            }),
            Err(failure) => Err(claude_failure_to_agnostic(failure)),
        },
        RawAiCompletionResult::Gemini(result) => match result {
            Ok(success) => Ok(AgnosticCompletionOutput {
                chunks: vec![CompletionOutputChunk::Text(success.content)],
                tokens_used: CompletionOutputTokensUsed {
                    input: success.input_tokens,
                    output: success.output_tokens,
                },
            }),
            Err(failure) => Err(gemini_failure_to_agnostic(failure)),
        },
    }
}

const MAX_RETRIES: usize = 2;
const BASE_RETRY_DELAY: Duration = Duration::from_millis(100);

fn retry_after_override(err: &AgnosticCompletionError) -> Option<Duration> {
    match err {
        AgnosticCompletionError::RateLimited {
            retry_after: Some(ra),
            ..
        } => Some(*ra),
        _ => None,
    }
}

fn build_attempt<'a, F, Fut>(
    provider: ProviderKind,
    input_index: usize,
    mut raw_op: F,
) -> BoxFuture<'a, ProviderAttempt>
where
    F: FnMut() -> Fut + Send + 'a,
    Fut: std::future::Future<Output = Result<AgnosticCompletionOutput, AgnosticCompletionError>>
        + Send
        + 'a,
{
    async move {
        let start = Instant::now();
        let mut retries: u32 = 0;
        let mut backoff = ExponentialBuilder::default()
            .with_min_delay(BASE_RETRY_DELAY)
            .with_max_times(MAX_RETRIES)
            .with_jitter()
            .build();

        let result = loop {
            match raw_op().await {
                Ok(output) => break Ok(output),
                Err(err) => {
                    if !err.is_transient() {
                        break Err(err);
                    }
                    let Some(backoff_delay) = backoff.next() else {
                        break Err(err);
                    };
                    let delay = retry_after_override(&err).unwrap_or(backoff_delay);
                    tokio::time::sleep(delay).await;
                    retries += 1;
                }
            }
        };

        ProviderAttempt {
            provider,
            input_index,
            result,
            retries,
            latency: start.elapsed(),
        }
    }
    .boxed()
}

pub async fn multi_infer<'a>(inputs: &'a MultiAiCompletionInputs<'a>) -> Vec<ProviderAttempt> {
    let mut in_flight: FuturesUnordered<BoxFuture<'a, ProviderAttempt>> = FuturesUnordered::new();

    for (input_index, input) in inputs.completion_inputs.iter().copied().enumerate() {
        let attempt_future = match input {
            AiCompletionInputs::Claude(client, command) => build_attempt(
                ProviderKind::Claude,
                input_index,
                move || async move {
                    let raw = claude_get_completion(client, command).await;
                    convert_raw_result_to_agnostic_output(RawAiCompletionResult::Claude(raw))
                },
            ),
            AiCompletionInputs::OpenAi(client, command) => build_attempt(
                ProviderKind::OpenAi,
                input_index,
                move || async move {
                    let raw = openai_get_completion(client, command).await;
                    convert_raw_result_to_agnostic_output(RawAiCompletionResult::OpenAi(raw))
                },
            ),
            AiCompletionInputs::Gemini(client, command) => build_attempt(
                ProviderKind::Gemini,
                input_index,
                move || async move {
                    let raw = gemini_get_completion(client, command).await;
                    convert_raw_result_to_agnostic_output(RawAiCompletionResult::Gemini(raw))
                },
            ),
        };
        in_flight.push(attempt_future);
    }

    let mut attempts = Vec::with_capacity(inputs.completion_inputs.len());
    while let Some(attempt) = in_flight.next().await {
        attempts.push(attempt);
    }
    attempts
}

#[cfg(test)]
mod tests {
    use crate::{
        AiCompletionInputs, ClaudeClient, ClaudeCompletionCommand, ClaudeMessage, ClaudeModel,
        ClaudeRole, CompletionOutputChunk, GeminiClient, GeminiCompletionCommand, GeminiMessage,
        GeminiModel, GeminiRole, MultiAiCompletionInputs, OpenAiClient, OpenAiCompletionCommand,
        OpenAiMessage, OpenAiModel, OpenAiRole, ProviderKind,
    };

    use super::multi_infer;

    #[tokio::test]
    async fn multi_infer_preserves_input_index_with_duplicate_same_provider_inputs() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_body(
                r#"{
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}
                }"#,
            )
            .expect(3)
            .create_async()
            .await;

        let client = OpenAiClient::new_with_base_url("k".to_string(), server.url());
        let cmd_a = OpenAiCompletionCommand {
            model: OpenAiModel::Gpt4oMini,
            system_prompt: None,
            messages: vec![OpenAiMessage {
                role: OpenAiRole::User,
                content: "a".to_string(),
            }],
            max_tokens: Some(8),
            temperature: None,
        };
        let cmd_b = OpenAiCompletionCommand {
            messages: vec![OpenAiMessage {
                role: OpenAiRole::User,
                content: "b".to_string(),
            }],
            ..cmd_a.clone()
        };
        let cmd_c = OpenAiCompletionCommand {
            messages: vec![OpenAiMessage {
                role: OpenAiRole::User,
                content: "c".to_string(),
            }],
            ..cmd_a.clone()
        };

        let inputs = [
            AiCompletionInputs::OpenAi(&client, &cmd_a),
            AiCompletionInputs::OpenAi(&client, &cmd_b),
            AiCompletionInputs::OpenAi(&client, &cmd_c),
        ];
        let multi = MultiAiCompletionInputs::new(&inputs);

        let mut attempts = multi_infer(&multi).await;
        assert_eq!(attempts.len(), 3);

        attempts.sort_by_key(|a| a.input_index);
        for (i, attempt) in attempts.iter().enumerate() {
            assert_eq!(attempt.input_index, i);
            assert_eq!(attempt.provider, ProviderKind::OpenAi);
            assert!(
                attempt.result.is_ok(),
                "duplicate same-provider inputs should all succeed independently"
            );
        }
    }

    #[tokio::test]
    async fn multi_infer_gemini_success_path_emits_real_text_and_tokens() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1beta/models/gemini-1.5-flash:generateContent")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "candidates": [{"content": {"parts": [{"text": "gemini fan-out works"}]}}],
                    "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 5}
                }"#,
            )
            .create_async()
            .await;

        let gemini_client = GeminiClient::new_with_base_url("k".to_string(), server.url());
        let gemini_command = GeminiCompletionCommand {
            model: GeminiModel::Gemini15Flash,
            system_prompt: None,
            messages: vec![GeminiMessage {
                role: GeminiRole::User,
                content: "go".to_string(),
            }],
            max_tokens: Some(16),
            temperature: None,
        };
        let inputs = [AiCompletionInputs::Gemini(&gemini_client, &gemini_command)];
        let multi = MultiAiCompletionInputs::new(&inputs);

        let attempts = multi_infer(&multi).await;
        assert_eq!(attempts.len(), 1);
        let attempt = &attempts[0];
        assert_eq!(attempt.provider, ProviderKind::Gemini);
        assert_eq!(attempt.input_index, 0);

        let output = attempt
            .result
            .as_ref()
            .expect("Gemini path must produce a real output, not ProviderStub");

        match output.chunks.as_slice() {
            [CompletionOutputChunk::Text(text)] => assert_eq!(text, "gemini fan-out works"),
            other => panic!("expected a single text chunk, got {other:?}"),
        }
        assert_eq!(output.tokens_used.input, 9);
        assert_eq!(output.tokens_used.output, 5);
    }

    #[tokio::test]
    async fn multi_infer_claude_success_path_emits_real_text_and_tokens() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/messages")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "content": [{"type": "text", "text": "claude fan-out works"}],
                    "usage": {"input_tokens": 12, "output_tokens": 7}
                }"#,
            )
            .create_async()
            .await;

        let claude_client = ClaudeClient::new_with_base_url("k".to_string(), server.url());
        let claude_command = ClaudeCompletionCommand {
            model: ClaudeModel::Sonnet46,
            system_prompt: None,
            messages: vec![ClaudeMessage {
                role: ClaudeRole::User,
                content: "go".to_string(),
            }],
            max_tokens: 16,
            temperature: None,
        };
        let inputs = [AiCompletionInputs::Claude(&claude_client, &claude_command)];
        let multi = MultiAiCompletionInputs::new(&inputs);

        let attempts = multi_infer(&multi).await;
        assert_eq!(attempts.len(), 1);
        let attempt = &attempts[0];
        assert_eq!(attempt.provider, ProviderKind::Claude);
        assert_eq!(attempt.input_index, 0);

        let output = attempt
            .result
            .as_ref()
            .expect("Claude path must produce a real output, not ProviderStub");

        match output.chunks.as_slice() {
            [CompletionOutputChunk::Text(text)] => assert_eq!(text, "claude fan-out works"),
            other => panic!("expected a single text chunk, got {other:?}"),
        }
        assert_eq!(output.tokens_used.input, 12);
        assert_eq!(output.tokens_used.output, 7);
    }

    #[tokio::test]
    async fn multi_infer_openai_success_path_emits_real_text_and_tokens() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                r#"{
                    "choices": [{"message": {"content": "fan-out works"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4}
                }"#,
            )
            .create_async()
            .await;

        let openai_client = OpenAiClient::new_with_base_url("k".to_string(), server.url());
        let openai_command = OpenAiCompletionCommand {
            model: OpenAiModel::Gpt4oMini,
            system_prompt: None,
            messages: vec![OpenAiMessage {
                role: OpenAiRole::User,
                content: "go".to_string(),
            }],
            max_tokens: Some(16),
            temperature: None,
        };
        let inputs = [AiCompletionInputs::OpenAi(&openai_client, &openai_command)];
        let multi = MultiAiCompletionInputs::new(&inputs);

        let attempts = multi_infer(&multi).await;
        assert_eq!(attempts.len(), 1);
        let attempt = &attempts[0];
        assert_eq!(attempt.provider, ProviderKind::OpenAi);
        assert_eq!(attempt.input_index, 0);

        let output = attempt
            .result
            .as_ref()
            .expect("OpenAI path must produce a real output, not ProviderStub");

        match output.chunks.as_slice() {
            [CompletionOutputChunk::Text(text)] => assert_eq!(text, "fan-out works"),
            other => panic!("expected a single text chunk, got {other:?}"),
        }
        assert_eq!(output.tokens_used.input, 10);
        assert_eq!(output.tokens_used.output, 4);
    }

    #[tokio::test(start_paused = true)]
    async fn multi_infer_openai_transient_503_drives_retry_then_surfaces_failure() {
        use crate::AgnosticCompletionError;

        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(503)
            .with_body(r#"{"error":{"message":"upstream busy"}}"#)
            .expect(3)
            .create_async()
            .await;

        let openai_client = OpenAiClient::new_with_base_url("k".to_string(), server.url());
        let openai_command = OpenAiCompletionCommand {
            model: OpenAiModel::Gpt4oMini,
            system_prompt: None,
            messages: vec![OpenAiMessage {
                role: OpenAiRole::User,
                content: "go".to_string(),
            }],
            max_tokens: Some(8),
            temperature: None,
        };
        let inputs = [AiCompletionInputs::OpenAi(&openai_client, &openai_command)];
        let multi = MultiAiCompletionInputs::new(&inputs);

        let attempts = multi_infer(&multi).await;
        assert_eq!(attempts.len(), 1);
        let attempt = &attempts[0];

        assert_eq!(attempt.provider, ProviderKind::OpenAi);
        assert_eq!(attempt.input_index, 0);
        assert_eq!(
            attempt.retries, 2,
            "DEFAULT_MAX_ATTEMPTS=3 → 1 initial + 2 retries"
        );

        let err = attempt
            .result
            .as_ref()
            .expect_err("503 is transient but should exhaust retries and surface as ServerError");
        match err {
            AgnosticCompletionError::ServerError {
                provider,
                status,
                message,
            } => {
                assert_eq!(*provider, ProviderKind::OpenAi);
                assert_eq!(*status, 503);
                assert_eq!(message.as_deref(), Some("upstream busy"));
            }
            other => panic!("expected ServerError, got {other:?}"),
        }

        mock.assert_async().await;
    }
}

//this is the higher level api and crate namesake. should feel like any other completion. does not need to conform to OpenAI spec; though it's probably smart to create a serde Deserialize struct to represent the OpenAPI spec, and create a conversion from/into  to make it super smooth to use this with said OpenAPI spec from the outside.
pub fn consortium_completion() {

    //this is where we'll have like PHase 1: intra-model consotrium output.
    //then phase 2: inter-model consortium output, using best-of for each model from phase 1
    //final output completion, though we'll want to maintain and return the others along the way, or provide callbacks/hooks to do somehting with them when theyr'e generated at least
}

const ORDERED_JUDGEMENT_SYSTEM_PROMPT: &'static str = r#"
WIP/TODO: Set up system prompt.

judge output based on the provided instructions + inputs given

give reasoning first,

xml style tag format, etc.
"#;

// #[derive(Deserialize)]
pub struct OrderedJudgementStructuredData {
    //this can be where we have the corresponding IDs in order, something like
    ordered_ids: Vec<String>,
}

pub enum SortableJudgementProvider {
    OpenAi,
    Claude,
    Gemini,
}

pub enum AiCompletionCommand {
    OpenAi(OpenAiCompletionCommand),
    Claude(ClaudeCompletionCommand),
    Gemini(GeminiCompletionCommand),
}
pub fn make_sortable_judgement_command(
    provider: &SortableJudgementProvider,
) -> AiCompletionCommand {
    match provider {
        SortableJudgementProvider::Claude => {
            AiCompletionCommand::Claude(ClaudeCompletionCommand::default())
        }
        SortableJudgementProvider::Gemini => {
            AiCompletionCommand::Gemini(GeminiCompletionCommand::default())
        }
        SortableJudgementProvider::OpenAi => {
            AiCompletionCommand::OpenAi(OpenAiCompletionCommand::default())
        }
    }
}
