mod ai_client_apis;

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use futures::{FutureExt, StreamExt};

use crate::ai_client_apis::{claude::*, gemini::*, openai::*};

pub use crate::ai_client_apis::claude::{ClaudeClient, ClaudeCompletionCommand};
pub use crate::ai_client_apis::gemini::{GeminiClient, GeminiCompletionCommand};
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
    #[error("{provider:?}: response malformed: {reason}")]
    MalformedResponse {
        provider: ProviderKind,
        reason: String,
    },
    #[error("{provider:?}: stub provider has no real client yet")]
    ProviderStub { provider: ProviderKind },
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
            | Self::MalformedResponse { provider, .. }
            | Self::ProviderStub { provider } => *provider,
        }
    }

    pub fn is_transient(&self) -> bool {
        match self {
            Self::Transport { .. } | Self::RateLimited { .. } | Self::ServerError { .. } => true,
            Self::Deserialize { .. }
            | Self::Auth { .. }
            | Self::InvalidRequest { .. }
            | Self::MalformedResponse { .. }
            | Self::ProviderStub { .. } => false,
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
            Ok(_success) => Ok(AgnosticCompletionOutput {
                chunks: vec![],
                tokens_used: CompletionOutputTokensUsed {
                    input: 0,
                    output: 0,
                },
            }),
            Err(_failure) => Err(AgnosticCompletionError::ProviderStub {
                provider: ProviderKind::Claude,
            }),
        },
        RawAiCompletionResult::Gemini(result) => match result {
            Ok(_success) => Ok(AgnosticCompletionOutput {
                chunks: vec![],
                tokens_used: CompletionOutputTokensUsed {
                    input: 0,
                    output: 0,
                },
            }),
            Err(_failure) => Err(AgnosticCompletionError::ProviderStub {
                provider: ProviderKind::Gemini,
            }),
        },
    }
}

const DEFAULT_MAX_ATTEMPTS: u32 = 3;
const DEFAULT_BASE_DELAY: Duration = Duration::from_millis(100);

struct RetryOutcome {
    result: Result<AgnosticCompletionOutput, AgnosticCompletionError>,
    retries: u32,
    latency: Duration,
}

async fn run_with_retry<F, Fut>(
    max_attempts: u32,
    base_delay: Duration,
    mut operation: F,
) -> RetryOutcome
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<AgnosticCompletionOutput, AgnosticCompletionError>>,
{
    let start = Instant::now();
    let mut retries: u32 = 0;
    loop {
        match operation().await {
            Ok(output) => {
                return RetryOutcome {
                    result: Ok(output),
                    retries,
                    latency: start.elapsed(),
                };
            }
            Err(err) => {
                let can_retry = err.is_transient() && retries + 1 < max_attempts;
                if !can_retry {
                    return RetryOutcome {
                        result: Err(err),
                        retries,
                        latency: start.elapsed(),
                    };
                }
                let delay = delay_for_attempt(&err, retries, base_delay);
                tokio::time::sleep(delay).await;
                retries += 1;
            }
        }
    }
}

fn delay_for_attempt(err: &AgnosticCompletionError, attempt: u32, base: Duration) -> Duration {
    if let AgnosticCompletionError::RateLimited {
        retry_after: Some(ra),
        ..
    } = err
    {
        return *ra;
    }
    let exp = attempt.min(6);
    let backoff_ms = (base.as_millis() as u64).saturating_mul(1u64 << exp);
    let jitter_ms = jitter_within(backoff_ms / 4);
    Duration::from_millis(backoff_ms.saturating_add(jitter_ms))
}

static JITTER_COUNTER: AtomicU64 = AtomicU64::new(0);

fn jitter_within(max_ms: u64) -> u64 {
    if max_ms == 0 {
        return 0;
    }
    let mut z = JITTER_COUNTER
        .fetch_add(1, Ordering::Relaxed)
        .wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    z % (max_ms + 1)
}

fn build_attempt<'a, F, Fut>(
    provider: ProviderKind,
    input_index: usize,
    raw_op: F,
) -> BoxFuture<'a, ProviderAttempt>
where
    F: FnMut() -> Fut + Send + 'a,
    Fut: std::future::Future<Output = Result<AgnosticCompletionOutput, AgnosticCompletionError>>
        + Send
        + 'a,
{
    async move {
        let outcome = run_with_retry(DEFAULT_MAX_ATTEMPTS, DEFAULT_BASE_DELAY, raw_op).await;
        ProviderAttempt {
            provider,
            input_index,
            result: outcome.result,
            retries: outcome.retries,
            latency: outcome.latency,
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
    use std::collections::HashSet;

    use crate::{
        AiCompletionInputs, ClaudeClient, ClaudeCompletionCommand, CompletionOutputChunk,
        GeminiClient, GeminiCompletionCommand, MultiAiCompletionInputs, OpenAiClient,
        OpenAiCompletionCommand, OpenAiMessage, OpenAiModel, OpenAiRole, ProviderKind,
    };

    use super::multi_infer;

    #[tokio::test]
    async fn multi_infer_returns_one_attempt_per_stub_input_with_failures_preserved() {
        let claude_client = ClaudeClient {};
        let claude_command = ClaudeCompletionCommand {};
        let gemini_client = GeminiClient {};
        let gemini_command_a = GeminiCompletionCommand {};
        let gemini_command_b = GeminiCompletionCommand {};

        let inputs = [
            AiCompletionInputs::Gemini(&gemini_client, &gemini_command_a),
            AiCompletionInputs::Claude(&claude_client, &claude_command),
            AiCompletionInputs::Gemini(&gemini_client, &gemini_command_b),
        ];
        let multi = MultiAiCompletionInputs::new(&inputs);

        let attempts = multi_infer(&multi).await;

        assert_eq!(
            attempts.len(),
            inputs.len(),
            "every input should produce exactly one ProviderAttempt"
        );

        let providers: HashSet<ProviderKind> = attempts.iter().map(|a| a.provider).collect();
        assert!(providers.contains(&ProviderKind::Claude));
        assert!(providers.contains(&ProviderKind::Gemini));

        let mut seen_indices: Vec<usize> = attempts.iter().map(|a| a.input_index).collect();
        seen_indices.sort();
        assert_eq!(
            seen_indices,
            vec![0, 1, 2],
            "every input slot must appear exactly once"
        );

        for attempt in &attempts {
            let err = attempt
                .result
                .as_ref()
                .expect_err("Claude/Gemini stubs always Err until their M2 slice lands");
            assert_eq!(err.provider(), attempt.provider);
            assert_eq!(attempt.retries, 0, "ProviderStub is non-transient");
        }
    }

    #[tokio::test]
    async fn multi_infer_preserves_input_index_with_duplicate_same_provider_inputs() {
        let claude_client = ClaudeClient {};
        let claude_command_a = ClaudeCompletionCommand {};
        let claude_command_b = ClaudeCompletionCommand {};
        let gemini_client = GeminiClient {};
        let gemini_command = GeminiCompletionCommand {};

        let inputs = [
            AiCompletionInputs::Claude(&claude_client, &claude_command_a),
            AiCompletionInputs::Gemini(&gemini_client, &gemini_command),
            AiCompletionInputs::Claude(&claude_client, &claude_command_b),
        ];
        let multi = MultiAiCompletionInputs::new(&inputs);

        let mut attempts = multi_infer(&multi).await;
        assert_eq!(attempts.len(), 3);

        attempts.sort_by_key(|a| a.input_index);

        assert_eq!(attempts[0].input_index, 0);
        assert_eq!(attempts[0].provider, ProviderKind::Claude);
        assert_eq!(attempts[1].input_index, 1);
        assert_eq!(attempts[1].provider, ProviderKind::Gemini);
        assert_eq!(attempts[2].input_index, 2);
        assert_eq!(attempts[2].provider, ProviderKind::Claude);
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

        let openai_client = OpenAiClient::new("k".to_string()).with_base_url(server.url());
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

        let openai_client = OpenAiClient::new("k".to_string()).with_base_url(server.url());
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
            AiCompletionCommand::Claude(ClaudeCompletionCommand {})
        }
        SortableJudgementProvider::Gemini => {
            AiCompletionCommand::Gemini(GeminiCompletionCommand {})
        }
        SortableJudgementProvider::OpenAi => {
            AiCompletionCommand::OpenAi(OpenAiCompletionCommand::default())
        }
    }
}
