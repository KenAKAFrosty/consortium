pub struct OpenAiClient {}

pub struct OpenAiCompletionCommand {}

#[derive(Debug)]
pub struct OpenAiCompletionSuccess {}
#[derive(Debug)]
pub struct OpenAiCompletionFailure {}

pub type OpenAiResult = Result<OpenAiCompletionSuccess, OpenAiCompletionFailure>;

pub async fn openai_get_completion(
    client: &OpenAiClient,
    command: &OpenAiCompletionCommand,
) -> OpenAiResult {
    Err(OpenAiCompletionFailure {})
}
