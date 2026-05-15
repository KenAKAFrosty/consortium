pub struct GeminiClient {}
pub struct GeminiCompletionCommand {}

#[derive(Debug)]
pub struct GeminiCompletionSuccess {}
#[derive(Debug)]
pub struct GeminiCompletionFailure {}

pub type GeminiResult = Result<GeminiCompletionSuccess, GeminiCompletionFailure>;

pub async fn gemini_get_completion(
    client: &GeminiClient,
    command: &GeminiCompletionCommand,
) -> GeminiResult {
    Err(GeminiCompletionFailure {})
}
