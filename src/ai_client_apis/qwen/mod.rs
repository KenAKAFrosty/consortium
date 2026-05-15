pub struct QwenClient {}

pub struct QwenCompletionCommand {}

#[derive(Debug)]
pub struct QwenCompletionSuccess {}
#[derive(Debug)]
pub struct QwenCompletionFailure {}

pub type QwenResult = Result<QwenCompletionSuccess, QwenCompletionFailure>;

pub async fn qwen_get_completion(
    client: &QwenClient,
    command: &QwenCompletionCommand,
) -> QwenResult {
    Err(QwenCompletionFailure {})
}
