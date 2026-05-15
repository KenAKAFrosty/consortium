pub struct ClaudeClient {}

pub struct ClaudeCompletionCommand {}

#[derive(Debug)]
pub struct ClaudeCompletionSuccess {}
#[derive(Debug)]
pub struct ClaudeCompletionFailure {}

pub type ClaudeResult = Result<ClaudeCompletionSuccess, ClaudeCompletionFailure>;

pub async fn claude_get_completion(
    client: &ClaudeClient,
    command: &ClaudeCompletionCommand,
) -> ClaudeResult {
    Err(ClaudeCompletionFailure {})
}
