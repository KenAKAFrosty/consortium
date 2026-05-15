pub struct DeepseekClient {}

pub struct DeepseekCompletionCommand {}

#[derive(Debug)]
pub struct DeepseekCompletionSuccess {}
#[derive(Debug)]
pub struct DeepseekCompletionFailure {}

pub type DeepseekResult = Result<DeepseekCompletionSuccess, DeepseekCompletionFailure>;

pub async fn deepseek_get_completion(
    client: &DeepseekClient,
    command: &DeepseekCompletionCommand,
) -> DeepseekResult {
    Err(DeepseekCompletionFailure {})
}
