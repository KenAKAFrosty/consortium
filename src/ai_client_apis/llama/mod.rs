pub struct LlamaClient {}

pub struct LlamaCompletionCommand {}

#[derive(Debug)]
pub struct LlamaCompletionSuccess {}
#[derive(Debug)]
pub struct LlamaCompletionFailure {}

pub type LlamaResult = Result<LlamaCompletionSuccess, LlamaCompletionFailure>;

pub async fn llama_get_completion(
    client: &LlamaClient,
    command: &LlamaCompletionCommand,
) -> LlamaResult {
    Err(LlamaCompletionFailure {})
}
