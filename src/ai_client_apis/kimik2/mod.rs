pub struct KimiK2Client {}

pub struct KimiK2CompletionCommand {}

#[derive(Debug)]
pub struct KimiK2CompletionSuccess {}
#[derive(Debug)]
pub struct KimiK2CompletionFailure {}

pub type KimiK2Result = Result<KimiK2CompletionSuccess, KimiK2CompletionFailure>;

pub async fn kimik2_get_completion(
    client: &KimiK2Client,
    command: &KimiK2CompletionCommand,
) -> KimiK2Result {
    Err(KimiK2CompletionFailure {})
}
