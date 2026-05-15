mod ai_client_apis;

use crate::ai_client_apis::{
    claude::*, deepseek::*, gemini::*, kimik2::*, llama::*, openai::*, qwen::*,
};

enum AiCompletionInputs<'a> {
    Gemini(&'a GeminiClient, &'a GeminiCompletionCommand),
    OpenAi(&'a OpenAiClient, &'a OpenAiCompletionCommand),
    Claude(&'a ClaudeClient, &'a ClaudeCompletionCommand),
    // KimiK2(&'a KimiK2Client, &'a KimiK2CompletionCommand),
    // Deepseek(&'a DeepseekClient, &'a DeepseekCompletionCommand),
    // Qwen(&'a QwenClient, &'a QwenCompletionCommand),
    // Llama(&'a LlamaClient, &'a LlamaCompletionCommand),
}

pub struct MultiAiCompletionInputs<'a> {
    completion_inputs: &'a Vec<AiCompletionInputs<'a>>,
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

#[derive(Debug)]
enum CompletionOutputImage<'a> {
    Base64(&'a str),
    Raw(&'a Vec<u8>), //will likely get the bytes package and use that here instead of a Vec<u8>
}
#[derive(Debug)]
enum CompletionOutputChunk<'a> {
    Text(&'a str),
    Image(CompletionOutputImage<'a>),
}

#[derive(Debug)]
//TOOD: Make this more detailed with breakdowns like # of reasoning tokens, or system tokens vs other input tokens, etc.
pub struct CompletionOutputTokensUsed {
    input: u64,
    output: u64,
}

#[derive(Debug)]
pub struct AgnosticCompletionOutput<'a> {
    chunks: Vec<CompletionOutputChunk<'a>>,
    tokens_used: CompletionOutputTokensUsed,
}

pub fn convert_raw_result_to_agnostic_output<'a>(
    raw_result: RawAiCompletionResult,
) -> AgnosticCompletionOutput<'a> {
    match raw_result {
        RawAiCompletionResult::OpenAi(result) => AgnosticCompletionOutput {
            chunks: vec![],
            tokens_used: CompletionOutputTokensUsed {
                input: 0,
                output: 0,
            },
        },
        RawAiCompletionResult::Claude(result) => AgnosticCompletionOutput {
            chunks: vec![],
            tokens_used: CompletionOutputTokensUsed {
                input: 0,
                output: 0,
            },
        },
        RawAiCompletionResult::Gemini(result) => AgnosticCompletionOutput {
            chunks: vec![],
            tokens_used: CompletionOutputTokensUsed {
                input: 0,
                output: 0,
            },
        },
    }
}

//this will all become async, but at time of writing was offline so could not add tokio etc)
pub fn multi_infer<'a>(inputs: &MultiAiCompletionInputs) -> Vec<AgnosticCompletionOutput<'a>> {
    println!("Running multi infer");

    //like here we could map into async functions and then run futures_unordered on them, etc.
    let completion_async_funcs = inputs
        .completion_inputs
        .iter()
        .map(async |input| match &input {
            AiCompletionInputs::Claude(client, command) => {
                RawAiCompletionResult::Claude(claude_get_completion(client, command).await)
            }
            AiCompletionInputs::Gemini(client, command) => {
                RawAiCompletionResult::Gemini(gemini_get_completion(client, command).await)
            }
            AiCompletionInputs::OpenAi(client, command) => {
                RawAiCompletionResult::OpenAi(openai_get_completion(client, command).await)
            } // AiCompletionInputs::Deepseek(client, command) => {
              //     RawAiCompletionResult::Deepseek(deepseek_get_completion(client, command).await)
              // }
              // AiCompletionInputs::KimiK2(client, command) => {
              //     RawAiCompletionResult::KimiK2(kimik2_get_completion(client, command).await)
              // }
              // AiCompletionInputs::Llama(client, command) => {
              //     RawAiCompletionResult::Llama(llama_get_completion(client, command).await)
              // }
              // AiCompletionInputs::Qwen(client, command) => {
              //     RawAiCompletionResult::Qwen(qwen_get_completion(client, command).await)
              // }
        });

    //once connected and can add futures crate, this is where we can do the futures_unordered thing. placeholder empty vec for now
    let raw_results: Vec<RawAiCompletionResult> = vec![];

    let agnostic_outputs: Vec<AgnosticCompletionOutput> = raw_results
        .into_iter()
        .map(|result| convert_raw_result_to_agnostic_output(result))
        .collect();

    // panic!("Not implemented");

    agnostic_outputs
}

#[cfg(test)]
mod tests {
    use crate::{
        AiCompletionInputs, MultiAiCompletionInputs,
        ai_client_apis::{
            gemini::{GeminiClient, GeminiCompletionCommand},
            openai::{OpenAiClient, OpenAiCompletionCommand},
        },
    };

    use super::multi_infer;
    #[test]
    fn multi_infer_placeholder_returns_empty_vec() {
        let commands = vec![
            AiCompletionInputs::Gemini(&GeminiClient {}, &GeminiCompletionCommand {}),
            AiCompletionInputs::OpenAi(&OpenAiClient {}, &OpenAiCompletionCommand {}),
        ];
        let result = multi_infer(&MultiAiCompletionInputs {
            completion_inputs: &commands,
        });
        assert!(result.is_empty());
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
pub struct OrderedJudgementStructuredData<'a> {
    //this can be where we have the corresponding IDs in order, something like
    ordered_ids: Vec<&'a str>,
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
            AiCompletionCommand::OpenAi(OpenAiCompletionCommand {})
        }
    }
}
