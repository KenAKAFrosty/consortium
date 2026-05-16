//! Multi-prompt dataset pipeline (M6a).
//!
//! [`DatasetBuilder`] configures provider slot templates, judges, an embedder,
//! and a diversification strategy. [`DatasetBuilder::build`] validates the
//! configuration eagerly and returns a [`DatasetRun`]. [`DatasetRun::execute`]
//! runs (optional) diversification once over the prompt batch, then yields a
//! [`futures::Stream`] of [`PromptOutcome`]s — exactly one per input prompt,
//! emitted in original `prompt_index` order even when diversification leaves
//! some prompts in-place as [`PromptOutcome::Skipped`].
//!
//! ## Prompt-to-slot planning boundary
//!
//! [`DatasetBuilder`] does **not** store prompt-specific
//! [`crate::AiCompletionInputs`]. Each [`SlotTemplate`] owns its provider
//! client and a small `plan` closure that produces a fresh, owned,
//! provider-typed command for a given prompt. Per-prompt commands live only
//! for the duration of one [`crate::consortium_completion`] call — the
//! orchestrator's `ConsortiumSlot<'_>` borrows from a per-iteration scratch
//! [`SlotCommand`] vector.
//!
//! ## Failure preservation
//!
//! Three explicit categories, matching the orchestrator's pattern:
//!
//! - Fatal setup errors: [`DatasetBuildError`] is surfaced eagerly by
//!   [`DatasetBuilder::build`] before any prompt work begins.
//! - Fatal runtime errors: [`DatasetRunError`] is surfaced by
//!   [`DatasetRun::execute`] *before* the stream is returned — e.g., the
//!   embedder failed and diversification cannot proceed.
//! - Per-prompt outcomes: every prompt produces exactly one
//!   [`PromptOutcome`] — `Completed`, `Skipped`, or `Failed`. A failing
//!   prompt never terminates the stream; subsequent prompts continue.
//!
//! ## What M6a deliberately is not
//!
//! Per-prompt execution is sequential. Embedding is a single batch call
//! (auto-chunking is M3b/M6b territory). Judge concurrency stays inherited
//! from M5a (sequential within a phase). Streaming events (`PhaseEvent`,
//! callback hooks) are M5c work, not M6a.

use std::collections::HashSet;
use std::sync::Arc;

use futures::stream::{self, Stream, StreamExt};
use serde::Serialize;
use tokio::io::{AsyncWrite, AsyncWriteExt};

use crate::diversification::{SelectionStrategy, StopCondition, select_diverse};
use crate::embeddings::{AgnosticEmbeddingError, Embedder};
use crate::orchestrator::{
    ConsortiumOutcome, ConsortiumSlot, JudgeProvider, consortium_completion,
};
use crate::{
    AiCompletionInputs, ClaudeClient, ClaudeCompletionCommand, GeminiClient,
    GeminiCompletionCommand, OpenAiClient, OpenAiCompletionCommand, ProviderKind,
};

// ============================================================================
// Slot templates — the prompt-to-slot planning boundary.
// ============================================================================

/// One configured model "template" in the dataset pipeline.
///
/// A template owns its provider client and a `plan` closure that produces a
/// fresh, owned, provider-typed command for a given prompt. Per-prompt
/// commands are not stored in the template — the planner runs once per
/// prompt and the resulting command lives only for one
/// [`crate::consortium_completion`] call.
///
/// Construct via [`SlotTemplate::openai`] / [`SlotTemplate::claude`] /
/// [`SlotTemplate::gemini`]. The planner closure returns
/// `Result<XxxCompletionCommand, E>` for any `E: Into<Box<dyn Error + Send +
/// Sync>>` — including `String`, `&'static str`, and any concrete
/// `std::error::Error` impl. Planner failures surface per-prompt as
/// [`PromptRunError::SlotPlanning`] and do not terminate the stream.
//
// `clippy::type_complexity` would push us to name the planner closure types.
// We deliberately keep them inline: the error envelope stays an inline
// `Box<dyn Error + Send + Sync>` (per the M6a design call to not expose a
// public `PlanError` alias as the main API story), and naming the closure
// types separately while leaving the error inline produces aliases that read
// worse than the inline form.
#[allow(clippy::type_complexity)]
pub enum SlotTemplate {
    OpenAi {
        client: OpenAiClient,
        model_label: String,
        samples: usize,
        plan: Arc<
            dyn Fn(
                    &str,
                )
                    -> Result<OpenAiCompletionCommand, Box<dyn std::error::Error + Send + Sync>>
                + Send
                + Sync,
        >,
    },
    Claude {
        client: ClaudeClient,
        model_label: String,
        samples: usize,
        plan: Arc<
            dyn Fn(
                    &str,
                )
                    -> Result<ClaudeCompletionCommand, Box<dyn std::error::Error + Send + Sync>>
                + Send
                + Sync,
        >,
    },
    Gemini {
        client: GeminiClient,
        model_label: String,
        samples: usize,
        plan: Arc<
            dyn Fn(
                    &str,
                )
                    -> Result<GeminiCompletionCommand, Box<dyn std::error::Error + Send + Sync>>
                + Send
                + Sync,
        >,
    },
}

impl SlotTemplate {
    pub fn openai<F, E>(
        client: OpenAiClient,
        model_label: impl Into<String>,
        samples: usize,
        plan: F,
    ) -> Self
    where
        F: Fn(&str) -> Result<OpenAiCompletionCommand, E> + Send + Sync + 'static,
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::OpenAi {
            client,
            model_label: model_label.into(),
            samples,
            plan: Arc::new(move |prompt| plan(prompt).map_err(Into::into)),
        }
    }

    pub fn claude<F, E>(
        client: ClaudeClient,
        model_label: impl Into<String>,
        samples: usize,
        plan: F,
    ) -> Self
    where
        F: Fn(&str) -> Result<ClaudeCompletionCommand, E> + Send + Sync + 'static,
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::Claude {
            client,
            model_label: model_label.into(),
            samples,
            plan: Arc::new(move |prompt| plan(prompt).map_err(Into::into)),
        }
    }

    pub fn gemini<F, E>(
        client: GeminiClient,
        model_label: impl Into<String>,
        samples: usize,
        plan: F,
    ) -> Self
    where
        F: Fn(&str) -> Result<GeminiCompletionCommand, E> + Send + Sync + 'static,
        E: Into<Box<dyn std::error::Error + Send + Sync>>,
    {
        Self::Gemini {
            client,
            model_label: model_label.into(),
            samples,
            plan: Arc::new(move |prompt| plan(prompt).map_err(Into::into)),
        }
    }

    pub fn model_label(&self) -> &str {
        match self {
            Self::OpenAi { model_label, .. }
            | Self::Claude { model_label, .. }
            | Self::Gemini { model_label, .. } => model_label,
        }
    }

    pub fn samples(&self) -> usize {
        match self {
            Self::OpenAi { samples, .. }
            | Self::Claude { samples, .. }
            | Self::Gemini { samples, .. } => *samples,
        }
    }

    pub fn provider(&self) -> ProviderKind {
        match self {
            Self::OpenAi { .. } => ProviderKind::OpenAi,
            Self::Claude { .. } => ProviderKind::Claude,
            Self::Gemini { .. } => ProviderKind::Gemini,
        }
    }

    /// Run this template's planner against `prompt`. The returned
    /// [`SlotCommand`] variant always matches `self`'s variant — that pairing
    /// is the invariant `process_prompt` relies on when building
    /// [`ConsortiumSlot`]s.
    fn plan_for(
        &self,
        prompt: &str,
    ) -> Result<SlotCommand, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::OpenAi { plan, .. } => plan(prompt).map(SlotCommand::OpenAi),
            Self::Claude { plan, .. } => plan(prompt).map(SlotCommand::Claude),
            Self::Gemini { plan, .. } => plan(prompt).map(SlotCommand::Gemini),
        }
    }
}

/// Per-iteration owned command — the storage `ConsortiumSlot<'_>` borrows from
/// for the duration of a single [`crate::consortium_completion`] call.
/// Crate-private: never escapes this module.
enum SlotCommand {
    OpenAi(OpenAiCompletionCommand),
    Claude(ClaudeCompletionCommand),
    Gemini(GeminiCompletionCommand),
}

// ============================================================================
// Builder + run.
// ============================================================================

/// Configurable, eagerly-validated builder for a dataset run.
///
/// Generic over the embedder type because [`Embedder`] uses native `async fn
/// in trait` and is not dyn-safe; static dispatch only. The embedder is
/// consumed by [`DatasetRun::execute`] and never stored in the per-prompt
/// stream state.
pub struct DatasetBuilder<E: Embedder> {
    slot_templates: Vec<SlotTemplate>,
    judges: Vec<Arc<dyn JudgeProvider>>,
    embedder: E,
    selection: SelectionStrategy,
    stop_condition: StopCondition,
}

impl<E: Embedder> DatasetBuilder<E> {
    /// Start a new builder. Defaults: no slots, no judges,
    /// [`SelectionStrategy::Centroid`], and a [`StopCondition`] that does not
    /// filter anything (`max_n = None`, `similarity_tripwire = None`).
    pub fn new(embedder: E) -> Self {
        Self {
            slot_templates: Vec::new(),
            judges: Vec::new(),
            embedder,
            selection: SelectionStrategy::Centroid,
            stop_condition: StopCondition {
                max_n: None,
                similarity_tripwire: None,
            },
        }
    }

    pub fn slot(mut self, template: SlotTemplate) -> Self {
        self.slot_templates.push(template);
        self
    }

    /// Attach a judge by value. Convenience wrapper around [`Self::judge_arc`]
    /// — wraps `judge` in an [`Arc`] internally so multiple slots can share
    /// the same configuration.
    pub fn judge<J: JudgeProvider + 'static>(mut self, judge: J) -> Self {
        self.judges.push(Arc::new(judge));
        self
    }

    pub fn judge_arc(mut self, judge: Arc<dyn JudgeProvider>) -> Self {
        self.judges.push(judge);
        self
    }

    pub fn selection(mut self, selection: SelectionStrategy) -> Self {
        self.selection = selection;
        self
    }

    pub fn stop_condition(mut self, stop: StopCondition) -> Self {
        self.stop_condition = stop;
        self
    }

    /// Validate the configuration and produce a runnable [`DatasetRun`].
    /// Fails fast on missing slots, missing judges, or zero-sample slots.
    pub fn build(self) -> Result<DatasetRun<E>, DatasetBuildError> {
        if self.slot_templates.is_empty() {
            return Err(DatasetBuildError::NoSlots);
        }
        if self.judges.is_empty() {
            return Err(DatasetBuildError::NoJudges);
        }
        for (i, t) in self.slot_templates.iter().enumerate() {
            if t.samples() == 0 {
                return Err(DatasetBuildError::ZeroSamples {
                    slot_index: i,
                    model_label: t.model_label().to_string(),
                });
            }
        }
        Ok(DatasetRun {
            slot_templates: self.slot_templates,
            judges: self.judges,
            embedder: self.embedder,
            selection: self.selection,
            stop_condition: self.stop_condition,
        })
    }
}

/// Validated, ready-to-execute dataset configuration.
pub struct DatasetRun<E: Embedder> {
    slot_templates: Vec<SlotTemplate>,
    judges: Vec<Arc<dyn JudgeProvider>>,
    embedder: E,
    selection: SelectionStrategy,
    stop_condition: StopCondition,
}

impl<E: Embedder> DatasetRun<E> {
    /// Run the pipeline against `prompts`. Returns a [`Stream`] of exactly
    /// one [`PromptOutcome`] per input prompt, in original `prompt_index`
    /// order.
    ///
    /// Diversification runs once over the full batch before streaming begins.
    /// When the configured [`StopCondition`] cannot exclude anything
    /// (`max_n.is_none() && similarity_tripwire.is_none()`), the embedder is
    /// not called and every prompt is selected — keeping the no-filter path
    /// cheap.
    ///
    /// Empty `prompts` yields an empty stream and does not call the embedder.
    pub async fn execute(
        self,
        prompts: Vec<String>,
    ) -> Result<impl Stream<Item = PromptOutcome>, DatasetRunError> {
        let no_filter = self.stop_condition.max_n.is_none()
            && self.stop_condition.similarity_tripwire.is_none();

        let selected: HashSet<usize> = if prompts.is_empty() || no_filter {
            (0..prompts.len()).collect()
        } else {
            let batch = self
                .embedder
                .embed(&prompts)
                .await
                .map_err(DatasetRunError::Embedding)?;
            if batch.vectors.len() != prompts.len() {
                return Err(DatasetRunError::EmbeddingCountMismatch {
                    expected: prompts.len(),
                    got: batch.vectors.len(),
                });
            }
            // Count was non-empty (we only embed when `prompts` is non-empty),
            // so batch.vectors has at least one row. Anchor the expected
            // dimension on row 0 and validate the rest before handing off —
            // `select_diverse` is documented to panic on mixed dimensions, and
            // the M6a failure contract requires malformed embedding batches
            // to surface as typed runtime errors instead.
            let expected_dim = batch.vectors[0].len();
            for (row_index, row) in batch.vectors.iter().enumerate().skip(1) {
                if row.len() != expected_dim {
                    return Err(DatasetRunError::EmbeddingDimensionMismatch {
                        row_index,
                        expected: expected_dim,
                        actual: row.len(),
                    });
                }
            }
            select_diverse(&batch.vectors, self.selection, self.stop_condition)
                .into_iter()
                .collect()
        };

        let DatasetRun {
            slot_templates,
            judges,
            ..
        } = self;

        let state = StreamState {
            slot_templates,
            judges,
            prompts,
            selected,
            next: 0,
        };

        Ok(stream::unfold(state, |mut s| async move {
            if s.next >= s.prompts.len() {
                return None;
            }
            let i = s.next;
            s.next += 1;
            let prompt = std::mem::take(&mut s.prompts[i]);

            let outcome = if s.selected.contains(&i) {
                match process_prompt(&s.slot_templates, &s.judges, &prompt).await {
                    Ok(outcome) => PromptOutcome::Completed {
                        prompt_index: i,
                        prompt,
                        outcome,
                    },
                    Err(error) => PromptOutcome::Failed {
                        prompt_index: i,
                        prompt,
                        error,
                    },
                }
            } else {
                PromptOutcome::Skipped {
                    prompt_index: i,
                    prompt,
                    reason: SkipReason::NotSelectedByDiversification,
                }
            };

            Some((outcome, s))
        }))
    }
}

struct StreamState {
    slot_templates: Vec<SlotTemplate>,
    judges: Vec<Arc<dyn JudgeProvider>>,
    prompts: Vec<String>,
    selected: HashSet<usize>,
    next: usize,
}

async fn process_prompt(
    templates: &[SlotTemplate],
    judges: &[Arc<dyn JudgeProvider>],
    prompt: &str,
) -> Result<ConsortiumOutcome, PromptRunError> {
    let mut commands: Vec<SlotCommand> = Vec::with_capacity(templates.len());
    for (i, t) in templates.iter().enumerate() {
        match t.plan_for(prompt) {
            Ok(cmd) => commands.push(cmd),
            Err(source) => {
                return Err(PromptRunError::SlotPlanning {
                    slot_index: i,
                    model_label: t.model_label().to_string(),
                    source,
                });
            }
        }
    }

    let slots: Vec<ConsortiumSlot<'_>> = templates
        .iter()
        .zip(commands.iter())
        .map(|(t, c)| {
            let input = match (t, c) {
                (SlotTemplate::OpenAi { client, .. }, SlotCommand::OpenAi(cmd)) => {
                    AiCompletionInputs::OpenAi(client, cmd)
                }
                (SlotTemplate::Claude { client, .. }, SlotCommand::Claude(cmd)) => {
                    AiCompletionInputs::Claude(client, cmd)
                }
                (SlotTemplate::Gemini { client, .. }, SlotCommand::Gemini(cmd)) => {
                    AiCompletionInputs::Gemini(client, cmd)
                }
                // `plan_for` guarantees the SlotCommand variant matches the
                // SlotTemplate variant of `self`. Any other pairing is a
                // programmer error inside this module.
                _ => unreachable!(
                    "template/command variant mismatch — SlotTemplate::plan_for invariant violated"
                ),
            };
            ConsortiumSlot {
                input,
                model_label: t.model_label().to_string(),
                samples: t.samples(),
            }
        })
        .collect();

    let judge_refs: Vec<&dyn JudgeProvider> = judges.iter().map(|a| a.as_ref()).collect();
    Ok(consortium_completion(&slots, &judge_refs).await)
}

// ============================================================================
// Outcomes + errors.
// ============================================================================

/// Result of processing one prompt from the input batch. Exactly one
/// `PromptOutcome` is emitted per input prompt, in original `prompt_index`
/// order.
#[derive(Debug)]
pub enum PromptOutcome {
    /// The prompt was selected by diversification, slot planning succeeded,
    /// and the consortium pipeline produced a [`ConsortiumOutcome`]. The
    /// outcome may itself report `winner = None` (every model failed
    /// end-to-end); that detail is preserved inside `outcome` rather than
    /// being escalated here.
    Completed {
        prompt_index: usize,
        prompt: String,
        outcome: ConsortiumOutcome,
    },
    /// The prompt was excluded by the configured [`SelectionStrategy`] /
    /// [`StopCondition`]. Emitted in-place (not moved to the end of the
    /// stream) so callers see one row per input prompt in original order.
    Skipped {
        prompt_index: usize,
        prompt: String,
        reason: SkipReason,
    },
    /// The prompt was selected but a fatal pre-orchestration step failed
    /// (currently only slot planning). Later prompts continue.
    Failed {
        prompt_index: usize,
        prompt: String,
        error: PromptRunError,
    },
}

impl PromptOutcome {
    pub fn prompt_index(&self) -> usize {
        match self {
            Self::Completed { prompt_index, .. }
            | Self::Skipped { prompt_index, .. }
            | Self::Failed { prompt_index, .. } => *prompt_index,
        }
    }

    pub fn prompt(&self) -> &str {
        match self {
            Self::Completed { prompt, .. }
            | Self::Skipped { prompt, .. }
            | Self::Failed { prompt, .. } => prompt,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    NotSelectedByDiversification,
}

/// Per-prompt runtime failure. Typed outer envelope; the planner's underlying
/// error is preserved as a boxed [`std::error::Error`] source so callers can
/// downcast or inspect without the dataset module needing to know the planner
/// closure's error type.
#[derive(Debug, thiserror::Error)]
pub enum PromptRunError {
    #[error("slot {slot_index} ({model_label}): planner failed: {source}")]
    SlotPlanning {
        slot_index: usize,
        model_label: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// Fatal setup error returned by [`DatasetBuilder::build`].
#[derive(Debug, thiserror::Error)]
pub enum DatasetBuildError {
    #[error("dataset builder must have at least one slot template")]
    NoSlots,
    #[error("dataset builder must have at least one judge")]
    NoJudges,
    #[error("slot {slot_index} ({model_label}) has samples = 0")]
    ZeroSamples {
        slot_index: usize,
        model_label: String,
    },
}

/// Fatal one-shot runtime error returned by [`DatasetRun::execute`] before
/// the stream is constructed. Per-prompt failures use [`PromptRunError`]
/// instead and do not bubble out here.
#[derive(Debug, thiserror::Error)]
pub enum DatasetRunError {
    #[error("embedding failed: {0}")]
    Embedding(#[source] AgnosticEmbeddingError),
    #[error("embedder returned {got} vectors for {expected} prompts")]
    EmbeddingCountMismatch { expected: usize, got: usize },
    /// The embedder returned a batch whose vectors do not all share the same
    /// dimension. `select_diverse` requires uniform dimensionality and is
    /// documented to panic otherwise; the dataset layer rejects the batch
    /// before that contract is reached.
    #[error(
        "embedder vector at row {row_index} has dimension {actual}, expected {expected} (row 0)"
    )]
    EmbeddingDimensionMismatch {
        row_index: usize,
        expected: usize,
        actual: usize,
    },
}

// ============================================================================
// JSONL row shape + writer.
// ============================================================================

/// Compact serialisable projection of a [`PromptOutcome`] suitable for one
/// JSONL row. Deliberately small for M6a — captures the cross-model winner's
/// content (if any), the skip/failure reason, but not the full
/// [`ConsortiumOutcome`] graph. Callers that need richer audit data should
/// consume the stream directly and write their own projection.
#[derive(Debug, Serialize)]
pub struct DatasetRow {
    pub prompt_index: usize,
    pub prompt: String,
    pub status: RowStatus,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RowStatus {
    Completed { winner: Option<RowWinner> },
    Skipped { reason: RowSkipReason },
    Failed { error: String },
}

#[derive(Debug, Serialize)]
pub struct RowWinner {
    pub model_label: String,
    pub provider: &'static str,
    pub content: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RowSkipReason {
    NotSelectedByDiversification,
}

impl PromptOutcome {
    /// Project this outcome into the small JSONL-friendly [`DatasetRow`]
    /// shape used by [`write_jsonl`].
    pub fn to_row(&self) -> DatasetRow {
        match self {
            Self::Completed {
                prompt_index,
                prompt,
                outcome,
            } => DatasetRow {
                prompt_index: *prompt_index,
                prompt: prompt.clone(),
                status: RowStatus::Completed {
                    winner: outcome
                        .phase_two
                        .as_ref()
                        .and_then(|p2| p2.winner.as_ref())
                        .map(|w| RowWinner {
                            model_label: w.model_label.clone(),
                            provider: provider_str(w.provider),
                            content: w.content.clone(),
                        }),
                },
            },
            Self::Skipped {
                prompt_index,
                prompt,
                reason,
            } => DatasetRow {
                prompt_index: *prompt_index,
                prompt: prompt.clone(),
                status: RowStatus::Skipped {
                    reason: match reason {
                        SkipReason::NotSelectedByDiversification => {
                            RowSkipReason::NotSelectedByDiversification
                        }
                    },
                },
            },
            Self::Failed {
                prompt_index,
                prompt,
                error,
            } => DatasetRow {
                prompt_index: *prompt_index,
                prompt: prompt.clone(),
                status: RowStatus::Failed {
                    error: error.to_string(),
                },
            },
        }
    }
}

fn provider_str(p: ProviderKind) -> &'static str {
    match p {
        ProviderKind::OpenAi => "openai",
        ProviderKind::Claude => "claude",
        ProviderKind::Gemini => "gemini",
    }
}

/// Write each [`PromptOutcome`] as one JSONL line and `flush()` after every
/// line, so a tail / reader sees finalized rows promptly and a crash
/// preserves everything written up to the last finalized prompt.
///
/// Stops at the first I/O error or serialization error. Serialization
/// failures are converted to [`std::io::ErrorKind::Other`].
pub async fn write_jsonl<W, S>(mut writer: W, mut stream: S) -> std::io::Result<()>
where
    W: AsyncWrite + Unpin,
    S: Stream<Item = PromptOutcome> + Unpin,
{
    while let Some(outcome) = stream.next().await {
        let row = outcome.to_row();
        let mut buf = serde_json::to_vec(&row).map_err(std::io::Error::other)?;
        buf.push(b'\n');
        writer.write_all(&buf).await?;
        writer.flush().await?;
    }
    Ok(())
}

// ============================================================================
// Tests.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use futures::StreamExt;
    use futures::future::BoxFuture;

    use crate::embeddings::{EmbeddingBatch, EmbeddingUsage};
    use crate::judge::JudgeRequest;
    use crate::orchestrator::{CrossModelPhaseOutcome, PhaseTwoWinner};
    use crate::{AgnosticCompletionError, OpenAiClient, OpenAiMessage, OpenAiModel, OpenAiRole};

    // ---------- test doubles ----------

    struct TestEmbedder {
        vectors: Vec<Vec<f32>>,
    }

    impl Embedder for TestEmbedder {
        async fn embed(&self, inputs: &[String]) -> Result<EmbeddingBatch, AgnosticEmbeddingError> {
            assert_eq!(
                inputs.len(),
                self.vectors.len(),
                "TestEmbedder: configured vector count must match prompt count"
            );
            Ok(EmbeddingBatch {
                vectors: self.vectors.clone(),
                usage: EmbeddingUsage { input_tokens: 0 },
            })
        }
    }

    /// Embedder that returns vectors of mixed dimensions — used to assert
    /// the dataset layer catches malformed embedding batches before they
    /// reach `select_diverse`, which would otherwise panic on dimension
    /// mismatch.
    struct JaggedEmbedder;

    impl Embedder for JaggedEmbedder {
        async fn embed(
            &self,
            inputs: &[String],
        ) -> Result<EmbeddingBatch, AgnosticEmbeddingError> {
            let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());
            for (i, _) in inputs.iter().enumerate() {
                // Row 0 is 3-d; subsequent rows are 4-d.
                let dim = if i == 0 { 3 } else { 4 };
                vectors.push(vec![0.0_f32; dim]);
            }
            Ok(EmbeddingBatch {
                vectors,
                usage: EmbeddingUsage { input_tokens: 0 },
            })
        }
    }

    /// Embedder that panics on use — used to assert the no-filter
    /// short-circuit really never calls the embedder.
    struct PanicEmbedder;

    impl Embedder for PanicEmbedder {
        async fn embed(
            &self,
            _inputs: &[String],
        ) -> Result<EmbeddingBatch, AgnosticEmbeddingError> {
            panic!("PanicEmbedder must not be called: dataset run took the no-filter fast path");
        }
    }

    struct FnJudge<F> {
        label: String,
        f: F,
    }

    impl<F> JudgeProvider for FnJudge<F>
    where
        F: Fn(JudgeRequest) -> Result<String, AgnosticCompletionError> + Send + Sync,
    {
        fn label(&self) -> &str {
            &self.label
        }

        fn invoke<'a>(
            &'a self,
            request: JudgeRequest,
        ) -> BoxFuture<'a, Result<String, AgnosticCompletionError>> {
            let r = (self.f)(request);
            Box::pin(async move { r })
        }
    }

    fn rank_in_order(req: JudgeRequest) -> Result<String, AgnosticCompletionError> {
        let n = req.user_message.matches("[c").count();
        let ids: Vec<String> = (1..=n).map(|i| format!("c{i}")).collect();
        Ok(format!(
            "<reasoning>prefer c1 first</reasoning><ranking>{}</ranking>",
            ids.join(",")
        ))
    }

    fn ok_openai_planner(
        prompt: &str,
    ) -> Result<OpenAiCompletionCommand, Box<dyn std::error::Error + Send + Sync>> {
        Ok(OpenAiCompletionCommand {
            model: OpenAiModel::Gpt4oMini,
            system_prompt: None,
            messages: vec![OpenAiMessage {
                role: OpenAiRole::User,
                content: prompt.to_string(),
            }],
            max_tokens: Some(8),
            temperature: None,
        })
    }

    // ---------- builder validation ----------

    #[test]
    fn build_rejects_no_slots() {
        let result = DatasetBuilder::new(PanicEmbedder)
            .judge(FnJudge {
                label: "j".to_string(),
                f: rank_in_order,
            })
            .build();
        assert!(matches!(result, Err(DatasetBuildError::NoSlots)));
    }

    #[test]
    fn build_rejects_no_judges() {
        let client = OpenAiClient::new("k".to_string());
        let result = DatasetBuilder::new(PanicEmbedder)
            .slot(SlotTemplate::openai(client, "a", 1, ok_openai_planner))
            .build();
        assert!(matches!(result, Err(DatasetBuildError::NoJudges)));
    }

    #[test]
    fn build_rejects_zero_samples_slot() {
        let client = OpenAiClient::new("k".to_string());
        let result = DatasetBuilder::new(PanicEmbedder)
            .slot(SlotTemplate::openai(client, "slot-a", 0, ok_openai_planner))
            .judge(FnJudge {
                label: "j".to_string(),
                f: rank_in_order,
            })
            .build();
        match result {
            Err(DatasetBuildError::ZeroSamples {
                slot_index,
                model_label,
            }) => {
                assert_eq!(slot_index, 0);
                assert_eq!(model_label, "slot-a");
            }
            Err(other) => panic!("expected ZeroSamples, got {other:?}"),
            Ok(_) => panic!("expected ZeroSamples, builder unexpectedly succeeded"),
        }
    }

    // ---------- happy path ----------

    #[tokio::test]
    async fn happy_path_emits_one_outcome_per_prompt_in_original_index_order() {
        // Two model slots, each backed by its own mockito server. Three
        // prompts; embedding layout makes Centroid+max_n=2 pick indices 0
        // and 2. The skipped prompt (1) must appear in-place in the stream.
        let mut server_a = mockito::Server::new_async().await;
        let _mock_a = server_a
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_body(
                r#"{
                    "choices": [{"message": {"content": "alpha"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}
                }"#,
            )
            .expect_at_least(4) // 2 selected prompts × 2 samples
            .create_async()
            .await;

        let mut server_b = mockito::Server::new_async().await;
        let _mock_b = server_b
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_body(
                r#"{
                    "choices": [{"message": {"content": "bravo"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}
                }"#,
            )
            .expect_at_least(4)
            .create_async()
            .await;

        let client_a = OpenAiClient::new_with_base_url("k".to_string(), server_a.url());
        let client_b = OpenAiClient::new_with_base_url("k".to_string(), server_b.url());

        let embedder = TestEmbedder {
            vectors: vec![
                vec![1.0, 0.0, 0.0],
                vec![0.9, 0.1, 0.0],
                vec![0.0, 1.0, 0.0],
            ],
        };
        let prompts = vec!["p0".to_string(), "p1".to_string(), "p2".to_string()];

        let run = DatasetBuilder::new(embedder)
            .slot(SlotTemplate::openai(
                client_a,
                "slot-a",
                2,
                ok_openai_planner,
            ))
            .slot(SlotTemplate::openai(
                client_b,
                "slot-b",
                2,
                ok_openai_planner,
            ))
            .judge(FnJudge {
                label: "j".to_string(),
                f: rank_in_order,
            })
            .selection(SelectionStrategy::Centroid)
            .stop_condition(StopCondition::with_max_n(2))
            .build()
            .expect("build");

        let stream = run.execute(prompts).await.expect("execute");
        let outcomes: Vec<PromptOutcome> = stream.collect().await;

        assert_eq!(outcomes.len(), 3, "one outcome per input prompt");
        assert_eq!(outcomes[0].prompt_index(), 0);
        assert_eq!(outcomes[1].prompt_index(), 1);
        assert_eq!(outcomes[2].prompt_index(), 2);
        assert_eq!(outcomes[0].prompt(), "p0");
        assert_eq!(outcomes[1].prompt(), "p1");
        assert_eq!(outcomes[2].prompt(), "p2");

        // Index 1 is the in-place skip — not moved to the tail.
        match &outcomes[1] {
            PromptOutcome::Skipped { reason, .. } => {
                assert_eq!(*reason, SkipReason::NotSelectedByDiversification);
            }
            other => panic!("expected Skipped at index 1, got {other:?}"),
        }

        // Indices 0 and 2 both produce a Phase 2 winner.
        for i in [0_usize, 2] {
            match &outcomes[i] {
                PromptOutcome::Completed { outcome, .. } => {
                    let winner = outcome
                        .phase_two
                        .as_ref()
                        .and_then(|p2| p2.winner.as_ref())
                        .expect("phase 2 winner");
                    assert!(
                        winner.content == "alpha" || winner.content == "bravo",
                        "unexpected winner content: {:?}",
                        winner.content
                    );
                }
                other => panic!("expected Completed at index {i}, got {other:?}"),
            }
        }
    }

    // ---------- per-prompt failure continuation ----------

    #[tokio::test]
    async fn failing_planner_yields_failed_then_continues_with_later_prompts() {
        // No diversification → embedder is never called (PanicEmbedder).
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
            .expect_at_least(2) // 2 successful prompts × 1 sample
            .create_async()
            .await;

        let client = OpenAiClient::new_with_base_url("k".to_string(), server.url());

        // String error type to also exercise `E: Into<Box<dyn Error + Send + Sync>>`.
        let plan = |prompt: &str| -> Result<OpenAiCompletionCommand, String> {
            if prompt.contains("fail") {
                Err("planner says no".to_string())
            } else {
                Ok(OpenAiCompletionCommand {
                    model: OpenAiModel::Gpt4oMini,
                    system_prompt: None,
                    messages: vec![OpenAiMessage {
                        role: OpenAiRole::User,
                        content: prompt.to_string(),
                    }],
                    max_tokens: Some(8),
                    temperature: None,
                })
            }
        };

        let run = DatasetBuilder::new(PanicEmbedder)
            .slot(SlotTemplate::openai(client, "slot-a", 1, plan))
            .judge(FnJudge {
                label: "j".to_string(),
                f: rank_in_order,
            })
            .build()
            .expect("build");

        let prompts = vec![
            "fail-me".to_string(),
            "ok-one".to_string(),
            "ok-two".to_string(),
        ];

        let stream = run.execute(prompts).await.expect("execute");
        let outcomes: Vec<PromptOutcome> = stream.collect().await;

        assert_eq!(outcomes.len(), 3);
        assert_eq!(outcomes[0].prompt_index(), 0);
        assert_eq!(outcomes[1].prompt_index(), 1);
        assert_eq!(outcomes[2].prompt_index(), 2);

        match &outcomes[0] {
            PromptOutcome::Failed { error, prompt, .. } => {
                assert_eq!(prompt, "fail-me");
                let PromptRunError::SlotPlanning {
                    slot_index,
                    model_label,
                    source,
                } = error;
                assert_eq!(*slot_index, 0);
                assert_eq!(model_label, "slot-a");
                assert!(
                    source.to_string().contains("planner says no"),
                    "unexpected planner source: {source}"
                );
            }
            other => panic!("expected Failed at index 0, got {other:?}"),
        }
        assert!(
            matches!(outcomes[1], PromptOutcome::Completed { .. }),
            "later prompt must still run after a planner failure"
        );
        assert!(matches!(outcomes[2], PromptOutcome::Completed { .. }));
    }

    // ---------- JSONL writer ----------

    #[tokio::test]
    async fn write_jsonl_emits_one_row_per_outcome_with_winner_projection() {
        // Drive write_jsonl directly with canned outcomes — keeps this test
        // independent of orchestration / mocks.
        let outcomes = vec![
            PromptOutcome::Completed {
                prompt_index: 0,
                prompt: "p0".to_string(),
                outcome: ConsortiumOutcome {
                    phase_one: Vec::new(),
                    phase_two: Some(CrossModelPhaseOutcome {
                        candidates: Vec::new(),
                        judge_outcomes: Vec::new(),
                        aggregated: None,
                        winner: Some(PhaseTwoWinner {
                            model_index: 0,
                            provider: ProviderKind::OpenAi,
                            model_label: "slot-a".to_string(),
                            content: "the winning content".to_string(),
                        }),
                    }),
                },
            },
            PromptOutcome::Skipped {
                prompt_index: 1,
                prompt: "p1".to_string(),
                reason: SkipReason::NotSelectedByDiversification,
            },
            PromptOutcome::Failed {
                prompt_index: 2,
                prompt: "p2".to_string(),
                error: PromptRunError::SlotPlanning {
                    slot_index: 0,
                    model_label: "slot-a".to_string(),
                    source: "nope".into(),
                },
            },
        ];

        let stream = stream::iter(outcomes);
        let mut buf: Vec<u8> = Vec::new();
        write_jsonl(&mut buf, stream).await.expect("write");

        let text = String::from_utf8(buf).expect("utf-8");
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3, "one JSONL line per outcome");

        let r0: serde_json::Value = serde_json::from_str(lines[0]).expect("row 0 json");
        assert_eq!(r0["prompt_index"], 0);
        assert_eq!(r0["prompt"], "p0");
        assert_eq!(r0["status"]["kind"], "completed");
        assert_eq!(r0["status"]["winner"]["model_label"], "slot-a");
        assert_eq!(r0["status"]["winner"]["provider"], "openai");
        assert_eq!(r0["status"]["winner"]["content"], "the winning content");

        let r1: serde_json::Value = serde_json::from_str(lines[1]).expect("row 1 json");
        assert_eq!(r1["prompt_index"], 1);
        assert_eq!(r1["status"]["kind"], "skipped");
        assert_eq!(r1["status"]["reason"], "not_selected_by_diversification");

        let r2: serde_json::Value = serde_json::from_str(lines[2]).expect("row 2 json");
        assert_eq!(r2["prompt_index"], 2);
        assert_eq!(r2["status"]["kind"], "failed");
        assert!(
            r2["status"]["error"]
                .as_str()
                .unwrap()
                .contains("planner failed"),
            "unexpected serialized error: {}",
            r2["status"]["error"]
        );
    }

    #[tokio::test]
    async fn execute_returns_typed_error_when_embedder_yields_mixed_dimension_vectors() {
        // JaggedEmbedder returns row-0 with dim 3 and row-1 with dim 4 —
        // exactly the case `select_diverse` would panic on. The dataset
        // layer must catch it and surface a typed `DatasetRunError` instead.
        let client = OpenAiClient::new("k".to_string());
        let run = DatasetBuilder::new(JaggedEmbedder)
            .slot(SlotTemplate::openai(client, "slot-a", 1, ok_openai_planner))
            .judge(FnJudge {
                label: "j".to_string(),
                f: rank_in_order,
            })
            .stop_condition(StopCondition::with_max_n(2))
            .build()
            .expect("build");

        let err = run
            .execute(vec!["p0".to_string(), "p1".to_string()])
            .await
            .err()
            .expect("execute must reject a jagged embedding batch");

        match err {
            DatasetRunError::EmbeddingDimensionMismatch {
                row_index,
                expected,
                actual,
            } => {
                assert_eq!(row_index, 1);
                assert_eq!(expected, 3);
                assert_eq!(actual, 4);
            }
            other => panic!("expected EmbeddingDimensionMismatch, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_prompts_yields_empty_stream_without_calling_embedder() {
        // PanicEmbedder asserts no embed() call — proves the empty-input
        // short-circuit really skips embedding even when a non-trivial
        // StopCondition is configured.
        let client = OpenAiClient::new("k".to_string());
        let run = DatasetBuilder::new(PanicEmbedder)
            .slot(SlotTemplate::openai(client, "slot-a", 1, ok_openai_planner))
            .judge(FnJudge {
                label: "j".to_string(),
                f: rank_in_order,
            })
            .stop_condition(StopCondition::with_max_n(5))
            .build()
            .expect("build");

        let stream = run.execute(Vec::new()).await.expect("execute");
        let outcomes: Vec<PromptOutcome> = stream.collect().await;
        assert!(outcomes.is_empty());
    }
}
