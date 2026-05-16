//! Two-phase consortium orchestrator for a single prompt.
//!
//! The orchestrator:
//!
//! 1. Fans out `N` completion samples per configured model slot (Phase 1
//!    sampling) using the existing M1 [`crate::multi_infer`] machinery — every
//!    `(slot, sample)` pair gets a distinct outer `input_index`, so callers can
//!    correlate every returned [`crate::ProviderAttempt`] back to its origin
//!    even when multiple samples reuse the same provider client.
//! 2. Ranks each slot's surviving samples with the configured judges, picks a
//!    per-model winner (Phase 1 judging).
//! 3. Ranks the per-model winners across models, picks the overall best
//!    (Phase 2).
//!
//! Failures are first-class outcomes at every layer:
//!
//! - Per-sample provider failures stay in [`ModelPhaseOutcome::samples`] as
//!   `Err` [`crate::ProviderAttempt`]s. They are excluded from the candidate
//!   pool but never dropped from the record.
//! - Per-judge failures (provider error or parse error) are preserved in
//!   [`JudgeOutcome::result`]; aggregation simply skips them.
//! - A slot with zero surviving samples produces a [`ModelPhaseOutcome`] with
//!   `winner = None` — that is the slot's terminal failure shape, not a
//!   dropped row.
//! - Phase 2 proceeds over whichever slots produced a winner; a fully-failed
//!   model is absent from [`CrossModelPhaseOutcome::candidates`] but visible in
//!   [`ConsortiumOutcome::phase_one`] with `winner = None`.
//!
//! Streaming / hooks (`mpsc` channel, callback trait) are intentionally not in
//! this slice — the in-memory result shape comes first; streaming can be added
//! later as an alternative surface over the same orchestration.
//!
//! ## Concurrency (M5b)
//!
//! Phase 1 sampling has always been a single concurrent [`crate::multi_infer`]
//! fan-out across every `(slot, sample)` pair. M5b extends concurrency to the
//! parts that were sequential under M5a:
//!
//! - **Per-slot judges in Phase 1.** The judges configured for a single slot
//!   are invoked concurrently against that slot's blind candidate set via a
//!   local [`futures::stream::FuturesUnordered`]. Results are reordered into
//!   original judge order before being written to [`ModelPhaseOutcome::judge_outcomes`].
//! - **Slot-level Phase 1.** The per-slot [`phase_one_for_slot`] futures
//!   themselves run concurrently inside another [`futures::stream::FuturesUnordered`],
//!   reordered by `slot_index` before becoming [`ConsortiumOutcome::phase_one`].
//! - **Phase 2 judges.** The cross-model judges run concurrently with the same
//!   reorder pattern.
//!
//! Concurrency stays single-task / cooperative — no [`tokio::spawn`], no `Send`
//! refactor. The orchestrator's future remains non-`Send` (it captures
//! `&dyn JudgeProvider` across `.await` points, and `dyn JudgeProvider` does
//! not pick up its supertrait `Sync` bound on the trait object itself), which
//! matches the existing dataset-stream contract from M6b.
//!
//! Public surface is unchanged: [`consortium_completion`] returns the same
//! [`ConsortiumOutcome`] shape and preserves every order, provenance, and
//! failure invariant from M5a even when internal futures complete out of
//! order.

use futures::FutureExt;
use futures::future::{BoxFuture, LocalBoxFuture};
use futures::stream::{FuturesUnordered, StreamExt};

use crate::judge::{
    AggregatedRanking, BlindCandidate, BlindId, Candidate, JudgeRequest, JudgementError,
    OrderedJudgement, aggregate_rankings, assign_blind_ids, judge_rank,
};
use crate::{
    AgnosticCompletionError, AgnosticCompletionOutput, AiCompletionInputs,
    CompletionOutputChunk, MultiAiCompletionInputs, ProviderAttempt, ProviderKind, multi_infer,
};

/// A judge model the orchestrator can invoke. Implementors wire whichever real
/// (or canned) provider serves as a judge.
///
/// The trait lives here rather than in [`crate::judge`] on purpose: the M4
/// judge primitives stay closure-based and provider-agnostic. The orchestrator
/// needs to call the same judge once per slot in Phase 1 and once in Phase 2,
/// which is awkward to model with [`FnOnce`] closures alone. A small trait
/// captures the "callable multiple times" shape without forcing a specific
/// provider path into the judge module itself.
///
/// `label` is purely for provenance in [`JudgeOutcome::judge_label`]; it is
/// never sent to the judge model.
pub trait JudgeProvider: Send + Sync {
    fn label(&self) -> &str;
    fn invoke<'a>(
        &'a self,
        request: JudgeRequest,
    ) -> BoxFuture<'a, Result<String, AgnosticCompletionError>>;
}

/// One configured model entry: which provider call to make, how many samples
/// to draw from it in Phase 1, and the label that follows winners through
/// provenance.
///
/// `samples` is the per-slot fan-out width. The orchestrator pushes `samples`
/// independent copies of `input` into a single [`multi_infer`] call — exactly
/// the duplicate-`input_index` use case M1 was designed for.
#[derive(Clone)]
pub struct ConsortiumSlot<'a> {
    pub input: AiCompletionInputs<'a>,
    pub model_label: String,
    pub samples: usize,
}

/// One sampling attempt within a slot, with its slot-local sample index
/// preserved so callers can trace winners back to the originating attempt.
#[derive(Debug)]
pub struct SampleAttempt {
    /// Position within the slot, `0..ConsortiumSlot::samples`.
    pub sample_index: usize,
    pub attempt: ProviderAttempt,
}

/// One Phase 1 candidate as the judges saw it, paired with the
/// orchestrator-assigned blind id and a back-pointer to the originating
/// sample in [`ModelPhaseOutcome::samples`].
///
/// Populated whenever a slot produced at least one successful sample,
/// including the singleton short-circuit case. That way any preserved
/// [`BlindId`] surfaced through [`JudgeOutcome`] or [`AggregatedRanking`] —
/// winning or not — is externally resolvable back to a concrete
/// [`SampleAttempt`] without callers relying on hidden ordering conventions.
#[derive(Debug, Clone)]
pub struct JudgedSample {
    pub blind_id: BlindId,
    pub sample_index: usize,
    pub content: String,
}

/// One judge's outcome for one ranking session. Failures are preserved here
/// rather than discarded so callers can audit which judges contributed to
/// aggregation and which did not.
#[derive(Debug)]
pub struct JudgeOutcome {
    pub judge_label: String,
    pub result: Result<OrderedJudgement, JudgementError>,
}

/// Resolved Phase 1 winner for a slot. `sample_index` indexes back into
/// [`ModelPhaseOutcome::samples`] so callers can recover the exact originating
/// attempt (timing, retries, raw output) without needing to re-clone it here.
#[derive(Debug, Clone)]
pub struct PhaseOneWinner {
    pub sample_index: usize,
    pub content: String,
    pub provider: ProviderKind,
    pub model_label: String,
}

/// Outcome of Phase 1 for a single slot.
///
/// `provider` is derived from the slot's [`AiCompletionInputs`] and is always
/// populated — even when every sample failed and the slot's
/// `ProviderAttempt`s could not supply it.
#[derive(Debug)]
pub struct ModelPhaseOutcome {
    pub model_label: String,
    pub provider: ProviderKind,
    pub samples: Vec<SampleAttempt>,
    /// Explicit blind-id provenance for the candidates the Phase 1 judges (would
    /// have) seen, in `BlindCandidate` order. Each entry maps a [`BlindId`] back
    /// to a [`SampleAttempt`] index in [`Self::samples`]. Empty only when no
    /// sample succeeded.
    pub judged: Vec<JudgedSample>,
    pub judge_outcomes: Vec<JudgeOutcome>,
    pub aggregated: Option<AggregatedRanking>,
    pub winner: Option<PhaseOneWinner>,
}

/// One Phase 2 candidate as the cross-model judges saw it. `blind_id` is the
/// orchestrator-assigned identifier paired into every [`JudgeRequest`];
/// `model_index` indexes back into [`ConsortiumOutcome::phase_one`] so the
/// full per-model history is one hop away from the cross-model record;
/// `content` is the exact text the judges received, equal to the originating
/// [`PhaseOneWinner::content`].
///
/// Populated for every slot that produced a [`PhaseOneWinner`], including the
/// singleton short-circuit case. That way any preserved [`BlindId`] from a
/// Phase 2 [`JudgeOutcome`] or [`AggregatedRanking`] resolves back to a
/// specific model slot without callers relying on hidden ordering
/// conventions.
#[derive(Debug, Clone)]
pub struct CrossModelCandidate {
    pub blind_id: BlindId,
    pub model_index: usize,
    pub provider: ProviderKind,
    pub model_label: String,
    pub content: String,
}

/// Resolved Phase 2 winner. `model_index` traces back through
/// [`ConsortiumOutcome::phase_one`] to the winning sample.
#[derive(Debug, Clone)]
pub struct PhaseTwoWinner {
    pub model_index: usize,
    pub provider: ProviderKind,
    pub model_label: String,
    pub content: String,
}

/// Outcome of the cross-model phase. `candidates` lists every model slot that
/// produced a Phase 1 winner, in `phase_one` order. `judge_outcomes` is empty
/// only when the cross-model phase short-circuited a single-candidate case;
/// otherwise it contains every judge's outcome, successes and failures alike.
#[derive(Debug)]
pub struct CrossModelPhaseOutcome {
    pub candidates: Vec<CrossModelCandidate>,
    pub judge_outcomes: Vec<JudgeOutcome>,
    pub aggregated: Option<AggregatedRanking>,
    pub winner: Option<PhaseTwoWinner>,
}

/// Top-level outcome of one consortium prompt.
///
/// `phase_two` is `None` only when no Phase 1 slot produced a winner — i.e.,
/// every configured model failed end-to-end. A `Some(CrossModelPhaseOutcome)`
/// whose `winner` is `None` means winners existed but every cross-model judge
/// failed; the slot-level winners remain visible via [`Self::phase_one`].
#[derive(Debug)]
pub struct ConsortiumOutcome {
    pub phase_one: Vec<ModelPhaseOutcome>,
    pub phase_two: Option<CrossModelPhaseOutcome>,
}

/// Run the two-phase consortium pipeline for a single prompt.
///
/// `slots` describes which model calls to make and how many samples to draw
/// from each. `judges` are invoked once per slot in Phase 1 (when the slot has
/// ≥2 surviving samples) and once in Phase 2 (when ≥2 slots have winners).
///
/// Singleton candidate sets in either phase short-circuit judging: the lone
/// candidate is the winner and no judge calls are made. The resulting
/// [`ModelPhaseOutcome`] or [`CrossModelPhaseOutcome`] reports `aggregated =
/// None` and an empty `judge_outcomes` in those cases. The candidate list
/// itself disambiguates "no judges ran because trivial" from "every judge
/// failed".
///
/// Empty `slots` returns an outcome with empty `phase_one` and `phase_two =
/// None`. Empty `judges` is accepted but degenerate: every non-singleton phase
/// will have `aggregated = None` and `winner = None` because no judge can vote.
pub async fn consortium_completion<'a>(
    slots: &'a [ConsortiumSlot<'a>],
    judges: &'a [&'a dyn JudgeProvider],
) -> ConsortiumOutcome {
    let total_samples: usize = slots.iter().map(|s| s.samples).sum();
    let mut fan_inputs: Vec<AiCompletionInputs<'a>> = Vec::with_capacity(total_samples);
    let mut routing: Vec<(usize, usize)> = Vec::with_capacity(total_samples);
    for (slot_index, slot) in slots.iter().enumerate() {
        for sample_index in 0..slot.samples {
            fan_inputs.push(slot.input);
            routing.push((slot_index, sample_index));
        }
    }

    let attempts = if fan_inputs.is_empty() {
        Vec::new()
    } else {
        let multi = MultiAiCompletionInputs::new(&fan_inputs);
        multi_infer(&multi).await
    };

    let mut by_slot: Vec<Vec<SampleAttempt>> = (0..slots.len()).map(|_| Vec::new()).collect();
    for attempt in attempts {
        let (slot_index, sample_index) = routing[attempt.input_index];
        by_slot[slot_index].push(SampleAttempt {
            sample_index,
            attempt,
        });
    }
    for bag in by_slot.iter_mut() {
        bag.sort_by_key(|s| s.sample_index);
    }

    // Phase 1 across slots runs concurrently inside a local FuturesUnordered;
    // outputs are reordered by slot_index so `phase_one` is deterministic
    // regardless of completion order. No `tokio::spawn` — keeps the
    // orchestrator's future single-task / non-`Send`, matching the dataset
    // stream contract.
    let mut slot_fanout: FuturesUnordered<LocalBoxFuture<'_, (usize, ModelPhaseOutcome)>> =
        FuturesUnordered::new();
    for (slot_index, slot) in slots.iter().enumerate() {
        let slot_samples = std::mem::take(&mut by_slot[slot_index]);
        let provider = slot.input.provider();
        slot_fanout.push(
            async move {
                let outcome = phase_one_for_slot(slot, provider, slot_samples, judges).await;
                (slot_index, outcome)
            }
            .boxed_local(),
        );
    }

    let mut phase_one_buf: Vec<Option<ModelPhaseOutcome>> =
        (0..slots.len()).map(|_| None).collect();
    while let Some((slot_index, outcome)) = slot_fanout.next().await {
        phase_one_buf[slot_index] = Some(outcome);
    }
    let phase_one: Vec<ModelPhaseOutcome> = phase_one_buf
        .into_iter()
        .map(|o| o.expect("every slot future writes its slot_index slot exactly once"))
        .collect();

    let phase_two = phase_two_outcome(&phase_one, judges).await;

    ConsortiumOutcome {
        phase_one,
        phase_two,
    }
}

async fn phase_one_for_slot<'a>(
    slot: &ConsortiumSlot<'a>,
    provider: ProviderKind,
    slot_samples: Vec<SampleAttempt>,
    judges: &'a [&'a dyn JudgeProvider],
) -> ModelPhaseOutcome {
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut sample_index_by_candidate: Vec<usize> = Vec::new();
    for sa in &slot_samples {
        if let Ok(output) = &sa.attempt.result {
            candidates.push(Candidate {
                content: collect_text_content(output),
                provider: sa.attempt.provider,
                model: slot.model_label.clone(),
            });
            sample_index_by_candidate.push(sa.sample_index);
        }
    }

    if candidates.is_empty() {
        return ModelPhaseOutcome {
            model_label: slot.model_label.clone(),
            provider,
            samples: slot_samples,
            judged: Vec::new(),
            judge_outcomes: Vec::new(),
            aggregated: None,
            winner: None,
        };
    }

    // Assign blind ids up front so the mapping is recorded in the public
    // outcome even when the singleton short-circuit skips judging — callers
    // can resolve any preserved BlindId back to a sample regardless.
    let (blind, blind_to_cand_idx) = assign_blind_ids(&candidates);
    let judged: Vec<JudgedSample> = blind
        .iter()
        .enumerate()
        .map(|(i, bc)| JudgedSample {
            blind_id: bc.id.clone(),
            sample_index: sample_index_by_candidate[i],
            content: bc.content.clone(),
        })
        .collect();

    if candidates.len() == 1 {
        let winner = PhaseOneWinner {
            sample_index: sample_index_by_candidate[0],
            content: candidates[0].content.clone(),
            provider,
            model_label: slot.model_label.clone(),
        };
        return ModelPhaseOutcome {
            model_label: slot.model_label.clone(),
            provider,
            samples: slot_samples,
            judged,
            judge_outcomes: Vec::new(),
            aggregated: None,
            winner: Some(winner),
        };
    }

    let (judge_outcomes, successful) = invoke_judges_in_parallel(&blind, judges).await;

    let aggregated = if successful.is_empty() {
        None
    } else {
        Some(aggregate_rankings(&successful))
    };

    let winner = aggregated
        .as_ref()
        .and_then(|agg| agg.ordered_ids.first())
        .map(|blind_id| {
            let cand_idx = blind_to_cand_idx[blind_id];
            PhaseOneWinner {
                sample_index: sample_index_by_candidate[cand_idx],
                content: candidates[cand_idx].content.clone(),
                provider,
                model_label: slot.model_label.clone(),
            }
        });

    ModelPhaseOutcome {
        model_label: slot.model_label.clone(),
        provider,
        samples: slot_samples,
        judged,
        judge_outcomes,
        aggregated,
        winner,
    }
}

/// Invoke every judge against `blind` concurrently inside a local
/// [`FuturesUnordered`], then reorder results by `judge_index` so the returned
/// [`JudgeOutcome`] vec mirrors the caller's original `judges` slice. Also
/// returns the cloned-out successful [`OrderedJudgement`]s in the same input
/// order, ready for [`aggregate_rankings`].
///
/// Concurrency is bounded by `judges.len()` — there is no explicit cap because
/// judge counts are configured up-front and stay small (single digits in
/// practice). No `tokio::spawn`; the futures run cooperatively in the calling
/// task. Result determinism comes from the reorder buffer, not from completion
/// order.
async fn invoke_judges_in_parallel<'a>(
    blind: &[BlindCandidate],
    judges: &'a [&'a dyn JudgeProvider],
) -> (Vec<JudgeOutcome>, Vec<OrderedJudgement>) {
    let mut in_flight: FuturesUnordered<LocalBoxFuture<'_, (usize, JudgeOutcome)>> =
        FuturesUnordered::new();
    for (judge_index, judge) in judges.iter().enumerate() {
        let label = judge.label().to_string();
        in_flight.push(
            async move {
                let result = judge_rank(blind, |req| judge.invoke(req)).await;
                (
                    judge_index,
                    JudgeOutcome {
                        judge_label: label,
                        result,
                    },
                )
            }
            .boxed_local(),
        );
    }

    let mut buf: Vec<Option<JudgeOutcome>> = (0..judges.len()).map(|_| None).collect();
    while let Some((idx, outcome)) = in_flight.next().await {
        buf[idx] = Some(outcome);
    }
    let judge_outcomes: Vec<JudgeOutcome> = buf
        .into_iter()
        .map(|o| o.expect("every judge future writes its judge_index slot exactly once"))
        .collect();
    let successful: Vec<OrderedJudgement> = judge_outcomes
        .iter()
        .filter_map(|jo| jo.result.as_ref().ok().cloned())
        .collect();
    (judge_outcomes, successful)
}

async fn phase_two_outcome<'a>(
    phase_one: &[ModelPhaseOutcome],
    judges: &'a [&'a dyn JudgeProvider],
) -> Option<CrossModelPhaseOutcome> {
    let winners: Vec<(usize, &PhaseOneWinner)> = phase_one
        .iter()
        .enumerate()
        .filter_map(|(i, po)| po.winner.as_ref().map(|w| (i, w)))
        .collect();

    if winners.is_empty() {
        return None;
    }

    // Assign blind ids up front so cross-model provenance (blind_id ->
    // model_index) is recorded even when the singleton short-circuit skips
    // judging. Callers can resolve any preserved BlindId from a Phase 2
    // judge result back to a slot through `candidates`.
    let candidates_for_judging: Vec<Candidate> = winners
        .iter()
        .map(|(_, w)| Candidate {
            content: w.content.clone(),
            provider: w.provider,
            model: w.model_label.clone(),
        })
        .collect();
    let (blind, blind_to_cand_idx) = assign_blind_ids(&candidates_for_judging);

    let cross_candidates: Vec<CrossModelCandidate> = winners
        .iter()
        .enumerate()
        .map(|(i, (idx, w))| CrossModelCandidate {
            blind_id: blind[i].id.clone(),
            model_index: *idx,
            provider: w.provider,
            model_label: w.model_label.clone(),
            content: w.content.clone(),
        })
        .collect();

    if winners.len() == 1 {
        let (model_index, w) = winners[0];
        return Some(CrossModelPhaseOutcome {
            candidates: cross_candidates,
            judge_outcomes: Vec::new(),
            aggregated: None,
            winner: Some(PhaseTwoWinner {
                model_index,
                provider: w.provider,
                model_label: w.model_label.clone(),
                content: w.content.clone(),
            }),
        });
    }

    let (judge_outcomes, successful) = invoke_judges_in_parallel(&blind, judges).await;

    let aggregated = if successful.is_empty() {
        None
    } else {
        Some(aggregate_rankings(&successful))
    };

    let winner = aggregated
        .as_ref()
        .and_then(|agg| agg.ordered_ids.first())
        .map(|blind_id| {
            let cand_idx = blind_to_cand_idx[blind_id];
            let (model_index, w) = winners[cand_idx];
            PhaseTwoWinner {
                model_index,
                provider: w.provider,
                model_label: w.model_label.clone(),
                content: w.content.clone(),
            }
        });

    Some(CrossModelPhaseOutcome {
        candidates: cross_candidates,
        judge_outcomes,
        aggregated,
        winner,
    })
}

fn collect_text_content(output: &AgnosticCompletionOutput) -> String {
    let mut buf = String::new();
    for chunk in &output.chunks {
        if let CompletionOutputChunk::Text(text) = chunk {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(text);
        }
    }
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::judge::{BlindId, JudgementParseError};
    use crate::{
        AiCompletionInputs, OpenAiClient, OpenAiCompletionCommand, OpenAiMessage, OpenAiModel,
        OpenAiRole, ProviderKind,
    };

    /// Test-only [`JudgeProvider`] backed by a synchronous closure. Each call
    /// produces a canned response; closures can vary their output based on the
    /// [`JudgeRequest`] (e.g., by inspecting which blind ids are present).
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
            let result = (self.f)(request);
            Box::pin(async move { result })
        }
    }

    fn ok_command(content: &str) -> OpenAiCompletionCommand {
        OpenAiCompletionCommand {
            model: OpenAiModel::Gpt4oMini,
            system_prompt: None,
            messages: vec![OpenAiMessage {
                role: OpenAiRole::User,
                content: content.to_string(),
            }],
            max_tokens: Some(8),
            temperature: None,
        }
    }

    #[tokio::test]
    async fn happy_path_two_phase_picks_a_winner_traceable_to_originating_sample() {
        // Two slots, two samples each. Slot A returns "alpha", slot B returns
        // "bravo". Each slot is backed by its own mockito server so the two
        // slots produce visibly distinct content.
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
            .expect_at_least(2)
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
            .expect_at_least(2)
            .create_async()
            .await;

        let client_a = OpenAiClient::new_with_base_url("k".to_string(), server_a.url());
        let client_b = OpenAiClient::new_with_base_url("k".to_string(), server_b.url());
        let cmd_a = ok_command("a");
        let cmd_b = ok_command("b");

        let slots = vec![
            ConsortiumSlot {
                input: AiCompletionInputs::OpenAi(&client_a, &cmd_a),
                model_label: "openai-a".to_string(),
                samples: 2,
            },
            ConsortiumSlot {
                input: AiCompletionInputs::OpenAi(&client_b, &cmd_b),
                model_label: "openai-b".to_string(),
                samples: 2,
            },
        ];

        // Two judges. Both rank "c1 first" in every session — for a 2-candidate
        // session this means: Phase 1 slot A picks sample_index=0; slot B picks
        // sample_index=0; Phase 2 picks slot A (the first slot in `phase_one`).
        let always_c1_then_rest = |req: JudgeRequest| -> Result<String, AgnosticCompletionError> {
            // Compose a ranking by enumerating each c-id in order — c1,c2 for
            // 2 candidates, c1,c2,c3 for 3, etc. The exact number is implicit
            // in the user message but we only need to support the sizes this
            // test produces (2).
            let n = req.user_message.matches("[c").count();
            let mut ids = Vec::with_capacity(n);
            for i in 1..=n {
                ids.push(format!("c{i}"));
            }
            Ok(format!(
                "<reasoning>c1 is the strongest, lexicographic order beyond that</reasoning><ranking>{}</ranking>",
                ids.join(",")
            ))
        };

        let j1 = FnJudge {
            label: "j1".to_string(),
            f: always_c1_then_rest,
        };
        let j2 = FnJudge {
            label: "j2".to_string(),
            f: always_c1_then_rest,
        };
        let judges: Vec<&dyn JudgeProvider> = vec![&j1, &j2];

        let outcome = consortium_completion(&slots, &judges).await;

        // Both slots produced 2 successful samples and both judges contributed.
        assert_eq!(outcome.phase_one.len(), 2);
        for (slot_index, po) in outcome.phase_one.iter().enumerate() {
            assert_eq!(po.provider, ProviderKind::OpenAi, "slot {slot_index}");
            assert_eq!(po.samples.len(), 2, "slot {slot_index}");
            for (i, sa) in po.samples.iter().enumerate() {
                assert_eq!(sa.sample_index, i);
                assert!(sa.attempt.result.is_ok(), "slot {slot_index} sample {i}");
            }
            assert_eq!(po.judge_outcomes.len(), 2);
            assert!(
                po.judge_outcomes.iter().all(|jo| jo.result.is_ok()),
                "all judges should succeed in the happy path"
            );
            assert!(po.aggregated.is_some());
        }

        // Phase 1 winners trace back to sample_index=0 via the c1 preference.
        let winner_a = outcome.phase_one[0].winner.as_ref().expect("slot A winner");
        assert_eq!(winner_a.sample_index, 0);
        assert_eq!(winner_a.content, "alpha");
        assert_eq!(winner_a.model_label, "openai-a");
        // The PhaseOneWinner.sample_index resolves to a real SampleAttempt with
        // a real Ok result — provenance survives end-to-end.
        let backref_a = &outcome.phase_one[0].samples[winner_a.sample_index];
        assert!(backref_a.attempt.result.is_ok());

        let winner_b = outcome.phase_one[1].winner.as_ref().expect("slot B winner");
        assert_eq!(winner_b.sample_index, 0);
        assert_eq!(winner_b.content, "bravo");
        assert_eq!(winner_b.model_label, "openai-b");

        // Phase 1 blind-id provenance is exposed: every successful sample
        // appears in `judged` with its blind id, and any preserved BlindId
        // from a JudgeOutcome (winning OR non-winning) resolves back to a
        // concrete SampleAttempt through `samples[sample_index]`.
        let slot_a = &outcome.phase_one[0];
        assert_eq!(slot_a.judged.len(), 2);
        let j1_ranking_a = slot_a.judge_outcomes[0]
            .result
            .as_ref()
            .expect("j1 ranking for slot A");
        let non_winning_blind_a = j1_ranking_a.ordered_ids[1].clone();
        let losing_judged = slot_a
            .judged
            .iter()
            .find(|j| j.blind_id == non_winning_blind_a)
            .expect("non-winning blind id resolves through judged");
        // The non-winning blind id should not be the winner's blind id —
        // judged[0] (sample_index=0) corresponds to c1 because assignment is
        // sequential.
        assert_ne!(losing_judged.sample_index, winner_a.sample_index);
        let losing_attempt = &slot_a.samples[losing_judged.sample_index];
        assert_eq!(losing_attempt.sample_index, losing_judged.sample_index);
        assert!(losing_attempt.attempt.result.is_ok());
        // Content recorded in `judged` matches what the judges actually saw.
        assert_eq!(losing_judged.content, "alpha");

        // Phase 2: two candidates, both judges rank c1 first → slot A wins.
        let phase_two = outcome.phase_two.expect("phase 2 should run");
        assert_eq!(phase_two.candidates.len(), 2);
        assert_eq!(phase_two.judge_outcomes.len(), 2);
        assert!(
            phase_two.judge_outcomes.iter().all(|jo| jo.result.is_ok()),
            "happy-path phase 2 judges should all succeed"
        );
        let phase_two_winner = phase_two.winner.as_ref().expect("phase 2 winner");
        assert_eq!(phase_two_winner.model_index, 0);
        assert_eq!(phase_two_winner.content, "alpha");
        assert_eq!(phase_two_winner.model_label, "openai-a");

        // The aggregated ranking is over the cross-model blind ids — first id
        // resolves through the candidate slice back to phase_one[0].
        let agg = phase_two.aggregated.as_ref().expect("phase 2 aggregation");
        assert_eq!(agg.ordered_ids.first(), Some(&BlindId::new("c1")));

        // Cross-model blind-id provenance: a non-winning BlindId from a
        // Phase 2 judge result must resolve back to a model_index through
        // `candidates`, not via hidden ordering. Pick the second id from j1's
        // cross-model ranking.
        let j1_cross = phase_two.judge_outcomes[0]
            .result
            .as_ref()
            .expect("j1 cross-model ranking");
        let non_winning_blind_p2 = j1_cross.ordered_ids[1].clone();
        let losing_cross = phase_two
            .candidates
            .iter()
            .find(|c| c.blind_id == non_winning_blind_p2)
            .expect("non-winning Phase 2 blind id resolves through candidates");
        assert_ne!(losing_cross.model_index, phase_two_winner.model_index);
        assert_eq!(losing_cross.model_label, "openai-b");
        assert_eq!(losing_cross.content, "bravo");
        // And model_index points back into phase_one as advertised.
        let losing_phase_one = &outcome.phase_one[losing_cross.model_index];
        assert_eq!(losing_phase_one.model_label, "openai-b");
    }

    #[tokio::test(start_paused = true)]
    async fn partial_failure_preserves_failed_attempts_and_failed_judges() {
        // Slot A: 2 samples, both 503 — every sample fails after retries.
        // Slot B: 2 samples, both succeed.
        // Judge j1 succeeds for slot B's phase 1; judge j2 returns a provider
        // error for slot B's phase 1. Slot A skips judging (no candidates).
        let mut server_a = mockito::Server::new_async().await;
        let _mock_a = server_a
            .mock("POST", "/v1/chat/completions")
            .with_status(503)
            .with_body(r#"{"error":{"message":"upstream busy"}}"#)
            .expect_at_least(6) // 2 samples * (1 + 2 retries) = 6
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
            .expect_at_least(2)
            .create_async()
            .await;

        let client_a = OpenAiClient::new_with_base_url("k".to_string(), server_a.url());
        let client_b = OpenAiClient::new_with_base_url("k".to_string(), server_b.url());
        let cmd_a = ok_command("a");
        let cmd_b = ok_command("b");

        let slots = vec![
            ConsortiumSlot {
                input: AiCompletionInputs::OpenAi(&client_a, &cmd_a),
                model_label: "openai-a".to_string(),
                samples: 2,
            },
            ConsortiumSlot {
                input: AiCompletionInputs::OpenAi(&client_b, &cmd_b),
                model_label: "openai-b".to_string(),
                samples: 2,
            },
        ];

        let always_c1_then_rest = |req: JudgeRequest| -> Result<String, AgnosticCompletionError> {
            let n = req.user_message.matches("[c").count();
            let mut ids = Vec::with_capacity(n);
            for i in 1..=n {
                ids.push(format!("c{i}"));
            }
            Ok(format!(
                "<reasoning>c1 wins</reasoning><ranking>{}</ranking>",
                ids.join(",")
            ))
        };
        let always_err = |_: JudgeRequest| -> Result<String, AgnosticCompletionError> {
            Err(AgnosticCompletionError::Auth {
                provider: ProviderKind::OpenAi,
                message: Some("bad judge key".to_string()),
            })
        };

        let j1 = FnJudge {
            label: "j1".to_string(),
            f: always_c1_then_rest,
        };
        let j2 = FnJudge {
            label: "j2".to_string(),
            f: always_err,
        };
        let judges: Vec<&dyn JudgeProvider> = vec![&j1, &j2];

        let outcome = consortium_completion(&slots, &judges).await;

        // Slot A: 2 failed sample attempts are preserved, no judging ran
        // (empty candidate set), no winner.
        let slot_a = &outcome.phase_one[0];
        assert_eq!(slot_a.model_label, "openai-a");
        assert_eq!(slot_a.provider, ProviderKind::OpenAi);
        assert_eq!(slot_a.samples.len(), 2, "failed attempts must be retained");
        for sa in &slot_a.samples {
            match &sa.attempt.result {
                Err(AgnosticCompletionError::ServerError { status, .. }) => {
                    assert_eq!(*status, 503)
                }
                other => panic!("expected ServerError, got {other:?}"),
            }
        }
        assert!(slot_a.judge_outcomes.is_empty());
        assert!(slot_a.aggregated.is_none());
        assert!(slot_a.winner.is_none());

        // Slot B: 2 successful samples; both judges' outcomes preserved (one
        // success, one failure). j1 still wins phase 1.
        let slot_b = &outcome.phase_one[1];
        assert_eq!(slot_b.samples.len(), 2);
        assert!(slot_b.samples.iter().all(|s| s.attempt.result.is_ok()));
        assert_eq!(slot_b.judge_outcomes.len(), 2, "both judges recorded");

        let j1_out = slot_b
            .judge_outcomes
            .iter()
            .find(|jo| jo.judge_label == "j1")
            .expect("j1 outcome");
        assert!(j1_out.result.is_ok());

        let j2_out = slot_b
            .judge_outcomes
            .iter()
            .find(|jo| jo.judge_label == "j2")
            .expect("j2 outcome must be preserved as Err");
        match &j2_out.result {
            Err(JudgementError::Provider(AgnosticCompletionError::Auth { message, .. })) => {
                assert_eq!(message.as_deref(), Some("bad judge key"));
            }
            other => panic!("expected JudgementError::Provider(Auth), got {other:?}"),
        }
        // Sanity: parse-error variant exists and we did not collapse j2's
        // failure into it.
        assert!(!matches!(
            j2_out.result,
            Err(JudgementError::Parse(JudgementParseError::MissingReasoningTag))
        ));

        let winner_b = slot_b.winner.as_ref().expect("slot B has a winner");
        assert_eq!(winner_b.content, "bravo");
        assert_eq!(winner_b.sample_index, 0);

        // Phase 2 short-circuits with one surviving slot: candidate list has
        // the one survivor, no judges invoked, winner is trivially slot B.
        let phase_two = outcome.phase_two.expect("phase 2 with one survivor");
        assert_eq!(phase_two.candidates.len(), 1);
        assert_eq!(phase_two.candidates[0].model_index, 1);
        assert!(phase_two.judge_outcomes.is_empty());
        assert!(phase_two.aggregated.is_none());
        let p2_winner = phase_two.winner.expect("phase 2 winner");
        assert_eq!(p2_winner.model_index, 1);
        assert_eq!(p2_winner.model_label, "openai-b");
        assert_eq!(p2_winner.content, "bravo");
    }

    // ---------- M5b: parallel judge fan-out ----------

    /// Test-only [`JudgeProvider`] that pins each invocation at a shared
    /// [`tokio::sync::Barrier`]. The barrier only releases once exactly
    /// `Barrier::new(n)` calls are concurrently suspended — direct evidence
    /// that the orchestrator keeps that many judges in flight at once. The
    /// caller chooses the barrier size to match its expected concurrency.
    /// `record` is the judge's label, pushed in the order each invocation
    /// crosses the barrier so a test can correlate completion order with
    /// emit order.
    struct BarrierJudge {
        label: String,
        barrier: std::sync::Arc<tokio::sync::Barrier>,
        active: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        max_seen: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        completion_order: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl JudgeProvider for BarrierJudge {
        fn label(&self) -> &str {
            &self.label
        }

        fn invoke<'a>(
            &'a self,
            request: JudgeRequest,
        ) -> BoxFuture<'a, Result<String, AgnosticCompletionError>> {
            use std::sync::atomic::Ordering;
            let barrier = self.barrier.clone();
            let active = self.active.clone();
            let max_seen = self.max_seen.clone();
            let completion_order = self.completion_order.clone();
            let label = self.label.clone();
            Box::pin(async move {
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                max_seen.fetch_max(now, Ordering::SeqCst);
                // Suspends until the barrier has released exactly `barrier`
                // judge calls — the central concurrency-proof pivot.
                barrier.wait().await;
                active.fetch_sub(1, Ordering::SeqCst);
                completion_order.lock().unwrap().push(label.clone());

                let n = request.user_message.matches("[c").count();
                let ids: Vec<String> = (1..=n).map(|i| format!("c{i}")).collect();
                Ok(format!(
                    "<reasoning>r</reasoning><ranking>{}</ranking>",
                    ids.join(",")
                ))
            })
        }
    }

    /// Judges within a single Phase 1 slot run concurrently — proven by a
    /// `tokio::sync::Barrier` sized to the judge count, which would deadlock
    /// if judges were invoked sequentially. After the barrier releases, the
    /// returned `judge_outcomes` must still be in original input order
    /// `[j1, j2, j3]` regardless of which judge's barrier wait resolved
    /// first.
    #[tokio::test]
    async fn phase_one_judges_run_in_parallel_within_a_slot() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex};
        use std::time::Duration;

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
            .expect_at_least(2)
            .create_async()
            .await;
        let client = OpenAiClient::new_with_base_url("k".to_string(), server.url());
        let cmd = ok_command("p");

        let slots = vec![ConsortiumSlot {
            input: AiCompletionInputs::OpenAi(&client, &cmd),
            model_label: "slot-a".to_string(),
            samples: 2,
        }];

        let parallelism = 3;
        let barrier = Arc::new(tokio::sync::Barrier::new(parallelism));
        let active = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let completion_order: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        let j1 = BarrierJudge {
            label: "j1".to_string(),
            barrier: barrier.clone(),
            active: active.clone(),
            max_seen: max_seen.clone(),
            completion_order: completion_order.clone(),
        };
        let j2 = BarrierJudge {
            label: "j2".to_string(),
            barrier: barrier.clone(),
            active: active.clone(),
            max_seen: max_seen.clone(),
            completion_order: completion_order.clone(),
        };
        let j3 = BarrierJudge {
            label: "j3".to_string(),
            barrier,
            active: active.clone(),
            max_seen: max_seen.clone(),
            completion_order: completion_order.clone(),
        };
        let judges: Vec<&dyn JudgeProvider> = vec![&j1, &j2, &j3];

        let outcome = tokio::time::timeout(
            Duration::from_secs(10),
            consortium_completion(&slots, &judges),
        )
        .await
        .expect("orchestrator hung — judges likely ran sequentially and the barrier deadlocked");

        assert_eq!(
            max_seen.load(Ordering::SeqCst),
            parallelism,
            "all judges should have been concurrently in flight at once"
        );
        assert_eq!(active.load(Ordering::SeqCst), 0);

        // judge_outcomes preserves the caller's input order regardless of
        // which judge completed the barrier-wait first.
        let slot_a = &outcome.phase_one[0];
        let labels: Vec<&str> = slot_a
            .judge_outcomes
            .iter()
            .map(|jo| jo.judge_label.as_str())
            .collect();
        assert_eq!(
            labels,
            vec!["j1", "j2", "j3"],
            "judge_outcomes must be in original input order even when futures complete out of order"
        );
        for jo in &slot_a.judge_outcomes {
            assert!(jo.result.is_ok(), "all parallel judges should succeed");
        }
    }

    /// Phase 2 cross-model judges run concurrently with the same
    /// barrier-pinning pattern, and the cross-model `judge_outcomes` are in
    /// original judge order.
    #[tokio::test]
    async fn phase_two_judges_run_in_parallel_with_preserved_order() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex};
        use std::time::Duration;

        // Two slots, single sample each, single completion content so each
        // slot trivially produces a Phase 1 winner without judges (singleton
        // short-circuit). Phase 2 then judges both winners across models.
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
            .expect_at_least(1)
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
            .expect_at_least(1)
            .create_async()
            .await;

        let client_a = OpenAiClient::new_with_base_url("k".to_string(), server_a.url());
        let client_b = OpenAiClient::new_with_base_url("k".to_string(), server_b.url());
        let cmd_a = ok_command("a");
        let cmd_b = ok_command("b");

        let slots = vec![
            ConsortiumSlot {
                input: AiCompletionInputs::OpenAi(&client_a, &cmd_a),
                model_label: "openai-a".to_string(),
                samples: 1,
            },
            ConsortiumSlot {
                input: AiCompletionInputs::OpenAi(&client_b, &cmd_b),
                model_label: "openai-b".to_string(),
                samples: 1,
            },
        ];

        let parallelism = 3;
        let barrier = Arc::new(tokio::sync::Barrier::new(parallelism));
        let active = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let completion_order: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        let j1 = BarrierJudge {
            label: "j1".to_string(),
            barrier: barrier.clone(),
            active: active.clone(),
            max_seen: max_seen.clone(),
            completion_order: completion_order.clone(),
        };
        let j2 = BarrierJudge {
            label: "j2".to_string(),
            barrier: barrier.clone(),
            active: active.clone(),
            max_seen: max_seen.clone(),
            completion_order: completion_order.clone(),
        };
        let j3 = BarrierJudge {
            label: "j3".to_string(),
            barrier,
            active: active.clone(),
            max_seen: max_seen.clone(),
            completion_order: completion_order.clone(),
        };
        let judges: Vec<&dyn JudgeProvider> = vec![&j1, &j2, &j3];

        let outcome = tokio::time::timeout(
            Duration::from_secs(10),
            consortium_completion(&slots, &judges),
        )
        .await
        .expect("orchestrator hung — Phase 2 judges likely ran sequentially");

        // Phase 1 slots use the singleton short-circuit (1 sample each, no
        // judges), so the only judge invocations happened in Phase 2.
        assert_eq!(
            max_seen.load(Ordering::SeqCst),
            parallelism,
            "Phase 2 should have run all judges concurrently"
        );
        let phase_two = outcome.phase_two.expect("phase 2 with two surviving slots");
        assert_eq!(phase_two.judge_outcomes.len(), 3);
        let labels: Vec<&str> = phase_two
            .judge_outcomes
            .iter()
            .map(|jo| jo.judge_label.as_str())
            .collect();
        assert_eq!(
            labels,
            vec!["j1", "j2", "j3"],
            "Phase 2 judge_outcomes must be in original input order"
        );
    }

    /// Slots within Phase 1 are judged concurrently — the slowest slot does
    /// not block faster slots from running their own judges. Even though
    /// Phase 1 sampling has always been concurrent via `multi_infer`, M5b
    /// makes the judging step concurrent across slots too. Determinism is
    /// proven by ordering `phase_one` strictly by slot_index.
    #[tokio::test(start_paused = true)]
    async fn phase_one_slots_run_concurrently_with_preserved_slot_order() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::Duration;

        // Three slots, single sample each. Slot 0's judge sleeps; slots 1
        // and 2 are fast. With sequential per-slot judging, slot 1 cannot
        // start before slot 0 finishes — so a "slot 1 finished before slot
        // 0 started judging" flag would never trip. Under M5b, the slot
        // futures run concurrently, so slot 1's judge starts and finishes
        // while slot 0 is still sleeping. We use `start_paused` plus
        // `tokio::time::sleep` so the slow slot's delay is virtual.
        let mut server_a = mockito::Server::new_async().await;
        let _ma = server_a
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_body(
                r#"{
                    "choices": [{"message": {"content": "S0-content"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}
                }"#,
            )
            .expect_at_least(2)
            .create_async()
            .await;
        let mut server_b = mockito::Server::new_async().await;
        let _mb = server_b
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_body(
                r#"{
                    "choices": [{"message": {"content": "S1-content"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}
                }"#,
            )
            .expect_at_least(2)
            .create_async()
            .await;
        let mut server_c = mockito::Server::new_async().await;
        let _mc = server_c
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_body(
                r#"{
                    "choices": [{"message": {"content": "S2-content"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}
                }"#,
            )
            .expect_at_least(2)
            .create_async()
            .await;

        let client_a = OpenAiClient::new_with_base_url("k".to_string(), server_a.url());
        let client_b = OpenAiClient::new_with_base_url("k".to_string(), server_b.url());
        let client_c = OpenAiClient::new_with_base_url("k".to_string(), server_c.url());
        let cmd_a = ok_command("a");
        let cmd_b = ok_command("b");
        let cmd_c = ok_command("c");

        let slots = vec![
            ConsortiumSlot {
                input: AiCompletionInputs::OpenAi(&client_a, &cmd_a),
                model_label: "slot-0".to_string(),
                samples: 2,
            },
            ConsortiumSlot {
                input: AiCompletionInputs::OpenAi(&client_b, &cmd_b),
                model_label: "slot-1".to_string(),
                samples: 2,
            },
            ConsortiumSlot {
                input: AiCompletionInputs::OpenAi(&client_c, &cmd_c),
                model_label: "slot-2".to_string(),
                samples: 2,
            },
        ];

        // `slot_0_finished` flips true at the end of slot 0's judging.
        // `slot_1_finished_before_slot_0` records whether slot 1's judging
        // completed while slot 0 was still pending — which can only happen
        // if slot-level Phase 1 work runs concurrently.
        let slot_0_finished = Arc::new(AtomicBool::new(false));
        let slot_1_finished_before_slot_0 = Arc::new(AtomicBool::new(false));

        struct OrderingJudge {
            label: String,
            slow_marker: String,
            slot1_marker: String,
            slow_delay: Duration,
            slot_0_finished: Arc<AtomicBool>,
            slot_1_finished_before_slot_0: Arc<AtomicBool>,
        }
        impl JudgeProvider for OrderingJudge {
            fn label(&self) -> &str {
                &self.label
            }
            fn invoke<'a>(
                &'a self,
                request: JudgeRequest,
            ) -> BoxFuture<'a, Result<String, AgnosticCompletionError>> {
                let slow_marker = self.slow_marker.clone();
                let slot1_marker = self.slot1_marker.clone();
                let slow_delay = self.slow_delay;
                let slot_0_finished = self.slot_0_finished.clone();
                let slot_1_finished_before_slot_0 =
                    self.slot_1_finished_before_slot_0.clone();
                Box::pin(async move {
                    let user = request.user_message;
                    let is_slow = user.contains(&slow_marker);
                    if is_slow {
                        tokio::time::sleep(slow_delay).await;
                    }
                    if user.contains(&slot1_marker)
                        && !slot_0_finished.load(Ordering::SeqCst)
                    {
                        slot_1_finished_before_slot_0.store(true, Ordering::SeqCst);
                    }
                    if is_slow {
                        slot_0_finished.store(true, Ordering::SeqCst);
                    }
                    let n = user.matches("[c").count();
                    let ids: Vec<String> = (1..=n).map(|i| format!("c{i}")).collect();
                    Ok(format!(
                        "<reasoning>r</reasoning><ranking>{}</ranking>",
                        ids.join(",")
                    ))
                })
            }
        }

        let j = OrderingJudge {
            label: "j".to_string(),
            slow_marker: "S0-content".to_string(),
            slot1_marker: "S1-content".to_string(),
            slow_delay: Duration::from_secs(60),
            slot_0_finished: slot_0_finished.clone(),
            slot_1_finished_before_slot_0: slot_1_finished_before_slot_0.clone(),
        };
        let judges: Vec<&dyn JudgeProvider> = vec![&j];

        let outcome = consortium_completion(&slots, &judges).await;

        assert!(
            slot_1_finished_before_slot_0.load(Ordering::SeqCst),
            "slot 1 should finish judging before slot 0 — proves slot futures run concurrently"
        );
        assert!(slot_0_finished.load(Ordering::SeqCst));

        // phase_one is in slot_index order regardless of completion order.
        assert_eq!(outcome.phase_one.len(), 3);
        let labels: Vec<&str> = outcome
            .phase_one
            .iter()
            .map(|po| po.model_label.as_str())
            .collect();
        assert_eq!(labels, vec!["slot-0", "slot-1", "slot-2"]);
        for po in &outcome.phase_one {
            assert!(po.winner.is_some());
        }
    }

    /// Failed judges must be preserved at their original judge_index even
    /// when judges run concurrently. Mixing Err and Ok results across the
    /// reorder buffer is the load-bearing case — a naive "push as they
    /// arrive" implementation would interleave them by completion order
    /// instead.
    #[tokio::test]
    async fn parallel_phase_one_preserves_failed_judges_at_their_input_index() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_body(
                r#"{
                    "choices": [{"message": {"content": "alpha"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}
                }"#,
            )
            .expect_at_least(2)
            .create_async()
            .await;
        let client = OpenAiClient::new_with_base_url("k".to_string(), server.url());
        let cmd = ok_command("p");

        let slots = vec![ConsortiumSlot {
            input: AiCompletionInputs::OpenAi(&client, &cmd),
            model_label: "slot-a".to_string(),
            samples: 2,
        }];

        // j1 succeeds, j2 fails, j3 succeeds, j4 fails. Under parallel
        // execution, futures can resolve in any order. The reorder buffer
        // must restore [Ok, Err, Ok, Err] by judge_index.
        let always_c1_then_rest = |req: JudgeRequest| -> Result<String, AgnosticCompletionError> {
            let n = req.user_message.matches("[c").count();
            let ids: Vec<String> = (1..=n).map(|i| format!("c{i}")).collect();
            Ok(format!(
                "<reasoning>c1 wins</reasoning><ranking>{}</ranking>",
                ids.join(",")
            ))
        };
        let always_err = |_: JudgeRequest| -> Result<String, AgnosticCompletionError> {
            Err(AgnosticCompletionError::Auth {
                provider: ProviderKind::OpenAi,
                message: Some("bad judge key".to_string()),
            })
        };

        let j1 = FnJudge {
            label: "j1".to_string(),
            f: always_c1_then_rest,
        };
        let j2 = FnJudge {
            label: "j2".to_string(),
            f: always_err,
        };
        let j3 = FnJudge {
            label: "j3".to_string(),
            f: always_c1_then_rest,
        };
        let j4 = FnJudge {
            label: "j4".to_string(),
            f: always_err,
        };
        let judges: Vec<&dyn JudgeProvider> = vec![&j1, &j2, &j3, &j4];

        let outcome = consortium_completion(&slots, &judges).await;
        let slot_a = &outcome.phase_one[0];

        assert_eq!(slot_a.judge_outcomes.len(), 4);
        let expected: [(&str, bool); 4] =
            [("j1", true), ("j2", false), ("j3", true), ("j4", false)];
        for (i, (label, is_ok)) in expected.iter().enumerate() {
            let jo = &slot_a.judge_outcomes[i];
            assert_eq!(jo.judge_label, *label, "judge_outcomes[{i}] label");
            assert_eq!(
                jo.result.is_ok(),
                *is_ok,
                "judge_outcomes[{i}] success state mismatched — reorder buffer should keep \
                 failed judges aligned with their original input position"
            );
            if !*is_ok {
                match &jo.result {
                    Err(JudgementError::Provider(AgnosticCompletionError::Auth {
                        message,
                        ..
                    })) => {
                        assert_eq!(message.as_deref(), Some("bad judge key"));
                    }
                    other => panic!("expected JudgementError::Provider(Auth), got {other:?}"),
                }
            }
        }

        // Aggregation still uses the two successful judges and picks a
        // winner, proving parallel fan-out did not break Borda.
        assert!(slot_a.aggregated.is_some());
        assert!(slot_a.winner.is_some());
    }
}
