use std::collections::{HashMap, HashSet};

use crate::AgnosticCompletionError;
use crate::ProviderKind;

/// A candidate completion submitted to the judge layer along with its source
/// metadata. Provider/model are kept here so callers can recover provenance after
/// judgment but are never sent to the judge model — the judge only sees the
/// neutral blind id.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub content: String,
    pub provider: ProviderKind,
    pub model: String,
}

/// Opaque, neutral identifier used in judge prompts and responses. Never carries
/// provider information.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlindId(String);

impl BlindId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for BlindId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A candidate as the judge will see it: only a blind id and the content.
#[derive(Debug, Clone)]
pub struct BlindCandidate {
    pub id: BlindId,
    pub content: String,
}

/// Assign sequential blind ids (`c1`, `c2`, ...) to each candidate. Returns the
/// blind-candidate view to send to judges and a reverse map from blind id to
/// the candidate's index in the original slice — provenance preserved without
/// leaking to the judge.
///
/// Sequential assignment is deterministic. Callers that need to defeat
/// position-based correlations between input order and provider identity should
/// shuffle `candidates` themselves before calling this function.
pub fn assign_blind_ids(
    candidates: &[Candidate],
) -> (Vec<BlindCandidate>, HashMap<BlindId, usize>) {
    let mut blind = Vec::with_capacity(candidates.len());
    let mut map = HashMap::with_capacity(candidates.len());
    for (i, candidate) in candidates.iter().enumerate() {
        let id = BlindId(format!("c{}", i + 1));
        map.insert(id.clone(), i);
        blind.push(BlindCandidate {
            id,
            content: candidate.content.clone(),
        });
    }
    (blind, map)
}

/// A judge's structured ranking output, parsed from the response payload.
#[derive(Debug, Clone)]
pub struct OrderedJudgement {
    /// Blind ids in the judge's preference order, best first. Length equals the
    /// number of candidates judged. No duplicates. No missing ids.
    pub ordered_ids: Vec<BlindId>,
    /// The reasoning text the judge produced inside the `<reasoning>` block,
    /// trimmed of surrounding whitespace.
    pub reasoning: String,
    /// The full raw response from the judge model, preserved for audit/debug.
    pub raw_response: String,
}

#[derive(Debug, thiserror::Error)]
pub enum JudgementParseError {
    #[error("judge response missing <reasoning>...</reasoning> block")]
    MissingReasoningTag,
    #[error("judge response missing <ranking>...</ranking> block")]
    MissingRankingTag,
    #[error("ranking block was empty")]
    EmptyRanking,
    #[error("ranking contained an unknown blind id: {id:?}")]
    UnknownId { id: String },
    #[error("ranking contained duplicate blind id: {id:?}")]
    DuplicateId { id: String },
    #[error("ranking was missing blind ids: {missing:?}")]
    MissingIds { missing: Vec<String> },
    #[error("judge response had non-whitespace text outside the required blocks: {snippet:?}")]
    ExtraneousText { snippet: String },
}

#[derive(Debug, thiserror::Error)]
pub enum JudgementError {
    #[error("provider call failed: {0}")]
    Provider(#[from] AgnosticCompletionError),
    #[error("judge response could not be parsed: {0}")]
    Parse(#[from] JudgementParseError),
}

/// Parse a raw judge response against the expected set of blind ids.
///
/// Whitespace around ids and inside the `<reasoning>` block is tolerated. The
/// `<ranking>` block must contain exactly the same set of ids as `expected_ids`,
/// each appearing exactly once. No duplicates. No unknowns. No empties.
///
/// The response must contain exactly two tagged blocks (`<reasoning>` then
/// `<ranking>`) with nothing but whitespace outside them. Prefatory text,
/// trailing chatter, or duplicate blocks are rejected as
/// [`JudgementParseError::ExtraneousText`] — the prompt forbids out-of-block
/// text, and accepting it would let judges silently break the contract.
pub fn parse_ordered_judgement(
    raw_response: &str,
    expected_ids: &HashSet<BlindId>,
) -> Result<OrderedJudgement, JudgementParseError> {
    let reasoning_span = locate_tag(raw_response, "reasoning")
        .ok_or(JudgementParseError::MissingReasoningTag)?;
    let ranking_span =
        locate_tag(raw_response, "ranking").ok_or(JudgementParseError::MissingRankingTag)?;

    // Reasoning must come before ranking, and the blocks must not overlap. If
    // either invariant fails, the duplicate / out-of-order block ends up in one
    // of the "outside" regions checked below and surfaces as ExtraneousText.
    let regions: [(usize, usize); 3] = [
        (0, reasoning_span.open_start),
        (reasoning_span.close_end, ranking_span.open_start),
        (ranking_span.close_end, raw_response.len()),
    ];
    for (start, end) in regions {
        if end < start {
            // Blocks overlap or ranking precedes reasoning — express as
            // extraneous text using the full slice from the earlier index.
            let lo = start.min(end);
            let hi = start.max(end);
            return Err(JudgementParseError::ExtraneousText {
                snippet: trimmed_snippet(&raw_response[lo..hi]),
            });
        }
        let region = &raw_response[start..end];
        if region.chars().any(|c| !c.is_whitespace()) {
            return Err(JudgementParseError::ExtraneousText {
                snippet: trimmed_snippet(region),
            });
        }
    }

    let reasoning = raw_response[reasoning_span.content_start..reasoning_span.content_end]
        .trim()
        .to_string();
    let ranking_block =
        &raw_response[ranking_span.content_start..ranking_span.content_end];

    let ids: Vec<String> = ranking_block
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if ids.is_empty() {
        return Err(JudgementParseError::EmptyRanking);
    }

    let mut seen: HashSet<String> = HashSet::with_capacity(ids.len());
    for id in &ids {
        let blind = BlindId(id.clone());
        if !expected_ids.contains(&blind) {
            return Err(JudgementParseError::UnknownId { id: id.clone() });
        }
        if !seen.insert(id.clone()) {
            return Err(JudgementParseError::DuplicateId { id: id.clone() });
        }
    }

    let provided: HashSet<&str> = ids.iter().map(|s| s.as_str()).collect();
    let mut missing: Vec<String> = expected_ids
        .iter()
        .filter(|id| !provided.contains(id.as_str()))
        .map(|id| id.as_str().to_string())
        .collect();
    if !missing.is_empty() {
        missing.sort();
        return Err(JudgementParseError::MissingIds { missing });
    }

    Ok(OrderedJudgement {
        ordered_ids: ids.into_iter().map(BlindId).collect(),
        reasoning,
        raw_response: raw_response.to_string(),
    })
}

/// Locked judge system prompt. The prompt:
/// - instructs the model to use `<reasoning>...</reasoning>` then
///   `<ranking>id1,id2,...</ranking>`,
/// - forbids ties,
/// - requires every candidate to appear exactly once,
/// - explicitly tells the judge it will not know which provider/model produced
///   any candidate.
pub const JUDGE_SYSTEM_PROMPT: &str = r#"You are a judge evaluating multiple candidate completions to the same prompt. Each candidate carries a short blind identifier (e.g., c1, c2, c3) and nothing else — you do not know which model or organization produced any of them.

Respond in two blocks, in this exact order, with no text outside the blocks:

1. <reasoning>...</reasoning> — a single block where you discuss the candidates by their blind identifiers, weighing their merits and weaknesses.
2. <ranking>id1,id2,...</ranking> — a single block listing every blind identifier exactly once, comma-separated, ordered from best to worst.

Rules:
- Every candidate identifier must appear in the ranking exactly once. No omissions, no duplicates.
- No ties. Each position is a strict preference.
- Do not speculate about the source model or organization. Judge the content alone.
- Do not write anything outside the two tagged blocks."#;

/// The pair of pieces a caller needs to wire into their chosen judge provider:
/// the system prompt and a user message containing the formatted candidates.
#[derive(Debug)]
pub struct JudgeRequest {
    pub system_prompt: &'static str,
    pub user_message: String,
}

/// Format the candidates into the user message body for a judge call. Only
/// blind ids are emitted — no provider or model attribution.
pub fn build_judge_user_message(candidates: &[BlindCandidate]) -> String {
    let mut buf = String::new();
    buf.push_str("Candidates to judge:\n\n");
    for candidate in candidates {
        buf.push_str(&format!("[{}]\n{}\n\n", candidate.id, candidate.content));
    }
    buf.push_str("Provide your reasoning and ranking in the required format.\n");
    buf
}

/// Invoke a judge model and parse its structured response.
///
/// `invoke_judge` receives a [`JudgeRequest`] containing the locked system
/// prompt and the assembled user message. The caller wires the request into
/// whichever provider serves as the judge and returns the raw response text.
///
/// Provider errors propagate as [`JudgementError::Provider`]. Parse / validation
/// failures propagate as [`JudgementError::Parse`].
pub async fn judge_rank<F, Fut>(
    candidates: &[BlindCandidate],
    invoke_judge: F,
) -> Result<OrderedJudgement, JudgementError>
where
    F: FnOnce(JudgeRequest) -> Fut,
    Fut: std::future::Future<Output = Result<String, AgnosticCompletionError>>,
{
    let request = JudgeRequest {
        system_prompt: JUDGE_SYSTEM_PROMPT,
        user_message: build_judge_user_message(candidates),
    };
    let raw = invoke_judge(request).await?;
    let expected: HashSet<BlindId> = candidates.iter().map(|c| c.id.clone()).collect();
    let judgement = parse_ordered_judgement(&raw, &expected)?;
    Ok(judgement)
}

#[derive(Debug, Clone)]
pub struct AggregatedRanking {
    /// Blind ids ordered by aggregate Borda score, best first. Ties broken
    /// lexicographically on the blind id string.
    pub ordered_ids: Vec<BlindId>,
    pub scores: HashMap<BlindId, u32>,
}

/// Aggregate multiple judges' orderings into a single ranking using Borda count:
/// the i-th-place candidate in a length-N ranking earns `N - i` points. Total
/// scores are summed across judges and ordered descending.
///
/// Tie-break: when total scores are equal, blind ids are ordered ascending
/// lexicographically (string comparison). This is deterministic but not numeric
/// — `c10` sorts before `c2`. Callers needing a different tie-break can
/// post-process `AggregatedRanking.scores`.
///
/// An empty input produces an empty aggregation.
///
/// # Panics
///
/// Panics if `rankings` contains rankings over different blind-id sets. Every
/// ranking must be over the same candidate universe — mixing rankings from
/// different judging sessions silently combines scores across unrelated
/// candidate pools, which is a programmer error at the call site (orchestrator
/// passed mismatched inputs). The invariant is checked against `rankings[0]`.
pub fn aggregate_rankings(rankings: &[OrderedJudgement]) -> AggregatedRanking {
    if rankings.is_empty() {
        return AggregatedRanking {
            ordered_ids: Vec::new(),
            scores: HashMap::new(),
        };
    }

    let first_universe: HashSet<&BlindId> = rankings[0].ordered_ids.iter().collect();
    for (i, ranking) in rankings.iter().enumerate().skip(1) {
        let universe: HashSet<&BlindId> = ranking.ordered_ids.iter().collect();
        if universe != first_universe {
            let expected = sorted_id_strings(&first_universe);
            let actual = sorted_id_strings(&universe);
            panic!(
                "aggregate_rankings: all rankings must be over the same blind-id set; \
                 rankings[0] has {expected:?} but rankings[{i}] has {actual:?}"
            );
        }
    }

    let mut scores: HashMap<BlindId, u32> = HashMap::new();

    for ranking in rankings {
        let n = ranking.ordered_ids.len() as u32;
        for (rank, id) in ranking.ordered_ids.iter().enumerate() {
            let points = n - rank as u32;
            *scores.entry(id.clone()).or_insert(0) += points;
        }
    }

    let mut ordered: Vec<(BlindId, u32)> =
        scores.iter().map(|(k, v)| (k.clone(), *v)).collect();
    ordered.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    AggregatedRanking {
        ordered_ids: ordered.into_iter().map(|(id, _)| id).collect(),
        scores,
    }
}

struct TagSpan {
    open_start: usize,
    content_start: usize,
    content_end: usize,
    close_end: usize,
}

fn locate_tag(haystack: &str, tag: &str) -> Option<TagSpan> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let open_start = haystack.find(&open)?;
    let content_start = open_start + open.len();
    let close_offset = haystack[content_start..].find(&close)?;
    let content_end = content_start + close_offset;
    let close_end = content_end + close.len();
    Some(TagSpan {
        open_start,
        content_start,
        content_end,
        close_end,
    })
}

fn sorted_id_strings(ids: &HashSet<&BlindId>) -> Vec<String> {
    let mut v: Vec<&BlindId> = ids.iter().copied().collect();
    v.sort();
    v.into_iter().map(|id| id.as_str().to_string()).collect()
}

fn trimmed_snippet(s: &str) -> String {
    let trimmed = s.trim();
    if trimmed.chars().count() <= 80 {
        trimmed.to_string()
    } else {
        let truncated: String = trimmed.chars().take(80).collect();
        format!("{truncated}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected(ids: &[&str]) -> HashSet<BlindId> {
        ids.iter().map(|s| BlindId::new(*s)).collect()
    }

    fn make_candidate(content: &str, provider: ProviderKind, model: &str) -> Candidate {
        Candidate {
            content: content.to_string(),
            provider,
            model: model.to_string(),
        }
    }

    #[test]
    fn assign_blind_ids_assigns_sequential_ids_preserving_provenance() {
        let candidates = vec![
            make_candidate("first", ProviderKind::OpenAi, "gpt-4o"),
            make_candidate("second", ProviderKind::Claude, "claude-opus-4-7"),
            make_candidate("third", ProviderKind::OpenAi, "gpt-4o"),
        ];

        let (blind, map) = assign_blind_ids(&candidates);

        assert_eq!(blind.len(), 3);
        assert_eq!(blind[0].id, BlindId::new("c1"));
        assert_eq!(blind[0].content, "first");
        assert_eq!(blind[1].id, BlindId::new("c2"));
        assert_eq!(blind[2].id, BlindId::new("c3"));

        // Provenance map: each blind id resolves back to the original index, so
        // two same-provider candidates with different content stay distinguishable.
        assert_eq!(map[&BlindId::new("c1")], 0);
        assert_eq!(map[&BlindId::new("c2")], 1);
        assert_eq!(map[&BlindId::new("c3")], 2);
        assert_eq!(candidates[map[&BlindId::new("c1")]].provider, ProviderKind::OpenAi);
        assert_eq!(candidates[map[&BlindId::new("c3")]].provider, ProviderKind::OpenAi);
        assert_eq!(candidates[map[&BlindId::new("c1")]].content, "first");
        assert_eq!(candidates[map[&BlindId::new("c3")]].content, "third");
    }

    #[test]
    fn build_judge_user_message_uses_only_blind_ids() {
        let blind = vec![
            BlindCandidate {
                id: BlindId::new("c1"),
                content: "alpha".to_string(),
            },
            BlindCandidate {
                id: BlindId::new("c2"),
                content: "beta".to_string(),
            },
        ];
        let msg = build_judge_user_message(&blind);
        assert!(msg.contains("[c1]"));
        assert!(msg.contains("alpha"));
        assert!(msg.contains("[c2]"));
        assert!(msg.contains("beta"));
        for forbidden in ["claude", "openai", "gemini", "anthropic", "cohere"] {
            assert!(
                !msg.to_lowercase().contains(forbidden),
                "user message must not leak provider names: contained {forbidden:?}"
            );
        }
    }

    #[test]
    fn judge_system_prompt_does_not_mention_provider_names() {
        let lower = JUDGE_SYSTEM_PROMPT.to_lowercase();
        for forbidden in ["claude", "openai", "gemini", "anthropic", "cohere", "gpt"] {
            assert!(
                !lower.contains(forbidden),
                "JUDGE_SYSTEM_PROMPT must stay provider-neutral: contained {forbidden:?}"
            );
        }
    }

    #[test]
    fn parse_valid_response_extracts_ordering_and_reasoning() {
        let raw = "<reasoning>c1 is the most thorough, c2 is brief but accurate, c3 hallucinated</reasoning><ranking>c1,c2,c3</ranking>";
        let result = parse_ordered_judgement(raw, &expected(&["c1", "c2", "c3"])).expect("parse");

        assert_eq!(
            result.ordered_ids,
            vec![BlindId::new("c1"), BlindId::new("c2"), BlindId::new("c3")]
        );
        assert!(result.reasoning.contains("c1 is the most thorough"));
        assert_eq!(result.raw_response, raw);
    }

    #[test]
    fn parse_tolerates_whitespace_around_ids_and_in_reasoning() {
        let raw = "<reasoning>\n  c2 has the cleanest structure  \n</reasoning>\n<ranking>  c2 ,  c1 ,c3  </ranking>";
        let result = parse_ordered_judgement(raw, &expected(&["c1", "c2", "c3"])).expect("parse");
        assert_eq!(
            result.ordered_ids,
            vec![BlindId::new("c2"), BlindId::new("c1"), BlindId::new("c3")]
        );
        assert_eq!(result.reasoning, "c2 has the cleanest structure");
    }

    #[test]
    fn parse_missing_reasoning_tag_errors() {
        let raw = "<ranking>c1,c2</ranking>";
        let err = parse_ordered_judgement(raw, &expected(&["c1", "c2"])).expect_err("missing");
        assert!(matches!(err, JudgementParseError::MissingReasoningTag));
    }

    #[test]
    fn parse_missing_ranking_tag_errors() {
        let raw = "<reasoning>thoughts</reasoning>";
        let err = parse_ordered_judgement(raw, &expected(&["c1", "c2"])).expect_err("missing");
        assert!(matches!(err, JudgementParseError::MissingRankingTag));
    }

    #[test]
    fn parse_empty_ranking_errors() {
        let raw = "<reasoning>x</reasoning><ranking>   </ranking>";
        let err = parse_ordered_judgement(raw, &expected(&["c1"])).expect_err("empty");
        assert!(matches!(err, JudgementParseError::EmptyRanking));
    }

    #[test]
    fn parse_unknown_id_errors() {
        let raw = "<reasoning>x</reasoning><ranking>c1,c99,c2</ranking>";
        let err =
            parse_ordered_judgement(raw, &expected(&["c1", "c2", "c3"])).expect_err("unknown");
        match err {
            JudgementParseError::UnknownId { id } => assert_eq!(id, "c99"),
            other => panic!("expected UnknownId, got {other:?}"),
        }
    }

    #[test]
    fn parse_duplicate_id_errors() {
        let raw = "<reasoning>x</reasoning><ranking>c1,c2,c1</ranking>";
        let err = parse_ordered_judgement(raw, &expected(&["c1", "c2", "c3"]))
            .expect_err("duplicate");
        match err {
            JudgementParseError::DuplicateId { id } => assert_eq!(id, "c1"),
            other => panic!("expected DuplicateId, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_prefatory_text() {
        let raw = "Sure! Here is my judgment.\n<reasoning>x</reasoning><ranking>c1,c2</ranking>";
        let err = parse_ordered_judgement(raw, &expected(&["c1", "c2"])).expect_err("prefatory");
        match err {
            JudgementParseError::ExtraneousText { snippet } => {
                assert!(snippet.contains("Sure"), "snippet was: {snippet}");
            }
            other => panic!("expected ExtraneousText, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_trailing_text() {
        let raw = "<reasoning>x</reasoning><ranking>c1,c2</ranking>\nDone!";
        let err = parse_ordered_judgement(raw, &expected(&["c1", "c2"])).expect_err("trailing");
        match err {
            JudgementParseError::ExtraneousText { snippet } => {
                assert!(snippet.contains("Done"), "snippet was: {snippet}");
            }
            other => panic!("expected ExtraneousText, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_text_between_blocks() {
        let raw = "<reasoning>x</reasoning>then<ranking>c1,c2</ranking>";
        let err = parse_ordered_judgement(raw, &expected(&["c1", "c2"])).expect_err("between");
        match err {
            JudgementParseError::ExtraneousText { snippet } => {
                assert!(snippet.contains("then"), "snippet was: {snippet}");
            }
            other => panic!("expected ExtraneousText, got {other:?}"),
        }
    }

    #[test]
    fn parse_rejects_duplicate_ranking_block() {
        // A second <ranking> block ends up in the post-first-ranking outside region.
        let raw = "<reasoning>x</reasoning><ranking>c1,c2</ranking><ranking>c2,c1</ranking>";
        let err = parse_ordered_judgement(raw, &expected(&["c1", "c2"])).expect_err("duplicate");
        assert!(matches!(err, JudgementParseError::ExtraneousText { .. }));
    }

    #[test]
    fn parse_rejects_ranking_before_reasoning() {
        // The ranking block ends up in the "before reasoning" outside region.
        let raw = "<ranking>c1,c2</ranking><reasoning>x</reasoning>";
        let err =
            parse_ordered_judgement(raw, &expected(&["c1", "c2"])).expect_err("out of order");
        assert!(matches!(err, JudgementParseError::ExtraneousText { .. }));
    }

    #[test]
    fn parse_allows_whitespace_around_blocks() {
        let raw = "\n\n<reasoning>x</reasoning>\n\n<ranking>c1,c2</ranking>\n";
        let result =
            parse_ordered_judgement(raw, &expected(&["c1", "c2"])).expect("whitespace ok");
        assert_eq!(
            result.ordered_ids,
            vec![BlindId::new("c1"), BlindId::new("c2")]
        );
    }

    #[test]
    fn parse_missing_id_errors_with_full_missing_list() {
        let raw = "<reasoning>x</reasoning><ranking>c1</ranking>";
        let err = parse_ordered_judgement(raw, &expected(&["c1", "c2", "c3"]))
            .expect_err("missing");
        match err {
            JudgementParseError::MissingIds { missing } => {
                assert_eq!(missing, vec!["c2".to_string(), "c3".to_string()]);
            }
            other => panic!("expected MissingIds, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn judge_rank_returns_parsed_judgement() {
        let candidates = vec![
            BlindCandidate {
                id: BlindId::new("c1"),
                content: "hello".to_string(),
            },
            BlindCandidate {
                id: BlindId::new("c2"),
                content: "world".to_string(),
            },
        ];

        let result = judge_rank(&candidates, |req| async move {
            assert_eq!(req.system_prompt, JUDGE_SYSTEM_PROMPT);
            assert!(req.user_message.contains("[c1]"));
            assert!(req.user_message.contains("hello"));
            assert!(req.user_message.contains("[c2]"));
            Ok(
                "<reasoning>c2 is shorter but accurate</reasoning><ranking>c2,c1</ranking>"
                    .to_string(),
            )
        })
        .await
        .expect("judge");

        assert_eq!(
            result.ordered_ids,
            vec![BlindId::new("c2"), BlindId::new("c1")]
        );
    }

    #[tokio::test]
    async fn judge_rank_propagates_provider_error() {
        let candidates = vec![BlindCandidate {
            id: BlindId::new("c1"),
            content: "x".to_string(),
        }];
        let result = judge_rank(&candidates, |_| async {
            Err(AgnosticCompletionError::Auth {
                provider: ProviderKind::OpenAi,
                message: Some("bad key".to_string()),
            })
        })
        .await;

        assert!(matches!(result, Err(JudgementError::Provider(_))));
    }

    #[tokio::test]
    async fn judge_rank_propagates_parse_error() {
        let candidates = vec![
            BlindCandidate {
                id: BlindId::new("c1"),
                content: "x".to_string(),
            },
            BlindCandidate {
                id: BlindId::new("c2"),
                content: "y".to_string(),
            },
        ];
        let result = judge_rank(&candidates, |_| async {
            Ok("malformed response with no tags".to_string())
        })
        .await;

        assert!(matches!(
            result,
            Err(JudgementError::Parse(
                JudgementParseError::MissingReasoningTag
            ))
        ));
    }

    fn ranking(ids: &[&str]) -> OrderedJudgement {
        OrderedJudgement {
            ordered_ids: ids.iter().map(|s| BlindId::new(*s)).collect(),
            reasoning: String::new(),
            raw_response: String::new(),
        }
    }

    #[test]
    fn aggregate_single_ranking_yields_borda_scores() {
        let agg = aggregate_rankings(&[ranking(&["c1", "c2", "c3"])]);
        // length 3 → c1 gets 3, c2 gets 2, c3 gets 1.
        assert_eq!(agg.scores[&BlindId::new("c1")], 3);
        assert_eq!(agg.scores[&BlindId::new("c2")], 2);
        assert_eq!(agg.scores[&BlindId::new("c3")], 1);
        assert_eq!(
            agg.ordered_ids,
            vec![BlindId::new("c1"), BlindId::new("c2"), BlindId::new("c3")]
        );
    }

    #[test]
    fn aggregate_unanimous_judges_preserves_order() {
        let agg = aggregate_rankings(&[
            ranking(&["c1", "c2", "c3"]),
            ranking(&["c1", "c2", "c3"]),
            ranking(&["c1", "c2", "c3"]),
        ]);
        assert_eq!(
            agg.ordered_ids,
            vec![BlindId::new("c1"), BlindId::new("c2"), BlindId::new("c3")]
        );
        assert_eq!(agg.scores[&BlindId::new("c1")], 9);
        assert_eq!(agg.scores[&BlindId::new("c2")], 6);
        assert_eq!(agg.scores[&BlindId::new("c3")], 3);
    }

    #[test]
    fn aggregate_disagreeing_judges_uses_sum_of_borda_scores() {
        // Judge A: c1, c2, c3 → c1=3, c2=2, c3=1
        // Judge B: c2, c3, c1 → c2=3, c3=2, c1=1
        // Sum:               → c1=4, c2=5, c3=3 → winner c2
        let agg = aggregate_rankings(&[
            ranking(&["c1", "c2", "c3"]),
            ranking(&["c2", "c3", "c1"]),
        ]);
        assert_eq!(
            agg.ordered_ids,
            vec![BlindId::new("c2"), BlindId::new("c1"), BlindId::new("c3")]
        );
    }

    #[test]
    fn aggregate_ties_broken_lexicographically() {
        // Two judges with mirrored orderings — every id ties.
        // Judge A: c1, c2, c3 → c1=3, c2=2, c3=1
        // Judge B: c3, c2, c1 → c3=3, c2=2, c1=1
        // Sum:               → c1=4, c2=4, c3=4. Lexicographic: c1, c2, c3.
        let agg = aggregate_rankings(&[
            ranking(&["c1", "c2", "c3"]),
            ranking(&["c3", "c2", "c1"]),
        ]);
        assert_eq!(
            agg.ordered_ids,
            vec![BlindId::new("c1"), BlindId::new("c2"), BlindId::new("c3")]
        );
        assert_eq!(agg.scores[&BlindId::new("c1")], 4);
        assert_eq!(agg.scores[&BlindId::new("c2")], 4);
        assert_eq!(agg.scores[&BlindId::new("c3")], 4);
    }

    #[test]
    fn aggregate_empty_input_yields_empty_aggregation() {
        let agg = aggregate_rankings(&[]);
        assert!(agg.ordered_ids.is_empty());
        assert!(agg.scores.is_empty());
    }

    #[test]
    #[should_panic(expected = "all rankings must be over the same blind-id set")]
    fn aggregate_panics_on_inconsistent_blind_id_universe() {
        let agg_input = [
            ranking(&["c1", "c2", "c3"]),
            ranking(&["c1", "c2", "c4"]), // c3 vs c4 — different universe
        ];
        let _ = aggregate_rankings(&agg_input);
    }

    #[test]
    #[should_panic(expected = "all rankings must be over the same blind-id set")]
    fn aggregate_panics_on_partial_overlap_universe() {
        let agg_input = [
            ranking(&["c1", "c2", "c3"]),
            ranking(&["c1", "c2"]), // missing c3 — different universe
        ];
        let _ = aggregate_rankings(&agg_input);
    }
}
