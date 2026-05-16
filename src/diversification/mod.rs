#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Pick the candidate whose cosine similarity to the running centroid of
    /// already-selected vectors is lowest. The centroid is the elementwise mean
    /// of the selected set.
    Centroid,
    /// Pick the candidate whose cosine similarity to its nearest already-selected
    /// neighbor is lowest (i.e., farthest from anything already selected).
    FarthestPointSampling,
}

#[derive(Debug, Clone, Copy)]
pub struct StopCondition {
    /// Stop once the selected set reaches this many vectors. `None` means no cap.
    pub max_n: Option<usize>,
    /// Stop when the next candidate's similarity-to-group exceeds this threshold —
    /// i.e., the closest remaining input is already "too similar" to the selected
    /// set to be worth adding. For `Centroid`, the threshold is checked against
    /// the candidate's similarity to the centroid. For `FarthestPointSampling`,
    /// against the candidate's similarity to its nearest selected neighbor.
    /// `None` means no tripwire.
    pub similarity_tripwire: Option<f32>,
}

impl StopCondition {
    pub fn with_max_n(max_n: usize) -> Self {
        Self {
            max_n: Some(max_n),
            similarity_tripwire: None,
        }
    }

    pub fn with_tripwire(tripwire: f32) -> Self {
        Self {
            max_n: None,
            similarity_tripwire: Some(tripwire),
        }
    }
}

/// Select a diverse subset of `embeddings` and return the indices of the chosen
/// vectors in the order they were picked. The first index returned is always 0
/// (the seed picks the first input). Subsequent indices depend on `strategy`.
///
/// Empty input returns an empty selection.
///
/// # Panics
///
/// Panics if `embeddings` contains vectors of different lengths. All vectors must
/// share the same dimension — this is an invariant of any well-formed embedding
/// batch from a single model. Mixed dimensions indicate a programmer error at
/// the call site (mixing batches from different providers/models, or a corrupted
/// aggregation), not a runtime condition worth modeling as a typed error.
pub fn select_diverse(
    embeddings: &[Vec<f32>],
    strategy: SelectionStrategy,
    stop: StopCondition,
) -> Vec<usize> {
    if embeddings.is_empty() {
        return Vec::new();
    }

    let dim = embeddings[0].len();
    for (i, vec) in embeddings.iter().enumerate().skip(1) {
        assert_eq!(
            vec.len(),
            dim,
            "select_diverse: embeddings must share the same dimension; \
             embeddings[0] has dim {dim} but embeddings[{i}] has dim {}",
            vec.len()
        );
    }

    let total = embeddings.len();
    let cap = stop.max_n.unwrap_or(total).min(total);
    if cap == 0 {
        return Vec::new();
    }

    let mut selected: Vec<usize> = Vec::with_capacity(cap);
    selected.push(0);

    while selected.len() < cap {
        let next = match strategy {
            SelectionStrategy::Centroid => next_by_centroid(embeddings, &selected),
            SelectionStrategy::FarthestPointSampling => next_by_fps(embeddings, &selected),
        };

        let Some((idx, group_similarity)) = next else {
            break;
        };

        if let Some(tripwire) = stop.similarity_tripwire {
            if group_similarity > tripwire {
                break;
            }
        }

        selected.push(idx);
    }

    selected
}

/// Returns (index, similarity_to_centroid) of the best next candidate, or None
/// if no unselected vector remains.
fn next_by_centroid(embeddings: &[Vec<f32>], selected: &[usize]) -> Option<(usize, f32)> {
    let centroid = running_centroid(embeddings, selected)?;
    let mut best: Option<(usize, f32)> = None;
    for (i, vec) in embeddings.iter().enumerate() {
        if selected.contains(&i) {
            continue;
        }
        let sim = cosine_similarity(vec, &centroid);
        match best {
            Some((_, best_sim)) if sim >= best_sim => {}
            _ => best = Some((i, sim)),
        }
    }
    best
}

/// Returns (index, similarity_to_nearest_selected) of the best next candidate,
/// or None if no unselected vector remains.
fn next_by_fps(embeddings: &[Vec<f32>], selected: &[usize]) -> Option<(usize, f32)> {
    let mut best: Option<(usize, f32)> = None;
    for (i, vec) in embeddings.iter().enumerate() {
        if selected.contains(&i) {
            continue;
        }
        let nearest_sim = selected
            .iter()
            .map(|&s| cosine_similarity(vec, &embeddings[s]))
            .fold(f32::NEG_INFINITY, f32::max);
        match best {
            Some((_, best_nearest)) if nearest_sim >= best_nearest => {}
            _ => best = Some((i, nearest_sim)),
        }
    }
    best
}

fn running_centroid(embeddings: &[Vec<f32>], selected: &[usize]) -> Option<Vec<f32>> {
    if selected.is_empty() {
        return None;
    }
    let dim = embeddings[selected[0]].len();
    let mut sum = vec![0.0_f32; dim];
    for &i in selected {
        for (d, x) in embeddings[i].iter().enumerate() {
            sum[d] += x;
        }
    }
    let n = selected.len() as f32;
    for x in &mut sum {
        *x /= n;
    }
    Some(sum)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine similarity requires equal lengths");
    let mut dot = 0.0_f32;
    let mut mag_a = 0.0_f32;
    let mut mag_b = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        mag_a += x * x;
        mag_b += y * y;
    }
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a.sqrt() * mag_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Four 3D clusters of two points each: three axis-aligned (+x, +y, +z) and
    /// one anti-diagonal (-1,-1,-1)/√3 region. Index pairs (0,1) → cluster 0,
    /// (2,3) → cluster 1, (4,5) → cluster 2, (6,7) → cluster 3.
    ///
    /// 2D fixtures with symmetric opposing clusters cause the running centroid
    /// to collapse to the origin once two opposing picks are made, after which
    /// cosine similarity loses direction. The 3D anti-diagonal fixture avoids
    /// that pathology while still being well-separated.
    fn four_clusters_3d() -> Vec<Vec<f32>> {
        vec![
            // Cluster 0: near +x
            vec![1.00, 0.00, 0.00],
            vec![0.95, 0.05, 0.05],
            // Cluster 1: near +y
            vec![0.00, 1.00, 0.00],
            vec![0.05, 0.95, 0.05],
            // Cluster 2: near +z
            vec![0.00, 0.00, 1.00],
            vec![0.05, 0.05, 0.95],
            // Cluster 3: anti-diagonal
            vec![-0.577, -0.577, -0.577],
            vec![-0.60, -0.55, -0.58],
        ]
    }

    fn cluster_of(idx: usize) -> usize {
        idx / 2
    }

    #[test]
    fn centroid_picks_one_per_cluster_with_max_n_four() {
        let embs = four_clusters_3d();
        let selected = select_diverse(
            &embs,
            SelectionStrategy::Centroid,
            StopCondition::with_max_n(4),
        );

        assert_eq!(selected.len(), 4);
        let clusters: HashSet<usize> = selected.iter().map(|&i| cluster_of(i)).collect();
        assert_eq!(
            clusters.len(),
            4,
            "Centroid should pick exactly one representative per cluster, picked: {selected:?}"
        );
    }

    #[test]
    fn farthest_point_sampling_picks_one_per_cluster_with_max_n_four() {
        let embs = four_clusters_3d();
        let selected = select_diverse(
            &embs,
            SelectionStrategy::FarthestPointSampling,
            StopCondition::with_max_n(4),
        );

        assert_eq!(selected.len(), 4);
        let clusters: HashSet<usize> = selected.iter().map(|&i| cluster_of(i)).collect();
        assert_eq!(
            clusters.len(),
            4,
            "FPS should pick exactly one representative per cluster, picked: {selected:?}"
        );
    }

    #[test]
    fn max_n_caps_selection_count() {
        let embs = four_clusters_3d();
        let selected = select_diverse(
            &embs,
            SelectionStrategy::FarthestPointSampling,
            StopCondition::with_max_n(2),
        );
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn tripwire_stops_when_next_candidate_too_similar_to_group() {
        // All vectors are near-identical → similarity ~1.0 between any pair.
        // With a tight tripwire (0.5), only the seed should survive.
        let embs = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.01, 0.01, 0.0],
            vec![1.0, 0.0, 0.02],
            vec![0.99, 0.0, 0.0],
        ];
        let selected = select_diverse(
            &embs,
            SelectionStrategy::Centroid,
            StopCondition::with_tripwire(0.5),
        );
        assert_eq!(
            selected.len(),
            1,
            "all vectors too similar — only the seed should be selected"
        );
        assert_eq!(selected[0], 0);
    }

    #[test]
    fn tripwire_does_not_stop_when_candidates_dissimilar_enough() {
        // Two well-separated points in 2D. Loose tripwire (0.5) lets both through.
        let embs = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.0]];
        let selected = select_diverse(
            &embs,
            SelectionStrategy::FarthestPointSampling,
            StopCondition::with_tripwire(0.5),
        );
        assert!(
            selected.len() >= 2,
            "orthogonal vectors should not trip the 0.5 wire, got: {selected:?}"
        );
    }

    #[test]
    #[should_panic(expected = "must share the same dimension")]
    fn select_diverse_panics_on_mixed_dimensions() {
        let embs = vec![
            vec![1.0_f32, 0.0],
            vec![0.0_f32, 1.0, 0.0],
        ];
        let _ = select_diverse(
            &embs,
            SelectionStrategy::Centroid,
            StopCondition::with_max_n(2),
        );
    }

    #[test]
    fn empty_input_returns_empty_selection() {
        let selected = select_diverse(
            &[],
            SelectionStrategy::Centroid,
            StopCondition::with_max_n(4),
        );
        assert!(selected.is_empty());
    }

    #[test]
    fn max_n_zero_returns_empty_selection() {
        let embs = four_clusters_3d();
        let selected = select_diverse(
            &embs,
            SelectionStrategy::Centroid,
            StopCondition {
                max_n: Some(0),
                similarity_tripwire: None,
            },
        );
        assert!(selected.is_empty());
    }

    #[test]
    fn cosine_similarity_orthogonal_is_zero() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_identical_is_one() {
        let a = [0.5_f32, 0.3, 0.8];
        let b = [0.5_f32, 0.3, 0.8];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_opposite_is_negative_one() {
        let a = [1.0_f32, 2.0, 3.0];
        let b = [-1.0_f32, -2.0, -3.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-6);
    }
}
