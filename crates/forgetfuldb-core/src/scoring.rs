//! Hybrid retrieval scoring.
//!
//! ```text
//! retrieval_score =
//!     0.45 * semantic_similarity
//!   + 0.20 * importance_score
//!   + 0.15 * recurrence_score
//!   + 0.10 * recency_score
//!   + 0.10 * pinned_boost
//!   - 0.20 * staleness_penalty
//! ```
//!
//! All inputs are expected in `[0, 1]`. The importance term should be the
//! decay-adjusted importance (`decay::decay_score`), so old unimportant
//! memories naturally sink without a separate decay term here.

use serde::{Deserialize, Serialize};

/// Weights for each component of the retrieval score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalWeights {
    pub semantic: f64,
    pub importance: f64,
    pub recurrence: f64,
    pub recency: f64,
    pub pinned_boost: f64,
    pub staleness_penalty: f64,
}

impl Default for RetrievalWeights {
    fn default() -> Self {
        RetrievalWeights {
            semantic: 0.45,
            importance: 0.20,
            recurrence: 0.15,
            recency: 0.10,
            pinned_boost: 0.10,
            staleness_penalty: 0.20,
        }
    }
}

/// Inputs to the retrieval score for one candidate memory.
#[derive(Debug, Clone, Copy)]
pub struct ScoreInputs {
    /// Blended vector/keyword similarity in [0, 1].
    pub semantic_similarity: f64,
    /// Decay-adjusted importance in [0, 1].
    pub importance: f64,
    pub recurrence: f64,
    pub recency: f64,
    pub pinned: bool,
    pub stale: bool,
}

/// Per-component breakdown returned to callers so the CLI/API can explain
/// *why* a memory was retrieved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub semantic_similarity: f64,
    pub importance: f64,
    pub recurrence: f64,
    pub recency: f64,
    pub pinned_boost: f64,
    pub staleness_penalty: f64,
    pub total: f64,
}

/// Compute the weighted retrieval score with a full breakdown.
pub fn retrieval_score(inputs: &ScoreInputs, w: &RetrievalWeights) -> ScoreBreakdown {
    let pinned_boost = if inputs.pinned { 1.0 } else { 0.0 };
    let staleness_penalty = if inputs.stale { 1.0 } else { 0.0 };
    let total = w.semantic * inputs.semantic_similarity
        + w.importance * inputs.importance
        + w.recurrence * inputs.recurrence
        + w.recency * inputs.recency
        + w.pinned_boost * pinned_boost
        - w.staleness_penalty * staleness_penalty;
    ScoreBreakdown {
        semantic_similarity: inputs.semantic_similarity,
        importance: inputs.importance,
        recurrence: inputs.recurrence,
        recency: inputs.recency,
        pinned_boost,
        staleness_penalty,
        total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_inputs() -> ScoreInputs {
        ScoreInputs {
            semantic_similarity: 0.8,
            importance: 0.6,
            recurrence: 0.4,
            recency: 0.5,
            pinned: false,
            stale: false,
        }
    }

    #[test]
    fn retrieval_score_matches_spec_formula() {
        let w = RetrievalWeights::default();
        let s = retrieval_score(&base_inputs(), &w);
        let expected = 0.45 * 0.8 + 0.20 * 0.6 + 0.15 * 0.4 + 0.10 * 0.5;
        assert!((s.total - expected).abs() < 1e-12);
    }

    #[test]
    fn pinned_memory_gets_boost() {
        let w = RetrievalWeights::default();
        let mut inputs = base_inputs();
        let unpinned = retrieval_score(&inputs, &w).total;
        inputs.pinned = true;
        let pinned = retrieval_score(&inputs, &w).total;
        assert!((pinned - unpinned - 0.10).abs() < 1e-12);
    }

    #[test]
    fn stale_memory_is_penalized() {
        let w = RetrievalWeights::default();
        let mut inputs = base_inputs();
        let fresh = retrieval_score(&inputs, &w).total;
        inputs.stale = true;
        let stale = retrieval_score(&inputs, &w).total;
        assert!((fresh - stale - 0.20).abs() < 1e-12);
        assert!(stale < fresh);
    }
}
