//! Hybrid retrieval scoring.
//!
//! ```text
//! final_score =
//!     0.60 * relevance
//!   + 0.15 * recency_score
//!   + 0.10 * importance_score
//!   + 0.10 * recurrence_score
//!   + 0.05 * graph_support
//!   + pinned_boost
//!   - staleness_penalty
//! ```
//!
//! All inputs are expected in `[0, 1]`. Candidate recall is relevance-only;
//! importance belongs in final ranking as a tie-breaker/retention signal.

use serde::{Deserialize, Serialize};

/// Weights for each component of the retrieval score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalWeights {
    /// Historical name kept for config/UI compatibility. In V2 this is the
    /// relevance term produced by candidate recall.
    pub semantic: f64,
    pub importance: f64,
    pub recurrence: f64,
    pub recency: f64,
    #[serde(default = "default_graph_support_weight")]
    pub graph_support: f64,
    pub pinned_boost: f64,
    pub staleness_penalty: f64,
}

impl Default for RetrievalWeights {
    fn default() -> Self {
        RetrievalWeights {
            semantic: 0.60,
            importance: 0.10,
            recurrence: 0.10,
            recency: 0.15,
            graph_support: default_graph_support_weight(),
            pinned_boost: 0.05,
            staleness_penalty: 0.20,
        }
    }
}

fn default_graph_support_weight() -> f64 {
    0.05
}

/// Inputs to the retrieval score for one candidate memory.
#[derive(Debug, Clone, Copy)]
pub struct ScoreInputs {
    /// Relevance from candidate recall in [0, 1].
    pub semantic_similarity: f64,
    /// Decay-adjusted importance in [0, 1].
    pub importance: f64,
    pub recurrence: f64,
    pub recency: f64,
    pub graph_support: f64,
    pub pinned: bool,
    pub stale: bool,
    /// The memory's salience in `[0, 1]`. Informational here — its effect
    /// is already baked into `importance` via salience-resisted decay — but
    /// surfaced in the breakdown so the UI can explain *why* an old memory
    /// survived.
    pub salience: f64,
}

/// Per-component breakdown returned to callers so the CLI/API can explain
/// *why* a memory was retrieved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    /// Same value as `semantic_similarity`, named for the V2 two-stage
    /// pipeline. The old field remains for API compatibility.
    #[serde(default)]
    pub relevance: f64,
    pub semantic_similarity: f64,
    pub importance: f64,
    pub recurrence: f64,
    pub recency: f64,
    #[serde(default)]
    pub graph_support: f64,
    pub pinned_boost: f64,
    pub staleness_penalty: f64,
    /// Multiplier applied to `total` because the memory is a verbatim
    /// conversational turn (chat raw event / episodic) rather than a
    /// distilled fact. 1.0 means no damping was applied.
    #[serde(default = "no_damping")]
    pub conversational_damping: f64,
    /// Additive boost from spreading activation: this memory is associated
    /// (co-occurs in past turns) with higher-scoring hits. 0 when spreading
    /// activation is off or the memory has no relevant associations.
    #[serde(default)]
    pub association_boost: f64,
    /// The memory's salience in `[0, 1]` (informational; folded into
    /// `importance` via decay resistance).
    #[serde(default)]
    pub salience: f64,
    pub total: f64,
}

fn no_damping() -> f64 {
    1.0
}

/// Compute the weighted retrieval score with a full breakdown.
pub fn retrieval_score(inputs: &ScoreInputs, w: &RetrievalWeights) -> ScoreBreakdown {
    let pinned_boost = if inputs.pinned { 1.0 } else { 0.0 };
    let staleness_penalty = if inputs.stale { 1.0 } else { 0.0 };
    let total = w.semantic * inputs.semantic_similarity
        + w.importance * inputs.importance
        + w.recurrence * inputs.recurrence
        + w.recency * inputs.recency
        + w.graph_support * inputs.graph_support
        + w.pinned_boost * pinned_boost
        - w.staleness_penalty * staleness_penalty;
    ScoreBreakdown {
        relevance: inputs.semantic_similarity,
        semantic_similarity: inputs.semantic_similarity,
        importance: inputs.importance,
        recurrence: inputs.recurrence,
        recency: inputs.recency,
        graph_support: inputs.graph_support,
        pinned_boost,
        staleness_penalty,
        conversational_damping: 1.0,
        association_boost: 0.0,
        salience: inputs.salience,
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
            graph_support: 0.0,
            pinned: false,
            stale: false,
            salience: 0.0,
        }
    }

    #[test]
    fn retrieval_score_matches_spec_formula() {
        let w = RetrievalWeights::default();
        let s = retrieval_score(&base_inputs(), &w);
        let expected = 0.60 * 0.8 + 0.10 * 0.6 + 0.10 * 0.4 + 0.15 * 0.5;
        assert!((s.total - expected).abs() < 1e-12);
    }

    #[test]
    fn pinned_memory_gets_boost() {
        let w = RetrievalWeights::default();
        let mut inputs = base_inputs();
        let unpinned = retrieval_score(&inputs, &w).total;
        inputs.pinned = true;
        let pinned = retrieval_score(&inputs, &w).total;
        assert!((pinned - unpinned - 0.05).abs() < 1e-12);
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
