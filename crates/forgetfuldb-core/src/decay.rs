//! Exponential forgetting curves.
//!
//! ```text
//! decay_score = importance_score * exp(-lambda * age_days)
//! ```
//!
//! Lambda is configurable per memory type. Defaults encode the desired
//! behavior: raw events evaporate within days, episodic memories fade over
//! weeks, semantic/procedural knowledge persists for months, and pinned
//! memories never decay.

use crate::types::MemoryType;
use serde::{Deserialize, Serialize};

/// Per-memory-type decay constants (per day).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayLambdas {
    pub raw_event: f64,
    pub episodic: f64,
    pub semantic: f64,
    pub procedural: f64,
    pub preference: f64,
    pub archive: f64,
}

impl Default for DecayLambdas {
    fn default() -> Self {
        DecayLambdas {
            // half-life ~2 days
            raw_event: 0.35,
            // half-life ~9 days
            episodic: 0.08,
            // half-life ~70 days
            semantic: 0.01,
            procedural: 0.01,
            // half-life ~35 days
            preference: 0.02,
            // already archived; decays like raw events for pruning purposes
            archive: 0.35,
        }
    }
}

impl DecayLambdas {
    pub fn for_type(&self, mt: MemoryType) -> f64 {
        match mt {
            MemoryType::RawEvent => self.raw_event,
            MemoryType::Episodic => self.episodic,
            MemoryType::Semantic => self.semantic,
            MemoryType::Procedural => self.procedural,
            MemoryType::Preference => self.preference,
            MemoryType::Archive => self.archive,
        }
    }
}

/// Decay-adjusted importance. Pinned memories do not decay at all.
pub fn decay_score(importance: f64, lambda: f64, age_days: f64, pinned: bool) -> f64 {
    if pinned {
        return importance;
    }
    importance * (-lambda * age_days.max(0.0)).exp()
}

/// Decay-adjusted importance with **salience resistance**: a formative
/// (high-salience) memory forgets more slowly. `salience` in `[0, 1]`
/// scales the effective decay rate down by up to `resist` (e.g. resist
/// 0.7 → a fully-salient memory decays at 30% of the base rate). Pinned
/// still short-circuits to no decay at all. This is where the salience
/// axis "keeps the formative" meets decay "forgets the unused".
pub fn decay_score_resisted(importance: f64, lambda: f64, age_days: f64, pinned: bool, salience: f64, resist: f64) -> f64 {
    if pinned {
        return importance;
    }
    let eff_lambda = lambda * (1.0 - resist.clamp(0.0, 1.0) * salience.clamp(0.0, 1.0));
    importance * (-eff_lambda * age_days.max(0.0)).exp()
}

/// Recency score in [0, 1] from days since last access (or creation).
/// Uses a gentle curve so something touched a week ago still registers.
pub fn recency_score(days_since_access: f64) -> f64 {
    (-0.1 * days_since_access.max(0.0)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_matches_formula() {
        let s = decay_score(0.8, 0.1, 10.0, false);
        let expected = 0.8 * (-1.0f64).exp();
        assert!((s - expected).abs() < 1e-12);
    }

    #[test]
    fn decay_decreases_with_age() {
        let young = decay_score(0.8, 0.35, 1.0, false);
        let old = decay_score(0.8, 0.35, 10.0, false);
        assert!(young > old);
        assert!(old > 0.0);
    }

    #[test]
    fn pinned_memories_do_not_decay() {
        let s = decay_score(0.8, 0.35, 365.0, true);
        assert_eq!(s, 0.8);
    }

    #[test]
    fn salience_slows_decay() {
        // Same memory, same age — the salient one retains far more.
        let dull = decay_score_resisted(0.8, 0.35, 20.0, false, 0.0, 0.7);
        let salient = decay_score_resisted(0.8, 0.35, 20.0, false, 1.0, 0.7);
        assert!(salient > dull);
        // A fully-salient memory decays at (1 - resist) of the base rate.
        let plain = decay_score(0.8, 0.35, 20.0, false);
        assert!((dull - plain).abs() < 1e-9, "salience 0 == plain decay");
        let expected_salient = decay_score(0.8, 0.35 * 0.3, 20.0, false);
        assert!((salient - expected_salient).abs() < 1e-9);
        // Pin still wins outright.
        assert_eq!(decay_score_resisted(0.8, 0.35, 365.0, true, 0.0, 0.7), 0.8);
    }

    #[test]
    fn raw_events_decay_faster_than_semantic() {
        let l = DecayLambdas::default();
        let raw = decay_score(0.8, l.for_type(MemoryType::RawEvent), 7.0, false);
        let semantic = decay_score(0.8, l.for_type(MemoryType::Semantic), 7.0, false);
        assert!(semantic > raw);
    }

    #[test]
    fn recency_is_bounded() {
        assert!((recency_score(0.0) - 1.0).abs() < 1e-12);
        assert!(recency_score(30.0) < 0.1);
        assert!(recency_score(30.0) > 0.0);
    }
}
