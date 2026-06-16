//! Salience — which memories resist forgetting, and why.
//!
//! Decay forgets the *unused*; salience keeps the *formative*. They are
//! orthogonal axes over the same memory: a memory can be old and rarely
//! touched (decayed) yet still formative (salient), and survive.
//!
//! Salience is **U-shaped**, not a line from boring → interesting. Both
//! extremes of the novelty axis are formative — *surprise* (a memory unlike
//! anything stored) and *habit* (a memory echoed by many others spread
//! evenly over time). The boring middle — a few near-duplicates clustered
//! in one moment — is a transient *burst*, not formative.
//!
//! Critically, all three behaviors fall out of **one** computation: the
//! distribution of a memory's near-neighbors over time.
//!
//! ```text
//!   sparse neighbors            -> surprise   (novel; keep)
//!   dense + temporally tight    -> burst      (-> gist collapse)
//!   dense + temporally spread   -> habit      (-> trait promotion; keep)
//! ```
//!
//! This module is the shared primitive. Consolidation reads it three ways
//! (salience, gist-collapse, habit-promotion); keeping it here, pure and
//! deterministic, makes each behavior independently testable and lets the
//! observability layer explain *why* a memory was kept.

use serde::{Deserialize, Serialize};

/// A near-neighbor of some target memory, in the two dimensions the
/// discriminator cares about: how similar, and how long ago.
#[derive(Debug, Clone, Copy)]
pub struct Neighbor {
    /// Cosine similarity in `[0, 1]` to the target memory.
    pub similarity: f64,
    /// Age of this neighbor in days (>= 0) relative to the analysis time.
    pub age_days: f64,
}

/// What a memory's neighbor structure says about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeighborClass {
    /// Nothing similar exists — novel, formative.
    Surprise,
    /// Many near-neighbors clustered in a short window — a one-off burst
    /// (consolidation collapses these to a single gist).
    Burst,
    /// Many near-neighbors spread evenly over time — a stable habit
    /// (consolidation promotes the underlying trait).
    Habit,
    /// Neither extreme.
    Ordinary,
}

/// Tunable thresholds for the discriminator. Defaults are reasonable for a
/// personal store; tune offline against real history, never guess live.
#[derive(Debug, Clone, Copy)]
pub struct NeighborParams {
    /// A neighbor "counts" toward density above this cosine.
    pub similarity_threshold: f64,
    /// Below this nearest-cosine, the memory is classed Surprise.
    pub surprise_max_sim: f64,
    /// Neighbor count that maps to density 1.0 (saturates).
    pub density_saturation: f64,
    /// Temporal spread at/below which a dense cluster is a Burst.
    pub tight_spread: f64,
    /// Temporal spread at/above which a dense cluster is a Habit.
    pub wide_spread: f64,
    /// Minimum density for the Habit class (avoids 2-point "habits").
    pub habit_min_density: f64,
}

impl Default for NeighborParams {
    fn default() -> Self {
        NeighborParams {
            similarity_threshold: 0.60,
            surprise_max_sim: 0.45,
            density_saturation: 8.0,
            tight_spread: 0.15,
            wide_spread: 0.40,
            habit_min_density: 0.35,
        }
    }
}

/// The neighbor structure of one memory and what it implies. Every field
/// is reported so the UI can explain a salience score component by
/// component.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NeighborStats {
    /// Near-neighbors above `similarity_threshold`.
    pub count: usize,
    /// Nearest cosine to any candidate (drives surprise).
    pub max_similarity: f64,
    /// `count` saturated to `[0, 1]`.
    pub density: f64,
    /// Age range of near-neighbors / history span, in `[0, 1]`.
    pub temporal_spread: f64,
    /// `1 - max_similarity` — novelty.
    pub surprise_term: f64,
    /// `density * temporal_spread` — stable-recurrence.
    pub habit_term: f64,
    pub class: NeighborClass,
}

/// Classify a memory by the distribution of its near-neighbors over time.
///
/// `neighbors` are the candidate similarities/ages (e.g. the top-K nearest
/// existing memories). `history_span_days` is the age of the oldest memory
/// — the window spread is normalized against. Pure and deterministic.
pub fn analyze_neighbors(neighbors: &[Neighbor], history_span_days: f64, p: &NeighborParams) -> NeighborStats {
    let max_similarity = neighbors.iter().map(|n| n.similarity).fold(0.0_f64, f64::max);
    let near: Vec<&Neighbor> = neighbors.iter().filter(|n| n.similarity >= p.similarity_threshold).collect();
    let count = near.len();
    let density = (count as f64 / p.density_saturation.max(1.0)).clamp(0.0, 1.0);

    // Temporal spread: the age range of near-neighbors, normalized by the
    // store's history span. 0 = all at one instant (a burst), 1 = spans the
    // whole history (a long-standing habit).
    let temporal_spread = if count >= 2 && history_span_days > 0.0 {
        let (mut lo, mut hi) = (f64::MAX, f64::MIN);
        for nb in &near {
            lo = lo.min(nb.age_days);
            hi = hi.max(nb.age_days);
        }
        ((hi - lo) / history_span_days).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let surprise_term = (1.0 - max_similarity).clamp(0.0, 1.0);
    let habit_term = density * temporal_spread;

    let class = if max_similarity < p.surprise_max_sim {
        NeighborClass::Surprise
    } else if count >= 2 && temporal_spread <= p.tight_spread {
        NeighborClass::Burst
    } else if count >= 2 && temporal_spread >= p.wide_spread && density >= p.habit_min_density {
        NeighborClass::Habit
    } else {
        NeighborClass::Ordinary
    };

    NeighborStats { count, max_similarity, density, temporal_spread, surprise_term, habit_term, class }
}

/// Final salience in `[0, 1]`: the U-shaped max of surprise and habit,
/// gated by a relevance signal so novel-*noise* (typos, garbage, off-topic
/// tangents) can't enshrine itself as "surprising".
pub fn salience(stats: &NeighborStats, relevance: f64) -> f64 {
    (stats.surprise_term.max(stats.habit_term) * relevance.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

/// A coarse content-quality gate in `[0, 1]`: substantive content scores
/// ~1, trivial/garbage (very short, no extracted entities) scores low.
/// This is the relevance signal that keeps novel *noise* — typos, "ok",
/// off-topic fragments — from being enshrined as surprising. A real
/// goal-relevance vector would refine this later (spec 2.7).
pub fn content_relevance(content_chars: usize, entity_count: usize) -> f64 {
    match (content_chars >= 16, entity_count > 0) {
        (true, true) => 1.0,
        (true, false) => 0.7,
        (false, true) => 0.4,
        (false, false) => 0.1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(similarity: f64, age_days: f64) -> Neighbor {
        Neighbor { similarity, age_days }
    }

    #[test]
    fn novel_memory_is_surprise_and_salient() {
        // Nothing remotely similar exists.
        let stats = analyze_neighbors(&[n(0.2, 1.0), n(0.1, 30.0)], 100.0, &NeighborParams::default());
        assert_eq!(stats.class, NeighborClass::Surprise);
        assert!(stats.surprise_term > 0.7, "far from everything = surprising: {}", stats.surprise_term);
        assert!(salience(&stats, 1.0) > 0.7);
    }

    #[test]
    fn many_similar_in_a_tight_window_is_a_burst() {
        // 5 near-identical memories all within a day of each other.
        let neighbors: Vec<Neighbor> = (0..5).map(|i| n(0.85, 10.0 + i as f64 * 0.1)).collect();
        let stats = analyze_neighbors(&neighbors, 100.0, &NeighborParams::default());
        assert_eq!(stats.class, NeighborClass::Burst);
        assert!(stats.temporal_spread < 0.15);
    }

    #[test]
    fn many_similar_spread_over_time_is_a_habit() {
        // 6 similar memories spread across the whole 100-day history.
        let neighbors: Vec<Neighbor> = (0..6).map(|i| n(0.8, i as f64 * 18.0)).collect();
        let stats = analyze_neighbors(&neighbors, 100.0, &NeighborParams::default());
        assert_eq!(stats.class, NeighborClass::Habit);
        assert!(stats.habit_term > 0.0);
        assert!(salience(&stats, 1.0) > 0.0);
    }

    #[test]
    fn relevance_gate_suppresses_novel_noise() {
        // A novel-but-garbage memory: high surprise, but relevance ~0.
        let stats = analyze_neighbors(&[n(0.1, 1.0)], 100.0, &NeighborParams::default());
        assert!(stats.surprise_term > 0.8);
        assert!(salience(&stats, 0.05) < 0.1, "garbage novelty must not enshrine itself");
        assert!(salience(&stats, 1.0) > 0.8, "relevant novelty is kept");
    }

    #[test]
    fn burst_and_habit_are_the_same_count_opposite_spread() {
        // Same neighbor count (5), only the temporal spread differs — the
        // single discriminator that separates burst from habit.
        let tight: Vec<Neighbor> = (0..5).map(|i| n(0.8, 10.0 + i as f64 * 0.05)).collect();
        let spread: Vec<Neighbor> = (0..5).map(|i| n(0.8, i as f64 * 20.0)).collect();
        let p = NeighborParams::default();
        assert_eq!(analyze_neighbors(&tight, 100.0, &p).class, NeighborClass::Burst);
        assert_eq!(analyze_neighbors(&spread, 100.0, &p).class, NeighborClass::Habit);
    }
}
