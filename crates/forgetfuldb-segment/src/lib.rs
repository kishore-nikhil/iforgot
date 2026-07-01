//! forgetfuldb-segment — Tier-2 embedding-space **surprise segmentation**.
//!
//! Turns an ordered stream of memory embeddings into contiguous **events** by
//! finding where the stream stops looking like its recent past. The pipeline:
//!
//! 1. normalize every embedding (FR-2)
//! 2. **surprise** = `1 − cosine(predict(prior_window), vᵢ)` (FR-1, FR-3)
//! 3. **threshold** — cut where surprise exceeds a rolling `μ + γσ` (FR-4)
//! 4. **refine** — snap each cut to its locally most coherent split (FR-5)
//!
//! No LLM, no clock, no RNG: given identical embeddings and config the output
//! is bit-identical (NFR-1, NFR-2). The crate decides the *units* of memory;
//! it never reads or writes decay, retrieval, or scoring (NFR-5).
//!
//! Two entry points: [`segment`] embeds strings with an [`EmbeddingProvider`]
//! and stamps provenance; [`segment_with_embeddings`] works on vectors you
//! already have (tests, stored embeddings) and carries the tier-3
//! `precomputed_surprise` hook. See `docs/surprise-segmentation.md`.

mod predictor;
mod refine;
mod surprise;
mod threshold;

pub use predictor::{
    predictor_for, CentroidPredictor, ExtrapolatePredictor, Predictor, WeightedPredictor,
};
pub use refine::{refine_boundaries, SimMatrix};
pub use surprise::surprise_signal;
pub use threshold::detect_boundaries;

use forgetfuldb_core::config::SegmentConfig;
use forgetfuldb_embed::EmbeddingProvider;

/// A contiguous half-open range of entry indices forming one event.
/// `end` is exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    pub start: usize,
    pub end: usize,
}

impl Event {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }
}

/// The segmentation of a stream: the events, the embedding space they were
/// computed in (FR-8), and the raw surprise signal (consumed by epoch
/// `drift_in` and the observability UI).
#[derive(Debug, Clone)]
pub struct SegmentResult {
    /// Contiguous, sorted, gap-free events covering `0..n`.
    pub events: Vec<Event>,
    /// Identity of the embedding model (`EmbeddingProvider::model_id`). `None`
    /// when segmenting raw vectors with no provider — the caller stamps it.
    pub embedding_model: Option<String>,
    /// Per-entry surprise; warm-up entries report `0.0`. Same length as input.
    pub surprise: Vec<f64>,
}

/// Segment an ordered stream of text entries. Embeds each with `embedder`,
/// records `embedder.model_id()` as provenance (FR-8), then delegates to
/// [`segment_with_embeddings`].
pub fn segment<E: EmbeddingProvider + ?Sized>(
    entries: &[String],
    embedder: &E,
    cfg: &SegmentConfig,
) -> SegmentResult {
    let embs: Vec<Vec<f32>> = entries.iter().map(|t| embedder.embed(t)).collect();
    let mut res = segment_with_embeddings(&embs, cfg, None);
    res.embedding_model = Some(embedder.model_id());
    res
}

/// Segment already-embedded vectors. `precomputed_surprise`, if `Some`,
/// bypasses FR-1..FR-3 and feeds straight into the threshold pass (the tier-3
/// hook, §12) — it must be one value per entry.
pub fn segment_with_embeddings(
    embs: &[Vec<f32>],
    cfg: &SegmentConfig,
    precomputed_surprise: Option<&[f64]>,
) -> SegmentResult {
    let n = embs.len();

    // FR-6 trivial cases.
    if n == 0 {
        return SegmentResult { events: Vec::new(), embedding_model: None, surprise: Vec::new() };
    }
    if n <= cfg.min_event_len {
        return SegmentResult {
            events: vec![Event { start: 0, end: n }],
            embedding_model: None,
            surprise: precomputed_surprise.map(|s| s.to_vec()).unwrap_or_else(|| vec![0.0; n]),
        };
    }

    // Surprise (FR-3) or the precomputed tier-3 signal. A precomputed signal
    // has no warm-up region.
    let (surprise, warmup) = match precomputed_surprise {
        Some(s) => (s.to_vec(), 0usize),
        None => {
            let pred = predictor_for(cfg.predictor, cfg.weight_decay);
            let s = surprise_signal(embs, pred.as_ref(), cfg.window_size);
            (s, cfg.window_size.min(n))
        }
    };

    // Candidate boundaries (FR-4).
    let candidates =
        detect_boundaries(&surprise, warmup, cfg.threshold_window, cfg.gamma, cfg.min_event_len);

    // Refinement (FR-5) snaps a cut to its most coherent position using the
    // embedding geometry. It is skipped on the precomputed-surprise path: that
    // signal may come from a different space (token logprobs) with no
    // trustworthy relationship to these vectors, so the detected boundaries
    // are authoritative.
    let refined = if precomputed_surprise.is_some() {
        candidates
    } else {
        let normed: Vec<Vec<f64>> = embs.iter().map(|e| surprise::normalize_f64(e)).collect();
        let sim = SimMatrix::new(&normed);
        let moved = refine_boundaries(&candidates, &sim, cfg.refine_radius, cfg.min_event_len, n);
        // Veto cuts that don't actually separate distinct content (volatility
        // artifacts) — the graph disposes what the surprise signal proposed.
        // A window-sized neighborhood averages out within-topic noise so a mere
        // spike in volatility isn't mistaken for a direction change.
        refine::coherence_gate(&moved, &normed, cfg.window_size.max(2), n)
    };

    let events = boundaries_to_events(&refined, n, cfg.min_event_len);
    SegmentResult { events, embedding_model: None, surprise }
}

/// Build contiguous events from sorted boundary indices, absorbing a trailing
/// remainder shorter than `min_event_len` into the previous event (FR-6).
fn boundaries_to_events(boundaries: &[usize], n: usize, min_event_len: usize) -> Vec<Event> {
    let mut bnds: Vec<usize> = boundaries.to_vec();
    while let Some(&last) = bnds.last() {
        if n - last < min_event_len {
            bnds.pop();
        } else {
            break;
        }
    }
    let mut events = Vec::with_capacity(bnds.len() + 1);
    let mut start = 0usize;
    for &b in &bnds {
        if b <= start {
            continue; // defensive: never emit an empty or backwards event
        }
        events.push(Event { start, end: b });
        start = b;
    }
    events.push(Event { start, end: n });
    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use forgetfuldb_core::config::{PredictorKind, SegmentConfig};

    fn cfg() -> SegmentConfig {
        SegmentConfig::default()
    }

    fn covers(events: &[Event], n: usize) {
        assert_eq!(events.first().map(|e| e.start), Some(0));
        assert_eq!(events.last().map(|e| e.end), Some(n));
        for w in events.windows(2) {
            assert_eq!(w[0].end, w[1].start, "events must be contiguous");
        }
        assert!(events.iter().all(|e| e.end > e.start), "no empty events");
    }

    #[test]
    fn empty_input_is_empty() {
        let r = segment_with_embeddings(&[], &cfg(), None);
        assert!(r.events.is_empty());
    }

    #[test]
    fn tiny_input_is_one_event() {
        let embs = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let r = segment_with_embeddings(&embs, &cfg(), None);
        assert_eq!(r.events, vec![Event { start: 0, end: 2 }]);
    }

    #[test]
    fn precomputed_surprise_bypasses_prediction() {
        // A precomputed spike at 15 with no warm-up region.
        let embs: Vec<Vec<f32>> = (0..30).map(|_| vec![1.0, 0.0]).collect();
        let mut s = vec![0.05; 30];
        s[15] = 0.95;
        let mut c = cfg();
        c.predictor = PredictorKind::Centroid; // must be ignored on this path
        let r = segment_with_embeddings(&embs, &c, Some(&s));
        covers(&r.events, 30);
        assert!(r.events.iter().any(|e| e.start == 15), "cut should follow the precomputed spike");
    }
}
