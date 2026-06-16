//! Epochs — segmenting a lifetime of memories into eras.
//!
//! Decay forgets, salience keeps, edges connect; **epochs organize**. As the
//! embedding stream moves through time its centre of mass drifts. When recent
//! memories drift far enough from the current era's identity — and *stay*
//! drifted (hysteresis), so a one-off tangent doesn't split an era — a new
//! epoch begins. The result is a sequence of drift-segmented eras the engine
//! can name and date exactly, which is the whole point: the model has no
//! clock, so "during the Clarity era" has to be computed here.
//!
//! This module is pure and deterministic — no SQLite, no embedding model, no
//! clock of its own. Like [`crate::salience`], that makes the mechanism
//! independently testable and the boundaries reproducible.

use serde::{Deserialize, Serialize};

/// One memory reduced to what epoch segmentation needs: when it happened and
/// where it sits in embedding space. Callers pass these **sorted by
/// `created_at`** (ascending).
#[derive(Debug, Clone)]
pub struct EpochPoint {
    pub created_at: i64,
    pub embedding: Vec<f32>,
}

/// A detected era: a contiguous run of memories in time whose embeddings
/// share a centre of mass. `ended_at` is `None` for the current (open) epoch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpochSpan {
    /// 0-based index in time order.
    pub ordinal: usize,
    pub started_at: i64,
    /// Exclusive end (the `created_at` of the first memory in the next era),
    /// or `None` while this is the latest era.
    pub ended_at: Option<i64>,
    /// Unit-normalized mean embedding of the era's members — its identity.
    pub centroid: Vec<f32>,
    pub member_count: usize,
    /// The drift (`1 − cosine`) from the previous era that opened this one.
    /// `0.0` for the first era.
    pub drift_in: f64,
}

/// Tunable thresholds for segmentation. Defaults suit a personal store; tune
/// offline against real history rather than guessing live.
#[derive(Debug, Clone, Copy)]
pub struct EpochParams {
    /// Cosine *distance* (`1 − cosine`) from the era centroid above which a
    /// memory counts as "drifting".
    pub drift_threshold: f64,
    /// Consecutive drifting memories required to actually cut a boundary. A
    /// single on-topic memory resets the run — this is the hysteresis that
    /// keeps a brief tangent from splitting an era.
    pub hysteresis_runs: usize,
    /// A closed era must have at least this many members…
    pub min_size: usize,
    /// …and span at least this many days. Together they stop micro-eras.
    pub min_days: f64,
}

impl Default for EpochParams {
    fn default() -> Self {
        EpochParams { drift_threshold: 0.35, hysteresis_runs: 3, min_size: 4, min_days: 5.0 }
    }
}

fn normalize(v: &[f32]) -> Vec<f64> {
    let norm = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    if norm <= f64::EPSILON {
        return vec![0.0; v.len()];
    }
    v.iter().map(|&x| x as f64 / norm).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Unit-normalized mean of a set of embeddings — the centroid direction.
fn centroid_of(points: &[EpochPoint]) -> Vec<f32> {
    if points.is_empty() {
        return Vec::new();
    }
    let dim = points[0].embedding.len();
    let mut sum = vec![0.0_f64; dim];
    for p in points {
        for (s, x) in sum.iter_mut().zip(normalize(&p.embedding)) {
            *s += x;
        }
    }
    let unit = normalize(&sum.iter().map(|&x| x as f32).collect::<Vec<_>>());
    unit.into_iter().map(|x| x as f32).collect()
}

fn days_between(a: i64, b: i64) -> f64 {
    ((b - a).max(0) as f64) / 86_400.0
}

/// Segment a time-ordered embedding stream into drift-bounded eras.
///
/// Single online pass: maintain the current era's centroid (the running mean
/// of its on-topic members) and a counter of consecutive drifting memories.
/// When that counter reaches `hysteresis_runs` — and the era being closed is
/// large and long enough — cut a boundary at the *start* of the drift run, so
/// the drifting memories seed the new era rather than polluting the old one.
pub fn segment(points: &[EpochPoint], p: &EpochParams) -> Vec<EpochSpan> {
    if points.is_empty() {
        return Vec::new();
    }
    let dim = points[0].embedding.len();

    let mut spans = Vec::new();
    let mut start = 0usize; // index of the current era's first member
    let mut sum = normalize(&points[0].embedding); // Σ normalized on-topic members
    let mut n_in = 1usize;
    let mut run = 0usize; // consecutive drifting members
    let mut run_start = 0usize;
    let mut cur_drift_in = 0.0; // drift that opened the current era

    let fold = |sum: &mut Vec<f64>, n_in: &mut usize, e: &[f64]| {
        for (s, x) in sum.iter_mut().zip(e) {
            *s += x;
        }
        *n_in += 1;
    };

    for i in 1..points.len() {
        let e = normalize(&points[i].embedding);
        // Cosine to the era's identity: sum is a vector of summed unit
        // vectors, so normalize it before the dot product.
        let centroid_dir = normalize(&sum.iter().map(|&x| x as f32).collect::<Vec<_>>());
        let drift = 1.0 - dot(&centroid_dir, &e);
        let drifting = drift > p.drift_threshold;

        if drifting {
            if run == 0 {
                run_start = i;
            }
            run += 1;
        } else {
            run = 0;
        }

        if run >= p.hysteresis_runs {
            let boundary = run_start;
            let old_count = boundary - start;
            let old_days = days_between(points[start].created_at, points[boundary - 1].created_at);
            if old_count >= p.min_size && old_days >= p.min_days {
                spans.push(EpochSpan {
                    ordinal: spans.len(),
                    started_at: points[start].created_at,
                    ended_at: Some(points[boundary].created_at),
                    centroid: centroid_of(&points[start..boundary]),
                    member_count: old_count,
                    drift_in: cur_drift_in,
                });
                // The drift run seeds the new era.
                start = boundary;
                sum = vec![0.0; dim];
                n_in = 0;
                for q in &points[boundary..=i] {
                    fold(&mut sum, &mut n_in, &normalize(&q.embedding));
                }
                cur_drift_in = drift;
                run = 0;
                continue;
            }
            // Too small/short to stand alone: absorb the drift into this era.
            run = 0;
            fold(&mut sum, &mut n_in, &e);
            continue;
        }

        // On-topic members define the era centroid; held-out drift members
        // (a run that hasn't yet triggered) are still era members by time,
        // they just don't steer the running drift comparison.
        if !drifting {
            fold(&mut sum, &mut n_in, &e);
        }
    }

    // Close the final, open era.
    spans.push(EpochSpan {
        ordinal: spans.len(),
        started_at: points[start].created_at,
        ended_at: None,
        centroid: centroid_of(&points[start..]),
        member_count: points.len() - start,
        drift_in: cur_drift_in,
    });
    spans
}

/// Which era a timestamp falls in, given the eras' `started_at` values in
/// ascending order. Returns the index of the last era whose start is `<= ts`
/// (everything before the first boundary belongs to era 0).
pub fn epoch_index_at(starts: &[i64], ts: i64) -> usize {
    match starts.binary_search(&ts) {
        Ok(i) => i,
        Err(0) => 0,
        Err(i) => i - 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4-D one-hot-ish embedding for topic `axis`, with a little spread so
    /// members of a topic aren't bit-identical.
    fn emb(axis: usize, jitter: f32) -> Vec<f32> {
        let mut v = vec![0.0_f32; 4];
        v[axis] = 1.0;
        v[(axis + 1) % 4] = jitter; // small off-axis component
        v
    }

    fn stream(specs: &[(usize, i64)]) -> Vec<EpochPoint> {
        specs
            .iter()
            .map(|&(axis, day)| EpochPoint { created_at: day * 86_400, embedding: emb(axis, 0.05) })
            .collect()
    }

    #[test]
    fn single_topic_stream_is_one_epoch() {
        let pts: Vec<_> = (0..12).map(|d| EpochPoint { created_at: d * 86_400, embedding: emb(0, 0.05 * (d % 3) as f32) }).collect();
        let spans = segment(&pts, &EpochParams::default());
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].ended_at, None);
        assert_eq!(spans[0].member_count, 12);
    }

    #[test]
    fn sharp_shift_creates_one_boundary() {
        // 8 days of topic 0, then 8 days of topic 2 (orthogonal).
        let mut specs: Vec<(usize, i64)> = (0..8).map(|d| (0, d)).collect();
        specs.extend((8..16).map(|d| (2, d)));
        let pts = stream(&specs);
        let spans = segment(&pts, &EpochParams::default());
        assert_eq!(spans.len(), 2, "a clean topic shift should yield two eras");
        // Boundary falls at the first topic-2 memory (day 8).
        assert_eq!(spans[0].ended_at, Some(8 * 86_400));
        assert_eq!(spans[1].started_at, 8 * 86_400);
        assert!(spans[1].drift_in > 0.5, "the second era opened on a large drift: {}", spans[1].drift_in);
    }

    #[test]
    fn brief_excursion_does_not_split() {
        // Topic 0 throughout, with a single off-topic memory in the middle.
        // One tangent must not reach the hysteresis run, so it's one era.
        let mut specs: Vec<(usize, i64)> = (0..6).map(|d| (0, d)).collect();
        specs.push((2, 6)); // the excursion
        specs.extend((7..13).map(|d| (0, d)));
        let pts = stream(&specs);
        let spans = segment(&pts, &EpochParams::default());
        assert_eq!(spans.len(), 1, "a one-off tangent must not split an era");
    }

    #[test]
    fn min_size_prevents_micro_epochs() {
        // A shift after only 2 memories: the era to close is below min_size,
        // so no boundary is cut.
        let mut specs: Vec<(usize, i64)> = (0..2).map(|d| (0, d)).collect();
        specs.extend((2..12).map(|d| (2, d)));
        let pts = stream(&specs);
        let spans = segment(&pts, &EpochParams { min_size: 4, ..Default::default() });
        assert_eq!(spans.len(), 1, "a 2-member lead-in can't stand as its own era");
    }

    #[test]
    fn min_days_prevents_short_epochs() {
        // Enough members, but all within a single day: too short to close.
        let mut specs: Vec<(usize, i64)> = (0..6).map(|_| (0, 0)).collect();
        specs.extend((0..8).map(|_| (2, 0)));
        let pts = stream(&specs);
        let spans = segment(&pts, &EpochParams { min_days: 5.0, ..Default::default() });
        assert_eq!(spans.len(), 1, "a same-day burst isn't a multi-era history");
    }

    #[test]
    fn tiny_or_empty_input_does_not_panic() {
        assert!(segment(&[], &EpochParams::default()).is_empty());
        let one = vec![EpochPoint { created_at: 0, embedding: emb(0, 0.0) }];
        assert_eq!(segment(&one, &EpochParams::default()).len(), 1);
    }

    #[test]
    fn three_eras_in_order() {
        // 0 → 1 → 2, each a clean week-long block.
        let mut specs: Vec<(usize, i64)> = (0..7).map(|d| (0, d)).collect();
        specs.extend((7..14).map(|d| (1, d)));
        specs.extend((14..21).map(|d| (2, d)));
        let pts = stream(&specs);
        let spans = segment(&pts, &EpochParams::default());
        assert_eq!(spans.len(), 3);
        assert_eq!(spans.iter().map(|s| s.ordinal).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!(spans[2].ended_at, None);
    }

    #[test]
    fn epoch_index_at_maps_timestamps_to_eras() {
        let starts = [0_i64, 100, 200];
        assert_eq!(epoch_index_at(&starts, -5), 0); // before everything → era 0
        assert_eq!(epoch_index_at(&starts, 0), 0);
        assert_eq!(epoch_index_at(&starts, 50), 0);
        assert_eq!(epoch_index_at(&starts, 100), 1);
        assert_eq!(epoch_index_at(&starts, 150), 1);
        assert_eq!(epoch_index_at(&starts, 250), 2);
    }
}
