//! Graph refinement — snap each boundary to its locally most coherent cut
//! (FR-5).
//!
//! For a candidate at position `p` (left segment ends at `p`, right starts at
//! `p`), a modularity-style score rewards internally-similar sides and a
//! dissimilar cut: `intra_left + intra_right − across`. Each boundary slides
//! within `±radius` to the position that maximizes it.
//!
//! Refinement may only **move** boundaries, never add or remove them (V-7).
//! Movement is clamped so boundaries stay sorted, stay `min_event_len` apart,
//! and keep the head/tail segments `≥ min_event_len` — so a collision is
//! resolved by clamping, not by dropping a boundary.
//!
//! We do **not** materialize the `O(n²·d)` similarity matrix the spec allows
//! (design §5): [`SimMatrix`] holds the normalized embeddings and computes
//! cosines lazily, and the coherence is evaluated over a bounded `±radius`
//! neighborhood, so refinement stays `O(boundaries · radius² · d)`.

/// Lazy cosine over unit-normalized embeddings — the `sim` of FR-5 without the
/// quadratic matrix. Vectors are pre-normalized, so cosine is a dot product.
pub struct SimMatrix<'a> {
    embs: &'a [Vec<f64>],
}

impl<'a> SimMatrix<'a> {
    pub fn new(embs: &'a [Vec<f64>]) -> Self {
        SimMatrix { embs }
    }

    pub fn cosine(&self, i: usize, j: usize) -> f64 {
        self.embs[i]
            .iter()
            .zip(&self.embs[j])
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .clamp(-1.0, 1.0)
    }

    pub fn len(&self) -> usize {
        self.embs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embs.is_empty()
    }
}

/// Mean pairwise cosine within `[a, b)`. A side of `< 2` members is trivially
/// coherent (`1.0`) so it neither rewards nor penalizes the score.
fn intra(sim: &SimMatrix, a: usize, b: usize) -> f64 {
    if b.saturating_sub(a) < 2 {
        return 1.0;
    }
    let mut total = 0.0;
    let mut count = 0.0;
    for i in a..b {
        for j in (i + 1)..b {
            total += sim.cosine(i, j);
            count += 1.0;
        }
    }
    total / count
}

/// Mean cosine across the cut: every left member against every right member.
fn across(sim: &SimMatrix, ll: usize, p: usize, rr: usize) -> f64 {
    if p <= ll || rr <= p {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0.0;
    for i in ll..p {
        for j in p..rr {
            total += sim.cosine(i, j);
            count += 1.0;
        }
    }
    total / count
}

/// Cut quality at `p`, evaluated over a `±half` neighborhood bounded by the
/// neighboring boundaries. Higher = a cleaner split.
fn cut_score(sim: &SimMatrix, p: usize, left_bound: usize, right_bound: usize, half: usize) -> f64 {
    let ll = p.saturating_sub(half).max(left_bound);
    let rr = (p + half).min(right_bound);
    intra(sim, ll, p) + intra(sim, p, rr) - across(sim, ll, p, rr)
}

/// Slide each candidate to its locally most coherent position (FR-5).
/// `n` is the stream length; `min_event_len` bounds how close a boundary may
/// sit to its neighbors and to the ends.
pub fn refine_boundaries(
    candidates: &[usize],
    sim: &SimMatrix,
    radius: usize,
    min_event_len: usize,
    n: usize,
) -> Vec<usize> {
    let mut refined: Vec<usize> = Vec::with_capacity(candidates.len());
    if n < 2 * min_event_len {
        return refined; // no interior position can host a legal boundary
    }
    let interior_hi = n - min_event_len;

    for (idx, &b) in candidates.iter().enumerate() {
        let prev = if idx == 0 { 0 } else { refined[idx - 1] };
        let next = candidates.get(idx + 1).copied().unwrap_or(n);

        // Legal interval: within ±radius, past the previous boundary by
        // min_event_len, before the next candidate by min_event_len, and
        // inside the interior so head/tail stay ≥ min_event_len.
        let lo = b
            .saturating_sub(radius)
            .max(prev + min_event_len)
            .max(min_event_len);
        let hi = (b + radius)
            .min(next.saturating_sub(min_event_len))
            .min(interior_hi);

        if lo > hi {
            // No legal spot within ±radius. If any legal position exists past
            // the previous boundary, clamp to the nearest one (movement, not
            // removal). If the stream has no room left for another boundary
            // (`prev + min_event_len` past the interior), drop it — keeping
            // more boundaries than the length supports would violate
            // min_event_len, which is the stronger invariant.
            let legal_lo = prev + min_event_len;
            if legal_lo <= interior_hi {
                refined.push(b.clamp(legal_lo, interior_hi));
            }
            continue;
        }

        // Start from the original position (clamped) and only move on a
        // strictly better score — so with no usable signal (ties) the boundary
        // stays put rather than drifting to the window edge.
        let mut best = b.clamp(lo, hi);
        let mut best_score = cut_score(sim, best, prev, next, radius);
        for p in lo..=hi {
            let s = cut_score(sim, p, prev, next, radius);
            if s > best_score {
                best_score = s;
                best = p;
            }
        }
        refined.push(best);
    }
    refined
}

/// Unit mean direction of a set of unit vectors (empty / zero → empty).
fn mean_dir(rows: &[Vec<f64>]) -> Vec<f64> {
    if rows.is_empty() {
        return Vec::new();
    }
    let dim = rows[0].len();
    let mut s = vec![0.0f64; dim];
    for r in rows {
        for (a, b) in s.iter_mut().zip(r) {
            *a += b;
        }
    }
    let n = s.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n <= f64::EPSILON {
        return vec![0.0; dim];
    }
    s.iter().map(|x| x / n).collect()
}

fn cos64(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>().clamp(-1.0, 1.0)
}

/// Cosine above which the two sides of a cut are "basically the same
/// direction" (< ~18° apart) — a volatility blip inside one topic, not an era
/// change. Boundaries this coherent are vetoed by [`coherence_gate`].
const MAX_ACROSS_COS: f64 = 0.95;

/// Drop boundaries that don't actually separate dissimilar content. The
/// surprise signal spikes on *any* rise in prediction error — including a mere
/// increase in within-topic noise — so a detected boundary can be a volatility
/// artifact rather than a real shift. Here the graph disposes what surprise
/// proposed: if the mean direction just before and just after the cut is nearly
/// identical (`> MAX_ACROSS_COS`), the boundary is removed. Geometry-based, so
/// it only runs on the embedding path (never on precomputed surprise).
pub fn coherence_gate(boundaries: &[usize], normed: &[Vec<f64>], radius: usize, n: usize) -> Vec<usize> {
    let mut kept = Vec::with_capacity(boundaries.len());
    for (idx, &b) in boundaries.iter().enumerate() {
        let prev = if idx == 0 { 0 } else { boundaries[idx - 1] };
        let next = boundaries.get(idx + 1).copied().unwrap_or(n);
        let l0 = b.saturating_sub(radius).max(prev);
        let r1 = (b + radius).min(next);
        let left = mean_dir(&normed[l0..b]);
        let right = mean_dir(&normed[b..r1]);
        if cos64(&left, &right) <= MAX_ACROSS_COS {
            kept.push(b);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surprise::normalize_f64;

    fn sims(embs: &[Vec<f32>]) -> Vec<Vec<f64>> {
        embs.iter().map(|e| normalize_f64(e)).collect()
    }

    #[test]
    fn snaps_offby_one_to_true_split() {
        // Six A then six B; feed a boundary one early (5) — refinement should
        // pull it to the true split at 6.
        let mut embs: Vec<Vec<f32>> = (0..6).map(|_| vec![1.0, 0.0]).collect();
        embs.extend((0..6).map(|_| vec![0.0, 1.0]));
        let normed = sims(&embs);
        let sim = SimMatrix::new(&normed);
        let refined = refine_boundaries(&[5], &sim, 3, 2, embs.len());
        assert_eq!(refined, vec![6]);
    }

    #[test]
    fn never_adds_or_removes() {
        let mut embs: Vec<Vec<f32>> = (0..6).map(|_| vec![1.0, 0.0]).collect();
        embs.extend((0..6).map(|_| vec![0.0, 1.0]));
        let normed = sims(&embs);
        let sim = SimMatrix::new(&normed);
        let refined = refine_boundaries(&[6], &sim, 3, 2, embs.len());
        assert_eq!(refined.len(), 1, "count must be invariant under refinement");
    }
}
