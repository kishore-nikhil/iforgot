//! Surprise — prediction error against the prior window (FR-2, FR-3).
//!
//! `surprise_i = 1 − cosine(predict(prior_window), v_i)`, clamped to `[0, 2]`.
//! Every embedding is L2-normalized on the way in (FR-2); a zero-norm vector
//! is treated as maximally surprising (`1.0`) rather than dividing by zero.
//! The first `window` entries have no full history and are **warm-up** —
//! surprise `0.0`, and (see [`crate::threshold`]) excluded from the rolling
//! statistics so they can never manufacture a boundary.

use crate::predictor::Predictor;

/// L2-normalize into `f32`, reporting whether the input was a zero vector.
pub(crate) fn normalize_f32(v: &[f32]) -> (Vec<f32>, bool) {
    let norm: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    if norm <= f64::EPSILON {
        (vec![0.0; v.len()], true)
    } else {
        (v.iter().map(|&x| (x as f64 / norm) as f32).collect(), false)
    }
}

/// L2-normalize into unit `f64` (used by the similarity source in refinement).
pub(crate) fn normalize_f64(v: &[f32]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    if norm <= f64::EPSILON {
        vec![0.0; v.len()]
    } else {
        v.iter().map(|&x| x as f64 / norm).collect()
    }
}

/// Cosine of two vectors as `f64`, guarding zero/degenerate inputs (→ 0.0).
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for k in 0..a.len().min(b.len()) {
        let (x, y) = (a[k] as f64, b[k] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= f64::EPSILON || nb <= f64::EPSILON {
        0.0
    } else {
        (dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0)
    }
}

/// One surprise value per entry (FR-3). The first `window` entries are warm-up
/// (`0.0`). For the rest, predict from the prior `window` normalized entries
/// and score `1 − cosine`. A zero-norm actual → `1.0`.
pub fn surprise_signal<P: Predictor + ?Sized>(embs: &[Vec<f32>], pred: &P, window: usize) -> Vec<f64> {
    let n = embs.len();
    let mut out = vec![0.0f64; n];
    if window == 0 || n == 0 {
        return out;
    }
    // FR-2: normalize every input once; remember which were zero vectors.
    let (normed, zero): (Vec<Vec<f32>>, Vec<bool>) = embs.iter().map(|e| normalize_f32(e)).unzip();

    for i in window..n {
        if zero[i] {
            out[i] = 1.0; // maximally surprising, never a divide-by-zero
            continue;
        }
        let predicted = pred.predict(&normed[i - window..i]);
        let cos = cosine(&predicted, &normed[i]);
        out[i] = (1.0 - cos).clamp(0.0, 2.0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predictor::WeightedPredictor;

    #[test]
    fn warmup_is_zero() {
        let embs: Vec<Vec<f32>> = (0..10).map(|_| vec![1.0, 0.0, 0.0]).collect();
        let s = surprise_signal(&embs, &WeightedPredictor { decay: 0.5 }, 4);
        assert!(s[0..4].iter().all(|&x| x == 0.0));
    }

    #[test]
    fn zero_vector_is_max_surprise() {
        // A zero vector at index 5 must yield surprise 1.0, no NaN.
        let mut embs: Vec<Vec<f32>> = (0..10).map(|_| vec![1.0, 0.0, 0.0]).collect();
        embs[5] = vec![0.0, 0.0, 0.0];
        let s = surprise_signal(&embs, &WeightedPredictor { decay: 0.5 }, 3);
        assert_eq!(s[5], 1.0);
        assert!(s.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn scale_invariance() {
        // Scaling every vector ×10 must not change the surprise signal (FR-2).
        let base: Vec<Vec<f32>> = (0..12)
            .map(|i| if i < 6 { vec![1.0, 0.0, 0.0] } else { vec![0.0, 1.0, 0.0] })
            .collect();
        let scaled: Vec<Vec<f32>> = base.iter().map(|v| v.iter().map(|x| x * 10.0).collect()).collect();
        let s1 = surprise_signal(&base, &WeightedPredictor { decay: 0.5 }, 3);
        let s2 = surprise_signal(&scaled, &WeightedPredictor { decay: 0.5 }, 3);
        for (a, b) in s1.iter().zip(&s2) {
            assert!((a - b).abs() < 1e-6, "scale changed surprise: {a} vs {b}");
        }
    }
}
