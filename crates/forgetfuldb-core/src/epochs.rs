//! Epochs — organizing a lifetime of memories into eras.
//!
//! The **boundaries** between eras are now found by `forgetfuldb-segment`
//! (tier-2 embedding-space *surprise* segmentation): recency-weighted
//! prediction error against a rolling `μ + γσ` threshold, refined by graph
//! modularity. That replaced the drift-and-hysteresis segmenter that used to
//! live here — see `docs/surprise-segmentation.md`.
//!
//! This module keeps only what the rest of the engine needs to *consume* eras,
//! and stays pure and deterministic (no SQLite, no model, no clock):
//! - [`centroid_of`] reduces an era's members to its identity direction, and
//! - [`epoch_index_at`] maps a timestamp to the era it falls in.
//!
//! The consolidator assembles the stored `Epoch` rows (timestamps, centroids,
//! summaries) from segment's index ranges; retrieval resolves an era ordinal to
//! its time window via [`epoch_index_at`].

/// Unit-normalized mean of a set of embeddings — an era's identity direction.
/// Each member is normalized first so a long vector can't dominate the mean.
/// Empty input (or a zero-sum) yields an empty / zero vector (no direction).
pub fn centroid_of(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    let dim = embeddings[0].len();
    let mut sum = vec![0.0f64; dim];
    for e in embeddings {
        let norm = e.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        if norm <= f64::EPSILON {
            continue;
        }
        for (s, &x) in sum.iter_mut().zip(e) {
            *s += x as f64 / norm;
        }
    }
    let mag = sum.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag <= f64::EPSILON {
        return vec![0.0; dim];
    }
    sum.iter().map(|&x| (x / mag) as f32).collect()
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

    #[test]
    fn centroid_of_is_unit_and_scale_invariant() {
        let a = centroid_of(&[vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]]);
        let mag: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 1e-5, "centroid must be unit norm");
        // Scaling a member must not change the direction.
        let b = centroid_of(&[vec![10.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]]);
        assert!((a[0] - b[0]).abs() < 1e-5);
    }

    #[test]
    fn centroid_of_handles_empty_and_zero() {
        assert!(centroid_of(&[]).is_empty());
        assert_eq!(centroid_of(&[vec![0.0, 0.0]]), vec![0.0, 0.0]);
    }
}
