//! Rolling threshold — EM-LLM `μ + γσ` boundary detection (FR-4).
//!
//! A position `i` is a candidate boundary when its surprise exceeds the local
//! normal: `surprise[i] > μ + γσ`, with `μ, σ` computed over a **trailing
//! window of real surprises** (`τ` back, warm-up excluded).
//!
//! ## Why warm-up is excluded (correctness-critical, see design §3.1)
//!
//! Taking the spec literally — `μ, σ` over `surprise[i−τ..i]` with warm-up
//! defined as `0.0` — breaks V-2. At `i = warmup` the trailing window is all
//! zeros, so `μ = σ = 0` and the threshold is `0`; the first in-topic surprise
//! then exceeds it and cuts a spurious boundary. So statistics are taken over
//! *real* surprises only, a boundary needs `MIN_REAL_SAMPLES`, and `σ` is
//! floored at `SIGMA_FLOOR` — with no real spread nothing is an outlier and a
//! flat single-topic stream yields zero boundaries.

/// Minimum real (non-warm-up) surprises in the trailing window before a
/// position is eligible to be a boundary. Two is the fewest that define a σ.
const MIN_REAL_SAMPLES: usize = 2;

/// Floor on σ so a near-constant window can't drive the threshold to ~0 and
/// let ordinary noise cut. Below any real spread, nothing is a boundary.
const SIGMA_FLOOR: f64 = 1e-9;

/// Absolute surprise a boundary must also clear (design §3.2). A purely
/// relative `μ + γσ` threshold flags ~16% of *any* stationary signal at
/// `γ = 1` — so a flat single-topic stream (V-2) would cut on its own noise.
/// Because surprise is a cosine *distance* in `[0, 2]`, there is a natural
/// absolute scale: intra-topic jitter sits near `0` (cosine ≈ 1) while a real
/// era change sits near `1` (near-orthogonal). This floor separates the two,
/// exactly as the old epochs `drift_threshold` did. A boundary must be BOTH a
/// local outlier AND an absolutely meaningful shift.
const MIN_ABS_SURPRISE: f64 = 0.10;

/// Candidate boundary indices (FR-4). `warmup` positions are skipped and never
/// counted in the statistics — pass `0` for a precomputed-surprise signal that
/// has no warm-up region (the tier-3 hook). `min_event_len` prevents cutting a
/// segment shorter than itself (the just-closed side); the trailing remainder
/// is enforced later when events are built.
pub fn detect_boundaries(
    surprise: &[f64],
    warmup: usize,
    tau: usize,
    gamma: f64,
    min_event_len: usize,
) -> Vec<usize> {
    let n = surprise.len();
    let mut out = Vec::new();
    let start = warmup.min(n);
    let mut last_boundary = 0usize;

    for i in start..n {
        // Trailing window of real surprises strictly before i.
        let lo = i.saturating_sub(tau).max(warmup);
        let win = &surprise[lo..i];
        if win.len() < MIN_REAL_SAMPLES {
            continue;
        }
        let mu = win.iter().sum::<f64>() / win.len() as f64;
        let var = win.iter().map(|s| (s - mu) * (s - mu)).sum::<f64>() / win.len() as f64;
        let sigma = var.sqrt().max(SIGMA_FLOOR);

        let is_local_outlier = surprise[i] > mu + gamma * sigma;
        let is_meaningful = surprise[i] > MIN_ABS_SURPRISE;
        if is_local_outlier && is_meaningful && i - last_boundary >= min_event_len {
            out.push(i);
            last_boundary = i;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_stream_has_no_boundaries() {
        // Uniform small surprise after warm-up: nothing exceeds its own local
        // noise. This is the V-2 mechanism in miniature.
        let mut s = vec![0.0; 30];
        for (i, v) in s.iter_mut().enumerate().skip(8) {
            // tiny deterministic ripple, no real outlier
            *v = 0.02 + 0.001 * ((i % 3) as f64);
        }
        assert!(detect_boundaries(&s, 8, 16, 1.0, 2).is_empty());
    }

    #[test]
    fn a_clear_spike_is_a_boundary() {
        let mut s = vec![0.0; 30];
        for v in s.iter_mut().skip(8) {
            *v = 0.02;
        }
        s[15] = 0.9; // a lone spike
        let b = detect_boundaries(&s, 8, 16, 1.0, 2);
        assert_eq!(b, vec![15]);
    }

    #[test]
    fn min_event_len_absorbs_adjacent_spikes() {
        let mut s = vec![0.0; 30];
        for v in s.iter_mut().skip(8) {
            *v = 0.02;
        }
        s[15] = 0.9;
        s[16] = 0.9; // one entry later — would make a length-1 segment
        let b = detect_boundaries(&s, 8, 16, 1.0, 2);
        assert_eq!(b, vec![15], "the adjacent spike must be absorbed by min_event_len");
    }

    #[test]
    fn no_boundary_before_two_real_samples() {
        // Even a big surprise right at warm-up's edge can't cut without enough
        // real history to define a σ (guards V-2's first-position failure).
        let mut s = vec![0.0; 20];
        s[8] = 0.9;
        s[9] = 0.9;
        let b = detect_boundaries(&s, 8, 16, 1.0, 2);
        assert!(b.is_empty(), "positions 8,9 lack 2 prior real samples");
    }
}
