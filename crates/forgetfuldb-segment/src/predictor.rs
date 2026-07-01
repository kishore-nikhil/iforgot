//! Predictors — form the expectation of the next embedding from the prior
//! window (FR-1). The prediction is compared against the entry that actually
//! arrived; the gap is *surprise*.
//!
//! Every predictor returns an **L2-normalized** vector, because cosine assumes
//! unit vectors. Inputs are already normalized by [`crate::surprise`] before
//! they reach a predictor.

use forgetfuldb_core::config::PredictorKind;

/// Forms the predicted next embedding from an ordered window (oldest first).
pub trait Predictor {
    /// `window` is the ordered prior entries, oldest first. The return value
    /// MUST be L2-normalized (FR-1).
    fn predict(&self, window: &[Vec<f32>]) -> Vec<f32>;
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
    if norm > f64::EPSILON {
        for x in v.iter_mut() {
            *x = (*x as f64 / norm) as f32;
        }
    }
}

/// Flat mean of the window — every prior entry weighted equally. The baseline
/// V-3 measures the weighted predictor against: on a moving topic a flat mean
/// lags the center and over-cuts.
pub struct CentroidPredictor;

impl Predictor for CentroidPredictor {
    fn predict(&self, window: &[Vec<f32>]) -> Vec<f32> {
        let dim = window.first().map(|v| v.len()).unwrap_or(0);
        let mut sum = vec![0f32; dim];
        for v in window {
            for (s, x) in sum.iter_mut().zip(v) {
                *s += *x;
            }
        }
        l2_normalize(&mut sum);
        sum
    }
}

/// Exponential recency weighting: the most recent entry gets weight `1`, the
/// one before it `decay`, then `decay²`, … This tracks a *moving* center, so a
/// gradual drift stays unsurprising (the prediction follows the arc) while a
/// hard jump still spikes. The default predictor.
pub struct WeightedPredictor {
    /// Per-step decay in `(0, 1]`. Smaller = more myopic (recent-dominated).
    pub decay: f64,
}

impl Predictor for WeightedPredictor {
    fn predict(&self, window: &[Vec<f32>]) -> Vec<f32> {
        let dim = window.first().map(|v| v.len()).unwrap_or(0);
        let mut sum = vec![0f32; dim];
        let n = window.len();
        for (idx, v) in window.iter().enumerate() {
            // age 0 = most recent (last in the window).
            let age = (n - 1 - idx) as i32;
            let w = self.decay.powi(age) as f32;
            for (s, x) in sum.iter_mut().zip(v) {
                *s += w * *x;
            }
        }
        l2_normalize(&mut sum);
        sum
    }
}

/// OPTIONAL / stubbed (FR-1). A real implementation would extrapolate the
/// window's trajectory (`v_last + (v_last − v_prev)`) so a steady drift is
/// predicted *ahead* rather than lagged. Until then it falls back to the
/// weighted predictor so selecting `"extrapolate"` is never a silent no-op.
///
/// TODO(tier-2): linear/Holt trajectory extrapolation + a V-3-style test that
/// it beats `WeightedPredictor` on a constant-velocity drift.
pub struct ExtrapolatePredictor;

impl Predictor for ExtrapolatePredictor {
    fn predict(&self, window: &[Vec<f32>]) -> Vec<f32> {
        WeightedPredictor { decay: 0.5 }.predict(window)
    }
}

/// Build the predictor selected by config.
pub fn predictor_for(kind: PredictorKind, weight_decay: f64) -> Box<dyn Predictor> {
    match kind {
        PredictorKind::Centroid => Box::new(CentroidPredictor),
        PredictorKind::Weighted => Box::new(WeightedPredictor { decay: weight_decay }),
        PredictorKind::Extrapolate => Box::new(ExtrapolatePredictor),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictions_are_unit_norm() {
        let win = vec![vec![3.0f32, 0.0, 0.0], vec![0.0, 4.0, 0.0]];
        for p in [
            Box::new(CentroidPredictor) as Box<dyn Predictor>,
            Box::new(WeightedPredictor { decay: 0.5 }),
        ] {
            let v = p.predict(&win);
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "predictor output must be unit norm, got {norm}");
        }
    }

    #[test]
    fn weighted_leans_recent() {
        // Window ends on the y-axis; weighting should pull the prediction
        // toward y more than a flat centroid does.
        let win = vec![vec![1.0f32, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let w = WeightedPredictor { decay: 0.3 }.predict(&win);
        let c = CentroidPredictor.predict(&win);
        assert!(w[1] > c[1], "weighted should favor the recent y-axis entry");
    }

    #[test]
    fn empty_window_does_not_panic() {
        assert!(CentroidPredictor.predict(&[]).is_empty());
    }
}
