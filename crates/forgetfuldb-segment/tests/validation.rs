//! Validation suite for `forgetfuldb-segment`, one test per spec item V-1..V-8
//! and V-11. All fixtures are deterministic — no network, no real model.

use forgetfuldb_core::config::{PredictorKind, SegmentConfig};
use forgetfuldb_embed::EmbeddingProvider;
use forgetfuldb_segment::{segment, segment_with_embeddings, Event, SegmentResult};

const DIM: usize = 16;

/// Deterministic per-entry jitter in `[0, mag)`, hashed from the index so
/// vectors in a block are distinct but reproducible (no RNG — NFR-2).
fn jitter(i: usize, salt: usize) -> f32 {
    let h = ((i as u64).wrapping_mul(2_654_435_761).wrapping_add(salt as u64 * 40_503)) % 1000;
    h as f32 / 1000.0
}

/// Unit-ish vector near basis axis `axis`, with small deterministic noise on a
/// couple of other axes.
fn near(axis: usize, i: usize, mag: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[axis] = 1.0;
    v[(axis + 3) % DIM] += mag * jitter(i, 1);
    v[(axis + 7) % DIM] += mag * jitter(i, 2);
    v
}

fn boundaries(r: &SegmentResult) -> Vec<usize> {
    r.events.iter().skip(1).map(|e| e.start).collect()
}

fn assert_covers(events: &[Event], n: usize) {
    assert_eq!(events.first().map(|e| e.start), Some(0));
    assert_eq!(events.last().map(|e| e.end), Some(n));
    for w in events.windows(2) {
        assert_eq!(w[0].end, w[1].start, "events must be contiguous");
    }
    assert!(events.iter().all(|e| e.end > e.start));
}

// ── V-1: three-era synthetic (the core test) ────────────────────────────────
#[test]
fn v1_three_era_exact_boundaries() {
    let mut embs = Vec::new();
    for i in 0..10 {
        embs.push(near(0, i, 0.02));
    }
    for i in 10..20 {
        embs.push(near(1, i, 0.02));
    }
    for i in 20..30 {
        embs.push(near(2, i, 0.02));
    }
    let r = segment_with_embeddings(&embs, &SegmentConfig::default(), None);
    assert_covers(&r.events, 30);
    assert_eq!(boundaries(&r), vec![10, 20], "clean orthogonal blocks must split exactly at 10 and 20");
}

// ── V-2: single-topic null ───────────────────────────────────────────────────
#[test]
fn v2_single_topic_null() {
    let embs: Vec<Vec<f32>> = (0..30).map(|i| near(0, i, 0.03)).collect();
    let r = segment_with_embeddings(&embs, &SegmentConfig::default(), None);
    assert_eq!(r.events, vec![Event { start: 0, end: 30 }], "a flat single topic must be one event");
}

// ── V-3: gradual drift, weighted tracks it, centroid over-cuts ───────────────
#[test]
fn v3_gradual_drift_weighted_beats_centroid() {
    // A smooth continuous rotation through the embedding plane — the topic
    // slides, no hard jump. Fast enough to be in the regime where the predictor
    // choice matters: a flat centroid lags the moving center and over-cuts,
    // while the recency-weighted predictor tracks it and stays under the floor.
    let step = 8.0_f32.to_radians();
    let mut embs = Vec::new();
    for i in 0..30 {
        let a = i as f32 * step;
        let mut v = vec![0.0f32; DIM];
        v[0] = a.cos();
        v[1] = a.sin();
        v[5] += 0.01 * jitter(i, 3);
        embs.push(v);
    }
    let weighted = SegmentConfig { predictor: PredictorKind::Weighted, ..SegmentConfig::default() };
    let centroid = SegmentConfig { predictor: PredictorKind::Centroid, ..SegmentConfig::default() };

    let wn = boundaries(&segment_with_embeddings(&embs, &weighted, None)).len();
    let cn = boundaries(&segment_with_embeddings(&embs, &centroid, None)).len();

    assert!(wn <= 2, "weighted should barely cut a smooth drift, got {wn}");
    assert!(wn < cn, "weighted ({wn}) must cut strictly fewer than centroid ({cn}) on a moving topic");
}

// ── V-4: normalization guard ─────────────────────────────────────────────────
#[test]
fn v4_normalization_guard() {
    // A zero vector (past the warm-up region) is present in BOTH streams, so
    // the only difference between them is scale — isolating the two properties:
    // scale-invariance (identical surprise) and zero-handling (surprise 1.0).
    let zero_at = 20;
    let mut base = Vec::new();
    for i in 0..15 {
        base.push(near(0, i, 0.02));
    }
    for i in 15..30 {
        base.push(near(1, i, 0.02));
    }
    base[zero_at] = vec![0.0; DIM];
    let scaled: Vec<Vec<f32>> = base.iter().map(|v| v.iter().map(|x| x * 10.0).collect()).collect();

    let cfg = SegmentConfig::default();
    let rb = segment_with_embeddings(&base, &cfg, None);
    let rs = segment_with_embeddings(&scaled, &cfg, None);

    assert!(rs.surprise.iter().all(|x| x.is_finite()), "no NaN/Inf with unnormalized + zero vectors");
    assert_eq!(rs.surprise[zero_at], 1.0, "the zero vector must be maximally surprising");
    // Scaling ×10 must not change the surprise signal anywhere (FR-2).
    for i in 0..30 {
        assert!((rb.surprise[i] - rs.surprise[i]).abs() < 1e-6, "scale changed surprise at {i}");
    }
}

// ── V-5: threshold locality (adapts to local variance) ───────────────────────
#[test]
fn v5_threshold_locality() {
    // Calm first half, then a volatile-but-same-topic second half: big
    // fluctuations that never leave topic A. A global threshold would fire on
    // the loud half; the local μ+γσ must not.
    let mut embs = Vec::new();
    for i in 0..15 {
        embs.push(near(0, i, 0.02)); // calm
    }
    for i in 15..30 {
        // loud noise on unrelated axes but the dominant component stays axis 0
        let mut v = near(0, i, 0.6);
        v[0] = 1.0;
        embs.push(v);
    }
    let cfg = SegmentConfig::default();
    let r = segment_with_embeddings(&embs, &cfg, None);

    // The local detector: at most the single calm→loud transition may cut; the
    // loud half's internal fluctuations must not.
    let in_loud = boundaries(&r).iter().filter(|&&b| b > 17).count();
    assert_eq!(in_loud, 0, "local threshold must not cut inside the volatile-but-stable half");

    // A GLOBAL threshold on the same surprise would fire repeatedly in the loud
    // half — confirming the locality actually did work.
    let global_hits = global_threshold_hits(&r.surprise, 8, 1.0);
    let loud_global = global_hits.iter().filter(|&&b| b > 17).count();
    assert!(loud_global > in_loud, "a global threshold should over-cut the loud half (got {loud_global})");
}

/// A deliberately naive global-threshold detector (μ+γσ over the whole signal)
/// used only as a foil for V-5.
fn global_threshold_hits(surprise: &[f64], warmup: usize, gamma: f64) -> Vec<usize> {
    let real: Vec<f64> = surprise.iter().copied().skip(warmup).collect();
    if real.len() < 2 {
        return Vec::new();
    }
    let mu = real.iter().sum::<f64>() / real.len() as f64;
    let var = real.iter().map(|s| (s - mu) * (s - mu)).sum::<f64>() / real.len() as f64;
    let sigma = var.sqrt();
    (warmup..surprise.len()).filter(|&i| surprise[i] > mu + gamma * sigma).collect()
}

// ── V-6: min_event_len enforcement (blocks of 10, 1, 19) ─────────────────────
#[test]
fn v6_min_event_len() {
    let mut embs = Vec::new();
    for i in 0..10 {
        embs.push(near(0, i, 0.02));
    }
    embs.push(near(1, 10, 0.02)); // the length-1 fragment
    for i in 11..30 {
        embs.push(near(2, i, 0.02));
    }
    let cfg = SegmentConfig { min_event_len: 2, ..SegmentConfig::default() };
    let r = segment_with_embeddings(&embs, &cfg, None);
    assert_covers(&r.events, 30);
    assert!(r.events.iter().all(|e| e.len() >= 2), "no event shorter than min_event_len");
    let b = boundaries(&r);
    assert!(!(b.contains(&10) && b.contains(&11)), "the length-1 fragment must be absorbed, not kept");
}

// ── V-7: refinement correctness (with ambiguous transition entries) ──────────
#[test]
fn v7_refinement_snaps_to_true_split() {
    use forgetfuldb_segment::{refine_boundaries, SimMatrix};

    // Block A, then two ambiguous transition entries that LEAN — entry 8 mostly
    // A, entry 9 mostly B — then block B. The unique coherence-maximizing split
    // is at 9 (the A-leaning entry joins A, the B-leaning entry joins B).
    let mut embs: Vec<Vec<f32>> = (0..8).map(|i| near(0, i, 0.02)).collect();
    let mut lean_a = vec![0.0f32; DIM];
    lean_a[0] = 0.9;
    lean_a[1] = 0.4;
    let mut lean_b = vec![0.0f32; DIM];
    lean_b[0] = 0.4;
    lean_b[1] = 0.9;
    embs.push(lean_a);
    embs.push(lean_b);
    embs.extend((10..18).map(|i| near(1, i, 0.02)));

    let normed: Vec<Vec<f64>> = embs
        .iter()
        .map(|v| {
            let n = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            v.iter().map(|x| *x as f64 / n).collect()
        })
        .collect();
    let sim = SimMatrix::new(&normed);

    // Feed a deliberately off-by-one candidate (8) — refinement must snap to 9.
    let refined = refine_boundaries(&[8], &sim, 3, 2, embs.len());
    assert_eq!(refined.len(), 1, "refinement must not add or remove boundaries");
    assert_eq!(refined[0], 9, "boundary should snap to the coherence-maximizing split");
}

// ── V-8: coverage & invariants + determinism (property test) ─────────────────
#[test]
fn v8_invariants_and_determinism() {
    // Deterministic LCG stands in for a property-test RNG (no new deps, NFR-2).
    let mut state: u64 = 0x1234_5678_9abc_def1;
    let mut rng = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };

    for _ in 0..200 {
        let n = (rng() % 60) as usize;
        let cfg = SegmentConfig {
            window_size: 1 + (rng() % 8) as usize,
            predictor: match rng() % 3 {
                0 => PredictorKind::Centroid,
                1 => PredictorKind::Weighted,
                _ => PredictorKind::Extrapolate,
            },
            weight_decay: 0.1 + (rng() % 9) as f64 / 10.0,
            threshold_window: 2 + (rng() % 20) as usize,
            gamma: (rng() % 30) as f64 / 10.0,
            refine_radius: (rng() % 5) as usize,
            min_event_len: 1 + (rng() % 4) as usize,
        };
        let embs: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let axis = (rng() % DIM as u32) as usize;
                near(axis, rng() as usize, 0.05)
            })
            .collect();

        let r1 = segment_with_embeddings(&embs, &cfg, None);
        // Coverage / ordering / no-overlap.
        if n == 0 {
            assert!(r1.events.is_empty());
            continue;
        }
        assert_covers(&r1.events, n);
        // Every event ≥ min_event_len except possibly the final remainder.
        for (k, e) in r1.events.iter().enumerate() {
            if k + 1 < r1.events.len() {
                assert!(e.len() >= cfg.min_event_len, "interior event shorter than min_event_len");
            }
        }
        // Determinism: identical inputs → identical output.
        let r2 = segment_with_embeddings(&embs, &cfg, None);
        assert_eq!(r1.events, r2.events, "segmentation must be deterministic");
        assert_eq!(r1.surprise, r2.surprise);
    }
}

// ── V-11: embedding-model provenance ─────────────────────────────────────────
struct StubEmbedder {
    id: &'static str,
}
impl EmbeddingProvider for StubEmbedder {
    fn name(&self) -> &'static str {
        "stub"
    }
    fn dim(&self) -> usize {
        DIM
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        // Map the first char to an axis so scripted eras are reproducible.
        let axis = (text.as_bytes().first().copied().unwrap_or(b'a') as usize) % DIM;
        near(axis, text.len(), 0.02)
    }
    fn model_id(&self) -> String {
        self.id.to_string()
    }
}

#[test]
fn v11_provenance_is_plumbed() {
    let entries: Vec<String> = (0..12).map(|i| format!("{}{}", if i < 6 { 'a' } else { 'q' }, i)).collect();
    let cfg = SegmentConfig::default();

    let r1 = segment(&entries, &StubEmbedder { id: "model-x:16" }, &cfg);
    let r2 = segment(&entries, &StubEmbedder { id: "model-y:16" }, &cfg);

    assert_eq!(r1.embedding_model.as_deref(), Some("model-x:16"));
    assert_eq!(r2.embedding_model.as_deref(), Some("model-y:16"));
    // Same geometry, different provenance — FR-8 is plumbing, not geometry.
    assert_eq!(r1.events, r2.events);
}
