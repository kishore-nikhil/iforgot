//! `cargo bench -p forgetfuldb-segment`
//!
//! Two reports, no external harness (NFR-4):
//!
//! 1. **Throughput** — wall-clock of the surprise → threshold → refine pipeline
//!    across stream sizes, to show the per-session/day batch cost stays cheap
//!    and to expose the documented O(n²) refinement ceiling.
//!
//! 2. **Segmentation quality (the "memory works better" measurement)** — on
//!    synthetic streams with *known* era boundaries, how accurately does each
//!    method recover them? Accurate boundaries are the whole point: they are
//!    the units "consolidate within, preserve across" and epoch retrieval act
//!    on, so recovering the true eras is what keeps a relevant memory from
//!    being merged across a real topic shift (preserved longer) while routine
//!    inside an era is compressed. We compare the shipped **surprise** segmenter
//!    against the **centroid-drift** method it replaced (reconstructed here) and
//!    a naive **fixed-chunk** baseline (the "always cut a fixed fraction"
//!    failure mode), scoring boundary precision / recall / F1.

use forgetfuldb_core::config::{PredictorKind, SegmentConfig};
use forgetfuldb_segment::{segment_with_embeddings, Event};
use std::time::Instant;

const DIM: usize = 32;

// ── deterministic synthetic streams ─────────────────────────────────────────

/// Deterministic mean-zero noise in `[-0.5, 0.5)`. Centered so that adding it
/// to a basis vector is genuine *volatility* around that direction, not a DC
/// bias that silently drifts the mean off-axis.
fn jitter(i: usize, salt: usize) -> f32 {
    let h = ((i as u64).wrapping_mul(2_654_435_761).wrapping_add(salt as u64 * 40_503)) % 1000;
    h as f32 / 1000.0 - 0.5
}

/// Unit-ish vector near basis axis `axis` with small deterministic noise.
fn near(axis: usize, i: usize, mag: f32) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    v[axis % DIM] = 1.0;
    v[(axis + 3) % DIM] += mag * jitter(i, 1);
    v[(axis + 7) % DIM] += mag * jitter(i, 2);
    v
}

/// A vector rotated `deg` degrees in the (0,1) plane.
fn rotated(deg: f32, i: usize) -> Vec<f32> {
    let a = deg.to_radians();
    let mut v = vec![0.0f32; DIM];
    v[0] = a.cos();
    v[1] = a.sin();
    v[5] += 0.01 * jitter(i, 3);
    v
}

/// One labeled scenario: the embeddings and the true boundary indices.
struct Scenario {
    name: &'static str,
    embs: Vec<Vec<f32>>,
    truth: Vec<usize>,
}

fn scenarios() -> Vec<Scenario> {
    let mut out = Vec::new();

    // K clean orthogonal eras of length L — the easy case everyone should pass.
    for (k, l) in [(3usize, 12usize), (5, 10)] {
        let mut embs = Vec::new();
        let mut truth = Vec::new();
        for era in 0..k {
            if era > 0 {
                truth.push(era * l);
            }
            for i in 0..l {
                embs.push(near(era * 4, era * l + i, 0.03));
            }
        }
        out.push(Scenario { name: if k == 3 { "clean_3_eras" } else { "clean_5_eras" }, embs, truth });
    }

    // Single topic — the null. Any boundary here is a false positive.
    {
        let embs: Vec<Vec<f32>> = (0..40).map(|i| near(0, i, 0.04)).collect();
        out.push(Scenario { name: "single_topic_null", embs, truth: vec![] });
    }

    // A slow drift that should NOT cut, then a hard jump that SHOULD. Rewards a
    // predictor that tracks a moving center instead of over-cutting the drift.
    {
        let mut embs = Vec::new();
        for i in 0..20 {
            embs.push(rotated(i as f32 * 1.5, i)); // gentle 1.5°/step arc
        }
        for i in 20..40 {
            embs.push(near(9, i, 0.03)); // orthogonal jump
        }
        out.push(Scenario { name: "gradual_then_jump", embs, truth: vec![20] });
    }

    // Calm first half, then loud-but-same-topic second half. A global threshold
    // over-cuts the loud half; the local one should not.
    {
        let mut embs = Vec::new();
        for i in 0..20 {
            embs.push(near(0, i, 0.02));
        }
        for i in 20..40 {
            let mut v = near(0, i, 0.7);
            v[0] = 1.0;
            embs.push(v);
        }
        out.push(Scenario { name: "volatile_stable", embs, truth: vec![] });
    }

    // A subtle but real shift (~40°, cosine ≈ 0.77) — below the old method's
    // fixed 0.35 drift threshold, so it's invisible to centroid-drift, but a
    // clear local outlier to the adaptive μ+γσ. This is the case the fixed
    // threshold was replaced to handle.
    {
        let at = |deg: f32, i: usize| {
            let a = deg.to_radians();
            let mut v = vec![0.0f32; DIM];
            v[0] = a.cos();
            v[1] = a.sin();
            v[9] += 0.02 * jitter(i, 4);
            v
        };
        let mut embs = Vec::new();
        for i in 0..16 {
            embs.push(at(0.0, i));
        }
        for i in 16..32 {
            embs.push(at(40.0, i));
        }
        out.push(Scenario { name: "subtle_shift_40deg", embs, truth: vec![16] });
    }

    // Two subtle shifts then a sharp one — old fixed threshold catches only the
    // orthogonal jump; the adaptive method should catch all three.
    {
        let at = |deg: f32, i: usize| {
            let a = deg.to_radians();
            let mut v = vec![0.0f32; DIM];
            v[0] = a.cos();
            v[1] = a.sin();
            v[9] += 0.02 * jitter(i, 4);
            v
        };
        let mut embs = Vec::new();
        for i in 0..12 {
            embs.push(at(0.0, i));
        }
        for i in 12..24 {
            embs.push(at(35.0, i));
        }
        for i in 24..36 {
            embs.push(at(70.0, i));
        }
        for i in 36..48 {
            embs.push(near(20, i, 0.03)); // orthogonal jump
        }
        out.push(Scenario { name: "subtle_x2_then_sharp", embs, truth: vec![12, 24, 36] });
    }

    // Uneven eras — realistic mix of long and short stretches.
    {
        let lens = [14usize, 8, 20, 10];
        let mut embs = Vec::new();
        let mut truth = Vec::new();
        let mut pos = 0;
        for (era, &l) in lens.iter().enumerate() {
            if era > 0 {
                truth.push(pos);
            }
            for i in 0..l {
                embs.push(near(era * 5, pos + i, 0.03));
            }
            pos += l;
        }
        out.push(Scenario { name: "uneven_eras", embs, truth });
    }

    out
}

// ── methods under comparison ────────────────────────────────────────────────

fn boundaries(events: &[Event]) -> Vec<usize> {
    events.iter().skip(1).map(|e| e.start).collect()
}

fn surprise_method(embs: &[Vec<f32>], predictor: PredictorKind) -> Vec<usize> {
    let cfg = SegmentConfig { predictor, ..SegmentConfig::default() };
    boundaries(&segment_with_embeddings(embs, &cfg, None).events)
}

/// A faithful index-domain port of the centroid-drift + hysteresis segmenter
/// this crate replaced (`core::epochs` before the cutover): maintain the era
/// centroid, count consecutive "drifting" memories, cut after `hysteresis` in a
/// row if the closed era clears `min_size`. Reconstructed here only as the
/// before/after baseline.
fn centroid_drift_method(embs: &[Vec<f32>]) -> Vec<usize> {
    const DRIFT: f64 = 0.35;
    const HYST: usize = 3;
    const MIN_SIZE: usize = 4;
    let norm = |v: &[f32]| -> Vec<f64> {
        let n = v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
        if n <= f64::EPSILON { vec![0.0; v.len()] } else { v.iter().map(|&x| x as f64 / n).collect() }
    };
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    if embs.is_empty() {
        return vec![];
    }
    let dim = embs[0].len();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut sum = norm(&embs[0]);
    let mut run = 0usize;
    let mut run_start = 0usize;
    for i in 1..embs.len() {
        let e = norm(&embs[i]);
        let dir = norm(&sum.iter().map(|&x| x as f32).collect::<Vec<_>>());
        let drift = 1.0 - dot(&dir, &e);
        if drift > DRIFT {
            if run == 0 {
                run_start = i;
            }
            run += 1;
        } else {
            run = 0;
        }
        if run >= HYST {
            if run_start - start >= MIN_SIZE {
                out.push(run_start);
                start = run_start;
                sum = vec![0.0; dim];
                for q in &embs[run_start..=i] {
                    for (s, x) in sum.iter_mut().zip(norm(q)) {
                        *s += x;
                    }
                }
            }
            run = 0;
            continue;
        }
        if drift <= DRIFT {
            for (s, x) in sum.iter_mut().zip(e) {
                *s += x;
            }
        }
    }
    out
}

/// The naive foil: cut every `chunk` entries regardless of content — the
/// "always cut a fixed fraction" failure mode the spec warns against.
fn fixed_chunk_method(n: usize, chunk: usize) -> Vec<usize> {
    (chunk..n).step_by(chunk).collect()
}

// ── scoring ──────────────────────────────────────────────────────────────────

/// Precision / recall / F1 of detected boundaries vs truth, ±1 tolerance.
fn score(detected: &[usize], truth: &[usize]) -> (f64, f64, f64) {
    let mut used = vec![false; detected.len()];
    let mut tp = 0usize;
    for &g in truth {
        if let Some(k) = detected
            .iter()
            .enumerate()
            .filter(|(k, &d)| !used[*k] && d.abs_diff(g) <= 1)
            .min_by_key(|(_, &d)| d.abs_diff(g))
            .map(|(k, _)| k)
        {
            used[k] = true;
            tp += 1;
        }
    }
    let fp = detected.len() - tp;
    let fn_ = truth.len() - tp;
    let precision = if tp + fp == 0 { 1.0 } else { tp as f64 / (tp + fp) as f64 };
    let recall = if tp + fn_ == 0 { 1.0 } else { tp as f64 / (tp + fn_) as f64 };
    let f1 = if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) };
    (precision, recall, f1)
}

// ── reports ──────────────────────────────────────────────────────────────────

fn throughput_report() {
    println!("\n== Throughput (surprise → threshold → refine, WeightedPredictor) ==");
    println!("{:>8}  {:>12}  {:>14}  {:>10}", "n", "total", "per-entry", "entries/s");
    let cfg = SegmentConfig::default();
    for &n in &[128usize, 512, 1024, 2048, 4096] {
        // A multi-era stream so refinement actually has boundaries to move.
        let embs: Vec<Vec<f32>> = (0..n).map(|i| near((i / 20) * 3, i, 0.03)).collect();
        // warm the caches, then time a few reps.
        let _ = segment_with_embeddings(&embs, &cfg, None);
        let reps = if n <= 1024 { 50 } else { 10 };
        let t = Instant::now();
        for _ in 0..reps {
            let r = segment_with_embeddings(&embs, &cfg, None);
            std::hint::black_box(&r.events);
        }
        let per_call = t.elapsed() / reps;
        let per_entry = per_call / n as u32;
        let eps = n as f64 / per_call.as_secs_f64();
        println!("{n:>8}  {:>12?}  {:>14?}  {:>10.0}", per_call, per_entry, eps);
    }
    println!("(refinement is O(boundaries·radius²·d); the surprise+threshold pass is O(n·d).)");
}

/// A named boundary detector: embeddings → detected boundary indices.
type Method = (&'static str, fn(&[Vec<f32>]) -> Vec<usize>);

fn quality_report() {
    let methods: [Method; 4] = [
        ("surprise (new, weighted)", |e| surprise_method(e, PredictorKind::Weighted)),
        ("surprise (centroid pred)", |e| surprise_method(e, PredictorKind::Centroid)),
        ("centroid-drift (old)", centroid_drift_method),
        ("fixed-chunk/10 (naive)", |e| fixed_chunk_method(e.len(), 10)),
    ];

    let scen = scenarios();
    println!("\n== Segmentation quality: boundary recovery vs known eras (±1) ==");
    println!("Higher F1 = memories grouped into their true eras = relevant items");
    println!("preserved across real shifts, routine compressed within.\n");

    // Per-scenario F1 table.
    print!("{:<20}", "scenario");
    for (name, _) in &methods {
        print!("{:>26}", name);
    }
    println!();
    let mut totals = [(0.0f64, 0.0f64, 0.0f64); 4];
    for s in &scen {
        print!("{:<20}", s.name);
        for (mi, (_, f)) in methods.iter().enumerate() {
            let (p, r, f1) = score(&f(&s.embs), &s.truth);
            totals[mi].0 += p;
            totals[mi].1 += r;
            totals[mi].2 += f1;
            print!("{:>26}", format!("F1={f1:.2} (P{p:.2}/R{r:.2})"));
        }
        println!();
    }

    // Aggregate.
    let n = scen.len() as f64;
    println!();
    print!("{:<20}", "AVERAGE");
    for t in &totals {
        print!("{:>26}", format!("F1={:.2} (P{:.2}/R{:.2})", t.2 / n, t.0 / n, t.1 / n));
    }
    println!("\n");
}

fn main() {
    println!("forgetfuldb-segment benchmarks");
    quality_report();
    throughput_report();
}
