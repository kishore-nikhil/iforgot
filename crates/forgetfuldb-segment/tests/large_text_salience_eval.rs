//! Evaluation of the large-text salience hypothesis (see
//! `docs/large-text-ingest.md`). Pastes a *made-up* cat-and-wolf story — content
//! no pretrained model has memorized, so this isolates the memory *system*, not
//! the LLM's priors — then: atomize → segment → score salience per chunk, in two
//! store states, and answer questions by retrieval.
//!
//! Run it and watch the numbers:  `cargo test -p forgetfuldb-segment --test
//! large_text_salience_eval -- --nocapture`
//!
//! Caveat: the default embedder is `hashed_bow` (lexical, not semantic), so
//! retrieval here is word-overlap. It demonstrates the mechanism, not semantic
//! depth (that needs a real local model).

use forgetfuldb_core::config::SegmentConfig;
use forgetfuldb_core::salience::{analyze_neighbors, salience, Neighbor, NeighborParams};
use forgetfuldb_embed::{cosine_similarity, EmbeddingProvider, HashedBagOfWords};
use forgetfuldb_segment::{segment_with_embeddings, Event};

/// A made-up story: four scenes (home / river / bargain / winter), each with a
/// distinct vocabulary so segmentation has real boundaries to find.
const STORY: &str = "Mira the striped cat lived in a sunny meadow beside the quiet village. \
Every morning Mira chased grasshoppers through the tall yellow meadow grass. \
The kind villagers left Mira warm bowls of milk near the meadow fence. \
One cold afternoon Mira wandered down to the rushing river. \
At the river Mira met Fenn a grey wolf drinking from the cold water. \
Fenn the wolf had traveled far along the river looking for food. \
Fenn offered Mira a bargain to guard the meadow from hungry foxes. \
In return Mira would share her evening fish with the grey wolf. \
The cat and the wolf sealed their bargain beside the flowing river. \
Through the long winter the wolf guarded the snowy meadow every night. \
No hungry fox ever again crept into the cold snowy meadow. \
Mira and Fenn stayed loyal friends until the warm spring returned.";

/// Split prose into sentence atoms — the minimal `atomize` the design doc puts
/// in `core::ingest`; inlined here so the eval is self-contained.
fn atomize(text: &str) -> Vec<String> {
    text.split(". ")
        .map(|s| s.trim().trim_end_matches('.').trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn embed_all(p: &dyn EmbeddingProvider, texts: &[String]) -> Vec<Vec<f32>> {
    texts.iter().map(|t| p.embed(t)).collect()
}

/// Turn segmented index ranges into their joined chunk text + embedding.
fn chunk_texts(atoms: &[String], events: &[Event]) -> Vec<String> {
    events.iter().map(|e| atoms[e.start..e.end].join(". ")).collect()
}

#[test]
fn large_text_salience_and_retrieval_eval() {
    let p = HashedBagOfWords::new(256);
    let atoms = atomize(STORY);
    let atom_embs = embed_all(&p, &atoms);

    // Small window: this is a short document, so the predictor can't need more
    // history than a scene is long (design §V-10 re-baseline reasoning).
    let cfg = SegmentConfig { window_size: 2, threshold_window: 4, ..SegmentConfig::default() };
    let result = segment_with_embeddings(&atom_embs, &cfg, None);
    let chunks = chunk_texts(&atoms, &result.events);
    let chunk_embs = embed_all(&p, &chunks);

    println!("\n=== atomized into {} sentences, segmented into {} chunks ===", atoms.len(), chunks.len());
    for (i, c) in chunks.iter().enumerate() {
        println!("  chunk {i}: {}", short(c));
    }
    assert!(chunks.len() >= 2, "the story should segment into multiple scenes, got {}", chunks.len());

    // ── Case 1: empty store — everything is novel ────────────────────────
    println!("\n=== Case 1: EMPTY store (no prior memory) ===");
    println!("With nothing to compare against, every chunk is maximally novel;");
    println!("salience cannot discriminate — retrieval relevance answers questions.");
    for (i, chunk) in chunks.iter().enumerate() {
        let stats = analyze_neighbors(&[], 1.0, &NeighborParams::default());
        let s = salience(&stats, 1.0);
        println!("  chunk {i}: surprise={:.2} salience={:.2}  [{}]", stats.surprise_term, s, short(chunk));
        assert!(stats.surprise_term > 0.9, "empty store ⇒ chunk should be maximally novel");
    }

    // ── Case 2: populated store — salience discriminates ─────────────────
    // Pre-seed a "cat in a meadow" memory. Now the meadow chunk is old news
    // (low surprise) while the river/wolf/bargain chunks are new (high surprise).
    println!("\n=== Case 2: store already knows 'a cat in a meadow' ===");
    let prior = p.embed("A striped cat lived in a sunny green meadow near a small village");
    let mut surprises = Vec::new();
    for (i, ce) in chunk_embs.iter().enumerate() {
        let sim = cosine_similarity(ce, &prior) as f64;
        let neighbors = [Neighbor { similarity: sim, age_days: 2.0 }];
        let stats = analyze_neighbors(&neighbors, 30.0, &NeighborParams::default());
        let s = salience(&stats, 1.0);
        surprises.push(stats.surprise_term);
        println!(
            "  chunk {i}: sim-to-known={:.2} surprise={:.2} salience={:.2}  [{}]",
            sim,
            stats.surprise_term,
            s,
            short(&chunks[i])
        );
    }
    // The chunk most similar to the seeded memory is the "meadow" one; it should
    // be the LEAST surprising. A genuinely new chunk should out-surprise it.
    let known_idx = argmax(&chunk_embs.iter().map(|c| cosine_similarity(c, &prior) as f64).collect::<Vec<_>>());
    let novel_max = surprises.iter().cloned().fold(0.0_f64, f64::max);
    println!(
        "  → most-known chunk is #{known_idx} (surprise={:.2}); most-novel chunk surprise={:.2}",
        surprises[known_idx], novel_max
    );
    assert!(
        surprises[known_idx] < novel_max - 0.05,
        "the already-known chunk must be less surprising than the most novel one"
    );
    assert!(
        chunks[known_idx].to_lowercase().contains("meadow"),
        "the least-surprising chunk should be the meadow/home scene"
    );

    // ── Questions: retrieval in different scenarios ──────────────────────
    println!("\n=== Questions (retrieval = cosine to chunks) ===");
    let present = [
        ("Who was the grey wolf drinking at the river?", &["wolf", "river", "fenn"][..]),
        ("What bargain was made to guard against hungry foxes?", &["bargain", "guard", "fox"][..]),
        ("Where was the sunny meadow near the quiet village?", &["meadow", "village"][..]),
    ];
    for (q, expect_any) in present {
        let (idx, score) = rank_top(&p, q, &chunk_embs);
        let hit = chunks[idx].to_lowercase();
        println!("  Q: {q}\n     → chunk {idx} (score {score:.2}): {}", short(&chunks[idx]));
        assert!(
            expect_any.iter().any(|k| hit.contains(k)),
            "top chunk for {q:?} should mention one of {expect_any:?}, got: {hit}"
        );
    }

    // A question about content that is NOT in the story: the system should
    // decline (top score below a confidence floor), not invent a match.
    let absent = "How did the fire dragon defend the stone castle tower?";
    let (aidx, ascore) = rank_top(&p, absent, &chunk_embs);
    println!("\n  Q (absent from story): {absent}\n     → best chunk {aidx} score {ascore:.2}");
    assert!(ascore < 0.25, "a question about absent content should fall below the confidence floor, got {ascore:.2}");

    println!("\n=== eval complete ===\n");
}

fn rank_top(p: &dyn EmbeddingProvider, query: &str, chunk_embs: &[Vec<f32>]) -> (usize, f64) {
    let q = p.embed(query);
    let sims: Vec<f64> = chunk_embs.iter().map(|c| cosine_similarity(c, &q) as f64).collect();
    let idx = argmax(&sims);
    (idx, sims[idx])
}

fn argmax(v: &[f64]) -> usize {
    v.iter().enumerate().fold((0, f64::MIN), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0
}

fn short(s: &str) -> String {
    if s.len() <= 60 {
        s.to_string()
    } else {
        format!("{}…", &s[..57])
    }
}
