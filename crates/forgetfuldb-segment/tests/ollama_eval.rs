//! The cat-and-wolf salience/retrieval eval run against a **real semantic
//! embedder** (`embeddinggemma` served by local Ollama), to see whether the
//! coarse scene-merging under `hashed_bow` was the embedder's fault (it is).
//!
//! Ignored by default (needs a running Ollama + the model). Run explicitly:
//!   ollama pull embeddinggemma
//!   cargo test -p forgetfuldb-segment --test ollama_eval -- --ignored --nocapture

use forgetfuldb_core::config::SegmentConfig;
use forgetfuldb_core::ingest::{atomize, structure_hint};
use forgetfuldb_core::salience::{analyze_neighbors, salience, Neighbor, NeighborParams};
use forgetfuldb_embed::{cosine_similarity, EmbeddingProvider, OllamaEmbeddings};

// Four scenes, ~5 sentences each — long enough that the first scene boundary
// clears the segmenter's warm-up (a 3-sentence scene would hide it).
const STORY: &str = "Mira the striped cat lived in a sunny meadow beside the quiet village. \
Every morning Mira chased grasshoppers through the tall meadow grass. \
The kind villagers left Mira warm bowls of milk near the wooden fence. \
Mira loved to nap on the smooth warm stones at the meadow's edge. \
The little cat knew every flower and hedgerow in the peaceful meadow. \
One cold afternoon Mira wandered down to the rushing river. \
At the river Mira met Fenn, a grey wolf drinking from the water. \
Fenn the wolf had traveled far along the river looking for food. \
The wolf's fur was matted and his eyes were tired from the long journey. \
Mira watched the big grey wolf warily from the mossy riverbank. \
Fenn offered Mira a bargain to guard the meadow from hungry foxes. \
In return Mira would share her evening fish with the grey wolf. \
The cat and the wolf sealed their bargain with a solemn nod. \
They agreed the wolf would patrol the fields while the cat kept watch. \
It was an unusual pact between a small cat and a large grey wolf. \
Through the long winter the wolf guarded the snowy meadow every night. \
No hungry fox ever again crept into the frozen snowy meadow. \
The villagers marveled at how safe their little valley had become. \
Mira and Fenn stayed loyal friends until the warm spring returned. \
Their strange friendship was remembered in the village for many years.";

fn provider() -> OllamaEmbeddings {
    OllamaEmbeddings::new("http://127.0.0.1:11434".to_string(), "embeddinggemma".to_string())
        .expect("Ollama must be running with the embeddinggemma model pulled")
}

fn clip(s: &str) -> String {
    if s.chars().count() <= 70 {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(67).collect::<String>())
    }
}

fn argmax(v: &[f64]) -> usize {
    v.iter().enumerate().fold((0, f64::MIN), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0
}

#[test]
#[ignore = "requires a running Ollama with embeddinggemma"]
fn cat_and_wolf_with_embeddinggemma() {
    let p = provider();
    println!("\nembedder: {} (dim {})", p.model_id(), p.dim());

    let atoms = atomize(STORY, structure_hint(STORY));
    let atom_embs: Vec<Vec<f32>> = atoms.iter().map(|a| p.embed(a)).collect();

    // Short document → small window so a scene shift isn't hidden by warm-up.
    let cfg = SegmentConfig { window_size: 3, threshold_window: 6, ..SegmentConfig::default() };
    let result = forgetfuldb_segment::segment_with_embeddings(&atom_embs, &cfg, None);
    let chunks: Vec<String> = result.events.iter().map(|e| atoms[e.start..e.end].join(" ")).collect();
    let chunk_embs: Vec<Vec<f32>> = chunks.iter().map(|c| p.embed(c)).collect();

    println!("\n=== segmented into {} chunks (vs 3 coarse under hashed_bow) ===", chunks.len());
    for (i, c) in chunks.iter().enumerate() {
        println!("  chunk {i}: {}", clip(c));
    }
    assert!(chunks.len() >= 2, "the story should segment into scenes");

    // ── Case 1: empty store — all novel ──
    println!("\n=== Case 1: EMPTY store ===");
    for (i, c) in chunks.iter().enumerate() {
        let stats = analyze_neighbors(&[], 1.0, &NeighborParams::default());
        println!("  chunk {i}: surprise={:.2}  [{}]", stats.surprise_term, clip(c));
    }

    // ── Case 2: the meadow scene is already known ──
    println!("\n=== Case 2: store already knows 'a cat in a meadow' ===");
    let prior = p.embed("A striped cat named Mira lived in a sunny green meadow near a village");
    let mut surprises = Vec::new();
    for (i, ce) in chunk_embs.iter().enumerate() {
        let sim = cosine_similarity(ce, &prior) as f64;
        let stats = analyze_neighbors(&[Neighbor { similarity: sim, age_days: 2.0 }], 30.0, &NeighborParams::default());
        surprises.push(stats.surprise_term);
        println!("  chunk {i}: sim-to-known={sim:.2} surprise={:.2} salience={:.2}  [{}]", stats.surprise_term, salience(&stats, 1.0), clip(&chunks[i]));
    }
    let known = argmax(&chunk_embs.iter().map(|c| cosine_similarity(c, &prior) as f64).collect::<Vec<_>>());
    let novel_max = surprises.iter().cloned().fold(0.0_f64, f64::max);
    println!("  → most-known chunk #{known} surprise={:.2}; most-novel surprise={:.2}", surprises[known], novel_max);
    assert!(surprises[known] < novel_max, "the already-known chunk must be less surprising than the most novel");
    assert!(chunks[known].to_lowercase().contains("meadow"), "the known chunk should be the meadow scene");

    // ── Questions ──
    // Note: a flowing *narrative* is semantically continuous, so the chunk
    // boundaries above are soft (unlike a distinct-topic doc). Question answering
    // is retrieval, though, and semantic embeddings retrieve the relevant
    // *sentence* cleanly — which is the "ask questions about the story" test.
    println!("\n=== Questions (sentence-level semantic retrieval) ===");
    let ask = |q: &str| -> (usize, f64) {
        let qe = p.embed(q);
        let sims: Vec<f64> = atom_embs.iter().map(|a| cosine_similarity(a, &qe) as f64).collect();
        let idx = argmax(&sims);
        (idx, sims[idx])
    };
    let questions = [
        ("Who did Mira meet at the river?", &["river", "wolf", "fenn"][..]),
        ("Where did the cat make its home?", &["meadow", "village"][..]),
        // "deal" ↔ "pact"/"agreed" is a semantic match no lexical embedder makes.
        ("What deal did the wolf offer the cat?", &["bargain", "pact", "agreed", "guard", "patrol", "fish", "fox"][..]),
        ("What happened during the winter?", &["winter", "snowy", "guard"][..]),
    ];
    let mut story_scores = Vec::new();
    for (q, expect) in questions {
        let (idx, score) = ask(q);
        story_scores.push(score);
        println!("  Q: {q}\n     → \"{}\" ({score:.2})", clip(&atoms[idx]));
        assert!(
            expect.iter().any(|k| atoms[idx].to_lowercase().contains(k)),
            "top sentence for {q:?} should mention one of {expect:?}, got: {}",
            atoms[idx]
        );
    }

    // Absent content: dense embeddings have a high baseline, so compare
    // *relative* — the off-topic question must score below every real one.
    let (_, absent) = ask("How did the fire dragon defend the stone castle tower?");
    let min_story = story_scores.iter().cloned().fold(f64::MAX, f64::min);
    println!("\n  Q (absent from story) → best score {absent:.2}; lowest story score {min_story:.2}");
    assert!(absent < min_story, "an off-topic question should score below every real one");

    println!("\n=== eval complete ===\n");
}
