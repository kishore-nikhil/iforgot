//! End-to-end integration tests: init database -> ingest -> retrieve ->
//! consolidate, against a real on-disk SQLite file.

use forgetfuldb_consolidate::{consolidate, ExtractiveSummarizer};
use forgetfuldb_core::config::Config;
use forgetfuldb_core::types::MemoryType;
use forgetfuldb_retrieve::{retrieve, RetrieveOptions};
use forgetfuldb_store::pipeline::{ingest, warm_bloom, IngestRequest};
use forgetfuldb_store::Store;
use std::path::PathBuf;

/// Unique temp path per test so parallel tests don't collide.
struct TempDb(PathBuf);

impl TempDb {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "forgetfuldb-test-{name}-{}-{}.sqlite3",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        TempDb(path)
    }
}

impl Drop for TempDb {
    fn drop(&mut self) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{}", self.0.display(), suffix));
        }
    }
}

fn ingest_text(store: &Store, cfg: &Config, text: &str, tags: Vec<String>, mt: Option<MemoryType>) -> String {
    let mut bloom = warm_bloom(store).unwrap();
    let provider = forgetfuldb_embed::create_provider(&cfg.embedding_backend, cfg.embedding_dim).unwrap();
    ingest(
        store,
        &mut bloom,
        provider.as_ref(),
        cfg,
        IngestRequest {
            text: text.to_string(),
            source: Some("chat".to_string()),
            tags,
            memory_type: mt,
            session_id: Some("session-1".to_string()),
            role: Some("user".to_string()),
        },
    )
    .unwrap()
    .memory()
    .id
    .clone()
}

#[test]
fn init_creates_database_with_schema() {
    let db = TempDb::new("init");
    let store = Store::open(&db.0).expect("database initializes");
    let stats = store.stats().expect("schema present");
    assert_eq!(stats.total_memories, 0);
    assert!(db.0.exists());
}

#[test]
fn ingest_then_retrieve_returns_relevant_context_pack() {
    let db = TempDb::new("ingest-retrieve");
    let cfg = Config { sqlite_path: db.0.display().to_string(), ..Config::default() };
    let store = Store::open(&db.0).unwrap();

    ingest_text(
        &store,
        &cfg,
        "Plot Perfect billing is handled by Stripe with monthly invoices",
        vec!["project:plotperfect".to_string()],
        Some(MemoryType::Semantic),
    );
    ingest_text(&store, &cfg, "Watered the ficus and repotted the basil", vec![], None);

    let provider = forgetfuldb_embed::create_provider(&cfg.embedding_backend, cfg.embedding_dim).unwrap();
    let pack = retrieve(
        &store,
        provider.as_ref(),
        &cfg,
        "What do I know about Plot Perfect billing?",
        &RetrieveOptions { top_k: 5, ..Default::default() },
    )
    .unwrap();

    assert!(!pack.memories.is_empty());
    assert!(pack.memories[0].item.content.to_lowercase().contains("billing"));
    assert!(pack.memories[0].score.total > 0.0);
    // Score breakdown is part of the output contract.
    let json = serde_json::to_value(&pack).unwrap();
    assert!(json["memories"][0]["score"]["semantic_similarity"].is_number());
}

#[test]
fn full_lifecycle_ingest_consolidate_retrieve() {
    let db = TempDb::new("lifecycle");
    let cfg = Config { sqlite_path: db.0.display().to_string(), ..Config::default() };
    let store = Store::open(&db.0).unwrap();

    // A topic cluster plus a duplicate phrasing.
    for text in [
        "plot perfect billing refunds were discussed in standup",
        "plot perfect billing needs prorated invoices",
        "decided plot perfect billing moves to usage based pricing",
        "billing plot perfect discussed refunds were in standup", // near-dup of #1 (same tokens)
    ] {
        ingest_text(&store, &cfg, text, vec!["project:plotperfect".to_string()], None);
    }
    let before = store.stats().unwrap().total_memories;

    let report = consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
    assert!(report.duplicates_merged >= 1, "near-duplicates should merge");
    assert!(report.clusters_summarized >= 1, "topic cluster should summarize");

    let after = store.stats().unwrap();
    // One row merged away, one summary added.
    assert_eq!(after.total_memories, before - report.duplicates_merged as i64 + report.clusters_summarized as i64);
    assert!(after.links > 0, "merge + summary create links");

    // The new semantic summary should be retrievable.
    let provider = forgetfuldb_embed::create_provider(&cfg.embedding_backend, cfg.embedding_dim).unwrap();
    let pack = retrieve(&store, provider.as_ref(), &cfg, "plot perfect billing", &RetrieveOptions::default()).unwrap();
    assert!(pack
        .memories
        .iter()
        .any(|m| m.item.memory_type == MemoryType::Semantic && m.item.source.as_deref() == Some("consolidation")));
}

#[test]
fn duplicate_ingest_is_detected_across_store_reopen() {
    let db = TempDb::new("dedup-reopen");
    let cfg = Config { sqlite_path: db.0.display().to_string(), ..Config::default() };
    {
        let store = Store::open(&db.0).unwrap();
        ingest_text(&store, &cfg, "the wifi password is in the kitchen drawer", vec![], None);
    }
    // Reopen: Bloom filter is rebuilt from stored hashes.
    let store = Store::open(&db.0).unwrap();
    let mut bloom = warm_bloom(&store).unwrap();
    let provider = forgetfuldb_embed::create_provider(&cfg.embedding_backend, cfg.embedding_dim).unwrap();
    let outcome = ingest(
        &store,
        &mut bloom,
        provider.as_ref(),
        &cfg,
        IngestRequest {
            text: "The wifi password is in the kitchen drawer".to_string(),
            source: None,
            tags: vec![],
            memory_type: None,
            session_id: None,
            role: None,
        },
    )
    .unwrap();
    assert!(outcome.is_duplicate());
    assert_eq!(store.stats().unwrap().total_memories, 1);
}
