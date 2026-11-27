use forgetfuldb_core::config::Config;
use forgetfuldb_core::ingest::content_hash;
use forgetfuldb_core::types::{InputMode, MemoryItem, MemoryType, SourceChunk, SourceDocument};
use forgetfuldb_embed::EmbeddingProvider;
use forgetfuldb_retrieve::{preview_at, retrieve, RetrieveOptions};
use forgetfuldb_store::{MemoryEdge, Store};
use serde_json::{json, Value};

const NOW: i64 = 1_700_000_000;

// Equal vectors deliberately exercise tied relevance and graph scores.
struct FixedEmbedding;

impl EmbeddingProvider for FixedEmbedding {
    fn name(&self) -> &'static str {
        "fixed-test"
    }
    fn dim(&self) -> usize {
        2
    }
    fn embed(&self, _: &str) -> Vec<f32> {
        vec![1.0, 0.0]
    }
}

fn memory(id: &str) -> MemoryItem {
    let mut item = MemoryItem::new(
        id.into(),
        format!("evidence {id}"),
        MemoryType::Semantic,
        content_hash(id),
        NOW,
    );
    item.embedding = Some(vec![1.0, 0.0]);
    item
}

fn snapshot(store: &Store) -> Value {
    json!({
        "memories": store.list_memories(None).unwrap(),
        "edges": store.list_edges().unwrap(),
        "links": store.all_links().unwrap(),
        "evidence": store.all_evidence().unwrap(),
        "chunks": store.all_source_chunks().unwrap(),
        "turns": store.list_chat_turns(100).unwrap(),
    })
}

fn graph_fixture(reverse: bool) -> Store {
    let store = Store::open_in_memory().unwrap();
    let mut ids = vec!["a", "b", "c", "d"];
    if reverse {
        ids.reverse();
    }
    for id in ids {
        store.insert_memory(&memory(id)).unwrap();
    }
    let mut edges = vec![("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")];
    if reverse {
        edges.reverse();
    }
    for (src, dst) in edges {
        store
            .upsert_edge(&MemoryEdge {
                src_id: src.into(),
                dst_id: dst.into(),
                edge_type: "sequence".into(),
                weight: 1.0,
                co_count: 1,
                created_at: NOW,
                last_activated: NOW,
            })
            .unwrap();
    }
    store
}

#[test]
fn previews_are_stable_across_runs_and_insertion_order_without_writes() {
    let mut cfg = Config {
        spreading_activation: true,
        ..Config::default()
    };
    cfg.spreading.inject_subgraph = true;
    cfg.spreading.seed_count = 1;
    cfg.spreading.max_hops = 2;
    let opts = RetrieveOptions {
        top_k: 1,
        ..Default::default()
    };
    let mut expected = None;
    for reverse in [false, true] {
        let store = graph_fixture(reverse);
        let before = snapshot(&store);
        // HashMap traversal paths receive fresh random hash seeds on each call.
        for _ in 0..20 {
            let pack = preview_at(&store, &FixedEmbedding, &cfg, "query", &opts, NOW).unwrap();
            assert!(pack.subgraph.is_some());
            assert!(pack.memories.len() > opts.top_k);
            let actual = serde_json::to_value(pack).unwrap();
            if let Some(expected) = &expected {
                assert_eq!(&actual, expected);
            } else {
                expected = Some(actual);
            }
        }
        assert_eq!(
            snapshot(&store),
            before,
            "preview must not rehearse or reinforce evidence"
        );
    }
}

#[test]
fn supplied_clock_controls_decay_and_recency_without_changing_access_state() {
    let store = Store::open_in_memory().unwrap();
    store.insert_memory(&memory("clock")).unwrap();
    let cfg = Config::default();
    let opts = RetrieveOptions::default();
    let first = preview_at(&store, &FixedEmbedding, &cfg, "query", &opts, NOW).unwrap();
    let later = preview_at(
        &store,
        &FixedEmbedding,
        &cfg,
        "query",
        &opts,
        NOW + 100 * 86_400,
    )
    .unwrap();
    assert_eq!(first.generated_at, NOW);
    assert_eq!(later.generated_at, NOW + 100 * 86_400);
    assert!(later.memories[0].score.importance < first.memories[0].score.importance);
    assert!(later.memories[0].score.recency < first.memories[0].score.recency);
    let stored = store.get_memory("clock").unwrap().unwrap();
    assert_eq!(stored.access_count, 0);
    assert_eq!(stored.last_accessed_at, None);
}

#[test]
fn live_retrieval_still_records_access() {
    let store = Store::open_in_memory().unwrap();
    store.insert_memory(&memory("live")).unwrap();
    retrieve(
        &store,
        &FixedEmbedding,
        &Config::default(),
        "query",
        &RetrieveOptions::default(),
    )
    .unwrap();
    let stored = store.get_memory("live").unwrap().unwrap();
    assert_eq!(stored.access_count, 1);
    assert!(stored.last_accessed_at.is_some());
    assert!(
        store.all_evidence().unwrap().is_empty(),
        "access must not become success evidence"
    );
}

#[test]
fn preview_uses_supplied_clock_for_source_chunk_expiry() {
    let store = Store::open_in_memory().unwrap();
    let mut anchor = memory("document");
    anchor.tags.push("source_doc:source".into());
    anchor.embedding = Some(vec![0.0, 1.0]);
    store.insert_memory(&anchor).unwrap();
    store
        .insert_source_document(&SourceDocument {
            id: "source".into(),
            raw_text_hash: "source-hash".into(),
            source_type: InputMode::PastedDocument,
            session_id: None,
            summary: None,
            entities: vec![],
            topics: vec![],
            created_at: NOW,
        })
        .unwrap();
    store
        .insert_source_chunk(&SourceChunk {
            id: "chunk".into(),
            source_id: "source".into(),
            chunk_index: 0,
            text: "the applicable verification condition".into(),
            summary: None,
            entities: vec![],
            topics: vec![],
            embedding: Some(vec![1.0, 0.0]),
            salience: 0.0,
            created_at: NOW,
        })
        .unwrap();
    let before = snapshot(&store);
    let mut cfg = Config {
        decay_lambda_raw: 1.0,
        chunk_match_min_sim: 0.5,
        ..Config::default()
    };
    cfg.consolidation_thresholds.archive_max_decay = 0.3;
    let opts = RetrieveOptions::default();
    let fresh = preview_at(&store, &FixedEmbedding, &cfg, "query", &opts, NOW).unwrap();
    assert_eq!(
        fresh.memories[0].item.content,
        "the applicable verification condition"
    );
    let aged = preview_at(
        &store,
        &FixedEmbedding,
        &cfg,
        "query",
        &opts,
        NOW + 2 * 86_400,
    )
    .unwrap();
    assert_eq!(aged.memories[0].item.content, anchor.content);
    assert_eq!(snapshot(&store), before);
}
