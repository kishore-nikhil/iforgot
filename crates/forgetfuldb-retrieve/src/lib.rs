//! forgetfuldb-retrieve
//!
//! Hybrid retrieval: blend vector similarity (placeholder embeddings) with
//! keyword/tag overlap, then weigh in decay-adjusted importance,
//! recurrence, recency, pinning and staleness using the formula in
//! `forgetfuldb_core::scoring`. The result is a compact "context pack"
//! ready to paste into an LLM prompt.
//!
//! v1 scans all candidate rows and computes cosine in process — a brute
//! force search that is comfortably fast for a personal memory store
//! (tens of thousands of rows). An ANN index can slot in behind the same
//! function signature later.

use anyhow::Result;
use forgetfuldb_core::config::Config;
use forgetfuldb_core::ingest::tokenize;
use forgetfuldb_core::scoring::{retrieval_score, ScoreBreakdown, ScoreInputs};
use forgetfuldb_core::types::{MemoryItem, MemoryType};
use forgetfuldb_core::{age_days, decay, now_unix};
use forgetfuldb_embed::{cosine_similarity, EmbeddingProvider};
use forgetfuldb_store::Store;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrieveOptions {
    pub top_k: usize,
    /// Stale memories are excluded unless explicitly requested.
    pub include_stale: bool,
    /// Archived memories are excluded unless explicitly requested.
    pub include_archived: bool,
}

impl Default for RetrieveOptions {
    fn default() -> Self {
        RetrieveOptions { top_k: 10, include_stale: false, include_archived: false }
    }
}

/// One retrieved memory plus the score breakdown explaining the ranking.
#[derive(Debug, Clone, Serialize)]
pub struct RetrievedMemory {
    #[serde(flatten)]
    pub item: MemoryItem,
    pub score: ScoreBreakdown,
}

/// Compact, JSON-serializable context for an LLM prompt.
#[derive(Debug, Clone, Serialize)]
pub struct ContextPack {
    pub query: String,
    pub generated_at: i64,
    pub memories: Vec<RetrievedMemory>,
}

/// Jaccard-style overlap between query tokens and memory tokens/tags/topic.
/// Complements the placeholder embedding with exact term matching.
pub fn keyword_overlap(query_tokens: &HashSet<String>, item: &MemoryItem) -> f64 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let mut item_tokens: HashSet<String> = tokenize(&item.content).into_iter().collect();
    for tag in &item.tags {
        // `project:plotperfect` matches both "project" and "plotperfect".
        for part in tag.split(':') {
            item_tokens.insert(part.to_lowercase());
        }
    }
    if let Some(topic) = &item.topic {
        item_tokens.insert(topic.to_lowercase());
    }
    for entity in &item.entities {
        item_tokens.insert(entity.to_lowercase());
    }
    let hits = query_tokens.iter().filter(|t| item_tokens.contains(*t)).count();
    hits as f64 / query_tokens.len() as f64
}

/// Run hybrid retrieval and return the top-k context pack. Retrieved
/// memories get their access metadata touched (retrieval is rehearsal).
pub fn retrieve(
    store: &Store,
    provider: &dyn EmbeddingProvider,
    cfg: &Config,
    query: &str,
    opts: &RetrieveOptions,
) -> Result<ContextPack> {
    let now = now_unix();
    let query_embedding = provider.embed(query);
    let query_tokens: HashSet<String> = tokenize(query).into_iter().collect();
    let lambdas = cfg.decay_lambdas();
    let weights = &cfg.retrieval_weights;

    let mut scored: Vec<RetrievedMemory> = Vec::new();
    for item in store.list_memories(None)? {
        if item.stale && !opts.include_stale {
            continue;
        }
        if item.memory_type == MemoryType::Archive && !opts.include_archived {
            continue;
        }

        let cosine = item
            .embedding
            .as_ref()
            .map(|e| cosine_similarity(&query_embedding, e) as f64)
            .unwrap_or(0.0)
            .max(0.0);
        let keywords = keyword_overlap(&query_tokens, &item);
        // Blend vector and lexical signals into one similarity in [0, 1].
        let semantic_similarity = (0.7 * cosine + 0.3 * keywords).clamp(0.0, 1.0);

        let lambda = lambdas.for_type(item.memory_type);
        let importance = decay::decay_score(
            item.importance_score,
            lambda,
            age_days(item.created_at, now),
            item.pinned,
        );
        let recency = decay::recency_score(age_days(
            item.last_accessed_at.unwrap_or(item.created_at),
            now,
        ));

        let breakdown = retrieval_score(
            &ScoreInputs {
                semantic_similarity,
                importance,
                recurrence: item.recurrence_score,
                recency,
                pinned: item.pinned,
                stale: item.stale,
            },
            weights,
        );
        scored.push(RetrievedMemory { item, score: breakdown });
    }

    scored.sort_by(|a, b| b.score.total.partial_cmp(&a.score.total).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(opts.top_k);

    // Retrieval counts as access: it slows future decay-driven cleanup.
    for hit in &scored {
        store.touch_memory(&hit.item.id, now)?;
    }

    Ok(ContextPack { query: query.to_string(), generated_at: now, memories: scored })
}

#[cfg(test)]
mod tests {
    use super::*;
    use forgetfuldb_store::pipeline::{ingest, warm_bloom, IngestRequest};

    fn setup() -> (Store, Box<dyn EmbeddingProvider>, Config) {
        let store = Store::open_in_memory().unwrap();
        let provider = forgetfuldb_embed::create_provider("hashed_bow", 128).unwrap();
        (store, provider, Config::default())
    }

    fn add(store: &Store, provider: &dyn EmbeddingProvider, cfg: &Config, text: &str, tags: Vec<String>) -> String {
        let mut bloom = warm_bloom(store).unwrap();
        let out = ingest(
            store,
            &mut bloom,
            provider,
            cfg,
            IngestRequest {
                text: text.to_string(),
                source: None,
                tags,
                memory_type: Some(MemoryType::Semantic),
                session_id: None,
                role: None,
            },
        )
        .unwrap();
        out.memory().id.clone()
    }

    #[test]
    fn relevant_memory_ranks_first() {
        let (store, provider, cfg) = setup();
        add(&store, provider.as_ref(), &cfg, "plot perfect billing runs on stripe invoices", vec![]);
        add(&store, provider.as_ref(), &cfg, "the cat sleeps on the windowsill every afternoon", vec![]);
        let pack = retrieve(&store, provider.as_ref(), &cfg, "plot perfect billing", &RetrieveOptions::default()).unwrap();
        assert!(pack.memories[0].item.content.contains("billing"));
        assert!(pack.memories[0].score.total > pack.memories[1].score.total);
    }

    #[test]
    fn stale_memories_hidden_unless_requested() {
        let (store, provider, cfg) = setup();
        let id = add(&store, provider.as_ref(), &cfg, "old billing fact about stripe", vec![]);
        store.set_stale(&id, true).unwrap();

        let hidden = retrieve(&store, provider.as_ref(), &cfg, "billing stripe", &RetrieveOptions::default()).unwrap();
        assert!(hidden.memories.is_empty());

        let opts = RetrieveOptions { include_stale: true, ..Default::default() };
        let shown = retrieve(&store, provider.as_ref(), &cfg, "billing stripe", &opts).unwrap();
        assert_eq!(shown.memories.len(), 1);
        assert_eq!(shown.memories[0].score.staleness_penalty, 1.0);
    }

    #[test]
    fn pinned_memory_outranks_equal_unpinned() {
        let (store, provider, cfg) = setup();
        // Two equally relevant memories; pin the second.
        add(&store, provider.as_ref(), &cfg, "billing detail alpha for stripe", vec![]);
        let pinned_id = add(&store, provider.as_ref(), &cfg, "billing detail bravo for stripe", vec![]);
        store.set_pinned(&pinned_id, true).unwrap();

        let pack = retrieve(&store, provider.as_ref(), &cfg, "billing stripe", &RetrieveOptions::default()).unwrap();
        assert_eq!(pack.memories[0].item.id, pinned_id);
        assert_eq!(pack.memories[0].score.pinned_boost, 1.0);
    }

    #[test]
    fn retrieval_touches_access_metadata() {
        let (store, provider, cfg) = setup();
        let id = add(&store, provider.as_ref(), &cfg, "remember the standup is at nine", vec![]);
        retrieve(&store, provider.as_ref(), &cfg, "standup time", &RetrieveOptions::default()).unwrap();
        let item = store.get_memory(&id).unwrap().unwrap();
        assert_eq!(item.access_count, 1);
        assert!(item.last_accessed_at.is_some());
    }
}
