//! The ingest workflow, shared by the CLI and HTTP server:
//!
//! 1. normalize text and hash content
//! 2. Bloom-filter pre-check ("probably seen before?") — dedup only,
//!    never used for retrieval
//! 3. store the raw event (if new)
//! 4. extract keywords/entities with local heuristics
//! 5. embed via the pluggable provider
//! 6. compute initial importance and decay
//! 7. persist the memory item
//!
//! On an exact duplicate, the existing memory's recurrence is reinforced
//! instead of inserting a new row — "I keep hearing this" makes a memory
//! stronger, like rehearsal in human memory.

use crate::{MemoryEdge, Store};
use anyhow::Result;
use forgetfuldb_core::config::Config;
use forgetfuldb_core::ids::new_id;
use forgetfuldb_core::ingest::{content_hash, extract_keywords, guess_topic, initial_importance, normalize};
use forgetfuldb_core::types::{MemoryItem, MemoryType, RawEvent, Session};
use forgetfuldb_core::{age_days, decay, now_unix};
use forgetfuldb_embed::EmbeddingProvider;
use forgetfuldb_prob::BloomFilter;
use std::collections::{HashMap, HashSet};

/// Ingest parameters from the CLI/API.
#[derive(Debug, Clone)]
pub struct IngestRequest {
    pub text: String,
    pub source: Option<String>,
    pub tags: Vec<String>,
    pub memory_type: Option<MemoryType>,
    pub session_id: Option<String>,
    pub role: Option<String>,
}

/// What happened during ingest.
#[derive(Debug, Clone)]
pub enum IngestOutcome {
    /// A new memory was stored.
    Stored(MemoryItem),
    /// Exact duplicate: the existing memory was reinforced instead.
    Reinforced(MemoryItem),
}

impl IngestOutcome {
    pub fn memory(&self) -> &MemoryItem {
        match self {
            IngestOutcome::Stored(m) | IngestOutcome::Reinforced(m) => m,
        }
    }

    pub fn is_duplicate(&self) -> bool {
        matches!(self, IngestOutcome::Reinforced(_))
    }
}

/// Meta keys recording which embedding model the stored vectors use.
pub const META_EMBED_BACKEND: &str = "embedding_backend";
pub const META_EMBED_MODEL: &str = "embedding_model";
pub const META_EMBED_DIM: &str = "embedding_dim";

/// Re-embed every memory with `provider` and record the new embedding
/// identity. Needed whenever the embedding model changes, because vectors
/// of different dimensions (or different models) are not comparable —
/// without this, retrieval silently returns zero similarity. `on_progress`
/// is called as `(done, total)` so a UI can show a bar. Returns the count.
pub fn reembed_all(
    store: &Store,
    provider: &dyn EmbeddingProvider,
    model: &str,
    mut on_progress: impl FnMut(usize, usize),
) -> Result<usize> {
    let items = store.list_memories(None)?;
    let total = items.len();
    for (i, item) in items.iter().enumerate() {
        store.set_embedding(&item.id, &provider.embed(&item.content))?;
        on_progress(i + 1, total);
    }
    store.set_meta(META_EMBED_BACKEND, provider.name())?;
    store.set_meta(META_EMBED_MODEL, model)?;
    store.set_meta(META_EMBED_DIM, &provider.dim().to_string())?;
    Ok(total)
}

/// Compare the active provider against what the stored vectors actually
/// are. Returns a human-readable warning when they differ (e.g. the model
/// changed but `reembed_all` hasn't run), else `None`. Based on a real
/// stored vector's length, so it fires even if the embedding identity was
/// never recorded (e.g. config edited by hand).
pub fn embedding_mismatch_warning(store: &Store, provider: &dyn EmbeddingProvider) -> Option<String> {
    let stored_dim = store
        .list_memories(None)
        .ok()?
        .iter()
        .find_map(|m| m.embedding.as_ref().map(|e| e.len()))?;
    if stored_dim == provider.dim() {
        return None;
    }
    let stored_model = store.get_meta(META_EMBED_MODEL).ok().flatten().unwrap_or_default();
    let from = if stored_model.is_empty() { String::new() } else { format!(" (model \"{stored_model}\")") };
    Some(format!(
        "stored embeddings are {stored_dim}-dim{from} but the active provider is {}-dim — retrieval will \
         mismatch until you re-embed (run `forgetfuldb reembed`, or `/embed` in chat)",
        provider.dim()
    ))
}

/// The association edge type for "retrieved into the same chat turn".
pub const EDGE_CO_OCCURRED: &str = "co_occurred";

/// Rebuild the `co_occurred` association edges from `chat_turns`: two
/// memories injected into the same turn are associated, and the more
/// recently (and more often) that happened, the stronger the edge.
///
/// Recomputed from scratch each call (idempotent — no unbounded growth):
/// weight = Σ over shared turns of `exp(-lambda * age_days(turn))`. Pairs
/// below `min_weight`, or referencing memories that no longer exist, are
/// dropped. Returns the number of edges written.
pub fn rebuild_cooccurrence_edges(store: &Store, lambda: f64, min_weight: f64, now: i64) -> Result<usize> {
    let alive: HashSet<String> = store.list_memories(None)?.into_iter().map(|m| m.id).collect();

    // (a, b) with a < b  ->  (summed weight, shared-turn count, latest ts)
    let mut pairs: HashMap<(String, String), (f64, i64, i64)> = HashMap::new();
    for turn in store.list_chat_turns(usize::MAX)? {
        let ids: Vec<&String> = turn.memory_ids.iter().filter(|id| alive.contains(*id)).collect();
        if ids.len() < 2 {
            continue;
        }
        let w = (-lambda * age_days(turn.created_at, now).max(0.0)).exp();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let (a, b) = if ids[i] <= ids[j] { (ids[i], ids[j]) } else { (ids[j], ids[i]) };
                let entry = pairs.entry((a.clone(), b.clone())).or_insert((0.0, 0, 0));
                entry.0 += w;
                entry.1 += 1;
                entry.2 = entry.2.max(turn.created_at);
            }
        }
    }

    store.clear_edges(EDGE_CO_OCCURRED)?;
    let mut written = 0;
    for ((src_id, dst_id), (weight, co_count, last_activated)) in pairs {
        if weight < min_weight {
            continue;
        }
        store.upsert_edge(&MemoryEdge {
            src_id,
            dst_id,
            edge_type: EDGE_CO_OCCURRED.to_string(),
            weight,
            co_count,
            created_at: now,
            last_activated,
        })?;
        written += 1;
    }
    Ok(written)
}

/// Build a Bloom filter warmed with every content hash already stored.
/// The filter is a fast pre-check only; the UNIQUE constraint on
/// `content_hash` is the authoritative dedup mechanism.
pub fn warm_bloom(store: &Store) -> Result<BloomFilter> {
    let hashes = store.all_content_hashes()?;
    let mut bloom = BloomFilter::with_capacity(hashes.len().max(10_000), 0.01);
    for h in &hashes {
        bloom.insert(h);
    }
    Ok(bloom)
}

pub fn ingest(
    store: &Store,
    bloom: &mut BloomFilter,
    provider: &dyn EmbeddingProvider,
    cfg: &Config,
    req: IngestRequest,
) -> Result<IngestOutcome> {
    let now = now_unix();
    let text = normalize(&req.text);
    anyhow::ensure!(!text.is_empty(), "cannot ingest empty text");
    let hash = content_hash(&text);

    // Bloom says "maybe seen" -> confirm against SQLite (false positives
    // are possible). Bloom says "never seen" -> skip the lookup entirely.
    if bloom.contains(&hash) {
        if let Some(existing) = store.get_memory_by_hash(&hash)? {
            return Ok(IngestOutcome::Reinforced(reinforce(store, existing, now)?));
        }
    }

    if let Some(session_id) = &req.session_id {
        store.upsert_session(&Session {
            id: session_id.clone(),
            title: None,
            created_at: now,
            updated_at: now,
        })?;
    }

    // Keep the verbatim input as a raw event for the record.
    store.insert_raw_event(&RawEvent {
        id: new_id("evt", &hash),
        session_id: req.session_id.clone(),
        role: req.role.clone(),
        content: text.clone(),
        created_at: now,
        content_hash: hash.clone(),
    })?;

    let memory_type = req.memory_type.unwrap_or(MemoryType::Episodic);
    let entities = extract_keywords(&text, 8);
    let topic = guess_topic(&req.tags, &entities);
    let importance = initial_importance(&text, memory_type, &req.tags);
    let lambda = cfg.decay_lambdas().for_type(memory_type);

    let mut item = MemoryItem::new(new_id("mem", &hash), text, memory_type, hash.clone(), now);
    item.source = req.source;
    item.topic = topic;
    item.entities = entities;
    item.tags = req.tags;
    item.importance_score = importance;
    item.decay_score = decay::decay_score(importance, lambda, 0.0, false);
    item.recency_score = 1.0;
    item.embedding = Some(provider.embed(&item.content));

    store.insert_memory(&item)?;
    bloom.insert(&hash);
    Ok(IngestOutcome::Stored(item))
}

/// Duplicate input strengthens the existing memory: recurrence climbs
/// (saturating at 1.0), access metadata is refreshed.
fn reinforce(store: &Store, mut item: MemoryItem, now: i64) -> Result<MemoryItem> {
    item.recurrence_score = (item.recurrence_score + 0.2).min(1.0);
    item.access_count += 1;
    item.last_accessed_at = Some(now);
    item.updated_at = now;
    store.update_memory(&item)?;
    Ok(item)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Store, BloomFilter, Box<dyn EmbeddingProvider>, Config) {
        let store = Store::open_in_memory().unwrap();
        let bloom = warm_bloom(&store).unwrap();
        let provider = forgetfuldb_embed::create_provider("hashed_bow", 64).unwrap();
        (store, bloom, provider, Config::default())
    }

    fn req(text: &str) -> IngestRequest {
        IngestRequest {
            text: text.to_string(),
            source: Some("test".into()),
            tags: vec![],
            memory_type: None,
            session_id: None,
            role: None,
        }
    }

    #[test]
    fn ingest_stores_new_memory() {
        let (store, mut bloom, provider, cfg) = setup();
        let out = ingest(&store, &mut bloom, provider.as_ref(), &cfg, req("plot perfect uses stripe")).unwrap();
        assert!(!out.is_duplicate());
        assert!(store.get_memory(&out.memory().id).unwrap().is_some());
        assert_eq!(store.stats().unwrap().raw_events, 1);
    }

    #[test]
    fn cooccurrence_edges_from_shared_turns() {
        let (store, mut bloom, provider, cfg) = setup();
        let a = ingest(&store, &mut bloom, provider.as_ref(), &cfg, req("alpha fact about billing")).unwrap().memory().id.clone();
        let b = ingest(&store, &mut bloom, provider.as_ref(), &cfg, req("bravo fact about demos")).unwrap().memory().id.clone();
        let c = ingest(&store, &mut bloom, provider.as_ref(), &cfg, req("charlie unrelated fact")).unwrap().memory().id.clone();

        // Two turns inject {a,b}; one turn injects {a,c}. So a-b is stronger.
        let turn = |id: &str, ids: Vec<String>| crate::ChatTurn {
            id: id.to_string(), session_id: None, created_at: now_unix(),
            user_text: "q".into(), assistant_text: "r".into(), model: "m".into(), backend: "b".into(),
            prompt_tokens: None, completion_tokens: None, total_duration_ms: None, llm_duration_ms: None,
            retrieve_duration_ms: 0, context_memory_count: ids.len() as i64, context_chars: 0, memory_ids: ids,
        };
        store.insert_chat_turn(&turn("t1", vec![a.clone(), b.clone()])).unwrap();
        store.insert_chat_turn(&turn("t2", vec![a.clone(), b.clone()])).unwrap();
        store.insert_chat_turn(&turn("t3", vec![a.clone(), c.clone()])).unwrap();

        let n = rebuild_cooccurrence_edges(&store, 0.02, 0.05, now_unix()).unwrap();
        assert_eq!(n, 2, "pairs a-b and a-c");
        let neighbors = store.neighbors(&a, EDGE_CO_OCCURRED).unwrap();
        let ab = neighbors.iter().find(|(id, _)| *id == b).unwrap().1;
        let ac = neighbors.iter().find(|(id, _)| *id == c).unwrap().1;
        assert!(ab > ac, "a-b co-occurred twice, a-c once: {ab} vs {ac}");

        // Idempotent: a second rebuild yields the same edge count.
        assert_eq!(rebuild_cooccurrence_edges(&store, 0.02, 0.05, now_unix()).unwrap(), 2);
    }

    #[test]
    fn duplicate_ingest_reinforces_instead_of_duplicating() {
        let (store, mut bloom, provider, cfg) = setup();
        ingest(&store, &mut bloom, provider.as_ref(), &cfg, req("the api key lives in 1password")).unwrap();
        // Same canonical content (case/whitespace differ).
        let out = ingest(&store, &mut bloom, provider.as_ref(), &cfg, req("The API  key lives in 1Password")).unwrap();
        assert!(out.is_duplicate());
        assert!(out.memory().recurrence_score > 0.0);
        assert_eq!(store.stats().unwrap().total_memories, 1);
    }
}
