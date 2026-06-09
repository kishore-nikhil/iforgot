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

use crate::Store;
use anyhow::Result;
use forgetfuldb_core::config::Config;
use forgetfuldb_core::ids::new_id;
use forgetfuldb_core::ingest::{content_hash, extract_keywords, guess_topic, initial_importance, normalize};
use forgetfuldb_core::types::{MemoryItem, MemoryType, RawEvent, Session};
use forgetfuldb_core::{decay, now_unix};
use forgetfuldb_embed::EmbeddingProvider;
use forgetfuldb_prob::BloomFilter;

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
