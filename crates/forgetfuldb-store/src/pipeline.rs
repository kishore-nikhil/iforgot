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
use forgetfuldb_core::ingest::{
    chunk_source_text, classify_input_mode, content_hash, extract_keywords,
    extract_memory_candidates, guess_topic, initial_importance, normalize,
};
use forgetfuldb_core::types::{
    EvidenceSource, EvidenceType, InputMode, MemoryCandidateType, MemoryEvidence, MemoryItem,
    MemoryType, RawEvent, Session, SourceChunk, SourceDocument,
};
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
pub fn embedding_mismatch_warning(
    store: &Store,
    provider: &dyn EmbeddingProvider,
) -> Option<String> {
    let stored_dim = store
        .list_memories(None)
        .ok()?
        .iter()
        .find_map(|m| m.embedding.as_ref().map(|e| e.len()))?;
    if stored_dim == provider.dim() {
        return None;
    }
    let stored_model = store
        .get_meta(META_EMBED_MODEL)
        .ok()
        .flatten()
        .unwrap_or_default();
    let from = if stored_model.is_empty() {
        String::new()
    } else {
        format!(" (model \"{stored_model}\")")
    };
    Some(format!(
        "stored embeddings are {stored_dim}-dim{from} but the active provider is {}-dim — retrieval will \
         mismatch until you re-embed (run `forgetfuldb reembed`, or `/embed` in chat)",
        provider.dim()
    ))
}

/// The association edge type for "retrieved into the same chat turn".
pub const EDGE_CO_OCCURRED: &str = "co_occurred";
/// Edge type for "close in embedding space" (cosine kNN). Unlike
/// co-occurrence (behavioral — recalled together), this is *semantic*: two
/// memories that mean similar things, even if never recalled together.
pub const EDGE_SEMANTIC: &str = "semantic_similar";
/// Edge type for "discussing A was followed by discussing B" — directional,
/// reconstructed from the order of chat turns within a session. The causal
/// signal that session conversational order carries (and that nothing else
/// captures).
pub const EDGE_SEQUENCE: &str = "sequence";

/// Rebuild `semantic_similar` edges: connect each memory to its nearest
/// neighbors in embedding space (cosine >= `min_sim`, up to `top_k`).
/// Recomputed from scratch (idempotent). Undirected (canonical src < dst);
/// weight is the cosine. Answers "what is *close in meaning*", distinct
/// from co-occurrence's "what is *recalled together*".
pub fn rebuild_semantic_edges(
    store: &Store,
    min_sim: f64,
    top_k: usize,
    now: i64,
) -> Result<usize> {
    let items: Vec<MemoryItem> = store
        .list_memories(None)?
        .into_iter()
        .filter(|m| m.embedding.is_some())
        .collect();

    // Canonical pair -> best cosine seen.
    let mut pairs: HashMap<(String, String), f64> = HashMap::new();
    for item in &items {
        let emb = item.embedding.as_ref().unwrap();
        let mut sims: Vec<(&str, f64)> = items
            .iter()
            .filter(|o| o.id != item.id)
            .filter_map(|o| {
                o.embedding.as_ref().map(|oe| {
                    (
                        o.id.as_str(),
                        forgetfuldb_embed::cosine_similarity(emb, oe) as f64,
                    )
                })
            })
            .filter(|(_, c)| *c >= min_sim)
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (other, cos) in sims.into_iter().take(top_k) {
            let (a, b) = if item.id.as_str() <= other {
                (item.id.as_str(), other)
            } else {
                (other, item.id.as_str())
            };
            let e = pairs.entry((a.to_string(), b.to_string())).or_insert(0.0);
            *e = e.max(cos);
        }
    }

    store.clear_edges(EDGE_SEMANTIC)?;
    let mut written = 0;
    for ((src_id, dst_id), weight) in pairs {
        store.upsert_edge(&MemoryEdge {
            src_id,
            dst_id,
            edge_type: EDGE_SEMANTIC.to_string(),
            weight,
            co_count: 1,
            created_at: now,
            last_activated: now,
        })?;
        written += 1;
    }
    Ok(written)
}

/// Rebuild `sequence` edges from session conversational order: for each
/// pair of consecutive turns in a session, link the top memories of the
/// earlier turn to the top memories of the later one (directional,
/// src=earlier). Weight is recency-decayed like co-occurrence. This is the
/// reasoning-path signal — "we discussed A, then B" — recovered from
/// `chat_turns`, which nothing else reads for structure.
pub fn rebuild_sequence_edges(
    store: &Store,
    lambda: f64,
    min_weight: f64,
    now: i64,
    top_per_turn: usize,
) -> Result<usize> {
    let alive: HashSet<String> = store
        .list_memories(None)?
        .into_iter()
        .map(|m| m.id)
        .collect();
    let turns = store.list_chat_turns(usize::MAX)?; // oldest first

    // Directional (src=earlier -> dst=later) pair -> (weight, count, latest).
    let mut pairs: HashMap<(String, String), (f64, i64, i64)> = HashMap::new();
    for win in turns.windows(2) {
        let (a, b) = (&win[0], &win[1]);
        if a.session_id.is_none() || a.session_id != b.session_id {
            continue; // only within one session
        }
        let w = (-lambda * age_days(b.created_at, now).max(0.0)).exp();
        let earlier: Vec<&String> = a
            .memory_ids
            .iter()
            .filter(|id| alive.contains(*id))
            .take(top_per_turn)
            .collect();
        let later: Vec<&String> = b
            .memory_ids
            .iter()
            .filter(|id| alive.contains(*id))
            .take(top_per_turn)
            .collect();
        for src in &earlier {
            for dst in &later {
                if src == dst {
                    continue;
                }
                let entry = pairs
                    .entry(((*src).clone(), (*dst).clone()))
                    .or_insert((0.0, 0, 0));
                entry.0 += w;
                entry.1 += 1;
                entry.2 = entry.2.max(b.created_at);
            }
        }
    }

    store.clear_edges(EDGE_SEQUENCE)?;
    let mut written = 0;
    for ((src_id, dst_id), (weight, co_count, last_activated)) in pairs {
        if weight < min_weight {
            continue;
        }
        store.upsert_edge(&MemoryEdge {
            src_id,
            dst_id,
            edge_type: EDGE_SEQUENCE.to_string(),
            weight,
            co_count,
            created_at: now,
            last_activated,
        })?;
        written += 1;
    }
    Ok(written)
}

/// Rebuild the `co_occurred` association edges from `chat_turns`: two
/// memories injected into the same turn are associated, and the more
/// recently (and more often) that happened, the stronger the edge.
///
/// Recomputed from scratch each call (idempotent — no unbounded growth):
/// weight = Σ over shared turns of `exp(-lambda * age_days(turn))`. Pairs
/// below `min_weight`, or referencing memories that no longer exist, are
/// dropped. Returns the number of edges written.
pub fn rebuild_cooccurrence_edges(
    store: &Store,
    lambda: f64,
    min_weight: f64,
    now: i64,
) -> Result<usize> {
    let alive: HashSet<String> = store
        .list_memories(None)?
        .into_iter()
        .map(|m| m.id)
        .collect();

    // (a, b) with a < b  ->  (summed weight, shared-turn count, latest ts)
    let mut pairs: HashMap<(String, String), (f64, i64, i64)> = HashMap::new();
    for turn in store.list_chat_turns(usize::MAX)? {
        let ids: Vec<&String> = turn
            .memory_ids
            .iter()
            .filter(|id| alive.contains(*id))
            .collect();
        if ids.len() < 2 {
            continue;
        }
        let w = (-lambda * age_days(turn.created_at, now).max(0.0)).exp();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let (a, b) = if ids[i] <= ids[j] {
                    (ids[i], ids[j])
                } else {
                    (ids[j], ids[i])
                };
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
    let input_mode = classify_input_mode(&req.text);
    let parse = extract_memory_candidates(&req.text, None);

    // Bloom says "maybe seen" -> confirm against SQLite (false positives
    // are possible). Bloom says "never seen" -> skip the lookup entirely.
    if bloom.contains(&hash) {
        if let Some(existing) = store.get_memory_by_hash(&hash)? {
            return Ok(IngestOutcome::Reinforced(reinforce(
                store,
                existing,
                req.session_id.clone(),
                now,
            )?));
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

    let source_document_id = if input_mode.is_long_source() {
        Some(store_source_document(
            store,
            &text,
            &hash,
            input_mode,
            req.session_id.clone(),
            now,
        )?)
    } else {
        None
    };

    let memory_type = req
        .memory_type
        .unwrap_or_else(|| inferred_memory_type(input_mode, &parse));
    let entities = extract_keywords(&text, 8);
    let topic = guess_topic(&req.tags, &entities);
    let mut importance = initial_importance(&text, memory_type, &req.tags);
    importance = v2_base_importance(memory_type, &parse, importance, input_mode);
    let lambda = cfg.decay_lambdas().for_type(memory_type);

    let mut item = MemoryItem::new(new_id("mem", &hash), text, memory_type, hash.clone(), now);
    item.source = req.source;
    item.topic = topic;
    item.entities = entities;
    item.tags = req.tags;
    if let Some(source_id) = &source_document_id {
        item.tags.push(format!("source_doc:{source_id}"));
        item.source
            .get_or_insert_with(|| input_mode.as_str().to_string());
    }
    item.importance_score = importance;
    item.confidence = parse.confidence.max(if input_mode.is_long_source() {
        0.35
    } else {
        0.5
    });
    item.decay_score = decay::decay_score(importance, lambda, 0.0, false);
    item.recency_score = 1.0;
    let embedding = provider.embed(&item.content);
    // Provisional salience from novelty alone (the free write-time signal):
    // 1 - max cosine to anything stored, gated by content quality so a
    // novel typo doesn't enshrine itself. Consolidation later revises this
    // with the full surprise/habit discriminator.
    let relevance = forgetfuldb_core::salience::content_relevance(
        item.content.chars().count(),
        item.entities.len(),
    );
    item.salience = provisional_salience(store, &embedding, relevance)?;
    item.embedding = Some(embedding);

    store.insert_memory(&item)?;
    record_ingest_evidence(store, &item, &parse, req.session_id.clone(), now)?;
    bloom.insert(&hash);
    Ok(IngestOutcome::Stored(item))
}

fn inferred_memory_type(
    input_mode: InputMode,
    parse: &forgetfuldb_core::types::ParseResult,
) -> MemoryType {
    if input_mode.is_long_source() {
        return MemoryType::RawEvent;
    }
    parse
        .candidates
        .iter()
        .find_map(|c| match c.candidate_type {
            MemoryCandidateType::PreferenceStrong | MemoryCandidateType::PreferenceWeak => {
                Some(MemoryType::Preference)
            }
            MemoryCandidateType::GoalCandidate
            | MemoryCandidateType::IdentityCandidate
            | MemoryCandidateType::ProjectFactCandidate => Some(MemoryType::Semantic),
            _ => None,
        })
        .unwrap_or(MemoryType::Episodic)
}

fn v2_base_importance(
    memory_type: MemoryType,
    parse: &forgetfuldb_core::types::ParseResult,
    fallback: f64,
    input_mode: InputMode,
) -> f64 {
    if input_mode.is_long_source() {
        return 0.10;
    }
    let candidate_base = parse
        .candidates
        .iter()
        .map(|c| match c.candidate_type {
            MemoryCandidateType::IdentityCandidate => 0.70,
            MemoryCandidateType::GoalCandidate => 0.60,
            MemoryCandidateType::PreferenceStrong => 0.50,
            MemoryCandidateType::PreferenceWeak => 0.30,
            MemoryCandidateType::EpisodicCandidate => 0.20,
            MemoryCandidateType::CorrectionSignal => 0.45,
            _ => 0.10,
        })
        .fold(0.0_f64, f64::max);
    let type_floor = match memory_type {
        MemoryType::Foundation => 0.70,
        MemoryType::Preference => 0.30,
        MemoryType::Semantic => 0.20,
        MemoryType::Procedural => 0.30,
        MemoryType::Episodic => 0.20,
        MemoryType::RawEvent => 0.10,
        MemoryType::Archive => 0.05,
    };
    fallback
        .min(0.65)
        .max(candidate_base)
        .max(type_floor)
        .clamp(0.05, 1.0)
}

fn store_source_document(
    store: &Store,
    text: &str,
    hash: &str,
    input_mode: InputMode,
    session_id: Option<String>,
    now: i64,
) -> Result<String> {
    let id = new_id("src", hash);
    let entities = extract_keywords(text, 16);
    let topics = entities.iter().take(5).cloned().collect::<Vec<_>>();
    let summary = Some(
        text.split_whitespace()
            .take(40)
            .collect::<Vec<_>>()
            .join(" "),
    );
    store.insert_source_document(&SourceDocument {
        id: id.clone(),
        raw_text_hash: hash.to_string(),
        source_type: input_mode,
        session_id,
        summary,
        entities,
        topics,
        created_at: now,
    })?;
    for (i, chunk_text) in chunk_source_text(text, 550, 80).into_iter().enumerate() {
        let chunk_hash = content_hash(&format!("{hash}:{i}:{chunk_text}"));
        let chunk_entities = extract_keywords(&chunk_text, 8);
        let chunk_topics = chunk_entities.iter().take(3).cloned().collect::<Vec<_>>();
        store.insert_source_chunk(&SourceChunk {
            id: new_id("chunk", &chunk_hash),
            source_id: id.clone(),
            chunk_index: i,
            text: chunk_text,
            summary: None,
            entities: chunk_entities,
            topics: chunk_topics,
        })?;
    }
    Ok(id)
}

fn record_ingest_evidence(
    store: &Store,
    item: &MemoryItem,
    parse: &forgetfuldb_core::types::ParseResult,
    session_id: Option<String>,
    now: i64,
) -> Result<()> {
    let mut evidence = Vec::new();
    if item.content.to_lowercase().contains("remember") {
        evidence.push((EvidenceType::ExplicitRememberRequest, 0.8));
    }
    if parse
        .candidates
        .iter()
        .any(|c| c.candidate_type == MemoryCandidateType::CorrectionSignal)
    {
        evidence.push((EvidenceType::UserCorrection, 0.9));
    }
    if parse.confidence >= 0.45 {
        evidence.push((EvidenceType::TopicRepeated, parse.confidence.min(0.6)));
    }
    for (idx, (evidence_type, strength)) in evidence.into_iter().enumerate() {
        store.insert_evidence(&MemoryEvidence {
            id: new_id(
                "ev",
                &format!("{}-{idx}-{}", item.id, evidence_type.as_str()),
            ),
            memory_id: item.id.clone(),
            evidence_type,
            strength,
            source: EvidenceSource::DeterministicExtractor,
            session_id: session_id.clone(),
            created_at: now,
        })?;
    }
    Ok(())
}

/// Write-time provisional salience from novelty: `1 - max cosine to any
/// stored memory`, gated by relevance so novel-noise stays low. O(n) scan
/// — runs on the ingest path (the background writer thread for chat), fine
/// at personal scale; the authoritative value comes from consolidation.
fn provisional_salience(store: &Store, embedding: &[f32], relevance: f64) -> Result<f64> {
    let mut max_cos = 0.0_f32;
    for other in store.list_memories(None)? {
        if let Some(e) = &other.embedding {
            max_cos = max_cos.max(forgetfuldb_embed::cosine_similarity(embedding, e));
        }
    }
    let surprise = (1.0 - max_cos as f64).clamp(0.0, 1.0);
    Ok((surprise * relevance.clamp(0.0, 1.0)).clamp(0.0, 1.0))
}

/// Duplicate input strengthens the existing memory: recurrence climbs
/// (saturating at 1.0), access metadata is refreshed.
fn reinforce(
    store: &Store,
    mut item: MemoryItem,
    session_id: Option<String>,
    now: i64,
) -> Result<MemoryItem> {
    item.recurrence_score = (item.recurrence_score + 0.2).min(1.0);
    item.access_count += 1;
    item.last_accessed_at = Some(now);
    item.updated_at = now;
    store.update_memory(&item)?;
    store.insert_evidence(&MemoryEvidence {
        id: new_id("ev", &format!("{}-duplicate-{now}", item.id)),
        memory_id: item.id.clone(),
        evidence_type: EvidenceType::TopicRepeated,
        strength: 0.5,
        source: EvidenceSource::DeterministicExtractor,
        session_id,
        created_at: now,
    })?;
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
        let out = ingest(
            &store,
            &mut bloom,
            provider.as_ref(),
            &cfg,
            req("plot perfect uses stripe"),
        )
        .unwrap();
        assert!(!out.is_duplicate());
        assert!(store.get_memory(&out.memory().id).unwrap().is_some());
        assert_eq!(store.stats().unwrap().raw_events, 1);
    }

    #[test]
    fn cooccurrence_edges_from_shared_turns() {
        let (store, mut bloom, provider, cfg) = setup();
        let a = ingest(
            &store,
            &mut bloom,
            provider.as_ref(),
            &cfg,
            req("alpha fact about billing"),
        )
        .unwrap()
        .memory()
        .id
        .clone();
        let b = ingest(
            &store,
            &mut bloom,
            provider.as_ref(),
            &cfg,
            req("bravo fact about demos"),
        )
        .unwrap()
        .memory()
        .id
        .clone();
        let c = ingest(
            &store,
            &mut bloom,
            provider.as_ref(),
            &cfg,
            req("charlie unrelated fact"),
        )
        .unwrap()
        .memory()
        .id
        .clone();

        // Two turns inject {a,b}; one turn injects {a,c}. So a-b is stronger.
        let turn = |id: &str, ids: Vec<String>| crate::ChatTurn {
            id: id.to_string(),
            session_id: None,
            created_at: now_unix(),
            user_text: "q".into(),
            assistant_text: "r".into(),
            model: "m".into(),
            backend: "b".into(),
            prompt_tokens: None,
            completion_tokens: None,
            total_duration_ms: None,
            llm_duration_ms: None,
            retrieve_duration_ms: 0,
            context_memory_count: ids.len() as i64,
            context_chars: 0,
            memory_ids: ids,
        };
        store
            .insert_chat_turn(&turn("t1", vec![a.clone(), b.clone()]))
            .unwrap();
        store
            .insert_chat_turn(&turn("t2", vec![a.clone(), b.clone()]))
            .unwrap();
        store
            .insert_chat_turn(&turn("t3", vec![a.clone(), c.clone()]))
            .unwrap();

        let n = rebuild_cooccurrence_edges(&store, 0.02, 0.05, now_unix()).unwrap();
        assert_eq!(n, 2, "pairs a-b and a-c");
        let neighbors = store.neighbors(&a, EDGE_CO_OCCURRED).unwrap();
        let ab = neighbors.iter().find(|(id, _)| *id == b).unwrap().1;
        let ac = neighbors.iter().find(|(id, _)| *id == c).unwrap().1;
        assert!(ab > ac, "a-b co-occurred twice, a-c once: {ab} vs {ac}");

        // Idempotent: a second rebuild yields the same edge count.
        assert_eq!(
            rebuild_cooccurrence_edges(&store, 0.02, 0.05, now_unix()).unwrap(),
            2
        );
    }

    #[test]
    fn duplicate_ingest_reinforces_instead_of_duplicating() {
        let (store, mut bloom, provider, cfg) = setup();
        ingest(
            &store,
            &mut bloom,
            provider.as_ref(),
            &cfg,
            req("the api key lives in 1password"),
        )
        .unwrap();
        // Same canonical content (case/whitespace differ).
        let out = ingest(
            &store,
            &mut bloom,
            provider.as_ref(),
            &cfg,
            req("The API  key lives in 1Password"),
        )
        .unwrap();
        assert!(out.is_duplicate());
        assert!(out.memory().recurrence_score > 0.0);
        assert_eq!(store.stats().unwrap().total_memories, 1);
    }

    #[test]
    fn long_input_becomes_source_document_and_chunks() {
        let (store, mut bloom, provider, cfg) = setup();
        let long = "# Rust memory graph\n\n".to_string()
            + &"This section discusses embeddings, recurrence, source chunks, and graph support. "
                .repeat(180);
        let out = ingest(&store, &mut bloom, provider.as_ref(), &cfg, req(&long)).unwrap();
        let item = out.memory();
        assert_eq!(item.memory_type, MemoryType::RawEvent);
        assert!(item.importance_score <= 0.10);

        let source_tag = item
            .tags
            .iter()
            .find_map(|t| t.strip_prefix("source_doc:"))
            .expect("source document tag");
        let doc = store
            .get_source_document(source_tag)
            .unwrap()
            .expect("source document row");
        assert_eq!(doc.source_type, InputMode::PastedDocument);
        assert!(store.source_chunks(source_tag).unwrap().len() > 1);
    }
}
