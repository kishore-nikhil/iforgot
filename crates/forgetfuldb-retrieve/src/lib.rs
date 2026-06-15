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
#[serde(default)]
pub struct RetrieveOptions {
    pub top_k: usize,
    /// Stale memories are excluded unless explicitly requested.
    pub include_stale: bool,
    /// Archived memories are excluded unless explicitly requested.
    pub include_archived: bool,
    /// Relevance gate: memories scoring below this are dropped even when
    /// `top_k` isn't filled. An empty result beats a misleading one.
    /// 0.0 (the default) keeps the historical "always return top_k"
    /// behavior.
    pub min_score: f64,
    /// Exclude memories ingested by this chat session (tagged
    /// `session:<id>`). The live conversation is already in the prompt as
    /// history; re-injecting it as "memories" wastes tokens and competes
    /// with itself.
    pub exclude_session: Option<String>,
    /// Score multiplier in (0, 1] for verbatim conversational memories
    /// (chat-sourced raw events / episodic turns). Distilled semantic,
    /// preference and procedural memories are unaffected. 1.0 disables.
    pub conversational_damping: f64,
    /// Restrict candidates to these memory types (None = all).
    pub memory_types: Option<Vec<MemoryType>>,
    /// Restrict candidates to memories created at/after this unix time.
    pub since: Option<i64>,
    /// Restrict candidates to memories created at/before this unix time.
    pub until: Option<i64>,
    /// Also return near-misses: memories that scored above zero but below
    /// `min_score`, with full breakdowns. For the retrieval inspector —
    /// "what almost got injected".
    pub debug: bool,
    /// Skip decay entirely and use raw importance. For *dated* / epoch
    /// queries ("everything I did in 2026"): decay governs ambient recall
    /// (what surfaces unprompted), but a temporal lookup is a different
    /// operation — "what happened in this interval", regardless of whether
    /// it has faded.
    pub bypass_decay: bool,
}

impl Default for RetrieveOptions {
    fn default() -> Self {
        RetrieveOptions {
            top_k: 10,
            include_stale: false,
            include_archived: false,
            min_score: 0.0,
            exclude_session: None,
            conversational_damping: 1.0,
            memory_types: None,
            since: None,
            until: None,
            debug: false,
            bypass_decay: false,
        }
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
    /// The relevance gate that was applied (0.0 = none).
    #[serde(default)]
    pub min_score: f64,
    /// Debug mode only: memories that scored above zero but below the
    /// gate. Never injected into prompts.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub near_misses: Vec<RetrievedMemory>,
}

/// Jaccard-style overlap between query tokens and memory tokens/tags/topic.
/// Complements the placeholder embedding with exact term matching.
pub fn keyword_overlap(query_tokens: &HashSet<String>, item: &MemoryItem) -> f64 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let mut item_tokens: HashSet<String> = tokenize(&item.content).into_iter().collect();
    for tag in &item.tags {
        // `session:<id>` is a system tag (used to exclude the live
        // session from retrieval), not content — don't let the word
        // "session" in a query match every chat memory.
        if tag.starts_with("session:") {
            continue;
        }
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

/// Cap on near-misses returned in debug mode.
const MAX_NEAR_MISSES: usize = 20;

/// Add a one-hop spreading-activation boost in place: each candidate gains
/// `spreading_factor · Σ (neighbor_base_score · w/(1+w))` over its
/// co-occurrence edges, capped at `spreading_factor` so associations
/// never dominate genuine relevance. Uses pre-boost scores throughout.
fn apply_spreading_activation(store: &Store, cfg: &Config, scored: &mut [RetrievedMemory]) -> Result<()> {
    use std::collections::HashMap;
    let edges = store.list_edges()?;
    if edges.is_empty() {
        return Ok(());
    }
    // Undirected adjacency for the co-occurrence edges.
    let mut adj: HashMap<&str, Vec<(&str, f64)>> = HashMap::new();
    for e in &edges {
        if e.edge_type != forgetfuldb_store::pipeline::EDGE_CO_OCCURRED {
            continue;
        }
        adj.entry(&e.src_id).or_default().push((&e.dst_id, e.weight));
        adj.entry(&e.dst_id).or_default().push((&e.src_id, e.weight));
    }
    // Snapshot pre-boost scores (owned keys, so the mutable pass below is
    // free of the borrow) so the spread can't cascade across hits.
    let base: HashMap<String, f64> = scored.iter().map(|m| (m.item.id.clone(), m.score.total)).collect();

    for m in scored.iter_mut() {
        let Some(neighbors) = adj.get(m.item.id.as_str()) else { continue };
        let raw: f64 = neighbors
            .iter()
            .filter_map(|(nbr, w)| base.get(*nbr).map(|s| s * (w / (1.0 + w))))
            .sum();
        let boost = (cfg.spreading_factor * raw).min(cfg.spreading_factor);
        if boost > 0.0 {
            m.score.association_boost = boost;
            m.score.total += boost;
        }
    }
    Ok(())
}

/// A verbatim conversational turn: chat-sourced and never distilled into
/// a semantic / preference / procedural fact.
fn is_conversational(item: &MemoryItem) -> bool {
    item.source.as_deref() == Some("chat")
        && matches!(item.memory_type, MemoryType::RawEvent | MemoryType::Episodic)
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
    let session_tag = opts.exclude_session.as_ref().map(|s| format!("session:{s}"));

    let mut scored: Vec<RetrievedMemory> = Vec::new();
    for item in store.list_memories(None)? {
        if item.stale && !opts.include_stale {
            continue;
        }
        if item.memory_type == MemoryType::Archive && !opts.include_archived {
            continue;
        }
        if let Some(types) = &opts.memory_types {
            if !types.contains(&item.memory_type) {
                continue;
            }
        }
        if opts.since.is_some_and(|s| item.created_at < s) || opts.until.is_some_and(|u| item.created_at > u) {
            continue;
        }
        // The excluded session's turns are already in the prompt as live
        // history — they must not come back as "memories" of themselves.
        if let Some(tag) = &session_tag {
            if item.tags.iter().any(|t| t == tag) {
                continue;
            }
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
        // Salience resists decay; a dated query bypasses decay entirely.
        let importance = if opts.bypass_decay {
            item.importance_score
        } else {
            decay::decay_score_resisted(
                item.importance_score,
                lambda,
                age_days(item.created_at, now),
                item.pinned,
                item.salience,
                cfg.salience_resist,
            )
        };
        let recency = decay::recency_score(age_days(
            item.last_accessed_at.unwrap_or(item.created_at),
            now,
        ));

        let mut breakdown = retrieval_score(
            &ScoreInputs {
                semantic_similarity,
                importance,
                recurrence: item.recurrence_score,
                recency,
                pinned: item.pinned,
                stale: item.stale,
                salience: item.salience,
            },
            weights,
        );
        // Verbatim conversational turns hijack chats when re-injected:
        // damp them so only strong matches survive, while distilled
        // semantic/preference/procedural memories rank normally.
        if opts.conversational_damping < 1.0 && is_conversational(&item) {
            breakdown.conversational_damping = opts.conversational_damping;
            breakdown.total *= opts.conversational_damping;
        }
        scored.push(RetrievedMemory { item, score: breakdown });
    }

    // Spreading activation: a memory that co-occurs (in past chat turns)
    // with strong hits gets an additive boost, so retrieving one memory
    // surfaces its companions. One hop, computed from pre-boost scores so
    // it can't cascade. Off unless enabled in config.
    if cfg.spreading_activation && cfg.spreading_factor > 0.0 {
        apply_spreading_activation(store, cfg, &mut scored)?;
    }

    scored.sort_by(|a, b| b.score.total.partial_cmp(&a.score.total).unwrap_or(std::cmp::Ordering::Equal));
    let mut near_misses = Vec::new();
    if opts.min_score > 0.0 {
        if opts.debug {
            near_misses = scored
                .iter()
                .filter(|m| m.score.total > 0.0 && m.score.total < opts.min_score)
                .take(MAX_NEAR_MISSES)
                .cloned()
                .collect();
        }
        scored.retain(|m| m.score.total >= opts.min_score);
    }
    scored.truncate(opts.top_k);

    // Retrieval counts as access: it slows future decay-driven cleanup.
    // Near-misses are NOT touched — looking at what almost matched must
    // not rehearse it.
    for hit in &scored {
        store.touch_memory(&hit.item.id, now)?;
    }

    Ok(ContextPack {
        query: query.to_string(),
        generated_at: now,
        memories: scored,
        min_score: opts.min_score,
        near_misses,
    })
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

    fn add_chat_turn(store: &Store, provider: &dyn EmbeddingProvider, cfg: &Config, text: &str, session: &str) -> String {
        let mut bloom = warm_bloom(store).unwrap();
        let out = ingest(
            store,
            &mut bloom,
            provider,
            cfg,
            IngestRequest {
                text: text.to_string(),
                source: Some("chat".to_string()),
                tags: vec![format!("session:{session}")],
                memory_type: Some(MemoryType::Episodic),
                session_id: Some(session.to_string()),
                role: Some("user".to_string()),
            },
        )
        .unwrap();
        out.memory().id.clone()
    }

    #[test]
    fn min_score_gates_weak_matches() {
        let (store, provider, cfg) = setup();
        add(&store, provider.as_ref(), &cfg, "the cat sleeps on the windowsill every afternoon", vec![]);

        let open = retrieve(&store, provider.as_ref(), &cfg, "plot perfect billing stripe", &RetrieveOptions::default()).unwrap();
        assert_eq!(open.memories.len(), 1, "no gate: weak match still returned");

        let gated_opts = RetrieveOptions { min_score: 0.4, ..Default::default() };
        let gated = retrieve(&store, provider.as_ref(), &cfg, "plot perfect billing stripe", &gated_opts).unwrap();
        assert!(gated.memories.is_empty(), "an unrelated memory must not pass the gate");

        // A direct hit still clears the same gate.
        add(&store, provider.as_ref(), &cfg, "plot perfect billing runs on stripe invoices", vec![]);
        let hit = retrieve(&store, provider.as_ref(), &cfg, "plot perfect billing stripe", &gated_opts).unwrap();
        assert_eq!(hit.memories.len(), 1);
        assert!(hit.memories[0].item.content.contains("billing"));
    }

    #[test]
    fn excluded_session_memories_are_skipped() {
        let (store, provider, cfg) = setup();
        add_chat_turn(&store, provider.as_ref(), &cfg, "standup moved to nine thirty", "live");
        add_chat_turn(&store, provider.as_ref(), &cfg, "standup notes go in the wiki", "old");

        let opts = RetrieveOptions { exclude_session: Some("live".to_string()), ..Default::default() };
        let pack = retrieve(&store, provider.as_ref(), &cfg, "standup", &opts).unwrap();
        assert_eq!(pack.memories.len(), 1);
        assert!(pack.memories[0].item.content.contains("wiki"));
    }

    #[test]
    fn conversational_turns_are_damped() {
        let (store, provider, cfg) = setup();
        // The same fact as a verbatim chat turn and as a distilled memory.
        add_chat_turn(&store, provider.as_ref(), &cfg, "billing for plot perfect runs on stripe", "old");
        add(&store, provider.as_ref(), &cfg, "plot perfect billing runs on stripe invoices", vec![]);

        let opts = RetrieveOptions { conversational_damping: 0.5, ..Default::default() };
        let pack = retrieve(&store, provider.as_ref(), &cfg, "plot perfect billing stripe", &opts).unwrap();

        let turn = pack.memories.iter().find(|m| m.item.memory_type == MemoryType::Episodic).unwrap();
        let fact = pack.memories.iter().find(|m| m.item.memory_type == MemoryType::Semantic).unwrap();
        assert_eq!(turn.score.conversational_damping, 0.5);
        assert_eq!(fact.score.conversational_damping, 1.0, "distilled memories are never damped");
        assert!(fact.score.total > turn.score.total, "the distilled fact must outrank the verbatim turn");
        assert_eq!(pack.memories[0].item.id, fact.item.id);
    }

    #[test]
    fn session_tags_do_not_leak_into_keyword_overlap() {
        let (store, provider, cfg) = setup();
        add_chat_turn(&store, provider.as_ref(), &cfg, "the deploy finished cleanly", "abc123");
        let pack = retrieve(&store, provider.as_ref(), &cfg, "session abc123", &RetrieveOptions::default()).unwrap();
        // Without the skip, keyword overlap alone would contribute 0.3
        // (both query tokens match the tag parts). Allow a little noise
        // from hashed-BoW collisions, but nowhere near that.
        assert!(
            pack.memories[0].score.semantic_similarity < 0.15,
            "querying the word 'session' must not match the system tag: {}",
            pack.memories[0].score.semantic_similarity
        );
    }

    #[test]
    fn debug_mode_reports_near_misses() {
        let (store, provider, cfg) = setup();
        add(&store, provider.as_ref(), &cfg, "plot perfect billing runs on stripe invoices", vec![]);
        add(&store, provider.as_ref(), &cfg, "the cat sleeps on the windowsill every afternoon", vec![]);

        let opts = RetrieveOptions { min_score: 0.4, debug: true, ..Default::default() };
        let pack = retrieve(&store, provider.as_ref(), &cfg, "plot perfect billing stripe", &opts).unwrap();
        assert_eq!(pack.memories.len(), 1);
        assert_eq!(pack.min_score, 0.4);
        assert_eq!(pack.near_misses.len(), 1, "the weak match shows up as a near-miss");
        assert!(pack.near_misses[0].item.content.contains("cat"));
        assert!(pack.near_misses[0].score.total < 0.4);

        // Without debug, near-misses stay private.
        let quiet = RetrieveOptions { min_score: 0.4, ..Default::default() };
        let pack = retrieve(&store, provider.as_ref(), &cfg, "plot perfect billing stripe", &quiet).unwrap();
        assert!(pack.near_misses.is_empty());
    }

    #[test]
    fn type_and_time_filters_restrict_candidates() {
        let (store, provider, cfg) = setup();
        let id = add(&store, provider.as_ref(), &cfg, "standup is at nine thirty", vec![]);

        let wrong_type = RetrieveOptions {
            memory_types: Some(vec![MemoryType::Preference]),
            ..Default::default()
        };
        assert!(retrieve(&store, provider.as_ref(), &cfg, "standup", &wrong_type).unwrap().memories.is_empty());

        let right_type = RetrieveOptions {
            memory_types: Some(vec![MemoryType::Semantic]),
            ..Default::default()
        };
        assert_eq!(retrieve(&store, provider.as_ref(), &cfg, "standup", &right_type).unwrap().memories[0].item.id, id);

        let future_only = RetrieveOptions { since: Some(now_unix() + 1000), ..Default::default() };
        assert!(retrieve(&store, provider.as_ref(), &cfg, "standup", &future_only).unwrap().memories.is_empty());

        let past_window = RetrieveOptions { until: Some(now_unix() + 1000), ..Default::default() };
        assert_eq!(retrieve(&store, provider.as_ref(), &cfg, "standup", &past_window).unwrap().memories.len(), 1);
    }

    #[test]
    fn spreading_activation_boosts_associated_memories() {
        let (store, provider, mut cfg) = setup();
        // A memory that matches the query, and an unrelated one that does not.
        let _hit = add(&store, provider.as_ref(), &cfg, "the standup is at nine thirty on mondays", vec![]);
        let companion = add(&store, provider.as_ref(), &cfg, "the sprint demo is on friday afternoon", vec![]);

        // Without an edge, the companion gets no boost.
        let base = retrieve(&store, provider.as_ref(), &cfg, "standup time", &RetrieveOptions::default()).unwrap();
        let base_companion = base.memories.iter().find(|m| m.item.id == companion).unwrap().score.total;

        // Associate them (as if they'd been retrieved together before) and
        // enable spreading activation.
        store
            .upsert_edge(&forgetfuldb_store::MemoryEdge {
                src_id: base.memories[0].item.id.clone().min(companion.clone()),
                dst_id: base.memories[0].item.id.clone().max(companion.clone()),
                edge_type: forgetfuldb_store::pipeline::EDGE_CO_OCCURRED.to_string(),
                weight: 3.0,
                co_count: 3,
                created_at: 0,
                last_activated: 0,
            })
            .unwrap();
        cfg.spreading_activation = true;

        let spread = retrieve(&store, provider.as_ref(), &cfg, "standup time", &RetrieveOptions::default()).unwrap();
        let boosted = spread.memories.iter().find(|m| m.item.id == companion).unwrap();
        assert!(boosted.score.association_boost > 0.0, "companion should gain an association boost");
        assert!(
            boosted.score.total > base_companion,
            "spreading activation should raise the companion's total ({} vs {})",
            boosted.score.total,
            base_companion
        );
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
