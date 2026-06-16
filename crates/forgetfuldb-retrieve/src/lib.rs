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

pub mod traverse;

use anyhow::Result;
use forgetfuldb_core::config::Config;
use forgetfuldb_core::ingest::tokenize;
use forgetfuldb_core::scoring::{retrieval_score, ScoreBreakdown, ScoreInputs};
use forgetfuldb_core::types::{MemoryItem, MemoryType};
use forgetfuldb_core::{age_days, decay, now_unix};
use forgetfuldb_embed::{cosine_similarity, EmbeddingProvider};
use forgetfuldb_store::Store;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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
    /// Query a whole era by its ordinal: resolves to that epoch's time window
    /// and implies `bypass_decay` (an era lookup wants everything from that
    /// stretch, faded or not). Explicit `since`/`until` take precedence.
    pub epoch_ordinal: Option<i64>,
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
            epoch_ordinal: None,
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
    /// The connected subgraph among the injected memories (multi-hop only):
    /// the paths that link them, so the prompt can show *how* they relate.
    /// `None` unless `spreading.inject_subgraph` is on.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub subgraph: Option<ContextSubgraph>,
}

/// The relationship structure among injected memories — a set of paths
/// through the typed graph, each a chain of memory ids and the edge type
/// joining each step.
#[derive(Debug, Clone, Serialize)]
pub struct ContextSubgraph {
    pub paths: Vec<ContextPath>,
}

/// One path: `nodes[i]` connects to `nodes[i+1]` via `edges[i]`, so
/// `edges.len() == nodes.len() - 1`.
#[derive(Debug, Clone, Serialize)]
pub struct ContextPath {
    pub nodes: Vec<String>,
    pub edges: Vec<String>,
}

impl ContextPack {
    /// Render the injected subgraph as a compact "connections" block for the
    /// prompt — `"A" —(led to)→ "B" —(related to)→ "C"` — resolving ids to
    /// short snippets from the pack's own memories. `None` if there's nothing
    /// to show.
    pub fn render_subgraph(&self) -> Option<String> {
        let sg = self.subgraph.as_ref()?;
        let snippet = |id: &str| -> Option<String> {
            self.memories.iter().find(|m| m.item.id == id).map(|m| {
                let text = m.item.summary.as_deref().unwrap_or(&m.item.content);
                let short: String = text.chars().take(60).collect();
                format!("\"{}\"", short.trim())
            })
        };
        let phrase = |edge: &str| match edge {
            "sequence" => "led to",
            "co_occurred" => "recalled with",
            "semantic_similar" => "related to",
            _ => "linked to",
        };
        let mut lines = Vec::new();
        for path in &sg.paths {
            let mut parts = Vec::new();
            let mut renderable = true;
            for (i, id) in path.nodes.iter().enumerate() {
                match snippet(id) {
                    Some(s) => {
                        if i > 0 {
                            parts.push(format!("—({})→", phrase(&path.edges[i - 1])));
                        }
                        parts.push(s);
                    }
                    None => {
                        renderable = false;
                        break;
                    }
                }
            }
            if renderable && parts.len() > 1 {
                lines.push(parts.join(" "));
            }
        }
        (!lines.is_empty()).then(|| lines.join("\n"))
    }
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

/// Multi-hop spreading activation. From the top-scoring seeds, activation
/// spreads over the typed graph (all three edge types) with per-hop decay,
/// and each reached candidate gains a boost `= spreading_factor · activation`,
/// **capped at `spreading_factor`** so an associated memory can never outrank
/// a genuine hit (conversational dominance). At `max_hops = 1` this is the
/// original one-hop boost. Returns the full reach map (paths + activation) so
/// the caller can also assemble the injected subgraph. Uses pre-boost scores
/// as seed strengths, so the spread can't cascade off its own boosts.
fn apply_spreading_activation(
    store: &Store,
    cfg: &Config,
    scored: &mut [RetrievedMemory],
) -> Result<HashMap<String, traverse::Reach>> {
    use forgetfuldb_store::pipeline::EDGE_SEQUENCE;

    let edges = store.list_edges()?;
    if edges.is_empty() {
        return Ok(HashMap::new());
    }
    // Adjacency over all edge types. co_occurred / semantic_similar are
    // undirected; sequence is directional (earlier → later), so it's followed
    // forward only.
    let mut adj: HashMap<String, Vec<traverse::AdjEdge>> = HashMap::new();
    for e in &edges {
        adj.entry(e.src_id.clone()).or_default().push(traverse::AdjEdge {
            dst: e.dst_id.clone(),
            edge_type: e.edge_type.clone(),
            weight: e.weight,
        });
        if e.edge_type != EDGE_SEQUENCE {
            adj.entry(e.dst_id.clone()).or_default().push(traverse::AdjEdge {
                dst: e.src_id.clone(),
                edge_type: e.edge_type.clone(),
                weight: e.weight,
            });
        }
    }

    let s = &cfg.spreading;
    let params = traverse::TraverseParams {
        max_hops: s.max_hops.max(1),
        hop_decay: s.hop_decay,
        activation_floor: s.activation_floor,
        co_occurred_factor: s.co_occurred_factor,
        semantic_factor: s.semantic_factor,
        sequence_factor: s.sequence_factor,
    };

    // Seeds: the strongest base hits (pre-boost scores).
    let mut seeds: Vec<(String, f64)> =
        scored.iter().filter(|m| m.score.total > 0.0).map(|m| (m.item.id.clone(), m.score.total)).collect();
    seeds.sort_by(|a, b| b.1.total_cmp(&a.1));
    seeds.truncate(s.seed_count.max(1));

    let reach = traverse::traverse(&seeds, &adj, &params);

    for m in scored.iter_mut() {
        if let Some(r) = reach.get(&m.item.id) {
            let boost = (cfg.spreading_factor * r.activation).min(cfg.spreading_factor);
            if boost > 0.0 {
                m.score.association_boost = boost;
                m.score.total += boost;
            }
        }
    }
    Ok(reach)
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

    // An epoch query is a dated lookup: resolve the ordinal to its
    // [started_at, ended_at) window and bypass decay (we want everything from
    // that era, faded or not). Explicit since/until win over the era's bounds.
    let (epoch_since, epoch_until, epoch_bypass) = match opts.epoch_ordinal {
        Some(ord) => match store.list_epochs()?.into_iter().find(|e| e.ordinal == ord) {
            // `ended_at` is exclusive (next era's start), so the inclusive
            // `until` is one second earlier.
            Some(e) => (Some(e.started_at), e.ended_at.map(|t| t - 1), true),
            None => (None, None, false),
        },
        None => (None, None, false),
    };
    let since = opts.since.or(epoch_since);
    let until = opts.until.or(epoch_until);
    let bypass_decay = opts.bypass_decay || epoch_bypass;

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
        if since.is_some_and(|s| item.created_at < s) || until.is_some_and(|u| item.created_at > u) {
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
        let importance = if bypass_decay {
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

    // Spreading activation: from the strongest hits, activation spreads over
    // the typed graph (K hops, per-hop decay), so retrieving one memory
    // surfaces its companions — and the paths that connect them. Computed
    // from pre-boost scores so it can't cascade off its own boosts. Off
    // unless enabled in config.
    let reach = if cfg.spreading_activation && cfg.spreading_factor > 0.0 {
        apply_spreading_activation(store, cfg, &mut scored)?
    } else {
        HashMap::new()
    };

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

    // Multi-hop subgraph: pull the connective memories (linked to the hits but
    // not themselves top-k) into the result and record the paths. Mutates
    // `scored`, so it runs before the touch loop below.
    let subgraph = inject_subgraph(store, &mut scored, &reach, cfg)?;

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
        subgraph,
    })
}

/// Assemble the injected subgraph from the walk, and pull the connective
/// memories it found into the result. These are memories *linked* to the hits
/// but not themselves top-k by similarity — the whole point of multi-hop — so
/// they're appended to `memories` (ranked after the direct hits, scored purely
/// by association) and their content becomes part of the prompt. Paths are
/// kept maximal (a prefix of a longer chain is dropped) and bounded by
/// `subgraph_max_nodes` newly-injected memories.
fn inject_subgraph(
    store: &Store,
    memories: &mut Vec<RetrievedMemory>,
    reach: &HashMap<String, traverse::Reach>,
    cfg: &Config,
) -> Result<Option<ContextSubgraph>> {
    if !cfg.spreading.inject_subgraph {
        return Ok(None);
    }
    let present: HashSet<String> = memories.iter().map(|m| m.item.id.clone()).collect();

    // Traversed-in paths, longest first (so the maximal chain wins the budget
    // and its prefixes fold into it), then strongest.
    let mut cands: Vec<&traverse::Reach> =
        reach.values().filter(|r| r.depth >= 1 && !r.edges.is_empty()).collect();
    cands.sort_by(|a, b| b.path.len().cmp(&a.path.len()).then(b.activation.total_cmp(&a.activation)));

    let max_nodes = cfg.spreading.subgraph_max_nodes;
    let mut paths: Vec<ContextPath> = Vec::new();
    let mut inject: Vec<String> = Vec::new(); // new connective ids, in order
    let mut injected: HashSet<String> = HashSet::new();

    for r in cands {
        if paths.iter().any(|q| q.nodes.len() > r.path.len() && q.nodes.starts_with(&r.path)) {
            continue; // prefix of a chain we already kept
        }
        let extra: Vec<String> =
            r.path.iter().filter(|id| !present.contains(*id) && !injected.contains(*id)).cloned().collect();
        if inject.len() + extra.len() > max_nodes {
            continue;
        }
        for id in extra {
            injected.insert(id.clone());
            inject.push(id);
        }
        paths.push(ContextPath { nodes: r.path.clone(), edges: r.edges.clone() });
    }
    if paths.is_empty() {
        return Ok(None);
    }

    // Inject the connective memories so the prompt actually contains them and
    // the token-cost metric counts them. Skip stale/archived ones.
    for id in &inject {
        if let Some(item) = store.get_memory(id)? {
            if item.stale || item.memory_type == MemoryType::Archive {
                continue;
            }
            let activation = reach.get(id).map(|r| r.activation).unwrap_or(0.0);
            memories.push(RetrievedMemory { item, score: association_score(activation) });
        }
    }
    Ok(Some(ContextSubgraph { paths }))
}

/// A score breakdown for a memory pulled in purely by graph association — its
/// only nonzero term is the activation it accumulated.
fn association_score(activation: f64) -> ScoreBreakdown {
    ScoreBreakdown {
        semantic_similarity: 0.0,
        importance: 0.0,
        recurrence: 0.0,
        recency: 0.0,
        pinned_boost: 0.0,
        staleness_penalty: 0.0,
        conversational_damping: 1.0,
        association_boost: activation,
        salience: 0.0,
        total: activation,
    }
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
    fn multi_hop_surfaces_linked_memories_below_direct_hits() {
        let (store, provider, mut cfg) = setup();
        let hit = add(&store, provider.as_ref(), &cfg, "the standup is at nine thirty on mondays", vec![]);
        let mid = add(&store, provider.as_ref(), &cfg, "the sprint demo is on friday afternoon", vec![]);
        let far = add(&store, provider.as_ref(), &cfg, "we ordered pizza for the team retro lunch", vec![]);

        let edge = |a: &str, b: &str| forgetfuldb_store::MemoryEdge {
            src_id: a.min(b).to_string(),
            dst_id: a.max(b).to_string(),
            edge_type: forgetfuldb_store::pipeline::EDGE_CO_OCCURRED.to_string(),
            weight: 5.0,
            co_count: 5,
            created_at: 0,
            last_activated: 0,
        };
        store.upsert_edge(&edge(&hit, &mid)).unwrap(); // hit — mid (1 hop)
        store.upsert_edge(&edge(&mid, &far)).unwrap(); // mid — far (2 hops from hit)

        cfg.spreading_activation = true;
        cfg.spreading.seed_count = 1; // only the direct hit seeds the walk

        // One hop: only the direct neighbor is reached; far gets nothing.
        cfg.spreading.max_hops = 1;
        let one = retrieve(&store, provider.as_ref(), &cfg, "standup time", &RetrieveOptions::default()).unwrap();
        let far_one = one.memories.iter().find(|m| m.item.id == far).unwrap();
        assert_eq!(far_one.score.association_boost, 0.0, "two hops away gets nothing at max_hops=1");

        // Two hops: far is reached and lifted — but never above the direct hit.
        cfg.spreading.max_hops = 2;
        let two = retrieve(&store, provider.as_ref(), &cfg, "standup time", &RetrieveOptions::default()).unwrap();
        let far_two = two.memories.iter().find(|m| m.item.id == far).unwrap();
        let hit_two = two.memories.iter().find(|m| m.item.id == hit).unwrap();
        assert!(far_two.score.association_boost > 0.0, "the 2-hop memory is reached and boosted");
        assert!(
            hit_two.score.total > far_two.score.total,
            "the direct hit must still outrank the traversed-in memory ({} vs {})",
            hit_two.score.total,
            far_two.score.total
        );
    }

    #[test]
    fn subgraph_injection_renders_connection_paths() {
        let (store, provider, mut cfg) = setup();
        let hit = add(&store, provider.as_ref(), &cfg, "the standup is at nine thirty on mondays", vec![]);
        let mid = add(&store, provider.as_ref(), &cfg, "the sprint demo is on friday afternoon", vec![]);
        let far = add(&store, provider.as_ref(), &cfg, "we ordered pizza for the team retro lunch", vec![]);

        // Directional sequence edges: hit → mid → far (a reasoning path).
        let seq = |a: &str, b: &str| forgetfuldb_store::MemoryEdge {
            src_id: a.to_string(),
            dst_id: b.to_string(),
            edge_type: forgetfuldb_store::pipeline::EDGE_SEQUENCE.to_string(),
            weight: 5.0,
            co_count: 5,
            created_at: 0,
            last_activated: 0,
        };
        store.upsert_edge(&seq(&hit, &mid)).unwrap();
        store.upsert_edge(&seq(&mid, &far)).unwrap();

        cfg.spreading_activation = true;
        cfg.spreading.seed_count = 1;
        cfg.spreading.max_hops = 2;
        cfg.spreading.inject_subgraph = true;

        let pack = retrieve(&store, provider.as_ref(), &cfg, "standup time", &RetrieveOptions::default()).unwrap();
        let sg = pack.subgraph.as_ref().expect("a subgraph is injected");
        // The full chain is one maximal path; the hit→mid prefix is folded in.
        assert!(
            sg.paths.iter().any(|p| p.nodes == vec![hit.clone(), mid.clone(), far.clone()]),
            "the 2-hop chain is a path"
        );
        assert!(!sg.paths.iter().any(|p| p.nodes == vec![hit.clone(), mid.clone()]), "the prefix path is dropped");

        let rendered = pack.render_subgraph().expect("renders to text");
        assert!(rendered.contains("led to"), "sequence edges render as 'led to': {rendered}");
        assert!(rendered.contains("pizza"), "the far memory's snippet appears: {rendered}");
    }

    #[test]
    fn subgraph_injects_connective_memories_beyond_top_k() {
        let (store, provider, mut cfg) = setup();
        let hit = add(&store, provider.as_ref(), &cfg, "the standup is at nine thirty on mondays", vec![]);
        let b = add(&store, provider.as_ref(), &cfg, "office plants need watering on tuesdays", vec![]);
        let c = add(&store, provider.as_ref(), &cfg, "the parking garage closes at midnight", vec![]);
        // Noise so a top_k=1 flat result excludes b and c.
        for i in 0..5 {
            add(&store, provider.as_ref(), &cfg, &format!("filler memory number {i} about nothing in particular"), vec![]);
        }
        let seq = |a: &str, d: &str| forgetfuldb_store::MemoryEdge {
            src_id: a.to_string(),
            dst_id: d.to_string(),
            edge_type: forgetfuldb_store::pipeline::EDGE_SEQUENCE.to_string(),
            weight: 6.0,
            co_count: 6,
            created_at: 0,
            last_activated: 0,
        };
        store.upsert_edge(&seq(&hit, &b)).unwrap(); // hit → b → c
        store.upsert_edge(&seq(&b, &c)).unwrap();

        cfg.spreading_activation = true;
        cfg.spreading.seed_count = 1;
        cfg.spreading.max_hops = 2;
        cfg.spreading.inject_subgraph = true;

        // Flat top-k is just the hit; the chain's other links aren't top-k.
        let opts = RetrieveOptions { top_k: 1, ..Default::default() };
        let pack = retrieve(&store, provider.as_ref(), &cfg, "standup time", &opts).unwrap();

        let ids: HashSet<&str> = pack.memories.iter().map(|m| m.item.id.as_str()).collect();
        assert!(ids.contains(hit.as_str()), "the direct hit is present");
        assert!(
            ids.contains(b.as_str()) && ids.contains(c.as_str()),
            "the connective memories are injected past top_k=1"
        );
        assert!(pack.memories.len() > opts.top_k, "subgraph injection grows the result beyond top_k");
        assert!(pack.render_subgraph().unwrap().contains("led to"), "the chain renders");
    }

    #[test]
    fn no_subgraph_unless_opted_in() {
        let (store, provider, mut cfg) = setup();
        let a = add(&store, provider.as_ref(), &cfg, "alpha standup note", vec![]);
        let b = add(&store, provider.as_ref(), &cfg, "bravo unrelated note", vec![]);
        store
            .upsert_edge(&forgetfuldb_store::MemoryEdge {
                src_id: a.clone().min(b.clone()),
                dst_id: a.clone().max(b.clone()),
                edge_type: forgetfuldb_store::pipeline::EDGE_CO_OCCURRED.to_string(),
                weight: 5.0,
                co_count: 5,
                created_at: 0,
                last_activated: 0,
            })
            .unwrap();
        cfg.spreading_activation = true; // boosts on, but subgraph off
        let pack = retrieve(&store, provider.as_ref(), &cfg, "standup", &RetrieveOptions::default()).unwrap();
        assert!(pack.subgraph.is_none(), "no subgraph unless inject_subgraph is set");
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

    #[test]
    fn epoch_ordinal_restricts_retrieval_to_that_era() {
        let (store, provider, cfg) = setup();
        let now = now_unix();
        let old = add(&store, provider.as_ref(), &cfg, "the gardening project notes from the early era", vec![]);
        let recent = add(&store, provider.as_ref(), &cfg, "the database migration notes from the current era", vec![]);
        // Age `old` into the past; `recent` stays at now.
        let mut m = store.get_memory(&old).unwrap().unwrap();
        m.created_at = now - 40 * 86_400;
        store.update_memory(&m).unwrap();

        // Era 0 = [now-50d, now-20d); era 1 = [now-20d, open).
        store
            .replace_epochs(&[
                forgetfuldb_store::Epoch {
                    id: "e0".into(),
                    ordinal: 0,
                    started_at: now - 50 * 86_400,
                    ended_at: Some(now - 20 * 86_400),
                    centroid: None,
                    label: None,
                    summary: None,
                    member_count: 1,
                    drift_in: 0.0,
                },
                forgetfuldb_store::Epoch {
                    id: "e1".into(),
                    ordinal: 1,
                    started_at: now - 20 * 86_400,
                    ended_at: None,
                    centroid: None,
                    label: None,
                    summary: None,
                    member_count: 1,
                    drift_in: 0.5,
                },
            ])
            .unwrap();

        // Querying era 0 returns only its member — even though it's 40 days
        // old (the era lookup bypasses decay).
        let pack0 = retrieve(
            &store,
            provider.as_ref(),
            &cfg,
            "notes",
            &RetrieveOptions { epoch_ordinal: Some(0), ..Default::default() },
        )
        .unwrap();
        let ids0: Vec<&str> = pack0.memories.iter().map(|m| m.item.id.as_str()).collect();
        assert!(ids0.contains(&old.as_str()), "era 0 query returns the old-era memory");
        assert!(!ids0.contains(&recent.as_str()), "era 0 query excludes the current-era memory");

        // Querying era 1 returns only the current-era memory.
        let pack1 = retrieve(
            &store,
            provider.as_ref(),
            &cfg,
            "notes",
            &RetrieveOptions { epoch_ordinal: Some(1), ..Default::default() },
        )
        .unwrap();
        let ids1: Vec<&str> = pack1.memories.iter().map(|m| m.item.id.as_str()).collect();
        assert!(ids1.contains(&recent.as_str()) && !ids1.contains(&old.as_str()));
    }
}
