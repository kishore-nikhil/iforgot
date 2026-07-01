//! forgetfuldb-consolidate
//!
//! The "sleep cycle" of ForgetfulDB. Run periodically (or via
//! `forgetfuldb consolidate`) to:
//!
//! 1. merge near-duplicate memories (cosine >= threshold)
//! 2. refresh recurrence scores from topic/entity frequency
//!    (Count-Min Sketch)
//! 3. cluster related memories by topic and write a summary memory
//!    (pluggable [`Summarizer`]; v1 is a dumb extractive one)
//! 4. promote frequently accessed episodic memories to semantic memory
//! 5. mark contradicted/updated memories stale
//! 6. archive decayed old raw events, and delete long-archived ones —
//!    keeping a reservoir-sampled representative trace

pub mod summarize;

pub use summarize::{ExtractiveSummarizer, Summarizer};

use anyhow::Result;
use forgetfuldb_core::config::Config;
use forgetfuldb_core::ids::new_id;
use forgetfuldb_core::ingest::content_hash;
use forgetfuldb_core::types::{EvidenceType, LinkRelation, MemoryItem, MemoryLink, MemoryType};
use forgetfuldb_core::{age_days, decay, now_unix};
use forgetfuldb_embed::cosine_similarity;
use forgetfuldb_prob::ReservoirSampler;
use forgetfuldb_store::Store;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

/// What one consolidation pass did.
#[derive(Debug, Default, Clone, Serialize)]
pub struct ConsolidationReport {
    pub duplicates_merged: usize,
    /// Bursts (dense, temporally-tight clusters of similar events) collapsed
    /// into a gist while keeping the one anomalous outlier.
    pub bursts_collapsed: usize,
    pub recurrence_updated: usize,
    pub clusters_summarized: usize,
    pub promoted_to_semantic: usize,
    /// Semantic/preference memories concluded as decay-exempt Foundation
    /// traits from accumulated habit evidence this pass.
    pub promoted_to_foundation: usize,
    /// Supersession (`Updates`) edges inferred from similar memories this pass
    /// (the staleness attack). Their targets are staled by `mark_contradicted_stale`.
    pub contradictions_inferred: usize,
    /// Previously-staled memories revived because their value was reasserted
    /// as the current one (the staleness call self-healed).
    pub revived: usize,
    pub marked_stale: usize,
    pub archived: usize,
    pub deleted: usize,
    /// Provenance of every summary memory created this pass.
    pub summaries: Vec<forgetfuldb_store::SummaryProvenance>,
    /// Co-occurrence association edges rebuilt from chat history.
    pub associations: usize,
    /// Memories whose salience was revised by the neighbor discriminator.
    pub salience_revised: usize,
    /// Memories whose noisy `topic` was refined from its cluster.
    pub topics_refined: usize,
    /// Memories whose importance was revised from V2 evidence/graph/decay.
    pub importance_revised: usize,
    /// `semantic_similar` (cosine kNN) edges rebuilt.
    pub semantic_edges: usize,
    /// `sequence` (session-order) edges rebuilt.
    pub sequence_edges: usize,
    /// Drift-segmented eras the timeline was partitioned into this pass.
    pub epochs: usize,
}

/// Run a full consolidation pass. Every pass is logged to the
/// `consolidation_runs` table so the observability UI can show what each
/// sleep cycle did.
pub fn consolidate(
    store: &Store,
    summarizer: &dyn Summarizer,
    cfg: &Config,
) -> Result<ConsolidationReport> {
    let mut report = ConsolidationReport::default();
    let now = now_unix();

    merge_duplicates(store, cfg, now, &mut report)?;
    collapse_bursts(store, summarizer, cfg, now, &mut report)?;
    refresh_recurrence(store, now, &mut report)?;
    refresh_importance(store, cfg, now, &mut report)?;
    revise_salience(store, now, &mut report)?;
    refine_topics(store, cfg, now, &mut report)?;
    summarize_clusters(store, summarizer, cfg, now, &mut report)?;
    promote_recurring(store, cfg, now, &mut report)?;
    promote_to_foundation(store, cfg, now, &mut report)?;
    infer_contradictions(store, cfg, &mut report)?;
    mark_contradicted_stale(store, &mut report)?;
    revive_reasserted(store, cfg, &mut report)?;
    archive_and_prune(store, cfg, now, &mut report)?;

    // Rebuild the association graph from scratch. Done last, after pruning,
    // so edges never point at deleted memories. Three distinct edge types,
    // each a different notion of "related":
    //   co_occurred     — recalled together (behavioral / Hebbian)
    //   semantic_similar — close in meaning (embedding kNN)
    //   sequence        — discussed one after another (causal / session order)
    report.associations = forgetfuldb_store::pipeline::rebuild_cooccurrence_edges(
        store,
        cfg.edge_decay_lambda,
        cfg.edge_min_weight,
        now,
    )?;
    report.semantic_edges = forgetfuldb_store::pipeline::rebuild_semantic_edges(
        store,
        cfg.semantic_edge_min_sim,
        cfg.semantic_edge_top_k,
        now,
    )?;
    report.sequence_edges = forgetfuldb_store::pipeline::rebuild_sequence_edges(
        store,
        cfg.edge_decay_lambda,
        cfg.edge_min_weight,
        now,
        2,
    )?;

    // Organize the (now-pruned) timeline into drift-segmented eras. Last,
    // because it reads the final surviving corpus; the boundaries it writes
    // guide the *next* pass's within-epoch consolidation.
    segment_epochs(store, summarizer, cfg, &mut report)?;

    store.log_consolidation_run(&forgetfuldb_store::ConsolidationRun {
        id: new_id("run", &format!("consolidate-{now}")),
        ran_at: now,
        duplicates_merged: report.duplicates_merged as i64,
        recurrence_updated: report.recurrence_updated as i64,
        clusters_summarized: report.clusters_summarized as i64,
        promoted: (report.promoted_to_semantic + report.promoted_to_foundation) as i64,
        marked_stale: report.marked_stale as i64,
        archived: report.archived as i64,
        pruned: report.deleted as i64,
        summaries: report.summaries.clone(),
    })?;

    Ok(report)
}

/// Authoritative salience pass: classify every memory by its near-neighbor
/// distribution over time (the shared discriminator) and set salience to
/// the U-shaped max of surprise and habit, gated by relevance. This is what
/// lets a formative memory resist the decay that buries the routine around
/// it. O(n^2) over active memories — fine at personal scale (the sleep
/// cycle is off the conversation path).
fn revise_salience(store: &Store, now: i64, report: &mut ConsolidationReport) -> Result<()> {
    use forgetfuldb_core::salience::salience;

    for (item, stats) in analyze_active(store, now)? {
        let relevance = forgetfuldb_core::salience::content_relevance(
            item.content.chars().count(),
            item.entities.len(),
        );
        let new_salience = salience(&stats, relevance);
        if (new_salience - item.salience).abs() > 1e-6 {
            let mut updated = item.clone();
            updated.salience = new_salience;
            updated.updated_at = now;
            store.update_memory(&updated)?;
            report.salience_revised += 1;
        }
    }
    Ok(())
}

/// Run the shared neighbor discriminator over every active (non-archive,
/// embedded) memory and return each item paired with its neighbor structure.
/// One O(n²) read of the corpus that the salience and Foundation passes both
/// consume — the "compute once, read many ways" primitive from the spec.
/// Archives are excluded: they're de-emphasized copies of pruned memories,
/// out of the active corpus, so they neither get analyzed nor count as
/// neighbors.
fn analyze_active(
    store: &Store,
    now: i64,
) -> Result<Vec<(MemoryItem, forgetfuldb_core::salience::NeighborStats)>> {
    use forgetfuldb_core::salience::{analyze_neighbors, Neighbor, NeighborParams};

    let items: Vec<MemoryItem> = store
        .list_memories(None)?
        .into_iter()
        .filter(|m| m.memory_type != MemoryType::Archive && m.embedding.is_some())
        .collect();
    if items.len() < 2 {
        return Ok(Vec::new());
    }
    // History span: age of the oldest memory, the window spread is measured
    // against.
    let oldest = items.iter().map(|m| m.created_at).min().unwrap_or(now);
    let history_span_days = age_days(oldest, now).max(1.0);
    let params = NeighborParams::default();

    let mut out = Vec::with_capacity(items.len());
    for item in &items {
        let emb = item.embedding.as_ref().unwrap();
        let neighbors: Vec<Neighbor> = items
            .iter()
            .filter(|o| o.id != item.id)
            .filter_map(|o| {
                o.embedding.as_ref().map(|oe| Neighbor {
                    similarity: cosine_similarity(emb, oe).max(0.0) as f64,
                    age_days: age_days(o.created_at, now),
                })
            })
            .collect();
        let stats = analyze_neighbors(&neighbors, history_span_days, &params);
        out.push((item.clone(), stats));
    }
    Ok(out)
}

/// Conclude Foundation traits from accumulated habit. A semantic/preference
/// memory whose near-neighbors form a *habit* — many of them, spread over a
/// long stretch of history — is no longer just a fact; it's an
/// identity-level trait ("initiated tic-tac-toe 4× over 3 months → likes
/// games"). Promote it to the decay-exempt Foundation tier. This is the
/// temporal inverse of a burst: only patterns that have proven themselves
/// over time graduate.
///
/// A habit is usually a *cluster* of similar memories; promoting every member
/// would mint a dozen copies of one trait. So we collapse each habit to a
/// single Foundation: strongest member first, and skip any candidate already
/// close to a memory we've made Foundational.
fn promote_to_foundation(
    store: &Store,
    cfg: &Config,
    now: i64,
    report: &mut ConsolidationReport,
) -> Result<()> {
    use forgetfuldb_core::salience::{NeighborClass, NeighborParams};

    let min_spread = cfg.consolidation_thresholds.foundation_min_temporal_spread;
    let min_neighbors = cfg.consolidation_thresholds.foundation_min_neighbors;
    let near = NeighborParams::default().similarity_threshold as f32;

    let analyzed = analyze_active(store, now)?;
    // Embeddings of memories already Foundational — the seeds a new trait must
    // not duplicate. Seeded with existing Foundations, grown as we promote.
    let mut foundation_embeddings: Vec<Vec<f32>> = analyzed
        .iter()
        .filter(|(m, _)| m.memory_type == MemoryType::Foundation)
        .filter_map(|(m, _)| m.embedding.clone())
        .collect();

    // Strongest candidates first, so the representative member of a habit
    // cluster is the one that becomes the trait.
    let mut candidates: Vec<(MemoryItem, NeighborClass, f64, usize)> = analyzed
        .into_iter()
        .filter(|(m, _)| !m.stale && !m.pinned)
        .filter(|(m, _)| matches!(m.memory_type, MemoryType::Semantic | MemoryType::Preference))
        .map(|(m, s)| (m, s.class, s.temporal_spread, s.count))
        .filter(|(_, class, spread, count)| {
            *class == NeighborClass::Habit && *spread >= min_spread && *count >= min_neighbors
        })
        .collect();
    candidates.sort_by(|a, b| b.0.importance_score.total_cmp(&a.0.importance_score));

    for (item, _, _, _) in candidates {
        let emb = match &item.embedding {
            Some(e) => e,
            None => continue,
        };
        // Skip if this trait is already represented (an existing or
        // just-promoted Foundation sits within the near-neighbor radius).
        if foundation_embeddings
            .iter()
            .any(|fe| cosine_similarity(emb, fe) >= near)
        {
            continue;
        }
        let mut updated = item.clone();
        updated.memory_type = MemoryType::Foundation;
        updated.importance_score = (updated.importance_score + 0.1).min(1.0);
        updated.updated_at = now;
        store.update_memory(&updated)?;
        foundation_embeddings.push(emb.clone());
        report.promoted_to_foundation += 1;
    }
    Ok(())
}

/// Pure merge logic: fold `dup` into `keep`. The surviving memory absorbs
/// the duplicate's rehearsal history and metadata, mimicking how repeated
/// experiences strengthen a single memory trace.
pub fn merge_pair(mut keep: MemoryItem, dup: &MemoryItem, now: i64) -> MemoryItem {
    keep.access_count += dup.access_count;
    keep.recurrence_score = (keep.recurrence_score + dup.recurrence_score + 0.2).min(1.0);
    keep.importance_score = keep.importance_score.max(dup.importance_score);
    keep.pinned = keep.pinned || dup.pinned;
    keep.created_at = keep.created_at.min(dup.created_at);
    keep.last_accessed_at = match (keep.last_accessed_at, dup.last_accessed_at) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    };
    for tag in &dup.tags {
        if !keep.tags.contains(tag) {
            keep.tags.push(tag.clone());
        }
    }
    for entity in &dup.entities {
        if !keep.entities.contains(entity) {
            keep.entities.push(entity.clone());
        }
    }
    keep.updated_at = now;
    keep
}

/// Find near-duplicate pairs by cosine similarity and merge each into the
/// more important/newer item. O(n^2) over active memories — fine for a
/// personal store, see README limitations.
fn merge_duplicates(
    store: &Store,
    cfg: &Config,
    now: i64,
    report: &mut ConsolidationReport,
) -> Result<()> {
    let threshold = cfg.consolidation_thresholds.duplicate_similarity as f32;
    let items: Vec<MemoryItem> = store
        .list_memories(None)?
        .into_iter()
        .filter(|m| m.memory_type != MemoryType::Archive && m.embedding.is_some())
        .collect();
    // Consolidate *within* an era, preserve *across*: a near-identical memory
    // from a different epoch is contextually distinct, so it isn't merged.
    let epoch_starts = epoch_starts(store)?;

    let mut removed: Vec<bool> = vec![false; items.len()];
    for i in 0..items.len() {
        if removed[i] {
            continue;
        }
        for j in (i + 1)..items.len() {
            if removed[j] {
                continue;
            }
            if !same_epoch(&epoch_starts, items[i].created_at, items[j].created_at) {
                continue;
            }
            let sim = cosine_similarity(
                items[i].embedding.as_ref().unwrap(),
                items[j].embedding.as_ref().unwrap(),
            );
            if sim >= threshold {
                // A Foundation trait always absorbs the duplicate, never the
                // reverse — merge_pair keeps the survivor's type, so the
                // decay-exempt trait must be the survivor. Otherwise keep the
                // higher-importance item; tie goes to the newer one.
                let i_found = items[i].memory_type.is_decay_exempt();
                let j_found = items[j].memory_type.is_decay_exempt();
                let keep_i = if i_found != j_found {
                    i_found
                } else {
                    items[i].importance_score > items[j].importance_score
                        || (items[i].importance_score == items[j].importance_score
                            && items[i].created_at >= items[j].created_at)
                };
                let (keep_idx, dup_idx) = if keep_i { (i, j) } else { (j, i) };
                let merged = merge_pair(items[keep_idx].clone(), &items[dup_idx], now);
                store.update_memory(&merged)?;
                store.insert_link(&MemoryLink {
                    source_id: items[dup_idx].id.clone(),
                    target_id: merged.id.clone(),
                    relation: LinkRelation::Duplicates,
                })?;
                store.delete_memory(&items[dup_idx].id)?;
                removed[dup_idx] = true;
                report.duplicates_merged += 1;
                if dup_idx == i {
                    break; // item i no longer exists
                }
            }
        }
    }
    Ok(())
}

/// Collapse *bursts* into a gist, keeping the anomaly. A burst is a dense
/// cluster of similar event memories packed into a tight time window — a
/// flurry, not a habit (which is the same density spread over time, and is
/// promoted instead of collapsed). The routine members are summarized into a
/// single gist and deleted; the one **outlier** — the member least like the
/// rest — survives, because the surprising thing in a flood is the part that
/// didn't fit.
///
/// This inverts the dedup-merge above, which keeps the *central* member: when
/// the cluster is a transient burst, the center is the disposable routine and
/// the edge is what's worth keeping.
fn collapse_bursts(
    store: &Store,
    summarizer: &dyn Summarizer,
    cfg: &Config,
    now: i64,
    report: &mut ConsolidationReport,
) -> Result<()> {
    use forgetfuldb_core::salience::NeighborParams;

    if !cfg.consolidation_thresholds.burst_collapse_enabled {
        return Ok(());
    }
    let min_size = cfg.consolidation_thresholds.burst_min_size;
    let params = NeighborParams::default();
    let sim_thresh = params.similarity_threshold as f32;

    // History span is measured against the whole corpus, so "tight" means
    // tight relative to the store's lifetime — same yardstick the salience
    // discriminator uses.
    let all = store.list_memories(None)?;
    let oldest = all.iter().map(|m| m.created_at).min().unwrap_or(now);
    let history_span_days = age_days(oldest, now).max(1.0);

    // Only transient event-level memories are collapsible. Distilled
    // knowledge (semantic/preference/foundation), pins, and stale rows are
    // never swept into a gist.
    let items: Vec<MemoryItem> = all
        .into_iter()
        .filter(|m| matches!(m.memory_type, MemoryType::RawEvent | MemoryType::Episodic))
        .filter(|m| !m.pinned && !m.stale && m.embedding.is_some())
        .collect();
    if items.len() < min_size {
        return Ok(());
    }
    // A burst lives within one era; don't sweep memories from different epochs
    // into the same gist (consolidate within, preserve across).
    let epoch_starts = epoch_starts(store)?;

    let sim = |a: usize, b: usize| {
        cosine_similarity(
            items[a].embedding.as_ref().unwrap(),
            items[b].embedding.as_ref().unwrap(),
        )
    };

    let mut consumed = vec![false; items.len()];
    for i in 0..items.len() {
        if consumed[i] {
            continue;
        }
        // Gather i's near-neighbors (similar, same era, not yet consumed).
        // Anything identical enough to merge is already gone (dedup ran
        // first), so a cluster here is "similar but distinct" — exactly a burst.
        let mut cluster: Vec<usize> = vec![i];
        #[allow(clippy::needless_range_loop)] // need j to index items, consumed, and sim(i, j)
        for j in 0..items.len() {
            if j == i || consumed[j] {
                continue;
            }
            if !same_epoch(&epoch_starts, items[i].created_at, items[j].created_at) {
                continue;
            }
            if sim(i, j) >= sim_thresh {
                cluster.push(j);
            }
        }
        if cluster.len() < min_size {
            continue;
        }
        // A burst is temporally *tight*: the members fall within a short slice
        // of the store's history. A cluster spread across time is a habit and
        // is left for Foundation promotion, not collapsed.
        let ages: Vec<f64> = cluster
            .iter()
            .map(|&k| age_days(items[k].created_at, now))
            .collect();
        let lo = ages.iter().cloned().fold(f64::MAX, f64::min);
        let hi = ages.iter().cloned().fold(f64::MIN, f64::max);
        if (hi - lo) / history_span_days > params.tight_spread {
            continue;
        }

        // The outlier is the member with the lowest mean similarity to the
        // rest — the part of the flood that didn't fit the pattern.
        let mean_sim = |k: usize| -> f64 {
            let others = cluster.iter().filter(|&&o| o != k);
            let (sum, n) = others.fold((0.0, 0usize), |(s, n), &o| (s + sim(k, o) as f64, n + 1));
            if n == 0 {
                1.0
            } else {
                sum / n as f64
            }
        };
        let outlier = *cluster
            .iter()
            .min_by(|&&a, &&b| mean_sim(a).total_cmp(&mean_sim(b)))
            .unwrap();
        let routine: Vec<usize> = cluster.iter().cloned().filter(|&k| k != outlier).collect();

        // Build the gist from the routine members and store it as one
        // episodic memory. The routine is then deleted — thinned, with the
        // gist as its trace — and the outlier kept and sharpened.
        let texts: Vec<&str> = routine.iter().map(|&k| items[k].content.as_str()).collect();
        let body = summarizer.summarize(&texts);
        let summary = if body.is_empty() {
            format!("Gist of {} similar events", routine.len())
        } else {
            format!("Gist of {} similar events: {body}", routine.len())
        };
        let hash = content_hash(&summary);
        if store.get_memory_by_hash(&hash)?.is_none() {
            let mut gist = MemoryItem::new(
                new_id("mem", &hash),
                summary.clone(),
                MemoryType::Episodic,
                hash,
                now,
            );
            gist.summary = Some(summary);
            gist.source = Some("consolidation".to_string());
            // Sit the gist at the burst's own moment, not "now".
            gist.created_at = routine
                .iter()
                .map(|&k| items[k].created_at)
                .max()
                .unwrap_or(now);
            gist.topic = items[outlier].topic.clone();
            gist.importance_score = 0.4;
            gist.decay_score = 0.4;
            store.insert_memory(&gist)?;
        }

        // Keep the anomaly, and bump it so it stands out now that the routine
        // around it is gone.
        let mut keep = items[outlier].clone();
        keep.importance_score = (keep.importance_score + 0.1).min(1.0);
        keep.salience = keep.salience.max(0.6);
        keep.updated_at = now;
        store.update_memory(&keep)?;

        for &k in &routine {
            store.delete_memory(&items[k].id)?;
            consumed[k] = true;
        }
        consumed[outlier] = true;
        report.bursts_collapsed += 1;
    }
    Ok(())
}

/// Refresh recurrence from V2 recurrence signals:
/// raw mentions, distinct sessions, distinct days, context diversity, and
/// accepted reuse evidence. Raw frequency is deliberately the weakest term.
fn refresh_recurrence(store: &Store, now: i64, report: &mut ConsolidationReport) -> Result<()> {
    let items = store.list_memories(None)?;
    let all_items = items.clone();
    let all_evidence = store.all_evidence()?;
    for mut item in items {
        let terms = recurrence_terms(&item);
        if terms.is_empty() {
            continue;
        }
        let mut raw_mention_count = 0usize;
        let mut sessions = HashSet::new();
        let mut days = HashSet::new();
        let mut contexts = HashSet::new();

        for other in &all_items {
            let other_terms = recurrence_terms(other);
            if !terms.iter().any(|t| other_terms.contains(t)) {
                continue;
            }
            let multiplier = recurrence_source_multiplier(other);
            raw_mention_count += multiplier;
            for tag in other.tags.iter().filter_map(|t| t.strip_prefix("session:")) {
                sessions.insert(tag.to_string());
            }
            days.insert(other.created_at / 86_400);
            for ctx in other_terms.into_iter().filter(|t| !terms.contains(t)) {
                contexts.insert(ctx);
            }
        }

        let accepted_reuse_count = all_evidence
            .iter()
            .filter(|e| e.memory_id == item.id)
            .filter(|e| {
                matches!(
                    e.evidence_type,
                    EvidenceType::RetrievalSuccess | EvidenceType::UserConfirmation
                )
            })
            .count();
        let context_diversity = (contexts.len() as f64 / 8.0).min(1.0);
        let recurrence = (0.10 * (raw_mention_count as f64).ln_1p()
            + 0.25 * (sessions.len() as f64).ln_1p()
            + 0.35 * (days.len() as f64).ln_1p()
            + 0.20 * context_diversity
            + 0.10 * (accepted_reuse_count as f64).ln_1p())
        .min(1.0);
        if (recurrence - item.recurrence_score).abs() > 0.05 && recurrence > item.recurrence_score {
            item.recurrence_score = recurrence;
            item.updated_at = now;
            store.update_memory(&item)?;
            report.recurrence_updated += 1;
        }
    }
    Ok(())
}

fn recurrence_terms(item: &MemoryItem) -> HashSet<String> {
    let mut terms = HashSet::new();
    if let Some(topic) = &item.topic {
        terms.insert(topic.to_lowercase());
    }
    for entity in &item.entities {
        terms.insert(entity.to_lowercase());
    }
    terms
}

fn recurrence_source_multiplier(item: &MemoryItem) -> usize {
    if item.source.as_deref() == Some("code_block") || item.source.as_deref() == Some("log_dump") {
        0
    } else if item.tags.iter().any(|t| t.starts_with("source_doc:")) {
        1
    } else {
        2
    }
}

fn refresh_importance(
    store: &Store,
    cfg: &Config,
    now: i64,
    report: &mut ConsolidationReport,
) -> Result<()> {
    let evidence = store.all_evidence()?;
    let mut evidence_by_memory: HashMap<String, Vec<_>> = HashMap::new();
    for ev in evidence {
        evidence_by_memory
            .entry(ev.memory_id.clone())
            .or_default()
            .push(ev);
    }
    let mut graph_support: HashMap<String, f64> = HashMap::new();
    for edge in store.list_edges()? {
        let contribution = edge.weight.max(0.0).ln_1p() / 6.0;
        for id in [edge.src_id, edge.dst_id] {
            let entry = graph_support.entry(id).or_insert(0.0);
            *entry = (*entry + contribution).min(1.0);
        }
    }

    for mut item in store.list_memories(None)? {
        if item.memory_type == MemoryType::Archive
            || item.pinned
            || item.memory_type.is_decay_exempt()
        {
            continue;
        }
        let evs = evidence_by_memory
            .get(&item.id)
            .cloned()
            .unwrap_or_default();
        let correction_count = evs
            .iter()
            .filter(|e| e.evidence_type == EvidenceType::UserCorrection)
            .count();
        let positive_evidence: f64 = evs
            .iter()
            .filter(|e| e.evidence_type != EvidenceType::UserCorrection)
            .map(|e| match e.evidence_type {
                EvidenceType::ExplicitRememberRequest => 0.30 * e.strength,
                EvidenceType::UserConfirmation | EvidenceType::RetrievalSuccess => {
                    0.25 * e.strength
                }
                EvidenceType::CrossSessionRecurrence | EvidenceType::CrossDayRecurrence => {
                    0.20 * e.strength
                }
                EvidenceType::SessionThemeSupport | EvidenceType::GraphClusterSupport => {
                    0.10 * e.strength
                }
                _ => 0.08 * e.strength,
            })
            .sum::<f64>()
            .min(0.35);
        let base = match item.memory_type {
            MemoryType::Foundation => 0.70,
            MemoryType::Preference => 0.35,
            MemoryType::Semantic | MemoryType::Procedural => 0.30,
            MemoryType::Episodic => 0.20,
            MemoryType::RawEvent => 0.10,
            MemoryType::Archive => 0.05,
        };
        let current_decay = decay::decay_score_resisted(
            item.importance_score,
            cfg.decay_lambdas().for_type(item.memory_type),
            age_days(item.created_at, now),
            item.pinned,
            item.salience,
            cfg.salience_resist,
        );
        let decay_loss = 1.0 - current_decay.clamp(0.0, 1.0);
        let correction_penalty = 0.40 * correction_count as f64;
        let revised = (base
            + positive_evidence
            + graph_support.get(&item.id).copied().unwrap_or(0.0) * 0.15
            + item.recurrence_score * 0.20
            - decay_loss * 0.25
            - correction_penalty)
            .clamp(0.05, 1.0);
        if (revised - item.importance_score).abs() > 0.05 {
            item.importance_score = revised;
            item.updated_at = now;
            store.update_memory(&item)?;
            report.importance_revised += 1;
        }
    }
    Ok(())
}

/// Summarize contiguous **temporal events** of episodic memory into semantic
/// gist memories. Replaces topic-string grouping with tier-2 surprise
/// segmentation ([`forgetfuldb_segment`]): a stretch of episodic experience
/// that hangs together in time and embedding space becomes one summary, rather
/// than every same-topic memory across all time collapsing together. Events run
/// over the survivors of `collapse_bursts` (which ran earlier and deleted its
/// members), so a burst is never summarized twice. See
/// `docs/surprise-segmentation.md` §9.
fn summarize_clusters(
    store: &Store,
    summarizer: &dyn Summarizer,
    cfg: &Config,
    now: i64,
    report: &mut ConsolidationReport,
) -> Result<()> {
    use forgetfuldb_segment::segment_with_embeddings;

    let min_size = cfg.consolidation_thresholds.cluster_min_size;
    let mut items: Vec<MemoryItem> = store
        .list_memories(Some(MemoryType::Episodic))?
        .into_iter()
        .filter(|m| !m.stale && m.embedding.is_some())
        .collect();
    items.sort_by_key(|m| m.created_at);
    retain_modal_embedding_dim(&mut items);
    if items.len() < min_size {
        return Ok(());
    }

    let embs: Vec<Vec<f32>> = items.iter().map(|m| m.embedding.clone().unwrap()).collect();
    let events = segment_with_embeddings(&embs, &cfg.segmentation, None).events;

    for ev in &events {
        let members = &items[ev.start..ev.end];
        if members.len() < min_size {
            continue;
        }
        let texts: Vec<&str> = members.iter().map(|m| m.content.as_str()).collect();
        let summary = summarizer.summarize(&texts);
        if summary.is_empty() {
            continue;
        }
        let hash = content_hash(&summary);
        // Skip if this exact summary already exists (idempotent passes).
        if store.get_memory_by_hash(&hash)?.is_some() {
            continue;
        }
        let mut item = MemoryItem::new(
            new_id("mem", &hash),
            summary.clone(),
            MemoryType::Semantic,
            hash,
            now,
        );
        item.summary = Some(summary);
        item.topic = modal_topic(members);
        item.source = Some("consolidation".to_string());
        item.importance_score = members
            .iter()
            .map(|m| m.importance_score)
            .fold(0.0, f64::max)
            .max(0.6);
        item.recurrence_score = (members.len() as f64 / 10.0).min(1.0);
        item.decay_score = item.importance_score;
        store.insert_memory(&item)?;
        for member in members {
            store.insert_link(&MemoryLink {
                source_id: item.id.clone(),
                target_id: member.id.clone(),
                relation: LinkRelation::DerivedFrom,
            })?;
        }
        report.summaries.push(forgetfuldb_store::SummaryProvenance {
            summary_id: item.id.clone(),
            source_ids: members.iter().map(|m| m.id.clone()).collect(),
        });
        report.clusters_summarized += 1;
    }
    Ok(())
}

/// The most common non-empty `topic` among an event's members (deterministic:
/// ties break alphabetically). Used to label a temporal summary now that
/// summaries are no longer grouped by a single topic string.
fn modal_topic(members: &[MemoryItem]) -> Option<String> {
    let mut topics: Vec<&str> = members.iter().filter_map(|m| m.topic.as_deref()).collect();
    if topics.is_empty() {
        return None;
    }
    topics.sort_unstable();
    let (mut best, mut best_n) = ("", 0usize);
    let mut i = 0;
    while i < topics.len() {
        let t = topics[i];
        let n = topics[i..].iter().take_while(|&&x| x == t).count();
        if n > best_n {
            best_n = n;
            best = t;
        }
        i += n;
    }
    Some(best.to_string())
}

/// Episodic memories rehearsed often enough graduate to semantic memory
/// (slower decay): repetition turns experience into knowledge.
fn promote_recurring(
    store: &Store,
    cfg: &Config,
    now: i64,
    report: &mut ConsolidationReport,
) -> Result<()> {
    let min_access = cfg.consolidation_thresholds.promote_min_access_count;
    for mut item in store.list_memories(Some(MemoryType::Episodic))? {
        if !item.stale && item.access_count >= min_access {
            item.memory_type = MemoryType::Semantic;
            item.importance_score = (item.importance_score + 0.1).min(1.0);
            item.updated_at = now;
            store.update_memory(&item)?;
            report.promoted_to_semantic += 1;
        }
    }
    Ok(())
}

/// Refine each memory's noisy single-token `topic` into a cluster-level
/// label. Memories that are similar *or* share a chat session are clustered
/// (union-find), and each cluster's members adopt its dominant signal — an
/// explicit `project:`/`topic:` tag (weighted) or the most-common entity. This
/// turns "topic = alphabetically-first word" into "topic = what this group is
/// about", which sharpens summaries, foundation promotion, and contradiction
/// candidate-gen. Explicit tags are never overwritten; deterministic
/// (vote count, then alphabetical) so labels converge instead of thrashing.
fn refine_topics(
    store: &Store,
    cfg: &Config,
    now: i64,
    report: &mut ConsolidationReport,
) -> Result<()> {
    if !cfg.consolidation_thresholds.topic_refine_enabled {
        return Ok(());
    }
    let sim_thresh = cfg.consolidation_thresholds.topic_cluster_sim as f32;

    let items: Vec<MemoryItem> = store
        .list_memories(None)?
        .into_iter()
        .filter(|m| m.memory_type != MemoryType::Archive && m.embedding.is_some())
        .collect();
    let n = items.len();
    if n < 2 {
        return Ok(());
    }

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let (ra, rb) = (find(parent, a), find(parent, b));
        if ra != rb {
            parent[ra] = rb;
        }
    }
    let mut parent: Vec<usize> = (0..n).collect();

    // Session cohesion: same chat session → same cluster.
    let mut by_session: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, m) in items.iter().enumerate() {
        for t in m.tags.iter().filter(|t| t.starts_with("session:")) {
            by_session.entry(t.as_str()).or_default().push(i);
        }
    }
    for idxs in by_session.values() {
        for w in idxs.windows(2) {
            union(&mut parent, w[0], w[1]);
        }
    }
    // Semantic cohesion.
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(
                items[i].embedding.as_ref().unwrap(),
                items[j].embedding.as_ref().unwrap(),
            );
            if sim >= sim_thresh {
                union(&mut parent, i, j);
            }
        }
    }

    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        clusters.entry(r).or_default().push(i);
    }

    let explicit_topic = |m: &MemoryItem| -> Option<String> {
        m.tags.iter().find_map(|t| {
            t.strip_prefix("project:")
                .or_else(|| t.strip_prefix("topic:"))
                .filter(|s| !s.is_empty())
                .map(String::from)
        })
    };

    for members in clusters.values() {
        if members.len() < 2 {
            continue; // a singleton keeps whatever it had
        }
        // Vote: explicit tag (weight 2) + each entity (weight 1). The label
        // shared across the cluster wins; alphabetical tie-break for stability.
        let mut votes: HashMap<String, i32> = HashMap::new();
        for &i in members {
            if let Some(t) = explicit_topic(&items[i]) {
                *votes.entry(t).or_insert(0) += 2;
            }
            for e in &items[i].entities {
                *votes.entry(e.clone()).or_insert(0) += 1;
            }
        }
        let mut tally: Vec<(String, i32)> = votes.into_iter().collect();
        tally.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let Some((canonical, _)) = tally.first() else {
            continue;
        };

        for &i in members {
            if explicit_topic(&items[i]).is_some()
                || items[i].topic.as_deref() == Some(canonical.as_str())
            {
                continue;
            }
            let mut updated = items[i].clone();
            updated.topic = Some(canonical.clone());
            updated.updated_at = now;
            store.update_memory(&updated)?;
            report.topics_refined += 1;
        }
    }
    Ok(())
}

/// The `started_at` of every stored era, ascending — the boundary list the
/// `consolidate within, preserve across` guards look memories up against.
/// Empty until the first `segment_epochs` runs, which makes the guards no-ops
/// on a fresh store (everything is one era).
fn epoch_starts(store: &Store) -> Result<Vec<i64>> {
    Ok(store.list_epochs()?.iter().map(|e| e.started_at).collect())
}

/// Whether two timestamps fall in the same era. With no epochs yet, all
/// memories are treated as one era (so consolidation behaves as before).
fn same_epoch(starts: &[i64], a: i64, b: i64) -> bool {
    use forgetfuldb_core::epochs::epoch_index_at;
    starts.is_empty() || epoch_index_at(starts, a) == epoch_index_at(starts, b)
}

/// Segment the surviving timeline into eras and persist them. The model has no
/// clock, so the engine computes the eras: tier-2 embedding-space *surprise*
/// segmentation ([`forgetfuldb_segment`]) finds the boundaries, and this
/// adapter turns its index ranges into stored `Epoch` rows — re-imposing the
/// `epoch_min_size` / `epoch_min_days` guard the surprise segmenter is (by
/// design) time-agnostic about. Each era is labeled with an extractive summary
/// of its most-salient members. See `docs/surprise-segmentation.md` §8.
fn segment_epochs(
    store: &Store,
    summarizer: &dyn Summarizer,
    cfg: &Config,
    report: &mut ConsolidationReport,
) -> Result<()> {
    use forgetfuldb_core::epochs::centroid_of;
    use forgetfuldb_segment::segment_with_embeddings;

    // Active, embedded memories in time order define the era stream. Archives
    // (de-emphasized) and unembedded rows don't shape an era's identity.
    let mut items: Vec<MemoryItem> = store
        .list_memories(None)?
        .into_iter()
        .filter(|m| m.memory_type != MemoryType::Archive && m.embedding.is_some())
        .collect();
    items.sort_by_key(|m| m.created_at);
    // Boundaries are only comparable within one embedding space (FR-8): keep a
    // single-dimension cohort so a half-finished re-embed can't mix spaces.
    retain_modal_embedding_dim(&mut items);
    if items.is_empty() {
        store.replace_epochs(&[])?;
        return Ok(());
    }

    let embs: Vec<Vec<f32>> = items.iter().map(|m| m.embedding.clone().unwrap()).collect();
    let result = segment_with_embeddings(&embs, &cfg.segmentation, None);

    // Re-impose the min-size / min-days guard (the segmenter is time-agnostic):
    // absorb any event too small or too short to stand as its own era.
    let events = merge_micro_eras(
        &result.events,
        &items,
        cfg.consolidation_thresholds.epoch_min_size,
        cfg.consolidation_thresholds.epoch_min_days,
    );

    let mut rows = Vec::with_capacity(events.len());
    for (ordinal, ev) in events.iter().enumerate() {
        let members = &items[ev.start..ev.end];
        let member_embs: Vec<Vec<f32>> = members.iter().map(|m| m.embedding.clone().unwrap()).collect();
        let started_at = members[0].created_at;
        // Exclusive end = the first memory of the next era, or open for the last.
        let ended_at = events.get(ordinal + 1).map(|next| items[next.start].created_at);
        // Drift that opened this era = the surprise at its first entry (0 for
        // the first era, which breaks from nothing).
        let drift_in = if ordinal == 0 { 0.0 } else { result.surprise[ev.start] };

        // Label/summary from the era's most-salient members — the gist of what
        // that stretch of time was about.
        let mut top: Vec<&MemoryItem> = members.iter().collect();
        top.sort_by(|a, b| b.salience.total_cmp(&a.salience));
        let texts: Vec<&str> = top.iter().take(5).map(|m| m.content.as_str()).collect();
        let summary = summarizer.summarize(&texts);
        rows.push(forgetfuldb_store::Epoch {
            id: new_id("epoch", &format!("{}-{}", ordinal, started_at)),
            ordinal: ordinal as i64,
            started_at,
            ended_at,
            centroid: Some(centroid_of(&member_embs)),
            label: Some(format!("era {}", ordinal + 1)),
            summary: (!summary.is_empty()).then_some(summary),
            member_count: members.len() as i64,
            drift_in,
        });
    }
    store.replace_epochs(&rows)?;
    report.epochs = rows.len();
    Ok(())
}

/// Keep only memories whose embedding has the most common dimensionality, so a
/// partially-migrated store (mixed embedding models/dims) can't feed
/// incomparable vectors into segmentation. Deterministic: ties break toward the
/// larger dimension.
fn retain_modal_embedding_dim(items: &mut Vec<MemoryItem>) {
    let mut dims: Vec<usize> = items.iter().filter_map(|m| m.embedding.as_ref().map(|e| e.len())).collect();
    if dims.is_empty() {
        return;
    }
    dims.sort_unstable();
    // Modal dim: longest run in the sorted list; tie → larger dim.
    let (mut best_dim, mut best_count) = (dims[0], 0usize);
    let mut i = 0;
    while i < dims.len() {
        let d = dims[i];
        let j = dims[i..].iter().take_while(|&&x| x == d).count();
        if j > best_count || (j == best_count && d > best_dim) {
            best_count = j;
            best_dim = d;
        }
        i += j;
    }
    items.retain(|m| m.embedding.as_ref().map(|e| e.len()) == Some(best_dim));
}

/// Absorb events too small (`< min_size` members) or too short (`< min_days`
/// span) to stand as their own era into the following span, so the surprise
/// segmenter's fine boundaries don't create micro-eras. The final (open) era is
/// always kept even if small. Preserves contiguous `0..n` coverage.
fn merge_micro_eras(
    events: &[forgetfuldb_segment::Event],
    items: &[MemoryItem],
    min_size: usize,
    min_days: f64,
) -> Vec<forgetfuldb_segment::Event> {
    use forgetfuldb_segment::Event;
    if events.is_empty() {
        return Vec::new();
    }
    let stands_alone = |ev: &Event| {
        let count = ev.end - ev.start;
        let days = age_days(items[ev.start].created_at, items[ev.end - 1].created_at);
        count >= min_size && days >= min_days
    };
    let mut merged: Vec<Event> = Vec::new();
    let mut cur = events[0];
    for ev in &events[1..] {
        if stands_alone(&cur) {
            merged.push(cur);
            cur = *ev;
        } else {
            cur.end = ev.end; // absorb the too-small era forward
        }
    }
    merged.push(cur);
    merged
}

/// Same-subject precision filter for contradiction candidacy: two memories
/// share a topic or an entity (embedding closeness alone can be coincidental).
fn same_subject(a: &MemoryItem, b: &MemoryItem) -> bool {
    (a.topic.is_some() && a.topic == b.topic) || a.entities.iter().any(|e| b.entities.contains(e))
}

/// Infer supersession from similar memories — the staleness attack. Candidate
/// pairs (cosine in the band below dedup, sharing a topic or entity) are
/// clustered; within each cluster the newest memory is the potential winner,
/// and an older member is superseded when the [`contradiction`] core judges it
/// so with enough confidence (a correction cue, or a singular-slot value change
/// backed by replacement-over-time). Writes an `Updates` edge winner→loser;
/// `mark_contradicted_stale` (next step) stales the loser. Reversible, opt-in,
/// and silent when unsure — false negatives are safe, false positives are not.
fn infer_contradictions(
    store: &Store,
    cfg: &Config,
    report: &mut ConsolidationReport,
) -> Result<()> {
    use forgetfuldb_core::contradiction::{classify_cardinality, judge, value_tokens};

    if !cfg.contradiction.enabled {
        return Ok(());
    }
    let lo = cfg.contradiction.candidate_min_sim as f32;
    let hi = cfg.consolidation_thresholds.duplicate_similarity as f32;
    let conf_threshold = cfg.contradiction.confidence_threshold;

    let items: Vec<MemoryItem> = store
        .list_memories(None)?
        .into_iter()
        .filter(|m| m.memory_type != MemoryType::Archive && !m.stale && m.embedding.is_some())
        .collect();
    let n = items.len();
    if n < 2 {
        return Ok(());
    }

    // Cluster candidate pairs (cosine in band ∧ same subject) by union-find.
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut parent: Vec<usize> = (0..n).collect();
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(
                items[i].embedding.as_ref().unwrap(),
                items[j].embedding.as_ref().unwrap(),
            );
            if sim >= lo && sim < hi && same_subject(&items[i], &items[j]) {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        clusters.entry(r).or_default().push(i);
    }

    for members in clusters.values() {
        if members.len() < 2 {
            continue;
        }
        // Cardinality from how the slot's distinct values sit in time:
        // grouped by value signature, replacement (sequential) vs accumulation.
        let mut spans: HashMap<Vec<String>, (i64, i64)> = HashMap::new();
        for &i in members {
            let sig = value_tokens(&items[i].content);
            let e = spans
                .entry(sig)
                .or_insert((items[i].created_at, items[i].created_at));
            e.0 = e.0.min(items[i].created_at);
            e.1 = e.1.max(items[i].created_at);
        }
        let cardinality = classify_cardinality(&spans.values().copied().collect::<Vec<_>>(), 0.2);

        // The newest member is the candidate winner; older differing members
        // are superseded when the verdict clears the confidence bar.
        let winner = *members
            .iter()
            .max_by_key(|&&i| items[i].created_at)
            .unwrap();
        for &i in members {
            if i == winner || items[i].created_at > items[winner].created_at {
                continue;
            }
            if let Some(v) = judge(&items[i], &items[winner], cardinality) {
                if v.confidence >= conf_threshold {
                    store.insert_link(&MemoryLink {
                        source_id: v.winner_id,
                        target_id: v.loser_id,
                        relation: LinkRelation::Updates,
                    })?;
                    report.contradictions_inferred += 1;
                }
            }
        }
    }
    Ok(())
}

/// Reversibility: a memory staled by a supersession edge is **revived** when
/// its value is reasserted as the *current* one — the newest live, same-subject
/// memory in its slot asserts the same value again ("actually, back to
/// Postgres"). This is the self-heal that makes an over-eager staling safe; it
/// also removes the supersession edge so the next pass doesn't re-stale it.
fn revive_reasserted(store: &Store, cfg: &Config, report: &mut ConsolidationReport) -> Result<()> {
    use forgetfuldb_core::contradiction::{correction_cue, value_tokens};

    // A memory's *current* asserted value: the cue's "new" target if it's a
    // correction ("from Postgres to SQLite" → SQLite), else its value tokens.
    // Without this, a migration statement that names the old value would look
    // like it reasserts it.
    let effective_value = |text: &str| -> Vec<String> {
        match correction_cue(text).and_then(|c| c.new) {
            Some(n) => vec![n.to_lowercase()],
            None => value_tokens(text),
        }
    };

    if !cfg.contradiction.enabled {
        return Ok(());
    }
    let lo = cfg.contradiction.candidate_min_sim as f32;

    let all: Vec<MemoryItem> = store
        .list_memories(None)?
        .into_iter()
        .filter(|m| m.memory_type != MemoryType::Archive && m.embedding.is_some())
        .collect();

    // Memories staled *by a supersession edge* (vs. some other reason).
    let staled_targets: std::collections::HashSet<String> = store
        .all_links()?
        .into_iter()
        .filter(|l| {
            matches!(
                l.relation,
                LinkRelation::Updates | LinkRelation::Contradicts
            )
        })
        .map(|l| l.target_id)
        .collect();

    for s in all
        .iter()
        .filter(|m| m.stale && staled_targets.contains(&m.id))
    {
        let sv = value_tokens(&s.content);
        if sv.is_empty() {
            continue; // can't confirm reassertion of a non-value-like value
        }
        let s_emb = s.embedding.as_ref().unwrap();
        // The current truth in this slot: the newest live, same-subject memory.
        let newest_live = all
            .iter()
            .filter(|m| !m.stale && m.id != s.id && m.created_at >= s.created_at)
            .filter(|m| same_subject(s, m))
            .filter(|m| cosine_similarity(s_emb, m.embedding.as_ref().unwrap()) >= lo)
            .max_by_key(|m| m.created_at);

        if let Some(w) = newest_live {
            let wv = effective_value(&w.content);
            // Revive only if that current memory reasserts s's value.
            if sv.iter().all(|t| wv.contains(t)) {
                store.set_stale(&s.id, false)?;
                store.clear_supersession_links_to(&s.id)?;
                report.revived += 1;
            }
        }
    }
    Ok(())
}

/// Any memory that is the target of a `contradicts` or `updates` link is
/// out of date by definition — mark it stale (kept, but hidden from
/// retrieval unless explicitly requested).
fn mark_contradicted_stale(store: &Store, report: &mut ConsolidationReport) -> Result<()> {
    for link in store.all_links()? {
        if matches!(
            link.relation,
            LinkRelation::Contradicts | LinkRelation::Updates
        ) {
            if let Some(target) = store.get_memory(&link.target_id)? {
                if !target.stale {
                    store.set_stale(&target.id, true)?;
                    report.marked_stale += 1;
                }
            }
        }
    }
    Ok(())
}

/// Forgetting proper: old, decayed, unpinned raw-event memories become
/// archives; long-archived ones are deleted. A reservoir sample of pruned
/// raw events survives as a single archive note, so the past is thinned,
/// not erased without trace.
fn archive_and_prune(
    store: &Store,
    cfg: &Config,
    now: i64,
    report: &mut ConsolidationReport,
) -> Result<()> {
    let lambdas = cfg.decay_lambdas();
    let archive_cutoff_days = cfg.archive_after_days;
    let delete_cutoff_days = cfg.delete_after_days;
    let max_decay = cfg.consolidation_thresholds.archive_max_decay;

    for item in store.list_memories(None)? {
        if item.pinned || item.memory_type.is_decay_exempt() {
            continue; // pins and Foundation traits never decay or get pruned
        }
        if item.salience >= cfg.salience_keep_threshold {
            continue; // formative (high-salience) memories are kept, like pins
        }
        let age = age_days(item.created_at, now);
        // Salience-resisted so a formative memory survives the pruning that
        // buries the routine around it.
        let current_decay = decay::decay_score_resisted(
            item.importance_score,
            lambdas.for_type(item.memory_type),
            age,
            item.pinned,
            item.salience,
            cfg.salience_resist,
        );
        match item.memory_type {
            MemoryType::RawEvent if age > archive_cutoff_days && current_decay < max_decay => {
                store.set_memory_type(&item.id, MemoryType::Archive)?;
                report.archived += 1;
            }
            MemoryType::Archive if age > delete_cutoff_days => {
                store.delete_memory(&item.id)?;
                report.deleted += 1;
            }
            _ => {}
        }
    }

    // Prune verbatim raw_events older than the delete window, keeping a
    // uniform reservoir sample as a representative trace.
    let cutoff = now - (delete_cutoff_days * 86_400.0) as i64;
    let old_events = store.raw_events_older_than(cutoff)?;
    if !old_events.is_empty() {
        let mut reservoir = ReservoirSampler::new(cfg.consolidation_thresholds.prune_sample_size);
        for ev in &old_events {
            reservoir.add(ev.content.clone());
        }
        let sample = reservoir.into_items();
        let note = format!(
            "Representative sample of {} pruned raw events: {}",
            old_events.len(),
            sample.join(" | ")
        );
        let hash = content_hash(&note);
        if store.get_memory_by_hash(&hash)?.is_none() {
            let mut item =
                MemoryItem::new(new_id("mem", &hash), note, MemoryType::Archive, hash, now);
            item.source = Some("consolidation".to_string());
            item.importance_score = 0.1;
            item.decay_score = 0.1;
            store.insert_memory(&item)?;
        }
        for ev in &old_events {
            store.delete_raw_event(&ev.id)?;
            report.deleted += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use forgetfuldb_embed::EmbeddingProvider;
    use forgetfuldb_store::pipeline::{ingest, warm_bloom, IngestRequest};

    fn setup() -> (Store, Box<dyn EmbeddingProvider>, Config) {
        let store = Store::open_in_memory().unwrap();
        let provider = forgetfuldb_embed::create_provider("hashed_bow", 128).unwrap();
        (store, provider, Config::default())
    }

    fn add(
        store: &Store,
        provider: &dyn EmbeddingProvider,
        cfg: &Config,
        text: &str,
        mt: MemoryType,
    ) -> String {
        let mut bloom = warm_bloom(store).unwrap();
        ingest(
            store,
            &mut bloom,
            provider,
            cfg,
            IngestRequest {
                text: text.to_string(),
                source: None,
                tags: vec!["project:test".to_string()],
                memory_type: Some(mt),
                session_id: None,
                role: None,
            },
        )
        .unwrap()
        .memory()
        .id
        .clone()
    }

    /// Ingest with a `session:<id>` tag (and no `project:` tag), so the
    /// memory's topic is the noisy guessed one — what topic-refinement fixes.
    fn add_session(
        store: &Store,
        provider: &dyn EmbeddingProvider,
        cfg: &Config,
        text: &str,
        session: &str,
    ) -> String {
        let mut bloom = warm_bloom(store).unwrap();
        ingest(
            store,
            &mut bloom,
            provider,
            cfg,
            IngestRequest {
                text: text.to_string(),
                source: None,
                tags: vec![format!("session:{session}")],
                memory_type: Some(MemoryType::Episodic),
                session_id: Some(session.to_string()),
                role: None,
            },
        )
        .unwrap()
        .memory()
        .id
        .clone()
    }

    /// Push a memory's creation back in time (created_at is set to "now" at
    /// ingest; behavior tests need aged memories).
    fn backdate(store: &Store, id: &str, days: i64) {
        let mut m = store.get_memory(id).unwrap().unwrap();
        m.created_at = now_unix() - days * 86_400;
        store.update_memory(&m).unwrap();
    }

    fn salience_of(store: &Store, id: &str) -> f64 {
        store.get_memory(id).unwrap().unwrap().salience
    }

    /// Insert a memory with a hand-crafted embedding and creation time —
    /// lets epoch tests control the embedding-space geometry exactly, rather
    /// than depending on hashed_bow's fuzzy lexical cosine.
    fn insert_emb(
        store: &Store,
        text: &str,
        embedding: Vec<f32>,
        mt: MemoryType,
        days_ago: i64,
    ) -> String {
        let now = now_unix();
        let hash = content_hash(text);
        let mut m = MemoryItem::new(
            new_id("mem", &hash),
            text.to_string(),
            mt,
            hash,
            now - days_ago * 86_400,
        );
        m.embedding = Some(embedding);
        m.importance_score = 0.6;
        store.insert_memory(&m).unwrap();
        m.id
    }

    /// A 2-D unit vector at `deg` degrees — a point on a topic circle, so the
    /// angular gap between two memories is exactly their cosine distance.
    fn unit2(deg: f64) -> Vec<f32> {
        let r = deg.to_radians();
        vec![r.cos() as f32, r.sin() as f32]
    }

    // ---- Eval Layer 1: behavior tests (each isolates one mechanism) ----

    #[test]
    fn eval_surprise_a_novel_memory_outscores_routine_on_salience() {
        let (store, provider, mut cfg) = setup();
        // Isolate the salience mechanism: disable dedup-merging *and*
        // burst-collapse so the routine cluster stays intact (either one
        // would erase the "many similar neighbors" signal this test is about).
        cfg.consolidation_thresholds.duplicate_similarity = 0.999;
        cfg.consolidation_thresholds.burst_collapse_enabled = false;
        // A cluster of mutually-similar routine memories (distinct trailing
        // word so they don't collapse to one identical token set)...
        for room in ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"] {
            add(
                &store,
                provider.as_ref(),
                &cfg,
                &format!("the daily standup meeting was held at nine in room {room}"),
                MemoryType::Episodic,
            );
        }
        // ...and one genuinely novel memory (disjoint vocabulary).
        let anomaly = add(
            &store,
            provider.as_ref(),
            &cfg,
            "a burst water pipe flooded the basement archive overnight",
            MemoryType::Episodic,
        );

        consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();

        // Compare against the most-salient surviving routine memory (some may
        // have merged) — the novel one must out-salience all of them. The
        // behavioral claim is relative: absolute values float with the
        // embedding backend (hashed_bow has a collision floor).
        let anomaly_sal = salience_of(&store, &anomaly);
        let routine_max = store
            .list_memories(None)
            .unwrap()
            .iter()
            .filter(|m| m.id != anomaly && m.source.as_deref() != Some("consolidation"))
            .map(|m| m.salience)
            .fold(0.0_f64, f64::max);
        assert!(
            anomaly_sal > routine_max + 0.1,
            "the novel memory should be clearly more salient: anomaly {anomaly_sal:.3} vs best routine {routine_max:.3}"
        );
    }

    #[test]
    fn eval_selective_forgetting_anomaly_survives_routine_is_archived() {
        let (store, provider, cfg) = setup();
        // 6 routine raw events + 1 anomaly, all aged past the archive window.
        let mut routine = Vec::new();
        for d in 0..6 {
            let id = add(
                &store,
                provider.as_ref(),
                &cfg,
                &format!("logged the routine nightly backup job run number {d} completed ok"),
                MemoryType::RawEvent,
            );
            backdate(&store, &id, 30);
            routine.push(id);
        }
        let anomaly = add(
            &store,
            provider.as_ref(),
            &cfg,
            "the production database was accidentally dropped during the migration incident",
            MemoryType::RawEvent,
        );
        backdate(&store, &anomaly, 30);

        consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();

        // The anomaly is formative — kept as a raw event, not archived.
        let kept = store.get_memory(&anomaly).unwrap().unwrap();
        assert_eq!(
            kept.memory_type,
            MemoryType::RawEvent,
            "the anomaly must survive (salience {:.3})",
            kept.salience
        );
        assert!(
            kept.salience >= cfg.salience_keep_threshold,
            "anomaly salience {:.3} should clear the keep bar",
            kept.salience
        );

        // The routine decayed into the archive (forgotten as distinct events).
        let archived = routine
            .iter()
            .filter(|id| {
                store
                    .get_memory(id)
                    .unwrap()
                    .map(|m| m.memory_type == MemoryType::Archive)
                    .unwrap_or(true) // pruned entirely also counts as forgotten
            })
            .count();
        assert!(
            archived >= 4,
            "most routine memories should be archived/forgotten, got {archived}/6"
        );
    }

    #[test]
    fn eval_habit_concludes_a_foundation_trait() {
        let (store, provider, mut cfg) = setup();
        // Keep the recurring memories distinct so dedup-merge doesn't collapse
        // the cluster before the habit can be observed.
        cfg.consolidation_thresholds.duplicate_similarity = 0.999;

        // One trait expressed many times, evenly across a long history (the
        // user keeps starting games) — a habit, not a one-off burst.
        let mut ids = Vec::new();
        for (day, when) in [
            (100, "monday"),
            (80, "tuesday"),
            (60, "wednesday"),
            (40, "thursday"),
            (20, "friday"),
            (1, "weekend"),
        ] {
            let id = add(
                &store,
                provider.as_ref(),
                &cfg,
                &format!("the user enjoys starting a game of tic tac toe on {when}"),
                MemoryType::Semantic,
            );
            backdate(&store, &id, day);
            ids.push(id);
        }

        consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();

        // The habit graduates to exactly one decay-exempt Foundation trait;
        // the rest of the cluster stays put (collapse-to-a-single-trait).
        let foundations: Vec<_> = ids
            .iter()
            .filter(|id| {
                store
                    .get_memory(id)
                    .unwrap()
                    .map(|m| m.memory_type == MemoryType::Foundation)
                    .unwrap_or(false)
            })
            .collect();
        assert_eq!(
            foundations.len(),
            1,
            "a long-standing habit should mint exactly one Foundation trait, got {}",
            foundations.len()
        );

        // And a Foundation trait is decay-exempt: it survives a prune pass
        // unconditionally, like a pin.
        let lambdas = cfg.decay_lambdas();
        assert_eq!(
            lambdas.for_type(MemoryType::Foundation),
            0.0,
            "Foundation must not decay"
        );
    }

    #[test]
    fn eval_burst_collapses_to_gist_keeping_the_anomaly() {
        let (store, provider, cfg) = setup();
        // A tight burst of similar-but-distinct routine events. They share a
        // long common prefix (so they cluster) but differ in one word (so
        // they're not exact duplicates the dedup pass would merge first).
        let mut routine = Vec::new();
        for topic in ["roadmap", "staffing", "budget", "tooling", "metrics"] {
            let id = add(
                &store,
                provider.as_ref(),
                &cfg,
                &format!("the weekly leadership sync covered the usual updates and discussed the {topic} plan"),
                MemoryType::Episodic,
            );
            routine.push(id);
        }
        // One anomaly in the same burst: shares the opening so it joins the
        // cluster, but it's the member least like the rest — the one to keep.
        let anomaly = add(
            &store,
            provider.as_ref(),
            &cfg,
            "the weekly leadership sync covered the usual updates until a major outage derailed everything",
            MemoryType::Episodic,
        );

        let report = consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
        let surviving_routine: Vec<_> = routine
            .iter()
            .filter(|id| store.get_memory(id).unwrap().is_some())
            .cloned()
            .collect();
        let gist = store
            .list_memories(None)
            .unwrap()
            .into_iter()
            .find(|m| m.content.starts_with("Gist of"));

        assert_eq!(
            report.bursts_collapsed, 1,
            "the burst should collapse exactly once"
        );
        assert!(
            store.get_memory(&anomaly).unwrap().is_some(),
            "the anomaly must survive the collapse"
        );
        assert!(
            surviving_routine.is_empty(),
            "the routine should be gone, replaced by the gist"
        );
        assert!(
            gist.is_some(),
            "a gist memory should capture the collapsed routine"
        );
    }

    #[test]
    fn eval_two_topic_stream_segments_into_two_eras() {
        let (store, _provider, mut cfg) = setup();
        cfg.consolidation_thresholds.burst_collapse_enabled = false; // keep the stream intact
                                                                     // Tight on-topic points sit within the 0.92 dup angle; only collapse
                                                                     // true duplicates so the six-per-era stream survives to be segmented.
        cfg.consolidation_thresholds.duplicate_similarity = 0.999;
        // The surprise segmenter needs `window_size` entries of warm-up before
        // the first detectable boundary (FR-3). This 12-note fixture puts the
        // A→B shift at index 6, so the default window of 8 would never see it —
        // a smaller window suits the miniature stream (real streams are larger).
        cfg.segmentation.window_size = 3;

        // Era A: six notes clustered near axis 0° (mutually similar but none a
        // duplicate), spread over ~10 days starting ~40 days ago.
        for (k, deg) in [0.0, 6.0, 12.0, 18.0, 24.0, 30.0].into_iter().enumerate() {
            insert_emb(
                &store,
                &format!("topic a note {k}"),
                unit2(deg),
                MemoryType::Semantic,
                40 - k as i64 * 2,
            );
        }
        // Era B: six notes near axis 90° — a clean rotation away — ~10 days later.
        for (k, deg) in [90.0, 96.0, 102.0, 108.0, 114.0, 120.0]
            .into_iter()
            .enumerate()
        {
            insert_emb(
                &store,
                &format!("topic b note {k}"),
                unit2(deg),
                MemoryType::Semantic,
                20 - k as i64 * 2,
            );
        }

        let report = consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
        assert_eq!(report.epochs, 2, "report should record two eras");

        let epochs = store.list_epochs().unwrap();
        assert_eq!(
            epochs.len(),
            2,
            "a clean topic rotation over time should yield two eras, got {}",
            epochs.len()
        );
        assert_eq!(epochs[0].member_count, 6, "era A holds its six members");
        assert!(epochs[0].ended_at.is_some(), "the first era is closed");
        assert_eq!(epochs[1].ended_at, None, "the latest era is open");
        assert!(
            epochs[1].drift_in > 0.3,
            "era 2 opened on real drift: {}",
            epochs[1].drift_in
        );
    }

    #[test]
    fn eval_consolidation_preserves_near_duplicates_across_epochs() {
        // Identical embedding, different text → cosine 1.0, normally merged.
        let emb = vec![1.0_f32, 0.0];

        // Control — same era: the two collapse to one (the merge still works).
        {
            let (store, _p, mut cfg) = setup();
            cfg.consolidation_thresholds.burst_collapse_enabled = false;
            let a = insert_emb(
                &store,
                "near dup alpha",
                emb.clone(),
                MemoryType::Semantic,
                30,
            );
            let b = insert_emb(
                &store,
                "near dup bravo",
                emb.clone(),
                MemoryType::Semantic,
                28,
            );
            consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
            let surviving = [&a, &b]
                .iter()
                .filter(|id| store.get_memory(id).unwrap().is_some())
                .count();
            assert_eq!(surviving, 1, "near-duplicates in one era should merge");
        }

        // Across two eras: the preserve-across guard keeps both.
        {
            let (store, _p, mut cfg) = setup();
            cfg.consolidation_thresholds.burst_collapse_enabled = false;
            let now = now_unix();
            let a = insert_emb(
                &store,
                "near dup alpha",
                emb.clone(),
                MemoryType::Semantic,
                30,
            );
            let b = insert_emb(
                &store,
                "near dup bravo",
                emb.clone(),
                MemoryType::Semantic,
                10,
            );
            // Two eras with a boundary at day 20: a (day 30) and b (day 10)
            // fall on opposite sides.
            store
                .replace_epochs(&[
                    forgetfuldb_store::Epoch {
                        id: "ep0".into(),
                        ordinal: 0,
                        started_at: now - 40 * 86_400,
                        ended_at: Some(now - 20 * 86_400),
                        centroid: None,
                        label: None,
                        summary: None,
                        member_count: 1,
                        drift_in: 0.0,
                    },
                    forgetfuldb_store::Epoch {
                        id: "ep1".into(),
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
            consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
            assert!(
                store.get_memory(&a).unwrap().is_some() && store.get_memory(&b).unwrap().is_some(),
                "near-duplicates in different eras must both survive"
            );
        }
    }

    #[test]
    fn refine_topics_converges_a_session_to_one_topic() {
        let (store, provider, mut cfg) = setup();
        cfg.consolidation_thresholds.burst_collapse_enabled = false; // keep the three intact
                                                                     // Same session, all about "payments", but guess_topic gives each a
                                                                     // *different* alphabetically-first word (alpha / complete / dashboard).
        let ids: Vec<String> = [
            "alpha release shipped for payments",
            "complete payments gateway integration",
            "dashboard for payments needs work",
        ]
        .iter()
        .map(|t| add_session(&store, provider.as_ref(), &cfg, t, "s1"))
        .collect();

        consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();

        let topics: Vec<String> = ids
            .iter()
            .map(|id| {
                store
                    .get_memory(id)
                    .unwrap()
                    .unwrap()
                    .topic
                    .unwrap_or_default()
            })
            .collect();
        assert!(
            topics.iter().all(|t| t == "payments"),
            "session members converge on the shared entity, not their noisy first word: {topics:?}"
        );
    }

    #[test]
    fn refine_topics_keeps_explicit_tags_and_is_stable() {
        let (store, provider, mut cfg) = setup();
        cfg.consolidation_thresholds.burst_collapse_enabled = false;
        // `add` tags everything project:test → explicit topic "test".
        let a = add(
            &store,
            provider.as_ref(),
            &cfg,
            "the billing invoice flow needs a review",
            MemoryType::Semantic,
        );
        add(
            &store,
            provider.as_ref(),
            &cfg,
            "billing invoice formatting bug was fixed",
            MemoryType::Semantic,
        );

        consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
        assert_eq!(
            store.get_memory(&a).unwrap().unwrap().topic.as_deref(),
            Some("test"),
            "an explicit project tag is never overwritten"
        );

        // Converged → a second pass refines nothing new (stable, no thrash).
        let second = consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
        assert_eq!(
            second.topics_refined, 0,
            "topics have converged; no churn on re-run"
        );
    }

    #[test]
    fn infer_contradiction_stales_the_superseded_memory() {
        let (store, provider, mut cfg) = setup();
        cfg.contradiction.enabled = true;
        // The 0.80 default band floor is tuned for real embeddings; hashed_bow
        // (lexical) runs lower, so lower the floor for the test (gotcha #5).
        cfg.contradiction.candidate_min_sim = 0.5;
        let old = add(
            &store,
            provider.as_ref(),
            &cfg,
            "the main database runs on Postgres",
            MemoryType::Semantic,
        );
        let new = add(
            &store,
            provider.as_ref(),
            &cfg,
            "the main database migrated from Postgres to SQLite",
            MemoryType::Semantic,
        );
        backdate(&store, &old, 10);

        let report = consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
        assert!(
            report.contradictions_inferred >= 1,
            "the migration should be inferred as a supersession"
        );
        assert!(
            store.get_memory(&old).unwrap().unwrap().stale,
            "the old Postgres memory is staled"
        );
        assert!(
            !store.get_memory(&new).unwrap().unwrap().stale,
            "the new memory survives"
        );
    }

    #[test]
    fn coexisting_preferences_are_not_staled() {
        let (store, provider, mut cfg) = setup();
        cfg.contradiction.enabled = true;
        cfg.contradiction.candidate_min_sim = 0.5; // hashed_bow range (see above)
        cfg.consolidation_thresholds.duplicate_similarity = 0.999; // keep both (don't merge)
        let coffee = add(
            &store,
            provider.as_ref(),
            &cfg,
            "I really prefer drinking coffee in the morning",
            MemoryType::Preference,
        );
        let tea = add(
            &store,
            provider.as_ref(),
            &cfg,
            "I really prefer drinking tea in the morning",
            MemoryType::Preference,
        );
        backdate(&store, &coffee, 10);

        let report = consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
        assert_eq!(
            report.contradictions_inferred, 0,
            "coffee and tea coexist — no cue, no replacement"
        );
        assert!(!store.get_memory(&coffee).unwrap().unwrap().stale);
        assert!(!store.get_memory(&tea).unwrap().unwrap().stale);
    }

    #[test]
    fn reasserting_a_value_revives_the_staled_memory() {
        let (store, provider, mut cfg) = setup();
        cfg.contradiction.enabled = true;
        cfg.contradiction.candidate_min_sim = 0.5;
        // A: Postgres. B: migrated to SQLite → stales A.
        let a = add(
            &store,
            provider.as_ref(),
            &cfg,
            "the main database runs on Postgres",
            MemoryType::Semantic,
        );
        let b = add(
            &store,
            provider.as_ref(),
            &cfg,
            "the main database migrated from Postgres to SQLite",
            MemoryType::Semantic,
        );
        backdate(&store, &a, 30);
        backdate(&store, &b, 20);
        consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
        assert!(
            store.get_memory(&a).unwrap().unwrap().stale,
            "A is staled by the migration"
        );

        // C: moved back to Postgres (newest) → supersedes B and reasserts A.
        let c = add(
            &store,
            provider.as_ref(),
            &cfg,
            "the main database moved back from SQLite to Postgres",
            MemoryType::Semantic,
        );
        backdate(&store, &c, 5);
        let report = consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();

        assert!(report.revived >= 1, "A should be revived");
        assert!(
            !store.get_memory(&a).unwrap().unwrap().stale,
            "A revived — Postgres is current again"
        );
        assert!(
            store.get_memory(&b).unwrap().unwrap().stale,
            "B (SQLite) is now the superseded one"
        );
    }

    #[test]
    fn contradiction_is_off_by_default() {
        let (store, provider, cfg) = setup(); // enabled = false
        let old = add(
            &store,
            provider.as_ref(),
            &cfg,
            "the main database runs on Postgres",
            MemoryType::Semantic,
        );
        add(
            &store,
            provider.as_ref(),
            &cfg,
            "the main database migrated from Postgres to SQLite",
            MemoryType::Semantic,
        );
        backdate(&store, &old, 10);
        let report = consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
        assert_eq!(
            report.contradictions_inferred, 0,
            "opt-in: nothing inferred unless enabled"
        );
        assert!(!store.get_memory(&old).unwrap().unwrap().stale);
    }

    #[test]
    fn merge_pair_combines_history() {
        let now = now_unix();
        let mut a = MemoryItem::new(
            "a".into(),
            "fact".into(),
            MemoryType::Episodic,
            "h1".into(),
            now - 100,
        );
        let mut b = MemoryItem::new(
            "b".into(),
            "fact again".into(),
            MemoryType::Episodic,
            "h2".into(),
            now,
        );
        a.access_count = 3;
        a.importance_score = 0.5;
        a.tags = vec!["x".into()];
        b.access_count = 2;
        b.importance_score = 0.7;
        b.pinned = true;
        b.tags = vec!["y".into()];

        let merged = merge_pair(a.clone(), &b, now);
        assert_eq!(merged.access_count, 5);
        assert_eq!(merged.importance_score, 0.7);
        assert!(merged.pinned);
        assert_eq!(merged.created_at, now - 100);
        assert!(merged.tags.contains(&"x".to_string()) && merged.tags.contains(&"y".to_string()));
        assert!(merged.recurrence_score > 0.0);
    }

    #[test]
    fn near_duplicates_are_merged() {
        let (store, provider, cfg) = setup();
        // Same token multiset -> cosine 1.0 under hashed bag-of-words,
        // but different canonical hashes so both rows insert.
        add(
            &store,
            provider.as_ref(),
            &cfg,
            "stripe handles plot perfect billing invoices",
            MemoryType::Episodic,
        );
        add(
            &store,
            provider.as_ref(),
            &cfg,
            "billing invoices plot perfect handles stripe",
            MemoryType::Episodic,
        );
        add(
            &store,
            provider.as_ref(),
            &cfg,
            "granite countertops were measured on tuesday",
            MemoryType::Episodic,
        );

        let report = consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
        assert_eq!(report.duplicates_merged, 1);
        // 3 ingested - 1 merged away (no clusters of 3+ on same topic remain)
        let stats = store.stats().unwrap();
        assert_eq!(
            stats
                .by_type
                .iter()
                .find(|(t, _)| t == "episodic")
                .unwrap()
                .1,
            2
        );
    }

    #[test]
    fn contradicted_memory_marked_stale() {
        let (store, provider, cfg) = setup();
        let old = add(
            &store,
            provider.as_ref(),
            &cfg,
            "the office is in building seven",
            MemoryType::Semantic,
        );
        let new = add(
            &store,
            provider.as_ref(),
            &cfg,
            "the office moved to building twelve",
            MemoryType::Semantic,
        );
        store
            .insert_link(&MemoryLink {
                source_id: new,
                target_id: old.clone(),
                relation: LinkRelation::Updates,
            })
            .unwrap();

        let report = consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
        assert_eq!(report.marked_stale, 1);
        assert!(store.get_memory(&old).unwrap().unwrap().stale);
    }

    #[test]
    fn frequently_accessed_episodic_promoted_to_semantic() {
        let (store, provider, cfg) = setup();
        let id = add(
            &store,
            provider.as_ref(),
            &cfg,
            "deploys happen from the main branch",
            MemoryType::Episodic,
        );
        for _ in 0..3 {
            store.touch_memory(&id, now_unix()).unwrap();
        }
        let report = consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
        assert_eq!(report.promoted_to_semantic, 1);
        assert_eq!(
            store.get_memory(&id).unwrap().unwrap().memory_type,
            MemoryType::Semantic
        );
    }

    #[test]
    fn topic_clusters_get_summarized() {
        let (store, provider, cfg) = setup();
        add(
            &store,
            provider.as_ref(),
            &cfg,
            "discussed plot perfect billing edge cases for refunds",
            MemoryType::Episodic,
        );
        add(
            &store,
            provider.as_ref(),
            &cfg,
            "plot perfect customers complained about invoice formatting",
            MemoryType::Episodic,
        );
        add(
            &store,
            provider.as_ref(),
            &cfg,
            "decided plot perfect will switch to usage based pricing",
            MemoryType::Episodic,
        );

        let report = consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
        assert!(report.clusters_summarized >= 1);
        let semantic = store.list_memories(Some(MemoryType::Semantic)).unwrap();
        assert!(semantic
            .iter()
            .any(|m| m.source.as_deref() == Some("consolidation")));
    }

    #[test]
    fn pinned_memories_survive_pruning() {
        let (store, provider, cfg) = setup();
        let id = add(
            &store,
            provider.as_ref(),
            &cfg,
            "never delete this pinned reminder",
            MemoryType::RawEvent,
        );
        store.set_pinned(&id, true).unwrap();
        // Backdate far past every cutoff.
        let mut item = store.get_memory(&id).unwrap().unwrap();
        item.created_at -= 200 * 86_400;
        store.update_memory(&item).unwrap();

        consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
        let survivor = store.get_memory(&id).unwrap().unwrap();
        assert_eq!(survivor.memory_type, MemoryType::RawEvent);
    }
}
