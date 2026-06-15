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
use forgetfuldb_core::types::{LinkRelation, MemoryItem, MemoryLink, MemoryType};
use forgetfuldb_core::{age_days, decay, now_unix};
use forgetfuldb_embed::cosine_similarity;
use forgetfuldb_prob::{CountMinSketch, ReservoirSampler};
use forgetfuldb_store::Store;
use serde::Serialize;
use std::collections::HashMap;

/// What one consolidation pass did.
#[derive(Debug, Default, Clone, Serialize)]
pub struct ConsolidationReport {
    pub duplicates_merged: usize,
    pub recurrence_updated: usize,
    pub clusters_summarized: usize,
    pub promoted_to_semantic: usize,
    pub marked_stale: usize,
    pub archived: usize,
    pub deleted: usize,
    /// Provenance of every summary memory created this pass.
    pub summaries: Vec<forgetfuldb_store::SummaryProvenance>,
    /// Co-occurrence association edges rebuilt from chat history.
    pub associations: usize,
    /// Memories whose salience was revised by the neighbor discriminator.
    pub salience_revised: usize,
    /// `semantic_similar` (cosine kNN) edges rebuilt.
    pub semantic_edges: usize,
    /// `sequence` (session-order) edges rebuilt.
    pub sequence_edges: usize,
}

/// Run a full consolidation pass. Every pass is logged to the
/// `consolidation_runs` table so the observability UI can show what each
/// sleep cycle did.
pub fn consolidate(store: &Store, summarizer: &dyn Summarizer, cfg: &Config) -> Result<ConsolidationReport> {
    let mut report = ConsolidationReport::default();
    let now = now_unix();

    merge_duplicates(store, cfg, now, &mut report)?;
    refresh_recurrence(store, now, &mut report)?;
    revise_salience(store, now, &mut report)?;
    summarize_clusters(store, summarizer, cfg, now, &mut report)?;
    promote_recurring(store, cfg, now, &mut report)?;
    mark_contradicted_stale(store, &mut report)?;
    archive_and_prune(store, cfg, now, &mut report)?;

    // Rebuild the association graph from scratch. Done last, after pruning,
    // so edges never point at deleted memories. Three distinct edge types,
    // each a different notion of "related":
    //   co_occurred     — recalled together (behavioral / Hebbian)
    //   semantic_similar — close in meaning (embedding kNN)
    //   sequence        — discussed one after another (causal / session order)
    report.associations =
        forgetfuldb_store::pipeline::rebuild_cooccurrence_edges(store, cfg.edge_decay_lambda, cfg.edge_min_weight, now)?;
    report.semantic_edges = forgetfuldb_store::pipeline::rebuild_semantic_edges(
        store,
        cfg.semantic_edge_min_sim,
        cfg.semantic_edge_top_k,
        now,
    )?;
    report.sequence_edges =
        forgetfuldb_store::pipeline::rebuild_sequence_edges(store, cfg.edge_decay_lambda, cfg.edge_min_weight, now, 2)?;

    store.log_consolidation_run(&forgetfuldb_store::ConsolidationRun {
        id: new_id("run", &format!("consolidate-{now}")),
        ran_at: now,
        duplicates_merged: report.duplicates_merged as i64,
        recurrence_updated: report.recurrence_updated as i64,
        clusters_summarized: report.clusters_summarized as i64,
        promoted: report.promoted_to_semantic as i64,
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
    use forgetfuldb_core::salience::{analyze_neighbors, salience, Neighbor, NeighborParams};

    // Archives are de-emphasized copies of pruned memories, kept for the
    // record but out of the active retrieval corpus — so they neither get a
    // salience nor count as neighbors for active memories.
    let items: Vec<MemoryItem> = store
        .list_memories(None)?
        .into_iter()
        .filter(|m| m.memory_type != MemoryType::Archive && m.embedding.is_some())
        .collect();
    if items.len() < 2 {
        return Ok(());
    }
    // History span: age of the oldest memory, the window spread is measured
    // against.
    let oldest = items.iter().map(|m| m.created_at).min().unwrap_or(now);
    let history_span_days = age_days(oldest, now).max(1.0);
    let params = NeighborParams::default();

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
        let relevance = forgetfuldb_core::salience::content_relevance(item.content.chars().count(), item.entities.len());
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
fn merge_duplicates(store: &Store, cfg: &Config, now: i64, report: &mut ConsolidationReport) -> Result<()> {
    let threshold = cfg.consolidation_thresholds.duplicate_similarity as f32;
    let items: Vec<MemoryItem> = store
        .list_memories(None)?
        .into_iter()
        .filter(|m| m.memory_type != MemoryType::Archive && m.embedding.is_some())
        .collect();

    let mut removed: Vec<bool> = vec![false; items.len()];
    for i in 0..items.len() {
        if removed[i] {
            continue;
        }
        for j in (i + 1)..items.len() {
            if removed[j] {
                continue;
            }
            let sim = cosine_similarity(
                items[i].embedding.as_ref().unwrap(),
                items[j].embedding.as_ref().unwrap(),
            );
            if sim >= threshold {
                // Keep the higher-importance item; tie goes to the newer one.
                let (keep_idx, dup_idx) = if items[i].importance_score > items[j].importance_score
                    || (items[i].importance_score == items[j].importance_score
                        && items[i].created_at >= items[j].created_at)
                {
                    (i, j)
                } else {
                    (j, i)
                };
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

/// Rebuild a Count-Min Sketch of topic/entity frequencies from the store
/// and refresh each memory's recurrence score from it. The sketch gives
/// approximate counts in O(1) space per key — slight overcounting is
/// acceptable for a soft score.
fn refresh_recurrence(store: &Store, now: i64, report: &mut ConsolidationReport) -> Result<()> {
    let items = store.list_memories(None)?;
    let mut cms = CountMinSketch::default_size();
    for item in &items {
        if let Some(topic) = &item.topic {
            cms.add(topic, 1);
        }
        for entity in &item.entities {
            cms.add(entity, 1);
        }
    }
    for mut item in items {
        let freq = item
            .topic
            .as_deref()
            .map(|t| cms.estimate(t))
            .unwrap_or(0)
            .max(item.entities.iter().map(|e| cms.estimate(e)).max().unwrap_or(0));
        // log-compress: freq 1 -> 0, 10 -> ~0.7, 30+ -> 1.0
        let recurrence = ((1.0 + freq as f64).ln() / (30.0f64).ln()).clamp(0.0, 1.0);
        if (recurrence - item.recurrence_score).abs() > 0.05 && recurrence > item.recurrence_score {
            item.recurrence_score = recurrence;
            item.updated_at = now;
            store.update_memory(&item)?;
            report.recurrence_updated += 1;
        }
    }
    Ok(())
}

/// Group active episodic memories by topic; clusters big enough get an
/// extractive summary stored as a semantic memory linked `derived_from`.
fn summarize_clusters(
    store: &Store,
    summarizer: &dyn Summarizer,
    cfg: &Config,
    now: i64,
    report: &mut ConsolidationReport,
) -> Result<()> {
    let min_size = cfg.consolidation_thresholds.cluster_min_size;
    let mut clusters: HashMap<String, Vec<MemoryItem>> = HashMap::new();
    for item in store.list_memories(Some(MemoryType::Episodic))? {
        if item.stale {
            continue;
        }
        if let Some(topic) = &item.topic {
            clusters.entry(topic.clone()).or_default().push(item);
        }
    }

    for (topic, members) in clusters {
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
        let mut item = MemoryItem::new(new_id("mem", &hash), summary.clone(), MemoryType::Semantic, hash, now);
        item.summary = Some(summary);
        item.topic = Some(topic.clone());
        item.source = Some("consolidation".to_string());
        item.importance_score = members
            .iter()
            .map(|m| m.importance_score)
            .fold(0.0, f64::max)
            .max(0.6);
        item.recurrence_score = (members.len() as f64 / 10.0).min(1.0);
        item.decay_score = item.importance_score;
        store.insert_memory(&item)?;
        for member in &members {
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

/// Episodic memories rehearsed often enough graduate to semantic memory
/// (slower decay): repetition turns experience into knowledge.
fn promote_recurring(store: &Store, cfg: &Config, now: i64, report: &mut ConsolidationReport) -> Result<()> {
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

/// Any memory that is the target of a `contradicts` or `updates` link is
/// out of date by definition — mark it stale (kept, but hidden from
/// retrieval unless explicitly requested).
fn mark_contradicted_stale(store: &Store, report: &mut ConsolidationReport) -> Result<()> {
    for link in store.all_links()? {
        if matches!(link.relation, LinkRelation::Contradicts | LinkRelation::Updates) {
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
fn archive_and_prune(store: &Store, cfg: &Config, now: i64, report: &mut ConsolidationReport) -> Result<()> {
    let lambdas = cfg.decay_lambdas();
    let archive_cutoff_days = cfg.archive_after_days;
    let delete_cutoff_days = cfg.delete_after_days;
    let max_decay = cfg.consolidation_thresholds.archive_max_decay;

    for item in store.list_memories(None)? {
        if item.pinned {
            continue; // pinned memories never decay or get pruned
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
            let mut item = MemoryItem::new(new_id("mem", &hash), note, MemoryType::Archive, hash, now);
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

    fn add(store: &Store, provider: &dyn EmbeddingProvider, cfg: &Config, text: &str, mt: MemoryType) -> String {
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

    // ---- Eval Layer 1: behavior tests (each isolates one mechanism) ----

    #[test]
    fn eval_surprise_a_novel_memory_outscores_routine_on_salience() {
        let (store, provider, mut cfg) = setup();
        // Isolate the salience mechanism: disable dedup-merging so the
        // routine cluster stays intact (merging it to one would erase the
        // "many similar neighbors" signal this test is about).
        cfg.consolidation_thresholds.duplicate_similarity = 0.999;
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
        assert_eq!(kept.memory_type, MemoryType::RawEvent, "the anomaly must survive (salience {:.3})", kept.salience);
        assert!(kept.salience >= cfg.salience_keep_threshold, "anomaly salience {:.3} should clear the keep bar", kept.salience);

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
        assert!(archived >= 4, "most routine memories should be archived/forgotten, got {archived}/6");
    }

    #[test]
    fn merge_pair_combines_history() {
        let now = now_unix();
        let mut a = MemoryItem::new("a".into(), "fact".into(), MemoryType::Episodic, "h1".into(), now - 100);
        let mut b = MemoryItem::new("b".into(), "fact again".into(), MemoryType::Episodic, "h2".into(), now);
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
        add(&store, provider.as_ref(), &cfg, "stripe handles plot perfect billing invoices", MemoryType::Episodic);
        add(&store, provider.as_ref(), &cfg, "billing invoices plot perfect handles stripe", MemoryType::Episodic);
        add(&store, provider.as_ref(), &cfg, "granite countertops were measured on tuesday", MemoryType::Episodic);

        let report = consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
        assert_eq!(report.duplicates_merged, 1);
        // 3 ingested - 1 merged away (no clusters of 3+ on same topic remain)
        let stats = store.stats().unwrap();
        assert_eq!(
            stats.by_type.iter().find(|(t, _)| t == "episodic").unwrap().1,
            2
        );
    }

    #[test]
    fn contradicted_memory_marked_stale() {
        let (store, provider, cfg) = setup();
        let old = add(&store, provider.as_ref(), &cfg, "the office is in building seven", MemoryType::Semantic);
        let new = add(&store, provider.as_ref(), &cfg, "the office moved to building twelve", MemoryType::Semantic);
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
        let id = add(&store, provider.as_ref(), &cfg, "deploys happen from the main branch", MemoryType::Episodic);
        for _ in 0..3 {
            store.touch_memory(&id, now_unix()).unwrap();
        }
        let report = consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
        assert_eq!(report.promoted_to_semantic, 1);
        assert_eq!(store.get_memory(&id).unwrap().unwrap().memory_type, MemoryType::Semantic);
    }

    #[test]
    fn topic_clusters_get_summarized() {
        let (store, provider, cfg) = setup();
        add(&store, provider.as_ref(), &cfg, "discussed plot perfect billing edge cases for refunds", MemoryType::Episodic);
        add(&store, provider.as_ref(), &cfg, "plot perfect customers complained about invoice formatting", MemoryType::Episodic);
        add(&store, provider.as_ref(), &cfg, "decided plot perfect will switch to usage based pricing", MemoryType::Episodic);

        let report = consolidate(&store, &ExtractiveSummarizer::default(), &cfg).unwrap();
        assert!(report.clusters_summarized >= 1);
        let semantic = store.list_memories(Some(MemoryType::Semantic)).unwrap();
        assert!(semantic.iter().any(|m| m.source.as_deref() == Some("consolidation")));
    }

    #[test]
    fn pinned_memories_survive_pruning() {
        let (store, provider, cfg) = setup();
        let id = add(&store, provider.as_ref(), &cfg, "never delete this pinned reminder", MemoryType::RawEvent);
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
