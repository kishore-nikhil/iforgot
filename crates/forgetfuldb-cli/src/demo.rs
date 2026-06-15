//! `forgetfuldb demo` — seed a self-contained demo store so the
//! observability UI has something worth looking at immediately:
//! ~200 memories spread over 60 simulated days across a handful of topic
//! clusters, with links (summaries, updates, contradictions), pinned and
//! stale examples, chat-turn metrics, and weekly consolidation runs.
//!
//! Deterministic by design (tiny LCG, fixed seed): no `rand` dependency,
//! and two runs produce the same store, which makes the UI demoable and
//! the seeder testable. Everything lands in its own directory — the real
//! memory store is never touched.

use anyhow::Result;
use forgetfuldb_core::config::Config;
use forgetfuldb_core::ids::new_id;
use forgetfuldb_core::ingest::content_hash;
use forgetfuldb_core::types::{LinkRelation, MemoryItem, MemoryLink, MemoryType};
use forgetfuldb_core::{decay, now_unix};
use forgetfuldb_store::{ChatTurn, ConsolidationRun, Store, SummaryProvenance};
use std::path::Path;

const DAY: i64 = 86_400;
const SIM_DAYS: i64 = 60;

/// Minimal deterministic generator (SplitMix-style).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() as i64).rem_euclid((hi - lo).max(1))
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.below(100) < pct
    }
}

/// Topic clusters: (topic, project tag, distilled facts, episodic events).
const CLUSTERS: &[(&str, &str, &[&str], &[&str])] = &[
    (
        "plotperfect",
        "project:plotperfect",
        &[
            "Plot Perfect billing runs on Stripe with monthly invoices",
            "Plot Perfect exports stories as EPUB and PDF",
            "Plot Perfect character sheets live in the characters/ collection",
            "Plot Perfect uses Firebase auth with Google sign-in only",
        ],
        &[
            "sketched the Plot Perfect story-arc editor",
            "fixed the Plot Perfect invoice webhook retries",
            "reviewed Plot Perfect onboarding funnel numbers",
            "tested Plot Perfect EPUB export on a 900-page draft",
        ],
    ),
    (
        "forgetfuldb",
        "project:iforgot",
        &[
            "ForgetfulDB stores memories in SQLite with WAL mode",
            "ForgetfulDB decay is exponential per memory type",
            "ForgetfulDB retrieval blends cosine similarity with keyword overlap",
            "ForgetfulDB consolidation merges duplicates above 0.92 cosine",
        ],
        &[
            "profiled ForgetfulDB retrieval on 10k rows",
            "tuned ForgetfulDB decay lambdas for episodic memories",
            "debugged a ForgetfulDB WAL checkpoint stall",
            "added a ForgetfulDB consolidation dry-run flag idea to the backlog",
        ],
    ),
    (
        "dayforge",
        "project:dayforge",
        &[
            "DayForge syncs tasks with CalDAV every fifteen minutes",
            "DayForge stores recurring schedules as RRULE strings",
            "DayForge widgets are built with WidgetKit timelines",
        ],
        &[
            "untangled a DayForge timezone bug around DST",
            "shipped the DayForge weekly review screen",
            "triaged DayForge sync conflicts from the beta group",
        ],
    ),
    (
        "standup",
        "work:standup",
        &[
            "the standup moved to nine thirty on Mondays",
            "sprint demos happen every second Friday",
        ],
        &[
            "presented the retrieval latency numbers at standup",
            "standup ran long discussing the Q3 roadmap",
            "paired with Sam after standup on the metrics dashboard",
        ],
    ),
    (
        "fitness",
        "life:fitness",
        &[
            "the gym is closed on public holidays",
            "tuesday and thursday are climbing days",
        ],
        &[
            "climbed two grades harder than last month",
            "skipped the gym for the release crunch",
        ],
    ),
];

const PREFERENCES: &[&str] = &[
    "I always prefer dark mode in every editor",
    "I like commit messages in present tense",
    "I prefer espresso over filter coffee",
    "I want summaries in bullet points, not prose",
    "I prefer metric units everywhere",
];

const PROCEDURES: &[&str] = &[
    "to release: bump the version, tag, run cargo publish, then announce",
    "to rotate the API key: regenerate in the dashboard, update 1Password, redeploy",
    "to back up the memory store: checkpoint WAL then copy the sqlite file",
    "to onboard a beta tester: send the TestFlight link and the welcome doc",
];

const USER_QUESTIONS: &[&str] = &[
    "what did I decide about billing?",
    "when is the standup again?",
    "summarize where DayForge stands",
    "what's left on the Plot Perfect launch list?",
    "how does retrieval scoring work?",
    "what did we ship last sprint?",
];

fn sprint_tag(day: i64) -> String {
    format!("sprint:2026-S{}", 10 + day / 14)
}

/// Build one memory with plausible scores for its age and type.
#[allow(clippy::too_many_arguments)]
fn make_memory(
    rng: &mut Lcg,
    provider: &dyn forgetfuldb_embed::EmbeddingProvider,
    cfg: &Config,
    now: i64,
    day: i64,
    text: String,
    memory_type: MemoryType,
    topic: &str,
    tag: &str,
) -> MemoryItem {
    let created_at = now - (SIM_DAYS - day) * DAY + rng.range(0, DAY / 2);
    let hash = content_hash(&text);
    let mut item = MemoryItem::new(new_id("mem", &hash), text, memory_type, hash, created_at);
    item.updated_at = created_at;
    item.topic = Some(topic.to_string());
    item.tags = vec![tag.to_string(), sprint_tag(day)];
    item.source = Some(if memory_type == MemoryType::RawEvent { "chat" } else { "demo" }.to_string());
    item.importance_score = match memory_type {
        MemoryType::Preference => 0.7 + rng.range(0, 20) as f64 / 100.0,
        MemoryType::Procedural => 0.65 + rng.range(0, 20) as f64 / 100.0,
        MemoryType::Semantic => 0.55 + rng.range(0, 30) as f64 / 100.0,
        MemoryType::Episodic => 0.4 + rng.range(0, 30) as f64 / 100.0,
        _ => 0.25 + rng.range(0, 20) as f64 / 100.0,
    };
    item.recurrence_score = rng.range(0, 60) as f64 / 100.0;
    item.access_count = rng.range(0, 8);
    if item.access_count > 0 {
        item.last_accessed_at = Some(created_at + rng.range(0, (SIM_DAYS - day).max(1) * DAY));
    }
    item.pinned = rng.chance(3);
    let lambda = cfg.decay_lambdas().for_type(memory_type);
    item.decay_score =
        decay::decay_score(item.importance_score, lambda, (SIM_DAYS - day) as f64, item.pinned);
    item.embedding = Some(provider.embed(&item.content));
    item
}

pub fn seed(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let db_path = dir.join("forgetfuldb-demo.sqlite3");
    // Clean slate on every run: the demo is disposable by contract.
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(dir.join(format!("forgetfuldb-demo.sqlite3{suffix}")));
    }
    // Stored relative: the config loader anchors relative sqlite paths
    // against the config file's own directory, so the demo dir stays
    // self-contained and relocatable.
    let cfg = Config {
        name: "demo".to_string(),
        sqlite_path: "forgetfuldb-demo.sqlite3".to_string(),
        ..Config::default()
    };
    cfg.save(&dir.join("forgetfuldb.toml"))?;

    let store = Store::open(&db_path)?;
    let provider = forgetfuldb_embed::create_provider_from_config(&cfg)?;
    let now = now_unix();
    let mut rng = Lcg(0x5EED);

    let mut all_ids: Vec<String> = Vec::new();
    let mut episodic_by_topic: Vec<(String, Vec<String>)> =
        CLUSTERS.iter().map(|(t, ..)| (t.to_string(), Vec::new())).collect();
    let mut fact_ids_by_topic: Vec<(String, Vec<String>)> = episodic_by_topic.clone();
    let mut memories = 0usize;

    // Day-by-day generation: a few episodic/raw events daily, distilled
    // facts sprinkled through the timeline.
    for day in 0..SIM_DAYS {
        let cluster_idx = rng.below(CLUSTERS.len() as u64) as usize;
        let (topic, tag, facts, events) = CLUSTERS[cluster_idx];

        for i in 0..(2 + rng.below(2)) {
            let event = events[rng.below(events.len() as u64) as usize];
            let item = make_memory(
                &mut rng, provider.as_ref(), &cfg, now, day,
                format!("{event} (day {day}.{i})"),
                MemoryType::Episodic, topic, tag,
            );
            episodic_by_topic[cluster_idx].1.push(item.id.clone());
            all_ids.push(item.id.clone());
            store.insert_memory(&item)?;
            memories += 1;
        }
        if rng.chance(70) {
            let event = events[rng.below(events.len() as u64) as usize];
            let item = make_memory(
                &mut rng, provider.as_ref(), &cfg, now, day,
                format!("assistant noted: {event}, follow-ups tracked (day {day})"),
                MemoryType::RawEvent, topic, tag,
            );
            all_ids.push(item.id.clone());
            store.insert_memory(&item)?;
            memories += 1;
        }
        if day % 4 == 0 {
            let fact = facts[(day as usize / 4) % facts.len()];
            let item = make_memory(
                &mut rng, provider.as_ref(), &cfg, now, day,
                format!("{fact} (noted day {day})"),
                MemoryType::Semantic, topic, tag,
            );
            fact_ids_by_topic[cluster_idx].1.push(item.id.clone());
            all_ids.push(item.id.clone());
            store.insert_memory(&item)?;
            memories += 1;
        }
        if day % 12 == 0 {
            let pref = PREFERENCES[(day as usize / 12) % PREFERENCES.len()];
            let item = make_memory(
                &mut rng, provider.as_ref(), &cfg, now, day,
                pref.to_string(), MemoryType::Preference, "preferences", "life:preferences",
            );
            all_ids.push(item.id.clone());
            store.insert_memory(&item)?;
            memories += 1;
        }
        if day % 15 == 1 {
            let proc = PROCEDURES[(day as usize / 15) % PROCEDURES.len()];
            let item = make_memory(
                &mut rng, provider.as_ref(), &cfg, now, day,
                proc.to_string(), MemoryType::Procedural, "procedures", "work:howto",
            );
            all_ids.push(item.id.clone());
            store.insert_memory(&item)?;
            memories += 1;
        }
    }

    // Contradiction/update pairs: a newer fact supersedes an older one,
    // which goes stale — exactly what the graph should make visible.
    let mut links = 0usize;
    for (idx, (_, _, facts, _)) in CLUSTERS.iter().enumerate().take(3) {
        let (topic, tag) = (CLUSTERS[idx].0, CLUSTERS[idx].1);
        let old = make_memory(
            &mut rng, provider.as_ref(), &cfg, now, 8,
            format!("{} (superseded draft)", facts[0]),
            MemoryType::Semantic, topic, tag,
        );
        let mut old = old;
        old.stale = true;
        let new_id_ = fact_ids_by_topic[idx].1.first().cloned();
        store.insert_memory(&old)?;
        memories += 1;
        all_ids.push(old.id.clone());
        if let Some(newer) = new_id_ {
            store.insert_link(&MemoryLink {
                source_id: newer,
                target_id: old.id.clone(),
                relation: LinkRelation::Updates,
            })?;
            links += 1;
        }
    }

    // Weekly consolidation runs; weeks 3 and 6 produce topic summaries
    // with full provenance (derived_from links + run log entries).
    let mut runs = Vec::new();
    for week in 1..=8i64 {
        let ran_at = now - (SIM_DAYS - week * 7).max(0) * DAY;
        let mut summaries = Vec::new();
        if week == 3 || week == 6 {
            let idx = (week as usize) % CLUSTERS.len();
            let (topic, tag, facts, _) = CLUSTERS[idx];
            let sources: Vec<String> =
                episodic_by_topic[idx].1.iter().take(4 + week as usize).cloned().collect();
            if sources.len() >= 3 {
                let day = week * 7;
                let mut summary = make_memory(
                    &mut rng, provider.as_ref(), &cfg, now, day,
                    format!("summary: {} — {} (week {week})", topic, facts[0]),
                    MemoryType::Semantic, topic, tag,
                );
                summary.source = Some("consolidation".to_string());
                summary.summary = Some(summary.content.clone());
                store.insert_memory(&summary)?;
                memories += 1;
                all_ids.push(summary.id.clone());
                for src in &sources {
                    store.insert_link(&MemoryLink {
                        source_id: summary.id.clone(),
                        target_id: src.clone(),
                        relation: LinkRelation::DerivedFrom,
                    })?;
                    links += 1;
                }
                summaries.push(SummaryProvenance { summary_id: summary.id.clone(), source_ids: sources });
            }
        }
        let run = ConsolidationRun {
            id: new_id("run", &format!("demo-{week}")),
            ran_at,
            duplicates_merged: rng.range(0, 4),
            recurrence_updated: rng.range(5, 30),
            clusters_summarized: summaries.len() as i64,
            promoted: rng.range(0, 3),
            marked_stale: rng.range(0, 2),
            archived: rng.range(0, 6),
            pruned: rng.range(0, 4),
            summaries,
        };
        store.log_consolidation_run(&run)?;
        runs.push(run);
    }

    // Chat-turn metrics: 1-2 turns most days, token counts and latencies
    // drifting upward as the store (and prompt) grows.
    let mut turns = 0usize;
    for day in 0..SIM_DAYS {
        if rng.chance(25) {
            continue;
        }
        for _ in 0..(1 + rng.below(2)) {
            let created_at = now - (SIM_DAYS - day) * DAY + rng.range(0, DAY);
            let q = USER_QUESTIONS[rng.below(USER_QUESTIONS.len() as u64) as usize];
            let injected = (0..rng.below(7))
                .filter_map(|_| {
                    if all_ids.is_empty() { None } else { Some(all_ids[rng.below(all_ids.len() as u64) as usize].clone()) }
                })
                .collect::<Vec<_>>();
            let context_chars = injected.len() as i64 * rng.range(60, 140);
            let turn = ChatTurn {
                id: new_id("turn", &format!("demo-{day}-{turns}")),
                session_id: Some(format!("session_demo_{}", day / 7)),
                created_at,
                user_text: q.to_string(),
                assistant_text: format!("(demo reply about: {q})"),
                model: "gemma4:12b".to_string(),
                backend: "ollama".to_string(),
                prompt_tokens: Some(700 + day * 12 + rng.range(0, 600)),
                completion_tokens: Some(rng.range(60, 550)),
                total_duration_ms: Some(rng.range(1200, 11_000)),
                llm_duration_ms: Some(rng.range(900, 9_500)),
                retrieve_duration_ms: rng.range(1, 16),
                context_memory_count: injected.len() as i64,
                context_chars,
                memory_ids: injected,
            };
            store.insert_chat_turn(&turn)?;
            turns += 1;
        }
    }

    // Build co-occurrence association edges from the seeded chat turns, so
    // the graph shows "used-together" links immediately.
    let assoc = forgetfuldb_store::pipeline::rebuild_cooccurrence_edges(
        &store,
        cfg.edge_decay_lambda,
        cfg.edge_min_weight,
        now,
    )?;

    println!("seeded demo store in {}:", dir.display());
    println!(
        "  {memories} memories | {links} links | {assoc} co-occurrence edges | {} consolidation runs | {turns} chat turns",
        runs.len()
    );
    println!("explore it:");
    println!("  forgetfuldb server --config {} --ui ui/dist", dir.join("forgetfuldb.toml").display());
    println!("  open http://127.0.0.1:8787/ui");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_seed_is_deterministic_and_populated() {
        let dir = std::env::temp_dir().join(format!("fdb-demo-test-{}", std::process::id()));
        seed(&dir).unwrap();
        let store = Store::open(&dir.join("forgetfuldb-demo.sqlite3")).unwrap();
        let stats = store.stats().unwrap();
        assert!(stats.total_memories >= 150, "expected a populated store, got {}", stats.total_memories);
        assert!(stats.links > 5);
        assert!(!store.list_consolidation_runs(20).unwrap().is_empty());
        assert!(!store.list_chat_turns(500).unwrap().is_empty());

        // Re-seeding is idempotent (clean slate, same generator seed).
        let first_total = stats.total_memories;
        drop(store);
        seed(&dir).unwrap();
        let store = Store::open(&dir.join("forgetfuldb-demo.sqlite3")).unwrap();
        assert_eq!(store.stats().unwrap().total_memories, first_total);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
