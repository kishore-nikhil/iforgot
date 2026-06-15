//! forgetfuldb-store
//!
//! SQLite persistence for ForgetfulDB using rusqlite (bundled SQLite, so
//! no system dependency on macOS). Migrations are embedded SQL files
//! tracked in a `schema_migrations` table.
//!
//! Also hosts [`pipeline`], the small orchestration layer for the ingest
//! workflow shared by the CLI and the HTTP server.

pub mod pipeline;

use anyhow::{Context, Result};
use forgetfuldb_core::types::{LinkRelation, MemoryItem, MemoryLink, MemoryType, RawEvent, Session};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::str::FromStr;

/// Embedded, ordered migrations. Add new `(name, sql)` pairs at the end;
/// applied ones are skipped via the `schema_migrations` table.
const MIGRATIONS: &[(&str, &str)] = &[
    ("0001_init", include_str!("../migrations/0001_init.sql")),
    ("0002_chat_turns", include_str!("../migrations/0002_chat_turns.sql")),
    ("0003_consolidation_runs", include_str!("../migrations/0003_consolidation_runs.sql")),
    ("0004_store_meta", include_str!("../migrations/0004_store_meta.sql")),
    ("0005_memory_edges", include_str!("../migrations/0005_memory_edges.sql")),
];

pub struct Store {
    conn: Connection,
}

impl Store {
    /// Open (creating if needed) the database at `path` and apply
    /// pending migrations.
    pub fn open(path: &Path) -> Result<Store> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let conn = Connection::open(path)
            .with_context(|| format!("opening sqlite db at {}", path.display()))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        // Multiple connections may share the file (e.g. the chat thread
        // reads while the background memory writer writes); wait briefly
        // instead of failing on transient lock contention.
        conn.pragma_update(None, "busy_timeout", 5000)?;
        let store = Store { conn };
        store.migrate()?;
        Ok(store)
    }

    /// In-memory database for tests.
    pub fn open_in_memory() -> Result<Store> {
        let store = Store { conn: Connection::open_in_memory()? };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_migrations (
                 name TEXT PRIMARY KEY,
                 applied_at INTEGER NOT NULL
             );",
        )?;
        for (name, sql) in MIGRATIONS {
            let applied: Option<String> = self
                .conn
                .query_row("SELECT name FROM schema_migrations WHERE name = ?1", [name], |r| r.get(0))
                .optional()?;
            if applied.is_none() {
                self.conn.execute_batch(sql).with_context(|| format!("applying migration {name}"))?;
                self.conn.execute(
                    "INSERT INTO schema_migrations (name, applied_at) VALUES (?1, ?2)",
                    params![name, forgetfuldb_core::now_unix()],
                )?;
            }
        }
        Ok(())
    }

    // ---- memory_items -------------------------------------------------

    pub fn insert_memory(&self, item: &MemoryItem) -> Result<()> {
        self.conn.execute(
            "INSERT INTO memory_items (
                 id, content, summary, memory_type, source, topic, entities, tags,
                 created_at, updated_at, last_accessed_at, access_count,
                 importance_score, recurrence_score, recency_score, decay_score,
                 confidence, stale, pinned, embedding, content_hash
             ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21)",
            params![
                item.id,
                item.content,
                item.summary,
                item.memory_type.as_str(),
                item.source,
                item.topic,
                serde_json::to_string(&item.entities)?,
                serde_json::to_string(&item.tags)?,
                item.created_at,
                item.updated_at,
                item.last_accessed_at,
                item.access_count,
                item.importance_score,
                item.recurrence_score,
                item.recency_score,
                item.decay_score,
                item.confidence,
                item.stale as i64,
                item.pinned as i64,
                item.embedding.as_ref().map(|e| encode_embedding(e)),
                item.content_hash,
            ],
        )?;
        Ok(())
    }

    pub fn update_memory(&self, item: &MemoryItem) -> Result<()> {
        self.conn.execute(
            "UPDATE memory_items SET
                 content = ?2, summary = ?3, memory_type = ?4, source = ?5, topic = ?6,
                 entities = ?7, tags = ?8, updated_at = ?9, last_accessed_at = ?10,
                 access_count = ?11, importance_score = ?12, recurrence_score = ?13,
                 recency_score = ?14, decay_score = ?15, confidence = ?16,
                 stale = ?17, pinned = ?18, embedding = ?19, content_hash = ?20
             WHERE id = ?1",
            params![
                item.id,
                item.content,
                item.summary,
                item.memory_type.as_str(),
                item.source,
                item.topic,
                serde_json::to_string(&item.entities)?,
                serde_json::to_string(&item.tags)?,
                item.updated_at,
                item.last_accessed_at,
                item.access_count,
                item.importance_score,
                item.recurrence_score,
                item.recency_score,
                item.decay_score,
                item.confidence,
                item.stale as i64,
                item.pinned as i64,
                item.embedding.as_ref().map(|e| encode_embedding(e)),
                item.content_hash,
            ],
        )?;
        Ok(())
    }

    pub fn get_memory(&self, id: &str) -> Result<Option<MemoryItem>> {
        self.conn
            .query_row(
                &format!("SELECT {MEMORY_COLUMNS} FROM memory_items WHERE id = ?1"),
                [id],
                row_to_memory,
            )
            .optional()
            .map_err(Into::into)
    }

    pub fn get_memory_by_hash(&self, content_hash: &str) -> Result<Option<MemoryItem>> {
        self.conn
            .query_row(
                &format!("SELECT {MEMORY_COLUMNS} FROM memory_items WHERE content_hash = ?1"),
                [content_hash],
                row_to_memory,
            )
            .optional()
            .map_err(Into::into)
    }

    /// All memories, optionally filtered by type. Fine for a personal
    /// database (thousands of rows); see README limitations for scale.
    pub fn list_memories(&self, memory_type: Option<MemoryType>) -> Result<Vec<MemoryItem>> {
        let mut items = Vec::new();
        match memory_type {
            Some(mt) => {
                let mut stmt = self.conn.prepare(&format!(
                    "SELECT {MEMORY_COLUMNS} FROM memory_items WHERE memory_type = ?1 ORDER BY created_at"
                ))?;
                let rows = stmt.query_map([mt.as_str()], row_to_memory)?;
                for row in rows {
                    items.push(row?);
                }
            }
            None => {
                let mut stmt = self.conn.prepare(&format!(
                    "SELECT {MEMORY_COLUMNS} FROM memory_items ORDER BY created_at"
                ))?;
                let rows = stmt.query_map([], row_to_memory)?;
                for row in rows {
                    items.push(row?);
                }
            }
        }
        Ok(items)
    }

    pub fn delete_memory(&self, id: &str) -> Result<bool> {
        let n = self.conn.execute("DELETE FROM memory_items WHERE id = ?1", [id])?;
        self.conn.execute(
            "DELETE FROM memory_links WHERE source_id = ?1 OR target_id = ?1",
            [id],
        )?;
        Ok(n > 0)
    }

    /// Record a retrieval hit: bump access_count and last_accessed_at.
    pub fn touch_memory(&self, id: &str, now: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE memory_items SET access_count = access_count + 1,
                 last_accessed_at = ?2 WHERE id = ?1",
            params![id, now],
        )?;
        Ok(())
    }

    pub fn set_pinned(&self, id: &str, pinned: bool) -> Result<bool> {
        let n = self.conn.execute(
            "UPDATE memory_items SET pinned = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, pinned as i64, forgetfuldb_core::now_unix()],
        )?;
        Ok(n > 0)
    }

    pub fn set_stale(&self, id: &str, stale: bool) -> Result<bool> {
        let n = self.conn.execute(
            "UPDATE memory_items SET stale = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, stale as i64, forgetfuldb_core::now_unix()],
        )?;
        Ok(n > 0)
    }

    pub fn set_memory_type(&self, id: &str, memory_type: MemoryType) -> Result<bool> {
        let n = self.conn.execute(
            "UPDATE memory_items SET memory_type = ?2, updated_at = ?3 WHERE id = ?1",
            params![id, memory_type.as_str(), forgetfuldb_core::now_unix()],
        )?;
        Ok(n > 0)
    }

    /// Every stored content hash — used to warm the Bloom filter at startup.
    pub fn all_content_hashes(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT content_hash FROM memory_items")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    // ---- memory_links --------------------------------------------------

    pub fn insert_link(&self, link: &MemoryLink) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO memory_links (source_id, target_id, relation) VALUES (?1, ?2, ?3)",
            params![link.source_id, link.target_id, link.relation.as_str()],
        )?;
        Ok(())
    }

    pub fn links_for(&self, memory_id: &str) -> Result<Vec<MemoryLink>> {
        let mut stmt = self.conn.prepare(
            "SELECT source_id, target_id, relation FROM memory_links
             WHERE source_id = ?1 OR target_id = ?1",
        )?;
        let rows = stmt.query_map([memory_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (source_id, target_id, relation) = row?;
            out.push(MemoryLink {
                source_id,
                target_id,
                relation: LinkRelation::from_str(&relation)
                    .map_err(|e| anyhow::anyhow!(e))?,
            });
        }
        Ok(out)
    }

    pub fn all_links(&self) -> Result<Vec<MemoryLink>> {
        let mut stmt = self.conn.prepare("SELECT source_id, target_id, relation FROM memory_links")?;
        let rows = stmt.query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (source_id, target_id, relation) = row?;
            out.push(MemoryLink {
                source_id,
                target_id,
                relation: LinkRelation::from_str(&relation).map_err(|e| anyhow::anyhow!(e))?,
            });
        }
        Ok(out)
    }

    // ---- raw_events ------------------------------------------------------

    pub fn insert_raw_event(&self, ev: &RawEvent) -> Result<()> {
        self.conn.execute(
            "INSERT INTO raw_events (id, session_id, role, content, created_at, content_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![ev.id, ev.session_id, ev.role, ev.content, ev.created_at, ev.content_hash],
        )?;
        Ok(())
    }

    pub fn raw_events_older_than(&self, cutoff: i64) -> Result<Vec<RawEvent>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, role, content, created_at, content_hash
             FROM raw_events WHERE created_at < ?1",
        )?;
        let rows = stmt.query_map([cutoff], |r| {
            Ok(RawEvent {
                id: r.get(0)?,
                session_id: r.get(1)?,
                role: r.get(2)?,
                content: r.get(3)?,
                created_at: r.get(4)?,
                content_hash: r.get(5)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn delete_raw_event(&self, id: &str) -> Result<()> {
        self.conn.execute("DELETE FROM raw_events WHERE id = ?1", [id])?;
        Ok(())
    }

    // ---- sessions --------------------------------------------------------

    pub fn upsert_session(&self, session: &Session) -> Result<()> {
        self.conn.execute(
            "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at,
                 title = COALESCE(excluded.title, sessions.title)",
            params![session.id, session.title, session.created_at, session.updated_at],
        )?;
        Ok(())
    }

    // ---- chat turn metrics ------------------------------------------------

    pub fn insert_chat_turn(&self, t: &ChatTurn) -> Result<()> {
        self.conn.execute(
            "INSERT INTO chat_turns (
                 id, session_id, created_at, user_text, assistant_text, model, backend,
                 prompt_tokens, completion_tokens, total_duration_ms, llm_duration_ms,
                 retrieve_duration_ms, context_memory_count, context_chars, memory_ids
             ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)",
            params![
                t.id,
                t.session_id,
                t.created_at,
                t.user_text,
                t.assistant_text,
                t.model,
                t.backend,
                t.prompt_tokens,
                t.completion_tokens,
                t.total_duration_ms,
                t.llm_duration_ms,
                t.retrieve_duration_ms,
                t.context_memory_count,
                t.context_chars,
                serde_json::to_string(&t.memory_ids)?,
            ],
        )?;
        Ok(())
    }

    /// Aggregate chat metrics: the raw material for context optimization.
    pub fn chat_metrics_summary(&self) -> Result<ChatMetricsSummary> {
        self.conn
            .query_row(
                "SELECT COUNT(*),
                        AVG(prompt_tokens), AVG(completion_tokens),
                        SUM(prompt_tokens), SUM(completion_tokens),
                        AVG(context_chars), AVG(context_memory_count),
                        AVG(retrieve_duration_ms), AVG(llm_duration_ms)
                 FROM chat_turns",
                [],
                |r| {
                    Ok(ChatMetricsSummary {
                        turns: r.get(0)?,
                        avg_prompt_tokens: r.get(1)?,
                        avg_completion_tokens: r.get(2)?,
                        total_prompt_tokens: r.get::<_, Option<i64>>(3)?.unwrap_or(0),
                        total_completion_tokens: r.get::<_, Option<i64>>(4)?.unwrap_or(0),
                        avg_context_chars: r.get(5)?,
                        avg_context_memories: r.get(6)?,
                        avg_retrieve_ms: r.get(7)?,
                        avg_llm_ms: r.get(8)?,
                    })
                },
            )
            .map_err(Into::into)
    }

    /// Most recent `limit` chat turns, returned oldest-first (chart-ready).
    pub fn list_chat_turns(&self, limit: usize) -> Result<Vec<ChatTurn>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, session_id, created_at, user_text, assistant_text, model, backend,
                    prompt_tokens, completion_tokens, total_duration_ms, llm_duration_ms,
                    retrieve_duration_ms, context_memory_count, context_chars, memory_ids
             FROM chat_turns ORDER BY created_at DESC, id DESC LIMIT ?1",
        )?;
        let mut turns: Vec<ChatTurn> = stmt
            .query_map([limit as i64], |r| {
                let memory_ids: String = r.get(14)?;
                Ok(ChatTurn {
                    id: r.get(0)?,
                    session_id: r.get(1)?,
                    created_at: r.get(2)?,
                    user_text: r.get(3)?,
                    assistant_text: r.get(4)?,
                    model: r.get(5)?,
                    backend: r.get(6)?,
                    prompt_tokens: r.get(7)?,
                    completion_tokens: r.get(8)?,
                    total_duration_ms: r.get(9)?,
                    llm_duration_ms: r.get(10)?,
                    retrieve_duration_ms: r.get(11)?,
                    context_memory_count: r.get(12)?,
                    context_chars: r.get(13)?,
                    memory_ids: serde_json::from_str(&memory_ids).unwrap_or_default(),
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        turns.reverse();
        Ok(turns)
    }

    // ---- store metadata (key/value) ---------------------------------------

    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        self.conn
            .query_row("SELECT value FROM store_meta WHERE key = ?1", [key], |r| r.get(0))
            .optional()
            .map_err(Into::into)
    }

    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO store_meta (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    /// Overwrite just the embedding vector of one memory (used when
    /// re-embedding the store under a new model). Leaves all scores intact.
    pub fn set_embedding(&self, id: &str, embedding: &[f32]) -> Result<()> {
        self.conn.execute(
            "UPDATE memory_items SET embedding = ?2 WHERE id = ?1",
            params![id, encode_embedding(embedding)],
        )?;
        Ok(())
    }

    // ---- consolidation runs ------------------------------------------------

    pub fn log_consolidation_run(&self, run: &ConsolidationRun) -> Result<()> {
        self.conn.execute(
            "INSERT INTO consolidation_runs (
                 id, ran_at, duplicates_merged, recurrence_updated, clusters_summarized,
                 promoted, marked_stale, archived, pruned, summaries
             ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![
                run.id,
                run.ran_at,
                run.duplicates_merged,
                run.recurrence_updated,
                run.clusters_summarized,
                run.promoted,
                run.marked_stale,
                run.archived,
                run.pruned,
                serde_json::to_string(&run.summaries)?,
            ],
        )?;
        Ok(())
    }

    /// Most recent `limit` consolidation runs, newest first.
    pub fn list_consolidation_runs(&self, limit: usize) -> Result<Vec<ConsolidationRun>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, ran_at, duplicates_merged, recurrence_updated, clusters_summarized,
                    promoted, marked_stale, archived, pruned, summaries
             FROM consolidation_runs ORDER BY ran_at DESC, id DESC LIMIT ?1",
        )?;
        let runs = stmt
            .query_map([limit as i64], |r| {
                let summaries: String = r.get(9)?;
                Ok(ConsolidationRun {
                    id: r.get(0)?,
                    ran_at: r.get(1)?,
                    duplicates_merged: r.get(2)?,
                    recurrence_updated: r.get(3)?,
                    clusters_summarized: r.get(4)?,
                    promoted: r.get(5)?,
                    marked_stale: r.get(6)?,
                    archived: r.get(7)?,
                    pruned: r.get(8)?,
                    summaries: serde_json::from_str(&summaries).unwrap_or_default(),
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(runs)
    }

    // ---- memory_edges (weighted associations) -----------------------------

    /// Insert or replace one weighted edge (canonical src < dst handled by
    /// the caller). Used by the co-occurrence rebuild.
    pub fn upsert_edge(&self, edge: &MemoryEdge) -> Result<()> {
        self.conn.execute(
            "INSERT INTO memory_edges (src_id, dst_id, edge_type, weight, co_count, created_at, last_activated)
             VALUES (?1,?2,?3,?4,?5,?6,?7)
             ON CONFLICT(src_id, dst_id, edge_type) DO UPDATE SET
                 weight = excluded.weight,
                 co_count = excluded.co_count,
                 last_activated = excluded.last_activated",
            params![
                edge.src_id,
                edge.dst_id,
                edge.edge_type,
                edge.weight,
                edge.co_count,
                edge.created_at,
                edge.last_activated,
            ],
        )?;
        Ok(())
    }

    /// Delete every edge of one type (the co-occurrence pass rebuilds them
    /// from scratch each run, so it clears first).
    pub fn clear_edges(&self, edge_type: &str) -> Result<()> {
        self.conn.execute("DELETE FROM memory_edges WHERE edge_type = ?1", [edge_type])?;
        Ok(())
    }

    /// Incrementally strengthen `co_occurred` edges among `ids` — the
    /// memories injected into one chat turn. Each pair's weight and count
    /// grow by one; new pairs are created. Additive (unlike the
    /// consolidation rebuild, which recomputes with decay), so this is the
    /// cheap live update that runs off the conversation path. Returns the
    /// number of pairs touched. A `+1` here equals the rebuild's
    /// contribution for an age-0 turn, so the two stay consistent.
    pub fn bump_cooccurrence_edges(&self, ids: &[String], now: i64) -> Result<usize> {
        // Unique, canonical (src < dst) pairs.
        let mut uniq: Vec<&String> = ids.iter().collect();
        uniq.sort();
        uniq.dedup();
        let mut pairs = 0;
        let tx = self.conn.unchecked_transaction()?;
        for i in 0..uniq.len() {
            for j in (i + 1)..uniq.len() {
                tx.execute(
                    "INSERT INTO memory_edges (src_id, dst_id, edge_type, weight, co_count, created_at, last_activated)
                     VALUES (?1, ?2, 'co_occurred', 1.0, 1, ?3, ?3)
                     ON CONFLICT(src_id, dst_id, edge_type) DO UPDATE SET
                         weight = weight + 1.0,
                         co_count = co_count + 1,
                         last_activated = excluded.last_activated",
                    params![uniq[i], uniq[j], now],
                )?;
                pairs += 1;
            }
        }
        tx.commit()?;
        Ok(pairs)
    }

    /// All weighted edges (for the graph view).
    pub fn list_edges(&self) -> Result<Vec<MemoryEdge>> {
        let mut stmt = self.conn.prepare(
            "SELECT src_id, dst_id, edge_type, weight, co_count, created_at, last_activated FROM memory_edges",
        )?;
        let edges = stmt
            .query_map([], |r| {
                Ok(MemoryEdge {
                    src_id: r.get(0)?,
                    dst_id: r.get(1)?,
                    edge_type: r.get(2)?,
                    weight: r.get(3)?,
                    co_count: r.get(4)?,
                    created_at: r.get(5)?,
                    last_activated: r.get(6)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(edges)
    }

    /// Neighbors of `id` via edges of `edge_type`, as `(other_id, weight)`.
    /// Edges are undirected, so both endpoints are checked.
    pub fn neighbors(&self, id: &str, edge_type: &str) -> Result<Vec<(String, f64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT CASE WHEN src_id = ?1 THEN dst_id ELSE src_id END, weight
             FROM memory_edges
             WHERE edge_type = ?2 AND (src_id = ?1 OR dst_id = ?1)",
        )?;
        let rows = stmt
            .query_map(params![id, edge_type], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        Ok(rows)
    }

    // ---- stats -----------------------------------------------------------

    pub fn stats(&self) -> Result<StoreStats> {
        let mut by_type = Vec::new();
        for mt in MemoryType::ALL {
            let count: i64 = self.conn.query_row(
                "SELECT COUNT(*) FROM memory_items WHERE memory_type = ?1",
                [mt.as_str()],
                |r| r.get(0),
            )?;
            by_type.push((mt.as_str().to_string(), count));
        }
        let total: i64 = self.conn.query_row("SELECT COUNT(*) FROM memory_items", [], |r| r.get(0))?;
        let stale: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memory_items WHERE stale = 1", [], |r| r.get(0))?;
        let pinned: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memory_items WHERE pinned = 1", [], |r| r.get(0))?;
        let raw_events: i64 = self.conn.query_row("SELECT COUNT(*) FROM raw_events", [], |r| r.get(0))?;
        let links: i64 = self.conn.query_row("SELECT COUNT(*) FROM memory_links", [], |r| r.get(0))?;
        let sessions: i64 = self.conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
        Ok(StoreStats { total_memories: total, by_type, stale, pinned, raw_events, links, sessions })
    }
}

/// One weighted association edge (e.g. `co_occurred`) from `memory_edges`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEdge {
    pub src_id: String,
    pub dst_id: String,
    pub edge_type: String,
    pub weight: f64,
    pub co_count: i64,
    pub created_at: i64,
    pub last_activated: i64,
}

/// Provenance of one summary memory created during consolidation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SummaryProvenance {
    pub summary_id: String,
    pub source_ids: Vec<String>,
}

/// One row of the `consolidation_runs` log table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationRun {
    pub id: String,
    pub ran_at: i64,
    pub duplicates_merged: i64,
    pub recurrence_updated: i64,
    pub clusters_summarized: i64,
    pub promoted: i64,
    pub marked_stale: i64,
    pub archived: i64,
    pub pruned: i64,
    pub summaries: Vec<SummaryProvenance>,
}

/// One row of the `chat_turns` metrics table.
#[derive(Debug, Clone, Serialize)]
pub struct ChatTurn {
    pub id: String,
    pub session_id: Option<String>,
    pub created_at: i64,
    pub user_text: String,
    pub assistant_text: String,
    pub model: String,
    pub backend: String,
    /// None when the backend didn't report token usage (estimates are the
    /// caller's job; NULLs keep the data honest).
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub total_duration_ms: Option<i64>,
    pub llm_duration_ms: Option<i64>,
    pub retrieve_duration_ms: i64,
    pub context_memory_count: i64,
    pub context_chars: i64,
    pub memory_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ChatMetricsSummary {
    pub turns: i64,
    pub avg_prompt_tokens: Option<f64>,
    pub avg_completion_tokens: Option<f64>,
    pub total_prompt_tokens: i64,
    pub total_completion_tokens: i64,
    pub avg_context_chars: Option<f64>,
    pub avg_context_memories: Option<f64>,
    pub avg_retrieve_ms: Option<f64>,
    pub avg_llm_ms: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct StoreStats {
    pub total_memories: i64,
    pub by_type: Vec<(String, i64)>,
    pub stale: i64,
    pub pinned: i64,
    pub raw_events: i64,
    pub links: i64,
    pub sessions: i64,
}

const MEMORY_COLUMNS: &str = "id, content, summary, memory_type, source, topic, entities, tags, \
    created_at, updated_at, last_accessed_at, access_count, importance_score, recurrence_score, \
    recency_score, decay_score, confidence, stale, pinned, embedding, content_hash";

fn row_to_memory(row: &Row<'_>) -> rusqlite::Result<MemoryItem> {
    let memory_type: String = row.get(3)?;
    let entities: Option<String> = row.get(6)?;
    let tags: Option<String> = row.get(7)?;
    let embedding: Option<Vec<u8>> = row.get(19)?;
    Ok(MemoryItem {
        id: row.get(0)?,
        content: row.get(1)?,
        summary: row.get(2)?,
        memory_type: MemoryType::from_str(&memory_type).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, e.into())
        })?,
        source: row.get(4)?,
        topic: row.get(5)?,
        entities: entities
            .map(|s| serde_json::from_str(&s).unwrap_or_default())
            .unwrap_or_default(),
        tags: tags
            .map(|s| serde_json::from_str(&s).unwrap_or_default())
            .unwrap_or_default(),
        created_at: row.get(8)?,
        updated_at: row.get(9)?,
        last_accessed_at: row.get(10)?,
        access_count: row.get(11)?,
        importance_score: row.get(12)?,
        recurrence_score: row.get(13)?,
        recency_score: row.get(14)?,
        decay_score: row.get(15)?,
        confidence: row.get(16)?,
        stale: row.get::<_, i64>(17)? != 0,
        pinned: row.get::<_, i64>(18)? != 0,
        embedding: embedding.map(|b| decode_embedding(&b)),
        content_hash: row.get(20)?,
    })
}

/// f32 vector -> little-endian byte BLOB.
pub fn encode_embedding(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Little-endian byte BLOB -> f32 vector.
pub fn decode_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use forgetfuldb_core::types::MemoryItem;

    fn sample_item(id: &str, content: &str) -> MemoryItem {
        let now = forgetfuldb_core::now_unix();
        let hash = forgetfuldb_core::ingest::content_hash(content);
        let mut item = MemoryItem::new(id.to_string(), content.to_string(), MemoryType::Episodic, hash, now);
        item.tags = vec!["project:test".to_string()];
        item.entities = vec!["billing".to_string()];
        item.embedding = Some(vec![0.1, 0.2, 0.3]);
        item
    }

    #[test]
    fn memory_roundtrip() {
        let store = Store::open_in_memory().unwrap();
        let item = sample_item("mem_1", "the billing system uses stripe");
        store.insert_memory(&item).unwrap();
        let back = store.get_memory("mem_1").unwrap().unwrap();
        assert_eq!(back.content, item.content);
        assert_eq!(back.memory_type, MemoryType::Episodic);
        assert_eq!(back.tags, item.tags);
        assert_eq!(back.embedding.unwrap().len(), 3);
    }

    #[test]
    fn duplicate_content_hash_rejected() {
        let store = Store::open_in_memory().unwrap();
        store.insert_memory(&sample_item("mem_1", "same text")).unwrap();
        let err = store.insert_memory(&sample_item("mem_2", "same text"));
        assert!(err.is_err(), "UNIQUE constraint should reject duplicate hash");
    }

    #[test]
    fn links_roundtrip() {
        let store = Store::open_in_memory().unwrap();
        store.insert_memory(&sample_item("mem_1", "fact one")).unwrap();
        store.insert_memory(&sample_item("mem_2", "fact two")).unwrap();
        store
            .insert_link(&MemoryLink {
                source_id: "mem_1".into(),
                target_id: "mem_2".into(),
                relation: LinkRelation::Updates,
            })
            .unwrap();
        let links = store.links_for("mem_2").unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].relation, LinkRelation::Updates);
    }

    #[test]
    fn cooccurrence_bump_is_additive_and_fast() {
        let store = Store::open_in_memory().unwrap();
        // A realistic store: 400 memories, plus a pre-existing edge web so
        // the bump isn't measured against an empty table.
        let ids: Vec<String> = (0..400).map(|i| format!("mem_{i:04}")).collect();
        for id in &ids {
            store.insert_memory(&sample_item(id, &format!("memory {id}"))).unwrap();
        }
        for chunk in ids.chunks(6).take(60) {
            store.bump_cooccurrence_edges(chunk, 1_000).unwrap();
        }

        // Additive: bumping the same pair twice doubles its weight.
        let pair = vec![ids[0].clone(), ids[1].clone()];
        store.bump_cooccurrence_edges(&pair, 2_000).unwrap();
        store.bump_cooccurrence_edges(&pair, 3_000).unwrap();
        let w = store.neighbors(&ids[0], "co_occurred").unwrap();
        let weight = w.iter().find(|(n, _)| n == &ids[1]).map(|(_, w)| *w).unwrap();
        assert!(weight >= 2.0, "two extra bumps should add to the weight, got {weight}");

        // Latency: a typical turn injects ~6 memories (15 pairs). Time many
        // such bumps and assert the per-bump cost is tiny (it runs on the
        // background writer, but cheap is still the point).
        let turn: Vec<String> = ids[10..16].to_vec();
        let iters = 200;
        let t0 = std::time::Instant::now();
        for k in 0..iters {
            store.bump_cooccurrence_edges(&turn, 4_000 + k).unwrap();
        }
        let per = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("co-occurrence bump (6 ids / 15 pairs): {per:.3} ms/turn over {iters} iters");
        assert!(per < 5.0, "bump should be well under 5ms/turn, was {per:.3} ms");
    }

    #[test]
    fn meta_roundtrip_and_set_embedding() {
        let store = Store::open_in_memory().unwrap();
        assert_eq!(store.get_meta("embedding_dim").unwrap(), None);
        store.set_meta("embedding_dim", "768").unwrap();
        store.set_meta("embedding_dim", "1024").unwrap(); // upsert
        assert_eq!(store.get_meta("embedding_dim").unwrap().as_deref(), Some("1024"));

        store.insert_memory(&sample_item("mem_1", "fact")).unwrap();
        store.set_embedding("mem_1", &[0.1, 0.2, 0.3, 0.4]).unwrap();
        let m = store.get_memory("mem_1").unwrap().unwrap();
        assert_eq!(m.embedding.unwrap().len(), 4, "embedding swapped, dimension changed");
    }

    #[test]
    fn touch_updates_access_metadata() {
        let store = Store::open_in_memory().unwrap();
        store.insert_memory(&sample_item("mem_1", "fact")).unwrap();
        store.touch_memory("mem_1", 12345).unwrap();
        let back = store.get_memory("mem_1").unwrap().unwrap();
        assert_eq!(back.access_count, 1);
        assert_eq!(back.last_accessed_at, Some(12345));
    }

    #[test]
    fn chat_turn_metrics_roundtrip() {
        let store = Store::open_in_memory().unwrap();
        let turn = ChatTurn {
            id: "turn_1".into(),
            session_id: Some("s1".into()),
            created_at: 1000,
            user_text: "hello".into(),
            assistant_text: "hi".into(),
            model: "gemma3:12b".into(),
            backend: "ollama".into(),
            prompt_tokens: Some(120),
            completion_tokens: Some(30),
            total_duration_ms: Some(900),
            llm_duration_ms: Some(800),
            retrieve_duration_ms: 4,
            context_memory_count: 3,
            context_chars: 250,
            memory_ids: vec!["mem_a".into()],
        };
        store.insert_chat_turn(&turn).unwrap();
        let summary = store.chat_metrics_summary().unwrap();
        assert_eq!(summary.turns, 1);
        assert_eq!(summary.total_prompt_tokens, 120);
        assert_eq!(summary.avg_completion_tokens, Some(30.0));
        assert_eq!(summary.avg_context_memories, Some(3.0));
    }

    #[test]
    fn embedding_codec_roundtrip() {
        let v = vec![0.5f32, -1.25, 3.75];
        assert_eq!(decode_embedding(&encode_embedding(&v)), v);
    }
}
