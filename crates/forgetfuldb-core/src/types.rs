//! The memory schema: strongly typed enums and records that mirror the
//! SQLite tables in `forgetfuldb-store`.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// What kind of memory an item is. Stored as a lowercase snake_case string
/// in SQLite so rows stay human-readable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    /// Verbatim event (chat turn, log line). Decays fast.
    RawEvent,
    /// "What happened" — an experience tied to a moment in time.
    Episodic,
    /// "What is true" — distilled facts, decays slowly.
    Semantic,
    /// "How to do things" — workflows and commands, decays slowly.
    Procedural,
    /// User preferences ("I like dark mode").
    Preference,
    /// Compressed/retired memory kept for the record, excluded from
    /// normal retrieval.
    Archive,
}

impl MemoryType {
    pub const ALL: [MemoryType; 6] = [
        MemoryType::RawEvent,
        MemoryType::Episodic,
        MemoryType::Semantic,
        MemoryType::Procedural,
        MemoryType::Preference,
        MemoryType::Archive,
    ];

    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryType::RawEvent => "raw_event",
            MemoryType::Episodic => "episodic",
            MemoryType::Semantic => "semantic",
            MemoryType::Procedural => "procedural",
            MemoryType::Preference => "preference",
            MemoryType::Archive => "archive",
        }
    }
}

impl FromStr for MemoryType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "raw_event" => Ok(MemoryType::RawEvent),
            "episodic" => Ok(MemoryType::Episodic),
            "semantic" => Ok(MemoryType::Semantic),
            "procedural" => Ok(MemoryType::Procedural),
            "preference" => Ok(MemoryType::Preference),
            "archive" => Ok(MemoryType::Archive),
            other => Err(format!("unknown memory type: {other}")),
        }
    }
}

impl fmt::Display for MemoryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Typed relation between two memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkRelation {
    Supports,
    Contradicts,
    Updates,
    Duplicates,
    DerivedFrom,
    BelongsToProject,
}

impl LinkRelation {
    pub fn as_str(&self) -> &'static str {
        match self {
            LinkRelation::Supports => "supports",
            LinkRelation::Contradicts => "contradicts",
            LinkRelation::Updates => "updates",
            LinkRelation::Duplicates => "duplicates",
            LinkRelation::DerivedFrom => "derived_from",
            LinkRelation::BelongsToProject => "belongs_to_project",
        }
    }
}

impl FromStr for LinkRelation {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "supports" => Ok(LinkRelation::Supports),
            "contradicts" => Ok(LinkRelation::Contradicts),
            "updates" => Ok(LinkRelation::Updates),
            "duplicates" => Ok(LinkRelation::Duplicates),
            "derived_from" => Ok(LinkRelation::DerivedFrom),
            "belongs_to_project" => Ok(LinkRelation::BelongsToProject),
            other => Err(format!("unknown link relation: {other}")),
        }
    }
}

impl fmt::Display for LinkRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One durable memory. Mirrors the `memory_items` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: String,
    pub content: String,
    pub summary: Option<String>,
    pub memory_type: MemoryType,
    pub source: Option<String>,
    pub topic: Option<String>,
    pub entities: Vec<String>,
    pub tags: Vec<String>,
    pub created_at: i64,
    pub updated_at: i64,
    pub last_accessed_at: Option<i64>,
    pub access_count: i64,
    pub importance_score: f64,
    pub recurrence_score: f64,
    pub recency_score: f64,
    pub decay_score: f64,
    /// How strongly this memory resists forgetting — U-shaped over novelty
    /// (high for both surprising and habitual memories). Distinct axis from
    /// decay: a salient memory can be old and untouched yet survive. Set
    /// provisionally at ingest, recomputed authoritatively at consolidation.
    #[serde(default)]
    pub salience: f64,
    pub confidence: f64,
    pub stale: bool,
    pub pinned: bool,
    /// Embedding vector, stored as a little-endian f32 BLOB in SQLite.
    /// Skipped in JSON output to keep context packs compact.
    #[serde(skip_serializing)]
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    pub content_hash: String,
}

impl MemoryItem {
    /// A fresh item with sane defaults; callers fill in scores.
    pub fn new(id: String, content: String, memory_type: MemoryType, content_hash: String, now: i64) -> Self {
        MemoryItem {
            id,
            content,
            summary: None,
            memory_type,
            source: None,
            topic: None,
            entities: Vec::new(),
            tags: Vec::new(),
            created_at: now,
            updated_at: now,
            last_accessed_at: None,
            access_count: 0,
            importance_score: 0.5,
            recurrence_score: 0.0,
            recency_score: 1.0,
            decay_score: 0.5,
            salience: 0.0,
            confidence: 1.0,
            stale: false,
            pinned: false,
            embedding: None,
            content_hash,
        }
    }
}

/// Mirrors the `memory_links` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLink {
    pub source_id: String,
    pub target_id: String,
    pub relation: LinkRelation,
}

/// Mirrors the `raw_events` table: the untouched input log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawEvent {
    pub id: String,
    pub session_id: Option<String>,
    pub role: Option<String>,
    pub content: String,
    pub created_at: i64,
    pub content_hash: String,
}

/// Mirrors the `sessions` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub title: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}
