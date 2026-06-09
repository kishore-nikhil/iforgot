//! forgetfuldb-core
//!
//! Pure, dependency-light building blocks shared by every other crate:
//!
//! - [`types`]: the memory schema (`MemoryItem`, `MemoryType`, links, ...)
//! - [`scoring`]: the hybrid retrieval scoring formula
//! - [`decay`]: exponential forgetting curves per memory type
//! - [`ingest`]: text normalization, hashing and importance heuristics
//! - [`config`]: the `forgetfuldb.toml` configuration model
//!
//! Nothing in this crate touches SQLite, the network, or an embedding
//! model — that keeps the "what is a memory and how is it scored" logic
//! trivially unit-testable.

pub mod config;
pub mod decay;
pub mod ids;
pub mod ingest;
pub mod scoring;
pub mod types;

/// Current unix timestamp in seconds.
pub fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Age in fractional days between two unix timestamps.
pub fn age_days(created_at: i64, now: i64) -> f64 {
    ((now - created_at).max(0) as f64) / 86_400.0
}
