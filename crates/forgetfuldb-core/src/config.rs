//! `forgetfuldb.toml` configuration model with serde defaults, so a
//! partial (or absent) config file always yields a working setup.

use crate::decay::DecayLambdas;
use crate::scoring::RetrievalWeights;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Path to the SQLite database file.
    pub sqlite_path: String,
    /// Embedding backend name. v1 ships `hashed_bow` (deterministic,
    /// model-free). Future: `fastembed`, `candle`, `llama_cpp`, `coreml`.
    pub embedding_backend: String,
    /// Dimensionality of placeholder embeddings.
    pub embedding_dim: usize,
    pub decay_lambda_raw: f64,
    pub decay_lambda_episodic: f64,
    pub decay_lambda_semantic: f64,
    pub decay_lambda_procedural: f64,
    pub decay_lambda_preference: f64,
    pub retrieval_weights: RetrievalWeights,
    pub consolidation_thresholds: ConsolidationThresholds,
    /// Raw events older than this become archive memories.
    pub archive_after_days: f64,
    /// Archived, unpinned memories older than this are deleted.
    pub delete_after_days: f64,
    /// ForgetfulDB never talks to the network when true (the default).
    pub local_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ConsolidationThresholds {
    /// Cosine similarity at or above which two memories are merged as
    /// duplicates.
    pub duplicate_similarity: f64,
    /// Minimum cluster size before a topic cluster gets a summary memory.
    pub cluster_min_size: usize,
    /// Access count at which an episodic memory is promoted to semantic.
    pub promote_min_access_count: i64,
    /// Decay score below which an old raw event is archived.
    pub archive_max_decay: f64,
    /// How many pruned raw events to keep as a representative sample
    /// (reservoir sampling) when deleting.
    pub prune_sample_size: usize,
}

impl Default for ConsolidationThresholds {
    fn default() -> Self {
        ConsolidationThresholds {
            duplicate_similarity: 0.92,
            cluster_min_size: 3,
            promote_min_access_count: 3,
            archive_max_decay: 0.05,
            prune_sample_size: 5,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        let lambdas = DecayLambdas::default();
        Config {
            sqlite_path: "forgetfuldb.sqlite3".to_string(),
            embedding_backend: "hashed_bow".to_string(),
            embedding_dim: 256,
            decay_lambda_raw: lambdas.raw_event,
            decay_lambda_episodic: lambdas.episodic,
            decay_lambda_semantic: lambdas.semantic,
            decay_lambda_procedural: lambdas.procedural,
            decay_lambda_preference: lambdas.preference,
            retrieval_weights: RetrievalWeights::default(),
            consolidation_thresholds: ConsolidationThresholds::default(),
            archive_after_days: 14.0,
            delete_after_days: 90.0,
            local_only: true,
        }
    }
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Config> {
        let raw = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&raw)?)
    }

    /// Load `path` if it exists, otherwise return defaults.
    pub fn load_or_default(path: &Path) -> anyhow::Result<Config> {
        if path.exists() {
            Config::load(path)
        } else {
            Ok(Config::default())
        }
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        std::fs::write(path, toml::to_string_pretty(self)?)?;
        Ok(())
    }

    /// Decay lambdas assembled from the individual config fields.
    pub fn decay_lambdas(&self) -> DecayLambdas {
        DecayLambdas {
            raw_event: self.decay_lambda_raw,
            episodic: self.decay_lambda_episodic,
            semantic: self.decay_lambda_semantic,
            procedural: self.decay_lambda_procedural,
            preference: self.decay_lambda_preference,
            archive: self.decay_lambda_raw,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_roundtrips_through_toml() {
        let cfg = Config::default();
        let text = toml::to_string_pretty(&cfg).unwrap();
        let back: Config = toml::from_str(&text).unwrap();
        assert_eq!(back.sqlite_path, cfg.sqlite_path);
        assert!(back.local_only);
    }

    #[test]
    fn partial_toml_uses_defaults() {
        let cfg: Config = toml::from_str("sqlite_path = \"/tmp/x.sqlite3\"").unwrap();
        assert_eq!(cfg.sqlite_path, "/tmp/x.sqlite3");
        assert_eq!(cfg.embedding_backend, "hashed_bow");
    }
}
