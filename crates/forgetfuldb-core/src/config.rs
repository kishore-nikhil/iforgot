//! `forgetfuldb.toml` configuration model with serde defaults, so a
//! partial (or absent) config file always yields a working setup.

use crate::decay::DecayLambdas;
use crate::scoring::RetrievalWeights;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Friendly name for this memory database, shown in banners and
    /// stats so it's always clear which store a session is talking to
    /// (e.g. "main" for the global store, a project name for local ones).
    pub name: String,
    /// Path to the SQLite database file. Relative paths are resolved
    /// against the directory containing the config file, never the
    /// process working directory.
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
    /// The chat backend URL must be localhost and the HTTP server binds
    /// 127.0.0.1.
    pub local_only: bool,
    /// Chat loop settings (used by forgetfuldb-agent / iforgot-chat).
    pub chat: ChatConfig,
    /// Tool integration settings (shell execution and future tools).
    pub tools: ToolsConfig,
}

/// Settings for the pluggable tool interface.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolsConfig {
    /// Master switch: may the assistant use tools at all?
    pub enabled: bool,
    /// Allow the built-in shell tool. Execution still requires
    /// per-command user confirmation in the CLI.
    pub shell_enabled: bool,
    /// Kill a shell command if it runs longer than this.
    pub shell_timeout_secs: u64,
    /// Let the HTTP server's `/tools/execute` actually run tools. Off by
    /// default: an HTTP endpoint can't ask a human to confirm, so this
    /// would be a remote shell. Enable only on a trusted local machine.
    pub allow_server_execute: bool,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        ToolsConfig {
            enabled: true,
            shell_enabled: true,
            shell_timeout_secs: 30,
            allow_server_execute: false,
        }
    }
}

/// Settings for the memory-wrapped chat loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChatConfig {
    /// "ollama" (native API, exact token metrics) or "openai_compat"
    /// (works with llama-server, LM Studio, anything OpenAI-shaped).
    pub backend: String,
    /// Base URL of the local LLM server.
    pub base_url: String,
    /// Model name to request. Empty means "not selected yet": the
    /// `iforgot` chat then lists the models installed on the backend and
    /// persists your choice here.
    pub model: String,
    /// How many memories to inject per turn.
    pub top_k: usize,
    /// How many past user/assistant exchanges to keep in the prompt.
    pub history_turns: usize,
    /// How long Ollama keeps the model loaded after a request (e.g.
    /// "30m", "1h", "-1" for forever). Avoids paying a full model reload
    /// after idle pauses. Ignored by openai_compat backends.
    pub keep_alive: String,
    /// Base system prompt; retrieved memories are appended to it.
    pub system_prompt: String,
}

/// The default persona: a developer's assistant that remembers, can run
/// AI-assisted shell commands, and is extensible with tools for private
/// local tasks. Override via `chat.system_prompt` in the config.
pub fn default_system_prompt() -> String {
    "You are iForgot, a local AI assistant for a software developer. You help with \
     coding and day-to-day tasks, you keep track of the user's long-term memories, and \
     you can run AI-assisted shell commands through your tools. Use the user's memories \
     when they are relevant, and say so plainly when you don't know something. Be \
     concise and practical. Format answers in Markdown."
        .to_string()
}

impl Default for ChatConfig {
    fn default() -> Self {
        ChatConfig {
            backend: "ollama".to_string(),
            base_url: "http://127.0.0.1:11434".to_string(),
            model: String::new(),
            top_k: 6,
            history_turns: 8,
            keep_alive: "30m".to_string(),
            system_prompt: default_system_prompt(),
        }
    }
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
            name: "main".to_string(),
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
            chat: ChatConfig::default(),
            tools: ToolsConfig::default(),
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

pub const CONFIG_FILE: &str = "forgetfuldb.toml";
pub const DB_FILE: &str = "forgetfuldb.sqlite3";

/// Where a resolved config came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigScope {
    /// `--config <path>` on the command line.
    Explicit,
    /// `./forgetfuldb.toml` in the working directory — a deliberate
    /// project-local memory store.
    Local,
    /// `~/.forgetfuldb/` — the shared store every session defaults to,
    /// so memories stay intact no matter where you launch from.
    Global,
}

impl ConfigScope {
    pub fn as_str(&self) -> &'static str {
        match self {
            ConfigScope::Explicit => "explicit",
            ConfigScope::Local => "local",
            ConfigScope::Global => "global",
        }
    }
}

/// A config plus where it lives and how it was found.
pub struct ResolvedConfig {
    pub config: Config,
    pub path: PathBuf,
    pub scope: ConfigScope,
    /// True when the global store is in use but a stray
    /// `forgetfuldb.sqlite3` sits in the working directory — likely
    /// orphaned memories from an older cwd-relative session worth
    /// telling the user about.
    pub stray_local_db: bool,
}

pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

/// Resolve the active config: `--config` flag, else `./forgetfuldb.toml`
/// if present, else the global `~/.forgetfuldb/` store (created on first
/// use). See [`resolve_from`] for the testable core.
pub fn resolve(explicit: Option<&Path>) -> anyhow::Result<ResolvedConfig> {
    let cwd = std::env::current_dir()?;
    let home = home_dir().ok_or_else(|| anyhow::anyhow!("cannot determine home directory (HOME unset)"))?;
    resolve_from(explicit, &cwd, &home)
}

pub fn resolve_from(explicit: Option<&Path>, cwd: &Path, home: &Path) -> anyhow::Result<ResolvedConfig> {
    if let Some(path) = explicit {
        let mut config = Config::load_or_default(path)?;
        anchor_sqlite_path(&mut config, path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(cwd));
        return Ok(ResolvedConfig {
            config,
            path: path.to_path_buf(),
            scope: ConfigScope::Explicit,
            stray_local_db: false,
        });
    }

    let local = cwd.join(CONFIG_FILE);
    if local.exists() {
        let mut config = Config::load(&local)?;
        anchor_sqlite_path(&mut config, cwd);
        return Ok(ResolvedConfig { config, path: local, scope: ConfigScope::Local, stray_local_db: false });
    }

    let dir = home.join(".forgetfuldb");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(CONFIG_FILE);
    let mut config = if path.exists() {
        Config::load(&path)?
    } else {
        // First use: write the global config with an ABSOLUTE database
        // path, so the store never depends on where a session starts.
        let c = Config {
            name: "main".to_string(),
            sqlite_path: dir.join(DB_FILE).display().to_string(),
            ..Config::default()
        };
        c.save(&path)?;
        c
    };
    anchor_sqlite_path(&mut config, &dir);
    Ok(ResolvedConfig {
        config,
        path,
        scope: ConfigScope::Global,
        stray_local_db: cwd.join(DB_FILE).exists(),
    })
}

/// Make a relative `sqlite_path` absolute against the config's own
/// directory, so the same config means the same database from any cwd.
fn anchor_sqlite_path(config: &mut Config, base: &Path) {
    let p = Path::new(&config.sqlite_path);
    if p.is_relative() {
        config.sqlite_path = base.join(p).display().to_string();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "forgetfuldb-cfg-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn global_store_is_created_with_absolute_db_path_and_name() {
        let cwd = temp_dir("cwd");
        let home = temp_dir("home");
        let resolved = resolve_from(None, &cwd, &home).unwrap();
        assert_eq!(resolved.scope, ConfigScope::Global);
        assert_eq!(resolved.config.name, "main");
        assert!(Path::new(&resolved.config.sqlite_path).is_absolute());
        assert!(resolved.config.sqlite_path.starts_with(home.join(".forgetfuldb").to_str().unwrap()));
        assert!(resolved.path.exists(), "global config persisted");
        assert!(!resolved.stray_local_db);

        // Second resolution loads the same store rather than recreating.
        let again = resolve_from(None, &cwd, &home).unwrap();
        assert_eq!(again.config.sqlite_path, resolved.config.sqlite_path);
    }

    #[test]
    fn local_config_wins_over_global_and_anchors_relative_db() {
        let cwd = temp_dir("cwd-local");
        let home = temp_dir("home-local");
        let local = Config { name: "myproject".to_string(), ..Config::default() };
        local.save(&cwd.join(CONFIG_FILE)).unwrap();

        let resolved = resolve_from(None, &cwd, &home).unwrap();
        assert_eq!(resolved.scope, ConfigScope::Local);
        assert_eq!(resolved.config.name, "myproject");
        // Relative sqlite_path anchored to the config's directory.
        assert_eq!(resolved.config.sqlite_path, cwd.join(DB_FILE).display().to_string());
    }

    #[test]
    fn stray_local_database_is_flagged() {
        let cwd = temp_dir("cwd-stray");
        let home = temp_dir("home-stray");
        std::fs::write(cwd.join(DB_FILE), b"old memories").unwrap();
        let resolved = resolve_from(None, &cwd, &home).unwrap();
        assert_eq!(resolved.scope, ConfigScope::Global);
        assert!(resolved.stray_local_db);
    }

    #[test]
    fn explicit_config_path_wins() {
        let cwd = temp_dir("cwd-exp");
        let home = temp_dir("home-exp");
        let path = cwd.join("custom.toml");
        let cfg = Config { name: "custom".to_string(), ..Config::default() };
        cfg.save(&path).unwrap();

        let resolved = resolve_from(Some(&path), &cwd, &home).unwrap();
        assert_eq!(resolved.scope, ConfigScope::Explicit);
        assert_eq!(resolved.config.name, "custom");
        assert!(!home.join(".forgetfuldb").exists(), "global store untouched");
    }

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
