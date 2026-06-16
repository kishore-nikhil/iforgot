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
    /// Embedding backend name. `hashed_bow` (default, deterministic,
    /// model-free, offline) or `ollama` (a real local embedding model).
    pub embedding_backend: String,
    /// Dimensionality of `hashed_bow` placeholder embeddings. Ignored by
    /// `ollama` (the model fixes its own dimension, probed at startup).
    pub embedding_dim: usize,
    /// Ollama embedding model name when `embedding_backend = "ollama"`
    /// (e.g. "embeddinggemma", "nomic-embed-text"). Empty otherwise.
    #[serde(default)]
    pub embedding_model: String,
    /// Base URL of the Ollama server used for embeddings. Must be
    /// localhost while `local_only` is set.
    #[serde(default = "default_embedding_base_url")]
    pub embedding_base_url: String,
    pub decay_lambda_raw: f64,
    pub decay_lambda_episodic: f64,
    pub decay_lambda_semantic: f64,
    pub decay_lambda_procedural: f64,
    pub decay_lambda_preference: f64,
    /// Decay rate for Foundation traits. Defaults to 0 (never decays);
    /// `serde(default)` keeps configs written before the Foundation tier
    /// loadable.
    #[serde(default)]
    pub decay_lambda_foundation: f64,
    pub retrieval_weights: RetrievalWeights,
    pub consolidation_thresholds: ConsolidationThresholds,
    /// Enable spreading activation: after base scoring, boost memories
    /// associated (co-occurring in past chat turns) with the top hits, so
    /// retrieving one memory can surface its companions. Off by default so
    /// behavior is unchanged until opted in.
    #[serde(default)]
    pub spreading_activation: bool,
    /// Strength of the spreading-activation boost (0 disables).
    #[serde(default = "default_spreading_factor")]
    pub spreading_factor: f64,
    /// Per-day decay applied to a co-occurrence when summing edge weight,
    /// so recent shared turns matter more than old ones.
    #[serde(default = "default_edge_decay_lambda")]
    pub edge_decay_lambda: f64,
    /// Co-occurrence edges below this weight are pruned (kept the graph
    /// from filling with one-off pairings).
    #[serde(default = "default_edge_min_weight")]
    pub edge_min_weight: f64,
    /// How strongly salience resists decay, in `[0, 1]`. A fully-salient
    /// memory decays at `(1 - salience_resist)` of the base rate (0.7 →
    /// 30%). 0 disables salience-based decay resistance.
    #[serde(default = "default_salience_resist")]
    pub salience_resist: f64,
    /// Salience at/above which a memory is **kept** through pruning,
    /// regardless of decay — the automatic counterpart to a manual pin.
    /// This is how a formative memory survives the archiving that buries
    /// the routine around it. 1.0 effectively disables the hard keep.
    #[serde(default = "default_salience_keep_threshold")]
    pub salience_keep_threshold: f64,
    /// Minimum cosine for a `semantic_similar` edge (kNN in meaning-space).
    #[serde(default = "default_semantic_edge_min_sim")]
    pub semantic_edge_min_sim: f64,
    /// Nearest neighbors linked per memory for `semantic_similar` edges.
    #[serde(default = "default_semantic_edge_top_k")]
    pub semantic_edge_top_k: usize,
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
    /// How many recent raw user messages are folded into the *retrieval
    /// query* alongside the current one. A vague follow-up ("something
    /// catchier") carries no topic on its own; with context the query
    /// still knows what the conversation is about. Affects retrieval
    /// only — the prompt sent to the model is unchanged. 0 disables.
    pub query_context_turns: usize,
    /// Retrieval score below which a memory is NOT injected into the
    /// prompt, even if `top_k` isn't filled. An empty memory block is
    /// better than a misleading one. 0.0 disables the gate.
    pub min_retrieval_score: f64,
    /// Score multiplier in (0, 1] applied to verbatim conversational
    /// memories (chat-sourced raw events and episodic turns) during chat
    /// retrieval. Old conversations re-injected verbatim are the main way
    /// a chat gets hijacked onto a stale topic; distilled semantic /
    /// preference / procedural memories are unaffected. 1.0 disables.
    pub conversational_damping: f64,
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
/// Default Ollama URL for embeddings (same host the chat backend uses).
pub fn default_embedding_base_url() -> String {
    "http://127.0.0.1:11434".to_string()
}

fn default_spreading_factor() -> f64 {
    0.15
}
fn default_edge_decay_lambda() -> f64 {
    0.02 // ~35-day half-life: associations fade slowly
}
fn default_edge_min_weight() -> f64 {
    0.1
}
fn default_salience_resist() -> f64 {
    0.7
}
fn default_salience_keep_threshold() -> f64 {
    0.6
}
fn default_semantic_edge_min_sim() -> f64 {
    0.55
}
fn default_semantic_edge_top_k() -> usize {
    5
}

pub fn default_system_prompt() -> String {
    "You are iForgot, a local AI assistant for a software developer on macOS. You help \
     with coding and everyday tasks, you remember the user's long-term context, and you \
     can run shell commands through your tools. When the user asks you to DO something on \
     their machine (find their IP, list files, check a process, etc.), propose the shell \
     command using the tool format below — do not merely describe it. NEVER invent or \
     fabricate command output: you only learn a command's result after the tool runs and \
     its real output is given to you. The user always confirms before anything runs, so \
     you don't need disclaimers or warnings about running commands. Use the user's \
     memories when relevant, say so plainly when you don't know, keep answers concise, \
     and format them in Markdown. Retrieved memories are background from past sessions: \
     when a memory conflicts with what the user is saying in the live conversation, the \
     conversation always takes precedence."
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
            query_context_turns: 2,
            min_retrieval_score: 0.25,
            conversational_damping: 0.6,
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
    /// Promote a semantic/preference memory to a Foundation trait once its
    /// near-neighbors form a *habit* (the discriminator's class) spread over
    /// at least this fraction of the store's history — evidence of a
    /// long-standing pattern, not a recent flurry.
    pub foundation_min_temporal_spread: f64,
    /// …and only with at least this many near-neighbors, so a couple of
    /// repeats can't mint a lifelong trait.
    pub foundation_min_neighbors: usize,
    /// Collapse *bursts* — dense clusters of similar event memories packed
    /// into a tight time window — into a single gist while keeping the one
    /// outlier (the anomaly). The temporal inverse of Foundation promotion.
    pub burst_collapse_enabled: bool,
    /// Minimum cluster size before a burst is collapsed to a gist. Below this
    /// it's just a few related notes, not a flood worth compressing.
    pub burst_min_size: usize,
}

impl Default for ConsolidationThresholds {
    fn default() -> Self {
        ConsolidationThresholds {
            duplicate_similarity: 0.92,
            cluster_min_size: 3,
            promote_min_access_count: 3,
            archive_max_decay: 0.05,
            prune_sample_size: 5,
            foundation_min_temporal_spread: 0.5,
            foundation_min_neighbors: 4,
            burst_collapse_enabled: true,
            burst_min_size: 4,
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
            embedding_model: String::new(),
            embedding_base_url: default_embedding_base_url(),
            decay_lambda_raw: lambdas.raw_event,
            decay_lambda_episodic: lambdas.episodic,
            decay_lambda_semantic: lambdas.semantic,
            decay_lambda_procedural: lambdas.procedural,
            decay_lambda_preference: lambdas.preference,
            decay_lambda_foundation: lambdas.foundation,
            retrieval_weights: RetrievalWeights::default(),
            consolidation_thresholds: ConsolidationThresholds::default(),
            spreading_activation: false,
            spreading_factor: default_spreading_factor(),
            edge_decay_lambda: default_edge_decay_lambda(),
            edge_min_weight: default_edge_min_weight(),
            salience_resist: default_salience_resist(),
            salience_keep_threshold: default_salience_keep_threshold(),
            semantic_edge_min_sim: default_semantic_edge_min_sim(),
            semantic_edge_top_k: default_semantic_edge_top_k(),
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
            foundation: self.decay_lambda_foundation,
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
    /// True when a "project-local" `forgetfuldb.toml` was found in the
    /// home directory itself. Home is most terminals' starting cwd, not a
    /// project: such a config silently splits memories away from the
    /// global store (usually a leftover from the old cwd-relative era),
    /// so frontends should warn.
    pub home_local_config: bool,
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
            home_local_config: false,
        });
    }

    let local = cwd.join(CONFIG_FILE);
    if local.exists() {
        let mut config = Config::load(&local)?;
        anchor_sqlite_path(&mut config, cwd);
        return Ok(ResolvedConfig {
            config,
            path: local,
            scope: ConfigScope::Local,
            stray_local_db: false,
            home_local_config: cwd == home,
        });
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
        home_local_config: false,
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
    fn local_config_in_home_directory_is_flagged() {
        let home = temp_dir("home-is-cwd");
        let stray = Config { name: "main".to_string(), ..Config::default() };
        stray.save(&home.join(CONFIG_FILE)).unwrap();

        // Launched from home itself: the local config wins but is flagged.
        let resolved = resolve_from(None, &home, &home).unwrap();
        assert_eq!(resolved.scope, ConfigScope::Local);
        assert!(resolved.home_local_config);

        // A genuine project dir with its own config is not flagged.
        let project = temp_dir("project");
        stray.save(&project.join(CONFIG_FILE)).unwrap();
        let resolved = resolve_from(None, &project, &home).unwrap();
        assert_eq!(resolved.scope, ConfigScope::Local);
        assert!(!resolved.home_local_config);
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
