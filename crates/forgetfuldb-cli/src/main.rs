//! forgetfuldb — local-first AI memory database CLI.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use forgetfuldb_consolidate::ExtractiveSummarizer;
use forgetfuldb_core::config::Config;
use forgetfuldb_core::types::MemoryType;
use forgetfuldb_retrieve::RetrieveOptions;
use forgetfuldb_store::pipeline::{ingest, warm_bloom, IngestRequest};
use forgetfuldb_store::Store;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[derive(Parser)]
#[command(name = "forgetfuldb", version, about = "Local-first AI memory database with human-like forgetting")]
struct Cli {
    /// Path to forgetfuldb.toml. Default: ./forgetfuldb.toml if present,
    /// otherwise the shared global store in ~/.forgetfuldb/
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create a config file and initialize its database.
    /// Default: project-local in the current directory; --global targets
    /// the shared ~/.forgetfuldb/ store every session uses by default.
    Init {
        /// Friendly name for this memory database (default: the
        /// directory name, or "main" for --global)
        #[arg(long)]
        name: Option<String>,
        /// Initialize the global store instead of a local one
        #[arg(long)]
        global: bool,
    },
    /// Store a new memory
    Ingest {
        #[arg(long)]
        text: String,
        #[arg(long)]
        source: Option<String>,
        /// Repeatable, e.g. --tag project:plotperfect --tag billing
        #[arg(long = "tag")]
        tags: Vec<String>,
        /// raw_event | episodic | semantic | procedural | preference
        #[arg(long)]
        memory_type: Option<String>,
        #[arg(long)]
        session: Option<String>,
        #[arg(long)]
        role: Option<String>,
    },
    /// Retrieve a context pack for a query
    Retrieve {
        #[arg(long)]
        query: String,
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        /// Include stale memories (off by default)
        #[arg(long)]
        include_stale: bool,
    },
    /// Run a consolidation pass (dedup, summarize, promote, archive)
    Consolidate,
    /// Show database statistics
    Stats,
    /// Show chat token/context metrics (recorded by iforgot chat & the proxy)
    Metrics,
    /// Show one memory with its score fields and links
    Inspect {
        #[arg(long)]
        id: String,
    },
    /// Move a memory to the archive type
    Archive {
        #[arg(long)]
        id: String,
    },
    /// Pin a memory so it never decays (or unpin with --off)
    Pin {
        #[arg(long)]
        id: String,
        #[arg(long)]
        off: bool,
    },
    /// Run the local HTTP API
    Server {
        #[arg(long, default_value_t = 8787)]
        port: u16,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if let Command::Init { name, global } = &cli.command {
        return init(cli.config.as_deref(), name.clone(), *global);
    }

    let resolved = forgetfuldb_core::config::resolve(cli.config.as_deref())?;
    if resolved.stray_local_db {
        eprintln!(
            "note: a {} exists in this directory, but the {} memory \"{}\" is in use \
             (run `forgetfuldb init` here to adopt the local one)",
            forgetfuldb_core::config::DB_FILE,
            resolved.scope.as_str(),
            resolved.config.name
        );
    }
    if resolved.home_local_config {
        eprintln!(
            "warning: using a {} found in your HOME directory, not the shared global store \
             (~/.forgetfuldb/). Move or delete {} if this is unintentional.",
            forgetfuldb_core::config::CONFIG_FILE,
            resolved.path.display()
        );
    }
    let cfg = resolved.config;

    match cli.command {
        Command::Init { .. } => unreachable!("handled above"),
        Command::Ingest { text, source, tags, memory_type, session, role } => {
            let store = open_store(&cfg)?;
            let mut bloom = warm_bloom(&store)?;
            let provider = forgetfuldb_embed::create_provider(&cfg.embedding_backend, cfg.embedding_dim)?;
            let memory_type = memory_type
                .as_deref()
                .map(MemoryType::from_str)
                .transpose()
                .map_err(|e| anyhow::anyhow!(e))?;
            let outcome = ingest(
                &store,
                &mut bloom,
                provider.as_ref(),
                &cfg,
                IngestRequest { text, source, tags, memory_type, session_id: session, role },
            )?;
            let verdict = if outcome.is_duplicate() { "reinforced existing" } else { "stored new" };
            println!("{verdict} memory {}", outcome.memory().id);
            println!("{}", serde_json::to_string_pretty(outcome.memory())?);
            Ok(())
        }
        Command::Retrieve { query, top_k, include_stale } => {
            let store = open_store(&cfg)?;
            let provider = forgetfuldb_embed::create_provider(&cfg.embedding_backend, cfg.embedding_dim)?;
            let opts = RetrieveOptions { top_k, include_stale, ..Default::default() };
            let pack = forgetfuldb_retrieve::retrieve(&store, provider.as_ref(), &cfg, &query, &opts)?;
            println!("{}", serde_json::to_string_pretty(&pack)?);
            Ok(())
        }
        Command::Consolidate => {
            let store = open_store(&cfg)?;
            let report = forgetfuldb_consolidate::consolidate(&store, &ExtractiveSummarizer::default(), &cfg)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::Stats => {
            let store = open_store(&cfg)?;
            let stats = store.stats()?;
            println!("memory name    : {}", cfg.name);
            println!("database       : {}", cfg.sqlite_path);
            println!("total memories : {}", stats.total_memories);
            for (mt, count) in &stats.by_type {
                println!("  {mt:<11}: {count}");
            }
            println!("stale          : {}", stats.stale);
            println!("pinned         : {}", stats.pinned);
            println!("raw events     : {}", stats.raw_events);
            println!("links          : {}", stats.links);
            println!("sessions       : {}", stats.sessions);
            Ok(())
        }
        Command::Metrics => {
            let store = open_store(&cfg)?;
            let m = store.chat_metrics_summary()?;
            if m.turns == 0 {
                println!("no chat turns recorded yet (use `iforgot` or the /v1 proxy)");
                return Ok(());
            }
            let fmt = |v: Option<f64>| v.map_or("?".to_string(), |x| format!("{x:.0}"));
            println!("chat turns      : {}", m.turns);
            println!("prompt tokens   : avg {} (total {})", fmt(m.avg_prompt_tokens), m.total_prompt_tokens);
            println!("reply tokens    : avg {} (total {})", fmt(m.avg_completion_tokens), m.total_completion_tokens);
            println!("context         : avg {} chars, avg {} memories/turn", fmt(m.avg_context_chars), fmt(m.avg_context_memories));
            println!("latency         : retrieve avg {} ms, llm avg {} ms", fmt(m.avg_retrieve_ms), fmt(m.avg_llm_ms));
            Ok(())
        }
        Command::Inspect { id } => {
            let store = open_store(&cfg)?;
            let item = store.get_memory(&id)?.with_context(|| format!("memory not found: {id}"))?;
            let links = store.links_for(&id)?;
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "memory": item, "links": links }))?);
            Ok(())
        }
        Command::Archive { id } => {
            let store = open_store(&cfg)?;
            anyhow::ensure!(store.set_memory_type(&id, MemoryType::Archive)?, "memory not found: {id}");
            println!("archived {id}");
            Ok(())
        }
        Command::Pin { id, off } => {
            let store = open_store(&cfg)?;
            anyhow::ensure!(store.set_pinned(&id, !off)?, "memory not found: {id}");
            println!("{} {id}", if off { "unpinned" } else { "pinned" });
            Ok(())
        }
        Command::Server { port } => {
            let runtime = tokio::runtime::Runtime::new()?;
            runtime.block_on(forgetfuldb_server::serve(cfg, port))
        }
    }
}

fn open_store(cfg: &Config) -> Result<Store> {
    Store::open(Path::new(&cfg.sqlite_path))
        .with_context(|| "is the database initialized? run `forgetfuldb init` first")
}

fn init(explicit: Option<&Path>, name: Option<String>, global: bool) -> Result<()> {
    use forgetfuldb_core::config::{home_dir, CONFIG_FILE, DB_FILE};

    let (config_path, default_name, absolute_db) = if let Some(path) = explicit {
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(|p| p.to_path_buf())
            .unwrap_or(std::env::current_dir()?);
        (path.to_path_buf(), dir_name(&parent), false)
    } else if global {
        let dir = home_dir()
            .context("cannot determine home directory (HOME unset)")?
            .join(".forgetfuldb");
        std::fs::create_dir_all(&dir)?;
        (dir.join(CONFIG_FILE), "main".to_string(), true)
    } else {
        let cwd = std::env::current_dir()?;
        (cwd.join(CONFIG_FILE), dir_name(&cwd), false)
    };

    if config_path.exists() {
        let mut cfg = Config::load(&config_path)?;
        if let Some(name) = name {
            cfg.name = name;
            cfg.save(&config_path)?;
            println!("renamed memory to \"{}\" in {}", cfg.name, config_path.display());
        } else {
            println!("config exists at {} (memory \"{}\")", config_path.display(), cfg.name);
        }
    } else {
        let mut cfg = Config { name: name.unwrap_or(default_name), ..Config::default() };
        if absolute_db {
            // The global store must never depend on the launch directory.
            cfg.sqlite_path = config_path
                .parent()
                .expect("global config has a parent dir")
                .join(DB_FILE)
                .display()
                .to_string();
        }
        cfg.save(&config_path)?;
        println!("wrote {} (memory \"{}\")", config_path.display(), cfg.name);
    }

    // Re-resolve through the normal path so the database lands exactly
    // where every other command will look for it.
    let resolved = forgetfuldb_core::config::resolve(Some(&config_path))?;
    Store::open(Path::new(&resolved.config.sqlite_path))?;
    println!("database ready at {}", resolved.config.sqlite_path);
    Ok(())
}

fn dir_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "main".to_string())
}
