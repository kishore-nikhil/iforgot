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
    /// Path to forgetfuldb.toml (default: ./forgetfuldb.toml)
    #[arg(long, global = true, default_value = "forgetfuldb.toml")]
    config: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create the config file (if missing) and initialize the database
    Init,
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
    let cfg = Config::load_or_default(&cli.config)?;

    match cli.command {
        Command::Init => init(&cli.config, &cfg),
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
            let opts = RetrieveOptions { top_k, include_stale, include_archived: false };
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

fn init(config_path: &Path, cfg: &Config) -> Result<()> {
    if !config_path.exists() {
        cfg.save(config_path)?;
        println!("wrote {}", config_path.display());
    } else {
        println!("config exists at {}", config_path.display());
    }
    Store::open(Path::new(&cfg.sqlite_path))?;
    println!("database ready at {}", cfg.sqlite_path);
    Ok(())
}
