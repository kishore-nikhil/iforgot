//! iforgot — casual terminal chat with a local LLM (Ollama / llama-server)
//! where memory updates itself: every message is ingested, every reply is
//! grounded in retrieved memories, and per-turn token metrics are logged
//! for context optimization.

use anyhow::Result;
use forgetfuldb_agent::{Agent, TurnResult};
use forgetfuldb_consolidate::ExtractiveSummarizer;
use forgetfuldb_core::config::Config;
use rustyline::error::ReadlineError;
use std::io::{IsTerminal, Write};
use std::path::PathBuf;

const CYAN: &str = "\x1b[36m";
const MAGENTA: &str = "\x1b[35m";
const GREEN: &str = "\x1b[32m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

const LOGO: &str = r#"
   _ _____                          _
  (_)  ___|__  _ __ __ _  ___  ___ | |_
  | | |_ / _ \| '__/ _` |/ _ \/ _ \| __|
  | |  _| (_) | |  | (_| | (_) | (_) | |_
  |_|_|  \___/|_|   \__, |\___/ \___/ \__|
                    |___/
"#;

/// Value of a `--flag value` argument, if present.
fn arg_value(flag: &str) -> Option<String> {
    std::env::args().skip_while(|a| a != flag).nth(1)
}

fn main() -> Result<()> {
    let config_path = arg_value("--config").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("forgetfuldb.toml"));
    let cfg = Config::load_or_default(&config_path)?;

    let runtime = tokio::runtime::Runtime::new()?;
    let mut agent = Agent::new(cfg)?;

    let color = std::io::stdout().is_terminal();
    let paint = move |code: &'static str| -> &'static str { if color { code } else { "" } };

    println!("{}{}{}", paint(CYAN), LOGO, paint(RESET));

    let mut editor = rustyline::DefaultEditor::new()?;

    // `--model name` sets and persists the model directly.
    if let Some(name) = arg_value("--model") {
        agent.set_model(&name, &config_path)?;
        println!("{}  model set to {name} (saved to {}){}", paint(DIM), config_path.display(), paint(RESET));
    }

    // No model selected, or the configured one isn't installed -> let the
    // user pick from what the backend actually has, and save the choice.
    match runtime.block_on(agent.backend.list_models()) {
        Ok(installed) => {
            let current = agent.backend.model().to_string();
            let missing = !current.is_empty()
                && !installed.iter().any(|m| forgetfuldb_agent::backend::model_matches(m, &current));
            if current.is_empty() || missing {
                if missing {
                    println!("{}  configured model '{current}' is not installed on the backend{}", paint(MAGENTA), paint(RESET));
                } else {
                    println!("{}  no model selected{}", paint(MAGENTA), paint(RESET));
                }
                match pick_model(&installed, &mut editor, paint)? {
                    Some(choice) => {
                        agent.set_model(&choice, &config_path)?;
                        println!("{}  model set to {choice} (saved to {}){}", paint(DIM), config_path.display(), paint(RESET));
                    }
                    None => anyhow::bail!(
                        "no model selected. Pull one (e.g. `ollama pull gemma3:12b`) and rerun, \
                         or pass --model <name>"
                    ),
                }
            }
        }
        Err(_) => {
            println!(
                "{}  warning: no LLM server answering at {} — start it (e.g. `ollama serve`) and try again{}",
                paint(MAGENTA),
                agent.backend.base_url(),
                paint(RESET)
            );
            anyhow::ensure!(
                !agent.backend.model().is_empty(),
                "no model selected and the LLM server is unreachable, so there is nothing to pick from. \
                 Start the server, or pass --model <name>"
            );
        }
    }

    println!(
        "{}  model {} via {} at {} | db {} | /help for commands{}",
        paint(DIM),
        agent.backend.model(),
        agent.backend.name(),
        agent.backend.base_url(),
        agent.cfg.sqlite_path,
        paint(RESET)
    );
    println!();
    let mut last_turn: Option<TurnResult> = None;

    loop {
        let line = match editor.readline(&format!("{}you ❯ {}", paint(GREEN), paint(RESET))) {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(e) => return Err(e.into()),
        };
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        let _ = editor.add_history_entry(input);

        if let Some(cmd) = input.strip_prefix('/') {
            if !handle_command(cmd, &mut agent, &last_turn, &runtime, &config_path, &mut editor, paint)? {
                break;
            }
            continue;
        }

        print!("{}iforgot ❯ {}", paint(MAGENTA), paint(RESET));
        std::io::stdout().flush()?;
        let result = runtime.block_on(agent.chat_turn(input, &mut |tok: &str| {
            print!("{tok}");
            let _ = std::io::stdout().flush();
        }));
        println!();
        match result {
            Ok(turn) => {
                let t = &turn.turn;
                println!(
                    "{}  ⏺ {} memories | prompt {} tok | reply {} tok | retrieve {}ms | llm {}ms{}",
                    paint(DIM),
                    t.context_memory_count,
                    t.prompt_tokens.map_or("?".into(), |v| v.to_string()),
                    t.completion_tokens.map_or("?".into(), |v| v.to_string()),
                    t.retrieve_duration_ms,
                    t.llm_duration_ms.map_or("?".into(), |v| v.to_string()),
                    paint(RESET)
                );
                println!();
                last_turn = Some(turn);
            }
            Err(e) => println!("{}  error: {e:#}{}\n", paint(MAGENTA), paint(RESET)),
        }
    }
    agent.flush(); // drain background memory writes before claiming safety
    println!("{}bye — your memories are safe in {}{}", paint(DIM), agent.cfg.sqlite_path, paint(RESET));
    Ok(())
}

/// Numbered model picker. Returns None on empty list or EOF.
fn pick_model(
    installed: &[String],
    editor: &mut rustyline::DefaultEditor,
    paint: impl Fn(&'static str) -> &'static str,
) -> Result<Option<String>> {
    if installed.is_empty() {
        println!("  the backend reports no installed models");
        return Ok(None);
    }
    println!("  models installed on the backend:");
    for (i, name) in installed.iter().enumerate() {
        println!("    {}{}{}. {name}", paint(CYAN), i + 1, paint(RESET));
    }
    loop {
        let line = match editor.readline(&format!("  select a model [1-{}] ❯ ", installed.len())) {
            Ok(line) => line,
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if let Ok(n) = input.parse::<usize>() {
            if (1..=installed.len()).contains(&n) {
                return Ok(Some(installed[n - 1].clone()));
            }
        }
        if let Some(name) = installed.iter().find(|m| forgetfuldb_agent::backend::model_matches(m, input)) {
            return Ok(Some(name.clone()));
        }
        println!("  enter a number 1-{} or an exact model name", installed.len());
    }
}

/// Returns false when the user asked to quit.
#[allow(clippy::too_many_arguments)]
fn handle_command(
    cmd: &str,
    agent: &mut Agent,
    last_turn: &Option<TurnResult>,
    runtime: &tokio::runtime::Runtime,
    config_path: &std::path::Path,
    editor: &mut rustyline::DefaultEditor,
    paint: impl Fn(&'static str) -> &'static str + Copy,
) -> Result<bool> {
    // Slash commands inspect the database, so settle any memory writes
    // still queued on the background writer first (milliseconds).
    agent.flush();
    let mut parts = cmd.split_whitespace();
    match (parts.next().unwrap_or(""), parts.next()) {
        ("quit", _) | ("exit", _) => return Ok(false),
        ("help", _) => {
            println!("  /model [name]    list installed models, or switch (and save) the model");
            println!("  /memories        show the memories behind the last answer (with scores)");
            println!("  /stats           memory database statistics");
            println!("  /metrics         token & context metrics across all chat turns");
            println!("  /consolidate     run the sleep cycle (dedup, summarize, promote, prune)");
            println!("  /pin <id>        pin a memory so it never decays");
            println!("  /unpin <id>      unpin a memory");
            println!("  /stale <id>      mark a memory stale (hidden from retrieval)");
            println!("  /inspect <id>    dump one memory as JSON");
            println!("  /quit            leave (history stays in the database)");
        }
        ("model", arg) => {
            let installed = runtime.block_on(agent.backend.list_models()).unwrap_or_default();
            match arg {
                Some(name) => {
                    let resolved = installed
                        .iter()
                        .find(|m| forgetfuldb_agent::backend::model_matches(m, name))
                        .cloned();
                    match (resolved, installed.is_empty()) {
                        (Some(name), _) => {
                            agent.set_model(&name, config_path)?;
                            println!("  model set to {name} (saved to {})", config_path.display());
                        }
                        // Backend unreachable: trust the user, still persist.
                        (None, true) => {
                            agent.set_model(name, config_path)?;
                            println!("  backend unreachable; model set to {name} unverified (saved)");
                        }
                        (None, false) => {
                            println!("  '{name}' is not installed. available: {}", installed.join(", "));
                        }
                    }
                }
                None => match pick_model(&installed, editor, paint)? {
                    Some(name) => {
                        agent.set_model(&name, config_path)?;
                        println!("  model set to {name} (saved to {})", config_path.display());
                    }
                    None => println!("  current model: {}", agent.backend.model()),
                },
            }
        }
        ("memories", _) => match last_turn {
            Some(turn) if !turn.pack.memories.is_empty() => {
                for m in &turn.pack.memories {
                    println!(
                        "  {:.3}  {}{}{}  [{}] {}",
                        m.score.total,
                        paint(DIM),
                        m.item.id,
                        paint(RESET),
                        m.item.memory_type,
                        m.item.content
                    );
                }
            }
            Some(_) => println!("  last turn used no memories"),
            None => println!("  no turns yet"),
        },
        ("stats", _) => {
            let s = agent.store.stats()?;
            println!("  memories {} | stale {} | pinned {} | raw events {} | links {}",
                s.total_memories, s.stale, s.pinned, s.raw_events, s.links);
            for (mt, n) in &s.by_type {
                if *n > 0 {
                    println!("    {mt}: {n}");
                }
            }
        }
        ("metrics", _) => {
            let m = agent.store.chat_metrics_summary()?;
            if m.turns == 0 {
                println!("  no chat turns recorded yet");
            } else {
                let fmt = |v: Option<f64>| v.map_or("?".to_string(), |x| format!("{x:.0}"));
                println!("  turns: {}", m.turns);
                println!("  prompt tokens : avg {} (total {})", fmt(m.avg_prompt_tokens), m.total_prompt_tokens);
                println!("  reply tokens  : avg {} (total {})", fmt(m.avg_completion_tokens), m.total_completion_tokens);
                println!("  context       : avg {} chars across {} memories/turn",
                    fmt(m.avg_context_chars), fmt(m.avg_context_memories));
                if let (Some(ctx), Some(prompt)) = (m.avg_context_chars, m.avg_prompt_tokens) {
                    // chars/4 ≈ tokens: rough context share of the prompt
                    let share = (ctx / 4.0) / prompt.max(1.0) * 100.0;
                    println!("  context share : ~{share:.0}% of prompt tokens");
                }
                println!("  latency       : retrieve avg {}ms | llm avg {}ms",
                    fmt(m.avg_retrieve_ms), fmt(m.avg_llm_ms));
            }
        }
        ("consolidate", _) => {
            let report = forgetfuldb_consolidate::consolidate(&agent.store, &ExtractiveSummarizer::default(), &agent.cfg)?;
            println!("  {}", serde_json::to_string(&report)?);
        }
        ("pin", Some(id)) => {
            anyhow::ensure!(agent.store.set_pinned(id, true)?, "memory not found: {id}");
            println!("  pinned {id}");
        }
        ("unpin", Some(id)) => {
            anyhow::ensure!(agent.store.set_pinned(id, false)?, "memory not found: {id}");
            println!("  unpinned {id}");
        }
        ("stale", Some(id)) => {
            anyhow::ensure!(agent.store.set_stale(id, true)?, "memory not found: {id}");
            println!("  marked stale {id}");
        }
        ("inspect", Some(id)) => match agent.store.get_memory(id)? {
            Some(item) => println!("{}", serde_json::to_string_pretty(&item)?),
            None => println!("  memory not found: {id}"),
        },
        (other, _) => println!("  unknown command /{other} — try /help"),
    }
    Ok(true)
}
