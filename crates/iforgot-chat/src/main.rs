//! iforgot — casual terminal chat with a local LLM (Ollama / llama-server)
//! where memory updates itself: every message is ingested, every reply is
//! grounded in retrieved memories, and per-turn token metrics are logged
//! for context optimization.

mod markdown;
mod spinner;

use anyhow::Result;
use forgetfuldb_agent::{Agent, TurnResult};
use forgetfuldb_consolidate::ExtractiveSummarizer;
use markdown::MarkdownStream;
use spinner::Spinner;
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
    let explicit = arg_value("--config").map(PathBuf::from);
    let resolved = forgetfuldb_core::config::resolve(explicit.as_deref())?;
    let config_path = resolved.path.clone();
    let scope = resolved.scope;
    let stray_local_db = resolved.stray_local_db;
    let home_local_config = resolved.home_local_config;

    let runtime = tokio::runtime::Runtime::new()?;
    let mut agent = Agent::new(resolved.config)?;

    let color = std::io::stdout().is_terminal();
    let paint = move |code: &'static str| -> &'static str { if color { code } else { "" } };

    println!("{}{}{}", paint(CYAN), LOGO, paint(RESET));
    println!(
        "{}  memory \"{}\" ({}) | db {}{}",
        paint(DIM),
        agent.cfg.name,
        scope.as_str(),
        agent.cfg.sqlite_path,
        paint(RESET)
    );
    if stray_local_db {
        println!(
            "{}  note: a {} exists in this directory from an earlier session, but the {} memory \
             \"{}\" is in use. To revive those memories run `forgetfuldb init` here and use --config, \
             or re-ingest what matters.{}",
            paint(MAGENTA),
            forgetfuldb_core::config::DB_FILE,
            scope.as_str(),
            agent.cfg.name,
            paint(RESET)
        );
    }
    if home_local_config {
        println!(
            "{}  warning: this session uses a {} found in your HOME directory, NOT the shared \
             global store (~/.forgetfuldb/). Sessions started elsewhere won't see these memories. \
             If that's unintentional, move or delete {} to fall back to the global store.{}",
            paint(MAGENTA),
            forgetfuldb_core::config::CONFIG_FILE,
            config_path.display(),
            paint(RESET)
        );
    }

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
        "{}  model {} via {} at {} | tools: {} | /help for commands{}",
        paint(DIM),
        agent.backend.model(),
        agent.backend.name(),
        agent.backend.base_url(),
        if agent.tool_list().is_empty() { "off" } else { "on" },
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
            if !handle_command(cmd, &mut agent, &last_turn, &runtime, &config_path, &mut editor, color, paint)? {
                break;
            }
            continue;
        }

        match run_turn(&mut agent, &runtime, &mut editor, input, color, paint) {
            Ok(turn) => last_turn = Some(turn),
            Err(e) => println!("{}  error: {e:#}{}\n", paint(MAGENTA), paint(RESET)),
        }
    }
    agent.flush(); // drain background memory writes before claiming safety
    println!(
        "{}bye — your memories are safe in \"{}\" ({}){}",
        paint(DIM),
        agent.cfg.name,
        agent.cfg.sqlite_path,
        paint(RESET)
    );
    Ok(())
}

/// Run a full conversational turn: stream the reply with live Markdown
/// formatting, then drive any tool calls the model proposes — showing the
/// command and running it only on confirmation, feeding the output back
/// for a follow-up answer, and repeating if the model chains another tool.
fn run_turn(
    agent: &mut Agent,
    runtime: &tokio::runtime::Runtime,
    editor: &mut rustyline::DefaultEditor,
    input: &str,
    color: bool,
    paint: impl Fn(&'static str) -> &'static str + Copy,
) -> Result<TurnResult> {
    let mut result = stream_reply(color, paint, |on_token| runtime.block_on(agent.chat_turn(input, on_token)))?;
    print_metrics(&result.turn, paint);

    while let Some(call) = result.pending_tool.clone() {
        let preview = agent.tool_preview(&call).unwrap_or_else(|| call.tool.clone());
        let approved = if agent.tool_requires_confirmation(&call) {
            println!(
                "{}  ⚙ {} wants to run:{}\n    {}{}{}",
                paint(MAGENTA),
                call.tool,
                paint(RESET),
                paint(CYAN),
                preview,
                paint(RESET)
            );
            let answer = editor.readline("  run it? [Enter/y = run, anything else = cancel] ❯ ")?;
            let a = answer.trim().to_lowercase();
            a.is_empty() || a == "y" || a == "yes"
        } else {
            true
        };

        let feedback = if approved {
            match agent.execute_tool(&call) {
                Ok(output) => {
                    println!("{}{}{}", paint(DIM), indent(&output), paint(RESET));
                    output
                }
                Err(e) => {
                    println!("{}  tool error: {e}{}", paint(MAGENTA), paint(RESET));
                    format!("error: {e}")
                }
            }
        } else {
            println!("{}  cancelled{}", paint(DIM), paint(RESET));
            "(the user declined to run this command)".to_string()
        };

        result = stream_reply(color, paint, |on_token| {
            runtime.block_on(agent.respond_to_tool(&call, &feedback, on_token))
        })?;
        print_metrics(&result.turn, paint);
    }
    Ok(result)
}

/// Print the assistant prefix, spin while waiting for the first token
/// (retrieval + model load + prompt eval), then stream tokens through the
/// Markdown formatter and flush the trailing partial line.
fn stream_reply(
    color: bool,
    paint: impl Fn(&'static str) -> &'static str,
    run: impl FnOnce(&mut dyn FnMut(&str)) -> Result<TurnResult>,
) -> Result<TurnResult> {
    print!("{}iforgot ❯ {}", paint(MAGENTA), paint(RESET));
    let _ = std::io::stdout().flush();
    let mut spinner = Spinner::start(color, paint(DIM), paint(RESET));
    let mut md = MarkdownStream::new(color);
    let mut awaiting_first_token = true;
    let result = run(&mut |tok: &str| {
        if awaiting_first_token {
            spinner.stop();
            awaiting_first_token = false;
        }
        print!("{}", md.push(tok));
        let _ = std::io::stdout().flush();
    });
    // Empty reply or backend error: the spinner is still running.
    spinner.stop();
    print!("{}", md.finish());
    println!();
    result
}

fn print_metrics(t: &forgetfuldb_store::ChatTurn, paint: impl Fn(&'static str) -> &'static str) {
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
}

/// Indent multi-line tool output so it reads as a block.
fn indent(text: &str) -> String {
    text.lines().map(|l| format!("    {l}")).collect::<Vec<_>>().join("\n")
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
    color: bool,
    paint: impl Fn(&'static str) -> &'static str + Copy,
) -> Result<bool> {
    // Slash commands inspect the database, so settle any memory writes
    // still queued on the background writer first (milliseconds).
    agent.flush();
    // Split into command word + raw remainder, so `/cmd` keeps spaces.
    let (head, rest) = match cmd.split_once(char::is_whitespace) {
        Some((h, r)) => (h, r.trim()),
        None => (cmd, ""),
    };
    let arg = if rest.is_empty() { None } else { Some(rest) };
    match (head, arg) {
        ("quit", _) | ("exit", _) => return Ok(false),
        ("help", _) => {
            println!("  /cmd <command>   run a shell command directly (e.g. /cmd ls -la)");
            println!("  /tools           list tools the assistant can request");
            println!("  /prompt          show the active system prompt");
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
            println!();
            println!("  {}Tip: just ask in plain English (\"show my IP\") and the assistant", paint(DIM));
            println!("  will propose a command you can approve.{}", paint(RESET));
        }
        ("cmd", Some(command)) | ("sh", Some(command)) | ("run", Some(command)) => {
            // You typed it explicitly, so it runs without a confirm prompt.
            match agent.execute_tool(&Agent::shell_call(command)) {
                Ok(output) => println!("{}{}{}", paint(DIM), indent(&output), paint(RESET)),
                Err(e) => println!("{}  {e}{}", paint(MAGENTA), paint(RESET)),
            }
        }
        ("cmd", None) | ("sh", None) | ("run", None) => {
            println!("  usage: /cmd <shell command>");
        }
        ("tools", _) => {
            let tools = agent.tool_list();
            if tools.is_empty() {
                println!("  tools are disabled (set tools.enabled = true in the config)");
            } else {
                for t in tools {
                    let confirm = if t.requires_confirmation { " (asks before running)" } else { "" };
                    println!("  {}{}{}: {} — args {}{confirm}", paint(CYAN), t.name, paint(RESET), t.description, t.usage);
                }
            }
        }
        ("prompt", _) => {
            let mut md = MarkdownStream::new(color);
            print!("{}", md.push(agent.cfg.chat.system_prompt.trim()));
            print!("{}", md.finish());
            println!();
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
