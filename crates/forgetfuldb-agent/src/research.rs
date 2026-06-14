//! The research agent: point it at a folder and it explores autonomously
//! with **read-only** commands, then distills what it learned into
//! long-term memories.
//!
//! Safety model: commands are executed by
//! [`ExploreTool`](forgetfuldb_tools::ExploreTool), which validates every
//! command against a read-only allowlist (no `rm`, no redirection, no
//! script execution) and pins the working directory to the researched
//! folder — that's why, unlike normal chat tools, no per-command user
//! confirmation is needed.
//!
//! Loop shape: the model gets a research system prompt and replies either
//! with one ```tool block (an explore command — executed, output fed
//! back) or with its final report. The report ends with a fenced
//! ```remember block whose lines become semantic memories tagged
//! `project:<folder>`, so later sessions can answer questions about the
//! project from memory alone.

use crate::backend::ChatMessage;
use crate::{Agent, WriteJob};
use anyhow::Result;
use forgetfuldb_core::types::MemoryType;
use forgetfuldb_store::pipeline::IngestRequest;
use forgetfuldb_tools::{ExploreTool, Tool, ToolCall};
use std::path::Path;

/// Hard cap on exploration commands per research run; after this the
/// model is told to write its report.
pub const RESEARCH_MAX_STEPS: usize = 12;

/// Cap per remembered fact so a rambling line can't bloat a memory.
const MAX_FACT_CHARS: usize = 300;

/// What happened during a research run.
pub struct ResearchReport {
    /// The model's final Markdown report (includes the remember block).
    pub summary: String,
    /// Exploration commands executed.
    pub steps: usize,
    /// Facts ingested as semantic memories.
    pub memories_stored: usize,
    /// Folder name used in the `project:<name>` tag.
    pub project: String,
}

fn research_prompt(root: &Path) -> String {
    format!(
        "You are a meticulous project researcher exploring the folder {root} on the user's \
         machine.\n\
         \n\
         HOW TO EXPLORE\n\
         Reply with ONLY a fenced block proposing one read-only command at a time:\n\
         ```tool\n{{\"tool\": \"explore\", \"args\": {{\"command\": \"ls -la\"}}}}\n```\n\
         You will get the command's output back. Allowed commands: ls, cat, head, tail, grep, \
         rg, find, wc, file, tree, du, stat, pwd (pipes between them are fine). Anything that \
         writes, deletes or executes is rejected. Paths are relative to {root}.\n\
         \n\
         STRATEGY\n\
         Start broad (ls / find by type), then read the README and manifest files \
         (Cargo.toml, package.json, requirements.txt, ...), then sample the most important \
         source files. Prefer head/grep over cat for big files. You have at most \
         {max_steps} commands — spend them wisely.\n\
         \n\
         THE REPORT\n\
         When you understand the project (or are told to finish), reply WITHOUT a tool block: \
         a concise Markdown report — what the project is, tech stack, how it's organized, key \
         files, anything notable (TODOs, tests, configs). End the report with a fenced block:\n\
         ```remember\n\
         one standalone fact per line, 3 to 8 lines\n\
         ```\n\
         Each line must make sense on its own months from now (name the project explicitly).",
        root = root.display(),
        max_steps = RESEARCH_MAX_STEPS,
    )
}

/// Extract an explore call from a model reply. The ```bash fallback is
/// only honored while exploring a reply WITHOUT a remember block — a
/// final report may quote shell snippets that must not be re-executed.
fn parse_explore_call(reply: &str) -> Option<ToolCall> {
    if let Some(call) = forgetfuldb_tools::parse_tool_call(reply) {
        if call.tool == "explore" || call.tool == "shell" {
            return Some(ToolCall { tool: "explore".to_string(), args: call.args });
        }
    }
    if reply.contains("```remember") {
        return None;
    }
    forgetfuldb_tools::extract_shell_command(reply)
        .map(|command| ToolCall { tool: "explore".to_string(), args: serde_json::json!({ "command": command }) })
}

/// Lines of the ```remember block, cleaned of bullets and blanks.
pub fn remember_lines(summary: &str) -> Vec<String> {
    let mut facts = Vec::new();
    let mut in_block = false;
    for line in summary.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            if in_block {
                break;
            }
            in_block = trimmed.trim_start_matches('`').trim().eq_ignore_ascii_case("remember");
            continue;
        }
        if in_block {
            let fact = trimmed.trim_start_matches(['-', '*']).trim();
            if !fact.is_empty() {
                facts.push(fact.chars().take(MAX_FACT_CHARS).collect());
            }
        }
    }
    facts
}

impl Agent {
    /// Research `path`: explore it with read-only commands, stream the
    /// model's commentary through `on_token`, announce each executed
    /// command through `on_step(step, command)`, and ingest the distilled
    /// facts as semantic memories tagged `project:<folder>`.
    ///
    /// Runs on a dedicated message thread — the chat history is neither
    /// used nor polluted.
    pub async fn research(
        &mut self,
        path: &Path,
        on_token: &mut dyn FnMut(&str),
        on_step: &mut dyn FnMut(usize, &str),
    ) -> Result<ResearchReport> {
        let root = path
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("cannot resolve {}: {e}", path.display()))?;
        anyhow::ensure!(root.is_dir(), "{} is not a directory", root.display());
        let project = root
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("project")
            .to_string();
        let explore = ExploreTool::new(root.clone(), self.cfg.tools.shell_timeout_secs);

        let mut messages = vec![
            ChatMessage::new("system", research_prompt(&root)),
            ChatMessage::new("user", format!("Begin researching {} now.", root.display())),
        ];
        let mut steps = 0usize;
        let mut summary = String::new();
        // Up to MAX_STEPS command rounds plus two wrap-up nudges; if the
        // model still insists on tools after that, its last reply stands.
        for _round in 0..(RESEARCH_MAX_STEPS + 2) {
            let (reply, _usage) = self.backend.chat_stream(&messages, on_token).await?;
            messages.push(ChatMessage::new("assistant", reply.clone()));
            match parse_explore_call(&reply) {
                Some(call) if steps < RESEARCH_MAX_STEPS => {
                    steps += 1;
                    let command = explore.preview(&call.args);
                    on_step(steps, &command);
                    let output = explore.execute(&call.args).unwrap_or_else(|e| format!("error: {e}"));
                    let nudge = if steps == RESEARCH_MAX_STEPS {
                        "\nThat was the last exploration command. Write the final report now, \
                         ending with the ```remember block."
                    } else {
                        ""
                    };
                    messages.push(ChatMessage::new(
                        "user",
                        format!("Output of `{command}`:\n```\n{output}\n```{nudge}"),
                    ));
                }
                Some(_) => {
                    messages.push(ChatMessage::new(
                        "user",
                        "No more commands available. Write the final report now, ending with \
                         the ```remember block.",
                    ));
                }
                None => {
                    summary = reply;
                    break;
                }
            }
        }
        if summary.is_empty() {
            summary = messages
                .iter()
                .rev()
                .find(|m| m.role == "assistant")
                .map(|m| m.content.clone())
                .unwrap_or_default();
        }

        // Distill: remember-block lines, or the report head as a fallback,
        // become semantic memories. Dedup in the ingest pipeline means
        // re-researching a project reinforces instead of duplicating.
        let mut facts = remember_lines(&summary);
        if facts.is_empty() && !summary.trim().is_empty() {
            facts.push(summary.trim().chars().take(600).collect());
        }
        let memories_stored = facts.len();
        for fact in facts {
            self.writer.submit(WriteJob::Ingest(IngestRequest {
                text: fact,
                source: Some("research".to_string()),
                tags: vec![format!("project:{project}")],
                memory_type: Some(MemoryType::Semantic),
                session_id: None,
                role: None,
            }));
        }

        Ok(ResearchReport { summary, steps, memories_stored, project })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remember_block_lines_become_facts() {
        let report = "# Report\nSome code:\n```bash\nls\n```\nDone.\n```remember\n- iforgot is a Rust memory DB\nuses SQLite with WAL\n\n* has 11 crates\n```\ntail";
        let facts = remember_lines(report);
        assert_eq!(
            facts,
            vec!["iforgot is a Rust memory DB", "uses SQLite with WAL", "has 11 crates"]
        );
    }

    #[test]
    fn no_remember_block_means_no_facts() {
        assert!(remember_lines("just a report\n```bash\nls\n```").is_empty());
    }

    #[test]
    fn explore_call_parsed_from_tool_block() {
        let reply = "```tool\n{\"tool\": \"explore\", \"args\": {\"command\": \"ls -la\"}}\n```";
        let call = parse_explore_call(reply).unwrap();
        assert_eq!(call.tool, "explore");
        assert_eq!(call.args["command"], "ls -la");
    }

    #[test]
    fn bash_fallback_works_but_not_inside_final_report() {
        let exploring = "Let me look:\n```bash\ncat README.md\n```";
        assert_eq!(parse_explore_call(exploring).unwrap().args["command"], "cat README.md");

        // A final report quoting a shell snippet must not be re-executed.
        let report = "Usage:\n```bash\ncargo run\n```\n```remember\nfact one\n```";
        assert!(parse_explore_call(report).is_none());
    }

    #[test]
    fn prompt_names_the_root_and_the_protocol() {
        let p = research_prompt(Path::new("/tmp/myproj"));
        assert!(p.contains("/tmp/myproj"));
        assert!(p.contains("```tool"));
        assert!(p.contains("```remember"));
    }
}
