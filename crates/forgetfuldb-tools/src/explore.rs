//! Read-only exploration tool for the research agent.
//!
//! Unlike [`ShellTool`](crate::ShellTool), this tool runs **without user
//! confirmation** — so it must be safe by construction, not by approval.
//! Safety comes from a strict validator, not from trusting the model:
//!
//! - Only programs on a hard-coded read-only allowlist may run (`ls`,
//!   `cat`, `grep`, `find`, ...). Every segment of a pipe / `;` chain is
//!   checked, so `cat x | rm -rf /` is rejected at the `rm`.
//! - Output redirection (`>`), command substitution (`` ` ``, `$(`,
//!   `<(`), and backgrounding (`&`) are rejected outright — they are the
//!   escape hatches that would let a "read-only" command write or execute.
//! - `find`'s write/exec flags (`-delete`, `-exec`, `-execdir`, `-ok`,
//!   `-okdir`, `-fprint*`) are rejected even though `find` itself is
//!   allowed.
//! - Commands run with the working directory pinned to the research root.
//!
//! The validator is deliberately conservative: a legitimate command that
//! merely *looks* dangerous (e.g. `grep '$(x)' f`) is rejected with a
//! reason the model can read and rephrase around.

use crate::Tool;
use anyhow::{Context, Result};
use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration;

/// Programs the explorer may run. Each must be read-only by nature.
const ALLOWED: &[&str] = &[
    "ls", "cat", "head", "tail", "grep", "rg", "find", "wc", "file", "tree", "du", "stat", "pwd",
];

/// Argument tokens that turn an allowed program into a writer/executor.
const FORBIDDEN_ARGS: &[&str] = &["-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprintf", "-fls"];

/// Substrings that are never allowed anywhere in the command line.
const FORBIDDEN_SUBSTRINGS: &[(&str, &str)] = &[
    (">", "output redirection"),
    ("`", "command substitution"),
    ("$(", "command substitution"),
    ("<(", "process substitution"),
    ("&", "backgrounding / chaining"),
];

/// Check that `command` only does read-only work. Returns the reason on
/// rejection so the model can correct itself.
pub fn validate_readonly(command: &str) -> std::result::Result<(), String> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return Err("empty command".to_string());
    }
    for (needle, why) in FORBIDDEN_SUBSTRINGS {
        if trimmed.contains(needle) {
            return Err(format!("'{needle}' is not allowed ({why})"));
        }
    }
    // Validate the head of every pipeline/sequence segment.
    for segment in trimmed.split(['|', ';', '\n']) {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        let mut tokens = segment.split_whitespace();
        let head = tokens.next().unwrap_or_default();
        if !ALLOWED.contains(&head) {
            return Err(format!(
                "'{head}' is not on the read-only allowlist ({})",
                ALLOWED.join(", ")
            ));
        }
        for token in tokens {
            if FORBIDDEN_ARGS.contains(&token) {
                return Err(format!("'{token}' is not allowed (it writes or executes)"));
            }
        }
    }
    Ok(())
}

/// A confirmation-free shell tool restricted to read-only commands, with
/// its working directory pinned to the folder being researched.
pub struct ExploreTool {
    root: PathBuf,
    timeout: Duration,
}

impl ExploreTool {
    pub fn new(root: PathBuf, timeout_secs: u64) -> Self {
        ExploreTool { root, timeout: Duration::from_secs(timeout_secs.clamp(1, 600)) }
    }

    pub fn root(&self) -> &PathBuf {
        &self.root
    }

    fn command_from_args(args: &Value) -> Result<String> {
        args.get("command")
            .and_then(Value::as_str)
            .map(str::to_string)
            .filter(|c| !c.trim().is_empty())
            .context("explore tool needs a non-empty \"command\" argument")
    }
}

impl Tool for ExploreTool {
    fn name(&self) -> &str {
        "explore"
    }

    fn description(&self) -> &str {
        "run a READ-ONLY shell command (ls, cat, head, tail, grep, rg, find, wc, file, tree, du, stat) inside the folder being researched"
    }

    fn usage(&self) -> &str {
        "{\"command\": \"<read-only command>\"}"
    }

    /// Safe by construction: validated allowlist, no confirmation needed.
    fn requires_confirmation(&self) -> bool {
        false
    }

    fn preview(&self, args: &Value) -> String {
        Self::command_from_args(args).unwrap_or_else(|_| "<invalid explore command>".to_string())
    }

    fn execute(&self, args: &Value) -> Result<String> {
        let command = Self::command_from_args(args)?;
        if let Err(reason) = validate_readonly(&command) {
            // Returned as tool output (not an error) so the agent loop
            // feeds it back to the model, which can then rephrase.
            return Ok(format!("rejected: {reason}. Use only read-only commands."));
        }
        crate::shell::run_with_timeout(&command, self.timeout, Some(&self.root))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn allows_read_only_commands_and_pipes() {
        for cmd in [
            "ls -la",
            "cat README.md",
            "grep -rn 'fn main' src",
            "find . -name '*.rs' -type f",
            "cat Cargo.toml | grep version",
            "head -50 src/main.rs; wc -l src/main.rs",
        ] {
            assert!(validate_readonly(cmd).is_ok(), "should allow: {cmd}");
        }
    }

    #[test]
    fn rejects_writes_deletes_and_escapes() {
        for cmd in [
            "rm -rf /",
            "ls; rm file",
            "cat x | xargs rm",
            "find . -name '*.tmp' -delete",
            "find . -exec rm {} \\;",
            "echo hi",
            "cat x > y",
            "ls && rm x",
            "cat `which ls`",
            "ls $(rm x)",
            "bash script.sh",
            "FOO=1 ls",
        ] {
            assert!(validate_readonly(cmd).is_err(), "should reject: {cmd}");
        }
    }

    #[test]
    fn executes_in_the_pinned_root() {
        let dir = std::env::temp_dir().join(format!("explore-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("marker.txt"), "hello explorer").unwrap();
        let tool = ExploreTool::new(dir.clone(), 10);

        let out = tool.execute(&json!({"command": "cat marker.txt"})).unwrap();
        assert!(out.contains("hello explorer"));
        assert!(!tool.requires_confirmation());

        let rejected = tool.execute(&json!({"command": "rm marker.txt"})).unwrap();
        assert!(rejected.starts_with("rejected:"));
        assert!(dir.join("marker.txt").exists(), "rm must never have run");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
