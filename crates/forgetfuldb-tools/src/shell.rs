//! Shell command tool.
//!
//! Runs a command through the system shell and returns its combined
//! stdout/stderr. Because this is the most dangerous built-in, it always
//! [`requires_confirmation`](Tool::requires_confirmation) — the LLM can
//! only propose a command; a human approves it before it runs.

use crate::Tool;
use anyhow::{Context, Result};
use serde_json::Value;
use std::process::{Command, Stdio};
use std::time::Duration;

/// Cap on returned output so a chatty command can't blow up the prompt or
/// terminal.
const MAX_OUTPUT_CHARS: usize = 8_000;

pub struct ShellTool {
    timeout: Duration,
}

impl ShellTool {
    pub fn new(timeout_secs: u64) -> Self {
        ShellTool { timeout: Duration::from_secs(timeout_secs.clamp(1, 600)) }
    }

    fn command_from_args(args: &Value) -> Result<String> {
        args.get("command")
            .and_then(Value::as_str)
            .map(str::to_string)
            .filter(|c| !c.trim().is_empty())
            .context("shell tool needs a non-empty \"command\" argument")
    }
}

impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "run a shell command on the user's machine and return its output"
    }

    fn usage(&self) -> &str {
        "{\"command\": \"<shell command>\"}"
    }

    fn requires_confirmation(&self) -> bool {
        true
    }

    fn preview(&self, args: &Value) -> String {
        Self::command_from_args(args).unwrap_or_else(|_| "<invalid shell command>".to_string())
    }

    fn execute(&self, args: &Value) -> Result<String> {
        let command = Self::command_from_args(args)?;
        run_with_timeout(&command, self.timeout)
    }
}

/// Run `command` through the platform shell, killing it if it exceeds
/// `timeout`. Output is captured after the process ends and truncated to
/// [`MAX_OUTPUT_CHARS`].
fn run_with_timeout(command: &str, timeout: Duration) -> Result<String> {
    let mut child = shell_command(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to launch shell command: {command}"))?;

    // Enforce the timeout by polling try_wait; kill a runaway process so
    // we never leak it. (std has no blocking wait-with-timeout, and a
    // 20ms poll is imperceptible for interactive commands.)
    let start = std::time::Instant::now();
    loop {
        match child.try_wait()? {
            Some(_status) => break,
            None => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok(format!(
                        "(command timed out after {}s and was terminated)",
                        timeout.as_secs()
                    ));
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }

    let output = child.wait_with_output().context("collecting command output")?;
    let mut text = String::new();
    text.push_str(&String::from_utf8_lossy(&output.stdout));
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.trim().is_empty() {
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        text.push_str(&stderr);
    }
    let mut text = text.trim().to_string();
    if text.is_empty() {
        text = format!("(command exited with {}, no output)", output.status);
    }
    if text.chars().count() > MAX_OUTPUT_CHARS {
        let truncated: String = text.chars().take(MAX_OUTPUT_CHARS).collect();
        text = format!("{truncated}\n…(output truncated)");
    }
    Ok(text)
}

#[cfg(unix)]
fn shell_command(command: &str) -> Command {
    let mut c = Command::new("sh");
    c.arg("-c").arg(command);
    c
}

#[cfg(windows)]
fn shell_command(command: &str) -> Command {
    let mut c = Command::new("cmd");
    c.arg("/C").arg(command);
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn runs_a_simple_command() {
        let tool = ShellTool::new(10);
        let out = tool.execute(&json!({"command": "echo hello from forgetfuldb"})).unwrap();
        assert!(out.contains("hello from forgetfuldb"));
    }

    #[test]
    fn captures_stderr() {
        let tool = ShellTool::new(10);
        let out = tool.execute(&json!({"command": "echo oops 1>&2"})).unwrap();
        assert!(out.contains("oops"));
    }

    #[test]
    fn missing_command_errors() {
        let tool = ShellTool::new(10);
        assert!(tool.execute(&json!({})).is_err());
        assert!(tool.execute(&json!({"command": "   "})).is_err());
    }

    #[test]
    fn preview_shows_the_command() {
        let tool = ShellTool::new(10);
        assert_eq!(tool.preview(&json!({"command": "ls -la"})), "ls -la");
        assert!(tool.requires_confirmation());
    }

    #[test]
    fn times_out_long_commands() {
        let tool = ShellTool::new(1);
        let out = tool.execute(&json!({"command": "sleep 5"})).unwrap();
        assert!(out.contains("timed out"));
    }
}
