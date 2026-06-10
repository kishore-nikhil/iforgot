//! forgetfuldb-tools
//!
//! A small, pluggable tool interface so the local assistant can *do*
//! things, not just talk. Adding a new capability is one type plus one
//! line of registration:
//!
//! ```ignore
//! struct Clock;
//! impl Tool for Clock {
//!     fn name(&self) -> &str { "clock" }
//!     fn description(&self) -> &str { "current time" }
//!     fn usage(&self) -> &str { "{}" }
//!     fn preview(&self, _args: &Value) -> String { "show the time".into() }
//!     fn execute(&self, _args: &Value) -> Result<String> { Ok(now_string()) }
//! }
//! registry.register(Box::new(Clock));
//! ```
//!
//! ## Safety model
//!
//! Tools can run arbitrary side effects (the built-in [`ShellTool`] runs
//! shell commands), so execution is **never automatic**:
//!
//! - The LLM can only *propose* a call by emitting a fenced ```tool``
//!   block. [`parse_tool_call`] extracts it; the frontend decides what to
//!   do with it.
//! - [`Tool::requires_confirmation`] lets a tool demand explicit user
//!   approval before running. The `iforgot` CLI shows the exact command
//!   and waits for Enter/y.
//! - The HTTP server lists tools but refuses to execute them unless the
//!   operator opts in (`tools.allow_server_execute`), because an HTTP
//!   endpoint can't ask a human first.

pub mod shell;

pub use shell::ShellTool;

use anyhow::Result;
use forgetfuldb_core::config::ToolsConfig;
use serde::Serialize;
use serde_json::Value;

/// A capability the assistant can invoke. Implementations must be cheap to
/// construct and safe to share across threads.
pub trait Tool: Send + Sync {
    /// Stable identifier the LLM uses to call the tool (e.g. `"shell"`).
    fn name(&self) -> &str;

    /// One-line description shown to the LLM so it knows when to use it.
    fn description(&self) -> &str;

    /// Argument shape, shown to the LLM (e.g. `{"command": "<shell>"}`).
    fn usage(&self) -> &str;

    /// Whether running this tool needs explicit user approval. Defaults to
    /// true — opt out only for read-only, side-effect-free tools.
    fn requires_confirmation(&self) -> bool {
        true
    }

    /// Human-readable rendering of what *this* call will do, shown in the
    /// confirmation prompt (e.g. the literal command to be run).
    fn preview(&self, args: &Value) -> String;

    /// Run the tool and return its textual output.
    fn execute(&self, args: &Value) -> Result<String>;
}

/// Lightweight description for listing tools (CLI `/tools`, server
/// `GET /tools`).
#[derive(Debug, Clone, Serialize)]
pub struct ToolInfo {
    pub name: String,
    pub description: String,
    pub usage: String,
    pub requires_confirmation: bool,
}

/// A tool call proposed by the LLM (or issued directly by the user).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub tool: String,
    pub args: Value,
}

/// Holds the registered tools and turns them into a prompt section.
#[derive(Default)]
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn empty() -> Self {
        ToolRegistry { tools: Vec::new() }
    }

    /// Build the default registry from config. Currently just the shell
    /// tool, gated by `tools.enabled` and `tools.shell_enabled`.
    pub fn from_config(cfg: &ToolsConfig) -> Self {
        let mut registry = ToolRegistry::empty();
        if cfg.enabled && cfg.shell_enabled {
            registry.register(Box::new(ShellTool::new(cfg.shell_timeout_secs)));
        }
        registry
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.iter().find(|t| t.name() == name).map(|t| t.as_ref())
    }

    pub fn list(&self) -> Vec<ToolInfo> {
        self.tools
            .iter()
            .map(|t| ToolInfo {
                name: t.name().to_string(),
                description: t.description().to_string(),
                usage: t.usage().to_string(),
                requires_confirmation: t.requires_confirmation(),
            })
            .collect()
    }

    /// Execute a call against the registered tool, or error if unknown.
    pub fn execute(&self, call: &ToolCall) -> Result<String> {
        let tool = self
            .get(&call.tool)
            .ok_or_else(|| anyhow::anyhow!("unknown tool '{}'", call.tool))?;
        tool.execute(&call.args)
    }

    /// The system-prompt section describing how to call tools. Empty when
    /// no tools are registered, so the prompt stays clean.
    pub fn prompt_section(&self) -> String {
        if self.tools.is_empty() {
            return String::new();
        }
        let mut out = String::from(
            "\n\nTOOLS\nYou can run local tools to help the user. To use one, reply with ONLY a \
             fenced code block tagged `tool` containing JSON, and nothing else:\n\
             ```tool\n{\"tool\": \"<name>\", \"args\": { ... }}\n```\n\
             You will then receive the tool's output and should answer the user using it. \
             Only request a tool when it genuinely helps; otherwise answer normally.\n\
             Available tools:\n",
        );
        for t in &self.tools {
            out.push_str(&format!("- {}: {} — args: {}\n", t.name(), t.description(), t.usage()));
        }
        out
    }
}

/// Extract a tool call from an LLM reply, if one is present.
///
/// Looks for a fenced code block (``` ... ```) whose JSON has a `"tool"`
/// key; falls back to treating the whole trimmed reply as that JSON. The
/// fenced-block requirement keeps commands merely *mentioned* in prose
/// from being parsed as calls.
pub fn parse_tool_call(reply: &str) -> Option<ToolCall> {
    for block in fenced_blocks(reply) {
        if let Some(call) = parse_json_object(&block) {
            return Some(call);
        }
    }
    parse_json_object(reply.trim())
}

/// Collect the contents of each ``` ... ``` fenced block.
fn fenced_blocks(text: &str) -> Vec<String> {
    let parts: Vec<&str> = text.split("```").collect();
    // Block contents are the odd-indexed segments between fences.
    parts.iter().skip(1).step_by(2).map(|s| s.to_string()).collect()
}

/// Parse the first `{...}` object in `s` and turn it into a ToolCall if it
/// has a `"tool"` field. Accepts either `{"tool":"x","args":{...}}` or a
/// flat `{"tool":"x", ...rest as args}` shape.
fn parse_json_object(s: &str) -> Option<ToolCall> {
    let start = s.find('{')?;
    let end = s.rfind('}')?;
    if end < start {
        return None;
    }
    let value: Value = serde_json::from_str(&s[start..=end]).ok()?;
    let tool = value.get("tool")?.as_str()?.to_string();
    let args = value.get("args").cloned().unwrap_or_else(|| value.clone());
    Some(ToolCall { tool, args })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fenced_tool_block() {
        let reply = "Sure, let me check.\n```tool\n{\"tool\": \"shell\", \"args\": {\"command\": \"ifconfig\"}}\n```";
        let call = parse_tool_call(reply).unwrap();
        assert_eq!(call.tool, "shell");
        assert_eq!(call.args["command"], "ifconfig");
    }

    #[test]
    fn ignores_command_merely_mentioned_in_prose() {
        let reply = "You could run `ifconfig` to see your IP, but I won't do it for you.";
        assert!(parse_tool_call(reply).is_none());
    }

    #[test]
    fn parses_bare_json_reply() {
        let reply = "{\"tool\":\"shell\",\"args\":{\"command\":\"ls\"}}";
        let call = parse_tool_call(reply).unwrap();
        assert_eq!(call.tool, "shell");
    }

    #[test]
    fn registry_prompt_section_and_lookup() {
        let cfg = ToolsConfig::default();
        let registry = ToolRegistry::from_config(&cfg);
        assert!(!registry.is_empty());
        assert!(registry.get("shell").is_some());
        assert!(registry.prompt_section().contains("shell"));
        assert_eq!(registry.list()[0].name, "shell");
    }

    #[test]
    fn disabled_config_yields_empty_registry() {
        let cfg = ToolsConfig { enabled: false, ..ToolsConfig::default() };
        let registry = ToolRegistry::from_config(&cfg);
        assert!(registry.is_empty());
        assert!(registry.prompt_section().is_empty());
    }
}
