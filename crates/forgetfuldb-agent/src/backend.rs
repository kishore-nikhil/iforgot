//! Local LLM chat backends.
//!
//! Two implementations behind one enum (an enum instead of a trait object
//! keeps async methods simple and dependency-free):
//!
//! - [`ChatBackend::Ollama`]: Ollama's native `/api/chat` (NDJSON
//!   streaming). Reports exact token counts and durations per turn.
//! - [`ChatBackend::OpenAiCompat`]: `/v1/chat/completions` (SSE
//!   streaming). Works with llama-server (llama.cpp), LM Studio, and
//!   anything else OpenAI-shaped. Usage reporting varies by server.
//!
//! Both speak plain HTTP to localhost. reqwest is built without TLS in
//! this workspace, so remote https endpoints are unreachable by
//! construction — local-only is enforced at the dependency level, and
//! [`ensure_local_url`] additionally rejects non-loopback hosts when
//! `local_only = true`.

use anyhow::{Context, Result};
use forgetfuldb_core::config::{ChatConfig, Config};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        ChatMessage { role: role.to_string(), content: content.into() }
    }
}

/// Token/timing usage reported by the backend. Fields stay `None` when a
/// backend doesn't report them — callers may estimate, but stored metrics
/// distinguish measured from missing.
#[derive(Debug, Clone, Default)]
pub struct ChatUsage {
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub total_duration_ms: Option<i64>,
    pub llm_duration_ms: Option<i64>,
}

pub enum ChatBackend {
    Ollama { base_url: String, model: String, keep_alive: String, client: reqwest::Client },
    OpenAiCompat { base_url: String, model: String, client: reqwest::Client },
}

/// Reject non-loopback chat URLs when `local_only` is set.
pub fn ensure_local_url(cfg: &Config) -> Result<()> {
    if !cfg.local_only {
        return Ok(());
    }
    let url = &cfg.chat.base_url;
    let host = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url)
        .split(['/', ':'])
        .next()
        .unwrap_or("");
    let local = matches!(host, "localhost" | "127.0.0.1" | "[::1]" | "::1");
    anyhow::ensure!(
        local,
        "local_only = true but chat base_url is '{url}'. \
         Point it at localhost, or set local_only = false explicitly."
    );
    Ok(())
}

impl ChatBackend {
    pub fn from_config(cfg: &Config) -> Result<ChatBackend> {
        ensure_local_url(cfg)?;
        let ChatConfig { backend, base_url, model, keep_alive, .. } = &cfg.chat;
        let base_url = base_url.trim_end_matches('/').to_string();
        let client = reqwest::Client::new();
        match backend.as_str() {
            "ollama" => Ok(ChatBackend::Ollama {
                base_url,
                model: model.clone(),
                keep_alive: keep_alive.clone(),
                client,
            }),
            "openai_compat" => Ok(ChatBackend::OpenAiCompat { base_url, model: model.clone(), client }),
            other => anyhow::bail!("unknown chat backend '{other}' (available: ollama, openai_compat)"),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ChatBackend::Ollama { .. } => "ollama",
            ChatBackend::OpenAiCompat { .. } => "openai_compat",
        }
    }

    pub fn model(&self) -> &str {
        match self {
            ChatBackend::Ollama { model, .. } | ChatBackend::OpenAiCompat { model, .. } => model,
        }
    }

    /// Switch the model for subsequent turns. Callers persist the change
    /// to the config file separately.
    pub fn set_model(&mut self, name: &str) {
        match self {
            ChatBackend::Ollama { model, .. } | ChatBackend::OpenAiCompat { model, .. } => {
                *model = name.to_string();
            }
        }
    }

    /// Models installed on the backend server (Ollama `/api/tags` or
    /// OpenAI-compatible `/v1/models`).
    pub async fn list_models(&self) -> Result<Vec<String>> {
        match self {
            ChatBackend::Ollama { base_url, client, .. } => {
                let v: Value = client
                    .get(format!("{base_url}/api/tags"))
                    .send()
                    .await?
                    .error_for_status()?
                    .json()
                    .await?;
                Ok(parse_ollama_tags(&v))
            }
            ChatBackend::OpenAiCompat { base_url, client, .. } => {
                let v: Value = client
                    .get(format!("{base_url}/v1/models"))
                    .send()
                    .await?
                    .error_for_status()?
                    .json()
                    .await?;
                Ok(parse_openai_models(&v))
            }
        }
    }

    pub fn base_url(&self) -> &str {
        match self {
            ChatBackend::Ollama { base_url, .. } | ChatBackend::OpenAiCompat { base_url, .. } => base_url,
        }
    }

    /// Cheap reachability probe so the chat UI can warn before the first
    /// turn instead of failing mid-conversation.
    pub async fn health(&self) -> bool {
        let url = match self {
            ChatBackend::Ollama { base_url, .. } => format!("{base_url}/api/tags"),
            ChatBackend::OpenAiCompat { base_url, .. } => format!("{base_url}/v1/models"),
        };
        match self {
            ChatBackend::Ollama { client, .. } | ChatBackend::OpenAiCompat { client, .. } => {
                client.get(url).send().await.map(|r| r.status().is_success()).unwrap_or(false)
            }
        }
    }

    /// Stream a chat completion, invoking `on_token` per token as it
    /// arrives. Returns the full reply plus whatever usage the backend
    /// reported.
    pub async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        on_token: &mut dyn FnMut(&str),
    ) -> Result<(String, ChatUsage)> {
        match self {
            ChatBackend::Ollama { base_url, model, keep_alive, client } => {
                // keep_alive holds the model in memory between turns so an
                // idle pause doesn't cost a full model reload.
                let body = json!({
                    "model": model,
                    "messages": messages,
                    "stream": true,
                    "keep_alive": keep_alive,
                });
                let resp = client
                    .post(format!("{base_url}/api/chat"))
                    .json(&body)
                    .send()
                    .await
                    .with_context(|| format!("is your LLM server running at {base_url}?"))?
                    .error_for_status()?;
                stream_lines(resp, on_token, parse_ollama_line).await
            }
            ChatBackend::OpenAiCompat { base_url, model, client } => {
                let body = json!({ "model": model, "messages": messages, "stream": true });
                let resp = client
                    .post(format!("{base_url}/v1/chat/completions"))
                    .json(&body)
                    .send()
                    .await
                    .with_context(|| format!("is your LLM server running at {base_url}?"))?
                    .error_for_status()?;
                stream_lines(resp, on_token, parse_openai_line).await
            }
        }
    }
}

/// Model names from Ollama's `/api/tags` response.
pub fn parse_ollama_tags(v: &Value) -> Vec<String> {
    v.get("models")
        .and_then(Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(|m| m.get("name").and_then(Value::as_str).map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Model ids from an OpenAI-compatible `/v1/models` response.
pub fn parse_openai_models(v: &Value) -> Vec<String> {
    v.get("data")
        .and_then(Value::as_array)
        .map(|models| {
            models
                .iter()
                .filter_map(|m| m.get("id").and_then(Value::as_str).map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Do two model names refer to the same model? Ollama treats `name` and
/// `name:latest` as equivalent.
pub fn model_matches(a: &str, b: &str) -> bool {
    a == b
        || a.strip_suffix(":latest").map(|s| s == b).unwrap_or(false)
        || b.strip_suffix(":latest").map(|s| s == a).unwrap_or(false)
}

/// What one streamed line contributed.
#[derive(Debug, Default, PartialEq)]
pub struct LineEvent {
    pub token: Option<String>,
    pub usage: Option<ChatUsageUpdate>,
    pub done: bool,
}

#[derive(Debug, Default, PartialEq)]
pub struct ChatUsageUpdate {
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub total_duration_ms: Option<i64>,
    pub llm_duration_ms: Option<i64>,
}

/// Parse one NDJSON line from Ollama's `/api/chat` stream.
pub fn parse_ollama_line(line: &str) -> Option<LineEvent> {
    let v: Value = serde_json::from_str(line.trim()).ok()?;
    let mut ev = LineEvent::default();
    if let Some(tok) = v.pointer("/message/content").and_then(Value::as_str) {
        if !tok.is_empty() {
            ev.token = Some(tok.to_string());
        }
    }
    if v.get("done").and_then(Value::as_bool) == Some(true) {
        ev.done = true;
        ev.usage = Some(ChatUsageUpdate {
            prompt_tokens: v.get("prompt_eval_count").and_then(Value::as_i64),
            completion_tokens: v.get("eval_count").and_then(Value::as_i64),
            // Ollama reports nanoseconds.
            total_duration_ms: v.get("total_duration").and_then(Value::as_i64).map(|ns| ns / 1_000_000),
            llm_duration_ms: v.get("eval_duration").and_then(Value::as_i64).map(|ns| ns / 1_000_000),
        });
    }
    Some(ev)
}

/// Parse one SSE line from an OpenAI-compatible stream.
pub fn parse_openai_line(line: &str) -> Option<LineEvent> {
    let line = line.trim();
    let data = line.strip_prefix("data:")?.trim();
    if data == "[DONE]" {
        return Some(LineEvent { done: true, ..Default::default() });
    }
    let v: Value = serde_json::from_str(data).ok()?;
    let mut ev = LineEvent::default();
    if let Some(tok) = v.pointer("/choices/0/delta/content").and_then(Value::as_str) {
        if !tok.is_empty() {
            ev.token = Some(tok.to_string());
        }
    }
    // Some servers (llama-server, newer Ollama) attach usage to a chunk.
    if let Some(usage) = v.get("usage").filter(|u| u.is_object()) {
        ev.usage = Some(ChatUsageUpdate {
            prompt_tokens: usage.get("prompt_tokens").and_then(Value::as_i64),
            completion_tokens: usage.get("completion_tokens").and_then(Value::as_i64),
            total_duration_ms: None,
            llm_duration_ms: None,
        });
    }
    Some(ev)
}

/// Drive a byte stream line by line through a parser, accumulating the
/// reply and usage.
async fn stream_lines(
    resp: reqwest::Response,
    on_token: &mut dyn FnMut(&str),
    parse: fn(&str) -> Option<LineEvent>,
) -> Result<(String, ChatUsage)> {
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut reply = String::new();
    let mut usage = ChatUsage::default();

    let handle_line = |line: &str, reply: &mut String, usage: &mut ChatUsage, on_token: &mut dyn FnMut(&str)| {
        if let Some(ev) = parse(line) {
            if let Some(tok) = ev.token {
                on_token(&tok);
                reply.push_str(&tok);
            }
            if let Some(u) = ev.usage {
                usage.prompt_tokens = u.prompt_tokens.or(usage.prompt_tokens);
                usage.completion_tokens = u.completion_tokens.or(usage.completion_tokens);
                usage.total_duration_ms = u.total_duration_ms.or(usage.total_duration_ms);
                usage.llm_duration_ms = u.llm_duration_ms.or(usage.llm_duration_ms);
            }
        }
    };

    while let Some(chunk) = stream.next().await {
        buf.extend_from_slice(&chunk?);
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            if let Ok(text) = std::str::from_utf8(&line) {
                if !text.trim().is_empty() {
                    handle_line(text, &mut reply, &mut usage, on_token);
                }
            }
        }
    }
    if let Ok(text) = std::str::from_utf8(&buf) {
        if !text.trim().is_empty() {
            handle_line(text, &mut reply, &mut usage, on_token);
        }
    }
    Ok((reply, usage))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_token_line_parses() {
        let ev = parse_ollama_line(r#"{"message":{"role":"assistant","content":"Hel"},"done":false}"#).unwrap();
        assert_eq!(ev.token.as_deref(), Some("Hel"));
        assert!(!ev.done);
    }

    #[test]
    fn ollama_final_line_carries_usage() {
        let ev = parse_ollama_line(
            r#"{"done":true,"prompt_eval_count":42,"eval_count":7,"total_duration":1500000000,"eval_duration":1200000000}"#,
        )
        .unwrap();
        assert!(ev.done);
        let u = ev.usage.unwrap();
        assert_eq!(u.prompt_tokens, Some(42));
        assert_eq!(u.completion_tokens, Some(7));
        assert_eq!(u.total_duration_ms, Some(1500));
    }

    #[test]
    fn openai_sse_lines_parse() {
        let ev = parse_openai_line(r#"data: {"choices":[{"delta":{"content":"Hi"}}]}"#).unwrap();
        assert_eq!(ev.token.as_deref(), Some("Hi"));
        assert!(parse_openai_line("data: [DONE]").unwrap().done);
        assert!(parse_openai_line(": keepalive comment").is_none());
        let with_usage =
            parse_openai_line(r#"data: {"choices":[],"usage":{"prompt_tokens":99,"completion_tokens":5}}"#).unwrap();
        assert_eq!(with_usage.usage.unwrap().prompt_tokens, Some(99));
    }

    #[test]
    fn model_list_parsing() {
        let tags = serde_json::json!({"models": [{"name": "gemma4:12b"}, {"name": "llama3.2:3b"}]});
        assert_eq!(parse_ollama_tags(&tags), vec!["gemma4:12b", "llama3.2:3b"]);
        let models = serde_json::json!({"data": [{"id": "qwen2.5-7b"}]});
        assert_eq!(parse_openai_models(&models), vec!["qwen2.5-7b"]);
        assert!(parse_ollama_tags(&serde_json::json!({})).is_empty());
    }

    #[test]
    fn model_name_matching_handles_latest_suffix() {
        assert!(model_matches("gemma4:12b", "gemma4:12b"));
        assert!(model_matches("llama3.2:latest", "llama3.2"));
        assert!(model_matches("llama3.2", "llama3.2:latest"));
        assert!(!model_matches("gemma4:12b", "gemma3:12b"));
    }

    #[test]
    fn local_only_rejects_remote_urls() {
        let mut cfg = Config::default();
        cfg.chat.base_url = "http://api.example.com:8080".to_string();
        assert!(ensure_local_url(&cfg).is_err());
        cfg.chat.base_url = "http://127.0.0.1:11434".to_string();
        assert!(ensure_local_url(&cfg).is_ok());
        cfg.chat.base_url = "http://localhost:8080/".to_string();
        assert!(ensure_local_url(&cfg).is_ok());
        cfg.local_only = false;
        cfg.chat.base_url = "http://api.example.com".to_string();
        assert!(ensure_local_url(&cfg).is_ok());
    }
}
