//! OpenAI-compatible memory proxy.
//!
//! `POST /v1/chat/completions` accepts the standard OpenAI request shape,
//! transparently wraps it with memory, and forwards it to the local LLM
//! (Ollama or llama-server — both speak the same endpoint):
//!
//! 1. the last user message is ingested into ForgetfulDB
//! 2. a context pack is retrieved and injected as a leading system message
//! 3. the (possibly model-defaulted) request is forwarded upstream
//! 4. the reply is ingested and a metrics row is recorded
//!
//! Point any OpenAI-compatible chat UI (Open WebUI, IDE plugins, ...) at
//! this server as its base URL and it gains long-term memory with zero
//! integration work. The LLM and the UI both stay unaware memory exists.
//!
//! Streaming (`"stream": true`) is passed through verbatim; in that mode
//! the assistant reply is not captured for ingestion (the bytes go
//! straight to the client) and token usage is not recorded — context
//! metrics still are. Non-streaming requests get full metrics.

use crate::{ApiError, SharedState};
use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use forgetfuldb_agent::backend::ChatUsage;
use forgetfuldb_agent::{finish_turn, memory_context_block, prepare_turn, PreparedTurn};
use serde_json::{json, Value};

/// Last `"role": "user"` message with string content, if any.
fn last_user_content(body: &Value) -> Option<String> {
    body.get("messages")?
        .as_array()?
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))?
        .get("content")?
        .as_str()
        .map(str::to_string)
}

/// Up to `limit` user messages preceding the last one (oldest first).
/// Folded into the retrieval query so vague follow-ups still retrieve
/// memories about the conversation's topic; never added to the prompt.
fn prior_user_contents(body: &Value, limit: usize) -> Vec<String> {
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut users: Vec<String> = messages
        .iter()
        .filter(|m| m.get("role").and_then(Value::as_str) == Some("user"))
        .filter_map(|m| m.get("content").and_then(Value::as_str).map(str::to_string))
        .collect();
    users.pop(); // the last user message is the query itself
    let skip = users.len().saturating_sub(limit);
    users.split_off(skip)
}

pub(crate) async fn chat_completions(
    State(state): State<SharedState>,
    Json(mut body): Json<Value>,
) -> Result<Response, ApiError> {
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let user_text = last_user_content(&body);

    // Phase 1 (sync, under lock): retrieve memories for the user message.
    // Read-only — ingestion happens in `record` after the LLM responds.
    // The lock is released before the LLM call so generation never blocks
    // other API requests.
    let (prepared, base_url, default_model, client) = {
        let app = state.lock().expect("state mutex poisoned");
        let prepared = match &user_text {
            Some(text) => {
                let sp = app.cfg.chat.system_prompt.clone();
                let query_context = prior_user_contents(&body, app.cfg.chat.query_context_turns);
                // No session exclusion here: the proxy's session id is the
                // constant "proxy", so excluding it would hide every
                // proxy-learned memory ever, not just this conversation.
                // Conversational damping covers the duplicate-injection
                // case instead.
                Some(prepare_turn(&app.store, app.provider.as_ref(), &app.cfg, &sp, &[], &query_context, None, text)?)
            }
            None => None,
        };
        (
            prepared,
            app.cfg.chat.base_url.trim_end_matches('/').to_string(),
            app.cfg.chat.model.clone(),
            app.http_client.clone(),
        )
    };

    // Inject the memory block as a system message just BEFORE the last
    // user message (not at position 0): everything earlier in the
    // conversation stays byte-identical across requests, so the LLM
    // server's prefix KV-cache stays valid. Also default the model if the
    // client didn't pick one.
    if let Some(prepared) = &prepared {
        let system = json!({ "role": "system", "content": memory_context_block(&prepared.pack) });
        if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
            let idx = messages
                .iter()
                .rposition(|m| m.get("role").and_then(Value::as_str) == Some("user"))
                .unwrap_or(messages.len());
            messages.insert(idx, system);
        }
    }
    if body.get("model").and_then(Value::as_str).is_none_or(str::is_empty) {
        if default_model.is_empty() {
            return Err(anyhow::anyhow!(
                "no model selected: pass \"model\" in the request, or set [chat] model in \
                 forgetfuldb.toml (run `iforgot` once to pick interactively)"
            )
            .into());
        }
        body["model"] = json!(default_model);
    }

    let upstream = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("forwarding to LLM at {base_url} failed: {e}"))?;

    let status = StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);

    if stream {
        // Record what we know now (context metrics; no tokens, no reply),
        // then hand the byte stream straight through.
        if let Some(prepared) = &prepared {
            record(&state, prepared, "", &ChatUsage::default(), &default_model)?;
        }
        let content_type = upstream
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("text/event-stream")
            .to_string();
        let response = Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, content_type)
            .body(Body::from_stream(upstream.bytes_stream()))
            .map_err(|e| anyhow::anyhow!(e))?;
        return Ok(response);
    }

    let reply_json: Value = upstream.json().await.map_err(|e| anyhow::anyhow!(e))?;
    if let Some(prepared) = &prepared {
        let reply = reply_json
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .unwrap_or("");
        let usage = ChatUsage {
            prompt_tokens: reply_json.pointer("/usage/prompt_tokens").and_then(Value::as_i64),
            completion_tokens: reply_json.pointer("/usage/completion_tokens").and_then(Value::as_i64),
            total_duration_ms: None,
            llm_duration_ms: None,
        };
        record(&state, prepared, reply, &usage, &default_model)?;
    }
    Ok((status, Json(reply_json)).into_response())
}

/// Phase 2 (sync, re-lock): ingest both messages and persist metrics.
fn record(
    state: &SharedState,
    prepared: &PreparedTurn,
    reply: &str,
    usage: &ChatUsage,
    model: &str,
) -> Result<(), ApiError> {
    let mut app = state.lock().expect("state mutex poisoned");
    let crate::AppState { store, bloom, provider, cfg, .. } = &mut *app;
    finish_turn(store, bloom, provider.as_ref(), cfg, "proxy", prepared, reply, usage, "openai_proxy", model)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_last_user_message() {
        let body = json!({
            "messages": [
                {"role": "system", "content": "be nice"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "second"}
            ]
        });
        assert_eq!(last_user_content(&body).unwrap(), "second");
        assert!(last_user_content(&json!({"messages": []})).is_none());
    }

    #[test]
    fn prior_user_contents_skips_the_current_query() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "second"},
                {"role": "user", "content": "current"}
            ]
        });
        assert_eq!(prior_user_contents(&body, 2), vec!["first", "second"]);
        assert_eq!(prior_user_contents(&body, 1), vec!["second"]);
        assert!(prior_user_contents(&body, 0).is_empty());
        assert!(prior_user_contents(&json!({"messages": []}), 2).is_empty());
    }
}
