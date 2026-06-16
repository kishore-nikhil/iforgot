//! forgetfuldb-agent
//!
//! The memory-wrapped chat loop. The LLM stays stateless; this crate makes
//! every turn update memory automatically:
//!
//! 1. the user's message is **ingested** (write) through the normal
//!    pipeline — dedup, hashing, importance scoring
//! 2. a context pack is **retrieved** (read) for the message — queried
//!    with recent conversation context, gated by relevance, with the live
//!    session excluded — and attached to the user message; retrieval
//!    itself bumps access counts, so reading is rehearsal
//! 3. the reply streams from the local LLM ([`backend::ChatBackend`]:
//!    Ollama native or any OpenAI-compatible server)
//! 4. the assistant's reply is ingested as a fast-decaying `raw_event`
//! 5. a [`forgetfuldb_store::ChatTurn`] metrics row is recorded (token
//!    counts, context share, latencies) for later context optimization
//!
//! UIs are thin frontends over this crate: the `iforgot` terminal chat
//! links it directly, and `forgetfuldb-server` reuses the same pieces for
//! its OpenAI-compatible memory proxy.

pub mod backend;
pub mod research;
pub mod writer;

pub use backend::{ChatBackend, ChatMessage, ChatUsage};
pub use research::{ResearchReport, RESEARCH_MAX_STEPS};
pub use writer::{MemoryWriter, WriteJob};

use anyhow::Result;
use forgetfuldb_core::config::Config;
use forgetfuldb_core::ids::new_id;
use forgetfuldb_core::now_unix;
use forgetfuldb_core::types::MemoryType;
use forgetfuldb_embed::EmbeddingProvider;
use forgetfuldb_prob::BloomFilter;
use forgetfuldb_retrieve::{ContextPack, RetrieveOptions};
use forgetfuldb_store::pipeline::{ingest, IngestRequest};
use forgetfuldb_store::{ChatTurn, Store};
use forgetfuldb_tools::{ToolCall, ToolRegistry};
use std::time::Instant;

/// Rough token estimate (~4 chars/token) used only when the backend
/// doesn't report real usage.
pub fn estimate_tokens(text: &str) -> i64 {
    (text.chars().count() as i64 + 3) / 4
}

/// Render the retrieved memories as a context block. The framing matters:
/// memories are *background* — small models otherwise latch onto an old
/// topic from storage instead of what the user is talking about right now.
pub fn memory_context_block(pack: &ContextPack) -> String {
    if pack.memories.is_empty() {
        return "(no stored memories matched this message yet)".to_string();
    }
    let mut out = String::from(
        "Background memories from past sessions, most relevant first. They may be \
         unrelated to the current conversation; if they conflict with it, the live \
         conversation takes precedence:\n",
    );
    for m in &pack.memories {
        out.push_str(&format!("- [{}] {}\n", m.item.memory_type, m.item.content));
    }
    out
}

/// Attach the memory block to the *current user message* instead of the
/// system prompt. The system prompt and history then stay byte-identical
/// across turns, so Ollama's prefix KV-cache survives and each turn only
/// evaluates the new tokens — with a 12B model that's the difference
/// between milliseconds and seconds before the first token.
pub fn wrap_user_message(user_text: &str, pack: &ContextPack) -> String {
    if pack.memories.is_empty() {
        return user_text.to_string();
    }
    format!("{}\n{}", memory_context_block(pack), user_text)
}

/// Everything computed before the LLM call: the sync, DB-touching half of
/// a turn. Split from the async LLM call so callers (e.g. the server) can
/// release the store lock while the model generates.
pub struct PreparedTurn {
    pub user_text: String,
    pub messages: Vec<ChatMessage>,
    pub pack: ContextPack,
    pub context_chars: i64,
    pub retrieve_duration_ms: i64,
}

/// Retrieve context and assemble the prompt. Read-only: ingestion happens
/// asynchronously via [`MemoryWriter`] (or synchronously in the proxy),
/// which also guarantees the current message can't rank as a "memory" of
/// itself in its own context pack.
///
/// `query_context` is the last few *raw* user messages (oldest first).
/// They are folded into the retrieval query only — never into the prompt —
/// so a vague follow-up ("something catchier") still retrieves memories
/// about the conversation's actual topic. `session_id`, when set, keeps
/// this session's own turns out of the pack: they're already in `history`.
#[allow(clippy::too_many_arguments)]
pub fn prepare_turn(
    store: &Store,
    provider: &dyn EmbeddingProvider,
    cfg: &Config,
    system_prompt: &str,
    history: &[ChatMessage],
    query_context: &[String],
    session_id: Option<&str>,
    user_text: &str,
) -> Result<PreparedTurn> {
    let t0 = Instant::now();
    let opts = RetrieveOptions {
        top_k: cfg.chat.top_k,
        min_score: cfg.chat.min_retrieval_score,
        exclude_session: session_id.map(str::to_string),
        conversational_damping: cfg.chat.conversational_damping,
        ..Default::default()
    };
    let query = if query_context.is_empty() {
        user_text.to_string()
    } else {
        format!("{}\n{}", query_context.join("\n"), user_text)
    };
    let pack = forgetfuldb_retrieve::retrieve(store, provider, cfg, &query, &opts)?;
    let retrieve_duration_ms = t0.elapsed().as_millis() as i64;
    let context_chars: i64 = pack.memories.iter().map(|m| m.item.content.chars().count() as i64).sum();

    // Static system prompt + verbatim history + memories attached to the
    // new user message: everything before the new message is identical to
    // the previous request, keeping the model's prefix KV-cache valid so
    // each turn only evaluates new tokens.
    let mut messages = Vec::with_capacity(history.len() + 2);
    messages.push(ChatMessage::new("system", system_prompt.to_string()));
    messages.extend_from_slice(history);
    messages.push(ChatMessage::new("user", wrap_user_message(user_text, &pack)));

    Ok(PreparedTurn {
        user_text: user_text.to_string(),
        messages,
        pack,
        context_chars,
        retrieve_duration_ms,
    })
}

/// The ingest request for one chat message (user or assistant role).
/// The `session:<id>` tag lets retrieval exclude the live session's own
/// turns from its context pack (they're already in the prompt as history).
pub fn chat_ingest_request(session_id: &str, role: &str, text: &str) -> IngestRequest {
    IngestRequest {
        text: text.to_string(),
        source: Some("chat".to_string()),
        tags: vec![format!("session:{session_id}")],
        // User turns: episodic by default (consolidation reclassifies).
        // Assistant turns: fast-decaying raw events.
        memory_type: if role == "assistant" { Some(MemoryType::RawEvent) } else { None },
        session_id: Some(session_id.to_string()),
        role: Some(role.to_string()),
    }
}

/// Assemble the metrics row for a completed turn.
pub fn build_chat_turn(
    session_id: &str,
    prepared: &PreparedTurn,
    reply: &str,
    usage: &ChatUsage,
    backend_name: &str,
    model: &str,
) -> ChatTurn {
    ChatTurn {
        id: new_id("turn", &prepared.user_text),
        session_id: Some(session_id.to_string()),
        created_at: now_unix(),
        user_text: prepared.user_text.clone(),
        assistant_text: reply.to_string(),
        model: model.to_string(),
        backend: backend_name.to_string(),
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_duration_ms: usage.total_duration_ms,
        llm_duration_ms: usage.llm_duration_ms,
        retrieve_duration_ms: prepared.retrieve_duration_ms,
        context_memory_count: prepared.pack.memories.len() as i64,
        context_chars: prepared.context_chars,
        memory_ids: prepared.pack.memories.iter().map(|m| m.item.id.clone()).collect(),
    }
}

/// Synchronous completion of a turn: ingest both messages and record
/// metrics. Used by the proxy; the `iforgot` chat uses [`MemoryWriter`]
/// instead so none of this sits on the conversation path.
#[allow(clippy::too_many_arguments)]
pub fn finish_turn(
    store: &Store,
    bloom: &mut BloomFilter,
    provider: &dyn EmbeddingProvider,
    cfg: &Config,
    session_id: &str,
    prepared: &PreparedTurn,
    reply: &str,
    usage: &ChatUsage,
    backend_name: &str,
    model: &str,
) -> Result<ChatTurn> {
    ingest(store, bloom, provider, cfg, chat_ingest_request(session_id, "user", &prepared.user_text))?;
    if !reply.trim().is_empty() {
        ingest(store, bloom, provider, cfg, chat_ingest_request(session_id, "assistant", reply))?;
    }
    let turn = build_chat_turn(session_id, prepared, reply, usage, backend_name, model);
    store.insert_chat_turn(&turn)?;
    // Live "used together" associations for the proxy path too.
    if turn.memory_ids.len() >= 2 {
        store.bump_cooccurrence_edges(&turn.memory_ids, now_unix())?;
    }
    Ok(turn)
}

/// Result of one full chat turn.
pub struct TurnResult {
    pub reply: String,
    pub pack: ContextPack,
    pub turn: ChatTurn,
    /// A tool the model is asking to run, parsed from its reply. The
    /// frontend decides whether to confirm and execute it.
    pub pending_tool: Option<ToolCall>,
}

/// Convenience wrapper owning everything a chat frontend needs.
/// Reads (retrieval) use this struct's store connection; all writes go
/// through the background [`MemoryWriter`] thread (WAL mode lets the two
/// connections coexist).
pub struct Agent {
    pub store: Store,
    writer: MemoryWriter,
    provider: Box<dyn EmbeddingProvider>,
    pub cfg: Config,
    pub backend: ChatBackend,
    pub tools: ToolRegistry,
    /// Base persona plus the tools section, composed once. Kept separate
    /// from `cfg.chat.system_prompt` so persisting config (on model
    /// switch) never bakes the tool instructions into the file.
    system_prompt: String,
    pub session_id: String,
    history: Vec<ChatMessage>,
    /// Raw recent user messages (oldest first, capped at
    /// `chat.query_context_turns`). History keeps the *wrapped* user
    /// messages — memory blocks included — so it can't be reused for
    /// retrieval queries without feeding old memories back into retrieval.
    recent_user_texts: Vec<String>,
}

impl Agent {
    pub fn new(mut cfg: Config) -> Result<Agent> {
        let store = Store::open(std::path::Path::new(&cfg.sqlite_path))?;
        // Build the embedding provider first. If a configured Ollama model
        // is unreachable at startup, don't brick the whole session — warn
        // and fall back to the built-in hashed_bow so chat still works (the
        // dimension-mismatch warning then nudges the user to fix/re-embed).
        let provider = match forgetfuldb_embed::create_provider_from_config(&cfg) {
            Ok(p) => p,
            Err(e) => {
                eprintln!(
                    "warning: embedding backend '{}' unavailable ({e}); falling back to hashed_bow for this session",
                    cfg.embedding_backend
                );
                cfg.embedding_backend = "hashed_bow".to_string();
                cfg.embedding_model = String::new();
                forgetfuldb_embed::create_provider_from_config(&cfg)?
            }
        };
        // Spawn the writer with the (possibly fallen-back) config so reads
        // and writes always embed with the same model.
        let writer = MemoryWriter::spawn(&cfg)?;
        let backend = ChatBackend::from_config(&cfg)?;
        let tools = ToolRegistry::from_config(&cfg.tools);
        let system_prompt = format!("{}{}", cfg.chat.system_prompt, tools.prompt_section());
        let session_id = new_id("session", "chat");
        Ok(Agent {
            store,
            writer,
            provider,
            cfg,
            backend,
            tools,
            system_prompt,
            session_id,
            history: Vec::new(),
            recent_user_texts: Vec::new(),
        })
    }

    /// Tools available this session.
    pub fn tool_list(&self) -> Vec<forgetfuldb_tools::ToolInfo> {
        self.tools.list()
    }

    /// Does running `call` need explicit user approval?
    pub fn tool_requires_confirmation(&self, call: &ToolCall) -> bool {
        self.tools.get(&call.tool).map(|t| t.requires_confirmation()).unwrap_or(true)
    }

    /// The literal action a call will perform (shown in the prompt).
    pub fn tool_preview(&self, call: &ToolCall) -> Option<String> {
        self.tools.get(&call.tool).map(|t| t.preview(&call.args))
    }

    /// Run a tool call directly. Used both for confirmed LLM proposals and
    /// for the `/cmd` slash command.
    pub fn execute_tool(&self, call: &ToolCall) -> Result<String> {
        self.tools.execute(call)
    }

    /// Build a `shell` tool call for a raw command string (for `/cmd`).
    pub fn shell_call(command: &str) -> ToolCall {
        ToolCall { tool: "shell".to_string(), args: serde_json::json!({ "command": command }) }
    }

    /// Switch the chat model for this session and persist the choice to
    /// `config_path`. The memory database is untouched: the chat model
    /// only affects generation, never storage or embeddings.
    pub fn set_model(&mut self, name: &str, config_path: &std::path::Path) -> Result<()> {
        self.backend.set_model(name);
        self.cfg.chat.model = name.to_string();
        self.cfg.save(config_path)?;
        Ok(())
    }

    /// Switch the embedding backend/model and **re-embed the whole store**,
    /// because vectors from different models are not comparable. The new
    /// provider is built and validated (Ollama reachable, model exists)
    /// *before* anything is touched, so a bad choice changes nothing. On
    /// success the retrieval provider is swapped, the background writer is
    /// respawned (so new ingests embed with the new model too), and the
    /// choice is persisted. Returns the number of memories re-embedded.
    pub fn set_embedding(
        &mut self,
        backend: &str,
        model: &str,
        config_path: &std::path::Path,
        on_progress: impl FnMut(usize, usize),
    ) -> Result<usize> {
        // Validate first: this is where an unreachable Ollama or a missing
        // model fails, leaving the session exactly as it was.
        let mut new_cfg = self.cfg.clone();
        new_cfg.embedding_backend = backend.to_string();
        new_cfg.embedding_model = model.to_string();
        let provider = forgetfuldb_embed::create_provider_from_config(&new_cfg)?;

        // Settle pending writes, then re-embed against the settled store.
        self.writer.flush();
        let label = if model.is_empty() { backend.to_string() } else { model.to_string() };
        let n = forgetfuldb_store::pipeline::reembed_all(&self.store, provider.as_ref(), &label, on_progress)?;

        // Commit: swap retrieval provider, respawn the writer with the new
        // config, persist. The old writer drops (its thread drains + joins).
        self.provider = provider;
        self.cfg = new_cfg;
        self.writer = MemoryWriter::spawn(&self.cfg)?;
        self.cfg.save(config_path)?;
        Ok(n)
    }

    /// A warning if the stored vectors don't match the active embedding
    /// provider's dimension (model changed without re-embedding), else None.
    pub fn embedding_warning(&self) -> Option<String> {
        forgetfuldb_store::pipeline::embedding_mismatch_warning(&self.store, self.provider.as_ref())
    }

    /// The active embedding identity, for the banner / `/embed` display.
    pub fn embedding_label(&self) -> String {
        if self.cfg.embedding_backend == "hashed_bow" {
            format!("hashed_bow ({}-dim, built-in)", self.provider.dim())
        } else {
            format!("{} ({}-dim, via {})", self.cfg.embedding_model, self.provider.dim(), self.cfg.embedding_backend)
        }
    }

    /// One full turn: retrieve context (the only synchronous memory work),
    /// queue the user-message ingest on the background writer, stream the
    /// reply (calling `on_token` per token), then queue the reply ingest
    /// and metrics. The conversation never waits on a memory write.
    pub async fn chat_turn(&mut self, user_text: &str, on_token: &mut dyn FnMut(&str)) -> Result<TurnResult> {
        // Settle any writes still queued from the previous turn so this
        // turn's retrieval sees them (read-your-own-writes). In practice
        // the queue drained long ago and this returns immediately.
        self.writer.flush();
        let prepared = prepare_turn(
            &self.store,
            self.provider.as_ref(),
            &self.cfg,
            &self.system_prompt,
            &self.history,
            &self.recent_user_texts,
            Some(&self.session_id),
            user_text,
        )?;

        // The user message is written to memory while the model generates.
        self.writer
            .submit(WriteJob::Ingest(chat_ingest_request(&self.session_id, "user", user_text)));

        // Remember the raw text for the next turn's retrieval query.
        self.recent_user_texts.push(user_text.to_string());
        let cap = self.cfg.chat.query_context_turns;
        if self.recent_user_texts.len() > cap {
            self.recent_user_texts.drain(..self.recent_user_texts.len() - cap);
        }

        let (reply, usage) = self.stream_and_record(&prepared, on_token).await?;
        // User-initiated turn: allow the ```bash code-block fallback.
        let pending_tool = self.parse_pending_tool(&reply, true);
        let turn = build_chat_turn(&self.session_id, &prepared, &reply, &usage, self.backend.name(), self.backend.model());

        Ok(TurnResult { reply, pack: prepared.pack, turn, pending_tool })
    }

    /// After a tool runs, feed its output back to the model for a follow-up
    /// answer. The output is sent as a fresh user message; we skip
    /// retrieval here (the prompt is already grounded) for speed. The model
    /// may chain another tool call, which the frontend handles the same way.
    pub async fn respond_to_tool(
        &mut self,
        call: &ToolCall,
        output: &str,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<TurnResult> {
        self.writer.flush();
        let preview = self.tool_preview(call).unwrap_or_else(|| call.tool.clone());
        let user_text = format!(
            "Output of tool `{}` ({}):\n```\n{}\n```\nUse this to answer my previous request.",
            call.tool, preview, output
        );
        let mut messages = Vec::with_capacity(self.history.len() + 2);
        messages.push(ChatMessage::new("system", self.system_prompt.clone()));
        messages.extend_from_slice(&self.history);
        messages.push(ChatMessage::new("user", user_text.clone()));
        let prepared = PreparedTurn {
            user_text,
            messages,
            pack: ContextPack {
                query: String::new(),
                generated_at: now_unix(),
                memories: vec![],
                min_score: 0.0,
                near_misses: vec![],
            },
            context_chars: 0,
            retrieve_duration_ms: 0,
        };

        // The tool output is worth remembering as a fast-decaying event.
        self.writer.submit(WriteJob::Ingest(chat_ingest_request(
            &self.session_id,
            "assistant",
            &format!("ran `{preview}` -> {output}"),
        )));

        let (reply, usage) = self.stream_and_record(&prepared, on_token).await?;
        // Follow-up turn: honor an explicit ```tool block (chaining) but
        // NOT the bash-code-block fallback, so a final answer that quotes a
        // command doesn't re-prompt the user in a loop.
        let pending_tool = self.parse_pending_tool(&reply, false);
        let turn = build_chat_turn(&self.session_id, &prepared, &reply, &usage, self.backend.name(), self.backend.model());
        Ok(TurnResult { reply, pack: prepared.pack, turn, pending_tool })
    }

    /// Shared tail of a turn: stream the reply, fill in usage, queue the
    /// assistant ingest + metrics, and update history.
    async fn stream_and_record(
        &mut self,
        prepared: &PreparedTurn,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<(String, ChatUsage)> {
        let t0 = Instant::now();
        let (reply, mut usage) = self.backend.chat_stream(&prepared.messages, on_token).await?;
        if usage.llm_duration_ms.is_none() {
            usage.llm_duration_ms = Some(t0.elapsed().as_millis() as i64);
        }
        if usage.total_duration_ms.is_none() {
            usage.total_duration_ms = Some(t0.elapsed().as_millis() as i64 + prepared.retrieve_duration_ms);
        }

        let turn = build_chat_turn(&self.session_id, prepared, &reply, &usage, self.backend.name(), self.backend.model());
        if !reply.trim().is_empty() {
            self.writer
                .submit(WriteJob::Ingest(chat_ingest_request(&self.session_id, "assistant", &reply)));
        }
        self.writer.submit(WriteJob::RecordTurn(turn));

        // History keeps the wrapped user message exactly as it was sent:
        // replaying different bytes next turn would invalidate the
        // model's prefix cache.
        self.history.push(prepared.messages.last().expect("prepared turn has a user message").clone());
        self.history.push(ChatMessage::new("assistant", reply.clone()));
        let max_msgs = self.cfg.chat.history_turns * 2;
        if self.history.len() > max_msgs {
            self.history.drain(..self.history.len() - max_msgs);
        }
        Ok((reply, usage))
    }

    /// Parse a tool call from a reply, if tools are registered.
    ///
    /// Two layers: first the structured ```tool block protocol; then, when
    /// `allow_shell_fallback` is set, a fallback that treats a plain
    /// ```bash / ```sh code block as a shell command. Small local models
    /// often ignore the JSON protocol and just *describe* the command in a
    /// code block, so the fallback is what makes "show my IP" actually
    /// offer to run something. It's only applied to user-initiated turns,
    /// not tool follow-ups, so a final answer that happens to quote a
    /// command can't trigger a re-prompt loop.
    fn parse_pending_tool(&self, reply: &str, allow_shell_fallback: bool) -> Option<ToolCall> {
        if self.tools.is_empty() {
            return None;
        }
        if let Some(call) = forgetfuldb_tools::parse_tool_call(reply) {
            if self.tools.get(&call.tool).is_some() {
                return Some(call);
            }
        }
        if allow_shell_fallback && self.tools.get("shell").is_some() {
            if let Some(command) = forgetfuldb_tools::extract_shell_command(reply) {
                return Some(Agent::shell_call(&command));
            }
        }
        None
    }

    /// Block until all queued memory writes have been applied.
    pub fn flush(&self) {
        self.writer.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forgetfuldb_store::pipeline::warm_bloom;

    fn setup() -> (Store, BloomFilter, Box<dyn EmbeddingProvider>, Config) {
        let store = Store::open_in_memory().unwrap();
        let bloom = warm_bloom(&store).unwrap();
        let provider = forgetfuldb_embed::create_provider("hashed_bow", 64).unwrap();
        (store, bloom, provider, Config::default())
    }

    #[test]
    fn prepare_turn_builds_cache_friendly_prompt() {
        let (store, mut bloom, provider, mut cfg) = setup();
        // This test is about prompt *structure*; disable the relevance
        // gate and damping so the seeded memory is always injected.
        cfg.chat.min_retrieval_score = 0.0;
        cfg.chat.conversational_damping = 1.0;
        // A stored memory so the context block is non-empty.
        ingest(&store, &mut bloom, provider.as_ref(), &cfg,
            chat_ingest_request("s0", "user", "I always prefer dark mode in every editor")).unwrap();

        let history = [ChatMessage::new("user", "earlier message"), ChatMessage::new("assistant", "earlier reply")];
        let prepared =
            prepare_turn(&store, provider.as_ref(), &cfg, &cfg.chat.system_prompt, &history, &[], None, "what theme do I like?")
                .unwrap();

        // system + 2 history + user
        assert_eq!(prepared.messages.len(), 4);
        // The system prompt is STATIC (cache-stable): memories live in the
        // user message instead.
        assert_eq!(prepared.messages[0].content, cfg.chat.system_prompt);
        assert_eq!(prepared.messages[1].content, "earlier message");
        let user_msg = &prepared.messages.last().unwrap().content;
        assert!(user_msg.contains("dark mode"), "memories attached to user turn");
        assert!(user_msg.ends_with("what theme do I like?"));
        // prepare_turn is read-only: no new memory rows.
        assert_eq!(store.stats().unwrap().total_memories, 1);
    }

    #[test]
    fn current_message_does_not_retrieve_itself() {
        let (store, _bloom, provider, cfg) = setup();
        let prepared =
            prepare_turn(&store, provider.as_ref(), &cfg, &cfg.chat.system_prompt, &[], &[], None, "a brand new statement")
                .unwrap();
        assert!(
            prepared.pack.memories.iter().all(|m| m.item.content != "a brand new statement"),
            "the just-sent message must not appear in its own context pack"
        );
    }

    #[test]
    fn finish_turn_records_metrics_and_ingests_reply() {
        let (store, mut bloom, provider, cfg) = setup();
        let prepared =
            prepare_turn(&store, provider.as_ref(), &cfg, &cfg.chat.system_prompt, &[], &[], None, "what is my editor theme?")
                .unwrap();
        let usage = ChatUsage {
            prompt_tokens: Some(100),
            completion_tokens: Some(12),
            total_duration_ms: Some(800),
            llm_duration_ms: Some(750),
        };
        let turn = finish_turn(
            &store, &mut bloom, provider.as_ref(), &cfg, "s1",
            &prepared, "You prefer dark mode.", &usage, "ollama", "gemma3:12b",
        )
        .unwrap();

        assert_eq!(turn.prompt_tokens, Some(100));
        // user message + assistant reply both in memory
        assert_eq!(store.stats().unwrap().total_memories, 2);
        let summary = store.chat_metrics_summary().unwrap();
        assert_eq!(summary.turns, 1);
        assert_eq!(summary.total_completion_tokens, 12);
    }

    #[test]
    fn memory_prompt_lists_retrieved_memories() {
        let (store, mut bloom, provider, cfg) = setup();
        ingest(
            &store, &mut bloom, provider.as_ref(), &cfg,
            IngestRequest {
                text: "standup is at nine thirty".into(),
                source: None, tags: vec![], memory_type: Some(MemoryType::Semantic),
                session_id: None, role: None,
            },
        )
        .unwrap();
        let pack = forgetfuldb_retrieve::retrieve(
            &store, provider.as_ref(), &cfg, "when is standup", &RetrieveOptions::default(),
        )
        .unwrap();
        let block = memory_context_block(&pack);
        assert!(block.contains("standup is at nine thirty"));
        assert!(block.contains("[semantic]"));
        let wrapped = wrap_user_message("when is standup", &pack);
        assert!(wrapped.ends_with("when is standup"));
        assert!(wrapped.contains("standup is at nine thirty"));
    }

    #[test]
    fn vague_followup_retrieves_via_conversation_context() {
        let (store, mut bloom, provider, mut cfg) = setup();
        cfg.chat.min_retrieval_score = 0.0;
        // A distilled fact from an older session about the live topic.
        ingest(&store, &mut bloom, provider.as_ref(), &cfg,
            IngestRequest {
                text: "plot perfect is a story writing app for creating characters".into(),
                source: Some("chat".into()), tags: vec![], memory_type: Some(MemoryType::Semantic),
                session_id: None, role: None,
            }).unwrap();

        // The follow-up alone names no topic at all.
        let bare = prepare_turn(
            &store, provider.as_ref(), &cfg, &cfg.chat.system_prompt, &[], &[], None,
            "something catchy and memorable",
        ).unwrap();
        // With the previous user message folded into the retrieval query,
        // the topic memory ranks despite the vague follow-up.
        let context = vec!["suggest names for my story writing app plot perfect".to_string()];
        let contextual = prepare_turn(
            &store, provider.as_ref(), &cfg, &cfg.chat.system_prompt, &[], &context, None,
            "something catchy and memorable",
        ).unwrap();

        let top_sim = |p: &PreparedTurn| p.pack.memories.first().map(|m| m.score.semantic_similarity).unwrap_or(0.0);
        assert!(
            top_sim(&contextual) > top_sim(&bare),
            "conversation context must raise similarity to the topic memory: {} vs {}",
            top_sim(&contextual), top_sim(&bare)
        );
        assert!(contextual.pack.memories[0].item.content.contains("story writing"));
        // The contextual query never leaks into the prompt itself.
        assert!(contextual.messages.last().unwrap().content.ends_with("something catchy and memorable"));
    }

    #[test]
    fn own_session_turns_are_not_reinjected_as_memories() {
        let (store, mut bloom, provider, mut cfg) = setup();
        cfg.chat.min_retrieval_score = 0.0;
        cfg.chat.conversational_damping = 1.0;
        // Same content ingested by two sessions.
        ingest(&store, &mut bloom, provider.as_ref(), &cfg,
            chat_ingest_request("live-session", "user", "the standup moved to nine thirty")).unwrap();
        ingest(&store, &mut bloom, provider.as_ref(), &cfg,
            chat_ingest_request("old-session", "user", "the demo is on friday afternoon")).unwrap();

        let prepared = prepare_turn(
            &store, provider.as_ref(), &cfg, &cfg.chat.system_prompt, &[], &[],
            Some("live-session"), "when is the standup and the demo?",
        ).unwrap();

        assert!(
            prepared.pack.memories.iter().all(|m| !m.item.content.contains("standup moved")),
            "the live session's own turns must not come back as memories"
        );
        assert!(
            prepared.pack.memories.iter().any(|m| m.item.content.contains("demo is on friday")),
            "other sessions' memories still retrieve"
        );
    }

    #[test]
    fn chat_ingest_tags_the_session() {
        let req = chat_ingest_request("s42", "user", "hello");
        assert_eq!(req.tags, vec!["session:s42".to_string()]);
    }

    #[test]
    fn token_estimate_is_sane() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
    }
}
