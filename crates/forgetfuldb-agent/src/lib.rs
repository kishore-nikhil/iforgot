//! forgetfuldb-agent
//!
//! The memory-wrapped chat loop. The LLM stays stateless; this crate makes
//! every turn update memory automatically:
//!
//! 1. the user's message is **ingested** (write) through the normal
//!    pipeline — dedup, hashing, importance scoring
//! 2. a context pack is **retrieved** (read) for the message and injected
//!    into the system prompt; retrieval itself bumps access counts, so
//!    reading is rehearsal
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
pub mod writer;

pub use backend::{ChatBackend, ChatMessage, ChatUsage};
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
use std::time::Instant;

/// Rough token estimate (~4 chars/token) used only when the backend
/// doesn't report real usage.
pub fn estimate_tokens(text: &str) -> i64 {
    (text.chars().count() as i64 + 3) / 4
}

/// Render the retrieved memories as a context block.
pub fn memory_context_block(pack: &ContextPack) -> String {
    if pack.memories.is_empty() {
        return "(no stored memories matched this message yet)".to_string();
    }
    let mut out = String::from("Relevant memories from long-term storage, most relevant first:\n");
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
pub fn prepare_turn(
    store: &Store,
    provider: &dyn EmbeddingProvider,
    cfg: &Config,
    history: &[ChatMessage],
    user_text: &str,
) -> Result<PreparedTurn> {
    let t0 = Instant::now();
    let opts = RetrieveOptions { top_k: cfg.chat.top_k, ..Default::default() };
    let pack = forgetfuldb_retrieve::retrieve(store, provider, cfg, user_text, &opts)?;
    let retrieve_duration_ms = t0.elapsed().as_millis() as i64;
    let context_chars: i64 = pack.memories.iter().map(|m| m.item.content.chars().count() as i64).sum();

    // Static system prompt + verbatim history + memories attached to the
    // new user message: everything before the new message is identical to
    // the previous request, keeping the model's prefix KV-cache valid so
    // each turn only evaluates new tokens.
    let mut messages = Vec::with_capacity(history.len() + 2);
    messages.push(ChatMessage::new("system", cfg.chat.system_prompt.clone()));
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
pub fn chat_ingest_request(session_id: &str, role: &str, text: &str) -> IngestRequest {
    IngestRequest {
        text: text.to_string(),
        source: Some("chat".to_string()),
        tags: vec![],
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
    Ok(turn)
}

/// Result of one full chat turn.
pub struct TurnResult {
    pub reply: String,
    pub pack: ContextPack,
    pub turn: ChatTurn,
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
    pub session_id: String,
    history: Vec<ChatMessage>,
}

impl Agent {
    pub fn new(cfg: Config) -> Result<Agent> {
        let store = Store::open(std::path::Path::new(&cfg.sqlite_path))?;
        let writer = MemoryWriter::spawn(&cfg)?;
        let provider = forgetfuldb_embed::create_provider(&cfg.embedding_backend, cfg.embedding_dim)?;
        let backend = ChatBackend::from_config(&cfg)?;
        let session_id = new_id("session", "chat");
        Ok(Agent { store, writer, provider, cfg, backend, session_id, history: Vec::new() })
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

    /// One full turn: retrieve context (the only synchronous memory work),
    /// queue the user-message ingest on the background writer, stream the
    /// reply (calling `on_token` per token), then queue the reply ingest
    /// and metrics. The conversation never waits on a memory write.
    pub async fn chat_turn(&mut self, user_text: &str, on_token: &mut dyn FnMut(&str)) -> Result<TurnResult> {
        // Settle any writes still queued from the previous turn so this
        // turn's retrieval sees them (read-your-own-writes). In practice
        // the queue drained long ago and this returns immediately.
        self.writer.flush();
        let prepared = prepare_turn(&self.store, self.provider.as_ref(), &self.cfg, &self.history, user_text)?;

        // The user message is written to memory while the model generates.
        self.writer
            .submit(WriteJob::Ingest(chat_ingest_request(&self.session_id, "user", user_text)));

        let t0 = Instant::now();
        let (reply, mut usage) = self.backend.chat_stream(&prepared.messages, on_token).await?;
        if usage.llm_duration_ms.is_none() {
            usage.llm_duration_ms = Some(t0.elapsed().as_millis() as i64);
        }
        if usage.total_duration_ms.is_none() {
            usage.total_duration_ms = Some(t0.elapsed().as_millis() as i64 + prepared.retrieve_duration_ms);
        }

        let turn = build_chat_turn(&self.session_id, &prepared, &reply, &usage, self.backend.name(), self.backend.model());
        if !reply.trim().is_empty() {
            self.writer
                .submit(WriteJob::Ingest(chat_ingest_request(&self.session_id, "assistant", &reply)));
        }
        self.writer.submit(WriteJob::RecordTurn(turn.clone()));

        // History keeps the wrapped user message exactly as it was sent:
        // replaying different bytes next turn would invalidate the
        // model's prefix cache.
        self.history.push(prepared.messages.last().expect("prepared turn has a user message").clone());
        self.history.push(ChatMessage::new("assistant", reply.clone()));
        let max_msgs = self.cfg.chat.history_turns * 2;
        if self.history.len() > max_msgs {
            self.history.drain(..self.history.len() - max_msgs);
        }

        Ok(TurnResult { reply, pack: prepared.pack, turn })
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
        let (store, mut bloom, provider, cfg) = setup();
        // A stored memory so the context block is non-empty.
        ingest(&store, &mut bloom, provider.as_ref(), &cfg,
            chat_ingest_request("s0", "user", "I always prefer dark mode in every editor")).unwrap();

        let history = [ChatMessage::new("user", "earlier message"), ChatMessage::new("assistant", "earlier reply")];
        let prepared =
            prepare_turn(&store, provider.as_ref(), &cfg, &history, "what theme do I like?").unwrap();

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
        let prepared = prepare_turn(&store, provider.as_ref(), &cfg, &[], "a brand new statement").unwrap();
        assert!(
            prepared.pack.memories.iter().all(|m| m.item.content != "a brand new statement"),
            "the just-sent message must not appear in its own context pack"
        );
    }

    #[test]
    fn finish_turn_records_metrics_and_ingests_reply() {
        let (store, mut bloom, provider, cfg) = setup();
        let prepared = prepare_turn(&store, provider.as_ref(), &cfg, &[], "what is my editor theme?").unwrap();
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
    fn token_estimate_is_sane() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
    }
}
