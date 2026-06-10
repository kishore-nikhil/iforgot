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

pub use backend::{ChatBackend, ChatMessage, ChatUsage};

use anyhow::Result;
use forgetfuldb_core::config::Config;
use forgetfuldb_core::ids::new_id;
use forgetfuldb_core::now_unix;
use forgetfuldb_core::types::MemoryType;
use forgetfuldb_embed::EmbeddingProvider;
use forgetfuldb_prob::BloomFilter;
use forgetfuldb_retrieve::{ContextPack, RetrieveOptions};
use forgetfuldb_store::pipeline::{ingest, warm_bloom, IngestRequest};
use forgetfuldb_store::{ChatTurn, Store};
use std::time::Instant;

/// Rough token estimate (~4 chars/token) used only when the backend
/// doesn't report real usage.
pub fn estimate_tokens(text: &str) -> i64 {
    (text.chars().count() as i64 + 3) / 4
}

/// Render the base system prompt plus the retrieved memories.
pub fn memory_system_prompt(base: &str, pack: &ContextPack) -> String {
    let mut out = String::from(base);
    if pack.memories.is_empty() {
        out.push_str("\n\n(no stored memories matched this message yet)");
    } else {
        out.push_str("\n\nRelevant memories, most relevant first:\n");
        for m in &pack.memories {
            out.push_str(&format!("- [{}] {}\n", m.item.memory_type, m.item.content));
        }
    }
    out
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

/// Ingest the user's message and assemble the prompt (steps 1–2).
pub fn prepare_turn(
    store: &Store,
    bloom: &mut BloomFilter,
    provider: &dyn EmbeddingProvider,
    cfg: &Config,
    session_id: &str,
    history: &[ChatMessage],
    user_text: &str,
) -> Result<PreparedTurn> {
    // Retrieve BEFORE ingesting, so the user's current message can't
    // rank as a "memory" of itself in its own context pack.
    let t0 = Instant::now();
    let opts = RetrieveOptions { top_k: cfg.chat.top_k, ..Default::default() };
    let pack = forgetfuldb_retrieve::retrieve(store, provider, cfg, user_text, &opts)?;
    let retrieve_duration_ms = t0.elapsed().as_millis() as i64;

    ingest(
        store,
        bloom,
        provider,
        cfg,
        IngestRequest {
            text: user_text.to_string(),
            source: Some("chat".to_string()),
            tags: vec![],
            memory_type: None, // episodic by default; consolidation reclassifies
            session_id: Some(session_id.to_string()),
            role: Some("user".to_string()),
        },
    )?;
    let context_chars: i64 = pack.memories.iter().map(|m| m.item.content.chars().count() as i64).sum();

    let mut messages = Vec::with_capacity(history.len() + 2);
    messages.push(ChatMessage::new("system", memory_system_prompt(&cfg.chat.system_prompt, &pack)));
    messages.extend_from_slice(history);
    messages.push(ChatMessage::new("user", user_text));

    Ok(PreparedTurn {
        user_text: user_text.to_string(),
        messages,
        pack,
        context_chars,
        retrieve_duration_ms,
    })
}

/// Ingest the assistant reply and record the metrics row (steps 4–5).
/// Returns the recorded turn.
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
    if !reply.trim().is_empty() {
        // Assistant turns decay fast unless consolidation finds they matter.
        ingest(
            store,
            bloom,
            provider,
            cfg,
            IngestRequest {
                text: reply.to_string(),
                source: Some("chat".to_string()),
                tags: vec![],
                memory_type: Some(MemoryType::RawEvent),
                session_id: Some(session_id.to_string()),
                role: Some("assistant".to_string()),
            },
        )?;
    }

    let turn = ChatTurn {
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
    };
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
pub struct Agent {
    pub store: Store,
    bloom: BloomFilter,
    provider: Box<dyn EmbeddingProvider>,
    pub cfg: Config,
    pub backend: ChatBackend,
    pub session_id: String,
    history: Vec<ChatMessage>,
}

impl Agent {
    pub fn new(cfg: Config) -> Result<Agent> {
        let store = Store::open(std::path::Path::new(&cfg.sqlite_path))?;
        let bloom = warm_bloom(&store)?;
        let provider = forgetfuldb_embed::create_provider(&cfg.embedding_backend, cfg.embedding_dim)?;
        let backend = ChatBackend::from_config(&cfg)?;
        let session_id = new_id("session", "chat");
        Ok(Agent { store, bloom, provider, cfg, backend, session_id, history: Vec::new() })
    }

    /// One full turn: ingest user message, retrieve, stream the reply
    /// (calling `on_token` per token), ingest the reply, record metrics.
    pub async fn chat_turn(&mut self, user_text: &str, on_token: &mut dyn FnMut(&str)) -> Result<TurnResult> {
        let prepared = prepare_turn(
            &self.store,
            &mut self.bloom,
            self.provider.as_ref(),
            &self.cfg,
            &self.session_id,
            &self.history,
            user_text,
        )?;

        let t0 = Instant::now();
        let (reply, mut usage) = self.backend.chat_stream(&prepared.messages, on_token).await?;
        if usage.llm_duration_ms.is_none() {
            usage.llm_duration_ms = Some(t0.elapsed().as_millis() as i64);
        }
        if usage.total_duration_ms.is_none() {
            usage.total_duration_ms = Some(t0.elapsed().as_millis() as i64 + prepared.retrieve_duration_ms);
        }

        let turn = finish_turn(
            &self.store,
            &mut self.bloom,
            self.provider.as_ref(),
            &self.cfg,
            &self.session_id,
            &prepared,
            &reply,
            &usage,
            self.backend.name(),
            self.backend.model(),
        )?;

        self.history.push(ChatMessage::new("user", user_text));
        self.history.push(ChatMessage::new("assistant", reply.clone()));
        let max_msgs = self.cfg.chat.history_turns * 2;
        if self.history.len() > max_msgs {
            self.history.drain(..self.history.len() - max_msgs);
        }

        Ok(TurnResult { reply, pack: prepared.pack, turn })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (Store, BloomFilter, Box<dyn EmbeddingProvider>, Config) {
        let store = Store::open_in_memory().unwrap();
        let bloom = warm_bloom(&store).unwrap();
        let provider = forgetfuldb_embed::create_provider("hashed_bow", 64).unwrap();
        (store, bloom, provider, Config::default())
    }

    #[test]
    fn prepare_turn_ingests_and_builds_prompt() {
        let (store, mut bloom, provider, cfg) = setup();
        let prepared = prepare_turn(
            &store,
            &mut bloom,
            provider.as_ref(),
            &cfg,
            "s1",
            &[ChatMessage::new("user", "earlier message"), ChatMessage::new("assistant", "earlier reply")],
            "I always prefer dark mode",
        )
        .unwrap();

        // User message was written to memory automatically.
        assert_eq!(store.stats().unwrap().total_memories, 1);
        // system + 2 history + user
        assert_eq!(prepared.messages.len(), 4);
        assert_eq!(prepared.messages[0].role, "system");
        assert!(prepared.messages[0].content.contains("iForgot"));
        assert_eq!(prepared.messages.last().unwrap().content, "I always prefer dark mode");
    }

    #[test]
    fn current_message_does_not_retrieve_itself() {
        let (store, mut bloom, provider, cfg) = setup();
        let prepared =
            prepare_turn(&store, &mut bloom, provider.as_ref(), &cfg, "s1", &[], "a brand new statement").unwrap();
        assert!(
            prepared.pack.memories.iter().all(|m| m.item.content != "a brand new statement"),
            "the just-sent message must not appear in its own context pack"
        );
    }

    #[test]
    fn finish_turn_records_metrics_and_ingests_reply() {
        let (store, mut bloom, provider, cfg) = setup();
        let prepared = prepare_turn(&store, &mut bloom, provider.as_ref(), &cfg, "s1", &[], "what is my editor theme?").unwrap();
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
        let prompt = memory_system_prompt("Base.", &pack);
        assert!(prompt.starts_with("Base."));
        assert!(prompt.contains("standup is at nine thirty"));
        assert!(prompt.contains("[semantic]"));
    }

    #[test]
    fn token_estimate_is_sane() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
    }
}
