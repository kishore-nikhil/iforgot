//! Ollama-backed embeddings: a real local embedding model (embeddinggemma,
//! nomic-embed-text, qwen3-embedding, …) served by Ollama, in place of the
//! deterministic `hashed_bow` placeholder.
//!
//! ## Why a worker thread
//!
//! [`EmbeddingProvider::embed`] is **synchronous**, but Ollama is an HTTP
//! call, and `embed` is invoked from both plain sync code (the CLI) and
//! from inside the tokio runtime (the server, the chat agent). A blocking
//! HTTP client cannot be created or driven on a tokio worker thread. So
//! the provider owns one dedicated OS thread that builds a
//! `reqwest::blocking::Client` and services embed requests over a channel;
//! `embed` just sends the text and blocks on the reply. The blocking
//! client therefore never touches the async runtime, and the trait stays
//! sync so nothing else in the system changes.
//!
//! Failures degrade rather than panic: a network/model error yields a
//! zero vector (which simply won't match anything in cosine space) plus a
//! logged warning, plus a non-zero exit only at construction time when the
//! probe fails — so a misconfigured model is caught immediately, not
//! silently.

use crate::EmbeddingProvider;
use anyhow::{anyhow, Result};
use std::sync::mpsc;
use std::thread::JoinHandle;

type EmbedResult = std::result::Result<Vec<f32>, String>;

struct Job {
    text: String,
    reply: mpsc::Sender<EmbedResult>,
}

/// Embeddings from a local Ollama model via `POST /api/embed`.
pub struct OllamaEmbeddings {
    model: String,
    dim: usize,
    tx: mpsc::Sender<Job>,
    _handle: JoinHandle<()>,
}

impl OllamaEmbeddings {
    /// Connect to `base_url` and probe `model` for its embedding
    /// dimension. Returns an error if Ollama is unreachable or the model
    /// isn't an embedding model (so the caller can fall back to
    /// `hashed_bow` and tell the user).
    pub fn new(base_url: String, model: String) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<Job>();
        let url = format!("{}/api/embed", base_url.trim_end_matches('/'));
        let model_for_thread = model.clone();
        let handle = std::thread::Builder::new()
            .name("forgetfuldb-embed".to_string())
            .spawn(move || {
                // Built HERE, on this non-tokio thread, so the blocking
                // client never collides with an async runtime.
                let client = reqwest::blocking::Client::new();
                for job in rx {
                    let _ = job.reply.send(embed_once(&client, &url, &model_for_thread, &job.text));
                }
            })?;

        let provider = OllamaEmbeddings { model: model.clone(), dim: 0, tx, _handle: handle };
        let probe = provider
            .call("dimension probe")
            .map_err(|e| anyhow!("Ollama embedding model '{model}' unavailable: {e}"))?;
        anyhow::ensure!(!probe.is_empty(), "Ollama model '{model}' returned an empty embedding");
        Ok(OllamaEmbeddings { dim: probe.len(), ..provider })
    }

    /// Send one text to the worker and block for its embedding.
    fn call(&self, text: &str) -> std::result::Result<Vec<f32>, String> {
        let (rtx, rrx) = mpsc::channel();
        self.tx
            .send(Job { text: text.to_string(), reply: rtx })
            .map_err(|_| "embedding worker thread is gone".to_string())?;
        rrx.recv().map_err(|_| "embedding worker dropped the reply".to_string())?
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

impl EmbeddingProvider for OllamaEmbeddings {
    fn name(&self) -> &'static str {
        "ollama"
    }

    /// Fold in the concrete model name so `embeddinggemma` and
    /// `nomic-embed-text` are distinguishable provenance, not both `"ollama"`.
    fn model_id(&self) -> String {
        format!("ollama:{}", self.model)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        match self.call(text) {
            Ok(mut v) => {
                // Guard the contract: callers and stored vectors assume a
                // fixed dimension.
                if v.len() != self.dim {
                    v.resize(self.dim, 0.0);
                }
                v
            }
            Err(e) => {
                eprintln!("forgetfuldb-embed: ollama embed failed ({e}); using zero vector");
                vec![0.0; self.dim]
            }
        }
    }
}

fn embed_once(client: &reqwest::blocking::Client, url: &str, model: &str, text: &str) -> EmbedResult {
    let resp = client
        .post(url)
        .json(&serde_json::json!({ "model": model, "input": text }))
        .send()
        .map_err(|e| e.to_string())?;
    let body: serde_json::Value = resp.json().map_err(|e| e.to_string())?;
    if let Some(err) = body.get("error").and_then(|e| e.as_str()) {
        return Err(err.to_string());
    }
    // /api/embed returns {"embeddings": [[...]]}; we send one input.
    let arr = body
        .get("embeddings")
        .and_then(|e| e.get(0))
        .and_then(|e| e.as_array())
        .ok_or_else(|| "response had no embeddings array".to_string())?;
    Ok(arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect())
}
