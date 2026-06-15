//! forgetfuldb-embed
//!
//! Abstraction over embedding providers. v1 ships a deterministic,
//! model-free placeholder (hashed bag-of-words with the signed hashing
//! trick) so the whole system works offline with zero model downloads.
//!
//! The [`EmbeddingProvider`] trait is the integration point for real
//! local models later: candle, fastembed, llama.cpp (GGUF embedding
//! models), or Core ML on Apple Silicon. Each becomes another
//! implementation selected by name in `forgetfuldb.toml`
//! (`embedding_backend = "..."`); nothing else in the system changes.

pub mod ollama;

pub use ollama::OllamaEmbeddings;

use forgetfuldb_core::config::Config;
use forgetfuldb_core::ingest::tokenize;
use std::hash::{Hash, Hasher};

/// A provider turns text into a fixed-size dense vector.
///
/// Implementations must be deterministic for the same input within a
/// database lifetime, because stored vectors are compared against freshly
/// embedded queries.
pub trait EmbeddingProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn dim(&self) -> usize;
    fn embed(&self, text: &str) -> Vec<f32>;
}

/// Deterministic placeholder embedding: each token is hashed into one of
/// `dim` buckets with a hash-derived sign (the "hashing trick"), then the
/// vector is L2-normalized. Captures lexical overlap, not true semantics —
/// good enough to exercise the retrieval pipeline end to end.
pub struct HashedBagOfWords {
    dim: usize,
}

impl HashedBagOfWords {
    pub fn new(dim: usize) -> Self {
        HashedBagOfWords { dim: dim.max(16) }
    }
}

impl EmbeddingProvider for HashedBagOfWords {
    fn name(&self) -> &'static str {
        "hashed_bow"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut vec = vec![0f32; self.dim];
        for token in tokenize(text) {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            token.hash(&mut hasher);
            let h = hasher.finish();
            let bucket = (h % self.dim as u64) as usize;
            let sign = if (h >> 63) == 0 { 1.0 } else { -1.0 };
            vec[bucket] += sign;
        }
        l2_normalize(&mut vec);
        vec
    }
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity in [-1, 1]; returns 0.0 for mismatched/zero vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Build a provider by backend name + dim. Handles the offline,
/// model-free `hashed_bow`; for networked backends like `ollama` use
/// [`create_provider_from_config`], which has the model name and URL.
pub fn create_provider(backend: &str, dim: usize) -> anyhow::Result<Box<dyn EmbeddingProvider>> {
    match backend {
        "hashed_bow" => Ok(Box::new(HashedBagOfWords::new(dim))),
        "ollama" => anyhow::bail!(
            "the 'ollama' embedding backend needs a model and URL — build it with \
             create_provider_from_config(&cfg)"
        ),
        other => anyhow::bail!(
            "unknown embedding backend '{other}' (available: hashed_bow, ollama; \
             candle/fastembed/coreml are on the roadmap)"
        ),
    }
}

/// True for localhost-style hosts, enforcing `local_only` for networked
/// embedding backends.
pub fn is_local_url(url: &str) -> bool {
    let host = url
        .split("://")
        .nth(1)
        .unwrap_or(url)
        .split('/')
        .next()
        .unwrap_or("")
        .rsplit_once(':')
        .map(|(h, _)| h)
        .unwrap_or("")
        .trim_matches(['[', ']']);
    matches!(host, "127.0.0.1" | "localhost" | "::1" | "0.0.0.0")
}

/// Build the embedding provider described by the config. This is the
/// entry point for real (networked) backends: `hashed_bow` stays the
/// default, `ollama` selects a local embedding model by name.
pub fn create_provider_from_config(cfg: &Config) -> anyhow::Result<Box<dyn EmbeddingProvider>> {
    match cfg.embedding_backend.as_str() {
        "hashed_bow" => Ok(Box::new(HashedBagOfWords::new(cfg.embedding_dim))),
        "ollama" => {
            anyhow::ensure!(
                !cfg.embedding_model.trim().is_empty(),
                "embedding_backend = \"ollama\" requires embedding_model (e.g. \"embeddinggemma\")"
            );
            if cfg.local_only {
                anyhow::ensure!(
                    is_local_url(&cfg.embedding_base_url),
                    "local_only is set, but embedding_base_url ({}) is not localhost",
                    cfg.embedding_base_url
                );
            }
            Ok(Box::new(OllamaEmbeddings::new(
                cfg.embedding_base_url.clone(),
                cfg.embedding_model.clone(),
            )?))
        }
        other => anyhow::bail!("unknown embedding backend '{other}' (available: hashed_bow, ollama)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_are_deterministic() {
        let p = HashedBagOfWords::new(64);
        assert_eq!(p.embed("billing invoices stripe"), p.embed("billing invoices stripe"));
    }

    #[test]
    fn similar_text_scores_higher_than_unrelated() {
        let p = HashedBagOfWords::new(256);
        let a = p.embed("plot perfect billing uses stripe invoices");
        let b = p.embed("billing invoices for plot perfect via stripe");
        let c = p.embed("granite countertop installation quote");
        assert!(cosine_similarity(&a, &b) > cosine_similarity(&a, &c));
    }

    #[test]
    fn cosine_of_self_is_one() {
        let p = HashedBagOfWords::new(64);
        let v = p.embed("hello world example");
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn local_url_detection() {
        for ok in ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:8080", "http://0.0.0.0:1"] {
            assert!(is_local_url(ok), "should be local: {ok}");
        }
        for bad in ["http://example.com:11434", "https://api.openai.com", "http://10.0.0.5:11434"] {
            assert!(!is_local_url(bad), "should be remote: {bad}");
        }
    }

    #[test]
    fn ollama_backend_needs_config_not_bare_name() {
        // create_provider can't build ollama (no model/url); the config
        // path is the only way in.
        assert!(create_provider("ollama", 256).is_err());
        assert!(create_provider("hashed_bow", 256).is_ok());
    }
}
