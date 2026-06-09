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

/// Build a provider from the config's `embedding_backend` name.
pub fn create_provider(backend: &str, dim: usize) -> anyhow::Result<Box<dyn EmbeddingProvider>> {
    match backend {
        "hashed_bow" => Ok(Box::new(HashedBagOfWords::new(dim))),
        other => anyhow::bail!(
            "unknown embedding backend '{other}' (available: hashed_bow; \
             candle/fastembed/llama_cpp/coreml are on the roadmap)"
        ),
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
}
