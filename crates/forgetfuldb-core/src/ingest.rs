//! Pure ingest-side helpers: normalization, content hashing, lightweight
//! keyword/entity extraction and the initial importance heuristic.
//!
//! Everything here is deterministic and local — no models, no network.

use crate::types::MemoryType;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

/// Whitespace-collapse and trim. The original casing is preserved for
/// storage; [`canonicalize`] is used for hashing/dedup.
pub fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Canonical form used only for content hashing: lowercase + collapsed
/// whitespace, so "Hello  World" and "hello world" dedup together.
pub fn canonicalize(text: &str) -> String {
    normalize(text).to_lowercase()
}

/// SHA-256 hex of the canonical form of `text`.
pub fn content_hash(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(canonicalize(text).as_bytes());
    let digest = hasher.finalize();
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Small English stopword list — enough to keep keyword extraction from
/// drowning in glue words, without pulling in an NLP dependency.
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had", "has", "have",
    "he", "her", "his", "i", "if", "in", "into", "is", "it", "its", "me", "my", "no", "not", "of",
    "on", "or", "our", "she", "so", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "to", "up", "was", "we", "were", "what", "when", "where", "which", "who",
    "will", "with", "you", "your", "about", "all", "also", "am", "been", "can", "do", "does",
    "did", "get", "got", "how", "just", "like", "more", "out", "over", "some", "than", "very",
];

pub fn is_stopword(word: &str) -> bool {
    STOPWORDS.contains(&word)
}

/// Lowercased alphanumeric tokens with stopwords removed.
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1 && !is_stopword(t))
        .map(|t| t.to_string())
        .collect()
}

/// Top-N tokens by frequency (ties broken alphabetically for determinism).
/// This is the v1 "entity extraction": cheap local heuristics first, a
/// real NER/LLM pass can replace it later behind the same signature.
pub fn extract_keywords(text: &str, max: usize) -> Vec<String> {
    let mut freq: HashMap<String, usize> = HashMap::new();
    for tok in tokenize(text) {
        *freq.entry(tok).or_insert(0) += 1;
    }
    let mut pairs: Vec<(String, usize)> = freq.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    pairs.into_iter().take(max).map(|(w, _)| w).collect()
}

/// Derive a topic: a `project:<name>` or `topic:<name>` tag wins, otherwise
/// the most frequent keyword.
pub fn guess_topic(tags: &[String], keywords: &[String]) -> Option<String> {
    for tag in tags {
        if let Some(rest) = tag.strip_prefix("project:").or_else(|| tag.strip_prefix("topic:")) {
            if !rest.is_empty() {
                return Some(rest.to_string());
            }
        }
    }
    keywords.first().cloned()
}

/// Cue words that suggest the user wants this remembered.
const IMPORTANCE_CUES: &[&str] = &[
    "always", "never", "remember", "important", "must", "prefer", "preference", "deadline",
    "password", "key", "decided", "decision", "rule",
];

/// Initial importance in [0.05, 1.0]: a base per memory type plus small
/// boosts for explicit cues, tags and substantial content.
pub fn initial_importance(text: &str, memory_type: MemoryType, tags: &[String]) -> f64 {
    let base = match memory_type {
        MemoryType::Preference => 0.75,
        MemoryType::Procedural => 0.70,
        MemoryType::Semantic => 0.65,
        MemoryType::Episodic => 0.50,
        MemoryType::RawEvent => 0.30,
        MemoryType::Archive => 0.10,
    };
    let lower = text.to_lowercase();
    let mut score: f64 = base;
    for cue in IMPORTANCE_CUES {
        if lower.contains(cue) {
            score += 0.05;
        }
    }
    if !tags.is_empty() {
        score += 0.05;
    }
    // Very short fragments ("ok", "thanks") are rarely worth keeping.
    if text.len() < 16 {
        score -= 0.15;
    }
    score.clamp(0.05, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_collapses_whitespace() {
        assert_eq!(normalize("  hello \n  world  "), "hello world");
    }

    #[test]
    fn hash_is_case_and_whitespace_insensitive() {
        assert_eq!(content_hash("Hello  World"), content_hash("hello world"));
        assert_ne!(content_hash("hello world"), content_hash("hello mars"));
    }

    #[test]
    fn keywords_skip_stopwords() {
        let kws = extract_keywords("the billing system and the billing invoices", 3);
        assert_eq!(kws[0], "billing");
        assert!(!kws.contains(&"the".to_string()));
    }

    #[test]
    fn topic_prefers_project_tag() {
        let tags = vec!["project:plotperfect".to_string()];
        let kws = vec!["billing".to_string()];
        assert_eq!(guess_topic(&tags, &kws).unwrap(), "plotperfect");
        assert_eq!(guess_topic(&[], &kws).unwrap(), "billing");
    }

    #[test]
    fn importance_reflects_type_and_cues() {
        let pref = initial_importance("I always prefer dark mode", MemoryType::Preference, &[]);
        let raw = initial_importance("ran ls in the terminal today", MemoryType::RawEvent, &[]);
        assert!(pref > raw);
        assert!(pref <= 1.0 && raw >= 0.05);
    }
}
