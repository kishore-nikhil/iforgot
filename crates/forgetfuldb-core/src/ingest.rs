//! Pure ingest-side helpers: normalization, content hashing, lightweight
//! keyword/entity extraction and the initial importance heuristic.
//!
//! Everything here is deterministic and local — no models, no network.

use crate::types::{
    ConversationFrame, InputMode, MemoryCandidate, MemoryCandidateType, MemoryType, ParseResult,
    UncertaintyReason,
};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

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

/// Classify an input before extraction so pasted/code/log material does not
/// explode into high-importance personal memories.
pub fn classify_input_mode(text: &str) -> InputMode {
    let len = text.chars().count();
    let lines = text.lines().count();
    let lower = text.to_lowercase();
    let code_markers = text.contains("```")
        || lower.contains("fn ")
        || lower.contains("class ")
        || lower.contains("import ")
        || lower.contains("use ")
        || lower.contains("const ")
        || lower.contains("let ")
        || lower.contains("=>")
        || lower.contains("{") && lower.contains("}");
    let log_markers = lower.contains("stack trace")
        || lower.contains("exception")
        || lower.contains("traceback")
        || lower.contains(" error ")
        || lower.contains("warn ")
        || lower.matches(':').count() > 20 && lower.matches('\n').count() > 8;
    let markdown_doc = text
        .lines()
        .filter(|l| l.trim_start().starts_with('#'))
        .count()
        >= 2
        || text
            .lines()
            .filter(|l| l.trim_start().starts_with("- "))
            .count()
            >= 6;
    let article_markers = lower.contains("abstract")
        || lower.contains("introduction")
        || lower.contains("references")
        || lower.contains("conclusion");

    if code_markers && (len > 300 || lines > 6) {
        InputMode::CodeBlock
    } else if log_markers {
        InputMode::LogDump
    } else if article_markers && len > 1_000 {
        InputMode::ArticleDraft
    } else if len > 3_000 || markdown_doc && len > 800 {
        InputMode::PastedDocument
    } else if len >= 500 || lines > 8 {
        InputMode::MixedMessage
    } else {
        InputMode::ConversationMessage
    }
}

fn first_phrase_after<'a>(lower: &'a str, cue: &str) -> Option<&'a str> {
    let idx = lower.find(cue)?;
    let after = lower[idx + cue.len()..]
        .trim_start_matches(|c: char| c.is_whitespace() || c == ':' || c == ',');
    let end = after
        .find(|c: char| matches!(c, '.' | '!' | '?' | '\n' | ';'))
        .unwrap_or(after.len());
    let phrase = after[..end].trim();
    (!phrase.is_empty()).then_some(phrase)
}

fn object_from_tokens(text: &str) -> Option<String> {
    let toks = tokenize(text);
    toks.into_iter()
        .find(|t| t != "love" && t != "prefer" && t != "hate" && t != "want" && t != "need")
}

fn active_scope(frame: Option<&ConversationFrame>) -> Option<String> {
    let f = frame?;
    f.active_location
        .as_ref()
        .map(|l| format!("location:{l}"))
        .or_else(|| f.active_project.as_ref().map(|p| format!("project:{p}")))
        .or_else(|| f.active_topics.first().map(|t| format!("topic:{t}")))
}

/// Lightweight deterministic extraction. It errs toward weak evidence on
/// ambiguity; callers decide whether to persist the candidate as memory,
/// evidence, or raw/source context.
pub fn extract_memory_candidates(text: &str, frame: Option<&ConversationFrame>) -> ParseResult {
    let input_mode = classify_input_mode(text);
    let lower = text.to_lowercase();
    let tokens = tokenize(text);
    let mut candidates = Vec::new();
    let mut reasons = Vec::new();

    if input_mode.is_long_source() {
        reasons.push(UncertaintyReason::LongInput);
        if input_mode == InputMode::PastedDocument {
            reasons.push(UncertaintyReason::PastedDocument);
        }
    }
    if text.trim().split_whitespace().count() < 3 {
        reasons.push(UncertaintyReason::FragmentSentence);
    }

    let scope = active_scope(frame);
    let has_user_subject = lower.contains("i ")
        || lower.starts_with("i'")
        || lower.starts_with("i ")
        || lower.contains(" my ");

    let strong_pref = [
        "i love ",
        "i really like ",
        "my favorite ",
        "i prefer ",
        "i hate ",
    ];
    for cue in strong_pref {
        if let Some(obj) = first_phrase_after(&lower, cue).and_then(object_from_tokens) {
            candidates.push(MemoryCandidate {
                candidate_type: MemoryCandidateType::PreferenceStrong,
                subject: Some("user".to_string()),
                predicate: Some(
                    if cue.contains("hate") {
                        "dislikes"
                    } else {
                        "likes"
                    }
                    .to_string(),
                ),
                object: Some(obj),
                scope: scope.clone(),
                confidence: 0.90,
                text: text.to_string(),
            });
            break;
        }
    }

    let weak_pref = [
        "i like ",
        "not my thing",
        "meh",
        "solid",
        "delicious",
        "slaps",
        "goated",
    ];
    if candidates.is_empty() {
        for cue in weak_pref {
            if lower.contains(cue) {
                let obj = first_phrase_after(&lower, cue)
                    .and_then(object_from_tokens)
                    .or_else(|| tokens.iter().rev().find(|t| t.len() > 2).cloned());
                if let Some(obj) = obj {
                    candidates.push(MemoryCandidate {
                        candidate_type: MemoryCandidateType::PreferenceWeak,
                        subject: Some("user".to_string()),
                        predicate: Some(
                            if cue == "not my thing" || cue == "meh" {
                                "weak_negative_sentiment"
                            } else {
                                "positive_sentiment"
                            }
                            .to_string(),
                        ),
                        object: Some(obj),
                        scope: scope.clone(),
                        confidence: if has_user_subject { 0.65 } else { 0.55 },
                        text: text.to_string(),
                    });
                    if !has_user_subject {
                        reasons.push(UncertaintyReason::WeakCueOnly);
                    }
                    break;
                }
            }
        }
    }

    for cue in ["i want ", "i need ", "my goal is ", "goal:"] {
        if let Some(goal) = first_phrase_after(&lower, cue) {
            candidates.push(MemoryCandidate {
                candidate_type: MemoryCandidateType::GoalCandidate,
                subject: Some("user".to_string()),
                predicate: Some("goal".to_string()),
                object: Some(goal.chars().take(120).collect()),
                scope: scope.clone(),
                confidence: 0.72,
                text: text.to_string(),
            });
            break;
        }
    }

    for cue in ["i am ", "i'm ", "my name is ", "call me "] {
        if let Some(value) = first_phrase_after(&lower, cue) {
            candidates.push(MemoryCandidate {
                candidate_type: MemoryCandidateType::IdentityCandidate,
                subject: Some("user".to_string()),
                predicate: Some("identity".to_string()),
                object: Some(value.chars().take(80).collect()),
                scope: None,
                confidence: 0.78,
                text: text.to_string(),
            });
            break;
        }
    }

    if lower.contains("actually")
        || lower.contains("don't like")
        || lower.contains("do not like")
        || lower.contains("correction")
    {
        candidates.push(MemoryCandidate {
            candidate_type: MemoryCandidateType::CorrectionSignal,
            subject: Some("user".to_string()),
            predicate: Some("correction".to_string()),
            object: tokens.first().cloned(),
            scope: scope.clone(),
            confidence: 0.80,
            text: text.to_string(),
        });
    }

    let mut seen = HashSet::new();
    for tok in tokens.iter().filter(|t| t.len() > 3).take(8) {
        if seen.insert(tok) {
            candidates.push(MemoryCandidate {
                candidate_type: MemoryCandidateType::EntityMention,
                subject: None,
                predicate: Some("mentions".to_string()),
                object: Some(tok.clone()),
                scope: scope.clone(),
                confidence: 0.35,
                text: tok.clone(),
            });
        }
    }

    if candidates.iter().all(|c| c.object.is_none()) {
        reasons.push(UncertaintyReason::MissingObject);
    }
    if candidates.is_empty() {
        reasons.push(UncertaintyReason::NoKnownEntity);
    }

    let best = candidates
        .iter()
        .map(|c| c.confidence)
        .fold(0.0_f64, f64::max);
    let penalty = if input_mode.is_long_source() {
        0.25
    } else {
        0.0
    } + 0.03 * reasons.len() as f64;
    ParseResult {
        candidates,
        confidence: (best - penalty).clamp(0.0, 1.0),
        uncertainty_reasons: reasons,
    }
}

/// Chunk long inputs on paragraph-ish boundaries. The policy approximates
/// the 300-700 token target without pulling in a tokenizer dependency.
pub fn chunk_source_text(text: &str, target_tokens: usize, overlap_tokens: usize) -> Vec<String> {
    let target = target_tokens.max(80);
    let overlap = overlap_tokens.min(target / 2);
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < words.len() {
        let end = (start + target).min(words.len());
        out.push(words[start..end].join(" "));
        if end == words.len() {
            break;
        }
        start = end.saturating_sub(overlap);
    }
    out
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
        if let Some(rest) = tag
            .strip_prefix("project:")
            .or_else(|| tag.strip_prefix("topic:"))
        {
            if !rest.is_empty() {
                return Some(rest.to_string());
            }
        }
    }
    keywords.first().cloned()
}

/// Cue words that suggest the user wants this remembered.
const IMPORTANCE_CUES: &[&str] = &[
    "always",
    "never",
    "remember",
    "important",
    "must",
    "prefer",
    "preference",
    "deadline",
    "password",
    "key",
    "decided",
    "decision",
    "rule",
];

/// Initial importance in [0.05, 1.0]: a base per memory type plus small
/// boosts for explicit cues, tags and substantial content.
pub fn initial_importance(text: &str, memory_type: MemoryType, tags: &[String]) -> f64 {
    let base = match memory_type {
        // Foundation is concluded by consolidation, not ingested directly;
        // if one is ever written by hand it should rank as a core trait.
        MemoryType::Foundation => 0.90,
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

    #[test]
    fn classifier_routes_long_markdown_as_source_material() {
        let text = "# One\n\n- alpha\n- beta\n- gamma\n- delta\n- epsilon\n- zeta\n\n".repeat(20);
        assert_eq!(classify_input_mode(&text), InputMode::PastedDocument);
        let parsed = extract_memory_candidates(&text, None);
        assert!(parsed
            .uncertainty_reasons
            .contains(&UncertaintyReason::LongInput));
        assert!(parsed.confidence < 0.75);
    }

    #[test]
    fn deterministic_extractor_finds_strong_preference() {
        let parsed = extract_memory_candidates("I love coffee in Philly.", None);
        let pref = parsed
            .candidates
            .iter()
            .find(|c| c.candidate_type == MemoryCandidateType::PreferenceStrong)
            .expect("strong preference candidate");
        assert_eq!(pref.subject.as_deref(), Some("user"));
        assert_eq!(pref.predicate.as_deref(), Some("likes"));
        assert_eq!(pref.object.as_deref(), Some("coffee"));
        assert!(parsed.confidence >= 0.75);
    }
}
