//! Inferred contradiction / supersession — the *sensor* half.
//!
//! The actuator already exists: consolidation stales the target of an
//! `Updates`/`Contradicts` link. What's missing is deciding, from two similar
//! memories, that one *supersedes* the other ("migrated to SQLite" obsoletes
//! "uses Postgres") and which direction. This module is the pure, deterministic
//! core of that decision — string/structure heuristics only, no SQLite, no
//! model — so it's unit-testable like [`crate::salience`] / `epochs`, and
//! fails *safe* (it stays silent when unsure rather than staling a true fact).
//!
//! It is precision-first by construction: the caller only ever feeds it
//! *candidate* pairs (already similar + same subject), so the cues here can be
//! liberal without firing on unrelated memories.

use crate::ingest::is_stopword;
use crate::types::MemoryItem;
use std::collections::HashSet;

/// Verbs that signal a switch from one value to another. Also excluded from
/// being mistaken for a *value*.
const SWITCH_VERBS: &[&str] = &[
    "switched", "switching", "switch", "migrated", "migrating", "migrate", "moved", "moving", "move",
    "changed", "changing", "change", "upgraded", "downgraded", "rotated", "rotate", "replaced",
    "replacing", "replace", "updated",
];

/// Whole-word markers that signal a correction even without explicit values.
const CUE_MARKERS: &[&str] = &["instead", "longer", "stopped", "deprecated", "superseded", "actually", "correction"];

/// What a correction cue spelled out, if anything.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Cue {
    pub old: Option<String>,
    pub new: Option<String>,
}

/// Whether a slot holds one value at a time or many at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cardinality {
    /// Values replace each other over time (current DB, current manager).
    Single,
    /// Values coexist (preferences, lists).
    Multi,
    /// Not enough evidence to tell.
    Unknown,
}

/// A supersession decision: `loser` is staled, `winner` survives.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub loser_id: String,
    pub winner_id: String,
    pub confidence: f64,
    pub reason: String,
}

/// Split into words, trimming surrounding punctuation but keeping case
/// (capitalization is a value signal) and keeping function words (cues need
/// "from"/"to"). Empty fragments dropped.
fn raw_words(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace())
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

/// A token that could be a subject or value: not glue, not a switch verb, and
/// either ≥2 chars or numeric.
fn significant(lower: &str) -> bool {
    !is_stopword(lower)
        && !SWITCH_VERBS.contains(&lower)
        && lower.chars().any(|c| c.is_alphanumeric())
        && (lower.chars().count() >= 2 || lower.chars().any(|c| c.is_numeric()))
}

/// Capitalized or numeric tokens are the most value-like (proper nouns,
/// counts, keys).
fn value_like(word: &str) -> bool {
    word.chars().next().is_some_and(|c| c.is_uppercase()) || word.chars().any(|c| c.is_numeric())
}

fn first_significant_after(raw: &[String], lower: &[String], start: usize) -> Option<String> {
    (start..raw.len()).find(|&i| significant(&lower[i])).map(|i| raw[i].clone())
}

/// Detect a correction cue and, where the phrasing spells it out, the old/new
/// values. `None` for a plain statement of fact.
pub fn correction_cue(text: &str) -> Option<Cue> {
    let raw = raw_words(text);
    if raw.is_empty() {
        return None;
    }
    let lw: Vec<String> = raw.iter().map(|w| w.to_lowercase()).collect();

    let mut cue = Cue::default();
    let mut found = false;

    // "from X …"
    if let Some(fi) = lw.iter().position(|w| w == "from") {
        cue.old = first_significant_after(&raw, &lw, fi + 1);
        found = true;
    }
    // "<switch verb> … to Y" / "from … to Y"
    if let Some(ti) = lw.iter().position(|w| w == "to") {
        let switched_before = lw[..ti].iter().any(|w| SWITCH_VERBS.contains(&w.as_str()) || w == "from");
        if switched_before {
            cue.new = first_significant_after(&raw, &lw, ti + 1);
            found = true;
        }
    }
    // "instead of X"
    if let Some(ii) = lw.iter().position(|w| w == "instead") {
        let after = lw[ii..].iter().position(|w| w == "of").map(|p| ii + p + 1).unwrap_or(ii + 1);
        cue.old = cue.old.or_else(|| first_significant_after(&raw, &lw, after));
        found = true;
    }
    // "no longer X" / "stopped … X"
    if lw.windows(2).any(|w| w[0] == "no" && w[1] == "longer") || lw.iter().any(|w| w == "stopped") {
        found = true;
        if cue.old.is_none() {
            if let Some(pi) = lw.iter().position(|w| w == "longer" || w == "stopped") {
                cue.old = first_significant_after(&raw, &lw, pi + 1);
            }
        }
    }
    // bare switch verb or other marker
    if lw.iter().any(|w| SWITCH_VERBS.contains(&w.as_str()) || CUE_MARKERS.contains(&w.as_str())) {
        found = true;
    }

    found.then_some(cue)
}

/// The noun phrase after the first determiner/possessive — the *slot* a
/// statement is about ("my **manager**", "the **api key**").
pub fn slot_phrase(text: &str) -> Option<String> {
    const DET: &[&str] = &["the", "a", "an", "my", "your", "his", "her", "our", "their", "its", "this", "that"];
    const BOUNDARY: &[&str] =
        &["is", "are", "was", "were", "be", "been", "to", "from", "of", "with", "on", "at", "in", "for", "by", "into", "as", "no", "not", "now"];

    let raw = raw_words(text);
    let lw: Vec<String> = raw.iter().map(|w| w.to_lowercase()).collect();
    let di = lw.iter().position(|w| DET.contains(&w.as_str()))?;

    let mut phrase = Vec::new();
    for i in (di + 1)..raw.len() {
        if BOUNDARY.contains(&lw[i].as_str()) || phrase.len() >= 4 {
            break;
        }
        phrase.push(raw[i].clone());
    }
    (!phrase.is_empty()).then(|| phrase.join(" "))
}

/// The differing values between two *near-identical* texts: the significant
/// token unique to each side, preferring value-like (capitalized/numeric)
/// ones. `None` if the texts diverge too much to be a clean value swap.
pub fn value_diff(a: &str, b: &str) -> Option<(String, String)> {
    let sig = |text: &str| -> Vec<String> {
        raw_words(text).into_iter().filter(|w| significant(&w.to_lowercase())).collect()
    };
    let aw = sig(a);
    let bw = sig(b);
    let aset: HashSet<String> = aw.iter().map(|w| w.to_lowercase()).collect();
    let bset: HashSet<String> = bw.iter().map(|w| w.to_lowercase()).collect();

    let a_only: Vec<&String> = aw.iter().filter(|w| !bset.contains(&w.to_lowercase())).collect();
    let b_only: Vec<&String> = bw.iter().filter(|w| !aset.contains(&w.to_lowercase())).collect();
    // A genuine value swap differs in only a few tokens, not whole sentences.
    if a_only.is_empty() || b_only.is_empty() || a_only.len() > 3 || b_only.len() > 3 {
        return None;
    }
    let pick = |v: &[&String]| -> String {
        v.iter().find(|w| value_like(w)).copied().or_else(|| v.last().copied()).cloned().unwrap()
    };
    Some((pick(&a_only), pick(&b_only)))
}

/// A memory's value signature: its value-like (capitalized/numeric) tokens,
/// lowercased, sorted and deduped. Two memories asserting the same value share
/// a signature — used to group a slot's memories by value when judging
/// replacement vs accumulation. Empty when the value isn't value-like (a
/// lowercase preference), which keeps such slots conservatively `Unknown`.
pub fn value_tokens(text: &str) -> Vec<String> {
    let mut v: Vec<String> = raw_words(text)
        .into_iter()
        .filter(|w| significant(&w.to_lowercase()) && value_like(w))
        .map(|w| w.to_lowercase())
        .collect();
    v.sort();
    v.dedup();
    v
}

/// Classify a slot from how its values sit in time: sequential, non-overlapping
/// spans → replacement (Single); overlapping → coexistence (Multi). `spans` is
/// each distinct value's `(earliest, latest)` mention. Fewer than two values,
/// or single-point spans, is too thin to judge → `Unknown`.
pub fn classify_cardinality(spans: &[(i64, i64)], overlap_threshold: f64) -> Cardinality {
    if spans.len() < 2 || spans.iter().all(|(s, e)| s == e) {
        return Cardinality::Unknown;
    }
    let mut max_ratio = 0.0_f64;
    for i in 0..spans.len() {
        for j in (i + 1)..spans.len() {
            let (s1, e1) = spans[i];
            let (s2, e2) = spans[j];
            let overlap = (e1.min(e2) - s1.max(s2)).max(0) as f64;
            let union = (e1.max(e2) - s1.min(s2)).max(1) as f64;
            max_ratio = max_ratio.max(overlap / union);
        }
    }
    if max_ratio > overlap_threshold {
        Cardinality::Multi
    } else {
        Cardinality::Single
    }
}

/// Decide whether `newer` supersedes `older` (the two are a candidate pair,
/// `older.created_at <= newer.created_at`). Confidence tiers let the caller
/// gate: a correction cue is decisive; a singular-slot value change is solid;
/// an unconfirmed value change is low-confidence (deferred, not acted on);
/// coexisting values are no contradiction at all.
pub fn judge(older: &MemoryItem, newer: &MemoryItem, cardinality: Cardinality) -> Option<Verdict> {
    let verdict = |confidence: f64, reason: String| Verdict {
        loser_id: older.id.clone(),
        winner_id: newer.id.clone(),
        confidence,
        reason,
    };

    // 1. An explicit correction in the newer memory is decisive and directional.
    if let Some(cue) = correction_cue(&newer.content) {
        let detail = match (cue.old.as_deref(), cue.new.as_deref()) {
            (Some(o), Some(n)) => format!("{o} → {n}"),
            (None, Some(n)) => format!("→ {n}"),
            (Some(o), None) => format!("{o} → (current)"),
            (None, None) => value_diff(&older.content, &newer.content).map(|(o, n)| format!("{o} → {n}")).unwrap_or_default(),
        };
        let confidence = if cue.old.is_some() || cue.new.is_some() { 0.9 } else { 0.82 };
        return Some(verdict(confidence, format!("correction cue: {detail}")));
    }

    // 2. A value swap, judged by the slot's cardinality.
    if let Some((o, n)) = value_diff(&older.content, &newer.content) {
        return match cardinality {
            Cardinality::Single => Some(verdict(0.7, format!("singular slot changed: {o} → {n}"))),
            Cardinality::Unknown => Some(verdict(0.4, format!("value changed (slot unconfirmed): {o} → {n}"))),
            Cardinality::Multi => None, // coexisting values — not a contradiction
        };
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MemoryType;

    fn mem(id: &str, content: &str, created_at: i64) -> MemoryItem {
        MemoryItem::new(id.into(), content.into(), MemoryType::Semantic, format!("h{id}"), created_at)
    }

    #[test]
    fn cue_from_to_extracts_both_values() {
        let c = correction_cue("we switched from Postgres to SQLite last week").unwrap();
        assert_eq!(c.old.as_deref(), Some("Postgres"));
        assert_eq!(c.new.as_deref(), Some("SQLite"));
    }

    #[test]
    fn cue_migrated_to_extracts_new_value() {
        let c = correction_cue("migrated the main database to SQLite").unwrap();
        assert_eq!(c.new.as_deref(), Some("SQLite"));
        assert_eq!(c.old, None);
    }

    #[test]
    fn cue_no_longer_and_instead() {
        assert!(correction_cue("I no longer use Postgres").is_some());
        assert_eq!(correction_cue("using SQLite instead of Postgres").unwrap().old.as_deref(), Some("Postgres"));
    }

    #[test]
    fn plain_statement_has_no_cue() {
        assert!(correction_cue("the main database runs on Postgres").is_none());
        assert!(correction_cue("I like coffee").is_none());
    }

    #[test]
    fn value_diff_picks_the_changed_value() {
        assert_eq!(
            value_diff("the database runs on Postgres", "the database runs on SQLite").unwrap(),
            ("Postgres".to_string(), "SQLite".to_string())
        );
        assert_eq!(
            value_diff("the rate limit is 100", "the rate limit is 500").unwrap(),
            ("100".to_string(), "500".to_string())
        );
    }

    #[test]
    fn value_diff_none_when_texts_diverge() {
        assert!(value_diff("the database runs on Postgres", "i had a big sandwich for lunch today").is_none());
    }

    #[test]
    fn slot_phrase_extracts_noun_after_determiner() {
        assert_eq!(slot_phrase("my manager is Bob").as_deref(), Some("manager"));
        assert_eq!(slot_phrase("the api key is abc123").as_deref(), Some("api key"));
        assert_eq!(slot_phrase("Postgres is great").as_deref(), None); // no determiner
    }

    #[test]
    fn cardinality_replacement_vs_accumulation() {
        assert_eq!(classify_cardinality(&[(0, 10), (20, 30)], 0.2), Cardinality::Single); // sequential
        assert_eq!(classify_cardinality(&[(0, 30), (10, 40)], 0.2), Cardinality::Multi); // overlapping
        assert_eq!(classify_cardinality(&[(5, 5), (9, 9)], 0.2), Cardinality::Unknown); // single points
    }

    #[test]
    fn judge_cue_supersedes_with_high_confidence() {
        let old = mem("a", "the database runs on Postgres", 100);
        let new = mem("b", "we migrated the database from Postgres to SQLite", 200);
        let v = judge(&old, &new, Cardinality::Unknown).unwrap();
        assert_eq!(v.loser_id, "a");
        assert_eq!(v.winner_id, "b");
        assert!(v.confidence >= 0.85, "cue should be decisive: {}", v.confidence);
        assert!(v.reason.contains("Postgres") && v.reason.contains("SQLite"));
    }

    #[test]
    fn judge_coexisting_values_are_not_a_contradiction() {
        let a = mem("a", "I like coffee", 100);
        let b = mem("b", "I like tea", 200);
        assert!(judge(&a, &b, Cardinality::Multi).is_none(), "coffee and tea both hold");
    }

    #[test]
    fn judge_singular_slot_change_is_solid_but_unknown_is_deferred() {
        let old = mem("a", "the rate limit is 100", 100);
        let new = mem("b", "the rate limit is 500", 200);
        let solid = judge(&old, &new, Cardinality::Single).unwrap();
        assert!(solid.confidence >= 0.6 && solid.confidence < 0.9);
        let deferred = judge(&old, &new, Cardinality::Unknown).unwrap();
        assert!(deferred.confidence < 0.5, "thin evidence stays below the acceptance bar");
    }
}
