//! The runtime `supersede_memory` tool — the opt-in, query-aware precision
//! layer over the deterministic offline contradiction detector.
//!
//! When the model, reading the memories injected this turn, notices that one
//! is an outdated version of another, it can ask to mark the stale one
//! superseded. This is free inference (the model is already reading them) and
//! query-aware, but it mutates memory from a side-channel, so the actuator
//! here is strict: both ids must be memories actually shown this turn (no
//! hallucinated ids), the staling is reversible (a flag, never a delete, that
//! the offline `revive_reasserted` pass can undo), and every call returns a
//! line for the log.

use anyhow::{bail, Context, Result};
use forgetfuldb_core::types::{LinkRelation, MemoryLink};
use forgetfuldb_store::Store;
use serde_json::Value;

/// A model-proposed supersession parsed from a `supersede_memory` tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupersedeRequest {
    pub stale_id: String,
    pub superseded_by_id: String,
    pub reason: String,
}

/// The prompt section advertising the tool. Appended to the system prompt only
/// when contradiction handling is enabled, so the default prompt is unchanged.
/// Requires the injected memory block to show ids (so the model can reference
/// them).
pub const SUPERSEDE_TOOL_PROMPT: &str = "\n\nMEMORY MAINTENANCE\n\
    If two of the background memories clearly conflict — one is an outdated \
    version of the other (an old value a newer memory corrects) — you may mark \
    the outdated one superseded by replying with ONLY a fenced `tool` block:\n\
    ```tool\n\
    {\"tool\": \"supersede_memory\", \"args\": {\"stale_id\": \"<id of the outdated memory>\", \
    \"superseded_by_id\": \"<id of the current one>\", \"reason\": \"<short why>\"}}\n\
    ```\n\
    Both ids must be memories shown to you above. Only do this when you are \
    confident; otherwise answer the user normally.";

/// Parse a `supersede_memory` tool call's args. `None` if the shape is wrong.
pub fn parse_supersede(args: &Value) -> Option<SupersedeRequest> {
    let stale_id = args.get("stale_id")?.as_str()?.trim().to_string();
    let superseded_by_id = args.get("superseded_by_id").and_then(Value::as_str).unwrap_or("").trim().to_string();
    let reason = args.get("reason").and_then(Value::as_str).unwrap_or("").trim().to_string();
    if stale_id.is_empty() {
        return None;
    }
    Some(SupersedeRequest { stale_id, superseded_by_id, reason })
}

/// Apply a model-proposed supersession against `retrieved_ids` — the ids
/// injected into *this* turn. Rejects any id not shown this turn (the
/// anti-hallucination guard), records an `Updates` edge, and reversibly
/// stales the loser. Returns a one-line log of what happened.
pub fn apply_supersede(store: &Store, retrieved_ids: &[String], req: &SupersedeRequest) -> Result<String> {
    let in_turn = |id: &str| retrieved_ids.iter().any(|r| r == id);

    if !in_turn(&req.stale_id) {
        bail!("refused: '{}' was not among the memories shown this turn", req.stale_id);
    }
    if !req.superseded_by_id.is_empty() && !in_turn(&req.superseded_by_id) {
        bail!("refused: '{}' was not among the memories shown this turn", req.superseded_by_id);
    }
    if req.superseded_by_id == req.stale_id {
        bail!("refused: a memory cannot supersede itself");
    }

    let target = store
        .get_memory(&req.stale_id)?
        .with_context(|| format!("no such memory: {}", req.stale_id))?;

    if !req.superseded_by_id.is_empty() {
        store.insert_link(&MemoryLink {
            source_id: req.superseded_by_id.clone(),
            target_id: req.stale_id.clone(),
            relation: LinkRelation::Updates,
        })?;
    }
    store.set_stale(&req.stale_id, true)?;

    let preview: String = target.content.chars().take(48).collect();
    let why = if req.reason.is_empty() { String::new() } else { format!(" ({})", req.reason) };
    Ok(format!("marked \"{}\" superseded{}", preview.trim(), why))
}

/// The structured prompt for the gated resolution call: shown only when a
/// conflict is *detected*, never in the main streaming chat. Asks the model to
/// pick a supersession among two id-tagged memories, or decline.
pub fn resolution_prompt(a_id: &str, a_content: &str, b_id: &str, b_content: &str) -> String {
    format!(
        "Two stored memories about the same thing may conflict — one may be an \
         outdated version of the other.\n\
         [{a_id}] {a_content}\n\
         [{b_id}] {b_content}\n\n\
         If one clearly supersedes the other, reply with ONLY this block:\n\
         ```tool\n\
         {{\"tool\": \"supersede_memory\", \"args\": {{\"stale_id\": \"<outdated id>\", \
         \"superseded_by_id\": \"<current id>\", \"reason\": \"<short why>\"}}}}\n\
         ```\n\
         If they are both still valid (not a real conflict), reply with exactly: NONE"
    )
}

/// Resolve one detected conflict pair via the model — the gated precision call.
/// `ask` runs the actual LLM (kept as a closure so the flow is testable with a
/// mock and so the caller owns the backend). Parses the model's tool block and,
/// if it proposes a supersession, applies it (validated + reversible). Returns
/// the log line, or `None` if the model declined or replied unusably.
pub fn resolve_pair<F>(
    store: &Store,
    retrieved_ids: &[String],
    a: (&str, &str),
    b: (&str, &str),
    ask: F,
) -> Result<Option<String>>
where
    F: FnOnce(&str) -> Result<String>,
{
    let reply = ask(&resolution_prompt(a.0, a.1, b.0, b.1))?;
    let Some(call) = forgetfuldb_tools::parse_tool_call(&reply) else {
        return Ok(None); // "NONE" or prose → no action
    };
    if call.tool != "supersede_memory" {
        return Ok(None);
    }
    let Some(req) = parse_supersede(&call.args) else {
        return Ok(None);
    };
    Ok(Some(apply_supersede(store, retrieved_ids, &req)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use forgetfuldb_core::now_unix;
    use forgetfuldb_core::types::{MemoryItem, MemoryType};

    fn store_with(items: &[(&str, &str)]) -> Store {
        let store = Store::open_in_memory().unwrap();
        for (id, content) in items {
            let m = MemoryItem::new((*id).into(), (*content).into(), MemoryType::Semantic, format!("h{id}"), now_unix());
            store.insert_memory(&m).unwrap();
        }
        store
    }

    #[test]
    fn parse_reads_the_three_fields() {
        let args = serde_json::json!({"stale_id": " a ", "superseded_by_id": "b", "reason": "rotated"});
        let req = parse_supersede(&args).unwrap();
        assert_eq!(req.stale_id, "a"); // trimmed
        assert_eq!(req.superseded_by_id, "b");
        assert_eq!(req.reason, "rotated");
        // Missing stale_id → no request.
        assert!(parse_supersede(&serde_json::json!({"reason": "x"})).is_none());
    }

    #[test]
    fn applies_and_stales_reversibly_with_an_edge() {
        let store = store_with(&[("a", "the api key is OLD"), ("b", "the api key is NEW")]);
        let retrieved = vec!["a".to_string(), "b".to_string()];
        let req = SupersedeRequest { stale_id: "a".into(), superseded_by_id: "b".into(), reason: "rotated".into() };

        let log = apply_supersede(&store, &retrieved, &req).unwrap();
        assert!(log.contains("superseded"));
        assert!(store.get_memory("a").unwrap().unwrap().stale, "the loser is staled");
        assert!(!store.get_memory("b").unwrap().unwrap().stale, "the winner survives");
        // The Updates edge is recorded (winner → loser), so the offline pass
        // can later revive it if the value is reasserted.
        let links = store.links_for("a").unwrap();
        assert!(links.iter().any(|l| l.source_id == "b" && l.target_id == "a" && l.relation == LinkRelation::Updates));
    }

    #[test]
    fn resolve_pair_applies_a_model_verdict() {
        let store = store_with(&[("a", "uses Postgres"), ("b", "migrated to SQLite")]);
        let retrieved = vec!["a".to_string(), "b".to_string()];
        // Mock LLM: returns a supersede tool block (a is outdated).
        let ask = |_prompt: &str| {
            Ok("```tool\n{\"tool\":\"supersede_memory\",\"args\":{\"stale_id\":\"a\",\"superseded_by_id\":\"b\",\"reason\":\"migrated\"}}\n```".to_string())
        };
        let out = resolve_pair(&store, &retrieved, ("a", "uses Postgres"), ("b", "migrated to SQLite"), ask).unwrap();
        assert!(out.is_some(), "the verdict is applied");
        assert!(store.get_memory("a").unwrap().unwrap().stale, "a staled per the model");
    }

    #[test]
    fn resolve_pair_does_nothing_when_model_declines() {
        let store = store_with(&[("a", "likes coffee"), ("b", "likes tea")]);
        let retrieved = vec!["a".to_string(), "b".to_string()];
        let ask = |_p: &str| Ok("NONE".to_string()); // both valid
        let out = resolve_pair(&store, &retrieved, ("a", "likes coffee"), ("b", "likes tea"), ask).unwrap();
        assert!(out.is_none());
        assert!(!store.get_memory("a").unwrap().unwrap().stale, "nothing staled");
    }

    #[test]
    fn rejects_ids_not_shown_this_turn() {
        let store = store_with(&[("a", "x"), ("b", "y")]);
        let retrieved = vec!["a".to_string()]; // only 'a' was shown
        // superseded_by 'b' wasn't shown → refuse.
        let req = SupersedeRequest { stale_id: "a".into(), superseded_by_id: "b".into(), reason: String::new() };
        assert!(apply_supersede(&store, &retrieved, &req).is_err());
        // stale_id not shown → refuse (anti-hallucination).
        let req2 = SupersedeRequest { stale_id: "z".into(), superseded_by_id: String::new(), reason: String::new() };
        assert!(apply_supersede(&store, &["a".to_string()], &req2).is_err());
        // 'a' untouched.
        assert!(!store.get_memory("a").unwrap().unwrap().stale);
    }
}
