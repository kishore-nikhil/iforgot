//! Baseline retrieval/prompt fixtures, not an evaluation of goal-aware behavior.
use forgetfuldb_agent::{memory_context_block, wrap_user_message};
use forgetfuldb_core::config::Config;
use forgetfuldb_core::ingest::content_hash;
use forgetfuldb_core::types::{MemoryItem, MemoryType};
use forgetfuldb_embed::{EmbeddingProvider, HashedBagOfWords};
use forgetfuldb_retrieve::{preview_at, RetrieveOptions};
use forgetfuldb_store::Store;
use serde::Deserialize;
use std::collections::HashSet;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    name: String,
    events: Vec<MemoryEvent>,
    turns: Vec<ReplayTurn>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryEvent {
    id: String,
    known_at: i64,
    text: String,
    memory_type: MemoryType,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayTurn {
    as_of: i64,
    user_text: String,
    expected_ids: Vec<String>,
    absent_ids: Vec<String>,
}

fn fixtures() -> Vec<Fixture> {
    [
        include_str!("fixtures/goal_memory/support.json"),
        include_str!("fixtures/goal_memory/coding.json"),
    ]
    .iter()
    .map(|text| serde_json::from_str(text).unwrap())
    .collect()
}

fn validate(fixture: &Fixture) {
    let ids: HashSet<_> = fixture.events.iter().map(|event| &event.id).collect();
    assert_eq!(
        ids.len(),
        fixture.events.len(),
        "duplicate fixture event ID"
    );
    assert!(fixture
        .events
        .windows(2)
        .all(|pair| pair[0].known_at <= pair[1].known_at));
    for turn in &fixture.turns {
        for id in turn.expected_ids.iter().chain(&turn.absent_ids) {
            assert!(ids.contains(id), "unsupported fixture reference {id}");
        }
        for id in &turn.expected_ids {
            let event = fixture.events.iter().find(|event| event.id == *id).unwrap();
            assert!(
                event.known_at <= turn.as_of,
                "expectation leaks a later event"
            );
            assert!(!turn.absent_ids.contains(id));
        }
    }
}

// Each turn gets a fresh prefix snapshot. Do not insert later events and
// pretend a created_at cutoff can undo their effect on existing memories.
fn store_at(fixture: &Fixture, as_of: i64, provider: &dyn EmbeddingProvider) -> Store {
    let store = Store::open_in_memory().unwrap();
    for event in fixture
        .events
        .iter()
        .filter(|event| event.known_at <= as_of)
    {
        let mut item = MemoryItem::new(
            event.id.clone(),
            event.text.clone(),
            event.memory_type,
            content_hash(&event.text),
            event.known_at,
        );
        item.embedding = Some(provider.embed(&event.text));
        store.insert_memory(&item).unwrap();
    }
    store
}

#[test]
fn support_and_coding_replays_are_repeatable_and_render_only_known_evidence() {
    let provider = HashedBagOfWords::new(128);
    let cfg = Config::default();
    // Broad recall is intentional: Phase 0 validates wiring, not relevance
    // quality. Later goal-aware comparisons must share the same configuration.
    let opts = RetrieveOptions {
        top_k: 10,
        ..Default::default()
    };
    for fixture in fixtures() {
        validate(&fixture);
        for turn in &fixture.turns {
            let store = store_at(&fixture, turn.as_of, &provider);
            let before = serde_json::to_value(store.list_memories(None).unwrap()).unwrap();
            let pack =
                preview_at(&store, &provider, &cfg, &turn.user_text, &opts, turn.as_of).unwrap();
            let rerun_store = store_at(&fixture, turn.as_of, &provider);
            let rerun = preview_at(
                &rerun_store,
                &provider,
                &cfg,
                &turn.user_text,
                &opts,
                turn.as_of,
            )
            .unwrap();
            assert_eq!(
                serde_json::to_value(&pack).unwrap(),
                serde_json::to_value(&rerun).unwrap()
            );
            let ids: HashSet<_> = pack.memories.iter().map(|hit| &hit.item.id).collect();
            let expected: HashSet<_> = turn.expected_ids.iter().collect();
            assert_eq!(ids, expected, "{} at {}", fixture.name, turn.as_of);

            let context = memory_context_block(&pack);
            let prompt = wrap_user_message(&turn.user_text, &pack);
            assert!(
                prompt.ends_with(&turn.user_text),
                "latest user message stays intact"
            );
            for event in &fixture.events {
                if turn.expected_ids.contains(&event.id) {
                    assert!(context.contains(&event.text));
                }
                if turn.absent_ids.contains(&event.id) {
                    assert!(!context.contains(&event.text));
                    assert!(!prompt.contains(&event.text));
                }
            }
            if pack.memories.is_empty() {
                assert_eq!(prompt, turn.user_text);
            } else {
                assert!(context.contains("the live conversation takes precedence"));
            }
            assert_eq!(
                serde_json::to_value(store.list_memories(None).unwrap()).unwrap(),
                before
            );
            assert!(store.all_evidence().unwrap().is_empty());
            assert!(store.list_chat_turns(10).unwrap().is_empty());
        }
    }
}

#[test]
fn adding_later_events_does_not_change_an_earlier_replay() {
    let provider = HashedBagOfWords::new(128);
    for mut fixture in fixtures() {
        let as_of = fixture.turns.last().unwrap().as_of;
        let before = store_at(&fixture, as_of, &provider);
        fixture.events.push(MemoryEvent {
            id: "future-correction".into(),
            known_at: as_of + 1000,
            text: "A later correction must not rewrite the earlier evidence.".into(),
            memory_type: MemoryType::Semantic,
        });
        let after = store_at(&fixture, as_of, &provider);
        let cfg = Config::default();
        let opts = RetrieveOptions::default();
        let a = preview_at(&before, &provider, &cfg, "what happened", &opts, as_of).unwrap();
        let b = preview_at(&after, &provider, &cfg, "what happened", &opts, as_of).unwrap();
        assert_eq!(
            serde_json::to_value(a).unwrap(),
            serde_json::to_value(b).unwrap()
        );
    }
}
