# Goal-memory replay fixtures

These are synthetic baseline stories for Phase 0, not measured model results.
Run with `cargo test -p forgetfuldb-agent --test goal_memory_replay`.

- `support.json`: earlier self-service goal, documented verification failure,
  changed objective and available specialists, followed by a later outcome.
- `coding.json`: integration-test failure, a repaired prerequisite, and a later
  test result that must not be visible when deciding whether to retry.

Each event has a stable `id`, `known_at` timestamp, text and memory type. Each
turn supplies `as_of`, the latest user text and expected/absent memory IDs.
The loader rejects unknown fields; tests validate IDs and event chronology.
Each turn builds a fresh SQLite prefix snapshot containing only known events.
The fixed clock controls retrieval scoring and staged-chunk expiry. The
`hashed_bow` provider and prompt renderer run locally, without an LLM or tools.

Tests check retrieved evidence, repeated output, prompt contents and absence
of memory writes. They deliberately use broad recall with a zero relevance
gate; passing does not establish goal alignment, semantic recall quality,
safe retry behavior or final-response improvement. Phase 1 adds goal and attempt
records; Phases 2–3 add projection assertions and scripted/live response checks.
Keep outcome receipt time separate from observation time when outcome events
are introduced, and preserve all related turns in one evaluation split.

`preview_at` is a scoring-clock API, not a historical database engine. Never
replay a past turn against a current store that already contains later edits,
evidence, summaries or graph updates. Build the appropriate prefix snapshot.
Project/scope filtering and goal-conditioned ranking land in later phases.
