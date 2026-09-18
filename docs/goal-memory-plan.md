# Goal-aware memory: phased delivery and PR stack

Status: implementation started, Phase 0. Baseline: `main` at
`93b455e45634bcb928f48dceda46bd2aec302b98` (reviewed 2026-09-18).
This execution plan adapts the proposed *Goal Memory Implementation and Test
Plan* dated 18 September 2026. Each phase below is a review checkpoint, not
a claim that the feature already exists. The repository owner reviews and merges.

## Product contract

Build a bounded, inspectable **memory projection** for the current task:
current objective, relevant prior goals, changed constraints, attempts,
evidence, conditional lessons and unknowns. A new explicit instruction wins
over historical similarity. Inferred intent is provisional and never grants
permission to execute an action. Casual conversation may have no goal.

The first demonstration changes from reducing support calls with self-service
to improving satisfaction for locked-out customers when human specialists
become available. Remember the failed verification path without treating the
old objective as current or an untested intervention as successful. The second
demonstration covers a failed coding action and a later repaired prerequisite.

Outcome prediction and TACOS routing are separate, deferred experiments.
Task success, usefulness of a memory, and preference recurrence remain
different signals. Mere retrieval must never count as verified success.

## Confirmed integration map

| Area | Reuse | Work needed |
| --- | --- | --- |
| core | Config defaults, candidate extraction, evidence enums | Goal revisions, attempts/events, transitions, projection contracts |
| store | Eleven migrations, memory/source/evidence records, chat turns | Atomic migrations, durable task/scope IDs, validated references, event idempotency, snapshots |
| retrieve | Relevance-first recall, scoring, graph traversal | Non-mutating frozen-time entry point, scope filtering through graphs, goal applicability, budgets |
| agent | Shared `prepare_turn`, tool proposals, background writer | Stable source IDs, task resumption, shared turn lifecycle, attempt recording |
| server | Proxy and inspector APIs | Distinct conversation/task identities, completed/cancelled stream recording, projection reads |
| UI | Retrieval inspector | Stored turn projections, evidence links, revisions and empty/pending/conflict states |
| consolidate | Evidence-derived importance, provenance links | Conditional lessons, correction handling, protection of unresolved references |

`PredictionSnapshot` and `OutcomeSnapshot` are memory-retention statistics,
not action predictions/outcomes. Preserve their meaning. `ConversationFrame`
exists but ingest passes `None`. Accepted-reuse evidence producers are unwired.
`docs/HANDOFF.md` exists; its branch, migration and some implementation claims
are outdated. Source and passing checks take precedence over that snapshot.

Baseline checks: 224 Rust tests passed, one Ollama integration test and one
documentation example skipped; UI production build passed; two pre-existing
Clippy style warnings in `core::ingest`. No live-model quality or latency result
has been established. The user's global SQLite database is not a test fixture.

## Delivery phases

| Phase | Feature-related PR slices, in dependency order | Exit gate |
| --- | --- | --- |
| 0 — Reproducible foundation | 0a: this plan, atomic migration safeguards, backed-up legacy fixture, clean Clippy baseline. 0b: non-mutating retrieval at a supplied clock, deterministic replay fixtures and model-free prompt checks. | Existing tests pass; migration failure leaves neither partial schema nor a completion marker; fixture upgrades preserve records; repeated replay is stable and does not write to memory. |
| 1 — Goals and attempts | 1a: explicit goal revisions, lifecycle and durable task/scope/source IDs behind `goal_tracking`. 1b: proposed/executed attempts, precondition signatures, append-only outcome events and exact active-task queries. | Refine/replace/cancel/complete/reopen/parallel transitions, restart recovery, invalid references, cross-scope rejection, transaction failure and duplicate events pass. |
| 2 — Memory projections | 2a: goal-conditioned retrieval, exact attempt assembly, applicability and protected token allocation. 2b: immutable JSON/text projection snapshots and shadow generation; optional bounded extractor with fallback. | New-goal precedence, relevant failures, changed conditions, evidence references, exclusions, frozen-time behavior and budget invariants pass. No tool execution during projection. |
| 3 — Chat and inspection | 3a: shared CLI/proxy lifecycle, stable API task identity and streaming completion/cancellation. 3b: `/goal`, `/projection`, `/attempts`, inspection API and UI snapshots; controlled `projection_injection`. | Both acceptance demonstrations check the injected context and final response; goal correction creates a revision; historical views stay stable; flag-off behavior is compatible. |
| 4 — Feedback and lessons | 4a: authoritative tool/business events and explicit human feedback behind `outcome_feedback`. 4b: conditional lessons, corrections, delayed outcomes, retention/deletion handling. | Duplicate/out-of-order/corrected events count once; contradictory provenance is retained; pending references survive pruning; explicit deletion removes derived exposure; retrieval earns no success credit. |
| 5 — Optional TACOS evaluation | Separate policy interface and feature flag; compare with simple outcome counts after ordinary goal memory is evaluated. | Held-out incremental benefit over Phase 4 with equivalent permissions/budgets. Otherwise keep the simpler policy. |

Initial review checkpoint: PRs 0a and 0b as one stack. Subsequent phases get
their own small stacks after review feedback is incorporated. A slice is
complete only when its stated gate passes; supporting APIs are not evidence
that goal-aware chat is already working.

### Progress

- 0a implemented: migrations commit atomically; a backed-up synthetic schema-0008
  fixture upgrades and reopens with memories, links, edges and raw events intact;
  failed SQL or a failed marker write rolls back. In-memory tests enforce foreign
  keys like disk stores. Workspace: 228 tests passed, two skipped; strict Clippy
  passed after clearing the two baseline style warnings.
- 0b next: deterministic non-mutating retrieval and replay fixtures.
- Phases 1–5 pending.

## Implementation decisions

- MVP scope boundary is a store/project, not multi-tenant authorization. Bind
  requests to an explicit scope. Project tags alone do not provide isolation.
  Scope filters must also constrain graph traversal, sources and inspector reads.
- Task identity survives session restart. Proxy requests need a documented
  conversation/task identifier instead of the current constant `proxy` identity.
  Ambiguous clients must not silently share an active goal.
- Keep goal revisions, attempt execution and event receipt distinct. Allocate
  stable source-event IDs before extraction or asynchronous ingestion; reject
  invented references. Tool proposals and declined actions are not execution.
- Add migrations, do not rewrite shipped schemas. Migration SQL and its marker
  commit together. New linked records use transactional validation, foreign
  references and producer idempotency keys; generated IDs alone are not dedup.
- Protect active/unresolved source references as soon as Phase 1 writes them;
  Phase 4 extends that policy to learned lessons. Explicit deletion wins over
  retention and must cover derived text as well as links.
- Add independent, default-off `goal_tracking`, `projection_injection`,
  `outcome_feedback` and `tacos` switches when their behavior lands. Shadow mode
  stores projections but neither injects them nor reinforces recalled memories.
- Start with explicit goals and deterministic transitions. Optional extraction
  failure must leave normal chat available; local-only mode must stay local.
- Start projection budget at 1,200 tokens, configurable. Reserve current goal,
  changed constraints and applicable failures first. Count full rendering with
  the active tokenizer or a conservative fallback; do not sever warning conditions.
- Graph chronology is not causality. Render sequence edges as temporal
  relationships in projections. Memory text is quoted evidence, not instructions.
- Permanent failures, transient timeouts and repaired prerequisites need different
  retry treatment. Include elapsed time, retry hints, budgets and idempotency.

## Replay and evaluation

Use fixed clocks, explicit IDs, deterministic embeddings and scripted model/tool
adapters. Build each replay store from only the events known at that point;
an `as_of` argument cannot undo later edits to a live database. Never claim
historical correctness by filtering creation timestamps alone.

Phase 0 seeds the support and coding stories as baseline retrieval fixtures.
They test infrastructure, not goal alignment or real response quality. Extend
them through Phases 1–3 with goal, attempt, outcome and expected-response records.
Full chat backend injection and cancellation simulation land with chat integration.

Before release, author 100 trajectories: 20 continuation, 20 changed/conflicting
goals, 20 unchanged failed attempts, 20 changed-condition/transient failures,
20 irrelevant/ambiguous/no-memory cases. Split 50 development / 50 locked test,
stratified by family; keep related turns, entities and paraphrases together.
Compare A (baseline), B (projections), C (attempts and verified lessons), then
D (optional TACOS) with the same model, context limit, tools and settings.

Report paired counts and uncertainty: goal alignment (pilot target 95%), applicable
recall (90%, plus precision), repeated failures (50% relative reduction when the
baseline denominator is nonzero), false avoidance (at most 5%), independently
verified task success, and unsupported-history rates. These are proposed targets,
not observed results. Scripted responses prove wiring, not model improvement.

Hard gates: zero deterministic goal-precedence/scope/provenance/feedback failures,
100% projection integrity, migration/restart correctness, and no unexplained
verified-task-success regression. Measure p50/p95 overhead at 1k/10k/50k memories
on named hardware and set the deployment latency budget before Phase 2 tuning.
Record extractor tokens, latency and cost separately. Real-model evaluation and
UI keyboard/narrow-screen checks are required before user-facing rollout.

## Stack and review workflow

Use `gh stack` with an explicit `main` base and `codex/goal-memory-*` branches.
Each PR describes its own behavior and tests; child PRs target their preceding
layer. Submit reviewable changes and leave merging to the repository owner.
After merges, use `gh stack sync` and revalidate affected layers before continuing.
Do not commit unrelated workspace changes.

Typical commands (only add a layer after the current slice is committed):

```sh
gh stack init --base main codex/goal-memory-foundation
gh stack add codex/goal-memory-replay
gh stack submit --auto --open
gh stack view
```

Relevant checks:

```sh
cargo test --workspace --locked --offline
cargo clippy --workspace --all-targets --locked --offline -- -D warnings
cd ui && npm run build
```
