# iforgot — a forgetting engine for agents

> **Positioning.** iforgot is a *forgetting engine* — a memory layer for
> agents — backed by SQLite. It is **not** a database. A database stores
> faithfully and forever; this engine actively decides what to keep, lets
> memories decay, consolidates routine into gist, promotes traits, segments
> time, connects memories, and (eventually) generates new connections
> offline. SQLite is persistence; iforgot is the retention / decay /
> salience / consolidation / segmentation / association *policy* on top.
> "ForgetfulDB" survives as the ironic product name — the joke is that it
> forgets.

## Why forgetting is the feature

Every long-lived agent accumulates context. The naive answer — keep
everything and retrieve by similarity — fails in three ways: the prompt
grows without bound, stale facts stay confidently wrong, and the signal
(what mattered) drowns in routine (what happened). Human memory solved
this by *forgetting well*: it is lossy, reconstructive, salience-weighted,
and consolidated during sleep. iforgot is an attempt to make those
dynamics explicit, inspectable, and — because the engine has a clock the
model lacks — in places **better than human** (exact dates, exact epochs).

## The six orthogonal mechanisms

Each is a distinct axis over the same memories. The one-line thesis:

> **decay forgets · salience keeps · habit reinforces · epochs organize ·
> edges connect · dreaming creates.**

| Axis | What it does | Status |
| --- | --- | --- |
| **Decay** | Forgets the unused — exponential `exp(-λt)`, per-type half-lives | ✅ shipped |
| **Salience** | Keeps the formative — U-shaped (surprise ∨ habit), resists decay | ✅ shipped |
| **Abstraction** | Turns repetition into traits (raw → episodic → semantic → foundation) | ✅ shipped (episodic→semantic, habit→foundation, burst→gist) |
| **Epochs** | Organizes a lifetime into eras (drift-segmented) | ✅ shipped (drift segmentation; calendar grid still planned) |
| **Edges** | Connects memories (typed graph + traversal) | ◐ partial (4 edge types + one-hop activation; multi-hop traversal planned) |
| **Dreaming** | *Creates* memories/connections offline (recombination) | ○ planned |

A unifying insight used throughout: **one neighbor-density-over-time
computation drives three behaviors.** Given a memory, find its
near-neighbors above a similarity threshold, then classify by the temporal
spread of those neighbors:

```
sparse neighbors            -> surprise   (novel; keep)
dense + temporally tight    -> burst      (one-off; collapse to gist)
dense + temporally spread    -> habit      (stable; promote the trait)
```

Built once (`forgetfuldb-core::salience`), read three ways. The critical
constraint everything routes around: **the model has no clock.** Every
notion of "now", "3 years ago", "annual", "during the Clarity era" is
computed by the engine from stored timestamps and injected — which is why
it can be made exact.

---

## What's implemented today

### Decay (`forgetfuldb-core::decay`)
Per-type exponential half-lives (raw ~2d, episodic ~9d, semantic ~70d,
preference ~35d). **Pins** are a true hard short-circuit (no decay, never
evicted, in all paths). Salience modulates the *effective* decay rate
(`decay_score_resisted`): a fully-salient memory forgets at `(1 - resist)`
of the base rate.

### Salience (`forgetfuldb-core::salience`, `salience` column)
U-shaped: `salience = max(surprise, habit)` gated by content relevance so
novel *noise* (typos, garbage) can't enshrine itself.
- **surprise** = `1 − max cosine to anything stored` (novelty)
- **habit** = `neighbor_density × temporal_spread`
- **provisional** value at ingest (novelty only, the free write-time
  signal); **authoritative** value recomputed each consolidation by the
  shared discriminator.
- A memory at/above `salience_keep_threshold` is **kept through pruning**,
  regardless of decay — the automatic counterpart to a manual pin. This is
  how a formative memory survives the archiving that buries the routine
  around it.

### Real embeddings (`forgetfuldb-embed`)
Pluggable provider; `hashed_bow` (offline, lexical) default, or a local
Ollama model (embeddinggemma, nomic-embed-text, …) for true semantic
distance — the prerequisite for surprise/salience and semantic edges to be
meaningful. Switchable live (`/embed`), with a re-embed migration.

### Typed association edges (`memory_edges`)
Three notions of "related", each a different traversal meaning:
- **`co_occurred`** — recalled into the same chat turn (behavioral /
  Hebbian, weighted, recency-decayed). Updated *live* per turn off the
  conversation path; rebuilt authoritatively at consolidation.
- **`semantic_similar`** — close in embedding space (cosine kNN). What is
  *close in meaning*, even if never recalled together.
- **`sequence`** — discussed one after another within a session
  (directional, causal). The reasoning-path signal recovered from
  `chat_turns` order, which nothing else reads.

Plus the consolidation-built `derived_from` (summary provenance) and
`duplicates` (dedup) links.

### Spreading activation (one hop, config-gated)
Retrieving a memory boosts its co-occurrence neighbors, so companions
surface even at low query similarity. Intentionally one-hop today
(non-cascading); multi-hop traversal is planned.

### Temporal query bypass
Dated / epoch queries (`bypass_decay`) skip decay and read raw importance:
decay governs *ambient* recall, but "what happened in this interval" is a
different operation, an index lookup.

### Consolidation — the "sleep cycle"
Dedup-merge → **burst-collapse (gist, keep the anomaly)** → recurrence
refresh → **salience revision** → cluster summaries → episodic→semantic
promotion → **habit→foundation promotion** → contradiction-staling →
archive/prune → **rebuild all three edge graphs**. Logged per run. Triggered
manually, or nightly via the opt-in launchd timer
(`forgetfuldb schedule install`).

### Foundation tier (`MemoryType::Foundation`)
Decay-exempt identity traits *concluded* from accumulated habit evidence: a
semantic/preference memory whose near-neighbors form a `Habit` (the
discriminator's class) spread over a long stretch of history graduates to
Foundation, which never decays and is never pruned — a pin reached
automatically rather than by hand. A habit *cluster* collapses to a single
trait (strongest member wins; near-twins are skipped).

### Gist-collapse keeping the anomaly (`collapse_bursts`)
The temporal inverse of Foundation promotion. A `Burst` — a dense cluster of
similar events packed into a tight window — is summarized into one gist and
the routine deleted, but the **outlier** (the member least like the rest, the
part of the flood that didn't fit) is kept and sharpened. Where dedup keeps
the *center*, a burst keeps the *edge*.

### Observability UI + live updates
A read-only React SPA (embedded in the binary) over the whole engine:
the memory graph (glowing nodes, time-scrubber, all edge types, salience
in the node panel), the retrieval inspector (per-component score bars,
salience flags, the active embedding model), consolidation diffs, metrics.
**SSE** (`/events`) pushes a change the instant the store is modified — by
this server or by a separate `iforgot` process (detected via SQLite
`data_version`) — so the graph updates live as you chat.

---

## What's planned (and where it plugs in)

Ordered by the critical path. Each is specced enough to build cleanly.

> ✅ **Done:** the **Foundation tier** and **gist-collapse keeping the
> anomaly** (items 1–2 below) now ship — see "What's implemented today". The
> remaining critical path starts at **epochs**.

1. ~~**Foundation tier**~~ — *shipped.* Decay-exempt trait memories
   *concluded* by consolidation from accumulated habit evidence ("user
   initiated tic-tac-toe 4× over 3 months → trait"). The identity layer.
2. ~~**Gist collapse keeping the anomaly**~~ — *shipped.* When the
   discriminator finds a *burst*, collapse the routine into one gist but
   **keep the outlier** (inverts the keep-the-central-member dedup).
3. **Epochs** — organic drift-segmented eras (windowed embedding-centroid
   drift + hysteresis, model-free) running orthogonally to a calendar grid.
   Prior art: **ES-Mem** (arXiv 2601.07582). Consolidate *within* an epoch,
   preserve *across*.
4. **Multi-hop edge traversal + subgraph injection** — retrieval becomes a
   path-walk of the typed graph with per-hop activation decay, and injects
   a *connected subgraph with paths* (so the model can reason over the
   chain), not a flat list. The difference between *retrieval* and
   *thinking*. Must preserve conversational dominance (relationships are
   hints, not overrides).
5. **Inferred contradiction detection** — read text, conclude
   "A supersedes B", write a contrastive edge — the direct attack on
   *staleness*, the hardest open problem in agent memory.
6. **Goal-conditioned retrieval** — bias scoring by a current intent
   vector; also supplies a real goal-relevance term for salience.
7. **Dreaming** — offline, sample *unconnected* memory pairs and test
   whether a connection (or a new *derived* memory — "both projects failed
   for the same reason") should exist. The only mechanism that *creates*.
   Strictly offline, low-confidence, pruned aggressively if never
   reinforced — the confabulation guards are the feature.
8. **Scale**: ANN index (HNSW) when brute force stops being instant; an
   **MCP server** so any agent can use iforgot as a memory tool.

---

## Evaluation philosophy

**Do not optimize against LoCoMo / LongMemEval.** ~94% of LoCoMo and ~85%
of LongMemEval questions need ≤2 prior sessions and assume stored info
stays permanently valid — the *opposite* of this project's thesis. They
structurally reward hoarding and penalize forgetting. Three layers
instead:

1. **Synthetic behavior tests (in CI, against real SQLite).** Deterministic
   streams with known answers, each isolating one mechanism — selective
   forgetting (anomaly survives, routine collapses), staleness, surprise
   salience, habit-vs-burst, epoch-boundary-with-hysteresis. *Shipped:*
   `forgetfuldb-consolidate` behavior tests + the discriminator unit tests.
2. **Retention efficiency — the real top-line metric.** Accuracy *per token
   of memory injected*. Every accuracy number paired with its token cost.
   *Shipped (cost side):* `chat_metrics_summary` computes the per-turn
   injected-token cost — injected tokens/turn, memory share of prompt,
   tokens/injected-memory — surfaced in `/metrics`, the CLI, and the UI
   Metrics view. This is the denominator that flatters forgetting: near-equal
   accuracy at a fraction of the tokens. The accuracy numerator comes from
   Layer 1 / Layer 3.
3. **Targeted external benchmarks** — MemoryAgentBench (report *only* the
   selective-forgetting axis), "Recall to Forgetting" — reported as
   efficiency-vs-recall *pairs*, never a raw recall chase.

---

## Design principle: every mechanism is isolable

The six axes are deliberately separate, separately testable, and
separately *inspectable* in the UI — so when a memory behaves
unexpectedly, you can see exactly which mechanism did it: the decay curve
(scrubber), the salience score (node panel + inspector flag), the edge
type that pulled it in (graph + spreading-activation breakdown), the
consolidation run that changed it (diff view). Debuggability is a
first-class constraint, not an afterthought.
