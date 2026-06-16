# iforgot — session handoff

> Drop-in context for a new session. Read this + [memory-architecture.md](memory-architecture.md)
> (the vision/roadmap) and you're caught up. This file is the *operational*
> state; the architecture doc is the *direction*.

## What this is

**iforgot / ForgetfulDB** — a **forgetting engine** (memory layer for
agents) backed by SQLite, written in Rust. *Not* a database: it decides
what to keep, decays the unused, keeps the formative (salience),
consolidates routine into gist, and connects memories into a typed graph.
Repo: `~/projects/iforgot`. Local-first, `local_only = true` (binds
127.0.0.1).

Two binaries: **`forgetfuldb`** (CLI: ingest/retrieve/consolidate/server/
demo/reembed/…) and **`iforgot`** (terminal chat with automatic memory).
Both installed to `~/.cargo/bin`.

## Crate map (`crates/`)

| Crate | Responsibility | Notable files |
| --- | --- | --- |
| `forgetfuldb-core` | scoring, decay, **salience**, **epochs**, config, types | `salience.rs` (neighbor discriminator), `epochs.rs` (drift segmentation), `decay.rs` (`decay_score_resisted`), `scoring.rs`, `config.rs` |
| `forgetfuldb-store` | SQLite persistence, migrations, pipeline | `lib.rs`, `pipeline.rs` (ingest + edge rebuilds), `migrations/0001..0008` |
| `forgetfuldb-embed` | embedding providers | `hashed_bow` (default, lexical) + `ollama.rs` (real) |
| `forgetfuldb-retrieve` | hybrid retrieval, spreading activation, `bypass_decay`, `epoch_ordinal` | `lib.rs` |
| `forgetfuldb-consolidate` | the "sleep cycle" | `lib.rs` (merge, gist-collapse, `revise_salience`, summaries, promote, foundation, stale, archive/prune, edge rebuilds, `segment_epochs`) |
| `forgetfuldb-prob` | bloom / CMS / HLL / reservoir (from scratch) | |
| `forgetfuldb-tools` | shell + read-only `explore` tools | |
| `forgetfuldb-agent` | chat loop, backend, **writer** (live edge bump), research | `lib.rs`, `writer.rs`, `research.rs` |
| `forgetfuldb-server` | axum API + **SSE** + embedded UI | `lib.rs` (build.rs embeds `ui/dist`) |
| `forgetfuldb-cli` | the `forgetfuldb` binary + `demo` seeder | `main.rs`, `demo.rs` |
| `iforgot-chat` | the `iforgot` binary | `main.rs`, `spinner.rs` |
| `ui/` | React+Vite observability SPA | `src/views/{GraphView,RetrievalView,ConsolidationView,MetricsView}.tsx`, `api.ts`, `usePoll.ts` |

## What's shipped (high level)

- **Retrieval quality**: contextual query (`query_context_turns`),
  relevance gate (`min_retrieval_score`), `conversational_damping`,
  `session:<id>` self-exclusion.
- **Salience** (U-shaped surprise∨habit) via the shared
  neighbor-density-over-time discriminator; resists decay; `salience_keep_threshold`
  keeps formative memories through pruning (auto-pin). Pin is a separate
  hard short-circuit.
- **Typed association graph** in `memory_edges`: `co_occurred` (recalled
  together, live per-turn + rebuilt), `semantic_similar` (cosine kNN),
  `sequence` (session order, directional). Plus `memory_links`
  (`derived_from`, `duplicates`, …). **Spreading activation** (one-hop,
  config-gated `spreading_activation`, default off).
- **Real embeddings** via Ollama (`/embed` in chat, `forgetfuldb reembed`),
  with dimension-mismatch warning + `store_meta` identity.
- **Consolidation** logs runs (`consolidation_runs` table) shown in the UI.
- **Observability UI** at `/ui` (embedded in binary): graph (glowing nodes,
  time-scrubber, all edge types, "living" animation, salience in node
  panel), retrieval inspector (per-component bars + near-misses + salience
  flag + active embedding model), consolidation diff, metrics. **SSE**
  (`/events`) → live updates (~1s) including from a separate `iforgot`
  process.
- **`/research <folder>`** read-only exploration agent; first-token spinner.
- **Eval Layer 1** behavior tests (selective forgetting, surprise salience,
  decay resistance, **habit→foundation**, **burst→gist**, **two-topic→two-
  eras**, **preserve-across-epochs**) in `forgetfuldb-consolidate`, plus the
  pure `epochs.rs` segmentation unit tests in `forgetfuldb-core`.
- **Foundation tier** (`MemoryType::Foundation`): decay-exempt identity
  traits *concluded* by consolidation from `NeighborClass::Habit` evidence
  (a habit cluster collapses to a single trait). Mirrors the pin exemption
  in decay/prune/merge. Migration `0007` widens the type CHECK.
- **Gist-collapse keeping the anomaly** (`collapse_bursts`): a dense,
  temporally-tight burst of similar events is summarized into one gist and
  deleted, but the **outlier** (least-central member) is kept and sharpened —
  the inverse of dedup's keep-the-center.
- **Retention-efficiency metric** (Layer 2 cost denominator): injected
  tokens/turn, memory share of prompt, tokens/injected-memory — computed in
  `chat_metrics_summary`, surfaced in `/metrics`, the CLI `metrics` command,
  and the UI Metrics view. Accuracy (numerator) still comes from the behavior
  tests/benchmarks; this is the cost it pairs against.
- **Nightly consolidation timer** (`forgetfuldb schedule install|status|
  uninstall`): writes a per-user launchd agent that runs `consolidate`
  nightly. Opt-in (the user runs it); macOS-only (Linux prints the cron line).
- **Epochs** (`core::epochs`, migration `0008`, `segment_epochs`): the
  timeline is partitioned into drift-segmented eras (windowed embedding-
  centroid drift + hysteresis, model-free). Rebuilt each consolidation into
  the `epochs` table (no `epoch_id` on `memory_items` — membership is a
  time-range lookup). Drives "consolidate within, preserve across" (merge &
  gist-collapse skip cross-epoch pairs) and `retrieve(epoch_ordinal=N)`
  (resolves to the era's window + `bypass_decay`). Surfaced at `/epochs`, in
  `stats`, and as the UI Metrics "epochs" strip.

State: ~142 tests pass, clippy clean, tsc clean.

## Commands

```bash
# build / verify
cargo test
cargo clippy --all-targets
cd ui && npm install && npm run build      # rebuild the SPA

# install (refreshes ~/.cargo/bin AND re-embeds the latest ui/dist)
cargo install --path crates/forgetfuldb-cli
cargo install --path crates/iforgot-chat

# run
forgetfuldb demo --dir demo                # seed a throwaway store (runs a consolidation pass)
forgetfuldb server                          # API + embedded /ui (global store)
forgetfuldb server --config demo/forgetfuldb.toml --ui ui/dist   # dev: disk UI, demo store
forgetfuldb consolidate                     # the sleep cycle (MANUAL — see gotchas)
iforgot                                      # chat; /embed /research /consolidate /model /help
```

## Gotchas (read before debugging)

1. **Config resolution**: `--config` > `./forgetfuldb.toml` > global
   `~/.forgetfuldb/`. The user's real store is the **global** one. A stray
   `~/forgetfuldb.toml` in `$HOME` triggers a warning (it once split the store).
2. **Consolidation must be triggered** — salience, semantic/sequence edges,
   summaries, promotion (incl. foundation), burst-collapse, and archiving
   only happen when `forgetfuldb consolidate` (or `/consolidate`, or
   `POST /consolidate`) runs. It is **not** automatic *unless* the user has
   run `forgetfuldb schedule install` (the nightly launchd timer). If
   "edges/salience aren't updating," either it's never been consolidated or
   the timer isn't installed (`forgetfuldb schedule status`).
3. **Embedded UI can go stale** on incremental `cargo build` (the
   `include_dir!` of `ui/dist` doesn't always recompile). `cargo install`
   does a fresh build so it's current. For UI dev, use `--ui ui/dist` (disk,
   always latest) and skip the re-embed.
4. **Switching embedding model requires re-embedding** (different dims are
   incomparable → silent zero-similarity). `/embed` does it automatically;
   **restart the server** after switching (it fixes its provider at startup).
5. **Tests use `hashed_bow`** (lexical, has a cosine-collision floor) — real
   semantic behavior needs `embeddinggemma`. Behavior tests assert *relative*
   not absolute values for this reason. The tokenizer drops 1-char tokens
   (bit me once: numeric suffixes collapsed and merged).
6. **`spreading_activation` reads `list_edges()` on the sync retrieval path** —
   cheap now, watch it if the edge table gets huge (ANN/indexing is the fix).
7. Date in this project's history is ~2026-06; consolidate relative dates.

## What's next (deferred, specced in memory-architecture.md)

Done since the last handoff: **Foundation tier**, **gist-collapse**, the
**retention-efficiency metric** (cost denominator), the **launchd nightly
timer**, and **epochs** (drift-segmented eras, ES-Mem arXiv 2601.07582).
Remaining, in dependency order: **multi-hop traversal + subgraph injection**
(retrieval that *thinks* along the edges — generalize the one-hop spreading
activation, inject a connected subgraph; the `epoch-multihop` branch is named
for it) → **contradiction inference** (the staleness attack) →
**goal-conditioned retrieval** → **dreaming** (offline recombination, the only
mechanism that *creates*) → **ANN index**, **MCP server**.

**Eval philosophy**: do NOT optimize LoCoMo/LongMemEval (they reward
hoarding); use **retention efficiency** (accuracy per injected token). The
cost denominator is now measured per turn; the accuracy numerator still comes
from the Layer-1 behavior tests / targeted benchmarks.
