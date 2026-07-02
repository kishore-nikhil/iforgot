# Large-text ingest — atomize → segment → salience → decay

> How ForgetfulDB absorbs a large paste (an HTML dump, a pasted doc, a log)
> without either storing it as one useless blob or flooding the store with
> fragments. Builds directly on `forgetfuldb-segment`
> (see [surprise-segmentation.md](surprise-segmentation.md)) and the existing
> `core::salience`.
>
> Status: **implemented, including the two-track (gist + verbatim) storage and
> retrieval design in §8.** Atomizer, semantic chunking + per-chunk salience,
> the durable gist, gist-first descent at retrieval, and real chunk decay/prune
> are all landed and tested (`cargo test --workspace`: 0 failures; touched
> crates clippy-clean). A multi-agent adversarial review (§8.4) found and this
> pass fixed 9 real defects in the gist/expansion machinery — read that section
> before trusting the mechanism's edge cases. The chunk→durable **promotion**
> pass (testing-effect, §8.2 point 4) and the classifier collapse (§6) remain
> design only. Evals: `forgetfuldb-segment/tests/large_text_salience_eval.rs`
> (isolated), `cat_and_wolf_end_to_end_eval` (real ingest path), and
> `ollama_eval.rs` (real `embeddinggemma`, `--ignored`).

## 1. The two questions a large paste raises

1. **Should this even be remembered?** Often *no*. "Edit this HTML for me" makes
   the HTML a **working artifact for this turn**, not a durable memory. A
   forgetting engine must not hoard it.
2. **If yes, in what units?** Not one blob (diluted embedding, all-or-nothing
   retrieval), not per-line fragments (incoherent). The right unit is a
   **coherent segment** — the output of surprise segmentation applied *inside*
   the document.

## 2. Pipeline

```
ingest(text)
  │  len(text) < THRESHOLD  ─────────────► normal single-memory path (regular chat)
  │
  └─ len(text) ≥ THRESHOLD  (large paste)
        1. atomize(text, hint)      split into atoms (sentences / code blocks / log lines)
        2. embed atoms              one vector each
        3. segment(atom_embs)       coherent chunks   ← forgetfuldb-segment, reused
        4. salience-score chunks    novelty × relevance  ← core::salience, reused
        5. stage with decay ∝ (1 − salience)   ← fast fade, not a hard drop (§4)
        6. promote on reuse         a chunk that gets retrieved & accepted is reinforced
```

Steps 3 and 4 are already built and tested. The new surface is small (§5).

## 3. The salience funnel — "keep only the relevant stuff"

This is `core::salience` applied per chunk. "Worth remembering" is **two
opposite similarity signals**, and getting the sign right is the whole trick:

- **relevance** = similarity to the current goal/query → want it **HIGH**.
- **surprise/novelty** = `1 − max cosine to anything already stored` → want it
  **LOW similarity** to memory (i.e. novel). High similarity means *redundant* —
  already known, don't restore it.

`salience = max(surprise, habit) × relevance` ([`salience.rs`](../crates/forgetfuldb-core/src/salience.rs)).
A boilerplate HTML `<head>` scores low novelty (generic) and low goal-relevance
→ fades. The one section with the real pricing logic scores high on both →
kept. **Score chunks, not atoms** — a sentence out of context is noise; a
coherent segment is the unit that carries meaning.

**Two surprises, don't conflate them:**

| | reference frame | decides |
| --- | --- | --- |
| segmentation surprise (`forgetfuldb-segment`) | the local *sequence* | **where** chunks break |
| novelty surprise (`core::salience`) | the *whole store* | **what** is worth keeping |

## 4. Fast decay, not a hard drop (graceful degradation)

Rather than thresholding salience into keep/drop, **stage every chunk but set
its decay rate inversely to salience**. A high-salience chunk decays slowly
(survives); a low-salience chunk decays fast (fades in hours/days unless
reused). This is strictly better than a cutoff:

- **No arbitrary threshold.** The forgetting curve does continuous selection
  over time instead of a brittle one-shot decision.
- **The "edit this HTML" case falls out for free.** The whole paste is briefly
  available for the current task and immediate follow-ups, then the irrelevant
  bulk fades — durable memory is never polluted, but nothing is lost mid-task.
- **Reuse rescues.** A staged chunk that gets retrieved and accepted is
  reinforced (importance/recurrence up, decay reset) and promotes to durable —
  the existing §V2 reinforcement path. What proves useful survives; what
  doesn't, evaporates.

Concretely: a staged chunk is inserted with `decay_score` seeded low
(∝ salience) and a **fast decay lambda**, so the existing archive/prune cycle
retires the unreused ones with no new machinery — just a decay-rate that
depends on salience.

## 5. Where the changes land

| Location | Change | New / reuse |
| --- | --- | --- |
| `core::ingest` | `atomize(text, hint) -> Vec<String>` (replaces the naive fixed-window `chunk_source_text`); collapse `classify_input_mode`'s 6-way enum to `needs_atomizing(text) -> bool` + `structure_hint(text) -> {Prose, Code, Log}` (the old markers become the atomizer's splitter hint, not a stored label) | **new** (small) |
| `forgetfuldb-segment` | none — `segment_with_embeddings` is reused as the intra-document chunker (it's time-agnostic, so it drops straight in) | reuse |
| `core::salience` | none — `analyze_neighbors` + `salience` scored per chunk | reuse |
| `forgetfuldb-store::pipeline` | new `stage_large_text(...)`: atomize → embed → segment → per-chunk salience (extends the existing [`provisional_salience`](../crates/forgetfuldb-store/src/pipeline.rs:606)) → insert staged chunks with decay ∝ (1 − salience); write `source_document` + `source_chunks` (schema 0009 exists) | **new** (orchestration) |
| `core::decay` / `types` | a staged chunk needs a fast per-memory decay rate (a `Staged` lifecycle marker or a decay-lambda override) | **new** (small) |
| consolidation | the §13 chunk→durable promotion on support (confidence ∧ relevance ∧ reuse) — already on the roadmap | **new** (later) |

Dependencies stay one-way: `store → {core, segment, embed}`. No cycle.

## 6. The classifier, honestly

The current [`classify_input_mode`](../crates/forgetfuldb-core/src/ingest.rs:59)
is brittle keyword+length heuristics ("I hit an `exception`" → `LogDump`). Its
*only* legitimate job is choosing an atomization strategy — never a stored
truth about the content. Two properties make a **loose** length gate safe:

- The **segmenter won't fragment coherent text** (proved: `single_topic → 1
  chunk`), so an over-triggered gate on a long-but-single-topic blob just yields
  one chunk.
- **Salience filters the junk** that atomizing surfaces.

So the gate can be roughly "is it big?"; the real work is downstream, in two
mechanisms that already exist and are tested.

## 7. Evaluation — results

A made-up cat-and-wolf story (content no pretrained model has memorized — so it
isolates the *memory system*, not the LLM's priors) is run through the **real
ingest pipeline** in `cat_and_wolf_end_to_end_eval`
(`store/src/pipeline.rs`). Observed:

- **Case 1 — empty store.** Every chunk salience = **1.00**. With nothing to
  compare against, salience *cannot discriminate* — it's **retrieval relevance**
  that answers a specific question, not salience. (A blank-slate agent correctly
  keeps the first thing it learns.)
- **Case 2 — meadow pre-seeded.** The meadow chunk drops to **0.57** (redundant)
  while the novel winter chunk stays at **0.92** — salience discriminates *only
  relative to what is already known*. The hypothesis working: keep the new, fade
  the known.
- **Fast decay, not drop.** Staged chunks decay at the fast raw-event rate,
  resisted by salience (`decay_score_resisted`). Day 1: routine 0.81 / novel
  0.88 — **both present** for the current task. Day 7: 0.23 / 0.42. Day 30:
  0.002 / 0.024 — the routine chunk has faded out, the novel one persists. No
  arbitrary drop; the forgetting curve selects.
- **Absent content.** A question about a dragon/castle scores **0.02** — below
  the confidence floor, so the system declines rather than inventing a match.

**Honest limitation (visible in the eval).** Under the default `hashed_bow`
embedder (lexical, not semantic), the four scenes share vocabulary (Mira,
meadow, wolf), so segmentation is **coarse** — the river and bargain scenes
merged into the first chunk, and both scene questions routed to it. The
*mechanism* (semantic chunking → per-chunk salience → decay) is proven
end-to-end; clean per-scene separation and retrieval need a real local model.

### 7.1 Against a real embedder (`embeddinggemma` via Ollama)

`forgetfuldb-segment/tests/ollama_eval.rs` (ignored by default; run with
`--ignored`) reruns the story through the local **Gemma** embedder (768-dim).
Findings, blunt:

- **Semantic salience is cleaner and grounded.** With "a cat in a meadow"
  pre-seeded, the meadow chunk is recognized as *semantically* known
  (surprise 0.32) vs. the novel chunks (~0.56) — not the lexical coincidence
  `hashed_bow` relies on.
- **Sentence retrieval works, including synonyms.** "What **deal** did the wolf
  offer?" retrieves *"It was an unusual **pact**…"* — a deal↔pact match no
  lexical embedder makes. Every story question beats an off-topic (dragon/castle)
  one, so the system still declines on absent content.
- **But narrative segmentation stays fuzzy — and that's the real lesson.** A
  flowing *story* is semantically **continuous** ("the cat wandered to the
  river" isn't surprising after "the cat napped in the meadow"), so the
  segmenter finds real semantic-surprise points that don't match human-authored
  "scenes." Surprise segmentation shines on **distinct-topic** documents
  (billing → shipping → refunds), not continuous narratives. Choose the eval
  content to the mechanism.
- **Operational note:** dense embeddings have a high similarity floor
  (off-topic ≈ 0.44 vs on-topic ≈ 0.51), so the retrieval-gate threshold
  (`min_retrieval_score`) needs recalibrating for a semantic backend — a margin,
  not an absolute, decides relevance.

## 8. Two-track storage & retrieval — the fuzzy-trace design

How the staged chunks become a *memory of the document* that erodes like human
memory does. Grounded in a code audit (multi-agent, 2026-07) plus three design
lenses (memory research, prior-art systems, retrieval architecture).

### 8.1 The audit, blunt (what was true before this section)

- **`source_chunks` was a dead-end table**: embedded and salience-scored at
  ingest, then never read by any query path — no retrieval scan, no server
  route, nothing. Write-only.
- **"Fast decay, not drop" was a property of the eval, not the system.** The
  day-1/7/30 numbers in §7 were computed inline in the test; the table had no
  timestamp column and no production code decayed, pruned, or retrieved a
  chunk.
- The only retrievable trace of a long doc was **one RawEvent blob at
  importance 0.10** with a diluted whole-document embedding, gone in days.
- Multi-hop (`retrieve::traverse`) exists but is **off by default**
  (`spreading_activation=false`, `max_hops=1`) and can never reach chunks:
  every edge writer draws ids from `memory_items` only, and traversal resolves
  ids via `get_memory` (unknown ids silently skipped).
- Trap: edge rebuilds **clear-and-rebuild** each edge type every consolidation
  from `chat_turns`/`memory_items` — any staging-time doc-structure edges would
  be wiped at the next sleep cycle unless the rebuild re-derives them.

### 8.2 The design (fuzzy-trace theory, engineered)

Humans encode a **verbatim trace and a gist trace in parallel; the verbatim
fades first** (Brainerd/Reyna), so recall shifts from quotation to
reconstruction. Two tracks:

| Track | What | Decay | Fate |
| --- | --- | --- | --- |
| **Gist** | one compact durable memory per document — the chunk-opening (boundary) sentences joined; `derived_from`-linked to the raw blob; `source_doc:` tag as the pointer down | semantic-tier (λ=0.01) | survives — the retrieval anchor |
| **Verbatim** | the staged chunks (embedding + salience) | raw-event λ, **resisted by salience** | fade unless used; pruned below the floor |

You remember *"cat befriended wolf to guard the meadow"* forever; the exact
phrasing, for a week. Storage decisions this implies:

1. **Gist at ingest** (slice 1): built from each chunk's opening atom — event
   segmentation theory says boundary content is privileged in recall, and it's
   deterministic and free. Stored as a Semantic memory with its own embedding.
2. **Read path at retrieval** (slice 2): when a final hit carries a
   `source_doc:<id>` tag, cosine-rank that document's chunks against the query
   and substitute the best chunk's text for the diluted blob/gist content — a
   *directed, scoped* lookup (tens of vectors), not spreading activation, so it
   can be on by default. This is the first production read of `source_chunks`
   and the prerequisite for every usage signal downstream.
3. **Real chunk decay** (slice 2b): `created_at` on chunks (migration 0011) +
   a consolidation prune pass — `decay_score_resisted(1, λ_raw, age, salience,
   resist)` below `archive_max_decay` → delete. At the default λ_raw=0.35 /
   resist=0.7: a routine chunk falls below the floor in ~9 days, a maximally
   salient one at ~28 — salience buys the **promotion window**, not
   permanence; the gist is what lasts. The doc row (hash + summary) outlives
   its chunks as the "where to look" pointer.
4. **Testing-effect promotion** (next): the only path from staged chunk →
   durable memory is being retrieved *and used* — relevance is proven by use,
   not predicted at ingest. No batch promotion. On promotion the chunk becomes
   a `memory_item` with `derived_from` (to gist) and `sequence` edges to its
   nearest *surviving* siblings — written so `rebuild_sequence_edges` re-derives
   them from `chunk_index` (see the rebuild-wipe trap above).
5. **Reconstructive fallback** (later): gist hits ∧ no surviving detail →
   re-read still-alive staged chunks with `bypass_decay` (epoch-query
   precedent); after they're gone, answer from gist with an explicit
   *detail-forgotten* marker. Never confabulate. Gate the re-read so the system
   doesn't quietly become a hoarder.

### 8.3 Retrieval contract (where multi-hop fits)

- **Gist-first descent** (slice 2) is the default read path for doc content.
- **Sequence-walk** for narrative adjacency ("what happened next") rides the
  existing `traverse.rs` (sequence factor 1.0) once promoted chunks enter the
  graph. Multi-hop *spreading* stays opt-in; adjacency at `max_hops=1` is the
  sane default.
- **Only promoted chunks enter the graph.** Staged chunks never get edge rows
  (bounded graph, trivial pruning); gist↔chunk provenance lives in
  `memory_links`, which traversal deliberately ignores (unknown edge type →
  factor 0.0).
- **Margin-based gating** (later): replace the absolute `min_score` with a
  pool-relative gate (z-score / floor+δ·range) — dense embedders have a high
  similarity floor (§7.1: 0.44 off-topic vs 0.51 on-topic), so relevance is a
  margin, not an absolute.
- **Token-budgeted assembly** (later): fill context by priority (direct hits →
  gist lines → detail → neighbors), degrading over-budget chunks to summaries —
  retention efficiency enforced at assembly time.

Held for later: entity spine (HippoRAG-style durable entity nodes for cross-doc
hops), replay-to-edges (dying chunks leave edges between durable memories),
goal-conditioned relevance (cosine to a session-goal centroid as the salience
relevance input), contextual chunk embeddings (doc-gist prefix; risk: prefix
homogenizes intra-doc vectors).

### 8.4 Adversarial review — 11 defects found, 9 fixed

A multi-agent review (3 dimensions x adversarial verification against the
actual code, several findings reproduced with real test cases) of the initial
§8.2/8.3 implementation found 11 confirmed defects. This is the honest record —
read it before extending the gist/expansion machinery.

**Fixed, all covered by regression tests:**

1. **Single-atom documents crashed ingest.** A short message misclassified as a
   long source (e.g. the word "exception" trips `LogDump`) atomizes to exactly
   one atom, so the gist text equaled the whole blob and the second
   `insert_memory` hit the `content_hash` `UNIQUE` constraint — `ingest()`
   returned `Err`, with a durable gist already persisted from the failed
   attempt. **Fix:** the gist hash is now salted (`gist:<source_id>:<text>`),
   which can never equal a plain-text hash of anything.
2. **The gist's hash was never added to the bloom filter**, so a later message
   equal to the gist's text skipped the authoritative dedup lookup and hit the
   same `UNIQUE` crash instead of returning `Reinforced`. **Fix:**
   `finalize_document_gist` now calls `bloom.insert`.
3. **Cross-document gist collisions.** Gist dedup was keyed on the
   boundary-sentence text alone, so two different documents sharing their first
   ≤8 chunk-opening sentences (e.g. monthly report templates) collapsed onto
   one gist tagged with only the *first* document's `source_id` — the second
   document's chunks became permanently unreachable. **Fix:** the same
   `source_id` salt (fix #1) makes every document's gist hash unique by
   construction; the dedup-hit branch now also verifies `source ==
   "document_gist"` and appends the new tag rather than silently adopting an
   unrelated memory.
4. **Blob self-depression.** The gist used to be inserted into `memory_items`
   *before* the blob's own `provisional_salience` lookup, so a long document's
   novelty was scored against a summary derived from itself. **Fix:** gist
   creation moved to *after* `store.insert_memory(&item)` for the blob in
   `ingest()` (see the ordering comment there).
5. **No relevance floor on chunk substitution (HIGH — reproduced).** The
   original comparison was "chunk cosine beats anchor cosine," unclamped. With
   `hashed_bow`'s sign-hashed vectors going negative, a query with **zero**
   real overlap to the store still "beat" a negative anchor and substituted an
   arbitrary unrelated chunk into the prompt — verified end-to-end with a
   nonsense query expanding into unrelated verbatim text. **Fix:** cosines are
   clamped to `[0, 1]`, and substitution additionally requires clearing
   `chunk_match_min_sim` (a new, dedicated config knob — deliberately **not**
   `semantic_edge_min_sim`: that default (0.55) turned out to be mistuned for
   `hashed_bow`, whose realistic same-topic cosine tops out near 0.47; see the
   field doc for the calibration data).
6. **One document, two top-k slots.** A document's gist and its verbatim blob
   both carry the same `source_doc:` tag and can both be top-k hits, each
   independently triggering expansion — wasting a slot on the same document
   twice. **Fix:** `expand_document_chunks` now tracks which `source_id`s have
   already claimed a slot (only on a *successful* expansion, so an
   unsuccessful first attempt doesn't block a later anchor from trying).
7. **Chunk text join destroyed structure.** Atoms were joined with a plain
   space regardless of structure hint, flattening a multi-line stack trace
   (Log) or function body (Code) into one unreadable line before it's injected
   into the prompt. **Fix:** the join separator now follows the same
   `structure_hint` used for atomization (`"\n"` for Log/Code, `" "` for
   Prose).

**Acknowledged, not fixed this pass (scope decisions):**

8. **Re-paste after chunk prune silently discards fresh detail.** If a user
   re-pastes the *exact same* document after its chunks have decayed away, the
   whole-document dedup short-circuit in `ingest()` returns `Reinforced`
   without ever reaching `store_source_document` — the chunks are not
   re-staged. This is a real gap, but restaging requires design work (should
   *every* duplicate re-ingest restage, or only when chunks are fully gone? is
   that itself the testing-effect promotion trigger?) that belongs with item 4
   of §8.2, not a bug-fix pass. Tracked as an open question (§9).
9. **Substitution happens after scoring/ranking/truncation**, so the displayed
   `ScoreBreakdown` describes the discarded anchor content, not the injected
   chunk. Low severity, and "fixing" it (rescoring post-substitution) risks
   destabilizing the min_score/truncate contract for a cosmetic inconsistency.
   Documented as a known tradeoff at the `expand_document_chunks` call site.

**Refuted** (mechanically accurate observations that don't produce an actual
failure in this codebase): migration non-atomicity (a systemic, pre-existing
property of every migration, not specific to this diff); `bypass_decay`/epoch
queries interacting with the chunk-decay floor (chunks aren't in the epoch/date
range machinery at all, so the interaction doesn't occur).

## 9. Open questions

- **Atomizer strategy per hint.** Prose → sentence/paragraph; code → block/
  function; log → line/timestamp. HTML/code structural splitting may want a
  dependency (or a cheap tag-depth heuristic first).
- **Length threshold.** `classify_input_mode` uses `> 3000` chars; §6 argues it
  can be loose. Tune once real inputs exist.
- **Gist quality on narratives.** Boundary-sentence gists are deterministic but
  extractive; a local-LLM (Ollama) summarizer behind the existing `Summarizer`
  trait is the upgrade path if they prove weak.
- **Promotion trigger fidelity.** "The agent used it" is a fuzzy signal; start
  with expanded-into-context + the turn completed, tighten later.
- **Re-paste after prune doesn't restage** (§8.4 item 8). Whole-document dedup
  short-circuits before chunks would ever be re-examined, so a user
  deliberately re-sharing a document to reinforce its detail is silently
  ignored once the chunks have faded. Candidate fix: on the `Reinforced` path,
  check whether the matched memory's document has zero live chunks and, if so,
  restage — needs to be designed as part of testing-effect promotion, not
  bolted on separately.
- **`chunk_match_min_sim` is a single global default (0.2)**, calibrated to
  `hashed_bow`. A real semantic embedder's noise floor sits far higher
  (§7.1: ~0.44) — this needs the same per-embedder recalibration the design
  already flags for `min_retrieval_score`, ideally superseded by the deferred
  margin-based gating (§8.3) rather than hand-tuned per backend.
