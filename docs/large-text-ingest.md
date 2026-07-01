# Large-text ingest — atomize → segment → salience → decay

> Design doc. How ForgetfulDB should absorb a large paste (an HTML dump, a
> pasted doc, a log) without either storing it as one useless blob or flooding
> the store with fragments. Builds directly on `forgetfuldb-segment`
> (see [surprise-segmentation.md](surprise-segmentation.md)) and the existing
> `core::salience`. Status: **design + evaluation** (the pipeline is specced
> here; the salience hypothesis is exercised by
> `crates/forgetfuldb-segment/tests/large_text_salience_eval.rs`).

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

## 7. Evaluation — the salience hypothesis

`crates/forgetfuldb-segment/tests/large_text_salience_eval.rs` pastes a made-up
cat-and-wolf story (content no pretrained model has memorized — so it isolates
the *memory system*, not the LLM's priors), atomizes and segments it, and tests:

- **Case 1 — empty store.** Every chunk is maximally novel (`surprise ≈ 1.0`):
  with nothing to compare against, salience *cannot discriminate* — it's
  **retrieval relevance** that answers a specific question, not salience. (A
  blank-slate agent correctly keeps the first thing it learns.)
- **Case 2 — populated store.** With a "cat in a meadow" memory pre-seeded, the
  meadow chunk scores **low** novelty (redundant) while the river/wolf/bargain
  chunks score **high** — salience discriminates *only relative to what is
  already known*. This is the hypothesis working: keep the new, fade the known.
- **Questions in scenarios.** Retrieval (cosine to chunks) returns the right
  scene for "who was the wolf at the river?" etc., and a question about content
  **not** in the story (a dragon, a castle) returns a top score below the
  confidence floor — the system declines rather than inventing a match.

**Caveat (honest):** the default embedder is `hashed_bow` (lexical bag-of-words,
not semantic), so retrieval works on word overlap; true semantic Q&A needs a
real local model (`embeddinggemma` via Ollama). The eval demonstrates the
*mechanism* (segment → salience → relevance routing), not semantic depth.

## 8. Open questions

- **Atomizer strategy per hint.** Prose → sentence/paragraph; code → block/
  function; log → line/timestamp. HTML/code structural splitting may want a
  dependency (or a cheap tag-depth heuristic first).
- **Length threshold.** `classify_input_mode` uses `> 3000` chars; §6 argues it
  can be loose. Tune once real inputs exist.
- **Decay-rate mapping.** The exact `salience → lambda` curve (and whether it's
  a new `Staged` type vs a per-row lambda override) — pick against the archive/
  prune cadence.
