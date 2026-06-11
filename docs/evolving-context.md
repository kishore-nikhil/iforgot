# Evolving context: the working-memory plan

Status: **design** — the short-term fixes below it are shipped; the
working-memory tier is the next major piece of memory work.

## The problem this solves

A chat turn has two sources of grounding today:

1. **Live history** — the last `history_turns` exchanges, verbatim.
2. **Long-term retrieval** — `top_k` memories matched against the user's
   message.

Both are real, but there's a gap between them. History is verbatim and
short; long-term memory is decayed and topic-agnostic. A conversation
that runs past the history window silently loses its own beginning, and
nothing in the system *understands what the current conversation is
about* — retrieval can only guess from the latest message.

The failure mode that motivated this design: the user spends a turn
describing a story-writing app, then says "something catchy people will
remember". The follow-up names no topic, the old session's memories do,
and the model answers about the wrong project entirely.

## Shipped mitigations (short-term fixes)

These landed first and stand on their own — see README "Context vs
memory" for the user-facing description:

| Fix | Where |
| --- | --- |
| Contextual retrieval query (`query_context_turns`): recent raw user messages are folded into the retrieval query, never the prompt | `forgetfuldb-agent::prepare_turn`, proxy |
| Relevance gate (`min_retrieval_score`): inject nothing rather than something misleading | `forgetfuldb-retrieve` |
| Conversational damping (`conversational_damping`): verbatim chat turns are down-weighted so old conversations can't hijack the live one | `forgetfuldb-retrieve` |
| Session exclusion: the live session's own turns (tagged `session:<id>`) never come back as "memories" | `forgetfuldb-retrieve`, `chat_ingest_request` |
| Precedence framing: the memory block and system prompt both say the live conversation wins on conflict | `memory_context_block`, default system prompt |

They make retrieval conversation-aware. They do **not** give the system
any *understanding* of the conversation — that's the working-memory tier.

## The design: a three-tier context

Human-memory framing, consistent with the rest of ForgetfulDB:

```text
┌────────────────────────────────────────────────────────────┐
│ 1. Live history        verbatim, last N exchanges          │  seconds–minutes
│ 2. Working memory      rolling summary of THIS session     │  the whole session
│ 3. Long-term memory    decayed, consolidated, retrieved    │  days–months
└────────────────────────────────────────────────────────────┘
```

Tier 2 is new: a **per-session rolling summary** — a few sentences that
say what the conversation is about, what was decided, and what's still
open. It is *the* evolving context: updated as the session unfolds,
consolidated into long-term memory when the session ends.

### How working memory behaves

- **Update cadence.** Re-summarized every turn (or every K turns /
  token-budget triggered) by the `Summarizer` trait — the same seam
  consolidation uses. Until the LLM summarizer exists (roadmap item 2),
  a heuristic extractive version can carry it: latest topic guess +
  entities + the first user message of the session.
- **Injection point.** Attached to the *current user message* alongside
  the memory block — never the system prompt — so the prefix KV-cache
  property is preserved (everything before the newest message stays
  byte-identical across turns).
- **Retrieval input.** The working-memory summary replaces/augments
  `query_context_turns` as the context half of the retrieval query. A
  summary is a better query than raw recent turns: it survives topic
  drift and stays bounded in tokens.
- **Storage.** A `working_memory` row per session (session_id, summary,
  updated_at, turn_count). It is *not* a `MemoryItem` while live — it
  must not decay, dedup, or be retrieved into other sessions.

### Session-end consolidation

When a session closes (explicit `/quit`, or staleness timeout during the
nightly consolidate run):

1. The final working-memory summary is ingested as an **episodic**
   memory ("what happened in that conversation"), tagged with the
   session.
2. Stable facts and preferences the summarizer flagged are ingested as
   **semantic** / **preference** memories — this is where "the user has
   an app called Plot Perfect" graduates from a chat turn to a fact.
3. The verbatim per-turn chat memories from that session become prime
   candidates for archive/prune — the summary now represents them, so
   the lossy-compression story (raw → episodic → semantic) plays out at
   session granularity. This is roadmap item 5 ("session-aware
   consolidation") falling out of the design for free.

### Why this beats just widening the knobs

- Raising `history_turns` grows the prompt linearly and still falls off
  a cliff at the window edge. The summary is O(1) tokens for the whole
  session.
- Raising `query_context_turns` feeds raw text into retrieval; after a
  topic shift, stale raw turns pollute the query. A maintained summary
  tracks the *current* topic.
- Long conversations stop being a memory hole: today a session's early
  decisions exist only as fast-decaying raw turns; with consolidation
  they leave as distilled facts.

## Implementation phases

1. **Heuristic working memory** (no model needed): extractive summary =
   session topic guess + accumulated entities + first user message.
   New `working_memory` table; injection next to the memory block;
   retrieval query uses it. Ship behind `[chat] working_memory = true`.
2. **LLM summarizer** (roadmap item 2): `Summarizer` impl backed by
   Ollama; abstractive rolling summary with a fixed token budget;
   fact/preference extraction at session end. The background
   `MemoryWriter` thread does the summarizing off the conversation path.
3. **Session-end consolidation**: wire phase 2's outputs into
   `forgetfuldb-consolidate`; mark summarized sessions' raw turns for
   early archive.
4. **Proxy support**: the proxy is stateless per request and uses the
   constant session id "proxy"; derive a stable per-conversation key
   (hash of the first user message) so proxied UIs get working memory
   and session exclusion too.

## Open questions

- Summary drift: a bad early summary can steer later updates. Mitigate
  by always re-summarizing from (previous summary + last K verbatim
  turns), never summary-of-summary alone.
- When the user corrects a memory mid-chat ("no, the standup is at 10
  now"), should working memory write a `contradicts` link immediately
  rather than waiting for session end? Probably yes — it's the cheapest
  moment to catch it.
- Token budget split: with a 4k-context local model, a reasonable
  starting allocation is ~50% history, ~15% working memory, ~20%
  memories, ~15% reply headroom — tune against `/metrics` data.
