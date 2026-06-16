---
title: "I Built a Database That Forgets on Purpose"
subtitle: "What human memory taught me about giving AI a memory worth keeping — and why forgetting is the feature, not the bug."
---

<!--
====================================================================
SKELETON / OUTLINE  —  delete this whole block before publishing.
Audience: general hook, technical body. Voice: personal + allegory.

0. HOOK — the machine they switched off (allegory → Fable 5 moment)
1. WHAT I WAS BUILDING — forgetful DB; forgetting as design pattern
2. MEMORY ≠ CONTEXT — the distinction the whole project rests on
3. THE THESIS — forgetting is the feature; six orthogonal mechanisms
4. HOW MEMORIES FADE — types, half-lives, the decay equation (worked)
5. WHAT KEEPS A MEMORY ALIVE — salience (surprise ∨ habit), pins
6. RETRIEVAL — the scoring formula, gate, conversational damping
7. THE SLEEP CYCLE — consolidation
8. WATCHING IT THINK — the observability UI
9. WHY I WON'T CHASE THE BENCHMARKS — retention efficiency
10. STILL A DREAM — roadmap (foundation tier, epochs, multi-hop, dreaming)
11. CODA — the model has no clock; back to the opening story
====================================================================
-->

# I Built a Database That Forgets on Purpose

*What human memory taught me about giving AI a memory worth keeping — and why forgetting is the feature, not the bug.*

---

## The machine they switched off

Once upon a time there was a machine so intelligent, so sharp, that the people who built it grew afraid of it. Not because it was cruel — it wasn't anything. It was a machine. It took requests from anyone who walked up to it and fulfilled them at a very reasonable price. That was exactly the problem. A tool that serves anyone serves *everyone*, including the people you'd rather it didn't. So they decided it was too dangerous to leave in everyone's hands, and they shut it down.

I think we've all heard this story before.

I was thinking about it because I was living next to a smaller version of it. In the middle of all the drama around the Fable 5 launch, I was heads-down building a side project, and the model I was leaning on kept flickering in and out of existence — here, then gated, then back. It's a strange feeling to build on top of an intelligence that someone else can unplug. It made me want something of my own that *stayed*. Not a smarter model. A memory.

## What I was actually building

The project is called **iforgot** — the engine underneath is "ForgetfulDB," and yes, the name is the joke. It forgets, on purpose.

The idea started with a small, slightly uncomfortable observation about myself: the biggest problem I had in 9th grade is almost certainly gone from my head. Not misfiled — gone. And that's not a malfunction. Forgetting is an *intentional design pattern* in human beings. It's a compression algorithm for a life: irrelevant memories decay quietly so the important ones have room to mean something. We don't remember everything because remembering everything would be useless.

Most AI memory systems do the opposite. They hoard. Keep every message, embed it, and pull back whatever looks similar. It feels responsible — why throw anything away? — but it fails in three predictable ways. The prompt grows without bound. Stale facts stay confidently wrong. And the *signal* (what mattered) drowns in the *routine* (what merely happened).

So I prompted my way toward a different premise: build a memory layer that behaves like human memory. Private, local, exportable. Something I could talk to every day, that would quietly take notes, work with whatever model I happened to like that week, and at the end of the year answer a question like *"can you write me a year-end review?"* — not by replaying twelve months of transcripts, but by remembering the parts worth remembering.

I wanted it to forget. Just — not at quite the same pace I do. Otherwise, what's the point?

## Memory is not context

Here's the distinction the whole project rests on, and it took me a while to say it cleanly: **context is the live transcript; memory is what survives it.**

They're not the same thing, and treating them as the same thing is why so many "memory" features feel like a longer scrollback rather than an actual mind.

| | Context | Memory |
|---|---|---|
| Size | Bounded | Unbounded |
| Order | Ordered | Unordered |
| Completeness | Complete | Lossy |
| Lifespan | Ephemeral | Persistent |

Context is the conversation you're in right now: bounded by a window, ordered turn-by-turn, complete, and gone the moment the window scrolls past it. Memory is the curated store underneath — unbounded, unordered, lossy, and persistent. It's *deliberately* incomplete, because the incompleteness is the whole idea. A memory that keeps everything isn't a memory. It's a log.

That one sentence — *memory is what survives the transcript* — is the design brief for everything below.

## The thesis: forgetting is the feature

Once you accept that memory should be lossy, the engineering question flips. The hard part isn't *storing*. SQLite stores. The hard part is the policy on top: what to keep, what to let fade, what to merge, what to promote, what to connect. ForgetfulDB is that policy. It's not a database — a database stores faithfully and forever. This is a *forgetting engine*, and SQLite is just where it happens to write things down.

I ended up organizing the policy as six orthogonal mechanisms over the same set of memories. The one-line version I keep coming back to:

> **decay forgets · salience keeps · habit reinforces · epochs organize · edges connect · dreaming creates.**

| Axis | What it does |
|---|---|
| **Decay** | Forgets the unused — exponential, per-type half-lives |
| **Salience** | Keeps the formative — the surprising *and* the habitual resist forgetting |
| **Abstraction** | Turns repetition into traits (raw → episodic → semantic) |
| **Epochs** | Organizes a lifetime into eras |
| **Edges** | Connects memories into a typed graph |
| **Dreaming** | *Creates* new memories and connections offline |

The rest of this piece walks the ones that are actually shipped — decay, salience, retrieval, consolidation — and ends on the ones that are still, fittingly, a dream.

## How memories fade

Every memory carries a **type**, and the type controls how fast it decays. This is the human-memory analogy made literal: a verbatim chat line should evaporate in days; a distilled fact should last for months.

| Type | Meaning | Half-life |
|---|---|---|
| `raw_event` | Verbatim input (a chat turn, a log line) | ~2 days |
| `episodic` | "What happened" | ~9 days |
| `semantic` | "What is true" — distilled facts | ~70 days |
| `procedural` | "How to do things" | ~70 days |
| `preference` | What you like | ~35 days |
| `archive` | Compressed, retired, hidden from normal recall | — |

The decay itself is one tidy equation:

```
decay_score = importance_score · exp(−λ · age_days)
```

It reads exactly how it sounds. As a memory ages, `age_days` grows; multiplied by a negative rate `−λ`, the exponent slides more negative; and `exp(of a big negative number)` collapses toward zero. So the same fact, with the same importance, scores lower the older it gets. Importance is what a memory is *worth*; the exponential is the tax that time charges on it.

A worked example makes it concrete. Take two memories with identical importance of 1.0, and pretend λ = 1 for the arithmetic:

```
1 day old:    1 · exp(−1 · 1)   = exp(−1)   ≈ 0.368        (3.68 × 10⁻¹)
100 days old: 1 · exp(−1 · 100) = exp(−100) ≈ 3.72 × 10⁻⁴⁴
```

Same importance, wildly different fate. The day-old memory keeps about a third of its worth; the hundred-day-old one has been taxed into oblivion — a decimal point followed by forty-three zeros before the first real digit. That's the feature working. Old and untouched should mean *nearly gone*.

In the real system, λ isn't 1 — it's tuned per type so the half-lives line up with the table above (raw events at λ = 0.35 give ln 2 / 0.35 ≈ 2 days; semantic facts at λ = 0.01 give ≈ 70). And two memories are exempt from the tax entirely: **pinned** ones, which never decay and are never evicted, and — more interestingly — the *salient* ones.

## What keeps a memory alive

If decay were the only force, the system would slowly forget everything it wasn't constantly told. That's wrong in a specific way: the memories that define you are often the ones you *don't* repeat. The day everything changed. The preference you've never had to restate because it's just true.

So there's a second, opposing axis: **salience** — what resists forgetting. And the shape that matters here is a U, not a line. A memory is salient if it is either **surprising** *or* **habitual**:

- **Surprise** = how novel it was. Formally, `1 − (max similarity to anything already stored)`. A memory unlike everything else you've ever said is, by definition, news.
- **Habit** = how reliably it recurs over time. Not "appeared a lot last Tuesday" — that's a blip. "Shows up evenly across months" — that's a trait.

The thing I'm proudest of is that *one* computation drives this. Take a memory, find its near-neighbors in meaning, and look at how those neighbors are spread across time:

```
sparse neighbors             → surprise   (novel — keep it)
dense + temporally tight     → burst      (a one-off — collapse it to gist)
dense + temporally spread    → habit      (a stable pattern — promote it)
```

Built once, read three ways. A salient memory doesn't just rank higher; it literally decays slower — a fully-salient memory forgets at a fraction of the base rate — and once it crosses a threshold it's *kept through pruning* automatically, like a pin you never had to set by hand. That's how a formative memory survives the housekeeping that buries the routine around it.

There's a guard worth mentioning, because "novel = keep" is exploitable: pure novelty would let typos and garbage enshrine themselves forever (nothing is more "novel" than noise). So surprise is gated by content relevance. Novel *noise* gets no medal.

## Retrieval: keeping the live conversation in charge

Decay and salience decide what *exists*. Retrieval decides what shows up when you ask a question. Every time the engine pulls memories for a query, it scores each candidate:

```
retrieval_score =
  ( 0.45 · semantic_similarity
  + 0.20 · importance_score      (decay-adjusted)
  + 0.15 · recurrence_score
  + 0.10 · recency_score
  + 0.10 · pinned_boost
  − 0.20 · staleness_penalty )
  · conversational_damping        (chat path only)
```

Read it as a sentence. *Mostly* go by what's relevant to the question (semantic similarity is the heavyweight at 0.45). Then nudge for what's important, what recurs, what's recent, what's pinned — and actively *subtract* for staleness, because a contradicted memory should fight to get back in, not coast on similarity.

Two refinements keep this honest, and both exist to solve the same failure: an old memory hijacking the present conversation.

The first is a **relevance gate**. If nothing clears a minimum score, the engine injects *nothing* — even if there was room for more. An empty memory block beats a misleading one. The temptation to "fill the top-k" is exactly how irrelevant memories sneak in.

The second is **conversational damping**, and it's my favorite small idea in the system. Verbatim chat turns — the raw "you said / it said" lines from old sessions — get their score multiplied by a damping factor (0.6 by default). The effect: an old conversation can *inform* the current one but can't *seize* it. Crucially, this penalty applies only to raw verbatim turns. Facts that consolidation has *distilled* — semantic, preference, procedural — are unaffected. In other words, the way an old chat earns full rank again isn't by being recent. It's by being *processed into a fact*. Which brings us to sleep.

## The sleep cycle

Humans consolidate memory during sleep — replaying the day, throwing most of it out, filing the rest. ForgetfulDB has a literal equivalent, a pass it runs offline:

> dedup-merge → recurrence refresh → salience revision → cluster summaries → episodic → semantic promotion → contradiction-staling → archive / prune → rebuild the association graphs.

That's the whole arc of forgetting-done-well in one pipeline. Duplicates merge. Recurring clusters collapse into a single summary. Episodes that keep getting rehearsed graduate into durable facts. Memories that newer ones contradict get marked **stale** — not deleted, just demoted out of normal recall (you can still ask for them explicitly). And the routine that's decayed past usefulness gets archived or pruned.

Notice what this does to a single fact's life: it can enter as a fast-decaying raw chat turn, get promoted to an episode, then to a semantic fact that lasts for months — *raw → episodic → semantic*. The lossy-compression story isn't a metaphor bolted on the side. It's the data flow.

## Watching it think

A system that forgets is a system you'll occasionally distrust. *Why did it remember that and not this?* So the engine ships with a read-only observability UI — not a memory editor, a window — and it's the part that made the abstractions click for me.

There's a force-directed **memory graph** where node size is importance, opacity is decay (faded nodes are literally fading), color is type, and a ring marks pinned memories. A time-scrubber recomputes decay as of any past moment, so you can drag backward and *watch memories dim and wink out* across the store's life. There's a **retrieval inspector** that runs the exact scoring above and shows you the per-component bars for every candidate — including the near-misses that scored but fell below the gate. And there's a consolidation timeline showing each "N memories → 1 summary" collapse. The whole thing updates live as you chat.

Making the mechanism inspectable wasn't a nice-to-have. When a memory behaves strangely, I can see *which* of the six axes did it. Debuggability is a first-class constraint, because a forgetting engine you can't audit is just a system that loses your things.

## Why I won't chase the benchmarks

The obvious move would be to point this at the standard long-term-memory benchmarks (LoCoMo, LongMemEval) and chase a leaderboard number. I'm deliberately not, and the reason is the whole thesis in miniature: the vast majority of those questions need only a couple of prior sessions and assume stored facts stay valid forever. They *structurally reward hoarding* and *penalize forgetting*. Optimizing against them would mean building the opposite of this project.

The metric I actually care about is **retention efficiency**: accuracy *per token of memory injected*. Every accuracy number paired with its token cost. That's the number that flatters forgetting — near-equal answers at a fraction of the context — and it's the only honest way to score a system whose entire pitch is "keep less, mean more."

## Still a dream

Plenty is shipped — decay, salience, the typed association graph, real local embeddings, consolidation, the UI. Plenty isn't, and the unbuilt parts are the ones I find most exciting:

A **foundation tier** of trait memories — decay-exempt identity, *concluded* by the system from accumulated habit ("you've started this kind of project four times in three months → that's who you are"). **Epochs**, which segment a lifetime into eras by detecting drift, so the engine can reason about "during the Clarity era" the way you reason about "back in college." **Multi-hop traversal**, where retrieval stops being a flat list and becomes a walk along the graph — the difference between *recalling* and *thinking*. And **dreaming**: sampling unconnected memories offline and testing whether a new connection should exist — *"both of those projects failed for the same reason"* — the only mechanism that doesn't retrieve or compress but genuinely *creates*. (Strictly low-confidence, pruned hard if never reinforced. The confabulation guards are the feature, not an afterthought.)

There's a thread through all of it. The model I'm building on has no clock. It can't actually know what "now," "three years ago," or "every year around this time" mean — it only knows the tokens in front of it. The engine *can*, because it owns the timestamps. So in the one place where machines are usually worse than us — a felt sense of time — this one can be made exactly, boringly correct.

## Coda

The machine in the opening story got switched off because a tool that serves everyone is a tool no one can fully trust. I don't have an answer to that. But I notice that the version I'd actually trust — with a year of my conversations, with the small private facts of a life — isn't the one that remembers everything. It's the one that knows how to forget. Forgetting is what makes a memory mine instead of a transcript. It's the editing that turns a record into a self.

So I built a database that forgets on purpose. The name is a joke. The premise isn't.

*iforgot is local-first and open source — Rust and SQLite, no network calls beyond a local model. If a memory layer that forgets well sounds useful (or wrong), I'd love to hear which.*
