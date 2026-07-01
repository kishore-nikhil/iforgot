# Tier-2 Embedding-Space Surprise Segmentation — design

> Design doc for the `forgetfuldb-segment` crate and its two integration
> cutovers. Read alongside [memory-architecture.md](memory-architecture.md)
> (the §Epochs and §Consolidation sections) and the implementation spec this
> doc resolves. Status: **implemented** — all four phases landed, V-1..V-11 +
> the full workspace suite green, `forgetfuldb-segment` clippy-clean. Supersedes
> the drift method in [`core::epochs`](../crates/forgetfuldb-core/src/epochs.rs)
> (its `segment()` is deleted; boundaries now come from this crate).
>
> **Benchmark headline** (`cargo bench -p forgetfuldb-segment`): on 8 synthetic
> streams with known era boundaries, the new surprise segmenter recovers
> boundaries at **F1 = 1.00**, vs **0.85** for the reconstructed old
> centroid-drift method (blind to sub-`0.35` shifts) and **0.19** for naive
> fixed-chunking. Full pipeline runs ~235 ns/entry (sub-ms for 4 096 entries).
> Two additions emerged during implementation and are documented below: the
> absolute surprise floor (§3.2) and the coherence gate (§3.3).

## 1. Why this exists / what changes

Today the engine segments memory two ways, both crude:

- **Epochs** (`core::epochs::segment`) cut lifetime eras on raw centroid
  drift + a hysteresis counter. Wired via `segment_epochs`
  ([consolidate/lib.rs:126](../crates/forgetfuldb-consolidate/src/lib.rs:126)),
  written to the `epochs` table, consumed by `retrieve(epoch_ordinal=N)`
  ([retrieve/lib.rs:382](../crates/forgetfuldb-retrieve/src/lib.rs:382)),
  consolidate-within/preserve-across, and the `/epochs` UI.
- **Cluster summaries** (`summarize_clusters`,
  [consolidate/lib.rs:717](../crates/forgetfuldb-consolidate/src/lib.rs:717))
  group episodic memories by their `topic` **string** (a `HashMap`), then
  summarize groups ≥ `cluster_min_size`. Not temporal, not centroid-based.

This work replaces **both segmentation mechanisms** with one principled
segmenter: recency-weighted prediction error (surprise) against a rolling
window, an EM-LLM `μ + γσ` rolling threshold, and a graph-modularity
refinement pass. No LLM — embeddings only.

Two product decisions drive the scope (both confirmed):

1. **The surprise segmenter replaces the epochs drift method.** `core::epochs`
   keeps its *types* and the retrieval-facing API; its boundary *algorithm*
   moves to `forgetfuldb-segment`.
2. **Cluster summaries become temporal episodes.** `summarize_clusters` stops
   grouping by topic string and summarizes contiguous temporal events instead.

The hard separation from the spec still holds: this crate decides the **units**
of memory. It never reads or writes decay, retrieval, or scoring.

## 2. Crate boundaries & the dependency cycle

The spec's literal dependency graph (`segment → core` for `SegmentConfig`,
plus "refactor `core::epochs` to consume `segment`") is a **circular crate
dependency** — `core → segment → core` — and will not compile.

**Resolution:** `forgetfuldb-segment` produces only **pure index ranges**
(`Event { start, end }`). All timestamp/centroid/`EpochSpan` assembly lives in
`forgetfuldb-consolidate`, which is already allowed to depend on everything.
`core::epochs` keeps `EpochSpan`, `epoch_index_at`, and the `centroid_of`
helper (retrieval and span-assembly need them) but **loses its `segment()`
drift function**.

```
forgetfuldb-core        (SegmentConfig, EpochSpan, epoch_index_at)
   ▲      ▲
   │      │
forgetfuldb-embed   forgetfuldb-segment   (Event ranges, surprise, refine)
   ▲      ▲              ▲
   └──────┴──────────────┘
        forgetfuldb-consolidate   (segment_epochs adapter, summary cutover)
```

All arrows point one way. `forgetfuldb-segment` depends on `forgetfuldb-core`
(only for `SegmentConfig` at the API boundary) and `forgetfuldb-embed` (for
`EmbeddingProvider`). It must **not** depend on `-store`, `-retrieve`, or
`-consolidate` (NFR-5).

## 3. The mechanism

For an ordered stream `e_1..e_n` with embeddings `v_1..v_n`:

1. **Normalize** every input vector (FR-2). Zero-norm vector → that entry's
   surprise is forced to `1.0` (max), never a divide-by-zero.
2. **Predict** `v_i` from the prior window `v_{i-W}..v_{i-1}` using a
   `Predictor`; predicted vector is L2-normalized before return (FR-1).
3. **Surprise** `s_i = clamp(1 − cosine(predict(window), v_i), 0, 2)` (FR-3).
   The first `W` entries have no full window → marked **warm-up** (see §3.1).
4. **Threshold** `i` is a candidate boundary when `s_i > μ + γσ`, with `μ,σ`
   over a *trailing* window of size `τ` (FR-4).
5. **Refine** slide each candidate ±`refine_radius` to the locally most
   coherent cut (FR-5).
6. **Emit** contiguous `Event` ranges covering `0..n` (FR-6).

### 3.1 Threshold definition — the warm-up exclusion (correctness-critical)

The spec says compute `μ,σ` over the literal trailing slice `surprise[i-τ..i]`,
with warm-up entries defined as surprise `0.0`. **Taken literally, this fails
V-2 (single-topic null).** At `i = W` the trailing window is all warm-up zeros,
so `μ = σ = 0`, and the threshold `μ + γσ = 0` — *any* positive in-topic
surprise exceeds it and cuts a spurious boundary at index `W`. `min_event_len`
does not save it (both sides are ≥ 2).

**Design rule (deviation from the literal spec, required for V-2):**

- Warm-up positions (`i < W`) are **excluded** from `μ/σ` statistics and can
  never be boundaries.
- `μ,σ` are computed over the **real** surprises in the trailing window
  (`surprise[max(W, i−τ) .. i]`).
- A boundary requires **≥ 2 real samples** in that window and `σ` floored at a
  small `ε` (e.g. `1e-9`). With `σ ≤ ε` and no real spread, nothing is an
  outlier, so nothing cuts.

This makes V-1 and V-2 both hold: in V-1 the A→B jump at index 10 dwarfs the
two in-block samples before it (boundary fires); in V-2 every surprise sits at
the local mean, so `s_i − μ ≈ 0 < γσ` everywhere (no boundary). It is also
why V-1's defaults work — block length 10 > window 8 leaves ≥ 2 real samples
before the first boundary. **This rule must be stated in the code and covered
by V-2.**

### 3.2 Absolute surprise floor (correctness-critical, implementation finding)

The warm-up exclusion (§3.1) is necessary but **not sufficient** for V-2. A
purely relative `μ + γσ` threshold flags ~16% of *any* stationary signal at
`γ = 1` (that's just where the 1σ tail sits), so a flat single-topic stream
still cuts on its own noise once there are ≥ 2 real samples. Because surprise is
a cosine *distance* in `[0, 2]`, there is a natural absolute scale: intra-topic
jitter sits near `0` (cosine ≈ 1), a real era change near `1` (near-orthogonal).
So a candidate must clear **both** the local outlier test *and* an absolute
floor `MIN_ABS_SURPRISE = 0.10` — exactly the role the old epochs
`drift_threshold` played. This is what actually makes V-2 yield zero and is the
single most load-bearing constant in the crate (`threshold.rs`).

### 3.3 Coherence gate (benchmark-driven addition, beyond the original spec)

The surprise signal spikes on *any* rise in prediction error — including a mere
increase in within-topic **volatility**, not just a topic change. The
comparative benchmark (§ below) caught this: a calm→loud transition inside one
topic produced a false boundary the old drift method didn't. Since refinement
(FR-5) may only *move* boundaries, a separate post-step removes them: for each
surviving boundary, compare the mean direction of a `window`-sized neighborhood
just before vs. just after the cut; if they are nearly identical
(`cosine > 0.95`, i.e. < ~18° apart) the cut is a volatility artifact and is
dropped. The surprise signal *proposes*; the graph *disposes*. Geometry-based,
so it runs only on the embedding path (never on precomputed surprise). With it,
the new method reaches F1 = 1.00 on every benchmark scenario including
`volatile_stable` (`refine.rs::coherence_gate`).

### 3.4 Predictors (FR-1)

| Predictor | Behavior | Status |
| --- | --- | --- |
| `CentroidPredictor` | flat mean of the window | implemented (baseline for V-3) |
| `WeightedPredictor` | exponential recency weights, `weight_decay` | **default** |
| `ExtrapolatePredictor` | linear extrapolation of the trajectory | **stub + TODO** (falls back to weighted) |

`WeightedPredictor` is the default because it tracks a moving center: on a
gradual drift (V-3) it follows the arc and stays unsurprised, where a flat
centroid lags and over-cuts. V-3 asserts weighted < centroid boundary count —
that test justifies the default; if it fails the weighting is wrong.

## 4. Public API (`forgetfuldb-segment`)

```rust
/// Half-open range of entry indices forming one event. `end` exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event { pub start: usize, pub end: usize }

/// Boundaries + the embedding space they were computed in (FR-8) + the raw
/// surprise signal (consumed by epoch `drift_in` and the /epochs UI).
pub struct SegmentResult {
    pub events: Vec<Event>,
    /// Identity of the embedding model (see §6). `None` when segmenting raw
    /// vectors with no provider — the caller stamps provenance.
    pub embedding_model: Option<String>,
    /// Per-entry surprise (warm-up entries report 0.0). Same length as input.
    pub surprise: Vec<f64>,
}

pub trait Predictor {
    /// `window` is ordered oldest-first; returns the L2-normalized prediction.
    fn predict(&self, window: &[Vec<f32>]) -> Vec<f32>;
}
pub struct CentroidPredictor;
pub struct WeightedPredictor { pub decay: f64 }
pub struct ExtrapolatePredictor; // stub

// Building blocks (unit-tested directly: V-3..V-7)
pub fn surprise_signal<P: Predictor>(embs: &[Vec<f32>], pred: &P, window: usize) -> Vec<f64>;
pub fn detect_boundaries(surprise: &[f64], tau: usize, gamma: f64, min_event_len: usize) -> Vec<usize>;
pub fn refine_boundaries(candidates: &[usize], sim: &SimSource, radius: usize) -> Vec<usize>;

// Top-level (FR-6). `precomputed_surprise` is the tier-3 seam (§12): if Some,
// it bypasses FR-1..FR-3 and feeds straight into detect_boundaries.
pub fn segment_with_embeddings(
    embs: &[Vec<f32>],
    cfg: &SegmentConfig,
    precomputed_surprise: Option<&[f64]>,
) -> SegmentResult;

pub fn segment<E: EmbeddingProvider>(
    entries: &[String],
    embedder: &E,
    cfg: &SegmentConfig,
) -> SegmentResult; // embeds, stamps embedding_model, calls the above
```

Edge cases (FR-6): empty → `events: []`; `n ≤ min_event_len` → one event
`{0, n}`; events are contiguous, sorted, gap-free, overlap-free, cover `0..n`.

## 5. Refinement & complexity (FR-5, NFR-3)

The objective: for a candidate at position `b`, pick the offset in
`[b−radius, b+radius]` maximizing mean intra-segment cosine on both sides minus
the across-cut cosine (a modularity-style cut).

**We do not materialize the full `O(n²·d)` similarity matrix.** The spec allows
it but flags it as the dominant cost. Refinement only inspects a `±radius`
neighborhood, and the intra-segment coherence is evaluated over a **bounded
local window** around the cut (not the entire segment, which could be hundreds
of entries). `SimSource` computes cosines lazily on demand, keeping refinement
`O(num_boundaries · radius² · d)` — well under the ceiling. The `O(n²)` ceiling
and the per-session/day scope assumption are documented; we do **not** silently
degrade above it.

Invariants (FR-5 / V-7): refinement may only **move** boundaries. The spec's
"keep the higher-surprise one on collision" is reconciled by **clamping** a
refined boundary so it cannot cross or coincide with its neighbors (preserves
`min_event_len`), rather than dropping one — so the count is invariant and V-7's
"never adds/removes" holds literally.

## 6. Embedding provenance (FR-8) — `EmbeddingProvider` change

`EmbeddingProvider::name()` returns a `&'static str` — `"ollama"` for *every*
Ollama model ([ollama.rs:86](../crates/forgetfuldb-embed/src/ollama.rs:86)).
That cannot distinguish `embeddinggemma` from `nomic-embed-text`, so it fails
FR-8 ("comparable only within the same embedding space").

Add a default-method identity to the trait (backward compatible):

```rust
pub trait EmbeddingProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn dim(&self) -> usize;
    fn embed(&self, text: &str) -> Vec<f32>;

    /// Stable identity of the embedding space. Must match the string the
    /// store records under META_EMBED_MODEL so provenance lines up with the
    /// existing dimension-mismatch guard (pipeline.rs:98).
    fn model_id(&self) -> String { format!("{}:{}", self.name(), self.dim()) }
}
```

`OllamaEmbeddings` overrides it to fold in `self.model` (e.g.
`"ollama:embeddinggemma"`). `HashedBagOfWords` uses the default
(`"hashed_bow:<dim>"`). V-11 asserts the recorded provenance changes when the
stub's reported id changes.

## 7. Config (FR-7, V-9)

New top-level block in `forgetfuldb.toml` + `SegmentConfig` in
`forgetfuldb-core` (mirrors the `[spreading]` / `[contradiction]` pattern):

```toml
[segmentation]
window_size = 8          # W: prior entries the predictor sees
predictor = "weighted"   # "centroid" | "weighted" | "extrapolate"
weight_decay = 0.5       # exponential decay for the weighted predictor
threshold_window = 16    # τ: trailing window for μ/σ
gamma = 1.0              # γ: std-devs above local mean = boundary
refine_radius = 3        # ± positions refinement may shift a cut
min_event_len = 2        # minimum entries per event
```

```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Predictor { Centroid, Weighted, Extrapolate }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SegmentConfig {
    pub window_size: usize,
    pub predictor: Predictor,
    pub weight_decay: f64,
    pub threshold_window: usize,
    pub gamma: f64,
    pub refine_radius: usize,
    pub min_event_len: usize,
}
```

- **Typed `Predictor` enum** gives FR-7's "unknown predictor string → error at
  load" for free (serde rejects unknown variants).
- **Range validation is net-new infra.** `Config::load` is currently just
  `toml::from_str` ([config.rs:409](../crates/forgetfuldb-core/src/config.rs:409)) —
  it validates nothing. Add `Config::validate(&self) -> Result<()>`, called at
  the end of `load()`, delegating to `SegmentConfig::validate()`. Reject
  `window_size == 0`, `threshold_window == 0`, `gamma < 0.0`,
  `min_event_len == 0`, `weight_decay` outside `(0, 1]` (V-9).

### 7.1 Migration of the old epoch knobs

The epoch knobs under `[consolidation_thresholds]`
([config.rs:337](../crates/forgetfuldb-core/src/config.rs:337)) split:

| Knob | Fate |
| --- | --- |
| `epoch_drift_threshold` | **removed** — superseded by `gamma` / `threshold_window` |
| `epoch_hysteresis_runs` | **removed** — superseded by the rolling threshold |
| `epoch_min_size` | **kept** — the epoch adapter still enforces it (§8) |
| `epoch_min_days` | **kept** — the epoch adapter still enforces it (§8) |

`Config::epoch_params()` shrinks to just `min_size` / `min_days`.

## 8. Integration A — epochs cutover (Phase 3)

`segment_epochs` becomes a thin adapter over `forgetfuldb-segment`:

1. Load surviving memories, sorted by `created_at` asc → parallel `embs` and
   `timestamps` vectors. **Missing/heterogeneous embeddings** (`embedding:
   Option<Vec<f32>>`, [types.rs:473](../crates/forgetfuldb-core/src/types.rs:473))
   are filtered to the active embedding cohort first — a memory with no vector,
   or one whose `dim` ≠ the active provider's, is excluded from the stream
   (mixing embedding spaces produces garbage surprise). This filter is the
   adapter's responsibility, not the segmenter's.
2. `let res = segment_with_embeddings(&embs, &cfg.segmentation, None);`
3. Map each `Event { start, end }` → `EpochSpan { ordinal, started_at:
   ts[start], ended_at: ts[end] (None for the last), centroid:
   centroid_of(embs[start..end]), member_count, drift_in: res.surprise[start] }`.
4. **Re-impose `min_days` / `min_size`** (consequence of replacing the drift
   method — the segmenter is time-agnostic by design). Merge any event whose
   member count `< epoch_min_size` **or** whose span `< epoch_min_days` into its
   neighbor, matching today's "absorb the too-small lead-in" behavior. Without
   this, epochs regress to splitting same-day bursts into eras.
5. `store.replace_epochs(&spans)` — **unchanged**. The `epochs` table schema,
   `store.list_epochs()`, and `retrieve(epoch_ordinal=N)` are untouched
   (verified: retrieval only does a time-window lookup,
   [retrieve/lib.rs:382](../crates/forgetfuldb-retrieve/src/lib.rs:382)).

`core::epochs::segment` + `EpochParams` + their 8 unit tests are deleted. The
behavior tests that assert epoch outcomes (`two-topic→two-eras`,
`preserve-across-epochs`, the hysteresis test) are **re-baselined** against the
new mechanism — re-verified, not blindly deleted.

## 9. Integration B — temporal summaries (Phase 4)

`summarize_clusters` stops grouping by `topic` string and summarizes temporal
events:

1. Active `raw_event` + `episodic` memories (both variants exist in
   `MemoryType`), sorted by `created_at` asc, filtered to the active embedding
   cohort (§8.1).
2. `segment_with_embeddings` → events.
3. Each event with `member_count ≥ cluster_min_size` → one extractive summary
   (semantic memory, `derived_from` links, importance/recurrence as today).

**Interaction with `collapse_bursts` (consequence C).** Burst-collapse runs
*earlier* in the pass ([lib.rs:84](../crates/forgetfuldb-consolidate/src/lib.rs:84))
and already deletes burst members (keeping the outlier). By the time summary
runs it sees only survivors, so a burst is not double-summarized; the lone kept
outlier falls below `cluster_min_size` and is left alone. Order is load-bearing
and documented.

**`refine_topics` survives** — it still feeds contradiction candidate-gen and
foundation promotion; only its summary consumer goes away.

**Known semantic change (open question §13.1):** a topic revisited across three
sessions used to yield one summary; temporal events yield three (one per
episode). This is the intended effect of Decision 2 but worth a conscious sign-off.

## 10. Validation map (V-1 .. V-11)

| Test | Where | Notes |
| --- | --- | --- |
| V-1 three-era synthetic | `segment` unit | exact boundaries 10 & 20; **core, non-negotiable** |
| V-2 single-topic null | `segment` unit | guards the warm-up exclusion §3.1; **core** |
| V-3 gradual drift | `segment` unit | weighted < centroid; **core**, justifies the default |
| V-4 normalization guard | `segment` unit | non-unit + zero vector → no NaN, zero ⇒ surprise 1.0 |
| V-5 threshold locality | `segment` unit | volatile-but-stable half cuts nothing |
| V-6 min_event_len | `segment` unit | length-1 fragment absorbed |
| V-7 refinement | `segment` unit | off-by-one candidate snaps to truth; count invariant |
| V-8 coverage/invariants | `segment` proptest | contiguous, sorted, covers `0..n`, deterministic (NFR-2) |
| V-9 config loading | `core::config` | defaults, bad predictor, out-of-range |
| V-10 consolidator smoke | `consolidate` | real SQLite, events align with scripted eras; decay/retrieval tables untouched |
| V-11 provenance | `segment` / `consolidate` | swapping stub id changes recorded provenance |

## 11. Phased delivery

1. **`forgetfuldb-segment`, standalone & pure** — FR-1..FR-6, the §3.1 threshold
   rule, the tier-3 seam, V-1..V-8. No integration. **Gate: V-1/V-2/V-3 green
   before anything wires in** — they prove the mechanism, not just the plumbing.
2. **Provenance + config** — trait `model_id()` (§6); `[segmentation]` block,
   typed `Predictor`, `Config::validate()` (§7); example toml; V-9, V-11.
3. **Epochs cutover** — the §8 adapter; delete `core::epochs::segment`;
   re-baseline epoch tests; confirm `epoch_ordinal` + `/epochs` UI unaffected.
4. **Summary cutover** — §9; V-10; confirm decay/retrieval tables untouched.

Each phase is `cargo test` + `cargo clippy` clean before the next.

## 12. Tier-3 seam (out of scope, but reserved)

`precomputed_surprise: Option<&[f64]>` on `segment_with_embeddings` lets a
future token-logprob (Bayesian) surprise signal bypass FR-1..FR-3 and feed
straight into the threshold pass. **Not wired to Ollama.** This is the only
tier-3 hook; everything else in §6 of the spec stays out (LLM boundary
descriptions, abstractive summarization, re-embedding migration).

## 13. Risks & open questions

- **13.1 (decision needed) Cross-time summary aggregation.** §9 replaces
  topic-aggregated summaries with per-episode summaries. Recommend accepting it
  (it's Decision 2). Fallback if you want both: layer temporal events *under*
  topic clusters rather than replacing — but that keeps two grouping mechanisms.
- **13.2 Re-baselining, not deleting.** The epoch behavior tests encode real
  product expectations. Under the new mechanism their boundaries may shift by an
  index; each must be re-derived and re-asserted, not loosened to pass.
- **13.3 `min_days` fidelity.** The adapter merge (§8.4) must reproduce today's
  "same-day burst is one era" guarantee. Covered by porting the existing
  `min_days_prevents_short_epochs` test to the adapter.
- **13.4 Determinism (NFR-2).** f64 math, no RNG, no `HashMap` in logic. The
  proptest (V-8) asserts bit-identical re-runs.
- **13.5 Scope/cost.** `segment_epochs` and the summary pass run over the whole
  corpus, not a per-session/day batch. The §5 lazy similarity keeps refinement
  cheap; the surprise+threshold pass is `O(n·d)`. Documented ceiling, no silent
  degradation.

## 14. Out of scope

Tier-3 Bayesian surprise (beyond the §12 seam), LLM boundary descriptions,
abstractive summarization (the segmenter outputs ranges only), and re-embedding
migration when the model changes (this crate only records provenance, FR-8).
