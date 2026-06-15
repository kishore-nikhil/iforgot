-- Weighted, decaying association edges between memories — distinct from
-- `memory_links` (typed but unweighted provenance: derived_from, updates,
-- contradicts, …). The first edge type is `co_occurred`: two memories
-- retrieved into the same chat turn. Weight is recency-decayed
-- co-occurrence strength, recomputed during consolidation from
-- `chat_turns.memory_ids`. Undirected edges are stored canonically with
-- src_id < dst_id so each pair has exactly one row per type.

CREATE TABLE IF NOT EXISTS memory_edges (
    src_id         TEXT NOT NULL,
    dst_id         TEXT NOT NULL,
    edge_type      TEXT NOT NULL,
    weight         REAL NOT NULL DEFAULT 0,
    co_count       INTEGER NOT NULL DEFAULT 0,
    created_at     INTEGER NOT NULL,
    last_activated INTEGER NOT NULL,
    PRIMARY KEY (src_id, dst_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_memory_edges_src ON memory_edges (src_id);
CREATE INDEX IF NOT EXISTS idx_memory_edges_dst ON memory_edges (dst_id);
