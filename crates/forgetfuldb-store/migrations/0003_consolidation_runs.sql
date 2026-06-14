-- Log of consolidation ("sleep cycle") runs, for the observability UI's
-- consolidation diff view. Counts are denormalized for cheap listing;
-- `summaries` holds the provenance detail as JSON:
--   [{"summary_id": "...", "source_ids": ["...", ...]}, ...]

CREATE TABLE IF NOT EXISTS consolidation_runs (
    id                  TEXT PRIMARY KEY,
    ran_at              INTEGER NOT NULL,
    duplicates_merged   INTEGER NOT NULL DEFAULT 0,
    recurrence_updated  INTEGER NOT NULL DEFAULT 0,
    clusters_summarized INTEGER NOT NULL DEFAULT 0,
    promoted            INTEGER NOT NULL DEFAULT 0,
    marked_stale        INTEGER NOT NULL DEFAULT 0,
    archived            INTEGER NOT NULL DEFAULT 0,
    pruned              INTEGER NOT NULL DEFAULT 0,
    summaries           TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_consolidation_runs_ran_at ON consolidation_runs (ran_at);
