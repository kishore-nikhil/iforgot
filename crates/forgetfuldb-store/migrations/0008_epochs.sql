-- Epochs: drift-segmented eras over the memory timeline. Derived data —
-- rebuilt from scratch each consolidation by forgetfuldb-core::epochs — so
-- there is no foreign key to memory_items; membership is a time-range lookup
-- (started_at <= created_at < ended_at). The current era has ended_at NULL.

CREATE TABLE IF NOT EXISTS epochs (
    id           TEXT PRIMARY KEY,
    ordinal      INTEGER NOT NULL,        -- 0-based era index in time order
    started_at   INTEGER NOT NULL,
    ended_at     INTEGER,                 -- NULL = current/open era
    centroid     BLOB,                    -- little-endian f32, era identity vector
    label        TEXT,                    -- human-readable name
    summary      TEXT,                    -- extractive summary of the era
    member_count INTEGER NOT NULL DEFAULT 0,
    drift_in     REAL NOT NULL DEFAULT 0  -- drift that opened this era (0 for the first)
);

CREATE INDEX IF NOT EXISTS idx_epochs_started ON epochs (started_at);
