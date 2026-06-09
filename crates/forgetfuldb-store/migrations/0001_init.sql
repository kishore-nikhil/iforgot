-- ForgetfulDB initial schema.

CREATE TABLE IF NOT EXISTS memory_items (
    id               TEXT PRIMARY KEY,
    content          TEXT NOT NULL,
    summary          TEXT,
    memory_type      TEXT NOT NULL CHECK (memory_type IN
                       ('raw_event','episodic','semantic','procedural','preference','archive')),
    source           TEXT,
    topic            TEXT,
    entities         TEXT,            -- JSON array of strings
    tags             TEXT,            -- JSON array of strings
    created_at       INTEGER NOT NULL,
    updated_at       INTEGER NOT NULL,
    last_accessed_at INTEGER,
    access_count     INTEGER NOT NULL DEFAULT 0,
    importance_score REAL NOT NULL DEFAULT 0.5,
    recurrence_score REAL NOT NULL DEFAULT 0.0,
    recency_score    REAL NOT NULL DEFAULT 1.0,
    decay_score      REAL NOT NULL DEFAULT 0.5,
    confidence       REAL NOT NULL DEFAULT 1.0,
    stale            INTEGER NOT NULL DEFAULT 0,
    pinned           INTEGER NOT NULL DEFAULT 0,
    embedding        BLOB,            -- little-endian f32 vector
    content_hash     TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_memory_items_type    ON memory_items (memory_type);
CREATE INDEX IF NOT EXISTS idx_memory_items_topic   ON memory_items (topic);
CREATE INDEX IF NOT EXISTS idx_memory_items_created ON memory_items (created_at);

CREATE TABLE IF NOT EXISTS memory_links (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation  TEXT NOT NULL CHECK (relation IN
                ('supports','contradicts','updates','duplicates','derived_from','belongs_to_project')),
    PRIMARY KEY (source_id, target_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_memory_links_target ON memory_links (target_id);

CREATE TABLE IF NOT EXISTS raw_events (
    id           TEXT PRIMARY KEY,
    session_id   TEXT,
    role         TEXT,
    content      TEXT NOT NULL,
    created_at   INTEGER NOT NULL,
    content_hash TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_raw_events_hash    ON raw_events (content_hash);
CREATE INDEX IF NOT EXISTS idx_raw_events_session ON raw_events (session_id);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
