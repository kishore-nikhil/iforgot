-- Foundation memory type: a decay-exempt trait concluded by consolidation
-- from accumulated habit evidence. SQLite cannot ALTER a CHECK constraint,
-- so rebuild memory_items with 'foundation' added to the allowed set
-- (canonical table redefinition: new table → copy → swap → reindex).

CREATE TABLE memory_items_new (
    id               TEXT PRIMARY KEY,
    content          TEXT NOT NULL,
    summary          TEXT,
    memory_type      TEXT NOT NULL CHECK (memory_type IN
                       ('raw_event','episodic','semantic','procedural','preference','foundation','archive')),
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
    content_hash     TEXT NOT NULL UNIQUE,
    salience         REAL NOT NULL DEFAULT 0.0
);

INSERT INTO memory_items_new
    SELECT id, content, summary, memory_type, source, topic, entities, tags,
           created_at, updated_at, last_accessed_at, access_count,
           importance_score, recurrence_score, recency_score, decay_score,
           confidence, stale, pinned, embedding, content_hash, salience
    FROM memory_items;

DROP TABLE memory_items;
ALTER TABLE memory_items_new RENAME TO memory_items;

CREATE INDEX IF NOT EXISTS idx_memory_items_type     ON memory_items (memory_type);
CREATE INDEX IF NOT EXISTS idx_memory_items_topic    ON memory_items (topic);
CREATE INDEX IF NOT EXISTS idx_memory_items_created  ON memory_items (created_at);
CREATE INDEX IF NOT EXISTS idx_memory_items_salience ON memory_items (salience);
