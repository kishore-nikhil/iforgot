-- V2 adaptive memory pipeline: evidence, source documents/chunks, themes,
-- and prediction/outcome snapshots. These are append-only/supporting tables;
-- the existing memory_items table remains the durable memory surface.

CREATE TABLE IF NOT EXISTS memory_evidence (
    id            TEXT PRIMARY KEY,
    memory_id     TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    strength      REAL NOT NULL,
    source        TEXT NOT NULL,
    session_id    TEXT,
    created_at    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_evidence_memory ON memory_evidence (memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_evidence_type   ON memory_evidence (evidence_type);
CREATE INDEX IF NOT EXISTS idx_memory_evidence_time   ON memory_evidence (created_at);

CREATE TABLE IF NOT EXISTS source_documents (
    id            TEXT PRIMARY KEY,
    raw_text_hash TEXT NOT NULL UNIQUE,
    source_type   TEXT NOT NULL,
    session_id    TEXT,
    summary       TEXT,
    entities      TEXT NOT NULL DEFAULT '[]',
    topics        TEXT NOT NULL DEFAULT '[]',
    created_at    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_source_documents_session ON source_documents (session_id);
CREATE INDEX IF NOT EXISTS idx_source_documents_type    ON source_documents (source_type);

CREATE TABLE IF NOT EXISTS source_chunks (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    summary     TEXT,
    entities    TEXT NOT NULL DEFAULT '[]',
    topics      TEXT NOT NULL DEFAULT '[]',
    UNIQUE(source_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_source_chunks_source ON source_chunks (source_id);

CREATE TABLE IF NOT EXISTS session_themes (
    id               TEXT PRIMARY KEY,
    label            TEXT NOT NULL,
    supporting_nodes TEXT NOT NULL DEFAULT '[]',
    confidence       REAL NOT NULL,
    session_id       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_session_themes_session ON session_themes (session_id);

CREATE TABLE IF NOT EXISTS prediction_snapshots (
    id                                      TEXT PRIMARY KEY,
    memory_id                               TEXT NOT NULL,
    predicted_importance                    REAL NOT NULL,
    predicted_lifetime_days                 REAL NOT NULL,
    predicted_consolidation_probability     REAL NOT NULL,
    model_version                           TEXT NOT NULL,
    created_at                              INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prediction_snapshots_memory ON prediction_snapshots (memory_id);

CREATE TABLE IF NOT EXISTS outcome_snapshots (
    id                TEXT PRIMARY KEY,
    memory_id         TEXT NOT NULL,
    actual_importance REAL NOT NULL,
    evidence_count    INTEGER NOT NULL,
    survived_days     INTEGER NOT NULL,
    correction_count  INTEGER NOT NULL,
    created_at        INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_outcome_snapshots_memory ON outcome_snapshots (memory_id);
