-- Per-turn chat metrics: the dataset for later context optimization
-- (context share of prompt, token throughput, retrieval cost).

CREATE TABLE IF NOT EXISTS chat_turns (
    id                   TEXT PRIMARY KEY,
    session_id           TEXT,
    created_at           INTEGER NOT NULL,
    user_text            TEXT NOT NULL,
    assistant_text       TEXT NOT NULL,
    model                TEXT NOT NULL,
    backend              TEXT NOT NULL,
    prompt_tokens        INTEGER,          -- NULL when the backend didn't report usage
    completion_tokens    INTEGER,
    total_duration_ms    INTEGER,
    llm_duration_ms      INTEGER,
    retrieve_duration_ms INTEGER NOT NULL,
    context_memory_count INTEGER NOT NULL,
    context_chars        INTEGER NOT NULL,
    memory_ids           TEXT NOT NULL     -- JSON array of injected memory ids
);

CREATE INDEX IF NOT EXISTS idx_chat_turns_session ON chat_turns (session_id);
CREATE INDEX IF NOT EXISTS idx_chat_turns_created ON chat_turns (created_at);
