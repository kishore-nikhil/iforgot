-- Chunk lifecycle: staged chunks now decay for real. created_at lets the
-- consolidation prune pass age each chunk (raw-event lambda, resisted by the
-- chunk's salience) and delete what has faded below the archive floor —
-- "fast decay, not drop" becomes a system property instead of an eval-only
-- demonstration. Legacy rows default to 0 (unix epoch) and are pruned on the
-- first pass: they predate the lifecycle and were never readable anyway.
ALTER TABLE source_chunks ADD COLUMN created_at INTEGER NOT NULL DEFAULT 0;
