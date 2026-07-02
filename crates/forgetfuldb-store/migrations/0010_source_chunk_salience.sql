-- Semantic source chunks. Chunks are now produced by surprise segmentation
-- (forgetfuldb-segment) over atomized text, and carry the embedding they were
-- segmented on plus a write-time salience (novelty x relevance). Both feed the
-- chunk -> durable-memory promotion decision and let low-salience chunks decay
-- fast instead of being dropped. Additive columns; existing rows default to no
-- embedding and salience 0.
ALTER TABLE source_chunks ADD COLUMN embedding BLOB;
ALTER TABLE source_chunks ADD COLUMN salience REAL NOT NULL DEFAULT 0;
