-- Salience: how strongly a memory resists forgetting, independent of how
-- recently it was used (decay). U-shaped — high for both the surprising
-- (novel) and the habitual (recurring evenly over time). Provisional value
-- set at ingest from novelty; the authoritative value is recomputed each
-- consolidation by the neighbor-density discriminator
-- (forgetfuldb-core::salience). 0 = unremarkable.

ALTER TABLE memory_items ADD COLUMN salience REAL NOT NULL DEFAULT 0.0;

CREATE INDEX IF NOT EXISTS idx_memory_items_salience ON memory_items (salience);
