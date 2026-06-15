-- Small key/value table for store-level metadata. Currently records the
-- embedding identity (backend, model, dim) the stored vectors were built
-- with, so a later switch to a different embedding model can be detected
-- and the store re-embedded instead of silently returning zero-similarity
-- (vectors of different dimensions are incomparable).

CREATE TABLE IF NOT EXISTS store_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
