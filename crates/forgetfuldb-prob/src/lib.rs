//! forgetfuldb-prob
//!
//! Small, dependency-free probabilistic data structures.
//!
//! IMPORTANT: the Bloom filter here is **not** used for semantic
//! retrieval. Bloom filters can only answer "have I probably seen this
//! exact key before?" — they know nothing about meaning or similarity.
//! ForgetfulDB uses them strictly as a fast pre-check for content-hash
//! deduplication; the SQLite `UNIQUE` constraint on `content_hash`
//! remains the authoritative dedup mechanism (Bloom filters can return
//! false positives, never false negatives).
//!
//! - [`BloomFilter`]: approximate "seen before" membership.
//! - [`CountMinSketch`]: approximate frequency of topics/entities.
//! - [`HyperLogLog`]: approximate distinct counts (optional, for stats).
//! - [`ReservoirSampler`]: uniform sample of a stream (used to keep a
//!   representative sample of pruned raw events).

mod bloom;
mod cms;
mod hll;
mod reservoir;

pub use bloom::BloomFilter;
pub use cms::CountMinSketch;
pub use hll::HyperLogLog;
pub use reservoir::ReservoirSampler;

use std::hash::{Hash, Hasher};

/// Seeded 64-bit hash built on the std SipHash hasher. Writing the seed
/// before the key gives us a cheap family of independent-enough hash
/// functions without external dependencies.
pub(crate) fn seeded_hash<T: Hash + ?Sized>(seed: u64, key: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut hasher);
    key.hash(&mut hasher);
    hasher.finish()
}
