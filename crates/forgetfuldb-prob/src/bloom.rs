//! Bloom filter for approximate "probably already ingested this content
//! hash" checks.
//!
//! NOT a retrieval index: it cannot rank, search, or compare meaning.
//! It only short-circuits exact-duplicate ingestion before we hit SQLite.
//! False positives are resolved by the database's UNIQUE constraint;
//! false negatives cannot occur.

use crate::seeded_hash;

pub struct BloomFilter {
    bits: Vec<u64>,
    /// Number of bits (m).
    m: u64,
    /// Number of hash functions (k).
    k: u32,
}

impl BloomFilter {
    /// Size the filter for `expected_items` at the given false-positive
    /// rate using the standard formulas m = -n ln p / (ln 2)^2 and
    /// k = (m/n) ln 2.
    pub fn with_capacity(expected_items: usize, fp_rate: f64) -> Self {
        let n = expected_items.max(1) as f64;
        let p = fp_rate.clamp(1e-9, 0.5);
        let ln2 = std::f64::consts::LN_2;
        let m = ((-n * p.ln()) / (ln2 * ln2)).ceil().max(64.0) as u64;
        let k = ((m as f64 / n) * ln2).round().clamp(1.0, 16.0) as u32;
        let words = m.div_ceil(64) as usize;
        BloomFilter { bits: vec![0u64; words], m, k }
    }

    /// Two seeded hashes combined via double hashing (Kirsch–Mitzenmacher)
    /// stand in for k independent hash functions.
    fn indexes(&self, key: &str) -> impl Iterator<Item = u64> + '_ {
        let h1 = seeded_hash(0x51_7c_c1_b7, key);
        let h2 = seeded_hash(0x85_eb_ca_6b, key) | 1; // odd, so it cycles
        let m = self.m;
        (0..self.k as u64).map(move |i| (h1.wrapping_add(i.wrapping_mul(h2))) % m)
    }

    pub fn insert(&mut self, key: &str) {
        let idx: Vec<u64> = self.indexes(key).collect();
        for bit in idx {
            self.bits[(bit / 64) as usize] |= 1u64 << (bit % 64);
        }
    }

    /// `true` means "probably seen"; `false` means "definitely never seen".
    pub fn contains(&self, key: &str) -> bool {
        self.indexes(key)
            .all(|bit| self.bits[(bit / 64) as usize] & (1u64 << (bit % 64)) != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_false_negatives() {
        let mut bf = BloomFilter::with_capacity(1000, 0.01);
        for i in 0..1000 {
            bf.insert(&format!("hash-{i}"));
        }
        for i in 0..1000 {
            assert!(bf.contains(&format!("hash-{i}")));
        }
    }

    #[test]
    fn unseen_keys_mostly_absent() {
        let mut bf = BloomFilter::with_capacity(1000, 0.01);
        for i in 0..1000 {
            bf.insert(&format!("hash-{i}"));
        }
        let false_positives = (0..1000)
            .filter(|i| bf.contains(&format!("other-{i}")))
            .count();
        // 1% target; allow generous slack to keep the test deterministic.
        assert!(false_positives < 50, "too many false positives: {false_positives}");
    }
}
