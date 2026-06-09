//! Count-Min Sketch: approximate frequency counting for topics and
//! entities in sub-linear space. Estimates can only over-count (hash
//! collisions add, never subtract), which is fine for recurrence scoring.

use crate::seeded_hash;

pub struct CountMinSketch {
    /// depth rows x width columns of u32 counters.
    rows: Vec<Vec<u32>>,
    width: usize,
}

impl CountMinSketch {
    pub fn new(width: usize, depth: usize) -> Self {
        let width = width.max(8);
        let depth = depth.clamp(1, 16);
        CountMinSketch {
            rows: vec![vec![0u32; width]; depth],
            width,
        }
    }

    /// Reasonable default for tracking a few thousand distinct keys.
    pub fn default_size() -> Self {
        CountMinSketch::new(2048, 4)
    }

    pub fn add(&mut self, key: &str, count: u32) {
        for (i, row) in self.rows.iter_mut().enumerate() {
            let idx = (seeded_hash(i as u64 + 1, key) as usize) % self.width;
            row[idx] = row[idx].saturating_add(count);
        }
    }

    /// Point estimate: the minimum across rows (hence "count-min").
    pub fn estimate(&self, key: &str) -> u32 {
        self.rows
            .iter()
            .enumerate()
            .map(|(i, row)| row[(seeded_hash(i as u64 + 1, key) as usize) % self.width])
            .min()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimates_never_undercount() {
        let mut cms = CountMinSketch::default_size();
        for _ in 0..7 {
            cms.add("billing", 1);
        }
        cms.add("onboarding", 3);
        assert!(cms.estimate("billing") >= 7);
        assert!(cms.estimate("onboarding") >= 3);
    }

    #[test]
    fn unseen_keys_estimate_near_zero() {
        let mut cms = CountMinSketch::default_size();
        for i in 0..100 {
            cms.add(&format!("topic-{i}"), 1);
        }
        // With 2048x4 counters and 100 keys, an unseen key should read 0.
        assert_eq!(cms.estimate("never-added"), 0);
    }
}
