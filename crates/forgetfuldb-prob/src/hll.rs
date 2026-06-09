//! HyperLogLog: approximate distinct counting, used only for cheap stats
//! (e.g. "roughly how many distinct topics have ever been seen").

use crate::seeded_hash;

pub struct HyperLogLog {
    /// 2^p registers; p=12 gives ~1.6% standard error in 4 KiB.
    registers: Vec<u8>,
    p: u32,
}

impl HyperLogLog {
    pub fn new(p: u32) -> Self {
        let p = p.clamp(4, 16);
        HyperLogLog {
            registers: vec![0u8; 1 << p],
            p,
        }
    }

    pub fn add(&mut self, key: &str) {
        let hash = seeded_hash(0x9e37_79b9, key);
        let idx = (hash >> (64 - self.p)) as usize;
        // Rank = position of the first 1-bit in the remaining bits.
        let remaining = hash << self.p;
        let rank = (remaining.leading_zeros() + 1).min(64 - self.p + 1) as u8;
        if rank > self.registers[idx] {
            self.registers[idx] = rank;
        }
    }

    pub fn estimate(&self) -> f64 {
        let m = self.registers.len() as f64;
        let alpha = match self.registers.len() {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m),
        };
        let sum: f64 = self.registers.iter().map(|&r| 2f64.powi(-(r as i32))).sum();
        let raw = alpha * m * m / sum;
        // Small-range correction: fall back to linear counting.
        let zeros = self.registers.iter().filter(|&&r| r == 0).count();
        if raw <= 2.5 * m && zeros > 0 {
            m * (m / zeros as f64).ln()
        } else {
            raw
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_is_roughly_accurate() {
        let mut hll = HyperLogLog::new(12);
        let n = 5000;
        for i in 0..n {
            hll.add(&format!("item-{i}"));
        }
        let est = hll.estimate();
        let err = (est - n as f64).abs() / n as f64;
        assert!(err < 0.10, "estimate {est} too far from {n}");
    }

    #[test]
    fn duplicates_do_not_inflate() {
        let mut hll = HyperLogLog::new(12);
        for _ in 0..1000 {
            hll.add("same-key");
        }
        assert!(hll.estimate() < 5.0);
    }
}
