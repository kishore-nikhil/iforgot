//! Reservoir sampling (Algorithm R): keep a uniform random sample of a
//! stream without knowing its length. Used when pruning old raw events so
//! a representative handful survives as an archive note.

pub struct ReservoirSampler<T> {
    capacity: usize,
    seen: u64,
    items: Vec<T>,
    rng_state: u64,
}

impl<T> ReservoirSampler<T> {
    pub fn new(capacity: usize) -> Self {
        ReservoirSampler {
            capacity: capacity.max(1),
            seen: 0,
            items: Vec::new(),
            // Seed from the clock; statistical quality needs are modest.
            rng_state: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0x853c_49e6_748f_ea9b)
                | 1,
        }
    }

    /// xorshift64* — tiny PRNG, good enough for sampling.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    pub fn add(&mut self, item: T) {
        self.seen += 1;
        if self.items.len() < self.capacity {
            self.items.push(item);
        } else {
            let j = self.next_u64() % self.seen;
            if (j as usize) < self.capacity {
                self.items[j as usize] = item;
            }
        }
    }

    pub fn seen(&self) -> u64 {
        self.seen
    }

    pub fn into_items(self) -> Vec<T> {
        self.items
    }

    pub fn items(&self) -> &[T] {
        &self.items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_exceeds_capacity() {
        let mut r = ReservoirSampler::new(5);
        for i in 0..1000 {
            r.add(i);
        }
        assert_eq!(r.items().len(), 5);
        assert_eq!(r.seen(), 1000);
    }

    #[test]
    fn keeps_everything_under_capacity() {
        let mut r = ReservoirSampler::new(10);
        for i in 0..3 {
            r.add(i);
        }
        assert_eq!(r.items(), &[0, 1, 2]);
    }
}
