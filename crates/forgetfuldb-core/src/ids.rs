//! Compact unique IDs without an external UUID dependency.
//!
//! IDs combine the current time, a process-wide counter and a hash of the
//! payload, e.g. `mem_018f3a2b9c4d1e07`. Uniqueness is ultimately enforced
//! by SQLite primary keys; this just makes collisions implausible.

use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

pub fn new_id(prefix: &str, payload: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let count = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    nanos.hash(&mut hasher);
    count.hash(&mut hasher);
    payload.hash(&mut hasher);
    format!("{prefix}_{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_prefixed() {
        let a = new_id("mem", "same payload");
        let b = new_id("mem", "same payload");
        assert_ne!(a, b);
        assert!(a.starts_with("mem_"));
    }
}
