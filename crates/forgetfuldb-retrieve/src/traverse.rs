//! Multi-hop spreading activation over the typed memory graph.
//!
//! Plain retrieval scores each memory against the query in isolation. This
//! turns that into a *walk*: from the top-scoring seeds, activation spreads
//! along `co_occurred` / `semantic_similar` / `sequence` edges, decaying each
//! hop, so a memory that doesn't match the query but is *connected* to one
//! that does can still surface — and carries the path that brought it in.
//!
//! Pure and deterministic (no SQLite, no embeddings), like
//! [`forgetfuldb_core::salience`] / `epochs`: the graph algorithm is testable
//! on its own, and the caller ([`crate::retrieve`]) owns the SQLite→adjacency
//! mapping and the scoring policy (capping, conversational dominance).
//!
//! The generalization is exact: at `max_hops == 1` over `co_occurred` edges
//! this is the original one-hop boost, just unrolled to depth K.

use std::collections::{HashMap, HashSet};

/// One outgoing edge in the adjacency the walk traverses. The caller resolves
/// these from `memory_edges` (and decides directionality — undirected edge
/// types are listed in both directions).
#[derive(Debug, Clone)]
pub struct AdjEdge {
    pub dst: String,
    pub edge_type: String,
    pub weight: f64,
}

/// What the walk concluded about one reached memory.
#[derive(Debug, Clone)]
pub struct Reach {
    /// Accumulated incoming activation across every path that reached this
    /// node — the boost the caller folds (capped) into the node's score. A
    /// seed's own base score is *not* counted here (it's already in its
    /// score); a seed only gets activation if the graph routes back to it.
    pub activation: f64,
    /// Hops from the nearest seed (0 for seeds).
    pub depth: usize,
    /// Node ids from the seed to this node, e.g. `[A, B, C]`.
    pub path: Vec<String>,
    /// Edge type per hop; `edges[i]` connects `path[i]` → `path[i+1]`, so
    /// `edges.len() == path.len() - 1`.
    pub edges: Vec<String>,
}

/// Tunable walk parameters. Per-edge-type factors let `sequence` (a causal /
/// reasoning-path edge) propagate more strongly than `semantic_similar`
/// (mere closeness). Defaults are conservative; tune offline.
#[derive(Debug, Clone, Copy)]
pub struct TraverseParams {
    /// How many edges out from a seed the walk may go. `1` reproduces the
    /// original one-hop boost.
    pub max_hops: usize,
    /// Per-hop attenuation in `(0, 1]`: hop 2 is worth `hop_decay` of hop 1.
    pub hop_decay: f64,
    /// Contributions below this are dropped — keeps a long chain of weak
    /// edges from leaking activation across the whole graph.
    pub activation_floor: f64,
    pub co_occurred_factor: f64,
    pub semantic_factor: f64,
    pub sequence_factor: f64,
}

impl Default for TraverseParams {
    fn default() -> Self {
        TraverseParams {
            max_hops: 2,
            hop_decay: 0.5,
            activation_floor: 0.01,
            co_occurred_factor: 0.8,
            semantic_factor: 0.6,
            sequence_factor: 1.0,
        }
    }
}

impl TraverseParams {
    /// Propagation factor for an edge type. Unknown types don't propagate.
    pub fn factor(&self, edge_type: &str) -> f64 {
        match edge_type {
            "co_occurred" => self.co_occurred_factor,
            "semantic_similar" => self.semantic_factor,
            "sequence" => self.sequence_factor,
            _ => 0.0,
        }
    }
}

/// Spread activation from `seeds` over `adj` and return what each reached
/// memory accumulated. Bounded layer-BFS: each node is *expanded* at most
/// once (the cycle/cost guard), but activation from every path that reaches
/// it still accumulates into its boost. A node propagates with its
/// first-arrival strength; the recorded path is the (shortest) first arrival.
pub fn traverse(seeds: &[(String, f64)], adj: &HashMap<String, Vec<AdjEdge>>, p: &TraverseParams) -> HashMap<String, Reach> {
    let mut best: HashMap<String, Reach> = HashMap::new();
    // Strength a node propagates with — its base score (seeds) or its
    // first-arrival contribution (others). Distinct from `activation`, which
    // is the accumulated incoming boost.
    let mut strength: HashMap<String, f64> = HashMap::new();

    for (id, base) in seeds {
        best.insert(id.clone(), Reach { activation: 0.0, depth: 0, path: vec![id.clone()], edges: vec![] });
        strength.insert(id.clone(), *base);
    }

    let mut frontier: Vec<String> = seeds.iter().map(|(id, _)| id.clone()).collect();
    let mut expanded: HashSet<String> = HashSet::new();

    for _ in 0..p.max_hops {
        frontier.sort(); // stable order → deterministic path tie-breaks
        let mut next: Vec<String> = Vec::new();
        let mut queued: HashSet<String> = HashSet::new();

        for u in &frontier {
            if !expanded.insert(u.clone()) {
                continue; // expand each node once
            }
            let u_strength = strength.get(u).copied().unwrap_or(0.0);
            if u_strength <= 0.0 {
                continue;
            }
            let (u_path, u_edges, u_depth) = {
                let r = &best[u];
                (r.path.clone(), r.edges.clone(), r.depth)
            };
            let Some(neighbors) = adj.get(u) else { continue };
            for e in neighbors {
                let factor = p.factor(&e.edge_type);
                if factor <= 0.0 || e.weight <= 0.0 {
                    continue;
                }
                let contrib = u_strength * p.hop_decay * factor * (e.weight / (1.0 + e.weight));
                if contrib < p.activation_floor {
                    continue;
                }
                match best.get_mut(&e.dst) {
                    Some(r) => r.activation += contrib, // another path into a known node
                    None => {
                        let mut path = u_path.clone();
                        path.push(e.dst.clone());
                        let mut edges = u_edges.clone();
                        edges.push(e.edge_type.clone());
                        best.insert(e.dst.clone(), Reach { activation: contrib, depth: u_depth + 1, path, edges });
                        strength.insert(e.dst.clone(), contrib);
                        if queued.insert(e.dst.clone()) {
                            next.push(e.dst.clone());
                        }
                    }
                }
            }
        }

        frontier = next;
        if frontier.is_empty() {
            break;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(dst: &str, kind: &str, w: f64) -> AdjEdge {
        AdjEdge { dst: dst.to_string(), edge_type: kind.to_string(), weight: w }
    }

    /// Build adjacency from `(src, dst, type, weight)`; `co_occurred` /
    /// `semantic_similar` are undirected (both directions), `sequence` is
    /// directed.
    fn graph(edges: &[(&str, &str, &str, f64)]) -> HashMap<String, Vec<AdjEdge>> {
        let mut adj: HashMap<String, Vec<AdjEdge>> = HashMap::new();
        for &(s, d, k, w) in edges {
            adj.entry(s.to_string()).or_default().push(e(d, k, w));
            if k != "sequence" {
                adj.entry(d.to_string()).or_default().push(e(s, k, w));
            }
        }
        adj
    }

    fn params(max_hops: usize) -> TraverseParams {
        TraverseParams { max_hops, ..Default::default() }
    }

    #[test]
    fn chain_reaches_two_hops_with_decay() {
        // A(seed) → B → C via directed sequence edges (nothing points back).
        let adj = graph(&[("A", "B", "sequence", 1.0), ("B", "C", "sequence", 1.0)]);
        let out = traverse(&[("A".into(), 1.0)], &adj, &params(2));

        assert!(out.contains_key("C"), "C is reachable in two hops");
        assert_eq!(out["C"].depth, 2);
        assert_eq!(out["C"].path, vec!["A", "B", "C"]);
        assert_eq!(out["C"].edges, vec!["sequence", "sequence"]);
        // Activation strictly attenuates with distance.
        assert!(out["B"].activation > out["C"].activation);
        assert!(out["C"].activation > 0.0);
        // With no back-edge, the seed gets no self-boost.
        assert_eq!(out["A"].activation, 0.0);
        assert_eq!(out["A"].depth, 0);
    }

    #[test]
    fn undirected_edges_route_activation_back_to_a_seed() {
        // co_occurred is undirected, so a dense neighbor feeds a little
        // activation back to the seed — a connected seed ranks higher.
        let adj = graph(&[("A", "B", "co_occurred", 1.0)]);
        let out = traverse(&[("A".into(), 1.0)], &adj, &params(2));
        assert!(out["A"].activation > 0.0, "the back-edge boosts the seed");
        assert!(out["B"].activation > out["A"].activation, "but the neighbor still gets more");
    }

    #[test]
    fn max_hops_one_is_the_old_one_hop_boost() {
        let adj = graph(&[("A", "B", "co_occurred", 1.0), ("B", "C", "co_occurred", 1.0)]);
        let out = traverse(&[("A".into(), 1.0)], &adj, &params(1));
        assert!(out.contains_key("B"), "direct neighbor is boosted");
        assert!(!out.contains_key("C"), "two hops away is out of reach at max_hops=1");
    }

    #[test]
    fn activation_floor_cuts_weak_deep_paths() {
        // Weak edges: each hop multiplies by 0.5(decay)·0.8(co factor)·(0.1/1.1).
        let adj = graph(&[("A", "B", "co_occurred", 0.1), ("B", "C", "co_occurred", 0.1)]);
        let p = TraverseParams { max_hops: 3, activation_floor: 0.02, ..Default::default() };
        let out = traverse(&[("A".into(), 1.0)], &adj, &p);
        assert!(out.contains_key("B"), "first weak hop ~0.036 clears the floor");
        assert!(!out.contains_key("C"), "second weak hop falls below the floor");
    }

    #[test]
    fn cycles_terminate() {
        // A ↔ B ↔ C ↔ A — every node points at the others.
        let adj = graph(&[
            ("A", "B", "co_occurred", 1.0),
            ("B", "C", "co_occurred", 1.0),
            ("C", "A", "co_occurred", 1.0),
        ]);
        let out = traverse(&[("A".into(), 1.0)], &adj, &params(5));
        // Terminates, and reaches every node exactly once (expand-once guard).
        assert_eq!(out.len(), 3);
        assert!(out.values().all(|r| r.activation.is_finite()));
    }

    #[test]
    fn edge_type_factor_weights_paths() {
        // From A: a sequence edge to S and a semantic edge to M, equal weight.
        // sequence_factor (1.0) > semantic_factor (0.6) → S gets more.
        let adj = graph(&[("A", "S", "sequence", 1.0), ("A", "M", "semantic_similar", 1.0)]);
        let out = traverse(&[("A".into(), 1.0)], &adj, &params(1));
        assert!(out["S"].activation > out["M"].activation, "the causal edge propagates more strongly");
    }

    #[test]
    fn multiple_paths_accumulate() {
        // A and A2 are both seeds, both adjacent to T: T's boost sums both.
        let adj = graph(&[("A", "T", "co_occurred", 1.0), ("A2", "T", "co_occurred", 1.0)]);
        let one = traverse(&[("A".into(), 1.0)], &adj, &params(1));
        let two = traverse(&[("A".into(), 1.0), ("A2".into(), 1.0)], &adj, &params(1));
        assert!(two["T"].activation > one["T"].activation, "a second path adds activation");
        assert!((two["T"].activation - 2.0 * one["T"].activation).abs() < 1e-9);
    }

    #[test]
    fn unknown_edge_types_do_not_propagate() {
        let adj = graph(&[("A", "B", "belongs_to_project", 1.0)]);
        let out = traverse(&[("A".into(), 1.0)], &adj, &params(2));
        assert!(!out.contains_key("B"), "only co_occurred/semantic/sequence carry activation");
    }
}
