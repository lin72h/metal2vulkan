//! CFG graph primitives: successor/predecessor adjacency and a dominator tree, over the
//! `BodyBlock` source CFG, plus the native-side SPIR-V (`crate::spirv_module::Block`, `Word` label)
//! equivalents used by the emitter's control-flow repair and structurizer (see the `spirv_*`
//! fns near the end of this file).
//!
//! This is the single home for low-level CFG graph analysis. Higher-level passes
//! (the loop-forest in [`super::loopforest`], the structurizer, merge inference)
//! build a [`Cfg`] once and query dominance/adjacency through it, instead of each
//! open-coding its own successor scan. The dominators themselves come from
//! [`crate::native::dominators`], which every dominator question in the crate goes through.

use super::blocks::block_successors;
use super::BodyBlock;
use crate::spirv_module::Block;
use crate::spirv_module::Operand;
use spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

/// Successor/predecessor adjacency of a function's source CFG. Only edges whose
/// target is a real block in the function are recorded — a branch to a
/// non-existent label is dropped, matching the natural-loop analysis contract.
pub(in crate::native) struct Cfg {
    /// The entry block (first block of the function body).
    pub(in crate::native) entry: String,
    /// Successors of each block, in terminator order (deduped/sorted for switches
    /// by [`block_successors`]); edges to a non-existent label are dropped, so
    /// every listed target is a real block of the function.
    pub(in crate::native) successors: HashMap<String, Vec<String>>,
    /// Predecessors of each block (inverse of `successors`, real edges only).
    pub(in crate::native) predecessors: HashMap<String, Vec<String>>,
    /// The set of real block names, for cheap membership tests.
    names: HashSet<String>,
    /// Block names in function order, entry first. The dominator computation speaks dense block
    /// indices, and this is the deterministic numbering it is handed.
    order: Vec<String>,
}

impl Cfg {
    /// Build the adjacency of `blocks`. Returns `None` for an empty function (no
    /// entry block). Edge construction order mirrors the block order so downstream
    /// reverse-postorder / dominator results are deterministic. Edges to a label
    /// that is not itself a block are dropped from both `successors` and
    /// `predecessors` (a branch to a non-existent label carries no CFG meaning).
    pub(in crate::native) fn from_blocks(blocks: &[BodyBlock]) -> Option<Cfg> {
        let entry = blocks.first()?.name.clone();
        // Only edges to real blocks count (a branch to a non-existent label is ignored here).
        let names: HashSet<String> = blocks.iter().map(|b| b.name.clone()).collect();
        let successors: HashMap<String, Vec<String>> = blocks
            .iter()
            .map(|b| {
                let s = block_successors(b)
                    .into_iter()
                    .filter(|t| names.contains(t.as_str()))
                    .collect();
                (b.name.clone(), s)
            })
            .collect();
        let mut predecessors: HashMap<String, Vec<String>> = HashMap::new();
        for b in blocks {
            for t in successors.get(&b.name).into_iter().flatten() {
                predecessors
                    .entry(t.clone())
                    .or_default()
                    .push(b.name.clone());
            }
        }
        Some(Cfg {
            entry,
            successors,
            predecessors,
            names,
            order: blocks.iter().map(|b| b.name.clone()).collect(),
        })
    }

    /// `true` if `name` is a real block of this function.
    pub(in crate::native) fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    /// The set of blocks reachable from `start` (inclusive) over real forward edges.
    pub(in crate::native) fn reachable_from(&self, start: &str) -> HashSet<String> {
        let mut reachable: HashSet<String> = HashSet::new();
        let mut stack = vec![start.to_string()];
        reachable.insert(start.to_string());
        while let Some(n) = stack.pop() {
            for t in self.successors.get(&n).into_iter().flatten() {
                if reachable.insert(t.clone()) {
                    stack.push(t.clone());
                }
            }
        }
        reachable
    }

    /// The immediate-dominator tree of this CFG.
    pub(in crate::native) fn dominators(&self) -> Dominators {
        let (idom, intervals) = named_dominators(&self.order, |name| {
            self.successors.get(name).map(Vec::as_slice).unwrap_or(&[])
        });
        Dominators { idom, intervals }
    }
}

/// Immediate dominators and Euler intervals of a CFG whose blocks are named.
///
/// `order` is the dense numbering, entry first; `successors` names each block's targets. Both the
/// forward dominators here and the reverse-graph post-dominators in [`super::loopforest`] are this
/// same question over a different graph, so both come through here to reach the crate's one
/// dominator computation in [`crate::native::dominators`].
pub(super) fn named_dominators<'a>(
    order: &'a [String],
    successors: impl Fn(&str) -> &'a [String],
) -> (HashMap<String, String>, HashMap<String, (usize, usize)>) {
    let index: HashMap<&str, usize> = order
        .iter()
        .enumerate()
        .map(|(position, name)| (name.as_str(), position))
        .collect();
    let dense: Vec<Vec<usize>> = order
        .iter()
        .map(|name| {
            successors(name)
                .iter()
                .filter_map(|target| index.get(target.as_str()).copied())
                .collect()
        })
        .collect();
    let predecessors = crate::native::dominators::build_predecessors(&dense);
    let (_, intervals, parents) = crate::native::dominators::dominance(&dense, &predecessors);
    let named_idom = parents
        .iter()
        .enumerate()
        .filter_map(|(block, parent)| Some((order[block].clone(), order[(*parent)?].clone())))
        .collect();
    let named_intervals = intervals
        .iter()
        .enumerate()
        .filter_map(|(block, interval)| Some((order[block].clone(), (*interval)?)))
        .collect();
    (named_idom, named_intervals)
}

/// Immediate-dominator relation of a source CFG (entry maps to itself).
#[derive(Clone, Debug, Default)]
pub(in crate::native) struct Dominators {
    /// Immediate dominator of each block (entry maps to itself).
    idom: HashMap<String, String>,
    /// Euler intervals of the immediate-dominator tree. Dominance is interval containment, avoiding
    /// an O(tree depth) parent walk at every query in large generated CFGs.
    intervals: HashMap<String, (usize, usize)>,
}

impl Dominators {
    /// `true` if `dominator` dominates `node` (walking the idom chain).
    pub(in crate::native) fn dominates(&self, dominator: &str, node: &str) -> bool {
        if dominator == node {
            return true;
        }
        let (Some(&(dom_in, dom_out)), Some(&(node_in, node_out))) =
            (self.intervals.get(dominator), self.intervals.get(node))
        else {
            return false;
        };
        dom_in <= node_in && node_out <= dom_out
    }

    /// Immediate dominator of `node` (the entry, and any node mapping to itself, returns `None`).
    pub(in crate::native) fn idom(&self, node: &str) -> Option<&str> {
        self.idom
            .get(node)
            .map(String::as_str)
            .filter(|d| *d != node)
    }
}

/// Blocks reachable from `start` (inclusive) over the given forward-edge adjacency.
///
/// Operates directly on a caller-built `successors` map rather than a [`Cfg`], for the
/// merge-inference passes in [`super::blocks`] that build a raw adjacency mid-computation
/// (sometimes with edges the [`Cfg`] real-block filter would drop). [`Cfg::reachable_from`]
/// is the equivalent query when a full `Cfg` is already in hand.
pub(super) fn reachable_from(
    start: &str,
    successors: &HashMap<String, Vec<String>>,
) -> HashSet<String> {
    let mut seen = HashSet::new();
    let mut stack = vec![start.to_string()];
    while let Some(block) = stack.pop() {
        if !seen.insert(block.clone()) {
            continue;
        }
        if let Some(next) = successors.get(&block) {
            stack.extend(next.iter().cloned());
        }
    }
    seen
}

// --- Owned SPIR-V (`Block`) CFG primitives -------------------------------------------------
//
// The above operate on the `BodyBlock` source CFG (String labels). The emitter's control-flow
// repair and structurizer also walk the post-emit owned `Block` layer (numeric `Word` labels),
// where the same successor/predecessor/reachability queries were open-coded four times over
// (`cloned_block_successors` in the emitter, `block_successor_ids` in cfg/repair, plus two more on
// the final passes). These are the single native-side home for the Word
// layer; the passes-side copies stay separate because the passes layer must not depend on `native`
// (that dependency would invert ownership). Behaviour is identical to the copies they replace.

/// Successor block labels of one owned `Block`, read from its terminator. Branch/BranchConditional
/// arms in operand order; switch default + case targets sorted and deduped. Non-terminating or
/// unstructured-terminator blocks yield no successors.
pub(in crate::native) fn spirv_block_successors(block: &Block) -> Vec<Word> {
    fn id_ref(operand: &Operand) -> Option<Word> {
        match operand {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        }
    }
    let Some(inst) = block.instructions.last() else {
        return Vec::new();
    };
    match inst.class.opcode {
        Op::Branch => inst.operands.first().and_then(id_ref).into_iter().collect(),
        Op::BranchConditional => inst
            .operands
            .iter()
            .skip(1)
            .take(2)
            .filter_map(id_ref)
            .collect(),
        Op::Switch => {
            let mut out = Vec::new();
            if let Some(default) = inst.operands.get(1).and_then(id_ref) {
                out.push(default);
            }
            let mut idx = 3;
            while idx < inst.operands.len() {
                if let Some(target) = inst.operands.get(idx).and_then(id_ref) {
                    out.push(target);
                }
                idx += 2;
            }
            out.sort_unstable();
            out.dedup();
            out
        }
        _ => Vec::new(),
    }
}

/// Forward-edge adjacency of an owned SPIR-V function body keyed by block label id
/// (via [`spirv_block_successors`]). Blocks without a label id are skipped.
pub(in crate::native) fn spirv_block_successors_by_label(
    blocks: &[Block],
) -> HashMap<Word, Vec<Word>> {
    blocks
        .iter()
        .filter_map(|block| {
            Some((
                block.label.as_ref()?.result_id?,
                spirv_block_successors(block),
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blk(name: &str, term: &str) -> BodyBlock {
        let name = format!("%{name}");
        let typed = crate::native::tir::lower_block_carrier(
            &name,
            &[term.to_string()],
            &std::collections::HashMap::new(),
        );
        BodyBlock {
            name,
            role: crate::native::cfg::BlockRole::Normal,
            typed: typed.map(Into::into),
        }
    }

    /// Definitional dominance oracle: `d` dominates `n` iff every path from the
    /// entry to `n` passes through `d`. Equivalently (for `d != n`, `n` reachable),
    /// `n` is NOT reachable from the entry once `d` is deleted from the graph. This
    /// is an independent check of [`Dominators`], reached through a different route than the
    /// dense-set oracle in [`crate::native::dominators`]: path reachability, not a fixpoint.
    fn oracle_dominates(cfg: &Cfg, dominator: &str, node: &str) -> bool {
        if dominator == node {
            return true;
        }
        // BFS from entry, never expanding through `dominator`.
        let mut seen: HashSet<&str> = HashSet::new();
        let mut stack: Vec<&str> = vec![cfg.entry.as_str()];
        if cfg.entry == dominator {
            // Entry dominates everything reachable; if we can't even start, node
            // is unreachable-from-entry-without-dominator, so dominated.
            return true;
        }
        seen.insert(cfg.entry.as_str());
        while let Some(cur) = stack.pop() {
            for t in cfg.successors.get(cur).into_iter().flatten() {
                if !cfg.contains(t) || t == dominator {
                    continue;
                }
                if seen.insert(t.as_str()) {
                    stack.push(t.as_str());
                }
            }
        }
        // `node` dominated by `dominator` iff it cannot be reached avoiding `dominator`.
        !seen.contains(node)
    }

    fn reachable_names(cfg: &Cfg) -> Vec<String> {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut stack = vec![cfg.entry.as_str()];
        seen.insert(cfg.entry.as_str());
        while let Some(cur) = stack.pop() {
            for t in cfg.successors.get(cur).into_iter().flatten() {
                if cfg.contains(t) && seen.insert(t.as_str()) {
                    stack.push(t.as_str());
                }
            }
        }
        let mut v: Vec<String> = seen.into_iter().map(str::to_string).collect();
        v.sort();
        v
    }

    /// Cross-check [`Dominators::dominates`] against the definitional oracle over
    /// every ordered pair of entry-reachable blocks.
    fn assert_dominance_matches_oracle(blocks: &[BodyBlock]) {
        let cfg = Cfg::from_blocks(blocks).expect("non-empty");
        let doms = cfg.dominators();
        let reachable = reachable_names(&cfg);
        for d in &reachable {
            for n in &reachable {
                assert_eq!(
                    doms.dominates(d, n),
                    oracle_dominates(&cfg, d, n),
                    "dominance mismatch: does {d} dominate {n}?"
                );
            }
        }
    }

    #[test]
    fn straight_line_chain() {
        let blocks = vec![
            blk("0", "br label %a"),
            blk("a", "br label %b"),
            blk("b", "ret void"),
        ];
        let cfg = Cfg::from_blocks(&blocks).unwrap();
        let doms = cfg.dominators();
        assert!(doms.dominates("%0", "%b"));
        assert!(doms.dominates("%a", "%b"));
        assert!(!doms.dominates("%b", "%a"));
        assert_eq!(doms.idom("%b"), Some("%a"));
        assert_eq!(doms.idom("%0"), None);
        assert_dominance_matches_oracle(&blocks);
    }

    #[test]
    fn diamond_merge_dominance() {
        // entry -> {t, f} -> m ; neither arm dominates m, entry does.
        let blocks = vec![
            blk("0", "br i1 %c, label %t, label %f"),
            blk("t", "br label %m"),
            blk("f", "br label %m"),
            blk("m", "ret void"),
        ];
        let cfg = Cfg::from_blocks(&blocks).unwrap();
        let doms = cfg.dominators();
        assert!(doms.dominates("%0", "%m"));
        assert!(!doms.dominates("%t", "%m"));
        assert!(!doms.dominates("%f", "%m"));
        assert_eq!(doms.idom("%m"), Some("%0"));
        assert_dominance_matches_oracle(&blocks);
    }

    #[test]
    fn self_loop_and_nested_loop_dominance() {
        // Nested loop shape: outer header o, inner header i, latch back-edges.
        let blocks = vec![
            blk("0", "br label %o"),
            blk("o", "br i1 %c0, label %i, label %x"),
            blk("i", "br i1 %c1, label %i, label %ol"),
            blk("ol", "br i1 %c2, label %o, label %x"),
            blk("x", "ret void"),
        ];
        let cfg = Cfg::from_blocks(&blocks).unwrap();
        let doms = cfg.dominators();
        // Outer header dominates everything inside the loop.
        assert!(doms.dominates("%o", "%i"));
        assert!(doms.dominates("%o", "%ol"));
        assert!(doms.dominates("%o", "%x"));
        // Inner header dominates its latch back to the outer latch.
        assert!(doms.dominates("%i", "%ol"));
        // But the outer latch does not dominate the inner header.
        assert!(!doms.dominates("%ol", "%i"));
        assert_dominance_matches_oracle(&blocks);
    }

    #[test]
    fn edges_to_missing_labels_are_dropped() {
        // Branch to %ghost (no such block) must not create a phantom node, and the
        // dangling edge is dropped from both successors and predecessors.
        let blocks = vec![
            blk("0", "br i1 %c, label %a, label %ghost"),
            blk("a", "ret void"),
        ];
        let cfg = Cfg::from_blocks(&blocks).unwrap();
        assert!(cfg.contains("%a"));
        assert!(!cfg.contains("%ghost"));
        assert_eq!(
            cfg.successors.get("%0").map(Vec::as_slice),
            Some(&["%a".to_string()][..])
        );
        assert!(!cfg.predecessors.contains_key("%ghost"));
        assert_dominance_matches_oracle(&blocks);
    }

    #[test]
    fn reachable_from_covers_forward_edges_only() {
        // %dead is not reachable from the entry; %m is (via both arms).
        let blocks = vec![
            blk("0", "br i1 %c, label %t, label %f"),
            blk("t", "br label %m"),
            blk("f", "br label %m"),
            blk("m", "ret void"),
            blk("dead", "br label %m"),
        ];
        let cfg = Cfg::from_blocks(&blocks).unwrap();
        let r = cfg.reachable_from("%0");
        assert!(r.contains("%0"));
        assert!(r.contains("%t") && r.contains("%f") && r.contains("%m"));
        assert!(!r.contains("%dead"));
        // From %m only %m is reachable (it returns).
        assert_eq!(cfg.reachable_from("%m"), HashSet::from(["%m".to_string()]));
    }

    #[test]
    fn empty_function_has_no_cfg() {
        assert!(Cfg::from_blocks(&[]).is_none());
    }
}
