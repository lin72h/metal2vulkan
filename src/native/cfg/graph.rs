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

// How many source CFGs this thread has built, for the redundant-work regression checks.
//
// Building a [`Cfg`] is the unit of whole-function analysis: the dominator tree, the natural-loop
// forest, and every ownership decision derived from either start here. A translator that re-derives
// one per construct instead of maintaining it is quadratic in the block count, which is how the
// 20-second per-attempt ceiling in `AGENTS.md` gets broken -- and the count, unlike a wall time,
// is the same number on every machine.
//
// Thread-local because the unit tests run concurrently in one binary while a single translation is
// sequential. Test builds only; there is no counter in a release library.
#[cfg(test)]
thread_local! {
    static CFG_BUILDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn cfg_builds_bump() {
    CFG_BUILDS.with(|count| count.set(count.get() + 1));
}

/// Source CFGs built on this thread while `work` ran.
///
/// See the note on `CFG_BUILDS` above for why this number, rather than a wall time or a
/// high-water mark, is what a redundant-work regression check should assert on.
#[cfg(test)]
pub(in crate::native) fn cfg_builds_during<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let before = CFG_BUILDS.with(std::cell::Cell::get);
    let value = work();
    (value, CFG_BUILDS.with(std::cell::Cell::get) - before)
}

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
        #[cfg(test)]
        cfg_builds_bump();
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
        Dominators {
            idom,
            intervals,
            pass_throughs: HashMap::new(),
        }
    }
}

/// The dominator relation of `blocks`, and nothing else.
///
/// [`super::loopforest::analyze`] also answers dominance, but it pays for back-edge discovery, a
/// natural-loop body closure per header, and a nesting comparison between every pair of loops
/// before it can. A caller that only ever asks "does A dominate B" should ask that question, not
/// the one that happens to contain it: the ownership passes re-derive dominance after every graph
/// edit, so the difference is a whole-analysis constant multiplied by the number of edits.
///
/// An empty function has no entry, hence no dominance relation; the empty [`Dominators`] answers
/// `false` to every query but the reflexive one, matching [`super::loopforest::analyze`] there.
pub(in crate::native) fn block_dominators(blocks: &[BodyBlock]) -> Dominators {
    Cfg::from_blocks(blocks)
        .map(|cfg| cfg.dominators())
        .unwrap_or_default()
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

/// Immediate-dominator relation of a source CFG (entry maps to itself), together with any
/// pass-through blocks spliced into that CFG since it was derived.
#[derive(Clone, Debug, Default)]
pub(in crate::native) struct Dominators {
    /// Immediate dominator of each block (entry maps to itself).
    idom: HashMap<String, String>,
    /// Euler intervals of the immediate-dominator tree. Dominance is interval containment, avoiding
    /// an O(tree depth) parent walk at every query in large generated CFGs.
    intervals: HashMap<String, (usize, usize)>,
    /// Each pass-through given to [`Self::record_pass_through`], resolved to the blocks of the CFG
    /// this relation was derived from. Empty unless a caller records one.
    pass_throughs: HashMap<String, Vec<String>>,
}

impl Dominators {
    /// `true` if `dominator` dominates `node`.
    ///
    /// Once a pass-through has been recorded, `dominator` must be a block of the CFG this was
    /// derived from. A recorded pass-through is never the target of a later split -- split targets
    /// are chosen before any splitting starts -- so no caller has reason to ask what one dominates,
    /// and this cannot answer it.
    pub(in crate::native) fn dominates(&self, dominator: &str, node: &str) -> bool {
        if dominator == node {
            return true;
        }
        if !self.pass_throughs.is_empty() {
            debug_assert!(
                !self.pass_throughs.contains_key(dominator),
                "a pass-through is never a split target, so nothing asks what it dominates"
            );
            if let Some(originals) = self.pass_throughs.get(node) {
                return !originals.is_empty()
                    && originals
                        .iter()
                        .all(|original| self.dominates_derived(dominator, original));
            }
        }
        self.dominates_derived(dominator, node)
    }

    /// Interval containment, for two blocks of the CFG this relation was derived from.
    fn dominates_derived(&self, dominator: &str, node: &str) -> bool {
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

    /// Record that `name` was spliced in front of a block, carrying exactly `predecessors`.
    ///
    /// The ownership passes repeatedly split an edge: some predecessors of a block `T` are
    /// redirected through a fresh block `S` whose only successor is `T`. Re-deriving the whole
    /// relation after each split is the obvious way to keep answering dominance questions, and it
    /// makes a pass that performs one split per construct quadratic in the block count -- the shape
    /// that breaks the per-translation time ceiling on generated shaders.
    ///
    /// It is also unnecessary, twice over:
    ///
    /// * **The blocks that were already there do not move.** Every entry-to-`X` path after the split
    ///   is an entry-to-`X` path before it with `S` spliced in, and vice versa. The pre-existing
    ///   blocks on it are the same ones, so dominance among them is exactly unchanged.
    /// * **A pass-through's own dominators follow from its predecessors.** Every path to `S` is a
    ///   path to one of its predecessors plus one edge, so a block other than `S` dominates `S`
    ///   exactly when it dominates all of them. Resolving predecessors that are themselves recorded
    ///   pass-throughs leaves blocks of the derived CFG, which the unchanged relation answers for.
    ///
    /// So a pass that only ever splices pass-throughs records each one instead of re-deriving. The
    /// recording is exact, not an approximation: [`Self::dominates`] then agrees with a relation
    /// freshly derived from the mutated block list, which is what
    /// `a_recorded_pass_through_answers_what_a_fresh_analysis_answers` checks.
    ///
    /// `predecessors` is the complete set of edges redirected through it; a pass-through with no
    /// predecessors is unreachable, and is then dominated by nothing, as in
    /// [`crate::native::dominators`].
    pub(in crate::native) fn record_pass_through(&mut self, name: &str, predecessors: &[String]) {
        let mut derived = Vec::new();
        for predecessor in predecessors {
            match self.pass_throughs.get(predecessor) {
                Some(resolved) => derived.extend(resolved.iter().cloned()),
                None => derived.push(predecessor.clone()),
            }
        }
        derived.sort();
        derived.dedup();
        self.pass_throughs.insert(name.to_string(), derived);
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

    /// Splice `name` in front of `target`, taking exactly the `%`-prefixed `predecessors` with it.
    ///
    /// The same surgery the ownership passes perform: the redirected predecessors branch to the new
    /// block, and the new block branches on to the old target. Nothing else moves.
    fn splice_pass_through(
        blocks: &mut Vec<BodyBlock>,
        name: &str,
        target: &str,
        predecessors: &[&str],
    ) {
        for block in blocks.iter_mut() {
            if !predecessors.contains(&block.name.as_str()) {
                continue;
            }
            let typed = block.typed_mut().expect("carrier");
            typed.redirect_successor(target, name);
        }
        let at = blocks
            .iter()
            .position(|block| block.name == target)
            .expect("target block");
        let mut spliced = blk(name.trim_start_matches('%'), &format!("br label {target}"));
        spliced.name = name.to_string();
        blocks.insert(at, spliced);
    }

    /// A recorded split must answer every dominance question the way a fresh analysis of the
    /// mutated block list answers it.
    ///
    /// This is the property [`Dominators::record_pass_through`] trades a re-analysis per split for.
    /// It holds
    /// because a pass-through elides out of every path through it, and it is worth checking rather
    /// than reasoning about: the failure mode is not a crash but an ownership pass quietly choosing
    /// a different set of predecessors to redirect, which surfaces much later as a structurally
    /// invalid module.
    #[test]
    fn a_recorded_pass_through_answers_what_a_fresh_analysis_answers() {
        // A loop whose exit `%m` is entered from inside (`%body`) and from outside (`%pre`), so the
        // exit has predecessors that a split must separate -- the shape the ownership passes split.
        let mut blocks = vec![
            blk("0", "br i1 %c, label %pre, label %h"),
            blk("pre", "br label %m"),
            blk("h", "br i1 %c1, label %body, label %m"),
            blk("body", "br i1 %c2, label %h, label %m"),
            blk("m", "ret void"),
        ];
        let mut dominance = block_dominators(&blocks);

        // Two splits in a row, the second carrying the first's pass-through: the recursive case.
        splice_pass_through(&mut blocks, "%s0", "%m", &["%h", "%body"]);
        dominance.record_pass_through("%s0", &["%h".to_string(), "%body".to_string()]);
        splice_pass_through(&mut blocks, "%s1", "%m", &["%s0", "%pre"]);
        dominance.record_pass_through("%s1", &["%s0".to_string(), "%pre".to_string()]);

        let cfg = Cfg::from_blocks(&blocks).expect("non-empty");
        let names = reachable_names(&cfg);
        assert!(names.contains(&"%s0".to_string()) && names.contains(&"%s1".to_string()));
        let originals = ["%0", "%pre", "%h", "%body", "%m"];
        for dominator in originals {
            for node in &names {
                assert_eq!(
                    dominance.dominates(dominator, node),
                    oracle_dominates(&cfg, dominator, node),
                    "recorded splits disagree with the split graph: does {dominator} dominate {node}?"
                );
            }
        }
    }

    /// An unreachable pass-through is dominated by nothing, matching an unreachable original block.
    #[test]
    fn a_pass_through_with_no_predecessors_is_dominated_by_nothing() {
        let blocks = vec![blk("0", "br label %a"), blk("a", "ret void")];
        let mut dominance = block_dominators(&blocks);
        dominance.record_pass_through("%s0", &[]);
        assert!(!dominance.dominates("%0", "%s0"));
        assert!(dominance.dominates("%s0", "%s0"));
    }
}
