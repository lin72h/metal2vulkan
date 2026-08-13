//! Source-CFG dominator tree + natural-loop nesting forest.
//!
//! This is the analysis foundation for a structured-by-construction CFG emission (the R2 north-star:
//! "compute dominators + natural loops on the source CFG up front and emit OpSelectionMerge/
//! OpLoopMerge structured-by-construction"). It is **pure analysis** over the `BodyBlock` source CFG
//! — it does not modify blocks or change emission. It exists so the emitter can identify loop
//! headers, their bodies/latches, their nesting, and their exit (merge) blocks from the actual CFG,
//! instead of the order-and-heuristic `infer_loop_merges` path that mis-handles nested loops (e.g.
//! a previously observed module shape, where the inner loop's merge coincides with the outer loop's continue — see
//! [[metal2vulkan-native-emitter]]).
//!
//! Dominators use the Cooper-Harvey-Kennedy iterative algorithm over reverse-postorder; natural
//! loops are the standard back-edge (`u -> v` where `v` dominates `u`) closure.

use super::blocks::block_successors;
use super::graph::{compute_idom, reverse_postorder, Cfg, Dominators};
use super::BodyBlock;
use std::collections::{HashMap, HashSet};

/// One natural loop discovered in the source CFG.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::native) struct NaturalLoop {
    /// The loop header (the back-edge target; dominates the whole body).
    pub(in crate::native) header: String,
    /// All blocks in the loop (header + every block that can reach a latch without leaving via the
    /// header).
    pub(in crate::native) body: Vec<String>,
    /// Back-edge sources (latches) targeting the header.
    pub(in crate::native) latches: Vec<String>,
    /// Blocks outside the loop that are branch targets of in-loop blocks — the structured merge
    /// candidates. Empty for an infinite loop.
    pub(in crate::native) exits: Vec<String>,
    /// Enclosing loop header, if this loop is nested. `None` for a top-level loop.
    pub(in crate::native) parent: Option<String>,
}

/// The loop nesting forest of a function's source CFG, plus the dominator relation used to build it.
#[derive(Clone, Debug, Default)]
pub(in crate::native) struct LoopForest {
    /// Natural loops, each keyed by header.
    pub(in crate::native) loops: Vec<NaturalLoop>,
    /// Immediate-dominator relation of the source CFG (from [`super::graph`]).
    doms: Dominators,
}

impl LoopForest {
    /// `true` if `dominator` dominates `node` (walking the idom chain).
    pub(in crate::native) fn dominates(&self, dominator: &str, node: &str) -> bool {
        self.doms.dominates(dominator, node)
    }

    /// The natural loop headed by `header`, if any.
    pub(in crate::native) fn loop_for_header(&self, header: &str) -> Option<&NaturalLoop> {
        self.loops.iter().find(|l| l.header == header)
    }

    /// Immediate dominator of `node` (the entry, and any node mapping to itself, returns `None`).
    pub(in crate::native) fn idom(&self, node: &str) -> Option<&str> {
        self.doms.idom(node)
    }

    /// Dominance-correct loop-merge assignment — the structured `OpLoopMerge` plan a
    /// structured-by-construction emission consumes (the eventual replacement for the order-heuristic
    /// `infer_loop_merges`). For each loop the plan records its single continue (latch) and single
    /// merge (exit) when the loop is *directly structurable*, and otherwise flags why it needs CFG
    /// restructuring (node-splitting) first. This is the planning layer of R2; it does not change
    /// emission on its own.
    pub(in crate::native) fn structured_plan(&self) -> Vec<LoopPlan> {
        self.structured_plan_ignoring_exits(&HashSet::new())
    }

    /// Build the same loop plan while excluding proven terminal targets from merge selection.
    ///
    /// A branch to a bare `unreachable` terminates that invocation inside the construct; it is not a
    /// continuation destination that needs to compete for `OpLoopMerge`. Callers retain the raw exit
    /// inventory on [`NaturalLoop`] and opt into this projection only after proving the ignored block
    /// is terminal. If every raw exit is ignored, keep the raw set: a loop with only terminal exits is
    /// not the same thing as a genuinely non-terminating loop and needs separate construction.
    pub(in crate::native) fn structured_plan_ignoring_exits(
        &self,
        ignored: &HashSet<String>,
    ) -> Vec<LoopPlan> {
        // Every block that is some loop's latch (a continue target).
        let all_latches: HashSet<&str> = self
            .loops
            .iter()
            .flat_map(|l| l.latches.iter().map(String::as_str))
            .collect();
        self.loops
            .iter()
            .map(|l| {
                let live_exits = l
                    .exits
                    .iter()
                    .filter(|exit| !ignored.contains(*exit))
                    .cloned()
                    .collect::<Vec<_>>();
                let exits = if live_exits.is_empty() {
                    &l.exits
                } else {
                    &live_exits
                };
                let mut restructure = Vec::new();
                if l.latches.len() > 1 {
                    restructure.push(Restructure::MultipleLatches);
                }
                if exits.len() > 1 {
                    restructure.push(Restructure::MultipleExits);
                }
                // A single exit that is also another loop's continue (the 2647a6f3 overlap): the
                // structured merge can't be that block; it must be split.
                if let [exit] = exits.as_slice() {
                    if all_latches.contains(exit.as_str()) {
                        restructure.push(Restructure::MergeIsEnclosingContinue);
                    }
                }
                if exits.is_empty() {
                    restructure.push(Restructure::NoExit);
                }
                LoopPlan {
                    header: l.header.clone(),
                    continue_block: l.latches.first().cloned(),
                    merge_block: exits.first().cloned(),
                    restructure,
                }
            })
            .collect()
    }
}

/// The structured `OpLoopMerge` plan for one loop. `restructure` is empty when the loop can be
/// emitted directly (single latch, single exit, exit not shared); otherwise it lists the CFG
/// transforms the emission must apply first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::native) struct LoopPlan {
    pub(in crate::native) header: String,
    /// The continue target (latch); `None` only for a degenerate loop with no latch.
    pub(in crate::native) continue_block: Option<String>,
    /// The merge block (exit); `None` for an infinite loop.
    pub(in crate::native) merge_block: Option<String>,
    /// Why this loop is not directly structurable (empty = directly structurable).
    pub(in crate::native) restructure: Vec<Restructure>,
}

/// A CFG transform a loop needs before it can carry a valid `OpLoopMerge`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::native) enum Restructure {
    /// More than one back-edge: needs a single synthesized latch.
    MultipleLatches,
    /// More than one exit: needs a single synthesized merge.
    MultipleExits,
    /// The (single) merge is also an enclosing loop's continue (`2647a6f3`): needs a split merge.
    MergeIsEnclosingContinue,
    /// No exit (infinite loop): needs a synthesized unreachable merge.
    NoExit,
}

/// Build the dominator tree + natural-loop nesting forest from a function's source blocks. `blocks`
/// must be in program order with `blocks[0]` the entry.
pub(in crate::native) fn analyze(blocks: &[BodyBlock]) -> LoopForest {
    let cfg = match Cfg::from_blocks(blocks) {
        Some(cfg) => cfg,
        None => return LoopForest::default(),
    };
    let doms = cfg.dominators();

    let mut forest = LoopForest {
        loops: Vec::new(),
        doms,
    };

    // Back-edges: u -> v with v dominating u. Group latches by header v.
    let mut latches_by_header: HashMap<String, Vec<String>> = HashMap::new();
    for b in blocks {
        for t in cfg.successors.get(&b.name).into_iter().flatten() {
            if cfg.contains(t) && forest.dominates(t, &b.name) {
                latches_by_header
                    .entry(t.clone())
                    .or_default()
                    .push(b.name.clone());
            }
        }
    }

    let mut loops: Vec<NaturalLoop> = latches_by_header
        .into_iter()
        .map(|(header, mut latches)| {
            latches.sort();
            latches.dedup();
            let body = natural_loop_body(&header, &latches, &cfg.predecessors);
            let body_set: HashSet<&str> = body.iter().map(String::as_str).collect();
            let mut exits = Vec::new();
            for n in &body {
                for t in cfg.successors.get(n).into_iter().flatten() {
                    if cfg.contains(t) && !body_set.contains(t.as_str()) {
                        exits.push(t.clone());
                    }
                }
            }
            exits.sort();
            exits.dedup();
            NaturalLoop {
                header,
                body,
                latches,
                exits,
                parent: None,
            }
        })
        .collect();

    // Nesting: parent of loop A = the loop B (B != A) whose body strictly contains A's body and is
    // the smallest such. (Reducible CFGs give a proper forest; ties can't occur for natural loops.)
    let bodies: Vec<HashSet<String>> = loops
        .iter()
        .map(|l| l.body.iter().cloned().collect())
        .collect();
    let headers: Vec<String> = loops.iter().map(|l| l.header.clone()).collect();
    let parents: Vec<Option<String>> = (0..loops.len())
        .map(|i| {
            let mut best: Option<usize> = None;
            for j in 0..loops.len() {
                if i == j || headers[i] == headers[j] {
                    continue;
                }
                // B (j) strictly contains A (i)?
                if bodies[j].contains(&headers[i])
                    && bodies[i].is_subset(&bodies[j])
                    && bodies[i].len() < bodies[j].len()
                {
                    match best {
                        Some(b) if bodies[b].len() <= bodies[j].len() => {}
                        _ => best = Some(j),
                    }
                }
            }
            best.map(|j| headers[j].clone())
        })
        .collect();
    for (i, parent) in parents.into_iter().enumerate() {
        loops[i].parent = parent;
    }

    loops.sort_by(|a, b| a.header.cmp(&b.header));
    forest.loops = loops;
    forest
}

/// Recompute dominance while retaining a previously proven natural-loop forest.
///
/// This is for edge-splitting rewrites that only redirect terminal destinations and append acyclic
/// leaf/pass-through blocks. Such rewrites cannot create or remove a back-edge, change a loop body,
/// or change loop nesting, but their new blocks still need current dominance. Avoiding natural-loop
/// rediscovery is material for generated functions with hundreds of nested/overlapping loop records.
pub(in crate::native) fn analyze_reusing_natural_loops(
    blocks: &[BodyBlock],
    loops: &[NaturalLoop],
) -> LoopForest {
    let Some(cfg) = Cfg::from_blocks(blocks) else {
        return LoopForest::default();
    };
    LoopForest {
        loops: loops.to_vec(),
        doms: cfg.dominators(),
    }
}

/// Natural loop of a back-edge set: header plus every block that can reach a latch without passing
/// back through the header.
fn natural_loop_body(
    header: &str,
    latches: &[String],
    predecessors: &HashMap<String, Vec<String>>,
) -> Vec<String> {
    let mut body: HashSet<String> = HashSet::from([header.to_string()]);
    let mut stack: Vec<String> = latches.to_vec();
    for l in latches {
        body.insert(l.clone());
    }
    while let Some(n) = stack.pop() {
        if n == header {
            continue;
        }
        for pred in predecessors.get(&n).into_iter().flatten() {
            if body.insert(pred.clone()) {
                stack.push(pred.clone());
            }
        }
    }
    let mut body: Vec<String> = body.into_iter().collect();
    body.sort();
    body
}

/// An irreducible region of the source CFG: a strongly-connected region that forms a cycle but has
/// more than one entry block (a multi-entry cycle). It has no single dominating header to carry an
/// `OpLoopMerge`, so the dominance loop forest is blind to it (no dominance back-edge targets it) and
/// the post-hoc repair cannot structure it; resolving one would need node-splitting / controlled
/// duplication to give the region a single entry. `entries` (sorted) are the region's entry blocks (a
/// predecessor outside the region, or the function entry); the secondary entries are the split
/// candidates. This detector is kept as a guard, but MEASURED ACROSS BROAD PRIVATE REGRESSION SETS the population is
/// EMPTY: `historical irreducible-region probes` finds 0 irreducible regions over all 16,071
/// frontier + banked rows (after `lower_unstructured_switches`). So node-splitting is NOT the lever for
/// the cfg frontier — those failures are all on REDUCIBLE CFGs (cost-budget repair blow-ups and the
/// `selection:cond-phi-shared/*` merge-synthesis rejects `structured_reject_reason` reports).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::native) struct IrreducibleRegion {
    pub(in crate::native) nodes: Vec<String>,
    pub(in crate::native) entries: Vec<String>,
}

/// Detect irreducible regions of a function's source CFG: maximal strongly-connected components that
/// form a cycle but have more than one entry. Returns them sorted by first node; an empty result
/// means the reachable CFG is reducible. `blocks[0]` must be the entry. Uses Kosaraju's two-pass SCC
/// (iterative, so the thousand-block MPS kernels do not blow the stack).
pub(in crate::native) fn irreducible_regions(blocks: &[BodyBlock]) -> Vec<IrreducibleRegion> {
    let cfg = match Cfg::from_blocks(blocks) {
        Some(cfg) => cfg,
        None => return Vec::new(),
    };
    // Real forward successors (already filtered to existing blocks) + the subgraph
    // reachable from entry (unreachable blocks are a separate spirv-val rule).
    let succ = &cfg.successors;
    let reachable = cfg.reachable_from(&cfg.entry);

    // Predecessors over the reachable subgraph (for the reverse-graph pass + entry detection).
    let mut preds: HashMap<String, Vec<String>> = HashMap::new();
    for n in &reachable {
        for t in succ.get(n).into_iter().flatten() {
            if reachable.contains(t) {
                preds.entry(t.clone()).or_default().push(n.clone());
            }
        }
    }

    // Kosaraju pass 1: iterative DFS over the forward graph, recording finish order.
    let mut visited: HashSet<String> = HashSet::new();
    let mut finish_order: Vec<String> = Vec::new();
    for start in blocks
        .iter()
        .map(|b| &b.name)
        .filter(|n| reachable.contains(*n))
    {
        if visited.contains(start) {
            continue;
        }
        let mut dfs: Vec<(String, usize)> = vec![(start.clone(), 0)];
        visited.insert(start.clone());
        while let Some((node, ci)) = dfs.last().cloned() {
            let next = succ
                .get(&node)
                .and_then(|s| s.iter().filter(|t| reachable.contains(*t)).nth(ci));
            match next {
                Some(child) => {
                    dfs.last_mut().unwrap().1 += 1;
                    if visited.insert(child.clone()) {
                        dfs.push((child.clone(), 0));
                    }
                }
                None => {
                    finish_order.push(node);
                    dfs.pop();
                }
            }
        }
    }

    // Kosaraju pass 2: iterative DFS over the reverse graph in reverse finish order; each tree is one
    // SCC.
    let mut assigned: HashSet<String> = HashSet::new();
    let mut regions: Vec<IrreducibleRegion> = Vec::new();
    for root in finish_order.iter().rev() {
        if assigned.contains(root) {
            continue;
        }
        let mut scc: Vec<String> = Vec::new();
        let mut st = vec![root.clone()];
        assigned.insert(root.clone());
        while let Some(n) = st.pop() {
            scc.push(n.clone());
            for p in preds.get(&n).into_iter().flatten() {
                if assigned.insert(p.clone()) {
                    st.push(p.clone());
                }
            }
        }

        // Is this SCC a cycle? (>1 node, or a single node with a self-edge.)
        let scc_set: HashSet<&str> = scc.iter().map(String::as_str).collect();
        let is_cycle = scc.len() > 1
            || succ
                .get(&scc[0])
                .is_some_and(|s| s.iter().any(|t| t == &scc[0]));
        if !is_cycle {
            continue;
        }

        // Entry blocks: an SCC node reached from outside the SCC, or the function entry itself.
        let mut entries: Vec<String> = scc
            .iter()
            .filter(|n| {
                **n == cfg.entry
                    || preds
                        .get(*n)
                        .into_iter()
                        .flatten()
                        .any(|p| !scc_set.contains(p.as_str()))
            })
            .cloned()
            .collect();
        if entries.len() > 1 {
            entries.sort();
            let mut nodes = scc;
            nodes.sort();
            regions.push(IrreducibleRegion { nodes, entries });
        }
    }
    regions.sort_by(|a, b| a.nodes.first().cmp(&b.nodes.first()));
    regions
}

/// Sentinel name for the synthesized unique exit used to root the post-dominator computation when a
/// function has several return/unreachable blocks. Cannot collide with an AIR label (`%`-prefixed).
const VIRTUAL_EXIT: &str = "@@virtual_exit";

/// Immediate post-dominators of a function's source CFG, keyed `block -> ipostdom(block)`. The
/// post-dominator tree is the dominator tree of the reverse CFG rooted at a virtual exit that all
/// real exit blocks (no real successor: `ret`/`unreachable`) flow into. A block whose ipostdom is the
/// virtual exit (it post-dominates to function end) is omitted, as is the virtual exit itself.
/// Returns an empty map if there is no exit block (e.g. a function that is one infinite loop).
///
/// This is the analysis half R2's structured-by-construction emission needs for SELECTIONS: the
/// structured `OpSelectionMerge` of a conditional/switch header in a reducible CFG is its immediate
/// post-dominator (where its arms reconverge) — see [`selection_merges`].
pub(in crate::native) fn post_idom(blocks: &[BodyBlock]) -> HashMap<String, String> {
    post_idom_cut(blocks, &HashSet::new())
}

/// [`post_idom`] over a CFG with a set of `(from, to)` edges CUT — treated as if `from` had no such
/// successor. Cutting a loop's in-body → merge edges (structured breaks) models the loop body as the
/// region within which an in-loop selection reconverges, so a guarded-break selection takes its merge
/// from the non-break arm instead of the loop merge the break otherwise makes its post-dominator. See
/// [`break_aware_selection_merges`].
fn post_idom_cut(blocks: &[BodyBlock], cut: &HashSet<(String, String)>) -> HashMap<String, String> {
    let cfg = match Cfg::from_blocks(blocks) {
        Some(cfg) => cfg,
        None => return HashMap::new(),
    };
    // Forward successors with the cut edges removed.
    let fsucc: HashMap<String, Vec<String>> = cfg
        .successors
        .iter()
        .map(|(n, ts)| {
            let kept: Vec<String> = ts
                .iter()
                .filter(|t| !cut.contains(&(n.clone(), (*t).clone())))
                .cloned()
                .collect();
            (n.clone(), kept)
        })
        .collect();

    // Reachability on the CUT graph: a block reachable only via a cut edge drops out (so a fully-cut
    // block — e.g. a do-while latch whose {break, continue} arms are both cut — becomes a terminal).
    let mut reachable: HashSet<String> = HashSet::new();
    let mut stack = vec![cfg.entry.clone()];
    reachable.insert(cfg.entry.clone());
    while let Some(n) = stack.pop() {
        for t in fsucc.get(&n).into_iter().flatten() {
            if reachable.insert(t.clone()) {
                stack.push(t.clone());
            }
        }
    }

    // Exit blocks = reachable blocks with no real successor in the cut graph; they flow into the
    // virtual exit.
    let exits: Vec<String> = reachable
        .iter()
        .filter(|n| fsucc.get(*n).map(|s| s.is_empty()).unwrap_or(true))
        .cloned()
        .collect();
    if exits.is_empty() {
        return HashMap::new();
    }

    // Reverse-CFG successors (predecessors in the forward cut CFG), with the virtual exit pointing back
    // to every real exit block. The reverse graph is rooted at the virtual exit.
    let mut rsucc: HashMap<String, Vec<String>> = HashMap::new();
    rsucc.insert(VIRTUAL_EXIT.to_string(), exits.clone());
    for n in &reachable {
        for t in fsucc.get(n).into_iter().flatten() {
            if reachable.contains(t) {
                rsucc.entry(t.clone()).or_default().push(n.clone());
            }
        }
    }
    for e in &exits {
        rsucc
            .entry(e.clone())
            .or_default()
            .push(VIRTUAL_EXIT.to_string());
    }

    // Reverse-postorder + predecessors of the reverse graph, then CHK dominators rooted at the
    // virtual exit = post-dominators of the forward graph.
    let rnames: HashSet<&str> = reachable
        .iter()
        .map(String::as_str)
        .chain(std::iter::once(VIRTUAL_EXIT))
        .collect();
    let rpo = reverse_postorder(VIRTUAL_EXIT, &rsucc, &rnames);
    let rpo_num: HashMap<&str, usize> = rpo
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let mut rpreds: HashMap<String, Vec<String>> = HashMap::new();
    for (n, ts) in &rsucc {
        for t in ts {
            if rnames.contains(t.as_str()) {
                rpreds.entry(t.clone()).or_default().push(n.clone());
            }
        }
    }
    let ridom = compute_idom(VIRTUAL_EXIT, &rpreds, &rpo, &rpo_num);

    ridom
        .into_iter()
        .filter(|(n, d)| n != VIRTUAL_EXIT && d != VIRTUAL_EXIT && n != d)
        .collect()
}

/// Structured `OpSelectionMerge` assignment for a reducible source CFG: `header -> merge` for every
/// block that branches more than one way (conditional/switch) and is NOT a loop header (those carry an
/// `OpLoopMerge` from the loop forest instead). The merge is the header's immediate post-dominator —
/// the block where its arms reconverge. Headers whose ipostdom is the function exit (no reconvergence
/// before return) are omitted. This + the loop forest's [`LoopForest::structured_plan`] are the two
/// halves of the structured plan R2 emits by construction.
pub(in crate::native) fn selection_merges(
    blocks: &[BodyBlock],
    forest: &LoopForest,
) -> HashMap<String, String> {
    // A switch's terminal `unreachable` case is a structured exit from the selection, not a second
    // reconvergence path. Switch lowering already applies this rule through `infer_switch_merges`;
    // apply the same contract to the source-CFG post-dominator analysis so the two planners cannot
    // disagree and strand an otherwise ordinary switch without an `OpSelectionMerge`.
    let unreachable_targets: HashSet<&str> = blocks
        .iter()
        .filter(|block| {
            block.typed.as_ref().is_some_and(|typed| {
                typed.insts.is_empty()
                    && matches!(
                        typed.terminator,
                        crate::native::tir::TirTerminator::Unreachable
                    )
            })
        })
        .map(|block| block.name.as_str())
        .collect();
    let mut terminal_switch_edges = HashSet::new();
    for block in blocks {
        let is_switch = block.typed.as_ref().is_some_and(|typed| {
            matches!(
                typed.terminator,
                crate::native::tir::TirTerminator::Switch { .. }
            )
        });
        if !is_switch {
            continue;
        }
        for target in block_successors(block) {
            if unreachable_targets.contains(target.as_str()) {
                terminal_switch_edges.insert((block.name.clone(), target));
            }
        }
    }
    let pidom = if terminal_switch_edges.is_empty() {
        post_idom(blocks)
    } else {
        post_idom_cut(blocks, &terminal_switch_edges)
    };
    selection_merges_from_pidom(blocks, forest, &pidom)
}

/// Break-aware [`selection_merges`]: computes each selection header's merge on a post-dominator graph
/// where every in-loop → loop-merge edge (a structured break) is cut, so a guarded-break selection
/// `if(!c) break; body` (arms = {loop-merge, in-loop}) reconverges at its non-break arm instead of the
/// loop merge the break arm would otherwise make its post-dominator. That stops such a selection from
/// CLAIMING the loop merge — the collision that makes `unique_selection_merges` redirect the do-while
/// latch's break edge and reject `branch-no-merge` (the `merge-inloop` class). `loop_merges` supplies
/// each loop's chosen merge. Used only on the reject-triggered break-aware `structured_plan` attempt, so
/// base-admitting functions stay byte-identical.
pub(in crate::native) fn break_aware_selection_merges(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    loop_merges: &HashMap<String, super::LoopMergeInfo>,
) -> HashMap<String, String> {
    let cfg = match Cfg::from_blocks(blocks) {
        Some(cfg) => cfg,
        None => return HashMap::new(),
    };
    // Cut a loop's GUARDED-BREAK edges to its merge block — a mid-body conditional `if(!c) break; body`
    // whose break arm otherwise pollutes the selection's post-dominator up to the loop merge. Keep the
    // loop's STRUCTURAL exit tests (the do-while latch `{merge, continue}` and the while header's
    // top-test), identified as any in-loop → merge edge whose source also branches to the loop's
    // continue target or back to the header. Keeping ≥1 exit leaves post-dominance defined; cutting the
    // guarded break moves the selection's merge to its non-break arm (the reconvergence the break hid).
    let mut cut: HashSet<(String, String)> = HashSet::new();
    for l in &forest.loops {
        let Some(info) = loop_merges.get(&l.header) else {
            continue;
        };
        for n in &l.body {
            // The header's own exit is a structural exit; keep it (the header is excluded from selection
            // merges regardless, but its edge keeps the loop exiting under the cut).
            if n == &l.header {
                continue;
            }
            let succs: Vec<&str> = cfg
                .successors
                .get(n)
                .into_iter()
                .flatten()
                .map(String::as_str)
                .collect();
            if !succs.iter().any(|s| *s == info.merge) {
                continue;
            }
            let is_structural_exit = succs
                .iter()
                .any(|s| *s == info.continue_target || *s == l.header);
            if is_structural_exit {
                continue;
            }
            cut.insert((n.clone(), info.merge.clone()));
        }
    }
    selection_merges_from_pidom(blocks, forest, &post_idom_cut(blocks, &cut))
}

/// Shared body of [`selection_merges`] / [`break_aware_selection_merges`]: assign each non-loop-header
/// block with ≥2 distinct successors (a conditional/switch header) the merge given by `pidom` (its
/// immediate post-dominator in whichever — plain or break-cut — post-dominator graph the caller built).
fn selection_merges_from_pidom(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    pidom: &HashMap<String, String>,
) -> HashMap<String, String> {
    let loop_headers: HashSet<&str> = forest.loops.iter().map(|l| l.header.as_str()).collect();
    let block_names: HashSet<&str> = blocks.iter().map(|b| b.name.as_str()).collect();
    let mut merges = HashMap::new();
    for b in blocks {
        if loop_headers.contains(b.name.as_str()) {
            continue;
        }
        // A conditional/switch header has >= 2 distinct real successors.
        let distinct: HashSet<String> = block_successors(b)
            .into_iter()
            .filter(|successor| block_names.contains(successor.as_str()))
            .collect();
        if distinct.len() < 2 {
            continue;
        }
        if let Some(merge) = pidom.get(&b.name) {
            merges.insert(b.name.clone(), merge.clone());
        }
    }
    merges
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mirrors `split_body_blocks`: block names carry the `%` prefix; the terminator lowers to the carrier.
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

    #[test]
    fn single_self_loop() {
        // entry -> h ; h -> h (back) / exit
        let blocks = vec![
            blk("0", "br label %h"),
            blk("h", "br i1 %c, label %h, label %x"),
            blk("x", "ret void"),
        ];
        let f = analyze(&blocks);
        assert_eq!(f.loops.len(), 1);
        let l = &f.loops[0];
        assert_eq!(l.header, "%h");
        assert_eq!(l.latches, vec!["%h".to_string()]);
        assert_eq!(l.exits, vec!["%x".to_string()]);
        assert_eq!(l.parent, None);
    }

    #[test]
    fn simple_loop_with_body() {
        // 0 -> h -> b -> h (back); h -> x
        let blocks = vec![
            blk("0", "br label %h"),
            blk("h", "br i1 %c, label %b, label %x"),
            blk("b", "br label %h"),
            blk("x", "ret void"),
        ];
        let f = analyze(&blocks);
        assert_eq!(f.loops.len(), 1);
        let l = f.loop_for_header("%h").unwrap();
        assert_eq!(l.latches, vec!["%b".to_string()]);
        assert!(l.body.contains(&"%h".to_string()) && l.body.contains(&"%b".to_string()));
        assert_eq!(l.exits, vec!["%x".to_string()]);
    }

    #[test]
    fn dominance_is_correct() {
        let blocks = vec![
            blk("0", "br i1 %c, label %a, label %b"),
            blk("a", "br label %m"),
            blk("b", "br label %m"),
            blk("m", "ret void"),
        ];
        let f = analyze(&blocks);
        assert!(f.dominates("%0", "%m"));
        assert!(f.dominates("%0", "%a"));
        assert!(!f.dominates("%a", "%m")); // m reachable via b, so a does not dominate m
        assert!(f.dominates("%m", "%m"));
    }

    #[test]
    fn nested_loops_form_a_forest() {
        // outer header H, inner header G nested inside.
        // 0 -> H ; H -> G ; G -> body -> G (inner back) / G -> L ; L -> H (outer back) / L -> X
        let blocks = vec![
            blk("0", "br label %H"),
            blk("H", "br label %G"),
            blk("G", "br i1 %c, label %body, label %L"),
            blk("body", "br label %G"),
            blk("L", "br i1 %d, label %H, label %X"),
            blk("X", "ret void"),
        ];
        let f = analyze(&blocks);
        assert_eq!(f.loops.len(), 2, "{:?}", f.loops);
        let outer = f.loop_for_header("%H").unwrap();
        let inner = f.loop_for_header("%G").unwrap();
        assert_eq!(outer.parent, None);
        assert_eq!(inner.parent, Some("%H".to_string()));
        // The inner loop is fully contained in the outer.
        let outer_body: HashSet<&str> = outer.body.iter().map(String::as_str).collect();
        for n in &inner.body {
            assert!(
                outer_body.contains(n.as_str()),
                "inner {n} not in outer body"
            );
        }
    }

    #[test]
    fn inner_merge_equals_outer_continue_is_detectable() {
        // The a previously observed module shape shape: inner loop's exit (merge) is the outer loop's latch
        // (continue). 0 -> H ; H -> G ; G -> b -> G (inner back) / G -> latch ;
        // latch -> H (outer back) / latch -> X. The inner loop (G) exits to `latch`, which is the
        // outer loop's back-edge source.
        let blocks = vec![
            blk("0", "br label %H"),
            blk("H", "br label %G"),
            blk("G", "br i1 %c, label %b, label %latch"),
            blk("b", "br label %G"),
            blk("latch", "br i1 %d, label %H, label %X"),
            blk("X", "ret void"),
        ];
        let f = analyze(&blocks);
        let inner = f.loop_for_header("%G").unwrap();
        let outer = f.loop_for_header("%H").unwrap();
        // Inner loop exits to `latch`; `latch` is the outer loop's latch — the overlap the structured
        // emission must split.
        assert!(inner.exits.contains(&"%latch".to_string()));
        assert!(outer.latches.contains(&"%latch".to_string()));
    }

    #[test]
    fn structured_plan_marks_simple_loop_directly_structurable() {
        let blocks = vec![
            blk("0", "br label %h"),
            blk("h", "br i1 %c, label %b, label %x"),
            blk("b", "br label %h"),
            blk("x", "ret void"),
        ];
        let plan = analyze(&blocks).structured_plan();
        assert_eq!(plan.len(), 1);
        assert!(plan[0].restructure.is_empty(), "{:?}", plan[0]);
        assert_eq!(plan[0].continue_block.as_deref(), Some("%b"));
        assert_eq!(plan[0].merge_block.as_deref(), Some("%x"));
    }

    #[test]
    fn structured_plan_flags_merge_is_enclosing_continue() {
        // 2647a6f3 shape: inner loop's merge == outer loop's continue.
        let blocks = vec![
            blk("0", "br label %H"),
            blk("H", "br label %G"),
            blk("G", "br i1 %c, label %b, label %latch"),
            blk("b", "br label %G"),
            blk("latch", "br i1 %d, label %H, label %X"),
            blk("X", "ret void"),
        ];
        let plan = analyze(&blocks).structured_plan();
        let inner = plan.iter().find(|p| p.header == "%G").unwrap();
        assert!(
            inner
                .restructure
                .contains(&Restructure::MergeIsEnclosingContinue),
            "inner loop should be flagged for split: {inner:?}"
        );
        let outer = plan.iter().find(|p| p.header == "%H").unwrap();
        assert!(
            outer.restructure.is_empty(),
            "outer loop is directly structurable: {outer:?}"
        );
    }

    #[test]
    fn structured_plan_flags_multiple_exits() {
        // A loop with two distinct exit targets needs a synthesized single merge.
        let blocks = vec![
            blk("0", "br label %h"),
            blk("h", "br i1 %c, label %body, label %x1"),
            blk("body", "br i1 %d, label %h, label %x2"),
            blk("x1", "ret void"),
            blk("x2", "ret void"),
        ];
        let plan = analyze(&blocks).structured_plan();
        let l = plan.iter().find(|p| p.header == "%h").unwrap();
        assert!(
            l.restructure.contains(&Restructure::MultipleExits),
            "two exits should be flagged: {l:?}"
        );
    }

    #[test]
    fn irreducible_back_edge_is_not_a_natural_loop() {
        // Two entries into the cycle a<->b: 0 -> a and 0 -> b; a -> b; b -> a. Neither a nor b
        // dominates the other, so no natural loop is reported (the header-dominates-latch rule fails).
        let blocks = vec![
            blk("0", "br i1 %c, label %a, label %b"),
            blk("a", "br label %b"),
            blk("b", "br i1 %d, label %a, label %x"),
            blk("x", "ret void"),
        ];
        let f = analyze(&blocks);
        assert_eq!(
            f.loops.len(),
            0,
            "irreducible cycle must not be a natural loop: {:?}",
            f.loops
        );
    }

    #[test]
    fn irreducible_regions_empty_for_reducible_loop() {
        let blocks = vec![
            blk("0", "br label %h"),
            blk("h", "br i1 %c, label %b, label %x"),
            blk("b", "br label %h"),
            blk("x", "ret void"),
        ];
        assert!(
            irreducible_regions(&blocks).is_empty(),
            "a reducible loop has no irreducible region"
        );
    }

    #[test]
    fn irreducible_regions_empty_for_nested_reducible_loops() {
        let blocks = vec![
            blk("0", "br label %H"),
            blk("H", "br label %G"),
            blk("G", "br i1 %c, label %body, label %L"),
            blk("body", "br label %G"),
            blk("L", "br i1 %d, label %H, label %X"),
            blk("X", "ret void"),
        ];
        assert!(
            irreducible_regions(&blocks).is_empty(),
            "nested reducible loops have no irreducible region"
        );
    }

    #[test]
    fn irreducible_regions_detects_multi_entry_cycle() {
        // The same 2-entry cycle the natural-loop forest is blind to: 0 -> a, 0 -> b; a <-> b.
        let blocks = vec![
            blk("0", "br i1 %c, label %a, label %b"),
            blk("a", "br label %b"),
            blk("b", "br i1 %d, label %a, label %x"),
            blk("x", "ret void"),
        ];
        let regions = irreducible_regions(&blocks);
        assert_eq!(regions.len(), 1, "{regions:?}");
        assert_eq!(regions[0].nodes, vec!["%a".to_string(), "%b".to_string()]);
        assert_eq!(
            regions[0].entries,
            vec!["%a".to_string(), "%b".to_string()],
            "both cycle nodes are entered from outside"
        );
    }

    #[test]
    fn irreducible_regions_ignores_single_entry_scc() {
        // A reducible loop whose body is a 2-node cycle entered only at the header: 0 -> h; h -> b;
        // b -> h (back) and b -> x. The {h,b} SCC has a single entry (h), so it is reducible.
        let blocks = vec![
            blk("0", "br label %h"),
            blk("h", "br label %b"),
            blk("b", "br i1 %d, label %h, label %x"),
            blk("x", "ret void"),
        ];
        assert!(
            irreducible_regions(&blocks).is_empty(),
            "single-entry SCC is reducible"
        );
    }

    #[test]
    fn selection_merge_is_immediate_post_dominator_of_diamond() {
        // if/else diamond: H -> a, b ; a -> m ; b -> m ; m -> ret. The selection merge of H is m.
        let blocks = vec![
            blk("H", "br i1 %c, label %a, label %b"),
            blk("a", "br label %m"),
            blk("b", "br label %m"),
            blk("m", "ret void"),
        ];
        let forest = analyze(&blocks);
        let merges = selection_merges(&blocks, &forest);
        assert_eq!(merges.get("%H").map(String::as_str), Some("%m"));
        // Non-branching blocks get no selection merge.
        assert!(!merges.contains_key("%a"));
    }

    #[test]
    fn selection_merge_nested_if() {
        // Nested if: outer H -> (inner G, else e) ; G -> g1,g2 -> gm ; gm -> m ; e -> m ; m -> ret.
        let blocks = vec![
            blk("H", "br i1 %c, label %G, label %e"),
            blk("G", "br i1 %d, label %g1, label %g2"),
            blk("g1", "br label %gm"),
            blk("g2", "br label %gm"),
            blk("gm", "br label %m"),
            blk("e", "br label %m"),
            blk("m", "ret void"),
        ];
        let forest = analyze(&blocks);
        let merges = selection_merges(&blocks, &forest);
        assert_eq!(merges.get("%H").map(String::as_str), Some("%m"));
        assert_eq!(merges.get("%G").map(String::as_str), Some("%gm"));
    }

    #[test]
    fn switch_merge_ignores_terminal_unreachable_arm() {
        // The default arm terminates the invocation and therefore exits the switch construct. The two
        // live cases still reconverge at m, which is the switch's structured selection merge.
        let blocks = vec![
            blk(
                "sw",
                "switch i32 %s, label %dead [ i32 0, label %a i32 1, label %b ]",
            ),
            blk("a", "br label %m"),
            blk("b", "br label %m"),
            blk("dead", "unreachable"),
            blk("m", "ret void"),
        ];
        let forest = analyze(&blocks);
        let merges = selection_merges(&blocks, &forest);
        assert_eq!(merges.get("%sw").map(String::as_str), Some("%m"));
    }

    #[test]
    fn selection_merge_skips_loop_header() {
        // A loop header's conditional (exit test) is an OpLoopMerge, not an OpSelectionMerge.
        let blocks = vec![
            blk("0", "br label %h"),
            blk("h", "br i1 %c, label %b, label %x"),
            blk("b", "br label %h"),
            blk("x", "ret void"),
        ];
        let forest = analyze(&blocks);
        let merges = selection_merges(&blocks, &forest);
        assert!(
            !merges.contains_key("%h"),
            "loop header must not get a selection merge: {merges:?}"
        );
    }

    #[test]
    fn break_aware_selection_merge_moves_guarded_break_off_loop_merge() {
        // do-while with a mid-body GUARDED break: h header; S is `if(!c) break; body`; latch is the
        // do-while bottom test {back-edge, merge}. entry -> h -> S; S -> body / m(break);
        // body -> latch; latch -> h(back) / m; m -> ret.
        let blocks = vec![
            blk("entry", "br label %h"),
            blk("h", "br label %S"),
            blk("S", "br i1 %c, label %body, label %m"),
            blk("body", "br label %latch"),
            blk("latch", "br i1 %d, label %h, label %m"),
            blk("m", "ret void"),
        ];
        let forest = analyze(&blocks);
        // The plain post-dominator makes S claim the loop merge (the break arm pollutes it).
        let plain = selection_merges(&blocks, &forest);
        assert_eq!(
            plain.get("%S").map(String::as_str),
            Some("%m"),
            "plain: {plain:?}"
        );
        // Break-aware: with the loop's merge=%m / continue=%latch, S reconverges at its non-break arm.
        let mut loop_merges = HashMap::new();
        loop_merges.insert(
            "%h".to_string(),
            super::super::LoopMergeInfo {
                merge: "%m".to_string(),
                continue_target: "%latch".to_string(),
            },
        );
        let ba = break_aware_selection_merges(&blocks, &forest, &loop_merges);
        assert_eq!(
            ba.get("%S").map(String::as_str),
            Some("%body"),
            "break-aware S should reconverge at its non-break arm: {ba:?}"
        );
        // The loop's structural exit test (latch) keeps the merge, so the loop still exits under the cut.
        assert_eq!(ba.get("%latch").map(String::as_str), Some("%m"), "{ba:?}");
    }

    #[test]
    fn post_idom_handles_multiple_returns() {
        // Two return blocks: H -> a (ret) / b (ret). With a virtual exit, H post-dominates to exit, so
        // it has no in-function ipostdom (omitted). a and b each post-dominate only themselves.
        let blocks = vec![
            blk("H", "br i1 %c, label %a, label %b"),
            blk("a", "ret void"),
            blk("b", "ret void"),
        ];
        let pidom = post_idom(&blocks);
        // H reconverges only at the (virtual) exit, so no selection merge inside the function.
        assert!(!pidom.contains_key("%H"), "{pidom:?}");
        let forest = analyze(&blocks);
        assert!(selection_merges(&blocks, &forest).is_empty());
    }
}
