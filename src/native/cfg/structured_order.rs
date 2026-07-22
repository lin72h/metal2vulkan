//! R2 relooper — structured block ordering (module 1 of the structured-by-construction emitter).
//!
//! The clean rewrite of CFG handling replaces the order-heuristic merges + the 14-pass post-hoc
//! `repair_structured_merges` fixpoint with a structured-by-construction emitter. Four measured
//! disproofs (journal 2026-06-25, AIR2VK-R2-INCR3/4/5 + the bf16 finding) established that the repair
//! cannot be bypassed incrementally: it is entangled with block ordering, single-predecessor cleanup,
//! and merge placement, so any partial wiring regresses the `--banked` floor. The emitter must
//! therefore produce SPIR-V that is *already* structured-valid — which begins with emitting blocks in
//! a structured order.
//!
//! This module is that ordering. SPIR-V's structured CFG requires every block to appear after its
//! dominator and every construct's merge block to appear after the whole construct. [`structured_order`]
//! produces such a permutation by a dominator-tree preorder DFS that defers each header's merge block
//! until after its construct body. It is a pure function over the dominator/loop forest
//! ([`super::loopforest`]); it IS the active emission order — [`super::structured_emit::structured_plan`]
//! calls it (module 1 of the structured-by-construction consumer), so the order it returns is the order
//! blocks are emitted in.
//!
//! Known gap (2026-06-27): on a few MPS kernels a loop body's inner `OpSelectionMerge` block is ordered
//! BEFORE the loop header that dominates it, so a use precedes its def ("ID has not been defined", 9
//! frontier cases — `02/ab4c2598`, `04/37318cb2`). The merge is deferred at `idom(merge)`, which here
//! lands on a pre-loop block. This is NOT fixable by changing the deferral node: `idom(merge)` does not
//! dominate the construct header (the extra predecessor is a genuine non-dominating pre-loop edge into
//! the merge), the legacy repair path mis-orders it identically, and forcing the merge into the loop
//! only surfaces the real blocker — the merge is reachable from inside AND outside the loop, so it
//! cannot be that selection's merge without node duplication / splitting. The fix is the structurizer
//! node-split rewrite, not a reorder. (See `kb/metal2vulkan-conformance.md`, frontier invalid-spirv clusters.)

use super::loopforest::LoopForest;
use super::BodyBlock;
use std::collections::{HashMap, HashSet};

/// Produce a structured emission order: the block names of `blocks` permuted so that every block
/// follows its immediate dominator and every construct's merge block follows the construct body.
///
/// Drives a dominator-tree preorder DFS from the entry (`blocks[0]`). `merge_of` supplies each
/// header's construct merge block (loop exit or selection merge; `None` for non-headers). Each merge
/// is deferred at its OWN immediate dominator (its dom-tree parent) — removed from that parent's
/// dominator-children and processed LAST among them — so the whole construct body is emitted before
/// the merge. Dominator-children are otherwise visited in program order, so the emission tracks the AIR
/// order where the structure does not force a change (keeping forward references forward). Unreachable
/// blocks (no path from entry) are appended in program order.
///
/// Deferring at `idom(merge)` rather than at the header is what makes loops correct: a selection
/// merge's idom IS its header (so this is unchanged for selections), but a loop's exit (merge) is
/// frequently dominated by an in-body exit guard rather than the header. Deferring only at the header
/// would leave such a merge as an ordinary child of the guard, where it can be emitted before later
/// body blocks dominated by that guard — the `OpLoopMerge`-merge-before-dominating-body class
/// (journal `00/f927b9f7`). Deferring at the guard (its idom) makes the merge that guard's last child,
/// after the rest of the loop body.
///
/// Iterative (explicit stack) to bound stack depth on the thousand-block MPS kernels.
pub(in crate::native) fn structured_order(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    merge_of: impl Fn(&str) -> Option<String>,
) -> Vec<String> {
    structured_order_with_loop_merges_last(blocks, forest, merge_of, false)
}

/// Terminal-prefix ordering variant. A terminal retry can place a loop merge and an inner selection
/// merge under the same immediate dominator after it funnels a return/unreachable loop exit. The
/// selection must close before the enclosing loop merge; source order alone can put the synthesized
/// loop dispatch first and violate dominance. This ordering applies only to that reject-triggered
/// terminal construction, leaving the ordinary path byte-identical.
pub(in crate::native) fn structured_order_terminal(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    merge_of: impl Fn(&str) -> Option<String>,
) -> Vec<String> {
    structured_order_with_loop_merges_last(blocks, forest, merge_of, true)
}

fn structured_order_with_loop_merges_last(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    merge_of: impl Fn(&str) -> Option<String>,
    defer_loop_merges_last: bool,
) -> Vec<String> {
    let pos: HashMap<&str, usize> = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.name.as_str(), i))
        .collect();

    // Dominator-tree children of each block, in program order.
    let mut children: HashMap<String, Vec<String>> = HashMap::new();
    for b in blocks {
        if let Some(d) = forest.idom(&b.name) {
            children
                .entry(d.to_string())
                .or_default()
                .push(b.name.clone());
        }
    }
    for kids in children.values_mut() {
        kids.sort_by_key(|n| pos.get(n.as_str()).copied().unwrap_or(usize::MAX));
    }

    // Merges deferred at each dom-tree node = the construct merges whose `idom` is that node. A merge
    // is reached exactly once (it is a dom-child only of its idom), so each appears in exactly one
    // node's defer list. Sorted/deduped by program order for determinism.
    let mut defer_at: HashMap<String, Vec<String>> = HashMap::new();
    let mut loop_merges = HashSet::new();
    for b in blocks {
        if let Some(m) = merge_of(&b.name) {
            if forest.loop_for_header(&b.name).is_some() {
                loop_merges.insert(m.clone());
            }
            if let Some(d) = forest.idom(&m) {
                let slot = defer_at.entry(d.to_string()).or_default();
                if !slot.contains(&m) {
                    slot.push(m);
                }
            }
        }
    }
    for ms in defer_at.values_mut() {
        if defer_loop_merges_last {
            ms.sort_by_key(|n| {
                (
                    loop_merges.contains(n) as u8,
                    pos.get(n.as_str()).copied().unwrap_or(usize::MAX),
                )
            });
        } else {
            ms.sort_by_key(|n| pos.get(n.as_str()).copied().unwrap_or(usize::MAX));
        }
    }

    let Some(entry) = blocks.first().map(|b| b.name.clone()) else {
        return Vec::new();
    };

    let mut order = Vec::with_capacity(blocks.len());
    let mut visited = HashSet::new();
    let mut stack = vec![entry];
    while let Some(b) = stack.pop() {
        if !visited.insert(b.clone()) {
            continue;
        }
        order.push(b.clone());

        let deferred = defer_at.get(&b).cloned().unwrap_or_default();
        let mut kids = children.get(&b).cloned().unwrap_or_default();
        if !deferred.is_empty() {
            kids.retain(|c| !deferred.contains(c));
        }
        // LIFO: push the deferred merges first (bottom of the stack) so they pop after every regular
        // child and every child's subtree (the whole construct body); among several deferred merges at
        // one node, push in reverse program order so they pop in program order. Then push regular
        // children in reverse program order so they pop in program order.
        for m in deferred.into_iter().rev() {
            stack.push(m);
        }
        for c in kids.into_iter().rev() {
            stack.push(c);
        }
    }

    // Defensive: a reducible CFG reaches every block from entry, but never drop a block.
    for b in blocks {
        if !visited.contains(&b.name) {
            order.push(b.name.clone());
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::super::loopforest::analyze;
    use super::*;

    fn bb(name: &str, term: &str) -> BodyBlock {
        let name = name.to_string();
        let typed = crate::native::tir::lower_block_carrier(
            &name,
            &[term.to_string()],
            &std::collections::HashMap::new(),
        );
        BodyBlock {
            name,
            role: crate::native::cfg::BlockRole::Normal,
            typed,
        }
    }

    fn order_of(blocks: &[BodyBlock], merges: &[(&str, &str)]) -> Vec<String> {
        let forest = analyze(blocks);
        let map: HashMap<String, String> = merges
            .iter()
            .map(|(h, m)| (h.to_string(), m.to_string()))
            .collect();
        structured_order(blocks, &forest, |h| map.get(h).cloned())
    }

    fn terminal_order_of(blocks: &[BodyBlock], merges: &[(&str, &str)]) -> Vec<String> {
        let forest = analyze(blocks);
        let map: HashMap<String, String> = merges
            .iter()
            .map(|(h, m)| (h.to_string(), m.to_string()))
            .collect();
        structured_order_terminal(blocks, &forest, |h| map.get(h).cloned())
    }

    /// if/else diamond: entry branches to %a/%b which both reconverge at %m. Structured order keeps the
    /// arms (program order) inside the construct and emits the merge last.
    #[test]
    fn if_diamond_emits_merge_last() {
        let blocks = vec![
            bb("%entry", "br i1 %c, label %a, label %b"),
            bb("%a", "br label %m"),
            bb("%b", "br label %m"),
            bb("%m", "ret void"),
        ];
        let order = order_of(&blocks, &[("%entry", "%m")]);
        assert_eq!(order, vec!["%entry", "%a", "%b", "%m"]);
    }

    /// A merge placed BEFORE its arms in program order is moved after them by the structured order.
    #[test]
    fn merge_before_arms_is_reordered_after() {
        let blocks = vec![
            bb("%entry", "br i1 %c, label %a, label %b"),
            bb("%m", "ret void"),
            bb("%a", "br label %m"),
            bb("%b", "br label %m"),
        ];
        let order = order_of(&blocks, &[("%entry", "%m")]);
        let pos = |n: &str| order.iter().position(|x| x == n).unwrap();
        assert!(pos("%a") < pos("%m") && pos("%b") < pos("%m"));
        assert_eq!(order[0], "%entry");
    }

    /// Single loop: header guards body/exit, body latches back. Exit (the loop merge) is emitted after
    /// the body.
    #[test]
    fn loop_emits_exit_after_body() {
        let blocks = vec![
            bb("%entry", "br label %h"),
            bb("%h", "br i1 %c, label %body, label %exit"),
            bb("%body", "br label %h"),
            bb("%exit", "ret void"),
        ];
        let order = order_of(&blocks, &[("%h", "%exit")]);
        assert_eq!(order, vec!["%entry", "%h", "%body", "%exit"]);
    }

    /// Nested if inside the then-arm: the inner merge precedes the outer merge, and both follow their
    /// bodies.
    #[test]
    fn nested_if_orders_inner_before_outer_merge() {
        let blocks = vec![
            bb("%entry", "br i1 %c0, label %outer_then, label %outer_merge"),
            bb(
                "%outer_then",
                "br i1 %c1, label %inner_then, label %inner_merge",
            ),
            bb("%inner_then", "br label %inner_merge"),
            bb("%inner_merge", "br label %outer_merge"),
            bb("%outer_merge", "ret void"),
        ];
        let order = order_of(
            &blocks,
            &[("%entry", "%outer_merge"), ("%outer_then", "%inner_merge")],
        );
        let pos = |n: &str| order.iter().position(|x| x == n).unwrap();
        assert_eq!(order[0], "%entry");
        assert!(pos("%inner_then") < pos("%inner_merge"));
        assert!(pos("%inner_merge") < pos("%outer_merge"));
        assert!(pos("%outer_then") < pos("%inner_merge"));
    }

    /// The `00/f927b9f7` loop class: a loop whose exit (merge) is dominated by an in-body exit guard,
    /// not the header, and the exit appears BEFORE the latch in program order. Deferring the merge at
    /// its idom (the guard `%b1`) keeps it after the whole loop body; deferring only at the header
    /// would emit the exit before the latch (`OpLoopMerge` merge before a dominating body block).
    #[test]
    fn loop_exit_dominated_by_body_guard_emits_after_body() {
        let blocks = vec![
            bb("%entry", "br label %h"),
            bb("%h", "br label %b1"),
            bb("%b1", "br i1 %c, label %b2, label %exit"),
            // %exit precedes the latch %b2 in program order — the order that triggers the bug.
            bb("%exit", "ret void"),
            bb("%b2", "br label %h"),
        ];
        // %h is the loop header; its structured merge is the loop exit %exit (idom(%exit) == %b1).
        let order = order_of(&blocks, &[("%h", "%exit")]);
        let pos = |n: &str| order.iter().position(|x| x == n).unwrap();
        assert!(
            pos("%exit") > pos("%b2"),
            "loop exit must follow the latch (whole body) — got {order:?}"
        );
        assert!(
            pos("%exit") > pos("%b1"),
            "loop exit must follow its dominating guard — got {order:?}"
        );
        assert_eq!(order[0], "%entry");
    }

    /// A loop exit and an in-loop selection merge can be deferred at the same dominator. The
    /// terminal-prefix plan must close the selection before the enclosing loop exits, even when the
    /// source puts the loop exit first.
    #[test]
    fn terminal_order_closes_selection_before_shared_deferred_loop_exit() {
        let blocks = vec![
            bb("%entry", "br label %loop.header"),
            bb("%loop.header", "br label %guard"),
            bb(
                "%guard",
                "br i1 %break_now, label %loop.exit, label %selection.merge",
            ),
            bb("%loop.exit", "ret void"),
            bb("%selection.merge", "br label %loop.latch"),
            bb("%loop.latch", "br label %loop.header"),
        ];
        let merges = [
            ("%loop.header", "%loop.exit"),
            ("%guard", "%selection.merge"),
        ];
        let ordinary = order_of(&blocks, &merges);
        let terminal = terminal_order_of(&blocks, &merges);
        let pos =
            |order: &[String], name: &str| order.iter().position(|block| block == name).unwrap();
        assert!(pos(&ordinary, "%loop.exit") < pos(&ordinary, "%selection.merge"));
        assert!(pos(&terminal, "%selection.merge") < pos(&terminal, "%loop.exit"));
    }

    /// Every block appears exactly once and the entry is first.
    #[test]
    fn order_is_a_permutation() {
        let blocks = vec![
            bb("%entry", "br i1 %c, label %a, label %b"),
            bb("%a", "br label %m"),
            bb("%b", "br label %m"),
            bb("%m", "ret void"),
        ];
        let order = order_of(&blocks, &[("%entry", "%m")]);
        let mut sorted = order.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), blocks.len());
        assert_eq!(order[0], "%entry");
    }
}
