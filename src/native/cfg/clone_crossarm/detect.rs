//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

/// Predecessor map: block name -> the blocks that branch to it.
pub(in crate::native) fn predecessors(blocks: &[BodyBlock]) -> HashMap<String, Vec<String>> {
    let mut preds: HashMap<String, Vec<String>> = HashMap::new();
    for b in blocks {
        for s in block_successors(b) {
            preds.entry(s).or_default().push(b.name.clone());
        }
    }
    preds
}

/// Find the first cross-arm violation: a multi-way header `H` (not a loop header) with an immediate
/// arm successor `A` that `H` does not dominate and that is not `H`'s natural merge or a loop
/// break/continue target. `A` is the shared block to privatize for `H`'s entry.
pub(in crate::native) fn find_cross_arm(blocks: &[BodyBlock]) -> Option<(String, String)> {
    let forest = analyze(blocks);
    let pidom = post_idom(blocks);
    let loop_headers: HashSet<&str> = forest.loops.iter().map(|l| l.header.as_str()).collect();
    // A loop's exit (merge) blocks and its header (continue) are legal structured break/continue
    // targets — but ONLY for an arm whose header sits INSIDE that loop. An arm edge from a header
    // OUTSIDE the loop to its merge is a sibling-arm cross-jump, not a break (the same enclosure rule
    // `structured_emit::plan_self_check_reason` enforces). So enclosure is tested per (header, target)
    // via natural-loop membership rather than a global break-target whitelist — without this, a header
    // in one arm whose arm jumps to a loop-merge living in a SIBLING arm is silently skipped and never
    // privatized (`00/a28d5623`: `%94`'s arm → `%272`, the merge of a loop in arm `%189`).
    let is_enclosing_break = |b: &str, a: &str| -> bool {
        forest.loops.iter().any(|l| {
            l.body.iter().any(|n| n == b) && (l.header == a || l.exits.iter().any(|e| e == a))
        })
    };
    let names: HashSet<&str> = blocks.iter().map(|b| b.name.as_str()).collect();
    for b in blocks {
        if loop_headers.contains(b.name.as_str()) {
            continue;
        }
        let succs = block_successors(b);
        let distinct: HashSet<&str> = succs
            .iter()
            .map(String::as_str)
            .filter(|t| names.contains(t))
            .collect();
        if distinct.len() < 2 {
            continue;
        }
        let merge = pidom.get(&b.name).map(String::as_str);
        // The first clone candidate drives the fixpoint, so preserve the source terminator's arm
        // order instead of iterating the deduplication HashSet.
        for a in succs
            .iter()
            .map(String::as_str)
            .filter(|a| distinct.contains(a))
        {
            if Some(a) == merge || is_enclosing_break(&b.name, a) {
                continue;
            }
            if loop_headers.contains(a) {
                // Privatizing a loop header as the region entry is the loop-aware case left to a
                // future increment.
                continue;
            }
            if !forest.dominates(&b.name, a) {
                return Some((b.name.clone(), a.to_string()));
            }
        }
    }
    None
}
