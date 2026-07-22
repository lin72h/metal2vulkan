//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

/// Find nested conditional continuations that need privatization before a unique selection merge can
/// be synthesized. A candidate has all of these structural properties:
///
/// * `header` is a non-loop two-arm selection whose natural merge is not header-dominated;
/// * a header-dominated block branches to `continuation`;
/// * the continuation itself is not header-dominated (an enclosing arm also reaches it);
/// * it is an intermediate continuation, not the natural merge, and can reach that merge.
///
/// The last condition prevents cloning ordinary exits/breaks. Headers are innermost-first so a nested
/// construct claims its continuation before its enclosing construct is considered.
pub(in crate::native) fn find_deep_shared_continuations(
    blocks: &[BodyBlock],
) -> Vec<(String, String)> {
    let forest = analyze(blocks);
    let selection = selection_merges(blocks, &forest);
    let names: HashSet<&str> = blocks.iter().map(|b| b.name.as_str()).collect();
    let loop_headers: HashSet<&str> = forest.loops.iter().map(|l| l.header.as_str()).collect();

    let depth = |name: &str| {
        let mut depth = 0usize;
        let mut cur = name;
        while let Some(parent) = forest.idom(cur) {
            depth += 1;
            cur = parent;
        }
        depth
    };
    let mut headers: Vec<&BodyBlock> = blocks
        .iter()
        .filter(|b| {
            !loop_headers.contains(b.name.as_str())
                && conditional_branch_targets(b).is_some()
                && selection.contains_key(&b.name)
        })
        .collect();
    headers.sort_by_key(|b| std::cmp::Reverse(depth(&b.name)));

    let mut found = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for header in headers {
        let Some(natural) = selection.get(&header.name) else {
            continue;
        };
        if forest.dominates(&header.name, natural) {
            continue;
        }
        let reaches_natural = reverse_reachable(blocks, natural, &names);
        for block in blocks {
            if !forest.dominates(&header.name, &block.name) {
                continue;
            }
            for continuation in block_successors(block) {
                if continuation == *natural
                    || !names.contains(continuation.as_str())
                    || forest.dominates(&header.name, &continuation)
                    || !reaches_natural.contains(&continuation)
                {
                    continue;
                }
                let pair = (header.name.clone(), continuation);
                if seen.insert(pair.clone()) {
                    found.push(pair);
                }
            }
        }
    }
    found
}

/// Find a switch-case tail that is shared with another case. Each returned `case_root` is a direct
/// target of a non-loop switch; `continuation` is reached from its dominated region but is not itself
/// case-root-dominated, so another case still enters it. A non-case continuation is eligible only when
/// the natural merge is externally reachable; a direct case-root continuation is eligible regardless,
/// because entering another case is forbidden by SPIR-V's case-construct rules.
pub(in crate::native) fn find_switch_case_shared_continuations(
    blocks: &[BodyBlock],
) -> Vec<(String, String)> {
    let forest = analyze(blocks);
    let selection = selection_merges(blocks, &forest);
    let names: HashSet<&str> = blocks.iter().map(|b| b.name.as_str()).collect();
    let loop_headers: HashSet<&str> = forest.loops.iter().map(|l| l.header.as_str()).collect();
    let loop_nodes: HashSet<&str> = forest
        .loops
        .iter()
        .flat_map(|l| l.body.iter().map(String::as_str))
        .collect();

    let depth = |name: &str| {
        let mut depth = 0usize;
        let mut cur = name;
        while let Some(parent) = forest.idom(cur) {
            depth += 1;
            cur = parent;
        }
        depth
    };
    let mut switches: Vec<&BodyBlock> = blocks
        .iter()
        .filter(|b| {
            !loop_headers.contains(b.name.as_str())
                && !loop_nodes.contains(b.name.as_str())
                // Switch-terminating detection via the carrier (`is_switch_block`, carrier-first with a
                // line fallback) instead of re-lexing `.lines.last()` — byte-identical by construction.
                && crate::native::cfg::structured_emit::is_switch_block(b)
                && selection.contains_key(&b.name)
        })
        .collect();
    switches.sort_by_key(|b| std::cmp::Reverse(depth(&b.name)));

    let mut found = Vec::new();
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for switch in switches {
        let Some(natural) = selection.get(&switch.name) else {
            continue;
        };
        let roots = block_successors(switch);
        let root_set: HashSet<&str> = roots.iter().map(String::as_str).collect();
        let merge_is_external = !forest.dominates(&switch.name, natural);
        let reaches_natural = reverse_reachable(blocks, natural, &names);
        for case_root in &roots {
            if case_root == natural
                || !names.contains(case_root.as_str())
                || !forest.dominates(&switch.name, case_root)
            {
                continue;
            }
            for block in blocks {
                if !forest.dominates(case_root, &block.name) {
                    continue;
                }
                for continuation in block_successors(block) {
                    if continuation == *natural
                        || !names.contains(continuation.as_str())
                        || forest.dominates(case_root, &continuation)
                        || !reaches_natural.contains(&continuation)
                        || (!merge_is_external && !root_set.contains(continuation.as_str()))
                    {
                        continue;
                    }
                    let pair = (case_root.clone(), continuation);
                    if seen.insert(pair.clone()) {
                        found.push(pair);
                    }
                }
            }
        }
    }
    found
}

/// The blocks that can reach `target`, including `target`, computed on the finite in-function CFG.
pub(in crate::native) fn reverse_reachable(
    blocks: &[BodyBlock],
    target: &str,
    names: &HashSet<&str>,
) -> HashSet<String> {
    let preds = predecessors(blocks);
    let mut seen = HashSet::new();
    let mut stack = vec![target.to_string()];
    while let Some(block) = stack.pop() {
        if !seen.insert(block.clone()) {
            continue;
        }
        for pred in preds.get(&block).into_iter().flatten() {
            if names.contains(pred.as_str()) && !seen.contains(pred) {
                stack.push(pred.clone());
            }
        }
    }
    seen
}

/// Find a DEFINITIVE cross-arm edge: an edge `B -> S` where `B` is dominated by one arm target `T0` of
/// a 2-arm conditional selection header `H` and `S` is dominated by the OTHER arm target `T1`. Such an
/// edge escapes `B`'s selection arm into the sibling arm (spirv-val "block B exits the selection headed
/// by H, but not via a structured exit") — the 32-row ADMIT structured-exit family the base
/// `structured_plan` self-check 2 misses (it only inspects a header's DIRECT arm successors, not a block
/// deeper in an arm branching to an ANCESTOR selection's sibling arm). Returns `(near, S)` where
/// `near = T0` is the arm target dominating `B` (the [`privatize_dominated_region`] `header`) and `S` is
/// the sibling-arm block to clone (its `arm`), or `None`. FP-free by construction: only edges whose
/// endpoints are EACH definitively dominated by a distinct arm target are reported, so a straddle/
/// irreducible block dominated by neither arm (the dead-#6 ambiguity) is never reported and a
/// spirv-val-legal selection is never flagged.
pub(in crate::native) fn find_cross_arm_edge(blocks: &[BodyBlock]) -> Option<(String, String)> {
    let forest = analyze(blocks);
    let loop_headers: HashSet<&str> = forest.loops.iter().map(|l| l.header.as_str()).collect();
    let names: HashSet<&str> = blocks.iter().map(|b| b.name.as_str()).collect();
    // Selection headers -> their two arm targets (skip loop headers — their conditional is the loop exit
    // test, and a break/continue is a legal exit, not a cross-arm jump).
    let targets: HashMap<&str, (String, String)> = blocks
        .iter()
        .filter(|h| !loop_headers.contains(h.name.as_str()))
        .filter_map(|h| {
            let (t0, t1) = conditional_branch_targets(h)?;
            if t0 == t1 || !names.contains(t0.as_str()) || !names.contains(t1.as_str()) {
                return None;
            }
            Some((h.name.as_str(), (t0, t1)))
        })
        .collect();
    // Edge-wise via idom walk (O(edges * dom-depth), NOT O(headers * blocks)): for a NON-forward edge
    // B -> S, walk B's idom chain; at each selection-header ancestor H the arm target B entered is the
    // walk's `child`, so the OTHER arm is `sibling` — if `dominates(sibling, S)`, B (arm `child`) escapes
    // into S's sibling arm. Return `(child, S)`: `child` is the near arm target = the
    // `privatize_dominated_region` `header`, `S` its `arm`.
    for b in blocks {
        for s in block_successors(b) {
            if !names.contains(s.as_str())
                || loop_headers.contains(s.as_str())
                || forest.dominates(&b.name, &s)
            {
                continue;
            }
            let mut child: &str = &b.name;
            while let Some(cur) = forest.idom(child) {
                if let Some((x, y)) = targets.get(cur) {
                    let sibling = if child == x {
                        Some(y)
                    } else if child == y {
                        Some(x)
                    } else {
                        None
                    };
                    if let Some(sib) = sibling {
                        if forest.dominates(sib, &s) {
                            return Some((child.to_string(), s));
                        }
                    }
                }
                child = cur;
            }
        }
    }
    None
}
