//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

const LOOP_HEADER_SELECTION_PREFIX: &str = "%metal2vulkan.lhsel.";

/// Refine loop-merge post-dominators that hide a real in-loop selection convergence. For a selection
/// header `H` in loop `L`, ordinary post-dominance sees `L`'s merge because some `H`-dominated paths
/// break out of `L`; SPIR-V permits those paths to leave the selection through the enclosing loop.
/// When every remaining path reaches one loop-local block `C`, `C` is the proper selection merge.
///
/// This is deliberately narrow: it only replaces an existing natural merge that is exactly the
/// enclosing loop merge, rejects nested-loop traversal and every non-role exit, and proves all arm
/// regions against the actual CFG. The caller enables it only in the final reject-triggered planning
/// tier, after the ordinary post-dominator and clone tiers have declined the function.
pub(in crate::native) fn refine_loop_exit_selection_merges(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    selection_merges: &mut HashMap<String, String>,
) {
    if blocks.len() > LOOP_EXIT_SELECTION_MAX_BLOCKS {
        return;
    }
    let headers: Vec<(String, String)> = selection_merges
        .iter()
        .map(|(header, merge)| (header.clone(), merge.clone()))
        .collect();
    for (header, natural) in headers {
        let mut enclosing: Vec<_> = forest
            .loops
            .iter()
            .filter(|loop_info| loop_info.header != header)
            .filter(|loop_info| loop_info.body.iter().any(|node| node == &header))
            .filter_map(|loop_info| {
                loop_merges
                    .get(&loop_info.header)
                    .filter(|info| info.merge == natural)
                    .map(|info| (loop_info, info))
            })
            .collect();
        // The innermost enclosing loop owns the only legal break/continue roles for this selection.
        enclosing.sort_by_key(|(loop_info, _)| loop_info.body.len());
        for (loop_info, info) in enclosing {
            if let Some(convergence) = loop_exit_selection_convergence(
                blocks,
                forest,
                loop_merges,
                loop_info,
                info,
                &header,
            ) {
                selection_merges.insert(header.clone(), convergence);
                break;
            }
        }
    }
}

/// Recover the merge of a conditional carried by the sole successor of a loop header when some
/// paths terminate instead of reaching the ordinary post-dominator. This is the structural form
/// produced when a genuine loop-header selection is lifted off the header, but the proof deliberately
/// uses CFG ownership rather than a synthesized name. A direct arm is a valid merge when every path
/// from the other arm either reaches it or exits through the enclosing loop/return/unreachable.
///
/// The search considers only the two direct arms and traverses each edge at most once per candidate,
/// so it remains available for large generated functions where the general convergence search is
/// intentionally bounded.
pub(in crate::native) fn refine_loop_entry_terminal_selection_merges(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    selection_merges: &mut HashMap<String, String>,
) {
    let by_name = blocks
        .iter()
        .map(|block| (block.name.as_str(), block))
        .collect::<HashMap<_, _>>();
    for loop_info in &forest.loops {
        let Some(info) = loop_merges.get(&loop_info.header) else {
            continue;
        };
        let Some(loop_header) = by_name.get(loop_info.header.as_str()) else {
            continue;
        };
        let successors = block_successors(loop_header);
        let [selection_header] = successors.as_slice() else {
            continue;
        };
        if selection_merges.contains_key(selection_header) {
            continue;
        }
        let Some(selection_block) = by_name.get(selection_header.as_str()) else {
            continue;
        };
        let Some((left, right)) = conditional_branch_targets(selection_block) else {
            continue;
        };
        let loop_body = loop_info
            .body
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        for (candidate, other) in [(&left, &right), (&right, &left)] {
            if !loop_body.contains(candidate.as_str())
                || candidate == &loop_info.header
                || candidate == &info.merge
                || candidate == &info.continue_target
            {
                continue;
            }
            if loop_entry_arm_reaches_merge_or_structured_exit(
                blocks, &by_name, &loop_body, loop_info, info, other, candidate,
            ) {
                selection_merges.insert(selection_header.clone(), candidate.clone());
                if crate::env_vars::spi_why() {
                    eprintln!(
                        "[spi-why]   loop-entry-terminal header={} owner={} merge={}",
                        selection_header, loop_info.header, candidate,
                    );
                }
                break;
            }
        }
        if !selection_merges.contains_key(selection_header) {
            if let Some(candidate) = loop_exit_selection_convergence(
                blocks,
                forest,
                loop_merges,
                loop_info,
                info,
                selection_header,
            ) {
                selection_merges.insert(selection_header.clone(), candidate.clone());
                if crate::env_vars::spi_why() {
                    eprintln!(
                        "[spi-why]   loop-entry-terminal header={} owner={} merge={}",
                        selection_header, loop_info.header, candidate,
                    );
                }
            }
        }
    }
}

/// Recover terminal-aware merges for ordinary selections nested inside loops. Only headers absent
/// from both ordinary post-dominance and the explicit terminal-owner map are considered. For each,
/// the innermost enclosing loop supplies the legal break/continue/back-edge exits, while
/// [`loop_exit_selection_convergence`] proves every remaining live path reaches the candidate.
pub(in crate::native) fn refine_nested_terminal_selection_merges(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    forced_terminal_merges: &HashMap<String, String>,
    selection_merges: &mut HashMap<String, String>,
) {
    let by_name = blocks
        .iter()
        .map(|block| (block.name.as_str(), block))
        .collect::<HashMap<_, _>>();
    let mut headers = blocks
        .iter()
        .filter(|block| conditional_branch_targets(block).is_some() || is_switch_block(block))
        .filter(|block| !selection_merges.contains_key(&block.name))
        .filter(|block| !forced_terminal_merges.contains_key(&block.name))
        .map(|block| block.name.clone())
        .collect::<Vec<_>>();
    headers.sort();
    for header in headers {
        let direct_enclosing_loop_exit = |target: &str| {
            forest.loops.iter().any(|candidate_loop| {
                candidate_loop.header != header
                    && candidate_loop.body.iter().any(|node| node == &header)
                    && (candidate_loop.header == target
                        || loop_merges.get(&candidate_loop.header).is_some_and(|role| {
                            role.merge == target || role.continue_target == target
                        }))
            })
        };
        let mut enclosing = forest
            .loops
            .iter()
            .filter(|loop_info| loop_info.header != header)
            .filter(|loop_info| loop_info.body.iter().any(|node| node == &header))
            .filter_map(|loop_info| {
                loop_merges
                    .get(&loop_info.header)
                    .map(|info| (loop_info, info))
            })
            .collect::<Vec<_>>();
        enclosing.sort_by_key(|(loop_info, _)| loop_info.body.len());
        if enclosing.is_empty() {
            // A loop-free guard can return through the same block used as a later loop's exit. The
            // convergence candidate on its live arm is real, but recording it directly leaves the
            // shared return inside two structural owners. The terminal planner owns the required
            // edge split: it first gives the loop a private return, then gives this guard a private
            // merge. Decline here so that ownership-producing path runs instead of admitting an
            // unmodified graph that spirv-val rejects as re-entering a selection construct.
            if let Some(candidate) = terminal_exit_convergence(blocks, forest, &header) {
                if !terminal_arm_enters_loop_with_shared_exit(blocks, forest, &header, &candidate) {
                    selection_merges.insert(header.clone(), candidate.clone());
                    if crate::env_vars::spi_why() {
                        eprintln!(
                            "[spi-why]   terminal-convergence header={} merge={}",
                            header, candidate,
                        );
                    }
                    continue;
                }
            }
        }
        for (loop_info, info) in enclosing {
            let direct = by_name
                .get(header.as_str())
                .and_then(|block| conditional_branch_targets(block))
                .and_then(|(left, right)| {
                    let loop_body = loop_info
                        .body
                        .iter()
                        .map(String::as_str)
                        .collect::<HashSet<_>>();
                    [(&left, &right), (&right, &left)]
                        .into_iter()
                        .find_map(|(candidate, other)| {
                            (loop_body.contains(candidate.as_str())
                                && candidate != &loop_info.header
                                && candidate != &info.merge
                                && candidate != &info.continue_target
                                && (direct_enclosing_loop_exit(other)
                                    || loop_entry_arm_reaches_merge_or_structured_exit(
                                        blocks, &by_name, &loop_body, loop_info, info, other,
                                        candidate,
                                    )))
                            .then(|| candidate.clone())
                        })
                });
            let candidate = direct.or_else(|| {
                loop_exit_selection_convergence(
                    blocks,
                    forest,
                    loop_merges,
                    loop_info,
                    info,
                    &header,
                )
            });
            if let Some(candidate) = candidate {
                selection_merges.insert(header.clone(), candidate.clone());
                if crate::env_vars::spi_why() {
                    eprintln!(
                        "[spi-why]   nested-loop-terminal header={} owner={} merge={}",
                        header, loop_info.header, candidate,
                    );
                }
                break;
            }
        }
        if crate::env_vars::spi_why() && !selection_merges.contains_key(&header) {
            let arms = by_name
                .get(header.as_str())
                .map(|block| block_successors(block))
                .unwrap_or_default();
            let roles = forest
                .loops
                .iter()
                .filter(|loop_info| loop_info.body.iter().any(|node| node == &header))
                .map(|loop_info| {
                    let role = loop_merges.get(&loop_info.header);
                    (
                        loop_info.header.as_str(),
                        role.map(|role| role.merge.as_str()),
                        role.map(|role| role.continue_target.as_str()),
                    )
                })
                .collect::<Vec<_>>();
            eprintln!(
                "[spi-why]   nested-loop-terminal-decline header={} arms={:?} roles={:?}",
                header, arms, roles,
            );
        }
    }
}

fn terminal_arm_enters_loop_with_shared_exit(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    header: &str,
    live_convergence: &str,
) -> bool {
    let Some((left, right)) = blocks
        .iter()
        .find(|block| block.name == header)
        .and_then(conditional_branch_targets)
    else {
        return false;
    };
    [left, right].into_iter().any(|arm| {
        terminal_exit_arm(blocks, forest, header, &arm).is_some_and(|terminal| {
            forest.loops.iter().any(|loop_info| {
                loop_info
                    .exits
                    .iter()
                    .any(|exit| exit == &terminal.return_block)
                    && block_reaches(blocks, live_convergence, &loop_info.header)
            })
        })
    })
}

fn block_reaches(blocks: &[BodyBlock], start: &str, target: &str) -> bool {
    let by_name = blocks
        .iter()
        .map(|block| (block.name.as_str(), block))
        .collect::<HashMap<_, _>>();
    let mut seen = HashSet::new();
    let mut pending = vec![start.to_string()];
    while let Some(current) = pending.pop() {
        if current == target {
            return true;
        }
        if !seen.insert(current.clone()) {
            continue;
        }
        if let Some(block) = by_name.get(current.as_str()) {
            pending.extend(block_successors(block));
        }
    }
    false
}

/// Re-run terminal convergence after construct-tree edge splits have exposed an enclosing
/// continuation that was not provable in the source CFG.
///
/// Direct terminal-guard composition deliberately works innermost-first. Once those guards own
/// private returns, an outer header can become the ordinary `H -> { nested-guard, continuation }`
/// shape whose live paths all reach `continuation`. The initial selection analysis cannot see that
/// ownership yet. Reuse the same terminal-convergence proof late, but accept only a merge that is
/// header-dominated and not already owned by a selection or loop role. Anything ambiguous remains
/// absent and therefore falls back rather than weakening structured-CFG ownership. A header that
/// already had a source-time natural merge is also excluded: failure to synthesize that merge is a
/// collision-repair problem, not evidence that it became a terminal selection.
#[cfg(test)]
pub(in crate::native) fn complete_construct_tree_terminal_convergences(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
    forced_terminal_merges: &HashMap<String, String>,
    source_selection_merges: &HashMap<String, String>,
    header_merges: &mut HashMap<String, String>,
) -> bool {
    let forest = analyze(blocks);
    let mut proposed = header_merges.clone();
    refine_nested_terminal_selection_merges(
        blocks,
        &forest,
        loop_merges,
        forced_terminal_merges,
        &mut proposed,
    );

    let loop_roles = loop_role_targets_with_passthroughs(blocks, loop_merges);
    let mut claimed = header_merges.values().cloned().collect::<HashSet<_>>();
    let mut additions = proposed
        .into_iter()
        .filter(|(header, _)| {
            !header_merges.contains_key(header) && !source_selection_merges.contains_key(header)
        })
        .collect::<Vec<_>>();
    additions.sort();

    let mut changed = false;
    for (header, merge) in additions {
        if loop_roles.contains(&merge)
            || claimed.contains(&merge)
            || !forest.dominates(&header, &merge)
        {
            if crate::env_vars::spi_why() {
                eprintln!(
                    "[spi-why]   late-terminal-convergence-decline header={} merge={} loop-role={} claimed={} dominated={}",
                    header,
                    merge,
                    loop_roles.contains(&merge),
                    claimed.contains(&merge),
                    forest.dominates(&header, &merge),
                );
            }
            continue;
        }
        if crate::env_vars::spi_why() {
            eprintln!(
                "[spi-why]   late-terminal-convergence header={} merge={}",
                header, merge,
            );
        }
        claimed.insert(merge.clone());
        header_merges.insert(header, merge);
        changed = true;
    }
    changed
}

#[allow(clippy::too_many_arguments)]
fn loop_entry_arm_reaches_merge_or_structured_exit(
    blocks: &[BodyBlock],
    by_name: &HashMap<&str, &BodyBlock>,
    loop_body: &HashSet<&str>,
    loop_info: &super::loopforest::NaturalLoop,
    info: &LoopMergeInfo,
    start: &str,
    candidate: &str,
) -> bool {
    let mut reached = false;
    let direct_structured_exit = start == loop_info.header
        || start == info.merge
        || start == info.continue_target
        || block_ends_in_void_return(blocks, start)
        || block_ends_in_unreachable(blocks, start);
    let mut seen = HashSet::new();
    let mut stack = vec![start.to_string()];
    while let Some(node) = stack.pop() {
        if node == candidate {
            reached = true;
            continue;
        }
        if node == loop_info.header || node == info.merge || node == info.continue_target {
            continue;
        }
        if block_ends_in_void_return(blocks, &node) || block_ends_in_unreachable(blocks, &node) {
            continue;
        }
        if !loop_body.contains(node.as_str()) {
            return false;
        }
        if !seen.insert(node.clone()) {
            continue;
        }
        let Some(block) = by_name.get(node.as_str()) else {
            return false;
        };
        let successors = block_successors(block);
        if successors.is_empty() {
            return false;
        }
        stack.extend(successors);
    }
    // `candidate` is itself the other direct arm, so it is reached by that side of the selection. A
    // *direct* break/continue/terminal opposite arm is also complete without reaching it. Do not
    // generalize this to a later exit: an ordinary live subgraph may have a deeper shared convergence
    // that must win over either direct arm.
    reached || direct_structured_exit
}

/// Find a loop-local convergence for one selection whose normal post-dominator is the loop merge.
/// Every path starting at every arm must either reach the returned block or exit via this loop's
/// merge/continue/header. A nested loop or any other exit makes the proof fail conservatively.
pub(in crate::native) fn loop_exit_selection_convergence(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    loop_info: &super::loopforest::NaturalLoop,
    info: &LoopMergeInfo,
    header: &str,
) -> Option<String> {
    let header_block = blocks.iter().find(|block| block.name == header)?;
    let arms = block_successors(header_block);
    if arms.len() < 2 {
        return None;
    }
    let loop_body: HashSet<&str> = loop_info.body.iter().map(String::as_str).collect();
    let nested_loop_nodes: HashSet<&str> = forest
        .loops
        .iter()
        .filter(|other| other.header != loop_info.header)
        .filter(|other| loop_body.contains(other.header.as_str()))
        .flat_map(|other| other.body.iter().map(String::as_str))
        .collect();
    let all_loop_roles: HashSet<&str> = loop_merges
        .values()
        .flat_map(|role| [role.merge.as_str(), role.continue_target.as_str()])
        .collect();
    let candidate_depth = |name: &str| {
        let mut depth = 0usize;
        let mut cur = name;
        while let Some(parent) = forest.idom(cur) {
            depth += 1;
            cur = parent;
        }
        depth
    };
    let mut candidates: Vec<(&str, usize)> = blocks
        .iter()
        .map(|block| block.name.as_str())
        .filter(|candidate| {
            *candidate != header
                && *candidate != loop_info.header
                && *candidate != info.merge
                && *candidate != info.continue_target
                && loop_body.contains(candidate)
                && !nested_loop_nodes.contains(candidate)
                && !all_loop_roles.contains(candidate)
        })
        .map(|candidate| (candidate, candidate_depth(candidate)))
        .collect();
    candidates.sort_by_key(|(candidate, depth)| (*depth, *candidate));

    let mut best: Option<(usize, usize, &str)> = None;
    for (candidate, depth) in candidates {
        // A candidate need not be reached by every arm: an arm that only breaks/continues the
        // enclosing loop is a legal structured exit. Prefer the candidate reached by the MOST arms,
        // then the earliest one. Arm coverage chooses a shared convergence (`latch`) over a one-arm
        // body block; minimum dominator depth avoids skipping that convergence for a later shared
        // block, which would make a switch or selection claim more of the loop than it owns.
        let mut valid = true;
        let mut reaching_arms = 0usize;
        for arm in &arms {
            let mut reached = false;
            let mut seen: HashSet<String> = HashSet::new();
            let mut stack = vec![arm.clone()];
            while let Some(node) = stack.pop() {
                if node == candidate {
                    reached = true;
                    continue;
                }
                // A loop break, continue, or direct back-edge is a legal structured exit from this
                // selection. It deliberately does not need to pass through the selection merge.
                if node == info.merge || node == info.continue_target || node == loop_info.header {
                    continue;
                }
                if block_ends_in_void_return(blocks, &node)
                    || block_ends_in_unreachable(blocks, &node)
                {
                    continue;
                }
                if !seen.insert(node.clone()) {
                    continue;
                }
                if !loop_body.contains(node.as_str()) || nested_loop_nodes.contains(node.as_str()) {
                    valid = false;
                    break;
                }
                let Some(block) = blocks.iter().find(|block| block.name == node) else {
                    valid = false;
                    break;
                };
                let successors = block_successors(block);
                if successors.is_empty() {
                    valid = false;
                    break;
                }
                stack.extend(successors);
            }
            if !valid {
                valid = false;
                break;
            }
            reaching_arms += reached as usize;
        }
        if valid && reaching_arms > 0 {
            let replace = best.as_ref().is_none_or(|current| {
                reaching_arms > current.0
                    || (reaching_arms == current.0 && depth < current.1)
                    || (reaching_arms == current.0 && depth == current.1 && candidate < current.2)
            });
            if replace {
                best = Some((reaching_arms, depth, candidate));
            }
        }
    }
    best.map(|(_, _, candidate)| candidate.to_string())
}

/// Insert a fresh unique merge block for `header` whose natural (shared) merge is `natural`: redirect
/// every `header`-dominated block that branches to `natural` to the new block, which branches on to
/// `natural`. Returns the new block name, or `None` if no header-region predecessor targets `natural`.
pub(in crate::native) fn synth_unique_selection_merge(
    blocks: &mut Vec<BodyBlock>,
    forest: &LoopForest,
    header: &str,
    natural: &str,
    counter: &mut usize,
) -> Option<String> {
    let preds: HashSet<String> = header_owned_merge_predecessors(blocks, forest, header, natural)
        .into_iter()
        .collect();
    if preds.is_empty() {
        return None;
    }
    let new_name = format!("{SPLIT_PREFIX}{SEL_TOKEN}{counter}");
    *counter += 1;
    for b in blocks.iter_mut() {
        if preds.contains(&b.name) {
            if let Some(t) = b.typed_mut() {
                t.redirect_successor(natural, &new_name);
            }
        }
    }
    let at = blocks
        .iter()
        .position(|b| b.name == natural)
        .unwrap_or(blocks.len());
    blocks.insert(
        at,
        synthetic_block(
            new_name.clone(),
            vec![format!("br label {natural}")],
            role_for_name(&new_name),
        ),
    );
    Some(new_name)
}

/// Phi-aware variant of [`synth_unique_selection_merge`]: when the shared `natural` merge carries a
/// phi, the header-region predecessors that get redirected to the fresh pass-through can no longer feed
/// `natural`'s phi directly. So the pass-through carries a merged phi over those redirected incomings
/// and `natural`'s phi is rebuilt to take that merged value via the pass-through edge (the
/// [`split_phi_overlap`] surgery, but with header-dominated predecessors instead of in-loop ones).
/// Returns the new unique merge block, or `None` if no header-dominated predecessor targets `natural`.
pub(in crate::native) fn synth_unique_selection_merge_phi(
    blocks: &mut Vec<BodyBlock>,
    forest: &LoopForest,
    header: &str,
    natural: &str,
    counter: &mut usize,
) -> Option<String> {
    let preds = header_owned_merge_predecessors(blocks, forest, header, natural);
    synth_unique_selection_merge_phi_explicit(blocks, &preds, natural, &HashSet::new(), counter)
}

/// The predecessors of `natural` that a selection headed by `header` owns, in block order.
///
/// One rule, in one place, because it is also the answer to "which edges did that split carry?" --
/// a caller keeping an incremental dominance relation across splits
/// ([`crate::native::cfg::graph::Dominators::record_pass_through`]) has to record exactly this set,
/// and a second copy of the rule would be a second thing to keep in step.
///
/// Construct-tree gateways are excluded: a route block stands for control that entered the region
/// from outside, so redirecting it would move edges the header does not own.
pub(in crate::native) fn header_owned_merge_predecessors(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    header: &str,
    natural: &str,
) -> Vec<String> {
    blocks
        .iter()
        .filter(|block| {
            block.role != BlockRole::ConstructTreeRoute
                && block_successors(block)
                    .iter()
                    .any(|target| target == natural)
                && forest.dominates(header, &block.name)
        })
        .map(|block| block.name.clone())
        .collect()
}

/// Phi-aware unique-merge synthesis over an explicit set of owned predecessors.
///
/// A construct-tree gateway can sit between an owned predecessor and the shared natural merge. In
/// that case redirect the owned predecessor around the gateway and copy the gateway's phi value onto
/// the new predecessor edge. If outside control still enters the gateway, its original natural-phi
/// incoming remains; otherwise the private merge replaces it completely.
pub(in crate::native) fn synth_unique_selection_merge_phi_explicit(
    blocks: &mut Vec<BodyBlock>,
    preds: &[String],
    natural: &str,
    routes_into_natural: &HashSet<String>,
    counter: &mut usize,
) -> Option<String> {
    let pred_set = preds.iter().map(String::as_str).collect::<HashSet<_>>();
    let mut redirects = Vec::<(String, String)>::new();
    for block in blocks.iter() {
        if !pred_set.contains(block.name.as_str()) {
            continue;
        }
        for target in block_successors(block) {
            if target == natural || routes_into_natural.contains(&target) {
                redirects.push((block.name.clone(), target));
            }
        }
    }
    if redirects.is_empty() {
        return None;
    }

    let direct_preds = redirects
        .iter()
        .filter(|(_, target)| target == natural)
        .map(|(pred, _)| pred.as_str())
        .collect::<HashSet<_>>();
    let route_preds = redirects
        .iter()
        .filter(|(_, target)| target != natural)
        .fold(
            HashMap::<&str, Vec<&str>>::new(),
            |mut map, (pred, route)| {
                map.entry(route.as_str()).or_default().push(pred.as_str());
                map
            },
        );
    let route_has_unredirected_predecessor = |route: &str| {
        blocks.iter().any(|block| {
            !pred_set.contains(block.name.as_str())
                && block_successors(block).iter().any(|target| target == route)
        })
    };

    let new_name = format!("{SPLIT_PREFIX}{SEL_TOKEN}{counter}");
    *counter += 1;

    // Rebuild `natural`'s phis: the redirected predecessors' incomings fold into a merged phi in the
    // pass-through; the rest stay, plus one `[merged, pass-through]` incoming. Driven off `natural`'s
    // TYPED carrier (line order preserved by `t.insts`). Aggregate phis (`phi_incoming: None`) carry no
    // typed incoming list, so they are skipped — such a phi fails primary emit → retry regardless (the
    // carrier is the sole emission substrate). Each primitive has a `== re-lower` unit test (kb
    // "STEP-1/STEP-2 decomposition").
    let nat_idx = blocks.iter().position(|b| b.name == natural)?;
    type TypedIncomings = Vec<(crate::native::ir::LlValue, String)>;
    // (merged pass-through phi name, phi type, redirected typed incomings).
    let mut passthrough_merges: Vec<(String, crate::native::ir::LlType, TypedIncomings)> =
        Vec::new();
    // (nat phi dst, kept typed incomings + the `[ merged, {new_name} ]` funnel).
    let mut nat_rewrites: Vec<(String, TypedIncomings)> = Vec::new();
    if let Some(t) = &blocks[nat_idx].typed {
        for inst in &t.insts {
            let (Some(dst), Some((ty, inc))) = (inst.result.clone(), inst.phi_incoming().clone())
            else {
                continue;
            };
            let mut typed_red = Vec::new();
            let mut kept_plus = Vec::new();
            for (value, predecessor) in inc {
                if direct_preds.contains(predecessor.as_str()) {
                    typed_red.push((value, predecessor));
                    continue;
                }
                if let Some(owned_preds) = route_preds.get(predecessor.as_str()) {
                    typed_red.extend(
                        owned_preds
                            .iter()
                            .map(|owned| (value.clone(), (*owned).to_string())),
                    );
                    if route_has_unredirected_predecessor(&predecessor) {
                        kept_plus.push((value, predecessor));
                    }
                    continue;
                }
                kept_plus.push((value, predecessor));
            }
            if typed_red.is_empty() {
                continue;
            }
            let merged = format!("{new_name}.phi{}", passthrough_merges.len());
            kept_plus.push((
                crate::native::ir::LlValue::Local(merged.clone()),
                new_name.clone(),
            ));
            passthrough_merges.push((merged, ty, typed_red));
            nat_rewrites.push((dst, kept_plus));
        }
    }
    if let Some(t) = blocks[nat_idx].typed_mut() {
        for (dst, kept_plus) in &nat_rewrites {
            t.set_phi_incomings(dst, kept_plus);
        }
    }

    for (predecessor, old_target) in &redirects {
        if let Some(block) = blocks.iter_mut().find(|block| block.name == *predecessor) {
            block.typed_mut()?.redirect_successor(old_target, &new_name);
        }
    }

    let at = blocks
        .iter()
        .position(|b| b.name == natural)
        .unwrap_or(blocks.len());
    // Pass-through block: build its carrier by pushing the typed merged phis onto a fresh
    // `br label {natural}` carrier (byte-identical to lowering the merged phi lines + terminator —
    // `push_value_phi`'s `== re-lower` test).
    let mut blk = crate::native::tir::lower_block_carrier(
        &new_name,
        &[format!("br label {natural}")],
        &std::collections::HashMap::new(),
    )?;
    for (merged, ty, typed_red) in &passthrough_merges {
        blk.push_value_phi(merged, ty, typed_red);
    }
    blocks.insert(
        at,
        BodyBlock {
            name: new_name.clone(),
            role: role_for_name(&new_name),
            typed: Some(blk.into()),
        },
    );

    Some(new_name)
}

/// Does the single-exit loop `header`'s merge block double as the natural selection merge
/// (post-dominator) of a conditional OUTSIDE this loop? If so, the block is claimed by two constructs
/// (the loop and the outer selection) and the loop needs a distinct synthesized merge before it can be
/// structured. `merge` is examined against `selection_merges`, restricted to headers not in this
/// loop's body (an in-loop conditional sharing the merge is a structured break, handled elsewhere).
pub(in crate::native) fn merge_collides_with_outer_selection_from(
    forest: &LoopForest,
    selection_merges: &HashMap<String, String>,
    header: &str,
    merge: &str,
    converge_inloop: bool,
) -> bool {
    let Some(l) = forest.loop_for_header(header) else {
        return false;
    };
    let body: HashSet<&str> = l.body.iter().map(String::as_str).collect();
    // Base: an OUTER selection (header not in this loop's body) whose post-dominator is the loop merge.
    if selection_merges
        .iter()
        .any(|(h, m)| m == merge && !body.contains(h.as_str()))
    {
        return true;
    }
    // `converge_inloop` (the reject-triggered 4th `structured_plan` attempt): also treat an IN-LOOP
    // selection sharing the loop's merge as a collision, so the existing `split_phi_overlap` gives the
    // loop a distinct merge and the in-loop selection can claim `natural` — resolving the dominant
    // `cond-phi-shared/loop-role/merge-inloop` reject class (07 and the 107-fn bucket). Only enabled for
    // the retry attempt on a base-REJECTING function, so base-admitting functions stay byte-identical.
    if converge_inloop {
        return selection_merges
            .iter()
            .any(|(h, m)| m == merge && body.contains(h.as_str()) && h.as_str() != header);
    }
    false
}

/// Split a no-phi merge==continue overlap: insert a fresh pass-through block that branches to the
/// shared `exit`, redirect the inner loop's in-body predecessors of `exit` to it, and return the new
/// block's name (the inner loop's distinct merge). The original `exit` is untouched as a branch
/// target of the *enclosing* loop, preserving the enclosing continue. Returns `None` if no in-body
/// predecessor actually targets `exit` (nothing to split).
pub(in crate::native) fn split_no_phi_overlap(
    blocks: &mut Vec<BodyBlock>,
    forest: &LoopForest,
    header: &str,
    exit: &str,
    counter: &mut usize,
) -> Option<String> {
    let body: Vec<String> = forest.loop_for_header(header)?.body.clone();
    // In-loop blocks whose terminator targets the shared exit.
    let preds: Vec<String> = body
        .iter()
        .filter(|name| {
            blocks
                .iter()
                .find(|b| &b.name == *name)
                .map(|b| block_successors(b).iter().any(|s| s == exit))
                .unwrap_or(false)
        })
        .cloned()
        .collect();
    if preds.is_empty() {
        return None;
    }

    let new_name = format!("{SPLIT_PREFIX}{counter}");
    *counter += 1;

    // Redirect each in-body predecessor's terminator from `exit` to the new block.
    for b in blocks.iter_mut() {
        if preds.iter().any(|p| p == &b.name) {
            if let Some(t) = b.typed_mut() {
                t.redirect_successor(exit, &new_name);
            }
        }
    }

    // Insert the pass-through block (unconditional branch to the original exit) right before `exit`
    // so it stays in a natural position; program order is otherwise irrelevant to correctness.
    let pass_through = synthetic_block(
        new_name.clone(),
        vec![format!("br label {exit}")],
        role_for_name(&new_name),
    );
    let insert_at = blocks
        .iter()
        .position(|b| b.name == exit)
        .unwrap_or(blocks.len());
    blocks.insert(insert_at, pass_through);

    Some(new_name)
}

/// do-while normalization (loop rotation): if the loop's `latch` (continue) block ends in a
/// conditional whose arms are the loop `header` (back-edge) and the loop `merge` (the exit test is at
/// the loop bottom), split off a fresh unconditional continue block `C` (`br header`), redirect the
/// latch's back-edge arm to `C`, and rewrite the header's phis to take the back-edge from `C` instead
/// of the latch. The latch then becomes an ordinary body block branching to `{C, merge}` — a
/// `{continue, merge}` break the selection layer already recognizes. Returns the new continue block
/// `C`, or `None` if the latch is not a do-while exit test (e.g. an ordinary unconditional latch).
pub(in crate::native) fn synth_dowhile_continue(
    blocks: &mut Vec<BodyBlock>,
    header: &str,
    latch: &str,
    merge: &str,
    counter: &mut usize,
) -> Option<String> {
    let (t, f) = {
        let lb = blocks.iter().find(|b| b.name == latch)?;
        conditional_branch_targets(lb)?
    };
    let arms = [t.as_str(), f.as_str()];
    if !(arms.contains(&header) && arms.contains(&merge)) {
        return None;
    }

    let cont = format!("{CONT_PREFIX}{counter}");
    *counter += 1;

    // Redirect the latch's back-edge arm (-> header) to the new continue block.
    if let Some(lb) = blocks.iter_mut().find(|b| b.name == latch) {
        if let Some(t) = lb.typed_mut() {
            t.redirect_successor(header, &cont);
        }
    }
    // Rewrite the header's phi incomings: the back-edge predecessor is now C, not the latch.
    if let Some(hb) = blocks.iter_mut().find(|b| b.name == header) {
        if let Some(t) = hb.typed_mut() {
            t.rewrite_phi_predecessor(latch, &cont);
        }
    }
    // Insert C just before the header so the back-edge source sits adjacent to its target.
    let at = blocks
        .iter()
        .position(|b| b.name == header)
        .unwrap_or(blocks.len());
    blocks.insert(
        at,
        synthetic_block(
            cont.clone(),
            vec![format!("br label {header}")],
            role_for_name(&cont),
        ),
    );
    Some(cont)
}

/// A loop header that ALSO ends in a 2-way conditional whose arms are both in-loop blocks (neither the
/// loop's merge nor its continue) is simultaneously a loop construct AND a selection construct. A single
/// block cannot carry both `OpLoopMerge` and `OpSelectionMerge`, so the emitter leaves the conditional
/// bare and spirv-val rejects it ("Selection must be structured"). Split the conditional off into a
/// fresh successor block: the loop header keeps its phis/body and an unconditional branch to the new
/// block, which inherits the conditional and becomes an ordinary selection header (its merge resolved by
/// [`selection_merges`] after the split). Phi incomings in the two arms that named the header as
/// predecessor are rewired to the new block. Returns the new selection block's name, or `None` if the
/// header is not this shape (a `while`-loop header — one arm is the merge — or a do-while self-latch is
/// left untouched).
pub(in crate::native) fn split_loop_header_selection(
    blocks: &mut Vec<BodyBlock>,
    header: &str,
    merge: &str,
    continue_target: &str,
    loop_body: &HashSet<String>,
    counter: &mut usize,
) -> Option<String> {
    let (t, f) = {
        let hb = blocks.iter().find(|b| b.name == header)?;
        conditional_branch_targets(hb)?
    };
    // Only a genuine in-loop selection: neither arm is the loop's merge/continue, and neither is the
    // header itself (a self-back-edge latch — that is the do-while rotation's job, not a selection).
    let roles = [merge, continue_target, header];
    if roles.contains(&t.as_str())
        || roles.contains(&f.as_str())
        || !loop_body.contains(&t)
        || !loop_body.contains(&f)
    {
        return None;
    }

    let sel = format!("{LOOP_HEADER_SELECTION_PREFIX}{counter}");
    *counter += 1;

    // Lift the header's conditional terminator into the new block (carrying it on a terminator-only
    // carrier); the header branches to it instead.
    let sel_carrier = {
        let hb = blocks.iter_mut().find(|b| b.name == header)?;
        let sel_carrier = hb.typed.as_ref()?.terminator_only_block(&sel);
        if let Some(t) = hb.typed_mut() {
            t.set_unconditional_branch(&sel);
        }
        sel_carrier
    };
    // Phi incomings in the arms that named the header as their predecessor now flow through `sel`.
    for arm in [&t, &f] {
        if let Some(ab) = blocks.iter_mut().find(|b| &b.name == arm) {
            if let Some(t) = ab.typed_mut() {
                t.rewrite_phi_predecessor(header, &sel);
            }
        }
    }
    // Insert `sel` immediately after the header so it sits at the top of the loop body.
    let at = blocks
        .iter()
        .position(|b| b.name == header)
        .map(|i| i + 1)
        .unwrap_or(blocks.len());
    blocks.insert(
        at,
        BodyBlock {
            name: sel.clone(),
            role: role_for_name(&sel),
            typed: Some(sel_carrier.into()),
        },
    );
    Some(sel)
}

/// The `switch` analog of [`split_loop_header_selection`]. A loop header that ALSO ends in a `switch`
/// whose targets are all in-loop blocks (none is the loop's merge, continue, or the header itself) is
/// simultaneously a loop construct AND a switch selection construct — illegal on one block (it would
/// need both `OpLoopMerge` and `OpSelectionMerge`). Lift the `switch` terminator into a fresh successor
/// block so the header branches unconditionally to it and the new block becomes an ordinary switch
/// header (its merge resolved by [`unique_selection_merges`]' switch path). Phi incomings in the
/// case/default targets that named the header as predecessor are rewired to the new block. Returns the
/// new block's name, or `None` if the header is not this shape — any switch target being the loop's
/// merge/continue/header is a structured break/continue switch, left to the relooper retry (and still
/// rejected by `structured_plan`'s loop-header-switch gate, which now fires only on that residue).
pub(in crate::native) fn split_loop_header_switch(
    blocks: &mut Vec<BodyBlock>,
    header: &str,
    merge: &str,
    continue_target: &str,
    loop_body: &HashSet<String>,
    counter: &mut usize,
) -> Option<String> {
    let targets = {
        let hb = blocks.iter().find(|b| b.name == header)?;
        let is_switch = is_switch_block(hb);
        if !is_switch {
            return None;
        }
        block_successors(hb)
    };
    // Only a genuine in-loop switch: no case/default target is the loop's merge/continue/header.
    let roles = [merge, continue_target, header];
    if targets
        .iter()
        .any(|t| roles.contains(&t.as_str()) || !loop_body.contains(t))
    {
        return None;
    }

    let sel = format!("%metal2vulkan.lhsw.{counter}");
    *counter += 1;

    // Lift the header's switch terminator into the new block (carrying it on a terminator-only carrier);
    // the header branches to it instead.
    let sel_carrier = {
        let hb = blocks.iter_mut().find(|b| b.name == header)?;
        let sel_carrier = hb.typed.as_ref()?.terminator_only_block(&sel);
        if let Some(t) = hb.typed_mut() {
            t.set_unconditional_branch(&sel);
        }
        sel_carrier
    };
    // Phi incomings in the switch targets that named the header as their predecessor flow through `sel`.
    for arm in &targets {
        if let Some(ab) = blocks.iter_mut().find(|b| &b.name == arm) {
            if let Some(t) = ab.typed_mut() {
                t.rewrite_phi_predecessor(header, &sel);
            }
        }
    }
    let at = blocks
        .iter()
        .position(|b| b.name == header)
        .map(|i| i + 1)
        .unwrap_or(blocks.len());
    blocks.insert(
        at,
        BodyBlock {
            name: sel.clone(),
            role: role_for_name(&sel),
            typed: Some(sel_carrier.into()),
        },
    );
    Some(sel)
}

/// If block `b` is inside a loop whose `{merge, continue}` are exactly its conditional arms
/// `{t, f}`, return that loop's continue block — the `OpSelectionMerge` for a structured
/// break-or-continue. The break arm (the loop merge) is a valid structured jump out of the selection,
/// so the selection's merge is the continue (the fall-through arm). Returns `None` if no enclosing
/// loop's roles match (then `b` is an ordinary selection, handled by the post-dominator path).
pub(in crate::native) fn loop_break_continue_merge(
    forest: &LoopForest,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    b: &str,
    t: &str,
    f: &str,
) -> Option<String> {
    for l in &forest.loops {
        if !l.body.iter().any(|n| n == b) {
            continue;
        }
        let Some(info) = loop_merges.get(&l.header) else {
            continue;
        };
        let arms = [t, f];
        if !arms.contains(&info.merge.as_str()) {
            continue;
        }
        // The break-mid-body shape: one arm is the loop merge (a `break`), the other is the continue
        // block (a separate latch, distinct from this block). That is a structured loop break, not a
        // selection — its OpSelectionMerge is the continue block. (The do-while self-latch shape, where
        // the block IS the latch and conditionally branches header/merge, is a different gap left for a
        // follow-up; here `continue_target == b` so it is excluded.)
        let other = if t == info.merge { f } else { t };
        if other == info.continue_target
            && info.merge != info.continue_target
            && info.continue_target != b
        {
            return Some(info.continue_target.clone());
        }
    }
    None
}

/// True when block `b` is a non-header block inside a loop whose conditional arms `{t, f}` are a
/// spirv-val-legal BARE break/continue: one arm is the loop's continue target OR its merge (a
/// structured `continue`/`break` exit), and the OTHER arm stays inside the same loop body (or is itself
/// a loop role). Such a block needs NO `OpSelectionMerge` — a bare `OpBranchConditional` with one arm =
/// loop continue/merge is structured even when the other arm re-converges from several in-loop
/// predecessors (empirically confirmed against spirv-val). This generalizes the narrow `{merge,
/// continue}` shape `loop_break_continue_merge` handles to `{any-in-loop, continue|merge}`, so the
/// acceptance gate can SKIP the block (emit a bare branch) rather than reject `branch-no-merge`.
/// Removing a purely-structural merge hint on an already-legal branch cannot change runtime control
/// flow, so this is sound by construction.
pub(in crate::native) fn bare_loop_exit_branch(
    forest: &LoopForest,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    b: &str,
    t: &str,
    f: &str,
) -> bool {
    for l in &forest.loops {
        if l.header == b || !l.body.iter().any(|n| n == b) {
            continue;
        }
        let Some(info) = loop_merges.get(&l.header) else {
            continue;
        };
        let is_role = |n: &str| n == info.merge || n == info.continue_target;
        // One arm is a structured break/continue (the loop's merge or continue target). The other arm
        // needs no merge either — a bare `OpBranchConditional` with one loop-role arm is structured
        // regardless of the other target (the break arm may lead to the merge through an exit block,
        // e.g. the synthesized `lmerge.sel*` re-convergence). The other arm must not be the loop header
        // (that would be a back-edge, not a forward arm).
        if (is_role(t) || is_role(f)) && t != l.header && f != l.header {
            return true;
        }
    }
    false
}

/// [`bare_loop_exit_branch`] after selection/loop synthesis may have inserted a single-successor
/// pass-through on the loop-role arm. The branch is still a bare loop break/continue when the target is
/// either the role itself or a role-tagged pass-through chain that reaches that role.
pub(in crate::native) fn bare_loop_exit_branch_with_passthroughs(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    b: &str,
    t: &str,
    f: &str,
) -> bool {
    let by_name = blocks
        .iter()
        .map(|block| (block.name.as_str(), block))
        .collect::<HashMap<_, _>>();
    for l in &forest.loops {
        if l.header == b || !l.body.iter().any(|n| n == b) {
            continue;
        }
        let Some(info) = loop_merges.get(&l.header) else {
            continue;
        };
        let is_role = |n: &str| loop_role_or_passthrough(&by_name, n, info);
        if (is_role(t) || is_role(f)) && t != l.header && f != l.header {
            return true;
        }
    }
    false
}

/// Final-CFG counterpart of [`bare_loop_exit_branch`]. Late construct-tree synthesis can replace a
/// loop header and thereby stale the header-keyed `LoopMergeInfo` map, while natural-loop analysis
/// still proves the branch shape directly: one arm stays in the loop and the other is an actual loop
/// exit. Such a conditional is a structured bare break and must not acquire an `OpSelectionMerge`.
pub(in crate::native) fn bare_natural_loop_exit_branch(
    forest: &LoopForest,
    block: &str,
    true_target: &str,
    false_target: &str,
) -> bool {
    forest.loops.iter().any(|loop_| {
        loop_.header != block
            && loop_.body.iter().any(|name| name == block)
            && ((loop_.body.iter().any(|name| name == true_target)
                && loop_.exits.iter().any(|name| name == false_target))
                || (loop_.body.iter().any(|name| name == false_target)
                    && loop_.exits.iter().any(|name| name == true_target)))
    })
}

/// Return the first target outside `b`'s dominated region that is still inside an enclosing
/// selection region. The caller uses this as a continuation split point: the nested branch remains
/// a structured selection by merging at a private pass-through before that shared continuation,
/// instead of emitting a bare conditional.
pub(in crate::native) fn enclosing_selection_region_exit_target(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    selection_merges: &HashMap<String, String>,
    b: &str,
    t: &str,
    f: &str,
    merge: Option<&str>,
) -> Option<String> {
    if b.starts_with(LOOP_HEADER_SELECTION_PREFIX) {
        return None;
    }
    for target in [t, f] {
        if merge.is_some_and(|merge| target == merge) {
            continue;
        }
        if !forest.dominates(b, target) {
            continue;
        }
        if let Some(exit) = first_enclosing_selection_region_exit(
            blocks,
            forest,
            loop_merges,
            selection_merges,
            b,
            target,
            merge,
        ) {
            return Some(exit);
        }
    }
    None
}

fn first_enclosing_selection_region_exit(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    selection_merges: &HashMap<String, String>,
    b: &str,
    target: &str,
    merge: Option<&str>,
) -> Option<String> {
    let by_name = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.name.clone(), index))
        .collect::<HashMap<_, _>>();
    let mut seen = HashSet::new();
    let mut stack = vec![target.to_string()];
    while let Some(node) = stack.pop() {
        if !seen.insert(node.clone()) {
            continue;
        }
        if merge.is_some_and(|merge| node == merge) {
            continue;
        }
        let Some(block) = by_name.get(node.as_str()).map(|index| &blocks[*index]) else {
            continue;
        };
        for successor in block_successors(block) {
            if merge.is_some_and(|merge| successor == merge) {
                continue;
            }
            if legal_enclosing_loop_exit(forest, loop_merges, &node, &successor) {
                continue;
            }
            if forest.dominates(b, &successor) {
                stack.push(successor);
            } else if exits_to_enclosing_selection_region(
                blocks,
                forest,
                selection_merges,
                &by_name,
                b,
                &successor,
            ) {
                return Some(successor);
            }
        }
    }
    None
}

fn legal_enclosing_loop_exit(
    forest: &LoopForest,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    source: &str,
    target: &str,
) -> bool {
    forest.loops.iter().any(|loop_info| {
        let encloses =
            loop_info.header == source || loop_info.body.iter().any(|name| name == source);
        encloses
            && (loop_info.header == target
                || loop_merges
                    .get(&loop_info.header)
                    .is_some_and(|info| info.merge == target || info.continue_target == target))
    })
}

/// Identify the one exact enclosing boundary crossed by an ordinary selection's owned source
/// region. Unlike the construct-tree sibling-region query, this declines loop-role branches,
/// conditional latches, terminal declarations, and regions with multiple distinct escapes. The
/// ordinary planner can therefore materialize the returned boundary while processing the header,
/// without revisiting or mutating already-constructed selections afterward.
pub(in crate::native) fn ordinary_selection_enclosing_boundary_target(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    source_selection_merges: &HashMap<String, String>,
    header: &str,
    merge: &str,
) -> Option<String> {
    let by_name = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.name.clone(), index))
        .collect::<HashMap<_, _>>();
    let header_block = &blocks[*by_name.get(header)?];
    let (true_target, false_target) = conditional_branch_targets(header_block)?;
    if bare_loop_exit_branch_with_passthroughs(
        blocks,
        forest,
        loop_merges,
        header,
        &true_target,
        &false_target,
    ) {
        return None;
    }
    let header_successors = block_successors(header_block);
    if forest.loops.iter().any(|loop_info| {
        loop_info.latches.iter().any(|latch| latch == header)
            && header_successors
                .iter()
                .any(|successor| successor == &loop_info.header)
    }) {
        return None;
    }
    if by_name
        .get(merge)
        .map(|index| &blocks[*index])
        .is_some_and(is_bare_unreachable)
    {
        return None;
    }

    let mut seen = HashSet::new();
    let mut stack = header_successors
        .into_iter()
        .filter(|target| target != merge)
        .collect::<Vec<_>>();
    let mut boundary = None;
    while let Some(node) = stack.pop() {
        if node == merge || !seen.insert(node.clone()) {
            continue;
        }
        let Some(block) = by_name.get(node.as_str()).map(|index| &blocks[*index]) else {
            continue;
        };
        if !forest.dominates(header, &node) {
            continue;
        }
        for successor in block_successors(block) {
            if successor == merge
                || legal_enclosing_loop_exit(forest, loop_merges, &node, &successor)
            {
                continue;
            }
            if forest.dominates(header, &successor) {
                stack.push(successor);
                continue;
            }
            if exits_to_enclosing_selection_region(
                blocks,
                forest,
                source_selection_merges,
                &by_name,
                &node,
                &successor,
            ) {
                if boundary
                    .as_ref()
                    .is_some_and(|current: &String| current != &successor)
                {
                    return None;
                }
                boundary = Some(successor);
            }
        }
    }
    boundary
}

fn dominated_region_exits(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    header: &str,
    merge: Option<&str>,
) -> HashSet<String> {
    blocks
        .iter()
        .filter(|block| forest.dominates(header, &block.name))
        .filter(|block| merge.is_none_or(|merge| block.name != merge))
        .flat_map(block_successors)
        .filter(|successor| merge.is_none_or(|merge| successor != merge))
        .filter(|successor| !forest.dominates(header, successor))
        .collect()
}

/// Source owners whose construction can expose a nested selection with both arms routing into it.
///
/// The ownership builder runs innermost-first, so the nested header is already registered when one
/// of these enclosing owners is produced. Indexing that dependency on the immutable source CFG lets
/// the enclosing producer finish only the routes it can unlock instead of revisiting the whole CFG
/// after construction.
#[derive(Default)]
pub(in crate::native) struct PureEnclosingSelectionOwners {
    dependents_by_owner: HashMap<String, HashSet<String>>,
}

pub(in crate::native) fn pure_enclosing_selection_owners(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    source_selection_merges: &HashMap<String, String>,
) -> PureEnclosingSelectionOwners {
    let by_name = blocks
        .iter()
        .map(|block| (block.name.as_str(), block))
        .collect::<HashMap<_, _>>();
    let mut dependents_by_owner: HashMap<String, HashSet<String>> = HashMap::new();
    for (header, merge) in source_selection_merges {
        let Some((left, right)) = by_name
            .get(header.as_str())
            .and_then(|block| conditional_branch_targets(block))
        else {
            continue;
        };
        if (left == *merge) == (right == *merge) {
            continue;
        }
        let mut current = header.as_str();
        while let Some(parent) = forest.idom(current) {
            if source_selection_merges.contains_key(parent) {
                dependents_by_owner
                    .entry(parent.to_string())
                    .or_default()
                    .insert(header.clone());
            }
            current = parent;
        }
    }
    PureEnclosingSelectionOwners {
        dependents_by_owner,
    }
}

/// Materialize selections exposed by construction of one indexed enclosing owner.
///
/// Unique-merge synthesis can produce `H -> { shared, M -> outer-merge }`, where an enclosing sibling
/// also enters `shared`. Treating H as a bare routing branch loses real conditional ownership and
/// forces a later whole-CFG completeness sweep to rediscover H. When the indexed enclosing owner is
/// registered, clone the shared arm for H,
/// transfer every nested header's merge assignment through the clone's structural rename map, and
/// refunnel H-owned predecessors into one private merge. Processing innermost-first means a later
/// enclosing clone carries already-complete nested ownership with it.
pub(in crate::native) fn materialize_pure_enclosing_selection_routes_for_owner(
    blocks: &mut Vec<BodyBlock>,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    header_merges: &mut HashMap<String, String>,
    indexed_owners: &PureEnclosingSelectionOwners,
    owner: &str,
    counter: &mut usize,
) -> bool {
    let Some(dependents) = indexed_owners.dependents_by_owner.get(owner) else {
        return false;
    };
    let mut dependents = dependents.clone();
    let mut changed = false;
    let mut declined = HashSet::new();
    loop {
        let forest = analyze(blocks);
        let by_name = blocks
            .iter()
            .enumerate()
            .map(|(index, block)| (block.name.clone(), index))
            .collect::<HashMap<_, _>>();
        let mut headers = header_merges
            .iter()
            .filter_map(|(header, merge)| {
                dependents.contains(header).then_some(())?;
                (!declined.contains(header)).then_some(())?;
                let block = &blocks[*by_name.get(header)?];
                conditional_branch_targets(block)?;
                Some((
                    header.clone(),
                    merge.clone(),
                    depth_from_forest(&forest, header),
                ))
            })
            .collect::<Vec<_>>();
        headers.sort_by(|left, right| right.2.cmp(&left.2).then(left.0.cmp(&right.0)));

        let mut candidate = None;
        for (header, merge, _) in headers {
            let block = &blocks[by_name[&header]];
            let Some((left, right)) = conditional_branch_targets(block) else {
                continue;
            };
            let other = match (left == merge, right == merge) {
                (true, false) => right,
                (false, true) => left,
                _ => continue,
            };
            let Some(merge_block) = by_name.get(&merge).map(|index| &blocks[*index]) else {
                continue;
            };
            let merge_successors = block_successors(merge_block);
            let [merge_successor] = merge_successors.as_slice() else {
                continue;
            };
            let merge_routes_out = !forest.dominates(&header, merge_successor)
                && exits_to_enclosing_selection_region(
                    blocks,
                    &forest,
                    header_merges,
                    &by_name,
                    &header,
                    merge_successor,
                );
            let other_routes_out = !forest.dominates(&header, &other)
                && exits_to_enclosing_selection_region(
                    blocks,
                    &forest,
                    header_merges,
                    &by_name,
                    &header,
                    &other,
                );
            if merge_routes_out && other_routes_out {
                candidate = Some((header, other));
                break;
            }
        }
        let Some((header, other)) = candidate else {
            break;
        };

        let mut next_counter = *counter;
        let Some(cloned) =
            crate::native::cfg::clone_crossarm::privatize_dominated_region_with_renames(
                blocks,
                &header,
                &other,
                &mut next_counter,
            )
        else {
            declined.insert(header);
            continue;
        };
        let mut next_blocks = cloned.blocks;
        let cloned_merges = header_merges
            .iter()
            .filter_map(|(owner, merge)| {
                Some((
                    owner.clone(),
                    cloned.renamed.get(owner)?.clone(),
                    cloned
                        .renamed
                        .get(merge)
                        .cloned()
                        .unwrap_or_else(|| merge.clone()),
                ))
            })
            .collect::<Vec<_>>();

        let next_forest = analyze(&next_blocks);
        let Some(natural) = selection_merges(&next_blocks, &next_forest).remove(&header) else {
            declined.insert(header);
            continue;
        };
        let claims = header_merges
            .iter()
            .filter(|(owner, _)| owner.as_str() != header)
            .map(|(_, merge)| merge.clone())
            .chain(cloned_merges.iter().map(|(_, _, merge)| merge.clone()))
            .collect::<HashSet<_>>();
        let loop_roles = loop_role_targets_with_passthroughs(&next_blocks, loop_merges);
        let private = if next_forest.dominates(&header, &natural)
            && !claims.contains(&natural)
            && !loop_roles.contains(&natural)
        {
            Some(natural)
        } else if block_has_phi(&next_blocks, &natural) {
            synth_unique_selection_merge_phi(
                &mut next_blocks,
                &next_forest,
                &header,
                &natural,
                &mut next_counter,
            )
        } else {
            synth_unique_selection_merge(
                &mut next_blocks,
                &next_forest,
                &header,
                &natural,
                &mut next_counter,
            )
        };
        let Some(private) = private else {
            declined.insert(header);
            continue;
        };
        *blocks = next_blocks;
        for (_, cloned_owner, _) in &cloned_merges {
            dependents.insert(cloned_owner.clone());
        }
        let mut new_owners = cloned_merges
            .into_iter()
            .map(|(_, owner, merge)| (owner, merge))
            .collect::<Vec<_>>();
        new_owners.push((header.clone(), private.clone()));
        let next_forest = analyze(blocks);
        header_merges.extend(new_owners.iter().cloned());
        let terminal_links = terminal_parent_links(blocks, &next_forest);
        let mut composition_owners = new_owners
            .iter()
            .map(|(owner, _)| owner.clone())
            .collect::<Vec<_>>();
        composition_owners.sort_by(|left, right| {
            depth_from_forest(&next_forest, right)
                .cmp(&depth_from_forest(&next_forest, left))
                .then(left.cmp(right))
        });
        for owner in composition_owners {
            let completed = complete_terminal_parent_ownership(
                blocks,
                header_merges,
                &terminal_links,
                &owner,
                &mut next_counter,
            );
            for completed_owner in completed {
                if let Some(completed_dependents) =
                    indexed_owners.dependents_by_owner.get(&completed_owner)
                {
                    dependents.extend(completed_dependents.iter().cloned());
                }
                if blocks
                    .iter()
                    .find(|block| block.name == completed_owner)
                    .is_some_and(is_switch_block)
                {
                    finalize_fully_terminal_switch(
                        blocks,
                        header_merges,
                        &completed_owner,
                        &mut next_counter,
                    );
                }
            }
        }
        *counter = next_counter;
        declined.clear();
        changed = true;
    }
    changed
}

fn depth_from_forest(forest: &LoopForest, name: &str) -> usize {
    let mut depth = 0usize;
    let mut current = name;
    while let Some(parent) = forest.idom(current) {
        depth += 1;
        current = parent;
    }
    depth
}

pub(in crate::native) fn bare_enclosing_selection_region_escape(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    selection_merges: &HashMap<String, String>,
    source: &str,
    target: &str,
) -> bool {
    if forest.dominates(source, target) {
        return false;
    }
    let by_name = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.name.clone(), index))
        .collect::<HashMap<_, _>>();
    let mut current = Some(source);
    while let Some(candidate) = current {
        if let Some(block) = by_name.get(candidate).map(|index| &blocks[*index]) {
            if !selection_merges.contains_key(candidate)
                && conditional_branch_targets(block).is_some()
                && !dominated_region_exits(blocks, forest, candidate, None).is_empty()
                && forest.dominates(candidate, source)
                && !forest.dominates(candidate, target)
                && exits_to_enclosing_selection_region(
                    blocks,
                    forest,
                    selection_merges,
                    &by_name,
                    candidate,
                    target,
                )
            {
                return true;
            }
        }
        current = forest.idom(candidate);
    }
    false
}

fn exits_to_enclosing_selection_region(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    selection_merges: &HashMap<String, String>,
    by_name: &HashMap<String, usize>,
    b: &str,
    target: &str,
) -> bool {
    let mut current = b;
    while let Some(parent) = forest.idom(current) {
        if let Some(merge) = selection_merges.get(parent) {
            if let Some(header) = by_name.get(parent).map(|index| &blocks[*index]) {
                if let Some((left, right)) = conditional_branch_targets(header) {
                    let in_left = forest.dominates(&left, b);
                    let in_right = forest.dominates(&right, b);
                    if in_left != in_right && (target == merge || forest.dominates(parent, target))
                    {
                        return true;
                    }
                }
            }
        }
        current = parent;
    }
    false
}

fn loop_role_or_passthrough(
    by_name: &HashMap<&str, &BodyBlock>,
    start: &str,
    info: &LoopMergeInfo,
) -> bool {
    let mut current = start.to_string();
    for _ in 0..=by_name.len() {
        if current == info.merge || current == info.continue_target {
            return true;
        }
        let Some(block) = by_name.get(current.as_str()) else {
            return false;
        };
        if !matches!(
            block.role,
            BlockRole::LMerge | BlockRole::ConstructTreeRoute
        ) {
            return false;
        }
        let successors = block_successors(block);
        if successors.len() != 1 || successors[0] == current {
            return false;
        }
        current = successors[0].clone();
    }
    false
}

/// Re-verify a structured plan produced with a bare-loop-exit skip: reject if ANY block escapes its
/// enclosing construct to a target that is neither dominated by it, its own merge, an enclosing loop's
/// merge/continue or header (a legal break/continue/back-edge), an enclosing selection/switch's assigned
/// merge (a legal break), nor a sibling target of an enclosing switch (case-to-case fall-through). Such
/// an escape is the spirv-val "block exits the selection ... but not via a structured exit" — a genuine
/// unstructured exit the base self-checks 2/3 conservatively skip (the dead-#6 straddle ambiguity), which
/// the skip can unmask because the function no longer rejects at the now-skipped block first. FP-safe by
/// construction: every legal structured exit in the emitted plan is one of the enumerated cases, all keyed
/// on the plan's OWN recorded merges/loop-roles, so a spirv-val-legal plan is never rejected.
pub(in crate::native) fn bare_exit_escape_reason(
    ordered: &[BodyBlock],
    header_merge: &HashMap<String, String>,
    loop_merges: &HashMap<String, LoopMergeInfo>,
) -> Option<&'static str> {
    let forest = analyze(ordered);
    let names: HashSet<&str> = ordered.iter().map(|b| b.name.as_str()).collect();
    let switch_targets: HashMap<&str, HashSet<String>> = ordered
        .iter()
        .filter(|b| is_switch_block(b))
        .map(|b| (b.name.as_str(), block_successors(b).into_iter().collect()))
        .collect();
    for b in ordered {
        for s in block_successors(b) {
            if s == b.name || !names.contains(s.as_str()) {
                continue;
            }
            // (a) forward into b's own dominated subtree; (b) b's own assigned selection/switch merge.
            if forest.dominates(&b.name, &s) || header_merge.get(&b.name).is_some_and(|m| m == &s) {
                continue;
            }
            // (c) a break/continue/back-edge to an ENCLOSING loop (b in the loop's body).
            let loop_ok = forest.loops.iter().any(|l| {
                let encloses = l.header == b.name || l.body.iter().any(|n| n == &b.name);
                encloses
                    && (l.header == s
                        || loop_merges
                            .get(&l.header)
                            .is_some_and(|i| i.merge == s || i.continue_target == s))
            });
            // (d) a break to an ENCLOSING selection/switch's assigned merge.
            let sel_break = header_merge
                .iter()
                .any(|(h, m)| m == &s && h != &b.name && forest.dominates(h, &b.name));
            // (e) case-to-case / case-to-default within an ENCLOSING switch.
            let case_ok = switch_targets.iter().any(|(h, tg)| {
                *h != b.name.as_str() && forest.dominates(h, &b.name) && tg.contains(&s)
            });
            let enclosing_selection_region_ok =
                bare_enclosing_selection_region_escape(ordered, &forest, header_merge, &b.name, &s);
            if loop_ok || sel_break || case_ok || enclosing_selection_region_ok {
                continue;
            }
            if crate::env_vars::exit_why() {
                eprintln!(
                    "[exit-why] source={} target={} selection-merge={:?} enclosing-loops={:?}",
                    b.name,
                    s,
                    header_merge.get(&b.name),
                    forest
                        .loops
                        .iter()
                        .filter(|natural_loop| {
                            natural_loop.header == b.name
                                || natural_loop.body.iter().any(|node| node == &b.name)
                        })
                        .map(|natural_loop| (
                            &natural_loop.header,
                            loop_merges.get(&natural_loop.header)
                        ))
                        .collect::<Vec<_>>(),
                );
            }
            return Some("bare-exit:unstructured-escape");
        }
    }
    None
}

/// Reject an edge that begins inside a declared SPIR-V loop construct but bypasses that loop's
/// merge/continue/header. Natural-loop SCC membership is insufficient here: a header-dominated,
/// exit-only arm remains part of the structured loop even though it cannot reach a backedge.
pub(in crate::native) fn dominance_loop_exit_escape_reason(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
) -> Option<&'static str> {
    let forest = analyze(blocks);
    let names = blocks
        .iter()
        .map(|block| block.name.as_str())
        .collect::<HashSet<_>>();
    for block in blocks {
        for successor in block_successors(block) {
            if successor == block.name || !names.contains(successor.as_str()) {
                continue;
            }
            for natural_loop in &forest.loops {
                let Some(info) = loop_merges.get(&natural_loop.header) else {
                    continue;
                };
                let source_inside = natural_loop.header == block.name
                    || (forest.dominates(&natural_loop.header, &block.name)
                        && !forest.dominates(&info.merge, &block.name));
                if !source_inside {
                    continue;
                }
                let target_inside = forest.dominates(&natural_loop.header, &successor)
                    && !forest.dominates(&info.merge, &successor);
                let target_is_role = natural_loop.header == successor
                    || info.merge == successor
                    || info.continue_target == successor;
                if target_inside || target_is_role {
                    continue;
                }
                if crate::env_vars::exit_why() {
                    eprintln!(
                        "[exit-why] source={} target={} loop={} merge={} continue={}",
                        block.name,
                        successor,
                        natural_loop.header,
                        info.merge,
                        info.continue_target,
                    );
                }
                return Some("loop-exit:dominance-owned-bypass");
            }
        }
    }
    None
}
