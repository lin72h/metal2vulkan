//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

/// Bound the terminal-return selector to the same small-CFG population as the other reject-only
/// cross-arm structurizers. This is a planner-cost guard, never an emission rule: large graphs retain
/// the established fallback path unchanged.
pub(in crate::native) const TERMINAL_EXIT_SELECTION_MAX_BLOCKS: usize = 300;

pub(in crate::native) const MAX_TERMINAL_EXIT_SELECTION_ROUNDS: usize = 16;

pub(in crate::native) const TERMINAL_EXIT_COUNTER_START: usize = 3_000_000;

/// A reject-only terminal-return rewrite plus the explicit selection merges it proves. A shared
/// function return cannot be an ordinary post-dominator selection merge, but a branch that reaches a
/// real `ret` is a legal structured exit. Each recorded merge is a private pass-through before the
/// shared continuation, or an enclosing continuation reached after an already-recorded terminal guard.
pub(in crate::native) struct TerminalExitSelectionPlan {
    pub(in crate::native) blocks: Vec<BodyBlock>,
    pub(in crate::native) merges: HashMap<String, String>,
}

/// Give the `ret void` exit of one simple natural loop a private return when that source return is
/// also reached from outside the loop. The loop may have that return as its only exit or pair it with
/// one empty `unreachable` exit, which the ordinary multi-exit funnel already knows how to dispatch.
/// This keeps an early-return prefix and the later loop from competing for the same structural exit.
/// The transform is deliberately narrower than general loop cloning: exactly one natural loop, one
/// latch, and at least one source predecessor on each side of the loop boundary. Redirecting the
/// loop-local edges is then a semantics-preserving edge split with no values or side effects to
/// duplicate.
pub(in crate::native) fn privatize_single_loop_return_exit(
    blocks: &[BodyBlock],
) -> Option<Vec<BodyBlock>> {
    if blocks.len() > TERMINAL_EXIT_SELECTION_MAX_BLOCKS {
        return None;
    }
    let forest = analyze(blocks);
    let [loop_info] = forest.loops.as_slice() else {
        return None;
    };
    if loop_info.latches.len() != 1 || !(1..=2).contains(&loop_info.exits.len()) {
        return None;
    }
    let return_exits: Vec<&String> = loop_info
        .exits
        .iter()
        .filter(|exit| {
            blocks
                .iter()
                .find(|block| block.name == exit.as_str())
                .is_some_and(is_bare_ret_void)
        })
        .collect();
    let [exit] = return_exits.as_slice() else {
        return None;
    };
    let exit = exit.as_str();
    if loop_info.exits.len() == 2
        && !loop_info
            .exits
            .iter()
            .any(|candidate| block_ends_in_unreachable(blocks, candidate))
    {
        return None;
    }
    let exit_block = blocks.iter().find(|block| block.name == exit)?;
    debug_assert!(exit_block
        .typed
        .as_ref()
        .is_some_and(|t| t.insts.is_empty()));

    let loop_nodes: HashSet<&str> = loop_info.body.iter().map(String::as_str).collect();
    let mut loop_predecessors = Vec::new();
    let mut has_outer_predecessor = false;
    for block in blocks {
        if !block_successors(block).iter().any(|target| target == exit) {
            continue;
        }
        if loop_nodes.contains(block.name.as_str()) {
            loop_predecessors.push(block.name.clone());
        } else {
            has_outer_predecessor = true;
        }
    }
    if loop_predecessors.is_empty() || !has_outer_predecessor {
        return None;
    }

    let names: HashSet<&str> = blocks.iter().map(|block| block.name.as_str()).collect();
    let mut id = TERMINAL_EXIT_COUNTER_START;
    let private_return = loop {
        let candidate = format!("{SPLIT_PREFIX}{TLOOPRET_TOKEN}{id}");
        if !names.contains(candidate.as_str()) {
            break candidate;
        }
        id += 1;
    };

    let mut out = blocks.to_vec();
    for predecessor in &loop_predecessors {
        let idx = out.iter().position(|block| block.name == *predecessor)?;
        // A predecessor that does not actually branch to `exit` cannot be redirected — mirror the old
        // `redirected == terminator` no-op guard via the carrier's successor set.
        if !block_successors(&out[idx]).iter().any(|s| s == exit) {
            return None;
        }
        out[idx]
            .typed
            .as_mut()?
            .redirect_successor(exit, &private_return);
    }
    // A terminal-LOOP private-return clone (`%metal2vulkan.lmerge.tloopret*`): an `lmerge`-family
    // block but NOT the `texitret` subtype, so it tags `LMerge` (via the name-classifying seam).
    let role = role_for_name(&private_return);
    out.push(synthetic_block(
        private_return,
        vec!["ret void".to_string()],
        role,
    ));
    Some(out)
}

/// Structure nested early-return guards without treating the shared return block as a selection
/// merge. Headers inside natural loops stay out of scope; a loop-free terminal prefix is valid even
/// when a later loop remains in the function.
pub(in crate::native) fn terminal_exit_selection_merges(
    blocks: &[BodyBlock],
) -> Option<TerminalExitSelectionPlan> {
    if blocks.len() > TERMINAL_EXIT_SELECTION_MAX_BLOCKS {
        return None;
    }

    let mut out = blocks.to_vec();
    let mut merges = HashMap::new();
    let mut counter = TERMINAL_EXIT_COUNTER_START;
    let mut changed = false;
    for _ in 0..MAX_TERMINAL_EXIT_SELECTION_ROUNDS {
        let forest = analyze(&out);
        let loop_nodes: HashSet<&str> = forest
            .loops
            .iter()
            .flat_map(|loop_info| loop_info.body.iter().map(String::as_str))
            .collect();
        let header_depth = |name: &str| {
            let mut depth = 0usize;
            let mut current = name;
            while let Some(parent) = forest.idom(current) {
                depth += 1;
                current = parent;
            }
            depth
        };
        let mut headers: Vec<(String, usize)> = out
            .iter()
            .filter(|block| !merges.contains_key(&block.name))
            .filter(|block| !loop_nodes.contains(block.name.as_str()))
            .filter(|block| conditional_branch_targets(block).is_some())
            .map(|block| (block.name.clone(), header_depth(&block.name)))
            .collect();
        // Inner terminal guards must claim their continuation first. An outer guard then sees the
        // inner private merge as an in-arm predecessor and receives its own bridge after it.
        headers.sort_by(|(left, left_depth), (right, right_depth)| {
            right_depth.cmp(left_depth).then(left.cmp(right))
        });

        let mut applied = false;
        'headers: for (header, _) in headers {
            if let Some(terminal) = terminal_exit_continuation(&out, &forest, &header) {
                // The private pass-through is the selection merge. It need only be dominated by the
                // header; its successor is outside the selection, so a shared continuation can remain
                // shared without cloning its side effects or control-flow tail.
                // Both rewrites must succeed together. Work on a candidate so an unsupported return
                // form leaves the reject-only planner's input byte-for-byte untouched.
                let mut candidate = out.clone();
                let mut candidate_counter = counter;
                let Some(merge) = synth_terminal_selection_merge(
                    &mut candidate,
                    &forest,
                    &header,
                    &terminal.continuation,
                    &mut candidate_counter,
                ) else {
                    continue;
                };
                let Some(_) = synth_terminal_return_clone(
                    &mut candidate,
                    &header,
                    &terminal.exit_arm,
                    &terminal.return_arm,
                    &mut candidate_counter,
                ) else {
                    continue;
                };
                out = candidate;
                counter = candidate_counter;
                merges.insert(header, merge);
                changed = true;
                applied = true;
                break 'headers;
            }

            // An enclosing conditional need not itself branch directly to `ret`: one arm may be an
            // already-structured terminal guard whose private merge flows to the other direct arm.
            // That direct arm is the enclosing selection merge; recording it prevents ordinary
            // post-dominance from incorrectly claiming the shared function return instead.
            if let Some(merge) = terminal_exit_bridge_merge(&out, &forest, &merges, &header) {
                merges.insert(header, merge);
                changed = true;
                applied = true;
                break 'headers;
            }

            if let Some(convergence) = terminal_exit_convergence(&out, &forest, &header) {
                let mut candidate = out.clone();
                let mut candidate_counter = counter;
                let Some(merge) = synth_terminal_selection_merge(
                    &mut candidate,
                    &forest,
                    &header,
                    &convergence,
                    &mut candidate_counter,
                ) else {
                    continue;
                };
                out = candidate;
                counter = candidate_counter;
                merges.insert(header, merge);
                changed = true;
                applied = true;
                break 'headers;
            }
        }
        if !applied {
            break;
        }
    }

    changed.then_some(TerminalExitSelectionPlan {
        blocks: out,
        merges,
    })
}

/// The terminal-prefix retry has the same phi ownership obligation as ordinary unique selection
/// merges: redirecting predecessors of a phi-carrying continuation must first funnel their incoming
/// values through a phi in the private merge. Choose the existing phi-aware surgery from the current
/// candidate rather than assuming a terminal continuation is value-free.
pub(in crate::native) fn synth_terminal_selection_merge(
    blocks: &mut Vec<BodyBlock>,
    forest: &LoopForest,
    header: &str,
    continuation: &str,
    counter: &mut usize,
) -> Option<String> {
    if block_has_phi(blocks, continuation) {
        synth_unique_selection_merge_phi(blocks, forest, header, continuation, counter)
    } else {
        synth_unique_selection_merge(blocks, forest, header, continuation, counter)
    }
}

/// Add explicit selection merges to terminal `ret`/`unreachable` dispatches introduced after the
/// early-return pre-plan has run. A loop's two-exit funnel produces this exact pair: the unreachable
/// arm is the selection merge and the returning arm receives its own private return so later paths
/// cannot enter the construct through a shared source return.
pub(in crate::native) fn terminal_unreachable_selection_merges(
    blocks: &[BodyBlock],
) -> Option<TerminalExitSelectionPlan> {
    if blocks.len() > TERMINAL_EXIT_SELECTION_MAX_BLOCKS {
        return None;
    }
    let mut out = blocks.to_vec();
    let mut merges = HashMap::new();
    let mut counter = TERMINAL_EXIT_COUNTER_START;
    let mut changed = false;
    for _ in 0..MAX_TERMINAL_EXIT_SELECTION_ROUNDS {
        let forest = analyze(&out);
        let loop_nodes: HashSet<&str> = forest
            .loops
            .iter()
            .flat_map(|loop_info| loop_info.body.iter().map(String::as_str))
            .collect();
        let depth = |name: &str| {
            let mut depth = 0usize;
            let mut current = name;
            while let Some(parent) = forest.idom(current) {
                depth += 1;
                current = parent;
            }
            depth
        };
        let mut headers: Vec<String> = out
            .iter()
            .filter(|block| !merges.contains_key(&block.name))
            .filter(|block| !loop_nodes.contains(block.name.as_str()))
            .filter(|block| conditional_branch_targets(block).is_some())
            .map(|block| block.name.clone())
            .collect();
        headers.sort_by(|left, right| depth(right).cmp(&depth(left)).then(left.cmp(right)));

        let mut applied = false;
        for header in headers {
            let Some((merge, exit_arm, return_arm)) =
                terminal_unreachable_selection(&out, &forest, &header)
            else {
                continue;
            };
            let mut candidate = out.clone();
            let mut candidate_counter = counter;
            if synth_terminal_return_clone(
                &mut candidate,
                &header,
                &exit_arm,
                &return_arm,
                &mut candidate_counter,
            )
            .is_none()
            {
                continue;
            }
            out = candidate;
            counter = candidate_counter;
            merges.insert(header, merge);
            changed = true;
            applied = true;
            break;
        }
        if !applied {
            break;
        }
    }
    changed.then_some(TerminalExitSelectionPlan {
        blocks: out,
        merges,
    })
}

/// A two-arm terminal guard's non-returning continuation and the direct arm that leaves the function.
pub(in crate::native) struct TerminalExitContinuation {
    pub(in crate::native) continuation: String,
    pub(in crate::native) exit_arm: String,
    pub(in crate::native) return_arm: TerminalReturnArm,
}

/// A direct return target, or the last header-private linear block that branches to it. The terminal
/// planner gives each such exit its own `ret void` block, so later code never re-enters a selection by
/// branching to a shared source return label.
pub(in crate::native) struct TerminalReturnArm {
    pub(in crate::native) return_block: String,
    pub(in crate::native) predecessor: Option<String>,
}

/// Find the direct non-returning arm of a two-arm early-return guard. The caller supplies a fresh
/// pass-through merge before this continuation, so it does not need to treat a shared continuation as
/// a construct-internal block or search arbitrary descendants for a post-dominator. The exiting arm is
/// either the real return block itself or a header-private, straight-line tail ending at one; any branch,
/// loop, unknown target, or shared pre-return work declines conservatively.
pub(in crate::native) fn terminal_exit_continuation(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    header: &str,
) -> Option<TerminalExitContinuation> {
    let block = blocks.iter().find(|block| block.name == header)?;
    let (left, right) = conditional_branch_targets(block)?;
    let left_exit = terminal_exit_arm(blocks, forest, header, &left);
    let right_exit = terminal_exit_arm(blocks, forest, header, &right);
    match (left_exit, right_exit) {
        (None, Some(return_arm)) => Some(TerminalExitContinuation {
            continuation: left,
            exit_arm: right,
            return_arm,
        }),
        (Some(return_arm), None) => Some(TerminalExitContinuation {
            continuation: right,
            exit_arm: left,
            return_arm,
        }),
        _ => None,
    }
}

/// A conditional whose two direct arms are a return route and an `unreachable` block is a complete
/// terminal selection: the unreachable arm is a legal merge, while the return route needs its own
/// private return block so later paths cannot enter the selection through a shared source return.
/// Keep the merge header-dominated so no unrelated control path enters it.
pub(in crate::native) fn terminal_unreachable_selection(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    header: &str,
) -> Option<(String, String, TerminalReturnArm)> {
    let block = blocks.iter().find(|block| block.name == header)?;
    let (left, right) = conditional_branch_targets(block)?;
    let left_return = terminal_exit_arm(blocks, forest, header, &left);
    let right_return = terminal_exit_arm(blocks, forest, header, &right);
    let left_unreachable = block_ends_in_unreachable(blocks, &left);
    let right_unreachable = block_ends_in_unreachable(blocks, &right);
    let (merge, exit_arm, return_arm) = match (
        left_return,
        right_return,
        left_unreachable,
        right_unreachable,
    ) {
        (Some(return_arm), None, false, true) => (right, left, return_arm),
        (None, Some(return_arm), true, false) => (left, right, return_arm),
        _ => return None,
    };
    forest
        .dominates(header, &merge)
        .then_some((merge, exit_arm, return_arm))
}

pub(in crate::native) fn block_ends_in_unreachable(blocks: &[BodyBlock], name: &str) -> bool {
    blocks
        .iter()
        .find(|block| block.name == name)
        .is_some_and(is_bare_unreachable)
}

/// True if `block` is a single-terminator block (no straight-line instructions) whose terminator is
/// exactly `ret void`. Reads the typed carrier when populated — byte-identical to the line check by
/// construction (`split_body_blocks` strips blank/comment-only lines, so `insts.is_empty()` ⟺
/// `lines.len() == 1`; and `RetEmit::Void` replays the same `strip_comment().trim()` == `"ret void"`
/// test the line branch runs) — else the line fallback (pre-`populate` window).
fn is_bare_ret_void(block: &BodyBlock) -> bool {
    block
        .typed
        .as_ref()
        .is_some_and(|t| t.insts.is_empty() && matches!(t.ret, crate::native::tir::RetEmit::Void))
}

/// True if `block` is a single-terminator block whose terminator is exactly `unreachable` (read from
/// the carrier, the sole substrate).
fn is_bare_unreachable(block: &BodyBlock) -> bool {
    block.typed.as_ref().is_some_and(|t| {
        t.insts.is_empty() && matches!(t.terminator, crate::native::tir::TirTerminator::Unreachable)
    })
}

/// Prove that one direct selection arm exits the function. Only `ret void` is supported: a direct
/// return target may be shared by other paths; otherwise every pre-return block must be
/// header-dominated and have one successor, so the tail remains entirely inside this selection and
/// cannot hide an unstructured branch.
pub(in crate::native) fn terminal_exit_arm(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    header: &str,
    arm: &str,
) -> Option<TerminalReturnArm> {
    if block_ends_in_void_return(blocks, arm) {
        return Some(TerminalReturnArm {
            return_block: arm.to_string(),
            predecessor: None,
        });
    }
    if !forest.dominates(header, arm) {
        return None;
    }
    let mut current = arm.to_string();
    let mut seen = HashSet::new();
    while seen.insert(current.clone()) {
        let block = blocks.iter().find(|block| block.name == current)?;
        if block_ends_in_void_return(blocks, &current) {
            return Some(TerminalReturnArm {
                return_block: current,
                predecessor: None,
            });
        }
        let successors = block_successors(block);
        if successors.len() != 1 {
            return None;
        }
        let next = successors.into_iter().next()?;
        if block_ends_in_void_return(blocks, &next) {
            return Some(TerminalReturnArm {
                return_block: next,
                predecessor: Some(current),
            });
        }
        if !forest.dominates(header, &next) {
            return None;
        }
        current = next;
    }
    None
}

/// Give one terminal selection arm a private `ret void` block. A shared source return label cannot be
/// an arm of a structured selection: later code may branch to that label after the selection merge,
/// which SPIR-V correctly diagnoses as re-entering the construct. The proven direct arm or its proven
/// private linear tail is redirected to a fresh return block instead.
pub(in crate::native) fn synth_terminal_return_clone(
    blocks: &mut Vec<BodyBlock>,
    header: &str,
    exit_arm: &str,
    return_arm: &TerminalReturnArm,
    counter: &mut usize,
) -> Option<String> {
    if !block_ends_in_void_return(blocks, &return_arm.return_block) {
        return None;
    }
    let (redirect_block, redirect_from) = match &return_arm.predecessor {
        Some(predecessor) => (predecessor.as_str(), return_arm.return_block.as_str()),
        None => (header, exit_arm),
    };
    let redirect_index = blocks
        .iter()
        .position(|block| block.name == redirect_block)?;
    // The redirect is a no-op (and this transform declines) unless the block actually branches to
    // `redirect_from` — the carrier successor set is the dual of the old `redirected == original` guard.
    if !block_successors(blocks.get(redirect_index)?)
        .iter()
        .any(|s| s == redirect_from)
    {
        return None;
    }

    let names: HashSet<&str> = blocks.iter().map(|block| block.name.as_str()).collect();
    let mut id = *counter;
    let return_name = loop {
        let candidate = format!("{SPLIT_PREFIX}{TEXITRET_TOKEN}{id}");
        if !names.contains(candidate.as_str()) {
            break candidate;
        }
        id += 1;
    };
    *counter = id + 1;
    blocks
        .get_mut(redirect_index)?
        .typed
        .as_mut()?
        .redirect_successor(redirect_from, &return_name);
    // THE terminal-exit private-return clone (`texitret`): stamp the role its two consumers
    // (`terminal_exit_convergence`, the single-exit-return synthesizer) read instead of the name.
    blocks.push(synthetic_block(
        return_name.clone(),
        vec!["ret void".to_string()],
        BlockRole::TerminalExitReturn,
    ));
    Some(return_name)
}

/// If one direct arm is a previously structured terminal guard and its private merge immediately
/// branches to the other direct arm, that other arm is this enclosing selection's continuation. The
/// dominance check keeps the continuation inside the enclosing construct; a shared or ambiguous tail
/// stays on the fallback path.
pub(in crate::native) fn terminal_exit_bridge_merge(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    terminal_merges: &HashMap<String, String>,
    header: &str,
) -> Option<String> {
    let block = blocks.iter().find(|block| block.name == header)?;
    let (left, right) = conditional_branch_targets(block)?;
    for (continuation, terminal_arm) in [(&left, &right), (&right, &left)] {
        let Some(terminal_merge) = terminal_merges.get(terminal_arm) else {
            continue;
        };
        let Some(merge_block) = blocks.iter().find(|block| block.name == *terminal_merge) else {
            continue;
        };
        let successors = block_successors(merge_block);
        if forest.dominates(header, continuation)
            && successors.len() == 1
            && successors.first() == Some(continuation)
        {
            return Some(continuation.clone());
        }
    }
    None
}

/// Find the first common continuation of a conditional whose alternative paths may terminate through
/// private return clones. Ordinary post-dominance omits this shape because the returns flow straight to
/// the virtual exit, but the continuation is still a valid selection merge: each non-returning path
/// reaches it, and every returning path is already a private structured exit. The search is restricted
/// to one header-dominated, loop-free region and is used only by the bounded terminal-return retry.
pub(in crate::native) fn terminal_exit_convergence(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    header: &str,
) -> Option<String> {
    let header_block = blocks.iter().find(|block| block.name == header)?;
    let arms = conditional_branch_targets(header_block)?;
    let loop_nodes: HashSet<&str> = forest
        .loops
        .iter()
        .flat_map(|loop_info| loop_info.body.iter().map(String::as_str))
        .collect();
    let private_return = |name: &str| {
        blocks
            .iter()
            .find(|block| block.name == name)
            .is_some_and(|block| block.role == BlockRole::TerminalExitReturn)
            && block_ends_in_void_return(blocks, name)
    };
    let candidate_depth = |name: &str| {
        let mut depth = 0usize;
        let mut current = name;
        while let Some(parent) = forest.idom(current) {
            depth += 1;
            current = parent;
        }
        depth
    };

    let mut candidates: Vec<(&str, usize)> = blocks
        .iter()
        .filter(|block| {
            block.name != header
                // Exclude every `%metal2vulkan.lmerge.*` synth block via its role tag rather than a
                // name-prefix decode (both `lmerge` roles: the plain merge and the texitret subtype).
                && !matches!(block.role, BlockRole::LMerge | BlockRole::TerminalExitReturn)
                && !loop_nodes.contains(block.name.as_str())
        })
        .map(|block| {
            let name = block.name.as_str();
            (name, candidate_depth(name))
        })
        .collect();
    candidates.sort_by_key(|(candidate, depth)| (*depth, *candidate));

    for (candidate, _) in candidates {
        let mut terminal_paths = 0usize;
        let mut valid = true;
        for arm in [&arms.0, &arms.1] {
            let mut reached = false;
            let mut saw_terminal = false;
            let mut seen = HashSet::new();
            let mut stack = vec![arm.clone()];
            while let Some(node) = stack.pop() {
                if node == candidate {
                    reached = true;
                    continue;
                }
                if private_return(&node) {
                    saw_terminal = true;
                    continue;
                }
                if !forest.dominates(header, &node) || loop_nodes.contains(node.as_str()) {
                    valid = false;
                    break;
                }
                if !seen.insert(node.clone()) {
                    continue;
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
            if !valid || !reached {
                valid = false;
                break;
            }
            terminal_paths += saw_terminal as usize;
        }
        if valid && terminal_paths > 0 {
            return Some(candidate.to_string());
        }
    }
    None
}
