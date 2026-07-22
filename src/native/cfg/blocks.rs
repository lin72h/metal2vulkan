use super::super::ir::{LlFunction, LlType, LlValue};
use super::super::parse::{strip_comment, LlSwitch};
use super::graph::{
    reachable_before_target, reachable_from, reachable_from_without_revisiting, Cfg,
};
use super::loopforest::analyze;
use super::{BlockRole, BodyBlock, LoopMergeInfo};
use std::collections::{HashMap, HashSet};

/// Prefix of every block label the switch-to-if lowering synthesizes (`%..._{block}_{case}` plus
/// `_merge`/`_cond` suffixes). One constant so the producer formats and any consumer test agree.
const SWITCH_BLOCK_PREFIX: &str = "%metal2vulkan_switch_";
/// Prefix of the synthetic bypass block a switch merge is rerouted through. The SET (label builder)
/// and the role-classifying seam ([`super::structured_emit::role_for_name`]) MUST use the same
/// string, so it lives here once and is shared.
pub(in crate::native) const SWITCH_BYPASS_PREFIX: &str = "%metal2vulkan_switch_bypass_";

pub(in crate::native) fn split_body_blocks(
    lines: &[String],
    entry_name: String,
    named_types: &HashMap<String, LlType>,
) -> Vec<BodyBlock> {
    let mut blocks = Vec::new();
    // AIR-sourced blocks are never synthesized roles (the `%metal2vulkan.*` synth prefixes are kept
    // distinct from any AIR label), so the initial split stamps `Normal`. The instruction lines are a
    // LOCAL accumulator lowered into each block's carrier (`lower_block_carrier` — the one place block
    // instructions are lexed) and then discarded; the carrier is the block's sole substrate.
    let mut cur_name = entry_name;
    let mut cur_lines: Vec<String> = Vec::new();
    for line in lines {
        let trimmed = strip_comment(line).trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(label) = trimmed.strip_suffix(':') {
            if !cur_lines.is_empty() || !blocks.is_empty() {
                let typed =
                    crate::native::tir::lower_block_carrier(&cur_name, &cur_lines, named_types);
                blocks.push(BodyBlock {
                    name: cur_name,
                    role: BlockRole::Normal,
                    typed,
                });
            }
            cur_name = format!("%{label}");
            cur_lines = Vec::new();
        } else {
            cur_lines.push(line.clone());
        }
    }
    let typed = crate::native::tir::lower_block_carrier(&cur_name, &cur_lines, named_types);
    blocks.push(BodyBlock {
        name: cur_name,
        role: BlockRole::Normal,
        typed,
    });
    blocks
}

/// Build a synthesized [`BodyBlock`] with its `typed` carrier populated from its own `lines`. For the
/// scaffolding blocks the structurizer synthesizes (`br`/`ret`/`phi` over primitive, vector, or pointer
/// SSA values), the module named-type table is not needed — those never carry an `extractvalue` into a
/// named struct — so an empty type map lowers them byte-identically to the emit-time re-parse (which
/// uses the real types); BC drift NONE proves the equivalence. Use this for synthetic scaffolding ONLY,
/// never for a clone of an arbitrary AIR block (whose lines may need the real type table). A block whose
/// lines do not lower (no terminator) keeps `typed = None` (the flip then re-lowers it).
pub(in crate::native) fn synthetic_block(
    name: String,
    lines: Vec<String>,
    role: BlockRole,
) -> BodyBlock {
    let typed = crate::native::tir::lower_block_carrier(&name, &lines, &HashMap::new());
    BodyBlock { name, role, typed }
}

pub(in crate::native) fn implicit_entry_block_name(f: &LlFunction) -> String {
    let mut next_numeric = 0;
    for (name, _) in &f.params {
        if let Some(n) = name.strip_prefix('%').and_then(|s| s.parse::<usize>().ok()) {
            if n >= next_numeric {
                next_numeric = n + 1;
            }
        }
    }
    format!("%{next_numeric}")
}

pub(in crate::native) fn infer_branch_merges(
    blocks: &[BodyBlock],
) -> HashMap<(String, String), String> {
    let by_name: HashMap<String, &BodyBlock> = blocks.iter().map(|b| (b.name.clone(), b)).collect();
    let successors: HashMap<String, Vec<String>> = blocks
        .iter()
        .map(|b| (b.name.clone(), block_successors(b)))
        .collect();
    let switch_merges = infer_switch_merges(blocks);
    let loop_merges = infer_loop_merges(blocks);
    let mut merges = HashMap::new();
    loop {
        let mut changed = false;
        for block in blocks {
            let Some((true_label, false_label)) = conditional_branch_targets(block) else {
                continue;
            };
            let key = (true_label.clone(), false_label.clone());
            if merges.contains_key(&key) {
                continue;
            }
            let true_exit = unconditional_branch_target(by_name.get(&true_label).copied());
            let false_exit = unconditional_branch_target(by_name.get(&false_label).copied());
            let true_merge = target_structured_merge(
                &true_label,
                &by_name,
                &merges,
                &switch_merges,
                &loop_merges,
            );
            let false_merge = target_structured_merge(
                &false_label,
                &by_name,
                &merges,
                &switch_merges,
                &loop_merges,
            );
            let true_reaches_false =
                reachable_from(&true_label, &successors).contains(&false_label);
            let false_reaches_true =
                reachable_from(&false_label, &successors).contains(&true_label);
            let is_loop_continue =
                is_loop_continue_branch(&block.name, &true_label, &false_label, &loop_merges);
            let merge = match (true_exit, false_exit) {
                (Some(a), Some(b)) if a == b => Some(a),
                (Some(a), _) if a == false_label => Some(a),
                (_, Some(b)) if b == true_label => Some(b),
                (Some(a), _)
                    if false_merge.as_ref() == Some(&a)
                        && all_paths_reach_target(&false_label, &a, &successors) =>
                {
                    Some(a)
                }
                (_, Some(b))
                    if true_merge.as_ref() == Some(&b)
                        && all_paths_reach_target(&true_label, &b, &successors) =>
                {
                    Some(b)
                }
                _ if !loop_merges.contains_key(&false_label)
                    && !switch_merges.contains_key(&false_label)
                    && conditional_merge_through_shared_body(
                        &true_label,
                        &false_label,
                        &by_name,
                    )
                    .is_some_and(|candidate| {
                        all_paths_reach_target(&true_label, &candidate, &successors)
                            && all_paths_reach_target(&false_label, &candidate, &successors)
                    }) =>
                {
                    conditional_merge_through_shared_body(&true_label, &false_label, &by_name)
                }
                _ if !loop_merges.contains_key(&true_label)
                    && !switch_merges.contains_key(&true_label)
                    && conditional_merge_through_shared_body(
                        &false_label,
                        &true_label,
                        &by_name,
                    )
                    .is_some_and(|candidate| {
                        all_paths_reach_target(&true_label, &candidate, &successors)
                            && all_paths_reach_target(&false_label, &candidate, &successors)
                    }) =>
                {
                    conditional_merge_through_shared_body(&false_label, &true_label, &by_name)
                }
                _ if true_merge.as_ref().zip(false_merge.as_ref()).is_some_and(
                    |(true_merge, false_merge)| {
                        true_merge == false_merge
                            && all_paths_reach_target(&true_label, true_merge, &successors)
                            && all_paths_reach_target(&false_label, true_merge, &successors)
                    },
                ) =>
                {
                    true_merge.clone()
                }
                _ if false_reaches_true
                    && !true_reaches_false
                    && !is_loop_continue
                    && all_paths_reach_target(&false_label, &true_label, &successors) =>
                {
                    Some(true_label.clone())
                }
                _ if true_reaches_false
                    && !false_reaches_true
                    && !is_loop_continue
                    && all_paths_reach_target(&true_label, &false_label, &successors) =>
                {
                    Some(false_label.clone())
                }
                _ if switch_merges.get(&true_label) == Some(&false_label) => {
                    Some(false_label.clone())
                }
                _ if switch_merges.get(&false_label) == Some(&true_label) => {
                    Some(true_label.clone())
                }
                _ if block_is_unreachable(&false_label, &by_name) => reachable_merge_after_header(
                    &block.name,
                    &true_label,
                    blocks,
                    &successors,
                    &by_name,
                ),
                _ if block_is_unreachable(&true_label, &by_name) => reachable_merge_after_header(
                    &block.name,
                    &false_label,
                    blocks,
                    &successors,
                    &by_name,
                ),
                _ if loop_merges.get(&true_label).is_some_and(|info| {
                    info.merge == false_label && info.continue_target != block.name
                }) =>
                {
                    Some(false_label.clone())
                }
                _ if loop_merges.get(&false_label).is_some_and(|info| {
                    info.merge == true_label && info.continue_target != block.name
                }) =>
                {
                    Some(true_label.clone())
                }
                (_, Some(b))
                    if b == true_label
                        && block_role_is_switch_bypass(&false_label, &by_name)
                        && switch_merges.values().any(|merge| merge == &true_label) =>
                {
                    Some(false_label.clone())
                }
                (Some(a), _)
                    if a == false_label
                        && block_role_is_switch_bypass(&true_label, &by_name)
                        && switch_merges.values().any(|merge| merge == &false_label) =>
                {
                    Some(true_label.clone())
                }
                _ if !is_loop_continue => common_reachable_merge_after_header(
                    &block.name,
                    &true_label,
                    &false_label,
                    blocks,
                    &successors,
                    &by_name,
                ),
                _ => None,
            };
            if let Some(merge) = merge {
                let branch_target_is_structured = loop_merges.contains_key(&true_label)
                    || loop_merges.contains_key(&false_label)
                    || switch_merges.contains_key(&true_label)
                    || switch_merges.contains_key(&false_label);
                if !branch_target_is_structured
                    && has_unrepairable_external_predecessor_to_branch_body(
                        &block.name,
                        &true_label,
                        &false_label,
                        &merge,
                        &successors,
                        &by_name,
                    )
                {
                    continue;
                }
                if !branch_target_is_structured
                    && branch_merge_path_has_unrepairable_external_entry(
                        &block.name,
                        &true_label,
                        &false_label,
                        &merge,
                        &successors,
                        &by_name,
                    )
                {
                    continue;
                }
                merges.insert(key, merge);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    merges
}

/// True when `label` names a synthesized switch-bypass block, read from the block's typed
/// [`BlockRole::SwitchBypass`] tag rather than decoding the `SWITCH_BYPASS_PREFIX` on the name.
/// Byte-identical to the retired name match: a block carries `SwitchBypass` iff its name has that
/// prefix (the `role_for_name` / synthesis-site invariant), and a bypass-named branch target always
/// has its block materialized here (created at the one synthesis site and branched to).
fn block_role_is_switch_bypass(label: &str, by_name: &HashMap<String, &BodyBlock>) -> bool {
    by_name
        .get(label)
        .is_some_and(|block| block.role == BlockRole::SwitchBypass)
}

fn is_loop_continue_branch(
    block: &str,
    true_label: &str,
    false_label: &str,
    loop_merges: &HashMap<String, LoopMergeInfo>,
) -> bool {
    loop_merges
        .get(true_label)
        .is_some_and(|info| info.merge == false_label && info.continue_target == block)
        || loop_merges
            .get(false_label)
            .is_some_and(|info| info.merge == true_label && info.continue_target == block)
}

fn conditional_merge_through_shared_body(
    body_label: &str,
    conditional_label: &str,
    by_name: &HashMap<String, &BodyBlock>,
) -> Option<String> {
    let (true_label, false_label) = conditional_branch_targets(by_name.get(conditional_label)?)?;
    if true_label == body_label {
        return Some(false_label);
    }
    if false_label == body_label {
        return Some(true_label);
    }
    None
}

fn common_reachable_merge_after_header(
    header: &str,
    true_label: &str,
    false_label: &str,
    blocks: &[BodyBlock],
    successors: &HashMap<String, Vec<String>>,
    by_name: &HashMap<String, &BodyBlock>,
) -> Option<String> {
    let start = blocks
        .iter()
        .position(|block| block.name == header)
        .map(|idx| idx + 1)?;
    let true_reachable = reachable_from_without_revisiting(true_label, header, successors);
    let false_reachable = reachable_from_without_revisiting(false_label, header, successors);
    blocks
        .iter()
        .skip(start)
        .map(|block| &block.name)
        .find(|candidate| {
            true_reachable.contains(*candidate)
                && false_reachable.contains(*candidate)
                && all_paths_reach_target_or_unreachable(true_label, candidate, successors, by_name)
                && all_paths_reach_target_or_unreachable(
                    false_label,
                    candidate,
                    successors,
                    by_name,
                )
        })
        .cloned()
}

fn reachable_merge_after_header(
    header: &str,
    label: &str,
    blocks: &[BodyBlock],
    successors: &HashMap<String, Vec<String>>,
    by_name: &HashMap<String, &BodyBlock>,
) -> Option<String> {
    let start = blocks
        .iter()
        .position(|block| block.name == header)
        .map(|idx| idx + 1)?;
    let reachable = reachable_from(label, successors);
    blocks
        .iter()
        .skip(start)
        .map(|block| &block.name)
        .find(|candidate| {
            reachable.contains(*candidate)
                && all_paths_reach_target_or_unreachable(label, candidate, successors, by_name)
        })
        .cloned()
}

fn all_paths_reach_target(
    start: &str,
    target: &str,
    successors: &HashMap<String, Vec<String>>,
) -> bool {
    fn walk(
        node: &str,
        target: &str,
        successors: &HashMap<String, Vec<String>>,
        visiting: &mut HashSet<String>,
        memo: &mut HashMap<String, bool>,
    ) -> bool {
        if node == target {
            return true;
        }
        if let Some(&cached) = memo.get(node) {
            return cached;
        }
        if !visiting.insert(node.to_string()) {
            return true;
        }
        let ok = successors
            .get(node)
            .filter(|next| !next.is_empty())
            .is_some_and(|next| {
                next.iter()
                    .all(|succ| walk(succ, target, successors, visiting, memo))
            });
        visiting.remove(node);
        memo.insert(node.to_string(), ok);
        ok
    }

    walk(
        start,
        target,
        successors,
        &mut HashSet::new(),
        &mut HashMap::new(),
    )
}

fn all_paths_reach_target_or_unreachable(
    start: &str,
    target: &str,
    successors: &HashMap<String, Vec<String>>,
    by_name: &HashMap<String, &BodyBlock>,
) -> bool {
    fn walk(
        node: &str,
        target: &str,
        successors: &HashMap<String, Vec<String>>,
        by_name: &HashMap<String, &BodyBlock>,
        visiting: &mut HashSet<String>,
        memo: &mut HashMap<String, bool>,
    ) -> bool {
        if node == target {
            return true;
        }
        if block_is_unreachable(node, by_name) {
            return true;
        }
        if let Some(&cached) = memo.get(node) {
            return cached;
        }
        if !visiting.insert(node.to_string()) {
            return true;
        }
        let ok = successors
            .get(node)
            .filter(|next| !next.is_empty())
            .is_some_and(|next| {
                next.iter()
                    .all(|succ| walk(succ, target, successors, by_name, visiting, memo))
            });
        visiting.remove(node);
        memo.insert(node.to_string(), ok);
        ok
    }

    walk(
        start,
        target,
        successors,
        by_name,
        &mut HashSet::new(),
        &mut HashMap::new(),
    )
}

fn block_is_unreachable(label: &str, by_name: &HashMap<String, &BodyBlock>) -> bool {
    // Read the block's structured terminator from its carrier (the sole substrate). A block with no
    // carrier lowered no terminator, so it cannot be `unreachable` (which IS a terminator) → false.
    by_name.get(label).is_some_and(|block| {
        block.typed.as_ref().is_some_and(|carrier| {
            matches!(
                carrier.terminator,
                crate::native::tir::TirTerminator::Unreachable
            )
        })
    })
}

fn has_unrepairable_external_predecessor_to_branch_body(
    header: &str,
    true_label: &str,
    false_label: &str,
    merge: &str,
    successors: &HashMap<String, Vec<String>>,
    by_name: &HashMap<String, &BodyBlock>,
) -> bool {
    let reachable = reachable_from(header, successors);
    [true_label, false_label]
        .into_iter()
        .filter(|target| **target != *merge)
        .any(|body| {
            let has_external_pred = successors.iter().any(|(pred, targets)| {
                pred != header
                    && pred != true_label
                    && pred != false_label
                    && !reachable.contains(pred)
                    && targets.iter().any(|target| target == body)
            });
            has_external_pred
                && !external_entry_is_repairable_selection_target(
                    body, header, merge, &reachable, successors, by_name,
                )
        })
}

fn branch_merge_path_has_unrepairable_external_entry(
    header: &str,
    true_label: &str,
    false_label: &str,
    merge: &str,
    successors: &HashMap<String, Vec<String>>,
    by_name: &HashMap<String, &BodyBlock>,
) -> bool {
    let header_reachable = reachable_from(header, successors);
    let mut reachable_to_merge = HashSet::new();
    reachable_to_merge.extend(reachable_before_target(true_label, merge, successors));
    reachable_to_merge.extend(reachable_before_target(false_label, merge, successors));
    reachable_to_merge.remove(header);
    reachable_to_merge.remove(merge);

    reachable_to_merge.into_iter().any(|node| {
        if !reachable_from(&node, successors).contains(merge) {
            return false;
        }
        let has_external_pred = successors.iter().any(|(pred, targets)| {
            !header_reachable.contains(pred) && targets.iter().any(|target| target == &node)
        });
        has_external_pred
            && !external_entry_is_repairable_selection_target(
                &node,
                header,
                merge,
                &header_reachable,
                successors,
                by_name,
            )
    })
}

fn external_entry_is_repairable_selection_target(
    node: &str,
    header: &str,
    merge: &str,
    header_reachable: &HashSet<String>,
    successors: &HashMap<String, Vec<String>>,
    by_name: &HashMap<String, &BodyBlock>,
) -> bool {
    if block_starts_with_phi(node, by_name) {
        return false;
    }
    if unconditional_branch_target(by_name.get(node).copied()) == Some(merge.to_string()) {
        return false;
    }
    std::iter::once(header)
        .chain(header_reachable.iter().map(String::as_str))
        .filter_map(|pred| successors.get(pred).map(|targets| (pred, targets)))
        .any(|(pred, targets)| {
            pred != node && targets.len() == 2 && targets.iter().any(|target| target == node)
        })
}

fn block_starts_with_phi(label: &str, by_name: &HashMap<String, &BodyBlock>) -> bool {
    // Read the block's first instruction from its carrier (the sole substrate). A block with no
    // carrier lowered no instructions, so its first inst is not a phi → false.
    by_name.get(label).is_some_and(|block| {
        block
            .typed
            .as_ref()
            .and_then(|carrier| carrier.insts.first())
            .is_some_and(|inst| inst.is_phi())
    })
}

fn target_structured_merge(
    label: &str,
    by_name: &HashMap<String, &BodyBlock>,
    branch_merges: &HashMap<(String, String), String>,
    switch_merges: &HashMap<String, String>,
    loop_merges: &HashMap<String, LoopMergeInfo>,
) -> Option<String> {
    if let Some(merge) = switch_merges.get(label) {
        return Some(merge.clone());
    }
    if let Some(info) = loop_merges.get(label) {
        return Some(info.merge.clone());
    }
    let (true_label, false_label) = conditional_branch_targets(by_name.get(label)?)?;
    branch_merges.get(&(true_label, false_label)).cloned()
}

pub(in crate::native) fn infer_loop_merges(blocks: &[BodyBlock]) -> HashMap<String, LoopMergeInfo> {
    let order: HashMap<String, usize> = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.name.clone(), i))
        .collect();
    let successors: HashMap<String, Vec<String>> = blocks
        .iter()
        .map(|b| (b.name.clone(), block_successors(b)))
        .collect();
    let mut predecessors: HashMap<String, Vec<String>> = HashMap::new();
    for (block, targets) in &successors {
        for target in targets {
            predecessors
                .entry(target.clone())
                .or_default()
                .push(block.clone());
        }
    }

    // Real dominator tree (CHK) over the source CFG's real-block edges — the single
    // dominance oracle, replacing the local path-DFS `dominates` this module used to carry.
    // `None` only when `blocks` is empty, in which case the loop below never runs.
    let doms = Cfg::from_blocks(blocks).map(|cfg| cfg.dominators());
    let mut merges = HashMap::new();
    for header in blocks {
        let Some(header_idx) = order.get(&header.name).copied() else {
            continue;
        };
        let header_reachable = reachable_from(&header.name, &successors);
        let back_preds = predecessors
            .get(&header.name)
            .into_iter()
            .flat_map(|preds| preds.iter())
            .filter(|pred| {
                header_reachable.contains(*pred)
                    && doms
                        .as_ref()
                        .is_some_and(|d| d.dominates(&header.name, pred))
                    && order
                        .get(*pred)
                        .copied()
                        .is_some_and(|idx| idx >= header_idx)
            })
            .cloned()
            .collect::<Vec<_>>();
        if back_preds.is_empty() {
            continue;
        }

        let loop_nodes = natural_loop_nodes(&header.name, &back_preds, &predecessors);
        let Some(continue_target) =
            infer_continue_target(&header.name, &back_preds, &loop_nodes, &successors, &order)
        else {
            continue;
        };
        let Some(merge) = infer_loop_exit(&header.name, &loop_nodes, blocks, &successors, &order)
        else {
            continue;
        };
        merges.insert(
            header.name.clone(),
            LoopMergeInfo {
                merge,
                continue_target,
            },
        );
    }
    merges
}

fn natural_loop_nodes(
    header: &str,
    back_preds: &[String],
    predecessors: &HashMap<String, Vec<String>>,
) -> HashSet<String> {
    let mut nodes = HashSet::from([header.to_string()]);
    let mut stack = back_preds.to_vec();
    while let Some(block) = stack.pop() {
        if !nodes.insert(block.clone()) || block == header {
            continue;
        }
        if let Some(preds) = predecessors.get(&block) {
            stack.extend(preds.iter().cloned());
        }
    }
    nodes
}

fn infer_continue_target(
    header: &str,
    back_preds: &[String],
    loop_nodes: &HashSet<String>,
    successors: &HashMap<String, Vec<String>>,
    order: &HashMap<String, usize>,
) -> Option<String> {
    if back_preds.len() == 1 && back_preds[0] != header {
        return Some(back_preds[0].clone());
    }

    let reachable = successors
        .get(header)?
        .iter()
        .map(|target| (target, reachable_from(target, successors)))
        .collect::<Vec<_>>();
    reachable
        .iter()
        .filter(|(target, _)| loop_nodes.contains(*target))
        .find(|(_, seen)| back_preds.iter().all(|pred| seen.contains(pred)))
        .map(|(target, _)| (*target).clone())
        .or_else(|| {
            back_preds
                .iter()
                .filter(|pred| pred.as_str() != header)
                .min_by_key(|pred| order.get(*pred).copied().unwrap_or(usize::MAX))
                .cloned()
        })
}

fn infer_loop_exit(
    header: &str,
    loop_nodes: &HashSet<String>,
    blocks: &[BodyBlock],
    successors: &HashMap<String, Vec<String>>,
    order: &HashMap<String, usize>,
) -> Option<String> {
    let mut exits = HashSet::new();
    for node in loop_nodes {
        for target in successors.get(node).into_iter().flatten() {
            if !loop_nodes.contains(target) {
                exits.insert(target.clone());
            }
        }
    }
    if exits.len() == 1 {
        return exits.into_iter().next();
    }
    let header_idx = order.get(header).copied()?;
    let reachable = exits
        .iter()
        .map(|exit| reachable_from(exit, successors))
        .collect::<Vec<_>>();
    blocks
        .iter()
        .skip(header_idx + 1)
        .map(|block| &block.name)
        .find(|candidate| reachable.iter().all(|seen| seen.contains(*candidate)))
        .cloned()
}

pub(in crate::native) fn infer_switch_merges(blocks: &[BodyBlock]) -> HashMap<String, String> {
    let by_name: HashMap<String, &BodyBlock> = blocks.iter().map(|b| (b.name.clone(), b)).collect();
    let order: HashMap<String, usize> = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.name.clone(), i))
        .collect();
    let successors: HashMap<String, Vec<String>> = blocks
        .iter()
        .map(|b| (b.name.clone(), block_successors(b)))
        .collect();
    let mut merges = HashMap::new();
    for block in blocks {
        let Some(targets) = switch_targets(block) else {
            continue;
        };
        let live_targets = targets
            .iter()
            .filter(|target| !block_is_unreachable(target, &by_name))
            .collect::<Vec<_>>();
        if live_targets.is_empty() {
            continue;
        }
        let reachable = live_targets
            .iter()
            .map(|target| (*target, reachable_from(target, &successors)))
            .collect::<Vec<_>>();
        let Some(start) = order.get(&block.name).map(|i| i + 1) else {
            continue;
        };
        for candidate in blocks.iter().skip(start) {
            if reachable
                .iter()
                .all(|(_, seen)| seen.contains(&candidate.name))
                && reachable
                    .iter()
                    .all(|(target, _)| all_paths_reach_target(target, &candidate.name, &successors))
            {
                merges.insert(block.name.clone(), candidate.name.clone());
                break;
            }
        }
    }
    merges
}

pub(in crate::native) fn lower_unstructured_switches(blocks: &[BodyBlock]) -> Vec<BodyBlock> {
    lower_switches(blocks, false)
}

/// Lower every switch that directly exits an enclosing natural loop while another arm remains in that
/// loop. SPIR-V cannot use a loop-breaking target as an `OpSwitch` case construct; a branch ladder
/// makes each equality test an ordinary selection, where the loop-exit arm is a legal structured exit.
/// This is used only by the reject-triggered loop-exit structurizer tier; the default lowering keeps
/// its existing byte behavior.
pub(in crate::native) fn lower_loop_exit_switches(blocks: &[BodyBlock]) -> Vec<BodyBlock> {
    lower_switches(blocks, true)
}

fn lower_switches(blocks: &[BodyBlock], force_loop_exit_ladders: bool) -> Vec<BodyBlock> {
    let normalized = normalize_switch_bypass_merges(blocks);
    let blocks = normalized.as_slice();
    let switch_merges = infer_switch_merges(blocks);
    let loop_exit_ladders = if force_loop_exit_ladders {
        loop_exit_switch_headers(blocks)
    } else {
        Default::default()
    };
    let successors: HashMap<String, Vec<String>> = blocks
        .iter()
        .map(|b| (b.name.clone(), block_successors(b)))
        .collect();
    let mut lowered = Vec::new();
    let mut phi_rewrites: HashMap<String, HashMap<String, Vec<String>>> = HashMap::new();
    let mut synthetic_index = 0usize;

    for block in blocks {
        // SPIR-V switch case constructs cannot converge at a shared intermediate block when another
        // case can bypass it. A branch ladder lets ordinary branch merge inference model that shape.
        // Source the block's switch from the typed carrier (`switch: switch_emit(term)`, populated for
        // every production switch block).
        let Some(switch) = switch_of(block) else {
            lowered.push(block.clone());
            continue;
        };
        let needs_ladder = loop_exit_ladders.contains(&block.name)
            || !switch_merges.contains_key(&block.name)
            || switch_case_order_requires_ladder(&block.name, &switch, &switch_merges, &successors);
        if !needs_ladder {
            lowered.push(block.clone());
            continue;
        }
        let Some(selector_ty) = text_type(&switch.selector.ty) else {
            lowered.push(block.clone());
            continue;
        };
        let Some(selector_value) = text_value(&switch.selector.value) else {
            lowered.push(block.clone());
            continue;
        };
        let cases = switch
            .cases
            .iter()
            .map(|(value, label)| text_value(value).map(|value| (value, label.clone())))
            .collect::<Option<Vec<_>>>();
        let Some(cases) = cases else {
            lowered.push(block.clone());
            continue;
        };

        let block_index = synthetic_index;
        synthetic_index += 1;
        if cases.is_empty() {
            let terminator = format!("br label {}", switch.default_label);
            // The block reuses the original switch block's instruction PREFIX + a synthetic `br` — build
            // its carrier from the source's typed prefix (`block.typed.insts` == the block's non-terminator
            // instructions) + the synthetic tail (no module named-types needed).
            let typed = block.typed.as_ref().and_then(|p| {
                crate::native::tir::lower_block_carrier_with_prefix(
                    &block.name,
                    p,
                    std::slice::from_ref(&terminator),
                )
            });
            lowered.push(BodyBlock {
                name: block.name.clone(),
                // Reuses the original block's name → preserves its role (a switch block is `Normal`).
                role: block.role,
                typed,
            });
            continue;
        }

        let mut target_predecessors: HashMap<String, Vec<String>> = HashMap::new();
        for (case_index, (case_value, label)) in cases.iter().enumerate() {
            let name = if case_index == 0 {
                block.name.clone()
            } else {
                format!("{SWITCH_BLOCK_PREFIX}{block_index}_{case_index}")
            };
            add_unique(
                target_predecessors.entry(label.clone()).or_default(),
                name.clone(),
            );
            let next_label = if case_index + 1 == cases.len() {
                switch.default_label.clone()
            } else {
                format!("{SWITCH_BLOCK_PREFIX}{block_index}_{}", case_index + 1)
            };
            let false_label = if case_index == 0 || case_index + 1 == cases.len() {
                next_label.clone()
            } else {
                format!("{SWITCH_BLOCK_PREFIX}{block_index}_{case_index}_merge")
            };
            let cond = format!("{SWITCH_BLOCK_PREFIX}{block_index}_{case_index}_cond");
            let tail = vec![
                format!("{cond} = icmp eq {selector_ty} {selector_value}, {case_value}"),
                format!("br i1 {cond}, label {label}, label {false_label}"),
            ];
            // case 0 copies the original switch block's instruction PREFIX (build its carrier from the
            // source's typed prefix + the synthetic `icmp`/`br` tail); every other case is a purely
            // synthetic block (carrier lowered directly from its own lines with empty named-types).
            let typed = if case_index == 0 {
                block.typed.as_ref().and_then(|p| {
                    crate::native::tir::lower_block_carrier_with_prefix(&name, p, &tail)
                })
            } else {
                crate::native::tir::lower_block_carrier(&name, &tail, &HashMap::new())
            };
            // case 0 reuses the original block's name (preserve its role); every other case is a
            // fresh `%metal2vulkan_switch_*` synth name, which is never a terminal-return clone.
            let role = if case_index == 0 {
                block.role
            } else {
                BlockRole::Normal
            };
            lowered.push(BodyBlock { name, role, typed });
            if false_label != next_label {
                let merge_lines = vec![format!("br label {next_label}")];
                let typed = crate::native::tir::lower_block_carrier(
                    &false_label,
                    &merge_lines,
                    &HashMap::new(),
                );
                lowered.push(BodyBlock {
                    name: false_label,
                    role: BlockRole::Normal,
                    typed,
                });
            }
        }
        let default_pred = if cases.len() == 1 {
            block.name.clone()
        } else {
            format!("{SWITCH_BLOCK_PREFIX}{block_index}_{}", cases.len() - 1)
        };
        add_unique(
            target_predecessors
                .entry(switch.default_label.clone())
                .or_default(),
            default_pred,
        );
        record_phi_rewrites(&mut phi_rewrites, &block.name, target_predecessors);
    }

    rewrite_lowered_switch_target_phis(&mut lowered, &phi_rewrites);
    lowered
}

/// Return switches in a natural-loop body that branch directly to that loop's sole exit while at least
/// one other arm stays in the body. Such a `switch` is source-valid, but its exiting case cannot be a
/// SPIR-V case construct; [`lower_loop_exit_switches`] lowers it to ordinary conditional branches.
fn loop_exit_switch_headers(blocks: &[BodyBlock]) -> HashSet<String> {
    let forest = analyze(blocks);
    let mut headers = HashSet::new();
    for loop_info in &forest.loops {
        let [merge] = loop_info.exits.as_slice() else {
            continue;
        };
        let body: HashSet<&str> = loop_info.body.iter().map(String::as_str).collect();
        for block in blocks {
            if block.name == loop_info.header
                || !body.contains(block.name.as_str())
                || !block.typed.as_ref().is_some_and(|t| {
                    matches!(
                        t.terminator,
                        crate::native::tir::TirTerminator::Switch { .. }
                    )
                })
            {
                continue;
            }
            let successors = block_successors(block);
            if successors.iter().any(|target| target == merge)
                && successors
                    .iter()
                    .any(|target| body.contains(target.as_str()))
            {
                headers.insert(block.name.clone());
            }
        }
    }
    headers
}

/// A block's switch, read from its typed carrier (`typed.switch`, `switch_emit(term)` — populated for
/// every production switch block). `None` for a non-switch block or a block with no carrier.
fn switch_of(block: &BodyBlock) -> Option<LlSwitch> {
    block.typed.as_ref()?.switch.clone()
}

fn switch_case_order_requires_ladder(
    block_name: &str,
    switch: &LlSwitch,
    switch_merges: &HashMap<String, String>,
    successors: &HashMap<String, Vec<String>>,
) -> bool {
    let case_targets = switch_case_targets_in_order(switch);
    let merge = switch_merges.get(block_name);
    let order = case_targets
        .iter()
        .enumerate()
        .map(|(idx, target)| (target.clone(), idx))
        .collect::<HashMap<_, _>>();
    if !order.contains_key(&switch.default_label)
        && !first_reachable_case_targets(
            &switch.default_label,
            merge.map(String::as_str),
            Some(block_name),
            &order,
            successors,
        )
        .is_empty()
    {
        return true;
    }
    if case_targets.len() < 2 {
        return false;
    }
    for (idx, target) in case_targets.iter().enumerate() {
        let reached_targets = first_reachable_case_targets(
            target,
            merge.map(String::as_str),
            None,
            &order,
            successors,
        );
        for reached in reached_targets {
            let Some(reached_idx) = order.get(&reached).copied() else {
                continue;
            };
            if idx + 1 != reached_idx {
                return true;
            }
        }
    }
    false
}

fn first_reachable_case_targets(
    start: &str,
    merge: Option<&str>,
    stop: Option<&str>,
    case_order: &HashMap<String, usize>,
    successors: &HashMap<String, Vec<String>>,
) -> HashSet<String> {
    let mut reached = HashSet::new();
    let mut seen = HashSet::new();
    let mut stack = successors.get(start).cloned().unwrap_or_default();
    while let Some(node) = stack.pop() {
        let is_boundary = merge == Some(node.as_str()) || stop == Some(node.as_str());
        if is_boundary || !seen.insert(node.clone()) {
            continue;
        }
        if node != start && case_order.contains_key(&node) {
            reached.insert(node);
            continue;
        }
        if let Some(next) = successors.get(&node) {
            stack.extend(next.iter().cloned());
        }
    }
    reached
}

fn switch_case_targets_in_order(switch: &LlSwitch) -> Vec<String> {
    let mut targets = Vec::new();
    let mut seen = HashSet::new();
    for (_, label) in &switch.cases {
        if seen.insert(label.clone()) {
            targets.push(label.clone());
        }
    }
    targets
}

fn record_phi_rewrites(
    rewrites: &mut HashMap<String, HashMap<String, Vec<String>>>,
    old_pred: &str,
    target_predecessors: HashMap<String, Vec<String>>,
) {
    for (target, new_preds) in target_predecessors {
        let pred_rewrites = rewrites.entry(target).or_default();
        let rewritten = pred_rewrites.entry(old_pred.to_string()).or_default();
        for new_pred in new_preds {
            add_unique(rewritten, new_pred);
        }
    }
}

fn rewrite_lowered_switch_target_phis(
    blocks: &mut [BodyBlock],
    rewrites: &HashMap<String, HashMap<String, Vec<String>>>,
) {
    for block in blocks {
        let Some(block_rewrites) = rewrites.get(&block.name) else {
            continue;
        };
        // Expand each target phi's incoming predecessors on the carrier directly (a switch predecessor
        // that lowered to a comparison ladder fans out into the ladder's leaf blocks).
        if let Some(t) = &mut block.typed {
            t.expand_phi_predecessors(block_rewrites);
        }
    }
}

fn add_unique(values: &mut Vec<String>, value: String) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn normalize_switch_bypass_merges(blocks: &[BodyBlock]) -> Vec<BodyBlock> {
    let mut normalized = blocks.to_vec();
    let mut synthetic_index = 0usize;
    while normalize_one_switch_bypass_merge(&mut normalized, &mut synthetic_index) {}
    normalized
}

fn normalize_one_switch_bypass_merge(
    blocks: &mut Vec<BodyBlock>,
    synthetic_index: &mut usize,
) -> bool {
    let unreachable = blocks
        .iter()
        .filter(|block| {
            block.typed.as_ref().is_some_and(|t| {
                matches!(t.terminator, crate::native::tir::TirTerminator::Unreachable)
            })
        })
        .map(|block| block.name.clone())
        .collect::<HashSet<_>>();
    let order: HashMap<String, usize> = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.name.clone(), i))
        .collect();
    let successors: HashMap<String, Vec<String>> = blocks
        .iter()
        .map(|b| (b.name.clone(), block_successors(b)))
        .collect();

    for block in blocks.clone() {
        let Some(targets) = switch_targets(&block) else {
            continue;
        };
        let live_targets = targets
            .iter()
            .filter(|target| !unreachable.contains(*target))
            .cloned()
            .collect::<Vec<_>>();
        if live_targets.is_empty() {
            continue;
        }
        let reachable = live_targets
            .iter()
            .map(|target| (target, reachable_from(target, &successors)))
            .collect::<Vec<_>>();
        let Some(start) = order.get(&block.name).map(|i| i + 1) else {
            continue;
        };
        let mut intermediate = None;
        let mut merge = None;
        for candidate in blocks.iter().skip(start) {
            if !reachable
                .iter()
                .all(|(_, seen)| seen.contains(&candidate.name))
            {
                continue;
            }
            if reachable
                .iter()
                .all(|(target, _)| all_paths_reach_target(target, &candidate.name, &successors))
            {
                merge = Some(candidate.name.clone());
                break;
            }
            intermediate.get_or_insert_with(|| candidate.name.clone());
        }
        let (Some(intermediate), Some(merge)) = (intermediate, merge) else {
            continue;
        };
        let intermediate_exit = blocks
            .iter()
            .find(|block| block.name == intermediate)
            .and_then(|block| unconditional_branch_target(Some(block)));
        if intermediate_exit != Some(merge.clone()) {
            continue;
        }
        if rewrite_switch_bypass_merge(
            blocks,
            &live_targets,
            &successors,
            &intermediate,
            &merge,
            *synthetic_index,
        ) {
            *synthetic_index += 1;
            return true;
        }
    }
    false
}

fn rewrite_switch_bypass_merge(
    blocks: &mut Vec<BodyBlock>,
    live_targets: &[String],
    successors: &HashMap<String, Vec<String>>,
    intermediate: &str,
    merge: &str,
    switch_index: usize,
) -> bool {
    let reachable_from_cases = live_targets
        .iter()
        .flat_map(|target| reachable_from(target, successors))
        .collect::<HashSet<_>>();
    let bypass_preds = blocks
        .iter()
        .filter(|block| block.name != intermediate && reachable_from_cases.contains(&block.name))
        .filter(|block| block_successors(block).iter().any(|target| target == merge))
        .map(|block| block.name.clone())
        .collect::<Vec<_>>();
    if bypass_preds.is_empty() {
        return false;
    }

    let Some(intermediate_idx) = blocks.iter().position(|block| block.name == intermediate) else {
        return false;
    };
    let Some(merge_idx) = blocks.iter().position(|block| block.name == merge) else {
        return false;
    };

    let bypass_set = bypass_preds.iter().cloned().collect::<HashSet<_>>();

    // The intermediate block's phi result names, read from its carrier — the value an inner merge-phi
    // incoming must name for the bypass topology to hold.
    let intermediate_phi_results: HashSet<String> = match &blocks[intermediate_idx].typed {
        Some(t) => t
            .insts
            .iter()
            .filter(|inst| inst.is_phi())
            .filter_map(|inst| inst.result.clone())
            .collect(),
        None => return false,
    };

    // Plan the intermediate-phi additions off the merge block's CARRIER (no `.lines` text). For each
    // merge phi: it must merge an incoming from the intermediate whose value names an intermediate phi,
    // and each of its bypass-predecessor incomings becomes a `[value, synthetic_bypass_label]` addition
    // on that intermediate phi. Keyed by the intermediate phi's result. An AGGREGATE merge phi
    // (`phi_incoming: None`) carries no readable incoming list, so the transform declines (the string
    // path bailed on it too — its intermediate-incoming / bypass typed-value reads failed); this class
    // is unused (gated by BC/G4/G5-PV), aggregate phis route to retry regardless.
    let typed_additions: HashMap<String, Vec<(LlValue, String)>> = {
        let Some(merge_carrier) = &blocks[merge_idx].typed else {
            return false;
        };
        let merge_phis: Vec<&crate::native::tir::TirInst> = merge_carrier
            .insts
            .iter()
            .filter(|inst| inst.is_phi())
            .collect();
        if merge_phis.is_empty() {
            return false;
        }
        let mut additions: HashMap<String, Vec<(LlValue, String)>> = HashMap::new();
        for merge_phi in merge_phis {
            let Some((_, incoming)) = &merge_phi.phi_incoming else {
                return false;
            };
            let Some((inner_value, _)) = incoming.iter().find(|(_, pred)| pred == intermediate)
            else {
                return false;
            };
            let LlValue::Local(inner_name) = inner_value else {
                return false;
            };
            if !intermediate_phi_results.contains(inner_name) {
                return false;
            }
            for (value, pred) in incoming
                .iter()
                .filter(|(_, pred)| bypass_set.contains(pred))
            {
                let label = synthetic_bypass_label(switch_index, pred);
                additions
                    .entry(inner_name.clone())
                    .or_default()
                    .push((value.clone(), label));
            }
        }
        additions
    };
    if typed_additions.is_empty() {
        return false;
    }

    // Pre-check the terminator redirects are all applicable (transactional, no mutation yet): a bypass
    // pred's carrier terminator must be a `br`/`br i1` (the string `rewrite_text_successor` declined a
    // `switch`; a `ret`/`unreachable` has no successor, so is never a bypass pred). Bail if any is not.
    let mut redirects = Vec::with_capacity(bypass_preds.len());
    for pred in &bypass_preds {
        let synthetic = synthetic_bypass_label(switch_index, pred);
        let Some(pred_idx) = blocks.iter().position(|block| &block.name == pred) else {
            return false;
        };
        let redirectable = blocks[pred_idx].typed.as_ref().is_some_and(|t| {
            matches!(
                t.terminator,
                crate::native::tir::TirTerminator::Br(_)
                    | crate::native::tir::TirTerminator::BrCond { .. }
            )
        });
        if !redirectable {
            return false;
        }
        redirects.push((pred_idx, pred.clone(), synthetic));
    }

    let mut insert_after = HashMap::new();
    for (pred_idx, pred, synthetic) in redirects {
        // Redirect the terminator successor merge -> synthetic bypass on the carrier (the typed dual of
        // the string `rewrite_text_successor`).
        if let Some(t) = &mut blocks[pred_idx].typed {
            t.redirect_successor(merge, &synthetic);
        }
        insert_after.insert(pred, synthetic);
    }
    // Extend the intermediate carrier's phis from the typed additions (merge-carrier-sourced values).
    // Every `typed_additions` key names an intermediate phi (checked against `intermediate_phi_results`
    // above), so each `append_phi_incoming` targets an existing carrier phi.
    if let Some(t) = &mut blocks[intermediate_idx].typed {
        for (result, extra) in &typed_additions {
            for (value, pred) in extra {
                t.append_phi_incoming(result, value.clone(), pred);
            }
        }
    }
    // Drop the bypass incomings from the merge carrier's phis (the same `!bypass_set` filter): a merge
    // phi that had a bypass incoming is non-aggregate (the plan loop bailed otherwise), so
    // `rebuild_phi_incomings` reproduces the same incoming set / operands / uses a re-lower of the
    // bypass-dropped line would.
    if let Some(t) = &mut blocks[merge_idx].typed {
        t.rebuild_phi_incomings(|pred| !bypass_set.contains(pred));
    }

    let mut rebuilt = Vec::with_capacity(blocks.len() + insert_after.len());
    for block in blocks.drain(..) {
        let synthetic = insert_after.get(&block.name).cloned();
        rebuilt.push(block);
        if let Some(synthetic) = synthetic {
            let lines = vec![format!("br label {intermediate}")];
            // Purely synthetic single-`br` bypass block — lower its carrier directly (empty named-types
            // is exact for a `br`).
            let typed =
                crate::native::tir::lower_block_carrier(&synthetic, &lines, &HashMap::new());
            rebuilt.push(BodyBlock {
                name: synthetic,
                // A `%metal2vulkan_switch_bypass_*` block — the one SwitchBypass synthesis site;
                // `infer_branch_merges` reads this role instead of decoding the bypass-prefixed name.
                role: BlockRole::SwitchBypass,
                typed,
            });
        }
    }
    *blocks = rebuilt;
    true
}

fn synthetic_bypass_label(switch_index: usize, pred: &str) -> String {
    format!(
        "{SWITCH_BYPASS_PREFIX}{}_{}",
        switch_index,
        pred.trim_start_matches('%')
    )
}

fn text_type(ty: &LlType) -> Option<String> {
    match ty {
        LlType::Bool => Some("i1".to_string()),
        LlType::Int(bits) => Some(format!("i{bits}")),
        _ => None,
    }
}

fn text_value(value: &LlValue) -> Option<String> {
    match value {
        LlValue::Local(name) | LlValue::Global(name) => Some(name.clone()),
        LlValue::Bool(value) => Some(value.to_string()),
        LlValue::Int(value) | LlValue::Hex(value) => Some(value.to_string()),
        LlValue::SignedInt(value) => Some(value.to_string()),
        _ => None,
    }
}

/// The successor labels of a typed terminator, applying this CFG layer's switch convention (a sorted,
/// deduped target set). Shared by the carrier read and the line fallback so both agree exactly.
fn successors_of_terminator(term: &crate::native::tir::TirTerminator) -> Vec<String> {
    let is_switch = matches!(term, crate::native::tir::TirTerminator::Switch { .. });
    let mut succ: Vec<String> = term.successors().into_iter().map(String::from).collect();
    if is_switch {
        succ.sort();
        succ.dedup();
    }
    succ
}

/// Conditional-branch targets `(true, false)` of a block, read from its typed carrier (the sole
/// substrate). `None` for a non-conditional terminator or a block with no carrier.
pub(in crate::native) fn conditional_branch_targets(block: &BodyBlock) -> Option<(String, String)> {
    match &block.typed.as_ref()?.terminator {
        crate::native::tir::TirTerminator::BrCond { t, f, .. } => Some((t.clone(), f.clone())),
        _ => None,
    }
}

/// Successor labels of a block, read from its typed carrier (the sole substrate). Empty for a block
/// with no carrier (no terminator lowered).
pub(in crate::native) fn block_successors(block: &BodyBlock) -> Vec<String> {
    match &block.typed {
        Some(carrier) => successors_of_terminator(&carrier.terminator),
        None => Vec::new(),
    }
}

fn switch_targets(block: &BodyBlock) -> Option<Vec<String>> {
    match &block.typed.as_ref()?.terminator {
        crate::native::tir::TirTerminator::Switch { default, cases, .. } => {
            let mut targets = Vec::with_capacity(cases.len() + 1);
            targets.push(default.clone());
            targets.extend(cases.iter().map(|(_, label)| label.clone()));
            targets.sort();
            targets.dedup();
            Some(targets)
        }
        _ => None,
    }
}

fn unconditional_branch_target(block: Option<&BodyBlock>) -> Option<String> {
    match &block?.typed.as_ref()?.terminator {
        crate::native::tir::TirTerminator::Br(t) => Some(t.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(name: &str, terminator: &str) -> BodyBlock {
        synthetic_block(
            name.to_string(),
            vec![terminator.to_string()],
            BlockRole::Normal,
        )
    }

    #[test]
    fn implicit_entry_block_name_uses_next_numeric_value() {
        let function = LlFunction {
            name: "main".to_string(),
            ret: LlType::Void,
            params: vec![
                ("%0".to_string(), LlType::Int(32)),
                ("%named".to_string(), LlType::Int(32)),
                ("%6".to_string(), LlType::Int(32)),
            ],
            blocks: Vec::new(),
        };

        assert_eq!(implicit_entry_block_name(&function), "%7");

        let function = LlFunction {
            name: "main".to_string(),
            ret: LlType::Void,
            params: vec![("%named".to_string(), LlType::Int(32))],
            blocks: Vec::new(),
        };

        assert_eq!(implicit_entry_block_name(&function), "%0");
    }

    #[test]
    fn branch_merge_ignores_backedge_reachability_through_header() {
        let blocks = vec![
            block("entry", "br label %header"),
            block("%header", "br i1 %cond, label %latch, label %body"),
            block("%body", "br label %inner"),
            block("%inner", "br i1 %done, label %latch, label %inner"),
            block("%latch", "br i1 %again, label %header, label %exit"),
            block("%exit", "ret void"),
        ];
        let merges = infer_branch_merges(&blocks);

        assert_eq!(
            merges.get(&("%latch".to_string(), "%body".to_string())),
            Some(&"%latch".to_string())
        );
    }

    #[test]
    fn branch_merge_ignores_latch_cycle_through_predecessor_to_header() {
        let blocks = vec![
            block("%95", "br i1 %outer, label %96, label %113"),
            block("%96", "br label %113"),
            block("%113", "br i1 %zero_order, label %122, label %115"),
            block("%115", "br label %125"),
            block("%122", "br i1 %next_y, label %44, label %47"),
            block("%125", "br i1 %inner_done, label %122, label %125"),
            block("%44", "br label %41"),
            block("%47", "br label %95"),
            block("%41", "br label %exit"),
            block("%exit", "ret void"),
        ];
        let merges = infer_branch_merges(&blocks);

        assert_eq!(
            merges.get(&("%122".to_string(), "%115".to_string())),
            Some(&"%122".to_string())
        );
    }

    #[test]
    fn branch_merge_keeps_cubemap_latch_before_accumulation_body() {
        let blocks = vec![
            block("%3", "br label %29"),
            block("%29", "br i1 %empty, label %41, label %33"),
            block("%31", "br i1 %zero_order, label %138, label %139"),
            block("%33", "br label %47"),
            block("%41", "br i1 %face_done, label %31, label %29"),
            block("%44", "br i1 %row_done, label %41, label %33"),
            block(
                "%47",
                "switch i32 %face, label %69 [ i32 0, label %54 i32 1, label %57 i32 2, label %60 i32 3, label %63 i32 4, label %66 ]",
            ),
            block("%54", "br label %73"),
            block("%57", "br label %73"),
            block("%60", "br label %73"),
            block("%63", "br label %73"),
            block("%66", "br label %73"),
            block("%69", "br label %73"),
            block("%73", "br i1 %order1, label %86, label %87"),
            block("%86", "br label %87"),
            block("%87", "br i1 %order2, label %88, label %95"),
            block("%88", "br label %95"),
            block("%95", "br i1 %order3, label %96, label %113"),
            block("%96", "br label %113"),
            block("%113", "br i1 %zero_order, label %122, label %115"),
            block("%115", "br label %125"),
            block("%122", "br i1 %texel_done, label %44, label %47"),
            block("%125", "br i1 %coeff_done, label %122, label %125"),
            block("%138", "ret void"),
            block("%139", "br i1 %out_done, label %138, label %139"),
        ];
        let lowered = lower_unstructured_switches(&blocks);
        let merges = infer_branch_merges(&lowered);

        assert_eq!(
            merges.get(&("%122".to_string(), "%115".to_string())),
            Some(&"%122".to_string()),
            "{merges:?}"
        );
    }

    /// A switch in a natural-loop body whose default directly breaks the loop cannot remain an
    /// `OpSwitch`: SPIR-V requires every case construct to remain under the switch header. The
    /// reject-only loop-exit lowering turns it into a comparison ladder, preserving the original
    /// default/case targets while making the loop-exit edge an ordinary conditional branch.
    #[test]
    fn loop_exit_switch_lowers_to_a_branch_ladder() {
        let blocks = vec![
            block("%entry", "br label %head"),
            block("%head", "br i1 %run, label %sw, label %exit"),
            block(
                "%sw",
                "switch i32 %tag, label %exit [ i32 0, label %latch i32 4, label %latch ]",
            ),
            block("%latch", "br label %head"),
            block("%exit", "ret void"),
        ];
        let lowered = lower_loop_exit_switches(&blocks);
        assert!(
            lowered.len() > blocks.len(),
            "the forced ladder adds a comparison block: {lowered:#?}"
        );
        assert!(
            !lowered.iter().any(|block| {
                block.typed.as_ref().is_some_and(|t| {
                    matches!(
                        t.terminator,
                        crate::native::tir::TirTerminator::Switch { .. }
                    )
                })
            }),
            "loop-exiting switch must be lowered: {lowered:#?}"
        );
        let first = lowered.iter().find(|block| block.name == "%sw").unwrap();
        assert!(
            block_successors(first)
                .iter()
                .any(|target| target == "%latch"),
            "first equality branch still reaches the case target: {first:#?}"
        );
        assert!(
            lowered.iter().any(|block| block_successors(block)
                .iter()
                .any(|target| target == "%exit")),
            "the ladder preserves the default loop-exit target: {lowered:#?}"
        );
    }
}
