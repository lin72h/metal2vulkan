use super::super::ir::{LlFunction, LlType, LlValue};
use super::super::parse::{strip_comment, LlSwitch};
use super::graph::{reachable_from, Cfg};
use super::loopforest::{analyze, post_idom};
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
                    crate::native::tir::lower_block_carrier(&cur_name, &cur_lines, named_types)
                        .map(Into::into);
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
    let typed =
        crate::native::tir::lower_block_carrier(&cur_name, &cur_lines, named_types).map(Into::into);
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
    let typed =
        crate::native::tir::lower_block_carrier(&name, &lines, &HashMap::new()).map(Into::into);
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

/// Index-based reachability scratch for branch-merge inference. Boolean vectors avoid cloning block
/// names into thousands of short-lived hash sets and keep repeated graph walks allocator-stable.
struct IndexedReachability {
    index: HashMap<String, usize>,
    names: Vec<String>,
    successors: Vec<Vec<usize>>,
    unreachable: Vec<bool>,
}

impl IndexedReachability {
    fn new(blocks: &[BodyBlock], successors: &HashMap<String, Vec<String>>) -> Self {
        let names = blocks.iter().map(|block| block.name.clone()).collect();
        let index = blocks
            .iter()
            .enumerate()
            .map(|(idx, block)| (block.name.clone(), idx))
            .collect::<HashMap<_, _>>();
        let successors = blocks
            .iter()
            .map(|block| {
                successors
                    .get(&block.name)
                    .into_iter()
                    .flatten()
                    .filter_map(|successor| index.get(successor).copied())
                    .collect()
            })
            .collect();
        let unreachable = blocks
            .iter()
            .map(|block| {
                block.typed.as_ref().is_some_and(|carrier| {
                    matches!(
                        carrier.terminator,
                        crate::native::tir::TirTerminator::Unreachable
                    )
                })
            })
            .collect();
        Self {
            index,
            names,
            successors,
            unreachable,
        }
    }

    fn reachable(&self, start: &str, excluded: Option<&str>) -> Vec<bool> {
        let mut seen = vec![false; self.successors.len()];
        let Some(&start) = self.index.get(start) else {
            return seen;
        };
        let excluded = excluded.and_then(|name| self.index.get(name).copied());
        let mut pending = vec![start];
        while let Some(node) = pending.pop() {
            if Some(node) == excluded || seen[node] {
                continue;
            }
            seen[node] = true;
            pending.extend(self.successors[node].iter().copied());
        }
        seen
    }

    fn contains(&self, reachable: &[bool], name: &str) -> bool {
        self.index
            .get(name)
            .is_some_and(|index| reachable.get(*index) == Some(&true))
    }

    fn all_paths_reach(&self, start: &str, target: &str, allow_unreachable: bool) -> bool {
        let (Some(&start), Some(&target)) = (self.index.get(start), self.index.get(target)) else {
            return false;
        };
        fn walk(
            graph: &IndexedReachability,
            node: usize,
            target: usize,
            allow_unreachable: bool,
            visiting: &mut [bool],
            memo: &mut [Option<bool>],
        ) -> bool {
            if node == target || (allow_unreachable && graph.unreachable[node]) {
                return true;
            }
            if let Some(cached) = memo[node] {
                return cached;
            }
            if visiting[node] {
                return true;
            }
            visiting[node] = true;
            let ok = !graph.successors[node].is_empty()
                && graph.successors[node].iter().copied().all(|successor| {
                    walk(graph, successor, target, allow_unreachable, visiting, memo)
                });
            visiting[node] = false;
            memo[node] = Some(ok);
            ok
        }
        walk(
            self,
            start,
            target,
            allow_unreachable,
            &mut vec![false; self.successors.len()],
            &mut vec![None; self.successors.len()],
        )
    }
}

pub(in crate::native) fn infer_branch_merges(
    blocks: &[BodyBlock],
) -> HashMap<(String, String), String> {
    let by_name: HashMap<String, &BodyBlock> = blocks.iter().map(|b| (b.name.clone(), b)).collect();
    let successors: HashMap<String, Vec<String>> = blocks
        .iter()
        .map(|b| (b.name.clone(), block_successors(b)))
        .collect();
    let reachability = IndexedReachability::new(blocks, &successors);
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
            // Keep reachability scratch bounded to this header. Caching the transitive closure for
            // every arm is O(V²) memory and made generated thousand-block functions exceed the
            // translation budget; two graph walks perform the same queries in O(V + E) live space.
            let true_reachable = reachability.reachable(&true_label, None);
            let false_reachable = reachability.reachable(&false_label, None);
            let true_reaches_false = reachability.contains(&true_reachable, &false_label);
            let false_reaches_true = reachability.contains(&false_reachable, &true_label);
            let is_loop_continue =
                is_loop_continue_branch(&block.name, &true_label, &false_label, &loop_merges);
            let merge = match (true_exit, false_exit) {
                (Some(a), Some(b)) if a == b => Some(a),
                (Some(a), _) if a == false_label => Some(a),
                (_, Some(b)) if b == true_label => Some(b),
                (Some(a), _)
                    if false_merge.as_ref() == Some(&a)
                        && reachability.all_paths_reach(&false_label, &a, false) =>
                {
                    Some(a)
                }
                (_, Some(b))
                    if true_merge.as_ref() == Some(&b)
                        && reachability.all_paths_reach(&true_label, &b, false) =>
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
                        reachability.all_paths_reach(&true_label, &candidate, false)
                            && reachability.all_paths_reach(&false_label, &candidate, false)
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
                        reachability.all_paths_reach(&true_label, &candidate, false)
                            && reachability.all_paths_reach(&false_label, &candidate, false)
                    }) =>
                {
                    conditional_merge_through_shared_body(&false_label, &true_label, &by_name)
                }
                _ if true_merge.as_ref().zip(false_merge.as_ref()).is_some_and(
                    |(true_merge, false_merge)| {
                        true_merge == false_merge
                            && reachability.all_paths_reach(&true_label, true_merge, false)
                            && reachability.all_paths_reach(&false_label, true_merge, false)
                    },
                ) =>
                {
                    true_merge.clone()
                }
                _ if false_reaches_true
                    && !true_reaches_false
                    && !is_loop_continue
                    && reachability.all_paths_reach(&false_label, &true_label, false) =>
                {
                    Some(true_label.clone())
                }
                _ if true_reaches_false
                    && !false_reaches_true
                    && !is_loop_continue
                    && reachability.all_paths_reach(&true_label, &false_label, false) =>
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
                    &reachability,
                    &true_reachable,
                ),
                _ if block_is_unreachable(&true_label, &by_name) => reachable_merge_after_header(
                    &block.name,
                    &false_label,
                    blocks,
                    &reachability,
                    &false_reachable,
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
                    &reachability,
                    &true_reachable,
                    &false_reachable,
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
                        &reachability,
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
                        &reachability,
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
    reachability: &IndexedReachability,
    true_reachable: &[bool],
    false_reachable: &[bool],
) -> Option<String> {
    let start = blocks
        .iter()
        .position(|block| block.name == header)
        .map(|idx| idx + 1)?;
    // Exclude a back-edge through the header from the candidate proof, matching the old bounded
    // traversal. If either arm can revisit the header, derive the exact no-revisit set locally.
    let true_without_header;
    let false_without_header;
    let true_reachable = if reachability.contains(true_reachable, header) {
        true_without_header = reachability.reachable(true_label, Some(header));
        &true_without_header
    } else {
        true_reachable
    };
    let false_reachable = if reachability.contains(false_reachable, header) {
        false_without_header = reachability.reachable(false_label, Some(header));
        &false_without_header
    } else {
        false_reachable
    };
    blocks
        .iter()
        .skip(start)
        .map(|block| &block.name)
        .find(|candidate| {
            reachability.contains(true_reachable, candidate)
                && reachability.contains(false_reachable, candidate)
                && reachability.all_paths_reach(true_label, candidate, true)
                && reachability.all_paths_reach(false_label, candidate, true)
        })
        .cloned()
}

fn reachable_merge_after_header(
    header: &str,
    label: &str,
    blocks: &[BodyBlock],
    reachability: &IndexedReachability,
    reachable: &[bool],
) -> Option<String> {
    let start = blocks
        .iter()
        .position(|block| block.name == header)
        .map(|idx| idx + 1)?;
    blocks
        .iter()
        .skip(start)
        .map(|block| &block.name)
        .find(|candidate| {
            reachability.contains(reachable, candidate)
                && reachability.all_paths_reach(label, candidate, true)
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
    reachability: &IndexedReachability,
    by_name: &HashMap<String, &BodyBlock>,
) -> bool {
    let reachable = reachability.reachable(header, None);
    [true_label, false_label]
        .into_iter()
        .filter(|target| **target != *merge)
        .any(|body| {
            let has_external_pred = successors.iter().any(|(pred, targets)| {
                pred != header
                    && pred != true_label
                    && pred != false_label
                    && !reachability.contains(&reachable, pred)
                    && targets.iter().any(|target| target == body)
            });
            has_external_pred
                && !external_entry_is_repairable_selection_target(
                    body,
                    header,
                    merge,
                    &reachable,
                    reachability,
                    successors,
                    by_name,
                )
        })
}

fn branch_merge_path_has_unrepairable_external_entry(
    header: &str,
    true_label: &str,
    false_label: &str,
    merge: &str,
    successors: &HashMap<String, Vec<String>>,
    reachability: &IndexedReachability,
    by_name: &HashMap<String, &BodyBlock>,
) -> bool {
    let header_reachable = reachability.reachable(header, None);
    let true_to_merge = reachability.reachable(true_label, Some(merge));
    let false_to_merge = reachability.reachable(false_label, Some(merge));

    reachability.names.iter().enumerate().any(|(index, node)| {
        if node == header
            || node == merge
            || (!true_to_merge[index] && !false_to_merge[index])
            || !reachability.contains(&reachability.reachable(node, None), merge)
        {
            return false;
        }
        let has_external_pred = successors.iter().any(|(pred, targets)| {
            !reachability.contains(&header_reachable, pred)
                && targets.iter().any(|target| target == node)
        });
        has_external_pred
            && !external_entry_is_repairable_selection_target(
                node,
                header,
                merge,
                &header_reachable,
                reachability,
                successors,
                by_name,
            )
    })
}

fn external_entry_is_repairable_selection_target(
    node: &str,
    header: &str,
    merge: &str,
    header_reachable: &[bool],
    reachability: &IndexedReachability,
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
        .chain(
            reachability
                .names
                .iter()
                .filter(|name| reachability.contains(header_reachable, name))
                .map(String::as_str),
        )
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

/// Infer only switches whose live targets are the merge itself or branch directly to one common
/// merge. This linear-time subset is used when a rejected function is too large for the complete
/// structured planner and the transitive-closure switch heuristic. It never guesses through an
/// intermediate region: every accepted arm provides a local edge proof.
pub(in crate::native) fn infer_direct_switch_merges(
    blocks: &[BodyBlock],
) -> HashMap<String, String> {
    let by_name = blocks
        .iter()
        .map(|block| (block.name.clone(), block))
        .collect::<HashMap<_, _>>();
    let order = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut merges = HashMap::new();
    for block in blocks {
        let Some(targets) = switch_targets(block) else {
            continue;
        };
        let live_targets = targets
            .into_iter()
            .filter(|target| !block_is_unreachable(target, &by_name))
            .collect::<Vec<_>>();
        let Some(first) = live_targets.first() else {
            continue;
        };
        let mut candidates = Vec::with_capacity(2);
        if let Some(target) = unconditional_branch_target(by_name.get(first).copied()) {
            candidates.push(target);
        }
        candidates.push(first.clone());
        candidates.dedup();
        let Some(header_order) = order.get(block.name.as_str()) else {
            continue;
        };
        let merge = candidates.into_iter().find(|candidate| {
            order
                .get(candidate.as_str())
                .is_some_and(|candidate_order| candidate_order > header_order)
                && live_targets.iter().all(|target| {
                    target == candidate
                        || unconditional_branch_target(by_name.get(target).copied()).as_ref()
                            == Some(candidate)
                })
        });
        if let Some(merge) = merge {
            merges.insert(block.name.clone(), merge);
        }
    }
    merges
}

/// Linear-time direct-reconvergence subset for conditional branches in oversized rejected CFGs.
/// A merge is accepted only when each arm is the merge or branches to it immediately.
pub(in crate::native) fn infer_direct_branch_merges(
    blocks: &[BodyBlock],
) -> HashMap<(String, String), String> {
    let by_name = blocks
        .iter()
        .map(|block| (block.name.clone(), block))
        .collect::<HashMap<_, _>>();
    let order = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut merges = HashMap::new();
    for block in blocks {
        let Some((on_true, on_false)) = conditional_branch_targets(block) else {
            continue;
        };
        let Some(header_order) = order.get(block.name.as_str()) else {
            continue;
        };
        let mut candidates = Vec::with_capacity(4);
        for arm in [&on_true, &on_false] {
            if let Some(target) = unconditional_branch_target(by_name.get(arm).copied()) {
                candidates.push(target);
            }
            candidates.push(arm.clone());
        }
        candidates.dedup();
        let merge = candidates.into_iter().find(|candidate| {
            order
                .get(candidate.as_str())
                .is_some_and(|candidate_order| candidate_order > header_order)
                && [&on_true, &on_false].into_iter().all(|arm| {
                    block_is_unreachable(arm, &by_name)
                        || arm == candidate
                        || unconditional_branch_target(by_name.get(arm).copied()).as_ref()
                            == Some(candidate)
                })
        });
        if let Some(merge) = merge {
            merges.insert((on_true, on_false), merge);
        }
    }
    merges
}

/// Infer conditional merges for an oversized rejected CFG with bounded graph storage. Direct local
/// reconvergence handles terminal-arm shapes that ordinary post-dominance cannot represent; the CHK
/// immediate-post-dominator tree then covers larger diamonds without building one reachability set
/// per branch. Loop headers are excluded because their conditional owns `OpLoopMerge`, not
/// `OpSelectionMerge`.
pub(in crate::native) fn infer_bounded_branch_merges_by_header(
    blocks: &[BodyBlock],
) -> HashMap<String, String> {
    let direct = infer_direct_branch_merges(blocks);
    let Some(cfg) = Cfg::from_blocks(blocks) else {
        return HashMap::new();
    };
    let dominators = cfg.dominators();
    let post_dominators = post_idom(blocks);
    let order = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut merges = HashMap::new();

    for block in blocks {
        let Some((on_true, on_false)) = conditional_branch_targets(block) else {
            continue;
        };
        if let Some(merge) = direct.get(&(on_true, on_false)) {
            merges.insert(block.name.clone(), merge.clone());
            continue;
        }
        let is_loop_header = cfg
            .predecessors
            .get(&block.name)
            .into_iter()
            .flatten()
            .any(|predecessor| dominators.dominates(&block.name, predecessor));
        if is_loop_header {
            continue;
        }
        let Some(merge) = post_dominators.get(&block.name) else {
            continue;
        };
        let ordered_after_header = order
            .get(block.name.as_str())
            .zip(order.get(merge.as_str()))
            .is_some_and(|(header, merge)| merge > header);
        if ordered_after_header {
            merges.insert(block.name.clone(), merge.clone());
        }
    }
    merges
}

/// Funnel the common two-way dispatch reached from both arms of an outer branch through one predicate
/// phi. The source shape
/// `H -> {A,B}; A -> {T,F}; B -> {T,F}` is irreducible as nested SPIR-V selections because `T` and
/// `F` are entered from sibling constructs. Rewriting it to
/// `H -> {A,B}; A/B -> J; J -> {T,F}` preserves the chosen predicate and gives both selections one
/// entry. Destination phis are funneled through matching value phis in `J`.
pub(in crate::native) fn funnel_shared_branch_dispatches(blocks: &[BodyBlock]) -> Vec<BodyBlock> {
    let mut current = blocks.to_vec();
    let mut counter = 0usize;
    // Each successful round turns both candidate arm terminators into unconditional branches, so
    // that source header can never match again. One round per original block is therefore a hard
    // convergence bound rather than an arbitrary workload-sized cap.
    for _ in 0..blocks.len() {
        let Some(next) = funnel_one_shared_branch_dispatch(&current, &mut counter) else {
            break;
        };
        current = next;
    }
    current
}

fn funnel_one_shared_branch_dispatch(
    blocks: &[BodyBlock],
    counter: &mut usize,
) -> Option<Vec<BodyBlock>> {
    let cfg = Cfg::from_blocks(blocks)?;
    let by_name = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let names = by_name.keys().copied().collect::<HashSet<_>>();

    'header: for header in blocks {
        let Some((left, right)) = conditional_branch_targets(header) else {
            continue;
        };
        if left == right {
            continue;
        }
        let (Some(&left_index), Some(&right_index)) =
            (by_name.get(left.as_str()), by_name.get(right.as_str()))
        else {
            continue;
        };
        let private_arm = |arm: &str| {
            cfg.predecessors
                .get(arm)
                .is_some_and(|preds| preds.len() == 1 && preds[0] == header.name)
        };
        if !private_arm(&left) || !private_arm(&right) {
            continue;
        }
        let Some(left_typed) = blocks[left_index].typed.as_deref() else {
            continue;
        };
        let Some(right_typed) = blocks[right_index].typed.as_deref() else {
            continue;
        };
        let crate::native::tir::TirTerminator::BrCond {
            cond: left_cond,
            t: left_true,
            f: left_false,
        } = &left_typed.terminator
        else {
            continue;
        };
        let crate::native::tir::TirTerminator::BrCond {
            cond: right_cond,
            t: right_true,
            f: right_false,
        } = &right_typed.terminator
        else {
            continue;
        };
        if left_true != right_true
            || left_false != right_false
            || left_true == left_false
            || !names.contains(left_true.as_str())
            || !names.contains(left_false.as_str())
        {
            continue;
        }
        let insertion = left_index.max(right_index) + 1;
        if [left_true, left_false].into_iter().any(|target| {
            by_name
                .get(target.as_str())
                .is_none_or(|target_index| *target_index < insertion)
        }) {
            continue;
        }

        let mut join_name = format!("%metal2vulkan.branch_funnel.{}", *counter);
        while names.contains(join_name.as_str()) {
            *counter += 1;
            join_name = format!("%metal2vulkan.branch_funnel.{}", *counter);
        }
        *counter += 1;
        let predicate = format!("{join_name}.predicate");
        let join_lines = vec![
            format!("{predicate} = phi i1 [ {left_cond}, {left} ], [ {right_cond}, {right} ]"),
            format!("br i1 {predicate}, label {left_true}, label {left_false}"),
        ];
        let Some(join_typed) =
            crate::native::tir::lower_block_carrier(&join_name, &join_lines, &HashMap::new())
        else {
            continue;
        };

        // Every destination phi must carry both soon-to-be-funnelled predecessors. The late CFG
        // repair could diagnose a malformed source, but this semantic transform declines it instead.
        let mut target_rewrites = Vec::new();
        let mut join_typed = join_typed;
        let mut join_phi_index = 0usize;
        for target in [left_true, left_false] {
            let target_index = *by_name.get(target.as_str())?;
            let Some(target_typed) = blocks[target_index].typed.as_deref() else {
                continue 'header;
            };
            let mut rewrites = Vec::new();
            for inst in &target_typed.insts {
                let Some((ty, incoming)) = &inst.phi_incoming else {
                    continue;
                };
                let from_left = incoming
                    .iter()
                    .filter(|(_, pred)| pred == &left)
                    .collect::<Vec<_>>();
                let from_right = incoming
                    .iter()
                    .filter(|(_, pred)| pred == &right)
                    .collect::<Vec<_>>();
                if from_left.is_empty() && from_right.is_empty() {
                    continue;
                }
                if from_left.len() != 1 || from_right.len() != 1 {
                    continue 'header;
                }
                let result = inst.result.as_ref()?;
                let phi_name = format!("{join_name}.phi.{join_phi_index}");
                join_phi_index += 1;
                join_typed.push_value_phi(
                    &phi_name,
                    ty,
                    &[
                        (from_left[0].0.clone(), left.clone()),
                        (from_right[0].0.clone(), right.clone()),
                    ],
                );
                let mut replacement = Vec::with_capacity(incoming.len() - 1);
                let mut inserted = false;
                for (value, pred) in incoming {
                    if pred == &left || pred == &right {
                        if !inserted {
                            replacement.push((LlValue::Local(phi_name.clone()), join_name.clone()));
                            inserted = true;
                        }
                    } else {
                        replacement.push((value.clone(), pred.clone()));
                    }
                }
                rewrites.push((result.clone(), replacement));
            }
            target_rewrites.push((target_index, rewrites));
        }

        let mut out = blocks.to_vec();
        for arm_index in [left_index, right_index] {
            let typed = std::sync::Arc::make_mut(out[arm_index].typed.as_mut()?);
            typed.set_unconditional_branch(&join_name);
        }
        for (target_index, rewrites) in target_rewrites {
            let typed = std::sync::Arc::make_mut(out[target_index].typed.as_mut()?);
            for (result, incoming) in rewrites {
                typed.set_phi_incomings(&result, &incoming);
            }
        }
        out.insert(
            insertion,
            BodyBlock {
                name: join_name,
                role: BlockRole::Normal,
                typed: Some(join_typed.into()),
            },
        );
        if crate::env_vars::retry_debug() {
            eprintln!(
                "[retry-debug] funnel shared dispatch: header={} arms=({}, {}) targets=({}, {})",
                header.name, left, right, left_true, left_false
            );
        }
        return Some(out);
    }
    None
}

/// Refunnel a deep short-circuit arm shared with its enclosing selection. For header `H`, immediate
/// arm `S`, and post-dominator `M`, every entry into `S` from the construct and every construct path
/// that bypasses `S` into `M` is redirected through `J`; `J` phis the route bit and branches to
/// `{S,M}`. This removes the cross-arm entry without cloning the (potentially very large) `S` region.
pub(in crate::native) fn refunnel_one_deep_shared_arm(
    blocks: &[BodyBlock],
    counter: &mut usize,
) -> Option<Vec<BodyBlock>> {
    let cfg = Cfg::from_blocks(blocks)?;
    let dominators = cfg.dominators();
    let post_dominators = post_idom(blocks);
    let by_name = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let names = by_name.keys().copied().collect::<HashSet<_>>();

    'header: for header in blocks {
        let Some((on_true, on_false)) = conditional_branch_targets(header) else {
            continue;
        };
        let Some(merge) = post_dominators.get(&header.name) else {
            continue;
        };
        if cfg
            .predecessors
            .get(&header.name)
            .into_iter()
            .flatten()
            .any(|pred| dominators.dominates(&header.name, pred))
        {
            continue;
        }
        for shared in [&on_true, &on_false] {
            if shared == merge {
                continue;
            }
            let candidate_entries = cfg
                .predecessors
                .get(shared)
                .into_iter()
                .flatten()
                .filter(|pred| dominators.dominates(&header.name, pred))
                .cloned()
                .collect::<Vec<_>>();
            if candidate_entries.len() < 2
                || !candidate_entries.iter().any(|pred| pred == &header.name)
            {
                continue;
            }
            let shared_reachable = cfg.reachable_from(shared);
            let shared_entries = candidate_entries
                .into_iter()
                .filter(|pred| !shared_reachable.contains(pred.as_str()))
                .collect::<Vec<_>>();
            if shared_entries.len() < 2 || !shared_entries.iter().any(|pred| pred == &header.name) {
                continue;
            }
            let bypass_entries = cfg
                .predecessors
                .get(merge)
                .into_iter()
                .flatten()
                .filter(|pred| {
                    dominators.dominates(&header.name, pred)
                        && !shared_reachable.contains(pred.as_str())
                })
                .cloned()
                .collect::<Vec<_>>();
            if bypass_entries.is_empty() {
                continue;
            }
            let mut all_entries = shared_entries
                .iter()
                .map(|pred| (pred.clone(), true))
                .chain(bypass_entries.iter().map(|pred| (pred.clone(), false)))
                .collect::<Vec<_>>();
            all_entries.sort_by_key(|(pred, _)| by_name.get(pred.as_str()).copied());
            all_entries.dedup_by(|left, right| left.0 == right.0);

            let mut join_name = format!("%metal2vulkan.branch_refunnel.{}", *counter);
            while names.contains(join_name.as_str()) {
                *counter += 1;
                join_name = format!("%metal2vulkan.branch_refunnel.{}", *counter);
            }
            *counter += 1;
            let route = format!("{join_name}.route");
            let incoming_text = all_entries
                .iter()
                .map(|(pred, takes_shared)| format!("[ {takes_shared}, {pred} ]"))
                .collect::<Vec<_>>()
                .join(", ");
            let join_lines = vec![
                format!("{route} = phi i1 {incoming_text}"),
                format!("br i1 {route}, label {shared}, label {merge}"),
            ];
            let Some(mut join_typed) =
                crate::native::tir::lower_block_carrier(&join_name, &join_lines, &HashMap::new())
            else {
                continue;
            };

            let mut target_rewrites = Vec::new();
            let mut join_phi_index = 0usize;
            for (target, entries) in [
                (shared.as_str(), shared_entries.as_slice()),
                (merge.as_str(), bypass_entries.as_slice()),
            ] {
                let target_index = *by_name.get(target)?;
                let Some(target_typed) = blocks[target_index].typed.as_deref() else {
                    continue 'header;
                };
                let entry_set = entries.iter().map(String::as_str).collect::<HashSet<_>>();
                let mut rewrites = Vec::new();
                for inst in &target_typed.insts {
                    let Some((ty, incoming)) = &inst.phi_incoming else {
                        continue;
                    };
                    let selected = incoming
                        .iter()
                        .filter(|(_, pred)| entry_set.contains(pred.as_str()))
                        .collect::<Vec<_>>();
                    if selected.is_empty() {
                        continue;
                    }
                    if selected.len() != entry_set.len() {
                        continue 'header;
                    }
                    let result = inst.result.as_ref()?;
                    let phi_name = format!("{join_name}.phi.{join_phi_index}");
                    join_phi_index += 1;
                    let join_incoming = selected
                        .iter()
                        .map(|(value, pred)| ((*value).clone(), pred.clone()))
                        .collect::<Vec<_>>();
                    join_typed.push_value_phi(&phi_name, ty, &join_incoming);
                    let mut replacement = Vec::with_capacity(incoming.len() + 1 - selected.len());
                    let mut inserted = false;
                    for (value, pred) in incoming {
                        if entry_set.contains(pred.as_str()) {
                            if !inserted {
                                replacement
                                    .push((LlValue::Local(phi_name.clone()), join_name.clone()));
                                inserted = true;
                            }
                        } else {
                            replacement.push((value.clone(), pred.clone()));
                        }
                    }
                    rewrites.push((result.clone(), replacement));
                }
                target_rewrites.push((target_index, rewrites));
            }

            let mut out = blocks.to_vec();
            for (pred, takes_shared) in &all_entries {
                let target = if *takes_shared { shared } else { merge };
                let pred_index = *by_name.get(pred.as_str())?;
                let typed = std::sync::Arc::make_mut(out[pred_index].typed.as_mut()?);
                typed.redirect_successor(target, &join_name);
            }
            for (target_index, rewrites) in target_rewrites {
                let typed = std::sync::Arc::make_mut(out[target_index].typed.as_mut()?);
                for (result, incoming) in rewrites {
                    typed.set_phi_incomings(&result, &incoming);
                }
            }
            let insertion = *by_name.get(merge.as_str())?;
            out.insert(
                insertion,
                BodyBlock {
                    name: join_name,
                    role: BlockRole::Normal,
                    typed: Some(join_typed.into()),
                },
            );
            return Some(out);
        }
    }
    None
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
            let typed = block
                .typed
                .as_ref()
                .and_then(|p| {
                    crate::native::tir::lower_block_carrier_with_prefix(
                        &block.name,
                        p,
                        std::slice::from_ref(&terminator),
                    )
                })
                .map(Into::into);
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
            }
            .map(Into::into);
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
                )
                .map(Into::into);
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
        if let Some(t) = block.typed_mut() {
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
        if let Some(t) = blocks[pred_idx].typed_mut() {
            t.redirect_successor(merge, &synthetic);
        }
        insert_after.insert(pred, synthetic);
    }
    // Extend the intermediate carrier's phis from the typed additions (merge-carrier-sourced values).
    // Every `typed_additions` key names an intermediate phi (checked against `intermediate_phi_results`
    // above), so each `append_phi_incoming` targets an existing carrier phi.
    if let Some(t) = blocks[intermediate_idx].typed_mut() {
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
    if let Some(t) = blocks[merge_idx].typed_mut() {
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
                crate::native::tir::lower_block_carrier(&synthetic, &lines, &HashMap::new())
                    .map(Into::into);
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
            byval_param_pointees: vec![None; 3],
            blocks: Vec::new(),
        };

        assert_eq!(implicit_entry_block_name(&function), "%7");

        let function = LlFunction {
            name: "main".to_string(),
            ret: LlType::Void,
            params: vec![("%named".to_string(), LlType::Int(32))],
            byval_param_pointees: vec![None],
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

    #[test]
    fn direct_switch_merge_accepts_only_local_common_reconvergence() {
        let blocks = vec![
            block(
                "%switch",
                "switch i32 %tag, label %dead [ i32 0, label %a i32 1, label %b ]",
            ),
            block("%a", "br label %merge"),
            block("%b", "br label %merge"),
            block("%dead", "unreachable"),
            block("%merge", "ret void"),
        ];
        assert_eq!(
            infer_direct_switch_merges(&blocks).get("%switch"),
            Some(&"%merge".to_string())
        );

        let mut divergent = blocks;
        divergent[2] = block("%b", "br label %other");
        divergent.push(block("%other", "ret void"));
        assert!(infer_direct_switch_merges(&divergent).is_empty());
    }

    #[test]
    fn direct_branch_merge_accepts_only_local_common_reconvergence() {
        let blocks = vec![
            block("%if", "br i1 %cond, label %a, label %b"),
            block("%a", "br label %merge"),
            block("%b", "br label %merge"),
            block("%merge", "ret void"),
        ];
        assert_eq!(
            infer_direct_branch_merges(&blocks).get(&("%a".to_string(), "%b".to_string())),
            Some(&"%merge".to_string())
        );

        let mut divergent = blocks;
        divergent[2] = block("%b", "br label %other");
        divergent.push(block("%other", "ret void"));
        assert!(infer_direct_branch_merges(&divergent).is_empty());

        let terminal = vec![
            block("%if", "br i1 %cond, label %live, label %dead"),
            block("%live", "br label %merge"),
            block("%dead", "unreachable"),
            block("%merge", "ret void"),
        ];
        assert_eq!(
            infer_direct_branch_merges(&terminal).get(&("%live".to_string(), "%dead".to_string())),
            Some(&"%merge".to_string())
        );
    }

    #[test]
    fn bounded_branch_merge_uses_post_dominance_but_excludes_loop_headers() {
        let blocks = vec![
            block("%if", "br i1 %cond, label %a, label %b"),
            block("%a", "br label %a_tail"),
            block("%a_tail", "br label %merge"),
            block("%b", "br label %b_tail"),
            block("%b_tail", "br label %merge"),
            block("%merge", "ret void"),
        ];
        assert_eq!(
            infer_bounded_branch_merges_by_header(&blocks).get("%if"),
            Some(&"%merge".to_string())
        );

        let loop_blocks = vec![
            block("%loop", "br i1 %cond, label %body, label %exit"),
            block("%body", "br label %loop"),
            block("%exit", "ret void"),
        ];
        assert!(infer_bounded_branch_merges_by_header(&loop_blocks).is_empty());
    }

    #[test]
    fn shared_branch_dispatch_is_funnelled_with_destination_phis() {
        let blocks = vec![
            block("%header", "br i1 %outer, label %left, label %right"),
            block("%left", "br i1 %left_cond, label %taken, label %done"),
            block("%right", "br i1 %right_cond, label %taken, label %done"),
            synthetic_block(
                "%taken".to_string(),
                vec![
                    "%value = phi i32 [ 1, %left ], [ 2, %right ]".to_string(),
                    "ret void".to_string(),
                ],
                BlockRole::Normal,
            ),
            block("%done", "ret void"),
        ];
        let out = funnel_shared_branch_dispatches(&blocks);
        let join = out
            .iter()
            .find(|block| block.name.starts_with("%metal2vulkan.branch_funnel."))
            .expect("shared dispatch receives a funnel");
        assert_eq!(block_successors(&out[1]), vec![join.name.clone()]);
        assert_eq!(block_successors(&out[2]), vec![join.name.clone()]);
        assert_eq!(
            conditional_branch_targets(join),
            Some(("%taken".to_string(), "%done".to_string()))
        );

        let taken = out.iter().find(|block| block.name == "%taken").unwrap();
        let incoming = &taken.typed.as_ref().unwrap().insts[0]
            .phi_incoming
            .as_ref()
            .unwrap()
            .1;
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].1, join.name);
        assert_eq!(
            join.typed
                .as_ref()
                .unwrap()
                .insts
                .iter()
                .filter(|inst| inst.opcode == "phi")
                .count(),
            2
        );
    }

    #[test]
    fn deep_shared_arm_is_refunnelled_without_cloning_its_region() {
        let blocks = vec![
            block("%header", "br i1 %outer, label %shared, label %inner"),
            block("%inner", "br label %dispatch"),
            block(
                "%dispatch",
                "br i1 %nested, label %via_shared, label %bypass",
            ),
            block("%via_shared", "br label %shared"),
            block("%bypass", "br label %merge"),
            block("%shared", "br label %merge"),
            block("%merge", "ret void"),
        ];
        let out = refunnel_one_deep_shared_arm(&blocks, &mut 0).unwrap();
        let join = out
            .iter()
            .find(|block| block.name.starts_with("%metal2vulkan.branch_refunnel."))
            .expect("deep shared arm receives a refunnel");
        assert_eq!(
            conditional_branch_targets(join),
            Some(("%shared".to_string(), "%merge".to_string()))
        );
        for predecessor in ["%header", "%via_shared", "%bypass"] {
            let block = out.iter().find(|block| block.name == predecessor).unwrap();
            assert!(block_successors(block).contains(&join.name));
        }
        assert_eq!(out.len(), blocks.len() + 1);
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
