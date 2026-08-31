//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

#[cfg(test)]
#[derive(Clone)]
struct ContinuePhiSplit {
    inside: Vec<(crate::native::ir::LlValue, String)>,
    outside: Vec<(crate::native::ir::LlValue, String)>,
}

/// Keep every declared loop continue single-entry in the finalized typed CFG.
///
/// A pass-through continue `C -> H` can retain predecessors from outside `H` when heuristic merge
/// inference or an earlier ownership transform selects `C` as the back-edge carrier. Redirect only
/// those non-`H`-dominated predecessors to `H`, moving their exact continue-phi values onto matching
/// header-phi edges. The transaction declines unless every header phi proves a complete mapping.
#[cfg(test)]
pub(in crate::native) fn normalize_loop_continue_external_predecessors(
    blocks: &mut Vec<BodyBlock>,
    loop_merges: &HashMap<String, LoopMergeInfo>,
) -> bool {
    let existing_order_violations = dominator_serialization_violations(blocks);
    let mut changed = false;
    while let Some((header, continue_target, outside_preds, continue_updates, header_updates)) =
        next_loop_continue_external_predecessor_plan(blocks, loop_merges)
    {
        let Some(candidate) = apply_loop_continue_external_predecessor_plan(
            blocks,
            &header,
            &continue_target,
            &outside_preds,
            &continue_updates,
            &header_updates,
        ) else {
            break;
        };
        *blocks = candidate;
        changed = true;
    }

    if changed {
        restore_new_dominator_serialization(blocks, &existing_order_violations);
    }
    changed
}

/// Give a conditional a legal selection boundary when its effective merge is a loop continue.
///
/// The finalized loop and branch maps are the ownership contract the emitter consumes. Normalize
/// their three exact collision shapes before label allocation: an enclosing selection moves to the
/// loop's outside boundary, an in-loop selection that reconverges at the continue gets a phi-aware
/// private pass-through, and a direct `{continue, loop-merge}` branch gets a private break edge.
pub(in crate::native) fn normalize_continue_selection_merge_targets(
    blocks: &mut Vec<BodyBlock>,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    branch_merges_by_header: &mut HashMap<String, String>,
) -> bool {
    let mut counter = next_selection_merge_suffix(blocks);
    let mut changed = false;
    while let Some(cfg) = super::super::graph::Cfg::from_blocks(blocks) {
        let dominators = cfg.dominators();
        let mut applied = false;
        let loop_headers = blocks
            .iter()
            .filter(|block| loop_merges.contains_key(&block.name))
            .map(|block| block.name.clone())
            .collect::<Vec<_>>();
        for loop_header in loop_headers {
            let info = &loop_merges[&loop_header];
            let selection_names = blocks
                .iter()
                .map(|block| block.name.clone())
                .collect::<Vec<_>>();
            for selection_name in selection_names {
                if loop_merges.contains_key(&selection_name)
                    || branch_merges_by_header.get(&selection_name) != Some(&info.continue_target)
                {
                    continue;
                }
                let Some((true_target, false_target)) = blocks
                    .iter()
                    .find(|block| block.name == selection_name)
                    .and_then(conditional_branch_targets)
                else {
                    continue;
                };

                // An enclosing selection cannot close inside the loop's continue construct.
                if dominators.dominates(&selection_name, &loop_header) {
                    let outside_arm = [true_target.as_str(), false_target.as_str()]
                        .into_iter()
                        .find(|target| !dominators.dominates(&loop_header, target));
                    let boundary = outside_arm
                        .filter(|outside| {
                            all_typed_paths_reach_without(
                                blocks,
                                &info.merge,
                                outside,
                                &[&loop_header],
                            )
                        })
                        .unwrap_or(&info.merge)
                        .to_string();
                    if boundary == info.continue_target {
                        continue;
                    }
                    branch_merges_by_header.insert(selection_name, boundary);
                    applied = true;
                    break;
                }

                let direct_break_continue = (true_target == info.continue_target
                    && false_target == info.merge)
                    || (false_target == info.continue_target && true_target == info.merge);
                if direct_break_continue {
                    let Some((candidate, private)) = split_direct_continue_selection_merge(
                        blocks,
                        &selection_name,
                        &info.merge,
                        &mut counter,
                    ) else {
                        continue;
                    };
                    *blocks = candidate;
                    branch_merges_by_header.insert(selection_name, private);
                    applied = true;
                    break;
                }

                if !all_typed_paths_reach_without(
                    blocks,
                    &true_target,
                    &info.continue_target,
                    &[&info.merge],
                ) || !all_typed_paths_reach_without(
                    blocks,
                    &false_target,
                    &info.continue_target,
                    &[&info.merge],
                ) {
                    continue;
                }
                let construct =
                    typed_reachable_before(blocks, &selection_name, &info.continue_target);
                let predecessors = blocks
                    .iter()
                    .filter(|block| {
                        construct.contains(&block.name)
                            && block_successors(block)
                                .iter()
                                .any(|target| target == &info.continue_target)
                    })
                    .map(|block| block.name.clone())
                    .collect::<Vec<_>>();
                if predecessors.is_empty() {
                    continue;
                }
                let continue_phis_are_typed = blocks
                    .iter()
                    .find(|block| block.name == info.continue_target)
                    .and_then(|block| block.typed.as_ref())
                    .is_some_and(|typed| {
                        typed
                            .insts
                            .iter()
                            .take_while(|inst| inst.is_phi())
                            .all(|inst| inst.phi_incoming().is_some())
                    });
                if !continue_phis_are_typed {
                    continue;
                }
                let mut candidate = blocks.clone();
                let private = if block_has_phi(&candidate, &info.continue_target) {
                    synth_unique_selection_merge_phi_explicit(
                        &mut candidate,
                        &predecessors,
                        &info.continue_target,
                        &HashSet::new(),
                        &mut counter,
                    )
                } else {
                    synth_unique_selection_merge_no_phi_explicit(
                        &mut candidate,
                        &predecessors,
                        &info.continue_target,
                        &HashSet::new(),
                        &mut counter,
                    )
                };
                let Some(private) = private else {
                    continue;
                };
                *blocks = candidate;
                branch_merges_by_header.insert(selection_name, private);
                applied = true;
                break;
            }
            if applied {
                break;
            }
        }
        if !applied {
            break;
        }
        changed = true;
    }
    changed
}

fn split_direct_continue_selection_merge(
    blocks: &[BodyBlock],
    selection: &str,
    loop_merge: &str,
    counter: &mut usize,
) -> Option<(Vec<BodyBlock>, String)> {
    let mut candidate = blocks.to_vec();
    let selection_idx = candidate.iter().position(|block| block.name == selection)?;
    let merge_idx = candidate
        .iter()
        .position(|block| block.name == loop_merge)?;
    let private = format!("{SPLIT_PREFIX}{SEL_TOKEN}{counter}");
    *counter += 1;
    candidate[selection_idx]
        .typed_mut()?
        .redirect_successor(loop_merge, &private);
    candidate[merge_idx]
        .typed_mut()?
        .rewrite_phi_predecessor(selection, &private);
    candidate.insert(
        merge_idx,
        synthetic_block(
            private.clone(),
            vec![format!("br label {loop_merge}")],
            role_for_name(&private),
        ),
    );
    Some((candidate, private))
}

fn all_typed_paths_reach_without(
    blocks: &[BodyBlock],
    start: &str,
    target: &str,
    forbidden: &[&str],
) -> bool {
    fn walk(
        by_name: &HashMap<&str, &BodyBlock>,
        label: &str,
        target: &str,
        forbidden: &[&str],
        visiting: &mut HashSet<String>,
        memo: &mut HashMap<String, bool>,
    ) -> bool {
        if label == target {
            return true;
        }
        if forbidden.contains(&label) {
            return false;
        }
        if let Some(cached) = memo.get(label) {
            return *cached;
        }
        if !visiting.insert(label.to_string()) {
            return true;
        }
        let result = by_name.get(label).is_some_and(|block| {
            let successors = block_successors(block);
            !successors.is_empty()
                && successors
                    .iter()
                    .all(|successor| walk(by_name, successor, target, forbidden, visiting, memo))
        });
        visiting.remove(label);
        memo.insert(label.to_string(), result);
        result
    }

    let by_name = blocks
        .iter()
        .map(|block| (block.name.as_str(), block))
        .collect::<HashMap<_, _>>();
    walk(
        &by_name,
        start,
        target,
        forbidden,
        &mut HashSet::new(),
        &mut HashMap::new(),
    )
}

fn typed_reachable_before(blocks: &[BodyBlock], start: &str, target: &str) -> HashSet<String> {
    let by_name = blocks
        .iter()
        .map(|block| (block.name.as_str(), block))
        .collect::<HashMap<_, _>>();
    let mut seen = HashSet::new();
    let mut pending = vec![start.to_string()];
    while let Some(label) = pending.pop() {
        if label == target || !seen.insert(label.clone()) {
            continue;
        }
        if let Some(block) = by_name.get(label.as_str()) {
            pending.extend(block_successors(block));
        }
    }
    seen
}

#[cfg(test)]
type ContinueNormalizationPlan = (
    String,
    String,
    HashSet<String>,
    Vec<(String, Vec<(crate::native::ir::LlValue, String)>)>,
    Vec<(String, Vec<(crate::native::ir::LlValue, String)>)>,
);

#[cfg(test)]
fn apply_loop_continue_external_predecessor_plan(
    blocks: &[BodyBlock],
    header: &str,
    continue_target: &str,
    outside_preds: &HashSet<String>,
    continue_updates: &[(String, Vec<(crate::native::ir::LlValue, String)>)],
    header_updates: &[(String, Vec<(crate::native::ir::LlValue, String)>)],
) -> Option<Vec<BodyBlock>> {
    // `BodyBlock` shares its immutable typed carrier through `Arc`; only blocks edited below are
    // copied. Adopt the candidate after every carrier has been proved present, so invariant drift
    // cannot expose half of the edge/phi transaction to subsequent structurization.
    let mut candidate = blocks.to_vec();
    let continue_idx = candidate
        .iter()
        .position(|block| block.name == continue_target)?;
    let header_idx = candidate.iter().position(|block| block.name == header)?;
    if candidate[continue_idx].typed.is_none()
        || candidate[header_idx].typed.is_none()
        || candidate
            .iter()
            .filter(|block| outside_preds.contains(&block.name))
            .any(|block| block.typed.is_none())
    {
        return None;
    }
    let typed = candidate[continue_idx].typed_mut()?;
    for (result, incoming) in continue_updates {
        typed.set_phi_incomings(result, incoming);
    }
    let typed = candidate[header_idx].typed_mut()?;
    for (result, incoming) in header_updates {
        typed.set_phi_incomings(result, incoming);
    }
    for block in &mut candidate {
        if outside_preds.contains(&block.name) {
            block
                .typed_mut()?
                .redirect_successor(continue_target, header);
        }
    }
    Some(candidate)
}

#[cfg(test)]
fn next_loop_continue_external_predecessor_plan(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
) -> Option<ContinueNormalizationPlan> {
    use crate::native::ir::LlValue;

    let cfg = super::super::graph::Cfg::from_blocks(blocks)?;
    let dominators = cfg.dominators();
    for block in blocks {
        let header = &block.name;
        let Some(info) = loop_merges.get(header) else {
            continue;
        };
        if info.continue_target == *header {
            continue;
        }
        let Some(continue_block) = blocks
            .iter()
            .find(|block| block.name == info.continue_target)
        else {
            continue;
        };
        let Some(continue_typed) = continue_block.typed.as_ref() else {
            continue;
        };
        if !matches!(
            &continue_typed.terminator,
            crate::native::tir::TirTerminator::Br(target) if target == header
        ) {
            continue;
        }
        let outside_preds = cfg
            .predecessors
            .get(&info.continue_target)
            .into_iter()
            .flatten()
            .filter(|pred| !dominators.dominates(header, pred))
            .cloned()
            .collect::<HashSet<_>>();
        if outside_preds.is_empty() {
            continue;
        }

        let mut splits = HashMap::<String, ContinuePhiSplit>::new();
        let mut continue_updates = Vec::new();
        let mut admissible = true;
        for inst in continue_typed.insts.iter().take_while(|inst| inst.is_phi()) {
            let (Some(result), Some((_, incoming))) =
                (inst.result.as_ref(), inst.phi_incoming().as_ref())
            else {
                admissible = false;
                break;
            };
            let (outside, inside): (Vec<_>, Vec<_>) = incoming
                .iter()
                .cloned()
                .partition(|(_, pred)| outside_preds.contains(pred));
            if outside.is_empty() {
                continue;
            }
            if inside.is_empty() {
                admissible = false;
                break;
            }
            continue_updates.push((result.clone(), inside.clone()));
            splits.insert(result.clone(), ContinuePhiSplit { inside, outside });
        }
        if !admissible {
            continue;
        }

        let Some(header_typed) = block.typed.as_ref() else {
            continue;
        };
        let mut header_updates = Vec::new();
        for inst in header_typed.insts.iter().take_while(|inst| inst.is_phi()) {
            let (Some(result), Some((_, incoming))) =
                (inst.result.as_ref(), inst.phi_incoming().as_ref())
            else {
                admissible = false;
                break;
            };
            let mut rewritten = Vec::new();
            let mut rewrote_continue = false;
            for (value, pred) in incoming {
                let split = if pred == &info.continue_target {
                    match value {
                        LlValue::Local(name) => splits.get(name),
                        _ => None,
                    }
                } else {
                    None
                };
                let Some(split) = split else {
                    rewritten.push((value.clone(), pred.clone()));
                    continue;
                };
                rewrote_continue = true;
                if !split.inside.is_empty() {
                    rewritten.push((value.clone(), pred.clone()));
                }
                rewritten.extend(split.outside.clone());
            }
            let rewritten_preds = rewritten
                .iter()
                .map(|(_, pred)| pred.as_str())
                .collect::<HashSet<_>>();
            if !rewrote_continue
                || outside_preds
                    .iter()
                    .any(|pred| !rewritten_preds.contains(pred.as_str()))
            {
                admissible = false;
                break;
            }
            header_updates.push((result.clone(), rewritten));
        }
        if admissible {
            return Some((
                header.clone(),
                info.continue_target.clone(),
                outside_preds,
                continue_updates,
                header_updates,
            ));
        }
    }
    None
}

#[cfg(test)]
fn dominator_serialization_violations(blocks: &[BodyBlock]) -> HashSet<(String, String)> {
    let Some(cfg) = super::super::graph::Cfg::from_blocks(blocks) else {
        return HashSet::new();
    };
    let dominators = cfg.dominators();
    let positions = blocks
        .iter()
        .enumerate()
        .map(|(position, block)| (block.name.as_str(), position))
        .collect::<HashMap<_, _>>();
    blocks
        .iter()
        .filter_map(|block| {
            let parent = dominators.idom(&block.name)?;
            (positions[parent] >= positions[block.name.as_str()])
                .then(|| (parent.to_string(), block.name.clone()))
        })
        .collect()
}

#[cfg(test)]
fn restore_new_dominator_serialization(
    blocks: &mut Vec<BodyBlock>,
    existing: &HashSet<(String, String)>,
) {
    loop {
        let Some(cfg) = super::super::graph::Cfg::from_blocks(blocks) else {
            return;
        };
        let dominators = cfg.dominators();
        let positions = blocks
            .iter()
            .enumerate()
            .map(|(position, block)| (block.name.clone(), position))
            .collect::<HashMap<_, _>>();
        let violation = blocks.iter().find_map(|block| {
            let parent = dominators.idom(&block.name)?;
            let parent_position = positions[parent];
            let node_position = positions[&block.name];
            (parent_position >= node_position
                && !existing.contains(&(parent.to_string(), block.name.clone())))
            .then_some((parent_position, node_position))
        });
        let Some((parent_position, node_position)) = violation else {
            return;
        };
        let parent = blocks.remove(parent_position);
        blocks.insert(node_position, parent);
    }
}

fn next_selection_merge_suffix(blocks: &[BodyBlock]) -> usize {
    let prefix = format!("{SPLIT_PREFIX}{SEL_TOKEN}");
    blocks
        .iter()
        .filter_map(|block| block.name.strip_prefix(&prefix))
        .filter_map(|suffix| suffix.parse::<usize>().ok())
        .max()
        .map_or(0, |max| max + 1)
}

#[derive(Clone, Copy)]
enum EmittedMergeKind {
    Loop,
    Branch,
    Switch,
}

#[derive(Clone)]
struct EmittedMergeClaim {
    header: String,
    target: String,
    kind: EmittedMergeKind,
}

/// Reclassify finalized loop claims whose header no longer has a natural backedge.
///
/// Typed ownership rewrites can eliminate the last backedge after the initial loop forest selected
/// a header. Such a claim must not survive as `OpLoopMerge`: an unconditional/terminal header owns
/// no construct marker, while a conditional or switch retains the former exit as its selection
/// merge. Update the exact maps consumed by emission before numeric labels exist.
#[cfg(test)]
pub(in crate::native) fn normalize_stale_emitted_loop_claims(
    blocks: &[BodyBlock],
    loop_merges: &mut HashMap<String, LoopMergeInfo>,
    branch_merges_by_header: &mut HashMap<String, String>,
    switch_merges: &mut HashMap<String, String>,
) -> bool {
    let forest = analyze(blocks);
    let stale = loop_merges
        .keys()
        .filter(|header| forest.loop_for_header(header).is_none())
        .cloned()
        .collect::<Vec<_>>();
    let mut changed = false;
    for header in stale {
        let Some(typed) = blocks
            .iter()
            .find(|block| block.name == header)
            .and_then(|block| block.typed.as_ref())
        else {
            continue;
        };
        let Some(info) = loop_merges.remove(&header) else {
            continue;
        };
        match &typed.terminator {
            crate::native::tir::TirTerminator::BrCond { .. } => {
                branch_merges_by_header.insert(header, info.merge);
            }
            crate::native::tir::TirTerminator::Switch { .. } => {
                switch_merges.insert(header, info.merge);
            }
            crate::native::tir::TirTerminator::Br(_)
            | crate::native::tir::TirTerminator::Unreachable
            | crate::native::tir::TirTerminator::Ret(_) => {}
        }
        changed = true;
    }
    changed
}

#[cfg(test)]
pub(super) fn phi_value_exact_eq(
    lhs: &crate::native::ir::LlValue,
    rhs: &crate::native::ir::LlValue,
) -> bool {
    use crate::native::ir::LlValue;
    let typed_values_eq =
        |lhs: &[crate::native::ir::TypedValue], rhs: &[crate::native::ir::TypedValue]| {
            lhs.len() == rhs.len()
                && lhs.iter().zip(rhs).all(|(lhs, rhs)| {
                    lhs.ty == rhs.ty && phi_value_exact_eq(&lhs.value, &rhs.value)
                })
        };
    match (lhs, rhs) {
        (LlValue::Local(lhs), LlValue::Local(rhs))
        | (LlValue::Global(lhs), LlValue::Global(rhs)) => lhs == rhs,
        (LlValue::Bool(lhs), LlValue::Bool(rhs)) => lhs == rhs,
        (LlValue::Int(lhs), LlValue::Int(rhs)) | (LlValue::Hex(lhs), LlValue::Hex(rhs)) => {
            lhs == rhs
        }
        (LlValue::SignedInt(lhs), LlValue::SignedInt(rhs)) => lhs == rhs,
        (LlValue::Float(lhs), LlValue::Float(rhs)) => lhs.to_bits() == rhs.to_bits(),
        (LlValue::HalfBits(lhs), LlValue::HalfBits(rhs))
        | (LlValue::BFloatBits(lhs), LlValue::BFloatBits(rhs)) => lhs == rhs,
        (LlValue::Vector(lhs), LlValue::Vector(rhs))
        | (LlValue::Array(lhs), LlValue::Array(rhs))
        | (LlValue::Struct(lhs), LlValue::Struct(rhs)) => typed_values_eq(lhs, rhs),
        (LlValue::Splat(lhs), LlValue::Splat(rhs)) => {
            lhs.ty == rhs.ty && phi_value_exact_eq(&lhs.value, &rhs.value)
        }
        (LlValue::Gep(lhs), LlValue::Gep(rhs)) => {
            lhs.inbounds == rhs.inbounds
                && lhs.source_ty == rhs.source_ty
                && lhs.base.ty == rhs.base.ty
                && phi_value_exact_eq(&lhs.base.value, &rhs.base.value)
                && typed_values_eq(&lhs.indices, &rhs.indices)
        }
        (
            LlValue::IntToPtr {
                source: lhs_source,
                destination: lhs_destination,
            },
            LlValue::IntToPtr {
                source: rhs_source,
                destination: rhs_destination,
            },
        ) => {
            lhs_destination == rhs_destination
                && lhs_source.ty == rhs_source.ty
                && phi_value_exact_eq(&lhs_source.value, &rhs_source.value)
        }
        (LlValue::Zero, LlValue::Zero) | (LlValue::Undef, LlValue::Undef) => true,
        _ => false,
    }
}

fn emitted_merge_claims(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
    branch_merges: &HashMap<(String, String), String>,
    branch_merges_by_header: &HashMap<String, String>,
    branch_merges_header_only: bool,
    switch_merges: &HashMap<String, String>,
) -> Vec<EmittedMergeClaim> {
    let mut claims = Vec::new();
    for block in blocks {
        let header = &block.name;
        if let Some(info) = loop_merges.get(header) {
            claims.push(EmittedMergeClaim {
                header: header.clone(),
                target: info.merge.clone(),
                kind: EmittedMergeKind::Loop,
            });
        }
        let Some(typed) = &block.typed else {
            continue;
        };
        match &typed.terminator {
            crate::native::tir::TirTerminator::BrCond { t, f, .. }
                if !loop_merges.contains_key(header) =>
            {
                let is_loop_continue = loop_merges
                    .get(t)
                    .is_some_and(|info| info.merge == *f && info.continue_target == *header)
                    || loop_merges
                        .get(f)
                        .is_some_and(|info| info.merge == *t && info.continue_target == *header);
                if is_loop_continue {
                    continue;
                }
                let target = branch_merges_by_header.get(header).cloned().or_else(|| {
                    (!branch_merges_header_only)
                        .then(|| branch_merges.get(&(t.clone(), f.clone())).cloned())
                        .flatten()
                });
                if let Some(target) = target {
                    claims.push(EmittedMergeClaim {
                        header: header.clone(),
                        target,
                        kind: EmittedMergeKind::Branch,
                    });
                }
            }
            crate::native::tir::TirTerminator::Switch { .. } => {
                if let Some(target) = switch_merges.get(header) {
                    claims.push(EmittedMergeClaim {
                        header: header.clone(),
                        target: target.clone(),
                        kind: EmittedMergeKind::Switch,
                    });
                }
            }
            _ => {}
        }
    }
    claims
}

/// Privatize shared direct arms only for conditionals/switches with finalized emitted merge claims.
///
/// The established typed cross-arm clone renames every cloned definition and partitions/mirrors phi
/// incomings. Materialize pair-keyed branch claims by header before cloning changes an arm label, then
/// let [`funnel_emitted_selection_merge_bypasses`] route each private clone through the already-declared
/// merge. Headers without an emitted merge are excluded rather than being made to look structured.
#[cfg(test)]
pub(in crate::native) fn privatize_emitted_shared_direct_selection_arms(
    blocks: &mut Vec<BodyBlock>,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    branch_merges: &HashMap<(String, String), String>,
    branch_merges_by_header: &mut HashMap<String, String>,
    branch_merges_header_only: bool,
    switch_merges: &HashMap<String, String>,
) -> bool {
    let claims = emitted_merge_claims(
        blocks,
        loop_merges,
        branch_merges,
        branch_merges_by_header,
        branch_merges_header_only,
        switch_merges,
    );
    let mut eligible_headers = HashSet::new();
    for claim in claims {
        match claim.kind {
            EmittedMergeKind::Loop => {}
            EmittedMergeKind::Branch => {
                eligible_headers.insert(claim.header.clone());
                branch_merges_by_header.insert(claim.header, claim.target);
            }
            EmittedMergeKind::Switch => {
                eligible_headers.insert(claim.header);
            }
        }
    }
    if eligible_headers.is_empty() {
        return false;
    }
    let mut privatized = super::clone_crossarm::privatize_trivial_cross_arm_for_emitted_headers(
        blocks,
        &eligible_headers,
    );
    if privatized.len() == blocks.len() {
        return false;
    }
    funnel_emitted_selection_merge_bypasses(
        &mut privatized,
        loop_merges,
        branch_merges,
        branch_merges_by_header,
        branch_merges_header_only,
        switch_merges,
    );
    if has_emitted_selection_merge_bypass(
        &privatized,
        loop_merges,
        branch_merges,
        branch_merges_by_header,
        branch_merges_header_only,
        switch_merges,
    ) {
        return false;
    }
    *blocks = privatized;
    true
}

#[cfg(test)]
fn has_emitted_selection_merge_bypass(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
    branch_merges: &HashMap<(String, String), String>,
    branch_merges_by_header: &HashMap<String, String>,
    branch_merges_header_only: bool,
    switch_merges: &HashMap<String, String>,
) -> bool {
    let claims = emitted_merge_claims(
        blocks,
        loop_merges,
        branch_merges,
        branch_merges_by_header,
        branch_merges_header_only,
        switch_merges,
    );
    let forest = analyze(blocks);
    let mut predecessors = HashMap::<String, HashSet<String>>::new();
    for block in blocks {
        for successor in block_successors(block) {
            predecessors
                .entry(successor)
                .or_default()
                .insert(block.name.clone());
        }
    }
    claims
        .into_iter()
        .filter(|claim| !matches!(claim.kind, EmittedMergeKind::Loop))
        .any(|claim| {
            let Some(merge_block) = blocks.iter().find(|block| block.name == claim.target) else {
                return false;
            };
            let Some(merge_typed) = merge_block.typed.as_ref() else {
                return false;
            };
            let crate::native::tir::TirTerminator::Br(target) = &merge_typed.terminator else {
                return false;
            };
            if !forest.dominates(&claim.header, &claim.target)
                || forest.dominates(&claim.header, target)
            {
                return false;
            }
            let merge_predecessors = predecessors.get(&claim.target).cloned().unwrap_or_default();
            predecessors
                .get(target)
                .into_iter()
                .flatten()
                .any(|predecessor| {
                    predecessor != &claim.target
                        && !merge_predecessors.contains(predecessor)
                        && forest.dominates(&claim.header, predecessor)
                })
        })
}

/// Route every header-owned edge through a declared pass-through selection merge before emission.
///
/// For `H -> { A -> M -> T, B -> T }`, `B` bypasses `H`'s declared merge `M`. When `M` is a direct
/// phi-prefix pass-through, move the exact `B -> T` phi values into `M` and redirect that edge to
/// `M`. The transaction declines unless every merge phi is represented by the target phi and every
/// bypass value is exact; no value or ownership is inferred from names.
#[cfg(test)]
pub(in crate::native) fn funnel_emitted_selection_merge_bypasses(
    blocks: &mut [BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
    branch_merges: &HashMap<(String, String), String>,
    branch_merges_by_header: &HashMap<String, String>,
    branch_merges_header_only: bool,
    switch_merges: &HashMap<String, String>,
) -> bool {
    let selection_claims = emitted_merge_claims(
        blocks,
        loop_merges,
        branch_merges,
        branch_merges_by_header,
        branch_merges_header_only,
        switch_merges,
    )
    .into_iter()
    .filter(|claim| !matches!(claim.kind, EmittedMergeKind::Loop))
    .collect::<Vec<_>>();
    let mut changed = false;
    loop {
        let forest = analyze(blocks);
        let mut predecessors = HashMap::<String, Vec<String>>::new();
        for block in blocks.iter() {
            for successor in block_successors(block) {
                let entry = predecessors.entry(successor).or_default();
                if !entry.contains(&block.name) {
                    entry.push(block.name.clone());
                }
            }
        }
        let mut plan = None;
        for claim in &selection_claims {
            let Some(merge_idx) = blocks.iter().position(|block| block.name == claim.target) else {
                continue;
            };
            let Some(merge_typed) = blocks[merge_idx].typed.as_ref() else {
                continue;
            };
            let crate::native::tir::TirTerminator::Br(target) = &merge_typed.terminator else {
                continue;
            };
            let target = target.clone();
            if !forest.dominates(&claim.header, &claim.target)
                || forest.dominates(&claim.header, &target)
            {
                continue;
            }
            if merge_typed
                .insts
                .iter()
                .any(|instruction| !instruction.is_phi())
            {
                continue;
            }
            let merge_predecessors = predecessors.get(&claim.target).cloned().unwrap_or_default();
            let merge_predecessor_set = merge_predecessors.iter().cloned().collect::<HashSet<_>>();
            let bypasses = predecessors
                .get(&target)
                .into_iter()
                .flatten()
                .filter(|predecessor| {
                    predecessor.as_str() != claim.target
                        && !merge_predecessor_set.contains(*predecessor)
                        && forest.dominates(&claim.header, predecessor)
                })
                .cloned()
                .collect::<HashSet<_>>();
            if bypasses.is_empty() {
                continue;
            }
            let Some(target_idx) = blocks.iter().position(|block| block.name == target) else {
                continue;
            };
            let Some(target_typed) = blocks[target_idx].typed.as_ref() else {
                continue;
            };
            let merge_phis = merge_typed
                .insts
                .iter()
                .take_while(|instruction| instruction.is_phi())
                .filter_map(|instruction| {
                    Some((
                        instruction.result.clone()?,
                        instruction.phi_incoming().as_ref()?.1.clone(),
                    ))
                })
                .collect::<HashMap<_, _>>();
            if merge_phis.len()
                != merge_typed
                    .insts
                    .iter()
                    .take_while(|instruction| instruction.is_phi())
                    .count()
            {
                continue;
            }
            let mut mapped_merge_phis = HashSet::new();
            let mut merge_phi_additions = Vec::new();
            let mut target_phi_updates = Vec::new();
            let mut used_results = blocks
                .iter()
                .filter_map(|block| block.typed.as_ref())
                .flat_map(|typed| &typed.insts)
                .filter_map(|instruction| instruction.result.clone())
                .collect::<HashSet<_>>();
            let mut funnel_counter = 0usize;
            let mut valid = true;
            for instruction in target_typed
                .insts
                .iter()
                .take_while(|instruction| instruction.is_phi())
            {
                let (Some(result), Some((ty, incoming))) = (
                    instruction.result.clone(),
                    instruction.phi_incoming().as_ref(),
                ) else {
                    valid = false;
                    break;
                };
                let Some((merge_value, _)) = incoming
                    .iter()
                    .find(|(_, predecessor)| predecessor == &claim.target)
                else {
                    valid = false;
                    break;
                };
                let bypass_incoming = incoming
                    .iter()
                    .filter(|(_, predecessor)| bypasses.contains(predecessor))
                    .cloned()
                    .collect::<Vec<_>>();
                if bypass_incoming.len() != bypasses.len() {
                    valid = false;
                    break;
                }
                let mut replacement = None;
                if let crate::native::ir::LlValue::Local(merge_result) = merge_value {
                    if merge_phis.contains_key(merge_result) {
                        if !mapped_merge_phis.insert(merge_result.clone()) {
                            valid = false;
                            break;
                        }
                        merge_phi_additions.push((merge_result.clone(), None, bypass_incoming));
                    } else if bypass_incoming
                        .iter()
                        .any(|(value, _)| !phi_value_exact_eq(value, merge_value))
                    {
                        let funnel_result = loop {
                            let candidate = format!("{}.funnel{funnel_counter}", claim.target);
                            funnel_counter += 1;
                            if used_results.insert(candidate.clone()) {
                                break candidate;
                            }
                        };
                        let mut funnel_incoming = merge_predecessors
                            .iter()
                            .map(|predecessor| (merge_value.clone(), predecessor.clone()))
                            .collect::<Vec<_>>();
                        funnel_incoming.extend(bypass_incoming);
                        merge_phi_additions.push((
                            funnel_result.clone(),
                            Some(ty.clone()),
                            funnel_incoming,
                        ));
                        replacement = Some(funnel_result);
                    }
                } else if bypass_incoming
                    .iter()
                    .any(|(value, _)| !phi_value_exact_eq(value, merge_value))
                {
                    let funnel_result = loop {
                        let candidate = format!("{}.funnel{funnel_counter}", claim.target);
                        funnel_counter += 1;
                        if used_results.insert(candidate.clone()) {
                            break candidate;
                        }
                    };
                    let mut funnel_incoming = merge_predecessors
                        .iter()
                        .map(|predecessor| (merge_value.clone(), predecessor.clone()))
                        .collect::<Vec<_>>();
                    funnel_incoming.extend(bypass_incoming);
                    merge_phi_additions.push((
                        funnel_result.clone(),
                        Some(ty.clone()),
                        funnel_incoming,
                    ));
                    replacement = Some(funnel_result);
                }
                target_phi_updates.push((
                    result,
                    incoming
                        .iter()
                        .filter(|(_, predecessor)| !bypasses.contains(predecessor))
                        .map(|(value, predecessor)| {
                            let value = if predecessor == &claim.target {
                                replacement
                                    .as_ref()
                                    .map(|result| crate::native::ir::LlValue::Local(result.clone()))
                                    .unwrap_or_else(|| value.clone())
                            } else {
                                value.clone()
                            };
                            (value, predecessor.clone())
                        })
                        .collect::<Vec<_>>(),
                ));
            }
            if !valid || mapped_merge_phis.len() != merge_phis.len() {
                continue;
            }
            plan = Some((
                merge_idx,
                target_idx,
                claim.target.clone(),
                target,
                bypasses,
                merge_phi_additions,
                target_phi_updates,
            ));
            break;
        }
        let Some((
            merge_idx,
            target_idx,
            merge,
            target,
            bypasses,
            merge_phi_additions,
            target_phi_updates,
        )) = plan
        else {
            break;
        };
        for block in blocks.iter_mut() {
            if bypasses.contains(&block.name) {
                if let Some(typed) = block.typed_mut() {
                    typed.redirect_successor(&target, &merge);
                }
            }
        }
        if let Some(typed) = blocks[merge_idx].typed_mut() {
            for (result, new_type, additions) in merge_phi_additions {
                if let Some(ty) = new_type {
                    typed.push_value_phi(&result, &ty, &additions);
                } else {
                    for (value, predecessor) in additions {
                        typed.append_phi_incoming(&result, value, &predecessor);
                    }
                }
            }
        }
        if let Some(typed) = blocks[target_idx].typed_mut() {
            for (result, incoming) in target_phi_updates {
                typed.set_phi_incomings(&result, &incoming);
            }
        }
        changed = true;
    }
    changed
}

/// Give every structurally owned emitted construct its own merge before numeric SPIR-V labels exist.
///
/// Fallback merge inference can assign one natural post-dominator to nested constructs or an exit
/// also entered from outside the construct. The typed CFG and the finalized effective merge maps
/// contain the exact incoming edges for every provably owned claim, so split shared and
/// non-dominated claims here instead of rediscovering them from emitted `Op*Merge` instructions.
/// Redirect only edges dominated by the claimant but not by the target: when the target is itself a
/// loop header, its target-dominated predecessors are backedges and must continue to enter it
/// directly. Phi values are folded through the private pass-through by the same typed primitive used
/// by the primary structurizer.
pub(in crate::native) fn privatize_reused_emitted_merge_targets(
    blocks: &mut Vec<BodyBlock>,
    loop_merges: &mut HashMap<String, LoopMergeInfo>,
    branch_merges: &HashMap<(String, String), String>,
    branch_merges_by_header: &mut HashMap<String, String>,
    branch_merges_header_only: bool,
    switch_merges: &mut HashMap<String, String>,
) -> bool {
    let claims = emitted_merge_claims(
        blocks,
        loop_merges,
        branch_merges,
        branch_merges_by_header,
        branch_merges_header_only,
        switch_merges,
    );

    let claim_counts = claims.iter().fold(HashMap::new(), |mut counts, claim| {
        *counts.entry(claim.target.clone()).or_insert(0usize) += 1;
        counts
    });
    let continue_owners = loop_merges
        .iter()
        .map(|(header, info)| (info.continue_target.as_str(), header.as_str()))
        .collect::<Vec<_>>();
    // Every question this pass asks the graph is a dominance question, so it asks for dominance and
    // not for the natural-loop forest built around it.
    let mut dominance = block_dominators(blocks);
    let mut shared = claims
        .into_iter()
        .filter(|claim| {
            claim_counts.get(&claim.target).copied().unwrap_or(0) > 1
                || !dominance.dominates(&claim.header, &claim.target)
                || continue_owners
                    .iter()
                    .any(|(target, owner)| *target == claim.target && *owner != claim.header)
        })
        .collect::<Vec<_>>();
    if shared.is_empty() {
        return false;
    }

    // Use inner-first ownership order.
    shared.sort_by_cached_key(|claim| {
        let dominated = blocks
            .iter()
            .filter(|block| dominance.dominates(&claim.header, &block.name))
            .count();
        (claim.target.clone(), dominated, claim.header.clone())
    });
    let mut counter = next_selection_merge_suffix(blocks);
    let mut changed = false;
    for claim in shared {
        // Each split is recorded rather than re-analyzed, so the synthesized pass-throughs
        // participate in the next claim's ownership question directly -- see
        // [`PassThroughDominance`] for why that is the same answer a fresh analysis gives, and for
        // what recomputing it per claim costs on a function with one construct per block.
        let predecessors = blocks
            .iter()
            .filter(|block| {
                dominance.dominates(&claim.header, &block.name)
                    && !dominance.dominates(&claim.target, &block.name)
                    && block_successors(block)
                        .iter()
                        .any(|target| target == &claim.target)
            })
            .map(|block| block.name.clone())
            .collect::<Vec<_>>();
        let private = if block_has_phi(blocks, &claim.target) {
            synth_unique_selection_merge_phi_explicit(
                blocks,
                &predecessors,
                &claim.target,
                &HashSet::new(),
                &mut counter,
            )
        } else {
            synth_unique_selection_merge_no_phi_explicit(
                blocks,
                &predecessors,
                &claim.target,
                &HashSet::new(),
                &mut counter,
            )
        };
        // A rejected/unstructured source plan can name a post-dominator reached only through arms
        // with external entries, leaving this header no directly owned predecessor to refunnel. Do
        // not mask a later semantic unsupported diagnostic (for example an unresolved authored
        // function-table call), and do not guess ownership. Successful plans have an exact split;
        // otherwise construction selects the raw-CFG representation.
        let Some(private) = private else {
            continue;
        };
        // The synthesis redirects exactly the predecessor edges named above, and `private` branches
        // only on to `claim.target`.
        dominance.record_pass_through(&private, &predecessors);
        match claim.kind {
            EmittedMergeKind::Loop => {
                loop_merges
                    .get_mut(&claim.header)
                    .expect("recorded loop merge claim")
                    .merge = private;
            }
            EmittedMergeKind::Branch => {
                branch_merges_by_header.insert(claim.header, private);
            }
            EmittedMergeKind::Switch => {
                switch_merges.insert(claim.header, private);
            }
        }
        changed = true;
    }
    changed
}

/// Give every loop a private merge after an ownership transform introduces an external predecessor
/// to the merge selected by the original loop forest.
///
/// Construct-tree routing can preserve the loop body exactly while making its old exit reachable
/// from an enclosing sibling. The loop header then no longer dominates that exit, so it cannot remain
/// the loop's declared merge. Reuse the existing phi-aware overlap split to funnel only in-loop exit
/// edges through a fresh merge; outside edges and their phi values remain on the old exit.
pub(in crate::native) fn privatize_nondominated_loop_merges(
    blocks: &mut Vec<BodyBlock>,
    loop_merges: &mut HashMap<String, LoopMergeInfo>,
) -> bool {
    let mut counter = blocks.len();
    while blocks
        .iter()
        .any(|block| block.name == format!("{SPLIT_PREFIX}{counter}"))
    {
        counter += 1;
    }
    let mut changed = false;
    loop {
        let forest = analyze(blocks);
        let mut polluted = loop_merges
            .iter()
            .filter_map(|(header, info)| {
                let loop_info = forest.loop_for_header(header)?;
                (!forest.dominates(header, &info.merge)).then_some((
                    loop_info.body.len(),
                    header.clone(),
                    info.merge.clone(),
                ))
            })
            .collect::<Vec<_>>();
        polluted.sort();
        let Some((_, header, merge)) = polluted.into_iter().next() else {
            break;
        };
        while blocks
            .iter()
            .any(|block| block.name == format!("{SPLIT_PREFIX}{counter}"))
        {
            counter += 1;
        }
        let split = if block_has_phi(blocks, &merge) {
            split_phi_overlap(blocks, &forest, &header, &merge, &mut counter)
        } else {
            split_no_phi_overlap(blocks, &forest, &header, &merge, &mut counter)
        };
        let Some(private_merge) = split else {
            break;
        };
        let Some(info) = loop_merges.get_mut(&header) else {
            break;
        };
        info.merge = private_merge;
        changed = true;
    }
    changed
}

pub(in crate::native) fn loop_role_targets_with_passthroughs(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
) -> HashSet<String> {
    let by_name = blocks
        .iter()
        .map(|block| (block.name.as_str(), block))
        .collect::<HashMap<_, _>>();
    let mut out = HashSet::new();
    for info in loop_merges.values() {
        add_loop_role_target_with_passthroughs(&by_name, &mut out, &info.merge);
        add_loop_role_target_with_passthroughs(&by_name, &mut out, &info.continue_target);
    }
    out
}

fn add_loop_role_target_with_passthroughs(
    by_name: &HashMap<&str, &BodyBlock>,
    out: &mut HashSet<String>,
    role: &str,
) {
    let mut current = role.to_string();
    for _ in 0..=by_name.len() {
        if !out.insert(current.clone()) {
            break;
        }
        let Some(block) = by_name.get(current.as_str()) else {
            break;
        };
        if !matches!(
            block.role,
            BlockRole::LMerge | BlockRole::ConstructTreeRoute
        ) {
            break;
        }
        let successors = block_successors(block);
        if successors.len() != 1 || successors[0] == current {
            break;
        }
        current = successors[0].clone();
    }
}

fn construct_tree_redirect_candidate(block: &BodyBlock) -> bool {
    matches!(block.role, BlockRole::Normal | BlockRole::LMerge)
}

/// Routes that still feed `natural` (directly). Used when a regional construct-tree wrapper has
/// rewritten a former Normal/LMerge reconvergence predecessor into a `ConstructTreeRoute` gateway:
/// the selection header no longer has a source-owned edge into `natural`, but the Normal/LMerge
/// blocks that *enter* those routes are still header-dominated and can be reclaimed onto a private
/// merge without rewriting the route edge itself.
fn construct_tree_routes_into_natural(blocks: &[BodyBlock], natural: &str) -> HashSet<String> {
    blocks
        .iter()
        .filter(|block| block.role == BlockRole::ConstructTreeRoute)
        .filter(|block| {
            block_successors(block)
                .iter()
                .any(|target| target == natural)
        })
        .map(|block| block.name.clone())
        .collect()
}

/// Header-dominated Normal/LMerge predecessors that either target `natural` directly or enter a
/// construct-tree route that targets `natural`. Empty when the ordinary direct sweep already covers
/// every reconvergence edge (routes are never claimed as ownership predecessors).
fn construct_tree_preds_reclaiming_routes(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    header: &str,
    natural: &str,
) -> Vec<String> {
    let routes = construct_tree_routes_into_natural(blocks, natural);
    if routes.is_empty() {
        return Vec::new();
    }
    let mut preds = Vec::new();
    for block in blocks {
        if !construct_tree_redirect_candidate(block) || !forest.dominates(header, &block.name) {
            continue;
        }
        let hits = block_successors(block)
            .into_iter()
            .any(|target| target == natural || routes.iter().any(|route| route == &target));
        if hits {
            preds.push(block.name.clone());
        }
    }
    preds.sort();
    preds.dedup();
    preds
}

/// Compute forest-driven loop merges, splitting merge==continue overlaps and pure self-latching
/// no-exit loops in place.
///
/// Returns the (possibly block-augmented) body in program order and a `header -> LoopMergeInfo` map
/// covering the loops this increment handles. The map is a drop-in for the subset of
/// `infer_loop_merges` results it covers; callers may union it with the heuristic map for the rest.
pub(in crate::native) fn forest_loop_merges(
    blocks: &[BodyBlock],
    converge_inloop: bool,
    multi_exit_clone: bool,
) -> (Vec<BodyBlock>, HashMap<String, LoopMergeInfo>) {
    let initial_forest = analyze(blocks);
    let mut loop_headers = initial_forest
        .loops
        .iter()
        .map(|loop_info| (loop_info.body.len(), loop_info.header.clone()))
        .collect::<Vec<_>>();
    loop_headers.sort();
    let mut out_blocks = blocks.to_vec();
    let mut merges = HashMap::new();
    let mut split_counter = 0usize;
    let mut collision_cache: Option<(LoopForest, HashMap<String, String>)> = None;

    // Inner ownership must be materialized first. Recompute the current forest and this header's plan
    // after every nested transform so an enclosing loop sees synthesized inner dispatch/backedge
    // blocks as part of its body and owns their exits too; reordering a stale source plan is insufficient.
    for (_, header) in loop_headers {
        let forest = analyze(&out_blocks);
        // A bare unreachable target terminates the invocation in-place; unlike a live continuation it
        // does not need to be selected as the loop's unique merge. Keep it in the CFG/construct, but
        // exclude it from merge-candidate cardinality after proving the exact terminal shape.
        let terminal_exits = out_blocks
            .iter()
            .filter(|block| block_ends_in_unreachable(&out_blocks, &block.name))
            .map(|block| block.name.clone())
            .collect::<HashSet<_>>();
        let current_plans = forest.structured_plan_ignoring_exits(&terminal_exits);
        let Some(plan) = current_plans.iter().find(|plan| plan.header == header) else {
            continue;
        };
        // A pure infinite loop with its header as the sole latch has no distinct continue target and
        // no source exit to name as its merge. Split the header's body away from its phi block, route
        // its old self-edge through a fresh continue, and add an unreachable merge. This preserves
        // non-termination while giving `OpLoopMerge` the distinct merge/continue blocks SPIR-V needs.
        // Kept deliberately narrow: a conditional/multi-block no-exit loop needs its own construction.
        if plan.restructure.as_slice() == [Restructure::NoExit]
            && plan.continue_block.as_deref() == Some(plan.header.as_str())
        {
            if let Some((merge, continue_target)) =
                synth_noexit_self_latch(&mut out_blocks, &plan.header, &mut split_counter)
            {
                collision_cache = None;
                merges.insert(
                    plan.header.clone(),
                    LoopMergeInfo {
                        merge,
                        continue_target,
                    },
                );
            }
            continue;
        }

        let (Some(continue_target), Some(merge_block)) =
            (plan.continue_block.clone(), plan.merge_block.clone())
        else {
            if crate::env_vars::flm_why() {
                eprintln!(
                    "[flm-why] header={} restructure={:?} continue={:?} merge={:?} uncovered=missing-role",
                    plan.header,
                    plan.restructure,
                    plan.continue_block,
                    plan.merge_block,
                );
            }
            continue;
        };

        if plan.restructure.is_empty() {
            // Loop-merge ⇄ selection-merge collision: if the loop's merge block is ALSO the
            // post-dominator (natural selection merge) of a conditional OUTSIDE this loop, a single
            // block would have to be both a loop merge and a selection merge — illegal in SPIR-V (a
            // merge block belongs to exactly one construct). Give the LOOP a distinct merge: redirect
            // its in-loop exit edges to `merge_block` through a fresh pass-through (with phi surgery
            // when `merge_block` carries a phi), leaving the original block as the enclosing
            // selection's merge. This is the dominant `cond-phi-shared/loop-role` frontier shape.
            if collision_cache.is_none() {
                let det_forest = analyze(&out_blocks);
                let det_selection_merges = selection_merges(&out_blocks, &det_forest);
                collision_cache = Some((det_forest, det_selection_merges));
            }
            let (det_forest, det_selection_merges) = collision_cache.as_ref().unwrap();
            let collides = merge_collides_with_outer_selection_from(
                det_forest,
                det_selection_merges,
                &plan.header,
                &merge_block,
                converge_inloop,
            );
            if crate::env_vars::flm_why() {
                eprintln!(
                    "[flm-why] header={} restructure={:?} merge={} collides={} has_phi={}",
                    plan.header,
                    plan.restructure,
                    merge_block,
                    collides,
                    block_has_phi(&out_blocks, &merge_block),
                );
            }
            let mut mutated = false;
            let merge_block = if collides {
                let split = if block_has_phi(&out_blocks, &merge_block) {
                    split_phi_overlap(
                        &mut out_blocks,
                        det_forest,
                        &plan.header,
                        &merge_block,
                        &mut split_counter,
                    )
                } else {
                    split_no_phi_overlap(
                        &mut out_blocks,
                        det_forest,
                        &plan.header,
                        &merge_block,
                        &mut split_counter,
                    )
                };
                mutated |= split.is_some();
                split.unwrap_or(merge_block)
            } else {
                merge_block
            };
            // do-while normalization: when the latch (continue) block itself ends in a conditional
            // that exits to the loop merge (the exit test is at the loop bottom), split off a separate
            // unconditional continue block so the latch becomes an ordinary {continue, merge} break.
            let rotated_continue = synth_dowhile_continue(
                &mut out_blocks,
                &plan.header,
                &continue_target,
                &merge_block,
                &mut split_counter,
            );
            mutated |= rotated_continue.is_some();
            let continue_target = rotated_continue.unwrap_or(continue_target);
            if mutated {
                collision_cache = None;
            }
            merges.insert(
                plan.header.clone(),
                LoopMergeInfo {
                    merge: merge_block,
                    continue_target,
                },
            );
            continue;
        }

        // Multiple-exit loops: funnel both exits through one synthesized dispatch merge (k == 2,
        // no-phi exits). On success the loop is single-exit; otherwise left for the fallback path.
        if plan.restructure.as_slice() == [Restructure::MultipleExits] {
            let exits = forest
                .loop_for_header(&plan.header)
                .map(|l| l.exits.clone())
                .unwrap_or_default();
            if let Some(new_merge) = synth_multi_exit_merge(
                &mut out_blocks,
                &forest,
                &plan.header,
                &exits,
                &mut split_counter,
            ) {
                collision_cache = None;
                // A dispatch exit arm reached ALSO from outside the loop (a shared exit) is not
                // `M`-dominated; clone its dominated forward region so both arms reconverge at an
                // `M`-dominated merge (M2 residual fix). Only in the reject-triggered clone attempt
                // (`multi_exit_clone`) — running it on every attempt would corrupt a function that
                // already admits at the base attempt (turning its valid dispatch into a reject→repair).
                if multi_exit_clone {
                    privatize_dispatch_shared_exits(
                        &mut out_blocks,
                        &new_merge,
                        &mut split_counter,
                    );
                }
                // The funnelled loop may now be a do-while (its latch conditionally exits to the new
                // dispatch merge); rotate that latch into a separate continue block too.
                let continue_target = synth_dowhile_continue(
                    &mut out_blocks,
                    &plan.header,
                    &continue_target,
                    &new_merge,
                    &mut split_counter,
                )
                .unwrap_or(continue_target);
                merges.insert(
                    plan.header.clone(),
                    LoopMergeInfo {
                        merge: new_merge,
                        continue_target,
                    },
                );
            }
            continue;
        }

        // Multiple-latch loops: unify the back-edges into one synthesized latch so the loop is
        // single-latch and directly structurable. Mirror of the multi-exit funnel on the continue side.
        // Fires only for a pure-`[MultipleLatches]` loop, which currently rejects, so it can only ADD
        // admissions (never corrupt an already-admitting fn).
        if plan.restructure.as_slice() == [Restructure::MultipleLatches] {
            let latches = forest
                .loop_for_header(&plan.header)
                .map(|l| l.latches.clone())
                .unwrap_or_default();
            if let Some(new_latch) = synth_multi_latch_continue(
                &mut out_blocks,
                &plan.header,
                &latches,
                &mut split_counter,
            ) {
                collision_cache = None;
                // The unified loop may now be a do-while (its single latch conditionally exits to the
                // merge); rotate that latch into a separate unconditional continue block too.
                let continue_target = synth_dowhile_continue(
                    &mut out_blocks,
                    &plan.header,
                    &new_latch,
                    &merge_block,
                    &mut split_counter,
                )
                .unwrap_or(new_latch);
                merges.insert(
                    plan.header.clone(),
                    LoopMergeInfo {
                        merge: merge_block,
                        continue_target,
                    },
                );
            }
            continue;
        }

        // Only the lone merge==continue overlap is handled here; anything else is left for the
        // existing path (and the later increments of the consumer).
        if plan.restructure.as_slice() != [Restructure::MergeIsEnclosingContinue] {
            if crate::env_vars::flm_why() {
                eprintln!(
                    "[flm-why] header={} restructure={:?} continue={} merge={} uncovered=unsupported-combination",
                    plan.header, plan.restructure, continue_target, merge_block,
                );
            }
            continue;
        }
        let split = if block_has_phi(&out_blocks, &merge_block) {
            // Phi-carrying shared block: the redirected predecessors' incomings are merged into a phi
            // in the synthesized pass-through block, and the shared block's phi takes that merged
            // value via the pass-through edge (preserving the enclosing continue's own incomings).
            split_phi_overlap(
                &mut out_blocks,
                &forest,
                &plan.header,
                &merge_block,
                &mut split_counter,
            )
        } else {
            split_no_phi_overlap(
                &mut out_blocks,
                &forest,
                &plan.header,
                &merge_block,
                &mut split_counter,
            )
        };
        if let Some(new_merge) = split {
            collision_cache = None;
            // The split redirected this loop's in-body predecessors of the shared merge (including its
            // latch) to `new_merge`; if the latch is now a do-while bottom test (conditionally branches
            // back to the header or out to `new_merge`), rotate it into a clean unconditional continue
            // too — exactly as the empty/MultipleExits branches do. Without this the latch stays both
            // the loop continue AND a conditional, which spirv-val rejects ("block exits the continue,
            // but not via a structured exit").
            let continue_target = synth_dowhile_continue(
                &mut out_blocks,
                &plan.header,
                &continue_target,
                &new_merge,
                &mut split_counter,
            )
            .unwrap_or(continue_target);
            merges.insert(
                plan.header.clone(),
                LoopMergeInfo {
                    merge: new_merge,
                    continue_target,
                },
            );
        }
    }

    // Loop-header-is-also-a-selection split: a header carrying both the loop's OpLoopMerge and a genuine
    // in-loop conditional/switch is illegal SPIR-V; give the conditional/switch its own block so the
    // header only branches unconditionally and the lifted block becomes a structurable selection/switch
    // header. Applied last (after all merge/continue are final) and sorted for determinism; each split is
    // local to one header. A header is at most one shape, so calling both is safe — each no-ops on the
    // other's terminator kind.
    let mut header_infos: Vec<(String, String, String)> = merges
        .iter()
        .map(|(h, i)| (h.clone(), i.merge.clone(), i.continue_target.clone()))
        .collect();
    header_infos.sort();
    let final_forest = analyze(&out_blocks);
    for (h, m, c) in header_infos {
        let Some(loop_info) = final_forest.loop_for_header(&h) else {
            continue;
        };
        let loop_body = loop_info.body.iter().cloned().collect::<HashSet<_>>();
        split_loop_header_selection(&mut out_blocks, &h, &m, &c, &loop_body, &mut split_counter);
        split_loop_header_switch(&mut out_blocks, &h, &m, &c, &loop_body, &mut split_counter);
    }

    (out_blocks, merges)
}

/// Structure a pure self-latching infinite loop. LLVM permits a header which carries its own phis,
/// executes the whole body, and ends in `br label %header`; SPIR-V requires an `OpLoopMerge` to name
/// distinct continue and merge blocks. The transform makes that implicit layout explicit:
///
/// ```text
/// preheader -> H(phi, body, br H)    =>    preheader -> H(phi, br B)
///                                             B(body, br C)
///                                             C(br H)
///                                             M(unreachable)
/// ```
///
/// Header phi back-edge predecessors are retargeted from `H` to `C`. The unreachable merge is never
/// executed, exactly matching the source no-exit loop, but remains a real block for `OpLoopMerge`.
/// Returns the `(merge, continue)` pair only for an unconditional direct self branch; broader no-exit
/// shapes remain on the fallback path.
pub(in crate::native) fn synth_noexit_self_latch(
    blocks: &mut Vec<BodyBlock>,
    header: &str,
    counter: &mut usize,
) -> Option<(String, String)> {
    let header_idx = blocks.iter().position(|b| b.name == header)?;
    // Self-latch guard: the header ends in an unconditional `br` back to itself
    // (`TirTerminator::Br(header)`). The carrier is the sole substrate now.
    let src_typed = blocks[header_idx].typed.clone()?;
    if !matches!(&src_typed.terminator, crate::native::tir::TirTerminator::Br(l) if l == header) {
        return None;
    }

    let id = *counter;
    *counter += 1;
    let body = format!("%metal2vulkan.noexit.body.{id}");
    let continue_target = format!("%metal2vulkan.noexit.cont.{id}");
    let merge = format!("%metal2vulkan.noexit.merge.{id}");

    // Build both new carriers off the SOURCE header carrier (kb "STEP-1/STEP-2 decomposition"). The body
    // is the header's NON-PHI instruction suffix (`insts[phi_count..]`, real types already resolved) + a
    // terminator the self-latch guard fixes to `br label {continue_target}` (the header's
    // `br label {header}` redirected). The header is the leading phi PREFIX (`insts[..phi_count]`) with
    // its self back-edge predecessor rewritten to the continue target + a `br label {body}` terminator.
    // Both byte-identical to re-lowering the rewritten lines by construction (`from_suffix`/`prefix`
    // reuse already-typed insts; `rewrite_phi_predecessor` has a `== re-lower` test).
    let header_name = blocks[header_idx].name.clone();
    let phi_count = src_typed.insts.iter().take_while(|i| i.is_phi()).count();
    let body_typed = crate::native::tir::lower_block_carrier_from_suffix(
        &body,
        &src_typed,
        phi_count,
        &format!("br label {continue_target}"),
    );
    blocks[header_idx].typed = crate::native::tir::lower_block_carrier_prefix(
        &header_name,
        &src_typed,
        phi_count,
        &format!("br label {body}"),
    )
    .map(|mut h| {
        h.rewrite_phi_predecessor(header, &continue_target);
        h
    })
    .map(Into::into);
    blocks.insert(
        header_idx + 1,
        BodyBlock {
            name: body,
            role: BlockRole::Normal,
            typed: body_typed.map(Into::into),
        },
    );
    let cont_typed = crate::native::tir::lower_block_carrier(
        &continue_target,
        &[format!("br label {header}")],
        &std::collections::HashMap::new(),
    );
    blocks.insert(
        header_idx + 2,
        BodyBlock {
            name: continue_target.clone(),
            role: BlockRole::Normal,
            typed: cont_typed.map(Into::into),
        },
    );
    // Keep this after the reachable source blocks. `structured_order` appends unreachable blocks, so
    // the merge follows the loop body while the pruning pass retains it through the loop declaration.
    let merge_typed = crate::native::tir::lower_block_carrier(
        &merge,
        &["unreachable".to_string()],
        &std::collections::HashMap::new(),
    );
    blocks.push(BodyBlock {
        name: merge.clone(),
        role: BlockRole::Normal,
        typed: merge_typed.map(Into::into),
    });

    Some((merge, continue_target))
}

/// Per-construct selection-merge synthesis (R2 relooper module 2).
///
/// Each conditional/switch header's natural merge is its immediate post-dominator
/// ([`selection_merges`]). A SPIR-V merge block belongs to exactly ONE construct, so where a
/// post-dominator is shared by two-or-more headers, or collides with a loop's merge/continue, this
/// gives the colliding header a fresh UNIQUE merge: it inserts an empty pass-through block, redirects
/// that header's region predecessors of the shared block to it, and branches it on to the shared
/// block (so other constructs still reach the original). Headers whose natural merge is unique are
/// left on it. Returns the augmented blocks + `branch_merges` (keyed `(true,false)`, the emitter's
/// `branch_merges` key) + `switch_merges` (keyed by header).
///
/// Scope: NO-PHI — a header whose shared merge carries a phi is skipped (omitted from the returned
/// maps), since rewiring it needs the phi-merge surgery [`split_phi_overlap`] does for loops; that is
/// the next sub-increment. Likewise this handles the *flat* collision (sibling/loop sharing); deeply
/// nested constructs that share a post-dominator are validated only by the wholesale floor A/B (module
/// 4), where the integration correctness this unit cannot prove is gated.
pub(in crate::native) fn unique_selection_merges(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
    break_aware: bool,
) -> (
    Vec<BodyBlock>,
    HashMap<(String, String), String>,
    HashMap<String, String>,
) {
    let (blocks, branch, _branch_by_header, switch) =
        unique_selection_merges_with_loop_exit(blocks, loop_merges, break_aware, false);
    (blocks, branch, switch)
}

/// [`unique_selection_merges`] with the reject-only loop-exit convergence extension enabled.
/// The extension is kept out of the ordinary helper so earlier planner alternatives stay
/// byte-identical; only the final `structured_plan` alternative asks it to replace a loop-merge
/// post-dominator with a proven in-loop convergence block.
pub(in crate::native) fn unique_selection_merges_with_loop_exit(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
    break_aware: bool,
    loop_exit_selection: bool,
) -> (
    Vec<BodyBlock>,
    HashMap<(String, String), String>,
    HashMap<String, String>,
    HashMap<String, String>,
) {
    unique_selection_merges_with_loop_exit_and_forced(
        blocks,
        loop_merges,
        break_aware,
        loop_exit_selection,
        &HashMap::new(),
    )
}

/// Variant of [`unique_selection_merges_with_loop_exit`] that accepts a small set of explicit
/// terminal-exit merges. A header in `forced_terminal_merges` has one arm that reaches a proved
/// function return, so ordinary post-dominance cannot name its non-return continuation. The caller
/// supplies the private continuation merge and this builder records it exactly like a synthesized
/// selection merge while leaving all other headers on the ordinary derivation.
pub(in crate::native) fn unique_selection_merges_with_loop_exit_and_forced(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
    break_aware: bool,
    loop_exit_selection: bool,
    forced_terminal_merges: &HashMap<String, String>,
) -> (
    Vec<BodyBlock>,
    HashMap<(String, String), String>,
    HashMap<String, String>,
    HashMap<String, String>,
) {
    unique_selection_merges_with_loop_exit_and_forced_inner(
        blocks,
        loop_merges,
        break_aware,
        loop_exit_selection,
        forced_terminal_merges,
        false,
    )
}

/// Construct-tree candidates may carry a stronger source ownership proof than the final synthesized
/// CFG dominance forest can express.  The regional wrapper is a flat state dispatcher: it preserves
/// the original edge semantics, but later merge synthesis can erase static dominance for a header
/// whose natural merge was owned in the pre-synthesis graph.  This variant keeps ordinary emission
/// byte-identical and only lets construct-tree-owned candidates retry the unique-merge split using
/// the immutable pre-synthesis forest when the current forest has no redirectable predecessor.
pub(in crate::native) fn unique_selection_merges_with_construct_tree_ownership(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
    break_aware: bool,
    loop_exit_selection: bool,
    forced_terminal_merges: &HashMap<String, String>,
) -> (
    Vec<BodyBlock>,
    HashMap<(String, String), String>,
    HashMap<String, String>,
    HashMap<String, String>,
) {
    unique_selection_merges_with_loop_exit_and_forced_inner(
        blocks,
        loop_merges,
        break_aware,
        loop_exit_selection,
        forced_terminal_merges,
        true,
    )
}

fn finish_construct_tree_selection_owner(
    blocks: &mut Vec<BodyBlock>,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    header_merges: &mut HashMap<String, String>,
    terminal_links: &TerminalParentLinks,
    pure_enclosing_owners: &PureEnclosingSelectionOwners,
    owner: &str,
    counter: &mut usize,
) {
    let completed =
        complete_terminal_parent_ownership(blocks, header_merges, terminal_links, owner, counter);
    for completed_owner in completed {
        if blocks
            .iter()
            .find(|block| block.name == completed_owner)
            .is_some_and(is_switch_block)
        {
            finalize_fully_terminal_switch(blocks, header_merges, &completed_owner, counter);
        }
        materialize_pure_enclosing_selection_routes_for_owner(
            blocks,
            loop_merges,
            header_merges,
            pure_enclosing_owners,
            &completed_owner,
            counter,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn unique_selection_merges_with_loop_exit_and_forced_inner(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
    break_aware: bool,
    loop_exit_selection: bool,
    forced_terminal_merges: &HashMap<String, String>,
    construct_tree_owned: bool,
) -> (
    Vec<BodyBlock>,
    HashMap<(String, String), String>,
    HashMap<String, String>,
    HashMap<String, String>,
) {
    let forest = analyze(blocks);
    let terminal_parent_links =
        construct_tree_owned.then(|| terminal_parent_links(blocks, &forest));
    // Break-aware selection merges (the reject-triggered 5th `structured_plan` attempt) reconverge a
    // guarded-break selection at its non-break arm instead of the loop merge, so it never claims the
    // loop merge and the do-while latch is never redirected (the `merge-inloop` fix at its source).
    let mut sel = if break_aware {
        break_aware_selection_merges(blocks, &forest, loop_merges)
    } else {
        selection_merges(blocks, &forest)
    };
    if loop_exit_selection {
        refine_loop_exit_selection_merges(blocks, &forest, loop_merges, &mut sel);
        refine_loop_entry_terminal_selection_merges(blocks, &forest, loop_merges, &mut sel);
        refine_nested_terminal_selection_merges(
            blocks,
            &forest,
            loop_merges,
            forced_terminal_merges,
            &mut sel,
        );
    } else if construct_tree_owned {
        let source_headers = sel.keys().cloned().collect::<HashSet<_>>();
        refine_nested_terminal_selection_merges(
            blocks,
            &forest,
            loop_merges,
            forced_terminal_merges,
            &mut sel,
        );
        sel.retain(|header, _| {
            source_headers.contains(header)
                || !fully_terminal_void_return_selection(blocks, &forest, header)
        });
    }
    let loop_headers: HashSet<&str> = forest.loops.iter().map(|l| l.header.as_str()).collect();
    let loop_roles = loop_role_targets_with_passthroughs(blocks, loop_merges);
    // How many selection headers claim each natural merge (>1 ⇒ shared ⇒ collision).
    let mut claims: HashMap<&str, usize> = HashMap::new();
    for m in sel.values() {
        *claims.entry(m.as_str()).or_default() += 1;
    }
    // This complete source ownership map is immutable while headers are materialized. Later
    // innermost-first edge splits may rename the concrete merge blocks, but cannot change which
    // enclosing source selection owns an escaping continuation.
    let mut source_selection_merges = sel.clone();
    source_selection_merges.extend(forced_terminal_merges.clone());
    let pure_enclosing_owners = construct_tree_owned
        .then(|| pure_enclosing_selection_owners(blocks, &forest, &source_selection_merges));

    let mut out = blocks.to_vec();
    let mut branch = HashMap::new();
    let mut switch = HashMap::new();
    // Keep assignments by header until every enclosing synth has finished. An outer pass-through may
    // redirect an already-processed inner header's arm, so a `(true, false)` key recorded eagerly can
    // become stale even though that header's declared merge is still correct.
    let mut header_merges: HashMap<String, String> = HashMap::new();
    let mut counter = if construct_tree_owned {
        next_selection_merge_suffix(&out)
    } else {
        0usize
    };

    // Pre-register EVERY structured break/continue latch — not only those with a post-idom selection
    // merge. After multi-exit + do-while rotation a latch has arms `{loop-merge, continue}` and no
    // in-function reconvergence (the arms diverge), so `selection_merges` omits it; without this pass
    // the completeness gate later rejects `branch-no-merge` on that latch (M2 residual / residual
    // `merge-inloop` after converge). Record the key from the CURRENT terminator before any selection
    // synth can rewrite it.
    let mut break_continue_blocks: HashSet<String> = HashSet::new();
    for b in blocks {
        if loop_headers.contains(b.name.as_str()) {
            continue;
        }
        let Some((t, f)) = conditional_branch_targets(b) else {
            continue;
        };
        if let Some(cont) = loop_break_continue_merge(&forest, loop_merges, &b.name, &t, &f) {
            header_merges.insert(b.name.clone(), cont.clone());
            if !loop_exit_selection {
                branch.insert((t, f), cont);
            }
            break_continue_blocks.insert(b.name.clone());
        }
    }

    let depth = |name: &str| {
        let mut d = 0usize;
        let mut cur = name;
        while let Some(p) = forest.idom(cur) {
            d += 1;
            cur = p;
        }
        d
    };

    // A construct-tree selection whose two arms are already proven terminal owns an unreachable
    // merge declaration. Assign it with the other headers instead of discovering the omission in a
    // post-synthesis completion sweep. The proof is graph-structural and the synthesized block is
    // disconnected, so it neither redirects an edge nor invalidates the source forest used below.
    if construct_tree_owned {
        let loop_nodes = forest
            .loops
            .iter()
            .flat_map(|loop_info| loop_info.body.iter().cloned())
            .collect::<HashSet<_>>();
        let mut terminal_headers = blocks
            .iter()
            .filter(|block| {
                !header_merges.contains_key(&block.name)
                    && !sel.contains_key(&block.name)
                    && !forced_terminal_merges.contains_key(&block.name)
                    && !loop_nodes.contains(&block.name)
                    && conditional_branch_targets(block).is_some()
            })
            .map(|block| (block.name.clone(), depth(&block.name)))
            .collect::<Vec<_>>();
        // Terminal declarations do not redirect edges. Preserve the established outermost-first
        // ownership order so nested terminal headers remain independent and deterministic.
        terminal_headers.sort_by(|left, right| left.1.cmp(&right.1).then(left.0.cmp(&right.0)));
        for (header, _) in terminal_headers {
            if let Some(merge) =
                synth_shared_void_return_selection_merge(&mut out, &forest, &header, &mut counter)
            {
                header_merges.insert(header.clone(), merge);
                finish_construct_tree_selection_owner(
                    &mut out,
                    loop_merges,
                    &mut header_merges,
                    terminal_parent_links
                        .as_ref()
                        .expect("construct-tree terminal links were indexed"),
                    pure_enclosing_owners
                        .as_ref()
                        .expect("construct-tree enclosing owners were indexed"),
                    &header,
                    &mut counter,
                );
            }
        }
    }

    // Process headers INNERMOST-FIRST (deepest dominator depth first): a nested construct must claim
    // its own arms before its enclosing construct does, else the outer header's region-predecessor
    // sweep grabs the inner arms.
    let mut headers: Vec<&BodyBlock> = blocks
        .iter()
        .filter(|b| {
            !loop_headers.contains(b.name.as_str())
                && (sel.contains_key(&b.name) || forced_terminal_merges.contains_key(&b.name))
                && !break_continue_blocks.contains(&b.name)
        })
        .collect();
    // Dominator depth can be proportional to the function size in generated ladder CFGs. Cache
    // each header's depth once instead of walking the idom chain again for every sort comparison.
    headers.sort_by_cached_key(|b| std::cmp::Reverse(depth(&b.name)));

    // Non-construct retries need dominance over the CURRENT blocks, including pass-through merges
    // synthesized for already-processed (inner) headers. Construct-tree ownership cannot afford to
    // re-analyze the 1k+ block regional graph after every merge split; it derives dominance from the
    // immutable source forest and carries synthesized predecessor origins through `ct_owned_preds`
    // instead.
    let mut cur_forest = analyze(&out);
    #[derive(Clone, Debug)]
    struct CtOwnedPred {
        block: String,
        origins: Vec<String>,
    }
    let mut ct_owned_preds: HashMap<(String, String), Vec<CtOwnedPred>> = HashMap::new();
    let mut construct_tree_bare_loop_role_headers = HashSet::new();
    let mut stopped_for_growth = false;
    for b in headers.iter().copied() {
        if let Some(mut merge) = forced_terminal_merges.get(&b.name).cloned() {
            header_merges.insert(b.name.clone(), merge.clone());
            if construct_tree_owned {
                finish_construct_tree_selection_owner(
                    &mut out,
                    loop_merges,
                    &mut header_merges,
                    terminal_parent_links
                        .as_ref()
                        .expect("construct-tree terminal links were indexed"),
                    pure_enclosing_owners
                        .as_ref()
                        .expect("construct-tree enclosing owners were indexed"),
                    &b.name,
                    &mut counter,
                );
                merge = header_merges[&b.name].clone();
            }
            if !loop_exit_selection {
                let cur_block = out.iter().find(|x| x.name == b.name).unwrap_or(b);
                if let Some((true_target, false_target)) = conditional_branch_targets(cur_block) {
                    branch.insert((true_target, false_target), merge);
                }
            }
            continue;
        }
        let Some(natural) = sel.get(&b.name).cloned() else {
            continue;
        };
        if let Some((true_target, false_target)) = conditional_branch_targets(b) {
            if construct_tree_owned
                && bare_loop_exit_branch(&forest, loop_merges, &b.name, &true_target, &false_target)
            {
                construct_tree_bare_loop_role_headers.insert(b.name.clone());
                continue;
            }
            let enclosing_boundary = if construct_tree_owned {
                enclosing_selection_region_exit_target(
                    blocks,
                    &forest,
                    loop_merges,
                    &source_selection_merges,
                    &b.name,
                    &true_target,
                    &false_target,
                    Some(&natural),
                )
            } else {
                ordinary_selection_enclosing_boundary_target(
                    blocks,
                    &forest,
                    loop_merges,
                    &source_selection_merges,
                    &b.name,
                    &natural,
                )
            };
            if let Some(exit_target) = enclosing_boundary {
                let synthesis_forest = if construct_tree_owned {
                    &forest
                } else {
                    &cur_forest
                };
                // Which edges the split below will carry, read before it moves them. Only the
                // `!construct_tree_owned` arm records into `cur_forest`; the construct-tree arm
                // deliberately keeps deciding ownership from the immutable source forest.
                let carried = if construct_tree_owned {
                    Vec::new()
                } else {
                    header_owned_merge_predecessors(&out, synthesis_forest, &b.name, &exit_target)
                };
                let synth = if block_has_phi(&out, &exit_target) {
                    synth_unique_selection_merge_phi(
                        &mut out,
                        synthesis_forest,
                        &b.name,
                        &exit_target,
                        &mut counter,
                    )
                } else {
                    synth_unique_selection_merge(
                        &mut out,
                        synthesis_forest,
                        &b.name,
                        &exit_target,
                        &mut counter,
                    )
                };
                if let Some(merge) = synth {
                    header_merges.insert(b.name.clone(), merge.clone());
                    if construct_tree_owned {
                        finish_construct_tree_selection_owner(
                            &mut out,
                            loop_merges,
                            &mut header_merges,
                            terminal_parent_links
                                .as_ref()
                                .expect("construct-tree terminal links were indexed"),
                            pure_enclosing_owners
                                .as_ref()
                                .expect("construct-tree enclosing owners were indexed"),
                            &b.name,
                            &mut counter,
                        );
                    } else {
                        cur_forest.record_pass_through(&merge, &carried);
                        if !loop_exit_selection {
                            let current =
                                out.iter().find(|block| block.name == b.name).unwrap_or(b);
                            if let Some((true_target, false_target)) =
                                conditional_branch_targets(current)
                            {
                                branch.insert((true_target, false_target), merge);
                            }
                        }
                    }
                    continue;
                }
            }
        }
        // A shared merge is a collision when >1 selection claims it OR it doubles as a loop
        // merge/continue — the `claims`/`loop_roles` tests. But the `claims` counter only sees
        // OTHER SELECTION headers' post-idoms; when the block this header reconverges on is ALSO
        // branched to directly by an ENCLOSING construct's arm, no selection collision is recorded,
        // yet the header does not dominate it (an outer edge reaches it without passing the header) —
        // the `selection:merge-not-dominated` reject. That is the SAME shared-merge shape, so treat a
        // non-dominated natural merge as a collision and let the pass-through synth insert a
        // header-dominated merge (its predecessors are all header-dominated). `plan_self_check_reason`
        // re-checks dominance afterward, so if synth cannot resolve it the plan still honestly rejects.
        let natural_has_phi = block_has_phi(&out, &natural);
        let natural_is_synthetic_merge = construct_tree_owned
            && !natural_has_phi
            && out
                .iter()
                .find(|block| block.name == natural)
                .is_some_and(|block| block.role == BlockRole::LMerge);
        let dominance_forest = if construct_tree_owned {
            &forest
        } else {
            &cur_forest
        };
        let collides = claims.get(natural.as_str()).copied().unwrap_or(0) > 1
            || loop_roles.contains(&natural)
            || natural_is_synthetic_merge
            || !dominance_forest.dominates(&b.name, &natural);
        let mut merge = if collides {
            // A shared merge carrying a phi needs the phi-merge surgery (merged phi in the synthesized
            // pass-through, `natural`'s phi rebuilt); a no-phi shared merge takes the plain split.
            // NOTE: do NOT skip structural loop-exit predecessors here — that is the disproven
            // LOOP_BREAK_PROTECT shape (dead-end #19): leaving a latch pointing at a loop merge an
            // enclosing selection claimed over-admits into "branches to the selection construct, but
            // not to the selection header". Break-aware selection merges fix the redirect at its
            // source by not claiming the loop merge; multi-exit shared-exit trampolines fix the
            // residual cross-arm on the dispatch itself.
            let source_owned_preds = if construct_tree_owned {
                out.iter()
                    .filter(|candidate| {
                        construct_tree_redirect_candidate(candidate)
                            && forest.dominates(&b.name, &candidate.name)
                            && block_successors(candidate)
                                .iter()
                                .any(|target| target == &natural)
                    })
                    .map(|candidate| candidate.name.clone())
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            // Frontier straddle residual after regional wrap: the only edges into `natural` may be
            // ConstructTreeRoute gateways. Reclaim the Normal/LMerge predecessors of those routes so
            // multi-header shared merges still get private splits (exact-eight routes stay unclaimed).
            let route_reclaimed_preds = if construct_tree_owned && source_owned_preds.is_empty() {
                construct_tree_preds_reclaiming_routes(&out, &forest, &b.name, &natural)
            } else {
                Vec::new()
            };
            let carried_preds = if construct_tree_owned {
                ct_owned_preds
                    .remove(&(b.name.clone(), natural.clone()))
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            let current_owned_preds = Vec::new();
            let mut explicit_no_phi_preds = source_owned_preds.clone();
            explicit_no_phi_preds.extend(route_reclaimed_preds.iter().cloned());
            explicit_no_phi_preds.extend(current_owned_preds);
            explicit_no_phi_preds.extend(carried_preds.iter().map(|pred| pred.block.clone()));
            explicit_no_phi_preds.sort();
            explicit_no_phi_preds.dedup();
            let mut propagated_origins = source_owned_preds.clone();
            propagated_origins.extend(route_reclaimed_preds.iter().cloned());
            for pred in &carried_preds {
                propagated_origins.extend(pred.origins.iter().cloned());
            }
            propagated_origins.sort();
            propagated_origins.dedup();
            let routes_into_natural = if construct_tree_owned && !route_reclaimed_preds.is_empty() {
                construct_tree_routes_into_natural(&out, &natural)
            } else {
                HashSet::new()
            };
            // As above: read before the split moves them, and only for the arms that record.
            let carried = if construct_tree_owned {
                Vec::new()
            } else {
                header_owned_merge_predecessors(&out, &cur_forest, &b.name, &natural)
            };
            let synth = if natural_has_phi {
                if construct_tree_owned && !explicit_no_phi_preds.is_empty() {
                    synth_unique_selection_merge_phi_explicit(
                        &mut out,
                        &explicit_no_phi_preds,
                        &natural,
                        &routes_into_natural,
                        &mut counter,
                    )
                } else if construct_tree_owned {
                    synth_unique_selection_merge_phi(
                        &mut out,
                        &forest,
                        &b.name,
                        &natural,
                        &mut counter,
                    )
                } else {
                    synth_unique_selection_merge_phi(
                        &mut out,
                        &cur_forest,
                        &b.name,
                        &natural,
                        &mut counter,
                    )
                }
            } else if construct_tree_owned {
                synth_unique_selection_merge_no_phi_explicit(
                    &mut out,
                    &explicit_no_phi_preds,
                    &natural,
                    &routes_into_natural,
                    &mut counter,
                )
            } else {
                let synth = synth_unique_selection_merge(
                    &mut out,
                    &cur_forest,
                    &b.name,
                    &natural,
                    &mut counter,
                );
                if synth.is_none() && construct_tree_owned {
                    synth_unique_selection_merge(&mut out, &forest, &b.name, &natural, &mut counter)
                } else {
                    synth
                }
            };
            match synth {
                Some(s) => {
                    if selection_synth_growth_exceeds_ladder_cap(blocks.len(), out.len()) {
                        stopped_for_growth = true;
                    } else if !construct_tree_owned {
                        // A unique-selection merge only splits forward edges with an acyclic
                        // pass-through. Natural loops and their nesting are unchanged, and so is
                        // dominance among the blocks that were already there -- so record the one
                        // block it added rather than re-deriving the relation. Re-deriving here made
                        // large generated CFGs revisit the whole graph once per split: it was 55% of
                        // every CFG rebuild the slowest corpus source performed.
                        cur_forest.record_pass_through(&s, &carried);
                    }
                    if construct_tree_owned && !propagated_origins.is_empty() {
                        for target in &headers {
                            if target.name == b.name {
                                continue;
                            }
                            if sel.get(&target.name) != Some(&natural) {
                                continue;
                            }
                            if propagated_origins
                                .iter()
                                .all(|origin| forest.dominates(&target.name, origin))
                            {
                                ct_owned_preds
                                    .entry((target.name.clone(), natural.clone()))
                                    .or_default()
                                    .push(CtOwnedPred {
                                        block: s.clone(),
                                        origins: propagated_origins.clone(),
                                    });
                            }
                        }
                    }
                    s
                }
                None => {
                    if crate::env_vars::spi_why() {
                        eprintln!(
                            "[spi-why]   selection-synth-decline header={} natural={} phi={} collides={}",
                            b.name, natural, natural_has_phi, collides,
                        );
                    }
                    continue;
                }
            }
        } else {
            natural
        };
        header_merges.insert(b.name.clone(), merge.clone());
        if construct_tree_owned {
            finish_construct_tree_selection_owner(
                &mut out,
                loop_merges,
                &mut header_merges,
                terminal_parent_links
                    .as_ref()
                    .expect("construct-tree terminal links were indexed"),
                pure_enclosing_owners
                    .as_ref()
                    .expect("construct-tree enclosing owners were indexed"),
                &b.name,
                &mut counter,
            );
            merge = header_merges[&b.name].clone();
        }
        if !loop_exit_selection {
            // Keep the ordinary path byte-identical: it historically keyed each merge as soon as the
            // header was processed. The final re-key below is only needed by the reject-only
            // loop-exit tier, where an outer synth intentionally rewrites nested loop-exit arms.
            let cur_block = out.iter().find(|x| x.name == b.name).unwrap_or(b);
            if is_switch_block(cur_block) {
                switch.insert(b.name.clone(), merge);
            } else if let Some((true_target, false_target)) = conditional_branch_targets(cur_block)
            {
                branch.insert((true_target, false_target), merge);
            }
        }
        if stopped_for_growth {
            if crate::env_vars::spi_why() {
                eprintln!(
                    "[spi-why]   selection-growth-stop source={} candidate={} last_header={}",
                    blocks.len(),
                    out.len(),
                    b.name,
                );
            }
            break;
        }
    }

    // Re-key from final terminators when a later transform can rewrite an already-recorded header. This
    // is intentionally after the full innermost-first synthesis pass: an enclosing unique-merge split
    // can redirect a nested header's arm after that nested header was assigned, and the emitter keys
    // conditional merges by the final target pair rather than the header name.
    let rekey_all =
        loop_exit_selection || !forced_terminal_merges.is_empty() || construct_tree_owned;
    if rekey_all || !break_continue_blocks.is_empty() {
        let final_forest = analyze(&out);
        let mut bare_loop_role_headers = Vec::new();
        for block in &out {
            if !rekey_all && !break_continue_blocks.contains(&block.name) {
                continue;
            }
            let Some(merge) = header_merges.get(&block.name).cloned() else {
                continue;
            };
            if crate::env_vars::spi_why() {
                let merge_block = out.iter().find(|candidate| candidate.name == merge);
                eprintln!(
                    "[spi-why]   final-selection header={} merge={} role={:?} successors={:?}",
                    block.name,
                    merge,
                    merge_block.map(|candidate| candidate.role),
                    merge_block.map(block_successors),
                );
            }
            if is_switch_block(block) {
                switch.insert(block.name.clone(), merge);
            } else if let Some((true_target, false_target)) = conditional_branch_targets(block) {
                let final_bare_loop_role = construct_tree_owned
                    && (construct_tree_bare_loop_role_headers.contains(&block.name)
                        || bare_natural_loop_exit_branch(
                            &final_forest,
                            &block.name,
                            &true_target,
                            &false_target,
                        )
                        || bare_loop_exit_branch_with_passthroughs(
                            &out,
                            &final_forest,
                            loop_merges,
                            &block.name,
                            &true_target,
                            &false_target,
                        ));
                if final_bare_loop_role {
                    // Late selection synthesis may wrap a loop merge/continue arm in a
                    // single-successor gateway after this header was correctly classified as a
                    // bare bottom-test branch. Re-keying it as a selection makes that gateway the
                    // declared merge even though it is reached only from the latch, so it cannot be
                    // dominated by the header. Preserve the loop-role branch as bare in the final
                    // maps as well as during the initial classification.
                    branch.remove(&(true_target.clone(), false_target.clone()));
                    branch.remove(&(false_target.clone(), true_target.clone()));
                    bare_loop_role_headers.push(block.name.clone());
                    continue;
                }
                branch.insert((true_target, false_target), merge);
            }
        }
        for header in bare_loop_role_headers {
            header_merges.remove(&header);
        }
    }
    (out, branch, header_merges, switch)
}

#[cfg(test)]
pub(in crate::native) fn repair_construct_tree_nondominated_selection_merges(
    blocks: &mut Vec<BodyBlock>,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    header_merges: &mut HashMap<String, String>,
    counter: &mut usize,
) {
    for _ in 0..8 {
        let forest = analyze(blocks);
        let mut merge_of = HashMap::new();
        for (header, info) in loop_merges {
            merge_of.insert(header.clone(), info.merge.clone());
        }
        merge_of.extend(
            header_merges
                .iter()
                .map(|(header, merge)| (header.clone(), merge.clone())),
        );
        let order = structured_order(blocks, &forest, |header| merge_of.get(header).cloned());
        let rank = order
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect::<HashMap<_, _>>();
        let mut headers = blocks
            .iter()
            .filter_map(|block| Some((block.name.clone(), header_merges.get(&block.name)?.clone())))
            .collect::<Vec<_>>();
        headers.sort_by_key(|(header, _)| rank.get(header.as_str()).copied().unwrap_or(usize::MAX));
        let mut repaired = false;
        for (header, merge) in headers {
            if terminal_exit_continuation(blocks, &forest, &header).is_some() {
                continue;
            }
            if forest.dominates(&header, &merge) {
                continue;
            }
            let Some(block) = blocks.iter().find(|block| block.name == header) else {
                continue;
            };
            let header_targets_merge = block_successors(block)
                .iter()
                .any(|target| target == &merge);
            let route_reclaim_available = !header_targets_merge
                && !construct_tree_preds_reclaiming_routes(blocks, &forest, &header, &merge)
                    .is_empty();
            // Ordinary polluted-merge repair needs a direct header edge into the merge. The
            // post-regional-wrapper residual has no such edge: reconvergence was rewritten into a
            // ConstructTreeRoute gateway, so reclaim Normal/LMerge predecessors of those routes.
            if !header_targets_merge && !route_reclaim_available {
                continue;
            }
            if let Some((t, f)) = conditional_branch_targets(block) {
                if bare_loop_exit_branch_with_passthroughs(
                    blocks,
                    &forest,
                    loop_merges,
                    &header,
                    &t,
                    &f,
                ) {
                    continue;
                }
            }
            if block_has_phi(blocks, &merge) {
                continue;
            }
            let mut repair_preds = blocks
                .iter()
                .filter(|candidate| {
                    construct_tree_redirect_candidate(candidate)
                        && forest.dominates(&header, &candidate.name)
                        && block_successors(candidate)
                            .iter()
                            .any(|target| target == &merge)
                })
                .map(|candidate| candidate.name.clone())
                .collect::<Vec<_>>();
            let routes_into_merge = if repair_preds.is_empty() {
                let reclaimed =
                    construct_tree_preds_reclaiming_routes(blocks, &forest, &header, &merge);
                repair_preds.extend(reclaimed);
                construct_tree_routes_into_natural(blocks, &merge)
            } else {
                HashSet::new()
            };
            repair_preds.sort();
            repair_preds.dedup();
            let split = synth_unique_selection_merge_no_phi_explicit(
                blocks,
                &repair_preds,
                &merge,
                &routes_into_merge,
                counter,
            );
            if let Some(private_merge) = split {
                header_merges.insert(header, private_merge);
                repaired = true;
                break;
            }
        }
        if !repaired {
            break;
        }
    }
}

#[cfg(test)]
pub(in crate::native) fn repair_construct_tree_passthrough_selection_merges(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
    header_merges: &mut HashMap<String, String>,
) {
    let forest = analyze(blocks);
    let by_name = blocks
        .iter()
        .map(|block| (block.name.as_str(), block))
        .collect::<HashMap<_, _>>();
    let loop_roles = loop_role_targets_with_passthroughs(blocks, loop_merges);
    let mut claims = header_merges
        .values()
        .fold(HashMap::new(), |mut claims, merge| {
            *claims.entry(merge.clone()).or_insert(0usize) += 1;
            claims
        });
    for _ in 0..blocks.len() {
        let mut changed = false;
        let assignments = header_merges
            .iter()
            .map(|(header, merge)| (header.clone(), merge.clone()))
            .collect::<Vec<_>>();
        for (header, merge) in assignments {
            if terminal_exit_continuation(blocks, &forest, &header).is_some() {
                continue;
            }
            let Some(merge_block) = by_name.get(merge.as_str()) else {
                continue;
            };
            if merge_block.role != BlockRole::LMerge {
                continue;
            }
            let successors = block_successors(merge_block);
            let [successor] = successors.as_slice() else {
                continue;
            };
            // Promoting a private pass-through back onto a block already owned by another
            // selection (or by a loop role) recreates the exact collision the private merge was
            // synthesized to avoid. Reserve targets as promotions happen so two private merges in
            // the same pass cannot both collapse onto an initially-unclaimed successor either.
            if loop_roles.contains(successor) || claims.get(successor).copied().unwrap_or(0) != 0 {
                continue;
            }
            if !forest.dominates(&header, successor) {
                continue;
            }
            let bypasses_merge = blocks.iter().any(|candidate| {
                candidate.name != merge
                    && forest.dominates(&header, &candidate.name)
                    && block_successors(candidate)
                        .iter()
                        .any(|target| target == successor)
            });
            if bypasses_merge {
                header_merges.insert(header, successor.clone());
                if let Some(count) = claims.get_mut(&merge) {
                    *count = count.saturating_sub(1);
                }
                *claims.entry(successor.clone()).or_insert(0) += 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

/// Replace a bypassed private pass-through with a phi-aware merge over every header-owned incoming
/// to its shared successor.
///
/// `H -> { old-private -> J, body -> J }` is not structured when outside paths also enter `J`: using
/// `old-private` as H's merge lets `body` bypass it, while promoting H directly to `J` would give the
/// construct an external entry. Split the H-owned incoming edges to `J` through one fresh private
/// merge. [`synth_unique_selection_merge_phi`] preserves `J`'s exact incoming values when it carries
/// phis; the no-phi variant performs the same ownership split without data surgery.
#[cfg(test)]
pub(in crate::native) fn repair_construct_tree_bypassed_passthrough_merges(
    blocks: &mut Vec<BodyBlock>,
    header_merges: &mut HashMap<String, String>,
    counter: &mut usize,
) -> bool {
    let mut changed = false;
    loop {
        let forest = analyze(blocks);
        let by_name = blocks
            .iter()
            .map(|block| (block.name.as_str(), block))
            .collect::<HashMap<_, _>>();
        let mut assignments = header_merges
            .iter()
            .map(|(header, merge)| (header.clone(), merge.clone()))
            .collect::<Vec<_>>();
        assignments.sort();
        let mut repair = None;
        for (header, merge) in assignments {
            // A direct terminal guard owns the pass-through on its live arm. Chasing its merge chain
            // to the first non-LMerge block can cross onto the returning arm and replace that local
            // contract with a terminal target. Leave it with the terminal constructor.
            if terminal_exit_continuation(blocks, &forest, &header).is_some() {
                continue;
            }
            let Some(merge_block) = by_name.get(merge.as_str()) else {
                continue;
            };
            if merge_block.role != BlockRole::LMerge {
                continue;
            }
            let mut chain = HashSet::new();
            let mut current = merge.clone();
            let successor = loop {
                if !chain.insert(current.clone()) {
                    break None;
                }
                let Some(block) = by_name.get(current.as_str()) else {
                    break None;
                };
                if block.role != BlockRole::LMerge {
                    break Some(current);
                }
                let successors = block_successors(block);
                let [next] = successors.as_slice() else {
                    break None;
                };
                current = next.clone();
            };
            let Some(successor) = successor else {
                continue;
            };
            let bypass = blocks.iter().any(|candidate| {
                !chain.contains(&candidate.name)
                    && forest.dominates(&header, &candidate.name)
                    && block_successors(candidate)
                        .iter()
                        .any(|target| target == &successor)
            });
            if bypass {
                repair = Some((header, successor));
                break;
            }
        }
        let Some((header, successor)) = repair else {
            break;
        };
        let private = if block_has_phi(blocks, &successor) {
            synth_unique_selection_merge_phi(blocks, &forest, &header, &successor, counter)
        } else {
            synth_unique_selection_merge(blocks, &forest, &header, &successor, counter)
        };
        let Some(private) = private else {
            break;
        };
        if crate::env_vars::spi_why() {
            eprintln!(
                "[spi-why]   bypass-refunnel header={} successor={} merge={}",
                header, successor, private,
            );
        }
        header_merges.insert(header, private);
        changed = true;
    }
    changed
}

fn synth_unique_selection_merge_no_phi_explicit(
    blocks: &mut Vec<BodyBlock>,
    preds: &[String],
    natural: &str,
    routes_into_natural: &HashSet<String>,
    counter: &mut usize,
) -> Option<String> {
    let pred_set: HashSet<&str> = preds.iter().map(String::as_str).collect();
    // (pred name, old successor to rewrite → private merge)
    let mut redirect: Vec<(String, String)> = Vec::new();
    for block in blocks.iter() {
        if !pred_set.contains(block.name.as_str()) {
            continue;
        }
        for target in block_successors(block) {
            if target == natural || routes_into_natural.contains(&target) {
                redirect.push((block.name.clone(), target));
            }
        }
    }
    if redirect.is_empty() {
        return None;
    }
    let new_name = format!("{SPLIT_PREFIX}{SEL_TOKEN}{counter}");
    *counter += 1;
    for (pred, old_target) in &redirect {
        if let Some(block) = blocks.iter_mut().find(|block| &block.name == pred) {
            if let Some(typed) = block.typed_mut() {
                typed.redirect_successor(old_target, &new_name);
            }
        }
    }
    let at = blocks
        .iter()
        .position(|block| block.name == natural)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn block(name: &str) -> BodyBlock {
        BodyBlock {
            name: name.to_string(),
            role: role_for_name(name),
            typed: crate::native::tir::lower_block_carrier(
                name,
                &["ret void".to_string()],
                &HashMap::new(),
            )
            .map(Into::into),
        }
    }

    #[test]
    fn construct_tree_selection_counter_skips_existing_sel_suffixes() {
        let blocks = vec![
            block("%entry"),
            block("%metal2vulkan.lmerge.sel0"),
            block("%metal2vulkan.lmerge.sel12"),
            block("%metal2vulkan.lmerge.selx"),
            block("%metal2vulkan.lmerge.99"),
        ];
        assert_eq!(next_selection_merge_suffix(&blocks), 13);
    }
}
