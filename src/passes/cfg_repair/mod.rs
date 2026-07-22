//! Structured-CFG merge, continue, and phi repair over the retained SPIR-V module.

use super::*;
use crate::passes::spirv_cfg::{block_successors, label_dominates};

/// Repair structured-CFG merge placement: `OpSelectionMerge`/`OpLoopMerge` must be the
/// second-to-last instruction in its block (immediately before the block's branch terminator). llc's
/// SPIR-V structurizer occasionally hoists a value computation (e.g. an `OpFNegate` for a select arm)
/// BETWEEN the merge and the conditional branch, leaving the merge mid-block (spirv-val rejects it).
/// The merge declaration has no result and only references label ids, so sliding it down to sit just
/// before the terminator is always semantics-preserving. We move it; the displaced instructions keep
/// their relative order and stay above it.
pub(in crate::passes) fn fix_merge_placement(ctx: &mut Ctx, entry_idx: usize) {
    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let insts = &mut ctx.module.functions[entry_idx].blocks[bi].instructions;
        let n = insts.len();
        if n < 2 {
            continue;
        }
        // Find a merge instruction that is NOT already at index n-2.
        let merge_pos = insts
            .iter()
            .position(|i| matches!(i.class.opcode, Op::SelectionMerge | Op::LoopMerge));
        if let Some(pos) = merge_pos {
            if pos != n - 2 {
                let merge = insts.remove(pos);
                // After removal the block has n-1 instructions; insert so the merge becomes the
                // second-to-last (just before the terminator at the new end).
                insts.insert(insts.len() - 1, merge);
            }
        }
    }
}

pub(in crate::passes) fn repair_continue_selection_merge_targets(ctx: &mut Ctx, entry_idx: usize) {
    loop {
        if !repair_one_continue_selection_merge_target(ctx, entry_idx) {
            break;
        }
    }
}

pub(in crate::passes) fn repair_loop_continue_external_predecessors(
    ctx: &mut Ctx,
    entry_idx: usize,
) {
    loop {
        if !repair_one_loop_continue_external_predecessor(ctx, entry_idx) {
            break;
        }
    }
}

pub(in crate::passes) fn repair_loop_continue_pass_through_targets(
    ctx: &mut Ctx,
    entry_idx: usize,
) {
    loop {
        if !repair_one_loop_continue_pass_through_target(ctx, entry_idx) {
            break;
        }
    }
}

pub(in crate::passes) fn repair_one_loop_continue_pass_through_target(
    ctx: &mut Ctx,
    entry_idx: usize,
) -> bool {
    let blocks = &mut ctx.module.functions[entry_idx].blocks;
    for header_idx in 0..blocks.len() {
        let Some(header_label) = blocks[header_idx]
            .label
            .as_ref()
            .and_then(|label| label.result_id)
        else {
            continue;
        };
        let Some(loop_merge_idx) = blocks[header_idx]
            .instructions
            .iter()
            .position(|inst| inst.class.opcode == Op::LoopMerge)
        else {
            continue;
        };
        let Some(loop_merge_label) = blocks[header_idx].instructions[loop_merge_idx]
            .operands
            .first()
            .and_then(id_ref_operand)
        else {
            continue;
        };
        let Some(continue_label) = blocks[header_idx].instructions[loop_merge_idx]
            .operands
            .get(1)
            .and_then(id_ref_operand)
        else {
            continue;
        };
        let Some(continue_idx) = block_index_by_label(blocks, continue_label) else {
            continue;
        };
        let Some(target) = branch_only_target(&blocks[continue_idx]) else {
            continue;
        };
        if target == header_label || target == loop_merge_label {
            continue;
        }
        if let Some(Operand::IdRef(continue_target)) = blocks[header_idx].instructions
            [loop_merge_idx]
            .operands
            .get_mut(1)
        {
            *continue_target = target;
            return true;
        }
    }
    false
}

pub(in crate::passes) fn repair_one_loop_continue_external_predecessor(
    ctx: &mut Ctx,
    entry_idx: usize,
) -> bool {
    let blocks = &ctx.module.functions[entry_idx].blocks;
    let predecessors = block_predecessors(blocks);
    let dominators = block_dominators(blocks);

    for header_idx in 0..blocks.len() {
        let Some(header_label) = blocks[header_idx]
            .label
            .as_ref()
            .and_then(|label| label.result_id)
        else {
            continue;
        };
        let Some(loop_merge) = blocks[header_idx]
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == Op::LoopMerge)
        else {
            continue;
        };
        let Some(continue_label) = loop_merge.operands.get(1).and_then(id_ref_operand) else {
            continue;
        };
        if continue_label == header_label {
            continue;
        }
        let Some(continue_idx) = block_index_by_label(blocks, continue_label) else {
            continue;
        };
        if phi_prefix_branch_target(&blocks[continue_idx]) != Some(header_label) {
            continue;
        }
        let Some(continue_preds) = predecessors.get(&continue_label).cloned() else {
            continue;
        };
        let outside_preds = continue_preds
            .iter()
            .copied()
            .filter(|pred| !label_dominates(&dominators, header_label, *pred))
            .collect::<HashSet<_>>();
        if outside_preds.is_empty() {
            continue;
        }

        let mut continue_phi_updates = Vec::new();
        let mut continue_phi_splits: HashMap<Word, PhiSplit> = HashMap::new();
        for (inst_idx, inst) in blocks[continue_idx].instructions.iter().enumerate() {
            if inst.class.opcode != Op::Phi {
                break;
            }
            let Some(result) = inst.result_id else {
                continue;
            };
            let (inside, outside) = split_phi_operands(&inst.operands, &outside_preds);
            if outside.is_empty() {
                continue;
            }
            if inside.is_empty() {
                continue_phi_updates.clear();
                continue_phi_splits.clear();
                break;
            }
            continue_phi_updates.push((inst_idx, inside.clone()));
            continue_phi_splits.insert(result, PhiSplit { inside, outside });
        }
        if !outside_preds.is_empty()
            && blocks[header_idx]
                .instructions
                .iter()
                .take_while(|inst| inst.class.opcode == Op::Phi)
                .any(|inst| {
                    !phi_predicates_after_continue_split(
                        inst,
                        continue_label,
                        &outside_preds,
                        &continue_phi_splits,
                    )
                })
        {
            continue;
        }

        let mut header_phi_updates = Vec::new();
        for (inst_idx, inst) in blocks[header_idx].instructions.iter().enumerate() {
            if inst.class.opcode != Op::Phi {
                break;
            }
            let Some(operands) =
                split_header_phi_continue_incoming(inst, continue_label, &continue_phi_splits)
            else {
                continue;
            };
            header_phi_updates.push((inst_idx, operands));
        }

        let blocks = &mut ctx.module.functions[entry_idx].blocks;
        for (inst_idx, operands) in continue_phi_updates {
            if let Some(inst) = blocks[continue_idx].instructions.get_mut(inst_idx) {
                inst.operands = operands;
            }
        }
        for (inst_idx, operands) in header_phi_updates {
            if let Some(inst) = blocks[header_idx].instructions.get_mut(inst_idx) {
                inst.operands = operands;
            }
        }
        for block in blocks.iter_mut() {
            let Some(label) = block.label.as_ref().and_then(|label| label.result_id) else {
                continue;
            };
            if outside_preds.contains(&label) {
                redirect_terminator_target(block, continue_label, header_label);
            }
        }
        return true;
    }
    false
}

pub(in crate::passes) fn repair_one_continue_selection_merge_target(
    ctx: &mut Ctx,
    entry_idx: usize,
) -> bool {
    let block_count = ctx.module.functions[entry_idx].blocks.len();
    for header_idx in 0..block_count {
        let Some(loop_merge) = ctx.module.functions[entry_idx].blocks[header_idx]
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == Op::LoopMerge)
        else {
            continue;
        };
        let Some(loop_merge_label) = loop_merge.operands.first().and_then(id_ref_operand) else {
            continue;
        };
        let Some(continue_label) = loop_merge.operands.get(1).and_then(id_ref_operand) else {
            continue;
        };

        for block_idx in 0..block_count {
            let Some(block_label) = ctx.module.functions[entry_idx].blocks[block_idx]
                .label
                .as_ref()
                .and_then(|label| label.result_id)
            else {
                continue;
            };
            let Some(selection_merge_idx) = ctx.module.functions[entry_idx].blocks[block_idx]
                .instructions
                .iter()
                .position(|inst| inst.class.opcode == Op::SelectionMerge)
            else {
                continue;
            };
            let selection_merge = ctx.module.functions[entry_idx].blocks[block_idx].instructions
                [selection_merge_idx]
                .operands
                .first()
                .and_then(id_ref_operand);
            if selection_merge != Some(continue_label) {
                continue;
            }
            let Some(term) = ctx.module.functions[entry_idx].blocks[block_idx]
                .instructions
                .last()
            else {
                continue;
            };
            if term.class.opcode != Op::BranchConditional {
                continue;
            }
            let Some(true_target) = term.operands.get(1).and_then(id_ref_operand) else {
                continue;
            };
            let Some(false_target) = term.operands.get(2).and_then(id_ref_operand) else {
                continue;
            };
            let branches_to_continue_and_merge = (true_target == continue_label
                && false_target == loop_merge_label)
                || (false_target == continue_label && true_target == loop_merge_label);
            if !branches_to_continue_and_merge {
                if all_paths_reach_target_without_labels(
                    &ctx.module.functions[entry_idx].blocks,
                    true_target,
                    continue_label,
                    &[loop_merge_label],
                ) && all_paths_reach_target_without_labels(
                    &ctx.module.functions[entry_idx].blocks,
                    false_target,
                    continue_label,
                    &[loop_merge_label],
                ) {
                    return split_selection_continue_merge(
                        ctx,
                        entry_idx,
                        block_idx,
                        selection_merge_idx,
                        continue_label,
                    );
                }
                continue;
            }
            let Some(loop_merge_idx) =
                block_index_by_label(&ctx.module.functions[entry_idx].blocks, loop_merge_label)
            else {
                continue;
            };

            let synthetic_label = ctx.module.fresh_id();
            let blocks = &mut ctx.module.functions[entry_idx].blocks;
            if let Some(Operand::IdRef(label)) = blocks[block_idx].instructions[selection_merge_idx]
                .operands
                .first_mut()
            {
                *label = synthetic_label;
            }
            redirect_terminator_target(&mut blocks[block_idx], loop_merge_label, synthetic_label);
            replace_phi_predecessor(blocks, loop_merge_label, block_label, synthetic_label);
            blocks.insert(
                loop_merge_idx,
                Block {
                    label: Some(Instruction::new(
                        Op::Label,
                        None,
                        Some(synthetic_label),
                        vec![],
                    )),
                    instructions: vec![Instruction::new(
                        Op::Branch,
                        None,
                        None,
                        vec![Operand::IdRef(loop_merge_label)],
                    )],
                },
            );
            return true;
        }
    }
    false
}

#[derive(Clone)]
pub(in crate::passes) struct PhiSplit {
    pub(in crate::passes) inside: Vec<Operand>,
    pub(in crate::passes) outside: Vec<Operand>,
}

pub(in crate::passes) fn split_phi_operands(
    operands: &[Operand],
    outside_preds: &HashSet<Word>,
) -> (Vec<Operand>, Vec<Operand>) {
    let mut inside = Vec::new();
    let mut outside = Vec::new();
    for pair in operands.chunks(2) {
        let [value, Operand::IdRef(pred)] = pair else {
            inside.extend_from_slice(pair);
            continue;
        };
        if outside_preds.contains(pred) {
            outside.push(value.clone());
            outside.push(Operand::IdRef(*pred));
        } else {
            inside.push(value.clone());
            inside.push(Operand::IdRef(*pred));
        }
    }
    (inside, outside)
}

pub(in crate::passes) fn phi_predicates_after_continue_split(
    inst: &Instruction,
    continue_label: Word,
    outside_preds: &HashSet<Word>,
    continue_phi_splits: &HashMap<Word, PhiSplit>,
) -> bool {
    let Some(operands) =
        split_header_phi_continue_incoming(inst, continue_label, continue_phi_splits)
    else {
        return false;
    };
    let preds = operands
        .chunks(2)
        .filter_map(|pair| match pair {
            [_, Operand::IdRef(pred)] => Some(*pred),
            _ => None,
        })
        .collect::<HashSet<_>>();
    outside_preds.iter().all(|pred| preds.contains(pred))
}

pub(in crate::passes) fn split_header_phi_continue_incoming(
    inst: &Instruction,
    continue_label: Word,
    continue_phi_splits: &HashMap<Word, PhiSplit>,
) -> Option<Vec<Operand>> {
    let mut changed = false;
    let mut operands = Vec::new();
    for pair in inst.operands.chunks(2) {
        let [value, Operand::IdRef(pred)] = pair else {
            operands.extend_from_slice(pair);
            continue;
        };
        if *pred != continue_label {
            operands.push(value.clone());
            operands.push(Operand::IdRef(*pred));
            continue;
        }
        let Operand::IdRef(value_id) = value else {
            operands.push(value.clone());
            operands.push(Operand::IdRef(*pred));
            continue;
        };
        let Some(split) = continue_phi_splits.get(value_id) else {
            operands.push(value.clone());
            operands.push(Operand::IdRef(*pred));
            continue;
        };
        changed = true;
        if !split.inside.is_empty() {
            operands.push(value.clone());
            operands.push(Operand::IdRef(*pred));
        }
        operands.extend(split.outside.clone());
    }
    changed.then_some(operands)
}

pub(in crate::passes) fn split_selection_continue_merge(
    ctx: &mut Ctx,
    entry_idx: usize,
    header_idx: usize,
    selection_merge_idx: usize,
    continue_label: Word,
) -> bool {
    let Some(header_label) = ctx.module.functions[entry_idx].blocks[header_idx]
        .label
        .as_ref()
        .and_then(|label| label.result_id)
    else {
        return false;
    };
    let Some(continue_idx) =
        block_index_by_label(&ctx.module.functions[entry_idx].blocks, continue_label)
    else {
        return false;
    };
    let construct_labels = reachable_before_target(
        &ctx.module.functions[entry_idx].blocks,
        header_label,
        continue_label,
    );
    let synthetic_label = ctx.module.fresh_id();

    let mut redirected_preds = HashSet::new();
    {
        let blocks = &mut ctx.module.functions[entry_idx].blocks;
        if let Some(Operand::IdRef(label)) = blocks[header_idx].instructions[selection_merge_idx]
            .operands
            .first_mut()
        {
            *label = synthetic_label;
        }
        for block in blocks.iter_mut() {
            let Some(pred) = block.label.as_ref().and_then(|label| label.result_id) else {
                continue;
            };
            if !construct_labels.contains(&pred) {
                continue;
            }
            if redirect_terminator_target(block, continue_label, synthetic_label) {
                redirected_preds.insert(pred);
            }
        }
    }
    if redirected_preds.is_empty() {
        return false;
    }

    let phi_splits = {
        let blocks = &ctx.module.functions[entry_idx].blocks;
        let Some(continue_block) = blocks.get(continue_idx) else {
            return false;
        };
        let mut splits = Vec::new();
        for (inst_idx, inst) in continue_block.instructions.iter().enumerate() {
            if inst.class.opcode != Op::Phi {
                break;
            }
            let mut kept = Vec::new();
            let mut redirected = Vec::new();
            for pair in inst.operands.chunks(2) {
                if pair.len() != 2 {
                    kept.extend_from_slice(pair);
                    continue;
                }
                let is_redirected =
                    matches!(pair[1], Operand::IdRef(pred) if redirected_preds.contains(&pred));
                if is_redirected {
                    redirected.extend_from_slice(pair);
                } else {
                    kept.extend_from_slice(pair);
                }
            }
            if !redirected.is_empty() {
                splits.push((inst_idx, inst.result_type, kept, redirected));
            }
        }
        splits
    };

    let mut synthetic_instructions = Vec::new();
    let mut phi_updates = Vec::new();
    for (inst_idx, result_type, mut kept, redirected) in phi_splits {
        let phi_id = ctx.module.fresh_id();
        synthetic_instructions.push(Instruction::new(
            Op::Phi,
            result_type,
            Some(phi_id),
            redirected,
        ));
        kept.push(Operand::IdRef(phi_id));
        kept.push(Operand::IdRef(synthetic_label));
        phi_updates.push((inst_idx, kept));
    }
    synthetic_instructions.push(Instruction::new(
        Op::Branch,
        None,
        None,
        vec![Operand::IdRef(continue_label)],
    ));

    let blocks = &mut ctx.module.functions[entry_idx].blocks;
    if let Some(continue_block) = blocks.get_mut(continue_idx) {
        for (inst_idx, operands) in phi_updates {
            if let Some(inst) = continue_block.instructions.get_mut(inst_idx) {
                inst.operands = operands;
            }
        }
    }
    blocks.insert(
        continue_idx,
        Block {
            label: Some(Instruction::new(
                Op::Label,
                None,
                Some(synthetic_label),
                vec![],
            )),
            instructions: synthetic_instructions,
        },
    );
    true
}

pub(in crate::passes) fn repair_phi_predecessor_edges(ctx: &mut Ctx, entry_idx: usize) {
    let blocks = &ctx.module.functions[entry_idx].blocks;
    let predecessors = block_predecessors(blocks);
    let dominators = block_dominators(blocks);
    let def_blocks = value_def_blocks(&ctx.module.functions[entry_idx]);

    let blocks = &mut ctx.module.functions[entry_idx].blocks;
    for block in blocks {
        let Some(label) = block.label.as_ref().and_then(|label| label.result_id) else {
            continue;
        };
        let Some(preds) = predecessors.get(&label) else {
            continue;
        };
        for inst in &mut block.instructions {
            if inst.class.opcode != Op::Phi {
                break;
            }
            let incoming = phi_incoming(inst);
            let incoming_preds = incoming
                .iter()
                .map(|(_, pred)| *pred)
                .collect::<HashSet<_>>();
            let actual_preds = preds.iter().copied().collect::<HashSet<_>>();
            let missing_preds = preds
                .iter()
                .copied()
                .filter(|pred| !incoming_preds.contains(pred))
                .collect::<Vec<_>>();
            for missing_pred in missing_preds {
                let mut replaced_stale = false;
                for pair in inst.operands.chunks_mut(2) {
                    let [value, Operand::IdRef(old_pred)] = pair else {
                        continue;
                    };
                    if actual_preds.contains(old_pred) {
                        continue;
                    }
                    if !phi_value_available_on_edge(value, missing_pred, &dominators, &def_blocks) {
                        continue;
                    }
                    *old_pred = missing_pred;
                    replaced_stale = true;
                    break;
                }
                if replaced_stale {
                    continue;
                }
                let Some((value, _)) = incoming.iter().find(|(value, _)| {
                    phi_value_available_on_edge(value, missing_pred, &dominators, &def_blocks)
                }) else {
                    continue;
                };
                inst.operands.push(value.clone());
                inst.operands.push(Operand::IdRef(missing_pred));
            }
            inst.operands = inst
                .operands
                .chunks(2)
                .filter(|pair| {
                    let [_, Operand::IdRef(pred)] = pair else {
                        return false;
                    };
                    actual_preds.contains(pred)
                })
                .flatten()
                .cloned()
                .collect();
        }
    }
}

pub(in crate::passes) fn block_index_by_label(blocks: &[Block], label_id: Word) -> Option<usize> {
    blocks
        .iter()
        .position(|block| block.label.as_ref().and_then(|label| label.result_id) == Some(label_id))
}

pub(in crate::passes) fn branch_only_target(block: &Block) -> Option<Word> {
    let [inst] = block.instructions.as_slice() else {
        return None;
    };
    (inst.class.opcode == Op::Branch)
        .then(|| inst.operands.first().and_then(id_ref_operand))
        .flatten()
}

pub(in crate::passes) fn phi_prefix_branch_target(block: &Block) -> Option<Word> {
    let mut iter = block.instructions.iter();
    let branch = loop {
        let inst = iter.next()?;
        if inst.class.opcode == Op::Phi {
            continue;
        }
        break inst;
    };
    if branch.class.opcode != Op::Branch || iter.next().is_some() {
        return None;
    }
    branch.operands.first().and_then(id_ref_operand)
}

pub(in crate::passes) fn block_predecessors(blocks: &[Block]) -> HashMap<Word, Vec<Word>> {
    let mut predecessors: HashMap<Word, Vec<Word>> = HashMap::new();
    for block in blocks {
        let Some(label) = block.label.as_ref().and_then(|label| label.result_id) else {
            continue;
        };
        for successor in block_successors(block) {
            let entry = predecessors.entry(successor).or_default();
            if !entry.contains(&label) {
                entry.push(label);
            }
        }
    }
    predecessors
}

pub(in crate::passes) fn all_paths_reach_target_without_labels(
    blocks: &[Block],
    start: Word,
    target: Word,
    forbidden: &[Word],
) -> bool {
    fn walk(
        blocks: &[Block],
        label: Word,
        target: Word,
        forbidden: &[Word],
        visiting: &mut HashSet<Word>,
        memo: &mut HashMap<Word, bool>,
    ) -> bool {
        if label == target {
            return true;
        }
        if forbidden.contains(&label) {
            return false;
        }
        if let Some(&cached) = memo.get(&label) {
            return cached;
        }
        if !visiting.insert(label) {
            return true;
        }
        let ok = block_index_by_label(blocks, label).is_some_and(|idx| {
            let successors = block_successors(&blocks[idx]);
            !successors.is_empty()
                && successors
                    .iter()
                    .all(|successor| walk(blocks, *successor, target, forbidden, visiting, memo))
        });
        visiting.remove(&label);
        memo.insert(label, ok);
        ok
    }

    walk(
        blocks,
        start,
        target,
        forbidden,
        &mut HashSet::new(),
        &mut HashMap::new(),
    )
}

pub(in crate::passes) fn reachable_before_target(
    blocks: &[Block],
    start: Word,
    target: Word,
) -> HashSet<Word> {
    let mut seen = HashSet::new();
    let mut stack = vec![start];
    while let Some(label) = stack.pop() {
        if label == target || !seen.insert(label) {
            continue;
        }
        let Some(idx) = block_index_by_label(blocks, label) else {
            continue;
        };
        stack.extend(block_successors(&blocks[idx]));
    }
    seen
}

pub(in crate::passes) fn id_ref_operand(operand: &Operand) -> Option<Word> {
    match operand {
        Operand::IdRef(id) => Some(*id),
        _ => None,
    }
}

pub(in crate::passes) fn redirect_terminator_target(
    block: &mut Block,
    from: Word,
    to: Word,
) -> bool {
    let Some(term) = block.instructions.last_mut() else {
        return false;
    };
    let mut changed = false;
    match term.class.opcode {
        Op::Branch => {
            if let Some(Operand::IdRef(label)) = term.operands.first_mut() {
                if *label == from {
                    *label = to;
                    changed = true;
                }
            }
        }
        Op::BranchConditional => {
            for operand in term.operands.iter_mut().skip(1).take(2) {
                if let Operand::IdRef(label) = operand {
                    if *label == from {
                        *label = to;
                        changed = true;
                    }
                }
            }
        }
        _ => {}
    }
    changed
}

pub(in crate::passes) fn replace_phi_predecessor(
    blocks: &mut [Block],
    target: Word,
    from: Word,
    to: Word,
) {
    let Some(block) = blocks
        .iter_mut()
        .find(|block| block.label.as_ref().and_then(|label| label.result_id) == Some(target))
    else {
        return;
    };
    for inst in &mut block.instructions {
        if inst.class.opcode != Op::Phi {
            break;
        }
        let mut idx = 1;
        while idx < inst.operands.len() {
            if inst.operands[idx] == Operand::IdRef(from) {
                inst.operands[idx] = Operand::IdRef(to);
            }
            idx += 2;
        }
    }
}

pub(in crate::passes) fn block_dominators(blocks: &[Block]) -> HashMap<Word, HashSet<Word>> {
    let labels = blocks
        .iter()
        .filter_map(|block| block.label.as_ref()?.result_id)
        .collect::<Vec<_>>();
    let Some(entry) = labels.first().copied() else {
        return HashMap::new();
    };
    let all_labels = labels.iter().copied().collect::<HashSet<_>>();
    let predecessors = block_predecessors(blocks);
    let mut dominators = labels
        .iter()
        .copied()
        .map(|label| {
            if label == entry {
                (label, HashSet::from([entry]))
            } else {
                (label, all_labels.clone())
            }
        })
        .collect::<HashMap<_, _>>();

    loop {
        let mut changed = false;
        for label in labels.iter().copied().filter(|label| *label != entry) {
            let mut next = predecessors
                .get(&label)
                .and_then(|preds| {
                    let mut iter = preds.iter();
                    let first = iter.next()?;
                    let mut out = dominators.get(first).cloned().unwrap_or_default();
                    for pred in iter {
                        let pred_doms = dominators.get(pred).cloned().unwrap_or_default();
                        out.retain(|dom| pred_doms.contains(dom));
                    }
                    Some(out)
                })
                .unwrap_or_default();
            next.insert(label);
            if dominators.get(&label) != Some(&next) {
                dominators.insert(label, next);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    dominators
}

pub(in crate::passes) fn value_def_blocks(function: &Function) -> HashMap<Word, Word> {
    let mut defs = HashMap::new();
    for block in &function.blocks {
        let Some(label) = block.label.as_ref().and_then(|label| label.result_id) else {
            continue;
        };
        for inst in &block.instructions {
            if let Some(result) = inst.result_id {
                defs.insert(result, label);
            }
        }
    }
    defs
}

pub(in crate::passes) fn phi_incoming(inst: &Instruction) -> Vec<(Operand, Word)> {
    inst.operands
        .chunks(2)
        .filter_map(|pair| {
            let [value, Operand::IdRef(pred)] = pair else {
                return None;
            };
            Some((value.clone(), *pred))
        })
        .collect()
}

pub(in crate::passes) fn phi_value_available_on_edge(
    value: &Operand,
    pred: Word,
    dominators: &HashMap<Word, HashSet<Word>>,
    def_blocks: &HashMap<Word, Word>,
) -> bool {
    let Operand::IdRef(value) = value else {
        return true;
    };
    let Some(def_block) = def_blocks.get(value).copied() else {
        return true;
    };
    dominators
        .get(&pred)
        .is_some_and(|pred_dominators| pred_dominators.contains(&def_block))
}
