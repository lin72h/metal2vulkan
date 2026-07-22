//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

pub(in crate::native::emitter) fn block_dominators(
    blocks: &[Block],
) -> HashMap<Word, HashSet<Word>> {
    let labels = blocks
        .iter()
        .filter_map(|block| block.label.as_ref()?.result_id)
        .collect::<Vec<_>>();
    let Some(entry) = labels.first().copied() else {
        return HashMap::new();
    };
    let label_indices = labels
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, label)| (label, idx))
        .collect::<HashMap<_, _>>();
    let word_len = labels.len().div_ceil(u64::BITS as usize);
    let mut full = vec![u64::MAX; word_len];
    clear_unused_bits(&mut full, labels.len());
    let mut predecessors = vec![Vec::<usize>::new(); labels.len()];
    for block in blocks {
        let Some(label) = block.label.as_ref().and_then(|label| label.result_id) else {
            continue;
        };
        let Some(label_idx) = label_indices.get(&label).copied() else {
            continue;
        };
        for successor in cloned_block_successors(block) {
            if let Some(successor_idx) = label_indices.get(&successor).copied() {
                predecessors[successor_idx].push(label_idx);
            }
        }
    }

    let mut dominators = vec![full.clone(); labels.len()];
    dominators[0] = vec![0; word_len];
    set_bit(&mut dominators[0], 0);

    loop {
        let mut changed = false;
        for idx in 1..labels.len() {
            let mut next = if let Some((first, rest)) = predecessors[idx].split_first() {
                let mut out = dominators[*first].clone();
                for pred in rest {
                    for word in 0..word_len {
                        out[word] &= dominators[*pred][word];
                    }
                }
                out
            } else {
                vec![0; word_len]
            };
            set_bit(&mut next, idx);
            if dominators[idx] != next {
                dominators[idx] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut out = HashMap::new();
    for (idx, label) in labels.iter().copied().enumerate() {
        let mut doms = HashSet::new();
        for (candidate_idx, candidate_label) in labels.iter().copied().enumerate() {
            if bit_is_set(&dominators[idx], candidate_idx) {
                doms.insert(candidate_label);
            }
        }
        out.insert(label, doms);
    }
    debug_assert!(out
        .get(&entry)
        .is_some_and(|doms| doms == &HashSet::from([entry])));
    out
}

pub(in crate::native::emitter) fn block_value_defs(blocks: &[Block]) -> HashMap<Word, Word> {
    let mut defs = HashMap::new();
    for block in blocks {
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

pub(in crate::native::emitter) fn has_non_leading_phi(block: &Block) -> bool {
    let mut saw_non_phi = false;
    for inst in &block.instructions {
        if inst.class.opcode == Op::Phi {
            if saw_non_phi {
                return true;
            }
        } else {
            saw_non_phi = true;
        }
    }
    false
}

pub(in crate::native::emitter) fn is_phi_incoming_materialization(inst: &Instruction) -> bool {
    matches!(
        inst.class.opcode,
        Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
    ) && inst.result_id.is_some()
}

pub(in crate::native::emitter) fn unique_phi_incoming_predecessor_for_value(
    blocks: &[Block],
    target_label: Word,
    value_id: Word,
) -> Option<Word> {
    let mut incoming_pred = None;
    for block in blocks {
        let label = block.label.as_ref().and_then(|label| label.result_id);
        for inst in &block.instructions {
            for operand in &inst.operands {
                if id_ref_operand(operand) != Some(value_id) {
                    continue;
                }
                let is_target_phi_incoming = label == Some(target_label)
                    && inst.class.opcode == Op::Phi
                    && inst.operands.chunks(2).any(|pair| {
                        matches!(
                            pair,
                            [Operand::IdRef(value), Operand::IdRef(_)] if *value == value_id
                        )
                    });
                if !is_target_phi_incoming {
                    return None;
                }
            }
            if label != Some(target_label) || inst.class.opcode != Op::Phi {
                continue;
            }
            for pair in inst.operands.chunks(2) {
                let [Operand::IdRef(value), Operand::IdRef(pred)] = pair else {
                    continue;
                };
                if *value != value_id {
                    continue;
                }
                match incoming_pred {
                    Some(existing) if existing != *pred => return None,
                    Some(_) => {}
                    None => incoming_pred = Some(*pred),
                }
            }
        }
    }
    incoming_pred
}

pub(in crate::native::emitter) fn materialization_operands_dominate_pred(
    inst: &Instruction,
    result: Word,
    target_label: Word,
    pred: Word,
    defs: &HashMap<Word, Word>,
    dominators: &HashMap<Word, HashSet<Word>>,
) -> bool {
    for operand in &inst.operands {
        let Some(id) = id_ref_operand(operand) else {
            continue;
        };
        if id == result {
            continue;
        }
        let Some(def_label) = defs.get(&id).copied() else {
            continue;
        };
        if def_label == target_label || !label_dominates(dominators, def_label, pred) {
            return false;
        }
    }
    true
}

pub(in crate::native::emitter) fn terminator_insert_index(block: &Block) -> usize {
    let term_idx = block.instructions.len().saturating_sub(1);
    if term_idx > 0
        && matches!(
            block.instructions[term_idx - 1].class.opcode,
            Op::SelectionMerge | Op::LoopMerge
        )
    {
        term_idx - 1
    } else {
        term_idx
    }
}
