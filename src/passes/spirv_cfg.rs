//! Passes-side SPIR-V (`crate::spirv_module::Block`, `Word` label) CFG primitives.
//!
//! The final passes walk the same retained owned-`Block` control-flow graph the native emitter
//! does. The native side has its own equivalents in `native::cfg::graph`; these cannot be shared
//! because the passes layer must not depend on `native` (that would invert the ownership
//! direction). This is the single
//! passes-side home for the Word-layer successor scan, replacing the byte-identical
//! copies formerly open-coded in `inline/mod.rs` and `lower/access.rs`.

use crate::spirv_module::Block;
use crate::spirv_module::Operand;
use spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

/// Successor block labels of one owned `Block`, read from its terminator. Branch/BranchConditional
/// arms in operand order; switch default + case targets sorted and deduped. Non-terminating or
/// unstructured-terminator blocks yield no successors.
pub(in crate::passes) fn block_successors(block: &Block) -> Vec<Word> {
    fn id_ref(operand: &Operand) -> Option<Word> {
        match operand {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        }
    }
    let Some(inst) = block.instructions.last() else {
        return Vec::new();
    };
    match inst.class.opcode {
        Op::Branch => inst.operands.first().and_then(id_ref).into_iter().collect(),
        Op::BranchConditional => inst
            .operands
            .iter()
            .skip(1)
            .take(2)
            .filter_map(id_ref)
            .collect(),
        Op::Switch => {
            let mut out = Vec::new();
            if let Some(default) = inst.operands.get(1).and_then(id_ref) {
                out.push(default);
            }
            let mut idx = 3;
            while idx < inst.operands.len() {
                if let Some(target) = inst.operands.get(idx).and_then(id_ref) {
                    out.push(target);
                }
                idx += 2;
            }
            out.sort_unstable();
            out.dedup();
            out
        }
        _ => Vec::new(),
    }
}

/// Forward-edge adjacency of an owned SPIR-V function body keyed by block label id
/// (via [`block_successors`]). Blocks without a label id are skipped.
pub(in crate::passes) fn block_successors_by_label(blocks: &[Block]) -> HashMap<Word, Vec<Word>> {
    blocks
        .iter()
        .filter_map(|block| Some((block.label.as_ref()?.result_id?, block_successors(block))))
        .collect()
}

/// Whether `dominator` dominates `label` in a precomputed Word-keyed dominator-set map
/// (`label -> {blocks that dominate it}`).
pub(in crate::passes) fn label_dominates(
    dominators: &HashMap<Word, HashSet<Word>>,
    dominator: Word,
    label: Word,
) -> bool {
    dominators
        .get(&label)
        .is_some_and(|labels| labels.contains(&dominator))
}
