//! Post-emit pre-phi materialization fixup — the sole survivor of the W4 repair-roster deletion.
//!
//! The ~4,800-line post-hoc CFG repair roster (the 4x fixpoint + `repair_*`/`split_*`/`clone_*`
//! surgery) was deleted at W4 (2026-07-16): `METAL2VULKAN_NO_REPAIR` proved the retry cascade ships
//! the full regression set spirv-val-valid without it. What remains here is the ONE fixup that runs on the
//! STRUCTURED (admitted) path, not on reject repair: relocating a pointer-phi's incoming access-chain
//! materialization out of the phi's own block. It is CFG-structure-agnostic (moves a single
//! materialization to its unique phi-incoming predecessor edge under dominance guards; never reorders
//! blocks or rewrites merges), so it is independent of the deleted roster.

use super::*;

impl Emitter {
    pub(in crate::native) fn repair_pre_phi_incoming_materializations(
        &mut self,
        blocks: &mut [Block],
    ) {
        // Relocating an instruction changes its defining block but never changes the CFG. Build the
        // compact dominator tree once; only the much smaller value-definition map must be refreshed
        // after a move. The former full dominator-set map was O(V^2) and rebuilt for every relocated
        // materialization, leaving hundreds of MiB of allocator high-water pages on large functions.
        let dominators = emitted_block_dominators(blocks);
        loop {
            let defs = block_value_defs(blocks);
            let mut moved = false;
            'scan: for target_idx in 0..blocks.len() {
                let Some(target_label) = blocks[target_idx]
                    .label
                    .as_ref()
                    .and_then(|label| label.result_id)
                else {
                    continue;
                };
                if !has_non_leading_phi(&blocks[target_idx]) {
                    continue;
                }
                for inst_idx in 0..blocks[target_idx].instructions.len() {
                    let inst = &blocks[target_idx].instructions[inst_idx];
                    if !is_phi_incoming_materialization(inst)
                        || !blocks[target_idx]
                            .instructions
                            .iter()
                            .skip(inst_idx + 1)
                            .any(|inst| inst.class.opcode == Op::Phi)
                    {
                        continue;
                    }
                    let Some(result) = inst.result_id else {
                        continue;
                    };
                    let Some(pred) =
                        unique_phi_incoming_predecessor_for_value(blocks, target_label, result)
                    else {
                        continue;
                    };
                    if pred == target_label
                        || !materialization_operands_dominate_pred(
                            inst,
                            result,
                            target_label,
                            pred,
                            &defs,
                            &dominators,
                        )
                    {
                        continue;
                    }
                    let Some(pred_idx) = block_index_by_label(blocks, pred) else {
                        continue;
                    };
                    let inst = blocks[target_idx].instructions.remove(inst_idx);
                    let insert_idx = terminator_insert_index(&blocks[pred_idx]);
                    blocks[pred_idx].instructions.insert(insert_idx, inst);
                    moved = true;
                    break 'scan;
                }
            }
            if !moved {
                break;
            }
        }

        // A non-phi instruction can remain between phis even when it is not itself a phi incoming
        // materialization (for example, index arithmetic rematerialized for an earlier pointer phi in
        // the same block). Every OpPhi semantically executes at block entry and its operands come from
        // predecessor edges, so stable-partitioning the residual phis ahead of ordinary instructions is
        // the canonical SPIR-V order. Materializations that actually feed a phi were relocated by the
        // dominance-checked loop above first; any unsupported same-block dependency remains invalid and
        // is still rejected by the caller's module-level validation gate.
        for block in blocks {
            if !has_non_leading_phi(block) {
                continue;
            }
            let instructions = std::mem::take(&mut block.instructions);
            let (mut phis, rest): (Vec<_>, Vec<_>) = instructions
                .into_iter()
                .partition(|inst| inst.class.opcode == Op::Phi);
            phis.extend(rest);
            block.instructions = phis;
        }
    }
}
