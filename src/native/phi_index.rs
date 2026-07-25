//! M4 (pointer-typing rewrite): "phi the index, not the pointer" lowering for an illegal
//! logical-pointer `OpPhi`.
//!
//! A pointer `OpPhi` whose result type is in a storage class `VariablePointers` cannot cover — i.e.
//! NOT `StorageBuffer`/`Workgroup` (so `Private`/`UniformConstant`/`Function`/…) — is rejected by
//! spirv-val: *"Instruction may only have a logical pointer operand in the StorageBuffer or Workgroup
//! storage classes with appropriate variable pointers capability"*. This is the shape a loop walking a
//! pointer through an aggregate emits (e.g. a constant lookup table `[2 x [4 x i32]]` indexed across
//! iterations) — the pointer is a phi of access chains into one base.
//!
//! When every incoming pointer of such a phi is an `(In)BoundsAccessChain` into the **same base** with
//! the **same arity**, the merge is expressible WITHOUT a pointer phi: phi each index position (plain
//! integer phis are always legal), rematerialize a single access chain after the phis, and replace the
//! pointer phi's uses with it. The pointer phi is removed; its now-dead arm access chains are left for
//! id-canonicalization to drop.
//!
//! Applied in `lib.rs`'s failure-triggered retry (adopt-if-VALIDATES), so it is floor-safe by
//! construction: a module that already validates never reaches it, and a rewrite that does not produce
//! a validating module is discarded.

use crate::spirv_module::Instruction;
use crate::spirv_module::Module;
use crate::spirv_module::Operand;
use spirv::{Op, StorageClass, Word};
use std::collections::HashMap;

/// Rewrite every eligible illegal logical-pointer `OpPhi` in the module into phi-the-index form.
/// Returns true if any phi was rewritten.
pub(super) fn rewrite_logical_pointer_phis(module: &mut Module) -> bool {
    rewrite_pointer_phis(module, false)
}

/// Like [`rewrite_logical_pointer_phis`], but ALSO rewrites `StorageBuffer`/`Workgroup` pointer
/// phis — legal SPIR-V under `VariablePointers(StorageBuffer)`, but a shape MoltenVK's SPIRV-Cross
/// MSL backend cannot always express (pipeline creation fails with `cannot initialize a variable of
/// type 'device float *' with an lvalue of type 'device float'`). The index-phi form is semantically
/// identical and needs no variable pointers, so it is strictly more portable.
pub(super) fn rewrite_variable_pointer_phis(module: &mut Module) -> bool {
    rewrite_pointer_phis(module, true)
}

/// Legalize an integer `OpPhi` whose result type is a NARROWER integer than one of its incoming
/// values by inserting an `OpUConvert` (== LLVM `trunc`) in that incoming's predecessor block, so the
/// phi operand matches the result type. This reconstructs a truncation the emitter drops when it
/// lowers a wide (`i64`/`ulong`) induction back-edge into a narrow (`i32`/`uint`) loop phi: the AIR phi
/// is `i32` (its init constant and result are `uint`, and its value is used as an array index) but a
/// parallel `i64` induction feeds the back edge, so the phi ends up with a `%ulong` incoming —
/// spirv-val rejects `OpPhi's result type does not match incoming value type`, and even ignoring
/// validity the missing trunc is byte-wrong.
///
/// Only NARROWING is handled: truncation keeps the low bits regardless of signedness (`OpUConvert` ==
/// LLVM `trunc`), so no sign analysis is needed. Widening would need the incoming's signedness and is
/// left alone. Floor-safe by construction — a valid module has every phi operand already matching the
/// result type, so this never fires on a validating module. Returns true if any operand was converted.
pub(super) fn rewrite_integer_width_phis(module: &mut Module) -> bool {
    // int type id -> (width, signedness). Only scalar OpTypeInt (vectors are OpTypeVector, skipped).
    let mut int_types: HashMap<Word, (u32, u32)> = HashMap::new();
    // global (module-scope) value id -> its result type, so a wide OpConstant incoming is typed too.
    let mut global_value_type: HashMap<Word, Word> = HashMap::new();
    for i in &module.types_global_values {
        if i.class.opcode == Op::TypeInt {
            if let (Some(rid), Some(Operand::LiteralBit32(w)), Some(Operand::LiteralBit32(s))) =
                (i.result_id, i.operands.first(), i.operands.get(1))
            {
                int_types.insert(rid, (*w, *s));
            }
        }
        if let (Some(rid), Some(rty)) = (i.result_id, i.result_type) {
            global_value_type.insert(rid, rty);
        }
    }
    if int_types.is_empty() {
        return false;
    }

    let mut next_id = module.header.as_ref().map(|h| h.bound).unwrap_or(0);
    let mut fresh = || {
        let id = next_id;
        next_id += 1;
        id
    };

    let mut any = false;
    for function in &mut module.functions {
        // value id -> result type, across the whole function, plus module-scope values.
        let mut value_type: HashMap<Word, Word> = global_value_type.clone();
        for block in &function.blocks {
            for inst in &block.instructions {
                if let (Some(rid), Some(rty)) = (inst.result_id, inst.result_type) {
                    value_type.insert(rid, rty);
                }
            }
        }
        // label id -> block index, for locating a phi's predecessor block.
        let mut label_to_block: HashMap<Word, usize> = HashMap::new();
        for (bi, block) in function.blocks.iter().enumerate() {
            if let Some(rid) = block.label.as_ref().and_then(|l| l.result_id) {
                label_to_block.insert(rid, bi);
            }
        }

        // Plan phase (immutable scan): a conversion per (phi operand) that narrows.
        struct Conv {
            phi_block: usize,
            phi_result: Word,
            value_pos: usize, // operand index in the phi of the value to replace
            pred_label: Word,
            value: Word,
            target_ty: Word,
        }
        let mut convs: Vec<Conv> = Vec::new();
        for (bi, block) in function.blocks.iter().enumerate() {
            for inst in &block.instructions {
                if inst.class.opcode != Op::Phi {
                    continue;
                }
                let Some(rty) = inst.result_type else {
                    continue;
                };
                let Some(&(rwidth, rsign)) = int_types.get(&rty) else {
                    continue; // not a scalar-int phi
                };
                let Some(phi_result) = inst.result_id else {
                    continue;
                };
                // Operands are [v0, b0, v1, b1, ...]; the value operands sit at even positions.
                let mut pos = 0;
                while pos + 1 < inst.operands.len() {
                    if let (Operand::IdRef(v), Operand::IdRef(b)) =
                        (&inst.operands[pos], &inst.operands[pos + 1])
                    {
                        if let Some(vty) = value_type.get(v) {
                            if let Some(&(vwidth, _)) = int_types.get(vty) {
                                // Narrowing only, result signedness 0 (OpUConvert requires it), and
                                // the predecessor block must be locatable.
                                if vwidth > rwidth && rsign == 0 && label_to_block.contains_key(b) {
                                    convs.push(Conv {
                                        phi_block: bi,
                                        phi_result,
                                        value_pos: pos,
                                        pred_label: *b,
                                        value: *v,
                                        target_ty: rty,
                                    });
                                }
                            }
                        }
                    }
                    pos += 2;
                }
            }
        }
        if convs.is_empty() {
            continue;
        }

        // Apply phase: one OpUConvert per unique (pred, value, target) inserted before the
        // predecessor's terminator (and before any structured merge), then repoint the phi operands.
        let mut conv_id: HashMap<(Word, Word, Word), Word> = HashMap::new();
        for c in &convs {
            conv_id
                .entry((c.pred_label, c.value, c.target_ty))
                .or_insert_with(&mut fresh);
        }
        // Materialize the converts (each unique key inserted once).
        let mut inserted: std::collections::HashSet<(Word, Word, Word)> = Default::default();
        for c in &convs {
            let key = (c.pred_label, c.value, c.target_ty);
            if !inserted.insert(key) {
                continue;
            }
            let new_id = conv_id[&key];
            let pred_idx = label_to_block[&c.pred_label];
            let block = &mut function.blocks[pred_idx];
            let mut insert_at = block.instructions.len().saturating_sub(1); // before terminator
            if insert_at > 0
                && matches!(
                    block.instructions[insert_at - 1].class.opcode,
                    Op::SelectionMerge | Op::LoopMerge
                )
            {
                insert_at -= 1; // keep the structured merge adjacent to its branch
            }
            block.instructions.insert(
                insert_at,
                Instruction::new(
                    Op::UConvert,
                    Some(c.target_ty),
                    Some(new_id),
                    vec![Operand::IdRef(c.value)],
                ),
            );
            any = true;
        }
        // Repoint each planned phi operand to its converted id.
        for c in &convs {
            let new_id = conv_id[&(c.pred_label, c.value, c.target_ty)];
            let block = &mut function.blocks[c.phi_block];
            if let Some(inst) = block
                .instructions
                .iter_mut()
                .find(|i| i.result_id == Some(c.phi_result))
            {
                if let Some(op) = inst.operands.get_mut(c.value_pos) {
                    *op = Operand::IdRef(new_id);
                }
            }
        }
    }

    if any {
        if let Some(header) = module.header.as_mut() {
            header.bound = next_id;
        }
    }
    any
}

struct SelectInductionPlan {
    block: usize,
    phi_result: Word,
    ptr_ty: Word,
    base: Word,
    chain_opcode: Op,
    prefix_indices: Vec<Word>,
    init_index: Word,
    init_pred: Word,
    index_ty: Word,
    select_id: Word,
    select_block: usize,
    cond: Word,
    advanced_on_true: bool,
    step: Word,
    advanced_ptr: Word,
    back_pred: Word,
}

fn rewrite_pointer_phis(module: &mut Module, include_variable_pointer_classes: bool) -> bool {
    // type-id -> OpTypePointer storage class, and the set of module-scope OpVariable ids (a legal,
    // everything-dominating access-chain base).
    let type_defs: HashMap<Word, Instruction> = module
        .types_global_values
        .iter()
        .filter_map(|i| i.result_id.map(|id| (id, i.clone())))
        .collect();
    let global_vars: std::collections::HashSet<Word> = module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::Variable)
        .filter_map(|i| i.result_id)
        .collect();
    let ptr_storage = |ptr_ty: Word| -> Option<StorageClass> {
        let inst = type_defs.get(&ptr_ty)?;
        if inst.class.opcode != Op::TypePointer {
            return None;
        }
        match inst.operands.first()? {
            Operand::StorageClass(s) => Some(*s),
            _ => None,
        }
    };

    let mut next_id = module.header.as_ref().map(|h| h.bound).unwrap_or(0);
    let mut fresh = || {
        let id = next_id;
        next_id += 1;
        id
    };

    let mut any = false;
    for function in &mut module.functions {
        // Eligible access-chain bases: any base that dominates the whole function, so a rematerialized
        // access chain is legal anywhere. That is a module-scope OpVariable OR a function-entry-block
        // OpVariable (`alloca` — SPIR-V requires all function variables in the first block, so they
        // dominate every block). A non-variable base (a derived pointer) is rejected, since proving its
        // dominance would need real dominator analysis.
        let mut eligible_bases = global_vars.clone();
        if let Some(entry) = function.blocks.first() {
            for inst in &entry.instructions {
                if inst.class.opcode == Op::Variable {
                    if let Some(rid) = inst.result_id {
                        eligible_bases.insert(rid);
                    }
                }
            }
        }
        // value-id -> defining instruction + result type/block, across the whole function.
        let mut value_def: HashMap<Word, Instruction> = HashMap::new();
        let mut value_type: HashMap<Word, Word> = HashMap::new();
        let mut value_block: HashMap<Word, usize> = HashMap::new();
        for (bi, block) in function.blocks.iter().enumerate() {
            for inst in &block.instructions {
                if let Some(rid) = inst.result_id {
                    value_def.insert(rid, inst.clone());
                    value_block.insert(rid, bi);
                    if let Some(rty) = inst.result_type {
                        value_type.insert(rid, rty);
                    }
                }
            }
        }
        let mut use_count: HashMap<Word, usize> = HashMap::new();
        for block in &function.blocks {
            for inst in &block.instructions {
                for op in &inst.operands {
                    if let Operand::IdRef(id) = op {
                        *use_count.entry(*id).or_default() += 1;
                    }
                }
            }
        }
        // Index operand types also come from module constants (e.g. `%uint_0`).
        let const_type = |id: Word| -> Option<Word> {
            value_type
                .get(&id)
                .copied()
                .or_else(|| type_defs.get(&id).and_then(|i| i.result_type))
        };
        let zero_by_type: HashMap<Word, Word> = type_defs
            .iter()
            .filter_map(|(&id, inst)| {
                let ty = inst.result_type?;
                is_zero_int_constant(&type_defs, inst).then_some((ty, id))
            })
            .collect();
        let zero_ids: std::collections::HashSet<Word> = type_defs
            .iter()
            .filter_map(|(&id, inst)| is_zero_int_constant(&type_defs, inst).then_some(id))
            .collect();

        // Plan rewrites first (immutable scan), then mutate the blocks.
        // Per index position: either reuse a single operand all arms share (REQUIRED for struct member
        // indices, which must stay OpConstant), or phi the differing per-arm operands (legal only for
        // array/vector indices — a struct member index that differs across arms makes the rewrite
        // invalid, but adopt-if-validates discards it, so it cannot regress).
        enum IndexSrc {
            Reuse(Word),
            Phi(Word, Vec<(Word, Word)>),
        }
        struct Plan {
            block: usize,
            phi_result: Word,
            ptr_ty: Word,
            base: Word,
            index_srcs: Vec<IndexSrc>,
        }
        let mut plans: Vec<Plan> = Vec::new();
        let mut induction_plans: Vec<SelectInductionPlan> = Vec::new();

        for (bi, block) in function.blocks.iter().enumerate() {
            for inst in &block.instructions {
                if inst.class.opcode != Op::Phi {
                    continue;
                }
                let Some(ptr_ty) = inst.result_type else {
                    continue;
                };
                let Some(storage) = ptr_storage(ptr_ty) else {
                    continue; // not a pointer phi
                };
                if !include_variable_pointer_classes
                    && matches!(
                        storage,
                        StorageClass::StorageBuffer | StorageClass::Workgroup
                    )
                {
                    continue; // legal under VariablePointers — leave it
                }
                let Some(phi_result) = inst.result_id else {
                    continue;
                };
                // Operands are [v0, b0, v1, b1, ...].
                let arms: Vec<(Word, Word)> = inst
                    .operands
                    .chunks(2)
                    .filter_map(|c| match (c.first(), c.get(1)) {
                        (Some(Operand::IdRef(v)), Some(Operand::IdRef(b))) => Some((*v, *b)),
                        _ => None,
                    })
                    .collect();
                if arms.len() < 2 || arms.len() * 2 != inst.operands.len() {
                    continue;
                }
                if arms.len() == 2 {
                    if let Some(plan) = select_induction_plan(
                        bi,
                        phi_result,
                        ptr_ty,
                        &arms,
                        &value_def,
                        &value_block,
                        &use_count,
                        &eligible_bases,
                        &const_type,
                        &zero_by_type,
                        &zero_ids,
                    ) {
                        induction_plans.push(plan);
                        continue;
                    }
                }
                // Every arm must be an (In)BoundsAccessChain into the SAME global-var base with equal
                // arity.
                let mut base: Option<Word> = None;
                let mut arity: Option<usize> = None;
                let mut arm_defs: Vec<(Instruction, Word)> = Vec::new(); // (access-chain, block_label)
                let mut ok = true;
                for (v, b) in &arms {
                    let Some(def) = value_def.get(v) else {
                        ok = false;
                        break;
                    };
                    if !matches!(def.class.opcode, Op::AccessChain | Op::InBoundsAccessChain) {
                        ok = false;
                        break;
                    }
                    let Some(Operand::IdRef(b0)) = def.operands.first() else {
                        ok = false;
                        break;
                    };
                    if !eligible_bases.contains(b0) {
                        ok = false;
                        break;
                    }
                    let k = def.operands.len() - 1;
                    match (base, arity) {
                        (None, None) => {
                            base = Some(*b0);
                            arity = Some(k);
                        }
                        (Some(pb), Some(pk)) if pb == *b0 && pk == k => {}
                        _ => {
                            ok = false;
                            break;
                        }
                    }
                    arm_defs.push((def.clone(), *b));
                }
                if !ok {
                    continue;
                }
                let (Some(base), Some(k)) = (base, arity) else {
                    continue;
                };
                if k == 0 {
                    continue;
                }
                // Per position: reuse a shared operand (keeps constant struct indices constant) or phi
                // the differing operands.
                let mut index_srcs: Vec<IndexSrc> = Vec::new();
                let mut positions_ok = true;
                for j in 0..k {
                    let mut entries: Vec<(Word, Word)> = Vec::new();
                    let mut idx_ty: Option<Word> = None;
                    for (def, blk) in &arm_defs {
                        let Some(Operand::IdRef(idx)) = def.operands.get(1 + j) else {
                            positions_ok = false;
                            break;
                        };
                        if idx_ty.is_none() {
                            idx_ty = const_type(*idx);
                        }
                        entries.push((*idx, *blk));
                    }
                    if !positions_ok {
                        break;
                    }
                    if entries.iter().all(|(v, _)| *v == entries[0].0) {
                        index_srcs.push(IndexSrc::Reuse(entries[0].0));
                    } else {
                        let Some(idx_ty) = idx_ty else {
                            positions_ok = false;
                            break;
                        };
                        index_srcs.push(IndexSrc::Phi(idx_ty, entries));
                    }
                }
                if !positions_ok {
                    continue;
                }
                plans.push(Plan {
                    block: bi,
                    phi_result,
                    ptr_ty,
                    base,
                    index_srcs,
                });
            }
        }

        if plans.is_empty() && induction_plans.is_empty() {
            continue;
        }

        let mut remap: HashMap<Word, Word> = HashMap::new();
        let mut remove_results: std::collections::HashSet<Word> = Default::default();

        // Apply select-fed pointer inductions first: `%p = phi(base, select(cond, gep(%p, step), %p))`
        // becomes an integer-index phi plus a rematerialized access chain from the stable base.
        for plan in &induction_plans {
            let index_phi_id = fresh();
            let next_sum_id = fresh();
            let next_index_id = fresh();
            let new_ptr_id = fresh();

            let block = &mut function.blocks[plan.block];
            block
                .instructions
                .retain(|i| i.result_id != Some(plan.phi_result));
            let phi_insert = block
                .instructions
                .iter()
                .position(|i| i.class.opcode != Op::Phi)
                .unwrap_or(block.instructions.len());
            block.instructions.insert(
                phi_insert,
                Instruction::new(
                    Op::Phi,
                    Some(plan.index_ty),
                    Some(index_phi_id),
                    vec![
                        Operand::IdRef(plan.init_index),
                        Operand::IdRef(plan.init_pred),
                        Operand::IdRef(next_index_id),
                        Operand::IdRef(plan.back_pred),
                    ],
                ),
            );
            let mut chain_args: Vec<Operand> = Vec::with_capacity(2 + plan.prefix_indices.len());
            chain_args.push(Operand::IdRef(plan.base));
            chain_args.extend(plan.prefix_indices.iter().copied().map(Operand::IdRef));
            chain_args.push(Operand::IdRef(index_phi_id));
            let after_phis = block
                .instructions
                .iter()
                .position(|i| i.class.opcode != Op::Phi)
                .unwrap_or(block.instructions.len());
            block.instructions.insert(
                after_phis,
                Instruction::new(
                    plan.chain_opcode,
                    Some(plan.ptr_ty),
                    Some(new_ptr_id),
                    chain_args,
                ),
            );

            let select_block = &mut function.blocks[plan.select_block];
            if let Some(select_pos) = select_block
                .instructions
                .iter()
                .position(|i| i.result_id == Some(plan.select_id))
            {
                select_block.instructions.insert(
                    select_pos,
                    Instruction::new(
                        Op::IAdd,
                        Some(plan.index_ty),
                        Some(next_sum_id),
                        vec![Operand::IdRef(index_phi_id), Operand::IdRef(plan.step)],
                    ),
                );
                let (true_value, false_value) = if plan.advanced_on_true {
                    (next_sum_id, index_phi_id)
                } else {
                    (index_phi_id, next_sum_id)
                };
                select_block.instructions.insert(
                    select_pos + 1,
                    Instruction::new(
                        Op::Select,
                        Some(plan.index_ty),
                        Some(next_index_id),
                        vec![
                            Operand::IdRef(plan.cond),
                            Operand::IdRef(true_value),
                            Operand::IdRef(false_value),
                        ],
                    ),
                );
            }

            remap.insert(plan.phi_result, new_ptr_id);
            remove_results.insert(plan.select_id);
            if use_count.get(&plan.advanced_ptr).copied().unwrap_or(0) == 1 {
                remove_results.insert(plan.advanced_ptr);
            }
            any = true;
        }

        // Apply: for each plan synthesize index phis + a rematerialized access chain, splice into the
        // block, and remap the old phi result to the new access chain.
        for plan in &plans {
            let block = &mut function.blocks[plan.block];
            let mut new_phis: Vec<Instruction> = Vec::new();
            let mut chain_args: Vec<Operand> = vec![Operand::IdRef(plan.base)];
            for src in &plan.index_srcs {
                match src {
                    IndexSrc::Reuse(id) => chain_args.push(Operand::IdRef(*id)),
                    IndexSrc::Phi(idx_ty, entries) => {
                        let idx_phi_id = fresh();
                        let mut ops: Vec<Operand> = Vec::with_capacity(entries.len() * 2);
                        for (val, blk) in entries {
                            ops.push(Operand::IdRef(*val));
                            ops.push(Operand::IdRef(*blk));
                        }
                        new_phis.push(Instruction::new(
                            Op::Phi,
                            Some(*idx_ty),
                            Some(idx_phi_id),
                            ops,
                        ));
                        chain_args.push(Operand::IdRef(idx_phi_id));
                    }
                }
            }
            let new_ptr_id = fresh();
            let access = Instruction::new(
                Op::InBoundsAccessChain,
                Some(plan.ptr_ty),
                Some(new_ptr_id),
                chain_args,
            );
            // Remove the old pointer phi.
            block
                .instructions
                .retain(|i| i.result_id != Some(plan.phi_result));
            // OpPhi must lead the block: prepend the new index phis, then place the access chain right
            // after the last (any) phi.
            for (k, p) in new_phis.into_iter().enumerate() {
                block.instructions.insert(k, p);
            }
            let after_phis = block
                .instructions
                .iter()
                .position(|i| i.class.opcode != Op::Phi)
                .unwrap_or(block.instructions.len());
            block.instructions.insert(after_phis, access);
            remap.insert(plan.phi_result, new_ptr_id);
            any = true;
        }

        // Replace every use of a remapped (old phi) id with its rematerialized access chain id.
        if !remap.is_empty() {
            for block in &mut function.blocks {
                for inst in &mut block.instructions {
                    for op in &mut inst.operands {
                        if let Operand::IdRef(id) = op {
                            if let Some(new) = remap.get(id) {
                                *id = *new;
                            }
                        }
                    }
                }
            }
        }

        if !remove_results.is_empty() {
            for block in &mut function.blocks {
                block
                    .instructions
                    .retain(|i| i.result_id.is_none_or(|id| !remove_results.contains(&id)));
            }
        }
    }

    if any {
        if let Some(header) = module.header.as_mut() {
            header.bound = next_id;
        }
    }
    any
}

fn select_induction_plan<F>(
    block: usize,
    phi_result: Word,
    ptr_ty: Word,
    arms: &[(Word, Word)],
    value_def: &HashMap<Word, Instruction>,
    value_block: &HashMap<Word, usize>,
    use_count: &HashMap<Word, usize>,
    eligible_bases: &std::collections::HashSet<Word>,
    const_type: &F,
    zero_by_type: &HashMap<Word, Word>,
    zero_ids: &std::collections::HashSet<Word>,
) -> Option<SelectInductionPlan>
where
    F: Fn(Word) -> Option<Word>,
{
    let mut base_arm: Option<(&Instruction, Word)> = None;
    let mut select_arm: Option<(&Instruction, Word)> = None;
    for (value, pred) in arms {
        let def = value_def.get(value)?;
        if matches!(def.class.opcode, Op::AccessChain | Op::InBoundsAccessChain) {
            base_arm = Some((def, *pred));
        } else if def.class.opcode == Op::Select && def.result_type == Some(ptr_ty) {
            select_arm = Some((def, *pred));
        } else {
            return None;
        }
    }

    let (base_def, init_pred) = base_arm?;
    let (select_def, back_pred) = select_arm?;
    let select_id = select_def.result_id?;
    if use_count.get(&select_id).copied().unwrap_or(0) != 1 {
        return None;
    }

    let Operand::IdRef(base) = base_def.operands.first()? else {
        return None;
    };
    if !eligible_bases.contains(base) || base_def.operands.len() < 2 {
        return None;
    }
    let mut base_indices = Vec::with_capacity(base_def.operands.len() - 1);
    for operand in &base_def.operands[1..] {
        let Operand::IdRef(index) = operand else {
            return None;
        };
        base_indices.push(*index);
    }
    let init_index = *base_indices.last()?;
    let prefix_indices = base_indices[..base_indices.len() - 1].to_vec();

    let [Operand::IdRef(cond), Operand::IdRef(true_value), Operand::IdRef(false_value)] =
        select_def.operands.as_slice()
    else {
        return None;
    };
    let (advanced_ptr, advanced_on_true) = if *true_value == phi_result {
        (*false_value, false)
    } else if *false_value == phi_result {
        (*true_value, true)
    } else {
        return None;
    };
    let advanced_def = value_def.get(&advanced_ptr)?;
    if !matches!(
        advanced_def.class.opcode,
        Op::PtrAccessChain | Op::InBoundsPtrAccessChain
    ) || advanced_def.result_type != Some(ptr_ty)
    {
        return None;
    }
    let [Operand::IdRef(advanced_base), Operand::IdRef(step)] = advanced_def.operands.as_slice()
    else {
        return None;
    };
    if *advanced_base != phi_result {
        return None;
    }
    let step_ty = const_type(*step)?;
    let init_ty = const_type(init_index)?;
    let init_index = if init_ty == step_ty {
        init_index
    } else if zero_ids.contains(&init_index) {
        *zero_by_type.get(&step_ty)?
    } else {
        return None;
    };
    let index_ty = step_ty;
    let select_block = *value_block.get(&select_id)?;

    Some(SelectInductionPlan {
        block,
        phi_result,
        ptr_ty,
        base: *base,
        chain_opcode: base_def.class.opcode,
        prefix_indices,
        init_index,
        init_pred,
        index_ty,
        select_id,
        select_block,
        cond: *cond,
        advanced_on_true,
        step: *step,
        advanced_ptr,
        back_pred,
    })
}

fn is_zero_int_constant(defs: &HashMap<Word, Instruction>, inst: &Instruction) -> bool {
    if let Some(ty) = inst.result_type {
        if !defs.is_empty()
            && defs
                .get(&ty)
                .is_none_or(|ty_inst| ty_inst.class.opcode != Op::TypeInt)
        {
            return false;
        }
    } else {
        return false;
    }
    matches!(
        (inst.class.opcode, inst.operands.as_slice()),
        (Op::ConstantNull, [])
            | (Op::Constant, [Operand::LiteralBit32(0)])
            | (Op::Constant, [Operand::LiteralBit64(0)])
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::{Block, Function, ModuleHeader};

    fn inst(op: Op, ty: Option<Word>, res: Option<Word>, ops: Vec<Operand>) -> Instruction {
        Instruction::new(op, ty, res, ops)
    }

    // A uint OpPhi whose loop-back incoming is a ulong value (a dropped `trunc i64 to i32` on a wide
    // induction back-edge) is legalized by inserting an OpUConvert in the predecessor block so the phi
    // operand matches the uint result type.
    #[test]
    fn narrows_ulong_incoming_of_uint_phi_with_uconvert_in_predecessor() {
        // types: uint=1 ulong=2  consts: uint_1=10 ulong_1=11  values: add=20 phi=30
        // labels: entry=40 header=41 latch=42
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(50));
        m.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            inst(
                Op::TypeInt,
                None,
                Some(2),
                vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(10),
                vec![Operand::LiteralBit32(1)],
            ),
            inst(
                Op::Constant,
                Some(2),
                Some(11),
                vec![Operand::LiteralBit32(1)],
            ),
        ];
        let entry = Block {
            label: Some(inst(Op::Label, None, Some(40), vec![])),
            instructions: vec![inst(Op::Branch, None, None, vec![Operand::IdRef(41)])],
        };
        // header: phi(uint) [uint_1, entry], [add(ulong), latch]; branch to latch.
        let header = Block {
            label: Some(inst(Op::Label, None, Some(41), vec![])),
            instructions: vec![
                inst(
                    Op::Phi,
                    Some(1),
                    Some(30),
                    vec![
                        Operand::IdRef(10),
                        Operand::IdRef(40),
                        Operand::IdRef(20),
                        Operand::IdRef(42),
                    ],
                ),
                inst(Op::Branch, None, None, vec![Operand::IdRef(42)]),
            ],
        };
        // latch: %20 = OpIAdd ulong ulong_1 ulong_1; branch back to header.
        let latch = Block {
            label: Some(inst(Op::Label, None, Some(42), vec![])),
            instructions: vec![
                inst(
                    Op::IAdd,
                    Some(2),
                    Some(20),
                    vec![Operand::IdRef(11), Operand::IdRef(11)],
                ),
                inst(Op::Branch, None, None, vec![Operand::IdRef(41)]),
            ],
        };
        let mut func = Function::new();
        func.blocks = vec![entry, header, latch];
        m.functions = vec![func];

        assert!(rewrite_integer_width_phis(&mut m));

        // An OpUConvert to uint of the ulong value was inserted in the latch (predecessor), before its
        // terminating branch.
        let latch = &m.functions[0].blocks[2];
        let conv = latch
            .instructions
            .iter()
            .find(|i| i.class.opcode == Op::UConvert)
            .expect("uconvert inserted in predecessor");
        assert_eq!(conv.result_type, Some(1)); // to uint
        assert_eq!(conv.operands[0], Operand::IdRef(20)); // of the ulong add
        assert_eq!(latch.instructions.last().unwrap().class.opcode, Op::Branch);
        // The phi's back-edge operand now references the converted id, not the raw ulong value.
        let header = &m.functions[0].blocks[1];
        let phi = header
            .instructions
            .iter()
            .find(|i| i.class.opcode == Op::Phi)
            .unwrap();
        assert_eq!(phi.operands[2], Operand::IdRef(conv.result_id.unwrap()));
        assert_eq!(phi.operands[3], Operand::IdRef(42));
        // The forward (entry) incoming, already uint, is untouched.
        assert_eq!(phi.operands[0], Operand::IdRef(10));
    }

    // A pointer phi over two access chains into the same Function alloca, sharing the constant first
    // index (`uint_0`, a struct member) and differing on the second (an array index), is rewritten to:
    // reuse the shared constant, phi only the differing index, and rematerialize the access chain. The
    // illegal pointer phi is removed and its use is repointed.
    #[test]
    fn rewrites_struct_alloca_pointer_phi_reusing_constant_first_index() {
        // ids: uint=1 v2float=2 struct=3 ptr_struct=4 ptr_v2f=5  uint_0=10 uint_1=11
        // alloca=20  armA=30 armB=31  varidx=40  phi=50 load=51  labels 19/29/32/49
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(60));
        m.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            inst(
                Op::TypeFloat,
                None,
                Some(2),
                vec![Operand::LiteralBit32(32)],
            ),
            inst(Op::TypeStruct, None, Some(3), vec![Operand::IdRef(2)]),
            inst(
                Op::TypePointer,
                None,
                Some(4),
                vec![
                    Operand::StorageClass(StorageClass::Function),
                    Operand::IdRef(3),
                ],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(5),
                vec![
                    Operand::StorageClass(StorageClass::Function),
                    Operand::IdRef(2),
                ],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(10),
                vec![Operand::LiteralBit32(0)],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(11),
                vec![Operand::LiteralBit32(1)],
            ),
        ];
        let entry = Block {
            label: Some(inst(Op::Label, None, Some(19), vec![])),
            instructions: vec![inst(
                Op::Variable,
                Some(4),
                Some(20),
                vec![Operand::StorageClass(StorageClass::Function)],
            )],
        };
        let arm_a = Block {
            label: Some(inst(Op::Label, None, Some(29), vec![])),
            instructions: vec![inst(
                Op::InBoundsAccessChain,
                Some(5),
                Some(30),
                vec![Operand::IdRef(20), Operand::IdRef(10), Operand::IdRef(11)],
            )],
        };
        let arm_b = Block {
            label: Some(inst(Op::Label, None, Some(32), vec![])),
            instructions: vec![
                inst(Op::Undef, Some(1), Some(40), vec![]),
                inst(
                    Op::InBoundsAccessChain,
                    Some(5),
                    Some(31),
                    vec![Operand::IdRef(20), Operand::IdRef(10), Operand::IdRef(40)],
                ),
            ],
        };
        let header = Block {
            label: Some(inst(Op::Label, None, Some(49), vec![])),
            instructions: vec![
                inst(
                    Op::Phi,
                    Some(5),
                    Some(50),
                    vec![
                        Operand::IdRef(30),
                        Operand::IdRef(29),
                        Operand::IdRef(31),
                        Operand::IdRef(32),
                    ],
                ),
                inst(Op::Load, Some(2), Some(51), vec![Operand::IdRef(50)]),
            ],
        };
        let mut func = Function::new();
        func.blocks = vec![entry, arm_a, arm_b, header];
        m.functions = vec![func];

        assert!(rewrite_logical_pointer_phis(&mut m));

        let hdr = &m.functions[0].blocks[3];
        // No pointer phi remains.
        assert!(!hdr
            .instructions
            .iter()
            .any(|i| i.class.opcode == Op::Phi && i.result_type == Some(5)));
        // An integer phi for the differing (second) index was synthesized.
        let int_phi = hdr
            .instructions
            .iter()
            .find(|i| i.class.opcode == Op::Phi && i.result_type == Some(1));
        let int_phi = int_phi.expect("differing index should be phi'd");
        let int_phi_id = int_phi.result_id.unwrap();
        // The rematerialized access chain reuses the shared constant first index and the index phi.
        let chain = hdr
            .instructions
            .iter()
            .find(|i| i.class.opcode == Op::InBoundsAccessChain)
            .expect("rematerialized access chain");
        assert_eq!(chain.operands[0], Operand::IdRef(20)); // base alloca
        assert_eq!(chain.operands[1], Operand::IdRef(10)); // reused constant struct index
        assert_eq!(chain.operands[2], Operand::IdRef(int_phi_id)); // phi'd array index
                                                                   // The load now reads the rematerialized pointer, not the removed phi.
        let load = hdr
            .instructions
            .iter()
            .find(|i| i.class.opcode == Op::Load)
            .unwrap();
        assert_eq!(load.operands[0], Operand::IdRef(chain.result_id.unwrap()));
    }

    #[test]
    fn rewrites_storage_buffer_select_induction_phi_to_index_phi() {
        // `%p = phi [base, entry], [select(cond, gep(%p, step), %p), latch]` is legal with
        // VariablePointersStorageBuffer, but it can be represented without pointer SSA merges by
        // phi'ing the trailing array index and rematerializing one access chain from the root buffer.
        let uint = 1;
        let block_ty = 2;
        let ptr_block = 3;
        let ptr_uint = 4;
        let zero = 10;
        let one = 11;
        let cond = 12;
        let buffer = 20;
        let base = 21;
        let ptr_phi = 22;
        let advanced = 23;
        let selected = 24;
        let loaded = 25;
        let entry_label = 30;
        let loop_label = 31;
        let latch_label = 32;
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(40));
        m.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(uint),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            inst(
                Op::Constant,
                Some(uint),
                Some(zero),
                vec![Operand::LiteralBit32(0)],
            ),
            inst(
                Op::Constant,
                Some(uint),
                Some(one),
                vec![Operand::LiteralBit32(1)],
            ),
            inst(
                Op::TypeRuntimeArray,
                None,
                Some(9),
                vec![Operand::IdRef(uint)],
            ),
            inst(
                Op::TypeStruct,
                None,
                Some(block_ty),
                vec![Operand::IdRef(9)],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(ptr_block),
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(block_ty),
                ],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(ptr_uint),
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(uint),
                ],
            ),
            inst(
                Op::Variable,
                Some(ptr_block),
                Some(buffer),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
        ];
        let entry = Block {
            label: Some(inst(Op::Label, None, Some(entry_label), vec![])),
            instructions: vec![inst(
                Op::AccessChain,
                Some(ptr_uint),
                Some(base),
                vec![
                    Operand::IdRef(buffer),
                    Operand::IdRef(zero),
                    Operand::IdRef(zero),
                ],
            )],
        };
        let loop_block = Block {
            label: Some(inst(Op::Label, None, Some(loop_label), vec![])),
            instructions: vec![
                inst(
                    Op::Phi,
                    Some(ptr_uint),
                    Some(ptr_phi),
                    vec![
                        Operand::IdRef(base),
                        Operand::IdRef(entry_label),
                        Operand::IdRef(selected),
                        Operand::IdRef(latch_label),
                    ],
                ),
                inst(
                    Op::PtrAccessChain,
                    Some(ptr_uint),
                    Some(advanced),
                    vec![Operand::IdRef(ptr_phi), Operand::IdRef(one)],
                ),
                inst(
                    Op::Select,
                    Some(ptr_uint),
                    Some(selected),
                    vec![
                        Operand::IdRef(cond),
                        Operand::IdRef(advanced),
                        Operand::IdRef(ptr_phi),
                    ],
                ),
                inst(
                    Op::Load,
                    Some(uint),
                    Some(loaded),
                    vec![Operand::IdRef(ptr_phi)],
                ),
            ],
        };
        let latch = Block {
            label: Some(inst(Op::Label, None, Some(latch_label), vec![])),
            instructions: vec![],
        };
        let mut func = Function::new();
        func.blocks = vec![entry, loop_block, latch];
        m.functions = vec![func];

        assert!(rewrite_variable_pointer_phis(&mut m));

        let loop_block = &m.functions[0].blocks[1];
        assert!(
            !loop_block
                .instructions
                .iter()
                .any(|i| i.class.opcode == Op::Phi && i.result_type == Some(ptr_uint)),
            "pointer phi should be removed: {loop_block:?}"
        );
        assert!(
            !loop_block
                .instructions
                .iter()
                .any(|i| i.class.opcode == Op::Select && i.result_type == Some(ptr_uint)),
            "pointer select should be removed: {loop_block:?}"
        );
        let index_phi = loop_block
            .instructions
            .iter()
            .find(|i| i.class.opcode == Op::Phi && i.result_type == Some(uint))
            .expect("index phi");
        let next_index = match index_phi.operands[2] {
            Operand::IdRef(id) => id,
            ref other => panic!("backedge index should be an id ref, got {other:?}"),
        };
        assert_eq!(index_phi.operands[0], Operand::IdRef(zero));
        assert_eq!(index_phi.operands[1], Operand::IdRef(entry_label));
        assert_eq!(index_phi.operands[3], Operand::IdRef(latch_label));
        let index_select = loop_block
            .instructions
            .iter()
            .find(|i| i.result_id == Some(next_index))
            .expect("next-index select");
        assert_eq!(index_select.class.opcode, Op::Select);
        assert_eq!(index_select.result_type, Some(uint));
        let remat = loop_block
            .instructions
            .iter()
            .find(|i| {
                matches!(i.class.opcode, Op::AccessChain | Op::InBoundsAccessChain)
                    && i.result_type == Some(ptr_uint)
                    && i.result_id != Some(base)
            })
            .expect("rematerialized pointer");
        let load = loop_block
            .instructions
            .iter()
            .find(|i| i.result_id == Some(loaded))
            .expect("load");
        assert_eq!(load.operands[0], Operand::IdRef(remat.result_id.unwrap()));
    }
}
