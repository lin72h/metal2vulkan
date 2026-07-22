//! Workgroup-rooted and interface-rooted pointer materialization.

use super::*;
use crate::passes::resources::rewrites::*;

/// Threadgroup memory params are spliced to fixed-size Workgroup arrays. Access chains rooted at the
/// param are already valid after `rewrite_pointer_storage`, but a direct AIR load/store or pointer
/// merge arm means element zero and cannot legally use the array variable itself as the pointer
/// operand. Re-root those direct uses through the array path to their expected leaf type.
pub(in crate::passes) fn rewrite_workgroup_root_access(
    ctx: &mut Ctx,
    entry_idx: usize,
    var: Word,
    defs: &HashMap<Word, Instruction>,
) {
    let types = combined_type_defs(ctx, defs);
    let value_types = combined_value_types(ctx, entry_idx);
    let Some(var_ptr_ty) = value_types.get(&var).copied() else {
        return;
    };
    let Some(array_ty) = ptr_pointee(&types, var_ptr_ty) else {
        return;
    };

    let mut want: Vec<Word> = vec![];
    for blk in &ctx.module.functions[entry_idx].blocks {
        for inst in &blk.instructions {
            match inst.class.opcode {
                Op::Load if inst.operands.first() == Some(&Operand::IdRef(var)) => {
                    if let Some(t) = inst.result_type {
                        if !want.contains(&t) {
                            want.push(t);
                        }
                    }
                }
                Op::Store if inst.operands.first() == Some(&Operand::IdRef(var)) => {
                    let Some(Operand::IdRef(object)) = inst.operands.get(1) else {
                        continue;
                    };
                    let Some(t) = value_types.get(object).copied() else {
                        continue;
                    };
                    if !want.contains(&t) {
                        want.push(t);
                    }
                }
                Op::Select | Op::Phi => {
                    let uses_var = if inst.class.opcode == Op::Select {
                        inst.operands
                            .get(1..)
                            .is_some_and(|arms| arms.contains(&Operand::IdRef(var)))
                    } else {
                        inst.operands
                            .iter()
                            .step_by(2)
                            .any(|operand| operand == &Operand::IdRef(var))
                    };
                    if !uses_var {
                        continue;
                    }
                    let Some(target) = inst.result_type.and_then(|ty| ptr_pointee(&types, ty))
                    else {
                        continue;
                    };
                    if !want.contains(&target) {
                        want.push(target);
                    }
                }
                _ => {}
            }
        }
    }
    if want.is_empty() {
        return;
    }

    let u0 = ctx.const_uint(0);
    let mut leaf_chain: HashMap<Word, Word> = HashMap::new();
    let mut injected: Vec<Instruction> = vec![];
    for target in want {
        let Some(path) = path_to_leaf(&types, array_ty, target) else {
            continue;
        };
        let ptr_ty = ctx.ty_ptr(StorageClass::Workgroup, target);
        let id = ctx.module.fresh_id();
        let mut ops = vec![Operand::IdRef(var)];
        for _ in &path {
            ops.push(Operand::IdRef(u0));
        }
        injected.push(Instruction::new(
            Op::InBoundsAccessChain,
            Some(ptr_ty),
            Some(id),
            ops,
        ));
        leaf_chain.insert(target, id);
    }
    if leaf_chain.is_empty() {
        return;
    }

    for blk in &mut ctx.module.functions[entry_idx].blocks {
        for inst in &mut blk.instructions {
            match inst.class.opcode {
                Op::Load if inst.operands.first() == Some(&Operand::IdRef(var)) => {
                    if let Some(&chain) = inst.result_type.as_ref().and_then(|t| leaf_chain.get(t))
                    {
                        inst.operands[0] = Operand::IdRef(chain);
                    }
                }
                Op::Store if inst.operands.first() == Some(&Operand::IdRef(var)) => {
                    let Some(Operand::IdRef(object)) = inst.operands.get(1) else {
                        continue;
                    };
                    if let Some(&chain) = value_types.get(object).and_then(|t| leaf_chain.get(t)) {
                        inst.operands[0] = Operand::IdRef(chain);
                    }
                }
                Op::Select | Op::Phi => {
                    let Some(chain) = inst
                        .result_type
                        .and_then(|ty| ptr_pointee(&types, ty))
                        .and_then(|target| leaf_chain.get(&target))
                        .copied()
                    else {
                        continue;
                    };
                    if inst.class.opcode == Op::Select {
                        for operand in inst.operands.iter_mut().skip(1) {
                            if *operand == Operand::IdRef(var) {
                                *operand = Operand::IdRef(chain);
                            }
                        }
                    } else {
                        for operand in inst.operands.iter_mut().step_by(2) {
                            if *operand == Operand::IdRef(var) {
                                *operand = Operand::IdRef(chain);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    if let Some(first) = ctx.module.functions[entry_idx].blocks.first_mut() {
        let at = first
            .instructions
            .iter()
            .position(|i| i.class.opcode != Op::Variable)
            .unwrap_or(first.instructions.len());
        for (k, chain) in injected.into_iter().enumerate() {
            first.instructions.insert(at + k, chain);
        }
    }
}

/// Rewrite the uses of a buffer param the backend collapsed into a bare element pointer. `block_ty`
/// is the StorageBuffer Block we point the var at (a `{ RuntimeArray<T> }` for a genuine array, or the
/// reconstructed struct). Each access chain rooted at the param is re-rooted at the var (with a
/// member-0 index prepended iff `prepend_member0`, for the runtime-array case); each direct OpLoad of
/// the param — the offset-0 leaf — is routed through an access chain to that leaf.
pub(in crate::passes) fn rewrite_collapsed_buffer(
    ctx: &mut Ctx,
    entry_idx: usize,
    pid: Word,
    var: Word,
    block_ty: Word,
    prepend_member0: bool,
    defs: &HashMap<Word, Instruction>,
) {
    let u0 = ctx.const_uint(0);

    // A combined type map (original defs + the types we synthesized) for leaf-path descent.
    let mut types = defs.clone();
    for g in &ctx.new_globals {
        if let Some(id) = g.result_id {
            types.entry(id).or_insert_with(|| g.clone());
        }
    }

    let value_types = combined_value_types(ctx, entry_idx);
    let desired_pointees = pointer_leaf_use_types(ctx, entry_idx, &value_types);

    // 1) Access chains based directly at the param: re-root at the buffer var; prepend member-0 only
    //    for the runtime-array wrapping (the original first index then indexes the runtime array).
    //    Some AIR record-0 member paths arrive as a single padded LLVM struct index even though the
    //    chosen wrapper is `{ RuntimeArray<compact-metadata-struct> }`; when the chain's user proves the
    //    intended compact member, route it through record 0 before falling back to plain array indexing.
    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let n_inst = ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .len();
        for ii in 0..n_inst {
            let (old_indices, result_type, result_id) = {
                let inst = &ctx.module.functions[entry_idx].blocks[bi].instructions[ii];
                if matches!(
                    inst.class.opcode,
                    Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
                ) && inst.operands.first() == Some(&Operand::IdRef(pid))
                {
                    (
                        inst.operands[1..].to_vec(),
                        inst.result_type,
                        inst.result_id,
                    )
                } else {
                    (vec![], None, None)
                }
            };
            if old_indices.is_empty() {
                continue;
            }

            let direct_root = if prepend_member0 {
                runtime_array_block_element_type(&types, block_ty)
            } else {
                Some(block_ty)
            };
            let remapped = direct_root.and_then(|root_ty| {
                remap_collapsed_direct_air_struct_access(
                    ctx,
                    &types,
                    root_ty,
                    &old_indices,
                    result_type,
                )
                .or_else(|| {
                    result_id
                        .and_then(|rid| desired_pointees.get(&rid).and_then(|ty| *ty))
                        .and_then(|desired_pointee| {
                            remap_direct_air_struct_access_to_pointee(
                                ctx,
                                &types,
                                root_ty,
                                &old_indices,
                                desired_pointee,
                            )
                        })
                })
            });

            if let Some((indices, pointee)) = remapped {
                let mut operands = vec![Operand::IdRef(var)];
                if prepend_member0 {
                    operands.push(Operand::IdRef(u0));
                    operands.push(Operand::IdRef(u0));
                }
                operands.extend(indices);
                let ptr_ty = ctx.ty_ptr(StorageClass::StorageBuffer, pointee);
                let inst = &mut ctx.module.functions[entry_idx].blocks[bi].instructions[ii];
                inst.operands = operands;
                inst.result_type = Some(ptr_ty);
                continue;
            }

            let inst = &mut ctx.module.functions[entry_idx].blocks[bi].instructions[ii];
            inst.operands[0] = Operand::IdRef(var);
            if prepend_member0 {
                inst.operands.insert(1, Operand::IdRef(u0));
            }
        }
    }

    // 2) Direct loads/stores of the param (offset-0 object): route each through an access chain to
    //    the exact aggregate or leaf type used by the instruction, one chain per distinct type.
    let mut want: Vec<Word> = vec![]; // distinct direct object types needing a chain
    for blk in &ctx.module.functions[entry_idx].blocks {
        for inst in &blk.instructions {
            match inst.class.opcode {
                Op::Load if inst.operands.first() == Some(&Operand::IdRef(pid)) => {
                    if let Some(t) = inst.result_type {
                        if !want.contains(&t) {
                            want.push(t);
                        }
                    }
                }
                Op::Store if inst.operands.first() == Some(&Operand::IdRef(pid)) => {
                    let Some(Operand::IdRef(object)) = inst.operands.get(1) else {
                        continue;
                    };
                    if let Some(t) = value_types.get(object).copied() {
                        if !want.contains(&t) {
                            want.push(t);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let mut leaf_chain: HashMap<Word, Word> = HashMap::new();
    let mut injected: Vec<Instruction> = vec![];
    for t in want {
        let Some(path) = path_to_leaf(&types, block_ty, t) else {
            continue;
        };
        let ptr_ty = ctx.ty_ptr(StorageClass::StorageBuffer, t);
        let id = ctx.module.fresh_id();
        let mut ops = vec![Operand::IdRef(var)];
        for _ in &path {
            ops.push(Operand::IdRef(u0));
        }
        injected.push(Instruction::new(
            Op::AccessChain,
            Some(ptr_ty),
            Some(id),
            ops,
        ));
        leaf_chain.insert(t, id);
    }
    // Point each direct load/store at its exact-type chain.
    for blk in &mut ctx.module.functions[entry_idx].blocks {
        for inst in &mut blk.instructions {
            match inst.class.opcode {
                Op::Load if inst.operands.first() == Some(&Operand::IdRef(pid)) => {
                    if let Some(&c) = inst.result_type.as_ref().and_then(|t| leaf_chain.get(t)) {
                        inst.operands[0] = Operand::IdRef(c);
                    }
                }
                Op::Store if inst.operands.first() == Some(&Operand::IdRef(pid)) => {
                    let Some(Operand::IdRef(object)) = inst.operands.get(1) else {
                        continue;
                    };
                    if let Some(&c) = value_types.get(object).and_then(|t| leaf_chain.get(t)) {
                        inst.operands[0] = Operand::IdRef(c);
                    }
                }
                _ => {}
            }
        }
    }
    if let Some(first) = ctx.module.functions[entry_idx].blocks.first_mut() {
        for (k, ld) in injected.into_iter().enumerate() {
            first.instructions.insert(k, ld);
        }
    }

    // 3) Defensive: any remaining direct use of the param (pass-through) routes to the first
    //    scalar/vector leaf at offset 0.
    let still_used = ctx.module.functions[entry_idx].blocks.iter().any(|b| {
        b.instructions.iter().any(|i| {
            i.operands
                .iter()
                .any(|o| matches!(o, Operand::IdRef(r) if *r == pid))
        })
    });
    if still_used {
        // Descend to the first non-aggregate leaf.
        let mut cur = block_ty;
        let mut path = vec![];
        while let Some(next) = types.get(&cur).and_then(member0_type) {
            path.push(0u32);
            cur = next;
        }
        let ptr_ty = ctx.ty_ptr(StorageClass::StorageBuffer, cur);
        let id = ctx.module.fresh_id();
        let mut ops = vec![Operand::IdRef(var)];
        for _ in &path {
            ops.push(Operand::IdRef(u0));
        }
        if let Some(first) = ctx.module.functions[entry_idx].blocks.first_mut() {
            first.instructions.insert(
                0,
                Instruction::new(Op::AccessChain, Some(ptr_ty), Some(id), ops),
            );
        }
        replace_id_in_function(&mut ctx.module.functions[entry_idx], pid, id);
    }
}

/// After a pointer param has been spliced to a real variable (a StorageBuffer buffer block, or a
/// Private zero var for an unmodeled pointer), every access-chain off it still produces a
/// `UniformConstant` element pointer (the type the backend chose). Vulkan requires the element
/// pointer's storage class to match the root variable's. Walk the entry block, find access chains
/// and pointer aliases transitively rooted at one of `root_vars`, and rewrite their result-type
/// pointer to `target_sc` (creating the pointer type if needed). OpLoad/OpStore through those
/// pointers are unaffected by class. Used both for buffers (-> StorageBuffer) and unmodeled pointer
/// params (-> Private).
pub(in crate::passes) fn rewrite_pointer_storage(
    ctx: &mut Ctx,
    entry_idx: usize,
    root_vars: &[Word],
    target_sc: StorageClass,
    defs: &HashMap<Word, Instruction>,
) -> Result<(), String> {
    let mut roots: HashSet<Word> = root_vars.iter().copied().collect();
    let direct_roots: HashSet<Word> = root_vars.iter().copied().collect();
    // iterate to a fixpoint over access chains so chained derefs are caught.
    let mut changed = true;
    let mut new_ptr_types: HashMap<(Word, Word), Word> = HashMap::new(); // (old ptr ty, pointee) -> new ptr ty
    let mut value_types = combined_value_types(ctx, entry_idx);
    let mut value_defs = combined_value_defs(ctx, entry_idx);
    let types = combined_type_defs(ctx, defs);
    let mut scaled_vector_strides = HashSet::new();
    let mut vector_stride_plans = vec![];

    while changed {
        changed = false;
        let n_blocks = ctx.module.functions[entry_idx].blocks.len();
        for bi in 0..n_blocks {
            let n_inst = ctx.module.functions[entry_idx].blocks[bi]
                .instructions
                .len();
            for ii in 0..n_inst {
                let (op, rty, rid, base, operands) = {
                    let inst = &ctx.module.functions[entry_idx].blocks[bi].instructions[ii];
                    let base = inst.operands.first().and_then(|o| match o {
                        Operand::IdRef(b) => Some(*b),
                        _ => None,
                    });
                    (
                        inst.class.opcode,
                        inst.result_type,
                        inst.result_id,
                        base,
                        inst.operands.clone(),
                    )
                };
                let rooted = if matches!(
                    op,
                    Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
                ) {
                    base.map(|id| roots.contains(&id)).unwrap_or(false)
                } else if matches!(op, Op::CopyObject)
                    || (op == Op::Bitcast
                        && rty
                            .and_then(|ty| pointer_pointee_including_new(ctx, &types, ty))
                            .is_some())
                {
                    base.map(|id| roots.contains(&id)).unwrap_or(false)
                } else if matches!(op, Op::Phi) {
                    ctx.module.functions[entry_idx].blocks[bi].instructions[ii]
                        .operands
                        .iter()
                        .step_by(2)
                        .any(|operand| matches!(operand, Operand::IdRef(id) if roots.contains(id)))
                } else if matches!(op, Op::Select) {
                    ctx.module.functions[entry_idx].blocks[bi].instructions[ii]
                        .operands
                        .iter()
                        .skip(1)
                        .any(|operand| matches!(operand, Operand::IdRef(id) if roots.contains(id)))
                } else {
                    false
                };
                if !rooted {
                    continue;
                }
                // This pointer result is buffer-derived; mark it as a root and rewrite its type.
                if let Some(rid) = rid {
                    if roots.insert(rid) {
                        changed = true;
                    }
                }
                if let Some(old_ty) = rty {
                    let pointee = if let Some(pointee) = rewritten_rooted_pointer_pointee(
                        ctx,
                        &types,
                        &value_types,
                        &roots,
                        op,
                        base,
                        &operands,
                    ) {
                        pointee
                    } else {
                        let old_pointee = pointer_pointee_including_new(ctx, defs, old_ty)
                            .ok_or_else(|| {
                                format!("buffer access-chain result type {old_ty} not a pointer")
                            })?;
                        canonical_rooted_access_pointee(
                            ctx,
                            &types,
                            &value_types,
                            &roots,
                            op,
                            base,
                            old_pointee,
                        )
                    };
                    if target_sc == StorageClass::StorageBuffer {
                        if let (Some(rid), Some((index, index_ty, lanes))) = (
                            rid,
                            rooted_vector_stride_plan(
                                ctx,
                                &types,
                                &value_types,
                                op,
                                old_ty,
                                pointee,
                                &operands,
                            ),
                        ) {
                            if scaled_vector_strides.insert(rid) {
                                vector_stride_plans.push((rid, index, index_ty, lanes));
                            }
                        }
                    }
                    let new_ty = if let Some(&t) = new_ptr_types.get(&(old_ty, pointee)) {
                        t
                    } else {
                        let t = ctx.ty_ptr(target_sc, pointee);
                        new_ptr_types.insert((old_ty, pointee), t);
                        t
                    };
                    if let Some(rid) = rid {
                        value_types.insert(rid, new_ty);
                    }
                    if matches!(op, Op::Phi | Op::Select) {
                        let replacements = rewritten_pointer_constant_operands(
                            ctx,
                            &mut value_types,
                            &mut value_defs,
                            op,
                            &operands,
                            new_ty,
                        );
                        if !replacements.is_empty() {
                            let inst =
                                &mut ctx.module.functions[entry_idx].blocks[bi].instructions[ii];
                            for (operand_idx, replacement) in replacements {
                                inst.operands[operand_idx] = Operand::IdRef(replacement);
                            }
                        }
                    }
                    ctx.module.functions[entry_idx].blocks[bi].instructions[ii].result_type =
                        Some(new_ty);
                }
                if target_sc == StorageClass::Workgroup
                    && op == Op::PtrAccessChain
                    && base.is_some_and(|id| direct_roots.contains(&id))
                {
                    let inst = ctx.module.functions[entry_idx].blocks[bi].instructions[ii].clone();
                    ctx.module.functions[entry_idx].blocks[bi].instructions[ii] = Instruction::new(
                        Op::InBoundsAccessChain,
                        inst.result_type,
                        inst.result_id,
                        inst.operands,
                    );
                }
            }
        }
    }

    // A vector-source GEP can be rooted at a scalar lane after interface buffer reconstruction:
    // `gep <N x E>, ptr %lane, %record` then needs `%record * N` in E units. The old result pointer
    // type is the proof that the PtrAccessChain element operand carried a vector stride; after the
    // rooted rewrite changes its pointee to E, preserve that byte offset explicitly. The later
    // subword-store pass decomposes the mismatched vector store into N scalar stores from this point.
    for (result, index, index_ty, lanes) in vector_stride_plans {
        let factor = ctx.const_int_of(index_ty, i64::from(lanes));
        let scaled = ctx.module.fresh_id();
        let scale = Instruction::new(
            Op::IMul,
            Some(index_ty),
            Some(scaled),
            vec![Operand::IdRef(index), Operand::IdRef(factor)],
        );
        for block in &mut ctx.module.functions[entry_idx].blocks {
            let Some(pos) = block
                .instructions
                .iter()
                .position(|inst| inst.result_id == Some(result))
            else {
                continue;
            };
            block.instructions[pos].operands[1] = Operand::IdRef(scaled);
            block.instructions.insert(pos, scale);
            break;
        }
    }
    Ok(())
}

pub(in crate::passes) fn rooted_vector_stride_plan(
    ctx: &Ctx,
    types: &HashMap<Word, Instruction>,
    value_types: &HashMap<Word, Word>,
    op: Op,
    old_ptr_ty: Word,
    new_pointee: Word,
    operands: &[Operand],
) -> Option<(Word, Word, u32)> {
    if op != Op::PtrAccessChain || operands.len() != 2 {
        return None;
    }
    let old_pointee = pointer_pointee_including_new(ctx, types, old_ptr_ty)?;
    let vector = types.get(&old_pointee)?;
    if vector.class.opcode != Op::TypeVector {
        return None;
    }
    let (Some(Operand::IdRef(element)), Some(Operand::LiteralBit32(lanes))) =
        (vector.operands.first(), vector.operands.get(1))
    else {
        return None;
    };
    if *lanes <= 1
        || (*element != new_pointee && !types_structurally_match(ctx, types, *element, new_pointee))
    {
        return None;
    }
    let Operand::IdRef(index) = operands[1] else {
        return None;
    };
    let index_ty = *value_types.get(&index)?;
    (types.get(&index_ty)?.class.opcode == Op::TypeInt).then_some((index, index_ty, *lanes))
}

pub(in crate::passes) fn rewritten_rooted_pointer_pointee(
    ctx: &Ctx,
    types: &HashMap<Word, Instruction>,
    value_types: &HashMap<Word, Word>,
    roots: &HashSet<Word>,
    op: Op,
    base: Option<Word>,
    operands: &[Operand],
) -> Option<Word> {
    if matches!(
        op,
        Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
    ) {
        let base_ptr_ty = value_types.get(&base?)?;
        let base_pointee = pointer_pointee_including_new(ctx, types, *base_ptr_ty)?;
        let access_operands = if op == Op::PtrAccessChain {
            operands.get(2..)?
        } else {
            operands.get(1..)?
        };
        return type_after_spirv_access_operands(types, base_pointee, access_operands);
    }
    if op == Op::CopyObject {
        let source_ty = value_types.get(&base?)?;
        return pointer_pointee_including_new(ctx, types, *source_ty);
    }
    if op == Op::Phi {
        return rooted_operand_pointer_pointee(ctx, types, value_types, roots, operands, 0, 2);
    }
    if op == Op::Select {
        return rooted_operand_pointer_pointee(ctx, types, value_types, roots, operands, 1, 1);
    }
    None
}

pub(in crate::passes) fn rooted_operand_pointer_pointee(
    ctx: &Ctx,
    types: &HashMap<Word, Instruction>,
    value_types: &HashMap<Word, Word>,
    roots: &HashSet<Word>,
    operands: &[Operand],
    skip: usize,
    step: usize,
) -> Option<Word> {
    let mut fallback = None;
    for operand in operands.iter().skip(skip).step_by(step) {
        let Operand::IdRef(id) = operand else {
            continue;
        };
        let Some(ptr_ty) = value_types.get(id) else {
            continue;
        };
        let Some(pointee) = pointer_pointee_including_new(ctx, types, *ptr_ty) else {
            continue;
        };
        if roots.contains(id) {
            return Some(pointee);
        }
        fallback.get_or_insert(pointee);
    }
    fallback
}
