//! Raw-word and raw-byte resource access rewrites.

use super::*;

#[derive(Clone, Debug)]
pub(in crate::passes) struct BufferAccessPath {
    pub(in crate::passes) root: Word,
    pub(in crate::passes) indices: Vec<Word>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::passes) enum RawStoreObject {
    Float32,
    Uint32,
}

pub(in crate::passes) fn rewrite_raw_word_alias_chains(
    ctx: &mut Ctx,
    entry_idx: usize,
    buffer_structs: &[(Word, Word)],
    defs: &HashMap<Word, Instruction>,
) -> Result<(), String> {
    let buffer_types: HashMap<Word, Word> = buffer_structs.iter().copied().collect();
    if buffer_types.is_empty() {
        return Ok(());
    }

    let mut raw_alias_vars: HashMap<Word, Word> = HashMap::new();
    let mut paths: HashMap<Word, BufferAccessPath> = HashMap::new();
    let mut types = combined_type_defs(ctx, defs);
    let mut value_types = combined_value_types(ctx, entry_idx);

    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let n_insts = ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .len();
        for ii in 0..n_insts {
            let snapshot = ctx.module.functions[entry_idx].blocks[bi].instructions[ii].clone();
            let Some(result) = snapshot.result_id else {
                continue;
            };
            if matches!(snapshot.class.opcode, Op::CopyObject | Op::Bitcast) {
                let Some(Operand::IdRef(source)) = snapshot.operands.first() else {
                    continue;
                };
                if let Some(path) = paths.get(source).cloned() {
                    if ptr_pointee(&types, snapshot.result_type.unwrap_or(0)).is_some() {
                        paths.insert(result, path);
                    }
                }
                continue;
            }
            if !matches!(
                snapshot.class.opcode,
                Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
            ) {
                continue;
            }
            let Some(Operand::IdRef(base)) = snapshot.operands.first() else {
                continue;
            };
            if ptr_pointee(&types, snapshot.result_type.unwrap_or(0)).is_none() {
                continue;
            }
            let indices = snapshot.operands[1..]
                .iter()
                .filter_map(|operand| match operand {
                    Operand::IdRef(id) => Some(*id),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if indices.len() != snapshot.operands.len().saturating_sub(1) {
                continue;
            }
            if buffer_types.contains_key(base) {
                paths.insert(
                    result,
                    BufferAccessPath {
                        root: *base,
                        indices,
                    },
                );
            } else if let Some(base_path) = paths.get(base).cloned() {
                let mut combined = base_path.indices;
                combined.extend(indices);
                paths.insert(
                    result,
                    BufferAccessPath {
                        root: base_path.root,
                        indices: combined,
                    },
                );
            }
        }
    }

    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let mut ii = 0;
        while ii
            < ctx.module.functions[entry_idx].blocks[bi]
                .instructions
                .len()
        {
            let snapshot = ctx.module.functions[entry_idx].blocks[bi].instructions[ii].clone();

            if matches!(
                snapshot.class.opcode,
                Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
            ) {
                if let Some(Operand::IdRef(base)) = snapshot.operands.first() {
                    let result_pointee = snapshot
                        .result_type
                        .and_then(|result_type| ptr_pointee(&types, result_type));
                    if buffer_types.contains_key(base)
                        && result_pointee.is_some_and(|ty| is_uint_type(&types, ty))
                    {
                        if buffer_types
                            .get(base)
                            .is_some_and(|ty| is_raw_uint_runtime_block(&types, *ty))
                        {
                            if let Some(word) =
                                direct_raw_root_word_index(&types, &snapshot.operands[1..])
                            {
                                let Some(binding) = descriptor_binding(&ctx.module, *base) else {
                                    ii += 1;
                                    continue;
                                };
                                let raw_var = match raw_alias_vars.get(base).copied() {
                                    Some(var) => var,
                                    None => {
                                        let var =
                                            create_raw_alias_buffer(ctx, binding, defs, &mut types);
                                        raw_alias_vars.insert(*base, var);
                                        var
                                    }
                                };
                                let ptr_uint = ctx.ty_ptr(
                                    StorageClass::StorageBuffer,
                                    result_pointee
                                        .ok_or("raw word alias: missing result pointee")?,
                                );
                                let zero = ctx.const_uint(0);
                                let inst = &mut ctx.module.functions[entry_idx].blocks[bi]
                                    .instructions[ii];
                                inst.result_type = Some(ptr_uint);
                                inst.operands = vec![
                                    Operand::IdRef(raw_var),
                                    Operand::IdRef(zero),
                                    Operand::IdRef(word),
                                ];
                                ii += 1;
                                continue;
                            }
                        }
                        let index_ids = snapshot.operands[1..]
                            .iter()
                            .map(|operand| match operand {
                                Operand::IdRef(id) => Some(*id),
                                _ => None,
                            })
                            .collect::<Option<Vec<_>>>();
                        if let (Some(indices), Some(block_ty)) =
                            (index_ids, buffer_types.get(base).copied())
                        {
                            if let Some((byte_offset, leaf_ty)) =
                                access_path_byte_offset_and_leaf_type(&types, block_ty, &indices)
                            {
                                if byte_offset % 4 == 0
                                    && result_pointee.is_some_and(|pointee| pointee != leaf_ty)
                                {
                                    let Some(binding) = descriptor_binding(&ctx.module, *base)
                                    else {
                                        ii += 1;
                                        continue;
                                    };
                                    let raw_var = match raw_alias_vars.get(base).copied() {
                                        Some(var) => var,
                                        None => {
                                            let var = create_raw_alias_buffer(
                                                ctx, binding, defs, &mut types,
                                            );
                                            raw_alias_vars.insert(*base, var);
                                            var
                                        }
                                    };
                                    let ptr_uint = ctx.ty_ptr(
                                        StorageClass::StorageBuffer,
                                        result_pointee
                                            .ok_or("raw word alias: missing result pointee")?,
                                    );
                                    let zero = ctx.const_uint(0);
                                    let word = ctx.const_uint(byte_offset / 4);
                                    let inst = &mut ctx.module.functions[entry_idx].blocks[bi]
                                        .instructions[ii];
                                    inst.result_type = Some(ptr_uint);
                                    inst.operands = vec![
                                        Operand::IdRef(raw_var),
                                        Operand::IdRef(zero),
                                        Operand::IdRef(word),
                                    ];
                                    ii += 1;
                                    continue;
                                }
                            }
                        }
                    }
                    if let Some(base_path) = paths.get(base).cloned() {
                        if result_pointee.is_some_and(|ty| is_uint_type(&types, ty)) {
                            if let Some(local_word) =
                                raw_word_chain_offset(&types, &snapshot.operands[1..])
                            {
                                if let Some(block_ty) = buffer_types.get(&base_path.root).copied() {
                                    if let Some(base_bytes) = access_path_byte_offset(
                                        &types,
                                        block_ty,
                                        &base_path.indices,
                                    ) {
                                        let combined_indices = snapshot.operands[1..]
                                            .iter()
                                            .map(|operand| match operand {
                                                Operand::IdRef(id) => Some(*id),
                                                _ => None,
                                            })
                                            .collect::<Option<Vec<_>>>()
                                            .map(|mut indices| {
                                                let mut combined = base_path.indices.clone();
                                                combined.append(&mut indices);
                                                combined
                                            });
                                        if let (Some(combined_indices), Some(pointee)) =
                                            (combined_indices, result_pointee)
                                        {
                                            if access_path_byte_offset_and_leaf_type(
                                                &types,
                                                block_ty,
                                                &combined_indices,
                                            )
                                            .is_some_and(|(_, leaf_ty)| leaf_ty == pointee)
                                            {
                                                ii += 1;
                                                continue;
                                            }
                                        }
                                        if base_bytes % 4 == 0 {
                                            let Some(binding) =
                                                descriptor_binding(&ctx.module, base_path.root)
                                            else {
                                                ii += 1;
                                                continue;
                                            };
                                            let raw_var = match raw_alias_vars
                                                .get(&base_path.root)
                                                .copied()
                                            {
                                                Some(var) => var,
                                                None => {
                                                    let var = create_raw_alias_buffer(
                                                        ctx, binding, defs, &mut types,
                                                    );
                                                    raw_alias_vars.insert(base_path.root, var);
                                                    var
                                                }
                                            };
                                            let word = ctx.const_uint(base_bytes / 4 + local_word);
                                            let ptr_uint = ctx.ty_ptr(
                                                StorageClass::StorageBuffer,
                                                result_pointee.ok_or(
                                                    "raw word alias: missing result pointee",
                                                )?,
                                            );
                                            let zero = ctx.const_uint(0);
                                            let inst = &mut ctx.module.functions[entry_idx].blocks
                                                [bi]
                                                .instructions[ii];
                                            inst.result_type = Some(ptr_uint);
                                            inst.operands = vec![
                                                Operand::IdRef(raw_var),
                                                Operand::IdRef(zero),
                                                Operand::IdRef(word),
                                            ];
                                            ii += 1;
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if snapshot.class.opcode == Op::Load {
                if let Some(rewrite) = plan_mismatched_raw_word_load_rewrite(
                    ctx,
                    &types,
                    &mut value_types,
                    &mut raw_alias_vars,
                    defs,
                    &buffer_types,
                    &paths,
                    &snapshot,
                ) {
                    let inserted = rewrite.prefix.len();
                    let insts = &mut ctx.module.functions[entry_idx].blocks[bi].instructions;
                    for (offset, inst) in rewrite.prefix.into_iter().enumerate() {
                        insts.insert(ii + offset, inst);
                    }
                    insts[ii + inserted] = rewrite.replacement;
                    ii += inserted + 1;
                    continue;
                }
            }
            if snapshot.class.opcode == Op::Store {
                if let Some(rewrite) = plan_raw_byte_store_rewrite(
                    ctx,
                    &types,
                    &mut value_types,
                    &mut raw_alias_vars,
                    defs,
                    &buffer_types,
                    &paths,
                    &snapshot,
                ) {
                    let inserted = rewrite.prefix.len();
                    let insts = &mut ctx.module.functions[entry_idx].blocks[bi].instructions;
                    for (offset, inst) in rewrite.prefix.into_iter().enumerate() {
                        insts.insert(ii + offset, inst);
                    }
                    insts[ii + inserted] = rewrite.replacement;
                    ii += inserted + 1;
                    continue;
                }
            }
            if let Some(rewrite) = plan_raw_byte_atomic_rewrite(
                ctx,
                &types,
                &mut value_types,
                &mut raw_alias_vars,
                defs,
                &buffer_types,
                &paths,
                &snapshot,
            ) {
                let inserted = rewrite.prefix.len();
                let insts = &mut ctx.module.functions[entry_idx].blocks[bi].instructions;
                for (offset, inst) in rewrite.prefix.into_iter().enumerate() {
                    insts.insert(ii + offset, inst);
                }
                insts[ii + inserted].operands[0] = Operand::IdRef(rewrite.ptr);
                ii += inserted + 1;
                continue;
            }
            ii += 1;
        }
    }

    let used = function_used_ids(&ctx.module.functions[entry_idx]);
    for block in &mut ctx.module.functions[entry_idx].blocks {
        block.instructions.retain(|inst| {
            if inst.class.opcode != Op::Bitcast {
                return true;
            }
            let Some(result) = inst.result_id else {
                return true;
            };
            if used.contains(&result) || !paths.contains_key(&result) {
                return true;
            }
            inst.result_type
                .and_then(|ty| ptr_pointee(&types, ty))
                .is_none()
        });
    }
    Ok(())
}

pub(in crate::passes) struct RawPointerRewrite {
    pub(in crate::passes) prefix: Vec<Instruction>,
    pub(in crate::passes) ptr: Word,
}

pub(in crate::passes) struct RawStoreRewrite {
    pub(in crate::passes) prefix: Vec<Instruction>,
    pub(in crate::passes) replacement: Instruction,
}

pub(in crate::passes) struct RawLoadRewrite {
    pub(in crate::passes) prefix: Vec<Instruction>,
    pub(in crate::passes) replacement: Instruction,
}

pub(in crate::passes) fn plan_mismatched_raw_word_load_rewrite(
    ctx: &mut Ctx,
    types: &HashMap<Word, Instruction>,
    value_types: &mut HashMap<Word, Word>,
    raw_alias_vars: &mut HashMap<Word, Word>,
    defs: &HashMap<Word, Instruction>,
    buffer_types: &HashMap<Word, Word>,
    paths: &HashMap<Word, BufferAccessPath>,
    load: &Instruction,
) -> Option<RawLoadRewrite> {
    let result = load.result_id?;
    let result_ty = load.result_type?;
    let result_kind = raw_store_object_kind(types, result_ty)?;
    let [Operand::IdRef(ptr), ..] = load.operands.as_slice() else {
        return None;
    };
    let ptr_ty = value_types.get(ptr).copied()?;
    let ptr_pointee = ptr_pointee(types, ptr_ty)?;
    if raw_store_object_kind(types, ptr_pointee) == Some(result_kind) {
        if let Some(rewrite) = plan_aligned_structured_leaf_load_rewrite(
            ctx,
            types,
            value_types,
            buffer_types,
            paths,
            *ptr,
            load,
            result,
            result_ty,
            result_kind,
        ) {
            return Some(rewrite);
        }
        return plan_structured_raw_word_load_rewrite(
            ctx,
            types,
            value_types,
            raw_alias_vars,
            defs,
            buffer_types,
            paths,
            *ptr,
            result,
            result_ty,
            result_kind,
        );
    }
    if let Some(rewrite) = plan_aligned_structured_leaf_load_rewrite(
        ctx,
        types,
        value_types,
        buffer_types,
        paths,
        *ptr,
        load,
        result,
        result_ty,
        result_kind,
    ) {
        return Some(rewrite);
    }
    if let Some(rewrite) = plan_structured_raw_word_load_rewrite(
        ctx,
        types,
        value_types,
        raw_alias_vars,
        defs,
        buffer_types,
        paths,
        *ptr,
        result,
        result_ty,
        result_kind,
    ) {
        return Some(rewrite);
    }
    let pointer = plan_raw_word_pointer_rewrite(
        ctx,
        types,
        value_types,
        raw_alias_vars,
        defs,
        buffer_types,
        paths,
        *ptr,
    )
    .or_else(|| {
        plan_structured_raw_word_pointer_rewrite(
            ctx,
            types,
            value_types,
            raw_alias_vars,
            defs,
            buffer_types,
            paths,
            *ptr,
        )
    })?;

    match result_kind {
        RawStoreObject::Uint32 => {
            let mut replacement = load.clone();
            replacement.operands = vec![Operand::IdRef(pointer.ptr)];
            Some(RawLoadRewrite {
                prefix: pointer.prefix,
                replacement,
            })
        }
        RawStoreObject::Float32 => {
            let uint = ctx.ty_uint();
            let raw = ctx.module.fresh_id();
            let mut prefix = pointer.prefix;
            prefix.push(Instruction::new(
                Op::Load,
                Some(uint),
                Some(raw),
                vec![Operand::IdRef(pointer.ptr)],
            ));
            value_types.insert(raw, uint);
            Some(RawLoadRewrite {
                prefix,
                replacement: Instruction::new(
                    Op::Bitcast,
                    Some(result_ty),
                    Some(result),
                    vec![Operand::IdRef(raw)],
                ),
            })
        }
    }
}

pub(in crate::passes) fn plan_raw_byte_store_rewrite(
    ctx: &mut Ctx,
    types: &HashMap<Word, Instruction>,
    value_types: &mut HashMap<Word, Word>,
    raw_alias_vars: &mut HashMap<Word, Word>,
    defs: &HashMap<Word, Instruction>,
    buffer_types: &HashMap<Word, Word>,
    paths: &HashMap<Word, BufferAccessPath>,
    store: &Instruction,
) -> Option<RawStoreRewrite> {
    let [Operand::IdRef(ptr), Operand::IdRef(object), ..] = store.operands.as_slice() else {
        return None;
    };
    let object_ty = value_types.get(object).copied()?;
    let object_kind = raw_store_object_kind(types, object_ty)?;
    let ptr_ty = value_types.get(ptr).copied()?;
    let ptr_pointee = ptr_pointee(types, ptr_ty)?;
    if raw_store_object_kind(types, ptr_pointee) == Some(object_kind) {
        return None;
    }

    let mut prelude = Vec::new();
    let object = match object_kind {
        RawStoreObject::Uint32 => *object,
        RawStoreObject::Float32 => {
            let converted = ctx.module.fresh_id();
            prelude.push(Instruction::new(
                Op::Bitcast,
                Some(ctx.ty_uint()),
                Some(converted),
                vec![Operand::IdRef(*object)],
            ));
            value_types.insert(converted, ctx.ty_uint());
            converted
        }
    };

    if let Some(mut rewrite) = plan_structured_raw_word_store_rewrite(
        ctx,
        types,
        value_types,
        raw_alias_vars,
        defs,
        buffer_types,
        paths,
        *ptr,
        object,
    ) {
        prelude.append(&mut rewrite.prefix);
        return Some(RawStoreRewrite {
            prefix: prelude,
            replacement: rewrite.replacement,
        });
    }

    let pointer = plan_raw_word_pointer_rewrite(
        ctx,
        types,
        value_types,
        raw_alias_vars,
        defs,
        buffer_types,
        paths,
        *ptr,
    )
    .or_else(|| {
        plan_structured_raw_word_pointer_rewrite(
            ctx,
            types,
            value_types,
            raw_alias_vars,
            defs,
            buffer_types,
            paths,
            *ptr,
        )
    })?;
    prelude.extend(pointer.prefix);

    let mut replacement = store.clone();
    replacement.operands = vec![Operand::IdRef(pointer.ptr), Operand::IdRef(object)];

    Some(RawStoreRewrite {
        prefix: prelude,
        replacement,
    })
}

fn plan_aligned_structured_leaf_load_rewrite(
    ctx: &mut Ctx,
    types: &HashMap<Word, Instruction>,
    value_types: &mut HashMap<Word, Word>,
    buffer_types: &HashMap<Word, Word>,
    paths: &HashMap<Word, BufferAccessPath>,
    ptr: Word,
    load: &Instruction,
    result: Word,
    result_ty: Word,
    result_kind: RawStoreObject,
) -> Option<RawLoadRewrite> {
    let path = paths.get(&ptr)?;
    let block_ty = buffer_types.get(&path.root).copied()?;
    let (byte_offset, leaf_ty) =
        access_path_byte_offset_and_leaf_type(types, block_ty, &path.indices)?;
    if byte_offset % 4 != 0 {
        return None;
    }
    if raw_store_object_kind(types, leaf_ty) == Some(result_kind) {
        return None;
    }

    if let Some(rewrite) = plan_structured_leaf_load(
        ctx,
        types,
        value_types,
        path,
        leaf_ty,
        load,
        result,
        result_ty,
        result_ty,
        false,
    ) {
        return Some(rewrite);
    }

    if result_kind == RawStoreObject::Float32 {
        let uint = ctx.ty_uint();
        if uint != result_ty {
            return plan_structured_leaf_load(
                ctx,
                types,
                value_types,
                path,
                leaf_ty,
                load,
                result,
                result_ty,
                uint,
                true,
            );
        }
    }

    None
}

#[allow(clippy::too_many_arguments)]
fn plan_structured_leaf_load(
    ctx: &mut Ctx,
    types: &HashMap<Word, Instruction>,
    value_types: &mut HashMap<Word, Word>,
    path: &BufferAccessPath,
    leaf_ty: Word,
    load: &Instruction,
    result: Word,
    result_ty: Word,
    load_ty: Word,
    bitcast_result: bool,
) -> Option<RawLoadRewrite> {
    let suffix = path_to_leaf(types, leaf_ty, load_ty)?;
    let ptr_ty = ctx.ty_ptr(StorageClass::StorageBuffer, load_ty);
    let leaf_ptr = ctx.module.fresh_id();
    let mut operands = Vec::with_capacity(1 + path.indices.len() + suffix.len());
    operands.push(Operand::IdRef(path.root));
    operands.extend(path.indices.iter().copied().map(Operand::IdRef));
    operands.extend(
        suffix
            .into_iter()
            .map(|index| Operand::IdRef(ctx.const_uint(index))),
    );

    let mut prefix = vec![Instruction::new(
        Op::AccessChain,
        Some(ptr_ty),
        Some(leaf_ptr),
        operands,
    )];
    value_types.insert(leaf_ptr, ptr_ty);

    let mut load_operands = vec![Operand::IdRef(leaf_ptr)];
    load_operands.extend(load.operands.iter().skip(1).cloned());
    if !bitcast_result {
        return Some(RawLoadRewrite {
            prefix,
            replacement: Instruction::new(Op::Load, Some(result_ty), Some(result), load_operands),
        });
    }

    let loaded = ctx.module.fresh_id();
    prefix.push(Instruction::new(
        Op::Load,
        Some(load_ty),
        Some(loaded),
        load_operands,
    ));
    value_types.insert(loaded, load_ty);
    Some(RawLoadRewrite {
        prefix,
        replacement: Instruction::new(
            Op::Bitcast,
            Some(result_ty),
            Some(result),
            vec![Operand::IdRef(loaded)],
        ),
    })
}

#[derive(Clone, Copy, Debug)]
pub(in crate::passes) struct StructuredRawWordAccess {
    pub(in crate::passes) raw_var: Word,
    pub(in crate::passes) byte_offset: u32,
}

pub(in crate::passes) fn plan_structured_raw_word_load_rewrite(
    ctx: &mut Ctx,
    types: &HashMap<Word, Instruction>,
    value_types: &mut HashMap<Word, Word>,
    raw_alias_vars: &mut HashMap<Word, Word>,
    defs: &HashMap<Word, Instruction>,
    buffer_types: &HashMap<Word, Word>,
    paths: &HashMap<Word, BufferAccessPath>,
    ptr: Word,
    result: Word,
    result_ty: Word,
    result_kind: RawStoreObject,
) -> Option<RawLoadRewrite> {
    let path = paths.get(&ptr)?;
    let block_ty = buffer_types.get(&path.root).copied()?;
    let byte_offset = access_path_byte_offset(types, block_ty, &path.indices)?;
    if byte_offset % 4 == 0 {
        return None;
    }
    let access =
        structured_raw_word_access(ctx, types, raw_alias_vars, defs, buffer_types, paths, ptr)?;

    let uint = ctx.ty_uint();
    let mut prefix = Vec::new();
    let low_word = raw_word_load_at_const_index(
        ctx,
        value_types,
        access.raw_var,
        access.byte_offset / 4,
        &mut prefix,
    );
    let high_word = raw_word_load_at_const_index(
        ctx,
        value_types,
        access.raw_var,
        access.byte_offset / 4 + 1,
        &mut prefix,
    );

    let low_shift = ctx.const_uint((access.byte_offset % 4) * 8);
    let shifted_low = ctx.module.fresh_id();
    prefix.push(Instruction::new(
        Op::ShiftRightLogical,
        Some(uint),
        Some(shifted_low),
        vec![Operand::IdRef(low_word), Operand::IdRef(low_shift)],
    ));
    value_types.insert(shifted_low, uint);

    let high_shift = ctx.const_uint(32 - (access.byte_offset % 4) * 8);
    let shifted_high = ctx.module.fresh_id();
    prefix.push(Instruction::new(
        Op::ShiftLeftLogical,
        Some(uint),
        Some(shifted_high),
        vec![Operand::IdRef(high_word), Operand::IdRef(high_shift)],
    ));
    value_types.insert(shifted_high, uint);

    match result_kind {
        RawStoreObject::Uint32 => Some(RawLoadRewrite {
            prefix,
            replacement: Instruction::new(
                Op::BitwiseOr,
                Some(uint),
                Some(result),
                vec![Operand::IdRef(shifted_low), Operand::IdRef(shifted_high)],
            ),
        }),
        RawStoreObject::Float32 => {
            let assembled = ctx.module.fresh_id();
            prefix.push(Instruction::new(
                Op::BitwiseOr,
                Some(uint),
                Some(assembled),
                vec![Operand::IdRef(shifted_low), Operand::IdRef(shifted_high)],
            ));
            value_types.insert(assembled, uint);
            Some(RawLoadRewrite {
                prefix,
                replacement: Instruction::new(
                    Op::Bitcast,
                    Some(result_ty),
                    Some(result),
                    vec![Operand::IdRef(assembled)],
                ),
            })
        }
    }
}

pub(in crate::passes) fn plan_structured_raw_word_store_rewrite(
    ctx: &mut Ctx,
    types: &HashMap<Word, Instruction>,
    value_types: &mut HashMap<Word, Word>,
    raw_alias_vars: &mut HashMap<Word, Word>,
    defs: &HashMap<Word, Instruction>,
    buffer_types: &HashMap<Word, Word>,
    paths: &HashMap<Word, BufferAccessPath>,
    ptr: Word,
    object: Word,
) -> Option<RawStoreRewrite> {
    let path = paths.get(&ptr)?;
    let block_ty = buffer_types.get(&path.root).copied()?;
    let byte_offset = access_path_byte_offset(types, block_ty, &path.indices)?;
    if byte_offset % 4 == 0 {
        return None;
    }
    let access =
        structured_raw_word_access(ctx, types, raw_alias_vars, defs, buffer_types, paths, ptr)?;

    let uint = ctx.ty_uint();
    let mut instructions = Vec::new();
    for byte in 0..4 {
        let source = if byte == 0 {
            object
        } else {
            let shifted = ctx.module.fresh_id();
            let shift = ctx.const_uint(byte * 8);
            instructions.push(Instruction::new(
                Op::ShiftRightLogical,
                Some(uint),
                Some(shifted),
                vec![Operand::IdRef(object), Operand::IdRef(shift)],
            ));
            value_types.insert(shifted, uint);
            shifted
        };
        raw_byte_store_from_u32_at_const_offset(
            ctx,
            value_types,
            access.raw_var,
            access.byte_offset + byte,
            source,
            &mut instructions,
        );
    }

    let replacement = instructions.pop()?;
    Some(RawStoreRewrite {
        prefix: instructions,
        replacement,
    })
}

pub(in crate::passes) fn structured_raw_word_access(
    ctx: &mut Ctx,
    types: &HashMap<Word, Instruction>,
    raw_alias_vars: &mut HashMap<Word, Word>,
    defs: &HashMap<Word, Instruction>,
    buffer_types: &HashMap<Word, Word>,
    paths: &HashMap<Word, BufferAccessPath>,
    ptr: Word,
) -> Option<StructuredRawWordAccess> {
    let path = paths.get(&ptr)?;
    let block_ty = buffer_types.get(&path.root).copied()?;
    let byte_offset = access_path_byte_offset(types, block_ty, &path.indices)?;
    let binding = descriptor_binding(&ctx.module, path.root)?;
    let raw_var = match raw_alias_vars.get(&path.root).copied() {
        Some(var) => var,
        None => {
            let mut type_defs = combined_type_defs(ctx, defs);
            let var = create_raw_alias_buffer(ctx, binding, defs, &mut type_defs);
            raw_alias_vars.insert(path.root, var);
            var
        }
    };
    Some(StructuredRawWordAccess {
        raw_var,
        byte_offset,
    })
}

pub(in crate::passes) fn raw_word_pointer_at_const_index(
    ctx: &mut Ctx,
    value_types: &mut HashMap<Word, Word>,
    raw_var: Word,
    word_index: u32,
    instructions: &mut Vec<Instruction>,
) -> Word {
    let uint = ctx.ty_uint();
    let ptr_uint_ty = ctx.ty_ptr(StorageClass::StorageBuffer, uint);
    let raw_ptr = ctx.module.fresh_id();
    let zero = ctx.const_uint(0);
    let word = ctx.const_uint(word_index);
    instructions.push(Instruction::new(
        Op::AccessChain,
        Some(ptr_uint_ty),
        Some(raw_ptr),
        vec![
            Operand::IdRef(raw_var),
            Operand::IdRef(zero),
            Operand::IdRef(word),
        ],
    ));
    value_types.insert(raw_ptr, ptr_uint_ty);
    raw_ptr
}

pub(in crate::passes) fn raw_word_load_at_const_index(
    ctx: &mut Ctx,
    value_types: &mut HashMap<Word, Word>,
    raw_var: Word,
    word_index: u32,
    instructions: &mut Vec<Instruction>,
) -> Word {
    let uint = ctx.ty_uint();
    let ptr = raw_word_pointer_at_const_index(ctx, value_types, raw_var, word_index, instructions);
    let word = ctx.module.fresh_id();
    instructions.push(Instruction::new(
        Op::Load,
        Some(uint),
        Some(word),
        vec![Operand::IdRef(ptr)],
    ));
    value_types.insert(word, uint);
    word
}

pub(in crate::passes) fn raw_byte_store_from_u32_at_const_offset(
    ctx: &mut Ctx,
    value_types: &mut HashMap<Word, Word>,
    raw_var: Word,
    byte_offset: u32,
    value: Word,
    instructions: &mut Vec<Instruction>,
) {
    let uint = ctx.ty_uint();
    let ptr =
        raw_word_pointer_at_const_index(ctx, value_types, raw_var, byte_offset / 4, instructions);
    let lane_shift = (byte_offset % 4) * 8;
    let byte_mask = ctx.const_uint(0xff);
    let byte = ctx.module.fresh_id();
    instructions.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(byte),
        vec![Operand::IdRef(value), Operand::IdRef(byte_mask)],
    ));
    value_types.insert(byte, uint);

    let shifted_byte = if lane_shift == 0 {
        byte
    } else {
        let shifted = ctx.module.fresh_id();
        let shift = ctx.const_uint(lane_shift);
        instructions.push(Instruction::new(
            Op::ShiftLeftLogical,
            Some(uint),
            Some(shifted),
            vec![Operand::IdRef(byte), Operand::IdRef(shift)],
        ));
        value_types.insert(shifted, uint);
        shifted
    };

    let shifted_mask = 0xffu32 << lane_shift;
    let inverted_mask = ctx.const_uint(!shifted_mask);
    let scope = ctx.const_uint(Scope::Device as u32);
    let semantics = ctx.const_uint(MemorySemantics::RELAXED.bits());
    let old = ctx.module.fresh_id();
    instructions.push(Instruction::new(
        Op::AtomicAnd,
        Some(uint),
        Some(old),
        vec![
            Operand::IdRef(ptr),
            Operand::IdScope(scope),
            Operand::IdMemorySemantics(semantics),
            Operand::IdRef(inverted_mask),
        ],
    ));
    value_types.insert(old, uint);
    let old = ctx.module.fresh_id();
    instructions.push(Instruction::new(
        Op::AtomicOr,
        Some(uint),
        Some(old),
        vec![
            Operand::IdRef(ptr),
            Operand::IdScope(scope),
            Operand::IdMemorySemantics(semantics),
            Operand::IdRef(shifted_byte),
        ],
    ));
    value_types.insert(old, uint);
}
