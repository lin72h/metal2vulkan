//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

#[derive(Clone, Debug)]
pub(in crate::passes) struct AccessChainProvenance {
    pub(in crate::passes) root: Word,
    pub(in crate::passes) indices: Vec<Word>,
    pub(in crate::passes) ptr_ty: Word,
    pub(in crate::passes) op: Op,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(in crate::passes) struct LocalPointerFieldKey {
    pub(in crate::passes) root: Word,
    pub(in crate::passes) indices: Vec<u32>,
}

pub(in crate::passes) fn recover_inlined_local_pointer_fields(ctx: &mut Ctx, entry_idx: usize) {
    let store_markers = local_pointer_field_store_markers(ctx);
    let load_markers = local_pointer_field_load_markers(ctx);
    if store_markers.is_empty() || load_markers.is_empty() {
        return;
    }

    let mut access_keys: HashMap<Word, LocalPointerFieldKey> = HashMap::new();
    let mut stored_fields: HashMap<LocalPointerFieldKey, Word> = HashMap::new();
    for blk in &ctx.module.functions[entry_idx].blocks {
        for inst in &blk.instructions {
            if let Some((result, key)) = local_pointer_field_access_key(ctx, &access_keys, inst) {
                access_keys.insert(result, key);
            }
            if inst.class.opcode != Op::Store {
                continue;
            }
            let (Some(Operand::IdRef(ptr)), Some(Operand::IdRef(value))) =
                (inst.operands.first(), inst.operands.get(1))
            else {
                continue;
            };
            let (Some(key), Some(source)) = (access_keys.get(ptr), store_markers.get(value)) else {
                continue;
            };
            stored_fields.insert(key.clone(), *source);
        }
    }
    if stored_fields.is_empty() {
        return;
    }

    let mut replacements = Vec::new();
    for blk in &ctx.module.functions[entry_idx].blocks {
        for inst in &blk.instructions {
            let Some(result) = inst.result_id else {
                continue;
            };
            let Some(key) = load_markers.get(&result) else {
                continue;
            };
            let Some(source) = stored_fields.get(key).copied() else {
                continue;
            };
            replacements.push((result, source));
        }
    }
    let func = &mut ctx.module.functions[entry_idx];
    for (from, to) in replacements {
        replace_id_in_function(func, from, to);
    }
}

pub(in crate::passes) fn local_pointer_field_store_markers(ctx: &Ctx) -> HashMap<Word, Word> {
    ctx.emit_sidecar
        .local_pointer_field_stores
        .iter()
        .map(|fact| (fact.id, fact.source))
        .collect()
}

pub(in crate::passes) fn local_pointer_field_load_markers(
    ctx: &Ctx,
) -> HashMap<Word, LocalPointerFieldKey> {
    ctx.emit_sidecar
        .local_pointer_field_loads
        .iter()
        .map(|fact| {
            (
                fact.id,
                LocalPointerFieldKey {
                    root: fact.root,
                    indices: fact.indices.clone(),
                },
            )
        })
        .collect()
}

pub(in crate::passes) fn local_pointer_field_access_key(
    ctx: &Ctx,
    access_keys: &HashMap<Word, LocalPointerFieldKey>,
    inst: &Instruction,
) -> Option<(Word, LocalPointerFieldKey)> {
    if !matches!(
        inst.class.opcode,
        Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
    ) {
        return None;
    }
    let result = inst.result_id?;
    let Some(Operand::IdRef(base)) = inst.operands.first() else {
        return None;
    };
    let mut key = access_keys
        .get(base)
        .cloned()
        .unwrap_or(LocalPointerFieldKey {
            root: *base,
            indices: Vec::new(),
        });
    for operand in &inst.operands[1..] {
        key.indices.push(access_index_u32(ctx, operand)?);
    }
    Some((result, key))
}

pub(in crate::passes) fn access_index_u32(ctx: &Ctx, operand: &Operand) -> Option<u32> {
    match operand {
        Operand::LiteralBit32(value) => Some(*value),
        Operand::IdRef(id) => const_u32(ctx, *id),
        _ => None,
    }
}

pub(in crate::passes) fn const_u32(ctx: &Ctx, id: Word) -> Option<u32> {
    ctx.module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
        .find(|inst| inst.class.opcode == Op::Constant && inst.result_id == Some(id))
        .and_then(|inst| match inst.operands.first() {
            Some(Operand::LiteralBit32(value)) => Some(*value),
            _ => None,
        })
}

/// Compose wrapper-style access chains after helper inlining. AIR canonicalization represents
/// pointer arithmetic as `{ [0 x T] }` GEPs. Before inlining, a helper parameter can legitimately be a
/// pointer to that wrapper; after inlining, the parameter may be replaced by an already-derived `T*`.
/// Then helper GEPs like `gep wrapper, %derived_t_ptr, 0, 0, 1` become invalid Logical SPIR-V:
/// `OpAccessChain %derived_t_ptr 0 1` tries to index through scalar/vector `T`. Track earlier access
/// chains and rewrite those derived chains back to the original root with a composed final index.
pub(in crate::passes) fn compose_derived_access_chains(ctx: &mut Ctx, entry_idx: usize) {
    let mut provenance: HashMap<Word, AccessChainProvenance> = HashMap::new();
    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let insts = ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .clone();
        let mut new_insts = Vec::with_capacity(insts.len());
        for mut inst in insts {
            if inst.class.opcode == Op::Select {
                rewrite_pointer_select_constant_operands(ctx, &mut inst);
                if let Some(result) = inst.result_id {
                    if let Some(prov) = selected_pointer_provenance(ctx, &provenance, &inst) {
                        provenance.insert(result, prov);
                    }
                }
                new_insts.push(inst);
                continue;
            }
            if !matches!(
                inst.class.opcode,
                Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
            ) {
                new_insts.push(inst);
                continue;
            }

            let mut pre = Vec::new();
            if let (Some(result), Some(ptr_ty), Some(Operand::IdRef(base))) =
                (inst.result_id, inst.result_type, inst.operands.first())
            {
                if let Some(prev) = provenance.get(base).cloned() {
                    let mut rewrote = false;
                    if let Some(ops) = compose_array_pointer_access(
                        ctx,
                        &prev,
                        ptr_ty,
                        &inst.operands[1..],
                        &mut pre,
                    ) {
                        inst = Instruction::new(prev.op, inst.result_type, inst.result_id, ops);
                        rewrote = true;
                    }
                    if !rewrote && pointer_pointee(ctx, prev.ptr_ty) == pointer_pointee(ctx, ptr_ty)
                    {
                        if let Some(offset) = derived_access_offset(ctx, &inst.operands[1..]) {
                            if let Some(last) = prev.indices.last().copied() {
                                if let Some(composed) =
                                    compose_access_index(ctx, last, offset, &mut pre)
                                {
                                    let mut indices = prev.indices.clone();
                                    if let Some(slot) = indices.last_mut() {
                                        *slot = composed;
                                    }
                                    let mut ops = vec![Operand::IdRef(prev.root)];
                                    ops.extend(indices.iter().copied().map(Operand::IdRef));
                                    inst = Instruction::new(
                                        prev.op,
                                        inst.result_type,
                                        inst.result_id,
                                        ops,
                                    );
                                }
                            }
                        }
                    }
                }

                if let Some(prov) = provenance_for_access_chain(ctx, &inst) {
                    provenance.insert(result, prov);
                }
            }
            new_insts.extend(pre);
            new_insts.push(inst);
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = new_insts;
    }
}

pub(in crate::passes) fn rewrite_pointer_select_constant_operands(
    ctx: &mut Ctx,
    inst: &mut Instruction,
) {
    let Some(result_type) = inst.result_type else {
        return;
    };
    if pointer_pointee(ctx, result_type).is_none() {
        return;
    }
    for operand in inst.operands.iter_mut().skip(1) {
        let Operand::IdRef(id) = operand else {
            continue;
        };
        let Some(opcode) = pointer_constant_opcode(ctx, *id) else {
            continue;
        };
        if value_result_type(ctx, *id) == Some(result_type) {
            continue;
        }
        *operand = Operand::IdRef(pointer_constant_for_type(ctx, opcode, result_type));
    }
}

pub(in crate::passes) fn pointer_constant_opcode(ctx: &Ctx, id: Word) -> Option<Op> {
    ctx.new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .find(|inst| inst.result_id == Some(id))
        .and_then(|inst| match inst.class.opcode {
            Op::ConstantNull | Op::Undef => Some(inst.class.opcode),
            _ => None,
        })
}

pub(in crate::passes) fn pointer_constant_for_type(ctx: &mut Ctx, opcode: Op, ty: Word) -> Word {
    if let Some(existing) = ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .find(|inst| inst.class.opcode == opcode && inst.result_type == Some(ty))
        .and_then(|inst| inst.result_id)
    {
        return existing;
    }
    let id = ctx.module.fresh_id();
    ctx.new_globals
        .push(Instruction::new(opcode, Some(ty), Some(id), vec![]));
    id
}

pub(in crate::passes) fn selected_pointer_provenance(
    ctx: &Ctx,
    provenance: &HashMap<Word, AccessChainProvenance>,
    inst: &Instruction,
) -> Option<AccessChainProvenance> {
    let result_type = inst.result_type?;
    pointer_pointee(ctx, result_type)?;
    let [_, Operand::IdRef(true_value), Operand::IdRef(false_value)] = inst.operands.as_slice()
    else {
        return None;
    };
    match (
        provenance.get(true_value).cloned(),
        is_constant_null_id(ctx, *true_value),
        provenance.get(false_value).cloned(),
        is_constant_null_id(ctx, *false_value),
    ) {
        (Some(mut prov), _, None, true) | (None, true, Some(mut prov), _) => {
            prov.ptr_ty = result_type;
            Some(prov)
        }
        _ => None,
    }
}

pub(in crate::passes) fn compose_array_pointer_access(
    ctx: &mut Ctx,
    prev: &AccessChainProvenance,
    ptr_ty: Word,
    operands: &[Operand],
    pre: &mut Vec<Instruction>,
) -> Option<Vec<Operand>> {
    let array_ty = pointer_pointee(ctx, prev.ptr_ty)?;
    let elem_ty = array_element_type(ctx, array_ty)?;
    if pointer_pointee(ctx, ptr_ty)? != elem_ty || prev.indices.is_empty() {
        return None;
    }
    match operands {
        [Operand::IdRef(member)] => {
            let mut ops = vec![Operand::IdRef(prev.root)];
            ops.extend(prev.indices.iter().copied().map(Operand::IdRef));
            ops.push(Operand::IdRef(*member));
            Some(ops)
        }
        [Operand::IdRef(offset), Operand::IdRef(member)] => {
            let last = prev.indices.last().copied()?;
            let composed = compose_access_index(ctx, last, *offset, pre)?;
            let mut indices = prev.indices.clone();
            if let Some(slot) = indices.last_mut() {
                *slot = composed;
            }
            let mut ops = vec![Operand::IdRef(prev.root)];
            ops.extend(indices.into_iter().map(Operand::IdRef));
            ops.push(Operand::IdRef(*member));
            Some(ops)
        }
        _ => None,
    }
}

pub(in crate::passes) fn array_element_type(ctx: &Ctx, ty: Word) -> Option<Word> {
    let def = type_def_of(ctx, ty)?;
    match def.class.opcode {
        Op::TypeArray | Op::TypeRuntimeArray => match def.operands.first() {
            Some(Operand::IdRef(elem)) => Some(*elem),
            _ => None,
        },
        _ => None,
    }
}

/// If `ty` is a 64-bit integer scalar type (`OpTypeInt 64 <signedness>`), return `Some(signed)`
/// where `signed` is true for a signed int; else `None`. Access-chain indices are always integer
/// SCALARS in SPIR-V (a vector index is invalid), so we only need the scalar case.
pub(in crate::passes) fn int64_signedness(ctx: &Ctx, ty: Word) -> Option<bool> {
    let def = type_def_of(ctx, ty)?;
    if def.class.opcode != Op::TypeInt {
        return None;
    }
    let width = match def.operands.first() {
        Some(Operand::LiteralBit32(w)) => *w,
        _ => return None,
    };
    if width != 64 {
        return None;
    }
    let signed = matches!(def.operands.get(1), Some(Operand::LiteralBit32(1)));
    Some(signed)
}

/// The constant integer value of a 64-bit `OpConstant` id, if it is one (and small enough to be an
/// access-chain index — which it always is). The owned operand stores a 64-bit constant as a single
/// `LiteralBit64`; a 32-bit one as `LiteralBit32` (handled here too for robustness).
pub(in crate::passes) fn const_i64_value(ctx: &Ctx, id: Word) -> Option<u64> {
    let def = type_def_of(ctx, id)?;
    if def.class.opcode != Op::Constant {
        return None;
    }
    match def.operands.first() {
        Some(Operand::LiteralBit64(w)) => Some(*w),
        Some(Operand::LiteralBit32(w)) => Some(*w as u64),
        _ => None,
    }
}

pub(in crate::passes) fn pointer_pointee(ctx: &Ctx, ptr_ty: Word) -> Option<Word> {
    let def = type_def_of(ctx, ptr_ty)?;
    if def.class.opcode != Op::TypePointer {
        return None;
    }
    match def.operands.get(1) {
        Some(Operand::IdRef(pointee)) => Some(*pointee),
        _ => None,
    }
}

pub(in crate::passes) fn derived_access_offset(ctx: &Ctx, operands: &[Operand]) -> Option<Word> {
    if operands.len() == 1 {
        return match operands.first() {
            Some(Operand::IdRef(id)) => Some(*id),
            _ => None,
        };
    }
    if operands.len() != 2 {
        return None;
    }
    let first = match operands.first() {
        Some(Operand::IdRef(id)) => *id,
        _ => return None,
    };
    let offset = match operands.get(1) {
        Some(Operand::IdRef(id)) => *id,
        _ => return None,
    };
    if first == offset || const_i64_value(ctx, first) == Some(0) {
        Some(offset)
    } else {
        None
    }
}

pub(in crate::passes) fn provenance_for_access_chain(
    ctx: &Ctx,
    inst: &Instruction,
) -> Option<AccessChainProvenance> {
    let ptr_ty = inst.result_type?;
    let root = match inst.operands.first() {
        Some(Operand::IdRef(root)) => *root,
        _ => return None,
    };
    let indices = inst.operands[1..]
        .iter()
        .map(|op| match op {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    pointer_pointee(ctx, ptr_ty)?;
    Some(AccessChainProvenance {
        root,
        indices,
        ptr_ty,
        op: inst.class.opcode,
    })
}

pub(in crate::passes) fn compose_access_index(
    ctx: &mut Ctx,
    base_index: Word,
    offset: Word,
    pre: &mut Vec<Instruction>,
) -> Option<Word> {
    if const_i64_value(ctx, offset) == Some(0) {
        return Some(base_index);
    }
    match (
        const_i64_value(ctx, base_index),
        const_i64_value(ctx, offset),
    ) {
        (Some(a), Some(b)) => return Some(const_int_like(ctx, base_index, a + b)),
        (_, Some(0)) => return Some(base_index),
        _ => {}
    }
    let base_ty = value_result_type(ctx, base_index)?;
    let offset_ty = value_result_type(ctx, offset)?;
    if base_ty != offset_ty {
        return None;
    }
    let result = ctx.module.fresh_id();
    pre.push(Instruction::new(
        Op::IAdd,
        Some(base_ty),
        Some(result),
        vec![Operand::IdRef(base_index), Operand::IdRef(offset)],
    ));
    Some(result)
}
