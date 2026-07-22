//! Private-resource atomic lowering and shared raw-access analysis.

use super::*;

/// Rewrite every `OpAtomic*` in the entry function whose pointer has Private storage class into the
/// equivalent non-atomic load/op/store. Private memory is per-invocation, so there is no contention
/// and the rewrite is semantically exact; it is also required, since SPIR-V's universal validation
/// rules forbid atomics on Private storage. This runs after absent (function-constant-gated) buffer
/// pointers have been re-classed to Private zero vars, which is the only way a Private atomic arises.
pub(in crate::passes) fn rewrite_private_pointer_atomics(ctx: &mut Ctx, entry_idx: usize) {
    // id -> result type and type id -> its OpTypePointer storage class, across globals and the entry
    // function, so we can test whether an atomic's pointer operand has Private storage.
    let mut result_types: HashMap<Word, Word> = HashMap::new();
    let mut ptr_storages: HashMap<Word, StorageClass> = HashMap::new();
    for inst in ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
    {
        if let (Some(rid), Some(rty)) = (inst.result_id, inst.result_type) {
            result_types.insert(rid, rty);
        }
        if inst.class.opcode == Op::TypePointer {
            if let (Some(rid), Some(Operand::StorageClass(sc))) =
                (inst.result_id, inst.operands.first())
            {
                ptr_storages.insert(rid, *sc);
            }
        }
    }
    for blk in &ctx.module.functions[entry_idx].blocks {
        for inst in &blk.instructions {
            if let (Some(rid), Some(rty)) = (inst.result_id, inst.result_type) {
                result_types.insert(rid, rty);
            }
        }
    }
    let is_private_ptr = |ptr: Word| -> bool {
        result_types.get(&ptr).and_then(|ty| ptr_storages.get(ty)) == Some(&StorageClass::Private)
    };

    let bool_ty = ctx.ty_bool();
    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let insts = ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .clone();
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts {
            if atomic_i32_value_operands(inst.class.opcode).is_none() {
                out.push(inst);
                continue;
            }
            let Some(&Operand::IdRef(ptr)) = inst.operands.first() else {
                out.push(inst);
                continue;
            };
            if !is_private_ptr(ptr) {
                out.push(inst);
                continue;
            }
            if lower_one_private_atomic(ctx, &inst, ptr, bool_ty, &mut out).is_none() {
                // Unsupported atomic shape (e.g. CompareExchange) — leave it; spirv-val will flag it
                // rather than us emitting something wrong.
                out.push(inst);
            }
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = out;
    }
}

/// Emit the non-atomic equivalent of one `OpAtomic*` on a Private pointer into `out`, returning
/// `Some(())` when handled. The atomic's result id (if any) takes the OLD value, matching Metal/SPIR-V
/// atomic semantics.
pub(in crate::passes) fn lower_one_private_atomic(
    ctx: &mut Ctx,
    inst: &Instruction,
    ptr: Word,
    bool_ty: Word,
    out: &mut Vec<Instruction>,
) -> Option<()> {
    // Operand layout for the handled ops: [pointer, scope, semantics, (value)].
    let value = match inst.operands.get(3) {
        Some(Operand::IdRef(v)) => Some(*v),
        _ => None,
    };
    let load_old = |out: &mut Vec<Instruction>, result_type: Word, result: Word| {
        out.push(Instruction::new(
            Op::Load,
            Some(result_type),
            Some(result),
            vec![Operand::IdRef(ptr)],
        ));
    };
    let store_new = |out: &mut Vec<Instruction>, new: Word| {
        out.push(Instruction::new(
            Op::Store,
            None,
            None,
            vec![Operand::IdRef(ptr), Operand::IdRef(new)],
        ));
    };
    match inst.class.opcode {
        Op::AtomicLoad => {
            load_old(out, inst.result_type?, inst.result_id?);
        }
        Op::AtomicStore => {
            store_new(out, value?);
        }
        Op::AtomicExchange => {
            load_old(out, inst.result_type?, inst.result_id?);
            store_new(out, value?);
        }
        Op::AtomicIAdd | Op::AtomicISub | Op::AtomicAnd | Op::AtomicOr | Op::AtomicXor => {
            let result_type = inst.result_type?;
            let old = inst.result_id?;
            let value = value?;
            load_old(out, result_type, old);
            let new = ctx.module.fresh_id();
            let op = match inst.class.opcode {
                Op::AtomicIAdd => Op::IAdd,
                Op::AtomicISub => Op::ISub,
                Op::AtomicAnd => Op::BitwiseAnd,
                Op::AtomicOr => Op::BitwiseOr,
                Op::AtomicXor => Op::BitwiseXor,
                _ => return None,
            };
            out.push(Instruction::new(
                op,
                Some(result_type),
                Some(new),
                vec![Operand::IdRef(old), Operand::IdRef(value)],
            ));
            store_new(out, new);
        }
        Op::AtomicSMax | Op::AtomicSMin | Op::AtomicUMax | Op::AtomicUMin => {
            let result_type = inst.result_type?;
            let old = inst.result_id?;
            let value = value?;
            load_old(out, result_type, old);
            // new = (value <cmp> old) ? value : old, where cmp selects the kept extreme.
            let cmp_op = match inst.class.opcode {
                Op::AtomicSMax => Op::SGreaterThan,
                Op::AtomicSMin => Op::SLessThan,
                Op::AtomicUMax => Op::UGreaterThan,
                Op::AtomicUMin => Op::ULessThan,
                _ => return None,
            };
            let cond = ctx.module.fresh_id();
            out.push(Instruction::new(
                cmp_op,
                Some(bool_ty),
                Some(cond),
                vec![Operand::IdRef(value), Operand::IdRef(old)],
            ));
            let new = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::Select,
                Some(result_type),
                Some(new),
                vec![
                    Operand::IdRef(cond),
                    Operand::IdRef(value),
                    Operand::IdRef(old),
                ],
            ));
            store_new(out, new);
        }
        _ => return None,
    }
    Some(())
}

pub(in crate::passes) fn plan_raw_byte_atomic_rewrite(
    ctx: &mut Ctx,
    types: &HashMap<Word, Instruction>,
    value_types: &mut HashMap<Word, Word>,
    raw_alias_vars: &mut HashMap<Word, Word>,
    defs: &HashMap<Word, Instruction>,
    buffer_types: &HashMap<Word, Word>,
    paths: &HashMap<Word, BufferAccessPath>,
    atomic: &Instruction,
) -> Option<RawPointerRewrite> {
    let value_operands = atomic_i32_value_operands(atomic.class.opcode)?;
    if atomic
        .result_type
        .is_some_and(|ty| !is_uint_type(types, ty))
    {
        return None;
    }
    for idx in value_operands {
        let Some(Operand::IdRef(value)) = atomic.operands.get(*idx) else {
            return None;
        };
        let value_ty = value_types.get(value).copied()?;
        if !is_uint_type(types, value_ty) {
            return None;
        }
    }
    let Some(Operand::IdRef(ptr)) = atomic.operands.first() else {
        return None;
    };
    if let Some(rewrite) = plan_raw_word_pointer_rewrite(
        ctx,
        types,
        value_types,
        raw_alias_vars,
        defs,
        buffer_types,
        paths,
        *ptr,
    ) {
        return Some(rewrite);
    }

    let ptr_ty = value_types.get(ptr).copied()?;
    let ptr_pointee = ptr_pointee(types, ptr_ty)?;
    let path = paths.get(ptr)?;
    let block_ty = buffer_types.get(&path.root).copied()?;
    let leaf_ty = access_path_leaf_type(types, block_ty, &path.indices)?;
    if ptr_pointee == leaf_ty {
        return None;
    }
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
}

pub(in crate::passes) fn access_path_leaf_type(
    types: &HashMap<Word, Instruction>,
    root_ty: Word,
    indices: &[Word],
) -> Option<Word> {
    access_path_byte_offset_and_leaf_type(types, root_ty, indices)
        .map(|(_, leaf_ty)| leaf_ty)
        .or_else(|| {
            let operands = indices
                .iter()
                .copied()
                .map(Operand::IdRef)
                .collect::<Vec<_>>();
            type_after_spirv_access_operands(types, root_ty, &operands)
        })
}

pub(in crate::passes) fn atomic_i32_value_operands(op: Op) -> Option<&'static [usize]> {
    match op {
        Op::AtomicLoad => Some(&[]),
        Op::AtomicStore
        | Op::AtomicExchange
        | Op::AtomicIAdd
        | Op::AtomicISub
        | Op::AtomicSMin
        | Op::AtomicUMin
        | Op::AtomicSMax
        | Op::AtomicUMax
        | Op::AtomicAnd
        | Op::AtomicOr
        | Op::AtomicFAddEXT => Some(&[3]),
        Op::AtomicCompareExchange => Some(&[4, 5]),
        _ => None,
    }
}

pub(in crate::passes) fn plan_raw_word_pointer_rewrite(
    ctx: &mut Ctx,
    types: &HashMap<Word, Instruction>,
    value_types: &mut HashMap<Word, Word>,
    raw_alias_vars: &mut HashMap<Word, Word>,
    defs: &HashMap<Word, Instruction>,
    buffer_types: &HashMap<Word, Word>,
    paths: &HashMap<Word, BufferAccessPath>,
    ptr: Word,
) -> Option<RawPointerRewrite> {
    let path = paths.get(&ptr)?;
    let block_ty = buffer_types.get(&path.root).copied()?;
    let byte_index = raw_byte_buffer_index(types, block_ty, &path.indices)?;
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

    let uint = ctx.ty_uint();
    let mut prefix = Vec::new();
    let word_index = raw_store_word_index(ctx, types, value_types, byte_index, &mut prefix)?;
    let ptr_uint_ty = ctx.ty_ptr(StorageClass::StorageBuffer, uint);
    let raw_ptr = ctx.module.fresh_id();
    let zero = ctx.const_uint(0);
    prefix.push(Instruction::new(
        Op::AccessChain,
        Some(ptr_uint_ty),
        Some(raw_ptr),
        vec![
            Operand::IdRef(raw_var),
            Operand::IdRef(zero),
            Operand::IdRef(word_index),
        ],
    ));
    value_types.insert(raw_ptr, ptr_uint_ty);

    Some(RawPointerRewrite {
        prefix,
        ptr: raw_ptr,
    })
}

pub(in crate::passes) fn plan_structured_raw_word_pointer_rewrite(
    ctx: &mut Ctx,
    types: &HashMap<Word, Instruction>,
    value_types: &mut HashMap<Word, Word>,
    raw_alias_vars: &mut HashMap<Word, Word>,
    defs: &HashMap<Word, Instruction>,
    buffer_types: &HashMap<Word, Word>,
    paths: &HashMap<Word, BufferAccessPath>,
    ptr: Word,
) -> Option<RawPointerRewrite> {
    let path = paths.get(&ptr)?;
    let block_ty = buffer_types.get(&path.root).copied()?;
    let mut prefix = Vec::new();
    let word_index = if let Some(byte_offset) =
        access_path_byte_offset(types, block_ty, &path.indices)
    {
        if byte_offset % 4 != 0 {
            return None;
        }
        ctx.const_uint(byte_offset / 4)
    } else {
        let dynamic_index = runtime_array_word_index(types, value_types, block_ty, &path.indices)?;
        coerce_storage_buffer_word_index_to_uint(
            ctx,
            types,
            value_types,
            dynamic_index,
            &mut prefix,
        )?
    };
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
    let uint = ctx.ty_uint();
    let ptr_uint_ty = ctx.ty_ptr(StorageClass::StorageBuffer, uint);
    let raw_ptr = ctx.module.fresh_id();
    let zero = ctx.const_uint(0);
    prefix.push(Instruction::new(
        Op::AccessChain,
        Some(ptr_uint_ty),
        Some(raw_ptr),
        vec![
            Operand::IdRef(raw_var),
            Operand::IdRef(zero),
            Operand::IdRef(word_index),
        ],
    ));
    value_types.insert(raw_ptr, ptr_uint_ty);
    Some(RawPointerRewrite {
        prefix,
        ptr: raw_ptr,
    })
}

pub(in crate::passes) fn runtime_array_word_index(
    types: &HashMap<Word, Instruction>,
    value_types: &HashMap<Word, Word>,
    block_ty: Word,
    indices: &[Word],
) -> Option<Word> {
    let [member, index] = indices else {
        return None;
    };
    if const_u32(types, *member) != Some(0) {
        return None;
    }
    let block = types.get(&block_ty)?;
    if block.class.opcode != Op::TypeStruct {
        return None;
    }
    let array = match block.operands.first() {
        Some(Operand::IdRef(array)) => *array,
        _ => return None,
    };
    let array_def = types.get(&array)?;
    if !matches!(array_def.class.opcode, Op::TypeArray | Op::TypeRuntimeArray) {
        return None;
    }
    let elem = match array_def.operands.first() {
        Some(Operand::IdRef(elem)) => *elem,
        _ => return None,
    };
    let (size, align) = ty_size_align(elem, types);
    if round_up(size, align) != 4 || !value_types.contains_key(index) {
        return None;
    }
    Some(*index)
}

pub(in crate::passes) fn coerce_storage_buffer_word_index_to_uint(
    ctx: &mut Ctx,
    types: &HashMap<Word, Instruction>,
    value_types: &mut HashMap<Word, Word>,
    index: Word,
    prefix: &mut Vec<Instruction>,
) -> Option<Word> {
    if let Some(index) = const_u32(types, index) {
        return Some(ctx.const_uint(index));
    }
    let uint = ctx.ty_uint();
    let index_ty = value_types.get(&index).copied()?;
    if is_uint_type(types, index_ty) {
        return Some(index);
    }
    if !is_integer_type_with_width(types, index_ty, 64) {
        return None;
    }
    let converted = ctx.module.fresh_id();
    prefix.push(Instruction::new(
        Op::UConvert,
        Some(uint),
        Some(converted),
        vec![Operand::IdRef(index)],
    ));
    value_types.insert(converted, uint);
    Some(converted)
}

pub(in crate::passes) fn combined_type_defs(
    ctx: &Ctx,
    defs: &HashMap<Word, Instruction>,
) -> HashMap<Word, Instruction> {
    let mut types = defs.clone();
    for inst in &ctx.new_globals {
        if let Some(id) = inst.result_id {
            types.insert(id, inst.clone());
        }
    }
    types
}

pub(in crate::passes) fn combined_value_types(ctx: &Ctx, entry_idx: usize) -> HashMap<Word, Word> {
    let mut value_types = HashMap::new();
    for inst in ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
    {
        if let (Some(id), Some(ty)) = (inst.result_id, inst.result_type) {
            value_types.insert(id, ty);
        }
    }
    for param in &ctx.module.functions[entry_idx].parameters {
        if let (Some(id), Some(ty)) = (param.result_id, param.result_type) {
            value_types.insert(id, ty);
        }
    }
    for block in &ctx.module.functions[entry_idx].blocks {
        for inst in &block.instructions {
            if let (Some(id), Some(ty)) = (inst.result_id, inst.result_type) {
                value_types.insert(id, ty);
            }
        }
    }
    value_types
}

pub(in crate::passes) fn raw_word_chain_offset(
    types: &HashMap<Word, Instruction>,
    operands: &[Operand],
) -> Option<u32> {
    match operands {
        [Operand::IdRef(word)] => const_u32(types, *word),
        [Operand::IdRef(first), Operand::IdRef(word)] if const_u32(types, *first) == Some(0) => {
            const_u32(types, *word)
        }
        _ => None,
    }
}

pub(in crate::passes) fn access_path_byte_offset(
    types: &HashMap<Word, Instruction>,
    root_ty: Word,
    indices: &[Word],
) -> Option<u32> {
    access_path_byte_offset_and_leaf_type(types, root_ty, indices).map(|(offset, _)| offset)
}

pub(in crate::passes) fn access_path_byte_offset_and_leaf_type(
    types: &HashMap<Word, Instruction>,
    root_ty: Word,
    indices: &[Word],
) -> Option<(u32, Word)> {
    let mut ty = root_ty;
    let mut offset = 0u32;
    for index in indices {
        let def = types.get(&ty)?;
        match def.class.opcode {
            Op::TypeStruct => {
                let member = const_u32(types, *index)? as usize;
                loop {
                    let def = types.get(&ty)?;
                    if def.class.opcode != Op::TypeStruct {
                        return None;
                    }
                    if member < def.operands.len() {
                        let mut member_offset = 0u32;
                        for (idx, operand) in def.operands.iter().enumerate() {
                            let Operand::IdRef(member_ty) = operand else {
                                return None;
                            };
                            let (size, align) = ty_size_align(*member_ty, types);
                            member_offset = round_up(member_offset, align);
                            if idx == member {
                                offset += member_offset;
                                ty = *member_ty;
                                break;
                            }
                            member_offset += size;
                        }
                        break;
                    }
                    if def.operands.len() != 1 {
                        return None;
                    }
                    let Some(Operand::IdRef(member_ty)) = def.operands.first() else {
                        return None;
                    };
                    ty = *member_ty;
                }
            }
            Op::TypeArray | Op::TypeRuntimeArray => {
                let elem = match def.operands.first() {
                    Some(Operand::IdRef(elem)) => *elem,
                    _ => return None,
                };
                let (size, align) = ty_size_align(elem, types);
                offset += const_u32(types, *index)? * round_up(size, align);
                ty = elem;
            }
            Op::TypeVector => {
                let elem = match def.operands.first() {
                    Some(Operand::IdRef(elem)) => *elem,
                    _ => return None,
                };
                let (size, _) = ty_size_align(elem, types);
                offset += const_u32(types, *index)? * size;
                ty = elem;
            }
            _ => return None,
        }
    }
    Some((offset, ty))
}

pub(in crate::passes) fn direct_raw_root_word_index(
    types: &HashMap<Word, Instruction>,
    operands: &[Operand],
) -> Option<Word> {
    match operands {
        [Operand::IdRef(first), Operand::IdRef(word)] if const_u32(types, *first) == Some(0) => {
            Some(*word)
        }
        _ => None,
    }
}

pub(in crate::passes) fn is_raw_uint_runtime_block(
    types: &HashMap<Word, Instruction>,
    block_ty: Word,
) -> bool {
    let Some(block) = types.get(&block_ty) else {
        return false;
    };
    if block.class.opcode != Op::TypeStruct || block.operands.len() != 1 {
        return false;
    }
    let Some(Operand::IdRef(runtime_array_ty)) = block.operands.first() else {
        return false;
    };
    let Some(runtime_array) = types.get(runtime_array_ty) else {
        return false;
    };
    if runtime_array.class.opcode != Op::TypeRuntimeArray {
        return false;
    }
    match runtime_array.operands.first() {
        Some(Operand::IdRef(elem)) => is_uint_type(types, *elem),
        _ => false,
    }
}

pub(in crate::passes) fn raw_byte_buffer_index(
    types: &HashMap<Word, Instruction>,
    root_ty: Word,
    indices: &[Word],
) -> Option<Word> {
    let [member, byte_index] = indices else {
        return None;
    };
    if const_u32(types, *member) != Some(0) {
        return None;
    }
    let root = types.get(&root_ty)?;
    if root.class.opcode != Op::TypeStruct {
        return None;
    }
    let array = match root.operands.first() {
        Some(Operand::IdRef(array)) => *array,
        _ => return None,
    };
    let array_def = types.get(&array)?;
    if !matches!(array_def.class.opcode, Op::TypeArray | Op::TypeRuntimeArray) {
        return None;
    }
    let elem = match array_def.operands.first() {
        Some(Operand::IdRef(elem)) => *elem,
        _ => return None,
    };
    is_uchar_type(types, elem).then_some(*byte_index)
}

pub(in crate::passes) fn raw_store_word_index(
    ctx: &mut Ctx,
    types: &HashMap<Word, Instruction>,
    value_types: &mut HashMap<Word, Word>,
    byte_index: Word,
    prefix: &mut Vec<Instruction>,
) -> Option<Word> {
    if let Some(byte_index) = const_u32(types, byte_index) {
        return (byte_index % 4 == 0).then(|| ctx.const_uint(byte_index / 4));
    }

    let uint = ctx.ty_uint();
    let byte_index_ty = value_types.get(&byte_index).copied()?;
    let byte_index = if is_uint_type(types, byte_index_ty) {
        byte_index
    } else if is_integer_type_with_width(types, byte_index_ty, 64) {
        let converted = ctx.module.fresh_id();
        prefix.push(Instruction::new(
            Op::UConvert,
            Some(uint),
            Some(converted),
            vec![Operand::IdRef(byte_index)],
        ));
        value_types.insert(converted, uint);
        converted
    } else {
        return None;
    };

    let word_index = ctx.module.fresh_id();
    let divisor = ctx.const_uint(4);
    prefix.push(Instruction::new(
        Op::UDiv,
        Some(uint),
        Some(word_index),
        vec![Operand::IdRef(byte_index), Operand::IdRef(divisor)],
    ));
    value_types.insert(word_index, uint);
    Some(word_index)
}

pub(in crate::passes) fn const_u32(types: &HashMap<Word, Instruction>, id: Word) -> Option<u32> {
    let inst = types.get(&id)?;
    if inst.class.opcode != Op::Constant {
        return None;
    }
    match inst.operands.first() {
        Some(Operand::LiteralBit32(value)) => Some(*value),
        Some(Operand::LiteralBit64(value)) => u32::try_from(*value).ok(),
        _ => None,
    }
}

pub(in crate::passes) fn is_uint_type(types: &HashMap<Word, Instruction>, id: Word) -> bool {
    types.get(&id).is_some_and(|inst| {
        inst.class.opcode == Op::TypeInt
            && inst.operands.first() == Some(&Operand::LiteralBit32(32))
            && inst.operands.get(1) == Some(&Operand::LiteralBit32(0))
    })
}

pub(in crate::passes) fn is_uchar_type(types: &HashMap<Word, Instruction>, id: Word) -> bool {
    types.get(&id).is_some_and(|inst| {
        inst.class.opcode == Op::TypeInt
            && inst.operands.first() == Some(&Operand::LiteralBit32(8))
            && inst.operands.get(1) == Some(&Operand::LiteralBit32(0))
    })
}

pub(in crate::passes) fn is_float32_type(types: &HashMap<Word, Instruction>, id: Word) -> bool {
    types.get(&id).is_some_and(|inst| {
        inst.class.opcode == Op::TypeFloat
            && inst.operands.first() == Some(&Operand::LiteralBit32(32))
    })
}

pub(in crate::passes) fn is_integer_type_with_width(
    types: &HashMap<Word, Instruction>,
    id: Word,
    bits: u32,
) -> bool {
    types.get(&id).is_some_and(|inst| {
        inst.class.opcode == Op::TypeInt
            && inst.operands.first() == Some(&Operand::LiteralBit32(bits))
    })
}

pub(in crate::passes) fn raw_store_object_kind(
    types: &HashMap<Word, Instruction>,
    id: Word,
) -> Option<RawStoreObject> {
    if is_uint_type(types, id) {
        Some(RawStoreObject::Uint32)
    } else if is_float32_type(types, id) {
        Some(RawStoreObject::Float32)
    } else {
        None
    }
}

pub(in crate::passes) fn descriptor_binding(module: &Module, var: Word) -> Option<u32> {
    module.annotations.iter().find_map(|inst| {
        if inst.class.opcode == Op::Decorate
            && inst.operands.first() == Some(&Operand::IdRef(var))
            && inst.operands.get(1) == Some(&Operand::Decoration(Decoration::Binding))
        {
            match inst.operands.get(2) {
                Some(Operand::LiteralBit32(binding)) => Some(*binding),
                _ => None,
            }
        } else {
            None
        }
    })
}

pub(in crate::passes) fn create_raw_alias_buffer(
    ctx: &mut Ctx,
    binding: u32,
    defs: &HashMap<Word, Instruction>,
    types: &mut HashMap<Word, Instruction>,
) -> Word {
    let uint = ctx.ty_uint();
    let runtime = ctx.ty_runtime_array(uint);
    let block = ctx.module.fresh_id();
    let block_inst = type_inst(Op::TypeStruct, block, vec![Operand::IdRef(runtime)]);
    ctx.new_globals.push(block_inst.clone());
    types.insert(block, block_inst);
    *types = combined_type_defs(ctx, defs);
    decorate_block_struct(ctx, block, types);

    let ptr_ty = ctx.ty_ptr(StorageClass::StorageBuffer, block);
    let var = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::Variable,
        Some(ptr_ty),
        Some(var),
        vec![Operand::StorageClass(StorageClass::StorageBuffer)],
    ));
    decorate_binding(&mut ctx.module, var, binding);
    ctx.interface.push(var);
    var
}

/// The member-0 / element-0 sub-type of an aggregate type instruction (struct member 0, array elem).
pub(in crate::passes) fn member0_type(def: &Instruction) -> Option<Word> {
    match def.class.opcode {
        Op::TypeStruct | Op::TypeArray | Op::TypeRuntimeArray => match def.operands.first() {
            Some(Operand::IdRef(m)) => Some(*m),
            _ => None,
        },
        _ => None,
    }
}

pub(in crate::passes) fn runtime_array_block_element_type(
    types: &HashMap<Word, Instruction>,
    block_ty: Word,
) -> Option<Word> {
    let runtime_array = member0_type(types.get(&block_ty)?)?;
    member0_type(types.get(&runtime_array)?)
}

/// Index path (all `0`s) from `block_ty` down to the offset-0 sub-element whose type is `target`,
/// descending member-0/element-0 at each composite level. The collapsed buffer's bare pointer is
/// `&block[0...0]`, so a direct load of `target` reaches it via this path off the block var.
pub(in crate::passes) fn path_to_leaf(
    types: &HashMap<Word, Instruction>,
    block_ty: Word,
    target: Word,
) -> Option<Vec<u32>> {
    let mut cur = block_ty;
    let mut path = vec![];
    for _ in 0..32 {
        if cur == target {
            return Some(path);
        }
        let next = member0_type(types.get(&cur)?)?;
        path.push(0);
        cur = next;
    }
    None
}
