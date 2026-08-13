//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

/// Retype a plain element load whose pointer carrier is an array by explicitly selecting element
/// zero. LLVM opaque pointers permit loading `T` from the address of `[N x T]`; Logical SPIR-V
/// requires the otherwise address-identical zero descent to produce `T*` first.
pub(in crate::passes) fn repair_load_through_array_pointer(ctx: &mut Ctx, entry_idx: usize) {
    let mut plans = HashMap::<(usize, usize), (Word, Word, StorageClass)>::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, load) in block.instructions.iter().enumerate() {
            if load.class.opcode != Op::Load || load.operands.len() != 1 {
                continue;
            }
            let (Some(result_ty), Some(Operand::IdRef(pointer))) =
                (load.result_type, load.operands.first())
            else {
                continue;
            };
            let Some(pointer_ty) = value_result_type(ctx, *pointer) else {
                continue;
            };
            let Some(pointer_def) = type_def_of(ctx, pointer_ty) else {
                continue;
            };
            let (Some(Operand::StorageClass(storage)), Some(Operand::IdRef(array_ty))) =
                (pointer_def.operands.first(), pointer_def.operands.get(1))
            else {
                continue;
            };
            let Some(array_def) = type_def_of(ctx, *array_ty) else {
                continue;
            };
            if !matches!(array_def.class.opcode, Op::TypeArray | Op::TypeRuntimeArray)
                || array_def.operands.first() != Some(&Operand::IdRef(result_ty))
            {
                continue;
            }
            plans.insert((bi, ii), (*pointer, result_ty, *storage));
        }
    }
    if plans.is_empty() {
        return;
    }
    let zero = ctx.const_uint(0);
    for bi in 0..ctx.module.functions[entry_idx].blocks.len() {
        let old = std::mem::take(&mut ctx.module.functions[entry_idx].blocks[bi].instructions);
        let mut rewritten = Vec::with_capacity(old.len());
        for (ii, mut instruction) in old.into_iter().enumerate() {
            let Some(&(pointer, result_ty, storage)) = plans.get(&(bi, ii)) else {
                rewritten.push(instruction);
                continue;
            };
            let element_pointer = ctx.module.fresh_id();
            rewritten.push(Instruction::new(
                Op::InBoundsAccessChain,
                Some(ctx.ty_ptr(storage, result_ty)),
                Some(element_pointer),
                vec![Operand::IdRef(pointer), Operand::IdRef(zero)],
            ));
            instruction.operands[0] = Operand::IdRef(element_pointer);
            rewritten.push(instruction);
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = rewritten;
    }
}

/// Lower a flat scalar offset through `{ RuntimeArray<Vector<T, N>> }` into its exact typed path.
/// Opaque LLVM pointers allow `T*` arithmetic from the wrapper's address; Logical SPIR-V instead
/// needs member zero, vector index `flat / N`, and lane `flat % N` spelled out explicitly.
pub(in crate::passes) fn rewrite_flat_scalar_ptr_access_through_vector_array(
    ctx: &mut Ctx,
    entry_idx: usize,
) {
    #[derive(Clone, Copy)]
    struct Plan {
        bi: usize,
        ii: usize,
        base: Word,
        flat: Word,
        index_ty: Word,
        lanes: u32,
        result_ty: Word,
        result: Word,
    }

    let mut plans = Vec::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, instruction) in block.instructions.iter().enumerate() {
            if instruction.class.opcode != Op::PtrAccessChain || instruction.operands.len() != 3 {
                continue;
            }
            let (
                Some(result_ty),
                Some(result),
                Some(Operand::IdRef(base)),
                Some(Operand::IdRef(element)),
                Some(Operand::IdRef(flat)),
            ) = (
                instruction.result_type,
                instruction.result_id,
                instruction.operands.first(),
                instruction.operands.get(1),
                instruction.operands.get(2),
            )
            else {
                continue;
            };
            if const_u32(ctx, *element) != Some(0) {
                continue;
            }
            let Some(result_scalar) = pointer_pointee(ctx, result_ty) else {
                continue;
            };
            let Some(base_ty) = value_result_type(ctx, *base) else {
                continue;
            };
            let Some(base_struct) = pointer_pointee(ctx, base_ty) else {
                continue;
            };
            let Some(struct_def) = type_def_of(ctx, base_struct) else {
                continue;
            };
            let [Operand::IdRef(array_ty)] = struct_def.operands.as_slice() else {
                continue;
            };
            if struct_def.class.opcode != Op::TypeStruct {
                continue;
            }
            let Some(array_def) = type_def_of(ctx, *array_ty) else {
                continue;
            };
            if !matches!(array_def.class.opcode, Op::TypeArray | Op::TypeRuntimeArray) {
                continue;
            }
            let Some(Operand::IdRef(vector_ty)) = array_def.operands.first() else {
                continue;
            };
            let Some(vector_def) = type_def_of(ctx, *vector_ty) else {
                continue;
            };
            let (Some(Operand::IdRef(component)), Some(Operand::LiteralBit32(lanes))) =
                (vector_def.operands.first(), vector_def.operands.get(1))
            else {
                continue;
            };
            if vector_def.class.opcode != Op::TypeVector
                || *component != result_scalar
                || *lanes == 0
            {
                continue;
            }
            let Some(index_ty) = value_result_type(ctx, *flat) else {
                continue;
            };
            let Some(index_def) = type_def_of(ctx, index_ty) else {
                continue;
            };
            if index_def.class.opcode != Op::TypeInt
                || !matches!(index_def.operands.get(1), Some(Operand::LiteralBit32(0)))
            {
                continue;
            }
            plans.push(Plan {
                bi,
                ii,
                base: *base,
                flat: *flat,
                index_ty,
                lanes: *lanes,
                result_ty,
                result,
            });
        }
    }
    if plans.is_empty() {
        return;
    }
    let by_site = plans
        .into_iter()
        .map(|plan| ((plan.bi, plan.ii), plan))
        .collect::<HashMap<_, _>>();
    let member_zero = ctx.const_uint(0);
    for bi in 0..ctx.module.functions[entry_idx].blocks.len() {
        let old = std::mem::take(&mut ctx.module.functions[entry_idx].blocks[bi].instructions);
        let mut rewritten = Vec::with_capacity(old.len());
        for (ii, instruction) in old.into_iter().enumerate() {
            let Some(plan) = by_site.get(&(bi, ii)).copied() else {
                rewritten.push(instruction);
                continue;
            };
            let (vector_index, lane) = if let Some(flat) = const_u32(ctx, plan.flat) {
                (
                    ctx.const_int_of(plan.index_ty, i64::from(flat / plan.lanes)),
                    ctx.const_int_of(plan.index_ty, i64::from(flat % plan.lanes)),
                )
            } else {
                let lanes = ctx.const_int_of(plan.index_ty, i64::from(plan.lanes));
                let vector_index = ctx.module.fresh_id();
                rewritten.push(Instruction::new(
                    Op::UDiv,
                    Some(plan.index_ty),
                    Some(vector_index),
                    vec![Operand::IdRef(plan.flat), Operand::IdRef(lanes)],
                ));
                let lane = ctx.module.fresh_id();
                rewritten.push(Instruction::new(
                    Op::UMod,
                    Some(plan.index_ty),
                    Some(lane),
                    vec![Operand::IdRef(plan.flat), Operand::IdRef(lanes)],
                ));
                (vector_index, lane)
            };
            rewritten.push(Instruction::new(
                Op::InBoundsAccessChain,
                Some(plan.result_ty),
                Some(plan.result),
                vec![
                    Operand::IdRef(plan.base),
                    Operand::IdRef(member_zero),
                    Operand::IdRef(vector_index),
                    Operand::IdRef(lane),
                ],
            ));
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = rewritten;
    }
}

/// Repair a `OpLoad %vecN <p>` whose pointer `<p>` is a SCALAR-pointee access chain that the emitter
/// produced by accumulating a chain of element-strided geps (`gep <N x scalar>, ptr, idx`) into a single
/// index off a VECTOR-pointer base, but emitted with the wrong opcode/type: it used
/// `OpInBoundsAccessChain %_ptr_SC_scalar %base %idx` (which indexes a COMPONENT of the vector and yields
/// a scalar pointer) where the AIR meant `OpPtrAccessChain %_ptr_SC_vecN %base %idx` (which STRIDES by the
/// whole vector). The result is a self-inconsistent module: `<p>` is typed `scalar*` yet the load reads a
/// `vecN` ("OpLoad Result Type vecN does not match Pointer's type"). When `<p> = (InBounds)AccessChain
/// %_ptr_SC_scalar %base %idx` with EXACTLY ONE index, `%base` is `_ptr_SC_vecN` (same storage class,
/// pointee = the loaded vector), `scalar` is that vector's component type, and EVERY use of `<p>` is a
/// load of that same `vecN`, the chain is rewritten to `OpPtrAccessChain %_ptr_SC_vecN %base %idx` — the
/// vector stride the gep chain encodes.
///
/// Byte-EXACT by construction: the AIR strides by `<N x scalar>` (the gep element type), so `%idx` is a
/// vector-stride count and the byte address is `%idx * sizeof(vecN)` from `%base` — exactly what
/// `OpPtrAccessChain` over `vecN*` computes; the prior component-index typing was the bug (a scalar pick
/// can never satisfy a `vecN` load). Floor-SAFE by construction: fires ONLY on the currently-INVALID
/// shape (a `vecN` load through a scalar-pointee chain whose base is a `vecN*`), which a valid/banked
/// module never contains; the all-uses-are-`vecN`-loads gate makes the retype of `<p>` safe. Decides
/// purely from IR structure (pointer/vector type defs), never a shader name. `OpPtrAccessChain` off a
/// select/phi pointer is legal here under `VariablePointersStorageBuffer` (the same capability the
/// sibling `OpPtrAccessChain` arms in these MPS kernels already rely on).
pub(in crate::passes) fn repair_vector_load_through_scalar_stride(ctx: &mut Ctx, entry_idx: usize) {
    let mut ptr_info: HashMap<Word, (StorageClass, Word)> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if inst.class.opcode == Op::TypePointer {
            if let (Some(id), Some(Operand::StorageClass(s)), Some(Operand::IdRef(p))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*s, *p));
            }
        }
    }
    // component type of a vector type id, if it is one.
    let vec_component = |ctx: &Ctx, ty: Word| -> Option<Word> {
        let def = type_def_of(ctx, ty)?;
        if def.class.opcode != Op::TypeVector {
            return None;
        }
        match def.operands.first() {
            Some(Operand::IdRef(c)) => Some(*c),
            _ => None,
        }
    };

    // value-id -> (block, inst) index for the entry function body, plus all use sites of a value.
    let mut def_at: HashMap<Word, (usize, usize)> = HashMap::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if let Some(id) = inst.result_id {
                def_at.insert(id, (bi, ii));
            }
        }
    }
    // For a candidate pointer `<p>`, every use must be `OpLoad <vecN> <p>` for the same vecN.
    let all_uses_are_vec_load = |ctx: &Ctx, p: Word, vec_ty: Word| -> bool {
        let mut count = 0usize;
        for block in &ctx.module.functions[entry_idx].blocks {
            for inst in &block.instructions {
                let references_p = inst
                    .operands
                    .iter()
                    .any(|o| matches!(o, Operand::IdRef(r) if *r == p));
                if !references_p {
                    continue;
                }
                if inst.class.opcode != Op::Load || inst.result_type != Some(vec_ty) {
                    return false;
                }
                // The pointer must be the load's pointer operand (first operand), not some other slot.
                if inst.operands.first() != Some(&Operand::IdRef(p)) {
                    return false;
                }
                count += 1;
            }
        }
        count > 0
    };

    // (block, inst of the access chain, new pointer-type id = base's pointer type)
    let mut edits: Vec<(usize, usize, Word)> = Vec::new();
    for block in &ctx.module.functions[entry_idx].blocks {
        for inst in &block.instructions {
            if inst.class.opcode != Op::Load {
                continue;
            }
            let Some(vec_ty) = inst.result_type else {
                continue;
            };
            let Some(component) = vec_component(ctx, vec_ty) else {
                continue;
            };
            let Some(Operand::IdRef(p)) = inst.operands.first() else {
                continue;
            };
            let Some(&(pbi, pii)) = def_at.get(p) else {
                continue;
            };
            let pdef = &ctx.module.functions[entry_idx].blocks[pbi].instructions[pii];
            if !matches!(pdef.class.opcode, Op::InBoundsAccessChain | Op::AccessChain) {
                continue;
            }
            // exactly one index (base + 1 index).
            if pdef.operands.len() != 2 {
                continue;
            }
            let Some(p_ty) = pdef.result_type else {
                continue;
            };
            let Some(&(p_sc, p_pointee)) = ptr_info.get(&p_ty) else {
                continue;
            };
            if p_pointee != component {
                continue;
            }
            let Some(Operand::IdRef(base)) = pdef.operands.first() else {
                continue;
            };
            let Some(base_ty) = value_result_type(ctx, *base) else {
                continue;
            };
            let Some(&(base_sc, base_pointee)) = ptr_info.get(&base_ty) else {
                continue;
            };
            if base_sc != p_sc || base_pointee != vec_ty {
                continue;
            }
            if !all_uses_are_vec_load(ctx, *p, vec_ty) {
                continue;
            }
            edits.push((pbi, pii, base_ty));
        }
    }
    for (bi, ii, new_ty) in edits {
        let inst = &mut ctx.module.functions[entry_idx].blocks[bi].instructions[ii];
        let result_id = inst.result_id;
        let operands = inst.operands.clone();
        *inst = Instruction::new(Op::PtrAccessChain, Some(new_ty), result_id, operands);
    }
}

/// Expand an invalid vector load through one word of a raw `{ RuntimeArray<uint> }` buffer into
/// consecutive scalar-word loads. The chain's final index names the first word; lane `n` reads
/// `index + n`, bitcasts that word to the declared 32-bit component, and reconstructs the vector.
pub(in crate::passes) fn repair_vector_load_through_raw_word_pointer(
    ctx: &mut Ctx,
    entry_idx: usize,
) {
    let mut ptr_info: HashMap<Word, (StorageClass, Word)> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if inst.class.opcode == Op::TypePointer {
            if let (Some(id), Some(Operand::StorageClass(storage)), Some(Operand::IdRef(pointee))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*storage, *pointee));
            }
        }
    }

    #[derive(Clone)]
    struct Plan {
        chain_block: usize,
        chain_inst: usize,
        chain_id: Word,
        chain: Instruction,
        vector_ty: Word,
        component_ty: Word,
        lanes: u32,
        strided_tail: bool,
        last_index: Word,
        index_ty: Word,
        word_offset: u32,
    }

    let mut definitions = HashMap::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if let Some(id) = inst.result_id {
                definitions.insert(id, (bi, ii));
            }
        }
    }
    let uint = ctx.ty_uint();
    let mut plans = HashMap::new();
    for block in &ctx.module.functions[entry_idx].blocks {
        for load in &block.instructions {
            if load.class.opcode != Op::Load || load.operands.len() != 1 {
                continue;
            }
            let Some(vector_ty) = load.result_type else {
                continue;
            };
            let Some(vector_def) = type_def_of(ctx, vector_ty) else {
                continue;
            };
            let (Some(Operand::IdRef(component_ty)), Some(Operand::LiteralBit32(lanes))) =
                (vector_def.operands.first(), vector_def.operands.get(1))
            else {
                continue;
            };
            if vector_def.class.opcode != Op::TypeVector
                || *lanes < 2
                || direct_scalar_width(ctx, *component_ty) != Some(32)
            {
                continue;
            }
            let Some(Operand::IdRef(chain_id)) = load.operands.first() else {
                continue;
            };
            let Some(&(chain_block, chain_inst)) = definitions.get(chain_id) else {
                continue;
            };
            let chain =
                &ctx.module.functions[entry_idx].blocks[chain_block].instructions[chain_inst];
            if !matches!(
                chain.class.opcode,
                Op::AccessChain | Op::InBoundsAccessChain
            ) || chain.operands.len() < 3
            {
                continue;
            }
            let Some(result_ptr_ty) = chain.result_type else {
                continue;
            };
            let Some((StorageClass::StorageBuffer, result_pointee)) =
                ptr_info.get(&result_ptr_ty).copied()
            else {
                continue;
            };
            let Some(Operand::IdRef(base)) = chain.operands.first() else {
                continue;
            };
            let Some(base_ptr_ty) = value_result_type(ctx, *base) else {
                continue;
            };
            let Some((StorageClass::StorageBuffer, base_pointee)) =
                ptr_info.get(&base_ptr_ty).copied()
            else {
                continue;
            };
            if single_member_array_scalar_elem(ctx, base_pointee) != Some(uint) {
                continue;
            }
            // Composed opaque-pointer GEPs can leave one or more zero-stride scalar tails after the
            // meaningful vector stride (`..., vector_index, 0, 0`). Once the type walk has reached a
            // scalar, each such trailing zero is an address-neutral `gep T, ptr, 0`; retain no more
            // than the canonical `[base, member, word, vector_stride]` form before recognizing it.
            // This normalization is local to the already-invalid vector-load shape, so it cannot
            // alter a valid aggregate descent.
            let mut effective_chain = chain.clone();
            let mut word_offset = 0;
            // A composed `gep vecN, ptr, trailing_vector` can follow the dynamic vector stride as
            // `..., vector_index, 0, trailing_vector`. Once the raw interface has erased the vector
            // type, recover that constant vector offset in raw words and canonicalize the chain.
            if effective_chain.operands.len() == 6
                && walk_into_type(ctx, base_pointee, &effective_chain.operands[1..3]) == Some(uint)
                && effective_chain
                    .operands
                    .get(4)
                    .and_then(|operand| match operand {
                        Operand::IdRef(id) => const_u32(ctx, *id),
                        _ => None,
                    })
                    == Some(0)
            {
                let trailing_vector =
                    effective_chain
                        .operands
                        .get(5)
                        .and_then(|operand| match operand {
                            Operand::IdRef(id) => const_u32(ctx, *id),
                            _ => None,
                        });
                if let Some(trailing_vector) = trailing_vector {
                    word_offset = trailing_vector.saturating_mul(*lanes);
                    effective_chain.operands.truncate(4);
                }
            }
            while effective_chain.operands.len() > 4
                && effective_chain
                    .operands
                    .last()
                    .and_then(|operand| match operand {
                        Operand::IdRef(id) => const_u32(ctx, *id),
                        _ => None,
                    })
                    == Some(0)
            {
                effective_chain.operands.pop();
            }
            let strided_tail = if walk_into_type(ctx, base_pointee, &effective_chain.operands[1..])
                == Some(uint)
                && result_pointee == uint
            {
                false
            } else if effective_chain.operands.len() == 4
                && walk_into_type(ctx, base_pointee, &effective_chain.operands[1..3]) == Some(uint)
                && result_pointee == vector_ty
            {
                true
            } else {
                continue;
            };
            let Some(Operand::IdRef(last_index)) = effective_chain.operands.last() else {
                continue;
            };
            let last_index = *last_index;
            let Some(index_ty) = value_result_type(ctx, last_index) else {
                continue;
            };
            plans.entry(*chain_id).or_insert_with(|| Plan {
                chain_block,
                chain_inst,
                chain_id: *chain_id,
                chain: effective_chain,
                vector_ty,
                component_ty: *component_ty,
                lanes: *lanes,
                strided_tail,
                last_index,
                index_ty,
                word_offset,
            });
        }
    }
    if plans.is_empty() {
        return;
    }

    // Every use of the scalar pointer must be the same exact plain vector load.
    plans.retain(|chain_id, plan| {
        let mut uses = 0;
        for inst in ctx.module.functions[entry_idx]
            .blocks
            .iter()
            .flat_map(|block| block.instructions.iter())
        {
            if inst.result_id == Some(*chain_id) {
                continue;
            }
            if !inst
                .operands
                .iter()
                .any(|operand| operand == &Operand::IdRef(*chain_id))
            {
                continue;
            }
            if inst.class.opcode != Op::Load
                || inst.operands.as_slice() != [Operand::IdRef(*chain_id)]
                || inst.result_type != Some(plan.vector_ty)
            {
                return false;
            }
            uses += 1;
        }
        uses > 0
    });
    if plans.is_empty() {
        return;
    }

    let chain_sites = plans
        .values()
        .map(|plan| ((plan.chain_block, plan.chain_inst), plan.chain_id))
        .collect::<HashMap<_, _>>();
    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let old = std::mem::take(&mut ctx.module.functions[entry_idx].blocks[bi].instructions);
        let mut rewritten = Vec::with_capacity(old.len() + 16);
        for (ii, inst) in old.into_iter().enumerate() {
            if chain_sites.contains_key(&(bi, ii)) {
                continue;
            }
            let Some(chain_id) = inst.operands.first().and_then(|operand| match operand {
                Operand::IdRef(id) if plans.contains_key(id) => Some(*id),
                _ => None,
            }) else {
                rewritten.push(inst);
                continue;
            };
            let plan = &plans[&chain_id];
            let Some(result) = inst.result_id else {
                rewritten.push(inst);
                continue;
            };
            let original_last_index = plan.last_index;
            let index_ty = plan.index_ty;
            let first_word = if plan.strided_tail {
                let scaled = ctx.module.fresh_id();
                let lanes = ctx.const_int_of(index_ty, i64::from(plan.lanes));
                rewritten.push(Instruction::new(
                    Op::IMul,
                    Some(index_ty),
                    Some(scaled),
                    vec![Operand::IdRef(original_last_index), Operand::IdRef(lanes)],
                ));
                let Some(Operand::IdRef(base_word)) = plan
                    .chain
                    .operands
                    .get(plan.chain.operands.len().saturating_sub(2))
                else {
                    rewritten.push(inst);
                    continue;
                };
                let base_word = if value_result_type(ctx, *base_word) == Some(index_ty) {
                    *base_word
                } else if let Some(value) = const_u32(ctx, *base_word) {
                    ctx.const_int_of(index_ty, i64::from(value))
                } else {
                    let converted = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::UConvert,
                        Some(index_ty),
                        Some(converted),
                        vec![Operand::IdRef(*base_word)],
                    ));
                    converted
                };
                let first = ctx.module.fresh_id();
                rewritten.push(Instruction::new(
                    Op::IAdd,
                    Some(index_ty),
                    Some(first),
                    vec![Operand::IdRef(base_word), Operand::IdRef(scaled)],
                ));
                if plan.word_offset == 0 {
                    first
                } else {
                    let offset_first = ctx.module.fresh_id();
                    let offset = ctx.const_int_of(index_ty, i64::from(plan.word_offset));
                    rewritten.push(Instruction::new(
                        Op::IAdd,
                        Some(index_ty),
                        Some(offset_first),
                        vec![Operand::IdRef(first), Operand::IdRef(offset)],
                    ));
                    offset_first
                }
            } else {
                original_last_index
            };
            let mut components = Vec::with_capacity(plan.lanes as usize);
            for lane in 0..plan.lanes {
                let lane_index = if lane == 0 {
                    first_word
                } else {
                    let index = ctx.module.fresh_id();
                    let offset = ctx.const_int_of(index_ty, i64::from(lane));
                    rewritten.push(Instruction::new(
                        Op::IAdd,
                        Some(index_ty),
                        Some(index),
                        vec![Operand::IdRef(first_word), Operand::IdRef(offset)],
                    ));
                    index
                };
                let pointer = ctx.module.fresh_id();
                let mut operands = plan.chain.operands.clone();
                if plan.strided_tail {
                    operands.pop();
                }
                *operands.last_mut().expect("raw word chain has an index") =
                    Operand::IdRef(lane_index);
                rewritten.push(Instruction::new(
                    plan.chain.class.opcode,
                    Some(ctx.ty_ptr(StorageClass::StorageBuffer, uint)),
                    Some(pointer),
                    operands,
                ));
                let word = ctx.module.fresh_id();
                rewritten.push(Instruction::new(
                    Op::Load,
                    Some(uint),
                    Some(word),
                    vec![Operand::IdRef(pointer)],
                ));
                let component = if plan.component_ty == uint {
                    word
                } else {
                    let component = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::Bitcast,
                        Some(plan.component_ty),
                        Some(component),
                        vec![Operand::IdRef(word)],
                    ));
                    component
                };
                components.push(Operand::IdRef(component));
            }
            rewritten.push(Instruction::new(
                Op::CompositeConstruct,
                Some(plan.vector_ty),
                Some(result),
                components,
            ));
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = rewritten;
    }
}

/// Repair a `OpLoad %scalar <p>` whose pointer `<p> = OpPtrAccessChain %_ptr_SC_vecN %base %idx…` is a
/// VECTOR pointer landed at a vector boundary, where the AIR meant to read the vector's component 0 at
/// that stride (`OpLoad %float (PtrAccessChain <4 x float>*, idx)` — the matrix-column gather emits
/// element 0 as a valid `load vecN`+`OpCompositeExtract 0` but elements 1..N-1 as this strided
/// scalar load). `<p>` is typed `vecN*` yet the load reads a `scalar` ("OpLoad Result Type scalar does not
/// match Pointer's type"). When `<p> = OpPtrAccessChain %_ptr_SC_vecN %base [indices]`, the load result is
/// the vector's COMPONENT scalar, and EVERY use of `<p>` is a load of that scalar, a trailing component-0
/// index is appended (`… %uint_0`) and `<p>` is retyped to `_ptr_SC_scalar` — the byte address is
/// unchanged (a `PtrAccessChain` over `vecN*` lands at `idx * sizeof(vecN)`, exactly a vector boundary, so
/// component 0 is byte offset 0), now well-typed for the scalar load.
///
/// Byte-EXACT by construction (component 0 of a vector at a vector boundary is the same 4 bytes the
/// invalid scalar load already addressed). Floor-SAFE by construction: fires ONLY on the currently-INVALID
/// shape (a scalar load through a `PtrAccessChain` whose result is a `vecN*`), which a valid/banked module
/// never contains; the all-uses-are-scalar-loads gate makes the retype safe. Decides purely from IR
/// structure, never a shader name.
pub(in crate::passes) fn repair_scalar_load_through_vector_ptr(ctx: &mut Ctx, entry_idx: usize) {
    let mut ptr_info: HashMap<Word, (StorageClass, Word)> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if inst.class.opcode == Op::TypePointer {
            if let (Some(id), Some(Operand::StorageClass(s)), Some(Operand::IdRef(p))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*s, *p));
            }
        }
    }
    let vec_component = |ctx: &Ctx, ty: Word| -> Option<Word> {
        let def = type_def_of(ctx, ty)?;
        if def.class.opcode != Op::TypeVector {
            return None;
        }
        match def.operands.first() {
            Some(Operand::IdRef(c)) => Some(*c),
            _ => None,
        }
    };
    // Find (or note absence of) the `_ptr_SC_scalar` pointer type already present in the module.
    let find_scalar_ptr = |sc: StorageClass, scalar: Word| -> Option<Word> {
        ptr_info
            .iter()
            .find(|(_, (s, p))| *s == sc && *p == scalar)
            .map(|(id, _)| *id)
    };

    let mut def_at: HashMap<Word, (usize, usize)> = HashMap::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if let Some(id) = inst.result_id {
                def_at.insert(id, (bi, ii));
            }
        }
    }
    let all_uses_are_scalar_load = |ctx: &Ctx, p: Word, scalar_ty: Word| -> bool {
        let mut count = 0usize;
        for block in &ctx.module.functions[entry_idx].blocks {
            for inst in &block.instructions {
                let refs = inst
                    .operands
                    .iter()
                    .any(|o| matches!(o, Operand::IdRef(r) if *r == p));
                if !refs {
                    continue;
                }
                if inst.class.opcode != Op::Load
                    || inst.result_type != Some(scalar_ty)
                    || inst.operands.first() != Some(&Operand::IdRef(p))
                {
                    return false;
                }
                count += 1;
            }
        }
        count > 0
    };

    // (block, inst of the PtrAccessChain, new result-pointer-type = scalar*)
    let mut edits: Vec<(usize, usize, Word)> = Vec::new();
    for block in &ctx.module.functions[entry_idx].blocks {
        for inst in &block.instructions {
            if inst.class.opcode != Op::Load {
                continue;
            }
            let Some(scalar_ty) = inst.result_type else {
                continue;
            };
            // result must be a scalar (not a vector/aggregate).
            if !matches!(
                type_def_of(ctx, scalar_ty).map(|d| d.class.opcode),
                Some(Op::TypeInt) | Some(Op::TypeFloat)
            ) {
                continue;
            }
            let Some(Operand::IdRef(p)) = inst.operands.first() else {
                continue;
            };
            let Some(&(pbi, pii)) = def_at.get(p) else {
                continue;
            };
            let pdef = &ctx.module.functions[entry_idx].blocks[pbi].instructions[pii];
            if pdef.class.opcode != Op::PtrAccessChain {
                continue;
            }
            let Some(p_ty) = pdef.result_type else {
                continue;
            };
            let Some(&(p_sc, p_pointee)) = ptr_info.get(&p_ty) else {
                continue;
            };
            // The chain points at a vector whose component is the loaded scalar.
            if vec_component(ctx, p_pointee) != Some(scalar_ty) {
                continue;
            }
            let Some(scalar_ptr_ty) = find_scalar_ptr(p_sc, scalar_ty) else {
                continue;
            };
            if !all_uses_are_scalar_load(ctx, *p, scalar_ty) {
                continue;
            }
            edits.push((pbi, pii, scalar_ptr_ty));
        }
    }

    let zero = ctx.const_uint(0);
    for (bi, ii, new_ty) in edits {
        let inst = &mut ctx.module.functions[entry_idx].blocks[bi].instructions[ii];
        inst.result_type = Some(new_ty);
        inst.operands.push(Operand::IdRef(zero));
    }
}

/// Drop the stores into a WRITE-ONLY-DEAD `Function`-storage variable when at least one of those stores
/// is type-INVALID. The native emitter materializes a thread-local pointer ARRAY (`[N x ulong]`) for an
/// AIR `[N x ptr]` alloca, fills it with the kernel's buffer pointers, then — when it lowers the dynamic
/// selection off that array as an `OpSelect`/`OpPhi` cascade over the buffer descriptors directly — never
/// READS the array back. The leftover stores `OpStore %slot %bufferPtr` write a StorageBuffer pointer into
/// a `Function ulong` slot ("OpStore Object type does not match Pointer's pointee"), an invalid residue of
/// a dead write. When EVERY use of a `Function` variable (transitively through access chains) is the
/// POINTER operand of an `OpStore` — never loaded, never a store OBJECT, never any other operand — the
/// variable is write-only and unobservable, so removing all its stores (and the now-dead access chains
/// feeding them) is byte-NEUTRAL: `Function` storage is per-invocation and never compared against a
/// golden, and a write with no subsequent read has no effect.
///
/// Byte-NEUTRAL by construction (write-only `Function` memory, no reachable read). Floor-SAFE by
/// construction: gated on at least one store into the variable being type-mismatched (object type ≠ slot
/// pointee) — a valid/banked module's stores are all well-typed, so nothing matches; a legitimately
/// write-only scratch array with well-typed stores is left untouched. Decides purely from IR structure
/// (storage class + use census + a store-type compare), never a shader name.
pub(in crate::passes) fn drop_writeonly_dead_local_array_stores(ctx: &mut Ctx, entry_idx: usize) {
    let mut ptr_info: HashMap<Word, (StorageClass, Word)> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if inst.class.opcode == Op::TypePointer {
            if let (Some(id), Some(Operand::StorageClass(s)), Some(Operand::IdRef(p))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*s, *p));
            }
        }
    }

    // Candidate Function-storage variables declared in the entry function's first block.
    let mut candidates: Vec<Word> = Vec::new();
    for block in &ctx.module.functions[entry_idx].blocks {
        for inst in &block.instructions {
            if inst.class.opcode == Op::Variable {
                if let (Some(id), Some(ty)) = (inst.result_id, inst.result_type) {
                    if matches!(ptr_info.get(&ty), Some((StorageClass::Function, _))) {
                        candidates.push(id);
                    }
                }
            }
        }
    }

    for var in candidates {
        // Transitive closure of access chains rooted at `var` (S = {var} ∪ chains based in S).
        let mut set: std::collections::HashSet<Word> = std::collections::HashSet::new();
        set.insert(var);
        loop {
            let mut grew = false;
            for block in &ctx.module.functions[entry_idx].blocks {
                for inst in &block.instructions {
                    if !matches!(
                        inst.class.opcode,
                        Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
                    ) {
                        continue;
                    }
                    let Some(id) = inst.result_id else { continue };
                    if set.contains(&id) {
                        continue;
                    }
                    if matches!(inst.operands.first(), Some(Operand::IdRef(b)) if set.contains(b)) {
                        set.insert(id);
                        grew = true;
                    }
                }
            }
            if !grew {
                break;
            }
        }

        // Census: every reference to an id in `set` must be either a chain in `set` using it as base,
        // or the POINTER operand of an OpStore. Otherwise the variable escapes / is read — bail.
        let mut write_only = true;
        let mut has_invalid_store = false;
        'census: for block in &ctx.module.functions[entry_idx].blocks {
            for inst in &block.instructions {
                match inst.class.opcode {
                    Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain => {
                        // A chain in `set` is fine (its base ∈ set). A chain NOT in set must not
                        // reference any set id (it would be reading through it).
                        let in_set = inst.result_id.map(|r| set.contains(&r)).unwrap_or(false);
                        if in_set {
                            continue;
                        }
                        if inst
                            .operands
                            .iter()
                            .any(|o| matches!(o, Operand::IdRef(r) if set.contains(r)))
                        {
                            write_only = false;
                            break 'census;
                        }
                    }
                    Op::Store => {
                        let ptr_op = inst.operands.first();
                        let obj_op = inst.operands.get(1);
                        let ptr_in = matches!(ptr_op, Some(Operand::IdRef(r)) if set.contains(r));
                        let obj_in = matches!(obj_op, Some(Operand::IdRef(r)) if set.contains(r));
                        if obj_in {
                            // The variable's value is being stored elsewhere — it escapes.
                            write_only = false;
                            break 'census;
                        }
                        if ptr_in {
                            // Type check: object's result type vs slot pointee.
                            if let (Some(Operand::IdRef(slot)), Some(Operand::IdRef(obj))) =
                                (ptr_op, obj_op)
                            {
                                let slot_pointee = value_result_type(ctx, *slot)
                                    .and_then(|t| ptr_info.get(&t).map(|(_, p)| *p));
                                let obj_ty = value_result_type(ctx, *obj);
                                if let (Some(sp), Some(ot)) = (slot_pointee, obj_ty) {
                                    if sp != ot {
                                        has_invalid_store = true;
                                    }
                                }
                            }
                        }
                    }
                    _ => {
                        if inst
                            .operands
                            .iter()
                            .any(|o| matches!(o, Operand::IdRef(r) if set.contains(r)))
                        {
                            write_only = false;
                            break 'census;
                        }
                    }
                }
            }
        }

        if !write_only || !has_invalid_store {
            continue;
        }

        // Remove the stores whose pointer ∈ set, and the access chains in `set` (now dead).
        for block in &mut ctx.module.functions[entry_idx].blocks {
            block.instructions.retain(|inst| match inst.class.opcode {
                Op::Store => {
                    !matches!(inst.operands.first(), Some(Operand::IdRef(r)) if set.contains(r))
                }
                Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain => {
                    !inst.result_id.map(|r| set.contains(&r)).unwrap_or(false)
                }
                _ => true,
            });
        }
    }
}

/// Drop access chains in the entry function whose result is UNUSED and whose chain is currently
/// INVALID (it over-indexes a non-composite or lands on a different pointee than declared). The
/// emitter sometimes leaves a dead address computation behind a value that a later prune/structurize
/// pass removed — e.g. a `device uchar*` element pointer re-indexed to `_ptr…_ushort` (a byte-pointer
/// reinterpret) whose load was pruned. An access chain has no side effects, so removing an unused one
/// is byte-NEUTRAL; gating on "currently invalid" keeps a valid/banked module provably untouched (its
/// every chain either is used or validates, so nothing matches). Runs to a fixpoint because dropping
/// one dead chain can orphan another it fed.
pub(in crate::passes) fn drop_dead_invalid_access_chains(ctx: &mut Ctx, entry_idx: usize) {
    let mut ptr_info: HashMap<Word, (StorageClass, Word)> = HashMap::new();
    let mut value_types = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if let (Some(id), Some(ty)) = (inst.result_id, inst.result_type) {
            value_types.insert(id, ty);
        }
        if inst.class.opcode == Op::TypePointer {
            if let (Some(id), Some(Operand::StorageClass(s)), Some(Operand::IdRef(p))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*s, *p));
            }
        }
    }
    for parameter in &ctx.module.functions[entry_idx].parameters {
        if let (Some(id), Some(ty)) = (parameter.result_id, parameter.result_type) {
            value_types.insert(id, ty);
        }
    }
    for block in &ctx.module.functions[entry_idx].blocks {
        for instruction in &block.instructions {
            if let (Some(id), Some(ty)) = (instruction.result_id, instruction.result_type) {
                value_types.insert(id, ty);
            }
        }
    }
    let chain_definitions = ctx.module.functions[entry_idx]
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .filter_map(|inst| {
            matches!(
                inst.class.opcode,
                Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
            )
            .then_some((inst.result_id?, inst.clone()))
        })
        .collect::<HashMap<_, _>>();
    // Build operand reference counts once, then peel newly dead parent chains with a worklist.
    // The former fixpoint rescanned the complete function after every ancestry layer, which became
    // quadratic when raw-byte replay expanded a large shader by tens of thousands of instructions.
    let mut references: HashMap<Word, usize> = HashMap::new();
    for block in &ctx.module.functions[entry_idx].blocks {
        for instruction in &block.instructions {
            for operand in &instruction.operands {
                if let Operand::IdRef(id) = operand {
                    *references.entry(*id).or_default() += 1;
                }
            }
        }
    }
    for instruction in &ctx.module.annotations {
        for operand in &instruction.operands {
            if let Operand::IdRef(id) = operand {
                *references.entry(*id).or_default() += 1;
            }
        }
    }
    let mut pending = chain_definitions
        .keys()
        .copied()
        .filter(|id| references.get(id).copied().unwrap_or(0) == 0)
        .collect::<Vec<_>>();
    let mut victims = HashSet::new();
    while let Some(result) = pending.pop() {
        if victims.contains(&result) || references.get(&result).copied().unwrap_or(0) != 0 {
            continue;
        }
        let Some(instruction) = chain_definitions.get(&result) else {
            continue;
        };
        // A dead, otherwise-valid PtrAccessChain may be the only remaining use of an invalid parent
        // after its load/store was rewritten. Remove that suffix too, but only when its ancestry
        // proves the invalidity; an unrelated valid dead chain is deliberately left byte-identical.
        if !invalid_access_chain_or_ancestor(
            ctx,
            &ptr_info,
            &value_types,
            &chain_definitions,
            instruction,
            &mut HashSet::new(),
        ) {
            continue;
        }
        victims.insert(result);
        for operand in &instruction.operands {
            let Operand::IdRef(id) = operand else {
                continue;
            };
            let Some(count) = references.get_mut(id) else {
                continue;
            };
            *count = count.saturating_sub(1);
            if *count == 0 && chain_definitions.contains_key(id) {
                pending.push(*id);
            }
        }
    }
    if victims.is_empty() {
        return;
    }
    for block in &mut ctx.module.functions[entry_idx].blocks {
        block.instructions.retain(|instruction| {
            !instruction
                .result_id
                .is_some_and(|result| victims.contains(&result))
        });
    }
}

fn invalid_access_chain_or_ancestor(
    ctx: &Ctx,
    ptr_info: &HashMap<Word, (StorageClass, Word)>,
    value_types: &HashMap<Word, Word>,
    definitions: &HashMap<Word, Instruction>,
    inst: &Instruction,
    seen: &mut HashSet<Word>,
) -> bool {
    let Some(result) = inst.result_id else {
        return false;
    };
    if !seen.insert(result) {
        return false;
    }
    let Some(result_type) = inst.result_type else {
        return false;
    };
    let Some(&(result_storage, result_pointee)) = ptr_info.get(&result_type) else {
        return false;
    };
    let Some(Operand::IdRef(base)) = inst.operands.first() else {
        return false;
    };
    let Some(base_ptr_ty) = value_types.get(base).copied() else {
        return false;
    };
    let Some(&(base_storage, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
        return false;
    };
    let index_start = if inst.class.opcode == Op::PtrAccessChain {
        // Operand 1 is the pointer-arithmetic element; it preserves the base pointee. Only later
        // operands descend through composites.
        2
    } else {
        1
    };
    let indices = inst.operands.get(index_start..).unwrap_or_default();
    let (reached, consumed) = walk_into_type_partial(ctx, base_pointee, indices);
    if result_storage != base_storage || consumed != indices.len() || reached != result_pointee {
        return true;
    }
    definitions.get(base).is_some_and(|parent| {
        invalid_access_chain_or_ancestor(ctx, ptr_info, value_types, definitions, parent, seen)
    })
}

/// Store-side analogue of [`lower_cross_member_subword_load`]: lower an `OpStore` of a wider scalar
/// through a struct-MEMBER pointer into per-member little-endian stores. A packed-struct write —
/// `OpStore %p %v` with `%p : _ptr…_uint` (struct member 8, a `uint`) and `%v : ulong` — spans member
/// 8 and member 9, so the faithful `OpStore` mismatches the pointee (spirv-val "OpStore Pointer's type
/// does not match Object's type"). The fix splits the object into the members the write covers and
/// stores each piece through that member's sibling chain.
///
/// Byte-EXACT by construction on a little-endian target AND floor-SAFE by construction: only fires on a
/// store whose object type MISMATCHES the member pointee (a valid module's stores match), and only when
/// the object EXACTLY tiles a run of FULL direct-scalar members from the addressed member (no partial
/// member — that would clobber bytes outside the write), so a banked module is never touched.
pub(in crate::passes) fn lower_cross_member_subword_store(ctx: &mut Ctx, entry_idx: usize) {
    let mut ptr_info: HashMap<Word, (StorageClass, Word)> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if inst.class.opcode == Op::TypePointer {
            if let (Some(id), Some(Operand::StorageClass(s)), Some(Operand::IdRef(p))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*s, *p));
            }
        }
    }
    let mut member_offset: HashMap<(Word, u32), u32> = HashMap::new();
    for inst in &ctx.module.annotations {
        if inst.class.opcode == Op::MemberDecorate {
            if let (
                Some(Operand::IdRef(sty)),
                Some(Operand::LiteralBit32(m)),
                Some(Operand::Decoration(Decoration::Offset)),
                Some(Operand::LiteralBit32(off)),
            ) = (
                inst.operands.first(),
                inst.operands.get(1),
                inst.operands.get(2),
                inst.operands.get(3),
            ) {
                member_offset.insert((*sty, *m), *off);
            }
        }
    }
    let mut chain_defs: HashMap<Word, Instruction> = HashMap::new();
    for block in &ctx.module.functions[entry_idx].blocks {
        for inst in &block.instructions {
            if matches!(inst.class.opcode, Op::AccessChain | Op::InBoundsAccessChain) {
                if let Some(rid) = inst.result_id {
                    chain_defs.insert(rid, inst.clone());
                }
            }
        }
    }

    struct StorePlan {
        bi: usize,
        ii: usize,
        obj: Word,
        obj_ty: Word,
        obj_bits: u32,
        base: Word,
        prefix: Vec<Operand>,
        storage: StorageClass,
        members: Vec<(u32, Word, u32)>, // (member_index, member_ty, member_bits)
    }
    let mut plans: Vec<StorePlan> = Vec::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if inst.class.opcode != Op::Store {
                continue;
            }
            let (Some(Operand::IdRef(ptr)), Some(Operand::IdRef(obj))) =
                (inst.operands.first(), inst.operands.get(1))
            else {
                continue;
            };
            let Some(chain) = chain_defs.get(ptr) else {
                continue;
            };
            let Some(ptr_ty) = chain.result_type else {
                continue;
            };
            let Some(&(storage, pointee)) = ptr_info.get(&ptr_ty) else {
                continue;
            };
            let Some(obj_ty) = value_result_type(ctx, *obj) else {
                continue;
            };
            if obj_ty == pointee {
                continue;
            }
            let Some(obj_bits) = direct_scalar_width(ctx, obj_ty) else {
                continue;
            };
            if obj_bits == 0 || obj_bits % 8 != 0 || obj_bits > 64 {
                continue;
            }
            let Some(Operand::IdRef(base)) = chain.operands.first() else {
                continue;
            };
            let indices: Vec<Operand> = chain.operands[1..].to_vec();
            if indices.len() < 2 {
                continue;
            }
            let Some(base_ptr_ty) = value_result_type(ctx, *base) else {
                continue;
            };
            let Some(&(_, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            let prefix = &indices[..indices.len() - 1];
            let (struct_ty, prefix_consumed) = walk_into_type_partial(ctx, base_pointee, prefix);
            if prefix_consumed != prefix.len() {
                continue;
            }
            let Some(sdef) = type_def_of(ctx, struct_ty) else {
                continue;
            };
            if sdef.class.opcode != Op::TypeStruct {
                continue;
            }
            let Some(Operand::IdRef(member_id)) = indices.last() else {
                continue;
            };
            let Some(member) = const_u32(ctx, *member_id) else {
                continue;
            };
            let Some(&base_off) = member_offset.get(&(struct_ty, member)) else {
                continue;
            };
            let write_lo = base_off;
            let Some(write_hi) = write_lo.checked_add(obj_bits / 8) else {
                continue;
            };
            // The object must tile a run of FULL members starting at the addressed one.
            let mut members: Vec<(u32, Word, u32)> = Vec::new();
            let mut cursor = write_lo;
            let mut ok = true;
            while cursor < write_hi {
                // The member starting exactly at `cursor`.
                let mut hit: Option<(u32, Word, u32)> = None;
                for m in 0..sdef.operands.len() as u32 {
                    if member_offset.get(&(struct_ty, m)).copied() != Some(cursor) {
                        continue;
                    }
                    let Some(Operand::IdRef(mty)) = sdef.operands.get(m as usize) else {
                        continue;
                    };
                    let Some(mbits) = direct_scalar_width(ctx, *mty) else {
                        continue;
                    };
                    hit = Some((m, *mty, mbits));
                    break;
                }
                let Some((m, mty, mbits)) = hit else {
                    ok = false;
                    break;
                };
                if cursor + mbits / 8 > write_hi {
                    ok = false; // a partial member — would clobber bytes outside the write.
                    break;
                }
                members.push((m, mty, mbits));
                cursor += mbits / 8;
            }
            if !ok || cursor != write_hi || members.len() < 2 {
                continue;
            }
            plans.push(StorePlan {
                bi,
                ii,
                obj: *obj,
                obj_ty,
                obj_bits,
                base: *base,
                prefix: prefix.to_vec(),
                storage,
                members,
            });
        }
    }
    if plans.is_empty() {
        return;
    }

    let mut by_block: HashMap<usize, Vec<StorePlan>> = HashMap::new();
    for plan in plans {
        by_block.entry(plan.bi).or_default().push(plan);
    }
    for (bi, mut block_plans) in by_block {
        block_plans.sort_by_key(|p| p.ii);
        let insts = ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .clone();
        let mut out: Vec<Instruction> = Vec::with_capacity(insts.len());
        let mut next = 0usize;
        for (ii, inst) in insts.into_iter().enumerate() {
            if next < block_plans.len() && block_plans[next].ii == ii {
                let plan = &block_plans[next];
                next += 1;
                let obj_uint_ty = uint_type_of_width(ctx, plan.obj_bits);
                let obj_uint = if plan.obj_ty == obj_uint_ty {
                    plan.obj
                } else {
                    let id = ctx.module.fresh_id();
                    out.push(Instruction::new(
                        Op::Bitcast,
                        Some(obj_uint_ty),
                        Some(id),
                        vec![Operand::IdRef(plan.obj)],
                    ));
                    id
                };
                let mut shift = 0u32;
                for &(m, mty, mbits) in &plan.members {
                    let member_uint_ty = uint_type_of_width(ctx, mbits);
                    // Extract this member's bits from the object.
                    let shifted = if shift == 0 {
                        obj_uint
                    } else {
                        let amt = ctx.const_uint(shift);
                        let id = ctx.module.fresh_id();
                        out.push(Instruction::new(
                            Op::ShiftRightLogical,
                            Some(obj_uint_ty),
                            Some(id),
                            vec![Operand::IdRef(obj_uint), Operand::IdRef(amt)],
                        ));
                        id
                    };
                    let narrowed = if plan.obj_bits == mbits {
                        shifted
                    } else {
                        let id = ctx.module.fresh_id();
                        out.push(Instruction::new(
                            Op::UConvert,
                            Some(member_uint_ty),
                            Some(id),
                            vec![Operand::IdRef(shifted)],
                        ));
                        id
                    };
                    let val = if mty == member_uint_ty {
                        narrowed
                    } else {
                        let id = ctx.module.fresh_id();
                        out.push(Instruction::new(
                            Op::Bitcast,
                            Some(mty),
                            Some(id),
                            vec![Operand::IdRef(narrowed)],
                        ));
                        id
                    };
                    let member_idx_id = ctx.const_uint(m);
                    let ptr_ty = ctx.ty_ptr(plan.storage, mty);
                    let mptr = ctx.module.fresh_id();
                    let mut ops = Vec::with_capacity(plan.prefix.len() + 2);
                    ops.push(Operand::IdRef(plan.base));
                    ops.extend(plan.prefix.iter().cloned());
                    ops.push(Operand::IdRef(member_idx_id));
                    out.push(Instruction::new(
                        Op::InBoundsAccessChain,
                        Some(ptr_ty),
                        Some(mptr),
                        ops,
                    ));
                    out.push(Instruction::new(
                        Op::Store,
                        None,
                        None,
                        vec![Operand::IdRef(mptr), Operand::IdRef(val)],
                    ));
                    shift += mbits;
                }
                continue;
            }
            out.push(inst);
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = out;
    }
}

/// Lower an `OpStore` of a wider scalar or homogeneous vector through a NARROWER scalar element
/// pointer (a byte/sub-word reinterpret store) into per-element little-endian stores. A byte-addressed
/// buffer write — `OpStore %844 %719`
/// with `%844 : _ptr…_uchar` (a `device uchar*` element pointer, e.g. from an `OpPtrAccessChain`) and
/// `%719 : uint` — mismatches the pointee (spirv-val "OpStore Pointer's type does not match Object's
/// type"). The emitter has no logical pointer bitcast to widen the byte pointer; the byte-correct
/// lowering splits the object into `obj_bits / pointee_bits` little-endian slots and stores each
/// through a sibling element pointer (`OpPtrAccessChain ptr k`), which is valid for an element pointer
/// in a PtrAccessChain-capable storage class.
///
/// Byte-EXACT by construction on a little-endian target (slot `k` of the object lands at byte `k` of
/// the pointer's element address — exactly where a native wide store would place it) AND floor-SAFE by
/// construction: only fires on a store whose object type already MISMATCHES the pointee (a valid/banked
/// module's stores match their pointee, so it never matches), and only for a scalar/vector object an
/// integral multiple wider than a direct-scalar pointee in a PtrAccessChain-capable storage class. The
/// pointer must be either an explicit `OpPtrAccessChain` element pointer or an access-chain result whose
/// final index enters an array/runtime-array element; bare pointer parameters and struct-member scalars
/// stay untouched.
pub(in crate::passes) fn lower_subword_scalar_store(ctx: &mut Ctx, entry_idx: usize) {
    let mut ptr_info: HashMap<Word, (StorageClass, Word)> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if inst.class.opcode == Op::TypePointer {
            if let (Some(id), Some(Operand::StorageClass(s)), Some(Operand::IdRef(p))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*s, *p));
            }
        }
    }

    // (bi, ii, ptr, storage, pointee_ty, pointee_bits, obj, obj_ty, obj_bits)
    struct StorePlan {
        bi: usize,
        ii: usize,
        ptr: Word,
        storage: StorageClass,
        pointee_ty: Word,
        pointee_bits: u32,
        obj: Word,
        obj_ty: Word,
        obj_bits: u32,
    }
    let mut plans: Vec<StorePlan> = Vec::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if inst.class.opcode != Op::Store {
                continue;
            }
            let (Some(Operand::IdRef(ptr)), Some(Operand::IdRef(obj))) =
                (inst.operands.first(), inst.operands.get(1))
            else {
                continue;
            };
            let Some(ptr_ty) = value_result_type(ctx, *ptr) else {
                continue;
            };
            let Some(&(storage, pointee_ty)) = ptr_info.get(&ptr_ty) else {
                continue;
            };
            if !ptr_access_chain_allowed_storage(storage) {
                continue;
            }
            let ptr_is_element = find_def_in_func(&ctx.module.functions[entry_idx], *ptr)
                .map(|d| store_split_pointer_is_element(ctx, &d, pointee_ty))
                .unwrap_or(false);
            if !ptr_is_element {
                continue;
            }
            let Some(obj_ty) = value_result_type(ctx, *obj) else {
                continue;
            };
            if obj_ty == pointee_ty {
                continue; // a matched store — a valid module lands here; never touched.
            }
            let Some(pointee_bits) = direct_scalar_width(ctx, pointee_ty) else {
                continue;
            };
            let Some(obj_bits) = direct_scalar_or_vector_width(ctx, obj_ty) else {
                continue;
            };
            if pointee_bits == 0
                || obj_bits <= pointee_bits
                || obj_bits % pointee_bits != 0
                || obj_bits > 64
            {
                continue;
            }
            plans.push(StorePlan {
                bi,
                ii,
                ptr: *ptr,
                storage,
                pointee_ty,
                pointee_bits,
                obj: *obj,
                obj_ty,
                obj_bits,
            });
        }
    }
    if plans.is_empty() {
        return;
    }

    let mut by_block: HashMap<usize, Vec<StorePlan>> = HashMap::new();
    for plan in plans {
        by_block.entry(plan.bi).or_default().push(plan);
    }
    for (bi, mut block_plans) in by_block {
        block_plans.sort_by_key(|p| p.ii);
        let insts = ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .clone();
        let mut out: Vec<Instruction> = Vec::with_capacity(insts.len());
        let mut next = 0usize;
        for (ii, inst) in insts.into_iter().enumerate() {
            if next < block_plans.len() && block_plans[next].ii == ii {
                let plan = &block_plans[next];
                next += 1;
                let slots = plan.obj_bits / plan.pointee_bits;
                let obj_uint_ty = uint_type_of_width(ctx, plan.obj_bits);
                let pointee_uint_ty = uint_type_of_width(ctx, plan.pointee_bits);
                let ptr_ty = ctx.ty_ptr(plan.storage, plan.pointee_ty);
                // Reinterpret the object to its same-width unsigned int once.
                let obj_uint = if plan.obj_ty == obj_uint_ty {
                    plan.obj
                } else {
                    let id = ctx.module.fresh_id();
                    out.push(Instruction::new(
                        Op::Bitcast,
                        Some(obj_uint_ty),
                        Some(id),
                        vec![Operand::IdRef(plan.obj)],
                    ));
                    id
                };
                for k in 0..slots {
                    // Slot k's bits, little-endian.
                    let shifted = if k == 0 {
                        obj_uint
                    } else {
                        let amt = ctx.const_uint(k * plan.pointee_bits);
                        let id = ctx.module.fresh_id();
                        out.push(Instruction::new(
                            Op::ShiftRightLogical,
                            Some(obj_uint_ty),
                            Some(id),
                            vec![Operand::IdRef(obj_uint), Operand::IdRef(amt)],
                        ));
                        id
                    };
                    // Narrow to the pointee width (truncate to the low pointee_bits).
                    let narrowed = if plan.obj_bits == plan.pointee_bits {
                        shifted
                    } else {
                        let id = ctx.module.fresh_id();
                        out.push(Instruction::new(
                            Op::UConvert,
                            Some(pointee_uint_ty),
                            Some(id),
                            vec![Operand::IdRef(shifted)],
                        ));
                        id
                    };
                    // Reinterpret to the pointee's declared type if it isn't that unsigned int.
                    let val = if plan.pointee_ty == pointee_uint_ty {
                        narrowed
                    } else {
                        let id = ctx.module.fresh_id();
                        out.push(Instruction::new(
                            Op::Bitcast,
                            Some(plan.pointee_ty),
                            Some(id),
                            vec![Operand::IdRef(narrowed)],
                        ));
                        id
                    };
                    // Sibling element pointer at +k (the chain pointer itself for k == 0).
                    let slot_ptr = if k == 0 {
                        plan.ptr
                    } else {
                        let kid = ctx.const_uint(k);
                        let id = ctx.module.fresh_id();
                        out.push(Instruction::new(
                            Op::PtrAccessChain,
                            Some(ptr_ty),
                            Some(id),
                            vec![Operand::IdRef(plan.ptr), Operand::IdRef(kid)],
                        ));
                        id
                    };
                    out.push(Instruction::new(
                        Op::Store,
                        None,
                        None,
                        vec![Operand::IdRef(slot_ptr), Operand::IdRef(val)],
                    ));
                }
                continue;
            }
            out.push(inst);
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = out;
    }
}

fn store_split_pointer_is_element(ctx: &Ctx, inst: &Instruction, pointee_ty: Word) -> bool {
    match inst.class.opcode {
        Op::PtrAccessChain => true,
        Op::AccessChain | Op::InBoundsAccessChain => {
            access_chain_final_index_enters_array_element(ctx, inst, pointee_ty)
        }
        _ => false,
    }
}

fn access_chain_final_index_enters_array_element(
    ctx: &Ctx,
    inst: &Instruction,
    pointee_ty: Word,
) -> bool {
    let Some(Operand::IdRef(base)) = inst.operands.first() else {
        return false;
    };
    let Some(base_ty) = value_result_type(ctx, *base) else {
        return false;
    };
    let Some(mut cur) = pointer_pointee(ctx, base_ty) else {
        return false;
    };
    let indices = &inst.operands[1..];
    let Some((last, prefix)) = indices.split_last() else {
        return false;
    };
    for op in prefix {
        let Some(def) = type_def_of(ctx, cur) else {
            return false;
        };
        cur = match def.class.opcode {
            Op::TypeStruct => {
                let Operand::IdRef(idx_id) = op else {
                    return false;
                };
                let Some(member) = const_u32(ctx, *idx_id) else {
                    return false;
                };
                match def.operands.get(member as usize) {
                    Some(Operand::IdRef(member_ty)) => *member_ty,
                    _ => return false,
                }
            }
            Op::TypeArray | Op::TypeRuntimeArray | Op::TypeVector | Op::TypeMatrix => {
                match def.operands.first() {
                    Some(Operand::IdRef(elem_ty)) => *elem_ty,
                    _ => return false,
                }
            }
            _ => return false,
        };
    }
    if !matches!(last, Operand::IdRef(_)) {
        return false;
    }
    let Some(def) = type_def_of(ctx, cur) else {
        return false;
    };
    if !matches!(def.class.opcode, Op::TypeArray | Op::TypeRuntimeArray) {
        return false;
    }
    match def.operands.first() {
        Some(Operand::IdRef(elem_ty)) => *elem_ty == pointee_ty,
        _ => false,
    }
}

pub(in crate::passes) fn direct_scalar_or_vector_width(ctx: &Ctx, ty: Word) -> Option<u32> {
    if let Some(bits) = direct_scalar_width(ctx, ty) {
        return Some(bits);
    }
    let vector = type_def_of(ctx, ty)?;
    if vector.class.opcode != Op::TypeVector {
        return None;
    }
    let (Some(Operand::IdRef(element)), Some(Operand::LiteralBit32(lanes))) =
        (vector.operands.first(), vector.operands.get(1))
    else {
        return None;
    };
    direct_scalar_width(ctx, *element)?.checked_mul(*lanes)
}

/// Find (or synthesize) the `OpTypeInt <bits> 0` (unsigned) type. For the widths these passes use
/// (8/16/32/64) the type already exists when the module loads that scalar — but synthesize as a
/// fallback so the helper is total. (Int8/Int16 carry capabilities the module already declares when it
/// has a uchar/ushort member, the only way a sub-word reinterpret arises.)
pub(in crate::passes) fn uint_type_of_width(ctx: &mut Ctx, bits: u32) -> Word {
    for inst in ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
    {
        if inst.class.opcode == Op::TypeInt
            && inst.operands.first() == Some(&Operand::LiteralBit32(bits))
            && inst.operands.get(1) == Some(&Operand::LiteralBit32(0))
        {
            if let Some(id) = inst.result_id {
                return id;
            }
        }
    }
    let id = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::TypeInt,
        None,
        Some(id),
        vec![Operand::LiteralBit32(bits), Operand::LiteralBit32(0)],
    ));
    id
}

/// One member's contribution to an assembled cross-member sub-word read. All fields are owned
/// scalars/Words computed in the read-only scan, so the mutating apply phase is mechanical.
pub(in crate::passes) struct SubwordPart {
    pub(in crate::passes) member_index: u32,
    pub(in crate::passes) member_offset: u32,
    pub(in crate::passes) member_ty: Word,
    pub(in crate::passes) member_bits: u32,
    pub(in crate::passes) shift_right: u32, // bits to drop off the bottom of the member before taking our bytes
    pub(in crate::passes) keep_bits: u32, // number of low bits that belong to this read after the right-shift
    pub(in crate::passes) shift_left: u32, // bit position of those bits within the result
}

/// Plan for rewriting one width-mismatched `OpLoad` into a cross-member byte assembly.
pub(in crate::passes) struct SubwordPlan {
    pub(in crate::passes) bi: usize,
    pub(in crate::passes) ii: usize,
    pub(in crate::passes) result_id: Word,
    pub(in crate::passes) result_ty: Word,
    pub(in crate::passes) result_bits: u32,
    pub(in crate::passes) base: Word,
    pub(in crate::passes) prefix: Vec<Operand>,
    pub(in crate::passes) storage: StorageClass,
    pub(in crate::passes) parts: Vec<SubwordPart>,
    pub(in crate::passes) exact_buffer_offset: Option<(Word, u64)>,
    pub(in crate::passes) addressed_member_offset: u32,
}

/// Lower a width-mismatched `OpLoad` of a wider scalar through a struct-MEMBER pointer into a
/// little-endian assembly of the members the read spans. This generalizes
/// [`remap_word_index_to_struct_member`] (a same-type byte-address remap) to a SUB-WORD read that
/// CROSSES member boundaries: an MPS/byte-addressed buffer reflected as a packed struct
/// (`{uint@0,float@4,uchar@8,uchar@9,ushort@10}`) reads e.g. a `ushort` at byte 9 — spanning the
/// `uchar` field-3 byte and the low byte of the `ushort` field-4 — so the emitter's faithful
/// `OpLoad %ushort` through the `_ptr…_uchar` member pointer mismatches the pointee
/// (spirv-val "OpLoad Result Type … does not match Pointer's type").
///
/// The fix reads each member whose byte range overlaps `[B, B+W)` (B = the addressed member's
/// `Offset`, W = the load's byte width), reinterprets it to its same-width unsigned int, masks+shifts
/// its contributing bytes to their position in the result, and ORs them — then bitcasts the assembled
/// unsigned int to the load's result type. Byte-EXACT by construction on a little-endian target: the
/// `Offset` decorations are the physical std430 layout the golden was captured with (the same address
/// oracle the word-index remap relies on), so the assembled value is exactly the W contiguous bytes at
/// byte B. Floor-SAFE by construction: only fires on a load whose result type already MISMATCHES the
/// member pointee (a valid/banked module's loads match their pointee, so it never matches), and only
/// when the spanned members EXACTLY tile `[B, B+W)` (no padding gap) and are all direct scalars — any
/// other shape leaves the load untouched.
pub(in crate::passes) fn lower_cross_member_subword_load(
    ctx: &mut Ctx,
    entry_idx: usize,
) -> Result<(), String> {
    let mut ptr_info: HashMap<Word, (StorageClass, Word)> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if inst.class.opcode == Op::TypePointer {
            if let (Some(id), Some(Operand::StorageClass(s)), Some(Operand::IdRef(p))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*s, *p));
            }
        }
    }
    let mut member_offset: HashMap<(Word, u32), u32> = HashMap::new();
    for inst in &ctx.module.annotations {
        if inst.class.opcode == Op::MemberDecorate {
            if let (
                Some(Operand::IdRef(sty)),
                Some(Operand::LiteralBit32(m)),
                Some(Operand::Decoration(Decoration::Offset)),
                Some(Operand::LiteralBit32(off)),
            ) = (
                inst.operands.first(),
                inst.operands.get(1),
                inst.operands.get(2),
                inst.operands.get(3),
            ) {
                member_offset.insert((*sty, *m), *off);
            }
        }
    }
    // Pre-scan every access chain so a load can recover its pointer's (base, indices) without a borrow.
    let mut chain_defs: HashMap<Word, Instruction> = HashMap::new();
    for block in &ctx.module.functions[entry_idx].blocks {
        for inst in &block.instructions {
            if matches!(inst.class.opcode, Op::AccessChain | Op::InBoundsAccessChain) {
                if let Some(rid) = inst.result_id {
                    chain_defs.insert(rid, inst.clone());
                }
            }
        }
    }
    let types = crate::passes::resources::rewrites::combined_type_defs(ctx, &HashMap::new());
    let value_types = crate::passes::resources::rewrites::combined_value_types(ctx, entry_idx);
    let exact_offsets = ctx
        .emit_sidecar
        .buffer_access_offsets
        .iter()
        .map(|fact| (fact.id, (fact.root, fact.byte_offset)))
        .collect::<HashMap<_, _>>();

    let mut plans: Vec<SubwordPlan> = Vec::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if inst.class.opcode != Op::Load {
                continue;
            }
            let (Some(result_ty), Some(result_id)) = (inst.result_type, inst.result_id) else {
                continue;
            };
            let Some(Operand::IdRef(ptr)) = inst.operands.first() else {
                continue;
            };
            // The pointer must be an access chain whose declared pointee MISMATCHES the load result.
            let Some(chain) = chain_defs.get(ptr) else {
                continue;
            };
            let Some(ptr_ty) = chain.result_type else {
                continue;
            };
            let Some(&(storage, pointee)) = ptr_info.get(&ptr_ty) else {
                continue;
            };
            if pointee == result_ty {
                continue; // a matched load — a valid module's every load lands here; never touched.
            }
            // The result must be a direct scalar no wider than 32 bits (mask constants stay in u32).
            let Some(result_bits) = direct_scalar_width(ctx, result_ty) else {
                continue;
            };
            if result_bits == 0 || result_bits % 8 != 0 || result_bits > 32 {
                continue;
            }
            // The chain's member prefix must walk cleanly to a struct; the trailing index is the
            // addressed member.
            let Some(Operand::IdRef(base)) = chain.operands.first() else {
                continue;
            };
            let indices: Vec<Operand> = chain.operands[1..].to_vec();
            if indices.len() < 2 {
                continue; // need a prefix into the buffer struct plus a member index.
            }
            let Some(base_ptr_ty) = value_result_type(ctx, *base) else {
                continue;
            };
            let Some(&(_, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            let prefix = &indices[..indices.len() - 1];
            let (struct_ty, prefix_consumed) = walk_into_type_partial(ctx, base_pointee, prefix);
            if prefix_consumed != prefix.len() {
                continue;
            }
            let Some(sdef) = type_def_of(ctx, struct_ty) else {
                continue;
            };
            if sdef.class.opcode != Op::TypeStruct {
                continue;
            }
            let Some(Operand::IdRef(member_id)) = indices.last() else {
                continue;
            };
            let Some(member) = const_u32(ctx, *member_id) else {
                continue;
            };
            let Some(&base_off) = member_offset.get(&(struct_ty, member)) else {
                continue;
            };
            let read_lo = base_off;
            let Some(read_hi) = read_lo.checked_add(result_bits / 8) else {
                continue;
            };
            // Collect every member whose byte range overlaps [read_lo, read_hi); they must EXACTLY
            // tile it (no gap), each a direct scalar.
            let mut covered: Vec<(u32, Word, u32, u32)> = Vec::new(); // (member, ty, off, size_bytes)
            for m in 0..sdef.operands.len() as u32 {
                let Some(&off) = member_offset.get(&(struct_ty, m)) else {
                    continue;
                };
                let Some(Operand::IdRef(mty)) = sdef.operands.get(m as usize) else {
                    continue;
                };
                let Some(mbits) = direct_scalar_width(ctx, *mty) else {
                    continue;
                };
                let msize = mbits / 8;
                if msize == 0 {
                    continue;
                }
                if off < read_hi && off + msize > read_lo {
                    covered.push((m, *mty, off, msize));
                }
            }
            if covered.len() < 2 {
                continue; // a same-member reinterpret — left to the dedicated handlers.
            }
            covered.sort_by_key(|c| c.2);
            // Contiguity: the spanned members must tile [read_lo, read_hi) with no padding gap.
            let mut cursor = covered[0].2;
            if cursor > read_lo {
                continue;
            }
            let mut tiled = true;
            for &(_, _, off, msize) in &covered {
                if off != cursor {
                    tiled = false;
                    break;
                }
                cursor = off + msize;
            }
            if !tiled || cursor < read_hi {
                continue;
            }
            let mut parts: Vec<SubwordPart> = Vec::new();
            for &(m, mty, off, msize) in &covered {
                let lo = read_lo.max(off);
                let hi = read_hi.min(off + msize);
                parts.push(SubwordPart {
                    member_index: m,
                    member_offset: off,
                    member_ty: mty,
                    member_bits: msize * 8,
                    shift_right: (lo - off) * 8,
                    keep_bits: (hi - lo) * 8,
                    shift_left: (lo - read_lo) * 8,
                });
            }
            plans.push(SubwordPlan {
                bi,
                ii,
                result_id,
                result_ty,
                result_bits,
                base: *base,
                prefix: prefix.to_vec(),
                storage,
                parts,
                exact_buffer_offset: inherited_exact_byte_offset(
                    *ptr,
                    &exact_offsets,
                    &chain_defs,
                    &types,
                    &value_types,
                    &mut HashSet::new(),
                ),
                addressed_member_offset: read_lo,
            });
        }
    }

    if plans.is_empty() {
        return Ok(());
    }

    // Apply: rebuild each affected block, splicing the assembly in place of the planned load.
    let mut by_block: HashMap<usize, Vec<SubwordPlan>> = HashMap::new();
    for plan in plans {
        by_block.entry(plan.bi).or_default().push(plan);
    }
    for (bi, mut block_plans) in by_block {
        block_plans.sort_by_key(|p| p.ii);
        let insts = ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .clone();
        let mut out: Vec<Instruction> = Vec::with_capacity(insts.len());
        let mut next = 0usize;
        for (ii, inst) in insts.into_iter().enumerate() {
            if next < block_plans.len() && block_plans[next].ii == ii {
                let plan = &block_plans[next];
                next += 1;
                emit_subword_assembly(ctx, plan, &mut out)?;
                continue;
            }
            out.push(inst);
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = out;
    }
    Ok(())
}

/// Emit the byte-assembly sequence for one [`SubwordPlan`] into `out`, with the final value carrying
/// the original load's result id.
pub(in crate::passes) fn emit_subword_assembly(
    ctx: &mut Ctx,
    plan: &SubwordPlan,
    out: &mut Vec<Instruction>,
) -> Result<(), String> {
    let result_uint = uint_type_of_width(ctx, plan.result_bits);
    let mut acc: Option<Word> = None;
    for part in &plan.parts {
        // Sibling member pointer (same base + prefix, this member's index).
        let member_idx_id = ctx.const_uint(part.member_index);
        let ptr_ty = ctx.ty_ptr(plan.storage, part.member_ty);
        let ptr = ctx.module.fresh_id();
        let mut ops = Vec::with_capacity(plan.prefix.len() + 2);
        ops.push(Operand::IdRef(plan.base));
        ops.extend(plan.prefix.iter().cloned());
        ops.push(Operand::IdRef(member_idx_id));
        out.push(Instruction::new(
            Op::InBoundsAccessChain,
            Some(ptr_ty),
            Some(ptr),
            ops,
        ));
        if let Some((root, addressed_offset)) = plan.exact_buffer_offset {
            let relative = i64::from(part.member_offset) - i64::from(plan.addressed_member_offset);
            if let Some(byte_offset) = addressed_offset.checked_add_signed(relative) {
                ctx.emit_sidecar.buffer_access_offsets.push(
                    crate::emit_sidecar::BufferAccessOffset {
                        id: ptr,
                        root,
                        byte_offset,
                    },
                );
            }
        }
        // Load and reinterpret to the member's same-width unsigned int.
        let loaded = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::Load,
            Some(part.member_ty),
            Some(loaded),
            vec![Operand::IdRef(ptr)],
        ));
        let member_uint = uint_type_of_width(ctx, part.member_bits);
        let mut cur = if part.member_ty == member_uint {
            loaded
        } else {
            let id = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::Bitcast,
                Some(member_uint),
                Some(id),
                vec![Operand::IdRef(loaded)],
            ));
            id
        };
        // Drop the low bytes that precede our overlap (in the member's own width).
        if part.shift_right > 0 {
            let amt = ctx.const_uint(part.shift_right);
            let id = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::ShiftRightLogical,
                Some(member_uint),
                Some(id),
                vec![Operand::IdRef(cur), Operand::IdRef(amt)],
            ));
            cur = id;
        }
        // Convert to the result width (zero-extend if narrower, truncate if wider).
        if part.member_bits != plan.result_bits {
            let id = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::UConvert,
                Some(result_uint),
                Some(id),
                vec![Operand::IdRef(cur)],
            ));
            cur = id;
        }
        // Keep only this member's bytes.
        if part.keep_bits < plan.result_bits {
            let mask = if part.keep_bits >= 32 {
                u32::MAX
            } else {
                (1u32 << part.keep_bits) - 1
            };
            let mask_id = ctx.const_int_of(result_uint, mask as i64);
            let id = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::BitwiseAnd,
                Some(result_uint),
                Some(id),
                vec![Operand::IdRef(cur), Operand::IdRef(mask_id)],
            ));
            cur = id;
        }
        // Place the bytes at their position in the result.
        if part.shift_left > 0 {
            let amt = ctx.const_uint(part.shift_left);
            let id = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::ShiftLeftLogical,
                Some(result_uint),
                Some(id),
                vec![Operand::IdRef(cur), Operand::IdRef(amt)],
            ));
            cur = id;
        }
        acc = Some(match acc {
            None => cur,
            Some(prev) => {
                let id = ctx.module.fresh_id();
                out.push(Instruction::new(
                    Op::BitwiseOr,
                    Some(result_uint),
                    Some(id),
                    vec![Operand::IdRef(prev), Operand::IdRef(cur)],
                ));
                id
            }
        });
    }
    let packed = acc.ok_or("covered.len() >= 2")?;
    // Bind the assembled unsigned int to the load's result id, bitcasting to the declared type if it
    // is not already that unsigned int (a float/half/signed result of the same width).
    let op = if plan.result_ty == result_uint {
        Op::CopyObject
    } else {
        Op::Bitcast
    };
    out.push(Instruction::new(
        op,
        Some(plan.result_ty),
        Some(plan.result_id),
        vec![Operand::IdRef(packed)],
    ));
    Ok(())
}

/// The defining instruction of value `id` within `func` (a cloned snapshot — callers inspect opcode
/// and operands without juggling a borrow against the surrounding mutation).
pub(in crate::passes) fn find_def_in_func(func: &Function, id: Word) -> Option<Instruction> {
    for b in &func.blocks {
        for i in &b.instructions {
            if i.result_id == Some(id) {
                return Some(i.clone());
            }
        }
    }
    None
}

/// The id of operand `n` of `inst`, when it is an `IdRef`.
pub(in crate::passes) fn operand_id(inst: &Instruction, n: usize) -> Option<Word> {
    match inst.operands.get(n) {
        Some(Operand::IdRef(x)) => Some(*x),
        _ => None,
    }
}

/// True when `inst` is `OpIMul(v, k)` (in either operand order) with a constant operand equal to `w`.
pub(in crate::passes) fn imul_by(ctx: &Ctx, inst: &Instruction, w: u32) -> bool {
    inst.class.opcode == Op::IMul
        && (0..2).any(|n| operand_id(inst, n).and_then(|x| const_u32(ctx, x)) == Some(w))
}

/// True when value `idx` is a flat-WORD-addressed index with element stride `w` words: a constant word
/// offset, `OpIMul(elem, w)`, or `OpIAdd(const, OpIMul(elem, w))` (either operand order). A `w`-strided
/// element index proves the address is `byte = 4*(const + elem*w)` — i.e. word-granular addressing of
/// the whole variable, not an into-element index. Any other shape returns false (the caller then leaves
/// the variable untouched rather than guess a layout).
pub(in crate::passes) fn index_is_word_addressed(
    ctx: &Ctx,
    func: &Function,
    idx: Word,
    w: u32,
) -> bool {
    if const_u32(ctx, idx).is_some() {
        return true;
    }
    let Some(def) = find_def_in_func(func, idx) else {
        return false;
    };
    match def.class.opcode {
        Op::IMul => imul_by(ctx, &def, w),
        Op::IAdd => {
            let a = operand_id(&def, 0);
            let b = operand_id(&def, 1);
            let a_const = a.and_then(|x| const_u32(ctx, x)).is_some();
            let b_const = b.and_then(|x| const_u32(ctx, x)).is_some();
            let a_mul = a
                .and_then(|x| find_def_in_func(func, x))
                .map(|d| imul_by(ctx, &d, w))
                .unwrap_or(false);
            let b_mul = b
                .and_then(|x| find_def_in_func(func, x))
                .map(|d| imul_by(ctx, &d, w))
                .unwrap_or(false);
            (a_const && b_mul) || (b_const && a_mul)
        }
        _ => false,
    }
}
