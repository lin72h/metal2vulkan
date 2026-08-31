//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

/// Trace `base` through `OpPhi` (all incoming values must converge), `OpCopyObject`, and element-0
/// access chains to the single array variable `base` is the element-0 pointer of. Returns the array
/// variable id when `base` provably equals `&array[0]` for ONE `[K x elem_scalar]` array of storage
/// class `sc`; `None` if provenance diverges to a different array, hits a non-element-0 chain, or any
/// leaf is an unknown definition.
pub(in crate::passes) fn trace_to_array_element_zero(
    ctx: &Ctx,
    func: &Function,
    base: Word,
    elem_scalar: Word,
    sc: StorageClass,
    ptr_info: &HashMap<Word, (StorageClass, Word)>,
) -> Option<Word> {
    let mut visited: HashSet<Word> = HashSet::new();
    let mut stack = vec![base];
    let mut found: Option<Word> = None;
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let def = find_def_in_func(func, id)?;
        match def.class.opcode {
            Op::Phi => {
                // Incoming (value, label) pairs; values sit at even operand positions.
                let mut k = 0;
                while k < def.operands.len() {
                    if let Some(Operand::IdRef(v)) = def.operands.get(k) {
                        stack.push(*v);
                    }
                    k += 2;
                }
            }
            Op::CopyObject => {
                let Some(Operand::IdRef(v)) = def.operands.first() else {
                    return None;
                };
                stack.push(*v);
            }
            Op::InBoundsAccessChain | Op::AccessChain => {
                // Must be exactly `arr[0]` — a single const-0 index into a `[K x elem_scalar]` array
                // of the SAME storage class.
                if def.operands.len() != 2 {
                    return None;
                }
                let Some(Operand::IdRef(arr)) = def.operands.first() else {
                    return None;
                };
                let Some(Operand::IdRef(idx)) = def.operands.get(1) else {
                    return None;
                };
                if const_u32(ctx, *idx) != Some(0) {
                    return None;
                }
                let arr_ptr_ty = value_result_type(ctx, *arr)?;
                let &(arr_sc, arr_pointee) = ptr_info.get(&arr_ptr_ty)?;
                if arr_sc != sc || array_element_type(ctx, arr_pointee) != Some(elem_scalar) {
                    return None;
                }
                match found {
                    Some(prev) if prev != *arr => return None,
                    _ => found = Some(*arr),
                }
            }
            _ => return None,
        }
    }
    found
}

/// Remap a raw WORD index that lands on a typed-struct binding back to the MEMBER index at the same
/// byte offset. The R4 raw byte-model (`emitter/memory.rs`, byte-offset `div_euclid(4)`) emits a
/// buffer access as `%buf <member-prefix> %uint_W` where the LAST index `W` is a WORD index (4-byte
/// stride) into the buffer's flat byte space. When the SAME buffer is ALSO accessed via typed member
/// GEPs, `build_stage_input` reconstructs the reflected typed struct as the binding type rather than
/// the
/// raw `{ RuntimeArray<uint> }` transport block (it sees struct-path chains and keeps the struct), so
/// the flat word index `W` is OUT OF BOUNDS for — or mistyped against — the struct's MEMBER indices
/// (spirv-val "OpInBoundsAccessChain cannot find index W into the structure ... has N members"). This
/// is the dual-use `FullChainConversionParams`-style family: a `{<4 x float>, <2 x i16>x3, i32x4}`
/// uniform read both via typed member GEPs and via a word-aligned reinterpret. Because the
/// reconstructed struct carries byte `Offset` decorations, word `W` (= byte `4W`) maps unambiguously
/// to the member `M` with `Offset[M] == 4W`; when member `M`'s type equals the access result pointee
/// the remap `W -> M` is byte-EXACT (identical byte address, identical scalar), reconciling the
/// dual-use buffer's word-index and member-index chains onto the one typed-struct binding.
///
/// Byte-safe / floor-safe by construction: only chains that are CURRENTLY INVALID (walking every index
/// over-runs the member count or reaches a type != the declared result pointee) are touched, the
/// member-prefix must walk cleanly to a STRUCT, and the remapped member must sit at EXACTLY byte `4W`
/// AND carry the result-pointee type (else bail) — so a banked/valid module (every member index in
/// range and correctly typed) never matches and the floor is provably unchanged. The remap selects the
/// identical byte address with the identical scalar type, so it is a byte no-op, not a reinterpret.
/// Decides purely from IR structure (type walk + member `Offset` decorations), never a shader name.
pub(in crate::passes) fn remap_word_index_to_struct_member(ctx: &mut Ctx, entry_idx: usize) {
    let value_types = function_value_types(ctx, entry_idx);
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
    // Struct member byte offsets — the address oracle for the word->member remap.
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

    let mut edits: Vec<(usize, usize, u32)> = Vec::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if !matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain) {
                continue;
            }
            let Some(result_type) = inst.result_type else {
                continue;
            };
            let Some(&(_, result_pointee)) = ptr_info.get(&result_type) else {
                continue;
            };
            let Some(Operand::IdRef(base)) = inst.operands.first() else {
                continue;
            };
            let indices: Vec<Operand> = inst.operands[1..].to_vec();
            if indices.is_empty() {
                continue;
            }
            let Some(base_ptr_ty) = value_types.get(base).copied() else {
                continue;
            };
            let Some(&(_, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            // Floor-safe gate: only currently-INVALID chains. A valid chain walks every index and
            // reaches exactly the declared result pointee — never touched.
            let (reached, consumed) = walk_into_type_partial(ctx, base_pointee, &indices);
            if consumed == indices.len() && reached == result_pointee {
                continue;
            }
            // The member-PREFIX (all but the trailing word index) must walk cleanly to a struct.
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
            // The trailing index must be a constant WORD index.
            let Some(Operand::IdRef(last_id)) = indices.last() else {
                continue;
            };
            let Some(word) = const_u32(ctx, *last_id) else {
                continue;
            };
            let Some(byte) = word.checked_mul(4) else {
                continue;
            };
            // The unique member sitting at EXACTLY byte 4W whose type is the result pointee.
            let mut found: Option<u32> = None;
            for m in 0..sdef.operands.len() {
                if member_offset.get(&(struct_ty, m as u32)).copied() != Some(byte) {
                    continue;
                }
                let Some(Operand::IdRef(mty)) = sdef.operands.get(m) else {
                    continue;
                };
                if *mty != result_pointee {
                    continue;
                }
                if found.is_some() {
                    found = None; // ambiguous — bail rather than guess
                    break;
                }
                found = Some(m as u32);
            }
            let Some(member) = found else {
                continue;
            };
            if member == word {
                continue;
            }
            edits.push((bi, ii, member));
        }
    }

    for (bi, ii, member) in edits {
        let member_id = ctx.const_uint(member);
        if let Some(last) = ctx.module.functions[entry_idx].blocks[bi].instructions[ii]
            .operands
            .last_mut()
        {
            *last = Operand::IdRef(member_id);
        }
    }
}

/// Remap a raw WORD index that OVERFLOWS the member-0 sub-struct it descended into, back to the
/// SIBLING top-level member of the OUTER buffer struct sitting at the same absolute byte offset.
///
/// This is the two-level generalization of [`remap_word_index_to_struct_member`]. The R4 raw
/// byte-model computes a buffer-relative flat WORD index and the emitter lowers it as
/// `%buf <const-prefix> %uint_W` where the prefix descends into an inner aggregate member (e.g.
/// `%buf %uint_0` into member 0 — a sub-struct at offset 0) and the trailing word `W` is the flat
/// index from the buffer base. When `W` over-runs the inner sub-struct's member count, the existing
/// single-level remap (which resolves byte `4W` against the PREFIX struct) bails — but the AIR's
/// flat index is BUFFER-relative, so byte `prefix_byte + 4W` actually lands on a SIBLING member of
/// the OUTER struct (e.g. a trailing scalar at a high byte offset, well past member 0). When that
/// absolute byte offset coincides EXACTLY with one top-level member `M` of the base buffer struct
/// whose type equals the access result pointee, the whole index list collapses to `%buf %uint_M` —
/// the identical byte address with the identical scalar, a byte no-op reconciling the dual-use
/// buffer's word-index chain onto the typed-struct binding.
///
/// Byte-safe / floor-safe by construction: fires ONLY on a CURRENTLY-INVALID chain (the full walk
/// over-runs before consuming every index), whose member-PREFIX is a non-empty run of CONSTANT
/// indices walking cleanly through structs/arrays to a known accumulated byte offset, whose trailing
/// index is a CONSTANT word, and only when the absolute byte offset `prefix_byte + 4W` matches the
/// `Offset` decoration of EXACTLY ONE top-level member of the base buffer struct AND that member's
/// type equals the result pointee. Any ambiguity, a non-constant index, a non-struct base pointee, or
/// a miss leaves the chain untouched — a banked/valid module never over-indexes, so the floor is
/// provably unchanged. Decides purely from IR structure (type walk + member `Offset`/`ArrayStride`
/// decorations), never a shader name.
pub(in crate::passes) fn remap_overflow_word_index_to_outer_member(
    ctx: &mut Ctx,
    entry_idx: usize,
) {
    let value_types = function_value_types(ctx, entry_idx);
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
    // Struct member byte offsets + array strides — the address oracle for the absolute-offset resolve.
    let mut member_offset: HashMap<(Word, u32), u32> = HashMap::new();
    let mut array_stride: HashMap<Word, u32> = HashMap::new();
    for inst in &ctx.module.annotations {
        match inst.class.opcode {
            Op::MemberDecorate => {
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
            Op::Decorate => {
                if let (
                    Some(Operand::IdRef(ty)),
                    Some(Operand::Decoration(Decoration::ArrayStride)),
                    Some(Operand::LiteralBit32(stride)),
                ) = (
                    inst.operands.first(),
                    inst.operands.get(1),
                    inst.operands.get(2),
                ) {
                    array_stride.insert(*ty, *stride);
                }
            }
            _ => {}
        }
    }

    // Accumulated absolute byte offset of walking a CONSTANT-index prefix into `base_pointee`.
    // Returns None on any non-constant index, a malformed walk, or a step whose stride/offset is
    // unknown (no ArrayStride / no member Offset).
    let prefix_byte = |start: Word, prefix: &[Operand]| -> Option<u32> {
        let mut cur = start;
        let mut byte: u32 = 0;
        for op in prefix {
            let Operand::IdRef(idx_id) = op else {
                return None;
            };
            let idx = const_u32(ctx, *idx_id)?;
            let def = type_def_of(ctx, cur)?;
            match def.class.opcode {
                Op::TypeStruct => {
                    byte = byte.checked_add(*member_offset.get(&(cur, idx))?)?;
                    cur = match def.operands.get(idx as usize) {
                        Some(Operand::IdRef(m)) => *m,
                        _ => return None,
                    };
                }
                Op::TypeArray | Op::TypeRuntimeArray => {
                    let stride = *array_stride.get(&cur)?;
                    byte = byte.checked_add(stride.checked_mul(idx)?)?;
                    cur = match def.operands.first() {
                        Some(Operand::IdRef(elem)) => *elem,
                        _ => return None,
                    };
                }
                // A vector/matrix step in the prefix would need a component-size stride; bail to keep
                // the address oracle decoration-grounded (these shapes don't occur for the flat-index
                // family and guessing a stride would risk a byte-wrong remap).
                _ => return None,
            }
        }
        Some(byte)
    };

    let mut edits: Vec<(usize, usize, u32)> = Vec::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if !matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain) {
                continue;
            }
            let Some(result_type) = inst.result_type else {
                continue;
            };
            let Some(&(_, result_pointee)) = ptr_info.get(&result_type) else {
                continue;
            };
            let Some(Operand::IdRef(base)) = inst.operands.first() else {
                continue;
            };
            let indices: Vec<Operand> = inst.operands[1..].to_vec();
            // Need a non-empty member PREFIX (≥1 descent) plus the trailing word index — the prefix is
            // what distinguishes this from the single-level remap (which handles the empty prefix).
            if indices.len() < 2 {
                continue;
            }
            let Some(base_ptr_ty) = value_types.get(base).copied() else {
                continue;
            };
            let Some(&(_, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            // The base buffer pointee must itself be a struct (the outer binding type) to host a
            // sibling member.
            let Some(bdef) = type_def_of(ctx, base_pointee) else {
                continue;
            };
            if bdef.class.opcode != Op::TypeStruct {
                continue;
            }
            // Floor-safe gate: only currently-INVALID chains (a valid chain walks every index to the
            // declared result pointee).
            let (reached, consumed) = walk_into_type_partial(ctx, base_pointee, &indices);
            if consumed == indices.len() && reached == result_pointee {
                continue;
            }
            let prefix = &indices[..indices.len() - 1];
            let Some(pbyte) = prefix_byte(base_pointee, prefix) else {
                continue;
            };
            let Some(Operand::IdRef(last_id)) = indices.last() else {
                continue;
            };
            let Some(word) = const_u32(ctx, *last_id) else {
                continue;
            };
            let Some(abs_byte) = word.checked_mul(4).and_then(|b| b.checked_add(pbyte)) else {
                continue;
            };
            // The unique TOP-LEVEL member of the base buffer struct at exactly `abs_byte` whose type is
            // the result pointee.
            let mut found: Option<u32> = None;
            for m in 0..bdef.operands.len() {
                if member_offset.get(&(base_pointee, m as u32)).copied() != Some(abs_byte) {
                    continue;
                }
                let Some(Operand::IdRef(mty)) = bdef.operands.get(m) else {
                    continue;
                };
                if *mty != result_pointee {
                    continue;
                }
                if found.is_some() {
                    found = None; // ambiguous — bail rather than guess
                    break;
                }
                found = Some(m as u32);
            }
            let Some(member) = found else {
                continue;
            };
            edits.push((bi, ii, member));
        }
    }

    for (bi, ii, member) in edits {
        let member_id = ctx.const_uint(member);
        // Collapse the whole index list to the single sibling-member index.
        let inst = &mut ctx.module.functions[entry_idx].blocks[bi].instructions[ii];
        inst.operands.truncate(1);
        inst.operands.push(Operand::IdRef(member_id));
    }
}

/// Remap a raw DYNAMIC flat word index `%buf <const-prefix> (%uint_W + %dyn)` onto the typed
/// top-level ARRAY member of the base buffer struct sitting at absolute byte `prefix_byte + 4W`.
///
/// This is the DYNAMIC-array-member generalization of [`remap_overflow_word_index_to_outer_member`]
/// (which collapses a CONSTANT word index onto a SCALAR sibling member). The R4 emitter lowers a typed
/// dynamic-array-member gep (`gep Buf, %p, 0, i32 M, i64 %dyn` — member `M` is `[K x scalar]`) as a
/// flat WORD index `%buf %uint_0 (%uint_W + %dyn)` where `W` is the member's word offset and `%dyn` the
/// element index. When the buffer descriptor stays a TYPED struct (because the SAME buffer is also read
/// by typed member geps), `%uint_0` descends into member 0 (a sub-struct) and the dynamic `%uint_W+%dyn`
/// indexes INTO it — illegal ("the <id> … into a structure must be an OpConstant"). When byte `4W`
/// (relative to the prefix) coincides EXACTLY with a top-level ARRAY member `M` of the base buffer struct
/// whose element type equals the access result pointee AND whose `ArrayStride` equals the word size (4),
/// the element index is precisely `%dyn`, so the chain collapses to `%buf %uint_M %dyn` — the identical
/// byte address with the identical scalar, a byte no-op reconciling the dual-use buffer's flat dynamic
/// word chain onto the typed-struct binding.
///
/// Byte-safe / floor-safe by construction: fires ONLY on a CURRENTLY-INVALID chain (the full walk stops
/// before consuming every index — a valid dynamic array access consumes all indices), whose member
/// PREFIX is a run of CONSTANT indices walking cleanly to a known accumulated byte offset, whose trailing
/// index is an `OpIAdd` of a CONSTANT word `W` and exactly one non-constant `%dyn`, whose result pointee
/// is a 32-bit scalar, and only when absolute byte `prefix_byte + 4W` matches the `Offset` of EXACTLY ONE
/// top-level member of the base buffer struct whose type is an `OpTypeArray` of that scalar with
/// `ArrayStride == 4`. Any ambiguity, a non-constant `W`, a non-4 stride, a non-scalar pointee, a
/// non-struct base, or a miss leaves the chain untouched — a banked/valid module never over-indexes, so
/// the floor is provably unchanged. Decides purely from IR structure (type walk + `Offset`/`ArrayStride`
/// decorations + the `OpIAdd const+dyn` shape), never a shader name.
pub(in crate::passes) fn remap_dynamic_word_index_to_array_member(ctx: &mut Ctx, entry_idx: usize) {
    let value_types = function_value_types(ctx, entry_idx);
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
    let mut array_stride: HashMap<Word, u32> = HashMap::new();
    for inst in &ctx.module.annotations {
        match inst.class.opcode {
            Op::MemberDecorate => {
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
            Op::Decorate => {
                if let (
                    Some(Operand::IdRef(ty)),
                    Some(Operand::Decoration(Decoration::ArrayStride)),
                    Some(Operand::LiteralBit32(stride)),
                ) = (
                    inst.operands.first(),
                    inst.operands.get(1),
                    inst.operands.get(2),
                ) {
                    array_stride.insert(*ty, *stride);
                }
            }
            _ => {}
        }
    }

    // result pointee must be a 32-bit scalar (so a 4-byte word stride == one element).
    let is_word_scalar = |ctx: &Ctx, ty: Word| -> bool {
        match type_def_of(ctx, ty) {
            Some(def) => match def.class.opcode {
                Op::TypeInt | Op::TypeFloat => {
                    matches!(def.operands.first(), Some(Operand::LiteralBit32(32)))
                }
                _ => false,
            },
            None => false,
        }
    };

    // value-id -> (opcode, operands) for the entry function body — to resolve the `OpIAdd const + dyn`.
    let mut value_def: HashMap<Word, (Op, Vec<Operand>)> = HashMap::new();
    for block in &ctx.module.functions[entry_idx].blocks {
        for inst in &block.instructions {
            if let Some(id) = inst.result_id {
                value_def.insert(id, (inst.class.opcode, inst.operands.clone()));
            }
        }
    }
    // Split an `OpIAdd` into (constant word W, dynamic operand id) regardless of operand order.
    let split_const_plus_dyn = |ctx: &Ctx, id: Word| -> Option<(u32, Word)> {
        let (op, ops) = value_def.get(&id)?;
        if *op != Op::IAdd {
            return None;
        }
        let (Operand::IdRef(a), Operand::IdRef(b)) = (ops.first()?, ops.get(1)?) else {
            return None;
        };
        match (const_u32(ctx, *a), const_u32(ctx, *b)) {
            (Some(w), None) => Some((w, *b)),
            (None, Some(w)) => Some((w, *a)),
            _ => None, // const+const folds elsewhere; dyn+dyn is not a flat word index
        }
    };

    let prefix_byte = |start: Word, prefix: &[Operand]| -> Option<u32> {
        let mut cur = start;
        let mut byte: u32 = 0;
        for op in prefix {
            let Operand::IdRef(idx_id) = op else {
                return None;
            };
            let idx = const_u32(ctx, *idx_id)?;
            let def = type_def_of(ctx, cur)?;
            match def.class.opcode {
                Op::TypeStruct => {
                    byte = byte.checked_add(*member_offset.get(&(cur, idx))?)?;
                    cur = match def.operands.get(idx as usize) {
                        Some(Operand::IdRef(m)) => *m,
                        _ => return None,
                    };
                }
                Op::TypeArray | Op::TypeRuntimeArray => {
                    let stride = *array_stride.get(&cur)?;
                    byte = byte.checked_add(stride.checked_mul(idx)?)?;
                    cur = match def.operands.first() {
                        Some(Operand::IdRef(elem)) => *elem,
                        _ => return None,
                    };
                }
                _ => return None,
            }
        }
        Some(byte)
    };

    // (block, inst, member M, dynamic id) edits.
    let mut edits: Vec<(usize, usize, u32, Word)> = Vec::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if !matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain) {
                continue;
            }
            let Some(result_type) = inst.result_type else {
                continue;
            };
            let Some(&(_, result_pointee)) = ptr_info.get(&result_type) else {
                continue;
            };
            if !is_word_scalar(ctx, result_pointee) {
                continue;
            }
            let Some(Operand::IdRef(base)) = inst.operands.first() else {
                continue;
            };
            let indices: Vec<Operand> = inst.operands[1..].to_vec();
            if indices.len() < 2 {
                continue;
            }
            let Some(base_ptr_ty) = value_types.get(base).copied() else {
                continue;
            };
            let Some(&(_, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            let Some(bdef) = type_def_of(ctx, base_pointee) else {
                continue;
            };
            if bdef.class.opcode != Op::TypeStruct {
                continue;
            }
            // Floor-safe gate: only currently-INVALID chains.
            let (reached, consumed) = walk_into_type_partial(ctx, base_pointee, &indices);
            if consumed == indices.len() && reached == result_pointee {
                continue;
            }
            let prefix = &indices[..indices.len() - 1];
            let Some(pbyte) = prefix_byte(base_pointee, prefix) else {
                continue;
            };
            let Some(Operand::IdRef(last_id)) = indices.last() else {
                continue;
            };
            let Some((word, dyn_id)) = split_const_plus_dyn(ctx, *last_id) else {
                continue;
            };
            let Some(abs_byte) = word.checked_mul(4).and_then(|b| b.checked_add(pbyte)) else {
                continue;
            };
            // The unique TOP-LEVEL member of the base buffer struct at exactly `abs_byte` that is an
            // ARRAY of the result-pointee scalar with a 4-byte ArrayStride (so `%dyn` == element index).
            let mut found: Option<u32> = None;
            for m in 0..bdef.operands.len() {
                if member_offset.get(&(base_pointee, m as u32)).copied() != Some(abs_byte) {
                    continue;
                }
                let Some(Operand::IdRef(mty)) = bdef.operands.get(m) else {
                    continue;
                };
                let Some(mdef) = type_def_of(ctx, *mty) else {
                    continue;
                };
                if mdef.class.opcode != Op::TypeArray {
                    continue;
                }
                if array_stride.get(mty).copied() != Some(4) {
                    continue;
                }
                let Some(Operand::IdRef(elem)) = mdef.operands.first() else {
                    continue;
                };
                if *elem != result_pointee {
                    continue;
                }
                if found.is_some() {
                    found = None; // ambiguous — bail rather than guess
                    break;
                }
                found = Some(m as u32);
            }
            let Some(member) = found else {
                continue;
            };
            edits.push((bi, ii, member, dyn_id));
        }
    }

    for (bi, ii, member, dyn_id) in edits {
        let member_id = ctx.const_uint(member);
        // Collapse to `%base %uint_M %dyn`.
        let inst = &mut ctx.module.functions[entry_idx].blocks[bi].instructions[ii];
        inst.operands.truncate(1);
        inst.operands.push(Operand::IdRef(member_id));
        inst.operands.push(Operand::IdRef(dyn_id));
    }
}

/// Remap a raw DYNAMIC flat word index `%buf <const-prefix> (W + S*idx)` onto a FIELD of the element
/// struct of a top-level `[K x struct]` ARRAY member, reconciling a uint-typed flat read of a (possibly
/// non-uint) 32-bit field via an inserted same-width `OpBitcast`.
///
/// Sibling of [`remap_dynamic_word_index_to_array_member`], which handles `[K x scalar]` members where
/// the result pointee already matches the element. MPS TRC kernels read fields of a uniform/storage
/// `[K x struct{N x float}]` member by FLAT word index then `OpBitcast` the loaded word to the field's
/// type; the emitter lowers the dynamic array-member gep as `%buf %uint_0 (W + S*idx)` — descending the
/// FIRST member (a sub-struct), then dynamically over-indexing it ("the <id> … into a structure must be
/// an OpConstant"). When the trailing index is `OpIAdd(W, OpIMul(idx, S))` and absolute byte
/// `prefix_byte + 4W` lands inside the FIRST element of a UNIQUE top-level array member `M` whose
/// `ArrayStride == 4*S` and whose element is a struct holding a 32-bit-scalar field `F` at that byte,
/// the access is byte-EXACTLY `%buf %uint_M %idx %uint_F` (element index `idx`, the `OpIMul` operand).
/// When the field's scalar type differs from the chain's result pointee (e.g. a `float` field read as
/// `uint`), the access chain is retyped to the field pointer and every load of it is split into
/// `OpLoad <field> ; OpBitcast <orig>` — both 32-bit, so the loaded value is bit-identical to the
/// original raw-word load + downstream bitcast.
///
/// Byte-safe / floor-safe by construction: fires ONLY on a CURRENTLY-INVALID chain (the type walk stops
/// before consuming every index — a valid dynamic access consumes all indices), whose prefix is a run of
/// CONSTANT indices walking to a known byte, whose trailing index is `OpIAdd` of a constant word `W` and
/// an `OpIMul` of one non-constant `idx` by a constant stride `S`, whose result pointee is a 32-bit
/// scalar, when `prefix_byte + 4W` falls within the first element of EXACTLY ONE top-level array member
/// whose ArrayStride is `4*S` and element is a struct with a 32-bit-scalar field at that byte, AND every
/// use of the chain is an `OpLoad` of a 32-bit scalar. Any miss/ambiguity (or a non-load use) leaves the
/// chain untouched — a banked/valid module never over-indexes, so the floor is provably unchanged.
/// Decides purely from IR structure (type walk + `Offset`/`ArrayStride` decorations + the
/// `OpIAdd const + OpIMul(dyn,const)` shape), never a shader name.
struct ExactWordTarget {
    path: Vec<u32>,
    ty: Word,
}

fn exact_word_path(
    ctx: &Ctx,
    member_offset: &HashMap<(Word, u32), u32>,
    array_stride: &HashMap<Word, u32>,
    ty: Word,
    byte_offset: u32,
) -> Option<ExactWordTarget> {
    let definition = type_def_of(ctx, ty)?;
    match definition.class.opcode {
        Op::TypeInt | Op::TypeFloat
            if byte_offset == 0
                && matches!(definition.operands.first(), Some(Operand::LiteralBit32(32))) =>
        {
            Some(ExactWordTarget {
                path: Vec::new(),
                ty,
            })
        }
        Op::TypeStruct => (0..definition.operands.len()).rev().find_map(|member| {
            let offset = member_offset.get(&(ty, member as u32)).copied()?;
            let relative = byte_offset.checked_sub(offset)?;
            let Operand::IdRef(member_ty) = definition.operands[member] else {
                return None;
            };
            let mut target =
                exact_word_path(ctx, member_offset, array_stride, member_ty, relative)?;
            target.path.insert(0, member as u32);
            Some(target)
        }),
        Op::TypeArray => {
            let (Some(Operand::IdRef(element)), Some(Operand::IdRef(length))) =
                (definition.operands.first(), definition.operands.get(1))
            else {
                return None;
            };
            let length = const_u32(ctx, *length)?;
            let stride = array_stride.get(&ty).copied()?;
            let index = byte_offset / stride;
            if index >= length {
                return None;
            }
            let mut target = exact_word_path(
                ctx,
                member_offset,
                array_stride,
                *element,
                byte_offset % stride,
            )?;
            target.path.insert(0, index);
            Some(target)
        }
        Op::TypeVector => {
            let (Some(Operand::IdRef(element)), Some(Operand::LiteralBit32(lanes))) =
                (definition.operands.first(), definition.operands.get(1))
            else {
                return None;
            };
            let element_definition = type_def_of(ctx, *element)?;
            let Some(Operand::LiteralBit32(width)) = element_definition.operands.first() else {
                return None;
            };
            let stride = width.checked_div(8)?;
            let index = byte_offset / stride;
            if index >= *lanes {
                return None;
            }
            let mut target = exact_word_path(
                ctx,
                member_offset,
                array_stride,
                *element,
                byte_offset % stride,
            )?;
            target.path.insert(0, index);
            Some(target)
        }
        _ => None,
    }
}

pub(in crate::passes) fn remap_dynamic_word_index_to_array_struct_field(
    ctx: &mut Ctx,
    entry_idx: usize,
) {
    let value_types = function_value_types(ctx, entry_idx);
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
    let mut array_stride: HashMap<Word, u32> = HashMap::new();
    for inst in &ctx.module.annotations {
        match inst.class.opcode {
            Op::MemberDecorate => {
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
            Op::Decorate => {
                if let (
                    Some(Operand::IdRef(ty)),
                    Some(Operand::Decoration(Decoration::ArrayStride)),
                    Some(Operand::LiteralBit32(stride)),
                ) = (
                    inst.operands.first(),
                    inst.operands.get(1),
                    inst.operands.get(2),
                ) {
                    array_stride.insert(*ty, *stride);
                }
            }
            _ => {}
        }
    }

    // result pointee must be a 32-bit scalar (so a 4-byte word stride == one element word).
    let is_word_scalar = |ctx: &Ctx, ty: Word| -> bool {
        match type_def_of(ctx, ty) {
            Some(def) => match def.class.opcode {
                Op::TypeInt | Op::TypeFloat => {
                    matches!(def.operands.first(), Some(Operand::LiteralBit32(32)))
                }
                _ => false,
            },
            None => false,
        }
    };

    // value-id -> (opcode, operands) for the entry function body.
    let mut value_def: HashMap<Word, (Op, Vec<Operand>)> = HashMap::new();
    for block in &ctx.module.functions[entry_idx].blocks {
        for inst in &block.instructions {
            if let Some(id) = inst.result_id {
                value_def.insert(id, (inst.class.opcode, inst.operands.clone()));
            }
        }
    }
    // Split `OpIAdd` into (constant word W, other operand id), regardless of order.
    let split_const_plus_other = |ctx: &Ctx, id: Word| -> Option<(u32, Word)> {
        let (op, ops) = value_def.get(&id)?;
        if *op != Op::IAdd {
            return None;
        }
        let (Operand::IdRef(a), Operand::IdRef(b)) = (ops.first()?, ops.get(1)?) else {
            return None;
        };
        match (const_u32(ctx, *a), const_u32(ctx, *b)) {
            (Some(w), None) => Some((w, *b)),
            (None, Some(w)) => Some((w, *a)),
            _ => None,
        }
    };
    // Split `OpIMul` into (dynamic id, constant stride S), regardless of order.
    let split_dyn_times_const = |ctx: &Ctx, id: Word| -> Option<(Word, u32)> {
        let (op, ops) = value_def.get(&id)?;
        if *op != Op::IMul {
            return None;
        }
        let (Operand::IdRef(a), Operand::IdRef(b)) = (ops.first()?, ops.get(1)?) else {
            return None;
        };
        match (const_u32(ctx, *a), const_u32(ctx, *b)) {
            (Some(s), None) => Some((*b, s)),
            (None, Some(s)) => Some((*a, s)),
            _ => None,
        }
    };

    let prefix_byte = |start: Word, prefix: &[Operand]| -> Option<u32> {
        let mut cur = start;
        let mut byte: u32 = 0;
        for op in prefix {
            let Operand::IdRef(idx_id) = op else {
                return None;
            };
            let idx = const_u32(ctx, *idx_id)?;
            let def = type_def_of(ctx, cur)?;
            match def.class.opcode {
                Op::TypeStruct => {
                    byte = byte.checked_add(*member_offset.get(&(cur, idx))?)?;
                    cur = match def.operands.get(idx as usize) {
                        Some(Operand::IdRef(m)) => *m,
                        _ => return None,
                    };
                }
                Op::TypeArray | Op::TypeRuntimeArray => {
                    let stride = *array_stride.get(&cur)?;
                    byte = byte.checked_add(stride.checked_mul(idx)?)?;
                    cur = match def.operands.first() {
                        Some(Operand::IdRef(elem)) => *elem,
                        _ => return None,
                    };
                }
                _ => return None,
            }
        }
        Some(byte)
    };

    // Map operand-uses of each value id to instruction sites, so we can require all-loads.
    let mut uses: HashMap<Word, Vec<(usize, usize, Op)>> = HashMap::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            for op in &inst.operands {
                if let Operand::IdRef(r) = op {
                    uses.entry(*r)
                        .or_default()
                        .push((bi, ii, inst.class.opcode));
                }
            }
        }
    }

    // (block, inst, result_id, member M, element dyn id, field F, field scalar type, storage).
    struct ChainEdit {
        bi: usize,
        ii: usize,
        cid: Word,
        member: u32,
        elem: Word,
        elem_bias: u32,
        field_path: Vec<u32>,
        field_ty: Word,
        storage: StorageClass,
    }
    let mut edits: Vec<ChainEdit> = Vec::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if !matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain) {
                continue;
            }
            let Some(result_type) = inst.result_type else {
                continue;
            };
            let Some(&(storage, result_pointee)) = ptr_info.get(&result_type) else {
                continue;
            };
            if !is_word_scalar(ctx, result_pointee) {
                continue;
            }
            let Some(cid) = inst.result_id else {
                continue;
            };
            let Some(Operand::IdRef(base)) = inst.operands.first() else {
                continue;
            };
            let indices: Vec<Operand> = inst.operands[1..].to_vec();
            if indices.len() < 2 {
                continue;
            }
            let Some(base_ptr_ty) = value_types.get(base).copied() else {
                continue;
            };
            let Some(&(_, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            let Some(bdef) = type_def_of(ctx, base_pointee) else {
                continue;
            };
            if bdef.class.opcode != Op::TypeStruct {
                continue;
            }
            // Floor-safe gate: only currently-INVALID chains.
            let (reached, consumed) = walk_into_type_partial(ctx, base_pointee, &indices);
            if consumed == indices.len() && reached == result_pointee {
                continue;
            }
            let prefix = &indices[..indices.len() - 1];
            let Some(pbyte) = prefix_byte(base_pointee, prefix) else {
                continue;
            };
            let Some(Operand::IdRef(last_id)) = indices.last() else {
                continue;
            };
            let Some((word, mul_id)) = split_const_plus_other(ctx, *last_id) else {
                continue;
            };
            let Some((elem_dyn, stride_words)) = split_dyn_times_const(ctx, mul_id) else {
                continue;
            };
            if stride_words == 0 {
                continue;
            }
            let Some(abs_byte) = word.checked_mul(4).and_then(|b| b.checked_add(pbyte)) else {
                continue;
            };
            let Some(elem_stride_bytes) = stride_words.checked_mul(4) else {
                continue;
            };
            // Unique TOP-LEVEL array member containing the constant address, with ArrayStride ==
            // 4*S and a struct element type. Split an address in any element into a constant
            // element bias plus its byte offset within that element; restricting this to element
            // zero incorrectly rejected otherwise ordinary affine accesses.
            let mut found: Option<(u32, u32, u32, Word)> = None; // (member M, element bias, field byte, elem struct type)
            for m in 0..bdef.operands.len() {
                let Some(&off) = member_offset.get(&(base_pointee, m as u32)) else {
                    continue;
                };
                if abs_byte < off {
                    continue;
                }
                let Some(Operand::IdRef(mty)) = bdef.operands.get(m) else {
                    continue;
                };
                let Some(mdef) = type_def_of(ctx, *mty) else {
                    continue;
                };
                if mdef.class.opcode != Op::TypeArray {
                    continue;
                }
                if array_stride.get(mty).copied() != Some(elem_stride_bytes) {
                    continue;
                }
                let (Some(Operand::IdRef(elem_ty)), Some(Operand::IdRef(length_id))) =
                    (mdef.operands.first(), mdef.operands.get(1))
                else {
                    continue;
                };
                let Some(length) = const_u32(ctx, *length_id) else {
                    continue;
                };
                let relative = abs_byte - off;
                let elem_bias = relative / elem_stride_bytes;
                if elem_bias >= length {
                    continue;
                }
                let Some(edef) = type_def_of(ctx, *elem_ty) else {
                    continue;
                };
                if edef.class.opcode != Op::TypeStruct {
                    continue;
                }
                if found.is_some() {
                    found = None; // ambiguous — bail rather than guess
                    break;
                }
                found = Some((m as u32, elem_bias, relative % elem_stride_bytes, *elem_ty));
            }
            let Some((member, elem_bias, field_byte, elem_ty)) = found else {
                continue;
            };
            // Resolve the exact constant path inside the array element. Nested matrix/array/struct
            // fields are as representable as direct fields; the former one-level lookup discarded
            // valid addresses inside those aggregates.
            let Some(target) =
                exact_word_path(ctx, &member_offset, &array_stride, elem_ty, field_byte)
            else {
                continue;
            };
            // Every use of the chain must be an OpLoad (so retyping the pointer is safe).
            let all_loads = uses
                .get(&cid)
                .map(|u| u.iter().all(|&(_, _, op)| op == Op::Load))
                .unwrap_or(false);
            if !all_loads {
                continue;
            }
            edits.push(ChainEdit {
                bi,
                ii,
                cid,
                member,
                elem: elem_dyn,
                elem_bias,
                field_path: target.path,
                field_ty: target.ty,
                storage,
            });
        }
    }

    if edits.is_empty() {
        return;
    }

    // For each retyped chain whose field type differs from a load's result type, split that load into
    // `OpLoad <field> ; OpBitcast <orig>`. Collect the load sites first (needs the chain edits' cids).
    let cid_field: HashMap<Word, Word> = edits.iter().map(|e| (e.cid, e.field_ty)).collect();
    // load site -> (field_ty, original load result type)
    let mut load_splits: HashMap<(usize, usize), (Word, Word)> = HashMap::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if inst.class.opcode != Op::Load {
                continue;
            }
            let Some(Operand::IdRef(ptr)) = inst.operands.first() else {
                continue;
            };
            let Some(&field_ty) = cid_field.get(ptr) else {
                continue;
            };
            let Some(orig_ty) = inst.result_type else {
                continue;
            };
            if orig_ty != field_ty {
                load_splits.insert((bi, ii), (field_ty, orig_ty));
            }
        }
    }

    // Retype + rewrite the access-chain instructions in place (no index shift).
    for e in &edits {
        let field_ptr_ty = ctx.ty_ptr(e.storage, e.field_ty);
        let member_id = ctx.const_uint(e.member);
        let field_ids = e
            .field_path
            .iter()
            .map(|field| ctx.const_uint(*field))
            .collect::<Vec<_>>();
        let base = match ctx.module.functions[entry_idx].blocks[e.bi].instructions[e.ii]
            .operands
            .first()
        {
            Some(Operand::IdRef(b)) => *b,
            _ => continue,
        };
        let inst = &mut ctx.module.functions[entry_idx].blocks[e.bi].instructions[e.ii];
        inst.result_type = Some(field_ptr_ty);
        inst.operands = vec![
            Operand::IdRef(base),
            Operand::IdRef(member_id),
            Operand::IdRef(e.elem),
        ];
        inst.operands
            .extend(field_ids.into_iter().map(Operand::IdRef));
    }

    // Apply load splits by rebuilding affected blocks (insert a field-typed load before each, and
    // turn the original load into a same-width OpBitcast preserving its id/type).
    let mut by_block: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(bi, ii) in load_splits.keys() {
        by_block.entry(bi).or_default().push(ii);
    }
    for (bi, mut iis) in by_block {
        iis.sort_unstable();
        let insts = ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .clone();
        let mut out: Vec<Instruction> = Vec::with_capacity(insts.len() + iis.len());
        let mut next = 0usize;
        for (ii, inst) in insts.into_iter().enumerate() {
            if next < iis.len() && iis[next] == ii {
                next += 1;
                let (field_ty, orig_ty) = load_splits[&(bi, ii)];
                let ptr = match inst.operands.first() {
                    Some(Operand::IdRef(p)) => *p,
                    _ => {
                        out.push(inst);
                        continue;
                    }
                };
                let mem_operand = inst.operands.get(1).cloned();
                let load_id = ctx.module.fresh_id();
                let mut load_ops = vec![Operand::IdRef(ptr)];
                if let Some(m) = mem_operand {
                    load_ops.push(m);
                }
                out.push(Instruction::new(
                    Op::Load,
                    Some(field_ty),
                    Some(load_id),
                    load_ops,
                ));
                out.push(Instruction::new(
                    Op::Bitcast,
                    Some(orig_ty),
                    inst.result_id,
                    vec![Operand::IdRef(load_id)],
                ));
            } else {
                out.push(inst);
            }
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = out;
    }

    // Materialize a nonzero constant element bias immediately before the rewritten chain. Locate
    // by result id after load splitting so inserted field-typed loads cannot stale block indices.
    for e in edits.iter().filter(|edit| edit.elem_bias != 0) {
        let Some(elem_ty) = value_result_type(ctx, e.elem) else {
            continue;
        };
        let bias = ctx.const_int_of(elem_ty, i64::from(e.elem_bias));
        let biased_elem = ctx.module.fresh_id();
        let Some((block_index, instruction_index)) = ctx.module.functions[entry_idx]
            .blocks
            .iter()
            .enumerate()
            .find_map(|(block_index, block)| {
                block
                    .instructions
                    .iter()
                    .position(|instruction| instruction.result_id == Some(e.cid))
                    .map(|instruction_index| (block_index, instruction_index))
            })
        else {
            continue;
        };
        let block = &mut ctx.module.functions[entry_idx].blocks[block_index];
        block.instructions.insert(
            instruction_index,
            Instruction::new(
                Op::IAdd,
                Some(elem_ty),
                Some(biased_elem),
                vec![Operand::IdRef(e.elem), Operand::IdRef(bias)],
            ),
        );
        if let Some(Operand::IdRef(element)) = block.instructions[instruction_index + 1]
            .operands
            .get_mut(2)
        {
            *element = biased_elem;
        }
    }
}
