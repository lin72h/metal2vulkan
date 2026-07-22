//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;
use crate::passes::stage_input::{layout_ty_size_align, round_up};

pub(in crate::passes) fn const_int_like(ctx: &mut Ctx, like: Word, value: u64) -> Word {
    let Some(ty) = value_result_type(ctx, like) else {
        return ctx.const_uint(value as u32);
    };
    let Some(def) = type_def_of(ctx, ty) else {
        return ctx.const_uint(value as u32);
    };
    if def.class.opcode != Op::TypeInt {
        return ctx.const_uint(value as u32);
    }
    match def.operands.first() {
        Some(Operand::LiteralBit32(64)) => {
            let id = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Constant,
                Some(ty),
                Some(id),
                vec![Operand::LiteralBit64(value)],
            ));
            id
        }
        _ => ctx.const_uint(value as u32),
    }
}

/// Narrow 64-bit-integer INDEX operands of an `OpAccessChain`/`OpInBoundsAccessChain`/
/// `OpPtrAccessChain` in the entry function to 32-bit-integer equivalents WHERE VALUE-PRESERVING.
/// NVIDIA's SPIR-V->NVVM compiler SEGFAULTed when an access-chain index was a 64-bit (`%ulong`)
/// value (that rail is retired; the narrowing is kept because 32-bit indices remain the more
/// portable form):
///   * a 64-bit CONSTANT index that fits in u32 -> the equal-valued 32-bit `OpConstant uint`;
///   * a 64-bit CONSTANT index that does NOT fit -> left 64-bit (truncating it would silently WRAP
///     the address: a degenerate-but-real AIR shape, e.g. copyKernel's synthesized
///     `MTLCopyArgs{0x1_00000001,...}`, indexes with i64 values >= 2^32, and Apple hardware resolves
///     the full 64-bit address — truncation made metal2vulkan write a texel Apple leaves untouched);
///   * a 64-bit SSA index -> left 64-bit for the same reason: whether it exceeds u32 is a runtime
///     property, and MoltenVK/spirv-val accept 64-bit access-chain indices.
///     The base pointer operand (operand 0) is left untouched — only fitting constant indices are
///     narrowed.
pub(in crate::passes) fn narrow_access_chain_indices(ctx: &mut Ctx, entry_idx: usize) {
    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let mut new_insts: Vec<Instruction> = vec![];
        let insts = ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .clone();
        for mut inst in insts {
            if !matches!(
                inst.class.opcode,
                Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
            ) {
                new_insts.push(inst);
                continue;
            }
            // Operand 0 is the base pointer; operands 1.. are the indices. Narrow each 64-bit index.
            let pre: Vec<Instruction> = vec![];
            for oi in 1..inst.operands.len() {
                let idx = match inst.operands[oi] {
                    Operand::IdRef(r) => r,
                    _ => continue,
                };
                let idx_ty = match value_result_type(ctx, idx) {
                    Some(t) => t,
                    None => continue,
                };
                let Some(signed) = int64_signedness(ctx, idx_ty) else {
                    continue; // already 32-bit (or narrower) -> leave it.
                };
                let _ = signed;
                if let Some(v) = const_i64_value(ctx, idx) {
                    // Constant index: reuse/synthesize the equal-valued 32-bit uint constant, but
                    // only when the value actually fits — truncating a >= 2^32 constant would
                    // silently wrap the address instead of keeping Apple's 64-bit resolution.
                    if u32::try_from(v).is_ok() {
                        let c32 = ctx.const_uint(v as u32);
                        inst.operands[oi] = Operand::IdRef(c32);
                    }
                }
                // Dynamic 64-bit indices stay 64-bit: whether they exceed u32 is a runtime
                // property, and truncation diverges from the 64-bit address math Apple hardware
                // performs for the same AIR.
            }
            new_insts.extend(pre);
            new_insts.push(inst);
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = new_insts;
    }
}

/// Enforce the SPIR-V rule that a scalar-integer arithmetic/bitwise op's operands share the result
/// type's bit width. Lowering can leave an operand at the wrong width: a value widened to `ulong` to
/// index a `StorageBuffer` runtime array (`%i32 -> OpUConvert %ulong`) is sometimes reused directly in
/// a 32-bit offset multiply, so the `ulong` id flows into an `OpIMul %uint`, which spirv-val rejects
/// ("arithmetic operands must have the same bit width as Result Type"). This is purely structural: for
/// each targeted op whose result is a scalar `OpTypeInt` of width W, any `IdRef` operand whose own
/// result type is a scalar `OpTypeInt` WIDER than W gets a truncating `OpUConvert` to width W inserted
/// immediately before the op, and the operand is repointed at it. Truncation recovers exactly the
/// low-W bits the narrow op is meant to consume (the value was widened FROM that width), so it is
/// byte-neutral for a correctly-typed module — every operand already matches W, no `OpUConvert` is
/// inserted, the bytes are identical. Keys on operand vs result bit width alone, never on any name.
/// Shifts are excluded: their Shift operand may legally differ in width from the result.
pub(in crate::passes) fn normalize_int_arith_operand_widths(ctx: &mut Ctx) {
    // scalar OpTypeInt id -> bit width.
    let mut int_width: HashMap<Word, u32> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if inst.class.opcode == Op::TypeInt {
            if let (Some(id), Some(Operand::LiteralBit32(w))) =
                (inst.result_id, inst.operands.first())
            {
                int_width.insert(id, *w);
            }
        }
    }
    // value id -> result-type id, for every value-producing instruction.
    let mut id_ty: HashMap<Word, Word> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if let (Some(r), Some(t)) = (inst.result_id, inst.result_type) {
            id_ty.insert(r, t);
        }
    }
    for f in &ctx.module.functions {
        for p in &f.parameters {
            if let (Some(r), Some(t)) = (p.result_id, p.result_type) {
                id_ty.insert(r, t);
            }
        }
        for b in &f.blocks {
            for i in &b.instructions {
                if let (Some(r), Some(t)) = (i.result_id, i.result_type) {
                    id_ty.insert(r, t);
                }
            }
        }
    }
    let is_target = |op: Op| {
        matches!(
            op,
            Op::IAdd
                | Op::ISub
                | Op::IMul
                | Op::UDiv
                | Op::SDiv
                | Op::UMod
                | Op::SRem
                | Op::SMod
                | Op::BitwiseAnd
                | Op::BitwiseOr
                | Op::BitwiseXor
                | Op::SNegate
                | Op::Not
        )
    };
    for fi in 0..ctx.module.functions.len() {
        for bi in 0..ctx.module.functions[fi].blocks.len() {
            // Take the instructions out so `ctx.module.fresh_id()` can allocate ids without a live borrow.
            let insts = std::mem::take(&mut ctx.module.functions[fi].blocks[bi].instructions);
            let mut new_insts: Vec<Instruction> = Vec::with_capacity(insts.len());
            for mut inst in insts {
                let coerce = is_target(inst.class.opcode)
                    .then_some(inst.result_type)
                    .flatten()
                    .and_then(|rt| int_width.get(&rt).copied().map(|w| (rt, w)));
                if let Some((res_ty, res_w)) = coerce {
                    // Dedup within this instruction so a value used twice truncates once.
                    let mut truncated: HashMap<Word, Word> = HashMap::new();
                    for oi in 0..inst.operands.len() {
                        let Operand::IdRef(id) = inst.operands[oi] else {
                            continue;
                        };
                        let too_wide = id_ty
                            .get(&id)
                            .and_then(|t| int_width.get(t))
                            .is_some_and(|&ow| ow > res_w);
                        if !too_wide {
                            continue;
                        }
                        let narrow = if let Some(&existing) = truncated.get(&id) {
                            existing
                        } else {
                            let fresh = ctx.module.fresh_id();
                            new_insts.push(Instruction::new(
                                Op::UConvert,
                                Some(res_ty),
                                Some(fresh),
                                vec![Operand::IdRef(id)],
                            ));
                            id_ty.insert(fresh, res_ty);
                            truncated.insert(id, fresh);
                            fresh
                        };
                        inst.operands[oi] = Operand::IdRef(narrow);
                    }
                }
                new_insts.push(inst);
            }
            ctx.module.functions[fi].blocks[bi].instructions = new_insts;
        }
    }
}

/// After helper inlining and access-chain composition, scalar pointer arithmetic can appear as
/// `OpInBoundsAccessChain %ptr_T %already_ptr_T %idx`. In Logical SPIR-V, that tries to index
/// through scalar `T`. Storage classes that Vulkan allows as `OpPtrAccessChain` bases use that form;
/// Private chains must have been composed back to an aggregate root first.
pub(in crate::passes) fn rewrite_scalar_pointer_arithmetic_access_chains(
    ctx: &mut Ctx,
    entry_idx: usize,
) {
    let mut pointer_storage = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if inst.class.opcode == Op::TypePointer {
            if let (Some(result), Some(Operand::StorageClass(storage))) =
                (inst.result_id, inst.operands.first())
            {
                pointer_storage.insert(result, *storage);
            }
        }
    }

    let mut id_types = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if let (Some(result), Some(result_type)) = (inst.result_id, inst.result_type) {
            id_types.insert(result, result_type);
        }
    }
    for function in &ctx.module.functions {
        for inst in &function.parameters {
            if let (Some(result), Some(result_type)) = (inst.result_id, inst.result_type) {
                id_types.insert(result, result_type);
            }
        }
        for block in &function.blocks {
            for inst in &block.instructions {
                if let (Some(result), Some(result_type)) = (inst.result_id, inst.result_type) {
                    id_types.insert(result, result_type);
                }
            }
        }
    }

    for block in &mut ctx.module.functions[entry_idx].blocks {
        for inst in &mut block.instructions {
            if inst.class.opcode != Op::InBoundsAccessChain {
                continue;
            }
            let Some(result_type) = inst.result_type else {
                continue;
            };
            if !pointer_storage
                .get(&result_type)
                .is_some_and(|storage| ptr_access_chain_allowed_storage(*storage))
            {
                continue;
            }
            let Some(Operand::IdRef(base)) = inst.operands.first() else {
                continue;
            };
            if id_types.get(base) != Some(&result_type) {
                continue;
            }
            *inst = Instruction::new(
                Op::PtrAccessChain,
                inst.result_type,
                inst.result_id,
                inst.operands.clone(),
            );
        }
    }
}

/// `OpPtrAccessChain` requires its Base pointer *type* to carry an `ArrayStride` decoration (the
/// element size it strides by), per VUID-StandaloneSpirv-None-10684. When the base points at a
/// scalar/vector element of a StorageBuffer (the scalar-pointer-arithmetic form produced above and
/// by the native emitter's `pointer_arithmetic_access_chain_op_for_storage` path), nothing in the
/// Block-layout pass decorates that pointer type — it is not an array/struct member — so spirv-val
/// rejects the chain. This walks every `OpPtrAccessChain` in the module and adds the missing
/// `ArrayStride = round_up(sizeof pointee)` to each distinct base pointer type, idempotently.
pub(in crate::passes) fn decorate_ptr_access_chain_base_strides(ctx: &mut Ctx) {
    let mut defs: HashMap<Word, Instruction> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if let Some(result) = inst.result_id {
            defs.insert(result, inst.clone());
        }
    }

    // Pointer types already carrying an ArrayStride (avoid emitting a duplicate decoration).
    let mut already: HashSet<Word> = HashSet::new();
    for ann in &ctx.module.annotations {
        if ann.class.opcode == Op::Decorate
            && ann.operands.get(1) == Some(&Operand::Decoration(Decoration::ArrayStride))
        {
            if let Some(Operand::IdRef(t)) = ann.operands.first() {
                already.insert(*t);
            }
        }
    }

    // Collect the pointer types that every OpPtrAccessChain strides through. The Base operand's
    // pointer type is the one the VUID requires ArrayStride on; for the scalar-pointer-arithmetic
    // (same-type) form base type == result type, but for a stride+descent chain (the
    // `rewrite_strided_descent_access_chains` output) the base points at a wider aggregate than the
    // result, so its type differs — decorate BOTH (a redundant ArrayStride on a non-base pointer type
    // is harmless; spirv-val enforces it only when the type is actually a PtrAccessChain Base).
    let mut base_ptr_types: Vec<Word> = Vec::new();
    for function in &ctx.module.functions {
        for block in &function.blocks {
            for inst in &block.instructions {
                if inst.class.opcode != Op::PtrAccessChain {
                    continue;
                }
                if let Some(t) = inst.result_type {
                    if !base_ptr_types.contains(&t) {
                        base_ptr_types.push(t);
                    }
                }
                if let Some(Operand::IdRef(base)) = inst.operands.first() {
                    if let Some(t) = value_result_type(ctx, *base) {
                        if !base_ptr_types.contains(&t) {
                            base_ptr_types.push(t);
                        }
                    }
                }
            }
        }
    }

    for ptr_ty in base_ptr_types {
        if already.contains(&ptr_ty) {
            continue;
        }
        let Some(def) = defs.get(&ptr_ty) else {
            continue;
        };
        if def.class.opcode != Op::TypePointer {
            continue;
        }
        // `ArrayStride` is an explicit layout decoration. Vulkan permits it on the logical
        // StorageBuffer/PhysicalStorageBuffer pointer view used by `OpPtrAccessChain`, but rejects
        // the same decoration on a Workgroup pointer type (VUID-StandaloneSpirv-None-10684).
        // Workgroup `OpPtrAccessChain`s retain their undecorated pointer type.
        let Some(Operand::StorageClass(storage)) = def.operands.first() else {
            continue;
        };
        if matches!(storage, StorageClass::Workgroup) {
            continue;
        }
        // OpTypePointer %storage %pointee — pointee is operand[1].
        let Some(Operand::IdRef(pointee)) = def.operands.get(1) else {
            continue;
        };
        let (size, align) = layout_ty_size_align(ctx, *pointee, &defs);
        let stride = round_up(size, align).max(1);
        ctx.module.annotations.push(Instruction::new(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(ptr_ty),
                Operand::Decoration(Decoration::ArrayStride),
                Operand::LiteralBit32(stride),
            ],
        ));
        already.insert(ptr_ty);
    }
}

pub(in crate::passes) fn ptr_access_chain_allowed_storage(storage: StorageClass) -> bool {
    matches!(
        storage,
        StorageClass::Workgroup | StorageClass::StorageBuffer | StorageClass::PhysicalStorageBuffer
    )
}

/// Walk a composite TYPE id through access-chain index operands (each an `IdRef` to an `OpConstant`),
/// returning the innermost reached type id, or `None` if a step indexes a non-composite. This is the
/// "index INTO the type" semantics (struct member by constant; array/runtime-array/vector/matrix
/// deref to element regardless of value) used to test InBounds validity and the PtrAccessChain
/// post-stride descent.
pub(in crate::passes) fn walk_into_type(
    ctx: &Ctx,
    mut cur: Word,
    indices: &[Operand],
) -> Option<Word> {
    for op in indices {
        let def = type_def_of(ctx, cur)?;
        cur = match def.class.opcode {
            Op::TypeStruct => {
                let Operand::IdRef(idx_id) = op else {
                    return None;
                };
                let cdef = type_def_of(ctx, *idx_id)?;
                if cdef.class.opcode != Op::Constant {
                    return None;
                }
                let member = match cdef.operands.first()? {
                    Operand::LiteralBit32(v) => *v as usize,
                    _ => return None,
                };
                match def.operands.get(member)? {
                    Operand::IdRef(m) => *m,
                    _ => return None,
                }
            }
            Op::TypeArray | Op::TypeRuntimeArray | Op::TypeVector | Op::TypeMatrix => {
                match def.operands.first()? {
                    Operand::IdRef(elem) => *elem,
                    _ => return None,
                }
            }
            _ => return None,
        };
    }
    Some(cur)
}

/// Walk a composite TYPE id through access-chain index operands AS FAR AS POSSIBLE, returning the
/// reached type id and how many indices were consumed. Unlike [`walk_into_type`] (which returns `None`
/// the moment a step indexes a non-composite), this stops gracefully at the first non-composite (or an
/// undecidable step) and reports the partial progress — used to detect a trailing over-index.
pub(in crate::passes) fn walk_into_type_partial(
    ctx: &Ctx,
    mut cur: Word,
    indices: &[Operand],
) -> (Word, usize) {
    for (n, op) in indices.iter().enumerate() {
        let Some(def) = type_def_of(ctx, cur) else {
            return (cur, n);
        };
        let next = match def.class.opcode {
            Op::TypeStruct => {
                let Operand::IdRef(idx_id) = op else {
                    return (cur, n);
                };
                let Some(cdef) = type_def_of(ctx, *idx_id) else {
                    return (cur, n);
                };
                if cdef.class.opcode != Op::Constant {
                    return (cur, n);
                }
                let member = match cdef.operands.first() {
                    Some(Operand::LiteralBit32(v)) => *v as usize,
                    _ => return (cur, n),
                };
                match def.operands.get(member) {
                    Some(Operand::IdRef(m)) => *m,
                    _ => return (cur, n),
                }
            }
            Op::TypeArray | Op::TypeRuntimeArray | Op::TypeVector | Op::TypeMatrix => {
                match def.operands.first() {
                    Some(Operand::IdRef(elem)) => *elem,
                    _ => return (cur, n),
                }
            }
            _ => return (cur, n),
        };
        cur = next;
    }
    (cur, indices.len())
}

/// Drop a TRAILING run of CONSTANT-ZERO over-indices from an INVALID member-access chain. The AIR
/// declares an MPS buffer's element struct from `air.struct_type_info`, which FLATTENS nested
/// single-member wrappers (`{{{uint}}}` → `uint`); the GEP keeps the full member-0 descent
/// (`base [0, N, 0, 0, 0]`), so under Logical addressing the chain reaches the flattened scalar and
/// then over-indexes it with the leftover `0`s — spirv-val "reached non-composite type while indexes
/// still remain". Because every dropped index is member-0 of a composite (byte offset 0), the leftover
/// descent lands at the SAME byte address as the reached scalar; dropping the trailing zeros is
/// byte-IDENTICAL.
///
/// Byte-safe / floor-safe by construction: only chains that are CURRENTLY INVALID (the partial walk
/// stops BEFORE consuming every index) are touched, and only when (1) every leftover index is an
/// `OpConstant 0`, (2) at least one index survives (a 0-index chain is not emitted), and (3) the result
/// pointee type EQUALS the scalar the surviving prefix reaches (so the truncated chain is valid and the
/// pointee is unchanged) — a banked/valid module (every chain fully walks) never matches. Decides purely
/// from IR structure (type walk + constant check), never a shader name.
pub(in crate::passes) fn drop_overindexed_zero_tail(ctx: &mut Ctx, entry_idx: usize) {
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

    let mut edits: Vec<(usize, usize, usize)> = Vec::new();
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
            let Some(base_ptr_ty) = value_result_type(ctx, *base) else {
                continue;
            };
            let Some(&(_, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            let (reached, consumed) = walk_into_type_partial(ctx, base_pointee, &indices);
            // Only currently-INVALID chains (a valid one walks every index); keep >= 1 index.
            if consumed >= indices.len() || consumed == 0 {
                continue;
            }
            // Every leftover index must be a constant 0 (a member-0 / zero-stride descent).
            let all_zero_tail = indices[consumed..].iter().all(|op| match op {
                Operand::IdRef(id) => const_u32(ctx, *id) == Some(0),
                _ => false,
            });
            if !all_zero_tail {
                continue;
            }
            // The surviving prefix must reach exactly the declared result pointee, AND that pointee must
            // be a direct scalar — so the dropped indices were descending INTO a scalar (provably a
            // byte no-op), not shifting a struct member offset.
            if reached != result_pointee || direct_scalar_width(ctx, reached).is_none() {
                continue;
            }
            edits.push((bi, ii, consumed));
        }
    }
    for (bi, ii, consumed) in edits {
        ctx.module.functions[entry_idx].blocks[bi].instructions[ii]
            .operands
            .truncate(1 + consumed);
    }
}

/// Re-root an over-index of a demoted array-element-0 pointer back onto the array.
///
/// metal2vulkan sometimes lowers an AIR `getelementptr [K x T], ptr %arr, i64 0, i64 %i` (element `%i` of a
/// function/threadgroup/device array) in TWO steps: first an element-0 pointer `%p = AC %arr %uint_0`
/// (a `_ptr_SC_T` to element 0), then the dynamic part `%r = AC %p %i` — which OVER-indexes the scalar
/// `%p` points at ("OpInBoundsAccessChain reached non-composite type while indexes still remain"). The
/// element-0 pointer may also be merged through one or more `OpPhi`/`OpCopyObject` before the
/// over-index (e.g. a loop-carried accumulator pointer). Since `%p` — and every phi arm it flows from —
/// provably equals `&%arr[0]`, the address `&(&%arr[0])[%i]` is byte-IDENTICAL to `&%arr[%i]`; the pass
/// re-roots the over-indexing chain onto `%arr` with the SAME single dynamic index.
///
/// **Byte-EXACT by construction**: element 0 + offset `i` is element `i` — the SAME byte address, for
/// ANY storage class (the array is element-contiguous; for StorageBuffer the `ArrayStride` is already
/// on the array type). This recovers the array provenance the two-step lowering lost — the size `K` is
/// NOT lost (the `[K x T]` array variable is declared; this is the demotion the prior BVH/W-c note
/// flagged, here resolved for the case where the array IS still declared and provenance converges).
/// **Floor-SAFE by construction**: fires ONLY on a CURRENTLY-INVALID chain (a base pointing at a direct
/// SCALAR equal to the result pointee, with exactly one remaining index — a valid module never
/// over-indexes a scalar) whose base provenance, traced through `OpPhi` (every incoming must converge)
/// and `OpCopyObject` and element-0 `AC`s, resolves to element 0 of ONE declared `[K x T]` array
/// variable (same storage class) whose element type equals the result pointee. Any divergence, a
/// non-element-0 chain, or an unknown provenance leaf leaves the chain untouched. Decides purely from
/// IR structure (provenance trace + type compare), never a shader name.
pub(in crate::passes) fn reroot_demoted_array_element_overindex(ctx: &mut Ctx, entry_idx: usize) {
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

    let func = ctx.module.functions[entry_idx].clone();
    let mut edits: Vec<(usize, usize, Word)> = Vec::new();
    for (bi, block) in func.blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if !matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain) {
                continue;
            }
            // Exactly one index (the dynamic element index) — the 1-D demoted-array shape.
            if inst.operands.len() != 2 {
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
            // CURRENTLY INVALID: base points at a direct SCALAR equal to the result pointee (a pure
            // re-root, not a reinterpret) — indexing a scalar over-runs.
            let Some(base_ptr_ty) = value_result_type(ctx, *base) else {
                continue;
            };
            let Some(&(base_sc, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            if base_pointee != result_pointee || direct_scalar_width(ctx, base_pointee).is_none() {
                continue;
            }
            // Provenance must converge to element 0 of ONE declared `[K x base_pointee]` array.
            let Some(array_id) =
                trace_to_array_element_zero(ctx, &func, *base, base_pointee, base_sc, &ptr_info)
            else {
                continue;
            };
            edits.push((bi, ii, array_id));
        }
    }
    for (bi, ii, array_id) in edits {
        // Re-root: replace the base operand with the array variable; the single index now selects the
        // array element (byte-identical to element 0 + that offset). Opcode/index preserved.
        ctx.module.functions[entry_idx].blocks[bi].instructions[ii].operands[0] =
            Operand::IdRef(array_id);
    }
}
