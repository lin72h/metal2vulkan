//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

/// The byte width and word width used by [`rewrite_raw_byte_pointer_wide_loads`].  This lowering is
/// deliberately limited to 32-bit scalar components: it reconstructs their native little-endian
/// representation from four unsigned-byte loads, then bitcasts the reconstructed word when the
/// requested component is a float or signed integer.
pub(in crate::passes) const RAW_BYTE_POINTER_ELEMENT_BITS: u32 = 8;

pub(in crate::passes) const RAW_BYTE_POINTER_WORD_BITS: u32 = 32;

pub(in crate::passes) const RAW_BYTE_POINTER_WORD_BYTES: u32 =
    RAW_BYTE_POINTER_WORD_BITS / RAW_BYTE_POINTER_ELEMENT_BITS;

/// Return `(component type, lane count)` for a scalar or vector whose components are direct 32-bit
/// integers/floats.  Matrices and aggregates intentionally do not match: they have layout rules
/// beyond this raw-byte replay's contiguous scalar lanes.
pub(in crate::passes) fn raw_byte_pointer_word_shape(ctx: &Ctx, ty: Word) -> Option<(Word, u32)> {
    let def = type_def_of(ctx, ty)?;
    let (component, lanes) = if def.class.opcode == Op::TypeVector {
        let (Some(Operand::IdRef(component)), Some(Operand::LiteralBit32(lanes))) =
            (def.operands.first(), def.operands.get(1))
        else {
            return None;
        };
        (*component, *lanes)
    } else {
        (ty, 1)
    };
    let component_def = type_def_of(ctx, component)?;
    if !matches!(component_def.class.opcode, Op::TypeInt | Op::TypeFloat) {
        return None;
    }
    match component_def.operands.first() {
        Some(Operand::LiteralBit32(width)) if *width == RAW_BYTE_POINTER_WORD_BITS => {
            Some((component, lanes))
        }
        _ => None,
    }
}

/// True for the unsigned-byte scalar type used as a raw StorageBuffer byte view.  Signed byte views
/// are deliberately left to a future lowering: `OpUConvert` would not be an unambiguous byte replay
/// for every signed producer.
pub(in crate::passes) fn is_unsigned_byte_scalar(ctx: &Ctx, ty: Word) -> bool {
    let Some(def) = type_def_of(ctx, ty) else {
        return false;
    };
    def.class.opcode == Op::TypeInt
        && def.operands.first() == Some(&Operand::LiteralBit32(RAW_BYTE_POINTER_ELEMENT_BITS))
        && def.operands.get(1) == Some(&Operand::LiteralBit32(0))
}

/// True for an integer scalar index width we can reproduce exactly with `Ctx::const_int_of`.
pub(in crate::passes) fn raw_byte_pointer_index_type(ctx: &Ctx, ty: Word) -> bool {
    let Some(def) = type_def_of(ctx, ty) else {
        return false;
    };
    def.class.opcode == Op::TypeInt
        && matches!(def.operands.first(), Some(Operand::LiteralBit32(32 | 64)))
}

/// Append a byte-exact little-endian load of one 32-bit component from `base + byte_offset`.
/// `base` and the resulting pointers have the same unsigned-byte pointer type.  The caller runs
/// [`decorate_ptr_access_chain_base_strides`] immediately afterward, which gives that pointer type
/// the required `ArrayStride = 1` for these `OpPtrAccessChain`s.
pub(in crate::passes) fn append_raw_byte_pointer_word_load(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    base: Word,
    base_ptr_ty: Word,
    byte_ty: Word,
    index_ty: Word,
    byte_offset: Word,
    component_ty: Word,
) -> Word {
    let uint_ty = ctx.ty_uint();
    let mut assembled: Option<Word> = None;
    for byte in 0..RAW_BYTE_POINTER_WORD_BYTES {
        let offset = if byte == 0 {
            byte_offset
        } else {
            let byte_const = ctx.const_int_of(index_ty, byte as i64);
            let id = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::IAdd,
                Some(index_ty),
                Some(id),
                vec![Operand::IdRef(byte_offset), Operand::IdRef(byte_const)],
            ));
            id
        };
        let ptr = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::PtrAccessChain,
            Some(base_ptr_ty),
            Some(ptr),
            vec![Operand::IdRef(base), Operand::IdRef(offset)],
        ));
        let raw_byte = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::Load,
            Some(byte_ty),
            Some(raw_byte),
            vec![Operand::IdRef(ptr)],
        ));
        let widened = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::UConvert,
            Some(uint_ty),
            Some(widened),
            vec![Operand::IdRef(raw_byte)],
        ));
        let shifted = if byte == 0 {
            widened
        } else {
            let shift = ctx.const_uint(byte * RAW_BYTE_POINTER_ELEMENT_BITS);
            let id = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::ShiftLeftLogical,
                Some(uint_ty),
                Some(id),
                vec![Operand::IdRef(widened), Operand::IdRef(shift)],
            ));
            id
        };
        assembled = Some(match assembled {
            None => shifted,
            Some(previous) => {
                let id = ctx.module.fresh_id();
                out.push(Instruction::new(
                    Op::BitwiseOr,
                    Some(uint_ty),
                    Some(id),
                    vec![Operand::IdRef(previous), Operand::IdRef(shifted)],
                ));
                id
            }
        });
    }
    let assembled = assembled.expect("a 32-bit raw-byte word has four bytes");
    let component = ctx.module.fresh_id();
    out.push(Instruction::new(
        if component_ty == uint_ty {
            Op::CopyObject
        } else {
            Op::Bitcast
        },
        Some(component_ty),
        Some(component),
        vec![Operand::IdRef(assembled)],
    ));
    component
}

/// Replace an INVALID wide scalar/vector load through a raw `StorageBuffer uchar*` with an explicit
/// little-endian byte replay.  LLVM's `getelementptr <N x T>, ptr addrspace(1) %raw, i64 i` means
/// `raw + i * sizeof(<N x T>)`; the native emitter may preserve that element-stride intent as an
/// `Op(InBounds)AccessChain` whose base pointee is the scalar `uchar`.  Logical SPIR-V instead treats
/// the one index as a descent *into* `uchar`, so the source instruction is invalid before its load can
/// execute ("reached non-composite type while indexes still remain").
///
/// The replacement computes `i * (lanes * 4)` in the original index type, accesses every byte with
/// `OpPtrAccessChain` on the original `uchar*`, assembles each 32-bit little-endian component, then
/// bitcasts it to the requested `int`/`float` component and reconstructs the vector.  Thus each byte
/// is read from precisely the address the source GEP names.  It supports a raw pointer produced by a
/// parameter, select, or phi equally: no buffer identity or descriptor aliasing is inferred.
///
/// Floor-safe by construction: it touches only an access chain that is CURRENTLY INVALID when walked
/// through an unsigned-byte pointee, has exactly one 32/64-bit scalar index, returns a 32-bit scalar
/// or vector pointer in the same StorageBuffer class, and whose EVERY use is a plain exact-typed
/// `OpLoad`.  No valid/banked access chain matches; stores, atomics, calls, pointer escapes, volatile
/// loads, and non-byte base views remain untouched.  The decision is entirely type/use topology, never
/// a function name, resource id, or a single-workload observation.
pub(in crate::passes) fn rewrite_raw_byte_pointer_wide_loads(ctx: &mut Ctx, entry_idx: usize) {
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

    #[derive(Clone, Copy)]
    struct Plan {
        bi: usize,
        ii: usize,
        chain_id: Word,
        base: Word,
        base_ptr_ty: Word,
        byte_ty: Word,
        index: Word,
        index_ty: Word,
        component_ty: Word,
        lanes: u32,
        result_pointee: Word,
    }

    let mut plans = Vec::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if !matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain)
                || inst.operands.len() != 2
            {
                continue;
            }
            let (Some(chain_id), Some(result_ptr_ty)) = (inst.result_id, inst.result_type) else {
                continue;
            };
            let Some(&(storage, result_pointee)) = ptr_info.get(&result_ptr_ty) else {
                continue;
            };
            if storage != StorageClass::StorageBuffer {
                continue;
            }
            let Some((component_ty, lanes)) = raw_byte_pointer_word_shape(ctx, result_pointee)
            else {
                continue;
            };
            let (Some(Operand::IdRef(base)), Some(Operand::IdRef(index))) =
                (inst.operands.first(), inst.operands.get(1))
            else {
                continue;
            };
            let Some(base_ptr_ty) = value_result_type(ctx, *base) else {
                continue;
            };
            let Some(&(base_storage, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            if base_storage != StorageClass::StorageBuffer
                || !is_unsigned_byte_scalar(ctx, base_pointee)
            {
                continue;
            }
            let Some(index_ty) = value_result_type(ctx, *index) else {
                continue;
            };
            if !raw_byte_pointer_index_type(ctx, index_ty) {
                continue;
            }
            // This is an over-index into the raw scalar byte pointee.  Keeping this explicit makes
            // the pass invalid-only even if a future emitter produces a superficially similar chain.
            if walk_into_type(ctx, base_pointee, &inst.operands[1..]).is_some() {
                continue;
            }
            plans.push(Plan {
                bi,
                ii,
                chain_id,
                base: *base,
                base_ptr_ty,
                byte_ty: base_pointee,
                index: *index,
                index_ty,
                component_ty,
                lanes,
                result_pointee,
            });
        }
    }
    if plans.is_empty() {
        return;
    }

    // A byte replay deliberately has no pointer result to hand on: every use must be a plain load
    // of the declared wide pointee.  Any memory-access operand (including Volatile/Aligned) is a
    // semantic boundary, so it disqualifies the entire chain rather than being guessed at.
    let plan_ids: HashSet<Word> = plans.iter().map(|p| p.chain_id).collect();
    let result_pointee_of: HashMap<Word, Word> = plans
        .iter()
        .map(|p| (p.chain_id, p.result_pointee))
        .collect();
    let mut load_sites: HashMap<Word, Vec<(usize, usize)>> = HashMap::new();
    let mut disqualified: HashSet<Word> = HashSet::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if inst.result_id.is_some_and(|id| plan_ids.contains(&id))
                && matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain)
            {
                continue;
            }
            for chain_id in inst.operands.iter().filter_map(|operand| match operand {
                Operand::IdRef(id) if plan_ids.contains(id) => Some(*id),
                _ => None,
            }) {
                let exact_plain_load = inst.class.opcode == Op::Load
                    && inst.result_type == result_pointee_of.get(&chain_id).copied()
                    && inst.operands.len() == 1
                    && inst.operands.first() == Some(&Operand::IdRef(chain_id));
                if exact_plain_load {
                    load_sites.entry(chain_id).or_default().push((bi, ii));
                } else {
                    disqualified.insert(chain_id);
                }
            }
        }
    }
    plans.retain(|plan| {
        !disqualified.contains(&plan.chain_id) && load_sites.contains_key(&plan.chain_id)
    });
    if plans.is_empty() {
        return;
    }

    let chain_at: HashMap<(usize, usize), Word> = plans
        .iter()
        .map(|plan| ((plan.bi, plan.ii), plan.chain_id))
        .collect();
    let plan_by_id: HashMap<Word, Plan> = plans.iter().map(|plan| (plan.chain_id, *plan)).collect();
    let load_at: HashMap<(usize, usize), Word> = plans
        .iter()
        .flat_map(|plan| {
            load_sites
                .get(&plan.chain_id)
                .into_iter()
                .flatten()
                .map(move |&site| (site, plan.chain_id))
        })
        .collect();

    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let old = std::mem::take(&mut ctx.module.functions[entry_idx].blocks[bi].instructions);
        let mut rewritten = Vec::with_capacity(old.len() + 32);
        for (ii, inst) in old.into_iter().enumerate() {
            if chain_at.contains_key(&(bi, ii)) {
                // The replacement is anchored at each direct load so the original pointer id cannot
                // escape and the index/base remain available along the original use path.
                continue;
            }
            let Some(chain_id) = load_at.get(&(bi, ii)).copied() else {
                rewritten.push(inst);
                continue;
            };
            let plan = plan_by_id[&chain_id];
            let result_id = inst
                .result_id
                .expect("raw-byte replay's exact typed load has a result id");
            let stride_bytes = plan
                .lanes
                .checked_mul(RAW_BYTE_POINTER_WORD_BYTES)
                .expect("SPIR-V vector lane count times word bytes fits u32");
            let stride = ctx.const_int_of(plan.index_ty, stride_bytes as i64);
            let base_offset = ctx.module.fresh_id();
            rewritten.push(Instruction::new(
                Op::IMul,
                Some(plan.index_ty),
                Some(base_offset),
                vec![Operand::IdRef(plan.index), Operand::IdRef(stride)],
            ));
            let mut components = Vec::with_capacity(plan.lanes as usize);
            for lane in 0..plan.lanes {
                let lane_offset = if lane == 0 {
                    base_offset
                } else {
                    let offset = ctx
                        .const_int_of(plan.index_ty, (lane * RAW_BYTE_POINTER_WORD_BYTES) as i64);
                    let id = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::IAdd,
                        Some(plan.index_ty),
                        Some(id),
                        vec![Operand::IdRef(base_offset), Operand::IdRef(offset)],
                    ));
                    id
                };
                let component = append_raw_byte_pointer_word_load(
                    ctx,
                    &mut rewritten,
                    plan.base,
                    plan.base_ptr_ty,
                    plan.byte_ty,
                    plan.index_ty,
                    lane_offset,
                    plan.component_ty,
                );
                components.push(Operand::IdRef(component));
            }
            if plan.lanes == 1 {
                rewritten.push(Instruction::new(
                    Op::CopyObject,
                    Some(plan.result_pointee),
                    Some(result_id),
                    components,
                ));
            } else {
                rewritten.push(Instruction::new(
                    Op::CompositeConstruct,
                    Some(plan.result_pointee),
                    Some(result_id),
                    components,
                ));
            }
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = rewritten;
    }
}

/// Lower an over-indexing access chain that reads/writes a sub-slot of a THREAD-LOCAL scalar variable
/// reused as a packed array of narrower scalars — the union shape an alloca gets when the AIR stores a
/// WIDE scalar (e.g. an `i64`) into it AND also addresses it as `float[2]`. `%c = AC
/// %_ptr_Function_float %slot %uint_N` over a scalar `_ptr_Function_ulong` base is illegal under Logical
/// addressing (indexing a non-composite), but its byte intent is exact: element `N` is bits
/// `[N*R, (N+1)*R)` of the wide slot (little-endian). The pass re-expresses each load/store through `%c`
/// as a shift/mask on the WHOLE slot:
///   - **load**: `OpLoad %slot`, `>> N*R` (if N>0), `OpUConvert` to the R-bit result, bitcast;
///   - **store** (thread-local, so the read-modify-write cannot race another thread on the other
///     bytes): `OpLoad %slot`, AND a not-mask that clears slot `N`, OR in `zext(obj) << N*R`, store back.
///     Byte-EXACT on a little-endian target; the wide-slot's own (matching-type) loads/stores are untouched.
///
/// The index may be a constant OR dynamic (the shift offset `N*R` is then computed at runtime; a slot
/// holds `SLOT_BITS/ELEM_BITS` elements, so the AIR's valid indices keep the shift in range — an
/// out-of-range index would be OOB in the original chain too).
///
/// Floor-safe by construction: only rewrites chains that are CURRENTLY INVALID (a single index
/// over-running a `Function`/`Private` DIRECT-SCALAR base whose width is an exact multiple of the
/// narrower result scalar, a constant index in range) and whose every use is an `OpLoad`/`OpStore` of
/// the result pointee — a banked/valid module (no scalar over-index) never matches. Restricted to a
/// 64-bit slot / 32-bit element (the `i64`-as-2×`i32` union shape); other widths fall through to leave
/// the chain for a later pass / the raw retry.
pub(in crate::passes) fn rewrite_scalar_slot_array_overindex(
    ctx: &mut Ctx,
    entry_idx: usize,
) -> Result<(), String> {
    const SLOT_BITS: u32 = 64;
    const ELEM_BITS: u32 = 32;

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

    struct Plan {
        bi: usize,
        ii: usize,
        ac_id: Word,
        base: Word,
        base_pointee: Word,
        result_pointee: Word,
        idx_id: Word,
        const_n: Option<u32>,
    }
    let mut plans: Vec<Plan> = Vec::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if !matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain) {
                continue;
            }
            let (Some(ac_id), Some(result_type)) = (inst.result_id, inst.result_type) else {
                continue;
            };
            // `%c = AC %base <idx…>` — base + at least one index. The element selector is the LAST
            // index; any LEADING indices must be constant 0 (the byte-0 descents of the phantom
            // single-member/array wrappers the AIR puts around the union slot, e.g. `%201 %uint_0
            // %uint_1`).
            if inst.operands.len() < 2 {
                continue;
            }
            let Operand::IdRef(base) = &inst.operands[0] else {
                continue;
            };
            let indices = &inst.operands[1..];
            let leading_all_zero = indices[..indices.len() - 1]
                .iter()
                .all(|op| matches!(op, Operand::IdRef(id) if const_u32(ctx, *id) == Some(0)));
            if !leading_all_zero {
                continue;
            }
            let Operand::IdRef(idx) = &indices[indices.len() - 1] else {
                continue;
            };
            let Some(&(sc_r, result_pointee)) = ptr_info.get(&result_type) else {
                continue;
            };
            if !matches!(sc_r, StorageClass::Function | StorageClass::Private) {
                continue;
            }
            if direct_scalar_width(ctx, result_pointee) != Some(ELEM_BITS) {
                continue;
            }
            // Base must be a pointer to a DIRECT WIDE SCALAR in the SAME thread-local storage class.
            let Some(base_ptr_ty) = value_result_type(ctx, *base) else {
                continue;
            };
            let Some(&(sc_b, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            if sc_b != sc_r || direct_scalar_width(ctx, base_pointee) != Some(SLOT_BITS) {
                continue;
            }
            // The element index is either a constant in range of the packed slot, or DYNAMIC (the AIR
            // only ever produces an in-range index — a slot holds SLOT_BITS/ELEM_BITS elements; an
            // out-of-range index would be OOB in the original chain too, so the shift reinterpret is
            // byte-faithful to the AIR's intent over the valid range).
            let const_n = const_u32(ctx, *idx);
            if let Some(n) = const_n {
                if (n + 1) * ELEM_BITS > SLOT_BITS {
                    continue;
                }
            }
            // Currently INVALID: walking `[idx]` into the scalar base pointee must fail.
            if walk_into_type(ctx, base_pointee, &inst.operands[1..]).is_some() {
                continue;
            }
            plans.push(Plan {
                bi,
                ii,
                ac_id,
                base: *base,
                base_pointee,
                result_pointee,
                idx_id: *idx,
                const_n,
            });
        }
    }
    if plans.is_empty() {
        return Ok(());
    }

    // Every use of each `%c` must be an OpLoad(result_pointee)/OpStore(_, result_pointee).
    let plan_ids: HashSet<Word> = plans.iter().map(|p| p.ac_id).collect();
    let result_pointee_of: HashMap<Word, Word> =
        plans.iter().map(|p| (p.ac_id, p.result_pointee)).collect();
    let mut use_sites: HashMap<Word, Vec<(usize, usize, bool)>> = HashMap::new();
    let mut disqualified: HashSet<Word> = HashSet::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if inst
                .result_id
                .map(|r| plan_ids.contains(&r))
                .unwrap_or(false)
                && matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain)
            {
                continue;
            }
            match inst.class.opcode {
                Op::Load => {
                    if let Some(Operand::IdRef(ptr)) = inst.operands.first() {
                        if plan_ids.contains(ptr) {
                            if inst.result_type == result_pointee_of.get(ptr).copied() {
                                use_sites.entry(*ptr).or_default().push((bi, ii, true));
                            } else {
                                disqualified.insert(*ptr);
                            }
                        }
                    }
                }
                Op::Store => {
                    if let Some(Operand::IdRef(ptr)) = inst.operands.first() {
                        if plan_ids.contains(ptr) {
                            let obj = match inst.operands.get(1) {
                                Some(Operand::IdRef(o)) => value_result_type(ctx, *o),
                                _ => None,
                            };
                            if obj == result_pointee_of.get(ptr).copied() {
                                use_sites.entry(*ptr).or_default().push((bi, ii, false));
                            } else {
                                disqualified.insert(*ptr);
                            }
                        }
                    }
                }
                _ => {
                    for op in &inst.operands {
                        if let Operand::IdRef(id) = op {
                            if plan_ids.contains(id) {
                                disqualified.insert(*id);
                            }
                        }
                    }
                }
            }
        }
    }
    plans.retain(|p| !disqualified.contains(&p.ac_id) && use_sites.contains_key(&p.ac_id));
    if plans.is_empty() {
        return Ok(());
    }

    let ulong_ty = ctx.ty_ulong();
    let uint_ty = ctx.ty_uint();
    let plan_by_id: HashMap<Word, (Word, Word, Word, Option<u32>)> = plans
        .iter()
        .map(|p| (p.ac_id, (p.base, p.base_pointee, p.idx_id, p.const_n)))
        .collect();
    let result_pointee_by_id: HashMap<Word, Word> =
        plans.iter().map(|p| (p.ac_id, p.result_pointee)).collect();
    let chain_at: HashMap<(usize, usize), Word> =
        plans.iter().map(|p| ((p.bi, p.ii), p.ac_id)).collect();
    let use_at: HashMap<(usize, usize), Word> = plans
        .iter()
        .flat_map(|p| {
            use_sites
                .get(&p.ac_id)
                .into_iter()
                .flatten()
                .map(move |&(bi, ii, _)| ((bi, ii), p.ac_id))
        })
        .collect();

    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let old = std::mem::take(&mut ctx.module.functions[entry_idx].blocks[bi].instructions);
        let mut newv: Vec<Instruction> = Vec::with_capacity(old.len() + 8);
        for (ii, inst) in old.into_iter().enumerate() {
            // The over-indexing chain itself is dead — every use now computes from the base directly.
            if chain_at.contains_key(&(bi, ii)) {
                continue;
            }
            if let Some(&ac_id) = use_at.get(&(bi, ii)) {
                let (base, base_pointee, idx_id, const_n) = plan_by_id[&ac_id];
                let result_pointee = result_pointee_by_id[&ac_id];
                // The bit offset of element N within the slot: `N*ELEM_BITS`, as a value id (None when
                // it is statically zero). Const folds to a literal; a dynamic index multiplies by
                // ELEM_BITS at runtime (valid: the slot holds SLOT_BITS/ELEM_BITS elements, so the
                // runtime index — and the shift — stays in range over the AIR's valid accesses).
                let shift_id: Option<Word> = match const_n {
                    Some(0) => None,
                    Some(n) => Some(ctx.const_uint(n * ELEM_BITS)),
                    None => {
                        let elem_const = ctx.const_uint(ELEM_BITS);
                        let id = ctx.module.fresh_id();
                        newv.push(Instruction::new(
                            Op::IMul,
                            Some(uint_ty),
                            Some(id),
                            vec![Operand::IdRef(idx_id), Operand::IdRef(elem_const)],
                        ));
                        Some(id)
                    }
                };
                // Load the whole slot and reinterpret it to a 64-bit unsigned int.
                let whole = ctx.module.fresh_id();
                newv.push(Instruction::new(
                    Op::Load,
                    Some(base_pointee),
                    Some(whole),
                    vec![Operand::IdRef(base)],
                ));
                let whole_u = if base_pointee == ulong_ty {
                    whole
                } else {
                    let id = ctx.module.fresh_id();
                    newv.push(Instruction::new(
                        Op::Bitcast,
                        Some(ulong_ty),
                        Some(id),
                        vec![Operand::IdRef(whole)],
                    ));
                    id
                };
                if inst.class.opcode == Op::Load {
                    let res = inst.result_id.ok_or("load has a result id")?;
                    // Shift the slot down to element N, truncate to 32 bits, bitcast to the result.
                    let shifted = match shift_id {
                        None => whole_u,
                        Some(sh) => {
                            let id = ctx.module.fresh_id();
                            newv.push(Instruction::new(
                                Op::ShiftRightLogical,
                                Some(ulong_ty),
                                Some(id),
                                vec![Operand::IdRef(whole_u), Operand::IdRef(sh)],
                            ));
                            id
                        }
                    };
                    let bitcast_needed = result_pointee != uint_ty;
                    let trunc = if bitcast_needed {
                        ctx.module.fresh_id()
                    } else {
                        res
                    };
                    newv.push(Instruction::new(
                        Op::UConvert,
                        Some(uint_ty),
                        Some(trunc),
                        vec![Operand::IdRef(shifted)],
                    ));
                    if bitcast_needed {
                        newv.push(Instruction::new(
                            Op::Bitcast,
                            Some(result_pointee),
                            Some(res),
                            vec![Operand::IdRef(trunc)],
                        ));
                    }
                } else {
                    // Read-modify-write: clear element N's ELEM_BITS, OR in the zero-extended object.
                    // The clear-mask is `~(low_elem_mask << shift)` — `OpNot` of the shifted low mask,
                    // so a dynamic shift is handled without a per-N constant.
                    let obj = match inst.operands.get(1) {
                        Some(Operand::IdRef(o)) => *o,
                        _ => return Err("scalar-slot store lost its object".to_string()),
                    };
                    let low_mask = ctx.const_int_of(ulong_ty, ((1u64 << ELEM_BITS) - 1) as i64);
                    let slot_mask = match shift_id {
                        None => low_mask,
                        Some(sh) => {
                            let id = ctx.module.fresh_id();
                            newv.push(Instruction::new(
                                Op::ShiftLeftLogical,
                                Some(ulong_ty),
                                Some(id),
                                vec![Operand::IdRef(low_mask), Operand::IdRef(sh)],
                            ));
                            id
                        }
                    };
                    let not_mask = ctx.module.fresh_id();
                    newv.push(Instruction::new(
                        Op::Not,
                        Some(ulong_ty),
                        Some(not_mask),
                        vec![Operand::IdRef(slot_mask)],
                    ));
                    let keep = ctx.module.fresh_id();
                    newv.push(Instruction::new(
                        Op::BitwiseAnd,
                        Some(ulong_ty),
                        Some(keep),
                        vec![Operand::IdRef(whole_u), Operand::IdRef(not_mask)],
                    ));
                    let obj_u = if result_pointee == uint_ty {
                        obj
                    } else {
                        let id = ctx.module.fresh_id();
                        newv.push(Instruction::new(
                            Op::Bitcast,
                            Some(uint_ty),
                            Some(id),
                            vec![Operand::IdRef(obj)],
                        ));
                        id
                    };
                    let obj_wide = ctx.module.fresh_id();
                    newv.push(Instruction::new(
                        Op::UConvert,
                        Some(ulong_ty),
                        Some(obj_wide),
                        vec![Operand::IdRef(obj_u)],
                    ));
                    let obj_shifted = match shift_id {
                        None => obj_wide,
                        Some(sh) => {
                            let id = ctx.module.fresh_id();
                            newv.push(Instruction::new(
                                Op::ShiftLeftLogical,
                                Some(ulong_ty),
                                Some(id),
                                vec![Operand::IdRef(obj_wide), Operand::IdRef(sh)],
                            ));
                            id
                        }
                    };
                    let combined = ctx.module.fresh_id();
                    newv.push(Instruction::new(
                        Op::BitwiseOr,
                        Some(ulong_ty),
                        Some(combined),
                        vec![Operand::IdRef(keep), Operand::IdRef(obj_shifted)],
                    ));
                    let stored = if base_pointee == ulong_ty {
                        combined
                    } else {
                        let id = ctx.module.fresh_id();
                        newv.push(Instruction::new(
                            Op::Bitcast,
                            Some(base_pointee),
                            Some(id),
                            vec![Operand::IdRef(combined)],
                        ));
                        id
                    };
                    newv.push(Instruction::new(
                        Op::Store,
                        None,
                        None,
                        vec![Operand::IdRef(base), Operand::IdRef(stored)],
                    ));
                }
                continue;
            }
            newv.push(inst);
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = newv;
    }
    Ok(())
}

/// Lower a reinterpreting scalar `OpLoad` whose Result Type does not match the access-chain pointee's
/// declared scalar type — the "N bits at address A" read Metal expresses through a typed buffer view.
///
/// An MPS-style heterogeneous-struct buffer is modelled in SPIR-V with every 4-byte slot declared as a
/// 32-bit `float`/`uint` (offsets sourced from `air.struct_type_info`). The AIR then reads a field at
/// its TRUE width through the 32-bit element pointer, which spirv-val rejects ("OpLoad Result Type
/// `<id>` does not match Pointer's type"):
///   - **widen** `OpLoad %ulong %p` — a 64-bit field is the two adjacent 32-bit slots; load slot `k`
///     and slot `k+1` and little-endian assemble (`lo | hi << 32`). Contiguity is PROVEN from the
///     declared member `Offset` / `ArrayStride` decorations, never assumed.
///   - **narrow** `OpLoad %uchar/%ushort %p` — a byte/short field is the low byte(s) of the slot; load
///     the slot and `OpUConvert`-truncate (little-endian, so the low bits are the byte at the address).
///   - **same-width** `OpLoad %uint %p` over a `float` slot (or vice-versa) — a bit reinterpret;
///     `OpLoad` the declared type then `OpBitcast`.
///     Each is the FAITHFUL value the load denotes (the exact bytes at the address every validated access
///     already trusts), so it is byte-correct on a little-endian target.
///
/// Byte-safe / floor-safe by construction: only loads that are CURRENTLY INVALID (Result Type ≠
/// declared pointee, i.e. spirv-val already rejects them) over a `StorageBuffer` pointer whose pointee
/// is a direct 32-bit scalar are touched — a banked/valid module (every load matches its pointee) never
/// matches, so the floor is provably untouched. The original access chain and all its other uses are
/// left intact; only the individual load instruction is rewritten (the widen forms a FRESH sibling
/// chain for slot `k+1`, never mutating the original). Decides purely from IR structure (storage class
/// + type walk + layout decorations), never a shader name.
pub(in crate::passes) fn rewrite_reinterpret_scalar_loads(ctx: &mut Ctx, entry_idx: usize) {
    // Pointer-type -> (storage class, pointee).
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

    // Declared layout: struct member byte offsets and array byte strides (the contiguity oracle).
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
                    Some(Operand::LiteralBit32(s)),
                ) = (
                    inst.operands.first(),
                    inst.operands.get(1),
                    inst.operands.get(2),
                ) {
                    array_stride.insert(*ty, *s);
                }
            }
            _ => {}
        }
    }

    let is_int = |ctx: &Ctx, ty: Word| -> bool {
        type_def_of(ctx, ty)
            .map(|d| d.class.opcode == Op::TypeInt)
            .unwrap_or(false)
    };
    let is_float = |ctx: &Ctx, ty: Word| -> bool {
        type_def_of(ctx, ty)
            .map(|d| d.class.opcode == Op::TypeFloat)
            .unwrap_or(false)
    };

    // Access-chain result id -> (block, instruction) for the defining chain (member-access or
    // PtrAccessChain — the latter is the byte-buffer element-stride form).
    let mut ac_at: HashMap<Word, (usize, usize)> = HashMap::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if matches!(
                inst.class.opcode,
                Op::InBoundsAccessChain | Op::AccessChain | Op::PtrAccessChain
            ) {
                if let Some(r) = inst.result_id {
                    ac_at.insert(r, (bi, ii));
                }
            }
        }
    }

    // The last-index of a fabricated sibling slot chain.
    #[derive(Clone)]
    enum SibIdx {
        Const(u32),        // a constant index value (struct member i+j, or constant array idx+j)
        DynAdd(Word, u32), // OpIAdd(base_index_value, j) — a dynamic array index + j
    }
    #[derive(Clone)]
    enum Kind {
        SameWidth, // W == V: reinterpret the bits via OpBitcast
        Narrow,    // W <  V: OpUConvert truncation to the low W bits
        Widen {
            // W = k*V: k-1 sibling slot chains, all sharing the original base + opcode + result ptr type.
            op: Op,
            base: Word,
            prefix: Vec<Operand>,
            siblings: Vec<SibIdx>,
        },
    }
    struct Plan {
        bi: usize,
        ii: usize,
        result_id: Word,
        result_ty: Word,
        pointee_ty: Word,
        ptr: Word,
        ptr_ty: Word,
        memops: Vec<Operand>,
        slot_v: u32,
        kind: Kind,
    }

    let mut plans: Vec<Plan> = Vec::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if inst.class.opcode != Op::Load {
                continue;
            }
            let (Some(result_id), Some(result_ty)) = (inst.result_id, inst.result_type) else {
                continue;
            };
            let Some(Operand::IdRef(ptr)) = inst.operands.first() else {
                continue;
            };
            let Some(ptr_ty) = value_result_type(ctx, *ptr) else {
                continue;
            };
            let Some(&(sc, pointee_ty)) = ptr_info.get(&ptr_ty) else {
                continue;
            };
            if sc != StorageClass::StorageBuffer {
                continue;
            }
            // Currently-INVALID only: a valid scalar load has Result Type == declared pointee.
            if result_ty == pointee_ty {
                continue;
            }
            let (Some(w), Some(v)) = (
                direct_scalar_width(ctx, result_ty),
                direct_scalar_width(ctx, pointee_ty),
            ) else {
                continue;
            };
            // A float slot is always a 32-bit word (the float<->uint slot the interface models); only
            // reinterpret it via the 32-bit `uint` bitcast. Integer slots may be any width (a uchar
            // byte buffer read wider, etc.).
            if is_float(ctx, pointee_ty) && v != 32 {
                continue;
            }
            let memops: Vec<Operand> = inst.operands.iter().skip(1).cloned().collect();
            let kind = if w == v {
                Kind::SameWidth
            } else if w < v {
                // Narrow truncation only lands in an integer result (OpUConvert).
                if !is_int(ctx, result_ty) {
                    continue;
                }
                Kind::Narrow
            } else {
                // Widen: integer result whose width is a whole multiple of the slot width.
                if !is_int(ctx, result_ty) || w % v != 0 {
                    continue;
                }
                let k = (w / v) as usize;
                let Some(&(abi, aii)) = ac_at.get(ptr) else {
                    continue;
                };
                let ac = &ctx.module.functions[entry_idx].blocks[abi].instructions[aii];
                let op = ac.class.opcode;
                let Some(Operand::IdRef(base)) = ac.operands.first() else {
                    continue;
                };
                let indices: Vec<Operand> = ac.operands[1..].to_vec();
                if indices.is_empty() {
                    continue;
                }
                let Some(base_ptr_ty) = value_result_type(ctx, *base) else {
                    continue;
                };
                let Some(&(_, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                    continue;
                };
                let slot_bytes = v / 8;
                let mut siblings: Vec<SibIdx> = Vec::with_capacity(k - 1);
                let mut ok = true;
                // Byte-buffer element-stride form: `OpPtrAccessChain %elem_ptr %p %elem` over a slot-
                // sized element (base ptr ArrayStride == slot) — the sibling bumps the single Element
                // index by j, striding j whole slots (contiguous by the stride decoration).
                if op == Op::PtrAccessChain {
                    if indices.len() != 1 {
                        continue;
                    }
                    if pointee_ty != base_pointee
                        || array_stride.get(&base_ptr_ty).copied() != Some(slot_bytes)
                    {
                        continue;
                    }
                    let Operand::IdRef(elem) = &indices[0] else {
                        continue;
                    };
                    if let Some(c) = const_u32(ctx, *elem) {
                        for j in 1..k as u32 {
                            siblings.push(SibIdx::Const(c + j));
                        }
                    } else {
                        for j in 1..k as u32 {
                            siblings.push(SibIdx::DynAdd(*elem, j));
                        }
                    }
                    if !ok || siblings.len() != k - 1 {
                        continue;
                    }
                    plans.push(Plan {
                        bi,
                        ii,
                        result_id,
                        result_ty,
                        pointee_ty,
                        ptr: *ptr,
                        ptr_ty,
                        memops,
                        slot_v: v,
                        kind: Kind::Widen {
                            op,
                            base: *base,
                            prefix: Vec::new(),
                            siblings,
                        },
                    });
                    continue;
                }
                let (prefix, last) = indices.split_at(indices.len() - 1);
                let last = &last[0];
                let Some(parent_ty) = walk_into_type(ctx, base_pointee, prefix) else {
                    continue;
                };
                let Some(pdef) = type_def_of(ctx, parent_ty) else {
                    continue;
                };
                match pdef.class.opcode {
                    Op::TypeStruct => {
                        // last index is a constant member i; members i..i+k-1 must be same-width
                        // scalars laid out contiguously (proven from the Offset decorations).
                        let Operand::IdRef(last_id) = last else {
                            continue;
                        };
                        let Some(i) = const_u32(ctx, *last_id) else {
                            continue;
                        };
                        let Some(&base_off) = member_offset.get(&(parent_ty, i)) else {
                            continue;
                        };
                        for j in 1..k as u32 {
                            let m = i + j;
                            let Some(Operand::IdRef(mty)) = pdef.operands.get(m as usize) else {
                                ok = false;
                                break;
                            };
                            if direct_scalar_width(ctx, *mty) != Some(v) {
                                ok = false;
                                break;
                            }
                            match member_offset.get(&(parent_ty, m)) {
                                Some(&off) if off == base_off + j * slot_bytes => {}
                                _ => {
                                    ok = false;
                                    break;
                                }
                            }
                            siblings.push(SibIdx::Const(m));
                        }
                    }
                    Op::TypeArray | Op::TypeRuntimeArray => {
                        // element must be the same-width slot scalar, contiguous (ArrayStride == slot).
                        let Some(Operand::IdRef(elem)) = pdef.operands.first() else {
                            continue;
                        };
                        if direct_scalar_width(ctx, *elem) != Some(v) {
                            continue;
                        }
                        if array_stride.get(&parent_ty).copied() != Some(slot_bytes) {
                            continue;
                        }
                        match last {
                            Operand::IdRef(last_id) => {
                                if let Some(c) = const_u32(ctx, *last_id) {
                                    for j in 1..k as u32 {
                                        siblings.push(SibIdx::Const(c + j));
                                    }
                                } else {
                                    for j in 1..k as u32 {
                                        siblings.push(SibIdx::DynAdd(*last_id, j));
                                    }
                                }
                            }
                            _ => continue,
                        }
                    }
                    _ => continue,
                }
                if !ok || siblings.len() != k - 1 {
                    continue;
                }
                Kind::Widen {
                    op,
                    base: *base,
                    prefix: prefix.to_vec(),
                    siblings,
                }
            };
            plans.push(Plan {
                bi,
                ii,
                result_id,
                result_ty,
                pointee_ty,
                ptr: *ptr,
                ptr_ty,
                memops,
                slot_v: v,
                kind,
            });
        }
    }
    if plans.is_empty() {
        return;
    }

    // Build the replacement instruction sequence for each rewritten load (needs fresh ids / constants).
    let uint_ty = ctx.ty_uint();
    let mut replacement: HashMap<(usize, usize), Vec<Instruction>> = HashMap::new();
    for plan in &plans {
        let pt = plan.pointee_ty;
        let rt = plan.result_ty;
        let mut seq: Vec<Instruction> = Vec::new();
        // Load the slot at the pointer in its DECLARED type.
        let lo = ctx.module.fresh_id();
        let mut lo_ops = vec![Operand::IdRef(plan.ptr)];
        lo_ops.extend(plan.memops.iter().cloned());
        seq.push(Instruction::new(Op::Load, Some(pt), Some(lo), lo_ops));
        // Coerce a slot value to its unsigned-int bit pattern.
        let to_word = |ctx: &mut Ctx, seq: &mut Vec<Instruction>, val: Word| -> Word {
            if is_float(ctx, pt) {
                let u = ctx.module.fresh_id();
                seq.push(Instruction::new(
                    Op::Bitcast,
                    Some(uint_ty),
                    Some(u),
                    vec![Operand::IdRef(val)],
                ));
                u
            } else {
                val
            }
        };
        match &plan.kind {
            Kind::SameWidth => {
                seq.push(Instruction::new(
                    Op::Bitcast,
                    Some(rt),
                    Some(plan.result_id),
                    vec![Operand::IdRef(lo)],
                ));
            }
            Kind::Narrow => {
                let src = to_word(ctx, &mut seq, lo);
                seq.push(Instruction::new(
                    Op::UConvert,
                    Some(rt),
                    Some(plan.result_id),
                    vec![Operand::IdRef(src)],
                ));
            }
            Kind::Widen {
                op,
                base,
                prefix,
                siblings,
            } => {
                let lo_i = to_word(ctx, &mut seq, lo);
                let lo_wide = ctx.module.fresh_id();
                seq.push(Instruction::new(
                    Op::UConvert,
                    Some(rt),
                    Some(lo_wide),
                    vec![Operand::IdRef(lo_i)],
                ));
                let mut acc = lo_wide;
                let n = siblings.len();
                for (idx, sib) in siblings.iter().enumerate() {
                    let j = (idx + 1) as u32;
                    let last_op = match sib {
                        SibIdx::Const(c) => Operand::IdRef(ctx.const_uint(*c)),
                        SibIdx::DynAdd(base_val, add) => {
                            let cadd = ctx.const_uint(*add);
                            let s = ctx.module.fresh_id();
                            seq.push(Instruction::new(
                                Op::IAdd,
                                Some(uint_ty),
                                Some(s),
                                vec![Operand::IdRef(*base_val), Operand::IdRef(cadd)],
                            ));
                            Operand::IdRef(s)
                        }
                    };
                    let pid = ctx.module.fresh_id();
                    let mut ops = vec![Operand::IdRef(*base)];
                    ops.extend(prefix.iter().cloned());
                    ops.push(last_op);
                    seq.push(Instruction::new(*op, Some(plan.ptr_ty), Some(pid), ops));
                    let hi = ctx.module.fresh_id();
                    let mut hi_ops = vec![Operand::IdRef(pid)];
                    hi_ops.extend(plan.memops.iter().cloned());
                    seq.push(Instruction::new(Op::Load, Some(pt), Some(hi), hi_ops));
                    let hi_i = to_word(ctx, &mut seq, hi);
                    let hi_wide = ctx.module.fresh_id();
                    seq.push(Instruction::new(
                        Op::UConvert,
                        Some(rt),
                        Some(hi_wide),
                        vec![Operand::IdRef(hi_i)],
                    ));
                    let shift = ctx.const_int_of(rt, (j * plan.slot_v) as i64);
                    let shifted = ctx.module.fresh_id();
                    seq.push(Instruction::new(
                        Op::ShiftLeftLogical,
                        Some(rt),
                        Some(shifted),
                        vec![Operand::IdRef(hi_wide), Operand::IdRef(shift)],
                    ));
                    let or_id = if idx + 1 == n {
                        plan.result_id
                    } else {
                        ctx.module.fresh_id()
                    };
                    seq.push(Instruction::new(
                        Op::BitwiseOr,
                        Some(rt),
                        Some(or_id),
                        vec![Operand::IdRef(acc), Operand::IdRef(shifted)],
                    ));
                    acc = or_id;
                }
            }
        }
        replacement.insert((plan.bi, plan.ii), seq);
    }

    // Splice each replacement in place of its original load.
    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let old = std::mem::take(&mut ctx.module.functions[entry_idx].blocks[bi].instructions);
        let mut newv: Vec<Instruction> = Vec::with_capacity(old.len() + 8);
        for (ii, inst) in old.into_iter().enumerate() {
            if let Some(seq) = replacement.get(&(bi, ii)) {
                newv.extend(seq.iter().cloned());
            } else {
                newv.push(inst);
            }
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = newv;
    }
}
