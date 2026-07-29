//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

/// Rewrite an INVALID single-DYNAMIC-index access chain into a single-member Block buffer that also
/// REINTERPRETS the element to a same-width scalar. The AIR lowered a `device float*` view of a
/// `device uint*` runtime-array buffer (or vice-versa) as
/// `OpInBoundsAccessChain %_ptr_StorageBuffer_float %buf %dyn` — ONE dynamic index applied directly to
/// the Block struct (illegal: spirv-val "the <id> passed to OpInBoundsAccessChain to index into a
/// structure must be an OpConstant") AND a `float` result over a `uint` runtime array. The byte-correct
/// lowering descends member-0 into the runtime array (`%buf %uint_0 %dyn` → a `uint` element pointer),
/// loads the `uint`, and `OpBitcast`s it to `float` — bit-EXACT because both are 32-bit, so it is
/// correct regardless of which type is the buffer's "real" layout (the bits are identical either way).
/// A store through the chain bitcasts the `float` object to `uint` first.
///
/// Byte-safe by construction — only chains that are CURRENTLY INVALID are touched, gated on:
/// (1) exactly ONE index, and it is NOT an `OpConstant` (a dynamic struct-member select — the illegal
///     form; a banked/valid module never matches, so the floor is provably untouched);
/// (2) the base pointee is a SINGLE-member struct whose member-0 is an array/runtime-array of a DIRECT
///     scalar element E (so member-0 insertion is unambiguous and `%dyn` is the array's element index);
/// (3) the result pointee P and E are BOTH direct int/float scalars of EQUAL bit width (so the
///     value reinterpret is a bit-exact `OpBitcast`, never a width change);
/// (4) EVERY use of the chain result is an `OpLoad`(P) or `OpStore`(_, P) — if any other use exists the
///     chain is skipped (we must not hand a retyped pointer to an unmodelled consumer).
/// Decides purely from IR structure (storage class + type walk + use kinds), never a shader name.
pub(in crate::passes) fn rewrite_dynamic_struct_index_reinterpret(
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

    // The single member-0 array element of a single-member struct, if that member is an array/
    // runtime-array of a direct scalar.
    let struct_single_array_elem = |ctx: &Ctx, struct_ty: Word| -> Option<Word> {
        let def = type_def_of(ctx, struct_ty)?;
        if def.class.opcode != Op::TypeStruct || def.operands.len() != 1 {
            return None;
        }
        let Operand::IdRef(member0) = def.operands.first()? else {
            return None;
        };
        let mdef = type_def_of(ctx, *member0)?;
        if !matches!(mdef.class.opcode, Op::TypeArray | Op::TypeRuntimeArray) {
            return None;
        }
        let Operand::IdRef(elem) = mdef.operands.first()? else {
            return None;
        };
        direct_scalar_width(ctx, *elem).map(|_| *elem)
    };

    // Phase 1: collect candidate chains (read-only).
    struct Plan {
        bi: usize,
        ii: usize,
        ac_id: Word,
        base: Word,
        dyn_idx: Word,
        elem_ty: Word,
        result_pointee: Word,
        sc: StorageClass,
        reinterpret: bool,
    }
    let mut plans: Vec<Plan> = Vec::new();
    let blocks = &ctx.module.functions[entry_idx].blocks;
    for (bi, block) in blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if !matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain) {
                continue;
            }
            // [base, idx0] — exactly one index.
            if inst.operands.len() != 2 {
                continue;
            }
            let (Some(ac_id), Some(result_type)) = (inst.result_id, inst.result_type) else {
                continue;
            };
            let Some(&(sc, result_pointee)) = ptr_info.get(&result_type) else {
                continue;
            };
            let Some(Operand::IdRef(base)) = inst.operands.first() else {
                continue;
            };
            let Some(Operand::IdRef(dyn_idx)) = inst.operands.get(1) else {
                continue;
            };
            // The single index must be DYNAMIC (a non-constant) — the illegal struct-member select.
            if const_u32(ctx, *dyn_idx).is_some() {
                continue;
            }
            let Some(base_ptr_ty) = value_result_type(ctx, *base) else {
                continue;
            };
            let Some(&(_, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            let Some(elem_ty) = struct_single_array_elem(ctx, base_pointee) else {
                continue;
            };
            // Same-width direct scalar reinterpret (or identity if the element already matches).
            let (Some(ew), Some(pw)) = (
                direct_scalar_width(ctx, elem_ty),
                direct_scalar_width(ctx, result_pointee),
            ) else {
                continue;
            };
            if ew != pw {
                continue;
            }
            plans.push(Plan {
                bi,
                ii,
                ac_id,
                base: *base,
                dyn_idx: *dyn_idx,
                elem_ty,
                result_pointee,
                sc,
                reinterpret: result_pointee != elem_ty,
            });
        }
    }
    if plans.is_empty() {
        return Ok(());
    }

    // Verify every use of each chain result is an OpLoad(result_pointee)/OpStore(_, result_pointee);
    // disqualify a chain otherwise. Returns the (bi, ii, is_load) of each accepted use.
    let mut use_sites: HashMap<Word, Vec<(usize, usize, bool)>> = HashMap::new();
    let mut disqualified: HashSet<Word> = HashSet::new();
    let plan_ids: HashSet<Word> = plans.iter().map(|p| p.ac_id).collect();
    let result_pointee_of: HashMap<Word, Word> =
        plans.iter().map(|p| (p.ac_id, p.result_pointee)).collect();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            // The defining AC instruction itself is not a "use".
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
                    // Any other operand reference to the chain disqualifies it.
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

    // Phase 2: allocate the member-0 constant + new element pointer types, then apply edits.
    let member0 = ctx.const_uint(0);
    let mut new_ptr_ty: HashMap<(StorageClass, Word), Word> = HashMap::new();
    for p in &plans {
        new_ptr_ty
            .entry((p.sc, p.elem_ty))
            .or_insert_with(|| ctx.ty_ptr(p.sc, p.elem_ty));
    }

    // Pre-allocate fresh ids for each load split (the new element-typed load id).
    let load_split_id: HashMap<(usize, usize), Word> = plans
        .iter()
        .flat_map(|p| use_sites.get(&p.ac_id).into_iter().flatten())
        .filter(|(_, _, is_load)| *is_load)
        .map(|&(bi, ii, _)| ((bi, ii), 0))
        .collect::<HashMap<_, _>>()
        .into_keys()
        .map(|k| (k, ctx.module.fresh_id()))
        .collect();
    let store_cast_id: HashMap<(usize, usize), Word> = plans
        .iter()
        .flat_map(|p| use_sites.get(&p.ac_id).into_iter().flatten())
        .filter(|(_, _, is_load)| !*is_load)
        .map(|&(bi, ii, _)| ((bi, ii), 0))
        .collect::<HashMap<_, _>>()
        .into_keys()
        .map(|k| (k, ctx.module.fresh_id()))
        .collect();

    // Rewrite the AC instructions in place (operand count goes 1 -> 2; result type to the element ptr).
    let ac_at: HashMap<(usize, usize), &Plan> = plans.iter().map(|p| ((p.bi, p.ii), p)).collect();
    let elem_ty_of_load: HashMap<(usize, usize), (Word, Word)> = plans
        .iter()
        .flat_map(|p| {
            use_sites
                .get(&p.ac_id)
                .into_iter()
                .flatten()
                .map(move |&(bi, ii, is_load)| ((bi, ii), (p.elem_ty, p.result_pointee, is_load)))
        })
        .map(|((bi, ii), (e, rp, _))| ((bi, ii), (e, rp)))
        .collect();

    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let old = std::mem::take(&mut ctx.module.functions[entry_idx].blocks[bi].instructions);
        let mut newv: Vec<Instruction> = Vec::with_capacity(old.len() + 4);
        for (ii, inst) in old.into_iter().enumerate() {
            if let Some(p) = ac_at.get(&(bi, ii)) {
                let new_ty = new_ptr_ty[&(p.sc, p.elem_ty)];
                newv.push(Instruction::new(
                    inst.class.opcode,
                    Some(new_ty),
                    Some(p.ac_id),
                    vec![
                        Operand::IdRef(p.base),
                        Operand::IdRef(member0),
                        Operand::IdRef(p.dyn_idx),
                    ],
                ));
                let _ = p.reinterpret;
                continue;
            }
            if let Some(&load_id) = load_split_id.get(&(bi, ii)) {
                let (elem_ty, result_pointee) = elem_ty_of_load[&(bi, ii)];
                let ptr = match inst.operands.first() {
                    Some(Operand::IdRef(p)) => *p,
                    _ => return Err("load split site lost its pointer operand".to_string()),
                };
                let res = inst.result_id.ok_or("load has a result id")?;
                // Preserve any trailing memory-access operand (e.g. Aligned) on the element load.
                let mut load_ops = vec![Operand::IdRef(ptr)];
                load_ops.extend(inst.operands.iter().skip(1).cloned());
                newv.push(Instruction::new(
                    Op::Load,
                    Some(elem_ty),
                    Some(load_id),
                    load_ops,
                ));
                if elem_ty == result_pointee {
                    // Identity (no reinterpret needed): rebind the original id to a copy of the load.
                    newv.push(Instruction::new(
                        Op::CopyObject,
                        Some(result_pointee),
                        Some(res),
                        vec![Operand::IdRef(load_id)],
                    ));
                } else {
                    newv.push(Instruction::new(
                        Op::Bitcast,
                        Some(result_pointee),
                        Some(res),
                        vec![Operand::IdRef(load_id)],
                    ));
                }
                continue;
            }
            if let Some(&cast_id) = store_cast_id.get(&(bi, ii)) {
                let (elem_ty, _result_pointee) = elem_ty_of_load[&(bi, ii)];
                let ptr = match inst.operands.first() {
                    Some(Operand::IdRef(p)) => *p,
                    _ => return Err("store cast site lost its pointer operand".to_string()),
                };
                let obj = match inst.operands.get(1) {
                    Some(Operand::IdRef(o)) => *o,
                    _ => return Err("store cast site lost its object operand".to_string()),
                };
                newv.push(Instruction::new(
                    Op::Bitcast,
                    Some(elem_ty),
                    Some(cast_id),
                    vec![Operand::IdRef(obj)],
                ));
                // Preserve any trailing memory-access operand (e.g. Aligned) on the store.
                let mut store_ops = vec![Operand::IdRef(ptr), Operand::IdRef(cast_id)];
                store_ops.extend(inst.operands.iter().skip(2).cloned());
                newv.push(Instruction::new(Op::Store, None, None, store_ops));
                continue;
            }
            newv.push(inst);
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = newv;
    }
    Ok(())
}

/// Rewrite an invalid dynamic sub-word view of a raw word buffer-block element.
///
/// This is the sub-word sibling of [`rewrite_dynamic_struct_index_reinterpret`]. Raw-buffer retries
/// model device memory as `{ RuntimeArray<uint> }`; a later typed `half*`/`ushort*`/`uchar*` view can
/// survive as `OpInBoundsAccessChain %ptr_half %buf %dyn`, which is invalid because `%dyn` indexes the
/// wrapper struct. It is also not a same-width reinterpret: `%dyn` is a sub-word element index over a
/// 32-bit backing word. The byte-correct lowering uses `%dyn / lanes_per_word` to address member-0's
/// `uint` word and `%dyn & (lanes_per_word - 1)` to extract or replace the selected 8/16-bit lane.
///
/// Floor-safe gates: one non-constant index, `StorageBuffer`, base pointee exactly a single-member
/// array/runtime-array of unsigned 32-bit words, result pointee exactly an 8/16-bit int/float scalar, and
/// every use is a plain `OpLoad`/`OpStore` of that pointee. Anything else remains visible to
/// spirv-val instead of guessing.
#[derive(Clone, Copy)]
struct DynamicSubwordPlan {
    bi: usize,
    ii: usize,
    opcode: Op,
    ac_id: Word,
    base: Word,
    dyn_idx: Word,
    index_ty: Word,
    result_pointee: Word,
    result_bits: u32,
}

pub(in crate::passes) fn rewrite_dynamic_struct_index_subword_reinterpret(
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
            if let (Some(id), Some(Operand::StorageClass(sc)), Some(Operand::IdRef(pointee))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*sc, *pointee));
            }
        }
    }

    let mut plans = Vec::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if !matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain)
                || inst.operands.len() != 2
            {
                continue;
            }
            let (Some(ac_id), Some(result_ptr_ty)) = (inst.result_id, inst.result_type) else {
                continue;
            };
            let Some(&(sc, result_pointee)) = ptr_info.get(&result_ptr_ty) else {
                continue;
            };
            let Some(result_bits) = direct_scalar_width(ctx, result_pointee) else {
                continue;
            };
            if sc != StorageClass::StorageBuffer
                || !(8..32).contains(&result_bits)
                || 32 % result_bits != 0
            {
                continue;
            }
            let (Some(Operand::IdRef(base)), Some(Operand::IdRef(dyn_idx))) =
                (inst.operands.first(), inst.operands.get(1))
            else {
                continue;
            };
            if const_u32(ctx, *dyn_idx).is_some() {
                continue;
            }
            let Some(index_ty) = value_result_type(ctx, *dyn_idx) else {
                continue;
            };
            if type_def_of(ctx, index_ty).is_none_or(|def| def.class.opcode != Op::TypeInt) {
                continue;
            }
            let Some(base_ptr_ty) = value_result_type(ctx, *base) else {
                continue;
            };
            let Some(&(_, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            let Some(elem_ty) = single_member_array_scalar_elem(ctx, base_pointee) else {
                continue;
            };
            if !is_unsigned_int_width(ctx, elem_ty, 32) {
                continue;
            }
            if walk_into_type(ctx, base_pointee, &inst.operands[1..]).is_some() {
                continue;
            }
            plans.push(DynamicSubwordPlan {
                bi,
                ii,
                opcode: inst.class.opcode,
                ac_id,
                base: *base,
                dyn_idx: *dyn_idx,
                index_ty,
                result_pointee,
                result_bits,
            });
        }
    }
    if plans.is_empty() {
        return Ok(());
    }

    let plan_ids: HashSet<Word> = plans.iter().map(|plan| plan.ac_id).collect();
    let result_pointee_of: HashMap<Word, Word> = plans
        .iter()
        .map(|plan| (plan.ac_id, plan.result_pointee))
        .collect();
    let mut use_sites: HashMap<Word, Vec<(usize, usize, bool)>> = HashMap::new();
    let mut disqualified = HashSet::new();
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
                let exact_load = inst.class.opcode == Op::Load
                    && inst.operands.len() == 1
                    && inst.result_type == result_pointee_of.get(&chain_id).copied();
                let exact_store = inst.class.opcode == Op::Store
                    && inst.operands.len() == 2
                    && inst.operands.first() == Some(&Operand::IdRef(chain_id))
                    && inst.operands.get(1).and_then(|operand| match operand {
                        Operand::IdRef(value) => value_result_type(ctx, *value),
                        _ => None,
                    }) == result_pointee_of.get(&chain_id).copied();
                if exact_load {
                    use_sites.entry(chain_id).or_default().push((bi, ii, true));
                } else if exact_store {
                    use_sites.entry(chain_id).or_default().push((bi, ii, false));
                } else {
                    disqualified.insert(chain_id);
                }
            }
        }
    }
    plans.retain(|plan| !disqualified.contains(&plan.ac_id) && use_sites.contains_key(&plan.ac_id));
    if plans.is_empty() {
        return Ok(());
    }

    let uint = ctx.ty_uint();
    let ptr_uint = ctx.ty_ptr(StorageClass::StorageBuffer, uint);
    let member0 = ctx.const_uint(0);
    let chain_at: HashSet<(usize, usize)> = plans.iter().map(|plan| (plan.bi, plan.ii)).collect();
    let plan_by_id: HashMap<Word, DynamicSubwordPlan> =
        plans.iter().map(|plan| (plan.ac_id, *plan)).collect();
    let use_at: HashMap<(usize, usize), Word> = plans
        .iter()
        .flat_map(|plan| {
            use_sites
                .get(&plan.ac_id)
                .into_iter()
                .flatten()
                .map(move |&(bi, ii, _)| ((bi, ii), plan.ac_id))
        })
        .collect();

    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let old = std::mem::take(&mut ctx.module.functions[entry_idx].blocks[bi].instructions);
        let mut rewritten = Vec::with_capacity(old.len() + 10);
        for (ii, inst) in old.into_iter().enumerate() {
            if chain_at.contains(&(bi, ii)) {
                continue;
            }
            let Some(chain_id) = use_at.get(&(bi, ii)).copied() else {
                rewritten.push(inst);
                continue;
            };
            let plan = plan_by_id[&chain_id];
            let (word_ptr, shift_bits) =
                emit_dynamic_halfword_address(ctx, &plan, ptr_uint, member0, &mut rewritten);
            match inst.class.opcode {
                Op::Load => {
                    let result_uint_ty = uint_type_of_width(ctx, plan.result_bits);
                    let result = inst.result_id.ok_or("subword load has a result id")?;
                    let loaded_word = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::Load,
                        Some(uint),
                        Some(loaded_word),
                        vec![Operand::IdRef(word_ptr)],
                    ));
                    let shifted = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::ShiftRightLogical,
                        Some(uint),
                        Some(shifted),
                        vec![Operand::IdRef(loaded_word), Operand::IdRef(shift_bits)],
                    ));
                    let mask = ctx.const_uint((1u32 << plan.result_bits) - 1);
                    let masked = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::BitwiseAnd,
                        Some(uint),
                        Some(masked),
                        vec![Operand::IdRef(shifted), Operand::IdRef(mask)],
                    ));
                    let narrowed = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::UConvert,
                        Some(result_uint_ty),
                        Some(narrowed),
                        vec![Operand::IdRef(masked)],
                    ));
                    let op = if plan.result_pointee == result_uint_ty {
                        Op::CopyObject
                    } else {
                        Op::Bitcast
                    };
                    rewritten.push(Instruction::new(
                        op,
                        Some(plan.result_pointee),
                        Some(result),
                        vec![Operand::IdRef(narrowed)],
                    ));
                }
                Op::Store => {
                    let result_uint_ty = uint_type_of_width(ctx, plan.result_bits);
                    let object = match inst.operands.get(1) {
                        Some(Operand::IdRef(value)) => *value,
                        _ => return Err("subword store lost its object operand".to_string()),
                    };
                    let object_bits = if plan.result_pointee == result_uint_ty {
                        object
                    } else {
                        let bits = ctx.module.fresh_id();
                        rewritten.push(Instruction::new(
                            Op::Bitcast,
                            Some(result_uint_ty),
                            Some(bits),
                            vec![Operand::IdRef(object)],
                        ));
                        bits
                    };
                    let object_word = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::UConvert,
                        Some(uint),
                        Some(object_word),
                        vec![Operand::IdRef(object_bits)],
                    ));
                    let shifted_object = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::ShiftLeftLogical,
                        Some(uint),
                        Some(shifted_object),
                        vec![Operand::IdRef(object_word), Operand::IdRef(shift_bits)],
                    ));
                    let lane_mask_base = ctx.const_uint((1u32 << plan.result_bits) - 1);
                    let lane_mask = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::ShiftLeftLogical,
                        Some(uint),
                        Some(lane_mask),
                        vec![Operand::IdRef(lane_mask_base), Operand::IdRef(shift_bits)],
                    ));
                    let keep_mask = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::Not,
                        Some(uint),
                        Some(keep_mask),
                        vec![Operand::IdRef(lane_mask)],
                    ));
                    let old_word = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::Load,
                        Some(uint),
                        Some(old_word),
                        vec![Operand::IdRef(word_ptr)],
                    ));
                    let kept_word = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::BitwiseAnd,
                        Some(uint),
                        Some(kept_word),
                        vec![Operand::IdRef(old_word), Operand::IdRef(keep_mask)],
                    ));
                    let merged = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::BitwiseOr,
                        Some(uint),
                        Some(merged),
                        vec![Operand::IdRef(kept_word), Operand::IdRef(shifted_object)],
                    ));
                    rewritten.push(Instruction::new(
                        Op::Store,
                        None,
                        None,
                        vec![Operand::IdRef(word_ptr), Operand::IdRef(merged)],
                    ));
                }
                _ => return Err("subword dynamic-chain use was not load/store".to_string()),
            }
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = rewritten;
    }
    Ok(())
}

fn is_unsigned_int_width(ctx: &Ctx, ty: Word, bits: u32) -> bool {
    type_def_of(ctx, ty).is_some_and(|def| {
        def.class.opcode == Op::TypeInt
            && def.operands.first() == Some(&Operand::LiteralBit32(bits))
            && def.operands.get(1) == Some(&Operand::LiteralBit32(0))
    })
}

fn emit_dynamic_halfword_address(
    ctx: &mut Ctx,
    plan: &DynamicSubwordPlan,
    ptr_uint: Word,
    member0: Word,
    out: &mut Vec<Instruction>,
) -> (Word, Word) {
    let lanes_per_word = 32 / plan.result_bits;
    let divisor = ctx.const_int_of(plan.index_ty, i64::from(lanes_per_word));
    let word_idx = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::UDiv,
        Some(plan.index_ty),
        Some(word_idx),
        vec![Operand::IdRef(plan.dyn_idx), Operand::IdRef(divisor)],
    ));
    let lane_mask = ctx.const_int_of(plan.index_ty, i64::from(lanes_per_word - 1));
    let lane = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::BitwiseAnd,
        Some(plan.index_ty),
        Some(lane),
        vec![Operand::IdRef(plan.dyn_idx), Operand::IdRef(lane_mask)],
    ));
    let uint = ctx.ty_uint();
    let lane32 = if plan.index_ty == uint {
        lane
    } else {
        let converted = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::UConvert,
            Some(uint),
            Some(converted),
            vec![Operand::IdRef(lane)],
        ));
        converted
    };
    let lane_bits = ctx.const_uint(plan.result_bits);
    let shift_bits = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::IMul,
        Some(uint),
        Some(shift_bits),
        vec![Operand::IdRef(lane32), Operand::IdRef(lane_bits)],
    ));
    let word_ptr = ctx.module.fresh_id();
    out.push(Instruction::new(
        plan.opcode,
        Some(ptr_uint),
        Some(word_ptr),
        vec![
            Operand::IdRef(plan.base),
            Operand::IdRef(member0),
            Operand::IdRef(word_idx),
        ],
    ));
    (word_ptr, shift_bits)
}

#[derive(Clone, Copy)]
struct DynamicWideWordPlan {
    bi: usize,
    ii: usize,
    opcode: Op,
    ac_id: Word,
    base: Word,
    dyn_idx: Word,
    index_ty: Word,
    result_pointee: Word,
    result_bits: u32,
}

/// Rewrite an invalid dynamic 64-bit scalar view of a raw word buffer-block element.
///
/// Raw-buffer retries use `{ RuntimeArray<uint> }`. A `ulong*`/`long*`/`double*` view can survive as
/// `OpInBoundsAccessChain %ptr_ulong %buf %dyn`: the single dynamic index illegally indexes the Block
/// struct, and the desired element spans two backing words. For plain loads/stores, lower `%dyn` to
/// `word = dyn * 2`, access member-0 at `word` and `word + 1`, and assemble/split the 64-bit scalar in
/// little-endian order.
pub(in crate::passes) fn rewrite_dynamic_struct_index_wide_word_reinterpret(
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
            if let (Some(id), Some(Operand::StorageClass(sc)), Some(Operand::IdRef(pointee))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*sc, *pointee));
            }
        }
    }

    let mut plans = Vec::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if !matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain)
                || inst.operands.len() != 2
            {
                continue;
            }
            let (Some(ac_id), Some(result_ptr_ty)) = (inst.result_id, inst.result_type) else {
                continue;
            };
            let Some(&(sc, result_pointee)) = ptr_info.get(&result_ptr_ty) else {
                continue;
            };
            let Some(result_bits) = direct_scalar_width(ctx, result_pointee) else {
                continue;
            };
            if sc != StorageClass::StorageBuffer || result_bits != 64 {
                continue;
            }
            let (Some(Operand::IdRef(base)), Some(Operand::IdRef(dyn_idx))) =
                (inst.operands.first(), inst.operands.get(1))
            else {
                continue;
            };
            if const_u32(ctx, *dyn_idx).is_some() {
                continue;
            }
            let Some(index_ty) = value_result_type(ctx, *dyn_idx) else {
                continue;
            };
            if type_def_of(ctx, index_ty).is_none_or(|def| def.class.opcode != Op::TypeInt) {
                continue;
            }
            let Some(base_ptr_ty) = value_result_type(ctx, *base) else {
                continue;
            };
            let Some(&(_, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            let Some(elem_ty) = single_member_array_scalar_elem(ctx, base_pointee) else {
                continue;
            };
            if !is_unsigned_int_width(ctx, elem_ty, 32) {
                continue;
            }
            if walk_into_type(ctx, base_pointee, &inst.operands[1..]).is_some() {
                continue;
            }
            plans.push(DynamicWideWordPlan {
                bi,
                ii,
                opcode: inst.class.opcode,
                ac_id,
                base: *base,
                dyn_idx: *dyn_idx,
                index_ty,
                result_pointee,
                result_bits,
            });
        }
    }
    if plans.is_empty() {
        return Ok(());
    }

    let plan_ids: HashSet<Word> = plans.iter().map(|plan| plan.ac_id).collect();
    let result_pointee_of: HashMap<Word, Word> = plans
        .iter()
        .map(|plan| (plan.ac_id, plan.result_pointee))
        .collect();
    let mut use_sites: HashMap<Word, Vec<(usize, usize, bool)>> = HashMap::new();
    let mut disqualified = HashSet::new();
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
                let exact_load = inst.class.opcode == Op::Load
                    && inst.operands.len() == 1
                    && inst.result_type == result_pointee_of.get(&chain_id).copied();
                let exact_store = inst.class.opcode == Op::Store
                    && inst.operands.len() == 2
                    && inst.operands.first() == Some(&Operand::IdRef(chain_id))
                    && inst.operands.get(1).and_then(|operand| match operand {
                        Operand::IdRef(value) => value_result_type(ctx, *value),
                        _ => None,
                    }) == result_pointee_of.get(&chain_id).copied();
                if exact_load {
                    use_sites.entry(chain_id).or_default().push((bi, ii, true));
                } else if exact_store {
                    use_sites.entry(chain_id).or_default().push((bi, ii, false));
                } else {
                    disqualified.insert(chain_id);
                }
            }
        }
    }
    plans.retain(|plan| !disqualified.contains(&plan.ac_id) && use_sites.contains_key(&plan.ac_id));
    if plans.is_empty() {
        return Ok(());
    }

    let uint = ctx.ty_uint();
    let ptr_uint = ctx.ty_ptr(StorageClass::StorageBuffer, uint);
    let member0 = ctx.const_uint(0);
    let chain_at: HashSet<(usize, usize)> = plans.iter().map(|plan| (plan.bi, plan.ii)).collect();
    let plan_by_id: HashMap<Word, DynamicWideWordPlan> =
        plans.iter().map(|plan| (plan.ac_id, *plan)).collect();
    let use_at: HashMap<(usize, usize), Word> = plans
        .iter()
        .flat_map(|plan| {
            use_sites
                .get(&plan.ac_id)
                .into_iter()
                .flatten()
                .map(move |&(bi, ii, _)| ((bi, ii), plan.ac_id))
        })
        .collect();

    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let old = std::mem::take(&mut ctx.module.functions[entry_idx].blocks[bi].instructions);
        let mut rewritten = Vec::with_capacity(old.len() + 10);
        for (ii, inst) in old.into_iter().enumerate() {
            if chain_at.contains(&(bi, ii)) {
                continue;
            }
            let Some(chain_id) = use_at.get(&(bi, ii)).copied() else {
                rewritten.push(inst);
                continue;
            };
            let plan = plan_by_id[&chain_id];
            let result_uint_ty = uint_type_of_width(ctx, plan.result_bits);
            let word_base = emit_dynamic_wide_word_base(ctx, &plan, &mut rewritten);
            match inst.class.opcode {
                Op::Load => {
                    let result = inst.result_id.ok_or("wide-word load has a result id")?;
                    let mut assembled: Option<Word> = None;
                    for word_lane in 0..2 {
                        let word_ptr = emit_dynamic_word_lane_pointer(
                            ctx,
                            &plan,
                            ptr_uint,
                            member0,
                            word_base,
                            word_lane,
                            &mut rewritten,
                        );
                        let loaded = ctx.module.fresh_id();
                        rewritten.push(Instruction::new(
                            Op::Load,
                            Some(uint),
                            Some(loaded),
                            vec![Operand::IdRef(word_ptr)],
                        ));
                        let widened = ctx.module.fresh_id();
                        rewritten.push(Instruction::new(
                            Op::UConvert,
                            Some(result_uint_ty),
                            Some(widened),
                            vec![Operand::IdRef(loaded)],
                        ));
                        let placed = if word_lane == 0 {
                            widened
                        } else {
                            let shift = ctx.const_uint(32);
                            let id = ctx.module.fresh_id();
                            rewritten.push(Instruction::new(
                                Op::ShiftLeftLogical,
                                Some(result_uint_ty),
                                Some(id),
                                vec![Operand::IdRef(widened), Operand::IdRef(shift)],
                            ));
                            id
                        };
                        assembled = Some(match assembled {
                            None => placed,
                            Some(prev) => {
                                let id = ctx.module.fresh_id();
                                rewritten.push(Instruction::new(
                                    Op::BitwiseOr,
                                    Some(result_uint_ty),
                                    Some(id),
                                    vec![Operand::IdRef(prev), Operand::IdRef(placed)],
                                ));
                                id
                            }
                        });
                    }
                    let packed =
                        assembled.ok_or("wide-word load should assemble at least one word")?;
                    let op = if plan.result_pointee == result_uint_ty {
                        Op::CopyObject
                    } else {
                        Op::Bitcast
                    };
                    rewritten.push(Instruction::new(
                        op,
                        Some(plan.result_pointee),
                        Some(result),
                        vec![Operand::IdRef(packed)],
                    ));
                }
                Op::Store => {
                    let object = match inst.operands.get(1) {
                        Some(Operand::IdRef(value)) => *value,
                        _ => return Err("wide-word store lost its object operand".to_string()),
                    };
                    let object_bits = if plan.result_pointee == result_uint_ty {
                        object
                    } else {
                        let bits = ctx.module.fresh_id();
                        rewritten.push(Instruction::new(
                            Op::Bitcast,
                            Some(result_uint_ty),
                            Some(bits),
                            vec![Operand::IdRef(object)],
                        ));
                        bits
                    };
                    for word_lane in 0..2 {
                        let shifted = if word_lane == 0 {
                            object_bits
                        } else {
                            let shift = ctx.const_uint(32);
                            let id = ctx.module.fresh_id();
                            rewritten.push(Instruction::new(
                                Op::ShiftRightLogical,
                                Some(result_uint_ty),
                                Some(id),
                                vec![Operand::IdRef(object_bits), Operand::IdRef(shift)],
                            ));
                            id
                        };
                        let word_value = ctx.module.fresh_id();
                        rewritten.push(Instruction::new(
                            Op::UConvert,
                            Some(uint),
                            Some(word_value),
                            vec![Operand::IdRef(shifted)],
                        ));
                        let word_ptr = emit_dynamic_word_lane_pointer(
                            ctx,
                            &plan,
                            ptr_uint,
                            member0,
                            word_base,
                            word_lane,
                            &mut rewritten,
                        );
                        rewritten.push(Instruction::new(
                            Op::Store,
                            None,
                            None,
                            vec![Operand::IdRef(word_ptr), Operand::IdRef(word_value)],
                        ));
                    }
                }
                _ => return Err("wide-word dynamic-chain use was not load/store".to_string()),
            }
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = rewritten;
    }
    Ok(())
}

fn emit_dynamic_wide_word_base(
    ctx: &mut Ctx,
    plan: &DynamicWideWordPlan,
    out: &mut Vec<Instruction>,
) -> Word {
    let words_per_value = ctx.const_int_of(plan.index_ty, i64::from(plan.result_bits / 32));
    let word_base = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::IMul,
        Some(plan.index_ty),
        Some(word_base),
        vec![
            Operand::IdRef(plan.dyn_idx),
            Operand::IdRef(words_per_value),
        ],
    ));
    word_base
}

fn emit_dynamic_word_lane_pointer(
    ctx: &mut Ctx,
    plan: &DynamicWideWordPlan,
    ptr_uint: Word,
    member0: Word,
    word_base: Word,
    word_lane: u32,
    out: &mut Vec<Instruction>,
) -> Word {
    let word_index = if word_lane == 0 {
        word_base
    } else {
        let offset = ctx.const_int_of(plan.index_ty, i64::from(word_lane));
        let id = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::IAdd,
            Some(plan.index_ty),
            Some(id),
            vec![Operand::IdRef(word_base), Operand::IdRef(offset)],
        ));
        id
    };
    let ptr = ctx.module.fresh_id();
    out.push(Instruction::new(
        plan.opcode,
        Some(ptr_uint),
        Some(ptr),
        vec![
            Operand::IdRef(plan.base),
            Operand::IdRef(member0),
            Operand::IdRef(word_index),
        ],
    ));
    ptr
}

/// Rewrite an invalid dynamic vector view of a single-member runtime-array buffer into scalar lane
/// loads or stores.  This is the vector counterpart of
/// [`rewrite_dynamic_struct_index_reinterpret`]: LLVM's `getelementptr <N x T>, ptr %buf, i` can
/// arrive as an access chain with one dynamic index directly into the Block struct, even when the
/// actual backing is `{ runtime-array<E> }`.  A struct member index must be constant in SPIR-V, and
/// the vector's element stride must be made explicit.
///
/// The replacement addresses lane `j` as `buf[0][i * N + j]`, then reconstructs/extracts the vector.
/// It is byte-exact when `E` and the vector component have the same direct-scalar width: that is the
/// same contiguous `N * sizeof(E)` range LLVM's vector GEP names.  The pass accepts only direct
/// exact-typed loads/stores of the otherwise-invalid pointer, so pointer escapes, atomics, memory
/// operands, non-vector aggregates, and already-valid chains are left untouched.
pub(in crate::passes) fn rewrite_dynamic_struct_index_vector_reinterpret(
    ctx: &mut Ctx,
    entry_idx: usize,
) -> Result<(), String> {
    #[derive(Clone, Copy)]
    struct Plan {
        bi: usize,
        ii: usize,
        opcode: Op,
        ac_id: Word,
        base: Word,
        dyn_idx: Word,
        index_ty: Word,
        elem_ty: Word,
        component_ty: Word,
        result_pointee: Word,
        sc: StorageClass,
        lanes: u32,
    }

    let mut ptr_info: HashMap<Word, (StorageClass, Word)> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if inst.class.opcode == Op::TypePointer {
            if let (Some(id), Some(Operand::StorageClass(sc)), Some(Operand::IdRef(pointee))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*sc, *pointee));
            }
        }
    }

    let vector_shape = |ctx: &Ctx, ty: Word| -> Option<(Word, u32)> {
        let def = type_def_of(ctx, ty)?;
        if def.class.opcode != Op::TypeVector {
            return None;
        }
        let (Operand::IdRef(component), Operand::LiteralBit32(lanes)) =
            (def.operands.first()?, def.operands.get(1)?)
        else {
            return None;
        };
        (*lanes >= 2 && direct_scalar_width(ctx, *component).is_some())
            .then_some((*component, *lanes))
    };
    let integer_index = |ctx: &Ctx, ty: Word| {
        type_def_of(ctx, ty).is_some_and(|def| def.class.opcode == Op::TypeInt)
    };

    let mut plans = Vec::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if !matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain)
                || inst.operands.len() != 2
            {
                continue;
            }
            let (Some(ac_id), Some(result_ptr_ty)) = (inst.result_id, inst.result_type) else {
                continue;
            };
            let Some(&(sc, result_pointee)) = ptr_info.get(&result_ptr_ty) else {
                continue;
            };
            let Some((component_ty, lanes)) = vector_shape(ctx, result_pointee) else {
                continue;
            };
            let (Some(Operand::IdRef(base)), Some(Operand::IdRef(dyn_idx))) =
                (inst.operands.first(), inst.operands.get(1))
            else {
                continue;
            };
            if const_u32(ctx, *dyn_idx).is_some() {
                continue;
            }
            let Some(index_ty) = value_result_type(ctx, *dyn_idx) else {
                continue;
            };
            if !integer_index(ctx, index_ty) {
                continue;
            }
            let Some(base_ptr_ty) = value_result_type(ctx, *base) else {
                continue;
            };
            let Some(&(_, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            let Some(elem_ty) = single_member_array_scalar_elem(ctx, base_pointee) else {
                continue;
            };
            if direct_scalar_width(ctx, elem_ty) != direct_scalar_width(ctx, component_ty) {
                continue;
            }
            // Preserve the invalid-only gate: a chain that already walks its base type is not ours
            // to alter, regardless of whether its result happens to be a vector.
            if walk_into_type(ctx, base_pointee, &inst.operands[1..]).is_some() {
                continue;
            }
            plans.push(Plan {
                bi,
                ii,
                opcode: inst.class.opcode,
                ac_id,
                base: *base,
                dyn_idx: *dyn_idx,
                index_ty,
                elem_ty,
                component_ty,
                result_pointee,
                sc,
                lanes,
            });
        }
    }
    if plans.is_empty() {
        return Ok(());
    }

    // A vector view has no legal pointer representation over the scalar runtime array.  Make sure
    // every use is a plain load/store before deleting that pointer and replaying its scalar lanes.
    let plan_ids: HashSet<Word> = plans.iter().map(|p| p.ac_id).collect();
    let result_pointee_of: HashMap<Word, Word> =
        plans.iter().map(|p| (p.ac_id, p.result_pointee)).collect();
    let mut use_sites: HashMap<Word, Vec<(usize, usize, bool)>> = HashMap::new();
    let mut disqualified = HashSet::new();
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
                let exact_load = inst.class.opcode == Op::Load
                    && inst.operands.len() == 1
                    && inst.result_type == result_pointee_of.get(&chain_id).copied();
                let exact_store = inst.class.opcode == Op::Store
                    && inst.operands.len() == 2
                    && matches!(inst.operands.first(), Some(Operand::IdRef(id)) if *id == chain_id)
                    && inst.operands.get(1).and_then(|operand| match operand {
                        Operand::IdRef(value) => value_result_type(ctx, *value),
                        _ => None,
                    }) == result_pointee_of.get(&chain_id).copied();
                if exact_load {
                    use_sites.entry(chain_id).or_default().push((bi, ii, true));
                } else if exact_store {
                    use_sites.entry(chain_id).or_default().push((bi, ii, false));
                } else {
                    disqualified.insert(chain_id);
                }
            }
        }
    }
    plans.retain(|plan| !disqualified.contains(&plan.ac_id) && use_sites.contains_key(&plan.ac_id));
    if plans.is_empty() {
        return Ok(());
    }

    let member0 = ctx.const_uint(0);
    let mut elem_ptr_ty = HashMap::new();
    for plan in &plans {
        elem_ptr_ty
            .entry((plan.sc, plan.elem_ty))
            .or_insert_with(|| ctx.ty_ptr(plan.sc, plan.elem_ty));
    }
    let plan_by_id: HashMap<Word, Plan> = plans.iter().map(|plan| (plan.ac_id, *plan)).collect();
    let chain_at: HashMap<(usize, usize), Word> = plans
        .iter()
        .map(|plan| ((plan.bi, plan.ii), plan.ac_id))
        .collect();
    let use_at: HashMap<(usize, usize), Word> = plans
        .iter()
        .flat_map(|plan| {
            use_sites
                .get(&plan.ac_id)
                .into_iter()
                .flatten()
                .map(move |&(bi, ii, _)| ((bi, ii), plan.ac_id))
        })
        .collect();

    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let old = std::mem::take(&mut ctx.module.functions[entry_idx].blocks[bi].instructions);
        let mut rewritten = Vec::with_capacity(old.len() + 8);
        for (ii, inst) in old.into_iter().enumerate() {
            if chain_at.contains_key(&(bi, ii)) {
                // All verified uses are replayed below, so the illegal pointer itself disappears.
                continue;
            }
            let Some(chain_id) = use_at.get(&(bi, ii)).copied() else {
                rewritten.push(inst);
                continue;
            };
            let plan = &plan_by_id[&chain_id];
            let lane_base = ctx.module.fresh_id();
            let lane_factor = ctx.const_int_of(plan.index_ty, i64::from(plan.lanes));
            rewritten.push(Instruction::new(
                Op::IMul,
                Some(plan.index_ty),
                Some(lane_base),
                vec![Operand::IdRef(plan.dyn_idx), Operand::IdRef(lane_factor)],
            ));

            let mut lane_ptrs = Vec::with_capacity(plan.lanes as usize);
            for lane in 0..plan.lanes {
                let lane_index = if lane == 0 {
                    lane_base
                } else {
                    let lane_offset = ctx.const_int_of(plan.index_ty, i64::from(lane));
                    let lane_index = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::IAdd,
                        Some(plan.index_ty),
                        Some(lane_index),
                        vec![Operand::IdRef(lane_base), Operand::IdRef(lane_offset)],
                    ));
                    lane_index
                };
                let ptr = ctx.module.fresh_id();
                rewritten.push(Instruction::new(
                    plan.opcode,
                    Some(elem_ptr_ty[&(plan.sc, plan.elem_ty)]),
                    Some(ptr),
                    vec![
                        Operand::IdRef(plan.base),
                        Operand::IdRef(member0),
                        Operand::IdRef(lane_index),
                    ],
                ));
                lane_ptrs.push(ptr);
            }

            match inst.class.opcode {
                Op::Load => {
                    let result = inst.result_id.ok_or("vector load has a result id")?;
                    let mut components = Vec::with_capacity(plan.lanes as usize);
                    for ptr in lane_ptrs {
                        let loaded = ctx.module.fresh_id();
                        rewritten.push(Instruction::new(
                            Op::Load,
                            Some(plan.elem_ty),
                            Some(loaded),
                            vec![Operand::IdRef(ptr)],
                        ));
                        let component = if plan.elem_ty == plan.component_ty {
                            loaded
                        } else {
                            let cast = ctx.module.fresh_id();
                            rewritten.push(Instruction::new(
                                Op::Bitcast,
                                Some(plan.component_ty),
                                Some(cast),
                                vec![Operand::IdRef(loaded)],
                            ));
                            cast
                        };
                        components.push(Operand::IdRef(component));
                    }
                    rewritten.push(Instruction::new(
                        Op::CompositeConstruct,
                        Some(plan.result_pointee),
                        Some(result),
                        components,
                    ));
                }
                Op::Store => {
                    let object = match inst.operands.get(1) {
                        Some(Operand::IdRef(value)) => *value,
                        _ => return Err("vector store lost its object operand".to_string()),
                    };
                    for (lane, ptr) in lane_ptrs.into_iter().enumerate() {
                        let component = ctx.module.fresh_id();
                        rewritten.push(Instruction::new(
                            Op::CompositeExtract,
                            Some(plan.component_ty),
                            Some(component),
                            vec![Operand::IdRef(object), Operand::LiteralBit32(lane as u32)],
                        ));
                        let element = if plan.elem_ty == plan.component_ty {
                            component
                        } else {
                            let cast = ctx.module.fresh_id();
                            rewritten.push(Instruction::new(
                                Op::Bitcast,
                                Some(plan.elem_ty),
                                Some(cast),
                                vec![Operand::IdRef(component)],
                            ));
                            cast
                        };
                        rewritten.push(Instruction::new(
                            Op::Store,
                            None,
                            None,
                            vec![Operand::IdRef(ptr), Operand::IdRef(element)],
                        ));
                    }
                }
                _ => return Err("vector dynamic-chain use was not load/store".to_string()),
            }
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = rewritten;
    }
    Ok(())
}

/// Replace an invalid dynamic member access into a homogeneous struct with value selection.
///
/// LLVM bitcasts sometimes expose `[N x T]` private/function storage as a SPIR-V `OpTypeStruct`
/// containing N identical `T` members. A dynamic `getelementptr` into the original array then becomes
/// `OpInBoundsAccessChain %ptr_T %base %dyn`, which is invalid because SPIR-V struct indices must be
/// constants. If every struct member has the same type and the dynamic pointer is only loaded, the
/// value is exactly a select over constant-member loads.
pub(in crate::passes) fn rewrite_dynamic_homogeneous_struct_index_load(
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
            if let (Some(id), Some(Operand::StorageClass(sc)), Some(Operand::IdRef(pointee))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*sc, *pointee));
            }
        }
    }

    fn homogeneous_struct_member_path(
        ctx: &Ctx,
        ty: Word,
        result_pointee: Word,
    ) -> Option<(Word, u32, Vec<u32>)> {
        let def = type_def_of(ctx, ty)?;
        if def.class.opcode != Op::TypeStruct || def.operands.is_empty() {
            return None;
        }
        let first = match def.operands.first()? {
            Operand::IdRef(member) => *member,
            _ => return None,
        };
        if def
            .operands
            .iter()
            .all(|operand| matches!(operand, Operand::IdRef(member) if *member == first))
            && first == result_pointee
        {
            return Some((first, def.operands.len() as u32, Vec::new()));
        }
        if def.operands.len() == 1 {
            let (member, len, mut prefix) =
                homogeneous_struct_member_path(ctx, first, result_pointee)?;
            prefix.insert(0, 0);
            Some((member, len, prefix))
        } else {
            None
        }
    }
    let bool_select_type = |ctx: &mut Ctx, ty: Word| -> Option<Word> {
        match type_def_of(ctx, ty).map(|def| def.class.opcode) {
            Some(Op::TypeInt | Op::TypeFloat | Op::TypeBool | Op::TypePointer) => {
                Some(ctx.ty_bool())
            }
            Some(Op::TypeVector) => {
                let def = type_def_of(ctx, ty)?;
                let Some(Operand::LiteralBit32(lanes)) = def.operands.get(1) else {
                    return None;
                };
                Some(ctx.ty_vec_bool(*lanes))
            }
            _ => None,
        }
    };
    let selectable_member_type = |ctx: &Ctx, ty: Word| -> bool {
        match type_def_of(ctx, ty).map(|def| def.class.opcode) {
            Some(Op::TypeInt | Op::TypeFloat | Op::TypeBool | Op::TypePointer) => true,
            Some(Op::TypeVector) => type_def_of(ctx, ty)
                .is_some_and(|def| matches!(def.operands.get(1), Some(Operand::LiteralBit32(_)))),
            _ => false,
        }
    };

    #[derive(Clone)]
    struct Plan {
        bi: usize,
        ii: usize,
        chain_id: Word,
        base: Word,
        dyn_idx: Word,
        index_ty: Word,
        member_ty: Word,
        ptr_ty: Word,
        opcode: Op,
        members: u32,
        prefix: Vec<u32>,
    }
    let mut plans = Vec::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if !matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain)
                || inst.operands.len() != 2
            {
                continue;
            }
            let (Some(chain_id), Some(ptr_ty)) = (inst.result_id, inst.result_type) else {
                continue;
            };
            let Some(&(_, result_pointee)) = ptr_info.get(&ptr_ty) else {
                continue;
            };
            let (Some(Operand::IdRef(base)), Some(Operand::IdRef(dyn_idx))) =
                (inst.operands.first(), inst.operands.get(1))
            else {
                continue;
            };
            if const_u32(ctx, *dyn_idx).is_some() {
                continue;
            }
            let Some(index_ty) = value_result_type(ctx, *dyn_idx) else {
                continue;
            };
            if type_def_of(ctx, index_ty).is_none_or(|def| def.class.opcode != Op::TypeInt) {
                continue;
            }
            let Some(base_ptr_ty) = value_result_type(ctx, *base) else {
                continue;
            };
            let Some(&(_, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            let Some((member_ty, members, prefix)) =
                homogeneous_struct_member_path(ctx, base_pointee, result_pointee)
            else {
                continue;
            };
            if !selectable_member_type(ctx, member_ty) {
                continue;
            }
            plans.push(Plan {
                bi,
                ii,
                chain_id,
                base: *base,
                dyn_idx: *dyn_idx,
                index_ty,
                member_ty,
                ptr_ty,
                opcode: inst.class.opcode,
                members,
                prefix,
            });
        }
    }
    if plans.is_empty() {
        return Ok(());
    }

    let plan_ids: HashSet<Word> = plans.iter().map(|plan| plan.chain_id).collect();
    let mut use_sites: HashMap<Word, Vec<(usize, usize)>> = HashMap::new();
    let mut disqualified = HashSet::new();
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
                if inst.class.opcode == Op::Load
                    && inst.operands.len() == 1
                    && matches!(inst.operands.first(), Some(Operand::IdRef(id)) if *id == chain_id)
                {
                    use_sites.entry(chain_id).or_default().push((bi, ii));
                } else {
                    disqualified.insert(chain_id);
                }
            }
        }
    }
    plans.retain(|plan| {
        !disqualified.contains(&plan.chain_id) && use_sites.contains_key(&plan.chain_id)
    });
    if plans.is_empty() {
        return Ok(());
    }

    let plan_by_id: HashMap<Word, Plan> = plans
        .iter()
        .map(|plan| (plan.chain_id, plan.clone()))
        .collect();
    let chain_at: HashMap<(usize, usize), Word> = plans
        .iter()
        .map(|plan| ((plan.bi, plan.ii), plan.chain_id))
        .collect();
    let use_at: HashMap<(usize, usize), Word> = plans
        .iter()
        .flat_map(|plan| {
            use_sites
                .get(&plan.chain_id)
                .into_iter()
                .flatten()
                .map(move |&(bi, ii)| ((bi, ii), plan.chain_id))
        })
        .collect();

    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let old = std::mem::take(&mut ctx.module.functions[entry_idx].blocks[bi].instructions);
        let mut rewritten = Vec::with_capacity(old.len() + 16);
        for (ii, inst) in old.into_iter().enumerate() {
            if chain_at.contains_key(&(bi, ii)) {
                continue;
            }
            let Some(chain_id) = use_at.get(&(bi, ii)).copied() else {
                rewritten.push(inst);
                continue;
            };
            let plan = &plan_by_id[&chain_id];
            let result = inst
                .result_id
                .ok_or("dynamic homogeneous struct load lost result id")?;
            let Some(cond_ty) = bool_select_type(ctx, plan.member_ty) else {
                rewritten.push(inst);
                continue;
            };
            let mut selected = None;
            for member in 0..plan.members {
                let member_idx = ctx.const_int_of(plan.index_ty, i64::from(member));
                let ptr = ctx.module.fresh_id();
                let mut chain_ops = Vec::with_capacity(plan.prefix.len() + 2);
                chain_ops.push(Operand::IdRef(plan.base));
                for prefix in &plan.prefix {
                    chain_ops.push(Operand::IdRef(ctx.const_uint(*prefix)));
                }
                chain_ops.push(Operand::IdRef(member_idx));
                rewritten.push(Instruction::new(
                    plan.opcode,
                    Some(plan.ptr_ty),
                    Some(ptr),
                    chain_ops,
                ));
                let loaded = ctx.module.fresh_id();
                rewritten.push(Instruction::new(
                    Op::Load,
                    Some(plan.member_ty),
                    Some(loaded),
                    vec![Operand::IdRef(ptr)],
                ));
                let Some(prev) = selected else {
                    selected = Some(loaded);
                    continue;
                };
                let cmp = ctx.module.fresh_id();
                rewritten.push(Instruction::new(
                    Op::IEqual,
                    Some(ctx.ty_bool()),
                    Some(cmp),
                    vec![Operand::IdRef(plan.dyn_idx), Operand::IdRef(member_idx)],
                ));
                let cond = if cond_ty == ctx.ty_bool() {
                    cmp
                } else {
                    let cond = ctx.module.fresh_id();
                    let lanes = type_def_of(ctx, cond_ty)
                        .and_then(|def| match def.operands.get(1) {
                            Some(Operand::LiteralBit32(lanes)) => Some(*lanes),
                            _ => None,
                        })
                        .ok_or("vector bool type missing lane count")?;
                    rewritten.push(Instruction::new(
                        Op::CompositeConstruct,
                        Some(cond_ty),
                        Some(cond),
                        (0..lanes).map(|_| Operand::IdRef(cmp)).collect(),
                    ));
                    cond
                };
                let select = if member + 1 == plan.members {
                    result
                } else {
                    ctx.module.fresh_id()
                };
                rewritten.push(Instruction::new(
                    Op::Select,
                    Some(plan.member_ty),
                    Some(select),
                    vec![
                        Operand::IdRef(cond),
                        Operand::IdRef(loaded),
                        Operand::IdRef(prev),
                    ],
                ));
                selected = Some(select);
            }
            if plan.members == 1 {
                let selected = selected.ok_or("homogeneous struct member load missing value")?;
                rewritten.push(Instruction::new(
                    Op::CopyObject,
                    Some(plan.member_ty),
                    Some(result),
                    vec![Operand::IdRef(selected)],
                ));
            }
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = rewritten;
    }
    Ok(())
}

/// The single member-0 element of a single-member struct `{ array/runtime-array<E> }`, when `E` is a
/// DIRECT scalar; `None` otherwise. (Shared shape test for the buffer-block reinterpret rewrites.)
pub(in crate::passes) fn single_member_array_scalar_elem(
    ctx: &Ctx,
    struct_ty: Word,
) -> Option<Word> {
    let def = type_def_of(ctx, struct_ty)?;
    if def.class.opcode != Op::TypeStruct || def.operands.len() != 1 {
        return None;
    }
    let Operand::IdRef(member0) = def.operands.first()? else {
        return None;
    };
    let mdef = type_def_of(ctx, *member0)?;
    if !matches!(mdef.class.opcode, Op::TypeArray | Op::TypeRuntimeArray) {
        return None;
    }
    let Operand::IdRef(elem) = mdef.operands.first()? else {
        return None;
    };
    direct_scalar_width(ctx, *elem).map(|_| *elem)
}

/// Rewrite an INVALID *nested-chained* element reinterpret over a buffer-block runtime array. The AIR
/// took a `device float*` element pointer into a `{ runtime-array<float> }` SSBO, REINTERPRET-cast it
/// to `device uint*`, and GEP'd it — which metal2vulkan lowers as a two-index chain over the float element:
///
/// ```text
///   %inner = OpInBoundsAccessChain %_ptr_StorageBuffer_float %buf  %uint_0 %dynF   ; element #dynF (VALID)
///   %out   = OpInBoundsAccessChain %_ptr_StorageBuffer_uint  %inner %uint_0 %dynU   ; reinterpret (INVALID)
/// ```
///
/// `%out`'s `[%uint_0, %dynU]` indexes into the scalar `float` `%inner` points at (spirv-val "reached
/// non-composite"). Because `float` and `uint` are both the array's element stride (4 bytes), the
/// byte-correct address is element `dynF + dynU` of the SAME float runtime array, so this re-roots
/// `%out` onto `%buf` at the summed index and loads the `float`, `OpBitcast`ing it to `uint`:
///
/// ```text
///   %sum   = OpIAdd %uint %dynF %dynU
///   %out   = OpInBoundsAccessChain %_ptr_StorageBuffer_float %buf %uint_0 %sum   ; float element ptr
///   ; at each load: %f = OpLoad %float %out ; %res = OpBitcast %uint %f
/// ```
///
/// Byte-EXACT: `dynF` strides whole floats and `dynU` strides whole uints, both 4 bytes, so
/// `(dynF + dynU) * 4` is the same byte offset the reinterpret-GEP addressed; the loaded bits are
/// identical under a same-width int/float `OpBitcast`. Stores bitcast the object back to the element
/// scalar first. `%inner` is left intact (it may have other, valid uses); only `%out` is re-rooted.
///
/// Byte-safe by construction — only chains that are CURRENTLY INVALID are touched, gated on:
/// (1) `%out` is a two-index chain `[%uint_0, %dynU]` whose result pointee P is a DIRECT scalar;
/// (2) its base `%inner` is itself a two-index chain `[%uint_0, %dynF]` into a SINGLE-member struct
///     `{ array/runtime-array<E> }` (so `%buf %uint_0 %sum` is a valid element pointer) whose element E
///     is a DIRECT scalar with `width(E) == width(P)` (so the value reinterpret is a bit-exact
///     `OpBitcast`, never a width change), and whose result pointee equals E;
/// (3) the base storage class permits the re-rooted descent (StorageBuffer/Workgroup/PSB);
/// (4) `%out` is CURRENTLY INVALID (the index walk over `%inner`'s pointee fails — always true for a
///     scalar pointee, so a banked/valid module never matches and the floor is provably untouched);
/// (5) EVERY use of `%out` is an `OpLoad`(P) or `OpStore`(_, P) — any other use skips the chain.
/// Decides purely from IR structure (storage class + type walk + use kinds), never a shader name.
pub(in crate::passes) fn rewrite_chained_element_reinterpret(
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

    // The defining chain instruction of every access-chain result (for resolving `%inner`).
    let mut chain_def: HashMap<Word, Instruction> = HashMap::new();
    for block in &ctx.module.functions[entry_idx].blocks {
        for inst in &block.instructions {
            if matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain) {
                if let Some(rid) = inst.result_id {
                    chain_def.insert(rid, inst.clone());
                }
            }
        }
    }

    // A two-index chain `[base, %uint_0, %idx]` -> (base, idx) when its first index is constant 0.
    let two_index_zero_lead = |ctx: &Ctx, inst: &Instruction| -> Option<(Word, Word)> {
        if inst.operands.len() != 3 {
            return None;
        }
        let (Operand::IdRef(base), Operand::IdRef(i0), Operand::IdRef(i1)) =
            (&inst.operands[0], &inst.operands[1], &inst.operands[2])
        else {
            return None;
        };
        if const_u32(ctx, *i0) != Some(0) {
            return None;
        }
        Some((*base, *i1))
    };

    struct Plan {
        bi: usize,
        ii: usize,
        ac_id: Word,
        buf: Word,
        inner_idx: Word,
        out_idx: Word,
        elem_ty: Word,
        result_pointee: Word,
        sc: StorageClass,
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
            let Some((inner, out_idx)) = two_index_zero_lead(ctx, inst) else {
                continue;
            };
            let Some(&(sc, result_pointee)) = ptr_info.get(&result_type) else {
                continue;
            };
            if !ptr_access_chain_allowed_storage(sc) {
                continue;
            }
            let Some(pw) = direct_scalar_width(ctx, result_pointee) else {
                continue;
            };
            // The base must itself be a `[base, %uint_0, %dynF]` chain into a single-member array struct.
            let Some(inner_inst) = chain_def.get(&inner) else {
                continue;
            };
            let Some((buf, inner_idx)) = two_index_zero_lead(ctx, inner_inst) else {
                continue;
            };
            let Some(inner_result_type) = inner_inst.result_type else {
                continue;
            };
            let Some(&(_, inner_pointee)) = ptr_info.get(&inner_result_type) else {
                continue;
            };
            let Some(buf_ptr_ty) = value_result_type(ctx, buf) else {
                continue;
            };
            let Some(&(_, buf_pointee)) = ptr_info.get(&buf_ptr_ty) else {
                continue;
            };
            let Some(elem_ty) = single_member_array_scalar_elem(ctx, buf_pointee) else {
                continue;
            };
            // `%inner` must genuinely point at the array's element scalar (so `dynF` is the element
            // index we sum with `dynU`), and that element width must match the reinterpret width.
            if inner_pointee != elem_ty {
                continue;
            }
            if direct_scalar_width(ctx, elem_ty) != Some(pw) {
                continue;
            }
            // `%out` must be CURRENTLY INVALID: walking `[%uint_0, %dynU]` into `%inner`'s scalar
            // pointee must fail (always does for a scalar — this is a belt-and-braces floor guard).
            if walk_into_type(ctx, inner_pointee, &inst.operands[1..]).is_some() {
                continue;
            }
            plans.push(Plan {
                bi,
                ii,
                ac_id,
                buf,
                inner_idx,
                out_idx,
                elem_ty,
                result_pointee,
                sc,
            });
        }
    }
    if plans.is_empty() {
        return Ok(());
    }

    // Every use of each `%out` must be an OpLoad(P)/OpStore(_, P); disqualify otherwise.
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

    // Phase 2: allocate the member-0 const, the summed-index ids, the element pointer types, and the
    // per-use load/store split ids.
    let member0 = ctx.const_uint(0);
    let uint_ty = ctx.ty_uint();
    let mut new_ptr_ty: HashMap<(StorageClass, Word), Word> = HashMap::new();
    let mut sum_id: HashMap<Word, Word> = HashMap::new();
    for p in &plans {
        new_ptr_ty
            .entry((p.sc, p.elem_ty))
            .or_insert_with(|| ctx.ty_ptr(p.sc, p.elem_ty));
        sum_id.insert(p.ac_id, ctx.module.fresh_id());
    }
    let load_split_id: HashMap<(usize, usize), Word> = plans
        .iter()
        .flat_map(|p| use_sites.get(&p.ac_id).into_iter().flatten())
        .filter(|(_, _, is_load)| *is_load)
        .map(|&(bi, ii, _)| ((bi, ii), 0))
        .collect::<HashMap<_, _>>()
        .into_keys()
        .map(|k| (k, ctx.module.fresh_id()))
        .collect();
    let store_cast_id: HashMap<(usize, usize), Word> = plans
        .iter()
        .flat_map(|p| use_sites.get(&p.ac_id).into_iter().flatten())
        .filter(|(_, _, is_load)| !*is_load)
        .map(|&(bi, ii, _)| ((bi, ii), 0))
        .collect::<HashMap<_, _>>()
        .into_keys()
        .map(|k| (k, ctx.module.fresh_id()))
        .collect();

    let ac_at: HashMap<(usize, usize), &Plan> = plans.iter().map(|p| ((p.bi, p.ii), p)).collect();
    let elem_ty_of_use: HashMap<(usize, usize), (Word, Word)> = plans
        .iter()
        .flat_map(|p| {
            use_sites
                .get(&p.ac_id)
                .into_iter()
                .flatten()
                .map(move |&(bi, ii, _)| ((bi, ii), (p.elem_ty, p.result_pointee)))
        })
        .collect();

    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let old = std::mem::take(&mut ctx.module.functions[entry_idx].blocks[bi].instructions);
        let mut newv: Vec<Instruction> = Vec::with_capacity(old.len() + 4);
        for (ii, inst) in old.into_iter().enumerate() {
            if let Some(p) = ac_at.get(&(bi, ii)) {
                // `%sum = OpIAdd %uint %dynF %dynU` then re-rooted float-element chain over `%buf`.
                let sum = sum_id[&p.ac_id];
                newv.push(Instruction::new(
                    Op::IAdd,
                    Some(uint_ty),
                    Some(sum),
                    vec![Operand::IdRef(p.inner_idx), Operand::IdRef(p.out_idx)],
                ));
                let new_ty = new_ptr_ty[&(p.sc, p.elem_ty)];
                newv.push(Instruction::new(
                    inst.class.opcode,
                    Some(new_ty),
                    Some(p.ac_id),
                    vec![
                        Operand::IdRef(p.buf),
                        Operand::IdRef(member0),
                        Operand::IdRef(sum),
                    ],
                ));
                continue;
            }
            if let Some(&load_id) = load_split_id.get(&(bi, ii)) {
                let (elem_ty, result_pointee) = elem_ty_of_use[&(bi, ii)];
                let ptr = match inst.operands.first() {
                    Some(Operand::IdRef(p)) => *p,
                    _ => return Err("load split site lost its pointer operand".to_string()),
                };
                let res = inst.result_id.ok_or("load has a result id")?;
                let mut load_ops = vec![Operand::IdRef(ptr)];
                load_ops.extend(inst.operands.iter().skip(1).cloned());
                newv.push(Instruction::new(
                    Op::Load,
                    Some(elem_ty),
                    Some(load_id),
                    load_ops,
                ));
                newv.push(Instruction::new(
                    Op::Bitcast,
                    Some(result_pointee),
                    Some(res),
                    vec![Operand::IdRef(load_id)],
                ));
                continue;
            }
            if let Some(&cast_id) = store_cast_id.get(&(bi, ii)) {
                let (elem_ty, _result_pointee) = elem_ty_of_use[&(bi, ii)];
                let ptr = match inst.operands.first() {
                    Some(Operand::IdRef(p)) => *p,
                    _ => return Err("store cast site lost its pointer operand".to_string()),
                };
                let obj = match inst.operands.get(1) {
                    Some(Operand::IdRef(o)) => *o,
                    _ => return Err("store cast site lost its object operand".to_string()),
                };
                newv.push(Instruction::new(
                    Op::Bitcast,
                    Some(elem_ty),
                    Some(cast_id),
                    vec![Operand::IdRef(obj)],
                ));
                let mut store_ops = vec![Operand::IdRef(ptr), Operand::IdRef(cast_id)];
                store_ops.extend(inst.operands.iter().skip(2).cloned());
                newv.push(Instruction::new(Op::Store, None, None, store_ops));
                continue;
            }
            newv.push(inst);
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = newv;
    }
    Ok(())
}

/// Rewrite an INVALID *nested-chained* element reinterpret over a NARROW (byte/short) buffer-block array
/// where the reinterpret WIDENS — a `device uchar*`/`device ushort*` element pointer reinterpret-cast to
/// a WIDER scalar (`ushort`/`uint`/`ulong`) and GEP'd:
///
/// ```text
///   %inner = OpInBoundsAccessChain %_ptr_StorageBuffer_uchar  %buf   %uint_0 %byteIdx   ; byte #byteIdx (VALID)
///   %out   = OpInBoundsAccessChain %_ptr_StorageBuffer_ushort %inner %k                 ; reinterpret (INVALID)
/// ```
///
/// `%out`'s single index `%k` strides whole `ushort`s over the `uchar` `%inner` points at (spirv-val
/// "reached non-composite"). The byte-correct address is byte `byteIdx + k*R` of the SAME `uchar` runtime
/// array, where `R = width(ushort)/width(uchar) = 2` is how many narrow slots one wide element spans. So
/// the ushort is the `R` contiguous narrow slots starting at `sum = byteIdx + k*R`, little-endian:
///
/// ```text
///   %mul = OpIMul %uint %k %R ; %sum = OpIAdd %uint %byteIdx %mul
///   ; LOAD:  for j in 0..R: load uchar at slot sum+j, UConvert to ushort, shift left j*V, OR-assemble
///   ; STORE: for j in 0..R: shift object right j*V, UConvert (truncate) to uchar, store at slot sum+j
/// ```
///
/// Byte-EXACT on a little-endian target: `k` strides `R` narrow slots and `j` walks the `R` slots of one
/// wide element, so `sum+j` is the exact byte the reinterpret-GEP+load/store would have touched, and the
/// LE assemble/split reproduces the same bytes the native wide access does. `%inner` is left intact (it
/// may have other valid uses); only `%out` is re-rooted and its loads/stores expanded.
///
/// Byte-safe / floor-safe by construction — only chains CURRENTLY INVALID are touched, gated on:
/// (1) `%out` is a single-index chain `[%inner, %k]` whose result pointee P is a DIRECT INT scalar;
/// (2) its base `%inner` is a two-index chain `[%uint_0, %byteIdx]` into a SINGLE-member struct
///     `{ array/runtime-array<E> }` whose element E is a DIRECT INT scalar with `width(P) = R*width(E)`,
///     `R >= 2` (a genuine widen — the narrow direction is the existing same-width/narrow passes), and
///     whose result pointee equals E;
/// (3) the base storage class is StorageBuffer (member-access descent over a real SSBO);
/// (4) `%out` is CURRENTLY INVALID (the index walk over `%inner`'s scalar pointee fails — always true for
///     a scalar, so a banked/valid module never matches and the floor is provably untouched);
/// (5) EVERY use of `%out` is an `OpLoad`(P) or `OpStore`(_, P) — any other use skips the chain.
/// Decides purely from IR structure (storage class + type walk + use kinds), never a shader name.
pub(in crate::passes) fn rewrite_byte_buffer_chained_reinterpret(
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

    let is_int = |ctx: &Ctx, ty: Word| -> bool {
        type_def_of(ctx, ty)
            .map(|d| d.class.opcode == Op::TypeInt)
            .unwrap_or(false)
    };

    // The defining chain instruction of every access-chain result (for resolving `%inner`).
    let mut chain_def: HashMap<Word, Instruction> = HashMap::new();
    for block in &ctx.module.functions[entry_idx].blocks {
        for inst in &block.instructions {
            if matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain) {
                if let Some(rid) = inst.result_id {
                    chain_def.insert(rid, inst.clone());
                }
            }
        }
    }

    struct Plan {
        bi: usize,
        ii: usize,
        ac_id: Word,
        buf: Word,
        inner_idx: Word,
        elem_ty: Word,
        result_pointee: Word,
        ratio: u32,
        slot_w: u32,
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
            // `%out = AC %inner %k` — base + exactly one index.
            if inst.operands.len() != 2 {
                continue;
            }
            let (Operand::IdRef(inner), Operand::IdRef(_out_idx)) =
                (&inst.operands[0], &inst.operands[1])
            else {
                continue;
            };
            let Some(&(sc, result_pointee)) = ptr_info.get(&result_type) else {
                continue;
            };
            if sc != StorageClass::StorageBuffer {
                continue;
            }
            let Some(w) = direct_scalar_width(ctx, result_pointee) else {
                continue;
            };
            if !is_int(ctx, result_pointee) {
                continue;
            }
            // `%inner` must itself be a `[base, %uint_0, %byteIdx]` chain into a single-member byte array.
            let Some(inner_inst) = chain_def.get(inner) else {
                continue;
            };
            if inner_inst.operands.len() != 3 {
                continue;
            }
            let (Operand::IdRef(buf), Operand::IdRef(i0), Operand::IdRef(inner_idx)) = (
                &inner_inst.operands[0],
                &inner_inst.operands[1],
                &inner_inst.operands[2],
            ) else {
                continue;
            };
            if const_u32(ctx, *i0) != Some(0) {
                continue;
            }
            let Some(inner_result_type) = inner_inst.result_type else {
                continue;
            };
            let Some(&(_, inner_pointee)) = ptr_info.get(&inner_result_type) else {
                continue;
            };
            let Some(buf_ptr_ty) = value_result_type(ctx, *buf) else {
                continue;
            };
            let Some(&(_, buf_pointee)) = ptr_info.get(&buf_ptr_ty) else {
                continue;
            };
            let Some(elem_ty) = single_member_array_scalar_elem(ctx, buf_pointee) else {
                continue;
            };
            if inner_pointee != elem_ty || !is_int(ctx, elem_ty) {
                continue;
            }
            let Some(v) = direct_scalar_width(ctx, elem_ty) else {
                continue;
            };
            // Genuine WIDEN only: P is R whole narrow slots, R >= 2 (same-width/narrow are other passes).
            if v == 0 || w % v != 0 {
                continue;
            }
            let ratio = w / v;
            if ratio < 2 {
                continue;
            }
            // `%out` must be CURRENTLY INVALID: walking `[%k]` into `%inner`'s scalar pointee must fail.
            if walk_into_type(ctx, inner_pointee, &inst.operands[1..]).is_some() {
                continue;
            }
            plans.push(Plan {
                bi,
                ii,
                ac_id,
                buf: *buf,
                inner_idx: *inner_idx,
                elem_ty,
                result_pointee,
                ratio,
                slot_w: v,
            });
        }
    }
    if plans.is_empty() {
        return Ok(());
    }

    // Every use of each `%out` must be an OpLoad(P)/OpStore(_, P); disqualify otherwise.
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

    // Phase 2: the byte-index `sum = inner_idx + out_idx*ratio` per plan, shared by every use.
    let member0 = ctx.const_uint(0);
    let uint_ty = ctx.ty_uint();
    let mut elem_ptr_ty: HashMap<Word, Word> = HashMap::new();
    let mut sum_id: HashMap<Word, Word> = HashMap::new();
    let mut ratio_const: HashMap<u32, Word> = HashMap::new();
    for p in &plans {
        elem_ptr_ty
            .entry(p.elem_ty)
            .or_insert_with(|| ctx.ty_ptr(StorageClass::StorageBuffer, p.elem_ty));
        sum_id.insert(p.ac_id, ctx.module.fresh_id());
        ratio_const
            .entry(p.ratio)
            .or_insert_with(|| ctx.const_uint(p.ratio));
    }

    let ac_at: HashMap<(usize, usize), Word> =
        plans.iter().map(|p| ((p.bi, p.ii), p.ac_id)).collect();
    let plan_by_id: HashMap<Word, (Word, Word, Word, Word, u32, u32)> = plans
        .iter()
        .map(|p| {
            (
                p.ac_id,
                (
                    p.buf,
                    p.elem_ty,
                    p.result_pointee,
                    p.inner_idx,
                    p.ratio,
                    p.slot_w,
                ),
            )
        })
        .collect();
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
            // The original `%out` chain becomes the `sum = inner_idx + out_idx*ratio` computation.
            if let Some(&ac_id) = ac_at.get(&(bi, ii)) {
                let (_buf, _e, _p, inner_idx, ratio, _v) = plan_by_id[&ac_id];
                let sum = sum_id[&ac_id];
                let out_idx = match inst.operands.get(1) {
                    Some(Operand::IdRef(o)) => *o,
                    _ => {
                        return Err(
                            "byte-buffer reinterpret chain lost its index operand".to_string()
                        )
                    }
                };
                let mul = ctx.module.fresh_id();
                newv.push(Instruction::new(
                    Op::IMul,
                    Some(uint_ty),
                    Some(mul),
                    vec![Operand::IdRef(out_idx), Operand::IdRef(ratio_const[&ratio])],
                ));
                newv.push(Instruction::new(
                    Op::IAdd,
                    Some(uint_ty),
                    Some(sum),
                    vec![Operand::IdRef(inner_idx), Operand::IdRef(mul)],
                ));
                continue;
            }
            // A load/store through `%out` expands into `ratio` little-endian narrow slot accesses.
            if let Some(&ac_id) = use_at.get(&(bi, ii)) {
                let (buf, elem_ty, result_pointee, _ii, ratio, slot_w) = plan_by_id[&ac_id];
                let sum = sum_id[&ac_id];
                let eptr = elem_ptr_ty[&elem_ty];
                // Per-slot element pointers: `AC %buf %uint_0 (sum + j)`.
                let mut slot_ptr = Vec::with_capacity(ratio as usize);
                for j in 0..ratio {
                    let idx = if j == 0 {
                        sum
                    } else {
                        let off = ctx.const_uint(j);
                        let id = ctx.module.fresh_id();
                        newv.push(Instruction::new(
                            Op::IAdd,
                            Some(uint_ty),
                            Some(id),
                            vec![Operand::IdRef(sum), Operand::IdRef(off)],
                        ));
                        id
                    };
                    let pid = ctx.module.fresh_id();
                    newv.push(Instruction::new(
                        Op::InBoundsAccessChain,
                        Some(eptr),
                        Some(pid),
                        vec![
                            Operand::IdRef(buf),
                            Operand::IdRef(member0),
                            Operand::IdRef(idx),
                        ],
                    ));
                    slot_ptr.push(pid);
                }
                if inst.class.opcode == Op::Load {
                    // LE assemble: OR_j ( zext(load slot j) << (j*slot_w) ).
                    let res = inst.result_id.ok_or("load has a result id")?;
                    let mut acc: Option<Word> = None;
                    for (j, &pid) in slot_ptr.iter().enumerate() {
                        let raw = ctx.module.fresh_id();
                        newv.push(Instruction::new(
                            Op::Load,
                            Some(elem_ty),
                            Some(raw),
                            vec![Operand::IdRef(pid)],
                        ));
                        let wide = ctx.module.fresh_id();
                        newv.push(Instruction::new(
                            Op::UConvert,
                            Some(result_pointee),
                            Some(wide),
                            vec![Operand::IdRef(raw)],
                        ));
                        let shifted = if j == 0 {
                            wide
                        } else {
                            let sh = ctx.const_uint(j as u32 * slot_w);
                            let sid = ctx.module.fresh_id();
                            newv.push(Instruction::new(
                                Op::ShiftLeftLogical,
                                Some(result_pointee),
                                Some(sid),
                                vec![Operand::IdRef(wide), Operand::IdRef(sh)],
                            ));
                            sid
                        };
                        acc = Some(match acc {
                            None => shifted,
                            Some(prev) => {
                                let last = j as u32 == ratio - 1;
                                let oid = if last { res } else { ctx.module.fresh_id() };
                                newv.push(Instruction::new(
                                    Op::BitwiseOr,
                                    Some(result_pointee),
                                    Some(oid),
                                    vec![Operand::IdRef(prev), Operand::IdRef(shifted)],
                                ));
                                oid
                            }
                        });
                    }
                    // ratio >= 2, so the OR above always produced `res`; nothing else to emit.
                    let _ = acc;
                } else {
                    // LE split-store: store_j( trunc(obj >> (j*slot_w)) ).
                    let obj = match inst.operands.get(1) {
                        Some(Operand::IdRef(o)) => *o,
                        _ => {
                            return Err(
                                "store through reinterpret chain lost its object".to_string()
                            )
                        }
                    };
                    for (j, &pid) in slot_ptr.iter().enumerate() {
                        let shifted = if j == 0 {
                            obj
                        } else {
                            let sh = ctx.const_uint(j as u32 * slot_w);
                            let sid = ctx.module.fresh_id();
                            newv.push(Instruction::new(
                                Op::ShiftRightLogical,
                                Some(result_pointee),
                                Some(sid),
                                vec![Operand::IdRef(obj), Operand::IdRef(sh)],
                            ));
                            sid
                        };
                        let narrow = ctx.module.fresh_id();
                        newv.push(Instruction::new(
                            Op::UConvert,
                            Some(elem_ty),
                            Some(narrow),
                            vec![Operand::IdRef(shifted)],
                        ));
                        newv.push(Instruction::new(
                            Op::Store,
                            None,
                            None,
                            vec![Operand::IdRef(pid), Operand::IdRef(narrow)],
                        ));
                    }
                }
                continue;
            }
            newv.push(inst);
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = newv;
    }
    Ok(())
}
