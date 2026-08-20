//! SPIR-V rewrite entry points: each applies one legalization/portability rewrite from a sibling
//! pass module (`psb`, `phi_index`, `cfg`, `relooper`, `constfold`, …) to an in-flight [`Module`].
//! Callers (the failure-triggered retry tiers in `lib.rs`) adopt a result only if it independently
//! validates, so every rewrite here is floor-safe by construction. Also holds the remaining
//! structural screens (`has_*`); public byte compatibility wrappers live at the `native` facade.

use super::*;

/// Whether `spv` contains the local selection emitted to guard a raw logical-buffer write whose
/// source byte offset has a dynamic term wider than the u32 address model.  The guard's true arm
/// performs the original write and its false arm falls through to the selection merge, preserving
/// Metal's robust no-write behavior when the complete i64 offset cannot be represented.
///
/// This deliberately identifies the emitted control-flow *shape*, not a source symbol or private capture
/// case.  A guard inserted into a source loop header moves that header's source terminator into the
/// guard continuation, which can expose a CFG-only validator error.  Callers can then reuse the
/// ordinary relooper on precisely that already-guarded graph; the resulting bytes are the same
/// candidate production's first CFG retry would otherwise adopt.
pub(crate) fn module_has_wide_raw_store_guard(module: &Module) -> bool {
    let defs: HashMap<Word, &Instruction> = module
        .all_inst_iter()
        .filter_map(|inst| inst.result_id.map(|id| (id, inst)))
        .collect();

    let is_u64_constant = |id: Word| {
        let Some(inst) = defs.get(&id) else {
            return false;
        };
        if inst.class.opcode != Op::Constant {
            return false;
        }
        let Some(ty) = inst.result_type else {
            return false;
        };
        let Some(type_inst) = defs.get(&ty) else {
            return false;
        };
        type_inst.class.opcode == Op::TypeInt
            && type_inst.operands.first() == Some(&Operand::LiteralBit32(64))
            && type_inst.operands.get(1) == Some(&Operand::LiteralBit32(0))
    };

    for function in &module.functions {
        let blocks: HashMap<Word, &crate::spirv_module::Block> = function
            .blocks
            .iter()
            .filter_map(|block| {
                block
                    .label
                    .as_ref()
                    .and_then(|label| label.result_id)
                    .map(|id| (id, block))
            })
            .collect();
        for block in &function.blocks {
            for (index, branch) in block.instructions.iter().enumerate() {
                if branch.class.opcode != Op::BranchConditional || index == 0 {
                    continue;
                }
                let (
                    Some(Operand::IdRef(condition)),
                    Some(Operand::IdRef(write_label)),
                    Some(Operand::IdRef(false_label)),
                ) = (
                    branch.operands.first(),
                    branch.operands.get(1),
                    branch.operands.get(2),
                )
                else {
                    continue;
                };
                let merge = &block.instructions[index - 1];
                if merge.class.opcode != Op::SelectionMerge {
                    continue;
                }
                let Some(Operand::IdRef(merge_label)) = merge.operands.first() else {
                    continue;
                };
                if false_label != merge_label {
                    continue;
                }
                let Some(compare) = defs.get(condition) else {
                    continue;
                };
                if compare.class.opcode != Op::ULessThanEqual
                    || !matches!(compare.operands.get(1), Some(Operand::IdRef(max)) if is_u64_constant(*max))
                {
                    continue;
                }
                let Some(write_block) = blocks.get(write_label) else {
                    continue;
                };
                let writes_then_merges = write_block
                    .instructions
                    .iter()
                    .any(|inst| is_spirv_memory_write(inst.class.opcode))
                    && write_block.instructions.last().is_some_and(|terminator| {
                        terminator.class.opcode == Op::Branch
                            && terminator.operands.first() == Some(&Operand::IdRef(*merge_label))
                    });
                if writes_then_merges {
                    return true;
                }
            }
        }
    }
    false
}

/// `OpStore` plus the atomic operations that can modify their first pointer operand.  The native
/// raw subword-store lowering is an atomic AND/OR read-modify-write, so the robust wide-offset
/// guard must recognize it as a write too.
fn is_spirv_memory_write(op: Op) -> bool {
    matches!(
        op,
        Op::Store
            | Op::AtomicStore
            | Op::AtomicExchange
            | Op::AtomicCompareExchange
            | Op::AtomicCompareExchangeWeak
            | Op::AtomicIIncrement
            | Op::AtomicIDecrement
            | Op::AtomicIAdd
            | Op::AtomicISub
            | Op::AtomicSMin
            | Op::AtomicUMin
            | Op::AtomicSMax
            | Op::AtomicUMax
            | Op::AtomicAnd
            | Op::AtomicOr
            | Op::AtomicXor
            | Op::AtomicFAddEXT
            | Op::AtomicFMinEXT
            | Op::AtomicFMaxEXT
    )
}

/// Drop byte-addressed Workgroup padding clears that the native emitter can produce from AIR
/// `llvm.memset.p3` over struct tail padding.
///
/// SPIR-V Logical addressing cannot express `OpPtrAccessChain %uchar*` from a `%struct*` base. When
/// the byte offset does not match any `Offset`-decorated struct member and every use is a zero
/// `OpStore`, the store only clears padding. Workgroup memory has already been zero-filled by the
/// harness prologue, and no typed load can observe the padding byte, so removing the invalid chain
/// and its zero stores is value-preserving.
pub(crate) fn drop_workgroup_struct_padding_byte_zero_stores_module(module: &mut Module) -> bool {
    use spirv::Decoration;

    let mut type_defs: HashMap<Word, &Instruction> = HashMap::new();
    let mut ptr_info: HashMap<Word, (StorageClass, Word)> = HashMap::new();
    let mut result_type: HashMap<Word, Word> = HashMap::new();
    let mut constants: HashMap<Word, u64> = HashMap::new();
    let mut member_offsets: HashMap<Word, HashMap<u32, u32>> = HashMap::new();

    for inst in module.all_inst_iter() {
        if let Some(id) = inst.result_id {
            if let Some(ty) = inst.result_type {
                result_type.insert(id, ty);
            }
            match inst.class.opcode {
                Op::TypeInt | Op::TypePointer | Op::TypeStruct => {
                    type_defs.insert(id, inst);
                }
                Op::Constant => {
                    let value = match inst.operands.first() {
                        Some(Operand::LiteralBit32(value)) => u64::from(*value),
                        Some(Operand::LiteralBit64(value)) => *value,
                        _ => continue,
                    };
                    constants.insert(id, value);
                }
                Op::ConstantNull => {
                    constants.insert(id, 0);
                }
                _ => {}
            }
        }
        if inst.class.opcode == Op::TypePointer {
            if let (Some(id), Some(Operand::StorageClass(sc)), Some(Operand::IdRef(pointee))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*sc, *pointee));
            }
        }
        if inst.class.opcode == Op::MemberDecorate {
            let (
                Some(Operand::IdRef(struct_ty)),
                Some(Operand::LiteralBit32(member)),
                Some(Operand::Decoration(Decoration::Offset)),
                Some(Operand::LiteralBit32(offset)),
            ) = (
                inst.operands.first(),
                inst.operands.get(1),
                inst.operands.get(2),
                inst.operands.get(3),
            )
            else {
                continue;
            };
            member_offsets
                .entry(*struct_ty)
                .or_default()
                .insert(*member, *offset);
        }
    }

    let is_uchar = |ty: Word| -> bool {
        matches!(
            type_defs.get(&ty).map(|inst| inst.operands.as_slice()),
            Some([Operand::LiteralBit32(8), Operand::LiteralBit32(0)])
        )
    };
    fn align_to(value: u32, align: u32) -> u32 {
        if align <= 1 {
            value
        } else {
            value.div_ceil(align) * align
        }
    }
    fn type_size_align(
        type_defs: &HashMap<Word, &Instruction>,
        constants: &HashMap<Word, u64>,
        ty: Word,
    ) -> Option<(u32, u32)> {
        let def = type_defs.get(&ty)?;
        match def.class.opcode {
            Op::TypeInt | Op::TypeFloat => {
                let Some(Operand::LiteralBit32(width)) = def.operands.first() else {
                    return None;
                };
                let bytes = width.div_ceil(8).max(1);
                Some((bytes, bytes))
            }
            Op::TypeVector => {
                let (Some(Operand::IdRef(component)), Some(Operand::LiteralBit32(count))) =
                    (def.operands.first(), def.operands.get(1))
                else {
                    return None;
                };
                let (size, align) = type_size_align(type_defs, constants, *component)?;
                Some((size.checked_mul(*count)?, align))
            }
            Op::TypeArray => {
                let (Some(Operand::IdRef(element)), Some(Operand::IdRef(length))) =
                    (def.operands.first(), def.operands.get(1))
                else {
                    return None;
                };
                let (size, align) = type_size_align(type_defs, constants, *element)?;
                let count = u32::try_from(*constants.get(length)?).ok()?;
                Some((align_to(size, align).checked_mul(count)?, align))
            }
            Op::TypeStruct => {
                let mut offset = 0u32;
                let mut max_align = 1u32;
                for operand in &def.operands {
                    let Operand::IdRef(member_ty) = operand else {
                        return None;
                    };
                    let (member_size, member_align) =
                        type_size_align(type_defs, constants, *member_ty)?;
                    max_align = max_align.max(member_align);
                    offset = align_to(offset, member_align);
                    offset = offset.checked_add(member_size)?;
                }
                Some((align_to(offset, max_align), max_align))
            }
            _ => None,
        }
    }
    let inferred_struct_offsets = |ty: Word| -> Option<HashSet<u32>> {
        let def = type_defs.get(&ty)?;
        if def.class.opcode != Op::TypeStruct {
            return None;
        }
        let mut offsets = HashSet::new();
        let mut offset = 0u32;
        for operand in &def.operands {
            let Operand::IdRef(member_ty) = operand else {
                return None;
            };
            let (member_size, member_align) = type_size_align(&type_defs, &constants, *member_ty)?;
            offset = align_to(offset, member_align);
            offsets.insert(offset);
            offset = offset.checked_add(member_size)?;
        }
        Some(offsets)
    };
    let is_struct_padding_offset = |ty: Word, offset: u32| -> bool {
        let Some(def) = type_defs.get(&ty) else {
            return false;
        };
        if def.class.opcode != Op::TypeStruct {
            return false;
        }
        if let Some(offsets) = member_offsets.get(&ty) {
            return !offsets
                .values()
                .any(|member_offset| *member_offset == offset);
        };
        inferred_struct_offsets(ty).is_some_and(|offsets| !offsets.contains(&offset))
    };

    let mut chain_base: HashMap<Word, (usize, usize, usize, Word, u32)> = HashMap::new();
    for (fi, function) in module.functions.iter().enumerate() {
        for (bi, block) in function.blocks.iter().enumerate() {
            for (ii, inst) in block.instructions.iter().enumerate() {
                if inst.class.opcode != Op::PtrAccessChain || inst.operands.len() != 2 {
                    continue;
                }
                let (Some(chain), Some(result_ptr_ty)) = (inst.result_id, inst.result_type) else {
                    continue;
                };
                let Some(&(StorageClass::Workgroup, result_pointee)) = ptr_info.get(&result_ptr_ty)
                else {
                    continue;
                };
                if !is_uchar(result_pointee) {
                    continue;
                }
                let (Some(Operand::IdRef(base)), Some(Operand::IdRef(offset_id))) =
                    (inst.operands.first(), inst.operands.get(1))
                else {
                    continue;
                };
                let Some(offset) = constants
                    .get(offset_id)
                    .and_then(|value| u32::try_from(*value).ok())
                else {
                    continue;
                };
                let Some(base_ptr_ty) = result_type.get(base).copied() else {
                    continue;
                };
                let Some(&(StorageClass::Workgroup, base_pointee)) = ptr_info.get(&base_ptr_ty)
                else {
                    continue;
                };
                if is_struct_padding_offset(base_pointee, offset) {
                    chain_base.insert(chain, (fi, bi, ii, base_pointee, offset));
                }
            }
        }
    }
    let mut changed_candidates = true;
    while changed_candidates {
        changed_candidates = false;
        for (fi, function) in module.functions.iter().enumerate() {
            for (bi, block) in function.blocks.iter().enumerate() {
                for (ii, inst) in block.instructions.iter().enumerate() {
                    if inst.class.opcode != Op::PtrAccessChain || inst.operands.len() != 2 {
                        continue;
                    }
                    let (Some(chain), Some(result_ptr_ty)) = (inst.result_id, inst.result_type)
                    else {
                        continue;
                    };
                    if chain_base.contains_key(&chain) {
                        continue;
                    }
                    let Some(&(StorageClass::Workgroup, result_pointee)) =
                        ptr_info.get(&result_ptr_ty)
                    else {
                        continue;
                    };
                    if !is_uchar(result_pointee) {
                        continue;
                    }
                    let (Some(Operand::IdRef(base)), Some(Operand::IdRef(offset_id))) =
                        (inst.operands.first(), inst.operands.get(1))
                    else {
                        continue;
                    };
                    let Some((_, _, _, root_struct, base_offset)) = chain_base.get(base).copied()
                    else {
                        continue;
                    };
                    let Some(offset) = constants
                        .get(offset_id)
                        .and_then(|value| u32::try_from(*value).ok())
                        .and_then(|offset| base_offset.checked_add(offset))
                    else {
                        continue;
                    };
                    if is_struct_padding_offset(root_struct, offset) {
                        chain_base.insert(chain, (fi, bi, ii, root_struct, offset));
                        changed_candidates = true;
                    }
                }
            }
        }
    }
    if chain_base.is_empty() {
        return false;
    }

    let mut zero_values: HashSet<Word> = constants
        .iter()
        .filter_map(|(id, value)| (*value == 0).then_some(*id))
        .collect();
    let mut changed_zero = true;
    while changed_zero {
        changed_zero = false;
        for inst in module.all_inst_iter() {
            let Some(id) = inst.result_id else {
                continue;
            };
            if zero_values.contains(&id) {
                continue;
            }
            let zero_operand0 = inst.operands.first().is_some_and(
                |operand| matches!(operand, Operand::IdRef(value) if zero_values.contains(value)),
            );
            let zero = match inst.class.opcode {
                Op::CopyObject | Op::UConvert | Op::SConvert | Op::Bitcast => zero_operand0,
                Op::ShiftRightLogical | Op::ShiftRightArithmetic | Op::ShiftLeftLogical => {
                    zero_operand0
                }
                _ => false,
            };
            if zero {
                zero_values.insert(id);
                changed_zero = true;
            }
        }
    }

    let mut store_sites: HashSet<(usize, usize, usize)> = HashSet::new();
    let mut disqualified: HashSet<Word> = HashSet::new();
    for (fi, function) in module.functions.iter().enumerate() {
        for (bi, block) in function.blocks.iter().enumerate() {
            for (ii, inst) in block.instructions.iter().enumerate() {
                for (oi, operand) in inst.operands.iter().enumerate() {
                    let Operand::IdRef(id) = operand else {
                        continue;
                    };
                    if !chain_base.contains_key(id) {
                        continue;
                    }
                    if inst.class.opcode == Op::PtrAccessChain
                        && oi == 0
                        && inst
                            .result_id
                            .is_some_and(|result| chain_base.contains_key(&result))
                    {
                        continue;
                    }
                    if inst.class.opcode == Op::Store
                        && oi == 0
                        && inst.operands.get(1).and_then(|operand| match operand {
                            Operand::IdRef(value) => zero_values.contains(value).then_some(0),
                            _ => None,
                        }) == Some(0)
                    {
                        store_sites.insert((fi, bi, ii));
                    } else {
                        disqualified.insert(*id);
                    }
                }
            }
        }
    }
    chain_base.retain(|chain, _| !disqualified.contains(chain));
    if chain_base.is_empty() || store_sites.is_empty() {
        return false;
    }

    let chain_sites: HashSet<(usize, usize, usize)> = chain_base
        .into_values()
        .map(|(fi, bi, ii, _, _)| (fi, bi, ii))
        .collect();
    let mut changed = false;
    for (fi, function) in module.functions.iter_mut().enumerate() {
        for (bi, block) in function.blocks.iter_mut().enumerate() {
            let old = std::mem::take(&mut block.instructions);
            block.instructions = old
                .into_iter()
                .enumerate()
                .filter_map(|(ii, inst)| {
                    if chain_sites.contains(&(fi, bi, ii)) || store_sites.contains(&(fi, bi, ii)) {
                        changed = true;
                        None
                    } else {
                        Some(inst)
                    }
                })
                .collect();
        }
    }
    changed
}

/// Remove debug/decorate records whose target id was deleted by a prior module rewrite.
///
/// Rewrites such as constant-branch pruning can delete function-constant helper globals, and a later
/// CFG rebuild may preserve their `OpName` records. `OpName` is non-semantic, but SPIR-V still
/// requires its target id to be defined. This cleanup never touches executable instructions or
/// interface declarations; it only drops annotations/debug names that already point at nothing.
pub(crate) fn drop_dangling_debug_targets_module(module: &mut Module) -> bool {
    let defined: HashSet<Word> = module
        .all_inst_iter()
        .filter_map(|inst| inst.result_id)
        .collect();
    let target_is_dangling = |inst: &Instruction| -> bool {
        matches!(inst.operands.first(), Some(Operand::IdRef(id)) if !defined.contains(id))
    };

    let before_names = module.debug_names.len();
    let before_annotations = module.annotations.len();
    module.debug_names.retain(|inst| !target_is_dangling(inst));
    module.annotations.retain(|inst| !target_is_dangling(inst));
    module.debug_names.len() != before_names || module.annotations.len() != before_annotations
}

/// Collapse every one-incoming `OpPhi` to its sole value and substitute its uses within the owning
/// function. CFG edge splitting legitimately creates this forwarding form; keeping it until
/// validation is both redundant and hazardous when an earlier interface rewrite refined the value
/// type but left the phi's stale carrier type behind. The rewrite is an SSA identity, bounded to one
/// small substitution map per function, and does not clone consumer subgraphs.
pub(crate) fn collapse_single_incoming_phis_module(module: &mut Module) -> bool {
    let mut changed = false;
    for function in &mut module.functions {
        changed |= constfold::collapse_trivial_phis(function);
    }
    if changed {
        drop_dangling_debug_targets_module(module);
    }
    changed
}

/// Replace already-invalid packed `i32` loads through a Private 16-bit vector word view with a
/// direct vector load plus a two-lane bitcast. Returns whether any load was legalized.
pub(crate) fn rewrite_private_vector_word_loads_module(module: &mut Module) -> bool {
    let changed = private_vector_word::rewrite_private_vector_word_loads(module);
    if changed {
        drop_dangling_debug_targets_module(module);
    }
    changed
}

/// Retype validator-invalid `OpSampledImage` instructions to the concrete image operand's sampled
/// type after interface refinement.
pub(crate) fn repair_sampled_image_result_types_module(module: &mut Module) -> bool {
    sampled_image_type::repair_sampled_image_result_types(module)
}

/// Apply the W1 PhysicalStorageBuffer64 lowering in place. Errors if no cross-binding pointer-merge
/// sub-graph was rewritten. The caller (the failure-triggered retry) adopts the result ONLY if it
/// independently validates, so this is floor-safe by construction.
pub(crate) fn rewrite_cross_binding_pointer_merges_module(
    module: &mut Module,
    layout: crate::reflect::DescriptorLayout,
) -> Result<(), String> {
    if !psb::rewrite_cross_binding_pointer_merges_with_layout(module, layout) {
        return Err("native emitter: no cross-binding pointer merge to rewrite".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}

/// Report whether `module` has a lowerable cross-binding pointer closure containing an `OpPhi`.
/// Callers use this cheap structural screen after the exact spirv-val diagnostic to decide whether
/// the phi could be the failure that warrants the PhysicalStorageBuffer64 primary candidate.
pub(crate) fn has_cross_binding_pointer_phi_module(module: &Module) -> bool {
    psb::has_cross_binding_pointer_phi(module)
}

/// Apply the PhysicalStorageBuffer64 lowering in place only when the cross-binding closure contains
/// an `OpPhi`. Ordinary cross-binding selects stay available to the Logical value-domain lowering;
/// a phi with post-merge dynamic accesses needs the address-table representation instead of
/// replaying values on predecessor edges. The caller still validates before adopting the module.
pub(crate) fn rewrite_cross_binding_pointer_phis_module(
    module: &mut Module,
    layout: crate::reflect::DescriptorLayout,
) -> Result<(), String> {
    if !psb::rewrite_cross_binding_pointer_phis_with_layout(module, layout) {
        return Err("native emitter: no cross-binding pointer phi to rewrite".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}

/// Lower the cross-binding pointer-merge sub-graph in place INTO THE VALUE DOMAIN (plain Logical
/// `StorageBuffer`), staying off PhysicalStorageBuffer64. Instead of selecting among POINTERS then
/// loading once, it loads from every candidate buffer and selects among the LOADED VALUES —
/// byte-exact (the selected value is the exact load Apple performs; discarded over-reads do not
/// fault), and MoltenVK-runnable (no buffer-device-address, which blocks compute-pipeline creation).
/// Errors if no cross-binding pointer merge was value-lowered. The caller (the failure-triggered
/// retry) adopts the result ONLY if it independently validates, so this is floor-safe by
/// construction; it is preferred over the PSB lowering when both validate.
pub(crate) fn rewrite_cross_binding_pointer_merges_to_values_module(
    module: &mut Module,
) -> Result<(), String> {
    if !psb_value_select::rewrite_cross_binding_pointer_merges_to_values(module) {
        return Err("native emitter: no cross-binding pointer merge to value-lower".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}

/// Lower an opaque image `OpSelect` through pure explicit-LOD sampling into ordinary value selects,
/// in place. Vulkan cannot select images directly without descriptor indexing, while cloning a pure
/// sample for each image leaf and selecting the sampled result stays in portable Logical SPIR-V. The
/// pass declines anything except an image-only select tree whose complete consumer closure is
/// explicit-LOD sampling.
pub(crate) fn rewrite_opaque_image_selects_module(module: &mut Module) -> Result<(), String> {
    if !opaque_image_select::rewrite_opaque_image_selects(module) {
        return Err("native emitter: no lowerable opaque image select".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}

/// Remodel every Workgroup variable accessed only as the float-as-int atomic idiom (the
/// `OpBitcast %_ptr_Workgroup_<int> %chain` → `OpAtomicSMin/SMax` pattern that spirv-val rejects as an
/// illegal logical-pointer bitcast) so its float leaves become the int the atomics use. Errors if no
/// variable was remodeled. Byte-safe by construction (Workgroup scratch, float↔int32 bit-identical,
/// layout-preserving clone, strict all-uses gate). The caller adopts the result ONLY if it
/// independently validates, so this is floor-safe by construction.
pub(crate) fn rewrite_workgroup_atomic_floats_module(module: &mut Module) -> Result<(), String> {
    if !wg_atomic::rewrite_workgroup_atomic_floats(module) {
        return Err("native emitter: no workgroup float-as-int atomic to remodel".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}

/// Retype every Function scalar-integer variable accessed ONLY as the sub-word packed-scalar idiom
/// (e.g. a `uint` alloca written as two `half` lanes then read whole) into a `<N x E>` vector, in
/// place. This drops the illegal scalar-indexing access chains' invalidity and value-bitcasts its
/// whole-word loads/stores. Errors if no variable was remodeled. Byte-safe by construction
/// (Function scratch, little-endian-identical vector layout). The caller adopts the result ONLY if
/// it independently validates, so this is floor-safe by construction.
pub(crate) fn rewrite_subword_packed_scalars_module(module: &mut Module) -> Result<(), String> {
    if !subword_pack::rewrite_subword_packed_scalars(module) {
        return Err("native emitter: no sub-word packed scalar to remodel".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}

/// Apply the M4 phi-the-index retry to an already-loaded module and re-finalize it in place. Errors
/// if no eligible illegal logical-pointer `OpPhi` was rewritten. The caller adopts the result ONLY
/// if it independently validates, so this is floor-safe by construction.
pub(crate) fn rewrite_logical_pointer_phis_retry_module(module: &mut Module) -> Result<(), String> {
    if !phi_index::rewrite_logical_pointer_phis(module) {
        return Err("native emitter: no logical pointer phi to rewrite".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}

/// Apply the M4 phi-the-index legalization to an in-flight module (the PRIMARY emit tail), in place.
/// Rewrites only ILLEGAL logical-pointer phis — a non-`StorageBuffer`/`Workgroup` (Private/Function/
/// UniformConstant) pointer `OpPhi`, which is always spirv-val-invalid — so it can only move an
/// already-invalid module toward valid, never touch a validating one (floor-safe by construction).
/// Same mechanism the retry tier ([`rewrite_logical_pointer_phis_retry_module`]) applies, hoisted
/// onto the no-retry path so these functions' PRIMARY emit validates instead of shipping via retry
/// rescue. Returns true if any phi was rewritten. The caller runs `canonicalize_ids` afterward.
pub(crate) fn rewrite_logical_pointer_phis_module(module: &mut Module) -> bool {
    phi_index::rewrite_logical_pointer_phis(module)
}

/// Lower an invalid direct-load `OpSelect` whose opaque-pointer arms cross logical storage classes
/// into per-arm loads followed by a value select. The structural pass refuses pointer escapes.
pub(crate) fn rewrite_mixed_storage_pointer_select_loads_module(module: &mut Module) -> bool {
    mixed_pointer_select::rewrite_mixed_storage_pointer_select_loads(module)
}

/// Legalize integer `OpPhi` result/incoming width mismatches in an in-flight module (the PRIMARY emit
/// tail) by truncating a wide incoming to the phi's narrower integer result type. Only touches phis
/// that are already spirv-val-INVALID (an integer phi whose operand type differs from its result
/// type), so it can only move an already-invalid module toward valid — floor-safe by construction.
/// See [`phi_index::rewrite_integer_width_phis`] for the mechanism. The caller runs `canonicalize_ids`
/// afterward. Returns true if any operand was converted.
pub(crate) fn rewrite_integer_width_phis_module(module: &mut Module) -> bool {
    phi_index::rewrite_integer_width_phis(module)
}

/// Register-demote any value in an in-flight module (the PRIMARY emit tail) whose defining block no
/// longer dominates a use — the loop-closed-SSA violation the `MultipleExits` funnel
/// (`synth_multi_exit_merge`) leaves behind (spirv-val: *"ID X defined in block B does not dominate its
/// use in block U"*). Spills the value to a function-scope `OpVariable` (stored after its def, loaded
/// before each non-dominated use). Only touches modules that ALREADY carry such a violation (a valid
/// module has every def dominating its uses), so it is floor-safe by construction. See
/// [`cfg::demote_nondominating_values`]. The caller runs `canonicalize_ids` afterward. Returns true if
/// any value was demoted.
pub(crate) fn demote_nondominating_values_module(module: &mut Module) -> bool {
    cfg::demote_nondominating_values(module)
}

/// Node-split a MULTI-ENTRY loop in an in-flight module (the PRIMARY emit tail) whose header is entered
/// by forward edges from two different selections' arms — the irreducible shape `structured_plan`
/// over-admits, spirv-val-INVALID (*"block X exits the selection headed by Y, but not via a structured
/// exit"*; the mlx-steel `steel_attention` family). Duplicates the loop region for the inner arm's
/// entry, routing the clone's exit to that selection's merge, so each loop is single-entry. Only fires
/// on a loop with ≥2 forward header entries (a valid loop is single-entry), so it is floor-safe by
/// construction. See [`cfg::split_multientry_loop_selection_exits`]. The caller runs `canonicalize_ids`
/// afterward. Returns true if any loop was split.
pub(crate) fn split_multientry_loop_selection_exits_module(module: &mut Module) -> bool {
    cfg::split_multientry_loop_selection_exits(module)
}

/// Lower a cross-binding pointer `OpSelect`/`OpPhi` (pointers into DISTINCT buffer bindings, spirv-val-
/// INVALID *"Variable pointers must point into the same structure"*) INTO THE VALUE DOMAIN in an
/// in-flight module (the PRIMARY emit tail): load from every candidate buffer, select among the LOADED
/// VALUES. This is the SAME portable value-domain form the `value_select` retry tier ships
/// (`rewrite_cross_binding_pointer_merges_to_values_module`); running it on the primary path makes the
/// direct emit valid instead of relying on retry-rescue. The caller runs `canonicalize_ids` afterward.
/// Returns true if a merge was value-lowered.
///
/// GUARDED to `StorageBuffer`-pointer merges — the genuine "Variable pointers must point into the same
/// structure" class the `value_select` retry rescues. A merge over LOGICAL (`Private`/`Function`/
/// `Workgroup`) pointers is a DIFFERENT population: the unmodeled-device-buffer placeholder family
/// (`emit_private_zero_pointer_value`, e.g. `01/a70fb990` `%_ptr_Private_float %86` dynamically indexed
/// → the "reached non-composite" error), which the PRIMARY error routes to `fc_promote_psb` /
/// pointer-typing retry, NOT `value_select`. Value-lowering such a Private merge does NOT resolve the
/// module (the non-composite issue remains) and DERAILS that pointer-typing rescue (`a70fb990` regressed
/// valid→FALLBACK), so it is excluded — mirroring production, where `value_select` never fires for a
/// non-`CrossBindingPointerMerge` error class. Decides purely from IR pointer storage class, never a
/// shader name.
pub(crate) fn value_lower_cross_binding_pointer_merges_module(module: &mut Module) -> bool {
    let defs = module
        .types_global_values
        .iter()
        .filter_map(|inst| inst.result_id.map(|id| (id, inst.clone())))
        .collect::<HashMap<_, _>>();
    let has_storage_buffer_pointer_merge = module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| matches!(inst.class.opcode, Op::Phi | Op::Select))
        .filter_map(|inst| inst.result_type)
        .any(|ty| ptr_storage(&defs, ty) == Some(StorageClass::StorageBuffer));
    if !has_storage_buffer_pointer_merge {
        return false;
    }
    psb_value_select::rewrite_cross_binding_pointer_merges_to_values(module)
}

/// Apply the phi-the-index rewrite in place to VARIABLE-pointer (`StorageBuffer`/`Workgroup`) phis.
/// These phis are legal SPIR-V under
/// `VariablePointersStorageBuffer` — spirv-val passes — but MoltenVK's SPIRV-Cross MSL backend
/// cannot always express them (pipeline creation fails with `cannot initialize a variable of type
/// 'device float *' with an lvalue of type 'device float'`). The index-phi form is semantically
/// identical (same base, same per-arm indices, one rematerialized access chain), so the caller runs
/// this as a PORTABILITY NORMALIZATION on the success path and adopts the result only if it
/// independently validates. Errors if no eligible phi was rewritten.
pub(crate) fn rewrite_variable_pointer_phis_module(module: &mut Module) -> Result<(), String> {
    if !phi_index::rewrite_variable_pointer_phis(module) {
        return Err("native emitter: no variable pointer phi to rewrite".to_string());
    }
    add_native_module_capabilities(module);
    crate::passes::lower_scalar_i64_arithmetic_module(module);
    // The i64 lowering can widen a synthesized index-phi backedge after phi-the-index has already
    // normalized it. Re-run the narrow-incoming legalization on that final shape before validation.
    phi_index::rewrite_integer_width_phis(module);
    add_native_module_capabilities(module);
    Ok(())
}

/// Apply static constant-branch pruning (function-constant dead-arm DCE) in place. Errors if
/// nothing was pruned. The caller (the failure-triggered retry) adopts the result ONLY if it
/// independently validates, and the transformation removes only statically-unreachable code +
/// unused pure values, so it is floor-safe AND semantics-preserving by construction.
pub(crate) fn prune_constant_branches_module(module: &mut Module) -> Result<(), String> {
    if !constfold::prune_constant_branches(module) {
        return Err("native emitter: no constant branch to prune".to_string());
    }
    drop_dangling_debug_targets_module(module);
    add_native_module_capabilities(module);
    Ok(())
}

/// Remove dead pure chains rooted at invalid logical-pointer nulls after late primary rewrites.
/// Preserve every result outside those chains explicitly, so this remains a focused legalization and
/// cannot become whole-module DCE or perturb unrelated diagnostic instructions.
pub(crate) fn drop_unused_values_module(module: &mut Module) -> bool {
    let defs = module
        .types_global_values
        .iter()
        .filter_map(|instruction| instruction.result_id.map(|id| (id, instruction.clone())))
        .collect::<HashMap<_, _>>();
    let mut removable = module
        .types_global_values
        .iter()
        .filter(|instruction| instruction.class.opcode == Op::ConstantNull)
        .filter_map(|instruction| {
            let id = instruction.result_id?;
            let storage = ptr_storage(&defs, instruction.result_type?)?;
            (!matches!(
                storage,
                StorageClass::StorageBuffer | StorageClass::Workgroup
            ))
            .then_some(id)
        })
        .collect::<HashSet<_>>();
    if removable.is_empty() {
        return false;
    }

    loop {
        let additions = module
            .all_inst_iter()
            .filter(|instruction| {
                constfold::is_pure(instruction.class.opcode) && instruction.result_id.is_some()
            })
            .filter(|instruction| {
                instruction
                    .operands
                    .iter()
                    .any(|operand| matches!(operand, Operand::IdRef(id) if removable.contains(id)))
            })
            .filter_map(|instruction| instruction.result_id)
            .filter(|id| !removable.contains(id))
            .collect::<Vec<_>>();
        if additions.is_empty() {
            break;
        }
        removable.extend(additions);
    }

    let preserved = module
        .all_inst_iter()
        .filter_map(|instruction| instruction.result_id)
        .filter(|id| !removable.contains(id))
        .collect::<HashSet<_>>();
    constfold::dce_preserving(module, &preserved)
}

/// Reconcile the storage class of every typed access-chain result with its actual base pointer.
///
/// SPIR-V requires these storage classes to match. Earlier lowering normally establishes that
/// invariant, but a late value substitution can replace a pointer merge with one of its concrete
/// roots after the access chain was typed. In particular, folding an AIR function-constant switch
/// can replace an `addrspace(2)` pointer carrier with a module-scope `Private` constant table while
/// leaving the chain's former `UniformConstant` result type behind. Retyping only the pointer storage
/// class is exact: the pointee and every index remain unchanged, and the base determines the only
/// legal storage class.
///
/// Run to a fixpoint because one repaired chain can itself be the base of a later chain. Returns
/// whether any result type changed.
pub(crate) fn reconcile_access_chain_storage_classes_module(module: &mut Module) -> bool {
    let mut changed = false;
    loop {
        let pointer_types = module
            .types_global_values
            .iter()
            .filter_map(|instruction| {
                if instruction.class.opcode != Op::TypePointer {
                    return None;
                }
                let id = instruction.result_id?;
                let (Some(Operand::StorageClass(storage)), Some(Operand::IdRef(pointee))) =
                    (instruction.operands.first(), instruction.operands.get(1))
                else {
                    return None;
                };
                Some((id, (*storage, *pointee)))
            })
            .collect::<HashMap<_, _>>();
        let value_types = module
            .all_inst_iter()
            .filter_map(|instruction| Some((instruction.result_id?, instruction.result_type?)))
            .collect::<HashMap<_, _>>();

        let repairs = module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .filter(|instruction| {
                matches!(
                    instruction.class.opcode,
                    Op::AccessChain
                        | Op::InBoundsAccessChain
                        | Op::PtrAccessChain
                        | Op::InBoundsPtrAccessChain
                )
            })
            .filter_map(|instruction| {
                let result = instruction.result_id?;
                let result_type = instruction.result_type?;
                let Operand::IdRef(base) = instruction.operands.first()? else {
                    return None;
                };
                let base_type = value_types.get(base)?;
                let (base_storage, _) = pointer_types.get(base_type)?;
                let (result_storage, pointee) = pointer_types.get(&result_type)?;
                (base_storage != result_storage).then_some((result, (*base_storage, *pointee)))
            })
            .collect::<Vec<_>>();
        if repairs.is_empty() {
            break;
        }

        let mut replacement_types = HashMap::new();
        for (_, key) in &repairs {
            if replacement_types.contains_key(key) {
                continue;
            }
            if let Some(existing) = pointer_types
                .iter()
                .find_map(|(id, candidate)| (candidate == key).then_some(*id))
            {
                replacement_types.insert(*key, existing);
                continue;
            }
            let id = module.fresh_id();
            module.types_global_values.push(Instruction::new(
                Op::TypePointer,
                None,
                Some(id),
                vec![Operand::StorageClass(key.0), Operand::IdRef(key.1)],
            ));
            replacement_types.insert(*key, id);
        }
        let repairs = repairs.into_iter().collect::<HashMap<_, _>>();
        for function in &mut module.functions {
            for block in &mut function.blocks {
                for instruction in &mut block.instructions {
                    let Some(result) = instruction.result_id else {
                        continue;
                    };
                    let Some(key) = repairs.get(&result) else {
                        continue;
                    };
                    instruction.result_type = replacement_types.get(key).copied();
                    changed = true;
                }
            }
        }
    }
    changed
}

/// Preserving form of [`prune_constant_branches_module`] for a primary module that still carries
/// typed sidecar roots. Returns whether pruning changed the module.
pub(crate) fn prune_constant_branches_module_preserving(
    module: &mut Module,
    preserved_global_ids: &[Word],
) -> bool {
    let roots = preserved_global_ids.iter().copied().collect();
    let changed = constfold::prune_constant_branches_preserving(module, &roots);
    let dropped = drop_dangling_debug_targets_module(module);
    changed || dropped
}

/// Whether the module contains an `OpFunctionCall` to a BODILESS `llvm.agx*` hardware-intrinsic
/// declaration (AGX matmul `igemm`, `load/store.with.emask`, …) — the structural trigger for the
/// primary-path FC prune in `primary_retry.rs`. Such a call is never executable on a Vulkan target:
/// no lowering exists and the declaration has no body. Keyed on the `llvm.agx` ABI-symbol namespace
/// via the `OpName` of bodiless functions (the emitter always names emitted declarations) — never a
/// shader name.
pub(crate) fn has_bodiless_agx_call_module(module: &Module) -> bool {
    use std::collections::HashSet;
    let agx_names: HashSet<spirv::Word> = module
        .debug_names
        .iter()
        .filter(|inst| inst.class.opcode == spirv::Op::Name)
        .filter_map(|inst| {
            let Operand::IdRef(id) = inst.operands.first()? else {
                return None;
            };
            let Operand::LiteralString(name) = inst.operands.get(1)? else {
                return None;
            };
            name.starts_with("llvm.agx").then_some(*id)
        })
        .collect();
    if agx_names.is_empty() {
        return false;
    }
    let bodiless: HashSet<spirv::Word> = module
        .functions
        .iter()
        .filter(|f| f.blocks.is_empty())
        .filter_map(|f| f.def.as_ref().and_then(|d| d.result_id))
        .filter(|id| agx_names.contains(id))
        .collect();
    if bodiless.is_empty() {
        return false;
    }
    module.functions.iter().any(|f| {
        f.blocks.iter().any(|b| {
            b.instructions.iter().any(|inst| {
                inst.class.opcode == spirv::Op::FunctionCall
                    && matches!(
                        inst.operands.first(),
                        Some(Operand::IdRef(callee)) if bodiless.contains(callee)
                    )
            })
        })
    })
}

/// Reconcile whole-buffer scalar fallback arms in place (byte-0 base of an FC-multiplexed
/// raw-modeled device buffer) to the merge's scalar pointee, so a cross-binding pointer merge's arms
/// share one pointee type (see [`buffer_arm_reconcile`]). Errors if nothing was reconciled. Applied
/// only in the adopt-if-VALIDATES `fc_promote_psb` retry (feeding PSB), so floor-safe by
/// construction; byte-EXACT (element 0 = offset 0 for either scalar element type).
pub(crate) fn reconcile_whole_buffer_scalar_arms_module(module: &mut Module) -> Result<(), String> {
    if !buffer_arm_reconcile::reconcile_whole_buffer_scalar_arms(module) {
        return Err("native emitter: no whole-buffer scalar arm to reconcile".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}

/// Apply the W2 relooper (switch-dispatch + register demotion) in place with the default block cap.
/// Errors if no function was rewritten. The caller (the failure-triggered cfg retry) adopts the
/// result ONLY if it independently validates, so this is floor-safe by construction.
pub(crate) fn rewrite_to_relooper_module(module: &mut Module) -> Result<(), String> {
    rewrite_to_relooper_module_capped(module, relooper::default_max_relooper_blocks())
}

/// Product-safe block budget for the function-constant-dead prune → relooper composition. Pruning
/// may shrink an oversized function below this ceiling; it never licenses a larger state machine.
pub const PRUNE_THEN_RELOOPER_MAX_BLOCKS: usize = 1024;

/// Higher retry budget for an unrepaired CFG that the structurizer emitted as a REJECT. The normal
/// relooper stays capped at 1024 blocks; the whole-function relooper independently clamps every
/// caller to its hard downstream-driver safety ceiling.
pub const CFG_EMIT_RELOOPER_MAX_BLOCKS: usize = 1024;

/// Like [`rewrite_to_relooper_module`] but with an explicit requested block cap. The native
/// whole-function relooper always clamps that request to its hard downstream-driver safety limit.
pub(crate) fn rewrite_to_relooper_module_capped(
    module: &mut Module,
    max_blocks: usize,
) -> Result<(), String> {
    if !relooper::rewrite_to_relooper(module, max_blocks) {
        return Err("native emitter: no function to relooper".to_string());
    }
    drop_dangling_debug_targets_module(module);
    add_native_module_capabilities(module);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::{Block, Function};
    use spirv::Decoration;

    fn inst(op: Op, ty: Option<Word>, id: Option<Word>, operands: Vec<Operand>) -> Instruction {
        Instruction::new(op, ty, id, operands)
    }

    #[test]
    fn access_chain_storage_follows_late_substituted_base() {
        let mut module = Module::default();
        module.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
            ),
            inst(
                Op::TypeVector,
                None,
                Some(2),
                vec![Operand::IdRef(1), Operand::LiteralBit32(2)],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(3),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(2),
                ],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(4),
                vec![
                    Operand::StorageClass(StorageClass::UniformConstant),
                    Operand::IdRef(2),
                ],
            ),
            inst(
                Op::TypeInt,
                None,
                Some(5),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            inst(
                Op::Constant,
                Some(5),
                Some(6),
                vec![Operand::LiteralBit32(0)],
            ),
            inst(
                Op::Variable,
                Some(3),
                Some(7),
                vec![Operand::StorageClass(StorageClass::Private)],
            ),
        ];
        module.functions = vec![Function {
            blocks: vec![Block {
                label: None,
                instructions: vec![inst(
                    Op::InBoundsAccessChain,
                    Some(4),
                    Some(8),
                    vec![Operand::IdRef(7), Operand::IdRef(6)],
                )],
            }],
            ..Default::default()
        }];

        assert!(reconcile_access_chain_storage_classes_module(&mut module));
        assert_eq!(
            module.functions[0].blocks[0].instructions[0].result_type,
            Some(3)
        );
        assert!(!reconcile_access_chain_storage_classes_module(&mut module));
    }

    fn workgroup_struct_byte_store_module(byte_offset: u32) -> Module {
        let mut module = Module::default();
        module.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
            ),
            inst(
                Op::TypeInt,
                None,
                Some(2),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            inst(
                Op::TypeStruct,
                None,
                Some(3),
                vec![Operand::IdRef(2), Operand::IdRef(1)],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(4),
                vec![
                    Operand::StorageClass(StorageClass::Workgroup),
                    Operand::IdRef(3),
                ],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(5),
                vec![
                    Operand::StorageClass(StorageClass::Workgroup),
                    Operand::IdRef(1),
                ],
            ),
            inst(
                Op::Constant,
                Some(2),
                Some(6),
                vec![Operand::LiteralBit32(byte_offset)],
            ),
            inst(Op::ConstantNull, Some(1), Some(7), vec![]),
            inst(
                Op::Variable,
                Some(4),
                Some(8),
                vec![Operand::StorageClass(StorageClass::Workgroup)],
            ),
        ];
        module.annotations = vec![
            inst(
                Op::MemberDecorate,
                None,
                None,
                vec![
                    Operand::IdRef(3),
                    Operand::LiteralBit32(0),
                    Operand::Decoration(Decoration::Offset),
                    Operand::LiteralBit32(0),
                ],
            ),
            inst(
                Op::MemberDecorate,
                None,
                None,
                vec![
                    Operand::IdRef(3),
                    Operand::LiteralBit32(1),
                    Operand::Decoration(Decoration::Offset),
                    Operand::LiteralBit32(4),
                ],
            ),
        ];
        module.functions = vec![Function {
            blocks: vec![Block {
                label: None,
                instructions: vec![
                    inst(
                        Op::PtrAccessChain,
                        Some(5),
                        Some(9),
                        vec![Operand::IdRef(8), Operand::IdRef(6)],
                    ),
                    inst(
                        Op::Store,
                        None,
                        None,
                        vec![Operand::IdRef(9), Operand::IdRef(7)],
                    ),
                ],
            }],
            ..Default::default()
        }];
        module
    }

    #[test]
    fn drops_workgroup_struct_padding_byte_zero_store() {
        let mut module = workgroup_struct_byte_store_module(5);

        assert!(drop_workgroup_struct_padding_byte_zero_stores_module(
            &mut module
        ));
        let insts = &module.functions[0].blocks[0].instructions;
        assert!(
            !insts.iter().any(|inst| inst.result_id == Some(9)),
            "padding byte chain should be removed"
        );
        assert!(
            !insts.iter().any(|inst| {
                inst.class.opcode == Op::Store
                    && matches!(inst.operands.first(), Some(Operand::IdRef(9)))
            }),
            "padding zero store should be removed"
        );
    }

    #[test]
    fn drops_derived_workgroup_struct_padding_byte_zero_stores() {
        let mut module = workgroup_struct_byte_store_module(6);
        module.types_global_values.extend([
            inst(
                Op::Constant,
                Some(2),
                Some(10),
                vec![Operand::LiteralBit32(1)],
            ),
            inst(
                Op::TypeInt,
                None,
                Some(12),
                vec![Operand::LiteralBit32(16), Operand::LiteralBit32(0)],
            ),
            inst(Op::ConstantNull, Some(12), Some(13), vec![]),
            inst(
                Op::Constant,
                Some(2),
                Some(16),
                vec![Operand::LiteralBit32(8)],
            ),
        ]);
        module.functions[0].blocks[0].instructions = vec![
            inst(
                Op::PtrAccessChain,
                Some(5),
                Some(9),
                vec![Operand::IdRef(8), Operand::IdRef(6)],
            ),
            inst(Op::UConvert, Some(1), Some(14), vec![Operand::IdRef(13)]),
            inst(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(9), Operand::IdRef(14)],
            ),
            inst(
                Op::PtrAccessChain,
                Some(5),
                Some(11),
                vec![Operand::IdRef(9), Operand::IdRef(10)],
            ),
            inst(
                Op::ShiftRightLogical,
                Some(12),
                Some(15),
                vec![Operand::IdRef(13), Operand::IdRef(16)],
            ),
            inst(Op::UConvert, Some(1), Some(17), vec![Operand::IdRef(15)]),
            inst(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(11), Operand::IdRef(17)],
            ),
        ];

        assert!(drop_workgroup_struct_padding_byte_zero_stores_module(
            &mut module
        ));
        let insts = &module.functions[0].blocks[0].instructions;
        assert!(
            !insts
                .iter()
                .any(|inst| inst.result_id == Some(9) || inst.result_id == Some(11)),
            "direct and derived padding byte chains should be removed"
        );
        assert!(
            !insts.iter().any(|inst| inst.class.opcode == Op::Store),
            "all zero padding byte stores should be removed"
        );
    }

    #[test]
    fn keeps_workgroup_struct_member_byte_zero_store() {
        let mut module = workgroup_struct_byte_store_module(4);

        assert!(!drop_workgroup_struct_padding_byte_zero_stores_module(
            &mut module
        ));
        let insts = &module.functions[0].blocks[0].instructions;
        assert!(insts.iter().any(|inst| inst.result_id == Some(9)));
    }

    #[test]
    fn drops_debug_records_for_undefined_targets() {
        let mut module = Module::default();
        module.types_global_values = vec![inst(
            Op::TypeInt,
            None,
            Some(1),
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        )];
        module.debug_names = vec![
            inst(
                Op::Name,
                None,
                None,
                vec![Operand::IdRef(1), Operand::LiteralString("live".into())],
            ),
            inst(
                Op::Name,
                None,
                None,
                vec![Operand::IdRef(99), Operand::LiteralString("dead".into())],
            ),
        ];
        module.annotations = vec![
            inst(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(1),
                    Operand::Decoration(Decoration::RelaxedPrecision),
                ],
            ),
            inst(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(99),
                    Operand::Decoration(Decoration::RelaxedPrecision),
                ],
            ),
        ];

        assert!(drop_dangling_debug_targets_module(&mut module));
        assert_eq!(module.debug_names.len(), 1);
        assert_eq!(module.annotations.len(), 1);
        assert_eq!(
            module.debug_names[0].operands.first(),
            Some(&Operand::IdRef(1))
        );
        assert_eq!(
            module.annotations[0].operands.first(),
            Some(&Operand::IdRef(1))
        );
    }
}
