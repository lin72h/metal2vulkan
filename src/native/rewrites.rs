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

/// Apply the W1 PhysicalStorageBuffer64 lowering in place. Errors if no cross-binding pointer-merge
/// sub-graph was rewritten. The caller (the failure-triggered retry) adopts the result ONLY if it
/// independently validates, so this is floor-safe by construction.
pub(crate) fn rewrite_cross_binding_pointer_merges_module(
    module: &mut Module,
) -> Result<(), String> {
    if !psb::rewrite_cross_binding_pointer_merges(module) {
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
pub(crate) fn rewrite_cross_binding_pointer_phis_module(module: &mut Module) -> Result<(), String> {
    if !psb::rewrite_cross_binding_pointer_phis(module) {
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
    add_native_module_capabilities(module);
    Ok(())
}

/// Preserving form of [`prune_constant_branches_module`] for a primary module that still carries
/// typed sidecar roots. Returns whether pruning changed the module.
pub(crate) fn prune_constant_branches_module_preserving(
    module: &mut Module,
    preserved_global_ids: &[Word],
) -> bool {
    let roots = preserved_global_ids.iter().copied().collect();
    constfold::prune_constant_branches_preserving(module, &roots)
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

/// Higher block budget for the function-constant-dead prune → relooper composition. Pruning can
/// shrink an otherwise oversized function below this ceiling while it remains above the normal
/// relooper budget.
pub const PRUNE_THEN_RELOOPER_MAX_BLOCKS: usize = 2048;

/// Bounded high cap for the relooper retry over an unrepaired CFG that the structurizer emitted as a
/// REJECT (complete but spirv-val-invalid bytes). The normal relooper stays capped at 1024 blocks;
/// this retry-only cap is sufficient for the complete graph a reject function emits, while remaining
/// below the 8192 diagnostic ceiling for the separately measured huge emitted-module path.
pub const CFG_EMIT_RELOOPER_MAX_BLOCKS: usize = 2048;

/// Like [`rewrite_to_relooper_module`] but with an explicit block cap. The prune-then-relooper
/// retry uses a higher cap than the default 1024: its input is a function whose statically-dead
/// function-constant arms were already pruned away (byte-correct DCE), so a >1024-block source can
/// land below the cap after pruning yet still exceed the default. Lifting the cap only for that
/// already-failing, adopt-if-validates path keeps the normal relooper's perf budget intact while
/// admitting the pruned huge-function cfg cases.
pub(crate) fn rewrite_to_relooper_module_capped(
    module: &mut Module,
    max_blocks: usize,
) -> Result<(), String> {
    if !relooper::rewrite_to_relooper(module, max_blocks) {
        return Err("native emitter: no function to relooper".to_string());
    }
    add_native_module_capabilities(module);
    Ok(())
}
