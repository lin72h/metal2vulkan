//! Native LLVM-IR -> Vulkan SPIR-V emitter.
//!
//! This starts as a deliberately small emitter for the post-sanitize AIR LLVM-IR subset and grows
//! construct-by-construct. It emits the same pre-interface shape the old `llc` step produced: a
//! Logical-addressing Shader module with the AIR entry as a normal function, typed parameters, and
//! an aggregate/vector return. The existing SPIR-V interface/lowering passes consume that module.

mod async_copy;
mod buffer_arm_reconcile;
mod cfg;
mod constfold;
// Read-only typed-SSA soundness / carrier-comparison / reject-census diagnostics (no emission path).
mod diagnostics;
mod emitter;
// Emitter entry points: default typed emit + the adopt-if-validates retry-tier variants.
mod emit_tiers;
// Retry-routing error classifiers (spirv-val / emit message text → `ValidationClass`/`EmitErrorClass`).
mod error_class;
mod imageblock;
// Non-recursive internal-function inliner (pre-emit AIR-text pass). Wired into the inline+SROA
// retry tiers (`emit_vulkan_spirv_inline_sroa`/`_raw`).
mod inline;
pub(crate) mod ir;
mod lex;
mod opaque_image_select;
mod parse;
mod phi_index;
mod psb;
mod psb_value_select;
mod relooper;
mod render;
// Byte-level SPIR-V rewrite adapters (the failure-triggered retry tiers' legalization passes).
mod rewrites;
mod sroa;
mod subword_pack;
mod vec_scalar_merge;
mod wg_atomic;
// Parse-once typed SSA/CFG carrier used by production analysis, restructuring, and emission.
mod tir;

use crate::spirv_module::Instruction;
use crate::spirv_module::Operand;
use crate::spirv_module::{load_bytes, Module};
pub(crate) use async_copy::lower_simdgroup_async_copy;
pub use diagnostics::{
    cond_other_witness_report, irreducible_region_report, param_pointee_check,
    straddle_region_report, straddle_witness_report, structured_reject_loop_classes,
    structured_reject_reasons, tir_gep_pointee_report, tir_pointee_check, tir_self_check,
    tir_storage_check, tir_structured_self_check, ParamPointeeStats, PointeeCheckStats,
    StorageCheckStats, TirCheckStats,
};
#[cfg(test)]
pub(in crate::native) use emit_tiers::emit_vulkan_spirv_from_typed_blocks;
pub use emit_tiers::{
    emit_vulkan_spirv, emit_vulkan_spirv_all_buffers_raw, emit_vulkan_spirv_all_buffers_raw_bda,
    emit_vulkan_spirv_all_buffers_raw_relooper_feed,
    emit_vulkan_spirv_all_buffers_raw_with_workgroup, emit_vulkan_spirv_construct_tree,
    emit_vulkan_spirv_inline_sroa, emit_vulkan_spirv_inline_sroa_raw,
    emit_vulkan_spirv_inline_sroa_raw_cfg_restructure,
    emit_vulkan_spirv_with_primitive_phi_metadata,
};
pub(crate) use emit_tiers::{
    emit_vulkan_spirv_all_buffers_raw_bda_with_sidecar,
    emit_vulkan_spirv_all_buffers_raw_relooper_feed_with_sidecar,
    emit_vulkan_spirv_all_buffers_raw_with_sidecar,
    emit_vulkan_spirv_all_buffers_raw_with_workgroup_sidecar,
    emit_vulkan_spirv_construct_tree_with_sidecar,
    emit_vulkan_spirv_inline_sroa_raw_cfg_restructure_with_sidecar,
    emit_vulkan_spirv_inline_sroa_raw_with_sidecar, emit_vulkan_spirv_inline_sroa_with_sidecar,
    emit_vulkan_spirv_with_primitive_phi_metadata_sidecar, emit_vulkan_spirv_with_sidecar,
};
use emitter::Emitter;
pub use error_class::{
    classify_emit_error, classify_validation_error, is_cfg_structurization_error,
    is_cross_binding_pointer_merge_error, is_dynamic_struct_index_error,
    is_logical_pointer_operand_error, is_logical_pointer_phi_error, is_pointer_typing_emit_error,
    is_pointer_typing_validation_error, EmitErrorClass, ValidationClass,
};
use ir::LlModule;
pub(crate) use rewrites::{
    demote_nondominating_values_module, rewrite_integer_width_phis_module,
    rewrite_logical_pointer_phis_module, split_multientry_loop_selection_exits_module,
    value_lower_cross_binding_pointer_merges_module,
};
pub(crate) use rewrites::{
    has_bodiless_agx_call_module, has_cross_binding_pointer_phi_module,
    module_has_wide_raw_store_guard, prune_constant_branches_module,
    prune_constant_branches_module_preserving, reconcile_whole_buffer_scalar_arms_module,
    rewrite_cross_binding_pointer_merges_module,
    rewrite_cross_binding_pointer_merges_to_values_module,
    rewrite_cross_binding_pointer_phis_module, rewrite_logical_pointer_phis_retry_module,
    rewrite_opaque_image_selects_module, rewrite_subword_packed_scalars_module,
    rewrite_to_relooper_module, rewrite_to_relooper_module_capped,
    rewrite_variable_pointer_phis_module, rewrite_workgroup_atomic_floats_module,
};
pub use rewrites::{CFG_EMIT_RELOOPER_MAX_BLOCKS, PRUNE_THEN_RELOOPER_MAX_BLOCKS};
use spirv::{Capability, Op, StorageClass, Word};
use std::collections::{HashMap, HashSet};

/// Compatibility wrapper for the public diagnostic/probe surface. Production retries use
/// [`rewrite_cross_binding_pointer_merges_module`] so adjacent module passes can compose without
/// serializing and reparsing between them.
pub fn rewrite_cross_binding_pointer_merges_bytes(spv: &[u8]) -> Result<Vec<u8>, String> {
    let mut module = load_bytes(spv).map_err(|e| format!("SPIR-V load: {e:?}"))?;
    rewrite_cross_binding_pointer_merges_module(&mut module)?;
    Ok(module
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect())
}

/// Compatibility wrapper for the public diagnostic/probe surface. Production retries use
/// [`rewrite_cross_binding_pointer_merges_to_values_module`] so adjacent module passes can compose
/// without serializing and reparsing between them.
pub fn rewrite_cross_binding_pointer_merges_to_values_bytes(spv: &[u8]) -> Result<Vec<u8>, String> {
    let mut module = load_bytes(spv).map_err(|e| format!("SPIR-V load: {e:?}"))?;
    rewrite_cross_binding_pointer_merges_to_values_module(&mut module)?;
    Ok(module
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect())
}

/// Compatibility wrapper for the public diagnostic/probe surface. Production retries use
/// [`prune_constant_branches_module`] so adjacent module passes can compose without serializing and
/// reparsing between them.
pub fn prune_constant_branches_bytes(spv: &[u8]) -> Result<Vec<u8>, String> {
    let mut module = load_bytes(spv).map_err(|e| format!("SPIR-V load: {e:?}"))?;
    prune_constant_branches_module(&mut module)?;
    Ok(module
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect())
}

/// Compatibility wrapper for the public diagnostic/probe surface. Production retries use
/// [`rewrite_to_relooper_module`] so adjacent module passes can compose without serializing and
/// reparsing between them.
pub fn rewrite_to_relooper_bytes(spv: &[u8]) -> Result<Vec<u8>, String> {
    let mut module = load_bytes(spv).map_err(|e| format!("SPIR-V load: {e:?}"))?;
    rewrite_to_relooper_module(&mut module)?;
    Ok(module
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect())
}

/// Explicit-cap compatibility wrapper for the public diagnostic/probe surface.
pub fn rewrite_to_relooper_bytes_capped(spv: &[u8], max_blocks: usize) -> Result<Vec<u8>, String> {
    let mut module = load_bytes(spv).map_err(|e| format!("SPIR-V load: {e:?}"))?;
    rewrite_to_relooper_module_capped(&mut module, max_blocks)?;
    Ok(module
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect())
}

pub(crate) fn close_module_capabilities(module: &mut Module) {
    add_native_module_capabilities(module);
}

fn add_native_module_capabilities(module: &mut Module) {
    crate::spirv_variable_ptr::lower_zero_base_storage_buffer_ptr_access_chains(module);
    let (has_storage_buffer_pointer_merge, has_other_pointer_merge) =
        crate::spirv_variable_ptr::variable_pointer_requirement(module);
    // This helper runs after post-finish module rewrites as well as fresh native emission. Those
    // rewrites can remove the last pointer-typed OpPhi/OpSelect/OpPtrAccessChain from a module that
    // already declared VariablePointers*. Keep the capability set closed over the current module
    // shape instead of only adding requirements; stale declarations are legal SPIR-V, but they force
    // native drivers down a variable-pointer compiler path.
    module
        .capabilities
        .retain(|inst| match inst.operands.as_slice() {
            [Operand::Capability(Capability::VariablePointersStorageBuffer)] => {
                has_storage_buffer_pointer_merge
            }
            [Operand::Capability(Capability::VariablePointers)] => has_other_pointer_merge,
            _ => true,
        });
    if has_storage_buffer_pointer_merge {
        require_capability(module, Capability::VariablePointersStorageBuffer);
    }
    if has_other_pointer_merge {
        require_capability(module, Capability::VariablePointers);
    }
}

fn require_capability(module: &mut Module, capability: Capability) {
    if module.capabilities.iter().any(|inst| {
        matches!(
            inst.operands.as_slice(),
            [Operand::Capability(existing)] if *existing == capability
        )
    }) {
        return;
    }
    module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(capability)],
    ));
}

fn ptr_storage(defs: &HashMap<Word, Instruction>, ptr_ty: Word) -> Option<StorageClass> {
    let inst = defs.get(&ptr_ty)?;
    if inst.class.opcode != Op::TypePointer {
        return None;
    }
    match inst.operands.first()? {
        Operand::StorageClass(storage) => Some(*storage),
        _ => None,
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod error_classifier_tests;
