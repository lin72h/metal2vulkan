//! Native LLVM-IR -> Vulkan SPIR-V emitter.
//!
//! It emits a pre-interface Logical-addressing Shader module with the AIR entry as a normal
//! function, typed parameters, and an aggregate/vector return. Retained-SPIR-V interface and
//! lowering passes consume that module.

mod async_copy;
mod cfg;
mod constfold;
// Read-only typed-SSA soundness / carrier-comparison / reject-census diagnostics (no emission path).
mod diagnostics;
mod emitter;
// Emitter entry points for the primary and structurally selected alternate representations.
mod emit_tiers;
mod imageblock;
// Non-recursive internal-function inliner (pre-emit AIR-text pass).
mod inline;
pub(crate) mod ir;
mod lex;
mod opaque_image_select;
mod owned_cfg;
mod parse;
mod private_vector_word;
mod psb;
mod psb_value_select;
pub(crate) mod ray_intersection;
mod relooper;
mod render;
// Owned-module construction rewrites and compatibility byte adapters.
mod rewrites;
mod vec_scalar_merge;
mod wg_atomic;

// Parse-once typed SSA/CFG carrier used by production analysis, restructuring, and emission.
mod tir;

use crate::spirv_module::Instruction;
use crate::spirv_module::Module;
use crate::spirv_module::Operand;
pub(crate) use async_copy::{lower_simdgroup_async_copy, lower_simdgroup_async_copy_owned};
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
    emit_vulkan_spirv_all_buffers_raw_with_workgroup,
    emit_vulkan_spirv_with_primitive_phi_metadata,
};
pub(crate) use emit_tiers::{
    emit_vulkan_spirv_all_buffers_raw_bda_with_sidecar,
    emit_vulkan_spirv_all_buffers_raw_relooper_feed_with_sidecar,
    emit_vulkan_spirv_all_buffers_raw_with_sidecar,
    emit_vulkan_spirv_all_buffers_raw_with_workgroup_sidecar, emit_vulkan_spirv_with_outcome,
    emit_vulkan_spirv_with_sidecar,
};
use emitter::Emitter;
use ir::LlModule;

pub(crate) fn inline_direct_function_pointer_consumers(
    san_ll: &str,
    direct_functions: &std::collections::HashSet<String>,
) -> String {
    inline::inline_direct_function_pointer_consumers(san_ll, direct_functions)
}

pub(crate) use owned_cfg::{owned_module_failure, OwnedModuleFailure};
pub(crate) use parse::{
    parse_return_type as parse_llvm_return_type, parse_type_prefix as parse_llvm_type_prefix,
};
pub(crate) use rewrites::close_private_vector_word_views_module;
#[cfg(test)]
pub(crate) use rewrites::{address_construction_count, reset_address_construction_counts};
pub(crate) use rewrites::{
    close_inlined_bda_pointer_tables_module, construct_cfg_functions_module,
    construct_interface_cross_binding_pointer_merges_module,
    construct_interface_cross_binding_pointer_phis_module,
    construct_interface_cross_binding_pointer_values_module, construct_opaque_image_selects_module,
    construct_physical_atomic_pointer_lvalues_module, construct_workgroup_atomic_floats_module,
    eliminate_dead_pointer_values_module, eliminate_dead_values_module,
    lower_unobserved_bda_aggregate_pointer_fields_module, prune_constant_branches_module,
    prune_constant_cfg_module_if_changed, prune_unused_null_and_undef_constants_module,
    unowned_selection_header_labels,
};
pub use rewrites::{BOUNDED_RELOOPER_MAX_BLOCKS, CFG_EMIT_RELOOPER_MAX_BLOCKS};
use spirv::{Capability, Op, StorageClass, Word};
use std::collections::{HashMap, HashSet};

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

#[cfg(test)]
mod tests;
