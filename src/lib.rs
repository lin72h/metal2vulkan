//! metal2vulkan — Metal AIR (LLVM bitcode) -> Vulkan SPIR-V, via a native LLVM-IR emitter.
//!
//! The native emitter produces `OpCapability Shader` / `Logical GLSL450` SPIR-V directly from
//! sanitized AIR LLVM IR. Crate-owned retained-SPIR-V passes then build the Vulkan stage interface,
//! lower residual AIR operations, normalize memory access and control flow, and finalize the module
//! (see [`passes`]).
//!
//! Pipeline: `.air|.ll` -> llvm-dis -> sanitize -> native Vulkan SPIR-V emit -> retained crate
//! module -> interface+lowering passes -> assemble -> spirv-val (vulkan1.3).

// `too_many_arguments` and `type_complexity` are threshold heuristics that fire pervasively and
// benignly across this translator: emit/lowering functions legitimately thread many typed
// parameters (stage + metas + module + ctx + resolved ids), and the IR/emit return shapes are
// genuinely nested domain types (e.g. `Result<Option<Vec<(Vec<u32>, LlType)>>, String>`). Factoring
// every such signature behind a wrapper struct or a one-off `type` alias would add indirection
// without improving clarity, so both are accepted crate-wide rather than scattered per-site.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub mod air_intrinsics;
pub mod as_shadow;
mod emit_sidecar;
pub mod env_vars;
mod fc_specialize;
mod layout;
pub mod linked_functions;
pub mod meta;
pub mod native;
pub mod passes;
mod passthrough;
mod primary_retry;
pub mod reflect;
mod retry;
mod spirv_binary;
mod spirv_module;
mod spirv_operand;
mod spirv_variable_ptr;
pub mod tools;
pub(crate) mod types;

pub use fc_specialize::{
    specialize_function_constant_bytes, specialize_function_constants,
    specialize_function_constants_zero,
};
pub use passthrough::{translate_passthrough, translate_vertex_observer};
use primary_retry::*;

use crate::spirv_module::{load_bytes as load_owned_module, Module};
use std::path::Path;

/// Detect the shader stage from the AIR's own `!air.vertex`/`!air.fragment`/`!air.kernel` metadata
/// (which SPIR-V emission later drops). This lets callers translate an AIR blob without separately
/// carrying its stage. Supplying the wrong stage can mis-map stage-interface roles, so prefer this
/// function when the metadata is present.
pub fn detect_stage(src: &str, tmp: &Path) -> Result<passes::Stage, String> {
    let ll = tools::air_to_sanitized_ll(src, tmp)?;
    if ll.contains("!air.vertex =") {
        Ok(passes::Stage::Vertex)
    } else if ll.contains("!air.fragment =") {
        Ok(passes::Stage::Fragment)
    } else if ll.contains("!air.kernel =") {
        Ok(passes::Stage::Kernel)
    } else {
        Err(
            "metal2vulkan: no !air.vertex/!air.fragment/!air.kernel stage metadata in module"
                .into(),
        )
    }
}

/// Translate an AIR bitcode or LLVM-IR file to Vulkan SPIR-V for `stage`.
///
/// The final module is validated with `spirv-val` under the Vulkan 1.3 environment. A primary module
/// that fails validation may enter the validation-gated retry cascade; only a validating candidate
/// is returned. `tmp` is caller-owned scratch space and may be reused sequentially, but callers
/// should give concurrent translations separate directories.
pub fn translate(src: &str, stage: passes::Stage, tmp: &Path) -> Result<Vec<u8>, String> {
    translate_with_options(src, stage, tmp, passes::TransformOptions::default())
}

pub fn translate_with_options(
    src: &str,
    stage: passes::Stage,
    tmp: &Path,
    options: passes::TransformOptions,
) -> Result<Vec<u8>, String> {
    let (san_ll, datalayout) = tools::air_to_sanitized_ll_with_datalayout(src, tmp)?;
    let datalayout = datalayout
        .as_deref()
        .map(layout::AirDataLayout::parse)
        .transpose()?;
    translate_sanitized_native_with_options_and_layout(&san_ll, stage, tmp, options, datalayout)
}

/// Translate already-sanitized LLVM IR through the native emitter.
pub fn translate_sanitized_native(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
) -> Result<Vec<u8>, String> {
    translate_sanitized_native_with_options(san_ll, stage, tmp, passes::TransformOptions::default())
}

/// Apply the shared pre-emit AIR lowering before any path derives stage metadata or retries from the
/// module. Retries re-emit the supplied text directly, so using the original intrinsic-bearing text
/// after the primary has already lowered it would make the two paths observe different programs.
/// Floor-safe: `lower_simdgroup_async_copy` is a no-op unless the module calls
/// `air.simdgroup_async_copy_2d` (such modules fail the emitter outright otherwise).
fn lower_async_copy_if_enabled(san_ll: &str) -> String {
    native::lower_simdgroup_async_copy(san_ll)
}

fn reject_unsupported_metal_linked_functions(san_ll: &str) -> Result<(), String> {
    if san_ll.contains(".MTL_VISIBLE_FN_REF") || san_ll.contains("!air.visible_function_references")
    {
        return Err(
            "native emitter: unsupported Metal visible function reference; dynamic linked \
             functions are not expressible in Logical SPIR-V"
                .into(),
        );
    }
    Ok(())
}

fn options_for_air(
    san_ll: &str,
    mut options: passes::TransformOptions,
) -> Result<passes::TransformOptions, String> {
    options
        .descriptor_layout
        .validate()
        .map_err(|error| error.to_string())?;
    options.validate_runtime_samplers()?;
    options.validate_runtime_storage_images()?;
    if san_ll.contains("air.compile.denorms_disable") {
        options.denorm_flush_to_zero_f32 = true;
    }
    // AIR's simdgroup ABI is 32 lanes. Vulkan implementations may expose wider native subgroups
    // (MoltenVK commonly exposes 64), so subgroup reductions/scans must retain 32-lane partitions
    // instead of silently adopting the driver's width.
    if san_ll.contains("@air.simd_") {
        options.simd_cluster32 = true;
    }
    Ok(options)
}

/// Per-stage interface metadata parsed once from sanitized AIR and shared by emission, passes,
/// reflection, and every retry tier. Kernel translation also retains the FC-buffer-promoted
/// projection used by one adopt-if-validates retry; both projections share one decoded metadata-node
/// table.
struct StageMeta {
    frag: Option<meta::FragMeta>,
    vert: Option<meta::VertMeta>,
    kern: Option<meta::KernMeta>,
    promoted_kern: Option<meta::KernMeta>,
    entry_name: Option<String>,
}

fn parse_stage_meta(san_ll: &str, stage: passes::Stage) -> StageMeta {
    match stage {
        passes::Stage::Fragment => {
            let (frag, entry_name) = meta::parse_air_fragment_meta_with_entry(san_ll);
            StageMeta {
                frag,
                vert: None,
                kern: None,
                promoted_kern: None,
                entry_name,
            }
        }
        passes::Stage::Vertex => {
            let (vert, entry_name) = meta::parse_air_vertex_meta_with_entry(san_ll);
            StageMeta {
                frag: None,
                vert,
                kern: None,
                promoted_kern: None,
                entry_name,
            }
        }
        passes::Stage::Kernel => {
            let (kern, promoted_kern, entry_name) = meta::parse_air_kernel_meta_variants(san_ll);
            StageMeta {
                frag: None,
                vert: None,
                kern,
                promoted_kern,
                entry_name,
            }
        }
    }
}

fn stage_buffer_layouts<'a>(
    stage: passes::Stage,
    frag: Option<&'a meta::FragMeta>,
    vert: Option<&'a meta::VertMeta>,
    kern: Option<&'a meta::KernMeta>,
) -> Option<&'a std::collections::HashMap<u32, meta::AirType>> {
    match stage {
        passes::Stage::Fragment => frag.map(|meta| &meta.buffer_layouts),
        passes::Stage::Vertex => vert.map(|meta| &meta.buffer_layouts),
        passes::Stage::Kernel => kern.map(|meta| &meta.buffer_layouts),
    }
}

fn is_runtime_storage_image_binding(binding: &reflect::ResourceBinding, metal_index: u32) -> bool {
    binding.metal_index == metal_index
        && (binding.kind == reflect::ResourceKind::StorageImage
            || matches!(
                binding.kind,
                reflect::ResourceKind::TextureArray
                    | reflect::ResourceKind::EmbeddedArgBufferTexture
            ) && binding.access == Some(reflect::ResourceAccess::Storage))
}

/// Build the [`reflect::ShaderReflection`] facade from the already-parsed stage metadata. Pure
/// re-shaping of parsed data — never touches the emitted SPIR-V, so the reflected translate paths
/// produce byte-identical bytes to their non-reflected siblings.
fn build_reflection(
    stage: passes::Stage,
    frag: Option<&meta::FragMeta>,
    vert: Option<&meta::VertMeta>,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    options: &passes::TransformOptions,
) -> Result<reflect::ShaderReflection, String> {
    let mut reflection = match stage {
        passes::Stage::Fragment => {
            reflect::ShaderReflection::from_fragment(&frag.cloned().unwrap_or_default(), entry_name)
        }
        passes::Stage::Vertex => {
            reflect::ShaderReflection::from_vertex(&vert.cloned().unwrap_or_default(), entry_name)
        }
        passes::Stage::Kernel => reflect::ShaderReflection::from_kernel(
            &kern.cloned().unwrap_or_default(),
            entry_name,
            options.kernel_local_size,
        ),
    };
    let runtime_sampler_indices = reflection
        .bindings
        .iter()
        .filter(|binding| binding.kind == reflect::ResourceKind::Sampler)
        .map(|binding| binding.metal_index)
        .collect::<std::collections::BTreeSet<_>>();
    reflection.runtime_sampler_specializations = options
        .runtime_sampler_states
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(metal_index, state)| {
            let metal_index = u32::try_from(metal_index).ok()?;
            if !runtime_sampler_indices.contains(&metal_index) {
                return None;
            }
            Some(reflect::RuntimeSamplerSpecialization {
                metal_index,
                state: state?,
            })
        })
        .collect();
    for (metal_index, state) in options
        .runtime_storage_image_states
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(metal_index, state)| Some((u32::try_from(metal_index).ok()?, state?)))
    {
        let mut applied = false;
        let spirv_format = state.format.explicit_format();
        for binding in reflection
            .bindings
            .iter_mut()
            .filter(|binding| is_runtime_storage_image_binding(binding, metal_index))
        {
            applied = true;
            if let Some(shape) = binding.texture_shape.as_mut() {
                shape.storage_format = spirv_format;
            }
        }
        if !applied {
            continue;
        }
        reflection.runtime_storage_image_specializations.push(
            reflect::RuntimeStorageImageSpecialization {
                metal_index,
                state,
                spirv_format,
            },
        );
    }
    reflection.apply_descriptor_layout(options.descriptor_layout)?;
    Ok(reflection)
}

fn validate_reflected_runtime_storage_images(
    reflection: &reflect::ShaderReflection,
    options: &passes::TransformOptions,
) -> Result<(), String> {
    for (metal_index, state) in options
        .runtime_storage_image_states
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(metal_index, state)| Some((u32::try_from(metal_index).ok()?, state?)))
    {
        let specialized = reflection
            .runtime_storage_image_specializations
            .iter()
            .any(|specialization| specialization.metal_index == metal_index);
        if !specialized {
            return Err(format!(
                "runtime storage image {metal_index}: no reflected storage-image binding exists for runtime format {:?}",
                state.format
            ));
        }
        let runtime_component = state.format.component();
        for binding in reflection
            .bindings
            .iter()
            .filter(|binding| is_runtime_storage_image_binding(binding, metal_index))
        {
            let Some(shape) = binding.texture_shape else {
                return Err(format!(
                    "runtime storage image {metal_index}: reflected storage-image binding has no texture shape"
                ));
            };
            if shape.component != runtime_component {
                return Err(format!(
                    "runtime storage image {metal_index}: AIR texels are {:?}, but runtime format {:?} is {runtime_component:?}",
                    shape.component, state.format
                ));
            }
        }
    }
    Ok(())
}

/// Diagnostic probe (used by `historical PSB dump probes`): run the exact production
/// translate path but with the W1 PhysicalStorageBuffer64 retry tiers DISABLED, returning the
/// pre-PSB emission the cascade would otherwise feed to the PSB rewrite. Lets the `--psb-dump` tool
/// apply `rewrite_cross_binding_pointer_merges_bytes` by hand for inspection without an env flip
/// (the retired `METAL2VULKAN_PSB=0`). Not a frontier signal; production always runs with PSB on.
pub fn translate_pre_psb_probe(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
) -> Result<Vec<u8>, String> {
    let lowered = lower_async_copy_if_enabled(san_ll);
    let san_ll = lowered.as_str();
    reject_unsupported_metal_linked_functions(san_ll)?;
    let stage_meta = parse_stage_meta(san_ll, stage);
    let options = options_for_air(san_ll, passes::TransformOptions::default())?;
    let datalayout = layout::AirDataLayout::from_ir(san_ll)?;
    translate_sanitized_with_meta(
        san_ll,
        stage,
        stage_meta.frag.as_ref(),
        stage_meta.vert.as_ref(),
        stage_meta.kern.as_ref(),
        stage_meta.promoted_kern.as_ref(),
        stage_meta.entry_name.as_deref(),
        tmp,
        options,
        false,
        datalayout,
    )
}

pub fn translate_sanitized_native_with_options(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
    options: passes::TransformOptions,
) -> Result<Vec<u8>, String> {
    let datalayout = layout::AirDataLayout::from_ir(san_ll)?;
    translate_sanitized_native_with_options_and_layout(san_ll, stage, tmp, options, datalayout)
}

fn translate_sanitized_native_with_options_and_layout(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
    options: passes::TransformOptions,
    datalayout: Option<layout::AirDataLayout>,
) -> Result<Vec<u8>, String> {
    // Lower `air.simdgroup_async_copy_2d` (+ its event/wait pair) to an explicit strided tile copy
    // BEFORE any meta parse or emit, so every path — the default emit AND every retry tier (which
    // re-emit from `san_ll`) — sees ordinary LLVM instead of the unhandled intrinsic. This is what lets
    // the FC-promote/prune/PSB retries fix the SECOND wall these kernels carry (a Private-demoted
    // pointer arm over-indexed as an array), which they could not before when the lowering lived only
    // inside `emit_vulkan_spirv` (the retries bypass that entry). Floor-safe by construction: the
    // rewrite is a no-op unless the module calls `air.simdgroup_async_copy_2d`, and such modules fail
    // the emitter outright today. See `native::async_copy` and its structural regression tests.
    let lowered = lower_async_copy_if_enabled(san_ll);
    let san_ll = lowered.as_str();
    reject_unsupported_metal_linked_functions(san_ll)?;
    let stage_meta = parse_stage_meta(san_ll, stage);
    let options = options_for_air(san_ll, options)?;
    translate_sanitized_with_meta(
        san_ll,
        stage,
        stage_meta.frag.as_ref(),
        stage_meta.vert.as_ref(),
        stage_meta.kern.as_ref(),
        stage_meta.promoted_kern.as_ref(),
        stage_meta.entry_name.as_deref(),
        tmp,
        options,
        true,
        datalayout,
    )
}

/// Translate sanitized AIR after resolving authored direct visible-function references and
/// function-table slots to exact linked AIR definitions. This is the portable Logical-SPIR-V
/// alternative to Metal's runtime function linker and function pointers.
pub fn translate_sanitized_native_linked_with_options(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
    options: passes::TransformOptions,
    linkage: &linked_functions::LinkedFunctionLinkage,
) -> Result<Vec<u8>, String> {
    let stage_name = match stage {
        passes::Stage::Kernel => "kernel",
        passes::Stage::Vertex => "vertex",
        passes::Stage::Fragment => "fragment",
    };
    let entry_name = meta::entry_name(san_ll, stage_name)
        .ok_or_else(|| format!("linked translation found no AIR {stage_name} entry"))?;
    let specialized =
        linked_functions::specialize_visible_function_tables(san_ll, &entry_name, linkage)?;
    let specialized =
        linked_functions::specialize_visible_function_references(&specialized, linkage)?;
    let specialized = linked_functions::specialize_opaque_triangle_intersection_tables(
        &specialized,
        &entry_name,
        linkage,
    )?;
    translate_sanitized_native_with_options(&specialized, stage, tmp, options)
}

/// Like [`translate`] but also returns the [`reflect::ShaderReflection`] needed to integrate the
/// resulting module.
///
/// Interface facts come from AIR metadata and the translator's descriptor ABI. Conservative buffer
/// footprints come from read-only analysis of the final adopted SPIR-V, after retry selection. The
/// analysis does not mutate the module, so the returned SPIR-V remains byte-identical to
/// [`translate`] for the same input, stage, and options.
pub fn translate_reflected(
    src: &str,
    stage: passes::Stage,
    tmp: &Path,
) -> Result<(Vec<u8>, reflect::ShaderReflection), String> {
    translate_reflected_with_options(src, stage, tmp, passes::TransformOptions::default())
}

pub fn translate_reflected_with_options(
    src: &str,
    stage: passes::Stage,
    tmp: &Path,
    options: passes::TransformOptions,
) -> Result<(Vec<u8>, reflect::ShaderReflection), String> {
    // Capture the `target datalayout` while sanitizing so both executable layout and reflection use
    // the same source contract without re-reading the source `.ll`. A sanitized-entry caller that
    // supplies no datalayout still leaves reflection.datalayout = None.
    let (san_ll, datalayout) = tools::air_to_sanitized_ll_with_datalayout(src, tmp)?;
    let parsed_datalayout = datalayout
        .as_deref()
        .map(layout::AirDataLayout::parse)
        .transpose()?;
    let (spv, mut reflection) = translate_sanitized_native_reflected_with_layout(
        &san_ll,
        stage,
        tmp,
        options,
        parsed_datalayout,
    )?;
    reflection.datalayout = datalayout;
    Ok((spv, reflection))
}

/// Reflect sanitized AIR without requiring its executable lowering to be supported yet.
///
/// Authored dependency validation uses this for link-time resources such as function tables: their
/// stage interface is fully described by AIR metadata even before indirect calls have been resolved
/// to linked function definitions. The returned shape is identical to the reflection attached to a
/// successful translated module.
pub fn reflect_sanitized(
    san_ll: &str,
    stage: passes::Stage,
    options: passes::TransformOptions,
) -> Result<reflect::ShaderReflection, String> {
    let lowered = lower_async_copy_if_enabled(san_ll);
    let san_ll = lowered.as_str();
    let stage_meta = parse_stage_meta(san_ll, stage);
    let options = options_for_air(san_ll, options)?;
    let mut reflection = build_reflection(
        stage,
        stage_meta.frag.as_ref(),
        stage_meta.vert.as_ref(),
        stage_meta.kern.as_ref(),
        stage_meta.entry_name.as_deref(),
        &options,
    )?;
    validate_reflected_runtime_storage_images(&reflection, &options)?;
    reflection.function_constants = meta::parse_function_constants(san_ll);
    reflection.refine_buffer_access_from_entry(san_ll);
    reflection.add_static_samplers(san_ll)?;
    if stage == passes::Stage::Kernel && has_device_address_pointer_load(san_ll) {
        reflection.add_buffer_address_table()?;
    }
    reflection.validate_descriptor_abi()?;
    Ok(reflection)
}

/// [`translate_sanitized_native_with_options`] plus the reflection facade. See [`translate_reflected`].
pub fn translate_sanitized_native_reflected(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
    options: passes::TransformOptions,
) -> Result<(Vec<u8>, reflect::ShaderReflection), String> {
    let datalayout = layout::AirDataLayout::from_ir(san_ll)?;
    translate_sanitized_native_reflected_with_layout(san_ll, stage, tmp, options, datalayout)
}

fn translate_sanitized_native_reflected_with_layout(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
    options: passes::TransformOptions,
    datalayout: Option<layout::AirDataLayout>,
) -> Result<(Vec<u8>, reflect::ShaderReflection), String> {
    let lowered = lower_async_copy_if_enabled(san_ll);
    let san_ll = lowered.as_str();
    reject_unsupported_metal_linked_functions(san_ll)?;
    let stage_meta = parse_stage_meta(san_ll, stage);
    let options = options_for_air(san_ll, options)?;
    let mut reflection = build_reflection(
        stage,
        stage_meta.frag.as_ref(),
        stage_meta.vert.as_ref(),
        stage_meta.kern.as_ref(),
        stage_meta.entry_name.as_deref(),
        &options,
    )?;
    validate_reflected_runtime_storage_images(&reflection, &options)?;
    reflection.function_constants = meta::parse_function_constants(san_ll);
    reflection.refine_buffer_access_from_entry(san_ll);
    reflection.add_static_samplers(san_ll)?;
    if stage == passes::Stage::Kernel && has_device_address_pointer_load(san_ll) {
        reflection.add_buffer_address_table()?;
    }
    reflection.validate_descriptor_abi()?;
    let spv = translate_sanitized_with_meta(
        san_ll,
        stage,
        stage_meta.frag.as_ref(),
        stage_meta.vert.as_ref(),
        stage_meta.kern.as_ref(),
        stage_meta.promoted_kern.as_ref(),
        stage_meta.entry_name.as_deref(),
        tmp,
        options,
        true,
        datalayout,
    )?;
    reflection.add_buffer_footprints(&spv)?;
    Ok((spv, reflection))
}

/// Emit + passes transform for a sanitized AIR module, stopping BEFORE spirv-val and the retry
/// cascade — the byte-drift-check (BC) gate's translate path. Returns the canonical SPIR-V bytes the
/// default path would hand to spirv-val on success (`Ok`), or the emit/passes error (`Err`). No
/// subprocess and no retry tiers: this is exactly the "primary emit path + passes layer" BC
/// regression-guards, and it is byte-identical to the pre-spirv-val stage of
/// [`translate_sanitized_native`] (the same async-copy lowering, meta parse, `emit_vulkan_spirv`, and
/// `finish_module` the default path runs before it validates). The retry cascade only fires when this
/// output fails spirv-val, so BC deliberately does NOT see it (retry output needs spirv-val to gate,
/// which the milestone battery's G4/G5 cover).
pub fn translate_native_no_retry(san_ll: &str, stage: passes::Stage) -> Result<Vec<u8>, String> {
    // Mirror the pre-spirv-val prologue of `translate_sanitized_native_with_options` exactly so BC
    // measures the bytes production would actually validate: async-copy lowering, then stage meta.
    let lowered = lower_async_copy_if_enabled(san_ll);
    reject_unsupported_metal_linked_functions(&lowered)?;
    let stage_meta = parse_stage_meta(&lowered, stage);
    translate_native_no_retry_with_meta(
        &lowered,
        stage,
        stage_meta.frag.as_ref(),
        stage_meta.vert.as_ref(),
        stage_meta.kern.as_ref(),
        stage_meta.entry_name.as_deref(),
    )
}

/// `translate_native_no_retry` after its shared AIR pre-lowering and one stage-meta parse. Keeping
/// this separate lets the validation-gated primary wrapper reuse both the exact lowered text and the
/// parsed carrier for its default candidate and retry context.
fn translate_native_no_retry_with_meta(
    san_ll: &str,
    stage: passes::Stage,
    frag: Option<&meta::FragMeta>,
    vert: Option<&meta::VertMeta>,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
) -> Result<Vec<u8>, String> {
    // `tools::emit_vulkan_spirv` is the in-process native emitter (no subprocess); `finish_module` is
    // the shared passes tail every translate path runs. Together they are the default path's output
    // up to (but not including) `spirv_val_bytes` and the tier cascade.
    emit_finish_primary_module(
        san_ll,
        stage,
        frag,
        vert,
        kern,
        entry_name,
        passes::TransformOptions::default(),
    )
    .map(|finished| finished.bytes)
}

/// Return the default-path-aligned PRIMARY candidate without entering the retry cascade. This is
/// intentionally separate from [`translate_native_no_retry`]: the byte baseline must stay an
/// in-process, pre-validator gate, while a cross-binding pointer phi can only be adopted after
/// spirv-val proves that it is the actual failing rule. Callers that measure the primary-validity
/// floor use this form with a per-worker temporary directory.
pub fn translate_native_primary_validated(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
) -> Result<Vec<u8>, String> {
    let lowered = lower_async_copy_if_enabled(san_ll);
    let san_ll = lowered.as_str();
    reject_unsupported_metal_linked_functions(san_ll)?;
    let stage_meta = parse_stage_meta(san_ll, stage);
    let finished = emit_finish_primary_module(
        san_ll,
        stage,
        stage_meta.frag.as_ref(),
        stage_meta.vert.as_ref(),
        stage_meta.kern.as_ref(),
        stage_meta.entry_name.as_deref(),
        passes::TransformOptions::default(),
    )?;
    let primary_module = finished.module;
    let primary = finished.bytes;
    let Err(validation_error) = tools::spirv_val_bytes(&primary, tmp) else {
        return Ok(primary);
    };
    // The corpus audit calls this primary-only gate before the full retry cascade. Make its existing
    // default-off dump control capture the actual first-invalid module, so a later retry error cannot
    // hide where validation first failed. Diagnostics must never affect translation if the requested
    // dump path itself is unavailable.
    if let Some(path) = env_vars::retry_dump() {
        let _ = std::fs::write(path, &primary);
    }
    if let Some(primary) = primary_xbind_phi_psb_for_validation_error(
        &primary_module,
        &primary,
        &validation_error,
        tmp,
        reflect::DescriptorLayout::default(),
    ) {
        return Ok(primary);
    }
    let retry = retry::RetryCtx::new(
        san_ll,
        stage,
        stage_meta.frag.as_ref(),
        stage_meta.vert.as_ref(),
        stage_meta.kern.as_ref(),
        stage_meta.promoted_kern.as_ref(),
        stage_meta.entry_name.as_deref(),
        tmp,
        passes::TransformOptions::default(),
        true,
        layout::AirDataLayout::from_ir(san_ll)?,
    );
    if let Some(primary) = primary_storage_buffer_raw_feed_if_needed(&validation_error, &retry) {
        return Ok(primary);
    }
    Ok(
        primary_primitive_phi_metadata_if_needed(&validation_error, &retry)
            .or_else(|| primary_pointer_typing_raw_if_needed(&validation_error, &retry))
            .or_else(|| {
                primary_pointer_typing_wide_raw_guard_chain_if_needed(
                    &validation_error,
                    &primary,
                    &primary_module,
                    &retry,
                )
            })
            .or_else(|| primary_other_prune_if_needed(&validation_error, &primary, tmp))
            .or_else(|| {
                primary_wide_raw_store_guard_cfg_chain_if_needed(
                    &validation_error,
                    &primary,
                    &primary_module,
                    &retry,
                )
            })
            .or_else(|| primary_construct_tree_if_needed(&validation_error, &retry))
            // Plain relooper before prune-then-relooper: the latter deliberately declines when the ordinary
            // relooper already validates (so the full retry cascade keeps the `val-cfg:relooper` tier label).
            // Primary-validated has no later relooper step, so adopt it here for the residual CFG class
            // production already ships via relooper (notably banked `09/de938adc`).
            .or_else(|| primary_relooper_if_needed(&validation_error, &primary, tmp))
            .or_else(|| primary_cfg_prune_then_relooper_if_needed(&validation_error, &primary, tmp))
            // When the primary bytes are too broken for an in-place relooper rewrite (e.g. undefined-id SSA
            // order after an over-admitted structured emit), re-emit through the production raw→relooper→PSB
            // composition so primary-validated tracks the shipped cascade for frontier cross-arm residuals.
            .or_else(|| {
                let original = finished
                    .original_module
                    .as_ref()
                    .map(assemble_finished_module)
                    .unwrap_or_else(|| primary.clone());
                primary_cfg_raw_then_relooper_if_needed(&validation_error, &original, &retry)
            })
            .or_else(|| {
                primary_dynamic_struct_index_inline_sroa_relooper_if_needed(
                    &validation_error,
                    &retry,
                )
            })
            .or_else(|| {
                primary_logical_pointer_inline_sroa_raw_if_needed(&validation_error, &retry)
            })
            .or_else(|| {
                primary_opaque_image_select_module_if_needed(
                    &validation_error,
                    &primary_module,
                    &primary,
                    tmp,
                )
            })
            .unwrap_or(primary),
    )
}

/// Run the interface+lowering passes on emitted SPIR-V bytes and assemble to a canonical byte
/// stream — the shared tail of every translate path (default and the R4 raw retry tiers). Extracted
/// so the probe path ([`translate_raw_tiers_probe`]) re-emits the raw tiers through the EXACT same
/// transform the retry uses, rather than a divergent approximation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FinishRewrites {
    Plain,
    Primary,
}

#[derive(Clone)]
struct FinishedModule {
    /// The same module represented by `bytes`, retained so validation-triggered rewrites can clone
    /// and mutate it without reparsing the serialization that `finish_module` just produced.
    module: Module,
    bytes: Vec<u8>,
    /// Canonical module before primary-only rewrites, retained for validation-triggered retry
    /// rewrites that must start before the speculative primary-only tail. Its bytes are assembled
    /// lazily only when primary validation fails, so a validating primary assembles once.
    original_module: Option<Module>,
}

#[cfg(test)]
thread_local! {
    static FINISH_ASSEMBLE_COUNT: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
}

fn assemble_finished_module(module: &Module) -> Vec<u8> {
    #[cfg(test)]
    FINISH_ASSEMBLE_COUNT.with(|count| count.set(count.get() + 1));
    module
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
}

#[cfg(test)]
fn reset_finish_assemble_count() {
    FINISH_ASSEMBLE_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
fn finish_assemble_count() -> usize {
    FINISH_ASSEMBLE_COUNT.with(std::cell::Cell::get)
}

fn finish_module(
    mut emitted: emit_sidecar::EmittedSpirv,
    stage: passes::Stage,
    frag: Option<&meta::FragMeta>,
    vert: Option<&meta::VertMeta>,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    air_data_layout: Option<&layout::AirDataLayout>,
    options: passes::TransformOptions,
    rewrites: FinishRewrites,
) -> Result<FinishedModule, String> {
    emitted.sidecar.air_data_layout = air_data_layout.cloned();
    let retry_debug = env_vars::retry_debug();
    if retry_debug {
        for mapping in &emitted.sidecar.air_struct_layout_mappings {
            match mapping.status {
                emit_sidecar::AirStructLayoutMappingStatus::MappedNatural => {}
                emit_sidecar::AirStructLayoutMappingStatus::MappedExplicit => eprintln!(
                    "[retry-debug] AIR struct layout param={} type={:?}: exact metadata differs from natural layout; using exact offsets",
                    mapping.param_index, mapping.struct_ty
                ),
                status => eprintln!(
                    "[retry-debug] AIR struct layout param={} type={:?}: unmapped ({status:?}); using datalayout-derived natural layout",
                    mapping.param_index, mapping.struct_ty
                ),
            }
        }
        eprintln!("[retry-debug] finish: passes start");
    }
    let (mut out, mut sidecar) = passes::transform_with_options_and_sidecar(
        emitted.module,
        emitted.sidecar,
        stage,
        frag,
        vert,
        kern,
        entry_name,
        options,
    )?;
    if retry_debug {
        eprintln!("[retry-debug] finish: passes complete; native rewrites start");
    }
    // M4: legalize illegal logical-pointer phis (Private/Function/UniformConstant `OpPhi`) on the
    // PRIMARY emit path via the phi-the-index rewrite, so those functions validate directly instead of
    // shipping only via retry rescue. Floor-safe by construction (it touches only already-invalid
    // logical-pointer phis, never a validating module).
    let mut native_pointer_changed = native::rewrite_logical_pointer_phis_module(&mut out);
    // Opaque AIR pointers can select a real StorageBuffer byte view against a typed Private fallback.
    // Such a pointer select is invalid under Logical addressing regardless of VariablePointers.
    // Load each concrete arm in its own storage domain and select values; pointer escapes remain an
    // honest failure.
    while native::rewrite_mixed_storage_pointer_select_loads_module(&mut out) {
        native_pointer_changed = true;
    }
    // M4: legalize integer-width phi mismatches (a narrow `uint` loop phi fed a wide `ulong` induction
    // back-edge value) by truncating the wide incoming to the phi's result type. Floor-safe (touches
    // only already-invalid integer phis). Runs after the pointer-phi rewrite so it can also legalize
    // any width mismatch that rewrite's synthesized index phis might introduce.
    native::rewrite_integer_width_phis_module(&mut out);
    // M4/Keystone-2: repair loop-closed-SSA violations the `MultipleExits` loop funnel leaves behind (a
    // loop-body value used raw in a post-loop block the restructured CFG can reach without the defining
    // block), by register-demoting the offending value to a function-scope OpVariable. Floor-safe by
    // construction — a validating module has every def dominating its uses, so this never fires on one.
    native_pointer_changed |= native::demote_nondominating_values_module(&mut out);
    // Workgroup struct-padding clears emitted as byte-addressed `uchar*` stores are not legal
    // Logical SPIR-V when rooted at a struct pointer. Drop the provably-padding zero stores before
    // validation so the normal retry cascade only sees semantic failures.
    native::drop_workgroup_struct_padding_byte_zero_stores_module(&mut out);
    native::drop_dangling_debug_targets_module(&mut out);
    // M4/Keystone-2: node-split a multi-entry loop whose header is entered from two different
    // selections' arms (the irreducible shape `structured_plan` over-admits — the mlx-steel
    // `steel_attention` family), so the PRIMARY structured emit validates instead of shipping only via
    // the relooper retry. Floor-safe by construction (a valid loop is single-entry). Runs after the phi
    // rewrites; the attention rows also need the integer-width phi legalization above.
    let native_cfg_changed = native::split_multientry_loop_selection_exits_module(&mut out);
    if native_pointer_changed {
        passes::repair_exact_raw_byte_loads_after_native_rewrites(&mut out, &sidecar, entry_name)?;
    }
    if retry_debug {
        eprintln!("[retry-debug] finish: native rewrites complete; cfg repair start");
    }
    // The complete-module rewrites above run after the main passes pipeline and can introduce or
    // redirect CFG edges, or synthesize index phis from pointer phis. Re-establish the shared
    // merge/continue/phi invariant before ids are canonicalized so late native composition cannot
    // bypass structured-CFG repair.
    if native_cfg_changed || native_pointer_changed {
        out = passes::repair_structured_cfg_after_native_rewrites(out, stage, entry_name)?;
    }
    // CFG repair can introduce a forwarding phi after the earlier loop-closed-SSA pass. Re-check the
    // completed graph at this mutation boundary; the repair is self-gating and touches only a value
    // whose definition does not dominate its use.
    if native::demote_nondominating_values_module(&mut out) {
        out = passes::repair_structured_cfg_after_native_rewrites(out, stage, entry_name)?;
    }
    // CFG repair can split an edge and leave a forwarding phi with exactly one incoming pair. Fold
    // that SSA identity before validation and canonicalization. Besides avoiding redundant phis,
    // this prevents a stale pre-interface pointer result type from wrapping a refined image value.
    native::collapse_single_incoming_phis_module(&mut out);
    while native::rewrite_mixed_storage_pointer_select_loads_module(&mut out) {}
    native::rewrite_private_vector_word_loads_module(&mut out);
    native::repair_sampled_image_result_types_module(&mut out);
    // Complete-module pointer rewrites and their exact-raw repair can rematerialize access chains
    // after the main transform's null cleanup. Re-close the invariant after the final producer.
    passes::neutralize_null_access_chains_after_native_rewrites(&mut out, entry_name)?;
    if retry_debug {
        eprintln!("[retry-debug] finish: cfg repair complete; canonicalize start");
    }
    // Renumber all ids into a deterministic, serialized-order canonical form. This format
    // normalization keeps equivalent producer paths directly comparable and SPIR-V-level diffs
    // meaningful.
    let mut retained_global_ids = sidecar
        .local_pointer_field_stores
        .iter()
        .map(|fact| fact.id)
        .collect::<Vec<_>>();
    passes::canonicalize_ids_and_remap_sidecar(&mut out, &mut retained_global_ids, &mut sidecar);
    if retry_debug {
        eprintln!("[retry-debug] finish: canonicalize complete");
    }
    let original_module = (rewrites == FinishRewrites::Primary).then(|| out.clone());
    if rewrites == FinishRewrites::Primary {
        let (primary_pointer_changed, primary_cfg_changed) =
            primary_retry::apply_primary_emit_rewrites_module(
                &mut out,
                &mut retained_global_ids,
                &mut sidecar,
            );
        if primary_pointer_changed {
            passes::repair_exact_raw_byte_loads_after_native_rewrites(
                &mut out, &sidecar, entry_name,
            )?;
        }
        // Primary-only rewrites deliberately run after canonical id remapping and can perform their
        // own CFG surgery. They are the final mutation boundary, so close the same structured-CFG
        // invariant here as well instead of assuming the earlier finish-time repair still applies.
        if primary_cfg_changed {
            out = passes::repair_structured_cfg_after_native_rewrites(out, stage, entry_name)?;
        }
        if native::demote_nondominating_values_module(&mut out) {
            out = passes::repair_structured_cfg_after_native_rewrites(out, stage, entry_name)?;
        }
        native::collapse_single_incoming_phis_module(&mut out);
        while native::rewrite_mixed_storage_pointer_select_loads_module(&mut out) {}
        native::rewrite_private_vector_word_loads_module(&mut out);
        native::repair_sampled_image_result_types_module(&mut out);
        passes::neutralize_null_access_chains_after_native_rewrites(&mut out, entry_name)?;
        native::drop_unused_values_module(&mut out);
    }
    // Every late producer above may substitute a pointer carrier after its users were typed. Close
    // the SPIR-V access-chain contract at the final module boundary: the result pointer keeps its
    // pointee but always inherits the actual base pointer's storage class.
    native::reconcile_access_chain_storage_classes_module(&mut out);
    passes::validate_descriptor_bindings(&out, options.descriptor_layout)?;
    let bytes = assemble_finished_module(&out);
    if retry_debug {
        eprintln!("[retry-debug] finish: assembly complete");
    }
    Ok(FinishedModule {
        module: out,
        bytes,
        original_module,
    })
}

/// Probe-only (used by `historical raw-residual probes`): emit the two R4 raw-retry tiers
/// for a module and return each tier's emitted+assembled SPIR-V bytes (`Ok`) or its emit error
/// (`Err`), through the EXACT same `finish_module` transform the production retry applies. The caller
/// spirv-vals each to measure the TRUE residual blocker once raw byte-offset modeling resolves the
/// buffer pointer typing: a pointer-merge-classified case whose raw tier emits clean but is then
/// rejected for a CFG/other reason is MULTIPLY-BLOCKED (R4 + R2/...), not a pure R4 case. Returns
/// `[tier1, tier2]` where tier1 = device/constant buffers raw, tier2 = + threadgroup buffers raw.
pub fn translate_raw_tiers_probe(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
) -> Vec<Result<Vec<u8>, String>> {
    let lowered = lower_async_copy_if_enabled(san_ll);
    let san_ll = lowered.as_str();
    if let Err(error) = reject_unsupported_metal_linked_functions(san_ll) {
        return vec![Err(error.clone()), Err(error)];
    }
    let stage_meta = parse_stage_meta(san_ll, stage);
    let opts = passes::TransformOptions::default();
    let air_data_layout = layout::AirDataLayout::from_ir(san_ll);
    let run = |emitted: Result<emit_sidecar::EmittedSpirv, String>| -> Result<Vec<u8>, String> {
        let air_data_layout = air_data_layout.as_ref().map_err(Clone::clone)?;
        emitted.and_then(|b| {
            finish_module(
                b,
                stage,
                stage_meta.frag.as_ref(),
                stage_meta.vert.as_ref(),
                stage_meta.kern.as_ref(),
                stage_meta.entry_name.as_deref(),
                air_data_layout.as_ref(),
                opts,
                FinishRewrites::Plain,
            )
            .map(|finished| finished.bytes)
        })
    };
    vec![
        run(tools::emit_vulkan_spirv_all_buffers_raw_with_sidecar(
            san_ll,
            tmp,
            stage_meta.kern.as_ref(),
            stage_meta.entry_name.as_deref(),
            stage_buffer_layouts(
                stage,
                stage_meta.frag.as_ref(),
                stage_meta.vert.as_ref(),
                stage_meta.kern.as_ref(),
            ),
        )),
        run(
            tools::emit_vulkan_spirv_all_buffers_raw_with_workgroup_sidecar(
                san_ll,
                tmp,
                stage_meta.kern.as_ref(),
                stage_meta.entry_name.as_deref(),
                stage_buffer_layouts(
                    stage,
                    stage_meta.frag.as_ref(),
                    stage_meta.vert.as_ref(),
                    stage_meta.kern.as_ref(),
                ),
            ),
        ),
    ]
}

/// Diagnostic probe: emit the BDA device-pointer tier (`emit_vulkan_spirv_all_buffers_raw_bda`) for a
/// case and run it through the same `finish_module` finalization the production `bda_retry` applies, so
/// the surviving spirv-val residual of a `raw store for Ptr(1)` BDA case can be inspected. Mirrors
/// [`translate_raw_tiers_probe`]; not a frontier signal.
pub fn translate_bda_probe(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
) -> Result<Vec<u8>, String> {
    let lowered = lower_async_copy_if_enabled(san_ll);
    let san_ll = lowered.as_str();
    reject_unsupported_metal_linked_functions(san_ll)?;
    let stage_meta = parse_stage_meta(san_ll, stage);
    let opts = passes::TransformOptions::default();
    let air_data_layout = layout::AirDataLayout::from_ir(san_ll)?;
    tools::emit_vulkan_spirv_all_buffers_raw_bda_with_sidecar(
        san_ll,
        tmp,
        stage_meta.kern.as_ref(),
        stage_meta.entry_name.as_deref(),
        stage_buffer_layouts(
            stage,
            stage_meta.frag.as_ref(),
            stage_meta.vert.as_ref(),
            stage_meta.kern.as_ref(),
        ),
    )
    .and_then(|b| {
        finish_module(
            b,
            stage,
            stage_meta.frag.as_ref(),
            stage_meta.vert.as_ref(),
            stage_meta.kern.as_ref(),
            stage_meta.entry_name.as_deref(),
            air_data_layout.as_ref(),
            opts,
            FinishRewrites::Plain,
        )
        .map(|finished| finished.bytes)
    })
}

fn translate_sanitized_with_meta(
    san_ll: &str,
    stage: passes::Stage,
    frag: Option<&meta::FragMeta>,
    vert: Option<&meta::VertMeta>,
    kern: Option<&meta::KernMeta>,
    promoted_kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    tmp: &Path,
    options: passes::TransformOptions,
    psb_retry_enabled: bool,
    datalayout: Option<layout::AirDataLayout>,
) -> Result<Vec<u8>, String> {
    // Build the retry cascade context (S2): the former per-tier closures are now methods on
    // `RetryCtx`, which captures the same locals (`san_ll`, `tmp`, the meta/stage/options `finish`
    // threads, and the A/B env gates read once here). `psb_retry_enabled` is always `true` on the
    // production path; only `translate_pre_psb_probe` passes `false`. Routing stays below.
    let rc = retry::RetryCtx::new(
        san_ll,
        stage,
        frag,
        vert,
        kern,
        promoted_kern,
        entry_name,
        tmp,
        options,
        psb_retry_enabled,
        datalayout,
    );
    // Within the relooper's hard cap, a large source CFG is owned by the global structurizers.
    // Above that cap, speculative construct-tree pre-routing can consume the entire translation
    // budget before a valid primary is even attempted; try the primary first and let validation
    // select later retries. Count structural LLVM blocks without parsing instruction bodies.
    let max_function_blocks = sanitized_max_function_basic_block_count(san_ll);
    if large_cfg_relooper_eligible(max_function_blocks) {
        if let Some(relooped) = rc.raw_feed_then_relooper() {
            return Ok(relooped);
        }
        if rc.large_cfg_construct_tree_eligible() {
            if let Some(constructed) = rc.construct_tree_retry() {
                return Ok(constructed);
            }
        } else if rc.retry_debug_on {
            eprintln!("[retry-debug] large cfg: construct-tree skipped after non-CFG feed failure");
        }
    }
    let retry_debug_on = rc.retry_debug_on;
    let has_device_address_pointer = has_device_address_pointer_load(san_ll);
    let primary_finished = tools::emit_vulkan_spirv_with_sidecar(
        san_ll,
        tmp,
        rc.kern,
        rc.entry_name,
        stage_buffer_layouts(rc.stage, rc.frag, rc.vert, rc.kern),
    )
    .and_then(|emitted| rc.finish_primary_carrier(emitted));
    let translated = match primary_finished {
        Ok(finished) => {
            // `finish_module` applies the in-process primary rewrites before assembly. Try the
            // phi-only PSB candidate only if spirv-val identifies its cross-binding pointer rule.
            // On any non-validating outcome the canonical pre-rewrite module is serialized lazily
            // to drive the retry cascade unchanged.
            let primary_module = finished.module;
            let primary = finished.bytes;
            let out_module = finished
                .original_module
                .expect("primary finish always retains canonical pre-rewrite module");
            let primary_validation = tools::spirv_val_bytes(&primary, tmp);
            if primary_validation.is_ok() {
                if retry_debug_on {
                    eprintln!("[retry-debug] default emission validated in-translate");
                }
                Ok(primary)
            } else {
                let out = assemble_finished_module(&out_module);
                let validation_error = primary_validation.expect_err("checked above");
                if retry_debug_on {
                    eprintln!(
                        "[retry-debug] primary module spirv-val failed: {}",
                        validation_error
                            .lines()
                            .take(4)
                            .collect::<Vec<_>>()
                            .join(" | ")
                    );
                }
                let primary_block_count = primary_module
                    .functions
                    .iter()
                    .map(|function| function.blocks.len())
                    .max()
                    .unwrap_or(0);
                // The whole-function relooper cannot own an emitted graph beyond its hard cap, but
                // static function-constant arms can be the only reason it grew past that ceiling.
                // First prune and reloop the already-built primary; only if it cannot shrink into
                // the hard budget, give the compact source CFG one construct-tree attempt. Raw and
                // SROA re-emissions remain skipped for this oversized case.
                if primary_block_count > native::CFG_EMIT_RELOOPER_MAX_BLOCKS {
                    primary_cfg_prune_then_relooper_if_needed(&validation_error, &primary, tmp)
                        .or_else(|| primary_construct_tree_if_needed(&validation_error, &rc))
                        .ok_or(validation_error)
                } else if let Some(primary) = primary_xbind_phi_psb_for_validation_error(
                    &primary_module,
                    &primary,
                    &validation_error,
                    tmp,
                    options.descriptor_layout,
                ) {
                    Ok(primary)
                } else if let Some(primary) =
                    primary_storage_buffer_raw_feed_if_needed(&validation_error, &rc)
                {
                    Ok(primary)
                } else if let Some(primary) =
                    primary_primitive_phi_metadata_if_needed(&validation_error, &rc)
                {
                    Ok(primary)
                } else if let Some(primary) =
                    primary_pointer_typing_raw_if_needed(&validation_error, &rc)
                {
                    Ok(primary)
                } else if let Some(primary) = primary_pointer_typing_wide_raw_guard_chain_if_needed(
                    &validation_error,
                    &primary,
                    &primary_module,
                    &rc,
                ) {
                    Ok(primary)
                } else if let Some(primary) =
                    primary_other_prune_if_needed(&validation_error, &primary, tmp)
                {
                    Ok(primary)
                } else if let Some(primary) = primary_wide_raw_store_guard_cfg_chain_if_needed(
                    &validation_error,
                    &primary,
                    &primary_module,
                    &rc,
                ) {
                    Ok(primary)
                } else if let Some(primary) =
                    primary_cfg_prune_if_needed(&validation_error, &primary, &rc)
                {
                    Ok(primary)
                } else if let Some(primary) =
                    primary_construct_tree_if_needed(&validation_error, &rc)
                {
                    Ok(primary)
                } else if let Some(primary) =
                    primary_cfg_prune_then_relooper_if_needed(&validation_error, &primary, tmp)
                {
                    Ok(primary)
                } else if let Some(primary) =
                    primary_dynamic_struct_index_inline_sroa_relooper_if_needed(
                        &validation_error,
                        &rc,
                    )
                {
                    Ok(primary)
                } else if let Some(primary) =
                    primary_logical_pointer_inline_sroa_raw_if_needed(&validation_error, &rc)
                {
                    Ok(primary)
                } else if let Some(primary) = primary_opaque_image_select_module_if_needed(
                    &validation_error,
                    &primary_module,
                    &primary,
                    tmp,
                ) {
                    Ok(primary)
                } else {
                    match tools::spirv_val_bytes(&out, tmp) {
                        Ok(()) => {
                            if retry_debug_on {
                                eprintln!("[retry-debug] default emission validated in-translate");
                            }
                            // Variable-pointer-phi portability normalization is applied uniformly to every
                            // returned module at the end of this function, so the success path just surfaces the
                            // validated bytes.
                            Ok(out)
                        }
                        // The produced module is structurally valid but mistyped — retry raw, keep the default
                        // module's bytes if raw does not validate.
                        Err(e)
                            if native::classify_validation_error(&e)
                                == native::ValidationClass::PointerTyping =>
                        {
                            let fc_promote_first =
                                validation_error_may_be_fc_private_placeholder(&e)
                                    .then(|| {
                                        rc.census(
                                            "val-ptr:fc_promote_logical",
                                            rc.fc_promote_logical(),
                                        )
                                    })
                                    .flatten();
                            Ok(fc_promote_first
                                // A descriptor-root/device-address phi reports as pointer typing on
                                // the Logical primary. Prefer the address-domain BDA emitter before
                                // the generic raw/CFG ladders, but only when LLVM `inttoptr` proves
                                // the physical-address mechanism is actually present.
                                .or_else(|| {
                                    has_device_address_pointer
                                        .then(|| rc.census("val-ptr:bda", rc.bda_retry()))
                                        .flatten()
                                })
                                .or_else(|| rc.census("val-ptr:raw_retry", rc.raw_retry()))
                                .or_else(|| rc.census("val-ptr:prune", rc.prune_retry(&out)))
                                // A Function scalar-integer alloca reinterpreted as smaller-element lanes (the
                                // `as_type<uint>(half2)` pack idiom) emits an illegal scalar-indexing access chain
                                // ("reached non-composite"); retype the variable to a `<N x elem>` vector.
                                .or_else(|| {
                                    rc.census(
                                        "val-ptr:subword_pack",
                                        rc.subword_pack_retry(&out_module),
                                    )
                                })
                                // Pruning a function-constant-dead arm can leave the surviving CFG unstructured
                                // (a switch/selection whose removed case breaks the construct nesting); restructure
                                // the pruned module with the relooper before giving up.
                                .or_else(|| {
                                    rc.census(
                                        "val-ptr:prune_then_relooper",
                                        rc.prune_then_relooper(&out),
                                    )
                                })
                                // The module is BOTH pointer-mistyped and carries an unstructured switch: the raw
                                // byte model fixes the typing but its emission still leaves the switch unstructured,
                                // so structure the raw bytes with the relooper. The raw emission may then still leave
                                // a cross-binding merge (the pointer mistyping was the surface symptom of a select
                                // among distinct buffers) — apply the PSB rewrite to it.
                                .or_else(|| {
                                    rc.raw_then_psb_chain(
                                        "val-ptr:raw_then_relooper",
                                        "val-ptr:raw_psb",
                                    )
                                })
                                // A local union alloca whose conflicting typed views live in a CALLEE emits a
                                // structurally-typed chain against the caller's byte-array storage (the DecodeETC2
                                // class: an over-indexed `OpInBoundsAccessChain %uint` on `[N x uchar]`). The text
                                // inliner collapses all views into one function, where the multi-view byte-array
                                // remodel + byte-reinterpret GEP lowering handles them. Adopt-if-validates.
                                .or_else(|| {
                                    rc.inline_sroa_chain(
                                        "val-ptr:inline_sroa",
                                        "val-ptr:inline_sroa_raw",
                                    )
                                })
                                // The TopK multi-destination shape can need BOTH pieces: inline+SROA+raw
                                // dissolves pointer-valued local aggregate fields, then the expanded helper
                                // exposes a merge-block overlap that needs the existing cross-arm restructure
                                // tier before validation can adopt it.
                                .or_else(|| {
                                    rc.census(
                                        "val-ptr:inline_sroa_raw_cfg_restructure",
                                        rc.inline_sroa_raw_cfg_restructure_retry(),
                                    )
                                })
                                // The FC-multiplexed cross-binding merge whose live buffers were Private-demoted to
                                // zero placeholders (a70fb990-class `ndArrayConvWinogradWeightsTransform`): the
                                // default emit surfaces as a pointer-typing wall (an over-indexed Private scalar
                                // placeholder / a cross-storage pointer merge), so it routes HERE rather than the
                                // cross-binding arm. Re-parse with the FC-gated buffers bound REAL, prune the dead
                                // dtype arms, reconcile the raw byte-0 arm to the merge pointee, and PSB-lower. Placed
                                // LAST so it only fires when every Logical tier above has declined — no
                                // currently-cleared case's adopted bytes change. Adopt-if-validates, byte-correct by
                                // construction (real buffers + offset-0 reconcile + byte-proven PSB).
                                .or_else(|| {
                                    rc.census("val-ptr:fc_promote_psb", rc.fc_promote_psb())
                                })
                                .unwrap_or(out))
                        }
                        // The repair fallback produced a module spirv-val rejects for a structured-CFG nesting
                        // violation — retry with the cross-arm restructure, then the general relooper on the
                        // emitted bytes. If every validation-gated recovery declines, return an honest
                        // translation error: known-invalid structured CFG bytes must never be shipped.
                        Err(e)
                            if native::classify_validation_error(&e)
                                == native::ValidationClass::CfgStructurization =>
                        {
                            {
                                if retry_debug_on {
                                    let head: Vec<&str> = e.lines().take(4).collect();
                                    eprintln!(
                                    "[retry-debug] default module failed cfg-class spirv-val: {}",
                                    head.join(" | ")
                                );
                                }
                                if let Some((blocks, reused_merges)) =
                                    native::systematic_reused_merges_beyond_relooper(&primary_module)
                                {
                                    if retry_debug_on {
                                        eprintln!(
                                            "[retry-debug] cfg fast fallback: function blocks={blocks} \
                                             exceeds relooper cap={} with {reused_merges} reused \
                                             merge targets",
                                            native::CFG_EMIT_RELOOPER_MAX_BLOCKS,
                                        );
                                    }
                                    if crate::env_vars::tier_census() {
                                        eprintln!("[tier-census] fallback");
                                    }
                                    return Err(e);
                                }
                                rc.census("val-cfg:construct_tree", rc.construct_tree_retry())
                                    .or_else(|| {
                                        rc.census("val-cfg:relooper", rc.relooper_retry(&out))
                                    })
                            }
                            .or_else(|| {
                                rc.census(
                                    "val-cfg:prune_then_relooper",
                                    rc.prune_then_relooper(&out),
                                )
                            })
                            // The relooper structures the CFG but its from-scratch rebuild can EXPOSE a
                            // cross-binding pointer merge (`Variable pointers must point into the same structure`
                            // on an `OpSelect %_ptr_StorageBuffer_*`) that the repaired form kept off the failing
                            // path. On the default (repaired) path that same select reaches spirv-val as a clean
                            // CrossBindingPointerMerge and `value_select_retry` lowers it; here it is stranded on
                            // the relooped bytes. Route the relooped module through the same value-domain (then PSB)
                            // lowering the cross-binding arm uses. Adopt-if-validates, floor-safe.
                            .or_else(|| {
                                rc.census(
                                    "val-cfg:relooper_then_value_select",
                                    rc.relooper_then_value_select(&out),
                                )
                            })
                            // The cfg violation can MASK a pointer-typing error (spirv-val reports one error per
                            // module): once the relooper structures the CFG, validation trips on the underlying
                            // mistyped access instead. The raw byte model dissolves the typing error and the
                            // relooper structures the raw bytes — same composite the pointer-typing arm uses.
                            .or_else(|| {
                                rc.census("val-cfg:raw_then_relooper", rc.raw_then_relooper())
                            })
                            // The cfg violation can also live in an internal helper whose pointer-parameter
                            // staging is what breaks downstream (e.g. an atomic CAS loop on a device pointer
                            // param, staged Private). Inlining the helper chain + SROA dissolves the staging and
                            // the collapsed body often structures cleanly. Adopt-if-validates, floor-safe.
                            .or_else(|| {
                                rc.inline_sroa_chain(
                                    "val-cfg:inline_sroa",
                                    "val-cfg:inline_sroa_raw",
                                )
                            })
                            // The straddle-loop-merge + cross-binding-phi cluster (05/MPSRNNBreakUpToOutputVecs):
                            // the relooper refuses it, `inline_sroa_raw+psb` leaves the CFG untouched (straddle
                            // reject → naive `infer_*` → invalid structured module), so neither clears it. Emit
                            // with the cross-arm restructure ENABLED (privatize + return-unify the rejected CFG),
                            // then PSB-rewrite the cross-binding pointer phi to 64-bit addresses. Adopt-if-
                            // validates, floor-safe.
                            .or_else(|| {
                                rc.census(
                                    "val-cfg:inline_sroa_raw_cfg_restructure",
                                    rc.inline_sroa_raw_cfg_restructure_retry(),
                                )
                            })
                            .ok_or(e)
                        }
                        // An illegal logical-pointer `OpPhi` (pointer-induction over an aggregate) — retry the
                        // phi-the-index rewrite, then fall back to FC-dead-arm pruning (the offending phi may
                        // live entirely in a statically-dead function-constant arm). Keep the default bytes if
                        // neither validates.
                        Err(e)
                            if native::classify_validation_error(&e)
                                == native::ValidationClass::LogicalPointerPhi =>
                        {
                            Ok(rc
                                .census("val-phi:phi_index", rc.phi_index_retry(&out_module))
                                .or_else(|| rc.census("val-phi:prune", rc.prune_retry(&out)))
                                .unwrap_or(out))
                        }
                        // A cross-binding pointer merge (OpSelect/OpPhi over distinct bindings) — retry the W1
                        // PhysicalStorageBuffer64 lowering, then fall back to FC-dead-arm pruning. Keep the
                        // default bytes if neither validates.
                        Err(e)
                            if native::classify_validation_error(&e)
                                == native::ValidationClass::CrossBindingPointerMerge =>
                        {
                            Ok(rc
                                .census("val-xbind:value_select", rc.value_select_retry(&out))
                                // The value-domain lowering (load-then-select) stays plain Logical StorageBuffer and
                                // is MoltenVK-runnable, so it is preferred over the PSB (buffer-device-address) form
                                // whenever it validates; PSB is the fallback for merges this bails on.
                                .or_else(|| rc.census("val-xbind:psb", rc.psb_retry(&out)))
                                // The default module may carry a buffer pointer bitcast PSB cannot model; the raw
                                // byte-model emission dissolves it, leaving a clean whole-buffer cross-binding select
                                // the PSB rewrite then lowers to physical addresses.
                                .or_else(|| rc.census("val-xbind:raw_psb", rc.raw_psb()))
                                .or_else(|| rc.census("val-xbind:prune", rc.prune_retry(&out)))
                                .unwrap_or(out))
                        }
                        // Any other validation error — last-resort FC-dead-arm pruning, then the Logical
                        // structured recovery tiers, and the relooper switch-dispatcher only as a final resort.
                        // All adopt-if-validates / floor-safe.
                        Err(_) => Ok(rc
                            .census("val-other:prune", rc.prune_retry(&out))
                            // A logical pointer type mismatch can be the surface form of a device-
                            // address hierarchy: one phi arm is a descriptor-rooted buffer and a
                            // backedge arm is produced by LLVM `inttoptr`. Only try the BDA emitter
                            // when that structural opcode is actually present; its address-domain phi
                            // lowering is adopted only after full SPIR-V validation.
                            .or_else(|| {
                                has_device_address_pointer
                                    .then(|| rc.census("val-other:bda", rc.bda_retry()))
                                    .flatten()
                            })
                            // A local union alloca whose conflicting typed views live in a CALLEE emits a
                            // structurally-typed chain against the caller's byte-array storage (the DecodeETC2
                            // class: `OpInBoundsAccessChain %uint` with indexes left over on `[N x uchar]`).
                            // The text inliner collapses all views into one function, where the byte-array
                            // remodel + byte-reinterpret GEP lowering handles them. Adopt-if-validates.
                            //
                            // Ordered BEFORE the relooper tiers below: inlining+SROA lets the collapsed body
                            // structure through the DEFAULT structurizer into genuine nested `OpLoopMerge` loops,
                            // whereas the relooper lowers the whole function to one `OpLoopMerge` + giant
                            // `OpSwitch` state machine that is spirv-val-VALID but SPIRV-Cross/MoltenVK
                            // miscompiles (nested loops routed through the shared outer merge → zero-trip). Both
                            // validate; the structured form is byte-portable and the relooper form is not, so a
                            // validating structured emission must be preferred. Floor-safe: a banked case
                            // validates on the DEFAULT emit and never enters this retry chain. Clears the 3
                            // `kern_tiled_da_gather_reduce` rows (default emit fails pointer-typing, lands here).
                            .or_else(|| {
                                rc.inline_sroa_chain(
                                    "val-other:inline_sroa",
                                    "val-other:inline_sroa_raw",
                                )
                            })
                            // Last-resort structurization (miscompile-prone, see above): the offending construct
                            // may live in a statically-dead arm (prune shrinks a huge function below the relooper
                            // cap), or the module may be both pointer-mistyped and carry an unstructured switch.
                            .or_else(|| {
                                rc.census(
                                    "val-other:prune_then_relooper",
                                    rc.prune_then_relooper(&out),
                                )
                            })
                            .or_else(|| {
                                rc.census("val-other:raw_then_relooper", rc.raw_then_relooper())
                            })
                            // The default error may mask a cross-binding pointer merge: prefer the Logical
                            // value-select lowering (MoltenVK/Metal-runnable) before any physical-address tier.
                            .or_else(|| {
                                rc.census("val-other:value_select", rc.value_select_retry(&out))
                            })
                            .unwrap_or(out)),
                    }
                }
            }
        }
        // A cross-storage select deliberately has no materializable Logical-SPIR-V pointer. If its
        // first opaque consumer is an internal-helper argument, erase that boundary first: typed
        // inlining carries the deferred select into the helper and its load/store handlers replay
        // the concrete arms in value space. This is both the semantically direct repair and far
        // cheaper than trying whole-module BDA/raw remodelling before the diagnosed mechanism.
        Err(emit_err)
            if native::classify_emit_error(&emit_err)
                == native::EmitErrorClass::DeferredPointerMaterialization =>
        {
            let selected_pointer = native::deferred_pointer_materialization_name(&emit_err)
                .expect("classified deferred pointer error carries its SSA value");
            rc.census(
                "emit-deferred-pointer:consumer-inline",
                rc.pointer_select_consumer_inline_retry(&selected_pointer),
            )
            .ok_or(emit_err)
        }
        // The default typed emission FAILED outright with a buffer/pointer-typing emit gap the raw
        // byte-offset model expresses (a reinterpret-load width mismatch or a missing pointer storage
        // class). Retry raw and adopt it only if it independently validates; the raw bytes may
        // themselves carry an unstructured switch, so also try raw-then-relooper before surfacing the
        // original emit error.
        Err(emit_err)
            if native::classify_emit_error(&emit_err) == native::EmitErrorClass::PointerTyping =>
        {
            rc.census("emit-ptr:raw_retry", rc.raw_retry())
                .or_else(|| rc.raw_then_psb_chain("emit-ptr:raw_then_relooper", "emit-ptr:raw_psb"))
                // A `missing pointer storage` gap can also be an Apple BVH builder that DEREFERENCES a
                // device pointer derived via `air.get_data_pointer_instance_acceleration_structure` without
                // ever STORING it (the `getMTLInstanceBounds` kernels) — the raw tiers don't model the
                // device address but the BDA passthrough does. Adopt-if-validates, so a `missing pointer
                // storage` case the BDA tier cannot emit (e.g. the TopK array-of-pointers) is untouched.
                .or_else(|| rc.census("emit-ptr:bda", rc.bda_retry()))
                .or_else(|| rc.census("emit-ptr:bda_then_relooper", rc.bda_then_relooper()))
                // The `missing pointer storage` can also be a Function-pointer store staged for a by-value
                // struct forwarded into an internal helper (the TopK MPS NDArray multi-destination class):
                // inline the helper chain + SROA the staging so the device pointer is used directly.
                // `inline_sroa` inlines the helper chain + SROAs the pointer staging; its `_raw`
                // escalation additionally models the surviving buffers raw so the TopK cases whose
                // lowered device-pointer array feeds typed byte-offset loads/stores emit legally.
                .or_else(|| {
                    rc.inline_sroa_chain("emit-ptr:inline_sroa", "emit-ptr:inline_sroa_raw")
                })
                .ok_or(emit_err)
        }
        // The default typed emission failed with an emit gap no classifier above matches — notably
        // `native emitter: raw store for Ptr(1) is not covered yet` (an Apple BVH builder storing/
        // dereferencing a device pointer), OR `unknown SSA value %N` from an N-way pointer multiplexer
        // over DISTINCT device buffers passed to an internal helper (the MPSmatrixEmbeddings class): the
        // typed emit defers each distinct-buffer pointer `select` into the `selected_pointers` side-table
        // WITHOUT a materialized id, and the call-arg consumption then hits `value_id` on an unmaterialized
        // select. `bda_retry` (kept FIRST so every case it already clears keeps its exact bytes) does not
        // model this. The all-buffers-raw re-emission DOES express it — the distinct-buffer multiplexer
        // becomes a whole-buffer cross-binding `OpSelect` over raw byte-array backings, which the relooper
        // structures and the PSB PhysicalStorageBuffer64 rewrite lowers to physical addresses
        // (`raw_then_relooper`'s relooped+psb sub-path). So mirror the pointer-typing emit arm's raw/
        // relooper/psb cascade here. Every tier is adopt-if-validates, so this is strictly additive
        // and floor-safe: a case that previously hard-failed can only become a validating pass, never the
        // reverse (a banked case validates on the default emit and never enters this arm). Clears the three
        // MPSmatrixEmbeddings chained-multiplexer frontier fails (b511a833/f8b6205d/37967b8c).
        Err(emit_err) => rc
            .census("emit-other:bda", rc.bda_retry())
            .or_else(|| rc.census("emit-other:bda_then_relooper", rc.bda_then_relooper()))
            .or_else(|| rc.raw_then_psb_chain("emit-other:raw_then_relooper", "emit-other:raw_psb"))
            .ok_or(emit_err),
    };
    // M-C1 tier-adoption census (env-gated, pure telemetry): report which cascade invocation site
    // (if any) produced the adopted bytes, so the regression run yields a per-site adoption histogram —
    // the named kill-list of never-firing tiers. `default` = the default emission validated or was
    // kept (no tier adopted); `fallback` = every tier declined and translation FALLBACKs.
    if crate::env_vars::tier_census() {
        let label = match &translated {
            Ok(_) => rc.adopted_tier().unwrap_or("default"),
            Err(_) => "fallback",
        };
        eprintln!("[tier-census] {label}");
    }
    // Final portability normalization on EVERY validating result, regardless of which arm produced
    // it. A `StorageBuffer`/`Workgroup` pointer `OpPhi` (a loop-carried pointer walked through one
    // buffer) is legal SPIR-V under VariablePointersStorageBuffer, so it never trips a retry — but
    // MoltenVK's SPIRV-Cross MSL backend cannot always express it, failing pipeline creation with
    // `cannot initialize a variable of type 'device float *' with an lvalue of type 'device float'`
    // (the MPSLSTM/MPSRNN class). Such a phi can survive not only the untouched success path but any
    // retry arm (e.g. value_select clears a cross-binding merge yet leaves an intra-buffer pointer
    // phi standing). A statically-dead arm can also be the only thing keeping a null/placeholder
    // pointer incoming alive, so run the existing constant-branch prune first when the module already
    // declares variable-pointer capability. The index-phi/pruned form is semantically identical and
    // needs no variable pointers, so adopt it whenever it independently validates; keep the arm's bytes
    // otherwise. Floor-safe by construction (adopt-if-validates).
    translated.map(|out| {
        let candidate = load_owned_module(&out).ok().and_then(|mut module| {
            let has_variable_pointer_capability = module.capabilities.iter().any(|inst| {
                matches!(
                    inst.operands.as_slice(),
                    [spirv_module::Operand::Capability(
                        spirv::Capability::VariablePointers
                            | spirv::Capability::VariablePointersStorageBuffer
                    )]
                )
            });
            let pruned = has_variable_pointer_capability
                && native::prune_constant_branches_module(&mut module).is_ok();
            let rewrote = native::rewrite_variable_pointer_phis_module(&mut module).is_ok();
            if !pruned && !rewrote {
                return None;
            }
            let bytes = module
                .assemble()
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>();
            rc.validates_dbg("variable_pointer_phi_normalization", &bytes)
                .then_some(bytes)
        });
        candidate.unwrap_or(out)
    })
}

fn has_device_address_pointer_load(san_ll: &str) -> bool {
    san_ll.lines().any(|line| {
        line.split_once('=').is_some_and(|(_, rhs)| {
            let rhs = rhs.trim_start();
            rhs.starts_with("inttoptr ")
                || (rhs.starts_with("load ptr addrspace(1)")
                    && rhs.split_once(',').is_some_and(|(_, pointer)| {
                        let pointer = pointer.trim_start();
                        pointer.starts_with("ptr addrspace(1) ")
                            || pointer.starts_with("ptr addrspace(2) ")
                    }))
        })
    })
}

fn sanitized_max_function_basic_block_count(san_ll: &str) -> usize {
    let mut current = None::<(usize, bool)>;
    let mut maximum = 0usize;
    let finish_count =
        |(labels, implicit): (usize, bool)| labels.saturating_add(usize::from(implicit)).max(1);
    for line in san_ll
        .lines()
        .filter_map(|line| line.split(';').next().map(str::trim))
    {
        if line.starts_with("define ") {
            if let Some(state) = current.replace((0, false)) {
                maximum = maximum.max(finish_count(state));
            }
        } else if line == "}" {
            if let Some(state) = current.take() {
                maximum = maximum.max(finish_count(state));
            }
        } else if line.ends_with(':') {
            if let Some((labels, _)) = current.as_mut() {
                *labels += 1;
            }
        } else if !line.is_empty() {
            if let Some((0, implicit)) = current.as_mut() {
                *implicit = true;
            }
        }
    }
    maximum.max(current.map(finish_count).unwrap_or_default())
}

fn large_cfg_relooper_eligible(max_function_blocks: usize) -> bool {
    max_function_blocks > native::LARGE_CFG_BLOCK_THRESHOLD
        && max_function_blocks <= native::CFG_EMIT_RELOOPER_MAX_BLOCKS
}

fn validation_error_may_be_fc_private_placeholder(error: &str) -> bool {
    error.contains("reached non-composite") && error.contains("_ptr_Private_")
}

fn validation_error_may_be_storage_buffer_overindex(error: &str) -> bool {
    error.contains("reached non-composite type while indexes still remain")
        && error.contains("_ptr_StorageBuffer_")
}

/// Canonicalize the ids of an emitted SPIR-V byte stream into the deterministic serialized-order form
/// the shipped pipeline applies in `finish_module`. Exposed for the byte-drift gates
/// (historical validation tooling byte-baseline-check` / `byte-determinism-check`), which compare native-emit
/// output in its canonical (shipped) form rather than the raw emission-order id numbering, which differs
/// benignly between processes and is normalized away here.
pub fn canonicalize_spirv_bytes(spv: &[u8]) -> Result<Vec<u8>, String> {
    let mut module = load_owned_module(spv).map_err(|e| format!("SPIR-V load: {e:?}"))?;
    passes::canonicalize_ids(&mut module);
    Ok(module
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect())
}

/// Disassemble SPIR-V bytes to spvasm text (for golden fixtures / debugging).
pub fn disassemble(spv: &[u8]) -> Result<String, String> {
    let m = load_owned_module(spv).map_err(|e| format!("SPIR-V load: {e:?}"))?;
    Ok(m.disassemble())
}

#[cfg(test)]
mod single_meta_parse_tests {
    use super::*;

    #[test]
    fn large_cfg_pretry_uses_the_largest_function_not_module_total() {
        let many_small = (0..400)
            .map(|index| {
                format!(
                    "define void @f{index}() {{\nentry:\n  br label %exit\nexit:\n  ret void\n}}\n"
                )
            })
            .collect::<String>();
        assert_eq!(sanitized_max_function_basic_block_count(&many_small), 2);

        let one_large = format!(
            "define void @large() {{\nentry:\n{}  ret void\n}}\n",
            (0..350)
                .map(|index| format!("b{index}:\n  br label %b{}\n", index + 1))
                .collect::<String>()
        );
        assert_eq!(sanitized_max_function_basic_block_count(&one_large), 351);
    }

    #[test]
    fn large_cfg_pretry_never_reloads_a_graph_beyond_the_relooper_cap() {
        assert!(!large_cfg_relooper_eligible(
            native::LARGE_CFG_BLOCK_THRESHOLD
        ));
        assert!(large_cfg_relooper_eligible(
            native::LARGE_CFG_BLOCK_THRESHOLD + 1
        ));
        assert!(large_cfg_relooper_eligible(
            native::CFG_EMIT_RELOOPER_MAX_BLOCKS
        ));
        assert!(!large_cfg_relooper_eligible(
            native::CFG_EMIT_RELOOPER_MAX_BLOCKS + 1
        ));
    }

    #[test]
    fn device_address_model_detects_loaded_device_pointers_not_local_pointer_staging() {
        assert!(has_device_address_pointer_load(
            "%p = load ptr addrspace(1), ptr addrspace(2) %field"
        ));
        assert!(has_device_address_pointer_load(
            "%p = load ptr addrspace(1), ptr addrspace(1) %field"
        ));
        assert!(has_device_address_pointer_load(
            "%p = inttoptr i64 %address to ptr addrspace(1)"
        ));
        assert!(!has_device_address_pointer_load(
            "%p = load ptr addrspace(1), ptr %local"
        ));
        assert!(!has_device_address_pointer_load(
            "%p = load ptr addrspace(2), ptr addrspace(2) %field"
        ));

        let ll = r#"
define void @k(ptr addrspace(1) %out, i64 %address) {
  %p = inttoptr i64 %address to ptr addrspace(1)
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1}
!4 = !{i32 1, !"air.thread_position_in_grid"}
"#;
        let reflection = reflect_sanitized(
            ll,
            passes::Stage::Kernel,
            passes::TransformOptions::default(),
        )
        .expect("reflect device-address kernel");
        let table = reflection
            .bindings
            .iter()
            .find(|binding| binding.kind == reflect::ResourceKind::BufferAddressTable)
            .expect("buffer-address table reflection");
        assert_eq!(
            table.descriptor.map(|descriptor| descriptor.binding),
            Some(reflect::SYNTHETIC_BINDING_BASE)
        );
    }

    #[test]
    fn fc_buffer_promotion_only_routes_private_placeholder_chains() {
        assert!(validation_error_may_be_fc_private_placeholder(
            "OpInBoundsAccessChain reached non-composite type; %x = OpInBoundsAccessChain %_ptr_Private_uint"
        ));
        assert!(!validation_error_may_be_fc_private_placeholder(
            "OpInBoundsAccessChain reached non-composite type; %x = OpInBoundsAccessChain %_ptr_StorageBuffer_uint"
        ));
    }

    #[test]
    fn raw_relooper_feed_only_routes_storage_buffer_overindex() {
        assert!(validation_error_may_be_storage_buffer_overindex(
            "OpInBoundsAccessChain reached non-composite type while indexes still remain: \
             %x = OpInBoundsAccessChain %_ptr_StorageBuffer_uint %root %index"
        ));
        assert!(!validation_error_may_be_storage_buffer_overindex(
            "OpInBoundsAccessChain reached non-composite type while indexes still remain: \
             %x = OpInBoundsAccessChain %_ptr_Private_uint %root %index"
        ));
    }

    const SIMPLE_KERNEL: &str = r#"
define void @k(ptr addrspace(1) %out) {
entry:
  store i32 7, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;

    #[test]
    fn production_kernel_emit_reuses_one_stage_meta_parse() {
        meta::reset_air_meta_parse_count();
        let stage_meta = parse_stage_meta(SIMPLE_KERNEL, passes::Stage::Kernel);
        tools::emit_vulkan_spirv_with_sidecar(
            SIMPLE_KERNEL,
            Path::new(""),
            stage_meta.kern.as_ref(),
            stage_meta.entry_name.as_deref(),
            stage_buffer_layouts(
                passes::Stage::Kernel,
                stage_meta.frag.as_ref(),
                stage_meta.vert.as_ref(),
                stage_meta.kern.as_ref(),
            ),
        )
        .expect("production emitter consumes threaded metadata");

        assert_eq!(meta::air_meta_parse_count(), 1);
    }

    #[test]
    fn validating_primary_finish_assembles_once() {
        let tmp =
            std::env::temp_dir().join(format!("metal2vulkan_finish_once_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        reset_finish_assemble_count();
        let spv = translate_sanitized_native(SIMPLE_KERNEL, passes::Stage::Kernel, &tmp)
            .expect("simple primary translation validates");
        tools::spirv_val_bytes(&spv, &tmp).expect("simple primary spirv-val");
        assert_eq!(
            finish_assemble_count(),
            1,
            "a validating primary must not assemble fallback bytes"
        );
    }
}
