//! metal2vulkan — Metal AIR (LLVM bitcode) -> Vulkan SPIR-V, via a native LLVM-IR emitter.
//!
//! The native emitter produces `OpCapability Shader` / `Logical GLSL450` SPIR-V directly from
//! sanitized AIR LLVM IR. Crate-owned retained-SPIR-V passes then build the Vulkan stage interface,
//! lower residual AIR operations, normalize memory access and control flow, and finalize the module
//! (see [`passes`]).
//!
//! Pipeline: `.air|.ll` -> llvm-dis -> sanitize -> native Vulkan SPIR-V emit -> retained crate
//! module -> interface+lowering passes -> assemble -> spirv-val (vulkan1.2).

// `too_many_arguments` and `type_complexity` are threshold heuristics that fire pervasively and
// benignly across this translator: emit/lowering functions legitimately thread many typed
// parameters (stage + metas + module + ctx + resolved ids), and the IR/emit return shapes are
// genuinely nested domain types (e.g. `Result<Option<Vec<(Vec<u32>, LlType)>>, String>`). Factoring
// every such signature behind a wrapper struct or a one-off `type` alias would add indirection
// without improving clarity, so both are accepted crate-wide rather than scattered per-site.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub mod air_intrinsics;
pub(crate) mod air_static_init;
pub mod as_shadow;
mod construction;
mod emission_order;
mod emit_sidecar;
pub mod env_vars;
mod fc_air_specialize;
mod fc_specialize;
pub(crate) mod float16;
mod layout;
pub mod linked_functions;
pub mod meta;
pub mod native;
pub mod passes;
mod passthrough;
pub mod reflect;
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
pub use passthrough::{
    translate_passthrough, translate_passthrough_specialized, translate_vertex_observer,
};

use crate::spirv_module::{load_bytes as load_owned_module, Module};
use std::borrow::Cow;
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
/// Construction selects a representation from AIR structure and owned-module invariants before
/// serialization. The single resulting module is then validated with `spirv-val` under the Vulkan
/// 1.2 environment; validator output never selects or repairs another representation. `tmp` is
/// caller-owned scratch space and may be reused sequentially, but callers should give concurrent
/// translations separate directories.
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

/// Apply the shared pre-emit AIR lowering before any representation derives stage metadata or emits
/// from the module. Alternate constructions re-emit the supplied text directly, so using the
/// original intrinsic-bearing text would make the representations observe different programs.
/// Floor-safe: `lower_simdgroup_async_copy` is a no-op unless the module calls
/// `air.simdgroup_async_copy_2d` (such modules fail the emitter outright otherwise).
fn lower_async_copy_if_enabled(san_ll: &str) -> Cow<'_, str> {
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
    if let Some(dispatch) = options.kernel_dispatch {
        dispatch.validate()?;
    }
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

/// Per-stage interface metadata parsed once from sanitized AIR and shared by emission, passes, and
/// reflection.
struct StageMeta {
    frag: Option<meta::FragMeta>,
    vert: Option<meta::VertMeta>,
    kern: Option<meta::KernMeta>,
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
                entry_name,
            }
        }
        passes::Stage::Vertex => {
            let (vert, entry_name) = meta::parse_air_vertex_meta_with_entry(san_ll);
            StageMeta {
                frag: None,
                vert,
                kern: None,
                entry_name,
            }
        }
        passes::Stage::Kernel => {
            let (kern, _, entry_name) = meta::parse_air_kernel_meta_variants(san_ll);
            StageMeta {
                frag: None,
                vert: None,
                kern,
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
    if let Some(dispatch) = options.kernel_dispatch {
        dispatch.validate()?;
    }
    if !matches!(stage, passes::Stage::Kernel) && options.kernel_dispatch.is_some() {
        return Err("kernel dispatch bounds are only valid for kernel stages".to_string());
    }
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
    if matches!(stage, passes::Stage::Kernel) {
        reflection.kernel_dispatch = Some(
            options
                .kernel_dispatch
                .unwrap_or_else(reflect::KernelDispatch::safe_default),
        );
    }
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

pub fn translate_sanitized_native_with_options(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
    options: passes::TransformOptions,
) -> Result<Vec<u8>, String> {
    if env_vars::retry_debug() {
        eprintln!("[retry-debug] translate: datalayout parse start");
    }
    let datalayout = layout::AirDataLayout::from_ir(san_ll)?;
    if env_vars::retry_debug() {
        eprintln!("[retry-debug] translate: datalayout parse complete");
    }
    translate_sanitized_native_with_options_and_layout(san_ll, stage, tmp, options, datalayout)
}

/// Translate sanitized AIR after baking exact Metal function-constant payloads into the AIR
/// initializer contract.
///
/// Specialization happens before metadata parsing, CFG construction, and resource-interface
/// lowering. This is required when a function constant controls resource presence or removes a CFG
/// arm: changing a finished SPIR-V initializer cannot faithfully restore structure already folded
/// under the default value. Each payload is exact little-endian scalar/vector storage for its AIR
/// `MTL_FC_INIT_<index>` type.
pub fn translate_sanitized_native_specialized_with_options(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
    options: passes::TransformOptions,
    function_constants: &[(u32, Vec<u8>)],
) -> Result<Vec<u8>, String> {
    let datalayout = layout::AirDataLayout::from_ir(san_ll)?;
    let specialized =
        fc_air_specialize::specialize_air_function_constants(san_ll, function_constants)?;
    translate_sanitized_native_with_options_and_layout(
        specialized.as_ref(),
        stage,
        tmp,
        options,
        datalayout,
    )
}

/// Translate an owned sanitized AIR module while allowing superseded preprocessing input to be
/// released before typed parsing. This is byte- and error-equivalent to
/// [`translate_sanitized_native_with_options`], but lowers peak memory for large modules that need a
/// source rewrite. Callers that must retain their source should use the borrowed API.
pub fn translate_sanitized_native_owned_with_options(
    san_ll: String,
    stage: passes::Stage,
    tmp: &Path,
    options: passes::TransformOptions,
) -> Result<Vec<u8>, String> {
    let datalayout = layout::AirDataLayout::from_ir(&san_ll)?;
    let lowered = native::lower_simdgroup_async_copy_owned(san_ll);
    translate_sanitized_native_pre_lowered_with_layout(&lowered, stage, tmp, options, datalayout)
}

fn translate_sanitized_native_with_options_and_layout(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
    options: passes::TransformOptions,
    datalayout: Option<layout::AirDataLayout>,
) -> Result<Vec<u8>, String> {
    // Lower `air.simdgroup_async_copy_2d` (+ its event/wait pair) to an explicit strided tile copy
    // before metadata parsing or emission, so the primary and alternate representations see the
    // same ordinary LLVM. The rewrite is a no-op unless the module calls the intrinsic, which the
    // emitter otherwise rejects. See `native::async_copy` and its structural regression tests.
    if env_vars::retry_debug() {
        eprintln!("[retry-debug] translate: AIR pre-lowering start");
    }
    let lowered = lower_async_copy_if_enabled(san_ll);
    let san_ll = lowered.as_ref();
    if env_vars::retry_debug() {
        eprintln!("[retry-debug] translate: AIR pre-lowering complete");
    }
    translate_sanitized_native_pre_lowered_with_layout(san_ll, stage, tmp, options, datalayout)
}

fn translate_sanitized_native_pre_lowered_with_layout(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
    options: passes::TransformOptions,
    datalayout: Option<layout::AirDataLayout>,
) -> Result<Vec<u8>, String> {
    reject_unsupported_metal_linked_functions(san_ll)?;
    if env_vars::retry_debug() {
        eprintln!("[retry-debug] translate: stage metadata parse start");
    }
    let stage_meta = parse_stage_meta(san_ll, stage);
    if env_vars::retry_debug() {
        eprintln!("[retry-debug] translate: stage metadata parse complete");
    }
    let options = options_for_air(san_ll, options)?;
    if env_vars::retry_debug() {
        eprintln!("[retry-debug] translate: construction core start");
    }
    translate_sanitized_with_meta(
        san_ll,
        stage,
        stage_meta.frag.as_ref(),
        stage_meta.vert.as_ref(),
        stage_meta.kern.as_ref(),
        stage_meta.entry_name.as_deref(),
        tmp,
        options,
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
    let specialized = specialize_linked_module(san_ll, stage, linkage)?;
    translate_sanitized_native_with_options(&specialized, stage, tmp, options)
}

/// Resolve exact direct references and function-table contents into one ordinary AIR module.
/// Consumers that need to inspect or diagnose the pre-emission linked program use the same
/// specialization contract as linked translation.
pub fn specialize_linked_module(
    san_ll: &str,
    stage: passes::Stage,
    linkage: &linked_functions::LinkedFunctionLinkage,
) -> Result<String, String> {
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
    Ok(specialized)
}

/// Linked translation with AIR-level function-constant specialization.
///
/// Linkage is resolved first so the same exact function-constant values specialize every retained
/// linked definition carrying the stable AIR initializer index.
pub fn translate_sanitized_native_linked_specialized_with_options(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
    options: passes::TransformOptions,
    linkage: &linked_functions::LinkedFunctionLinkage,
    function_constants: &[(u32, Vec<u8>)],
) -> Result<Vec<u8>, String> {
    let specialized = specialize_linked_module(san_ll, stage, linkage)?;
    translate_sanitized_native_specialized_with_options(
        &specialized,
        stage,
        tmp,
        options,
        function_constants,
    )
}

/// Like [`translate`] but also returns the [`reflect::ShaderReflection`] needed to integrate the
/// resulting module.
///
/// Interface facts come from AIR metadata and the translator's descriptor ABI. Conservative buffer
/// footprints come from read-only analysis of the final constructed and validated SPIR-V. The
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
/// to linked function definitions.
///
/// The returned shape is identical only in its fields, not in what fills them: two descriptor
/// classes are decided by the constructed module, and this path builds none.
///
/// Whether the emitter needs a buffer-address table is decided by the constructed pointer graph, so
/// this path asks the emitter's own predicate: it parses the source and runs
/// `requires_device_address_model`, the same question `emit_vulkan_spirv_with_sidecar` asks. That is
/// still not the finished module -- over 2880 corpus sources it disagrees with what the emitter
/// emits on 8, four in each direction. The AIR text scan it replaced disagreed on 63, and 13 of
/// those reported no table for a module that declares one.
/// The descriptors the passes synthesize to type an AIR value -- `SynthesizedNullTexture` and
/// `SynthesizedReadSampler` -- are absent entirely, because whether one survives depends on whether
/// anything in the finished module consumed its value.
///
/// Three per-binding fields are likewise the declaration rather than the module: a texture's
/// `texture_shape` is the shape its AIR type name implies, which is not always the image the
/// emitter binds; a buffer's `access` is the declared classification, not widened to the loads and
/// stores the module performs; and `footprint` is absent. `translate_sanitized_native_reflected`
/// has a module and reads all of these off it instead.
pub fn reflect_sanitized(
    san_ll: &str,
    stage: passes::Stage,
    options: passes::TransformOptions,
) -> Result<reflect::ShaderReflection, String> {
    let lowered = lower_async_copy_if_enabled(san_ll);
    let san_ll = lowered.as_ref();
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
    if stage == passes::Stage::Kernel
        && native::requires_device_address_model_for_source(
            san_ll,
            stage_meta.kern.as_ref(),
            stage_meta.entry_name.as_deref(),
        )
    {
        reflection.add_buffer_address_table()?;
    }
    reflection.validate_descriptor_abi()?;
    Ok(reflection)
}

/// Reflect sanitized AIR after baking exact Metal function-constant payloads into its resource
/// and control-flow contract.
///
/// Function constants can gate stage arguments in AIR metadata. Reflection must therefore consume
/// the same specialized AIR as translation; reflecting the disabled default and specializing only
/// during emission would omit resources selected by an authored non-default value.
pub fn reflect_sanitized_specialized(
    san_ll: &str,
    stage: passes::Stage,
    options: passes::TransformOptions,
    function_constants: &[(u32, Vec<u8>)],
) -> Result<reflect::ShaderReflection, String> {
    let specialized =
        fc_air_specialize::specialize_air_function_constants(san_ll, function_constants)?;
    reflect_sanitized(specialized.as_ref(), stage, options)
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
    let san_ll = lowered.as_ref();
    reject_unsupported_metal_linked_functions(san_ll)?;
    let stage_meta = parse_stage_meta(san_ll, stage);
    let options = options_for_air(san_ll, options)?;
    passes::validate_kernel_dispatch_options(stage, options)?;
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
    reflection.validate_descriptor_abi()?;
    let finished = translate_sanitized_with_meta_prevalidated_carrier(
        san_ll,
        stage,
        stage_meta.frag.as_ref(),
        stage_meta.vert.as_ref(),
        stage_meta.kern.as_ref(),
        stage_meta.entry_name.as_deref(),
        tmp,
        options,
        datalayout,
    )?;
    // The emitter decides whether the constructed pointer graph needs an address table; reflection
    // no longer predicts that from the AIR text when a module is in hand to be read.
    reflection.reconcile_buffer_address_table(&finished.module);
    // Nothing in the AIR metadata describes a descriptor the passes invented to type an AIR value,
    // so only the module and the pass that made it can say a consumer has to bind one.
    reflection.report_synthesized_placeholders(
        &finished.module,
        &finished.placeholder_descriptor_bindings,
    );
    // The image the emitter binds is not always the one the AIR type name implies -- a texel-read
    // cube binds as a 2D array -- and the view a consumer creates has to match the module.
    reflection.reconcile_texture_shapes(&finished.module);
    reflection.add_buffer_footprints(&finished.module)?;
    reflection.validate_descriptor_abi()?;
    Ok((finished.bytes, reflection))
}

/// Emit and finish the primary representation without invoking `spirv-val`. This is the byte-drift
/// boundary for the primary emitter: it includes the same AIR lowering, metadata parse, passes, and
/// owned-module construction checks used by production, but it does not select an alternate
/// representation when those checks reject the primary one.
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
/// this separate lets the validating primary wrapper reuse the exact lowered text and parsed
/// carrier.
fn translate_native_no_retry_with_meta(
    san_ll: &str,
    stage: passes::Stage,
    frag: Option<&meta::FragMeta>,
    vert: Option<&meta::VertMeta>,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
) -> Result<Vec<u8>, String> {
    // `tools::emit_vulkan_spirv` is the in-process native emitter (no subprocess); `finish_module` is
    // the shared passes tail every translate path runs. Together they are the primary construction
    // up to (but not including) `spirv_val_bytes`.
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

/// Emit and finish the primary construction without invoking the external validator. Shared by the
/// byte-drift boundary and the primary-validating facade so both observe the exact product module.
fn emit_finish_primary_module(
    san_ll: &str,
    stage: passes::Stage,
    frag: Option<&meta::FragMeta>,
    vert: Option<&meta::VertMeta>,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    options: passes::TransformOptions,
) -> Result<FinishedModule, String> {
    passes::validate_kernel_dispatch_options(stage, options)?;
    let air_data_layout = crate::layout::AirDataLayout::from_ir(san_ll)?;
    tools::emit_vulkan_spirv_with_sidecar(
        san_ll,
        Path::new(""),
        kern,
        entry_name,
        stage_buffer_layouts(stage, frag, vert, kern),
    )
    .and_then(|emitted| {
        finish_module(
            emitted,
            stage,
            frag,
            vert,
            kern,
            entry_name,
            air_data_layout.as_ref(),
            options,
            FinishConstruction::Primary,
        )
        .map_err(|failure| failure.error)
    })
}

/// Construct and validate only the primary representation. This is intentionally separate from
/// [`translate_native_no_retry`]: callers that measure primary validity use this form with a
/// per-worker temporary directory.
pub fn translate_native_primary_validated(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
) -> Result<Vec<u8>, String> {
    let lowered = lower_async_copy_if_enabled(san_ll);
    let san_ll = lowered.as_ref();
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
    if let Some(path) = env_vars::retry_dump() {
        let _ = std::fs::write(path, &finished.bytes);
    }
    tools::spirv_val_bytes(&finished.bytes, tmp)?;
    Ok(finished.bytes)
}
/// Run interface and lowering passes on an owned emitted module and assemble one canonical byte
/// stream. Primary and alternate representations share this construction boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FinishConstruction {
    Plain,
    Primary,
    RawRelooper,
}

#[derive(Clone)]
struct FinishedModule {
    /// The same module represented by `bytes`, retained for owned construction consumers.
    module: Module,
    bytes: Vec<u8>,
    /// Bindings of descriptors the passes synthesized with no Metal argument behind them, filtered
    /// to those this module still declares. Only the passes know these exist; reflection reports
    /// them from here so a consumer's descriptor-set layout covers them.
    placeholder_descriptor_bindings: Vec<u32>,
}

impl FinishedModule {
    /// Seal one finished owned module and its only serialized representation together.
    fn new(module: Module, placeholder_descriptor_bindings: Vec<u32>) -> Self {
        let bytes = assemble_finished_module(&module);
        Self {
            module,
            bytes,
            placeholder_descriptor_bindings,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FinishFailureKind {
    Other,
    RawBufferConstruction,
    CfgConstruction,
}

#[derive(Debug)]
struct FinishFailure {
    kind: FinishFailureKind,
    error: String,
}

impl FinishFailure {
    fn cfg(error: String) -> Self {
        Self {
            kind: FinishFailureKind::CfgConstruction,
            error,
        }
    }
}

impl From<String> for FinishFailure {
    fn from(error: String) -> Self {
        Self {
            kind: FinishFailureKind::Other,
            error,
        }
    }
}

impl From<native::OwnedModuleFailure> for FinishFailure {
    fn from(failure: native::OwnedModuleFailure) -> Self {
        match failure {
            native::OwnedModuleFailure::Invalid(error)
            | native::OwnedModuleFailure::TypeConstruction(error) => Self::from(error),
            native::OwnedModuleFailure::RawBufferConstruction(error) => Self {
                kind: FinishFailureKind::RawBufferConstruction,
                error,
            },
            native::OwnedModuleFailure::CfgConstruction(error) => Self::cfg(error),
        }
    }
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
    construction: FinishConstruction,
) -> Result<FinishedModule, FinishFailure> {
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
                // Not a disagreement: a byte-addressed buffer is emitted as its raw contents, so
                // there are no members for the declared offsets to land on in the first place.
                emit_sidecar::AirStructLayoutMappingStatus::EmittedIsUntypedBuffer => eprintln!(
                    "[retry-debug] AIR struct layout param={} type={:?}: emitted as an untyped buffer; declared offsets do not apply",
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
    let passes::Transformed {
        module: mut out,
        mut sidecar,
        placeholder_descriptor_bindings,
    } = passes::transform_with_options_and_sidecar(
        emitted.module,
        emitted.sidecar,
        stage,
        frag,
        vert,
        kern,
        entry_name,
        options,
    )?;
    native::close_inlined_bda_pointer_tables_module(&mut out);
    let preserved_pointer_facts = sidecar
        .local_pointer_field_stores
        .iter()
        .map(|fact| fact.id)
        .collect::<std::collections::HashSet<_>>();
    if native::lower_unobserved_bda_aggregate_pointer_fields_module(&mut out)? {
        native::eliminate_dead_values_module(&mut out, &preserved_pointer_facts);
    }
    if retry_debug {
        eprintln!("[retry-debug] finish: passes complete; canonicalize start");
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
    if construction == FinishConstruction::Primary {
        // Static conditions are part of the constructed program, not a validator diagnosis. Remove
        // their unreachable CFG before the primary candidate is assembled, without treating every
        // otherwise-unused emitted computation as an optimization candidate.
        native::prune_constant_cfg_module_if_changed(&mut out);
    }
    native::prune_unused_null_and_undef_constants_module(&mut out);
    // A complete source-ownership rejection chooses the bounded relooper representation after the
    // ordinary lowering and static-CFG construction have established the final reachable value and
    // pointer graph. The decision is carried from source-CFG planning; no serialized output is
    // parsed and no validator failure participates.
    let mut cfg_construction_functions = sidecar.ownership_plan_rejected_functions.clone();
    cfg_construction_functions.extend(
        sidecar
            .post_lowering_cfg_construction_functions
            .iter()
            .cloned(),
    );
    native::construct_cfg_functions_module(&mut out, &cfg_construction_functions)
        .map_err(FinishFailure::cfg)?;
    native::construct_physical_atomic_pointer_lvalues_module(&mut out);
    // Every instruction-deleting step of this boundary has run. Dead-value elimination, constant-CFG
    // pruning, and CFG construction all delete uses after `transform` established global liveness,
    // so a variable can outlive its last use here; re-establish liveness before the module is sealed.
    // An interface entry naming a descriptor binding no instruction touches is a demand on every
    // consumer that builds its layout from the interface. This runs ahead of the constant sweep
    // below so the initializers it strands are collected in the same step.
    passes::drop_unreferenced_global_variables(&mut out);
    // CFG construction rematerializes values used by the selected representation. Discard any
    // null/undef constants that become dead in that final graph; under PhysicalStorageBuffer64 an
    // otherwise unreferenced logical-pointer null is still structurally invalid SPIR-V.
    native::prune_unused_null_and_undef_constants_module(&mut out);
    if construction == FinishConstruction::RawRelooper {
        // CFG construction mints its final ids after the shared pre-CFG canonicalization.
        // Canonicalize the completed selected representation before its only owned check and
        // serialization, matching the deterministic output contract without sealing an
        // intermediate module.
        passes::canonicalize_ids(&mut out);
    }
    if let Some(failure) = native::owned_module_failure(&out) {
        if let Some(path) = env_vars::retry_dump() {
            let _ = std::fs::write(path, assemble_finished_module(&out));
        }
        return Err(failure.into());
    }
    passes::validate_descriptor_bindings(&out, options.descriptor_layout)?;
    // Steps after `transform` delete instructions and can strand a placeholder's last use, and the
    // liveness pass above then takes its variable. Keep only the ones this finished module still
    // declares, so the list reflection reads never names a descriptor that is no longer there.
    let declared = spirv_module::descriptor_bindings_in_set(&out, options.descriptor_layout.set);
    let placeholder_descriptor_bindings = placeholder_descriptor_bindings
        .into_iter()
        .filter(|binding| declared.contains(binding))
        .collect::<Vec<_>>();
    let finished = FinishedModule::new(out, placeholder_descriptor_bindings);
    if retry_debug {
        eprintln!("[retry-debug] finish: assembly complete");
    }
    Ok(finished)
}

/// Diagnostic probe: construct both raw-buffer representations through the same `finish_module`
/// boundary used by production and return their bytes or construction errors. The caller may
/// validate them independently to inspect whether raw byte-offset modeling closes a type invariant.
/// Returns `[device_raw, device_and_workgroup_raw]`.
pub fn translate_raw_tiers_probe(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
) -> Vec<Result<Vec<u8>, String>> {
    let lowered = lower_async_copy_if_enabled(san_ll);
    let san_ll = lowered.as_ref();
    if let Err(error) = reject_unsupported_metal_linked_functions(san_ll) {
        return vec![Err(error.clone()), Err(error)];
    }
    let stage_meta = parse_stage_meta(san_ll, stage);
    let opts = passes::TransformOptions::default();
    if let Err(error) = passes::validate_kernel_dispatch_options(stage, opts) {
        return vec![Err(error.clone()), Err(error)];
    }
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
                FinishConstruction::Plain,
            )
            .map(|finished| finished.bytes)
            .map_err(|failure| failure.error)
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
            &Default::default(),
            &Default::default(),
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
                &Default::default(),
                &Default::default(),
            ),
        ),
    ]
}

/// Diagnostic probe: force the BDA device-pointer model
/// (`emit_vulkan_spirv_all_buffers_raw_bda`) and run it through production finalization so a
/// surviving validation residual can be inspected. Mirrors [`translate_raw_tiers_probe`]; not a
/// frontier signal.
pub fn translate_bda_probe(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
) -> Result<Vec<u8>, String> {
    let lowered = lower_async_copy_if_enabled(san_ll);
    let san_ll = lowered.as_ref();
    reject_unsupported_metal_linked_functions(san_ll)?;
    let stage_meta = parse_stage_meta(san_ll, stage);
    let opts = passes::TransformOptions::default();
    passes::validate_kernel_dispatch_options(stage, opts)?;
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
        &Default::default(),
        &Default::default(),
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
            FinishConstruction::Plain,
        )
        .map(|finished| finished.bytes)
        .map_err(|failure| failure.error)
    })
}

fn translate_sanitized_with_meta(
    san_ll: &str,
    stage: passes::Stage,
    frag: Option<&meta::FragMeta>,
    vert: Option<&meta::VertMeta>,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    tmp: &Path,
    options: passes::TransformOptions,
    datalayout: Option<layout::AirDataLayout>,
) -> Result<Vec<u8>, String> {
    // Shared pre-emission boundary for the production and diagnostic construction paths. The
    // reflected facade performs this same check before reflection analysis, then enters the
    // prevalidated core below.
    passes::validate_kernel_dispatch_options(stage, options)?;
    translate_sanitized_with_meta_prevalidated_carrier(
        san_ll, stage, frag, vert, kern, entry_name, tmp, options, datalayout,
    )
    .map(|finished| finished.bytes)
}

/// Translation core after the facade has enforced the dispatch contract. The reflected facade runs
/// the same check before its potentially expensive reflection analysis, then enters here so the
/// dispatch contract is scanned only once.
fn translate_sanitized_with_meta_prevalidated_carrier(
    san_ll: &str,
    stage: passes::Stage,
    frag: Option<&meta::FragMeta>,
    vert: Option<&meta::VertMeta>,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    tmp: &Path,
    options: passes::TransformOptions,
    datalayout: Option<layout::AirDataLayout>,
) -> Result<FinishedModule, String> {
    let retry_debug_on = env_vars::retry_debug();
    if retry_debug_on {
        eprintln!("[retry-debug] translate: construction context start");
    }
    let rc = construction::ConstructionCtx::new(
        san_ll, stage, frag, vert, kern, entry_name, tmp, options, datalayout,
    );
    if retry_debug_on {
        eprintln!("[retry-debug] translate: primary emission start");
    }
    let primary_emitted = tools::emit_vulkan_spirv_with_outcome(
        san_ll,
        tmp,
        rc.kern,
        rc.entry_name,
        stage_buffer_layouts(rc.stage, rc.frag, rc.vert, rc.kern),
    );
    let primary_finished = match primary_emitted {
        Ok(emitted) => {
            rc.remember_ordinary_plan_rejections(&emitted);
            rc.finish_primary_carrier(emitted)
        }
        Err(failure) => {
            rc.remember_ordinary_plan_rejection_set(&failure.ordinary_plan_rejected_functions);
            rc.remember_ownership_plan_rejection_set(&failure.ownership_plan_rejected_functions);
            Err(failure.error)
        }
    };
    let translated = match primary_finished {
        Ok(finished) => Ok(finished),
        // A complete source-ownership rejection is a construction fact, including when the typed
        // interface expansion grows beyond the bounded relooper or exposes an incompatible phi
        // carrier. Select the raw-buffer CFG representation directly while every candidate is still
        // owned; validate only the one finished construction and never parse it back for repair.
        Err(emit_err) if rc.needs_raw_construction() => {
            let constructed = rc.construct_raw().map_err(|construction_error| {
                format!("{emit_err}; raw construction failed: {construction_error}")
            })?;
            Ok(constructed)
        }
        Err(emit_err) => Err(emit_err),
    }
    .and_then(|constructed| {
        // This is the sole production validation boundary. Its verdict is returned directly and
        // cannot select another representation or trigger another construction pass.
        if let Some(path) = env_vars::retry_dump() {
            let _ = std::fs::write(path, &constructed.bytes);
        }
        tools::spirv_val_bytes(&constructed.bytes, tmp)?;
        if retry_debug_on {
            eprintln!("[retry-debug] constructed module validated in-translate");
        }
        Ok(constructed)
    });
    // Compatibility telemetry now records only whether construction produced a module. It does not
    // expose representation selection and cannot affect product behavior.
    if crate::env_vars::tier_census() {
        let label = match &translated {
            Ok(_) => "default",
            Err(_) => "fallback",
        };
        eprintln!("[tier-census] {label}");
    }
    translated
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
    fn specialized_reflection_includes_function_constant_gated_argument_buffer() {
        let ll = r#"
@enabled.MTL_FC_INIT_7_b = internal addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@enabled_pred = internal addrspace(2) global i8 0, align 1

declare i1 @air.is_function_constant_defined(ptr addrspace(2))

define internal void @_GLOBAL__sub_I_enabled() section "air.static_init" {
  %value = load i8, ptr addrspace(2) @enabled.MTL_FC_INIT_7_b
  %defined = call i1 @air.is_function_constant_defined(ptr addrspace(2) @enabled.MTL_FC_INIT_7_b)
  %set = icmp ne i8 %value, 0
  %selected = select i1 %defined, i1 %set, i1 false
  %byte = zext i1 %selected to i8
  store i8 %byte, ptr addrspace(2) @enabled_pred
  ret void
}

define void @k(ptr addrspace(2) %args) {
  ret void
}

!air.vertex = !{!0}
!air.function_constants = !{!8}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.function_constant", !4, !"air.indirect_buffer", !"air.location_index", i32 30, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Args"}
!4 = !{ptr addrspace(2) @enabled_pred, !"bool", !"enabled"}
!5 = !{i32 0, i32 8, i32 0, !"void", !"data", !"air.indirect_argument", !6}
!6 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"void"}
!8 = !{ptr addrspace(2) @enabled.MTL_FC_INIT_7_b, !"bool", !"enabled", i32 7, i1 false}
"#;

        let default = reflect_sanitized(
            ll,
            passes::Stage::Vertex,
            passes::TransformOptions::default(),
        )
        .expect("reflect default");
        assert!(!default
            .bindings
            .iter()
            .any(|binding| binding.metal_index == 30));

        let disabled = reflect_sanitized_specialized(
            ll,
            passes::Stage::Vertex,
            passes::TransformOptions::default(),
            &[(7, vec![0])],
        )
        .expect("reflect explicitly disabled");
        assert!(!disabled
            .bindings
            .iter()
            .any(|binding| binding.metal_index == 30));

        let enabled = reflect_sanitized_specialized(
            ll,
            passes::Stage::Vertex,
            passes::TransformOptions::default(),
            &[(7, vec![1])],
        )
        .expect("reflect enabled");
        assert!(enabled.bindings.iter().any(|binding| {
            binding.kind == reflect::ResourceKind::Buffer && binding.metal_index == 30
        }));
        assert!(enabled.bindings.iter().any(|binding| {
            binding.kind == reflect::ResourceKind::EmbeddedArgBufferBuffer
                && binding
                    .embedded_source
                    .is_some_and(|source| source.buffer_index == 30 && source.field_offset == 0)
        }));
    }

    #[test]
    fn metadata_only_reflection_reports_the_address_table_a_device_pointer_needs() {
        let ll = r#"
define void @k(ptr addrspace(1) %out, i64 %address) {
  %p = inttoptr i64 %address to ptr addrspace(1)
  %v = load i32, ptr addrspace(1) %p, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
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

        // The same `inttoptr` with nothing reading it needs no address model, and the emitter
        // builds none. A text scan cannot tell the two apart -- it sees the token either way, and
        // reported a table for 49 corpus modules that declare none.
        let dead = ll.replace(
            "  %v = load i32, ptr addrspace(1) %p, align 4\n  store i32 %v, ptr addrspace(1) %out, align 4\n",
            "",
        );
        assert!(!reflect_sanitized(
            &dead,
            passes::Stage::Kernel,
            passes::TransformOptions::default(),
        )
        .expect("reflect dead-pointer kernel")
        .bindings
        .iter()
        .any(|binding| binding.kind == reflect::ResourceKind::BufferAddressTable));
    }

    /// The predicate is the emitter's, so it sees what a text scan cannot. This kernel loads a
    /// device pointer out of a *struct member* through a GEP -- there is no `inttoptr`, and the
    /// load's pointer operand is a temporary rather than a literal `ptr addrspace(1)` operand, so
    /// a line-prefix scan of the AIR reads nothing. The pointer graph says otherwise, and the
    /// emitted module carries the address model, so reflection has to report the table.
    #[test]
    fn metadata_only_reflection_sees_a_device_pointer_a_text_scan_cannot() {
        let ll = r#"
%struct.Handles = type { ptr addrspace(1), i32 }

define void @k(ptr addrspace(1) %handles, ptr addrspace(1) %out) {
entry:
  %slot = getelementptr inbounds %struct.Handles, ptr addrspace(1) %handles, i64 0, i32 0
  %device = load ptr addrspace(1), ptr addrspace(1) %slot, align 8
  %value = load i32, ptr addrspace(1) %device, align 4
  store i32 %value, ptr addrspace(1) %out, align 4
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"Handles", !"air.arg_name", !"handles"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
        let metadata_only = reflect_sanitized(
            ll,
            passes::Stage::Kernel,
            passes::TransformOptions::default(),
        )
        .expect("reflect device-address kernel");
        assert!(
            metadata_only
                .bindings
                .iter()
                .any(|binding| binding.kind == reflect::ResourceKind::BufferAddressTable),
            "{:?}",
            metadata_only
                .bindings
                .iter()
                .map(|binding| binding.kind)
                .collect::<Vec<_>>()
        );

        // And it agrees with what translation actually emits, which is the point of asking the
        // emitter's predicate rather than a second one.
        let tmp = std::env::temp_dir().join(format!(
            "metal2vulkan_metadata_address_table_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&tmp);
        let (_, translated) = translate_sanitized_native_reflected(
            ll,
            passes::Stage::Kernel,
            &tmp,
            passes::TransformOptions::default(),
        )
        .expect("translate device-address kernel");
        let _ = std::fs::remove_dir_all(&tmp);
        let table_bindings = |reflection: &reflect::ShaderReflection| {
            reflection
                .bindings
                .iter()
                .filter(|binding| binding.kind == reflect::ResourceKind::BufferAddressTable)
                .filter_map(|binding| binding.descriptor.map(|descriptor| descriptor.binding))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            table_bindings(&metadata_only),
            table_bindings(&translated),
            "metadata-only and reflected translation must agree on the address table"
        );
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
    fn owned_type_failure_cannot_select_alternate_construction() {
        let failure = FinishFailure::from(native::OwnedModuleFailure::TypeConstruction(
            "invalid owned type graph".to_string(),
        ));
        assert_eq!(failure.kind, FinishFailureKind::Other);
        assert_eq!(failure.error, "invalid owned type graph");
    }

    #[test]
    fn validating_primary_finish_assembles_once() {
        let tmp =
            std::env::temp_dir().join(format!("metal2vulkan_finish_once_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp);
        reset_finish_assemble_count();
        spirv_module::reset_load_bytes_count();
        let spv = translate_sanitized_native(SIMPLE_KERNEL, passes::Stage::Kernel, &tmp)
            .expect("simple primary translation validates");
        tools::spirv_val_bytes(&spv, &tmp).expect("simple primary spirv-val");
        assert_eq!(
            finish_assemble_count(),
            1,
            "a validating primary must not assemble fallback bytes"
        );
        assert_eq!(
            spirv_module::load_bytes_count(),
            0,
            "production translation must not parse its serialized output for repair"
        );
    }

    #[test]
    fn selected_raw_relooper_finish_assembles_once() {
        let tmp = std::env::temp_dir().join(format!(
            "metal2vulkan_raw_relooper_finish_once_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&tmp);
        let stage_meta = parse_stage_meta(SIMPLE_KERNEL, passes::Stage::Kernel);
        let construction = construction::ConstructionCtx::new(
            SIMPLE_KERNEL,
            passes::Stage::Kernel,
            stage_meta.frag.as_ref(),
            stage_meta.vert.as_ref(),
            stage_meta.kern.as_ref(),
            stage_meta.entry_name.as_deref(),
            &tmp,
            passes::TransformOptions::default(),
            layout::AirDataLayout::from_ir(SIMPLE_KERNEL).expect("AIR datalayout"),
        );
        reset_finish_assemble_count();
        spirv_module::reset_load_bytes_count();
        native::reset_address_construction_counts();

        let finished = construction
            .construct_raw_relooper()
            .expect("raw relooper construction");
        tools::spirv_val_bytes(&finished.bytes, &tmp).expect("raw relooper spirv-val");
        assert_eq!(
            finish_assemble_count(),
            1,
            "the selected raw-relooper representation must have no intermediate assembly"
        );
        assert_eq!(
            spirv_module::load_bytes_count(),
            0,
            "raw-relooper construction must remain owned through final assembly"
        );
        assert_eq!(
            native::address_construction_count(),
            1,
            "raw-relooper address closure must be owned by interface construction"
        );
        let _ = std::fs::remove_dir_all(tmp);
    }

    #[test]
    fn reflected_translation_retains_the_finished_owned_module() {
        let tmp = std::env::temp_dir().join(format!(
            "metal2vulkan_reflected_owned_module_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&tmp);
        spirv_module::reset_load_bytes_count();

        let (spv, reflection) = translate_sanitized_native_reflected(
            SIMPLE_KERNEL,
            passes::Stage::Kernel,
            &tmp,
            passes::TransformOptions::default(),
        )
        .expect("simple reflected translation validates");

        assert!(!spv.is_empty());
        assert!(reflection.bindings.iter().any(|binding| {
            binding.kind == reflect::ResourceKind::Buffer && binding.footprint.is_some()
        }));
        assert_eq!(
            spirv_module::load_bytes_count(),
            0,
            "reflection must analyze the finished owned module without parsing output bytes"
        );
    }
}
