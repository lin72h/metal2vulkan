//! metal2vulkan — Metal AIR (LLVM bitcode) -> Vulkan SPIR-V, via a native LLVM-IR emitter.
//!
//! The decisive difference from the legacy `metal2vulkanspirv` crate: that one targeted the OpenCL/Kernel
//! SPIR-V backend (`spirv64-unknown-unknown`), which emits UNSTRUCTURED, Physical64/OpenCL-dialect
//! SPIR-V, then did a ~1300-line regex dialect rewrite + a hand-written structurizer to reach a
//! Vulkan-legal module. Here the native emitter produces `OpCapability Shader` / `Logical GLSL450`
//! SPIR-V directly from sanitized AIR LLVM IR. What remains is the stage *interface* — done as a
//! handful of passes on the crate-owned SPIR-V module representation (see `passes/`).
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

pub mod as_shadow;
mod emit_sidecar;
pub mod env_vars;
mod fc_specialize;
mod layout;
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

pub use fc_specialize::specialize_function_constants;
pub use passthrough::translate_passthrough;
use primary_retry::*;

use crate::spirv_module::{load_bytes as load_owned_module, Module};
use std::path::Path;

/// Translate an AIR/`.ll` shader to Vulkan SPIR-V bytes for the given stage. On success returns the
/// assembled SPIR-V words. The caller validates with `tools::spirv_val`.
/// Detect the shader stage from the AIR's own `!air.vertex`/`!air.fragment`/`!air.kernel` metadata
/// (which the compiler emits and SPIR-V emission later drops). Lets the caller translate a captured
/// AIR blob without knowing its stage — the kext forwards raw AIR; the stage is intrinsic to the
/// module. A fragment shader run as `--stage vertex` (or vice-versa) mis-maps the [[position]] role,
/// so detecting beats guessing.
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

pub fn translate(src: &str, stage: passes::Stage, tmp: &Path) -> Result<Vec<u8>, String> {
    translate_with_options(src, stage, tmp, passes::TransformOptions::default())
}

pub fn translate_with_options(
    src: &str,
    stage: passes::Stage,
    tmp: &Path,
    options: passes::TransformOptions,
) -> Result<Vec<u8>, String> {
    let san_ll = tools::air_to_sanitized_ll(src, tmp)?;
    translate_sanitized_native_with_options(&san_ll, stage, tmp, options)
}

/// Translate already-sanitized LLVM IR through the native emitter. The LLVM `llc` backend and its
/// crash-workaround passes are not used.
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
) -> reflect::ShaderReflection {
    match stage {
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
    }
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
    let stage_meta = parse_stage_meta(san_ll, stage);
    translate_sanitized_with_meta(
        san_ll,
        stage,
        stage_meta.frag.as_ref(),
        stage_meta.vert.as_ref(),
        stage_meta.kern.as_ref(),
        stage_meta.promoted_kern.as_ref(),
        stage_meta.entry_name.as_deref(),
        tmp,
        passes::TransformOptions::default(),
        false,
    )
}

pub fn translate_sanitized_native_with_options(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
    options: passes::TransformOptions,
) -> Result<Vec<u8>, String> {
    // Lower `air.simdgroup_async_copy_2d` (+ its event/wait pair) to an explicit strided tile copy
    // BEFORE any meta parse or emit, so every path — the default emit AND every retry tier (which
    // re-emit from `san_ll`) — sees ordinary LLVM instead of the unhandled intrinsic. This is what lets
    // the FC-promote/prune/PSB retries fix the SECOND wall these kernels carry (a Private-demoted
    // pointer arm over-indexed as an array), which they could not before when the lowering lived only
    // inside `emit_vulkan_spirv` (the retries bypass that entry). Floor-safe by construction: the
    // rewrite is a no-op unless the module calls `air.simdgroup_async_copy_2d`, and such modules fail
    // the emitter outright today. See `native::async_copy` + journal AIR2VK-ASYNC-COPY-CLEAR.
    let lowered = lower_async_copy_if_enabled(san_ll);
    let san_ll = lowered.as_str();
    let stage_meta = parse_stage_meta(san_ll, stage);
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
    )
}

/// Like [`translate`] but also returns the [`reflect::ShaderReflection`] — the stage-interface
/// facade (descriptor bindings, vertex attributes, varyings, render targets) the translator already
/// parsed, so a downstream consumer never re-reflects the emitted SPIR-V. The SPIR-V bytes are
/// byte-identical to [`translate`]; reflection is a pure re-shaping of the parsed metadata.
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
    // Capture the `target datalayout` before sanitization drops it, so the reflection carries it and
    // the consumer never re-reads the source `.ll` from disk. (The sanitized-entry sibling has no
    // source datalayout to recover, so it leaves reflection.datalayout = None.)
    let (san_ll, datalayout) = tools::air_to_sanitized_ll_with_datalayout(src, tmp)?;
    let (spv, mut reflection) = translate_sanitized_native_reflected(&san_ll, stage, tmp, options)?;
    reflection.datalayout = datalayout;
    Ok((spv, reflection))
}

/// [`translate_sanitized_native_with_options`] plus the reflection facade. See [`translate_reflected`].
pub fn translate_sanitized_native_reflected(
    san_ll: &str,
    stage: passes::Stage,
    tmp: &Path,
    options: passes::TransformOptions,
) -> Result<(Vec<u8>, reflect::ShaderReflection), String> {
    let lowered = lower_async_copy_if_enabled(san_ll);
    let san_ll = lowered.as_str();
    let stage_meta = parse_stage_meta(san_ll, stage);
    let mut reflection = build_reflection(
        stage,
        stage_meta.frag.as_ref(),
        stage_meta.vert.as_ref(),
        stage_meta.kern.as_ref(),
        stage_meta.entry_name.as_deref(),
        &options,
    );
    // Function constants are a cross-stage IR fact (not in the per-stage meta): scan the sanitized
    // IR once so a consumer can discover the module's spec-ids without walking SPIR-V.
    reflection.function_constants = meta::parse_function_constants(san_ll);
    // AIR constexpr samplers are module globals rather than entry parameters, so stage metadata
    // does not carry them. Reflect their decoded state and the same first-free sampler-band
    // allocation used by the interface pass before returning the consumer contract.
    reflection.add_static_samplers(san_ll)?;
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
    )?;
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
    // `tools::llc_vulkan_spirv` is the in-process native emitter (no subprocess); `finish_module` is
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
    if let Some(primary) = primary_xbind_phi_psb_for_validation_error(
        &primary_module,
        &primary,
        &validation_error,
        tmp,
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
    );
    Ok(primary_primitive_phi_metadata_if_needed(
        &validation_error,
        san_ll,
        stage,
        stage_meta.frag.as_ref(),
        stage_meta.vert.as_ref(),
        stage_meta.kern.as_ref(),
        stage_meta.entry_name.as_deref(),
        tmp,
        passes::TransformOptions::default(),
    )
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
            san_ll,
            stage,
            stage_meta.frag.as_ref(),
            stage_meta.vert.as_ref(),
            stage_meta.kern.as_ref(),
            stage_meta.entry_name.as_deref(),
            tmp,
            passes::TransformOptions::default(),
        )
    })
    .or_else(|| {
        primary_logical_pointer_inline_sroa_raw_if_needed(
            &validation_error,
            san_ll,
            stage,
            stage_meta.frag.as_ref(),
            stage_meta.vert.as_ref(),
            stage_meta.kern.as_ref(),
            stage_meta.entry_name.as_deref(),
            tmp,
            passes::TransformOptions::default(),
        )
    })
    .or_else(|| {
        primary_opaque_image_select_module_if_needed(
            &validation_error,
            &primary_module,
            &primary,
            tmp,
        )
    })
    .unwrap_or(primary))
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
    emitted: emit_sidecar::EmittedSpirv,
    stage: passes::Stage,
    frag: Option<&meta::FragMeta>,
    vert: Option<&meta::VertMeta>,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    options: passes::TransformOptions,
    rewrites: FinishRewrites,
) -> Result<FinishedModule, String> {
    let (mut out, sidecar) = passes::transform_with_options_and_sidecar(
        emitted.module,
        emitted.sidecar,
        stage,
        frag,
        vert,
        kern,
        entry_name,
        options,
    )?;
    // M4: legalize illegal logical-pointer phis (Private/Function/UniformConstant `OpPhi`) on the
    // PRIMARY emit path via the phi-the-index rewrite, so those functions validate directly instead of
    // shipping only via retry rescue. Floor-safe by construction (it touches only already-invalid
    // logical-pointer phis, never a validating module).
    native::rewrite_logical_pointer_phis_module(&mut out);
    // M4: legalize integer-width phi mismatches (a narrow `uint` loop phi fed a wide `ulong` induction
    // back-edge value) by truncating the wide incoming to the phi's result type. Floor-safe (touches
    // only already-invalid integer phis). Runs after the pointer-phi rewrite so it can also legalize
    // any width mismatch that rewrite's synthesized index phis might introduce.
    native::rewrite_integer_width_phis_module(&mut out);
    // M4/Keystone-2: repair loop-closed-SSA violations the `MultipleExits` loop funnel leaves behind (a
    // loop-body value used raw in a post-loop block the restructured CFG can reach without the defining
    // block), by register-demoting the offending value to a function-scope OpVariable. Floor-safe by
    // construction — a validating module has every def dominating its uses, so this never fires on one.
    native::demote_nondominating_values_module(&mut out);
    // M4/Keystone-2: node-split a multi-entry loop whose header is entered from two different
    // selections' arms (the irreducible shape `structured_plan` over-admits — the mlx-steel
    // `steel_attention` family), so the PRIMARY structured emit validates instead of shipping only via
    // the relooper retry. Floor-safe by construction (a valid loop is single-entry). Runs after the phi
    // rewrites; the attention rows also need the integer-width phi legalization above.
    native::split_multientry_loop_selection_exits_module(&mut out);
    // Renumber all ids into a deterministic, serialized-order canonical form. This format
    // normalization keeps equivalent producer paths directly comparable and SPIR-V-level diffs
    // meaningful.
    let mut retained_global_ids = sidecar
        .local_pointer_field_stores
        .iter()
        .map(|fact| fact.id)
        .collect::<Vec<_>>();
    passes::canonicalize_ids_and_remap(&mut out, &mut retained_global_ids);
    let original_module = (rewrites == FinishRewrites::Primary).then(|| out.clone());
    if rewrites == FinishRewrites::Primary {
        primary_retry::apply_primary_emit_rewrites_module(&mut out, &mut retained_global_ids);
    }
    let bytes = assemble_finished_module(&out);
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
    let stage_meta = parse_stage_meta(san_ll, stage);
    let opts = passes::TransformOptions::default();
    let run = |emitted: Result<emit_sidecar::EmittedSpirv, String>| -> Result<Vec<u8>, String> {
        emitted.and_then(|b| {
            finish_module(
                b,
                stage,
                stage_meta.frag.as_ref(),
                stage_meta.vert.as_ref(),
                stage_meta.kern.as_ref(),
                stage_meta.entry_name.as_deref(),
                opts,
                FinishRewrites::Plain,
            )
            .map(|finished| finished.bytes)
        })
    };
    vec![
        run(tools::llc_vulkan_spirv_all_buffers_raw_with_sidecar(
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
            tools::llc_vulkan_spirv_all_buffers_raw_with_workgroup_sidecar(
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
    let stage_meta = parse_stage_meta(san_ll, stage);
    let opts = passes::TransformOptions::default();
    tools::llc_vulkan_spirv_all_buffers_raw_bda_with_sidecar(
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
    );
    let retry_debug_on = rc.retry_debug_on;
    let translated = match tools::llc_vulkan_spirv_with_sidecar(
        san_ll,
        tmp,
        rc.kern,
        rc.entry_name,
        stage_buffer_layouts(rc.stage, rc.frag, rc.vert, rc.kern),
    )
    .and_then(|b| {
        finish_module(
            b,
            rc.stage,
            rc.frag,
            rc.vert,
            rc.kern,
            rc.entry_name,
            rc.options,
            FinishRewrites::Primary,
        )
    }) {
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
                if let Some(primary) = primary_xbind_phi_psb_for_validation_error(
                    &primary_module,
                    &primary,
                    &validation_error,
                    tmp,
                ) {
                    Ok(primary)
                } else if let Some(primary) = primary_primitive_phi_metadata_if_needed(
                    &validation_error,
                    san_ll,
                    rc.stage,
                    rc.frag,
                    rc.vert,
                    rc.kern,
                    rc.entry_name,
                    tmp,
                    rc.options,
                ) {
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
                        san_ll,
                        rc.stage,
                        rc.frag,
                        rc.vert,
                        rc.kern,
                        rc.entry_name,
                        tmp,
                        rc.options,
                    )
                {
                    Ok(primary)
                } else if let Some(primary) = primary_logical_pointer_inline_sroa_raw_if_needed(
                    &validation_error,
                    san_ll,
                    rc.stage,
                    rc.frag,
                    rc.vert,
                    rc.kern,
                    rc.entry_name,
                    tmp,
                    rc.options,
                ) {
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
                            let fc_promote_first = e
                                .contains("reached non-composite")
                                .then(|| {
                                    rc.census("val-ptr:fc_promote_logical", rc.fc_promote_logical())
                                })
                                .flatten();
                            Ok(fc_promote_first
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
                        // emitted bytes; keep the default bytes if neither validates.
                        Err(e)
                            if native::classify_validation_error(&e)
                                == native::ValidationClass::CfgStructurization =>
                        {
                            Ok({
                                if retry_debug_on {
                                    let head: Vec<&str> = e.lines().take(4).collect();
                                    eprintln!(
                                    "[retry-debug] default module failed cfg-class spirv-val: {}",
                                    head.join(" | ")
                                );
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
                            // Last resort for a HUGE (>1024-block) emitted module whose only blocker is the cfg
                            // nesting violation — lift the relooper cap (see `relooper_retry_huge`). Clears
                            // 02/07ef16ba (4630 blocks, reducible, no secondary wall).
                            .or_else(|| {
                                rc.census("val-cfg:relooper_huge", rc.relooper_retry_huge(&out))
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
                            .unwrap_or(out))
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
    // phi standing). The index-phi form is semantically identical and needs no variable pointers, so
    // adopt it whenever it independently validates; keep the arm's bytes otherwise. Floor-safe by
    // construction (adopt-if-validates).
    translated.map(|out| {
        let candidate = load_owned_module(&out).ok().and_then(|mut module| {
            native::rewrite_variable_pointer_phis_module(&mut module).ok()?;
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
        tools::llc_vulkan_spirv_with_sidecar(
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
