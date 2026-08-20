//! The primary emit path plus its validation-error retry tiers. `translate_native_primary_validated`
//! emits once, then `emit_finish_primary`/`apply_*`/`primary_*_if_needed` try an ordered cascade of
//! structural rewrites, each re-validated, until one passes spirv-val (or all are exhausted). Byte
//! behavior is identical to when this lived in `lib.rs`; extracted for cohesion.

use super::*;
use crate::spirv_module::{load_bytes, Module};

fn module_bytes(module: &Module) -> Vec<u8> {
    module
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
}

fn prune_constant_branches_candidate(spv: &[u8]) -> Option<Module> {
    let mut module = load_bytes(spv).ok()?;
    native::prune_constant_branches_module(&mut module).ok()?;
    Some(module)
}

fn relooper_candidate(spv: &[u8]) -> Option<Vec<u8>> {
    let mut module = load_bytes(spv).ok()?;
    native::rewrite_to_relooper_module(&mut module).ok()?;
    Some(module_bytes(&module))
}

/// A raw logical-buffer write with a wide dynamic offset carries a local robust-write selection:
/// write only when the full i64 byte sum fits the u32 logical address model.  When that selection
/// occurs in a source loop header, its continuation owns the source terminator and can expose a
/// CFG-only validator error.  Reuse the exact ordered prefix of production's CFG cascade through
/// raw→relooper, but only for that emitted structural guard and only after each candidate validates.
/// This preserves the production candidate order (ordinary relooper, prune, high-cap, value lowering,
/// then raw→relooper) rather than bypassing a more portable earlier form.
pub(crate) fn primary_wide_raw_store_guard_cfg_chain_if_needed(
    validation_error: &str,
    spv: &[u8],
    module: &Module,
    retry: &retry::RetryCtx<'_>,
) -> Option<Vec<u8>> {
    if native::classify_validation_error(validation_error)
        != native::ValidationClass::CfgStructurization
        || !native::module_has_wide_raw_store_guard(module)
    {
        return None;
    }
    retry
        .census("val-cfg:relooper", retry.relooper_retry(spv))
        .or_else(|| {
            retry.census(
                "val-cfg:prune_then_relooper",
                retry.prune_then_relooper(spv),
            )
        })
        .or_else(|| {
            retry.census(
                "val-cfg:relooper_then_value_select",
                retry.relooper_then_value_select(spv),
            )
        })
        .or_else(|| retry.census("val-cfg:raw_then_relooper", retry.raw_then_relooper()))
}

/// Emit + `finish_module` the default (structured) way, then apply the in-process primary-emit
/// rewrites. This is the default path's output up to (but not including) `spirv_val_bytes` and the
/// tier cascade. Shared by the no-retry baseline path and its validation-gated primary-floor wrapper,
/// so both begin from the same structured bytes as the production default attempt. (The former
/// Keystone-2 `STRUCTURED_EXIT_GATE` demote-to-repair tail was removed with the W4 repair-roster
/// deletion — the MPS attention-kernel `structured-exit` family now ships via the retry cascade.)
pub(super) fn emit_finish_primary_module(
    san_ll: &str,
    stage: passes::Stage,
    frag: Option<&meta::FragMeta>,
    vert: Option<&meta::VertMeta>,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    options: passes::TransformOptions,
) -> Result<FinishedModule, String> {
    let air_data_layout = crate::layout::AirDataLayout::from_ir(san_ll)?;
    let structured = tools::emit_vulkan_spirv_with_sidecar(
        san_ll,
        Path::new(""),
        kern,
        entry_name,
        stage_buffer_layouts(stage, frag, vert, kern),
    )
    .and_then(|b| {
        finish_module(
            b,
            stage,
            frag,
            vert,
            kern,
            entry_name,
            air_data_layout.as_ref(),
            options,
            FinishRewrites::Primary,
        )
    })?;
    // Callers choose the byte-only BC boundary or retain the module for validation-gated rewrites.
    // The pointer-phi PSB form remains outside this helper, so a speculative physical rewrite cannot
    // perturb the subprocess-free baseline.
    Ok(structured)
}

/// PhysicalStorageBuffer64 lowering for a cross-binding POINTER PHI. A `select` can replay values at
/// its consumer in plain Logical storage, but a phi's chosen edge has no predicate at that consumer
/// and its dynamic access indices may be defined only after the merge. The PSB address table keeps
/// that pointer selection representable and leaves ordinary selects to
/// [`apply_primary_emit_rewrites_module`].
/// This pass is deliberately kept OUT of `finish_module`, so it touches only the primary candidate
/// and never a retry re-emit.
pub(crate) fn apply_xbind_phi_psb(
    module: &Module,
    spv: &[u8],
    descriptor_layout: crate::reflect::DescriptorLayout,
) -> Vec<u8> {
    let mut candidate = module.clone();
    if native::rewrite_cross_binding_pointer_phis_module(&mut candidate, descriptor_layout).is_err()
    {
        return spv.to_vec();
    }
    candidate
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
}

/// Build the purely in-process primary candidate on the already-loaded, already-canonicalized
/// module. Keeping both rewrites in memory retains the typed sidecar roots that replaced legacy
/// debug-marker liveness. The ordering and second canonicalization after value lowering match the
/// former bytes adapters exactly. Retry re-emits use [`FinishRewrites::Plain`] and never call this.
pub(crate) fn apply_primary_emit_rewrites_module(
    module: &mut Module,
    retained_global_ids: &mut [spirv::Word],
    sidecar: &mut crate::emit_sidecar::EmitSidecar,
) -> (bool, bool) {
    let pointer_changed = native::value_lower_cross_binding_pointer_merges_module(module);
    if pointer_changed {
        passes::canonicalize_ids_and_remap_sidecar(module, retained_global_ids, sidecar);
    }
    let cfg_changed = apply_agx_fc_prune_module(module, retained_global_ids);
    (pointer_changed, cfg_changed)
}

pub(crate) fn primary_xbind_phi_psb_for_validation_error(
    module: &Module,
    spv: &[u8],
    validation_error: &str,
    tmp: &Path,
    descriptor_layout: crate::reflect::DescriptorLayout,
) -> Option<Vec<u8>> {
    if native::classify_validation_error(validation_error)
        != native::ValidationClass::CrossBindingPointerMerge
    {
        return None;
    }
    if !native::has_cross_binding_pointer_phi_module(module) {
        return None;
    }
    let primary = apply_xbind_phi_psb(module, spv, descriptor_layout);
    (primary != spv && tools::spirv_val_bytes(&primary, tmp).is_ok()).then_some(primary)
}

/// Re-emit a primary only after its unseeded form proves a pointer-typing failure. The optional
/// parser mode requires an exact primitive metadata type, a matching post-phi GEP source type, and
/// the declared byte extent; the re-emission is still adopted only when it validates (or its
/// cross-binding pointer phi validates after the physical address lowering).
pub(crate) fn primary_primitive_phi_metadata_if_needed(
    validation_error: &str,
    retry: &retry::RetryCtx<'_>,
) -> Option<Vec<u8>> {
    if native::classify_validation_error(validation_error) != native::ValidationClass::PointerTyping
    {
        return None;
    }
    let emitted = native::emit_vulkan_spirv_with_primitive_phi_metadata_sidecar(
        retry.san_ll,
        retry.kern,
        retry.entry_name,
        stage_buffer_layouts(retry.stage, retry.frag, retry.vert, retry.kern),
    )
    .ok()?;
    let primary = retry.finish_primary_carrier(emitted).ok()?;
    let primary_module = primary.module;
    let primary = primary.bytes;
    match tools::spirv_val_bytes(&primary, retry.tmp) {
        Ok(()) => Some(primary),
        Err(error) => primary_xbind_phi_psb_for_validation_error(
            &primary_module,
            &primary,
            &error,
            retry.tmp,
            retry.options.descriptor_layout,
        ),
    }
}

/// A validator-proven StorageBuffer over-index already identifies the raw-buffer model and makes
/// primitive-phi metadata probing irrelevant. Feed that byte view straight to the relooper before
/// either retry repeats the expensive native structured-plan search.
pub(crate) fn primary_storage_buffer_raw_feed_if_needed(
    validation_error: &str,
    retry: &retry::RetryCtx<'_>,
) -> Option<Vec<u8>> {
    validation_error_may_be_storage_buffer_overindex(validation_error)
        .then(|| {
            retry.census(
                "val-ptr:raw_feed_then_relooper",
                retry.raw_feed_then_relooper(),
            )
        })
        .flatten()
}

/// Promote the first production pointer-typing retry as a primary candidate. For the
/// `reached non-composite` Private-placeholder class, try the FC-buffer-promoted Logical projection
/// first: if an FC-wrapped buffer is actually used, binding it as the real descriptor preserves the
/// intended pointer better than the all-buffer-raw fallback's private zero scratch. Otherwise keep the
/// established all-buffer-raw first choice. Every candidate is retained only when it independently
/// validates.
pub(crate) fn primary_pointer_typing_raw_if_needed(
    validation_error: &str,
    retry: &retry::RetryCtx<'_>,
) -> Option<Vec<u8>> {
    if native::classify_validation_error(validation_error) != native::ValidationClass::PointerTyping
    {
        return None;
    }
    if validation_error_may_be_fc_private_placeholder(validation_error) {
        if let Some(bytes) = retry.fc_promote_logical() {
            return Some(bytes);
        }
    }
    retry.raw_retry()
}

/// A pointer-typing primary can become a CFG-only failure after all buffers are modeled raw.  The
/// wide raw-write guard is the structural proof for that composition; replay the exact production
/// pointer branch between the ordinary raw retry and its raw→relooper/PSB tail so an earlier prune or
/// subword rewrite retains its established priority.  Every invoked retry independently validates.
pub(crate) fn primary_pointer_typing_wide_raw_guard_chain_if_needed(
    validation_error: &str,
    spv: &[u8],
    module: &Module,
    retry: &retry::RetryCtx<'_>,
) -> Option<Vec<u8>> {
    if native::classify_validation_error(validation_error) != native::ValidationClass::PointerTyping
        || !retry.raw_reemit_has_wide_raw_store_guard()
    {
        return None;
    }
    retry
        .census("val-ptr:prune", retry.prune_retry(spv))
        .or_else(|| retry.census("val-ptr:subword_pack", retry.subword_pack_retry(module)))
        .or_else(|| {
            retry.census(
                "val-ptr:prune_then_relooper",
                retry.prune_then_relooper(spv),
            )
        })
        .or_else(|| retry.raw_then_psb_chain("val-ptr:raw_then_relooper", "val-ptr:raw_psb"))
}

/// Promote the existing `val-other:prune` retry only for an `Other` primary validation error. This
/// covers the function-constant-dead table shape where an optional buffer's Private zero placeholder
/// is selected with a real StorageBuffer arm; static branch pruning removes the disabled arm. If that
/// semantics-preserving prune exposes only a structured-CFG rejection, apply the existing capped
/// prune-then-relooper sequence before adoption. The raw byte baseline remains unmodified, and every
/// candidate is independently validated. Other validation classes retain their established
/// higher-priority repair routing.
pub(crate) fn primary_other_prune_if_needed(
    validation_error: &str,
    spv: &[u8],
    tmp: &Path,
) -> Option<Vec<u8>> {
    if native::classify_validation_error(validation_error) != native::ValidationClass::Other {
        return None;
    }
    let mut pruned_module = prune_constant_branches_candidate(spv)?;
    let pruned = module_bytes(&pruned_module);
    if pruned == spv {
        return None;
    }
    match tools::spirv_val_bytes(&pruned, tmp) {
        Ok(()) => return Some(pruned),
        Err(error)
            if native::classify_validation_error(&error)
                != native::ValidationClass::CfgStructurization =>
        {
            return None;
        }
        Err(_) => {}
    }
    native::rewrite_to_relooper_module_capped(
        &mut pruned_module,
        native::PRUNE_THEN_RELOOPER_MAX_BLOCKS,
    )
    .ok()?;
    let primary = module_bytes(&pruned_module);
    (primary != spv && tools::spirv_val_bytes(&primary, tmp).is_ok()).then_some(primary)
}

/// Prune statically-dead function-constant arms before source-level CFG re-emission when the
/// primary failure is itself a structured-control-flow rule. If pruning alone validates, it removes
/// both the dead CFG violation and the need to parse/emit/finish a construct-tree candidate.
pub(crate) fn primary_cfg_prune_if_needed(
    validation_error: &str,
    spv: &[u8],
    retry: &retry::RetryCtx<'_>,
) -> Option<Vec<u8>> {
    if native::classify_validation_error(validation_error)
        != native::ValidationClass::CfgStructurization
    {
        return None;
    }
    retry.census("val-cfg:prune", retry.prune_retry(spv))
}

/// Adopt the ordinary relooper rewrite when the primary fails a CFG structural rule and the relooper
/// produces a validator-clean module.
///
/// The full retry cascade already owns this form under the `val-cfg:relooper` census label
/// (`relooper_retry`). Historically `primary_cfg_prune_then_relooper_if_needed` *declined* when the
/// plain relooper worked so that cascade could keep the tier label — but
/// [`crate::translate_native_primary_validated`] has no later relooper step, so the banked compound-loop
/// residual (`09/de938adc`, shared merge block / `loop:MultipleExits+MultipleLatches[k=2,phi=1]`) stayed
/// PV-invalid even though production already ships the relooped bytes. This helper closes that gap:
/// validation-gated, structural (not name-keyed), and byte-identical for every primary that already
/// spirv-val-passes (it never fires on them).
pub(crate) fn primary_relooper_if_needed(
    validation_error: &str,
    spv: &[u8],
    tmp: &Path,
) -> Option<Vec<u8>> {
    if native::classify_validation_error(validation_error)
        != native::ValidationClass::CfgStructurization
    {
        return None;
    }
    let candidate = relooper_candidate(spv)?;
    (candidate != spv && tools::spirv_val_bytes(&candidate, tmp).is_ok()).then_some(candidate)
}

/// Promote the production `val-cfg:raw_then_relooper` (+ PSB) cascade into the primary-validated path
/// when the primary fails a CFG structurization rule that plain relooper / prune-then-relooper cannot
/// rewrite on the primary bytes. The frontier `selection:cross-arm-shared` residual
/// (`08/2d2d8c4f`, `10/1754ce19`) admits under `structured_plan` but still emits a validator-invalid
/// module (structured-exit / undefined-id SSA order); production ships via raw re-emit → relooper →
/// PSB. This helper reuses that exact retry composition, validation-gated and not name-keyed.
pub(crate) fn primary_cfg_raw_then_relooper_if_needed(
    validation_error: &str,
    spv: &[u8],
    retry: &retry::RetryCtx<'_>,
) -> Option<Vec<u8>> {
    if native::classify_validation_error(validation_error)
        != native::ValidationClass::CfgStructurization
    {
        return None;
    }
    retry
        .census(
            "val-cfg:relooper_then_value_select",
            retry.relooper_then_value_select(spv),
        )
        .or_else(|| retry.census("val-cfg:raw_then_relooper", retry.raw_then_relooper()))
}

/// Promote the existing `val-cfg:prune_then_relooper` retry only when the primary fails a CFG
/// structural rule and the ordinary relooper cannot validate it. Function-constant pruning can remove
/// dead branches that make a large reducer exceed the ordinary relooper's scope; the capped reloop of
/// the smaller, reachable CFG is already the production retry form. Checking the ordinary relooper
/// first preserves that retry's priority for every case it can already repair on the full cascade;
/// the primary-validated path adopts the plain relooper via [`primary_relooper_if_needed`] instead.
pub(crate) fn primary_cfg_prune_then_relooper_if_needed(
    validation_error: &str,
    spv: &[u8],
    tmp: &Path,
) -> Option<Vec<u8>> {
    if native::classify_validation_error(validation_error)
        != native::ValidationClass::CfgStructurization
    {
        return None;
    }
    if relooper_candidate(spv)
        .is_some_and(|candidate| tools::spirv_val_bytes(&candidate, tmp).is_ok())
    {
        return None;
    }
    let mut pruned_module = prune_constant_branches_candidate(spv)?;
    let pruned = module_bytes(&pruned_module);
    if pruned == spv {
        return None;
    }
    native::rewrite_to_relooper_module_capped(
        &mut pruned_module,
        native::PRUNE_THEN_RELOOPER_MAX_BLOCKS,
    )
    .ok()?;
    let primary = module_bytes(&pruned_module);
    (primary != spv && tools::spirv_val_bytes(&primary, tmp).is_ok()).then_some(primary)
}

/// Promote the narrow R2 construct-tree repair before broader CFG rewrites when the primary fails a
/// structured-CFG validation rule. The retry itself re-emits from source and independently validates
/// the finished module, so this helper only gates it to the relevant error class and preserves the
/// existing tier-census label.
pub(crate) fn primary_construct_tree_if_needed(
    validation_error: &str,
    retry: &retry::RetryCtx<'_>,
) -> Option<Vec<u8>> {
    if native::classify_validation_error(validation_error)
        != native::ValidationClass::CfgStructurization
    {
        return None;
    }
    retry.census("val-cfg:construct_tree", retry.construct_tree_retry())
}

/// Promote the narrow inline/SROA retry for a dynamic structure index when it validates directly, or
/// compose it with the relooper only after its error class changes to CFG structurization. The
/// source-level index remains dynamic on the raw primary, so no pointer retype is guessed: the
/// existing inline pass removes the caller-owned local byte view, and the existing relooper
/// restructures the resulting otherwise-valid CFG. Any other first or second validation class
/// declines.
pub(crate) fn primary_dynamic_struct_index_inline_sroa_relooper_if_needed(
    validation_error: &str,
    retry: &retry::RetryCtx<'_>,
) -> Option<Vec<u8>> {
    if native::classify_validation_error(validation_error) != native::ValidationClass::Other
        || !native::is_dynamic_struct_index_error(validation_error)
    {
        return None;
    }
    let inlined = tools::emit_vulkan_spirv_inline_sroa_with_sidecar(
        retry.san_ll,
        retry.tmp,
        retry.kern,
        retry.entry_name,
        stage_buffer_layouts(retry.stage, retry.frag, retry.vert, retry.kern),
    )
    .and_then(|spv| retry.finish(spv))
    .ok()?;
    match tools::spirv_val_bytes(&inlined, retry.tmp) {
        Ok(()) => return Some(inlined),
        Err(inline_error)
            if native::classify_validation_error(&inline_error)
                == native::ValidationClass::CfgStructurization => {}
        Err(_) => return None,
    }
    let primary = relooper_candidate(&inlined)?;
    tools::spirv_val_bytes(&primary, retry.tmp)
        .is_ok()
        .then_some(primary)
}

/// Promote the existing inline+SROA+raw retry only when the raw primary's exact validator rule is an
/// illegal Logical pointer operand. Inlining exposes the float-CAS helper's pointer at the caller and
/// raw addressing represents its `float`/`i32` reinterpret without `OpBitcast`; the candidate is
/// adopted only after its own validation. The no-retry byte baseline remains on the original emission.
pub(crate) fn primary_logical_pointer_inline_sroa_raw_if_needed(
    validation_error: &str,
    retry: &retry::RetryCtx<'_>,
) -> Option<Vec<u8>> {
    if native::classify_validation_error(validation_error) != native::ValidationClass::Other
        || !native::is_logical_pointer_operand_error(validation_error)
    {
        return None;
    }
    let primary = tools::emit_vulkan_spirv_inline_sroa_raw_with_sidecar(
        retry.san_ll,
        retry.tmp,
        retry.kern,
        retry.entry_name,
        stage_buffer_layouts(retry.stage, retry.frag, retry.vert, retry.kern),
    )
    .and_then(|spv| retry.finish(spv))
    .ok()?;
    tools::spirv_val_bytes(&primary, retry.tmp)
        .is_ok()
        .then_some(primary)
}

/// Move a stale opaque-image select/phi into its pure sampling/query consumers after the raw primary
/// reports either the broad `Other` select diagnostic or the exact pointer-result/image-incoming phi
/// mismatch. The post-emit pass requires a fully image-only merge and rejects every write, sparse,
/// opaque, or non-dominating use, so it cannot turn a pointer repair into a resource-policy guess.
/// The result is retained only after independent spirv-val.
/// `finish_module` has already loaded, transformed, and canonicalized this module; clone that
/// retained carrier for the speculative rewrite instead of reparsing its just-assembled bytes.
pub(crate) fn primary_opaque_image_select_module_if_needed(
    validation_error: &str,
    module: &Module,
    spv: &[u8],
    tmp: &Path,
) -> Option<Vec<u8>> {
    let class = native::classify_validation_error(validation_error);
    let opaque_merge_mismatch = (class == native::ValidationClass::PointerTyping
        && validation_error.contains("OpPhi's result type")
        && validation_error.contains("does not match incoming value"))
        || (validation_error.contains("Expected both objects to be of Result Type: Select")
            && validation_error.contains("_ptr_UniformConstant_"));
    if class != native::ValidationClass::Other && !opaque_merge_mismatch {
        return None;
    }
    let mut primary_module = module.clone();
    native::rewrite_opaque_image_selects_module(&mut primary_module).ok()?;
    let primary = primary_module
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    (primary != spv && tools::spirv_val_bytes(&primary, tmp).is_ok()).then_some(primary)
}

/// PRIMARY-emit FC prune for modules that CALL a bodiless `llvm.agx*` hardware-intrinsic declaration.
/// Such a call is never executable on a Vulkan target (no
/// lowering exists, the declaration has no body) and is frequently spirv-val-INVALID — the
/// declaration's generic pointer param types mismatch the concrete call args. The affected kernels
/// are Metal function-constant specialized with the agx arm FC-dead (function constants are modeled
/// at their disabled default, matching how the goldens were captured), so the static constant-branch
/// prune — the SAME transform the `val-other:prune` retry applies post-validation-failure — removes
/// the arm, and the primary emit produces the bytes that retry ships today. Triggered structurally
/// by [`native::has_bodiless_agx_call_module`] (the `llvm.agx` ABI-symbol namespace, never a shader
/// name). The result is retained only when that exact call is gone after pruning, so an AGX call in
/// LIVE code cannot cause an unrelated constant branch to alter the primary candidate.
fn apply_agx_fc_prune_module(module: &mut Module, retained_global_ids: &[spirv::Word]) -> bool {
    if !native::has_bodiless_agx_call_module(module) {
        return false;
    }
    let original = module.clone();
    let changed = native::prune_constant_branches_module_preserving(module, retained_global_ids);
    if !changed || native::has_bodiless_agx_call_module(module) {
        *module = original;
        false
    } else {
        native::close_module_capabilities(module);
        true
    }
}
