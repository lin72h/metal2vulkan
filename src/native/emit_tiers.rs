//! Emitter entry points: the default Logical typed emit ([`emit_vulkan_spirv`]) plus the
//! adopt-if-validates retry-tier variants (inline+SROA, all-buffers-raw, workgroup-raw, BDA,
//! relooper-feed). Each variant composes pre-parse AIR lowerings + parse + [`Emitter`] with a
//! different buffer/CFG model; `lib.rs`'s retry cascade selects among them on emit/validation
//! failure. See [`super::error_class`] for the routing classifiers.

use super::*;
use crate::meta;

pub fn emit_vulkan_spirv(san_ll: &str) -> Result<Vec<u8>, String> {
    let kern = meta::parse_air_kernel_meta(san_ll);
    let entry_name = meta::entry_name(san_ll, "kernel");
    Ok(emit_vulkan_spirv_with_sidecar(
        san_ll,
        kern.as_ref(),
        entry_name.as_deref(),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
    )?
    .into_bytes())
}

/// Test adapter for CFG synthesis: parse the function shell so parameters/module state follow the
/// ordinary path, replace its body with the finalized typed carriers, then run the unchanged emitter.
/// This avoids serializing a carrier through the deliberately partial debug-text renderer and proves
/// synthetic blocks are consumable by the real graph-driven emission substrate.
#[cfg(test)]
pub(in crate::native) fn emit_vulkan_spirv_from_typed_blocks(
    function_shell: &str,
    blocks: Vec<crate::native::cfg::BodyBlock>,
) -> Result<Vec<u8>, String> {
    let mut parsed = LlModule::parse(function_shell)?;
    let [function] = parsed.functions.as_mut_slice() else {
        return Err(format!(
            "native emitter: typed-block test adapter needs exactly one function, got {}",
            parsed.functions.len()
        ));
    };
    function.blocks = blocks;
    Ok(finalize_emission(Emitter::new(parsed), None)?.into_bytes())
}

pub(crate) fn emit_vulkan_spirv_with_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    emit_vulkan_spirv_inner(san_ll, false, kern, entry_name, buffer_layouts)
}

/// Re-emit with the reject-only construct-tree own-arm candidate enabled. This is a retry tier, not
/// the default path: the caller adopts the result only if the whole finished module validates.
pub fn emit_vulkan_spirv_construct_tree(san_ll: &str) -> Result<Vec<u8>, String> {
    let kern = meta::parse_air_kernel_meta(san_ll);
    let entry_name = meta::entry_name(san_ll, "kernel");
    Ok(emit_vulkan_spirv_construct_tree_with_sidecar(
        san_ll,
        kern.as_ref(),
        entry_name.as_deref(),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
    )?
    .into_bytes())
}

pub(crate) fn emit_vulkan_spirv_construct_tree_with_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    let san_ll = async_copy::lower_simdgroup_async_copy(san_ll);
    let san_ll = vec_scalar_merge::lower_vector_scalar_pointer_merge(&san_ll);
    let parsed = LlModule::parse_with_stage_meta(&san_ll, kern, entry_name)?;
    finalize_emission(Emitter::new(parsed).with_construct_tree(), buffer_layouts)
}

/// Re-emit with the narrowly scoped primitive metadata inference for a cross-buffer pointer phi.
/// This is NOT a normal emitter mode: callers must first prove the raw primary has the relevant
/// pointer-typing failure, then independently validate the result before adoption.
pub fn emit_vulkan_spirv_with_primitive_phi_metadata(san_ll: &str) -> Result<Vec<u8>, String> {
    let kern = meta::parse_air_kernel_meta(san_ll);
    let entry_name = meta::entry_name(san_ll, "kernel");
    Ok(emit_vulkan_spirv_with_primitive_phi_metadata_sidecar(
        san_ll,
        kern.as_ref(),
        entry_name.as_deref(),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
    )?
    .into_bytes())
}

pub(crate) fn emit_vulkan_spirv_with_primitive_phi_metadata_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    emit_vulkan_spirv_inner(san_ll, true, kern, entry_name, buffer_layouts)
}

fn emit_vulkan_spirv_inner(
    san_ll: &str,
    primitive_phi_metadata: bool,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    // Lower `air.simdgroup_async_copy_2d` (+ its event/wait pair) to an explicit strided tile copy
    // before parse. Modules that also need pointer cleanup are handled by the ordinary retry tiers,
    // which see this lowering because the production entry applies it before re-emission. This
    // entry's copy is retained for direct callers; already-lowered text is a no-op guard. See
    // `async_copy` and its structural regression tests. Floor-safe: only fires on async-copy modules,
    // which fail the emitter outright otherwise.
    let san_ll = async_copy::lower_simdgroup_async_copy(san_ll);
    // Scalarize any scalar/vector pointer-merge before parse (floor-safe: a no-op unless the module
    // carries a `<N x T>*`/`T*` merge the emitter rejects outright). See `vec_scalar_merge`.
    let san_ll = vec_scalar_merge::lower_vector_scalar_pointer_merge(&san_ll);
    let parsed = if primitive_phi_metadata {
        LlModule::parse_with_primitive_phi_metadata_and_stage_meta(&san_ll, kern, entry_name)?
    } else {
        LlModule::parse_with_stage_meta(&san_ll, kern, entry_name)?
    };
    finalize_emission(Emitter::new(parsed), buffer_layouts)
}

fn finalize_emission(
    emitter: Emitter,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    let (mut module, sidecar) = emitter.emit_with_sidecar(buffer_layouts)?;
    add_native_module_capabilities(&mut module);
    Ok(crate::emit_sidecar::EmittedSpirv { module, sidecar })
}

/// Emit after INLINING non-recursive internal helper calls and promoting the resulting entry-stored,
/// non-escaping Function allocas (scalar-replacement + aggregate round-trip fold). This reaches the
/// MPS NDArray multi-destination kernels (TopK) that stage a device-buffer-pointer array through a
/// Function `MPSNDArrays` struct forwarded by value into a helper that reads the array_ref at a static
/// index: in Logical SPIR-V a pointer cannot live in memory/a struct member, so the default emit bails
/// `missing pointer storage` on the Function-pointer store. Inlining collapses the helper boundary and
/// SROA store-forwards the staging away, leaving the device buffer pointer used directly — a shape the
/// emitter accepts. Byte-neutral (inlining + store-forwarding preserve semantics exactly) and
/// floor-safe (an adopt-if-validates retry tier; both passes are no-ops on a module lacking the shape).
/// See the internal `inline` and `sroa` modules.
pub fn emit_vulkan_spirv_inline_sroa(san_ll: &str) -> Result<Vec<u8>, String> {
    let kern = meta::parse_air_kernel_meta(san_ll);
    let entry_name = meta::entry_name(san_ll, "kernel");
    Ok(emit_vulkan_spirv_inline_sroa_with_sidecar(
        san_ll,
        kern.as_ref(),
        entry_name.as_deref(),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
    )?
    .into_bytes())
}

pub(crate) fn emit_vulkan_spirv_inline_sroa_with_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    let inlined = inline::inline_nonrecursive_internal_calls(san_ll);
    let sroad = sroa::promote_entry_allocas_and_fold_aggregates(&inlined);
    let san_ll = vec_scalar_merge::lower_vector_scalar_pointer_merge(&sroad);
    let parsed = LlModule::parse_with_stage_meta(&san_ll, kern, entry_name)?;
    finalize_emission(Emitter::new(parsed), buffer_layouts)
}

pub(crate) fn emit_vulkan_spirv_pointer_select_consumer_inline_with_sidecar(
    san_ll: &str,
    selected_pointer: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    let inlined = inline::inline_pointer_select_consumer(san_ll, entry_name, selected_pointer);
    let parsed = LlModule::parse_with_stage_meta(&inlined, kern, entry_name)?;
    // Inserting a multiblock helper at a callsite can make the ordinary structurizer's local clone
    // heuristics expand a large entry CFG. Emit the exact rewritten blocks as a bounded relooper feed;
    // the retry owns the whole-module structurization and adopts only validating output.
    finalize_emission(Emitter::new(parsed).with_relooper_feed(), buffer_layouts)
}

/// Like [`emit_vulkan_spirv_inline_sroa`], but ALSO models every device/constant buffer raw (the
/// word-`RuntimeArray` byte-offset model) before emission. After inlining + SROA collapse the helper
/// boundary, the surviving device buffers are byte-addressed (`i8*`/`RuntimeArray<uchar>`) and the
/// kernel loads/stores typed values (float/uint) at byte offsets — a reinterpret the Logical typed
/// emit cannot express (it would need an illegal pointer bitcast, so it either emits an invalid
/// `OpLoad %float` from a `uchar*` or bails `cannot reinterpret load of byte pointer`). The raw model
/// declares those buffers as `RuntimeArray<uint>` and forms each typed access by word offset + value
/// bitcast — byte-exact and Logical-legal. This is the composition the TopK multi-destination kernels
/// need: the inline+SROA lowers the `%10` device-pointer array to a direct select over byte buffers,
/// and the raw model makes the byte-offset float/uint traffic on those buffers valid. Adopt-if-
/// validates (a retry tier), so floor-safe by construction.
pub fn emit_vulkan_spirv_inline_sroa_raw(san_ll: &str) -> Result<Vec<u8>, String> {
    let kern = meta::parse_air_kernel_meta(san_ll);
    let entry_name = meta::entry_name(san_ll, "kernel");
    Ok(emit_vulkan_spirv_inline_sroa_raw_with_sidecar(
        san_ll,
        kern.as_ref(),
        entry_name.as_deref(),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
    )?
    .into_bytes())
}

pub(crate) fn emit_vulkan_spirv_inline_sroa_raw_with_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    let inlined = inline::inline_nonrecursive_internal_calls(san_ll);
    let sroad = sroa::promote_entry_allocas_and_fold_aggregates(&inlined);
    let san_ll = vec_scalar_merge::lower_vector_scalar_pointer_merge(&sroad);
    let mut parsed = LlModule::parse_with_stage_meta(&san_ll, kern, entry_name)?;
    mark_all_device_buffers_raw(&mut parsed, false);
    finalize_emission(Emitter::new(parsed), buffer_layouts)
}

/// Like [`emit_vulkan_spirv_inline_sroa_raw`], but ALSO enables the R2 cross-arm restructure
/// (`with_cfg_restructure`) so a function whose (inlined+raw) CFG `structured_plan` rejects gets its
/// cross-arm shared regions privatized and its returns unified before emission — the raw byte model
/// alone leaves the CFG untouched, so a straddle/cross-arm reject falls to the naive `infer_*` merges
/// and emits an invalid structured module (`Block N is already a merge block` / `branches to the
/// selection construct, but not to the header`). This is the missing composition for the
/// straddle-loop-merge + cross-binding-phi cluster (`MPSRNNBreakUpToOutputVecs`/05): the cross-arm
/// restructure fixes the CFG, the raw model makes the byte-offset device traffic Logical-legal, and a
/// following PSB rewrite dissolves the cross-binding pointer phi into 64-bit addresses. Adopt-if-
/// validates (a retry tier), so floor-safe by construction — a case that emits+validates on the
/// default path never reaches it.
pub fn emit_vulkan_spirv_inline_sroa_raw_cfg_restructure(san_ll: &str) -> Result<Vec<u8>, String> {
    let kern = meta::parse_air_kernel_meta(san_ll);
    let entry_name = meta::entry_name(san_ll, "kernel");
    Ok(
        emit_vulkan_spirv_inline_sroa_raw_cfg_restructure_with_sidecar(
            san_ll,
            kern.as_ref(),
            entry_name.as_deref(),
            kern.as_ref().map(|meta| &meta.buffer_layouts),
        )?
        .into_bytes(),
    )
}

pub(crate) fn emit_vulkan_spirv_inline_sroa_raw_cfg_restructure_with_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    let inlined = inline::inline_nonrecursive_internal_calls(san_ll);
    let sroad = sroa::promote_entry_allocas_and_fold_aggregates(&inlined);
    let san_ll = vec_scalar_merge::lower_vector_scalar_pointer_merge(&sroad);
    let mut parsed = LlModule::parse_with_stage_meta(&san_ll, kern, entry_name)?;
    mark_all_device_buffers_raw(&mut parsed, false);
    finalize_emission(Emitter::new(parsed).with_cfg_restructure(), buffer_layouts)
}

/// Emit with every device/constant (`addrspace(1)`/`addrspace(2)`) buffer pointer param modeled raw
/// (byte-offset access on a `RuntimeArray<uint>` backing, which is view-agnostic). The R4 ground-truth
/// raw retry (`translate`'s pipeline) falls back to this for a module whose default typed emission
/// produces a structurally-valid-but-mistyped buffer access — the dominant pointer-merge frontier
/// class, where the buffer's declared SPIR-V block (its Metal argument-metadata layout) is a
/// divergent type tree from the AIR `getelementptr` view (a field split across two declared members,
/// a scalar buried in a declared sub-struct, or a view that traverses past a declared scalar leaf).
pub fn emit_vulkan_spirv_all_buffers_raw(san_ll: &str) -> Result<Vec<u8>, String> {
    let kern = meta::parse_air_kernel_meta(san_ll);
    let entry_name = meta::entry_name(san_ll, "kernel");
    Ok(emit_vulkan_spirv_all_buffers_raw_with_sidecar(
        san_ll,
        kern.as_ref(),
        entry_name.as_deref(),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
    )?
    .into_bytes())
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_with_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    let san_ll = vec_scalar_merge::lower_vector_scalar_pointer_merge(san_ll);
    let mut parsed = LlModule::parse_with_stage_meta(&san_ll, kern, entry_name)?;
    mark_all_device_buffers_raw(&mut parsed, false);
    finalize_emission(Emitter::new(parsed), buffer_layouts)
}

/// Like [`emit_vulkan_spirv_all_buffers_raw`], but additionally models every threadgroup
/// (`addrspace(3)`) buffer pointer param raw (`RuntimeArray`/concrete-vector byte-offset access). The
/// R4 ground-truth raw retry escalates to this as a SECOND tier: a module whose default typed emission
/// fails with a pointer-typing error first retries with only device/constant buffers raw (the proven
/// first-tier win), and only if THAT raw module also fails spirv-val does it retry with workgroup
/// buffers raw too. Two tiers (rather than one mark that always includes addrspace 3) is what keeps
/// the first-tier win floor-safe: marking a threadgroup buffer raw can break an otherwise-valid raw
/// module (workgroup byte-offset access is not always expressible under Logical addressing), which
/// would un-adopt a first-tier raw result and regress it. Each tier is adopted only if it
/// independently validates, so the escalation can only add wins, never remove them.
pub fn emit_vulkan_spirv_all_buffers_raw_with_workgroup(san_ll: &str) -> Result<Vec<u8>, String> {
    let kern = meta::parse_air_kernel_meta(san_ll);
    let entry_name = meta::entry_name(san_ll, "kernel");
    Ok(emit_vulkan_spirv_all_buffers_raw_with_workgroup_sidecar(
        san_ll,
        kern.as_ref(),
        entry_name.as_deref(),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
    )?
    .into_bytes())
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_with_workgroup_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    let san_ll = vec_scalar_merge::lower_vector_scalar_pointer_merge(san_ll);
    let mut parsed = LlModule::parse_with_stage_meta(&san_ll, kern, entry_name)?;
    mark_all_device_buffers_raw(&mut parsed, true);
    finalize_emission(Emitter::new(parsed), buffer_layouts)
}

/// Emit the all-device/constant-buffer raw view with the structured-plan attempt forced off (the
/// `relooper_feed` path) so a caller that immediately runs the relooper can rebuild the CFG directly
/// from a guaranteed-unstructured complete module. This is an intermediate-only form:
/// it intentionally omits branch/loop structured merge hints, and it must never be adopted before
/// the relooper (and any required pointer rewrite) independently validates the result.
pub fn emit_vulkan_spirv_all_buffers_raw_relooper_feed(san_ll: &str) -> Result<Vec<u8>, String> {
    let kern = meta::parse_air_kernel_meta(san_ll);
    let entry_name = meta::entry_name(san_ll, "kernel");
    Ok(
        emit_vulkan_spirv_all_buffers_raw_relooper_feed_with_sidecar(
            san_ll,
            kern.as_ref(),
            entry_name.as_deref(),
            kern.as_ref().map(|meta| &meta.buffer_layouts),
        )?
        .into_bytes(),
    )
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_relooper_feed_with_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    let san_ll = vec_scalar_merge::lower_vector_scalar_pointer_merge(san_ll);
    let mut parsed = LlModule::parse_with_stage_meta(&san_ll, kern, entry_name)?;
    mark_all_device_buffers_raw(&mut parsed, false);
    finalize_emission(Emitter::new(parsed).with_relooper_feed(), buffer_layouts)
}

/// Emit with every device/constant buffer modeled raw AND device-pointer (BDA) modeling enabled: a
/// device pointer (`addrspace(1)`) LOADED from a buffer word is its real 64-bit address (an
/// `OpConvertUToPtr` PhysicalStorageBuffer64 pointer), so the kernel can STORE it (a verbatim 8-byte
/// copy) and DEREFERENCE it (`address + struct/array offset`). This is the honest lowering of the
/// "BDA" frontier class — the Apple BVH builders that load a device pointer from one buffer, store it
/// into another, and walk it as a `MTLSWBVH*` struct (the `raw store for Ptr(1) is not covered yet`
/// emit gap). Byte-correct by construction: the stored bytes are the exact loaded address, the deref
/// is `address + offset` with no tag-bit manipulation (verified across the cluster), and the address
/// is a real Vulkan device address under `buffer_device_address`. The default Logical emit is never
/// altered (this is an adopt-if-validates retry tier). The internal emitter's
/// `with_bda_device_pointers` mode owns the address conversion.
pub fn emit_vulkan_spirv_all_buffers_raw_bda(san_ll: &str) -> Result<Vec<u8>, String> {
    let kern = meta::parse_air_kernel_meta(san_ll);
    let entry_name = meta::entry_name(san_ll, "kernel");
    Ok(emit_vulkan_spirv_all_buffers_raw_bda_with_sidecar(
        san_ll,
        kern.as_ref(),
        entry_name.as_deref(),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
    )?
    .into_bytes())
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_bda_with_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    let san_ll = vec_scalar_merge::lower_vector_scalar_pointer_merge(san_ll);
    let mut parsed = LlModule::parse_with_stage_meta(&san_ll, kern, entry_name)?;
    mark_all_device_buffers_raw(&mut parsed, false);
    finalize_emission(
        Emitter::new(parsed).with_bda_device_pointers(),
        buffer_layouts,
    )
}

/// BDA counterpart of [`emit_vulkan_spirv_all_buffers_raw_relooper_feed`]: emit the raw+BDA
/// physical-address model while deliberately skipping CFG repair so the retry cascade can rebuild
/// guarded/unstructured control flow with the relooper before validation/adoption.
pub(crate) fn emit_vulkan_spirv_all_buffers_raw_bda_relooper_feed_with_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    let san_ll = vec_scalar_merge::lower_vector_scalar_pointer_merge(san_ll);
    let mut parsed = LlModule::parse_with_stage_meta(&san_ll, kern, entry_name)?;
    mark_all_device_buffers_raw(&mut parsed, false);
    finalize_emission(
        Emitter::new(parsed)
            .with_bda_device_pointers()
            .with_relooper_feed(),
        buffer_layouts,
    )
}

/// Mark every device/constant (`addrspace(1)`/`addrspace(2)`) buffer pointer param of every function
/// raw in `parsed.raw_buffer_params`. With `include_workgroup`, also marks threadgroup
/// (`addrspace(3)`) buffer params (the second-tier escalation — see
/// [`emit_vulkan_spirv_all_buffers_raw_with_workgroup`]).
fn mark_all_device_buffers_raw(parsed: &mut LlModule, include_workgroup: bool) {
    let keys: Vec<(String, String)> = parsed
        .functions
        .iter()
        .flat_map(|f| {
            f.params.iter().filter_map(move |(name, ty)| {
                let raw = matches!(ty, ir::LlType::Ptr(1 | 2))
                    || (include_workgroup && matches!(ty, ir::LlType::Ptr(3)));
                raw.then(|| (f.name.clone(), name.clone()))
            })
        })
        .collect();
    for key in keys {
        parsed.raw_buffer_params.insert(key);
    }
}
