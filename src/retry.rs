//! The failure-triggered retry cascade for [`crate::translate_sanitized_with_meta`] (refactor S2).
//!
//! When the default typed emission fails — either spirv-val rejects the produced module or the
//! native emitter errors outright — translation escalates through a fixed ladder of retry *tiers*,
//! each of which re-emits or byte-rewrites the module a different way and is **adopted only if its
//! output independently validates** (`adopt-if-VALIDATES`). That discipline is what makes every tier
//! floor-safe: a module that validates on the default path never enters the cascade, so no tier can
//! regress a passing (banked) case; a module that reaches a tier already failed, and a non-validating
//! retry result is discarded. The relative ORDER of the tiers under each error class is load-bearing
//! (e.g. value-select before PSB because MoltenVK cannot compute-pipeline a buffer-device-address
//! module; inline+SROA before the relooper because the relooper's switch-dispatch form is
//! spirv-val-valid but SPIRV-Cross-miscompiled) — the per-tier doc comments are the spec.
//!
//! This module holds only the tier *mechanisms*. The routing (which classes try which tiers, in what
//! order) stays in [`crate::translate_sanitized_with_meta`] as the `match` over
//! [`native::classify_validation_error`] / [`native::classify_emit_error`]. Each tier was a local
//! closure in that function; extracting them here (S2) is a mechanical move — the captured locals
//! (`san_ll`, `tmp`, the meta/stage/options `finish` threads, `retry_debug_on`)
//! became [`RetryCtx`] fields, and each closure body is preserved verbatim.

use crate::spirv_module::{load_bytes, Module};
use crate::{
    finish_module, meta, native, passes, stage_buffer_layouts, tools, FinishRewrites,
    FinishedModule,
};
use std::collections::HashMap;
use std::path::Path;

fn module_bytes(module: &Module) -> Vec<u8> {
    module
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
}

fn load_relooper_module(bytes: &[u8]) -> Result<Module, String> {
    let mut module = load_bytes(bytes).map_err(|error| format!("SPIR-V load: {error:?}"))?;
    native::rewrite_to_relooper_module(&mut module)?;
    Ok(module)
}

/// Shared context for the retry tiers: the sanitized IR, the temp dir spirv-val runs against, the
/// module metadata `finish` re-applies on every re-emit, and the A/B env gates read once at
/// construction. Every tier borrows this immutably (`&self`).
pub(crate) struct RetryCtx<'a> {
    pub(crate) san_ll: &'a str,
    pub(crate) stage: passes::Stage,
    pub(crate) frag: Option<&'a meta::FragMeta>,
    pub(crate) vert: Option<&'a meta::VertMeta>,
    pub(crate) kern: Option<&'a meta::KernMeta>,
    pub(crate) promoted_kern: Option<&'a meta::KernMeta>,
    pub(crate) entry_name: Option<&'a str>,
    pub(crate) tmp: &'a Path,
    pub(crate) options: passes::TransformOptions,
    pub(crate) air_data_layout: Option<crate::layout::AirDataLayout>,
    /// `METAL2VULKAN_RETRY_DEBUG` present — trace each tier's emit/validate outcome to stderr.
    pub(crate) retry_debug_on: bool,
    /// Whether the W1 PhysicalStorageBuffer64 tiers are enabled. Always `true` on every production
    /// translate; the only caller that passes `false` is the `--psb-dump` diagnostic probe
    /// ([`crate::translate_pre_psb_probe`]), which wants the pre-PSB emission so it can apply the PSB
    /// rewrite by hand for inspection.
    psb_retry_enabled: bool,
    /// M-C1 tier-adoption census: the label of the last cascade invocation site whose result was
    /// adopted (set by [`Self::census`]). `None` means no tier adopted (default emission kept). Only
    /// meaningful when `METAL2VULKAN_TIER_CENSUS` is set; the census wrapping is a pure passthrough so
    /// emitted bytes are unchanged whether or not it is on.
    adopted_tier: std::cell::Cell<Option<&'static str>>,
    /// Whether the large-CFG feed's last validation failure is a control-flow shape the global
    /// construct tree can change. A pointer/type failure survives CFG-only re-emission, so skipping
    /// that second full-module candidate avoids a guaranteed-futile memory spike.
    large_cfg_construct_tree_eligible: std::cell::Cell<bool>,
    /// M-C2 raw re-emit cache. The "every device/constant buffer modeled raw" re-emission
    /// (`emit_vulkan_spirv_all_buffers_raw` + `finish`) is the shared front of `raw_retry`,
    /// `raw_psb`, and `raw_then_relooper` — up to three cascade sites, each also escalating to the
    /// `_with_workgroup` variant, so a single cascade recomputed the same expensive native emit up to
    /// ~6×. Memoize each variant's serialized `Result` once per translate; every site reuses it.
    /// Byte-neutral on the deterministic majority (identical input → identical `finish`), and for a
    /// byte-nondeterministic row it merely makes the three sites agree on ONE valid raw form instead
    /// of drawing three independent samples — both forms are adopt-if-VALIDATES, so pass/fail (hence
    /// G4/G5) is unchanged. `None` = not computed yet; `Some(Err)` caches an emit failure too.
    raw_reemit_cache: std::cell::RefCell<Option<Result<Vec<u8>, String>>>,
    raw_reemit_wg_cache: std::cell::RefCell<Option<Result<Vec<u8>, String>>>,
    /// M-C2 counterpart for BDA re-emission. `bda_retry` runs before `bda_then_relooper`; caching lets
    /// the second tier skip a full BDA feed when the ordinary BDA emit already failed in the
    /// straight-line graph walk.
    bda_reemit_cache: std::cell::RefCell<Option<Result<Vec<u8>, String>>>,
    /// The construct-tree tier can be reached once by the large-CFG pre-route and again after the
    /// ordinary primary identifies the same CFG validation class. Its source re-emission, finish,
    /// and internal retry chain are deterministic for this translation, so retain the final
    /// adopt/decline verdict and never repeat that work. A declined tier caches only `None`; a
    /// successful tier retains its serialized bytes, not an owned module graph.
    construct_tree_retry_cache: std::cell::RefCell<Option<Option<Vec<u8>>>>,
}

impl<'a> RetryCtx<'a> {
    fn buffer_layouts(&self) -> Option<&'a HashMap<u32, meta::AirType>> {
        stage_buffer_layouts(self.stage, self.frag, self.vert, self.kern)
    }

    /// Build the context, reading the A/B env gates once (as the function entry did). `psb_retry_enabled`
    /// is passed explicitly (always `true` in production) rather than read from env, so the `--psb-dump`
    /// diagnostic can request the pre-PSB emission without an env flip.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        san_ll: &'a str,
        stage: passes::Stage,
        frag: Option<&'a meta::FragMeta>,
        vert: Option<&'a meta::VertMeta>,
        kern: Option<&'a meta::KernMeta>,
        promoted_kern: Option<&'a meta::KernMeta>,
        entry_name: Option<&'a str>,
        tmp: &'a Path,
        options: passes::TransformOptions,
        psb_retry_enabled: bool,
        air_data_layout: Option<crate::layout::AirDataLayout>,
    ) -> Self {
        RetryCtx {
            san_ll,
            stage,
            frag,
            vert,
            kern,
            promoted_kern,
            entry_name,
            tmp,
            options,
            air_data_layout,
            retry_debug_on: crate::env_vars::retry_debug(),
            psb_retry_enabled,
            adopted_tier: std::cell::Cell::new(None),
            large_cfg_construct_tree_eligible: std::cell::Cell::new(true),
            raw_reemit_cache: std::cell::RefCell::new(None),
            raw_reemit_wg_cache: std::cell::RefCell::new(None),
            bda_reemit_cache: std::cell::RefCell::new(None),
            construct_tree_retry_cache: std::cell::RefCell::new(None),
        }
    }

    /// Memoized "all device/constant buffers raw" re-emission (`emit_vulkan_spirv_all_buffers_raw` +
    /// `finish`), computed once per translate and shared by `raw_retry`/`raw_psb`/`raw_then_relooper`.
    /// Retain only serialized bytes, not the much larger owned module graph, so later retry tiers can
    /// build one candidate at a time within the per-translation memory ceiling.
    fn raw_reemit_finished(&self) -> Result<Vec<u8>, String> {
        let mut cache = self.raw_reemit_cache.borrow_mut();
        if cache.is_none() {
            *cache = Some(
                tools::emit_vulkan_spirv_all_buffers_raw_with_sidecar(
                    self.san_ll,
                    self.tmp,
                    self.kern,
                    self.entry_name,
                    self.buffer_layouts(),
                )
                .and_then(|b| self.finish(b)),
            );
        }
        cache.as_ref().unwrap().clone()
    }

    fn raw_reemit(&self) -> Result<Vec<u8>, String> {
        self.raw_reemit_finished()
    }

    /// Memoized `_with_workgroup` escalation of [`Self::raw_reemit_finished`] (threadgroup buffers
    /// modeled raw too). Shared by the same three sites' `or_else` fallbacks.
    fn raw_reemit_wg_finished(&self) -> Result<Vec<u8>, String> {
        let mut cache = self.raw_reemit_wg_cache.borrow_mut();
        if cache.is_none() {
            *cache = Some(
                tools::emit_vulkan_spirv_all_buffers_raw_with_workgroup_sidecar(
                    self.san_ll,
                    self.tmp,
                    self.kern,
                    self.entry_name,
                    self.buffer_layouts(),
                )
                .and_then(|b| self.finish(b)),
            );
        }
        cache.as_ref().unwrap().clone()
    }

    fn raw_reemit_wg(&self) -> Result<Vec<u8>, String> {
        self.raw_reemit_wg_finished()
    }

    fn bda_reemit_finished(&self) -> Result<Vec<u8>, String> {
        let mut cache = self.bda_reemit_cache.borrow_mut();
        if cache.is_none() {
            *cache = Some(
                tools::emit_vulkan_spirv_all_buffers_raw_bda_with_sidecar(
                    self.san_ll,
                    self.tmp,
                    self.kern,
                    self.entry_name,
                    self.buffer_layouts(),
                )
                .and_then(|b| self.finish(b)),
            );
        }
        cache.as_ref().unwrap().clone()
    }

    fn bda_reemit(&self) -> Result<Vec<u8>, String> {
        self.bda_reemit_finished()
    }

    /// Raw re-emission whose rejected-CFG repair is intentionally skipped because this caller feeds
    /// the bytes straight to the relooper. This is a fallback only after both normal raw emissions
    /// exhausted their repair budget, so existing raw retry output remains byte-identical.
    fn raw_reemit_relooper_feed(&self) -> Result<Vec<u8>, String> {
        tools::emit_vulkan_spirv_all_buffers_raw_relooper_feed_with_sidecar(
            self.san_ll,
            self.tmp,
            self.kern,
            self.entry_name,
            self.buffer_layouts(),
        )
        .and_then(|b| self.finish(b))
    }

    /// Fast composition for a validator-proven raw-buffer typing failure that also needs CFG
    /// structurization. The feed emitter deliberately omits the native structured-plan search: its
    /// output is consumed immediately by the relooper, which reconstructs structured control flow
    /// from the same reachable CFG. As with every retry, no bytes are adopted unless the resulting
    /// module independently validates.
    pub(crate) fn raw_feed_then_relooper(&self) -> Option<Vec<u8>> {
        let raw_feed = self.raw_reemit_relooper_feed();
        if self.retry_debug_on {
            if let Err(e) = &raw_feed {
                eprintln!("[retry-debug] raw_feed_then_relooper: raw feed emit failed: {e}");
            }
        }
        raw_feed.ok().and_then(|mut bytes| {
            if self.record_large_cfg_feed_validation("raw_feed_then_relooper(feed)", &bytes) {
                return Some(bytes);
            }
            // Source-level block counts can fit the relooper while interface/CFG legalization
            // expands the emitted function beyond its hard cap. A non-CFG validator error from
            // that feed must not suppress construct-tree: the relooper cannot consume this graph,
            // and construct-tree is still the sole owner of its source-level control flow.
            if emitted_graph_exceeds_relooper_cap(&bytes) {
                self.large_cfg_construct_tree_eligible.set(true);
            }
            // Interface lowering can turn an AIR pointer phi over texture parameters into an
            // image-object phi while retaining the stale pointer result type. Lower that proven
            // image-only closure into value-domain reads before the relooper analyzes pointer SSA;
            // otherwise it may rematerialize an already-invalid pointer carrier.
            if let Ok(mut module) = load_bytes(&bytes) {
                if native::rewrite_opaque_image_selects_module(&mut module).is_ok() {
                    bytes = module_bytes(&module);
                    if self.record_large_cfg_feed_validation(
                        "raw_feed_then_relooper(feed+opaque_image)",
                        &bytes,
                    ) {
                        return Some(bytes);
                    }
                    if emitted_graph_exceeds_relooper_cap(&bytes) {
                        self.large_cfg_construct_tree_eligible.set(true);
                    }
                }
            }
            self.adopt_relooped_raw(&bytes)
        })
    }

    fn record_large_cfg_feed_validation(&self, label: &str, bytes: &[u8]) -> bool {
        match tools::spirv_val_bytes(bytes, self.tmp) {
            Ok(()) => {
                self.large_cfg_construct_tree_eligible.set(false);
                true
            }
            Err(error) => {
                self.large_cfg_construct_tree_eligible.set(
                    native::classify_validation_error(&error)
                        == native::ValidationClass::CfgStructurization,
                );
                if self.retry_debug_on {
                    let head = error.lines().take(4).collect::<Vec<_>>().join(" | ");
                    eprintln!("[retry-debug] {label}: emitted, spirv-val failed: {head}");
                }
                false
            }
        }
    }

    pub(crate) fn large_cfg_construct_tree_eligible(&self) -> bool {
        self.large_cfg_construct_tree_eligible.get()
    }

    /// M-C1 census passthrough: tag `r` with the invocation-site label `tier` when it is `Some` (a
    /// tier adopted), then return `r` unchanged. Because the `or_else` arms it wraps are atomic
    /// (each yields `Some` only when its whole chain validated), exactly one `census` call records
    /// per translate — the adopting site. A pure identity on the bytes, so BC is unaffected.
    pub(crate) fn census(&self, tier: &'static str, r: Option<Vec<u8>>) -> Option<Vec<u8>> {
        if r.is_some() {
            self.adopted_tier.set(Some(tier));
        }
        r
    }

    /// The label recorded by the last adopting [`Self::census`] this translate, or `None` if no tier
    /// adopted (default emission kept / hard fallback).
    pub(crate) fn adopted_tier(&self) -> Option<&'static str> {
        self.adopted_tier.get()
    }

    /// Run the passes layer + id-canonicalization over freshly-emitted SPIR-V, re-applying this
    /// module's stage/metadata/options. Every tier that re-emits threads its bytes through here.
    pub(crate) fn finish(
        &self,
        emitted: crate::emit_sidecar::EmittedSpirv,
    ) -> Result<Vec<u8>, String> {
        self.finish_carrier(emitted).map(|finished| finished.bytes)
    }

    fn finish_carrier(
        &self,
        emitted: crate::emit_sidecar::EmittedSpirv,
    ) -> Result<FinishedModule, String> {
        self.finish_carrier_with_rewrites(emitted, FinishRewrites::Plain)
    }

    fn finish_carrier_with_rewrites(
        &self,
        emitted: crate::emit_sidecar::EmittedSpirv,
        rewrites: FinishRewrites,
    ) -> Result<FinishedModule, String> {
        finish_module(
            emitted,
            self.stage,
            self.frag,
            self.vert,
            self.kern,
            self.entry_name,
            self.air_data_layout.as_ref(),
            self.options,
            rewrites,
        )
    }

    pub(crate) fn finish_primary_carrier(
        &self,
        emitted: crate::emit_sidecar::EmittedSpirv,
    ) -> Result<FinishedModule, String> {
        self.finish_carrier_with_rewrites(emitted, FinishRewrites::Primary)
    }

    /// True iff `bytes` independently passes spirv-val — the adopt-if-VALIDATES gate every tier uses.
    fn validates(&self, bytes: &[u8]) -> bool {
        tools::spirv_val_bytes(bytes, self.tmp).is_ok()
    }

    /// [`Self::validates`] with `METAL2VULKAN_RETRY_DEBUG` tracing of the pass/fail and the first
    /// spirv-val error lines.
    pub(crate) fn validates_dbg(&self, name: &str, bytes: &[u8]) -> bool {
        match tools::spirv_val_bytes(bytes, self.tmp) {
            Ok(()) => {
                if self.retry_debug_on {
                    eprintln!("[retry-debug] {name}: VALIDATES");
                }
                true
            }
            Err(e) => {
                if self.retry_debug_on {
                    let head: Vec<&str> = e.lines().take(4).collect();
                    eprintln!(
                        "[retry-debug] {name}: emitted, spirv-val failed: {}",
                        head.join(" | ")
                    );
                }
                false
            }
        }
    }

    // R4 ground-truth raw retry (the dominant frontier lever — buffer pointer-merge). The default
    // typed emission resolves a buffer access against the buffer's declared SPIR-V block (its Metal
    // argument-metadata layout), which can be a structurally divergent type tree from the AIR
    // `getelementptr` view; the access chain is then either valid-in-shape-but-mistyped (spirv-val
    // rejects it) or inexpressible (the emitter errors outright). In EITHER failure mode we re-run the
    // whole pipeline with every device/constant buffer modeled raw (byte-offset access, view-agnostic)
    // and adopt it ONLY if it independently validates. Floor-safe by construction: a valid module
    // (every banked case) emits and validates on the default path, so the retry never runs and the
    // bytes are byte-identical; a module reaching the retry already failed (never banked), and the raw
    // result is taken only if it validates, so neither the floor nor a failing module's diagnostics can
    // regress. Degrades to the default outcome if spirv-val is unavailable or the raw re-emit does not
    // validate.
    //
    // Tier 1: device/constant buffers raw (the dominant pointer-merge win). Tier 2: escalate to also
    // modeling threadgroup (addrspace 3) buffers raw, which fixes the over-index class on workgroup
    // buffers tier 1 leaves untouched. Each tier adopted only if it independently validates, so the
    // escalation can only add wins on top of tier 1, never regress one.
    pub(crate) fn raw_retry(&self) -> Option<Vec<u8>> {
        self.raw_reemit()
            .ok()
            .filter(|b| self.validates(b))
            .or_else(|| self.raw_reemit_wg().ok().filter(|b| self.validates(b)))
    }

    /// Whether the normal all-buffer-raw re-emission that a later raw→relooper retry would consume
    /// contains the native wide-offset robust-write guard.  This is a structural eligibility probe:
    /// callers still invoke the ordinary retry tier and adopt its bytes only after spirv-val.  It
    /// deliberately follows the same raw / raw-with-workgroup choice as [`Self::raw_then_relooper`]
    /// before that tier considers its unrepaired-feed last resort.
    pub(crate) fn raw_reemit_has_wide_raw_store_guard(&self) -> bool {
        self.raw_reemit_finished()
            .ok()
            .or_else(|| self.raw_reemit_wg_finished().ok())
            .and_then(|bytes| load_bytes(&bytes).ok())
            .is_some_and(|module| native::module_has_wide_raw_store_guard(&module))
    }

    // M4 phi-the-index retry — an illegal logical-pointer `OpPhi` (a pointer phi in Private/
    // UniformConstant/Function storage, which VariablePointers cannot cover) is rewritten to phi the
    // access-chain INDICES and rematerialize the pointer, when every arm is an access chain into one
    // base. Clones the retained pre-primary module (no re-emit or reparse) and is adopted ONLY if the
    // rewrite independently validates — same adopt-if-VALIDATES floor-safety as the raw/cfg retries: a
    // module that validates on the default path never reaches it, and a non-validating rewrite is
    // discarded.
    pub(crate) fn phi_index_retry(&self, module: &Module) -> Option<Vec<u8>> {
        let mut candidate = module.clone();
        native::rewrite_logical_pointer_phis_retry_module(&mut candidate).ok()?;
        let bytes = candidate
            .assemble()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        self.validates(&bytes).then_some(bytes)
    }

    // Sub-word packed-scalar retry — a Function scalar-integer alloca written through SMALLER-element
    // access chains (the `as_type<uint>(half2)` / `uchar4` pack idiom) emits an illegal
    // `OpInBoundsAccessChain %_ptr_Function_<elem> %var %idx` indexing a scalar ("reached non-composite").
    // Retype the variable to a `<N x elem>` vector so the lane access chains index a vector component
    // (legal) and value-bitcast its whole-word loads/stores. Clones the retained pre-primary module
    // and is adopted ONLY if it independently validates — same adopt-if-VALIDATES floor-safety as the
    // other retries; byte-safe by construction (Function scratch, little-endian-identical layout).
    pub(crate) fn subword_pack_retry(&self, module: &Module) -> Option<Vec<u8>> {
        let mut candidate = module.clone();
        native::rewrite_subword_packed_scalars_module(&mut candidate).ok()?;
        let bytes = candidate
            .assemble()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        self.validates(&bytes).then_some(bytes)
    }

    // Static constant-branch pruning retry — a Metal function-constant-gated optional feature (an
    // optional mask buffer, a `do_causal` flag, ...) compiles to a dead arm that over-indexes the
    // demoted Private placeholder of its unbound buffer ("reached non-composite"). We model function
    // constants at their disabled default (the golden was captured the same way), so the arm never
    // executes: const-fold the predicate branches and DCE the dead code (placeholder GEPs and the
    // pointer phis carrying them included). Operates on the already-emitted default bytes and is
    // adopted ONLY if it independently validates — same adopt-if-VALIDATES floor-safety as the other
    // retries, and the transformation removes only statically-dead code so it is semantics-preserving.
    pub(crate) fn prune_retry(&self, out: &[u8]) -> Option<Vec<u8>> {
        let mut candidate = load_bytes(out).ok()?;
        native::prune_constant_branches_module(&mut candidate).ok()?;
        let bytes = candidate
            .assemble()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        self.validates(&bytes).then_some(bytes)
    }

    // W1 PhysicalStorageBuffer64 retry — a cross-binding pointer `OpSelect`/`OpPhi` (pointers from
    // distinct descriptor bindings, illegal under Logical addressing) is rewritten to the PSB form
    // (merged buffers as physical-address pointers sourced from a synthesized address table). Operates
    // on the already-emitted default bytes and is adopted ONLY if the rewrite independently validates —
    // same adopt-if-VALIDATES floor-safety as the raw/cfg/phi retries.
    // Enabled on every production translate (`psb_retry_enabled`); only `--psb-dump` disables it to
    // inspect the pre-PSB emission.
    pub(crate) fn psb_retry(&self, out: &[u8]) -> Option<Vec<u8>> {
        if !self.psb_retry_enabled {
            return None;
        }
        let mut candidate = load_bytes(out).ok()?;
        native::rewrite_cross_binding_pointer_merges_module(
            &mut candidate,
            self.options.descriptor_layout,
        )
        .ok()?;
        let bytes = candidate
            .assemble()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        self.validates(&bytes).then_some(bytes)
    }

    // Value-domain cross-binding retry — the SAME cross-binding pointer merge, but lowered into the
    // VALUE domain (load from every candidate buffer, select among the loaded VALUES) so the module
    // stays plain Logical `StorageBuffer`. This is byte-exact by construction (the selected value is
    // the exact load Apple performs) and — unlike the PSB lowering — MoltenVK can create a COMPUTE
    // pipeline from it (buffer-device-address access blocks compute-pipeline creation). Preferred over
    // `psb_retry`: tried FIRST so a Logical value-lowered module wins when it validates, with PSB as the
    // fallback for merges this pass bails on (a store through the merge, an opaque use). Operates on the
    // already-emitted default bytes, adopt-if-VALIDATES (same floor-safety as the other retries).
    pub(crate) fn value_select_retry(&self, out: &[u8]) -> Option<Vec<u8>> {
        let bytes = self.value_select_rewrite(out)?;
        self.validates_dbg("value_select", &bytes).then_some(bytes)
    }

    fn value_select_rewrite(&self, out: &[u8]) -> Option<Vec<u8>> {
        let mut candidate = load_bytes(out).ok()?;
        match native::rewrite_cross_binding_pointer_merges_to_values_module(&mut candidate) {
            Ok(()) => Some(module_bytes(&candidate)),
            Err(e) => {
                if self.retry_debug_on {
                    eprintln!("[retry-debug] value_select: bailed: {e}");
                }
                None
            }
        }
    }

    // raw-then-PSB retry — a cross-binding merge whose DEFAULT emission also carries a buffer pointer
    // bitcast (a `device float*` viewed as `uint*`, etc.) that the PSB rewrite cannot model, but whose
    // RAW byte-model emission eliminates the bitcast and leaves only a clean WHOLE-buffer
    // `OpSelect`/`OpPhi` over the buffer variables. The raw emission is then handed to the same PSB
    // rewrite (which models a whole-buffer cross-binding select: each buffer's device address from the
    // address table, ConvertUToPtr to a physical struct pointer, the indexing access chain re-applied
    // physically). Adopt ONLY if it independently validates — same floor-safety as `raw_retry`/
    // `psb_retry`; byte-gated on radv like every PSB module. Enabled on every production translate
    // (`psb_retry_enabled`).
    pub(crate) fn raw_psb(&self) -> Option<Vec<u8>> {
        if !self.psb_retry_enabled {
            return None;
        }
        let raw = self
            .raw_reemit()
            .ok()
            .or_else(|| self.raw_reemit_wg().ok())?;
        let mut raw_module = load_bytes(&raw).ok()?;
        if let Some(valid) =
            self.psb_then_wg_remodel_module(raw_module.clone(), "raw_psb", "raw_psb+wg")
        {
            return Some(valid);
        }
        // The raw model can expose a large but reducible CFG to the PSB rewrite. When the rewrite
        // adds physical-address setup inside that graph, retained merge repair can leave an invalid
        // back-edge even though the same raw traffic is otherwise sound. Structure the raw module
        // first, then apply the identical whole-buffer PSB lowering to the rebuilt graph. This is
        // the same generic raw → relooper → PSB composition used by `raw_then_relooper`; it runs
        // only after the existing raw-PSB result failed validation, so successful pre-relooper output is
        // byte-identical.
        let relooped = native::rewrite_to_relooper_module(&mut raw_module);
        if self.retry_debug_on {
            if let Err(e) = &relooped {
                eprintln!("[retry-debug] raw_psb: relooper rewrite failed: {e}");
            }
        }
        relooped.ok().and_then(|()| {
            self.psb_then_wg_remodel_module(raw_module, "raw_psb(relooped)", "raw_psb(relooped+wg)")
        })
    }

    /// Apply the cross-binding PSB whole-buffer rewrite to `bytes` and adopt if it validates; else run
    /// the Workgroup float-as-int atomic remodel on the PSB result and adopt if THAT validates. Returns
    /// `None` when neither validates, or when the PSB rewrite errors (no cross-binding merge present).
    ///
    /// The second step exists because a module can carry BOTH illegal constructs at once — a
    /// cross-binding device `OpSelect`/`OpPhi` AND a Workgroup float-atomic pointer bitcast (the BVH
    /// `binFragmentsTemporalSplitKernel` shape: it selects among distinct device buffers *and* min/max
    /// a threadgroup float AABB via `OpBitcast %_ptr_Workgroup_int` → `OpAtomicSMin/SMax`). spirv-val
    /// reports one wall per module, so the PSB rewrite dissolves the device select but leaves the WG
    /// bitcast standing; `rewrite_workgroup_atomic_floats_module` then retypes the Workgroup variable's
    /// float leaves to the int the atomics use and the bitcast disappears. Byte-safe by construction
    /// (Workgroup is shader-internal scratch, float↔int32 is a bit-identical 32-bit reinterpret,
    /// layout-preserving), adopt-if-validates. Shared by every PSB-bearing retry tier so any of them can
    /// clear the combined shape — not just `raw_psb` — since the raw/PSB re-emit bypasses the lowering
    /// pipeline where the Ctx-based WG remodel lives.
    pub(crate) fn psb_then_wg_remodel(
        &self,
        bytes: &[u8],
        label_psb: &str,
        label_wg: &str,
    ) -> Option<Vec<u8>> {
        let candidate = load_bytes(bytes).ok()?;
        self.psb_then_wg_remodel_module(candidate, label_psb, label_wg)
    }

    fn psb_then_wg_remodel_module(
        &self,
        mut candidate: Module,
        label_psb: &str,
        label_wg: &str,
    ) -> Option<Vec<u8>> {
        native::rewrite_cross_binding_pointer_merges_module(
            &mut candidate,
            self.options.descriptor_layout,
        )
        .ok()?;
        let psb = candidate
            .assemble()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        if self.validates_dbg(label_psb, &psb) {
            return Some(psb);
        }
        native::rewrite_workgroup_atomic_floats_module(&mut candidate).ok()?;
        let bytes = candidate
            .assemble()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        self.validates_dbg(label_wg, &bytes).then_some(bytes)
    }

    // FC-promote → prune → reconcile → PSB retry — the FC-multiplexed cross-binding device merge
    // (a70fb990-class, `ndArrayConvWinogradWeightsTransform`): the kernel FC-dispatches to one of N
    // per-dtype template variants that access the SAME device buffers at CONFLICTING element types
    // (`float*` in the live variant, `half*`/`bfloat*`/`uchar*` in the dead ones). Unable to pick one
    // pointee, the default emit models those buffers raw and DEMOTES the merge arm to a Private zero
    // placeholder, so the cross-binding pointer merge reads ZEROS — spirv-val-passable but byte-WRONG
    // (proven, see kb). This retry selects the StageMeta projection where FC-gated `air.buffer`
    // params bind as REAL StorageBuffer, so the live variant's buffers hold real device data; then
    // (1) prunes the FC dead arms (only the live variant's merge survives), (2) reconciles the raw
    // byte-0 whole-buffer fallback arm to the merge's scalar pointee (byte-EXACT: element 0 = offset
    // 0 for either scalar element type), and (3) applies the PSB
    // PhysicalStorageBuffer64 cross-binding lowering the arms now admit. Adopt ONLY if the result
    // validates — floor-safe by construction (the default path and every golden are untouched; a golden
    // validates on the default path and never reaches this retry). Byte-correct by construction (real
    // buffers + exact offset-0 reconcile + byte-proven PSB mechanism); `synth=false` so byte-axis
    // [M]/pending like the other PSB-tier clears. Kernel-stage only; `psb_retry_enabled`. Decides
    // purely from IR structure (the FC wrapper ABI marker, block shape, cross-binding merge) — never a
    // shader name.
    pub(crate) fn fc_promote_psb(&self) -> Option<Vec<u8>> {
        if !self.psb_retry_enabled {
            return None;
        }
        if !matches!(self.stage, passes::Stage::Kernel) {
            return None;
        }
        let kern_p = self.promoted_kern?;
        let emit = tools::emit_vulkan_spirv_with_sidecar(
            self.san_ll,
            self.tmp,
            Some(kern_p),
            self.entry_name,
            Some(&kern_p.buffer_layouts),
        )
        .ok()?;
        let mut finished = finish_module(
            emit,
            self.stage,
            None,
            None,
            Some(kern_p),
            self.entry_name,
            self.air_data_layout.as_ref(),
            self.options,
            FinishRewrites::Plain,
        )
        .ok()?;
        native::prune_constant_branches_module(&mut finished.module).ok()?;
        native::reconcile_whole_buffer_scalar_arms_module(&mut finished.module).ok()?;
        native::rewrite_cross_binding_pointer_merges_module(
            &mut finished.module,
            self.options.descriptor_layout,
        )
        .ok()?;
        let psb = finished
            .module
            .assemble()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        Some(psb).filter(|b| self.validates(b))
    }

    // FC-promote logical retry — a function-constant-wrapped `air.buffer` argument can still be
    // genuinely used by the emitted body. The default metadata projection intentionally leaves
    // FC-wrapped buffers absent because some conditional buffers are not valid descriptor bindings.
    // When the default module then fails validation on a Private scalar placeholder access, retry the
    // promoted kernel projection by itself before the heavier raw/PSB tiers. This adopts only if the
    // promoted Logical module independently validates, so default-valid rows stay byte-identical and
    // truly-conditional invalid descriptors remain parked.
    pub(crate) fn fc_promote_logical(&self) -> Option<Vec<u8>> {
        if !matches!(self.stage, passes::Stage::Kernel) {
            return None;
        }
        let kern_p = self.promoted_kern?;
        let emitted = tools::emit_vulkan_spirv_with_sidecar(
            self.san_ll,
            self.tmp,
            Some(kern_p),
            self.entry_name,
            Some(&kern_p.buffer_layouts),
        )
        .and_then(|b| {
            finish_module(
                b,
                self.stage,
                None,
                None,
                Some(kern_p),
                self.entry_name,
                self.air_data_layout.as_ref(),
                self.options,
                FinishRewrites::Plain,
            )
            .map(|finished| finished.bytes)
        });
        if self.retry_debug_on {
            if let Err(error) = &emitted {
                eprintln!("[retry-debug] fc_promote_logical emit/finish failed: {error}");
            }
        }
        let bytes = emitted.ok()?;
        self.validates_dbg("fc_promote_logical", &bytes)
            .then_some(bytes)
    }

    // BDA retry — the `raw store for Ptr(1) is not covered yet` emit gap: an Apple BVH builder that
    // loads a device pointer from one buffer, stores it into another, and dereferences it as a
    // `MTLSWBVH*` struct. Re-emit with every device buffer raw AND device-pointer (BDA) modeling on —
    // the loaded pointer becomes its real 64-bit address (`OpConvertUToPtr` PhysicalStorageBuffer64),
    // the store is a verbatim 8-byte copy, the deref is `address + offset`. Adopt ONLY if it
    // independently validates (floor-safe; the default Logical emit is never touched). Byte-correct by
    // construction (exact loaded address bits, no tag-bit manipulation across the cluster); `synth=false`
    // so byte-axis [M]/pending like the other PSB-tier clears.
    pub(crate) fn bda_retry(&self) -> Option<Vec<u8>> {
        let bytes = match self.bda_reemit() {
            Ok(bytes) => bytes,
            Err(error) => {
                if self.retry_debug_on {
                    eprintln!("[retry-debug] bda emit/finish failed: {error}");
                }
                return None;
            }
        };
        self.validates_dbg("bda", &bytes).then_some(bytes)
    }

    /// BDA + relooper retry: the BDA physical-address model clears raw `store ptr addrspace(1)`
    /// shapes, but the resulting CFG can still need the same relooper repair as the raw retry path.
    /// Emit a BDA relooper-feed intermediate, rebuild the CFG, and adopt
    /// only if the final module validates. Plain [`Self::bda_retry`] stays first so existing BDA wins
    /// keep their bytes. If the ordinary BDA emit already failed in the straight-line typed graph
    /// walk, skip the feed because CFG repair cannot make the missing body opcode emit.
    pub(crate) fn bda_then_relooper(&self) -> Option<Vec<u8>> {
        if let Err(error) = self.bda_reemit_finished() {
            if native::is_graph_walk_unmigrated_emit_error(&error) {
                if self.retry_debug_on {
                    eprintln!(
                        "[retry-debug] bda_then_relooper: bda feed skipped after typed graph-walk emit failure"
                    );
                }
                return None;
            }
        }
        let fed = tools::emit_vulkan_spirv_all_buffers_raw_bda_relooper_feed_with_sidecar(
            self.san_ll,
            self.tmp,
            self.kern,
            self.entry_name,
            self.buffer_layouts(),
        )
        .and_then(|b| self.finish(b));
        if self.retry_debug_on {
            if let Err(error) = &fed {
                eprintln!("[retry-debug] bda_then_relooper: bda feed emit failed: {error}");
            }
        }
        let fed = fed.ok()?;
        let relooped = load_relooper_module(&fed);
        if self.retry_debug_on {
            if let Err(error) = &relooped {
                eprintln!("[retry-debug] bda_then_relooper: relooper rewrite failed: {error}");
            }
        }
        let bytes = module_bytes(&relooped.ok()?);
        self.validates_dbg("bda_then_relooper", &bytes)
            .then_some(bytes)
    }

    // Inline + SROA retry — the MPS NDArray multi-destination (TopK) `missing pointer storage` class:
    // a device-buffer-pointer array staged through a Function `MPSNDArrays` struct forwarded by value
    // into a helper that reads the array_ref at a static index. Inlining the non-recursive helper chain
    // collapses the call boundary and scalar-replacement store-forwards the Function staging away,
    // leaving the device buffer pointer used directly (a Logical-legal shape). Byte-neutral (inlining +
    // store-forwarding preserve semantics), adopt-if-validates so floor-safe.
    pub(crate) fn inline_sroa_retry(&self) -> Option<Vec<u8>> {
        let debug = crate::env_vars::retry_debug();
        let emitted = tools::emit_vulkan_spirv_inline_sroa_with_sidecar(
            self.san_ll,
            self.tmp,
            self.kern,
            self.entry_name,
            self.buffer_layouts(),
        )
        .and_then(|b| self.finish(b));
        if debug {
            match &emitted {
                Ok(bytes) => {
                    if let Err(e) = tools::spirv_val_bytes(bytes, self.tmp) {
                        eprintln!("[retry-debug] inline_sroa emitted but spirv-val failed: {e}");
                        if let Some(path) = crate::env_vars::retry_dump() {
                            let _ = std::fs::write(&path, bytes);
                            eprintln!("[retry-debug] wrote failing inline_sroa module to {path:?}");
                        }
                    }
                }
                Err(e) => eprintln!("[retry-debug] inline_sroa emit/finish failed: {e}"),
            }
        }
        let bytes = emitted.ok()?;
        if self.validates(&bytes) {
            return Some(bytes);
        }
        // Relooper escalation on the SAME emitted bytes (no re-emit): the collapsed module can be
        // right in every way EXCEPT its control flow — retained merge repair can mishandle the collapsed
        // loops' merge nesting (the astc class: a loop's declared merge stranded outside its
        // enclosing selection). The relooper structures any reducible CFG byte-neutrally.
        let r = load_relooper_module(&bytes).map(|module| module_bytes(&module));
        if self.retry_debug_on {
            if let Err(e) = &r {
                eprintln!("[retry-debug] inline_sroa relooper rewrite failed: {e}");
            }
        }
        r.ok()
            .filter(|b| self.validates_dbg("inline_sroa_relooped", b))
    }

    /// Remove only call boundaries that consume pointer-select results. Unlike the broad historical
    /// inline+SROA tier, this has work proportional to the diagnosed helper bodies rather than the
    /// complete internal call graph.
    pub(crate) fn pointer_select_consumer_inline_retry(
        &self,
        selected_pointer: &str,
    ) -> Option<Vec<u8>> {
        let emitted = tools::emit_vulkan_spirv_pointer_select_consumer_inline_with_sidecar(
            self.san_ll,
            selected_pointer,
            self.tmp,
            self.kern,
            self.entry_name,
            self.buffer_layouts(),
        )
        .and_then(|module| self.finish(module));
        if self.retry_debug_on {
            if let Err(error) = &emitted {
                eprintln!("[retry-debug] pointer-select consumer inline failed: {error}");
            }
        }
        let feed = emitted.ok()?;
        if self.validates_dbg("pointer_select_consumer_inline", &feed) {
            return Some(feed);
        }
        if self.retry_debug_on {
            if let Some(path) = crate::env_vars::retry_dump() {
                let path = std::path::PathBuf::from(path).with_extension("consumer-inline.spv");
                if std::fs::write(&path, &feed).is_ok() {
                    eprintln!(
                        "[retry-debug] wrote failing pointer-select consumer module to {path:?}"
                    );
                }
            }
        }
        let relooped = load_relooper_module(&feed).map(|module| module_bytes(&module));
        if self.retry_debug_on {
            if let Err(error) = &relooped {
                eprintln!("[retry-debug] pointer-select consumer relooper failed: {error}");
            }
        }
        let bytes = relooped.ok()?;
        self.validates_dbg("pointer_select_consumer_inline_relooped", &bytes)
            .then_some(bytes)
    }

    // Escalation of `inline_sroa_retry`: after the inline+SROA collapse, the surviving device buffers
    // are byte-addressed and the kernel loads/stores typed values (float/uint) at byte offsets — a
    // reinterpret Logical typed emit cannot express (invalid `OpLoad %float` from a `uchar*`, or an
    // emit bail). Model those buffers raw (word `RuntimeArray` + byte-offset word access + value
    // bitcast) so the traffic is valid. The TopK multi-destination class needs BOTH: inline+SROA to
    // lower the `%10` pointer array to a direct select over byte buffers, and raw to make the typed
    // byte-offset access on them legal. Adopt-if-validates, so floor-safe.
    pub(crate) fn inline_sroa_raw_retry(&self) -> Option<Vec<u8>> {
        let r = tools::emit_vulkan_spirv_inline_sroa_raw_with_sidecar(
            self.san_ll,
            self.tmp,
            self.kern,
            self.entry_name,
            self.buffer_layouts(),
        )
        .and_then(|b| self.finish(b));
        if self.retry_debug_on {
            if let Err(e) = &r {
                eprintln!("[retry-debug] inline_sroa_raw emit failed: {e}");
            }
        }
        let finished = r.ok()?;
        if self.validates_dbg("inline_sroa_raw", &finished) {
            return Some(finished);
        }
        // The lowered device-pointer array selects among DISTINCT buffer bindings (an `OpSelect` over
        // whole-buffer struct-base pointers), illegal under Logical addressing. Apply the PSB
        // PhysicalStorageBuffer64 cross-binding rewrite, as the `raw_psb` tier does — and, if a
        // Workgroup float-atomic bitcast survives it, the WG remodel too. Adopt only if it then
        // validates; a module WITHOUT a cross-binding merge (the rewrite errors) falls through to the
        // relooper escalation below rather than bailing the whole tier.
        let mut finished_module = load_bytes(&finished).ok()?;
        if let Some(valid) = self.psb_then_wg_remodel_module(
            finished_module.clone(),
            "inline_sroa_raw+psb",
            "inline_sroa_raw+psb+wg",
        ) {
            return Some(valid);
        }
        // The collapsed+raw module can be right in every way EXCEPT its control flow (the
        // scatter_forward class: the raw model dissolves the atomic pointer reinterpret the inlining
        // exposed, and only the CFG wall remains). The relooper structures it byte-neutrally.
        let relooped = native::rewrite_to_relooper_module_capped(
            &mut finished_module,
            native::CFG_EMIT_RELOOPER_MAX_BLOCKS,
        )
        .and_then(|()| {
            passes::repair_relooped_access_chains(&mut finished_module, self.entry_name)?;
            Ok(())
        });
        if self.retry_debug_on {
            if let Err(e) = &relooped {
                eprintln!("[retry-debug] inline_sroa_raw relooper rewrite failed: {e}");
            }
        }
        relooped.ok()?;
        let relooped_bytes = module_bytes(&finished_module);
        if self.validates_dbg("inline_sroa_raw_relooped", &relooped_bytes) {
            return Some(relooped_bytes);
        }
        if !self.psb_retry_enabled {
            return None;
        }
        self.psb_then_wg_remodel_module(
            finished_module,
            "inline_sroa_raw_relooped+psb",
            "inline_sroa_raw_relooped+psb+wg",
        )
    }

    // Combined inline+SROA → raw → cross-arm-restructure → PSB retry — the straddle-loop-merge +
    // cross-binding-phi cluster (`MPSRNNBreakUpToOutputVecs`/05). Its default emit fails cfg-class
    // spirv-val (`Block N is already a merge block`), the relooper refuses it (`no function to
    // relooper`), and `inline_sroa_raw+psb` still fails cfg (`branches to the selection construct, but
    // not to the header`) because the raw byte model leaves the CFG untouched and the straddle reject
    // falls to the naive `infer_*` merges. This tier emits with the cross-arm restructure ENABLED so
    // the CFG is privatized+return-unified before emission, THEN applies the PSB cross-binding rewrite
    // (and the WG float-atomic remodel if that bitcast survives) to dissolve the cross-binding pointer
    // phi into 64-bit addresses. Adopt ONLY if it independently validates — floor-safe by construction
    // (a case that emits+validates on the default path never reaches here).
    pub(crate) fn inline_sroa_raw_cfg_restructure_retry(&self) -> Option<Vec<u8>> {
        let r = tools::emit_vulkan_spirv_inline_sroa_raw_cfg_restructure_with_sidecar(
            self.san_ll,
            self.tmp,
            self.kern,
            self.entry_name,
            self.buffer_layouts(),
        )
        .and_then(|b| self.finish(b));
        if self.retry_debug_on {
            if let Err(e) = &r {
                eprintln!("[retry-debug] inline_sroa_raw_cfg_restructure emit failed: {e}");
            }
        }
        let finished = r.ok()?;
        if self.validates_dbg("inline_sroa_raw_cfg_restructure", &finished) {
            return Some(finished);
        }
        self.psb_then_wg_remodel(
            &finished,
            "inline_sroa_raw_cfg_restructure+psb",
            "inline_sroa_raw_cfg_restructure+psb+wg",
        )
    }

    /// The `inline_sroa` → `inline_sroa_raw` escalation pair, run in order under the two given census
    /// labels. Five cascade arms (`val-ptr`, `val-cfg`, `val-other`, `emit-ptr`, `emit-other`) all end
    /// with this identical adjacent pair; routing them through one helper keeps a single definition of
    /// the escalation while preserving each arm's exact tier order and static census labels — so it is
    /// byte-identical to the inlined `.or_else` chain it replaces.
    pub(crate) fn inline_sroa_chain(
        &self,
        inline_label: &'static str,
        raw_label: &'static str,
    ) -> Option<Vec<u8>> {
        self.census(inline_label, self.inline_sroa_retry())
            .or_else(|| self.census(raw_label, self.inline_sroa_raw_retry()))
    }

    /// The `raw_then_relooper` → `raw_psb` escalation pair, run in order under the two given census
    /// labels. Three cascade arms (`val-ptr`, `emit-ptr`, `emit-other`) end with this identical
    /// adjacent pair; routing them through one helper keeps a single definition of the escalation
    /// while preserving each arm's exact tier order and static census labels — so it is byte-identical
    /// to the inlined `.or_else` chain it replaces. (The `val-other` arm interleaves a `value_select`
    /// tier between the two, so it does NOT use this helper.)
    pub(crate) fn raw_then_psb_chain(
        &self,
        relooper_label: &'static str,
        psb_label: &'static str,
    ) -> Option<Vec<u8>> {
        self.census(relooper_label, self.raw_then_relooper())
            .or_else(|| self.census(psb_label, self.raw_psb()))
    }

    /// R2 construct-tree own-arm retry. Re-emits the normal Logical model with only the bounded
    /// construct-tree candidate enabled, and adopts it only if the finished module validates.
    pub(crate) fn construct_tree_retry(&self) -> Option<Vec<u8>> {
        if let Some(cached) = self.construct_tree_retry_cache.borrow().as_ref() {
            if self.retry_debug_on {
                eprintln!("[retry-debug] construct_tree: reused cached verdict");
            }
            return cached.clone();
        }
        let result = self.construct_tree_retry_uncached();
        *self.construct_tree_retry_cache.borrow_mut() = Some(result.clone());
        result
    }

    fn construct_tree_retry_uncached(&self) -> Option<Vec<u8>> {
        if self.retry_debug_on {
            eprintln!("[retry-debug] construct_tree: emit start");
        }
        let emitted = tools::emit_vulkan_spirv_construct_tree_with_sidecar(
            self.san_ll,
            self.tmp,
            self.kern,
            self.entry_name,
            self.buffer_layouts(),
        );
        if self.retry_debug_on {
            if let Ok(emitted) = &emitted {
                let blocks = emitted
                    .module
                    .functions
                    .iter()
                    .map(|function| function.blocks.len())
                    .sum::<usize>();
                let instructions = emitted
                    .module
                    .functions
                    .iter()
                    .flat_map(|function| &function.blocks)
                    .map(|block| block.instructions.len())
                    .sum::<usize>();
                let operands = emitted
                    .module
                    .functions
                    .iter()
                    .flat_map(|function| &function.blocks)
                    .flat_map(|block| &block.instructions)
                    .map(|instruction| instruction.operands.len())
                    .sum::<usize>();
                let global_instructions = emitted.module.global_inst_iter().count();
                let global_operands = emitted
                    .module
                    .global_inst_iter()
                    .map(|instruction| instruction.operands.len())
                    .sum::<usize>();
                let sidecar = &emitted.sidecar;
                let mapped_layouts = sidecar
                    .air_struct_layout_mappings
                    .iter()
                    .filter(|mapping| mapping.status.is_mapped())
                    .count();
                let unmapped_layouts = sidecar.air_struct_layout_mappings.len() - mapped_layouts;
                eprintln!(
                    "[retry-debug] construct_tree: emit complete blocks={blocks} instructions={instructions} operands={operands} globals={global_instructions} global-operands={global_operands} sidecar-address={} sidecar-buffer-fields={} sidecar-layout-mapped={mapped_layouts} sidecar-layout-unmapped={unmapped_layouts} sidecar-local-stores={} sidecar-local-loads={} sidecar-local-dynamic={}",
                    sidecar.buffer_address_words.len(),
                    sidecar.buffer_pointer_field_loads.len()
                        + sidecar.buffer_pointer_dynamic_field_loads.len(),
                    sidecar.local_pointer_field_stores.len(),
                    sidecar.local_pointer_field_loads.len(),
                    sidecar.local_pointer_dynamic_field_loads.len(),
                );
            } else {
                eprintln!("[retry-debug] construct_tree: emit failed");
            }
        }
        let r = emitted.and_then(|emitted| {
            if self.retry_debug_on {
                eprintln!("[retry-debug] construct_tree: finish start");
            }
            self.finish(emitted)
        });
        if self.retry_debug_on {
            if let Err(e) = &r {
                eprintln!("[retry-debug] construct_tree emit failed: {e}");
            }
        }
        let finished = r.ok()?;
        // Keep the retry dump contract useful when the primary emitter cannot produce bytes: the
        // construct-tree candidate is then the first complete module available for inspecting a
        // finish-time validation failure. Diagnostics remain best-effort and never affect routing.
        if let Some(path) = crate::env_vars::retry_dump() {
            let _ = std::fs::write(path, &finished);
        }
        if self.validates_dbg("construct_tree", &finished) {
            return Some(finished);
        }
        // Structuring and emitted-helper inlining can expose pure pointer phis whose values are
        // unused (for example opaque callback fields that are structurally always null). Their dead
        // `OpConstantNull %_ptr_UniformConstant_*` producer is independently invalid under Vulkan's
        // Logical addressing rules even though neither the phi nor the producer has a semantic use.
        // Reuse the module-wide transitive DCE from the constant-prune retry before trying
        // representation-changing pointer rewrites. The retry is adopt-if-validates and the pass
        // removes only values unreachable from side effects/module roots.
        if let Some(pruned) = self.prune_retry(&finished) {
            if self.retry_debug_on {
                eprintln!("[retry-debug] construct_tree+dead_value_prune: VALIDATES");
            }
            return Some(pruned);
        }
        // A construct-tree candidate can still exceed the relooper cap only because statically dead
        // function-constant arms survived into the finished graph. Prune those exact constant arms
        // before handing the smaller residual CFG to the existing capped relooper; do not rescan or
        // re-emit the AIR source, and adopt only after independent validation.
        if let Some(pruned_relooped) = self.prune_then_relooper(&finished) {
            return Some(pruned_relooped);
        }
        // The construct-tree candidate may clear the source-level ownership problem yet retain a
        // smaller reducible cross-arm exit after interface/value legalization. Hand that finished
        // module directly to the general relooper before representation-changing pointer retries.
        // This composition was previously reachable only as an accidental tail of value-select.
        if let Some(relooped) = self.relooper_retry(&finished) {
            return Some(relooped);
        }
        if let Some(value_selected) = self.value_select_rewrite(&finished) {
            match tools::spirv_val_bytes(&value_selected, self.tmp) {
                Ok(()) => {
                    if self.retry_debug_on {
                        eprintln!("[retry-debug] construct_tree+value_select: VALIDATES");
                    }
                    return Some(value_selected);
                }
                Err(error)
                    if native::classify_validation_error(&error)
                        == native::ValidationClass::CfgStructurization =>
                {
                    if self.retry_debug_on {
                        let head: Vec<&str> = error.lines().take(4).collect();
                        eprintln!(
                            "[retry-debug] construct_tree+value_select: cfg residual: {}",
                            head.join(" | ")
                        );
                    }
                    if let Some(pruned) = self.prune_then_relooper(&value_selected) {
                        if self.retry_debug_on {
                            eprintln!(
                                "[retry-debug] construct_tree+value_select+prune_then_relooper: VALIDATES"
                            );
                        }
                        return Some(pruned);
                    }
                    if let Some(relooped) = self.relooper_retry(&value_selected) {
                        return Some(relooped);
                    }
                }
                Err(error) => {
                    if self.retry_debug_on {
                        let head: Vec<&str> = error.lines().take(4).collect();
                        eprintln!(
                            "[retry-debug] construct_tree+value_select: emitted, spirv-val failed: {}",
                            head.join(" | ")
                        );
                    }
                }
            }
        }
        self.psb_then_wg_remodel(&finished, "construct_tree+psb", "construct_tree+psb+wg")
    }

    // W2 relooper retry — a control-flow structurization failure (the cfg frontier class: switch-in-
    // loop, multi-exit loops, irregular selection merges) that the cross-arm cfg restructure above does
    // not repair is lowered to the general relooper form (one switch-dispatch loop + state variable +
    // register demotion), which structures ANY reducible CFG mechanically. Operates on already-emitted
    // bytes and is adopted ONLY if it independently validates — same adopt-if-VALIDATES floor-safety.
    pub(crate) fn relooper_retry(&self, out: &[u8]) -> Option<Vec<u8>> {
        let r = load_relooper_module(out).map(|module| module_bytes(&module));
        if self.retry_debug_on {
            if let Err(e) = &r {
                eprintln!("[retry-debug] relooper_retry rewrite failed: {e}");
            }
        }
        r.ok().filter(|b| self.validates_dbg("relooper_retry", b))
    }

    // Relooper-then-cross-binding retry — a cfg-class failure whose relooped form structures the CFG
    // fine but then trips spirv-val on a cross-binding pointer merge the relooper's from-scratch
    // rebuild EXPOSES (`Variable pointers must point into the same structure` on an
    // `OpSelect %_ptr_StorageBuffer_*`). On the DEFAULT (repaired) path this same function reaches
    // spirv-val as a clean `CrossBindingPointerMerge` and `value_select_retry` lowers it; but when the
    // module first fails cfg-class (e.g. repair disabled / a merge-collision the structured plan can't
    // clear), the CfgStructurization arm structures via the relooper and the exposed pointer select is
    // never handed to the value-domain lowering. This tier closes that routing gap: reloop, then prefer
    // the Logical value-domain lowering (MoltenVK-runnable), falling back to the PSB whole-buffer form —
    // exactly the composition the `CrossBindingPointerMerge` arm uses, applied to the relooped bytes.
    // Adopt-if-VALIDATES (floor-safe: a banked case validates on the default emit and never reaches
    // here). The PSB leg is `psb_retry_enabled` (always on in production).
    pub(crate) fn relooper_then_value_select(&self, out: &[u8]) -> Option<Vec<u8>> {
        let relooped_module = load_relooper_module(out).ok()?;
        let mut value_candidate = relooped_module.clone();
        if native::rewrite_cross_binding_pointer_merges_to_values_module(&mut value_candidate)
            .is_ok()
        {
            let bytes = value_candidate
                .assemble()
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>();
            if self.validates_dbg("relooper_then_value_select", &bytes) {
                return Some(bytes);
            }
        }
        if !self.psb_retry_enabled {
            return None;
        }
        self.psb_then_wg_remodel_module(
            relooped_module,
            "relooper_then_psb",
            "relooper_then_psb+wg",
        )
    }

    // Combined prune-then-relooper retry — a huge (>1024-block) function whose statically-dead
    // function-constant arms inflate its block count past the relooper's block cap. Const-fold the
    // dead arms first (byte-correct DCE — removes only code that never runs at the disabled FC
    // default the golden was captured with), which can shrink the function below the cap, THEN
    // relooper the smaller module. Both transforms are semantics-preserving (prune deletes only dead
    // code; the relooper restructures the SAME reachable CFG), so the result is byte-correct, and it
    // is adopted ONLY if it independently validates — same adopt-if-VALIDATES floor-safety. Returns
    // None when nothing prunes (so the plain relooper_retry remains the operative tier).
    pub(crate) fn prune_then_relooper(&self, out: &[u8]) -> Option<Vec<u8>> {
        let mut pruned_module = load_bytes(out).ok()?;
        native::prune_constant_branches_module(&mut pruned_module).ok()?;
        // Function-constant pruning can shrink an oversized source below the same hard product cap;
        // a residual graph above that cap remains an honest fallback.
        let r = native::rewrite_to_relooper_module_capped(
            &mut pruned_module,
            native::PRUNE_THEN_RELOOPER_MAX_BLOCKS,
        )
        .map(|()| module_bytes(&pruned_module));
        if self.retry_debug_on {
            if let Err(e) = &r {
                eprintln!("[retry-debug] prune_then_relooper rewrite failed: {e}");
            }
        }
        r.ok()
            .filter(|b| self.validates_dbg("prune_then_relooper", b))
    }

    // Combined raw-then-relooper retry — a module that is BOTH pointer-mistyped (the raw byte model
    // fixes that) AND carries an unstructured CFG the relooper structures (a switch/selection whose
    // case branches the default typed emission left unstructured). Neither the plain `raw_retry` (its
    // raw bytes still fail spirv-val on the CFG) nor `prune_then_relooper` (it reloops the DEFAULT
    // bytes, which keep the pointer-type error) clears this composite; only running the relooper ON the
    // raw bytes does. Re-emit every device/constant buffer raw, then structure the result with the
    // relooper (first plain, then prune-dead-FC-arms-and-relooper for a switch inflated past the cap),
    // and adopt ONLY if it independently validates. Byte-correct by construction (the raw byte model is
    // a faithful byte view, golden-verified on banked cases; the relooper structurizes the SAME
    // reachable CFG, byte-neutrally) and floor-safe by construction (adopt-if-validates — a banked case
    // emits + validates on the default path and never reaches here). If both ordinary raw emissions
    // fail in the straight-line typed graph walk, skip the unrepaired CFG feed: it only changes
    // merge/structuring work around emitted blocks and cannot make an unmigrated body opcode emit.
    pub(crate) fn raw_then_relooper(&self) -> Option<Vec<u8>> {
        let raw1 = self.raw_reemit();
        if self.retry_debug_on {
            if let Err(e) = &raw1 {
                eprintln!("[retry-debug] raw_then_relooper: raw emit failed: {e}");
            }
        }
        let raw_bytes = match raw1 {
            Ok(bytes) => bytes,
            Err(raw1_err) => {
                let raw2 = self.raw_reemit_wg();
                if self.retry_debug_on {
                    if let Err(e) = &raw2 {
                        eprintln!("[retry-debug] raw_then_relooper: raw+wg emit failed: {e}");
                    }
                }
                match raw2 {
                    Ok(bytes) => bytes,
                    Err(raw2_err) => {
                        if native::is_graph_walk_unmigrated_emit_error(&raw1_err)
                            && native::is_graph_walk_unmigrated_emit_error(&raw2_err)
                        {
                            if self.retry_debug_on {
                                eprintln!(
                                    "[retry-debug] raw_then_relooper: raw feed skipped after typed graph-walk emit failures"
                                );
                            }
                            return None;
                        }
                        let raw_feed = self.raw_reemit_relooper_feed();
                        if self.retry_debug_on {
                            if let Err(e) = &raw_feed {
                                eprintln!(
                                    "[retry-debug] raw_then_relooper: raw feed emit failed: {e}"
                                );
                            }
                        }
                        raw_feed.ok()?
                    }
                }
            }
        };
        self.adopt_relooped_raw(&raw_bytes)
    }

    fn adopt_relooped_raw(&self, raw_bytes: &[u8]) -> Option<Vec<u8>> {
        // Adopt a relooped raw module if it validates as-is, OR — when it still carries a WHOLE-buffer
        // cross-binding select (the pointer mistyping's real shape, e.g. the `createBVHNodesKernelMotion`
        // BVH builders: a ~1340-block unstructured switch that, once structured, selects among distinct
        // device buffers `OpSelect %_ptr_StorageBuffer__struct %cond %bufA %bufB`) and PSB is enabled —
        // after the PSB whole-buffer lowering to PhysicalStorageBuffer64. This reuses the already-relooped
        // bytes (no second reloop), so the only added cost over a plain relooper adopt is a PSB rewrite +
        // one spirv-val on the cases that don't validate as-is. Byte-correct by construction (raw byte
        // model = faithful byte view; relooper = byte-neutral structurization of the SAME reachable CFG;
        // PSB whole-buffer = byte-correct device-address lowering) and floor-safe by construction
        // (adopt-if-validates — a banked case never reaches here).
        let adopt = |relooped_module: Module| -> Option<Vec<u8>> {
            let relooped = module_bytes(&relooped_module);
            if self.validates_dbg("raw_then_relooper(relooped)", &relooped) {
                return Some(relooped);
            }
            if !self.psb_retry_enabled {
                return None;
            }
            // PSB whole-buffer lowering, then the WG float-atomic remodel if that bitcast survives —
            // the combined cross-binding-select + Workgroup-float-atomic BVH shape reaches this tier
            // (`binFragmentsTemporalSplitKernel`) and needs both dissolved.
            self.psb_then_wg_remodel_module(
                relooped_module,
                "raw_then_relooper(relooped+psb)",
                "raw_then_relooper(relooped+psb+wg)",
            )
        };
        let relooped = load_relooper_module(raw_bytes);
        if self.retry_debug_on {
            if let Err(e) = &relooped {
                eprintln!("[retry-debug] raw_then_relooper: relooper rewrite failed: {e}");
            }
        }
        relooped.ok().and_then(adopt).or_else(|| {
            let mut pruned_module = load_bytes(raw_bytes).ok()?;
            native::prune_constant_branches_module(&mut pruned_module).ok()?;
            native::rewrite_to_relooper_module_capped(
                &mut pruned_module,
                native::PRUNE_THEN_RELOOPER_MAX_BLOCKS,
            )
            .ok()
            .and_then(|()| adopt(pruned_module))
        })
    }
}

fn emitted_graph_exceeds_relooper_cap(bytes: &[u8]) -> bool {
    load_bytes(bytes).is_ok_and(|module| module_exceeds_relooper_cap(&module))
}

fn module_exceeds_relooper_cap(module: &Module) -> bool {
    module
        .functions
        .iter()
        .any(|function| function.blocks.len() > native::CFG_EMIT_RELOOPER_MAX_BLOCKS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::{Block, Function};

    fn module_with_blocks(blocks: usize) -> Module {
        let mut module = Module::new();
        let mut function = Function::new();
        function.blocks.resize_with(blocks, Block::new);
        module.functions.push(function);
        module
    }

    #[test]
    fn post_finish_graph_over_relooper_cap_keeps_construct_tree_eligible() {
        assert!(!module_exceeds_relooper_cap(&module_with_blocks(
            native::CFG_EMIT_RELOOPER_MAX_BLOCKS
        )));
        assert!(module_exceeds_relooper_cap(&module_with_blocks(
            native::CFG_EMIT_RELOOPER_MAX_BLOCKS + 1
        )));
    }
}
