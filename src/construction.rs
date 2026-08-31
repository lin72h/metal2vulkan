//! Pre-validation construction alternatives selected from owned CFG and type facts.
//!
//! The primary representation is finished and structurally checked while it remains an owned
//! module. When those checks prove that representation cannot satisfy the source contract, this
//! context constructs the raw-buffer or raw-CFG representation. Only the single finished result is
//! serialized and validated; validator output never selects another representation.

use crate::{
    finish_module, meta, passes, stage_buffer_layouts, tools, FinishConstruction,
    FinishFailureKind, FinishedModule,
};
use std::collections::HashMap;
use std::path::Path;

/// Shared source, metadata, and option context for pre-validation representation construction.
pub(crate) struct ConstructionCtx<'a> {
    pub(crate) san_ll: &'a str,
    pub(crate) stage: passes::Stage,
    pub(crate) frag: Option<&'a meta::FragMeta>,
    pub(crate) vert: Option<&'a meta::VertMeta>,
    pub(crate) kern: Option<&'a meta::KernMeta>,
    pub(crate) entry_name: Option<&'a str>,
    pub(crate) tmp: &'a Path,
    pub(crate) options: passes::TransformOptions,
    pub(crate) air_data_layout: Option<crate::layout::AirDataLayout>,
    ordinary_plan_rejections: std::cell::RefCell<std::collections::HashSet<String>>,
    ownership_plan_rejections: std::cell::RefCell<std::collections::HashSet<String>>,
    post_lowering_cfg_construction: std::cell::Cell<bool>,
    primary_cfg_construction_failure: std::cell::Cell<bool>,
    primary_raw_buffer_construction_failure: std::cell::Cell<bool>,
}

impl<'a> ConstructionCtx<'a> {
    fn buffer_layouts(&self) -> Option<&'a HashMap<u32, meta::AirType>> {
        stage_buffer_layouts(self.stage, self.frag, self.vert, self.kern)
    }

    /// Build the construction context shared by primary and structurally selected representations.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        san_ll: &'a str,
        stage: passes::Stage,
        frag: Option<&'a meta::FragMeta>,
        vert: Option<&'a meta::VertMeta>,
        kern: Option<&'a meta::KernMeta>,
        entry_name: Option<&'a str>,
        tmp: &'a Path,
        options: passes::TransformOptions,
        air_data_layout: Option<crate::layout::AirDataLayout>,
    ) -> Self {
        ConstructionCtx {
            san_ll,
            stage,
            frag,
            vert,
            kern,
            entry_name,
            tmp,
            options,
            air_data_layout,
            ordinary_plan_rejections: std::cell::RefCell::new(Default::default()),
            ownership_plan_rejections: std::cell::RefCell::new(Default::default()),
            post_lowering_cfg_construction: std::cell::Cell::new(false),
            primary_cfg_construction_failure: std::cell::Cell::new(false),
            primary_raw_buffer_construction_failure: std::cell::Cell::new(false),
        }
    }

    /// Run the shared owned-module construction tail over a freshly emitted representation.
    #[cfg(test)]
    pub(crate) fn finish(
        &self,
        emitted: crate::emit_sidecar::EmittedSpirv,
    ) -> Result<Vec<u8>, String> {
        self.finish_carrier(emitted).map(|finished| finished.bytes)
    }

    #[cfg(test)]
    fn finish_carrier(
        &self,
        emitted: crate::emit_sidecar::EmittedSpirv,
    ) -> Result<FinishedModule, String> {
        self.finish_carrier_with_construction(emitted, FinishConstruction::Plain)
    }

    fn finish_carrier_with_construction(
        &self,
        emitted: crate::emit_sidecar::EmittedSpirv,
        construction: FinishConstruction,
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
            construction,
        )
        .map_err(|failure| failure.error)
    }

    pub(crate) fn finish_primary_carrier(
        &self,
        emitted: crate::emit_sidecar::EmittedSpirv,
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
            FinishConstruction::Primary,
        )
        .map_err(|failure| {
            match failure.kind {
                FinishFailureKind::CfgConstruction => {
                    self.primary_cfg_construction_failure.set(true);
                }
                FinishFailureKind::RawBufferConstruction => {
                    self.primary_raw_buffer_construction_failure.set(true);
                }
                FinishFailureKind::Other => {}
            }
            failure.error
        })
    }

    pub(crate) fn remember_ordinary_plan_rejections(
        &self,
        emitted: &crate::emit_sidecar::EmittedSpirv,
    ) {
        self.ordinary_plan_rejections.borrow_mut().extend(
            emitted
                .sidecar
                .ordinary_plan_rejected_functions
                .iter()
                .cloned(),
        );
        self.ownership_plan_rejections.borrow_mut().extend(
            emitted
                .sidecar
                .ownership_plan_rejected_functions
                .iter()
                .cloned(),
        );
        self.post_lowering_cfg_construction.set(
            self.post_lowering_cfg_construction.get()
                || !emitted
                    .sidecar
                    .post_lowering_cfg_construction_functions
                    .is_empty(),
        );
    }

    pub(crate) fn remember_ordinary_plan_rejection_set(
        &self,
        rejected: &std::collections::HashSet<String>,
    ) {
        self.ordinary_plan_rejections
            .borrow_mut()
            .extend(rejected.iter().cloned());
    }

    pub(crate) fn has_cfg_plan_rejections(&self) -> bool {
        !self.ordinary_plan_rejections.borrow().is_empty()
            || !self.ownership_plan_rejections.borrow().is_empty()
            || self.post_lowering_cfg_construction.get()
            || self.primary_cfg_construction_failure.get()
    }

    pub(crate) fn needs_raw_construction(&self) -> bool {
        self.has_cfg_plan_rejections() || self.primary_raw_buffer_construction_failure.get()
    }

    /// Build the selected raw-buffer representation. CFG ownership failures additionally select
    /// the bounded relooper feed; a pure buffer-type failure keeps the ordinary CFG representation.
    pub(crate) fn construct_raw(&self) -> Result<FinishedModule, String> {
        if self.has_cfg_plan_rejections() {
            return self.construct_raw_relooper();
        }
        let ordinary_rejections = self.ordinary_plan_rejections.borrow();
        let ownership_rejections = self.ownership_plan_rejections.borrow();
        let mut emitted = tools::emit_vulkan_spirv_all_buffers_raw_with_sidecar(
            self.san_ll,
            self.tmp,
            self.kern,
            self.entry_name,
            self.buffer_layouts(),
            &ordinary_rejections,
            &ownership_rejections,
        )?;
        // The all-raw representation can still contain a source pointer phi/select spanning
        // distinct buffer descriptors even when its CFG needed no reconstruction. Tell final
        // interface construction up front to own that closure in the address domain.
        emitted.sidecar.construct_cross_binding_addresses = true;
        self.finish_carrier_with_construction(emitted, FinishConstruction::Primary)
    }

    /// Build the raw-buffer CFG representation selected by a source ownership rejection. The raw
    /// feed remains an owned module through final CFG and address-domain construction, its owned
    /// checks, canonicalization, and its sole assembly. No intermediate representation is sealed,
    /// and no validator result selects a rewrite.
    pub(crate) fn construct_raw_relooper(&self) -> Result<FinishedModule, String> {
        let mut emitted = tools::emit_vulkan_spirv_all_buffers_raw_relooper_feed_with_sidecar(
            self.san_ll,
            self.tmp,
            self.kern,
            self.entry_name,
            self.buffer_layouts(),
        )?;
        emitted.sidecar.construct_cross_binding_addresses = true;
        self.finish_carrier_with_construction(emitted, FinishConstruction::RawRelooper)
    }

    pub(crate) fn remember_ownership_plan_rejection_set(
        &self,
        rejected: &std::collections::HashSet<String>,
    ) {
        self.ownership_plan_rejections
            .borrow_mut()
            .extend(rejected.iter().cloned());
    }
}
