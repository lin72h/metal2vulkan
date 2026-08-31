// Dominator/loop-forest analysis backing the structured-by-construction path (`structured_plan`).
// A rejected local plan is recorded as an owned construction fact so the module can select its raw
// CFG representation before serialization.
// Low-level source-CFG graph primitives (successor/predecessor adjacency + CHK
// dominator tree) shared by the loop-forest and structurizer analyses.
pub(in crate::native) mod graph;
pub(super) mod loopforest;
// Cross-arm node-splitting (single-entry tail duplication) used while constructing structured CFGs.
pub(super) mod clone_crossarm;
// Compact immediate-dominator analysis over the emitted CFG, shared by late rewrites.
mod exit_check;
// Forest-driven emission consumer: loop-merge computation + merge==continue overlap split for
// `structured_plan`, the live default structurization path.
pub(super) mod structured_emit;
// Structured block ordering (dominator-tree preorder, merge-last) consumed by `structured_plan`.
mod blocks;
pub(super) mod structured_order;

pub(super) use clone_crossarm::rename_tokens;
pub(in crate::native) use exit_check::EmittedDominators;
pub(super) use structured_emit::{
    cond_other_witness_lines, cond_phi_shared_witness_lines, construct_tree_gate_witness_lines,
    construct_tree_reject_reason, exceeds_local_structured_plan_budget,
    privatize_reused_emitted_merge_targets, renest_cond_phi_shared_own_arm,
    renest_loop_exit_sibling, renest_straddle_loop_merge, renest_whole_cfg_dispatch,
    requires_loop_exit_sibling_dispatch, requires_shared_loop_entry_ownership,
    straddle_witness_lines, structured_plan, structured_plan_construct_tree,
    structured_reject_reason, CROSS_ARM_EDGE_MAX_BLOCKS,
};

#[cfg(test)]
pub(super) use blocks::split_body_blocks;
pub(super) use blocks::{
    funnel_shared_branch_dispatches, implicit_entry_block_name, index_branch_merges_by_header,
    infer_bounded_branch_merges_by_header, infer_branch_merges, infer_direct_branch_merges,
    infer_direct_switch_merges, infer_loop_merges, infer_switch_merges,
    lower_unstructured_switches, refunnel_one_deep_shared_arm, split_source_body_blocks,
    switch_default_is_inferred_merge,
};
#[cfg(test)]
pub(super) fn id_ref_operand(operand: &crate::spirv_module::Operand) -> Option<spirv::Word> {
    let crate::spirv_module::Operand::IdRef(id) = operand else {
        return None;
    };
    Some(*id)
}

#[derive(Clone, Debug)]
pub(super) struct LoopMergeInfo {
    pub(super) merge: String,
    pub(super) continue_target: String,
}

/// The structural ROLE of a synthesized block, stamped by the synthesizer that creates it so
/// consumers recover the role from a typed tag instead of decoding it from the block's name. The
/// synthesized name is still emitted as the block label / `OpName`, but no production decision
/// reads it.
/// `Normal` is every ordinary block (an AIR label or a synthesized block whose role no consumer yet
/// queries); non-`Normal` variants are added as each name-prefix consumer migrates to the tag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum BlockRole {
    /// An ordinary block — an AIR-sourced block or a synthesized block no role consumer queries.
    #[default]
    Normal,
    /// A structurizer-synthesized `%metal2vulkan.lmerge.*` block (a selection/loop merge, a
    /// pass-through, a loop-return privatization, …) that is NOT the terminal-exit-return subtype.
    /// `terminal_exit_convergence` excludes these from its convergence candidate set via this tag
    /// instead of matching `starts_with("%metal2vulkan.lmerge.")` on the name.
    LMerge,
    /// A synthesized terminal-exit private-return clone (a `ret void` stub the terminal-return
    /// structurizer redirects an exit edge to). Named `%metal2vulkan.lmerge.texitret*` — a specific
    /// `lmerge` subtype; the convergence matcher (`terminal_exit_convergence`) and the
    /// single-exit-return synthesizer read this tag instead of matching `is_terminal_exit_return_name`
    /// on the name. Both `LMerge` and `TerminalExitReturn` are `%metal2vulkan.lmerge.*` blocks.
    TerminalExitReturn,
    /// A synthesized switch-bypass block (`%metal2vulkan_switch_bypass_*`) inserted between a switch
    /// predecessor and a shared merge so branch-merge inference can model the reconvergence. Branch
    /// inference (`infer_branch_merges`) recognizes these via this tag instead of matching the
    /// `SWITCH_BYPASS_PREFIX` on the branch-target name.
    SwitchBypass,
    /// Construct-tree route scaffolding: dispatchers and pass-through gateways that preserve an
    /// original inter-construct edge. Selection-merge repair must not treat these blocks as owned by a
    /// source conditional merely because the materialized dispatcher is statically dominated by that
    /// conditional in the wrapper CFG.
    ConstructTreeRoute,
}

#[derive(Clone, Debug)]
pub(super) struct BodyBlock {
    pub(super) name: String,
    /// The block's structural role (see [`BlockRole`]). `Normal` for every AIR-sourced block and
    /// every synthesized block whose role no consumer queries; stamped non-`Normal` at the synthesis
    /// site for the roles production code decides on.
    pub(super) role: BlockRole,
    /// The typed lowering of this block — its instructions + structured terminator (see
    /// [`crate::native::tir::TirBlock`]) — the SOLE substrate for CFG analysis, structurization, and
    /// emission. Built at parse time from the AIR text (`split_body_blocks` → `lower_block_carrier`, the
    /// one place block instructions are lexed) and rebuilt via the typed `phi_edit`/terminator primitives
    /// at every synthesis/mutation site. `None` only for a block whose lines did not lower (no
    /// terminator); such a block is a fail-visible `build_from_blocks` error, not a re-lower fallback.
    pub(super) typed: Option<std::sync::Arc<crate::native::tir::TirBlock>>,
}

impl BodyBlock {
    /// Mutable access to the typed carrier with copy-on-write semantics. Planner candidates share
    /// immutable AIR instruction payload; only a block whose transform actually edits it is cloned.
    pub(super) fn typed_mut(&mut self) -> Option<&mut crate::native::tir::TirBlock> {
        self.typed.as_mut().map(std::sync::Arc::make_mut)
    }

    /// The block's instruction + terminator lines, rendered from its typed carrier (the sole substrate).
    /// TEST-ONLY: the CFG-restructuring unit tests were written against the retired `.lines` field; this
    /// reproduces those lines from the carrier so the assertions read structured output as text. Panics
    /// on a carrier-less block (a test fixture must lower).
    #[cfg(test)]
    pub(super) fn lines(&self) -> Vec<String> {
        crate::native::tir::render_block_lines(
            self.typed
                .as_ref()
                .expect("test fixture block must have a typed carrier"),
        )
    }
}
