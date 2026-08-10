//! Retry-routing error classifiers: pure substring predicates over SPIRV-Tools / native-emitter
//! message text that decide which failure-recovery retry a translate error routes to, plus the
//! aggregate [`ValidationClass`]/[`EmitErrorClass`] enums `lib.rs`'s cascade matches on. Keeping the
//! validator-text matching in this one adapter means a SPIRV-Tools message change touches one place.
//! Unit coverage lives in `error_classifier_tests`.

use crate::spirv_module::{Module, Operand};
use spirv::{Op, Word};
use std::collections::HashMap;

/// A systematic shared-merge failure this large is outside the bounded CFG recovery envelope. The
/// threshold deliberately excludes isolated/local structurizer mistakes, which still receive the
/// complete retry ladder.
const SYSTEMATIC_REUSED_MERGE_TARGETS: usize = 8;

pub(crate) fn systematic_reused_merges_beyond_relooper(module: &Module) -> Option<(usize, usize)> {
    module.functions.iter().find_map(|function| {
        if function.blocks.len() <= super::CFG_EMIT_RELOOPER_MAX_BLOCKS {
            return None;
        }
        let mut claims = HashMap::<Word, usize>::new();
        for instruction in function.blocks.iter().flat_map(|block| &block.instructions) {
            if !matches!(instruction.class.opcode, Op::SelectionMerge | Op::LoopMerge) {
                continue;
            }
            if let Some(Operand::IdRef(merge)) = instruction.operands.first() {
                *claims.entry(*merge).or_default() += 1;
            }
        }
        let reused = claims.values().filter(|&&count| count > 1).count();
        (reused >= SYSTEMATIC_REUSED_MERGE_TARGETS).then_some((function.blocks.len(), reused))
    })
}

#[cfg(test)]
mod systematic_merge_tests {
    use super::*;
    use crate::spirv_module::{Block, Function, Instruction};

    fn module_with_reused_merges(blocks: usize, reused_targets: usize) -> Module {
        let mut function = Function::new();
        function.blocks.resize_with(blocks, Block::new);
        for target in 0..reused_targets {
            for claim in 0..2 {
                function.blocks[target * 2 + claim]
                    .instructions
                    .push(Instruction::new(
                        Op::SelectionMerge,
                        None,
                        None,
                        vec![Operand::IdRef(target as Word + 1)],
                    ));
            }
        }
        let mut module = Module::new();
        module.functions.push(function);
        module
    }

    #[test]
    fn systematic_merge_fast_fallback_requires_both_size_and_repetition() {
        assert!(
            systematic_reused_merges_beyond_relooper(&module_with_reused_merges(
                super::super::CFG_EMIT_RELOOPER_MAX_BLOCKS,
                SYSTEMATIC_REUSED_MERGE_TARGETS,
            ))
            .is_none()
        );
        assert!(
            systematic_reused_merges_beyond_relooper(&module_with_reused_merges(
                super::super::CFG_EMIT_RELOOPER_MAX_BLOCKS + 1,
                SYSTEMATIC_REUSED_MERGE_TARGETS - 1,
            ))
            .is_none()
        );
        assert_eq!(
            systematic_reused_merges_beyond_relooper(&module_with_reused_merges(
                super::super::CFG_EMIT_RELOOPER_MAX_BLOCKS + 1,
                SYSTEMATIC_REUSED_MERGE_TARGETS,
            )),
            Some((
                super::super::CFG_EMIT_RELOOPER_MAX_BLOCKS + 1,
                SYSTEMATIC_REUSED_MERGE_TARGETS,
            ))
        );
    }
}

/// Whether a spirv-val error message is one of the buffer pointer-typing failures the raw byte-offset
/// path repairs (the pointer-merge frontier class). Anything else — a CFG/back-edge error, a missing
/// capability, or a spirv-val spawn failure — returns false so the ground-truth retry is skipped.
pub fn is_pointer_typing_validation_error(err: &str) -> bool {
    err.contains("does not match Pointer")
        || err.contains("does not match Object")
        || err.contains("does not match the type that results from indexing into the Composite")
        || err.contains("reached non-composite")
        || err.contains("OpInBoundsAccessChain cannot find index")
        // Anchored: the real spirv-val message is "... does not match the type used to index ..."
        // (an access-chain index-type mismatch). The bare "to index" substring also matched
        // unrelated wordings ("prior to indexing", "failed to index"); "used to index" pins the
        // access-chain family without the spurious catches. (M-C3 classifier hardening.)
        || err.contains("used to index")
}

/// Whether a native-emitter ERROR (the default typed emission returned `Err` *before* any spirv-val
/// runs) is a buffer/pointer-typing gap the raw byte-offset model can express: a reinterpret-load
/// width mismatch (a buffer's declared pointee width differs from the load's result width, so the
/// typed access chain cannot be formed) or a missing pointer storage class. The ground-truth raw retry
/// uses this to also recover from an outright emit failure — not only a spirv-val rejection of a
/// produced module — by re-emitting with buffers modeled raw and adopting that only if it independently
/// validates. Floor-safe: a banked case emits and validates on the default path, so it never reaches
/// the retry; an emit failure was already a hard FALLBACK, so the raw attempt can only add a win.
pub fn is_pointer_typing_emit_error(err: &str) -> bool {
    err.contains("bit width mismatch")
        || err.contains("missing pointer storage")
        || err.contains("cannot reinterpret load")
        // A typed Workgroup argument pointer that the default path can only reinterpret to the raw
        // word view via an illegal logical-pointer `OpBitcast`. The all-buffers-raw-with-workgroup
        // retry models the callee buffer raw and passes the argument without a bitcast, so this routes
        // to that retry instead of emitting (and shipping) the rejected module.
        || err.contains("cannot reinterpret workgroup pointer")
        // The other illegal logical-pointer `OpBitcast` reinterpret sites the default path used to ship
        // a rejected module from: a cross-address-space pointer reinterpret, and a byte (i8) pointer
        // reinterpreted to a wider load type. Both are inexpressible under Logical addressing without a
        // pointer bitcast, so they route to the raw byte/word-offset retry instead.
        || err.contains("cannot reinterpret pointer")
        || err.contains("cannot reinterpret load of byte pointer")
        // A store THROUGH a direct pointer select whose arms can point into distinct buffers — illegal
        // as a pointer `OpSelect`+store under Logical addressing. Routes to the all-buffers-raw retry,
        // which lowers the select + store as byte-offset operations on the raw backings.
        || err.contains("cannot store through reinterpreted pointer select")
}

/// Whether a native-emitter error came from the typed straight-line body dispatcher after it failed
/// to find a migrated lowering/carrier for one instruction. CFG-only retry feeds cannot repair this
/// class: they change structured merge handling around already-emitted blocks, but this error occurs
/// before the block body is complete.
pub fn is_graph_walk_unmigrated_emit_error(err: &str) -> bool {
    err.contains("reason=graph_walk_unmigrated_opcode")
}

/// Whether a spirv-val rejection of a produced module (`branches to the selection construct`,
/// `back-edge`, `header block`, `does not dominate`, …) is a control-flow structurization failure the
/// relooper retry tiers can restructure. Used by [`classify_validation_error`] to route a rejected
/// module to the `CfgStructurization` cascade; a false positive only wastes a retry that is discarded
/// unless it independently validates. Anything pointer/parser/intrinsic-shaped returns false so the
/// cfg retry stays off that path. The needle set is spirv-val phrasings only: the legacy emit-error
/// `cost budget` / `clone budget` needles were deleted at cleanup-plan S2 (the repair roster that
/// produced them died at W4, and [`classify_emit_error`] no longer consults this predicate).
pub fn is_cfg_structurization_error(err: &str) -> bool {
    let m = err.to_ascii_lowercase();
    let has = |needle: &str| m.contains(needle);
    has("structured cfg")
        || has("back-edge")
        || has("back edge")
        || has("loop header")
        || has("loop construct")
        || has("selection construct")
        || has("merge block")
        || has("header block")
        || has("continue target")
        || has("structured exit")
        || has("structurally dominate")
        || has("does not dominate")
        // "Block N appears in the binary before its dominator M": a pure block-ORDERING violation in
        // the structurizer's output — the relooper rebuilds every block into one switch in a
        // dominance-valid order, so it can repair this. Adopt-if-validates keeps it floor-safe.
        || has("before its dominator")
        || has("must be structured")
        || has("case construct")
        // Structured-order / unrepaired merge emission can leave a use before its def in the
        // serialized block stream. spirv-val reports "ID N has not been defined" — the same class of
        // CFG emission failure the relooper / raw→relooper cascade rebuilds. Without this needle the
        // message falls through to Other and primary-validated never adopts the production rescue.
        || has("has not been defined")
}

/// Whether a spirv-val error is the illegal-logical-pointer-`OpPhi` rejection the M4 phi-the-index
/// rewrite ([`rewrite_logical_pointer_phis_retry_module`]) targets: a pointer `OpPhi`/`OpSelect` in
/// a storage class `VariablePointers` cannot cover (Private/UniformConstant/Function). Used by
/// `lib.rs`'s retry to route a rejected module to the rewrite; a false positive only wastes a retry
/// that is discarded unless the rewrite independently validates.
pub fn is_logical_pointer_phi_error(err: &str) -> bool {
    err.contains("may only have a logical pointer operand")
}

/// Whether a spirv-val error is the cross-binding pointer-merge rejection the W1 PhysicalStorageBuffer64
/// rewrite ([`rewrite_cross_binding_pointer_merges_bytes`]) targets: an `OpSelect`/`OpPhi` over pointers
/// from DISTINCT descriptor bindings, illegal under Logical addressing ("Variable pointers must point
/// into the same structure"). Used by `lib.rs`'s retry to route a rejected module to the PSB rewrite; a
/// false positive only wastes a retry that is discarded unless the rewrite independently validates.
pub fn is_cross_binding_pointer_merge_error(err: &str) -> bool {
    err.contains("Variable pointers must point into the same structure")
}

/// Whether a spirv-val rejection is the dynamic-structure-index rule ("Index into a structure must be
/// an OpConstant"): an access chain indexed a struct member with a runtime value, illegal in
/// structured SPIR-V. Classifies as [`ValidationClass::Other`] (no dedicated tier), but `lib.rs`'s
/// `primary_dynamic_struct_index_inline_sroa_relooper` rewrite refines within `Other` on exactly this
/// message. Kept here so validator-text matching lives only in this adapter (cleanup-plan S1).
pub fn is_dynamic_struct_index_error(err: &str) -> bool {
    err.contains("into a structure must be an OpConstant")
}

/// Whether a spirv-val rejection is the illegal-logical-pointer-operand rule ("Instruction may not
/// have a logical pointer operand"): an instruction other than an allowed pointer op took a logical
/// pointer where the storage class forbids it. Classifies as [`ValidationClass::Other`], but `lib.rs`'s
/// `primary_logical_pointer_inline_sroa_raw` rewrite refines within `Other` on exactly this message.
/// Distinct from [`is_logical_pointer_phi_error`] (the `OpPhi`-specific rule). Kept here so
/// validator-text matching lives only in this adapter (cleanup-plan S1).
pub fn is_logical_pointer_operand_error(err: &str) -> bool {
    err.contains("Instruction may not have a logical pointer operand")
}

/// Structural class of a spirv-val rejection of an *emitted* module, used to route the
/// failure-triggered retry cascade in `lib.rs`. This is the single classifier over the substring
/// matchers above (§2.2 fact 5 in the refactor plan): a SPIRV-Tools message change now touches one
/// place. The precedence here is load-bearing — it MUST equal the order of `lib.rs`'s
/// `Err(e) if is_*(e)` guard chain (pointer-typing, then cfg, then logical-pointer-phi, then
/// cross-binding), so that `match classify_validation_error(e)` fires exactly the arm the guard chain
/// did for an error that matches more than one family (spirv-val reports one error per module, but the
/// families are not provably disjoint).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationClass {
    /// Buffer/pointer mistyping the raw byte-offset path repairs — [`is_pointer_typing_validation_error`].
    PointerTyping,
    /// Structured-CFG nesting/dominance violation — [`is_cfg_structurization_error`].
    CfgStructurization,
    /// Illegal logical-pointer `OpPhi` — [`is_logical_pointer_phi_error`].
    LogicalPointerPhi,
    /// `OpSelect`/`OpPhi` over pointers from distinct bindings — [`is_cross_binding_pointer_merge_error`].
    CrossBindingPointerMerge,
    /// None of the retry-routable families (keep the default bytes).
    Other,
}

/// Classify a spirv-val rejection into its [`ValidationClass`]. Equivalent, arm-for-arm, to the
/// ordered `Err(e) if is_*(e)` guard chain it replaces.
pub fn classify_validation_error(err: &str) -> ValidationClass {
    if is_pointer_typing_validation_error(err) {
        ValidationClass::PointerTyping
    } else if is_cfg_structurization_error(err) {
        ValidationClass::CfgStructurization
    } else if is_logical_pointer_phi_error(err) {
        ValidationClass::LogicalPointerPhi
    } else if is_cross_binding_pointer_merge_error(err) {
        ValidationClass::CrossBindingPointerMerge
    } else {
        ValidationClass::Other
    }
}

/// Structural class of a native-emitter `Err` returned *before* any spirv-val runs, used to route the
/// emit-failure retry cascade in `lib.rs`. Only the pointer-typing gap has a dedicated raw-byte
/// recovery arm; everything else (including any residual CFG-shaped emit gap) routes to the BDA/raw
/// last-resort cascade. (The dedicated cfg-emit-error arm was removed with the W4 repair-roster
/// deletion: the roster's repair fixpoint was the sole producer of a cfg-class emit *error*, so no
/// emit path returns one anymore — a reject function now emits complete unrepaired bytes that fail
/// *validation*, routing through the validation-side cfg cascade instead.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmitErrorClass {
    /// Buffer/pointer-typing gap the raw byte model can express — [`is_pointer_typing_emit_error`].
    PointerTyping,
    /// Any other emit failure.
    Other,
}

/// Classify a native-emitter emit `Err` into its [`EmitErrorClass`]. Equivalent, arm-for-arm, to the
/// ordered `Err(emit_err) if is_*(emit_err)` guard chain it replaces.
pub fn classify_emit_error(err: &str) -> EmitErrorClass {
    if is_pointer_typing_emit_error(err) {
        EmitErrorClass::PointerTyping
    } else {
        EmitErrorClass::Other
    }
}
