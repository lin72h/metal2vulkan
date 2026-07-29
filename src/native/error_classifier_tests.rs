//! Unit coverage for the retry-routing error classifiers in the parent module (§6 gap-list:
//! `native/mod.rs` error classifiers had zero isolated tests). These pure substring predicates decide
//! which failure-recovery retry a translate error routes to, keyed on SPIRV-Tools / native-emitter
//! message text — so a message-wording change silently mis-routes (or drops) a retry. The tests pin:
//!   * each predicate on a representative real message (positive) and an unrelated one (negative),
//!   * the aggregate `classify_*` routing, and
//!   * the LOAD-BEARING precedence: `classify_validation_error` must resolve a message matching more
//!     than one family to the same arm the ordered `is_*` guard chain in `lib.rs` fires first.

use super::{
    classify_emit_error, classify_validation_error, is_cfg_structurization_error,
    is_cross_binding_pointer_merge_error, is_dynamic_struct_index_error,
    is_logical_pointer_operand_error, is_logical_pointer_phi_error, is_pointer_typing_emit_error,
    is_pointer_typing_validation_error, EmitErrorClass, ValidationClass,
};

#[test]
fn other_class_refinement_predicates_match_their_exact_messages() {
    // Both refine within ValidationClass::Other for the two primary-emit rewrites in lib.rs; they
    // live here so validator-text matching stays in one adapter (S1).
    assert!(is_dynamic_struct_index_error(
        "Index into a structure must be an OpConstant"
    ));
    assert_eq!(
        classify_validation_error("Index into a structure must be an OpConstant"),
        ValidationClass::Other
    );
    assert!(!is_dynamic_struct_index_error(
        "OpPhi may only have a logical pointer operand"
    ));

    assert!(is_logical_pointer_operand_error(
        "Instruction may not have a logical pointer operand"
    ));
    assert_eq!(
        classify_validation_error("Instruction may not have a logical pointer operand"),
        ValidationClass::Other
    );
    // Distinct from the OpPhi-specific logical-pointer rule.
    assert!(!is_logical_pointer_operand_error(
        "OpPhi may only have a logical pointer operand"
    ));
    assert!(!is_logical_pointer_phi_error(
        "Instruction may not have a logical pointer operand"
    ));
}

#[test]
fn pointer_typing_validation_matches_access_chain_messages() {
    assert!(is_pointer_typing_validation_error(
        "OpInBoundsAccessChain cannot find index 3"
    ));
    assert!(is_pointer_typing_validation_error(
        "Result Type does not match Pointer type"
    ));
    assert!(is_pointer_typing_validation_error(
        "Index 2 is out of bounds: reached non-composite type"
    ));
    // Anchored access-chain index-type mismatch: the real message keys on "used to index".
    assert!(is_pointer_typing_validation_error(
        "Result Type does not match the type used to index into the composite"
    ));
    assert!(is_pointer_typing_validation_error(
        "The Object type (OpTypePointer) does not match the type that results from indexing into the Composite (OpTypePointer)."
    ));
    assert!(!is_pointer_typing_validation_error(
        "some unrelated parser failure"
    ));
    // M-C3: the anchored "used to index" no longer catches spurious "to index" wordings that the
    // bare substring did — those must fall through to `Other`, not misroute to the raw retry.
    assert!(!is_pointer_typing_validation_error(
        "diagnostic emitted prior to indexing the resource table"
    ));
    assert!(!is_pointer_typing_validation_error(
        "the pass failed to index the descriptor set"
    ));
}

#[test]
fn cfg_structurization_is_case_insensitive_over_its_family() {
    assert!(is_cfg_structurization_error(
        "Structured CFG rejected: unstructured back-edge"
    ));
    assert!(is_cfg_structurization_error(
        "Block 5 has a back-edge to a non-header block"
    ));
    // The classifier lowercases first, so an upper-case SPIRV-Tools phrasing still routes.
    assert!(is_cfg_structurization_error(
        "Block 5 appears in the binary before its dominator 3"
    ));
    assert!(is_cfg_structurization_error("BACK-EDGE detected"));
    // The naive-fallback message the repair path emits for a function that REJECTS `structured_plan`
    // (an S15 reject class) and falls back to `infer_*`+repair — two headers ending up sharing a
    // post-dominator merge block. Captured fresh 2026-07-08 from all three M-B1 repair-unique blockers
    // (05/b00a8a8d, 06/389737b0, 07/777b9bb4), which each produce it verbatim under `NO_REPAIR=1`. It
    // MUST route to CfgStructurization so the cfg restructure retry tier is what gets a shot at these.
    assert!(is_cfg_structurization_error(
        "Block '699[%699]' is already a merge block for another header"
    ));
    assert!(!is_cfg_structurization_error("bit width mismatch"));
}

/// M-C3 classifier hardening — the captured spirv-val cfg-message set.
///
/// The nine message FAMILIES below are every distinct structured-CFG rejection SPIRV-Tools actually
/// emits for the frontier private capture set, captured 2026-07-08 by running
/// `METAL2VULKAN_RETRY_DEBUG=1 historical validation tooling and de-duplicating the
/// `[retry-debug] default module failed cfg-class spirv-val:` lines (ids elided). Each MUST route to
/// `CfgStructurization` so the cfg-restructure / relooper retry tiers are what get a shot at it — a
/// message that fell through to `Other` (or was captured by a higher-precedence pointer family) would
/// silently drop the cfg-specific recovery. Pinning them here means a future needle edit (or a
/// SPIRV-Tools wording change) that breaks routing fails a unit test instead of a execution gate.
///
/// KEY FINDING (why the 17 needles are broad substrings, NOT anchored full phrases): a single needle
/// legitimately catches MULTIPLE distinct real phrasings, so anchoring to any one of them would lose
/// the others. `"merge block"` catches both `"... does not structurally dominate the merge block ..."`
/// (an admitted-module dominance reject) AND `"... is already a merge block for another header"` (the
/// naive-fallback the repair path emits for an S15-rejected function — the M-B1 blocker message above);
/// `"case construct"` catches the `"Multiple case constructs ..."`, `"Case construct that targets ..."`,
/// and invalid-branch phrasings. The breadth is load-bearing, so M-C3's "replace broad substrings with
/// anchored patterns" is verified UNNECESSARY for these needles — every one that fires here catches a
/// real cfg message, and every real cfg message is caught. The `"used to index"` anchoring (in
/// `is_pointer_typing_validation_error`) remains the correct fix for the ONE needle that had a genuine
/// spurious-substring problem.
#[test]
fn cfg_structurization_routes_captured_spirv_val_messages() {
    for msg in [
        "Block '4122[%4122]' appears in the binary before its dominator '4540[%4540]'",
        "block <ID> '600[%600]' exits the selection headed by <ID> '583[%583]', \
         but not via a structured exit",
        "block <ID> 4953 branches to the loop construct, but not to the loop header <ID> 4929",
        "block <ID> 2640 branches to the selection construct, \
         but not to the selection header <ID> 1849",
        "Case construct that targets '193[%193]' has invalid branch to block '226[%226]' \
         (not another case construct, corresponding merge, outer loop merge or outer loop continue)",
        "Multiple case constructs have branches to the case construct that targets '501[%501]'",
        "ID '1234[%1234]' defined in block '10[%10]' does not dominate its use in block '20[%20]'",
        "Selection must be structured",
        "The selection construct with the selection header '742[%742]' \
         does not structurally dominate the merge block '3760[%3760]'",
    ] {
        assert_eq!(
            classify_validation_error(msg),
            ValidationClass::CfgStructurization,
            "captured spirv-val cfg message must route to CfgStructurization: {msg}"
        );
    }
}

#[test]
fn logical_pointer_phi_and_cross_binding_match_their_exact_messages() {
    assert!(is_logical_pointer_phi_error(
        "OpPhi may only have a logical pointer operand"
    ));
    assert!(!is_logical_pointer_phi_error(
        "Variable pointers must point into the same structure"
    ));
    assert!(is_cross_binding_pointer_merge_error(
        "Variable pointers must point into the same structure or array"
    ));
    assert!(!is_cross_binding_pointer_merge_error(
        "OpPhi may only have a logical pointer operand"
    ));
}

#[test]
fn pointer_typing_emit_matches_reinterpret_and_storage_gaps() {
    assert!(is_pointer_typing_emit_error("load bit width mismatch"));
    assert!(is_pointer_typing_emit_error(
        "missing pointer storage class"
    ));
    assert!(is_pointer_typing_emit_error(
        "cannot reinterpret load of byte pointer"
    ));
    assert!(is_pointer_typing_emit_error(
        "cannot reinterpret workgroup pointer"
    ));
    assert!(!is_pointer_typing_emit_error(
        "Structured CFG rejected: unstructured back-edge"
    ));
}

#[test]
fn classify_validation_routes_each_family() {
    assert_eq!(
        classify_validation_error("OpInBoundsAccessChain cannot find index 0"),
        ValidationClass::PointerTyping
    );
    assert_eq!(
        classify_validation_error("loop header block 4 does not dominate its merge"),
        ValidationClass::CfgStructurization
    );
    assert_eq!(
        classify_validation_error("ID '4994[%4994]' has not been defined"),
        ValidationClass::CfgStructurization
    );
    assert_eq!(
        classify_validation_error("OpPhi may only have a logical pointer operand"),
        ValidationClass::LogicalPointerPhi
    );
    assert_eq!(
        classify_validation_error("Variable pointers must point into the same structure"),
        ValidationClass::CrossBindingPointerMerge
    );
    assert_eq!(
        classify_validation_error("unrelated intrinsic lowering failure"),
        ValidationClass::Other
    );
}

#[test]
fn classify_validation_precedence_is_pointer_typing_before_cfg() {
    // A message that trips BOTH the pointer-typing family ("does not match Pointer") and the cfg
    // family ("back-edge") must resolve to PointerTyping — the first arm of the guard chain.
    let both = "Result does not match Pointer type at a back-edge";
    assert!(is_pointer_typing_validation_error(both));
    assert!(is_cfg_structurization_error(both));
    assert_eq!(
        classify_validation_error(both),
        ValidationClass::PointerTyping
    );
}

#[test]
fn classify_validation_precedence_phi_before_cross_binding() {
    // Logical-pointer-phi outranks cross-binding when a message matches both.
    let both = "OpPhi may only have a logical pointer operand; \
                Variable pointers must point into the same structure";
    assert!(is_logical_pointer_phi_error(both));
    assert!(is_cross_binding_pointer_merge_error(both));
    assert_eq!(
        classify_validation_error(both),
        ValidationClass::LogicalPointerPhi
    );
}

#[test]
fn classify_emit_routes_pointer_typing_then_other() {
    assert_eq!(
        classify_emit_error("load bit width mismatch"),
        EmitErrorClass::PointerTyping
    );
    // The dedicated cfg-emit-error arm was removed with the W4 repair-roster deletion (no emit path
    // returns a cfg-class emit error anymore); a residual cfg-shaped emit gap now routes to Other.
    assert_eq!(
        classify_emit_error("native emitter: structured cfg rejected"),
        EmitErrorClass::Other
    );
    assert_eq!(
        classify_emit_error("out of registers"),
        EmitErrorClass::Other
    );
    // A message in both emit families still resolves to PointerTyping (checked first).
    let both = "missing pointer storage in a structured cfg block";
    assert!(is_pointer_typing_emit_error(both));
    assert!(is_cfg_structurization_error(both));
    assert_eq!(classify_emit_error(both), EmitErrorClass::PointerTyping);
}
