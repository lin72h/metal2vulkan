//! S1: the typed error classifiers (`classify_validation_error` / `classify_emit_error`) must be
//! arm-for-arm equivalent to the prior `Err(e) if is_*(e)` guard chains in `lib.rs`. These tests
//! feed captured/representative spirv-val and native-emitter messages — including the exact
//! substrings each matcher keys on — and assert the classifier both (a) picks the expected class and
//! (b) equals that guard chain computed in the same precedence. Pure-static (no external tools).

use super::super::{
    classify_emit_error, classify_validation_error, is_cfg_structurization_error,
    is_cross_binding_pointer_merge_error, is_deferred_pointer_materialization_emit_error,
    is_graph_walk_unmigrated_emit_error, is_logical_pointer_phi_error,
    is_pointer_typing_emit_error, is_pointer_typing_validation_error, EmitErrorClass,
    ValidationClass,
};

/// The prior validation guard chain, in lib.rs precedence order — what the classifier must match.
fn guard_chain_validation(err: &str) -> ValidationClass {
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

fn guard_chain_emit(err: &str) -> EmitErrorClass {
    if is_deferred_pointer_materialization_emit_error(err) {
        EmitErrorClass::DeferredPointerMaterialization
    } else if is_pointer_typing_emit_error(err) {
        EmitErrorClass::PointerTyping
    } else {
        EmitErrorClass::Other
    }
}

// Representative spirv-val rejections (realistic full lines embedding each keyed substring).
const VALIDATION_CASES: &[(&str, ValidationClass)] = &[
    (
        "error: line 42: OpAccessChain Result Type does not match the type in the base composite; \
         result type does not match Pointer type",
        ValidationClass::PointerTyping,
    ),
    (
        "error: OpStore Pointer type and Object type do not match: does not match Object",
        ValidationClass::PointerTyping,
    ),
    (
        "OpInBoundsAccessChain reached non-composite type while indexes still remain to be traversed",
        ValidationClass::PointerTyping,
    ),
    (
        "error: OpInBoundsAccessChain cannot find index 3 in the structure",
        ValidationClass::PointerTyping,
    ),
    (
        "error: Result Type does not match the type used to index",
        ValidationClass::PointerTyping,
    ),
    (
        "error: The continue construct with the continue target has invalid back-edge",
        ValidationClass::CfgStructurization,
    ),
    (
        "error: Block 12 must be structured as a selection construct",
        ValidationClass::CfgStructurization,
    ),
    (
        "error: Block 7 appears in the binary before its dominator 3",
        ValidationClass::CfgStructurization,
    ),
    (
        "error: Header block 4 does not strictly dominate its merge block 9",
        ValidationClass::CfgStructurization,
    ),
    (
        "error: Structured CFG rejected: unstructured back-edge",
        ValidationClass::CfgStructurization,
    ),
    (
        "error: In Logical addressing OpPhi variable may only have a logical pointer operand",
        ValidationClass::LogicalPointerPhi,
    ),
    (
        "error: Variable pointers must point into the same structure or elements of the same array",
        ValidationClass::CrossBindingPointerMerge,
    ),
    (
        "error: some unrelated capability is missing for OpFoo",
        ValidationClass::Other,
    ),
    ("", ValidationClass::Other),
];

const EMIT_CASES: &[(&str, EmitErrorClass)] = &[
    (
        "native emitter: selected pointer %1557 missing metadata",
        EmitErrorClass::DeferredPointerMaterialization,
    ),
    (
        "native emitter: reinterpret bit width mismatch: buffer pointee 4 vs load 8",
        EmitErrorClass::PointerTyping,
    ),
    (
        "native emitter: missing pointer storage for %ptr",
        EmitErrorClass::PointerTyping,
    ),
    (
        "native emitter: cannot reinterpret load of byte pointer %p to <4 x float>",
        EmitErrorClass::PointerTyping,
    ),
    (
        "native emitter: cannot reinterpret workgroup pointer %tg",
        EmitErrorClass::PointerTyping,
    ),
    (
        "native emitter: cannot store through reinterpreted pointer select %sel",
        EmitErrorClass::PointerTyping,
    ),
    // The dedicated cfg-emit-error arm was removed with the W4 repair-roster deletion (no emit path
    // returns a cfg-class emit error anymore), so a residual cfg-shaped emit message routes to Other.
    (
        "native emitter: structured cfg rejected for an unstructured back-edge",
        EmitErrorClass::Other,
    ),
    (
        "native emitter: structured cfg has a non-header merge",
        EmitErrorClass::Other,
    ),
    (
        "native emitter: unknown callee @air.intersect.f32.i32",
        EmitErrorClass::Other,
    ),
    (
        "native emitter: instruction not handled by the typed graph walk \
         (reason=graph_walk_unmigrated_opcode, opcode=load)",
        EmitErrorClass::Other,
    ),
    ("", EmitErrorClass::Other),
];

#[test]
fn validation_classifier_matches_expected_and_guard_chain() {
    for (msg, expected) in VALIDATION_CASES {
        let got = classify_validation_error(msg);
        assert_eq!(got, *expected, "classify_validation_error({msg:?})");
        assert_eq!(
            got,
            guard_chain_validation(msg),
            "classifier diverged from the prior guard chain for {msg:?}"
        );
    }
}

#[test]
fn emit_classifier_matches_expected_and_guard_chain() {
    for (msg, expected) in EMIT_CASES {
        let got = classify_emit_error(msg);
        assert_eq!(got, *expected, "classify_emit_error({msg:?})");
        assert_eq!(
            got,
            guard_chain_emit(msg),
            "classifier diverged from the prior guard chain for {msg:?}"
        );
    }
}

#[test]
fn graph_walk_unmigrated_emit_error_is_detected_for_retry_feed_budgeting() {
    assert!(is_graph_walk_unmigrated_emit_error(
        "native emitter: instruction not handled by the typed graph walk \
         (reason=graph_walk_unmigrated_opcode, opcode=load)"
    ));
    assert!(!is_graph_walk_unmigrated_emit_error(
        "native emitter: missing pointer storage for %ptr"
    ));
}

#[test]
fn deferred_pointer_witness_maps_cfg_clone_back_to_source_ssa() {
    assert_eq!(
        super::super::deferred_pointer_materialization_name(
            "native emitter: selected pointer %xa4000000_1557 missing metadata"
        ),
        Some("%1557".to_string())
    );
}

// The precedence is load-bearing: a message matching BOTH pointer-typing and cfg families must
// classify as pointer-typing (the first guard arm), exactly as the old chain did.
#[test]
fn precedence_prefers_pointer_typing_over_cfg() {
    let both = "does not match Pointer; and also a back-edge violation in the loop header";
    assert!(is_pointer_typing_validation_error(both));
    assert!(is_cfg_structurization_error(both));
    assert_eq!(
        classify_validation_error(both),
        ValidationClass::PointerTyping
    );
    assert_eq!(guard_chain_validation(both), ValidationClass::PointerTyping);
}
