//! A `[[sample_mask]]` return member is written, not dropped.
//!
//! Metal's `[[sample_mask]]` output is a coverage mask: each bit the fragment clears removes that
//! sample from the write. Shaders use it for custom alpha-to-coverage, for order-independent
//! transparency dithering, and for hair/foliage — anywhere the shape inside a pixel is finer than
//! the pixel. A shader that returns a mask and has it dropped writes *every* covered sample, so the
//! cutout it was computing simply does not happen.
//!
//! Nothing else in the pipeline notices. The output-member walk recognised `air.render_target`,
//! `air.depth` and `air.stencil`, and skipped any member whose role it did not know — a `continue`,
//! not a diagnostic — so the module validated, exposed the same descriptors, and reflected
//! identically. That silence is the reason this needs an authored test: the corpus sample carries
//! only one shader that would exercise it, and every automatic check passes either way.
//!
//! SPIR-V's `SampleMask` is an *array* of `uint` (element N covering samples 32N..32N+31) while
//! Metal returns one scalar, so the write is an access chain into element 0 rather than a store to
//! the variable — the same shape `ClipDistance` already needed, which is why the two now share one
//! lowering.

use metal2vulkan::passes::Stage;
use metal2vulkan::{disassemble, translate_sanitized_native};
use std::collections::HashMap;
use std::path::PathBuf;

fn tmp() -> PathBuf {
    let d = std::env::temp_dir().join(format!("m2v_fragment_sample_mask_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

/// A fragment returning `{ color, sample_mask }` — the mask as one member of a return struct.
const MASK_WITH_COLOR: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define <{ <4 x float>, i32 }> @frag(<4 x float> %pos, i32 %mask) {
entry:
  %r0 = insertvalue <{ <4 x float>, i32 }> undef, <4 x float> %pos, 0
  %r1 = insertvalue <{ <4 x float>, i32 }> %r0, i32 %mask, 1
  ret <{ <4 x float>, i32 }> %r1
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!3 = !{!"air.sample_mask", !"air.arg_type_name", !"uint", !"air.arg_name", !"coverage"}
!4 = !{!5, !6}
!5 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!6 = !{i32 1, !"air.fragment_input", !"generated(m)", !"air.flat", !"air.arg_type_name", !"uint", !"air.arg_name", !"mask"}
"#;

/// A depth/coverage-only fragment: the mask IS the return value, so it never passes through a
/// return struct. That arm of the output lowering decides the builtin separately.
const MASK_ALONE: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define i32 @frag(<4 x float> %pos, i32 %mask) {
entry:
  ret i32 %mask
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !4}
!1 = !{!3}
!3 = !{!"air.sample_mask", !"air.arg_type_name", !"uint", !"air.arg_name", !"coverage"}
!4 = !{!5, !6}
!5 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!6 = !{i32 1, !"air.fragment_input", !"generated(m)", !"air.flat", !"air.arg_type_name", !"uint", !"air.arg_name", !"mask"}
"#;

fn translate_to_asm(ll: &str) -> String {
    let spv = translate_sanitized_native(ll, Stage::Fragment, &tmp()).expect("translate");
    disassemble(&spv).expect("disassemble")
}

/// Every `%id = <opcode> <operands…>` definition in the module, tokenized. The crate's own
/// disassembler prints numeric ids, so the assertions below walk the definition graph rather than
/// matching friendly type names that only `spirv-dis` invents.
fn definitions(asm: &str) -> HashMap<String, Vec<String>> {
    asm.lines()
        .map(str::trim)
        .filter_map(|line| {
            let (result, rhs) = line.split_once(" = ")?;
            result.starts_with('%').then(|| {
                (
                    result.to_string(),
                    rhs.split_whitespace().map(str::to_string).collect(),
                )
            })
        })
        .collect()
}

/// The id of the variable decorated `BuiltIn SampleMask`.
fn sample_mask_var(asm: &str) -> Option<String> {
    asm.lines().map(str::trim).find_map(|line| {
        let rest = line.strip_prefix("OpDecorate ")?;
        let (id, tail) = rest.split_once(' ')?;
        (tail == "BuiltIn SampleMask").then(|| id.to_string())
    })
}

#[test]
fn a_sample_mask_return_member_reaches_the_builtin() {
    for source in [MASK_WITH_COLOR, MASK_ALONE] {
        let asm = translate_to_asm(source);
        let var = sample_mask_var(&asm)
            .unwrap_or_else(|| panic!("no variable decorated BuiltIn SampleMask:\n{asm}"));
        assert!(
            asm.lines()
                .any(|line| line.contains("OpEntryPoint") && line.contains(&var)),
            "the mask variable must be part of the entry point interface:\n{asm}"
        );
    }
}

/// The value must land in element 0 of the array, not in the variable itself — a store straight to
/// a `uint[1]` Output is a type mismatch, and a scalar-typed variable is the wrong builtin shape.
#[test]
fn the_mask_is_stored_through_element_zero_of_the_array() {
    for source in [MASK_WITH_COLOR, MASK_ALONE] {
        let asm = translate_to_asm(source);
        let defs = definitions(&asm);
        let var = sample_mask_var(&asm).expect("SampleMask variable");

        let variable = &defs[&var];
        assert_eq!(variable.first().map(String::as_str), Some("OpVariable"));
        let pointer = &defs[&variable[1]];
        assert_eq!(
            pointer.first().map(String::as_str),
            Some("OpTypePointer"),
            "{asm}"
        );
        assert_eq!(
            defs[&pointer[2]].first().map(String::as_str),
            Some("OpTypeArray"),
            "SampleMask must be an array-typed Output variable:\n{asm}"
        );

        let (chain, index) = defs
            .iter()
            .find_map(|(result, rhs)| {
                (rhs.first().map(String::as_str) == Some("OpAccessChain")
                    && rhs.get(2) == Some(&var))
                .then(|| (result.clone(), rhs[3].clone()))
            })
            .unwrap_or_else(|| panic!("no OpAccessChain into {var}:\n{asm}"));
        let constant = &defs[&index];
        assert_eq!(constant.first().map(String::as_str), Some("OpConstant"));
        assert_eq!(
            constant.last().map(String::as_str),
            Some("0"),
            "the mask covers samples 0..31, which is element 0:\n{asm}"
        );
        assert!(
            asm.lines()
                .map(str::trim)
                .any(|line| line.starts_with(&format!("OpStore {chain} "))),
            "the mask must be stored through the access chain:\n{asm}"
        );
    }
}

/// A fragment that declares no mask must not sprout one — an always-present `SampleMask` output
/// makes every pipeline sample-masked whether the shader asked or not.
#[test]
fn a_fragment_without_a_mask_declares_no_sample_mask_output() {
    let plain = MASK_WITH_COLOR.replace("air.sample_mask", "air.render_target");
    assert_ne!(plain, MASK_WITH_COLOR);
    let asm = translate_to_asm(&plain);
    assert!(
        sample_mask_var(&asm).is_none(),
        "no SampleMask builtin should appear:\n{asm}"
    );
}
