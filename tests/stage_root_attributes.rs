//! Every attribute an AIR stage root carries is read, or the translation refuses.
//!
//! All three stage roots — `!air.kernel`, `!air.vertex`, `!air.fragment` — state their per-entry
//! attributes in the same place: extra operands on the root node, past the
//! `(function, outputs, inputs)` triple. They do not state them in the same *form*. `air.patch`
//! and `air.max_work_group_size` are references to a keyed node; `early_fragment_tests` is a bare
//! string operand on the root itself. A decode written for one form cannot see the other.
//!
//! That is how `[[early_fragment_tests]]` went missing. The vertex decode searched the tail for
//! `air.patch` and the other two stages did not look at all, so 51 of a 2880-source corpus sample
//! declared early depth testing and were emitted with Vulkan's default late test. The difference
//! is not cosmetic: under early tests a fragment the depth test rejects executes none of the
//! body's stores, while under late tests the same shader performs every buffer, texture and
//! imageblock write and only its color output is thrown away. Both modules validate, bind and
//! reflect identically.
//!
//! `air.max_work_group_size` — `[[max_total_threads_per_threadgroup(N)]]` — was unread on 439
//! sources in the same sample. It states the largest threadgroup the body was compiled for, so a
//! caller-requested `LocalSize` wider than it is outside the entry's own contract.
//!
//! The point of reading the tail in one place is the last case here: an attribute this translator
//! has never seen becomes a named refusal from whichever stage carries it, instead of the silence
//! that let the first two sit unread.

use metal2vulkan::passes::{Stage, TransformOptions};
use metal2vulkan::reflect::KernelDispatch;
use metal2vulkan::{
    disassemble, reflect_sanitized, translate_sanitized_native,
    translate_sanitized_native_with_options,
};
use std::path::PathBuf;

fn tmp() -> PathBuf {
    let d = std::env::temp_dir().join(format!("m2v_stage_root_attributes_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

/// A fragment that writes a device buffer and returns a color. `ATTR` is spliced into the root's
/// attribute tail, so the only difference between the arms below is the attribute itself.
const FRAGMENT: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @frag(<4 x float> %pos, ptr addrspace(1) %out) {
entry:
  store float 1.000000e+00, ptr addrspace(1) %out, align 4
  ret <4 x float> %pos
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3ATTR}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float*", !"air.arg_name", !"out"}
"#;

/// A fragment returning `{ color, depth }`: the body computes the value the depth test compares.
const FRAGMENT_WITH_DEPTH: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define <{ <4 x float>, float }> @frag(<4 x float> %pos) {
entry:
  %d = extractelement <4 x float> %pos, i32 2
  %r0 = insertvalue <{ <4 x float>, float }> undef, <4 x float> %pos, 0
  %r1 = insertvalue <{ <4 x float>, float }> %r0, float %d, 1
  ret <{ <4 x float>, float }> %r1
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !4ATTR}
!1 = !{!2, !3}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!"air.depth", !"air.any", !"air.arg_type_name", !"float"}
!4 = !{!5}
!5 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
"#;

/// The same shape for `[[stencil]]`, whose value is an integer reference rather than a depth.
const FRAGMENT_WITH_STENCIL: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define <{ <4 x float>, i32 }> @frag(<4 x float> %pos) {
entry:
  %d = extractelement <4 x float> %pos, i32 2
  %s = bitcast float %d to i32
  %r0 = insertvalue <{ <4 x float>, i32 }> undef, <4 x float> %pos, 0
  %r1 = insertvalue <{ <4 x float>, i32 }> %r0, i32 %s, 1
  ret <{ <4 x float>, i32 }> %r1
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !4ATTR}
!1 = !{!2, !3}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!"air.stencil", !"air.arg_type_name", !"uint"}
!4 = !{!5}
!5 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
"#;

/// A kernel whose root carries `ATTR` as its fourth operand.
const KERNEL: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define void @k(ptr addrspace(1) %out) {
entry:
  store float 1.000000e+00, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2ATTR}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float*", !"air.arg_name", !"out"}
"#;

/// An ordinary vertex whose root carries `ATTR` where a post-tessellation one carries `air.patch`.
const VERTEX: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @vert(i32 %vid) {
entry:
  %f = sitofp i32 %vid to float
  %v0 = insertelement <4 x float> undef, float %f, i32 0
  %v1 = insertelement <4 x float> %v0, float 0.000000e+00, i32 1
  %v2 = insertelement <4 x float> %v1, float 0.000000e+00, i32 2
  %v3 = insertelement <4 x float> %v2, float 1.000000e+00, i32 3
  ret <4 x float> %v3
}

!air.vertex = !{!0}
!0 = !{ptr @vert, !1, !3ATTR}
!1 = !{!2}
!2 = !{!"air.position", !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
"#;

/// Splice an attribute tail into a template. `attr` is the operand text plus any node definitions
/// it needs; an empty `attr` yields the same shader with a bare root.
fn with_attribute(template: &str, attr: &str) -> String {
    let spliced = template.replace("ATTR", attr);
    assert_ne!(spliced, template, "template has no ATTR splice point");
    spliced
}

fn translate(template: &str, attr: &str, stage: Stage) -> Result<String, String> {
    let ll = with_attribute(template, attr);
    let spv = translate_sanitized_native(&ll, stage, &tmp()).map_err(|error| error.to_string())?;
    Ok(disassemble(&spv).expect("disassemble"))
}

/// Translate a kernel under an explicit whole-workgroup dispatch, so the requested local size
/// reaches SPIR-V as a `LocalSize` execution mode rather than as specialization constants.
fn dispatch(template: &str, attr: &str, local_size: [u32; 3]) -> Result<String, String> {
    let ll = with_attribute(template, attr);
    let options = TransformOptions {
        kernel_local_size: local_size,
        kernel_dispatch: Some(KernelDispatch::Workgroups),
        ..TransformOptions::default()
    };
    let spv = translate_sanitized_native_with_options(&ll, Stage::Kernel, &tmp(), options)?;
    Ok(disassemble(&spv).expect("disassemble"))
}

/// The `OpExecutionMode` modes declared for the module's single entry point.
fn execution_modes(asm: &str) -> Vec<String> {
    asm.lines()
        .map(str::trim)
        .filter_map(|line| line.strip_prefix("OpExecutionMode "))
        .filter_map(|rest| rest.split_once(' ').map(|(_, mode)| mode.to_string()))
        .collect()
}

#[test]
fn early_fragment_tests_reaches_the_execution_mode() {
    let declared = translate(FRAGMENT, ", !\"early_fragment_tests\"", Stage::Fragment)
        .expect("fragment declaring early_fragment_tests translates");
    assert!(
        execution_modes(&declared)
            .iter()
            .any(|mode| mode == "EarlyFragmentTests"),
        "a fragment declaring early_fragment_tests must say so in SPIR-V:\n{declared}"
    );
}

/// The other half: a fragment that did NOT ask for early testing must not be given it. Early
/// testing is observable — it suppresses the stores of a depth-rejected fragment — so adding it
/// unasked is as wrong as dropping it.
#[test]
fn a_fragment_that_did_not_ask_for_early_tests_does_not_get_them() {
    let plain = translate(FRAGMENT, "", Stage::Fragment).expect("plain fragment translates");
    assert!(
        !execution_modes(&plain)
            .iter()
            .any(|mode| mode == "EarlyFragmentTests"),
        "nothing declared early_fragment_tests:\n{plain}"
    );
    assert!(
        execution_modes(&plain)
            .iter()
            .any(|mode| mode == "OriginUpperLeft"),
        "the fragment's ordinary modes must survive:\n{plain}"
    );
}

/// `[[early_fragment_tests]]` runs the depth and stencil tests before the body, so the body
/// cannot be the source of a value either test compares. Metal rejects the pair; so must this.
#[test]
fn early_fragment_tests_and_a_written_test_value_are_refused() {
    for (template, role) in [
        (FRAGMENT_WITH_DEPTH, "air.depth"),
        (FRAGMENT_WITH_STENCIL, "air.stencil"),
    ] {
        let error = translate(template, ", !\"early_fragment_tests\"", Stage::Fragment)
            .expect_err("a fragment cannot write a test value and demand the test run first");
        assert!(
            error.contains("early_fragment_tests") && error.contains(role),
            "the refusal must name both halves of the contradiction: {error}"
        );

        // Each half alone is fine, so the refusal is about the pair, not about either one.
        translate(template, "", Stage::Fragment)
            .unwrap_or_else(|error| panic!("writing {role} alone must translate: {error}"));
    }
}

/// The default dispatch is 64 threads. A kernel compiled for at most 32 must not be emitted with
/// a `LocalSize` twice the size its own body was built for.
/// A `LocalSize` is a request from the caller; the ceiling is a fact about the body. A dispatch
/// of 64 threads must not be emitted for a kernel compiled for at most 32, and must be emitted
/// unchanged for one compiled for 64.
#[test]
fn a_dispatch_past_the_declared_threadgroup_ceiling_is_refused() {
    let ceiling =
        |threads: u32| format!(", !6}}\n!6 = !{{!\"air.max_work_group_size\", i32 {threads}");

    let error =
        dispatch(KERNEL, &ceiling(32), [64, 1, 1]).expect_err("64 threads is past a ceiling of 32");
    assert!(
        error.contains("air.max_work_group_size") && error.contains("32"),
        "the refusal must name the ceiling it read: {error}"
    );

    // The same ceiling with a dispatch that fits inside it, and the same dispatch under a ceiling
    // that admits it: the check is about the pair, not about either value.
    for (attr, local_size) in [
        (ceiling(32), [8u32, 2, 2]),
        (ceiling(64), [64, 1, 1]),
        (ceiling(729), [9, 9, 9]),
    ] {
        let [x, y, z] = local_size;
        let asm = dispatch(KERNEL, &attr, local_size)
            .unwrap_or_else(|error| panic!("{x}x{y}x{z} must fit `{attr}`: {error}"));
        assert!(
            execution_modes(&asm)
                .iter()
                .any(|mode| mode == &format!("LocalSize {x} {y} {z}")),
            "the requested local size must still be emitted:\n{asm}"
        );
    }
}

/// The ceiling reflection reports is the ceiling translation enforces.
///
/// The refusal above is only usable if a consumer can find out what it is before asking. Reflection
/// reports `air.max_work_group_size` and does not enforce it -- discovering the bound is the reason
/// to call it -- so the two numbers have to be the same number, and this drives both off one
/// declaration. A kernel that declares nothing reports nothing and accepts any dispatch.
#[test]
fn the_reflected_threadgroup_ceiling_is_the_one_translation_enforces() {
    let declared = with_attribute(KERNEL, ", !6}\n!6 = !{!\"air.max_work_group_size\", i32 64");
    let reflection = reflect_sanitized(&declared, Stage::Kernel, TransformOptions::default())
        .expect("reflection reports the ceiling rather than enforcing it");
    let ceiling = reflection
        .max_work_group_size
        .expect("the kernel declares a ceiling");
    assert_eq!(ceiling, 64);

    // At the ceiling and past it, driven off the number reflection just reported.
    let at = [8, ceiling / 8, 1];
    let past = [8, ceiling / 8, 2];
    dispatch(
        KERNEL,
        ", !6}\n!6 = !{!\"air.max_work_group_size\", i32 64",
        at,
    )
    .expect("a dispatch of exactly the reported ceiling translates");
    let error = dispatch(
        KERNEL,
        ", !6}\n!6 = !{!\"air.max_work_group_size\", i32 64",
        past,
    )
    .expect_err("a dispatch past the reported ceiling is refused");
    assert!(
        error.contains(&ceiling.to_string()),
        "the refusal must cite the ceiling reflection reported: {error}"
    );

    // No declaration, no ceiling, and nothing to enforce.
    let plain = with_attribute(KERNEL, "");
    let plain_reflection = reflect_sanitized(&plain, Stage::Kernel, TransformOptions::default())
        .expect("a kernel with no ceiling still reflects");
    assert_eq!(plain_reflection.max_work_group_size, None);
    dispatch(KERNEL, "", [16, 16, 4]).expect("an undeclared ceiling bounds nothing");
}

/// The structural half: an attribute no stage models is refused by every stage that can carry one,
/// naming what it read. Silence here is what let the two attributes above go unread.
#[test]
fn an_attribute_no_stage_models_is_refused_by_name() {
    let cases = [
        (FRAGMENT, Stage::Fragment, "fragment"),
        (KERNEL, Stage::Kernel, "kernel"),
        (VERTEX, Stage::Vertex, "vertex"),
    ];
    for (template, stage, label) in cases {
        // Every arm translates without the attribute, so the refusal below is caused by it.
        translate(template, "", stage)
            .unwrap_or_else(|error| panic!("{label} must translate with a bare root: {error}"));

        let error = match translate(template, ", !\"air.invented_stage_attribute\"", stage) {
            Ok(asm) => panic!("{label} accepted an unmodelled attribute:\n{asm}"),
            Err(error) => error,
        };
        assert!(
            error.contains("air.invented_stage_attribute"),
            "the {label} refusal must name the attribute it could not read: {error}"
        );
    }
}

/// An attribute this translator models for a *different* stage is still unmodelled here. A kernel
/// carrying `early_fragment_tests` is not a kernel with early depth testing.
#[test]
fn an_attribute_belonging_to_another_stage_is_not_silently_accepted() {
    let error = translate(KERNEL, ", !\"early_fragment_tests\"", Stage::Kernel)
        .expect_err("a kernel has no fragment tests to run early");
    assert!(
        error.contains("early_fragment_tests"),
        "the refusal must name the attribute: {error}"
    );
}
