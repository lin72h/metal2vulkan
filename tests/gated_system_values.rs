//! A system value behind an off function constant costs the module nothing.
//!
//! `[[function_constant(x)]]` on an entry parameter means the parameter exists only when the
//! pipeline supplies `x` as true. Nothing supplies function-constant values at translate time, so a
//! gated parameter is absent and its uses are in statically dead code.
//!
//! For a *resource* that changes nothing: the descriptor stays in the layout either way, because
//! the pipeline layout has to match what the application binds. For a *system value* it matters. A
//! builtin declared for an absent parameter adds a variable to the entry point interface, and for
//! `[[viewport_array_index]]` and `[[render_target_array_index]]` it adds `ShaderViewportIndex` /
//! `ShaderLayer` to the module — device features the shader does not use, which narrow the set of
//! devices that can create the pipeline for nothing. Measured over a 2880-source corpus sample, 30
//! shaders declared exactly that.
//!
//! The vertex and kernel decodes never had the problem: their role reader collapses a gated
//! argument back to the function-constant wrapper unless it is a promoted resource. The fragment
//! reader looked past the wrapper unconditionally, which is the inconsistency this pins — from both
//! ends, because a decode that dropped the *ungated* case would satisfy half of it.

use metal2vulkan::passes::Stage;
use metal2vulkan::{disassemble, translate_sanitized_native};
use std::path::PathBuf;

fn tmp() -> PathBuf {
    let d = std::env::temp_dir().join(format!("m2v_gated_system_values_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

/// A fragment reading `[[viewport_array_index]]` behind an off-by-default function constant.
/// Substituting `GATE` away leaves the same shader with the parameter unconditionally present.
const GATED_VIEWPORT: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

@layered.MTL_FC_INIT_0_b = internal addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1

define <4 x float> @frag(<4 x float> %pos, i32 %viewport) {
entry:
  %f = uitofp i32 %viewport to float
  %v0 = insertelement <4 x float> undef, float %f, i32 0
  %v1 = insertelement <4 x float> %v0, float 0.000000e+00, i32 1
  %v2 = insertelement <4 x float> %v1, float 0.000000e+00, i32 2
  %v3 = insertelement <4 x float> %v2, float 1.000000e+00, i32 3
  ret <4 x float> %v3
}

!air.fragment = !{!0}
!air.function_constants = !{!7}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!5 = !{i32 1, GATE!"air.viewport_array_index", !"air.arg_type_name", !"uint", !"air.arg_name", !"viewport"}
!7 = !{ptr addrspace(2) @layered.MTL_FC_INIT_0_b, !"bool", !"layered", i32 0, i1 false}
"#;

fn translate_to_asm(gate: &str) -> String {
    let ll = GATED_VIEWPORT.replace("GATE", gate);
    assert_ne!(
        ll, GATED_VIEWPORT,
        "the gate placeholder must be substituted"
    );
    let spv = translate_sanitized_native(&ll, Stage::Fragment, &tmp()).expect("translate");
    disassemble(&spv).expect("disassemble")
}

fn mentions(asm: &str, needle: &str) -> bool {
    asm.lines().any(|line| line.contains(needle))
}

#[test]
fn a_gated_off_system_value_declares_no_builtin_and_no_capability() {
    let asm = translate_to_asm(r#"!"air.function_constant", !7, "#);
    assert!(
        !mentions(&asm, "BuiltIn ViewportIndex"),
        "an absent parameter must not claim its builtin:\n{asm}"
    );
    assert!(
        !mentions(&asm, "ShaderViewportIndex"),
        "and must not make the module require the device feature:\n{asm}"
    );
}

/// The other end: the same parameter without the gate must still be wired, or the fix would read as
/// "drop viewport indices" rather than "drop absent ones".
#[test]
fn the_same_system_value_ungated_is_still_wired() {
    let asm = translate_to_asm("");
    assert!(
        mentions(&asm, "BuiltIn ViewportIndex"),
        "a present parameter must be wired to its builtin:\n{asm}"
    );
    assert!(
        mentions(&asm, "OpCapability ShaderViewportIndex"),
        "and the module must declare what reading it needs:\n{asm}"
    );
}

/// A gated *resource* is the contrasting case: its descriptor is part of the pipeline layout the
/// application binds against, so it stays whether the constant is on or off.
#[test]
fn a_gated_off_resource_keeps_its_descriptor() {
    const GATED_BUFFER: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

@enabled.MTL_FC_INIT_0_b = internal addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1

define <4 x float> @frag(<4 x float> %pos, ptr addrspace(2) %cfg) {
entry:
  %n = load i32, ptr addrspace(2) %cfg
  %f = uitofp i32 %n to float
  %v0 = insertelement <4 x float> undef, float %f, i32 0
  %v1 = insertelement <4 x float> %v0, float 0.000000e+00, i32 1
  %v2 = insertelement <4 x float> %v1, float 0.000000e+00, i32 2
  %v3 = insertelement <4 x float> %v2, float 1.000000e+00, i32 3
  ret <4 x float> %v3
}

!air.fragment = !{!0}
!air.function_constants = !{!7}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!5 = !{i32 1, !"air.function_constant", !7, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"cfg"}
!7 = !{ptr addrspace(2) @enabled.MTL_FC_INIT_0_b, !"bool", !"enabled", i32 0, i1 false}
"#;

    let spv = translate_sanitized_native(GATED_BUFFER, Stage::Fragment, &tmp()).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        mentions(&asm, "Binding"),
        "a gated resource keeps the descriptor the pipeline layout declares:\n{asm}"
    );
}
