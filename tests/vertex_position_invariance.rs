//! `[[position, invariant]]` reaches the module as `Invariant`.
//!
//! Metal's `invariant` on a vertex position is a bit-exactness guarantee across pipelines: the same
//! input vertex, transformed by the same arithmetic in two different pipeline states, must produce
//! the identical clip position. Renderers depend on it wherever two passes have to agree about
//! depth — a depth prepass and the shading pass that tests `EQUAL` against it, or a stencil-marking
//! pass and its consumer. Without it a driver is free to reassociate or contract the arithmetic
//! differently per pipeline, and the two passes disagree by an ulp, which shows up as z-fighting
//! along every triangle.
//!
//! Vulkan spells the same guarantee `OpDecorate <position> Invariant`. AIR spells it as an
//! `air.invariant` marker on the position's output metadata node, next to the markers describing
//! its type and name — so a decode that reads only the role sees a plain `air.position` and drops
//! the request. Nothing downstream notices: the module validates, the interface is unchanged, and
//! reflection reports the same bindings. 20 of a 2880-source corpus sample declare it.
//!
//! `fragment_input_interpolation.rs` pins the same class on the other side of the pipeline — an AIR
//! qualifier whose only trace in SPIR-V is a decoration, and whose loss is invisible to every
//! automatic check.

use metal2vulkan::passes::Stage;
use metal2vulkan::{disassemble, translate_sanitized_native};
use std::path::PathBuf;

fn tmp() -> PathBuf {
    let d = std::env::temp_dir().join(format!("m2v_position_invariance_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

/// A vertex shader returning `{ position, uv }`, with the position declared invariant. The second
/// member exists so the position is one member of a struct rather than the whole return value —
/// the two travel through different arms of the output lowering.
const INVARIANT_VERTEX: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define <{ <4 x float>, <2 x float> }> @vert(<4 x float> %p, <2 x float> %uv) {
entry:
  %r0 = insertvalue <{ <4 x float>, <2 x float> }> undef, <4 x float> %p, 0
  %r1 = insertvalue <{ <4 x float>, <2 x float> }> %r0, <2 x float> %uv, 1
  ret <{ <4 x float>, <2 x float> }> %r1
}

!air.vertex = !{!0}
!0 = !{ptr @vert, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.position", !"air.invariant", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!"air.vertex_output", !"generated(uv)", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!4 = !{!5, !6}
!5 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}
!6 = !{i32 1, !"air.vertex_input", !"air.location_index", i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
"#;

/// A vertex shader whose whole return value is the position — the non-struct arm of the same
/// lowering, which decides the decoration separately.
const BARE_INVARIANT_VERTEX: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

define <4 x float> @vert(<4 x float> %p) {
entry:
  ret <4 x float> %p
}

!air.vertex = !{!0}
!0 = !{ptr @vert, !1, !3}
!1 = !{!2}
!2 = !{!"air.position", !"air.invariant", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!3 = !{!4}
!4 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"p"}
"#;

fn translate_to_asm(ll: &str) -> String {
    let spv = translate_sanitized_native(ll, Stage::Vertex, &tmp()).expect("translate");
    disassemble(&spv).expect("disassemble")
}

/// The id of the variable decorated `BuiltIn Position`, and whether it is also `Invariant`.
fn position_is_invariant(asm: &str) -> bool {
    let position = asm
        .lines()
        .map(str::trim)
        .find_map(|line| {
            let rest = line.strip_prefix("OpDecorate ")?;
            let (id, tail) = rest.split_once(' ')?;
            (tail == "BuiltIn Position").then(|| id.to_string())
        })
        .unwrap_or_else(|| panic!("no variable decorated BuiltIn Position:\n{asm}"));
    asm.lines()
        .map(str::trim)
        .any(|line| line == format!("OpDecorate {position} Invariant"))
}

#[test]
fn an_invariant_position_is_decorated_invariant() {
    assert!(
        position_is_invariant(&translate_to_asm(INVARIANT_VERTEX)),
        "`air.invariant` on a struct member must reach the Position variable"
    );
    assert!(
        position_is_invariant(&translate_to_asm(BARE_INVARIANT_VERTEX)),
        "`air.invariant` on a bare position return must reach it too"
    );
}

/// The decoration is a promise about the whole pipeline, so claiming it where Metal did not is as
/// wrong as dropping it: a driver may forgo optimizations to honour it.
#[test]
fn a_position_metal_did_not_declare_invariant_is_not_decorated() {
    for source in [INVARIANT_VERTEX, BARE_INVARIANT_VERTEX] {
        let plain = source.replace(r#", !"air.invariant""#, "");
        assert_ne!(plain, source, "the marker must be removed");
        assert!(
            !position_is_invariant(&translate_to_asm(&plain)),
            "a position without `air.invariant` must not be decorated:\n{plain}"
        );
    }
}

/// The marker is read off the member it sits on, not off the module.
#[test]
fn invariance_does_not_leak_to_the_other_outputs() {
    let asm = translate_to_asm(INVARIANT_VERTEX);
    let invariant_ids = asm
        .lines()
        .map(str::trim)
        .filter_map(|line| line.strip_suffix(" Invariant")?.strip_prefix("OpDecorate "))
        .count();
    assert_eq!(
        invariant_ids, 1,
        "only the position is invariant; the `uv` varying is not:\n{asm}"
    );
}
