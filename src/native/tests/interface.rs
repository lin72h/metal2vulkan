#![allow(unused_imports)]
use super::super::cfg::{
    id_ref_operand, infer_branch_merges, infer_loop_merges, infer_switch_merges,
    lower_unstructured_switches, split_body_blocks, BodyBlock,
};
use super::super::emit_vulkan_spirv;
use super::super::emitter::Emitter;
use super::super::ir::{LlType, LlValue};
use super::super::parse::{parse_type, parse_typed_value};
use super::*;
use crate::passes::{self, Stage};
use crate::spirv_module::load_bytes;
use crate::spirv_module::Operand;
use crate::spirv_module::{Block, Instruction};
use crate::{disassemble, meta, tools};
use spirv::{BuiltIn, Capability, Decoration, Op, Scope, SelectionControl, StorageClass, Word};
use std::collections::{HashMap, HashSet};

#[test]
fn native_fragment_depth_return_maps_to_frag_depth() {
    let ll = r#"
source_filename = "synth_depth"
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <{ float }> @synth_depth(<4 x float> %position) local_unnamed_addr #0 {
  %z = extractelement <4 x float> %position, i64 2
  %2 = insertvalue <{ float }> undef, float %z, 0
  ret <{ float }> %2
}

attributes #0 = { nounwind }

!air.fragment = !{!0}
!0 = !{ptr @synth_depth, !1, !3}
!1 = !{!2}
!2 = !{!"air.depth", !"air.depth_qualifier", !"air.any", !"air.arg_type_name", !"float", !"air.arg_name", !"depth"}
!3 = !{!4}
!4 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
"#;
    let tmp =
        std::env::temp_dir().join(format!("metal2vulkan_native_depth_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpEntryPoint Fragment"), "{asm}");
    assert!(asm.contains("BuiltIn FragDepth"), "{asm}");
    assert!(asm.contains("OpStore"), "{asm}");
    assert!(!asm.contains("Location 0"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_vertex_viewport_array_index_declares_vulkan_capability() {
    let ll = r#"
source_filename = "synth_viewport_index"
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <{ <4 x float>, float, i16, i16, float }> @vmain(i16 %viewport, i16 %layer) local_unnamed_addr #0 {
  %pos0 = insertelement <4 x float> poison, float 0.000000e+00, i64 0
  %pos1 = insertelement <4 x float> %pos0, float 0.000000e+00, i64 1
  %pos2 = insertelement <4 x float> %pos1, float 0.000000e+00, i64 2
  %pos3 = insertelement <4 x float> %pos2, float 1.000000e+00, i64 3
  %out0 = insertvalue <{ <4 x float>, float, i16, i16, float }> undef, <4 x float> %pos3, 0
  %out1 = insertvalue <{ <4 x float>, float, i16, i16, float }> %out0, float 5.000000e-01, 1
  %out2 = insertvalue <{ <4 x float>, float, i16, i16, float }> %out1, i16 %layer, 2
  %out3 = insertvalue <{ <4 x float>, float, i16, i16, float }> %out2, i16 %viewport, 3
  %out4 = insertvalue <{ <4 x float>, float, i16, i16, float }> %out3, float -1.000000e+00, 4
  ret <{ <4 x float>, float, i16, i16, float }> %out4
}

attributes #0 = { nounwind }

!air.vertex = !{!0}
!0 = !{ptr @vmain, !1, !7}
!1 = !{!2, !3, !4, !5, !6}
!2 = !{!"air.position", !"air.arg_type_name", !"float4"}
!3 = !{!"air.vertex_output", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"varying"}
!4 = !{!"air.render_target_array_index", !"air.arg_type_name", !"ushort", !"air.arg_name", !"layer"}
!5 = !{!"air.viewport_array_index", !"air.arg_type_name", !"ushort", !"air.arg_name", !"viewport"}
!6 = !{!"air.clip_distance", !"air.arg_type_name", !"float", !"air.arg_name", !"clip"}
!7 = !{!8, !9}
!8 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!9 = !{i32 1, !"air.instance_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"iid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_viewport_index_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Vertex, &tmp).expect("translate");
    let module = load_bytes(&out).expect("parse transformed");
    let defs: HashMap<Word, &Instruction> = module
        .types_global_values
        .iter()
        .map(|inst| (inst.result_id.unwrap_or(0), inst))
        .collect();
    assert!(module.capabilities.iter().any(|inst| matches!(
        inst.operands.as_slice(),
        [Operand::Capability(Capability::ShaderViewportIndex)]
    )));
    assert!(module.capabilities.iter().any(|inst| matches!(
        inst.operands.as_slice(),
        [Operand::Capability(Capability::ShaderLayer)]
    )));
    assert!(module.capabilities.iter().any(|inst| matches!(
        inst.operands.as_slice(),
        [Operand::Capability(Capability::ClipDistance)]
    )));
    assert_eq!(
        module.header.as_ref().map(|header| header.version()),
        Some((1, 5))
    );
    for builtin in [BuiltIn::Layer, BuiltIn::ViewportIndex] {
        let var = module
            .annotations
            .iter()
            .find_map(|inst| match inst.operands.as_slice() {
                [
                    Operand::IdRef(var),
                    Operand::Decoration(Decoration::BuiltIn),
                    Operand::BuiltIn(found),
                ] if *found == builtin => Some(*var),
                _ => None,
            })
            .unwrap_or_else(|| panic!("{builtin:?} output missing"));
        let var_ty = defs
            .get(&var)
            .and_then(|inst| inst.result_type)
            .unwrap_or_else(|| panic!("{builtin:?} variable has no type"));
        let pointee = defs
            .get(&var_ty)
            .and_then(|inst| match inst.operands.get(1) {
                Some(Operand::IdRef(pointee)) => Some(*pointee),
                _ => None,
            })
            .unwrap_or_else(|| panic!("{builtin:?} variable type is not a pointer"));
        assert!(
            defs.get(&pointee).is_some_and(|inst| {
                inst.class.opcode == Op::TypeInt
                    && inst.operands.first() == Some(&Operand::LiteralBit32(32))
            }),
            "{builtin:?} output pointee is not 32-bit int"
        );
    }
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("BuiltIn ViewportIndex"), "{asm}");
    assert!(asm.contains("BuiltIn Layer"), "{asm}");
    assert!(asm.contains("BuiltIn ClipDistance"), "{asm}");
    assert!(asm.contains("OpAccessChain"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fragment_viewport_array_index_uses_builtin_input() {
    let ll = r#"
source_filename = "synth_fragment_viewport_index"
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define i32 @frag(i32 %viewport) local_unnamed_addr #0 {
  ret i32 %viewport
}

attributes #0 = { nounwind }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"uint", !"air.arg_name", !"color"}
!3 = !{!4}
!4 = !{i32 0, !"air.viewport_array_index", !"air.arg_type_name", !"uint", !"air.arg_name", !"viewport"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fragment_viewport_index_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let module = load_bytes(&out).expect("parse transformed");
    let viewport_var = module
        .annotations
        .iter()
        .find_map(|inst| match inst.operands.as_slice() {
            [
                Operand::IdRef(var),
                Operand::Decoration(Decoration::BuiltIn),
                Operand::BuiltIn(BuiltIn::ViewportIndex),
            ] => Some(*var),
            _ => None,
        })
        .expect("ViewportIndex builtin decoration");
    assert!(module.capabilities.iter().any(|inst| matches!(
        inst.operands.as_slice(),
        [Operand::Capability(Capability::ShaderViewportIndex)]
    )));
    assert!(!module.annotations.iter().any(|inst| {
        inst.class.opcode == Op::Decorate
            && inst.operands.first() == Some(&Operand::IdRef(viewport_var))
            && inst.operands.get(1) == Some(&Operand::Decoration(Decoration::Location))
    }));
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("BuiltIn ViewportIndex"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_vertex_clip_distance_array_uses_array_builtin_output() {
    let ll = r#"
source_filename = "synth_clip_distance_array"
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <{ <4 x float>, [4 x float] }> @vmain() local_unnamed_addr #0 {
  %pos0 = insertelement <4 x float> poison, float 0.000000e+00, i64 0
  %pos1 = insertelement <4 x float> %pos0, float 0.000000e+00, i64 1
  %pos2 = insertelement <4 x float> %pos1, float 0.000000e+00, i64 2
  %pos3 = insertelement <4 x float> %pos2, float 1.000000e+00, i64 3
  %clip0 = insertvalue [4 x float] undef, float 1.000000e+00, 0
  %clip1 = insertvalue [4 x float] %clip0, float 2.000000e+00, 1
  %clip2 = insertvalue [4 x float] %clip1, float 3.000000e+00, 2
  %clip3 = insertvalue [4 x float] %clip2, float 4.000000e+00, 3
  %out0 = insertvalue <{ <4 x float>, [4 x float] }> undef, <4 x float> %pos3, 0
  %out1 = insertvalue <{ <4 x float>, [4 x float] }> %out0, [4 x float] %clip3, 1
  ret <{ <4 x float>, [4 x float] }> %out1
}

attributes #0 = { nounwind }

!air.vertex = !{!0}
!0 = !{ptr @vmain, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.position", !"air.arg_type_name", !"float4"}
!3 = !{!"air.clip_distance", !"air.clip_distance_array_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"clip"}
!4 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_clip_distance_array_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Vertex, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpCapability ClipDistance"), "{asm}");
    assert!(asm.contains("BuiltIn ClipDistance"), "{asm}");
    assert!(asm.contains("OpTypeArray"), "{asm}");
    assert!(asm.contains("OpStore"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fragment_undef_color_depth_return_skips_color_store() {
    let ll = r#"
source_filename = "synth_undef_color_depth"
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <{ <4 x float>, float }> @frag(<4 x float> %position) local_unnamed_addr #0 {
  %depth = extractelement <4 x float> %position, i64 2
  %out = insertvalue <{ <4 x float>, float }> undef, float %depth, 1
  ret <{ <4 x float>, float }> %out
}

attributes #0 = { nounwind }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
!3 = !{!"air.depth", !"air.depth_qualifier", !"air.any", !"air.arg_type_name", !"float", !"air.arg_name", !"depth"}
!4 = !{!5}
!5 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_fragment_undef_color_depth_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpEntryPoint Fragment"), "{asm}");
    assert!(asm.contains("BuiltIn FragDepth"), "{asm}");
    assert!(!asm.contains("Location 0"), "{asm}");
    let depth_var = asm
        .lines()
        .find_map(|line| {
            (line.contains("OpDecorate") && line.contains("BuiltIn FragDepth"))
                .then(|| line.split_whitespace().nth(1))
                .flatten()
        })
        .expect("depth output var");
    assert!(
        asm.lines()
            .any(|line| line.contains(&format!("OpStore {depth_var} "))),
        "{asm}"
    );
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fragment_multiple_return_values_all_store_outputs() {
    let ll = r#"
source_filename = "synth_multi_return_fragment"
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @frag(float %x) local_unnamed_addr #0 {
entry:
  %cond = fcmp olt float %x, 0.000000e+00
  br i1 %cond, label %early, label %late

early:
  ret <4 x float> zeroinitializer

late:
  %v0 = insertelement <4 x float> poison, float 1.000000e+00, i64 0
  %v1 = insertelement <4 x float> %v0, float 2.000000e+00, i64 1
  %v2 = insertelement <4 x float> %v1, float 3.000000e+00, i64 2
  %v3 = insertelement <4 x float> %v2, float 4.000000e+00, i64 3
  ret <4 x float> %v3
}

attributes #0 = { nounwind }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fragment_multi_return_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpEntryPoint Fragment"), "{asm}");
    assert!(!asm.contains("OpReturnValue"), "{asm}");
    assert!(asm.matches("OpStore").count() >= 2, "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fragment_point_coord_maps_to_pointcoord_builtin() {
    let ll = r#"
source_filename = "synth_point_coord"
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <4 x float> @frag(<2 x float> %pointCoord) local_unnamed_addr #0 {
entry:
  %x = extractelement <2 x float> %pointCoord, i64 0
  %y = extractelement <2 x float> %pointCoord, i64 1
  %v0 = insertelement <4 x float> poison, float %x, i64 0
  %v1 = insertelement <4 x float> %v0, float %y, i64 1
  %v2 = insertelement <4 x float> %v1, float 0.000000e+00, i64 2
  %v3 = insertelement <4 x float> %v2, float 1.000000e+00, i64 3
  ret <4 x float> %v3
}

attributes #0 = { nounwind }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.point_coord", !"air.arg_type_name", !"float2", !"air.arg_name", !"pointCoord"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_fragment_point_coord_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("BuiltIn PointCoord"), "{asm}");
    assert!(asm.contains("OpEntryPoint Fragment"), "{asm}");
    assert!(asm.contains("OpLoad"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fragment_primitive_id_maps_to_primitiveid_builtin() {
    let ll = r#"
source_filename = "synth_primitive_id"
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <{ i32, float }> @frag(<4 x float> %position, i32 %primitiveId) local_unnamed_addr #0 {
entry:
  %next = add i32 %primitiveId, 1
  %z = extractelement <4 x float> %position, i64 2
  %out0 = insertvalue <{ i32, float }> undef, i32 %next, 0
  %out1 = insertvalue <{ i32, float }> %out0, float %z, 1
  ret <{ i32, float }> %out1
}

attributes #0 = { nounwind }

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"uint"}
!3 = !{!"air.render_target", i32 1, i32 0, !"air.arg_type_name", !"float"}
!4 = !{!5, !6}
!5 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!6 = !{i32 1, !"air.primitive_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"primitiveId"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_fragment_primitive_id_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("BuiltIn PrimitiveId"), "{asm}");
    assert!(asm.contains("OpEntryPoint Fragment"), "{asm}");
    assert!(asm.contains("OpLoad"), "{asm}");
    assert!(
        !asm.lines()
            .any(|line| line.contains("OpUndef") && line.contains("%1")),
        "{asm}"
    );
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fragment_color_input_maps_to_input_attachment() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define <4 x float> @frag(<4 x float> %color0) {
entry:
  ret <4 x float> %color0
}

!air.fragment = !{!0}
!air.compile_options = !{!5}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"color0"}
!5 = !{!"air.compile.framebuffer_fetch_enable"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_fragment_color_input_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability InputAttachment"), "{asm}");
    assert!(asm.contains("SubpassData"), "{asm}");
    assert!(asm.contains("InputAttachmentIndex 0"), "{asm}");
    assert!(asm.contains("Binding 96"), "{asm}");
    assert!(asm.contains("OpImageRead"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fragment_color2_input_maps_to_input_attachment2() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define float @frag(float %color2) {
entry:
  ret float %color2
}

!air.fragment = !{!0}
!air.compile_options = !{!5}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float"}
!3 = !{!4}
!4 = !{i32 0, !"air.render_target", i32 2, !"air.arg_type_name", !"float", !"air.arg_name", !"color2"}
!5 = !{!"air.compile.framebuffer_fetch_enable"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_fragment_color2_input_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("InputAttachmentIndex 2"), "{asm}");
    assert!(asm.contains("Binding 98"), "{asm}");
}

#[test]
fn native_fragment_scalar_color_input_reads_vec4_then_extracts() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define float @frag(float %color0) {
entry:
  ret float %color0
}

!air.fragment = !{!0}
!air.compile_options = !{!5}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float"}
!3 = !{!4}
!4 = !{i32 0, !"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float", !"air.arg_name", !"color0"}
!5 = !{!"air.compile.framebuffer_fetch_enable"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_fragment_scalar_color_input_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let module = load_bytes(&spv).expect("parse transformed");
    let defs: HashMap<Word, &Instruction> = module
        .types_global_values
        .iter()
        .map(|inst| (inst.result_id.unwrap_or(0), inst))
        .collect();
    assert!(
        module.functions.iter().any(|function| {
            function.blocks.iter().any(|block| {
                block.instructions.iter().any(|inst| {
                    inst.class.opcode == Op::ImageRead
                        && inst.result_type.is_some_and(|result_type| {
                            defs.get(&result_type).is_some_and(|ty| {
                                ty.class.opcode == Op::TypeVector
                                    && ty.operands.get(1) == Some(&Operand::LiteralBit32(4))
                            })
                        })
                })
            })
        }),
        "scalar color input was not read as a vec4"
    );
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCompositeExtract"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_denorms_disable_f32_stays_portable_and_compact() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define <4 x float> @frag(<4 x float> %position) {
entry:
  %w = fmul <4 x float> %position, <float 0x36A0000000000000, float 0x36A0000000000000, float 0x36A0000000000000, float 0x36A0000000000000>
  %q = fdiv <4 x float> %w, %w
  ret <4 x float> %q
}

!air.fragment = !{!0}
!air.compile_options = !{!5}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float4"}
!5 = !{!"air.compile.denorms_disable"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_fragment_denorms_disable_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("OpCapability DenormFlushToZero"), "{asm}");
    assert!(!asm.contains("DenormFlushToZero"), "{asm}");
    assert!(!asm.contains("2143289344"), "{asm}");
    assert!(!asm.contains("OpULessThan"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_denorms_disable_f16_stays_portable_and_compact() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %out) {
entry:
  %h = load half, ptr addrspace(1) %out, align 2
  %m = fmul half %h, 0xH0001
  %q = fdiv half %m, %m
  store half %q, ptr addrspace(1) %out, align 2
  ret void
}

!air.kernel = !{!0}
!air.compile_options = !{!4}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"half", !"air.arg_name", !"out"}
!4 = !{!"air.compile.denorms_disable"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_denorms_disable_f16_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("OpCapability DenormFlushToZero"), "{asm}");
    assert!(!asm.contains("DenormFlushToZero"), "{asm}");
    assert!(!asm.contains("32767"), "{asm}");
    assert!(!asm.contains("32256"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_denorms_disable_f32_to_f16_convert_stays_portable_and_compact() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define <4 x half> @frag(<4 x float> %position) {
entry:
  %h = tail call fast <4 x half> @air.convert.f.v4f16.f.v4f32(<4 x float> %position)
  ret <4 x half> %h
}

declare <4 x half> @air.convert.f.v4f16.f.v4f32(<4 x float>)

!air.fragment = !{!0}
!air.compile_options = !{!5}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!3 = !{!4}
!4 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float4"}
!5 = !{!"air.compile.denorms_disable"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_fragment_denorms_disable_f16_clamp_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("OpCapability DenormFlushToZero"), "{asm}");
    assert!(!asm.contains("DenormFlushToZero"), "{asm}");
    assert!(!asm.contains("65504"), "{asm}");
    assert!(!asm.contains("-65504"), "{asm}");
    assert!(!asm.contains("OpFOrdGreaterThan"), "{asm}");
    assert!(!asm.contains("OpFOrdLessThan"), "{asm}");
    assert!(asm.contains("OpFConvert"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_stage_in_lowers_to_indexed_storage_buffer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(i32 %idx, <3 x i32> %point, ptr addrspace(1) %out) {
entry:
  %x = extractelement <3 x i32> %point, i64 0
  %slot = zext i32 %idx to i64
  %ptr = getelementptr i32, ptr addrspace(1) %out, i64 %slot
  store i32 %x, ptr addrspace(1) %ptr, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"idx"}
!4 = !{i32 1, !"air.stage_in", !"air.location_index", i32 6, i32 1, !"air.arg_type_name", !"uint3", !"air.arg_name", !"point"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_stage_in_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("BuiltIn GlobalInvocationId"), "{asm}");
    assert!(asm.contains("Binding 0"), "{asm}");
    assert!(asm.contains("Binding 1"), "{asm}");
    assert!(asm.contains("ArrayStride 16"), "{asm}");
    assert!(asm.contains("OpAccessChain"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_quoted_kernel_entry_lowers_buffers_to_storage_bindings() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @"re::df::pack"(ptr addrspace(1) readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) writeonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(2) readonly align 4 captures(none) dereferenceable(4) "air-buffer-no-alias" %2, i32 %3) {
entry:
  %count = load i32, ptr addrspace(2) %2, align 4
  %active = icmp ugt i32 %count, %3
  br i1 %active, label %body, label %exit

body:
  %slot = zext i32 %3 to i64
  %in = getelementptr inbounds <3 x float>, ptr addrspace(1) %0, i64 %slot
  %v = load <3 x float>, ptr addrspace(1) %in, align 16
  %x = extractelement <3 x float> %v, i64 0
  %out = getelementptr inbounds [3 x float], ptr addrspace(1) %1, i64 %slot, i64 0
  store float %x, ptr addrspace(1) %out, align 4
  br label %exit

exit:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @"re::df::pack", !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = distinct !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float3", !"air.arg_name", !"src"}
!4 = distinct !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 12, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"packed_float3", !"air.arg_name", !"dst"}
!5 = distinct !{i32 2, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"count"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"idx"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_quoted_kernel_buffers_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Binding 0"), "{asm}");
    assert!(asm.contains("Binding 1"), "{asm}");
    assert!(asm.contains("Binding 4"), "{asm}");
    assert!(asm.contains("BuiltIn GlobalInvocationId"), "{asm}");
    assert!(asm.contains("StorageBuffer"), "{asm}");
    assert!(!asm.contains("OpVariable %_ptr_Private__struct"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fragment_bool_input_uses_flat_uint_interface() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define <4 x float> @frag(i1 %flag) {
entry:
  %r = select i1 %flag, float 1.000000e+00, float 0.000000e+00
  %v0 = insertelement <4 x float> undef, float %r, i32 0
  %v1 = insertelement <4 x float> %v0, float 0.000000e+00, i32 1
  %v2 = insertelement <4 x float> %v1, float 0.000000e+00, i32 2
  %v3 = insertelement <4 x float> %v2, float 1.000000e+00, i32 3
  ret <4 x float> %v3
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.fragment_input", !"generated(flag)", !"air.flat", !"air.arg_type_name", !"bool", !"air.arg_name", !"flag"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_fragment_bool_input_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let module = load_bytes(&spv).expect("load spv");
    let input = interface_variable_at_location(&module, StorageClass::Input, 0)
        .expect("location 0 input variable");
    let input_ty = variable_pointee_type(&module, input).expect("input pointee type");
    assert!(
        module.types_global_values.iter().any(|inst| {
            inst.class.opcode == Op::TypeInt
                && inst.result_id == Some(input_ty)
                && inst.operands.first() == Some(&Operand::LiteralBit32(32))
                && inst.operands.get(1) == Some(&Operand::LiteralBit32(0))
        }),
        "{}",
        disassemble(&spv).expect("disassemble")
    );
    assert!(
        module.annotations.iter().any(|inst| {
            inst.class.opcode == Op::Decorate
                && inst.operands == [Operand::IdRef(input), Operand::Decoration(Decoration::Flat)]
        }),
        "{}",
        disassemble(&spv).expect("disassemble")
    );
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpINotEqual"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fragment_air_flat_float_input_is_decorated_flat() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define <4 x float> @frag(<4 x float> %color) {
entry:
  ret <4 x float> %color
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.fragment_input", !"generated(color)", !"air.flat", !"air.arg_type_name", !"float4", !"air.arg_name", !"color"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_fragment_float_flat_input_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let module = load_bytes(&spv).expect("load spv");
    let input = interface_variable_at_location(&module, StorageClass::Input, 0)
        .expect("location 0 input variable");
    assert!(
        module.annotations.iter().any(|inst| {
            inst.class.opcode == Op::Decorate
                && inst.operands == [Operand::IdRef(input), Operand::Decoration(Decoration::Flat)]
        }),
        "{}",
        disassemble(&spv).expect("disassemble")
    );
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fragment_signed_int_input_uses_signed_interface_type() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define <4 x float> @frag(i32 %index) {
entry:
  %as_float = sitofp i32 %index to float
  %v0 = insertelement <4 x float> undef, float %as_float, i32 0
  %v1 = insertelement <4 x float> %v0, float 0.000000e+00, i32 1
  %v2 = insertelement <4 x float> %v1, float 0.000000e+00, i32 2
  %v3 = insertelement <4 x float> %v2, float 1.000000e+00, i32 3
  ret <4 x float> %v3
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.fragment_input", !"generated(index)", !"air.flat", !"air.arg_type_name", !"int", !"air.arg_name", !"index"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_fragment_signed_int_input_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let module = load_bytes(&spv).expect("load spv");
    let input = interface_variable_at_location(&module, StorageClass::Input, 0)
        .expect("location 0 input variable");
    let input_ty = variable_pointee_type(&module, input).expect("input pointee type");
    assert!(
        module.types_global_values.iter().any(|inst| {
            inst.class.opcode == Op::TypeInt
                && inst.result_id == Some(input_ty)
                && inst.operands.first() == Some(&Operand::LiteralBit32(32))
                && inst.operands.get(1) == Some(&Operand::LiteralBit32(1))
        }),
        "{}",
        disassemble(&spv).expect("disassemble")
    );
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpBitcast"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fragment_signed_render_target_uses_signed_output_type() {
    let ll = r#"
source_filename = "case.metal"

define <4 x i32> @solid_rgba8_sint() {
  ret <4 x i32> <i32 -128, i32 -64, i32 63, i32 127>
}

!air.fragment = !{!0}
!0 = !{ptr @solid_rgba8_sint, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"int4"}
!3 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_fragment_signed_output_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let module = load_bytes(&spv).expect("load spv");
    let output = interface_variable_at_location(&module, StorageClass::Output, 0)
        .expect("location 0 output variable");
    let output_ty = variable_pointee_type(&module, output).expect("output pointee type");
    assert!(
        is_signed_i32_vector(&module, output_ty, 4),
        "{}",
        disassemble(&spv).expect("disassemble")
    );
}

#[test]
fn native_fragment_ushort_render_target_uses_uint_output_type() {
    let ll = r#"
source_filename = "case.metal"

define i16 @solid_r16_uint() {
  ret i16 65535
}

!air.fragment = !{!0}
!0 = !{ptr @solid_r16_uint, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"ushort"}
!3 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_fragment_ushort_output_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let module = load_bytes(&spv).expect("load spv");
    let output = interface_variable_at_location(&module, StorageClass::Output, 0)
        .expect("location 0 output variable");
    let output_ty = variable_pointee_type(&module, output).expect("output pointee type");
    assert!(
        module.types_global_values.iter().any(|inst| {
            inst.class.opcode == Op::TypeInt
                && inst.result_id == Some(output_ty)
                && inst.operands.first() == Some(&Operand::LiteralBit32(32))
                && inst.operands.get(1) == Some(&Operand::LiteralBit32(0))
        }),
        "{}",
        disassemble(&spv).expect("disassemble")
    );
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpUConvert"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_static_initializer_runs_before_entry_body() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@flag = internal addrspace(2) global i8 0, align 1
@sink = internal addrspace(2) global i8 0, align 1

define internal void @_GLOBAL__sub_I_test.metal() section "air.static_init" {
entry:
  store i8 1, ptr addrspace(2) @flag
  ret void
}

define void @main() {
entry:
  %v = load i8, ptr addrspace(2) @flag
  store i8 %v, ptr addrspace(2) @sink
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_static_init_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let function_ids = module
        .debug_names
        .iter()
        .filter_map(|instruction| match instruction.operands.as_slice() {
            [Operand::IdRef(id), Operand::LiteralString(name)] => Some((name.as_str(), *id)),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    let main_id = function_ids["main"];
    let init_id = function_ids["_GLOBAL__sub_I_test.metal"];
    let main = module
        .functions
        .iter()
        .find(|function| function.def.as_ref().and_then(|def| def.result_id) == Some(main_id))
        .expect("emitted main function");
    let first_executable = main.blocks[0]
        .instructions
        .iter()
        .find(|instruction| instruction.class.opcode != Op::Variable)
        .expect("main executable instruction");
    assert_eq!(
        first_executable.class.opcode,
        Op::Store,
        "call-free typed initializer body must precede entry work"
    );
    assert!(
        main.blocks[0].instructions.iter().all(|instruction| {
            instruction.class.opcode != Op::FunctionCall
                || instruction.operands.first() != Some(&Operand::IdRef(init_id))
        }),
        "call-free typed initializer must not cross the seam as a helper call"
    );
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    let init_store = asm
        .find("OpStore")
        .unwrap_or_else(|| panic!("missing inlined static-init store in {asm}"));
    let entry_load = asm
        .find("OpLoad")
        .unwrap_or_else(|| panic!("missing entry load in {asm}"));
    assert!(init_store < entry_load, "{asm}");
    assert!(!asm.contains("_GLOBAL__sub_I"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_chained_multiblock_initializers_complete_in_source_order() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@sink = internal addrspace(2) global i32 0, align 4

define internal void @_GLOBAL__sub_I_first.metal() section "air.static_init" {
entry:
  %condition = icmp eq i32 0, 0
  br i1 %condition, label %then, label %merge
then:
  br label %merge
merge:
  %value = add i32 1, 2
  store i32 %value, ptr addrspace(2) @sink
  ret void
}

define internal void @_GLOBAL__sub_I_second.metal() section "air.static_init" {
entry:
  %condition = icmp eq i32 0, 0
  br i1 %condition, label %then, label %merge
then:
  br label %merge
merge:
  %value = mul i32 3, 4
  store i32 %value, ptr addrspace(2) @sink
  ret void
}

define void @main() {
entry:
  %value = load i32, ptr addrspace(2) @sink
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{}
"#;
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let main_id = module
        .debug_names
        .iter()
        .find_map(|instruction| match instruction.operands.as_slice() {
            [Operand::IdRef(id), Operand::LiteralString(name)] if name == "main" => Some(*id),
            _ => None,
        })
        .expect("main id");
    let main = module
        .functions
        .iter()
        .find(|function| function.def.as_ref().and_then(|def| def.result_id) == Some(main_id))
        .expect("emitted main");
    assert!(
        main.blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .all(|instruction| instruction.class.opcode != Op::FunctionCall),
        "both structurally eligible constructor CFGs must move before serialization"
    );
    let marker_block = |opcode| {
        main.blocks
            .iter()
            .position(|block| {
                block
                    .instructions
                    .iter()
                    .any(|instruction| instruction.class.opcode == opcode)
            })
            .unwrap_or_else(|| panic!("missing {opcode:?} marker in emitted main"))
    };
    let first = marker_block(Op::IAdd);
    let second = marker_block(Op::IMul);
    let entry_work = marker_block(Op::Load);
    let labels = main
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(index, block)| Some((block.label.as_ref()?.result_id?, index)))
        .collect::<HashMap<_, _>>();
    let successors = |block: &Block| {
        let Some(terminator) = block.instructions.last() else {
            return Vec::new();
        };
        match terminator.class.opcode {
            Op::Branch => terminator
                .operands
                .first()
                .and_then(id_ref_operand)
                .into_iter()
                .filter_map(|label| labels.get(&label).copied())
                .collect(),
            Op::BranchConditional => terminator
                .operands
                .iter()
                .skip(1)
                .take(2)
                .filter_map(id_ref_operand)
                .filter_map(|label| labels.get(&label).copied())
                .collect(),
            Op::Switch => terminator
                .operands
                .iter()
                .skip(1)
                .step_by(2)
                .filter_map(id_ref_operand)
                .filter_map(|label| labels.get(&label).copied())
                .collect(),
            _ => Vec::new(),
        }
    };
    let reaches = |from: usize, to: usize| {
        let mut seen = HashSet::new();
        let mut pending = vec![from];
        while let Some(block) = pending.pop() {
            if block == to {
                return true;
            }
            if seen.insert(block) {
                pending.extend(successors(&main.blocks[block]));
            }
        }
        false
    };
    assert!(
        reaches(first, second),
        "first constructor must reach second"
    );
    assert!(
        reaches(second, entry_work),
        "second constructor must reach entry work"
    );
    assert!(
        !reaches(second, first) && !reaches(entry_work, second),
        "constructor ordering must not be cyclic or reversed"
    );
}

#[test]
fn native_kernel_extra_compute_builtins_bind_expected_values() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
	define void @main(i32 %lsize, i32 %lid, i32 %gsize, i32 %gid, i32 %thread_id, i32 %quad_lane, i32 %quad_group, i32 %lane, i32 %simd_group, i32 %simd_width, i32 %num_simd_groups) {
	entry:
	  %a = add i32 %lsize, %lid
	  %b = add i32 %a, %gsize
	  %c = add i32 %b, %gid
	  %d = add i32 %c, %thread_id
	  %q0 = add i32 %d, %quad_lane
	  %q = add i32 %q0, %quad_group
	  %e = add i32 %q, %lane
	  %f = add i32 %e, %simd_group
	  %g0 = add i32 %f, %simd_width
	  %g = add i32 %g0, %num_simd_groups
	  %ok = icmp uge i32 %g, 0
	  ret void
	}

	!air.kernel = !{!0}
	!0 = !{ptr @main, !1, !2}
	!1 = !{}
	!2 = !{!3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13}
	!3 = !{i32 0, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lsize"}
	!4 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lid"}
	!5 = !{i32 2, !"air.threadgroups_per_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gsize"}
	!6 = !{i32 3, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
	!7 = !{i32 4, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"thread_id"}
	!8 = !{i32 5, !"air.thread_index_in_quadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"quad_lane"}
	!9 = !{i32 6, !"air.quadgroup_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"quad_group"}
	!10 = !{i32 7, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lane"}
	!11 = !{i32 8, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_group"}
	!12 = !{i32 9, !"air.threads_per_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_width"}
	!13 = !{i32 10, !"air.simdgroups_per_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"num_simd_groups"}
	"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_extra_compute_builtins_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(
        module,
        Stage::Kernel,
        None,
        None,
        meta::parse_air_kernel_meta(ll).as_ref(),
        meta::entry_name(ll, "kernel").as_deref(),
    )
    .expect("interface transform")
    .assemble()
    .iter()
    .flat_map(|w| w.to_le_bytes())
    .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("BuiltIn LocalInvocationId"), "{asm}");
    assert!(asm.contains("BuiltIn NumWorkgroups"), "{asm}");
    assert!(asm.contains("BuiltIn WorkgroupId"), "{asm}");
    assert!(asm.contains("BuiltIn LocalInvocationIndex"), "{asm}");
    assert_eq!(asm.matches("OpBitwiseAnd").count(), 2, "{asm}");
    assert_eq!(asm.matches("OpShiftRightLogical").count(), 2, "{asm}");
    assert!(!asm.contains("OpUndef"), "{asm}");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.contains("64")),
        "{asm}"
    );
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.contains("32")),
        "{asm}"
    );
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.ends_with(" 2")),
        "{asm}"
    );
    assert_eq!(
        asm.lines()
            .filter(|line| line.contains("OpCompositeExtract"))
            .count(),
        3,
        "{asm}"
    );
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_map_screen_to_physical_1x1_map_lowers_to_zero() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(2) %map_data, ptr addrspace(1) %out, <2 x i32> %tid) {
entry:
  %coord = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %tid)
  %mapped = tail call fast <2 x float> @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x float> %coord, ptr addrspace(2) %map_data, i32 0)
  %slot = getelementptr inbounds [1 x <2 x float>], ptr addrspace(1) %out, i64 0, i64 0
  store <2 x float> %mapped, ptr addrspace(1) %slot, align 8
  ret void
}

declare <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32>)
declare <2 x float> @air.map_screen_to_physical_coordinates.v2f32.p2i8.i32(<2 x float>, ptr addrspace(2), i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"metal::rasterization_rate_map_data", !"air.arg_name", !"map_data"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_map_screen_to_physical_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCompositeConstruct"), "{asm}");
    assert!(!asm.contains("OpCopyObject"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("map_screen_to_physical"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fragment_array_size_query_extracts_array_layer_count() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define <4 x float> @frag(<4 x float> %position, ptr addrspace(1) %tex) {
entry:
  %layers = tail call i32 @air.get_array_size_texture_2d_array(ptr addrspace(1) %tex)
  %f = tail call fast float @air.convert.f.f32.u.i32(i32 %layers)
  %v0 = insertelement <4 x float> undef, float %f, i64 0
  %v1 = insertelement <4 x float> %v0, float 0.000000e+00, i64 1
  %v2 = insertelement <4 x float> %v1, float 0.000000e+00, i64 2
  %v3 = insertelement <4 x float> %v2, float 1.000000e+00, i64 3
  ret <4 x float> %v3
}

declare i32 @air.get_array_size_texture_2d_array(ptr addrspace(1))
declare float @air.convert.f.f32.u.i32(i32)

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5}
!4 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d_array<float, sample>", !"air.arg_name", !"tex"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_fragment_array_size_query_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageQuerySizeLod"), "{asm}");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpCompositeExtract") && line.ends_with(" 2")),
        "{asm}"
    );
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_uint2_compute_builtins_validate() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(<2 x i32> %tid, <2 x i32> %threads) {
entry:
  %x = extractelement <2 x i32> %tid, i64 0
  %y = extractelement <2 x i32> %tid, i64 1
  %sx = extractelement <2 x i32> %threads, i64 0
  %sy = extractelement <2 x i32> %threads, i64 1
  %row = mul i32 %sy, %y
  %col = add i32 %sx, %x
  %sum = add i32 %row, %col
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint2", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint2", !"air.arg_name", !"threads"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_uint2_builtins_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("BuiltIn LocalInvocationId"), "{asm}");
    assert!(asm.contains("OpCompositeConstruct"), "{asm}");
    assert!(
        !asm.lines().any(|line| {
            line.contains("OpCompositeExtract")
                && (line.contains("%uint_64 0") || line.contains("%uint_64 1"))
        }),
        "{asm}"
    );
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}
