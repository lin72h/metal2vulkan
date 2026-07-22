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
use spirv::{Decoration, Op, Scope, SelectionControl, StorageClass, Word};
use std::collections::{HashMap, HashSet};

#[test]
fn native_array_ref_texture_lowers_to_descriptor_array() {
    // A runtime-indexed `array_ref<texture2d>` argument is a descriptor array, not a single image.
    // The backend emits real per-element handle loads (`load ptr addrspace(1), ptr %texarg`) that a
    // single-image binding turns into an illegal `OpLoad` of a pointer FROM an image value
    // ("not a logical pointer"). The interface `ImageArray` binding + `materialize_texture_array_loads`
    // declare `OpTypeArray %image N` in UniformConstant and rewrite each element handle load to
    // `OpAccessChain %arrayvar %idx` + `OpLoad %image`, so the size-query/sample lowering sees an
    // ordinary loaded image. This is the shape that clears the texture-array shapes.
    let ll = r#"
%"struct.metal::texture2d" = type { ptr addrspace(1) }

define void @widthTexArray(ptr readonly captures(none) %0, ptr addrspace(1) noundef writeonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load ptr addrspace(1), ptr %0, align 8
  %4 = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) readonly captures(none) %3, i32 0) #1
  store i32 %4, ptr addrspace(1) %1, align 4
  ret void
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1) readonly captures(none), i32) local_unnamed_addr #1

attributes #0 = { convergent nounwind }
attributes #1 = { convergent nounwind memory(none) }

!air.kernel = !{!0}
!0 = !{ptr @widthTexArray, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"array_ref<texture2d<float, sample>>", !"air.arg_name", !"imgs"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_array_ref_texture_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    // Declared as an image descriptor array, accessed by index — not a single-image OpLoad of the var.
    assert!(
        asm.contains("OpTypeArray"),
        "expected image array type:\n{asm}"
    );
    assert!(
        asm.contains("OpAccessChain") && asm.contains("OpImageQuerySizeLod"),
        "expected indexed image access + size query:\n{asm}"
    );
    assert!(
        !asm.contains("is not a logical pointer"),
        "must not emit an illegal pointer load:\n{asm}"
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
fn native_get_null_texture_models_unmodeled_placeholder() {
    // `air.get_null_texture_2d()` yields a NULL texture handle that the MPS NDArray kernels store
    // into a vestigial struct field but never sample. The native emitter previously bailed
    // (`missing pointer storage`) because the result pointer had no storage class. It is now modeled
    // as an unmodeled placeholder pointer, so the store validates; the kernel emits a valid module.
    let ll = r#"
define void @k(ptr addrspace(1) %out) {
entry:
  %slot = alloca ptr addrspace(1), align 8
  %tex = call ptr addrspace(1) @air.get_null_texture_2d()
  store ptr addrspace(1) %tex, ptr %slot, align 8
  store i32 0, ptr addrspace(1) %out, align 4
  ret void
}

declare ptr addrspace(1) @air.get_null_texture_2d()

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let spv = emit_vulkan_spirv(ll).expect("null-texture kernel must emit a module");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpFunction"), "{asm}");
}

#[test]
fn native_is_null_texture_of_get_null_texture_lowers_to_true() {
    // `air.is_null_texture(%t)` where `%t` came from `air.get_null_texture_2d()` must lower to a
    // constant TRUE. The get_null_texture handle never crosses the emitter->reparse seam as a
    // recognizable image, so the passes-layer null tracking can't see it; the emitter therefore
    // consumes is_null_texture directly, keyed on the value it itself synthesized. Branching on the
    // result keeps it live (a realistic "is this texture bound?" use) and forces it into the module.
    let ll = r#"
define void @k(ptr addrspace(1) %out) {
entry:
  %tex = call ptr addrspace(1) @air.get_null_texture_2d()
  %isnull = call i1 @air.is_null_texture_2d(ptr addrspace(1) %tex)
  br i1 %isnull, label %null, label %bound
null:
  store i32 1, ptr addrspace(1) %out, align 4
  br label %done
bound:
  store i32 0, ptr addrspace(1) %out, align 4
  br label %done
done:
  ret void
}

declare ptr addrspace(1) @air.get_null_texture_2d()
declare i1 @air.is_null_texture_2d(ptr addrspace(1))

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let spv = emit_vulkan_spirv(ll).expect("is-null-texture kernel must emit a module");
    let asm = disassemble(&spv).expect("disassemble");
    // The predicate is the constant TRUE, not FALSE: a synthesized null texture IS null.
    assert!(
        asm.contains("OpConstantTrue"),
        "is_null_texture(get_null_texture()) must lower to constant TRUE; got:\n{asm}"
    );
}

#[test]
fn native_kernel_texture_sampler_metadata_bindings() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %tex, ptr addrspace(2) %sampler) {
entry:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 5, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.sampler", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"sampler"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_texture_sampler_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpTypeImage") && line.contains(" 2D ")),
        "{asm}"
    );
    assert!(asm.contains("OpTypeSampler"), "{asm}");
    assert!(asm.contains("Binding 37"), "{asm}");
    assert!(asm.contains("Binding 66"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_texture_sample_without_lod_uses_explicit_lod_zero() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %tex, ptr addrspace(2) %sampler, ptr addrspace(1) %out) {
entry:
  %c0 = insertelement <2 x float> poison, float 2.500000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 7.500000e-01, i32 1
  %sample = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %sampler, <2 x float> %coord)
  %color = extractvalue { <4 x float>, i8 } %sample, 0
  %x = extractelement <4 x float> %color, i64 0
  store float %x, ptr addrspace(1) %out, align 4
  ret void
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"sampler"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_sample_explicit_lod0_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageSampleExplicitLod"), "{asm}");
    assert!(!asm.contains("OpImageSampleImplicitLod"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_texture_sample_const_offset_uses_image_operand() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %tex, ptr addrspace(2) %sampler, ptr addrspace(1) %out) {
entry:
  %c0 = insertelement <2 x float> poison, float 2.500000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 7.500000e-01, i32 1
  %sample = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %sampler, <2 x float> %coord, i1 true, <2 x i32> <i32 1, i32 -1>, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %color = extractvalue { <4 x float>, i8 } %sample, 0
  store <4 x float> %color, ptr addrspace(1) %out, align 16
  ret void
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"sampler"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_sample_const_offset_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let sample = asm
        .lines()
        .find(|line| line.contains("OpImageSampleExplicitLod"))
        .expect("find OpImageSampleExplicitLod");
    assert!(sample.contains("Lod"), "{sample}\n\n{asm}");
    assert!(sample.contains("ConstOffset"), "{sample}\n\n{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_texture_sample_dynamic_offset_adjusts_coordinate() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %tex, ptr addrspace(2) %sampler, ptr addrspace(1) %out, i32 %lane) {
entry:
  %c0 = insertelement <2 x float> poison, float 2.500000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 7.500000e-01, i32 1
  %x = or i32 %lane, -8
  %o0 = insertelement <2 x i32> undef, i32 %x, i32 0
  %offset = insertelement <2 x i32> %o0, i32 -5, i32 1
  %sample = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %sampler, <2 x float> %coord, i1 true, <2 x i32> %offset, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %color = extractvalue { <4 x float>, i8 } %sample, 0
  store <4 x float> %color, ptr addrspace(1) %out, align 16
  ret void
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"sampler"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4*", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"int", !"air.arg_name", !"lane"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_sample_dynamic_offset_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let sample = asm
        .lines()
        .find(|line| line.contains("OpImageSampleExplicitLod"))
        .expect("find OpImageSampleExplicitLod");
    assert!(sample.contains("Lod"), "{sample}\n\n{asm}");
    assert!(!sample.contains("ConstOffset"), "{sample}\n\n{asm}");
    assert!(asm.contains("OpImageQuerySizeLod"), "{asm}");
    assert!(asm.contains("OpFDiv"), "{asm}");
    assert!(asm.contains("OpFAdd"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_depth_sample_uses_explicit_lod_zero() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %depth, ptr addrspace(2) %sampler, ptr addrspace(1) %out) {
entry:
  %c0 = insertelement <2 x float> poison, float 2.500000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 7.500000e-01, i32 1
  %sample = tail call { float, i8 } @air.sample_depth_2d.f32(ptr addrspace(1) %depth, ptr addrspace(2) %sampler, i32 0, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value = extractvalue { float, i8 } %sample, 0
  store float %value, ptr addrspace(1) %out, align 4
  ret void
}

declare { float, i8 } @air.sample_depth_2d.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d<float, sample>", !"air.arg_name", !"depth"}
!4 = !{i32 1, !"air.sampler", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"sampler"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_depth_sample_explicit_lod0_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageSampleExplicitLod"), "{asm}");
    assert!(!asm.contains("OpImageSampleImplicitLod"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fragment_selected_static_sampler_uses_valid_sampler_operand() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant i64 -9188470239253757879, align 8
@__air_sampler_state.1 = internal addrspace(2) constant i64 -9188470239253755831, align 8

define <4 x float> @frag(<4 x float> %position, <2 x float> %coord, ptr addrspace(1) %tex) {
entry:
  %selected = select i1 true, ptr addrspace(2) @__air_sampler_state, ptr addrspace(2) @__air_sampler_state.1
  %sample = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) %selected, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %color = extractvalue { <4 x float>, i8 } %sample, 0
  ret <4 x float> %color
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.fragment = !{!0}
!air.sampler_states = !{!8, !9}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5, !6}
!4 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!5 = !{i32 1, !"air.fragment_input", !"generated(coord)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"coord"}
!6 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!8 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!9 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.1}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_selected_static_sampler_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpTypeSampler"), "{asm}");
    assert!(asm.contains("OpSampledImage"), "{asm}");
    for line in asm.lines().filter(|line| line.contains("OpSampledImage")) {
        assert!(!line.contains("Private"), "{line}");
    }
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gather_texture_accepts_null_offset_vector() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020416, i64 0], align 8

define void @k(ptr addrspace(1) %tex, ptr addrspace(1) %out) {
entry:
  %c0 = insertelement <2 x float> poison, float 5.000000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 5.000000e-01, i32 1
  %gather = tail call { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)
  %color = extractvalue { <4 x float>, i8 } %gather, 0
  store <4 x float> %color, ptr addrspace(1) %out, align 16
  ret void
}

declare { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4*", !"air.arg_name", !"out"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_gather_null_offset_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageGather"), "{asm}");
    for line in asm.lines().filter(|line| line.contains("OpImageGather")) {
        assert!(!line.contains("ConstOffset"), "{line}");
    }
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gather_texture_uint_result_uses_integer_image_type() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020416, i64 0], align 8

define void @k(ptr addrspace(1) %tex, ptr addrspace(1) %out) {
entry:
  %c0 = insertelement <2 x float> poison, float 5.000000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 5.000000e-01, i32 1
  %gather = tail call { <4 x i32>, i8 } @air.gather_texture_2d.u.v4i32(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)
  %color = extractvalue { <4 x i32>, i8 } %gather, 0
  store <4 x i32> %color, ptr addrspace(1) %out, align 16
  ret void
}

declare { <4 x i32>, i8 } @air.gather_texture_2d.u.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<uint, sample>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"int4*", !"air.arg_name", !"out"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_gather_uint_texture_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageGather"), "{asm}");
    assert!(asm.contains("OpTypeImage"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gather_texture_from_byval_array_dynamic_field_uses_single_sampled_binding() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

%"struct.metal::texture2d" = type { ptr addrspace(1) }

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020416, i64 0], align 8

define void @k(i32 %idx, ptr readonly byval([2 x %"struct.metal::texture2d"]) %textures, ptr addrspace(1) %out) {
entry:
  %base = bitcast ptr %textures to ptr
  %wide = zext i32 %idx to i64
  %slot = getelementptr inbounds [2 x %"struct.metal::texture2d"], ptr %base, i64 0, i64 %wide, i32 0
  %tex = load ptr addrspace(1), ptr %slot, align 8
  %c0 = insertelement <2 x float> poison, float 5.000000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 5.000000e-01, i32 1
  %gather = tail call { <4 x i32>, i8 } @air.gather_texture_2d.u.v4i32(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)
  %color = extractvalue { <4 x i32>, i8 } %gather, 0
  tail call void @air.write_texture_2d.u.v4i32(ptr addrspace(1) %out, <2 x i32> zeroinitializer, <4 x i32> %color, i32 0, i32 2)
  ret void
}

declare { <4 x i32>, i8 } @air.gather_texture_2d.u.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)
declare void @air.write_texture_2d.u.v4i32(ptr addrspace(1), <2 x i32>, <4 x i32>, i32, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"idx"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 2, !"air.sample", !"air.arg_type_name", !"array<texture2d<uint, sample>, 2>", !"air.arg_name", !"textures"}
!5 = !{i32 2, !"air.texture", !"air.location_index", i32 4, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<uint, write>", !"air.arg_name", !"out"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_gather_byval_texture_array_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageGather"), "{asm}");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gather_texture_uint_pixel_sampler_extracts_integer_components() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601050697, i64 0], align 8

define void @k(ptr addrspace(1) %tex, ptr addrspace(1) %out) {
entry:
  %c0 = insertelement <2 x float> poison, float 5.000000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 5.000000e-01, i32 1
  %gather = tail call { <4 x i32>, i8 } @air.gather_texture_2d.u.v4i32(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)
  %color = extractvalue { <4 x i32>, i8 } %gather, 0
  store <4 x i32> %color, ptr addrspace(1) %out, align 16
  ret void
}

declare { <4 x i32>, i8 } @air.gather_texture_2d.u.v4i32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<uint, sample>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"int4*", !"air.arg_name", !"out"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_gather_uint_pixel_texture_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(asm.matches("OpImageFetch").count(), 4, "{asm}");
    for line in asm
        .lines()
        .filter(|line| line.contains("OpCompositeExtract"))
    {
        assert!(!line.contains("%float"), "{line}\n{asm}");
    }
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gather_texture_half_result_converts_struct_member() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020416, i64 0], align 8

define void @k(ptr addrspace(1) %tex, ptr addrspace(1) %out) {
entry:
  %c0 = insertelement <2 x float> poison, float 5.000000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 5.000000e-01, i32 1
  %gather = tail call { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)
  %color = extractvalue { <4 x half>, i8 } %gather, 0
  store <4 x half> %color, ptr addrspace(1) %out, align 8
  ret void
}

declare { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4*", !"air.arg_name", !"out"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_gather_half_result_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageGather"), "{asm}");
    assert!(asm.contains("OpFConvert"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gather_texture_2d_array_uses_layer_coordinate() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020416, i64 0], align 8

define void @k(ptr addrspace(1) %tex, ptr addrspace(1) %out) {
entry:
  %c0 = insertelement <2 x float> poison, float 5.000000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 5.000000e-01, i32 1
  %gather = tail call { <4 x half>, i8 } @air.gather_texture_2d_array.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i32 1, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)
  %color = extractvalue { <4 x half>, i8 } %gather, 0
  store <4 x half> %color, ptr addrspace(1) %out, align 8
  ret void
}

declare { <4 x half>, i8 } @air.gather_texture_2d_array.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i32, i1, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d_array<half, sample>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4*", !"air.arg_name", !"out"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_gather_2d_array_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load spv");
    assert!(
        module
            .functions
            .iter()
            .flat_map(|func| &func.blocks)
            .flat_map(|block| &block.instructions)
            .any(|inst| inst.class.opcode == Op::ImageGather),
        "{asm}"
    );
    assert!(
        module.types_global_values.iter().any(|inst| {
            inst.class.opcode == Op::TypeImage
                && inst.operands.get(1) == Some(&Operand::Dim(spirv::Dim::Dim2D))
                && inst.operands.get(3) == Some(&Operand::LiteralBit32(1))
        }),
        "{asm}"
    );
    assert!(
        module
            .functions
            .iter()
            .flat_map(|func| &func.blocks)
            .flat_map(|block| &block.instructions)
            .any(|inst| inst.class.opcode == Op::CompositeConstruct
                && inst
                    .result_type
                    .is_some_and(|ty| is_float_vector(&module, ty, 3))),
        "{asm}"
    );
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gather_texture_accepts_splat_offset_vector() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020416, i64 0], align 8

define void @k(ptr addrspace(1) %tex, ptr addrspace(1) %out) {
entry:
  %c0 = insertelement <2 x float> poison, float 5.000000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 5.000000e-01, i32 1
  %gather = tail call { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> splat (i32 -7), i32 0, i32 0)
  %color = extractvalue { <4 x float>, i8 } %gather, 0
  store <4 x float> %color, ptr addrspace(1) %out, align 16
  ret void
}

declare { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4*", !"air.arg_name", !"out"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_gather_splat_offset_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageGather"), "{asm}");
    for line in asm.lines().filter(|line| line.contains("OpImageGather")) {
        assert!(line.contains("ConstOffset"), "{line}\n\n{asm}");
    }
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gather_texture_accepts_literal_offset_vector() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020416, i64 0], align 8

define void @k(ptr addrspace(1) %tex, ptr addrspace(1) %out) {
entry:
  %c0 = insertelement <2 x float> poison, float 5.000000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 5.000000e-01, i32 1
  %gather = tail call { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> <i32 -7, i32 -5>, i32 0, i32 0)
  %color = extractvalue { <4 x float>, i8 } %gather, 0
  store <4 x float> %color, ptr addrspace(1) %out, align 16
  ret void
}

declare { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4*", !"air.arg_name", !"out"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_gather_literal_offset_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageGather"), "{asm}");
    for line in asm.lines().filter(|line| line.contains("OpImageGather")) {
        assert!(line.contains("ConstOffset"), "{line}\n\n{asm}");
    }
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gather_texture_dynamic_offset_adjusts_coordinate() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020416, i64 0], align 8

define void @k(ptr addrspace(1) %tex, ptr addrspace(1) %out, i32 %lane) {
entry:
  %c0 = insertelement <2 x float> poison, float 5.000000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 5.000000e-01, i32 1
  %x = or i32 %lane, -8
  %o0 = insertelement <2 x i32> undef, i32 %x, i32 0
  %offset = insertelement <2 x i32> %o0, i32 -5, i32 1
  %gather = tail call { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> %offset, i32 0, i32 0)
  %color = extractvalue { <4 x float>, i8 } %gather, 0
  store <4 x float> %color, ptr addrspace(1) %out, align 16
  ret void
}

declare { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4*", !"air.arg_name", !"out"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"int", !"air.arg_name", !"lane"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_gather_dynamic_offset_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let gather = asm
        .lines()
        .find(|line| line.contains("OpImageGather"))
        .expect("find OpImageGather");
    assert!(!gather.contains("ConstOffset"), "{gather}\n\n{asm}");
    assert!(asm.contains("OpImageQuerySizeLod"), "{asm}");
    assert!(asm.contains("OpFDiv"), "{asm}");
    assert!(asm.contains("OpFAdd"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gather_texture_pixel_sampler_lowers_to_fetches() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant i64 -9188470239253725111, align 8

define void @k(ptr addrspace(1) %tex, ptr addrspace(1) %out) {
entry:
  %c0 = insertelement <2 x float> poison, float 4.000000e+00, i32 0
  %coord = insertelement <2 x float> %c0, float 6.000000e+00, i32 1
  %gather = tail call { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> <i32 -1, i32 3>, i32 0, i32 0)
  %color = extractvalue { <4 x half>, i8 } %gather, 0
  store <4 x half> %color, ptr addrspace(1) %out, align 8
  ret void
}

declare { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4*", !"air.arg_name", !"out"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_gather_pixel_sampler_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(asm.matches("OpImageFetch").count(), 4, "{asm}");
    // The gather footprint base is floor(coord - 0.5): the coordinate is shifted by half a texel
    // (OpFSub against 0.5) before flooring. For this integer coord (4,6) with const offset (-1,3)
    // the four fetches land at (2,9),(3,9),(3,8),(2,8) — identical to the old whole-texel-bias form,
    // but computed via the half-texel shift, which is also correct for fractional coordinates.
    assert!(asm.contains("OpFSub"), "{asm}");
    assert!(asm.contains("0.5"), "{asm}");
    assert!(asm.contains("OpFConvert"), "{asm}");
    assert!(!asm.contains("OpSampledImage"), "{asm}");
    assert!(!asm.contains("OpImageGather"), "{asm}");
    assert!(!asm.contains("OpImageSample"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gather_texture_pixel_center_sampler_skips_high_bit_bias() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant i64 -9188470239253725111, align 8

define void @k(<2 x i16> %index, ptr addrspace(1) %tex, ptr addrspace(1) %out) {
entry:
  %base = shl <2 x i16> %index, splat (i16 1)
  %basef = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16> %base)
  %coord = fadd fast <2 x float> %basef, splat (float 5.000000e-01)
  %gather = tail call { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i32 2, i32 0)
  %color = extractvalue { <4 x half>, i8 } %gather, 0
  store <4 x half> %color, ptr addrspace(1) %out, align 8
  ret void
}

declare { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)
declare <2 x float> @air.convert.f.v2f32.u.v2i16(<2 x i16>)

!air.kernel = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"index"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"tex"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4*", !"air.arg_name", !"out"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_gather_pixel_center_sampler_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(asm.matches("OpImageFetch").count(), 4, "{asm}");
    assert!(asm.contains("OpFAdd"), "{asm}");
    assert!(!asm.contains(" -1"), "{asm}");
    assert!(!asm.contains("OpSampledImage"), "{asm}");
    assert!(!asm.contains("OpImageGather"), "{asm}");
    assert!(!asm.contains("OpImageSample"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gather_texture_array_pixel_sampler_integer_coord_uses_half_texel_footprint() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601050697, i64 0], align 8

define void @k(<2 x i32> %index, ptr addrspace(1) %tex, ptr addrspace(1) %out) {
entry:
  %base = shl <2 x i32> %index, splat (i32 1)
  %coord = tail call fast <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32> %base)
  %gather = tail call { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)
  %color = extractvalue { <4 x half>, i8 } %gather, 0
  store <4 x half> %color, ptr addrspace(1) %out, align 8
  ret void
}

declare { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)
declare <2 x float> @air.convert.f.v2f32.u.v2i32(<2 x i32>)

!air.kernel = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"index"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"tex"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4*", !"air.arg_name", !"out"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_gather_array_pixel_sampler_integer_coord_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(asm.matches("OpImageFetch").count(), 4, "{asm}");
    // Footprint base is floor(coord - 0.5): the coordinate is shifted by half a texel (OpFSub 0.5)
    // before flooring, so a runtime even-integer coord c gathers the c-1..c row exactly as Apple's
    // hardware gather does — without a static whole-texel bias that is wrong for fractional coords.
    assert!(asm.contains("OpFSub"), "{asm}");
    assert!(asm.contains("0.5"), "{asm}");
    assert!(asm.contains("OpFConvert"), "{asm}");
    assert!(!asm.contains("OpSampledImage"), "{asm}");
    assert!(!asm.contains("OpImageGather"), "{asm}");
    assert!(!asm.contains("OpImageSample"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gather_write_texture_pixel_sampler_uses_storage_reads() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant i64 -9188470239253725111, align 8

define void @k(ptr addrspace(1) %tex, ptr addrspace(1) %out) {
entry:
  %c0 = insertelement <2 x float> poison, float 4.000000e+00, i32 0
  %coord = insertelement <2 x float> %c0, float 6.000000e+00, i32 1
  %gather = tail call { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> <i32 -1, i32 3>, i32 0, i32 0)
  %color = extractvalue { <4 x half>, i8 } %gather, 0
  store <4 x half> %color, ptr addrspace(1) %out, align 8
  ret void
}

declare { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4*", !"air.arg_name", !"out"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_gather_write_texture_pixel_sampler_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(asm.matches("OpImageRead").count(), 4, "{asm}");
    assert!(asm.contains("OpImageQuerySize "), "{asm}");
    assert!(!asm.contains("OpImageFetch"), "{asm}");
    assert!(!asm.contains("OpImageQuerySizeLod"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gather_texture_array_pixel_sampler_lowers_to_fetches() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601050697, i64 0], align 8

define void @k(ptr addrspace(1) %tex, ptr addrspace(1) %out) {
entry:
  %c0 = insertelement <2 x float> poison, float 5.000000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 5.000000e-01, i32 1
  %gather = tail call { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i32 2, i32 0)
  %color = extractvalue { <4 x half>, i8 } %gather, 0
  store <4 x half> %color, ptr addrspace(1) %out, align 8
  ret void
}

declare { <4 x half>, i8 } @air.gather_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4*", !"air.arg_name", !"out"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_gather_array_pixel_sampler_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(asm.matches("OpImageFetch").count(), 4, "{asm}");
    assert!(asm.contains("OpCompositeExtract"), "{asm}");
    assert!(!asm.contains("OpSampledImage"), "{asm}");
    assert!(!asm.contains("OpImageGather"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gather_depth_lowers_to_component_zero_gather() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020416, i64 0], align 8

define void @k(ptr addrspace(1) %depth, ptr addrspace(1) %out) {
entry:
  %c0 = insertelement <2 x float> poison, float 5.000000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 5.000000e-01, i32 1
  %gather = tail call { <4 x float>, i8 } @air.gather_depth_2d.v4f32(ptr addrspace(1) %depth, ptr addrspace(2) @__air_sampler_state, i32 1, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i32 0)
  %values = extractvalue { <4 x float>, i8 } %gather, 0
  store <4 x float> %values, ptr addrspace(1) %out, align 16
  ret void
}

declare { <4 x float>, i8 } @air.gather_depth_2d.v4f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, i1, <2 x i32>, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d<float, sample>", !"air.arg_name", !"depth"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4*", !"air.arg_name", !"out"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp =
        std::env::temp_dir().join(format!("metal2vulkan_gather_depth_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    // A REAL component-0 gather (normalized-coordinate sampler): OpImageGather of the R channel,
    // never the retired zero-null harness contract.
    assert!(asm.contains("OpImageGather"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gather_depth_2d_array_uses_layer_coordinate() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601020416, i64 0], align 8

define void @k(ptr addrspace(1) %depth, ptr addrspace(1) %out) {
entry:
  %c0 = insertelement <2 x float> poison, float 5.000000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 5.000000e-01, i32 1
  %gather = tail call { <4 x float>, i8 } @air.gather_depth_2d_array.v4f32(ptr addrspace(1) %depth, ptr addrspace(2) @__air_sampler_state, i32 1, <2 x float> %coord, i32 2, i1 true, <2 x i32> zeroinitializer, i32 0)
  %values = extractvalue { <4 x float>, i8 } %gather, 0
  store <4 x float> %values, ptr addrspace(1) %out, align 16
  ret void
}

declare { <4 x float>, i8 } @air.gather_depth_2d_array.v4f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x float>, i32, i1, <2 x i32>, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"depth2d_array<float, sample>", !"air.arg_name", !"depth"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4*", !"air.arg_name", !"out"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_gather_depth_2d_array_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load spv");
    assert!(
        module
            .functions
            .iter()
            .flat_map(|func| &func.blocks)
            .flat_map(|block| &block.instructions)
            .filter(|inst| inst.class.opcode == Op::ImageFetch)
            .count()
            == 4,
        "{asm}"
    );
    assert!(asm.contains("OpImageQuerySizeLod"), "{asm}");
    assert!(asm.contains("OpULessThan"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(asm.contains("OpConstantNull"), "{asm}");
    assert!(asm.contains(" Floor "), "{asm}");
    assert!(!asm.contains("OpImageGather"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_unused_texture1d_uses_metadata_dimension() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %lut, ptr addrspace(1) %out) {
entry:
  store i32 1, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture1d<float, sample>", !"air.arg_name", !"lut", !"air.arg_unused"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_unused_texture1d_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpTypeImage"), "{asm}");
    assert!(asm.contains(" 1D "), "{asm}");
    assert!(asm.contains("Binding 32"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_uint_write_texture_uses_uint_storage_image() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %dst, <2 x i32> %tid) {
entry:
  %v0 = insertelement <4 x i32> undef, i32 1, i64 0
  %v1 = insertelement <4 x i32> %v0, i32 2, i64 1
  %v2 = insertelement <4 x i32> %v1, i32 3, i64 2
  %v3 = insertelement <4 x i32> %v2, i32 4, i64 3
  tail call void @air.write_texture_2d.u.v4i32(ptr addrspace(1) %dst, <2 x i32> %tid, <4 x i32> %v3, i32 0, i32 2)
  ret void
}

declare void @air.write_texture_2d.u.v4i32(ptr addrspace(1), <2 x i32>, <4 x i32>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 7, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<uint, write>", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_uint_write_texture_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Rgba8ui"), "{asm}");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(asm.contains("Binding 39"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_helper_wrapped_write_texture_uses_single_storage_image() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Out = type <{ %Tex, <2 x i16> }>
%Tex = type { ptr addrspace(1) }

define void @k(ptr addrspace(1) %dst) {
entry:
  %out = alloca %Out, align 8
  %tex_field = getelementptr inbounds %Out, ptr %out, i64 0, i32 0, i32 0
  store ptr addrspace(1) %dst, ptr %tex_field, align 8
  %coord_field = getelementptr inbounds %Out, ptr %out, i64 0, i32 1
  store <2 x i16> zeroinitializer, ptr %coord_field, align 8
  call void @helper(ptr %out)
  ret void
}

define internal void @helper(ptr %out) {
entry:
  %coord_field = getelementptr inbounds %Out, ptr %out, i64 0, i32 1
  %coord = load <2 x i16>, ptr %coord_field, align 8
  %tex_field = getelementptr inbounds %Out, ptr %out, i64 0, i32 0, i32 0
  %tex = load ptr addrspace(1), ptr %tex_field, align 8
  tail call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %tex, <2 x i16> %coord, <4 x half> zeroinitializer, i16 0, i32 2)
  ret void
}

declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"dst"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_helper_wrapped_write_texture_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_helper_wrapped_texture_query_resolves_loaded_image() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Out = type { %Tex }
%Tex = type { ptr addrspace(1) }

define void @k(ptr addrspace(1) %src, ptr addrspace(1) %out) {
entry:
  %wrap = alloca %Out, align 8
  %tex_field = getelementptr inbounds %Out, ptr %wrap, i64 0, i32 0, i32 0
  store ptr addrspace(1) %src, ptr %tex_field, align 8
  call void @helper(ptr %wrap, ptr addrspace(1) %out)
  ret void
}

define internal void @helper(ptr %wrap, ptr addrspace(1) %out) {
entry:
  %tex_field = getelementptr inbounds %Out, ptr %wrap, i64 0, i32 0, i32 0
  %tex = load ptr addrspace(1), ptr %tex_field, align 8
  %w = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %tex, i32 0)
  store i32 %w, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_helper_wrapped_texture_query_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageQuerySizeLod"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_helper_wrapped_write_texture_query_resolves_loaded_image() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Out = type { %Tex }
%Tex = type { ptr addrspace(1) }

define void @k(ptr addrspace(1) %dst, ptr addrspace(1) %out) {
entry:
  %wrap = alloca %Out, align 8
  %tex_field = getelementptr inbounds %Out, ptr %wrap, i64 0, i32 0, i32 0
  store ptr addrspace(1) %dst, ptr %tex_field, align 8
  call void @helper(ptr %wrap, ptr addrspace(1) %out)
  ret void
}

define internal void @helper(ptr %wrap, ptr addrspace(1) %out) {
entry:
  %tex_field = getelementptr inbounds %Out, ptr %wrap, i64 0, i32 0, i32 0
  %tex = load ptr addrspace(1), ptr %tex_field, align 8
  %w = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %tex, i32 0)
  store i32 %w, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_helper_wrapped_write_texture_query_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageQuerySizeLod"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_helper_wrapped_multi_texture_query_resolves_field_source() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Out = type { %Tex }
%Tex = type { ptr addrspace(1) }

define void @k(ptr addrspace(1) %src0, ptr addrspace(1) %src1, ptr addrspace(1) %dst, ptr addrspace(1) %out) {
entry:
  %wrap = alloca %Out, align 8
  %tex_field = getelementptr inbounds %Out, ptr %wrap, i64 0, i32 0, i32 0
  store ptr addrspace(1) %src0, ptr %tex_field, align 8
  call void @helper(ptr %wrap, ptr addrspace(1) %out)
  ret void
}

define internal void @helper(ptr %wrap, ptr addrspace(1) %out) {
entry:
  %tex_field = getelementptr inbounds %Out, ptr %wrap, i64 0, i32 0, i32 0
  %tex = load ptr addrspace(1), ptr %tex_field, align 8
  %w = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %tex, i32 0)
  store i32 %w, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"src0"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"src1"}
!5 = !{i32 2, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"dst"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_helper_wrapped_multi_texture_query_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageQuerySizeLod"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_selected_texture_query_clones_queries_before_select() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %src0, ptr addrspace(1) %src1, ptr addrspace(1) %out, i32 %tid) {
entry:
  %cond = icmp eq i32 %tid, 0
  %tex = select i1 %cond, ptr addrspace(1) %src0, ptr addrspace(1) %src1
  %w = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %tex, i32 0)
  %h = tail call i32 @air.get_height_texture_2d(ptr addrspace(1) %tex, i32 0)
  store i32 %w, ptr addrspace(1) %out, align 4
  %out1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %h, ptr addrspace(1) %out1, align 4
  ret void
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare i32 @air.get_height_texture_2d(ptr addrspace(1), i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"src0"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"src1"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint2", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_selected_texture_query_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load native spv");
    let pointer_types = module
        .types_global_values
        .iter()
        .filter_map(|inst| {
            (inst.class.opcode == Op::TypePointer)
                .then_some(inst.result_id)
                .flatten()
        })
        .collect::<HashSet<_>>();
    let pointer_select = module.all_inst_iter().find(|inst| {
        inst.class.opcode == Op::Select
            && inst
                .result_type
                .is_some_and(|ty| pointer_types.contains(&ty))
    });
    assert!(pointer_select.is_none(), "{pointer_select:?}\n{asm}");
    assert_eq!(asm.matches("OpImageQuerySizeLod").count(), 4, "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_imageblock_slice_write_uses_private_scratch_and_storage_image() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @k(%"struct.metal::_imageblock_base" %img_blk, ptr addrspace(1) %dst, <2 x i32> %gid, <2 x i16> %tid) {
entry:
  %width = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %dst, i32 0)
  %ptr = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  store <4 x half> zeroinitializer, ptr addrspace(4) %ptr, align 8
  tail call void @air.write_imageblock_slice_to_texture_2d.v4f16(ptr addrspace(1) %dst, ptr addrspace(4) %ptr, i1 false, <2 x i16> zeroinitializer, <2 x i16> zeroinitializer, <2 x i32> %gid, i32 0, i1 false, i32 2)
  ret void
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.write_imageblock_slice_to_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i32>, i32, i1, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6, !7}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 8, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{i32 0, i32 8, i32 0, !"half4", !"v"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"dst"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}
!7 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_imageblock_slice_write_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Rgba16f"), "{asm}");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(!asm.contains("air.imageblock"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_imageblock_slice_write_zero_extent_writes_transparent_zero() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @k(%"struct.metal::_imageblock_base" %img_blk, ptr addrspace(1) %dst, <2 x i16> %gid, <2 x i16> %tid) {
entry:
  %ptr = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  store <4 x half> zeroinitializer, ptr addrspace(4) %ptr, align 8
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.v4f16(ptr addrspace(1) %dst, ptr addrspace(4) %ptr, i1 true, <2 x i16> zeroinitializer, <2 x i16> <i16 0, i16 1>, <2 x i16> %gid, i16 0, i1 false, i32 2)
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.write_imageblock_slice_to_texture_2d.i16.v4f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6, !7}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 8, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{i32 0, i32 8, i32 0, !"half4", !"v"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"dst"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!7 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_imageblock_zero_extent_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(asm.contains("OpIEqual"), "{asm}");
    assert!(asm.contains("OpConstantNull"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(asm.contains("4294967295"), "{asm}");
    assert!(!asm.contains("air.imageblock"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_imageblock_slice_write_retypes_byte_offset_half4() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @k(%"struct.metal::_imageblock_base" %img_blk, ptr addrspace(1) %dst, <2 x i16> %gid, <2 x i16> %tid) {
entry:
  %ptr = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  %field = getelementptr inbounds i8, ptr addrspace(4) %ptr, i64 8
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.v4f16(ptr addrspace(1) %dst, ptr addrspace(4) %field, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> %gid, i16 0, i1 false, i32 2)
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.write_imageblock_slice_to_texture_2d.i16.v4f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6, !7}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 16, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{i32 0, i32 8, i32 0, !"half4", !"a", i32 8, i32 8, i32 0, !"half4", !"b"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"dst"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!7 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_imageblock_slice_write_byte_offset_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Rgba16f"), "{asm}");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(!asm.contains("air.imageblock"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_imageblock_slice_write_two_channel_half_pads_to_v4() {
    // A 2-channel imageblock write (`.i16.v2f16`): the texel type comes from the intrinsic-name
    // suffix (`v2f16` = <2 x half>), the byte-offset field pointer is reinterpreted to it and loaded,
    // and the <2 x half> texel is converted to <2 x float>, channel-extracted, and padded to the v4
    // OpImageWrite expects (extra channels zero). Was previously rejected "unsupported texel component".
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @k(%"struct.metal::_imageblock_base" %img_blk, ptr addrspace(1) %dst, <2 x i16> %gid, <2 x i16> %tid) {
entry:
  %store_ptr = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  store <2 x half> zeroinitializer, ptr addrspace(4) %store_ptr, align 4
  %write_ptr = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.v2f16(ptr addrspace(1) %dst, ptr addrspace(4) %write_ptr, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> %gid, i16 0, i1 false, i32 2)
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.write_imageblock_slice_to_texture_2d.i16.v2f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6, !7}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 4, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{i32 0, i32 4, i32 0, !"half2", !"v"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"dst"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!7 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_imageblock_slice_write_v2f16_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(asm.contains("OpCompositeExtract"), "{asm}");
    assert!(!asm.contains("air.imageblock"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_imageblock_slice_write_int16_widens_to_v4_sint() {
    // A non-float imageblock write (`.i16.v4i16`) into a SIGNED-integer storage image
    // (`texture2d<short, write>` → Sint sampled type): the <4 x i16> texel is sign-extended (SConvert)
    // to the image's 32-bit sint sampled type and written with OpImageWrite. Was previously rejected
    // "non-float imageblock write"; validated byte-exact vs the Apple goldens on the two regression cases
    // (10/ae048d1d rwppCornerResponseShort, 13/15ce2819 rwppCornerDetectFirstPass4x4Short).
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @k(%"struct.metal::_imageblock_base" %img_blk, ptr addrspace(1) %dst, <2 x i16> %gid, <2 x i16> %tid) {
entry:
  %store_ptr = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  store <4 x i16> zeroinitializer, ptr addrspace(4) %store_ptr, align 8
  %write_ptr = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.v4i16(ptr addrspace(1) %dst, ptr addrspace(4) %write_ptr, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> %gid, i16 0, i1 false, i32 2)
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.write_imageblock_slice_to_texture_2d.i16.v4i16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6, !7}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 8, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{i32 0, i32 8, i32 0, !"short4", !"v"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<short, write>", !"air.arg_name", !"dst"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!7 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_imageblock_slice_write_v4i16_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(asm.contains("OpSConvert"), "{asm}");
    assert!(!asm.contains("air.imageblock"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_imageblock_dimensions_lower_to_single_slot_extent() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @k(%"struct.metal::_imageblock_base" %img_blk, ptr addrspace(1) %dst, <2 x i16> %gid, <2 x i16> %tid) {
entry:
  %ptr = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  store <4 x float> zeroinitializer, ptr addrspace(4) %ptr, align 16
  %width = tail call i16 @air.get_imageblock_width()
  %size_x = insertelement <2 x i16> undef, i16 %width, i64 0
  %height = tail call i16 @air.get_imageblock_height()
  %size = insertelement <2 x i16> %size_x, i16 %height, i64 1
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.v4f32(ptr addrspace(1) %dst, ptr addrspace(4) %ptr, i1 false, <2 x i16> %size, <2 x i16> zeroinitializer, <2 x i16> %gid, i16 0, i1 false, i32 2)
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare i16 @air.get_imageblock_width()
declare i16 @air.get_imageblock_height()
declare void @air.write_imageblock_slice_to_texture_2d.i16.v4f32(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6, !7}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 16, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{i32 0, i32 16, i32 0, !"float4", !"v"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"dst"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!7 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_imageblock_dimensions_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.trim_end().ends_with(" 1")),
        "{asm}"
    );
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(!asm.contains("air.get_imageblock"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_imageblock_data_uses_metadata_float4_pointee() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @k(%"struct.metal::_imageblock_base" %img_blk, <2 x i16> %tid) {
entry:
  %ptr = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  store <4 x float> zeroinitializer, ptr addrspace(4) %ptr, align 16
  %loaded = load <4 x float>, ptr addrspace(4) %ptr, align 16
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 16, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{i32 0, i32 16, i32 0, !"float4", !"v"}
!5 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_imageblock_data_float4_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpStore"), "{asm}");
    assert!(asm.contains("OpLoad"), "{asm}");
    assert!(asm.contains("OpTypeFloat 32"), "{asm}");
    assert!(!asm.contains("OpTypeFloat 16"), "{asm}");
    assert!(!asm.contains("OpBitcast %_ptr_"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_imageblock_data_calls_share_private_scratch() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @k(%"struct.metal::_imageblock_base" %img_blk, ptr addrspace(1) %dst, <2 x i16> %gid, <2 x i16> %tid) {
entry:
  %store_ptr = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  store <4 x half> zeroinitializer, ptr addrspace(4) %store_ptr, align 8
  %write_ptr = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  tail call void @air.write_imageblock_slice_to_texture_2d.v4f16(ptr addrspace(1) %dst, ptr addrspace(4) %write_ptr, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> %gid, i16 0, i1 false, i32 2)
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.write_imageblock_slice_to_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6, !7}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 8, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{i32 0, i32 8, i32 0, !"half4", !"v"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"dst"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!7 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_imageblock_shared_scratch_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let private_vars = asm
        .lines()
        .filter(|line| line.contains("OpVariable") && line.contains("Private"))
        .count();
    assert_eq!(private_vars, 1, "{asm}");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_imageblock_private_byte_view_loads_typed_half_lane() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @k(%"struct.metal::_imageblock_base" %img_blk, <2 x i16> %tid) {
entry:
  %base = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  store half 0xH3C00, ptr addrspace(4) %base, align 2
  %byte_lane = getelementptr inbounds i8, ptr addrspace(4) %base, i64 2
  %lane = bitcast ptr addrspace(4) %byte_lane to ptr addrspace(4)
  store half 0xH4000, ptr addrspace(4) %lane, align 2
  %value = load half, ptr addrspace(4) %lane, align 2
  store half %value, ptr addrspace(4) %base, align 2
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 8, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{i32 0, i32 2, i32 0, !"half", !"a", i32 2, i32 2, i32 0, !"half", !"b", i32 4, i32 2, i32 0, !"half", !"c", i32 6, i32 2, i32 0, !"half", !"d"}
!5 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_imageblock_private_byte_half_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpLoad"), "{asm}");
    assert!(!asm.contains("OpPtrAccessChain"), "{asm}");
    assert!(!asm.contains("OpBitcast %_ptr_"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_cross_coordinate_imageblock_uses_shared_workgroup_cells() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @k(%"struct.metal::_imageblock_base" %img_blk, <2 x i16> %tid, <2 x i16> %threads) {
entry:
  %own = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  store half 0xH3C00, ptr addrspace(4) %own, align 2
  tail call void @air.wg.barrier(i32 8, i32 1)
  %neighbor_coord = add <2 x i16> %tid, <i16 1, i16 0>
  %neighbor = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %neighbor_coord, i32 0, i16 0)
  %value = load half, ptr addrspace(4) %neighbor, align 2
  store half %value, ptr addrspace(4) %own, align 2
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 2, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{i32 0, i32 2, i32 0, !"half", !"v"}
!5 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
!6 = !{i32 2, !"air.threads_per_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"threads"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_cross_coordinate_imageblock_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpVariable") && line.contains("Workgroup")),
        "{asm}"
    );
    assert!(
        !asm.lines()
            .any(|line| line.contains("OpVariable") && line.contains("Private")),
        "{asm}"
    );
    assert!(asm.contains("OpIMul"), "{asm}");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.ends_with(" 512")),
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
fn native_kernel_cross_coordinate_imageblock_write_reads_struct_prefix_without_pointer_bitcast() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @k(%"struct.metal::_imageblock_base" %img_blk, ptr addrspace(1) %dst, <2 x i16> %gid, <2 x i16> %tid, <2 x i16> %threads) {
entry:
  %own = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  store half 0xH3C00, ptr addrspace(4) %own, align 2
  %neighbor_coord = add <2 x i16> %tid, <i16 1, i16 0>
  %neighbor = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %neighbor_coord, i32 0, i16 0)
  store half 0xH4000, ptr addrspace(4) %neighbor, align 2
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.f16(ptr addrspace(1) %dst, ptr addrspace(4) %own, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> %gid, i16 0, i1 false, i32 2)
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.write_imageblock_slice_to_texture_2d.i16.f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6, !7, !8}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 8, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{i32 0, i32 2, i32 0, !"half", !"luma", i32 4, i32 4, i32 0, !"half2", !"chroma"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"dst"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!7 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
!8 = !{i32 4, !"air.threads_per_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"threads"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_cross_coordinate_imageblock_prefix_write_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(!asm.contains("OpBitcast %_ptr_Workgroup"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_private_imageblock_byte_field_keeps_complete_cell_layout() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @k(%"struct.metal::_imageblock_base" %img_blk, ptr addrspace(1) %dst, <2 x i16> %gid, <2 x i16> %tid) {
entry:
  %cell = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  %luma = bitcast ptr addrspace(4) %cell to ptr addrspace(4)
  store half 0xH3C00, ptr addrspace(4) %luma, align 2
  %chroma = getelementptr inbounds i8, ptr addrspace(4) %cell, i64 4
  store <2 x half> zeroinitializer, ptr addrspace(4) %chroma, align 4
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.f16(ptr addrspace(1) %dst, ptr addrspace(4) %cell, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> %gid, i16 0, i1 false, i32 2)
  tail call void @air.write_imageblock_slice_to_texture_2d.i16.v2f16(ptr addrspace(1) %dst, ptr addrspace(4) %chroma, i1 false, <2 x i16> zeroinitializer, <2 x i16> undef, <2 x i16> %gid, i16 0, i1 false, i32 2)
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
declare void @air.write_imageblock_slice_to_texture_2d.i16.f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)
declare void @air.write_imageblock_slice_to_texture_2d.i16.v2f16(ptr addrspace(1), ptr addrspace(4), i1, <2 x i16>, <2 x i16>, <2 x i16>, i16, i1, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6, !7}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 8, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{i32 0, i32 2, i32 0, !"half", !"luma", i32 4, i32 4, i32 0, !"half2", !"chroma"}
!5 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"dst"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!7 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_private_imageblock_byte_field_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(asm.matches("OpImageWrite").count(), 2, "{asm}");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(!asm.contains("OpBitcast %_ptr_Private"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_backend_imageblock_dimensions_use_shared_workgroup_cells() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

define void @k(%"struct.metal::_imageblock_base" %img_blk, <2 x i16> %tid) {
entry:
  %ptr = tail call ptr addrspace(4) @air.imageblock_data(<2 x i16> %tid, i32 0, i16 0)
  store i32 7, ptr addrspace(4) %ptr, align 4
  ret void
}

declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 4, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{i32 0, i32 4, i32 0, !"uint", !"v"}
!5 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
!apv.imageblock_dimensions = !{!6}
!6 = !{i32 4, i32 1}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_imageblock_workgroup_scratch_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpVariable") && line.contains("Workgroup")),
        "{asm}"
    );
    assert!(
        !asm.lines()
            .any(|line| line.contains("OpVariable") && line.contains("Private")),
        "{asm}"
    );
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_helper_wrapped_sample_and_write_textures_use_single_images() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Out = type <{ %Tex, %Tex, <2 x i16> }>
%Tex = type { ptr addrspace(1) }

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601053257, i64 0], align 8

define void @k(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
entry:
  %out = alloca %Out, align 8
  %src_field = getelementptr inbounds %Out, ptr %out, i64 0, i32 0, i32 0
  store ptr addrspace(1) %src, ptr %src_field, align 8
  %dst_field = getelementptr inbounds %Out, ptr %out, i64 0, i32 1, i32 0
  store ptr addrspace(1) %dst, ptr %dst_field, align 8
  %coord_field = getelementptr inbounds %Out, ptr %out, i64 0, i32 2
  store <2 x i16> zeroinitializer, ptr %coord_field, align 8
  call void @helper(ptr %out)
  ret void
}

define internal void @helper(ptr %out) {
entry:
  %coord_field = getelementptr inbounds %Out, ptr %out, i64 0, i32 2
  %coord = load <2 x i16>, ptr %coord_field, align 8
  %src_field = getelementptr inbounds %Out, ptr %out, i64 0, i32 0, i32 0
  %src = load ptr addrspace(1), ptr %src_field, align 8
  %dst_field = getelementptr inbounds %Out, ptr %out, i64 0, i32 1, i32 0
  %dst = load ptr addrspace(1), ptr %dst_field, align 8
  %sample = tail call <4 x half> @air.sample_texture_2d.v4f16(ptr addrspace(1) %src, ptr addrspace(2) @__air_sampler_state, <2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  tail call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %dst, <2 x i16> %coord, <4 x half> %sample, i16 0, i32 2)
  ret void
}

declare <4 x half> @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"dst"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_helper_wrapped_sample_write_textures_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm.contains("OpImageSampleExplicitLod") || asm.contains("OpImageFetch"),
        "{asm}"
    );
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_separate_helper_wrapped_sample_and_write_textures_recover_field_sources() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%In = type { %Tex }
%Out = type <{ %Tex, <2 x i16>, i8 }>
%Tex = type { ptr addrspace(1) }

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601053257, i64 0], align 8

define void @k(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
entry:
  %out = alloca %Out, align 8
  %in = alloca %In, align 8
  %dst_field = getelementptr inbounds %Out, ptr %out, i64 0, i32 0, i32 0
  store ptr addrspace(1) %dst, ptr %dst_field, align 8
  %coord_field = getelementptr inbounds %Out, ptr %out, i64 0, i32 1
  store <2 x i16> zeroinitializer, ptr %coord_field, align 8
  %flag_field = getelementptr inbounds %Out, ptr %out, i64 0, i32 2
  store i8 0, ptr %flag_field, align 4
  %src_field = getelementptr inbounds %In, ptr %in, i64 0, i32 0, i32 0
  store ptr addrspace(1) %src, ptr %src_field, align 8
  call void @helper(ptr %in, ptr %out)
  ret void
}

define internal void @helper(ptr %in, ptr %out) {
entry:
  %coord_field = getelementptr inbounds %Out, ptr %out, i64 0, i32 1
  %coord = load <2 x i16>, ptr %coord_field, align 8
  %src_field = getelementptr inbounds %In, ptr %in, i64 0, i32 0, i32 0
  %src = load ptr addrspace(1), ptr %src_field, align 8
  %sample = tail call <4 x half> @air.sample_texture_2d.v4f16(ptr addrspace(1) %src, ptr addrspace(2) @__air_sampler_state, <2 x float> zeroinitializer, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %dst_field = getelementptr inbounds %Out, ptr %out, i64 0, i32 0, i32 0
  %dst = load ptr addrspace(1), ptr %dst_field, align 8
  tail call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %dst, <2 x i16> %coord, <4 x half> %sample, i16 0, i32 2)
  ret void
}

declare <4 x half> @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!5}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"dst"}
!5 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_helper_wrapped_separate_sample_write_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm.contains("OpImageSampleExplicitLod") || asm.contains("OpImageFetch"),
        "{asm}"
    );
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_uint_array_write_texture_combines_layer_coord() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %dst, <3 x i32> %tid) {
entry:
  %x = extractelement <3 x i32> %tid, i64 0
  %y = extractelement <3 x i32> %tid, i64 1
  %layer = extractelement <3 x i32> %tid, i64 2
  %xy0 = insertelement <2 x i32> poison, i32 %x, i64 0
  %xy = insertelement <2 x i32> %xy0, i32 %y, i64 1
  %v0 = insertelement <4 x i32> undef, i32 1, i64 0
  %v1 = insertelement <4 x i32> %v0, i32 2, i64 1
  %v2 = insertelement <4 x i32> %v1, i32 3, i64 2
  %v3 = insertelement <4 x i32> %v2, i32 4, i64 3
  tail call void @air.write_texture_2d_array.u.v4i32(ptr addrspace(1) %dst, <2 x i32> %xy, i32 %layer, <4 x i32> %v3, i32 0, i32 2)
  ret void
}

declare void @air.write_texture_2d_array.u.v4i32(ptr addrspace(1), <2 x i32>, i32, <4 x i32>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.write", !"air.arg_type_name", !"texture2d_array<uint, write>", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_uint_array_write_texture_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Rgba8ui"), "{asm}");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(asm.contains("Binding 34"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_cube_write_texture_combines_face_coord() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %dst, <2 x i32> %xy) {
entry:
  %v0 = insertelement <4 x float> poison, float 2.500000e-01, i64 0
  %v1 = insertelement <4 x float> %v0, float 5.000000e-01, i64 1
  %v2 = insertelement <4 x float> %v1, float 7.500000e-01, i64 2
  %color = insertelement <4 x float> %v2, float 1.000000e+00, i64 3
  tail call void @air.write_texture_cube.v4f32(ptr addrspace(1) %dst, <2 x i32> %xy, i32 2, <4 x float> %color, i32 0, i32 2)
  ret void
}

declare void @air.write_texture_cube.v4f32(ptr addrspace(1), <2 x i32>, i32, <4 x float>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texturecube<float, write>", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"xy"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_cube_write_texture_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Cube 0 0 0 2 Rgba32f"), "{asm}");
    assert!(asm.contains("OpCompositeConstruct"), "{asm}");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(asm.contains("Binding 32"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_read_texture_array_widens_narrow_coord_before_layer_combine() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %src, ptr addrspace(1) %dst, <2 x i16> %gid) {
entry:
  %sampler = tail call ptr addrspace(2) @air.get_read_sampler()
  %read = tail call { <4 x half>, i8 } @air.read_texture_2d_array.i16.v4f16(ptr addrspace(1) %src, ptr addrspace(2) %sampler, <2 x i16> %gid, i16 0, <2 x i16> zeroinitializer, i16 0, i32 0)
  %texel = extractvalue { <4 x half>, i8 } %read, 0
  tail call void @air.write_texture_2d_array.i16.v4f16(ptr addrspace(1) %dst, <2 x i16> %gid, i16 0, <4 x half> %texel, i16 0, i32 2)
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { <4 x half>, i8 } @air.read_texture_2d_array.i16.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x i16>, i16, <2 x i16>, i16, i32)
declare void @air.write_texture_2d_array.i16.v4f16(ptr addrspace(1), <2 x i16>, i16, <4 x half>, i16, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d_array<half, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 2, i32 1, !"air.write", !"air.arg_type_name", !"texture2d_array<half, write>", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_read_texture_array_narrow_coord_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(asm.contains("OpCompositeConstruct"), "{asm}");
    assert!(asm.contains("OpImageFetch"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_read_depth_2d_i16_with_sampler_uses_vector_coord() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %depth, ptr addrspace(1) %out, <2 x i16> %gid) {
entry:
  %sampler = tail call ptr addrspace(2) @air.get_read_sampler()
  %read = tail call { float, i8 } @air.read_depth_2d.i16.f32(ptr addrspace(1) %depth, ptr addrspace(2) %sampler, i32 1, <2 x i16> %gid, <2 x i16> zeroinitializer, i16 0, i32 1)
  %value = extractvalue { float, i8 } %read, 0
  store float %value, ptr addrspace(1) %out, align 4
  ret void
}

declare ptr addrspace(2) @air.get_read_sampler()
declare { float, i8 } @air.read_depth_2d.i16.f32(ptr addrspace(1), ptr addrspace(2), i32, <2 x i16>, <2 x i16>, i16, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"depth"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float*", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_read_depth_sampler_coord_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load spv");
    let fetch = module
        .all_inst_iter()
        .find(|inst| inst.class.opcode == Op::ImageFetch)
        .expect("image fetch");
    let coord = match fetch.operands.get(1) {
        Some(Operand::IdRef(id)) => *id,
        other => panic!("unexpected image fetch coordinate operand: {other:?}"),
    };
    let coord_ty = module
        .all_inst_iter()
        .find(|inst| inst.result_id == Some(coord))
        .and_then(|inst| inst.result_type)
        .expect("coordinate type");
    let coord_ty_inst = module
        .all_inst_iter()
        .find(|inst| inst.result_id == Some(coord_ty))
        .expect("coordinate type definition");
    assert_eq!(coord_ty_inst.class.opcode, Op::TypeVector, "{asm}");
    assert!(
        matches!(
            coord_ty_inst.operands.get(1),
            Some(Operand::LiteralBit32(2))
        ),
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
fn native_read_depth_2d_i16_sample_index_uses_vector_coord() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %depth, ptr addrspace(1) %out, <2 x i16> %gid) {
entry:
  %read = tail call { float, i8 } @air.read_depth_2d.i16.f32(ptr addrspace(1) %depth, i32 1, <2 x i16> %gid, i16 0, i32 1)
  %value = extractvalue { float, i8 } %read, 0
  store float %value, ptr addrspace(1) %out, align 4
  ret void
}

declare { float, i8 } @air.read_depth_2d.i16.f32(ptr addrspace(1), i32, <2 x i16>, i16, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"depth2d<float, read>", !"air.arg_name", !"depth"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float*", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_read_depth_sample_index_coord_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load spv");
    let fetch = module
        .all_inst_iter()
        .find(|inst| inst.class.opcode == Op::ImageFetch)
        .expect("image fetch");
    let coord = match fetch.operands.get(1) {
        Some(Operand::IdRef(id)) => *id,
        other => panic!("unexpected image fetch coordinate operand: {other:?}"),
    };
    let coord_ty = module
        .all_inst_iter()
        .find(|inst| inst.result_id == Some(coord))
        .and_then(|inst| inst.result_type)
        .expect("coordinate type");
    let coord_ty_inst = module
        .all_inst_iter()
        .find(|inst| inst.result_id == Some(coord_ty))
        .expect("coordinate type definition");
    assert_eq!(coord_ty_inst.class.opcode, Op::TypeVector, "{asm}");
    assert!(
        matches!(
            coord_ty_inst.operands.get(1),
            Some(Operand::LiteralBit32(2))
        ),
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
fn native_kernel_pixel_sampler_array_sample_lowers_to_fetch() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601050697, i64 0], align 8

define void @k(ptr addrspace(1) %src, ptr addrspace(1) %dst, <2 x i16> %gid) {
entry:
  %c0 = insertelement <2 x float> poison, float 1.000000e+00, i32 0
  %coord = insertelement <2 x float> %c0, float 2.000000e+00, i32 1
  %sample = tail call <4 x half> @air.sample_texture_2d_array.v4f16(ptr addrspace(1) %src, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i16 0, i1 false, float 0.000000e+00, i32 0)
  tail call void @air.write_texture_2d_array.i16.v4f16(ptr addrspace(1) %dst, <2 x i16> %gid, i16 0, <4 x half> %sample, i16 0, i32 2)
  ret void
}

declare <4 x half> @air.sample_texture_2d_array.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i16, i1, float, i32)
declare void @air.write_texture_2d_array.i16.v4f16(ptr addrspace(1), <2 x i16>, i16, <4 x half>, i16, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d_array<half, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d_array<half, write>", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_pixel_sampler_array_sample_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageFetch"), "{asm}");
    assert!(asm.contains("OpConvertFToU"), "{asm}");
    assert!(!asm.contains("OpSampledImage"), "{asm}");
    assert!(!asm.contains("OpImageSample"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_pixel_linear_array_sampler_lowers_to_weighted_fetches() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601053184, i64 0], align 8

define void @k(ptr addrspace(1) %src, ptr addrspace(1) %dst, <2 x i16> %gid) {
entry:
  %c0 = insertelement <2 x float> poison, float 1.000000e+00, i32 0
  %coord = insertelement <2 x float> %c0, float 2.000000e+00, i32 1
  %sample = tail call { <4 x half>, i8 } @air.sample_texture_2d_array.v4f16(ptr addrspace(1) %src, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i32 1, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %color = extractvalue { <4 x half>, i8 } %sample, 0
  tail call void @air.write_texture_2d_array.i16.v4f16(ptr addrspace(1) %dst, <2 x i16> %gid, i16 1, <4 x half> %color, i16 0, i32 2)
  ret void
}

declare { <4 x half>, i8 } @air.sample_texture_2d_array.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i32, i1, <2 x i32>, i1, float, float, i32)
declare void @air.write_texture_2d_array.i16.v4f16(ptr addrspace(1), <2 x i16>, i16, <4 x half>, i16, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d_array<half, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d_array<half, write>", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_pixel_linear_array_sampler_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(asm.matches("OpImageFetch").count(), 4, "{asm}");
    assert!(asm.contains("OpImageQuerySizeLod"), "{asm}");
    assert!(asm.contains(" Floor "), "{asm}");
    assert!(asm.contains("OpVectorTimesScalar"), "{asm}");
    assert_eq!(asm.matches("OpFOrdEqual").count(), 4, "{asm}");
    assert!(asm.contains("OpFAdd"), "{asm}");
    assert!(asm.contains("OpLogicalAnd"), "{asm}");
    assert!(!asm.contains("OpSampledImage"), "{asm}");
    assert!(!asm.contains("OpImageSample"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_integer_pixel_linear_sampler_lowers_to_fetch() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601053257, i64 0], align 8

define void @k(ptr addrspace(1) %src, ptr addrspace(1) %dst, <2 x i16> %gid) {
entry:
  %c0 = insertelement <2 x float> poison, float 2.500000e-01, i32 0
  %coord = insertelement <2 x float> %c0, float 7.500000e-01, i32 1
  %sample = tail call { <4 x i16>, i8 } @air.sample_texture_2d.u.v4i16(ptr addrspace(1) %src, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %color = extractvalue { <4 x i16>, i8 } %sample, 0
  tail call void @air.write_texture_2d.i16.u.v4i16(ptr addrspace(1) %dst, <2 x i16> %gid, <4 x i16> %color, i16 0, i32 2)
  ret void
}

declare { <4 x i16>, i8 } @air.sample_texture_2d.u.v4i16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare void @air.write_texture_2d.i16.u.v4i16(ptr addrspace(1), <2 x i16>, <4 x i16>, i16, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<uint, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<ushort, write>", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_integer_pixel_linear_sampler_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageFetch"), "{asm}");
    assert!(asm.contains("OpConvertFToS"), "{asm}");
    // Metal's pixel-space nearest fetch selects texel floor(coord); the coord is
    // floored before ConvertFToS (truncation would pick the wrong texel at negative
    // coordinates), matching the normalized-nearest and pixel-linear paths.
    assert!(asm.contains(" Floor "), "{asm}");
    assert!(!asm.contains("OpFMul"), "{asm}");
    assert!(!asm.contains("OpSampledImage"), "{asm}");
    assert!(!asm.contains("OpImageSample"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_pixel_sampler_zero_address_guards_fetch_result() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601050624, i64 0], align 8

define void @k(ptr addrspace(1) %src, ptr addrspace(1) %dst, <2 x i16> %gid) {
entry:
  %signed = tail call fast float @air.convert.f.f32.s.i32(i32 1)
  %coord_x = fadd fast float %signed, 8.000000e+00
  %c0 = insertelement <2 x float> poison, float %coord_x, i32 0
  %coord = insertelement <2 x float> %c0, float 2.000000e+00, i32 1
  %sample = tail call <4 x half> @air.sample_texture_2d.v4f16(ptr addrspace(1) %src, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  tail call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %dst, <2 x i16> %gid, <4 x half> %sample, i16 0, i32 2)
  ret void
}

declare float @air.convert.f.f32.s.i32(i32)
declare <4 x half> @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_pixel_sampler_zero_address_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageFetch"), "{asm}");
    assert!(asm.contains("OpImageQuerySizeLod"), "{asm}");
    assert!(asm.contains("OpLogicalAnd"), "{asm}");
    assert!(asm.contains("OpConstantNull"), "{asm}");
    assert_eq!(asm.matches("OpTypeInt 32 1").count(), 1, "{asm}");
    assert!(!asm.contains("OpSampledImage"), "{asm}");
    assert!(!asm.contains("OpImageSample"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_pixel_sampler_2d_sample_offset_lowers_to_fetch_coord_add() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601050697, i64 0], align 8

define void @k(ptr addrspace(1) %src, ptr addrspace(1) %dst, <2 x i16> %gid) {
entry:
  %c0 = insertelement <2 x float> poison, float 1.000000e+00, i32 0
  %coord = insertelement <2 x float> %c0, float 2.000000e+00, i32 1
  %sample = tail call <4 x half> @air.sample_texture_2d.v4f16(ptr addrspace(1) %src, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> <i32 3, i32 0>, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  tail call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %dst, <2 x i16> %gid, <4 x half> %sample, i16 0, i32 2)
  ret void
}

declare <4 x half> @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!6}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!6 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_pixel_sampler_2d_offset_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageFetch"), "{asm}");
    assert!(asm.contains("OpIAdd"), "{asm}");
    assert!(!asm.contains("OpSampledImage"), "{asm}");
    assert!(!asm.contains("OpImageSample"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_pixel_sampler_dynamic_offset_lowers_to_fetch_coord_add() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 34901797601050697, i64 0], align 8

define void @k(ptr addrspace(1) %src, ptr addrspace(1) %dst, <2 x i16> %gid, i32 %lane) {
entry:
  %c0 = insertelement <2 x float> poison, float 1.000000e+00, i32 0
  %coord = insertelement <2 x float> %c0, float 2.000000e+00, i32 1
  %x = or i32 %lane, -8
  %o0 = insertelement <2 x i32> undef, i32 %x, i32 0
  %offset = insertelement <2 x i32> %o0, i32 0, i32 1
  %sample = tail call <4 x half> @air.sample_texture_2d.v4f16(ptr addrspace(1) %src, ptr addrspace(2) @__air_sampler_state, <2 x float> %coord, i1 true, <2 x i32> %offset, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  tail call void @air.write_texture_2d.i16.v4f16(ptr addrspace(1) %dst, <2 x i16> %gid, <4 x half> %sample, i16 0, i32 2)
  ret void
}

declare <4 x half> @air.sample_texture_2d.v4f16(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare void @air.write_texture_2d.i16.v4f16(ptr addrspace(1), <2 x i16>, <4 x half>, i16, i32)

!air.kernel = !{!0}
!air.sampler_states = !{!7}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<half, sample>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<half, write>", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"int", !"air.arg_name", !"lane"}
!7 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_pixel_sampler_dynamic_offset_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageFetch"), "{asm}");
    assert!(asm.contains("OpIAdd"), "{asm}");
    assert!(!asm.contains("OpSampledImage"), "{asm}");
    assert!(!asm.contains("OpImageSample"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_texture_buffer_write_uses_storage_texel_buffer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %dst, i32 %tid) {
entry:
  %v0 = insertelement <4 x float> undef, float 2.500000e-01, i64 0
  %v1 = insertelement <4 x float> %v0, float 5.000000e-01, i64 1
  %v2 = insertelement <4 x float> %v1, float 7.500000e-01, i64 2
  %v3 = insertelement <4 x float> %v2, float 1.000000e+00, i64 3
  tail call void @air.write_texture_buffer_1d.v4f32(ptr addrspace(1) %dst, i32 %tid, <4 x float> %v3, i32 2)
  ret void
}

declare void @air.write_texture_buffer_1d.v4f32(ptr addrspace(1), i32, <4 x float>, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture_buffer<float, write>", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_texture_buffer_write_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability SampledBuffer"), "{asm}");
    assert!(asm.contains("OpCapability ImageBuffer"), "{asm}");
    assert!(asm.contains("Buffer 0 0 0 2 Rgba32f"), "{asm}");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(asm.contains("Binding 32"), "{asm}");
    assert!(!asm.contains("OpCapability Sampled1D"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_texture1d_write_declares_storage_image1d_capability() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %dst, i16 %x) {
entry:
  %v0 = insertelement <4 x float> undef, float 2.500000e-01, i64 0
  %v1 = insertelement <4 x float> %v0, float 5.000000e-01, i64 1
  %v2 = insertelement <4 x float> %v1, float 7.500000e-01, i64 2
  %v3 = insertelement <4 x float> %v2, float 1.000000e+00, i64 3
  tail call void @air.write_texture_1d.i16.v4f32(ptr addrspace(1) %dst, i16 %x, <4 x float> %v3, i16 0, i32 2)
  ret void
}

declare void @air.write_texture_1d.i16.v4f32(ptr addrspace(1), i16, <4 x float>, i16, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture1d<float, write>", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort", !"air.arg_name", !"x"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_texture1d_write_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability Sampled1D"), "{asm}");
    assert!(asm.contains("OpCapability Image1D"), "{asm}");
    assert!(asm.contains("1D 0 0 0 2 Rgba32f"), "{asm}");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_ushort_texture_read_write_uses_16bit_uint_images() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %src, ptr addrspace(1) %dst, <2 x i16> %tid) {
entry:
  %coord = shl <2 x i16> %tid, <i16 1, i16 1>
  %sample = tail call { <4 x i16>, i8 } @air.read_texture_2d.i16.u.v4i16(ptr addrspace(1) %src, <2 x i16> %coord, i16 0, i32 1)
  %color = extractvalue { <4 x i16>, i8 } %sample, 0
  %wide = zext <4 x i16> %color to <4 x i32>
  %quot = udiv <4 x i32> %wide, %wide
  %zero = icmp eq <4 x i32> %wide, zeroinitializer
  %selected = select <4 x i1> %zero, <4 x i32> zeroinitializer, <4 x i32> %quot
  %narrow = trunc <4 x i32> %selected to <4 x i16>
  %isnull = tail call i1 @air.is_null_texture_2d(ptr addrspace(1) %dst)
  tail call void @air.write_texture_2d.i16.u.v4i16(ptr addrspace(1) %dst, <2 x i16> %coord, <4 x i16> %narrow, i16 0, i32 2)
  ret void
}

declare { <4 x i16>, i8 } @air.read_texture_2d.i16.u.v4i16(ptr addrspace(1), <2 x i16>, i16, i32)
declare i1 @air.is_null_texture_2d(ptr addrspace(1))
declare void @air.write_texture_2d.i16.u.v4i16(ptr addrspace(1), <2 x i16>, <4 x i16>, i16, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<ushort, read>", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<ushort, write>", !"air.arg_name", !"dst"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_ushort_texture_read_write_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Rgba16ui"), "{asm}");
    assert!(asm.contains("OpImageFetch"), "{asm}");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(asm.contains("OpUDiv"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        if let Err(err) = tools::spirv_val_bytes(&spv, &tmp) {
            panic!("spirv-val: {err}\n{asm}");
        }
    }
}

#[test]
fn native_kernel_same_texture_read_write_uses_storage_image_read() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %tex, <2 x i32> %tid) {
entry:
  %sample = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %tex, <2 x i32> %tid, i32 0, i32 3)
  %color = extractvalue { <4 x float>, i8 } %sample, 0
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %tex, <2 x i32> %tid, <4 x float> %color, i32 0, i32 3)
  ret void
}

declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, i32, i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_same_texture_read_write_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Rgba32f"), "{asm}");
    assert!(asm.contains("OpImageRead"), "{asm}");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(!asm.contains("OpImageFetch"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_write_texture_size_query_uses_storage_query() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %dst, <2 x i32> %tid) {
entry:
  %w = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %dst, i32 0)
  %h = tail call i32 @air.get_height_texture_2d(ptr addrspace(1) %dst, i32 0)
  %wf = tail call fast float @air.convert.f.f32.u.i32(i32 %w)
  %hf = tail call fast float @air.convert.f.f32.u.i32(i32 %h)
  %v0 = insertelement <4 x float> undef, float %wf, i64 0
  %v1 = insertelement <4 x float> %v0, float %hf, i64 1
  %v2 = insertelement <4 x float> %v1, float 0.000000e+00, i64 2
  %v3 = insertelement <4 x float> %v2, float 1.000000e+00, i64 3
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %dst, <2 x i32> %tid, <4 x float> %v3, i32 0, i32 2)
  ret void
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare i32 @air.get_height_texture_2d(ptr addrspace(1), i32)
declare float @air.convert.f.f32.u.i32(i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 7, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_write_texture_size_query_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageQuerySize "), "{asm}");
    assert!(!asm.contains("OpImageQuerySizeLod"), "{asm}");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_write_texture_mip_query_uses_single_level_constant() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %dst, <2 x i32> %tid) {
entry:
  %levels = tail call i32 @air.get_num_mip_levels_texture_2d(ptr addrspace(1) %dst)
  %levels_f = tail call fast float @air.convert.f.f32.u.i32(i32 %levels)
  %v0 = insertelement <4 x float> undef, float %levels_f, i64 0
  %v1 = insertelement <4 x float> %v0, float 0.000000e+00, i64 1
  %v2 = insertelement <4 x float> %v1, float 0.000000e+00, i64 2
  %v3 = insertelement <4 x float> %v2, float 1.000000e+00, i64 3
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %dst, <2 x i32> %tid, <4 x float> %v3, i32 0, i32 2)
  ret void
}

declare i32 @air.get_num_mip_levels_texture_2d(ptr addrspace(1))
declare float @air.convert.f.f32.u.i32(i32)
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 7, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_write_texture_mip_query_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("OpImageQueryLevels"), "{asm}");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_texture_fence_lowers_to_image_memory_barrier() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"

define void @texture_fence(ptr addrspace(1) %tex) {
entry:
  tail call void @air.fence_texture_2d(ptr addrspace(1) %tex)
  ret void
}

declare void @air.fence_texture_2d(ptr addrspace(1))

!air.kernel = !{!0}
!0 = !{ptr @texture_fence, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"tex"}
"#;
    let tmp =
        std::env::temp_dir().join(format!("metal2vulkan_texture_fence_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpMemoryBarrier"), "{asm}");
    assert!(!asm.contains("air.fence_texture"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}
