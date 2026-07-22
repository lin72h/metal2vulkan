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
use std::path::PathBuf;

#[test]
fn native_llvm_float_minmax_calls_lower_as_extinsts() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main() {
entry:
  %mx = tail call fast float @llvm.maxnum.f32(float 1.000000e+00, float 2.000000e+00)
  %mn = tail call fast float @llvm.minnum.f32(float %mx, float 3.000000e+00)
  ret void
}

declare float @llvm.maxnum.f32(float, float)
declare float @llvm.minnum.f32(float, float)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_llvm_minmax_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.lines().any(|line| line.contains(" FMax ")), "{asm}");
    assert!(asm.lines().any(|line| line.contains(" FMin ")), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("llvm.maxnum"), "{asm}");
    assert!(!asm.contains("llvm.minnum"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_integer_clamp_uses_integer_extinsts() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main(i32 %x, i32 %y) {
entry:
  %u = tail call i32 @air.clamp.u.i32(i32 %x, i32 0, i32 255)
  %s = tail call i32 @air.clamp.s.i32(i32 %y, i32 -4, i32 4)
  ret void
}

declare i32 @air.clamp.u.i32(i32, i32, i32)
declare i32 @air.clamp.s.i32(i32, i32, i32)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_intclamp_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains(" UClamp "), "{asm}");
    assert!(asm.contains(" SClamp "), "{asm}");
    assert!(!asm.contains(" FClamp "), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_abs_diff_u8_vector_lowers_to_integer_minmax_sub() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %out) {
entry:
  %a0 = insertelement <3 x i8> poison, i8 9, i64 0
  %a1 = insertelement <3 x i8> %a0, i8 2, i64 1
  %a = insertelement <3 x i8> %a1, i8 200, i64 2
  %b0 = insertelement <3 x i8> poison, i8 3, i64 0
  %b1 = insertelement <3 x i8> %b0, i8 7, i64 1
  %b = insertelement <3 x i8> %b1, i8 5, i64 2
  %d = tail call <3 x i8> @air.abs_diff.u.v3i8(<3 x i8> %a, <3 x i8> %b)
  store <3 x i8> %d, ptr addrspace(1) %out, align 4
  ret void
}

declare <3 x i8> @air.abs_diff.u.v3i8(<3 x i8>, <3 x i8>)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uchar3*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_abs_diff_u8_vector_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains(" UMax "), "{asm}");
    assert!(asm.contains(" UMin "), "{asm}");
    assert!(asm.contains("OpISub"), "{asm}");
    assert!(!asm.contains("abs_diff"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_unsigned_saturating_add_sub_vectors_lower_to_compare_select() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define <3 x i8> @sat_vec(<3 x i8> %a, <3 x i8> %b) {
entry:
  %sum = call <3 x i8> @air.add_sat.u.v3i8(<3 x i8> %a, <3 x i8> %b)
  %diff = call <3 x i8> @air.sub_sat.u.v3i8(<3 x i8> %sum, <3 x i8> %b)
  ret <3 x i8> %diff
}

declare <3 x i8> @air.add_sat.u.v3i8(<3 x i8>, <3 x i8>)
declare <3 x i8> @air.sub_sat.u.v3i8(<3 x i8>, <3 x i8>)
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpIAdd"), "{asm}");
    assert!(asm.contains("OpISub"), "{asm}");
    assert!(asm.contains("OpULessThan"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
}

#[test]
fn native_air_unsigned_i16_rhadd_widens_before_rounding() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i16 @rhadd(i16 %a, i16 %b) {
entry:
  %avg = call i16 @air.rhadd.u.i16(i16 %a, i16 %b)
  ret i16 %avg
}

declare i16 @air.rhadd.u.i16(i16, i16)
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(asm.contains("OpIAdd"), "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("air.rhadd.u.i16"), "{asm}");
}

#[test]
fn native_air_pack_unorm4x8_f16_converts_to_float_before_extinst() {
    let ll = r#"
source_filename = "pack_unorm_f16"
target datalayout = "e-p:64:64"
target triple = "air64-apple-macosx14.0.0"

define void @main(ptr addrspace(1) %out) {
entry:
  %packed = tail call i32 @air.pack.unorm4x8.v4f16(<4 x half> <half 0xH0000, half 0xH3800, half 0xH3C00, half 0xH3400>)
  store i32 %packed, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.pack.unorm4x8.v4f16(<4 x half>)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_pack_unorm4x8_f16_{}",
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
    assert!(asm.contains("OpFConvert"), "{asm}");
    assert!(asm.contains("PackUnorm4x8"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_convert_bfloat_widens_and_narrows_through_f32() {
    // bf16 has no SPIR-V type, so a bfloat value is modeled as its `OpTypeInt 16` bit pattern. An
    // `air.convert` whose source or dest is bf16 must widen the bits to f32 / narrow f32 to bits
    // around the float conversion, otherwise an OpConvertFToS/OpFConvert is fed an integer and
    // spirv-val rejects it with "expected float input". This exercises bf16->sint, bf16->f32, and
    // f32->bf16 in one kernel.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %in, ptr addrspace(1) %outi, ptr addrspace(1) %outbf) {
entry:
  %bf = load bfloat, ptr addrspace(1) %in, align 2
  %asi = tail call i32 @air.convert.s.i32.f.bf16(bfloat %bf)
  store i32 %asi, ptr addrspace(1) %outi, align 4
  %asf = tail call float @air.convert.f.f32.f.bf16(bfloat %bf)
  %back = tail call bfloat @air.convert.f.bf16.f.f32(float %asf)
  store bfloat %back, ptr addrspace(1) %outbf, align 2
  ret void
}

declare i32 @air.convert.s.i32.f.bf16(bfloat)
declare float @air.convert.f.f32.f.bf16(bfloat)
declare bfloat @air.convert.f.bf16.f.f32(float)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"bfloat*", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int*", !"air.arg_name", !"outi"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"bfloat*", !"air.arg_name", !"outbf"}
"#;
    let tmp =
        std::env::temp_dir().join(format!("metal2vulkan_convert_bf16_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    // bf16->f32 widen (shift left 16 + bitcast to float) and f32->bf16 narrow (shift right 16).
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    // The float->sint convert now sees a real float input.
    assert!(asm.contains("OpConvertFToS"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_bfloat_vector_arithmetic_rounds_through_float() {
    // bf16 has no SPIR-V float type — a `<N x bfloat>` value is modeled as its `Vector(Int(16), N)` bit
    // pattern. A vector bf16 arithmetic op (`fadd <4 x bfloat>`) must widen each lane to f32, do the op
    // in `Vector(Float, N)`, then re-narrow to bf16 bits. The scalar-only bf16 guard used to fall
    // through for a vector, emitting a type-invalid `OpFAdd` on the u16-vector storage type (spirv-val:
    // "Expected floating scalar or vector type as Result Type: FAdd").
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out) {
entry:
  %va = load <4 x bfloat>, ptr addrspace(1) %a, align 8
  %vb = load <4 x bfloat>, ptr addrspace(1) %b, align 8
  %sum = fadd fast <4 x bfloat> %va, %vb
  store <4 x bfloat> %sum, ptr addrspace(1) %out, align 8
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"bfloat4*", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"bfloat4*", !"air.arg_name", !"b"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"bfloat4*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!("metal2vulkan_bf16_vec_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    // Widen (shift-left-16 + bitcast to float) and narrow (shift-right-16 + uconvert) both present — the
    // bf16 vector round-trip through f32. If the scalar-only guard had fallen through, the add would be a
    // direct `OpFAdd` on the u16-vector storage with no surrounding shifts.
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    // spirv-val is the authoritative check: it rejects an `OpFAdd` whose result type is an integer
    // vector ("Expected floating scalar or vector type"), which is exactly the bug this guards.
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_integer_casts_lower_to_spirv_conversions() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i32 @casts(i32 %i) {
entry:
  %wide = zext i32 %i to i64
  %signed = sext i32 %i to i64
  %narrow = trunc i64 %signed to i32
  ret i32 %narrow
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(asm.contains("OpSConvert"), "{asm}");
}

#[test]
fn native_vector_integer_casts_lower_to_spirv_conversions() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define <2 x i16> @vector_casts(<2 x i16> %v) {
entry:
  %wide = zext <2 x i16> %v to <2 x i32>
  %signed = sext <2 x i16> %v to <2 x i32>
  %narrow = trunc <2 x i32> %wide to <2 x i16>
  ret <2 x i16> %narrow
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(asm.contains("OpSConvert"), "{asm}");
}

#[test]
fn native_air_same_kind_integer_convert_lowers_to_spirv_conversion() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %dst_u, ptr addrspace(1) %dst_s, ptr addrspace(1) %src) {
entry:
  %v = load <2 x i16>, ptr addrspace(1) %src, align 4
  %wide_u = tail call <2 x i32> @air.convert.u.v2i32.u.v2i16(<2 x i16> %v)
  %wide_s = tail call <2 x i32> @air.convert.s.v2i32.s.v2i16(<2 x i16> %v)
  store <2 x i32> %wide_u, ptr addrspace(1) %dst_u, align 8
  store <2 x i32> %wide_s, ptr addrspace(1) %dst_s, align 8
  ret void
}

declare <2 x i32> @air.convert.u.v2i32.u.v2i16(<2 x i16>)
declare <2 x i32> @air.convert.s.v2i32.s.v2i16(<2 x i16>)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint2", !"air.arg_name", !"dst_u"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"int2", !"air.arg_name", !"dst_s"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"ushort2", !"air.arg_name", !"src"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_same_kind_integer_convert_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(asm.contains("OpSConvert"), "{asm}");
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
fn native_air_mixed_sign_integer_convert_width_change_uses_spirv_conversion() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %dst_u16, ptr addrspace(1) %dst_s16, ptr addrspace(1) %src) {
entry:
  %v = load <2 x i32>, ptr addrspace(1) %src, align 8
  %narrow_u = tail call <2 x i16> @air.convert.u.v2i16.s.v2i32(<2 x i32> %v)
  %narrow_s = tail call <2 x i16> @air.convert.s.v2i16.u.v2i32(<2 x i32> %v)
  store <2 x i16> %narrow_u, ptr addrspace(1) %dst_u16, align 4
  store <2 x i16> %narrow_s, ptr addrspace(1) %dst_s16, align 4
  ret void
}

declare <2 x i16> @air.convert.u.v2i16.s.v2i32(<2 x i32>)
declare <2 x i16> @air.convert.s.v2i16.u.v2i32(<2 x i32>)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"ushort2", !"air.arg_name", !"dst_u16"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"short2", !"air.arg_name", !"dst_s16"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"int2", !"air.arg_name", !"src"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_mixed_sign_integer_convert_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpSConvert"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(
        !asm.lines()
            .any(|line| line.contains(" OpBitcast ") && line.contains("v2ushort")),
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
fn native_ptrtoint_lowers_to_zero_integer_placeholder() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i64 @pointer_address(ptr addrspace(1) %p) {
entry:
  %addr = ptrtoint ptr addrspace(1) %p to i64
  %out = add i64 %addr, 64
  ret i64 %out
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpIAdd"), "{asm}");
    assert!(!asm.contains("OpConvertPtrToU"), "{asm}");
    assert!(!asm.contains("OpBitcast"), "{asm}");
}

#[test]
fn native_i1_uses_bool_type_for_logic_and_extension() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i32 @bool_to_int(i32 %x) {
entry:
  %cmp = icmp ne i32 %x, 0
  %not = xor i1 %cmp, true
  %out = zext i1 %not to i32
  ret i32 %out
}

define i1 @int8_to_bool(i8 %x) {
entry:
  %out = trunc i8 %x to i1
  ret i1 %out
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpTypeBool"), "{asm}");
    assert!(!asm.contains("OpTypeInt 1 "), "{asm}");
    assert!(asm.contains("OpLogicalNotEqual"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(asm.contains("OpINotEqual"), "{asm}");
}

#[test]
fn native_float_half_conversions_lower_to_fconvert() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define <2 x float> @half_roundtrip(float %x, <2 x float> %v) {
entry:
  %h = fptrunc float %x to half
  %f = fpext half %h to float
  %vh = fptrunc <2 x float> %v to <2 x half>
  %vf = fpext <2 x half> %vh to <2 x float>
  %out = insertelement <2 x float> %vf, float %f, i64 0
  ret <2 x float> %out
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpTypeFloat 16"), "{asm}");
    assert_eq!(asm.matches("OpFConvert").count(), 4, "{asm}");
}

#[test]
fn native_integer_to_float_casts_lower_to_spirv_conversions() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define <2 x float> @casts(i32 %x, <2 x i32> %v) {
entry:
  %signed = sitofp i32 %x to float
  %unsigned = uitofp i32 %x to float
  %vec = sitofp <2 x i32> %v to <2 x float>
  %out0 = insertelement <2 x float> %vec, float %signed, i64 0
  %out1 = insertelement <2 x float> %out0, float %unsigned, i64 1
  ret <2 x float> %out1
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpConvertSToF"), "{asm}");
    assert!(asm.contains("OpConvertUToF"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
}

#[test]
fn native_bfloat_load_store_uses_u16_storage_and_bit_shifts() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
entry:
  %b = load bfloat, ptr addrspace(1) %src, align 2
  %f = fpext bfloat %b to float
  %sum = fadd float %f, 1.000000e+00
  %out = fptrunc float %sum to bfloat
  store bfloat %out, ptr addrspace(1) %dst, align 2
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"bfloat", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"bfloat", !"air.arg_name", !"dst"}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability Int8"), "{asm}");
    assert!(asm.contains("OpCapability Int16"), "{asm}");
    assert!(asm.contains("OpTypeInt 16 0"), "{asm}");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(asm.contains("OpBitcast"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(!asm.contains("OpTypeFloat 16"), "{asm}");
    assert!(!asm.contains("OpFConvert"), "{asm}");
}

#[test]
fn native_bfloat_hex_literal_stores_u16_bits() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %dst) {
entry:
  store bfloat 0xR0000, ptr addrspace(1) %dst, align 2
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"bfloat", !"air.arg_name", !"dst"}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpTypeInt 16 0"), "{asm}");
    assert_eq!(asm.matches("OpTypeInt 16 0").count(), 1, "{asm}");
    assert!(!asm.contains("OpTypeFloat 16"), "{asm}");
}

#[test]
fn native_llvm_bfloat_fmuladd_lowers_through_f32() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main() {
entry:
  %out = tail call bfloat @llvm.fmuladd.bf16(bfloat 0xR3f80, bfloat 0xR4000, bfloat 0xR4040)
  ret void
}

declare bfloat @llvm.fmuladd.bf16(bfloat, bfloat, bfloat)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_bfloat_fma_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    let transformed = load_bytes(&out).expect("load transformed spv");
    let fma_insts = transformed
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| {
            inst.class.opcode == Op::ExtInst
                && inst.operands.get(1) == Some(&Operand::LiteralExtInstInteger(50))
        })
        .collect::<Vec<_>>();
    assert_eq!(fma_insts.len(), 1, "{asm}");
    let fma_result_type = fma_insts[0].result_type.expect("fma result type");
    let fma_type = transformed
        .types_global_values
        .iter()
        .find(|inst| inst.result_id == Some(fma_result_type))
        .expect("fma result type definition");
    assert_eq!(fma_type.class.opcode, Op::TypeFloat, "{asm}");
    assert_eq!(
        fma_type.operands.first(),
        Some(&Operand::LiteralBit32(32)),
        "{asm}"
    );
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(asm.contains("OpBitcast"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_bfloat_arithmetic_lowers_through_f32() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main() {
entry:
  %add = fadd bfloat 0xR3f80, 0xR4000
  %mul = fmul bfloat %add, 0xR4040
  %sub = fsub bfloat %mul, 0xR3f80
  %div = fdiv bfloat %sub, 0xR4000
  %neg = fneg bfloat %div
  ret void
}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_bfloat_arith_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble transformed");
    let transformed = load_bytes(&out).expect("load transformed spv");
    let float_ty = transformed
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeFloat
                && inst.operands.first() == Some(&Operand::LiteralBit32(32))
        })
        .and_then(|inst| inst.result_id)
        .expect("float type");
    let ushort_ty = transformed
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeInt
                && inst.operands.first() == Some(&Operand::LiteralBit32(16))
                && inst.operands.get(1) == Some(&Operand::LiteralBit32(0))
        })
        .and_then(|inst| inst.result_id)
        .expect("ushort type");
    for op in [Op::FAdd, Op::FMul, Op::FSub, Op::FDiv, Op::FNegate] {
        let insts = transformed
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .filter(|inst| inst.class.opcode == op)
            .collect::<Vec<_>>();
        assert_eq!(insts.len(), 1, "{op:?}\n{asm}");
        assert_eq!(insts[0].result_type, Some(float_ty), "{op:?}\n{asm}");
        assert_ne!(insts[0].result_type, Some(ushort_ty), "{op:?}\n{asm}");
    }
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_vector_store_into_byte_remodeled_union_uses_byte_stores() {
    // A `store <4 x i32>` through the union view of an alloca the multi-view remodel backed with a
    // `[16 x i8]` byte array: the raw store must decompose to byte stores (a word-shaped `[0][k]`
    // chain is type-invalid against the byte-array variable).
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%union.V = type { <4 x i32> }
define void @k(ptr addrspace(1) %out, i32 %idx) {
entry:
  %u = alloca %union.V, align 16
  %v = load <4 x i32>, ptr addrspace(1) %out, align 16
  %m = getelementptr inbounds %union.V, ptr %u, i64 0, i32 0
  store <4 x i32> %v, ptr %m, align 16
  %wide = zext i32 %idx to i64
  %b = getelementptr inbounds [16 x i8], ptr %u, i64 0, i64 %wide
  %byte = load i8, ptr %b, align 1
  %w = zext i8 %byte to i32
  store i32 %w, ptr addrspace(1) %out, align 4
  ret void
}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_union_byte_store_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        let out = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_scalar_narrowing_load_reads_low_bits_of_wide_slot() {
    // An i64 union slot is reinterpret-loaded as a 32-bit float: the result is the LOW 32 bits at the
    // slot's address (little-endian), so the emitter loads the i64, UConvert-truncates to i32, and
    // bitcasts to float — a narrowing scalar reinterpret confined WITHIN the slot (no sibling read).
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%union.U = type { i64 }

define float @i64_slot_as_float() {
entry:
  %scratch = alloca %union.U, align 8
  %slot = getelementptr inbounds %union.U, ptr %scratch, i64 0, i32 0
  store i64 4607182418800017408, ptr %slot, align 8
  %v = load float, ptr %slot, align 4
  ret float %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(asm.contains("OpBitcast"), "{asm}");
    assert!(
        !asm.contains("reinterpret load bit width mismatch"),
        "{asm}"
    );
}

#[test]
fn native_scalar_narrowing_store_preserves_high_bits_of_wide_slot() {
    // A 32-bit float is reinterpret-stored into an i64 union slot: only the low 32 bits (the bytes at
    // the slot's address) may change, so the emitter read-modify-writes — load the i64, clear its low
    // 32 bits (>> 32 then << 32), OR in the zero-extended float bits, store back.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%union.U = type { i64 }

define void @float_into_i64_slot(float %f) {
entry:
  %scratch = alloca %union.U, align 8
  %slot = getelementptr inbounds %union.U, ptr %scratch, i64 0, i32 0
  store float %f, ptr %slot, align 4
  ret void
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpBitwiseOr"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(!asm.contains("does not match Object"), "{asm}");
}

#[test]
fn native_fast_float_arithmetic_and_store_lower() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.S = type { float }
define void @fast_store(ptr addrspace(1) %p, float %x) {
entry:
  %g = getelementptr inbounds %struct.S, ptr addrspace(1) %p, i64 0, i32 0
  %mul = fmul fast float %x, 4.000000e+00
  store float %mul, ptr addrspace(1) %g
  ret void
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpFMul"), "{asm}");
    assert!(asm.contains("OpStore"), "{asm}");
    assert!(asm.contains("4"), "{asm}");
}

#[test]
fn native_vector_float_arithmetic_materializes_literal_operands() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define <3 x float> @vector_float_ops(<3 x float> %x) {
entry:
  %mul = fmul fast <3 x float> %x, <float 0x3FE99999A0000000, float poison, float poison>
  %add = fadd fast <3 x float> %mul, <float 0x3FC99999A0000000, float poison, float poison>
  %sub = fsub fast <3 x float> %add, <float 1.000000e+00, float poison, float poison>
  %div = fdiv fast <3 x float> %sub, <float 2.000000e+00, float poison, float poison>
  ret <3 x float> %div
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpConstantComposite"), "{asm}");
    assert!(asm.contains("OpFMul"), "{asm}");
    assert!(asm.contains("OpFAdd"), "{asm}");
    assert!(asm.contains("OpFSub"), "{asm}");
    assert!(asm.contains("OpFDiv"), "{asm}");
}

#[test]
fn native_one_lane_vectors_lower_as_scalars() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k() {
entry:
  %shuf = shufflevector <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>, <4 x float> undef, <1 x i32> <i32 3>
  %ins = insertelement <1 x float> poison, float 2.000000e+00, i64 0
  br i1 true, label %left, label %right

left:
  br label %join

right:
  br label %join

join:
  %phi = phi <1 x float> [ %shuf, %left ], [ %ins, %right ]
  %sum = fadd <1 x float> %phi, splat (float 3.000000e+00)
  %out = extractelement <1 x float> %sum, i64 0
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!("metal2vulkan_vector1_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        !asm.lines()
            .any(|line| line.contains("OpTypeVector") && line.contains(" 1")),
        "{asm}"
    );
    assert!(!asm.contains("OpVectorShuffle"), "{asm}");
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
fn native_fneg_lowers_scalar_vector_and_half() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define <2 x float> @neg_vec(float %x, <2 x float> %v) {
entry:
  %sx = fneg fast float %x
  %vx = fneg fast <2 x float> %v
  %out = insertelement <2 x float> %vx, float %sx, i32 0
  ret <2 x float> %out
}

define half @neg_half(half %x) {
entry:
  %n = fneg half %x
  ret half %n
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert_eq!(asm.matches("OpFNegate").count(), 3, "{asm}");
    assert!(asm.contains("OpTypeFloat 16"), "{asm}");
}

#[test]
fn native_record_array_rewrite_uses_interface_element_struct_type() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_atomic.69" = type { i32 }

define void @k(ptr addrspace(1) %histogram, ptr addrspace(2) %index_ptr) {
entry:
  %idx = load i32, ptr addrspace(2) %index_ptr, align 4
  %idx64 = zext i32 %idx to i64
  %record = getelementptr inbounds %"struct.metal::_atomic.69", ptr addrspace(1) %histogram, i64 %idx64
  %field = getelementptr inbounds %"struct.metal::_atomic.69", ptr addrspace(1) %record, i64 0, i32 0
  store i32 0, ptr addrspace(1) %field, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"metal::_atomic", !"air.arg_name", !"histogram"}
!4 = !{i32 0, i32 4, i32 0, !"uint", !"__s"}
!5 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"index"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_record_array_atomic_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_vector_literals_materialize_for_insert_and_call_args() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define float @dotty(float %x) {
entry:
  %vecinit = insertelement <2 x float> <float undef, float 2.000000e+00>, float %x, i64 0
  %d = tail call fast float @air.dot.v2f32(<2 x float> %vecinit, <2 x float> <float 3.000000e+00, float 4.000000e+00>)
  ret float %d
}

declare float @air.dot.v2f32(<2 x float>, <2 x float>)
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpConstantComposite"), "{asm}");
    assert!(asm.contains("OpCompositeInsert"), "{asm}");
    assert!(asm.contains("OpFunctionCall"), "{asm}");
}

#[test]
fn native_typed_hex_float_literals_lower_as_float_constants() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define float @hex_weights(<3 x float> %x) {
entry:
  %d = tail call fast float @air.dot.v3f32(<3 x float> %x, <3 x float> <float 0x3FCB333340000000, float 0x3FE6E48E80000000, float 0x3FB2752540000000>)
  ret float %d
}

declare float @air.dot.v3f32(<3 x float>, <3 x float>)
"#;
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let float_ty = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeFloat && inst.operands == [Operand::LiteralBit32(32)]
        })
        .and_then(|inst| inst.result_id)
        .expect("float32 type");
    let float_constants = module
        .types_global_values
        .iter()
        .filter(|inst| inst.class.opcode == Op::Constant && inst.result_type == Some(float_ty))
        .filter_map(|inst| match inst.operands.first() {
            Some(Operand::LiteralBit32(bits)) => Some(*bits),
            _ => None,
        })
        .collect::<HashSet<_>>();
    for bits in [
        (f64::from_bits(0x3FCB333340000000) as f32).to_bits(),
        (f64::from_bits(0x3FE6E48E80000000) as f32).to_bits(),
        (f64::from_bits(0x3FB2752540000000) as f32).to_bits(),
    ] {
        assert!(float_constants.contains(&bits), "{float_constants:?}");
    }
}

#[test]
fn native_half_hex_literals_lower_as_float16_constants() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define half @half_ops(half %x) {
entry:
  %mul = fmul fast half %x, 0xH4000
  %add = fadd fast half %mul, 0xH3C00
  ret half %add
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpTypeFloat 16"), "{asm}");
    assert!(asm.contains("OpFMul"), "{asm}");
    assert!(asm.contains("OpFAdd"), "{asm}");
}

#[test]
fn native_flagged_integer_ops_and_negative_literals_lower() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i32 @int_ops(i32 %x) {
entry:
  %shift = shl nsw i32 %x, 1
  %add = add nsw i32 %shift, -3
  %sub = sub nsw i32 %add, 2
  ret i32 %sub
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpIAdd"), "{asm}");
    assert!(asm.contains("OpISub"), "{asm}");
}

#[test]
fn native_bitwise_integer_ops_lower() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i32 @bit_ops(i32 %x) {
entry:
  %and = and i32 %x, 255
  %or = or i32 %and, 16
  %xor = xor i32 %or, 1
  %shr = ashr i32 %xor, 1
  ret i32 %shr
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpBitwiseOr"), "{asm}");
    assert!(asm.contains("OpBitwiseXor"), "{asm}");
    assert!(asm.contains("OpShiftRightArithmetic"), "{asm}");
}

#[test]
fn native_popcount_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main() {
entry:
  %count = tail call i32 @air.popcount.i32(i32 305419896)
  ret void
}

declare i32 @air.popcount.i32(i32)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_popcount_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpBitCount"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_popcount_i64_splits_to_i32_bitcounts() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main() {
entry:
  %count = tail call i64 @air.popcount.i64(i64 -1)
  ret void
}

declare i64 @air.popcount.i64(i64)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_popcount_i64_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    let transformed = load_bytes(&out).expect("load transformed spv");
    let uint_ty = transformed
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeInt
                && inst.operands.first() == Some(&Operand::LiteralBit32(32))
                && inst.operands.get(1) == Some(&Operand::LiteralBit32(0))
        })
        .and_then(|inst| inst.result_id)
        .expect("uint type");
    let ulong_ty = transformed
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeInt
                && inst.operands.first() == Some(&Operand::LiteralBit32(64))
                && inst.operands.get(1) == Some(&Operand::LiteralBit32(0))
        })
        .and_then(|inst| inst.result_id)
        .expect("ulong type");
    let bitcounts = transformed
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| inst.class.opcode == Op::BitCount)
        .collect::<Vec<_>>();
    assert_eq!(bitcounts.len(), 2, "{asm}");
    for inst in bitcounts {
        assert_eq!(inst.result_type, Some(uint_ty), "{asm}");
        assert_ne!(inst.result_type, Some(ulong_ty), "{asm}");
    }
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(asm.contains("OpIAdd"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_popcount_v2i64_splits_to_v2i32_bitcounts() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main() {
entry:
  %v0 = insertelement <2 x i64> poison, i64 -1, i32 0
  %v1 = insertelement <2 x i64> %v0, i64 4294967296, i32 1
  %count = tail call <2 x i64> @air.popcount.v2i64(<2 x i64> %v1)
  ret void
}

declare <2 x i64> @air.popcount.v2i64(<2 x i64>)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_popcount_v2i64_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    let transformed = load_bytes(&out).expect("load transformed spv");
    let uint_ty = transformed
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeInt
                && inst.operands.first() == Some(&Operand::LiteralBit32(32))
                && inst.operands.get(1) == Some(&Operand::LiteralBit32(0))
        })
        .and_then(|inst| inst.result_id)
        .expect("uint type");
    let ulong_ty = transformed
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeInt
                && inst.operands.first() == Some(&Operand::LiteralBit32(64))
                && inst.operands.get(1) == Some(&Operand::LiteralBit32(0))
        })
        .and_then(|inst| inst.result_id)
        .expect("ulong type");
    let v2uint_ty = transformed
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeVector
                && inst.operands.first() == Some(&Operand::IdRef(uint_ty))
                && inst.operands.get(1) == Some(&Operand::LiteralBit32(2))
        })
        .and_then(|inst| inst.result_id)
        .expect("v2uint type");
    let v2ulong_ty = transformed
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeVector
                && inst.operands.first() == Some(&Operand::IdRef(ulong_ty))
                && inst.operands.get(1) == Some(&Operand::LiteralBit32(2))
        })
        .and_then(|inst| inst.result_id)
        .expect("v2ulong type");
    let bitcounts = transformed
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| inst.class.opcode == Op::BitCount)
        .collect::<Vec<_>>();
    assert_eq!(bitcounts.len(), 2, "{asm}");
    for inst in bitcounts {
        assert_eq!(inst.result_type, Some(v2uint_ty), "{asm}");
        assert_ne!(inst.result_type, Some(v2ulong_ty), "{asm}");
    }
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(asm.contains("OpIAdd"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fast_exp2_log2_vector_lower() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main() {
entry:
  %v0 = insertelement <3 x float> poison, float 1.250000e-01, i32 0
  %v1 = insertelement <3 x float> %v0, float 2.500000e-01, i32 1
  %v2 = insertelement <3 x float> %v1, float 5.000000e-01, i32 2
  %exp = tail call fast <3 x float> @air.fast_exp2.v3f32(<3 x float> %v2)
  %log = tail call fast <3 x float> @air.fast_log2.v3f32(<3 x float> %exp)
  %lane = extractelement <3 x float> %log, i32 0
  %sink = fcmp oge float %lane, 0.000000e+00
  ret void
}

declare <3 x float> @air.fast_exp2.v3f32(<3 x float>)
declare <3 x float> @air.fast_log2.v3f32(<3 x float>)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fast_exp2_log2_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains(" Exp2 "), "{asm}");
    assert!(asm.contains(" Log2 "), "{asm}");
    assert!(!asm.contains(" Exp "), "{asm}");
    assert!(!asm.contains(" Log "), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fast_inverse_tanh_log10_powr_vector_lower() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main() {
entry:
  %v0 = insertelement <4 x float> poison, float 1.250000e-01, i32 0
  %v1 = insertelement <4 x float> %v0, float 2.500000e-01, i32 1
  %v2 = insertelement <4 x float> %v1, float 5.000000e-01, i32 2
  %v3 = insertelement <4 x float> %v2, float 7.500000e-01, i32 3
  %asin = tail call fast <4 x float> @air.fast_asin.v4f32(<4 x float> %v3)
  %acos = tail call fast <4 x float> @air.fast_acos.v4f32(<4 x float> %v3)
  %sinh = tail call fast <4 x float> @air.fast_sinh.v4f32(<4 x float> %v3)
  %cosh = tail call fast <4 x float> @air.fast_cosh.v4f32(<4 x float> %v3)
  %tanh = tail call fast <4 x float> @air.fast_tanh.v4f32(<4 x float> %v3)
  %asinh = tail call fast <4 x float> @air.asinh.v4f32(<4 x float> %v3)
  %acosh = tail call fast <4 x float> @air.acosh.v4f32(<4 x float> %v3)
  %atanh = tail call fast <4 x float> @air.atanh.v4f32(<4 x float> %v3)
  %log10 = tail call fast <4 x float> @air.fast_log10.v4f32(<4 x float> %v3)
  %powr = tail call fast <4 x float> @air.fast_powr.v4f32(<4 x float> %v3, <4 x float> %v3)
  %a = fadd fast <4 x float> %asin, %acos
  %b = fadd fast <4 x float> %a, %sinh
  %c = fadd fast <4 x float> %b, %cosh
  %d = fadd fast <4 x float> %c, %tanh
  %e = fadd fast <4 x float> %d, %asinh
  %f = fadd fast <4 x float> %e, %acosh
  %g = fadd fast <4 x float> %f, %atanh
  %h = fadd fast <4 x float> %g, %log10
  %i = fadd fast <4 x float> %h, %powr
  %lane = extractelement <4 x float> %i, i32 0
  %sink = fcmp oge float %lane, 0.000000e+00
  ret void
}

declare <4 x float> @air.fast_asin.v4f32(<4 x float>)
declare <4 x float> @air.fast_acos.v4f32(<4 x float>)
declare <4 x float> @air.fast_sinh.v4f32(<4 x float>)
declare <4 x float> @air.fast_cosh.v4f32(<4 x float>)
declare <4 x float> @air.fast_tanh.v4f32(<4 x float>)
declare <4 x float> @air.asinh.v4f32(<4 x float>)
declare <4 x float> @air.acosh.v4f32(<4 x float>)
declare <4 x float> @air.atanh.v4f32(<4 x float>)
declare <4 x float> @air.fast_log10.v4f32(<4 x float>)
declare <4 x float> @air.fast_powr.v4f32(<4 x float>, <4 x float>)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fast_inverse_tanh_log10_powr_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains(" Asin "), "{asm}");
    assert!(asm.contains(" Acos "), "{asm}");
    assert!(asm.contains(" Sinh "), "{asm}");
    assert!(asm.contains(" Cosh "), "{asm}");
    // air.fast_tanh lowers to Metal's exp2-based formula (overflow-to-NaN faithful), not GLSL Tanh.
    assert!(!asm.contains(" Tanh "), "{asm}");
    assert!(asm.contains(" Exp2 "), "{asm}");
    assert!(asm.contains(" Asinh "), "{asm}");
    assert!(asm.contains(" Acosh "), "{asm}");
    assert!(asm.contains(" Atanh "), "{asm}");
    assert!(asm.contains(" Log "), "{asm}");
    assert!(asm.contains(" Pow "), "{asm}");
    assert!(asm.contains("OpFDiv"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_half_pow_uses_abs_base() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main() {
entry:
  %v0 = insertelement <3 x half> poison, half 0xHBC00, i32 0
  %v1 = insertelement <3 x half> %v0, half 0xH4000, i32 1
  %v2 = insertelement <3 x half> %v1, half 0xHC000, i32 2
  %e0 = insertelement <3 x half> poison, half 0xH4066, i32 0
  %e1 = shufflevector <3 x half> %e0, <3 x half> poison, <3 x i32> zeroinitializer
  %pow = tail call fast <3 x half> @air.pow.v3f16(<3 x half> %v2, <3 x half> %e1)
  %lane = extractelement <3 x half> %pow, i32 0
  %sink = fcmp oge half %lane, 0xH0000
  ret void
}

declare <3 x half> @air.pow.v3f16(<3 x half>, <3 x half>)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_half_pow_abs_base_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains(" FAbs "), "{asm}");
    assert!(asm.contains(" Pow "), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fast_round_vector_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main() {
entry:
  %v0 = insertelement <4 x float> poison, float 1.250000e+00, i32 0
  %v1 = insertelement <4 x float> %v0, float -1.750000e+00, i32 1
  %v2 = insertelement <4 x float> %v1, float 2.500000e+00, i32 2
  %v3 = insertelement <4 x float> %v2, float -2.500000e+00, i32 3
  %rounded = tail call fast <4 x float> @air.fast_round.v4f32(<4 x float> %v3)
  %lane = extractelement <4 x float> %rounded, i32 0
  %sink = fcmp oge float %lane, 0.000000e+00
  ret void
}

declare <4 x float> @air.fast_round.v4f32(<4 x float>)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fast_round_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains(" Round "), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_fast_rint_vector_lowers_to_round_even() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main() {
entry:
  %v0 = insertelement <4 x float> poison, float 1.250000e+00, i32 0
  %v1 = insertelement <4 x float> %v0, float -1.750000e+00, i32 1
  %v2 = insertelement <4 x float> %v1, float 2.500000e+00, i32 2
  %v3 = insertelement <4 x float> %v2, float -2.500000e+00, i32 3
  %rounded = tail call fast <4 x float> @air.fast_rint.v4f32(<4 x float> %v3)
  %lane = extractelement <4 x float> %rounded, i32 0
  %sink = fcmp oge float %lane, 0.000000e+00
  ret void
}

declare <4 x float> @air.fast_rint.v4f32(<4 x float>)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fast_rint_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains(" RoundEven "), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_round_half_vector_round_trips_through_float() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main(<2 x half> %v) {
entry:
  %rounded = tail call fast <2 x half> @air.round.v2f16(<2 x half> %v)
  %lane = extractelement <2 x half> %rounded, i32 0
  %sink = fcmp oge half %lane, 0xH0000
  ret void
}

declare <2 x half> @air.round.v2f16(<2 x half>)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_round_half_vector_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpFConvert"), "{asm}");
    assert!(asm.contains(" Round "), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_integer_division_and_remainder_ops_lower() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i32 @div_rem_ops(i32 %x, i32 %y) {
entry:
  %sd = sdiv i32 %x, 3
  %sr = srem i32 %x, 5
  %ud = udiv i32 %y, 7
  %ur = urem i32 %y, 11
  %a = add i32 %sd, %sr
  %b = add i32 %ud, %ur
  %out = add i32 %a, %b
  ret i32 %out
}

define <2 x i32> @vec_sdiv(<2 x i32> %x) {
entry:
  %d = sdiv <2 x i32> %x, <i32 3, i32 3>
  ret <2 x i32> %d
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert_eq!(asm.matches("OpSDiv").count(), 3, "{asm}");
    assert!(!asm.contains("OpSRem"), "{asm}");
    assert!(asm.contains("OpUDiv"), "{asm}");
    assert!(asm.contains("OpUMod"), "{asm}");
}

#[test]
fn native_integer_remainder_zero_denominators_are_guarded() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main(i32 %tid, ptr addrspace(1) %out) {
entry:
  %zero = sub i32 %tid, %tid
  %ur = urem i32 %tid, %zero
  %sr = srem i32 %tid, %zero
  %sum = add i32 %ur, %sr
  store i32 %sum, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_guard_remainder_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpUMod"), "{asm}");
    assert!(asm.contains("OpSDiv"), "{asm}");
    assert!(!asm.contains("OpSRem"), "{asm}");
    assert!(
        asm.matches("OpIEqual").count() >= 2,
        "expected zero-denominator checks for urem and srem\n{asm}"
    );
    assert!(
        asm.matches("OpSelect").count() >= 2,
        "expected selected safe denominators for urem and srem\n{asm}"
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
fn native_extractvalue_lowers_struct_member_extract() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define <4 x float> @extract_color({ <4 x float>, i8 } %r) {
entry:
  %c = extractvalue { <4 x float>, i8 } %r, 0
  ret <4 x float> %c
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpCompositeExtract"), "{asm}");
}

#[test]
fn native_splat_constants_materialize_as_vectors() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define <2 x float> @splat_ops(<2 x float> %x) {
entry:
  %mul = fmul fast <2 x float> %x, splat (float 5.000000e-01)
  ret <2 x float> %mul
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpConstantComposite"), "{asm}");
    assert!(asm.contains("OpFMul"), "{asm}");
}

#[test]
fn native_vector_fcmp_returns_vector_bool() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i1 @any_lt(<4 x float> %x) {
entry:
  %cmp = fcmp fast olt <4 x float> %x, splat (float 5.000000e-01)
  %any = tail call i1 @air.any.v4i1(<4 x i1> %cmp)
  ret i1 %any
}

declare i1 @air.any.v4i1(<4 x i1>)
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpFOrdLessThan"), "{asm}");
    assert!(asm.contains("OpTypeVector"), "{asm}");
    assert!(asm.contains("OpFunctionCall"), "{asm}");
}

#[test]
fn native_vector_select_accepts_vector_bool_mask() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define <4 x float> @mask(<4 x float> %x, <4 x float> %y) {
entry:
  %cmp = fcmp fast olt <4 x float> %x, %y
  %out = select <4 x i1> %cmp, <4 x float> %x, <4 x float> splat (float 1.000000e+00)
  ret <4 x float> %out
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpFOrdLessThan"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
}

#[test]
fn native_dynamic_vector_extract_insert_lower() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define <4 x float> @lane(<4 x float> %x, <4 x float> %y, i32 %idx) {
entry:
  %v = extractelement <4 x float> %x, i32 %idx
  %out = insertelement <4 x float> %y, float %v, i32 %idx
  ret <4 x float> %out
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpVectorExtractDynamic"), "{asm}");
    assert!(asm.contains("OpVectorInsertDynamic"), "{asm}");
}

#[test]
fn native_integer_comparisons_lower_scalar_and_vector_results() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i1 @scalar_cmp(i8 %x) {
entry:
  %c = icmp ugt i8 %x, 1
  ret i1 %c
}

define <2 x i1> @vector_cmp(<2 x i8> %x) {
entry:
  %c = icmp ne <2 x i8> %x, zeroinitializer
  ret <2 x i1> %c
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpUGreaterThan"), "{asm}");
    assert!(asm.contains("OpINotEqual"), "{asm}");
    assert!(asm.contains("OpTypeBool"), "{asm}");
    assert!(asm.contains("OpTypeVector"), "{asm}");
}

#[test]
fn native_wide_vector_shuffle_uses_composite_construct() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define <6 x i32> @wide_shuffle(<4 x i32> %a, <4 x i32> %b) {
entry:
  %wide_a = shufflevector <4 x i32> %a, <4 x i32> poison, <6 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison>
  %wide_b = shufflevector <4 x i32> %b, <4 x i32> poison, <6 x i32> <i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %out = shufflevector <6 x i32> %wide_a, <6 x i32> %wide_b, <6 x i32> <i32 0, i32 6, i32 1, i32 2, i32 3, i32 poison>
  ret <6 x i32> %out
}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("OpVectorShuffle"), "{asm}");
    assert!(asm.contains("OpCompositeConstruct"), "{asm}");
    assert!(asm.contains("OpCompositeExtract"), "{asm}");
}

#[test]
fn native_select_with_bool_literal_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define <4 x float> @pick(<4 x float> %a, <4 x float> %b) {
entry:
  %r = select i1 true, <4 x float> %a, <4 x float> %b
  ret <4 x float> %r
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpConstantTrue"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
}

#[test]
fn native_parser_accepts_array_of_named_struct_constants() {
    let value = parse_typed_value(
        "[2 x %struct.Point] [%struct.Point { <2 x float> <float 1.0, float 2.0>, <2 x float> <float 3.0, float 4.0> }, %struct.Point { <2 x float> <float 5.0, float 6.0>, <2 x float> <float 7.0, float 8.0> }]",
    )
    .expect("parse array of struct constants");
    assert_eq!(
        value.ty,
        LlType::Array(Box::new(LlType::Named("%struct.Point".into())), 2)
    );
    let LlValue::Array(rows) = value.value else {
        panic!("expected outer array");
    };
    assert_eq!(rows.len(), 2);
    for row in rows {
        assert_eq!(row.ty, LlType::Named("%struct.Point".into()));
        let LlValue::Struct(fields) = row.value else {
            panic!("expected struct row");
        };
        assert_eq!(fields.len(), 2);
        assert!(fields.iter().all(
            |field| matches!(field.ty, LlType::Vector(ref elem, 2) if **elem == LlType::Float)
        ));
    }
}

#[test]
fn native_signed_integer_minmax_uses_signed_extinst_type() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"

define void @k(ptr addrspace(1) %out) {
entry:
  %min = tail call i32 @air.min.s.i32(i32 -122, i32 117)
  %max = tail call i32 @air.max.s.i32(i32 -122, i32 117)
  store i32 %min, ptr addrspace(1) %out, align 4
  %next = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %max, ptr addrspace(1) %next, align 4
  ret void
}

declare i32 @air.min.s.i32(i32, i32)
declare i32 @air.max.s.i32(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_signed_integer_minmax_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load spv");
    let signed_ints = module
        .types_global_values
        .iter()
        .filter_map(|inst| {
            (inst.class.opcode == Op::TypeInt
                && inst.operands.first() == Some(&Operand::LiteralBit32(32))
                && inst.operands.get(1) == Some(&Operand::LiteralBit32(1)))
            .then_some(inst.result_id)
            .flatten()
        })
        .collect::<HashSet<_>>();
    for glsl_op in [39, 42] {
        let inst = module
            .all_inst_iter()
            .find(|inst| {
                inst.class.opcode == Op::ExtInst
                    && inst.operands.get(1) == Some(&Operand::LiteralExtInstInteger(glsl_op))
            })
            .unwrap_or_else(|| panic!("missing signed min/max op {glsl_op}\n{asm}"));
        assert!(
            inst.result_type.is_some_and(|ty| signed_ints.contains(&ty)),
            "signed min/max op {glsl_op} used unsigned result type\n{asm}"
        );
    }
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}
