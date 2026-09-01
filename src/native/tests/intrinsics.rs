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
fn native_llvm_umax_lowers_to_compare_select() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define i64 @main(i64 %a, i64 %b) {
entry:
  %m = tail call i64 @llvm.umax.i64(i64 %a, i64 %b)
  ret i64 %m
}

declare i64 @llvm.umax.i64(i64, i64)
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpUGreaterThan"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("llvm.umax"), "{asm}");
}

#[test]
fn native_llvm_usub_sat_lowers_to_compare_select() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define i32 @sat32(i32 %a, i32 %b) {
entry:
  %m = call i32 @llvm.usub.sat.i32(i32 %a, i32 %b)
  ret i32 %m
}

define i64 @sat64(i64 %a, i64 %b) {
entry:
  %m = call i64 @llvm.usub.sat.i64(i64 %a, i64 %b)
  ret i64 %m
}

declare i32 @llvm.usub.sat.i32(i32, i32)
declare i64 @llvm.usub.sat.i64(i64, i64)
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpULessThan"), "{asm}");
    assert!(asm.contains("OpISub"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("llvm.usub.sat"), "{asm}");
}

#[test]
fn native_parse_type_skips_leading_fast_math_flags() {
    assert_eq!(
        parse_type("nnan ninf nsz arcp afn <2 x half>").expect("parse flagged vector type"),
        LlType::Vector(Box::new(LlType::Half), 2)
    );
    assert_eq!(
        parse_type("volatile <4 x half>").expect("parse volatile vector type"),
        LlType::Vector(Box::new(LlType::Half), 4)
    );
}

#[test]
fn native_fast_multiply_add_contracts_without_changing_precise_arithmetic() {
    let ll = r#"
define float @long_fast_multiply_add(float %a, float %b, float %c) {
entry:
  %p0 = fmul fast float %a, %b
  %p1 = fmul fast float %a, %b
  %p2 = fmul fast float %a, %b
  %p3 = fmul fast float %a, %b
  %p4 = fmul fast float %a, %b
  %p5 = fmul fast float %a, %b
  %p6 = fmul fast float %a, %b
  %p7 = fmul fast float %a, %b
  %p8 = fmul fast float %a, %b
  %p9 = fmul fast float %a, %b
  %s0 = fadd fast float %c, %p0
  %s1 = fadd fast float %s0, %p1
  %s2 = fadd fast float %s1, %p2
  %s3 = fadd fast float %s2, %p3
  %s4 = fadd fast float %s3, %p4
  %s5 = fadd fast float %s4, %p5
  %s6 = fadd fast float %s5, %p6
  %s7 = fadd fast float %s6, %p7
  %s8 = fadd fast float %s7, %p8
  %s9 = fadd fast float %s8, %p9
  ret float %s9
}

define float @precise_multiply_add(float %a, float %b, float %c) {
entry:
  %product = fmul float %a, %b
  %sum = fadd float %product, %c
  ret float %sum
}

define float @shared_fast_product(float %a, float %b, float %c) {
entry:
  %product = fmul fast float %a, %b
  %sum = fadd fast float %product, %c
  %both = fadd fast float %sum, %product
  ret float %both
}

define float @product_rooted_sum(float %a, float %b, float %c, float %d) {
entry:
  %left = fmul fast float %a, %b
  %right = fmul fast float %c, %d
  %root = fadd fast float %left, %right
  %tail = fmul fast float %a, %d
  %sum = fadd fast float %root, %tail
  ret float %sum
}
"#;
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load spv");
    let fmas = module
        .all_inst_iter()
        .filter(|instruction| {
            instruction.class.opcode == Op::ExtInst
                && matches!(
                    instruction.operands.get(1),
                    Some(Operand::LiteralExtInstInteger(number))
                        if *number == spirv::GlslStd450Op::Fma as u32
                )
        })
        .count();
    assert_eq!(fmas, 10);
    assert_eq!(
        module
            .all_inst_iter()
            .filter(|instruction| instruction.class.opcode == Op::FAdd)
            .count(),
        5
    );
}

#[test]
fn native_short_fast_signed_sum_materializes_structural_groups() {
    let ll = r#"
define float @signed_sum(float %a, float %b, float %c, float %d, float %acc) {
entry:
  %positive0 = fmul fast float %a, 1.000000e+00
  %positive1 = fmul fast float %b, 2.000000e+00
  %negative0 = fmul fast float %c, -1.000000e+00
  %negative1 = fmul fast float %d, -2.000000e+00
  %sum0 = fadd fast float %positive0, %positive1
  %sum1 = fadd fast float %sum0, %acc
  %sum2 = fadd fast float %sum1, %negative0
  %sum3 = fadd fast float %sum2, %negative1
  ret float %sum3
}
"#;
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load spv");
    assert_eq!(
        module
            .all_inst_iter()
            .filter(|instruction| instruction.class.opcode == Op::BitReverse)
            .count(),
        6
    );
}

#[test]
fn native_seven_term_signed_sum_groups_difference_products_by_coefficient_sign() {
    let ll = r#"
define float @signed_sum_with_differences(float %a, float %b, float %acc) {
entry:
  %positive0 = fmul fast float %a, 3.000000e-01
  %negative0 = fmul fast float %a, -8.000000e-01
  %negative1 = fmul fast float %b, -8.000000e-01
  %positive1 = fmul fast float %b, 3.000000e-01
  %difference0 = fsub fast float %a, %b
  %difference_product0 = fmul fast float %difference0, 5.000000e-01
  %difference1 = fsub fast float %b, %a
  %difference_product1 = fmul fast float %difference1, 9.000000e-01
  %sum0 = fadd fast float %acc, %positive0
  %sum1 = fadd fast float %sum0, %negative0
  %sum2 = fadd fast float %sum1, %negative1
  %sum3 = fadd fast float %sum2, %positive1
  %sum4 = fadd fast float %sum3, %difference_product0
  %sum5 = fadd fast float %sum4, %difference_product1
  ret float %sum5
}

define void @stored_signed_sum_with_differences(float %a, float %b, float %acc, ptr %out) {
entry:
  %positive0 = fmul fast float %a, 3.000000e-01
  %negative0 = fmul fast float %a, -8.000000e-01
  %negative1 = fmul fast float %b, -8.000000e-01
  %positive1 = fmul fast float %b, 3.000000e-01
  %difference0 = fsub fast float %a, %b
  %difference_product0 = fmul fast float %difference0, 5.000000e-01
  %difference1 = fsub fast float %b, %a
  %difference_product1 = fmul fast float %difference1, 9.000000e-01
  %sum0 = fadd fast float %acc, %positive0
  %sum1 = fadd fast float %sum0, %negative0
  %sum2 = fadd fast float %sum1, %negative1
  %sum3 = fadd fast float %sum2, %positive1
  %sum4 = fadd fast float %sum3, %difference_product0
  %sum5 = fadd fast float %sum4, %difference_product1
  store float %sum5, ptr %out, align 4
  ret void
}
"#;
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load spv");
    assert_eq!(
        module
            .all_inst_iter()
            .filter(|instruction| instruction.class.opcode == Op::BitReverse)
            .count(),
        4
    );
}

#[test]
fn native_ten_term_fast_sums_preserve_structural_partitions() {
    let ll = r#"
define float @product_seed(float %a, float %b, float %acc) {
entry:
  %p0 = fmul fast float %a, 1.000000e+00
  %p1 = fmul fast float %b, 2.000000e+00
  %n0 = fmul fast float %a, -1.000000e+00
  %n1 = fmul fast float %a, -2.000000e+00
  %n2 = fmul fast float %b, -3.000000e+00
  %n3 = fmul fast float %b, -4.000000e+00
  %difference0 = fsub fast float %a, %b
  %difference_product0 = fmul fast float %difference0, 3.000000e+00
  %difference1 = fsub fast float %b, %a
  %difference_product1 = fmul fast float %difference1, 4.000000e+00
  %left = fmul fast float %a, 5.000000e+00
  %right = fmul fast float %b, 6.000000e+00
  %product_difference = fsub fast float %left, %right
  %s0 = fadd fast float %p0, %p1
  %s1 = fadd fast float %s0, %acc
  %s2 = fadd fast float %s1, %n0
  %s3 = fadd fast float %s2, %n1
  %s4 = fadd fast float %s3, %n2
  %s5 = fadd fast float %s4, %n3
  %s6 = fadd fast float %s5, %difference_product0
  %s7 = fadd fast float %s6, %difference_product1
  %s8 = fadd fast float %s7, %product_difference
  ret float %s8
}

define float @accumulator_seed(float %a, float %b, float %acc) {
entry:
  %n0 = fmul fast float %a, -1.000000e+00
  %n1 = fmul fast float %a, -2.000000e+00
  %p0 = fmul fast float %b, 1.000000e+00
  %p1 = fmul fast float %b, 2.000000e+00
  %n2 = fmul fast float %a, -3.000000e+00
  %difference0 = fsub fast float %a, %b
  %difference_product0 = fmul fast float %difference0, 3.000000e+00
  %n3 = fmul fast float %b, -4.000000e+00
  %difference1 = fsub fast float %b, %a
  %difference_product1 = fmul fast float %difference1, 4.000000e+00
  %difference2 = fsub fast float %a, %b
  %difference_product2 = fmul fast float %difference2, 5.000000e+00
  %s0 = fadd fast float %acc, %n0
  %s1 = fadd fast float %s0, %n1
  %s2 = fadd fast float %s1, %p0
  %s3 = fadd fast float %s2, %p1
  %s4 = fadd fast float %s3, %n2
  %s5 = fadd fast float %s4, %difference_product0
  %s6 = fadd fast float %s5, %n3
  %s7 = fadd fast float %s6, %difference_product1
  %s8 = fadd fast float %s7, %difference_product2
  ret float %s8
}
"#;
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load spv");
    assert_eq!(
        module
            .all_inst_iter()
            .filter(|instruction| instruction.class.opcode == Op::BitReverse)
            .count(),
        40
    );
}

#[test]
fn native_air_is_uniform_lowers_to_subgroup_vote() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i32 %v, ptr addrspace(1) %out) {
entry:
  %same = tail call i1 @air.is_uniform.i32(i32 %v)
  %word = select i1 %same, i32 1, i32 0
  store i32 %word, ptr addrspace(1) %out, align 4
  ret void
}

declare i1 @air.is_uniform.i32(i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_is_uniform_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(asm.contains("OpCapability GroupNonUniformVote"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformAllEqual"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_llvm_assume_is_dropped_after_lowering() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i32 %v, ptr addrspace(1) %out) {
entry:
  %same = icmp eq i32 %v, %v
  tail call void @llvm.assume(i1 %same)
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

declare void @llvm.assume(i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp =
        std::env::temp_dir().join(format!("metal2vulkan_native_assume_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("llvm.assume"), "{asm}");
    assert!(!asm.contains("llvm_assume"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_agx3_yield_scheduling_hint_is_dropped_after_lowering() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(i16 %delay, ptr addrspace(1) %out) {
entry:
  tail call void @llvm.agx3.yield(i16 %delay)
  store i16 %delay, ptr addrspace(1) %out, align 2
  ret void
}

declare void @llvm.agx3.yield(i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"ushort", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_agx3_yield_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("llvm.agx3.yield"), "{asm}");
    assert!(!asm.contains("llvm_agx3_yield"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_pack_unorm4x8_lowers_to_glsl_extinst() {
    let ll = r#"
source_filename = "pack_unorm"
target datalayout = "e-p:64:64"
target triple = "air64-apple-macosx14.0.0"

define void @main(ptr addrspace(1) %out) {
entry:
  %packed = tail call i32 @air.pack.unorm4x8.v4f32(<4 x float> <float 0.000000e+00, float 5.000000e-01, float 1.000000e+00, float 2.500000e-01>)
  store i32 %packed, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.pack.unorm4x8.v4f32(<4 x float>)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_pack_unorm4x8_{}",
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
fn native_air_pack_unorm_rgb565_f16_lowers_to_exact_fields() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(ptr addrspace(1) %out) {
entry:
  %packed = call i16 @air.pack.unorm.rgb565.v3f16(<3 x half> <half 0xH3C00, half 0xH3800, half 0xH0000>)
  store i16 %packed, ptr addrspace(1) %out, align 2
  ret void
}

declare i16 @air.pack.unorm.rgb565.v3f16(<3 x half>)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"ushort*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_pack_unorm_rgb565_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpFConvert"), "{asm}");
    assert_eq!(asm.matches(" FClamp ").count(), 3, "{asm}");
    assert_eq!(asm.matches(" Round ").count(), 3, "{asm}");
    assert_eq!(asm.matches("OpShiftLeftLogical").count(), 2, "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn native_coherent_air_store_and_fence_lower_to_memory_ops() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"

define void @fence_update(ptr addrspace(1) %timestamp, ptr addrspace(2) %value) {
entry:
  %v = load i32, ptr addrspace(2) %value, align 4
  %old = tail call i32 @air.load.system_coherent.volatile.i32.p1i32(ptr addrspace(1) %timestamp)
  %next = add i32 %v, %old
  tail call void @air.store.system_coherent.volatile.i32.p1i32(i32 %next, ptr addrspace(1) %timestamp)
  tail call void @air.atomic.fence(i32 1, i32 5, i32 3)
  ret void
}

declare i32 @air.load.system_coherent.volatile.i32.p1i32(ptr addrspace(1))
declare void @air.store.system_coherent.volatile.i32.p1i32(i32, ptr addrspace(1))
declare void @air.atomic.fence(i32, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @fence_update, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"timestamp"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"value"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_coherent_air_store_fence_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpStore"), "{asm}");
    assert!(asm.contains("OpLoad"), "{asm}");
    assert!(asm.contains("OpMemoryBarrier"), "{asm}");
    assert!(!asm.contains("air.load.system_coherent"), "{asm}");
    assert!(!asm.contains("air.store.system_coherent"), "{asm}");
    assert!(!asm.contains("air.atomic.fence"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_call_pointee_propagation_is_monotonic_for_conflicting_helpers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @k(ptr addrspace(1) %p) {
entry:
  call void @takes_i32(ptr addrspace(1) %p)
  call void @takes_float(ptr addrspace(1) %p)
  ret void
}

define void @takes_i32(ptr addrspace(1) %q) {
entry:
  %g = getelementptr inbounds i32, ptr addrspace(1) %q, i64 0
  ret void
}

define void @takes_float(ptr addrspace(1) %q) {
entry:
  %g = getelementptr inbounds float, ptr addrspace(1) %q, i64 0
  ret void
}
"#;
    let ir = super::super::ir::LlModule::parse(ll).expect("parse");
    assert_eq!(
        ir.ptr_pointees
            .get(&("takes_i32".to_string(), "%q".to_string())),
        Some(&LlType::Int(32))
    );
    assert_eq!(
        ir.ptr_pointees
            .get(&("takes_float".to_string(), "%q".to_string())),
        Some(&LlType::Float)
    );
    assert!(
        !ir.ptr_pointees
            .contains_key(&("k".to_string(), "%p".to_string())),
        "{:?}",
        ir.ptr_pointees
    );
}

#[test]
fn native_air_declaration_call_is_named_for_lowering() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define float @k(float %x) {
entry:
  %y = tail call fast float @air.fast_ceil.f32(float %x)
  ret float %y
}

declare float @air.fast_ceil.f32(float)
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpName"), "{asm}");
    assert!(asm.contains("air.fast_ceil.f32"), "{asm}");
    assert!(asm.contains("OpFunctionCall"), "{asm}");
}

#[test]
fn native_agx2_cluster_number_uses_padded_local_invocation_coordinates() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"
define void @k(ptr addrspace(1) %out) {
entry:
  %cluster = tail call i32 @llvm.agx2.cluster.num()
  store i32 %cluster, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @llvm.agx2.cluster.num()

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_agx2_cluster_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions {
            kernel_local_size: [10, 8, 1],
            ..passes::TransformOptions::default()
        },
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("BuiltIn LocalInvocationId"), "{asm}");
    assert!(asm.contains("BuiltIn WorkgroupSize"), "{asm}");
    assert_eq!(asm.matches("SpecId").count(), 3, "{asm}");
    assert!(asm.contains("OpCompositeExtract"), "{asm}");
    assert!(asm.contains("OpIMul"), "{asm}");
    assert!(!asm.contains("llvm.agx2.cluster.num"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.ends_with(" 16")),
        "{asm}"
    );
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn native_void_air_call_is_emitted_for_lowering() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @discard() {
entry:
  tail call void @air.discard_fragment()
  ret void
}

declare void @air.discard_fragment()
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("air.discard_fragment"), "{asm}");
    assert!(asm.contains("OpFunctionCall"), "{asm}");
}

#[test]
fn native_visible_function_table_indirect_group_call_is_noop() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %table = inttoptr i64 0 to ptr addrspace(1)
  %size = tail call i32 @air.get_size_visible_function_table(ptr addrspace(1) %table)
  %fp = tail call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 %size)
  %cast = bitcast ptr %fp to ptr
  tail call void %cast(ptr addrspace(2) null) #1, !air.function_groups !0
  ret void
}

declare i32 @air.get_size_visible_function_table(ptr addrspace(1))
declare ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1), i32)

attributes #1 = { convergent nobuiltin nounwind "no-builtins" }
!air.function_groups = !{!0}
!0 = !{!"air.function_group", !"rayGen"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_visible_function_group_{}",
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
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(
        !asm.contains("air.get_size_visible_function_table"),
        "{asm}"
    );
    assert!(
        !asm.contains("air.get_function_pointer_visible_function_table"),
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
fn native_visible_function_table_value_call_reports_indirect_function_pointer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define <4 x float> @main() {
entry:
  %table = inttoptr i64 0 to ptr addrspace(1)
  %fp = tail call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 0)
  %cast = bitcast ptr %fp to ptr
  %result = call fast <4 x float> %cast(ptr addrspace(2) null)
  ret <4 x float> %result
}

declare ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1), i32)
"#;
    let err = emit_vulkan_spirv(ll).expect_err("indirect value call should be unsupported");
    assert!(err.contains("unsupported indirect call"), "{err}");
    assert!(err.contains("function pointer %cast"), "{err}");
    assert!(!err.contains("graph_walk_unmigrated_opcode"), "{err}");
}

#[test]
fn native_visible_function_reference_fails_before_retry_cascade() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  tail call void @postfixPrimary_f.MTL_VISIBLE_FN_REF(ptr addrspace(2) null)
  ret void
}

declare void @postfixPrimary_f.MTL_VISIBLE_FN_REF(ptr addrspace(2)) section "air.externally_defined"
!air.visible_function_references = !{!0}
!0 = !{!"air.visible_function_reference", ptr @postfixPrimary_f.MTL_VISIBLE_FN_REF, !"postfixPrimary_f"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_visible_function_ref_{}",
        std::process::id()
    ));
    let err = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp)
        .expect_err("direct Metal visible function references are unsupported");
    assert!(
        err.contains("unsupported Metal visible function reference"),
        "{err}"
    );
    assert!(err.contains("Logical SPIR-V"), "{err}");
}

/// A post-tessellation vertex function: `air.patch` states the domain and control-point count, and
/// `air.patch_control_point_input` names the per-control-point fetch the evaluation stage calls.
const PATCH_CONTROL_POINT_LL: &str = r#"
target triple = "spirv-unknown-vulkan1.2"
define <{ <4 x float> }> @main(ptr %patch, <2 x float> %position_in_patch) {
entry:
  %cp = tail call { <3 x float> } @control.MTL_CONTROL_POINT_FN(i32 0, ptr %patch)
  %pos = insertvalue <{ <4 x float> }> undef, <4 x float> zeroinitializer, 0
  ret <{ <4 x float> }> %pos
}

declare { <3 x float> } @control.MTL_CONTROL_POINT_FN(i32, ptr) section "air.externally_defined"
!air.vertex = !{!0}
!0 = !{ptr @main, !1, !2, !7}
!1 = !{!3}
!2 = !{!4, !8}
!3 = !{!"air.position", !"air.arg_type_name", !"float4"}
!4 = !{i32 0, !"air.patch_control_point_input", !5, !6}
!5 = !{!"air.patch_control_point_function", ptr @control.MTL_CONTROL_POINT_FN}
!6 = !{!"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float3"}
!7 = !{!"air.patch", !"triangle", !"air.patch_control_point", i32 3}
!8 = !{i32 1, !"air.position_in_patch", !"air.arg_type_name", !"float2"}
"#;

#[test]
fn native_patch_control_point_reference_lowers_to_tessellation_evaluation() {
    let ll = PATCH_CONTROL_POINT_LL;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_patch_control_point_ref_{}",
        std::process::id()
    ));
    let out = crate::translate_sanitized_native(ll, Stage::Vertex, &tmp)
        .expect("patch control points lower through tessellation evaluation");
    let asm = disassemble(&out).expect("disassemble tessellation evaluation module");
    assert!(asm.contains("OpEntryPoint TessellationEvaluation"), "{asm}");
    assert!(
        asm.contains("OpExecutionMode") && asm.contains("Triangles"),
        "{asm}"
    );
    assert!(asm.contains("BuiltIn TessCoord"), "{asm}");
    assert!(!asm.contains("MTL_CONTROL_POINT_FN"), "{asm}");
}

/// The same function with an `air.patch` node naming a domain the translator does not model. Metal
/// would still run it as a post-tessellation evaluation shader; dropping the shape turns it into an
/// ordinary vertex shader that validates, binds and reflects while drawing the wrong geometry.
#[test]
fn native_unreadable_patch_domain_is_refused_rather_than_dropped() {
    let ll = PATCH_CONTROL_POINT_LL.replace("!\"triangle\"", "!\"tetrahedron\"");
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_unreadable_patch_domain_{}",
        std::process::id()
    ));
    let err = crate::translate_sanitized_native(&ll, Stage::Vertex, &tmp)
        .expect_err("a patch domain with no lowering must not become an ordinary vertex shader");
    assert!(err.contains("tessellation patch"), "{err}");
    assert!(err.contains("no tessellation domain"), "{err}");
}

/// An `air.patch` node that states a domain but no control-point count is the same hazard: the
/// count is what sizes every per-patch input the pipeline wires.
#[test]
fn native_patch_without_a_control_point_count_is_refused() {
    let ll = PATCH_CONTROL_POINT_LL.replace(
        "!\"air.patch\", !\"triangle\", !\"air.patch_control_point\", i32 3",
        "!\"air.patch\", !\"triangle\"",
    );
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_patch_no_control_point_{}",
        std::process::id()
    ));
    let err = crate::translate_sanitized_native(&ll, Stage::Vertex, &tmp).expect_err(
        "a patch with no control-point count must not become an ordinary vertex shader",
    );
    assert!(err.contains("air.patch_control_point"), "{err}");
}

/// The patch node is found by what it says, not by where it sits. Metal writes it third in the
/// vertex root today; a root that grows another entry must not turn a tessellation shader into a
/// plain vertex one, nor read a non-patch node as a patch.
#[test]
fn native_patch_node_is_found_by_its_marker_not_its_position() {
    let ll = PATCH_CONTROL_POINT_LL
        .replace("!0 = !{ptr @main, !1, !2, !7}", "!0 = !{ptr @main, !1, !2, !9, !7}")
        .replace(
            "!8 = !{i32 1, !\"air.position_in_patch\", !\"air.arg_type_name\", !\"float2\"}",
            "!8 = !{i32 1, !\"air.position_in_patch\", !\"air.arg_type_name\", !\"float2\"}\n!9 = !{}",
        );
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_patch_marker_position_{}",
        std::process::id()
    ));
    let out = crate::translate_sanitized_native(&ll, Stage::Vertex, &tmp)
        .expect("the patch node is still found past an unrelated root entry");
    let asm = disassemble(&out).expect("disassemble tessellation evaluation module");
    assert!(asm.contains("OpEntryPoint TessellationEvaluation"), "{asm}");
    assert!(asm.contains("Triangles"), "{asm}");
}

#[test]
fn native_reverse_bits_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %rev = tail call i32 @air.reverse_bits.i32(i32 305419896)
  ret void
}

declare i32 @air.reverse_bits.i32(i32)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_reverse_bits_{}",
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
    assert!(asm.contains("OpBitReverse"), "{asm}");
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
fn native_llvm_bswap_i32_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(i32 %x) {
entry:
  %swapped = tail call i32 @llvm.bswap.i32(i32 %x)
  ret void
}

declare i32 @llvm.bswap.i32(i32)
"#;
    let tmp =
        std::env::temp_dir().join(format!("metal2vulkan_native_bswap_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpBitwiseOr"), "{asm}");
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
fn native_llvm_abs_i32_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(i32 %x) {
entry:
  %abs = tail call i32 @llvm.abs.i32(i32 %x, i1 true)
  ret void
}

declare i32 @llvm.abs.i32(i32, i1 immarg)
"#;
    let tmp = std::env::temp_dir().join(format!("metal2vulkan_native_abs_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpExtInst"), "{asm}");
    assert!(asm.contains("SAbs"), "{asm}");
    assert!(!asm.contains("llvm.abs"), "{asm}");
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
fn native_rotate_i32_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(i32 %x, i32 %shift) {
entry:
  %rot = tail call i32 @air.rotate.i32(i32 %x, i32 %shift)
  ret void
}

declare i32 @air.rotate.i32(i32, i32)
"#;
    let tmp =
        std::env::temp_dir().join(format!("metal2vulkan_native_rotate_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(asm.contains("OpBitwiseOr"), "{asm}");
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
fn native_llvm_fshl_lowers_every_spirv_integer_width() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %rot8 = call i8 @llvm.fshl.i8(i8 18, i8 52, i8 3)
  %rot16 = call i16 @llvm.fshl.i16(i16 4660, i16 22136, i16 5)
  %rot32 = call i32 @llvm.fshl.i32(i32 305419896, i32 2596069104, i32 7)
  %rot64 = call i64 @llvm.fshl.i64(i64 1311768467463790320, i64 -81985529216486896, i64 11)
  ret void
}

declare i8 @llvm.fshl.i8(i8, i8, i8)
declare i16 @llvm.fshl.i16(i16, i16, i16)
declare i32 @llvm.fshl.i32(i32, i32, i32)
declare i64 @llvm.fshl.i64(i64, i64, i64)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_llvm_fshl_widths_{}",
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
    assert!(asm.matches("OpShiftLeftLogical").count() >= 4, "{asm}");
    assert!(asm.matches("OpShiftRightLogical").count() >= 4, "{asm}");
    assert!(asm.matches("OpBitwiseOr").count() >= 4, "{asm}");
    assert!(!asm.contains("llvm_fshl"), "{asm}");
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
fn native_fragment_derivatives_and_dot_are_typed_at_construction() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define <4 x float> @frag() {
entry:
  %dx = tail call float @air.dfdx.f32(float 2.000000e+00)
  %width_half = tail call half @air.fwidth.f16(half 0xH3C00)
  %width = fpext half %width_half to float
  %dot = tail call float @air.dot.v2f32(<2 x float> <float 1.000000e+00, float 2.000000e+00>, <2 x float> <float 3.000000e+00, float 4.000000e+00>)
  %sum0 = fadd float %dx, %width
  %sum1 = fadd float %sum0, %dot
  %lane = insertelement <4 x float> poison, float %sum1, i32 0
  %color = shufflevector <4 x float> %lane, <4 x float> poison, <4 x i32> zeroinitializer
  ret <4 x float> %color
}

declare float @air.dfdx.f32(float)
declare half @air.fwidth.f16(half)
declare float @air.dot.v2f32(<2 x float>, <2 x float>)

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_derivative_dot_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpDPdx"), "{asm}");
    assert!(asm.contains("OpFwidth"), "{asm}");
    assert!(asm.contains("OpDot"), "{asm}");
    assert!(asm.matches("OpFConvert").count() >= 2, "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
}

#[test]
fn native_dot_rejects_nonvector_air_operands_before_assembly() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define <4 x float> @frag() {
entry:
  %dot = tail call float @air.dot.f32(float 1.000000e+00, float 2.000000e+00)
  %lane = insertelement <4 x float> poison, float %dot, i32 0
  %color = shufflevector <4 x float> %lane, <4 x float> poison, <4 x i32> zeroinitializer
  ret <4 x float> %color
}

declare float @air.dot.f32(float, float)

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_invalid_dot_{}",
        std::process::id()
    ));
    let error = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp)
        .expect_err("scalar air.dot operands must fail during construction");
    assert!(
        error.contains("operands are not identical vectors of its result component type"),
        "{error}"
    );
}

#[test]
fn native_extract_bits_u32_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %bits = tail call i32 @air.extract_bits.u.i32(i32 305419896, i32 4, i32 8)
  ret void
}

declare i32 @air.extract_bits.u.i32(i32, i32, i32)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_extract_bits_{}",
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
    assert!(asm.contains("OpBitFieldUExtract"), "{asm}");
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
fn native_narrow_bitfield_intrinsics_widen_the_vulkan_base_to_i32() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %extracted = tail call i16 @air.extract_bits.s.i16(i16 4660, i32 4, i32 8)
  %inserted = tail call i16 @air.insert_bits.u.i16(i16 %extracted, i16 10, i32 4, i32 8)
  %vector_extracted = tail call <2 x i16> @air.extract_bits.u.v2i16(<2 x i16> <i16 4660, i16 22136>, i32 4, i32 8)
  %vector_inserted = tail call <2 x i16> @air.insert_bits.u.v2i16(<2 x i16> %vector_extracted, <2 x i16> <i16 10, i16 11>, i32 4, i32 8)
  ret void
}

declare i16 @air.extract_bits.s.i16(i16, i32, i32)
declare i16 @air.insert_bits.u.i16(i16, i16, i32, i32)
declare <2 x i16> @air.extract_bits.u.v2i16(<2 x i16>, i32, i32)
declare <2 x i16> @air.insert_bits.u.v2i16(<2 x i16>, <2 x i16>, i32, i32)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_narrow_bitfields_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpBitFieldSExtract"), "{asm}");
    assert!(asm.contains("OpBitFieldUExtract"), "{asm}");
    assert!(asm.matches("OpBitFieldInsert").count() >= 2, "{asm}");
    assert!(asm.matches("OpUConvert").count() >= 10, "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_extract_bits_u64_avoids_maintenance9_bitfield_opcode() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %bits = tail call i64 @air.extract_bits.u.i64(i64 81985529216486895, i32 40, i32 8)
  ret void
}

declare i64 @air.extract_bits.u.i64(i64, i32, i32)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_extract_bits_u64_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(!asm.contains("OpBitFieldUExtract"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_insert_bits_u32_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %bits = tail call i32 @air.insert_bits.u.i32(i32 305419896, i32 10, i32 4, i32 8)
  ret void
}

declare i32 @air.insert_bits.u.i32(i32, i32, i32, i32)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_insert_bits_{}",
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
    assert!(asm.contains("OpBitFieldInsert"), "{asm}");
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
fn native_insert_bits_u64_avoids_maintenance9_bitfield_opcode() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %bits = tail call i64 @air.insert_bits.u.i64(i64 81985529216486895, i64 10, i32 40, i32 8)
  ret void
}

declare i64 @air.insert_bits.u.i64(i64, i64, i32, i32)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_insert_bits_u64_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(!asm.contains("OpBitFieldInsert"), "{asm}");
    assert!(asm.contains("OpBitwiseOr"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_ctz_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %count = tail call i32 @air.ctz.i32(i32 305419896, i1 false)
  ret void
}

declare i32 @air.ctz.i32(i32, i1)
"#;
    let tmp = std::env::temp_dir().join(format!("metal2vulkan_native_ctz_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpIEqual"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(!asm.contains("OpBitCount"), "{asm}");
    assert!(!asm.contains("FindILsb"), "{asm}");
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
fn native_llvm_cttz_i32_zero_undef_lowers_to_find_lsb() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %count = tail call i32 @llvm.cttz.i32(i32 305419896, i1 true)
  ret void
}

declare i32 @llvm.cttz.i32(i32, i1)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_llvm_cttz_zero_undef_{}",
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
    assert!(asm.contains("OpExtInst"), "{asm}");
    assert!(asm.contains("FindILsb"), "{asm}");
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
fn native_llvm_cttz_i32_defined_zero_lowers_to_select_width() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %count = tail call i32 @llvm.cttz.i32(i32 305419896, i1 false)
  ret void
}

declare i32 @llvm.cttz.i32(i32, i1)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_llvm_cttz_defined_zero_{}",
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
    assert!(asm.contains("OpExtInst"), "{asm}");
    assert!(asm.contains("FindILsb"), "{asm}");
    assert!(asm.contains("OpIEqual"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
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
fn native_clz_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %count = tail call i32 @air.clz.i32(i32 305419896, i1 false)
  ret void
}

declare i32 @air.clz.i32(i32, i1)
"#;
    let tmp = std::env::temp_dir().join(format!("metal2vulkan_native_clz_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpIEqual"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(!asm.contains("OpBitCount"), "{asm}");
    assert!(!asm.contains("FindUMsb"), "{asm}");
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
fn native_fast_tan_and_fmod_lower() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %tan = tail call fast float @air.fast_tan.f32(float 5.000000e-01)
  %v0 = insertelement <2 x float> poison, float 5.500000e+00, i32 0
  %v1 = insertelement <2 x float> %v0, float -5.500000e+00, i32 1
  %d0 = insertelement <2 x float> poison, float 2.000000e+00, i32 0
  %d1 = insertelement <2 x float> %d0, float 2.000000e+00, i32 1
  %fmodv = tail call fast <2 x float> @air.fast_fmod.v2f32(<2 x float> %v1, <2 x float> %d1)
  %fmods = tail call fast float @air.fast_fmod.f32(float -5.500000e+00, float 2.000000e+00)
  %lane = extractelement <2 x float> %fmodv, i32 0
  %sum0 = fadd fast float %tan, %fmods
  %sum1 = fadd fast float %sum0, %lane
  %sink = fcmp oge float %sum1, 0.000000e+00
  ret void
}

declare float @air.fast_tan.f32(float)
declare <2 x float> @air.fast_fmod.v2f32(<2 x float>, <2 x float>)
declare float @air.fast_fmod.f32(float, float)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fast_tan_fmod_{}",
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
    assert!(asm.contains(" Tan "), "{asm}");
    assert!(asm.contains(" Trunc "), "{asm}");
    assert!(asm.contains("OpFDiv"), "{asm}");
    assert!(asm.contains("OpFMul"), "{asm}");
    assert!(asm.contains("OpFSub"), "{asm}");
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
fn native_fast_pi_transcendentals_lower() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %cos = tail call fast float @air.fast_cospi.f32(float 5.000000e-01)
  %sin = tail call fast float @air.fast_sinpi.f32(float 2.500000e-01)
  %tan = tail call fast float @air.fast_tanpi.f32(float 1.250000e-01)
  %sum0 = fadd fast float %cos, %sin
  %sum1 = fadd fast float %sum0, %tan
  %sink = fcmp oge float %sum1, 0.000000e+00
  ret void
}

declare float @air.fast_cospi.f32(float)
declare float @air.fast_sinpi.f32(float)
declare float @air.fast_tanpi.f32(float)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fast_pi_transcendentals_{}",
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
    assert!(asm.contains(" Cos "), "{asm}");
    assert!(asm.contains(" Sin "), "{asm}");
    assert!(asm.contains(" Tan "), "{asm}");
    assert!(asm.matches("OpFMul").count() >= 3, "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn native_fast_sincos_f32_zeroes_large_arguments() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main(float %x) {
entry:
  %cos = tail call fast float @air.fast_cos.f32(float %x)
  %sin = tail call fast float @air.fast_sin.f32(float %x)
  %sum = fadd fast float %cos, %sin
  %sink = fcmp oge float %sum, 0.000000e+00
  ret void
}

declare float @air.fast_cos.f32(float)
declare float @air.fast_sin.f32(float)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fast_sincos_large_{}",
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
    assert!(asm.contains(" Trunc "), "{asm}");
    assert!(asm.contains(" Cos "), "{asm}");
    assert!(asm.contains(" Sin "), "{asm}");
    assert!(asm.contains("OpFDiv"), "{asm}");
    assert!(asm.contains("OpFSub"), "{asm}");
    assert!(asm.contains("OpFOrdGreaterThanEqual"), "{asm}");
    assert!(asm.contains("1073741800"), "{asm}");
    assert!(asm.matches("OpSelect").count() >= 2, "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn native_fast_ldexp_lowers_to_exp2_scale() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %scaled = tail call fast float @air.fast_ldexp.f32(float 1.250000e+00, i32 3)
  %sink = fcmp oge float %scaled, 0.000000e+00
  ret void
}

declare float @air.fast_ldexp.f32(float, i32)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fast_ldexp_{}",
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
    assert!(asm.contains("OpConvertSToF"), "{asm}");
    assert!(asm.contains(" Exp2 "), "{asm}");
    assert!(asm.contains("OpFMul"), "{asm}");
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
fn native_llvm_fabs_lowers_to_extinst() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %abs = tail call fast float @llvm.fabs.f32(float -1.250000e+00)
  %sink = fcmp oge float %abs, 0.000000e+00
  ret void
}

declare float @llvm.fabs.f32(float)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_llvm_fabs_{}",
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
fn native_fast_atan_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @main() {
entry:
  %atan = tail call fast float @air.fast_atan.f32(float 5.000000e-01)
  %sink = fcmp oge float %atan, 0.000000e+00
  ret void
}

declare float @air.fast_atan.f32(float)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fast_atan_{}",
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
    assert!(asm.contains(" Atan "), "{asm}");
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
fn native_ignores_lifetime_intrinsics_with_generic_ptrs() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.2"
define void @life() {
entry:
  call void @llvm.lifetime.start.p0(ptr undef)
  call void @llvm.lifetime.end.p0(ptr undef)
  ret void
}

declare void @llvm.lifetime.start.p0(ptr)
declare void @llvm.lifetime.end.p0(ptr)
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(!asm.contains("llvm.lifetime"), "{asm}");
    assert!(asm.contains("OpReturn"), "{asm}");
}
