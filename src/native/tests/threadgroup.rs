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
fn native_workgroup_array_vector_load_uses_first_element() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@tile = internal unnamed_addr addrspace(3) global [256 x <4 x half>] undef, align 8

define void @k(ptr addrspace(1) %out) {
entry:
  %v = load <4 x half>, ptr addrspace(3) @tile, align 8
  store <4 x half> %v, ptr addrspace(1) %out, align 8
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"half4*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_workgroup_array_vec_load_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(!asm.contains("OpBitcast"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_ushort2_threads_per_threadgroup_is_constant_vector() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main(<2 x i16> %lsize) {
entry:
  %x = extractelement <2 x i16> %lsize, i32 0
  %ok = icmp uge i16 %x, 0
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.threads_per_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"lsize"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_ushort2_threads_per_threadgroup_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpConstantComposite"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_simd_shuffle_and_barrier_lower_to_subgroup_ops() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(i32 %v, i16 %lane) {
entry:
  %out = tail call i32 @air.simd_shuffle.u.i32(i32 %v, i16 %lane)
  %signed = tail call i32 @air.simd_shuffle.s.i32(i32 %out, i16 %lane)
  %ok = icmp uge i32 %signed, 0
  tail call void @air.simdgroup.barrier(i32 0, i32 1)
  ret void
}

declare i32 @air.simd_shuffle.u.i32(i32, i16)
declare i32 @air.simd_shuffle.s.i32(i32, i16)
declare void @air.simdgroup.barrier(i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp =
        std::env::temp_dir().join(format!("metal2vulkan_simd_shuffle_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformShuffleRelative"),
        "{asm}"
    );
    assert_eq!(asm.matches("OpGroupNonUniformShuffle").count(), 2, "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(asm.contains("OpControlBarrier"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_quad_sum_lowers_to_shuffle_xor_butterfly() {
    // quad_sum has no quad-scoped reduction op in SPIR-V; it lowers to an XOR butterfly over the two
    // intra-quad swap axes (mask 1, then mask 2), so every lane ends with the full quad sum. Two
    // GroupNonUniformShuffleXor + two FAdd, using only the Shuffle capability (no GroupNonUniformQuad).
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(float %v, ptr addrspace(1) %out) {
entry:
  %sum = tail call float @air.quad_sum.f32(float %v)
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.quad_sum.f32(float)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!("metal2vulkan_quad_sum_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(
        !asm.contains("OpCapability GroupNonUniformQuad"),
        "quad_sum must not require the GroupNonUniformQuad capability: {asm}"
    );
    assert_eq!(
        asm.matches("OpGroupNonUniformShuffleXor").count(),
        2,
        "{asm}"
    );
    assert_eq!(asm.matches("OpFAdd").count(), 2, "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_simd_sum_clusters_reduce_only_under_transform_option() {
    // air.simd_sum reduces over Metal's 32-lane simdgroup. By default the translator emits a
    // whole-subgroup GroupNonUniformFAdd Reduce; TransformOptions::simd_cluster32 (F4 promoted it
    // from the former METAL2VULKAN_SIMD_CLUSTER32 env knob, surfaced as the --simd-cluster32 CLI
    // flag) turns it into a ClusteredReduce over a 32-lane cluster + the GroupNonUniformClustered
    // capability so a wider-subgroup driver still reduces over exactly 32 lanes. Default off is
    // byte-neutral; this locks the option→ctx threading the env read used to carry.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(float %v, ptr addrspace(1) %out) {
entry:
  %sum = tail call float @air.simd_sum.f32(float %v)
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.simd_sum.f32(float)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!("metal2vulkan_simd_sum_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);

    // Default options: whole-subgroup Reduce, no clustering, no clustered capability.
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpGroupNonUniformFAdd"), "{asm}");
    assert!(
        !asm.contains("ClusteredReduce"),
        "default lowering must be a whole-subgroup Reduce, not clustered: {asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformClustered"),
        "{asm}"
    );

    // Opt-in via TransformOptions: ClusteredReduce over 32 lanes + the clustered capability.
    let opts = passes::TransformOptions {
        simd_cluster32: true,
        ..passes::TransformOptions::default()
    };
    let spv = crate::translate_sanitized_native_with_options(ll, Stage::Kernel, &tmp, opts)
        .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("ClusteredReduce"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformClustered"),
        "{asm}"
    );
}

#[test]
fn native_air_simd_is_first_lowers_to_subgroup_elect() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %out) {
entry:
  %first = tail call i1 @air.simd_is_first()
  %word = select i1 %first, i32 1, i32 0
  store i32 %word, ptr addrspace(1) %out, align 4
  ret void
}

declare i1 @air.simd_is_first()

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simd_is_first_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformElect"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_quad_all_lowers_to_subgroup_vote() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(i32 %v, ptr addrspace(1) %out) {
entry:
  %ok = icmp ne i32 %v, 0
  %all = tail call i1 @air.quad_all(i1 %ok)
  %word = select i1 %all, i32 1, i32 0
  store i32 %word, ptr addrspace(1) %out, align 4
  ret void
}

declare i1 @air.quad_all(i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_quad_all_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(asm.contains("OpCapability GroupNonUniformVote"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformAll"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_simd_any_all_lower_to_subgroup_vote() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(i32 %v, ptr addrspace(1) %out) {
entry:
  %ok = icmp ne i32 %v, 0
  %any = tail call i1 @air.simd_any(i1 %ok)
  %all = tail call i1 @air.simd_all(i1 %ok)
  %any_word = select i1 %any, i32 1, i32 0
  %all_word = select i1 %all, i32 2, i32 0
  %word = or i32 %any_word, %all_word
  store i32 %word, ptr addrspace(1) %out, align 4
  ret void
}

declare i1 @air.simd_any(i1)
declare i1 @air.simd_all(i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simd_any_all_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(asm.contains("OpCapability GroupNonUniformVote"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformAny"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformAll"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_simd_ballot_i64_lowers_to_subgroup_ballot() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(i32 %v, ptr addrspace(1) %out) {
entry:
  %ok = icmp ne i32 %v, 0
  %mask = tail call i64 @air.simd_ballot.i64(i1 %ok)
  %lo = trunc i64 %mask to i32
  store i32 %lo, ptr addrspace(1) %out, align 4
  ret void
}

declare i64 @air.simd_ballot.i64(i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simd_ballot_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(asm.contains("OpCapability GroupNonUniformBallot"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformBallot"), "{asm}");
    assert!(asm.contains("OpCompositeExtract"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpBitwiseOr"), "{asm}");
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
fn native_air_get_simdgroup_size_i16_lowers_to_width_constant() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %out) {
entry:
  %width16 = tail call i16 @air.get_simdgroup_size.i16()
  %width = zext i16 %width16 to i32
  store i32 %width, ptr addrspace(1) %out, align 4
  ret void
}

declare i16 @air.get_simdgroup_size.i16()

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simdgroup_size_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.trim_end().ends_with(" 32")),
        "{asm}"
    );
    assert!(
        !asm.contains("air.get_simdgroup_size"),
        "intrinsic call survived lowering:\n{asm}"
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
fn native_air_simd_shuffle_and_fill_down_lowers_to_clustered_subgroup_shuffle() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(half %x, half %fill, i16 %delta, ptr addrspace(1) %out) {
entry:
  %width = tail call i16 @air.get_simdgroup_size.i16()
  %down = tail call half @air.simd_shuffle_and_fill_down.f16(half %x, half %fill, i16 %delta, i16 %width)
  %f = fpext half %down to float
  store float %f, ptr addrspace(1) %out, align 4
  ret void
}

declare i16 @air.get_simdgroup_size.i16()
declare half @air.simd_shuffle_and_fill_down.f16(half, half, i16, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_shuffle_fill_down_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformShuffleRelative"),
        "{asm}"
    );
    assert!(asm.contains("SubgroupLocalInvocationId"), "{asm}");
    assert_eq!(
        asm.lines()
            .filter(|line| line.contains("= OpGroupNonUniformShuffle "))
            .count(),
        2,
        "{asm}"
    );
    assert!(asm.contains("OpUMod"), "{asm}");
    assert!(asm.contains("OpISub"), "{asm}");
    assert!(asm.contains("OpULessThan"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(
        !asm.contains("air.simd_shuffle_and_fill_down"),
        "intrinsic call survived lowering:\n{asm}"
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
fn native_air_simd_broadcast_lowers_to_subgroup_shuffle() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(float %x, i16 %lane, ptr addrspace(1) %out) {
entry:
  %sx = tail call float @air.simd_broadcast.f32(float %x, i16 %lane)
  store float %sx, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.simd_broadcast.f32(float, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_broadcast_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformShuffleRelative"),
        "{asm}"
    );
    assert!(asm.contains("OpGroupNonUniformShuffle"), "{asm}");
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
fn native_air_simd_shuffle_down_lowers_to_32_lane_absolute_shuffle() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(float %x, half %h, i16 %delta, ptr addrspace(1) %out) {
entry:
  %sx = tail call float @air.simd_shuffle_down.f32(float %x, i16 %delta)
  %sh = tail call half @air.simd_shuffle_down.f16(half %h, i16 %delta)
  %hf = fpext half %sh to float
  %sum = fadd float %sx, %hf
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.simd_shuffle_down.f32(float, i16)
declare half @air.simd_shuffle_down.f16(half, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_shuffle_down_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformShuffleRelative"),
        "{asm}"
    );
    assert_eq!(asm.matches("OpGroupNonUniformShuffle").count(), 2, "{asm}");
    assert!(!asm.contains("OpGroupNonUniformShuffleDown"), "{asm}");
    assert!(asm.contains("SubgroupLocalInvocationId"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
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
fn native_air_simd_shuffle_rotate_down_lowers_to_wrapping_subgroup_shuffle() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(float %x, i16 %delta, ptr addrspace(1) %out) {
entry:
  %sx = tail call float @air.simd_shuffle_rotate_down.f32(float %x, i16 %delta)
  store float %sx, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.simd_shuffle_rotate_down.f32(float, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_shuffle_rotate_down_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformShuffleRelative"),
        "{asm}"
    );
    assert_eq!(asm.matches("OpGroupNonUniformShuffle").count(), 1, "{asm}");
    assert!(asm.contains("OpIAdd"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(!asm.contains("air.simd_shuffle_rotate_down"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_simd_shuffle_up_lowers_to_32_lane_absolute_shuffle() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(i32 %x, i16 %delta, ptr addrspace(1) %out) {
entry:
  %sx = tail call i32 @air.simd_shuffle_up.u.i32(i32 %x, i16 %delta)
  store i32 %sx, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.simd_shuffle_up.u.i32(i32, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"int", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_shuffle_up_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformShuffleRelative"),
        "{asm}"
    );
    assert!(asm.contains("OpGroupNonUniformShuffle"), "{asm}");
    assert!(!asm.contains("OpGroupNonUniformShuffleUp"), "{asm}");
    assert!(asm.contains("SubgroupLocalInvocationId"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
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
fn native_air_simd_shuffle_xor_lowers_to_subgroup_shuffle_xor() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(half %x, i16 %mask, ptr addrspace(1) %out) {
entry:
  %sx = tail call half @air.simd_shuffle_xor.f16(half %x, i16 %mask)
  %f = fpext half %sx to float
  store float %f, ptr addrspace(1) %out, align 4
  ret void
}

declare half @air.simd_shuffle_xor.f16(half, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_shuffle_xor_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffle"),
        "{asm}"
    );
    assert!(
        !asm_has_line(&asm, "OpCapability GroupNonUniformShuffleRelative"),
        "{asm}"
    );
    assert!(asm.contains("OpGroupNonUniformShuffleXor"), "{asm}");
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
fn native_air_quad_shuffle_down_vector_lowers_to_subgroup_shuffle_down() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(<4 x float> %x, i16 %delta, ptr addrspace(1) %out) {
entry:
  %sx = tail call <4 x float> @air.quad_shuffle_down.v4f32(<4 x float> %x, i16 %delta)
  store <4 x float> %sx, ptr addrspace(1) %out, align 16
  ret void
}

declare <4 x float> @air.quad_shuffle_down.v4f32(<4 x float>, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_quad_shuffle_down_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformShuffleRelative"),
        "{asm}"
    );
    assert_eq!(
        asm.matches("OpGroupNonUniformShuffleDown").count(),
        1,
        "{asm}"
    );
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
fn native_air_simd_prefix_exclusive_sum_lowers_to_subgroup_scan() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(float %x, ptr addrspace(1) %out) {
entry:
  %sum = tail call float @air.simd_prefix_exclusive_sum.f32(float %x)
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.simd_prefix_exclusive_sum.f32(float)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp =
        std::env::temp_dir().join(format!("metal2vulkan_native_prefix_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformArithmetic"),
        "{asm}"
    );
    assert!(asm.contains("OpGroupNonUniformFAdd"), "{asm}");
    assert!(asm.contains("ExclusiveScan"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_simd_sum_lowers_to_subgroup_reduce() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(float %x, ptr addrspace(1) %out) {
entry:
  %sum = tail call float @air.simd_sum.f32(float %x)
  store float %sum, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.simd_sum.f32(float)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simd_sum_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformArithmetic"),
        "{asm}"
    );
    assert!(asm.contains("OpGroupNonUniformFAdd"), "{asm}");
    assert!(asm.contains("Reduce"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_simd_or_i8_lowers_to_subgroup_bitwise_reduce() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(i8 %x, ptr addrspace(1) %out) {
entry:
  %mask = tail call i8 @air.simd_or.u.i8(i8 %x)
  %wide = zext i8 %mask to i32
  store i32 %wide, ptr addrspace(1) %out, align 4
  ret void
}

declare i8 @air.simd_or.u.i8(i8)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simd_or_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformArithmetic"),
        "{asm}"
    );
    assert!(asm.contains("OpCapability Int8"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformBitwiseOr"), "{asm}");
    assert!(asm.contains("Reduce"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_simd_inclusive_sum_min_max_lower_to_subgroup_arithmetic() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(float %x, ptr addrspace(1) %out) {
entry:
  %sum = tail call float @air.simd_prefix_inclusive_sum.f32(float %x)
  %min = tail call float @air.simd_min.f32(float %x)
  %max = tail call float @air.simd_max.f32(float %x)
  %a = fadd fast float %sum, %min
  %b = fadd fast float %a, %max
  store float %b, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.simd_prefix_inclusive_sum.f32(float)
declare float @air.simd_min.f32(float)
declare float @air.simd_max.f32(float)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simd_extrema_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCapability GroupNonUniform"), "{asm}");
    assert!(
        asm_has_line(&asm, "OpCapability GroupNonUniformArithmetic"),
        "{asm}"
    );
    assert!(asm.contains("OpGroupNonUniformFAdd"), "{asm}");
    assert!(asm.contains("InclusiveScan"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformFMin"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformFMax"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_threads_per_grid_shares_num_workgroups_builtin() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(<2 x i32> %grid_size, <2 x i32> %group_count) {
entry:
  %sx = extractelement <2 x i32> %grid_size, i64 0
  %sy = extractelement <2 x i32> %grid_size, i64 1
  %gx = extractelement <2 x i32> %group_count, i64 0
  %gy = extractelement <2 x i32> %group_count, i64 1
  %sum0 = add i32 %sx, %sy
  %sum1 = add i32 %gx, %gy
  %sum = add i32 %sum0, %sum1
  %ok = icmp uge i32 %sum, 0
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"grid_size"}
!4 = !{i32 1, !"air.threadgroups_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"group_count"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_threads_per_grid_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions {
            kernel_local_size: [32, 2, 1],
            kernel_threads_per_grid: None,
            simd_cluster32: false,
            ..passes::TransformOptions::default()
        },
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("LocalSize 32 2 1"), "{asm}");
    assert!(asm.contains("BuiltIn NumWorkgroups"), "{asm}");
    assert_eq!(asm.matches("BuiltIn NumWorkgroups").count(), 1, "{asm}");
    assert!(asm.contains("OpIMul"), "{asm}");
    assert!(!asm.contains("OpUndef"), "{asm}");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.contains("32")),
        "{asm}"
    );
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.contains("2")),
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
fn native_kernel_threads_per_grid_accepts_exact_dispatch_threads_shape() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(<2 x i32> %grid_size) {
entry:
  %sx = extractelement <2 x i32> %grid_size, i64 0
  %sy = extractelement <2 x i32> %grid_size, i64 1
  %sum = add i32 %sx, %sy
  %ok = icmp uge i32 %sum, 0
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"grid_size"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_exact_threads_per_grid_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions {
            kernel_local_size: [5, 2, 1],
            kernel_threads_per_grid: Some([21, 3, 1]),
            simd_cluster32: false,
            ..passes::TransformOptions::default()
        },
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("LocalSize 5 2 1"), "{asm}");
    assert!(!asm.contains("BuiltIn NumWorkgroups"), "{asm}");
    assert!(!asm.contains("OpIMul"), "{asm}");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.contains("21")),
        "{asm}"
    );
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.contains("3")),
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
fn native_kernel_threadgroup_packed_float1_record_array_validates() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Temp = type { [1 x float], i32 }

define void @k(ptr addrspace(3) %temp, i32 %i) {
entry:
  %idx = zext i32 %i to i64
  %value = getelementptr inbounds %struct.Temp, ptr addrspace(3) %temp, i64 %idx, i32 0, i64 0
  store float 1.000000e+00, ptr addrspace(3) %value, align 4
  %count = getelementptr inbounds %struct.Temp, ptr addrspace(3) %temp, i64 %idx, i32 1
  store i32 %i, ptr addrspace(3) %count, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.struct_type_info", !5, !"air.arg_type_name", !"Temp", !"air.arg_name", !"temp"}
!4 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!5 = !{i32 0, i32 4, i32 0, !"packed_float1", !"value", i32 4, i32 4, i32 0, !"uint", !"count"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_packed_float1_record_array_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Workgroup"), "{asm}");
    assert!(asm.contains("OpTypeArray"), "{asm}");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.contains("512")),
        "{asm}"
    );
    assert!(!asm.contains("DescriptorSet"), "{asm}");
    assert!(!asm.contains("Binding"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_threadgroup_param_direct_scalar_load_uses_element_zero() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(3) %temp, i32 %i) {
entry:
  %idx = zext i32 %i to i64
  %slot = getelementptr inbounds float, ptr addrspace(3) %temp, i64 %idx
  store float 2.000000e+00, ptr addrspace(3) %temp, align 4
  %root = load float, ptr addrspace(3) %temp, align 4
  store float %root, ptr addrspace(3) %slot, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"temp"}
!4 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_threadgroup_direct_scalar_root_load_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load spv");
    let workgroup_vars = module
        .types_global_values
        .iter()
        .filter_map(|inst| {
            (inst.class.opcode == Op::Variable
                && inst.operands.first() == Some(&Operand::StorageClass(StorageClass::Workgroup)))
            .then_some(inst.result_id?)
        })
        .collect::<HashSet<_>>();
    assert!(!workgroup_vars.is_empty(), "{asm}");
    let direct_workgroup_load = module
        .functions
        .iter()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .any(|inst| {
            inst.class.opcode == Op::Load
                && inst
                    .operands
                    .first()
                    .and_then(id_ref_operand)
                    .is_some_and(|id| workgroup_vars.contains(&id))
        });
    assert!(!direct_workgroup_load, "{asm}");
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
fn native_threadgroup_atomic_i32_lowers_to_workgroup_spirv() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_atomic" = type { i32 }
@local_counts = internal addrspace(3) global [1 x %"struct.metal::_atomic"] zeroinitializer, align 4

define void @k() {
entry:
  %p = getelementptr inbounds [1 x %"struct.metal::_atomic"], ptr addrspace(3) @local_counts, i64 0, i64 0, i32 0
  tail call void @air.atomic.local.store.i32(ptr addrspace(3) %p, i32 0, i32 0, i32 1, i1 true)
  tail call void @air.wg.barrier(i32 2, i32 1)
  %v = tail call i32 @air.atomic.local.load.i32(ptr addrspace(3) %p, i32 0, i32 1, i1 true)
  %old = tail call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) %p, i32 1, i32 0, i32 1, i1 true)
  %signed_old = tail call i32 @air.atomic.local.add.s.i32(ptr addrspace(3) %p, i32 -1, i32 0, i32 1, i1 true)
  %signed_max_old = tail call i32 @air.atomic.local.max.s.i32(ptr addrspace(3) %p, i32 -3, i32 0, i32 1, i1 true)
  %max_old = tail call i32 @air.atomic.local.max.u.i32(ptr addrspace(3) %p, i32 9, i32 0, i32 1, i1 true)
  %signed_min_old = tail call i32 @air.atomic.local.min.s.i32(ptr addrspace(3) %p, i32 -9, i32 0, i32 1, i1 true)
  %min_old = tail call i32 @air.atomic.local.min.u.i32(ptr addrspace(3) %p, i32 3, i32 0, i32 1, i1 true)
  %masked = tail call i32 @air.atomic.local.and.u.i32(ptr addrspace(3) %p, i32 255, i32 0, i32 1, i1 true)
  %mask = tail call i32 @air.atomic.local.or.u.i32(ptr addrspace(3) %p, i32 2, i32 0, i32 1, i1 true)
  ret void
}

declare void @air.atomic.local.store.i32(ptr addrspace(3), i32, i32, i32, i1)
declare void @air.wg.barrier(i32, i32)
declare i32 @air.atomic.local.load.i32(ptr addrspace(3), i32, i32, i1)
declare i32 @air.atomic.local.add.u.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.add.s.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.max.s.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.max.u.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.min.s.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.min.u.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.and.u.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.or.u.i32(ptr addrspace(3), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_atomic_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Workgroup"), "{asm}");
    assert!(asm.contains("OpControlBarrier"), "{asm}");
    assert!(asm.contains("OpAtomicStore"), "{asm}");
    assert!(asm.contains("OpAtomicLoad"), "{asm}");
    assert!(asm.matches("OpAtomicIAdd").count() >= 2, "{asm}");
    assert!(asm.contains("OpAtomicSMax"), "{asm}");
    assert!(asm.contains("OpAtomicUMax"), "{asm}");
    assert!(asm.contains("OpAtomicSMin"), "{asm}");
    assert!(asm.contains("OpAtomicUMin"), "{asm}");
    assert!(asm.contains("OpAtomicAnd"), "{asm}");
    assert!(asm.contains("OpAtomicOr"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_threadgroup_atomic_fixed_loop_is_flattened_and_unrolled() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_atomic" = type { i32 }
@local_counts = internal addrspace(3) global [4 x %"struct.metal::_atomic"] zeroinitializer, align 4

define void @k() {
entry:
  %initp = getelementptr inbounds [4 x %"struct.metal::_atomic"], ptr addrspace(3) @local_counts, i64 0, i64 0, i32 0
  store i32 0, ptr addrspace(3) %initp, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %bin64 = zext i32 %i to i64
  %p = getelementptr inbounds [4 x %"struct.metal::_atomic"], ptr addrspace(3) @local_counts, i64 0, i64 %bin64, i32 0
  %old = tail call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) %p, i32 1, i32 0, i32 1, i1 true)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, 4
  br i1 %done, label %exit, label %loop

exit:
  tail call void @air.wg.barrier(i32 2, i32 1)
  ret void
}

declare void @air.wg.barrier(i32, i32)
declare i32 @air.atomic.local.add.u.i32(ptr addrspace(3), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_atomic_unroll_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("OpTypeStruct %uint"), "{asm}");
    assert_eq!(asm.matches("OpAtomicIAdd").count(), 4, "{asm}");
    assert!(!asm.contains("OpLoopMerge"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_simdgroup_matrix_8x8_lowers_through_scalar_array() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main(ptr addrspace(1) %out) {
entry:
  %a = insertelement <64 x float> zeroinitializer, float 1.000000e+00, i64 0
  %b = insertelement <64 x float> zeroinitializer, float 2.000000e+00, i64 0
  %c = insertelement <64 x float> zeroinitializer, float 3.000000e+00, i64 0
  %m = tail call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float> %a, <64 x float> %b, <64 x float> %c)
  %lane = extractelement <64 x float> %m, i64 0
  store float %lane, ptr addrspace(1) %out, align 4
  ret void
}

declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(<64 x float>, <64 x float>, <64 x float>)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simdgroup_matrix_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(!asm.contains("OpTypeVector %float 64"), "{asm}");
    assert!(asm.contains("OpTypeArray"), "{asm}");
    assert!(asm.contains("OpFMul"), "{asm}");
    assert!(asm.contains("OpFAdd"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_simdgroup_matrix_8x8_full_pipeline_lowers_and_validates() {
    // Exercises all four simdgroup_matrix 8x8 lowerings end to end: load (f16 and f32) from device
    // buffers, init_diag, the mixed-precision multiply_accumulate (v64f32 A, v64f16 B), and store to a
    // device buffer. The descriptor vectors carry the documented `<elements_per_row, 8>` / `<1,
    // elements_per_row>` shape (leading dimension = component 0 of the first descriptor vector).
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main(ptr addrspace(1) %in_h, ptr addrspace(1) %in_f, ptr addrspace(1) %out) {
entry:
  %pv1 = insertelement <2 x i64> <i64 poison, i64 8>, i64 8, i64 0
  %pv2 = insertelement <2 x i64> <i64 1, i64 poison>, i64 8, i64 1
  %ph = getelementptr inbounds half, ptr addrspace(1) %in_h, i64 0
  %pf = getelementptr inbounds float, ptr addrspace(1) %in_f, i64 0
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i64 0
  %A = call <64 x half> @air.simdgroup_matrix_8x8_load.v64f16.p1f16(ptr addrspace(1) %ph, <2 x i64> %pv1, <2 x i64> %pv2, <2 x i64> zeroinitializer)
  %B = call <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(ptr addrspace(1) %pf, <2 x i64> %pv1, <2 x i64> %pv2, <2 x i64> zeroinitializer)
  %C = call <64 x float> @air.simdgroup_matrix_8x8_init_diag.v64f32.f32(float 1.000000e+00)
  %D = call <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float> %B, <64 x half> %A, <64 x float> %C)
  %scaled = fmul fast <64 x float> %D, %D
  call void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float> %scaled, ptr addrspace(1) %po, <2 x i64> %pv1, <2 x i64> %pv2, <2 x i64> zeroinitializer)
  ret void
}

declare <64 x half> @air.simdgroup_matrix_8x8_load.v64f16.p1f16(ptr addrspace(1), <2 x i64>, <2 x i64>, <2 x i64>)
declare <64 x float> @air.simdgroup_matrix_8x8_load.v64f32.p1f32(ptr addrspace(1), <2 x i64>, <2 x i64>, <2 x i64>)
declare <64 x float> @air.simdgroup_matrix_8x8_init_diag.v64f32.f32(float)
declare <64 x float> @air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f16.v64f32(<64 x float>, <64 x half>, <64 x float>)
declare void @air.simdgroup_matrix_8x8_store.v64f32.p1f32(<64 x float>, ptr addrspace(1), <2 x i64>, <2 x i64>, <2 x i64>)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"half", !"air.arg_name", !"in_h"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"in_f"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_simdgroup_matrix_full_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble transformed");
    // The <64 x float> matrix must be modeled as an array, and the wide fmul must be scalarized
    // (no OpFMul over a 64-lane array type).
    assert!(!asm.contains("OpTypeVector %float 64"), "{asm}");
    assert!(asm.contains("OpTypeArray"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}
