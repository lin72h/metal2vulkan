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
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
fn native_air_is_uniform_lowers_to_subgroup_vote() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
fn native_void_air_call_is_emitted_for_lowering() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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

#[test]
fn native_reverse_bits_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
fn native_extract_bits_u32_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
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
fn native_insert_bits_u32_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
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
fn native_ctz_intrinsic_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
target triple = "spirv-unknown-vulkan1.3"
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
