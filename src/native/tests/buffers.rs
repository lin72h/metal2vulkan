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
use spirv::{Capability, Decoration, Op, Scope, SelectionControl, StorageClass, Word};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[test]
fn native_kernel_reinterprets_metadata_ulong_field_as_uint2() {
    let ll = r#"
source_filename = "case.metal"

%struct.CCDebugInfo = type { %union.anon.121 }
%union.anon.121 = type { <2 x i32> }

define void @CC_LogComputeFunction(ptr addrspace(1) noundef readonly align 8 captures(none) dereferenceable(8) "air-buffer-no-alias" %0, ptr addrspace(1) noundef writeonly align 8 captures(none) dereferenceable(8) "air-buffer-no-alias" %1) local_unnamed_addr #0 {
  %3 = getelementptr inbounds %struct.CCDebugInfo, ptr addrspace(1) %0, i64 0, i32 0, i32 0
  %4 = load <2 x i32>, ptr addrspace(1) %3, align 8
  %5 = getelementptr inbounds %struct.CCDebugInfo, ptr addrspace(1) %1, i64 0, i32 0, i32 0
  store <2 x i32> %4, ptr addrspace(1) %5, align 8
  ret void
}

attributes #0 = { nounwind }

!air.kernel = !{!0}
!0 = !{ptr @CC_LogComputeFunction, !1, !2}
!1 = !{}
!2 = !{!3, !6}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"CCDebugInfo", !"air.arg_name", !"logInfo"}
!4 = !{!"air.struct_type_info", !5, i32 0, i32 8, i32 0, !"CCDebugInfo::(anonymous)", !""}
!5 = !{i32 0, i32 8, i32 0, !"uint2", !"frame_id_uint", i32 0, i32 8, i32 0, !"ulong", !"frame_id"}
!6 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"CCDebugInfo", !"air.arg_name", !"logBuffer"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_uint2_ulong_reinterpret_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    // The size-guarded GEP-source override adopts the member-isomorphic LLVM `{{<2 x i32>}}` view
    // over the overlapping union metadata (`{uint2@0, ulong@0}`, same 8-byte extent), so the load
    // is a direct v2uint access chain — no ulong storage and no shift/or reassembly.
    assert!(asm.contains("OpLoad"), "{asm}");
    assert!(asm.contains("OpStore"), "{asm}");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(!asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(!asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(!asm.contains("OpBitwiseOr"), "{asm}");
    assert!(!asm.contains("OpTypeInt 64"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_bda_device_pointer_load_store_deref_lowers_to_physical() {
    // BDA mode: a device pointer (`addrspace(1)`) LOADED from a buffer becomes its real 64-bit address.
    // The kernel loads `%p` from `%in`, STORES it into `%out` (a verbatim 8-byte copy), and
    // DEREFERENCES it (`%p[0]` as float). The store lowers to an Int(64) word write; the deref lowers
    // to `OpConvertUToPtr` of the loaded address to a PhysicalStorageBuffer pointer + an Aligned load.
    // The module switches to the PhysicalStorageBuffer64 addressing model. Structural — never name-keyed.
    let ll = r#"
define void @k(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %p = load ptr addrspace(1), ptr addrspace(1) %in, align 8
  store ptr addrspace(1) %p, ptr addrspace(1) %out, align 8
  %wide = load i64, ptr addrspace(1) %out, align 8
  %narrow = trunc i64 %wide to i32
  %wide_gep = getelementptr inbounds i8, ptr addrspace(1) %p, i64 %wide
  %mixed_gep = getelementptr inbounds i8, ptr addrspace(1) %wide_gep, i32 %narrow
  %f = load float, ptr addrspace(1) %mixed_gep, align 4
  store float %f, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"in"}
"#;
    let spv = crate::native::emit_vulkan_spirv_all_buffers_raw_bda(ll).expect("bda emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm.contains("PhysicalStorageBuffer64"),
        "expected PhysicalStorageBuffer64 addressing model:\n{asm}"
    );
    assert!(
        asm.contains("OpCapability PhysicalStorageBufferAddresses"),
        "{asm}"
    );
    assert!(
        asm.contains("OpConvertUToPtr"),
        "expected a device-address deref:\n{asm}"
    );
    assert!(
        asm.contains("OpSConvert"),
        "mixed-width GEP composition must sign-extend its i32 offset:\n{asm}"
    );
}

#[test]
fn native_bda_pointer_phi_merges_buffer_root_with_child_device_address() {
    let ll = r#"
define internal void @walk(ptr addrspace(1) %root, ptr addrspace(1) %out) {
entry:
  br label %walk

walk:
  %cursor = phi ptr addrspace(1) [ %root, %entry ], [ %child, %body ]
  %done = icmp eq ptr addrspace(1) %cursor, null
  br i1 %done, label %exit, label %body

body:
  %child_addr = load i64, ptr addrspace(1) %cursor, align 8
  %child = inttoptr i64 %child_addr to ptr addrspace(1)
  br label %walk

exit:
  store i32 1, ptr addrspace(1) %out, align 4
  ret void
}

define void @k(ptr addrspace(1) %root, ptr addrspace(1) %out) {
entry:
  call void @walk(ptr addrspace(1) %root, ptr addrspace(1) %out)
  %opaque = insertvalue { ptr addrspace(1) } poison, ptr addrspace(1) null, 0
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"root"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_bda_address_phi_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_bda_probe(ll, Stage::Kernel, &tmp).expect("BDA translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.matches("OpPhi").count() >= 2, "{asm}");
    assert!(asm.contains("OpConvertUToPtr"), "{asm}");
    assert!(
        !asm.contains("OpConstantNull %_ptr_UniformConstant_uchar"),
        "a provably-null opaque aggregate field must use integer zero in BDA mode:\n{asm}"
    );
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn native_bda_inlined_helper_atomic_uses_a_physical_word_pointer() {
    let ll = r#"
define internal void @increment_leaf(ptr addrspace(1) %counters, i32 %index) {
entry:
  %slot = getelementptr inbounds i32, ptr addrspace(1) %counters, i32 %index
  %old = call i32 @air.atomic.global.add.s.i32(ptr addrspace(1) %slot, i32 1, i32 0, i32 2, i1 true)
  ret void
}

define internal void @increment(ptr addrspace(1) %counters, i32 %index) {
entry:
  br label %body

body:
  call void @increment_leaf(ptr addrspace(1) %counters, i32 %index)
  ret void
}

define void @k(ptr addrspace(1) %counters) {
entry:
  call void @increment(ptr addrspace(1) %counters, i32 3)
  ret void
}

declare i32 @air.atomic.global.add.s.i32(ptr addrspace(1), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"counters"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_bda_helper_atomic_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_bda_probe(ll, Stage::Kernel, &tmp).expect("BDA translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpAtomicIAdd"), "{asm}");
    assert!(asm.contains("OpConvertUToPtr"), "{asm}");
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn native_mtl_force_not_checked_i64_load_uses_bda_inttoptr_address() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"

define void @k(ptr addrspace(1) %out, ptr addrspace(1) %src) {
entry:
  %addr = load i64, ptr addrspace(1) %src, align 8
  %p = inttoptr i64 %addr to ptr addrspace(1)
  %field = getelementptr inbounds i8, ptr addrspace(1) %p, i64 8
  %v = tail call i64 @mtl.force_not_checked.load.i64.p1(ptr addrspace(1) %field)
  store i64 %v, ptr addrspace(1) %out, align 8
  ret void
}

declare extern_weak i64 @mtl.force_not_checked.load.i64.p1(ptr addrspace(1)) section "air.externally_defined"

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"src"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_mtl_force_not_checked_bda_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("PhysicalStorageBuffer64"), "{asm}");
    assert!(asm.contains("OpConvertUToPtr"), "{asm}");
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
fn native_direct_buffer_pointer_store_loads_runtime_address_sidecar() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %out, ptr addrspace(1) %source) {
entry:
  store ptr addrspace(1) %source, ptr addrspace(1) %out, align 8
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"source"}
"#;
    let kern = meta::parse_air_kernel_meta(ll);
    let entry_name = meta::entry_name(ll, "kernel");
    let emitted = crate::native::emit_vulkan_spirv_all_buffers_raw_with_sidecar(
        ll,
        kern.as_ref(),
        entry_name.as_deref(),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
    )
    .expect("emit raw-buffer tier with typed sidecar");
    let mut address_words = emitted
        .sidecar
        .buffer_address_words
        .iter()
        .map(|fact| (fact.param_index, fact.component))
        .collect::<Vec<_>>();
    address_words.sort_unstable();
    assert_eq!(address_words, vec![(1, 0), (1, 1)]);
    assert!(emitted
        .sidecar
        .buffer_address_words
        .iter()
        .all(|fact| fact.id != 0));

    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_direct_buffer_address_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Binding 640"), "{asm}");
    assert!(asm.contains("ArrayStride 8"), "{asm}");
    assert!(asm.contains("OpAccessChain"), "{asm}");
    assert!(!asm.contains("metal2vulkan.buffer_address_word"), "{asm}");
    assert!(!asm.contains("OpConvertPtrToU"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_bda_as_data_pointer_intrinsic_is_device_address_passthrough() {
    // `air.get_data_pointer_instance_acceleration_structure(%p)` is modeled (paravirt AS ABI) as an
    // IDENTITY passthrough of its device-pointer argument. `%p` is loaded from a buffer (BDA-eligible),
    // so under BDA mode the intrinsic result aliases `%p`'s device address: the store copies it verbatim
    // and the field-offset deref reads through it as a PhysicalStorageBuffer pointer — the plain-BDA path.
    let ll = r#"
define void @k(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %p = load ptr addrspace(1), ptr addrspace(1) %in, align 8
  %d = call ptr addrspace(1) @air.get_data_pointer_instance_acceleration_structure(ptr addrspace(1) %p)
  store ptr addrspace(1) %d, ptr addrspace(1) %out, align 8
  %g = getelementptr inbounds i8, ptr addrspace(1) %d, i64 136
  %f = load float, ptr addrspace(1) %g, align 4
  store float %f, ptr addrspace(1) %out, align 4
  ret void
}

declare ptr addrspace(1) @air.get_data_pointer_instance_acceleration_structure(ptr addrspace(1))

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"in"}
"#;
    let spv = crate::native::emit_vulkan_spirv_all_buffers_raw_bda(ll).expect("bda emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm.contains("PhysicalStorageBuffer64"),
        "AS data-pointer passthrough must reach the device-address path:\n{asm}"
    );
    assert!(
        asm.contains("OpConvertUToPtr"),
        "expected a device-address deref:\n{asm}"
    );
}

#[test]
fn native_acceleration_structure_shadow_lowers_count_child_and_payload_store() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %as, ptr addrspace(1) %out, i32 %idx) {
entry:
  %count = call i32 @air.get_instance_count_instance_acceleration_structure(ptr addrspace(1) %as)
  store i32 %count, ptr addrspace(1) %out, align 4
  %child = call ptr addrspace(1) @air.get_primitive_acceleration_structure_instance_acceleration_structure(ptr addrspace(1) %as, i32 %idx)
  %child_slot = getelementptr inbounds i8, ptr addrspace(1) %out, i64 8
  store ptr addrspace(1) %child, ptr addrspace(1) %child_slot, align 8
  %bits = ptrtoint ptr addrspace(1) %child to i64
  %bits_slot = getelementptr inbounds i8, ptr addrspace(1) %out, i64 16
  store i64 %bits, ptr addrspace(1) %bits_slot, align 8
  ret void
}

declare i32 @air.get_instance_count_instance_acceleration_structure(ptr addrspace(1))
declare ptr addrspace(1) @air.get_primitive_acceleration_structure_instance_acceleration_structure(ptr addrspace(1), i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.instance_acceleration_structure", !"air.location_index", i32 8, i32 1, !"air.read", !"air.arg_type_name", !"acceleration_structure<instancing>", !"air.arg_name", !"as"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"idx"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_as_shadow_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Binding 8"), "{asm}");
    assert!(asm.contains("OpTypeInt 64"), "{asm}");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("OpPtrEqual"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_unused_primitive_acceleration_structure_needs_no_vulkan_binding() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %as, ptr addrspace(1) %out) {
entry:
  store i32 7, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.primitive_acceleration_structure", !"air.location_index", i32 5, i32 1, !"air.read", !"air.arg_type_name", !"acceleration_structure<>", !"air.arg_name", !"as"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_primitive_as_shadow_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("Binding 5"), "{asm}");
    assert!(asm.contains("Binding 0"), "{asm}");
}

#[test]
fn native_callback_free_single_instance_triangle_query_uses_shadow_binding() {
    let ll = r#"
define void @k(ptr addrspace(1) %out, ptr addrspace(1) %as, ptr addrspace(1) %table) {
entry:
  %hit = call { i32, float, i32, i32, ptr addrspace(1), i32, i32, <2 x float>, i1 } @air.intersect.instancing.triangle_data(<3 x float> <float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, <3 x float> <float 0.000000e+00, float 0.000000e+00, float -1.000000e+00>, float 0.000000e+00, float 1.000000e+01, ptr addrspace(1) %as, i32 255, ptr addrspace(1) %table, ptr null, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 -1, i32 0, i1 false, i1 false)
  %instance = extractvalue { i32, float, i32, i32, ptr addrspace(1), i32, i32, <2 x float>, i1 } %hit, 6
  store i32 %instance, ptr addrspace(1) %out, align 4
  ret void
}

declare { i32, float, i32, i32, ptr addrspace(1), i32, i32, <2 x float>, i1 } @air.intersect.instancing.triangle_data(<3 x float>, <3 x float>, float, float, ptr addrspace(1), i32, ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.instance_acceleration_structure", !"air.location_index", i32 5, i32 1, !"air.read", !"air.arg_type_name", !"acceleration_structure<instancing>", !"air.arg_name", !"as"}
!5 = !{i32 2, !"air.intersection_function_table", !"air.location_index", i32 6}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_single_instance_intersection_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Binding 5"), "{asm}");
    assert!(!asm.contains("Binding 6"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_callback_free_multi_level_query_returns_path_and_writes_ids() {
    let ll = r#"
define void @k(ptr addrspace(1) %out, ptr addrspace(1) %as, ptr addrspace(1) %table) {
entry:
  %instance_ids = alloca i32, align 4
  %user_instance_ids = alloca i32, align 4
  store i32 -1, ptr %instance_ids, align 4
  store i32 -1, ptr %user_instance_ids, align 4
  %hit = call { i32, float, i32, i32, ptr addrspace(1), i8 } @air.intersect.multi_level_instancing(<3 x float> <float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, <3 x float> <float 0.000000e+00, float 0.000000e+00, float -1.000000e+00>, float 0.000000e+00, float 1.000000e+01, ptr addrspace(1) %as, i32 255, ptr addrspace(1) %table, ptr null, i64 0, i8 2, ptr %instance_ids, ptr %user_instance_ids, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 -1, i32 0, i1 false, i1 false)
  %opaque = extractvalue { i32, float, i32, i32, ptr addrspace(1), i8 } %hit, 4
  %opaque_is_null = icmp eq ptr addrspace(1) %opaque, null
  %opaque_is_null32 = zext i1 %opaque_is_null to i32
  %path_length8 = extractvalue { i32, float, i32, i32, ptr addrspace(1), i8 } %hit, 5
  %path_length = zext i8 %path_length8 to i32
  %instance_id = load i32, ptr %instance_ids, align 4
  %user_instance_id = load i32, ptr %user_instance_ids, align 4
  store i32 %path_length, ptr addrspace(1) %out, align 4
  %out_instance = getelementptr i32, ptr addrspace(1) %out, i64 1
  store i32 %instance_id, ptr addrspace(1) %out_instance, align 4
  %out_user = getelementptr i32, ptr addrspace(1) %out, i64 2
  store i32 %user_instance_id, ptr addrspace(1) %out_user, align 4
  %out_opaque = getelementptr i32, ptr addrspace(1) %out, i64 3
  store i32 %opaque_is_null32, ptr addrspace(1) %out_opaque, align 4
  ret void
}

declare { i32, float, i32, i32, ptr addrspace(1), i8 } @air.intersect.multi_level_instancing(<3 x float>, <3 x float>, float, float, ptr addrspace(1), i32, ptr addrspace(1), ptr, i64, i8, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint4", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.instance_acceleration_structure", !"air.location_index", i32 5, i32 1, !"air.read", !"air.arg_type_name", !"acceleration_structure<instancing>", !"air.arg_name", !"as"}
!5 = !{i32 2, !"air.intersection_function_table", !"air.location_index", i32 6}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_multi_level_intersection_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Binding 5"), "{asm}");
    assert!(!asm.contains("Binding 6"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_callback_free_instance_world_space_data_uses_identity_transform() {
    let ll = r#"
define void @k(ptr addrspace(1) %out, ptr addrspace(1) %as, ptr addrspace(1) %table) {
entry:
  %hit = call { i32, float, i32, i32, ptr addrspace(1), i32, i32, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } @air.intersect.instancing.world_space_data(<3 x float> <float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, <3 x float> <float 0.000000e+00, float 0.000000e+00, float -1.000000e+00>, float 0.000000e+00, float 1.000000e+01, ptr addrspace(1) %as, i32 255, ptr addrspace(1) %table, ptr null, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 -1, i32 0, i1 false, i1 false)
  %world_to_object_x = extractvalue { i32, float, i32, i32, ptr addrspace(1), i32, i32, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } %hit, 7
  %xx = extractelement <3 x float> %world_to_object_x, i32 0
  store float %xx, ptr addrspace(1) %out, align 4
  ret void
}

declare { i32, float, i32, i32, ptr addrspace(1), i32, i32, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float>, <3 x float> } @air.intersect.instancing.world_space_data(<3 x float>, <3 x float>, float, float, ptr addrspace(1), i32, ptr addrspace(1), ptr, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.instance_acceleration_structure", !"air.location_index", i32 5, i32 1, !"air.read", !"air.arg_type_name", !"acceleration_structure<instancing>", !"air.arg_name", !"as"}
!5 = !{i32 2, !"air.intersection_function_table", !"air.location_index", i32 6}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_instance_world_space_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Binding 5"), "{asm}");
    assert!(!asm.contains("Binding 6"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_ignores_llvm_metadata_root_globals() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@llvm.used = appending global [1 x ptr] [ptr @helper], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x ptr] [ptr @helper], section "llvm.metadata"

define void @main() {
entry:
  ret void
}

define internal void @helper() {
entry:
  ret void
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(!asm.contains("llvm.used"), "{asm}");
    assert!(!asm.contains("llvm.compiler.used"), "{asm}");
}

#[test]
fn sanitized_ll_drops_llvm_metadata_root_globals() {
    let ll = r#"
target triple = "air64-apple-macosx"
@llvm.used = appending global [1 x ptr] [ptr @helper], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x ptr] [ptr @helper], section "llvm.metadata"

define void @main() {
entry:
  ret void
}

define internal void @helper() {
entry:
  ret void
}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_sanitize_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let src = tmp.join("metadata.ll");
    std::fs::write(&src, ll).expect("write fixture ll");
    let sanitized = tools::air_to_sanitized_ll(src.to_str().unwrap(), &tmp).expect("sanitize .ll");
    assert!(sanitized.contains(tools::VULKAN_TRIPLE), "{sanitized}");
    assert!(!sanitized.contains("@llvm.used"), "{sanitized}");
    assert!(!sanitized.contains("@llvm.compiler.used"), "{sanitized}");
}

#[test]
fn native_memcpy_to_typed_alloca_first_field_uses_copy_memory() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Light = type { <4 x float>, <4 x float>, <3 x float> }
%struct.Params = type { %struct.Light, i32 }

define void @main() {
entry:
  %src = alloca %struct.Light, align 16
  %dst = alloca %struct.Params, align 16
  %src_raw = bitcast ptr %src to ptr
  %dst_raw = bitcast ptr %dst to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %dst_raw, ptr %src_raw, i64 48, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_typed_memcpy_{}",
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
    assert!(asm.contains("OpCopyMemory"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("llvm.memcpy"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_memcpy_between_named_prefix_wrappers_lowers_both_directions() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Frame = type { %union.FrameUnion }
%union.FrameUnion = type { %struct.FrameInner }
%struct.FrameInner = type { <3 x float>, <3 x float>, <3 x float> }
%struct.Geometry = type { %struct.Frame, <3 x float> }

define void @main() {
entry:
  %frame = alloca %struct.Frame, align 16
  %geometry = alloca %struct.Geometry, align 16
  %frame_raw = bitcast ptr %frame to ptr
  %geometry_raw = bitcast ptr %geometry to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %geometry_raw, ptr %frame_raw, i64 48, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %frame_raw, ptr %geometry_raw, i64 48, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_named_prefix_memcpy_{}",
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
    assert_eq!(asm.matches("OpCopyMemory").count(), 2, "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("llvm.memcpy"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_memcpy_from_named_wrapper_to_bare_array_lowers_by_elements() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Matrix = type { [3 x <3 x float>] }

define void @main() {
entry:
  %wrapped = alloca %struct.Matrix, align 16
  %bare = alloca [3 x <3 x float>], align 16
  %wrapped_raw = bitcast ptr %wrapped to ptr
  %bare_raw = bitcast ptr %bare to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %bare_raw, ptr %wrapped_raw, i64 48, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_wrapper_to_array_memcpy_{}",
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
    assert_eq!(asm.matches("OpCopyMemory").count(), 1, "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("llvm.memcpy"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn native_memcpy_from_opaque_byval_array_preserves_explicit_pointee() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Wrapper = type { [3 x i64] }

define void @main(ptr readonly byval([3 x i64]) %src) {
entry:
  %dst = alloca %struct.Wrapper, align 8
  %field = getelementptr inbounds %struct.Wrapper, ptr %dst, i64 0, i32 0
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 dereferenceable(24) %field, ptr align 8 dereferenceable(24) %src, i64 24, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_byval_memcpy_{}",
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
    assert_eq!(asm.matches("OpCopyMemory").count(), 1, "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("llvm.memcpy"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn native_memcpy_struct_prefix_skips_trailing_padding_field() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Aggregate = type { <4 x float>, i32, float, [4 x i8] }

define void @main() {
entry:
  %src = alloca %struct.Aggregate, align 16
  %dst = alloca %struct.Aggregate, align 16
  %src_raw = bitcast ptr %src to ptr
  %dst_raw = bitcast ptr %dst to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr %dst_raw, ptr %src_raw, i64 24, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_prefix_memcpy_{}",
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
    assert_eq!(asm.matches("OpCopyMemory").count(), 3, "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("llvm.memcpy"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_raw_buffer_memcpy_preserves_byte_offsets() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @main(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
entry:
  %src_head = load i32, ptr addrspace(1) %src, align 4
  %src_off = getelementptr inbounds i8, ptr addrspace(1) %src, i64 136
  %dst_off = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 32
  tail call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) noundef align 4 dereferenceable(24) %dst_off, ptr addrspace(1) noundef align 8 dereferenceable(24) %src_off, i64 24, i1 false)
  ret void
}

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"dst"}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("OpCopyMemory"), "{asm}");
    let module = load_bytes(&spv).expect("load native spv");
    let constants = module
        .types_global_values
        .iter()
        .filter_map(|inst| match (inst.class.opcode, inst.operands.last()) {
            (Op::Constant, Some(Operand::LiteralBit32(value))) => Some(*value),
            _ => None,
        })
        .collect::<HashSet<_>>();
    for word_offset in [32, 52, 34, 39] {
        assert!(
            constants.contains(&word_offset),
            "missing word offset {word_offset} in {constants:?}\n{asm}"
        );
    }
}

#[test]
fn native_raw_buffer_zero_memset_clears_entire_byte_range() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

%Counter = type { i32, [12 x i8] }

define void @main(i32 %tid, ptr addrspace(1) %counters) {
entry:
  %idx = zext i32 %tid to i64
  %tail = getelementptr inbounds %Counter, ptr addrspace(1) %counters, i64 %idx, i32 1, i64 0
  tail call void @llvm.memset.p1.i64(ptr addrspace(1) noundef align 4 dereferenceable(12) %tail, i8 0, i64 12, i1 false)
  ret void
}

declare void @llvm.memset.p1.i64(ptr addrspace(1), i8, i64, i1)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !5, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Counter", !"air.arg_name", !"counters"}
!5 = !{i32 0, i32 4, i32 0, !"uint", !"value"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_raw_zero_memset_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("llvm.memset"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert_eq!(asm.matches("OpStore").count(), 12, "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_raw_buffer_memcpy_preserves_partial_byte_range() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

%Particle = type { [3 x float], half, half, [3 x float], half, i16, float, half, half }

define void @main(ptr addrspace(1) %src, ptr addrspace(1) %dst, i32 %tid) {
entry:
  %idx = zext i32 %tid to i64
  %src_vel = getelementptr inbounds %Particle, ptr addrspace(1) %src, i64 %idx, i32 3
  %src_raw = bitcast ptr addrspace(1) %src_vel to ptr addrspace(1)
  %dst_vel = getelementptr inbounds %Particle, ptr addrspace(1) %dst, i64 %idx, i32 3
  %dst_raw = bitcast ptr addrspace(1) %dst_vel to ptr addrspace(1)
  tail call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) noundef align 4 dereferenceable(14) %dst_raw, ptr addrspace(1) noundef align 4 dereferenceable(14) %src_raw, i64 14, i1 false)
  ret void
}

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 40, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Particle", !"air.arg_name", !"src"}
!4 = !{i32 0, i32 12, i32 0, !"packed_float3", !"position", i32 12, i32 2, i32 0, !"half", !"size", i32 14, i32 2, i32 0, !"half", !"angle", i32 16, i32 12, i32 0, !"packed_float3", !"velocity", i32 28, i32 2, i32 0, !"half", !"angularVelocity", i32 30, i32 2, i32 0, !"short", !"colorIndex", i32 32, i32 4, i32 0, !"float", !"depth", i32 36, i32 2, i32 0, !"half", !"wigglePhase", i32 38, i32 2, i32 0, !"half", !"wiggleFrequency"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 40, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Particle", !"air.arg_name", !"dst"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_partial_raw_memcpy_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("OpCopyMemory"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("llvm.memcpy"), "{asm}");
    assert!(asm.matches("OpAtomicAnd").count() >= 14, "{asm}");
    assert!(asm.matches("OpAtomicOr").count() >= 14, "{asm}");
    let module = load_bytes(&spv).expect("load native spv");
    let constants = module
        .types_global_values
        .iter()
        .filter_map(|inst| match (inst.class.opcode, inst.operands.last()) {
            (Op::Constant, Some(Operand::LiteralBit32(value))) => Some(*value),
            _ => None,
        })
        .collect::<HashSet<_>>();
    for byte_offset in [16, 17, 28, 29] {
        assert!(
            constants.contains(&byte_offset),
            "missing byte offset {byte_offset} in {constants:?}\n{asm}"
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

#[test]
fn native_typed_constant_memcpy_to_dynamic_raw_buffer_preserves_words() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

%struct.Params = type { [4 x float] }

define void @main(ptr addrspace(1) %bytes, ptr addrspace(2) %params) {
entry:
  %base = load i32, ptr addrspace(1) %bytes, align 4
  %base64 = zext i32 %base to i64
  %dyn = getelementptr inbounds i8, ptr addrspace(1) %bytes, i64 %base64
  %dst = getelementptr inbounds i8, ptr addrspace(1) %dyn, i64 32
  %src = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 0
  %src_raw = bitcast ptr addrspace(2) %src to ptr addrspace(2)
  tail call void @llvm.memcpy.p1.p2.i64(ptr addrspace(1) align 16 dereferenceable(16) %dst, ptr addrspace(2) align 16 dereferenceable(16) %src_raw, i64 16, i1 false)
  ret void
}

declare void @llvm.memcpy.p1.p2.i64(ptr addrspace(1), ptr addrspace(2), i64, i1)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"bytes"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!5 = !{!"air.struct_type_info", !6, i32 0, i32 4, i32 4, !"float", !"values"}
!6 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_typed_to_raw_memcpy_{}",
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
    assert!(!asm.contains("llvm.memcpy"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("OpCopyMemory"), "{asm}");
    assert_eq!(asm.matches("OpStore").count(), 4, "{asm}");
    assert!(asm.contains("OpBitcast"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_metadata_struct_memcpy_to_private_alloca_uses_copy_memory() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

%Eyes = type { [4 x <2 x float>], [4 x <2 x float>], i32 }

define void @main(ptr addrspace(2) %eyes) {
entry:
  %dst = alloca %Eyes, align 8
  %dst_raw = bitcast ptr %dst to ptr
  %src_raw = bitcast ptr addrspace(2) %eyes to ptr addrspace(2)
  tail call void @llvm.memcpy.p0.p2.i64(ptr align 8 dereferenceable(72) %dst_raw, ptr addrspace(2) align 8 dereferenceable(72) %src_raw, i64 72, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p2.i64(ptr, ptr addrspace(2), i64, i1)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 72, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 72, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"simple_lens_model_eyes", !"air.arg_name", !"eyes"}
!4 = !{i32 0, i32 8, i32 4, !"float2", !"leftEyes", i32 32, i32 8, i32 4, !"float2", !"rightEyes", i32 64, i32 4, i32 0, !"int", !"numValidEyes"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_metadata_struct_memcpy_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCopyMemory"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("llvm.memcpy"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_metadata_struct_memcpy_with_trailing_padding_copies_fields() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

%Matrix = type { [4 x <4 x float>] }
%Agg = type <{ %Matrix, <4 x float>, i32, float, float, [4 x i8] }>

define void @main(i32 %tid, ptr addrspace(2) %src) {
entry:
  %dst = alloca %Agg, align 16
  %idx = zext i32 %tid to i64
  %src_elem = getelementptr inbounds %Agg, ptr addrspace(2) %src, i64 %idx
  %dst_raw = bitcast ptr %dst to ptr
  %src_raw = bitcast ptr addrspace(2) %src_elem to ptr addrspace(2)
  tail call void @llvm.memcpy.p0.p2.i64(ptr align 16 dereferenceable(96) %dst_raw, ptr addrspace(2) align 16 dereferenceable(96) %src_raw, i64 96, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p2.i64(ptr, ptr addrspace(2), i64, i1)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 96, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 96, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"da_tile_aggregation_t", !"air.arg_name", !"src"}
!5 = !{i32 0, i32 64, i32 0, !"float4x4", !"AtA", i32 64, i32 16, i32 0, !"float4", !"Atb", i32 80, i32 4, i32 0, !"uint", !"pixel_cnt", i32 84, i32 4, i32 0, !"float", !"min_distance", i32 88, i32 4, i32 0, !"float", !"max_distance"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_metadata_struct_padding_memcpy_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    // The size-guarded GEP-source override keeps the LLVM `%Agg` view (trailing `[4 x i8]` pad
    // included), so the 96-byte memcpy covers all six members and lowers per-field: 4 matrix
    // columns + float4 + uint + 2 floats + 4 pad bytes = 12 OpCopyMemory. Copying the pad bytes
    // matches Apple's memcpy semantics (the byte count covers them).
    assert_eq!(asm.matches("OpCopyMemory").count(), 12, "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("llvm.memcpy"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_raw_metadata_memcpy_through_private_vector_struct_preserves_padding() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

%Corner = type { <2 x float>, float }

define void @main(ptr addrspace(1) %dst, ptr addrspace(1) %src, i32 %i, i32 %j) {
entry:
  %tmp = alloca %Corner, align 8
  %ip = zext i32 %i to i64
  %jp = zext i32 %j to i64
  %dst_i = getelementptr inbounds %Corner, ptr addrspace(1) %dst, i64 %ip
  %src_j = getelementptr inbounds %Corner, ptr addrspace(1) %src, i64 %jp
  %tmp_raw = bitcast ptr %tmp to ptr
  %dst_raw = bitcast ptr addrspace(1) %dst_i to ptr addrspace(1)
  %src_raw = bitcast ptr addrspace(1) %src_j to ptr addrspace(1)
  call void @llvm.memcpy.p0.p1.i64(ptr align 8 dereferenceable(16) %tmp_raw, ptr addrspace(1) align 8 dereferenceable(16) %dst_raw, i64 16, i1 false)
  call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 8 dereferenceable(16) %dst_raw, ptr addrspace(1) align 8 dereferenceable(16) %src_raw, i64 16, i1 false)
  call void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) align 8 dereferenceable(16) %src_raw, ptr align 8 dereferenceable(16) %tmp_raw, i64 16, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p1.i64(ptr, ptr addrspace(1), i64, i1)
declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)
declare void @llvm.memcpy.p1.p0.i64(ptr addrspace(1), ptr, i64, i1)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6, !7}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Corner", !"air.arg_name", !"dst"}
!4 = !{i32 0, i32 8, i32 0, !"float2", !"corner", i32 8, i32 4, i32 0, !"float", !"score"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Corner", !"air.arg_name", !"src"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!7 = !{i32 3, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"j"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_raw_vector_struct_memcpy_{}",
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
    assert!(!asm.contains("OpCopyMemory"), "{asm}");
    assert!(!asm.contains("llvm.memcpy"), "{asm}");
    assert!(asm.matches("OpStore").count() >= 8, "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_record_array_metadata_gep_remaps_padding_fields() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

%Constants = type <{ i32, i32, [8 x i8], <2 x i32>, i8, [3 x i8], i32 }>

define void @main(ptr addrspace(2) %constants, ptr addrspace(1) %out, i32 %idx) {
entry:
  %tail = getelementptr inbounds %Constants, ptr addrspace(2) %constants, i64 0, i32 6
  %tail_value = load i32, ptr addrspace(2) %tail, align 4
  %idx64 = zext i32 %idx to i64
  %record_head = getelementptr inbounds %Constants, ptr addrspace(2) %constants, i64 %idx64, i32 0
  %head_value = load i32, ptr addrspace(2) %record_head, align 4
  %sum = add i32 %tail_value, %head_value
  store i32 %sum, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 32, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 32, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"constants_t", !"air.arg_name", !"constants"}
!4 = !{i32 0, i32 4, i32 0, !"uint", !"head0", i32 4, i32 4, i32 0, !"uint", !"head1", i32 16, i32 8, i32 0, !"uint2", !"dims", i32 24, i32 1, i32 0, !"uchar", !"flag", i32 28, i32 4, i32 0, !"uint", !"tail"}
!5 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint*", !"air.arg_name", !"out"}
!6 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"idx"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_record_array_metadata_gep_padding_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let uint_ty = uint32_type_id(&asm);
    assert!(asm.contains(&format!("OpConstant  {uint_ty}  4")), "{asm}");
    assert!(!asm.contains(&format!("OpConstant  {uint_ty}  6")), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_padded_struct_memcpy_destination_becomes_raw() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%BBox = type { <3 x float>, <3 x float> }

define void @copy(i32 %tid, ptr addrspace(2) %index, ptr addrspace(1) %dst, ptr addrspace(1) %src) {
entry:
  %is_zero = icmp eq i32 %tid, 0
  br i1 %is_zero, label %copy_block, label %done

copy_block:
  %idx = load i32, ptr addrspace(2) %index, align 4
  %idx64 = zext i32 %idx to i64
  %dst_slot = getelementptr inbounds %BBox, ptr addrspace(1) %dst, i64 %idx64
  %dst_raw = bitcast ptr addrspace(1) %dst_slot to ptr addrspace(1)
  %src_raw = bitcast ptr addrspace(1) %src to ptr addrspace(1)
  tail call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) align 16 dereferenceable(32) %dst_raw, ptr addrspace(1) align 16 dereferenceable(32) %src_raw, i64 32, i1 false)
  br label %done

done:
  ret void
}

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)

!air.kernel = !{!0}
!0 = !{ptr @copy, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !7}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"index"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !6, !"air.arg_type_size", i32 32, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"_MPSAxisAlignedBoundingBox", !"air.arg_name", !"dst"}
!6 = !{i32 0, i32 16, i32 0, !"float3", !"min", i32 16, i32 16, i32 0, !"float3", !"max"}
!7 = !{i32 3, !"air.buffer", !"air.buffer_size", i32 32, !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !6, !"air.arg_type_size", i32 32, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"_MPSAxisAlignedBoundingBox", !"air.arg_name", !"src"}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("OpCopyMemory"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(asm.matches("OpStore").count() >= 8, "{asm}");
    let module = load_bytes(&spv).expect("load native spv");
    let constants = module
        .types_global_values
        .iter()
        .filter_map(|inst| match (inst.class.opcode, inst.operands.last()) {
            (Op::Constant, Some(Operand::LiteralBit32(value))) => Some(*value),
            _ => None,
        })
        .collect::<HashSet<_>>();
    for word_offset in [0, 4, 7, 8] {
        assert!(
            constants.contains(&word_offset),
            "missing word offset {word_offset} in {constants:?}\n{asm}"
        );
    }
}

#[test]
fn native_raw_buffer_vector_i32_store_splits_to_words() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %buf) {
entry:
  %head = load i32, ptr addrspace(1) %buf, align 4
  %typed = bitcast ptr addrspace(1) %buf to ptr addrspace(1)
  %slot = getelementptr inbounds <4 x i32>, ptr addrspace(1) %typed, i64 1
  %v0 = insertelement <4 x i32> poison, i32 %head, i64 0
  %v1 = insertelement <4 x i32> %v0, i32 20, i64 1
  %v2 = insertelement <4 x i32> %v1, i32 30, i64 2
  %v3 = insertelement <4 x i32> %v2, i32 40, i64 3
  store <4 x i32> %v3, ptr addrspace(1) %slot, align 16
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"buf"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_raw_vec_i32_store_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        whole_workgroup_options(),
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(asm.matches("OpCompositeExtract").count(), 4, "{asm}");
    assert_eq!(asm.matches("OpStore").count(), 4, "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_direct_wide_vector_buffer_store_uses_aggregate_pointer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %out) {
entry:
  store <6 x i32> zeroinitializer, ptr addrspace(1) %out, align 32
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 32, !"air.arg_type_align_size", i32 32, !"air.arg_type_name", !"uint6", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_direct_wide_vector_store_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let module = load_bytes(&spv).expect("load spv");
    let uint = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeInt
                && inst.operands == [Operand::LiteralBit32(32), Operand::LiteralBit32(0)]
        })
        .and_then(|inst| inst.result_id)
        .expect("uint type");
    let array = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeArray
                && inst.operands.first() == Some(&Operand::IdRef(uint))
        })
        .and_then(|inst| inst.result_id)
        .expect("uint array type");
    let ptr_array = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypePointer
                && inst.operands
                    == [
                        Operand::StorageClass(StorageClass::StorageBuffer),
                        Operand::IdRef(array),
                    ]
        })
        .and_then(|inst| inst.result_id)
        .expect("array storage pointer");
    let ptr_uint = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypePointer
                && inst.operands
                    == [
                        Operand::StorageClass(StorageClass::StorageBuffer),
                        Operand::IdRef(uint),
                    ]
        })
        .and_then(|inst| inst.result_id);
    let access_chain_types = module
        .all_inst_iter()
        .filter(|inst| inst.class.opcode == Op::AccessChain)
        .filter_map(|inst| inst.result_type)
        .collect::<Vec<_>>();
    assert!(access_chain_types.contains(&ptr_array));
    assert!(ptr_uint.is_none_or(|ptr_uint| !access_chain_types.contains(&ptr_uint)));
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_workgroup_global_array_scalar_store_uses_first_element() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@counts = internal unnamed_addr addrspace(3) global [33 x i16] undef, align 2

define void @k(ptr addrspace(1) %out) {
entry:
  store i16 0, ptr addrspace(3) @counts, align 2
  %slot = getelementptr inbounds [33 x i16], ptr addrspace(3) @counts, i64 0, i64 0
  %v = load i16, ptr addrspace(3) %slot, align 2
  store i16 %v, ptr addrspace(1) %out, align 2
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"ushort*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_workgroup_array_scalar_store_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load native spv");
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
    // The deterministic threadgroup zero-init prologue legitimately whole-stores OpConstantNull
    // into each Workgroup variable; only the kernel BODY's element stores must go through access
    // chains, so exclude the null-fill stores from the direct-store assertion.
    let null_ids = module
        .types_global_values
        .iter()
        .filter_map(|inst| (inst.class.opcode == Op::ConstantNull).then_some(inst.result_id?))
        .collect::<HashSet<_>>();
    let direct_workgroup_store = module
        .functions
        .iter()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .any(|inst| {
            inst.class.opcode == Op::Store
                && inst
                    .operands
                    .first()
                    .and_then(id_ref_operand)
                    .is_some_and(|id| workgroup_vars.contains(&id))
                && inst
                    .operands
                    .get(1)
                    .and_then(id_ref_operand)
                    .is_some_and(|id| !null_ids.contains(&id))
        });
    assert!(!direct_workgroup_store, "{asm}");
    let workgroup_access_chains = module
        .functions
        .iter()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .filter_map(|inst| {
            (inst.class.opcode == Op::InBoundsAccessChain
                && inst
                    .operands
                    .first()
                    .and_then(id_ref_operand)
                    .is_some_and(|id| workgroup_vars.contains(&id)))
            .then_some(inst.result_id?)
        })
        .collect::<HashSet<_>>();
    assert!(!workgroup_access_chains.is_empty(), "{asm}");
    assert!(
        module
            .functions
            .iter()
            .flat_map(|func| &func.blocks)
            .flat_map(|block| &block.instructions)
            .any(|inst| inst.class.opcode == Op::Store
                && inst
                    .operands
                    .first()
                    .and_then(id_ref_operand)
                    .is_some_and(|id| workgroup_access_chains.contains(&id))),
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
fn native_workgroup_global_array_vector_store_uses_first_element() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@faces = internal unnamed_addr addrspace(3) global [8 x <4 x float>] undef, align 16

define void @k(ptr addrspace(1) %out) {
entry:
  store <4 x float> zeroinitializer, ptr addrspace(3) @faces, align 16
  %slot = getelementptr inbounds [8 x <4 x float>], ptr addrspace(3) @faces, i64 0, i64 0
  %v = load <4 x float>, ptr addrspace(3) %slot, align 16
  store <4 x float> %v, ptr addrspace(1) %out, align 16
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float4*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_workgroup_array_vector_store_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load native spv");
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
    // The deterministic threadgroup zero-init prologue legitimately whole-stores OpConstantNull
    // into each Workgroup variable; only the kernel BODY's element stores must go through access
    // chains, so exclude the null-fill stores from the direct-store assertion.
    let null_ids = module
        .types_global_values
        .iter()
        .filter_map(|inst| (inst.class.opcode == Op::ConstantNull).then_some(inst.result_id?))
        .collect::<HashSet<_>>();
    let direct_workgroup_store = module
        .functions
        .iter()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .any(|inst| {
            inst.class.opcode == Op::Store
                && inst
                    .operands
                    .first()
                    .and_then(id_ref_operand)
                    .is_some_and(|id| workgroup_vars.contains(&id))
                && inst
                    .operands
                    .get(1)
                    .and_then(id_ref_operand)
                    .is_some_and(|id| !null_ids.contains(&id))
        });
    assert!(!direct_workgroup_store, "{asm}");
    let workgroup_access_chains = module
        .functions
        .iter()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .filter_map(|inst| {
            (inst.class.opcode == Op::InBoundsAccessChain
                && inst
                    .operands
                    .first()
                    .and_then(id_ref_operand)
                    .is_some_and(|id| workgroup_vars.contains(&id)))
            .then_some(inst.result_id?)
        })
        .collect::<HashSet<_>>();
    assert!(!workgroup_access_chains.is_empty(), "{asm}");
    assert!(
        module
            .functions
            .iter()
            .flat_map(|func| &func.blocks)
            .flat_map(|block| &block.instructions)
            .any(|inst| inst.class.opcode == Op::Store
                && inst
                    .operands
                    .first()
                    .and_then(id_ref_operand)
                    .is_some_and(|id| workgroup_access_chains.contains(&id))),
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
fn native_memcpy_from_void_buffer_to_typed_struct_copies_raw_words() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Draw = type { i32, i32, i32, i32, i32 }

define void @main(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
entry:
  %src_head_ptr = bitcast ptr addrspace(1) %src to ptr addrspace(1)
  %src_head = load i32, ptr addrspace(1) %src_head_ptr, align 4
  %dst_tail = getelementptr inbounds %Draw, ptr addrspace(1) %dst, i64 0, i32 4
  %dst_raw = bitcast ptr addrspace(1) %dst to ptr addrspace(1)
  tail call void @llvm.memcpy.p1.p1.i64(ptr addrspace(1) %dst_raw, ptr addrspace(1) %src, i64 20, i1 false)
  ret void
}

declare void @llvm.memcpy.p1.p1.i64(ptr addrspace(1), ptr addrspace(1), i64, i1)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"void", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 20, !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"MTLDrawIndexedPrimitivesIndirectArguments", !"air.arg_name", !"dst"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_raw_to_typed_memcpy_{}",
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
    assert_eq!(asm.matches("OpStore").count(), 5, "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("llvm.memcpy"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_char_buffer_struct_array_stores_lower_as_raw_words() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Src = type { i32, i32 }
%Dst = type { i32, i32, i32, i32 }

define void @main(i32 %tid, ptr addrspace(1) %src, ptr addrspace(1) %dst) {
entry:
  %src_header_byte = getelementptr inbounds i8, ptr addrspace(1) %src, i64 28
  %src_header = bitcast ptr addrspace(1) %src_header_byte to ptr addrspace(1)
  %count = load i32, ptr addrspace(1) %src_header, align 4
  %src_off_byte = getelementptr inbounds i8, ptr addrspace(1) %src, i64 96
  %src_off_ptr = bitcast ptr addrspace(1) %src_off_byte to ptr addrspace(1)
  %off = load i64, ptr addrspace(1) %src_off_ptr, align 8
  %src_base_byte = getelementptr inbounds i8, ptr addrspace(1) %src, i64 %off
  %src_base = bitcast ptr addrspace(1) %src_base_byte to ptr addrspace(1)
  %idx = zext i32 %tid to i64
  %src0p = getelementptr inbounds %Src, ptr addrspace(1) %src_base, i64 %idx, i32 0
  %src0 = load i32, ptr addrspace(1) %src0p, align 4
  %dst_base = bitcast ptr addrspace(1) %dst to ptr addrspace(1)
  %dst0p = getelementptr inbounds %Dst, ptr addrspace(1) %dst_base, i64 %idx, i32 0
  store i32 %src0, ptr addrspace(1) %dst0p, align 4
  %dst1p = getelementptr inbounds %Dst, ptr addrspace(1) %dst_base, i64 %idx, i32 1
  store i32 %count, ptr addrspace(1) %dst1p, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"src"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"dst"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_char_struct_store_{}",
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
    assert!(asm.contains("OpStore"), "{asm}");
    for line in asm
        .lines()
        .filter(|line| line.contains("OpInBoundsAccessChain"))
    {
        let operand_count = line
            .split_once("OpInBoundsAccessChain")
            .map(|(_, operands)| operands.split_whitespace().count())
            .unwrap_or(0);
        assert!(
            operand_count <= 4,
            "raw word access chain should not index through a scalar: {line}\n{asm}"
        );
    }
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
fn native_raw_i64_copy_uses_access_alignment_for_dynamic_byte_base() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @main(i32 %tid, ptr addrspace(1) %src, ptr addrspace(1) %dst) {
entry:
  %src_base = load i64, ptr addrspace(1) %src, align 8
  %dst_base = load i64, ptr addrspace(1) %dst, align 8
  %src_byte = getelementptr inbounds i8, ptr addrspace(1) %src, i64 %src_base
  %dst_byte = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 %dst_base
  %idx = zext i32 %tid to i64
  %src_item = getelementptr inbounds i64, ptr addrspace(1) %src_byte, i64 %idx
  %dst_item = getelementptr inbounds i64, ptr addrspace(1) %dst_byte, i64 %idx
  %word = load i64, ptr addrspace(1) %src_item, align 8
  store i64 %word, ptr addrspace(1) %dst_item, align 8
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"src"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"dst"}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpUDiv"), "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(asm.contains("OpStore"), "{asm}");
    assert!(!asm.contains("OpCopyMemory"), "{asm}");
}

#[test]
fn native_raw_unaligned_i32_store_splits_to_byte_atomics() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @StorePacked(i32 %idx, ptr addrspace(1) %bytes) {
entry:
  %idx64 = zext i32 %idx to i64
  %slot = getelementptr inbounds i8, ptr addrspace(1) %bytes, i64 %idx64
  %word = bitcast ptr addrspace(1) %slot to ptr addrspace(1)
  store i32 287454020, ptr addrspace(1) %word, align 1
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @StorePacked, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"idx"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"bytes"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_raw_unaligned_i32_store_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(asm.matches("OpAtomicAnd").count(), 4, "{asm}");
    assert_eq!(asm.matches("OpAtomicOr").count(), 4, "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_raw_unaligned_float_load_reassembles_bytes() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @LoadPacked(i32 %idx, ptr addrspace(1) %bytes, ptr addrspace(1) %out) {
entry:
  %idx64 = zext i32 %idx to i64
  %slot = getelementptr inbounds i8, ptr addrspace(1) %bytes, i64 %idx64
  %word = bitcast ptr addrspace(1) %slot to ptr addrspace(1)
  %v = load float, ptr addrspace(1) %word, align 1
  %dst = getelementptr inbounds float, ptr addrspace(1) %out, i64 %idx64
  store float %v, ptr addrspace(1) %dst, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @LoadPacked, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"idx"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"bytes"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_raw_unaligned_float_load_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpUDiv"), "{asm}");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpBitwiseOr"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_indirect_buffer_pointer_fields_use_raw_placeholder_pointers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Params = type { i16, ptr addrspace(1), i32 }

define void @main(i32 %tid, ptr addrspace(2) %params) {
entry:
  tail call void @helper(i32 %tid, ptr addrspace(2) %params)
  ret void
}

define internal void @helper(i32 %tid, ptr addrspace(2) %params) {
entry:
  %kindp = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 0
  %kind = load i16, ptr addrspace(2) %kindp
  %field = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 1
  %base = load ptr addrspace(1), ptr addrspace(2) %field
  %arg = getelementptr inbounds i8, ptr addrspace(1) %base, i64 4
  store i32 %tid, ptr addrspace(1) %arg, align 4
  tail call void @llvm.memcpy.p1.p2.i64(ptr addrspace(1) %arg, ptr addrspace(2) %params, i64 4, i1 false)
  tail call void @air.set_object_buffer_render_command.p1i8(ptr addrspace(1) null, i32 %tid, ptr addrspace(1) %arg, i32 2)
  ret void
}

declare void @air.set_object_buffer_render_command.p1i8(ptr addrspace(1), i32, ptr addrspace(1), i32)
declare void @llvm.memcpy.p1.p2.i64(ptr addrspace(1), ptr addrspace(2), i64, i1)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.thread_position_in_grid"}
!4 = !{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_name", !"params"}
!5 = !{i32 0, i32 2, i32 0, !"ushort", !"kind", !"air.indirect_argument", !6, i32 8, i32 8, i32 0, !"uchar", !"payload", !"air.indirect_argument", !7, i32 16, i32 4, i32 0, !"uint", !"count", !"air.indirect_argument", !8}
!6 = !{}
!7 = !{}
!8 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_indirect_buffer_{}",
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
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("llvm.memcpy"), "{asm}");
    assert!(!asm.contains("OpStore"), "{asm}");
    assert!(!asm.contains("RuntimeArray %_ptr"), "{asm}");
    assert!(!asm.contains("OpPtrAccessChain"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_direct_vector_store_infers_storage_buffer_element() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %out) {
entry:
  store <4 x i32> <i32 40, i32 80, i32 120, i32 255>, ptr addrspace(1) %out, align 16
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint4", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_direct_vector_store_{}",
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
    assert!(asm.contains("OpTypeRuntimeArray"), "{asm}");
    assert!(asm.contains("OpTypeVector") && asm.contains(" 4"), "{asm}");
    assert!(asm.contains("ArrayStride 16"), "{asm}");
    assert!(!asm.contains("OpTypeInt 8 0"), "{asm}");
    assert!(asm.lines().any(|line| line.contains("OpStore")), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_vector_stride_from_scalar_lane_scales_and_splits_store() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
entry:
  %srcp = getelementptr inbounds <4 x i16>, ptr addrspace(1) %src, i64 3
  %value = load <4 x i16>, ptr addrspace(1) %srcp, align 8
  %lane2 = getelementptr inbounds <4 x i16>, ptr addrspace(1) %dst, i64 0, i64 2
  %lane2_alias = bitcast ptr addrspace(1) %lane2 to ptr addrspace(1)
  %record3_lane2 = getelementptr inbounds <4 x i16>, ptr addrspace(1) %lane2_alias, i64 3
  store <4 x i16> %value, ptr addrspace(1) %record3_lane2, align 8
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ushort4", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ushort4", !"air.arg_name", !"dst"}
"#;
    let spv = crate::translate_native_no_retry(ll, Stage::Kernel).expect("primary translate");
    let module = load_bytes(&spv).expect("load native spv");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpIMul"), "vector stride not scaled:\n{asm}");
    assert_eq!(
        module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .filter(|inst| inst.class.opcode == Op::Store)
            .count(),
        4,
        "the vector payload must split into four scalar stores:\n{asm}"
    );
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        let tmp = std::env::temp_dir().join(format!(
            "metal2vulkan_vector_stride_scalar_lane_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&tmp);
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_raw_uint_struct_metadata_reconstructs_typed_layout() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::matrix" = type { [3 x <3 x float>] }
%struct.Params = type { <3 x float>, %"struct.metal::matrix" }

define void @k(ptr addrspace(1) %out, ptr addrspace(2) %params) {
entry:
  %basep = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 0
  %base = load <3 x float>, ptr addrspace(2) %basep, align 16
  %rowp = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 1, i32 0, i64 2
  %row = load <3 x float>, ptr addrspace(2) %rowp, align 16
  %sum = fadd <3 x float> %base, %row
  store <3 x float> %sum, ptr addrspace(1) %out, align 16
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float3", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 80, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!5 = !{i32 0, i32 16, i32 0, !"float3", !"base", i32 16, i32 48, i32 0, !"float3x3", !"matrix", i32 64, i32 1, i32 0, !"bool", !"enabled", i32 65, i32 1, i32 0, !"bool", !"mode", i32 66, i32 2, i32 0, !"ushort", !"count", i32 68, i32 1, i32 0, !"bool", !"flag"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_raw_uint_struct_metadata_{}",
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
    assert!(
        !asm.lines()
            .any(|line| line.contains("OpTypeRuntimeArray %10")),
        "{asm}"
    );
    assert!(
        asm.lines().any(|line| line.contains("OpTypeStruct %6 %8")),
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
fn native_kernel_void_tail_call_propagates_array_buffer_pointee() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @sum_sh(ptr addrspace(1) %out, ptr addrspace(1) %input) {
entry:
  tail call fastcc void @helper(ptr addrspace(1) %out, ptr addrspace(1) %input)
  ret void
}

define internal fastcc void @helper(ptr addrspace(1) %out, ptr addrspace(1) %input) {
entry:
  %rgb = getelementptr inbounds [3 x float], ptr addrspace(1) %input, i64 2, i64 1
  %v = load float, ptr addrspace(1) %rgb, align 4
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @sum_sh, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 12, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"packed_float3", !"air.arg_name", !"input"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_tail_call_array_buffer_{}",
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
    assert!(asm.contains("OpTypeRuntimeArray"), "{asm}");
    assert!(asm.contains("OpTypeArray"), "{asm}");
    assert!(asm.contains("ArrayStride 12"), "{asm}");
    assert!(!asm.contains("OpTypeInt 8 0"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_inline_raw_buffer_gep_argument_keeps_descriptor_root_and_offset() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"

define void @k(ptr addrspace(1) %out) {
entry:
  %shifted = getelementptr inbounds i32, ptr addrspace(1) %out, i64 4
  tail call fastcc void @helper(ptr addrspace(1) %shifted)
  ret void
}

define internal fastcc void @helper(ptr addrspace(1) %dst) {
entry:
  store i32 7, ptr addrspace(1) %dst, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_inline_raw_buffer_gep_arg_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(asm.matches("Binding 0").count(), 1, "{asm}");
    assert!(
        !asm.lines()
            .any(|line| line.contains("OpVariable %_ptr_Private_uint Private")),
        "offset helper store must remain descriptor-backed:\n{asm}"
    );
    assert!(
        asm.lines()
            .any(|line| line.contains("OpInBoundsAccessChain")),
        "descriptor-backed helper store must preserve the four-word offset:\n{asm}"
    );
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn native_indirect_helper_gep_drops_signed_zero_record_index() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Params = type <{ ptr addrspace(2), i32, i32, i32, [4 x i8], ptr addrspace(2), [20 x i8] }>

define void @main(i32 %tid, ptr addrspace(1) %params) {
entry:
  %tid64 = zext i32 %tid to i64
  %record = getelementptr inbounds %struct.Params, ptr addrspace(1) %params, i64 %tid64
  tail call void @helper(ptr addrspace(1) %record)
  ret void
}

define internal void @helper(ptr addrspace(1) %params) {
entry:
  %count_field = getelementptr inbounds %struct.Params, ptr addrspace(1) %params, i64 0, i32 5
  %count_buffer = load ptr addrspace(2), ptr addrspace(1) %count_field
  %count = load i32, ptr addrspace(2) %count_buffer
  %out = getelementptr inbounds %struct.Params, ptr addrspace(1) %params, i64 0, i32 6, i64 0
  store i32 %count, ptr addrspace(1) %out
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.thread_position_in_grid"}
!4 = !{i32 1, !"air.indirect_buffer", !"air.location_index", i32 4, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !5, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!5 = !{i32 0, i32 8, i32 0, !"uchar", !"inputArguments", !"air.indirect_argument", !6, i32 8, i32 4, i32 0, !"int", !"commandType", !"air.indirect_argument", !7, i32 12, i32 4, i32 0, !"uint", !"commandStride", !"air.indirect_argument", !8, i32 16, i32 4, i32 0, !"uint", !"commandIndex", !"air.indirect_argument", !9, i32 24, i32 8, i32 0, !"uint", !"commandCount", !"air.indirect_argument", !10, i32 32, i32 1, i32 20, !"uchar", !"outputArguments", !"air.indirect_argument", !11}
!6 = !{}
!7 = !{}
!8 = !{}
!9 = !{}
!10 = !{}
!11 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_indirect_helper_gep_{}",
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
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("OpPtrAccessChain"), "{asm}");
    assert!(
        !asm.lines()
            .any(|line| line.contains("OpInBoundsAccessChain") && line.contains("%uint_0 %uint_6")),
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
fn native_kernel_thread_position_uint3_binds_full_global_invocation_id() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main(<3 x i32> %tid) {
entry:
  %ok = icmp uge <3 x i32> %tid, zeroinitializer
  %all = tail call i1 @air.all.v3i1(<3 x i1> %ok)
  ret void
}

declare i1 @air.all.v3i1(<3 x i1>)

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_thread_position_uint3_{}",
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
    assert!(asm.contains("BuiltIn GlobalInvocationId"), "{asm}");
    assert!(asm.contains("OpUGreaterThanEqual"), "{asm}");
    assert!(!asm.contains("OpCompositeExtract %uint"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_air_quad_shuffle_float_lowers_to_quad_local_subgroup_shuffle() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(float %x, i16 %lane, ptr addrspace(1) %out) {
entry:
  %sx = tail call float @air.quad_shuffle.f32(float %x, i16 %lane)
  store float %sx, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.quad_shuffle.f32(float, i16)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_quad_shuffle_{}",
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
    assert!(asm.contains("BuiltIn SubgroupLocalInvocationId"), "{asm}");
    assert!(asm.contains("OpGroupNonUniformShuffle"), "{asm}");
    assert!(asm.contains("OpBitwiseAnd"), "{asm}");
    assert!(asm.contains("OpIAdd"), "{asm}");
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
fn native_selected_buffer_pointer_gep_preserves_typed_pointee() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(i32 %gid, ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out) {
entry:
  %idx = zext i32 %gid to i64
  %cond = icmp eq i32 %gid, 0
  %selected = select i1 %cond, ptr addrspace(1) %a, ptr addrspace(1) %b
  %src = getelementptr inbounds i32, ptr addrspace(1) %selected, i64 %idx
  %value = load i32, ptr addrspace(1) %src, align 4
  %dst = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %idx
  store i32 %value, ptr addrspace(1) %dst, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"a"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_selected_buffer_gep_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(!asm.contains("_ptr_StorageBuffer_uchar"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_selected_buffer_pointer_gep_store_replays_values_per_arm() {
    // The selected GEP has one concrete access chain per buffer. A direct pointer `OpSelect` before
    // the store is illegal when those chains root in distinct StorageBuffer bindings, even though
    // their pointee types agree. The store must stay in the value domain: load both values, select
    // the selected/new value per arm, then store through each original arm.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(i32 %gid, ptr addrspace(1) %a, ptr addrspace(1) %b) {
entry:
  %idx = zext i32 %gid to i64
  %cond = icmp eq i32 %gid, 0
  %selected = select i1 %cond, ptr addrspace(1) %a, ptr addrspace(1) %b
  %dst = getelementptr inbounds i32, ptr addrspace(1) %selected, i64 %idx
  store i32 %gid, ptr addrspace(1) %dst, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"a"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_selected_buffer_gep_store_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_native_no_retry(ll, Stage::Kernel).expect("primary translate");
    let module = load_bytes(&spv).expect("load native spv");
    let asm = disassemble(&spv).expect("disassemble");
    let pointer_selects = module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| {
            inst.class.opcode == Op::Select
                && inst
                    .result_type
                    .is_some_and(|ty| pointer_type_storage_class(&module, ty).is_some())
        })
        .count();
    let value_selects = module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| inst.class.opcode == Op::Select)
        .count();
    assert_eq!(pointer_selects, 0, "{asm}");
    assert!(value_selects >= 2, "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_selected_i8_buffer_bitcast_vector_load_uses_raw_arms() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %gid) {
entry:
  %cond = icmp eq i32 %gid, 0
  %idx = zext i32 %gid to i64
  %selected = select i1 %cond, ptr addrspace(1) %a, ptr addrspace(1) %b
  %slot = getelementptr inbounds i8, ptr addrspace(1) %selected, i64 %idx
  %wide = bitcast ptr addrspace(1) %slot to ptr addrspace(1)
  %value = load <4 x i16>, ptr addrspace(1) %wide, align 8
  store <4 x i16> %value, ptr addrspace(1) %out, align 8
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"b"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ushort4", !"air.arg_name", !"out"}
!6 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_selected_i8_buffer_bitcast_vector_load_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load native spv");
    let pointer_bitcasts = module
        .functions
        .iter()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| {
            inst.class.opcode == Op::Bitcast
                && inst
                    .result_type
                    .is_some_and(|ty| pointer_type_storage_class(&module, ty).is_some())
        })
        .count();
    assert_eq!(pointer_bitcasts, 0, "{asm}");
    let v4u16_loads = module
        .functions
        .iter()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| {
            inst.class.opcode == Op::Load
                && inst
                    .result_type
                    .is_some_and(|ty| is_unsigned_int_vector(&module, ty, 16, 4))
        })
        .count();
    assert_eq!(v4u16_loads, 0, "{asm}");
    let v4u16_constructs = module
        .functions
        .iter()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| {
            inst.class.opcode == Op::CompositeConstruct
                && inst
                    .result_type
                    .is_some_and(|ty| is_unsigned_int_vector(&module, ty, 16, 4))
        })
        .count();
    assert_eq!(v4u16_constructs, 2, "{asm}");
    assert!(
        module
            .functions
            .iter()
            .flat_map(|func| &func.blocks)
            .flat_map(|block| &block.instructions)
            .any(|inst| inst.class.opcode == Op::Select
                && inst
                    .result_type
                    .is_some_and(|ty| is_unsigned_int_vector(&module, ty, 16, 4))),
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
fn native_selected_i8_buffer_bitcast_vector_store_uses_selected_raw_pointer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %a, ptr addrspace(1) %b, i32 %gid) {
entry:
  %cond = icmp eq i32 %gid, 0
  %selected = select i1 %cond, ptr addrspace(1) %a, ptr addrspace(1) %b
  %offset = zext i32 %gid to i64
  %slot = getelementptr inbounds i8, ptr addrspace(1) %selected, i64 %offset
  %wide = bitcast ptr addrspace(1) %slot to ptr addrspace(1)
  store <4 x i16> <i16 1, i16 2, i16 3, i16 4>, ptr addrspace(1) %wide, align 8
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"b"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_selected_i8_buffer_bitcast_vector_store_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load native spv");
    let pointer_selects = module
        .functions
        .iter()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| {
            inst.class.opcode == Op::Select
                && inst
                    .result_type
                    .is_some_and(|ty| pointer_type_storage_class(&module, ty).is_some())
        })
        .count();
    assert_eq!(pointer_selects, 0, "{asm}");
    let selection_merges = module
        .functions
        .iter()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| inst.class.opcode == Op::SelectionMerge)
        .count();
    assert!(selection_merges > 0, "{asm}");
    assert_no_pointer_bitcasts(&spv);
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_existing_struct_buffer_uses_air_member_offsets() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

%struct.Params = type { <3 x float>, <2 x float>, [8 x i8] }

define void @k(ptr addrspace(2) readonly align 16 %params, ptr addrspace(1) %out) {
entry:
  %field = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 1
  %value = load <2 x float>, ptr addrspace(2) %field, align 16
  store <2 x float> %value, ptr addrspace(1) %out, align 8
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 32, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!4 = !{i32 0, i32 12, i32 0, !"float3", !"a", i32 16, i32 8, i32 0, !"float2", !"b"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"float2*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_existing_struct_offsets_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let kern = meta::parse_air_kernel_meta(ll);
    let emitted = crate::native::emit_vulkan_spirv_with_sidecar(
        ll,
        kern.as_ref(),
        Some("k"),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
    )
    .expect("emit with AIR layout sidecar");
    assert!(
        emitted
            .sidecar
            .air_struct_offsets
            .values()
            .any(|offsets| offsets == &[0, 16, 24]),
        "{:?}",
        emitted.sidecar.air_struct_offsets
    );
    assert_eq!(
        emitted.sidecar.air_struct_layout_mappings[0].status,
        crate::emit_sidecar::AirStructLayoutMappingStatus::MappedNatural
    );
    assert!(
        emitted
            .sidecar
            .buffer_access_offsets
            .iter()
            .any(|fact| fact.byte_offset == 16),
        "the field GEP must preserve its exact source byte address: {:?}",
        emitted.sidecar.buffer_access_offsets
    );
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpMemberDecorate"), "{asm}");
    assert!(asm.contains("Offset 16"), "{asm}");
    assert!(!asm.contains("Offset 12"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_record_array_buffer_clones_block_element_struct() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

%struct.Params = type { i32 }

define void @k(ptr addrspace(2) %direct, ptr addrspace(1) %records, ptr addrspace(1) %out, i32 %idx) {
entry:
  %dptr = getelementptr inbounds %struct.Params, ptr addrspace(2) %direct, i64 0, i32 0
  %d = load i32, ptr addrspace(2) %dptr, align 4
  %idx64 = zext i32 %idx to i64
  %rptr = getelementptr inbounds %struct.Params, ptr addrspace(1) %records, i64 %idx64, i32 0
  %r = load i32, ptr addrspace(1) %rptr, align 4
  %sum = add i32 %d, %r
  store i32 %sum, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5, !7, !8}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"direct"}
!4 = !{i32 0, i32 4, i32 0, !"uint", !"x"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !6, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"records"}
!6 = !{i32 0, i32 4, i32 0, !"uint", !"x"}
!7 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint*", !"air.arg_name", !"out"}
!8 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"idx"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_record_array_clone_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load spv");
    let block_types = module
        .annotations
        .iter()
        .filter_map(|inst| {
            if inst.class.opcode != Op::Decorate {
                return None;
            }
            match inst.operands.as_slice() {
                [Operand::IdRef(target), Operand::Decoration(Decoration::Block)] => Some(*target),
                _ => None,
            }
        })
        .collect::<HashSet<_>>();
    let runtime_array_block_elements = module
        .types_global_values
        .iter()
        .filter_map(|inst| match inst.operands.first() {
            Some(Operand::IdRef(elem)) if inst.class.opcode == Op::TypeRuntimeArray => Some(*elem),
            _ => None,
        })
        .filter(|elem| block_types.contains(elem))
        .collect::<Vec<_>>();
    assert!(asm.contains("OpTypeRuntimeArray"), "{asm}");
    assert!(
        runtime_array_block_elements.is_empty(),
        "runtime array elements must not be Block-decorated: {runtime_array_block_elements:?}\n{asm}"
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
fn native_existing_struct_offsets_skip_backend_padding_arrays() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

%struct.Params = type <{ i32, [4 x i8], <2 x i32>, float, float }>

define void @k(ptr addrspace(2) readonly align 8 %params, ptr addrspace(1) %out) {
entry:
  %field = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 2
  %value = load <2 x i32>, ptr addrspace(2) %field, align 8
  store <2 x i32> %value, ptr addrspace(1) %out, align 8
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 24, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!4 = !{i32 0, i32 4, i32 0, !"uint", !"a", i32 8, i32 8, i32 0, !"uint2", !"b", i32 16, i32 4, i32 0, !"float", !"c", i32 20, i32 4, i32 0, !"float", !"d"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"uint2*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_existing_struct_padding_offsets_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let transformed = load_bytes(&spv).expect("load transformed spv");
    // The size-guarded GEP-source override keeps the member-isomorphic LLVM struct, so the
    // backend `[4 x i8]` pad is a real member carrying its byte-cursor offset (4) and the real
    // fields keep their AIR offsets (0/8/16/20). GEP ordinals are verbatim (member 2 = uint2@8).
    let struct_ty = transformed
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == Op::TypeStruct && inst.operands.len() == 5)
        .and_then(|inst| inst.result_id)
        .unwrap_or_else(|| panic!("five-member params struct\n{asm}"));
    let mut offsets = vec![None; 5];
    for inst in &transformed.annotations {
        if inst.class.opcode != Op::MemberDecorate {
            continue;
        }
        let [Operand::IdRef(target), Operand::LiteralBit32(member), Operand::Decoration(Decoration::Offset), Operand::LiteralBit32(offset)] =
            inst.operands.as_slice()
        else {
            continue;
        };
        if *target == struct_ty && (*member as usize) < offsets.len() {
            offsets[*member as usize] = Some(*offset);
        }
    }
    assert_eq!(
        offsets,
        vec![Some(0), Some(4), Some(8), Some(16), Some(20)],
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
fn native_existing_struct_offsets_place_unaligned_padding_at_byte_cursor() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

%struct.Params = type <{ i32, i8, [3 x i8], <2 x i32>, float }>

define void @k(ptr addrspace(2) readonly align 8 %params, ptr addrspace(1) %out) {
entry:
  %field = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 3
  %value = load <2 x i32>, ptr addrspace(2) %field, align 8
  store <2 x i32> %value, ptr addrspace(1) %out, align 8
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 20, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!4 = !{i32 0, i32 4, i32 0, !"uint", !"a", i32 4, i32 1, i32 0, !"uchar", !"flag", i32 8, i32 8, i32 0, !"uint2", !"b", i32 16, i32 4, i32 0, !"float", !"c"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"uint2*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_existing_struct_unaligned_padding_offsets_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let transformed = load_bytes(&spv).expect("load transformed spv");
    let struct_ty = transformed
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == Op::TypeStruct && inst.operands.len() == 5)
        .and_then(|inst| inst.result_id)
        .expect("five-member params struct");
    let mut offsets = vec![None; 5];
    for inst in &transformed.annotations {
        if inst.class.opcode != Op::MemberDecorate {
            continue;
        }
        let [Operand::IdRef(target), Operand::LiteralBit32(member), Operand::Decoration(Decoration::Offset), Operand::LiteralBit32(offset)] =
            inst.operands.as_slice()
        else {
            continue;
        };
        if *target == struct_ty && (*member as usize) < offsets.len() {
            offsets[*member as usize] = Some(*offset);
        }
    }
    assert_eq!(
        offsets,
        vec![Some(0), Some(4), Some(5), Some(8), Some(16)],
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
fn native_kernel_local_size_option_updates_execution_and_builtin() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(<2 x i32> %threads) {
entry:
  %sx = extractelement <2 x i32> %threads, i64 0
  %sy = extractelement <2 x i32> %threads, i64 1
  %sum = add i32 %sx, %sy
  %ok = icmp uge i32 %sum, 0
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint2", !"air.arg_name", !"threads"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_local_size_option_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions {
            kernel_local_size: [256, 2, 1],
            simd_cluster32: false,
            ..passes::TransformOptions::default()
        },
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("LocalSize 256 2 1"), "{asm}");
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.contains("256")),
        "{asm}"
    );
    assert!(
        asm.lines()
            .any(|line| line.contains("OpConstant") && line.contains("2")),
        "{asm}"
    );
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
fn native_raw_stores_check_wide_byte_offset_before_u32_narrowing() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %offsets, ptr addrspace(1) %dst) {
entry:
  %offset = load i64, ptr addrspace(1) %offsets, align 8
  %target = getelementptr inbounds i8, ptr addrspace(1) %dst, i64 %offset
  store i32 7, ptr addrspace(1) %target, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"offsets"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"dst"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_wide_raw_store_guard_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let unchecked = crate::translate_native_no_retry(ll, Stage::Kernel).expect("primary emit");
    let unchecked_module = load_bytes(&unchecked).expect("primary module loads");
    assert!(
        crate::native::module_has_wide_raw_store_guard(&unchecked_module),
        "wide raw store must retain its structural robust-store guard"
    );
    let subword_ll = ll.replacen("store i32 7,", "store i16 7,", 1);
    let subword_unchecked =
        crate::translate_native_no_retry(&subword_ll, Stage::Kernel).expect("subword primary emit");
    let subword_asm = disassemble(&subword_unchecked).expect("disassemble subword store");
    assert!(subword_asm.contains("OpAtomicAnd"), "{subword_asm}");
    assert!(subword_asm.contains("OpAtomicOr"), "{subword_asm}");
    let subword_module = load_bytes(&subword_unchecked).expect("subword primary module loads");
    assert!(
        crate::native::module_has_wide_raw_store_guard(&subword_module),
        "wide raw subword store must retain its structural robust-store guard"
    );
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpULessThanEqual"), "{asm}");
    assert!(asm.contains("OpSelectionMerge"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_threadgroup_struct_record_memcpy_scalarizes_to_leaf_indices() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Tile = type { i32, i32, i32, i32 }

define void @k(ptr addrspace(3) %scratch, i32 %i) {
entry:
  %idx = zext i32 %i to i64
  %src_idx = add i64 %idx, 1
  %dst = getelementptr inbounds %struct.Tile, ptr addrspace(3) %scratch, i64 %idx
  %src = getelementptr inbounds %struct.Tile, ptr addrspace(3) %scratch, i64 %src_idx
  tail call void @llvm.memcpy.p3.p3.i64(ptr addrspace(3) %dst, ptr addrspace(3) %src, i64 16, i1 false)
  ret void
}

declare void @llvm.memcpy.p3.p3.i64(ptr addrspace(3), ptr addrspace(3), i64, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.struct_type_info", !5, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Tile", !"air.arg_name", !"scratch"}
!4 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!5 = !{i32 0, i32 4, i32 0, !"uint", !"a", i32 4, i32 4, i32 0, !"uint", !"b", i32 8, i32 4, i32 0, !"uint", !"c", i32 12, i32 4, i32 0, !"uint", !"d"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_threadgroup_struct_memcpy_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load spv");
    let workgroup_var = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::Variable
                && inst.operands.first() == Some(&Operand::StorageClass(StorageClass::Workgroup))
        })
        .and_then(|inst| inst.result_id)
        .expect("workgroup var");
    let array_ty = variable_pointee_type(&module, workgroup_var).expect("workgroup array type");
    let elem_ty = module
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == Op::TypeArray && inst.result_id == Some(array_ty))
        .and_then(|inst| inst.operands.first())
        .and_then(|operand| match operand {
            Operand::IdRef(elem_ty) => Some(*elem_ty),
            _ => None,
        })
        .expect("workgroup array element type");
    let member_types = module
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == Op::TypeStruct && inst.result_id == Some(elem_ty))
        .map(|inst| {
            inst.operands
                .iter()
                .filter_map(id_ref_operand)
                .collect::<Vec<_>>()
        })
        .expect("workgroup array element struct type");
    assert_eq!(member_types.len(), 4, "{asm}");
    assert!(
        member_types.iter().all(|ty| *ty == member_types[0]),
        "{asm}"
    );
    let uint_ptr = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypePointer
                && inst.operands
                    == [
                        Operand::StorageClass(StorageClass::Workgroup),
                        Operand::IdRef(member_types[0]),
                    ]
        })
        .and_then(|inst| inst.result_id)
        .expect("workgroup member pointer type");
    let constants = module
        .all_inst_iter()
        .filter_map(|inst| {
            if inst.class.opcode != Op::Constant {
                return None;
            }
            match (inst.result_id, inst.operands.first()) {
                (Some(id), Some(Operand::LiteralBit32(value))) => Some((id, *value)),
                _ => None,
            }
        })
        .collect::<HashMap<_, _>>();
    let member_chains = module
        .functions
        .iter()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .filter_map(|inst| {
            if !matches!(inst.class.opcode, Op::AccessChain | Op::InBoundsAccessChain)
                || inst.result_type != Some(uint_ptr)
                || inst.operands.first() != Some(&Operand::IdRef(workgroup_var))
                || inst.operands.len() != 3
            {
                return None;
            }
            inst.operands
                .get(2)
                .and_then(id_ref_operand)
                .and_then(|id| constants.get(&id).copied())
        })
        .collect::<HashSet<_>>();
    let expected_members = [0, 1, 2, 3].into_iter().collect::<HashSet<_>>();
    assert_eq!(member_chains, expected_members, "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(!asm.contains("llvm.memcpy"), "{asm}");
    assert!(asm.matches("OpLoad").count() >= 4, "{asm}");
    assert!(asm.matches("OpStore").count() >= 4, "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_threadgroup_struct_member_store_splits_flattened_record_index() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Tile = type { i32, i32, i32, i32 }

define void @k(ptr addrspace(3) %scratch, i32 %i) {
entry:
  %idx = zext i32 %i to i64
  %field = getelementptr inbounds %struct.Tile, ptr addrspace(3) %scratch, i64 %idx, i32 1
  store i32 %i, ptr addrspace(3) %field, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.struct_type_info", !5, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Tile", !"air.arg_name", !"scratch"}
!4 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!5 = !{i32 0, i32 4, i32 0, !"uint", !"a", i32 4, i32 4, i32 0, !"uint", !"b", i32 8, i32 4, i32 0, !"uint", !"c", i32 12, i32 4, i32 0, !"uint", !"d"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_threadgroup_struct_member_flat_index_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load spv");
    let uint = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeInt
                && inst.operands == [Operand::LiteralBit32(32), Operand::LiteralBit32(0)]
        })
        .and_then(|inst| inst.result_id)
        .expect("uint type");
    let uint_ptr = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypePointer
                && inst.operands
                    == [
                        Operand::StorageClass(StorageClass::Workgroup),
                        Operand::IdRef(uint),
                    ]
        })
        .and_then(|inst| inst.result_id)
        .expect("workgroup uint pointer");
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
    let has_leaf_chain = module
        .functions
        .iter()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .any(|inst| {
            matches!(inst.class.opcode, Op::AccessChain | Op::InBoundsAccessChain)
                && inst.result_type == Some(uint_ptr)
                && inst
                    .operands
                    .first()
                    .and_then(id_ref_operand)
                    .is_some_and(|base| workgroup_vars.contains(&base))
                && inst.operands.len() == 3
        });
    assert!(has_leaf_chain, "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_kernel_threadgroup_param_uses_pointer_addrspace_without_metadata_address_space() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(3) %temp, i32 %i) {
entry:
  %idx = zext i32 %i to i64
  %value = getelementptr inbounds float, ptr addrspace(3) %temp, i64 %idx
  store float 1.000000e+00, ptr addrspace(3) %value, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"temp"}
!4 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_threadgroup_addrspace_fallback_{}",
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
fn native_raw_pointer_load_eq_null_uses_payload_nullness() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Params = type { ptr addrspace(1), i32 }

define void @main(ptr addrspace(2) %params) {
entry:
  %field = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 0
  %out = load ptr addrspace(1), ptr addrspace(2) %field
  %isnull = icmp eq ptr addrspace(1) %out, null
  br i1 %isnull, label %done, label %write

write:
  store i32 7, ptr addrspace(1) %out, align 4
  br label %done

done:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_name", !"params"}
!4 = !{i32 0, i32 8, i32 0, !"uint", !"out", !"air.indirect_argument", !5, i32 8, i32 4, i32 0, !"uint", !"tag", !"air.indirect_argument", !6}
!5 = !{}
!6 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_raw_ptr_nullness_{}",
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
    assert!(asm.contains("OpIEqual"), "{asm}");
    assert!(asm.contains("OpLogicalAnd"), "{asm}");
    assert!(!asm.contains("OpPtrEqual"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_raw_pointer_loads_compare_serialized_payloads() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Params = type { ptr addrspace(1), ptr addrspace(1) }

define void @main(ptr addrspace(2) %params) {
entry:
  %a_field = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 0
  %a = load ptr addrspace(1), ptr addrspace(2) %a_field
  %b_field = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 1
  %b = load ptr addrspace(1), ptr addrspace(2) %b_field
  %same = icmp eq ptr addrspace(1) %a, %b
  br i1 %same, label %done, label %check_different

check_different:
  %different = icmp ne ptr addrspace(1) %a, %b
  br i1 %different, label %write, label %done

write:
  store i32 7, ptr addrspace(1) %a, align 4
  br label %done

done:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 16, !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_name", !"params"}
!4 = !{i32 0, i32 8, i32 0, !"uint", !"a", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"uint", !"b", !"air.indirect_argument", !6}
!5 = !{}
!6 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_raw_ptr_payload_eq_{}",
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
    assert!(asm.matches("OpIEqual").count() >= 6, "{asm}");
    assert!(asm.matches("OpLogicalAnd").count() >= 3, "{asm}");
    assert!(asm.contains("OpLogicalNot"), "{asm}");
    assert!(!asm.contains("OpPtrEqual"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_direct_pointer_param_icmp_folds_identity() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i1 @same(ptr addrspace(1) %a) {
entry:
  %c = icmp eq ptr addrspace(1) %a, %a
  ret i1 %c
}

define i1 @distinct_eq(ptr addrspace(1) %a, ptr addrspace(1) %b) {
entry:
  %c = icmp eq ptr addrspace(1) %a, %b
  ret i1 %c
}

define i1 @distinct_ne(ptr addrspace(1) %a, ptr addrspace(1) %b) {
entry:
  %c = icmp ne ptr addrspace(1) %a, %b
  ret i1 %c
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpConstantTrue"), "{asm}");
    assert!(asm.contains("OpConstantFalse"), "{asm}");
    assert!(!asm.contains("OpIEqual"), "{asm}");
    assert!(!asm.contains("OpINotEqual"), "{asm}");
}

#[test]
fn native_pointer_icmp_compares_flattened_gep_provenance_indices() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@rot = internal addrspace(2) constant [2 x [4 x i32]] [[4 x i32] [i32 13, i32 15, i32 26, i32 6], [4 x i32] [i32 17, i32 29, i32 16, i32 24]], align 4

define i32 @walk(i32 %row) {
entry:
  %masked = and i32 %row, 1
  %idx = zext i32 %masked to i64
  %end = getelementptr inbounds [2 x [4 x i32]], ptr addrspace(2) @rot, i64 0, i64 %idx, i64 4
  %begin = getelementptr inbounds [2 x [4 x i32]], ptr addrspace(2) @rot, i64 0, i64 %idx, i64 0
  br label %loop

loop:
  %p = phi ptr addrspace(2) [ %begin, %entry ], [ %next, %body ]
  %acc = phi i32 [ 0, %entry ], [ %sum, %body ]
  %value = load i32, ptr addrspace(2) %p, align 4
  %sum = add i32 %acc, %value
  %next = getelementptr inbounds i32, ptr addrspace(2) %p, i64 1
  %done = icmp eq ptr addrspace(2) %next, %end
  br i1 %done, label %exit, label %body

body:
  br label %loop

exit:
  ret i32 %sum
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpIEqual"), "{asm}");
    assert!(asm.contains("OpLogicalAnd"), "{asm}");
    assert!(!asm.contains("OpPtrEqual"), "{asm}");
}

#[test]
fn native_pointer_icmp_reserves_forward_gep_phi_root_provenance() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@rot = internal addrspace(2) constant [2 x [4 x i32]] [[4 x i32] [i32 13, i32 15, i32 26, i32 6], [4 x i32] [i32 17, i32 29, i32 16, i32 24]], align 4

define i32 @walk(i64 %row) {
entry:
  %end = getelementptr inbounds [2 x [4 x i32]], ptr addrspace(2) @rot, i64 0, i64 %row, i64 4
  %start = getelementptr inbounds [2 x [4 x i32]], ptr addrspace(2) @rot, i64 0, i64 %row, i64 0
  br label %loop

loop:
  %p = phi ptr addrspace(2) [ %next, %loop ], [ %start, %entry ]
  %acc = phi i32 [ %value, %loop ], [ 0, %entry ]
  %value = load i32, ptr addrspace(2) %p, align 4
  %next = getelementptr inbounds i32, ptr addrspace(2) %p, i64 1
  %done = icmp eq ptr addrspace(2) %next, %end
  br i1 %done, label %exit, label %loop

exit:
  ret i32 %acc
}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_forward_gep_phi_root_icmp_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble");
    assert!(asm.contains("OpIEqual"), "{asm}");
    assert!(!asm.contains("OpPtrEqual"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_function_pointer_icmp_compares_loop_cursor_to_end_pointer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Box = type { [3 x float], [3 x float], float }

define void @k() {
entry:
  %arr = alloca [2 x %struct.Box], align 4
  %start = getelementptr inbounds [2 x %struct.Box], ptr %arr, i64 0, i64 0
  %end = getelementptr inbounds [2 x %struct.Box], ptr %arr, i64 0, i64 2
  br label %loop

loop:
  %p = phi ptr [ %start, %entry ], [ %next, %loop ]
  %f = getelementptr inbounds %struct.Box, ptr %p, i64 0, i32 2
  store float 1.0, ptr %f, align 4
  %next = getelementptr inbounds %struct.Box, ptr %p, i64 1
  %done = icmp eq ptr %next, %end
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_function_pointer_loop_end_icmp_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble");
    assert!(asm.contains("OpIEqual"), "{asm}");
    assert!(!asm.contains("OpPtrEqual"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_entry_pointer_param_eq_null_uses_bound_resource_nullness() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i1 @k(ptr addrspace(2) %maybe) {
entry:
  %isnull = icmp eq ptr addrspace(2) %maybe, null
  ret i1 %isnull
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"maybe"}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpConstantFalse"), "{asm}");
    assert!(!asm.contains("OpPtrEqual"), "{asm}");
}

#[test]
fn native_alloca_eq_null_uses_intrinsic_nonnullness() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i1 @k() {
entry:
  %slot = alloca i32, align 4
  %isnull = icmp eq ptr %slot, null
  ret i1 %isnull
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpConstantFalse"), "{asm}");
    assert!(!asm.contains("OpPtrEqual"), "{asm}");
}

#[test]
fn native_literal_null_eq_null_folds_true() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i1 @k() {
entry:
  %isnull = icmp eq ptr null, null
  ret i1 %isnull
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpConstantTrue"), "{asm}");
    assert!(!asm.contains("OpPtrEqual"), "{asm}");
}

#[test]
fn typed_inline_keeps_helper_parameter_semantics_until_emission_finishes() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i1 @k() {
entry:
  %slot = alloca i32, align 4
  %same = call i1 @same_pointer(ptr %slot, ptr %slot)
  ret i1 %same
}

define internal i1 @same_pointer(ptr %left, ptr %right) {
entry:
  %same = icmp eq ptr %left, %right
  ret i1 %same
}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let module = load_bytes(&spv).expect("load native spv");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(
        module
            .functions
            .iter()
            .filter(|function| !function.blocks.is_empty())
            .count(),
        1,
        "the helper body is typed-inlined before emission"
    );
    assert!(asm.contains("OpConstantFalse"), "{asm}");
    let definitions = module
        .all_inst_iter()
        .filter_map(|instruction| instruction.result_id)
        .collect::<HashSet<_>>();
    assert!(
        module.all_inst_iter().all(|instruction| {
            instruction.operands.iter().all(|operand| match operand {
                Operand::IdRef(id) => definitions.contains(id),
                _ => true,
            })
        }),
        "deferred helper-parameter ids must be fully substituted"
    );
}

#[test]
fn typed_inline_materializes_pointer_field_gep_for_helper_parameter_store() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Wrap = type { ptr addrspace(1), i64 }

define void @k(ptr addrspace(1) %src) {
entry:
  %wrap = alloca %Wrap, align 8
  call void @init(ptr %wrap, ptr addrspace(1) %src)
  ret void
}

define internal void @init(ptr %w, ptr addrspace(1) %p) {
entry:
  %field = getelementptr inbounds %Wrap, ptr %w, i64 0, i32 0
  store ptr addrspace(1) %p, ptr %field, align 8
  ret void
}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let module = load_bytes(&spv).expect("load native spv");
    assert_eq!(
        module
            .functions
            .iter()
            .filter(|function| !function.blocks.is_empty())
            .count(),
        1,
        "the helper body is typed-inlined before emission"
    );
    let definitions = module
        .all_inst_iter()
        .filter_map(|instruction| instruction.result_id)
        .collect::<HashSet<_>>();
    assert!(
        module.all_inst_iter().all(|instruction| {
            instruction.operands.iter().all(|operand| match operand {
                Operand::IdRef(id) => definitions.contains(id),
                _ => true,
            })
        }),
        "inlined pointer-field store must not reference a missing helper GEP"
    );
}

#[test]
fn typed_inline_records_storage_for_extracted_pointer_field() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Ext = type { ptr addrspace(2), ptr addrspace(2), ptr addrspace(2) }
%Entity = type { float, i32 }

define void @k(ptr addrspace(2) %a, ptr addrspace(2) %b, ptr addrspace(2) %c, ptr addrspace(1) %out) {
entry:
  %ext0 = insertvalue %Ext poison, ptr addrspace(2) %a, 0
  %ext1 = insertvalue %Ext %ext0, ptr addrspace(2) %b, 1
  %ext2 = insertvalue %Ext %ext1, ptr addrspace(2) %c, 2
  %value = call i32 @read_entity(%Ext %ext2)
  store i32 %value, ptr addrspace(1) %out, align 4
  ret void
}

define internal i32 @read_entity(%Ext %ext) {
entry:
  %entity = extractvalue %Ext %ext, 2
  %field = getelementptr inbounds %Entity, ptr addrspace(2) %entity, i64 0, i32 1
  %value = load i32, ptr addrspace(2) %field, align 4
  ret i32 %value
}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let module = load_bytes(&spv).expect("load native SPIR-V");
    assert_eq!(
        module
            .functions
            .iter()
            .filter(|function| !function.blocks.is_empty())
            .count(),
        1,
        "the helper body is typed-inlined before emission"
    );
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCompositeExtract"), "{asm}");
}

#[test]
fn native_by_value_buffer_member_keeps_llvm_leading_zero_during_flattening() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%View = type { [944 x i8], i32 }
%Wrapper = type { ptr addrspace(2) }

define void @k(ptr addrspace(2) %view, ptr addrspace(1) %out) {
entry:
  %wrapped = insertvalue %Wrapper poison, ptr addrspace(2) %view, 0
  %value = call fastcc i32 @read_member(%Wrapper %wrapped)
  store i32 %value, ptr addrspace(1) %out, align 4
  ret void
}

define internal fastcc i32 @read_member(%Wrapper %wrapped) {
entry:
  %view = extractvalue %Wrapper %wrapped, 0
  %field = getelementptr inbounds %View, ptr addrspace(2) %view, i64 0, i32 1
  %value = load i32, ptr addrspace(2) %field, align 4
  ret i32 %value
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 948, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 948, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"View", !"air.arg_name", !"view"}
!4 = !{i32 0, i32 1, i32 944, !"uchar", !"padding", i32 944, i32 4, i32 0, !"uint", !"value"}
!5 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_by_value_buffer_member_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp)
        .expect("translate by-value buffer member");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        !asm.contains("OpPtrAccessChain %_ptr_StorageBuffer_View"),
        "{asm}"
    );
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn typed_inline_retains_pruned_helper_type_capability() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k() {
entry:
  %slot = alloca i32, align 4
  %same = call ptr @identity(ptr %slot)
  ret void
}

define internal ptr @identity(ptr %pointer) {
entry:
  ret ptr %pointer
}
"#;
    let module =
        load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native SPIR-V");
    assert!(
        module.capabilities.iter().any(|instruction| {
            matches!(
                instruction.operands.as_slice(),
                [Operand::Capability(Capability::Int8)]
            )
        }),
        "pruning a helper must retain the type capability its residual emission requested"
    );
    assert_eq!(
        module
            .functions
            .iter()
            .filter(|function| !function.blocks.is_empty())
            .count(),
        1,
        "the capability is retained after the helper body is pruned"
    );
}

#[test]
fn native_internal_pointer_param_eq_null_uses_callsite_nonnullness() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %out) {
entry:
  %slot = alloca i32, align 4
  call fastcc void @nonnull_helper(ptr %slot)
  call fastcc void @nullable_helper(ptr null)
  ret void
}

define internal fastcc void @nonnull_helper(ptr %p) {
entry:
  %isnull = icmp eq ptr %p, null
  ret void
}

define internal fastcc void @nullable_helper(ptr %p) {
entry:
  %isnull = icmp eq ptr %p, null
  ret void
}
"#;
    let ir = super::super::ir::LlModule::parse(ll).expect("parse");
    let emitter = Emitter::new(ir.clone());
    let nonnull = emitter
        .infer_function_param_nonnull(&ir.functions)
        .expect("infer nonnull params");
    assert!(nonnull.contains(&("nonnull_helper".to_string(), 0)));
    assert!(!nonnull.contains(&("nullable_helper".to_string(), 0)));
    let nullness = emitter
        .infer_function_param_nullness(&ir.functions)
        .expect("infer observed nullness params");
    assert!(!nullness.contains(&("nonnull_helper".to_string(), 0)));
    assert!(nullness.contains(&("nullable_helper".to_string(), 0)));
}

#[test]
fn native_multiblock_helper_carries_nullable_pointer_shadow() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %out, i1 %choose) {
entry:
  %isnull = call i1 @nullable_helper(ptr addrspace(2) null, i1 %choose)
  %word = zext i1 %isnull to i32
  store i32 %word, ptr addrspace(1) %out, align 4
  ret void
}

define internal i1 @nullable_helper(ptr addrspace(2) %maybe, i1 %choose) {
entry:
  br i1 %choose, label %check, label %other
check:
  %isnull = icmp eq ptr addrspace(2) %maybe, null
  ret i1 %isnull
other:
  ret i1 false
}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let module = load_bytes(&spv).expect("load native SPIR-V");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        module
            .functions
            .iter()
            .any(|function| function.parameters.len() == 3),
        "the helper carries its two authored parameters plus one nullness shadow"
    );
    assert!(asm.contains("OpConstantTrue"), "{asm}");
}

#[test]
fn native_internal_pointer_param_from_entry_gep_is_nonnull() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Params = type { [4 x i32], [4 x i32] }

define void @k(ptr addrspace(2) %params) {
entry:
  %field = getelementptr inbounds %Params, ptr addrspace(2) %params, i64 0, i32 1, i64 0
  call fastcc void @helper(ptr addrspace(2) %field)
  ret void
}

define internal fastcc void @helper(ptr addrspace(2) %maybe) {
entry:
  %isnull = icmp eq ptr addrspace(2) %maybe, null
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 32, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 32, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
"#;
    let ir = super::super::ir::LlModule::parse(ll).expect("parse");
    let emitter = Emitter::new(ir.clone());
    let nonnull = emitter
        .infer_function_param_nonnull(&ir.functions)
        .expect("infer nonnull params");
    assert!(nonnull.contains(&("helper".to_string(), 0)));

    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpConstantFalse"), "{asm}");
    assert!(!asm.contains("OpPtrEqual"), "{asm}");
}

#[test]
fn native_internal_pointer_param_from_nonzero_inbounds_loaded_gep_is_nonnull() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Params = type { [4 x i32], [4 x i32] }
%Args = type { ptr addrspace(2) }

define void @k(ptr %args) {
entry:
  %slot = getelementptr inbounds %Args, ptr %args, i64 0, i32 0
  %base = load ptr addrspace(2), ptr %slot
  %field = getelementptr inbounds %Params, ptr addrspace(2) %base, i64 0, i32 1, i64 0
  call fastcc void @helper(ptr addrspace(2) %field)
  ret void
}

define internal fastcc void @helper(ptr addrspace(2) %maybe) {
entry:
  %isnull = icmp eq ptr addrspace(2) %maybe, null
  ret void
}
"#;
    let ir = super::super::ir::LlModule::parse_with_stage_meta(ll, None, Some("k")).expect("parse");
    let emitter = Emitter::new(ir.clone());
    let nonnull = emitter
        .infer_function_param_nonnull(&ir.functions)
        .expect("infer nonnull params");
    assert!(nonnull.contains(&("helper".to_string(), 0)));
}

#[test]
fn native_inttoptr_lowers_to_unmodeled_pointer_value() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @carry_int_pointer(i1 %cond, i64 %addr, ptr addrspace(2) %fallback) {
entry:
  br i1 %cond, label %from_int, label %from_param

from_int:
  %p = inttoptr i64 %addr to ptr addrspace(2)
  br label %merge

from_param:
  br label %merge

merge:
  %m = phi ptr addrspace(2) [ %p, %from_int ], [ %fallback, %from_param ]
  ret void
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpVariable"), "{asm}");
    assert!(asm.contains("Private"), "{asm}");
    assert!(asm.contains("OpPhi"), "{asm}");
    assert!(!asm.contains("OpBitcast"), "{asm}");
}

#[test]
fn native_unmodeled_gep_to_pointer_field_uses_byte_placeholder() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Header = type { i64, ptr addrspace(2) }
define void @pointer_field(i64 %addr) {
entry:
  %base = inttoptr i64 %addr to ptr addrspace(2)
  %field = getelementptr inbounds %Header, ptr addrspace(2) %base, i64 0, i32 1
  %p = load ptr addrspace(2), ptr addrspace(2) %field
  %elt = getelementptr inbounds i32, ptr addrspace(2) %p, i64 0
  store i32 1, ptr addrspace(2) %elt
  ret void
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpVariable"), "{asm}");
    assert!(!asm.contains("_ptr_Private__ptr_"), "{asm}");
}

#[test]
fn native_unmodeled_pointer_placeholder_uses_storage_only_pointee() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Inner = type { ptr addrspace(2), i64 }
%Outer = type { %Inner, i32 }
define void @main() {
entry:
  %base = inttoptr i64 0 to ptr addrspace(1)
  %field = getelementptr inbounds %Outer, ptr addrspace(1) %base, i64 0, i32 0
  ret void
}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_storage_only_placeholder_{}",
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
    assert!(!asm.contains("_ptr_Private__ptr_"), "{asm}");
    assert!(!asm.contains("OpTypeStruct %_ptr_UniformConstant"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_gep_preserves_device_addrspace_for_loads() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.S = type { i32 }
define i32 @load_device(ptr addrspace(1) %p) {
entry:
  %g = getelementptr inbounds %struct.S, ptr addrspace(1) %p, i64 0, i32 0
  %v = load i32, ptr addrspace(1) %g
  ret i32 %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(asm.contains("OpLoad"), "{asm}");
}

#[test]
fn native_threadgroup_atomic_param_struct_gep_keeps_backing_array_index() {
    // A threadgroup-buffer ENTRY PARAM is backed post-interface by an oversized Workgroup array of
    // its logical pointee ([512 x T]). A `gep T, ptr %param, 0, 0` must keep BOTH indices (element
    // index into the backing array + member index), else the atomic pointer stops at the wrapper
    // struct and OpAtomicStore/Load reject the non-scalar pointee (mergeLines_parallel a8dfbc01).
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_atomic" = type { i32 }

define void @k(ptr addrspace(3) noundef align 4 captures(none) dereferenceable(4) "air-buffer-no-alias" %0, <2 x i16> noundef %1) local_unnamed_addr {
  %3 = getelementptr inbounds %"struct.metal::_atomic", ptr addrspace(3) %0, i64 0, i32 0
  tail call void @air.atomic.local.store.i32(ptr addrspace(3) captures(none) %3, i32 0, i32 0, i32 1, i1 true)
  %4 = tail call i32 @air.atomic.local.load.i32(ptr addrspace(3) captures(none) %3, i32 0, i32 1, i1 true)
  ret void
}

declare void @air.atomic.local.store.i32(ptr addrspace(3) captures(none), i32, i32, i32, i1)
declare i32 @air.atomic.local.load.i32(ptr addrspace(3) captures(none), i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.struct_type_info", !5, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"metal::_atomic", !"air.arg_name", !"counter"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
!5 = !{i32 0, i32 4, i32 0, !"uint", !"value", !"air.atomic"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_atomic_param_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpAtomicStore"), "{asm}");
    assert!(asm.contains("OpAtomicLoad"), "{asm}");
    // The atomic backing array is flattened from `[N x struct { uint }]` to `[N x uint]`, so the
    // atomic pointer should chain directly to the scalar element with one index.
    let store_ptr = asm
        .lines()
        .find(|l| l.contains("OpAtomicStore"))
        .and_then(|l| l.split_whitespace().find(|w| w.starts_with('%')))
        .expect("OpAtomicStore pointer operand")
        .to_string();
    let chain = asm
        .lines()
        .find(|l| l.contains(&format!("{store_ptr} = OpInBoundsAccessChain")))
        .expect("atomic pointer access chain");
    let n_indices = chain.split(" = ").nth(1).map_or(0, |rhs| {
        rhs.split_whitespace()
            .filter(|w| w.starts_with('%'))
            .count()
            - 2 // type + base
    });
    assert_eq!(
        n_indices, 1,
        "expected flattened scalar element index: {chain}"
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
fn native_threadgroup_atomic_global_struct_pointer_peels_i32_field() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_atomic" = type { i32 }
@counter = internal addrspace(3) global %"struct.metal::_atomic" zeroinitializer, align 4

define void @k() {
entry:
  tail call void @air.atomic.local.store.i32(ptr addrspace(3) @counter, i32 0, i32 0, i32 1, i1 true)
  tail call void @air.wg.barrier(i32 2, i32 1)
  %v = tail call i32 @air.atomic.local.load.i32(ptr addrspace(3) @counter, i32 0, i32 1, i1 true)
  %old = tail call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) @counter, i32 1, i32 0, i32 1, i1 true)
  ret void
}

declare void @air.atomic.local.store.i32(ptr addrspace(3), i32, i32, i32, i1)
declare void @air.wg.barrier(i32, i32)
declare i32 @air.atomic.local.load.i32(ptr addrspace(3), i32, i32, i1)
declare i32 @air.atomic.local.add.u.i32(ptr addrspace(3), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_atomic_struct_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        whole_workgroup_options(),
    )
    .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(asm.contains("OpAtomicStore"), "{asm}");
    assert!(asm.contains("OpAtomicLoad"), "{asm}");
    assert!(asm.contains("OpAtomicIAdd"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_threadgroup_atomic_multifield_global_peels_first_atomic_field() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_atomic" = type { i32 }
%struct.Counters = type { %"struct.metal::_atomic", %"struct.metal::_atomic", %"struct.metal::_atomic" }
@counters = internal addrspace(3) global %struct.Counters zeroinitializer, align 4

define void @k() {
entry:
  tail call void @air.atomic.local.store.i32(ptr addrspace(3) @counters, i32 0, i32 0, i32 1, i1 true)
  %old = tail call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) @counters, i32 1, i32 0, i32 1, i1 true)
  ret void
}

declare void @air.atomic.local.store.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.add.u.i32(ptr addrspace(3), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_atomic_multifield_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpAtomicStore"), "{asm}");
    assert!(asm.contains("OpAtomicIAdd"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_threadgroup_atomic_union_singleton_array_peels_first_scalar_field() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_atomic" = type { i32 }
%union.atomicUint = type { [1 x %"struct.metal::_atomic"] }
@counter = internal addrspace(3) global %union.atomicUint undef, align 4

define void @k() {
entry:
  %old = tail call i32 @air.atomic.local.max.u.i32(ptr addrspace(3) @counter, i32 7, i32 0, i32 1, i1 true)
  ret void
}

declare i32 @air.atomic.local.max.u.i32(ptr addrspace(3), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !1}
!1 = !{}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_atomic_union_singleton_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(asm.contains("OpAtomicUMax"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_raw_atomic_array_load_uses_storage_buffer_pointer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

%"struct.metal::_atomic" = type { i32 }
%struct.Counters = type <{ [16 x %"struct.metal::_atomic"] }>

define void @k(ptr addrspace(1) %bytes, ptr addrspace(2) %base_ptr, ptr addrspace(2) %idx_ptr, ptr addrspace(1) %out) {
entry:
  %base = load i32, ptr addrspace(2) %base_ptr, align 4
  %base64 = zext i32 %base to i64
  %raw = getelementptr inbounds i8, ptr addrspace(1) %bytes, i64 %base64
  %counters = bitcast ptr addrspace(1) %raw to ptr addrspace(1)
  %idx = load i32, ptr addrspace(2) %idx_ptr, align 4
  %idx64 = zext i32 %idx to i64
  %slot = getelementptr inbounds %struct.Counters, ptr addrspace(1) %counters, i64 0, i32 0, i64 %idx64, i32 0
  %v = tail call i32 @air.atomic.global.load.i32(ptr addrspace(1) %slot, i32 0, i32 2, i1 true)
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

declare i32 @air.atomic.global.load.i32(ptr addrspace(1), i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"bytes"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"base"}
!5 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"idx"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_raw_atomic_array_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpAtomicLoad"), "{asm}");
    assert!(
        !asm.contains("OpVariable %_ptr_Workgroup_uint Workgroup"),
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

/// The atomic-float-min/max idiom (MPS BVH bounding boxes): a device `<3 x float>*` buffer whose
/// float lanes are updated with signed-integer atomics on the reinterpreted bits
/// (`air.atomic.global.{min,max}.s.i32`). `atomic_i32_pointer_id` cannot form an `i32*` from a
/// `<3 x float>` pointee under Logical addressing, so the failure-triggered raw retry marks the buffer
/// raw and lowers the atomics as uint-word `OpAtomicSMin`/`OpAtomicSMax` on the `RuntimeArray<uint>`
/// backing. Without the retry trigger covering this error, translate fails ("atomic i32 pointer
/// targets Vector(Float, 3)").
#[test]
fn native_device_atomic_int_on_float3_buffer_lowers_via_raw_word() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %bbox) {
entry:
  %l0 = getelementptr inbounds <3 x float>, ptr addrspace(1) %bbox, i64 0, i64 0
  %p0 = bitcast ptr addrspace(1) %l0 to ptr addrspace(1)
  %o0 = tail call i32 @air.atomic.global.min.s.i32(ptr addrspace(1) %p0, i32 1056964608, i32 0, i32 2, i1 true)
  %l2 = getelementptr inbounds <3 x float>, ptr addrspace(1) %bbox, i64 0, i64 2
  %p2 = bitcast ptr addrspace(1) %l2 to ptr addrspace(1)
  %o2 = tail call i32 @air.atomic.global.max.s.i32(ptr addrspace(1) %p2, i32 2, i32 0, i32 2, i1 true)
  ret void
}

declare i32 @air.atomic.global.min.s.i32(ptr addrspace(1), i32, i32, i32, i1)
declare i32 @air.atomic.global.max.s.i32(ptr addrspace(1), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float3", !"air.arg_name", !"bbox"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_atomic_int_float3_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpAtomicSMin"), "{asm}");
    assert!(asm.contains("OpAtomicSMax"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_device_atomic_i32_lowers_to_device_scope_spirv() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %count) {
entry:
  %old = tail call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %count, i32 1, i32 0, i32 2, i1 true)
  ret void
}

declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"count"}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpAtomicIAdd"), "{asm}");
    let module = load_bytes(&spv).expect("load native spv");
    let atomic = module
        .all_inst_iter()
        .find(|inst| inst.class.opcode == Op::AtomicIAdd)
        .expect("atomic add");
    let scope_id = match atomic.operands.get(1) {
        Some(Operand::IdScope(id) | Operand::IdRef(id)) => *id,
        other => panic!("unexpected atomic scope operand: {other:?}"),
    };
    let scope_value = module
        .all_inst_iter()
        .find(|inst| inst.result_id == Some(scope_id) && inst.class.opcode == Op::Constant)
        .and_then(|inst| match inst.operands.first() {
            Some(Operand::LiteralBit32(value)) => Some(*value),
            _ => None,
        })
        .expect("scope constant");
    assert_eq!(scope_value, Scope::Device as u32, "{asm}");
}

#[test]
fn native_device_atomic_store_i32_lowers_to_device_scope_spirv() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %count) {
entry:
  tail call void @air.atomic.global.store.i32(ptr addrspace(1) %count, i32 0, i32 0, i32 2, i1 true)
  ret void
}

declare void @air.atomic.global.store.i32(ptr addrspace(1), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"count"}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpAtomicStore"), "{asm}");
    let module = load_bytes(&spv).expect("load native spv");
    let atomic = module
        .all_inst_iter()
        .find(|inst| inst.class.opcode == Op::AtomicStore)
        .expect("atomic store");
    let scope_id = match atomic.operands.get(1) {
        Some(Operand::IdScope(id) | Operand::IdRef(id)) => *id,
        other => panic!("unexpected atomic scope operand: {other:?}"),
    };
    let scope_value = module
        .all_inst_iter()
        .find(|inst| inst.result_id == Some(scope_id) && inst.class.opcode == Op::Constant)
        .and_then(|inst| match inst.operands.first() {
            Some(Operand::LiteralBit32(value)) => Some(*value),
            _ => None,
        })
        .expect("scope constant");
    assert_eq!(scope_value, Scope::Device as u32, "{asm}");
}

#[test]
fn native_device_atomic_cmpxchg_weak_i32_updates_compare_pointer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define i32 @atomic_cmpxchg(ptr addrspace(1) %slot, i32 %expected, i32 %desired) {
entry:
  %compare = alloca i32, align 4
  store i32 %expected, ptr %compare, align 4
  %old = call i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1) %slot, ptr %compare, i32 %desired, i32 0, i32 0, i32 2, i1 true)
  %seen = load i32, ptr %compare, align 4
  %sum = add i32 %old, %seen
  ret i32 %sum
}

declare i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1), ptr, i32, i32, i32, i32, i1)
"#;
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let atomic = module
        .all_inst_iter()
        .find(|inst| inst.class.opcode == Op::AtomicCompareExchange)
        .expect("atomic cmpxchg");
    let scope_id = match atomic.operands.get(1) {
        Some(Operand::IdScope(id) | Operand::IdRef(id)) => *id,
        other => panic!("unexpected atomic scope operand: {other:?}"),
    };
    let scope_value = module
        .all_inst_iter()
        .find(|inst| inst.result_id == Some(scope_id) && inst.class.opcode == Op::Constant)
        .and_then(|inst| match inst.operands.first() {
            Some(Operand::LiteralBit32(value)) => Some(*value),
            _ => None,
        })
        .expect("scope constant");
    assert_eq!(scope_value, Scope::Device as u32);
    assert!(
        module
            .all_inst_iter()
            .any(|inst| inst.class.opcode == Op::Store),
        "expected compare pointer update store"
    );
}

#[test]
fn native_device_atomic_f32_add_lowers_to_ext_atomic_float() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

%"struct.metal::_atomic" = type { float }

define void @k(ptr addrspace(1) %stats, ptr addrspace(1) %out) {
entry:
  %slot = getelementptr inbounds %"struct.metal::_atomic", ptr addrspace(1) %stats, i64 1, i32 0
  %old = tail call fast float @air.atomic.global.add.f32(ptr addrspace(1) %slot, float 1.250000e+00, i32 0, i32 2, i1 true)
  store float %old, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.atomic.global.add.f32(ptr addrspace(1), float, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"stats"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_device_atomic_f32_add_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        asm.contains("OpExtension \"SPV_EXT_shader_atomic_float_add\""),
        "{asm}"
    );
    assert!(asm.contains("OpCapability AtomicFloat32AddEXT"), "{asm}");
    assert!(asm.contains("OpAtomicFAddEXT"), "{asm}");
    assert_no_pointer_bitcasts(&spv);
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_device_atomic_f32_sub_lowers_to_negated_ext_atomic_float_add() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

%"struct.metal::_atomic" = type { float }

define void @k(ptr addrspace(1) %stats, ptr addrspace(1) %out) {
entry:
  %slot = getelementptr inbounds %"struct.metal::_atomic", ptr addrspace(1) %stats, i64 1, i32 0
  %old = tail call fast float @air.atomic.global.sub.f32(ptr addrspace(1) %slot, float 1.250000e+00, i32 0, i32 2, i1 true)
  store float %old, ptr addrspace(1) %out, align 4
  ret void
}

declare float @air.atomic.global.sub.f32(ptr addrspace(1), float, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"stats"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_device_atomic_f32_sub_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    // Atomic float subtract lowers to an atomic float add of the negated operand.
    assert!(
        asm.contains("OpExtension \"SPV_EXT_shader_atomic_float_add\""),
        "{asm}"
    );
    assert!(asm.contains("OpCapability AtomicFloat32AddEXT"), "{asm}");
    assert!(asm.contains("OpFNegate"), "{asm}");
    assert!(asm.contains("OpAtomicFAddEXT"), "{asm}");
    assert_no_pointer_bitcasts(&spv);
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_unmodeled_device_atomic_cmpxchg_uses_workgroup_scratch() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define i32 @atomic_cmpxchg_unmodeled(i32 %expected, i32 %desired) {
entry:
  %slot = inttoptr i64 0 to ptr addrspace(1)
  %compare = alloca i32, align 4
  store i32 %expected, ptr %compare, align 4
  %old = call i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1) %slot, ptr %compare, i32 %desired, i32 0, i32 0, i32 2, i1 true)
  ret i32 %old
}

declare i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1), ptr, i32, i32, i32, i32, i1)
"#;
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let atomic = module
        .all_inst_iter()
        .find(|inst| inst.class.opcode == Op::AtomicCompareExchange)
        .expect("atomic cmpxchg");
    let ptr = match atomic.operands.first() {
        Some(Operand::IdRef(id)) => *id,
        other => panic!("unexpected atomic pointer operand: {other:?}"),
    };
    let ptr_def = module
        .all_inst_iter()
        .find(|inst| inst.result_id == Some(ptr))
        .expect("atomic pointer definition");
    assert_eq!(ptr_def.class.opcode, Op::Variable);
    assert!(
        matches!(
            ptr_def.operands.first(),
            Some(Operand::StorageClass(StorageClass::Workgroup))
        ),
        "{ptr_def:?}"
    );
}

#[test]
fn native_device_atomic_umax_i32_lowers_through_kernel_transform() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Counter = type { i32 }

define void @k(ptr addrspace(1) %max_value) {
entry:
  %slot = getelementptr inbounds %struct.Counter, ptr addrspace(1) %max_value, i64 0, i32 0
  %old = tail call i32 @air.atomic.global.max.u.i32(ptr addrspace(1) %slot, i32 7, i32 0, i32 2, i1 true)
  ret void
}

declare i32 @air.atomic.global.max.u.i32(ptr addrspace(1), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"max_value"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_device_atomic_umax_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpAtomicUMax"), "{asm}");
    let module = load_bytes(&spv).expect("load native spv");
    let atomic = module
        .all_inst_iter()
        .find(|inst| inst.class.opcode == Op::AtomicUMax)
        .expect("atomic umax");
    let scope_id = match atomic.operands.get(1) {
        Some(Operand::IdScope(id) | Operand::IdRef(id)) => *id,
        other => panic!("unexpected atomic scope operand: {other:?}"),
    };
    let scope_value = module
        .all_inst_iter()
        .find(|inst| inst.result_id == Some(scope_id) && inst.class.opcode == Op::Constant)
        .and_then(|inst| match inst.operands.first() {
            Some(Operand::LiteralBit32(value)) => Some(*value),
            _ => None,
        })
        .expect("scope constant");
    assert_eq!(scope_value, Scope::Device as u32, "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_device_atomic_i32_variants_lower_through_kernel_transform() {
    let cases = [
        (
            "air.atomic.global.add.s.i32",
            Op::AtomicIAdd,
            "OpAtomicIAdd",
        ),
        ("air.atomic.global.and.u.i32", Op::AtomicAnd, "OpAtomicAnd"),
        ("air.atomic.global.or.u.i32", Op::AtomicOr, "OpAtomicOr"),
        (
            "air.atomic.global.xchg.i32",
            Op::AtomicExchange,
            "OpAtomicExchange",
        ),
        (
            "air.atomic.global.max.s.i32",
            Op::AtomicSMax,
            "OpAtomicSMax",
        ),
        (
            "air.atomic.global.min.s.i32",
            Op::AtomicSMin,
            "OpAtomicSMin",
        ),
        (
            "air.atomic.global.min.u.i32",
            Op::AtomicUMin,
            "OpAtomicUMin",
        ),
        (
            "air.atomic.global.sub.s.i32",
            Op::AtomicISub,
            "OpAtomicISub",
        ),
        (
            "air.atomic.global.sub.u.i32",
            Op::AtomicISub,
            "OpAtomicISub",
        ),
    ];
    for (callee, opcode, opname) in cases {
        let ll = format!(
            r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Counter = type {{ i32 }}

define void @k(ptr addrspace(1) %value) {{
entry:
  %slot = getelementptr inbounds %struct.Counter, ptr addrspace(1) %value, i64 0, i32 0
  %old = tail call i32 @{callee}(ptr addrspace(1) %slot, i32 7, i32 0, i32 2, i1 true)
  ret void
}}

declare i32 @{callee}(ptr addrspace(1), i32, i32, i32, i1)

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"int", !"air.arg_name", !"value"}}
"#
        );
        let tmp = std::env::temp_dir().join(format!(
            "metal2vulkan_device_atomic_minmax_{}_{}",
            opname,
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&tmp);
        let spv = crate::translate_sanitized_native(&ll, Stage::Kernel, &tmp)
            .unwrap_or_else(|e| panic!("translate {callee}: {e}"));
        let asm = disassemble(&spv).expect("disassemble");
        assert!(asm.contains(opname), "{asm}");
        let module = load_bytes(&spv).expect("load native spv");
        let atomic = module
            .all_inst_iter()
            .find(|inst| inst.class.opcode == opcode)
            .unwrap_or_else(|| panic!("missing {opname}"));
        let scope_id = match atomic.operands.get(1) {
            Some(Operand::IdScope(id) | Operand::IdRef(id)) => *id,
            other => panic!("unexpected atomic scope operand: {other:?}"),
        };
        let scope_value = module
            .all_inst_iter()
            .find(|inst| inst.result_id == Some(scope_id) && inst.class.opcode == Op::Constant)
            .and_then(|inst| match inst.operands.first() {
                Some(Operand::LiteralBit32(value)) => Some(*value),
                _ => None,
            })
            .expect("scope constant");
        assert_eq!(scope_value, Scope::Device as u32, "{asm}");
        if std::process::Command::new("spirv-val")
            .arg("--version")
            .output()
            .is_ok()
        {
            tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
        }
    }
}

#[test]
fn native_threadgroup_global_array_opaque_gep_uses_access_chain() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@shared = internal addrspace(3) global [400 x float] undef, align 4
@nested = internal addrspace(3) global [4 x [324 x float]] undef, align 4

define void @k(i32 %i) {
entry:
  %idx = zext i32 %i to i64
  %p = getelementptr inbounds float, ptr addrspace(3) @shared, i64 %idx
  store float 1.000000e+00, ptr addrspace(3) %p, align 4
  %nested = getelementptr inbounds [324 x float], ptr addrspace(3) @nested, i64 0, i64 %idx
  store float 2.000000e+00, ptr addrspace(3) %nested, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_global_array_gep_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(!asm.contains("OpPtrAccessChain"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_threadgroup_scalar_array_struct_view_folds_field_index() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.half6 = type { half, half, half, half, half, half }
@randomCoords = internal unnamed_addr addrspace(3) global [1024 x half] undef, align 8

define void @k(i32 %i, ptr addrspace(1) %out) {
entry:
  %idx = zext i32 %i to i64
  %slot0 = getelementptr inbounds %struct.half6, ptr addrspace(3) @randomCoords, i64 %idx, i32 0
  %v0 = load half, ptr addrspace(3) %slot0, align 2
  %slot5 = getelementptr inbounds %struct.half6, ptr addrspace(3) @randomCoords, i64 %idx, i32 5
  %v5 = load half, ptr addrspace(3) %slot5, align 2
  store half %v0, ptr addrspace(1) %out, align 2
  %out5 = getelementptr inbounds half, ptr addrspace(1) %out, i64 1
  store half %v5, ptr addrspace(1) %out5, align 2
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"half", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_scalar_array_struct_view_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load spv");
    let half = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeFloat && inst.operands == [Operand::LiteralBit32(16)]
        })
        .and_then(|inst| inst.result_id)
        .expect("half type");
    let half_ptr = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypePointer
                && inst.operands
                    == [
                        Operand::StorageClass(StorageClass::Workgroup),
                        Operand::IdRef(half),
                    ]
        })
        .and_then(|inst| inst.result_id)
        .expect("workgroup half pointer");
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
    let random_coord_chains = module
        .functions
        .iter()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| {
            matches!(inst.class.opcode, Op::AccessChain | Op::InBoundsAccessChain)
                && inst.result_type == Some(half_ptr)
                && inst
                    .operands
                    .first()
                    .and_then(id_ref_operand)
                    .is_some_and(|base| workgroup_vars.contains(&base))
        })
        .collect::<Vec<_>>();
    assert!(!random_coord_chains.is_empty(), "{asm}");
    assert!(
        random_coord_chains
            .iter()
            .all(|inst| inst.operands.len() == 2),
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
fn native_threadgroup_union_global_raw_array_gep_indexes_first_scalar_field() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_atomic" = type { i32 }
%union.histogram_t = type { [64 x %"struct.metal::_atomic"] }
@wholeFaceHistogram = internal addrspace(3) global %union.histogram_t undef, align 4

define void @k(i32 %i) {
entry:
  %idx64 = zext i32 %i to i64
  %raw = getelementptr inbounds [64 x i32], ptr addrspace(3) @wholeFaceHistogram, i64 0, i64 %idx64
  store i32 0, ptr addrspace(3) %raw, align 4
  %atomic = getelementptr inbounds %union.histogram_t, ptr addrspace(3) @wholeFaceHistogram, i64 0, i32 0, i64 %idx64, i32 0
  %old = tail call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) %atomic, i32 1, i32 0, i32 1, i1 true)
  ret void
}

declare i32 @air.atomic.local.add.u.i32(ptr addrspace(3), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_union_raw_array_gep_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(!asm.contains("OpPtrAccessChain"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_threadgroup_load_accepts_constant_gep_pointer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@tg_values = internal unnamed_addr addrspace(3) global [128 x i32] undef, align 4

define void @k(ptr addrspace(1) %out) {
entry:
  %p = getelementptr inbounds [128 x i32], ptr addrspace(3) @tg_values, i64 0, i64 127
  store i32 7, ptr addrspace(3) %p, align 4
  %v = load i32, ptr addrspace(3) getelementptr inbounds ([128 x i32], ptr addrspace(3) @tg_values, i64 0, i64 127), align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_constant_gep_load_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Workgroup"), "{asm}");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
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
fn native_threadgroup_vector_global_i32_gep_reinterprets_words() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@shared = internal addrspace(3) global [64 x <4 x i16>] undef, align 8

define void @k(i32 %i, ptr addrspace(1) %out) {
entry:
  %idx = zext i32 %i to i64
  %p = getelementptr inbounds i32, ptr addrspace(3) @shared, i64 %idx
  store i32 %i, ptr addrspace(3) %p, align 4
  %loaded = load i32, ptr addrspace(3) %p, align 4
  store i32 %loaded, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_vector_word_gep_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpVectorExtractDynamic"), "{asm}");
    // The i32-view store writes exactly one word: two 16-bit component stores, never a
    // full-vector read-modify-write (which races against neighbouring-word writers).
    assert!(!asm.contains("OpVectorInsertDynamic"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_threadgroup_vector_bitcast_i64_store_reinterprets_whole_vector() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@shared = internal addrspace(3) global [64 x <4 x i16>] undef, align 8

define void @main() {
entry:
  %slot = getelementptr inbounds <4 x i16>, ptr addrspace(3) @shared, i64 0
  %raw = bitcast ptr addrspace(3) %slot to ptr addrspace(3)
  store i64 81985529216486895, ptr addrspace(3) %raw, align 8
  %loaded = load i64, ptr addrspace(3) %raw, align 8
  ret void
}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_vector_i64_store_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let spv = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load native spv");
    let function_insts = module
        .functions
        .iter()
        .flat_map(|func| &func.blocks)
        .flat_map(|block| &block.instructions)
        .collect::<Vec<_>>();
    let stored_objects = function_insts
        .iter()
        .filter(|inst| inst.class.opcode == Op::Store)
        .filter_map(|inst| inst.operands.get(1).and_then(id_ref_operand))
        .collect::<HashSet<_>>();
    assert!(
        function_insts.iter().any(|inst| {
            inst.class.opcode == Op::Bitcast
                && inst
                    .result_id
                    .is_some_and(|id| stored_objects.contains(&id))
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

#[test]
fn native_threadgroup_vector_bitcast_wide_store_splits_to_vector_slots() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@shared = internal addrspace(3) global [64 x <4 x half>] undef, align 8

define void @main() {
entry:
  %slot = getelementptr inbounds <4 x half>, ptr addrspace(3) @shared, i64 0
  %raw = bitcast ptr addrspace(3) %slot to ptr addrspace(3)
  store <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>, ptr addrspace(3) %raw, align 16
  ret void
}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_vector_wide_store_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let spv = passes::transform(module, Stage::Kernel, None, None, None, Some("main"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(asm.matches("OpCompositeConstruct").count(), 2, "{asm}");
    assert_eq!(asm.matches("OpBitcast").count(), 2, "{asm}");
    assert_eq!(asm.matches("OpPtrAccessChain").count(), 0, "{asm}");
    assert_eq!(asm.matches("OpInBoundsAccessChain").count(), 2, "{asm}");
    // Two element stores from the split wide store; the threadgroup zero-init prologue adds one
    // OpConstantNull whole-array store on top — exclude it from the split-store count.
    let null_ids: Vec<&str> = asm
        .lines()
        .filter(|line| line.contains("= OpConstantNull"))
        .filter_map(|line| line.split_whitespace().next())
        .collect();
    let body_stores = asm
        .lines()
        .filter(|line| line.trim_start().starts_with("OpStore"))
        .filter(|line| {
            !null_ids
                .iter()
                .any(|id| line.split_whitespace().any(|token| token == *id))
        })
        .count();
    assert_eq!(body_stores, 2, "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_threadgroup_vector_param_gep_then_raw_i32_reinterprets_words() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@shared = internal addrspace(3) global [64 x <4 x i16>] undef, align 8

define void @k(i32 %i, ptr addrspace(1) %out) {
entry:
  tail call void @helper(i32 %i, ptr addrspace(3) @shared, ptr addrspace(1) %out)
  ret void
}

define internal void @helper(i32 %i, ptr addrspace(3) %scratch, ptr addrspace(1) %out) {
entry:
  %idx = zext i32 %i to i64
  %vecp = getelementptr inbounds <4 x i16>, ptr addrspace(3) %scratch, i64 %idx
  store <4 x i16> zeroinitializer, ptr addrspace(3) %vecp, align 8
  store <4 x i16> zeroinitializer, ptr addrspace(3) %scratch, align 8
  %base = load <4 x i16>, ptr addrspace(3) %scratch, align 8
  %raw = bitcast ptr addrspace(3) %scratch to ptr addrspace(3)
  %wordp = getelementptr inbounds i32, ptr addrspace(3) %raw, i64 %idx
  store i32 %i, ptr addrspace(3) %wordp, align 4
  %loaded = load i32, ptr addrspace(3) %wordp, align 4
  store i32 %loaded, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_vector_param_word_gep_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let native_spv = emit_vulkan_spirv(ll).expect("native emit");
    let native_asm = disassemble(&native_spv).expect("disassemble native");
    assert!(
        !native_asm.contains("OpVectorInsertDynamic"),
        "{native_asm}"
    );
    assert!(
        native_asm.contains("OpVectorExtractDynamic"),
        "{native_asm}"
    );
    assert!(
        !native_asm
            .lines()
            .any(|line| line.contains(" OpBitcast ") && line.contains("_ptr_")),
        "{native_asm}"
    );
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("OpVectorInsertDynamic"), "{asm}");
    assert!(asm.contains("OpVectorExtractDynamic"), "{asm}");
    assert!(
        !asm.lines()
            .any(|line| line.contains(" OpBitcast ") && line.contains("_ptr_")),
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
fn native_threadgroup_i32_param_gep_uses_callsite_vector_pointee() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@shared = internal addrspace(3) global [64 x <4 x i16>] undef, align 8

define void @k(i32 %i, ptr addrspace(1) %out) {
entry:
  tail call void @prefix(i32 %i, ptr addrspace(3) @shared, ptr addrspace(1) %out)
  ret void
}

define internal void @prefix(i32 %i, ptr addrspace(3) %scratch, ptr addrspace(1) %out) {
entry:
  %idx = zext i32 %i to i64
  %wordp = getelementptr inbounds i32, ptr addrspace(3) %scratch, i64 %idx
  store i32 %i, ptr addrspace(3) %wordp, align 4
  %loaded = load i32, ptr addrspace(3) %wordp, align 4
  store i32 %loaded, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_i32_param_word_gep_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpVectorExtractDynamic"), "{asm}");
    assert!(!asm.contains("OpVectorInsertDynamic"), "{asm}");
    assert!(
        !asm.contains("OpInBoundsAccessChain %_ptr_Workgroup_uint"),
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
fn native_kernel_threadgroup_mixed_half_uint_scratch_uses_raw_word_array() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(3) %scratch, ptr addrspace(1) %out, i32 %i) {
entry:
  %idx = zext i32 %i to i64
  %halfp = getelementptr inbounds half, ptr addrspace(3) %scratch, i64 0
  store half 0xH3C00, ptr addrspace(3) %halfp, align 2
  %raw = bitcast ptr addrspace(3) %halfp to ptr addrspace(3)
  %wordp = getelementptr inbounds i32, ptr addrspace(3) %raw, i64 %idx
  store i32 %i, ptr addrspace(3) %wordp, align 4
  %loaded = load i32, ptr addrspace(3) %wordp, align 4
  store i32 %loaded, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"scratch"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_kernel_threadgroup_mixed_half_uint_scratch_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load spv");
    let uint = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::TypeInt
                && inst.operands == [Operand::LiteralBit32(32), Operand::LiteralBit32(0)]
        })
        .and_then(|inst| inst.result_id)
        .expect("uint type");
    let workgroup_var = module
        .types_global_values
        .iter()
        .find(|inst| {
            inst.class.opcode == Op::Variable
                && inst.operands.first() == Some(&Operand::StorageClass(StorageClass::Workgroup))
        })
        .and_then(|inst| inst.result_id)
        .expect("workgroup var");
    let workgroup_array = variable_pointee_type(&module, workgroup_var).expect("workgroup array");
    let array_def = module
        .types_global_values
        .iter()
        .find(|inst| inst.class.opcode == Op::TypeArray && inst.result_id == Some(workgroup_array))
        .expect("workgroup array type");
    let [Operand::IdRef(elem_ty), Operand::IdRef(len_id)] = array_def.operands.as_slice() else {
        panic!("workgroup scratch should be a fixed array: {asm}");
    };
    assert_eq!(*elem_ty, uint, "{asm}");
    let len = module
        .all_inst_iter()
        .find(|inst| inst.class.opcode == Op::Constant && inst.result_id == Some(*len_id))
        .and_then(|inst| match inst.operands.first() {
            Some(Operand::LiteralBit32(value)) => Some(*value),
            _ => None,
        })
        .expect("workgroup array length");
    assert_eq!(len, 2048, "{asm}");
    assert!(asm.contains("OpAtomicOr"), "{asm}");
    assert!(!asm.contains("_arr_uchar_uint_512"), "{asm}");
    assert!(
        !asm.contains("OpPtrAccessChain %_ptr_Workgroup_uint"),
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
fn native_threadgroup_dynamic_gep_allows_negative_constant_rebase() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(3) %scratch, ptr addrspace(1) %out, i32 %i) {
entry:
  %ok = icmp uge i32 %i, 128
  br i1 %ok, label %body, label %exit

body:
  %idx = zext i32 %i to i64
  %halfp = getelementptr inbounds half, ptr addrspace(3) %scratch, i64 0
  store half 0xH3C00, ptr addrspace(3) %halfp, align 2
  %raw = bitcast ptr addrspace(3) %halfp to ptr addrspace(3)
  %advanced = getelementptr inbounds i32, ptr addrspace(3) %raw, i64 %idx
  %rebased = getelementptr inbounds i32, ptr addrspace(3) %advanced, i64 -128
  store i32 %i, ptr addrspace(3) %rebased, align 4
  %loaded = load i32, ptr addrspace(3) %rebased, align 4
  %as_float = bitcast i32 %loaded to float
  store float %as_float, ptr addrspace(1) %out, align 4
  br label %exit

exit:
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"scratch"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_threadgroup_negative_raw_rebase_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load spv");
    assert!(
        module.all_inst_iter().any(|inst| {
            inst.class.opcode == Op::Constant
                && inst.operands.last() == Some(&Operand::LiteralBit32(4294967168))
        }),
        "{asm}"
    );
    assert!(asm.contains("OpIAdd"), "{asm}");
    assert!(asm.contains("Workgroup"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_pointer_bitcast_load_reinterprets_loaded_bits() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.S = type { float }
define i32 @load_bits(ptr addrspace(1) %p) {
entry:
  %field = getelementptr inbounds %struct.S, ptr addrspace(1) %p, i64 0, i32 0
  %bits_ptr = bitcast ptr addrspace(1) %field to ptr addrspace(1)
  %bits = load i32, ptr addrspace(1) %bits_ptr
  ret i32 %bits
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert_eq!(asm.matches("OpLoad").count(), 1, "{asm}");
    assert!(asm.contains("OpBitcast"), "{asm}");
}

#[test]
fn native_pointer_bitcast_load_reads_first_aggregate_word() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.TextureBufferClearParams = type { i32, %union.anon }
%union.anon = type { [4 x float] }
define void @k(ptr addrspace(1) %dst, ptr addrspace(2) %clear, i32 %tid) {
entry:
  %field = getelementptr inbounds %struct.TextureBufferClearParams, ptr addrspace(2) %clear, i64 0, i32 1
  %bits_ptr = bitcast ptr addrspace(2) %field to ptr addrspace(2)
  %bits = load i32, ptr addrspace(2) %bits_ptr, align 4
  %v0 = insertelement <4 x i32> undef, i32 %bits, i64 0
  %v1 = insertelement <4 x i32> %v0, i32 %bits, i64 1
  %v2 = insertelement <4 x i32> %v1, i32 %bits, i64 2
  %v3 = insertelement <4 x i32> %v2, i32 %bits, i64 3
  tail call void @air.write_texture_buffer_1d.u.v4i32(ptr addrspace(1) %dst, i32 %tid, <4 x i32> %v3, i32 2)
  ret void
}

declare void @air.write_texture_buffer_1d.u.v4i32(ptr addrspace(1), i32, <4 x i32>, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture_buffer<uint, write>", !"air.arg_name", !"dst"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 20, !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 20, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"TextureBufferClearParams", !"air.arg_name", !"clear"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_aggregate_reinterpret_load_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpImageWrite"), "{asm}");
    assert!(asm.contains("OpBitcast"), "{asm}");
    assert!(!asm.contains("OpBitcast %_ptr_"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

/// Loading a `<3 x i32>` from a `[4 x <3 x float>]` local reinterprets the leading `<3 x float>`
/// element lane-for-lane: access-chain element 0, load the vector, then `OpBitcast` to the int vector
/// (a legal same-size numeric-vector bitcast, unlike a pointer bitcast). Without this the emitter fell
/// back with "cannot reinterpret load … Array(Vector(Float, 3), 4) to Vector(Int(32), 3)".
#[test]
fn native_leading_vector_aggregate_load_bitcasts_to_int_vector() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define <3 x i32> @load_leading_vec() {
entry:
  %buf = alloca [4 x <3 x float>], align 16
  %v = load <3 x i32>, ptr %buf, align 16
  ret <3 x i32> %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpBitcast"), "{asm}");
    assert!(!asm.contains("non-bitcastable"), "{asm}");
    assert!(!asm.contains("OpBitcast %_ptr_"), "{asm}");
}

#[test]
fn native_scalar_array_load_rebuilds_same_shape_vector() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define <3 x float> @load_array_as_vector() {
entry:
  %buf = alloca [3 x float], align 16
  %value = load <3 x float>, ptr %buf, align 16
  ret <3 x float> %value
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpLoad"), "{asm}");
    assert!(asm.contains("OpCompositeConstruct"), "{asm}");
    assert!(!asm.contains("OpBitcast %_ptr_"), "{asm}");
}

#[test]
fn native_vector_store_rebuilds_same_shape_scalar_array() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @store_vector_as_array(<3 x float> %value) {
entry:
  %buf = alloca [3 x float], align 16
  store <3 x float> %value, ptr %buf, align 16
  ret void
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpCompositeConstruct"), "{asm}");
    assert!(asm.contains("OpStore"), "{asm}");
}

#[test]
fn native_pointer_bitcast_load_packs_leading_float_fields_as_i64() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Keypoint = type { float, float, i32 }
define i64 @load_prefix_bits(ptr %p) {
entry:
  %field = getelementptr inbounds %struct.Keypoint, ptr %p, i64 0, i32 2
  %keep = load i32, ptr %field, align 4
  %alias = bitcast ptr %p to ptr
  %bits = load i64, ptr %alias, align 4
  ret i64 %bits
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpBitwiseOr"), "{asm}");
    assert!(!asm.contains("non-bitcastable"), "{asm}");
}

#[test]
fn native_pointer_bitcast_store_splits_i64_into_leading_float_fields() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Keypoint = type { float, float, i32 }
define void @store_prefix_bits(ptr %p, i64 %bits) {
entry:
  %field = getelementptr inbounds %struct.Keypoint, ptr %p, i64 0, i32 2
  store i32 7, ptr %field, align 4
  %alias = bitcast ptr %p to ptr
  store i64 %bits, ptr %alias, align 4
  ret void
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(asm.matches("OpBitcast").count() >= 2, "{asm}");
    assert!(!asm.contains("reinterpret"), "{asm}");
}

#[test]
fn native_pointer_bitcast_load_reads_first_pointer_field() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Cache = type { ptr addrspace(1), float }
define i32 @load_first_pointer_field(ptr addrspace(1) %buf) {
entry:
  %cache = alloca %struct.Cache, align 8
  %field = getelementptr inbounds %struct.Cache, ptr %cache, i64 0, i32 0
  store ptr addrspace(1) %buf, ptr %field, align 8
  %bits_ptr = bitcast ptr %cache to ptr
  %loaded = load ptr addrspace(1), ptr %bits_ptr, align 8
  %word_ptr = getelementptr inbounds i32, ptr addrspace(1) %loaded, i64 0
  %word = load i32, ptr addrspace(1) %word_ptr, align 4
  ret i32 %word
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(!asm.contains("non-bitcastable"), "{asm}");
}

#[test]
fn native_local_aggregate_byte_view_uses_array_storage_for_dynamic_index() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%union.ETCBlock = type { %struct.Bits }
%struct.Bits = type { i64 }
define void @k(ptr addrspace(1) %out, i32 %idx) {
entry:
  %block = alloca %union.ETCBlock, align 8
  %alias = bitcast ptr %block to ptr
  %wide = zext i32 %idx to i64
  %byte = getelementptr inbounds [8 x i8], ptr %alias, i64 0, i64 %wide
  store i8 7, ptr %byte, align 1
  %field = getelementptr inbounds %union.ETCBlock, ptr %block, i64 0, i32 0, i32 0
  %bits = load i64, ptr %field, align 8
  %dst = getelementptr inbounds i64, ptr addrspace(1) %out, i64 0
  store i64 %bits, ptr addrspace(1) %dst, align 8
  ret void
}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_local_aggregate_byte_view_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("k"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_local_aggregate_multi_view_remodels_to_byte_array() {
    // One alloca reinterpreted through CONFLICTING same-size views (a byte-fill `[8 x i8]` GEP, a
    // `{ {i32}, {i32} }` struct view, and the original `{ { i64 } }` union view). The byte array is
    // the universal receiver: the remodel must pick it so every typed view lowers through the
    // byte-reinterpret GEP + byte-assembled load path instead of emitting an over-indexed chain.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%union.ETCBlock = type { %struct.anon.9 }
%struct.anon.9 = type { i64 }
%struct.anon = type { %union.anon, %union.anon }
%union.anon = type { %struct.anon.0 }
%struct.anon.0 = type { i32 }
define void @k(ptr addrspace(1) %out, i32 %idx) {
entry:
  %block = alloca %union.ETCBlock, align 4
  %alias = bitcast ptr %block to ptr
  %wide = zext i32 %idx to i64
  %byte = getelementptr inbounds [8 x i8], ptr %alias, i64 0, i64 %wide
  store i8 7, ptr %byte, align 1
  %view = bitcast ptr %block to ptr
  %hi = getelementptr inbounds %struct.anon, ptr %view, i64 0, i32 1, i32 0, i32 0
  %word = load i32, ptr %hi, align 4
  %field = getelementptr %union.ETCBlock, ptr %block, i64 0, i32 0, i32 0
  %bits = load i64, ptr %field, align 4
  %low = trunc i64 %bits to i32
  %sum = add i32 %word, %low
  store i32 %sum, ptr addrspace(1) %out, align 4
  ret void
}

"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_local_aggregate_multi_view_{}",
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
fn native_local_scalar_byte_view_packs_half_lanes_in_byte_storage() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %out) {
entry:
  %slot = alloca float, align 4
  %low_alias = bitcast ptr %slot to ptr
  store half 0xH3C00, ptr %low_alias, align 2
  %high_alias = bitcast ptr %slot to ptr
  %high = getelementptr inbounds i8, ptr %high_alias, i64 2
  store half 0xH4000, ptr %high, align 2
  %packed = load float, ptr %slot, align 4
  store float %packed, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_local_scalar_byte_view_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        !asm.contains("OpPtrAccessChain %_ptr_Function_uchar"),
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
fn native_global_byte_table_reinterpret_view_remodels_to_byte_array() {
    // A packed all-i8-leaf constant table addressed through a DIFFERENT aggregate view with dynamic
    // indices (`[2 x [4 x i8]]` over a struct declaration — the astc `unquantizedWeightTable`
    // shape). A structural chain would dynamically index a struct (invalid); the global must be
    // declared as its flat byte array with the initializer byte image so the byte-array raw paths
    // lower every view.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@tbl = internal unnamed_addr addrspace(2) constant <{ [4 x i8], <{ i8, i8, [2 x i8] }> }> <{ [4 x i8] c"", <{ i8, i8, [2 x i8] }> <{ i8 5, i8 6, [2 x i8] zeroinitializer }> }>, align 1
define void @k(ptr addrspace(1) %out, i32 %r, i32 %c) {
entry:
  %rw = zext i32 %r to i64
  %cw = zext i32 %c to i64
  %p = getelementptr inbounds [2 x [4 x i8]], ptr addrspace(2) @tbl, i64 0, i64 %rw, i64 %cw
  %b = load i8, ptr addrspace(2) %p, align 1
  %w = zext i8 %b to i32
  store i32 %w, ptr addrspace(1) %out, align 4
  ret void
}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_global_byte_view_{}",
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
fn native_local_aggregate_callee_view_valid_after_inline_sroa() {
    // The DecodeETC2 shape: the caller byte-fills a union alloca, and the CONFLICTING struct view
    // lives in an internal callee, so the default per-function emission produces a structurally
    // typed chain against the caller's byte-array storage. The text inliner collapses the views
    // into one function where the multi-view byte-array remodel handles them; the retry tier in
    // `translate_sanitized_native` adopts this emission when it validates.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%union.ETCBlock = type { %struct.anon.9 }
%struct.anon.9 = type { i64 }
%struct.anon = type { %union.anon, %union.anon }
%union.anon = type { %struct.anon.0 }
%struct.anon.0 = type { i32 }
define void @k(ptr addrspace(1) %out, i32 %idx) {
  %block = alloca %union.ETCBlock, align 4
  %alias = bitcast ptr %block to ptr
  %wide = zext i32 %idx to i64
  %byte = getelementptr inbounds [8 x i8], ptr %alias, i64 0, i64 %wide
  store i8 7, ptr %byte, align 1
  %word = call fastcc i32 @helper(ptr noundef nonnull %block)
  store i32 %word, ptr addrspace(1) %out, align 4
  ret void
}
define internal fastcc i32 @helper(ptr noundef %0) {
  %view = bitcast ptr %0 to ptr
  %hi = getelementptr inbounds %struct.anon, ptr %view, i64 0, i32 1, i32 0, i32 0
  %word = load i32, ptr %hi, align 4
  ret i32 %word
}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_local_aggregate_callee_view_{}",
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
fn native_function_aggregate_pointer_field_store_load_replays_pointer_value() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Cache = type { ptr addrspace(1), ptr addrspace(1), float }
define void @store_load_pointer_field(ptr addrspace(1) %buf) {
entry:
  %cache = alloca %struct.Cache, align 8
  %field = getelementptr inbounds %struct.Cache, ptr %cache, i64 0, i32 0
  store ptr addrspace(1) %buf, ptr %field, align 8
  %loaded = load ptr addrspace(1), ptr %field, align 8
  ret void
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpCopyObject"), "{asm}");
    assert!(asm.contains("OpStore"), "{asm}");
}

#[test]
fn native_local_pointer_field_load_preserves_stored_pointer_storage() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Cache = type { ptr }
define i8 @load_local_pointer_field() {
entry:
  %slot = alloca i8, align 1
  %cache = alloca %struct.Cache, align 8
  store i8 7, ptr %slot, align 1
  %field = getelementptr inbounds %struct.Cache, ptr %cache, i64 0, i32 0
  store ptr %slot, ptr %field, align 8
  %loaded = load ptr, ptr %field, align 8
  %byte = load i8, ptr %loaded, align 1
  ret i8 %byte
}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load spv");
    let copy = module
        .all_inst_iter()
        .find(|inst| inst.class.opcode == Op::CopyObject)
        .expect("copy object");
    let source = match copy.operands.first() {
        Some(Operand::IdRef(id)) => *id,
        other => panic!("unexpected copy source {other:?}"),
    };
    let source_type = module
        .all_inst_iter()
        .find(|inst| inst.result_id == Some(source))
        .and_then(|inst| inst.result_type)
        .expect("source type");
    assert_eq!(copy.result_type, Some(source_type), "{asm}");
}

#[test]
fn native_concrete_pointer_round_trips_through_opaque_by_value_wrapper() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Wrapper = type { ptr, i64 }
define void @k(ptr addrspace(1) %out) {
entry:
  %slot = alloca i64, align 8
  store i64 7, ptr %slot, align 8
  %with_pointer = insertvalue %Wrapper poison, ptr %slot, 0
  %complete = insertvalue %Wrapper %with_pointer, i64 1, 1
  %loaded_pointer = extractvalue %Wrapper %complete, 0
  %value = load i64, ptr %loaded_pointer, align 8
  %truncated = trunc i64 %value to i32
  store i32 %truncated, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_opaque_pointer_wrapper_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp)
        .expect("opaque by-value wrapper must preserve its concrete pointer");
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
fn native_inlined_byte_struct_view_extracts_from_same_size_scalar_alloca() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Flags = type { i8, i8, i8, i8, i8, i8, i8, i8 }
define void @k(ptr addrspace(1) %out) {
entry:
  %bits = alloca i64, align 8
  %alias = bitcast ptr %bits to ptr
  store i64 257, ptr %bits, align 8
  %flag = call fastcc i8 @read_flag(ptr %alias)
  %word = zext i8 %flag to i32
  store i32 %word, ptr addrspace(1) %out, align 4
  ret void
}

define internal fastcc i8 @read_flag(ptr %flags) {
entry:
  %field = getelementptr inbounds %Flags, ptr %flags, i64 0, i32 1
  %value = load i8, ptr %field, align 1
  %set = icmp ne i8 %value, 0
  br i1 %set, label %present, label %absent
present:
  ret i8 %value
absent:
  ret i8 0
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint*", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_scalar_alloca_byte_struct_view_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
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
fn native_function_pointer_array_dynamic_load_selects_stored_pointer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @store_via_pointer_table(ptr addrspace(1) %a, ptr addrspace(1) %b, i32 %idx) {
entry:
  %table = alloca [2 x ptr addrspace(1)], align 8
  %slot0 = getelementptr inbounds [2 x ptr addrspace(1)], ptr %table, i64 0, i64 0
  store ptr addrspace(1) %a, ptr %slot0, align 8
  %slot1 = getelementptr inbounds [2 x ptr addrspace(1)], ptr %table, i64 0, i64 1
  store ptr addrspace(1) %b, ptr %slot1, align 8
  %wide = zext i32 %idx to i64
  %slot = getelementptr inbounds [2 x ptr addrspace(1)], ptr %table, i64 0, i64 %wide
  %dst = load ptr addrspace(1), ptr %slot, align 8
  %value = getelementptr inbounds float, ptr addrspace(1) %dst, i64 0
  store float 1.000000e+00, ptr addrspace(1) %value, align 4
  ret void
}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(asm.contains("OpIEqual"), "{asm}");
    assert!(!asm.contains("non-bitcastable result Ptr"), "{asm}");
}

#[test]
fn native_function_pointer_matrix_dynamic_load_selects_stored_pointer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @store_via_pointer_matrix(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, ptr addrspace(1) %d, i32 %row, i32 %column) {
entry:
  %table = alloca [2 x [2 x ptr addrspace(1)]], align 8
  %slot00 = getelementptr inbounds [2 x [2 x ptr addrspace(1)]], ptr %table, i64 0, i64 0, i64 0
  store ptr addrspace(1) %a, ptr %slot00, align 8
  %slot01 = getelementptr inbounds [2 x [2 x ptr addrspace(1)]], ptr %table, i64 0, i64 0, i64 1
  store ptr addrspace(1) %b, ptr %slot01, align 8
  %slot10 = getelementptr inbounds [2 x [2 x ptr addrspace(1)]], ptr %table, i64 0, i64 1, i64 0
  store ptr addrspace(1) %c, ptr %slot10, align 8
  %slot11 = getelementptr inbounds [2 x [2 x ptr addrspace(1)]], ptr %table, i64 0, i64 1, i64 1
  store ptr addrspace(1) %d, ptr %slot11, align 8
  %wide_row = zext i32 %row to i64
  %wide_column = zext i32 %column to i64
  %slot = getelementptr inbounds [2 x [2 x ptr addrspace(1)]], ptr %table, i64 0, i64 %wide_row, i64 %wide_column
  %dst = load ptr addrspace(1), ptr %slot, align 8
  store float 1.000000e+00, ptr addrspace(1) %dst, align 4
  ret void
}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpLogicalAnd"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(!asm.contains("non-bitcastable result Ptr"), "{asm}");
}

#[test]
fn native_bound_buffer_pointer_array_preserves_dynamic_table_sources() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @load_via_pointer_table(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %out, i32 %idx) {
entry:
  %table = alloca [2 x ptr addrspace(1)], align 8
  %slot0 = getelementptr inbounds [2 x ptr addrspace(1)], ptr %table, i64 0, i64 0
  store ptr addrspace(1) %a, ptr %slot0, align 8
  %slot1 = getelementptr inbounds [2 x ptr addrspace(1)], ptr %table, i64 0, i64 1
  store ptr addrspace(1) %b, ptr %slot1, align 8
  %wide = zext i32 %idx to i64
  %slot = getelementptr inbounds [2 x ptr addrspace(1)], ptr %table, i64 0, i64 %wide
  %src = load ptr addrspace(1), ptr %slot, align 8
  %value = load i8, ptr addrspace(1) %src, align 1
  store i8 %value, ptr addrspace(1) %out, align 1
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @load_via_pointer_table, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 1, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 1, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"b"}
!5 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 1, !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_bound_pointer_table_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_native_primary_validated(ll, Stage::Kernel, &tmp)
        .expect("bound pointer table primary must validate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpSelect"), "{asm}");
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
fn native_pointer_bitcast_widening_vector_load_pads_extra_lane() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define float @load_wide_row() {
entry:
  %matrix = alloca [1 x <3 x float>], align 16
  %row = getelementptr inbounds [1 x <3 x float>], ptr %matrix, i64 0, i64 0
  store <3 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, ptr %row, align 16
  %alias = bitcast ptr %row to ptr
  %wide = load <4 x float>, ptr %alias, align 16
  %lane = extractelement <4 x float> %wide, i32 2
  ret float %lane
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpCompositeExtract"), "{asm}");
    assert!(asm.contains("OpCompositeConstruct"), "{asm}");
}

#[test]
fn native_pointer_bitcast_vector_load_store_uses_scalar_lanes() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @half4_alias() {
entry:
  %buf = alloca [16 x half], align 16
  %src = getelementptr inbounds [16 x half], ptr %buf, i64 0, i64 0
  %src_alias = bitcast ptr %src to ptr
  %v = load <4 x half>, ptr %src_alias, align 8
  %dst = getelementptr inbounds [16 x half], ptr %buf, i64 0, i64 4
  %dst_alias = bitcast ptr %dst to ptr
  store <4 x half> %v, ptr %dst_alias, align 8
  ret void
}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_half4_scalar_lanes_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load native spv");
    let out = passes::transform(module, Stage::Kernel, None, None, None, Some("half4_alias"))
        .expect("interface transform")
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let asm = disassemble(&out).expect("disassemble transformed");
    assert!(asm.contains("OpCompositeConstruct"), "{asm}");
    assert!(asm.contains("OpCompositeExtract"), "{asm}");
    assert!(
        !asm.contains("reinterpret load bit width mismatch"),
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
fn native_ptr_network_widen_whole_vs_part_emits_valid_primary() {
    // KEYSTONE-1 soundness gate: a device buffer dereferenced at BOTH `<4 x float>` (whole) and
    // `float` (part) is a whole-vs-part network. The KEYSTONE-1 widen scalarizes the whole
    // access to four `float` ops, so every access on the buffer is a consistent `float*`. Unlike the
    // WHOLE_PART flip (dead-end #15), this NARROWS to the finest granularity — there is no partial
    // retyping — so the PRIMARY emit (no retry) must be spirv-val-VALID. Asserting that here is what
    // keeps the flip honest: a `--list-fail EMPTY` that only holds via retry-rescue is NOT sufficiency.
    let ll = r#"
source_filename = "case.metal"
define void @wvp(ptr addrspace(1) noundef align 4 "air-buffer-no-alias" %0) local_unnamed_addr #0 {
  %v = load <4 x float>, ptr addrspace(1) %0, align 16
  %p = getelementptr inbounds float, ptr addrspace(1) %0, i64 4
  %s = load float, ptr addrspace(1) %p, align 4
  %e = extractelement <4 x float> %v, i32 0
  %t = fadd float %e, %s
  %q = getelementptr inbounds float, ptr addrspace(1) %0, i64 8
  store float %t, ptr addrspace(1) %q, align 4
  ret void
}
attributes #0 = { nounwind }
!air.kernel = !{!0}
!0 = !{ptr @wvp, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 0, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buf"}
"#;
    // Scalarize whole-vs-part deterministically (bypass the env flag), then emit through the PRIMARY,
    // NO-RETRY translate path (`translate_native_no_retry` — buffer-metadata modeling + the shared
    // passes tail, but NO spirv-val gate and NO retry cascade). Asserting spirv-val here proves the
    // primary emit is valid on its own, so the eventual default flip does not lean on retry-rescue.
    let widened = super::super::vec_scalar_merge::lower_with_widen_for_test(ll);
    assert!(
        !widened.contains("load <4 x float>"),
        "whole-vs-part load not scalarized:\n{widened}"
    );

    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_ptr_network_widen_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_native_no_retry(&widened, Stage::Kernel).expect("primary emit");
    let asm = disassemble(&out).expect("disassemble");
    if std::env::var("DBG_ASM").is_ok() {
        eprintln!("{asm}");
    }
    // The whole-vector load is gone, rebuilt from four scalar loads via insertelement.
    assert!(
        asm.matches("OpCompositeInsert").count() >= 4,
        "vector not rebuilt from scalar lanes:\n{asm}"
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
fn native_same_width_vector_reinterpret_store_bitcasts_object() {
    // A <2 x float> (8 bytes) stored through a <4 x half> (8 bytes) pointer is a byte-identical
    // reinterpret: the emitter OpBitcasts the object to the pointee vector before the store.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @v2f_into_v4h(<2 x float> %v) {
entry:
  %buf = alloca <4 x half>, align 8
  store <2 x float> %v, ptr %buf, align 8
  ret void
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpBitcast"), "{asm}");
    assert!(!asm.contains("does not match Object"), "{asm}");
}

#[test]
fn native_pointer_bitcast_i32_word_load_constructs_i16_vector() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%union.Everything = type { <4 x i32> }

define <4 x i16> @load_half_words() {
entry:
  %scratch = alloca %union.Everything, align 16
  %word2 = getelementptr inbounds %union.Everything, ptr %scratch, i64 0, i32 0, i64 2
  store i32 131073, ptr %word2, align 4
  %word3 = getelementptr inbounds %union.Everything, ptr %scratch, i64 0, i32 0, i64 3
  store i32 262147, ptr %word3, align 4
  %alias = bitcast ptr %word2 to ptr
  %v = load <4 x i16>, ptr %alias, align 8
  ret <4 x i16> %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpBitcast"), "{asm}");
    assert!(asm.contains("OpCompositeExtract"), "{asm}");
    assert!(asm.contains("OpCompositeConstruct"), "{asm}");
    assert!(
        !asm.contains("reinterpret load bit width mismatch"),
        "{asm}"
    );
}

#[test]
fn native_pointer_bitcast_narrowing_vector_store_drops_extra_lane() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::matrix" = type { [3 x <3 x float>] }
define void @main() {
entry:
  %v0 = insertelement <4 x float> poison, float 1.000000e+00, i32 0
  %v1 = insertelement <4 x float> %v0, float 2.000000e+00, i32 1
  %v2 = insertelement <4 x float> %v1, float 3.000000e+00, i32 2
  %v3 = insertelement <4 x float> %v2, float 4.000000e+00, i32 3
  %matrix = alloca [1 x %"struct.metal::matrix"], align 16
  %slot = getelementptr inbounds [1 x %"struct.metal::matrix"], ptr %matrix, i64 0, i64 0
  %row0 = bitcast ptr %slot to ptr
  %row1 = getelementptr inbounds [1 x %"struct.metal::matrix"], ptr %matrix, i64 0, i64 0, i32 0, i64 1
  %row1_alias = bitcast ptr %row1 to ptr
  store <4 x float> %v3, ptr %row0, align 16
  store <4 x float> %v3, ptr %row1_alias, align 16
  ret void
}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_narrowing_vector_store_{}",
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
    assert_eq!(asm.matches("OpVectorShuffle").count(), 2, "{asm}");
    assert!(asm.contains(" 0 1 2"), "{asm}");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_i64_load_store_reinterprets_i32_pair_struct() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Pair = type { i32, i32 }
define void @copy_pair(ptr addrspace(1) %dst, ptr addrspace(1) %src, i64 %idx) {
entry:
  %src_record = getelementptr inbounds %struct.Pair, ptr addrspace(1) %src, i64 %idx
  %src_bits = bitcast ptr addrspace(1) %src_record to ptr addrspace(1)
  %bits = load i64, ptr addrspace(1) %src_bits, align 4
  %dst_record = getelementptr inbounds %struct.Pair, ptr addrspace(1) %dst, i64 %idx
  %dst_bits = bitcast ptr addrspace(1) %dst_record to ptr addrspace(1)
  store i64 %bits, ptr addrspace(1) %dst_bits, align 4
  ret void
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpCompositeExtract"), "{asm}");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(asm.contains("OpBitwiseOr"), "{asm}");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
}

#[test]
fn native_loaded_pointer_can_feed_gep() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.S = type { ptr addrspace(1) }
define i32 @load_loaded_pointer(ptr addrspace(2) %p, i64 %idx) {
entry:
  %field = getelementptr inbounds %struct.S, ptr addrspace(2) %p, i64 0, i32 0
  %base = load ptr addrspace(1), ptr addrspace(2) %field
  %elt = getelementptr inbounds i32, ptr addrspace(1) %base, i64 %idx
  %v = load i32, ptr addrspace(1) %elt
  ret i32 %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpLoad"), "{asm}");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
}

#[test]
fn native_i8_pointer_can_load_wider_type() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i32 @load_from_byte_pointer(ptr addrspace(2) %p, i64 %idx) {
entry:
  %word = shl i64 %idx, 2
  %byte = getelementptr inbounds i8, ptr addrspace(2) %p, i64 %word
  %alias = bitcast ptr addrspace(2) %byte to ptr addrspace(2)
  %v = load i32, ptr addrspace(2) %alias
  ret i32 %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpUDiv"), "{asm}");
    assert!(asm.contains("OpLoad"), "{asm}");
    assert!(
        !asm.lines()
            .any(|line| line.contains(" OpBitcast ") && line.contains("_ptr_")),
        "{asm}"
    );
}

#[test]
fn native_i8_buffer_can_store_wider_depth_stencil_words() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"

define void @CopyD32S8ToBuffer(<2 x i32> %tid, ptr addrspace(1) %depth, ptr addrspace(1) %stencil, ptr addrspace(1) %output, ptr addrspace(2) %rowPitch) {
entry:
  %depthSample = tail call { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %depth, <2 x i32> %tid, i32 0, i32 1)
  %depthVec = extractvalue { <4 x float>, i8 } %depthSample, 0
  %depthValue = extractelement <4 x float> %depthVec, i64 0
  %stencilSample = tail call { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1) %stencil, <2 x i32> %tid, i32 0, i32 1)
  %stencilVec = extractvalue { <4 x i32>, i8 } %stencilSample, 0
  %stencilValue = extractelement <4 x i32> %stencilVec, i64 0
  %pitch = load i32, ptr addrspace(2) %rowPitch, align 4
  %y = extractelement <2 x i32> %tid, i64 1
  %rowBytes32 = mul i32 %pitch, %y
  %rowBytes = zext i32 %rowBytes32 to i64
  %row = getelementptr inbounds i8, ptr addrspace(1) %output, i64 %rowBytes
  %x = extractelement <2 x i32> %tid, i64 0
  %xBytes32 = shl i32 %x, 3
  %xBytes = zext i32 %xBytes32 to i64
  %slot = getelementptr inbounds i8, ptr addrspace(1) %row, i64 %xBytes
  %depthOut = bitcast ptr addrspace(1) %slot to ptr addrspace(1)
  store float %depthValue, ptr addrspace(1) %depthOut, align 4
  %stencilSlot = getelementptr inbounds i8, ptr addrspace(1) %slot, i64 4
  %stencilOut = bitcast ptr addrspace(1) %stencilSlot to ptr addrspace(1)
  store i32 %stencilValue, ptr addrspace(1) %stencilOut, align 4
  ret void
}

declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, i32, i32)
declare { <4 x i32>, i8 } @air.read_texture_2d.u.v4i32(ptr addrspace(1), <2 x i32>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @CopyD32S8ToBuffer, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6, !7}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"threadID"}
!4 = !{i32 1, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"depthView"}
!5 = !{i32 2, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<uint, read>", !"air.arg_name", !"stencilView"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"output"}
!7 = !{i32 4, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"rowAndPlanePitch"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_i8_buffer_raw_stores_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpBitcast"), "{asm}");
    assert!(asm.contains("OpUDiv"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_i8_buffer_bitcast_float_load_uses_raw_offset() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @ByteFloatLoad(ptr addrspace(1) %bytes, ptr addrspace(1) %out, i32 %tid) {
entry:
  %byte32 = shl i32 %tid, 2
  %byte = zext i32 %byte32 to i64
  %slot = getelementptr inbounds i8, ptr addrspace(1) %bytes, i64 %byte
  %floatSlot = bitcast ptr addrspace(1) %slot to ptr addrspace(1)
  %v = load float, ptr addrspace(1) %floatSlot, align 4
  %dst = getelementptr inbounds float, ptr addrspace(1) %out, i32 %tid
  store float %v, ptr addrspace(1) %dst, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @ByteFloatLoad, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"bytes"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_i8_buffer_bitcast_float_load_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpUDiv"), "{asm}");
    let binding0_count = asm
        .lines()
        .filter(|line| line.contains("OpDecorate") && line.contains(" Binding 0"))
        .count();
    assert_eq!(binding0_count, 1, "{asm}");
    assert!(
        asm.lines().any(|line| line.contains(" OpBitcast ")),
        "{asm}"
    );
    assert!(
        !asm.lines()
            .any(|line| line.contains(" OpBitcast ") && line.contains("_ptr_")),
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
fn native_raw_struct_array_dynamic_member_index_preserves_offset() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

%Counters = type <{ i32, [16 x i32] }>

define void @ParticleLike(i32 %idx, ptr addrspace(2) %header, ptr addrspace(1) %data, ptr addrspace(1) %out) {
entry:
  %base32 = load i32, ptr addrspace(2) %header, align 4
  %base64 = sext i32 %base32 to i64
  %base = getelementptr inbounds i8, ptr addrspace(1) %data, i64 %base64
  %typed = bitcast ptr addrspace(1) %base to ptr addrspace(1)
  %idx64 = zext i32 %idx to i64
  %slot = getelementptr inbounds %Counters, ptr addrspace(1) %typed, i64 0, i32 1, i64 %idx64
  %v = load i32, ptr addrspace(1) %slot, align 4
  %dst = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %idx64
  store i32 %v, ptr addrspace(1) %dst, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @ParticleLike, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"idx"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"header"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"data"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_raw_struct_array_dynamic_index_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpLoad"), "{asm}");
    assert!(!asm.contains("raw buffer offset is not modelable"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_raw_i8_load_uses_dynamic_byte_lane() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @ByteLane(i32 %tid, ptr addrspace(1) %bytes, ptr addrspace(1) %out) {
entry:
  %wide_i32 = shl i32 %tid, 2
  %wide_i64 = zext i32 %wide_i32 to i64
  %wide_byte = getelementptr inbounds i8, ptr addrspace(1) %bytes, i64 %wide_i64
  %wide_ptr = bitcast ptr addrspace(1) %wide_byte to ptr addrspace(1)
  %wide = load <4 x i8>, ptr addrspace(1) %wide_ptr, align 4
  %wide0 = extractelement <4 x i8> %wide, i64 0
  %idx = zext i32 %tid to i64
  %byte_ptr = getelementptr inbounds i8, ptr addrspace(1) %bytes, i64 %idx
  %byte = load i8, ptr addrspace(1) %byte_ptr, align 1
  %byte32 = zext i8 %byte to i32
  %wide32 = zext i8 %wide0 to i32
  %sum = add i32 %byte32, %wide32
  %dst = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %idx
  store i32 %sum, ptr addrspace(1) %dst, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @ByteLane, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"bytes"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_raw_i8_dynamic_lane_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpVectorExtractDynamic"), "{asm}");
    assert!(asm.contains("OpUDiv"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_raw_i8_vector_source_gep_load_uses_scalar_byte_loads() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @ByteVectorGep(i32 %tid, ptr addrspace(1) %bytes, ptr addrspace(1) %out) {
entry:
  %lane = zext i32 %tid to i64
  %base_lane = getelementptr inbounds <4 x i8>, ptr addrspace(1) %bytes, i64 0, i64 %lane
  %base_bytes = bitcast ptr addrspace(1) %base_lane to ptr addrspace(1)
  %quad_i32 = shl i32 %tid, 2
  %quad = zext i32 %quad_i32 to i64
  %quad_ptr = getelementptr inbounds <4 x i8>, ptr addrspace(1) %base_bytes, i64 %quad
  %wide = load <4 x i8>, ptr addrspace(1) %quad_ptr, align 4
  %wide0 = extractelement <4 x i8> %wide, i64 0
  %idx = zext i32 %tid to i64
  %dst = getelementptr inbounds i8, ptr addrspace(1) %out, i64 %idx
  store i8 %wide0, ptr addrspace(1) %dst, align 1
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @ByteVectorGep, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uchar4", !"air.arg_name", !"bytes"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_raw_i8_vector_source_gep_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCompositeConstruct"), "{asm}");
    assert!(
        !asm.lines().any(|line| line.contains("OpLoad %v4uchar")),
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
fn native_raw_pointer_load_from_dynamic_byte_offset_uses_placeholder() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @PointerLoad(i32 %idx, ptr addrspace(1) %bytes, ptr addrspace(1) %out) {
entry:
  %idx64 = zext i32 %idx to i64
  %slot = getelementptr inbounds i8, ptr addrspace(1) %bytes, i64 %idx64
  %ptr_slot = bitcast ptr addrspace(1) %slot to ptr addrspace(1)
  %loaded = load ptr addrspace(1), ptr addrspace(1) %ptr_slot, align 1
  %dst = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %idx64
  store i32 0, ptr addrspace(1) %dst, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @PointerLoad, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"idx"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"bytes"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_raw_pointer_load_dynamic_byte_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpVariable"), "{asm}");
    assert!(!asm.contains("raw dynamic byte stride"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_raw_scalar_array_reinterpret_gep_composes_byte_offsets() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @DecodePackedRGB8(i32 %x, ptr addrspace(1) %src, ptr addrspace(2) %stride, ptr addrspace(1) %out) {
entry:
  %row_stride = load i32, ptr addrspace(2) %stride, align 4
  %row_bytes = mul i32 %row_stride, %x
  %row64 = zext i32 %row_bytes to i64
  %row_ptr = getelementptr inbounds [3 x i8], ptr addrspace(1) %src, i64 0, i64 %row64
  %typed = bitcast ptr addrspace(1) %row_ptr to ptr addrspace(1)
  %x64 = zext i32 %x to i64
  %rptr = getelementptr inbounds [3 x i8], ptr addrspace(1) %typed, i64 %x64, i64 0
  %gptr = getelementptr inbounds [3 x i8], ptr addrspace(1) %typed, i64 %x64, i64 1
  %bptr = getelementptr inbounds [3 x i8], ptr addrspace(1) %typed, i64 %x64, i64 2
  %r = load i8, ptr addrspace(1) %rptr, align 1
  %g = load i8, ptr addrspace(1) %gptr, align 1
  %b = load i8, ptr addrspace(1) %bptr, align 1
  %r32 = zext i8 %r to i32
  %g32 = zext i8 %g to i32
  %b32 = zext i8 %b to i32
  %rg = add i32 %r32, %g32
  %rgb = add i32 %rg, %b32
  %dst = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %x64
  store i32 %rgb, ptr addrspace(1) %dst, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @DecodePackedRGB8, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"x"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 3, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"packed_uchar3", !"air.arg_name", !"src"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"stride"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_raw_scalar_array_gep_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_native_no_retry(ll, Stage::Kernel).expect("primary emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpIMul"), "{asm}");
    assert!(
        !asm.lines().any(|line| {
            line.contains("OpPtrAccessChain") && line.contains("_ptr_StorageBuffer_uchar")
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

#[test]
fn native_local_pointer_table_gep_infers_buffer_param_pointees() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %a, ptr addrspace(1) %b, i32 %which, i64 %offset, float %v) {
entry:
  %table = alloca [2 x ptr addrspace(1)], align 8
  %slot0 = getelementptr inbounds [2 x ptr addrspace(1)], ptr %table, i64 0, i64 0
  store ptr addrspace(1) %a, ptr %slot0, align 8
  %slot1 = getelementptr inbounds [2 x ptr addrspace(1)], ptr %table, i64 0, i64 1
  store ptr addrspace(1) %b, ptr %slot1, align 8
  %idx = zext i32 %which to i64
  %slot = getelementptr inbounds [2 x ptr addrspace(1)], ptr %table, i64 0, i64 %idx
  %base = load ptr addrspace(1), ptr %slot, align 8
  %ptr = getelementptr inbounds float, ptr addrspace(1) %base, i64 %offset
  store float %v, ptr addrspace(1) %ptr, align 4
  ret void
}
"#;
    let ir = super::super::ir::LlModule::parse(ll).expect("parse");
    assert_eq!(
        ir.ptr_pointees.get(&("k".to_string(), "%a".to_string())),
        Some(&LlType::Float)
    );
    assert_eq!(
        ir.ptr_pointees.get(&("k".to_string(), "%b".to_string())),
        Some(&LlType::Float)
    );
}

#[test]
fn native_local_pointer_table_dynamic_gep_selects_per_arm_access_chains() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %a, ptr addrspace(1) %b, i32 %which, i64 %offset, float %v) {
entry:
  %table = alloca [2 x ptr addrspace(1)], align 8
  %slot0 = getelementptr inbounds [2 x ptr addrspace(1)], ptr %table, i64 0, i64 0
  store ptr addrspace(1) %a, ptr %slot0, align 8
  %slot1 = getelementptr inbounds [2 x ptr addrspace(1)], ptr %table, i64 0, i64 1
  store ptr addrspace(1) %b, ptr %slot1, align 8
  %idx = zext i32 %which to i64
  %slot = getelementptr inbounds [2 x ptr addrspace(1)], ptr %table, i64 0, i64 %idx
  %base = load ptr addrspace(1), ptr %slot, align 8
  %ptr = getelementptr inbounds float, ptr addrspace(1) %base, i64 %offset
  store float %v, ptr addrspace(1) %ptr, align 4
  ret void
}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(asm.contains("OpAccessChain"), "{asm}");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(!asm.contains("OpPtrAccessChain"), "{asm}");
}

#[test]
fn native_vector_store_reinterprets_same_width_scalar_lanes() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @store_half2_lanes(ptr addrspace(3) %scratch, i32 %idx, <4 x float> %packed) {
entry:
  %wide = zext i32 %idx to i64
  %slot = getelementptr inbounds <2 x half>, ptr addrspace(3) %scratch, i64 %wide
  %cast = bitcast ptr addrspace(3) %slot to ptr addrspace(3)
  store <4 x float> %packed, ptr addrspace(3) %cast, align 16
  ret void
}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpCompositeExtract"), "{asm}");
    assert!(asm.contains("OpBitcast"), "{asm}");
    assert!(asm.contains("OpStore"), "{asm}");
}

#[test]
fn native_scalar_store_reinterprets_same_width_pointee() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %buf, i16 %tid) {
entry:
  %idx = zext i16 %tid to i64
  %slot = getelementptr inbounds i32, ptr addrspace(1) %buf, i64 %idx
  %cast = bitcast ptr addrspace(1) %slot to ptr addrspace(1)
  store float 1.000000e+00, ptr addrspace(1) %cast, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"buf"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_scalar_store_reinterpret_{}",
        std::process::id()
    ));
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpBitcast"), "{asm}");
    assert!(asm.contains("OpStore"), "{asm}");
}

#[test]
fn native_selected_raw_byte_pointer_half_store_keeps_raw_offset() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %a, i32 %i) {
entry:
  %wide = zext i32 %i to i64
  %cond = icmp eq i32 %i, 0
  %wordp = getelementptr inbounds i32, ptr addrspace(1) %a, i64 %wide
  %word = load i32, ptr addrspace(1) %wordp, align 4
  %abyte = getelementptr inbounds i8, ptr addrspace(1) %a, i64 0
  %bbyte = getelementptr inbounds i8, ptr addrspace(1) %a, i64 4
  %sel = select i1 %cond, ptr addrspace(1) %abyte, ptr addrspace(1) %bbyte
  %halfp = getelementptr inbounds half, ptr addrspace(1) %sel, i64 %wide
  store half 0xH3C00, ptr addrspace(1) %halfp, align 2
  store i32 %word, ptr addrspace(1) %wordp, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_selected_raw_half_store_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(!asm.contains("_ptr_Private_uint"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_raw_load_accepts_mul_by_aligned_select_byte_step() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i32 @load_raw_step(ptr addrspace(2) %p, i1 %wide, i32 %idx) {
entry:
  %tag = load i32, ptr addrspace(2) %p
  %step32 = select i1 %wide, i32 16, i32 36
  %bytes32 = mul i32 %idx, %step32
  %bytes = zext i32 %bytes32 to i64
  %byte = getelementptr inbounds i8, ptr addrspace(2) %p, i64 %bytes
  %v = load i32, ptr addrspace(2) %byte
  %out = add i32 %v, %tag
  ret i32 %out
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpUDiv"), "{asm}");
    assert!(asm.contains("OpLoad"), "{asm}");
}

#[test]
fn native_raw_half_load_uses_dynamic_lane_for_half_stride() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define half @load_raw_half(ptr addrspace(1) %p, i32 %idx) {
entry:
  %tag = load i32, ptr addrspace(1) %p
  %idx64 = zext i32 %idx to i64
  %halfp = getelementptr inbounds half, ptr addrspace(1) %p, i64 %idx64
  %v = load half, ptr addrspace(1) %halfp
  ret half %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpVectorExtractDynamic"), "{asm}");
    assert!(asm.contains("OpLoad"), "{asm}");
}

#[test]
fn native_raw_subword_stores_use_dynamic_lane_read_modify_write() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @store_raw_subwords(ptr addrspace(1) %p, i32 %idx, half %h, i16 %s, i8 %b) {
entry:
  %tag = load i32, ptr addrspace(1) %p
  %idx64 = zext i32 %idx to i64
  %bytep = getelementptr inbounds i8, ptr addrspace(1) %p, i64 %idx64
  %halfp = bitcast ptr addrspace(1) %bytep to ptr addrspace(1)
  store half %h, ptr addrspace(1) %halfp, align 2
  %next = getelementptr inbounds i8, ptr addrspace(1) %bytep, i64 1
  %shortp = bitcast ptr addrspace(1) %next to ptr addrspace(1)
  store i16 %s, ptr addrspace(1) %shortp, align 1
  store i8 %b, ptr addrspace(1) %bytep, align 1
  ret void
}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpAtomicAnd"), "{asm}");
    assert!(asm.contains("OpAtomicOr"), "{asm}");
    assert!(asm.contains("OpBitcast"), "{asm}");
    assert!(asm.contains("OpBitwiseXor"), "{asm}");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
}

#[test]
fn native_raw_i16_load_assembles_unaligned_byte_offset() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i16 @load_unaligned_u16(ptr addrspace(1) %p, i32 %idx) {
entry:
  %idx64 = zext i32 %idx to i64
  %bytep = getelementptr inbounds i8, ptr addrspace(1) %p, i64 %idx64
  %arrayp = bitcast ptr addrspace(1) %bytep to ptr addrspace(1)
  %shortp = getelementptr inbounds [2 x i16], ptr addrspace(1) %arrayp, i64 %idx64, i64 0
  %v = load i16, ptr addrspace(1) %shortp, align 1
  ret i16 %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpUConvert"), "{asm}");
}

#[test]
fn native_raw_i64_load_combines_two_words() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define i64 @load_raw_i64(ptr addrspace(2) %p, i64 %idx) {
entry:
  %tag = load i32, ptr addrspace(2) %p
  %byte = getelementptr inbounds i8, ptr addrspace(2) %p, i64 8
  %v = load i64, ptr addrspace(2) %byte
  %tag64 = zext i32 %tag to i64
  %out = xor i64 %v, %tag64
  ret i64 %out
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpBitwiseOr"), "{asm}");
    assert!(asm.contains("OpBitwiseXor"), "{asm}");
}

#[test]
fn native_direct_load_infers_pointer_pointee_type() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define float @load_scalar(ptr addrspace(2) %p) {
entry:
  %v = load float, ptr addrspace(2) %p
  ret float %v
}
"#;
    let module = load_bytes(emit_vulkan_spirv(ll).expect("native emit")).expect("load spv");
    let param_ptr_ty = module.functions[0].parameters[0]
        .result_type
        .expect("param pointer type");
    let ptr = module
        .types_global_values
        .iter()
        .find(|inst| inst.result_id == Some(param_ptr_ty))
        .expect("pointer type");
    let Operand::IdRef(pointee) = ptr.operands[1] else {
        panic!("pointer type should have pointee operand");
    };
    let pointee_ty = module
        .types_global_values
        .iter()
        .find(|inst| inst.result_id == Some(pointee))
        .expect("pointee type");
    assert_eq!(pointee_ty.class.opcode, Op::TypeFloat);
    assert_eq!(pointee_ty.operands[0], Operand::LiteralBit32(32));
}

#[test]
fn native_gep_keeps_dynamic_first_index_for_element_buffers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define float @load_indexed(ptr addrspace(1) %p, i64 %idx) {
entry:
  %g = getelementptr inbounds float, ptr addrspace(1) %p, i64 %idx
  %v = load float, ptr addrspace(1) %g
  ret float %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(asm.contains("OpLoad"), "{asm}");
}

#[test]
fn native_identity_bitcast_param_gep_uses_access_chain() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define float @load_indexed_alias(ptr addrspace(1) %p, i64 %idx) {
entry:
  %alias = bitcast ptr addrspace(1) %p to ptr addrspace(1)
  %g = getelementptr inbounds float, ptr addrspace(1) %alias, i64 %idx
  %v = load float, ptr addrspace(1) %g
  ret float %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(!asm.contains("OpPtrAccessChain"), "{asm}");
}

#[test]
fn native_kernel_packed_buffer_metadata_keeps_record_lane_geps_valid() {
    let ll = r#"
source_filename = "packed-buffer"

define void @packed_kernel(ptr addrspace(1) %src, ptr addrspace(1) %dst, <3 x i32> %gid) {
entry:
  %x = extractelement <3 x i32> %gid, i64 0
  %idx = zext i32 %x to i64
  %src0 = getelementptr inbounds [3 x float], ptr addrspace(1) %src, i64 %idx, i64 0
  %a = load float, ptr addrspace(1) %src0, align 4
  %src1 = getelementptr inbounds [3 x float], ptr addrspace(1) %src, i64 0, i64 1
  %b = load float, ptr addrspace(1) %src1, align 4
  %dst0 = getelementptr inbounds [2 x float], ptr addrspace(1) %dst, i64 %idx, i64 0
  store float %a, ptr addrspace(1) %dst0, align 4
  %dst1 = getelementptr inbounds [2 x float], ptr addrspace(1) %dst, i64 0, i64 1
  store float %b, ptr addrspace(1) %dst1, align 4
  ret void
}

!air.version = !{!0}
!air.language_version = !{!1}
!air.kernel = !{!2}
!0 = !{i32 2, i32 8, i32 0}
!1 = !{!"Metal", i32 3, i32 0, i32 0}
!2 = !{ptr @packed_kernel, !3, !4}
!3 = !{}
!4 = !{!5, !6, !7}
!5 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"packed_float3", !"air.arg_name", !"src"}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"packed_float2", !"air.arg_name", !"dst"}
!7 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_packed_buffer_gep_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble");
    if std::env::var("DBG_ASM").is_ok() {
        eprintln!("{asm}");
    }
    assert!(asm.contains("OpTypeRuntimeArray"), "{asm}");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_integer_alloca_retyped_by_half_array_gep_view() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define half @local_half_array_view(i64 %idx, half %value) {
entry:
  %slot = alloca i64, align 8
  %alias = bitcast ptr %slot to ptr
  store i64 0, ptr %slot, align 8
  %p = getelementptr inbounds [4 x half], ptr %alias, i64 0, i64 %idx
  store half %value, ptr %p, align 2
  %v = load half, ptr %p, align 2
  ret half %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpTypeArray"), "{asm}");
    assert!(asm.contains("OpConstantNull"), "{asm}");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
}

#[test]
fn native_collapses_redundant_wrapper_gep_on_derived_pointer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%air.arrblk.0 = type { [0 x float] }
define float @wrapped(ptr addrspace(1) %b, i64 %idx) {
entry:
  %arrayidx = getelementptr inbounds %air.arrblk.0, ptr addrspace(1) %b, i64 0, i32 0, i64 %idx
  %aircanon.0 = getelementptr inbounds %air.arrblk.0, ptr addrspace(1) %arrayidx, i64 0, i32 0, i64 0
  %v = load float, ptr addrspace(1) %aircanon.0
  ret float %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    let chains = asm.matches("OpInBoundsAccessChain").count();
    assert_eq!(chains, 1, "{asm}");
    assert!(asm.contains("OpLoad"), "{asm}");
}

#[test]
fn native_composes_zero_wrapper_gep_offsets_from_derived_pointer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%air.arrblk.0 = type { [0 x <4 x float>] }
define <4 x float> @wrapped_offset(ptr addrspace(2) %b) {
entry:
  %arrayidx = getelementptr inbounds %air.arrblk.0, ptr addrspace(2) %b, i64 0, i32 0, i64 7
  %next = getelementptr inbounds %air.arrblk.0, ptr addrspace(2) %arrayidx, i64 0, i32 0, i64 1
  %v = load <4 x float>, ptr addrspace(2) %next
  ret <4 x float> %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    let access_chains = asm
        .lines()
        .filter(|line| line.contains("OpInBoundsAccessChain"))
        .collect::<Vec<_>>();
    assert_eq!(access_chains.len(), 2, "{asm}");
    assert!(asm.lines().any(|line| line.ends_with(" 8")), "{asm}");
    assert!(!asm.lines().any(|line| line.ends_with(" 1")), "{asm}");
}

#[test]
fn native_composes_dynamic_zero_wrapper_gep_offsets() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%air.arrblk.0 = type { [0 x <4 x float>] }
define <4 x float> @wrapped_offset(ptr addrspace(2) %b, i64 %idx) {
entry:
  %arrayidx = getelementptr inbounds %air.arrblk.0, ptr addrspace(2) %b, i64 0, i32 0, i64 %idx
  %next = getelementptr inbounds %air.arrblk.0, ptr addrspace(2) %arrayidx, i64 0, i32 0, i64 1
  %v = load <4 x float>, ptr addrspace(2) %next
  ret <4 x float> %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpIAdd"), "{asm}");
    assert_eq!(asm.matches("OpInBoundsAccessChain").count(), 2, "{asm}");
}

#[test]
fn native_composes_linear_scalar_gep_offsets() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define float @scalar_offset(ptr addrspace(2) %b, i64 %idx) {
entry:
  %base = getelementptr inbounds float, ptr addrspace(2) %b, i64 %idx
  %next = getelementptr inbounds float, ptr addrspace(2) %base, i64 1
  %v = load float, ptr addrspace(2) %next
  ret float %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpIAdd"), "{asm}");
    let access_chains = asm
        .lines()
        .filter(|line| line.contains("OpInBoundsAccessChain"))
        .collect::<Vec<_>>();
    assert_eq!(access_chains.len(), 2, "{asm}");
    let first_result = access_chains[0].split('=').next().unwrap().trim();
    assert!(
        !access_chains[1].contains(&format!(" {first_result} ")),
        "{asm}"
    );
}

#[test]
fn native_whole_buffer_param_select_loads_values_before_selecting() {
    // A select between two WHOLE metadata-declared `air.buffer` params (an L/R buffer pair) then a
    // typed load at offset 0 — no GEP anywhere. The arms are direct entry params, which the
    // deferred load-typed select path used to reject wholesale (the texture-arm guard); a
    // metadata-declared data buffer arm is safe to load-and-select
    // (CC_CopyVirtualPixelStatsToReadbackTexture 0f853fb0).
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

define void @k(ptr addrspace(1) %0, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %1, ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %2, <2 x i16> noundef %3) local_unnamed_addr {
  %5 = extractelement <2 x i16> %3, i64 0
  %6 = icmp eq i16 %5, 1
  %7 = select i1 %6, ptr addrspace(1) %2, ptr addrspace(1) %1
  %8 = load <4 x half>, ptr addrspace(1) %7, align 8
  %9 = tail call fast <4 x float> @air.convert.f.v4f32.f.v4f16(<4 x half> %8)
  %10 = insertelement <2 x i16> %3, i16 2, i64 1
  tail call void @air.write_texture_2d.i16.v4f32(ptr addrspace(1) captures(none) %0, <2 x i16> %10, <4 x float> %9, i16 0, i32 3)
  ret void
}

declare void @air.write_texture_2d.i16.v4f32(ptr addrspace(1) captures(none), <2 x i16>, <4 x float>, i16, i32)
declare <4 x float> @air.convert.f.v4f32.f.v4f16(<4 x half>)

!air.kernel = !{!15}
!15 = !{ptr @k, !16, !17}
!16 = !{}
!17 = !{!18, !19, !20, !21}
!18 = !{i32 0, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.arg_type_name", !"texture2d<float, read_write>", !"air.arg_name", !"tex"}
!19 = !{i32 1, !"air.buffer", !"air.location_index", i32 12, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"bufL"}
!20 = !{i32 2, !"air.buffer", !"air.location_index", i32 13, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"bufR"}
!21 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"gid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_whole_buffer_select_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    // The load must be per-arm loads + a VALUE select — never an OpSelect of buffer pointers
    // consumed by a typed OpLoad.
    let pointer_selects = asm
        .lines()
        .filter(|line| line.contains("OpSelect") && line.contains("_ptr_"))
        .collect::<Vec<_>>();
    assert!(pointer_selects.is_empty(), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_storage_buffer_pointer_select_loads_values_before_selecting() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Buffer = type { [4 x half], [4 x half] }
define half @main(ptr addrspace(1) %buf, i1 %cond, i32 %idx) {
entry:
  %a = getelementptr inbounds %Buffer, ptr addrspace(1) %buf, i64 0, i32 0
  %b = getelementptr inbounds %Buffer, ptr addrspace(1) %buf, i64 0, i32 1
  %sel = select i1 %cond, ptr addrspace(1) %a, ptr addrspace(1) %b
  %p = getelementptr inbounds [4 x half], ptr addrspace(1) %sel, i64 0, i32 %idx
  %v = load half, ptr addrspace(1) %p, align 2
  ret half %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    let pointer_selects = asm
        .lines()
        .filter(|line| {
            line.contains("OpSelect")
                && (line.contains("_ptr_StorageBuffer") || line.contains("_ptr_UniformConstant"))
        })
        .collect::<Vec<_>>();
    let value_loads = asm.lines().filter(|line| line.contains("OpLoad")).count();
    assert!(pointer_selects.is_empty(), "{asm}");
    assert_eq!(value_loads, 2, "{asm}");
    assert!(
        asm.lines().any(|line| {
            line.contains("OpSelect")
                && !line.contains("_ptr_StorageBuffer")
                && !line.contains("_ptr_UniformConstant")
        }),
        "{asm}"
    );
}

#[test]
fn native_mixed_private_uniform_pointer_select_replays_loaded_values() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Config = type { i32, i32 }
@default_config = internal addrspace(2) global %Config zeroinitializer, align 4

define i32 @main(ptr addrspace(2) %argument_config, i1 %use_default) {
entry:
  %selected = select i1 %use_default, ptr addrspace(2) @default_config, ptr addrspace(2) %argument_config
  %field = getelementptr inbounds %Config, ptr addrspace(2) %selected, i64 0, i32 1
  %value = load i32, ptr addrspace(2) %field, align 4
  ret i32 %value
}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let asm = disassemble(&spv).expect("disassemble");
    let pointer_selects = asm
        .lines()
        .filter(|line| line.contains("OpSelect") && line.contains("_ptr_"))
        .collect::<Vec<_>>();
    assert!(pointer_selects.is_empty(), "{asm}");
    assert_eq!(asm.matches("OpLoad").count(), 2, "{asm}");
    assert_eq!(asm.matches("OpSelect").count(), 1, "{asm}");
}

#[test]
fn native_inlined_helper_forwards_mixed_storage_pointer_select() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"
%Config = type { i32, i32 }
@default_config = internal addrspace(2) global %Config zeroinitializer, align 4

define void @k(ptr addrspace(1) %out, ptr addrspace(2) %argument_config) {
entry:
  %selected = select i1 true, ptr addrspace(2) @default_config, ptr addrspace(2) %argument_config
  %value = call i32 @read_second(ptr addrspace(2) %selected)
  store i32 %value, ptr addrspace(1) %out, align 4
  ret void
}

define internal i32 @read_second(ptr addrspace(2) %config) {
entry:
  %field = getelementptr inbounds %Config, ptr addrspace(2) %config, i64 0, i32 1
  %value = load i32, ptr addrspace(2) %field, align 4
  ret i32 %value
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"int", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Config", !"air.arg_name", !"argument_config"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_inline_selected_pointer_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let pointer_selects = asm
        .lines()
        .filter(|line| line.contains("OpSelect") && line.contains("_ptr_"))
        .collect::<Vec<_>>();
    assert!(pointer_selects.is_empty(), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
    assert!(asm.matches("OpLoad").count() >= 2, "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn native_selected_pointer_reads_narrow_prefix_in_value_space() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"
@narrow = internal addrspace(2) global i32 7, align 4

define <4 x float> @frag(<4 x float> %position, ptr addrspace(2) %wide, i1 %runtime) {
entry:
  %declared = load i64, ptr addrspace(2) %wide, align 8
  %selected = select i1 %runtime, ptr addrspace(2) %wide, ptr addrspace(2) @narrow
  %value = load i32, ptr addrspace(2) %selected, align 4
  %bits = bitcast i32 %value to float
  %out = insertelement <4 x float> zeroinitializer, float %bits, i32 0
  ret <4 x float> %out
}

!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5, !6}
!4 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"wide"}
!6 = !{i32 2, !"air.fragment_input", !"runtime", !"air.flat", !"air.arg_type_name", !"bool", !"air.arg_name", !"runtime"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_selected_pointer_narrow_prefix_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Fragment, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpUConvert"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
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
fn native_scalar_store_to_private_aggregate_root_targets_first_field() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Config = type { i8, i32 }
@config = internal addrspace(2) global %Config zeroinitializer, align 4

define void @initialize(i8 %value) {
entry:
  store i8 %value, ptr addrspace(2) @config, align 4
  ret void
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert_eq!(asm.matches("OpInBoundsAccessChain").count(), 1, "{asm}");
    let store = asm
        .lines()
        .find(|line| line.contains("OpStore"))
        .expect("scalar store");
    assert!(!store.contains("%config "), "{asm}");
}

#[test]
fn native_selected_storage_buffer_gep_chain_loads_values_before_selecting() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%Buffer = type { [8 x float], [8 x float] }
define float @main(ptr addrspace(1) %buf, i1 %cond, i32 %base, i32 %idx) {
entry:
  %a = getelementptr inbounds %Buffer, ptr addrspace(1) %buf, i64 0, i32 0, i32 0
  %b = getelementptr inbounds %Buffer, ptr addrspace(1) %buf, i64 0, i32 1, i32 0
  %sel = select i1 %cond, ptr addrspace(1) %a, ptr addrspace(1) %b
  %p = getelementptr inbounds float, ptr addrspace(1) %sel, i32 %base
  %q = getelementptr inbounds float, ptr addrspace(1) %p, i32 %idx
  %v = load float, ptr addrspace(1) %q, align 4
  ret float %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    let pointer_selects = asm
        .lines()
        .filter(|line| {
            line.contains("OpSelect")
                && (line.contains("_ptr_StorageBuffer") || line.contains("_ptr_UniformConstant"))
        })
        .collect::<Vec<_>>();
    let value_loads = asm.lines().filter(|line| line.contains("OpLoad")).count();
    assert!(pointer_selects.is_empty(), "{asm}");
    assert_eq!(value_loads, 2, "{asm}");
    assert!(
        asm.lines().any(|line| {
            line.contains("OpSelect")
                && !line.contains("_ptr_StorageBuffer")
                && !line.contains("_ptr_UniformConstant")
        }),
        "{asm}"
    );
}

#[test]
fn native_pointer_select_null_arm_load_selects_values() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define float @main(ptr addrspace(2) %buf, i1 %cond, i64 %idx) {
entry:
  %p = getelementptr inbounds float, ptr addrspace(2) %buf, i64 %idx
  %sel = select i1 %cond, ptr addrspace(2) null, ptr addrspace(2) %p
  %v = load float, ptr addrspace(2) %sel, align 4
  ret float %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    let pointer_types = asm
        .lines()
        .filter(|line| line.contains("OpTypePointer"))
        .filter_map(|line| line.split_whitespace().next())
        .collect::<HashSet<_>>();
    let pointer_selects = asm
        .lines()
        .filter(|line| {
            line.contains("OpSelect")
                && line
                    .split_whitespace()
                    .nth(3)
                    .is_some_and(|ty| pointer_types.contains(ty))
        })
        .collect::<Vec<_>>();
    let float_type = asm
        .lines()
        .find(|line| line.contains("OpTypeFloat 32"))
        .and_then(|line| line.split_whitespace().next())
        .unwrap_or_else(|| panic!("missing float type:\n{asm}"));
    assert!(pointer_selects.is_empty(), "{asm}");
    assert!(
        asm.lines().any(|line| {
            line.contains("OpSelect") && line.split_whitespace().nth(3) == Some(float_type)
        }),
        "{asm}"
    );
    assert!(asm.lines().any(|line| line.contains("OpLoad")), "{asm}");
}

#[test]
fn native_pointer_select_null_arm_gep_uses_concrete_arm() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define float @main(ptr addrspace(1) %buf, i1 %cond, i64 %base, i64 %idx) {
entry:
  %p = getelementptr inbounds float, ptr addrspace(1) %buf, i64 %base
  %sel = select i1 %cond, ptr addrspace(1) null, ptr addrspace(1) %p
  %q = getelementptr inbounds float, ptr addrspace(1) %sel, i64 %idx
  %v = load float, ptr addrspace(1) %q, align 4
  ret float %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    let pointer_types = asm
        .lines()
        .filter(|line| line.contains("OpTypePointer"))
        .filter_map(|line| line.split_whitespace().next())
        .collect::<HashSet<_>>();
    let pointer_selects = asm
        .lines()
        .filter(|line| {
            line.contains("OpSelect")
                && line
                    .split_whitespace()
                    .nth(3)
                    .is_some_and(|ty| pointer_types.contains(ty))
        })
        .collect::<Vec<_>>();
    assert!(pointer_selects.is_empty(), "{asm}");
    assert!(
        asm.lines()
            .filter(|line| line.contains("OpInBoundsAccessChain"))
            .count()
            >= 2,
        "{asm}"
    );
}

#[test]
fn native_pointer_select_null_arm_call_arg_materializes_selected_pointer() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"
define internal fastcc float @helper(ptr addrspace(1) %p, i32 %count) {
entry:
  %x = getelementptr inbounds [3 x float], ptr addrspace(1) %p, i64 1, i64 0
  %v = load float, ptr addrspace(1) %x, align 4
  ret float %v
}

define void @main(ptr addrspace(1) %buf, ptr addrspace(1) %out, i32 %tid) {
entry:
  %cond = icmp ne i32 %tid, 0
  %base = zext i32 %tid to i64
  %p = getelementptr inbounds [3 x float], ptr addrspace(1) %buf, i64 %base
  %sel = select i1 %cond, ptr addrspace(1) %p, ptr addrspace(1) null
  %n = select i1 %cond, i32 2, i32 0
  %v = tail call fast fastcc float @helper(ptr addrspace(1) %sel, i32 %n)
  %dst = getelementptr inbounds float, ptr addrspace(1) %out, i64 %base
  store float %v, ptr addrspace(1) %dst, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 12, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"packed_float3", !"air.arg_name", !"buf"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_pointer_select_call_arg_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpSelect"), "{asm}");
    assert!(!asm.contains("OpPtrAccessChain"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_buffer_pointer_select_null_arm_rewrites_to_value_select() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @k(ptr addrspace(1) %buf, ptr addrspace(2) %flag) {
entry:
  %flag_value = load i32, ptr addrspace(2) %flag, align 4
  %cond = icmp eq i32 %flag_value, 0
  %p = getelementptr inbounds float, ptr addrspace(1) %buf, i64 0
  %sel = select i1 %cond, ptr addrspace(1) null, ptr addrspace(1) %p
  %v = load float, ptr addrspace(1) %sel, align 4
  store float %v, ptr addrspace(1) %p, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"buf"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"flag"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_buffer_pointer_select_null_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let pointer_types = asm
        .lines()
        .filter(|line| line.contains("OpTypePointer"))
        .filter_map(|line| line.split_whitespace().next())
        .collect::<HashSet<_>>();
    let pointer_selects = asm
        .lines()
        .filter(|line| {
            line.contains("OpSelect")
                && line
                    .split_whitespace()
                    .nth(3)
                    .is_some_and(|ty| pointer_types.contains(ty))
        })
        .collect::<Vec<_>>();
    let float_type = asm
        .lines()
        .find(|line| line.contains("OpTypeFloat 32"))
        .and_then(|line| line.split_whitespace().next())
        .unwrap_or_else(|| panic!("missing float type:\n{asm}"));
    assert!(pointer_selects.is_empty(), "{asm}");
    assert!(
        asm.lines().any(|line| {
            line.contains("OpSelect") && line.split_whitespace().nth(3) == Some(float_type)
        }),
        "{asm}"
    );
    assert!(
        asm.lines().any(|line| {
            line.contains("OpTypePointer StorageBuffer")
                && line.split_whitespace().last() == Some(float_type)
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

#[test]
fn native_workgroup_global_does_not_reuse_block_decorated_buffer_struct() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::_atomic" = type { i32 }

@tg = internal addrspace(3) global %"struct.metal::_atomic" zeroinitializer, align 4

define void @k(ptr addrspace(1) %histogram, i32 %i) {
entry:
  tail call void @air.atomic.local.store.i32(ptr addrspace(3) @tg, i32 0, i32 0, i32 1, i1 true)
  %idx = zext i32 %i to i64
  %slot = getelementptr inbounds %"struct.metal::_atomic", ptr addrspace(1) %histogram, i64 %idx, i32 0
  %local = tail call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) @tg, i32 1, i32 0, i32 1, i1 true)
  %global = tail call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %slot, i32 %local, i32 0, i32 2, i1 true)
  ret void
}

declare void @air.atomic.local.store.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.local.add.u.i32(ptr addrspace(3), i32, i32, i32, i1)
declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"metal::_atomic", !"air.arg_name", !"histogram"}
!4 = !{i32 0, i32 4, i32 0, !"uint", !"__s"}
!5 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_workgroup_block_alias_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    let module = load_bytes(&spv).expect("load spv");
    let block_types = module
        .annotations
        .iter()
        .filter_map(|inst| {
            if inst.class.opcode != Op::Decorate {
                return None;
            }
            match inst.operands.as_slice() {
                [Operand::IdRef(target), Operand::Decoration(Decoration::Block)] => Some(*target),
                _ => None,
            }
        })
        .collect::<HashSet<_>>();
    let workgroup_block_pointees = module
        .types_global_values
        .iter()
        .filter(|inst| {
            inst.class.opcode == Op::Variable
                && inst.operands.first() == Some(&Operand::StorageClass(StorageClass::Workgroup))
        })
        .filter_map(|inst| inst.result_id)
        .filter_map(|var| variable_pointee_type(&module, var))
        .filter(|pointee| block_types.contains(pointee))
        .collect::<Vec<_>>();
    assert!(
        workgroup_block_pointees.is_empty(),
        "Workgroup variables must not point at Block-decorated struct types: {workgroup_block_pointees:?}\n{asm}"
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
fn native_bool_vector_index_lowers_to_integer_dynamic_index() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define half @pick(<2 x half> %v, half %x, i1 %idx) {
entry:
  %old = extractelement <2 x half> %v, i1 %idx
  %next = insertelement <2 x half> %v, half %x, i1 %idx
  %new = extractelement <2 x half> %next, i32 0
  %out = fadd half %old, %new
  ret half %out
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpVectorExtractDynamic"), "{asm}");
    assert!(asm.contains("OpVectorInsertDynamic"), "{asm}");
    assert!(asm.contains("OpSelect"), "{asm}");
}

#[test]
fn native_fast_sincos_writes_cosine_pointer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main() {
entry:
  %slot = alloca float
  %sin = call fast float @air.fast_sincos.f32(float 1.000000e+00, ptr %slot)
  %cos = load float, ptr %slot
  %sum = fadd fast float %sin, %cos
  ret void
}

declare float @air.fast_sincos.f32(float, ptr)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_fast_sincos_{}",
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
    assert!(asm.contains(" Sin "), "{asm}");
    assert!(asm.contains(" Cos "), "{asm}");
    assert!(asm.contains("OpStore"), "{asm}");
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
fn native_sincos_f16_writes_cosine_pointer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main() {
entry:
  %slot = alloca half
  %sin = call fast half @air.sincos.f16(half 0xH3C00, ptr %slot)
  %cos = load half, ptr %slot
  %sum = fadd fast half %sin, %cos
  ret void
}

declare half @air.sincos.f16(half, ptr)
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_sincos_f16_{}",
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
    assert!(asm.contains("OpCapability Float16"), "{asm}");
    assert!(asm.contains(" Sin "), "{asm}");
    assert!(asm.contains(" Cos "), "{asm}");
    assert!(asm.contains("OpStore"), "{asm}");
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
fn native_half_buffer_loads_as_float_vector() {
    // A `device half*` buffer read as `<4 x float>` (16 contiguous bytes = 8 halfs = 4 floats) — an
    // MPS half-buffer reinterpret. The emitter has no logical pointer bitcast, so it must read the 8
    // contiguous halfs and little-endian–pack them into a v4uint, then bitcast to v4float.
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
define void @main(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %p = getelementptr inbounds half, ptr addrspace(1) %in, i64 4
  %v = load <4 x float>, ptr addrspace(1) %p, align 16
  store <4 x float> %v, ptr addrspace(1) %out, align 16
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_name", !"half", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_name", !"float4", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_half_to_v4float_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble transformed");
    // The 8 halfs are packed into floats: each lane shifts the high half by 16 and ORs.
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpBitwiseOr"), "{asm}");
    // The packed v4uint is bitcast to the v4float result.
    assert!(asm.contains("OpBitcast"), "{asm}");
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

#[test]
fn native_promoted_lut_global_can_be_loaded() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@.air.lut.0 = private unnamed_addr addrspace(2) constant [1 x float] [float 1.000000e+00]
define float @lut() {
entry:
  %p = getelementptr inbounds [1 x float], ptr @.air.lut.0, i64 0, i64 0
  %v = load float, ptr %p
  ret float %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("Private"), "{asm}");
    assert!(asm.contains("OpConstantComposite"), "{asm}");
    assert!(!asm.contains("OpConstantNull"), "{asm}");
    assert!(asm.contains("OpLoad"), "{asm}");
}

#[test]
fn native_promoted_lut_global_preserves_array_initializer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@.air.lut.0 = private unnamed_addr addrspace(2) constant [2 x float] [float 1.000000e+00, float 2.000000e+00]
define float @lut(i64 %idx) {
entry:
  %p = getelementptr inbounds [2 x float], ptr @.air.lut.0, i64 0, i64 %idx
  %v = load float, ptr %p
  ret float %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpConstantComposite"), "{asm}");
    assert!(!asm.contains("OpConstantNull"), "{asm}");
    assert!(asm.contains("OpLoad"), "{asm}");
}

#[test]
fn native_promoted_struct_global_preserves_zero_member_index() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@.air.lut.0 = private unnamed_addr addrspace(2) constant <{ [20 x [3 x [2 x i16]]], [12 x [3 x [2 x i16]]] }> <{ [20 x [3 x [2 x i16]]] zeroinitializer, [12 x [3 x [2 x i16]]] zeroinitializer }>
define i16 @lut() {
entry:
  %p = getelementptr inbounds <{ [20 x [3 x [2 x i16]]], [12 x [3 x [2 x i16]]] }>, ptr addrspace(2) @.air.lut.0, i64 0, i32 0, i64 19, i64 1, i64 0
  %v = load i16, ptr addrspace(2) %p
  ret i16 %v
}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(asm.contains("OpLoad"), "{asm}");
}

#[test]
fn native_inline_struct_global_gep_preserves_zero_member_index() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@.air.lut.0 = private unnamed_addr addrspace(2) constant <{ [20 x [3 x [2 x i16]]], [12 x [3 x [2 x i16]]] }> <{ [20 x [3 x [2 x i16]]] zeroinitializer, [12 x [3 x [2 x i16]]] zeroinitializer }>
define i16 @read(ptr addrspace(2) %p) {
entry:
  %v = load i16, ptr addrspace(2) %p
  ret i16 %v
}
define void @lut(ptr addrspace(1) writeonly %out) {
entry:
  %v = call i16 @read(ptr addrspace(2) getelementptr inbounds (<{ [20 x [3 x [2 x i16]]], [12 x [3 x [2 x i16]]] }>, ptr addrspace(2) @.air.lut.0, i64 0, i32 0, i64 19, i64 1, i64 0))
  store i16 %v, ptr addrspace(1) %out, align 2
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @lut, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 2, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"ushort", !"air.arg_name", !"out"}
"#;
    let spv = crate::native::emit_vulkan_spirv_inline_sroa(ll).expect("native emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(!asm.contains("OpFunctionCall"), "{asm}");
}

#[test]
fn native_member0_array_element_gep_from_flat_scalar_global_uses_view_indices() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@.air.lut.0 = private unnamed_addr addrspace(2) constant <{ [20 x [3 x [2 x i16]]], [12 x [3 x [2 x i16]]] }> <{ [20 x [3 x [2 x i16]]] zeroinitializer, [12 x [3 x [2 x i16]]] zeroinitializer }>
define i16 @lut() {
entry:
  %p = getelementptr inbounds [3 x [2 x i16]], ptr addrspace(2) @.air.lut.0, i64 19, i64 1, i64 0
  %v = load i16, ptr addrspace(2) %p
  ret i16 %v
}
"#;
    let spv = emit_vulkan_spirv(ll).expect("native emit");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(!asm.contains("OpTypeStruct"), "{asm}");
    let access = asm
        .lines()
        .find(|line| line.contains("OpInBoundsAccessChain"))
        .unwrap_or_else(|| panic!("missing access chain:\n{asm}"));
    assert_eq!(
        access.matches('%').count(),
        6,
        "expected result, type, root, and three view index ids:\n{access}\n{asm}"
    );
}

#[test]
fn native_gep_through_vector_yields_lane_pointer() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.uniform" = type { [2 x <4 x float>] }
define float @lane(ptr addrspace(2) %u) {
entry:
  %p = getelementptr inbounds %"struct.uniform", ptr addrspace(2) %u, i64 0, i32 0, i64 1, i64 2
  %v = load float, ptr addrspace(2) %p
  ret float %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
    assert!(asm.contains("OpLoad"), "{asm}");
}

#[test]
fn native_gep_strips_nuw_nusw_flags() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%"struct.metal::matrix.21" = type { [1 x <4 x float>] }
define <4 x float> @matrix(ptr addrspace(2) %m) {
entry:
  %p = getelementptr inbounds nuw %"struct.metal::matrix.21", ptr addrspace(2) %m, i64 0, i32 0, i64 0
  %v = load <4 x float>, ptr addrspace(2) %p
  ret <4 x float> %v
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpInBoundsAccessChain"), "{asm}");
}

#[test]
fn native_parser_accepts_typed_null_pointer_literal() {
    let value = parse_typed_value("ptr addrspace(2) noundef null").expect("parse typed null");
    assert_eq!(value.ty, LlType::Ptr(2));
    assert!(matches!(value.value, LlValue::Zero));
}

#[test]
fn native_global_byte_string_semicolon_is_not_comment() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
@bytes = internal unnamed_addr addrspace(2) constant [4 x i8] c"A;B\00", align 1

define void @keep() {
entry:
  ret void
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpConstantComposite"), "{asm}");
    assert!(asm.contains("OpVariable"), "{asm}");
}

#[test]
fn native_global_array_of_struct_constants_lowers() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Point = type { <2 x float>, <2 x float> }
@points = private unnamed_addr addrspace(2) constant [2 x %struct.Point] [%struct.Point { <2 x float> <float 1.0, float 2.0>, <2 x float> <float 3.0, float 4.0> }, %struct.Point { <2 x float> <float 5.0, float 6.0>, <2 x float> <float 7.0, float 8.0> }], align 16

define void @keep() {
entry:
  ret void
}
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpConstantComposite"), "{asm}");
    assert!(asm.contains("OpVariable"), "{asm}");
}

#[test]
fn native_constant_selected_private_table_retypes_access_chain_after_folding() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"
@first = internal unnamed_addr addrspace(2) constant [2 x <2 x i8>] [<2 x i8> zeroinitializer, <2 x i8> <i8 1, i8 2>], align 2
@second = internal unnamed_addr addrspace(2) constant [2 x <2 x i8>] [<2 x i8> <i8 3, i8 4>, <2 x i8> <i8 5, i8 6>], align 2

define void @k(ptr addrspace(1) %out, i32 %index) {
entry:
  %selected = select i1 true, ptr addrspace(2) @first, ptr addrspace(2) @second
  %wide = zext i32 %index to i64
  %element = getelementptr inbounds <2 x i8>, ptr addrspace(2) %selected, i64 %wide
  %value = load <2 x i8>, ptr addrspace(2) %element, align 2
  store <2 x i8> %value, ptr addrspace(1) %out, align 2
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 2, !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 2, !"air.arg_type_align_size", i32 2, !"air.arg_type_name", !"char2", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"index"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_private_table_storage_reconcile_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let module = load_bytes(&spv).expect("load spv");
    for instruction in module.all_inst_iter().filter(|instruction| {
        matches!(
            instruction.class.opcode,
            Op::AccessChain
                | Op::InBoundsAccessChain
                | Op::PtrAccessChain
                | Op::InBoundsPtrAccessChain
        )
    }) {
        let Operand::IdRef(base) = instruction.operands[0] else {
            continue;
        };
        let base_type = module
            .all_inst_iter()
            .find_map(|candidate| {
                (candidate.result_id == Some(base)).then_some(candidate.result_type)
            })
            .flatten()
            .expect("access-chain base type");
        assert_eq!(
            pointer_type_storage_class(&module, instruction.result_type.expect("result type")),
            pointer_type_storage_class(&module, base_type),
            "access-chain result must inherit its base storage class"
        );
    }
    tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    let _ = std::fs::remove_dir_all(tmp);
}

#[test]
fn native_direct_struct_buffer_drops_gep_root_zero_after_metadata_rebuild() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"
%Header = type { %Atomic, i32 }
%Atomic = type { i32 }

define void @k(ptr addrspace(1) %src, ptr addrspace(1) %out) {
entry:
  %field = getelementptr inbounds %Header, ptr addrspace(1) %src, i64 0, i32 0, i32 0
  %v = load i32, ptr addrspace(1) %field, align 4
  %dst = getelementptr inbounds i32, ptr addrspace(1) %out, i64 0
  store i32 %v, ptr addrspace(1) %dst, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 12, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Header", !"air.arg_name", !"src"}
!4 = !{!"air.struct_type_info", !5, i32 0, i32 8, i32 0, !"Header::flags", !"flags", i32 8, i32 4, i32 0, !"uint", !"tail"}
!5 = !{i32 0, i32 4, i32 0, !"uint", !"low", i32 4, i32 4, i32 0, !"uint", !"high"}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_direct_struct_buffer_root_zero_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        !asm.lines().any(|line| {
            line.contains("OpInBoundsAccessChain")
                && line.contains("_ptr_StorageBuffer_uint")
                && line.matches("%uint_0").count() >= 3
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

#[test]
fn native_packed_single_lane_struct_buffer_uses_one_storage_binding() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"
%Out = type { [1 x i32], [1 x i32], [1 x i32] }

define void @k(ptr addrspace(1) %out) {
entry:
  call fastcc void @helper(ptr addrspace(1) %out)
  ret void
}

define internal fastcc void @helper(ptr addrspace(1) %out) {
entry:
  %min = getelementptr inbounds %Out, ptr addrspace(1) %out, i64 0, i32 0, i64 0
  store i32 11, ptr addrspace(1) %min, align 4
  %max = getelementptr inbounds %Out, ptr addrspace(1) %out, i64 0, i32 1, i64 0
  store i32 22, ptr addrspace(1) %max, align 4
  %avg = getelementptr inbounds %Out, ptr addrspace(1) %out, i64 0, i32 2, i64 0
  store i32 33, ptr addrspace(1) %avg, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 12, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 12, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Out", !"air.arg_name", !"out"}
!4 = !{i32 0, i32 4, i32 0, !"packed_int1", !"min", i32 4, i32 4, i32 0, !"packed_int1", !"max", i32 8, i32 4, i32 0, !"packed_int1", !"avg"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_packed_single_lane_struct_buffer_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(asm.matches("Binding 0").count(), 1, "{asm}");
    assert!(
        !asm.contains("OpTypeRuntimeArray %uint"),
        "packed struct should not create a raw uint alias\n{asm}"
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
fn native_helper_keeps_scalar_constant_pointee_over_global_array_arg() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"
@weights = internal addrspace(2) constant [7 x half] [half 0xH3C00, half 0xH4000, half 0xH4200, half 0xH4400, half 0xH4500, half 0xH4600, half 0xH4700], align 2

define void @k(ptr addrspace(1) %out, i32 %idx) {
entry:
  call fastcc void @helper(ptr addrspace(1) %out, ptr addrspace(2) @weights, i32 %idx)
  ret void
}

define internal fastcc void @helper(ptr addrspace(1) %out, ptr addrspace(2) %weights, i32 %idx) {
entry:
  %first = load half, ptr addrspace(2) %weights, align 2
  %wide = zext i32 %idx to i64
  %p = getelementptr inbounds half, ptr addrspace(2) %weights, i64 %wide
  %second = load half, ptr addrspace(2) %p, align 2
  %sum = fadd half %first, %second
  store half %sum, ptr addrspace(1) %out, align 2
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"half", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"idx"}
"#;
    let ir = super::super::ir::LlModule::parse(ll).expect("parse");
    assert_eq!(
        ir.ptr_pointees
            .get(&("helper".to_string(), "%weights".to_string())),
        Some(&LlType::Half)
    );
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_scalar_constant_call_array_arg_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        !asm.lines().any(|line| {
            line.contains("OpLoad %half %weights") || line.contains("OpLoad %half @_")
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

#[test]
fn native_direct_call_pointer_result_carries_storage_to_followup_gep() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"
%Matrix = type { [4 x <4 x float>] }
%View = type { [2 x %Matrix] }

define void @k(ptr addrspace(2) %view, ptr addrspace(1) %out, i16 %eye) {
entry:
  %ret = tail call ptr addrspace(2) @helper(i16 %eye, ptr addrspace(2) %view)
  %lane = getelementptr inbounds %Matrix, ptr addrspace(2) %ret, i64 0, i32 0, i64 0
  %value = load <4 x float>, ptr addrspace(2) %lane, align 16
  store <4 x float> %value, ptr addrspace(1) %out, align 16
  ret void
}

declare ptr addrspace(2) @helper(i16, ptr addrspace(2))
"#;
    let asm = disassemble(&emit_vulkan_spirv(ll).expect("native emit")).expect("disassemble");
    assert!(asm.contains("OpFunctionCall"), "{asm}");
    assert!(asm.contains("OpLoad"), "{asm}");
}

#[test]
fn native_union_head_bitcast_load_does_not_create_same_binding_raw_alias() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"
%Params = type { %Union }
%Union = type { %ColorUint }
%ColorUint = type { i32, i32, i32, i32 }

define void @k(ptr addrspace(2) %params, ptr addrspace(1) %out) {
entry:
  %union = getelementptr inbounds %Params, ptr addrspace(2) %params, i64 0, i32 0
  %alias = bitcast ptr addrspace(2) %union to ptr addrspace(2)
  %v = load float, ptr addrspace(2) %alias, align 4
  store float %v, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !8}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!4 = !{!"air.struct_type_info", !5, i32 0, i32 16, i32 0, !"Union", !"color"}
!5 = !{!"air.struct_type_info", !6, i32 0, i32 4, i32 0, !"ColorUchar", !"uchar", !"air.struct_type_info", !7, i32 0, i32 16, i32 0, !"ColorUint", !"uint32"}
!6 = !{i32 0, i32 4, i32 0, !"uint", !"packed"}
!7 = !{i32 0, i32 4, i32 0, !"uint", !"x", i32 4, i32 4, i32 0, !"uint", !"y", i32 8, i32 4, i32 0, !"uint", !"z", i32 12, i32 4, i32 0, !"uint", !"w"}
!8 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_union_head_bitcast_no_alias_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(
        asm.matches(" Binding 0").count(),
        1,
        "buffer(0) must not also grow a raw same-binding StorageBuffer alias:\n{asm}"
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
fn native_metadata_byte_field_word_load_store_uses_raw_alias() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"
%Header = type { %Flags }
%Flags = type { i32, i32, i32 }

define void @k(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
entry:
  %sp = getelementptr inbounds %Header, ptr addrspace(1) %src, i64 0, i32 0, i32 2
  %v = load i32, ptr addrspace(1) %sp, align 4
  %dp = getelementptr inbounds %Header, ptr addrspace(1) %dst, i64 0, i32 0, i32 2
  store i32 %v, ptr addrspace(1) %dp, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 12, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Header", !"air.arg_name", !"src"}
!4 = !{!"air.struct_type_info", !5, i32 0, i32 12, i32 0, !"Flags", !"flags"}
!5 = !{i32 0, i32 4, i32 0, !"uint", !"a", i32 4, i32 4, i32 0, !"uint", !"b", i32 8, i32 1, i32 0, !"bool", !"c"}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 12, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Header", !"air.arg_name", !"dst"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_metadata_byte_field_word_alias_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        !asm.lines().any(|line| {
            line.contains("OpLoad %uint") && line.contains("_ptr_StorageBuffer_uchar")
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

    let layout = crate::reflect::DescriptorLayout {
        set: 5,
        ..Default::default()
    };
    let custom = crate::translate_sanitized_native_with_options(
        ll,
        Stage::Kernel,
        &tmp,
        passes::TransformOptions::default()
            .with_descriptor_layout(layout)
            .expect("custom raw-alias layout"),
    )
    .expect("custom raw-alias translation");
    let custom_asm = disassemble(&custom).expect("disassemble custom raw alias");
    assert!(custom_asm.contains("DescriptorSet 5"), "{custom_asm}");
    assert!(!custom_asm.contains("DescriptorSet 0"), "{custom_asm}");
    tools::spirv_val_bytes(&custom, &tmp).expect("custom raw-alias spirv-val");
}

#[test]
fn native_unaligned_metadata_byte_field_word_load_store_uses_raw_alias() {
    let ll = r#"
target triple = "air64-apple-macosx14.0.0"
%Header = type { %Flags }
%Flags = type { i32, i32, i8, i8, i32 }

define void @k(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
entry:
  %sp = getelementptr inbounds %Header, ptr addrspace(1) %src, i64 0, i32 0, i32 3
  %v = load i32, ptr addrspace(1) %sp, align 1
  %dp = getelementptr inbounds %Header, ptr addrspace(1) %dst, i64 0, i32 0, i32 3
  store i32 %v, ptr addrspace(1) %dp, align 1
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Header", !"air.arg_name", !"src"}
!4 = !{!"air.struct_type_info", !5, i32 0, i32 16, i32 0, !"Flags", !"flags"}
!5 = !{i32 0, i32 4, i32 0, !"uint", !"a", i32 4, i32 4, i32 0, !"uint", !"b", i32 8, i32 1, i32 0, !"bool", !"c", i32 9, i32 1, i32 0, !"bool", !"d", i32 12, i32 4, i32 0, !"uint", !"e"}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Header", !"air.arg_name", !"dst"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_unaligned_metadata_byte_field_word_alias_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    assert!(
        !asm.lines().any(|line| {
            line.contains("OpLoad %uint") && line.contains("_ptr_StorageBuffer_uchar")
        }),
        "{asm}"
    );
    assert!(asm.contains("OpShiftLeftLogical"), "{asm}");
    assert!(asm.contains("OpShiftRightLogical"), "{asm}");
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
fn native_raw_induction_phi_stays_raw_under_network_seed() {
    // A fully-RAW single-root pointer INDUCTION: a loop-carried phi whose arms are the raw buffer
    // root (via identity bitcast) and a forward GEP off the phi itself, dereferenced uniformly at
    // `<4 x float>`. The access census seeds the network (M-A2 def-site recording), but the
    // typed merge path cannot express the phi against the root's raw `{ [0 x i32] }` declaration —
    // the raw byte/word-index phi must keep the claim (raw_only_induction_phi). Pre-fix this
    // emitted `OpPhi %_ptr_StorageBuffer_v4float` with a `runtimearr_uint` incoming (spirv-val
    // INVALID, the MPSRNNLSTMRecursionCombined banked floor family).
    let ll = r#"
define void @raw_ptr_induction(ptr addrspace(1) noundef readonly captures(none) "air-buffer-no-alias" %0, ptr addrspace(1) noundef writeonly "air-buffer-no-alias" %1, <3 x i32> noundef %2) local_unnamed_addr #0 {
  %4 = bitcast ptr addrspace(1) %0 to ptr addrspace(1)
  br label %5

5:                                                ; preds = %5, %3
  %6 = phi i32 [ 0, %3 ], [ %14, %5 ]
  %7 = phi ptr addrspace(1) [ %4, %3 ], [ %13, %5 ]
  %8 = phi float [ 0.000000e+00, %3 ], [ %12, %5 ]
  %9 = load <4 x float>, ptr addrspace(1) %7, align 16
  %10 = extractelement <4 x float> %9, i64 0
  %11 = extractelement <4 x float> %9, i64 1
  %12 = fadd fast float %8, %10
  %13 = getelementptr inbounds <4 x float>, ptr addrspace(1) %7, i64 1
  %14 = add nuw i32 %6, 1
  %15 = icmp ult i32 %14, 8
  br i1 %15, label %5, label %16

16:                                               ; preds = %5
  %17 = fadd fast float %12, %11
  %18 = extractelement <3 x i32> %2, i64 0
  %19 = zext i32 %18 to i64
  %20 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %19
  store float %17, ptr addrspace(1) %20, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @raw_ptr_induction, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 128, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"src"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 4, !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"dst"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_raw_induction_phi_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    // The PRIMARY (no-retry) path — the `--primary-val-check` surface the family regresses on.
    let out = crate::translate_native_no_retry(ll, Stage::Kernel).expect("no-retry translate");
    let asm = disassemble(&out).expect("disassemble");
    // The pointer phi must be modeled as a RAW index phi, never a pointer-typed OpPhi against the
    // raw block declaration (which mistypes one arm).
    assert!(
        !asm.lines().any(|l| l.contains("OpPhi %_ptr_")),
        "expected no pointer-typed OpPhi (raw index phi instead):\n{asm}"
    );
    // Both buffers must be REALLY modeled: without the raw claim the seeded network defers to the
    // typed merge path, which cannot express the raw root, and the whole src network degrades to
    // unmodeled zero placeholders (src loses its binding and the loads become OpConstantNull).
    assert!(asm.contains("Binding 0"), "src buffer unbound:\n{asm}");
    assert!(asm.contains("Binding 1"), "dst buffer unbound:\n{asm}");
    assert!(
        asm.lines().filter(|l| l.contains("OpLoad")).count() >= 4,
        "expected the <4 x float> load modeled as real raw word loads:\n{asm}"
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
fn native_dead_null_storage_buffer_pointer_phi_prunes_before_final_normalization() {
    let ll = r#"
define void @dead_null_phi(ptr addrspace(1) noundef writeonly "air-buffer-no-alias" %out, ptr addrspace(1) noundef readonly "air-buffer-no-alias" %src, <3 x i32> %gid) local_unnamed_addr #0 {
entry:
  br i1 false, label %null_arm, label %real_arm

null_arm:
  br label %merge

real_arm:
  %p = getelementptr inbounds i8, ptr addrspace(1) %src, i64 4
  br label %merge

merge:
  %q = phi ptr addrspace(1) [ null, %null_arm ], [ %p, %real_arm ]
  %v = load i8, ptr addrspace(1) %q, align 1
  store i8 %v, ptr addrspace(1) %out, align 1
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @dead_null_phi, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"src"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_dead_null_phi_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble");
    assert!(!asm.contains("VariablePointersStorageBuffer"), "{asm}");
    assert!(
        !asm.lines()
            .any(|line| line.contains("OpPhi %_ptr_StorageBuffer")),
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
fn native_forward_unmodeled_device_gep_phi_uses_placeholder_not_pointer_phi() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"
%struct.Box = type { [3 x float], [3 x float] }
%struct.Table = type { ptr addrspace(1), ptr addrspace(1) }

define float @k(ptr addrspace(2) %table, i1 %cond, i64 %idx) {
entry:
  %f0 = getelementptr inbounds %struct.Table, ptr addrspace(2) %table, i64 0, i32 0
  %b0 = load ptr addrspace(1), ptr addrspace(2) %f0, align 8
  %f1 = getelementptr inbounds %struct.Table, ptr addrspace(2) %table, i64 0, i32 1
  %b1 = load ptr addrspace(1), ptr addrspace(2) %f1, align 8
  br i1 %cond, label %lhs, label %rhs

merge:
  %p = phi ptr addrspace(1) [ %lp, %lhs ], [ %rp, %rhs ]
  %x = getelementptr inbounds %struct.Box, ptr addrspace(1) %p, i64 0, i32 0, i64 0
  %v = load float, ptr addrspace(1) %x, align 4
  ret float %v

lhs:
  %lp = getelementptr inbounds %struct.Box, ptr addrspace(1) %b0, i64 %idx
  br label %merge

rhs:
  %rp = getelementptr inbounds %struct.Box, ptr addrspace(1) %b1, i64 %idx
  br label %merge
}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_native_forward_unmodeled_gep_phi_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let out = crate::translate_sanitized_native(ll, Stage::Kernel, &tmp).expect("translate");
    let asm = disassemble(&out).expect("disassemble");
    assert!(
        !asm.lines().any(|line| line.contains("OpPhi %_ptr_")),
        "unmodeled forward-GEP arms must not emit a pointer phi:\n{asm}"
    );
    assert!(
        !asm.lines().any(|line| line.contains("OpSelect %_ptr_")),
        "unmodeled forward-GEP arms must not emit a pointer select:\n{asm}"
    );
    assert!(
        asm.contains("OpTypePointer Private"),
        "expected private placeholder backing for the unmodeled pointer:\n{asm}"
    );
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&out, &tmp).expect("spirv-val");
    }
}

/// An AIR vector whose store size is smaller than its datalayout allocation size (`<3 x i8>` under
/// `v24:32:32`, `<3 x i16>` under `v48:64:64`, `<3 x float>` under `v96:128:128`) must push the
/// member that follows it to the next allocation boundary, exactly as LLVM's `StructLayout`
/// consumes `alignTo(storeSize, abiAlign)` per member. Emitting the store-size boundary instead
/// leaves every later member one lane early, so Vulkan reads the wrong bytes even though the
/// module is structurally valid SPIR-V.
const VECTOR_ALLOCATION_STRIDE_LL: &str = r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-n8:16:32"
target triple = "air64-apple-macosx10.15.0"

%struct.Image = type <{ i8, i8, i8, i8, <3 x i8>, i8, i8, [2 x i8] }>

define void @k(ptr addrspace(2) noalias readonly dereferenceable(12) %cfg, ptr addrspace(1) %out) {
entry:
  %vp = getelementptr inbounds %struct.Image, ptr addrspace(2) %cfg, i64 0, i32 4
  %v = load <3 x i8>, ptr addrspace(2) %vp, align 4
  %tp = getelementptr inbounds %struct.Image, ptr addrspace(2) %cfg, i64 0, i32 5
  %t = load i8, ptr addrspace(2) %tp, align 4
  %up = getelementptr inbounds %struct.Image, ptr addrspace(2) %cfg, i64 0, i32 6
  %u = load i8, ptr addrspace(2) %up, align 1
  %lane = extractelement <3 x i8> %v, i32 2
  %s0 = add i8 %lane, %t
  %s1 = add i8 %s0, %u
  %z = zext i8 %s1 to i32
  store i32 %z, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 12, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Image", !"air.arg_name", !"cfg"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint*", !"air.arg_name", !"out"}
"#;

fn vector_allocation_stride_with_metadata() -> String {
    let with_reference = VECTOR_ALLOCATION_STRIDE_LL.replacen(
        "!\"air.arg_type_size\", i32 12",
        "!\"air.struct_type_info\", !5, !\"air.arg_type_size\", i32 12",
        1,
    );
    format!(
        "{with_reference}\n!5 = !{{i32 0, i32 1, i32 0, !\"uchar\", !\"a\", i32 1, i32 1, i32 0, !\"uchar\", !\"b\", i32 2, i32 1, i32 0, !\"uchar\", !\"c\", i32 3, i32 1, i32 0, !\"uchar\", !\"d\", i32 4, i32 4, i32 0, !\"uchar3\", !\"v\", i32 8, i32 1, i32 0, !\"uchar\", !\"t\", i32 9, i32 1, i32 0, !\"uchar\", !\"u\"}}\n"
    )
}

const NONCANONICAL_VECTOR_ALIGNMENT_LL: &str = r#"
target datalayout = "e-p:64:64-v24:64:64-n8:16:32"
target triple = "air64-apple-macosx10.15.0"

%struct.Params = type <{ <3 x i8>, i8 }>

define void @k(ptr addrspace(2) noalias readonly dereferenceable(16) %params, ptr addrspace(1) %out) {
entry:
  %field = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 1
  %value = load i8, ptr addrspace(2) %field, align 8
  %wide = zext i8 %value to i32
  store i32 %wide, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint*", !"air.arg_name", !"out"}
"#;

/// Member offsets of the one emitted struct that holds a three-lane 8-bit vector, found
/// structurally through the type graph (`OpTypeInt 8` -> `OpTypeVector .. 3` -> `OpTypeStruct`).
fn three_lane_byte_vector_struct_offsets(asm: &str) -> Vec<(u32, u32)> {
    let result_id = |line: &str| -> Option<String> {
        let name = line.split_whitespace().next()?;
        name.starts_with('%').then(|| name.to_string())
    };
    let defines = |line: &str, opcode: &str, operands: &[&str]| -> Option<String> {
        let (lhs, rhs) = line.split_once('=')?;
        let mut words = rhs.split_whitespace();
        (words.next()? == opcode).then_some(())?;
        let rest = words.collect::<Vec<_>>();
        (rest == operands).then(|| lhs.trim().to_string())
    };
    let byte_ty = asm
        .lines()
        .map(str::trim)
        .find_map(|line| defines(line, "OpTypeInt", &["8", "0"]))
        .unwrap_or_else(|| panic!("no 8-bit integer type was emitted:\n{asm}"));
    let vector_ty = asm
        .lines()
        .map(str::trim)
        .find_map(|line| defines(line, "OpTypeVector", &[byte_ty.as_str(), "3"]))
        .unwrap_or_else(|| panic!("no <3 x i8> type was emitted:\n{asm}"));
    let struct_ty = asm
        .lines()
        .map(str::trim)
        .filter(|line| line.contains("OpTypeStruct"))
        .find(|line| line.split_whitespace().any(|word| word == vector_ty))
        .and_then(result_id)
        .unwrap_or_else(|| panic!("no struct with a <3 x i8> member was emitted:\n{asm}"));
    let mut offsets = asm
        .lines()
        .map(str::trim)
        .filter_map(|line| {
            let rest = line.strip_prefix("OpMemberDecorate ")?;
            let mut parts = rest.split_whitespace();
            (parts.next()? == struct_ty).then_some(())?;
            let member: u32 = parts.next()?.parse().ok()?;
            (parts.next()? == "Offset").then_some(())?;
            Some((member, parts.next()?.parse().ok()?))
        })
        .collect::<Vec<(u32, u32)>>();
    offsets.sort_unstable();
    offsets
}

#[test]
fn native_three_lane_vector_member_advances_by_its_allocation_size() {
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_vector_allocation_stride_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let spv = crate::translate_sanitized_native(VECTOR_ALLOCATION_STRIDE_LL, Stage::Kernel, &tmp)
        .expect("translate");
    let asm = disassemble(&spv).expect("disassemble");
    // `<3 x i8>` sits at 4 and allocates four bytes, so member 5 starts at 8 — not at the
    // store-size boundary 7 — and the trailing padding array lands at 10.
    assert_eq!(
        three_lane_byte_vector_struct_offsets(&asm),
        vec![
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 8),
            (6, 9),
            (7, 10)
        ],
        "{asm}"
    );
    if std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok()
    {
        tools::spirv_val_bytes(&spv, &tmp).expect("spirv-val");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn native_uchar3_struct_preserves_authoritative_air_offsets() {
    let ll = vector_allocation_stride_with_metadata();
    let kern = meta::parse_air_kernel_meta(&ll);
    let emitted = crate::native::emit_vulkan_spirv_with_sidecar(
        &ll,
        kern.as_ref(),
        Some("k"),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
    )
    .expect("emit with AIR layout sidecar");

    assert_eq!(
        emitted.sidecar.air_struct_layout_mappings,
        vec![crate::emit_sidecar::AirStructLayoutMapping {
            param_index: 0,
            struct_ty: emitted.sidecar.air_struct_layout_mappings[0].struct_ty,
            status: crate::emit_sidecar::AirStructLayoutMappingStatus::MappedNatural,
        }]
    );
    assert!(
        emitted
            .sidecar
            .air_struct_offsets
            .values()
            .any(|offsets| offsets == &[0, 1, 2, 3, 4, 8, 9]),
        "{:?}",
        emitted.sidecar.air_struct_offsets
    );
}

#[test]
fn native_struct_layout_shape_mismatch_is_typed_sidecar_evidence() {
    let ll = vector_allocation_stride_with_metadata().replace("!\"uchar3\"", "!\"float3\"");
    let kern = meta::parse_air_kernel_meta(&ll);
    let emitted = crate::native::emit_vulkan_spirv_with_sidecar(
        &ll,
        kern.as_ref(),
        Some("k"),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
    )
    .expect("emit mismatched AIR layout");

    assert_eq!(
        emitted.sidecar.air_struct_layout_mappings[0].status,
        crate::emit_sidecar::AirStructLayoutMappingStatus::EmittedShapeMismatch
    );
    assert!(emitted.sidecar.air_struct_offsets.is_empty());
}

#[test]
fn file_translation_threads_source_datalayout_into_emitted_offsets() {
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_source_datalayout_layout_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let source = tmp.join("vector-layout.ll");
    std::fs::write(&source, NONCANONICAL_VECTOR_ALIGNMENT_LL).expect("write source fixture");
    let spv = crate::translate(
        source.to_str().expect("UTF-8 fixture path"),
        Stage::Kernel,
        &tmp,
    )
    .expect("translate source file");
    let asm = disassemble(&spv).expect("disassemble");

    assert_eq!(
        three_lane_byte_vector_struct_offsets(&asm),
        vec![(0, 0), (1, 8)],
        "{asm}"
    );
    let _ = std::fs::remove_file(source);
    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn construct_tree_retry_three_lane_vector_member_advances_by_its_allocation_size() {
    // The layout contract belongs to every candidate the retry cascade can adopt, not only the
    // primary emit. `construct_tree` is the validation-gated tier the reproducing compositor
    // fragment shader actually lands on.
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_vector_allocation_stride_retry_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let kern = meta::parse_air_kernel_meta(VECTOR_ALLOCATION_STRIDE_LL);
    let retry = crate::retry::RetryCtx::new(
        VECTOR_ALLOCATION_STRIDE_LL,
        Stage::Kernel,
        None,
        None,
        kern.as_ref(),
        None,
        Some("k"),
        &tmp,
        passes::TransformOptions::default(),
        true,
        crate::layout::AirDataLayout::from_ir(VECTOR_ALLOCATION_STRIDE_LL)
            .expect("parse datalayout"),
    );
    let spv = retry
        .construct_tree_retry()
        .expect("construct_tree candidate");
    let asm = disassemble(&spv).expect("disassemble");
    assert_eq!(
        three_lane_byte_vector_struct_offsets(&asm),
        vec![
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 8),
            (6, 9),
            (7, 10)
        ],
        "{asm}"
    );
    let _ = std::fs::remove_dir_all(&tmp);
}
