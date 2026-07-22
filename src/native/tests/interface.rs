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
fn native_fragment_depth_return_maps_to_frag_depth() {
    let ll = r#"
source_filename = "synth_depth"
target datalayout = "e-p:64:64:64"
target triple = "air64-apple-macosx14.0.0"

define <{ float }> @synth_depth(float %0) local_unnamed_addr #0 {
  %2 = insertvalue <{ float }> undef, float %0, 0
  ret <{ float }> %2
}

attributes #0 = { nounwind }

!air.fragment = !{!0}
!0 = !{ptr @synth_depth, !1, !3}
!1 = !{!2}
!2 = !{!"air.depth", !"air.depth_qualifier", !"air.any", !"air.arg_type_name", !"float", !"air.arg_name", !"depth"}
!3 = !{!4}
!4 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float"}
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
