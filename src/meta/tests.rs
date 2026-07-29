use super::*;
use AirScalar::*;

const FRAG_LL: &str = r#"
!air.fragment = !{!15}
!15 = !{ptr @F, !16, !18}
!16 = !{!17}
!17 = !{!"air.render_target", i32 0, i32 0}
!18 = !{!19, !20, !21, !22, !23, !24}
!19 = !{i32 0, !"air.position", !"air.center"}
!20 = !{i32 1, !"air.fragment_input", !"generated", !"air.arg_type_name", !"float2", !"air.arg_name", !"texCoord"}
!21 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"texture2d<float, sample>"}
!22 = !{i32 3, !"air.buffer", !"air.buffer_size", i32 32, !"air.location_index", i32 5, i32 1}
!23 = !{i32 4, !"air.point_coord", !"air.arg_type_name", !"float2", !"air.arg_name", !"pointCoord"}
!24 = !{i32 5, !"air.front_facing", !"air.arg_type_name", !"bool", !"air.arg_name", !"front"}
"#;

#[test]
fn fragment_roles() {
    let m = parse_air_fragment_meta(FRAG_LL).unwrap();
    assert_eq!(m.n_render_targets, 1);
    assert_eq!(m.render_target_indices, vec![0]);
    assert_eq!(m.role_of(0), Some(&FragRole::Position));
    assert_eq!(m.role_of(1), Some(&FragRole::Varying(0)));
    assert_eq!(m.role_of(2), Some(&FragRole::Texture(0)));
    assert_eq!(m.role_of(3), Some(&FragRole::Buffer(5)));
    assert_eq!(m.role_of(4), Some(&FragRole::PointCoord));
    assert_eq!(m.role_of(5), Some(&FragRole::FrontFacing));
    assert_eq!(m.texture_type_name(2), Some("texture2d<float, sample>"));
    assert_eq!(m.varying_type(0), Some("float2"));
    assert_eq!(m.varying_name(0), Some("texCoord"));
    assert_eq!(m.varying_user_semantic(0), Some("generated"));
    assert!(!m.varying_is_flat(0));
}

#[test]
fn fragment_flat_varying_metadata() {
    let ll = FRAG_LL.replace(
        r#"!20 = !{i32 1, !"air.fragment_input", !"generated", !"air.arg_type_name", !"float2", !"air.arg_name", !"texCoord"}"#,
        r#"!20 = !{i32 1, !"air.fragment_input", !"generated", !"air.flat", !"air.arg_type_name", !"float2", !"air.arg_name", !"texCoord"}"#,
    );
    let m = parse_air_fragment_meta(&ll).unwrap();
    assert!(m.varying_is_flat(0));
}

#[test]
fn fragment_function_constant_location_uses_static_init_default() {
    let ll = r#"
@_ZL32__metal_implicit_attr_int_expr_1.78 = internal addrspace(2) global i32 0, align 4
@_ZN2RB6Shader8Constant13_shader_stateE.MTL_FC_INIT_0_Dv4_j = internal unnamed_addr addrspace(2) externally_initialized constant <4 x i32> undef, section "air.fc_initializer", align 16

define internal void @_GLOBAL__sub_I_shader_filter_distance.metal() section "air.static_init" {
entry:
  %1 = load <4 x i32>, ptr addrspace(2) @_ZN2RB6Shader8Constant13_shader_stateE.MTL_FC_INIT_0_Dv4_j
  %2 = extractelement <4 x i32> %1, i64 0
  %12 = and i32 %2, 131072
  %13 = icmp eq i32 %12, 0
  %14 = select i1 %13, i32 4, i32 6
  store i32 %14, ptr addrspace(2) @_ZL32__metal_implicit_attr_int_expr_1.78
  ret void
}

!air.fragment = !{!0}
!0 = !{ptr @F, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0}
!3 = !{!4}
!4 = !{i32 0, !"air.function_constant", !5, !"air.texture", !"air.location_index", ptr addrspace(2) @_ZL32__metal_implicit_attr_int_expr_1.78, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<half, read>", !"air.arg_name", !"dest_tex"}
!5 = !{ptr addrspace(2) @__metal_implicit_fc_pred_1.80, !"bool", !"uses_dest"}
"#;
    let m = parse_air_fragment_meta(ll).unwrap();
    assert_eq!(m.role_of(0), Some(&FragRole::Texture(4)));
}

#[test]
fn fragment_function_constant_render_target_disabled_by_default() {
    let ll = r#"
!air.fragment = !{!0}
!0 = !{ptr @F, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!3 = !{i32 7, !"air.function_constant", !5, !"air.render_target", i32 1, i32 0, !"air.arg_type_name", !"half2"}
!4 = !{}
!5 = !{ptr addrspace(2) @__metal_implicit_fc_pred_1, !"bool", !"uses_coverage"}
"#;
    let m = parse_air_fragment_meta(ll).unwrap();
    assert_eq!(m.n_render_targets, 1);
    assert_eq!(m.render_target_members, vec![(0, 0)]);
    assert_eq!(m.render_target_type_name(1), None);
}

#[test]
fn fragment_function_constant_render_target_static_true_is_recognized() {
    let ll = r#"
@__metal_implicit_fc_pred_1 = internal unnamed_addr addrspace(2) global i8 1, align 1
!air.fragment = !{!0}
!0 = !{ptr @F, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!3 = !{i32 7, !"air.function_constant", !5, !"air.render_target", i32 1, i32 0, !"air.arg_type_name", !"half2"}
!4 = !{}
!5 = !{ptr addrspace(2) @__metal_implicit_fc_pred_1, !"bool", !"uses_coverage"}
"#;
    let m = parse_air_fragment_meta(ll).unwrap();
    assert_eq!(m.n_render_targets, 2);
    assert_eq!(m.render_target_members, vec![(0, 0), (1, 1)]);
    assert_eq!(m.render_target_type_name(1), Some("half2"));
}

#[test]
fn fragment_color_input_uses_render_target_location() {
    let ll = r#"
!air.fragment = !{!0}
!0 = !{ptr @F, !1, !4}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!4 = !{!5}
!5 = !{i32 2, !"air.render_target", i32 2, !"air.arg_type_name", !"float", !"air.arg_name", !"d0"}
"#;
    let m = parse_air_fragment_meta(ll).unwrap();
    assert_eq!(m.role_of(2), Some(&FragRole::ColorInput(2)));
    assert_eq!(m.color_input_type_name(2), Some("float"));
}

#[test]
fn fragment_render_target_location_uses_static_init_default() {
    let ll = r#"
@_ZL32__metal_implicit_attr_int_expr_5 = internal addrspace(2) global i32 0, align 4
@_ZN2RB6Shader8Constant13_shader_stateE.MTL_FC_INIT_0_Dv4_j = internal unnamed_addr addrspace(2) externally_initialized constant <4 x i32> undef, section "air.fc_initializer", align 16

define internal void @_GLOBAL__sub_I_shader.metal() section "air.static_init" {
entry:
  %1 = load <4 x i32>, ptr addrspace(2) @_ZN2RB6Shader8Constant13_shader_stateE.MTL_FC_INIT_0_Dv4_j
  %2 = extractelement <4 x i32> %1, i64 0
  %3 = and i32 %2, 131072
  %4 = icmp eq i32 %3, 0
  %5 = select i1 %4, i32 4, i32 6
  store i32 %5, ptr addrspace(2) @_ZL32__metal_implicit_attr_int_expr_5
  ret void
}

!air.fragment = !{!0}
!0 = !{ptr @F, !1, !4}
!1 = !{!2, !3}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4"}
!3 = !{!"air.render_target", ptr addrspace(2) @_ZL32__metal_implicit_attr_int_expr_5, !"air.arg_type_name", !"half4"}
!4 = !{}
"#;
    let m = parse_air_fragment_meta(ll).unwrap();
    assert_eq!(m.render_target_members, vec![(0, 0), (1, 4)]);
    assert_eq!(m.render_target_indices, vec![0, 4]);
}

#[test]
fn fragment_render_target_indices() {
    let ll = FRAG_LL.replace(
        "!17 = !{!\"air.render_target\", i32 0, i32 0}",
        "!17 = !{!\"air.render_target\", i32 3, i32 0}",
    );
    let m = parse_air_fragment_meta(&ll).unwrap();
    assert_eq!(m.n_render_targets, 1);
    assert_eq!(m.render_target_members, vec![(0, 3)]);
    assert_eq!(m.render_target_indices, vec![3]);
}

#[test]
fn fragment_render_target_type_names() {
    let ll = FRAG_LL.replace(
        "!17 = !{!\"air.render_target\", i32 0, i32 0}",
        "!17 = !{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"int4\"}",
    );
    let m = parse_air_fragment_meta(&ll).unwrap();
    assert_eq!(m.render_target_type_name(0), Some("int4"));
}

#[test]
fn fragment_stencil_output_is_not_a_render_target() {
    let ll = FRAG_LL.replace(
        "!17 = !{!\"air.render_target\", i32 0, i32 0}",
        "!17 = !{!\"air.stencil\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"stencil\"}",
    );
    let m = parse_air_fragment_meta(&ll).unwrap();
    assert_eq!(m.n_render_targets, 0);
    assert!(m.render_target_members.is_empty());
    assert!(m.render_target_indices.is_empty());
    assert_eq!(m.stencil_members, vec![0]);
    assert!(m.is_stencil_member(0));
    assert!(!m.is_stencil_member(1));
}

#[test]
fn fragment_depth_output_is_not_a_render_target() {
    let ll = FRAG_LL.replace(
        "!17 = !{!\"air.render_target\", i32 0, i32 0}",
        "!17 = !{!\"air.depth\", !\"air.depth_qualifier\", !\"air.any\", !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"depth\"}",
    );
    let m = parse_air_fragment_meta(&ll).unwrap();
    assert_eq!(m.n_render_targets, 0);
    assert!(m.render_target_members.is_empty());
    assert_eq!(m.depth_members, vec![0]);
    assert!(m.is_depth_member(0));
    assert!(!m.is_depth_member(1));
}

const VERT_LL: &str = r#"
!air.vertex = !{!15}
!15 = !{ptr @V, !16, !19}
!16 = !{!17, !18}
!19 = !{!20, !21, !22}
!20 = !{i32 0, !"air.vertex_input", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float3", !"air.arg_name", !"position"}
!21 = !{i32 1, !"air.vertex_input", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!22 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 64, !"air.location_index", i32 4, i32 1}
"#;

#[test]
fn vertex_roles() {
    let m = parse_air_vertex_meta(VERT_LL).unwrap();
    assert_eq!(m.role_of(0), Some(&VertRole::VertexInput(0)));
    assert_eq!(m.role_of(1), Some(&VertRole::VertexInput(1)));
    assert_eq!(m.role_of(2), Some(&VertRole::Buffer(4)));
    assert_eq!(m.vertex_input_type(0), Some("float3"));
    assert_eq!(m.vertex_input_name(0), Some("position"));
    assert_eq!(m.vertex_input_type(1), Some("float2"));
    assert_eq!(m.vertex_input_name(1), Some("uv"));
}

// `[[vertex_id]]` / `[[instance_id]]` builtins -> VertexId/InstanceId roles (mirrors render_tri.air).
const VERT_BUILTIN_LL: &str = r#"
!air.vertex = !{!14}
!14 = !{ptr @vmain, !15, !17}
!15 = !{!16, !20, !21}
!16 = !{!"air.position", !"air.arg_type_name", !"float4"}
!17 = !{!18, !19}
!18 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!19 = !{i32 1, !"air.instance_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"iid"}
!20 = !{!"air.viewport_array_index", !"air.arg_type_name", !"uint", !"air.arg_name", !"viewport"}
!21 = !{!"air.clip_distance", !"air.arg_type_name", !"float", !"air.arg_name", !"clip"}
"#;

#[test]
fn vertex_builtin_roles() {
    let m = parse_air_vertex_meta(VERT_BUILTIN_LL).unwrap();
    assert_eq!(m.role_of(0), Some(&VertRole::VertexId));
    assert_eq!(m.role_of(1), Some(&VertRole::InstanceId));
    assert_eq!(m.output_role_of(0), Some(&VertOutRole::Position));
    assert_eq!(m.output_role_of(1), Some(&VertOutRole::ViewportArrayIndex));
    assert_eq!(m.output_role_of(2), Some(&VertOutRole::ClipDistance));
}

#[test]
fn vertex_render_target_array_index_role() {
    let ll = r#"
!air.vertex = !{!0}
!0 = !{ptr @vmain, !1, !5}
!1 = !{!2, !3, !4}
!2 = !{!"air.position", !"air.arg_type_name", !"float4"}
!3 = !{!"air.render_target_array_index", !"air.arg_type_name", !"uint", !"air.arg_name", !"layer"}
!4 = !{!"air.vertex_output", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"varying"}
!5 = !{}
"#;
    let m = parse_air_vertex_meta(ll).unwrap();
    assert_eq!(m.output_role_of(0), Some(&VertOutRole::Position));
    assert_eq!(
        m.output_role_of(1),
        Some(&VertOutRole::RenderTargetArrayIndex)
    );
    assert_eq!(m.output_role_of(2), Some(&VertOutRole::Varying(0)));
}

#[test]
fn vertex_function_constant_output_is_disabled_by_default() {
    let ll = r#"
!air.vertex = !{!0}
!0 = !{ptr @vmain, !1, !6}
!1 = !{!2, !3, !5}
!2 = !{!"air.position", !"air.arg_type_name", !"float4"}
!3 = !{!"air.function_constant", !4, !"air.vertex_output", !"air.arg_type_name", !"float4", !"air.arg_name", !"optional"}
!4 = !{ptr addrspace(2) @__metal_implicit_fc_pred_0, !"bool", !"enabled"}
!5 = !{!"air.vertex_output", !"air.arg_type_name", !"float", !"air.arg_name", !"varying"}
!6 = !{}
"#;
    let m = parse_air_vertex_meta(ll).unwrap();
    assert_eq!(m.output_role_of(0), Some(&VertOutRole::Position));
    assert_eq!(
        m.output_role_of(1),
        Some(&VertOutRole::FunctionConstantDisabled)
    );
    assert_eq!(m.output_role_of(2), Some(&VertOutRole::Varying(0)));
}

// `!air.kernel`: buffers + compute builtins (mirrors multi_add.air + harvested app kernels).
const KERN_LL: &str = r#"
!air.kernel = !{!14}
!14 = !{ptr @k, !15, !16}
!15 = !{}
!16 = !{!17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30}
!17 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"float", !"air.arg_name", !"a"}
!18 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"float", !"air.arg_name", !"b"}
!19 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.arg_type_name", !"float", !"air.arg_name", !"c"}
!20 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
!21 = !{i32 4, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lsize"}
!22 = !{i32 5, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lid"}
!23 = !{i32 6, !"air.threadgroups_per_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gsize"}
!24 = !{i32 7, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}
!25 = !{i32 8, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"thread_id"}
!26 = !{i32 9, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"lane"}
!27 = !{i32 10, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_group"}
!28 = !{i32 11, !"air.threads_per_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"grid_size"}
!29 = !{i32 12, !"air.threads_per_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"simd_width"}
!30 = !{i32 13, !"air.simdgroups_per_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"num_simd_groups"}
"#;

#[test]
fn kernel_roles() {
    let m = parse_air_kernel_meta(KERN_LL).unwrap();
    assert_eq!(m.role_of(0), Some(&KernRole::Buffer(0)));
    assert_eq!(m.role_of(1), Some(&KernRole::Buffer(1)));
    assert_eq!(m.role_of(2), Some(&KernRole::Buffer(2)));
    assert_eq!(m.buffer_address_space(0), Some(1));
    assert_eq!(m.buffer_address_space(1), Some(1));
    assert_eq!(m.buffer_address_space(2), Some(1));
    assert_eq!(m.buffer_type_name(0), Some("float"));
    assert_eq!(m.buffer_type_name(1), Some("float"));
    assert_eq!(m.buffer_type_name(2), Some("float"));
    assert_eq!(m.role_of(3), Some(&KernRole::ThreadPositionInGrid));
    assert_eq!(m.role_of(4), Some(&KernRole::ThreadsPerThreadgroup));
    assert_eq!(m.role_of(5), Some(&KernRole::ThreadPositionInThreadgroup));
    assert_eq!(m.role_of(6), Some(&KernRole::ThreadgroupsPerGrid));
    assert_eq!(m.role_of(7), Some(&KernRole::ThreadgroupPositionInGrid));
    assert_eq!(m.role_of(8), Some(&KernRole::ThreadIndexInThreadgroup));
    assert_eq!(m.role_of(9), Some(&KernRole::ThreadIndexInSimdgroup));
    assert_eq!(m.role_of(10), Some(&KernRole::SimdgroupIndexInThreadgroup));
    assert_eq!(m.role_of(11), Some(&KernRole::ThreadsPerGrid));
    assert_eq!(m.role_of(12), Some(&KernRole::ThreadsPerSimdgroup));
    assert_eq!(m.role_of(13), Some(&KernRole::SimdgroupsPerThreadgroup));
}

#[test]
fn kernel_texture_sampler_roles() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 5, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<uint, read>", !"air.arg_name", !"inTex"}
!4 = !{i32 1, !"air.sampler", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
"#;
    let m = parse_air_kernel_meta(ll).unwrap();
    assert_eq!(m.role_of(0), Some(&KernRole::Texture(5)));
    assert_eq!(m.texture_type_name(0), Some("texture2d<uint, read>"));
    assert_eq!(m.role_of(1), Some(&KernRole::Sampler(2)));
}

#[test]
fn kernel_stage_in_is_preserved_as_role() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.stage_in", !"air.location_index", i32 7, i32 1, !"air.arg_type_name", !"uint3", !"air.arg_name", !"pointIndices"}
"#;
    let m = parse_air_kernel_meta(ll).unwrap();
    assert_eq!(m.role_of(0), Some(&KernRole::StageInput(7)));
}

#[test]
fn kernel_indirect_buffer_is_a_buffer_role() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 1, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!4 = !{i32 0, i32 4, i32 0, !"uint", !"count", !"air.indirect_argument", !5, i32 4, i32 4, i32 0, !"uint", !"stride", !"air.indirect_argument", !6}
!5 = !{}
!6 = !{}
"#;
    let m = parse_air_kernel_meta(ll).unwrap();
    assert_eq!(m.role_of(1), Some(&KernRole::Buffer(3)));
    assert_eq!(m.buffer_address_space(1), Some(2));
    assert_eq!(m.buffer_type_size(1), Some(8));
    assert!(matches!(m.layout_of(1), Some(AirType::Struct(fields)) if fields.len() == 2));
}

#[test]
fn kernel_acceleration_structure_introspection_uses_shadow_role() {
    let ll = r#"
define void @k(ptr addrspace(1) %as, ptr addrspace(1) %out) {
  %count = call i32 @air.get_instance_count_instance_acceleration_structure(ptr addrspace(1) %as)
  store i32 %count, ptr addrspace(1) %out
  ret void
}
declare i32 @air.get_instance_count_instance_acceleration_structure(ptr addrspace(1))

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.instance_acceleration_structure", !"air.location_index", i32 8, i32 1, !"air.read", !"air.arg_type_name", !"acceleration_structure<instancing>", !"air.arg_name", !"as"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;
    let m = parse_air_kernel_meta(ll).unwrap();
    assert_eq!(
        m.role_of(0),
        Some(&KernRole::AccelerationStructureShadow(8))
    );
    assert_eq!(m.role_of(1), Some(&KernRole::Buffer(0)));
}

#[test]
fn kernel_imageblock_layout_is_not_a_buffer_layout() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 16, !"air.struct_type_info", !4, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{i32 0, i32 16, i32 0, !"float4", !"v"}
"#;
    let m = parse_air_kernel_meta(ll).unwrap();
    assert_eq!(m.role_of(0), Some(&KernRole::Other));
    assert!(m.layout_of(0).is_none());
    assert!(matches!(
        m.imageblock_layout_of(0),
        Some(AirType::Struct(fields))
            if matches!(
                fields.as_slice(),
                [AirMember {
                    offset: 0,
                    ty: AirType::Vec {
                        scalar: Float,
                        lanes: 4
                    }
                }]
            )
    ));
}

#[test]
fn kernel_function_constant_imageblock_keeps_its_cell_layout() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.function_constant", !4, !"air.imageblock", !"explicit", !"air.imageblock_data_size", i32 16, !"air.struct_type_info", !5, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"imageblock<ImageBlockData, layout_explicit>", !"air.arg_name", !"imgBlk"}
!4 = !{ptr addrspace(2) @__metal_implicit_fc_pred_0, !"bool", !"enabled"}
!5 = !{i32 0, i32 16, i32 0, !"float4", !"v"}
"#;
    let m = parse_air_kernel_meta(ll).unwrap();
    assert_eq!(m.role_of(0), Some(&KernRole::Other));
    assert!(matches!(
        m.imageblock_layout_of(0),
        Some(AirType::Struct(fields))
            if matches!(
                fields.as_slice(),
                [AirMember {
                    offset: 0,
                    ty: AirType::Vec {
                        scalar: Float,
                        lanes: 4
                    }
                }]
            )
    ));
}

#[test]
fn kernel_threadgroup_buffer_records_address_space() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.struct_type_info", !4, !"air.arg_type_name", !"Temp", !"air.arg_name", !"temp"}
!4 = !{i32 0, i32 4, i32 0, !"packed_float1", !"value"}
"#;
    let m = parse_air_kernel_meta(ll).unwrap();
    assert_eq!(m.role_of(3), Some(&KernRole::Buffer(0)));
    assert_eq!(m.buffer_address_space(3), Some(3));
    assert!(matches!(m.layout_of(3), Some(AirType::Struct(fields)) if fields.len() == 1));
}

#[test]
fn kernel_threadgroup_buffer_infers_address_space_from_signature() {
    let ll = r#"
define void @k(ptr addrspace(3) %temp, i32 %i) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"temp"}
!4 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"i"}
"#;
    let m = parse_air_kernel_meta(ll).unwrap();
    assert_eq!(m.role_of(0), Some(&KernRole::Buffer(0)));
    assert_eq!(m.buffer_address_space(0), Some(3));
}

#[test]
fn kernel_meta_variants_share_one_parse_and_preserve_fc_buffer_projection() {
    let ll = r#"
define void @k(ptr addrspace(1) %optional) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.function_constant", !4, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"optional"}
!4 = !{i32 0}
"#;

    let (default, promoted, entry) = parse_air_kernel_meta_variants(ll);
    let default = default.expect("default kernel metadata");
    let promoted = promoted.expect("promoted kernel metadata");

    assert_eq!(entry.as_deref(), Some("k"));
    assert_eq!(default.role_of(0), Some(&KernRole::Other));
    assert_eq!(promoted.role_of(0), Some(&KernRole::Buffer(7)));
    assert_eq!(promoted.buffer_address_space(0), Some(1));
    assert_eq!(promoted.buffer_type_size(0), Some(4));
    assert_eq!(promoted.buffer_type_name(0), Some("float"));
}

#[test]
fn primitive_air_type_names_include_64_bit_integers() {
    assert_eq!(
        primitive_air_type_from_name("ulong"),
        Some(AirType::Scalar(ULong))
    );
    assert_eq!(
        primitive_air_type_from_name("long2"),
        Some(AirType::Vec {
            scalar: SLong,
            lanes: 2
        })
    );
}

#[test]
fn nested_bitfield_struct_type_info_uses_declared_storage() {
    let ll = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !4, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"Header", !"air.arg_name", !"header"}
!4 = !{!"air.struct_type_info", !5, i32 0, i32 4, i32 0, !"Header::(anonymous)", !"flags", i32 4, i32 4, i32 0, !"float", !"tail"}
!5 = !{i32 0, i32 4, i32 0, !"uint", !"mode", i32 0, i32 1, i32 0, !"bool", !"enabled", i32 1, i32 4, i32 0, !"uint", !"depth"}
"#;
    let m = parse_air_kernel_meta(ll).unwrap();
    assert_eq!(
        m.layout_of(0),
        Some(&AirType::Struct(vec![
            AirMember {
                offset: 0,
                ty: AirType::Scalar(UInt)
            },
            AirMember {
                offset: 4,
                ty: AirType::Scalar(Float)
            }
        ]))
    );
}

#[test]
fn kernel_entry_name() {
    assert_eq!(entry_name(KERN_LL, "kernel").as_deref(), Some("k"));
}

#[test]
fn quoted_kernel_entry_name() {
    let ll = r#"
define void @"re::df::pack"(ptr addrspace(1) %out) {
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @"re::df::pack", !1, !2}
!1 = !{}
!2 = !{!3}
!3 = distinct !{i32 0, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;
    let (meta, _, entry) = parse_air_kernel_meta_variants(ll);
    assert_eq!(entry.as_deref(), Some("re::df::pack"));
    let meta = meta.expect("kernel metadata");
    assert_eq!(meta.role_of(0), Some(&KernRole::Buffer(7)));
    assert_eq!(meta.buffer_address_space(0), Some(1));
}

// `air.struct_type_info` reconstruction, including a matrix (float3x4 -> { [3 x float4] }) and a
// NESTED struct (mirrors ColorCorrectionParametric { float3x4, ColorCurve{float3 x2} }).
const RECON_LL: &str = r#"
!air.fragment = !{!0}
!0 = !{ptr @F, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0}
!3 = !{!4, !5, !9}
!4 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 16, !"air.struct_type_info", !6, !"air.arg_type_name", !"Pair"}
!5 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 96, !"air.struct_type_info", !7, !"air.arg_type_name", !"PCC"}
!6 = !{i32 0, i32 8, i32 0, !"packed_float2", !"_a", i32 8, i32 8, i32 0, !"packed_float2", !"_b", i32 16, i32 4, i32 0, !"packed_float1", !"_c"}
!7 = !{i32 0, i32 48, i32 0, !"float3x4", !"_m", !"air.struct_type_info", !8, i32 48, i32 48, i32 0, !"Curve", !"_c"}
!8 = !{i32 0, i32 16, i32 0, !"float3", !"_x", i32 16, i32 16, i32 0, !"float3", !"_y"}
!9 = !{i32 2, !"air.buffer", !"air.buffer_size", i32 32, !"air.struct_type_info", !10, !"air.arg_type_name", !"HistogramLayout"}
!10 = !{i32 0, i32 4, i32 4, !"uint", !"indices", i32 16, i32 4, i32 4, !"uint", !"counts"}
"#;

#[test]
fn struct_type_info_reconstruction() {
    let m = parse_air_fragment_meta(RECON_LL).unwrap();
    // Pair { packed_float2, packed_float2, packed_float1 }
    assert_eq!(
        m.layout_of(0),
        Some(&AirType::Struct(vec![
            AirMember {
                offset: 0,
                ty: AirType::PackedVec {
                    scalar: Float,
                    lanes: 2
                }
            },
            AirMember {
                offset: 8,
                ty: AirType::PackedVec {
                    scalar: Float,
                    lanes: 2
                }
            },
            AirMember {
                offset: 16,
                ty: AirType::PackedVec {
                    scalar: Float,
                    lanes: 1
                }
            }
        ]))
    );
    // PCC { float3x4 matrix, Curve { float3, float3 } }
    assert_eq!(
        m.layout_of(1),
        Some(&AirType::Struct(vec![
            AirMember {
                offset: 0,
                ty: AirType::Matrix {
                    scalar: Float,
                    cols: 3,
                    rows: 4
                }
            },
            AirMember {
                offset: 48,
                ty: AirType::Struct(vec![
                    AirMember {
                        offset: 0,
                        ty: AirType::Vec {
                            scalar: Float,
                            lanes: 3
                        }
                    },
                    AirMember {
                        offset: 16,
                        ty: AirType::Vec {
                            scalar: Float,
                            lanes: 3
                        }
                    }
                ])
            },
        ]))
    );
    // HistogramLayout { uint indices[4], uint counts[4] }
    assert_eq!(
        m.layout_of(2),
        Some(&AirType::Struct(vec![
            AirMember {
                offset: 0,
                ty: AirType::Array {
                    elem: Box::new(AirType::Scalar(UInt)),
                    len: 4
                }
            },
            AirMember {
                offset: 16,
                ty: AirType::Array {
                    elem: Box::new(AirType::Scalar(UInt)),
                    len: 4
                }
            },
        ]))
    );
}

#[test]
fn parse_function_constants_reads_fc_initializer_globals() {
    // R1.7: discover [[function_constant(N)]] from the Apple FC ABI marker `.MTL_FC_INIT_<N>_...`
    // (section "air.fc_initializer"), keyed only on that documented marker — never a shader name.
    let ll = "\
target triple = \"air64-apple-macosx\"
@_ZL32__metal_implicit_attr_int_expr_1.78 = internal addrspace(2) global i32 0, align 4
@_ZN2RB6Shader8Constant13_shader_stateE.MTL_FC_INIT_0_Dv4_j = internal unnamed_addr addrspace(2) externally_initialized constant <4 x i32> undef, section \"air.fc_initializer\", align 16
@some.other.MTL_FC_INIT_3_i = internal addrspace(2) externally_initialized constant i32 undef, section \"air.fc_initializer\", align 4
define void @k() {
entry:
  %1 = load <4 x i32>, ptr addrspace(2) @_ZN2RB6Shader8Constant13_shader_stateE.MTL_FC_INIT_0_Dv4_j
  ret void
}
";
    let fcs = parse_function_constants(ll);
    assert_eq!(fcs.len(), 2, "two distinct FC indices");
    // Sorted by index.
    assert_eq!(fcs[0].index, 0);
    assert_eq!(fcs[0].name, "_ZN2RB6Shader8Constant13_shader_stateE");
    assert_eq!(fcs[0].type_name, "<4 x i32>");
    assert_eq!(fcs[1].index, 3);
    assert_eq!(fcs[1].type_name, "i32");
    // The `load` use of the same global must NOT create a duplicate; the plain implicit global is
    // ignored (no MTL_FC_INIT marker).
    assert!(parse_function_constants("@x = global i32 0").is_empty());
}
