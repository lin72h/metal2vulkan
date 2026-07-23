//! Unit coverage for the reflection facade: parse a known AIR fixture into meta, build the
//! [`ShaderReflection`], and assert the exported binding numbers match the ABI convention the
//! interface pass decorates (buffers = index, textures = 32+n, samplers = 64+n, colors = 96+n,
//! all in descriptor set 0).

use super::*;
use crate::meta::{
    parse_air_fragment_meta, parse_air_vertex_meta, EmbeddedTexture, KernMeta, KernRole,
};

const FRAG_LL: &str = r#"
!air.fragment = !{!15}
!15 = !{ptr @F, !16, !18}
!16 = !{!17}
!17 = !{!"air.render_target", i32 0, i32 0}
!18 = !{!19, !20, !21, !22, !23}
!19 = !{i32 0, !"air.position", !"air.center"}
!20 = !{i32 1, !"air.fragment_input", !"generated", !"air.arg_type_name", !"float2", !"air.arg_name", !"texCoord"}
!21 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"texture2d<float, sample>"}
!22 = !{i32 3, !"air.buffer", !"air.buffer_size", i32 32, !"air.location_index", i32 5, i32 1}
!23 = !{i32 4, !"air.sampler", !"air.location_index", i32 2, i32 1}
"#;

#[test]
fn fragment_reflection_matches_abi_convention() {
    let meta = parse_air_fragment_meta(FRAG_LL).unwrap();
    let r = ShaderReflection::from_fragment(&meta, Some("myFragment"));

    assert_eq!(r.reflection_version, REFLECTION_VERSION);
    assert_eq!(r.stage, ShaderStage::Fragment);
    assert_eq!(r.entry_point.as_deref(), Some("myFragment"));

    // Texture(0) -> set 0, binding 32.
    let tex = r.binding_at(ResourceKind::Texture, 0).expect("texture 0");
    assert_eq!(
        tex.descriptor,
        Some(DescriptorLocation {
            set: 0,
            binding: TEXTURE_BINDING_BASE
        })
    );
    assert_eq!(tex.param_index, Some(2));
    assert_eq!(tex.type_name.as_deref(), Some("texture2d<float, sample>"));
    // A `sample` texture classifies Sampled (M2).
    assert_eq!(tex.access, Some(ResourceAccess::Sampled));

    // Buffer at Metal index 5 -> binding 5 (no base offset).
    let buf = r.binding_at(ResourceKind::Buffer, 5).expect("buffer 5");
    assert_eq!(
        buf.descriptor,
        Some(DescriptorLocation { set: 0, binding: 5 })
    );
    // R1.6: the fragment buffer's declared byte size is exported from `air.buffer_size` (32 in the
    // fixture) instead of being dropped, so a consumer never re-parses the AIR arg metadata.
    assert_eq!(buf.declared_size, Some(32));

    // Sampler(2) -> set 0, binding 66.
    let smp = r.binding_at(ResourceKind::Sampler, 2).expect("sampler 2");
    assert_eq!(
        smp.descriptor,
        Some(DescriptorLocation {
            set: 0,
            binding: SAMPLER_BINDING_BASE + 2
        })
    );

    // The interpolated varying is exported with its type/name/semantic.
    let v = r
        .varyings
        .iter()
        .find(|v| v.location == 0)
        .expect("varying 0");
    assert_eq!(v.type_name.as_deref(), Some("float2"));
    assert_eq!(v.name.as_deref(), Some("texCoord"));
    assert_eq!(v.user_semantic.as_deref(), Some("generated"));

    // The single render target at member 0 -> location 0.
    assert_eq!(r.render_targets.len(), 1);
    assert_eq!(r.render_targets[0].member_index, 0);
    assert_eq!(r.render_targets[0].location, 0);

    // Position/varying inputs are NOT descriptor resources.
    assert!(r
        .bindings
        .iter()
        .all(|b| b.kind != ResourceKind::ColorInput));
}

#[test]
fn reflected_static_samplers_stay_in_sampler_band_and_carry_state() {
    let ll = r#"
target triple = "spirv-unknown-vulkan1.3"

@__air_sampler_state.118 = internal addrspace(2) constant i64 -9188470239253755319, align 8
@__air_sampler_state.119 = internal addrspace(2) constant i64 -9188470239253757806, align 8

define <4 x float> @frag(<4 x float> %position, <2 x float> %coord, ptr addrspace(1) %tex, ptr addrspace(2) %runtime_sampler, <4 x float> %color0) {
entry:
  %sample0 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state.118, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value0 = extractvalue { <4 x float>, i8 } %sample0, 0
  %sample1 = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %tex, ptr addrspace(2) @__air_sampler_state.119, <2 x float> %coord, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %value1 = extractvalue { <4 x float>, i8 } %sample1, 0
  %value = fadd <4 x float> %value0, %value1
  ret <4 x float> %value
}

declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)

!air.fragment = !{!0}
!air.sampler_states = !{!9, !8}
!air.compile_options = !{!10}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5, !6, !7, !11}
!4 = !{i32 0, !"air.position", !"air.center", !"air.arg_type_name", !"float4"}
!5 = !{i32 1, !"air.fragment_input", !"generated(coord)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2"}
!6 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>"}
!7 = !{i32 3, !"air.sampler", !"air.location_index", i32 0, i32 1}
!8 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.118}
!9 = !{!"air.sampler_state", ptr addrspace(2) @__air_sampler_state.119}
!10 = !{!"air.compile.framebuffer_fetch_enable"}
!11 = !{i32 4, !"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
"#;
    let tmp = std::env::temp_dir().join(format!(
        "metal2vulkan_reflect_static_sampler_{}",
        std::process::id()
    ));
    let _ = std::fs::create_dir_all(&tmp);
    let (spv, reflection) = crate::translate_sanitized_native_reflected(
        ll,
        crate::passes::Stage::Fragment,
        &tmp,
        crate::passes::TransformOptions::default(),
    )
    .expect("translate");
    let asm = crate::disassemble(&spv).expect("disassemble");
    assert!(asm.contains("Binding 64"), "{asm}");
    assert!(asm.contains("Binding 65"), "{asm}");
    assert!(asm.contains("Binding 66"), "{asm}");
    assert!(asm.contains("Binding 96"), "{asm}");
    assert!(!asm.contains("Binding 97"), "{asm}");
    assert!(!asm.contains("Binding 98"), "{asm}");

    let static_samplers = reflection
        .bindings
        .iter()
        .filter(|binding| binding.kind == ResourceKind::StaticSampler)
        .collect::<Vec<_>>();
    assert_eq!(static_samplers.len(), 2);
    assert_eq!(
        static_samplers
            .iter()
            .filter_map(|binding| binding.descriptor.map(|descriptor| descriptor.binding))
            .collect::<Vec<_>>(),
        vec![65, 66]
    );
    let linear = static_samplers[0]
        .static_sampler
        .expect("linear static state");
    assert_eq!(linear.min_filter, SamplerFilter::Linear);
    assert_eq!(linear.mag_filter, SamplerFilter::Linear);
    assert_eq!(linear.address_mode_s, SamplerAddressMode::ClampToEdge);
    assert_eq!(linear.address_mode_t, SamplerAddressMode::ClampToEdge);
    assert_eq!(linear.address_mode_r, SamplerAddressMode::ClampToEdge);
    assert_eq!(linear.compare_function, SamplerCompareFunction::Never);
    assert_eq!(linear.lod_min_clamp, 0.0);
    assert_eq!(linear.lod_max_clamp, 65504.0);

    let repeat = static_samplers[1]
        .static_sampler
        .expect("repeat static state");
    assert_eq!(repeat.min_filter, SamplerFilter::Nearest);
    assert_eq!(repeat.mag_filter, SamplerFilter::Nearest);
    assert_eq!(repeat.address_mode_s, SamplerAddressMode::Repeat);
    assert_eq!(repeat.address_mode_t, SamplerAddressMode::Repeat);
    assert_eq!(repeat.address_mode_r, SamplerAddressMode::Repeat);
}

#[test]
fn fragment_and_vertex_buffer_address_space_exported_when_known() {
    // R1.6: when the AIR carries a buffer arg's address space, reflection reports it (device=1,
    // constant=2) for the fragment and vertex stages the way it already did for kernels — no guess
    // when absent. Drive the reflect mapping directly from meta (the parser fixture covers extraction).
    use crate::meta::{FragRole, VertRole};
    let mut frag = crate::meta::FragMeta {
        roles: vec![(0, FragRole::Buffer(0)), (1, FragRole::Buffer(1))],
        ..Default::default()
    };
    frag.buffer_address_spaces.insert(0, 2); // constant
    frag.buffer_type_sizes.insert(0, 64);
    // param 1 has no recorded address space -> reflection reports None (not a guessed default).
    let rf = ShaderReflection::from_fragment(&frag, Some("f"));
    let b0 = rf.binding_at(ResourceKind::Buffer, 0).expect("buffer 0");
    assert_eq!(b0.address_space, Some(2));
    assert_eq!(b0.declared_size, Some(64));
    let b1 = rf.binding_at(ResourceKind::Buffer, 1).expect("buffer 1");
    assert_eq!(b1.address_space, None);
    assert_eq!(b1.declared_size, None);

    let mut vert = crate::meta::VertMeta {
        roles: vec![(0, VertRole::Buffer(3))],
        ..Default::default()
    };
    vert.buffer_address_spaces.insert(0, 1); // device
    let rv = ShaderReflection::from_vertex(&vert, Some("v"));
    let vb = rv
        .binding_at(ResourceKind::Buffer, 3)
        .expect("vertex buffer 3");
    assert_eq!(vb.address_space, Some(1));
}

const VERT_LL: &str = r#"
!air.vertex = !{!5}
!5 = !{ptr @V, !6, !8}
!6 = !{!7}
!7 = !{i32 0, !"air.position", !"air.center"}
!8 = !{!9, !10}
!9 = !{i32 0, !"air.vertex_input", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos", !"air.location_index", i32 0, i32 1}
!10 = !{i32 1, !"air.buffer", !"air.location_index", i32 2, i32 1}
"#;

#[test]
fn vertex_reflection_exports_attributes_and_buffer() {
    let meta = parse_air_vertex_meta(VERT_LL).unwrap();
    let r = ShaderReflection::from_vertex(&meta, Some("myVertex"));
    assert_eq!(r.stage, ShaderStage::Vertex);

    // Buffer at Metal index 2 -> binding 2.
    let buf = r.binding_at(ResourceKind::Buffer, 2).expect("buffer 2");
    assert_eq!(
        buf.descriptor,
        Some(DescriptorLocation { set: 0, binding: 2 })
    );

    // Vertex attribute at location 0 exported with its type/name.
    let attr = r
        .vertex_attributes
        .iter()
        .find(|a| a.location == 0)
        .expect("attribute 0");
    assert_eq!(attr.type_name.as_deref(), Some("float4"));
    assert_eq!(attr.name.as_deref(), Some("pos"));
}

#[test]
fn kernel_reflection_threadgroup_buffer_has_no_descriptor() {
    // Build a KernMeta directly: a device buffer at index 0, a threadgroup buffer at index 1
    // (address space 3), a texture at index 0, and one embedded arg-buffer texture.
    let mut meta = KernMeta {
        roles: vec![
            (0, KernRole::Buffer(0)),
            (1, KernRole::Buffer(1)),
            (2, KernRole::Texture(0)),
        ],
        ..Default::default()
    };
    meta.buffer_address_spaces.insert(0, 1); // device
    meta.buffer_address_spaces
        .insert(1, ADDRESS_SPACE_THREADGROUP);
    meta.buffer_type_sizes.insert(0, 64);
    meta.texture_type_names
        .insert(2, "texture2d<float, sample>".to_string());
    meta.embedded_textures.push(EmbeddedTexture {
        buffer_index: 0,
        field_offset: 8,
        dim: spirv::Dim::Dim2D,
        comp: crate::passes::ImageComp::Float,
        synthetic_texture_index: 3,
    });

    let r = ShaderReflection::from_kernel(&meta, Some("myKernel"), [64, 1, 1]);
    assert_eq!(r.stage, ShaderStage::Kernel);
    assert_eq!(r.local_size, Some([64, 1, 1]));

    // Device buffer 0 -> descriptor binding 0, declared size present.
    let dev = r
        .binding_at(ResourceKind::Buffer, 0)
        .expect("device buffer");
    assert_eq!(
        dev.descriptor,
        Some(DescriptorLocation { set: 0, binding: 0 })
    );
    assert_eq!(dev.declared_size, Some(64));
    assert_eq!(dev.address_space, Some(1));

    // Threadgroup buffer 1 -> NO descriptor.
    let tg = r
        .binding_at(ResourceKind::ThreadgroupBuffer, 1)
        .expect("threadgroup buffer");
    assert_eq!(tg.descriptor, None);
    assert_eq!(tg.address_space, Some(ADDRESS_SPACE_THREADGROUP));

    // Texture 0 -> binding 32.
    let tex = r.binding_at(ResourceKind::Texture, 0).expect("texture");
    assert_eq!(
        tex.descriptor,
        Some(DescriptorLocation {
            set: 0,
            binding: TEXTURE_BINDING_BASE
        })
    );

    // Embedded arg-buffer texture at synthetic index 3 -> binding 35.
    let emb = r
        .binding_at(ResourceKind::EmbeddedArgBufferTexture, 3)
        .expect("embedded texture");
    assert_eq!(
        emb.descriptor,
        Some(DescriptorLocation {
            set: 0,
            binding: TEXTURE_BINDING_BASE + 3
        })
    );
    assert_eq!(emb.param_index, None);
    // R1.5: the arg-buffer source (owning buffer + field offset) is exported, not just the index.
    assert_eq!(
        emb.embedded_source,
        Some(EmbeddedArgBuffer {
            buffer_index: 0,
            field_offset: 8
        })
    );
    // The embedded texture's decoded shape rides `texture_shape` too (always sampled 2D here).
    let shape = emb.texture_shape.expect("embedded texture_shape");
    assert_eq!(shape.dimension, crate::meta::TextureDimension::D2);
    assert_eq!(shape.component, crate::meta::TextureComponent::Float);
    assert!(!shape.writable);
}

#[test]
fn imageblock_layouts_exported_for_kernel() {
    // R1.8: a kernel [[imageblock]] tile carries no descriptor, so it is not a ResourceBinding;
    // its reconstructed struct layout is parsed into KernMeta but was dropped. Export it sorted by
    // param index so a consumer can size the threadgroup tile without re-parsing.
    use crate::meta::{AirScalar, AirType};
    let mut meta = KernMeta {
        roles: vec![(0, KernRole::Buffer(0))],
        ..Default::default()
    };
    let tile = AirType::Vec {
        scalar: AirScalar::Float,
        lanes: 4,
    };
    meta.imageblock_layouts.insert(2, tile.clone());
    let r = ShaderReflection::from_kernel(&meta, Some("k"), [8, 8, 1]);
    assert_eq!(r.imageblock_layouts.len(), 1);
    assert_eq!(r.imageblock_layouts[0].param_index, 2);
    assert_eq!(r.imageblock_layouts[0].type_layout, tile);
    // A fragment shader carries none.
    let frag = parse_air_fragment_meta(FRAG_LL).unwrap();
    assert!(ShaderReflection::from_fragment(&frag, None)
        .imageblock_layouts
        .is_empty());
}

#[test]
fn vertex_builtins_report_index_and_position_usage() {
    // R1.4: vertex-stage builtin usage (vertex_id / instance_id / position) is exported instead of
    // dropped, so a consumer never walks the emitted module for OpDecorate BuiltIn.
    let mut meta = crate::meta::VertMeta {
        roles: vec![
            (0, VertRole::VertexId),
            (1, VertRole::InstanceId),
            (2, VertRole::Buffer(0)),
        ],
        ..Default::default()
    };
    meta.output_roles = vec![crate::meta::VertOutRole::Position];
    let r = ShaderReflection::from_vertex(&meta, Some("v"));
    let b = r.vertex_builtins.expect("vertex_builtins present");
    assert!(b.uses_vertex_index);
    assert!(b.uses_instance_index);
    assert!(b.writes_position);

    // A vertex shader that reads no index builtins and writes no position reports all-false.
    let plain = crate::meta::VertMeta {
        roles: vec![(0, VertRole::Buffer(0))],
        ..Default::default()
    };
    let rp = ShaderReflection::from_vertex(&plain, Some("v"));
    let bp = rp.vertex_builtins.expect("vertex_builtins present");
    assert_eq!(bp, VertexBuiltins::default());

    // Non-vertex stages carry no vertex builtins.
    let kern = KernMeta::default();
    assert_eq!(
        ShaderReflection::from_kernel(&kern, Some("k"), [1, 1, 1]).vertex_builtins,
        None
    );
}

#[test]
fn texture_access_classification_matches_declared_qualifier() {
    // write/read_write textures classify as storage images; sample/read as sampled; array_ref as a
    // sampled descriptor array (M2, mirroring the interface pass's texture_arg_storage).
    let cases = [
        (
            0u32,
            "texture2d<float, sample>",
            ResourceKind::Texture,
            ResourceAccess::Sampled,
        ),
        (
            1,
            "texture2d<uint, read>",
            ResourceKind::Texture,
            ResourceAccess::Sampled,
        ),
        (
            2,
            "texture2d<uint, write>",
            ResourceKind::StorageImage,
            ResourceAccess::Storage,
        ),
        (
            3,
            "texture2d<float, read_write>",
            ResourceKind::StorageImage,
            ResourceAccess::Storage,
        ),
        (
            4,
            "array_ref<texture2d<float, sample>>",
            ResourceKind::TextureArray,
            ResourceAccess::Sampled,
        ),
    ];
    let mut meta = KernMeta {
        roles: cases
            .iter()
            .enumerate()
            .map(|(i, (n, ..))| (i as u32, KernRole::Texture(*n)))
            .collect(),
        ..Default::default()
    };
    for (i, (_, name, ..)) in cases.iter().enumerate() {
        meta.texture_type_names.insert(i as u32, name.to_string());
    }
    let r = ShaderReflection::from_kernel(&meta, Some("k"), [1, 1, 1]);
    for (n, _, kind, access) in cases {
        let b = r
            .binding_at(kind, n)
            .unwrap_or_else(|| panic!("binding {n}"));
        assert_eq!(b.kind, kind, "kind for texture {n}");
        assert_eq!(b.access, Some(access), "access for texture {n}");
    }
}

#[test]
fn texture_shape_exports_dimensionality_arrayed_multisampled_component() {
    // R1.1: the decoded TextureShape is exported per texture binding, so a consumer never re-parses
    // the type name or walks the emitted OpTypeImage. One decoder (meta::texture_shape_from_name)
    // feeds both the emit path and this reflection.
    use crate::meta::{TextureComponent as TC, TextureDimension as TD};
    let cases = [
        (
            0u32,
            "texture1d<float, sample>",
            TD::D1,
            false,
            false,
            TC::Float,
        ),
        (
            1,
            "texture1d_array<float, sample>",
            TD::D1,
            true,
            false,
            TC::Float,
        ),
        (2, "texture2d<uint, read>", TD::D2, false, false, TC::Uint),
        (
            3,
            "texture2d_array<int, sample>",
            TD::D2,
            true,
            false,
            TC::Sint,
        ),
        (
            4,
            "texture2d_ms<float, read>",
            TD::D2,
            false,
            true,
            TC::Float,
        ),
        (
            5,
            "texture3d<half, sample>",
            TD::D3,
            false,
            false,
            TC::Float,
        ),
        (
            6,
            "texturecube<float, sample>",
            TD::Cube,
            false,
            false,
            TC::Float,
        ),
        (
            7,
            "texturecube_array<float, sample>",
            TD::Cube,
            true,
            false,
            TC::Float,
        ),
        (
            8,
            "texture_buffer<uint, read>",
            TD::Buffer,
            false,
            false,
            TC::Uint,
        ),
    ];
    let mut meta = KernMeta {
        roles: cases
            .iter()
            .enumerate()
            .map(|(i, (n, ..))| (i as u32, KernRole::Texture(*n)))
            .collect(),
        ..Default::default()
    };
    for (i, (_, name, ..)) in cases.iter().enumerate() {
        meta.texture_type_names.insert(i as u32, name.to_string());
    }
    let r = ShaderReflection::from_kernel(&meta, Some("k"), [1, 1, 1]);
    for (n, name, dim, arrayed, ms, comp) in cases {
        let b = r
            .bindings
            .iter()
            .find(|b| b.metal_index == n && b.texture_shape.is_some())
            .unwrap_or_else(|| panic!("texture binding {n} ({name})"));
        let shape = b.texture_shape.expect("texture_shape present");
        assert_eq!(shape.dimension, dim, "dim for {name}");
        assert_eq!(shape.arrayed, arrayed, "arrayed for {name}");
        assert_eq!(shape.multisampled, ms, "multisampled for {name}");
        assert_eq!(shape.component, comp, "component for {name}");
        // These are all sample/read textures — no storage format.
        assert_eq!(shape.storage_format, None, "storage_format for {name}");
    }
}

#[test]
fn storage_image_texel_format_exported() {
    // R1.2: a write/read_write texture exports the OpTypeImage texel format the emitter chose,
    // decoded once (meta::texture_shape_from_name) and shared with the emit path so they cannot
    // diverge. The consumer stops walking SPIR-V for the format operand.
    use crate::meta::TextureFormat as TF;
    let cases = [
        (0u32, "texture2d<float, write>", TF::Rgba32f),
        (1, "texture2d<half, write>", TF::Rgba16f),
        (2, "texture2d<uint, write>", TF::Rgba8ui),
        (3, "texture2d<ushort, read_write>", TF::Rgba16ui),
        (4, "texture2d<int, write>", TF::Rgba8i),
    ];
    let mut meta = KernMeta {
        roles: cases
            .iter()
            .enumerate()
            .map(|(i, (n, ..))| (i as u32, KernRole::Texture(*n)))
            .collect(),
        ..Default::default()
    };
    for (i, (_, name, _)) in cases.iter().enumerate() {
        meta.texture_type_names.insert(i as u32, name.to_string());
    }
    let r = ShaderReflection::from_kernel(&meta, Some("k"), [1, 1, 1]);
    for (n, name, fmt) in cases {
        let b = r
            .binding_at(ResourceKind::StorageImage, n)
            .unwrap_or_else(|| panic!("storage image {n} ({name})"));
        let shape = b.texture_shape.expect("texture_shape present");
        assert_eq!(shape.storage_format, Some(fmt), "storage_format for {name}");
        assert!(shape.writable, "writable for {name}");
    }
}

#[test]
fn constant_buffer_is_read_only_device_is_unknown() {
    // Constant address space -> ReadOnly; device address space -> None (needs SPIR-V dataflow). (M2)
    let mut meta = KernMeta {
        roles: vec![(0, KernRole::Buffer(0)), (1, KernRole::Buffer(1))],
        ..Default::default()
    };
    meta.buffer_address_spaces.insert(0, ADDRESS_SPACE_CONSTANT);
    meta.buffer_address_spaces.insert(1, 1); // device
    let r = ShaderReflection::from_kernel(&meta, Some("k"), [1, 1, 1]);
    assert_eq!(
        r.binding_at(ResourceKind::Buffer, 0).unwrap().access,
        Some(ResourceAccess::ReadOnly)
    );
    assert_eq!(r.binding_at(ResourceKind::Buffer, 1).unwrap().access, None);
}

#[cfg(feature = "serde")]
#[test]
fn reflection_serde_round_trips() {
    // A consumer persists reflection alongside cached SPIR-V; a serialize→deserialize round trip must
    // preserve every field (including the embedded AirType layout) so a cache hit skips re-reflection.
    let meta = parse_air_fragment_meta(FRAG_LL).unwrap();
    let r = ShaderReflection::from_fragment(&meta, Some("myFragment"));
    let json = serde_json::to_string(&r).expect("serialize reflection");
    let back: ShaderReflection = serde_json::from_str(&json).expect("deserialize reflection");
    assert_eq!(r, back);
    assert_eq!(back.reflection_version, REFLECTION_VERSION);
}

#[cfg(feature = "serde")]
#[test]
fn reflection_serde_covers_every_v3_field() {
    // R6: the persisted-cache contract. Build a reflection with EVERY field populated to a
    // non-default value — including the translate-path-only fields a stage builder never fills
    // (function_constants, datalayout, imageblock_layouts) and the storage-format / embedded-source
    // texture facts — then prove serialize→deserialize is loss-free. `assert_eq!` over the whole
    // struct is the coverage: any field serde drops or reshapes fails it.
    use crate::meta::{texture_shape_from_name, AirScalar, AirType, FunctionConstant};
    let storage_tex = texture_shape_from_name("texture2d<float, write>");
    assert!(storage_tex.writable && storage_tex.storage_format.is_some());
    let r = ShaderReflection {
        reflection_version: REFLECTION_VERSION,
        stage: ShaderStage::Kernel,
        entry_point: Some("k".to_string()),
        bindings: vec![
            ResourceBinding {
                kind: ResourceKind::Buffer,
                metal_index: 0,
                descriptor: ResourceBinding::descriptor_at(BUFFER_BINDING_BASE, 0),
                param_index: Some(0),
                address_space: Some(2),
                declared_size: Some(64),
                type_layout: Some(AirType::Vec {
                    scalar: AirScalar::Float,
                    lanes: 4,
                }),
                type_name: Some("float4".to_string()),
                texture_shape: None,
                embedded_source: None,
                access: Some(ResourceAccess::ReadOnly),
                static_sampler: None,
            },
            ResourceBinding {
                kind: ResourceKind::Texture,
                metal_index: 0,
                descriptor: ResourceBinding::descriptor_at(TEXTURE_BINDING_BASE, 0),
                param_index: None,
                address_space: None,
                declared_size: None,
                type_layout: None,
                type_name: Some("texture2d<float, write>".to_string()),
                texture_shape: Some(storage_tex),
                embedded_source: Some(EmbeddedArgBuffer {
                    buffer_index: 0,
                    field_offset: 8,
                }),
                access: Some(ResourceAccess::Storage),
                static_sampler: None,
            },
            ResourceBinding {
                kind: ResourceKind::StaticSampler,
                metal_index: 1,
                descriptor: ResourceBinding::descriptor_at(SAMPLER_BINDING_BASE, 1),
                param_index: None,
                address_space: None,
                declared_size: None,
                type_layout: None,
                type_name: None,
                texture_shape: None,
                embedded_source: None,
                access: None,
                static_sampler: Some(
                    StaticSamplerState::from_air_words([
                        0x807b_ff00_0008_0a49,
                        0x0000_0000_0000_3c00,
                    ])
                    .expect("static sampler"),
                ),
            },
        ],
        vertex_attributes: vec![VertexAttribute {
            location: 0,
            type_name: Some("float4".to_string()),
            name: Some("pos".to_string()),
        }],
        varyings: vec![Varying {
            location: 0,
            type_name: Some("float2".to_string()),
            name: Some("uv".to_string()),
            user_semantic: Some("user(uv)".to_string()),
        }],
        render_targets: vec![RenderTarget {
            member_index: 0,
            location: 0,
            type_name: Some("float4".to_string()),
        }],
        depth_members: vec![1],
        stencil_members: vec![2],
        local_size: Some([8, 8, 1]),
        vertex_builtins: Some(VertexBuiltins {
            uses_vertex_index: true,
            uses_instance_index: true,
            writes_position: true,
        }),
        imageblock_layouts: vec![ImageblockLayout {
            param_index: 3,
            type_layout: AirType::Scalar(AirScalar::Float),
        }],
        datalayout: Some("e-p:64:64-i64:64-n32:64".to_string()),
        function_constants: vec![FunctionConstant {
            index: 0,
            name: "myConst".to_string(),
            type_name: "i32".to_string(),
        }],
    };
    let json = serde_json::to_string(&r).expect("serialize reflection");
    let back: ShaderReflection = serde_json::from_str(&json).expect("deserialize reflection");
    assert_eq!(r, back);
    assert_eq!(back.reflection_version, REFLECTION_VERSION);
}

#[test]
fn abi_base_constants_are_the_contract() {
    assert_eq!(RESOURCE_DESCRIPTOR_SET, 0);
    assert_eq!(BUFFER_BINDING_BASE, 0);
    assert_eq!(TEXTURE_BINDING_BASE, 32);
    assert_eq!(SAMPLER_BINDING_BASE, 64);
    assert_eq!(COLOR_INPUT_BINDING_BASE, 96);
}
