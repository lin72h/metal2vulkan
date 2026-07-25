use crate::{
    seeded_texture_bytes_for_extent, seeded_unit_rgba32_float_texture_bytes, DataFormat, Extent3d,
    TextureInput, TextureRole,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TextureKind {
    Dim1d,
    Dim1dArray,
    Dim3d,
    Plain,
    Dim2dArray,
    Cube,
    CubeArray,
}

pub(crate) fn texture_seed_bytes(
    input: &TextureInput,
    kind: TextureKind,
    extent: Extent3d,
) -> Vec<u8> {
    if matches!(kind, TextureKind::Cube | TextureKind::CubeArray)
        && input.format == DataFormat::Rgba32Float
        && input.role == TextureRole::Sampled
    {
        return seeded_unit_rgba32_float_texture_bytes(extent);
    }
    seeded_texture_bytes_for_extent(input, extent)
}

pub(crate) fn texture_seed_extent(extent: Extent3d, kind: TextureKind) -> Extent3d {
    match kind {
        TextureKind::Dim1d => Extent3d::new(extent.width, 1, 1),
        TextureKind::Dim1dArray => Extent3d::new(extent.width, 1, extent.depth.max(1)),
        TextureKind::Cube => Extent3d::new(extent.width, extent.height, 6),
        TextureKind::CubeArray => {
            Extent3d::new(extent.width, extent.height, 6 * extent.depth.max(1))
        }
        TextureKind::Plain | TextureKind::Dim2dArray | TextureKind::Dim3d => extent,
    }
}

pub(crate) fn texture_output_extent(extent: Extent3d, kind: TextureKind) -> Extent3d {
    match kind {
        TextureKind::Dim1d => Extent3d::new(extent.width, 1, 1),
        TextureKind::Dim1dArray => Extent3d::new(extent.width, 1, extent.depth.max(1)),
        TextureKind::Cube => Extent3d::new(extent.width, extent.height, 1),
        TextureKind::CubeArray => Extent3d::new(extent.width, extent.height, extent.depth.max(1)),
        TextureKind::Plain | TextureKind::Dim2dArray | TextureKind::Dim3d => extent,
    }
}

#[cfg(any(target_os = "macos", test))]
pub(crate) fn texture_layer_count(extent: Extent3d, kind: TextureKind) -> usize {
    match kind {
        TextureKind::Dim1dArray | TextureKind::Dim2dArray => extent.depth.max(1) as usize,
        TextureKind::Cube => 6,
        TextureKind::CubeArray => extent.depth.max(1) as usize,
        TextureKind::Dim1d | TextureKind::Dim3d | TextureKind::Plain => 1,
    }
}

pub(crate) fn fragment_writes_depth(sanitized_ll: &str) -> bool {
    sanitized_ll
        .lines()
        .any(|line| line.contains(r#""air.depth""#))
}

pub(crate) fn texture_kind(sanitized_ll: Option<&str>, texture_location: u32) -> TextureKind {
    let Some(sanitized_ll) = sanitized_ll else {
        return TextureKind::Plain;
    };
    if let Some(meta) = metal2vulkan::meta::parse_air_kernel_meta(sanitized_ll) {
        let mut params = meta
            .roles
            .iter()
            .filter_map(|(param_idx, role)| {
                matches!(role, metal2vulkan::meta::KernRole::Texture(location) if *location == texture_location)
                    .then_some(*param_idx)
            })
            .collect::<Vec<_>>();
        params.sort_unstable();
        if let Some(param_idx) = params.into_iter().next() {
            return texture_kind_from_type_name(meta.texture_type_name(param_idx));
        }
    }
    if let Some(meta) = metal2vulkan::meta::parse_air_fragment_meta(sanitized_ll) {
        let mut params = meta
            .roles
            .iter()
            .filter_map(|(param_idx, role)| {
                matches!(role, metal2vulkan::meta::FragRole::Texture(location) if *location == texture_location)
                    .then_some(*param_idx)
            })
            .collect::<Vec<_>>();
        params.sort_unstable();
        if let Some(param_idx) = params.into_iter().next() {
            return texture_kind_from_type_name(meta.texture_type_name(param_idx));
        }
    }
    if let Some(meta) = metal2vulkan::meta::parse_air_vertex_meta(sanitized_ll) {
        let mut params = meta
            .roles
            .iter()
            .filter_map(|(param_idx, role)| {
                matches!(role, metal2vulkan::meta::VertRole::Texture(location) if *location == texture_location)
                    .then_some(*param_idx)
            })
            .collect::<Vec<_>>();
        params.sort_unstable();
        if let Some(param_idx) = params.into_iter().next() {
            return texture_kind_from_type_name(meta.texture_type_name(param_idx));
        }
    }
    TextureKind::Plain
}

pub(crate) fn texture_kind_from_type_name(name: Option<&str>) -> TextureKind {
    match name {
        Some(name) if name.starts_with("texture1d<") => TextureKind::Dim1d,
        Some(name) if name.starts_with("texture1d_array<") => TextureKind::Dim1dArray,
        Some(name) if name.starts_with("texture3d<") => TextureKind::Dim3d,
        Some(name) if name.starts_with("texture2d_array") || name.starts_with("depth2d_array") => {
            TextureKind::Dim2dArray
        }
        Some(name) if name.starts_with("texturecube_array<") => TextureKind::CubeArray,
        Some(name) if name.starts_with("texturecube<") => TextureKind::Cube,
        _ => TextureKind::Plain,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Seed;

    #[test]
    fn static_type_names_map_to_shared_texture_kinds() {
        assert_eq!(
            texture_kind_from_type_name(Some("texture1d<float, sample>")),
            TextureKind::Dim1d
        );
        assert_eq!(
            texture_kind_from_type_name(Some("texture1d_array<float, sample>")),
            TextureKind::Dim1dArray
        );
        assert_eq!(
            texture_kind_from_type_name(Some("texture3d<float, write>")),
            TextureKind::Dim3d
        );
        assert_eq!(
            texture_kind_from_type_name(Some("texture2d_array<float, read>")),
            TextureKind::Dim2dArray
        );
        assert_eq!(
            texture_kind_from_type_name(Some("depth2d_array<float, sample>")),
            TextureKind::Dim2dArray
        );
        assert_eq!(
            texture_kind_from_type_name(Some("texturecube<float, sample>")),
            TextureKind::Cube
        );
        assert_eq!(
            texture_kind_from_type_name(Some("texturecube_array<float, sample>")),
            TextureKind::CubeArray
        );
        assert_eq!(
            texture_kind_from_type_name(Some("texture2d<float, sample>")),
            TextureKind::Plain
        );
        assert_eq!(texture_kind_from_type_name(None), TextureKind::Plain);
    }

    #[test]
    fn air_metadata_selects_texture_kind_by_stage_and_location() {
        let kernel = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 5, i32 1, !"air.read", !"air.arg_type_name", !"texture3d<float, read>"}
"#;
        let duplicated_kernel_location = r#"
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.function_constant", !5, !"air.texture", !"air.location_index", i32 5, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>"}
!4 = !{i32 1, !"air.function_constant", !6, !"air.texture", !"air.location_index", i32 5, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d_array<float, sample>"}
!5 = !{}
!6 = !{}
"#;
        let fragment = r#"
!air.fragment = !{!0}
!0 = !{ptr @f, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0}
!3 = !{!4}
!4 = !{i32 0, !"air.texture", !"air.location_index", i32 6, i32 1, !"air.sample", !"air.arg_type_name", !"texturecube<float, sample>"}
"#;
        let vertex = r#"
!air.vertex = !{!0}
!0 = !{ptr @v, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 7, i32 1, !"air.read", !"air.arg_type_name", !"texture1d<float, read>"}
"#;

        assert_eq!(texture_kind(Some(kernel), 5), TextureKind::Dim3d);
        assert_eq!(
            texture_kind(Some(duplicated_kernel_location), 5),
            TextureKind::Plain
        );
        assert_eq!(texture_kind(Some(fragment), 6), TextureKind::Cube);
        assert_eq!(texture_kind(Some(vertex), 7), TextureKind::Dim1d);
        assert_eq!(texture_kind(Some(kernel), 99), TextureKind::Plain);
        assert_eq!(texture_kind(None, 5), TextureKind::Plain);
    }

    #[test]
    fn sampled_float_cube_uses_shared_unit_seed_override() {
        let extent = Extent3d::new(2, 2, 6);
        let input = TextureInput {
            index: 0,
            format: DataFormat::Rgba32Float,
            extent,
            role: TextureRole::Sampled,
            seed: Seed::ZeroForTest {
                reason: "shared texture helper test",
            },
        };

        assert_eq!(
            texture_seed_bytes(&input, TextureKind::Cube, extent),
            seeded_unit_rgba32_float_texture_bytes(extent)
        );
        assert_eq!(
            texture_seed_bytes(&input, TextureKind::Plain, extent),
            seeded_texture_bytes_for_extent(&input, extent)
        );
    }

    #[test]
    fn shared_shape_policy_handles_1d_cube_and_arrays() {
        let extent = Extent3d::new(8, 8, 3);
        assert_eq!(
            texture_seed_extent(extent, TextureKind::Dim1d),
            Extent3d::new(8, 1, 1)
        );
        assert_eq!(
            texture_seed_extent(extent, TextureKind::Dim1dArray),
            Extent3d::new(8, 1, 3)
        );
        assert_eq!(
            texture_output_extent(extent, TextureKind::Dim1d),
            Extent3d::new(8, 1, 1)
        );
        assert_eq!(
            texture_output_extent(extent, TextureKind::Dim1dArray),
            Extent3d::new(8, 1, 3)
        );
        assert_eq!(
            texture_seed_extent(extent, TextureKind::Cube),
            Extent3d::new(8, 8, 6)
        );
        assert_eq!(
            texture_output_extent(extent, TextureKind::Cube),
            Extent3d::new(8, 8, 1)
        );
        assert_eq!(
            texture_seed_extent(extent, TextureKind::CubeArray),
            Extent3d::new(8, 8, 18)
        );
        assert_eq!(
            texture_output_extent(extent, TextureKind::CubeArray),
            Extent3d::new(8, 8, 3)
        );
        assert_eq!(texture_layer_count(extent, TextureKind::Dim2dArray), 3);
        assert_eq!(texture_layer_count(extent, TextureKind::Dim1dArray), 3);
        assert_eq!(texture_layer_count(extent, TextureKind::Cube), 6);
        assert_eq!(
            texture_layer_count(
                texture_seed_extent(extent, TextureKind::CubeArray),
                TextureKind::CubeArray
            ),
            18
        );
        assert_eq!(texture_layer_count(extent, TextureKind::Dim3d), 1);
    }

    #[test]
    fn depth_output_metadata_is_detected() {
        assert!(fragment_writes_depth(r#"!1 = !{!"air.depth", i32 0}"#));
        assert!(!fragment_writes_depth(
            r#"!1 = !{!"air.render_target", i32 0}"#
        ));
    }
}
