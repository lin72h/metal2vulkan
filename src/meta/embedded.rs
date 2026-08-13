use super::types::{struct_member_starts_at, tokenize, Tok};
use super::{arg_type_name, role_strings, texture_shape_from_name, BufferAccess, TextureFormat};
use crate::passes::ImageComp;
use spirv::Dim;
use std::collections::HashMap;

/// A texture that lives inside an `air.indirect_buffer` argument buffer, marked
/// `air.indirect_argument` → `air.texture` in the buffer's `air.struct_type_info`, and used by the
/// shader body through an AIR texture intrinsic.
///
/// The translator materializes a synthetic UniformConstant sampled image for it and the validation
/// harness binds the SAME deterministically-seeded texture on both the Apple-oracle side (encoded
/// into the arg buffer via `MTLArgumentEncoder`) and the Vulkan-runner side (the reflected synthetic
/// descriptor). The authored identity remains the owning buffer plus field offset; the synthetic
/// texture index `K` is translator-owned (see [`embedded_synthetic_texture_index`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddedTexture {
    /// Kernel parameter index of the owning `air.indirect_buffer` argument.
    pub buffer_param_index: u32,
    /// Kernel parameter index of the owning `air.indirect_buffer` argument.
    pub buffer_index: u32,
    /// Byte offset of the texture handle within the argument-buffer struct.
    pub field_offset: u32,
    pub field_ordinal: u32,
    /// Metal argument-encoder index (`[[id(n)]]`) of this texture inside the argument buffer.
    pub argument_index: u32,
    /// Texture dimensionality (2D, with `arrayed` carrying the 2D-array distinction).
    pub dim: Dim,
    pub arrayed: bool,
    /// Sampled/storage component type (Float for `texture2d<float, read/write>`).
    pub comp: ImageComp,
    /// Storage-image format for write/read_write textures; `None` for sampled/read textures.
    pub storage_format: Option<TextureFormat>,
    /// Fixed number of opaque texture handles in this argument-buffer field. `None` denotes one
    /// handle; embedded runtime `array_ref` fields are not yet representable by Metal's fixed
    /// argument-buffer layout metadata.
    pub array_length: Option<u32>,
    /// The synthetic texture index `K` this embedded texture binds at.
    pub synthetic_texture_index: u32,
}

/// One resource-handle member of an AIR indirect argument buffer. This is the common structural
/// coordinate used by authored embedded resources, Metal argument encoding, and static lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddedArgument {
    pub buffer_param_index: u32,
    pub buffer_index: u32,
    pub field_ordinal: u32,
    pub field_offset: u32,
    pub argument_index: u32,
    /// Nested Metal buffer location for an `air.buffer` field; absent for constants, textures, and
    /// function-table handles represented by the same structural coordinate.
    pub resource_buffer_index: Option<u32>,
    pub resource_address_space: Option<u32>,
    pub resource_declared_size: Option<u32>,
    pub resource_access: Option<BufferAccess>,
}

/// The synthetic texture index `K` for embedded argument-buffer textures, derived so the translator
/// (binding decoration) and the validation harness (model + oracle) agree WITHOUT any name key.
///
/// **ABI convention:** `K = 1 + max(top-level air.texture air.location_index)`, or `0` when the
/// kernel has no top-level textures. This guarantees `K` never collides with a real top-level
/// texture's location index, so the synthetic sampled image binds at a free
/// `TEXTURE_BINDING_BASE + K` slot. `texture_locations` is the set of `air.location_index` values of
/// the kernel's top-level texture args.
pub fn embedded_synthetic_texture_index(texture_locations: &[u32]) -> u32 {
    texture_locations.iter().copied().max().map_or(0, |m| m + 1)
}

/// True iff the shader body calls an AIR texture intrinsic. These are stable AIR intrinsic
/// symbols, not shader identifiers. A declaration line is only emitted when the body actually calls it,
/// so its presence in the `.ll` text is a sound structural signal.
pub(super) fn body_uses_texture_intrinsic(ll: &str) -> bool {
    ll.contains("@air.sample_")
        || ll.contains("@air.gather_")
        || ll.contains("@air.read_texture")
        || ll.contains("@air.write_texture")
        || ll.contains("@air.write_imageblock_slice_to_texture")
}

/// Scan each `air.indirect_buffer`'s `air.struct_type_info` for members marked
/// `air.indirect_argument` whose nested node is an `air.texture`. Each such texture is
/// surfaced as an `EmbeddedTexture` with a synthetic index K, K+1, … (K from
/// [`embedded_synthetic_texture_index`]) so the translator and harness agree on the binding without
/// a name key. Plain 2D and 2D-array float/int textures share this structural contract.
pub(super) fn detect_embedded_textures(
    nodes: &HashMap<u32, String>,
    indirect_buffer_struct_refs: &[(u32, u32, u32)],
    top_level_texture_locations: &[u32],
) -> Vec<EmbeddedTexture> {
    let mut out = vec![];
    let mut next_k = embedded_synthetic_texture_index(top_level_texture_locations);
    for &(buffer_param_index, buffer_index, sref) in indirect_buffer_struct_refs {
        for (field_ordinal, field_offset, tex_node_ref) in embedded_argument_members(nodes, sref) {
            let Some(tex_node) = nodes.get(&tex_node_ref) else {
                continue;
            };
            let strs = role_strings(tex_node);
            // Structural gate: the nested node must be an `air.texture` with an access class the
            // read/write lowerings can express.
            if !strs.iter().any(|s| s == "texture")
                || !strs
                    .iter()
                    .any(|s| matches!(s.as_str(), "sample" | "read" | "write" | "read_write"))
            {
                continue;
            }
            let name = arg_type_name(tex_node).unwrap_or_default();
            let shape = texture_shape_from_name(&name);
            let dim = shape.dimension.to_spirv_dim();
            if dim != Dim::Dim2D {
                continue;
            }
            out.push(EmbeddedTexture {
                buffer_param_index,
                buffer_index,
                field_offset,
                field_ordinal,
                argument_index: super::location_index(tex_node, field_ordinal),
                dim,
                arrayed: shape.arrayed,
                comp: shape.component.to_image_comp(),
                storage_format: shape.storage_format,
                array_length: shape.array_length,
                synthetic_texture_index: next_k,
            });
            next_k += 1;
        }
    }
    out
}

pub(super) fn detect_embedded_arguments(
    nodes: &HashMap<u32, String>,
    indirect_buffer_struct_refs: &[(u32, u32, u32)],
) -> Vec<EmbeddedArgument> {
    indirect_buffer_struct_refs
        .iter()
        .flat_map(|&(buffer_param_index, buffer_index, sref)| {
            embedded_argument_members(nodes, sref)
                .into_iter()
                .filter_map(move |(field_ordinal, field_offset, node_ref)| {
                    let node = nodes.get(&node_ref)?;
                    let is_buffer = role_strings(node).iter().any(|role| role == "buffer");
                    Some(EmbeddedArgument {
                        buffer_param_index,
                        buffer_index,
                        field_ordinal,
                        field_offset,
                        argument_index: super::location_index(node, field_ordinal),
                        resource_buffer_index: is_buffer
                            .then(|| super::location_index(node, field_ordinal)),
                        resource_address_space: is_buffer
                            .then(|| super::address_space(node))
                            .flatten(),
                        resource_declared_size: is_buffer
                            .then(|| super::i32_after_marker(node, "air.arg_type_size"))
                            .flatten(),
                        resource_access: is_buffer
                            .then(|| super::declared_buffer_access(node))
                            .flatten(),
                    })
                })
        })
        .collect()
}

/// Yield `(field_offset, nested_node_ref)` for every `air.indirect_argument` member of an
/// `air.struct_type_info` node. Each member of a struct-type-info node is a 5-tuple
/// `i32 offset, i32 size, i32 array_len, !"type", !"name"`, optionally PREFIXED by
/// `!"air.indirect_argument", !N` — a SUFFIX that trails the member's `name` string (e.g.
/// `i32 0, i32 8, i32 0, !"texture2d<...>", !"texture", !"air.indirect_argument", !22`), mirroring the
/// tuple layout `parse_struct_info` walks. We recover the member `offset` and the `!N` of the
/// `air.indirect_argument` suffix that belongs to THAT member, so the caller can classify the nested
/// node. (Nested `air.struct_type_info` members are walked but never carry a top-level embedded
/// texture, so their suffix is ignored.)
fn embedded_argument_members(nodes: &HashMap<u32, String>, sref: u32) -> Vec<(u32, u32, u32)> {
    let Some(body) = nodes.get(&sref) else {
        return vec![];
    };
    let toks = tokenize(body);
    let mut out = vec![];
    let mut i = 0;
    let mut field_ordinal = 0u32;
    while i < toks.len() {
        // Optional nested-struct prefix (`!"air.struct_type_info", !N`).
        if let (Some(Tok::Str(s)), Some(Tok::Ref(_))) = (toks.get(i), toks.get(i + 1)) {
            if s == "air.struct_type_info" {
                i += 2;
            }
        }
        // Member tuple: offset, size, array_len, type-name (name string follows).
        let offset = match (
            toks.get(i),
            toks.get(i + 1),
            toks.get(i + 2),
            toks.get(i + 3),
        ) {
            (Some(Tok::Int(off)), Some(Tok::Int(_)), Some(Tok::Int(_)), Some(Tok::Str(_))) => *off,
            _ => break,
        };
        i += 5; // 3 ints + type + name
                // Trailing tokens up to the next member: an `!"air.indirect_argument", !N` SUFFIX here binds
                // node `!N` to THIS member (its `offset`). Use `struct_member_starts_at` (which does NOT treat
                // the `air.indirect_argument` marker as a member start) as the stop test, so the suffix is
                // scanned rather than skipped; the scan ends at the next real member tuple.
        while i < toks.len() && !struct_member_starts_at(&toks, i) {
            if let (Some(Tok::Str(s)), Some(Tok::Ref(x))) = (toks.get(i), toks.get(i + 1)) {
                if s == "air.indirect_argument" {
                    out.push((field_ordinal, offset, *x));
                }
            }
            i += 1;
        }
        field_ordinal += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::TextureFormat;

    #[test]
    fn embedded_texture_use_gate_includes_sampling_and_gathering() {
        assert!(body_uses_texture_intrinsic(
            "call <4 x float> @air.sample_texture_2d(...)"
        ));
        assert!(body_uses_texture_intrinsic(
            "call <4 x float> @air.gather_texture_2d(...)"
        ));
        assert!(!body_uses_texture_intrinsic(
            "declare void @unrelated_texture_helper()"
        ));
    }

    #[test]
    fn embedded_texture_detection_includes_read_and_write_fields() {
        let mut nodes = HashMap::new();
        nodes.insert(
            10,
            r#"i32 0, i32 8, i32 0, !"texture2d<float, read>", !"input", !"air.indirect_argument", !20, i32 8, i32 8, i32 0, !"texture2d<float, write>", !"output", !"air.indirect_argument", !21"#.to_string(),
        );
        nodes.insert(
            20,
            r#"i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>""#.to_string(),
        );
        nodes.insert(
            21,
            r#"i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>""#.to_string(),
        );

        let textures = detect_embedded_textures(&nodes, &[(4, 0, 10)], &[]);
        assert_eq!(textures.len(), 2);
        assert_eq!(textures[0].buffer_param_index, 4);
        assert_eq!(textures[0].field_offset, 0);
        assert_eq!(textures[0].argument_index, 0);
        assert_eq!(textures[0].storage_format, None);
        assert!(!textures[0].arrayed);
        assert_eq!(textures[0].synthetic_texture_index, 0);
        assert_eq!(textures[1].field_offset, 8);
        assert_eq!(textures[1].argument_index, 1);
        assert_eq!(textures[1].storage_format, Some(TextureFormat::R32f));
        assert_eq!(textures[1].array_length, None);
        assert!(!textures[1].arrayed);
        assert_eq!(textures[1].synthetic_texture_index, 1);
    }

    #[test]
    fn embedded_fixed_depth_array_preserves_handle_count_and_image_shape() {
        let mut nodes = HashMap::new();
        nodes.insert(
            10,
            r#"i32 0, i32 16, i32 0, !"array<depth2d_array<float, sample>, 2>", !"depth", !"air.indirect_argument", !20"#.to_string(),
        );
        nodes.insert(
            20,
            r#"i32 0, !"air.texture", !"air.location_index", i32 4, i32 2, !"air.sample", !"air.arg_type_name", !"array<depth2d_array<float, sample>, 2>""#.to_string(),
        );

        let textures = detect_embedded_textures(&nodes, &[(1, 16, 10)], &[]);
        assert_eq!(textures.len(), 1);
        assert_eq!(textures[0].argument_index, 4);
        assert_eq!(textures[0].array_length, Some(2));
        assert!(textures[0].arrayed);
        assert_eq!(textures[0].storage_format, None);
    }

    #[test]
    fn embedded_argument_detection_classifies_only_nested_buffers_as_device_resources() {
        let mut nodes = HashMap::new();
        nodes.insert(
            10,
            r#"i32 0, i32 4, i32 0, !"uint", !"count", !"air.indirect_argument", !20, i32 8, i32 8, i32 0, !"float", !"values", !"air.indirect_argument", !21"#.to_string(),
        );
        nodes.insert(
            20,
            r#"i32 0, !"air.indirect_constant", !"air.location_index", i32 2, i32 1"#.to_string(),
        );
        nodes.insert(
            21,
            r#"i32 1, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.address_space", i32 1"#.to_string(),
        );

        let arguments = detect_embedded_arguments(&nodes, &[(3, 5, 10)]);
        assert_eq!(arguments.len(), 2);
        assert_eq!(arguments[0].resource_buffer_index, None);
        assert_eq!(arguments[1].buffer_param_index, 3);
        assert_eq!(arguments[1].buffer_index, 5);
        assert_eq!(arguments[1].field_offset, 8);
        assert_eq!(arguments[1].argument_index, 7);
        assert_eq!(arguments[1].resource_buffer_index, Some(7));
        assert_eq!(arguments[1].resource_address_space, Some(1));
    }
}
