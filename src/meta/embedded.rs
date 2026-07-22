use super::types::{struct_member_starts_at, tokenize, Tok};
use super::{arg_type_name, role_strings, texture_shape_from_name};
use crate::passes::ImageComp;
use spirv::Dim;
use std::collections::HashMap;

/// A texture that lives inside an `air.indirect_buffer` argument buffer, marked
/// `air.indirect_argument` → `air.texture` in the buffer's `air.struct_type_info`, and read by the
/// kernel body through an integer-coord `air.read_texture` (an exact texel fetch, no filtering).
///
/// The translator materializes a synthetic UniformConstant sampled image for it and the validation
/// harness binds the SAME deterministically-seeded texture on both the Apple-oracle side (encoded
/// into the arg buffer via `MTLArgumentEncoder`) and the Vulkan-runner side (a plain `TextureInput`),
/// so all layers agree on one number: the synthetic texture index `K` (see
/// [`embedded_synthetic_texture_index`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddedTexture {
    /// Kernel parameter index of the owning `air.indirect_buffer` argument.
    pub buffer_index: u32,
    /// Byte offset of the texture handle within the argument-buffer struct.
    pub field_offset: u32,
    /// Texture dimensionality (only 2D is currently detected/supported).
    pub dim: Dim,
    /// Sampled component type (Float for `texture2d<float, sample>`).
    pub comp: ImageComp,
    /// The synthetic texture index `K` this embedded texture binds at.
    pub synthetic_texture_index: u32,
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

/// True iff the kernel body calls an `air.read_texture*` intrinsic — an integer-coord exact texel
/// fetch. `air.read_texture` is a stable AIR intrinsic symbol (part of the AIR ABI), not a shader
/// identifier. A declaration line (`declare ... @air.read_texture_2d.*`) is only emitted when the
/// body actually calls it, so its presence in the `.ll` text is a sound structural signal.
pub(super) fn body_uses_read_texture(ll: &str) -> bool {
    ll.contains("@air.read_texture")
}

/// Scan each `air.indirect_buffer`'s `air.struct_type_info` for members marked
/// `air.indirect_argument` whose nested node is an `air.texture` (sampled). Each such texture is
/// surfaced as an `EmbeddedTexture` with a synthetic index K, K+1, … (K from
/// [`embedded_synthetic_texture_index`]) so the translator and harness agree on the binding without
/// a name key. Only 2D sampled float/int textures are detected (the shape the read path supports).
pub(super) fn detect_embedded_textures(
    nodes: &HashMap<u32, String>,
    indirect_buffer_struct_refs: &[(u32, u32)],
    top_level_texture_locations: &[u32],
) -> Vec<EmbeddedTexture> {
    let mut out = vec![];
    let mut next_k = embedded_synthetic_texture_index(top_level_texture_locations);
    for &(buffer_index, sref) in indirect_buffer_struct_refs {
        for (field_offset, tex_node_ref) in embedded_texture_members(nodes, sref) {
            let Some(tex_node) = nodes.get(&tex_node_ref) else {
                continue;
            };
            let strs = role_strings(tex_node);
            // Structural gate: the nested node must be an `air.texture` that is `air.sample`-capable.
            if !strs.iter().any(|s| s == "texture") || !strs.iter().any(|s| s == "sample") {
                continue;
            }
            let name = arg_type_name(tex_node).unwrap_or_default();
            let shape = texture_shape_from_name(&name);
            let dim = shape.dimension.to_spirv_dim();
            // Only the plain 2D non-arrayed shape is supported by the read lowering today.
            if dim != Dim::Dim2D || shape.arrayed {
                continue;
            }
            out.push(EmbeddedTexture {
                buffer_index,
                field_offset,
                dim,
                comp: shape.component.to_image_comp(),
                synthetic_texture_index: next_k,
            });
            next_k += 1;
        }
    }
    out
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
fn embedded_texture_members(nodes: &HashMap<u32, String>, sref: u32) -> Vec<(u32, u32)> {
    let Some(body) = nodes.get(&sref) else {
        return vec![];
    };
    let toks = tokenize(body);
    let mut out = vec![];
    let mut i = 0;
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
                    out.push((offset, *x));
                }
            }
            i += 1;
        }
    }
    out
}
