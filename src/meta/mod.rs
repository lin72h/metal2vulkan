//! AIR stage-interface metadata, parsed from the ORIGINAL `.ll` text BEFORE llc runs (llc drops the
//! `!air.fragment` / `!air.vertex` metadata nodes). Each entry-function parameter, in SPIR-V
//! `OpFunctionParameter` order (which llc preserves), gets a role telling the interface pass what
//! Vulkan binding to synthesize for it. Ported faithfully from `metal2vulkanspirv::parse_air_fragment_meta`,
//! plus a sibling `!air.vertex` parser the old crate handled structurally rather than from metadata.

use std::collections::HashMap;

mod embedded;
mod function_constants;
mod globals;
mod textures;
mod types;
use embedded::{body_uses_read_texture, detect_embedded_textures};
pub use embedded::{embedded_synthetic_texture_index, EmbeddedTexture};
pub use function_constants::{parse_function_constants, FunctionConstant};
use globals::{location_index_with_static, static_init_int_global_values};
pub use textures::{
    texture_shape_from_name, TextureComponent, TextureDimension, TextureFormat, TextureShape,
};
use types::{parse_struct_info, struct_info_ref};
pub use types::{primitive_air_type_from_name, AirMember, AirScalar, AirType};

/// Role of a single fragment-shader entry parameter, keyed by its parameter index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FragRole {
    /// `[[position]]` -> Input BuiltIn FragCoord (often unused).
    Position,
    /// `[[stage_in]]` interpolated input -> Input var at Location N (N = order among fragment_inputs).
    Varying(u32),
    /// `[[texture(n)]]` -> UniformConstant sampled image.
    Texture(u32),
    /// `[[sampler(n)]]` -> UniformConstant sampler.
    Sampler(u32),
    /// `[[buffer(n)]]` -> Uniform/StorageBuffer block.
    Buffer(u32),
    /// `[[color(n)]]` framebuffer-fetch input -> Vulkan input attachment (out of scope this milestone).
    ColorInput(u32),
    /// Anything we don't model.
    Other,
}

/// A fragment shader's decoded parameter roles + render-target count/indices.
#[derive(Clone, Debug, Default)]
pub struct FragMeta {
    /// `(param_idx, role)` — one per fragment input, in declaration order.
    pub roles: Vec<(u32, FragRole)>,
    /// `fragment_input` Location -> AIR type name (`float2`, `float4`, ...). Used by passthrough
    /// vertex synthesis when the pipeline binds a built-in vertex slot.
    pub varying_types: HashMap<u32, String>,
    /// `fragment_input` Location -> Metal field/argument name, when AIR metadata carries one. The
    /// Metal oracle uses this to generate a vertex struct that Apple's pipeline linker matches.
    pub varying_names: HashMap<u32, String>,
    /// `fragment_input` Location -> Metal user semantic, such as `user(texturecoord)`, when AIR
    /// metadata carries one.
    pub varying_user_semantics: HashMap<u32, String>,
    /// number of `air.render_target` outputs (MRT count; 1 for the common single-output case).
    pub n_render_targets: u32,
    /// Return-struct member index -> color attachment Location for actual `air.render_target`
    /// outputs. Non-color outputs such as `air.stencil` are deliberately absent.
    pub render_target_members: Vec<(u32, u32)>,
    /// Return-struct member index -> AIR render-target type name (`float4`, `int4`, ...).
    pub render_target_type_names: HashMap<u32, String>,
    /// Return-struct member indices tagged as `air.depth` (`[[depth(...)]]`).
    pub depth_members: Vec<u32>,
    /// Return-struct member indices tagged as `air.stencil` (`[[stencil]]`).
    pub stencil_members: Vec<u32>,
    /// Render-target locations in AIR output metadata order. A single-output fragment can legally
    /// write a nonzero MRT slot, e.g. coverage shaders writing `[[color(1)]]`.
    pub render_target_indices: Vec<u32>,
    /// `param_idx -> reconstructed struct layout` for buffer args that carry `air.struct_type_info`.
    /// Used to rebuild the real struct when the backend collapsed the buffer into a bare pointer.
    pub buffer_layouts: HashMap<u32, AirType>,
    /// `param_idx -> AIR address space` for buffer args, when the AIR node carries it (device=1,
    /// constant=2). Populated only from `air.address_space` / the function param pointer address
    /// space — absent (not guessed) when the IR does not state it. Mirrors
    /// [`KernMeta::buffer_address_spaces`] for the fragment stage.
    pub buffer_address_spaces: HashMap<u32, u32>,
    /// `param_idx -> declared AIR buffer byte size` (`air.arg_type_size` / `air.buffer_size`), when
    /// the AIR node carries it. Mirrors [`KernMeta::buffer_type_sizes`] for the fragment stage.
    pub buffer_type_sizes: HashMap<u32, u32>,
    /// `param_idx -> AIR texture argument type name`, e.g. `texture2d<uint, read>`.
    pub texture_type_names: HashMap<u32, String>,
}

impl FragMeta {
    pub fn role_of(&self, idx: u32) -> Option<&FragRole> {
        self.roles.iter().find(|(i, _)| *i == idx).map(|(_, r)| r)
    }
    pub fn layout_of(&self, idx: u32) -> Option<&AirType> {
        self.buffer_layouts.get(&idx)
    }
    pub fn texture_type_name(&self, idx: u32) -> Option<&str> {
        self.texture_type_names.get(&idx).map(String::as_str)
    }
    pub fn varying_type(&self, loc: u32) -> Option<&str> {
        self.varying_types.get(&loc).map(String::as_str)
    }
    pub fn varying_name(&self, loc: u32) -> Option<&str> {
        self.varying_names.get(&loc).map(String::as_str)
    }
    pub fn varying_user_semantic(&self, loc: u32) -> Option<&str> {
        self.varying_user_semantics.get(&loc).map(String::as_str)
    }
    pub fn render_target_location_for_member(&self, member_idx: u32) -> Option<u32> {
        self.render_target_members
            .iter()
            .find_map(|(member, location)| (*member == member_idx).then_some(*location))
    }
    pub fn render_target_type_name(&self, member_idx: u32) -> Option<&str> {
        self.render_target_type_names
            .get(&member_idx)
            .map(String::as_str)
    }
    pub fn is_depth_member(&self, member_idx: u32) -> bool {
        self.depth_members.contains(&member_idx)
    }
    pub fn is_stencil_member(&self, member_idx: u32) -> bool {
        self.stencil_members.contains(&member_idx)
    }
}

/// Role of a single vertex-shader entry parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VertRole {
    /// `[[stage_in]]` / `[[attribute(n)]]` vertex attribute -> Input var at Location N.
    VertexInput(u32),
    /// `[[buffer(n)]]` -> Uniform/StorageBuffer block.
    Buffer(u32),
    /// `[[texture(n)]]` -> sampled image (vertex texture fetch). WindowServer vertices read data via
    /// textures, so the vertex stage needs the same texture/sampler handling as the fragment stage.
    Texture(u32),
    /// `[[sampler(n)]]` -> sampler.
    Sampler(u32),
    /// `[[vertex_id]]` -> Input BuiltIn VertexIndex (32-bit uint).
    VertexId,
    /// `[[instance_id]]` -> Input BuiltIn InstanceIndex (32-bit uint).
    InstanceId,
    Other,
}

/// Role of a single vertex return-struct member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VertOutRole {
    /// `[[position]]` -> Output BuiltIn Position.
    Position,
    /// `[[point_size]]` -> Output BuiltIn PointSize. Does not consume a Location.
    PointSize,
    /// `[[viewport_array_index]]` -> Output BuiltIn ViewportIndex. Does not consume a Location.
    ViewportArrayIndex,
    /// User varying output -> Output var at Location N.
    Varying(u32),
    Other,
}

/// A vertex shader's decoded parameter roles. The OUTPUT struct (member 0 = `[[position]]`, the rest
/// follows `output_roles`; point-size-style builtins do not consume varying locations.
#[derive(Clone, Debug, Default)]
pub struct VertMeta {
    pub roles: Vec<(u32, VertRole)>,
    pub output_roles: Vec<VertOutRole>,
    /// Vertex input Location -> AIR type name (`float2`, `float4`, ...). Used by conformance
    /// oracles that must synthesize a Metal vertex descriptor before pipeline reflection exists.
    pub vertex_input_types: HashMap<u32, String>,
    /// Vertex input Location -> Metal argument name, when AIR metadata carries one.
    pub vertex_input_names: HashMap<u32, String>,
    /// `param_idx -> reconstructed struct layout` for buffer args (see `FragMeta::buffer_layouts`).
    pub buffer_layouts: HashMap<u32, AirType>,
    /// `param_idx -> AIR address space` for buffer args, when the AIR carries it (see
    /// [`FragMeta::buffer_address_spaces`]).
    pub buffer_address_spaces: HashMap<u32, u32>,
    /// `param_idx -> declared AIR buffer byte size` for buffer args, when the AIR carries it (see
    /// [`FragMeta::buffer_type_sizes`]).
    pub buffer_type_sizes: HashMap<u32, u32>,
    /// `param_idx -> AIR texture argument type name`, e.g. `texture2d<uint, read>`.
    pub texture_type_names: HashMap<u32, String>,
}

impl VertMeta {
    pub fn role_of(&self, idx: u32) -> Option<&VertRole> {
        self.roles.iter().find(|(i, _)| *i == idx).map(|(_, r)| r)
    }
    pub fn layout_of(&self, idx: u32) -> Option<&AirType> {
        self.buffer_layouts.get(&idx)
    }
    pub fn texture_type_name(&self, idx: u32) -> Option<&str> {
        self.texture_type_names.get(&idx).map(String::as_str)
    }
    pub fn output_role_of(&self, idx: u32) -> Option<&VertOutRole> {
        self.output_roles.get(idx as usize)
    }
    pub fn vertex_input_type(&self, loc: u32) -> Option<&str> {
        self.vertex_input_types.get(&loc).map(String::as_str)
    }
    pub fn vertex_input_name(&self, loc: u32) -> Option<&str> {
        self.vertex_input_names.get(&loc).map(String::as_str)
    }
}

/// Role of a single compute-kernel entry parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernRole {
    /// AIR `buffer` argument. `KernMeta::buffer_address_space` distinguishes device/constant
    /// resource buffers from threadgroup scratch buffers.
    Buffer(u32),
    /// `[[texture(n)]]` -> UniformConstant sampled image.
    Texture(u32),
    /// `[[sampler(n)]]` -> UniformConstant sampler.
    Sampler(u32),
    /// Host-populated StorageBuffer shadow for an opaque Metal acceleration structure used by AIR
    /// introspection intrinsics. Bound at the resource's `air.location_index`; see `as_shadow`.
    AccelerationStructureShadow(u32),
    /// `[[threads_per_threadgroup]]` (`uint` or `uint3`) -> the execution local size.
    /// Scalar params receive `64`; vector params receive `(64, 1, 1)` for the current harness.
    ThreadsPerThreadgroup,
    /// `[[thread_position_in_threadgroup]]` (`uint` or `uint3`) -> LocalInvocationId.
    /// Scalar params receive component .x; vector params receive the full v3uint.
    ThreadPositionInThreadgroup,
    /// `[[threadgroups_per_grid]]` (`uint` or `uint3`) -> NumWorkgroups.
    /// Scalar params receive component .x; vector params receive the full v3uint.
    ThreadgroupsPerGrid,
    /// `[[threads_per_grid]]` (`uint` or `uint3`) -> NumWorkgroups * LocalSize.
    /// Scalar params receive component .x; vector params receive the requested vector prefix.
    ThreadsPerGrid,
    /// `[[threadgroup_position_in_grid]]` (`uint` or `uint3`) -> WorkgroupId.
    /// Scalar params receive component .x; vector params receive the full v3uint.
    ThreadgroupPositionInGrid,
    /// `[[thread_index_in_threadgroup]]` (`uint`) -> LocalInvocationIndex.
    ThreadIndexInThreadgroup,
    /// `[[thread_index_in_quadgroup]]` (`uint`) -> lane within the current 4-wide quadgroup.
    ThreadIndexInQuadgroup,
    /// `[[quadgroup_index_in_threadgroup]]` (`uint`) -> quadgroup number within the threadgroup.
    QuadgroupIndexInThreadgroup,
    /// `[[thread_index_in_simdgroup]]` (`uint`) -> lane within the current 32-wide simdgroup.
    ThreadIndexInSimdgroup,
    /// `[[simdgroup_index_in_threadgroup]]` (`uint`) -> simdgroup number within the threadgroup.
    SimdgroupIndexInThreadgroup,
    /// `[[threads_per_simdgroup]]` (`uint`) -> the AIR simdgroup width.
    ThreadsPerSimdgroup,
    /// `[[simdgroups_per_threadgroup]]` (`uint`) -> number of 32-wide AIR simdgroups in the local size.
    SimdgroupsPerThreadgroup,
    /// `[[thread_position_in_grid]]` (`uint` or `uint3`) -> GlobalInvocationId.
    /// Scalar params receive component .x; vector params receive the full v3uint.
    ThreadPositionInGrid,
    Other,
}

/// A compute kernel's decoded parameter roles. The body has no return value (a `void` kernel that
/// writes through its buffer pointers), so no output handling is needed.
#[derive(Clone, Debug, Default)]
pub struct KernMeta {
    pub roles: Vec<(u32, KernRole)>,
    /// `param_idx -> reconstructed struct layout` for buffer args (see `FragMeta::buffer_layouts`).
    pub buffer_layouts: HashMap<u32, AirType>,
    /// `param_idx -> reconstructed air.imageblock_data layout` for imageblock args.
    pub imageblock_layouts: HashMap<u32, AirType>,
    /// `param_idx -> AIR address space` for buffer args. Address space 3 is threadgroup memory.
    pub buffer_address_spaces: HashMap<u32, u32>,
    /// `param_idx -> declared AIR buffer argument byte size`, from `air.arg_type_size` or
    /// `air.buffer_size` when present.
    pub buffer_type_sizes: HashMap<u32, u32>,
    /// `param_idx -> AIR buffer argument type name`, e.g. `char`, `void`, or a struct name.
    pub buffer_type_names: HashMap<u32, String>,
    /// `param_idx -> AIR texture argument type name`, e.g. `texture2d<uint, read>`.
    pub texture_type_names: HashMap<u32, String>,
    /// Textures EMBEDDED inside an `air.indirect_buffer` argument buffer (via `air.indirect_argument`
    /// → nested `air.texture`) that the kernel body reads with an integer-coord `air.read_texture`.
    /// Each is surfaced as a standalone sampled-image resource so the read lowers to a real
    /// `OpImageFetch` instead of a null. See [`EmbeddedTexture`].
    pub embedded_textures: Vec<EmbeddedTexture>,
}

impl KernMeta {
    pub fn role_of(&self, idx: u32) -> Option<&KernRole> {
        self.roles.iter().find(|(i, _)| *i == idx).map(|(_, r)| r)
    }
    pub fn layout_of(&self, idx: u32) -> Option<&AirType> {
        self.buffer_layouts.get(&idx)
    }
    pub fn imageblock_layout_of(&self, idx: u32) -> Option<&AirType> {
        self.imageblock_layouts.get(&idx)
    }
    pub fn buffer_address_space(&self, idx: u32) -> Option<u32> {
        self.buffer_address_spaces.get(&idx).copied()
    }
    pub fn buffer_type_size(&self, idx: u32) -> Option<u32> {
        self.buffer_type_sizes.get(&idx).copied()
    }
    pub fn buffer_type_name(&self, idx: u32) -> Option<&str> {
        self.buffer_type_names.get(&idx).map(String::as_str)
    }
    pub fn texture_type_name(&self, idx: u32) -> Option<&str> {
        self.texture_type_names.get(&idx).map(String::as_str)
    }
}

/// Parse `!air.kernel` into a `KernMeta`. Structure (RE'd from the compute fixtures):
///   `!air.kernel = !{!N}`; `!N = !{ptr @k, !EMPTY, !IN}`;
///   `!IN = !{!a, !b, ...}` each `!{i32 idx, !"air.buffer"|"air.texture"|..., ...}`.
pub fn parse_air_kernel_meta(ll: &str) -> Option<KernMeta> {
    parse_air_kernel_meta_with(ll, false)
}

/// Like [`parse_air_kernel_meta`], but with `promote_fc_buffers` controlling whether a
/// `[[function_constant]]`-gated `air.buffer` param is classified as a REAL StorageBuffer binding
/// (`true`) or left as the default possibly-absent Private placeholder (`false`). Every production
/// entry point uses `false`; only the adopt-if-validates `fc_promote_psb` retry passes `true` (see
/// [`fc_promoted_role`]). Standalone callers can request either projection; production parses both
/// projections from one shared metadata-node table before the transform pipeline runs.
pub fn parse_air_kernel_meta_with(ll: &str, promote_fc_buffers: bool) -> Option<KernMeta> {
    let nodes = collect_nodes(ll);
    let entry = entry_name_from_nodes(ll, "kernel", &nodes);
    parse_air_kernel_meta_with_nodes(ll, promote_fc_buffers, &nodes, entry.as_deref())
}

/// Parse the default and FC-buffer-promoted kernel projections from one metadata-node table. The
/// retry cascade needs both projections, but their only difference is how a stable
/// `air.function_constant` wrapper classifies a wrapped buffer; collecting and decoding the AIR
/// metadata twice is unnecessary.
pub(crate) fn parse_air_kernel_meta_variants(
    ll: &str,
) -> (Option<KernMeta>, Option<KernMeta>, Option<String>) {
    let nodes = collect_nodes(ll);
    let entry = entry_name_from_nodes(ll, "kernel", &nodes);
    let default = parse_air_kernel_meta_with_nodes(ll, false, &nodes, entry.as_deref());
    let promoted = parse_air_kernel_meta_with_nodes(ll, true, &nodes, entry.as_deref());
    (default, promoted, entry)
}

fn parse_air_kernel_meta_with_nodes(
    ll: &str,
    promote_fc_buffers: bool,
    nodes: &HashMap<u32, String>,
    entry: Option<&str>,
) -> Option<KernMeta> {
    let root = stage_root(ll, "kernel")?;
    let rootc = nodes.get(&root)?;
    let param_address_spaces = entry
        .and_then(|name| function_param_pointer_address_spaces(ll, name))
        .unwrap_or_default();
    let refs = refs_in(rootc);
    // The argument-info list is the SECOND ref (`!EMPTY` is the first — an empty placeholder node).
    let in_ref = *refs.get(1)?;

    let mut roles = vec![];
    let mut buffer_layouts = HashMap::new();
    let mut imageblock_layouts = HashMap::new();
    let mut buffer_address_spaces = HashMap::new();
    let mut buffer_type_sizes = HashMap::new();
    let mut buffer_type_names = HashMap::new();
    let mut texture_type_names = HashMap::new();
    // `air.location_index` of every top-level `air.texture` arg — the basis for the synthetic
    // embedded-texture index K (see `embedded_synthetic_texture_index`).
    let mut top_level_texture_locations: Vec<u32> = vec![];
    // `(buffer_param_index, struct_type_info_node_ref)` for each `air.indirect_buffer` arg, so
    // embedded-texture detection can run once K is known (after the whole arg list is scanned).
    let mut indirect_buffer_struct_refs: Vec<(u32, u32)> = vec![];
    for r in refs_in(nodes.get(&in_ref)?) {
        let Some(node) = nodes.get(&r) else { continue };
        let Some(idx) = first_i32(node) else { continue };
        let layout = struct_info_ref(node).and_then(|sref| parse_struct_info(nodes, sref, 0));
        let strs = role_strings(node);
        let Some(first) = fc_promoted_role(&strs, promote_fc_buffers) else {
            continue;
        };
        let role = match first {
            "buffer" | "indirect_buffer" => {
                if first == "indirect_buffer" {
                    if let Some(sref) = struct_info_ref(node) {
                        // Key by the buffer's `air.location_index` (the Metal `[[buffer(N)]]` slot the
                        // harness binds), NOT the AIR argument position — they differ (e.g. arg 2 but
                        // buffer(0)). The oracle/runner both index buffers by location.
                        indirect_buffer_struct_refs.push((location_index(node, idx), sref));
                    }
                }
                if let Some(t) = layout.clone() {
                    buffer_layouts.insert(idx, t);
                }
                buffer_address_spaces.insert(
                    idx,
                    address_space(node)
                        .or_else(|| param_address_spaces.get(&idx).copied())
                        .unwrap_or(1),
                );
                if let Some(name) = arg_type_name(node) {
                    buffer_type_names.insert(idx, name);
                }
                if let Some(size) = i32_after_marker(node, "air.arg_type_size")
                    .or_else(|| i32_after_marker(node, "air.buffer_size"))
                {
                    buffer_type_sizes.insert(idx, size);
                }
                KernRole::Buffer(location_index(node, idx))
            }
            "texture" => {
                if let Some(name) = arg_type_name(node) {
                    texture_type_names.insert(idx, name);
                }
                let loc = location_index(node, idx);
                top_level_texture_locations.push(loc);
                KernRole::Texture(loc)
            }
            "instance_acceleration_structure" | "primitive_acceleration_structure"
                if body_uses_acceleration_structure_shadow(ll) =>
            {
                KernRole::AccelerationStructureShadow(location_index(node, idx))
            }
            "imageblock" => {
                if let Some(t) = layout {
                    imageblock_layouts.insert(idx, t);
                }
                KernRole::Other
            }
            "sampler" => KernRole::Sampler(location_index(node, idx)),
            "threads_per_threadgroup" => KernRole::ThreadsPerThreadgroup,
            "thread_position_in_threadgroup" => KernRole::ThreadPositionInThreadgroup,
            "threadgroups_per_grid" => KernRole::ThreadgroupsPerGrid,
            "threads_per_grid" => KernRole::ThreadsPerGrid,
            "threadgroup_position_in_grid" => KernRole::ThreadgroupPositionInGrid,
            "thread_index_in_threadgroup" => KernRole::ThreadIndexInThreadgroup,
            "thread_index_in_quadgroup" => KernRole::ThreadIndexInQuadgroup,
            "quadgroup_index_in_threadgroup" => KernRole::QuadgroupIndexInThreadgroup,
            "thread_index_in_simdgroup" => KernRole::ThreadIndexInSimdgroup,
            "simdgroup_index_in_threadgroup" => KernRole::SimdgroupIndexInThreadgroup,
            "threads_per_simdgroup" => KernRole::ThreadsPerSimdgroup,
            "simdgroups_per_threadgroup" => KernRole::SimdgroupsPerThreadgroup,
            "thread_position_in_grid" => KernRole::ThreadPositionInGrid,
            _ => KernRole::Other,
        };
        roles.push((idx, role));
    }
    // Detect argument-buffer-embedded textures that the body reads via an integer-coord
    // `air.read_texture` (an exact texel fetch). Gated purely on AIR structure/semantics — the
    // `air.indirect_argument` → `air.texture` marker chain and the `air.read_texture` intrinsic — so
    // it cannot key on any shader name. The body must actually READ (not merely sample/store) an
    // embedded texture for us to surface it; a `sample`/other use is left untouched.
    let embedded_textures = if body_uses_read_texture(ll) {
        detect_embedded_textures(
            nodes,
            &indirect_buffer_struct_refs,
            &top_level_texture_locations,
        )
    } else {
        vec![]
    };
    Some(KernMeta {
        roles,
        buffer_layouts,
        imageblock_layouts,
        buffer_address_spaces,
        buffer_type_sizes,
        buffer_type_names,
        texture_type_names,
        embedded_textures,
    })
}

fn body_uses_acceleration_structure_shadow(ll: &str) -> bool {
    ll.contains("@air.get_instance_count_instance_acceleration_structure")
        || ll.contains("@air.get_primitive_acceleration_structure_instance_acceleration_structure")
}

/// Collect every `!N = !{...}` metadata node body, keyed by N. Shared by both stage parsers.
fn collect_nodes(ll: &str) -> HashMap<u32, String> {
    #[cfg(test)]
    AIR_META_PARSE_COUNT.with(|count| count.set(count.get() + 1));
    let mut nodes = HashMap::new();
    for l in ll.lines() {
        let l = l.trim();
        let Some(rest) = l.strip_prefix('!') else {
            continue;
        };
        // expect "<digits> = !{<body>}"
        let Some(eq) = rest.find(" = !{") else {
            continue;
        };
        let Ok(id) = rest[..eq].parse::<u32>() else {
            continue;
        };
        let body = &rest[eq + " = !{".len()..];
        let body = body.strip_suffix('}').unwrap_or(body);
        nodes.insert(id, body.to_string());
    }
    nodes
}

#[cfg(test)]
thread_local! {
    static AIR_META_PARSE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_air_meta_parse_count() {
    AIR_META_PARSE_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn air_meta_parse_count() -> usize {
    AIR_META_PARSE_COUNT.with(std::cell::Cell::get)
}

/// The entry function NAME from `!air.<stage> = !{!N}; !N = !{ptr @<name>, ...}`. The Vulkan backend
/// does NOT inline helpers, so a module has many functions; this names the real entry.
pub fn entry_name(ll: &str, stage: &str) -> Option<String> {
    let nodes = collect_nodes(ll);
    entry_name_from_nodes(ll, stage, &nodes)
}

fn entry_name_from_nodes(ll: &str, stage: &str, nodes: &HashMap<u32, String>) -> Option<String> {
    let root = stage_root(ll, stage)?;
    let body = nodes.get(&root)?;
    // body like: "ptr @BlurComposite, !16, !18"
    let at = body.find('@')?;
    let after = &body[at + 1..];
    let name: String = after
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '.' || *c == '$')
        .collect();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

fn function_param_pointer_address_spaces(ll: &str, name: &str) -> Option<HashMap<u32, u32>> {
    let params = function_param_list(ll, name)?;
    let mut out = HashMap::new();
    for (idx, param) in split_top_level_commas(&params).into_iter().enumerate() {
        let param = param.trim_start();
        if !param.starts_with("ptr") {
            continue;
        }
        if let Some(addrspace) = param.find("addrspace(").and_then(|pos| {
            let after = &param[pos + "addrspace(".len()..];
            let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            digits.parse::<u32>().ok()
        }) {
            out.insert(idx as u32, addrspace);
        }
    }
    Some(out)
}

fn function_param_list(ll: &str, name: &str) -> Option<String> {
    let needle = format!("@{name}(");
    let start = ll.find(&needle)? + needle.len();
    let mut depth = 1u32;
    let mut end = start;
    for (off, ch) in ll[start..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    end = start + off;
                    break;
                }
            }
            _ => {}
        }
    }
    (depth == 0).then(|| ll[start..end].to_string())
}

fn split_top_level_commas(s: &str) -> Vec<String> {
    let mut items = Vec::new();
    let mut start = 0usize;
    let mut angle_depth = 0i32;
    let mut paren_depth = 0i32;
    let mut brace_depth = 0i32;
    for (idx, ch) in s.char_indices() {
        match ch {
            '<' => angle_depth += 1,
            '>' => angle_depth -= 1,
            '(' => paren_depth += 1,
            ')' => paren_depth -= 1,
            '{' => brace_depth += 1,
            '}' => brace_depth -= 1,
            ',' if angle_depth == 0 && paren_depth == 0 && brace_depth == 0 => {
                items.push(s[start..idx].trim().to_string());
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }
    let tail = s[start..].trim();
    if !tail.is_empty() {
        items.push(tail.to_string());
    }
    items
}

/// Find the single operand of `!air.<stage> = !{!N}` and return N.
fn stage_root(ll: &str, stage: &str) -> Option<u32> {
    let needle = format!("!air.{stage} = !{{!");
    for l in ll.lines() {
        let l = l.trim();
        if let Some(rest) = l.strip_prefix(&needle) {
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            return digits.parse().ok();
        }
    }
    None
}

/// The metadata refs (`!N`) appearing in a node body, in order. Skips `ptr @func` operands.
fn refs_in(body: &str) -> Vec<u32> {
    body.split(',')
        .filter_map(|s| {
            s.trim()
                .strip_prefix('!')
                .and_then(|x| x.parse::<u32>().ok())
        })
        .collect()
}

/// The `i32 N` immediate (parameter index) inside a metadata node body, if present.
fn first_i32(body: &str) -> Option<u32> {
    let mut it = body.split_whitespace().peekable();
    while let Some(tok) = it.next() {
        if tok == "i32" {
            if let Some(n) = it.peek() {
                let n = n.trim_end_matches(',');
                if let Ok(v) = n.parse::<u32>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn i32_after_marker(body: &str, marker: &str) -> Option<u32> {
    let marker = format!("!\"{marker}\"");
    let pos = body.find(&marker)?;
    first_i32(&body[pos + marker.len()..])
}

fn location_index(body: &str, fallback: u32) -> u32 {
    i32_after_marker(body, "air.location_index").unwrap_or(fallback)
}

fn address_space(body: &str) -> Option<u32> {
    i32_after_marker(body, "air.address_space")
}

/// The first `air.<role>` string literal in a node body (e.g. `air.texture`).
fn role_strings(body: &str) -> Vec<String> {
    let mut out = vec![];
    let mut rest = body;
    while let Some(p) = rest.find("!\"air.") {
        let after = &rest[p + 6..];
        let end = after.find('"').unwrap_or(after.len());
        out.push(after[..end].to_string());
        rest = &after[end..];
    }
    out
}

/// The primary arg-kind role string for an argument node, looking past a leading
/// `air.function_constant` wrapper. A conditionally-present (function-constant-gated) resource
/// emits `!"air.function_constant", !REF` BEFORE its real `!"air.<role>"` marker, so a naive
/// "first role string" sees `function_constant` and mis-classifies the argument as `Other`. The
/// real role is the first marker that isn't the wrapper. `air.function_constant` is a stable AIR
/// metadata-ABI symbol, not a shader identifier.
fn primary_role(strs: &[String]) -> Option<&str> {
    strs.iter()
        .map(String::as_str)
        .find(|s| *s != "function_constant")
}

/// Like `primary_role`, but only looks past the `air.function_constant` wrapper when the wrapped
/// resource is a TEXTURE, IMAGEBLOCK, or BUFFER (the latter only when `promote_buffers` is set).
/// An imageblock has no descriptor binding to promote; recognizing its stable ABI marker merely keeps
/// its metadata-described cell type available to the native emitter. Promoting a wrapped texture is
/// additive (it gains its own sampled/storage image binding). Promoting a wrapped BUFFER binds it as a
/// REAL StorageBuffer instead of the "possibly-absent → Private zero placeholder" default; on the
/// DEFAULT path this regresses byte-conformant goldens (a genuinely-absent fc buffer must stay Private,
/// and the conditionally-present binding emits an invalid `ArrayStride`-decorated array-of-Block), so
/// `promote_buffers` is FALSE on every default parse. It is set only by the adopt-if-validates
/// `fc_promote_psb` retry (an FC-multiplexed kernel whose live dtype variant's buffers ARE present and
/// hold real data — demoting them to Private zeros makes the cross-binding pointer merge read zeros,
/// byte-wrong; keeping them real StorageBuffer lets the FC prune + PSB lower the merge byte-correctly).
/// Dispatching on stable AIR ABI markers, not shader names.
fn fc_promoted_role(strs: &[String], promote_buffers: bool) -> Option<&str> {
    let first = strs.first().map(String::as_str)?;
    if first == "function_constant" {
        return Some(match primary_role(strs) {
            Some("texture") => "texture",
            Some("imageblock") => "imageblock",
            Some("buffer") if promote_buffers => "buffer",
            _ => first,
        });
    }
    Some(first)
}

fn string_after_marker(body: &str, marker: &str) -> Option<String> {
    let marker = format!("!\"{marker}\"");
    let pos = body.find(&marker)?;
    let after = &body[pos + marker.len()..];
    let pos = after.find("!\"")?;
    let value = &after[pos + 2..];
    let end = value.find('"')?;
    Some(value[..end].to_string())
}

fn arg_type_name(body: &str) -> Option<String> {
    string_after_marker(body, "air.arg_type_name")
}

fn arg_name(body: &str) -> Option<String> {
    string_after_marker(body, "air.arg_name")
}

/// Parse `!air.fragment` into a `FragMeta`. Structure (RE'd Phase 5.11):
///   `!air.fragment = !{!N}`; `!N = !{ptr @func, !OUT, !IN}`;
///   `!IN = !{!a, !b, ...}` each `!{i32 idx, !"air.<role>", ...}`; `!OUT = list of render targets`.
pub fn parse_air_fragment_meta(ll: &str) -> Option<FragMeta> {
    parse_air_fragment_meta_with_entry(ll).0
}

pub(crate) fn parse_air_fragment_meta_with_entry(ll: &str) -> (Option<FragMeta>, Option<String>) {
    let nodes = collect_nodes(ll);
    let entry = entry_name_from_nodes(ll, "fragment", &nodes);
    let meta = parse_air_fragment_meta_with_nodes(ll, &nodes, entry.as_deref());
    (meta, entry)
}

fn parse_air_fragment_meta_with_nodes(
    ll: &str,
    nodes: &HashMap<u32, String>,
    entry: Option<&str>,
) -> Option<FragMeta> {
    let static_int_globals = static_init_int_global_values(ll);
    let root = stage_root(ll, "fragment")?;
    let rootc = nodes.get(&root)?;
    let refs = refs_in(rootc);
    let (out_ref, in_ref) = (*refs.first()?, *refs.get(1)?);
    let render_target_members: Vec<(u32, u32)> = nodes
        .get(&out_ref)
        .map(|c| {
            refs_in(c)
                .into_iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    let node = nodes.get(&r)?;
                    let is_render_target = role_strings(node)
                        .first()
                        .map(|role| role == "render_target")
                        .unwrap_or(false);
                    is_render_target.then(|| (i as u32, first_i32(node).unwrap_or(i as u32)))
                })
                .collect()
        })
        .unwrap_or_default();
    let depth_members: Vec<u32> = nodes
        .get(&out_ref)
        .map(|c| {
            refs_in(c)
                .into_iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    let node = nodes.get(&r)?;
                    let is_depth = role_strings(node)
                        .first()
                        .map(|role| role == "depth")
                        .unwrap_or(false);
                    is_depth.then_some(i as u32)
                })
                .collect()
        })
        .unwrap_or_default();
    let stencil_members: Vec<u32> = nodes
        .get(&out_ref)
        .map(|c| {
            refs_in(c)
                .into_iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    let node = nodes.get(&r)?;
                    let is_stencil = role_strings(node)
                        .first()
                        .map(|role| role == "stencil")
                        .unwrap_or(false);
                    is_stencil.then_some(i as u32)
                })
                .collect()
        })
        .unwrap_or_default();
    let render_target_indices: Vec<u32> = render_target_members
        .iter()
        .map(|(_, location)| *location)
        .collect();
    let n_render_targets = render_target_indices.len() as u32;
    let render_target_type_names: HashMap<u32, String> = nodes
        .get(&out_ref)
        .map(|c| {
            refs_in(c)
                .into_iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    let node = nodes.get(&r)?;
                    let is_render_target = role_strings(node)
                        .first()
                        .map(|role| role == "render_target")
                        .unwrap_or(false);
                    if is_render_target {
                        arg_type_name(node).map(|name| (i as u32, name))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let mut roles = vec![];
    let mut buffer_layouts = HashMap::new();
    let mut varying_types = HashMap::new();
    let mut varying_names = HashMap::new();
    let mut varying_user_semantics = HashMap::new();
    let mut texture_type_names = HashMap::new();
    let mut varying_loc = 0u32;
    let mut color_input_idx = 0u32;
    let mut buffer_address_spaces = HashMap::new();
    let mut buffer_type_sizes = HashMap::new();
    // AIR function-param pointer address spaces for the fragment entry, the fallback the kernel
    // parser uses when a buffer arg node omits `air.address_space`.
    let param_address_spaces = entry
        .and_then(|name| function_param_pointer_address_spaces(ll, name))
        .unwrap_or_default();
    for r in refs_in(nodes.get(&in_ref)?) {
        let Some(node) = nodes.get(&r) else { continue };
        let Some(idx) = first_i32(node) else { continue };
        // Reconstruct any buffer struct layout (used when the backend collapsed the buffer pointer).
        if let Some(sref) = struct_info_ref(node) {
            if let Some(t) = parse_struct_info(nodes, sref, 0) {
                buffer_layouts.insert(idx, t);
            }
        }
        let strs = role_strings(node);
        let Some(role_str) = primary_role(&strs) else {
            continue;
        };
        let role = match role_str {
            "position" => FragRole::Position,
            "fragment_input" => {
                let l = varying_loc;
                varying_loc += 1;
                if let Some(name) = arg_type_name(node) {
                    varying_types.insert(l, name);
                }
                if let Some(name) = arg_name(node) {
                    varying_names.insert(l, name);
                }
                if let Some(semantic) = string_after_marker(node, "air.fragment_input") {
                    varying_user_semantics.insert(l, semantic);
                }
                FragRole::Varying(l)
            }
            "texture" => {
                if let Some(name) = arg_type_name(node) {
                    texture_type_names.insert(idx, name);
                }
                FragRole::Texture(location_index_with_static(node, idx, &static_int_globals))
            }
            "sampler" => {
                FragRole::Sampler(location_index_with_static(node, idx, &static_int_globals))
            }
            "buffer" | "indirect_buffer" => {
                // Populate address space / declared size ONLY when the IR actually carries them
                // (no invented default), so reflection never reports a guessed value.
                if let Some(space) =
                    address_space(node).or_else(|| param_address_spaces.get(&idx).copied())
                {
                    buffer_address_spaces.insert(idx, space);
                }
                if let Some(size) = i32_after_marker(node, "air.arg_type_size")
                    .or_else(|| i32_after_marker(node, "air.buffer_size"))
                {
                    buffer_type_sizes.insert(idx, size);
                }
                FragRole::Buffer(location_index_with_static(node, idx, &static_int_globals))
            }
            // an air.render_target in the INPUT list = framebuffer fetch ([[color(n)]]).
            "render_target" => {
                let ci = color_input_idx;
                color_input_idx += 1;
                FragRole::ColorInput(ci)
            }
            _ => FragRole::Other,
        };
        roles.push((idx, role));
    }
    Some(FragMeta {
        roles,
        varying_types,
        varying_names,
        varying_user_semantics,
        n_render_targets,
        render_target_members,
        render_target_type_names,
        depth_members,
        stencil_members,
        render_target_indices,
        buffer_layouts,
        buffer_address_spaces,
        buffer_type_sizes,
        texture_type_names,
    })
}

/// Parse `!air.vertex` into a `VertMeta`. `!air.vertex = !{!N}`; `!N = !{ptr @func, !OUT, !IN}`;
/// `!IN` entries are `!{i32 idx, !"air.vertex_input"|"air.buffer", ...}`.
pub fn parse_air_vertex_meta(ll: &str) -> Option<VertMeta> {
    parse_air_vertex_meta_with_entry(ll).0
}

pub(crate) fn parse_air_vertex_meta_with_entry(ll: &str) -> (Option<VertMeta>, Option<String>) {
    let nodes = collect_nodes(ll);
    let entry = entry_name_from_nodes(ll, "vertex", &nodes);
    let meta = parse_air_vertex_meta_with_nodes(ll, &nodes, entry.as_deref());
    (meta, entry)
}

fn parse_air_vertex_meta_with_nodes(
    ll: &str,
    nodes: &HashMap<u32, String>,
    entry: Option<&str>,
) -> Option<VertMeta> {
    let static_int_globals = static_init_int_global_values(ll);
    let root = stage_root(ll, "vertex")?;
    let rootc = nodes.get(&root)?;
    let refs = refs_in(rootc);
    let out_ref = *refs.first()?;
    let in_ref = *refs.get(1)?;

    let mut output_roles = vec![];
    let mut out_loc = 0u32;
    for r in refs_in(nodes.get(&out_ref)?) {
        let Some(node) = nodes.get(&r) else { continue };
        let strs = role_strings(node);
        let Some(first) = fc_promoted_role(&strs, false) else {
            continue;
        };
        let role = match first {
            "position" => VertOutRole::Position,
            "point_size" => VertOutRole::PointSize,
            "viewport_array_index" => VertOutRole::ViewportArrayIndex,
            "vertex_output" => {
                let l = location_index_with_static(node, out_loc, &static_int_globals);
                out_loc += 1;
                VertOutRole::Varying(l)
            }
            _ => VertOutRole::Other,
        };
        output_roles.push(role);
    }

    let mut roles = vec![];
    let mut vertex_input_types = HashMap::new();
    let mut vertex_input_names = HashMap::new();
    let mut buffer_layouts = HashMap::new();
    let mut buffer_address_spaces = HashMap::new();
    let mut buffer_type_sizes = HashMap::new();
    let mut texture_type_names = HashMap::new();
    let param_address_spaces = entry
        .and_then(|name| function_param_pointer_address_spaces(ll, name))
        .unwrap_or_default();
    let mut vin_loc = 0u32;
    for r in refs_in(nodes.get(&in_ref)?) {
        let Some(node) = nodes.get(&r) else { continue };
        let Some(idx) = first_i32(node) else { continue };
        if let Some(sref) = struct_info_ref(node) {
            if let Some(t) = parse_struct_info(nodes, sref, 0) {
                buffer_layouts.insert(idx, t);
            }
        }
        let strs = role_strings(node);
        let Some(first) = fc_promoted_role(&strs, false) else {
            continue;
        };
        let role = match first {
            "vertex_input" => {
                let l = location_index_with_static(node, vin_loc, &static_int_globals);
                vin_loc += 1;
                if let Some(name) = arg_type_name(node) {
                    vertex_input_types.insert(l, name);
                }
                if let Some(name) = arg_name(node) {
                    vertex_input_names.insert(l, name);
                }
                VertRole::VertexInput(l)
            }
            "buffer" | "indirect_buffer" => {
                if let Some(space) =
                    address_space(node).or_else(|| param_address_spaces.get(&idx).copied())
                {
                    buffer_address_spaces.insert(idx, space);
                }
                if let Some(size) = i32_after_marker(node, "air.arg_type_size")
                    .or_else(|| i32_after_marker(node, "air.buffer_size"))
                {
                    buffer_type_sizes.insert(idx, size);
                }
                VertRole::Buffer(location_index_with_static(node, idx, &static_int_globals))
            }
            "texture" => {
                if let Some(name) = arg_type_name(node) {
                    texture_type_names.insert(idx, name);
                }
                VertRole::Texture(location_index_with_static(node, idx, &static_int_globals))
            }
            "sampler" => {
                VertRole::Sampler(location_index_with_static(node, idx, &static_int_globals))
            }
            "vertex_id" => VertRole::VertexId,
            "instance_id" => VertRole::InstanceId,
            _ => VertRole::Other,
        };
        roles.push((idx, role));
    }
    Some(VertMeta {
        roles,
        output_roles,
        vertex_input_types,
        vertex_input_names,
        buffer_layouts,
        buffer_address_spaces,
        buffer_type_sizes,
        texture_type_names,
    })
}

#[cfg(test)]
mod tests;
