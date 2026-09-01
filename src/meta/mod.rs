//! AIR stage-interface metadata parsed from sanitized `.ll` before SPIR-V emission. The parser maps
//! each entry-function parameter to the role that the interface pass uses to synthesize Vulkan
//! bindings, stage inputs, and stage outputs.

use std::collections::HashMap;

mod embedded;
mod function_constants;
mod globals;
mod intersections;
mod textures;
mod types;
use embedded::{body_uses_texture_intrinsic, detect_embedded_arguments, detect_embedded_textures};
pub use embedded::{embedded_synthetic_texture_index, EmbeddedArgument, EmbeddedTexture};
pub use function_constants::{parse_function_constants, FunctionConstant};
use globals::{location_index_with_static, static_init_int_global_values};
pub(crate) use globals::{static_init_foldable_global_values, StaticIntValue};
pub use intersections::{
    AirIntersectionFamily, AirIntersectionInstancing, AirIntersectionResultField,
};
pub use textures::{
    texture_shape_from_name, TextureComponent, TextureDimension, TextureFormat, TextureShape,
    TEXTURE_HANDLE_ARRAY_DESCRIPTOR_COUNT,
};
pub(crate) use types::storage_air_type_for_size;
use types::{parse_struct_info, struct_info_ref, tokenize, Tok};
pub use types::{primitive_air_type_from_name, AirMember, AirScalar, AirType};

/// Whether stable AIR argument metadata describes a runtime array of device-buffer addresses.
pub fn is_device_buffer_array_type_name(name: &str) -> bool {
    name.chars()
        .filter(|character| !character.is_whitespace())
        .eq("array_ref<void>".chars())
}

/// Role of a single fragment-shader entry parameter, keyed by its parameter index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FragRole {
    /// `[[position]]` -> Input BuiltIn FragCoord (often unused).
    Position,
    /// `[[point_coord]]` -> Input BuiltIn PointCoord.
    PointCoord,
    /// `[[front_facing]]` -> Input BuiltIn FrontFacing.
    FrontFacing,
    /// `[[barycentric_coord]]` -> Input BuiltIn `BaryCoordKHR` / `BaryCoordNoPerspKHR` (float3).
    ///
    /// AIR states the perspective axis on the same node, exactly as it does for a varying, and
    /// SPIR-V spells the two as different builtins rather than as a decoration — so the flag has to
    /// travel with the role.
    BarycentricCoord { no_perspective: bool },
    /// `[[primitive_id]]` -> Input BuiltIn PrimitiveId (32-bit uint).
    PrimitiveId,
    /// `[[sample_id]]` -> Input BuiltIn SampleId (32-bit uint).
    SampleId,
    /// `[[sample_mask]]` on an *argument* -> Input BuiltIn `SampleMask`, the coverage the
    /// rasterizer produced for this fragment. AIR spells the input form `air.sample_mask_in` to
    /// distinguish it from the return member; SPIR-V uses one builtin in two storage classes.
    SampleMaskIn,
    /// `[[viewport_array_index]]` -> Input BuiltIn ViewportIndex (32-bit uint).
    ViewportArrayIndex,
    /// `[[render_target_array_index]]` -> Input BuiltIn Layer (32-bit uint).
    RenderTargetArrayIndex,
    /// `[[stage_in]]` interpolated input -> Input var at Location N (N = order among fragment_inputs).
    Varying(u32),
    /// `[[texture(n)]]` -> UniformConstant sampled image.
    Texture(u32),
    /// `[[sampler(n)]]` -> UniformConstant sampler.
    Sampler(u32),
    /// A Metal visible-function table resolved during authored dependency linking.
    VisibleFunctionTable(u32),
    /// A Metal intersection-function table resolved during authored dependency linking.
    IntersectionFunctionTable(u32),
    /// `[[buffer(n)]]` -> Uniform/StorageBuffer block.
    Buffer(u32),
    /// `[[color(n)]]` framebuffer-fetch input -> Vulkan input attachment (out of scope this milestone).
    ColorInput(u32),
    /// A custom fragment `[[imageblock_data]]` projection. Its fields are mapped to the master
    /// imageblock layout by AIR user semantic, never by the source struct or argument name.
    ImageblockData,
    /// Anything we don't model.
    Other,
}

/// Declared AIR access on a buffer argument. This is a conservative contract: it may be broader
/// than the specialized body actually uses, but it never understates reads or writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

/// Where inside a pixel AIR asked a `fragment_input` varying to be sampled.
///
/// Metal spells this as the first half of an interpolation attribute — `[[center_perspective]]`,
/// `[[centroid_no_perspective]]`, `[[sample_perspective]]` — and AIR emits it as its own marker
/// alongside the perspective marker. `air.center` is the default and needs no SPIR-V decoration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VaryingSampling {
    /// `air.center` — sampled at the pixel center. Vulkan's default; no decoration.
    #[default]
    Center,
    /// `air.centroid` — sampled inside the covered area of the primitive (`Centroid`).
    Centroid,
    /// `air.sample` — sampled per covered sample, which forces per-sample shading (`Sample`,
    /// capability `SampleRateShading`).
    Sample,
}

/// How AIR asked a `fragment_input` varying to be interpolated.
///
/// One record per varying rather than one set per qualifier: AIR states the whole interpolation
/// attribute on the argument node, and reading only the part the emitter happened to support is
/// how `air.no_perspective` and `air.centroid` were silently dropped for as long as only
/// `air.flat` was decoded. `interpolation_markers_are_decoded_or_deliberately_ignored` in
/// `src/meta/tests.rs` pins the full marker inventory against this record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaryingInterpolation {
    /// `air.flat` — not interpolated at all. AIR states this instead of, not alongside, the
    /// perspective/sampling pair, so it takes precedence over both fields below.
    pub flat: bool,
    /// `air.no_perspective` — interpolated linearly in screen space rather than
    /// perspective-correct (`NoPerspective`). Its complement `air.perspective` is Vulkan's
    /// default and needs no decoration.
    pub no_perspective: bool,
    /// Where in the pixel the interpolated value is taken from.
    pub sampling: VaryingSampling,
}

impl VaryingInterpolation {
    /// Decode the interpolation attribute from an argument node's `air.*` marker list.
    fn from_role_strings(strs: &[String]) -> Self {
        let has = |marker: &str| strs.iter().any(|s| s == marker);
        Self {
            flat: has("flat"),
            no_perspective: has("no_perspective"),
            sampling: if has("centroid") {
                VaryingSampling::Centroid
            } else if has("sample") {
                VaryingSampling::Sample
            } else {
                VaryingSampling::Center
            },
        }
    }
}

/// A fragment shader's decoded parameter roles + render-target count/indices.
#[derive(Clone, Debug, Default)]
pub struct FragMeta {
    /// `(param_idx, role)` — one per fragment input, in declaration order.
    pub roles: Vec<(u32, FragRole)>,
    /// `(parameter index, AIR role)` for every enabled entry parameter whose role has no
    /// lowering.
    ///
    /// An unrecognised parameter is bound to a zero value so the body stays well formed. That is
    /// right for a function-constant-disabled resource, which Metal itself defines as absent, and
    /// wrong for anything else: a `[[barycentric_coord]]` the emitter does not model becomes a
    /// silent zero, and everything computed from it is wrong in a module that validates. Emission
    /// rejects on a non-empty list instead.
    pub unmodelled_input_params: Vec<(u32, String)>,
    /// Every attribute on the `!air.fragment` root this stage has no model for, as it reads in
    /// AIR. See [`FragMeta::early_fragment_tests`]: the root's attribute tail changes what the
    /// stage does, so an entry carrying an attribute nothing consumes must be refused rather than
    /// emitted as if the root had carried nothing.
    pub unmodelled_stage_attributes: Vec<String>,
    /// `[[early_fragment_tests]]`: depth and stencil testing happens before the fragment body.
    ///
    /// This is not an optimisation hint. A fragment the depth test rejects runs no part of the
    /// body, so none of its buffer, texture or imageblock stores happen; under the default late
    /// test the same shader performs every store and only its color output is discarded.
    pub early_fragment_tests: bool,
    /// Descriptor-backed render-target planes used by implicit imageblock load/store intrinsics.
    /// Detected from the module's intrinsic calls, which is a property of the body rather than of
    /// the stage — the interface pass materializes the plane wherever it lowers one of those calls,
    /// so every stage that can carry them has to be able to report them.
    pub implicit_imageblock_attachments: Vec<ImplicitImageblockAttachment>,
    /// `fragment_input` Location -> AIR type name (`float2`, `float4`, ...). Used by passthrough
    /// vertex synthesis when the pipeline binds a built-in vertex slot.
    pub varying_types: HashMap<u32, String>,
    /// `fragment_input` Location -> Metal field/argument name, when AIR metadata carries one. The
    /// Metal oracle uses this to generate a vertex struct that Apple's pipeline linker matches.
    pub varying_names: HashMap<u32, String>,
    /// `fragment_input` Location -> Metal user semantic, such as `user(texturecoord)`, when AIR
    /// metadata carries one.
    pub varying_user_semantics: HashMap<u32, String>,
    /// `fragment_input` Location -> the interpolation attribute AIR declared for it. Absent means
    /// AIR said nothing, which is Vulkan's default (perspective-correct, pixel center).
    pub varying_interpolation: HashMap<u32, VaryingInterpolation>,
    /// number of `air.render_target` outputs (MRT count; 1 for the common single-output case).
    pub n_render_targets: u32,
    /// Return-struct member index -> color attachment Location for actual `air.render_target`
    /// outputs. Non-color outputs such as `air.stencil` are deliberately absent.
    pub render_target_members: Vec<(u32, u32)>,
    /// Return-struct member index -> AIR render-target type name (`float4`, `int4`, ...).
    pub render_target_type_names: HashMap<u32, String>,
    /// Return-struct member indices tagged as `air.depth` (`[[depth(...)]]`).
    pub depth_members: Vec<u32>,
    /// Conservative depth-test relation declared by `[[depth(...)]]`.
    pub depth_qualifier: Option<DepthQualifier>,
    /// Return-struct member indices tagged as `air.stencil` (`[[stencil]]`).
    pub stencil_members: Vec<u32>,
    /// Return-struct member indices tagged as `air.sample_mask` (`[[sample_mask]]`).
    ///
    /// The value is a coverage mask: a sample whose bit the shader clears is not written, which is
    /// how alpha-to-coverage and custom MSAA resolves are expressed. Vulkan spells it as the
    /// `SampleMask` builtin, an array of `uint` rather than the scalar Metal returns.
    pub sample_mask_members: Vec<u32>,
    /// `(member index, AIR role)` for every enabled return member whose role the emitter has no
    /// lowering for. See [`FRAGMENT_OUTPUT_ROLES`].
    pub unmodelled_output_members: Vec<(u32, String)>,
    /// Custom per-pixel fragment imageblock master plus the input/output projections that expose
    /// subsets of its fields. `None` when the fragment carries no `air.imageblock_master` contract.
    pub fragment_imageblock: Option<FragmentImageblock>,
    /// Render-target locations in AIR output metadata order. A single-output fragment can legally
    /// write a nonzero MRT slot, e.g. coverage shaders writing `[[color(1)]]`.
    pub render_target_indices: Vec<u32>,
    /// `param_idx -> reconstructed struct layout` for buffer args that carry `air.struct_type_info`.
    /// Used to rebuild the real struct when native emission represents the buffer as a bare pointer.
    pub buffer_layouts: HashMap<u32, AirType>,
    /// `param_idx -> AIR address space` for buffer args, when the AIR node carries it (device=1,
    /// constant=2). Populated only from `air.address_space` / the function param pointer address
    /// space — absent (not guessed) when the IR does not state it. Mirrors
    /// [`KernMeta::buffer_address_spaces`] for the fragment stage.
    pub buffer_address_spaces: HashMap<u32, u32>,
    /// `param_idx -> declared AIR buffer byte size` (`air.arg_type_size` / `air.buffer_size`), when
    /// the AIR node carries it. Mirrors [`KernMeta::buffer_type_sizes`] for the fragment stage.
    pub buffer_type_sizes: HashMap<u32, u32>,
    /// `param_idx -> air.buffer_size` for reference-like arguments whose metadata bounds the
    /// reachable object to exactly that many bytes. Kept separate from `buffer_type_sizes`, which
    /// may only be an unbounded pointer's element/pointee size.
    pub buffer_object_sizes: HashMap<u32, u32>,
    /// `param_idx -> AIR buffer argument type name` for every stage.
    pub buffer_type_names: HashMap<u32, String>,
    /// `param_idx -> declared AIR read/write qualifier`.
    pub buffer_accesses: HashMap<u32, BufferAccess>,
    /// `param_idx -> AIR texture argument type name`, e.g. `texture2d<uint, read>`.
    pub texture_type_names: HashMap<u32, String>,
    /// `param_idx -> the descriptor count `air.location_index` states, when it states more than one.
    ///
    /// A texture or sampler argument above `1` is a handle ARRAY occupying that many descriptors at
    /// its Metal slot. Absent for the ordinary single-descriptor argument and for a count spelled as
    /// a function-constant global.
    pub declared_descriptor_counts: HashMap<u32, u32>,
    /// Framebuffer-fetch color input Location -> AIR render-target type name, e.g. `float4`.
    pub color_input_type_names: HashMap<u32, String>,
    pub embedded_textures: Vec<EmbeddedTexture>,
    pub embedded_arguments: Vec<EmbeddedArgument>,
}

/// One field in AIR's custom fragment-imageblock master layout.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FragmentImageblockMember {
    pub offset: u32,
    pub size: u32,
    pub type_name: String,
    pub semantic: String,
    pub raster_order_group: u32,
}

/// One field exposed by an entry input or return-value imageblock projection.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FragmentImageblockProjectionMember {
    pub projection_member: u32,
    pub master_member: u32,
}

/// A partial struct view of a custom fragment imageblock.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FragmentImageblockProjection {
    /// Entry parameter index for an input projection, or return-struct member index for an output.
    pub interface_index: u32,
    pub members: Vec<FragmentImageblockProjectionMember>,
}

/// AIR's exact custom fragment-imageblock ABI.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FragmentImageblock {
    pub sample_size: u32,
    pub members: Vec<FragmentImageblockMember>,
    pub inputs: Vec<FragmentImageblockProjection>,
    pub outputs: Vec<FragmentImageblockProjection>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DepthQualifier {
    Any,
    Less,
    Greater,
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
    /// See [`Self::declared_descriptor_counts`].
    pub fn declared_descriptor_count(&self, idx: u32) -> Option<u32> {
        self.declared_descriptor_counts.get(&idx).copied()
    }
    pub fn color_input_type_name(&self, location: u32) -> Option<&str> {
        self.color_input_type_names
            .get(&location)
            .map(String::as_str)
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
    /// The interpolation attribute AIR declared for the varying at `loc`, defaulting to Vulkan's
    /// own default when AIR declared none.
    pub fn varying_interpolation(&self, loc: u32) -> VaryingInterpolation {
        self.varying_interpolation
            .get(&loc)
            .copied()
            .unwrap_or_default()
    }
    pub fn varying_is_flat(&self, loc: u32) -> bool {
        self.varying_interpolation(loc).flat
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
    pub fn is_sample_mask_member(&self, member_idx: u32) -> bool {
        self.sample_mask_members.contains(&member_idx)
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
    /// A Metal visible-function table resolved during authored dependency linking.
    VisibleFunctionTable(u32),
    /// A Metal intersection-function table resolved during authored dependency linking.
    IntersectionFunctionTable(u32),
    /// `[[vertex_id]]` -> Input BuiltIn VertexIndex (32-bit uint).
    VertexId,
    /// `[[instance_id]]` -> Input BuiltIn InstanceIndex (32-bit uint).
    InstanceId,
    /// Opaque Metal patch handle consumed only by the metadata-named control-point accessor.
    PatchControlPoints,
    /// Per-patch user input at the AIR location.
    PatchInput(u32),
    /// `[[position_in_patch]]` -> the leading components of Vulkan `TessCoord`.
    PositionInPatch,
    /// `[[patch_id]]` -> Vulkan `PrimitiveId` in tessellation evaluation.
    PatchId,
    /// Metal vertex-amplification identifiers have no Vulkan tessellation builtin and are exposed
    /// through the translator's per-patch system-input locations.
    AmplificationId,
    AmplificationCount,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PatchDomain {
    Triangle,
    Quad,
    Isoline,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatchControlPointField {
    pub location: u32,
    pub type_name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TessellationMeta {
    pub domain: PatchDomain,
    pub control_point_count: u32,
    pub control_point_function: Option<String>,
    pub control_point_fields: Vec<PatchControlPointField>,
}

/// Role of a single vertex return-struct member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VertOutRole {
    /// `[[position]]` -> Output BuiltIn Position.
    Position,
    /// `[[point_size]]` -> Output BuiltIn PointSize. Does not consume a Location.
    PointSize,
    /// `[[clip_distance]]` -> Output BuiltIn ClipDistance. Does not consume a Location.
    ClipDistance,
    /// `[[viewport_array_index]]` -> Output BuiltIn ViewportIndex. Does not consume a Location.
    ViewportArrayIndex,
    /// `[[render_target_array_index]]` -> Output BuiltIn Layer. Does not consume a Location.
    RenderTargetArrayIndex,
    /// User varying output -> Output var at Location N.
    Varying(u32),
    /// Function-constant-gated output disabled by the translator's default-zero FC model.
    FunctionConstantDisabled,
    Other,
}

/// A vertex shader's decoded parameter roles. The OUTPUT struct (member 0 = `[[position]]`, the rest
/// follows `output_roles`; point-size-style builtins do not consume varying locations.
#[derive(Clone, Debug, Default)]
pub struct VertMeta {
    pub roles: Vec<(u32, VertRole)>,
    /// `(parameter index, AIR role)` for every enabled entry parameter whose role has no
    /// lowering.
    ///
    /// An unrecognised parameter is bound to a zero value so the body stays well formed. That is
    /// right for a function-constant-disabled resource, which Metal itself defines as absent, and
    /// wrong for anything else: a `[[barycentric_coord]]` the emitter does not model becomes a
    /// silent zero, and everything computed from it is wrong in a module that validates. Emission
    /// rejects on a non-empty list instead.
    pub unmodelled_input_params: Vec<(u32, String)>,
    /// Descriptor-backed render-target planes used by implicit imageblock load/store intrinsics.
    /// Detected from the module's intrinsic calls, which is a property of the body rather than of
    /// the stage — the interface pass materializes the plane wherever it lowers one of those calls,
    /// so every stage that can carry them has to be able to report them.
    pub implicit_imageblock_attachments: Vec<ImplicitImageblockAttachment>,
    /// Entry parameter index -> AIR type name. Tessellation system values use this to expose the
    /// exact cross-stage scalar type instead of forcing executors to infer it from a location.
    pub parameter_type_names: HashMap<u32, String>,
    pub output_roles: Vec<VertOutRole>,
    /// `(member index, AIR role)` for every output member decoded as [`VertOutRole::Other`].
    ///
    /// The vertex output walk gives an unrecognised member the next free user Location, so an
    /// unmodelled role does not vanish — it becomes a varying the pipeline never wired, at a
    /// Location it takes from a real one. Reporting it lets emission reject instead.
    pub unmodelled_output_members: Vec<(u32, String)>,
    /// Output member indices AIR marked `air.invariant` — Metal `[[position, invariant]]`.
    ///
    /// The guarantee is bit-exact: the same vertex fed through two pipelines that both declare it
    /// must produce the identical clip position, which is what lets a depth-prepass and the pass
    /// that tests against it agree instead of z-fighting. Vulkan spells it `OpDecorate … Invariant`.
    /// A translation that drops it stays valid and reflects identically, so nothing but this record
    /// carries the request across.
    pub invariant_outputs: Vec<u32>,
    /// User-varying output Location -> AIR type name.
    pub output_varying_types: HashMap<u32, String>,
    /// User-varying output Location -> Metal field name.
    pub output_varying_names: HashMap<u32, String>,
    /// User-varying output Location -> Metal linker semantic.
    pub output_varying_user_semantics: HashMap<u32, String>,
    /// Vertex input Location -> AIR type name (`float2`, `float4`, ...). Used by conformance
    /// oracles that must synthesize a Metal vertex descriptor before pipeline reflection exists.
    pub vertex_input_types: HashMap<u32, String>,
    /// Vertex input Location -> Metal argument name, when AIR metadata carries one.
    pub vertex_input_names: HashMap<u32, String>,
    /// Per-patch tessellation input Location -> AIR type name.
    pub patch_input_types: HashMap<u32, String>,
    /// Per-patch tessellation input Location -> Metal argument name.
    pub patch_input_names: HashMap<u32, String>,
    /// `param_idx -> reconstructed struct layout` for buffer args (see `FragMeta::buffer_layouts`).
    pub buffer_layouts: HashMap<u32, AirType>,
    /// `param_idx -> AIR address space` for buffer args, when the AIR carries it (see
    /// [`FragMeta::buffer_address_spaces`]).
    pub buffer_address_spaces: HashMap<u32, u32>,
    /// `param_idx -> declared AIR buffer byte size` for buffer args, when the AIR carries it (see
    /// [`FragMeta::buffer_type_sizes`]).
    pub buffer_type_sizes: HashMap<u32, u32>,
    pub buffer_object_sizes: HashMap<u32, u32>,
    pub buffer_type_names: HashMap<u32, String>,
    pub buffer_accesses: HashMap<u32, BufferAccess>,
    /// `param_idx -> AIR texture argument type name`, e.g. `texture2d<uint, read>`.
    pub texture_type_names: HashMap<u32, String>,
    /// `param_idx -> the descriptor count `air.location_index` states, when it states more than one.
    ///
    /// A texture or sampler argument above `1` is a handle ARRAY occupying that many descriptors at
    /// its Metal slot. Absent for the ordinary single-descriptor argument and for a count spelled as
    /// a function-constant global.
    pub declared_descriptor_counts: HashMap<u32, u32>,
    pub embedded_textures: Vec<EmbeddedTexture>,
    pub embedded_arguments: Vec<EmbeddedArgument>,
    pub tessellation: Option<TessellationMeta>,
    /// Why an `air.patch` node the function does carry could not be decoded into
    /// [`VertMeta::tessellation`], if that happened.
    ///
    /// The two are not interchangeable with `tessellation: None`. A vertex function with no patch
    /// node is an ordinary vertex shader; one whose patch node did not decode is a post-tessellation
    /// evaluation shader whose domain, spacing, winding and per-patch inputs would all go missing,
    /// and the module that results is valid, binds, reflects, and draws the wrong geometry. The
    /// stage-input pass refuses on this rather than emitting it.
    pub undecoded_patch_shape: Option<String>,
    /// Every attribute on the `!air.vertex` root this stage has no model for, as it reads in AIR.
    /// Mirrors [`FragMeta::unmodelled_stage_attributes`].
    pub unmodelled_stage_attributes: Vec<String>,
}

impl VertMeta {
    pub fn is_tessellation_evaluation(&self) -> bool {
        self.tessellation.is_some()
    }

    pub fn tessellation_system_input_location(&self, role: &VertRole) -> Option<u32> {
        let base = self
            .roles
            .iter()
            .filter_map(|(_, role)| match role {
                VertRole::PatchInput(location) => Some(*location),
                _ => None,
            })
            .chain(
                self.tessellation
                    .iter()
                    .flat_map(|meta| meta.control_point_fields.iter().map(|field| field.location)),
            )
            .max()
            .map_or(0, |location| location + 1);
        match role {
            VertRole::InstanceId => Some(base),
            VertRole::AmplificationId => Some(base + 1),
            VertRole::AmplificationCount => Some(base + 2),
            _ => None,
        }
    }
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
    /// See [`Self::declared_descriptor_counts`].
    pub fn declared_descriptor_count(&self, idx: u32) -> Option<u32> {
        self.declared_descriptor_counts.get(&idx).copied()
    }
    pub fn output_role_of(&self, idx: u32) -> Option<&VertOutRole> {
        self.output_roles.get(idx as usize)
    }
    /// Whether AIR marked output member `idx` `air.invariant`.
    pub fn output_is_invariant(&self, idx: u32) -> bool {
        self.invariant_outputs.contains(&idx)
    }
    pub fn output_varying_type(&self, loc: u32) -> Option<&str> {
        self.output_varying_types.get(&loc).map(String::as_str)
    }
    pub fn output_varying_name(&self, loc: u32) -> Option<&str> {
        self.output_varying_names.get(&loc).map(String::as_str)
    }
    pub fn output_varying_user_semantic(&self, loc: u32) -> Option<&str> {
        self.output_varying_user_semantics
            .get(&loc)
            .map(String::as_str)
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
    /// An AIR primitive acceleration structure that is not consumed by an AIR intersection
    /// intrinsic. Metal still binds the native object; Vulkan needs no descriptor.
    PrimitiveAccelerationStructure(u32),
    /// An AIR primitive acceleration structure consumed by AIR intersection lowering. Metal binds
    /// the native object and Vulkan exposes authored triangle geometry through a StorageBuffer.
    PrimitiveAccelerationStructureShadow(u32),
    /// A Metal visible-function table. Logical SPIR-V has no descriptor for the opaque table;
    /// authored linking resolves its entries before ordinary interface lowering.
    VisibleFunctionTable(u32),
    /// A Metal intersection-function table. Like visible tables, this is a link-time authored
    /// resource rather than a Vulkan descriptor.
    IntersectionFunctionTable(u32),
    /// `[[threads_per_threadgroup]]` or `[[dispatch_threads_per_threadgroup]]` (`uint` or `uint3`)
    /// -> the execution local size. Scalar params receive `64`; vector params receive `(64, 1, 1)`
    /// for the current harness.
    ///
    /// Metal distinguishes the two only under `dispatchThreads:`, where a final partial threadgroup
    /// reports a smaller `threads_per_threadgroup` than the size the dispatch asked for. Vulkan has
    /// no partial workgroups — `vkCmdDispatch` issues whole ones — so both AIR roles denote the same
    /// value here, and the declared local size is that value.
    ThreadsPerThreadgroup,
    /// `[[thread_position_in_threadgroup]]` (`uint` or `uint3`) -> LocalInvocationId.
    /// Scalar params receive component .x; vector params receive the full v3uint.
    ThreadPositionInThreadgroup,
    /// `[[threadgroups_per_grid]]` (`uint` or `uint3`) -> NumWorkgroups.
    /// Scalar params receive component .x; vector params receive the full v3uint.
    ThreadgroupsPerGrid,
    /// `[[threads_per_grid]]` (`uint` or `uint3`) -> the selected kernel dispatch grid. Whole-
    /// workgroup dispatches derive it as NumWorkgroups * LocalSize; exact-thread dispatches read
    /// the complete logical grid from the region payload.
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
    /// Kernel `[[stage_in]]` attribute data. Metal feeds this through a stage-input descriptor keyed
    /// by `air.location_index`; Vulkan lowering needs an explicit per-invocation data ABI and must
    /// not silently bind it to zero.
    StageInput(u32),
    Other,
}

/// A compute kernel's decoded parameter roles. The body has no return value (a `void` kernel that
/// writes through its buffer pointers), so no output handling is needed.
#[derive(Clone, Debug, Default)]
pub struct KernMeta {
    pub roles: Vec<(u32, KernRole)>,
    /// `(parameter index, AIR role)` for every enabled entry parameter whose role has no
    /// lowering. See [`FragMeta::unmodelled_input_params`].
    pub unmodelled_input_params: Vec<(u32, String)>,
    /// Parameter indices of explicit imageblocks whose AIR node carries
    /// `air.alias_implicit_imageblock`.
    ///
    /// An explicit imageblock is ordinarily tile-local scratch, and the emitter gives it Private
    /// storage. This marker says the opposite: the storage *is* the implicit imageblock -- the
    /// render targets the rasterizer already wrote -- so the kernel's first read is of framebuffer
    /// content, and its writes have to land back there. Private scratch is neither, so the marker
    /// cannot be dropped.
    pub aliased_implicit_imageblock_params: Vec<u32>,
    /// Every attribute on the `!air.kernel` root this stage has no model for, as it reads in AIR.
    /// Mirrors [`FragMeta::unmodelled_stage_attributes`].
    pub unmodelled_stage_attributes: Vec<String>,
    /// `[[max_total_threads_per_threadgroup(N)]]`: the largest threadgroup the entry was compiled
    /// to run. A dispatch wider than this is outside what the AIR body was built for, so the
    /// requested `LocalSize` is checked against it rather than emitted unread.
    pub max_work_group_size: Option<u32>,
    /// Function-constant-wrapped buffer parameter index -> Metal buffer location. Multiple mutually
    /// exclusive typed alternatives may intentionally share one location.
    pub function_constant_buffer_locations: HashMap<u32, u32>,
    /// `param_idx -> reconstructed struct layout` for buffer args (see `FragMeta::buffer_layouts`).
    pub buffer_layouts: HashMap<u32, AirType>,
    /// `param_idx -> reconstructed air.imageblock_data layout` for imageblock args.
    pub imageblock_layouts: HashMap<u32, AirType>,
    /// Descriptor-backed render-target planes used by implicit imageblock load/store intrinsics.
    pub implicit_imageblock_attachments: Vec<ImplicitImageblockAttachment>,
    /// `param_idx -> AIR address space` for buffer args. Address space 3 is threadgroup memory.
    pub buffer_address_spaces: HashMap<u32, u32>,
    /// `param_idx -> declared AIR buffer argument byte size`, from `air.arg_type_size` or
    /// `air.buffer_size` when present.
    pub buffer_type_sizes: HashMap<u32, u32>,
    /// `param_idx -> air.buffer_size` when AIR declares one exact reference-object extent.
    pub buffer_object_sizes: HashMap<u32, u32>,
    /// `param_idx -> AIR buffer argument type name`, e.g. `char`, `void`, or a struct name.
    pub buffer_type_names: HashMap<u32, String>,
    /// `param_idx -> declared AIR read/write qualifier`.
    pub buffer_accesses: HashMap<u32, BufferAccess>,
    /// `param_idx -> AIR texture argument type name`, e.g. `texture2d<uint, read>`.
    pub texture_type_names: HashMap<u32, String>,
    /// `param_idx -> the descriptor count `air.location_index` states, when it states more than one.
    ///
    /// A texture or sampler argument above `1` is a handle ARRAY occupying that many descriptors at
    /// its Metal slot. Absent for the ordinary single-descriptor argument and for a count spelled as
    /// a function-constant global.
    pub declared_descriptor_counts: HashMap<u32, u32>,
    /// `param_idx -> AIR kernel stage-input scalar/vector type name`.
    pub stage_input_type_names: HashMap<u32, String>,
    /// Textures EMBEDDED inside an `air.indirect_buffer` argument buffer (via `air.indirect_argument`
    /// → nested `air.texture`) that the kernel body reads/writes with AIR texture intrinsics. Each is
    /// surfaced as a standalone image resource so the read/write lowers to a real descriptor instead of a
    /// private placeholder. See [`EmbeddedTexture`].
    pub embedded_textures: Vec<EmbeddedTexture>,
    /// Every resource-handle member carried by an `air.indirect_buffer`, including handles whose
    /// concrete kind is supplied by an authored manifest and verified from its AIR use.
    pub embedded_arguments: Vec<EmbeddedArgument>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImplicitImageblockAttachment {
    pub attachment: u32,
    pub data_rate: u32,
    pub max_index: Option<u32>,
    pub format: TextureFormat,
    pub reads: bool,
    pub writes: bool,
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
    /// See [`Self::declared_descriptor_counts`].
    pub fn declared_descriptor_count(&self, idx: u32) -> Option<u32> {
        self.declared_descriptor_counts.get(&idx).copied()
    }

    /// Synthetic buffer slots used for kernel `[[stage_in]]` attributes.
    ///
    /// Metal supplies stage-input data through ordinary buffer-table slots selected by the pipeline
    /// descriptor. Vulkan exposes the same arrays as read-only storage buffers. Keeping allocation
    /// here makes reflection and lowering consume one ABI decision instead of duplicating it.
    pub fn stage_input_bindings(&self) -> HashMap<u32, u32> {
        let mut occupied = self
            .roles
            .iter()
            .filter_map(|(_, role)| match role {
                KernRole::Buffer(binding)
                | KernRole::AccelerationStructureShadow(binding)
                | KernRole::PrimitiveAccelerationStructure(binding)
                | KernRole::PrimitiveAccelerationStructureShadow(binding)
                | KernRole::VisibleFunctionTable(binding)
                | KernRole::IntersectionFunctionTable(binding) => Some(*binding),
                _ => None,
            })
            .collect::<std::collections::HashSet<_>>();
        let mut next = 0u32;
        let mut bindings = HashMap::new();
        for (param_index, role) in &self.roles {
            if !matches!(role, KernRole::StageInput(_)) {
                continue;
            }
            while occupied.contains(&next) {
                next = next.saturating_add(1);
            }
            occupied.insert(next);
            bindings.insert(*param_index, next);
        }
        bindings
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
/// the internal `fc_promoted_role` classifier). Standalone callers can request either projection;
/// production parses both
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
    let static_int_globals = static_init_int_global_values(ll);
    let resource_location =
        |node: &str, fallback: u32| location_index_with_static(node, fallback, &static_int_globals);
    let param_address_spaces = entry
        .and_then(|name| function_param_pointer_address_spaces(ll, name))
        .unwrap_or_default();
    let refs = refs_in(rootc);
    // The argument-info list is the SECOND ref (`!EMPTY` is the first — an empty placeholder node).
    let in_ref = *refs.get(1)?;
    let mut max_work_group_size = None;
    let mut unmodelled_stage_attributes = vec![];
    for attribute in stage_root_attributes(rootc, nodes) {
        match attribute {
            StageAttribute::MaxWorkGroupSize(size) => max_work_group_size = Some(size),
            other => unmodelled_stage_attributes.push(other.describe()),
        }
    }

    let mut roles = vec![];
    let mut unmodelled_input_params: Vec<(u32, String)> = vec![];
    let mut aliased_implicit_imageblock_params: Vec<u32> = vec![];
    let mut function_constant_buffer_locations = HashMap::new();
    let mut buffer_layouts = HashMap::new();
    let mut imageblock_layouts = HashMap::new();
    let mut buffer_address_spaces = HashMap::new();
    let mut buffer_type_sizes = HashMap::new();
    let mut buffer_object_sizes = HashMap::new();
    let mut buffer_type_names = HashMap::new();
    let mut buffer_accesses = HashMap::new();
    let mut texture_type_names = HashMap::new();
    let mut declared_descriptor_counts = HashMap::new();
    let mut stage_input_type_names = HashMap::new();
    // `air.location_index` of every top-level `air.texture` arg — the basis for the synthetic
    // embedded-texture index K (see `embedded_synthetic_texture_index`).
    let mut top_level_texture_locations: Vec<u32> = vec![];
    // `(buffer_param_index, struct_type_info_node_ref)` for each `air.indirect_buffer` arg, so
    // embedded-texture detection can run once K is known (after the whole arg list is scanned).
    let mut indirect_buffer_struct_refs: Vec<(u32, u32, u32)> = vec![];
    for r in refs_in(nodes.get(&in_ref)?) {
        let Some(node) = nodes.get(&r) else { continue };
        let Some(idx) = first_i32(node) else { continue };
        let layout = struct_info_ref(node).and_then(|sref| parse_struct_info(nodes, sref, 0));
        let strs = role_strings(node);
        if let Some(count) = declared_descriptor_count(node) {
            declared_descriptor_counts.insert(idx, count);
        }
        if strs.first().map(String::as_str) == Some("function_constant")
            && primary_role(&strs) == Some("buffer")
        {
            function_constant_buffer_locations.insert(idx, resource_location(node, idx));
        }
        let Some(mut first) = fc_promoted_role(&strs, promote_fc_buffers) else {
            continue;
        };
        if primary_role(&strs) == Some("texture") && resource_location(node, idx) == u32::MAX {
            first = "function_constant";
        }
        if let Some(declared) = declared_role(&strs) {
            if !KERNEL_INPUT_ROLES.contains(&declared)
                && metadata_enabled_by_default(node, nodes, &static_int_globals)
            {
                unmodelled_input_params.push((idx, declared.to_string()));
            }
        }
        let role = match first {
            "buffer" | "indirect_buffer" => {
                if first == "indirect_buffer" {
                    if let Some(sref) = struct_info_ref(node) {
                        // Key by the buffer's `air.location_index` (the Metal `[[buffer(N)]]` slot the
                        // harness binds), NOT the AIR argument position — they differ (e.g. arg 2 but
                        // buffer(0)). The oracle/runner both index buffers by location.
                        indirect_buffer_struct_refs.push((idx, resource_location(node, idx), sref));
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
                if let Some(size) = i32_after_marker(node, "air.buffer_size") {
                    buffer_object_sizes.insert(idx, size);
                }
                if let Some(access) = declared_buffer_access(node) {
                    buffer_accesses.insert(idx, access);
                }
                KernRole::Buffer(location_index_with_static(node, idx, &static_int_globals))
            }
            "texture" => {
                if let Some(name) = arg_type_name(node) {
                    texture_type_names.insert(idx, name);
                }
                let loc = resource_location(node, idx);
                top_level_texture_locations.push(loc);
                KernRole::Texture(loc)
            }
            "instance_acceleration_structure" if body_uses_acceleration_structure_shadow(ll) => {
                KernRole::AccelerationStructureShadow(resource_location(node, idx))
            }
            "primitive_acceleration_structure" => {
                let binding = resource_location(node, idx);
                if ll.contains("@air.intersect.") {
                    KernRole::PrimitiveAccelerationStructureShadow(binding)
                } else {
                    KernRole::PrimitiveAccelerationStructure(binding)
                }
            }
            "visible_function_table" => {
                KernRole::VisibleFunctionTable(resource_location(node, idx))
            }
            "intersection_function_table" => {
                KernRole::IntersectionFunctionTable(resource_location(node, idx))
            }
            "imageblock" => {
                if let Some(t) = layout {
                    imageblock_layouts.insert(idx, t);
                }
                if strs.iter().any(|s| s == "alias_implicit_imageblock") {
                    aliased_implicit_imageblock_params.push(idx);
                }
                KernRole::Other
            }
            "sampler" => KernRole::Sampler(resource_location(node, idx)),
            "threads_per_threadgroup" | "dispatch_threads_per_threadgroup" => {
                KernRole::ThreadsPerThreadgroup
            }
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
            "stage_in" => {
                if let Some(name) = arg_type_name(node) {
                    stage_input_type_names.insert(idx, name);
                }
                KernRole::StageInput(resource_location(node, idx))
            }
            _ => KernRole::Other,
        };
        roles.push((idx, role));
    }
    // Detect argument-buffer-embedded textures that the body uses through AIR texture
    // intrinsics. Gated purely on AIR structure/semantics — the `air.indirect_argument` →
    // `air.texture` marker chain plus stable AIR intrinsic families — so it cannot key on any shader
    // name. The body must actually use a texture intrinsic for us to surface it.
    let embedded_textures = if body_uses_texture_intrinsic(ll) {
        detect_embedded_textures(
            nodes,
            &indirect_buffer_struct_refs,
            &top_level_texture_locations,
        )
    } else {
        vec![]
    };
    let embedded_arguments = detect_embedded_arguments(nodes, &indirect_buffer_struct_refs);
    let implicit_imageblock_attachments = detect_implicit_imageblock_attachments(ll)?;
    Some(KernMeta {
        roles,
        unmodelled_input_params,
        aliased_implicit_imageblock_params,
        unmodelled_stage_attributes,
        max_work_group_size,
        function_constant_buffer_locations,
        buffer_layouts,
        imageblock_layouts,
        implicit_imageblock_attachments,
        buffer_address_spaces,
        buffer_type_sizes,
        buffer_object_sizes,
        buffer_type_names,
        buffer_accesses,
        texture_type_names,
        declared_descriptor_counts,
        stage_input_type_names,
        embedded_textures,
        embedded_arguments,
    })
}

/// Decode the stable AIR implicit-imageblock intrinsic suffix to its exact storage plane format.
/// `Ok(None)` means the symbol is not in this intrinsic family; an unknown family suffix is an
/// explicit error so reflection and corpus capability audits cannot silently omit a new ABI shape.
pub fn implicit_imageblock_texture_format(name: &str) -> Result<Option<TextureFormat>, String> {
    let suffix = name
        .strip_prefix("air.load.implicit_imageblock.")
        .or_else(|| name.strip_prefix("air.store.implicit_imageblock."));
    let Some(suffix) = suffix else {
        return Ok(None);
    };
    let format = match suffix {
        "f16" => TextureFormat::R16f,
        "v2f16" => TextureFormat::Rg16f,
        "v4f16" => TextureFormat::Rgba16f,
        "f32" => TextureFormat::R32f,
        "v4f32" => TextureFormat::Rgba32f,
        "i32" => TextureFormat::R32ui,
        _ => {
            return Err(format!(
                "{name} has unsupported implicit imageblock texel type"
            ))
        }
    };
    Ok(Some(format))
}

fn detect_implicit_imageblock_attachments(ll: &str) -> Option<Vec<ImplicitImageblockAttachment>> {
    let mut attachments =
        std::collections::BTreeMap::<(u32, u32, TextureFormat), ImplicitImageblockAttachment>::new(
        );
    for line in ll.lines() {
        let Some(at) = line.find("@air.") else {
            continue;
        };
        let call = &line[at + 1..];
        let Some(open) = call.find('(') else { continue };
        let name = &call[..open];
        let (reads, writes, value_prefix) = if name.starts_with("air.load.implicit_imageblock.") {
            (true, false, 0usize)
        } else if name.starts_with("air.store.implicit_imageblock.") {
            (false, true, 1usize)
        } else {
            continue;
        };
        let Some(close) = call[open + 1..].find(')') else {
            continue;
        };
        let args = split_top_level_commas(&call[open + 1..open + 1 + close]);
        let Some(attachment) = args
            .get(value_prefix)
            .and_then(|arg| typed_u32_constant(arg))
        else {
            continue;
        };
        let index = args
            .get(value_prefix + 2)
            .and_then(|arg| typed_u32_constant(arg));
        let Some(data_rate) = args
            .get(value_prefix + 3)
            .and_then(|arg| typed_u32_constant(arg))
        else {
            continue;
        };
        let format = implicit_imageblock_texture_format(name).ok().flatten()?;
        let entry = attachments
            .entry((attachment, data_rate, format))
            .or_insert(ImplicitImageblockAttachment {
                attachment,
                data_rate,
                max_index: index,
                format,
                reads: false,
                writes: false,
            });
        entry.reads |= reads;
        entry.writes |= writes;
        entry.max_index = match (entry.max_index, index) {
            (Some(left), Some(right)) => Some(left.max(right)),
            _ => None,
        };
    }
    Some(attachments.into_values().collect())
}

fn typed_u32_constant(value: &str) -> Option<u32> {
    value.split_whitespace().last()?.parse().ok()
}

fn body_uses_acceleration_structure_shadow(ll: &str) -> bool {
    ll.contains("@air.get_instance_count_instance_acceleration_structure")
        || ll.contains("@air.get_primitive_acceleration_structure_instance_acceleration_structure")
        || ll.lines().any(|line| {
            let Some(start) = line.find("@air.intersect.") else {
                return false;
            };
            let Some(end) = line[start + 1..].find('(') else {
                return false;
            };
            let callee = &line[start + 1..start + 1 + end];
            AirIntersectionFamily::parse(callee)
                .ok()
                .flatten()
                .is_some_and(|family| family.instancing != AirIntersectionInstancing::None)
        })
}

/// Whether every AIR intersection call in this module has an implemented structural lowering.
///
/// Validation tooling uses the same product-owned decision as translation so its authorability
/// inventory cannot drift into a second, independently maintained intrinsic allowlist.
pub fn air_intersection_calls_are_supported(ll: &str) -> bool {
    crate::native::ray_intersection::all_air_intersection_calls_are_lowerable(ll)
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
        // expect "<digits> = !{<body>}" or "<digits> = distinct !{<body>}"
        let Some((eq, prefix_len)) =
            rest.find(" = !{")
                .map(|eq| (eq, " = !{".len()))
                .or_else(|| {
                    rest.find(" = distinct !{")
                        .map(|eq| (eq, " = distinct !{".len()))
                })
        else {
            continue;
        };
        let Ok(id) = rest[..eq].parse::<u32>() else {
            continue;
        };
        let body = &rest[eq + prefix_len..];
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
    // body like: `ptr @BlurComposite, !16, !18` or `ptr @"re::df::pack", !16, !18`.
    let at = body.find('@')?;
    let after = &body[at + 1..];
    let name = if let Some(quoted) = after.strip_prefix('"') {
        quoted_symbol_name(quoted)?
    } else {
        after
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '.' || *c == '$')
            .collect()
    };
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

fn pointer_symbol(body: &str) -> Option<String> {
    let at = body.find('@')?;
    let after = &body[at + 1..];
    let name = if let Some(quoted) = after.strip_prefix('"') {
        quoted_symbol_name(quoted)?
    } else {
        after
            .chars()
            .take_while(|c| c.is_alphanumeric() || matches!(*c, '_' | '.' | '$'))
            .collect()
    };
    (!name.is_empty()).then_some(name)
}

fn quoted_symbol_name(s: &str) -> Option<String> {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' => return Some(out),
            '\\' => {
                let hi = chars.peek().copied();
                let mut clone = chars.clone();
                let lo = {
                    clone.next();
                    clone.peek().copied()
                };
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    if hi.is_ascii_hexdigit() && lo.is_ascii_hexdigit() {
                        chars.next();
                        chars.next();
                        let byte = u8::from_str_radix(&format!("{hi}{lo}"), 16).ok()?;
                        out.push(byte as char);
                        continue;
                    }
                }
                out.push(chars.next().unwrap_or('\\'));
            }
            _ => out.push(ch),
        }
    }
    None
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
    let unquoted = format!("@{name}(");
    let quoted = format!("@\"{name}\"(");
    let (start, needle_len) = ll
        .find(&unquoted)
        .map(|start| (start, unquoted.len()))
        .or_else(|| ll.find(&quoted).map(|start| (start, quoted.len())))?;
    let start = start + needle_len;
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

/// One per-entry attribute carried by a stage root node, past its
/// `(function, outputs, inputs)` operand triple.
///
/// All three roots -- `!air.kernel`, `!air.vertex`, `!air.fragment` -- grow their entry
/// attributes in that same tail, but not in the same form: some are references to a keyed node
/// (`!{!"air.patch", ...}`), some are bare strings (`!"early_fragment_tests"`). A stage that
/// scans the tail for only the one attribute it knows cannot tell "no attribute" from "an
/// attribute I have never seen", so a new one is dropped in silence. Decoding the whole tail in
/// one place is what makes the difference observable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum StageAttribute {
    /// `!{!"air.patch", !"<domain>", !"air.patch_control_point", i32 N}` on a post-tessellation
    /// vertex function. Carries the node body; the vertex decode reads the shape out of it.
    Patch(String),
    /// `!"early_fragment_tests"` -- `[[early_fragment_tests]]`. Depth and stencil tests run
    /// before the fragment body, so a fragment the test rejects performs none of the body's
    /// stores. Vulkan spells it `OpExecutionMode ... EarlyFragmentTests`.
    EarlyFragmentTests,
    /// `!{!"air.max_work_group_size", i32 N}` -- `[[max_total_threads_per_threadgroup(N)]]`. The
    /// largest threadgroup this entry was compiled to run.
    MaxWorkGroupSize(u32),
    /// A tail operand this translator has no model for. Carried out so a stage can refuse rather
    /// than emit a module that silently lacks whatever the attribute asked for.
    Unrecognized(String),
}

impl StageAttribute {
    /// How the attribute reads back in a refusal message.
    pub(crate) fn describe(&self) -> String {
        match self {
            StageAttribute::Patch(_) => "air.patch".to_string(),
            StageAttribute::EarlyFragmentTests => "early_fragment_tests".to_string(),
            StageAttribute::MaxWorkGroupSize(_) => "air.max_work_group_size".to_string(),
            StageAttribute::Unrecognized(text) => text.clone(),
        }
    }
}

/// Decode the attribute tail of a stage root node body.
///
/// The first three operands are the entry function pointer, its output node and its input node;
/// every operand after them is an attribute. Recognition is by the attribute's own ABI marker,
/// never by its position in the tail.
fn stage_root_attributes(rootc: &str, nodes: &HashMap<u32, String>) -> Vec<StageAttribute> {
    let operands = split_top_level_commas(rootc);
    operands
        .iter()
        .skip(3)
        .filter_map(|operand| {
            let referenced = operand
                .strip_prefix('!')
                .and_then(|digits| digits.parse::<u32>().ok())
                .and_then(|id| nodes.get(&id));
            Some(match referenced {
                Some(node) if node.contains("!\"air.patch\"") => {
                    StageAttribute::Patch(node.clone())
                }
                Some(node) if node.contains("!\"air.max_work_group_size\"") => {
                    match i32_after_marker(node, "air.max_work_group_size") {
                        Some(size) => StageAttribute::MaxWorkGroupSize(size),
                        None => StageAttribute::Unrecognized(
                            "air.max_work_group_size states no thread count".to_string(),
                        ),
                    }
                }
                // A node with no operands states nothing, so there is nothing here to model and
                // nothing emission could drop. Refusing it would cost a translation for no safety.
                Some(node) if node.trim().is_empty() => return None,
                Some(node) => StageAttribute::Unrecognized(format!("!{{{node}}}")),
                None if operand == "!\"early_fragment_tests\"" => {
                    StageAttribute::EarlyFragmentTests
                }
                None => StageAttribute::Unrecognized(operand.clone()),
            })
        })
        .collect()
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
    match location_operands(body).map(|operands| operands.index) {
        Some(LocationOperand::Literal(index)) => index,
        _ => fallback,
    }
}

/// How many descriptors an argument node states it occupies, when AIR states more than one.
///
/// `None` for the ordinary single-descriptor argument, and for a count spelled as a
/// function-constant global -- the consumer picks that length at pipeline creation, so there is no
/// compile-time array to size.
fn declared_descriptor_count(body: &str) -> Option<u32> {
    match location_operands(body)?.count? {
        LocationOperand::Literal(count) if count > 1 => Some(count),
        _ => None,
    }
}

/// One operand of the `air.location_index` pair: either a compile-time value or a pointer to a
/// function-constant global whose value the consumer chooses at pipeline creation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LocationOperand {
    Literal(u32),
    Global(String),
}

/// The two operands `air.location_index` always carries: the resource's Metal slot, and how many
/// descriptors the argument occupies at that slot.
///
/// `!"air.location_index", i32 4, i32 2` is `[[texture(4)]]` holding two texture handles. The count
/// is `1` for an ordinary resource; above `1` the argument is a handle ARRAY, and every corpus
/// declaration above `1` agrees with the `array<..., N>` length in the type name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LocationOperands {
    pub index: LocationOperand,
    pub count: Option<LocationOperand>,
}

/// Decode the `air.location_index` operand pair POSITIONALLY.
///
/// Either operand may be a literal or a global, and all four combinations occur -- measured over
/// 14579 corpus sources: 112946 `(i32, i32)`, 3404 `(ptr, i32)`, 1447 `(ptr, ptr)`, 439
/// `(i32, ptr)`, and no other shape. Scanning forward for "the next `i32`" or "the next `@`"
/// therefore reads the COUNT whenever the slot is spelled the other way: a texture argument at
/// `[[texture(1)]]` with a function-constant array count would be bound at the count's value
/// instead of at 1. 69 corpus texture nodes are that shape.
pub(crate) fn location_operands(body: &str) -> Option<LocationOperands> {
    let marker = "!\"air.location_index\"";
    let position = body.find(marker)?;
    let mut operands = split_metadata_operands(&body[position + marker.len()..])
        .into_iter()
        .map(|operand| {
            if let Some(literal) = operand.strip_prefix("i32 ") {
                return literal.trim().parse().ok().map(LocationOperand::Literal);
            }
            let at = operand.find('@')?;
            let name = operand[at..]
                .chars()
                .take_while(|character| {
                    !character.is_whitespace() && !matches!(*character, ',' | ')' | '(' | '[' | ']')
                })
                .collect::<String>();
            (name.len() > 1).then_some(LocationOperand::Global(name))
        });
    Some(LocationOperands {
        index: operands.next().flatten()?,
        count: operands.next().flatten(),
    })
}

/// Split an AIR metadata operand list on the commas that separate operands. A `!"..."` string
/// operand may itself contain commas (`!"array<texture2d<half, sample>, 2>"`), so a plain
/// `split(',')` would tear one operand into three and shift every position after it.
fn split_metadata_operands(tail: &str) -> Vec<String> {
    let mut operands = vec![];
    let mut current = String::new();
    let mut quoted = false;
    for character in tail.chars() {
        match character {
            '"' => {
                quoted = !quoted;
                current.push(character);
            }
            ',' if !quoted => {
                operands.push(std::mem::take(&mut current).trim().to_string());
            }
            _ => current.push(character),
        }
    }
    operands.push(current.trim().to_string());
    operands.retain(|operand| !operand.is_empty());
    operands
}

fn render_target_location(
    body: &str,
    fallback: u32,
    static_int_globals: &HashMap<String, u32>,
) -> u32 {
    global_after_marker(body, "air.render_target")
        .and_then(|global| static_int_globals.get(&global).copied())
        .or_else(|| i32_after_marker(body, "air.render_target"))
        .unwrap_or(fallback)
}

fn global_after_marker(body: &str, marker: &str) -> Option<String> {
    let marker = format!("!\"{marker}\"");
    let pos = body.find(&marker)?;
    let after = &body[pos + marker.len()..];
    let at = after.find('@')?;
    let name = after[at..]
        .chars()
        .take_while(|c| !c.is_whitespace() && !matches!(*c, ',' | ')' | '(' | '[' | ']'))
        .collect::<String>();
    (name.len() > 1).then_some(name)
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

/// The role marker an argument node declares, or `None` when it declares none.
///
/// A bare `!"air.function_constant"` parameter *is* the function constant: there is no role marker
/// behind the wrapper, so reading past it lands on the node's own type/name qualifiers and reports
/// `arg_type_name` as though it were a role. [`fc_promoted_role`] already models this — it collapses
/// a gated argument to the wrapper unless the role is one the emitter binds anyway — so the two
/// stay consistent by sharing it.
fn declared_role(strs: &[String]) -> Option<&str> {
    match fc_promoted_role(strs, true)? {
        "function_constant" | "function_constant_disabled" => None,
        role => Some(role),
    }
}

fn function_constant_gate_global(body: &str, nodes: &HashMap<u32, String>) -> Option<String> {
    if role_strings(body).first().map(String::as_str) != Some("function_constant") {
        return None;
    }
    refs_in(body)
        .into_iter()
        .filter_map(|r| nodes.get(&r))
        .find_map(|node| {
            let at = node.find('@')?;
            let name = node[at..]
                .chars()
                .take_while(|c| !c.is_whitespace() && !matches!(*c, ',' | ')' | '(' | '[' | ']'))
                .collect::<String>();
            (name.len() > 1).then_some(name)
        })
}

fn metadata_enabled_by_default(
    body: &str,
    nodes: &HashMap<u32, String>,
    static_int_globals: &HashMap<String, u32>,
) -> bool {
    function_constant_gate_global(body, nodes)
        .map(|global| {
            static_int_globals
                .get(&global)
                .is_some_and(|value| *value != 0)
        })
        .unwrap_or(true)
}

/// Specialize metadata-only `air.function_constant` wrappers from statically evaluated AIR
/// predicates. A true predicate exposes the wrapped role as unconditional. A false predicate is
/// marked disabled so interface construction cannot retain an unreachable descriptor or output.
/// Unknown predicates keep the wrapper and the ordinary unspecialized projection.
pub(crate) fn specialize_function_constant_metadata(ll: &str) -> String {
    let nodes = collect_nodes(ll);
    let static_int_globals = static_init_int_global_values(ll);
    let marker = "!\"air.function_constant\"";
    let mut output = String::with_capacity(ll.len());

    for line in ll.split_inclusive('\n') {
        let value = function_constant_gate_global(line, &nodes)
            .and_then(|global| static_int_globals.get(&global).copied());
        match value {
            Some(0) => {
                output.push_str(&line.replacen(marker, "!\"air.function_constant_disabled\"", 1))
            }
            Some(_) => {
                let rewritten = line.find(marker).and_then(|start| {
                    let after_marker = &line[start + marker.len()..];
                    let next_role = after_marker.find("!\"air.")?;
                    Some(format!("{}{}", &line[..start], &after_marker[next_role..]))
                });
                if let Some(rewritten) = rewritten {
                    output.push_str(&rewritten);
                } else {
                    output.push_str(line);
                }
            }
            None => output.push_str(line),
        }
    }
    output
}

/// Like `primary_role`, but only looks past the `air.function_constant` wrapper when the wrapped
/// resource is a TEXTURE, IMAGEBLOCK, FUNCTION TABLE, or BUFFER (the latter only when
/// `promote_buffers` is set).
/// An imageblock has no descriptor binding to promote; recognizing its stable ABI marker merely keeps
/// its metadata-described cell type available to the native emitter. A wrapped texture with a real
/// location remains a descriptor even when its predicate defaults false: SPIR-V specialization may
/// enable the resource-using arm, and substituting another texture or erasing the binding would be
/// semantically wrong. A `-1` location remains absent until AIR specialization assigns a binding.
/// Promoting a wrapped BUFFER binds it as a
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
            Some("visible_function_table") => "visible_function_table",
            Some("intersection_function_table") => "intersection_function_table",
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

fn declared_buffer_access(body: &str) -> Option<BufferAccess> {
    if body.contains("air.read_write") {
        Some(BufferAccess::ReadWrite)
    } else if body.contains("air.write") {
        Some(BufferAccess::WriteOnly)
    } else if body.contains("air.read") {
        Some(BufferAccess::ReadOnly)
    } else {
        None
    }
}

fn arg_name(body: &str) -> Option<String> {
    string_after_marker(body, "air.arg_name")
}

fn ref_after_marker(body: &str, marker: &str) -> Option<u32> {
    let marker = format!("!\"{marker}\"");
    let after = body.get(body.find(&marker)? + marker.len()..)?;
    let bang = after.find('!')?;
    let digits = after[bang + 1..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    digits.parse().ok()
}

fn parse_fragment_imageblock_master(body: &str) -> Option<Vec<FragmentImageblockMember>> {
    let toks = tokenize(body);
    let mut members = Vec::new();
    let mut i = 0;
    while i + 4 < toks.len() {
        let (offset, size, type_name, semantic) = match (
            toks.get(i),
            toks.get(i + 1),
            toks.get(i + 2),
            toks.get(i + 3),
            toks.get(i + 4),
        ) {
            (
                Some(Tok::Int(offset)),
                Some(Tok::Int(size)),
                Some(Tok::Int(_array_len)),
                Some(Tok::Str(type_name)),
                Some(Tok::Str(semantic)),
            ) => (*offset, *size, type_name.clone(), semantic.clone()),
            _ => return None,
        };
        i += 5;
        let raster_order_group = match (toks.get(i), toks.get(i + 1)) {
            (Some(Tok::Str(marker)), Some(Tok::Int(group)))
                if marker == "air.raster_order_group" =>
            {
                i += 2;
                *group
            }
            _ => return None,
        };
        members.push(FragmentImageblockMember {
            offset,
            size,
            type_name,
            semantic,
            raster_order_group,
        });
    }
    (!members.is_empty() && i == toks.len()).then_some(members)
}

fn parse_fragment_imageblock_projection(
    nodes: &HashMap<u32, String>,
    node: &str,
    interface_index: u32,
    master_members: &[FragmentImageblockMember],
) -> Option<FragmentImageblockProjection> {
    let projection_ref = struct_info_ref(node)?;
    let projection = nodes.get(&projection_ref)?;
    let toks = tokenize(projection);
    let mut members = Vec::new();
    let mut i = 0;
    let mut projection_member = 0;
    while i + 4 < toks.len() {
        let semantic = match (
            toks.get(i),
            toks.get(i + 1),
            toks.get(i + 2),
            toks.get(i + 3),
            toks.get(i + 4),
        ) {
            (
                Some(Tok::Int(_)),
                Some(Tok::Int(_)),
                Some(Tok::Int(_)),
                Some(Tok::Str(_)),
                Some(Tok::Str(semantic)),
            ) => semantic,
            _ => return None,
        };
        let master_member = master_members
            .iter()
            .position(|member| member.semantic == *semantic)? as u32;
        members.push(FragmentImageblockProjectionMember {
            projection_member,
            master_member,
        });
        projection_member += 1;
        i += 5;
        if matches!(toks.get(i), Some(Tok::Str(marker)) if marker == "air.raster_order_group")
            && matches!(toks.get(i + 1), Some(Tok::Int(_)))
        {
            i += 2;
        }
    }
    (!members.is_empty() && i == toks.len()).then_some(FragmentImageblockProjection {
        interface_index,
        members,
    })
}

fn parse_fragment_imageblock(
    nodes: &HashMap<u32, String>,
    out_ref: u32,
    in_ref: u32,
) -> Option<FragmentImageblock> {
    let output_nodes = nodes
        .get(&out_ref)
        .map(|body| refs_in(body))
        .unwrap_or_default();
    let input_nodes = nodes
        .get(&in_ref)
        .map(|body| refs_in(body))
        .unwrap_or_default();
    let imageblock_node = output_nodes
        .iter()
        .chain(input_nodes.iter())
        .filter_map(|id| nodes.get(id))
        .find(|body| primary_role(&role_strings(body)) == Some("imageblock_data"))?;
    // Some AIR producers attach an explicit `air.imageblock_master` to a narrow projection. Others
    // pass the complete imageblock-data struct directly, in which case its own struct metadata is
    // the master layout. Both contracts carry the same offset/size/type/semantic/ROG tuple stream;
    // use that structure rather than requiring the optional indirection.
    let master_ref = ref_after_marker(imageblock_node, "air.imageblock_master")
        .or_else(|| struct_info_ref(imageblock_node))?;
    let master_members = parse_fragment_imageblock_master(nodes.get(&master_ref)?)?;
    let sample_size = i32_after_marker(imageblock_node, "air.imageblock_data_size")?;

    let outputs = output_nodes
        .iter()
        .enumerate()
        .filter_map(|(index, id)| {
            let node = nodes.get(id)?;
            (primary_role(&role_strings(node)) == Some("imageblock_data"))
                .then(|| {
                    parse_fragment_imageblock_projection(nodes, node, index as u32, &master_members)
                })
                .flatten()
        })
        .collect();
    let inputs = input_nodes
        .iter()
        .filter_map(|id| {
            let node = nodes.get(id)?;
            let index = first_i32(node)?;
            (primary_role(&role_strings(node)) == Some("imageblock_data"))
                .then(|| parse_fragment_imageblock_projection(nodes, node, index, &master_members))
                .flatten()
        })
        .collect();
    Some(FragmentImageblock {
        sample_size,
        members: master_members,
        inputs,
        outputs,
    })
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

/// The fragment entry-parameter roles the emitter models.
///
/// Any other enabled role lands in [`FragMeta::unmodelled_input_params`] and is rejected at
/// emission. `render_target` appears here because an `air.render_target` in the *input* list is
/// framebuffer fetch (`[[color(n)]]`), not an output.
/// The fragment entry-parameter roles that name a SPIR-V builtin rather than a bound resource.
///
/// A resource keeps its descriptor whether or not its function constant is on: the pipeline layout
/// has to match what the application binds either way. A system value does not. When its constant
/// is off the parameter is absent, and declaring the builtin anyway puts a variable in the entry
/// point interface — and, for `viewport_array_index` and `render_target_array_index`, a device
/// capability in the module — for a value the shader cannot read.
///
/// The vertex and kernel decodes already drop a gated-off system value, because `fc_promoted_role`
/// collapses everything but a promoted resource back to the wrapper. The fragment decode reads the
/// role past the wrapper unconditionally, so this list is how it makes the same distinction.
pub const FRAGMENT_SYSTEM_VALUE_ROLES: &[&str] = &[
    "barycentric_coord",
    "front_facing",
    "point_coord",
    "position",
    "primitive_id",
    "render_target_array_index",
    "sample_id",
    "sample_mask_in",
    "viewport_array_index",
];

pub const FRAGMENT_INPUT_ROLES: &[&str] = &[
    "barycentric_coord",
    "buffer",
    "fragment_input",
    "front_facing",
    "imageblock_data",
    "indirect_buffer",
    "intersection_function_table",
    "point_coord",
    "position",
    "primitive_id",
    "render_target",
    "render_target_array_index",
    "sample_id",
    "sample_mask_in",
    "sampler",
    "texture",
    "viewport_array_index",
    "visible_function_table",
];

/// The vertex entry-parameter roles the emitter models. See [`FRAGMENT_INPUT_ROLES`].
pub const VERTEX_INPUT_ROLES: &[&str] = &[
    "amplification_count",
    "amplification_id",
    "buffer",
    "indirect_buffer",
    "instance_id",
    "intersection_function_table",
    "patch_control_point_input",
    "patch_id",
    "patch_input",
    "position_in_patch",
    "sampler",
    "texture",
    "vertex_id",
    "vertex_input",
    "visible_function_table",
];

/// The kernel entry-parameter roles the emitter models. See [`FRAGMENT_INPUT_ROLES`].
pub const KERNEL_INPUT_ROLES: &[&str] = &[
    "buffer",
    "dispatch_threads_per_threadgroup",
    "imageblock",
    "indirect_buffer",
    "instance_acceleration_structure",
    "intersection_function_table",
    "primitive_acceleration_structure",
    "quadgroup_index_in_threadgroup",
    "sampler",
    "simdgroup_index_in_threadgroup",
    "simdgroups_per_threadgroup",
    "stage_in",
    "texture",
    "thread_index_in_quadgroup",
    "thread_index_in_simdgroup",
    "thread_index_in_threadgroup",
    "thread_position_in_grid",
    "thread_position_in_threadgroup",
    "threadgroup_position_in_grid",
    "threadgroups_per_grid",
    "threads_per_grid",
    "threads_per_simdgroup",
    "threads_per_threadgroup",
    "visible_function_table",
];

/// The fragment return-member roles the emitter lowers.
///
/// A member carrying any other role is reported through `FragMeta::unmodelled_output_members` and
/// rejected at emission rather than skipped. AIR states an output because the shader writes it, so
/// a member nothing knows how to write is a value that silently disappears — the shape that let
/// `[[sample_mask]]` be dropped without a diagnostic for as long as it was unrecognised.
pub const FRAGMENT_OUTPUT_ROLES: &[&str] = &[
    "render_target",
    "depth",
    "stencil",
    "sample_mask",
    "imageblock_data",
];

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
    let mut early_fragment_tests = false;
    let mut unmodelled_stage_attributes = vec![];
    for attribute in stage_root_attributes(rootc, nodes) {
        match attribute {
            StageAttribute::EarlyFragmentTests => early_fragment_tests = true,
            other => unmodelled_stage_attributes.push(other.describe()),
        }
    }
    let fragment_imageblock = parse_fragment_imageblock(nodes, out_ref, in_ref);
    let render_target_members: Vec<(u32, u32)> = nodes
        .get(&out_ref)
        .map(|c| {
            refs_in(c)
                .into_iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    let node = nodes.get(&r)?;
                    let roles = role_strings(node);
                    let is_render_target = primary_role(&roles) == Some("render_target");
                    (is_render_target
                        && metadata_enabled_by_default(node, nodes, &static_int_globals))
                    .then(|| {
                        (
                            i as u32,
                            render_target_location(node, i as u32, &static_int_globals),
                        )
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    // Which return members carry a given non-color output role. One reader for every such role, so
    // adding one is a call rather than another copy of this walk — `air.sample_mask` was dropped
    // for as long as it was the role nobody had copied the walk for.
    let output_members_with_role = |role: &str| -> Vec<u32> {
        nodes
            .get(&out_ref)
            .map(|c| {
                refs_in(c)
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, r)| {
                        let node = nodes.get(&r)?;
                        (primary_role(&role_strings(node)) == Some(role)
                            && metadata_enabled_by_default(node, nodes, &static_int_globals))
                        .then_some(i as u32)
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let depth_members = output_members_with_role("depth");
    let depth_qualifier = nodes.get(&out_ref).and_then(|c| {
        refs_in(c).into_iter().find_map(|r| {
            let node = nodes.get(&r)?;
            (primary_role(&role_strings(node)) == Some("depth"))
                .then(|| string_after_marker(node, "air.depth_qualifier"))
                .flatten()
                .and_then(|qualifier| match qualifier.as_str() {
                    "air.any" => Some(DepthQualifier::Any),
                    "air.less" => Some(DepthQualifier::Less),
                    "air.greater" => Some(DepthQualifier::Greater),
                    _ => None,
                })
        })
    });
    let stencil_members = output_members_with_role("stencil");
    let sample_mask_members = output_members_with_role("sample_mask");
    let unmodelled_output_members: Vec<(u32, String)> = nodes
        .get(&out_ref)
        .map(|c| {
            refs_in(c)
                .into_iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    let node = nodes.get(&r)?;
                    let roles = role_strings(node);
                    let role = declared_role(&roles)?;
                    (!FRAGMENT_OUTPUT_ROLES.contains(&role)
                        && metadata_enabled_by_default(node, nodes, &static_int_globals))
                    .then(|| (i as u32, role.to_string()))
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
                    let roles = role_strings(node);
                    let is_render_target = primary_role(&roles) == Some("render_target");
                    if is_render_target
                        && metadata_enabled_by_default(node, nodes, &static_int_globals)
                    {
                        arg_type_name(node).map(|name| (i as u32, name))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let mut roles = vec![];
    let mut unmodelled_input_params: Vec<(u32, String)> = vec![];
    let mut buffer_layouts = HashMap::new();
    let mut varying_types = HashMap::new();
    let mut varying_names = HashMap::new();
    let mut varying_user_semantics = HashMap::new();
    let mut varying_interpolation = HashMap::new();
    let mut texture_type_names = HashMap::new();
    let mut declared_descriptor_counts = HashMap::new();
    let mut color_input_type_names = HashMap::new();
    let mut varying_loc = 0u32;
    let mut buffer_address_spaces = HashMap::new();
    let mut buffer_type_sizes = HashMap::new();
    let mut buffer_object_sizes = HashMap::new();
    let mut buffer_type_names = HashMap::new();
    let mut buffer_accesses = HashMap::new();
    let mut indirect_buffer_struct_refs: Vec<(u32, u32, u32)> = Vec::new();
    // AIR function-param pointer address spaces for the fragment entry, the fallback the kernel
    // parser uses when a buffer arg node omits `air.address_space`.
    let param_address_spaces = entry
        .and_then(|name| function_param_pointer_address_spaces(ll, name))
        .unwrap_or_default();
    for r in refs_in(nodes.get(&in_ref)?) {
        let Some(node) = nodes.get(&r) else { continue };
        let Some(idx) = first_i32(node) else { continue };
        // Reconstruct any buffer struct layout used when native emission produces a bare pointer.
        if let Some(sref) = struct_info_ref(node) {
            if let Some(t) = parse_struct_info(nodes, sref, 0) {
                buffer_layouts.insert(idx, t);
            }
        }
        let strs = role_strings(node);
        if let Some(count) = declared_descriptor_count(node) {
            declared_descriptor_counts.insert(idx, count);
        }
        let Some(role_str) = primary_role(&strs) else {
            continue;
        };
        // A gated-off system value is absent; falling through to `Other` binds a zero for it.
        let role_str = if FRAGMENT_SYSTEM_VALUE_ROLES.contains(&role_str)
            && !metadata_enabled_by_default(node, nodes, &static_int_globals)
        {
            ""
        } else {
            role_str
        };
        if let Some(declared) = declared_role(&strs) {
            if !FRAGMENT_INPUT_ROLES.contains(&declared)
                && metadata_enabled_by_default(node, nodes, &static_int_globals)
            {
                unmodelled_input_params.push((idx, declared.to_string()));
            }
        }
        let role = match role_str {
            "position" => FragRole::Position,
            "point_coord" => FragRole::PointCoord,
            "front_facing" => FragRole::FrontFacing,
            "barycentric_coord" => FragRole::BarycentricCoord {
                no_perspective: VaryingInterpolation::from_role_strings(&strs).no_perspective,
            },
            "primitive_id" => FragRole::PrimitiveId,
            "sample_id" => FragRole::SampleId,
            "sample_mask_in" => FragRole::SampleMaskIn,
            "viewport_array_index" => FragRole::ViewportArrayIndex,
            "render_target_array_index" => FragRole::RenderTargetArrayIndex,
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
                varying_interpolation.insert(l, VaryingInterpolation::from_role_strings(&strs));
                FragRole::Varying(l)
            }
            "texture" if location_index_with_static(node, idx, &static_int_globals) != u32::MAX => {
                if let Some(name) = arg_type_name(node) {
                    texture_type_names.insert(idx, name);
                }
                FragRole::Texture(location_index_with_static(node, idx, &static_int_globals))
            }
            "sampler" => {
                FragRole::Sampler(location_index_with_static(node, idx, &static_int_globals))
            }
            "visible_function_table" => FragRole::VisibleFunctionTable(location_index_with_static(
                node,
                idx,
                &static_int_globals,
            )),
            "intersection_function_table" => FragRole::IntersectionFunctionTable(
                location_index_with_static(node, idx, &static_int_globals),
            ),
            "buffer" | "indirect_buffer" => {
                if role_str == "indirect_buffer" {
                    if let Some(sref) = struct_info_ref(node) {
                        indirect_buffer_struct_refs.push((
                            idx,
                            location_index_with_static(node, idx, &static_int_globals),
                            sref,
                        ));
                    }
                }
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
                if let Some(size) = i32_after_marker(node, "air.buffer_size") {
                    buffer_object_sizes.insert(idx, size);
                }
                if let Some(name) = arg_type_name(node) {
                    buffer_type_names.insert(idx, name);
                }
                if let Some(access) = declared_buffer_access(node) {
                    buffer_accesses.insert(idx, access);
                }
                FragRole::Buffer(location_index_with_static(node, idx, &static_int_globals))
            }
            // an air.render_target in the INPUT list = framebuffer fetch ([[color(n)]]).
            "render_target" => {
                let location = render_target_location(node, idx, &static_int_globals);
                if let Some(name) = arg_type_name(node) {
                    color_input_type_names.insert(location, name);
                }
                FragRole::ColorInput(location)
            }
            "imageblock_data" => FragRole::ImageblockData,
            _ => FragRole::Other,
        };
        roles.push((idx, role));
    }
    let top_level_texture_locations = roles
        .iter()
        .filter_map(|(_, role)| match role {
            FragRole::Texture(location) => Some(*location),
            _ => None,
        })
        .collect::<Vec<_>>();
    Some(FragMeta {
        roles,
        unmodelled_stage_attributes,
        early_fragment_tests,
        unmodelled_input_params,
        implicit_imageblock_attachments: detect_implicit_imageblock_attachments(ll)?,
        varying_types,
        varying_names,
        varying_user_semantics,
        varying_interpolation,
        n_render_targets,
        render_target_members,
        render_target_type_names,
        depth_members,
        depth_qualifier,
        stencil_members,
        sample_mask_members,
        unmodelled_output_members,
        fragment_imageblock,
        render_target_indices,
        buffer_layouts,
        buffer_address_spaces,
        buffer_type_sizes,
        buffer_object_sizes,
        buffer_type_names,
        buffer_accesses,
        texture_type_names,
        declared_descriptor_counts,
        color_input_type_names,
        embedded_textures: if body_uses_texture_intrinsic(ll) {
            detect_embedded_textures(
                nodes,
                &indirect_buffer_struct_refs,
                &top_level_texture_locations,
            )
        } else {
            Vec::new()
        },
        embedded_arguments: detect_embedded_arguments(nodes, &indirect_buffer_struct_refs),
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
    // Read the whole attribute tail, not just the one attribute this stage models. A patch node is
    // recognised by what it says (`air.patch`) rather than by where it sits, and anything else the
    // root carries becomes a named refusal instead of silence.
    let mut patch_node = None;
    let mut unmodelled_stage_attributes = vec![];
    for attribute in stage_root_attributes(rootc, nodes) {
        match attribute {
            StageAttribute::Patch(node) => patch_node = Some(node),
            other => unmodelled_stage_attributes.push(other.describe()),
        }
    }
    let mut undecoded_patch_shape = None;
    let patch_shape = patch_node.as_deref().and_then(|node| {
        let domain = if node.contains("!\"quad\"") {
            PatchDomain::Quad
        } else if node.contains("!\"triangle\"") {
            PatchDomain::Triangle
        } else if node.contains("!\"isoline\"") {
            PatchDomain::Isoline
        } else {
            // Dropping the shape here would emit an ordinary vertex shader: no domain, no spacing,
            // no winding, and a per-patch input set the pipeline never wires. Carry the failure out
            // so the passes can refuse instead.
            undecoded_patch_shape = Some("air.patch names no tessellation domain".to_string());
            return None;
        };
        match i32_after_marker(node, "air.patch_control_point") {
            Some(count) => Some((domain, count)),
            None => {
                undecoded_patch_shape =
                    Some("air.patch states no air.patch_control_point count".to_string());
                None
            }
        }
    });

    let mut output_roles = vec![];
    let mut invariant_outputs = vec![];
    let mut unmodelled_output_members = vec![];
    let mut output_varying_types = HashMap::new();
    let mut output_varying_names = HashMap::new();
    let mut output_varying_user_semantics = HashMap::new();
    let mut out_loc = 0u32;
    for r in refs_in(nodes.get(&out_ref)?) {
        let Some(node) = nodes.get(&r) else { continue };
        let strs = role_strings(node);
        let Some(first) = fc_promoted_role(&strs, false) else {
            continue;
        };
        let role = match first {
            "function_constant" | "function_constant_disabled" => {
                VertOutRole::FunctionConstantDisabled
            }
            "position" => VertOutRole::Position,
            "point_size" => VertOutRole::PointSize,
            "clip_distance" => VertOutRole::ClipDistance,
            "viewport_array_index" => VertOutRole::ViewportArrayIndex,
            "render_target_array_index" => VertOutRole::RenderTargetArrayIndex,
            "vertex_output" => {
                let l = location_index_with_static(node, out_loc, &static_int_globals);
                out_loc += 1;
                if let Some(name) = arg_type_name(node) {
                    output_varying_types.insert(l, name);
                }
                if let Some(name) = arg_name(node) {
                    output_varying_names.insert(l, name);
                }
                if let Some(semantic) = string_after_marker(node, "air.vertex_output") {
                    output_varying_user_semantics.insert(l, semantic);
                }
                VertOutRole::Varying(l)
            }
            _ => VertOutRole::Other,
        };
        if strs.iter().any(|s| s == "invariant") {
            invariant_outputs.push(output_roles.len() as u32);
        }
        if matches!(role, VertOutRole::Other) {
            unmodelled_output_members.push((output_roles.len() as u32, first.to_string()));
        }
        output_roles.push(role);
    }

    let mut roles = vec![];
    let mut unmodelled_input_params: Vec<(u32, String)> = vec![];
    let mut parameter_type_names = HashMap::new();
    let mut vertex_input_types = HashMap::new();
    let mut vertex_input_names = HashMap::new();
    let mut patch_input_types = HashMap::new();
    let mut patch_input_names = HashMap::new();
    let mut buffer_layouts = HashMap::new();
    let mut buffer_address_spaces = HashMap::new();
    let mut buffer_type_sizes = HashMap::new();
    let mut buffer_object_sizes = HashMap::new();
    let mut buffer_type_names = HashMap::new();
    let mut buffer_accesses = HashMap::new();
    let mut texture_type_names = HashMap::new();
    let mut declared_descriptor_counts = HashMap::new();
    let mut indirect_buffer_struct_refs = Vec::new();
    let mut patch_control_point = None;
    let param_address_spaces = entry
        .and_then(|name| function_param_pointer_address_spaces(ll, name))
        .unwrap_or_default();
    let mut vin_loc = 0u32;
    for r in refs_in(nodes.get(&in_ref)?) {
        let Some(node) = nodes.get(&r) else { continue };
        let Some(idx) = first_i32(node) else { continue };
        if let Some(name) = arg_type_name(node) {
            parameter_type_names.insert(idx, name);
        }
        if let Some(sref) = struct_info_ref(node) {
            if let Some(t) = parse_struct_info(nodes, sref, 0) {
                buffer_layouts.insert(idx, t);
            }
        }
        let strs = role_strings(node);
        if let Some(count) = declared_descriptor_count(node) {
            declared_descriptor_counts.insert(idx, count);
        }
        let Some(mut first) = fc_promoted_role(&strs, false) else {
            continue;
        };
        if primary_role(&strs) == Some("texture")
            && location_index_with_static(node, idx, &static_int_globals) == u32::MAX
        {
            first = "function_constant";
        }
        if primary_role(&strs) == Some("patch_input") {
            first = "patch_input";
        }
        if let Some(declared) = declared_role(&strs) {
            if !VERTEX_INPUT_ROLES.contains(&declared)
                && metadata_enabled_by_default(node, nodes, &static_int_globals)
            {
                unmodelled_input_params.push((idx, declared.to_string()));
            }
        }
        let role =
            match first {
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
                    if first == "indirect_buffer" {
                        if let Some(sref) = struct_info_ref(node) {
                            indirect_buffer_struct_refs.push((
                                idx,
                                location_index_with_static(node, idx, &static_int_globals),
                                sref,
                            ));
                        }
                    }
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
                    if let Some(size) = i32_after_marker(node, "air.buffer_size") {
                        buffer_object_sizes.insert(idx, size);
                    }
                    if let Some(name) = arg_type_name(node) {
                        buffer_type_names.insert(idx, name);
                    }
                    if let Some(access) = declared_buffer_access(node) {
                        buffer_accesses.insert(idx, access);
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
                "visible_function_table" => VertRole::VisibleFunctionTable(
                    location_index_with_static(node, idx, &static_int_globals),
                ),
                "intersection_function_table" => VertRole::IntersectionFunctionTable(
                    location_index_with_static(node, idx, &static_int_globals),
                ),
                "vertex_id" => VertRole::VertexId,
                "instance_id" => VertRole::InstanceId,
                "patch_control_point_input" => {
                    let refs = refs_in(node);
                    let function = refs
                        .first()
                        .and_then(|reference| nodes.get(reference))
                        .and_then(|body| pointer_symbol(body));
                    let fields = refs
                        .iter()
                        .skip(1)
                        .filter_map(|reference| nodes.get(reference))
                        .map(|field| PatchControlPointField {
                            location: location_index_with_static(field, 0, &static_int_globals),
                            type_name: arg_type_name(field),
                        })
                        .collect::<Vec<_>>();
                    if let Some(function) = function {
                        patch_control_point = Some((function, fields));
                    }
                    VertRole::PatchControlPoints
                }
                "patch_input" => {
                    let location = location_index_with_static(node, idx, &static_int_globals);
                    if let Some(name) = arg_type_name(node) {
                        patch_input_types.insert(location, name);
                    }
                    if let Some(name) = arg_name(node) {
                        patch_input_names.insert(location, name);
                    }
                    VertRole::PatchInput(location)
                }
                "position_in_patch" => VertRole::PositionInPatch,
                "patch_id" => VertRole::PatchId,
                "amplification_id" => VertRole::AmplificationId,
                "amplification_count" => VertRole::AmplificationCount,
                _ => VertRole::Other,
            };
        roles.push((idx, role));
    }
    let top_level_texture_locations = roles
        .iter()
        .filter_map(|(_, role)| match role {
            VertRole::Texture(location) => Some(*location),
            _ => None,
        })
        .collect::<Vec<_>>();
    Some(VertMeta {
        roles,
        unmodelled_input_params,
        implicit_imageblock_attachments: detect_implicit_imageblock_attachments(ll)?,
        parameter_type_names,
        output_roles,
        invariant_outputs,
        unmodelled_output_members,
        output_varying_types,
        output_varying_names,
        output_varying_user_semantics,
        vertex_input_types,
        vertex_input_names,
        patch_input_types,
        patch_input_names,
        buffer_layouts,
        buffer_address_spaces,
        buffer_type_sizes,
        buffer_object_sizes,
        buffer_type_names,
        buffer_accesses,
        texture_type_names,
        declared_descriptor_counts,
        embedded_textures: if body_uses_texture_intrinsic(ll) {
            detect_embedded_textures(
                nodes,
                &indirect_buffer_struct_refs,
                &top_level_texture_locations,
            )
        } else {
            Vec::new()
        },
        embedded_arguments: detect_embedded_arguments(nodes, &indirect_buffer_struct_refs),
        undecoded_patch_shape,
        unmodelled_stage_attributes,
        tessellation: patch_shape.map(|(domain, control_point_count)| {
            let (control_point_function, control_point_fields) = patch_control_point
                .map(|(function, fields)| (Some(function), fields))
                .unwrap_or_default();
            TessellationMeta {
                domain,
                control_point_count,
                control_point_function,
                control_point_fields,
            }
        }),
    })
}

#[cfg(test)]
mod tests;
