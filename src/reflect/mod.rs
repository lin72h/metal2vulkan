//! Public shader-reflection facade.
//!
//! The translator already parses everything a downstream consumer needs to bind a shader — the
//! stage-interface metadata in [`crate::meta`] (`FragMeta`/`VertMeta`/`KernMeta`) plus the fixed
//! descriptor-ABI convention the stage-input/output passes apply (`crate::passes::stage_input` and
//! `crate::passes::stage_output`). Historically
//! that knowledge was DROPPED at the crate boundary: `translate_*` returned bare `Result<Vec<u8>,
//! String>`, forcing consumers to re-reflect the produced SPIR-V and re-hardcode the ABI bases.
//!
//! This module exposes that knowledge as one consumer-shaped [`ShaderReflection`] value, computed as
//! a distilled facade over the parser-shaped meta structs plus the ABI constants below. The binding
//! numbers here are the SAME ones the interface pass decorates the module with (all in descriptor
//! [`RESOURCE_DESCRIPTOR_SET`], via `BASE + metal_index`); this is a pure re-shaping of already-parsed
//! data and never re-reads the emitted SPIR-V, so it is byte-neutral on the translator.

use crate::meta::{
    texture_shape_from_name, AirType, FragMeta, FragRole, FunctionConstant, KernMeta, KernRole,
    TextureComponent, TextureDimension, TextureShape, VertMeta, VertOutRole, VertRole,
};

/// Schema version of [`ShaderReflection`]. Bump on any breaking change to the serialized shape so a
/// consumer's persisted reflection cache invalidates cleanly rather than
/// deserializing stale fields. See plan Workstream M3.
///
/// v2 (cleanup-plan Workstream R): the shape grew the consumer-readiness fields — per-binding typed
/// `texture_shape` (dimension/arrayed/multisampled/component/writable/storage_format) and
/// `embedded_source`; stage-level `vertex_builtins`, `imageblock_layouts`, `function_constants`, and
/// the source `datalayout`; plus fragment/vertex buffer `address_space`/`declared_size` population.
///
/// v3 reports AIR-embedded constexpr samplers as `StaticSampler` bindings with their decoded state.
pub const REFLECTION_VERSION: u32 = 3;

/// The single descriptor set every Metal-facing resource is bound in. The interface pass hardcodes
/// `DescriptorSet 0` for every resource (buffers, textures, samplers, color inputs).
pub const RESOURCE_DESCRIPTOR_SET: u32 = 0;

/// Descriptor binding base for `[[buffer(n)]]` resources: the binding is the Metal buffer index `n`
/// directly (range `0..32`).
pub const BUFFER_BINDING_BASE: u32 = 0;
/// Descriptor binding base for `[[texture(n)]]` resources: binding = `TEXTURE_BINDING_BASE + n`.
pub const TEXTURE_BINDING_BASE: u32 = 32;
/// Descriptor binding base for `[[sampler(n)]]` resources: binding = `SAMPLER_BINDING_BASE + n`.
pub const SAMPLER_BINDING_BASE: u32 = 64;
/// Descriptor binding base for `[[color(n)]]` framebuffer-fetch inputs (Vulkan input attachments):
/// binding = `COLOR_INPUT_BINDING_BASE + n`.
pub const COLOR_INPUT_BINDING_BASE: u32 = 96;

/// AIR address space 2 = constant memory (`constant` / `const device`) — a read-only buffer.
pub const ADDRESS_SPACE_CONSTANT: u32 = 2;
/// AIR address space 3 = threadgroup memory. A `[[buffer(n)]]` in this space becomes a Workgroup
/// `OpVariable` and consumes NO descriptor (its [`ResourceBinding::descriptor`] is `None`).
pub const ADDRESS_SPACE_THREADGROUP: u32 = 3;

/// The shader stage of a reflected module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ShaderStage {
    Vertex,
    Fragment,
    Kernel,
}

impl From<crate::passes::Stage> for ShaderStage {
    fn from(stage: crate::passes::Stage) -> Self {
        match stage {
            crate::passes::Stage::Vertex => ShaderStage::Vertex,
            crate::passes::Stage::Fragment => ShaderStage::Fragment,
            crate::passes::Stage::Kernel => ShaderStage::Kernel,
        }
    }
}

/// The kind of a bound shader resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResourceKind {
    /// A `[[buffer(n)]]` device/constant buffer. Bound at `BUFFER_BINDING_BASE + n`.
    Buffer,
    /// A `[[buffer(n)]]` in threadgroup address space — a Workgroup variable, no descriptor.
    ThreadgroupBuffer,
    /// A `[[texture(n)]]` sampled image. Bound at `TEXTURE_BINDING_BASE + n`.
    Texture,
    /// A runtime-indexed texture descriptor array. Bound at `TEXTURE_BINDING_BASE + n`.
    /// See `ResourceBinding::access` to distinguish sampled from storage arrays.
    TextureArray,
    /// A write-only storage image (`texture` with `access::write`). Bound at `TEXTURE_BINDING_BASE + n`.
    StorageImage,
    /// A `[[sampler(n)]]`. Bound at `SAMPLER_BINDING_BASE + n`.
    Sampler,
    /// An AIR-embedded constexpr sampler. It has no Metal argument index and is populated from
    /// [`ResourceBinding::static_sampler`] at the reflected descriptor location.
    StaticSampler,
    /// A `[[color(n)]]` framebuffer-fetch input (Vulkan input attachment). Bound at
    /// `COLOR_INPUT_BINDING_BASE + n`.
    ColorInput,
    /// A host-populated StorageBuffer shadow of an opaque acceleration structure. Bound at the
    /// resource's Metal buffer index.
    AccelerationStructureShadow,
    /// A texture embedded inside an `air.indirect_buffer` argument buffer, surfaced as a standalone
    /// sampled image. Bound at `TEXTURE_BINDING_BASE + synthetic_index`.
    EmbeddedArgBufferTexture,
}

/// The descriptor location the interface pass decorates a resource with. Absent for resources that
/// consume no descriptor (threadgroup buffers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DescriptorLocation {
    pub set: u32,
    pub binding: u32,
}

/// Per-binding access classification. Populated at translate time from the declared Metal access:
/// texture access from the type-name qualifier (`sample`/`read` → `Sampled`, `write`/`read_write` →
/// `Storage`), and buffer access from the constant address space (`ReadOnly`). A DEVICE buffer's
/// precise read-vs-write requires IR dataflow the facade does not carry, so it stays `None` (the
/// consumer determines it SPIR-V-side). See plan Workstream M2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResourceAccess {
    /// A buffer read but never written by the shader.
    ReadOnly,
    /// A buffer written by the shader.
    ReadWrite,
    /// A sampled texture (`OpTypeImage Sampled=1`), read through a sampler.
    Sampled,
    /// A storage image (`OpTypeImage Sampled=2`), read/written directly.
    Storage,
}

/// Minification or magnification filtering encoded in an AIR constexpr sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SamplerFilter {
    Nearest,
    Linear,
    Bicubic,
}

/// Mipmap filtering encoded in an AIR constexpr sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SamplerMipFilter {
    None,
    Nearest,
    Linear,
}

/// Texture-coordinate addressing encoded in an AIR constexpr sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SamplerAddressMode {
    ClampToZero,
    ClampToEdge,
    Repeat,
    MirroredRepeat,
    ClampToBorder,
}

/// Coordinate convention encoded in an AIR constexpr sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SamplerCoordinates {
    Normalized,
    Pixel,
}

/// Comparison mode encoded in an AIR constexpr sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SamplerCompareFunction {
    None,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    Always,
    Never,
}

/// Border color encoded in an AIR constexpr sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SamplerBorderColor {
    TransparentBlack,
    OpaqueBlack,
    OpaqueWhite,
}

/// Reduction mode encoded in an AIR constexpr sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SamplerReduction {
    WeightedAverage,
    Minimum,
    Maximum,
}

/// Fully decoded AIR constexpr sampler state.
///
/// The raw words remain available for forward-compatible diagnostics. Consumers should use the
/// typed fields and reject unsupported modes rather than reinterpret the AIR ABI themselves.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StaticSamplerState {
    pub min_filter: SamplerFilter,
    pub mag_filter: SamplerFilter,
    pub mip_filter: SamplerMipFilter,
    pub address_mode_s: SamplerAddressMode,
    pub address_mode_t: SamplerAddressMode,
    pub address_mode_r: SamplerAddressMode,
    pub coordinates: SamplerCoordinates,
    pub compare_function: SamplerCompareFunction,
    pub max_anisotropy: u32,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub border_color: SamplerBorderColor,
    pub reduction: SamplerReduction,
    pub lod_bias: f32,
    pub raw_words: [u64; 2],
}

impl StaticSamplerState {
    pub(crate) fn from_air_words(words: [u64; 2]) -> Result<Self, String> {
        let word = words[0];
        let border_color = match (word >> 56) & 0x3 {
            0 => SamplerBorderColor::TransparentBlack,
            1 => SamplerBorderColor::OpaqueBlack,
            2 => SamplerBorderColor::OpaqueWhite,
            value => return Err(format!("unsupported AIR sampler border-color code {value}")),
        };
        let address = |shift: u32| match (word >> shift) & 0x7_u64 {
            0 if border_color != SamplerBorderColor::TransparentBlack => {
                Ok(SamplerAddressMode::ClampToBorder)
            }
            0 => Ok(SamplerAddressMode::ClampToZero),
            1 => Ok(SamplerAddressMode::ClampToEdge),
            2 => Ok(SamplerAddressMode::Repeat),
            3 => Ok(SamplerAddressMode::MirroredRepeat),
            value => Err(format!("unsupported AIR sampler address code {value}")),
        };
        let filter = |shift: u32| match (word >> shift) & 0x3_u64 {
            0 => Ok(SamplerFilter::Nearest),
            1 => Ok(SamplerFilter::Linear),
            2 => Ok(SamplerFilter::Bicubic),
            value => Err(format!("unsupported AIR sampler filter code {value}")),
        };
        let mip_filter = match (word >> 13) & 0x3 {
            0 => SamplerMipFilter::None,
            1 => SamplerMipFilter::Nearest,
            2 => SamplerMipFilter::Linear,
            value => return Err(format!("unsupported AIR sampler mip-filter code {value}")),
        };
        let coordinates = match (word >> 15) & 0x1 {
            0 => SamplerCoordinates::Normalized,
            _ => SamplerCoordinates::Pixel,
        };
        let compare_function = match (word >> 16) & 0xf {
            0 => SamplerCompareFunction::None,
            1 => SamplerCompareFunction::Less,
            2 => SamplerCompareFunction::LessEqual,
            3 => SamplerCompareFunction::Greater,
            4 => SamplerCompareFunction::GreaterEqual,
            5 => SamplerCompareFunction::Equal,
            6 => SamplerCompareFunction::NotEqual,
            7 => SamplerCompareFunction::Always,
            8 => SamplerCompareFunction::Never,
            value => {
                return Err(format!(
                    "unsupported AIR sampler compare-function code {value}"
                ))
            }
        };
        let reduction = match (word >> 58) & 0x3 {
            0 => SamplerReduction::WeightedAverage,
            1 => SamplerReduction::Minimum,
            2 => SamplerReduction::Maximum,
            value => return Err(format!("unsupported AIR sampler reduction code {value}")),
        };
        let min_half = (((word >> 32) & 0xff) as u16) << 8;
        let max_half = ((word >> 40) & 0xffff) as u16;
        let bias_half = (words[1] & 0xffff) as u16;
        Ok(Self {
            min_filter: filter(11)?,
            mag_filter: filter(9)?,
            mip_filter,
            address_mode_s: address(0)?,
            address_mode_t: address(3)?,
            address_mode_r: address(6)?,
            coordinates,
            compare_function,
            max_anisotropy: (((word >> 20) & 0xf) as u32) + 1,
            lod_min_clamp: half_to_f32(min_half),
            lod_max_clamp: half_to_f32(max_half),
            border_color,
            reduction,
            lod_bias: half_to_f32(bias_half),
            raw_words: words,
        })
    }

    pub(crate) fn uses_pixel_nearest(self) -> bool {
        self.uses_pixel_coordinates() && !self.uses_linear_filter()
    }

    pub(crate) fn uses_linear_filter(self) -> bool {
        self.min_filter == SamplerFilter::Linear && self.mag_filter == SamplerFilter::Linear
    }

    pub(crate) fn uses_pixel_coordinates(self) -> bool {
        self.coordinates == SamplerCoordinates::Pixel
    }

    pub(crate) fn spatial_clamps_to_zero(self, dimension: usize) -> bool {
        matches!(
            [
                self.address_mode_s,
                self.address_mode_t,
                self.address_mode_r
            ]
            .get(dimension),
            Some(SamplerAddressMode::ClampToZero)
        )
    }
}

fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 if fraction == 0 => sign,
        0 => {
            let leading = 31 - fraction.leading_zeros();
            let normalized_fraction = (fraction << (10 - leading)) & 0x03ff;
            let exponent = 127 - 14 - (10 - leading);
            sign | (exponent << 23) | (normalized_fraction << 13)
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        _ => sign | (u32::from(exponent) + 112) << 23 | (fraction << 13),
    };
    f32::from_bits(value)
}

/// The argument-buffer source of a translator-synthesized embedded texture: which
/// `air.indirect_buffer` kernel argument carries it and at what byte offset the texture handle sits.
/// Only the translator knows this mapping (the guest stream never names the synthetic texture), so a
/// consumer needs it to source/seed the texture the synthetic binding reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EmbeddedArgBuffer {
    /// Kernel parameter index of the owning `air.indirect_buffer` argument.
    pub buffer_index: u32,
    /// Byte offset of the texture handle within the argument-buffer struct.
    pub field_offset: u32,
}

/// One bound shader resource: its Metal index, descriptor location, and any decoded layout facts.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResourceBinding {
    pub kind: ResourceKind,
    /// The Metal resource index (`n` in `[[buffer(n)]]`/`[[texture(n)]]`/`[[sampler(n)]]`), or the
    /// synthetic index for an embedded argument-buffer texture.
    pub metal_index: u32,
    /// The SPIR-V descriptor location, or `None` for a resource that consumes no descriptor.
    pub descriptor: Option<DescriptorLocation>,
    /// The entry-parameter index this resource came from (SPIR-V `OpFunctionParameter` order), when
    /// applicable. `None` for synthesized resources (embedded argument-buffer textures).
    pub param_index: Option<u32>,
    /// For a buffer: the raw AIR address space (3 = threadgroup). `None` for non-buffers / when absent.
    pub address_space: Option<u32>,
    /// For a buffer: the declared AIR argument byte size, when the metadata carries one.
    pub declared_size: Option<u32>,
    /// For a buffer with `air.struct_type_info`: the reconstructed AIR aggregate layout.
    pub type_layout: Option<AirType>,
    /// The AIR argument type name (`texture2d<uint, read>`, a struct name, `char`, …), when carried.
    pub type_name: Option<String>,
    /// For a texture binding: the decoded shape (dimensionality, arrayed, multisampled, component,
    /// access qualifier), so a consumer need not re-parse the type name or walk the emitted
    /// `OpTypeImage`. `None` for non-textures / when no type name was carried.
    pub texture_shape: Option<TextureShape>,
    /// For an `EmbeddedArgBufferTexture`: the arg-buffer argument + offset it was synthesized from.
    /// `None` for every other binding.
    pub embedded_source: Option<EmbeddedArgBuffer>,
    /// Per-binding access, once Workstream M2 computes it; `None` otherwise.
    pub access: Option<ResourceAccess>,
    /// Decoded AIR state for [`ResourceKind::StaticSampler`]; `None` for every other kind.
    pub static_sampler: Option<StaticSamplerState>,
}

impl ResourceBinding {
    fn descriptor_at(base: u32, metal_index: u32) -> Option<DescriptorLocation> {
        Some(DescriptorLocation {
            set: RESOURCE_DESCRIPTOR_SET,
            binding: base.saturating_add(metal_index),
        })
    }
}

/// A vertex-stage input attribute (`[[attribute(n)]]` / `[[stage_in]]`).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexAttribute {
    /// SPIR-V input Location.
    pub location: u32,
    /// AIR type name (`float2`, `float4`, …), when carried.
    pub type_name: Option<String>,
    /// Metal argument name, when carried.
    pub name: Option<String>,
}

/// A varying: a fragment `[[stage_in]]` input, or a vertex user-varying output, at a Location.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Varying {
    /// SPIR-V Location.
    pub location: u32,
    /// AIR type name, when carried.
    pub type_name: Option<String>,
    /// Metal argument/field name, when carried.
    pub name: Option<String>,
    /// Metal user semantic (`user(texturecoord)`), when carried.
    pub user_semantic: Option<String>,
}

/// Vertex-stage builtin usage: which SPIR-V builtins the entry consumes or writes. Parsed into
/// `VertMeta` roles; a consumer uses these to decide whether to bind vertex/instance index sources
/// and whether the pipeline emits clip position, without walking the emitted module for `OpDecorate
/// BuiltIn`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VertexBuiltins {
    /// The entry reads `[[vertex_id]]` (SPIR-V `VertexIndex`).
    pub uses_vertex_index: bool,
    /// The entry reads `[[instance_id]]` (SPIR-V `InstanceIndex`).
    pub uses_instance_index: bool,
    /// The entry writes `[[position]]` (SPIR-V `Position`).
    pub writes_position: bool,
}

/// A fragment render-target output.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RenderTarget {
    /// Return-struct member index.
    pub member_index: u32,
    /// Color attachment Location.
    pub location: u32,
    /// AIR render-target type name (`float4`, `int4`, …), when carried.
    pub type_name: Option<String>,
}

/// A kernel `[[imageblock]]` threadgroup tile: the entry-parameter index and its reconstructed AIR
/// struct layout. An imageblock consumes NO descriptor (it is threadgroup-local storage), so it is
/// not a `ResourceBinding`; a consumer needs the layout to size the tile.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImageblockLayout {
    /// Entry-parameter index of the imageblock argument.
    pub param_index: u32,
    /// The reconstructed AIR aggregate layout of the imageblock struct.
    pub type_layout: AirType,
}

/// The consumer-shaped reflection of one translated shader. Built as a facade over the parser-shaped
/// [`FragMeta`]/[`VertMeta`]/[`KernMeta`]; every binding number matches what the interface pass
/// decorated the emitted module with.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShaderReflection {
    /// Schema version, always [`REFLECTION_VERSION`] at build time.
    pub reflection_version: u32,
    pub stage: ShaderStage,
    /// The ORIGINAL Metal entry-point function name (the emitted SPIR-V `OpEntryPoint` string is
    /// always `"main"`, so the meaningful identity a consumer keys on is this name).
    pub entry_point: Option<String>,
    /// Every bound resource, in entry-parameter order (synthesized embedded textures last).
    pub bindings: Vec<ResourceBinding>,
    /// Vertex input attributes (vertex stage only).
    pub vertex_attributes: Vec<VertexAttribute>,
    /// Varyings: fragment `[[stage_in]]` inputs, or vertex user-varying outputs.
    pub varyings: Vec<Varying>,
    /// Fragment render-target outputs (fragment stage only).
    pub render_targets: Vec<RenderTarget>,
    /// Fragment return-struct members tagged `[[depth]]`.
    pub depth_members: Vec<u32>,
    /// Fragment return-struct members tagged `[[stencil]]`.
    pub stencil_members: Vec<u32>,
    /// Kernel GLCompute local size (`[x, y, z]`), when the stage is a kernel.
    pub local_size: Option<[u32; 3]>,
    /// Vertex-stage builtin usage (`Some` only for the vertex stage).
    pub vertex_builtins: Option<VertexBuiltins>,
    /// Kernel `[[imageblock]]` threadgroup tiles (kernel stage only), sorted by parameter index.
    pub imageblock_layouts: Vec<ImageblockLayout>,
    /// The source LLVM-IR `target datalayout` string, when the reflected translate started from an
    /// unsanitized module (sanitization strips it). A consumer uses it to lay out struct members
    /// without re-reading the source `.ll`. `None` when translated from already-sanitized IR.
    pub datalayout: Option<String>,
    /// Metal `[[function_constant(N)]]` inventory (index/name/type), so a consumer can discover the
    /// module's spec-ids without scanning SPIR-V. Populated by the reflected translate paths; empty
    /// when reflection is built directly from meta (the `from_*` builders do not scan IR).
    pub function_constants: Vec<FunctionConstant>,
}

impl ShaderReflection {
    /// Build reflection for a fragment shader from its parsed meta and (optional) entry name.
    pub fn from_fragment(meta: &FragMeta, entry_point: Option<&str>) -> Self {
        let mut bindings = Vec::new();
        for (idx, role) in &meta.roles {
            let idx = *idx;
            let binding = match role {
                FragRole::Buffer(n) => ResourceBinding {
                    kind: ResourceKind::Buffer,
                    metal_index: *n,
                    descriptor: ResourceBinding::descriptor_at(BUFFER_BINDING_BASE, *n),
                    param_index: Some(idx),
                    address_space: meta.buffer_address_spaces.get(&idx).copied(),
                    declared_size: meta.buffer_type_sizes.get(&idx).copied(),
                    type_layout: meta.buffer_layouts.get(&idx).cloned(),
                    type_name: None,
                    texture_shape: None,
                    embedded_source: None,
                    access: None,
                    static_sampler: None,
                },
                FragRole::Texture(n) => texture_binding(*n, Some(idx), &meta.texture_type_names),
                FragRole::Sampler(n) => sampler_binding(*n, Some(idx)),
                FragRole::ColorInput(n) => ResourceBinding {
                    kind: ResourceKind::ColorInput,
                    metal_index: *n,
                    descriptor: ResourceBinding::descriptor_at(COLOR_INPUT_BINDING_BASE, *n),
                    param_index: Some(idx),
                    address_space: None,
                    declared_size: None,
                    type_layout: None,
                    type_name: None,
                    texture_shape: None,
                    embedded_source: None,
                    access: None,
                    static_sampler: None,
                },
                FragRole::Position
                | FragRole::PointCoord
                | FragRole::FrontFacing
                | FragRole::PrimitiveId
                | FragRole::SampleId
                | FragRole::ViewportArrayIndex
                | FragRole::Varying(_)
                | FragRole::Other => {
                    continue;
                }
            };
            bindings.push(binding);
        }
        let varyings = meta
            .varying_types
            .keys()
            .chain(meta.varying_names.keys())
            .chain(meta.varying_user_semantics.keys())
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|loc| Varying {
                location: loc,
                type_name: meta.varying_types.get(&loc).cloned(),
                name: meta.varying_names.get(&loc).cloned(),
                user_semantic: meta.varying_user_semantics.get(&loc).cloned(),
            })
            .collect();
        let render_targets = meta
            .render_target_members
            .iter()
            .map(|(member, location)| RenderTarget {
                member_index: *member,
                location: *location,
                type_name: meta.render_target_type_names.get(member).cloned(),
            })
            .collect();
        ShaderReflection {
            reflection_version: REFLECTION_VERSION,
            stage: ShaderStage::Fragment,
            entry_point: entry_point.map(str::to_string),
            bindings,
            vertex_attributes: Vec::new(),
            varyings,
            render_targets,
            depth_members: meta.depth_members.clone(),
            stencil_members: meta.stencil_members.clone(),
            local_size: None,
            vertex_builtins: None,
            imageblock_layouts: Vec::new(),
            datalayout: None,
            function_constants: Vec::new(),
        }
    }

    /// Build reflection for a vertex shader from its parsed meta and (optional) entry name.
    pub fn from_vertex(meta: &VertMeta, entry_point: Option<&str>) -> Self {
        let vertex_builtins = VertexBuiltins {
            uses_vertex_index: meta.roles.iter().any(|(_, r)| *r == VertRole::VertexId),
            uses_instance_index: meta.roles.iter().any(|(_, r)| *r == VertRole::InstanceId),
            writes_position: meta.output_roles.contains(&VertOutRole::Position),
        };
        let mut bindings = Vec::new();
        for (idx, role) in &meta.roles {
            let idx = *idx;
            let binding = match role {
                VertRole::Buffer(n) => ResourceBinding {
                    kind: ResourceKind::Buffer,
                    metal_index: *n,
                    descriptor: ResourceBinding::descriptor_at(BUFFER_BINDING_BASE, *n),
                    param_index: Some(idx),
                    address_space: meta.buffer_address_spaces.get(&idx).copied(),
                    declared_size: meta.buffer_type_sizes.get(&idx).copied(),
                    type_layout: meta.buffer_layouts.get(&idx).cloned(),
                    type_name: None,
                    texture_shape: None,
                    embedded_source: None,
                    access: None,
                    static_sampler: None,
                },
                VertRole::Texture(n) => texture_binding(*n, Some(idx), &meta.texture_type_names),
                VertRole::Sampler(n) => sampler_binding(*n, Some(idx)),
                VertRole::VertexInput(_)
                | VertRole::VertexId
                | VertRole::InstanceId
                | VertRole::Other => continue,
            };
            bindings.push(binding);
        }
        let vertex_attributes = meta
            .vertex_input_types
            .keys()
            .chain(meta.vertex_input_names.keys())
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|loc| VertexAttribute {
                location: loc,
                type_name: meta.vertex_input_types.get(&loc).cloned(),
                name: meta.vertex_input_names.get(&loc).cloned(),
            })
            .collect();
        let varyings = meta
            .output_roles
            .iter()
            .filter_map(|role| match role {
                VertOutRole::Varying(loc) => Some(Varying {
                    location: *loc,
                    type_name: None,
                    name: None,
                    user_semantic: None,
                }),
                _ => None,
            })
            .collect();
        ShaderReflection {
            reflection_version: REFLECTION_VERSION,
            stage: ShaderStage::Vertex,
            entry_point: entry_point.map(str::to_string),
            bindings,
            vertex_attributes,
            varyings,
            render_targets: Vec::new(),
            depth_members: Vec::new(),
            stencil_members: Vec::new(),
            local_size: None,
            vertex_builtins: Some(vertex_builtins),
            imageblock_layouts: Vec::new(),
            datalayout: None,
            function_constants: Vec::new(),
        }
    }

    /// Build reflection for a compute kernel from its parsed meta, entry name, and local size.
    pub fn from_kernel(meta: &KernMeta, entry_point: Option<&str>, local_size: [u32; 3]) -> Self {
        let mut bindings = Vec::new();
        for (idx, role) in &meta.roles {
            let idx = *idx;
            let binding = match role {
                KernRole::Buffer(n) => {
                    let address_space = meta.buffer_address_spaces.get(&idx).copied();
                    let threadgroup = address_space == Some(ADDRESS_SPACE_THREADGROUP);
                    let kind = if threadgroup {
                        ResourceKind::ThreadgroupBuffer
                    } else {
                        ResourceKind::Buffer
                    };
                    ResourceBinding {
                        kind,
                        metal_index: *n,
                        descriptor: if threadgroup {
                            None
                        } else {
                            ResourceBinding::descriptor_at(BUFFER_BINDING_BASE, *n)
                        },
                        param_index: Some(idx),
                        address_space,
                        declared_size: meta.buffer_type_sizes.get(&idx).copied(),
                        type_layout: meta.buffer_layouts.get(&idx).cloned(),
                        type_name: meta.buffer_type_names.get(&idx).cloned(),
                        texture_shape: None,
                        embedded_source: None,
                        access: buffer_access_from_address_space(address_space),
                        static_sampler: None,
                    }
                }
                KernRole::Texture(n) => texture_binding(*n, Some(idx), &meta.texture_type_names),
                KernRole::Sampler(n) => sampler_binding(*n, Some(idx)),
                KernRole::AccelerationStructureShadow(n) => ResourceBinding {
                    kind: ResourceKind::AccelerationStructureShadow,
                    metal_index: *n,
                    descriptor: ResourceBinding::descriptor_at(BUFFER_BINDING_BASE, *n),
                    param_index: Some(idx),
                    address_space: None,
                    declared_size: None,
                    type_layout: None,
                    type_name: None,
                    texture_shape: None,
                    embedded_source: None,
                    access: None,
                    static_sampler: None,
                },
                _ => continue,
            };
            bindings.push(binding);
        }
        for embedded in &meta.embedded_textures {
            bindings.push(ResourceBinding {
                kind: ResourceKind::EmbeddedArgBufferTexture,
                metal_index: embedded.synthetic_texture_index,
                descriptor: ResourceBinding::descriptor_at(
                    TEXTURE_BINDING_BASE,
                    embedded.synthetic_texture_index,
                ),
                param_index: None,
                address_space: None,
                declared_size: None,
                type_layout: None,
                type_name: None,
                texture_shape: Some(TextureShape {
                    dimension: TextureDimension::from_spirv_dim(embedded.dim),
                    arrayed: false,
                    multisampled: false,
                    component: TextureComponent::from_image_comp(embedded.comp),
                    writable: false,
                    array_ref: false,
                    storage_format: None,
                }),
                embedded_source: Some(EmbeddedArgBuffer {
                    buffer_index: embedded.buffer_index,
                    field_offset: embedded.field_offset,
                }),
                access: Some(ResourceAccess::Sampled),
                static_sampler: None,
            });
        }
        ShaderReflection {
            reflection_version: REFLECTION_VERSION,
            stage: ShaderStage::Kernel,
            entry_point: entry_point.map(str::to_string),
            bindings,
            vertex_attributes: Vec::new(),
            varyings: Vec::new(),
            render_targets: Vec::new(),
            depth_members: Vec::new(),
            stencil_members: Vec::new(),
            local_size: Some(local_size),
            vertex_builtins: None,
            imageblock_layouts: {
                let mut ibs: Vec<ImageblockLayout> = meta
                    .imageblock_layouts
                    .iter()
                    .map(|(idx, ty)| ImageblockLayout {
                        param_index: *idx,
                        type_layout: ty.clone(),
                    })
                    .collect();
                ibs.sort_by_key(|ib| ib.param_index);
                ibs
            },
            datalayout: None,
            function_constants: Vec::new(),
        }
    }

    /// The binding for a given resource kind + Metal index, if present.
    pub fn binding_at(&self, kind: ResourceKind, metal_index: u32) -> Option<&ResourceBinding> {
        self.bindings
            .iter()
            .find(|b| b.kind == kind && b.metal_index == metal_index)
    }

    pub(crate) fn add_static_samplers(&mut self, ll: &str) -> Result<(), String> {
        let constants = parse_static_sampler_constants(ll)?;
        if constants.is_empty() {
            return Ok(());
        }
        let mut occupied = self
            .bindings
            .iter()
            .filter_map(|binding| binding.descriptor.map(|descriptor| descriptor.binding))
            .collect::<std::collections::BTreeSet<_>>();
        for words in constants {
            let binding = (SAMPLER_BINDING_BASE..COLOR_INPUT_BINDING_BASE)
                .find(|binding| !occupied.contains(binding))
                .ok_or_else(|| {
                    format!(
                        "AIR constexpr sampler count exceeds descriptor band \
                         [{SAMPLER_BINDING_BASE},{COLOR_INPUT_BINDING_BASE})"
                    )
                })?;
            occupied.insert(binding);
            self.bindings.push(ResourceBinding {
                kind: ResourceKind::StaticSampler,
                metal_index: binding - SAMPLER_BINDING_BASE,
                descriptor: Some(DescriptorLocation {
                    set: RESOURCE_DESCRIPTOR_SET,
                    binding,
                }),
                param_index: None,
                address_space: None,
                declared_size: None,
                type_layout: None,
                type_name: None,
                texture_shape: None,
                embedded_source: None,
                access: None,
                static_sampler: Some(StaticSamplerState::from_air_words(words)?),
            });
        }
        Ok(())
    }
}

fn parse_static_sampler_constants(ll: &str) -> Result<Vec<[u64; 2]>, String> {
    let mut globals = std::collections::HashMap::<String, [u64; 2]>::new();
    let mut nodes = std::collections::HashMap::<u32, String>::new();
    let mut root = None;

    for raw in ll.lines() {
        let line = raw.trim();
        if line.starts_with('@') && line.contains(" constant ") {
            let Some((name, _)) = line.split_once(" = ") else {
                continue;
            };
            let mut values = line.split("i64 ").skip(1).filter_map(|tail| {
                tail.split(|ch: char| ch == ',' || ch == ']' || ch.is_whitespace())
                    .find(|token| !token.is_empty())
                    .and_then(|token| token.parse::<i64>().ok())
                    .map(|value| value as u64)
            });
            if let Some(first) = values.next() {
                globals.insert(name.to_string(), [first, values.next().unwrap_or(0)]);
            }
            continue;
        }
        if let Some(body) = line.strip_prefix("!air.sampler_states = !{") {
            root = Some(metadata_refs(body));
            continue;
        }
        let Some(rest) = line.strip_prefix('!') else {
            continue;
        };
        let Some((id, body)) = rest.split_once(" = !{") else {
            continue;
        };
        if let Ok(id) = id.parse::<u32>() {
            nodes.insert(id, body.trim_end_matches('}').to_string());
        }
    }

    let Some(root) = root else {
        return Ok(Vec::new());
    };
    let mut constants = Vec::with_capacity(root.len());
    for node_id in root {
        let body = nodes
            .get(&node_id)
            .ok_or_else(|| format!("AIR sampler-state metadata node !{node_id} is missing"))?;
        if !body.contains("!\"air.sampler_state\"") {
            return Err(format!(
                "AIR sampler-state root references non-sampler node !{node_id}"
            ));
        }
        let name = global_name(body)
            .ok_or_else(|| format!("AIR sampler-state node !{node_id} has no global"))?;
        let words = globals
            .get(name)
            .copied()
            .ok_or_else(|| format!("AIR sampler-state global {name} has no i64 initializer"))?;
        constants.push((name.to_string(), words));
    }
    constants.sort_by_key(|(name, _)| static_sampler_name_order(name));
    Ok(constants.into_iter().map(|(_, words)| words).collect())
}

fn metadata_refs(body: &str) -> Vec<u32> {
    body.split('!')
        .skip(1)
        .filter_map(|tail| {
            let digits = tail
                .chars()
                .take_while(|character| character.is_ascii_digit())
                .collect::<String>();
            (!digits.is_empty())
                .then(|| digits.parse::<u32>().ok())
                .flatten()
        })
        .collect()
}

fn global_name(body: &str) -> Option<&str> {
    let start = body.find('@')?;
    let rest = &body[start..];
    let end = rest
        .find(|character: char| {
            character.is_whitespace() || matches!(character, ',' | ')' | ']' | '}')
        })
        .unwrap_or(rest.len());
    Some(&rest[..end])
}

fn static_sampler_name_order(name: &str) -> u64 {
    name.trim_start_matches('@')
        .strip_prefix("__air_sampler_state")
        .and_then(|suffix| suffix.strip_prefix('.'))
        .and_then(|suffix| suffix.parse::<u64>().ok())
        .unwrap_or(0)
}

fn texture_binding(
    n: u32,
    param_index: Option<u32>,
    type_names: &std::collections::HashMap<u32, String>,
) -> ResourceBinding {
    let type_name = param_index.and_then(|idx| type_names.get(&idx).cloned());
    let texture_shape = type_name.as_deref().map(texture_shape_from_name);
    let (kind, access) = classify_texture(texture_shape.as_ref());
    ResourceBinding {
        kind,
        metal_index: n,
        descriptor: ResourceBinding::descriptor_at(TEXTURE_BINDING_BASE, n),
        param_index,
        address_space: None,
        declared_size: None,
        type_layout: None,
        type_name,
        texture_shape,
        embedded_source: None,
        access: Some(access),
        static_sampler: None,
    }
}

/// Classify a texture argument from its decoded [`TextureShape`], matching the interface pass's
/// `texture_arg_storage` (M2). A `write`/`read_write` texture lowers to a storage image; texture
/// handle arrays are runtime-indexed descriptor arrays whose access follows the inner texture
/// qualifier; everything else is a sampled image. This is the DECLARED access — the authoritative
/// Metal qualifier — so it is translate-time exact for a top-level texture argument. No decoded
/// shape (no type name carried) falls back to a plain sampled 2D texture.
fn classify_texture(shape: Option<&TextureShape>) -> (ResourceKind, ResourceAccess) {
    let Some(shape) = shape else {
        return (ResourceKind::Texture, ResourceAccess::Sampled);
    };
    if shape.array_ref {
        (
            ResourceKind::TextureArray,
            if shape.writable {
                ResourceAccess::Storage
            } else {
                ResourceAccess::Sampled
            },
        )
    } else if shape.writable {
        (ResourceKind::StorageImage, ResourceAccess::Storage)
    } else {
        (ResourceKind::Texture, ResourceAccess::Sampled)
    }
}

/// Access for a buffer given its raw AIR address space: the CONSTANT space is read-only. A DEVICE
/// buffer may be written, and proving it is not requires IR dataflow the facade does not carry — so
/// its access stays `None` (the consumer determines it SPIR-V-side). Only kernel metadata carries
/// address spaces; fragment/vertex buffers report `None`.
fn buffer_access_from_address_space(address_space: Option<u32>) -> Option<ResourceAccess> {
    match address_space {
        Some(ADDRESS_SPACE_CONSTANT) => Some(ResourceAccess::ReadOnly),
        _ => None,
    }
}

fn sampler_binding(n: u32, param_index: Option<u32>) -> ResourceBinding {
    ResourceBinding {
        kind: ResourceKind::Sampler,
        metal_index: n,
        descriptor: ResourceBinding::descriptor_at(SAMPLER_BINDING_BASE, n),
        param_index,
        address_space: None,
        declared_size: None,
        type_layout: None,
        type_name: None,
        texture_shape: None,
        embedded_source: None,
        access: None,
        static_sampler: None,
    }
}

#[cfg(test)]
mod tests;
