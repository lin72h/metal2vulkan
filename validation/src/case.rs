use crate::hash::sha256_bytes;
use crate::jsonl::to_sorted_json_string;
use base64::Engine as _;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AuthoredCase {
    pub air_sha256: String,
    pub case_id: String,
    pub name: String,
    pub entry: String,
    pub stage: Stage,
    #[serde(default)]
    pub buffers: Vec<BufferResource>,
    #[serde(default)]
    pub argument_buffer_buffers: Vec<ArgumentBufferBufferResource>,
    #[serde(default)]
    pub threadgroup_memory: Vec<ThreadgroupMemoryResource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imageblock: Option<ImageblockResource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fragment_imageblock: Option<FragmentImageblockResource>,
    #[serde(default)]
    pub acceleration_structures: Vec<AccelerationStructureResource>,
    #[serde(default)]
    pub visible_function_references: Vec<LinkedFunctionResource>,
    #[serde(default)]
    pub visible_function_tables: Vec<FunctionTableResource>,
    #[serde(default)]
    pub intersection_function_tables: Vec<IntersectionFunctionTableResource>,
    #[serde(default)]
    pub argument_buffer_intersection_function_tables:
        Vec<ArgumentBufferIntersectionFunctionTableResource>,
    #[serde(default)]
    pub textures: Vec<TextureResource>,
    #[serde(default)]
    pub texture_arrays: Vec<TextureArrayResource>,
    #[serde(default)]
    pub argument_buffer_textures: Vec<ArgumentBufferTextureResource>,
    #[serde(default)]
    pub samplers: Vec<SamplerResource>,
    #[serde(default)]
    pub render_targets: Vec<RenderTargetResource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_stencil: Option<DepthStencilResource>,
    #[serde(default)]
    pub vertex_inputs: Vec<VertexInput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vertex_observation: Option<VertexObservation>,
    #[serde(default)]
    pub kernel_stage_inputs: Vec<KernelStageInput>,
    #[serde(default)]
    pub function_constants: Vec<FunctionConstant>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dispatch: Option<Dispatch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draw: Option<Draw>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tessellation: Option<TessellationDraw>,
    pub output: OutputSelection,
    pub compare: Comparison,
    pub execution_safety: ExecutionSafety,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authored_by: Option<String>,
}

#[cfg(test)]
pub(crate) fn combined_depth_stencil_test_case(air_sha256: String, entry: String) -> AuthoredCase {
    AuthoredCase {
        air_sha256,
        case_id: "test-fragment-depth-stencil".into(),
        name: "fragment-depth-stencil-smoke".into(),
        entry,
        stage: Stage::Fragment,
        buffers: vec![],
        argument_buffer_buffers: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: Some(DepthStencilResource {
            dimensions: [1, 1],
            initial_depth_b64: Some("AACAPw==".into()),
            initial_stencil_b64: Some("qw==".into()),
        }),
        vertex_inputs: vec![],
        vertex_observation: None,
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: None,
        draw: Some(Draw {
            primitive: Primitive::Triangle,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
        }),
        tessellation: None,
        output: OutputSelection::Stencil {
            origin: [0, 0],
            dimensions: [1, 1],
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        rationale: None,
        authored_by: Some("codex:gpt-5.6-sol".into()),
    }
}

#[cfg(test)]
pub(crate) fn vector_function_constant_test_case(
    air_sha256: String,
    entry: String,
) -> AuthoredCase {
    AuthoredCase {
        air_sha256,
        case_id: "test-vector-function-constant".into(),
        name: "vector-function-constant-smoke".into(),
        entry,
        stage: Stage::Kernel,
        buffers: vec![BufferResource {
            binding: 0,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some("q6urqw==".into()),
        }],
        argument_buffer_buffers: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        kernel_stage_inputs: vec![],
        function_constants: vec![FunctionConstant {
            index: 0,
            scalar_type: ScalarType::U32,
            lanes: 4,
            bytes_b64: "AQAAAAIAAAADAAAABAAAAA==".into(),
        }],
        dispatch: Some(Dispatch {
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
        }),
        draw: None,
        tessellation: None,
        output: OutputSelection::Buffer {
            binding: 0,
            offset: 0,
            length: 4,
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        rationale: None,
        authored_by: Some("codex:gpt-5.6-sol".into()),
    }
}

#[cfg(test)]
pub(crate) fn narrow_implicit_imageblock_test_case(
    air_sha256: String,
    entry: String,
) -> AuthoredCase {
    AuthoredCase {
        air_sha256,
        case_id: "test-narrow-implicit-imageblock".into(),
        name: "narrow-implicit-imageblock-smoke".into(),
        entry,
        stage: Stage::Kernel,
        buffers: vec![],
        argument_buffer_buffers: vec![],
        threadgroup_memory: vec![],
        imageblock: Some(ImageblockResource {
            dimensions: [16, 16],
            implicit_coverage: Some(ImplicitImageblockCoverage::FullSingleSample),
        }),
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        samplers: vec![],
        render_targets: vec![RenderTargetResource {
            index: 0,
            format: TextureFormat::Rg16Float,
            dimensions: [16, 16],
            initial_bytes_b64: base64::engine::general_purpose::STANDARD
                .encode([0x00, 0x3c, 0x00, 0x40].repeat(16 * 16)),
        }],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: Some(Dispatch {
            grid: [16, 16, 1],
            threads_per_threadgroup: [16, 16, 1],
        }),
        draw: None,
        tessellation: None,
        output: OutputSelection::RenderTarget {
            index: 0,
            origin: [0, 0],
            dimensions: [1, 1],
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        rationale: None,
        authored_by: Some("codex:gpt-5.6-sol".into()),
    }
}

#[cfg(test)]
pub(crate) fn fragment_imageblock_test_case(air_sha256: String, entry: String) -> AuthoredCase {
    let initial_half = base64::engine::general_purpose::STANDARD.encode([0x00, 0x3c]);
    AuthoredCase {
        air_sha256,
        case_id: "test-fragment-imageblock".into(),
        name: "fragment-imageblock-smoke".into(),
        entry,
        stage: Stage::Fragment,
        buffers: vec![],
        argument_buffer_buffers: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: Some(FragmentImageblockResource {
            dimensions: [1, 1],
            members: vec![FragmentImageblockMemberResource {
                semantic: "user(depth)".into(),
                format: FragmentImageblockFormat::Half,
                role: ResourceRole::InOut,
                bytes_b64: None,
                initial_bytes_b64: Some(initial_half),
            }],
        }),
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        samplers: vec![],
        render_targets: vec![RenderTargetResource {
            index: 0,
            format: TextureFormat::Rgba16Float,
            dimensions: [1, 1],
            initial_bytes_b64: base64::engine::general_purpose::STANDARD.encode([0u8; 8]),
        }],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: None,
        draw: Some(Draw {
            primitive: Primitive::Triangle,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
        }),
        tessellation: None,
        output: OutputSelection::FragmentImageblock {
            semantic: "user(depth)".into(),
            origin: [0, 0],
            dimensions: [1, 1],
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        rationale: None,
        authored_by: Some("codex:gpt-5.6-sol".into()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Kernel,
    Vertex,
    Fragment,
}

impl Stage {
    pub fn product(self) -> metal2vulkan::passes::Stage {
        match self {
            Self::Kernel => metal2vulkan::passes::Stage::Kernel,
            Self::Vertex => metal2vulkan::passes::Stage::Vertex,
            Self::Fragment => metal2vulkan::passes::Stage::Fragment,
        }
    }

    pub fn metadata_label(self) -> &'static str {
        match self {
            Self::Kernel => "Kernel",
            Self::Vertex => "Vertex",
            Self::Fragment => "Fragment",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ResourceRole {
    Input,
    Output,
    InOut,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BufferResource {
    pub binding: u32,
    pub role: ResourceRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_bytes_b64: Option<String>,
}

/// A device-buffer handle encoded inside an `air.indirect_buffer` argument. Its identity is the
/// owner buffer plus field offset; product reflection supplies the Metal argument index.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArgumentBufferBufferResource {
    pub buffer_binding: u32,
    pub field_offset: u32,
    pub role: ResourceRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_bytes_b64: Option<String>,
}

/// A dynamically sized Metal `threadgroup` buffer argument.
///
/// It occupies the Metal buffer-index namespace but is emitted as descriptor-free Vulkan
/// `Workgroup` storage. Contents are deliberately not authorable: both APIs leave this memory
/// uninitialized, so a meaningful case must initialize every byte it reads in the shader.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ThreadgroupMemoryResource {
    pub binding: u32,
    pub length: u32,
}

/// Compute imageblock dimensions and, for implicit layouts, the authored raster coverage that
/// makes attachment writes persistent. Explicit imageblock storage remains shader-initialized and
/// therefore has no host bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImageblockResource {
    pub dimensions: [u32; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub implicit_coverage: Option<ImplicitImageblockCoverage>,
}

/// Per-pixel custom fragment imageblock planes. Members are identified by the stable AIR user
/// semantic shared by the master and its entry projections, never by a source struct/member name.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FragmentImageblockResource {
    pub dimensions: [u32; 2],
    pub members: Vec<FragmentImageblockMemberResource>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FragmentImageblockMemberResource {
    pub semantic: String,
    #[serde(default, skip_serializing_if = "FragmentImageblockFormat::is_half")]
    pub format: FragmentImageblockFormat,
    pub role: ResourceRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_bytes_b64: Option<String>,
}

/// Exact scalar/vector texel representation of one custom fragment-imageblock plane.
///
/// This mirrors the product's structurally supported AIR master-member types. Keeping it in the
/// authored literal makes byte extents and backend formats explicit without relying on source
/// member names or guessing from payload length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FragmentImageblockFormat {
    #[default]
    Half,
    Half4,
    Uchar4,
    Ushort,
}

impl FragmentImageblockFormat {
    pub fn from_air_type(type_name: &str, size: u32) -> Option<Self> {
        match (type_name, size) {
            ("half", 2) => Some(Self::Half),
            ("half4", 8) => Some(Self::Half4),
            ("uchar4", 4) => Some(Self::Uchar4),
            ("ushort", 2) => Some(Self::Ushort),
            _ => None,
        }
    }

    pub fn air_type_name(self) -> &'static str {
        match self {
            Self::Half => "half",
            Self::Half4 => "half4",
            Self::Uchar4 => "uchar4",
            Self::Ushort => "ushort",
        }
    }

    pub fn byte_size(self) -> usize {
        self.texture_format().bytes_per_pixel()
    }

    pub fn texture_format(self) -> TextureFormat {
        match self {
            Self::Half => TextureFormat::R16Float,
            Self::Half4 => TextureFormat::Rgba16Float,
            Self::Uchar4 => TextureFormat::Rgba8Uint,
            Self::Ushort => TextureFormat::R16Uint,
        }
    }

    pub fn is_half(&self) -> bool {
        *self == Self::Half
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ImplicitImageblockCoverage {
    /// A deterministic fullscreen triangle covers every pixel and the sole raster sample before
    /// the authored tile kernel runs. Color writes are disabled during this coverage prepass, so
    /// the authored render-target bytes remain the imageblock's initial values.
    FullSingleSample,
}

/// A literal acceleration structure shared by the Metal oracle and Vulkan candidate.
///
/// Instance resources use the existing host-defined child-reference shadow ABI. Primitive
/// resources carry tightly packed little-endian `float3` triangle vertices (nine floats, or 36
/// bytes, per triangle), from which Metal builds the native BLAS and Vulkan builds its shadow.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AccelerationStructureResource {
    pub binding: u32,
    #[serde(
        default,
        skip_serializing_if = "AccelerationStructureKind::is_instance"
    )]
    pub kind: AccelerationStructureKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub primitive_triangles_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub child_references: Vec<u64>,
}

/// An explicitly sized Metal function table with zero or more populated slots.
///
/// Every populated entry resolves a retained, hash-identified non-entry AIR module and exact LLVM
/// function symbol from the same harvested metallib provenance as the stage entry. Indices absent
/// from `entries` are authored null slots. The index is the value consumed by AIR table lookup
/// intrinsics.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FunctionTableResource {
    pub binding: u32,
    /// Total table capacity, including explicitly null slots.
    pub size: u32,
    pub entries: Vec<FunctionTableEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FunctionTableEntry {
    pub index: u32,
    pub module_sha256: String,
    pub function: String,
}

/// One logical function named by AIR's direct `air.visible_function_reference` metadata.
/// The dependency module is retained separately in the private library-module store.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LinkedFunctionResource {
    pub module_sha256: String,
    pub function: String,
}

/// An intersection-function table can contain linked procedural callbacks or Metal's explicit
/// opaque-triangle sentinel. The latter is a real table entry, not a missing/null slot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IntersectionFunctionTableResource {
    pub binding: u32,
    pub size: u32,
    pub entries: Vec<IntersectionFunctionTableEntry>,
}

/// An intersection-function table handle encoded inside an `air.indirect_buffer` argument.
/// Its stable identity is the owning Metal buffer binding plus the member's byte offset; the AIR
/// metadata supplies the backend argument-encoder index.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArgumentBufferIntersectionFunctionTableResource {
    pub buffer_binding: u32,
    pub field_offset: u32,
    pub size: u32,
    pub entries: Vec<IntersectionFunctionTableEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum IntersectionFunctionTableEntry {
    Linked {
        index: u32,
        module_sha256: String,
        function: String,
    },
    OpaqueTriangle {
        index: u32,
        signature: Vec<IntersectionFunctionSignature>,
    },
}

impl IntersectionFunctionTableEntry {
    pub fn index(&self) -> u32 {
        match self {
            Self::Linked { index, .. } | Self::OpaqueTriangle { index, .. } => *index,
        }
    }
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum IntersectionFunctionSignature {
    Instancing,
    TriangleData,
    WorldSpaceData,
    InstanceMotion,
    PrimitiveMotion,
    ExtendedLimits,
    MaxLevels,
    IntersectionFunctionBuffer,
    UserData,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AccelerationStructureKind {
    #[default]
    Instance,
    Primitive,
}

impl AccelerationStructureKind {
    fn is_instance(&self) -> bool {
        *self == Self::Instance
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TextureResource {
    pub binding: u32,
    pub role: ResourceRole,
    pub texture_type: TextureType,
    pub format: TextureFormat,
    pub dimensions: [u32; 3],
    #[serde(default = "default_one", skip_serializing_if = "is_one")]
    pub sample_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_bytes_b64: Option<String>,
}

/// A texture handle encoded inside an `air.indirect_buffer` argument.
///
/// The structural identity is the owning Metal buffer binding plus field byte offset. Product
/// reflection supplies the backend-specific Metal argument index and synthetic Vulkan descriptor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArgumentBufferTextureResource {
    pub buffer_binding: u32,
    pub field_offset: u32,
    pub role: ResourceRole,
    pub texture_type: TextureType,
    pub format: TextureFormat,
    pub dimensions: [u32; 3],
    #[serde(default = "default_one", skip_serializing_if = "is_one")]
    pub sample_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_bytes_b64: Option<String>,
}

/// A Metal texture-handle array. Element order is the descriptor/Metal handle index used by AIR.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TextureArrayResource {
    pub binding: u32,
    pub role: ResourceRole,
    pub texture_type: TextureType,
    pub format: TextureFormat,
    #[serde(default = "default_one", skip_serializing_if = "is_one")]
    pub sample_count: u32,
    pub elements: Vec<TextureArrayElement>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TextureArrayElement {
    pub dimensions: [u32; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_bytes_b64: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TextureType {
    Buffer,
    D1,
    D1Array,
    D2,
    D2Array,
    D2Multisample,
    D2MultisampleArray,
    D3,
    Cube,
    CubeArray,
}

const fn default_one() -> u32 {
    1
}

const fn is_one(value: &u32) -> bool {
    *value == 1
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TextureFormat {
    R8Unorm,
    Rgba8Unorm,
    Rgba8Uint,
    Rgba8Sint,
    R16Float,
    R16Uint,
    Rg16Float,
    Rgba16Float,
    Rgba16Uint,
    R32Uint,
    R32Sint,
    R32Float,
    Rgba32Uint,
    Rgba32Sint,
    Rgba32Float,
    Depth32Float,
}

impl TextureFormat {
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::R8Unorm => 1,
            Self::R16Float | Self::R16Uint => 2,
            Self::Rgba8Unorm
            | Self::Rgba8Uint
            | Self::Rgba8Sint
            | Self::R32Uint
            | Self::R32Sint
            | Self::R32Float => 4,
            Self::Rg16Float => 4,
            Self::Rgba16Float | Self::Rgba16Uint => 8,
            Self::Rgba32Uint | Self::Rgba32Sint | Self::Rgba32Float => 16,
            Self::Depth32Float => 4,
        }
    }

    fn runtime_storage_specialization(
        self,
    ) -> Result<metal2vulkan::reflect::RuntimeStorageImageState, String> {
        use metal2vulkan::reflect::{
            RuntimeStorageImageCapabilities, RuntimeStorageImageFormat, RuntimeStorageImageState,
        };
        let format = match self {
            Self::R8Unorm => RuntimeStorageImageFormat::R8Unorm,
            Self::Rgba8Unorm => RuntimeStorageImageFormat::Rgba8Unorm,
            Self::Rgba8Uint => RuntimeStorageImageFormat::Rgba8Uint,
            Self::Rgba8Sint => RuntimeStorageImageFormat::Rgba8Sint,
            Self::R16Float => RuntimeStorageImageFormat::R16Float,
            Self::R16Uint => RuntimeStorageImageFormat::R16Uint,
            Self::Rg16Float => RuntimeStorageImageFormat::Rg16Float,
            Self::Rgba16Float => RuntimeStorageImageFormat::Rgba16Float,
            Self::Rgba16Uint => RuntimeStorageImageFormat::Rgba16Uint,
            Self::R32Uint => RuntimeStorageImageFormat::R32Uint,
            Self::R32Sint => RuntimeStorageImageFormat::R32Sint,
            Self::R32Float => RuntimeStorageImageFormat::R32Float,
            Self::Rgba32Uint => RuntimeStorageImageFormat::Rgba32Uint,
            Self::Rgba32Sint => RuntimeStorageImageFormat::Rgba32Sint,
            Self::Rgba32Float => RuntimeStorageImageFormat::Rgba32Float,
            Self::Depth32Float => {
                return Err("depth formats cannot specialize a storage image".into());
            }
        };
        Ok(RuntimeStorageImageState {
            format,
            capabilities: RuntimeStorageImageCapabilities {
                storage_image: true,
                storage_image_atomic: matches!(self, Self::R32Uint | Self::R32Sint),
                read_without_format: false,
                write_without_format: false,
            },
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SamplerResource {
    pub binding: u32,
    pub address_mode: SamplerAddressMode,
    pub min_filter: SamplerFilter,
    pub mag_filter: SamplerFilter,
    pub mip_filter: SamplerMipFilter,
    pub normalized_coordinates: bool,
}

impl SamplerResource {
    pub(crate) fn runtime_specialization(&self) -> metal2vulkan::reflect::RuntimeSamplerState {
        use metal2vulkan::reflect::{
            RuntimeSamplerState, SamplerAddressMode as ProductAddressMode, SamplerBorderColor,
            SamplerCompareFunction, SamplerCoordinates, SamplerFilter as ProductFilter,
            SamplerMipFilter as ProductMipFilter, SamplerReduction,
        };
        let address = match self.address_mode {
            SamplerAddressMode::ClampToEdge => ProductAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToZero => ProductAddressMode::ClampToZero,
            SamplerAddressMode::Repeat => ProductAddressMode::Repeat,
            SamplerAddressMode::MirroredRepeat => ProductAddressMode::MirroredRepeat,
        };
        let filter = |filter| match filter {
            SamplerFilter::Nearest => ProductFilter::Nearest,
            SamplerFilter::Linear => ProductFilter::Linear,
        };
        let mip_filter = match self.mip_filter {
            SamplerMipFilter::NotMipmapped => ProductMipFilter::None,
            SamplerMipFilter::Nearest => ProductMipFilter::Nearest,
            SamplerMipFilter::Linear => ProductMipFilter::Linear,
        };
        RuntimeSamplerState {
            min_filter: filter(self.min_filter),
            mag_filter: filter(self.mag_filter),
            mip_filter,
            address_mode_s: address,
            address_mode_t: address,
            address_mode_r: address,
            coordinates: if self.normalized_coordinates {
                SamplerCoordinates::Normalized
            } else {
                SamplerCoordinates::Pixel
            },
            compare_function: SamplerCompareFunction::None,
            max_anisotropy: 1,
            lod_min_clamp: 0.0,
            lod_max_clamp: if self.normalized_coordinates {
                65504.0
            } else {
                0.0
            },
            border_color: SamplerBorderColor::TransparentBlack,
            reduction: SamplerReduction::WeightedAverage,
            lod_bias: 0.0,
        }
    }
}

pub(crate) fn product_transform_options(
    case: &AuthoredCase,
) -> Result<metal2vulkan::passes::TransformOptions, String> {
    let mut options = match case.dispatch.as_ref() {
        Some(dispatch) => metal2vulkan::passes::TransformOptions {
            kernel_local_size: dispatch.threads_per_threadgroup,
            ..metal2vulkan::passes::TransformOptions::default()
        },
        None => metal2vulkan::passes::TransformOptions::default(),
    };
    if case.stage == Stage::Fragment {
        options.raster_sample_count = Some(1);
    }
    for sampler in &case.samplers {
        options =
            options.with_runtime_sampler(sampler.binding, sampler.runtime_specialization())?;
    }
    for texture in &case.textures {
        if texture.role != ResourceRole::Input {
            options = options.with_runtime_storage_image(
                texture.binding,
                texture.format.runtime_storage_specialization()?,
            )?;
        }
    }
    for texture in &case.texture_arrays {
        if texture.role != ResourceRole::Input {
            options = options.with_runtime_storage_image(
                texture.binding,
                texture.format.runtime_storage_specialization()?,
            )?;
        }
    }
    Ok(options)
}

pub(crate) fn product_transform_options_with_reflection(
    case: &AuthoredCase,
    reflection: &metal2vulkan::reflect::ShaderReflection,
) -> Result<metal2vulkan::passes::TransformOptions, String> {
    let mut options = product_transform_options(case)?;
    for texture in &case.argument_buffer_textures {
        if texture.role == ResourceRole::Input {
            continue;
        }
        let indices = reflection
            .bindings
            .iter()
            .filter(|binding| {
                binding.kind == metal2vulkan::reflect::ResourceKind::EmbeddedArgBufferTexture
            })
            .filter_map(|binding| {
                let source = binding.embedded_source?;
                let count = binding.descriptor.map_or(1, |descriptor| descriptor.count);
                let delta = texture.field_offset.checked_sub(source.field_offset)?;
                (source.buffer_index == texture.buffer_binding
                    && delta % 8 == 0
                    && delta / 8 < count)
                    .then_some(binding.metal_index)
            })
            .collect::<std::collections::BTreeSet<_>>();
        if indices.is_empty() {
            return Err(format!(
                "argument-buffer storage texture {}+{} has no reflected embedded binding",
                texture.buffer_binding, texture.field_offset
            ));
        }
        let state = texture.format.runtime_storage_specialization()?;
        for index in indices {
            if let Some(existing) = usize::try_from(index)
                .ok()
                .and_then(|index| options.runtime_storage_image_states.get(index))
                .copied()
                .flatten()
            {
                if existing != state {
                    return Err(format!(
                        "argument-buffer storage texture {}+{} conflicts with runtime format {:?} already assigned to reflected resource index {index}",
                        texture.buffer_binding, texture.field_offset, existing.format
                    ));
                }
            }
            options = options.with_runtime_storage_image(index, state)?;
        }
    }
    Ok(options)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SamplerAddressMode {
    ClampToEdge,
    ClampToZero,
    Repeat,
    MirroredRepeat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SamplerFilter {
    Nearest,
    Linear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SamplerMipFilter {
    NotMipmapped,
    Nearest,
    Linear,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RenderTargetResource {
    pub index: u32,
    pub format: TextureFormat,
    pub dimensions: [u32; 2],
    pub initial_bytes_b64: String,
}

/// Literal graphics depth/stencil attachment. Each present aspect is tightly packed in pixel
/// order: little-endian `f32` depth values and one byte per stencil value.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DepthStencilResource {
    pub dimensions: [u32; 2],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_depth_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_stencil_b64: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AttributeInput {
    pub location: u32,
    pub format: AttributeFormat,
    pub stride: u32,
    pub bytes_b64: String,
}

/// Per-vertex graphics attributes and per-thread compute stage inputs share the same literal record
/// layout. Their containing manifest fields determine the stage-specific fetch semantics.
pub type VertexInput = AttributeInput;
pub type KernelStageInput = AttributeInput;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AttributeFormat {
    Char,
    Char2,
    Char3,
    Char4,
    Uchar,
    Uchar2,
    Uchar3,
    Uchar4,
    Short,
    Short2,
    Short3,
    Short4,
    Ushort,
    Ushort2,
    Ushort3,
    Ushort4,
    Half,
    Half2,
    Half3,
    Half4,
    Float,
    Float2,
    Float3,
    Float4,
    Uint,
    Uint2,
    Uint3,
    Uint4,
    Int,
    Int2,
    Int3,
    Int4,
}

impl AttributeFormat {
    pub fn from_air_type_name(name: &str) -> Option<Self> {
        Some(match name {
            "char" => Self::Char,
            "char2" => Self::Char2,
            "char3" => Self::Char3,
            "char4" => Self::Char4,
            "uchar" => Self::Uchar,
            "uchar2" => Self::Uchar2,
            "uchar3" => Self::Uchar3,
            "uchar4" => Self::Uchar4,
            "short" => Self::Short,
            "short2" => Self::Short2,
            "short3" => Self::Short3,
            "short4" => Self::Short4,
            "ushort" => Self::Ushort,
            "ushort2" => Self::Ushort2,
            "ushort3" => Self::Ushort3,
            "ushort4" => Self::Ushort4,
            "half" => Self::Half,
            "half2" => Self::Half2,
            "half3" => Self::Half3,
            "half4" => Self::Half4,
            "float" => Self::Float,
            "float2" => Self::Float2,
            "float3" => Self::Float3,
            "float4" => Self::Float4,
            "uint" => Self::Uint,
            "uint2" => Self::Uint2,
            "uint3" => Self::Uint3,
            "uint4" => Self::Uint4,
            "int" => Self::Int,
            "int2" => Self::Int2,
            "int3" => Self::Int3,
            "int4" => Self::Int4,
            _ => return None,
        })
    }

    pub fn byte_size(self) -> u32 {
        match self {
            Self::Char | Self::Uchar => 1,
            Self::Char2 | Self::Uchar2 => 2,
            Self::Char3 | Self::Uchar3 => 3,
            Self::Char4 | Self::Uchar4 => 4,
            Self::Short | Self::Ushort | Self::Half => 2,
            Self::Short2 | Self::Ushort2 | Self::Half2 => 4,
            Self::Short3 | Self::Ushort3 | Self::Half3 => 6,
            Self::Short4 | Self::Ushort4 | Self::Half4 => 8,
            Self::Float | Self::Uint | Self::Int => 4,
            Self::Float2 | Self::Uint2 | Self::Int2 => 8,
            Self::Float3 | Self::Uint3 | Self::Int3 => 12,
            Self::Float4 | Self::Uint4 | Self::Int4 => 16,
        }
    }

    pub fn air_type_name(self) -> &'static str {
        match self {
            Self::Char => "char",
            Self::Char2 => "char2",
            Self::Char3 => "char3",
            Self::Char4 => "char4",
            Self::Uchar => "uchar",
            Self::Uchar2 => "uchar2",
            Self::Uchar3 => "uchar3",
            Self::Uchar4 => "uchar4",
            Self::Short => "short",
            Self::Short2 => "short2",
            Self::Short3 => "short3",
            Self::Short4 => "short4",
            Self::Ushort => "ushort",
            Self::Ushort2 => "ushort2",
            Self::Ushort3 => "ushort3",
            Self::Ushort4 => "ushort4",
            Self::Half => "half",
            Self::Half2 => "half2",
            Self::Half3 => "half3",
            Self::Half4 => "half4",
            Self::Float => "float",
            Self::Float2 => "float2",
            Self::Float3 => "float3",
            Self::Float4 => "float4",
            Self::Uint => "uint",
            Self::Uint2 => "uint2",
            Self::Uint3 => "uint3",
            Self::Uint4 => "uint4",
            Self::Int => "int",
            Self::Int2 => "int2",
            Self::Int3 => "int3",
            Self::Int4 => "int4",
        }
    }

    /// Runtime-array stride used by the product's Vulkan StorageBuffer lowering.
    pub fn storage_buffer_stride(self) -> u32 {
        match self {
            Self::Char3 | Self::Uchar3 => 4,
            Self::Short3 | Self::Ushort3 | Self::Half3 => 8,
            Self::Float3 | Self::Uint3 | Self::Int3 => 16,
            _ => self.byte_size(),
        }
    }

    pub fn supports_tessellation_interface(self) -> bool {
        !matches!(
            self,
            Self::Char
                | Self::Char2
                | Self::Char3
                | Self::Char4
                | Self::Uchar
                | Self::Uchar2
                | Self::Uchar3
                | Self::Uchar4
        )
    }

    pub fn supports_tessellation_system_value(self) -> bool {
        matches!(self, Self::Short | Self::Ushort | Self::Int | Self::Uint)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FunctionConstant {
    pub index: u32,
    pub scalar_type: ScalarType,
    #[serde(default = "default_one", skip_serializing_if = "is_one")]
    pub lanes: u32,
    pub bytes_b64: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ScalarType {
    Bool,
    U8,
    I8,
    U16,
    I16,
    F16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
}

impl ScalarType {
    pub fn byte_size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }

    pub fn from_metal_abi_type_encoding(encoding: &str) -> Option<(Self, u32)> {
        let (lanes, scalar) = if let Some(vector) = encoding.strip_prefix("Dv") {
            let (lanes, scalar) = vector.split_once('_')?;
            (lanes.parse().ok()?, scalar)
        } else {
            (1, encoding)
        };
        if !(1..=4).contains(&lanes) {
            return None;
        }
        let scalar = match scalar {
            "b" => Self::Bool,
            "c" => Self::I8,
            "h" => Self::U8,
            "s" => Self::I16,
            "t" => Self::U16,
            "i" => Self::I32,
            "j" => Self::U32,
            "l" => Self::I64,
            "m" => Self::U64,
            "Dh" => Self::F16,
            "f" => Self::F32,
            "d" => Self::F64,
            _ => return None,
        };
        Some((scalar, lanes))
    }

    pub fn supports_metal_function_constant(self) -> bool {
        self != Self::F64
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Dispatch {
    pub grid: [u32; 3],
    pub threads_per_threadgroup: [u32; 3],
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Draw {
    pub primitive: Primitive,
    pub vertex_start: u32,
    pub vertex_count: u32,
    pub instance_count: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TessellationDraw {
    pub factors: Vec<TessellationFactors>,
    pub instance_count: u32,
    pub amplification_count: u32,
    #[serde(default)]
    pub control_points: Vec<AttributeInput>,
    #[serde(default)]
    pub patch_inputs: Vec<AttributeInput>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TessellationFactors {
    /// IEEE-754 binary16 bit patterns in Metal/Vulkan outer-factor order.
    pub edge_f16: Vec<u16>,
    /// IEEE-754 binary16 bit patterns in Metal/Vulkan inner-factor order.
    pub inside_f16: Vec<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Primitive {
    Point,
    Line,
    LineStrip,
    Triangle,
    TriangleStrip,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OutputSelection {
    Buffer {
        binding: u32,
        offset: u64,
        length: u64,
    },
    ArgumentBufferBuffer {
        buffer_binding: u32,
        field_offset: u32,
        offset: u64,
        length: u64,
    },
    Texture {
        binding: u32,
        origin: [u32; 3],
        dimensions: [u32; 3],
    },
    TextureArrayElement {
        binding: u32,
        element: u32,
        origin: [u32; 3],
        dimensions: [u32; 3],
    },
    ArgumentBufferTexture {
        buffer_binding: u32,
        field_offset: u32,
        origin: [u32; 3],
        dimensions: [u32; 3],
    },
    RenderTarget {
        index: u32,
        origin: [u32; 2],
        dimensions: [u32; 2],
    },
    Depth {
        origin: [u32; 2],
        dimensions: [u32; 2],
    },
    Stencil {
        origin: [u32; 2],
        dimensions: [u32; 2],
    },
    FragmentImageblock {
        semantic: String,
        origin: [u32; 2],
        dimensions: [u32; 2],
    },
}

/// Vertex-stage value routed through the generated fragment observer into render target zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum VertexObservation {
    Position,
    Varying { location: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Comparison {
    Exact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionSafety {
    /// The AIR CFG is structurally acyclic.
    LoopFree,
    /// Authored inputs and function constants give every reachable loop a finite bound.
    AuthoredBounded,
}

#[derive(Serialize)]
struct SemanticCase<'a> {
    air_sha256: &'a str,
    entry: &'a str,
    stage: Stage,
    buffers: Vec<&'a BufferResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    argument_buffer_buffers: Vec<&'a ArgumentBufferBufferResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    threadgroup_memory: Vec<&'a ThreadgroupMemoryResource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    imageblock: &'a Option<ImageblockResource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fragment_imageblock: &'a Option<FragmentImageblockResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    acceleration_structures: Vec<&'a AccelerationStructureResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    visible_function_references: Vec<&'a LinkedFunctionResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    visible_function_tables: Vec<&'a FunctionTableResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    intersection_function_tables: Vec<&'a IntersectionFunctionTableResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    argument_buffer_intersection_function_tables:
        Vec<&'a ArgumentBufferIntersectionFunctionTableResource>,
    textures: Vec<&'a TextureResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    texture_arrays: Vec<&'a TextureArrayResource>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    argument_buffer_textures: Vec<&'a ArgumentBufferTextureResource>,
    samplers: Vec<&'a SamplerResource>,
    render_targets: Vec<&'a RenderTargetResource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    depth_stencil: &'a Option<DepthStencilResource>,
    vertex_inputs: Vec<&'a VertexInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vertex_observation: &'a Option<VertexObservation>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    kernel_stage_inputs: Vec<&'a KernelStageInput>,
    function_constants: Vec<&'a FunctionConstant>,
    dispatch: &'a Option<Dispatch>,
    draw: &'a Option<Draw>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tessellation: &'a Option<TessellationDraw>,
    output: &'a OutputSelection,
    compare: &'a Comparison,
    execution_safety: ExecutionSafety,
}

impl AuthoredCase {
    pub fn is_rasterization_disabled_vertex(&self) -> bool {
        self.stage == Stage::Vertex && self.vertex_observation.is_none()
    }

    fn semantic(&self) -> SemanticCase<'_> {
        SemanticCase {
            air_sha256: &self.air_sha256,
            entry: &self.entry,
            stage: self.stage,
            buffers: sorted_refs(&self.buffers, |resource| resource.binding),
            argument_buffer_buffers: sorted_refs(&self.argument_buffer_buffers, |resource| {
                (resource.buffer_binding, resource.field_offset)
            }),
            threadgroup_memory: sorted_refs(&self.threadgroup_memory, |resource| resource.binding),
            imageblock: &self.imageblock,
            fragment_imageblock: &self.fragment_imageblock,
            acceleration_structures: sorted_refs(&self.acceleration_structures, |resource| {
                resource.binding
            }),
            visible_function_references: sorted_refs(
                &self.visible_function_references,
                |resource| (resource.function.clone(), resource.module_sha256.clone()),
            ),
            visible_function_tables: sorted_refs(&self.visible_function_tables, |resource| {
                resource.binding
            }),
            intersection_function_tables: sorted_refs(
                &self.intersection_function_tables,
                |resource| resource.binding,
            ),
            argument_buffer_intersection_function_tables: sorted_refs(
                &self.argument_buffer_intersection_function_tables,
                |resource| (resource.buffer_binding, resource.field_offset),
            ),
            textures: sorted_refs(&self.textures, |resource| resource.binding),
            texture_arrays: sorted_refs(&self.texture_arrays, |resource| resource.binding),
            argument_buffer_textures: sorted_refs(&self.argument_buffer_textures, |resource| {
                (resource.buffer_binding, resource.field_offset)
            }),
            samplers: sorted_refs(&self.samplers, |resource| resource.binding),
            render_targets: sorted_refs(&self.render_targets, |resource| resource.index),
            depth_stencil: &self.depth_stencil,
            vertex_inputs: sorted_refs(&self.vertex_inputs, |resource| resource.location),
            vertex_observation: &self.vertex_observation,
            kernel_stage_inputs: sorted_refs(&self.kernel_stage_inputs, |resource| {
                resource.location
            }),
            function_constants: sorted_refs(&self.function_constants, |constant| constant.index),
            dispatch: &self.dispatch,
            draw: &self.draw,
            tessellation: &self.tessellation,
            output: &self.output,
            compare: &self.compare,
            execution_safety: self.execution_safety,
        }
    }

    pub fn computed_case_id(&self) -> Result<String, String> {
        let canonical = to_sorted_json_string(self.semantic())
            .map_err(|error| format!("canonicalize case: {error}"))?;
        Ok(sha256_bytes(canonical.as_bytes()))
    }

    pub fn computed_input_sha256(&self) -> Result<String, String> {
        #[derive(Serialize)]
        struct Inputs<'a> {
            case_id: &'a str,
            buffers: Vec<&'a BufferResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            argument_buffer_buffers: Vec<&'a ArgumentBufferBufferResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            threadgroup_memory: Vec<&'a ThreadgroupMemoryResource>,
            #[serde(skip_serializing_if = "Option::is_none")]
            imageblock: &'a Option<ImageblockResource>,
            #[serde(skip_serializing_if = "Option::is_none")]
            fragment_imageblock: &'a Option<FragmentImageblockResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            acceleration_structures: Vec<&'a AccelerationStructureResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            visible_function_references: Vec<&'a LinkedFunctionResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            visible_function_tables: Vec<&'a FunctionTableResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            intersection_function_tables: Vec<&'a IntersectionFunctionTableResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            argument_buffer_intersection_function_tables:
                Vec<&'a ArgumentBufferIntersectionFunctionTableResource>,
            textures: Vec<&'a TextureResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            texture_arrays: Vec<&'a TextureArrayResource>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            argument_buffer_textures: Vec<&'a ArgumentBufferTextureResource>,
            samplers: Vec<&'a SamplerResource>,
            render_targets: Vec<&'a RenderTargetResource>,
            #[serde(skip_serializing_if = "Option::is_none")]
            depth_stencil: &'a Option<DepthStencilResource>,
            vertex_inputs: Vec<&'a VertexInput>,
            #[serde(skip_serializing_if = "Option::is_none")]
            vertex_observation: &'a Option<VertexObservation>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            kernel_stage_inputs: Vec<&'a KernelStageInput>,
            function_constants: Vec<&'a FunctionConstant>,
            dispatch: &'a Option<Dispatch>,
            draw: &'a Option<Draw>,
            #[serde(skip_serializing_if = "Option::is_none")]
            tessellation: &'a Option<TessellationDraw>,
            output: &'a OutputSelection,
        }
        let canonical = to_sorted_json_string(Inputs {
            case_id: &self.case_id,
            buffers: sorted_refs(&self.buffers, |resource| resource.binding),
            argument_buffer_buffers: sorted_refs(&self.argument_buffer_buffers, |resource| {
                (resource.buffer_binding, resource.field_offset)
            }),
            threadgroup_memory: sorted_refs(&self.threadgroup_memory, |resource| resource.binding),
            imageblock: &self.imageblock,
            fragment_imageblock: &self.fragment_imageblock,
            acceleration_structures: sorted_refs(&self.acceleration_structures, |resource| {
                resource.binding
            }),
            visible_function_references: sorted_refs(
                &self.visible_function_references,
                |resource| (resource.function.clone(), resource.module_sha256.clone()),
            ),
            visible_function_tables: sorted_refs(&self.visible_function_tables, |resource| {
                resource.binding
            }),
            intersection_function_tables: sorted_refs(
                &self.intersection_function_tables,
                |resource| resource.binding,
            ),
            argument_buffer_intersection_function_tables: sorted_refs(
                &self.argument_buffer_intersection_function_tables,
                |resource| (resource.buffer_binding, resource.field_offset),
            ),
            textures: sorted_refs(&self.textures, |resource| resource.binding),
            texture_arrays: sorted_refs(&self.texture_arrays, |resource| resource.binding),
            argument_buffer_textures: sorted_refs(&self.argument_buffer_textures, |resource| {
                (resource.buffer_binding, resource.field_offset)
            }),
            samplers: sorted_refs(&self.samplers, |resource| resource.binding),
            render_targets: sorted_refs(&self.render_targets, |resource| resource.index),
            depth_stencil: &self.depth_stencil,
            vertex_inputs: sorted_refs(&self.vertex_inputs, |resource| resource.location),
            vertex_observation: &self.vertex_observation,
            kernel_stage_inputs: sorted_refs(&self.kernel_stage_inputs, |resource| {
                resource.location
            }),
            function_constants: sorted_refs(&self.function_constants, |constant| constant.index),
            dispatch: &self.dispatch,
            draw: &self.draw,
            tessellation: &self.tessellation,
            output: &self.output,
        })
        .map_err(|error| format!("canonicalize inputs: {error}"))?;
        Ok(sha256_bytes(canonical.as_bytes()))
    }

    pub fn validate_literal_resources(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        validate_hash("air_sha256", &self.air_sha256, &mut errors);
        validate_hash("case_id", &self.case_id, &mut errors);
        if self.name.trim().is_empty() {
            errors.push("name must not be empty".into());
        }
        if self.entry.trim().is_empty() {
            errors.push("entry must not be empty".into());
        }
        if self.execution_safety == ExecutionSafety::AuthoredBounded
            && self
                .rationale
                .as_deref()
                .is_none_or(|rationale| rationale.trim().is_empty())
        {
            errors.push(
                "authored_bounded execution_safety requires a rationale identifying the finite loop bound"
                    .into(),
            );
        }
        if [
            self.dispatch.is_some(),
            self.draw.is_some(),
            self.tessellation.is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count()
            != 1
        {
            errors.push("exactly one of dispatch, draw, or tessellation is required".into());
        }
        match self.stage {
            Stage::Kernel if self.dispatch.is_none() => {
                errors.push("kernel case requires dispatch".into())
            }
            Stage::Vertex if self.draw.is_none() && self.tessellation.is_none() => {
                errors.push("vertex case requires draw or tessellation".into())
            }
            Stage::Fragment if self.draw.is_none() => {
                errors.push("fragment case requires draw".into())
            }
            _ => {}
        }
        if let Some(dispatch) = &self.dispatch {
            validate_dimensions("dispatch.grid", &dispatch.grid, &mut errors);
            validate_dimensions(
                "dispatch.threads_per_threadgroup",
                &dispatch.threads_per_threadgroup,
                &mut errors,
            );
            if let Some(imageblock) = &self.imageblock {
                validate_dimensions("imageblock.dimensions", &imageblock.dimensions, &mut errors);
                if imageblock.dimensions
                    != [
                        dispatch.threads_per_threadgroup[0],
                        dispatch.threads_per_threadgroup[1],
                    ]
                {
                    errors.push(format!(
                        "imageblock dimensions {:?} must equal threadgroup x/y dimensions {:?}",
                        imageblock.dimensions,
                        &dispatch.threads_per_threadgroup[..2]
                    ));
                }
            }
        }
        if self.imageblock.is_some() && self.stage != Stage::Kernel {
            errors.push("imageblock is valid only for kernel cases".into());
        }
        if let Some(imageblock) = &self.fragment_imageblock {
            if self.stage != Stage::Fragment {
                errors.push("fragment_imageblock is valid only for fragment cases".into());
            }
            validate_dimensions(
                "fragment_imageblock.dimensions",
                &imageblock.dimensions,
                &mut errors,
            );
            let mut semantics = HashSet::new();
            for member in &imageblock.members {
                if !semantics.insert(&member.semantic) {
                    errors.push(format!(
                        "duplicate fragment imageblock member semantic {}",
                        member.semantic
                    ));
                }
            }
            for member in &imageblock.members {
                let expected = checked_extent_bytes(
                    "fragment imageblock member",
                    &imageblock.dimensions,
                    member.format.byte_size(),
                    &mut errors,
                );
                validate_role_bytes_with_len(
                    &format!("fragment imageblock member {}", member.semantic),
                    member.role,
                    member.bytes_b64.as_deref(),
                    member.initial_bytes_b64.as_deref(),
                    expected,
                    &mut errors,
                );
            }
        }
        if let Some(draw) = &self.draw {
            if draw.vertex_count == 0 {
                errors.push("draw.vertex_count must be nonzero".into());
            }
            if draw.instance_count == 0 {
                errors.push("draw.instance_count must be nonzero".into());
            }
        }
        if let Some(tessellation) = &self.tessellation {
            if self.stage != Stage::Vertex {
                errors.push("tessellation is valid only for vertex cases".into());
            }
            if tessellation.factors.is_empty() {
                errors.push("tessellation.factors must contain at least one patch".into());
            }
            if tessellation.instance_count == 0 {
                errors.push("tessellation.instance_count must be nonzero".into());
            }
            if tessellation.amplification_count != 1 {
                errors.push(
                    "tessellation.amplification_count must be 1 for the single-layer observation contract"
                        .into(),
                );
            }
            for (patch, factors) in tessellation.factors.iter().enumerate() {
                for (kind, values) in [
                    ("edge_f16", &factors.edge_f16),
                    ("inside_f16", &factors.inside_f16),
                ] {
                    for bits in values {
                        if bits & 0x8000 != 0 || bits & 0x7fff == 0 || bits & 0x7c00 == 0x7c00 {
                            errors.push(format!(
                                "tessellation patch {patch} {kind} contains non-positive or non-finite binary16 {bits:#06x}"
                            ));
                        } else if *bits > 0x5400 {
                            errors.push(format!(
                                "tessellation patch {patch} {kind} factor exceeds 64"
                            ));
                        }
                    }
                }
            }
            ensure_unique(
                "tessellation control-point location",
                tessellation
                    .control_points
                    .iter()
                    .map(|input| input.location),
                &mut errors,
            );
            ensure_unique(
                "tessellation patch-input location",
                tessellation.patch_inputs.iter().map(|input| input.location),
                &mut errors,
            );
            validate_attribute_inputs(
                "tessellation control point",
                &tessellation.control_points,
                None,
                &mut errors,
            );
            validate_attribute_inputs(
                "tessellation patch input",
                &tessellation.patch_inputs,
                None,
                &mut errors,
            );
        }

        let mut buffer_bindings = HashSet::new();
        for buffer in &self.buffers {
            if !buffer_bindings.insert(buffer.binding) {
                errors.push(format!("duplicate buffer binding {}", buffer.binding));
            }
            validate_role_bytes(
                &format!("buffer binding {}", buffer.binding),
                buffer.role,
                buffer.bytes_b64.as_deref(),
                buffer.initial_bytes_b64.as_deref(),
                &mut errors,
            );
        }
        let mut argument_buffer_buffer_keys = HashSet::new();
        for buffer in &self.argument_buffer_buffers {
            let key = (buffer.buffer_binding, buffer.field_offset);
            if !argument_buffer_buffer_keys.insert(key) {
                errors.push(format!(
                    "duplicate argument-buffer buffer at buffer {} offset {}",
                    buffer.buffer_binding, buffer.field_offset
                ));
            }
            if !buffer_bindings.contains(&buffer.buffer_binding) {
                errors.push(format!(
                    "argument-buffer buffer at offset {} owns undeclared buffer {}",
                    buffer.field_offset, buffer.buffer_binding
                ));
            }
            validate_role_bytes(
                &format!(
                    "argument-buffer buffer at buffer {} offset {}",
                    buffer.buffer_binding, buffer.field_offset
                ),
                buffer.role,
                buffer.bytes_b64.as_deref(),
                buffer.initial_bytes_b64.as_deref(),
                &mut errors,
            );
        }
        let mut acceleration_structure_bindings = HashSet::new();
        let mut threadgroup_bindings = HashSet::new();
        for resource in &self.threadgroup_memory {
            if !threadgroup_bindings.insert(resource.binding) {
                errors.push(format!(
                    "duplicate threadgroup-memory binding {}",
                    resource.binding
                ));
            }
            if resource.length == 0 {
                errors.push(format!(
                    "threadgroup-memory binding {} length must be nonzero",
                    resource.binding
                ));
            }
            if buffer_bindings.contains(&resource.binding) {
                errors.push(format!(
                    "binding {} is declared as both a buffer and threadgroup memory",
                    resource.binding
                ));
            }
        }
        for resource in &self.acceleration_structures {
            if !acceleration_structure_bindings.insert(resource.binding) {
                errors.push(format!(
                    "duplicate acceleration-structure binding {}",
                    resource.binding
                ));
            }
            if buffer_bindings.contains(&resource.binding) {
                errors.push(format!(
                    "binding {} is declared as both a buffer and an acceleration structure",
                    resource.binding
                ));
            }
            if threadgroup_bindings.contains(&resource.binding) {
                errors.push(format!(
                    "binding {} is declared as both threadgroup memory and an acceleration structure",
                    resource.binding
                ));
            }
            match resource.kind {
                AccelerationStructureKind::Instance => {
                    if resource.primitive_triangles_b64.is_some() {
                        errors.push(format!(
                            "instance acceleration-structure binding {} cannot declare primitive triangles",
                            resource.binding
                        ));
                    }
                    if resource.child_references.len() > u32::MAX as usize {
                        errors.push(format!(
                            "acceleration-structure binding {} has too many instances",
                            resource.binding
                        ));
                    }
                }
                AccelerationStructureKind::Primitive => {
                    if !resource.child_references.is_empty() {
                        errors.push(format!(
                            "primitive acceleration-structure binding {} cannot declare child references",
                            resource.binding
                        ));
                    }
                    validate_primitive_triangles(resource, &mut errors);
                }
            }
        }
        let mut visible_reference_names = HashSet::new();
        for reference in &self.visible_function_references {
            validate_hash(
                &format!(
                    "visible function reference {:?} module_sha256",
                    reference.function
                ),
                &reference.module_sha256,
                &mut errors,
            );
            if reference.function.trim().is_empty() {
                errors.push("visible function reference name must not be empty".into());
            } else if !visible_reference_names.insert(reference.function.as_str()) {
                errors.push(format!(
                    "duplicate visible function reference {:?}",
                    reference.function
                ));
            }
        }
        let mut function_table_bindings = HashSet::new();
        for table in &self.visible_function_tables {
            let kind = "visible-function";
            if !function_table_bindings.insert(table.binding) {
                errors.push(format!(
                    "duplicate function-table buffer binding {}",
                    table.binding
                ));
            }
            for (other, occupied) in [
                ("buffer", buffer_bindings.contains(&table.binding)),
                (
                    "threadgroup memory",
                    threadgroup_bindings.contains(&table.binding),
                ),
                (
                    "acceleration structure",
                    acceleration_structure_bindings.contains(&table.binding),
                ),
            ] {
                if occupied {
                    errors.push(format!(
                        "binding {} is declared as both a {kind} table and {other}",
                        table.binding
                    ));
                }
            }
            if table.size == 0 {
                errors.push(format!(
                    "{kind} table binding {} size must be nonzero",
                    table.binding
                ));
            }
            let mut previous = None;
            for entry in &table.entries {
                if entry.index >= table.size {
                    errors.push(format!(
                        "{kind} table binding {} entry {} exceeds size {}",
                        table.binding, entry.index, table.size
                    ));
                }
                if previous.is_some_and(|index| index >= entry.index) {
                    errors.push(format!(
                        "{kind} table binding {} entries must be sorted by unique index",
                        table.binding
                    ));
                }
                previous = Some(entry.index);
                validate_hash(
                    &format!(
                        "{kind} table binding {} entry {} module_sha256",
                        table.binding, entry.index
                    ),
                    &entry.module_sha256,
                    &mut errors,
                );
                if entry.function.trim().is_empty() {
                    errors.push(format!(
                        "{kind} table binding {} entry {} function must not be empty",
                        table.binding, entry.index
                    ));
                }
            }
        }
        for table in &self.intersection_function_tables {
            let kind = "intersection-function";
            if !function_table_bindings.insert(table.binding) {
                errors.push(format!(
                    "duplicate function-table buffer binding {}",
                    table.binding
                ));
            }
            for (other, occupied) in [
                ("buffer", buffer_bindings.contains(&table.binding)),
                (
                    "threadgroup memory",
                    threadgroup_bindings.contains(&table.binding),
                ),
                (
                    "acceleration structure",
                    acceleration_structure_bindings.contains(&table.binding),
                ),
            ] {
                if occupied {
                    errors.push(format!(
                        "binding {} is declared as both a {kind} table and {other}",
                        table.binding
                    ));
                }
            }
            if table.size == 0 {
                errors.push(format!(
                    "{kind} table binding {} size must be nonzero",
                    table.binding
                ));
            }
            let mut previous = None;
            for entry in &table.entries {
                let index = entry.index();
                if index >= table.size {
                    errors.push(format!(
                        "{kind} table binding {} entry {index} exceeds size {}",
                        table.binding, table.size
                    ));
                }
                if previous.is_some_and(|previous| previous >= index) {
                    errors.push(format!(
                        "{kind} table binding {} entries must be sorted by unique index",
                        table.binding
                    ));
                }
                previous = Some(index);
                match entry {
                    IntersectionFunctionTableEntry::Linked {
                        module_sha256,
                        function,
                        ..
                    } => {
                        validate_hash(
                            &format!(
                                "{kind} table binding {} entry {index} module_sha256",
                                table.binding
                            ),
                            module_sha256,
                            &mut errors,
                        );
                        if function.trim().is_empty() {
                            errors.push(format!(
                                "{kind} table binding {} entry {index} function must not be empty",
                                table.binding
                            ));
                        }
                    }
                    IntersectionFunctionTableEntry::OpaqueTriangle { signature, .. } => {
                        if signature.windows(2).any(|pair| pair[0] >= pair[1]) {
                            errors.push(format!(
                                "{kind} table binding {} opaque entry {index} signature flags must be sorted and unique",
                                table.binding
                            ));
                        }
                    }
                }
            }
        }
        let mut embedded_table_keys = HashSet::new();
        for table in &self.argument_buffer_intersection_function_tables {
            let key = (table.buffer_binding, table.field_offset);
            let label = format!(
                "argument-buffer intersection-function table at buffer {} offset {}",
                table.buffer_binding, table.field_offset
            );
            if !embedded_table_keys.insert(key) {
                errors.push(format!("duplicate {label}"));
            }
            if !buffer_bindings.contains(&table.buffer_binding) {
                errors.push(format!("{label} owns an undeclared buffer"));
            }
            if table.size == 0 {
                errors.push(format!("{label} size must be nonzero"));
            }
            let mut previous = None;
            for entry in &table.entries {
                let index = entry.index();
                if index >= table.size {
                    errors.push(format!("{label} entry {index} exceeds size {}", table.size));
                }
                if previous.is_some_and(|previous| previous >= index) {
                    errors.push(format!("{label} entries must be sorted by unique index"));
                }
                previous = Some(index);
                match entry {
                    IntersectionFunctionTableEntry::Linked {
                        module_sha256,
                        function,
                        ..
                    } => {
                        validate_hash(
                            &format!("{label} entry {index} module_sha256"),
                            module_sha256,
                            &mut errors,
                        );
                        if function.trim().is_empty() {
                            errors
                                .push(format!("{label} entry {index} function must not be empty"));
                        }
                    }
                    IntersectionFunctionTableEntry::OpaqueTriangle { signature, .. } => {
                        if signature.windows(2).any(|pair| pair[0] >= pair[1]) {
                            errors.push(format!(
                                "{label} opaque entry {index} signature flags must be sorted and unique"
                            ));
                        }
                    }
                }
            }
        }
        let mut texture_bindings = HashSet::new();
        for texture in &self.textures {
            if !texture_bindings.insert(texture.binding) {
                errors.push(format!("duplicate texture binding {}", texture.binding));
            }
            validate_dimensions(
                &format!("texture binding {} dimensions", texture.binding),
                &texture.dimensions,
                &mut errors,
            );
            if let Err(error) = crate::literal::texture_layout(
                texture.texture_type,
                texture.dimensions,
                texture.sample_count,
            ) {
                errors.push(format!("texture binding {}: {error}", texture.binding));
            }
            if texture.sample_count > 1 && texture.role != ResourceRole::Input {
                errors.push(format!(
                    "multisample texture binding {} must have input role",
                    texture.binding
                ));
            }
            let expected = checked_texture_extent_bytes(
                &format!("texture binding {}", texture.binding),
                &texture.dimensions,
                texture.sample_count,
                texture.format.bytes_per_pixel(),
                &mut errors,
            );
            validate_role_bytes_with_len(
                &format!("texture binding {}", texture.binding),
                texture.role,
                texture.bytes_b64.as_deref(),
                texture.initial_bytes_b64.as_deref(),
                expected,
                &mut errors,
            );
        }
        for array in &self.texture_arrays {
            if array.elements.is_empty() {
                errors.push(format!(
                    "texture-array binding {} must have at least one element",
                    array.binding
                ));
            }
            for element in 0..array.elements.len() {
                let Some(slot) = array.binding.checked_add(element as u32) else {
                    errors.push(format!(
                        "texture-array binding {} slot range overflows",
                        array.binding
                    ));
                    break;
                };
                if !texture_bindings.insert(slot) {
                    errors.push(format!(
                        "texture-array binding {} element {element} overlaps Metal texture slot {slot}",
                        array.binding
                    ));
                }
            }
            if array.elements.len()
                > metal2vulkan::meta::TEXTURE_HANDLE_ARRAY_DESCRIPTOR_COUNT as usize
            {
                errors.push(format!(
                    "texture-array binding {} has {} elements, maximum is {}",
                    array.binding,
                    array.elements.len(),
                    metal2vulkan::meta::TEXTURE_HANDLE_ARRAY_DESCRIPTOR_COUNT
                ));
            }
            for (element_index, element) in array.elements.iter().enumerate() {
                let label = format!(
                    "texture-array binding {} element {element_index}",
                    array.binding
                );
                validate_dimensions(
                    &format!("{label} dimensions"),
                    &element.dimensions,
                    &mut errors,
                );
                if let Err(error) = crate::literal::texture_layout(
                    array.texture_type,
                    element.dimensions,
                    array.sample_count,
                ) {
                    errors.push(format!("{label}: {error}"));
                }
                if array.sample_count > 1 && array.role != ResourceRole::Input {
                    errors.push(format!("multisample {label} must have input role"));
                }
                let expected = checked_texture_extent_bytes(
                    &label,
                    &element.dimensions,
                    array.sample_count,
                    array.format.bytes_per_pixel(),
                    &mut errors,
                );
                validate_role_bytes_with_len(
                    &label,
                    array.role,
                    element.bytes_b64.as_deref(),
                    element.initial_bytes_b64.as_deref(),
                    expected,
                    &mut errors,
                );
            }
        }
        let mut argument_buffer_texture_keys = HashSet::new();
        for texture in &self.argument_buffer_textures {
            let key = (texture.buffer_binding, texture.field_offset);
            if !argument_buffer_texture_keys.insert(key) {
                errors.push(format!(
                    "duplicate argument-buffer texture at buffer {} offset {}",
                    texture.buffer_binding, texture.field_offset
                ));
            }
            if !buffer_bindings.contains(&texture.buffer_binding) {
                errors.push(format!(
                    "argument-buffer texture at offset {} owns undeclared buffer {}",
                    texture.field_offset, texture.buffer_binding
                ));
            }
            validate_dimensions(
                &format!(
                    "argument-buffer texture at buffer {} offset {} dimensions",
                    texture.buffer_binding, texture.field_offset
                ),
                &texture.dimensions,
                &mut errors,
            );
            if let Err(error) = crate::literal::texture_layout(
                texture.texture_type,
                texture.dimensions,
                texture.sample_count,
            ) {
                errors.push(format!(
                    "argument-buffer texture at buffer {} offset {}: {error}",
                    texture.buffer_binding, texture.field_offset
                ));
            }
            if texture.sample_count > 1 && texture.role != ResourceRole::Input {
                errors.push(format!(
                    "multisample argument-buffer texture at buffer {} offset {} must have input role",
                    texture.buffer_binding, texture.field_offset
                ));
            }
            let expected = checked_texture_extent_bytes(
                &format!(
                    "argument-buffer texture at buffer {} offset {}",
                    texture.buffer_binding, texture.field_offset
                ),
                &texture.dimensions,
                texture.sample_count,
                texture.format.bytes_per_pixel(),
                &mut errors,
            );
            validate_role_bytes_with_len(
                &format!(
                    "argument-buffer texture at buffer {} offset {}",
                    texture.buffer_binding, texture.field_offset
                ),
                texture.role,
                texture.bytes_b64.as_deref(),
                texture.initial_bytes_b64.as_deref(),
                expected,
                &mut errors,
            );
        }
        ensure_unique(
            "sampler binding",
            self.samplers.iter().map(|sampler| sampler.binding),
            &mut errors,
        );
        ensure_unique(
            "render target index",
            self.render_targets.iter().map(|target| target.index),
            &mut errors,
        );
        for target in &self.render_targets {
            validate_dimensions(
                &format!("render target {} dimensions", target.index),
                &target.dimensions,
                &mut errors,
            );
            let expected = checked_extent_bytes(
                &format!("render target {}", target.index),
                &target.dimensions,
                target.format.bytes_per_pixel(),
                &mut errors,
            );
            validate_b64_len(
                &format!("render target {} initial_bytes_b64", target.index),
                &target.initial_bytes_b64,
                expected,
                &mut errors,
            );
        }
        if let Some(attachment) = &self.depth_stencil {
            validate_dimensions(
                "depth/stencil attachment",
                &attachment.dimensions,
                &mut errors,
            );
            let pixels = attachment.dimensions[0]
                .checked_mul(attachment.dimensions[1])
                .map(u64::from);
            if attachment.initial_depth_b64.is_none() && attachment.initial_stencil_b64.is_none() {
                errors
                    .push("depth/stencil attachment requires at least one authored aspect".into());
            }
            if let Some(bytes) = &attachment.initial_depth_b64 {
                validate_b64_len(
                    "depth/stencil initial_depth_b64",
                    bytes,
                    pixels.and_then(|count| usize::try_from(count.checked_mul(4)?).ok()),
                    &mut errors,
                );
            }
            if let Some(bytes) = &attachment.initial_stencil_b64 {
                validate_b64_len(
                    "depth/stencil initial_stencil_b64",
                    bytes,
                    pixels.and_then(|count| usize::try_from(count).ok()),
                    &mut errors,
                );
            }
            if self.stage != Stage::Fragment {
                errors.push("depth_stencil is valid only for fragment cases".into());
            }
        }
        ensure_unique(
            "vertex input location",
            self.vertex_inputs.iter().map(|input| input.location),
            &mut errors,
        );
        let required_vertices = self
            .draw
            .as_ref()
            .and_then(|draw| draw.vertex_start.checked_add(draw.vertex_count));
        validate_attribute_inputs(
            "vertex input",
            &self.vertex_inputs,
            required_vertices,
            &mut errors,
        );
        if !self.vertex_inputs.is_empty() && self.stage != Stage::Vertex {
            errors.push("vertex_inputs are valid only for vertex cases".into());
        }
        match (self.stage, self.vertex_observation) {
            (Stage::Vertex, None) => {
                if self.tessellation.is_some() {
                    errors.push(
                        "rasterization-disabled vertex execution does not accept tessellation"
                            .into(),
                    );
                }
                if !self.render_targets.is_empty() || self.depth_stencil.is_some() {
                    errors.push(
                        "rasterization-disabled vertex execution does not accept attachments"
                            .into(),
                    );
                }
                if matches!(
                    self.output,
                    OutputSelection::RenderTarget { .. }
                        | OutputSelection::Depth { .. }
                        | OutputSelection::Stencil { .. }
                        | OutputSelection::FragmentImageblock { .. }
                ) {
                    errors.push(
                        "rasterization-disabled vertex output must select a shader resource".into(),
                    );
                }
            }
            (Stage::Vertex, Some(_)) => {
                if self.render_targets.len() != 1 || self.render_targets[0].index != 0 {
                    errors.push("vertex observation requires exactly render target zero".into());
                }
                if !matches!(self.output, OutputSelection::RenderTarget { index: 0, .. }) {
                    errors.push("vertex observation output must select render target zero".into());
                }
            }
            (_, Some(_)) => errors.push("vertex_observation is valid only for vertex cases".into()),
            (_, None) => {}
        }
        ensure_unique(
            "kernel stage-input location",
            self.kernel_stage_inputs.iter().map(|input| input.location),
            &mut errors,
        );
        validate_attribute_inputs(
            "kernel stage input",
            &self.kernel_stage_inputs,
            self.dispatch.as_ref().map(|dispatch| dispatch.grid[0]),
            &mut errors,
        );
        for input in &self.kernel_stage_inputs {
            let expected = input.format.storage_buffer_stride();
            if input.stride != expected {
                errors.push(format!(
                    "kernel stage input {} stride {} must equal product storage-buffer stride {expected}",
                    input.location, input.stride
                ));
            }
        }
        if !self.kernel_stage_inputs.is_empty() && self.stage != Stage::Kernel {
            errors.push("kernel_stage_inputs are valid only for kernel cases".into());
        }
        ensure_unique(
            "function constant index",
            self.function_constants.iter().map(|fc| fc.index),
            &mut errors,
        );
        for fc in &self.function_constants {
            if !(1..=4).contains(&fc.lanes) {
                errors.push(format!(
                    "function constant {} lanes must be in 1..=4",
                    fc.index
                ));
            }
            validate_b64_len(
                &format!("function constant {} bytes_b64", fc.index),
                &fc.bytes_b64,
                fc.scalar_type.byte_size().checked_mul(fc.lanes as usize),
                &mut errors,
            );
            if fc.scalar_type == ScalarType::Bool
                && base64::engine::general_purpose::STANDARD
                    .decode(&fc.bytes_b64)
                    .is_ok_and(|bytes| bytes.iter().any(|byte| !matches!(byte, 0 | 1)))
            {
                errors.push(format!(
                    "function constant {} bool must be encoded as 0 or 1",
                    fc.index
                ));
            }
        }
        self.validate_output_selection(&mut errors);
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn validate_output_selection(&self, errors: &mut Vec<String>) {
        match &self.output {
            OutputSelection::Buffer {
                binding,
                offset,
                length,
            } => {
                if *length == 0 {
                    errors.push("buffer output length must be nonzero".into());
                }
                let Some(resource) = self.buffers.iter().find(|item| item.binding == *binding)
                else {
                    errors.push(format!("buffer output binding {binding} is not declared"));
                    return;
                };
                if resource.role == ResourceRole::Input {
                    errors.push(format!(
                        "buffer output binding {binding} has input-only role"
                    ));
                }
                if let Some(bytes) = initial_bytes(
                    resource.bytes_b64.as_deref(),
                    resource.initial_bytes_b64.as_deref(),
                ) {
                    if offset.saturating_add(*length) > bytes.len() as u64 {
                        errors.push(format!(
                            "buffer output [{offset}, {}) exceeds binding {binding} length {}",
                            offset.saturating_add(*length),
                            bytes.len()
                        ));
                    }
                }
            }
            OutputSelection::ArgumentBufferBuffer {
                buffer_binding,
                field_offset,
                offset,
                length,
            } => {
                let Some(resource) = self.argument_buffer_buffers.iter().find(|item| {
                    item.buffer_binding == *buffer_binding && item.field_offset == *field_offset
                }) else {
                    errors.push(format!(
                        "argument-buffer buffer at buffer {buffer_binding} offset {field_offset} is not declared"
                    ));
                    return;
                };
                if resource.role == ResourceRole::Input {
                    errors.push(format!(
                        "argument-buffer buffer at buffer {buffer_binding} offset {field_offset} has input-only role"
                    ));
                }
                if *length == 0 {
                    errors.push("argument-buffer buffer output length must be nonzero".into());
                }
                if offset.checked_add(*length).is_none() {
                    errors.push("argument-buffer buffer output range overflows".into());
                }
            }
            OutputSelection::Texture {
                binding,
                origin,
                dimensions,
            } => {
                let Some(resource) = self.textures.iter().find(|item| item.binding == *binding)
                else {
                    errors.push(format!("texture output binding {binding} is not declared"));
                    return;
                };
                if resource.role == ResourceRole::Input {
                    errors.push(format!(
                        "texture output binding {binding} has input-only role"
                    ));
                }
                validate_region(
                    "texture output",
                    origin,
                    dimensions,
                    &resource.dimensions,
                    errors,
                );
            }
            OutputSelection::TextureArrayElement {
                binding,
                element,
                origin,
                dimensions,
            } => {
                let Some(resource) = self
                    .texture_arrays
                    .iter()
                    .find(|item| item.binding == *binding)
                else {
                    errors.push(format!(
                        "texture-array output binding {binding} is not declared"
                    ));
                    return;
                };
                if resource.role == ResourceRole::Input {
                    errors.push(format!(
                        "texture-array output binding {binding} has input-only role"
                    ));
                }
                let Some(selected) = resource.elements.get(*element as usize) else {
                    errors.push(format!(
                        "texture-array output binding {binding} element {element} is not declared"
                    ));
                    return;
                };
                validate_region(
                    "texture-array element output",
                    origin,
                    dimensions,
                    &selected.dimensions,
                    errors,
                );
            }
            OutputSelection::ArgumentBufferTexture {
                buffer_binding,
                field_offset,
                origin,
                dimensions,
            } => {
                let Some(resource) = self.argument_buffer_textures.iter().find(|item| {
                    item.buffer_binding == *buffer_binding && item.field_offset == *field_offset
                }) else {
                    errors.push(format!(
                        "argument-buffer texture at buffer {buffer_binding} offset {field_offset} is not declared"
                    ));
                    return;
                };
                if resource.role == ResourceRole::Input {
                    errors.push(format!(
                        "argument-buffer texture at buffer {buffer_binding} offset {field_offset} has input-only role"
                    ));
                }
                validate_region(
                    "argument-buffer texture output",
                    origin,
                    dimensions,
                    &resource.dimensions,
                    errors,
                );
            }
            OutputSelection::RenderTarget {
                index,
                origin,
                dimensions,
            } => {
                let Some(resource) = self.render_targets.iter().find(|item| item.index == *index)
                else {
                    errors.push(format!(
                        "render target output index {index} is not declared"
                    ));
                    return;
                };
                validate_region(
                    "render target output",
                    origin,
                    dimensions,
                    &resource.dimensions,
                    errors,
                );
            }
            OutputSelection::Depth { origin, dimensions } => {
                let Some(resource) = &self.depth_stencil else {
                    errors.push("depth output requires a depth/stencil attachment".into());
                    return;
                };
                if resource.initial_depth_b64.is_none() {
                    errors.push("depth output requires an authored depth aspect".into());
                }
                validate_region(
                    "depth output",
                    origin,
                    dimensions,
                    &resource.dimensions,
                    errors,
                );
            }
            OutputSelection::Stencil { origin, dimensions } => {
                let Some(resource) = &self.depth_stencil else {
                    errors.push("stencil output requires a depth/stencil attachment".into());
                    return;
                };
                if resource.initial_stencil_b64.is_none() {
                    errors.push("stencil output requires an authored stencil aspect".into());
                }
                validate_region(
                    "stencil output",
                    origin,
                    dimensions,
                    &resource.dimensions,
                    errors,
                );
            }
            OutputSelection::FragmentImageblock {
                semantic,
                origin,
                dimensions,
            } => {
                let Some(imageblock) = &self.fragment_imageblock else {
                    errors.push(
                        "fragment imageblock output requires a fragment_imageblock resource".into(),
                    );
                    return;
                };
                let Some(member) = imageblock
                    .members
                    .iter()
                    .find(|member| member.semantic == *semantic)
                else {
                    errors.push(format!(
                        "fragment imageblock output semantic {semantic} is not declared"
                    ));
                    return;
                };
                if member.role == ResourceRole::Input {
                    errors.push(format!(
                        "fragment imageblock output semantic {semantic} has input-only role"
                    ));
                }
                validate_region(
                    "fragment imageblock output",
                    origin,
                    dimensions,
                    &imageblock.dimensions,
                    errors,
                );
            }
        }
    }
}

fn sorted_refs<T, K: Ord>(values: &[T], key: impl Fn(&T) -> K) -> Vec<&T> {
    let mut values = values.iter().collect::<Vec<_>>();
    values.sort_by_key(|value| key(value));
    values
}

fn validate_hash(field: &str, value: &str, errors: &mut Vec<String>) {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        errors.push(format!("{field} must be 64 hexadecimal characters"));
    } else if value.bytes().any(|byte| byte.is_ascii_uppercase()) {
        errors.push(format!("{field} must use lowercase hexadecimal"));
    }
}

fn validate_dimensions<const N: usize>(field: &str, dims: &[u32; N], errors: &mut Vec<String>) {
    if dims.contains(&0) {
        errors.push(format!("{field} dimensions must be nonzero"));
    }
}

fn checked_extent_bytes<const N: usize>(
    field: &str,
    dimensions: &[u32; N],
    bytes_per_element: usize,
    errors: &mut Vec<String>,
) -> Option<usize> {
    let bytes = dimensions.iter().try_fold(1usize, |size, dimension| {
        size.checked_mul(*dimension as usize)
    });
    match bytes.and_then(|size| size.checked_mul(bytes_per_element)) {
        Some(bytes) => Some(bytes),
        None => {
            errors.push(format!("{field} byte size overflows host limits"));
            None
        }
    }
}

fn checked_texture_extent_bytes(
    field: &str,
    dimensions: &[u32; 3],
    sample_count: u32,
    bytes_per_element: usize,
    errors: &mut Vec<String>,
) -> Option<usize> {
    let bytes_per_texel = bytes_per_element.checked_mul(sample_count as usize);
    match bytes_per_texel {
        Some(bytes_per_texel) => checked_extent_bytes(field, dimensions, bytes_per_texel, errors),
        None => {
            errors.push(format!("{field} byte size overflows host limits"));
            None
        }
    }
}

fn validate_attribute_inputs(
    label: &str,
    inputs: &[AttributeInput],
    required_records: Option<u32>,
    errors: &mut Vec<String>,
) {
    for input in inputs {
        if input.stride < input.format.byte_size() {
            errors.push(format!(
                "{label} {} stride {} is smaller than format size {}",
                input.location,
                input.stride,
                input.format.byte_size()
            ));
        }
        validate_b64_len(
            &format!("{label} {} bytes_b64", input.location),
            &input.bytes_b64,
            None,
            errors,
        );
        if let (Some(records), Ok(bytes)) = (
            required_records,
            base64::engine::general_purpose::STANDARD.decode(&input.bytes_b64),
        ) {
            match records
                .checked_mul(input.stride)
                .map(|required| required as usize)
            {
                Some(required) if bytes.len() < required => errors.push(format!(
                    "{label} {} has {} bytes, but execution requires at least {required}",
                    input.location,
                    bytes.len()
                )),
                None => errors.push(format!("{label} {} byte range overflows", input.location)),
                _ => {}
            }
        }
    }
}

fn ensure_unique(field: &str, values: impl IntoIterator<Item = u32>, errors: &mut Vec<String>) {
    let mut seen = HashSet::new();
    for value in values {
        if !seen.insert(value) {
            errors.push(format!("duplicate {field} {value}"));
        }
    }
}

fn validate_role_bytes(
    label: &str,
    role: ResourceRole,
    bytes: Option<&str>,
    initial: Option<&str>,
    errors: &mut Vec<String>,
) {
    validate_role_bytes_with_len(label, role, bytes, initial, None, errors);
}

fn validate_role_bytes_with_len(
    label: &str,
    role: ResourceRole,
    bytes: Option<&str>,
    initial: Option<&str>,
    expected_len: Option<usize>,
    errors: &mut Vec<String>,
) {
    let selected = match role {
        ResourceRole::Input => match (bytes, initial) {
            (Some(value), None) => Some(("bytes_b64", value)),
            _ => {
                errors.push(format!(
                    "{label} input requires only bytes_b64 (no inferred or initial bytes)"
                ));
                None
            }
        },
        ResourceRole::Output | ResourceRole::InOut => match (bytes, initial) {
            (None, Some(value)) => Some(("initial_bytes_b64", value)),
            _ => {
                errors.push(format!(
                    "{label} output/in_out requires only initial_bytes_b64"
                ));
                None
            }
        },
    };
    if let Some((field, value)) = selected {
        validate_b64_len(&format!("{label} {field}"), value, expected_len, errors);
    }
}

fn validate_b64_len(
    field: &str,
    value: &str,
    expected_len: Option<usize>,
    errors: &mut Vec<String>,
) {
    match base64::engine::general_purpose::STANDARD.decode(value) {
        Ok(bytes) if bytes.is_empty() => errors.push(format!("{field} must not be empty")),
        Ok(bytes) if expected_len.is_some_and(|expected| bytes.len() != expected) => {
            errors.push(format!(
                "{field} decodes to {} bytes, expected {}",
                bytes.len(),
                expected_len.unwrap_or_default()
            ))
        }
        Ok(_) => {}
        Err(error) => errors.push(format!("{field} is invalid base64: {error}")),
    }
}

fn validate_primitive_triangles(
    resource: &AccelerationStructureResource,
    errors: &mut Vec<String>,
) {
    let label = format!(
        "primitive acceleration-structure binding {} triangles",
        resource.binding
    );
    let Some(encoded) = resource.primitive_triangles_b64.as_deref() else {
        errors.push(format!("{label} are required"));
        return;
    };
    let bytes = match base64::engine::general_purpose::STANDARD.decode(encoded) {
        Ok(bytes) => bytes,
        Err(error) => {
            errors.push(format!("{label} are not valid base64: {error}"));
            return;
        }
    };
    const TRIANGLE_BYTES: usize = 9 * std::mem::size_of::<f32>();
    if bytes.is_empty() || bytes.len() % TRIANGLE_BYTES != 0 {
        errors.push(format!(
            "{label} must contain one or more tightly packed 36-byte float3 triangles, got {} bytes",
            bytes.len()
        ));
        return;
    }
    for (index, word) in bytes.chunks_exact(4).enumerate() {
        let value = f32::from_le_bytes(word.try_into().expect("four-byte chunk"));
        if !value.is_finite() {
            errors.push(format!("{label} float {index} is not finite"));
        }
    }
}

fn initial_bytes(bytes: Option<&str>, initial: Option<&str>) -> Option<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(bytes.or(initial)?)
        .ok()
}

fn validate_region<const N: usize>(
    label: &str,
    origin: &[u32; N],
    dimensions: &[u32; N],
    extent: &[u32; N],
    errors: &mut Vec<String>,
) {
    validate_dimensions(label, dimensions, errors);
    for axis in 0..N {
        if origin[axis]
            .checked_add(dimensions[axis])
            .is_none_or(|end| end > extent[axis])
        {
            errors.push(format!("{label} exceeds declared extent on axis {axis}"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example() -> AuthoredCase {
        AuthoredCase {
            air_sha256: "11".repeat(32),
            case_id: String::new(),
            name: "writes-one-word".into(),
            entry: "kernel_main".into(),
            stage: Stage::Kernel,
            buffers: vec![BufferResource {
                binding: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some("q6urqw==".into()),
            }],
            argument_buffer_buffers: vec![],
            threadgroup_memory: vec![],
            imageblock: None,
            fragment_imageblock: None,
            acceleration_structures: vec![],
            visible_function_references: vec![],
            visible_function_tables: vec![],
            intersection_function_tables: vec![],
            argument_buffer_intersection_function_tables: vec![],
            textures: vec![],
            texture_arrays: vec![],
            argument_buffer_textures: vec![],
            samplers: vec![],
            render_targets: vec![],
            depth_stencil: None,
            vertex_inputs: vec![],
            vertex_observation: None,
            kernel_stage_inputs: vec![],
            function_constants: vec![],
            dispatch: Some(Dispatch {
                grid: [1, 1, 1],
                threads_per_threadgroup: [1, 1, 1],
            }),
            draw: None,
            tessellation: None,
            output: OutputSelection::Buffer {
                binding: 0,
                offset: 0,
                length: 4,
            },
            compare: Comparison::Exact,
            execution_safety: ExecutionSafety::LoopFree,
            rationale: Some("documentation does not identify semantics".into()),
            authored_by: Some("test".into()),
        }
    }

    #[test]
    fn authored_runtime_sampler_state_is_the_product_specialization_input() {
        let mut case = example();
        case.samplers.push(SamplerResource {
            binding: 3,
            address_mode: SamplerAddressMode::MirroredRepeat,
            min_filter: SamplerFilter::Nearest,
            mag_filter: SamplerFilter::Nearest,
            mip_filter: SamplerMipFilter::NotMipmapped,
            normalized_coordinates: false,
        });
        let options = product_transform_options(&case).expect("authored specialization");
        assert_eq!(options.kernel_local_size, [1, 1, 1]);
        assert_eq!(
            options.kernel_dispatch, None,
            "authored execution must exercise the product's safe dynamic-grid default"
        );
        assert_eq!(
            options.runtime_sampler_states[3],
            Some(case.samplers[0].runtime_specialization())
        );
        assert!(options.runtime_sampler_states[..3]
            .iter()
            .all(Option::is_none));

        case.samplers[0].mag_filter = SamplerFilter::Linear;
        let error = product_transform_options(&case)
            .expect_err("mixed pixel filters must not reach a Vulkan executor");
        assert!(error.contains("mixed min/mag"), "{error}");
    }

    #[test]
    fn authored_storage_texture_format_is_the_product_specialization_input() {
        let mut case = example();
        case.textures.push(TextureResource {
            binding: 4,
            role: ResourceRole::Output,
            texture_type: TextureType::D2,
            format: TextureFormat::Rgba8Unorm,
            dimensions: [2, 2, 1],
            sample_count: 1,
            bytes_b64: None,
            initial_bytes_b64: None,
        });
        case.texture_arrays.push(TextureArrayResource {
            binding: 6,
            role: ResourceRole::InOut,
            texture_type: TextureType::D2,
            format: TextureFormat::R32Uint,
            sample_count: 1,
            elements: vec![TextureArrayElement {
                dimensions: [1, 1, 1],
                bytes_b64: None,
                initial_bytes_b64: Some("AAAAAA==".into()),
            }],
        });

        let options = product_transform_options(&case).expect("authored storage specialization");
        assert_eq!(
            options.runtime_storage_image_states[4]
                .expect("single storage texture")
                .format,
            metal2vulkan::reflect::RuntimeStorageImageFormat::Rgba8Unorm
        );
        let array = options.runtime_storage_image_states[6].expect("storage texture array");
        assert_eq!(
            array.format,
            metal2vulkan::reflect::RuntimeStorageImageFormat::R32Uint
        );
        assert!(array.capabilities.storage_image_atomic);
    }

    #[test]
    fn authored_argument_buffer_storage_format_uses_reflected_synthetic_identity() {
        let air = r#"
%Args = type <{ %"struct.metal::texture2d" }>
%"struct.metal::texture2d" = type { ptr addrspace(1) }

define void @k(ptr addrspace(2) %args) {
entry:
  %field = getelementptr inbounds %Args, ptr addrspace(2) %args, i64 0, i32 0, i32 0
  %tex = load ptr addrspace(1), ptr addrspace(2) %field, align 8
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %tex, <2 x i32> zeroinitializer, <4 x float> zeroinitializer, i32 0, i32 2)
  ret void
}

declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_name", !"Args", !"air.arg_name", !"args"}
!4 = !{i32 0, i32 8, i32 0, !"texture2d<float, write>", !"output", !"air.indirect_argument", !5}
!5 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"output"}
"#;
        let mut case = example();
        case.argument_buffer_textures
            .push(ArgumentBufferTextureResource {
                buffer_binding: 0,
                field_offset: 0,
                role: ResourceRole::Output,
                texture_type: TextureType::D2,
                format: TextureFormat::Rgba8Unorm,
                dimensions: [1, 1, 1],
                sample_count: 1,
                bytes_b64: None,
                initial_bytes_b64: None,
            });
        let unspecialized = metal2vulkan::reflect_sanitized(
            air,
            metal2vulkan::passes::Stage::Kernel,
            product_transform_options(&case).unwrap(),
        )
        .expect("unspecialized embedded reflection");
        let options = product_transform_options_with_reflection(&case, &unspecialized)
            .expect("reflected embedded specialization");
        assert_eq!(
            options.runtime_storage_image_states[0]
                .expect("synthetic embedded resource")
                .format,
            metal2vulkan::reflect::RuntimeStorageImageFormat::Rgba8Unorm
        );
        let specialized =
            metal2vulkan::reflect_sanitized(air, metal2vulkan::passes::Stage::Kernel, options)
                .expect("specialized embedded reflection");
        assert_eq!(
            specialized.runtime_storage_image_specializations[0].metal_index,
            0
        );
        assert_eq!(
            specialized.runtime_storage_image_specializations[0].spirv_format,
            Some(metal2vulkan::meta::TextureFormat::Rgba8)
        );

        case.argument_buffer_textures[0].format = TextureFormat::Rgba8Uint;
        let incompatible = product_transform_options_with_reflection(&case, &unspecialized)
            .expect("construct incompatible reflected specialization");
        let error =
            metal2vulkan::reflect_sanitized(air, metal2vulkan::passes::Stage::Kernel, incompatible)
                .expect_err("authored integer format cannot satisfy float AIR texels");
        assert!(error.contains("AIR texels are Float"), "{error}");
    }

    #[test]
    fn fragment_depth_stencil_attachment_has_exact_aspect_bytes_and_selection() {
        let mut case = example();
        case.case_id = "00".repeat(32);
        case.stage = Stage::Fragment;
        case.buffers.clear();
        case.dispatch = None;
        case.draw = Some(Draw {
            primitive: Primitive::Triangle,
            vertex_start: 0,
            vertex_count: 3,
            instance_count: 1,
        });
        case.depth_stencil = Some(DepthStencilResource {
            dimensions: [1, 1],
            initial_depth_b64: Some("AAAAAA==".into()),
            initial_stencil_b64: None,
        });
        case.output = OutputSelection::Depth {
            origin: [0, 0],
            dimensions: [1, 1],
        };
        assert_eq!(case.validate_literal_resources(), Ok(()));

        case.depth_stencil.as_mut().unwrap().initial_depth_b64 = Some("AA==".into());
        assert!(case
            .validate_literal_resources()
            .unwrap_err()
            .iter()
            .any(|error| error.contains("expected 4")));
    }

    #[test]
    fn tessellation_system_values_have_one_integer_scalar_contract() {
        assert!(AttributeFormat::Ushort.supports_tessellation_system_value());
        assert!(AttributeFormat::Int.supports_tessellation_system_value());
        assert!(!AttributeFormat::Ushort2.supports_tessellation_system_value());
        assert!(!AttributeFormat::Float.supports_tessellation_system_value());
        assert!(!AttributeFormat::Uchar.supports_tessellation_system_value());
    }

    #[test]
    fn captured_function_constant_abi_shapes_are_exactly_authorable() {
        assert_eq!(
            ScalarType::from_metal_abi_type_encoding("b"),
            Some((ScalarType::Bool, 1))
        );
        assert_eq!(
            ScalarType::from_metal_abi_type_encoding("Dv4_j"),
            Some((ScalarType::U32, 4))
        );
        assert_eq!(
            ScalarType::from_metal_abi_type_encoding("Dv3_Dh"),
            Some((ScalarType::F16, 3))
        );
        assert!(ScalarType::from_metal_abi_type_encoding("Dv8_j").is_none());
    }

    #[test]
    fn vertex_side_effect_case_uses_draw_without_raster_attachments() {
        let mut case = example();
        case.stage = Stage::Vertex;
        case.dispatch = None;
        case.draw = Some(Draw {
            primitive: Primitive::Point,
            vertex_start: 0,
            vertex_count: 1,
            instance_count: 1,
        });
        case.case_id = case.computed_case_id().unwrap();
        case.validate_literal_resources().unwrap();
        assert!(case.is_rasterization_disabled_vertex());

        case.render_targets.push(RenderTargetResource {
            index: 0,
            format: TextureFormat::Rgba32Float,
            dimensions: [1, 1],
            initial_bytes_b64: "q6urq6urq6urq6urq6urqw==".into(),
        });
        case.case_id = case.computed_case_id().unwrap();
        assert!(case
            .validate_literal_resources()
            .unwrap_err()
            .iter()
            .any(|error| error.contains("does not accept attachments")));
    }

    #[test]
    fn documentation_and_manifest_field_order_do_not_change_identity() {
        let mut case = example();
        let id = case.computed_case_id().unwrap();
        case.rationale = Some("different explanation".into());
        case.authored_by = Some("someone else".into());
        case.name = "renamed-slot".into();
        assert_eq!(case.computed_case_id().unwrap(), id);

        case.case_id = id;
        let json = serde_json::to_value(&case).unwrap();
        let mut entries = json.as_object().unwrap().iter().collect::<Vec<_>>();
        entries.reverse();
        let reversed = serde_json::Value::Object(
            entries
                .into_iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        );
        let reparsed: AuthoredCase = serde_json::from_value(reversed).unwrap();
        assert_eq!(reparsed.computed_case_id().unwrap(), case.case_id);
    }

    #[test]
    fn authored_bounded_execution_requires_an_explicit_bound_rationale() {
        let mut case = example();
        case.execution_safety = ExecutionSafety::AuthoredBounded;
        case.rationale = None;
        case.case_id = case.computed_case_id().unwrap();
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(errors.contains("finite loop bound"), "{errors}");

        case.rationale = Some("input count=4 bounds the only loop to four iterations".into());
        case.validate_literal_resources().unwrap();
    }

    #[test]
    fn resource_array_order_does_not_change_semantic_or_input_identity() {
        let mut first = example();
        first.buffers.push(BufferResource {
            binding: 7,
            role: ResourceRole::Input,
            bytes_b64: Some("AQIDBA==".into()),
            initial_bytes_b64: None,
        });
        first.case_id = first.computed_case_id().unwrap();
        let first_input = first.computed_input_sha256().unwrap();

        let mut reordered = first.clone();
        reordered.buffers.reverse();
        assert_eq!(reordered.computed_case_id().unwrap(), first.case_id);
        assert_eq!(reordered.computed_input_sha256().unwrap(), first_input);
    }

    #[test]
    fn direct_visible_function_reference_is_semantic_input() {
        let mut case = example();
        let original_case = case.computed_case_id().unwrap();
        case.case_id = original_case.clone();
        let original_input = case.computed_input_sha256().unwrap();
        case.visible_function_references
            .push(LinkedFunctionResource {
                module_sha256: "22".repeat(32),
                function: "linked".into(),
            });
        assert_ne!(case.computed_case_id().unwrap(), original_case);
        assert_ne!(case.computed_input_sha256().unwrap(), original_input);
    }

    #[test]
    fn canonical_identity_and_input_digest_are_golden() {
        let mut case = example();
        case.case_id = case.computed_case_id().unwrap();
        assert_eq!(
            case.case_id,
            "209b8f187ed6b052352d63ba7c8445362bab6b49c61e9b96006522bb7d880a82"
        );
        assert_eq!(
            case.computed_input_sha256().unwrap(),
            "f4627a35e4ef25183ec50b1b61fa81231185fbe6792ffa36a41ac9ac9af477ee"
        );
    }

    #[test]
    fn malformed_resources_are_rejected_without_repair() {
        let mut case = example();
        case.buffers[0].initial_bytes_b64 = None;
        case.buffers[0].bytes_b64 = Some("not base64".into());
        case.dispatch.as_mut().unwrap().grid[0] = 0;
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(errors.contains("dimensions must be nonzero"), "{errors}");
        assert!(
            errors.contains("requires only initial_bytes_b64"),
            "{errors}"
        );
    }

    #[test]
    fn function_table_capacity_is_independent_of_populated_slots() {
        let mut case = example();
        case.intersection_function_tables
            .push(IntersectionFunctionTableResource {
                binding: 6,
                size: 4,
                entries: vec![],
            });
        case.case_id = case.computed_case_id().unwrap();
        case.validate_literal_resources().unwrap();

        case.intersection_function_tables[0].size = 0;
        case.case_id = case.computed_case_id().unwrap();
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(errors.contains("size must be nonzero"), "{errors}");

        case.intersection_function_tables[0] = IntersectionFunctionTableResource {
            binding: 6,
            size: 2,
            entries: vec![IntersectionFunctionTableEntry::Linked {
                index: 2,
                module_sha256: "22".repeat(32),
                function: "intersection".into(),
            }],
        };
        case.case_id = case.computed_case_id().unwrap();
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(errors.contains("entry 2 exceeds size 2"), "{errors}");
    }

    #[test]
    fn opaque_triangle_is_a_typed_populated_intersection_slot() {
        let mut case = example();
        case.intersection_function_tables
            .push(IntersectionFunctionTableResource {
                binding: 6,
                size: 1,
                entries: vec![IntersectionFunctionTableEntry::OpaqueTriangle {
                    index: 0,
                    signature: vec![
                        IntersectionFunctionSignature::TriangleData,
                        IntersectionFunctionSignature::IntersectionFunctionBuffer,
                    ],
                }],
            });
        case.case_id = case.computed_case_id().unwrap();
        case.validate_literal_resources().unwrap();
        let json = serde_json::to_string(&case).unwrap();
        assert!(json.contains(r#""kind":"opaque_triangle""#));
        assert!(!json.contains("module_sha256"));

        case.intersection_function_tables[0].entries =
            vec![IntersectionFunctionTableEntry::OpaqueTriangle {
                index: 0,
                signature: vec![
                    IntersectionFunctionSignature::IntersectionFunctionBuffer,
                    IntersectionFunctionSignature::TriangleData,
                ],
            }];
        case.case_id = case.computed_case_id().unwrap();
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(
            errors.contains("signature flags must be sorted and unique"),
            "{errors}"
        );
    }

    #[test]
    fn argument_buffer_intersection_table_uses_owner_and_field_identity() {
        let mut case = example();
        case.argument_buffer_intersection_function_tables.push(
            ArgumentBufferIntersectionFunctionTableResource {
                buffer_binding: 0,
                field_offset: 8,
                size: 1,
                entries: vec![IntersectionFunctionTableEntry::OpaqueTriangle {
                    index: 0,
                    signature: vec![IntersectionFunctionSignature::IntersectionFunctionBuffer],
                }],
            },
        );
        case.case_id = case.computed_case_id().unwrap();
        case.validate_literal_resources().unwrap();

        case.argument_buffer_intersection_function_tables[0].buffer_binding = 31;
        case.case_id = case.computed_case_id().unwrap();
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(errors.contains("owns an undeclared buffer"), "{errors}");
    }

    #[test]
    fn missing_and_unknown_manifest_fields_are_rejected() {
        let case = example();
        let mut value = serde_json::to_value(case).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("guessed_seed".into(), serde_json::json!(7));
        assert!(serde_json::from_value::<AuthoredCase>(value).is_err());

        let mut value = serde_json::to_value(example()).unwrap();
        value.as_object_mut().unwrap().remove("output");
        assert!(serde_json::from_value::<AuthoredCase>(value).is_err());
    }

    #[test]
    fn draw_inputs_and_bool_constants_are_range_checked() {
        let mut case = example();
        case.stage = Stage::Vertex;
        case.dispatch = None;
        case.draw = Some(Draw {
            primitive: Primitive::Triangle,
            vertex_start: 1,
            vertex_count: 3,
            instance_count: 1,
        });
        case.vertex_inputs.push(VertexInput {
            location: 0,
            format: AttributeFormat::Float,
            stride: 4,
            bytes_b64: "AAAAAAAAAAA=".into(),
        });
        case.function_constants.push(FunctionConstant {
            index: 0,
            scalar_type: ScalarType::Bool,
            lanes: 1,
            bytes_b64: "Ag==".into(),
        });
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(
            errors.contains("execution requires at least 16"),
            "{errors}"
        );
        assert!(
            errors.contains("bool must be encoded as 0 or 1"),
            "{errors}"
        );
    }

    #[test]
    fn kernel_stage_inputs_use_the_products_runtime_array_stride() {
        let mut case = example();
        case.case_id = "22".repeat(32);
        let original_id = case.computed_case_id().unwrap();
        case.kernel_stage_inputs.push(KernelStageInput {
            location: 6,
            format: AttributeFormat::Uint3,
            stride: 12,
            bytes_b64: "AAAAAAAAAAAAAAAAAAAAAA==".into(),
        });
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(
            errors.contains("must equal product storage-buffer stride 16"),
            "{errors}"
        );
        case.kernel_stage_inputs[0].stride = 16;
        assert!(case.validate_literal_resources().is_ok());
        assert_ne!(original_id, case.computed_case_id().unwrap());

        case.kernel_stage_inputs[0].format = AttributeFormat::Half3;
        case.kernel_stage_inputs[0].stride = 8;
        case.kernel_stage_inputs[0].bytes_b64 = "AAAAAAAAAAA=".into();
        assert!(case.validate_literal_resources().is_ok());
    }

    #[test]
    fn acceleration_structures_are_semantic_and_cannot_alias_buffers() {
        let mut case = example();
        let original_id = case.computed_case_id().unwrap();
        case.acceleration_structures
            .push(AccelerationStructureResource {
                binding: 0,
                kind: AccelerationStructureKind::Instance,
                primitive_triangles_b64: None,
                child_references: vec![0x1234],
            });
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(errors.contains("both a buffer and an acceleration structure"));
        assert_ne!(original_id, case.computed_case_id().unwrap());
        assert_ne!(
            case.computed_input_sha256().unwrap(),
            example().computed_input_sha256().unwrap()
        );
    }

    #[test]
    fn primitive_acceleration_structures_require_exact_finite_triangle_bytes() {
        let mut case = example();
        case.case_id = "22".repeat(32);
        case.acceleration_structures
            .push(AccelerationStructureResource {
                binding: 5,
                kind: AccelerationStructureKind::Primitive,
                primitive_triangles_b64: Some(
                    base64::engine::general_purpose::STANDARD.encode([0u8; 35]),
                ),
                child_references: vec![],
            });
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(errors.contains("36-byte float3 triangles"), "{errors}");

        let mut bytes = [0u8; 36];
        bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        case.acceleration_structures[0].primitive_triangles_b64 =
            Some(base64::engine::general_purpose::STANDARD.encode(bytes));
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(errors.contains("is not finite"), "{errors}");

        case.acceleration_structures[0].primitive_triangles_b64 =
            Some(base64::engine::general_purpose::STANDARD.encode([0u8; 36]));
        if let Err(errors) = case.validate_literal_resources() {
            panic!("{}", errors.join("\n"));
        }
    }

    #[test]
    fn argument_buffer_textures_use_structural_identity_and_literal_output() {
        let mut case = example();
        case.argument_buffer_textures
            .push(ArgumentBufferTextureResource {
                buffer_binding: 0,
                field_offset: 16,
                role: ResourceRole::Output,
                texture_type: TextureType::D2,
                format: TextureFormat::R32Uint,
                dimensions: [1, 1, 1],
                sample_count: 1,
                bytes_b64: None,
                initial_bytes_b64: Some("q6urqw==".into()),
            });
        case.output = OutputSelection::ArgumentBufferTexture {
            buffer_binding: 0,
            field_offset: 16,
            origin: [0, 0, 0],
            dimensions: [1, 1, 1],
        };
        case.case_id = case.computed_case_id().unwrap();
        case.validate_literal_resources().unwrap();
        let resources = crate::literal::LiteralResources::prepare(&case).unwrap();
        assert_eq!(resources.argument_buffer_textures.len(), 1);
        assert_eq!(
            resources.argument_buffer_textures[0]
                .select([0, 0, 0], [1, 1, 1])
                .unwrap(),
            [0xab; 4]
        );
    }

    #[test]
    fn argument_buffer_buffers_are_owned_literals_and_selectable_outputs() {
        let mut case = example();
        case.argument_buffer_buffers
            .push(ArgumentBufferBufferResource {
                buffer_binding: 0,
                field_offset: 0,
                role: ResourceRole::Output,
                bytes_b64: None,
                initial_bytes_b64: Some("q6urqw==".into()),
            });
        case.output = OutputSelection::ArgumentBufferBuffer {
            buffer_binding: 0,
            field_offset: 0,
            offset: 0,
            length: 4,
        };
        case.case_id = case.computed_case_id().unwrap();
        case.validate_literal_resources().unwrap();
        let resources = crate::literal::LiteralResources::prepare(&case).unwrap();
        assert_eq!(resources.argument_buffer_buffers.len(), 1);
        assert_eq!(resources.argument_buffer_buffers[0].bytes, [0xab; 4]);
    }

    #[test]
    fn texture_array_elements_are_ordered_literals_and_reserve_metal_slots() {
        let mut case = example();
        case.texture_arrays.push(TextureArrayResource {
            binding: 4,
            role: ResourceRole::Input,
            texture_type: TextureType::D2,
            format: TextureFormat::R32Uint,
            sample_count: 1,
            elements: vec![
                TextureArrayElement {
                    dimensions: [1, 1, 1],
                    bytes_b64: Some("AQAAAA==".into()),
                    initial_bytes_b64: None,
                },
                TextureArrayElement {
                    dimensions: [1, 1, 1],
                    bytes_b64: Some("AgAAAA==".into()),
                    initial_bytes_b64: None,
                },
            ],
        });
        case.case_id = case.computed_case_id().unwrap();
        case.validate_literal_resources().unwrap();
        let resources = crate::literal::LiteralResources::prepare(&case).unwrap();
        assert_eq!(resources.texture_arrays[0].elements[0].bytes, [1, 0, 0, 0]);
        assert_eq!(resources.texture_arrays[0].elements[1].bytes, [2, 0, 0, 0]);

        case.textures.push(TextureResource {
            binding: 5,
            role: ResourceRole::Input,
            texture_type: TextureType::D2,
            format: TextureFormat::R32Uint,
            dimensions: [1, 1, 1],
            sample_count: 1,
            bytes_b64: Some("AwAAAA==".into()),
            initial_bytes_b64: None,
        });
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(errors.contains("overlaps Metal texture slot 5"), "{errors}");
    }

    #[test]
    fn multisample_array_literals_keep_layers_and_samples_independent() {
        use base64::Engine as _;

        let mut case = example();
        case.textures.push(TextureResource {
            binding: 3,
            role: ResourceRole::Input,
            texture_type: TextureType::D2MultisampleArray,
            format: TextureFormat::R32Uint,
            dimensions: [2, 1, 3],
            sample_count: 4,
            bytes_b64: Some(base64::engine::general_purpose::STANDARD.encode([0u8; 96])),
            initial_bytes_b64: None,
        });
        case.case_id = case.computed_case_id().unwrap();
        case.validate_literal_resources().unwrap();
        let resources = crate::literal::LiteralResources::prepare(&case).unwrap();
        let layout = resources.textures[0].layout().unwrap();
        assert_eq!(layout.array_layers, 3);
        assert_eq!(layout.sample_count, 4);

        case.textures[0].role = ResourceRole::Output;
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(errors.contains("multisample texture binding 3 must have input role"));
    }

    #[test]
    fn imageblock_dimensions_are_explicit_and_match_product_local_size() {
        let mut case = example();
        case.imageblock = Some(ImageblockResource {
            dimensions: [1, 1],
            implicit_coverage: None,
        });
        case.case_id = case.computed_case_id().unwrap();
        case.validate_literal_resources().unwrap();

        case.imageblock.as_mut().unwrap().dimensions = [2, 1];
        let errors = case.validate_literal_resources().unwrap_err().join("\n");
        assert!(
            errors.contains("must equal threadgroup x/y dimensions"),
            "{errors}"
        );
    }
}
