//! Public shader-reflection facade.
//!
//! The translator already parses everything a downstream consumer needs to bind a shader — the
//! stage-interface metadata in [`crate::meta`] (`FragMeta`/`VertMeta`/`KernMeta`) plus the selected
//! descriptor-ABI convention the stage-input/output passes apply (`crate::passes::stage_input` and
//! `crate::passes::stage_output`). Historically
//! that knowledge was DROPPED at the crate boundary: `translate_*` returned bare `Result<Vec<u8>,
//! String>`, forcing consumers to re-reflect the produced SPIR-V and re-hardcode the ABI bases.
//!
//! This module exposes that knowledge as one consumer-shaped [`ShaderReflection`] value. Interface
//! declarations come from the parser-shaped AIR metadata and the ABI constants below; buffer access
//! footprints are then derived from the final constructed SPIR-V module before its owned carrier is
//! released. The binding numbers here are the SAME ones the interface pass decorates. The default uses
//! [`RESOURCE_DESCRIPTOR_SET`] and the
//! exported default ranges; [`DescriptorLayout`] can select a complete stage-local alternative.
//! Reflection never mutates the module, so reflected and non-reflected
//! translation remain byte-identical.

use crate::meta::{
    texture_shape_from_name, AirType, BufferAccess, FragMeta, FragRole, FunctionConstant, KernMeta,
    KernRole, TextureComponent, TextureDimension, TextureShape, VertMeta, VertOutRole, VertRole,
};
use crate::spirv_module::Module;

mod footprint;

/// Schema version of [`ShaderReflection`]. Bump on any breaking change to the serialized shape so a
/// consumer's persisted reflection cache invalidates cleanly rather than deserializing stale fields.
///
/// Notable schema milestones (see `CHANGELOG.md` for release-level history):
///
/// v2 added the core consumer-readiness fields — per-binding typed
/// `texture_shape` (dimension/arrayed/multisampled/component/writable/storage_format) and
/// `embedded_source`; stage-level `vertex_builtins`, `imageblock_layouts`, `function_constants`, and
/// the source `datalayout`; plus fragment/vertex buffer `address_space`/`declared_size` population.
///
/// v3 reports AIR-embedded constexpr samplers as `StaticSampler` bindings with their decoded state.
///
/// v4 adds conservative buffer extent classes, all-stage buffer type names / declared sizes, and
/// declared buffer access (including write-only).
///
/// v5 adds the Metal argument-encoder index for embedded argument-buffer textures.
///
/// v6 adds descriptor counts and fixed texture-handle array lengths.
///
/// v7 exposes kernel stage-input attributes as reflected read-only buffer resources, including both
/// their AIR attribute location and the shared synthetic Metal/Vulkan buffer slot.
///
/// v14 exposes device buffers embedded in AIR argument buffers and their nested Metal resource
/// index, enabling consumers to encode the Metal handle and populate the Vulkan device address.
/// v18 preserves each function constant's Metal ABI type encoding so consumers can bind exact
/// signed scalar/vector values rather than guessing from LLVM's signless integer types.
///
/// v21 adds conservative static and invocation-strided buffer access footprints. The footprint is
/// derived from the same finished owned SPIR-V module that supplies the returned bytes.
///
/// v22 replaces the overlapping 32-wide descriptor bases with checked, non-overlapping resource
/// bands and reserves a separate high range for translator-owned descriptors.
/// v24 records caller-provided runtime storage-image format specialization and the exact explicit
/// SPIR-V format (or formatless `Unknown`) emitted for each Metal texture index.
/// v25 extends that runtime specialization contract to writable textures embedded in argument
/// buffers, keyed by their reflected synthetic resource index.
/// v26 rejects component-incompatible runtime formats at the metadata-only reflection boundary,
/// matching executable translation's specialization contract.
/// v27 records the versioned effective descriptor layout used by the returned SPIR-V.
/// v28 records the original kernel dispatch-bound contract.
/// v29 makes that per-dispatch push-constant grid the safe default for every kernel; whole-workgroup
/// dispatch is now an explicit caller assertion.
/// v30 replaces surplus-invocation culling with exact boundary-region decomposition. Exact-thread
/// kernels expose three local-size specialization constants and a complete logical-grid payload.
/// v31 reports implicit imageblock render-target planes at every stage whose module carries the
/// `air.load/store.implicit_imageblock.*` intrinsics, not only in kernels. The interface pass
/// materializes the plane from the call, which is a property of the body rather than of the stage.
/// Reflected translation also reports the buffer-address table the finished module declares instead
/// of the one an AIR text scan predicted.
/// v37 reports a buffer member AIR names without describing as `AirType::Opaque { size }`. Earlier
/// versions reported every such member as a 32-bit `Float`, which named a type AIR never stated and
/// understated members of any other size.
/// v38 decides `reflect_sanitized`'s buffer-address table with the emitter's own device-address
/// predicate instead of an AIR text scan. Over 2880 corpus sources the scan disagreed with the
/// finished module 63 times and the predicate disagrees 8 times; reflected translation, which reads
/// the table off the module, is unchanged.
pub const REFLECTION_VERSION: u32 = 38;

/// Size in bytes of the twelve tightly packed `u32` values used by exact-thread dispatches: thread
/// grid, thread base, threadgroup base, and total threadgroup grid (three dimensions each).
pub const KERNEL_DISPATCH_PUSH_CONSTANT_SIZE: u32 = 48;

/// Vulkan specialization-constant ids which select the exact local size of a decomposed dispatch
/// region. Consumers must specialize all three values when creating each region pipeline.
pub const KERNEL_LOCAL_SIZE_SPEC_IDS: [u32; 3] = [0, 1, 2];

/// Default byte offset of the exact-thread dispatch-region payload.
pub const DEFAULT_KERNEL_DISPATCH_PUSH_CONSTANT_OFFSET: u32 = 0;

/// Reflected byte range occupied by the dynamic kernel grid in Vulkan push-constant storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KernelDispatchPushConstantRange {
    pub offset: u32,
    pub size: u32,
}

/// How a translated compute kernel obtains the exact Metal execution grid.
///
/// Vulkan gives one pipeline a fixed workgroup size. Metal `dispatchThreads` can make each boundary
/// workgroup smaller, so exact-thread launches are decomposed into at most eight rectangular
/// regions. Each region gets its exact local size plus logical thread and threadgroup bases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KernelDispatch {
    /// The caller used whole workgroups. `[[threads_per_grid]]` is derived as
    /// `NumWorkgroups * LocalSize`, and no invocation cull is needed.
    Workgroups,
    /// A fixed Metal `dispatchThreads` grid baked into this module.
    ThreadsFixed { threads_per_grid: [u32; 3] },
    /// A Metal `dispatchThreads` grid selected at dispatch time.
    ThreadsDynamic { offset: u32 },
}

impl KernelDispatch {
    /// Safe default for a kernel whose per-dispatch launch form is not fixed at translation time.
    pub const fn safe_default() -> Self {
        Self::ThreadsDynamic {
            offset: DEFAULT_KERNEL_DISPATCH_PUSH_CONSTANT_OFFSET,
        }
    }
}

impl KernelDispatch {
    pub fn validate(self) -> Result<(), String> {
        let offset = match self {
            Self::Workgroups => return Ok(()),
            Self::ThreadsFixed { .. } => DEFAULT_KERNEL_DISPATCH_PUSH_CONSTANT_OFFSET,
            Self::ThreadsDynamic { offset } => offset,
        };
        if offset % 4 != 0 {
            return Err(format!(
                "kernel grid push-constant offset {offset} is not 4-byte aligned"
            ));
        }
        offset
            .checked_add(KERNEL_DISPATCH_PUSH_CONSTANT_SIZE)
            .ok_or_else(|| "kernel grid push-constant range overflows u32".to_string())?;
        Ok(())
    }

    /// Push-constant byte range the consumer must make visible to the compute stage.
    pub const fn push_constant_range(self) -> Option<KernelDispatchPushConstantRange> {
        match self {
            Self::ThreadsDynamic { offset }
                if offset
                    .checked_add(KERNEL_DISPATCH_PUSH_CONSTANT_SIZE)
                    .is_some() =>
            {
                Some(KernelDispatchPushConstantRange {
                    offset,
                    size: KERNEL_DISPATCH_PUSH_CONSTANT_SIZE,
                })
            }
            Self::ThreadsFixed { .. } => Some(KernelDispatchPushConstantRange {
                offset: DEFAULT_KERNEL_DISPATCH_PUSH_CONSTANT_OFFSET,
                size: KERNEL_DISPATCH_PUSH_CONSTANT_SIZE,
            }),
            Self::Workgroups => None,
            Self::ThreadsDynamic { .. } => None,
        }
    }

    /// Build the exact sequence of Vulkan dispatch regions for this Metal launch.
    pub fn plan(
        self,
        nominal_local_size: [u32; 3],
        dynamic_threads_per_grid: Option<[u32; 3]>,
    ) -> Result<KernelDispatchPlan, String> {
        if nominal_local_size.contains(&0) {
            return Err("kernel local-size dimensions must be non-zero".to_string());
        }
        let threads_per_grid = match self {
            Self::Workgroups => {
                return Err("whole-workgroup dispatches do not use an exact-thread plan".to_string())
            }
            Self::ThreadsFixed { threads_per_grid } => {
                if dynamic_threads_per_grid.is_some_and(|grid| grid != threads_per_grid) {
                    return Err(
                        "runtime thread grid does not match fixed kernel dispatch".to_string()
                    );
                }
                threads_per_grid
            }
            Self::ThreadsDynamic { .. } => dynamic_threads_per_grid
                .ok_or_else(|| "dynamic kernel dispatch requires a thread grid".to_string())?,
        };
        KernelDispatchPlan::new(threads_per_grid, nominal_local_size)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KernelDispatchRegion {
    pub local_size: [u32; 3],
    pub group_count: [u32; 3],
    pub thread_base: [u32; 3],
    pub threadgroup_base: [u32; 3],
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KernelDispatchPlan {
    pub threads_per_grid: [u32; 3],
    pub threadgroups_per_grid: [u32; 3],
    pub regions: Vec<KernelDispatchRegion>,
}

impl KernelDispatchPlan {
    /// Encode one returned region's complete push-constant payload.
    pub fn push_constants(&self, region: KernelDispatchRegion) -> [u32; 12] {
        let mut words = [0; 12];
        words[0..3].copy_from_slice(&self.threads_per_grid);
        words[3..6].copy_from_slice(&region.thread_base);
        words[6..9].copy_from_slice(&region.threadgroup_base);
        words[9..12].copy_from_slice(&self.threadgroups_per_grid);
        words
    }

    fn new(threads_per_grid: [u32; 3], nominal_local_size: [u32; 3]) -> Result<Self, String> {
        let threadgroups_per_grid = std::array::from_fn(|dimension| {
            threads_per_grid[dimension].div_ceil(nominal_local_size[dimension])
        });
        let mut regions = Vec::with_capacity(8);
        for mask in 0_u8..8 {
            let mut local_size = nominal_local_size;
            let mut group_count = [0; 3];
            let mut thread_base = [0; 3];
            let mut threadgroup_base = [0; 3];
            let mut nonempty = true;
            for dimension in 0..3 {
                let full = threads_per_grid[dimension] / nominal_local_size[dimension];
                let tail = threads_per_grid[dimension] % nominal_local_size[dimension];
                if mask & (1 << dimension) == 0 {
                    group_count[dimension] = full;
                } else if tail == 0 {
                    nonempty = false;
                } else {
                    local_size[dimension] = tail;
                    group_count[dimension] = 1;
                    thread_base[dimension] = full * nominal_local_size[dimension];
                    threadgroup_base[dimension] = full;
                }
                nonempty &= group_count[dimension] != 0;
            }
            if nonempty {
                regions.push(KernelDispatchRegion {
                    local_size,
                    group_count,
                    thread_base,
                    threadgroup_base,
                });
            }
        }
        Ok(Self {
            threads_per_grid,
            threadgroups_per_grid,
            regions,
        })
    }
}

/// Version of the descriptor-layout contract and its default values.
pub const DESCRIPTOR_LAYOUT_VERSION: u32 = 1;

/// Default descriptor set for every Metal-facing resource.
pub const RESOURCE_DESCRIPTOR_SET: u32 = 0;

/// One half-open descriptor-binding band. `binding(index)` projects a Metal resource index into the
/// selected ABI band; out-of-band indices remain unsupported
/// instead of saturating into another resource class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DescriptorBindingRange {
    pub start: u32,
    pub end: u32,
}

impl DescriptorBindingRange {
    pub const fn from_base_count(base: u32, count: u32) -> Result<Self, DescriptorLayoutError> {
        match base.checked_add(count) {
            Some(end) => Ok(Self { start: base, end }),
            None => Err(DescriptorLayoutError::RangeOverflow { base, count }),
        }
    }

    pub const fn binding(self, index: u32) -> Option<u32> {
        let Some(width) = self.end.checked_sub(self.start) else {
            return None;
        };
        if index < width {
            Some(self.start + index)
        } else {
            None
        }
    }

    pub const fn contains(self, binding: u32) -> bool {
        binding >= self.start && binding < self.end
    }

    pub const fn len(self) -> Option<u32> {
        self.end.checked_sub(self.start)
    }

    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// Descriptor binding base for `[[buffer(n)]]` resources: the binding is the Metal buffer index `n`
/// directly (range `0..32`).
pub const BUFFER_BINDING_BASE: u32 = 0;
pub const BUFFER_BINDING_RANGE: DescriptorBindingRange =
    DescriptorBindingRange { start: 0, end: 32 };
/// Descriptor binding base for `[[texture(n)]]` resources: binding = `TEXTURE_BINDING_BASE + n`.
pub const TEXTURE_BINDING_BASE: u32 = 32;
/// Number of texture argument-table entries exposed by the Metal ABI. This source limit is
/// independent of the selected descriptor layout's capacity.
pub const TEXTURE_ARGUMENT_COUNT: u32 = 128;
pub const TEXTURE_ARGUMENT_COUNT_USIZE: usize = TEXTURE_ARGUMENT_COUNT as usize;
pub const TEXTURE_BINDING_RANGE: DescriptorBindingRange = DescriptorBindingRange {
    start: TEXTURE_BINDING_BASE,
    end: TEXTURE_BINDING_BASE + TEXTURE_ARGUMENT_COUNT,
};
/// Descriptor binding base for `[[sampler(n)]]` resources: binding = `SAMPLER_BINDING_BASE + n`.
pub const SAMPLER_BINDING_BASE: u32 = TEXTURE_BINDING_RANGE.end;
pub const SAMPLER_BINDING_RANGE: DescriptorBindingRange = DescriptorBindingRange {
    start: SAMPLER_BINDING_BASE,
    end: 192,
};
/// Number of sampler argument-table entries exposed by the Metal ABI. The remainder of
/// [`SAMPLER_BINDING_RANGE`] is reserved for synthesized static samplers.
pub const SAMPLER_ARGUMENT_COUNT: u32 = 16;
pub const SAMPLER_ARGUMENT_COUNT_USIZE: usize = SAMPLER_ARGUMENT_COUNT as usize;
/// Descriptor binding base for `[[color(n)]]` framebuffer-fetch inputs (Vulkan input attachments):
/// binding = `COLOR_INPUT_BINDING_BASE + n`.
pub const COLOR_INPUT_BINDING_BASE: u32 = SAMPLER_BINDING_RANGE.end;
pub const COLOR_INPUT_BINDING_RANGE: DescriptorBindingRange = DescriptorBindingRange {
    start: COLOR_INPUT_BINDING_BASE,
    end: 200,
};
/// Descriptor binding base for implicit imageblock render-target planes. Each attachment occupies
/// three storage-image bindings, one for AIR data rates 0 (default), 1 (color), and 2 (sample).
pub const IMAGEBLOCK_BINDING_BASE: u32 = COLOR_INPUT_BINDING_RANGE.end;
pub const IMAGEBLOCK_DATA_RATE_STRIDE: u32 = 3;
pub const IMAGEBLOCK_BINDING_RANGE: DescriptorBindingRange = DescriptorBindingRange {
    start: IMAGEBLOCK_BINDING_BASE,
    end: 224,
};
/// Descriptor binding base for custom fragment `[[imageblock_data]]` master fields. Each master
/// member occupies one storage-image binding, preserving independent formats and raster-order
/// groups without conflating custom tile data with color-attachment imageblocks.
pub const FRAGMENT_IMAGEBLOCK_BINDING_BASE: u32 = IMAGEBLOCK_BINDING_RANGE.end;
pub const FRAGMENT_IMAGEBLOCK_BINDING_RANGE: DescriptorBindingRange = DescriptorBindingRange {
    start: FRAGMENT_IMAGEBLOCK_BINDING_BASE,
    end: 480,
};
/// Descriptor band for Metal textures emitted as Vulkan storage images. Sampled and storage
/// images are distinct Vulkan descriptor types, so even conditional AIR arguments cannot share the
/// sampled-texture binding for the same Metal index.
pub const STORAGE_TEXTURE_BINDING_BASE: u32 = FRAGMENT_IMAGEBLOCK_BINDING_RANGE.end;
pub const STORAGE_TEXTURE_BINDING_RANGE: DescriptorBindingRange = DescriptorBindingRange {
    start: STORAGE_TEXTURE_BINDING_BASE,
    end: STORAGE_TEXTURE_BINDING_BASE + TEXTURE_ARGUMENT_COUNT,
};
/// Start of the default translator-owned descriptor range (currently direct-buffer address tables),
/// after every fixed Metal-facing band.
pub const SYNTHETIC_BINDING_BASE: u32 = 640;

/// Complete descriptor layout selected for one independently translated stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DescriptorLayout {
    pub version: u32,
    pub set: u32,
    pub buffers: DescriptorBindingRange,
    pub sampled_textures: DescriptorBindingRange,
    pub samplers: DescriptorBindingRange,
    pub color_inputs: DescriptorBindingRange,
    pub imageblocks: DescriptorBindingRange,
    pub fragment_imageblocks: DescriptorBindingRange,
    pub storage_textures: DescriptorBindingRange,
    pub synthetic: DescriptorBindingRange,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorLayoutError {
    UnsupportedVersion {
        actual: u32,
        expected: u32,
    },
    ReversedRange {
        class: &'static str,
        start: u32,
        end: u32,
    },
    OverlappingRanges {
        left: &'static str,
        left_range: DescriptorBindingRange,
        right: &'static str,
        right_range: DescriptorBindingRange,
    },
    RangeOverflow {
        base: u32,
        count: u32,
    },
}

impl std::fmt::Display for DescriptorLayoutError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVersion { actual, expected } => write!(
                formatter,
                "descriptor layout version {actual} is unsupported; expected {expected}"
            ),
            Self::ReversedRange { class, start, end } => write!(
                formatter,
                "descriptor layout {class} range [{start},{end}) is reversed"
            ),
            Self::OverlappingRanges {
                left,
                left_range,
                right,
                right_range,
            } => write!(
                formatter,
                "descriptor layout ranges {left} [{},{}) and {right} [{},{}) overlap",
                left_range.start, left_range.end, right_range.start, right_range.end
            ),
            Self::RangeOverflow { base, count } => write!(
                formatter,
                "descriptor binding range base {base} plus count {count} overflows u32"
            ),
        }
    }
}

impl std::error::Error for DescriptorLayoutError {}

pub const DEFAULT_DESCRIPTOR_LAYOUT: DescriptorLayout = DescriptorLayout {
    version: DESCRIPTOR_LAYOUT_VERSION,
    set: RESOURCE_DESCRIPTOR_SET,
    buffers: BUFFER_BINDING_RANGE,
    sampled_textures: TEXTURE_BINDING_RANGE,
    samplers: SAMPLER_BINDING_RANGE,
    color_inputs: COLOR_INPUT_BINDING_RANGE,
    imageblocks: IMAGEBLOCK_BINDING_RANGE,
    fragment_imageblocks: FRAGMENT_IMAGEBLOCK_BINDING_RANGE,
    storage_textures: STORAGE_TEXTURE_BINDING_RANGE,
    synthetic: DescriptorBindingRange {
        start: SYNTHETIC_BINDING_BASE,
        end: SYNTHETIC_BINDING_BASE + 32,
    },
};

impl Default for DescriptorLayout {
    fn default() -> Self {
        DEFAULT_DESCRIPTOR_LAYOUT
    }
}

impl DescriptorLayout {
    pub fn validate(self) -> Result<(), DescriptorLayoutError> {
        if self.version != DESCRIPTOR_LAYOUT_VERSION {
            return Err(DescriptorLayoutError::UnsupportedVersion {
                actual: self.version,
                expected: DESCRIPTOR_LAYOUT_VERSION,
            });
        }
        let ranges = [
            ("buffers", self.buffers),
            ("sampled textures", self.sampled_textures),
            ("samplers", self.samplers),
            ("color inputs", self.color_inputs),
            ("imageblocks", self.imageblocks),
            ("fragment imageblocks", self.fragment_imageblocks),
            ("storage textures", self.storage_textures),
            ("synthetic descriptors", self.synthetic),
        ];
        for (name, range) in ranges {
            if range.start > range.end {
                return Err(DescriptorLayoutError::ReversedRange {
                    class: name,
                    start: range.start,
                    end: range.end,
                });
            }
        }
        for (index, (left_name, left)) in ranges.iter().copied().enumerate() {
            if left.is_empty() {
                continue;
            }
            for (right_name, right) in ranges.iter().copied().skip(index + 1) {
                if !right.is_empty() && left.start < right.end && right.start < left.end {
                    return Err(DescriptorLayoutError::OverlappingRanges {
                        left: left_name,
                        left_range: left,
                        right: right_name,
                        right_range: right,
                    });
                }
            }
        }
        Ok(())
    }

    pub const fn buffer_binding(self, index: u32) -> Option<u32> {
        self.buffers.binding(index)
    }

    pub const fn sampled_texture_binding(self, index: u32) -> Option<u32> {
        self.sampled_textures.binding(index)
    }

    pub const fn storage_texture_binding(self, index: u32) -> Option<u32> {
        self.storage_textures.binding(index)
    }

    pub const fn sampler_binding(self, index: u32) -> Option<u32> {
        if index < SAMPLER_ARGUMENT_COUNT {
            self.samplers.binding(index)
        } else {
            None
        }
    }

    pub const fn color_input_binding(self, index: u32) -> Option<u32> {
        self.color_inputs.binding(index)
    }

    pub const fn imageblock_binding(self, attachment: u32, data_rate: u32) -> Option<u32> {
        if data_rate >= IMAGEBLOCK_DATA_RATE_STRIDE {
            return None;
        }
        let Some(offset) = attachment.checked_mul(IMAGEBLOCK_DATA_RATE_STRIDE) else {
            return None;
        };
        let Some(offset) = offset.checked_add(data_rate) else {
            return None;
        };
        self.imageblocks.binding(offset)
    }

    pub const fn fragment_imageblock_binding(self, member: u32) -> Option<u32> {
        self.fragment_imageblocks.binding(member)
    }
}

pub const fn buffer_resource_binding(index: u32) -> Option<u32> {
    BUFFER_BINDING_RANGE.binding(index)
}

pub const fn texture_resource_binding(index: u32) -> Option<u32> {
    TEXTURE_BINDING_RANGE.binding(index)
}

pub const fn storage_texture_resource_binding(index: u32) -> Option<u32> {
    STORAGE_TEXTURE_BINDING_RANGE.binding(index)
}

pub const fn sampler_resource_binding(index: u32) -> Option<u32> {
    if index < SAMPLER_ARGUMENT_COUNT {
        SAMPLER_BINDING_RANGE.binding(index)
    } else {
        None
    }
}

pub const fn color_input_resource_binding(index: u32) -> Option<u32> {
    COLOR_INPUT_BINDING_RANGE.binding(index)
}

/// Storage-image descriptor used to emulate one implicit imageblock render-target/data-rate plane.
pub const fn imageblock_resource_binding(attachment: u32, data_rate: u32) -> Option<u32> {
    if data_rate >= IMAGEBLOCK_DATA_RATE_STRIDE {
        return None;
    }
    let Some(offset) = attachment.checked_mul(IMAGEBLOCK_DATA_RATE_STRIDE) else {
        return None;
    };
    let Some(offset) = offset.checked_add(data_rate) else {
        return None;
    };
    IMAGEBLOCK_BINDING_RANGE.binding(offset)
}

pub const fn fragment_imageblock_resource_binding(master_member: u32) -> Option<u32> {
    FRAGMENT_IMAGEBLOCK_BINDING_RANGE.binding(master_member)
}

/// AIR address space 1 = device memory (`device`) — a descriptor-backed storage buffer.
pub const ADDRESS_SPACE_DEVICE: u32 = 1;
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
    TessellationEvaluation,
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
    /// A kernel `[[stage_in]]` attribute array. `metal_index` is the synthetic Metal buffer slot,
    /// and `stage_input_location` is its AIR attribute location.
    KernelStageInput,
    /// A `[[texture(n)]]` sampled image. Bound at `TEXTURE_BINDING_BASE + n`.
    Texture,
    /// A runtime-indexed texture descriptor array. See [`ResourceBinding::access`] to select the
    /// sampled- or storage-texture binding band.
    TextureArray,
    /// A write-capable storage image (`texture` with `access::write` or `access::read_write`). Bound
    /// at `STORAGE_TEXTURE_BINDING_BASE + n`.
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
    /// An authored primitive acceleration structure. Metal binds the native object. Its descriptor
    /// is present only when Vulkan intersection lowering consumes the triangle-geometry shadow.
    PrimitiveAccelerationStructure,
    /// An authored Metal visible-function table resolved during dependency linking. It consumes no
    /// Vulkan descriptor after indirect calls are specialized to linked functions.
    VisibleFunctionTable,
    /// An authored Metal intersection-function table resolved during dependency linking.
    IntersectionFunctionTable,
    /// A texture embedded inside an `air.indirect_buffer` argument buffer, surfaced as a standalone
    /// sampled or storage image in the corresponding texture binding band.
    EmbeddedArgBufferTexture,
    /// A device buffer embedded inside an `air.indirect_buffer`. The Vulkan module dereferences the
    /// device address written into the owner field, so this consumes no descriptor of its own.
    EmbeddedArgBufferBuffer,
    /// Synthesized table of Vulkan buffer device addresses indexed by Metal buffer location. Used
    /// by the BDA retry tier for direct device-buffer parameters.
    BufferAddressTable,
    /// A placeholder sampled image the translator binds for `air.get_null_texture_*()`, at the first
    /// binding in the sampled-texture band no Metal texture claims.
    ///
    /// No Metal argument corresponds to it. It exists because a function-constant-gated optional
    /// attachment still has to yield a texture handle, and it is reported only when the shader
    /// actually reads through that handle -- an unread one is retracted during translation. A
    /// consumer must bind an image of [`ResourceBinding::texture_shape`] here; what it contains is
    /// not observed, since Metal's null texture reads as zero.
    SynthesizedNullTexture,
    /// A placeholder sampler the translator binds for `air.get_read_sampler()`, at the first binding
    /// in the sampler band no Metal sampler claims.
    ///
    /// No Metal argument corresponds to it. It exists because AIR threads a sampler pointer into
    /// the sampler-less `texture.read(coord)`, and it is reported only when something consumes that
    /// value -- otherwise it is retracted during translation. A consumer must bind a sampler here.
    SynthesizedReadSampler,
}

/// The descriptor location the interface pass decorates a resource with. Absent for resources that
/// consume no descriptor (threadgroup buffers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DescriptorLocation {
    pub set: u32,
    pub binding: u32,
    /// Number of descriptors occupied at this binding. Top-level texture-handle arrays use the
    /// reflected ABI capacity, embedded fixed arrays use their exact length, and scalar resources
    /// use `1`.
    pub count: u32,
}

/// Per-binding access classification. Populated at translate time from Metal's declared access and
/// tightened by specialized-entry parameter attributes (`readnone`, `readonly`, `writeonly`).
/// Ambiguous device buffers retain the conservative declared result or `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResourceAccess {
    /// The specialized entry does not dereference this buffer. Its descriptor may still be bound,
    /// but no buffer contents need to be staged for this shader invocation.
    Unused,
    /// A buffer read but never written by the shader.
    ReadOnly,
    /// A buffer written but never read by the shader.
    WriteOnly,
    /// A buffer both read and written by the shader, or conservatively declared for both.
    ReadWrite,
    /// A sampled texture (`OpTypeImage Sampled=1`), read through a sampler.
    Sampled,
    /// A storage image (`OpTypeImage Sampled=2`), read/written directly.
    Storage,
}

/// Conservative byte-extent classification for a buffer binding.
///
/// Consumers may narrow a staged buffer window only for [`BufferExtent::Object`]. `Unbounded` and
/// `Unknown` both require retaining the complete caller-provided window. Every classification is an
/// over-approximation: an uncertain pointer must never be reported as a bounded object, because an
/// understated extent silently corrupts shader reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BufferExtent {
    /// AIR declares a reference-like object whose reachable extent is exactly `bytes`.
    Object { bytes: u32 },
    /// AIR declares a pointer/array element size but carries no array length.
    Unbounded,
    /// The metadata does not distinguish a bounded object from an unbounded pointer.
    Unknown,
}

/// Conservative byte footprint of every memory access rooted at one reflected buffer descriptor.
///
/// A consumer may narrow staging to the union of [`Self::static_ranges`] and the ranges obtained by
/// bounding [`Self::strided_accesses`] only when [`Self::has_unbounded_access`] is false. A true
/// unbounded flag means at least one reachable dereference could not be expressed by this schema;
/// the complete caller-provided buffer window must then remain available.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BufferFootprint {
    /// Coalesced half-open byte ranges whose addresses are independent of draw/dispatch indices.
    pub static_ranges: Vec<BufferByteRange>,
    /// Accesses whose byte address is an affine expression of stable Vulkan invocation builtins.
    pub strided_accesses: Vec<BufferStridedAccess>,
    /// Whether any access rooted at this binding could not be represented conservatively above.
    pub has_unbounded_access: bool,
}

/// One half-open byte interval `[offset, offset + size)` in a buffer binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BufferByteRange {
    pub offset: u64,
    pub size: u64,
}

/// One buffer access at `base_offset + sum(index * stride)`, spanning `access_size` bytes.
///
/// Terms are sorted by [`BufferIndexSource`] and duplicate sources are combined. The expression is
/// deliberately limited to stable invocation inputs whose bounds a draw/dispatch consumer knows;
/// data-dependent indices are reported through [`BufferFootprint::has_unbounded_access`] instead.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BufferStridedAccess {
    pub base_offset: u64,
    pub access_size: u64,
    pub terms: Vec<BufferStrideTerm>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BufferStrideTerm {
    /// Draw/dispatch index used by this address term.
    pub source: BufferIndexSource,
    /// Bytes added to the address for each increment of `source`.
    pub stride: u64,
}

/// Stable draw/dispatch index that participates in a reflected affine buffer address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BufferIndexSource {
    VertexIndex,
    InstanceIndex,
    GlobalInvocationIdX,
    GlobalInvocationIdY,
    GlobalInvocationIdZ,
    LocalInvocationIdX,
    LocalInvocationIdY,
    LocalInvocationIdZ,
    WorkgroupIdX,
    WorkgroupIdY,
    WorkgroupIdZ,
    LocalInvocationIndex,
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

/// Pipeline-provided state for one dynamically bound Metal sampler.
///
/// AIR carries the sampler's Metal index but not the state selected when the pipeline is created.
/// Supplying this state lets translation replace operations that Vulkan forbids with an
/// unnormalized-coordinate sampler by equivalent shader-side image fetches. Unlike
/// [`StaticSamplerState`], this has no AIR-encoded raw words because it comes from the caller.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RuntimeSamplerState {
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
}

impl RuntimeSamplerState {
    /// Reject pipeline-provided state the sampler lowering cannot reproduce, before translation
    /// mutates anything. The emulation constraints live on [`StaticSamplerState`] -- the type the
    /// lowering actually consumes -- so caller-supplied and AIR-encoded state answer to one rule.
    pub(crate) fn validate(self) -> Result<(), String> {
        if self.max_anisotropy == 0 {
            return Err("runtime sampler max_anisotropy must be at least 1".into());
        }
        if !self.lod_min_clamp.is_finite()
            || !self.lod_max_clamp.is_finite()
            || !self.lod_bias.is_finite()
        {
            return Err("runtime sampler LOD bounds and bias must be finite".into());
        }
        if self.lod_min_clamp > self.lod_max_clamp {
            return Err(format!(
                "runtime sampler minimum LOD {} exceeds maximum LOD {}",
                self.lod_min_clamp, self.lod_max_clamp
            ));
        }
        self.lowering_state().validate_lowering()
    }

    /// Project pipeline-provided state into the sampler-lowering representation shared by the
    /// translator and consumers that create the matching Vulkan sampler. `raw_words` is zeroed
    /// because runtime state has no AIR constexpr encoding.
    pub fn lowering_state(self) -> StaticSamplerState {
        StaticSamplerState {
            min_filter: self.min_filter,
            mag_filter: self.mag_filter,
            mip_filter: self.mip_filter,
            address_mode_s: self.address_mode_s,
            address_mode_t: self.address_mode_t,
            address_mode_r: self.address_mode_r,
            coordinates: self.coordinates,
            compare_function: self.compare_function,
            max_anisotropy: self.max_anisotropy,
            lod_min_clamp: self.lod_min_clamp,
            lod_max_clamp: self.lod_max_clamp,
            border_color: self.border_color,
            reduction: self.reduction,
            lod_bias: self.lod_bias,
            raw_words: [0; 2],
        }
    }
}

/// One runtime sampler state applied to every AIR sampler parameter at the same Metal index.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RuntimeSamplerSpecialization {
    pub metal_index: u32,
    pub state: RuntimeSamplerState,
}

/// Concrete storage-image format supplied by the pipeline for a dynamically bound Metal texture.
///
/// Most formats have an exact SPIR-V `ImageFormat`. `Bgra8Unorm` deliberately does not: SPIR-V has
/// no BGRA storage-image format token, so it can be used only through `ImageFormat::Unknown` when the
/// host exposes the operation-specific formatless storage-image features.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RuntimeStorageImageFormat {
    R8Unorm,
    Rgba8Unorm,
    Bgra8Unorm,
    R16Float,
    Rg16Float,
    Rg32Float,
    Rgba16Float,
    R32Float,
    Rgba32Float,
    R16Uint,
    R32Uint,
    Rgba8Uint,
    Rgba16Uint,
    Rgba32Uint,
    R32Sint,
    Rgba8Sint,
    Rgba32Sint,
}

impl RuntimeStorageImageFormat {
    pub(crate) fn component(self) -> crate::meta::TextureComponent {
        use crate::meta::TextureComponent;
        match self {
            Self::R8Unorm
            | Self::Rgba8Unorm
            | Self::Bgra8Unorm
            | Self::R16Float
            | Self::Rg16Float
            | Self::Rg32Float
            | Self::Rgba16Float
            | Self::R32Float
            | Self::Rgba32Float => TextureComponent::Float,
            Self::R16Uint
            | Self::R32Uint
            | Self::Rgba8Uint
            | Self::Rgba16Uint
            | Self::Rgba32Uint => TextureComponent::Uint,
            Self::R32Sint | Self::Rgba8Sint | Self::Rgba32Sint => TextureComponent::Sint,
        }
    }

    pub(crate) fn explicit_format(self) -> Option<crate::meta::TextureFormat> {
        use crate::meta::TextureFormat;
        match self {
            Self::R8Unorm => Some(TextureFormat::R8),
            Self::Rgba8Unorm => Some(TextureFormat::Rgba8),
            Self::R16Float => Some(TextureFormat::R16f),
            Self::Rg16Float => Some(TextureFormat::Rg16f),
            Self::Rg32Float => Some(TextureFormat::Rg32f),
            Self::Rgba16Float => Some(TextureFormat::Rgba16f),
            Self::R32Float => Some(TextureFormat::R32f),
            Self::Rgba32Float => Some(TextureFormat::Rgba32f),
            Self::R16Uint => Some(TextureFormat::R16ui),
            Self::R32Uint => Some(TextureFormat::R32ui),
            Self::Rgba8Uint => Some(TextureFormat::Rgba8ui),
            Self::Rgba16Uint => Some(TextureFormat::Rgba16ui),
            Self::Rgba32Uint => Some(TextureFormat::Rgba32ui),
            Self::R32Sint => Some(TextureFormat::R32i),
            Self::Rgba8Sint => Some(TextureFormat::Rgba8i),
            Self::Rgba32Sint => Some(TextureFormat::Rgba32i),
            Self::Bgra8Unorm => None,
        }
    }

    pub(crate) fn supports_atomics(self) -> bool {
        matches!(self, Self::R32Uint | Self::R32Sint)
    }
}

/// Host features relevant to one runtime storage-image specialization.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RuntimeStorageImageCapabilities {
    /// The runtime format supports Vulkan storage-image usage.
    pub storage_image: bool,
    /// The runtime format supports storage-image atomics.
    pub storage_image_atomic: bool,
    /// `shaderStorageImageReadWithoutFormat` is enabled on the target device.
    pub read_without_format: bool,
    /// `shaderStorageImageWriteWithoutFormat` is enabled on the target device.
    pub write_without_format: bool,
}

/// Pipeline-provided state for one dynamically bound Metal storage texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RuntimeStorageImageState {
    pub format: RuntimeStorageImageFormat,
    pub capabilities: RuntimeStorageImageCapabilities,
}

impl RuntimeStorageImageState {
    pub(crate) fn validate(self) -> Result<(), String> {
        if !self.capabilities.storage_image {
            return Err(format!(
                "runtime format {:?} lacks storage-image format support",
                self.format
            ));
        }
        Ok(())
    }
}

/// One runtime storage-image specialization reflected alongside the executable SPIR-V.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RuntimeStorageImageSpecialization {
    /// Metal texture index for a top-level binding, or the translator-assigned synthetic index
    /// reported by the matching `EmbeddedArgBufferTexture` binding.
    pub metal_index: u32,
    pub state: RuntimeStorageImageState,
    /// Exact explicit SPIR-V format, or `None` when the emitted `OpTypeImage` uses `Unknown`.
    pub spirv_format: Option<crate::meta::TextureFormat>,
}

impl StaticSamplerState {
    /// Decode the two `i64` words AIR stores for a `constexpr sampler`.
    ///
    /// The complete bit map, so that what is read and what is not are both stated rather than
    /// implied by the shifts below. Counts are over the 1084 static samplers in a 2880-source
    /// corpus.
    ///
    /// | `words[0]` | Field |
    /// |---|---|
    /// | 0-2, 3-5, 6-8 | `address_mode_s` / `_t` / `_r` |
    /// | 9-10, 11-12, 13-14 | `mag_filter` / `min_filter` / `mip_filter` |
    /// | 15 | `coordinates` |
    /// | 16-19 | `compare_function` |
    /// | 20-23 | `max_anisotropy - 1` |
    /// | 24-31 | **not read** (always zero) |
    /// | 32-39 | high byte of the `lod_min_clamp` half; the low byte is assumed zero (always zero) |
    /// | 40-55 | `lod_max_clamp` half |
    /// | 56-57, 58-59 | `border_color` / `reduction` |
    /// | 60-62 | **not read** (always zero) |
    /// | 63 | **not read**, and set on 151 of them, with no correlate among the decoded fields |
    ///
    /// | `words[1]` | Field |
    /// |---|---|
    /// | 0-15 | `lod_bias` half |
    /// | 16-63 | **not read** (always zero) |
    ///
    /// Bit 63 is a gap, not a decision: no evidence says what it selects, and every translation
    /// carrying it is currently accepted. An unrecognized *enum code* in a field this does read is
    /// a different matter and returns `Err`, since the alternative would be inventing a filter or
    /// address mode. `raw_words` is retained so a consumer can act on a bit this does not decode.
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

    /// Reject state the sampler lowering cannot reproduce exactly.
    ///
    /// Vulkan has no unnormalized-coordinate sampler with Metal's semantics, so a
    /// `coord::pixel` sampler is emulated with shader-side image fetches at level zero. Anything
    /// the fetch cannot express has to fail the translation rather than be silently dropped: the
    /// emitted module would sample the right texture with the wrong filter, the wrong anisotropy,
    /// or the wrong border, and validate either way.
    ///
    /// Both sources of sampler state answer to this: pipeline-provided state through
    /// [`RuntimeSamplerState::validate`], and AIR's constexpr encoding where the static sampler is
    /// lowered. Two copies of these rules drift, and they had -- the caller's copy refused a LOD
    /// maximum the AIR path accepted 531 times over the corpus.
    pub(crate) fn validate_lowering(self) -> Result<(), String> {
        if self.coordinates != SamplerCoordinates::Pixel {
            return Ok(());
        }
        if self.min_filter != self.mag_filter {
            return Err(
                "pixel-coordinate samplers with mixed min/mag filters are unsupported because AIR does not expose the pipeline derivative state needed to select the filter"
                    .into(),
            );
        }
        if self.mip_filter == SamplerMipFilter::Linear {
            return Err(
                "pixel-coordinate samplers with linear mip filtering are unsupported".into(),
            );
        }
        if self.max_anisotropy > 1 {
            return Err("pixel-coordinate sampler anisotropy cannot be emulated exactly".into());
        }
        if self.lod_bias != 0.0 {
            return Err("pixel-coordinate sampler LOD bias cannot be emulated exactly".into());
        }
        // Only the minimum matters. The fetch reads level zero, so a minimum above zero excludes
        // the level the emulation reads; a maximum cannot, since it is never below the minimum.
        // Metal's default maximum is the half-precision limit rather than zero -- 531 of the 535
        // pixel-coordinate static samplers in the corpus carry exactly that -- so demanding zero
        // rejected the ordinary case.
        if self.lod_min_clamp != 0.0 {
            return Err(format!(
                "pixel-coordinate sampler minimum LOD clamp {} excludes the level zero the emulation reads",
                self.lod_min_clamp
            ));
        }
        if self.reduction != SamplerReduction::WeightedAverage {
            return Err("pixel-coordinate sampler min/max reduction is unsupported".into());
        }
        if self.border_color != SamplerBorderColor::TransparentBlack
            && [
                self.address_mode_s,
                self.address_mode_t,
                self.address_mode_r,
            ]
            .contains(&SamplerAddressMode::ClampToBorder)
        {
            return Err(
                "pixel-coordinate samplers support only transparent-black border emulation".into(),
            );
        }
        Ok(())
    }

    pub(crate) fn uses_pixel_nearest(self) -> bool {
        self.uses_pixel_coordinates()
            && self.min_filter == SamplerFilter::Nearest
            && self.mag_filter == SamplerFilter::Nearest
    }

    pub(crate) fn uses_linear_filter(self) -> bool {
        self.min_filter == SamplerFilter::Linear && self.mag_filter == SamplerFilter::Linear
    }

    pub(crate) fn uses_bicubic_filter(self) -> bool {
        self.min_filter == SamplerFilter::Bicubic && self.mag_filter == SamplerFilter::Bicubic
    }

    pub(crate) fn uses_pixel_coordinates(self) -> bool {
        self.coordinates == SamplerCoordinates::Pixel
    }

    pub(crate) fn spatial_clamps_to_zero(self, dimension: usize) -> bool {
        match self.spatial_address_mode(dimension) {
            Some(SamplerAddressMode::ClampToZero) => true,
            Some(SamplerAddressMode::ClampToBorder) => {
                self.border_color == SamplerBorderColor::TransparentBlack
            }
            _ => false,
        }
    }

    pub(crate) fn spatial_address_mode(self, dimension: usize) -> Option<SamplerAddressMode> {
        [
            self.address_mode_s,
            self.address_mode_t,
            self.address_mode_r,
        ]
        .get(dimension)
        .copied()
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
    /// Kernel parameter index of the owning `air.indirect_buffer`.
    pub buffer_param_index: u32,
    /// Kernel parameter index of the owning `air.indirect_buffer` argument.
    pub buffer_index: u32,
    /// Byte offset of the texture handle within the argument-buffer struct.
    pub field_offset: u32,
    /// Zero-based member ordinal used by the LLVM GEP that loads this handle.
    pub field_ordinal: u32,
    /// Metal argument-encoder index (`[[id(n)]]`) of the texture field.
    pub argument_index: u32,
    /// Nested Metal `[[buffer(n)]]` location when this field is an `air.buffer` resource handle.
    /// Consumers write that buffer's native handle/address into the owner argument buffer.
    pub resource_buffer_index: Option<u32>,
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
    /// AIR attribute location for [`ResourceKind::KernelStageInput`]; `None` otherwise.
    pub stage_input_location: Option<u32>,
    /// For a buffer: the raw AIR address space (1 = device, 2 = constant, 3 = threadgroup). `None`
    /// for non-buffers or when metadata does not carry it.
    pub address_space: Option<u32>,
    /// For a buffer: the declared AIR argument byte size, when the metadata carries one.
    pub declared_size: Option<u32>,
    /// For a buffer: whether AIR bounds the binding to one object or leaves its array extent open.
    /// `None` for non-buffer resources.
    pub extent: Option<BufferExtent>,
    /// Final-module byte footprint for descriptor-backed `Buffer`, `KernelStageInput`, and
    /// `AccelerationStructureShadow` resources. `None` for other resources, threadgroup memory, and
    /// metadata-only reflection that did not translate a SPIR-V module.
    pub footprint: Option<BufferFootprint>,
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
    /// Declared or structurally tightened access classification; `None` when AIR does not provide
    /// enough information for a conservative classification.
    pub access: Option<ResourceAccess>,
    /// Decoded AIR state for [`ResourceKind::StaticSampler`]; `None` for every other kind.
    pub static_sampler: Option<StaticSamplerState>,
}

impl ResourceBinding {
    fn descriptor_at(binding: Option<u32>) -> Option<DescriptorLocation> {
        Some(DescriptorLocation {
            set: RESOURCE_DESCRIPTOR_SET,
            binding: binding?,
            count: 1,
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

/// One descriptor-backed implicit-imageblock render-target/data-rate plane. The storage image is
/// 2D-arrayed and the AIR color/sample index selects its array layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImplicitImageblockAttachment {
    pub attachment: u32,
    pub data_rate: u32,
    pub max_index: Option<u32>,
    pub binding: u32,
    pub format: crate::meta::TextureFormat,
    pub access: ResourceAccess,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FragmentImageblockMember {
    pub offset: u32,
    pub size: u32,
    pub type_name: String,
    pub semantic: String,
    pub raster_order_group: u32,
    /// Storage-image binding when at least one entry projection accesses this member.
    pub binding: Option<u32>,
    pub access: ResourceAccess,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FragmentImageblock {
    pub sample_size: u32,
    pub members: Vec<FragmentImageblockMember>,
    pub inputs: Vec<crate::meta::FragmentImageblockProjection>,
    pub outputs: Vec<crate::meta::FragmentImageblockProjection>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TessellationInterface {
    pub domain: crate::meta::PatchDomain,
    pub control_point_count: u32,
    pub control_point_locations: Vec<u32>,
    pub patch_input_locations: Vec<u32>,
    pub control_point_attributes: Vec<TessellationAttribute>,
    pub patch_attributes: Vec<TessellationAttribute>,
    pub instance_id: Option<TessellationAttribute>,
    pub amplification_id: Option<TessellationAttribute>,
    pub amplification_count: Option<TessellationAttribute>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TessellationAttribute {
    pub location: u32,
    pub type_name: Option<String>,
}

fn tessellation_system_attribute(
    meta: &VertMeta,
    expected_role: &VertRole,
) -> Option<TessellationAttribute> {
    let (parameter, role) = meta
        .roles
        .iter()
        .find(|(_, role)| *role == *expected_role)?;
    Some(TessellationAttribute {
        location: meta.tessellation_system_input_location(role)?,
        type_name: meta.parameter_type_names.get(parameter).cloned(),
    })
}

/// Describe the implicit imageblock render-target planes the interface pass materializes for a
/// module's `air.load/store.implicit_imageblock.*` calls.
///
/// Those calls are detected from the module body, not from the stage, and the interface pass
/// materializes a descriptor wherever it lowers one. Every stage therefore reports them the same
/// way: a stage that reported nothing here would leave a consumer building a descriptor-set layout
/// that does not cover a binding the module it was handed declares.
fn implicit_imageblock_planes(
    attachments: &[crate::meta::ImplicitImageblockAttachment],
) -> Vec<ImplicitImageblockAttachment> {
    attachments
        .iter()
        .map(|attachment| ImplicitImageblockAttachment {
            attachment: attachment.attachment,
            data_rate: attachment.data_rate,
            max_index: attachment.max_index,
            binding: imageblock_resource_binding(attachment.attachment, attachment.data_rate)
                .unwrap_or(IMAGEBLOCK_BINDING_RANGE.end),
            format: attachment.format,
            access: match (attachment.reads, attachment.writes) {
                (true, true) => ResourceAccess::ReadWrite,
                (true, false) => ResourceAccess::ReadOnly,
                (false, true) => ResourceAccess::WriteOnly,
                (false, false) => ResourceAccess::Unused,
            },
        })
        .collect()
}

/// The `Binding` decoration on `variable`, if the module gives it one.
/// The image an emitted descriptor variable is declared with: what a consumer must match when it
/// creates the view it binds there.
#[derive(Clone, Copy, PartialEq, Eq)]
struct EmittedImage {
    dimension: TextureDimension,
    arrayed: bool,
    multisampled: bool,
    component: TextureComponent,
    writable: bool,
    storage_format: Option<crate::meta::TextureFormat>,
}

impl EmittedImage {
    fn of(
        types: &std::collections::HashMap<spirv::Word, &crate::spirv_module::Instruction>,
        image: spirv::Word,
    ) -> Option<Self> {
        let instruction = types.get(&image)?;
        if instruction.class.opcode != spirv::Op::TypeImage {
            return None;
        }
        let literal = |index: usize| match instruction.operands.get(index) {
            Some(crate::spirv_module::Operand::LiteralBit32(value)) => Some(*value),
            _ => None,
        };
        let dimension = match instruction.operands.get(1) {
            Some(crate::spirv_module::Operand::Dim(dim)) => TextureDimension::from_spirv_dim(*dim),
            _ => return None,
        };
        // The sampled type is the image's component class: a float, or an integer whose
        // signedness operand separates `Sint` from `Uint`.
        let sampled_type = match instruction.operands.first() {
            Some(crate::spirv_module::Operand::IdRef(scalar)) => types.get(scalar)?,
            _ => return None,
        };
        let component = match (sampled_type.class.opcode, sampled_type.operands.get(1)) {
            (spirv::Op::TypeFloat, _) => TextureComponent::Float,
            (spirv::Op::TypeInt, Some(crate::spirv_module::Operand::LiteralBit32(1))) => {
                TextureComponent::Sint
            }
            (spirv::Op::TypeInt, _) => TextureComponent::Uint,
            _ => return None,
        };
        let storage_format = match instruction.operands.get(6) {
            Some(crate::spirv_module::Operand::ImageFormat(format)) => {
                crate::meta::TextureFormat::from_spirv_format(*format)
            }
            _ => None,
        };
        Some(Self {
            dimension,
            arrayed: literal(3)? != 0,
            multisampled: literal(4)? != 0,
            component,
            writable: literal(5)? == 2,
            storage_format,
        })
    }
}

/// The type a pointer type points at, looking through the array wrappers a descriptor array adds.
fn pointee_of(
    types: &std::collections::HashMap<spirv::Word, &crate::spirv_module::Instruction>,
    pointer: spirv::Word,
) -> Option<spirv::Word> {
    let instruction = types.get(&pointer)?;
    if instruction.class.opcode != spirv::Op::TypePointer {
        return None;
    }
    let mut pointee = match instruction.operands.get(1) {
        Some(crate::spirv_module::Operand::IdRef(pointee)) => *pointee,
        _ => return None,
    };
    while let Some(element) = types.get(&pointee).and_then(|instruction| {
        matches!(
            instruction.class.opcode,
            spirv::Op::TypeArray | spirv::Op::TypeRuntimeArray
        )
        .then(|| match instruction.operands.first() {
            Some(crate::spirv_module::Operand::IdRef(element)) => Some(*element),
            _ => None,
        })
        .flatten()
    }) {
        pointee = element;
    }
    Some(pointee)
}

impl EmittedImage {
    /// The reflected shape of this image on its own, for a descriptor with no AIR type name behind
    /// it. The two array fields describe a descriptor array, which a synthesized placeholder never
    /// is.
    fn texture_shape(self) -> TextureShape {
        TextureShape {
            dimension: self.dimension,
            arrayed: self.arrayed,
            multisampled: self.multisampled,
            component: self.component,
            writable: self.writable,
            array_ref: false,
            array_length: None,
            storage_format: self.storage_format,
        }
    }
}

fn descriptor_binding_of(module: &Module, variable: spirv::Word) -> Option<u32> {
    module.annotations.iter().find_map(|annotation| {
        match (
            annotation.class.opcode,
            annotation.operands.first(),
            annotation.operands.get(1),
            annotation.operands.get(2),
        ) {
            (
                spirv::Op::Decorate,
                Some(crate::spirv_module::Operand::IdRef(target)),
                Some(crate::spirv_module::Operand::Decoration(spirv::Decoration::Binding)),
                Some(crate::spirv_module::Operand::LiteralBit32(binding)),
            ) if *target == variable => Some(*binding),
            _ => None,
        }
    })
}

/// Consumer-shaped reflection of one translated shader.
///
/// Interface declarations are built from parser-shaped AIR metadata and the translator's shared
/// descriptor ABI. Successful reflected translation additionally analyzes the final constructed
/// SPIR-V for conservative buffer footprints before releasing its owned carrier. Every reported
/// binding number matches the module returned alongside this value.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShaderReflection {
    /// Schema version, always [`REFLECTION_VERSION`] at build time.
    pub reflection_version: u32,
    /// Effective descriptor-set and binding-range ABI used by the returned SPIR-V module.
    #[cfg_attr(feature = "serde", serde(default))]
    pub descriptor_layout: DescriptorLayout,
    pub stage: ShaderStage,
    /// The ORIGINAL Metal entry-point function name (the emitted SPIR-V `OpEntryPoint` string is
    /// always `"main"`, so the meaningful identity a consumer keys on is this name).
    pub entry_point: Option<String>,
    /// Every bound resource: AIR entry resources first, followed by translator-synthesized resources.
    pub bindings: Vec<ResourceBinding>,
    /// All resource-handle fields declared inside shader argument buffers. Authored resources use
    /// this shared coordinate to obtain the Metal argument index without re-parsing AIR metadata.
    pub argument_buffer_fields: Vec<EmbeddedArgBuffer>,
    /// Vertex input attributes (vertex stage only).
    pub vertex_attributes: Vec<VertexAttribute>,
    /// Varyings: fragment `[[stage_in]]` inputs, or vertex user-varying outputs.
    pub varyings: Vec<Varying>,
    /// Fragment render-target outputs (fragment stage only).
    pub render_targets: Vec<RenderTarget>,
    /// Fragment return-struct members tagged `[[depth]]`.
    pub depth_members: Vec<u32>,
    /// AIR depth-output qualifier used to derive the graphics pipeline's depth comparison.
    pub depth_qualifier: Option<crate::meta::DepthQualifier>,
    /// Fragment return-struct members tagged `[[stencil]]`.
    pub stencil_members: Vec<u32>,
    /// Kernel GLCompute local size (`[x, y, z]`), when the stage is a kernel.
    pub local_size: Option<[u32; 3]>,
    /// Kernel dispatch/grid ABI, when the stage is a kernel. A push-constant grid must be populated
    /// before each dispatch using the reflected byte range.
    pub kernel_dispatch: Option<KernelDispatch>,
    /// Vertex-stage builtin usage (`Some` only for the vertex stage).
    pub vertex_builtins: Option<VertexBuiltins>,
    /// Vulkan tessellation-evaluation interface synthesized from Metal patch metadata.
    pub tessellation: Option<TessellationInterface>,
    /// Kernel `[[imageblock]]` threadgroup tiles (kernel stage only), sorted by parameter index.
    pub imageblock_layouts: Vec<ImageblockLayout>,
    /// Implicit imageblock attachment planes consumed by stable AIR load/store intrinsics.
    pub implicit_imageblock_attachments: Vec<ImplicitImageblockAttachment>,
    /// Custom fragment imageblock master/projection ABI. Every master field maps to the reflected
    /// storage-image binding and is serialized independently from kernel/implicit imageblocks.
    pub fragment_imageblock: Option<FragmentImageblock>,
    /// The source LLVM-IR `target datalayout` string, when the reflected translate started from an
    /// unsanitized module (sanitization strips it). A consumer uses it to lay out struct members
    /// without re-reading the source `.ll`. `None` when translated from already-sanitized IR.
    pub datalayout: Option<String>,
    /// Pipeline-provided sampler states used to specialize dynamically bound Metal samplers.
    /// Entries are sorted by `metal_index`; each state applies to every runtime sampler binding at
    /// that index and describes the executable SPIR-V returned with this reflection.
    #[cfg_attr(feature = "serde", serde(default))]
    pub runtime_sampler_specializations: Vec<RuntimeSamplerSpecialization>,
    /// Pipeline-provided storage-image formats used to specialize dynamically bound Metal
    /// textures. The explicit/formatless choice matches the executable SPIR-V returned alongside
    /// this reflection.
    #[cfg_attr(feature = "serde", serde(default))]
    pub runtime_storage_image_specializations: Vec<RuntimeStorageImageSpecialization>,
    /// Metal `[[function_constant(N)]]` inventory (index/name/type), so a consumer can discover the
    /// module's spec-ids without scanning SPIR-V. Populated by the reflected translate paths; empty
    /// when reflection is built directly from meta (the `from_*` builders do not scan IR).
    pub function_constants: Vec<FunctionConstant>,
}

impl ShaderReflection {
    pub(crate) fn apply_descriptor_layout(
        &mut self,
        layout: DescriptorLayout,
    ) -> Result<(), String> {
        layout.validate().map_err(|error| error.to_string())?;
        for resource in &mut self.bindings {
            let Some(descriptor) = resource.descriptor.as_mut() else {
                continue;
            };
            let storage_texture = matches!(resource.kind, ResourceKind::StorageImage)
                || matches!(
                    resource.kind,
                    ResourceKind::TextureArray | ResourceKind::EmbeddedArgBufferTexture
                ) && resource.access == Some(ResourceAccess::Storage);
            let binding = match resource.kind {
                ResourceKind::Buffer
                | ResourceKind::KernelStageInput
                | ResourceKind::AccelerationStructureShadow
                | ResourceKind::PrimitiveAccelerationStructure => {
                    layout.buffer_binding(resource.metal_index)
                }
                ResourceKind::Texture
                | ResourceKind::TextureArray
                | ResourceKind::StorageImage
                | ResourceKind::EmbeddedArgBufferTexture => {
                    if storage_texture {
                        layout.storage_texture_binding(resource.metal_index)
                    } else {
                        layout.sampled_texture_binding(resource.metal_index)
                    }
                }
                ResourceKind::Sampler => layout.sampler_binding(resource.metal_index),
                ResourceKind::ColorInput => layout.color_input_binding(resource.metal_index),
                ResourceKind::StaticSampler
                | ResourceKind::BufferAddressTable
                | ResourceKind::SynthesizedNullTexture
                | ResourceKind::SynthesizedReadSampler => {
                    return Err(format!(
                        "cannot reconfigure reflection after synthesized {:?} resources were added",
                        resource.kind
                    ));
                }
                ResourceKind::ThreadgroupBuffer
                | ResourceKind::VisibleFunctionTable
                | ResourceKind::IntersectionFunctionTable
                | ResourceKind::EmbeddedArgBufferBuffer => None,
            }
            .ok_or_else(|| {
                format!(
                    "{:?} resource {} exceeds the selected descriptor layout",
                    resource.kind, resource.metal_index
                )
            })?;
            descriptor.set = layout.set;
            descriptor.binding = binding;
        }
        for attachment in &mut self.implicit_imageblock_attachments {
            attachment.binding = layout
                .imageblock_binding(attachment.attachment, attachment.data_rate)
                .ok_or_else(|| {
                    format!(
                        "implicit imageblock attachment {} rate {} exceeds the selected descriptor layout",
                        attachment.attachment, attachment.data_rate
                    )
                })?;
        }
        if let Some(imageblock) = &mut self.fragment_imageblock {
            for (index, member) in imageblock.members.iter_mut().enumerate() {
                if member.binding.is_some() {
                    member.binding = Some(
                        layout
                            .fragment_imageblock_binding(index as u32)
                            .ok_or_else(|| {
                                format!(
                                    "fragment imageblock member {index} exceeds the selected descriptor layout"
                                )
                            })?,
                    );
                }
            }
        }
        self.descriptor_layout = layout;
        Ok(())
    }

    /// Validate the public descriptor ABI as a closed, collision-free mapping. Translation calls
    /// this before returning reflection; direct metadata-only users may call it after constructing
    /// a `ShaderReflection` with the `from_*` helpers.
    pub fn validate_descriptor_abi(&self) -> Result<(), String> {
        self.descriptor_layout
            .validate()
            .map_err(|error| error.to_string())?;
        match (self.stage, self.kernel_dispatch) {
            (ShaderStage::Kernel, Some(dispatch)) => dispatch.validate()?,
            (ShaderStage::Kernel, None) => {
                return Err("kernel reflection is missing its dispatch-grid contract".to_string())
            }
            (_, Some(_)) => {
                return Err(
                    "non-kernel reflection carries a kernel dispatch-grid contract".to_string(),
                )
            }
            (_, None) => {}
        }
        // AIR states an argument's size (`air.arg_type_size`) and the member layout inside it
        // (`air.struct_type_info`) as two independent facts, reconstructed here independently. A
        // layout that reaches past the declared size is describing bytes the argument does not
        // have: some member's storage was mistaken, and every member after it sits at an offset
        // the shader never reads. A consumer packs its upload at these offsets, so the
        // disagreement is silent data corruption rather than anything a driver reports, and the
        // emitted SPIR-V does not settle it -- a buffer whose reconstruction is that far off is
        // represented as raw bytes and declares no struct type to compare against. Reaching short
        // is normal, since the declared size is a `sizeof` and carries tail padding.
        for resource in &self.bindings {
            let (Some(layout), Some(declared)) = (&resource.type_layout, resource.declared_size)
            else {
                continue;
            };
            let Some(extent) = crate::layout::air_metadata_extent(layout) else {
                continue;
            };
            if extent > u64::from(declared) {
                return Err(format!(
                    "reflection reports a {extent}-byte member layout for the {declared}-byte argument of {:?}({})",
                    resource.kind, resource.metal_index
                ));
            }
        }
        // A consumer walks `bindings` and acts once per entry: it allocates a descriptor, writes
        // it, and sizes its per-resource budgets from the count. Two entries equal in every field
        // therefore describe one resource twice and ask for that work twice, while carrying no
        // information the first entry did not. Genuinely distinct resources always differ
        // somewhere — the Metal index, the entry-parameter index, or, for an argument-buffer
        // resident, the field offset inside its owning buffer — so equality is the test, not the
        // `(set, binding)` pair, which raw-word alias buffers legitimately share.
        for (index, resource) in self.bindings.iter().enumerate() {
            if let Some(first) = self.bindings[..index]
                .iter()
                .position(|other| other == resource)
            {
                return Err(format!(
                    "reflection reports {:?}({}) twice, identically, at bindings[{first}] and bindings[{index}]",
                    resource.kind, resource.metal_index
                ));
            }
        }

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum DescriptorClass {
            StorageBuffer,
            SampledImage,
            UniformTexelBuffer,
            StorageImage,
            StorageTexelBuffer,
            Sampler,
            InputAttachment,
        }

        let mut occupied =
            std::collections::BTreeMap::<(u32, u32), (String, DescriptorClass, u32)>::new();
        let mut record = |location: DescriptorLocation,
                          owner: String,
                          class: DescriptorClass|
         -> Result<(), String> {
            if location.set != self.descriptor_layout.set {
                return Err(format!(
                    "{owner} uses descriptor set {}, expected {}",
                    location.set, self.descriptor_layout.set
                ));
            }
            if location.count == 0 {
                return Err(format!("{owner} has zero descriptor count"));
            }
            if let Some((previous, previous_class, previous_count)) = occupied.insert(
                (location.set, location.binding),
                (owner.clone(), class, location.count),
            ) {
                if previous_class != class {
                    return Err(format!(
                        "descriptor set {} binding {} is shared incompatibly by {previous} ({previous_class:?}, count {previous_count}) and {owner} ({class:?}, count {})",
                        location.set, location.binding, location.count
                    ));
                }
            }
            Ok(())
        };

        for resource in &self.bindings {
            let storage_texture = matches!(resource.kind, ResourceKind::StorageImage)
                || matches!(
                    resource.kind,
                    ResourceKind::TextureArray | ResourceKind::EmbeddedArgBufferTexture
                ) && resource.access == Some(ResourceAccess::Storage);
            let texel_buffer = resource
                .texture_shape
                .is_some_and(|shape| shape.dimension == TextureDimension::Buffer);
            let class = match resource.kind {
                ResourceKind::Buffer
                | ResourceKind::KernelStageInput
                | ResourceKind::AccelerationStructureShadow
                | ResourceKind::PrimitiveAccelerationStructure
                | ResourceKind::BufferAddressTable => DescriptorClass::StorageBuffer,
                ResourceKind::Texture
                | ResourceKind::TextureArray
                | ResourceKind::StorageImage
                | ResourceKind::EmbeddedArgBufferTexture => {
                    if storage_texture && texel_buffer {
                        DescriptorClass::StorageTexelBuffer
                    } else if storage_texture {
                        DescriptorClass::StorageImage
                    } else if texel_buffer {
                        DescriptorClass::UniformTexelBuffer
                    } else {
                        DescriptorClass::SampledImage
                    }
                }
                ResourceKind::Sampler
                | ResourceKind::StaticSampler
                | ResourceKind::SynthesizedReadSampler => DescriptorClass::Sampler,
                ResourceKind::SynthesizedNullTexture => DescriptorClass::SampledImage,
                ResourceKind::ColorInput => DescriptorClass::InputAttachment,
                ResourceKind::ThreadgroupBuffer
                | ResourceKind::VisibleFunctionTable
                | ResourceKind::IntersectionFunctionTable
                | ResourceKind::EmbeddedArgBufferBuffer => DescriptorClass::StorageBuffer,
            };
            let expected = match resource.kind {
                ResourceKind::Buffer
                | ResourceKind::KernelStageInput
                | ResourceKind::AccelerationStructureShadow => {
                    Some(self.descriptor_layout.buffer_binding(resource.metal_index))
                }
                ResourceKind::PrimitiveAccelerationStructure if resource.descriptor.is_some() => {
                    Some(self.descriptor_layout.buffer_binding(resource.metal_index))
                }
                ResourceKind::Texture
                | ResourceKind::TextureArray
                | ResourceKind::StorageImage
                | ResourceKind::EmbeddedArgBufferTexture => Some(if storage_texture {
                    self.descriptor_layout
                        .storage_texture_binding(resource.metal_index)
                } else {
                    self.descriptor_layout
                        .sampled_texture_binding(resource.metal_index)
                }),
                ResourceKind::Sampler => {
                    Some(self.descriptor_layout.sampler_binding(resource.metal_index))
                }
                ResourceKind::StaticSampler => Some(
                    self.descriptor_layout
                        .samplers
                        .binding(resource.metal_index),
                ),
                ResourceKind::ColorInput => Some(
                    self.descriptor_layout
                        .color_input_binding(resource.metal_index),
                ),
                // The two synthesized placeholders take the first binding in their band that no
                // Metal argument claims, so there is no index to recompute the expectation from.
                // The band check below is what constrains them.
                ResourceKind::BufferAddressTable
                | ResourceKind::SynthesizedNullTexture
                | ResourceKind::SynthesizedReadSampler => None,
                ResourceKind::ThreadgroupBuffer
                | ResourceKind::PrimitiveAccelerationStructure
                | ResourceKind::VisibleFunctionTable
                | ResourceKind::IntersectionFunctionTable
                | ResourceKind::EmbeddedArgBufferBuffer => {
                    if resource.descriptor.is_some() {
                        return Err(format!(
                            "{:?} resource {} unexpectedly consumes a descriptor",
                            resource.kind, resource.metal_index
                        ));
                    }
                    continue;
                }
            };
            let owner = format!("{:?}({})", resource.kind, resource.metal_index);
            if let Some(band) = match resource.kind {
                ResourceKind::SynthesizedNullTexture => {
                    Some(self.descriptor_layout.sampled_textures)
                }
                ResourceKind::SynthesizedReadSampler => Some(self.descriptor_layout.samplers),
                _ => None,
            } {
                let location = resource
                    .descriptor
                    .ok_or_else(|| format!("{owner} is missing its descriptor"))?;
                if !band.contains(location.binding) {
                    return Err(format!(
                        "{owner} uses binding {}, outside its descriptor band [{},{})",
                        location.binding, band.start, band.end
                    ));
                }
                record(location, owner, class)?;
                continue;
            }
            if resource.kind == ResourceKind::BufferAddressTable {
                let location = resource
                    .descriptor
                    .ok_or_else(|| format!("{owner} is missing its descriptor"))?;
                if !self.descriptor_layout.synthetic.contains(location.binding) {
                    return Err(format!(
                        "{owner} uses binding {}, outside synthetic descriptor range [{},{})",
                        location.binding,
                        self.descriptor_layout.synthetic.start,
                        self.descriptor_layout.synthetic.end
                    ));
                }
                record(location, owner, class)?;
                continue;
            }
            let expected = expected
                .flatten()
                .ok_or_else(|| format!("{owner} exceeds its descriptor ABI band"))?;
            let location = resource
                .descriptor
                .ok_or_else(|| format!("{owner} is missing its descriptor"))?;
            if location.binding != expected {
                return Err(format!(
                    "{owner} uses binding {}, expected {expected}",
                    location.binding
                ));
            }
            record(location, owner, class)?;
        }

        for attachment in &self.implicit_imageblock_attachments {
            let owner = format!(
                "implicit imageblock attachment {} rate {}",
                attachment.attachment, attachment.data_rate
            );
            let expected = self
                .descriptor_layout
                .imageblock_binding(attachment.attachment, attachment.data_rate)
                .ok_or_else(|| format!("{owner} exceeds its descriptor ABI band"))?;
            if attachment.binding != expected {
                return Err(format!(
                    "{owner} uses binding {}, expected {expected}",
                    attachment.binding
                ));
            }
            record(
                DescriptorLocation {
                    set: self.descriptor_layout.set,
                    binding: attachment.binding,
                    count: 1,
                },
                owner,
                DescriptorClass::StorageImage,
            )?;
        }
        if let Some(imageblock) = &self.fragment_imageblock {
            for (index, member) in imageblock.members.iter().enumerate() {
                let Some(binding) = member.binding else {
                    continue;
                };
                let owner = format!("fragment imageblock member {index}");
                let expected = self
                    .descriptor_layout
                    .fragment_imageblock_binding(index as u32)
                    .ok_or_else(|| format!("{owner} exceeds its descriptor ABI band"))?;
                if binding != expected {
                    return Err(format!(
                        "{owner} uses binding {binding}, expected {expected}"
                    ));
                }
                record(
                    DescriptorLocation {
                        set: self.descriptor_layout.set,
                        binding,
                        count: 1,
                    },
                    owner,
                    DescriptorClass::StorageImage,
                )?;
            }
        }
        let mut specialized_sampler_indices = std::collections::BTreeSet::new();
        for specialization in &self.runtime_sampler_specializations {
            specialization.state.validate().map_err(|error| {
                format!(
                    "runtime sampler {} specialization is invalid: {error}",
                    specialization.metal_index
                )
            })?;
            if specialization.metal_index >= SAMPLER_ARGUMENT_COUNT {
                return Err(format!(
                    "runtime sampler specialization index {} exceeds Metal sampler range 0..{SAMPLER_ARGUMENT_COUNT}",
                    specialization.metal_index
                ));
            }
            if !specialized_sampler_indices.insert(specialization.metal_index) {
                return Err(format!(
                    "runtime sampler {} is specialized more than once",
                    specialization.metal_index
                ));
            }
            if !self.bindings.iter().any(|binding| {
                binding.kind == ResourceKind::Sampler
                    && binding.metal_index == specialization.metal_index
            }) {
                return Err(format!(
                    "runtime sampler {} specialization has no matching AIR sampler binding",
                    specialization.metal_index
                ));
            }
        }
        Ok(())
    }

    /// Enrich descriptor-backed buffer bindings from the final constructed SPIR-V module.
    pub(crate) fn add_buffer_footprints(&mut self, module: &Module) -> Result<(), String> {
        footprint::attach_buffer_footprints(self, module)
    }

    /// Report the buffer-address tables the finished module actually declares, replacing whatever
    /// `add_buffer_address_table` predicted.
    ///
    /// The table is a translator-owned descriptor: the emitter creates one when the constructed
    /// pointer graph needs it. Reflection predicted that from a text scan of the AIR — an
    /// `inttoptr`, or a `load ptr` through a device-address pointer — and two independent
    /// derivations of one fact disagree. Measured over 2880 corpus sources: 13 modules declared a
    /// table reflection omitted, and 26 reflected one the module never declared. The first
    /// under-binds a consumer's descriptor set; the second demands a buffer nothing reads.
    ///
    /// The module is the fact. Callers that have one observe it here; `reflect_sanitized`, which
    /// never builds a module, keeps the prediction and is documented as an approximation.
    /// Correct each reported `texture_shape` to the image the module actually declares at that
    /// binding.
    ///
    /// The shape is decoded from the AIR type name, and the emitter does not always bind the image
    /// that name implies. A `texturecube` that is only texel-read binds as a `Dim2D` ARRAY image,
    /// because SPIR-V has no cube texel fetch (`resources::discovery`); reflection still said
    /// `Cube`. A consumer that follows it creates a `VK_IMAGE_VIEW_TYPE_CUBE` view for a descriptor
    /// whose shader variable is a 2D array, and Vulkan requires the view type to match the image
    /// type's `Dim`/`Arrayed`, so the dispatch is invalid. Measured over 2880 corpus sources: 2
    /// modules, both cube textures reflected as `Cube` while the module declares 2D arrayed.
    ///
    /// Only a binding whose image variables all declare the SAME image type is corrected. A
    /// function-constant-gated texture argument can put several differently-shaped variables on one
    /// binding -- 67 corpus modules do -- and there the module has no single answer to read; the
    /// consumer selects one by choosing its function constants, and the metadata-derived shape is
    /// left alone.
    ///
    /// `array_ref` and `array_length` describe the descriptor array, not the image, and are left to
    /// the metadata that decides the descriptor count.
    pub(crate) fn reconcile_texture_shapes(&mut self, module: &Module) {
        let types = module
            .types_global_values
            .iter()
            .filter_map(|instruction| Some((instruction.result_id?, instruction)))
            .collect::<std::collections::HashMap<_, _>>();
        let mut declared = std::collections::HashMap::<u32, Option<EmittedImage>>::new();
        for instruction in module
            .types_global_values
            .iter()
            .filter(|instruction| instruction.class.opcode == spirv::Op::Variable)
        {
            let Some(variable) = instruction.result_id else {
                continue;
            };
            let Some(binding) = descriptor_binding_of(module, variable) else {
                continue;
            };
            let Some(image) = instruction
                .result_type
                .and_then(|pointer| pointee_of(&types, pointer))
                .and_then(|pointee| EmittedImage::of(&types, pointee))
            else {
                continue;
            };
            match declared.entry(binding) {
                std::collections::hash_map::Entry::Occupied(mut seen) => {
                    if seen.get() != &Some(image) {
                        seen.insert(None);
                    }
                }
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(Some(image));
                }
            }
        }
        for resource in &mut self.bindings {
            let Some(shape) = resource.texture_shape.as_mut() else {
                continue;
            };
            let Some(binding) = resource.descriptor.map(|location| location.binding) else {
                continue;
            };
            let Some(Some(image)) = declared.get(&binding) else {
                continue;
            };
            shape.dimension = image.dimension;
            shape.arrayed = image.arrayed;
            shape.multisampled = image.multisampled;
            shape.component = image.component;
            shape.writable = image.writable;
            shape.storage_format = image.storage_format;
        }
    }

    /// Report the descriptors the passes synthesized with no Metal argument behind them.
    ///
    /// `air.get_read_sampler()` and `air.get_null_texture_*()` each make the interface pass bind a
    /// real resource so an AIR value has a legal SPIR-V type. Nothing in the AIR metadata describes
    /// those bindings, so reflection built from metadata alone reports a descriptor-set layout that
    /// does not cover the module -- the same shape of gap `reconcile_buffer_address_table` closes
    /// for the address table. `bindings` is what translation kept after retracting every
    /// placeholder nothing consumed, filtered to what the finished module still declares; the
    /// module supplies the resource class, so nothing here is inferred from binding numbers.
    pub(crate) fn report_synthesized_placeholders(&mut self, module: &Module, bindings: &[u32]) {
        self.bindings.retain(|resource| {
            !matches!(
                resource.kind,
                ResourceKind::SynthesizedNullTexture | ResourceKind::SynthesizedReadSampler
            )
        });
        let types = module
            .types_global_values
            .iter()
            .filter_map(|instruction| Some((instruction.result_id?, instruction)))
            .collect::<std::collections::HashMap<_, _>>();
        let mut null_textures = 0;
        let mut read_samplers = 0;
        for &binding in bindings {
            let Some(pointee) = module
                .types_global_values
                .iter()
                .filter(|instruction| instruction.class.opcode == spirv::Op::Variable)
                .filter(|instruction| {
                    instruction
                        .result_id
                        .map(|variable| descriptor_binding_of(module, variable) == Some(binding))
                        == Some(true)
                })
                .find_map(|instruction| pointee_of(&types, instruction.result_type?))
            else {
                continue;
            };
            let (kind, metal_index, texture_shape, access) = match types
                .get(&pointee)
                .map(|instruction| instruction.class.opcode)
            {
                Some(spirv::Op::TypeImage) => {
                    let index = null_textures;
                    null_textures += 1;
                    (
                        ResourceKind::SynthesizedNullTexture,
                        index,
                        EmittedImage::of(&types, pointee).map(EmittedImage::texture_shape),
                        Some(ResourceAccess::Sampled),
                    )
                }
                Some(spirv::Op::TypeSampler) => {
                    let index = read_samplers;
                    read_samplers += 1;
                    (ResourceKind::SynthesizedReadSampler, index, None, None)
                }
                _ => continue,
            };
            self.bindings.push(ResourceBinding {
                kind,
                metal_index,
                descriptor: Some(DescriptorLocation {
                    set: self.descriptor_layout.set,
                    binding,
                    count: 1,
                }),
                param_index: None,
                stage_input_location: None,
                address_space: None,
                declared_size: None,
                extent: None,
                footprint: None,
                type_layout: None,
                type_name: None,
                texture_shape,
                embedded_source: None,
                access,
                static_sampler: None,
            });
        }
    }

    pub(crate) fn reconcile_buffer_address_table(&mut self, module: &Module) {
        let mut declared = module
            .types_global_values
            .iter()
            .filter(|instruction| instruction.class.opcode == spirv::Op::Variable)
            .filter_map(|instruction| instruction.result_id)
            .filter_map(|variable| descriptor_binding_of(module, variable))
            .filter(|binding| self.descriptor_layout.synthetic.contains(*binding))
            .collect::<Vec<_>>();
        declared.sort_unstable();
        declared.dedup();
        self.bindings
            .retain(|resource| resource.kind != ResourceKind::BufferAddressTable);
        for (index, binding) in declared.into_iter().enumerate() {
            self.bindings.push(ResourceBinding {
                kind: ResourceKind::BufferAddressTable,
                metal_index: index as u32,
                descriptor: Some(DescriptorLocation {
                    set: self.descriptor_layout.set,
                    binding,
                    count: 1,
                }),
                param_index: None,
                stage_input_location: None,
                address_space: None,
                declared_size: None,
                extent: None,
                footprint: None,
                type_layout: None,
                type_name: None,
                texture_shape: None,
                embedded_source: None,
                access: Some(ResourceAccess::ReadOnly),
                static_sampler: None,
            });
        }
    }

    pub(crate) fn add_buffer_address_table(&mut self) -> Result<(), String> {
        if self
            .bindings
            .iter()
            .any(|binding| binding.kind == ResourceKind::BufferAddressTable)
        {
            return Ok(());
        }
        let occupied = self
            .bindings
            .iter()
            .filter_map(|resource| resource.descriptor.map(|location| location.binding))
            .chain(
                self.implicit_imageblock_attachments
                    .iter()
                    .map(|attachment| attachment.binding),
            )
            .chain(
                self.fragment_imageblock
                    .iter()
                    .flat_map(|imageblock| &imageblock.members)
                    .filter_map(|member| member.binding),
            )
            .collect::<std::collections::BTreeSet<_>>();
        let binding = (self.descriptor_layout.synthetic.start
            ..self.descriptor_layout.synthetic.end)
            .find(|binding| !occupied.contains(binding))
            .ok_or_else(|| {
                "descriptor binding space exhausted for buffer-address table".to_string()
            })?;
        self.bindings.push(ResourceBinding {
            kind: ResourceKind::BufferAddressTable,
            metal_index: 0,
            descriptor: Some(DescriptorLocation {
                set: self.descriptor_layout.set,
                binding,
                count: 1,
            }),
            param_index: None,
            stage_input_location: None,
            address_space: None,
            declared_size: None,
            extent: None,
            footprint: None,
            type_layout: None,
            type_name: None,
            texture_shape: None,
            embedded_source: None,
            access: Some(ResourceAccess::ReadOnly),
            static_sampler: None,
        });
        Ok(())
    }

    /// Build reflection for a fragment shader from its parsed meta and (optional) entry name.
    pub fn from_fragment(meta: &FragMeta, entry_point: Option<&str>) -> Self {
        let mut bindings = Vec::new();
        for (idx, role) in &meta.roles {
            let idx = *idx;
            let binding = match role {
                FragRole::Buffer(n) => ResourceBinding {
                    kind: ResourceKind::Buffer,
                    metal_index: *n,
                    descriptor: ResourceBinding::descriptor_at(buffer_resource_binding(*n)),
                    param_index: Some(idx),
                    stage_input_location: None,
                    address_space: meta.buffer_address_spaces.get(&idx).copied(),
                    declared_size: meta.buffer_type_sizes.get(&idx).copied(),
                    extent: Some(buffer_extent(
                        meta.buffer_object_sizes.get(&idx).copied(),
                        meta.buffer_type_sizes.get(&idx).copied(),
                        meta.buffer_type_names.get(&idx),
                    )),
                    footprint: None,
                    type_layout: meta.buffer_layouts.get(&idx).cloned(),
                    type_name: meta.buffer_type_names.get(&idx).cloned(),
                    texture_shape: None,
                    embedded_source: None,
                    access: buffer_access(
                        meta.buffer_accesses.get(&idx).copied(),
                        meta.buffer_address_spaces.get(&idx).copied(),
                    ),
                    static_sampler: None,
                },
                FragRole::Texture(n) => texture_binding(*n, Some(idx), &meta.texture_type_names),
                FragRole::Sampler(n) => sampler_binding(*n, Some(idx)),
                FragRole::VisibleFunctionTable(n) => {
                    function_table_binding(ResourceKind::VisibleFunctionTable, *n, idx)
                }
                FragRole::IntersectionFunctionTable(n) => {
                    function_table_binding(ResourceKind::IntersectionFunctionTable, *n, idx)
                }
                FragRole::ColorInput(n) => ResourceBinding {
                    kind: ResourceKind::ColorInput,
                    metal_index: *n,
                    descriptor: ResourceBinding::descriptor_at(color_input_resource_binding(*n)),
                    param_index: Some(idx),
                    stage_input_location: None,
                    address_space: None,
                    declared_size: None,
                    extent: None,
                    footprint: None,
                    type_layout: None,
                    type_name: meta.color_input_type_names.get(n).cloned(),
                    texture_shape: None,
                    embedded_source: None,
                    access: None,
                    static_sampler: None,
                },
                FragRole::Position
                | FragRole::PointCoord
                | FragRole::FrontFacing
                | FragRole::BarycentricCoord { .. }
                | FragRole::PrimitiveId
                | FragRole::SampleId
                | FragRole::SampleMaskIn
                | FragRole::ViewportArrayIndex
                | FragRole::RenderTargetArrayIndex
                | FragRole::Varying(_)
                | FragRole::ImageblockData
                | FragRole::Other => {
                    continue;
                }
            };
            bindings.push(binding);
        }
        append_embedded_resources(
            &mut bindings,
            &meta.embedded_textures,
            &meta.embedded_arguments,
        );
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
            descriptor_layout: DescriptorLayout::default(),
            stage: ShaderStage::Fragment,
            entry_point: entry_point.map(str::to_string),
            bindings,
            argument_buffer_fields: embedded_argument_fields(&meta.embedded_arguments),
            vertex_attributes: Vec::new(),
            varyings,
            render_targets,
            depth_members: meta.depth_members.clone(),
            depth_qualifier: meta.depth_qualifier,
            stencil_members: meta.stencil_members.clone(),
            local_size: None,
            kernel_dispatch: None,
            vertex_builtins: None,
            tessellation: None,
            imageblock_layouts: Vec::new(),
            implicit_imageblock_attachments: implicit_imageblock_planes(
                &meta.implicit_imageblock_attachments,
            ),
            fragment_imageblock: meta.fragment_imageblock.as_ref().map(|imageblock| {
                FragmentImageblock {
                    sample_size: imageblock.sample_size,
                    members: imageblock
                        .members
                        .iter()
                        .enumerate()
                        .map(|(index, member)| {
                            let index = index as u32;
                            let reads = imageblock.inputs.iter().any(|projection| {
                                projection
                                    .members
                                    .iter()
                                    .any(|projected| projected.master_member == index)
                            });
                            let writes = imageblock.outputs.iter().any(|projection| {
                                projection
                                    .members
                                    .iter()
                                    .any(|projected| projected.master_member == index)
                            });
                            FragmentImageblockMember {
                                offset: member.offset,
                                size: member.size,
                                type_name: member.type_name.clone(),
                                semantic: member.semantic.clone(),
                                raster_order_group: member.raster_order_group,
                                binding: (reads || writes)
                                    .then(|| fragment_imageblock_resource_binding(index))
                                    .flatten(),
                                access: match (reads, writes) {
                                    (true, true) => ResourceAccess::ReadWrite,
                                    (true, false) => ResourceAccess::ReadOnly,
                                    (false, true) => ResourceAccess::WriteOnly,
                                    (false, false) => ResourceAccess::Unused,
                                },
                            }
                        })
                        .collect(),
                    inputs: imageblock.inputs.clone(),
                    outputs: imageblock.outputs.clone(),
                }
            }),
            datalayout: None,
            runtime_sampler_specializations: Vec::new(),
            runtime_storage_image_specializations: Vec::new(),
            function_constants: Vec::new(),
        }
    }

    /// Build reflection for a vertex shader from its parsed meta and (optional) entry name.
    pub fn from_vertex(meta: &VertMeta, entry_point: Option<&str>) -> Self {
        let is_tessellation = meta.is_tessellation_evaluation();
        let vertex_builtins = VertexBuiltins {
            uses_vertex_index: meta.roles.iter().any(|(_, r)| *r == VertRole::VertexId),
            uses_instance_index: !is_tessellation
                && meta.roles.iter().any(|(_, r)| *r == VertRole::InstanceId),
            writes_position: meta.output_roles.contains(&VertOutRole::Position),
        };
        let mut bindings = Vec::new();
        for (idx, role) in &meta.roles {
            let idx = *idx;
            let binding = match role {
                VertRole::Buffer(n) => ResourceBinding {
                    kind: ResourceKind::Buffer,
                    metal_index: *n,
                    descriptor: ResourceBinding::descriptor_at(buffer_resource_binding(*n)),
                    param_index: Some(idx),
                    stage_input_location: None,
                    address_space: meta.buffer_address_spaces.get(&idx).copied(),
                    declared_size: meta.buffer_type_sizes.get(&idx).copied(),
                    extent: Some(buffer_extent(
                        meta.buffer_object_sizes.get(&idx).copied(),
                        meta.buffer_type_sizes.get(&idx).copied(),
                        meta.buffer_type_names.get(&idx),
                    )),
                    footprint: None,
                    type_layout: meta.buffer_layouts.get(&idx).cloned(),
                    type_name: meta.buffer_type_names.get(&idx).cloned(),
                    texture_shape: None,
                    embedded_source: None,
                    access: buffer_access(
                        meta.buffer_accesses.get(&idx).copied(),
                        meta.buffer_address_spaces.get(&idx).copied(),
                    ),
                    static_sampler: None,
                },
                VertRole::Texture(n) => texture_binding(*n, Some(idx), &meta.texture_type_names),
                VertRole::Sampler(n) => sampler_binding(*n, Some(idx)),
                VertRole::VisibleFunctionTable(n) => {
                    function_table_binding(ResourceKind::VisibleFunctionTable, *n, idx)
                }
                VertRole::IntersectionFunctionTable(n) => {
                    function_table_binding(ResourceKind::IntersectionFunctionTable, *n, idx)
                }
                VertRole::VertexInput(_)
                | VertRole::VertexId
                | VertRole::InstanceId
                | VertRole::PatchControlPoints
                | VertRole::PatchInput(_)
                | VertRole::PositionInPatch
                | VertRole::PatchId
                | VertRole::AmplificationId
                | VertRole::AmplificationCount
                | VertRole::Other => continue,
            };
            bindings.push(binding);
        }
        append_embedded_resources(
            &mut bindings,
            &meta.embedded_textures,
            &meta.embedded_arguments,
        );
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
                    type_name: meta.output_varying_types.get(loc).cloned(),
                    name: meta.output_varying_names.get(loc).cloned(),
                    user_semantic: meta.output_varying_user_semantics.get(loc).cloned(),
                }),
                _ => None,
            })
            .collect();
        ShaderReflection {
            reflection_version: REFLECTION_VERSION,
            descriptor_layout: DescriptorLayout::default(),
            stage: if is_tessellation {
                ShaderStage::TessellationEvaluation
            } else {
                ShaderStage::Vertex
            },
            entry_point: entry_point.map(str::to_string),
            bindings,
            argument_buffer_fields: embedded_argument_fields(&meta.embedded_arguments),
            vertex_attributes,
            varyings,
            render_targets: Vec::new(),
            depth_members: Vec::new(),
            depth_qualifier: None,
            stencil_members: Vec::new(),
            local_size: None,
            kernel_dispatch: None,
            vertex_builtins: Some(vertex_builtins),
            tessellation: meta.tessellation.as_ref().map(|tessellation| {
                let mut patch_input_locations = meta
                    .roles
                    .iter()
                    .filter_map(|(_, role)| match role {
                        VertRole::PatchInput(location) => Some(*location),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                patch_input_locations.sort_unstable();
                let patch_attributes = patch_input_locations
                    .iter()
                    .map(|location| TessellationAttribute {
                        location: *location,
                        type_name: meta.patch_input_types.get(location).cloned(),
                    })
                    .collect();
                TessellationInterface {
                    domain: tessellation.domain,
                    control_point_count: tessellation.control_point_count,
                    control_point_locations: tessellation
                        .control_point_fields
                        .iter()
                        .map(|field| field.location)
                        .collect(),
                    patch_input_locations,
                    control_point_attributes: tessellation
                        .control_point_fields
                        .iter()
                        .map(|field| TessellationAttribute {
                            location: field.location,
                            type_name: field.type_name.clone(),
                        })
                        .collect(),
                    patch_attributes,
                    instance_id: tessellation_system_attribute(meta, &VertRole::InstanceId),
                    amplification_id: tessellation_system_attribute(
                        meta,
                        &VertRole::AmplificationId,
                    ),
                    amplification_count: tessellation_system_attribute(
                        meta,
                        &VertRole::AmplificationCount,
                    ),
                }
            }),
            imageblock_layouts: Vec::new(),
            implicit_imageblock_attachments: implicit_imageblock_planes(
                &meta.implicit_imageblock_attachments,
            ),
            fragment_imageblock: None,
            datalayout: None,
            runtime_sampler_specializations: Vec::new(),
            runtime_storage_image_specializations: Vec::new(),
            function_constants: Vec::new(),
        }
    }

    /// Build reflection for a compute kernel from its parsed meta, entry name, and local size.
    pub fn from_kernel(meta: &KernMeta, entry_point: Option<&str>, local_size: [u32; 3]) -> Self {
        let mut bindings = Vec::new();
        let stage_input_bindings = meta.stage_input_bindings();
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
                            ResourceBinding::descriptor_at(buffer_resource_binding(*n))
                        },
                        param_index: Some(idx),
                        stage_input_location: None,
                        address_space,
                        declared_size: meta.buffer_type_sizes.get(&idx).copied(),
                        extent: Some(buffer_extent(
                            meta.buffer_object_sizes.get(&idx).copied(),
                            meta.buffer_type_sizes.get(&idx).copied(),
                            meta.buffer_type_names.get(&idx),
                        )),
                        footprint: None,
                        type_layout: meta.buffer_layouts.get(&idx).cloned(),
                        type_name: meta.buffer_type_names.get(&idx).cloned(),
                        texture_shape: None,
                        embedded_source: None,
                        access: buffer_access(
                            meta.buffer_accesses.get(&idx).copied(),
                            address_space,
                        ),
                        static_sampler: None,
                    }
                }
                KernRole::Texture(n) => texture_binding(*n, Some(idx), &meta.texture_type_names),
                KernRole::Sampler(n) => sampler_binding(*n, Some(idx)),
                KernRole::StageInput(location) => {
                    let metal_index = stage_input_bindings
                        .get(&idx)
                        .copied()
                        .expect("stage-input binding was allocated");
                    ResourceBinding {
                        kind: ResourceKind::KernelStageInput,
                        metal_index,
                        descriptor: ResourceBinding::descriptor_at(buffer_resource_binding(
                            metal_index,
                        )),
                        param_index: Some(idx),
                        stage_input_location: Some(*location),
                        address_space: None,
                        declared_size: None,
                        extent: Some(BufferExtent::Unbounded),
                        footprint: None,
                        type_layout: None,
                        type_name: meta.stage_input_type_names.get(&idx).cloned(),
                        texture_shape: None,
                        embedded_source: None,
                        access: Some(ResourceAccess::ReadOnly),
                        static_sampler: None,
                    }
                }
                KernRole::AccelerationStructureShadow(n) => ResourceBinding {
                    kind: ResourceKind::AccelerationStructureShadow,
                    metal_index: *n,
                    descriptor: ResourceBinding::descriptor_at(buffer_resource_binding(*n)),
                    param_index: Some(idx),
                    stage_input_location: None,
                    address_space: None,
                    declared_size: None,
                    extent: None,
                    footprint: None,
                    type_layout: None,
                    type_name: None,
                    texture_shape: None,
                    embedded_source: None,
                    access: None,
                    static_sampler: None,
                },
                KernRole::PrimitiveAccelerationStructure(n) => ResourceBinding {
                    kind: ResourceKind::PrimitiveAccelerationStructure,
                    metal_index: *n,
                    descriptor: None,
                    param_index: Some(idx),
                    stage_input_location: None,
                    address_space: None,
                    declared_size: None,
                    extent: None,
                    footprint: None,
                    type_layout: None,
                    type_name: Some("acceleration_structure<>".into()),
                    texture_shape: None,
                    embedded_source: None,
                    access: Some(ResourceAccess::ReadOnly),
                    static_sampler: None,
                },
                KernRole::PrimitiveAccelerationStructureShadow(n) => ResourceBinding {
                    kind: ResourceKind::PrimitiveAccelerationStructure,
                    metal_index: *n,
                    descriptor: ResourceBinding::descriptor_at(buffer_resource_binding(*n)),
                    param_index: Some(idx),
                    stage_input_location: None,
                    address_space: None,
                    declared_size: None,
                    extent: Some(BufferExtent::Unbounded),
                    footprint: None,
                    type_layout: None,
                    type_name: Some("acceleration_structure<>".into()),
                    texture_shape: None,
                    embedded_source: None,
                    access: Some(ResourceAccess::ReadOnly),
                    static_sampler: None,
                },
                KernRole::VisibleFunctionTable(n) => {
                    function_table_binding(ResourceKind::VisibleFunctionTable, *n, idx)
                }
                KernRole::IntersectionFunctionTable(n) => {
                    function_table_binding(ResourceKind::IntersectionFunctionTable, *n, idx)
                }
                _ => continue,
            };
            bindings.push(binding);
        }
        append_embedded_resources(
            &mut bindings,
            &meta.embedded_textures,
            &meta.embedded_arguments,
        );
        ShaderReflection {
            reflection_version: REFLECTION_VERSION,
            descriptor_layout: DescriptorLayout::default(),
            stage: ShaderStage::Kernel,
            entry_point: entry_point.map(str::to_string),
            bindings,
            argument_buffer_fields: embedded_argument_fields(&meta.embedded_arguments),
            vertex_attributes: Vec::new(),
            varyings: Vec::new(),
            render_targets: Vec::new(),
            depth_members: Vec::new(),
            depth_qualifier: None,
            stencil_members: Vec::new(),
            local_size: Some(local_size),
            kernel_dispatch: Some(KernelDispatch::safe_default()),
            vertex_builtins: None,
            tessellation: None,
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
            implicit_imageblock_attachments: implicit_imageblock_planes(
                &meta.implicit_imageblock_attachments,
            ),
            fragment_imageblock: None,
            datalayout: None,
            runtime_sampler_specializations: Vec::new(),
            runtime_storage_image_specializations: Vec::new(),
            function_constants: Vec::new(),
        }
    }

    /// The binding for a given resource kind + Metal index, if present.
    pub fn binding_at(&self, kind: ResourceKind, metal_index: u32) -> Option<&ResourceBinding> {
        self.bindings
            .iter()
            .find(|b| b.kind == kind && b.metal_index == metal_index)
    }

    /// Tighten declared buffer access using LLVM parameter attributes on the translated entry.
    /// `readonly` / `writeonly` apply transitively to calls made by the function and are therefore
    /// stronger than AIR's sometimes-broad `air.read_write` metadata. A parameter absent from the
    /// body is classified `Unused`; ambiguous uses retain the conservative declared classification.
    pub(crate) fn refine_buffer_access_from_entry(&mut self, ll: &str) {
        let Some(entry) = self.entry_point.as_deref() else {
            return;
        };
        let Some((args, body)) = llvm_entry_args_and_body(ll, entry) else {
            return;
        };
        for binding in &mut self.bindings {
            if !matches!(
                binding.kind,
                ResourceKind::Buffer | ResourceKind::ThreadgroupBuffer
            ) {
                continue;
            }
            let Some(param_index) = binding
                .param_index
                .and_then(|idx| usize::try_from(idx).ok())
            else {
                continue;
            };
            let Some(arg) = args.get(param_index) else {
                continue;
            };
            let Some(name) = percent_tokens(arg).last().copied() else {
                continue;
            };
            if !ssa_token_occurs(body, name) || llvm_arg_has_attribute(arg, "readnone") {
                binding.access = Some(ResourceAccess::Unused);
            } else if llvm_arg_has_attribute(arg, "readonly") {
                binding.access = Some(ResourceAccess::ReadOnly);
            } else if llvm_arg_has_attribute(arg, "writeonly") {
                binding.access = Some(ResourceAccess::WriteOnly);
            }
        }
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
            let binding = (self.descriptor_layout.samplers.start
                ..self.descriptor_layout.samplers.end)
                .find(|binding| !occupied.contains(binding))
                .ok_or_else(|| {
                    format!(
                        "AIR constexpr sampler count exceeds descriptor band \
                         [{},{})",
                        self.descriptor_layout.samplers.start, self.descriptor_layout.samplers.end
                    )
                })?;
            occupied.insert(binding);
            self.bindings.push(ResourceBinding {
                kind: ResourceKind::StaticSampler,
                metal_index: binding - self.descriptor_layout.samplers.start,
                descriptor: Some(DescriptorLocation {
                    set: self.descriptor_layout.set,
                    binding,
                    count: 1,
                }),
                param_index: None,
                stage_input_location: None,
                address_space: None,
                declared_size: None,
                extent: None,
                footprint: None,
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

fn llvm_entry_args_and_body<'a>(ll: &'a str, entry: &str) -> Option<(Vec<&'a str>, &'a str)> {
    let plain = format!("@{entry}(");
    let quoted = format!("@\"{entry}\"(");
    let start = ll.lines().position(|line| {
        line.trim_start().starts_with("define ")
            && (line.contains(&plain) || line.contains(&quoted))
    })?;
    let byte_start = ll
        .lines()
        .take(start)
        .map(|line| line.len() + 1)
        .sum::<usize>();
    let function = &ll[byte_start..];
    // Search from the entry symbol, rather than taking the first `{` / `(` in the header: an LLVM
    // function may return an aggregate such as `<{ i32, i32 }>`, and those delimiters precede its
    // parameter list.
    let symbol = function.find(&plain).or_else(|| function.find(&quoted))?;
    let open = function[symbol..].find('(')? + symbol;
    let mut depth = 0u32;
    let mut close = None;
    for (offset, character) in function[open..].char_indices() {
        match character {
            '(' => depth += 1,
            ')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    close = Some(open + offset);
                    break;
                }
            }
            _ => {}
        }
    }
    let close = close?;
    let header_end = function[close + 1..].find('{')? + close + 1;
    let args = split_top_level_llvm(&function[open + 1..close]);
    let tail = &function[header_end + 1..];
    let body_end = tail.find("\n}").unwrap_or(tail.len());
    Some((args, &tail[..body_end]))
}

fn split_top_level_llvm(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut depth = 0i32;
    for (index, character) in text.char_indices() {
        match character {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' => depth -= 1,
            ',' if depth == 0 => {
                out.push(text[start..index].trim());
                start = index + 1;
            }
            _ => {}
        }
    }
    if !text[start..].trim().is_empty() {
        out.push(text[start..].trim());
    }
    out
}

fn percent_tokens(text: &str) -> Vec<&str> {
    text.split('%')
        .skip(1)
        .filter_map(|tail| {
            let token = tail
                .split(|character: char| {
                    !(character.is_ascii_alphanumeric() || character == '_' || character == '.')
                })
                .next()?;
            (!token.is_empty()).then_some(token)
        })
        .collect()
}

fn ssa_token_occurs(text: &str, name: &str) -> bool {
    percent_tokens(text).contains(&name)
}

fn llvm_arg_has_attribute(arg: &str, attribute: &str) -> bool {
    arg.split(|character: char| !(character.is_ascii_alphanumeric() || character == '_'))
        .any(|token| token == attribute)
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
    let descriptor_binding = if access == ResourceAccess::Storage {
        storage_texture_resource_binding(n)
    } else {
        texture_resource_binding(n)
    };
    let mut descriptor = ResourceBinding::descriptor_at(descriptor_binding);
    if kind == ResourceKind::TextureArray {
        descriptor.as_mut().expect("texture descriptor").count =
            crate::meta::TEXTURE_HANDLE_ARRAY_DESCRIPTOR_COUNT;
    }
    ResourceBinding {
        kind,
        metal_index: n,
        descriptor,
        param_index,
        stage_input_location: None,
        address_space: None,
        declared_size: None,
        extent: None,
        footprint: None,
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

fn buffer_extent(
    object_size: Option<u32>,
    declared_size: Option<u32>,
    type_name: Option<&String>,
) -> BufferExtent {
    match (object_size, declared_size, type_name) {
        (Some(bytes), _, _) => BufferExtent::Object { bytes },
        (None, Some(_), _) | (None, None, Some(_)) => BufferExtent::Unbounded,
        (None, None, None) => BufferExtent::Unknown,
    }
}

fn buffer_access(
    declared: Option<BufferAccess>,
    address_space: Option<u32>,
) -> Option<ResourceAccess> {
    match declared {
        Some(BufferAccess::ReadOnly) => Some(ResourceAccess::ReadOnly),
        Some(BufferAccess::WriteOnly) => Some(ResourceAccess::WriteOnly),
        Some(BufferAccess::ReadWrite) => Some(ResourceAccess::ReadWrite),
        None if address_space == Some(ADDRESS_SPACE_CONSTANT) => Some(ResourceAccess::ReadOnly),
        None => None,
    }
}

fn append_embedded_resources(
    bindings: &mut Vec<ResourceBinding>,
    textures: &[crate::meta::EmbeddedTexture],
    arguments: &[crate::meta::EmbeddedArgument],
) {
    for embedded in textures {
        let descriptor_binding = if embedded.storage_format.is_some() {
            storage_texture_resource_binding(embedded.synthetic_texture_index)
                .unwrap_or(STORAGE_TEXTURE_BINDING_RANGE.end)
        } else {
            texture_resource_binding(embedded.synthetic_texture_index)
                .unwrap_or(TEXTURE_BINDING_RANGE.end)
        };
        bindings.push(ResourceBinding {
            kind: ResourceKind::EmbeddedArgBufferTexture,
            metal_index: embedded.synthetic_texture_index,
            descriptor: Some(DescriptorLocation {
                set: RESOURCE_DESCRIPTOR_SET,
                binding: descriptor_binding,
                count: embedded.array_length.unwrap_or(1),
            }),
            param_index: None,
            stage_input_location: None,
            address_space: None,
            declared_size: None,
            extent: None,
            footprint: None,
            type_layout: None,
            type_name: None,
            texture_shape: Some(TextureShape {
                dimension: TextureDimension::from_spirv_dim(embedded.dim),
                arrayed: embedded.arrayed,
                multisampled: false,
                component: TextureComponent::from_image_comp(embedded.comp),
                writable: embedded.storage_format.is_some(),
                array_ref: embedded.array_length.is_some(),
                array_length: embedded.array_length,
                storage_format: embedded.storage_format,
            }),
            embedded_source: Some(EmbeddedArgBuffer {
                buffer_param_index: embedded.buffer_param_index,
                buffer_index: embedded.buffer_index,
                field_offset: embedded.field_offset,
                field_ordinal: embedded.field_ordinal,
                argument_index: embedded.argument_index,
                resource_buffer_index: None,
            }),
            access: Some(if embedded.storage_format.is_some() {
                ResourceAccess::Storage
            } else {
                ResourceAccess::Sampled
            }),
            static_sampler: None,
        });
    }
    for argument in arguments {
        let Some(resource_index) = argument.resource_buffer_index else {
            continue;
        };
        bindings.push(ResourceBinding {
            kind: ResourceKind::EmbeddedArgBufferBuffer,
            metal_index: resource_index,
            descriptor: None,
            param_index: None,
            stage_input_location: None,
            address_space: argument.resource_address_space,
            declared_size: argument.resource_declared_size,
            extent: Some(buffer_extent(None, argument.resource_declared_size, None)),
            footprint: None,
            type_layout: None,
            type_name: None,
            texture_shape: None,
            embedded_source: Some(EmbeddedArgBuffer {
                buffer_param_index: argument.buffer_param_index,
                buffer_index: argument.buffer_index,
                field_offset: argument.field_offset,
                field_ordinal: argument.field_ordinal,
                argument_index: argument.argument_index,
                resource_buffer_index: Some(resource_index),
            }),
            access: buffer_access(argument.resource_access, argument.resource_address_space),
            static_sampler: None,
        });
    }
}

fn embedded_argument_fields(arguments: &[crate::meta::EmbeddedArgument]) -> Vec<EmbeddedArgBuffer> {
    arguments
        .iter()
        .map(|argument| EmbeddedArgBuffer {
            buffer_param_index: argument.buffer_param_index,
            buffer_index: argument.buffer_index,
            field_offset: argument.field_offset,
            field_ordinal: argument.field_ordinal,
            argument_index: argument.argument_index,
            resource_buffer_index: argument.resource_buffer_index,
        })
        .collect()
}

fn sampler_binding(n: u32, param_index: Option<u32>) -> ResourceBinding {
    ResourceBinding {
        kind: ResourceKind::Sampler,
        metal_index: n,
        descriptor: ResourceBinding::descriptor_at(sampler_resource_binding(n)),
        param_index,
        stage_input_location: None,
        address_space: None,
        declared_size: None,
        extent: None,
        footprint: None,
        type_layout: None,
        type_name: None,
        texture_shape: None,
        embedded_source: None,
        access: None,
        static_sampler: None,
    }
}

fn function_table_binding(
    kind: ResourceKind,
    metal_index: u32,
    param_index: u32,
) -> ResourceBinding {
    let type_name = match kind {
        ResourceKind::VisibleFunctionTable => "visible_function_table",
        ResourceKind::IntersectionFunctionTable => "intersection_function_table",
        _ => unreachable!("function-table helper requires a function-table kind"),
    };
    ResourceBinding {
        kind,
        metal_index,
        descriptor: None,
        param_index: Some(param_index),
        stage_input_location: None,
        address_space: None,
        declared_size: None,
        extent: None,
        footprint: None,
        type_layout: None,
        type_name: Some(type_name.into()),
        texture_shape: None,
        embedded_source: None,
        access: Some(ResourceAccess::ReadOnly),
        static_sampler: None,
    }
}

#[cfg(test)]
mod tests;
