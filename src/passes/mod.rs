//! Retained-SPIR-V transformation pipeline. The native emitter (`native/`, with its primary
//! structured-control-flow planning in `native/cfg/`) produces crate-owned Logical GLSL450 SPIR-V.
//! These passes then close both the Vulkan interface and residual AIR semantics. They:
//!   1. turn entry parameters into Vulkan interface variables by their AIR role
//!      (varying -> Input@Location, texture -> UniformConstant image, sampler -> UniformConstant
//!      sampler, buffer -> StorageBuffer Block@set/binding);
//!   2. turn the entry's return value into Output variable(s) @Location (MRT = struct split);
//!   3. lower the residual `air.*` OpFunctionCalls (sample -> OpImageSample*, math -> GLSL.std.450,
//!      dfdx/dfdy -> OpDPdx/OpDPdy, discard -> OpKill, ...);
//!   4. normalize typed access and Workgroup memory; and
//!   5. synthesize OpEntryPoint + OpExecutionMode, close capabilities, and remove dead declarations.
//!
//! Everything operates on the crate-owned module representation (`crate::spirv_module::Module`),
//! whose instructions, operands, functions, and blocks are crate-owned nodes.

use crate::meta::{
    AirScalar, AirType, FragMeta, FragRole, KernMeta, KernRole, VertMeta, VertOutRole, VertRole,
};
use crate::reflect::{
    RuntimeSamplerState, RuntimeStorageImageState, StaticSamplerState,
    SAMPLER_ARGUMENT_COUNT_USIZE, TEXTURE_ARGUMENT_COUNT_USIZE,
};
use crate::spirv_module::{is_block_terminator, Block, Function, Instruction, Module, Operand};
use spirv::{
    BuiltIn, Decoration, Dim, FunctionControl, ImageFormat, MemorySemantics, Op, Scope,
    StorageClass, Word,
};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    Vertex,
    Fragment,
    /// Metal compute kernel (`!air.kernel`). GLCompute entry, LocalSize (64,1,1) default, compute
    /// thread/grid builtins -> Vulkan builtins or local-size constants, `air.buffer` -> SSBO.
    Kernel,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransformOptions {
    /// Complete descriptor-set and binding layout for this independently translated stage.
    pub descriptor_layout: crate::reflect::DescriptorLayout,
    pub kernel_local_size: [u32; 3],
    /// Explicit kernel dispatch contract. `None` selects dynamic exact-thread region decomposition
    /// with its payload at offset zero. `Workgroups` explicitly asserts that every dispatched
    /// workgroup is complete.
    pub kernel_dispatch: Option<crate::reflect::KernelDispatch>,
    /// M-D2: lower `air.simd_{sum,min,max,and,or,xor}` REDUCE as a `GroupNonUniform ClusteredReduce`
    /// over a 32-lane cluster (Metal's simdgroup width) instead of a whole-subgroup Reduce, so a
    /// driver whose subgroup is WIDER than 32 still reduces over exactly the 32 lanes Apple's `simd_*`
    /// intrinsics define. Default false → emitted bytes are identical to the whole-subgroup form; the
    /// caller opts in only for a wider-subgroup driver (see kb conformance M-D2, pending G7).
    pub simd_cluster32: bool,
    /// AIR `air.compile.denorms_disable` requests flush-to-zero behavior for floating-point
    /// denormals. Vulkan exposes this only through optional float-controls features, so the native
    /// path records the request but does not emit a portability-breaking execution mode.
    pub denorm_flush_to_zero_f32: bool,
    /// Raster sample count selected by the graphics pipeline. AIR can query this through
    /// `air.get_num_samples.i32`, while Vulkan has no shader instruction for the same pipeline
    /// state. Supply the exact pipeline value when translating such a module; leaving it `None`
    /// keeps unknown state honest and makes the intrinsic fail visibly.
    pub raster_sample_count: Option<u32>,
    /// Pipeline sampler state by Metal `[[sampler(n)]]` index. AIR does not carry dynamically bound
    /// sampler state, but pixel-coordinate samplers require operation-aware shader specialization
    /// to remain legal in Vulkan. Unspecified slots retain ordinary runtime sampling.
    pub runtime_sampler_states: [Option<RuntimeSamplerState>; SAMPLER_ARGUMENT_COUNT_USIZE],
    /// Runtime surface format and host features by top-level Metal `[[texture(n)]]` index or
    /// reflected synthetic embedded-texture index. Only storage-image bindings consume these
    /// entries; sampled textures retain their AIR-derived image type.
    pub runtime_storage_image_states:
        [Option<RuntimeStorageImageState>; TEXTURE_ARGUMENT_COUNT_USIZE],
}

impl Default for TransformOptions {
    fn default() -> Self {
        Self {
            descriptor_layout: crate::reflect::DescriptorLayout::default(),
            kernel_local_size: [64, 1, 1],
            kernel_dispatch: None,
            simd_cluster32: false,
            denorm_flush_to_zero_f32: false,
            raster_sample_count: None,
            runtime_sampler_states: [None; SAMPLER_ARGUMENT_COUNT_USIZE],
            runtime_storage_image_states: [None; TEXTURE_ARGUMENT_COUNT_USIZE],
        }
    }
}

impl TransformOptions {
    /// Select the complete kernel dispatch/grid contract. Push-constant offsets are validated before
    /// translation mutates the module.
    pub fn with_kernel_dispatch(
        mut self,
        dispatch: crate::reflect::KernelDispatch,
    ) -> Result<Self, String> {
        dispatch.validate()?;
        self.kernel_dispatch = Some(dispatch);
        Ok(self)
    }

    pub fn with_descriptor_layout(
        mut self,
        layout: crate::reflect::DescriptorLayout,
    ) -> Result<Self, crate::reflect::DescriptorLayoutError> {
        layout.validate()?;
        self.descriptor_layout = layout;
        Ok(self)
    }

    /// Specialize one dynamically bound Metal sampler. Indices outside the Metal sampler argument
    /// table and malformed numeric state fail before translation mutates the module.
    pub fn with_runtime_sampler(
        mut self,
        metal_index: u32,
        state: RuntimeSamplerState,
    ) -> Result<Self, String> {
        state.validate()?;
        let slot = usize::try_from(metal_index)
            .ok()
            .filter(|slot| *slot < SAMPLER_ARGUMENT_COUNT_USIZE)
            .ok_or_else(|| {
                format!(
                    "Metal sampler index {metal_index} exceeds runtime specialization range 0..{}",
                    SAMPLER_ARGUMENT_COUNT_USIZE
                )
            })?;
        self.runtime_sampler_states[slot] = Some(state);
        Ok(self)
    }

    pub(crate) fn validate_runtime_samplers(self) -> Result<(), String> {
        for (index, state) in self.runtime_sampler_states.iter().copied().enumerate() {
            if let Some(state) = state {
                state
                    .validate()
                    .map_err(|error| format!("runtime sampler {index}: {error}"))?;
            }
        }
        Ok(())
    }

    /// Specialize one dynamically bound Metal storage texture. Use the Metal texture index for a
    /// top-level binding, or the reflected synthetic `metal_index` for an embedded argument-buffer
    /// texture. The runtime format and host feature facts are validated before translation mutates
    /// the module.
    pub fn with_runtime_storage_image(
        mut self,
        metal_index: u32,
        state: RuntimeStorageImageState,
    ) -> Result<Self, String> {
        state.validate()?;
        let slot = usize::try_from(metal_index)
            .ok()
            .filter(|slot| *slot < TEXTURE_ARGUMENT_COUNT_USIZE)
            .ok_or_else(|| {
                format!(
                    "storage-image resource index {metal_index} exceeds runtime specialization range 0..{}",
                    TEXTURE_ARGUMENT_COUNT_USIZE
                )
            })?;
        self.runtime_storage_image_states[slot] = Some(state);
        Ok(self)
    }

    pub(crate) fn validate_runtime_storage_images(self) -> Result<(), String> {
        for (index, state) in self
            .runtime_storage_image_states
            .iter()
            .copied()
            .enumerate()
        {
            if let Some(state) = state {
                state
                    .validate()
                    .map_err(|error| format!("runtime storage image {index}: {error}"))?;
            }
        }
        Ok(())
    }
}

pub(crate) fn validate_kernel_dispatch_options(
    stage: Stage,
    options: TransformOptions,
) -> Result<(), String> {
    if !matches!(stage, Stage::Kernel) {
        return Ok(());
    }
    if options.kernel_local_size.contains(&0) {
        return Err("kernel LocalSize dimensions must be non-zero".to_string());
    }
    Ok(())
}

/// The sampled component type of a texture (the `OpTypeImage` "Sampled Type"). Metal integer
/// textures (`air.read_texture_2d.u.v4i32` / `.s.v4i32`) must produce an int-typed image so the
/// OpImageFetch result is `v4uint`/`v4int` and matches the AIR result struct member — a float image
/// (the common case) would emit `v4float` and fail spirv-val on the struct construct.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ImageComp {
    Float,
    Uint,
    Sint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuntimeStorageImageUse {
    Read,
    Write,
    Atomic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FragmentImageblockFormat {
    image_format: ImageFormat,
    component: ImageComp,
    bits: u32,
    lanes: u32,
}

fn fragment_imageblock_format(type_name: &str) -> Option<FragmentImageblockFormat> {
    let (image_format, component, bits, lanes) = match type_name {
        "half" => (ImageFormat::R16f, ImageComp::Float, 16, 1),
        "half4" => (ImageFormat::Rgba16f, ImageComp::Float, 16, 4),
        "uchar4" => (ImageFormat::Rgba8ui, ImageComp::Uint, 8, 4),
        "ushort" => (ImageFormat::R16ui, ImageComp::Uint, 16, 1),
        _ => return None,
    };
    Some(FragmentImageblockFormat {
        image_format,
        component,
        bits,
        lanes,
    })
}

/// Typed cache key for the singleton types the finalize pass synthesizes (cleanup-plan S4). Each has
/// at most one instance per module and carries its own scan predicate in `finalize.rs`, so keying the
/// memo by this enum replaces the former ad-hoc `"fnvoid"`/`"uchar"`/`"ushort"` string keys.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::passes) enum SingletonType {
    /// `OpTypeFunction` returning void with no parameters (`void ()`).
    FnVoid,
    /// `OpTypeInt 8 0` (unsigned char).
    Int8,
    /// `OpTypeInt 16 0` (unsigned short).
    Int16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::passes) struct ImageShape {
    pub(in crate::passes) dim: Dim,
    pub(in crate::passes) arrayed: bool,
    pub(in crate::passes) comp: ImageComp,
    pub(in crate::passes) multisampled: bool,
}

/// Typed cache key for the synthesized array type + numeric constants the lowering passes memoize
/// (cleanup-plan C3): replaces the former prefix-namespaced string keys (`arr_…`, `ci_…`, `cf_…`,
/// `ch_…`) with a structural enum. Each builder keeps its exact scan / no-scan behavior — this only
/// changes the key representation, not when an instance is minted vs reused — so it is byte-neutral.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(in crate::passes) enum SynthCacheKey {
    /// `OpTypeArray <elem> <len-const>` (keyed by element type + the length `OpConstant` id).
    Array { elem: Word, len_const: Word },
    /// `OpConstant <int_ty> <value>` synthesized by `const_int_of` (reuses an existing match).
    ConstInt { int_ty: Word, value: i64 },
    /// `OpConstant <float> <bits>` synthesized by `const_float`, keyed by the f32 bit pattern.
    ConstFloat { bits: u32 },
    /// `OpConstant <half> <bits>` synthesized by `const_half`, keyed by the binary16 bit pattern.
    ConstHalf { bits: u16 },
    /// `OpTypeInt <bits> <signed>` synthesized by `conversions::integer_type` (reuses an existing
    /// match).
    IntType { bits: u32, signed: bool },
    /// `OpTypeVector <elem> <lanes>` synthesized by `conversions::vector_type`.
    VecType { elem: Word, lanes: u32 },
    /// `OpConstantComposite <ty> <value×lanes>` synthesized by `access::const_composite_splat`.
    CompositeSplat { ty: Word, value: Word, lanes: u32 },
    /// The singleton `SubgroupLocalInvocationId` builtin Input variable (`matrix_shuffle`).
    SubgroupLocalInvocationIdInputVar,
}

/// All state needed while rewriting; the module owns result-id allocation.
struct Ctx {
    module: Module,
    emit_sidecar: crate::emit_sidecar::EmitSidecar,
    /// The AIR stage currently being lowered. Sampling rules differ by execution model: GLCompute
    /// cannot use implicit-LOD sampling unless derivative-group execution modes are present.
    stage: Stage,
    /// id of the GLSL.std.450 ext-inst import, created lazily.
    glsl_ext: Option<Word>,
    /// interface variable ids to list on OpEntryPoint (Input/Output for <=1.3; everything for 1.4+).
    interface: Vec<Word>,
    /// new type/constant/variable instructions to append to types_global_values.
    new_globals: Vec<Instruction>,
    /// memoized synthesized array-type + numeric-constant ids, keyed structurally by
    /// [`SynthCacheKey`]. Used by the bespoke synthesizers that carry their own scan predicate or
    /// mint-fresh scheme (`ty_array`, `const_int_of`, `const_float`, `const_half`); each keeps its
    /// exact scan / no-scan behavior, this map only changes the key representation.
    synth_cache: HashMap<SynthCacheKey, Word>,
    /// Singleton finalize-synthesized types, memoized by their typed identity rather than a string key
    /// (cleanup-plan S4): the `void ()` function type, and the `i8`/`i16` scalar int types. Each carries
    /// its own module scan predicate (a `void ()` OpTypeFunction; an OpTypeInt of the given width) so it
    /// reuses an existing matching type before minting one; at most one of each exists per module.
    singleton_types: HashMap<SingletonType, Word>,
    /// Memoized `get_or_create` results keyed structurally by the exact `(op, result_type, operands)`
    /// tuple, replacing the hand-written textual keys the ~23 `ty_*`/`const_*` builders used to pass
    /// (refactor S11). Memoization-only: `get_or_create` still linear-scans `types_global_values` +
    /// `new_globals` on a cache miss, so the returned id (and thus output bytes) is unchanged; the
    /// structural key merely unifies builders that previously used distinct strings for the same type.
    struct_cache: HashMap<(Op, Option<Word>, Vec<Operand>), Word>,
    /// Selected-function result types indexed for a pass that performs repeated structural value
    /// queries. The index is installed only while that pass runs and misses fall through to the
    /// module, so lazily synthesized globals retain their ordinary lookup behavior.
    phase_value_types: Option<HashMap<Word, Word>>,
    /// Compact positions for type definitions during the module-wide memory phase. Lookups verify
    /// the result id at the recorded position, so an insertion that shifts either vector safely
    /// falls back to the ordinary search.
    phase_type_positions: Option<HashMap<Word, (bool, usize)>>,
    /// loaded-image id -> (Dim, arrayed), so the sample lowering builds the matching sampled-image
    /// type + coordinate shape.
    image_dims: HashMap<Word, (Dim, bool)>,
    /// loaded-image id -> sampled component type (default Float), so integer textures fetch/sample
    /// into a matching int vector.
    image_comp: HashMap<Word, ImageComp>,
    /// loaded-image ids whose AIR texture type is multisampled (`texture*_ms`), so reads use an MS
    /// image type and pass a Sample image operand instead of treating the sample id as a mip LOD.
    image_multisampled: HashSet<Word>,
    /// loaded-image ids that are STORAGE images (`OpTypeImage Sampled=2` with an explicit ImageFormat),
    /// the write-texture binding. `air.write_texture_*` lowers to `OpImageWrite` only on these.
    image_storage: HashSet<Word>,
    /// texture-array descriptor variable id -> (element `OpTypeImage` id, (Dim, arrayed), comp, MS). Set
    /// by the interface `ParamBinding::ImageArray` binding for runtime-indexed texture-handle arrays.
    /// `materialize_texture_array_loads` reads it to turn each `OpLoad` of a handle from a
    /// dynamically-indexed array element into `OpAccessChain %arrayvar %idx` + `OpLoad %image`.
    image_array_vars: HashMap<Word, (Word, (Dim, bool), ImageComp, bool)>,
    /// StorageBuffer descriptor variables introduced for AIR buffer parameters. Retained until the
    /// memory phase has consumed every source-shaped access so resource construction can omit only
    /// dead pointer projections rooted at an exact bound buffer.
    bound_buffer_vars: HashSet<Word>,
    /// loaded-image ids synthesized by `air.get_null_texture_*()`.
    null_image_values: HashSet<Word>,
    /// composite type ids already given explicit Offset/ArrayStride layout (dedup; a type decorated
    /// twice is a validation error).
    laid_out: HashSet<Word>,
    /// Struct type ids reconstructed from `air.struct_type_info`, with their exact AIR member offsets.
    air_struct_offsets: HashMap<Word, Vec<u32>>,
    air_data_layout: Option<crate::layout::AirDataLayout>,
    descriptor_layout: crate::reflect::DescriptorLayout,
    /// GLCompute LocalSize and the value exposed to AIR `[[threads_per_threadgroup]]`.
    kernel_local_size: [u32; 3],
    /// Per-dimension constants used by the execution mode and shader-visible local-size values.
    /// Exact-thread dispatches use specialization constants; whole-workgroup dispatches use
    /// ordinary constants.
    kernel_local_size_ids: Option<[Word; 3]>,
    kernel_workgroup_size_id: Option<Word>,
    /// Shared source for AIR grid builtins and exact boundary-region decomposition.
    kernel_dispatch: crate::reflect::KernelDispatch,
    /// M-D2 simd-reduce clustering opt-in (see [`TransformOptions::simd_cluster32`]).
    simd_cluster32: bool,
    /// Exact graphics-pipeline sample count used to lower `air.get_num_samples.i32`.
    raster_sample_count: Option<u32>,
    runtime_sampler_states: [Option<RuntimeSamplerState>; SAMPLER_ARGUMENT_COUNT_USIZE],
    runtime_storage_image_states: [Option<RuntimeStorageImageState>; TEXTURE_ARGUMENT_COUNT_USIZE],
    /// Storage-image variable/load ids carrying caller-provided runtime format state.
    runtime_storage_image_values: HashMap<Word, (u32, RuntimeStorageImageState)>,
    /// Metal texture indices whose storage-image binding consumed its specialization entry.
    applied_runtime_storage_image_indices: HashSet<u32>,
    /// lazily-created default sampler variable id, for `air.get_read_sampler()` (a sampler-less
    /// `texture.read` still passes a sampler operand AIR-side; we synthesize one valid sampler).
    default_sampler_var: Option<Word>,
    /// Descriptor variables the translator invented to give an AIR value a legal SPIR-V type, with
    /// no Metal argument behind them, mapped to the binding each was decorated with. Retracted when
    /// unconsumed (see [`stage_input::drop_unconsumed_placeholder_descriptor_loads`]); the ones that
    /// survive are reported to reflection, since no Metal argument would otherwise describe them.
    placeholder_descriptor_vars: HashMap<Word, u32>,
    /// Loaded sampler value ids with an exact static or pipeline-provided state.
    sampler_states: HashMap<Word, StaticSamplerState>,
    /// Direct loads of runtime sampler bindings that were given pipeline state.
    specialized_runtime_sampler_values: HashSet<Word>,
    /// Sampler-typed aliases whose incoming values do not share one exact state. A sampling
    /// operation cannot soundly specialize one of these values without cloning the operation per
    /// incoming state, so it must fail visibly instead of falling back to native sampling.
    ambiguous_sampler_states: HashSet<Word>,
    /// lazily-created default (null) float image variables for `air.get_null_texture_*()`, keyed by
    /// (Dim, arrayed) so a 2D and a 3D null texture get distinct bindings/types.
    default_null_image_vars: HashMap<(Dim, bool), Word>,
    /// `(render-target attachment, AIR imageblock data rate)` -> storage-image variable/type/format.
    /// The image is 2D-arrayed; the AIR color/sample index selects its array layer.
    implicit_imageblock_vars: HashMap<(u32, u32), (Word, Word, ImageFormat)>,
    /// Custom fragment-imageblock master member -> storage-image variable/type. Unlike implicit
    /// attachment imageblocks these slots are selected by master-field ordinal.
    fragment_imageblock_vars: HashMap<u32, (Word, Word)>,
    /// The custom fragment imageblock contract requires ordered per-pixel access.
    uses_fragment_imageblock: bool,
    fragment_imageblock_coord_var: Option<Word>,
    /// Fragment output rewrite created a `BuiltIn FragDepth` Output variable.
    writes_frag_depth: bool,
}

impl Ctx {
    #[cfg(test)]
    fn new(module: Module) -> Self {
        Self::with_options(module, Stage::Kernel, TransformOptions::default())
    }

    #[cfg(test)]
    fn with_options(module: Module, stage: Stage, options: TransformOptions) -> Self {
        Self::with_options_and_sidecar(
            module,
            crate::emit_sidecar::EmitSidecar::default(),
            stage,
            options,
        )
    }

    fn with_options_and_sidecar(
        module: Module,
        emit_sidecar: crate::emit_sidecar::EmitSidecar,
        stage: Stage,
        options: TransformOptions,
    ) -> Self {
        let mut module = module;
        module.sync_id_bound_from_instructions();
        let air_struct_offsets = emit_sidecar.air_struct_offsets.clone();
        let air_data_layout = emit_sidecar.air_data_layout.clone();
        Ctx {
            module,
            emit_sidecar,
            stage,
            glsl_ext: None,
            interface: vec![],
            new_globals: vec![],
            synth_cache: HashMap::new(),
            singleton_types: HashMap::new(),
            struct_cache: HashMap::new(),
            phase_value_types: None,
            phase_type_positions: None,
            image_dims: HashMap::new(),
            image_comp: HashMap::new(),
            image_multisampled: HashSet::new(),
            image_storage: HashSet::new(),
            image_array_vars: HashMap::new(),
            bound_buffer_vars: HashSet::new(),
            null_image_values: HashSet::new(),
            laid_out: HashSet::new(),
            air_struct_offsets,
            air_data_layout,
            descriptor_layout: options.descriptor_layout,
            kernel_local_size: options.kernel_local_size,
            kernel_local_size_ids: None,
            kernel_workgroup_size_id: None,
            kernel_dispatch: options
                .kernel_dispatch
                .unwrap_or_else(crate::reflect::KernelDispatch::safe_default),
            simd_cluster32: options.simd_cluster32,
            raster_sample_count: options.raster_sample_count,
            runtime_sampler_states: options.runtime_sampler_states,
            runtime_storage_image_states: options.runtime_storage_image_states,
            runtime_storage_image_values: HashMap::new(),
            applied_runtime_storage_image_indices: HashSet::new(),
            default_sampler_var: None,
            sampler_states: HashMap::new(),
            specialized_runtime_sampler_values: HashSet::new(),
            ambiguous_sampler_states: HashSet::new(),
            default_null_image_vars: HashMap::new(),
            placeholder_descriptor_vars: HashMap::new(),
            implicit_imageblock_vars: HashMap::new(),
            fragment_imageblock_vars: HashMap::new(),
            uses_fragment_imageblock: false,
            fragment_imageblock_coord_var: None,
            writes_frag_depth: false,
        }
    }

    fn add_capability(&mut self, capability: spirv::Capability) {
        if self.module.capabilities.iter().any(|instruction| {
            instruction.operands.first() == Some(&Operand::Capability(capability))
        }) {
            return;
        }
        self.module.capabilities.push(Instruction::new(
            Op::Capability,
            None,
            None,
            vec![Operand::Capability(capability)],
        ));
    }

    fn specialize_storage_image_format(
        &mut self,
        metal_index: u32,
        air_format: ImageFormat,
        component: ImageComp,
    ) -> Result<(ImageFormat, Option<RuntimeStorageImageState>), String> {
        let Some(state) = usize::try_from(metal_index)
            .ok()
            .and_then(|index| self.runtime_storage_image_states.get(index))
            .copied()
            .flatten()
        else {
            return Ok((air_format, None));
        };
        let air_component = crate::meta::TextureComponent::from_image_comp(component);
        let runtime_component = state.format.component();
        if air_component != runtime_component {
            return Err(format!(
                "runtime storage image {metal_index}: AIR texels are {air_component:?}, but runtime format {:?} is {runtime_component:?}",
                state.format
            ));
        }
        self.applied_runtime_storage_image_indices
            .insert(metal_index);
        let format = state
            .format
            .explicit_format()
            .map(crate::meta::TextureFormat::to_spirv_format)
            .unwrap_or(ImageFormat::Unknown);
        if storage_image_format_needs_extended_capability(format) {
            self.add_capability(spirv::Capability::StorageImageExtendedFormats);
        }
        Ok((format, Some(state)))
    }

    fn register_runtime_storage_image_value(
        &mut self,
        value: Word,
        metal_index: u32,
        state: Option<RuntimeStorageImageState>,
    ) {
        if let Some(state) = state {
            self.runtime_storage_image_values
                .insert(value, (metal_index, state));
        }
    }

    fn require_runtime_storage_image_use(
        &mut self,
        image: Word,
        usage: RuntimeStorageImageUse,
    ) -> Result<(), String> {
        let specializations = self.runtime_storage_image_origins(image);
        if specializations.is_empty() {
            return Ok(());
        }
        let first_format = specializations[0].1.format.explicit_format();
        if specializations
            .iter()
            .any(|(_, state)| state.format.explicit_format() != first_format)
        {
            return Err(
                "differently formatted runtime storage images cannot be selected into one image value"
                    .into(),
            );
        }
        for (metal_index, state) in specializations {
            if usage == RuntimeStorageImageUse::Atomic {
                if !state.format.supports_atomics() {
                    return Err(format!(
                        "runtime storage image {metal_index}: format {:?} cannot implement storage-image atomics",
                        state.format
                    ));
                }
                if !state.capabilities.storage_image_atomic {
                    return Err(format!(
                        "runtime storage image {metal_index}: format {:?} lacks storage-image atomic support",
                        state.format
                    ));
                }
            }
            if state.format.explicit_format().is_some() {
                continue;
            }
            let (supported, capability, operation) = match usage {
                RuntimeStorageImageUse::Read => (
                    state.capabilities.read_without_format,
                    spirv::Capability::StorageImageReadWithoutFormat,
                    "read",
                ),
                RuntimeStorageImageUse::Write => (
                    state.capabilities.write_without_format,
                    spirv::Capability::StorageImageWriteWithoutFormat,
                    "write",
                ),
                RuntimeStorageImageUse::Atomic => {
                    return Err(format!(
                        "runtime storage image {metal_index}: formatless storage-image atomics are unsupported"
                    ));
                }
            };
            if !supported {
                return Err(format!(
                    "runtime storage image {metal_index}: format {:?} requires host {operation}-without-format support",
                    state.format
                ));
            }
            self.add_capability(capability);
        }
        Ok(())
    }

    fn runtime_storage_image_origins(&self, image: Word) -> Vec<(u32, RuntimeStorageImageState)> {
        let mut pending = vec![image];
        let mut visited = HashSet::new();
        let mut origins = Vec::new();
        while let Some(value) = pending.pop() {
            if !visited.insert(value) {
                continue;
            }
            if let Some(origin) = self.runtime_storage_image_values.get(&value).copied() {
                if !origins.contains(&origin) {
                    origins.push(origin);
                }
                continue;
            }
            let Some(instruction) = self
                .module
                .types_global_values
                .iter()
                .chain(self.new_globals.iter())
                .chain(
                    self.module
                        .functions
                        .iter()
                        .flat_map(|function| function.blocks.iter())
                        .flat_map(|block| block.instructions.iter()),
                )
                .find(|instruction| instruction.result_id == Some(value))
            else {
                continue;
            };
            match instruction.class.opcode {
                Op::CopyObject | Op::Load => {
                    if let Some(Operand::IdRef(source)) = instruction.operands.first() {
                        pending.push(*source);
                    }
                }
                Op::Select => {
                    pending.extend(instruction.operands.iter().skip(1).filter_map(|operand| {
                        match operand {
                            Operand::IdRef(source) => Some(*source),
                            _ => None,
                        }
                    }))
                }
                Op::Phi => pending.extend(instruction.operands.iter().enumerate().filter_map(
                    |(index, operand)| match (index % 2, operand) {
                        (0, Operand::IdRef(source)) => Some(*source),
                        _ => None,
                    },
                )),
                _ => {}
            }
        }
        origins.sort_unstable_by_key(|(metal_index, _)| *metal_index);
        origins
    }

    fn validate_runtime_storage_image_bindings(&self) -> Result<(), String> {
        for (metal_index, state) in self
            .runtime_storage_image_states
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, state)| Some((u32::try_from(index).ok()?, state?)))
        {
            if !self
                .applied_runtime_storage_image_indices
                .contains(&metal_index)
            {
                return Err(format!(
                    "runtime storage image {metal_index}: no storage-image binding exists for runtime format {:?}",
                    state.format
                ));
            }
        }
        Ok(())
    }
}

fn storage_image_format_needs_extended_capability(format: ImageFormat) -> bool {
    matches!(
        format,
        ImageFormat::Rg32f
            | ImageFormat::Rg16f
            | ImageFormat::R11fG11fB10f
            | ImageFormat::R16f
            | ImageFormat::Rgba16
            | ImageFormat::Rgb10A2
            | ImageFormat::Rg16
            | ImageFormat::Rg8
            | ImageFormat::R16
            | ImageFormat::R8
            | ImageFormat::Rgba16Snorm
            | ImageFormat::Rg16Snorm
            | ImageFormat::Rg8Snorm
            | ImageFormat::R16Snorm
            | ImageFormat::R8Snorm
            | ImageFormat::Rg32i
            | ImageFormat::Rg16i
            | ImageFormat::Rg8i
            | ImageFormat::R16i
            | ImageFormat::R8i
            | ImageFormat::Rgb10a2ui
            | ImageFormat::Rg32ui
            | ImageFormat::Rg16ui
            | ImageFormat::Rg8ui
            | ImageFormat::R16ui
            | ImageFormat::R8ui
    )
}

/// A small helper to build a type/constant instruction and register it in the cache + new_globals.
fn type_inst(op: Op, result: Word, operands: Vec<Operand>) -> Instruction {
    Instruction::new(op, None, Some(result), operands)
}

/// Convert an f32 to its IEEE-754 binary16 (half) bit pattern. Round-to-nearest-even; handles the
/// small set of exact values we synthesize (0.0, 1.0) plus general finite values for robustness.
pub(crate) fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x007f_ffff;
    if ((bits >> 23) & 0xff) == 0xff {
        // inf / nan
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    if exp >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if exp <= 0 {
        // subnormal / underflow to zero (sufficient for 0.0; our synthesized consts are 0/1).
        if exp < -10 {
            return sign;
        }
        let mant = (mant | 0x0080_0000) >> (1 - exp);
        let rounded = (mant + 0x0000_1000) >> 13;
        return sign | rounded as u16;
    }
    let half_mant = (mant + 0x0000_1000) >> 13; // round to nearest even (approx)
    sign | ((exp as u16) << 10) | half_mant as u16
}

/// Find the entry function by name (the Vulkan backend does NOT inline helpers, so a module has many
/// functions with bodies; only one is the AIR stage entry). Falls back to the first bodied function.
fn find_entry_index(module: &Module, entry_name: Option<&str>) -> Option<usize> {
    if let Some(name) = entry_name {
        // match via OpName <id> "<name>" -> the function whose def result_id is that id.
        let mut id_for_name = None;
        for inst in &module.debug_names {
            if inst.class.opcode == Op::Name {
                if let (Some(Operand::IdRef(id)), Some(Operand::LiteralString(s))) =
                    (inst.operands.first(), inst.operands.get(1))
                {
                    if s == name {
                        id_for_name = Some(*id);
                    }
                }
            }
        }
        if let Some(id) = id_for_name {
            if let Some(p) = module.functions.iter().position(|f| {
                !f.blocks.is_empty() && f.def.as_ref().and_then(|d| d.result_id) == Some(id)
            }) {
                return Some(p);
            }
        }
    }
    module.functions.iter().position(|f| !f.blocks.is_empty())
}

/// Carry exact sampler state through the sampler-typed SSA joins that can remain after interface
/// binding. A join is usable only when every incoming value has the same known state; mixing a
/// specialized sampler with an unspecialized or differently specialized sampler is recorded as
/// ambiguous and rejected by the image-call lowering.
fn propagate_sampler_state_aliases(ctx: &mut Ctx, entry_idx: usize) {
    let sampler_ty = ctx.ty_sampler();
    let aliases = ctx.module.functions[entry_idx]
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .filter_map(|instruction| {
            if instruction.result_type != Some(sampler_ty) {
                return None;
            }
            let result = instruction.result_id?;
            let sources = match instruction.class.opcode {
                Op::CopyObject => instruction
                    .operands
                    .first()
                    .and_then(|operand| match operand {
                        Operand::IdRef(source) => Some(vec![*source]),
                        _ => None,
                    })
                    .unwrap_or_default(),
                Op::Select => instruction
                    .operands
                    .iter()
                    .skip(1)
                    .filter_map(|operand| match operand {
                        Operand::IdRef(source) => Some(*source),
                        _ => None,
                    })
                    .collect(),
                Op::Phi => instruction
                    .operands
                    .iter()
                    .step_by(2)
                    .filter_map(|operand| match operand {
                        Operand::IdRef(source) => Some(*source),
                        _ => None,
                    })
                    .collect(),
                _ => return None,
            };
            (!sources.is_empty()).then_some((result, sources))
        })
        .collect::<Vec<_>>();
    // Blocks need not be serialized in dominance order. Seed the aliases that directly consume a
    // classified state, then visit their dependents once. This is bounded by the sampler-alias graph
    // rather than a whole-module fixpoint. An unresolved sibling input is conservatively ambiguous,
    // which is the same contract as an unspecialized input at a join.
    let mut dependents = HashMap::<Word, Vec<usize>>::new();
    for (index, (_, sources)) in aliases.iter().enumerate() {
        for source in sources {
            dependents.entry(*source).or_default().push(index);
        }
    }
    let mut queued = vec![false; aliases.len()];
    let mut queue = VecDeque::new();
    for (index, (_, sources)) in aliases.iter().enumerate() {
        if sources.iter().any(|source| {
            ctx.sampler_states.contains_key(source) || ctx.ambiguous_sampler_states.contains(source)
        }) {
            queued[index] = true;
            queue.push_back(index);
        }
    }
    while let Some(index) = queue.pop_front() {
        let (result, sources) = &aliases[index];
        let mut exact = None;
        let mut saw_known = false;
        let mut conflict = false;
        for source in sources {
            if ctx.ambiguous_sampler_states.contains(source) {
                conflict = true;
                saw_known = true;
                continue;
            }
            let Some(state) = ctx.sampler_states.get(source).copied() else {
                conflict = true;
                continue;
            };
            saw_known = true;
            if exact.is_some_and(|existing| existing != state) {
                conflict = true;
            } else {
                exact = Some(state);
            }
        }
        if !saw_known {
            continue;
        }
        if conflict {
            ctx.ambiguous_sampler_states.insert(*result);
        } else if let Some(state) = exact {
            ctx.sampler_states.insert(*result, state);
        }
        for dependent in dependents.get(result).into_iter().flatten() {
            if !queued[*dependent] {
                queued[*dependent] = true;
                queue.push_back(*dependent);
            }
        }
    }
}

#[cfg(test)]
mod sampler_state_alias_tests {
    use super::*;
    use crate::spirv_module::ModuleHeader;

    #[test]
    fn pass_context_allocator_advances_past_definitions_missing_from_the_header_bound() {
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(2));
        let mut ctx = Ctx::new(module);
        ctx.new_globals.push(Instruction::new(
            Op::TypePointer,
            None,
            Some(7),
            vec![
                Operand::StorageClass(StorageClass::Workgroup),
                Operand::IdRef(1),
            ],
        ));

        assert_eq!(ctx.ty_sampler(), 8);
        assert_eq!(
            ctx.module.header.as_ref().map(|header| header.bound),
            Some(9)
        );
    }

    #[test]
    fn propagation_follows_alias_sources_in_later_blocks() {
        let mut module = Module::new();
        module
            .types_global_values
            .push(Instruction::new(Op::TypeSampler, None, Some(1), vec![]));
        module.functions.push(Function {
            def: None,
            parameters: vec![],
            blocks: vec![
                Block {
                    label: None,
                    instructions: vec![Instruction::new(
                        Op::CopyObject,
                        Some(1),
                        Some(5),
                        vec![Operand::IdRef(4)],
                    )],
                },
                Block {
                    label: None,
                    instructions: vec![Instruction::new(
                        Op::CopyObject,
                        Some(1),
                        Some(4),
                        vec![Operand::IdRef(3)],
                    )],
                },
            ],
            end: None,
        });
        let state = StaticSamplerState::from_air_words([34901797601020416, 0])
            .expect("decode sampler state");
        let mut ctx = Ctx::new(module);
        ctx.sampler_states.insert(3, state);

        propagate_sampler_state_aliases(&mut ctx, 0);

        assert_eq!(ctx.sampler_states.get(&4), Some(&state));
        assert_eq!(ctx.sampler_states.get(&5), Some(&state));
        assert!(ctx.ambiguous_sampler_states.is_empty());
    }
}

/// Map every existing type id -> its defining instruction (clone), for type inspection.
fn type_defs(module: &Module) -> HashMap<Word, Instruction> {
    let mut m = HashMap::new();
    for inst in &module.types_global_values {
        if let Some(rid) = inst.result_id {
            m.insert(rid, inst.clone());
        }
    }
    m
}

/// The element type of a pointer type id, if it is OpTypePointer.
fn ptr_pointee(defs: &HashMap<Word, Instruction>, ptr: Word) -> Option<Word> {
    let inst = defs.get(&ptr)?;
    if inst.class.opcode == Op::TypePointer {
        if let Operand::IdRef(p) = inst.operands[1] {
            return Some(p);
        }
    }
    None
}

/// The storage class of a pointer type id.
fn ptr_storage(defs: &HashMap<Word, Instruction>, ptr: Word) -> Option<StorageClass> {
    let inst = defs.get(&ptr)?;
    if inst.class.opcode == Op::TypePointer {
        if let Operand::StorageClass(sc) = inst.operands[0] {
            return Some(sc);
        }
    }
    None
}

impl Ctx {
    /// Get-or-create skeleton shared by the `ty_*`/`const_*` synthesizers (refactor S6/S11): return
    /// the cached id for the structural `(op, result_type, operands)` key; else the id of the first
    /// existing `types_global_values`/`new_globals` instruction with exactly that shape; else allocate
    /// a fresh id, append the instruction, and cache it. `result_type` is `None` for type declarations
    /// (matching `type_inst`) and `Some(ty)` for constants. Because every foldable opcode here is
    /// fixed-arity, full-operand equality reproduces each caller's original per-operand scan predicate
    /// exactly. S11 replaced the caller-supplied textual key with this structural key: it is
    /// memoization-only (the linear scan still decides the returned id), so it is byte-identical while
    /// removing the ~23 hand-maintained key strings and collapsing builders that shared a shape.
    fn get_or_create(&mut self, op: Op, result_type: Option<Word>, operands: Vec<Operand>) -> Word {
        let key = (op, result_type, operands.clone());
        if let Some(&id) = self.struct_cache.get(&key) {
            return id;
        }
        let mut highest_definition = 0;
        let mut existing = None;
        for inst in self
            .module
            .types_global_values
            .iter()
            .chain(self.new_globals.iter())
        {
            highest_definition = highest_definition.max(inst.result_id.unwrap_or(0));
            if inst.class.opcode == op
                && inst.result_type == result_type
                && inst.operands == operands
            {
                if let Some(rid) = inst.result_id {
                    existing.get_or_insert(rid);
                }
            }
        }
        if self.module.id_bound() <= highest_definition {
            self.module
                .set_id_bound(highest_definition.saturating_add(1));
        }
        if let Some(id) = existing {
            self.struct_cache.insert(key, id);
            return id;
        }
        let id = self.module.fresh_id();
        self.new_globals
            .push(Instruction::new(op, result_type, Some(id), operands));
        self.struct_cache.insert(key, id);
        id
    }

    /// Get or create OpTypeFloat 32. (Width matters: a half shader has an `OpTypeFloat 16` that must
    /// not be mistaken for `float`.)
    fn ty_float(&mut self) -> Word {
        self.get_or_create(Op::TypeFloat, None, vec![Operand::LiteralBit32(32)])
    }

    /// Get or create OpTypeBool.
    fn ty_bool(&mut self) -> Word {
        self.get_or_create(Op::TypeBool, None, vec![])
    }

    /// Get or create OpTypeVector bool N.
    fn ty_vec_bool(&mut self, n: u32) -> Word {
        let b = self.ty_bool();
        self.get_or_create(
            Op::TypeVector,
            None,
            vec![Operand::IdRef(b), Operand::LiteralBit32(n)],
        )
    }

    /// Get or create OpTypeVector float N.
    fn ty_vecf(&mut self, n: u32) -> Word {
        let f = self.ty_float();
        self.get_or_create(
            Op::TypeVector,
            None,
            vec![Operand::IdRef(f), Operand::LiteralBit32(n)],
        )
    }

    /// Get or create OpTypeVector half N.
    fn ty_vech(&mut self, n: u32) -> Word {
        let h = self.ty_half();
        self.get_or_create(
            Op::TypeVector,
            None,
            vec![Operand::IdRef(h), Operand::LiteralBit32(n)],
        )
    }

    /// Get or create OpTypeVector unsigned 16-bit int N.
    fn ty_vec_u16(&mut self, n: u32) -> Word {
        let u16_ty = self.ty_int16();
        self.get_or_create(
            Op::TypeVector,
            None,
            vec![Operand::IdRef(u16_ty), Operand::LiteralBit32(n)],
        )
    }

    /// Get or create OpTypePointer <sc> <pointee>.
    fn ty_ptr(&mut self, sc: StorageClass, pointee: Word) -> Word {
        self.get_or_create(
            Op::TypePointer,
            None,
            vec![Operand::StorageClass(sc), Operand::IdRef(pointee)],
        )
    }

    /// Get or create OpTypeArray <elem> <len-const>.
    fn ty_array(&mut self, elem: Word, len: u32) -> Word {
        let len_c = self.const_uint(len);
        let key = SynthCacheKey::Array {
            elem,
            len_const: len_c,
        };
        if let Some(&id) = self.synth_cache.get(&key) {
            return id;
        }
        let id = self.module.fresh_id();
        self.new_globals.push(type_inst(
            Op::TypeArray,
            id,
            vec![Operand::IdRef(elem), Operand::IdRef(len_c)],
        ));
        self.synth_cache.insert(key, id);
        id
    }

    /// Build the SPIR-V type for a reconstructed AIR aggregate (`air.struct_type_info`). Mirrors the
    /// original Metal layout so flattened access-chain indices land correctly: a matrix
    /// `floatCxR` becomes `{ [cols x vec(rows)] }` (the `metal::matrix` wrapper struct), a struct
    /// becomes an `OpTypeStruct` of its members in order.
    fn build_air_type(&mut self, t: &AirType) -> Word {
        match t {
            AirType::Scalar(scalar) => self.ty_air_scalar(*scalar),
            AirType::Vec { scalar, lanes } => self.ty_air_vec(*scalar, *lanes),
            AirType::PackedVec { scalar, lanes } => {
                let elem = self.ty_air_scalar(*scalar);
                self.ty_array(elem, *lanes)
            }
            AirType::Array { elem, len } => {
                let elem_ty = self.build_air_type(elem);
                self.ty_array(elem_ty, *len)
            }
            AirType::Matrix { scalar, cols, rows } => {
                let col = self.ty_air_vec(*scalar, *rows);
                let arr = self.ty_array(col, *cols);
                let st = self.module.fresh_id();
                self.new_globals
                    .push(type_inst(Op::TypeStruct, st, vec![Operand::IdRef(arr)]));
                st
            }
            AirType::Struct(members) => {
                let mtys: Vec<Word> = members.iter().map(|m| self.build_air_type(&m.ty)).collect();
                let st = self.module.fresh_id();
                let offsets: Vec<u32> = members.iter().map(|m| m.offset).collect();
                if offsets.windows(2).all(|w| w[1] > w[0]) {
                    self.air_struct_offsets.insert(st, offsets);
                }
                self.new_globals.push(type_inst(
                    Op::TypeStruct,
                    st,
                    mtys.into_iter().map(Operand::IdRef).collect(),
                ));
                st
            }
        }
    }

    fn ty_air_scalar(&mut self, scalar: AirScalar) -> Word {
        match scalar {
            AirScalar::Float => self.ty_float(),
            AirScalar::Half => self.ty_half(),
            AirScalar::UInt => self.ty_uint(),
            AirScalar::ULong | AirScalar::SLong => self.ty_ulong(),
            AirScalar::UShort | AirScalar::SShort => self.ty_int16(),
            // LLVM integer values are signless. Keep buffer storage leaves as unsigned words so
            // reconstructed `int`/`packed_intN` layouts type-check against emitted loads/stores;
            // signed interpretation is carried by the signed SPIR-V opcodes.
            AirScalar::SInt => self.ty_uint(),
            AirScalar::UChar | AirScalar::Bool => self.ty_int8(),
        }
    }

    fn ty_air_vec(&mut self, scalar: AirScalar, lanes: u32) -> Word {
        match scalar {
            AirScalar::Float => self.ty_vecf(lanes),
            AirScalar::Half => self.ty_vech(lanes),
            AirScalar::UInt => self.ty_vec_uint(lanes),
            AirScalar::ULong | AirScalar::SLong => self.ty_vec_ulong(lanes),
            AirScalar::UShort | AirScalar::SShort => self.ty_vec_u16(lanes),
            AirScalar::SInt => self.ty_vec_uint(lanes),
            AirScalar::UChar | AirScalar::Bool => {
                let elem = self.ty_int8();
                self.ty_array(elem, lanes)
            }
        }
    }

    /// Get or create OpTypeRuntimeArray <elem>. Used to wrap a `device T*` (or a homogeneous struct
    /// emitted as a bare element pointer) as a StorageBuffer Block `{ RuntimeArray<T> }`
    /// so Logical SPIR-V can express `buf[i]`.
    fn ty_runtime_array(&mut self, elem: Word) -> Word {
        self.get_or_create(Op::TypeRuntimeArray, None, vec![Operand::IdRef(elem)])
    }

    /// Get or create OpTypeImage with the given sampled component type, 2D/1D, sampled.
    fn ty_image(&mut self, dim: Dim, arrayed: bool, comp: ImageComp) -> Word {
        self.ty_image_ms(dim, arrayed, comp, false)
    }

    fn ty_image_ms(
        &mut self,
        dim: Dim,
        arrayed: bool,
        comp: ImageComp,
        multisampled: bool,
    ) -> Word {
        let f = match comp {
            ImageComp::Float => self.ty_float(),
            ImageComp::Uint => self.ty_uint(),
            ImageComp::Sint => self.ty_sint(),
        };
        self.get_or_create(
            Op::TypeImage,
            None,
            vec![
                Operand::IdRef(f),
                Operand::Dim(dim),
                Operand::LiteralBit32(0), // Depth: not a depth image
                Operand::LiteralBit32(arrayed as u32), // Arrayed
                Operand::LiteralBit32(multisampled as u32), // MS
                Operand::LiteralBit32(1), // Sampled: 1 = used with sampler
                Operand::ImageFormat(ImageFormat::Unknown),
            ],
        )
    }

    /// Get or create an input-attachment image type (`subpassInput`/`SubpassData`).
    fn ty_input_attachment(&mut self, sampled: Word) -> Word {
        self.get_or_create(
            Op::TypeImage,
            None,
            vec![
                Operand::IdRef(sampled),
                Operand::Dim(Dim::DimSubpassData),
                Operand::LiteralBit32(0), // Depth
                Operand::LiteralBit32(0), // Arrayed
                Operand::LiteralBit32(0), // MS
                Operand::LiteralBit32(2), // Sampled: 2 = read without sampler
                Operand::ImageFormat(ImageFormat::Unknown),
            ],
        )
    }

    /// Get or create a STORAGE `OpTypeImage` (Sampled=2) with an explicit `ImageFormat`, the
    /// write-texture binding. The sampled component type must match the storage image format so
    /// `OpImageWrite` accepts float, uint, or sint texels.
    fn ty_storage_image(
        &mut self,
        dim: Dim,
        arrayed: bool,
        fmt: ImageFormat,
        comp: ImageComp,
    ) -> Word {
        let sampled = match comp {
            ImageComp::Float => self.ty_float(),
            ImageComp::Uint => self.ty_uint(),
            ImageComp::Sint => self.ty_sint(),
        };
        self.get_or_create(
            Op::TypeImage,
            None,
            vec![
                Operand::IdRef(sampled),
                Operand::Dim(dim),
                Operand::LiteralBit32(0),              // Depth
                Operand::LiteralBit32(arrayed as u32), // Arrayed
                Operand::LiteralBit32(0),              // MS
                Operand::LiteralBit32(2),              // Sampled: 2 = read/write (storage)
                Operand::ImageFormat(fmt),
            ],
        )
    }

    fn ty_sampler(&mut self) -> Word {
        self.get_or_create(Op::TypeSampler, None, vec![])
    }

    fn ty_sampled_image(&mut self, image: Word) -> Word {
        self.get_or_create(Op::TypeSampledImage, None, vec![Operand::IdRef(image)])
    }

    fn ty_void(&mut self) -> Word {
        self.get_or_create(Op::TypeVoid, None, vec![])
    }

    /// OpTypeInt 32 unsigned.
    fn ty_uint(&mut self) -> Word {
        self.get_or_create(
            Op::TypeInt,
            None,
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
        )
    }

    /// OpTypeInt 64 unsigned.
    fn ty_ulong(&mut self) -> Word {
        self.get_or_create(
            Op::TypeInt,
            None,
            vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
        )
    }

    /// OpTypeInt 32 signed.
    fn ty_sint(&mut self) -> Word {
        self.get_or_create(
            Op::TypeInt,
            None,
            vec![Operand::LiteralBit32(32), Operand::LiteralBit32(1)],
        )
    }

    /// Get or create OpTypeVector sint N.
    fn ty_vec_sint(&mut self, n: u32) -> Word {
        let s = self.ty_sint();
        self.get_or_create(
            Op::TypeVector,
            None,
            vec![Operand::IdRef(s), Operand::LiteralBit32(n)],
        )
    }

    /// Get or create OpTypeVector uint N (e.g. v3uint for the GlobalInvocationId builtin).
    fn ty_vec_uint(&mut self, n: u32) -> Word {
        let u = self.ty_uint();
        self.get_or_create(
            Op::TypeVector,
            None,
            vec![Operand::IdRef(u), Operand::LiteralBit32(n)],
        )
    }

    /// Get or create OpTypeVector ulong N.
    fn ty_vec_ulong(&mut self, n: u32) -> Word {
        let u = self.ty_ulong();
        self.get_or_create(
            Op::TypeVector,
            None,
            vec![Operand::IdRef(u), Operand::LiteralBit32(n)],
        )
    }

    /// OpConstant uint <v>.
    fn const_uint(&mut self, v: u32) -> Word {
        let uint = self.ty_uint();
        self.get_or_create(Op::Constant, Some(uint), vec![Operand::LiteralBit32(v)])
    }

    fn kernel_local_size_ids(&mut self) -> [Word; 3] {
        if let Some(ids) = self.kernel_local_size_ids {
            return ids;
        }
        let ids = if matches!(
            self.kernel_dispatch,
            crate::reflect::KernelDispatch::Workgroups
        ) {
            self.kernel_local_size.map(|value| self.const_uint(value))
        } else {
            let uint_ty = self.ty_uint();
            std::array::from_fn(|dimension| {
                let id = self.module.fresh_id();
                self.new_globals.push(Instruction::new(
                    Op::SpecConstant,
                    Some(uint_ty),
                    Some(id),
                    vec![Operand::LiteralBit32(self.kernel_local_size[dimension])],
                ));
                self.module.annotations.push(Instruction::new(
                    Op::Decorate,
                    None,
                    None,
                    vec![
                        Operand::IdRef(id),
                        Operand::Decoration(Decoration::SpecId),
                        Operand::LiteralBit32(
                            crate::reflect::KERNEL_LOCAL_SIZE_SPEC_IDS[dimension],
                        ),
                    ],
                ));
                id
            })
        };
        self.kernel_local_size_ids = Some(ids);
        ids
    }

    fn kernel_workgroup_size_id(&mut self) -> Word {
        if let Some(id) = self.kernel_workgroup_size_id {
            return id;
        }
        let ids = self.kernel_local_size_ids();
        let vector_ty = self.ty_vec_uint(3);
        let id = self.module.fresh_id();
        self.new_globals.push(Instruction::new(
            Op::SpecConstantComposite,
            Some(vector_ty),
            Some(id),
            ids.into_iter().map(Operand::IdRef).collect(),
        ));
        self.module.annotations.push(Instruction::new(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(id),
                Operand::Decoration(Decoration::BuiltIn),
                Operand::BuiltIn(BuiltIn::WorkgroupSize),
            ],
        ));
        self.kernel_workgroup_size_id = Some(id);
        id
    }

    /// OpConstantTrue/OpConstantFalse of an existing bool type.
    fn const_bool_of(&mut self, bool_ty: Word, value: bool) -> Word {
        let op = if value {
            Op::ConstantTrue
        } else {
            Op::ConstantFalse
        };
        self.get_or_create(op, Some(bool_ty), vec![])
    }

    /// Constant integer `v` of the given integer type `int_ty` (any width/signedness). Reuses an
    /// existing matching `OpConstant`. Used to synthesize `0`/`1` of a value's own int type for
    /// bool<->int convert lowerings.
    fn const_int_of(&mut self, int_ty: Word, v: i64) -> Word {
        let key = SynthCacheKey::ConstInt { int_ty, value: v };
        if let Some(&id) = self.synth_cache.get(&key) {
            return id;
        }
        let bits = self
            .module
            .types_global_values
            .iter()
            .chain(self.new_globals.iter())
            .find_map(|inst| {
                if inst.class.opcode == Op::TypeInt && inst.result_id == Some(int_ty) {
                    match inst.operands.first() {
                        Some(Operand::LiteralBit32(bits)) => Some(*bits),
                        _ => None,
                    }
                } else {
                    None
                }
            })
            .unwrap_or(32);
        let lit = if bits == 64 {
            Operand::LiteralBit64(v as u64)
        } else {
            Operand::LiteralBit32(v as u32)
        };
        for inst in self
            .module
            .types_global_values
            .iter()
            .chain(self.new_globals.iter())
        {
            if inst.class.opcode == Op::Constant
                && inst.result_type == Some(int_ty)
                && inst.operands.first() == Some(&lit)
            {
                if let Some(rid) = inst.result_id {
                    self.synth_cache.insert(key, rid);
                    return rid;
                }
            }
        }
        let id = self.module.fresh_id();
        self.new_globals.push(Instruction::new(
            Op::Constant,
            Some(int_ty),
            Some(id),
            vec![lit],
        ));
        self.synth_cache.insert(key, id);
        id
    }

    /// Constant float 0.0/1.0 etc.
    fn const_float(&mut self, bits: f32) -> Word {
        let key = SynthCacheKey::ConstFloat {
            bits: bits.to_bits(),
        };
        if let Some(&id) = self.synth_cache.get(&key) {
            return id;
        }
        let f = self.ty_float();
        let id = self.module.fresh_id();
        self.new_globals.push(type_inst(
            Op::Constant,
            id,
            vec![Operand::LiteralBit32(bits.to_bits())],
        ));
        // fix result_type (Constant needs it)
        let last = self.new_globals.last_mut().unwrap();
        last.result_type = Some(f);
        self.synth_cache.insert(key, id);
        id
    }

    /// Get or create OpTypeFloat 16 (`half`). Reuses an existing type and adds the `Float16`
    /// capability requirement implicitly (asserted in finalize).
    fn ty_half(&mut self) -> Word {
        self.get_or_create(Op::TypeFloat, None, vec![Operand::LiteralBit32(16)])
    }

    /// Constant `half` of value `v` (0.0/1.0 etc.), encoded as the IEEE-754 binary16 bit pattern.
    fn const_half(&mut self, v: f32) -> Word {
        let bits = f32_to_f16_bits(v);
        let key = SynthCacheKey::ConstHalf { bits };
        if let Some(&id) = self.synth_cache.get(&key) {
            return id;
        }
        let h = self.ty_half();
        let id = self.module.fresh_id();
        self.new_globals.push(Instruction::new(
            Op::Constant,
            Some(h),
            Some(id),
            vec![Operand::LiteralBit32(bits as u32)],
        ));
        self.synth_cache.insert(key, id);
        id
    }

    fn glsl(&mut self) -> Word {
        if let Some(id) = self.glsl_ext {
            return id;
        }
        // reuse if present
        for inst in &self.module.ext_inst_imports {
            if let Some(Operand::LiteralString(s)) = inst.operands.first() {
                if s == "GLSL.std.450" {
                    let id = inst.result_id.unwrap();
                    self.glsl_ext = Some(id);
                    return id;
                }
            }
        }
        let id = self.module.fresh_id();
        self.module.ext_inst_imports.push(Instruction::new(
            Op::ExtInstImport,
            None,
            Some(id),
            vec![Operand::LiteralString("GLSL.std.450".into())],
        ));
        self.glsl_ext = Some(id);
        id
    }
}

mod access;
mod agx_cluster;
mod air_calls;
mod emitted_inline;
mod finalize;
#[cfg(test)]
mod lowering_regression_tests;
mod module_cleanup;
pub(crate) use module_cleanup::{drop_dangling_debug, drop_unreferenced_global_variables};
mod prune;
mod resources;
mod spirv_cfg;
mod stage_input;
mod stage_output;
mod type_singletons;
mod value_queries;
mod workgroup;

use access::{
    compose_derived_access_chains, decorate_ptr_access_chain_base_strides,
    drop_overindexed_zero_tail, drop_writeonly_dead_local_array_stores,
    expose_nullable_memory_bases, guard_integer_division_by_zero, hoist_function_variables,
    lower_cross_member_subword_load, lower_cross_member_subword_store,
    lower_private_byte_aggregate_reinterpret, lower_private_low_byte_word_load,
    lower_private_memory_atomics, lower_scalar_i64_arithmetic_to_u32_halves,
    lower_subword_scalar_store, materialize_inlined_local_pointer_field_stores,
    narrow_access_chain_indices, neutralize_null_access_chains,
    neutralize_private_placeholder_access_chains, recover_inlined_local_dynamic_pointer_fields,
    recover_inlined_local_pointer_fields, recover_unique_local_pointer_field_loads,
    remap_dynamic_word_index_to_array_member, remap_dynamic_word_index_to_array_struct_field,
    remap_overflow_word_index_to_outer_member, remap_word_index_to_struct_member,
    remodel_workgroup_flatword_aggregate, remodel_workgroup_floatarray_atomic_as_uint,
    remodel_workgroup_single_field_struct_array, reroot_demoted_array_element_overindex,
    retype_demoted_copymemory_placeholder, retype_private_direct_memory_placeholders,
    rewrite_byte_buffer_chained_reinterpret, rewrite_chained_element_reinterpret,
    rewrite_dynamic_homogeneous_struct_index_load, rewrite_dynamic_struct_index_reinterpret,
    rewrite_dynamic_struct_index_subword_reinterpret,
    rewrite_dynamic_struct_index_vector_reinterpret,
    rewrite_dynamic_struct_index_wide_word_reinterpret, rewrite_exact_raw_byte_block_memory,
    rewrite_flat_scalar_ptr_access_through_vector_array, rewrite_raw_byte_pointer_direct_loads,
    rewrite_raw_byte_pointer_wide_loads, rewrite_raw_byte_pointer_wide_stores,
    rewrite_reinterpret_scalar_loads, rewrite_scalar_pointer_arithmetic_access_chains,
    rewrite_scalar_slot_array_overindex, rewrite_strided_descent_access_chains,
    rewrite_thread_local_aggregate_prefix_stores, split_workgroup_ptr_access_chain_descent,
};
use air_calls::lower_air_calls;
use emitted_inline::{
    compose_chained_access_chains, inline_selected_helpers, prune_unreferenced_functions,
};
use finalize::finalize;
use prune::prune_unreachable_blocks;
use resources::rewrites::{rewrite_affine_raw_word_loads, rewrite_exact_raw_word_loads};
use resources::*;
use stage_input::{
    build_stage_input, load_kernel_dispatch_component, materialize_kernel_dispatch_field,
};
use stage_output::rewrite_return;
use value_queries::*;
use workgroup::*;

/// OpName id -> string for every residual intrinsic function. Shared by the inline pass (to skip
/// calls that must be lowered, not inlined) and the lowering pass. Lives in the parent so both
/// submodules reach it without a sibling dependency.
fn air_names(module: &Module) -> HashMap<Word, String> {
    let mut m = HashMap::new();
    for inst in &module.debug_names {
        if inst.class.opcode == Op::Name {
            if let (Some(Operand::IdRef(id)), Some(Operand::LiteralString(s))) =
                (inst.operands.first(), inst.operands.get(1))
            {
                if s.starts_with("air.")
                    || s.starts_with("llvm.fabs.")
                    || s.starts_with("llvm.fmuladd.")
                    || is_agx2_matmad_symbol(s)
                    || s == "llvm.agx3.edgecheck"
                    || s == "llvm.agx3.yield"
                    || s == "llvm.agx3.igemm.v8i32.i64.i64.v8i32"
                    || s == "llvm.agx2.cluster.num"
                    || s.starts_with("llvm.agx3.load.with.emask.global.")
                    || s.starts_with("llvm.agx3.store.with.emask.global.")
                    || s.starts_with("llvm.bswap.")
                    || s.starts_with("llvm.maxnum.")
                    || s.starts_with("llvm.minnum.")
                    || s == "llvm.assume"
                {
                    m.insert(*id, s.clone());
                }
            }
        }
    }
    m
}

fn is_agx2_matmad_symbol(name: &str) -> bool {
    matches!(
        name,
        "llvm.agx2.f16matmad4x4.v2f16"
            | "llvm.agx2.f32matmad4x4.v2f32"
            | "llvm.agx2.f16matmad8x8.v2f16"
            | "llvm.agx2.f32matmad8x8.v2f32"
    )
}

/// Apply every bodied helper splice to the native emitter's completed typed SPIR-V graph before it
/// is handed to the final passes.
///
/// This producer-side invocation does not prune functions or run module-wide post-inline recovery.
/// Those cleanup operations retain their downstream phase in the final passes.
pub(crate) fn inline_all_emitted_helpers(
    mut module: Module,
    emit_sidecar: crate::emit_sidecar::EmitSidecar,
    entry_name: Option<&str>,
) -> Result<(Module, crate::emit_sidecar::EmitSidecar), String> {
    // Some emitter CFG paths retain later `OpLabel`s inside the first SPIR-V block until the first
    // serialize/load boundary. Reproduce the loader's block partition in memory so this invocation
    // and the former residual invocation saw the same graph.
    let (partitioned, changed) = partition_embedded_blocks(module);
    module = partitioned;
    // `stage` and transform options are not consulted by the inliner. Supplying the ordinary kernel
    // defaults constructs the same allocation/type caches the later residual Ctx starts with.
    let mut ctx = Ctx::with_options_and_sidecar(
        module,
        emit_sidecar,
        Stage::Kernel,
        TransformOptions::default(),
    );
    let entry_idx = find_entry_index(&ctx.module, entry_name)
        .ok_or_else(|| "no entry function with a body found before emitted inlining".to_string())?;
    let entry_id = ctx.module.functions[entry_idx]
        .def
        .as_ref()
        .and_then(|instruction| instruction.result_id);
    let selected_ids = ctx
        .module
        .functions
        .iter()
        .filter(|function| !function.blocks.is_empty())
        .filter_map(|function| {
            let id = function
                .def
                .as_ref()
                .and_then(|instruction| instruction.result_id)?;
            (Some(id) != entry_id).then_some(id)
        })
        .collect::<HashSet<_>>();
    if !changed || selected_ids.is_empty() {
        emitted_inline::complete_inlined_access_chain_descent(&mut ctx, entry_idx);
        ctx.module.types_global_values.append(&mut ctx.new_globals);
        return Ok((ctx.module, ctx.emit_sidecar));
    }
    inline_selected_helpers(&mut ctx, entry_idx, &selected_ids)?;
    crate::native::close_inlined_bda_pointer_tables_module(&mut ctx.module);
    emitted_inline::complete_inlined_access_chain_descent(&mut ctx, entry_idx);
    if let Some((function, block, terminator, instructions)) =
        first_detached_instruction(&ctx.module)
    {
        return Err(format!(
            "emitted inliner produced instructions after a terminator \
             (reason=emitted_inline_detached_instruction, function={function}, block={block}, \
             first_terminator={terminator}, instructions={instructions})"
        ));
    }
    ctx.module.types_global_values.append(&mut ctx.new_globals);
    Ok((ctx.module, ctx.emit_sidecar))
}

/// Complete the function-constant prune/interface transaction by lowering any aggregate-stride
/// Workgroup pointer chain exposed by its new final pointer graph.
pub(crate) fn lower_specialized_workgroup_ptr_access_chains(module: Module) -> Module {
    let mut ctx = Ctx::with_options_and_sidecar(
        module,
        crate::emit_sidecar::EmitSidecar::default(),
        Stage::Kernel,
        TransformOptions::default(),
    );
    split_workgroup_ptr_access_chain_descent(&mut ctx, 0);
    ctx.module
}

/// Reproduce SPIR-V loader block partitioning without crossing the words boundary.
///
/// Consumes and returns the module plus `false` without mutation when the instruction stream would
/// not parse: every embedded label must immediately follow a block terminator, and every resulting
/// block must end in one.
fn partition_embedded_blocks(mut module: Module) -> (Module, bool) {
    let is_terminator = |instruction: &Instruction| is_block_terminator(instruction.class.opcode);
    let can_partition = module.functions.iter().all(|function| {
        function.blocks.iter().all(|block| {
            let mut segment_has_terminator = false;
            for instruction in &block.instructions {
                if instruction.class.opcode == Op::Label {
                    if !segment_has_terminator {
                        return false;
                    }
                    segment_has_terminator = false;
                } else if segment_has_terminator {
                    return false;
                } else if is_terminator(instruction) {
                    segment_has_terminator = true;
                }
            }
            segment_has_terminator
        })
    });
    if !can_partition {
        return (module, false);
    }

    module.functions = module
        .functions
        .into_iter()
        .map(|mut function| {
            let mut blocks = Vec::new();
            for block in function.blocks {
                let mut current = Block {
                    label: block.label,
                    instructions: Vec::new(),
                };
                for instruction in block.instructions {
                    if instruction.class.opcode == Op::Label {
                        blocks.push(current);
                        current = Block {
                            label: Some(instruction),
                            instructions: Vec::new(),
                        };
                    } else {
                        current.instructions.push(instruction);
                    }
                }
                blocks.push(current);
            }
            function.blocks = blocks;
            function
        })
        .collect();
    (module, true)
}

fn first_detached_instruction(module: &Module) -> Option<(usize, usize, usize, usize)> {
    for (function_index, function) in module.functions.iter().enumerate() {
        for (block_index, block) in function.blocks.iter().enumerate() {
            let terminator = block.instructions.iter().position(|instruction| {
                matches!(
                    instruction.class.opcode,
                    Op::Branch
                        | Op::BranchConditional
                        | Op::Switch
                        | Op::Return
                        | Op::ReturnValue
                        | Op::Kill
                        | Op::Unreachable
                )
            });
            if let Some(terminator) =
                terminator.filter(|index| index + 1 != block.instructions.len())
            {
                return Some((
                    function_index,
                    block_index,
                    terminator,
                    block.instructions.len(),
                ));
            }
        }
    }
    None
}

/// Replace every use of id `from` with `to` across a function body (operands only). Shared by the
/// inline + interface passes.
fn replace_id_in_function(func: &mut Function, from: Word, to: Word) {
    for blk in &mut func.blocks {
        for inst in &mut blk.instructions {
            for op in &mut inst.operands {
                if let Operand::IdRef(r) = op {
                    if *r == from {
                        *r = to;
                    }
                }
            }
        }
    }
}

/// Top-level driver: take the native emitter's SPIR-V module and produce a Vulkan entry-pointed module.
/// `entry_name` is the AIR stage entry function name (from `!air.<stage>` metadata) used to pick the
/// right function among the emitted functions.
/// Renumber every result id into a deterministic canonical form: walk all instructions in serialized
/// order and assign new sequential ids on first appearance, then remap every id reference (result id,
/// result type, and the three id-bearing operand kinds). The serialized order is fully Vec-ordered
/// (deterministic), so this makes the assembled bytes reproducible regardless of the HashMap-order-
/// sensitive id allocation during emission. Pure renumbering — semantically identical SPIR-V.
pub(crate) fn canonicalize_ids(module: &mut Module) {
    canonicalize_ids_and_remap(module, &mut []);
}

/// Canonicalize module ids while keeping a small caller-owned set of typed sidecar ids aligned with
/// the same remap. The tracked ids are not SPIR-V roots by themselves; callers use them to preserve
/// semantic sidecar facts through later in-memory rewrites after debug-marker removal.
pub(crate) fn canonicalize_ids_and_remap(module: &mut Module, tracked_ids: &mut [Word]) {
    let _ = canonicalize_ids_and_collect_remap(module, tracked_ids);
}

pub(crate) fn canonicalize_ids_and_remap_sidecar(
    module: &mut Module,
    tracked_ids: &mut [Word],
    sidecar: &mut crate::emit_sidecar::EmitSidecar,
) {
    let remap = canonicalize_ids_and_collect_remap(module, tracked_ids);
    sidecar.remap_ids(&remap);
}

fn canonicalize_ids_and_collect_remap(
    module: &mut Module,
    tracked_ids: &mut [Word],
) -> HashMap<Word, Word> {
    let mut remap: HashMap<Word, Word> = HashMap::new();
    let mut next: Word = 1;
    for inst in module.all_inst_iter() {
        if let Some(result_id) = inst.result_id {
            remap.entry(result_id).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            });
        }
    }
    let map = |w: Word| remap.get(&w).copied().unwrap_or(w);
    for inst in module.all_inst_iter_mut() {
        if let Some(result_type) = inst.result_type.as_mut() {
            *result_type = map(*result_type);
        }
        if let Some(result_id) = inst.result_id.as_mut() {
            *result_id = map(*result_id);
        }
        for operand in &mut inst.operands {
            match operand {
                Operand::IdRef(w) | Operand::IdMemorySemantics(w) | Operand::IdScope(w) => {
                    *w = map(*w);
                }
                _ => {}
            }
        }
    }
    module.set_id_bound(next);
    for id in tracked_ids {
        *id = map(*id);
    }
    remap
}

#[cfg(test)]
mod canonicalize_tests {
    use super::*;
    use crate::spirv_module::ModuleHeader;

    #[test]
    fn canonicalize_remaps_typed_sidecar_ids_with_the_module() {
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(81));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeInt,
                None,
                Some(50),
                vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
            ),
            Instruction::new(Op::ConstantNull, Some(50), Some(80), vec![]),
        ];
        let mut tracked = [80];

        canonicalize_ids_and_remap(&mut module, &mut tracked);

        assert_eq!(tracked, [2]);
        assert_eq!(module.types_global_values[1].result_id, Some(2));
        assert_eq!(module.header.as_ref().map(|header| header.bound), Some(3));
    }
}

#[cfg(test)]
mod emitted_inline_tests {
    use super::*;
    use crate::spirv_module::ModuleHeader;
    use spirv::FunctionControl;

    fn function(id: Word, label: Word, instructions: Vec<Instruction>) -> Function {
        Function {
            def: Some(Instruction::new(
                Op::Function,
                Some(2),
                Some(id),
                vec![
                    Operand::FunctionControl(FunctionControl::NONE),
                    Operand::IdRef(3),
                ],
            )),
            end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
            parameters: Vec::new(),
            blocks: vec![Block {
                label: Some(Instruction::new(Op::Label, None, Some(label), vec![])),
                instructions,
            }],
        }
    }

    #[test]
    fn emitted_selection_skips_a_prepass_intermediate_the_old_inliner_cannot_observe() {
        let entry_instructions = vec![
            Instruction::new(Op::FunctionCall, None, None, vec![Operand::IdRef(20)]),
            Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(11)]),
            Instruction::new(Op::UConvert, Some(5), Some(12), vec![Operand::IdRef(6)]),
            Instruction::new(Op::Return, None, None, vec![]),
        ];
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(100));
        module.debug_names = vec![
            Instruction::new(
                Op::Name,
                None,
                None,
                vec![
                    Operand::IdRef(10),
                    Operand::LiteralString("main".to_string()),
                ],
            ),
            Instruction::new(
                Op::Name,
                None,
                None,
                vec![
                    Operand::IdRef(20),
                    Operand::LiteralString("helper".to_string()),
                ],
            ),
        ];
        module.functions = vec![
            function(10, 11, entry_instructions.clone()),
            function(
                20,
                21,
                vec![Instruction::new(Op::Return, None, None, vec![])],
            ),
        ];

        let (module, _) = inline_all_emitted_helpers(
            module,
            crate::emit_sidecar::EmitSidecar::default(),
            Some("main"),
        )
        .expect("invalid intermediate follows the historical retry route");

        assert_eq!(
            module.functions[0].blocks[0].instructions,
            entry_instructions
        );
        assert_eq!(
            module.functions.len(),
            2,
            "the selected helper remains untouched"
        );
    }

    #[test]
    fn emitted_closure_partitions_embedded_labels_like_the_loader() {
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(100));
        module.debug_names = vec![Instruction::new(
            Op::Name,
            None,
            None,
            vec![
                Operand::IdRef(10),
                Operand::LiteralString("main".to_string()),
            ],
        )];
        module.functions = vec![
            function(
                10,
                11,
                vec![
                    Instruction::new(Op::FunctionCall, None, None, vec![Operand::IdRef(20)]),
                    Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(12)]),
                    Instruction::new(Op::Label, None, Some(12), vec![]),
                    Instruction::new(Op::Return, None, None, vec![]),
                ],
            ),
            function(
                20,
                21,
                vec![Instruction::new(Op::Return, None, None, vec![])],
            ),
        ];

        let (module, _) = inline_all_emitted_helpers(
            module,
            crate::emit_sidecar::EmitSidecar::default(),
            Some("main"),
        )
        .expect("complete emitted closure");

        assert_eq!(module.functions[0].blocks.len(), 2);
        assert_eq!(
            module.functions[0].blocks[1]
                .label
                .as_ref()
                .and_then(|label| label.result_id),
            Some(12)
        );
        assert!(
            module.functions[0]
                .blocks
                .iter()
                .flat_map(|block| &block.instructions)
                .all(|instruction| instruction.class.opcode != Op::FunctionCall),
            "the helper is spliced after reproducing the serialized block partition"
        );
    }

    #[test]
    fn emitted_closure_selects_bodied_function_ids_without_debug_names() {
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(100));
        module.debug_names = vec![Instruction::new(
            Op::Name,
            None,
            None,
            vec![
                Operand::IdRef(10),
                Operand::LiteralString("main".to_string()),
            ],
        )];
        module.functions = vec![
            function(
                10,
                11,
                vec![
                    Instruction::new(Op::FunctionCall, None, None, vec![Operand::IdRef(20)]),
                    Instruction::new(Op::Return, None, None, vec![]),
                ],
            ),
            function(
                20,
                21,
                vec![Instruction::new(Op::Return, None, None, vec![])],
            ),
        ];

        let (module, _) = inline_all_emitted_helpers(
            module,
            crate::emit_sidecar::EmitSidecar::default(),
            Some("main"),
        )
        .expect("complete emitted closure");

        assert!(
            module.functions[0].blocks[0]
                .instructions
                .iter()
                .all(|instruction| instruction.class.opcode != Op::FunctionCall),
            "all bodied function ids are selected independently of OpName"
        );
    }
}

#[cfg(test)]
pub(crate) fn transform(
    module: Module,
    stage: Stage,
    frag: Option<&FragMeta>,
    vert: Option<&VertMeta>,
    kern: Option<&KernMeta>,
    entry_name: Option<&str>,
) -> Result<Module, String> {
    transform_with_options(
        module,
        stage,
        frag,
        vert,
        kern,
        entry_name,
        TransformOptions {
            kernel_dispatch: matches!(stage, Stage::Kernel)
                .then_some(crate::reflect::KernelDispatch::Workgroups),
            ..TransformOptions::default()
        },
    )
}

#[cfg(test)]
pub(crate) fn transform_with_options(
    module: Module,
    stage: Stage,
    frag: Option<&FragMeta>,
    vert: Option<&VertMeta>,
    kern: Option<&KernMeta>,
    entry_name: Option<&str>,
    options: TransformOptions,
) -> Result<Module, String> {
    transform_with_options_and_sidecar(
        module,
        crate::emit_sidecar::EmitSidecar::default(),
        stage,
        frag,
        vert,
        kern,
        entry_name,
        options,
    )
    .map(
        |Transformed {
             module, sidecar, ..
         }| {
            let mut sidecar = sidecar;
            // This test-only byte boundary intentionally discards the sidecar. Drop its source-layout
            // oracle roots too so direct transform tests observe the same serialized type cleanup as a
            // completed product translation after main-pipeline exact-access replay.
            sidecar.buffer_root_source_types.clear();
            let mut ctx = Ctx::with_options_and_sidecar(module, sidecar, stage, options);
            module_cleanup::gc_dead_globals(&mut ctx);
            ctx.module
        },
    )
}

pub(crate) fn validate_descriptor_bindings(
    module: &Module,
    layout: crate::reflect::DescriptorLayout,
) -> Result<(), String> {
    resources::validate_descriptor_binding_classes(module, layout)
}

#[cfg(test)]
mod phase_contract_tests {
    use super::*;
    use crate::spirv_module::ModuleHeader;

    /// A module whose only pointer type is still staged in `new_globals`.
    ///
    /// Every phase between the interface pass and the splice looks like this, so if the verdict were
    /// taken from `ctx.module` alone it would read "references undefined id" for all of them and the
    /// trace would say nothing about where a real violation started.
    fn ctx_with_staged_pointer_type() -> Ctx {
        let mut module = Module::new();
        // `ModuleHeader::new` defaults to the grammar's 1.6; the owned contract is written against
        // the Vulkan 1.2 target environment, which caps SPIR-V at 1.5.
        let mut header = ModuleHeader::new(10);
        header.set_version(1, 5);
        module.header = Some(header);
        // `Linkage` rather than an entry point: the owned contract accepts either, and this fixture
        // exists to exercise the staged-globals merge, not to be a runnable shader.
        module.capabilities = [spirv::Capability::Shader, spirv::Capability::Linkage]
            .map(|capability| {
                Instruction::new(
                    Op::Capability,
                    None,
                    None,
                    vec![Operand::Capability(capability)],
                )
            })
            .to_vec();
        module.memory_model = Some(Instruction::new(
            Op::MemoryModel,
            None,
            None,
            vec![
                Operand::AddressingModel(spirv::AddressingModel::Logical),
                Operand::MemoryModel(spirv::MemoryModel::GLSL450),
            ],
        ));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            // Uses %2, which only exists in `new_globals` below.
            Instruction::new(
                Op::Variable,
                Some(2),
                Some(3),
                vec![Operand::StorageClass(StorageClass::Private)],
            ),
        ];
        let mut ctx = Ctx::new(module);
        ctx.new_globals.push(Instruction::new(
            Op::TypePointer,
            None,
            Some(2),
            vec![
                Operand::StorageClass(StorageClass::Private),
                Operand::IdRef(1),
            ],
        ));
        ctx
    }

    #[test]
    fn a_staged_global_counts_as_declared() {
        assert_eq!(
            phase_contract_verdict(&ctx_with_staged_pointer_type()),
            Vec::<String>::new()
        );
    }

    /// The same context with the staging buffer emptied: now %2 really is undefined, and the verdict
    /// must both fire and name the id, which is the whole reason the trace is worth reading.
    #[test]
    fn a_genuinely_missing_global_is_reported_by_id() {
        let mut ctx = ctx_with_staged_pointer_type();
        ctx.new_globals.clear();
        let verdicts = phase_contract_verdict(&ctx);
        assert!(
            verdicts
                .iter()
                .any(|verdict| verdict.contains("references undefined id %2")),
            "verdict must name the missing id, got: {verdicts:?}"
        );
    }

    /// Two independent violations at once: a debug name pointing at an id nothing defines, and a
    /// variable whose result type is another variable rather than a type declaration.
    ///
    /// Lowering leaves dangling debug names behind constantly and repairs them later, so a trace
    /// that stopped at the first failure would report only the name for every phase the name
    /// survived -- and the reader would blame whichever phase happens to run after the repair.
    #[test]
    fn a_transient_debug_name_does_not_hide_the_violation_under_it() {
        let mut ctx = ctx_with_staged_pointer_type();
        ctx.module.debug_names.push(Instruction::new(
            Op::Name,
            None,
            None,
            vec![
                Operand::IdRef(9),
                Operand::LiteralString("erased".to_string()),
            ],
        ));
        // Result type %3 is the Private variable declared above, not a type.
        ctx.module.types_global_values.push(Instruction::new(
            Op::Variable,
            Some(3),
            Some(4),
            vec![Operand::StorageClass(StorageClass::Private)],
        ));
        let verdicts = phase_contract_verdict(&ctx);
        assert!(
            verdicts
                .iter()
                .any(|verdict| verdict.contains("references undefined id %9")),
            "the dangling debug name must be reported, got: {verdicts:?}"
        );
        assert!(
            verdicts
                .iter()
                .any(|verdict| verdict.contains("result type is not a type declaration")),
            "the violation under it must be reported too, got: {verdicts:?}"
        );
    }
}

/// Every owned-construction verdict against the module a phase just produced, empty when it holds.
///
/// Phases stage new type/constant declarations in `new_globals` and only splice them into the module
/// at the end. Checking the module alone would therefore report an undefined id for every phase in
/// between, which drowns out the verdict this exists to show. Merging the staged globals in
/// reproduces the module the pipeline is actually building.
///
/// All of them, not just the first, because the first is routinely the least interesting: an
/// in-flight module often carries a transient violation a later phase repairs -- a debug `OpName`
/// left pointing at an id a rewrite replaced is the common one -- and the gate's first-failure
/// answer would report only that for every phase it survives, hiding the structural violation the
/// reader is trying to locate.
fn phase_contract_verdict(ctx: &Ctx) -> Vec<String> {
    let mut module = ctx.module.clone();
    module
        .types_global_values
        .extend(ctx.new_globals.iter().cloned());
    crate::native::owned_module_failures(&module, usize::MAX)
        .into_iter()
        .map(|failure| {
            let (kind, error) = match failure {
                crate::native::OwnedModuleFailure::Invalid(error) => ("invalid", error),
                crate::native::OwnedModuleFailure::TypeConstruction(error) => {
                    ("type-construction", error)
                }
                crate::native::OwnedModuleFailure::CfgConstruction(error) => {
                    ("cfg-construction", error)
                }
                crate::native::OwnedModuleFailure::RawBufferConstruction(error) => {
                    ("raw-buffer-construction", error)
                }
            };
            format!("{kind}: {error}")
        })
        .collect()
}

/// Print whether the in-flight module satisfies the owned construction contract, tagged with the
/// phase that just finished.
///
/// This is a debugging aid, not a gate. A module part-way through lowering is *allowed* to violate
/// the contract -- several phases exist precisely to repair what an earlier one left behind -- so a
/// single bad verdict means nothing on its own. What the trace gives you is the shape of the run:
/// the phase where a violation first appears and is never repaired is the one that introduced it,
/// and the message is the one the finished module would have failed with. Without it, a contract
/// failure reported at the end says nothing about which phase produced it.
fn report_phase_contract(ctx: &Ctx, phase: &str) {
    let verdicts = phase_contract_verdict(ctx);
    if verdicts.is_empty() {
        eprintln!("[pass-contract] {phase}: ok");
    }
    for verdict in verdicts {
        eprintln!("[pass-contract] {phase}: {verdict}");
    }
}

/// What `transform_with_options_and_sidecar` produced: the finished module, the emit-seam sidecar
/// it consumed, and the descriptor facts that only the passes are in a position to know.
pub(crate) struct Transformed {
    pub(crate) module: Module,
    pub(crate) sidecar: crate::emit_sidecar::EmitSidecar,
    /// Bindings of descriptors the passes synthesized with no Metal argument behind them, still
    /// present at this boundary. Reflection reports these; nothing else describes them.
    pub(crate) placeholder_descriptor_bindings: Vec<u32>,
}

pub(crate) fn transform_with_options_and_sidecar(
    module: Module,
    emit_sidecar: crate::emit_sidecar::EmitSidecar,
    stage: Stage,
    frag: Option<&FragMeta>,
    vert: Option<&VertMeta>,
    kern: Option<&KernMeta>,
    entry_name: Option<&str>,
    options: TransformOptions,
) -> Result<Transformed, String> {
    if matches!(stage, Stage::Kernel) && options.kernel_local_size.contains(&0) {
        return Err("kernel LocalSize dimensions must be non-zero".to_string());
    }
    if let Some(dispatch) = options.kernel_dispatch {
        dispatch.validate()?;
    }
    if !matches!(stage, Stage::Kernel) && options.kernel_dispatch.is_some() {
        return Err("kernel dispatch bounds are only valid for kernel stages".to_string());
    }
    let mut ctx = Ctx::with_options_and_sidecar(module, emit_sidecar, stage, options);
    let retry_debug = crate::env_vars::retry_debug();
    let pass_contract = crate::env_vars::pass_contract();
    // A macro rather than a closure so it can read `ctx.module` at the point it is written: a closure
    // capturing `&ctx` would hold a borrow across the whole pipeline, which is exactly where the
    // phases need `&mut ctx`.
    macro_rules! debug_phase {
        ($phase:expr) => {
            if retry_debug {
                eprintln!("[retry-debug] passes: {}", $phase);
            }
            if pass_contract {
                report_phase_contract(&ctx, $phase);
            }
        };
    }
    let mut entry_idx = find_entry_index(&ctx.module, entry_name)
        .ok_or_else(|| "no entry function with a body found".to_string())?;
    debug_phase!("cleanup start");
    // Producer-side closure already made the entry self-contained. Preserve the former residual
    // inliner's two post-splice cleanup operations at this phase.
    compose_chained_access_chains(&mut ctx, entry_idx);
    prune_unreferenced_functions(&mut ctx, entry_idx);
    entry_idx = find_entry_index(&ctx.module, entry_name)
        .ok_or_else(|| "entry vanished after helper cleanup".to_string())?;
    recover_inlined_local_pointer_fields(&mut ctx, entry_idx);
    compose_derived_access_chains(&mut ctx, entry_idx);
    agx_cluster::lower_agx2_cluster_numbers(&mut ctx, entry_idx, &stage)?;
    neutralize_null_access_chains(&mut ctx, entry_idx);

    debug_phase!("interface start");
    // 1a) decoded params -> stage-input/resource vars and entry-body replacements. Preserve the
    //     original type snapshot for the immediately-following return rewrite.
    let input_defs = build_stage_input(&mut ctx, entry_idx, &stage, frag, vert, kern)?;
    ctx.validate_runtime_storage_image_bindings()?;
    materialize_inlined_local_pointer_field_stores(&mut ctx, entry_idx);
    // 1b) return -> output vars.
    rewrite_return(&mut ctx, entry_idx, &stage, frag, vert, &input_defs)?;
    // Turn each runtime-indexed texture-array element handle load into an OpAccessChain+OpLoad
    // into the descriptor array declared by the interface `ImageArray` binding, so the sample/query
    // lowering below sees an ordinary loaded image (no-op unless an ImageArray binding exists). Runs
    // BEFORE `recover_inlined_local_pointer_fields`, which otherwise rewrites the per-element handle
    // load into a local-pointer-field marker and severs the access chain this pass keys on.
    resources::materialize_texture_array_loads(&mut ctx, entry_idx);
    hoist_function_variables(&mut ctx, entry_idx);
    // Entry texture parameters become loaded image ids only after interface binding. Replay any
    // helper-field markers that still reference those parameters before image calls lower.
    recover_inlined_local_pointer_fields(&mut ctx, entry_idx);
    // The replay above can reveal that a helper-local runtime table is populated by fixed elements
    // of one descriptor array. Re-run the materializer so it can replace that table's pointer select
    // with a descriptor-array access using the exact recorded store mapping.
    resources::materialize_texture_array_loads(&mut ctx, entry_idx);
    // Resource binding must get first refusal on pointer-shaped texture handles. Only after exact
    // single-image and descriptor-array roots have been materialized may remaining private
    // placeholders be neutralized as genuinely unmodeled memory.
    neutralize_private_placeholder_access_chains(&mut ctx, entry_idx)?;
    propagate_sampler_state_aliases(&mut ctx, entry_idx);

    debug_phase!("air lowering start");
    // 2) lower residual air.* calls inside the entry function.
    ctx.phase_value_types = Some(function_value_types(&ctx, entry_idx));
    let air_lowering = lower_air_calls(&mut ctx, entry_idx);
    ctx.phase_value_types = None;
    air_lowering?;
    // AIR calls can materialize opaque handles after interface binding's earlier resource-wrapper
    // collapse (notably `air.get_null_texture_*`). Re-establish the same aggregate invariant for
    // those late values before any subsequent pass observes their former pointer-shaped fields.
    resources::collapse_late_pointer_and_opaque_wrappers(&mut ctx, entry_idx)?;
    recover_unique_local_pointer_field_loads(&mut ctx, entry_idx);
    recover_inlined_local_dynamic_pointer_fields(&mut ctx, entry_idx)?;
    lower_private_memory_atomics(&mut ctx, entry_idx);

    // 2c) define otherwise-undefined integer divide-by-zero side arms before drivers can materialize
    //     divergent values from guarded select expressions.
    guard_integer_division_by_zero(&mut ctx, entry_idx);

    debug_phase!("memory lowering start");
    ctx.phase_type_positions = Some(
        ctx.new_globals
            .iter()
            .enumerate()
            .filter_map(|(index, instruction)| Some((instruction.result_id?, (true, index))))
            .chain(
                ctx.module
                    .types_global_values
                    .iter()
                    .enumerate()
                    .filter_map(|(index, instruction)| {
                        Some((instruction.result_id?, (false, index)))
                    }),
            )
            .collect(),
    );
    // 2d) narrow 64-bit access-chain INDEX operands to 32-bit. NVIDIA's SPIR-V->NVVM compiler crashes
    //     when an access-chain index is a 64-bit (`%ulong`) value (a driver-fragility class), so we
    //     value-preservingly rewrite each i64 index to a 32-bit equivalent. After this the i64 index
    //     constants/types are dead and `finalize` can drop the now-unused `OpCapability Int64`.
    for function_idx in 0..ctx.module.functions.len() {
        narrow_access_chain_indices(&mut ctx, function_idx);
    }
    for function_idx in 0..ctx.module.functions.len() {
        expose_nullable_memory_bases(&mut ctx, function_idx);
    }
    compose_derived_access_chains(&mut ctx, entry_idx);
    rewrite_scalar_pointer_arithmetic_access_chains(&mut ctx, entry_idx);
    for function_idx in 0..ctx.module.functions.len() {
        drop_overindexed_zero_tail(&mut ctx, function_idx);
    }
    for function_idx in 0..ctx.module.functions.len() {
        lower_private_low_byte_word_load(&mut ctx, function_idx);
    }
    reroot_demoted_array_element_overindex(&mut ctx, entry_idx);
    remap_word_index_to_struct_member(&mut ctx, entry_idx);
    remap_overflow_word_index_to_outer_member(&mut ctx, entry_idx);
    for function_idx in 0..ctx.module.functions.len() {
        remap_dynamic_word_index_to_array_member(&mut ctx, function_idx);
    }
    for function_idx in 0..ctx.module.functions.len() {
        remap_dynamic_word_index_to_array_struct_field(&mut ctx, function_idx);
    }
    drop_writeonly_dead_local_array_stores(&mut ctx, entry_idx);
    lower_cross_member_subword_load(&mut ctx, entry_idx)?;
    lower_cross_member_subword_store(&mut ctx, entry_idx);
    lower_subword_scalar_store(&mut ctx, entry_idx);
    for function_idx in 0..ctx.module.functions.len() {
        rewrite_strided_descent_access_chains(&mut ctx, function_idx);
    }
    for function_idx in 0..ctx.module.functions.len() {
        rewrite_dynamic_struct_index_reinterpret(&mut ctx, function_idx)?;
    }
    for function_idx in 0..ctx.module.functions.len() {
        rewrite_dynamic_struct_index_subword_reinterpret(&mut ctx, function_idx)?;
    }
    for function_idx in 0..ctx.module.functions.len() {
        rewrite_dynamic_struct_index_wide_word_reinterpret(&mut ctx, function_idx)?;
    }
    for function_idx in 0..ctx.module.functions.len() {
        rewrite_dynamic_struct_index_vector_reinterpret(&mut ctx, function_idx)?;
    }
    for function_idx in 0..ctx.module.functions.len() {
        rewrite_dynamic_homogeneous_struct_index_load(&mut ctx, function_idx)?;
    }
    for function_idx in 0..ctx.module.functions.len() {
        rewrite_chained_element_reinterpret(&mut ctx, function_idx)?;
    }
    for function_idx in 0..ctx.module.functions.len() {
        rewrite_byte_buffer_chained_reinterpret(&mut ctx, function_idx)?;
    }
    // The raw-byte replay introduces PtrAccessChain byte pointers.  Decorate their common uchar
    // base type immediately below, together with every pre-existing PtrAccessChain base.
    for function_idx in 0..ctx.module.functions.len() {
        rewrite_raw_byte_pointer_direct_loads(&mut ctx, function_idx);
    }
    for function_idx in 0..ctx.module.functions.len() {
        rewrite_raw_byte_pointer_wide_loads(&mut ctx, function_idx);
    }
    for function_idx in 0..ctx.module.functions.len() {
        rewrite_raw_byte_pointer_wide_stores(&mut ctx, function_idx);
    }
    decorate_ptr_access_chain_base_strides(&mut ctx);
    // Runs AFTER the stride decoration so the byte-buffer (PtrAccessChain) widen can read the base
    // pointer's ArrayStride to prove slot contiguity.
    rewrite_reinterpret_scalar_loads(&mut ctx, entry_idx);
    rewrite_scalar_slot_array_overindex(&mut ctx, entry_idx)?;
    remodel_workgroup_flatword_aggregate(&mut ctx, entry_idx);
    remodel_workgroup_single_field_struct_array(&mut ctx, entry_idx);
    remodel_workgroup_floatarray_atomic_as_uint(&mut ctx, entry_idx)?;
    lower_private_byte_aggregate_reinterpret(&mut ctx, entry_idx)?;
    retype_private_direct_memory_placeholders(&mut ctx);
    retype_demoted_copymemory_placeholder(&mut ctx, entry_idx);
    for function_idx in 0..ctx.module.functions.len() {
        rewrite_thread_local_aggregate_prefix_stores(&mut ctx, function_idx);
    }
    // Memory/interface rewrites above can expose a concrete pointer only after the first late
    // wrapper pass ran. Re-close the no-pointer-carrier invariant before finalization so an exact
    // insert/extract round trip cannot retain the aggregate field's stale opaque-pointer type.
    resources::collapse_late_pointer_and_opaque_wrappers(&mut ctx, entry_idx)?;
    // The newly concrete root can be a canonical raw-byte buffer block. Replay wide loads that were
    // formerly hidden behind the stale private pointer, then decorate the PtrAccessChains introduced
    // by that replay just as in the primary raw-memory phase above.
    for function_idx in 0..ctx.module.functions.len() {
        rewrite_exact_raw_byte_block_memory(&mut ctx, function_idx);
    }
    for function_idx in 0..ctx.module.functions.len() {
        rewrite_raw_byte_pointer_wide_loads(&mut ctx, function_idx);
    }
    rewrite_flat_scalar_ptr_access_through_vector_array(&mut ctx, entry_idx);
    for function_idx in 0..ctx.module.functions.len() {
        rewrite_exact_raw_word_loads(&mut ctx, function_idx);
    }
    for function_idx in 0..ctx.module.functions.len() {
        rewrite_affine_raw_word_loads(&mut ctx, function_idx);
    }
    let bound_buffer_vars = ctx.bound_buffer_vars.clone();
    resources::retire_dead_pointer_projections(
        &mut ctx,
        entry_idx,
        bound_buffer_vars.iter().copied(),
    );
    decorate_ptr_access_chain_base_strides(&mut ctx);
    // Every consumer of the discarded aggregate layout has now replayed its exact byte address.
    // Release the source-only type roots so final global GC can remove that obsolete type graph.
    ctx.emit_sidecar.buffer_root_source_types.clear();
    // Late raw-memory rewrites can introduce new access chains rooted in the same unnamed,
    // null-initialized Private placeholders that represent absent optional resources. Reapply the
    // structural placeholder closure after every pointer-producing memory pass has run.
    neutralize_private_placeholder_access_chains(&mut ctx, entry_idx)?;
    // The memory phase now owns the final caller/helper access paths. Publish its synthesized type
    // dependencies and close every packed Private-vector word view before later phases can observe
    // an unrepresentable Logical access chain. A late pointer phi retains the established
    // address-domain construction, while general select fallback waits for final module construction.
    ctx.phase_type_positions = None;
    ctx.module.types_global_values.append(&mut ctx.new_globals);
    // This is the last memory phase that can expose Workgroup pointer uses. Construct the complete
    // module-wide float-as-int atomic storage graph here, including nested pointee trees and retained
    // helper functions, so final assembly never needs a generic atomic repair sweep.
    // Close liveness first: structurization can leave pure pointer phis and their access chains after
    // every observable consumer was removed. Those dead references are not part of the storage
    // contract and must not make the constructor reject an otherwise-complete live use graph.
    let preserved_pointer_facts = ctx
        .emit_sidecar
        .local_pointer_field_stores
        .iter()
        .map(|fact| fact.id)
        .collect::<HashSet<_>>();
    crate::native::eliminate_dead_pointer_values_module(&mut ctx.module, &preserved_pointer_facts);
    crate::native::construct_workgroup_atomic_floats_module(&mut ctx.module);
    crate::native::close_private_vector_word_views_module(&mut ctx.module);
    if let Some(address_table) =
        crate::native::construct_interface_cross_binding_pointer_phis_module(
            &mut ctx.module,
            ctx.descriptor_layout,
        )
    {
        ctx.interface_buffer_var(address_table);
    }
    // Cross-binding pointer construction is the final memory transform that can introduce a
    // PtrAccessChain after the primary storage/stride closure. Reapply that opcode's base-storage
    // invariant to the newly constructed address-domain graph before finalization.
    decorate_ptr_access_chain_base_strides(&mut ctx);

    debug_phase!("integer lowering start");
    // 2i) Some conformant Vulkan devices expose `shaderInt64` but execute scalar 64-bit arithmetic
    //     via a fragile native path. Rebuild add/sub/mul from 32-bit/16-bit pieces; this is the same
    //     modular result as the original `ulong` arithmetic and is structural over integer width.
    lower_scalar_i64_arithmetic_to_u32_halves(&mut ctx);

    debug_phase!("workgroup/finalize start");
    // 2g) deterministic threadgroup memory: zero-fill every Workgroup variable at kernel entry
    //     (stores of OpConstantNull + one control barrier), the candidate half of the harness's
    //     defined refinement of Metal's undefined threadgroup contents. Kernel-only: Workgroup
    //     variables exist only in compute, and the barrier is only legal there.
    if matches!(stage, Stage::Kernel) {
        workgroup::unroll_small_workgroup_atomic_loops(&mut ctx, entry_idx);
        workgroup::zero_initialize_workgroup_memory(&mut ctx, entry_idx);
    }
    // 2f) Drop blocks made unreachable by typed pruning and specialization. spirv-val tolerates
    //     them; SPIRV-Cross (MoltenVK) throws on them at pipeline creation. See `prune.rs`.
    prune_unreachable_blocks(&mut ctx.module);

    resources::sink_loop_header_texture_array_loads(&mut ctx, entry_idx);
    // Every local-pointer marker consumer has run. These facts exist only across the emitter/pass
    // seam; keeping their sentinel ids rooted through final global collection can serialize a dead
    // pointer-typed null after later type reconstruction. Retire the consumed facts before final
    // liveness is computed.
    ctx.emit_sidecar.local_pointer_field_stores.clear();
    ctx.emit_sidecar.aggregate_pointer_values.clear();
    // 3) finalize: append synthesized globals, drop dead air.* decls, add entry point + exec modes,
    //    bump the bound.
    finalize(&mut ctx, entry_idx, &stage, vert)?;
    // Interface finalization can replace an optional scalar parameter with a Private scalar slot
    // while retaining LLVM's zero-only GEP descent. Close those identity chains on the complete
    // interface graph before the owned type check.
    for function_idx in 0..ctx.module.functions.len() {
        drop_overindexed_zero_tail(&mut ctx, function_idx);
    }
    for function_idx in 0..ctx.module.functions.len() {
        lower_private_low_byte_word_load(&mut ctx, function_idx);
    }
    // Finalization and the interface closures above can synthesize access chains after the primary
    // memory phase. Re-establish the portable index-width invariant on every retained function at
    // the last pointer-producing boundary.
    for function_idx in 0..ctx.module.functions.len() {
        narrow_access_chain_indices(&mut ctx, function_idx);
    }
    // Run on the complete type graph: retained helper functions can carry a Workgroup aggregate
    // stride that entry-focused lowering could not see until all synthesized globals were appended.
    split_workgroup_ptr_access_chain_descent(&mut ctx, entry_idx);
    // Finalization and the last Workgroup split are pointer-producing boundaries. Seal
    // PtrAccessChain storage against the complete value graph, then publish any pointer types that
    // sealing synthesized before global liveness runs.
    decorate_ptr_access_chain_base_strides(&mut ctx);
    ctx.module.types_global_values.append(&mut ctx.new_globals);
    // `air.get_read_sampler()` and `air.get_null_texture_*()` each needed a real descriptor to type
    // their result; where nothing ever consumed that value, retract the synthesis before liveness
    // runs so the descriptor goes too.
    stage_input::drop_unconsumed_placeholder_descriptor_loads(&mut ctx);
    // The closures above are the last of this boundary's instruction changes, and any of them can
    // delete a variable's last use. Re-establish global liveness against the finished bodies before
    // the collection that would otherwise root a stranded variable at its own interface entry.
    module_cleanup::drop_unreferenced_global_variables(&mut ctx.module);
    module_cleanup::gc_dead_globals(&mut ctx);
    debug_phase!("complete");

    let placeholder_descriptor_bindings = surviving_placeholder_bindings(&ctx);
    Ok(Transformed {
        module: ctx.module,
        sidecar: ctx.emit_sidecar,
        placeholder_descriptor_bindings,
    })
}

/// The bindings of the synthesized placeholder descriptors that still exist in the finished module.
///
/// A placeholder whose loads were retracted has already lost its variable to
/// `drop_unreferenced_global_variables`; asking the module which ones remain is what keeps this
/// list from claiming a descriptor that is no longer there.
fn surviving_placeholder_bindings(ctx: &Ctx) -> Vec<u32> {
    let mut bindings = ctx
        .module
        .types_global_values
        .iter()
        .filter(|instruction| instruction.class.opcode == Op::Variable)
        .filter_map(|instruction| instruction.result_id)
        .filter_map(|variable| ctx.placeholder_descriptor_vars.get(&variable).copied())
        .collect::<Vec<_>>();
    bindings.sort_unstable();
    bindings.dedup();
    bindings
}
