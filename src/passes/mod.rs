//! Retained-SPIR-V transformation pipeline. The native emitter (`native/`, with its primary
//! structured-control-flow planning in `native/cfg/`) produces crate-owned Logical GLSL450 SPIR-V.
//! These passes then close both the Vulkan interface and residual AIR semantics. They:
//!   1. turn entry parameters into Vulkan interface variables by their AIR role
//!      (varying -> Input@Location, texture -> UniformConstant image, sampler -> UniformConstant
//!      sampler, buffer -> StorageBuffer Block@set/binding);
//!   2. turn the entry's return value into Output variable(s) @Location (MRT = struct split);
//!   3. lower the residual `air.*` OpFunctionCalls (sample -> OpImageSample*, math -> GLSL.std.450,
//!      dfdx/dfdy -> OpDPdx/OpDPdy, discard -> OpKill, ...);
//!   4. normalize typed access, Workgroup memory, and the narrow CFG shapes that require retained
//!      module repair; and
//!   5. synthesize OpEntryPoint + OpExecutionMode, close capabilities, and remove dead declarations.
//!
//! Everything operates on the crate-owned module representation (`crate::spirv_module::Module`),
//! whose instructions, operands, functions, and blocks are crate-owned nodes.

use crate::meta::{
    AirScalar, AirType, FragMeta, FragRole, KernMeta, KernRole, VertMeta, VertOutRole, VertRole,
};
use crate::reflect::{RuntimeSamplerState, StaticSamplerState, SAMPLER_ARGUMENT_COUNT_USIZE};
use crate::spirv_module::{is_block_terminator, Block, Function, Instruction, Module, Operand};
use spirv::{
    BuiltIn, Decoration, Dim, FunctionControl, ImageFormat, MemorySemantics, Op, Scope,
    StorageClass, Word,
};
use std::collections::{HashMap, HashSet};

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
    pub kernel_local_size: [u32; 3],
    /// Exact Metal `[[threads_per_grid]]` value when it is not derivable as
    /// Vulkan NumWorkgroups * LocalSize (non-uniform `dispatchThreads` tail regions).
    pub kernel_threads_per_grid: Option<[u32; 3]>,
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
}

impl Default for TransformOptions {
    fn default() -> Self {
        Self {
            kernel_local_size: [64, 1, 1],
            kernel_threads_per_grid: None,
            simd_cluster32: false,
            denorm_flush_to_zero_f32: false,
            raster_sample_count: None,
            runtime_sampler_states: [None; SAMPLER_ARGUMENT_COUNT_USIZE],
        }
    }
}

impl TransformOptions {
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
    /// loaded-image ids synthesized by `air.get_null_texture_*()`.
    null_image_values: HashSet<Word>,
    /// composite type ids already given explicit Offset/ArrayStride layout (dedup; a type decorated
    /// twice is a validation error).
    laid_out: HashSet<Word>,
    /// Struct type ids reconstructed from `air.struct_type_info`, with their exact AIR member offsets.
    air_struct_offsets: HashMap<Word, Vec<u32>>,
    air_data_layout: Option<crate::layout::AirDataLayout>,
    /// GLCompute LocalSize and the value exposed to AIR `[[threads_per_threadgroup]]`.
    kernel_local_size: [u32; 3],
    /// Optional exact value exposed to AIR `[[threads_per_grid]]`.
    kernel_threads_per_grid: Option<[u32; 3]>,
    /// M-D2 simd-reduce clustering opt-in (see [`TransformOptions::simd_cluster32`]).
    simd_cluster32: bool,
    /// Exact graphics-pipeline sample count used to lower `air.get_num_samples.i32`.
    raster_sample_count: Option<u32>,
    runtime_sampler_states: [Option<RuntimeSamplerState>; SAMPLER_ARGUMENT_COUNT_USIZE],
    /// lazily-created default sampler variable id, for `air.get_read_sampler()` (a sampler-less
    /// `texture.read` still passes a sampler operand AIR-side; we synthesize one valid sampler).
    default_sampler_var: Option<Word>,
    /// Loaded sampler value ids that came from AIR-embedded `__air_sampler_state` globals.
    sampler_states: HashMap<Word, StaticSamplerState>,
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
        module.sync_id_bound_from_header();
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
            image_dims: HashMap::new(),
            image_comp: HashMap::new(),
            image_multisampled: HashSet::new(),
            image_storage: HashSet::new(),
            image_array_vars: HashMap::new(),
            null_image_values: HashSet::new(),
            laid_out: HashSet::new(),
            air_struct_offsets,
            air_data_layout,
            kernel_local_size: options.kernel_local_size,
            kernel_threads_per_grid: options.kernel_threads_per_grid,
            simd_cluster32: options.simd_cluster32,
            raster_sample_count: options.raster_sample_count,
            runtime_sampler_states: options.runtime_sampler_states,
            default_sampler_var: None,
            sampler_states: HashMap::new(),
            default_null_image_vars: HashMap::new(),
            implicit_imageblock_vars: HashMap::new(),
            fragment_imageblock_vars: HashMap::new(),
            uses_fragment_imageblock: false,
            fragment_imageblock_coord_var: None,
            writes_frag_depth: false,
        }
    }
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
        for inst in self
            .module
            .types_global_values
            .iter()
            .chain(self.new_globals.iter())
        {
            if inst.class.opcode == op
                && inst.result_type == result_type
                && inst.operands == operands
            {
                if let Some(rid) = inst.result_id {
                    self.struct_cache.insert(key, rid);
                    return rid;
                }
            }
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
mod cfg_repair;
mod emitted_inline;
mod finalize;
#[cfg(test)]
mod lowering_regression_tests;
mod module_cleanup;
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
    drop_dead_invalid_access_chains, drop_overindexed_zero_tail,
    drop_writeonly_dead_local_array_stores, expose_nullable_access_chain_bases,
    fix_noop_width_converts, guard_integer_division_by_zero, hoist_function_variables,
    lower_cross_member_subword_load, lower_cross_member_subword_store,
    lower_private_byte_aggregate_reinterpret, lower_private_memory_atomics,
    lower_scalar_i64_arithmetic_to_u32_halves, lower_subword_scalar_store,
    narrow_access_chain_indices, neutralize_null_access_chains,
    neutralize_private_placeholder_access_chains, normalize_int_arith_operand_widths,
    normalize_scalar_store_types, recover_inlined_local_dynamic_pointer_fields,
    recover_inlined_local_pointer_fields, recover_unique_local_pointer_field_loads,
    remap_dynamic_word_index_to_array_member, remap_dynamic_word_index_to_array_struct_field,
    remap_overflow_word_index_to_outer_member, remap_word_index_to_struct_member,
    remodel_workgroup_flatword_aggregate, remodel_workgroup_floatarray_atomic_as_uint,
    remodel_workgroup_single_field_struct_array, repair_load_through_array_pointer,
    repair_scalar_load_through_vector_ptr, repair_vector_load_through_raw_word_pointer,
    repair_vector_load_through_scalar_stride, reroot_demoted_array_element_overindex,
    retype_demoted_copymemory_placeholder, rewrite_byte_buffer_chained_reinterpret,
    rewrite_chained_element_reinterpret, rewrite_dynamic_homogeneous_struct_index_load,
    rewrite_dynamic_struct_index_reinterpret, rewrite_dynamic_struct_index_subword_reinterpret,
    rewrite_dynamic_struct_index_vector_reinterpret,
    rewrite_dynamic_struct_index_wide_word_reinterpret, rewrite_exact_raw_byte_block_loads,
    rewrite_flat_scalar_ptr_access_through_vector_array, rewrite_raw_byte_pointer_direct_loads,
    rewrite_raw_byte_pointer_wide_loads, rewrite_raw_byte_pointer_wide_stores,
    rewrite_reinterpret_scalar_loads, rewrite_scalar_pointer_arithmetic_access_chains,
    rewrite_scalar_slot_array_overindex, rewrite_strided_descent_access_chains,
    split_workgroup_ptr_access_chain_descent,
};
use air_calls::lower_air_calls;
use emitted_inline::{
    compose_chained_access_chains, inline_selected_helpers, prune_unreferenced_functions,
};
use finalize::finalize;
use prune::prune_unreachable_blocks;
use resources::rewrites::{rewrite_affine_raw_word_loads, rewrite_exact_raw_word_loads};
use resources::*;
use stage_input::build_stage_input;
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
                    || s == "llvm.agx2.cluster.num"
                    || matches!(
                        s.as_str(),
                        "llvm.agx3.load.with.emask.global.v4i8"
                            | "llvm.agx3.load.with.emask.global.v4i16"
                            | "llvm.agx3.load.with.emask.global.v4i32"
                            | "llvm.agx3.store.with.emask.global.v4i8"
                            | "llvm.agx3.store.with.emask.global.v4i16"
                            | "llvm.agx3.store.with.emask.global.v4i32"
                    )
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
    if !partition_embedded_blocks(&mut module) {
        return Ok((module, emit_sidecar));
    }
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
    if selected_ids.is_empty() {
        return Ok((ctx.module, ctx.emit_sidecar));
    }
    inline_selected_helpers(&mut ctx, entry_idx, &selected_ids)?;
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

pub(crate) fn repair_relooped_access_chains(
    module: &mut Module,
    entry_name: Option<&str>,
) -> Result<(), String> {
    let input = std::mem::take(module);
    let mut ctx = Ctx::with_options_and_sidecar(
        input,
        crate::emit_sidecar::EmitSidecar::default(),
        Stage::Kernel,
        TransformOptions::default(),
    );
    let entry_idx = find_entry_index(&ctx.module, entry_name).ok_or_else(|| {
        "no entry function with a body found for relooped access repair".to_string()
    })?;
    rewrite_dynamic_struct_index_reinterpret(&mut ctx, entry_idx)?;
    rewrite_dynamic_struct_index_subword_reinterpret(&mut ctx, entry_idx)?;
    rewrite_dynamic_struct_index_wide_word_reinterpret(&mut ctx, entry_idx)?;
    rewrite_dynamic_struct_index_vector_reinterpret(&mut ctx, entry_idx)?;
    rewrite_dynamic_homogeneous_struct_index_load(&mut ctx, entry_idx)?;
    ctx.module.types_global_values.append(&mut ctx.new_globals);
    *module = ctx.module;
    Ok(())
}

/// Re-close exact raw-byte leaf accesses after complete-module pointer rewrites. Those rewrites run
/// after the main transform and can expose a typed aggregate carrier only at this boundary; retain
/// the emitter sidecar until the carrier's constant descendants have been replayed as exact bytes.
pub(crate) fn repair_exact_raw_byte_loads_after_native_rewrites(
    module: &mut Module,
    emit_sidecar: &crate::emit_sidecar::EmitSidecar,
    entry_name: Option<&str>,
) -> Result<(), String> {
    let input = std::mem::take(module);
    let mut ctx = Ctx::with_options_and_sidecar(
        input,
        emit_sidecar.clone(),
        Stage::Kernel,
        TransformOptions::default(),
    );
    let entry_idx = find_entry_index(&ctx.module, entry_name).ok_or_else(|| {
        "no entry function with a body found for late raw-byte repair".to_string()
    })?;
    rewrite_exact_raw_byte_block_loads(&mut ctx, entry_idx);
    drop_dead_invalid_access_chains(&mut ctx, entry_idx);
    decorate_ptr_access_chain_base_strides(&mut ctx);
    ctx.module.types_global_values.append(&mut ctx.new_globals);
    *module = ctx.module;
    Ok(())
}

/// Re-close null-pointer access invariants after complete-module pointer-phi rewrites. Phi-the-index
/// can rematerialize a chain from an incoming null root after the main transform's early null cleanup;
/// dereferencing that arm was already undefined in LLVM, so preserve the established poisoned-value
/// lowering instead of emitting an invalid logical-pointer chain.
pub(crate) fn neutralize_null_access_chains_after_native_rewrites(
    module: &mut Module,
    entry_name: Option<&str>,
) -> Result<(), String> {
    let input = std::mem::take(module);
    let mut ctx = Ctx::with_options_and_sidecar(
        input,
        crate::emit_sidecar::EmitSidecar::default(),
        Stage::Kernel,
        TransformOptions::default(),
    );
    let entry_idx = find_entry_index(&ctx.module, entry_name).ok_or_else(|| {
        "no entry function with a body found for late null-access repair".to_string()
    })?;
    neutralize_null_access_chains(&mut ctx, entry_idx);
    ctx.module.types_global_values.append(&mut ctx.new_globals);
    *module = ctx.module;
    Ok(())
}

/// Repair aggregate-stride Workgroup pointer chains after an external module rewrite (notably
/// function-constant branch pruning) has changed the final pointer graph.
pub(crate) fn repair_specialized_workgroup_ptr_access_chains(module: &mut Module) {
    let owned = std::mem::replace(module, Module::new());
    let mut ctx = Ctx::with_options_and_sidecar(
        owned,
        crate::emit_sidecar::EmitSidecar::default(),
        Stage::Kernel,
        TransformOptions::default(),
    );
    split_workgroup_ptr_access_chain_descent(&mut ctx, 0);
    *module = ctx.module;
}

pub(crate) fn lower_scalar_i64_arithmetic_module(module: &mut Module) {
    let owned = std::mem::take(module);
    let mut ctx = Ctx::with_options_and_sidecar(
        owned,
        crate::emit_sidecar::EmitSidecar::default(),
        Stage::Kernel,
        TransformOptions::default(),
    );
    lower_scalar_i64_arithmetic_to_u32_halves(&mut ctx);
    ctx.module.types_global_values.append(&mut ctx.new_globals);
    *module = ctx.module;
}

/// Reproduce SPIR-V loader block partitioning without crossing the words boundary.
///
/// Returns `false` without mutation when the instruction stream would not parse: every embedded
/// label must immediately follow a block terminator, and every resulting block must end in one.
fn partition_embedded_blocks(module: &mut Module) -> bool {
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
        return false;
    }

    for function in &mut module.functions {
        let mut blocks = Vec::new();
        for block in std::mem::take(&mut function.blocks) {
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
    }
    true
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

fn repair_structured_cfg(ctx: &mut Ctx, entry_idx: usize) {
    let mut phase_started = std::time::Instant::now();
    let mut debug_size = |ctx: &Ctx, phase: &str| {
        if crate::env_vars::retry_debug() {
            let blocks = &ctx.module.functions[entry_idx].blocks;
            let instructions = blocks
                .iter()
                .map(|block| block.instructions.len())
                .sum::<usize>();
            let operands = blocks
                .iter()
                .flat_map(|block| &block.instructions)
                .map(|instruction| instruction.operands.len())
                .sum::<usize>();
            eprintln!(
                "[retry-debug] cfg-repair {phase}: blocks={} instructions={instructions} operands={operands} phase_ms={}",
                blocks.len(),
                phase_started.elapsed().as_millis()
            );
            phase_started = std::time::Instant::now();
        }
    };
    debug_size(ctx, "start");
    // A merge instruction must immediately precede its terminator, and loop continues must not also
    // serve as selection merges. Establish those local prerequisites before the mutually dependent
    // edge/phi repairs.
    cfg_repair::fix_merge_placement(ctx, entry_idx);
    cfg_repair::funnel_selection_merge_bypasses(ctx, entry_idx);
    cfg_repair::privatize_shared_direct_selection_arms(ctx, entry_idx);
    cfg_repair::repair_loop_continue_pass_through_targets(ctx, entry_idx);
    debug_size(ctx, "local-edges");

    // Fixing a loop-continue's external predecessor can leave a stale phi incoming, while repairing
    // that phi edge can expose another external predecessor. Iterate to a checked fixpoint.
    const CONTINUE_PHI_REPAIR_CAP: usize = 8;
    let mut continue_phi_converged = false;
    for _ in 0..CONTINUE_PHI_REPAIR_CAP {
        let continue_changed =
            cfg_repair::repair_loop_continue_external_predecessors(ctx, entry_idx);
        let phi_changed = cfg_repair::repair_phi_predecessor_edges(ctx, entry_idx);
        if !continue_changed && !phi_changed {
            continue_phi_converged = true;
            break;
        }
    }
    debug_assert!(
        continue_phi_converged,
        "loop-continue/phi-edge repair did not converge within {CONTINUE_PHI_REPAIR_CAP} rounds"
    );
    // External-predecessor repair establishes whether a selection is nested in a loop or encloses
    // it. Only then can a selection that named the old continue choose the correct merge boundary.
    cfg_repair::repair_continue_selection_merge_targets(ctx, entry_idx);
    debug_size(ctx, "continue-phi-fixpoint");

    // Large reducible functions can contain many nested conditionals with the same natural
    // post-dominator. Give every structured header a private merge before repairing loop ordering.
    cfg_repair::split_reused_merge_targets(ctx, entry_idx);
    debug_size(ctx, "private-merges");
    cfg_repair::split_merges_that_are_enclosing_continues(ctx, entry_idx);
    debug_size(ctx, "private-continue-merges");

    // Loop-header synthesis and merge ordering feed each other: moving a merge after its dominators
    // can reveal a serialized backward edge to a selection header. Iterate to convergence.
    const LOOP_STRUCTURE_REPAIR_CAP: usize = 16;
    let mut loop_structure_converged = false;
    for _ in 0..LOOP_STRUCTURE_REPAIR_CAP {
        let stale_changed = cfg_repair::downgrade_stale_loop_merges(ctx, entry_idx);
        let order_changed = cfg_repair::repair_dominator_block_order(ctx, entry_idx);
        let loop_changed = cfg_repair::repair_unmarked_natural_loops(ctx, entry_idx);
        debug_size(ctx, "loop-round");
        if !stale_changed && !order_changed && !loop_changed {
            loop_structure_converged = true;
            break;
        }
    }
    debug_assert!(
        loop_structure_converged,
        "loop-header/order repair did not converge within {LOOP_STRUCTURE_REPAIR_CAP} rounds"
    );
    cfg_repair::privatize_nondominated_construct_merges(ctx, entry_idx);
    // Loop-header synthesis can insert a pass-through between the original preheader/backedge set
    // and a phi block after the earlier edge fixpoint has run. Reconcile those final predecessor
    // identities, including moving a multi-edge value merge into the new header when necessary.
    cfg_repair::repair_phi_predecessor_edges(ctx, entry_idx);
}

/// Re-establish the same structured-CFG invariant after finish-time native module rewrites.
///
/// Those rewrites intentionally operate on a complete SPIR-V module and can introduce or redirect
/// control-flow edges after the main lowering pass. Routing the result through this shared phase
/// keeps the two layers compositional instead of requiring every native rewrite to duplicate every
/// merge, continue, and phi repair rule.
pub(crate) fn repair_structured_cfg_after_native_rewrites(
    module: Module,
    stage: Stage,
    entry_name: Option<&str>,
) -> Result<Module, String> {
    let mut ctx = Ctx::with_options_and_sidecar(
        module,
        crate::emit_sidecar::EmitSidecar::default(),
        stage,
        TransformOptions::default(),
    );
    let entry_idx = find_entry_index(&ctx.module, entry_name)
        .ok_or_else(|| "no entry function with a body found after native rewrites".to_string())?;
    repair_structured_cfg(&mut ctx, entry_idx);
    Ok(ctx.module)
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
        TransformOptions::default(),
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
    .map(|(module, mut sidecar)| {
        // This test-only byte boundary intentionally discards the sidecar. Drop its source-layout
        // oracle roots too so direct transform tests observe the same serialized type cleanup as a
        // completed product translation after the late repair seam.
        sidecar.buffer_root_source_types.clear();
        let mut ctx = Ctx::with_options_and_sidecar(module, sidecar, stage, options);
        module_cleanup::gc_dead_globals(&mut ctx);
        ctx.module
    })
}

pub(crate) fn validate_descriptor_bindings(module: &Module) -> Result<(), String> {
    resources::validate_descriptor_binding_classes(module)
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
) -> Result<(Module, crate::emit_sidecar::EmitSidecar), String> {
    if matches!(stage, Stage::Kernel) && options.kernel_local_size.contains(&0) {
        return Err("kernel LocalSize dimensions must be non-zero".to_string());
    }
    let mut ctx = Ctx::with_options_and_sidecar(module, emit_sidecar, stage, options);
    let retry_debug = crate::env_vars::retry_debug();
    let debug_phase = |phase: &str| {
        if retry_debug {
            eprintln!("[retry-debug] passes: {phase}");
        }
    };

    let mut entry_idx = find_entry_index(&ctx.module, entry_name)
        .ok_or_else(|| "no entry function with a body found".to_string())?;

    debug_phase("cleanup start");
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

    debug_phase("interface start");
    // 1a) decoded params -> stage-input/resource vars and entry-body replacements. Preserve the
    //     original type snapshot for the immediately-following return rewrite.
    let input_defs = build_stage_input(&mut ctx, entry_idx, &stage, frag, vert, kern)?;
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

    debug_phase("air lowering start");
    // 2) lower residual air.* calls inside the entry function.
    lower_air_calls(&mut ctx, entry_idx)?;
    // AIR calls can materialize opaque handles after interface binding's earlier resource-wrapper
    // collapse (notably `air.get_null_texture_*`). Re-establish the same aggregate invariant for
    // those late values before any subsequent pass observes their former pointer-shaped fields.
    resources::collapse_late_pointer_and_opaque_wrappers(&mut ctx, entry_idx)?;
    recover_unique_local_pointer_field_loads(&mut ctx, entry_idx);
    recover_inlined_local_dynamic_pointer_fields(&mut ctx, entry_idx)?;
    lower_private_memory_atomics(&mut ctx, entry_idx);

    // 2b) repair width-preserving int/float converts (illegal in SPIR-V) that arise from binding a
    //     narrower AIR param (`ushort [[vertex_id]]`) to the 32-bit Vulkan builtin -> a same-width
    //     `OpUConvert` the body's original widening cast compiled to. A no-op `OpCopyObject` is correct.
    fix_noop_width_converts(&mut ctx, entry_idx);

    // 2c) define otherwise-undefined integer divide-by-zero side arms before drivers can materialize
    //     divergent values from guarded select expressions.
    guard_integer_division_by_zero(&mut ctx, entry_idx);

    debug_phase("cfg repair start");
    // 2d) establish the complete structured-CFG invariant. The same compositional phase is rerun
    // after finish_module's native module rewrites, because those late rewrites can create new CFG
    // edges and must not bypass the invariant established here.
    repair_structured_cfg(&mut ctx, entry_idx);

    debug_phase("memory lowering start");
    // 2e) narrow 64-bit access-chain INDEX operands to 32-bit. NVIDIA's SPIR-V->NVVM compiler crashes
    //     when an access-chain index is a 64-bit (`%ulong`) value (a driver-fragility class), so we
    //     value-preservingly rewrite each i64 index to a 32-bit equivalent. After this the i64 index
    //     constants/types are dead and `finalize` can drop the now-unused `OpCapability Int64`.
    narrow_access_chain_indices(&mut ctx, entry_idx);
    expose_nullable_access_chain_bases(&mut ctx, entry_idx);
    compose_derived_access_chains(&mut ctx, entry_idx);
    rewrite_scalar_pointer_arithmetic_access_chains(&mut ctx, entry_idx);
    drop_overindexed_zero_tail(&mut ctx, entry_idx);
    reroot_demoted_array_element_overindex(&mut ctx, entry_idx);
    remap_word_index_to_struct_member(&mut ctx, entry_idx);
    remap_overflow_word_index_to_outer_member(&mut ctx, entry_idx);
    remap_dynamic_word_index_to_array_member(&mut ctx, entry_idx);
    remap_dynamic_word_index_to_array_struct_field(&mut ctx, entry_idx);
    repair_vector_load_through_scalar_stride(&mut ctx, entry_idx);
    repair_vector_load_through_raw_word_pointer(&mut ctx, entry_idx);
    repair_scalar_load_through_vector_ptr(&mut ctx, entry_idx);
    drop_writeonly_dead_local_array_stores(&mut ctx, entry_idx);
    lower_cross_member_subword_load(&mut ctx, entry_idx)?;
    lower_cross_member_subword_store(&mut ctx, entry_idx);
    normalize_scalar_store_types(&mut ctx, entry_idx);
    lower_subword_scalar_store(&mut ctx, entry_idx);
    drop_dead_invalid_access_chains(&mut ctx, entry_idx);
    rewrite_strided_descent_access_chains(&mut ctx, entry_idx);
    rewrite_dynamic_struct_index_reinterpret(&mut ctx, entry_idx)?;
    rewrite_dynamic_struct_index_subword_reinterpret(&mut ctx, entry_idx)?;
    rewrite_dynamic_struct_index_wide_word_reinterpret(&mut ctx, entry_idx)?;
    rewrite_dynamic_struct_index_vector_reinterpret(&mut ctx, entry_idx)?;
    rewrite_dynamic_homogeneous_struct_index_load(&mut ctx, entry_idx)?;
    rewrite_chained_element_reinterpret(&mut ctx, entry_idx)?;
    rewrite_byte_buffer_chained_reinterpret(&mut ctx, entry_idx)?;
    // The raw-byte replay introduces PtrAccessChain byte pointers.  Decorate their common uchar
    // base type immediately below, together with every pre-existing PtrAccessChain base.
    rewrite_raw_byte_pointer_direct_loads(&mut ctx, entry_idx);
    rewrite_raw_byte_pointer_wide_loads(&mut ctx, entry_idx);
    rewrite_raw_byte_pointer_wide_stores(&mut ctx, entry_idx);
    decorate_ptr_access_chain_base_strides(&mut ctx);
    // Runs AFTER the stride decoration so the byte-buffer (PtrAccessChain) widen can read the base
    // pointer's ArrayStride to prove slot contiguity.
    rewrite_reinterpret_scalar_loads(&mut ctx, entry_idx);
    rewrite_scalar_slot_array_overindex(&mut ctx, entry_idx)?;
    remodel_workgroup_flatword_aggregate(&mut ctx, entry_idx);
    remodel_workgroup_single_field_struct_array(&mut ctx, entry_idx);
    remodel_workgroup_floatarray_atomic_as_uint(&mut ctx, entry_idx)?;
    lower_private_byte_aggregate_reinterpret(&mut ctx, entry_idx)?;
    retype_demoted_copymemory_placeholder(&mut ctx, entry_idx);
    // Memory/interface rewrites above can expose a concrete pointer only after the first late
    // wrapper pass ran. Re-close the no-pointer-carrier invariant before finalization so an exact
    // insert/extract round trip cannot retain the aggregate field's stale opaque-pointer type.
    resources::collapse_late_pointer_and_opaque_wrappers(&mut ctx, entry_idx)?;
    // The newly concrete root can be a canonical raw-byte buffer block. Replay wide loads that were
    // formerly hidden behind the stale private pointer, then decorate the PtrAccessChains introduced
    // by that replay just as in the primary raw-memory phase above.
    rewrite_exact_raw_byte_block_loads(&mut ctx, entry_idx);
    rewrite_raw_byte_pointer_wide_loads(&mut ctx, entry_idx);
    repair_load_through_array_pointer(&mut ctx, entry_idx);
    rewrite_flat_scalar_ptr_access_through_vector_array(&mut ctx, entry_idx);
    rewrite_exact_raw_word_loads(&mut ctx, entry_idx);
    rewrite_affine_raw_word_loads(&mut ctx, entry_idx);
    drop_dead_invalid_access_chains(&mut ctx, entry_idx);
    decorate_ptr_access_chain_base_strides(&mut ctx);
    // Late raw-memory rewrites can introduce new access chains rooted in the same unnamed,
    // null-initialized Private placeholders that represent absent optional resources. Reapply the
    // structural placeholder closure after every pointer-producing memory pass has run.
    neutralize_private_placeholder_access_chains(&mut ctx, entry_idx)?;

    debug_phase("integer lowering start");
    // 2h) width-normalize integer arithmetic operands. A value widened to `ulong` for a StorageBuffer
    //     access-chain index that is then reused in a 32-bit offset multiply leaves an `OpIMul %uint`
    //     with a `ulong` operand, which spirv-val rejects. This inserts a truncating `OpUConvert` for
    //     any arithmetic/bitwise operand wider than its result type — byte-neutral when widths already
    //     match, purely structural (bit width only, no name matching). Runs module-wide, last, so it
    //     repairs the mismatch regardless of which earlier lowering pass introduced it.
    normalize_int_arith_operand_widths(&mut ctx);
    // 2i) Some conformant Vulkan devices expose `shaderInt64` but execute scalar 64-bit arithmetic
    //     via a fragile native path. Rebuild add/sub/mul from 32-bit/16-bit pieces; this is the same
    //     modular result as the original `ulong` arithmetic and is structural over integer width.
    lower_scalar_i64_arithmetic_to_u32_halves(&mut ctx);

    debug_phase("workgroup/finalize start");
    // 2g) deterministic threadgroup memory: zero-fill every Workgroup variable at kernel entry
    //     (stores of OpConstantNull + one control barrier), the candidate half of the harness's
    //     defined refinement of Metal's undefined threadgroup contents. Kernel-only: Workgroup
    //     variables exist only in compute, and the barrier is only legal there.
    if matches!(stage, Stage::Kernel) {
        workgroup::unroll_small_workgroup_atomic_loops(&mut ctx, entry_idx);
        workgroup::zero_initialize_workgroup_memory(&mut ctx, entry_idx);
    }
    // 2f) drop blocks unreachable from the entry (orphaned constructs the structured-CFG repair's
    //     clone-then-rewire steps can leave behind). spirv-val tolerates them; SPIRV-Cross (MoltenVK)
    //     throws on them at pipeline creation. See `prune.rs`.
    prune_unreachable_blocks(&mut ctx.module);

    resources::sink_loop_header_texture_array_loads(&mut ctx, entry_idx);
    // 3) finalize: append synthesized globals, drop dead air.* decls, add entry point + exec modes,
    //    bump the bound.
    finalize(&mut ctx, entry_idx, &stage, vert)?;
    // Run on the complete type graph: retained helper functions can carry a Workgroup aggregate
    // stride that entry-focused lowering could not see until all synthesized globals were appended.
    split_workgroup_ptr_access_chain_descent(&mut ctx, entry_idx);
    debug_phase("complete");

    Ok((ctx.module, ctx.emit_sidecar))
}
