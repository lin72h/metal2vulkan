//! Rewrite decoded AIR entry parameters into Vulkan stage-input and resource variables.

use super::*;
use crate::meta::primitive_air_type_from_name;
use crate::passes::access::{is_unsigned_byte_scalar, single_member_array_scalar_elem};
use crate::passes::stage_output::handle_static_sampler;
mod decorations;
mod kernel_grid;
mod kernel_values;
mod layout;

pub(in crate::passes) use decorations::*;
pub(in crate::passes) use kernel_grid::{
    bind_kernel_grid_push_constant_once, load_kernel_dispatch_component,
    materialize_kernel_dispatch_field,
};
pub(in crate::passes) use kernel_values::const_ivec;
use kernel_values::{
    bind_kernel_uvec3_builtin, bind_kernel_uvec3_builtin_var, const_kernel_local_size,
    is_raw_uint_buffer_block,
};
use layout::*;

// Layout size/align helpers are reused by the lower pass to decorate OpPtrAccessChain base
// pointer types with ArrayStride (a sibling-module of interface cannot reach `layout` directly).
pub(in crate::passes) use layout::{
    decorate_block_struct, drop_unconsumed_placeholder_descriptor_loads, layout_ty_size_align,
    round_up,
};

mod air_layout;
pub(in crate::passes) use air_layout::*;
const WORKGROUP_MEMORY_ELEMENTS: u32 = 512;

/// What an entry parameter became, so the body can be patched to read from it.
pub(in crate::passes) fn fragment_imageblock_projection_type_matches(
    defs: &HashMap<Word, Instruction>,
    ty: Word,
    format: FragmentImageblockFormat,
) -> bool {
    let scalar = if format.lanes == 1 {
        ty
    } else {
        let Some(definition) = defs.get(&ty) else {
            return false;
        };
        match definition.operands.as_slice() {
            [Operand::IdRef(scalar), Operand::LiteralBit32(lanes)]
                if definition.class.opcode == Op::TypeVector && *lanes == format.lanes =>
            {
                *scalar
            }
            _ => return false,
        }
    };
    let Some(definition) = defs.get(&scalar) else {
        return false;
    };
    match format.component {
        ImageComp::Float => {
            definition.class.opcode == Op::TypeFloat
                && definition.operands.as_slice() == [Operand::LiteralBit32(format.bits)]
        }
        ImageComp::Uint => {
            definition.class.opcode == Op::TypeInt
                && definition.operands.as_slice()
                    == [Operand::LiteralBit32(format.bits), Operand::LiteralBit32(0)]
        }
        ImageComp::Sint => false,
    }
}

pub(in crate::passes) enum ParamBinding {
    /// A loadable interface var (Input varying / vertex attribute): replace param uses with an
    /// OpLoad of `var` (type = the param's value type) inserted at function start.
    LoadVar { var: Word, ty: Word },
    /// A scalar fragment bool varying. Vulkan user IO cannot use OpTypeBool, so the interface slot is
    /// a flat uint and the loaded value is compared against zero at function entry.
    LoadVarBoolFromUint { var: Word, bool_ty: Word },
    /// A scalar builtin Input var (`VertexIndex`/`InstanceIndex`, a 32-bit uint) feeding a NARROWER
    /// integer param (`ushort [[instance_id]]`, an i16): load the uint then `OpUConvert` it down to the
    /// param's own width, so the body's 16-bit uses (`OpBitwiseAnd %ushort`) are width-consistent.
    LoadVarConverted {
        var: Word,
        load_ty: Word,
        param_ty: Word,
    },
    /// A loadable interface var whose signedness was recovered from AIR metadata while the LLVM body
    /// still uses the original signless integer type id. Load the interface type, then bitcast to the
    /// body's parameter type.
    LoadVarBitcast {
        var: Word,
        load_ty: Word,
        param_ty: Word,
    },
    /// Load a scalar builtin Input var, mask it, and splice the scalar result.
    LoadVarBitAnd {
        var: Word,
        load_ty: Word,
        param_ty: Word,
        mask: u32,
    },
    /// Load a scalar builtin Input var, shift it right, and splice the scalar result.
    LoadVarShiftRight {
        var: Word,
        load_ty: Word,
        param_ty: Word,
        shift: u32,
    },
    /// A vector builtin Input var (e.g. GlobalInvocationId, a v3uint) whose `comp`-th component feeds
    /// a scalar param: load the vector then OpCompositeExtract the 32-bit component, converting when the
    /// AIR param is narrower.
    LoadVarComponent {
        var: Word,
        vec_ty: Word,
        scalar_ty: Word,
        out_ty: Word,
        comp: u32,
    },
    /// A vector builtin Input var whose leading components feed a smaller AIR vector, e.g. Vulkan's
    /// v3uint LocalInvocationId -> AIR `uint2 [[thread_position_in_threadgroup]]`.
    LoadVarVectorPrefix {
        var: Word,
        vec_ty: Word,
        scalar_ty: Word,
        prefix_ty: Word,
        out_ty: Word,
        lanes: u32,
    },
    /// Vulkan's NumWorkgroups builtin multiplied by the kernel LocalSize, shaped as AIR
    /// `[[threads_per_grid]]`.
    LoadThreadsPerGrid {
        var: Word,
        vec_ty: Word,
        out_ty: Word,
        lanes: u32,
    },
    /// One three-component field loaded from the exact-thread dispatch payload.
    LoadKernelDispatchField {
        var: Word,
        first_member: u32,
        out_ty: Word,
        lanes: u32,
    },
    /// A Vulkan grid builtin plus one three-component base from the dispatch payload.
    LoadBuiltinPlusKernelDispatchField {
        builtin_var: Word,
        dispatch_var: Word,
        first_member: u32,
        out_ty: Word,
        lanes: u32,
    },
    /// The pipeline-specialized local size, shaped as the AIR parameter type.
    LoadKernelLocalSize { out_ty: Word, lanes: u32 },
    /// The number of 32-wide SIMD groups in the pipeline-specialized local size.
    LoadKernelSimdgroupsPerThreadgroup { out_ty: Word },
    /// An image variable (texture): param uses are the sample call's texture operand; replace param
    /// id with an OpLoad of the image at use. We record the var + image type + its (Dim, arrayed).
    Image {
        var: Word,
        image_ty: Word,
        dim: (Dim, bool),
        comp: ImageComp,
        multisampled: bool,
    },
    /// A runtime-indexed texture ARRAY (`array_ref<texture>`): a descriptor array of sampled or storage
    /// images. Declared as `OpTypeArray %image N` in UniformConstant. Unlike `Image`, the param is NOT
    /// replaced by a loaded image at function top; it is spliced to the array VARIABLE, and
    /// `materialize_texture_array_loads` turns each per-element handle load into
    /// `OpAccessChain %arrayvar %idx` + `OpLoad %image` at the use site. `elem_image_ty` is the element
    /// `OpTypeImage`; the pass records `(var -> (elem_image_ty, dim, comp))` in `ctx.image_array_vars`.
    ImageArray {
        var: Word,
        elem_image_ty: Word,
        dim: (Dim, bool),
        comp: ImageComp,
        multisampled: bool,
        runtime_specialization: Option<(u32, crate::reflect::RuntimeStorageImageState)>,
    },
    /// A write-only storage-image variable (`OpTypeImage Sampled=2` + ImageFormat): param uses are the
    /// `air.write_texture_*` texture operand; replace the param id with an OpLoad of the storage image.
    /// The loaded id is recorded in `ctx.image_storage` so the write lowering emits `OpImageWrite`.
    StorageImage {
        var: Word,
        image_ty: Word,
        dim: (Dim, bool),
        comp: ImageComp,
        runtime_specialization: Option<(u32, crate::reflect::RuntimeStorageImageState)>,
    },
    /// A framebuffer-fetch `[[color(n)]]` input. Vulkan exposes these as subpass input attachments,
    /// read with `OpImageRead` at the current fragment location.
    InputAttachment {
        var: Word,
        image_ty: Word,
        read_ty: Word,
        param_ty: Word,
    },
    /// A custom fragment-imageblock projection reconstructed by storage-image reads at FragCoord.
    FragmentImageblockProjection {
        coord_var: Word,
        param_ty: Word,
        members: Vec<(Word, Word, Word, FragmentImageblockFormat)>,
    },
    /// A sampler variable: replace param uses with an OpLoad of the sampler.
    Sampler {
        var: Word,
        specialized_state: Option<StaticSamplerState>,
    },
    /// A buffer block variable, with the lowering of the body's param uses (see `BufWrap`).
    Buffer { var: Word, wrap: BufWrap },
    /// A compute `[[stage_in]]` value. Vulkan has no compute-stage attribute stream, so the
    /// translator exposes each attribute as a read-only StorageBuffer array and indexes it by
    /// `GlobalInvocationId.x`.
    StageInput {
        var: Word,
        value_ty: Word,
        index_var: Word,
        dispatch_var: Option<Word>,
    },
    /// Threadgroup memory (`air.buffer` with `air.address_space = 3`): a fixed Workgroup array.
    WorkgroupMemory { var: Word },
    /// A non-resource value that can be spliced directly into parameter uses.
    Value { val: Word },
    /// Position builtin / unused: bind to a zero-initialized value of the param type.
    ZeroValue { val: Word },
    /// An unmodeled *pointer* param (e.g. an unbound `constant T&` buffer): bind it to a real Private
    /// zero-initialized variable of the pointee type rather than an `OpUndef` pointer. An undef pointer
    /// whose pointee is a non-opaque data aggregate, dereferenced via OpAccessChain/OpLoad, is both
    /// illegal Vulkan (UniformConstant data pointers, VUID-04655) and — even when it slips past
    /// spirv-val as an OpUndef — SEGFAULTs NVIDIA's libnvidia-glvkspirv SPIR-V->NVVM compiler. Routing
    /// the reads through a Private zero var yields the same "absent resource reads zero" semantics with
    /// a storage class NVIDIA compiles cleanly. Derived access chains are re-storage-classed to Private
    /// in `apply_bindings`.
    ZeroPointer { var: Word },
}

fn texture_type_is_handle_array(name: &str) -> bool {
    let compact = name
        .chars()
        .filter(|ch| !ch.is_ascii_whitespace())
        .collect::<String>();
    compact.contains("array_ref<texture") || compact.contains("array<texture")
}

/// How a buffer param's body uses are lowered.
pub(in crate::passes) enum BufWrap {
    /// The AIR pointee is a heterogeneous struct emitted as a struct pointer. The body's
    /// access chains already index struct members off the param — just splice the var id.
    Direct,
    /// Native emission kept a struct pointer, but at least one AIR use indexes an implicit array of those
    /// structs (`buffer[N].field`). The StorageBuffer is `{ RuntimeArray<Struct> }`; direct struct
    /// member paths are routed to record 0, and non-direct paths keep their original first operand as
    /// the record index.
    RecordArray { block_ty: Word, elem_ty: Word },
    /// Native emission represents the buffer as a bare element pointer (`T*`) with physical access
    /// chains against it (illegal in Logical SPIR-V). `block_ty` is the StorageBuffer Block we point
    /// the var at; we re-root the body's access chains at the var and route direct loads through the
    /// offset-0 leaf. `prepend_member0` is true for a genuine `device T*` array wrapped as
    /// `{ RuntimeArray<T> }` (the original first index then indexes the runtime array); false for a
    /// reconstructed struct (the original indices already navigate it).
    Collapsed {
        block_ty: Word,
        prepend_member0: bool,
        typed_aliases: Vec<(Word, Word)>,
    },
}

fn required_resource_binding(param: Word, binding: Option<u32>) -> Result<u32, String> {
    binding.ok_or_else(|| {
        format!("descriptor-backed entry parameter %{param} has no AIR descriptor ABI binding")
    })
}

/// Rewrite the native emitter's entry parameters into Vulkan interface variables by AIR role, and
/// its return value into Output variables.
pub(super) fn build_stage_input(
    ctx: &mut Ctx,
    entry_idx: usize,
    stage: &Stage,
    frag: Option<&FragMeta>,
    vert: Option<&VertMeta>,
    kern: Option<&KernMeta>,
) -> Result<HashMap<Word, Instruction>, String> {
    let descriptor_layout = ctx.descriptor_layout;
    let defs = type_defs(&ctx.module);
    let params: Vec<(Word, Word)> = ctx.module.functions[entry_idx]
        .parameters
        .iter()
        .map(|p| {
            Ok((
                p.result_id.ok_or("entry param missing result id")?,
                p.result_type.ok_or("entry param missing result type")?,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let texture_type_hints = texture_type_hints(&params, stage, frag, vert, kern);
    let tex_dims = texture_dims(ctx, entry_idx, &texture_type_hints);
    // Write-capable textures (bound as storage images). A texture that is also sampled through a
    // sampler stays in `tex_dims` and binds as a sampled image; plain `read_texture` on the same
    // resource is a storage-image `OpImageRead` and can share this binding.
    let mut wtex_candidates = texture_storage_hints(&params, stage, frag, vert, kern);
    for (pid, shape) in write_texture_dims(ctx, entry_idx) {
        if shape.2 == ImageFormat::R32ui {
            // Image atomics require one of Vulkan's scalar atomic storage formats; the call-site ABI
            // is more specific than the broad metadata `texture2d<uint, read_write>` spelling.
            wtex_candidates.insert(pid, shape);
        } else {
            wtex_candidates.entry(pid).or_insert(shape);
        }
    }
    let sampled_required = sampled_binding_required_operands(ctx, entry_idx);
    let wtex_dims: HashMap<Word, (Dim, bool, ImageFormat, ImageComp)> = wtex_candidates
        .into_iter()
        .filter(|(pid, _)| !sampled_required.contains(pid))
        .collect();

    // Plan a binding for each parameter, allocating descriptor bindings and locations in role order.
    let mut bindings: Vec<(Word, ParamBinding)> = vec![];
    // Buffer params whose pointer storage class must become Uniform, with their struct type id.
    let mut buffer_structs: Vec<(Word, Word)> = vec![]; // (var_id, struct_ty)
                                                        // The single FragCoord Input var: a fragment shader may carry MORE THAN ONE `position` param (e.g.
                                                        // an FC-specialized shader threads the pixel position into several helpers), but Vulkan forbids two
                                                        // interface variables decorated with the same builtin. Create FragCoord once and share it.
    let mut fragcoord_var: Option<Word> = None;
    let mut pointcoord_var: Option<Word> = None;
    let mut front_facing_var: Option<Word> = None;
    let mut primitive_id_var: Option<Word> = None;
    let mut sample_id_var: Option<Word> = None;
    let mut layer_var: Option<Word> = None;
    let mut tess_coord_var: Option<Word> = None;
    let mut local_invocation_index_var: Option<Word> = None;
    let mut num_workgroups_var: Option<Word> = None;
    let mut global_invocation_id_var: Option<Word> = None;
    let mut kernel_grid_push_constant_var: Option<Word> = None;
    if let Some(range) = ctx.kernel_dispatch.push_constant_range() {
        bind_kernel_grid_push_constant_once(ctx, &mut kernel_grid_push_constant_var, range.offset);
    }
    let stage_input_bindings = kern.map(KernMeta::stage_input_bindings).unwrap_or_default();

    // A parameter no role recognises is bound to a zero value further down so the body stays well
    // formed. That is correct for a function-constant-disabled resource — Metal defines it as
    // absent — and wrong for a system value the emitter simply does not model: the shader reads a
    // zero where the hardware would have given it a barycentric coordinate or a sample mask, and
    // the module validates, binds and reflects exactly as if nothing were missing. Reject those
    // instead, naming the role.
    let unmodelled = match stage {
        Stage::Fragment => frag.map(|meta| meta.unmodelled_input_params.as_slice()),
        Stage::Vertex => vert.map(|meta| meta.unmodelled_input_params.as_slice()),
        Stage::Kernel => kern.map(|meta| meta.unmodelled_input_params.as_slice()),
    }
    .unwrap_or_default();
    if let Some((param, role)) = unmodelled.first() {
        return Err(format!(
            "entry parameter {param} declares AIR role `air.{role}`, which has no lowering; \
             emitting the module would silently read a zero in its place"
        ));
    }

    for (i, (pid, pty)) in params.iter().enumerate() {
        let idx = i as u32;
        let role_is = |s: &str| match stage {
            Stage::Fragment => match frag.and_then(|m| m.role_of(idx)) {
                Some(FragRole::Position) => s == "position",
                Some(FragRole::PointCoord) => s == "point_coord",
                Some(FragRole::FrontFacing) => s == "front_facing",
                Some(FragRole::PrimitiveId) => s == "primitive_id",
                Some(FragRole::SampleId) => s == "sample_id",
                Some(FragRole::ViewportArrayIndex) => s == "viewport_array_index",
                Some(FragRole::RenderTargetArrayIndex) => s == "render_target_array_index",
                Some(FragRole::Varying(_)) => s == "varying",
                Some(FragRole::Texture(_)) => s == "texture",
                Some(FragRole::Sampler(_)) => s == "sampler",
                Some(FragRole::Buffer(_)) => s == "buffer",
                Some(FragRole::ColorInput(_)) => s == "color_input",
                Some(FragRole::ImageblockData) => s == "imageblock_data",
                _ => s == "other",
            },
            Stage::Vertex => match vert.and_then(|m| m.role_of(idx)) {
                Some(VertRole::VertexInput(_)) => s == "varying",
                Some(VertRole::Buffer(_)) => s == "buffer",
                Some(VertRole::Texture(_)) => s == "texture",
                Some(VertRole::Sampler(_)) => s == "sampler",
                Some(VertRole::VertexId) => s == "vertex_id",
                Some(VertRole::InstanceId) => s == "instance_id",
                Some(VertRole::PatchControlPoints) => s == "patch_control_points",
                Some(VertRole::PatchInput(_)) => s == "patch_input",
                Some(VertRole::PositionInPatch) => s == "position_in_patch",
                Some(VertRole::PatchId) => s == "patch_id",
                Some(VertRole::AmplificationId) => s == "amplification_id",
                Some(VertRole::AmplificationCount) => s == "amplification_count",
                _ => s == "other",
            },
            Stage::Kernel => match kern.and_then(|m| m.role_of(idx)) {
                Some(
                    KernRole::Buffer(_)
                    | KernRole::AccelerationStructureShadow(_)
                    | KernRole::PrimitiveAccelerationStructureShadow(_),
                ) => s == "buffer",
                Some(KernRole::Texture(_)) => s == "texture",
                Some(KernRole::Sampler(_)) => s == "sampler",
                Some(KernRole::ThreadsPerThreadgroup) => s == "threads_per_threadgroup",
                Some(KernRole::ThreadPositionInThreadgroup) => {
                    s == "thread_position_in_threadgroup"
                }
                Some(KernRole::ThreadgroupsPerGrid) => s == "threadgroups_per_grid",
                Some(KernRole::ThreadsPerGrid) => s == "threads_per_grid",
                Some(KernRole::ThreadgroupPositionInGrid) => s == "threadgroup_position_in_grid",
                Some(KernRole::ThreadIndexInThreadgroup) => s == "thread_index_in_threadgroup",
                Some(KernRole::ThreadIndexInQuadgroup) => s == "thread_index_in_quadgroup",
                Some(KernRole::QuadgroupIndexInThreadgroup) => {
                    s == "quadgroup_index_in_threadgroup"
                }
                Some(KernRole::ThreadIndexInSimdgroup) => s == "thread_index_in_simdgroup",
                Some(KernRole::SimdgroupIndexInThreadgroup) => {
                    s == "simdgroup_index_in_threadgroup"
                }
                Some(KernRole::ThreadsPerSimdgroup) => s == "threads_per_simdgroup",
                Some(KernRole::SimdgroupsPerThreadgroup) => s == "simdgroups_per_threadgroup",
                Some(KernRole::ThreadPositionInGrid) => s == "thread_position_in_grid",
                Some(KernRole::StageInput(_)) => s == "stage_in",
                _ => s == "other",
            },
        };
        let loc = match stage {
            Stage::Fragment => match frag.and_then(|m| m.role_of(idx)) {
                Some(FragRole::Varying(l)) => *l,
                _ => 0,
            },
            Stage::Vertex => match vert.and_then(|m| m.role_of(idx)) {
                Some(VertRole::VertexInput(l)) => *l,
                _ => 0,
            },
            Stage::Kernel => 0,
        };
        let resource_binding = match stage {
            Stage::Fragment => match frag.and_then(|m| m.role_of(idx)) {
                Some(FragRole::Texture(b)) => {
                    Some(texture_resource_binding(descriptor_layout, *b)?)
                }
                Some(FragRole::Sampler(b)) => {
                    Some(sampler_resource_binding(descriptor_layout, *b)?)
                }
                Some(FragRole::Buffer(b)) => Some(buffer_resource_binding(descriptor_layout, *b)?),
                Some(FragRole::ColorInput(b)) => {
                    Some(color_input_resource_binding(descriptor_layout, *b)?)
                }
                _ => None,
            },
            Stage::Vertex => match vert.and_then(|m| m.role_of(idx)) {
                Some(VertRole::Buffer(b)) => Some(buffer_resource_binding(descriptor_layout, *b)?),
                Some(VertRole::Texture(b)) => {
                    Some(texture_resource_binding(descriptor_layout, *b)?)
                }
                Some(VertRole::Sampler(b)) => {
                    Some(sampler_resource_binding(descriptor_layout, *b)?)
                }
                _ => None,
            },
            Stage::Kernel => match kern.and_then(|m| m.role_of(idx)) {
                Some(
                    KernRole::Buffer(b)
                    | KernRole::AccelerationStructureShadow(b)
                    | KernRole::PrimitiveAccelerationStructureShadow(b),
                ) => Some(buffer_resource_binding(descriptor_layout, *b)?),
                Some(KernRole::Texture(b)) => {
                    Some(texture_resource_binding(descriptor_layout, *b)?)
                }
                Some(KernRole::Sampler(b)) => {
                    Some(sampler_resource_binding(descriptor_layout, *b)?)
                }
                _ => None,
            },
        };
        let storage_resource_binding = match stage {
            Stage::Fragment => match frag.and_then(|m| m.role_of(idx)) {
                Some(FragRole::Texture(binding)) => Some(storage_texture_resource_binding(
                    descriptor_layout,
                    *binding,
                )?),
                _ => None,
            },
            Stage::Vertex => match vert.and_then(|m| m.role_of(idx)) {
                Some(VertRole::Texture(binding)) => Some(storage_texture_resource_binding(
                    descriptor_layout,
                    *binding,
                )?),
                _ => None,
            },
            Stage::Kernel => match kern.and_then(|m| m.role_of(idx)) {
                Some(KernRole::Texture(binding)) => Some(storage_texture_resource_binding(
                    descriptor_layout,
                    *binding,
                )?),
                _ => None,
            },
        };
        let metal_texture_index = match stage {
            Stage::Fragment => frag.and_then(|meta| match meta.role_of(idx) {
                Some(FragRole::Texture(index)) => Some(*index),
                _ => None,
            }),
            Stage::Vertex => vert.and_then(|meta| match meta.role_of(idx) {
                Some(VertRole::Texture(index)) => Some(*index),
                _ => None,
            }),
            Stage::Kernel => kern.and_then(|meta| match meta.role_of(idx) {
                Some(KernRole::Texture(index)) => Some(*index),
                _ => None,
            }),
        };

        let is_threadgroup_buffer = matches!(stage, Stage::Kernel)
            && role_is("buffer")
            && (kern.and_then(|m| m.buffer_address_space(idx)) == Some(3)
                || ptr_storage(&defs, *pty) == Some(StorageClass::Workgroup));

        // Runtime-indexed texture arrays are declared as `array_ref<texture...>` or fixed
        // `array<texture...>` in AIR metadata. Both are descriptor ARRAYs, not single images: the
        // native emission produces per-element handle loads (`load ptr addrspace(1), gep %argbuf, %idx`) that
        // must become `OpAccessChain` into a UniformConstant image array.
        let texture_type_name = match stage {
            Stage::Fragment => frag.and_then(|m| m.texture_type_name(idx)),
            Stage::Vertex => vert.and_then(|m| m.texture_type_name(idx)),
            Stage::Kernel => kern.and_then(|m| m.texture_type_name(idx)),
        };
        let is_array_texture = texture_type_name.is_some_and(texture_type_is_handle_array);

        if let (Stage::Kernel, Some(KernRole::StageInput(_))) =
            (stage, kern.and_then(|m| m.role_of(idx)))
        {
            let rta = ctx.ty_runtime_array(*pty);
            let block_ty = ctx.module.fresh_id();
            ctx.new_globals.push(type_inst(
                Op::TypeStruct,
                block_ty,
                vec![Operand::IdRef(rta)],
            ));
            let pptr = ctx.ty_ptr(StorageClass::StorageBuffer, block_ty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(pptr),
                Some(var),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ));
            let binding = stage_input_bindings
                .get(&idx)
                .copied()
                .ok_or_else(|| format!("kernel stage_in parameter {idx} missing binding"))?;
            let binding = buffer_resource_binding(descriptor_layout, binding)?;
            decorate_binding(&mut ctx.module, var, descriptor_layout.set, binding);
            let index_var = bind_kernel_v3uint_builtin_once(
                ctx,
                &mut global_invocation_id_var,
                BuiltIn::GlobalInvocationId,
            );
            bindings.push((
                *pid,
                ParamBinding::StageInput {
                    var,
                    value_ty: *pty,
                    index_var,
                    dispatch_var: kernel_grid_push_constant_var,
                },
            ));
            buffer_structs.push((var, block_ty));
        } else if role_is("texture") && is_array_texture {
            let (elem_image_ty, dim, arrayed, comp, multisampled, runtime_specialization) =
                if let Some((dim, arrayed, fmt, comp)) = wtex_dims.get(pid).copied() {
                    let metal_index = metal_texture_index
                        .ok_or("storage texture array has no Metal texture index")?;
                    let (fmt, state) =
                        ctx.specialize_storage_image_format(metal_index, fmt, comp)?;
                    (
                        ctx.ty_storage_image(dim, arrayed, fmt, comp),
                        dim,
                        arrayed,
                        comp,
                        false,
                        state.map(|state| (metal_index, state)),
                    )
                } else {
                    let shape = tex_dims
                        .get(pid)
                        .copied()
                        .or_else(|| texture_type_hints.get(pid).copied())
                        .unwrap_or(ImageShape {
                            dim: Dim::Dim2D,
                            arrayed: false,
                            comp: ImageComp::Float,
                            multisampled: false,
                        });
                    (
                        ctx.ty_image_ms(shape.dim, shape.arrayed, shape.comp, shape.multisampled),
                        shape.dim,
                        shape.arrayed,
                        shape.comp,
                        shape.multisampled,
                        None,
                    )
                };
            // The array length is a runtime function constant (`nImagesFC`), not a compile-time value,
            // so over-declare to `air.max_textures` (128). Vulkan lets an argument-buffer texture array
            // be over-sized; only accessed descriptors need be valid, and spirv-val does not bounds-check
            // a dynamic `OpAccessChain` index. A fixed `OpTypeArray` avoids the RuntimeDescriptorArray
            // capability; the index is `air.is_uniform`-marked, so no ShaderNonUniform is needed either.
            let array_ty = ctx.ty_array(
                elem_image_ty,
                crate::meta::TEXTURE_HANDLE_ARRAY_DESCRIPTOR_COUNT,
            );
            let pptr = ctx.ty_ptr(StorageClass::UniformConstant, array_ty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(pptr),
                Some(var),
                vec![Operand::StorageClass(StorageClass::UniformConstant)],
            ));
            let binding = required_resource_binding(
                *pid,
                if wtex_dims.contains_key(pid) {
                    storage_resource_binding
                } else {
                    resource_binding
                },
            )?;
            decorate_binding(&mut ctx.module, var, descriptor_layout.set, binding);
            ctx.interface_buffer_var(var);
            bindings.push((
                *pid,
                ParamBinding::ImageArray {
                    var,
                    elem_image_ty,
                    dim: (dim, arrayed),
                    comp,
                    multisampled,
                    runtime_specialization,
                },
            ));
        } else if role_is("texture") && wtex_dims.contains_key(pid) {
            // Write-only texture -> storage image (Sampled=2 + ImageFormat), lowered via OpImageWrite.
            let (dim, arrayed, fmt, comp) = wtex_dims
                .get(pid)
                .copied()
                .ok_or("write-texture dims missing for bound param")?;
            let metal_index =
                metal_texture_index.ok_or("storage texture has no Metal texture index")?;
            let (fmt, runtime_state) =
                ctx.specialize_storage_image_format(metal_index, fmt, comp)?;
            let image_ty = ctx.ty_storage_image(dim, arrayed, fmt, comp);
            let pptr = ctx.ty_ptr(StorageClass::UniformConstant, image_ty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(pptr),
                Some(var),
                vec![Operand::StorageClass(StorageClass::UniformConstant)],
            ));
            let binding = required_resource_binding(*pid, storage_resource_binding)?;
            decorate_binding(&mut ctx.module, var, descriptor_layout.set, binding);
            ctx.interface_buffer_var(var);
            bindings.push((
                *pid,
                ParamBinding::StorageImage {
                    var,
                    image_ty,
                    dim: (dim, arrayed),
                    comp,
                    runtime_specialization: runtime_state.map(|state| (metal_index, state)),
                },
            ));
        } else if role_is("texture") {
            let shape = tex_dims
                .get(pid)
                .copied()
                .or_else(|| texture_type_hints.get(pid).copied())
                .unwrap_or(ImageShape {
                    dim: Dim::Dim2D,
                    arrayed: false,
                    comp: ImageComp::Float,
                    multisampled: false,
                });
            let image_ty =
                ctx.ty_image_ms(shape.dim, shape.arrayed, shape.comp, shape.multisampled);
            let pptr = ctx.ty_ptr(StorageClass::UniformConstant, image_ty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(pptr),
                Some(var),
                vec![Operand::StorageClass(StorageClass::UniformConstant)],
            ));
            let binding = required_resource_binding(*pid, resource_binding)?;
            decorate_binding(&mut ctx.module, var, descriptor_layout.set, binding);
            ctx.interface_buffer_var(var); // SPIR-V 1.4+ lists every resource on the entry interface.
            bindings.push((
                *pid,
                ParamBinding::Image {
                    var,
                    image_ty,
                    dim: (shape.dim, shape.arrayed),
                    comp: shape.comp,
                    multisampled: shape.multisampled,
                },
            ));
        } else if let Some(color_index) = color_input_index(stage, frag, idx) {
            let (sampled_ty, read_ty) = input_attachment_read_types(ctx, &defs, *pty)?;
            let image_ty = ctx.ty_input_attachment(sampled_ty);
            let pptr = ctx.ty_ptr(StorageClass::UniformConstant, image_ty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(pptr),
                Some(var),
                vec![Operand::StorageClass(StorageClass::UniformConstant)],
            ));
            let binding = required_resource_binding(*pid, resource_binding)?;
            decorate_binding(&mut ctx.module, var, descriptor_layout.set, binding);
            decorate_input_attachment_index(&mut ctx.module, var, color_index);
            ctx.interface_buffer_var(var);
            bindings.push((
                *pid,
                ParamBinding::InputAttachment {
                    var,
                    image_ty,
                    read_ty,
                    param_ty: *pty,
                },
            ));
        } else if role_is("sampler") {
            let sty = ctx.ty_sampler();
            let pptr = ctx.ty_ptr(StorageClass::UniformConstant, sty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(pptr),
                Some(var),
                vec![Operand::StorageClass(StorageClass::UniformConstant)],
            ));
            let binding = required_resource_binding(*pid, resource_binding)?;
            decorate_binding(&mut ctx.module, var, descriptor_layout.set, binding);
            ctx.interface_buffer_var(var);
            let metal_index = match stage {
                Stage::Fragment => frag.and_then(|meta| match meta.role_of(idx) {
                    Some(FragRole::Sampler(index)) => Some(*index),
                    _ => None,
                }),
                Stage::Vertex => vert.and_then(|meta| match meta.role_of(idx) {
                    Some(VertRole::Sampler(index)) => Some(*index),
                    _ => None,
                }),
                Stage::Kernel => kern.and_then(|meta| match meta.role_of(idx) {
                    Some(KernRole::Sampler(index)) => Some(*index),
                    _ => None,
                }),
            }
            .ok_or_else(|| format!("sampler parameter {pid} has no Metal sampler index"))?;
            let specialized_state = usize::try_from(metal_index)
                .ok()
                .and_then(|index| ctx.runtime_sampler_states.get(index))
                .copied()
                .flatten()
                .map(RuntimeSamplerState::lowering_state);
            bindings.push((
                *pid,
                ParamBinding::Sampler {
                    var,
                    specialized_state,
                },
            ));
        } else if is_threadgroup_buffer {
            let pointee = ptr_pointee(&defs, *pty)
                .ok_or_else(|| format!("threadgroup param {pid} type {pty} is not a pointer"))?;
            let layout_ty = kern
                .and_then(|m| m.layout_of(idx))
                .map(|layout| build_workgroup_air_type(ctx, layout));
            let array_ty = if layout_ty.is_none() && is_raw_workgroup_array(&defs, pointee) {
                pointee
            } else {
                ctx.ty_array(layout_ty.unwrap_or(pointee), WORKGROUP_MEMORY_ELEMENTS)
            };
            let ptr_ty = ctx.ty_ptr(StorageClass::Workgroup, array_ty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(ptr_ty),
                Some(var),
                vec![Operand::StorageClass(StorageClass::Workgroup)],
            ));
            bindings.push((*pid, ParamBinding::WorkgroupMemory { var }));
        } else if role_is("buffer") {
            // pty is OpTypePointer UniformConstant %pointee. If pointee is a (heterogeneous) struct,
            // native emission kept it as a struct pointer — make a StorageBuffer var pointing at it and the
            // body's member access chains just work. If pointee is a bare scalar/vector (a genuine
            // `device float4*` array, or a homogeneous struct emitted as a `T*` that the body
            // indexes `[i]`), physical array-stride access is illegal in Logical SPIR-V: wrap it as a
            // Block `{ RuntimeArray<pointee> }` and remember the element type so the body's param uses
            // get rewritten (member-0 prepend on chains; &buf[0] for direct loads).
            let pointee = ptr_pointee(&defs, *pty)
                .ok_or_else(|| format!("buffer param {pid} type {pty} is not a pointer"))?;
            let pointee_op = defs.get(&pointee).map(|d| d.class.opcode);
            let is_struct = pointee_op == Some(Op::TypeStruct);
            let is_raw_uint_block = is_raw_uint_buffer_block(&defs, pointee);
            // A scalar pointee means native emission flattened the buffer to a flat scalar array and the
            // access chain indices are flat element offsets, NOT struct-navigation indices — so it must
            // use the RuntimeArray wrapping, not struct reconstruction (which would mis-index). A vector
            // pointee means native emission preserved the struct shape (multi-level nav), so reconstruct.
            let pointee_is_scalar =
                matches!(pointee_op, Some(Op::TypeFloat | Op::TypeInt | Op::TypeBool));
            // The AIR struct layout for this buffer (present iff it carries `air.struct_type_info`).
            let layout = match stage {
                Stage::Fragment => frag.and_then(|m| m.layout_of(idx)),
                Stage::Vertex => vert.and_then(|m| m.layout_of(idx)),
                Stage::Kernel => kern.and_then(|m| m.layout_of(idx)),
            };
            let primitive_buffer_air_type = match stage {
                Stage::Kernel => kern
                    .and_then(|m| m.buffer_type_name(idx))
                    .and_then(primitive_air_type_from_name),
                Stage::Fragment | Stage::Vertex => None,
            };
            let mut typed_alias_elements = Vec::new();
            let (struct_ty, wrap) = if is_raw_uint_block {
                let carries_indirect_arguments = kern.is_some_and(|meta| {
                    meta.embedded_arguments
                        .iter()
                        .any(|argument| argument.buffer_param_index == idx)
                });
                if carries_indirect_arguments {
                    // An AIR indirect buffer physically contains opaque 64-bit resource handles.
                    // When the emitter selected its raw-word transport, retain that representation:
                    // rebuilding the source struct would replace handle slots with their pointee
                    // types, making the two payload words impossible to address or populate.
                    (pointee, BufWrap::Direct)
                } else if let Some(at) = layout
                    .filter(|_| buffer_has_access_chains(&ctx.module.functions[entry_idx], *pid))
                {
                    let st = ctx.build_air_type(at);
                    let mut layout_defs = defs.clone();
                    for g in &ctx.new_globals {
                        if let Some(id) = g.result_id {
                            layout_defs.entry(id).or_insert_with(|| g.clone());
                        }
                    }
                    let all_chains_match_struct = buffer_access_chains_match_struct_path(
                        &layout_defs,
                        &ctx.module.functions[entry_idx],
                        *pid,
                        st,
                    );
                    let has_struct_chain = all_chains_match_struct
                        || buffer_has_struct_path_access_chain(
                            &layout_defs,
                            &ctx.module.functions[entry_idx],
                            *pid,
                            st,
                        );
                    let flat_scalar_element = body_buf_flat_scalar_element_type(
                        ctx,
                        &ctx.module.functions[entry_idx],
                        *pid,
                    );
                    if flat_scalar_element.is_some()
                        || ctx.emit_sidecar.all_device_buffers_raw
                        || ctx.emit_sidecar.flat_raw_buffer_params.contains(&idx)
                    {
                        // A proven `[0, scalar-index]` view must remain the native raw-word block:
                        // its index can be dynamic, while SPIR-V struct-member indices cannot. The
                        // same representation is required when the producer selected a raw
                        // interface because no single storage-compatible aggregate exists. Preserve
                        // it even if another endpoint happens to match the AIR metadata aggregate.
                        (
                            pointee,
                            BufWrap::Collapsed {
                                block_ty: pointee,
                                prepend_member0: !access_chains_include_wrapper_member0(
                                    &defs,
                                    &ctx.module.functions[entry_idx],
                                    *pid,
                                ),
                                typed_aliases: vec![],
                            },
                        )
                    } else if has_struct_chain {
                        (
                            st,
                            BufWrap::Collapsed {
                                block_ty: st,
                                prepend_member0: false,
                                typed_aliases: vec![],
                            },
                        )
                    } else {
                        (pointee, BufWrap::Direct)
                    }
                } else {
                    // Native raw-buffer params are already `{ RuntimeArray<uint> }` transport blocks.
                    // Use that block directly; wrapping it as an array element would create an illegal
                    // RuntimeArray<Block>.
                    (pointee, BufWrap::Direct)
                }
            } else if is_struct
                && struct_buffer_needs_record_array(
                    &defs,
                    &ctx.module.functions[entry_idx],
                    *pid,
                    pointee,
                )
            {
                // Native emission kept the real struct type, but AIR uses the first GEP index as an implicit
                // record index (`buffer[N].field`). Wrap the struct in a runtime array so nonzero or
                // dynamic record indices stay legal under Logical SPIR-V.
                let elem = ctx.clone_type_for_record_array_element(pointee, &defs);
                let rta = ctx.ty_runtime_array(elem);
                let st = ctx.module.fresh_id();
                ctx.new_globals
                    .push(type_inst(Op::TypeStruct, st, vec![Operand::IdRef(rta)]));
                (
                    st,
                    BufWrap::RecordArray {
                        block_ty: st,
                        elem_ty: elem,
                    },
                )
            } else if is_struct {
                // Native emission kept the real struct and AIR only indexes record 0; index it off the var
                // directly.
                (pointee, BufWrap::Direct)
            } else if let Some(at) = layout {
                // Native emission represents a structured AIR buffer as a bare pointer. If the body has
                // access chains that type-check against the original AIR struct layout, preserve that
                // layout. This includes scalar first fields (`buf.field0`) whose SPIR-V shape is also a
                // single-index chain. Only fall back to RuntimeArray when the chains do not match the
                // struct and the body is reading flat `buf[i]` elements.
                let has_access_chains =
                    buffer_has_access_chains(&ctx.module.functions[entry_idx], *pid);
                let flat_elem = pointee_is_scalar
                    .then(|| body_buf_elem_type(ctx, &ctx.module.functions[entry_idx], *pid))
                    .flatten()
                    .or_else(|| {
                        body_buf_flat_scalar_element_type(
                            ctx,
                            &ctx.module.functions[entry_idx],
                            *pid,
                        )
                    });
                if !has_access_chains {
                    let rta = ctx.ty_runtime_array(pointee);
                    let st = ctx.module.fresh_id();
                    ctx.new_globals
                        .push(type_inst(Op::TypeStruct, st, vec![Operand::IdRef(rta)]));
                    (
                        st,
                        BufWrap::Collapsed {
                            block_ty: st,
                            prepend_member0: true,
                            typed_aliases: vec![],
                        },
                    )
                } else {
                    let st = ctx.build_air_type(at);
                    let mut layout_defs = defs.clone();
                    for g in &ctx.new_globals {
                        if let Some(id) = g.result_id {
                            layout_defs.entry(id).or_insert_with(|| g.clone());
                        }
                    }
                    if let Some(elem) = flat_elem {
                        let already_has_wrapper_member0 = access_chains_include_wrapper_member0(
                            &defs,
                            &ctx.module.functions[entry_idx],
                            *pid,
                        );
                        let rta = ctx.ty_runtime_array(elem);
                        let st = ctx.module.fresh_id();
                        ctx.new_globals.push(type_inst(
                            Op::TypeStruct,
                            st,
                            vec![Operand::IdRef(rta)],
                        ));
                        (
                            st,
                            BufWrap::Collapsed {
                                block_ty: st,
                                prepend_member0: !already_has_wrapper_member0,
                                typed_aliases: vec![],
                            },
                        )
                    } else if buffer_access_chains_match_struct_path(
                        &layout_defs,
                        &ctx.module.functions[entry_idx],
                        *pid,
                        st,
                    ) {
                        (
                            st,
                            BufWrap::Collapsed {
                                block_ty: st,
                                prepend_member0: false,
                                typed_aliases: vec![],
                            },
                        )
                    } else if !pointee_is_scalar
                        && buffer_has_multi_index_access_chains(
                            &ctx.module.functions[entry_idx],
                            *pid,
                        )
                        && struct_buffer_needs_record_array(
                            &layout_defs,
                            &ctx.module.functions[entry_idx],
                            *pid,
                            st,
                        )
                    {
                        let rta = ctx.ty_runtime_array(st);
                        let block = ctx.module.fresh_id();
                        ctx.new_globals.push(type_inst(
                            Op::TypeStruct,
                            block,
                            vec![Operand::IdRef(rta)],
                        ));
                        (
                            block,
                            BufWrap::RecordArray {
                                block_ty: block,
                                elem_ty: st,
                            },
                        )
                    } else {
                        (
                            st,
                            BufWrap::Collapsed {
                                block_ty: st,
                                prepend_member0: false,
                                typed_aliases: vec![],
                            },
                        )
                    }
                }
            } else {
                // Genuine `device T*` array (no struct info): wrap as `{ RuntimeArray<T> }`. Source
                // canonicalization can represent this as `{[0 x T]}` while the emitted parameter is
                // `T*`; accesses may therefore retain the wrapper's member-0 index as
                // `%p %uint_0 %i`. Re-root those without prepending; all other array/vector element
                // chains still need the StorageBuffer block member inserted.
                if let Some(air_ty) = primitive_buffer_air_type.as_ref().filter(|air_ty| {
                    !matches!(**air_ty, AirType::Scalar(_))
                        && buffer_has_multi_index_access_chains(
                            &ctx.module.functions[entry_idx],
                            *pid,
                        )
                }) {
                    let elem = ctx.build_air_type(air_ty);
                    let rta = ctx.ty_runtime_array(elem);
                    let st = ctx.module.fresh_id();
                    ctx.new_globals
                        .push(type_inst(Op::TypeStruct, st, vec![Operand::IdRef(rta)]));
                    (
                        st,
                        BufWrap::RecordArray {
                            block_ty: st,
                            elem_ty: elem,
                        },
                    )
                } else {
                    let already_has_wrapper_member0 = access_chains_include_wrapper_member0(
                        &defs,
                        &ctx.module.functions[entry_idx],
                        *pid,
                    );
                    // An entry buffer parameter can be a bare `uchar*` even though the inlined body
                    // indexes it as a `float`/`v2float` array — the
                    // RuntimeArray element must match what the body READS, not the mistyped pointee
                    // (else the chain indexes a uchar array but loads a float). Prefer the body's
                    // single-index element type.
                    let body_elements =
                        body_buf_elem_types(ctx, &ctx.module.functions[entry_idx], *pid);
                    let elem = body_elements.first().copied().unwrap_or(pointee);
                    if body_elements.len() > 1
                        && body_elements
                            .iter()
                            .all(|element| buffer_typed_alias_element(&defs, *element))
                    {
                        typed_alias_elements = body_elements;
                    }
                    let rta = ctx.ty_runtime_array(elem);
                    let st = ctx.module.fresh_id();
                    ctx.new_globals
                        .push(type_inst(Op::TypeStruct, st, vec![Operand::IdRef(rta)]));
                    (
                        st,
                        BufWrap::Collapsed {
                            block_ty: st,
                            prepend_member0: !already_has_wrapper_member0,
                            typed_aliases: vec![],
                        },
                    )
                }
            };
            let source_layout_ty = layout.map(|air_layout| ctx.build_air_type(air_layout));
            let uptr = ctx.ty_ptr(StorageClass::StorageBuffer, struct_ty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(uptr),
                Some(var),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ));
            if let Some(source_layout_ty) = source_layout_ty.filter(|_| {
                single_member_array_scalar_elem(ctx, struct_ty)
                    .is_some_and(|element| is_unsigned_byte_scalar(ctx, element))
            }) {
                ctx.emit_sidecar
                    .buffer_root_source_types
                    .insert(var, source_layout_ty);
            }
            let binding = required_resource_binding(*pid, resource_binding)?;
            decorate_binding(&mut ctx.module, var, descriptor_layout.set, binding);
            let mut wrap = wrap;
            if let BufWrap::Collapsed { typed_aliases, .. } = &mut wrap {
                let primary_element = single_member_array_scalar_elem(ctx, struct_ty);
                for element in typed_alias_elements
                    .into_iter()
                    .filter(|element| Some(*element) != primary_element)
                {
                    let runtime_array = ctx.ty_runtime_array(element);
                    let alias_block = ctx.module.fresh_id();
                    ctx.new_globals.push(type_inst(
                        Op::TypeStruct,
                        alias_block,
                        vec![Operand::IdRef(runtime_array)],
                    ));
                    let alias_pointer = ctx.ty_ptr(StorageClass::StorageBuffer, alias_block);
                    let alias_var = ctx.module.fresh_id();
                    ctx.new_globals.push(Instruction::new(
                        Op::Variable,
                        Some(alias_pointer),
                        Some(alias_var),
                        vec![Operand::StorageClass(StorageClass::StorageBuffer)],
                    ));
                    decorate_binding(&mut ctx.module, alias_var, descriptor_layout.set, binding);
                    typed_aliases.push((element, alias_var));
                    buffer_structs.push((alias_var, alias_block));
                }
            }
            bindings.push((*pid, ParamBinding::Buffer { var, wrap }));
            buffer_structs.push((var, struct_ty));
        } else if role_is("varying") {
            if matches!(stage, Stage::Fragment) && is_scalar_bool(&defs, *pty) {
                let uint_ty = ctx.ty_uint();
                let pptr = ctx.ty_ptr(StorageClass::Input, uint_ty);
                let var = ctx.module.fresh_id();
                ctx.new_globals.push(Instruction::new(
                    Op::Variable,
                    Some(pptr),
                    Some(var),
                    vec![Operand::StorageClass(StorageClass::Input)],
                ));
                decorate_location(&mut ctx.module, var, loc);
                decorate_flat(&mut ctx.module, var);
                ctx.interface.push(var);
                bindings.push((
                    *pid,
                    ParamBinding::LoadVarBoolFromUint { var, bool_ty: *pty },
                ));
            } else if matches!(stage, Stage::Fragment) && type_contains_bool(&defs, *pty) {
                return Err(format!(
                    "fragment bool stage input at location {loc} is unsupported: Vulkan user \
                     Input/Output interfaces cannot use OpTypeBool"
                ));
            } else {
                let interface_ty = match stage {
                    Stage::Fragment => fragment_varying_interface_type(ctx, frag, loc, *pty, &defs),
                    Stage::Vertex => vertex_attribute_interface_type(ctx, vert, loc, *pty, &defs),
                    Stage::Kernel => *pty,
                };
                // Input var of the param value type at Location loc; load at entry.
                let pptr = ctx.ty_ptr(StorageClass::Input, interface_ty);
                let var = ctx.module.fresh_id();
                ctx.new_globals.push(Instruction::new(
                    Op::Variable,
                    Some(pptr),
                    Some(var),
                    vec![Operand::StorageClass(StorageClass::Input)],
                ));
                decorate_location(&mut ctx.module, var, loc);
                // A fragment input carries the interpolation attribute AIR declared for it, or
                // `Flat` when Vulkan forbids interpolating its scalar type at all (integer /
                // 64-bit float; VUID-StandaloneSpirv-Flat-04744).
                if matches!(stage, Stage::Fragment) {
                    decorate_interpolation(
                        &mut ctx.module,
                        var,
                        frag.map(|m| m.varying_interpolation(loc))
                            .unwrap_or_default(),
                        fragment_input_needs_flat(&defs, *pty),
                    );
                }
                ctx.interface.push(var);
                if interface_ty == *pty {
                    bindings.push((*pid, ParamBinding::LoadVar { var, ty: *pty }));
                } else if matches!(stage, Stage::Vertex)
                    && (type_int_shape(&defs, *pty).is_some()
                        || defs.get(pty).is_some_and(|definition| {
                            definition.class.opcode == Op::TypeVector
                                && definition
                                    .operands
                                    .first()
                                    .and_then(|operand| match operand {
                                        Operand::IdRef(element) => Some(*element),
                                        _ => None,
                                    })
                                    .is_some_and(|element| type_int_shape(&defs, element).is_some())
                        }))
                {
                    let binding = if matches!(
                        (
                            integer_component_width(ctx, interface_ty),
                            integer_component_width(ctx, *pty)
                        ),
                        (Some(interface_bits), Some(param_bits)) if interface_bits == param_bits
                    ) {
                        // AIR integers are signless in the function body, while vertex metadata
                        // carries the fetch format's signedness. Equal-width representations need
                        // only preserve their bits; SPIR-V forbids integer conversion when the
                        // component widths are equal.
                        ParamBinding::LoadVarBitcast {
                            var,
                            load_ty: interface_ty,
                            param_ty: *pty,
                        }
                    } else {
                        ParamBinding::LoadVarConverted {
                            var,
                            load_ty: interface_ty,
                            param_ty: *pty,
                        }
                    };
                    bindings.push((*pid, binding));
                } else {
                    bindings.push((
                        *pid,
                        ParamBinding::LoadVarBitcast {
                            var,
                            load_ty: interface_ty,
                            param_ty: *pty,
                        },
                    ));
                }
            }
        } else if role_is("viewport_array_index") {
            let uint_ty = ctx.ty_uint();
            let pptr = ctx.ty_ptr(StorageClass::Input, uint_ty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(pptr),
                Some(var),
                vec![Operand::StorageClass(StorageClass::Input)],
            ));
            decorate_builtin(&mut ctx.module, var, BuiltIn::ViewportIndex);
            decorate_flat(&mut ctx.module, var);
            ctx.interface.push(var);
            if *pty == uint_ty {
                bindings.push((*pid, ParamBinding::LoadVar { var, ty: uint_ty }));
            } else {
                bindings.push((
                    *pid,
                    ParamBinding::LoadVarConverted {
                        var,
                        load_ty: uint_ty,
                        param_ty: *pty,
                    },
                ));
            }
        } else if role_is("render_target_array_index") {
            let uint_ty = ctx.ty_uint();
            let var = bind_kernel_uint_builtin_once(ctx, &mut layer_var, BuiltIn::Layer);
            decorate_flat(&mut ctx.module, var);
            if *pty == uint_ty {
                bindings.push((*pid, ParamBinding::LoadVar { var, ty: uint_ty }));
            } else {
                bindings.push((
                    *pid,
                    ParamBinding::LoadVarConverted {
                        var,
                        load_ty: uint_ty,
                        param_ty: *pty,
                    },
                ));
            }
        } else if role_is("patch_input") {
            let location = match vert.and_then(|meta| meta.role_of(idx)) {
                Some(VertRole::PatchInput(location)) => *location,
                _ => unreachable!("patch_input role has a location"),
            };
            let pptr = ctx.ty_ptr(StorageClass::Input, *pty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(pptr),
                Some(var),
                vec![Operand::StorageClass(StorageClass::Input)],
            ));
            decorate_location(&mut ctx.module, var, location);
            decorate_patch(&mut ctx.module, var);
            ctx.interface.push(var);
            bindings.push((*pid, ParamBinding::LoadVar { var, ty: *pty }));
        } else if role_is("position_in_patch") {
            let vec_ty = ctx.ty_vecf(3);
            let var = if let Some(var) = tess_coord_var {
                var
            } else {
                let pptr = ctx.ty_ptr(StorageClass::Input, vec_ty);
                let var = ctx.module.fresh_id();
                ctx.new_globals.push(Instruction::new(
                    Op::Variable,
                    Some(pptr),
                    Some(var),
                    vec![Operand::StorageClass(StorageClass::Input)],
                ));
                decorate_builtin(&mut ctx.module, var, BuiltIn::TessCoord);
                ctx.interface.push(var);
                tess_coord_var = Some(var);
                var
            };
            let binding = match tess_coord_prefix_lanes(&defs, *pty, ctx.ty_float())? {
                None => ParamBinding::LoadVar { var, ty: vec_ty },
                Some(lanes) => ParamBinding::LoadVarVectorPrefix {
                    var,
                    vec_ty,
                    scalar_ty: ctx.ty_float(),
                    prefix_ty: *pty,
                    out_ty: *pty,
                    lanes,
                },
            };
            bindings.push((*pid, binding));
        } else if role_is("patch_id") {
            let uint_ty = ctx.ty_uint();
            let var =
                bind_kernel_uint_builtin_once(ctx, &mut primitive_id_var, BuiltIn::PrimitiveId);
            if *pty == uint_ty {
                bindings.push((*pid, ParamBinding::LoadVar { var, ty: uint_ty }));
            } else {
                bindings.push((
                    *pid,
                    ParamBinding::LoadVarConverted {
                        var,
                        load_ty: uint_ty,
                        param_ty: *pty,
                    },
                ));
            }
        } else if matches!(stage, Stage::Vertex)
            && vert.is_some_and(VertMeta::is_tessellation_evaluation)
            && (role_is("instance_id")
                || role_is("amplification_id")
                || role_is("amplification_count"))
        {
            let role = vert
                .and_then(|meta| meta.role_of(idx))
                .expect("decoded role");
            let location = vert
                .and_then(|meta| meta.tessellation_system_input_location(role))
                .expect("tessellation system input location");
            let pptr = ctx.ty_ptr(StorageClass::Input, *pty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(pptr),
                Some(var),
                vec![Operand::StorageClass(StorageClass::Input)],
            ));
            decorate_location(&mut ctx.module, var, location);
            decorate_patch(&mut ctx.module, var);
            ctx.interface.push(var);
            bindings.push((*pid, ParamBinding::LoadVar { var, ty: *pty }));
        } else if role_is("vertex_id") || role_is("instance_id") {
            // `[[vertex_id]]`/`[[instance_id]]` -> Input BuiltIn VertexIndex/InstanceIndex. Vulkan
            // requires this builtin to be a 32-bit int Input; the AIR `uint` param lowers to `%uint`,
            // which we use as both the var type and the load type. The body then reads the loaded value
            // (and may OpUConvert it to ulong before indexing — left intact). Without this the param
            // would fall through to ZeroValue/OpUndef and every vertex would read the same LUT entry.
            let builtin = if role_is("vertex_id") {
                BuiltIn::VertexIndex
            } else {
                BuiltIn::InstanceIndex
            };
            let uint_ty = ctx.ty_uint();
            // The builtin var is always a 32-bit uint (Vulkan's VertexIndex/InstanceIndex type). If the
            // AIR param is a NARROWER int (`ushort`), load uint then UConvert down to the param width.
            let pptr = ctx.ty_ptr(StorageClass::Input, uint_ty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(pptr),
                Some(var),
                vec![Operand::StorageClass(StorageClass::Input)],
            ));
            decorate_builtin(&mut ctx.module, var, builtin);
            ctx.interface.push(var);
            if *pty == uint_ty {
                bindings.push((*pid, ParamBinding::LoadVar { var, ty: uint_ty }));
            } else {
                bindings.push((
                    *pid,
                    ParamBinding::LoadVarConverted {
                        var,
                        load_ty: uint_ty,
                        param_ty: *pty,
                    },
                ));
            }
        } else if role_is("threads_per_threadgroup") {
            let lanes = scalar_or_vector_component(&defs, *pty)
                .and_then(|(_, lanes)| lanes)
                .unwrap_or(1);
            bindings.push((
                *pid,
                ParamBinding::LoadKernelLocalSize {
                    out_ty: *pty,
                    lanes,
                },
            ));
        } else if role_is("thread_position_in_threadgroup") {
            bind_kernel_uvec3_builtin(
                ctx,
                &defs,
                &mut bindings,
                *pid,
                *pty,
                BuiltIn::LocalInvocationId,
            );
        } else if role_is("threadgroups_per_grid") {
            if matches!(
                ctx.kernel_dispatch,
                crate::reflect::KernelDispatch::Workgroups
            ) {
                let var = bind_kernel_v3uint_builtin_once(
                    ctx,
                    &mut num_workgroups_var,
                    BuiltIn::NumWorkgroups,
                );
                bind_kernel_uvec3_builtin_var(ctx, &defs, &mut bindings, *pid, *pty, var);
            } else {
                let var = kernel_grid_push_constant_var.expect("exact dispatch payload bound");
                let lanes = scalar_or_vector_component(&defs, *pty)
                    .and_then(|(_, lanes)| lanes)
                    .unwrap_or(1);
                bindings.push((
                    *pid,
                    ParamBinding::LoadKernelDispatchField {
                        var,
                        first_member: 9,
                        out_ty: *pty,
                        lanes,
                    },
                ));
            }
        } else if role_is("threads_per_grid") {
            match ctx.kernel_dispatch {
                crate::reflect::KernelDispatch::ThreadsFixed { .. }
                | crate::reflect::KernelDispatch::ThreadsDynamic { .. } => {
                    let var = kernel_grid_push_constant_var.expect("exact dispatch payload bound");
                    let lanes = scalar_or_vector_component(&defs, *pty)
                        .and_then(|(_, lanes)| lanes)
                        .unwrap_or(1);
                    bindings.push((
                        *pid,
                        ParamBinding::LoadKernelDispatchField {
                            var,
                            first_member: 0,
                            out_ty: *pty,
                            lanes,
                        },
                    ));
                }
                crate::reflect::KernelDispatch::Workgroups => {
                    let var = bind_kernel_v3uint_builtin_once(
                        ctx,
                        &mut num_workgroups_var,
                        BuiltIn::NumWorkgroups,
                    );
                    bind_kernel_threads_per_grid(ctx, &defs, &mut bindings, *pid, *pty, var);
                }
            }
        } else if role_is("threadgroup_position_in_grid") {
            if matches!(
                ctx.kernel_dispatch,
                crate::reflect::KernelDispatch::Workgroups
            ) {
                bind_kernel_uvec3_builtin(
                    ctx,
                    &defs,
                    &mut bindings,
                    *pid,
                    *pty,
                    BuiltIn::WorkgroupId,
                );
            } else {
                let builtin_var = bind_kernel_v3uint_builtin(ctx, BuiltIn::WorkgroupId);
                let dispatch_var =
                    kernel_grid_push_constant_var.expect("exact dispatch payload bound");
                let lanes = scalar_or_vector_component(&defs, *pty)
                    .and_then(|(_, lanes)| lanes)
                    .unwrap_or(1);
                bindings.push((
                    *pid,
                    ParamBinding::LoadBuiltinPlusKernelDispatchField {
                        builtin_var,
                        dispatch_var,
                        first_member: 6,
                        out_ty: *pty,
                        lanes,
                    },
                ));
            }
        } else if role_is("thread_index_in_threadgroup") {
            let uint_ty = ctx.ty_uint();
            let var = bind_kernel_uint_builtin_once(
                ctx,
                &mut local_invocation_index_var,
                BuiltIn::LocalInvocationIndex,
            );
            if *pty == uint_ty {
                bindings.push((*pid, ParamBinding::LoadVar { var, ty: uint_ty }));
            } else {
                bindings.push((
                    *pid,
                    ParamBinding::LoadVarConverted {
                        var,
                        load_ty: uint_ty,
                        param_ty: *pty,
                    },
                ));
            }
        } else if role_is("thread_index_in_quadgroup") {
            let uint_ty = ctx.ty_uint();
            let var = bind_kernel_uint_builtin_once(
                ctx,
                &mut local_invocation_index_var,
                BuiltIn::LocalInvocationIndex,
            );
            bindings.push((
                *pid,
                ParamBinding::LoadVarBitAnd {
                    var,
                    load_ty: uint_ty,
                    param_ty: *pty,
                    mask: 3,
                },
            ));
        } else if role_is("quadgroup_index_in_threadgroup") {
            let uint_ty = ctx.ty_uint();
            let var = bind_kernel_uint_builtin_once(
                ctx,
                &mut local_invocation_index_var,
                BuiltIn::LocalInvocationIndex,
            );
            bindings.push((
                *pid,
                ParamBinding::LoadVarShiftRight {
                    var,
                    load_ty: uint_ty,
                    param_ty: *pty,
                    shift: 2,
                },
            ));
        } else if role_is("thread_index_in_simdgroup") {
            let uint_ty = ctx.ty_uint();
            let var = bind_kernel_uint_builtin_once(
                ctx,
                &mut local_invocation_index_var,
                BuiltIn::LocalInvocationIndex,
            );
            bindings.push((
                *pid,
                ParamBinding::LoadVarBitAnd {
                    var,
                    load_ty: uint_ty,
                    param_ty: *pty,
                    mask: 31,
                },
            ));
        } else if role_is("simdgroup_index_in_threadgroup") {
            let uint_ty = ctx.ty_uint();
            let var = bind_kernel_uint_builtin_once(
                ctx,
                &mut local_invocation_index_var,
                BuiltIn::LocalInvocationIndex,
            );
            bindings.push((
                *pid,
                ParamBinding::LoadVarShiftRight {
                    var,
                    load_ty: uint_ty,
                    param_ty: *pty,
                    shift: 5,
                },
            ));
        } else if role_is("threads_per_simdgroup") {
            let val = const_kernel_local_size(ctx, &defs, *pty, [32, 1, 1])
                .unwrap_or_else(|| ctx.const_uint(32));
            bindings.push((*pid, ParamBinding::Value { val }));
        } else if role_is("simdgroups_per_threadgroup") {
            bindings.push((
                *pid,
                ParamBinding::LoadKernelSimdgroupsPerThreadgroup { out_ty: *pty },
            ));
        } else if role_is("thread_position_in_grid") {
            let var = bind_kernel_v3uint_builtin_once(
                ctx,
                &mut global_invocation_id_var,
                BuiltIn::GlobalInvocationId,
            );
            if matches!(
                ctx.kernel_dispatch,
                crate::reflect::KernelDispatch::Workgroups
            ) {
                bind_kernel_uvec3_builtin_var(ctx, &defs, &mut bindings, *pid, *pty, var);
            } else {
                let dispatch_var =
                    kernel_grid_push_constant_var.expect("exact dispatch payload bound");
                let lanes = scalar_or_vector_component(&defs, *pty)
                    .and_then(|(_, lanes)| lanes)
                    .unwrap_or(1);
                bindings.push((
                    *pid,
                    ParamBinding::LoadBuiltinPlusKernelDispatchField {
                        builtin_var: var,
                        dispatch_var,
                        first_member: 3,
                        out_ty: *pty,
                        lanes,
                    },
                ));
            }
        } else if role_is("position") {
            // FragCoord builtin input (vec4). Often unused; bind a load if the type is vec4, else zero.
            let v4 = ctx.ty_vecf(4);
            if *pty == v4 {
                let var = if let Some(v) = fragcoord_var {
                    v // reuse the single FragCoord var (duplicate builtins are illegal)
                } else {
                    let pptr = ctx.ty_ptr(StorageClass::Input, v4);
                    let var = ctx.module.fresh_id();
                    ctx.new_globals.push(Instruction::new(
                        Op::Variable,
                        Some(pptr),
                        Some(var),
                        vec![Operand::StorageClass(StorageClass::Input)],
                    ));
                    decorate_builtin(&mut ctx.module, var, BuiltIn::FragCoord);
                    ctx.interface.push(var);
                    fragcoord_var = Some(var);
                    var
                };
                bindings.push((*pid, ParamBinding::LoadVar { var, ty: v4 }));
            } else {
                let z = ctx.const_zero(*pty, &defs);
                bindings.push((*pid, ParamBinding::ZeroValue { val: z }));
            }
        } else if role_is("point_coord") {
            let v2 = ctx.ty_vecf(2);
            if *pty == v2 {
                let var = if let Some(v) = pointcoord_var {
                    v
                } else {
                    let pptr = ctx.ty_ptr(StorageClass::Input, v2);
                    let var = ctx.module.fresh_id();
                    ctx.new_globals.push(Instruction::new(
                        Op::Variable,
                        Some(pptr),
                        Some(var),
                        vec![Operand::StorageClass(StorageClass::Input)],
                    ));
                    decorate_builtin(&mut ctx.module, var, BuiltIn::PointCoord);
                    ctx.interface.push(var);
                    pointcoord_var = Some(var);
                    var
                };
                bindings.push((*pid, ParamBinding::LoadVar { var, ty: v2 }));
            } else {
                let z = ctx.const_zero(*pty, &defs);
                bindings.push((*pid, ParamBinding::ZeroValue { val: z }));
            }
        } else if role_is("front_facing") {
            let bool_ty = ctx.ty_bool();
            if *pty == bool_ty {
                let var = if let Some(v) = front_facing_var {
                    v
                } else {
                    let pptr = ctx.ty_ptr(StorageClass::Input, bool_ty);
                    let var = ctx.module.fresh_id();
                    ctx.new_globals.push(Instruction::new(
                        Op::Variable,
                        Some(pptr),
                        Some(var),
                        vec![Operand::StorageClass(StorageClass::Input)],
                    ));
                    decorate_builtin(&mut ctx.module, var, BuiltIn::FrontFacing);
                    ctx.interface.push(var);
                    front_facing_var = Some(var);
                    var
                };
                bindings.push((*pid, ParamBinding::LoadVar { var, ty: bool_ty }));
            } else {
                let z = ctx.const_zero(*pty, &defs);
                bindings.push((*pid, ParamBinding::ZeroValue { val: z }));
            }
        } else if role_is("primitive_id") {
            let uint_ty = ctx.ty_uint();
            let var =
                bind_kernel_uint_builtin_once(ctx, &mut primitive_id_var, BuiltIn::PrimitiveId);
            decorate_flat(&mut ctx.module, var);
            if *pty == uint_ty {
                bindings.push((*pid, ParamBinding::LoadVar { var, ty: uint_ty }));
            } else {
                bindings.push((
                    *pid,
                    ParamBinding::LoadVarConverted {
                        var,
                        load_ty: uint_ty,
                        param_ty: *pty,
                    },
                ));
            }
        } else if role_is("sample_id") {
            let uint_ty = ctx.ty_uint();
            let var = bind_kernel_uint_builtin_once(ctx, &mut sample_id_var, BuiltIn::SampleId);
            decorate_flat(&mut ctx.module, var);
            if *pty == uint_ty {
                bindings.push((*pid, ParamBinding::LoadVar { var, ty: uint_ty }));
            } else {
                bindings.push((
                    *pid,
                    ParamBinding::LoadVarConverted {
                        var,
                        load_ty: uint_ty,
                        param_ty: *pty,
                    },
                ));
            }
        } else if role_is("imageblock_data") {
            let imageblock = frag
                .and_then(|meta| meta.fragment_imageblock.as_ref())
                .ok_or_else(|| {
                    format!(
                        "fragment imageblock parameter {idx} has no decoded AIR layout contract"
                    )
                })?;
            let projection = imageblock
                .inputs
                .iter()
                .find(|projection| projection.interface_index == idx)
                .ok_or_else(|| format!("fragment imageblock parameter {idx} has no projection"))?;
            let projected_types = defs
                .get(pty)
                .filter(|definition| definition.class.opcode == Op::TypeStruct)
                .ok_or_else(|| {
                    format!("fragment imageblock parameter {idx} is not a struct value")
                })?
                .operands
                .iter()
                .filter_map(|operand| match operand {
                    Operand::IdRef(ty) => Some(*ty),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if projected_types.len() != projection.members.len() {
                return Err(format!(
                    "fragment imageblock parameter {idx} exposes {} fields but AIR projects {}",
                    projected_types.len(),
                    projection.members.len()
                ));
            }
            let coord_var = if let Some(var) = fragcoord_var {
                var
            } else {
                let coord_ty = ctx.ty_vecf(4);
                let pointer_ty = ctx.ty_ptr(StorageClass::Input, coord_ty);
                let var = ctx.module.fresh_id();
                ctx.new_globals.push(Instruction::new(
                    Op::Variable,
                    Some(pointer_ty),
                    Some(var),
                    vec![Operand::StorageClass(StorageClass::Input)],
                ));
                decorate_builtin(&mut ctx.module, var, BuiltIn::FragCoord);
                ctx.interface.push(var);
                fragcoord_var = Some(var);
                ctx.fragment_imageblock_coord_var = Some(var);
                var
            };
            let mut members = Vec::with_capacity(projection.members.len());
            for (projected_ty, projected) in
                projected_types.into_iter().zip(projection.members.iter())
            {
                let master = imageblock
                    .members
                    .get(projected.master_member as usize)
                    .ok_or_else(|| {
                        format!(
                            "fragment imageblock parameter {idx} references missing master member {}",
                            projected.master_member
                        )
                    })?;
                let format = fragment_imageblock_format(&master.type_name).ok_or_else(|| {
                    format!(
                        "fragment imageblock parameter {idx} member {} has unsupported master type {}",
                        projected.projection_member, master.type_name
                    )
                })?;
                if !fragment_imageblock_projection_type_matches(&defs, projected_ty, format) {
                    return Err(format!(
                        "fragment imageblock parameter {idx} member {} does not match AIR master type {}",
                        projected.projection_member, master.type_name
                    ));
                }
                let (image_var, image_ty) =
                    ctx.fragment_imageblock_var(projected.master_member, &master.type_name)?;
                members.push((image_var, image_ty, projected_ty, format));
            }
            bindings.push((
                *pid,
                ParamBinding::FragmentImageblockProjection {
                    coord_var,
                    param_ty: *pty,
                    members,
                },
            ));
        } else if let Some(pointee) = data_pointer_pointee(&defs, *pty) {
            // Unmodeled *pointer* param (an unbound `constant T&`/buffer that no role recognized). An
            // OpUndef of a data-pointer type would be dereferenced by the body's OpAccessChain/OpLoad —
            // illegal (UniformConstant data, VUID-04655) and a hard SEGFAULT in NVIDIA's SPIR-V->NVVM
            // compiler even when spirv-val passes the undef. Bind it to a Private zero var instead so
            // the body reads zeros through a class NVIDIA compiles. Chains rewritten in apply_bindings.
            let var = ctx.zero_private_var(pointee);
            bindings.push((*pid, ParamBinding::ZeroPointer { var }));
        } else {
            // Unmodeled parameter: bind a zero/undef value of its type so the body stays well-formed.
            let z = ctx.const_zero(*pty, &defs);
            bindings.push((*pid, ParamBinding::ZeroValue { val: z }));
        }
    }

    // Block-decorate buffer structs + member offsets (std140-ish; we trust the AIR member layout and
    // emit offsets from the struct's member types). The map must include synthesized wrapper structs,
    // so merge new_globals into the type-def view.
    let mut all_defs = defs.clone();
    for g in &ctx.new_globals {
        if let Some(id) = g.result_id {
            all_defs.entry(id).or_insert_with(|| g.clone());
        }
    }
    split_explicit_layout_type_aliases(ctx, &buffer_structs, &mut all_defs);
    for g in &ctx.new_globals {
        if let Some(id) = g.result_id {
            all_defs.entry(id).or_insert_with(|| g.clone());
        }
    }
    // Several buffer params can share ONE Block struct type (e.g. multi_add: three `device float*`
    // args the backend wrapped into the same `{ RuntimeArray<float> }`). A Block/Offset decoration on a
    // given struct id must appear exactly once, so decorate each distinct struct only the first time.
    let mut block_decorated: HashSet<Word> = HashSet::new();
    for (var, struct_ty) in &buffer_structs {
        if block_decorated.insert(*struct_ty) {
            decorate_block_struct(ctx, *struct_ty, &all_defs);
        }
        ctx.interface_buffer_var(*var); // SPIR-V 1.4+ lists all globals; harmless for <=1.3.
    }

    // Convert the module-scope static sampler `__air_sampler_state` (an AIR-embedded default sampler,
    // an OpVariable UniformConstant with a constant array initializer) into a real sampler resource
    // and rewrite its `OpBitcast ... %__air_sampler_state` uses into a load of that sampler.
    handle_static_sampler(ctx)?;
    include_existing_private_globals(ctx);

    // Register textures embedded in an argument buffer (via `air.indirect_argument` → `air.texture`,
    // read/written by AIR texture intrinsics) as standalone images BEFORE applying param bindings. This
    // lands them in `ctx.image_dims`/`ctx.image_comp` and, for writable fields, `ctx.image_storage`, so
    // private-placeholder helper operands can recover the real descriptor.
    let embedded_textures = match stage {
        Stage::Fragment => frag.map(|meta| meta.embedded_textures.as_slice()),
        Stage::Vertex => vert.map(|meta| meta.embedded_textures.as_slice()),
        Stage::Kernel => kern.map(|meta| meta.embedded_textures.as_slice()),
    };
    register_embedded_textures(ctx, entry_idx, embedded_textures)?;

    // Apply param bindings to the body: drop params, then splice replacements.
    apply_bindings(ctx, entry_idx, bindings, &buffer_structs, &all_defs)?;
    if frag
        .and_then(|meta| meta.fragment_imageblock.as_ref())
        .is_some()
    {
        ctx.uses_fragment_imageblock = true;
        ctx.fragment_imageblock_coord_var = fragcoord_var;
        let block = ctx.module.functions[entry_idx]
            .blocks
            .first_mut()
            .ok_or_else(|| "fragment imageblock entry has no block".to_string())?;
        let insert_at = block
            .instructions
            .iter()
            .position(|instruction| instruction.class.opcode != Op::Variable)
            .unwrap_or(block.instructions.len());
        block.instructions.insert(
            insert_at,
            Instruction::new(Op::BeginInvocationInterlockEXT, None, None, vec![]),
        );
    }
    lower_patch_control_point_calls(ctx, entry_idx, vert, &all_defs)?;
    lower_buffer_address_facts(ctx, entry_idx, kern)?;

    Ok(defs)
}

/// Whether a direct buffer element view can own a descriptor alias at the same binding. Storage
/// buffer descriptors do not encode their element type; retaining one scalar/vector numeric view
/// per statically typed access lets Logical SPIR-V preserve each pointer type without retyping an
/// alias after construction. Aggregates stay on the layout-aware reconstruction path.
fn buffer_typed_alias_element(defs: &HashMap<Word, Instruction>, ty: Word) -> bool {
    let Some(definition) = defs.get(&ty) else {
        return false;
    };
    if matches!(definition.class.opcode, Op::TypeInt | Op::TypeFloat) {
        return true;
    }
    if definition.class.opcode != Op::TypeVector {
        return false;
    }
    let (Some(Operand::IdRef(element)), Some(Operand::LiteralBit32(lanes))) =
        (definition.operands.first(), definition.operands.get(1))
    else {
        return false;
    };
    (2..=4).contains(lanes)
        && defs
            .get(element)
            .is_some_and(|element| matches!(element.class.opcode, Op::TypeInt | Op::TypeFloat))
}

fn integer_component_width(ctx: &Ctx, ty: Word) -> Option<u32> {
    let definition = ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
        .find(|instruction| instruction.result_id == Some(ty))?;
    match definition.class.opcode {
        Op::TypeInt => match definition.operands.first() {
            Some(Operand::LiteralBit32(bits)) => Some(*bits),
            _ => None,
        },
        Op::TypeVector => match definition.operands.first() {
            Some(Operand::IdRef(element)) => integer_component_width(ctx, *element),
            _ => None,
        },
        _ => None,
    }
}

fn tess_coord_prefix_lanes(
    defs: &HashMap<Word, Instruction>,
    param_ty: Word,
    float_ty: Word,
) -> Result<Option<u32>, String> {
    match scalar_or_vector_component(defs, param_ty) {
        Some((component, Some(3))) if component == float_ty => Ok(None),
        Some((component, Some(2))) if component == float_ty => Ok(Some(2)),
        _ => Err("position_in_patch parameter must be float2 or float3".to_string()),
    }
}

fn lower_patch_control_point_calls(
    ctx: &mut Ctx,
    entry_idx: usize,
    vert: Option<&VertMeta>,
    defs: &HashMap<Word, Instruction>,
) -> Result<(), String> {
    let Some(tessellation) = vert.and_then(|meta| meta.tessellation.as_ref()) else {
        return Ok(());
    };
    let Some(control_point_function) = tessellation.control_point_function.as_ref() else {
        return Ok(());
    };
    let function_id = ctx.module.debug_names.iter().find_map(|instruction| {
        let [Operand::IdRef(id), Operand::LiteralString(name)] = instruction.operands.as_slice()
        else {
            return None;
        };
        (instruction.class.opcode == Op::Name && name == control_point_function).then_some(*id)
    });
    let Some(function_id) = function_id else {
        return Err(format!(
            "tessellation control-point function {:?} has no emitted declaration",
            control_point_function
        ));
    };
    let call_result_type = ctx.module.functions[entry_idx]
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .find(|instruction| {
            instruction.class.opcode == Op::FunctionCall
                && instruction.operands.first() == Some(&Operand::IdRef(function_id))
        })
        .and_then(|instruction| instruction.result_type);
    let Some(call_result_type) = call_result_type else {
        return Ok(());
    };
    let member_types = defs
        .get(&call_result_type)
        .filter(|definition| definition.class.opcode == Op::TypeStruct)
        .map(|definition| {
            definition
                .operands
                .iter()
                .filter_map(|operand| match operand {
                    Operand::IdRef(id) => Some(*id),
                    _ => None,
                })
                .collect::<Vec<_>>()
        })
        .ok_or("tessellation control-point accessor must return a struct")?;
    if member_types.len() != tessellation.control_point_fields.len() {
        return Err(format!(
            "tessellation control-point metadata has {} fields but accessor returns {}",
            tessellation.control_point_fields.len(),
            member_types.len()
        ));
    }

    let mut inputs = Vec::with_capacity(member_types.len());
    for (member_ty, field) in member_types
        .iter()
        .copied()
        .zip(&tessellation.control_point_fields)
    {
        let array_ty = ctx.ty_array(member_ty, tessellation.control_point_count);
        let pointer_ty = ctx.ty_ptr(StorageClass::Input, array_ty);
        let var = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::Variable,
            Some(pointer_ty),
            Some(var),
            vec![Operand::StorageClass(StorageClass::Input)],
        ));
        decorate_location(&mut ctx.module, var, field.location);
        ctx.interface.push(var);
        inputs.push((var, member_ty));
    }

    for block_idx in 0..ctx.module.functions[entry_idx].blocks.len() {
        let old = ctx.module.functions[entry_idx].blocks[block_idx]
            .instructions
            .clone();
        let mut rewritten = Vec::with_capacity(old.len());
        for instruction in old {
            if instruction.class.opcode != Op::FunctionCall
                || instruction.operands.first() != Some(&Operand::IdRef(function_id))
            {
                rewritten.push(instruction);
                continue;
            }
            let result = instruction
                .result_id
                .ok_or("tessellation control-point call has no result")?;
            let result_type = instruction
                .result_type
                .ok_or("tessellation control-point call has no result type")?;
            if result_type != call_result_type {
                return Err(
                    "tessellation control-point accessor has inconsistent return types".into(),
                );
            }
            let index = instruction
                .operands
                .get(1)
                .cloned()
                .ok_or("tessellation control-point call has no index")?;
            let mut members = Vec::with_capacity(inputs.len());
            for (var, member_ty) in &inputs {
                let member_pointer_ty = ctx.ty_ptr(StorageClass::Input, *member_ty);
                let pointer = ctx.module.fresh_id();
                rewritten.push(Instruction::new(
                    Op::AccessChain,
                    Some(member_pointer_ty),
                    Some(pointer),
                    vec![Operand::IdRef(*var), index.clone()],
                ));
                let member = ctx.module.fresh_id();
                rewritten.push(Instruction::new(
                    Op::Load,
                    Some(*member_ty),
                    Some(member),
                    vec![Operand::IdRef(pointer)],
                ));
                members.push(Operand::IdRef(member));
            }
            rewritten.push(Instruction::new(
                Op::CompositeConstruct,
                Some(result_type),
                Some(result),
                members,
            ));
        }
        ctx.module.functions[entry_idx].blocks[block_idx].instructions = rewritten;
    }
    Ok(())
}

/// Materialize a UniformConstant image for each argument-buffer-embedded texture the meta
/// pass surfaced (see `KernMeta::embedded_textures`), decorate it in the sampled- or storage-texture
/// band at index `K`
/// (K = the synthetic texture index the meta pass assigned via `embedded_synthetic_texture_index`,
/// the SAME convention the validation harness uses to bind the seeded texture), load it at entry, and
/// register the loaded image in `image_dims`/`image_comp` plus `image_storage` for writable fields.
///
/// This is what turns the use of an argument-buffer-embedded texture (whose handle is a private
/// pointer loaded from the arg buffer) into a real `OpImageFetch`/`OpImageWrite`: the texture lowerings
/// fall back to the unambiguous sampled/storage image when the operand is a private pointer, and this
/// registration provides that image. Gated entirely on AIR structure, never on a shader name.
fn register_embedded_textures(
    ctx: &mut Ctx,
    entry_idx: usize,
    embedded: Option<&[crate::meta::EmbeddedTexture]>,
) -> Result<(), String> {
    let descriptor_layout = ctx.descriptor_layout;
    let Some(embedded) = embedded else {
        return Ok(());
    };
    if embedded.is_empty() {
        return Ok(());
    }
    let mut loads: Vec<Instruction> = vec![];
    let mut replacements = Vec::new();
    for tex in embedded.iter().copied() {
        if tex.array_length == Some(0) {
            continue;
        }
        let Some(buffer_root) = ctx.module.functions[entry_idx]
            .parameters
            .get(tex.buffer_param_index as usize)
            .and_then(|parameter| parameter.result_id)
        else {
            continue;
        };
        let (image_ty, runtime_specialization) = if let Some(format) = tex.storage_format {
            let (format, state) = ctx.specialize_storage_image_format(
                tex.synthetic_texture_index,
                format.to_spirv_format(),
                tex.comp,
            )?;
            (
                ctx.ty_storage_image(tex.dim, tex.arrayed, format, tex.comp),
                state,
            )
        } else {
            (ctx.ty_image(tex.dim, tex.arrayed, tex.comp), None)
        };
        let binding_ty = tex
            .array_length
            .map(|length| ctx.ty_array(image_ty, length))
            .unwrap_or(image_ty);
        let pptr = ctx.ty_ptr(StorageClass::UniformConstant, binding_ty);
        let var = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::Variable,
            Some(pptr),
            Some(var),
            vec![Operand::StorageClass(StorageClass::UniformConstant)],
        ));
        // ABI: bind in the sampled- or storage-texture band at index K.
        let binding = if tex.storage_format.is_some() {
            storage_texture_resource_binding(descriptor_layout, tex.synthetic_texture_index)?
        } else {
            texture_resource_binding(descriptor_layout, tex.synthetic_texture_index)?
        };
        decorate_binding(&mut ctx.module, var, descriptor_layout.set, binding);
        ctx.interface_buffer_var(var); // SPIR-V 1.4+ lists every resource on the entry interface.
        if let Some(length) = tex.array_length {
            ctx.image_array_vars
                .insert(var, (image_ty, (tex.dim, tex.arrayed), tex.comp, false));
            ctx.register_runtime_storage_image_value(
                var,
                tex.synthetic_texture_index,
                runtime_specialization,
            );
            for fact in &mut ctx.emit_sidecar.buffer_pointer_field_loads {
                let end = u64::from(tex.field_offset) + u64::from(length) * 8;
                if fact.root == buffer_root
                    && fact.byte_offset >= u64::from(tex.field_offset)
                    && fact.byte_offset < end
                    && (fact.byte_offset - u64::from(tex.field_offset)) % 8 == 0
                {
                    fact.root = var;
                    fact.byte_offset -= u64::from(tex.field_offset);
                }
            }
            for fact in &mut ctx.emit_sidecar.buffer_pointer_dynamic_field_loads {
                if fact.root == buffer_root && fact.byte_offset == u64::from(tex.field_offset) {
                    fact.root = var;
                    fact.byte_offset = 0;
                }
            }
            continue;
        }
        let lid = ctx.module.fresh_id();
        loads.push(Instruction::new(
            Op::Load,
            Some(image_ty),
            Some(lid),
            vec![Operand::IdRef(var)],
        ));
        ctx.image_dims.insert(lid, (tex.dim, tex.arrayed));
        ctx.image_comp.insert(lid, tex.comp);
        if tex.storage_format.is_some() {
            ctx.image_storage.insert(lid);
            ctx.register_runtime_storage_image_value(
                var,
                tex.synthetic_texture_index,
                runtime_specialization,
            );
            ctx.register_runtime_storage_image_value(
                lid,
                tex.synthetic_texture_index,
                runtime_specialization,
            );
        }
        replacements.extend(
            ctx.emit_sidecar
                .buffer_pointer_field_loads
                .iter()
                .filter(|fact| {
                    fact.root == buffer_root && fact.byte_offset == u64::from(tex.field_offset)
                })
                .map(|fact| (fact.id, lid)),
        );
    }
    for (placeholder, image) in replacements {
        replace_id_in_function(&mut ctx.module.functions[entry_idx], placeholder, image);
    }
    // Insert the loads at the top of the entry block, AFTER any leading OpVariables (SPIR-V requires
    // function-local OpVariables to be the first instructions of the entry block). apply_bindings
    // inserts its own loads by the same rule afterward; both land after the variables.
    if let Some(first) = ctx.module.functions[entry_idx].blocks.first_mut() {
        let at = first
            .instructions
            .iter()
            .position(|i| i.class.opcode != Op::Variable)
            .unwrap_or(first.instructions.len());
        for (k, ld) in loads.into_iter().enumerate() {
            first.instructions.insert(at + k, ld);
        }
    }
    Ok(())
}

impl Ctx {
    fn clone_type_for_record_array_element(
        &mut self,
        ty: Word,
        defs: &HashMap<Word, Instruction>,
    ) -> Word {
        let mut memo = HashMap::new();
        self.clone_type_for_record_array_element_inner(ty, defs, &mut memo)
    }

    fn clone_type_for_record_array_element_inner(
        &mut self,
        ty: Word,
        defs: &HashMap<Word, Instruction>,
        memo: &mut HashMap<Word, Word>,
    ) -> Word {
        if let Some(&cloned) = memo.get(&ty) {
            return cloned;
        }
        let Some(def) = defs.get(&ty).cloned() else {
            return ty;
        };
        match def.class.opcode {
            Op::TypeStruct => {
                let members = def
                    .operands
                    .iter()
                    .map(|op| match op {
                        Operand::IdRef(member_ty) => Operand::IdRef(
                            self.clone_type_for_record_array_element_inner(*member_ty, defs, memo),
                        ),
                        other => other.clone(),
                    })
                    .collect::<Vec<_>>();
                let cloned = self.module.fresh_id();
                memo.insert(ty, cloned);
                if let Some(offsets) = self.air_struct_offsets.get(&ty).cloned() {
                    self.air_struct_offsets.insert(cloned, offsets);
                }
                self.new_globals
                    .push(type_inst(Op::TypeStruct, cloned, members));
                cloned
            }
            Op::TypeArray | Op::TypeRuntimeArray => {
                let Some(Operand::IdRef(elem)) = def.operands.first() else {
                    return ty;
                };
                let cloned_elem = self.clone_type_for_record_array_element_inner(*elem, defs, memo);
                if cloned_elem == *elem {
                    return ty;
                }
                let mut operands = def.operands.clone();
                operands[0] = Operand::IdRef(cloned_elem);
                let cloned = self.module.fresh_id();
                memo.insert(ty, cloned);
                self.new_globals
                    .push(type_inst(def.class.opcode, cloned, operands));
                cloned
            }
            _ => ty,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_buffer_alias_elements_include_numeric_vectors() {
        let defs = HashMap::from([
            (
                1,
                Instruction::new(
                    Op::TypeFloat,
                    None,
                    Some(1),
                    vec![Operand::LiteralBit32(32)],
                ),
            ),
            (
                2,
                Instruction::new(
                    Op::TypeVector,
                    None,
                    Some(2),
                    vec![Operand::IdRef(1), Operand::LiteralBit32(4)],
                ),
            ),
            (
                3,
                Instruction::new(Op::TypeStruct, None, Some(3), vec![Operand::IdRef(1)]),
            ),
        ]);

        assert!(buffer_typed_alias_element(&defs, 1));
        assert!(buffer_typed_alias_element(&defs, 2));
        assert!(!buffer_typed_alias_element(&defs, 3));
    }
    use crate::spirv_module::Instruction;
    use crate::spirv_module::ModuleHeader;
    use crate::spirv_module::Operand;
    use spirv::Op;

    fn ty(op: Op, id: u32, operands: Vec<Operand>) -> Instruction {
        Instruction::new(op, None, Some(id), operands)
    }

    #[test]
    fn tess_coord_binding_preserves_float3_and_only_truncates_float2() {
        let mut defs = HashMap::new();
        defs.insert(1, ty(Op::TypeFloat, 1, vec![Operand::LiteralBit32(32)]));
        defs.insert(
            2,
            ty(
                Op::TypeVector,
                2,
                vec![Operand::IdRef(1), Operand::LiteralBit32(2)],
            ),
        );
        defs.insert(
            3,
            ty(
                Op::TypeVector,
                3,
                vec![Operand::IdRef(1), Operand::LiteralBit32(3)],
            ),
        );
        defs.insert(
            4,
            ty(
                Op::TypeVector,
                4,
                vec![Operand::IdRef(1), Operand::LiteralBit32(4)],
            ),
        );

        assert_eq!(tess_coord_prefix_lanes(&defs, 3, 1), Ok(None));
        assert_eq!(tess_coord_prefix_lanes(&defs, 2, 1), Ok(Some(2)));
        assert!(tess_coord_prefix_lanes(&defs, 4, 1).is_err());
    }

    // A fragment Input of integer (or 64-bit float) component type cannot be interpolated and needs a
    // Flat decoration; a 32-bit-float scalar/vector is interpolated and must NOT be flagged.
    #[test]
    fn fragment_input_needs_flat_only_for_integer_and_double() {
        let mut defs = HashMap::new();
        defs.insert(
            1,
            ty(
                Op::TypeInt,
                1,
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
        );
        defs.insert(2, ty(Op::TypeFloat, 2, vec![Operand::LiteralBit32(32)]));
        defs.insert(3, ty(Op::TypeFloat, 3, vec![Operand::LiteralBit32(64)]));
        // %4 = <4 x uint>, %5 = <2 x float32>, %6 = <3 x double>
        defs.insert(
            4,
            ty(
                Op::TypeVector,
                4,
                vec![Operand::IdRef(1), Operand::LiteralBit32(4)],
            ),
        );
        defs.insert(
            5,
            ty(
                Op::TypeVector,
                5,
                vec![Operand::IdRef(2), Operand::LiteralBit32(2)],
            ),
        );
        defs.insert(
            6,
            ty(
                Op::TypeVector,
                6,
                vec![Operand::IdRef(3), Operand::LiteralBit32(3)],
            ),
        );
        assert!(fragment_input_needs_flat(&defs, 1), "uint scalar");
        assert!(!fragment_input_needs_flat(&defs, 2), "float32 scalar");
        assert!(fragment_input_needs_flat(&defs, 3), "double scalar");
        assert!(fragment_input_needs_flat(&defs, 4), "uint vector");
        assert!(!fragment_input_needs_flat(&defs, 5), "float32 vector");
        assert!(fragment_input_needs_flat(&defs, 6), "double vector");
    }

    #[test]
    fn type_contains_bool_descends_through_vectors() {
        let mut defs = HashMap::new();
        defs.insert(1, ty(Op::TypeBool, 1, vec![]));
        defs.insert(2, ty(Op::TypeFloat, 2, vec![Operand::LiteralBit32(32)]));
        defs.insert(
            3,
            ty(
                Op::TypeVector,
                3,
                vec![Operand::IdRef(1), Operand::LiteralBit32(2)],
            ),
        );
        defs.insert(
            4,
            ty(
                Op::TypeVector,
                4,
                vec![Operand::IdRef(2), Operand::LiteralBit32(2)],
            ),
        );

        assert!(type_contains_bool(&defs, 1));
        assert!(type_contains_bool(&defs, 3));
        assert!(!type_contains_bool(&defs, 2));
        assert!(!type_contains_bool(&defs, 4));
    }

    // layout_types_reachable_from collects every struct/array reached from the Block struct (these all
    // get explicit layout), but not scalars/vectors.
    #[test]
    fn layout_types_reachable_from_walks_nested_struct_and_array_members() {
        let mut defs = HashMap::new();
        // %1 = uint (scalar)
        defs.insert(
            1,
            ty(
                Op::TypeInt,
                1,
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
        );
        // %2 = struct { uint }  -- nested, also a Workgroup pointee in the real bug
        defs.insert(2, ty(Op::TypeStruct, 2, vec![Operand::IdRef(1)]));
        // %3 = struct { uint }  -- nested inside an array
        defs.insert(3, ty(Op::TypeStruct, 3, vec![Operand::IdRef(1)]));
        // %4 = array of %3
        defs.insert(
            4,
            ty(Op::TypeArray, 4, vec![Operand::IdRef(3), Operand::IdRef(1)]),
        );
        // %5 = Block struct { uint, %2, %4 }  -- the buffer struct (root)
        defs.insert(
            5,
            ty(
                Op::TypeStruct,
                5,
                vec![Operand::IdRef(1), Operand::IdRef(2), Operand::IdRef(4)],
            ),
        );
        // %6 = an unrelated struct, NOT reachable from the root
        defs.insert(6, ty(Op::TypeStruct, 6, vec![Operand::IdRef(1)]));

        let roots: HashSet<Word> = [5].into_iter().collect();
        let reachable = layout_types_reachable_from(&roots, &defs);

        assert!(reachable.contains(&5), "root block struct");
        assert!(reachable.contains(&2), "directly nested struct");
        assert!(reachable.contains(&3), "struct nested through an array");
        assert!(reachable.contains(&4), "array on the layout path");
        assert!(!reachable.contains(&1), "scalars are not layout composites");
        assert!(!reachable.contains(&6), "unrelated struct not reachable");
    }

    #[test]
    fn split_workgroup_layout_aliases_clones_nested_aggregate_paths() {
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(50));
        module.types_global_values = vec![
            ty(
                Op::TypeInt,
                1,
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            Instruction::new(
                Op::Constant,
                Some(1),
                Some(2),
                vec![Operand::LiteralBit32(4)],
            ),
            ty(Op::TypeArray, 3, vec![Operand::IdRef(1), Operand::IdRef(2)]),
            ty(Op::TypeStruct, 4, vec![Operand::IdRef(3)]),
            ty(Op::TypeStruct, 5, vec![Operand::IdRef(4)]),
            ty(Op::TypeStruct, 6, vec![Operand::IdRef(4)]),
            ty(Op::TypeArray, 7, vec![Operand::IdRef(6), Operand::IdRef(2)]),
            ty(
                Op::TypePointer,
                8,
                vec![
                    Operand::StorageClass(StorageClass::Workgroup),
                    Operand::IdRef(7),
                ],
            ),
            Instruction::new(
                Op::Variable,
                Some(8),
                Some(9),
                vec![Operand::StorageClass(StorageClass::Workgroup)],
            ),
            ty(
                Op::TypePointer,
                10,
                vec![
                    Operand::StorageClass(StorageClass::Function),
                    Operand::IdRef(4),
                ],
            ),
            ty(
                Op::TypePointer,
                11,
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(4),
                ],
            ),
            ty(
                Op::TypePointer,
                12,
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(5),
                ],
            ),
            Instruction::new(
                Op::Variable,
                Some(12),
                Some(13),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
        ];

        let mut defs = module
            .types_global_values
            .iter()
            .filter_map(|inst| inst.result_id.map(|id| (id, inst.clone())))
            .collect::<HashMap<_, _>>();
        let mut ctx = Ctx::new(module);
        split_explicit_layout_type_aliases(&mut ctx, &[(13, 5)], &mut defs);

        let pointer_pointee = |id| {
            defs.get(&id)
                .and_then(|inst| inst.operands.get(1))
                .and_then(|operand| match operand {
                    Operand::IdRef(pointee) => Some(*pointee),
                    _ => None,
                })
                .expect("pointer pointee")
        };
        let workgroup_root = pointer_pointee(8);
        let function_pointee = pointer_pointee(10);
        assert_ne!(workgroup_root, 7, "array-root Workgroup graph cloned");
        assert_ne!(
            function_pointee, 4,
            "aggregate-copy Function pointer cloned"
        );
        assert_eq!(pointer_pointee(11), 4, "StorageBuffer keeps laid-out type");
        assert_eq!(pointer_pointee(12), 5, "Block pointer remains unchanged");

        let cloned_graph =
            layout_types_reachable_from(&[workgroup_root].into_iter().collect(), &defs);
        assert!(!cloned_graph.contains(&3), "shared decorated array removed");
        assert!(
            !cloned_graph.contains(&4),
            "shared decorated struct removed"
        );
        let root_pos = ctx
            .module
            .types_global_values
            .iter()
            .position(|inst| inst.result_id == Some(workgroup_root))
            .expect("cloned root definition");
        let pointer_pos = ctx
            .module
            .types_global_values
            .iter()
            .position(|inst| inst.result_id == Some(8))
            .expect("Workgroup pointer definition");
        assert!(root_pos < pointer_pos, "clone defined before pointer use");
        assert_eq!(ctx.ty_ptr(StorageClass::Workgroup, workgroup_root), 8);
        assert_ne!(ctx.ty_ptr(StorageClass::Workgroup, 7), 8);
    }

    #[test]
    fn split_explicit_layout_aliases_isolates_function_only_structs() {
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(20));
        module.types_global_values = vec![
            ty(
                Op::TypeInt,
                1,
                vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
            ),
            ty(Op::TypeStruct, 2, vec![Operand::IdRef(1)]),
            ty(
                Op::TypePointer,
                3,
                vec![
                    Operand::StorageClass(StorageClass::Function),
                    Operand::IdRef(2),
                ],
            ),
            ty(
                Op::TypePointer,
                4,
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(2),
                ],
            ),
            ty(
                Op::TypePointer,
                7,
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(2),
                ],
            ),
            Instruction::new(
                Op::Variable,
                Some(4),
                Some(5),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
            Instruction::new(Op::ConstantNull, Some(2), Some(6), vec![]),
        ];

        let mut defs = module
            .types_global_values
            .iter()
            .filter_map(|inst| inst.result_id.map(|id| (id, inst.clone())))
            .collect::<HashMap<_, _>>();
        let mut ctx = Ctx::new(module);
        split_explicit_layout_type_aliases(&mut ctx, &[(5, 2)], &mut defs);

        let pointee = |id| match defs.get(&id).and_then(|inst| inst.operands.get(1)) {
            Some(Operand::IdRef(pointee)) => *pointee,
            _ => panic!("pointer pointee"),
        };
        assert_ne!(pointee(3), 2, "Function pointer receives undecorated clone");
        assert_eq!(
            pointee(7),
            pointee(3),
            "Private pointer receives the same undecorated clone"
        );
        assert_eq!(pointee(4), 2, "StorageBuffer keeps laid-out type");
        assert_eq!(
            ctx.module
                .types_global_values
                .iter()
                .find(|instruction| instruction.result_id == Some(6))
                .and_then(|instruction| instruction.result_type),
            Some(pointee(3)),
            "pre-existing aggregate values follow the unlaid Function type"
        );
    }

    #[test]
    fn split_explicit_layout_aliases_isolates_nested_block_root() {
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(20));
        module.types_global_values = vec![
            ty(
                Op::TypeInt,
                1,
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            ty(Op::TypeStruct, 2, vec![Operand::IdRef(1)]),
            ty(Op::TypeStruct, 3, vec![Operand::IdRef(2)]),
            ty(
                Op::TypePointer,
                4,
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(2),
                ],
            ),
            Instruction::new(
                Op::Variable,
                Some(4),
                Some(5),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
            ty(
                Op::TypePointer,
                6,
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(3),
                ],
            ),
            Instruction::new(
                Op::Variable,
                Some(6),
                Some(7),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
        ];
        let mut defs = module
            .types_global_values
            .iter()
            .filter_map(|instruction| instruction.result_id.map(|id| (id, instruction.clone())))
            .collect::<HashMap<_, _>>();
        let mut ctx = Ctx::new(module);

        split_explicit_layout_type_aliases(&mut ctx, &[(5, 2), (7, 3)], &mut defs);

        let nested = match defs
            .get(&3)
            .and_then(|definition| definition.operands.first())
        {
            Some(Operand::IdRef(member)) => *member,
            _ => panic!("outer block member"),
        };
        assert_ne!(nested, 2, "nested occurrence receives a non-Block clone");
        assert_eq!(
            defs.get(&nested).map(|definition| definition.class.opcode),
            Some(Op::TypeStruct)
        );
        assert_eq!(
            defs.get(&2)
                .and_then(|definition| definition.operands.first()),
            Some(&Operand::IdRef(1)),
            "independently bound root remains unchanged"
        );
    }
}
