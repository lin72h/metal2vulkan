//! Rewrite decoded AIR entry parameters into Vulkan stage-input and resource variables.

use super::*;
use crate::meta::primitive_air_type_from_name;
use crate::passes::stage_output::handle_static_sampler;
mod decorations;
mod kernel_values;
mod layout;

pub(in crate::passes) use decorations::*;
pub(in crate::passes) use kernel_values::const_ivec;
use kernel_values::{
    bind_kernel_uvec3_builtin, bind_kernel_uvec3_builtin_var, const_kernel_local_size,
    is_raw_uint_buffer_block,
};
use layout::*;

// Layout size/align helpers are reused by the lower pass to decorate OpPtrAccessChain base
// pointer types with ArrayStride (a sibling-module of interface cannot reach `layout` directly).
pub(in crate::passes) use layout::{
    decorate_block_struct, layout_ty_size_align, round_up, ty_size_align,
};

mod air_layout;
pub(in crate::passes) use air_layout::*;
const WORKGROUP_MEMORY_ELEMENTS: u32 = 512;

/// What an entry parameter became, so the body can be patched to read from it.
pub(in crate::passes) enum ParamBinding {
    /// A loadable interface var (Input varying / vertex attribute): replace param uses with an
    /// OpLoad of `var` (type = the param's value type) inserted at function start.
    LoadVar { var: Word, ty: Word },
    /// A scalar builtin Input var (`VertexIndex`/`InstanceIndex`, a 32-bit uint) feeding a NARROWER
    /// integer param (`ushort [[instance_id]]`, an i16): load the uint then `OpUConvert` it down to the
    /// param's own width, so the body's 16-bit uses (`OpBitwiseAnd %ushort`) are width-consistent.
    LoadVarConverted {
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
    /// An image variable (texture): param uses are the sample call's texture operand; replace param
    /// id with an OpLoad of the image at use. We record the var + image type + its (Dim, arrayed).
    Image {
        var: Word,
        image_ty: Word,
        dim: (Dim, bool),
        comp: ImageComp,
    },
    /// A runtime-indexed texture ARRAY (`array_ref<texture>`): a descriptor array of sampled images.
    /// Declared as `OpTypeArray %image N` in UniformConstant. Unlike `Image`, the param is NOT replaced
    /// by a loaded image at function top; it is spliced to the array VARIABLE, and
    /// `materialize_texture_array_loads` turns each per-element handle load into
    /// `OpAccessChain %arrayvar %idx` + `OpLoad %image` at the use site. `elem_image_ty` is the element
    /// `OpTypeImage`; the pass records `(var -> (elem_image_ty, dim, comp))` in `ctx.image_array_vars`.
    ImageArray {
        var: Word,
        elem_image_ty: Word,
        dim: (Dim, bool),
        comp: ImageComp,
    },
    /// A write-only storage-image variable (`OpTypeImage Sampled=2` + ImageFormat): param uses are the
    /// `air.write_texture_*` texture operand; replace the param id with an OpLoad of the storage image.
    /// The loaded id is recorded in `ctx.image_storage` so the write lowering emits `OpImageWrite`.
    StorageImage {
        var: Word,
        image_ty: Word,
        dim: (Dim, bool),
        comp: ImageComp,
    },
    /// A framebuffer-fetch `[[color(n)]]` input. Vulkan exposes these as subpass input attachments,
    /// read with `OpImageRead` at the current fragment location.
    InputAttachment {
        var: Word,
        image_ty: Word,
        read_ty: Word,
        param_ty: Word,
    },
    /// A sampler variable: replace param uses with an OpLoad of the sampler.
    Sampler { var: Word },
    /// A buffer block variable, with the lowering of the body's param uses (see `BufWrap`).
    Buffer { var: Word, wrap: BufWrap },
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

/// How a buffer param's body uses are lowered.
pub(in crate::passes) enum BufWrap {
    /// The AIR pointee was a (heterogeneous) struct the backend kept as a struct pointer. The body's
    /// access chains already index struct members off the param — just splice the var id.
    Direct,
    /// The backend kept a struct pointer, but at least one AIR use indexes an implicit array of those
    /// structs (`buffer[N].field`). The StorageBuffer is `{ RuntimeArray<Struct> }`; direct struct
    /// member paths are routed to record 0, and non-direct paths keep their original first operand as
    /// the record index.
    RecordArray { block_ty: Word, elem_ty: Word },
    /// The backend collapsed the buffer into a bare element pointer (`T*`) and emits physical access
    /// chains against it (illegal in Logical SPIR-V). `block_ty` is the StorageBuffer Block we point
    /// the var at; we re-root the body's access chains at the var and route direct loads through the
    /// offset-0 leaf. `prepend_member0` is true for a genuine `device T*` array wrapped as
    /// `{ RuntimeArray<T> }` (the original first index then indexes the runtime array); false for a
    /// reconstructed struct (the original indices already navigate it).
    Collapsed {
        block_ty: Word,
        prepend_member0: bool,
    },
}

/// Rewrite the entry function's parameters into Vulkan interface variables (Input/UniformConstant/
/// Uniform) by AIR role, and its return value into Output variable(s). The bridge from the raw llc
/// module to a Vulkan-pointed entry.
pub(super) fn build_stage_input(
    ctx: &mut Ctx,
    entry_idx: usize,
    stage: &Stage,
    frag: Option<&FragMeta>,
    vert: Option<&VertMeta>,
    kern: Option<&KernMeta>,
) -> Result<HashMap<Word, Instruction>, String> {
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
        wtex_candidates.entry(pid).or_insert(shape);
    }
    let wtex_dims: HashMap<Word, (Dim, bool, ImageFormat, ImageComp)> = wtex_candidates
        .into_iter()
        .filter(|(pid, _)| !tex_dims.contains_key(pid))
        .collect();

    // Plan a binding for each parameter, allocating descriptor bindings and locations in role order.
    let mut binding_ctr: u32 = 0;
    let mut bindings: Vec<(Word, ParamBinding)> = vec![];
    // Buffer params whose pointer storage class must become Uniform, with their struct type id.
    let mut buffer_structs: Vec<(Word, Word)> = vec![]; // (var_id, struct_ty)
                                                        // The single FragCoord Input var: a fragment shader may carry MORE THAN ONE `position` param (e.g.
                                                        // an FC-specialized shader threads the pixel position into several helpers), but Vulkan forbids two
                                                        // interface variables decorated with the same builtin. Create FragCoord once and share it.
    let mut fragcoord_var: Option<Word> = None;
    let mut local_invocation_index_var: Option<Word> = None;
    let mut num_workgroups_var: Option<Word> = None;
    let mut global_invocation_id_var: Option<Word> = None;

    for (i, (pid, pty)) in params.iter().enumerate() {
        let idx = i as u32;
        let role_is = |s: &str| match stage {
            Stage::Fragment => match frag.and_then(|m| m.role_of(idx)) {
                Some(FragRole::Position) => s == "position",
                Some(FragRole::Varying(_)) => s == "varying",
                Some(FragRole::Texture(_)) => s == "texture",
                Some(FragRole::Sampler(_)) => s == "sampler",
                Some(FragRole::Buffer(_)) => s == "buffer",
                Some(FragRole::ColorInput(_)) => s == "color_input",
                _ => s == "other",
            },
            Stage::Vertex => match vert.and_then(|m| m.role_of(idx)) {
                Some(VertRole::VertexInput(_)) => s == "varying",
                Some(VertRole::Buffer(_)) => s == "buffer",
                Some(VertRole::Texture(_)) => s == "texture",
                Some(VertRole::Sampler(_)) => s == "sampler",
                Some(VertRole::VertexId) => s == "vertex_id",
                Some(VertRole::InstanceId) => s == "instance_id",
                _ => s == "other",
            },
            Stage::Kernel => match kern.and_then(|m| m.role_of(idx)) {
                Some(KernRole::Buffer(_) | KernRole::AccelerationStructureShadow(_)) => {
                    s == "buffer"
                }
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
                Some(FragRole::Texture(b)) => Some(texture_resource_binding(*b)),
                Some(FragRole::Sampler(b)) => Some(sampler_resource_binding(*b)),
                Some(FragRole::Buffer(b)) => Some(*b),
                Some(FragRole::ColorInput(b)) => Some(color_input_resource_binding(*b)),
                _ => None,
            },
            Stage::Vertex => match vert.and_then(|m| m.role_of(idx)) {
                Some(VertRole::Buffer(b)) => Some(*b),
                Some(VertRole::Texture(b)) => Some(texture_resource_binding(*b)),
                Some(VertRole::Sampler(b)) => Some(sampler_resource_binding(*b)),
                _ => None,
            },
            Stage::Kernel => match kern.and_then(|m| m.role_of(idx)) {
                Some(KernRole::Buffer(b) | KernRole::AccelerationStructureShadow(b)) => Some(*b),
                Some(KernRole::Texture(b)) => Some(texture_resource_binding(*b)),
                Some(KernRole::Sampler(b)) => Some(sampler_resource_binding(*b)),
                _ => None,
            },
        };

        let is_threadgroup_buffer = matches!(stage, Stage::Kernel)
            && role_is("buffer")
            && (kern.and_then(|m| m.buffer_address_space(idx)) == Some(3)
                || ptr_storage(&defs, *pty) == Some(StorageClass::Workgroup));

        // A runtime-indexed texture array is declared `array_ref<texture...>` in the AIR arg type
        // metadata. Such a param is a descriptor ARRAY, not a single image: the backend emits real
        // per-element handle loads (`load ptr addrspace(1), gep %argbuf, %idx`) that a single-image
        // binding turns into an illegal `OpLoad` of a pointer FROM the image value. Route these to the
        // `ImageArray` binding + `materialize_texture_array_loads`. Floor-safe: every `array_ref`
        // texture fails today (the emitter never aliases its handle loads to the param), so no currently
        // passing case is on this path.
        let texture_type_name = match stage {
            Stage::Fragment => frag.and_then(|m| m.texture_type_name(idx)),
            Stage::Vertex => vert.and_then(|m| m.texture_type_name(idx)),
            Stage::Kernel => kern.and_then(|m| m.texture_type_name(idx)),
        };
        let is_array_texture = texture_type_name
            .is_some_and(|n| n.contains("array_ref<") || n.contains("array_ref <"));

        if role_is("texture") && !wtex_dims.contains_key(pid) && is_array_texture {
            let (dim, arrayed, comp) = tex_dims
                .get(pid)
                .copied()
                .or_else(|| texture_type_hints.get(pid).copied())
                .unwrap_or((Dim::Dim2D, false, ImageComp::Float));
            let elem_image_ty = ctx.ty_image(dim, arrayed, comp);
            // The array length is a runtime function constant (`nImagesFC`), not a compile-time value,
            // so over-declare to `air.max_textures` (128). Vulkan lets an argument-buffer texture array
            // be over-sized; only accessed descriptors need be valid, and spirv-val does not bounds-check
            // a dynamic `OpAccessChain` index. A fixed `OpTypeArray` avoids the RuntimeDescriptorArray
            // capability; the index is `air.is_uniform`-marked, so no ShaderNonUniform is needed either.
            let array_ty = ctx.ty_array(elem_image_ty, 128);
            let pptr = ctx.ty_ptr(StorageClass::UniformConstant, array_ty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(pptr),
                Some(var),
                vec![Operand::StorageClass(StorageClass::UniformConstant)],
            ));
            let binding = allocate_resource_binding(&mut binding_ctr, resource_binding);
            decorate_binding(&mut ctx.module, var, binding);
            ctx.interface_buffer_var(var);
            bindings.push((
                *pid,
                ParamBinding::ImageArray {
                    var,
                    elem_image_ty,
                    dim: (dim, arrayed),
                    comp,
                },
            ));
        } else if role_is("texture") && wtex_dims.contains_key(pid) {
            // Write-only texture -> storage image (Sampled=2 + ImageFormat), lowered via OpImageWrite.
            let (dim, arrayed, fmt, comp) = wtex_dims
                .get(pid)
                .copied()
                .ok_or("write-texture dims missing for bound param")?;
            let image_ty = ctx.ty_storage_image(dim, arrayed, fmt, comp);
            let pptr = ctx.ty_ptr(StorageClass::UniformConstant, image_ty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(pptr),
                Some(var),
                vec![Operand::StorageClass(StorageClass::UniformConstant)],
            ));
            let binding = allocate_resource_binding(&mut binding_ctr, resource_binding);
            decorate_binding(&mut ctx.module, var, binding);
            ctx.interface_buffer_var(var);
            bindings.push((
                *pid,
                ParamBinding::StorageImage {
                    var,
                    image_ty,
                    dim: (dim, arrayed),
                    comp,
                },
            ));
        } else if role_is("texture") {
            let (dim, arrayed, comp) = tex_dims
                .get(pid)
                .copied()
                .or_else(|| texture_type_hints.get(pid).copied())
                .unwrap_or((Dim::Dim2D, false, ImageComp::Float));
            let image_ty = ctx.ty_image(dim, arrayed, comp);
            let pptr = ctx.ty_ptr(StorageClass::UniformConstant, image_ty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(pptr),
                Some(var),
                vec![Operand::StorageClass(StorageClass::UniformConstant)],
            ));
            let binding = allocate_resource_binding(&mut binding_ctr, resource_binding);
            decorate_binding(&mut ctx.module, var, binding);
            ctx.interface_buffer_var(var); // SPIR-V 1.4+ lists every resource on the entry interface.
            bindings.push((
                *pid,
                ParamBinding::Image {
                    var,
                    image_ty,
                    dim: (dim, arrayed),
                    comp,
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
            let binding = allocate_resource_binding(&mut binding_ctr, resource_binding);
            decorate_binding(&mut ctx.module, var, binding);
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
            let binding = allocate_resource_binding(&mut binding_ctr, resource_binding);
            decorate_binding(&mut ctx.module, var, binding);
            ctx.interface_buffer_var(var);
            bindings.push((*pid, ParamBinding::Sampler { var }));
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
            // the backend kept it as a struct pointer — make a StorageBuffer var pointing at it and the
            // body's member access chains just work. If pointee is a bare scalar/vector (a genuine
            // `device float4*` array, OR a homogeneous struct the backend collapsed into a `T*` that it
            // indexes `[i]`), physical array-stride access is illegal in Logical SPIR-V: wrap it as a
            // Block `{ RuntimeArray<pointee> }` and remember the element type so the body's param uses
            // get rewritten (member-0 prepend on chains; &buf[0] for direct loads).
            let pointee = ptr_pointee(&defs, *pty)
                .ok_or_else(|| format!("buffer param {pid} type {pty} is not a pointer"))?;
            let pointee_op = defs.get(&pointee).map(|d| d.class.opcode);
            let is_struct = pointee_op == Some(Op::TypeStruct);
            let is_raw_uint_block = is_raw_uint_buffer_block(&defs, pointee);
            // A SCALAR pointee means the backend FLATTENED the buffer to a flat scalar array and the
            // access chain indices are flat element offsets, NOT struct-navigation indices — so it must
            // use the RuntimeArray wrapping, not struct reconstruction (which would mis-index). A vector
            // pointee means the backend preserved the struct shape (multi-level nav), so reconstruct.
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
            let (struct_ty, wrap) = if is_raw_uint_block {
                if let Some(at) = layout
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
                    if has_struct_chain {
                        (
                            st,
                            BufWrap::Collapsed {
                                block_ty: st,
                                prepend_member0: false,
                            },
                        )
                    } else {
                        // Native raw-buffer params are already `{ RuntimeArray<uint> }` transport
                        // blocks. Use that block directly when the body's chains do not match the AIR
                        // struct metadata; those are raw word/byte paths, not structured member paths.
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
                // Backend kept the real struct type, but AIR uses the first GEP index as an implicit
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
                // Backend kept the real struct and AIR only indexes record 0; index it off the var
                // directly.
                (pointee, BufWrap::Direct)
            } else if let Some(at) = layout {
                // Backend collapsed a structured AIR buffer into a bare pointer. If the body has
                // access chains that type-check against the original AIR struct layout, preserve that
                // layout. This includes scalar first fields (`buf.field0`) whose SPIR-V shape is also a
                // single-index chain. Only fall back to RuntimeArray when the chains do not match the
                // struct and the body is reading flat `buf[i]` elements.
                let has_access_chains =
                    buffer_has_access_chains(&ctx.module.functions[entry_idx], *pid);
                let flat_elem = if pointee_is_scalar {
                    body_buf_elem_type(ctx, &ctx.module.functions[entry_idx], *pid)
                } else {
                    None
                };
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
                    if buffer_access_chains_match_struct_path(
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
                            },
                        )
                    } else if buffer_has_multi_index_access_chains(
                        &ctx.module.functions[entry_idx],
                        *pid,
                    ) && struct_buffer_needs_record_array(
                        &layout_defs,
                        &ctx.module.functions[entry_idx],
                        *pid,
                        st,
                    ) {
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
                    } else if let Some(elem) = flat_elem {
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
                            },
                        )
                    } else {
                        (
                            st,
                            BufWrap::Collapsed {
                                block_ty: st,
                                prepend_member0: false,
                            },
                        )
                    }
                }
            } else {
                // Genuine `device T*` array (no struct info): wrap as `{ RuntimeArray<T> }`. When
                // pre-llc canonicalization wrapped this buffer (`{[0 x T]}`), llc can collapse it back
                // to `T*` but keep the wrapper's member-0 index, so accesses already read
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
                    // llc can type the entry buffer param as a bare `uchar*` (opaque-pointer default)
                    // even though the inlined body indexes it as a `float`/`v2float` array — the
                    // RuntimeArray element must match what the body READS, not the mistyped pointee
                    // (else the chain indexes a uchar array but loads a float). Prefer the body's
                    // single-index element type.
                    let elem = body_buf_elem_type(ctx, &ctx.module.functions[entry_idx], *pid)
                        .unwrap_or(pointee);
                    let rta = ctx.ty_runtime_array(elem);
                    let st = ctx.module.fresh_id();
                    ctx.new_globals
                        .push(type_inst(Op::TypeStruct, st, vec![Operand::IdRef(rta)]));
                    (
                        st,
                        BufWrap::Collapsed {
                            block_ty: st,
                            prepend_member0: !already_has_wrapper_member0,
                        },
                    )
                }
            };
            let uptr = ctx.ty_ptr(StorageClass::StorageBuffer, struct_ty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(uptr),
                Some(var),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ));
            let binding = allocate_resource_binding(&mut binding_ctr, resource_binding);
            decorate_binding(&mut ctx.module, var, binding);
            bindings.push((*pid, ParamBinding::Buffer { var, wrap }));
            buffer_structs.push((var, struct_ty));
        } else if role_is("varying") {
            // Input var of the param value type at Location loc; load at entry.
            let pptr = ctx.ty_ptr(StorageClass::Input, *pty);
            let var = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Variable,
                Some(pptr),
                Some(var),
                vec![Operand::StorageClass(StorageClass::Input)],
            ));
            decorate_location(&mut ctx.module, var, loc);
            // Fragment Input varyings of integer / 64-bit-float type cannot be interpolated and must
            // be Flat-decorated (VUID-StandaloneSpirv-Flat-04744; banked `2ec0065d`, `110901bc`).
            if matches!(stage, Stage::Fragment) && fragment_input_needs_flat(&defs, *pty) {
                decorate_flat(&mut ctx.module, var);
            }
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
            // Expose the same LocalSize that finalize writes into the GLCompute execution mode.
            let [x, y, z] = ctx.kernel_local_size;
            let val = const_kernel_local_size(ctx, &defs, *pty, [x, y, z])
                .unwrap_or_else(|| ctx.const_uint(x));
            bindings.push((*pid, ParamBinding::Value { val }));
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
            let var = bind_kernel_v3uint_builtin_once(
                ctx,
                &mut num_workgroups_var,
                BuiltIn::NumWorkgroups,
            );
            bind_kernel_uvec3_builtin_var(ctx, &defs, &mut bindings, *pid, *pty, var);
        } else if role_is("threads_per_grid") {
            if let Some(threads) = ctx.kernel_threads_per_grid {
                let val = const_kernel_local_size(ctx, &defs, *pty, threads)
                    .unwrap_or_else(|| ctx.const_uint(threads[0]));
                bindings.push((*pid, ParamBinding::Value { val }));
            } else {
                let var = bind_kernel_v3uint_builtin_once(
                    ctx,
                    &mut num_workgroups_var,
                    BuiltIn::NumWorkgroups,
                );
                bind_kernel_threads_per_grid(ctx, &defs, &mut bindings, *pid, *pty, var);
            }
        } else if role_is("threadgroup_position_in_grid") {
            bind_kernel_uvec3_builtin(ctx, &defs, &mut bindings, *pid, *pty, BuiltIn::WorkgroupId);
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
            let simdgroups = ctx.kernel_local_size[0].div_ceil(32).max(1);
            let val = const_kernel_local_size(ctx, &defs, *pty, [simdgroups, 1, 1])
                .unwrap_or_else(|| ctx.const_uint(simdgroups));
            bindings.push((*pid, ParamBinding::Value { val }));
        } else if role_is("thread_position_in_grid") {
            let var = bind_kernel_v3uint_builtin_once(
                ctx,
                &mut global_invocation_id_var,
                BuiltIn::GlobalInvocationId,
            );
            bind_kernel_uvec3_builtin_var(ctx, &defs, &mut bindings, *pid, *pty, var);
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
    split_workgroup_block_type_aliases(ctx, &buffer_structs, &mut all_defs);
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
    handle_static_sampler(ctx, &mut binding_ctr);
    include_existing_private_globals(ctx);

    // Register textures embedded in an argument buffer (via `air.indirect_argument` → `air.texture`,
    // read by an integer-coord `air.read_texture`) as standalone sampled images BEFORE applying param
    // bindings. This lands them in `ctx.image_dims`/`ctx.image_comp` so the read-texture lowering's
    // `single_sampled_image_for_private_read` fallback finds exactly one sampled image and fetches
    // from it, instead of nulling the read.
    register_embedded_textures(ctx, entry_idx, kern, &mut binding_ctr);

    // Apply param bindings to the body: drop params, then splice replacements.
    apply_bindings(ctx, entry_idx, bindings, &buffer_structs, &all_defs)?;
    lower_buffer_address_facts(ctx, entry_idx, kern)?;

    Ok(defs)
}

/// Materialize a UniformConstant sampled image for each argument-buffer-embedded texture the meta
/// pass surfaced (see `KernMeta::embedded_textures`), decorate it at `TEXTURE_BINDING_BASE + K`
/// (K = the synthetic texture index the meta pass assigned via `embedded_synthetic_texture_index`,
/// the SAME convention the validation harness uses to bind the seeded texture), load it at entry, and
/// register the loaded image in `image_dims`/`image_comp`.
///
/// This is what turns the read of an argument-buffer-embedded texture (whose handle is a private
/// pointer loaded from the arg buffer) into a real `OpImageFetch`: the read-texture lowering falls
/// back to the single non-storage sampled image in `image_dims` when the operand is a private pointer,
/// and this registration provides exactly that image. Gated entirely on AIR structure (the meta pass
/// only fills `embedded_textures` for the `air.indirect_argument`→`air.texture`+`air.read_texture`
/// shape), never on a name.
fn register_embedded_textures(
    ctx: &mut Ctx,
    entry_idx: usize,
    kern: Option<&KernMeta>,
    binding_ctr: &mut u32,
) {
    let Some(kern) = kern else { return };
    if kern.embedded_textures.is_empty() {
        return;
    }
    let embedded = kern.embedded_textures.clone();
    let mut loads: Vec<Instruction> = vec![];
    for tex in embedded {
        let image_ty = ctx.ty_image(tex.dim, false, tex.comp);
        let pptr = ctx.ty_ptr(StorageClass::UniformConstant, image_ty);
        let var = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::Variable,
            Some(pptr),
            Some(var),
            vec![Operand::StorageClass(StorageClass::UniformConstant)],
        ));
        // ABI: bind at TEXTURE_BINDING_BASE + K. `allocate_resource_binding` with a fixed target
        // reserves exactly this slot (it does not consume the running counter).
        let binding = allocate_resource_binding(
            binding_ctr,
            Some(texture_resource_binding(tex.synthetic_texture_index)),
        );
        decorate_binding(&mut ctx.module, var, binding);
        ctx.interface_buffer_var(var); // SPIR-V 1.4+ lists every resource on the entry interface.
        let lid = ctx.module.fresh_id();
        loads.push(Instruction::new(
            Op::Load,
            Some(image_ty),
            Some(lid),
            vec![Operand::IdRef(var)],
        ));
        ctx.image_dims.insert(lid, (tex.dim, false));
        ctx.image_comp.insert(lid, tex.comp);
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
    use crate::spirv_module::Instruction;
    use crate::spirv_module::ModuleHeader;
    use crate::spirv_module::Operand;
    use spirv::Op;

    fn ty(op: Op, id: u32, operands: Vec<Operand>) -> Instruction {
        Instruction::new(op, None, Some(id), operands)
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
        split_workgroup_block_type_aliases(&mut ctx, &[(13, 5)], &mut defs);

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
}
