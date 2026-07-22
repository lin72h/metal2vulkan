use super::*;

pub(in crate::passes) fn data_pointer_pointee(
    defs: &HashMap<Word, Instruction>,
    pty: Word,
) -> Option<Word> {
    let pointee = ptr_pointee(defs, pty)?;
    let opaque = matches!(
        defs.get(&pointee).map(|d| d.class.opcode),
        Some(
            Op::TypeImage
                | Op::TypeSampler
                | Op::TypeSampledImage
                | Op::TypeAccelerationStructureKHR
        )
    );
    if opaque {
        None
    } else {
        Some(pointee)
    }
}

/// Whether access chains rooted directly at `pid` already carry the synthetic wrapper-struct member
/// index (`%p %uint_0 %i`). A genuine `device T*` array of arrays/vectors can also produce multi-index
/// chains (`%p %dynamic %member`), and those still need the wrapper member inserted after re-rooting
/// onto the StorageBuffer variable.
pub(in crate::passes) fn access_chains_include_wrapper_member0(
    defs: &HashMap<Word, Instruction>,
    func: &Function,
    pid: Word,
) -> bool {
    let mut saw_chain = false;
    for blk in &func.blocks {
        for inst in &blk.instructions {
            if matches!(
                inst.class.opcode,
                Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
            ) && inst.operands.first() == Some(&Operand::IdRef(pid))
            {
                let indices = &inst.operands[1..];
                if indices.len() < 2 {
                    return false;
                }
                saw_chain = true;
                let Some(Operand::IdRef(first_index)) = indices.first() else {
                    return false;
                };
                if const_int_literal(defs, *first_index) != Some(0) {
                    return false;
                }
            }
        }
    }
    saw_chain
}

fn const_int_literal(defs: &HashMap<Word, Instruction>, id: Word) -> Option<u64> {
    let inst = defs.get(&id)?;
    if inst.class.opcode != Op::Constant {
        return None;
    }
    match inst.operands.first() {
        Some(Operand::LiteralBit64(v)) => Some(*v),
        Some(Operand::LiteralBit32(v)) => Some(*v as u64),
        _ => None,
    }
}

fn type_after_access_indices(
    defs: &HashMap<Word, Instruction>,
    root_ty: Word,
    indices: &[Operand],
) -> Option<Word> {
    let mut cur = root_ty;
    for idx in indices {
        let def = defs.get(&cur)?;
        cur = match def.class.opcode {
            Op::TypeStruct => {
                let Operand::IdRef(idx_id) = idx else {
                    return None;
                };
                let member = const_int_literal(defs, *idx_id)? as usize;
                match def.operands.get(member) {
                    Some(Operand::IdRef(member_ty)) => *member_ty,
                    _ => return None,
                }
            }
            Op::TypeArray | Op::TypeRuntimeArray | Op::TypeVector => match def.operands.first() {
                Some(Operand::IdRef(elem_ty)) => *elem_ty,
                _ => return None,
            },
            _ => return None,
        };
    }
    Some(cur)
}

fn value_result_type(ctx: &Ctx, func: &Function, value: Word) -> Option<Word> {
    for global in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if global.result_id == Some(value) {
            return global.result_type;
        }
    }
    for param in &func.parameters {
        if param.result_id == Some(value) {
            return param.result_type;
        }
    }
    for block in &func.blocks {
        for inst in &block.instructions {
            if inst.result_id == Some(value) {
                return inst.result_type;
            }
        }
    }
    None
}

pub(in crate::passes) fn access_chain_matches_direct_struct_path(
    defs: &HashMap<Word, Instruction>,
    struct_ty: Word,
    indices: &[Operand],
    result_ptr_ty: Option<Word>,
) -> bool {
    let Some(result_ptr_ty) = result_ptr_ty else {
        return false;
    };
    let Some(result_pointee) = ptr_pointee(defs, result_ptr_ty) else {
        return false;
    };
    type_after_access_indices(defs, struct_ty, indices) == Some(result_pointee)
}

/// AIR `ptr addrspace(2) %buffer, i64 N, ...` on a struct-typed buffer param can mean record `N` of an
/// implicit array of that struct. The native emitter correctly omits LLVM GEP's leading zero for
/// normal record-0 member paths, so do not key off a lone `%buffer %uint_1`: that is usually struct
/// member 1. Instead, find chains whose existing indices do not type-check as direct struct-member
/// paths. Those are the record-array uses that need a `{ RuntimeArray<Struct> }` wrapper.
pub(in crate::passes) fn struct_buffer_needs_record_array(
    defs: &HashMap<Word, Instruction>,
    func: &Function,
    pid: Word,
    struct_ty: Word,
) -> bool {
    for blk in &func.blocks {
        for inst in &blk.instructions {
            if matches!(
                inst.class.opcode,
                Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
            ) && inst.operands.first() == Some(&Operand::IdRef(pid))
                && !access_chain_matches_direct_struct_path(
                    defs,
                    struct_ty,
                    &inst.operands[1..],
                    inst.result_type,
                )
            {
                return true;
            }
        }
    }
    false
}

pub(in crate::passes) fn buffer_has_access_chains(func: &Function, pid: Word) -> bool {
    func.blocks.iter().any(|blk| {
        blk.instructions.iter().any(|inst| {
            matches!(
                inst.class.opcode,
                Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
            ) && inst.operands.first() == Some(&Operand::IdRef(pid))
        })
    })
}

pub(in crate::passes) fn buffer_has_multi_index_access_chains(func: &Function, pid: Word) -> bool {
    func.blocks.iter().any(|blk| {
        blk.instructions.iter().any(|inst| {
            matches!(
                inst.class.opcode,
                Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
            ) && inst.operands.first() == Some(&Operand::IdRef(pid))
                && inst.operands.len() > 2
        })
    })
}

pub(in crate::passes) fn buffer_access_chains_match_struct_path(
    defs: &HashMap<Word, Instruction>,
    func: &Function,
    pid: Word,
    struct_ty: Word,
) -> bool {
    for blk in &func.blocks {
        for inst in &blk.instructions {
            if matches!(
                inst.class.opcode,
                Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
            ) && inst.operands.first() == Some(&Operand::IdRef(pid))
                && !access_chain_matches_direct_struct_path(
                    defs,
                    struct_ty,
                    &inst.operands[1..],
                    inst.result_type,
                )
            {
                return false;
            }
        }
    }
    true
}

pub(in crate::passes) fn buffer_has_struct_path_access_chain(
    defs: &HashMap<Word, Instruction>,
    func: &Function,
    pid: Word,
    struct_ty: Word,
) -> bool {
    func.blocks.iter().any(|blk| {
        blk.instructions.iter().any(|inst| {
            matches!(
                inst.class.opcode,
                Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
            ) && inst.operands.first() == Some(&Operand::IdRef(pid))
                && access_chain_matches_direct_struct_path(
                    defs,
                    struct_ty,
                    &inst.operands[1..],
                    inst.result_type,
                )
        })
    })
}

/// The ELEMENT type the body actually reads/writes a buffer param as. For indexed uses this is the
/// pointee of a single-index `OpAccessChain %pid %i`; for direct scalar/vector load/store uses it is
/// the loaded/stored value type. Used when llc typed the entry param as a bare `uchar*`
/// (opaque-pointer default) yet the inlined body accesses it as a `float`/`v2float`/`v4uint` array.
/// The interface's `RuntimeArray` element type must match the body access, not the mistyped param
/// pointee, else `spirv-val` sees pointer/result-type mismatches.
/// Returns None if no access roots at `pid` (then keep the declared pointee).
pub(in crate::passes) fn body_buf_elem_type(ctx: &Ctx, func: &Function, pid: Word) -> Option<Word> {
    let defs = type_defs(&ctx.module);
    for blk in &func.blocks {
        for inst in &blk.instructions {
            if matches!(inst.class.opcode, Op::AccessChain | Op::InBoundsAccessChain)
                && inst.operands.first() == Some(&Operand::IdRef(pid))
                && inst.operands.len() == 2
            // exactly one index -> bare buf[i]
            {
                // result_type is a pointer to the element type.
                if let Some(rt) = inst.result_type {
                    if let Some(elem) = ptr_pointee(&defs, rt) {
                        return Some(elem);
                    }
                }
            }
            if inst.class.opcode == Op::Load && inst.operands.first() == Some(&Operand::IdRef(pid))
            {
                if let Some(result_type) = inst.result_type {
                    return Some(result_type);
                }
            }
            if inst.class.opcode == Op::Store && inst.operands.first() == Some(&Operand::IdRef(pid))
            {
                if let Some(Operand::IdRef(value)) = inst.operands.get(1) {
                    if let Some(result_type) = value_result_type(ctx, func, *value) {
                        return Some(result_type);
                    }
                }
            }
        }
    }
    None
}

/// Map each texture PARAMETER id -> (Dim, arrayed, comp) by scanning the entry's sampled/read texture
/// calls (arg 0 is the texture operand, which at this stage is still the param id). The callee name
/// encodes the dimension (`sample_texture_2d`, ...) and the sampled component type via its suffix
/// (`.u.v4i32` = uint, `.s.v4i32` = sint, else float).
pub(in crate::passes) fn texture_dims(
    ctx: &Ctx,
    entry_idx: usize,
    type_hints: &HashMap<Word, (Dim, bool, ImageComp)>,
) -> HashMap<Word, (Dim, bool, ImageComp)> {
    let names = air_names(&ctx.module);
    let write_textures = write_texture_operands(ctx, entry_idx, &names);
    let mut out = HashMap::new();
    // Textures that are ever sampled/gathered with a normalized direction/coordinate. A cube texture
    // in this set must stay `Dim Cube`; one outside it is only ever texel-fetched (`read`) or
    // size-queried, and Vulkan forbids `OpImageFetch` on a Cube image, so it is re-typed below.
    let mut direction_sampled: HashSet<Word> = HashSet::new();
    for blk in &ctx.module.functions[entry_idx].blocks {
        for inst in &blk.instructions {
            if inst.class.opcode != Op::FunctionCall {
                continue;
            }
            let Some(Operand::IdRef(callee)) = inst.operands.first() else {
                continue;
            };
            let Some(name) = names.get(callee) else {
                continue;
            };
            if name.starts_with("air.sample_") || name.starts_with("air.gather_") {
                if let Some(Operand::IdRef(tex)) = inst.operands.get(1) {
                    direction_sampled.insert(*tex);
                }
            }
            if !(name.starts_with("air.sample_texture")
                || name.starts_with("air.sample_depth")
                || name.starts_with("air.read_texture")
                || name.starts_with("air.read_depth")
                || name.starts_with("air.is_null_texture")
                || name.starts_with("air.get_width_texture")
                || name.starts_with("air.get_height_texture")
                || name.starts_with("air.get_depth_texture")
                || name.starts_with("air.get_array_size_texture")
                || name.starts_with("air.get_width_depth")
                || name.starts_with("air.get_height_depth")
                || name.starts_with("air.get_depth_depth")
                || name.starts_with("air.get_num_mip_levels_texture")
                || name.starts_with("air.get_num_samples_texture"))
            {
                continue;
            }
            // arg 0 (operand index 1) is the texture.
            let Some(Operand::IdRef(tex)) = inst.operands.get(1) else {
                continue;
            };
            if ((name.starts_with("air.read_texture") || name.starts_with("air.read_depth"))
                || is_size_query(name)
                || name.starts_with("air.is_null_texture"))
                && write_textures.contains(tex)
            {
                continue;
            }
            let (dim, arrayed) = sample_dim(name);
            let comp = texture_comp(name)
                .or_else(|| type_hints.get(tex).map(|(_, _, comp)| *comp))
                .unwrap_or(ImageComp::Float);
            // Let a suffix-derived integer use refine a provisional Float classification from an
            // earlier ambiguous query. Valid Metal does not mix float and integer texel views for the
            // same texture argument.
            out.entry(*tex)
                .and_modify(|e: &mut (Dim, bool, ImageComp)| {
                    if e.2 == ImageComp::Float {
                        e.2 = comp;
                    }
                })
                .or_insert((dim, arrayed, comp));
        }
    }
    // Vulkan forbids `OpImageFetch` on a `Dim Cube` image (cube texel fetch does not exist in
    // SPIR-V), so a cube texture that is only ever texel-read/size-queried binds as a 2D ARRAY
    // image instead: a cube IS a 6-layer 2D array, and the AIR read/write arg shape already passes
    // the face index in the array-layer slot, so the ordinary arrayed fetch path applies verbatim.
    // A cube that is ever direction-sampled keeps `Dim Cube` (sampling needs the cube view; if such
    // a texture is ALSO texel-read the fetch still fails validation — a real remaining gap rather
    // than a silently wrong emit). Cube ARRAYS are left untouched: their reads carry both face and
    // element and need a fused layer computation this re-type alone would silently drop.
    for (tex, entry) in out.iter_mut() {
        if entry.0 == Dim::DimCube && !entry.1 && !direction_sampled.contains(tex) {
            *entry = (Dim::Dim2D, true, entry.2);
        }
    }
    out
}

fn write_texture_operands(
    ctx: &Ctx,
    entry_idx: usize,
    names: &HashMap<Word, String>,
) -> HashSet<Word> {
    let mut out = HashSet::new();
    for blk in &ctx.module.functions[entry_idx].blocks {
        for inst in &blk.instructions {
            if inst.class.opcode != Op::FunctionCall {
                continue;
            }
            let Some(Operand::IdRef(callee)) = inst.operands.first() else {
                continue;
            };
            if !names
                .get(callee)
                .map(|name| is_write_texture_name(name))
                .unwrap_or(false)
            {
                continue;
            }
            if let Some(Operand::IdRef(tex)) = inst.operands.get(1) {
                out.insert(*tex);
            }
        }
    }
    out
}

fn is_size_query(name: &str) -> bool {
    name.starts_with("air.get_width_texture")
        || name.starts_with("air.get_height_texture")
        || name.starts_with("air.get_depth_texture")
        || name.starts_with("air.get_array_size_texture")
        || name.starts_with("air.get_width_depth")
        || name.starts_with("air.get_height_depth")
        || name.starts_with("air.get_depth_depth")
        || name.starts_with("air.get_num_mip_levels_texture")
        || name.starts_with("air.get_num_samples_texture")
}

/// Map each WRITE-texture PARAMETER id -> (Dim, arrayed, ImageFormat, ImageComp) by scanning the
/// entry's `air.write_texture_*` calls (arg 0 is the texture operand = the param id at this stage).
/// The texel type suffix (`.u.v4i32` / `.s.v4i32` / `.v4f16` / `.v4f32`) picks the storage image
/// format. A texture that is BOTH sampled through a sampler and written is not handled here (it would
/// need a read_write storage image with format coexisting with sampling — out of scope); such a param
/// falls back to the sampled-image binding and write lowering aborts -> shader FALLBACKs cleanly.
pub(in crate::passes) fn write_texture_dims(
    ctx: &Ctx,
    entry_idx: usize,
) -> HashMap<Word, (Dim, bool, ImageFormat, ImageComp)> {
    let names = air_names(&ctx.module);
    let mut out = HashMap::new();
    for blk in &ctx.module.functions[entry_idx].blocks {
        for inst in &blk.instructions {
            if inst.class.opcode != Op::FunctionCall {
                continue;
            }
            let Some(Operand::IdRef(callee)) = inst.operands.first() else {
                continue;
            };
            let Some(name) = names.get(callee) else {
                continue;
            };
            if !is_write_texture_name(name) {
                continue;
            }
            let Some(Operand::IdRef(tex)) = inst.operands.get(1) else {
                continue;
            };
            let (dim, arrayed) = sample_dim(name);
            let (fmt, comp) = if name.contains(".u.") && name.contains(".v4i16") {
                (ImageFormat::Rgba16ui, ImageComp::Uint)
            } else if name.contains(".u.") {
                (ImageFormat::Rgba8ui, ImageComp::Uint)
            } else if name.contains(".s.") {
                (ImageFormat::Rgba8i, ImageComp::Sint)
            } else if name.ends_with(".v4f16") || name.contains(".v4f16") {
                (ImageFormat::Rgba16f, ImageComp::Float)
            } else {
                (ImageFormat::Rgba32f, ImageComp::Float)
            };
            out.entry(*tex).or_insert((dim, arrayed, fmt, comp));
        }
    }
    out
}

fn is_write_texture_name(name: &str) -> bool {
    name.starts_with("air.write_texture")
        || name.starts_with("air.write_imageblock_slice_to_texture")
}

/// Map each metadata-declared write-capable texture PARAMETER id to its storage-image shape. Helper
/// inlining can wrap textures in small value structs, so `air.write_texture_*` may see a later
/// `CompositeExtract` rather than the original parameter id. The AIR interface metadata still names
/// the parameter's access mode and texture type.
pub(in crate::passes) fn texture_storage_hints(
    params: &[(Word, Word)],
    stage: &Stage,
    frag: Option<&FragMeta>,
    vert: Option<&VertMeta>,
    kern: Option<&KernMeta>,
) -> HashMap<Word, (Dim, bool, ImageFormat, ImageComp)> {
    params
        .iter()
        .enumerate()
        .filter_map(|(i, (pid, _))| {
            let idx = i as u32;
            let name = match stage {
                Stage::Fragment => frag.and_then(|m| m.texture_type_name(idx)),
                Stage::Vertex => vert.and_then(|m| m.texture_type_name(idx)),
                Stage::Kernel => kern.and_then(|m| m.texture_type_name(idx)),
            }?;
            texture_arg_storage(name).map(|shape| (*pid, shape))
        })
        .collect()
}

/// Parse the sampled component type a texture callee name implies. Metal integer textures encode
/// signedness as a `.u.`/`.s.` segment before the return-type token (`air.read_texture_2d.u.v4i32`).
/// Size queries carry no return-vector suffix, so their caller must fall back to AIR metadata.
fn texture_comp(name: &str) -> Option<ImageComp> {
    if name.contains(".u.") {
        Some(ImageComp::Uint)
    } else if name.contains(".s.") {
        Some(ImageComp::Sint)
    } else if is_size_query(name) || name.starts_with("air.is_null_texture") {
        None
    } else {
        Some(ImageComp::Float)
    }
}

pub(in crate::passes) fn texture_arg_comp(name: &str) -> ImageComp {
    crate::meta::texture_shape_from_name(name)
        .component
        .to_image_comp()
}

fn texture_arg_storage(name: &str) -> Option<(Dim, bool, ImageFormat, ImageComp)> {
    let shape = crate::meta::texture_shape_from_name(name);
    let fmt = shape.storage_format?.to_spirv_format();
    Some((
        shape.dimension.to_spirv_dim(),
        shape.arrayed,
        fmt,
        shape.component.to_image_comp(),
    ))
}

pub(in crate::passes) fn texture_arg_dim(name: &str) -> (Dim, bool) {
    let shape = crate::meta::texture_shape_from_name(name);
    (shape.dimension.to_spirv_dim(), shape.arrayed)
}

/// Parse the (Dim, arrayed) a sample callee name implies.
fn sample_dim(name: &str) -> (Dim, bool) {
    if name.contains("texture_buffer") {
        (Dim::DimBuffer, false)
    } else if name.contains("_1d_array") {
        (Dim::Dim1D, true)
    } else if name.contains("_2d_ms_array") {
        (Dim::Dim2D, true)
    } else if name.contains("_1d") {
        (Dim::Dim1D, false)
    } else if name.contains("_3d") {
        (Dim::Dim3D, false)
    } else if name.contains("_cube_array") {
        (Dim::DimCube, true)
    } else if name.contains("_cube") {
        (Dim::DimCube, false)
    } else if name.contains("_2d_array") {
        (Dim::Dim2D, true)
    } else {
        (Dim::Dim2D, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_arg_comp_reads_nested_texture_array_scalar() {
        assert_eq!(
            texture_arg_comp("array<texture2d<uint, sample>, 2>"),
            ImageComp::Uint
        );
        assert_eq!(
            texture_arg_comp("array<texture2d<int, sample>, 2>"),
            ImageComp::Sint
        );
        assert_eq!(
            texture_arg_comp("array<texture2d<half, sample>, 2>"),
            ImageComp::Float
        );
    }
}
