//! Structural image-value resolution and image/type coercion.

use super::*;

pub(in crate::passes) fn resolve_image_value(ctx: &Ctx, value: Word) -> Word {
    let mut current = value;
    for _ in 0..8 {
        if ctx.image_dims.contains_key(&current) || ctx.image_storage.contains(&current) {
            if !value_is_pointer(ctx, current) {
                return current;
            }
            if let Some(loaded) = single_loaded_value(ctx, current) {
                current = loaded;
                continue;
            }
        }
        let Some(inst) = value_inst(ctx, current) else {
            return current;
        };
        match inst.class.opcode {
            Op::CompositeExtract => {
                let Some(Operand::IdRef(composite)) = inst.operands.first() else {
                    return current;
                };
                let path = literal_path(&inst.operands[1..]);
                let Some(source) = resolve_composite_insert_path(ctx, *composite, &path) else {
                    return current;
                };
                current = source;
            }
            Op::CopyObject => {
                let Some(Operand::IdRef(source)) = inst.operands.first() else {
                    return current;
                };
                current = *source;
            }
            Op::Load => {
                let Some(Operand::IdRef(pointer)) = inst.operands.first() else {
                    return current;
                };
                let Some(stored) = single_stored_value(ctx, *pointer) else {
                    return current;
                };
                current = stored;
            }
            Op::Variable if value_is_pointer(ctx, current) => {
                let Some(loaded) = single_loaded_value(ctx, current) else {
                    return current;
                };
                current = loaded;
            }
            _ => return current,
        }
    }
    current
}

pub(in crate::passes) fn resolve_composite_insert_path(
    ctx: &Ctx,
    value: Word,
    path: &[u32],
) -> Option<Word> {
    let inst = value_inst(ctx, value)?;
    match inst.class.opcode {
        Op::CompositeInsert => {
            let inserted = match inst.operands.first()? {
                Operand::IdRef(id) => *id,
                _ => return None,
            };
            let base = match inst.operands.get(1)? {
                Operand::IdRef(id) => *id,
                _ => return None,
            };
            let insert_path = literal_path(&inst.operands[2..]);
            if insert_path == path {
                Some(inserted)
            } else {
                resolve_composite_insert_path(ctx, base, path)
            }
        }
        Op::CopyObject => match inst.operands.first()? {
            Operand::IdRef(source) => resolve_composite_insert_path(ctx, *source, path),
            _ => None,
        },
        _ => None,
    }
}

pub(in crate::passes) fn literal_path(operands: &[Operand]) -> Vec<u32> {
    operands
        .iter()
        .filter_map(|operand| match operand {
            Operand::LiteralBit32(value) => Some(*value),
            _ => None,
        })
        .collect()
}

pub(in crate::passes) fn value_inst(ctx: &Ctx, value: Word) -> Option<&Instruction> {
    ctx.module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
        .chain(
            ctx.module
                .functions
                .iter()
                .flat_map(|function| function.blocks.iter())
                .flat_map(|block| block.instructions.iter()),
        )
        .find(|inst| inst.result_id == Some(value))
}

pub(in crate::passes) fn single_stored_value(ctx: &Ctx, pointer: Word) -> Option<Word> {
    let mut found = None;
    for inst in ctx
        .module
        .functions
        .iter()
        .flat_map(|function| function.blocks.iter())
        .flat_map(|block| block.instructions.iter())
    {
        if inst.class.opcode != Op::Store {
            continue;
        }
        if inst.operands.first() != Some(&Operand::IdRef(pointer)) {
            continue;
        }
        let Some(Operand::IdRef(value)) = inst.operands.get(1) else {
            return None;
        };
        if found.replace(*value).is_some() {
            return None;
        }
    }
    found
}

pub(in crate::passes) fn single_loaded_value(ctx: &Ctx, pointer: Word) -> Option<Word> {
    let mut found = None;
    for inst in ctx
        .module
        .functions
        .iter()
        .flat_map(|function| function.blocks.iter())
        .flat_map(|block| block.instructions.iter())
    {
        if inst.class.opcode != Op::Load {
            continue;
        }
        if inst.operands.first() != Some(&Operand::IdRef(pointer)) {
            continue;
        }
        let value = inst.result_id?;
        if found.replace(value).is_some() {
            return None;
        }
    }
    found
}

pub(in crate::passes) fn image_is_storage(ctx: &Ctx, img: Word) -> bool {
    if ctx.image_storage.contains(&img) {
        return true;
    }
    let Some(ty) = value_result_type(ctx, img) else {
        return false;
    };
    let Some(def) = type_def_of(ctx, ty) else {
        return false;
    };
    if def.class.opcode != Op::TypeImage {
        return false;
    }
    matches!(def.operands.get(5), Some(Operand::LiteralBit32(2)))
}

pub(in crate::passes) fn single_storage_image_for_private_write(
    ctx: &Ctx,
    img: Word,
) -> Option<Word> {
    // Helper wrappers can store a write texture inside a Function aggregate and load it again from a
    // callee. The native emitter models Function pointer fields as integer storage, so if field replay
    // misses the cross-function case, the inlined write sees a Private zero pointer. When metadata
    // still produced exactly one storage-image binding, that binding is the only legal write target.
    if !texture_operand_is_private_pointer(ctx, img) || ctx.image_storage.len() != 1 {
        return None;
    }
    ctx.image_storage.iter().copied().next()
}

pub(in crate::passes) fn single_image_for_private_query(ctx: &Ctx, img: Word) -> Option<Word> {
    // Texture size/level/sample queries carry no access-mode suffix, so a helper-wrapper private
    // placeholder is recoverable only when the interface has exactly one real image binding total.
    if !texture_operand_is_private_pointer(ctx, img) {
        return None;
    }
    let mut candidates = ctx
        .image_dims
        .keys()
        .copied()
        .filter(|id| !ctx.null_image_values.contains(id));
    let candidate = candidates.next()?;
    if candidates.next().is_some() {
        return None;
    }
    Some(candidate)
}

pub(in crate::passes) fn single_sampled_image_for_private_read(
    ctx: &Ctx,
    img: Word,
) -> Option<Word> {
    // Same helper-wrapper failure mode as writes, but for a sampled/read texture operand. Keep this
    // to the unambiguous case: exactly one non-storage, non-null image binding exists.
    if !texture_operand_is_private_pointer(ctx, img) {
        return None;
    }
    let mut candidates = ctx
        .image_dims
        .keys()
        .copied()
        .filter(|id| !ctx.image_storage.contains(id) && !ctx.null_image_values.contains(id));
    let candidate = candidates.next()?;
    if candidates.next().is_some() {
        return None;
    }
    Some(candidate)
}

pub(in crate::passes) fn describe_value(ctx: &Ctx, value: Word) -> String {
    let Some(inst) = value_inst(ctx, value) else {
        return "no defining instruction".to_string();
    };
    let operands = inst
        .operands
        .iter()
        .map(|operand| match operand {
            Operand::IdRef(id) => format!("IdRef({id}: {})", describe_value(ctx, *id)),
            _ => format!("{operand:?}"),
        })
        .collect::<Vec<_>>()
        .join(", ");
    let stores = store_count(ctx, value);
    format!("{:?} [{}] stores={stores}", inst.class.opcode, operands)
}

pub(in crate::passes) fn store_count(ctx: &Ctx, pointer: Word) -> usize {
    ctx.module
        .functions
        .iter()
        .flat_map(|function| function.blocks.iter())
        .flat_map(|block| block.instructions.iter())
        .filter(|inst| {
            inst.class.opcode == Op::Store
                && inst.operands.first() == Some(&Operand::IdRef(pointer))
        })
        .count()
}

pub(in crate::passes) fn texture_operand_is_private_pointer(ctx: &Ctx, img: Word) -> bool {
    let Some(ty) = value_result_type(ctx, img) else {
        return false;
    };
    let Some(def) = type_def_of(ctx, ty) else {
        return false;
    };
    def.class.opcode == Op::TypePointer
        && matches!(
            def.operands.first(),
            Some(Operand::StorageClass(StorageClass::Private))
        )
}

pub(in crate::passes) fn value_is_pointer(ctx: &Ctx, value: Word) -> bool {
    let Some(ty) = value_result_type(ctx, value) else {
        return false;
    };
    type_def_of(ctx, ty)
        .map(|def| def.class.opcode == Op::TypePointer)
        .unwrap_or(false)
}

pub(in crate::passes) fn lower_null_texture_result(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
) -> Result<Vec<Instruction>, String> {
    let rdef = type_def_of(ctx, rty);
    let is_struct = rdef
        .as_ref()
        .map(|d| d.class.opcode == Op::TypeStruct)
        .unwrap_or(false);
    if is_struct {
        let member0 = rdef
            .as_ref()
            .and_then(|d| d.operands.first())
            .and_then(|o| match o {
                Operand::IdRef(m) => Some(*m),
                _ => None,
            })
            .ok_or("null texture result struct missing member 0")?;
        let zero_color = const_null_of(ctx, member0);
        let i8u = ctx.ty_int8();
        let undef8 = ctx.module.fresh_id();
        return Ok(vec![
            Instruction::new(Op::Undef, Some(i8u), Some(undef8), vec![]),
            Instruction::new(
                Op::CompositeConstruct,
                Some(rty),
                Some(res),
                vec![Operand::IdRef(zero_color), Operand::IdRef(undef8)],
            ),
        ]);
    }

    let zero = const_null_of(ctx, rty);
    Ok(vec![Instruction::new(
        Op::CopyObject,
        Some(rty),
        Some(res),
        vec![Operand::IdRef(zero)],
    )])
}

pub(in crate::passes) fn const_null_of(ctx: &mut Ctx, ty: Word) -> Word {
    let id = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::ConstantNull,
        Some(ty),
        Some(id),
        vec![],
    ));
    id
}

pub(in crate::passes) fn coerce_same_shape_integer(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    value: Word,
    value_ty: Word,
    target_ty: Word,
) -> Result<Word, String> {
    if value_ty == target_ty {
        return Ok(value);
    }
    let Some(value_shape) = integer_shape(ctx, value_ty) else {
        return Ok(value);
    };
    let Some(target_shape) = integer_shape(ctx, target_ty) else {
        return Ok(value);
    };
    if value_shape.1 != target_shape.1 {
        return Ok(value);
    }
    let cast = ctx.module.fresh_id();
    let op = if value_shape.0 == target_shape.0 {
        Op::Bitcast
    } else if integer_is_signed(ctx, target_ty).unwrap_or(false) {
        Op::SConvert
    } else {
        Op::UConvert
    };
    out.push(Instruction::new(
        op,
        Some(target_ty),
        Some(cast),
        vec![Operand::IdRef(value)],
    ));
    Ok(cast)
}

pub(in crate::passes) fn integer_shape(ctx: &Ctx, ty: Word) -> Option<(u32, u32)> {
    let def = type_def_of(ctx, ty)?;
    match def.class.opcode {
        Op::TypeInt => {
            let bits = match def.operands.first()? {
                Operand::LiteralBit32(bits) => *bits,
                _ => return None,
            };
            Some((bits, 1))
        }
        Op::TypeVector => {
            let elem = match def.operands.first()? {
                Operand::IdRef(elem) => *elem,
                _ => return None,
            };
            let lanes = match def.operands.get(1)? {
                Operand::LiteralBit32(lanes) => *lanes,
                _ => return None,
            };
            let (bits, elem_lanes) = integer_shape(ctx, elem)?;
            (elem_lanes == 1).then_some((bits, lanes))
        }
        _ => None,
    }
}

pub(in crate::passes) fn integer_is_signed(ctx: &Ctx, ty: Word) -> Option<bool> {
    let def = type_def_of(ctx, ty)?;
    match def.class.opcode {
        Op::TypeInt => {
            let signed = match def.operands.get(1)? {
                Operand::LiteralBit32(signed) => *signed,
                _ => return None,
            };
            Some(signed != 0)
        }
        Op::TypeVector => {
            let elem = match def.operands.first()? {
                Operand::IdRef(elem) => *elem,
                _ => return None,
            };
            integer_is_signed(ctx, elem)
        }
        _ => None,
    }
}

pub(in crate::passes) fn gather_const_or_dynamic_offset(
    ctx: &Ctx,
    value: Word,
) -> Result<(Option<[i32; 2]>, Option<Word>), String> {
    let ty = value_result_type(ctx, value).ok_or("air.gather_texture offset has no type")?;
    let def = type_def_of(ctx, ty).ok_or("air.gather_texture offset type is undefined")?;
    let lanes = match def.class.opcode {
        Op::TypeVector => match def.operands.get(1) {
            Some(Operand::LiteralBit32(lanes)) => *lanes,
            _ => return Err("air.gather_texture offset vector missing length".into()),
        },
        _ => return Err("air.gather_texture offset is not an integer vector".into()),
    };
    if lanes != 2 {
        return Err("air.gather_texture offset has unexpected vector length".into());
    }
    match const_i32_component_slots::<2>(ctx, value) {
        Ok(Some(slots)) => {
            let mut values = [0; 2];
            for (idx, slot) in slots.into_iter().enumerate() {
                let Some(value) = slot else {
                    return Err("air.gather_texture offset component is undef".into());
                };
                values[idx] = value;
            }
            Ok((Some(values), None))
        }
        Ok(None) => Ok((None, Some(value))),
        Err(err)
            if err.contains("offset component is not i32")
                || err.contains("offset composite insert base is not constant") =>
        {
            Ok((None, Some(value)))
        }
        Err(err) => Err(err),
    }
}

pub(in crate::passes) fn const_i32_components<const N: usize>(
    ctx: &Ctx,
    value: Word,
    expected_lanes: u32,
) -> Result<Option<[i32; N]>, String> {
    let ty = value_result_type(ctx, value).ok_or("air.gather_texture offset has no type")?;
    let def = type_def_of(ctx, ty).ok_or("air.gather_texture offset type is undefined")?;
    let lanes = match def.class.opcode {
        Op::TypeVector => match def.operands.get(1) {
            Some(Operand::LiteralBit32(lanes)) => *lanes,
            _ => return Err("air.gather_texture offset vector missing length".into()),
        },
        _ => return Err("air.gather_texture offset is not an integer vector".into()),
    };
    if lanes != expected_lanes || lanes as usize != N {
        return Err("air.gather_texture offset has unexpected vector length".into());
    }
    let slots = const_i32_component_slots::<N>(ctx, value)?
        .ok_or("air.gather_texture offset is not constant")?;
    let mut values = [0; N];
    for (idx, slot) in slots.into_iter().enumerate() {
        let Some(value) = slot else {
            return Err("air.gather_texture offset component is undef".into());
        };
        values[idx] = value;
    }
    Ok(Some(values))
}

pub(in crate::passes) fn const_i32_component_slots<const N: usize>(
    ctx: &Ctx,
    value: Word,
) -> Result<Option<[Option<i32>; N]>, String> {
    let Some(inst) = value_inst(ctx, value) else {
        return Ok(None);
    };
    match inst.class.opcode {
        Op::ConstantNull => Ok(Some([Some(0); N])),
        Op::ConstantComposite | Op::CompositeConstruct => {
            let mut values = [None; N];
            for (idx, operand) in inst.operands.iter().enumerate().take(N) {
                let Operand::IdRef(id) = operand else {
                    return Err("air.gather_texture offset component is not an id".into());
                };
                values[idx] = Some(const_i32_scalar(ctx, *id).ok_or_else(|| {
                    format!(
                        "air.gather_texture offset component is not i32: {:?}",
                        value_inst(ctx, *id).map(|inst| inst.class.opcode)
                    )
                })?);
            }
            Ok(Some(values))
        }
        Op::CompositeInsert => {
            if inst.operands.len() != 3 {
                return Err(
                    "air.gather_texture offset composite insert has unexpected shape".into(),
                );
            }
            let Operand::IdRef(component) = inst.operands[0] else {
                return Err(
                    "air.gather_texture offset composite insert component is not an id".into(),
                );
            };
            let Operand::IdRef(base) = inst.operands[1] else {
                return Err("air.gather_texture offset composite insert base is not an id".into());
            };
            let Operand::LiteralBit32(lane) = inst.operands[2] else {
                return Err(
                    "air.gather_texture offset composite insert lane is not literal".into(),
                );
            };
            let lane = lane as usize;
            if lane >= N {
                return Err(
                    "air.gather_texture offset composite insert lane is out of range".into(),
                );
            }
            let mut values = const_i32_component_slots::<N>(ctx, base)?
                .ok_or("air.gather_texture offset composite insert base is not constant")?;
            values[lane] = Some(const_i32_scalar(ctx, component).ok_or_else(|| {
                format!(
                    "air.gather_texture offset component is not i32: {:?}",
                    value_inst(ctx, component).map(|inst| inst.class.opcode)
                )
            })?);
            Ok(Some(values))
        }
        Op::Undef => Ok(Some([None; N])),
        _ => Ok(None),
    }
}

pub(in crate::passes) fn const_i32_scalar(ctx: &Ctx, value: Word) -> Option<i32> {
    let inst = const_inst(ctx, value)?;
    if inst.class.opcode != Op::Constant {
        return None;
    }
    match inst.operands.first()? {
        Operand::LiteralBit32(value) => Some(*value as i32),
        _ => None,
    }
}

pub(in crate::passes) fn const_inst(ctx: &Ctx, value: Word) -> Option<&Instruction> {
    ctx.module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
        .find(|inst| inst.result_id == Some(value))
}

pub(in crate::passes) fn const_sint_vec(ctx: &mut Ctx, values: &[i32]) -> Word {
    let ty = ctx.ty_vec_sint(values.len() as u32);
    let int_ty = ctx.ty_sint();
    let operands = values
        .iter()
        .copied()
        .map(|value| Operand::IdRef(ctx.const_int_of(int_ty, value as i64)))
        .collect();
    let id = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::ConstantComposite,
        Some(ty),
        Some(id),
        operands,
    ));
    id
}

/// The v4 vector type an OpImageFetch/OpImageSample on image `img` must produce: v4uint / v4int for an
/// integer texture (recorded in `image_comp`), else the supplied float `v4`.
pub(in crate::passes) fn image_fetch_v4(ctx: &mut Ctx, img: Word, v4: Word) -> Word {
    match ctx.image_comp.get(&img).copied() {
        Some(crate::passes::ImageComp::Uint) => ctx.ty_vec_uint(4),
        Some(crate::passes::ImageComp::Sint) => ctx.ty_vec_sint(4),
        _ => v4,
    }
}

pub(in crate::passes) fn vector_element_type(ctx: &Ctx, ty: Word) -> Option<Word> {
    let def = type_def_of(ctx, ty)?;
    if def.class.opcode != Op::TypeVector {
        return None;
    }
    match def.operands.first()? {
        Operand::IdRef(elem) => Some(*elem),
        _ => None,
    }
}

/// Build the coordinate operand for an OpImageSample matching `(dim, arrayed)`. AIR's spatial coord is
/// `coord` (arg[2]); for arrayed samples the integer layer index is arg[3], which we convert to float
/// and append as the last coordinate component (Vulkan array sampling encodes the layer in the coord).
pub(in crate::passes) fn sample_coord_components(
    ctx: &mut Ctx,
    coord: Word,
    ncomp: u32,
    out: &mut Vec<Instruction>,
) -> Result<Vec<Operand>, String> {
    // The coord may be a value built earlier in THIS lowering and still buffered in `out` (e.g. the
    // combined array/cube coord from `build_sample_coord`, a `CompositeConstruct` not yet committed to
    // the module). `value_result_type` only scans `ctx.module`/`new_globals`, so resolve the in-progress
    // `out` buffer first before falling back to the committed module.
    let ty = out
        .iter()
        .rev()
        .find(|i| i.result_id == Some(coord))
        .and_then(|i| i.result_type)
        .or_else(|| value_result_type(ctx, coord))
        .ok_or("air.sample_texture coord has no type")?;
    let def = type_def_of(ctx, ty).ok_or("air.sample_texture coord type is undefined")?;
    match def.class.opcode {
        Op::TypeFloat if ncomp == 1 => Ok(vec![Operand::IdRef(coord)]),
        Op::TypeVector => {
            let Some(Operand::IdRef(elem)) = def.operands.first() else {
                return Err("air.sample_texture vector coord missing element type".into());
            };
            let Some(Operand::LiteralBit32(n)) = def.operands.get(1) else {
                return Err("air.sample_texture vector coord missing length".into());
            };
            let is_float_elem = type_def_of(ctx, *elem)
                .map(|e| e.class.opcode == Op::TypeFloat)
                .unwrap_or(false);
            if !is_float_elem || *n != ncomp {
                return Err("air.sample_texture unexpected vector coord shape".into());
            }
            let mut comps = Vec::new();
            for c in 0..*n {
                let id = ctx.module.fresh_id();
                out.push(Instruction::new(
                    Op::CompositeExtract,
                    Some(ctx.ty_float()),
                    Some(id),
                    vec![Operand::IdRef(coord), Operand::LiteralBit32(c)],
                ));
                comps.push(Operand::IdRef(id));
            }
            Ok(comps)
        }
        _ => Err("air.sample_texture unsupported coord shape".into()),
    }
}

pub(in crate::passes) fn sample_layer_to_float(
    ctx: &mut Ctx,
    layer: Word,
    out: &mut Vec<Instruction>,
) -> Result<Word, String> {
    let ty = value_result_type(ctx, layer).ok_or("air.sample_texture layer has no type")?;
    let def = type_def_of(ctx, ty).ok_or("air.sample_texture layer type is undefined")?;
    if def.class.opcode == Op::TypeFloat {
        return Ok(layer);
    }
    if def.class.opcode != Op::TypeInt {
        return Err("air.sample_texture layer is not scalar int/float".into());
    }
    let signed = matches!(def.operands.get(1), Some(Operand::LiteralBit32(1)));
    let layer_f = ctx.module.fresh_id();
    out.push(Instruction::new(
        if signed {
            Op::ConvertSToF
        } else {
            Op::ConvertUToF
        },
        Some(ctx.ty_float()),
        Some(layer_f),
        vec![Operand::IdRef(layer)],
    ));
    Ok(layer_f)
}

pub(in crate::passes) fn build_sample_coord(
    ctx: &mut Ctx,
    dim: Dim,
    arrayed: bool,
    coord: Word,
    args: &[Word],
    out: &mut Vec<Instruction>,
) -> Result<Word, String> {
    if !arrayed {
        return Ok(coord); // 1D->float, 2D->v2float, 3D/cube->v3float: pass the AIR coord directly.
    }
    // Arrayed samples encode the layer as the final float coordinate component.
    let layer_i = args
        .get(3)
        .copied()
        .ok_or("air.sample_texture array texture missing layer")?;
    let spatial = match dim {
        Dim::Dim1D => 1,
        Dim::Dim2D => 2,
        Dim::DimCube | Dim::Dim3D => 3,
        _ => return Err("air.sample_texture unsupported arrayed dimension".into()),
    };
    build_arrayed_sample_coord(ctx, spatial, coord, layer_i, out)
}
