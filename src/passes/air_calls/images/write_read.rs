//! Texture image read and write lowering.

use super::*;

/// Lower `air.write_texture_<dim>.<coord>.<texel>(image, coord, texel, lod, access)` -> `OpImageWrite`.
/// The image (arg0) must be a bound STORAGE image (Sampled=2 with an explicit ImageFormat). AIR passes
/// non-arrayed writes as `(image, coord, texel, lod, access)`, cube writes as
/// `(image, coord, face, texel, lod, access)`, and array writes as
/// `(image, coord, layer, texel, lod, access)`. We coerce float writes to `<4 x float>` and integer
/// writes to the storage image's signedness, while the coord becomes the integer coordinate shape the
/// image expects. Anything we can't express (non-storage image, unexpected texel/coord shape, a LOD
/// operand a basic storage image can't take) -> Err so the shader FALLBACKs cleanly rather than
/// emitting an invalid OpImageWrite.
pub(in crate::passes) fn lower_write(
    ctx: &mut Ctx,
    args: &[Word],
    v4: Word,
) -> Result<Vec<Instruction>, String> {
    if args.len() < 3 {
        return Err("air.write_texture missing image/coord/texel".into());
    }
    let mut img = resolve_image_value(ctx, args[0]);
    if !image_is_storage(ctx, img) {
        if let Some(storage_img) = single_storage_image_for_private_write(ctx, img) {
            img = storage_img;
        } else {
            // Not a storage image (e.g. a texture also sampled, bound as Sampled=1) -> can't
            // OpImageWrite.
            return Err(format!(
                "air.write_texture on non-storage image id {img} ({})",
                describe_value(ctx, img)
            ));
        }
    }
    let mut out = vec![];

    let (dim, arrayed) = ctx
        .image_dims
        .get(&img)
        .copied()
        .unwrap_or((Dim::Dim2D, false));
    if crate::env_vars::tex_dbg() {
        let tys: Vec<String> = args
            .iter()
            .map(|a| {
                value_result_type(ctx, *a)
                    .and_then(|t| type_def_of(ctx, t))
                    .map(|d| format!("{}:{:?}", a, d.class.opcode))
                    .unwrap_or_else(|| format!("{a}:?"))
            })
            .collect();
        eprintln!("WRITE-TEX img {img} dim {dim:?} arrayed {arrayed} args {tys:?}");
    }
    let (coord, layer, texel) = if dim == Dim::DimCube && !arrayed {
        if args.len() < 4 {
            return Err("air.write_texture cube write missing face/texel".into());
        }
        (args[1], Some(args[2]), args[3])
    } else if arrayed {
        if args.len() < 4 {
            return Err("air.write_texture array write missing layer/texel".into());
        }
        (args[1], Some(args[2]), args[3])
    } else {
        (args[1], None, args[2])
    };

    let coord32 = build_fetch_coord(ctx, dim, arrayed, coord, layer, &mut out)?;

    let comp = ctx
        .image_comp
        .get(&img)
        .copied()
        .unwrap_or(crate::passes::ImageComp::Float);
    let texel32 = match comp {
        crate::passes::ImageComp::Float => match vector_shape(ctx, texel) {
            Some((elem, 4)) if is_float_width(ctx, elem, 32) => texel,
            Some((elem, 4)) if is_float_width(ctx, elem, 16) => {
                let t = ctx.module.fresh_id();
                out.push(Instruction::new(
                    Op::FConvert,
                    Some(v4),
                    Some(t),
                    vec![Operand::IdRef(texel)],
                ));
                t
            }
            _ => return Err("air.write_texture: unsupported texel shape".into()),
        },
        crate::passes::ImageComp::Uint | crate::passes::ImageComp::Sint => {
            let texel_ty = value_result_type(ctx, texel)
                .ok_or("air.write_texture: integer texel has no result type")?;
            if !matches!(resolve::integer_shape(ctx, texel_ty), Some((_bits, 4))) {
                return Err("air.write_texture: unsupported integer texel shape".into());
            }
            let target_ty = match comp {
                crate::passes::ImageComp::Uint => ctx.ty_vec_uint(4),
                crate::passes::ImageComp::Sint => ctx.ty_vec_sint(4),
                crate::passes::ImageComp::Float => {
                    return Err("air.write_texture: Float comp in the integer texel path".into())
                }
            };
            coerce_same_shape_integer(ctx, &mut out, texel, texel_ty, target_ty)?
        }
    };

    let texel32 = preserve_defined_texel_lanes(ctx, &mut out, img, coord32, texel, texel32);

    out.push(Instruction::new(
        Op::ImageWrite,
        None,
        None,
        vec![
            Operand::IdRef(img),
            Operand::IdRef(coord32),
            Operand::IdRef(texel32),
        ],
    ));
    Ok(out)
}

fn preserve_defined_texel_lanes(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    img: Word,
    coord32: Word,
    source_texel: Word,
    write_texel: Word,
) -> Word {
    let undefined = (0..4)
        .map(|lane| texel_lane_is_statically_undef(ctx, out, source_texel, lane))
        .collect::<Vec<_>>();
    if undefined.iter().all(|is_undef| !*is_undef) {
        return write_texel;
    }

    let Some(write_ty) = pending_or_module_result_type(ctx, out, write_texel) else {
        return write_texel;
    };
    let lane_ty = element_type(ctx, write_ty);
    let current = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ImageRead,
        Some(write_ty),
        Some(current),
        vec![Operand::IdRef(img), Operand::IdRef(coord32)],
    ));

    let mut merged = current;
    for (lane, is_undef) in undefined.into_iter().enumerate() {
        if is_undef {
            continue;
        }
        let lane_value = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::CompositeExtract,
            Some(lane_ty),
            Some(lane_value),
            vec![
                Operand::IdRef(write_texel),
                Operand::LiteralBit32(lane as u32),
            ],
        ));
        let next = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::CompositeInsert,
            Some(write_ty),
            Some(next),
            vec![
                Operand::IdRef(lane_value),
                Operand::IdRef(merged),
                Operand::LiteralBit32(lane as u32),
            ],
        ));
        merged = next;
    }
    merged
}

fn pending_or_module_result_type(ctx: &Ctx, pending: &[Instruction], value: Word) -> Option<Word> {
    pending
        .iter()
        .rev()
        .find(|inst| inst.result_id == Some(value))
        .and_then(|inst| inst.result_type)
        .or_else(|| value_result_type(ctx, value))
}

fn texel_lane_is_statically_undef(
    ctx: &Ctx,
    pending: &[Instruction],
    value: Word,
    lane: usize,
) -> bool {
    let Some(inst) = pending
        .iter()
        .rev()
        .find(|inst| inst.result_id == Some(value))
        .or_else(|| value_inst(ctx, value))
    else {
        return false;
    };
    match inst.class.opcode {
        Op::Undef => true,
        Op::CompositeConstruct => inst
            .operands
            .get(lane)
            .and_then(|operand| match operand {
                Operand::IdRef(id) => Some(texel_lane_is_statically_undef(ctx, pending, *id, 0)),
                _ => None,
            })
            .unwrap_or(false),
        Op::CompositeInsert => {
            let inserted = inst.operands.first().and_then(|operand| match operand {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            let base = inst.operands.get(1).and_then(|operand| match operand {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            });
            let target_lane = inst.operands.get(2).and_then(|operand| match operand {
                Operand::LiteralBit32(index) => Some(*index as usize),
                _ => None,
            });
            if target_lane == Some(lane) {
                inserted
                    .map(|id| texel_lane_is_statically_undef(ctx, pending, id, 0))
                    .unwrap_or(false)
            } else {
                base.map(|id| texel_lane_is_statically_undef(ctx, pending, id, lane))
                    .unwrap_or(false)
            }
        }
        Op::CopyObject | Op::UConvert | Op::SConvert | Op::FConvert | Op::Bitcast => inst
            .operands
            .first()
            .and_then(|operand| match operand {
                Operand::IdRef(id) => Some(texel_lane_is_statically_undef(ctx, pending, *id, lane)),
                _ => None,
            })
            .unwrap_or(false),
        _ => false,
    }
}

/// Lower `air.read_texture_<dim>.<ret>`: a sampler-less texel read by integer coordinate (Metal's
/// `texture.read(uint2 coord, lod)`). AIR args: a0=texture(loaded image), a1=integer coord, a2=lod,
/// (a3=access flag, ignored). Sampled images use `OpImageFetch` with the integer coord + Lod;
/// read-write storage images use `OpImageRead` and ignore the LOD because storage images are not
/// mipmapped in the current harness. The read produces the image component vector, then converts to
/// the AIR result shape when needed.
pub(in crate::passes) fn lower_read(
    ctx: &mut Ctx,
    _name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
    v4: Word,
) -> Result<Vec<Instruction>, String> {
    let (res, rty) = match (res, rty) {
        (Some(r), Some(t)) => (r, t),
        _ => return Err("air.read_texture has no result".into()),
    };
    if args.len() < 2 {
        return Err("air.read_texture missing texture/coord".into());
    }
    // AIR `read_texture_<dim>` comes in several arg shapes:
    //   * no-sampler: (texture, coord, lod, access)                       -> coord=a1, lod=a2
    //   * array no-sampler: (texture, coord, layer, lod, access)          -> coord=a1, layer=a2, lod=a3
    //   * with-sampler: (texture, sampler, coord, lod, access)            -> coord=a2, lod=a3
    //   * array with-sampler: (texture, sampler, coord, layer, lod, access) -> coord=a2, layer=a3, lod=a4
    //   * with-sampler+offset: (texture, sampler, coord, offset, lod, access) -> coord=a2, lod=a4
    // Distinguish by whether a1 is the integer COORD (no-sampler) or a sampler value (with-sampler):
    // the coord is an integer scalar/vector; a sampler is a pointer/sampler type.
    let mut img = resolve_image_value(ctx, args[0]);
    if texture_operand_is_private_pointer(ctx, img) {
        if let Some(sampled_img) = single_sampled_image_for_private_read(ctx, img) {
            img = sampled_img;
        } else {
            return lower_null_texture_result(ctx, res, rty);
        }
    }
    let (dim, arrayed) = ctx
        .image_dims
        .get(&img)
        .copied()
        .unwrap_or((Dim::Dim2D, false));
    // A cube read carries the face index in the array-layer slot (`(tex[, sampler], coord, face,
    // lod, access)`), so parse it with the arrayed shape.
    let cube = dim == Dim::DimCube && !arrayed;
    let (coord, layer, lod) = if value_is_int_or_intvec(ctx, args[1]) {
        if arrayed || cube {
            (args[1], args.get(2).copied(), args.get(3).copied())
        } else {
            (args[1], None, args.get(2).copied())
        }
    } else if arrayed || cube {
        let lod_idx = if args.len() >= 7 { 5 } else { 4 };
        (
            *args.get(2).ok_or("air.read_texture missing coord")?,
            args.get(3).copied(),
            args.get(lod_idx).copied(),
        )
    } else if args.len() == 5 {
        (
            *args.get(2).ok_or("air.read_texture missing coord")?,
            None,
            args.get(3).copied(),
        )
    } else {
        (
            *args.get(2).ok_or("air.read_texture missing coord")?,
            None,
            args.get(4).copied(),
        )
    };
    let mut out = vec![];
    // OpImageFetch's result vector component type must match the image's sampled type. For integer
    // textures the fetch yields v4uint/v4int, not v4float.
    let fetch_v4 = image_fetch_v4(ctx, img, v4);
    let color = ctx.module.fresh_id();
    if cube && !image_is_storage(ctx, img) {
        // Vulkan has no cube texel fetch (`OpImageFetch` forbids `Dim Cube`). A cube that reaches
        // this path stayed `Dim Cube` because it is ALSO direction-sampled elsewhere, so lower the
        // fetch to an equivalent direction sample: reconstruct the direction that hits the exact
        // texel center of (coord, face) and sample with a nearest sampler at explicit LOD. Nearest
        // sampling at a texel center returns that texel's bytes exactly.
        let face = layer.ok_or("air.read_texture cube read missing face")?;
        let sampler_arg = (!value_is_int_or_intvec(ctx, args[1])).then_some(args[1]);
        cube_fetch_as_center_sample(
            ctx,
            &mut out,
            img,
            coord,
            face,
            fetch_v4,
            color,
            sampler_arg,
        )?;
    } else {
        let coord32 = build_fetch_coord(ctx, dim, arrayed, coord, layer, &mut out)?;
        let lod32 = match lod {
            Some(lod) => Some(coerce_image_coord32(
                ctx,
                lod,
                &mut out,
                "air.read_texture lod",
            )?),
            None => None,
        };
        let mut ops = vec![Operand::IdRef(img), Operand::IdRef(coord32)];
        let op = if image_is_storage(ctx, img) {
            Op::ImageRead
        } else {
            if let Some(lod) = lod32 {
                ops.push(Operand::ImageOperands(spirv::ImageOperands::LOD));
                ops.push(Operand::IdRef(lod));
            }
            Op::ImageFetch
        };
        out.push(Instruction::new(op, Some(fetch_v4), Some(color), ops));
    }
    // Result shape: bare `<4 x half>` -> FConvert; `{<4 x half>, i8}` struct -> build; v4float -> copy.
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
            });
        let c = match member0 {
            Some(m) if m != v4 && is_half_vector(ctx, m) => {
                let cc = ctx.module.fresh_id();
                out.push(Instruction::new(
                    Op::FConvert,
                    Some(m),
                    Some(cc),
                    vec![Operand::IdRef(color)],
                ));
                cc
            }
            Some(m) => coerce_same_shape_integer(ctx, &mut out, color, fetch_v4, m)?,
            _ => color,
        };
        let i8u = ctx.ty_int8();
        let undef8 = ctx.module.fresh_id();
        out.push(Instruction::new(Op::Undef, Some(i8u), Some(undef8), vec![]));
        out.push(Instruction::new(
            Op::CompositeConstruct,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(c), Operand::IdRef(undef8)],
        ));
    } else if rty != v4 && is_half_vector(ctx, rty) {
        out.push(Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(color)],
        ));
    } else if resolve::integer_shape(ctx, rty).is_some() {
        let c = coerce_same_shape_integer(ctx, &mut out, color, fetch_v4, rty)?;
        out.push(Instruction::new(
            Op::CopyObject,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(c)],
        ));
    } else {
        out.push(Instruction::new(
            Op::CopyObject,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(color)],
        ));
    }
    Ok(out)
}

/// Lower `air.read_depth_<dim>.f32`: read an integer-coordinate depth texel through the current
/// RGBA8-backed harness image, extract component 0, and rebuild AIR's `{float, i8}` result shape.
pub(in crate::passes) fn lower_read_depth(
    ctx: &mut Ctx,
    _name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
    v4: Word,
) -> Result<Vec<Instruction>, String> {
    let (res, rty) = match (res, rty) {
        (Some(r), Some(t)) => (r, t),
        _ => return Err("air.read_depth has no result".into()),
    };
    if args.len() < 3 {
        return Err("air.read_depth missing texture/coord".into());
    }
    let mut img = resolve_image_value(ctx, args[0]);
    if texture_operand_is_private_pointer(ctx, img) {
        if let Some(sampled_img) = single_sampled_image_for_private_read(ctx, img) {
            img = sampled_img;
        } else {
            return lower_null_texture_result(ctx, res, rty);
        }
    }
    let (dim, arrayed) = ctx
        .image_dims
        .get(&img)
        .copied()
        .unwrap_or((Dim::Dim2D, false));
    let comp = ctx
        .image_comp
        .get(&img)
        .copied()
        .unwrap_or(crate::passes::ImageComp::Float);
    if comp != crate::passes::ImageComp::Float {
        return Err("air.read_depth on non-float texture".into());
    }

    // AIR read_depth forms mirror read_texture, with an extra sample-index operand before the coord
    // on depth reads. Array forms insert a scalar `layer` operand immediately AFTER the coord (same
    // placement read_texture uses), e.g. `air.read_depth_2d_array.f32(tex, sampler, sample_index,
    // coord, layer, offset, lod, access)`:
    //   * no-sampler: (texture, coord, lod, access) -> coord=a1, lod=a2
    //     array: (texture, coord, layer, lod, access) -> coord=a1, layer=a2, lod=a3
    //   * no-sampler/sample-index: (texture, sample_index, coord, lod, access) -> coord=a2, lod=a3
    //     array: (texture, sample_index, coord, layer, lod, access) -> coord=a2, layer=a3, lod=a4
    //   * with-sampler: (texture, sampler, sample_index, coord, offset, lod, access) -> coord=a3, lod=a5
    //     array (+offset): (texture, sampler, sample_index, coord, layer, offset, lod, access)
    //       -> coord=a3, layer=a4, lod=a6
    //   * with-sampler/no-offset: (texture, sampler, sample_index, coord, lod, access) -> coord=a3, lod=a4
    //     array/no-offset: (texture, sampler, sample_index, coord, layer, lod, access)
    //       -> coord=a3, layer=a4, lod=a5
    let (coord, layer, lod) = if value_is_int_or_intvec(ctx, args[1]) {
        // No-sampler: a1 is the coord, unless a1 is a scalar sample-index before a vector coord.
        let coord_idx = if vector_shape(ctx, args[1]).is_none()
            && args
                .get(2)
                .is_some_and(|coord| vector_shape(ctx, *coord).is_some())
        {
            2
        } else {
            1
        };
        let coord = *args.get(coord_idx).ok_or("air.read_depth missing coord")?;
        if arrayed {
            (
                coord,
                args.get(coord_idx + 1).copied(),
                args.get(coord_idx + 2).copied(),
            )
        } else {
            (coord, None, args.get(coord_idx + 1).copied())
        }
    } else {
        let coord = *args.get(3).ok_or("air.read_depth missing coord")?;
        if arrayed {
            // layer at a4; an optional offset sits between layer and lod (8 args ⇒ offset present).
            (
                coord,
                args.get(4).copied(),
                args.get(if args.len() >= 8 { 6 } else { 5 }).copied(),
            )
        } else {
            (
                coord,
                None,
                args.get(if args.len() >= 7 { 5 } else { 4 }).copied(),
            )
        }
    };
    let mut out = vec![];
    let coord32 = build_fetch_coord(ctx, dim, arrayed, coord, layer, &mut out)?;
    let lod32 = match lod {
        Some(lod) => Some(coerce_image_coord32(
            ctx,
            lod,
            &mut out,
            "air.read_depth lod",
        )?),
        None => None,
    };
    let color = ctx.module.fresh_id();
    let mut ops = vec![Operand::IdRef(img), Operand::IdRef(coord32)];
    if let Some(lod) = lod32 {
        ops.push(Operand::ImageOperands(spirv::ImageOperands::LOD));
        ops.push(Operand::IdRef(lod));
    }
    out.push(Instruction::new(Op::ImageFetch, Some(v4), Some(color), ops));
    let depth = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::CompositeExtract,
        Some(ctx.ty_float()),
        Some(depth),
        vec![Operand::IdRef(color), Operand::LiteralBit32(0)],
    ));

    let rdef = type_def_of(ctx, rty);
    let is_struct = rdef
        .as_ref()
        .map(|d| d.class.opcode == Op::TypeStruct)
        .unwrap_or(false);
    if is_struct {
        let i8u = ctx.ty_int8();
        let undef8 = ctx.module.fresh_id();
        out.push(Instruction::new(Op::Undef, Some(i8u), Some(undef8), vec![]));
        out.push(Instruction::new(
            Op::CompositeConstruct,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(depth), Operand::IdRef(undef8)],
        ));
    } else {
        out.push(Instruction::new(
            Op::CopyObject,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(depth)],
        ));
    }
    Ok(out)
}
