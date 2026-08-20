//! Depth sampling and comparison lowering.

use super::*;

/// Lower `air.sample_depth_<dim>.f32`: sample through the existing color-image path and return the
/// first component as the depth scalar. The harness currently seeds captured depth textures as
/// RGBA8_UNORM images, so this intentionally preserves the sampled-image contract instead of
/// introducing true depth-image formats before the cross-host evidence needs them.
pub(in crate::passes) fn lower_sample_depth(
    ctx: &mut Ctx,
    _name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
    v4: Word,
) -> Result<Vec<Instruction>, String> {
    let (res, rty) = match (res, rty) {
        (Some(r), Some(t)) => (r, t),
        _ => return Err("air.sample_depth has no result".into()),
    };
    if args.len() < 4 {
        return Err("air.sample_depth missing texture/sampler/coord".into());
    }
    let (mut img, samp, coord) = (resolve_image_value(ctx, args[0]), args[1], args[3]);
    if texture_operand_is_private_pointer(ctx, img) {
        if let Some(sampled_img) = single_sampled_image_for_private_read(ctx, img) {
            img = sampled_img;
        } else {
            return lower_null_texture_result(ctx, res, rty);
        }
    }
    let mut out = vec![];
    img = load_image_if_pointer(ctx, img, &mut out);
    let (fallback_dim, fallback_arrayed, fallback_comp) = image_shape_or_recorded(ctx, img);
    let (img_ty, dim, arrayed, comp) =
        sampled_operand_image_info(ctx, img, fallback_dim, fallback_arrayed, fallback_comp);
    if comp != crate::passes::ImageComp::Float {
        return Err("air.sample_depth on non-float texture".into());
    }
    let mut sample_args = vec![args[0], args[1], coord];
    if arrayed {
        sample_args.push(
            args.get(4)
                .copied()
                .ok_or("air.sample_depth array texture missing layer")?,
        );
        sample_args.extend_from_slice(&args[5..]);
    } else {
        sample_args.extend_from_slice(&args[4..]);
    }
    let pixel_state = ctx
        .sampler_states
        .get(&samp)
        .copied()
        .filter(|state| state.uses_pixel_coordinates());
    let color = if let Some(state) = pixel_state {
        let lod = ctx.const_uint(0);
        if state.uses_linear_filter() && matches!(dim, Dim::Dim2D | Dim::Dim3D) {
            lower_pixel_linear_sample(
                ctx,
                state,
                img,
                dim,
                arrayed,
                coord,
                &sample_args,
                lod,
                v4,
                &mut out,
            )?
        } else if state.uses_pixel_nearest() {
            let fetch = build_pixel_fetch_coord(
                ctx,
                state,
                img,
                dim,
                arrayed,
                coord,
                &sample_args,
                lod,
                &mut out,
            )?;
            let fetched = ctx.module.fresh_id();
            push_image_read_or_fetch(ctx, &mut out, img, fetch.coord, Some(lod), v4, fetched);
            if let Some(in_bounds) = fetch.in_bounds {
                let guarded = ctx.module.fresh_id();
                let zero = const_null_of(ctx, v4);
                out.push(Instruction::new(
                    Op::Select,
                    Some(v4),
                    Some(guarded),
                    vec![
                        Operand::IdRef(in_bounds),
                        Operand::IdRef(fetched),
                        Operand::IdRef(zero),
                    ],
                ));
                guarded
            } else {
                fetched
            }
        } else {
            return Err(format!(
                "pixel-coordinate depth sampling does not support {dim:?} with {:?} filtering",
                state.min_filter
            ));
        }
    } else {
        let si_ty = ctx.ty_sampled_image(img_ty);
        let si = ctx.module.fresh_id();
        let color = ctx.module.fresh_id();
        let samp = valid_sampler_value(ctx, samp, &mut out)?;
        out.push(Instruction::new(
            Op::SampledImage,
            Some(si_ty),
            Some(si),
            vec![Operand::IdRef(img), Operand::IdRef(samp)],
        ));
        // Depth-sample AIR has an extra scalar ABI operand before its spatial coordinate:
        // texture, sampler, control, coord, then layer for arrayed images. Ordinary color samples
        // put coord/layer at operands 2/3, so preserve the depth ABI explicitly.
        let coord_for_sample = if arrayed {
            let layer = args[4];
            let spatial = match dim {
                Dim::Dim1D => 1,
                Dim::Dim2D => 2,
                Dim::DimCube | Dim::Dim3D => 3,
                _ => return Err("air.sample_depth unsupported arrayed dimension".into()),
            };
            build_arrayed_sample_coord(ctx, spatial, coord, layer, &mut out)?
        } else {
            coord
        };
        push_image_sample(
            ctx,
            &mut out,
            v4,
            color,
            si,
            coord_for_sample,
            None,
            false,
            None,
        );
        color
    };
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

/// Lower `air.sample_compare_depth_<dim>.f32`: sample the RGBA8-backed depth fixture, extract
/// component 0, compare it against the AIR reference value, and return 1.0 or 0.0 in the AIR
/// `{float, i8}` shape. This mirrors `lower_sample_depth`'s current harness contract; it is not a
/// Vulkan Dref/comparison-sampler lowering.
pub(in crate::passes) fn lower_sample_compare_depth(
    ctx: &mut Ctx,
    _name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
    v4: Word,
) -> Result<Vec<Instruction>, String> {
    let (res, rty) = match (res, rty) {
        (Some(r), Some(t)) => (r, t),
        _ => return Err("air.sample_compare_depth has no result".into()),
    };
    if args.len() < 5 {
        return Err("air.sample_compare_depth missing texture/sampler/coord/reference".into());
    }
    let (mut img, samp, coord) = (resolve_image_value(ctx, args[0]), args[1], args[3]);
    if texture_operand_is_private_pointer(ctx, img) {
        if let Some(sampled_img) = single_sampled_image_for_private_read(ctx, img) {
            img = sampled_img;
        } else {
            return lower_null_texture_result(ctx, res, rty);
        }
    }
    let mut out = vec![];
    img = load_image_if_pointer(ctx, img, &mut out);
    let (fallback_dim, fallback_arrayed, fallback_comp) = image_shape_or_recorded(ctx, img);
    let (img_ty, dim, arrayed, comp) =
        sampled_operand_image_info(ctx, img, fallback_dim, fallback_arrayed, fallback_comp);
    if comp != crate::passes::ImageComp::Float {
        return Err("air.sample_compare_depth on non-float texture".into());
    }
    // Compare-depth uses the depth-sample ABI, not the ordinary color-sample ABI:
    // texture, sampler, control, spatial coord, [array layer], reference, flags...
    let (coord_for_sample, reference) = if arrayed {
        let layer = args
            .get(4)
            .copied()
            .ok_or("air.sample_compare_depth array texture missing layer")?;
        let reference = args
            .get(5)
            .copied()
            .ok_or("air.sample_compare_depth array texture missing reference")?;
        let spatial = match dim {
            Dim::Dim1D => 1,
            Dim::Dim2D => 2,
            Dim::DimCube | Dim::Dim3D => 3,
            _ => return Err("air.sample_compare_depth unsupported arrayed dimension".into()),
        };
        (
            build_arrayed_sample_coord(ctx, spatial, coord, layer, &mut out)?,
            reference,
        )
    } else {
        (coord, args[4])
    };
    let float_ty = ctx.ty_float();
    let bool_ty = ctx.ty_bool();
    let depth = ctx.module.fresh_id();
    let shadow = ctx.module.fresh_id();
    let one = ctx.const_float(1.0);
    let zero = ctx.const_float(0.0);
    let sampler_state = ctx.sampler_states.get(&samp).copied();
    if sampler_state
        .map(|state| state.compare_function == crate::reflect::SamplerCompareFunction::None)
        .unwrap_or(false)
    {
        return Err("air.sample_compare_depth requires a sampler with comparison enabled".into());
    }
    let color = if let Some(state) = sampler_state.filter(|state| state.uses_pixel_coordinates()) {
        let mut sample_args = vec![args[0], args[1], coord];
        let trailing = if arrayed {
            sample_args.push(args[4]);
            args.get(6..).unwrap_or_default()
        } else {
            args.get(5..).unwrap_or_default()
        };
        sample_args.extend_from_slice(trailing);
        let lod = ctx.const_uint(0);
        if state.uses_linear_filter() && matches!(dim, Dim::Dim2D | Dim::Dim3D) {
            lower_pixel_linear_sample(
                ctx,
                state,
                img,
                dim,
                arrayed,
                coord,
                &sample_args,
                lod,
                v4,
                &mut out,
            )?
        } else if state.uses_pixel_nearest() {
            let fetch = build_pixel_fetch_coord(
                ctx,
                state,
                img,
                dim,
                arrayed,
                coord,
                &sample_args,
                lod,
                &mut out,
            )?;
            let fetched = ctx.module.fresh_id();
            push_image_read_or_fetch(ctx, &mut out, img, fetch.coord, Some(lod), v4, fetched);
            if let Some(in_bounds) = fetch.in_bounds {
                let guarded = ctx.module.fresh_id();
                let zero_color = const_null_of(ctx, v4);
                out.push(Instruction::new(
                    Op::Select,
                    Some(v4),
                    Some(guarded),
                    vec![
                        Operand::IdRef(in_bounds),
                        Operand::IdRef(fetched),
                        Operand::IdRef(zero_color),
                    ],
                ));
                guarded
            } else {
                fetched
            }
        } else {
            return Err(format!(
                "pixel-coordinate depth comparison does not support {dim:?} with {:?} filtering",
                state.min_filter
            ));
        }
    } else {
        let si_ty = ctx.ty_sampled_image(img_ty);
        let si = ctx.module.fresh_id();
        let color = ctx.module.fresh_id();
        let valid_sampler = valid_sampler_value(ctx, samp, &mut out)?;
        out.push(Instruction::new(
            Op::SampledImage,
            Some(si_ty),
            Some(si),
            vec![Operand::IdRef(img), Operand::IdRef(valid_sampler)],
        ));
        push_image_sample(
            ctx,
            &mut out,
            v4,
            color,
            si,
            coord_for_sample,
            None,
            false,
            None,
        );
        color
    };
    out.push(Instruction::new(
        Op::CompositeExtract,
        Some(float_ty),
        Some(depth),
        vec![Operand::IdRef(color), Operand::LiteralBit32(0)],
    ));
    let compare = sampler_state
        .map(|state| state.compare_function)
        // Unspecialized runtime/selected samplers do not carry exact state through the AIR value
        // graph; preserve the pre-existing depth <= reference relation for that path.
        .unwrap_or(crate::reflect::SamplerCompareFunction::GreaterEqual);
    let passed = match compare {
        crate::reflect::SamplerCompareFunction::None
        | crate::reflect::SamplerCompareFunction::Never => ctx.const_bool_of(bool_ty, false),
        crate::reflect::SamplerCompareFunction::Always => ctx.const_bool_of(bool_ty, true),
        compare => {
            let passed = ctx.module.fresh_id();
            let opcode = match compare {
                crate::reflect::SamplerCompareFunction::Less => Op::FOrdLessThan,
                crate::reflect::SamplerCompareFunction::LessEqual => Op::FOrdLessThanEqual,
                crate::reflect::SamplerCompareFunction::Greater => Op::FOrdGreaterThan,
                crate::reflect::SamplerCompareFunction::GreaterEqual => Op::FOrdGreaterThanEqual,
                crate::reflect::SamplerCompareFunction::Equal => Op::FOrdEqual,
                crate::reflect::SamplerCompareFunction::NotEqual => Op::FOrdNotEqual,
                crate::reflect::SamplerCompareFunction::None
                | crate::reflect::SamplerCompareFunction::Always
                | crate::reflect::SamplerCompareFunction::Never => unreachable!(),
            };
            // Metal compares the incoming reference (the new value) against the sampled depth
            // (the existing value), matching MTLCompareFunction's ordering contract.
            out.push(Instruction::new(
                opcode,
                Some(bool_ty),
                Some(passed),
                vec![Operand::IdRef(reference), Operand::IdRef(depth)],
            ));
            passed
        }
    };
    out.push(Instruction::new(
        Op::Select,
        Some(float_ty),
        Some(shadow),
        vec![
            Operand::IdRef(passed),
            Operand::IdRef(one),
            Operand::IdRef(zero),
        ],
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
            vec![Operand::IdRef(shadow), Operand::IdRef(undef8)],
        ));
    } else {
        out.push(Instruction::new(
            Op::CopyObject,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(shadow)],
        ));
    }
    Ok(out)
}

pub(in crate::passes) fn find_sample_lod(
    ctx: &mut Ctx,
    arrayed: bool,
    args: &[Word],
    out: &mut Vec<Instruction>,
) -> Option<Word> {
    // Ordinary AIR sample forms put coord at arg[2]. Arrayed forms consume arg[3] as the layer, so a
    // scalar float after that is the explicit Metal `level(...)` operand. Non-LOD flags are integers.
    let start = if arrayed { 4 } else { 3 };
    args.get(start..)?
        .iter()
        .find_map(|arg| sample_lod_as_f32(ctx, *arg, out))
}

pub(in crate::passes) fn sample_spatial_dims(dim: Dim) -> Option<usize> {
    match dim {
        Dim::Dim1D => Some(1),
        Dim::Dim2D => Some(2),
        Dim::Dim3D => Some(3),
        _ => None,
    }
}

pub(in crate::passes) fn push_image_sample(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    result_ty: Word,
    result: Word,
    sampled_image: Word,
    coord: Word,
    explicit_lod: Option<Word>,
    force_lod0: bool,
    const_offset: Option<Word>,
) {
    let lod = explicit_lod.or_else(|| {
        (force_lod0 || matches!(ctx.stage, Stage::Kernel)).then(|| ctx.const_float(0.0))
    });
    let mut operands = vec![Operand::IdRef(sampled_image), Operand::IdRef(coord)];
    let mut image_operands = spirv::ImageOperands::empty();
    if lod.is_some() {
        image_operands |= spirv::ImageOperands::LOD;
    }
    if const_offset.is_some() {
        image_operands |= spirv::ImageOperands::CONST_OFFSET;
    }
    if !image_operands.is_empty() {
        operands.push(Operand::ImageOperands(image_operands));
        if let Some(lod) = lod {
            operands.push(Operand::IdRef(lod));
        }
        if let Some(offset) = const_offset {
            operands.push(Operand::IdRef(offset));
        }
    }
    out.push(Instruction::new(
        if lod.is_some() {
            Op::ImageSampleExplicitLod
        } else {
            Op::ImageSampleImplicitLod
        },
        Some(result_ty),
        Some(result),
        operands,
    ));
}

pub(in crate::passes) fn sample_lod_as_f32(
    ctx: &mut Ctx,
    lod: Word,
    out: &mut Vec<Instruction>,
) -> Option<Word> {
    let ty = value_result_type(ctx, lod)?;
    let def = type_def_of(ctx, ty)?;
    if def.class.opcode != Op::TypeFloat {
        return None;
    }
    let width = match def.operands.first() {
        Some(Operand::LiteralBit32(w)) => *w,
        _ => return None,
    };
    if width == 32 {
        return Some(lod);
    }
    if width == 16 {
        let widened = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FConvert,
            Some(ctx.ty_float()),
            Some(widened),
            vec![Operand::IdRef(lod)],
        ));
        return Some(widened);
    }
    None
}
