//! Color texture sampling and pixel-filter lowering.

use super::*;

/// Lower air.sample_texture_<dim>: combine the texture+sampler operands via OpSampledImage and emit
/// OpImageSampleExplicitLod when AIR carries a scalar float LOD operand, when integer textures require
/// it, or when a compute kernel cannot legally use implicit derivatives.
/// AIR call args: a0=texture(loaded image), a1=sampler(loaded sampler), a2=coord, then optional
/// layer/LOD/flags. Result is AIR's {vecN,i8}; we reproduce it as OpCompositeConstruct so the
/// downstream CompositeExtract 0 still yields the color.
pub(in crate::passes) fn lower_sample(
    ctx: &mut Ctx,
    _name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
    v4: Word,
) -> Result<Vec<Instruction>, String> {
    let (res, rty) = match (res, rty) {
        (Some(r), Some(t)) => (r, t),
        _ => return Err("air.sample has no result".into()),
    };
    if args.len() < 3 {
        return Err("air.sample missing texture/sampler/coord".into());
    }
    let (mut img, samp, coord) = (resolve_image_value(ctx, args[0]), args[1], args[2]);
    if texture_operand_is_private_pointer(ctx, img) {
        if let Some(sampled_img) = single_sampled_image_for_private_read(ctx, img) {
            img = sampled_img;
        } else {
            return lower_null_texture_result(ctx, res, rty);
        }
    }
    // Build the sampled-image type matching the texture's dimension (recorded when the image var was
    // created). Defaults to 2D for the common case.
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
    // The sample result vector component type must match the image's sampled type (v4uint/v4int for
    // integer textures); a float sample with explicit LOD is needed at fragment scope for integer
    // textures (implicit-LOD sampling of integer images is invalid).
    let sample_v4 = image_fetch_v4(ctx, img, v4);
    let is_int_tex = comp != crate::passes::ImageComp::Float;

    let mut out = vec![];
    if let Some(sampler_state) = ctx
        .sampler_states
        .get(&samp)
        .copied()
        .filter(|state| state.uses_pixel_coordinates())
    {
        // Pixel-coordinate linear float sampling is emulated in-shader (per-tap fetch + lerp) for
        // 2D AND 3D: Vulkan forbids an unnormalized-coordinates sampler on a Dim3d image view, so
        // a plain OpImageSample with raw pixel coords can never be the 3D lowering (vulkano rejects
        // the pipeline/descriptor pairing at dispatch).
        let pixel_linear = comp == crate::passes::ImageComp::Float
            && sampler_state.uses_linear_filter()
            && matches!(dim, Dim::Dim2D | Dim::Dim3D);
        let pixel_fetch = comp != crate::passes::ImageComp::Float
            || sampler_state.uses_pixel_nearest()
            || arrayed;
        if pixel_linear || pixel_fetch {
            let lod = match find_sample_lod(ctx, arrayed, args, &mut out) {
                Some(lod) => sample_lod_to_fetch_lod(ctx, lod, &mut out)?,
                None => ctx.const_uint(0),
            };
            if pixel_linear {
                let color = lower_pixel_linear_sample(
                    ctx,
                    sampler_state,
                    img,
                    dim,
                    arrayed,
                    coord,
                    args,
                    lod,
                    sample_v4,
                    &mut out,
                )?;
                return finish_sample_result(ctx, res, rty, color, sample_v4, out);
            }
            let fetch = build_pixel_fetch_coord(
                ctx,
                sampler_state,
                img,
                dim,
                arrayed,
                coord,
                args,
                lod,
                &mut out,
            )?;
            let mut color = ctx.module.fresh_id();
            push_image_read_or_fetch(ctx, &mut out, img, fetch.coord, Some(lod), sample_v4, color);
            if let Some(in_bounds) = fetch.in_bounds {
                let guarded = ctx.module.fresh_id();
                let zero_color = const_null_of(ctx, sample_v4);
                out.push(Instruction::new(
                    Op::Select,
                    Some(sample_v4),
                    Some(guarded),
                    vec![
                        Operand::IdRef(in_bounds),
                        Operand::IdRef(color),
                        Operand::IdRef(zero_color),
                    ],
                ));
                color = guarded;
            }
            return finish_sample_result(ctx, res, rty, color, sample_v4, out);
        }
    }
    if is_int_tex {
        if let Some(sampler_state) = ctx.sampler_states.get(&samp).copied() {
            if sample_uses_normalized_coords(ctx, arrayed, args) {
                let lod = match find_sample_lod(ctx, arrayed, args, &mut out) {
                    Some(lod) => sample_lod_to_fetch_lod(ctx, lod, &mut out)?,
                    None => ctx.const_uint(0),
                };
                let fetch = build_normalized_nearest_fetch_coord(
                    ctx,
                    sampler_state,
                    img,
                    dim,
                    arrayed,
                    coord,
                    args,
                    lod,
                    &mut out,
                )?;
                let mut color = ctx.module.fresh_id();
                push_image_read_or_fetch(
                    ctx,
                    &mut out,
                    img,
                    fetch.coord,
                    Some(lod),
                    sample_v4,
                    color,
                );
                if let Some(in_bounds) = fetch.in_bounds {
                    let guarded = ctx.module.fresh_id();
                    let zero_color = const_null_of(ctx, sample_v4);
                    out.push(Instruction::new(
                        Op::Select,
                        Some(sample_v4),
                        Some(guarded),
                        vec![
                            Operand::IdRef(in_bounds),
                            Operand::IdRef(color),
                            Operand::IdRef(zero_color),
                        ],
                    ));
                    color = guarded;
                }
                return finish_sample_result(ctx, res, rty, color, sample_v4, out);
            }
        }
    }

    let img_ty = ctx.ty_image(dim, arrayed, comp);
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

    // Compose the sample coordinate to match the image dimension. AIR passes the spatial coord in
    // arg[2]; for *_array samples the layer index is the NEXT integer arg (arg[3]). The combined
    // coordinate has (#spatial-dims + arrayed) float components.
    let mut coord_for_sample = build_sample_coord(ctx, dim, arrayed, coord, args, &mut out)?;
    let explicit_lod = find_sample_lod(ctx, arrayed, args, &mut out);
    let spatial = sample_spatial_dims(dim);
    let (const_offset, dynamic_offset) = if let Some(spatial) = spatial {
        let (const_offset, dynamic_offset) =
            sample_const_or_dynamic_offset(ctx, arrayed, args, spatial as u32)?;
        let const_offset = const_offset.and_then(|offset| {
            if offset.iter().all(|delta| *delta == 0) {
                None
            } else {
                Some(const_sint_vec(ctx, &offset))
            }
        });
        (const_offset, dynamic_offset)
    } else {
        (None, None)
    };
    if let (Some(spatial), Some(offset)) = (spatial, dynamic_offset) {
        coord_for_sample = apply_dynamic_sample_offset(
            ctx,
            img,
            arrayed,
            coord_for_sample,
            offset,
            sample_uses_normalized_coords(ctx, arrayed, args),
            spatial,
            &mut out,
        )?;
    }
    push_image_sample(
        ctx,
        &mut out,
        sample_v4,
        color,
        si,
        coord_for_sample,
        explicit_lod,
        is_int_tex,
        const_offset,
    );

    finish_sample_result(ctx, res, rty, color, sample_v4, out)
}

/// Clamp a float pixel-space sample coordinate component to the finite range `[-9.0, size + 9.0]`
/// before it is floored / converted to an integer texel index. For finite coordinates this
/// preserves sampling behavior: sample offsets are bounded to [-8, 7] texels by the AIR contract,
/// so any value at or below -9 (or at or beyond size + 9) still resolves purely through the
/// address mode after the offset is applied, and the downstream per-tap clamp / in-bounds guard
/// yields the identical edge texel or border zero for the clamped value. Non-finite coordinates,
/// however, poison the integer conversion — `OpConvertFToS` of inf/NaN is undefined, and the
/// linear-filter fraction `inf - floor(inf)` is NaN, which no zero-weight guard can mask — while a
/// hardware sampler resolves them through the address mode like any other far-out-of-range
/// coordinate. `NClamp` maps a NaN coordinate to the low bound deterministically.
pub(in crate::passes) fn clamp_pixel_coord_component_finite(
    ctx: &mut Ctx,
    comp: Word,
    size: Word,
    size_is_vector: bool,
    axis: u32,
    out: &mut Vec<Instruction>,
) -> Word {
    let float_ty = ctx.ty_float();
    let uint = ctx.ty_uint();
    let glsl = ctx.glsl();
    let size_axis = if size_is_vector {
        let c = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::CompositeExtract,
            Some(uint),
            Some(c),
            vec![Operand::IdRef(size), Operand::LiteralBit32(axis)],
        ));
        c
    } else {
        size
    };
    let size_f = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ConvertUToF,
        Some(float_ty),
        Some(size_f),
        vec![Operand::IdRef(size_axis)],
    ));
    let slack = ctx.const_float(9.0);
    let hi = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::FAdd,
        Some(float_ty),
        Some(hi),
        vec![Operand::IdRef(size_f), Operand::IdRef(slack)],
    ));
    let lo = ctx.const_float(-9.0);
    let clamped = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ExtInst,
        Some(float_ty),
        Some(clamped),
        vec![
            Operand::IdRef(glsl),
            Operand::LiteralExtInstInteger(81), // NClamp
            Operand::IdRef(comp),
            Operand::IdRef(lo),
            Operand::IdRef(hi),
        ],
    ));
    clamped
}

#[allow(clippy::too_many_arguments)]
pub(in crate::passes) fn lower_pixel_linear_sample(
    ctx: &mut Ctx,
    sampler_state: AirStaticSamplerState,
    img: Word,
    dim: Dim,
    arrayed: bool,
    coord: Word,
    args: &[Word],
    lod: Word,
    sample_v4: Word,
    out: &mut Vec<Instruction>,
) -> Result<Word, String> {
    let spatial: usize = match dim {
        Dim::Dim2D => 2,
        Dim::Dim3D => 3,
        _ => return Err("air.sample_texture pixel linear sample unsupported dimension".into()),
    };
    let (offset, dynamic_offset) =
        sample_const_or_dynamic_offset(ctx, arrayed, args, spatial as u32)?;
    let dynamic_offset = dynamic_offset
        .map(|offset| dynamic_i32_integer_offset_components(ctx, offset, spatial, out))
        .transpose()?;
    let layer = if arrayed {
        Some(
            args.get(3)
                .copied()
                .ok_or("air.sample_texture array texture missing layer")?,
        )
    } else {
        None
    };
    let size = query_image_size(ctx, img, spatial, arrayed, lod, out);
    let float_ty = ctx.ty_float();
    let sint = ctx.ty_sint();
    let bool_ty = ctx.ty_bool();
    let glsl = ctx.glsl();
    let zero_f = ctx.const_float(0.0);
    let one_f = ctx.const_float(1.0);
    let half_f = ctx.const_float(0.5);
    let mut base = Vec::with_capacity(spatial);
    let mut frac = Vec::with_capacity(spatial);
    for (idx, comp) in sample_coord_components(ctx, coord, spatial as u32, out)?
        .into_iter()
        .enumerate()
    {
        let Operand::IdRef(comp) = comp else {
            return Err("air.sample_texture pixel coord component is not an id".into());
        };
        let comp = clamp_pixel_coord_component_finite(ctx, comp, size, true, idx as u32, out);
        let biased_coord = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FSub,
            Some(float_ty),
            Some(biased_coord),
            vec![Operand::IdRef(comp), Operand::IdRef(half_f)],
        ));
        let floor = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::ExtInst,
            Some(float_ty),
            Some(floor),
            vec![
                Operand::IdRef(glsl),
                Operand::LiteralExtInstInteger(8),
                Operand::IdRef(biased_coord),
            ],
        ));
        let mut base_i = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::ConvertFToS,
            Some(sint),
            Some(base_i),
            vec![Operand::IdRef(floor)],
        ));
        if let Some(offset) = &offset {
            let delta = offset[idx];
            if delta != 0 {
                let shifted = ctx.module.fresh_id();
                let delta = ctx.const_int_of(sint, delta as i64);
                out.push(Instruction::new(
                    Op::IAdd,
                    Some(sint),
                    Some(shifted),
                    vec![Operand::IdRef(base_i), Operand::IdRef(delta)],
                ));
                base_i = shifted;
            }
        } else if let Some(offset) = &dynamic_offset {
            let shifted = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::IAdd,
                Some(sint),
                Some(shifted),
                vec![Operand::IdRef(base_i), Operand::IdRef(offset[idx])],
            ));
            base_i = shifted;
        }
        let frac_i = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FSub,
            Some(float_ty),
            Some(frac_i),
            vec![Operand::IdRef(biased_coord), Operand::IdRef(floor)],
        ));
        base.push(base_i);
        frac.push(frac_i);
    }

    // Per-axis lerp weights: [1 - frac, frac] for each spatial axis. The tap loop walks the
    // 2^spatial corner taps (bilinear for 2D, trilinear for 3D).
    let mut axis_weights = Vec::with_capacity(spatial);
    for &frac_axis in frac.iter().take(spatial) {
        let one_minus = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FSub,
            Some(float_ty),
            Some(one_minus),
            vec![Operand::IdRef(one_f), Operand::IdRef(frac_axis)],
        ));
        axis_weights.push([one_minus, frac_axis]);
    }
    let zero_color = const_null_of(ctx, sample_v4);
    let mut acc = zero_color;
    for tap in 0..(1usize << spatial) {
        let tap_offset: Vec<i32> = (0..spatial)
            .map(|axis| ((tap >> axis) & 1) as i32)
            .collect();
        let tap_coord = pixel_linear_tap_coord(
            ctx,
            sampler_state,
            dim,
            arrayed,
            &base,
            &tap_offset,
            layer,
            size,
            out,
        )?;
        let fetched = ctx.module.fresh_id();
        push_image_read_or_fetch(
            ctx,
            out,
            img,
            tap_coord.coord,
            Some(lod),
            sample_v4,
            fetched,
        );
        let color = if let Some(in_bounds) = tap_coord.in_bounds {
            let guarded = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::Select,
                Some(sample_v4),
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
        };
        let mut weight = axis_weights[0][tap & 1];
        for (axis, weights) in axis_weights.iter().enumerate().skip(1) {
            let combined = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::FMul,
                Some(float_ty),
                Some(combined),
                vec![
                    Operand::IdRef(weight),
                    Operand::IdRef(weights[(tap >> axis) & 1]),
                ],
            ));
            weight = combined;
        }
        let weighted = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::VectorTimesScalar,
            Some(sample_v4),
            Some(weighted),
            vec![Operand::IdRef(color), Operand::IdRef(weight)],
        ));
        let weight_is_zero = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FOrdEqual,
            Some(bool_ty),
            Some(weight_is_zero),
            vec![Operand::IdRef(weight), Operand::IdRef(zero_f)],
        ));
        let contribution = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::Select,
            Some(sample_v4),
            Some(contribution),
            vec![
                Operand::IdRef(weight_is_zero),
                Operand::IdRef(zero_color),
                Operand::IdRef(weighted),
            ],
        ));
        let sum = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FAdd,
            Some(sample_v4),
            Some(sum),
            vec![Operand::IdRef(acc), Operand::IdRef(contribution)],
        ));
        acc = sum;
    }
    Ok(acc)
}

pub(in crate::passes) fn finish_sample_result(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    color: Word,
    color_ty: Word,
    mut out: Vec<Instruction>,
) -> Result<Vec<Instruction>, String> {
    // The AIR sample result shape varies by texture component type:
    //   * f32 textures: `air.sample_texture_*.v4f32` returns a `{ <4 x float>, i8 }` STRUCT — the body
    //     then `CompositeExtract`s member 0. We rebuild that struct from the v4float color + undef i8.
    //   * half textures: `air.sample_texture_2d.v4f16` (and `air.read_texture_2d.v4f16`) return a bare
    //     `<4 x half>` VECTOR — we `OpFConvert` the v4float sample down to v4half and yield it directly.
    // Classify `rty` (struct vs vector) and emit the matching tail so both validate.
    let rdef = type_def_of(ctx, rty);
    let is_struct = rdef
        .as_ref()
        .map(|d| d.class.opcode == Op::TypeStruct)
        .unwrap_or(false);
    if is_struct {
        // Determine the struct's member-0 (color) type; if it is a half vector, convert first.
        let member0 = rdef
            .as_ref()
            .and_then(|d| d.operands.first())
            .and_then(|o| match o {
                Operand::IdRef(m) => Some(*m),
                _ => None,
            });
        let color_for_struct = match member0 {
            Some(m) if m != color_ty && is_half_vector(ctx, m) => {
                let c = ctx.module.fresh_id();
                out.push(Instruction::new(
                    Op::FConvert,
                    Some(m),
                    Some(c),
                    vec![Operand::IdRef(color)],
                ));
                c
            }
            Some(m) => coerce_same_shape_integer(ctx, &mut out, color, color_ty, m)?,
            _ => color,
        };
        let i8u = ctx.ty_int8();
        let undef8 = ctx.module.fresh_id();
        out.push(Instruction::new(Op::Undef, Some(i8u), Some(undef8), vec![]));
        out.push(Instruction::new(
            Op::CompositeConstruct,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(color_for_struct), Operand::IdRef(undef8)],
        ));
    } else if rty != color_ty && is_half_vector(ctx, rty) {
        // Bare half-vector result: FConvert the v4float sample to v4half.
        out.push(Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(color)],
        ));
    } else if resolve::integer_shape(ctx, rty).is_some() {
        let c = coerce_same_shape_integer(ctx, &mut out, color, color_ty, rty)?;
        out.push(Instruction::new(
            Op::CopyObject,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(c)],
        ));
    } else {
        // Bare v4float result: copy the color through.
        out.push(Instruction::new(
            Op::CopyObject,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(color)],
        ));
    }
    Ok(out)
}
