//! Floating-point and imageblock AIR call lowering.

use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ImageblockConversion {
    None,
    Float,
    Bitcast,
}

pub(in crate::passes) fn lower_implicit_imageblock_load(
    ctx: &mut Ctx,
    name: &str,
    result: Word,
    result_ty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if args.len() != 4 {
        return Err(format!(
            "{name} expects attachment, coordinate, index, and data rate"
        ));
    }
    let (attachment, data_rate) = implicit_imageblock_constants(ctx, name, args)?;
    let (format, comp, storage_ty, conversion, lanes) =
        implicit_imageblock_format(ctx, name, result_ty)?;
    let (var, image_ty) = ctx.implicit_imageblock_var(attachment, data_rate, format, comp)?;
    let mut out = Vec::new();
    let image = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Load,
        Some(image_ty),
        Some(image),
        vec![Operand::IdRef(var)],
    ));
    ctx.image_dims.insert(image, (Dim::Dim2D, true));
    ctx.image_comp.insert(image, comp);
    ctx.image_storage.insert(image);
    let coord = build_fetch_coord(ctx, Dim::Dim2D, true, args[1], Some(args[2]), &mut out)?;
    let read_result = if conversion != ImageblockConversion::None || lanes != 4 {
        ctx.module.fresh_id()
    } else {
        result
    };
    out.push(Instruction::new(
        Op::ImageRead,
        Some(storage_ty),
        Some(read_result),
        vec![Operand::IdRef(image), Operand::IdRef(coord)],
    ));
    let converted_source = if lanes == 1 {
        let component = if conversion != ImageblockConversion::None {
            ctx.module.fresh_id()
        } else {
            result
        };
        let component_ty = match comp {
            ImageComp::Float => ctx.ty_float(),
            ImageComp::Uint => ctx.ty_uint(),
            ImageComp::Sint => ctx.ty_sint(),
        };
        out.push(Instruction::new(
            Op::CompositeExtract,
            Some(component_ty),
            Some(component),
            vec![Operand::IdRef(read_result), Operand::LiteralBit32(0)],
        ));
        component
    } else if lanes < 4 {
        let prefix = ctx.module.fresh_id();
        let prefix_ty = match comp {
            ImageComp::Float => ctx.ty_vecf(lanes),
            ImageComp::Uint => ctx.ty_vec_uint(lanes),
            ImageComp::Sint => ctx.ty_vec_sint(lanes),
        };
        out.push(Instruction::new(
            Op::VectorShuffle,
            Some(prefix_ty),
            Some(prefix),
            std::iter::once(Operand::IdRef(read_result))
                .chain(std::iter::once(Operand::IdRef(read_result)))
                .chain((0..lanes).map(Operand::LiteralBit32))
                .collect(),
        ));
        prefix
    } else {
        read_result
    };
    if conversion != ImageblockConversion::None {
        let opcode = match conversion {
            ImageblockConversion::Float => Op::FConvert,
            ImageblockConversion::Bitcast => Op::Bitcast,
            ImageblockConversion::None => unreachable!(),
        };
        out.push(Instruction::new(
            opcode,
            Some(result_ty),
            Some(result),
            vec![Operand::IdRef(converted_source)],
        ));
    }
    Ok(out)
}

pub(in crate::passes) fn lower_implicit_imageblock_store(
    ctx: &mut Ctx,
    name: &str,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if args.len() != 5 {
        return Err(format!(
            "{name} expects value, attachment, coordinate, index, and data rate"
        ));
    }
    let abi_args = &args[1..];
    let (attachment, data_rate) = implicit_imageblock_constants(ctx, name, abi_args)?;
    let value_ty = value_result_type(ctx, args[0])
        .ok_or_else(|| format!("{name} value has no result type"))?;
    let (format, comp, storage_ty, conversion, lanes) =
        implicit_imageblock_format(ctx, name, value_ty)?;
    let (var, image_ty) = ctx.implicit_imageblock_var(attachment, data_rate, format, comp)?;
    let mut out = Vec::new();
    let image = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Load,
        Some(image_ty),
        Some(image),
        vec![Operand::IdRef(var)],
    ));
    ctx.image_dims.insert(image, (Dim::Dim2D, true));
    ctx.image_comp.insert(image, comp);
    ctx.image_storage.insert(image);
    let coord = build_fetch_coord(
        ctx,
        Dim::Dim2D,
        true,
        abi_args[1],
        Some(abi_args[2]),
        &mut out,
    )?;
    let value = if conversion != ImageblockConversion::None {
        let storage_value_ty = if lanes == 1 {
            match conversion {
                ImageblockConversion::Float => ctx.ty_float(),
                ImageblockConversion::Bitcast => match comp {
                    ImageComp::Float => ctx.ty_float(),
                    ImageComp::Uint => ctx.ty_uint(),
                    ImageComp::Sint => ctx.ty_sint(),
                },
                ImageblockConversion::None => unreachable!(),
            }
        } else if lanes < 4 {
            match comp {
                ImageComp::Float => ctx.ty_vecf(lanes),
                ImageComp::Uint => ctx.ty_vec_uint(lanes),
                ImageComp::Sint => ctx.ty_vec_sint(lanes),
            }
        } else {
            storage_ty
        };
        let converted = ctx.module.fresh_id();
        let opcode = match conversion {
            ImageblockConversion::Float => Op::FConvert,
            ImageblockConversion::Bitcast => Op::Bitcast,
            ImageblockConversion::None => unreachable!(),
        };
        out.push(Instruction::new(
            opcode,
            Some(storage_value_ty),
            Some(converted),
            vec![Operand::IdRef(args[0])],
        ));
        converted
    } else {
        args[0]
    };
    let value = if lanes == 1 {
        let texel = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::CompositeConstruct,
            Some(storage_ty),
            Some(texel),
            vec![Operand::IdRef(value); 4],
        ));
        texel
    } else if lanes < 4 {
        let texel = ctx.module.fresh_id();
        let operands = std::iter::once(Operand::IdRef(value))
            .chain(std::iter::once(Operand::IdRef(value)))
            .chain(
                (0..4)
                    .map(|lane| Operand::LiteralBit32(if lane < lanes { lane } else { u32::MAX })),
            )
            .collect();
        out.push(Instruction::new(
            Op::VectorShuffle,
            Some(storage_ty),
            Some(texel),
            operands,
        ));
        texel
    } else {
        value
    };
    out.push(Instruction::new(
        Op::ImageWrite,
        None,
        None,
        vec![
            Operand::IdRef(image),
            Operand::IdRef(coord),
            Operand::IdRef(value),
        ],
    ));
    Ok(out)
}

fn implicit_imageblock_constants(
    ctx: &Ctx,
    name: &str,
    args: &[Word],
) -> Result<(u32, u32), String> {
    let attachment = constant_u32(ctx, args[0])
        .ok_or_else(|| format!("{name} requires a constant render-target attachment"))?;
    let data_rate = constant_u32(ctx, args[3])
        .ok_or_else(|| format!("{name} requires a constant imageblock data rate"))?;
    Ok((attachment, data_rate))
}

fn implicit_imageblock_format(
    ctx: &mut Ctx,
    name: &str,
    value_ty: Word,
) -> Result<(ImageFormat, ImageComp, Word, ImageblockConversion, u32), String> {
    use crate::meta::TextureFormat;

    let format = crate::meta::implicit_imageblock_texture_format(name)?
        .ok_or_else(|| format!("{name} is not an implicit imageblock intrinsic"))?;
    let lowered = match format {
        TextureFormat::R16f => {
            require_storage_image_extended_formats(ctx);
            (
                ImageFormat::R16f,
                ImageComp::Float,
                ctx.ty_vecf(4),
                ImageblockConversion::Float,
                1,
            )
        }
        TextureFormat::Rg16f => {
            require_storage_image_extended_formats(ctx);
            (
                ImageFormat::Rg16f,
                ImageComp::Float,
                ctx.ty_vecf(4),
                ImageblockConversion::Float,
                2,
            )
        }
        TextureFormat::Rgba16f => (
            ImageFormat::Rgba16f,
            ImageComp::Float,
            ctx.ty_vecf(4),
            ImageblockConversion::Float,
            4,
        ),
        TextureFormat::R32f => {
            require_storage_image_extended_formats(ctx);
            (
                ImageFormat::R32f,
                ImageComp::Float,
                ctx.ty_vecf(4),
                ImageblockConversion::None,
                1,
            )
        }
        TextureFormat::Rgba32f => (
            ImageFormat::Rgba32f,
            ImageComp::Float,
            value_ty,
            ImageblockConversion::None,
            4,
        ),
        TextureFormat::R32ui => {
            require_storage_image_extended_formats(ctx);
            (
                ImageFormat::R32ui,
                ImageComp::Uint,
                ctx.ty_vec_uint(4),
                ImageblockConversion::Bitcast,
                1,
            )
        }
        _ => return Err(format!("{name} has unsupported implicit imageblock format")),
    };
    Ok(lowered)
}

fn require_storage_image_extended_formats(ctx: &mut Ctx) {
    let capability = spirv::Capability::StorageImageExtendedFormats;
    if ctx
        .module
        .capabilities
        .iter()
        .any(|instruction| instruction.operands.as_slice() == [Operand::Capability(capability)])
    {
        return;
    }
    ctx.module.capabilities.push(Instruction::new(
        Op::Capability,
        None,
        None,
        vec![Operand::Capability(capability)],
    ));
}

/// Lower the residual floating-point / transcendental AIR intrinsic family and the generic
/// GLSL.std.450 ext-inst tail (sincos, cospi/sinpi/tanpi, exp10, fmod, log10, integer & float
/// min/max/clamp, min3/max3/fmedian3, bfloat fmuladd, ldexp, fast_tanh, the `glsl_extinst` map,
/// saturate). This is the terminal dispatch stage of `lower_one`: every call that reached here is
/// either handled or rejected with the "unhandled air.* intrinsic" error. Guard order is preserved.
pub(in crate::passes) fn lower_float_math(
    ctx: &mut Ctx,
    name: &str,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if (name.starts_with("air.fast_sincos.") || name.starts_with("air.sincos.")) && args.len() == 2
    {
        let ext = ctx.glsl();
        let cos = ctx.module.fresh_id();
        return Ok(vec![
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(res),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(GLSLstd450::Sin as u32),
                    Operand::IdRef(args[0]),
                ],
            ),
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(cos),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(GLSLstd450::Cos as u32),
                    Operand::IdRef(args[0]),
                ],
            ),
            Instruction::new(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(args[1]), Operand::IdRef(cos)],
            ),
        ]);
    }
    // `cospi/sinpi/tanpi(x) = {cos,sin,tan}(pi*x)` and `exp10(x) = exp2(x*log2(10))` have no direct
    // GLSL.std.450 op; build them as a constant pre-multiply followed by the base transcendental.
    // Scalar f16/f32 only (the observed signatures); a vector form FALLBACKs honestly.
    if (name.starts_with("air.cospi.") || name.starts_with("air.fast_cospi.")) && args.len() == 1 {
        return lower_premul_glsl_unary(
            ctx,
            res,
            rty,
            args[0],
            std::f32::consts::PI,
            GLSLstd450::Cos,
        );
    }
    if (name.starts_with("air.sinpi.") || name.starts_with("air.fast_sinpi.")) && args.len() == 1 {
        return lower_premul_glsl_unary(
            ctx,
            res,
            rty,
            args[0],
            std::f32::consts::PI,
            GLSLstd450::Sin,
        );
    }
    if (name.starts_with("air.tanpi.") || name.starts_with("air.fast_tanpi.")) && args.len() == 1 {
        return lower_premul_glsl_unary(
            ctx,
            res,
            rty,
            args[0],
            std::f32::consts::PI,
            GLSLstd450::Tan,
        );
    }
    if (name.starts_with("air.exp10.") || name.starts_with("air.fast_exp10.")) && args.len() == 1 {
        return lower_premul_glsl_unary(
            ctx,
            res,
            rty,
            args[0],
            std::f32::consts::LOG2_10,
            GLSLstd450::Exp2,
        );
    }
    // Metal's fmod follows C fmod semantics: x - y * trunc(x / y). GLSL.std.450 only exposes
    // Modf, and SPIR-V's OpFMod uses floor-style modulus semantics, so build the trunc form
    // explicitly. The same operations are valid for scalar and vector float types.
    if name.starts_with("air.fast_fmod.") || name.starts_with("air.fmod.") {
        if args.len() != 2 {
            return Err(format!("{name} expects two operands"));
        }
        let ext = ctx.glsl();
        let quotient = ctx.module.fresh_id();
        let truncated = ctx.module.fresh_id();
        let product = ctx.module.fresh_id();
        return Ok(vec![
            Instruction::new(
                Op::FDiv,
                Some(rty),
                Some(quotient),
                vec![Operand::IdRef(args[0]), Operand::IdRef(args[1])],
            ),
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(truncated),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(GLSLstd450::Trunc as u32),
                    Operand::IdRef(quotient),
                ],
            ),
            Instruction::new(
                Op::FMul,
                Some(rty),
                Some(product),
                vec![Operand::IdRef(args[1]), Operand::IdRef(truncated)],
            ),
            Instruction::new(
                Op::FSub,
                Some(rty),
                Some(res),
                vec![Operand::IdRef(args[0]), Operand::IdRef(product)],
            ),
        ]);
    }
    // GLSL.std.450 has natural Log but no Log10. Metal's log10(x) is log(x) / ln(10); build the
    // divisor with the same scalar/vector shape as the result type.
    if (name.starts_with("air.fast_log10.") || name.starts_with("air.log10.")) && args.len() == 1 {
        let ext = ctx.glsl();
        let logged = ctx.module.fresh_id();
        let n = vector_len(ctx, rty);
        let ln10 = splat_or_scalar(ctx, rty, std::f32::consts::LN_10, n);
        return Ok(vec![
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(logged),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(GLSLstd450::Log as u32),
                    Operand::IdRef(args[0]),
                ],
            ),
            Instruction::new(
                Op::FDiv,
                Some(rty),
                Some(res),
                vec![Operand::IdRef(logged), Operand::IdRef(ln10)],
            ),
        ]);
    }
    // Integer min/max use GLSL's integer ext-inst variants. The generic `air.min`/`air.max`
    // matcher below is float-only; using FMin/FMax for e.g. `air.min.s.i16` emits invalid
    // SPIR-V (`FMin` with a ushort result).
    if args.len() == 2 {
        let int_minmax = if name.starts_with("air.min.s.") {
            Some(GLSLstd450::SMin)
        } else if name.starts_with("air.min.u.") {
            Some(GLSLstd450::UMin)
        } else if name.starts_with("air.max.s.") {
            Some(GLSLstd450::SMax)
        } else if name.starts_with("air.max.u.") {
            Some(GLSLstd450::UMax)
        } else {
            None
        };
        if let Some(op) = int_minmax {
            if matches!(op, GLSLstd450::SMin | GLSLstd450::SMax) {
                return lower_signed_integer_minmax(ctx, res, rty, args, op);
            }
            let ext = ctx.glsl();
            return Ok(vec![Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(res),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(op as u32),
                    Operand::IdRef(args[0]),
                    Operand::IdRef(args[1]),
                ],
            )]);
        }
    }
    // Integer clamp uses GLSL's integer clamp variants. The generic `clamp` matcher below is
    // float-only; using FClamp for e.g. `air.clamp.u.i32` emits invalid SPIR-V.
    if args.len() == 3 {
        let int_clamp = if name.starts_with("air.clamp.s.") {
            Some(GLSLstd450::SClamp)
        } else if name.starts_with("air.clamp.u.") {
            Some(GLSLstd450::UClamp)
        } else {
            None
        };
        if let Some(op) = int_clamp {
            let ext = ctx.glsl();
            return Ok(vec![Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(res),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(op as u32),
                    Operand::IdRef(args[0]),
                    Operand::IdRef(args[1]),
                    Operand::IdRef(args[2]),
                ],
            )]);
        }
    }
    // AIR's `*min3`/`*max3` helpers are ternary. GLSL.std.450 min/max ops are binary, so
    // fold them into two ext-inst operations instead of forwarding all three operands to one
    // invalid OpExtInst.
    if args.len() == 3 {
        let ternary_minmax = if name.contains("fmax3") {
            Some(GLSLstd450::FMax)
        } else if name.contains("fmin3") {
            Some(GLSLstd450::FMin)
        } else if name.starts_with("air.max3.u.") {
            Some(GLSLstd450::UMax)
        } else if name.starts_with("air.min3.u.") {
            Some(GLSLstd450::UMin)
        } else if name.starts_with("air.max3.s.") {
            Some(GLSLstd450::SMax)
        } else if name.starts_with("air.min3.s.") {
            Some(GLSLstd450::SMin)
        } else {
            None
        };
        if let Some(op) = ternary_minmax {
            let ext = ctx.glsl();
            let tmp = ctx.module.fresh_id();
            return Ok(vec![
                Instruction::new(
                    Op::ExtInst,
                    Some(rty),
                    Some(tmp),
                    vec![
                        Operand::IdRef(ext),
                        Operand::LiteralExtInstInteger(op as u32),
                        Operand::IdRef(args[0]),
                        Operand::IdRef(args[1]),
                    ],
                ),
                Instruction::new(
                    Op::ExtInst,
                    Some(rty),
                    Some(res),
                    vec![
                        Operand::IdRef(ext),
                        Operand::LiteralExtInstInteger(op as u32),
                        Operand::IdRef(tmp),
                        Operand::IdRef(args[2]),
                    ],
                ),
            ]);
        }
    }
    if name.contains("fmedian3") && args.len() == 3 {
        let ext = ctx.glsl();
        let min_ab = ctx.module.fresh_id();
        let max_ab = ctx.module.fresh_id();
        let min_max_ab_c = ctx.module.fresh_id();
        return Ok(vec![
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(min_ab),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(GLSLstd450::FMin as u32),
                    Operand::IdRef(args[0]),
                    Operand::IdRef(args[1]),
                ],
            ),
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(max_ab),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(GLSLstd450::FMax as u32),
                    Operand::IdRef(args[0]),
                    Operand::IdRef(args[1]),
                ],
            ),
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(min_max_ab_c),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(GLSLstd450::FMin as u32),
                    Operand::IdRef(max_ab),
                    Operand::IdRef(args[2]),
                ],
            ),
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(res),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(GLSLstd450::FMax as u32),
                    Operand::IdRef(min_ab),
                    Operand::IdRef(min_max_ab_c),
                ],
            ),
        ]);
    }
    if is_llvm_bfloat_fmuladd(name) {
        return lower_bfloat_fmuladd(ctx, name, res, rty, args);
    }
    if is_air_math(name, "ldexp") && args.len() == 2 {
        return lower_fast_ldexp(ctx, name, res, rty, args);
    }
    // Metal's FAST tanh is exp2-based — tanh(x) = (t - 1)/(t + 1) with t = exp2(x * 2*log2(e)) —
    // so t overflows to +inf for large positive x and the quotient is inf/inf = NaN. Apple goldens
    // carry those NaNs (the Espresso batchnorm family), where GLSL Tanh saturates to 1.0. Emit the
    // faithful formula for the fast variant only; precise air.tanh stays GLSL Tanh below.
    if name.starts_with("air.fast_tanh.") && args.len() == 1 {
        return lower_fast_tanh(ctx, res, rty, args[0]);
    }
    if name.starts_with("air.fast_cos.") && args.len() == 1 && is_f32_scalar_or_vector(ctx, rty) {
        return Ok(lower_fast_trig(ctx, res, rty, args[0], GLSLstd450::Cos));
    }
    if name.starts_with("air.fast_sin.") && args.len() == 1 && is_f32_scalar_or_vector(ctx, rty) {
        return Ok(lower_fast_trig(ctx, res, rty, args[0], GLSLstd450::Sin));
    }
    // GLSL.std.450 ext-inst math.
    if let Some(glsl_op) = glsl_extinst(name) {
        if args.len() == 1 && matches!(glsl_op, GLSLstd450::Round | GLSLstd450::RoundEven) {
            return Ok(half_glsl_unary(ctx, glsl_op, res, rty, args[0]));
        }
        if args.len() == 2 && is_air_math(name, "pow") {
            // Metal's `pow(x, y)` widens to the sign-magnitude form `pow(|x|, y)` rather than the
            // IEEE `exp2(y*log2(x))` that yields NaN for x < 0: Apple goldens carry a finite
            // magnitude for a negative base (the value's sign is reapplied downstream by the shader,
            // e.g. a later multiply). GLSL Pow leaves x < 0 undefined (NaN on the Apple GPU via
            // MoltenVK), so emit the abs form. The half path widens to float around the pow; the f32
            // FAST variant emits the abs-pow directly. Non-negative base is unchanged (|x| == x), so
            // no currently-passing case regresses. `powr` (x >= 0 by contract) never reaches here.
            if is_half_scalar_or_vector(ctx, rty) {
                return Ok(half_abs_pow(ctx, res, rty, args[0], args[1]));
            }
            if name.starts_with("air.fast_pow.") && is_f32_scalar_or_vector(ctx, rty) {
                return Ok(f32_abs_pow(ctx, res, rty, args[0], args[1]));
            }
        }
        if args.len() == 3 && matches!(glsl_op, GLSLstd450::FMix) {
            return Ok(lower_endpoint_preserving_mix(
                ctx, res, rty, args[0], args[1], args[2],
            ));
        }
        let ext = ctx.glsl();
        let mut ops = vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(glsl_op as u32),
        ];
        for a in args {
            ops.push(Operand::IdRef(*a));
        }
        return Ok(vec![Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(res),
            ops,
        )]);
    }
    // saturate(x) = FClamp(x, 0, 1); needs synthesized 0/1 constants of the result ELEMENT type
    // (half `air.saturate.f16` -> half 0/1; float -> float 0/1; else spirv-val rejects the mismatch).
    if name.contains("saturate") && args.len() == 1 {
        let ext = ctx.glsl();
        let (zero, one) = scalar_zero_one(ctx, rty);
        // FClamp on a vector takes vector clamp operands; but GLSL FClamp accepts scalar edges only
        // for scalar x. For vectors we must splat — build constant composites.
        let (lo, hi) = clamp_edges(ctx, rty, zero, one);
        return Ok(vec![Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(res),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::FClamp as u32),
                Operand::IdRef(args[0]),
                Operand::IdRef(lo),
                Operand::IdRef(hi),
            ],
        )]);
    }

    Err(format!("unhandled air.* intrinsic: {name}"))
}

fn lower_endpoint_preserving_mix(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    x: Word,
    y: Word,
    t: Word,
) -> Vec<Instruction> {
    let ext = ctx.glsl();
    let lanes = vector_len(ctx, rty);
    let bool_ty = if lanes == 1 {
        ctx.ty_bool()
    } else {
        ctx.ty_vec_bool(lanes)
    };
    let (zero, one) = scalar_zero_one(ctx, rty);
    let (zero, one) = clamp_edges(ctx, rty, zero, one);
    let raw = ctx.module.fresh_id();
    let is_zero = ctx.module.fresh_id();
    let is_one = ctx.module.fresh_id();
    let zero_selected = ctx.module.fresh_id();
    vec![
        Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(raw),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::FMix as u32),
                Operand::IdRef(x),
                Operand::IdRef(y),
                Operand::IdRef(t),
            ],
        ),
        Instruction::new(
            Op::FOrdEqual,
            Some(bool_ty),
            Some(is_zero),
            vec![Operand::IdRef(t), Operand::IdRef(zero)],
        ),
        Instruction::new(
            Op::Select,
            Some(rty),
            Some(zero_selected),
            vec![
                Operand::IdRef(is_zero),
                Operand::IdRef(x),
                Operand::IdRef(raw),
            ],
        ),
        Instruction::new(
            Op::FOrdEqual,
            Some(bool_ty),
            Some(is_one),
            vec![Operand::IdRef(t), Operand::IdRef(one)],
        ),
        Instruction::new(
            Op::Select,
            Some(rty),
            Some(res),
            vec![
                Operand::IdRef(is_one),
                Operand::IdRef(y),
                Operand::IdRef(zero_selected),
            ],
        ),
    ]
}

fn lower_fast_trig(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    x: Word,
    op: GLSLstd450,
) -> Vec<Instruction> {
    let ext = ctx.glsl();
    let lanes = vector_len(ctx, rty);
    let bool_ty = if lanes == 1 {
        ctx.ty_bool()
    } else {
        ctx.ty_vec_bool(lanes)
    };
    let zero = ctx.const_float(0.0);
    let threshold = ctx.const_float(1_073_741_824.0);
    let two_pi = ctx.const_float(std::f32::consts::TAU);
    let (zero, threshold) = clamp_edges(ctx, rty, zero, threshold);
    let (_, two_pi) = clamp_edges(ctx, rty, zero, two_pi);
    let abs = ctx.module.fresh_id();
    let quotient = ctx.module.fresh_id();
    let periods = ctx.module.fresh_id();
    let scaled_periods = ctx.module.fresh_id();
    let reduced = ctx.module.fresh_id();
    let raw = ctx.module.fresh_id();
    let too_large = ctx.module.fresh_id();
    vec![
        Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(abs),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::FAbs as u32),
                Operand::IdRef(x),
            ],
        ),
        Instruction::new(
            Op::FDiv,
            Some(rty),
            Some(quotient),
            vec![Operand::IdRef(x), Operand::IdRef(two_pi)],
        ),
        Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(periods),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::Trunc as u32),
                Operand::IdRef(quotient),
            ],
        ),
        Instruction::new(
            Op::FMul,
            Some(rty),
            Some(scaled_periods),
            vec![Operand::IdRef(periods), Operand::IdRef(two_pi)],
        ),
        Instruction::new(
            Op::FSub,
            Some(rty),
            Some(reduced),
            vec![Operand::IdRef(x), Operand::IdRef(scaled_periods)],
        ),
        Instruction::new(
            Op::FOrdGreaterThanEqual,
            Some(bool_ty),
            Some(too_large),
            vec![Operand::IdRef(abs), Operand::IdRef(threshold)],
        ),
        Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(raw),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(op as u32),
                Operand::IdRef(reduced),
            ],
        ),
        Instruction::new(
            Op::Select,
            Some(rty),
            Some(res),
            vec![
                Operand::IdRef(too_large),
                Operand::IdRef(zero),
                Operand::IdRef(raw),
            ],
        ),
    ]
}

pub(in crate::passes) fn lower_signed_integer_minmax(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    args: &[Word],
    op: GLSLstd450,
) -> Result<Vec<Instruction>, String> {
    let signed_ty =
        signed_integer_type_like(ctx, rty).ok_or("air signed integer min/max result is not int")?;
    let mut out = Vec::new();
    let lhs = bitcast_integer_to_type(ctx, &mut out, args[0], signed_ty)?;
    let rhs = bitcast_integer_to_type(ctx, &mut out, args[1], signed_ty)?;
    let signed_res = if signed_ty == rty {
        res
    } else {
        ctx.module.fresh_id()
    };
    let ext = ctx.glsl();
    out.push(Instruction::new(
        Op::ExtInst,
        Some(signed_ty),
        Some(signed_res),
        vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(op as u32),
            Operand::IdRef(lhs),
            Operand::IdRef(rhs),
        ],
    ));
    if signed_res != res {
        out.push(Instruction::new(
            Op::Bitcast,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(signed_res)],
        ));
    }
    Ok(out)
}

pub(in crate::passes) fn bitcast_integer_to_type(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    value: Word,
    target_ty: Word,
) -> Result<Word, String> {
    let value_ty = value_result_type(ctx, value).ok_or("air signed min/max arg has no type")?;
    if value_ty == target_ty {
        return Ok(value);
    }
    let value_shape =
        integer_shape(ctx, value_ty).ok_or("air signed min/max arg is not integer-shaped")?;
    let target_shape =
        integer_shape(ctx, target_ty).ok_or("air signed min/max target is not integer-shaped")?;
    if value_shape != target_shape {
        return Err("air signed min/max arg shape does not match result".into());
    }
    let cast = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Bitcast,
        Some(target_ty),
        Some(cast),
        vec![Operand::IdRef(value)],
    ));
    Ok(cast)
}

pub(in crate::passes) fn signed_integer_type_like(ctx: &mut Ctx, ty: Word) -> Option<Word> {
    let def = type_def_of(ctx, ty)?;
    match def.class.opcode {
        Op::TypeInt => {
            let bits = match def.operands.first()? {
                Operand::LiteralBit32(bits) => *bits,
                _ => return None,
            };
            Some(signed_integer_scalar_type(ctx, bits))
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
            let signed_elem = signed_integer_type_like(ctx, elem)?;
            Some(integer_vector_type(ctx, signed_elem, lanes))
        }
        _ => None,
    }
}

pub(in crate::passes) fn signed_integer_scalar_type(ctx: &mut Ctx, bits: u32) -> Word {
    for inst in ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
    {
        if inst.class.opcode == Op::TypeInt
            && inst.operands.first() == Some(&Operand::LiteralBit32(bits))
            && inst.operands.get(1) == Some(&Operand::LiteralBit32(1))
        {
            if let Some(id) = inst.result_id {
                return id;
            }
        }
    }
    let id = ctx.module.fresh_id();
    ctx.new_globals.push(type_inst(
        Op::TypeInt,
        id,
        vec![Operand::LiteralBit32(bits), Operand::LiteralBit32(1)],
    ));
    id
}

pub(in crate::passes) fn integer_vector_type(ctx: &mut Ctx, elem: Word, lanes: u32) -> Word {
    for inst in ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
    {
        if inst.class.opcode == Op::TypeVector
            && inst.operands.first() == Some(&Operand::IdRef(elem))
            && inst.operands.get(1) == Some(&Operand::LiteralBit32(lanes))
        {
            if let Some(id) = inst.result_id {
                return id;
            }
        }
    }
    let id = ctx.module.fresh_id();
    ctx.new_globals.push(type_inst(
        Op::TypeVector,
        id,
        vec![Operand::IdRef(elem), Operand::LiteralBit32(lanes)],
    ));
    id
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

pub(in crate::passes) fn lower_imageblock_slice_write(
    ctx: &mut Ctx,
    name: &str,
    args: &[Word],
    v4: Word,
) -> Result<Vec<Instruction>, String> {
    if args.len() < 6 {
        return Err("air.write_imageblock_slice_to_texture missing operands".into());
    }
    let ptr_ty = value_result_type(ctx, args[1])
        .ok_or("air.write_imageblock_slice_to_texture pointer has no result type")?;
    let texel_ty = pointer_pointee_type(ctx, ptr_ty)
        .ok_or("air.write_imageblock_slice_to_texture pointer is not typed")?;
    let storage = ptr_storage(&type_defs(&ctx.module), ptr_ty).unwrap_or(StorageClass::Private);
    let write_texel_ty = imageblock_slice_texel_type(ctx, name, v4).unwrap_or(texel_ty);
    let texel_ptr = if write_texel_ty == texel_ty {
        args[1]
    } else {
        // Logical SPIR-V cannot reinterpret one pointer type as another. An imageblock cell may
        // carry several metadata-described fields while the AIR write intrinsic names the first
        // texel field directly (for example `{ half, half2 }` written through the `.f16` form).
        // Follow only an exact zero-offset aggregate path to that field; this is a real typed
        // subobject, unlike the former pointer OpBitcast. If the AIR type has no such field, keep
        // the layout gap visible instead of inventing a byte-level reinterpretation.
        let path = imageblock_zero_offset_subobject_path(ctx, texel_ty, write_texel_ty)
            .ok_or_else(|| {
                format!(
                    "air.write_imageblock_slice_to_texture cannot view pointee type %{texel_ty} \
                     as texel type %{write_texel_ty} through a zero-offset aggregate field"
                )
            })?;
        let retyped_ptr_ty = ctx.ty_ptr(storage, write_texel_ty);
        let retyped = ctx.module.fresh_id();
        let mut operands = Vec::with_capacity(path.len() + 1);
        operands.push(Operand::IdRef(args[1]));
        operands.extend(
            path.into_iter()
                .map(|index| Operand::IdRef(ctx.const_uint(index))),
        );
        let mut out = vec![Instruction::new(
            Op::InBoundsAccessChain,
            Some(retyped_ptr_ty),
            Some(retyped),
            operands,
        )];
        let texel = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::Load,
            Some(write_texel_ty),
            Some(texel),
            vec![Operand::IdRef(retyped)],
        ));
        return lower_imageblock_slice_write_texel(ctx, args, v4, texel, write_texel_ty, out);
    };
    let texel = ctx.module.fresh_id();
    let out = vec![Instruction::new(
        Op::Load,
        Some(write_texel_ty),
        Some(texel),
        vec![Operand::IdRef(texel_ptr)],
    )];
    lower_imageblock_slice_write_texel(ctx, args, v4, texel, write_texel_ty, out)
}

/// Return the exact zero-offset aggregate-member path from `source` to `target`.
///
/// Imageblock write intrinsics encode their texel type in the stable AIR ABI symbol, while the
/// pointer itself can still name the complete metadata-described cell. The only legal Logical
/// SPIR-V narrowing is an access chain to a real contained subobject. Struct member zero and array
/// element zero share the parent's byte address, so recursively following their first child is
/// sufficient and deliberately does not model arbitrary same-size reinterprets.
pub(in crate::passes) fn imageblock_zero_offset_subobject_path(
    ctx: &Ctx,
    source: Word,
    target: Word,
) -> Option<Vec<u32>> {
    if source == target {
        return Some(Vec::new());
    }
    let source_def = type_def_of(ctx, source)?;
    let first_child = match source_def.class.opcode {
        Op::TypeStruct | Op::TypeArray => match source_def.operands.first() {
            Some(Operand::IdRef(child)) => *child,
            _ => return None,
        },
        _ => return None,
    };
    let mut path = imageblock_zero_offset_subobject_path(ctx, first_child, target)?;
    path.insert(0, 0);
    Some(path)
}

pub(in crate::passes) fn lower_imageblock_slice_write_texel(
    ctx: &mut Ctx,
    args: &[Word],
    v4: Word,
    texel: Word,
    texel_ty: Word,
    mut out: Vec<Instruction>,
) -> Result<Vec<Instruction>, String> {
    let mut img = resolve_image_value(ctx, args[0]);
    if !image_is_storage(ctx, img) {
        img = single_storage_image_for_private_write(ctx, img).ok_or_else(|| {
            format!("air.write_imageblock_slice_to_texture on non-storage image id {img}")
        })?;
    }
    ctx.require_runtime_storage_image_use(img, RuntimeStorageImageUse::Write)?;
    let (dim, arrayed) = ctx
        .image_dims
        .get(&img)
        .copied()
        .unwrap_or((Dim::Dim2D, false));
    // The imageblock-slice ABI puts the destination coordinate at operand 5 and, for an array
    // texture, its slice at operand 6.
    let layer = if arrayed {
        Some(
            args.get(6)
                .copied()
                .ok_or("air.write_imageblock_slice_to_texture array texture missing layer")?,
        )
    } else {
        None
    };
    let coord32 = build_fetch_coord(ctx, dim, arrayed, args[5], layer, &mut out)?;
    let region_gate =
        gate_imageblock_region_in_bounds(ctx, args, img, dim, arrayed, coord32, &mut out)?;
    let comp = ctx
        .image_comp
        .get(&img)
        .copied()
        .unwrap_or(ImageComp::Float);
    // A scalar texel (single-channel imageblock write, e.g. `air.write_imageblock_slice_to_texture_2d
    // .f16`) has no vector shape; treat it as 1 lane. OpImageWrite always takes a 4-component texel,
    // so a sub-v4 texel is padded to v4 with zeros (the extra channels are ignored by single/2/3-
    // channel storage formats).
    let (elem, lanes) = vector_type_shape(ctx, texel_ty).unwrap_or((texel_ty, 1));
    let texel32 = if comp != ImageComp::Float {
        // Integer (Sint/Uint) imageblock write, e.g. `air.write_imageblock_slice_to_texture_2d
        // .i16.v4i16` into an Rgba16Sint/Uint storage image: build a v4 of the image's 32-bit
        // integer sampled type (`ty_sint`/`ty_uint`), sign/zero-extending the narrower channels
        // per the format's signedness. OpImageWrite's texel type must match the storage image's
        // integer sampled type, exactly as the float arm matches its float sampled type.
        build_int_write_texel(ctx, comp, texel, elem, lanes, &mut out)?
    } else if lanes == 4 {
        if is_f32_scalar(ctx, elem) {
            texel
        } else if is_half_scalar(ctx, elem) {
            let converted = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::FConvert,
                Some(v4),
                Some(converted),
                vec![Operand::IdRef(texel)],
            ));
            converted
        } else {
            return Err(
                "air.write_imageblock_slice_to_texture: unsupported texel component".into(),
            );
        }
    } else if lanes == 1 {
        // Convert the scalar channel to f32, then build a v4 (channel 0 = value, 1..3 = 0).
        let f32_ty = ctx.ty_float();
        let scalar32 = if is_f32_scalar(ctx, elem) {
            texel
        } else if is_half_scalar(ctx, elem) {
            let converted = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::FConvert,
                Some(f32_ty),
                Some(converted),
                vec![Operand::IdRef(texel)],
            ));
            converted
        } else {
            return Err(
                "air.write_imageblock_slice_to_texture: unsupported texel component".into(),
            );
        };
        let zero = ctx.const_float(0.0);
        let padded = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::CompositeConstruct,
            Some(v4),
            Some(padded),
            vec![
                Operand::IdRef(scalar32),
                Operand::IdRef(zero),
                Operand::IdRef(zero),
                Operand::IdRef(zero),
            ],
        ));
        padded
    } else if lanes == 2 || lanes == 3 {
        // A 2- or 3-channel texel (e.g. `.v2f16`): convert to a float vector of the same lane count,
        // extract its channels, and build a v4 padding the unused channels with 0 (OpImageWrite always
        // takes a 4-component texel; the storage format ignores the extra channels).
        let f32_ty = ctx.ty_float();
        let src32 = if is_f32_scalar(ctx, elem) {
            texel
        } else if is_half_scalar(ctx, elem) {
            let fvec = ctx.ty_vecf(lanes);
            let converted = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::FConvert,
                Some(fvec),
                Some(converted),
                vec![Operand::IdRef(texel)],
            ));
            converted
        } else {
            return Err(
                "air.write_imageblock_slice_to_texture: unsupported texel component".into(),
            );
        };
        let zero = ctx.const_float(0.0);
        let mut comps = Vec::with_capacity(4);
        for i in 0..lanes {
            let c = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::CompositeExtract,
                Some(f32_ty),
                Some(c),
                vec![Operand::IdRef(src32), Operand::LiteralBit32(i)],
            ));
            comps.push(Operand::IdRef(c));
        }
        for _ in lanes..4 {
            comps.push(Operand::IdRef(zero));
        }
        let padded = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::CompositeConstruct,
            Some(v4),
            Some(padded),
            comps,
        ));
        padded
    } else {
        return Err("air.write_imageblock_slice_to_texture: texel is not v4".into());
    };
    let texel32_ty = match comp {
        ImageComp::Float => v4,
        ImageComp::Sint => ctx.ty_vec_sint(4),
        ImageComp::Uint => ctx.ty_vec_uint(4),
    };
    let texel32 = zero_texel_for_empty_imageblock_region(
        ctx,
        texel32,
        texel32_ty,
        region_gate.empty,
        &mut out,
    )?;
    out.push(Instruction::new(
        Op::ImageWrite,
        None,
        None,
        vec![
            Operand::IdRef(img),
            Operand::IdRef(region_gate.coord),
            Operand::IdRef(texel32),
        ],
    ));
    Ok(out)
}

fn zero_texel_for_empty_imageblock_region(
    ctx: &mut Ctx,
    texel: Word,
    texel_ty: Word,
    region_empty: Word,
    out: &mut Vec<Instruction>,
) -> Result<Word, String> {
    let zero_texel = ctx.get_or_create(Op::ConstantNull, Some(texel_ty), vec![]);
    let selected = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Select,
        Some(texel_ty),
        Some(selected),
        vec![
            Operand::IdRef(region_empty),
            Operand::IdRef(zero_texel),
            Operand::IdRef(texel),
        ],
    ));
    Ok(selected)
}

/// Build the `OpImageWrite` texel for a non-float (Sint/Uint) imageblock slice write: the storage
/// image's integer sampled type is 32-bit (`ty_sint`/`ty_uint`), so a narrower channel texel (the
/// common `.v4i16` form) is widened to a v4 of that 32-bit int, sign-extended for Sint and
/// zero-extended for Uint. Only the v4 shape is handled (both regression cases are `.v4i16`); a sub-v4
/// integer texel is left as an honest Err so a mis-lowering never ships instead of failing loudly.
pub(in crate::passes) fn build_int_write_texel(
    ctx: &mut Ctx,
    comp: ImageComp,
    texel: Word,
    elem: Word,
    lanes: u32,
    out: &mut Vec<Instruction>,
) -> Result<Word, String> {
    if lanes != 4 {
        return Err(
            "air.write_imageblock_slice_to_texture: non-float imageblock write with non-v4 texel"
                .into(),
        );
    }
    let (v4int_ty, signed) = match comp {
        ImageComp::Sint => (ctx.ty_vec_sint(4), true),
        ImageComp::Uint => (ctx.ty_vec_uint(4), false),
        ImageComp::Float => {
            return Err("build_int_write_texel called for a float image".into());
        }
    };
    // Already a 32-bit int of the target signedness → OpImageWrite accepts it directly (a same-width
    // SConvert/UConvert would be spirv-val-invalid, so this guard is required, not just an optimization).
    let already32 = (signed && is_int_scalar_width(ctx, elem, 32))
        || (!signed && is_uint_scalar_width(ctx, elem, 32));
    if already32 {
        return Ok(texel);
    }
    let op = if signed { Op::SConvert } else { Op::UConvert };
    let converted = ctx.module.fresh_id();
    out.push(Instruction::new(
        op,
        Some(v4int_ty),
        Some(converted),
        vec![Operand::IdRef(texel)],
    ));
    Ok(converted)
}

struct ImageblockRegionGate {
    coord: Word,
    empty: Word,
}

/// Gate an `air.write_imageblock_slice_to_texture_*` store on its destination region when the call
/// uses the implicit imageblock extent, and report whether the region has a zero spatial extent. The
/// implicit extent is the imageblock dimensions, which for a compute kernel are the threadgroup x/y
/// dimensions. The Apple GPU discards the whole implicit-region write when that region extends past
/// the texture bounds, so OR the write coordinate with all-ones in that case and let Vulkan's OOB
/// store rule drop the single `OpImageWrite` this lowering emits. Explicit zero-area regions keep
/// the destination coordinate and write a transparent-zero texel; explicit non-empty regions that
/// extend past the texture bounds are discarded by the same out-of-bounds coordinate gate.
fn gate_imageblock_region_in_bounds(
    ctx: &mut Ctx,
    args: &[Word],
    img: Word,
    dim: Dim,
    arrayed: bool,
    coord32: Word,
    out: &mut Vec<Instruction>,
) -> Result<ImageblockRegionGate, String> {
    let spatial: u32 = match dim {
        Dim::Dim1D | Dim::DimBuffer => 1,
        Dim::Dim3D => 3,
        _ => 2,
    };
    let ncomp = spatial + u32::from(arrayed);
    let uint = ctx.ty_uint();
    let bool_ty = ctx.ty_bool();
    // Texture dimensions. The image is a storage image (checked by the caller), so the query has
    // no LOD operand; the bound storage view is single-mip.
    let size_ty = if ncomp == 1 {
        uint
    } else {
        ctx.ty_vec_uint(ncomp)
    };
    let dims = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ImageQuerySize,
        Some(size_ty),
        Some(dims),
        vec![Operand::IdRef(img)],
    ));
    let extract = |ctx: &mut Ctx, out: &mut Vec<Instruction>, composite: Word, index: u32| {
        let id = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::CompositeExtract,
            Some(uint),
            Some(id),
            vec![Operand::IdRef(composite), Operand::LiteralBit32(index)],
        ));
        id
    };
    // Block region size per axis: the explicit size operand when the has-size flag is set, else the
    // imageblock (= threadgroup) dimensions. A non-constant flag selects at runtime.
    let has_size_flag = value_def_instruction(ctx, args[2]).map(|def| def.class.opcode);
    let explicit_size_def = value_def_instruction(ctx, args[4]).map(|def| def.class.opcode);
    let explicit_usable = !matches!(explicit_size_def, Some(Op::Undef) | None);
    let implicit = [
        ctx.const_uint(ctx.kernel_local_size[0]),
        ctx.const_uint(ctx.kernel_local_size[1]),
    ];
    let explicit = if explicit_usable && !matches!(has_size_flag, Some(Op::ConstantFalse)) {
        // args[4] is a `<2 x i16>` size; widen to uint2 and split into components.
        let src_ty = value_result_type(ctx, args[4])
            .ok_or("air.write_imageblock_slice_to_texture: size operand has no type")?;
        let wide = if scalar_bit_width(ctx, src_ty) == 32 {
            args[4]
        } else {
            let uint2 = ctx.ty_vec_uint(2);
            let id = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::UConvert,
                Some(uint2),
                Some(id),
                vec![Operand::IdRef(args[4])],
            ));
            id
        };
        Some([extract(ctx, out, wide, 0), extract(ctx, out, wide, 1)])
    } else {
        None
    };
    let mut region = implicit;
    if let Some(explicit) = explicit {
        match has_size_flag {
            Some(Op::ConstantTrue) => region = explicit,
            Some(Op::ConstantFalse) => {}
            _ => {
                // Runtime flag: select per axis.
                for axis in 0..2 {
                    let id = ctx.module.fresh_id();
                    out.push(Instruction::new(
                        Op::Select,
                        Some(uint),
                        Some(id),
                        vec![
                            Operand::IdRef(args[2]),
                            Operand::IdRef(explicit[axis]),
                            Operand::IdRef(implicit[axis]),
                        ],
                    ));
                    region[axis] = id;
                }
            }
        }
    }
    // fits = origin + region_size <= dims, per spatial axis (x, and y for 2D/3D images). The origin
    // comes from the already-widened write coordinate; both operands are <= 0xffff so the u32 add
    // cannot wrap. In parallel, track whether any spatial size component is zero so the caller can
    // model Apple's transparent-zero texel for zero-area explicit writes.
    let zero = ctx.const_uint(0);
    let mut fits: Option<Word> = None;
    let mut empty: Option<Word> = None;
    for axis in 0..spatial.min(2) {
        let origin = if ncomp == 1 {
            coord32
        } else {
            extract(ctx, out, coord32, axis)
        };
        let dim_axis = if ncomp == 1 {
            dims
        } else {
            extract(ctx, out, dims, axis)
        };
        let end = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::IAdd,
            Some(uint),
            Some(end),
            vec![
                Operand::IdRef(origin),
                Operand::IdRef(region[axis as usize]),
            ],
        ));
        let axis_empty = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::IEqual,
            Some(bool_ty),
            Some(axis_empty),
            vec![Operand::IdRef(region[axis as usize]), Operand::IdRef(zero)],
        ));
        let axis_fits = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::ULessThanEqual,
            Some(bool_ty),
            Some(axis_fits),
            vec![Operand::IdRef(end), Operand::IdRef(dim_axis)],
        ));
        fits = Some(match fits {
            None => axis_fits,
            Some(prev) => {
                let both = ctx.module.fresh_id();
                out.push(Instruction::new(
                    Op::LogicalAnd,
                    Some(bool_ty),
                    Some(both),
                    vec![Operand::IdRef(prev), Operand::IdRef(axis_fits)],
                ));
                both
            }
        });
        empty = Some(match empty {
            None => axis_empty,
            Some(prev) => {
                let either = ctx.module.fresh_id();
                out.push(Instruction::new(
                    Op::LogicalOr,
                    Some(bool_ty),
                    Some(either),
                    vec![Operand::IdRef(prev), Operand::IdRef(axis_empty)],
                ));
                either
            }
        });
    }
    let fits = fits.ok_or("gate_imageblock_region_in_bounds: at least one spatial axis")?;
    let empty = empty.ok_or("gate_imageblock_region_in_bounds: at least one spatial axis")?;
    // mask = fits ? 0 : 0xffffffff; OR-ing it into the coordinate forces the store out of bounds
    // (and thus discarded) exactly when the implicit/runtime-selected block region does not fit.
    let all_ones = ctx.const_uint(u32::MAX);
    let mask = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Select,
        Some(uint),
        Some(mask),
        vec![
            Operand::IdRef(fits),
            Operand::IdRef(zero),
            Operand::IdRef(all_ones),
        ],
    ));
    let (mask_vec, coord_ty) = if ncomp == 1 {
        (mask, uint)
    } else {
        let vec_ty = ctx.ty_vec_uint(ncomp);
        let id = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::CompositeConstruct,
            Some(vec_ty),
            Some(id),
            vec![Operand::IdRef(mask); ncomp as usize],
        ));
        (id, vec_ty)
    };
    let gated = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::BitwiseOr,
        Some(coord_ty),
        Some(gated),
        vec![Operand::IdRef(coord32), Operand::IdRef(mask_vec)],
    ));
    Ok(ImageblockRegionGate {
        coord: gated,
        empty,
    })
}

pub(in crate::passes) fn imageblock_slice_texel_type(
    ctx: &mut Ctx,
    name: &str,
    v4: Word,
) -> Option<Word> {
    // The texel component type is the final dotted suffix of the intrinsic name, e.g.
    // `air.write_imageblock_slice_to_texture_2d.i16.v2f16` -> `v2f16`. `vNfM` is an M-bit-float
    // N-vector (N in 2..=4); a bare `fM` is a scalar (1 lane). The half/float scalar+vector forms are
    // all handled by `lower_imageblock_slice_write_texel`; the name is authoritative for the texel
    // type so the field pointer is reinterpreted to it before the load.
    let suffix = name.rsplit('.').next()?;
    let (lanes, elem) = match suffix.strip_prefix('v') {
        Some(rest) => {
            let split = rest.find('f')?;
            (rest[..split].parse::<u32>().ok()?, &rest[split..])
        }
        None => (1, suffix),
    };
    if !(1..=4).contains(&lanes) {
        return None;
    }
    let half = match elem {
        "f16" => true,
        "f32" => false,
        _ => return None,
    };
    Some(match (half, lanes) {
        (true, 1) => ctx.ty_half(),
        (true, n) => ctx.ty_vech(n),
        (false, 1) => ctx.ty_float(),
        (false, 4) => v4,
        (false, n) => ctx.ty_vecf(n),
    })
}

pub(in crate::passes) fn pointer_pointee_type(ctx: &Ctx, ptr_ty: Word) -> Option<Word> {
    let def = type_def_of(ctx, ptr_ty)?;
    if def.class.opcode != Op::TypePointer {
        return None;
    }
    match def.operands.get(1) {
        Some(Operand::IdRef(pointee)) => Some(*pointee),
        _ => None,
    }
}
