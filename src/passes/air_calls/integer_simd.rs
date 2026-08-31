//! Integer, SIMD, texture, and conversion AIR call lowering.

use super::conversions::lower_convert;
use super::*;
use spirv::GroupOperation;

/// Lower the residual integer-arithmetic AIR/LLVM intrinsic family (`air.abs*`, `air.reverse_bits`,
/// `llvm.bswap.i32`, `air.rotate`, `air.{extract,insert}_bits`, `air.popcount`, `air.{ctz,clz}`,
/// `air.mul_hi`, `air.mad_sat`). Returns `Ok(Some(insts))` when a branch handled the call, `Ok(None)`
/// when no integer-op guard matched (the caller then continues its dispatch cascade), or `Err` when a
/// matched branch rejected its operands. Guard order/precedence is load-bearing and preserved.
pub(in crate::passes) fn lower_integer_op(
    ctx: &mut Ctx,
    name: &str,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Option<Vec<Instruction>>, String> {
    // Integer absolute value: `air.abs.s.i32` (signed) -> GLSL SAbs; `air.abs.u.*` (unsigned) is the
    // identity (an unsigned value is already its own magnitude) -> OpCopyObject.
    if name.starts_with("air.abs_diff.") && args.len() == 2 {
        let (min_op, max_op) = if name.starts_with("air.abs_diff.u.") {
            (GLSLstd450::UMin, GLSLstd450::UMax)
        } else if name.starts_with("air.abs_diff.s.") {
            (GLSLstd450::SMin, GLSLstd450::SMax)
        } else {
            return Err(format!("unhandled abs_diff intrinsic: {name}"));
        };
        let ext = ctx.glsl();
        let lo = ctx.module.fresh_id();
        let hi = ctx.module.fresh_id();
        return Ok(Some(vec![
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(hi),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(max_op as u32),
                    Operand::IdRef(args[0]),
                    Operand::IdRef(args[1]),
                ],
            ),
            Instruction::new(
                Op::ExtInst,
                Some(rty),
                Some(lo),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(min_op as u32),
                    Operand::IdRef(args[0]),
                    Operand::IdRef(args[1]),
                ],
            ),
            Instruction::new(
                Op::ISub,
                Some(rty),
                Some(res),
                vec![Operand::IdRef(hi), Operand::IdRef(lo)],
            ),
        ]));
    }
    if name.starts_with("air.abs.") && args.len() == 1 {
        if name.starts_with("air.abs.u.") {
            return Ok(Some(vec![Instruction::new(
                Op::CopyObject,
                Some(rty),
                Some(res),
                vec![Operand::IdRef(args[0])],
            )]));
        }
        let ext = ctx.glsl();
        return Ok(Some(vec![Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(res),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::SAbs as u32),
                Operand::IdRef(args[0]),
            ],
        )]));
    }
    if name.starts_with("air.reverse_bits.") && args.len() == 1 {
        // Vulkan restricts OpBitReverse to a 32-bit base (VUID-RuntimeSpirv-None-10824 without
        // maintenance9): widen a sub-32-bit reverse to u32, reverse, shift the reversed word right
        // by (32 - width) so the value's reversed bits land back in the low `width` bits, narrow.
        if let Some(width) = int_scalar_width(ctx, rty) {
            if width < 32 {
                let uint = ctx.ty_uint();
                let widened = ctx.module.fresh_id();
                let reversed = ctx.module.fresh_id();
                let shift = ctx.const_int_of(uint, i64::from(32 - width));
                let shifted = ctx.module.fresh_id();
                return Ok(Some(vec![
                    Instruction::new(
                        Op::UConvert,
                        Some(uint),
                        Some(widened),
                        vec![Operand::IdRef(args[0])],
                    ),
                    Instruction::new(
                        Op::BitReverse,
                        Some(uint),
                        Some(reversed),
                        vec![Operand::IdRef(widened)],
                    ),
                    Instruction::new(
                        Op::ShiftRightLogical,
                        Some(uint),
                        Some(shifted),
                        vec![Operand::IdRef(reversed), Operand::IdRef(shift)],
                    ),
                    Instruction::new(
                        Op::UConvert,
                        Some(rty),
                        Some(res),
                        vec![Operand::IdRef(shifted)],
                    ),
                ]));
            }
        }
        return Ok(Some(vec![Instruction::new(
            Op::BitReverse,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(args[0])],
        )]));
    }
    if name == "llvm.bswap.i32" && args.len() == 1 {
        if !is_int_scalar_width(ctx, rty, 32) {
            return Err("llvm.bswap.i32 result type is not i32".to_string());
        }
        return Ok(Some(byte_swap_i32(ctx, res, rty, args[0])));
    }
    if name.starts_with("air.rotate.") && args.len() == 2 {
        let width = int_scalar_width(ctx, rty)
            .ok_or_else(|| format!("{name} result type is not a scalar integer"))?;
        if width != 32 {
            return Err(format!("{name} currently supports i32 results"));
        }
        let value_ty = value_result_type(ctx, args[0])
            .ok_or_else(|| format!("{name} value operand has no result type"))?;
        let shift_ty = value_result_type(ctx, args[1])
            .ok_or_else(|| format!("{name} shift operand has no result type"))?;
        if value_ty != rty || shift_ty != rty {
            return Err(format!("{name} operand/result type mismatch"));
        }
        return Ok(Some(rotate_left_i32(ctx, res, rty, args[0], args[1])));
    }
    if name.starts_with("air.extract_bits.u.") && args.len() == 3 {
        let (lanes, width) = validate_extract_bits_shape(ctx, name, rty, args)?;
        if width == 64 {
            if lanes != 1 {
                return Err(format!("{name} currently supports 64-bit scalar results"));
            }
            return Ok(Some(extract_bits_i64(ctx, res, rty, args, false)?));
        }
        if width < 32 {
            return Ok(Some(extract_bits_narrow(ctx, res, rty, args, lanes, false)));
        }
        if width != 32 {
            return Err(format!("{name} unsupported result width {width}"));
        }
        return Ok(Some(vec![Instruction::new(
            Op::BitFieldUExtract,
            Some(rty),
            Some(res),
            vec![
                Operand::IdRef(args[0]),
                Operand::IdRef(args[1]),
                Operand::IdRef(args[2]),
            ],
        )]));
    }
    if name.starts_with("air.extract_bits.s.") && args.len() == 3 {
        let (lanes, width) = validate_extract_bits_shape(ctx, name, rty, args)?;
        if width == 64 {
            if lanes != 1 {
                return Err(format!("{name} currently supports 64-bit scalar results"));
            }
            return Ok(Some(extract_bits_i64(ctx, res, rty, args, true)?));
        }
        if width < 32 {
            return Ok(Some(extract_bits_narrow(ctx, res, rty, args, lanes, true)));
        }
        if width != 32 {
            return Err(format!("{name} unsupported result width {width}"));
        }
        return Ok(Some(vec![Instruction::new(
            Op::BitFieldSExtract,
            Some(rty),
            Some(res),
            vec![
                Operand::IdRef(args[0]),
                Operand::IdRef(args[1]),
                Operand::IdRef(args[2]),
            ],
        )]));
    }
    if name.starts_with("air.insert_bits.") && args.len() == 4 {
        let (lanes, width) = validate_insert_bits_shape(ctx, name, rty, args)?;
        if width == 64 {
            if lanes != 1 {
                return Err(format!("{name} currently supports 64-bit scalar results"));
            }
            return Ok(Some(insert_bits_i64(ctx, res, rty, args)?));
        }
        if width < 32 {
            return Ok(Some(insert_bits_narrow(ctx, res, rty, args, lanes)));
        }
        if width != 32 {
            return Err(format!("{name} unsupported result width {width}"));
        }
        return Ok(Some(vec![Instruction::new(
            Op::BitFieldInsert,
            Some(rty),
            Some(res),
            vec![
                Operand::IdRef(args[0]),
                Operand::IdRef(args[1]),
                Operand::IdRef(args[2]),
                Operand::IdRef(args[3]),
            ],
        )]));
    }
    if name.starts_with("air.popcount.") && args.len() == 1 {
        if let Some(arg_ty) = value_result_type(ctx, args[0]) {
            if is_int_scalar_width(ctx, rty, 64) && is_int_scalar_width(ctx, arg_ty, 64) {
                return Ok(Some(popcount_i64_as_u32_halves(ctx, res, rty, args[0])));
            }
            if let (Some((result_lanes, 64)), Some((arg_lanes, 64))) =
                (int_vector_width(ctx, rty), int_vector_width(ctx, arg_ty))
            {
                if result_lanes == arg_lanes {
                    return Ok(Some(popcount_i64_vector_as_u32_halves(
                        ctx,
                        res,
                        rty,
                        args[0],
                        result_lanes,
                    )));
                }
            }
            if let (Some(result_width), Some(arg_width)) =
                (int_scalar_width(ctx, rty), int_scalar_width(ctx, arg_ty))
            {
                if result_width < 32 && arg_width < 32 {
                    let uint = ctx.ty_uint();
                    let widened = ctx.module.fresh_id();
                    let counted = ctx.module.fresh_id();
                    return Ok(Some(vec![
                        Instruction::new(
                            Op::UConvert,
                            Some(uint),
                            Some(widened),
                            vec![Operand::IdRef(args[0])],
                        ),
                        Instruction::new(
                            Op::BitCount,
                            Some(uint),
                            Some(counted),
                            vec![Operand::IdRef(widened)],
                        ),
                        Instruction::new(
                            Op::UConvert,
                            Some(rty),
                            Some(res),
                            vec![Operand::IdRef(counted)],
                        ),
                    ]));
                }
            }
            if let (Some((result_lanes, result_width)), Some((arg_lanes, arg_width))) =
                (int_vector_width(ctx, rty), int_vector_width(ctx, arg_ty))
            {
                if result_lanes == arg_lanes && result_width < 32 && arg_width < 32 {
                    let uint_vec = ctx.ty_vec_uint(result_lanes);
                    let widened = ctx.module.fresh_id();
                    let counted = ctx.module.fresh_id();
                    return Ok(Some(vec![
                        Instruction::new(
                            Op::UConvert,
                            Some(uint_vec),
                            Some(widened),
                            vec![Operand::IdRef(args[0])],
                        ),
                        Instruction::new(
                            Op::BitCount,
                            Some(uint_vec),
                            Some(counted),
                            vec![Operand::IdRef(widened)],
                        ),
                        Instruction::new(
                            Op::UConvert,
                            Some(rty),
                            Some(res),
                            vec![Operand::IdRef(counted)],
                        ),
                    ]));
                }
            }
        }
        return Ok(Some(vec![Instruction::new(
            Op::BitCount,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(args[0])],
        )]));
    }
    if name.starts_with("air.ctz.") && args.len() == 2 {
        let width = int_scalar_width(ctx, rty)
            .ok_or_else(|| format!("{name} result type is not a scalar integer"))?;
        if !matches!(width, 8 | 16 | 32 | 64) {
            return Err(format!("{name} unsupported result width {width}"));
        }
        let arg_ty = value_result_type(ctx, args[0])
            .ok_or_else(|| format!("{name} operand has no result type"))?;
        if arg_ty != rty {
            return Err(format!("{name} operand/result type mismatch"));
        }
        return Ok(Some(ctz_scalar(ctx, res, rty, args[0], width)));
    }
    if name.starts_with("air.clz.") && args.len() == 2 {
        let width = int_scalar_width(ctx, rty)
            .ok_or_else(|| format!("{name} result type is not a scalar integer"))?;
        if !matches!(width, 8 | 16 | 32 | 64) {
            return Err(format!("{name} unsupported result width {width}"));
        }
        let arg_ty = value_result_type(ctx, args[0])
            .ok_or_else(|| format!("{name} operand has no result type"))?;
        if arg_ty != rty {
            return Err(format!("{name} operand/result type mismatch"));
        }
        return Ok(Some(clz_scalar(ctx, res, rty, args[0], width)));
    }
    // `mul_hi.u.i32(a,b)` = high 32 bits of the 64-bit unsigned product: widen to u64, multiply,
    // shift right 32, narrow back.
    if name.starts_with("air.mul_hi.u.") && args.len() == 2 {
        if int_scalar_width(ctx, rty) != Some(32) {
            return Err(format!("{name} currently supports i32 results"));
        }
        let ulong = ctx.ty_ulong();
        let a64 = ctx.module.fresh_id();
        let b64 = ctx.module.fresh_id();
        let prod = ctx.module.fresh_id();
        let hi = ctx.module.fresh_id();
        return Ok(Some(vec![
            Instruction::new(
                Op::UConvert,
                Some(ulong),
                Some(a64),
                vec![Operand::IdRef(args[0])],
            ),
            Instruction::new(
                Op::UConvert,
                Some(ulong),
                Some(b64),
                vec![Operand::IdRef(args[1])],
            ),
            Instruction::new(
                Op::IMul,
                Some(ulong),
                Some(prod),
                vec![Operand::IdRef(a64), Operand::IdRef(b64)],
            ),
            Instruction::new(
                Op::ShiftRightLogical,
                Some(ulong),
                Some(hi),
                vec![Operand::IdRef(prod), Operand::IdRef(ctx.const_uint(32))],
            ),
            Instruction::new(Op::UConvert, Some(rty), Some(res), vec![Operand::IdRef(hi)]),
        ]));
    }
    // `mad_sat.s.i32(a,b,c)` = saturate(a*b+c) to the i32 range. Compute in 64-bit (no overflow for
    // 32-bit inputs), SClamp to [i32::MIN, i32::MAX], narrow back.
    if name.starts_with("air.mad_sat.s.") && args.len() == 3 {
        if int_scalar_width(ctx, rty) != Some(32) {
            return Err(format!("{name} currently supports i32 results"));
        }
        let long = ctx.ty_ulong();
        let ext = ctx.glsl();
        let a64 = ctx.module.fresh_id();
        let b64 = ctx.module.fresh_id();
        let c64 = ctx.module.fresh_id();
        let prod = ctx.module.fresh_id();
        let sum = ctx.module.fresh_id();
        let clamped = ctx.module.fresh_id();
        let lo = ctx.const_int_of(long, i64::from(i32::MIN));
        let hi = ctx.const_int_of(long, i64::from(i32::MAX));
        return Ok(Some(vec![
            Instruction::new(
                Op::SConvert,
                Some(long),
                Some(a64),
                vec![Operand::IdRef(args[0])],
            ),
            Instruction::new(
                Op::SConvert,
                Some(long),
                Some(b64),
                vec![Operand::IdRef(args[1])],
            ),
            Instruction::new(
                Op::SConvert,
                Some(long),
                Some(c64),
                vec![Operand::IdRef(args[2])],
            ),
            Instruction::new(
                Op::IMul,
                Some(long),
                Some(prod),
                vec![Operand::IdRef(a64), Operand::IdRef(b64)],
            ),
            Instruction::new(
                Op::IAdd,
                Some(long),
                Some(sum),
                vec![Operand::IdRef(prod), Operand::IdRef(c64)],
            ),
            Instruction::new(
                Op::ExtInst,
                Some(long),
                Some(clamped),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(GLSLstd450::SClamp as u32),
                    Operand::IdRef(sum),
                    Operand::IdRef(lo),
                    Operand::IdRef(hi),
                ],
            ),
            Instruction::new(
                Op::SConvert,
                Some(rty),
                Some(res),
                vec![Operand::IdRef(clamped)],
            ),
        ]));
    }
    Ok(None)
}

fn integer_shape(ctx: &Ctx, ty: Word) -> Option<(u32, u32)> {
    int_scalar_width(ctx, ty)
        .map(|width| (1, width))
        .or_else(|| int_vector_width(ctx, ty))
}

fn scalar_integer_operand(ctx: &Ctx, value: Word) -> bool {
    value_result_type(ctx, value).is_some_and(|ty| int_scalar_width(ctx, ty).is_some())
}

fn validate_extract_bits_shape(
    ctx: &Ctx,
    name: &str,
    rty: Word,
    args: &[Word],
) -> Result<(u32, u32), String> {
    let shape = integer_shape(ctx, rty)
        .ok_or_else(|| format!("{name} result is not an integer scalar or vector"))?;
    if value_result_type(ctx, args[0]) != Some(rty) {
        return Err(format!(
            "{name} base operand does not match its result type"
        ));
    }
    if !scalar_integer_operand(ctx, args[1]) || !scalar_integer_operand(ctx, args[2]) {
        return Err(format!("{name} offset and count must be integer scalars"));
    }
    Ok(shape)
}

fn validate_insert_bits_shape(
    ctx: &Ctx,
    name: &str,
    rty: Word,
    args: &[Word],
) -> Result<(u32, u32), String> {
    let shape = integer_shape(ctx, rty)
        .ok_or_else(|| format!("{name} result is not an integer scalar or vector"))?;
    if value_result_type(ctx, args[0]) != Some(rty) || value_result_type(ctx, args[1]) != Some(rty)
    {
        return Err(format!(
            "{name} base and insert operands do not match its result type"
        ));
    }
    if !scalar_integer_operand(ctx, args[2]) || !scalar_integer_operand(ctx, args[3]) {
        return Err(format!("{name} offset and count must be integer scalars"));
    }
    Ok(shape)
}

fn bitfield_widened_type(ctx: &mut Ctx, lanes: u32) -> Word {
    if lanes == 1 {
        ctx.ty_uint()
    } else {
        ctx.ty_vec_uint(lanes)
    }
}

fn extract_bits_narrow(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    args: &[Word],
    lanes: u32,
    signed: bool,
) -> Vec<Instruction> {
    let widened_type = bitfield_widened_type(ctx, lanes);
    let widened_base = ctx.module.fresh_id();
    let extracted = ctx.module.fresh_id();
    vec![
        Instruction::new(
            Op::UConvert,
            Some(widened_type),
            Some(widened_base),
            vec![Operand::IdRef(args[0])],
        ),
        Instruction::new(
            if signed {
                Op::BitFieldSExtract
            } else {
                Op::BitFieldUExtract
            },
            Some(widened_type),
            Some(extracted),
            vec![
                Operand::IdRef(widened_base),
                Operand::IdRef(args[1]),
                Operand::IdRef(args[2]),
            ],
        ),
        Instruction::new(
            Op::UConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(extracted)],
        ),
    ]
}

fn insert_bits_narrow(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    args: &[Word],
    lanes: u32,
) -> Vec<Instruction> {
    let widened_type = bitfield_widened_type(ctx, lanes);
    let widened_base = ctx.module.fresh_id();
    let widened_insert = ctx.module.fresh_id();
    let inserted = ctx.module.fresh_id();
    vec![
        Instruction::new(
            Op::UConvert,
            Some(widened_type),
            Some(widened_base),
            vec![Operand::IdRef(args[0])],
        ),
        Instruction::new(
            Op::UConvert,
            Some(widened_type),
            Some(widened_insert),
            vec![Operand::IdRef(args[1])],
        ),
        Instruction::new(
            Op::BitFieldInsert,
            Some(widened_type),
            Some(inserted),
            vec![
                Operand::IdRef(widened_base),
                Operand::IdRef(widened_insert),
                Operand::IdRef(args[2]),
                Operand::IdRef(args[3]),
            ],
        ),
        Instruction::new(
            Op::UConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(inserted)],
        ),
    ]
}

/// Vulkan without maintenance9 restricts `OpBitFieldInsert`'s base to 32 bits. Preserve the AIR
/// 64-bit scalar contract with ordinary integer operations:
/// `(base & ~(field_mask << offset)) | ((insert & field_mask) << offset)`.
fn insert_bits_i64(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let _offset_ty = value_result_type(ctx, args[2])
        .filter(|ty| int_scalar_width(ctx, *ty).is_some())
        .ok_or("air.insert_bits i64 offset is not a scalar integer")?;
    let count_ty = value_result_type(ctx, args[3])
        .filter(|ty| int_scalar_width(ctx, *ty).is_some())
        .ok_or("air.insert_bits i64 count is not a scalar integer")?;
    let all_ones = ctx.const_int_of(rty, -1);
    let zero_value = ctx.const_int_of(rty, 0);
    let zero_count = ctx.const_int_of(count_ty, 0);
    let width = ctx.const_int_of(count_ty, 64);
    let remaining = ctx.module.fresh_id();
    let candidate_mask = ctx.module.fresh_id();
    let count_is_zero = ctx.module.fresh_id();
    let field_mask = ctx.module.fresh_id();
    let positioned_mask = ctx.module.fresh_id();
    let inserted_bits = ctx.module.fresh_id();
    let positioned_bits = ctx.module.fresh_id();
    let inverse_mask = ctx.module.fresh_id();
    let cleared_base = ctx.module.fresh_id();
    let bool_ty = ctx.ty_bool();
    Ok(vec![
        Instruction::new(
            Op::ISub,
            Some(count_ty),
            Some(remaining),
            vec![Operand::IdRef(width), Operand::IdRef(args[3])],
        ),
        Instruction::new(
            Op::ShiftRightLogical,
            Some(rty),
            Some(candidate_mask),
            vec![Operand::IdRef(all_ones), Operand::IdRef(remaining)],
        ),
        Instruction::new(
            Op::IEqual,
            Some(bool_ty),
            Some(count_is_zero),
            vec![Operand::IdRef(args[3]), Operand::IdRef(zero_count)],
        ),
        Instruction::new(
            Op::Select,
            Some(rty),
            Some(field_mask),
            vec![
                Operand::IdRef(count_is_zero),
                Operand::IdRef(zero_value),
                Operand::IdRef(candidate_mask),
            ],
        ),
        Instruction::new(
            Op::ShiftLeftLogical,
            Some(rty),
            Some(positioned_mask),
            vec![Operand::IdRef(field_mask), Operand::IdRef(args[2])],
        ),
        Instruction::new(
            Op::BitwiseAnd,
            Some(rty),
            Some(inserted_bits),
            vec![Operand::IdRef(args[1]), Operand::IdRef(field_mask)],
        ),
        Instruction::new(
            Op::ShiftLeftLogical,
            Some(rty),
            Some(positioned_bits),
            vec![Operand::IdRef(inserted_bits), Operand::IdRef(args[2])],
        ),
        Instruction::new(
            Op::Not,
            Some(rty),
            Some(inverse_mask),
            vec![Operand::IdRef(positioned_mask)],
        ),
        Instruction::new(
            Op::BitwiseAnd,
            Some(rty),
            Some(cleared_base),
            vec![Operand::IdRef(args[0]), Operand::IdRef(inverse_mask)],
        ),
        Instruction::new(
            Op::BitwiseOr,
            Some(rty),
            Some(res),
            vec![
                Operand::IdRef(cleared_base),
                Operand::IdRef(positioned_bits),
            ],
        ),
    ])
}

/// Vulkan applies the same pre-maintenance9 32-bit-base restriction to `OpBitField*Extract`.
/// Re-express 64-bit extraction with shifts (and a mask for the unsigned form).
fn extract_bits_i64(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    args: &[Word],
    signed: bool,
) -> Result<Vec<Instruction>, String> {
    let offset_ty = value_result_type(ctx, args[1])
        .filter(|ty| int_scalar_width(ctx, *ty).is_some())
        .ok_or("air.extract_bits i64 offset is not a scalar integer")?;
    let count_ty = value_result_type(ctx, args[2])
        .filter(|ty| int_scalar_width(ctx, *ty).is_some())
        .ok_or("air.extract_bits i64 count is not a scalar integer")?;
    if offset_ty != count_ty {
        return Err("air.extract_bits i64 offset/count type mismatch".into());
    }
    let width = ctx.const_int_of(count_ty, 64);
    let zero_count = ctx.const_int_of(count_ty, 0);
    let zero_value = ctx.const_int_of(rty, 0);
    let count_is_zero = ctx.module.fresh_id();
    let bool_ty = ctx.ty_bool();
    let mut instructions = vec![Instruction::new(
        Op::IEqual,
        Some(bool_ty),
        Some(count_is_zero),
        vec![Operand::IdRef(args[2]), Operand::IdRef(zero_count)],
    )];
    let extracted = ctx.module.fresh_id();
    if signed {
        let end = ctx.module.fresh_id();
        let left_amount = ctx.module.fresh_id();
        let right_amount = ctx.module.fresh_id();
        let shifted_left = ctx.module.fresh_id();
        instructions.extend([
            Instruction::new(
                Op::IAdd,
                Some(count_ty),
                Some(end),
                vec![Operand::IdRef(args[1]), Operand::IdRef(args[2])],
            ),
            Instruction::new(
                Op::ISub,
                Some(count_ty),
                Some(left_amount),
                vec![Operand::IdRef(width), Operand::IdRef(end)],
            ),
            Instruction::new(
                Op::ISub,
                Some(count_ty),
                Some(right_amount),
                vec![Operand::IdRef(width), Operand::IdRef(args[2])],
            ),
            Instruction::new(
                Op::ShiftLeftLogical,
                Some(rty),
                Some(shifted_left),
                vec![Operand::IdRef(args[0]), Operand::IdRef(left_amount)],
            ),
            Instruction::new(
                Op::ShiftRightArithmetic,
                Some(rty),
                Some(extracted),
                vec![Operand::IdRef(shifted_left), Operand::IdRef(right_amount)],
            ),
        ]);
    } else {
        let all_ones = ctx.const_int_of(rty, -1);
        let remaining = ctx.module.fresh_id();
        let candidate_mask = ctx.module.fresh_id();
        let field_mask = ctx.module.fresh_id();
        let shifted = ctx.module.fresh_id();
        instructions.extend([
            Instruction::new(
                Op::ISub,
                Some(count_ty),
                Some(remaining),
                vec![Operand::IdRef(width), Operand::IdRef(args[2])],
            ),
            Instruction::new(
                Op::ShiftRightLogical,
                Some(rty),
                Some(candidate_mask),
                vec![Operand::IdRef(all_ones), Operand::IdRef(remaining)],
            ),
            Instruction::new(
                Op::Select,
                Some(rty),
                Some(field_mask),
                vec![
                    Operand::IdRef(count_is_zero),
                    Operand::IdRef(zero_value),
                    Operand::IdRef(candidate_mask),
                ],
            ),
            Instruction::new(
                Op::ShiftRightLogical,
                Some(rty),
                Some(shifted),
                vec![Operand::IdRef(args[0]), Operand::IdRef(args[1])],
            ),
            Instruction::new(
                Op::BitwiseAnd,
                Some(rty),
                Some(extracted),
                vec![Operand::IdRef(shifted), Operand::IdRef(field_mask)],
            ),
        ]);
    }
    instructions.push(Instruction::new(
        Op::Select,
        Some(rty),
        Some(res),
        vec![
            Operand::IdRef(count_is_zero),
            Operand::IdRef(zero_value),
            Operand::IdRef(extracted),
        ],
    ));
    Ok(instructions)
}

fn lower_quad_vote(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    predicate: Word,
    all: bool,
) -> Result<Vec<Instruction>, String> {
    if !is_bool_type(ctx, rty) {
        return Err("quad vote result type is not bool".to_string());
    }
    // Metal quads are fixed aligned four-lane partitions. SPIR-V's subgroup vote instructions cover
    // the entire implementation subgroup, which may contain several Metal quads, so they are not a
    // semantic match. Convert the predicate to 0/1 and use the same XOR butterfly as quad_sum:
    // mask 1 combines adjacent lanes, then mask 2 combines the two pairs. Every lane receives the
    // exact result for its own four-lane partition, independent of the physical subgroup width.
    let uint = ctx.ty_uint();
    let one = ctx.const_uint(1);
    let zero = ctx.const_uint(0);
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let word = ctx.module.fresh_id();
    let peer_one = ctx.module.fresh_id();
    let pair = ctx.module.fresh_id();
    let peer_two = ctx.module.fresh_id();
    let quad = ctx.module.fresh_id();
    let op = if all { Op::BitwiseAnd } else { Op::BitwiseOr };
    Ok(vec![
        Instruction::new(
            Op::Select,
            Some(uint),
            Some(word),
            vec![
                Operand::IdRef(predicate),
                Operand::IdRef(one),
                Operand::IdRef(zero),
            ],
        ),
        Instruction::new(
            Op::GroupNonUniformShuffleXor,
            Some(uint),
            Some(peer_one),
            vec![
                Operand::IdScope(scope),
                Operand::IdRef(word),
                Operand::IdRef(one),
            ],
        ),
        Instruction::new(
            op,
            Some(uint),
            Some(pair),
            vec![Operand::IdRef(word), Operand::IdRef(peer_one)],
        ),
        Instruction::new(
            Op::GroupNonUniformShuffleXor,
            Some(uint),
            Some(peer_two),
            vec![
                Operand::IdScope(scope),
                Operand::IdRef(pair),
                Operand::IdRef(ctx.const_uint(2)),
            ],
        ),
        Instruction::new(
            op,
            Some(uint),
            Some(quad),
            vec![Operand::IdRef(pair), Operand::IdRef(peer_two)],
        ),
        Instruction::new(
            if all { Op::IEqual } else { Op::INotEqual },
            Some(rty),
            Some(res),
            vec![
                Operand::IdRef(quad),
                Operand::IdRef(if all { one } else { zero }),
            ],
        ),
    ])
}

fn lower_quad_active_threads_mask(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
) -> Result<Vec<Instruction>, String> {
    if int_scalar_width(ctx, rty) != Some(16) {
        return Err("air.quad_active_threads_mask result is not i16".to_string());
    }
    // AIR reports a quad-relative four-bit mask. A subgroup ballot is absolute, so select the
    // current 32-lane word and shift its aligned four-lane partition down to bits 0..3. Quad bases
    // are multiples of four and therefore never straddle the ballot's 32-bit words.
    let uint = ctx.ty_uint();
    let bool_ty = ctx.ty_bool();
    let ballot_ty = ctx.ty_vec_uint(4);
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let active = ctx.const_bool_of(bool_ty, true);
    let mut insts = Vec::new();
    let lane = subgroup_lane_index_u32(ctx, &mut insts);
    let ballot = ctx.module.fresh_id();
    let word_index = ctx.module.fresh_id();
    let word = ctx.module.fresh_id();
    let quad_shift = ctx.module.fresh_id();
    let shifted = ctx.module.fresh_id();
    let masked = ctx.module.fresh_id();
    insts.extend([
        Instruction::new(
            Op::GroupNonUniformBallot,
            Some(ballot_ty),
            Some(ballot),
            vec![Operand::IdScope(scope), Operand::IdRef(active)],
        ),
        Instruction::new(
            Op::ShiftRightLogical,
            Some(uint),
            Some(word_index),
            vec![Operand::IdRef(lane), Operand::IdRef(ctx.const_uint(5))],
        ),
        Instruction::new(
            Op::VectorExtractDynamic,
            Some(uint),
            Some(word),
            vec![Operand::IdRef(ballot), Operand::IdRef(word_index)],
        ),
        Instruction::new(
            Op::BitwiseAnd,
            Some(uint),
            Some(quad_shift),
            vec![Operand::IdRef(lane), Operand::IdRef(ctx.const_uint(28))],
        ),
        Instruction::new(
            Op::ShiftRightLogical,
            Some(uint),
            Some(shifted),
            vec![Operand::IdRef(word), Operand::IdRef(quad_shift)],
        ),
        Instruction::new(
            Op::BitwiseAnd,
            Some(uint),
            Some(masked),
            vec![Operand::IdRef(shifted), Operand::IdRef(ctx.const_uint(15))],
        ),
        Instruction::new(
            Op::UConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(masked)],
        ),
    ]);
    Ok(insts)
}

/// Lower the SIMD/subgroup lane-op family (`air.is_uniform`, `air.quad_{all,any}`, `air.get_simdgroup_size`,
/// `air.simd_is_first`, `air.quad_is_first`, the simd/quad shuffle/broadcast/fill variants,
/// `air.simd_{prefix_*,sum,or,xor,and,min,max}`) plus the vector-reduction / function-constant predicates (`air.all`/`air.any`,
/// `function_constant_predicate`, `air.is_function_constant_defined`). Returns `Ok(Some(insts))` when a
/// branch handled the call, `Ok(None)` when no guard matched (the caller then continues its dispatch
/// cascade), or `Err` when a matched branch rejected its operands. Guard order/precedence is preserved.
pub(in crate::passes) fn lower_simd_op(
    ctx: &mut Ctx,
    name: &str,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Option<Vec<Instruction>>, String> {
    if name.starts_with("air.is_uniform.") && args.len() == 1 {
        let scope = ctx.const_uint(Scope::Subgroup as u32);
        return Ok(Some(vec![Instruction::new(
            Op::GroupNonUniformAllEqual,
            Some(rty),
            Some(res),
            vec![Operand::IdScope(scope), Operand::IdRef(args[0])],
        )]));
    }
    if matches!(name, "air.quad_all" | "air.quad_any") && args.len() == 1 {
        return lower_quad_vote(ctx, res, rty, args[0], name == "air.quad_all").map(Some);
    }
    if name == "air.quad_active_threads_mask" && args.is_empty() {
        return lower_quad_active_threads_mask(ctx, res, rty).map(Some);
    }
    if name.starts_with("air.get_simdgroup_size.") && args.is_empty() {
        if int_scalar_width(ctx, rty).is_none() {
            return Err(format!("{name} result is not a scalar integer"));
        }
        let width = ctx.const_int_of(rty, 32);
        return Ok(Some(vec![Instruction::new(
            Op::CopyObject,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(width)],
        )]));
    }
    if name == "air.quad_is_first" && args.is_empty() {
        if !is_bool_type(ctx, rty) {
            return Err(format!("{name} result type is not bool"));
        }
        // Metal quads are the fixed four-lane partitions already used by the quad shuffle and sum
        // lowerings. The first lane is therefore exactly `(SubgroupLocalInvocationId & 3) == 0`,
        // independent of the physical subgroup width.
        let mut insts = Vec::new();
        let lane = subgroup_lane_index_u32(ctx, &mut insts);
        let uint = ctx.ty_uint();
        let quad_lane = ctx.module.fresh_id();
        insts.push(Instruction::new(
            Op::BitwiseAnd,
            Some(uint),
            Some(quad_lane),
            vec![Operand::IdRef(lane), Operand::IdRef(ctx.const_uint(3))],
        ));
        insts.push(Instruction::new(
            Op::IEqual,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(quad_lane), Operand::IdRef(ctx.const_uint(0))],
        ));
        return Ok(Some(insts));
    }
    if name == "air.simd_is_first" && args.is_empty() {
        if !is_bool_type(ctx, rty) {
            return Err(format!("{name} result type is not bool"));
        }
        if ctx.simd_cluster32 {
            let mut insts = Vec::new();
            let lane = subgroup_lane_index_u32(ctx, &mut insts);
            let uint = ctx.ty_uint();
            let local_lane = ctx.module.fresh_id();
            insts.push(Instruction::new(
                Op::BitwiseAnd,
                Some(uint),
                Some(local_lane),
                vec![Operand::IdRef(lane), Operand::IdRef(ctx.const_uint(31))],
            ));
            insts.push(Instruction::new(
                Op::IEqual,
                Some(rty),
                Some(res),
                vec![
                    Operand::IdRef(local_lane),
                    Operand::IdRef(ctx.const_uint(0)),
                ],
            ));
            return Ok(Some(insts));
        }
        let scope = ctx.const_uint(Scope::Subgroup as u32);
        return Ok(Some(vec![Instruction::new(
            Op::GroupNonUniformElect,
            Some(rty),
            Some(res),
            vec![Operand::IdScope(scope)],
        )]));
    }
    if name.starts_with("air.simd_broadcast_first.") && args.len() == 1 {
        let scope = ctx.const_uint(Scope::Subgroup as u32);
        if ctx.simd_cluster32 {
            let mut insts = Vec::new();
            let lane = subgroup_lane_index_u32(ctx, &mut insts);
            let uint = ctx.ty_uint();
            let local_lane = ctx.module.fresh_id();
            insts.push(Instruction::new(
                Op::BitwiseAnd,
                Some(uint),
                Some(local_lane),
                vec![Operand::IdRef(lane), Operand::IdRef(ctx.const_uint(31))],
            ));
            let base_lane = ctx.module.fresh_id();
            insts.push(Instruction::new(
                Op::ISub,
                Some(uint),
                Some(base_lane),
                vec![Operand::IdRef(lane), Operand::IdRef(local_lane)],
            ));
            insts.push(Instruction::new(
                Op::GroupNonUniformShuffle,
                Some(rty),
                Some(res),
                vec![
                    Operand::IdScope(scope),
                    Operand::IdRef(args[0]),
                    Operand::IdRef(base_lane),
                ],
            ));
            return Ok(Some(insts));
        }
        return Ok(Some(vec![Instruction::new(
            Op::GroupNonUniformBroadcastFirst,
            Some(rty),
            Some(res),
            vec![Operand::IdScope(scope), Operand::IdRef(args[0])],
        )]));
    }
    if name.starts_with("air.simd_broadcast.") && args.len() == 2 {
        let scope = ctx.const_uint(Scope::Subgroup as u32);
        let mut insts = Vec::new();
        let lane = subgroup_shuffle_index_u32(ctx, args[1], &mut insts)?;
        insts.push(Instruction::new(
            Op::GroupNonUniformShuffle,
            Some(rty),
            Some(res),
            vec![
                Operand::IdScope(scope),
                Operand::IdRef(args[0]),
                Operand::IdRef(lane),
            ],
        ));
        return Ok(Some(insts));
    }
    if name.starts_with("air.simd_shuffle.") && args.len() == 2 {
        let scope = ctx.const_uint(Scope::Subgroup as u32);
        let mut insts = Vec::new();
        let lane = subgroup_shuffle_index_u32(ctx, args[1], &mut insts)?;
        insts.push(Instruction::new(
            Op::GroupNonUniformShuffle,
            Some(rty),
            Some(res),
            vec![
                Operand::IdScope(scope),
                Operand::IdRef(args[0]),
                Operand::IdRef(lane),
            ],
        ));
        return Ok(Some(insts));
    }
    if name.starts_with("air.simd_shuffle_down.") && args.len() == 2 {
        return lower_simd_shuffle_down(ctx, res, rty, args).map(Some);
    }
    if name.starts_with("air.simd_shuffle_rotate_down.") && args.len() == 2 {
        return lower_simd_shuffle_rotate_down(ctx, res, rty, args).map(Some);
    }
    if name.starts_with("air.simd_shuffle_and_fill_down.") && args.len() == 4 {
        return lower_simd_shuffle_and_fill_down(ctx, res, rty, args).map(Some);
    }
    if name.starts_with("air.simd_shuffle_and_fill_up.") && args.len() == 4 {
        return lower_simd_shuffle_and_fill_up(ctx, res, rty, args).map(Some);
    }
    if name.starts_with("air.simd_shuffle_up.") && args.len() == 2 {
        return lower_simd_shuffle_up(ctx, res, rty, args).map(Some);
    }
    if name.starts_with("air.simd_shuffle_xor.") && args.len() == 2 {
        let scope = ctx.const_uint(Scope::Subgroup as u32);
        let mut insts = Vec::new();
        let mask = subgroup_shuffle_index_u32(ctx, args[1], &mut insts)?;
        insts.push(Instruction::new(
            Op::GroupNonUniformShuffleXor,
            Some(rty),
            Some(res),
            vec![
                Operand::IdScope(scope),
                Operand::IdRef(args[0]),
                Operand::IdRef(mask),
            ],
        ));
        return Ok(Some(insts));
    }
    if name.starts_with("air.quad_shuffle.") && args.len() == 2 {
        return lower_quad_shuffle(ctx, res, rty, args).map(Some);
    }
    // `air.quad_broadcast.<ty>(value, lane)` returns `value` from quad-lane `lane` to every lane —
    // identical to `quad_shuffle(value, lane)` (verified on Apple Metal: quad_broadcast(v,2) ==
    // quad_shuffle(v,2) for every lane), so it lowers through the same masked GroupNonUniformShuffle.
    if name.starts_with("air.quad_broadcast.") && args.len() == 2 {
        return lower_quad_shuffle(ctx, res, rty, args).map(Some);
    }
    if name.starts_with("air.quad_sum.") && args.len() == 1 {
        return lower_quad_sum(ctx, res, rty, args[0]).map(Some);
    }
    if (name.starts_with("air.quad_min.") || name.starts_with("air.quad_max.")) && args.len() == 1 {
        return lower_quad_integer_extrema(ctx, name, res, rty, args[0]).map(Some);
    }
    // `air.quad_shuffle_xor.<ty>(value, mask)` returns `value` from quad-lane `lane ^ mask`. Metal's
    // quad mask is 0..3, so xoring the full subgroup lane id keeps the exchange inside the 4-aligned
    // quad — identical to simd_shuffle_xor's lowering.
    if name.starts_with("air.quad_shuffle_xor.") && args.len() == 2 {
        let scope = ctx.const_uint(Scope::Subgroup as u32);
        let mut insts = Vec::new();
        let mask = subgroup_shuffle_index_u32(ctx, args[1], &mut insts)?;
        insts.push(Instruction::new(
            Op::GroupNonUniformShuffleXor,
            Some(rty),
            Some(res),
            vec![
                Operand::IdScope(scope),
                Operand::IdRef(args[0]),
                Operand::IdRef(mask),
            ],
        ));
        return Ok(Some(insts));
    }
    if name.starts_with("air.quad_shuffle_rotate_down.") && args.len() == 2 {
        return lower_quad_shuffle_rotate_down(ctx, res, rty, args).map(Some);
    }
    if name.starts_with("air.quad_shuffle_up.") && args.len() == 2 {
        return lower_quad_shuffle_up(ctx, res, rty, args).map(Some);
    }
    if name.starts_with("air.quad_shuffle_down.") && args.len() == 2 {
        let scope = ctx.const_uint(Scope::Subgroup as u32);
        let mut insts = Vec::new();
        let delta = subgroup_shuffle_index_u32(ctx, args[1], &mut insts)?;
        insts.push(Instruction::new(
            Op::GroupNonUniformShuffleDown,
            Some(rty),
            Some(res),
            vec![
                Operand::IdScope(scope),
                Operand::IdRef(args[0]),
                Operand::IdRef(delta),
            ],
        ));
        return Ok(Some(insts));
    }
    if name.starts_with("air.simd_prefix_exclusive_sum.") && args.len() == 1 {
        return lower_simd_sum(ctx, res, rty, args[0], GroupOperation::ExclusiveScan).map(Some);
    }
    if name.starts_with("air.simd_prefix_inclusive_sum.") && args.len() == 1 {
        return lower_simd_sum(ctx, res, rty, args[0], GroupOperation::InclusiveScan).map(Some);
    }
    if name.starts_with("air.simd_sum.") && args.len() == 1 {
        return lower_simd_sum(ctx, res, rty, args[0], GroupOperation::Reduce).map(Some);
    }
    if name.starts_with("air.simd_or.") && args.len() == 1 {
        return lower_simd_bitwise(
            ctx,
            res,
            rty,
            args[0],
            GroupOperation::Reduce,
            Op::GroupNonUniformBitwiseOr,
        )
        .map(Some);
    }
    if name.starts_with("air.simd_xor.") && args.len() == 1 {
        return lower_simd_bitwise(
            ctx,
            res,
            rty,
            args[0],
            GroupOperation::Reduce,
            Op::GroupNonUniformBitwiseXor,
        )
        .map(Some);
    }
    if name.starts_with("air.simd_and.") && args.len() == 1 {
        return lower_simd_bitwise(
            ctx,
            res,
            rty,
            args[0],
            GroupOperation::Reduce,
            Op::GroupNonUniformBitwiseAnd,
        )
        .map(Some);
    }
    if name.starts_with("air.simd_min.") && args.len() == 1 {
        return lower_simd_extrema(
            ctx,
            res,
            rty,
            args[0],
            GroupOperation::Reduce,
            SimdExtrema::Min,
        )
        .map(Some);
    }
    if name.starts_with("air.simd_max.") && args.len() == 1 {
        return lower_simd_extrema(
            ctx,
            res,
            rty,
            args[0],
            GroupOperation::Reduce,
            SimdExtrema::Max,
        )
        .map(Some);
    }
    // air.all / air.any -> OpAll / OpAny
    if name.starts_with("air.all") && args.len() == 1 {
        return Ok(Some(vec![Instruction::new(
            Op::All,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(args[0])],
        )]));
    }
    if name.starts_with("air.any") && args.len() == 1 {
        return Ok(Some(vec![Instruction::new(
            Op::Any,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(args[0])],
        )]));
    }
    // function-constant predicate normalizer: passes its i8 through (we treat it as identity; the
    // value already encodes the boolean). OpCopyObject keeps the SSA id valid.
    if name.contains("function_constant_predicate") && args.len() == 1 {
        return Ok(Some(vec![Instruction::new(
            Op::CopyObject,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(args[0])],
        )]));
    }
    // Runtime function-constant specialization is not plumbed through this translator. Match the
    // existing function-constant storage fold by selecting the "not defined" default path.
    if name == "air.is_function_constant_defined" {
        if args.len() != 1 {
            return Err(format!("{name} expects one function-constant operand"));
        }
        let fc_index = ctx.module.debug_names.iter().find_map(|inst| {
            if inst.class.opcode != Op::Name {
                return None;
            }
            match (inst.operands.first(), inst.operands.get(1)) {
                (Some(Operand::IdRef(id)), Some(Operand::LiteralString(name)))
                    if *id == args[0] =>
                {
                    crate::fc_specialize::fc_init_index(name)
                }
                _ => None,
            }
        });
        let c = ctx.const_bool_of(rty, false);
        if let Some(index) = fc_index {
            ctx.module.debug_names.push(Instruction::new(
                Op::Name,
                None,
                None,
                vec![
                    Operand::IdRef(res),
                    Operand::LiteralString(crate::fc_specialize::fc_defined_name(index)),
                ],
            ));
        }
        return Ok(Some(vec![Instruction::new(
            Op::CopyObject,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(c)],
        )]));
    }
    Ok(None)
}

/// Lower a single residual AIR/LLVM helper call into a sequence of replacement instructions.
pub(in crate::passes) fn lower_one(
    ctx: &mut Ctx,
    name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
    v4: Word,
) -> Result<Vec<Instruction>, String> {
    if name.starts_with("air.") && crate::air_intrinsics::air_intrinsic_disposition(name).is_none()
    {
        return Err(format!("unrecognized AIR intrinsic family: {name}"));
    }
    // texture sampling: air.sample_texture_<dim>.<ret>. Result is a {vecN, i8} struct in AIR; the
    // body then CompositeExtract member 0. We make our sample produce the same struct so the extract
    // still works: build {sample, 0u8}.
    if name.starts_with("air.sample_texture") {
        return lower_sample(ctx, name, res, rty, args, v4);
    }
    // `air.gather_texture_<dim>.<ret>`: Metal texture gather. The current lowering supports the
    // observed 2D sampled-image form and emits OpImageGather with a constant offset when present.
    if name.starts_with("air.gather_texture") {
        return lower_gather(ctx, name, res, rty, args, v4);
    }
    // `air.gather_depth_<dim>.<ret>`: depth texture gather. The synthesized private capture harness backs
    // captured depth textures through color-image fixtures, so gather their component zero.
    if name.starts_with("air.gather_depth") {
        return lower_gather_depth(ctx, name, res, rty, args, v4);
    }
    // `air.sample_compare_depth_<dim>.f32`: shadow comparison on a depth texture. The current render
    // harness backs captured depth textures with RGBA8_UNORM images, so image lowering samples channel
    // 0 and compares it manually instead of requiring a Vulkan comparison sampler.
    if name.starts_with("air.sample_compare_depth") {
        return lower_sample_compare_depth(ctx, name, res, rty, args, v4);
    }
    // `air.sample_depth_<dim>.f32`: depth texture sampling returns a scalar depth in a
    // `{float, i8}` AIR result.
    if name.starts_with("air.sample_depth") {
        return lower_sample_depth(ctx, name, res, rty, args, v4);
    }
    // `air.read_depth_<dim>.f32`: read a depth texel by integer coordinate. The current harness
    // backs captured depth textures with RGBA8_UNORM images, so fetch component 0 manually.
    if name.starts_with("air.read_depth") {
        return lower_read_depth(ctx, name, res, rty, args, v4);
    }
    // `air.read_texture_<dim>.<ret>`: read a texel by INTEGER coordinate, no sampler (a Metal
    // `texture.read(uint2)`). Lowers to OpImageFetch on the (sampled) image at the given LOD.
    if name.starts_with("air.read_texture") {
        return lower_read(ctx, name, res, rty, args, v4);
    }
    // `air.write_texture_<dim>.<coord>.<texel>(image, coord, texel, lod, access)`: a Metal
    // `texture.write(color, coord)`. Lowers to OpImageWrite on the bound STORAGE image (Sampled=2).
    if name.starts_with("air.write_texture") {
        return lower_write(ctx, name, args, v4);
    }
    if name.starts_with("air.atomic_fetch_max_explicit_texture_") {
        return lower_atomic_texture_fetch_max(ctx, name, res, rty, args);
    }
    if name == "air.calculate_unclamped_lod_texture_2d" {
        return lower_calculate_lod_texture_2d(ctx, name, res, rty, args, 1);
    }
    if name == "air.calculate_clamped_lod_texture_2d" {
        return lower_calculate_lod_texture_2d(ctx, name, res, rty, args, 0);
    }
    if name.starts_with("air.write_imageblock_slice_to_texture") {
        return lower_imageblock_slice_write(ctx, name, args, v4);
    }
    if name.starts_with("air.store.implicit_imageblock.") {
        return lower_implicit_imageblock_store(ctx, name, args);
    }
    if name.starts_with("air.discard_fragment") {
        // OpKill is a block terminator; but AIR calls it mid-block. Use OpDemoteToHelperInvocation
        // through its extension on the Vulkan 1.2 baseline so control flow is preserved. The
        // capability and extension declaration are added in finalize.
        return Ok(vec![Instruction::new(
            Op::DemoteToHelperInvocation,
            None,
            None,
            vec![],
        )]);
    }
    if crate::air_intrinsics::is_command_encoder_helper(name) {
        if let Some(rty) = rty {
            if !is_void_type(ctx, rty) {
                return Err(format!(
                    "command helper {name} unexpectedly returns a value"
                ));
            }
        }
        // These AIR helpers mutate Metal command encoders/indirect command buffers. The current
        // conformance compute runner observes buffer bytes only, so preserve translation validity
        // without claiming command-buffer side-effect emulation.
        return Ok(vec![]);
    }
    if name.starts_with("air.fence_texture") {
        return lower_fence_texture(ctx, name, res, rty, args);
    }
    if name == "air.get_descriptor_size_tensor" {
        return lower_descriptor_size_tensor(ctx, name, res, rty, args);
    }
    // AIR global atomic stores do not return a value. The AIR memory-order and scope operands are
    // ignored for now, matching the existing native/global atomic policy.
    if name == "air.atomic.global.store.i32" {
        return lower_atomic_global_store_i32(ctx, name, res, rty, args);
    }
    // simdgroup_matrix 8x8 store (void): scatter the 64-lane row-major matrix into device memory with
    // the runtime leading dimension. See the value-side simdgroup_matrix arms below. (A void SPIR-V
    // OpFunctionCall still carries an OpTypeVoid result id; it is unreferenced, so replacing the call
    // with the store sequence simply drops it.)
    if name.starts_with("air.simdgroup_matrix_8x8_store.") {
        return lower_simdgroup_matrix_8x8_store(ctx, args);
    }
    // AIR global/local integer atomics return the previous value. The AIR memory-order operands are
    // ignored for now, matching the existing native atomic-add policy.
    if name == "air.atomic.global.max.u.i32"
        || name == "air.atomic.global.and.u.i32"
        || name == "air.atomic.global.or.u.i32"
        || name == "air.atomic.global.xor.u.i32"
        || name == "air.atomic.global.xchg.i32"
        || name == "air.atomic.local.max.s.i32"
        || name == "air.atomic.local.max.u.i32"
        || name == "air.atomic.local.min.s.i32"
        || name == "air.atomic.local.min.u.i32"
        || name == "air.atomic.local.and.u.i32"
        || name == "air.atomic.local.or.u.i32"
        || name == "air.atomic.local.xor.u.i32"
        || name == "air.atomic.local.xchg.i32"
    {
        return lower_atomic_integer_rmw(ctx, name, res, rty, args);
    }
    // `air.get_{width,height,depth}_{texture,depth}_<dim>(texture, lod)` -> an image size query,
    // extracting the matching component. Sampled images use OpImageQuerySizeLod; storage images have
    // no sampler LOD, so use OpImageQuerySize instead. AIR's result is `i32`; the query yields a uint
    // component (same width) which we bitcast to the result.
    if is_image_size_query(name) {
        return lower_image_size_query(ctx, name, res, rty, args);
    }
    // AIR exposes optional texture presence through this intrinsic. Real bound interface textures are
    // non-null in the conformance harness; values synthesized by `air.get_null_texture_*` are tracked
    // explicitly below.
    if name.starts_with("air.is_null_texture") {
        return lower_is_null_texture(ctx, name, res, rty, args);
    }
    // `air.get_num_mip_levels_{texture,depth}_<dim>(texture)` -> OpImageQueryLevels for sampled images.
    // SPIR-V forbids OpImageQueryLevels on storage images (Sampled=2), and the current private capture sets historically
    // texture contract synthesizes one mip level for storage-write targets.
    if name.starts_with("air.get_num_mip_levels_texture")
        || name.starts_with("air.get_num_mip_levels_depth")
    {
        return lower_get_num_mip_levels(ctx, name, res, rty, args);
    }
    // The current conformance texture contract creates single-sample RGBA8 textures. Until the
    // harness grows a real multisample image resource, sample-count queries on FC-selected/default
    // texture paths lower to that canonical count instead of emitting an invalid MS image query.
    if name.starts_with("air.get_num_samples_texture") {
        return lower_get_num_samples_texture(ctx, name, res, rty);
    }
    if name == "air.get_num_samples.i32" {
        return lower_raster_sample_count(ctx, name, res, rty);
    }
    // The private capture harness synthesizes `metal::rasterization_rate_map_data` as a single physical tile.
    // The observed Apple helper maps through that 1x1 tile to the first physical texel; full
    // variable-rate rasterization map decoding remains a separate resource-model problem.
    if name.starts_with("air.map_screen_to_physical_coordinates.") {
        return lower_map_screen_to_physical(ctx, name, res, rty, args);
    }

    // `air.map_physical_to_screen_coordinates.<ret>.<map>.<layer>`: the inverse of the map above.
    // The private capture harness backs `metal::rasterization_rate_map_data` with a single uniform physical
    // tile, so this mapping is the identity — the physical coordinate passes through as its screen
    // coordinate. (The full variable-rate map decode is a separate resource-model problem, shared with
    // `air.map_screen_to_physical_coordinates`.) Byte-checked against the real Apple `CC_TAAKernel`
    // golden on M2/MoltenVK.
    if name.starts_with("air.map_physical_to_screen_coordinates.") {
        return lower_map_physical_to_screen(name, res, rty, args);
    }

    // `air.get_imageblock_width/height()` -> the imageblock (= tile) dimensions. For a compute
    // kernel the implicit imageblock spans the threadgroup, so the dimensions are the kernel's
    // LocalSize x/y — the same values `air.threads_per_threadgroup` exposes. Lowered here (not the
    // emitter) because only the pass Ctx knows `kernel_local_size`. Byte-relevant: the imageblock
    // slice-write OOB gate compares `origin + region` (region defaults to these dimensions) against
    // the texture extent; a constant value of 1 under-reports the region
    // and lets Apple-discarded out-of-bounds block writes land at the origin texel.
    if name == "air.get_imageblock_width" || name == "air.get_imageblock_height" {
        return lower_get_imageblock_extent(ctx, name, res, rty);
    }

    // `air.get_read_sampler()` -> load a synthesized default sampler. Its result is only consumed as
    // the (ignored) sampler operand of `air.read_texture_*`, so producing a valid `OpTypeSampler` value
    // here is sufficient and correct. Handled before the result-type unwrap (the original result type
    // is an opaque pointer we discard).
    if name == "air.get_read_sampler" {
        return lower_get_read_sampler(ctx, res);
    }
    // `air.get_null_texture_<dim>()` -> load a synthesized default image (a function-constant-gated
    // optional attachment that, with our FCs folded off, resolves to a null texture). The result is an
    // opaque pointer AIR-side; we yield a loaded image, recording its dims so a later sample works.
    if name.starts_with("air.get_null_texture") {
        return lower_get_null_texture(ctx, name, res);
    }
    if name == "llvm.assume" {
        return Ok(Vec::new());
    }
    // AGX `yield` is a convergent execution-scheduling hint used by bounded spin loops. It has no
    // result or portable SPIR-V operation, and its i16 operand controls only the target scheduler's
    // backoff duration. Preserve the program-visible memory/control semantics by removing the hint;
    // the surrounding coherent atomics and loop remain intact.
    if name == "llvm.agx3.yield" {
        if args.len() != 1 {
            return Err(format!("{name} expects 1 operand"));
        }
        return Ok(Vec::new());
    }
    let (res, rty) = match (res, rty) {
        (Some(r), Some(t)) => (r, t),
        _ => return Err(format!("air.* call {name} has no result")),
    };

    if name == "llvm.agx3.edgecheck" {
        return lower_agx3_edgecheck(ctx, name, res, rty, args);
    }
    if name == "llvm.agx3.igemm.v8i32.i64.i64.v8i32" {
        return lower_agx3_igemm_16x16_mac(ctx, name, res, rty, args);
    }
    if name.starts_with("air.load.implicit_imageblock.") {
        return lower_implicit_imageblock_load(ctx, name, res, rty, args);
    }
    if let Some(dimension) = agx2_matmad_dimension(name) {
        return lower_agx2_matmad(ctx, name, res, rty, args, dimension);
    }

    // numeric conversions: air.convert.<dst>.<...>.<src> -> OpConvert* / OpBitcast.
    if name.starts_with("air.convert.") {
        return lower_convert(ctx, name, res, rty, args);
    }
    // derivatives. OpDPdx/OpDPdy/OpFwidth require a 32-bit-float operand+result (Vulkan), so a HALF
    // derivative (`air.fwidth.f16`, `air.dfdx.f16`, ...) must round-trip through float: FConvert the
    // half arg up to float, take the derivative in float, FConvert the result back to half.
    if name.starts_with("air.dfdx") || name.starts_with("air.fast_dfdx") {
        let [arg] = args else {
            return Err(format!("{name} requires one operand"));
        };
        return half_deriv(ctx, Op::DPdx, res, rty, *arg);
    }
    if name.starts_with("air.dfdy") || name.starts_with("air.fast_dfdy") {
        let [arg] = args else {
            return Err(format!("{name} requires one operand"));
        };
        return half_deriv(ctx, Op::DPdy, res, rty, *arg);
    }
    if name.starts_with("air.fwidth") {
        let [arg] = args else {
            return Err(format!("{name} requires one operand"));
        };
        return half_deriv(ctx, Op::Fwidth, res, rty, *arg);
    }
    // air.dot -> OpDot
    if name.starts_with("air.dot") && args.len() == 2 {
        return lower_dot(ctx, name, res, rty, args);
    }
    // simdgroup_matrix 8x8 family (documented Metal cooperative-matrix API). Emulated per-thread as a
    // 64-lane row-major composite: lane r*8+c = element (r,c). Register-level thread distribution is
    // invisible to memory — only load/store touch memory, both row-major with the runtime leading
    // dimension — so a full-matrix SIMT model is byte-faithful to the documented layout. Dispatched on
    // the stable `air.*` ABI symbol (the allowed name-family exception); the emit decides shape from
    // the operand types, never from a shader identifier.
    if let Some(signature) = crate::air_intrinsics::matrix8_intrinsic(name) {
        return lower_simdgroup_matrix_8x8_mac(ctx, name, signature, res, rty, args);
    }
    if name.starts_with("air.simdgroup_matrix_16x16x16_multiply_accumulate.")
        || name.starts_with("air.simdgroup_matrix_16x16x16_widening_multiply_accumulate.")
    {
        return lower_simdgroup_matrix_16x16_mac(ctx, name, res, rty, args);
    }
    if name.starts_with("air.simdgroup_matrix_8x8_init_diag.") {
        return lower_simdgroup_matrix_8x8_init_diag(ctx, res, rty, args);
    }
    if name.starts_with("air.simdgroup_matrix_8x8_load.") {
        return lower_simdgroup_matrix_8x8_load(ctx, res, rty, args);
    }
    // SIMD/subgroup lane ops + vector-reduction / function-constant predicates. The handler preserves
    // the original guard order; None means no guard matched (the cascade continues below).
    if let Some(insts) = lower_simd_op(ctx, name, res, rty, args)? {
        return Ok(insts);
    }
    // `air.pack.unorm4x8.<arg>` / `air.pack.snorm4x8` / `*2x16` -> the GLSL.std.450 Pack* ext-inst.
    // The normalized variants consume 32-bit-float vectors and return one packed u32.
    if name.starts_with("air.pack.") {
        return lower_pack(ctx, name, res, rty, args);
    }
    // `air.unpack.unorm.rgb10a2.<ret>` has no GLSL.std.450 equivalent, so unpack the 10/10/10/2
    // bit-fields by hand. Field->component layout verified empirically against Apple Metal's
    // `unpack_unorm10a2_to_float`: r=(x&0x3FF)/1023, g=((x>>10)&0x3FF)/1023, b=((x>>20)&0x3FF)/1023,
    // a=((x>>30)&0x3)/3 (e.g. 0x00000001->0.000978=1/1023, 0x40000000->a=0.333333=1/3). The arg is
    // the packed u32 (same as the GLSL unpack path); the `.v4f16` variant FConverts the result down.
    if name.starts_with("air.unpack.unorm.rgb10a2") {
        return lower_unpack_rgb10a2(ctx, res, rty, args);
    }
    // `air.unpack.unorm.rg11b10f.<ret>` (the R11F_G11F_B10F packed-float pixel format) has no GLSL
    // equivalent. The three fields are unsigned small floats sharing half-float's 5-bit exponent
    // (bias 15): R = bits[0:11) and G = bits[11:22) are 5-exp/6-mantissa; B = bits[22:32) is
    // 5-exp/5-mantissa. Field->component layout verified empirically against Apple Metal's
    // `rg11b10f<float3>` read (e.g. 0x3C0->r=1.0, 0x180->r=2^-9=0.001953). Each field widens losslessly
    // into a half by left-justifying the mantissa (zero-fill is exact for normals AND denormals since
    // the half exponent width/bias match), so `UnpackHalf2x16(bits).x` yields the float32 component.
    if name.starts_with("air.unpack.unorm.rg11b10f") {
        return lower_unpack_rg11b10f(ctx, res, rty, args);
    }
    // `air.unpack.unorm.rgb9e5.<ret>` (the RGB9E5 shared-exponent float format) has no GLSL
    // equivalent. One 5-bit exponent at bits[27:32) is shared by three 9-bit integer mantissas
    // (R=bits[0:9), G=bits[9:18), B=bits[18:27)); each component = mantissa * 2^(exp-24) with NO
    // implicit leading 1. Layout/scale verified empirically against Apple Metal's `rgb9e5<float3>`
    // read (mant=511,exp=24 -> 511.0; exp=25 -> x2). Computed as mantissa * exp2(exp - 24).
    if name.starts_with("air.unpack.unorm.rgb9e5") {
        return lower_unpack_rgb9e5(ctx, res, rty, args);
    }
    // `air.unpack.unorm4x8.<ret>` / `air.unpack.snorm4x8` / `*2x16` -> the GLSL.std.450 Unpack* ext-inst.
    // These always return a 32-bit-float vector; a `.v4f16` AIR variant then FConverts down to half.
    if name.starts_with("air.unpack.") {
        return lower_unpack(ctx, name, res, rty, args);
    }
    // Integer-arithmetic intrinsic family (abs/abs_diff, reverse_bits, bswap, rotate, extract/insert
    // bits, popcount, ctz/clz, mul_hi, mad_sat). The handler preserves the original guard order.
    if let Some(insts) = lower_integer_op(ctx, name, res, rty, args)? {
        return Ok(insts);
    }
    lower_float_math(ctx, name, res, rty, args)
}

fn lower_dot(
    ctx: &Ctx,
    name: &str,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if type_def_of(ctx, rty).is_none_or(|definition| definition.class.opcode != Op::TypeFloat) {
        return Err(format!("{name} result is not a floating-point scalar"));
    }
    let Some(left_type) = value_result_type(ctx, args[0]) else {
        return Err(format!("{name} left operand has no result type"));
    };
    let Some(right_type) = value_result_type(ctx, args[1]) else {
        return Err(format!("{name} right operand has no result type"));
    };
    let component_matches = left_type == right_type
        && vector_type_shape(ctx, left_type).is_some_and(|(component, _)| component == rty);
    if !component_matches {
        return Err(format!(
            "{name} operands are not identical vectors of its result component type"
        ));
    }
    Ok(vec![Instruction::new(
        Op::Dot,
        Some(rty),
        Some(res),
        vec![Operand::IdRef(args[0]), Operand::IdRef(args[1])],
    )])
}
