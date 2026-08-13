//! Bfloat and GLSL extended-instruction AIR call lowering.

use super::*;

pub(in crate::passes) fn is_llvm_bfloat_fmuladd(name: &str) -> bool {
    name.starts_with("llvm.fmuladd.") && name.contains("bf16")
}

pub(in crate::passes) fn lower_bfloat_fmuladd(
    ctx: &mut Ctx,
    name: &str,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if args.len() != 3 {
        return Err(format!("{name} expects 3 operands"));
    }
    if !is_u16_scalar(ctx, rty) {
        return Err(format!(
            "{name} currently expects a scalar u16 bfloat storage result"
        ));
    }
    for arg in args {
        let arg_ty = value_result_type(ctx, *arg)
            .ok_or_else(|| format!("{name} has an operand with no result type"))?;
        if !is_u16_scalar(ctx, arg_ty) {
            return Err(format!(
                "{name} currently expects scalar u16 bfloat storage operands"
            ));
        }
    }

    let mut out = vec![];
    let a = bfloat_to_f32(ctx, &mut out, args[0]);
    let b = bfloat_to_f32(ctx, &mut out, args[1]);
    let c = bfloat_to_f32(ctx, &mut out, args[2]);
    let fma = ctx.module.fresh_id();
    let ext = ctx.glsl();
    let float = ctx.ty_float();
    out.push(Instruction::new(
        Op::ExtInst,
        Some(float),
        Some(fma),
        vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(GLSLstd450::Fma as u32),
            Operand::IdRef(a),
            Operand::IdRef(b),
            Operand::IdRef(c),
        ],
    ));
    f32_to_bfloat(ctx, &mut out, fma, rty, res);
    Ok(out)
}

pub(in crate::passes) fn bfloat_to_f32(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    value: Word,
) -> Word {
    let uint = ctx.ty_uint();
    let float = ctx.ty_float();
    let widened = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::UConvert,
        Some(uint),
        Some(widened),
        vec![Operand::IdRef(value)],
    ));
    let shift = ctx.const_uint(16);
    let shifted = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ShiftLeftLogical,
        Some(uint),
        Some(shifted),
        vec![Operand::IdRef(widened), Operand::IdRef(shift)],
    ));
    let result = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Bitcast,
        Some(float),
        Some(result),
        vec![Operand::IdRef(shifted)],
    ));
    result
}

pub(in crate::passes) fn f32_to_bfloat(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    value: Word,
    result_type: Word,
    result: Word,
) {
    let uint = ctx.ty_uint();
    let bits = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Bitcast,
        Some(uint),
        Some(bits),
        vec![Operand::IdRef(value)],
    ));
    let shift = ctx.const_uint(16);
    let high = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ShiftRightLogical,
        Some(uint),
        Some(high),
        vec![Operand::IdRef(bits), Operand::IdRef(shift)],
    ));
    let one = ctx.const_uint(1);
    let lsb = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(lsb),
        vec![Operand::IdRef(high), Operand::IdRef(one)],
    ));
    let bias_base = ctx.const_uint(0x7fff);
    let bias = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(bias),
        vec![Operand::IdRef(bias_base), Operand::IdRef(lsb)],
    ));
    let rounded = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(rounded),
        vec![Operand::IdRef(bits), Operand::IdRef(bias)],
    ));
    let shifted = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ShiftRightLogical,
        Some(uint),
        Some(shifted),
        vec![Operand::IdRef(rounded), Operand::IdRef(shift)],
    ));
    let shifted = select_canonical_bfloat_nan_bits(ctx, out, bits, shifted);
    out.push(Instruction::new(
        Op::UConvert,
        Some(result_type),
        Some(result),
        vec![Operand::IdRef(shifted)],
    ));
}

fn select_canonical_bfloat_nan_bits(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    bits: Word,
    narrowed: Word,
) -> Word {
    let uint = ctx.ty_uint();
    let bool_ty = ctx.ty_bool();
    let exp_mask = ctx.const_uint(0x7f80_0000);
    let mant_mask = ctx.const_uint(0x007f_ffff);
    let zero = ctx.const_uint(0);

    let exp_bits = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(exp_bits),
        vec![Operand::IdRef(bits), Operand::IdRef(exp_mask)],
    ));
    let exp_all_ones = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::IEqual,
        Some(bool_ty),
        Some(exp_all_ones),
        vec![Operand::IdRef(exp_bits), Operand::IdRef(exp_mask)],
    ));

    let mant_bits = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(mant_bits),
        vec![Operand::IdRef(bits), Operand::IdRef(mant_mask)],
    ));
    let mant_nonzero = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::INotEqual,
        Some(bool_ty),
        Some(mant_nonzero),
        vec![Operand::IdRef(mant_bits), Operand::IdRef(zero)],
    ));

    let is_nan = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::LogicalAnd,
        Some(bool_ty),
        Some(is_nan),
        vec![Operand::IdRef(exp_all_ones), Operand::IdRef(mant_nonzero)],
    ));

    let canonical_nan = ctx.const_uint(0x7fc0);
    let selected = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Select,
        Some(uint),
        Some(selected),
        vec![
            Operand::IdRef(is_nan),
            Operand::IdRef(canonical_nan),
            Operand::IdRef(narrowed),
        ],
    ));
    selected
}

/// GLSL.std.450 opcode for the simple unary/binary/ternary math intrinsics.
pub(in crate::passes) fn glsl_extinst(name: &str) -> Option<GLSLstd450> {
    use GLSLstd450::*;
    // atan2 is binary; keep it early for clarity. The ExtInst caller forwards all args.
    Some(if is_air_math(name, "atan2") {
        Atan2
    } else if is_air_math(name, "asin") {
        Asin
    } else if is_air_math(name, "acos") {
        Acos
    } else if is_air_math(name, "atan") {
        Atan
    } else if name.starts_with("llvm.maxnum.")
        || is_air_math(name, "fmax")
        || is_air_math(name, "max")
    {
        FMax
    } else if name.starts_with("llvm.minnum.")
        || is_air_math(name, "fmin")
        || is_air_math(name, "min")
    {
        FMin
    } else if is_air_math(name, "sqrt") {
        Sqrt
    } else if is_air_math(name, "rsqrt") {
        InverseSqrt
    } else if name.starts_with("llvm.fabs.") || is_air_math(name, "fabs") {
        FAbs
    } else if is_air_math(name, "pow") || is_air_math(name, "powr") {
        Pow
    } else if is_air_math(name, "sign") {
        FSign
    } else if is_air_math(name, "mix") {
        FMix
    } else if is_air_math(name, "floor") {
        Floor
    } else if is_air_math(name, "ceil") {
        Ceil
    } else if is_air_math(name, "round") {
        Round
    } else if is_air_math(name, "rint") {
        RoundEven
    } else if is_air_math(name, "trunc") {
        Trunc
    } else if is_air_math(name, "tan") {
        Tan
    } else if is_air_math(name, "sinh") {
        Sinh
    } else if is_air_math(name, "cosh") {
        Cosh
    } else if is_air_math(name, "tanh") {
        Tanh
    } else if is_air_math(name, "asinh") {
        Asinh
    } else if is_air_math(name, "acosh") {
        Acosh
    } else if is_air_math(name, "atanh") {
        Atanh
    } else if is_air_math(name, "sin") {
        Sin
    } else if is_air_math(name, "cos") {
        Cos
    } else if is_air_math(name, "exp2") {
        Exp2
    } else if is_air_math(name, "exp") {
        Exp
    } else if is_air_math(name, "log2") {
        Log2
    } else if is_air_math(name, "log") {
        Log
    } else if is_air_math(name, "fract") {
        Fract
    } else if is_air_math(name, "fma") || name.starts_with("llvm.fmuladd.") {
        Fma
    } else if is_air_math(name, "clamp") {
        FClamp
    } else {
        return None;
    })
}

pub(in crate::passes) fn is_air_math(name: &str, stem: &str) -> bool {
    let Some(rest) = name.strip_prefix("air.") else {
        return false;
    };
    let rest = rest.strip_prefix("fast_").unwrap_or(rest);
    rest.starts_with(stem) && rest.as_bytes().get(stem.len()) == Some(&b'.')
}

/// GLSL.std.450 extended-instruction numbers (subset we emit).
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
#[derive(Clone, Copy)]
pub(in crate::passes) enum GLSLstd450 {
    Round = 1,
    RoundEven = 2,
    SAbs = 5,
    FAbs = 4,
    FSign = 6,
    Floor = 8,
    Ceil = 9,
    Fract = 10,
    Sin = 13,
    Cos = 14,
    Tan = 15,
    Asin = 16,
    Acos = 17,
    Atan = 18,
    Sinh = 19,
    Cosh = 20,
    Tanh = 21,
    Asinh = 22,
    Acosh = 23,
    Atanh = 24,
    Atan2 = 25,
    Pow = 26,
    Exp = 27,
    Log = 28,
    Exp2 = 29,
    Log2 = 30,
    Sqrt = 31,
    InverseSqrt = 32,
    FMin = 37,
    UMin = 38,
    SMin = 39,
    FMax = 40,
    UMax = 41,
    SMax = 42,
    FClamp = 43,
    UClamp = 44,
    SClamp = 45,
    FMix = 46,
    Trunc = 3,
    Fma = 50,
    Ldexp = 53,
    PackSnorm4x8 = 54,
    PackUnorm4x8 = 55,
    PackSnorm2x16 = 56,
    PackUnorm2x16 = 57,
    PackHalf2x16 = 58,
    UnpackSnorm2x16 = 60,
    UnpackUnorm2x16 = 61,
    UnpackHalf2x16 = 62,
    UnpackSnorm4x8 = 63,
    UnpackUnorm4x8 = 64,
}
