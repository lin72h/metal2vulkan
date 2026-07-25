//! `air.convert` lowering helpers owned by the AIR-call subsystem.

use super::*;

/// True if a type token (e.g. `i1`, `v3i1`) denotes a BOOL: an `i1` scalar/vector, with the `1` not
/// followed by another digit (so `i16`/`v2i16` don't match).
pub(in crate::passes) fn is_i1_type_token(tok: &str) -> bool {
    let t = tok.strip_prefix('v').map_or(tok, |rest| {
        // skip the leading vector count digits.
        rest.trim_start_matches(|c: char| c.is_ascii_digit())
    });
    t == "i1"
}

/// True if `rty`'s (element) type is an integer.
fn is_int_result(ctx: &Ctx, rty: Word) -> bool {
    let elem = element_type(ctx, rty);
    type_def_of(ctx, elem)
        .map(|d| d.class.opcode == Op::TypeInt)
        .unwrap_or(false)
}

/// Component count of value `v` (from its result type): vector -> N, scalar -> 1.
fn vector_len_of_value(ctx: &Ctx, v: Word) -> u32 {
    value_result_type(ctx, v)
        .map(|t| vector_len(ctx, t))
        .unwrap_or(1)
}

/// An integer constant of value `iv` shaped like `rty` (scalar -> the int const; vector -> a splat),
/// matching `rty`'s element width/signedness.
fn int_splat_or_scalar(ctx: &mut Ctx, rty: Word, iv: i64, n: u32) -> Word {
    let elem = element_type(ctx, rty);
    let s = ctx.const_int_of(elem, iv);
    if n <= 1 {
        s
    } else {
        splat(ctx, rty, s, n)
    }
}

/// air.convert.<dstkind><...>.<srckind>... -> OpConvert* on the result type. Kinds: f=float,
/// s=signed int, u=unsigned int. We read the first and last kind letters off the mangled name.
pub(super) fn lower_convert(
    ctx: &mut Ctx,
    name: &str,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    // `air.convert` names are `air.convert.<dstkind>.<dsttype>.<srckind>.<srctype>`. An `i1` token is a
    // BOOL; whether it is the DEST type or the SOURCE type decides the lowering direction.
    // e.g. air.convert.f.v3f32.u.v3i1 (i1 = src) / air.convert.u.i1.f.f32 (i1 = dst).
    let parts: Vec<&str> = name.trim_start_matches("air.convert.").split('.').collect();
    // The type tokens are at even-ish positions; precisely, dst type = parts[1], src type = last part.
    let dst_type = parts.get(1).copied().unwrap_or("");
    let src_type = parts.last().copied().unwrap_or("");

    // Bool SOURCE (`...u.v3i1`): a vector/scalar of i1 -> numeric via OpSelect of 1/0 (OpConvert*
    // rejects bool input). The result element type may be float OR int (`air.convert.s.i32.u.i1`).
    if is_i1_type_token(src_type) {
        let n = vector_len(ctx, rty);
        let elem_is_int = is_int_result(ctx, rty);
        let (one, zero) = if elem_is_int {
            (
                int_splat_or_scalar(ctx, rty, 1, n),
                int_splat_or_scalar(ctx, rty, 0, n),
            )
        } else {
            (
                splat_or_scalar(ctx, rty, 1.0, n),
                splat_or_scalar(ctx, rty, 0.0, n),
            )
        };
        return Ok(vec![Instruction::new(
            Op::Select,
            Some(rty),
            Some(res),
            vec![
                Operand::IdRef(args[0]),
                Operand::IdRef(one),
                Operand::IdRef(zero),
            ],
        )]);
    }
    // Bool DEST (`air.convert.u.i1.f.f32`): a numeric -> i1 (bool) via a `!= 0` comparison. The source
    // kind (the last single-letter token) picks float vs int compare. `rty` here is `%bool`.
    if is_i1_type_token(dst_type) {
        // src kind = the kind letter immediately before the src type.
        let src_kind = parts
            .get(parts.len().saturating_sub(2))
            .and_then(|p| p.chars().next())
            .unwrap_or('f');
        let n = vector_len_of_value(ctx, args[0]);
        if src_kind == 'f' {
            let zero = splat_or_scalar(ctx, value_result_type(ctx, args[0]).unwrap_or(rty), 0.0, n);
            return Ok(vec![Instruction::new(
                Op::FUnordNotEqual,
                Some(rty),
                Some(res),
                vec![Operand::IdRef(args[0]), Operand::IdRef(zero)],
            )]);
        }
        let src_ty = value_result_type(ctx, args[0]).unwrap_or(rty);
        let zero = int_splat_or_scalar(ctx, src_ty, 0, n);
        return Ok(vec![Instruction::new(
            Op::INotEqual,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(args[0]), Operand::IdRef(zero)],
        )]);
    }
    // find dst kind and src kind by scanning for single-letter f/s/u tokens.
    let kinds: Vec<char> = parts
        .iter()
        .filter(|p| p.len() == 1 && matches!(p.chars().next().unwrap(), 'f' | 's' | 'u'))
        .map(|p| p.chars().next().unwrap())
        .collect();
    let (dst, src) = match (kinds.first(), kinds.last()) {
        (Some(d), Some(s)) if kinds.len() >= 2 => (*d, *s),
        _ => return Err(format!("cannot parse convert kinds from {name}")),
    };
    // bfloat16 has no SPIR-V type, so the emitter models a bf16 value as its raw `OpTypeInt 16` bit
    // pattern (the top 16 bits of an f32). A convert whose source or dest is bf16 (`...f.bf16` /
    // `f.bf16...`) therefore can't go straight through OpConvert*: the int16-typed operand fails
    // "expected float input". Widen the bf16 bits to f32 at the source / narrow f32 to bf16 bits at
    // the dest, around the existing float<->int conversion. `bf16` is a stable AIR type token.
    if token_is_bf16(src_type) || token_is_bf16(dst_type) {
        return lower_convert_bf16(ctx, res, rty, args, dst_type, src_type, dst, src);
    }
    if (dst, src) == ('f', 's') {
        let mut out = Vec::new();
        let (signed, _) = bitcast_to_integer_signedness(ctx, &mut out, args[0], true)?;
        out.push(Instruction::new(
            Op::ConvertSToF,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(signed)],
        ));
        return Ok(out);
    }
    if (dst, src) == ('u', 's') || (dst, src) == ('s', 'u') {
        let mut out = Vec::new();
        let signed_source = src == 's';
        let (input, input_ty) =
            bitcast_to_integer_signedness(ctx, &mut out, args[0], signed_source)?;
        let op = if scalar_bit_width(ctx, input_ty) == scalar_bit_width(ctx, rty)
            && vector_len(ctx, input_ty) == vector_len(ctx, rty)
        {
            Op::Bitcast
        } else if signed_source {
            Op::SConvert
        } else {
            Op::UConvert
        };
        out.push(Instruction::new(
            op,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(input)],
        ));
        return Ok(out);
    }
    if matches!((dst, src), ('u' | 's', 'f')) {
        if let Some(out) = lower_float_to_narrow_int_convert(ctx, res, rty, args[0], dst)? {
            return Ok(out);
        }
    }
    let op = match (dst, src) {
        ('f', 'u') => Op::ConvertUToF,
        ('u', 'f') => Op::ConvertFToU,
        ('s', 'f') => Op::ConvertFToS,
        ('u', 'u') => Op::UConvert,
        ('s', 's') => Op::SConvert,
        // float<->float of differing widths (half<->float): a single OpFConvert handles both fpext and
        // fptrunc, scalar or vector (`air.convert.f.v4f32.f.v4f16`, `...v4f16.f.v4f32`, `v2f32.f.v2f16`).
        ('f', 'f') => Op::FConvert,
        _ => return Err(format!("unhandled convert kinds {dst}->{src} in {name}")),
    };
    Ok(vec![Instruction::new(
        op,
        Some(rty),
        Some(res),
        vec![Operand::IdRef(args[0])],
    )])
}

fn lower_float_to_narrow_int_convert(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    value: Word,
    dst: char,
) -> Result<Option<Vec<Instruction>>, String> {
    let dst_width = scalar_bit_width(ctx, rty);
    if dst_width == 0 || dst_width >= 32 {
        return Ok(None);
    }
    let src_ty = value_result_type(ctx, value).ok_or("air.convert float source has no type")?;
    let src_elem = element_type(ctx, src_ty);
    let Some(src_def) = type_def_of(ctx, src_elem) else {
        return Err("air.convert float source type is undefined".into());
    };
    if src_def.class.opcode != Op::TypeFloat {
        return Ok(None);
    }

    let (lo, hi) = float_to_int_bounds(dst, dst_width)?;
    let n = vector_len(ctx, src_ty);
    let lo = splat_or_scalar(ctx, src_ty, lo, n);
    let hi = splat_or_scalar(ctx, src_ty, hi, n);
    let clamped = ctx.module.fresh_id();
    let ext = ctx.glsl();
    let op = if dst == 'u' {
        Op::ConvertFToU
    } else {
        Op::ConvertFToS
    };
    Ok(Some(vec![
        Instruction::new(
            Op::ExtInst,
            Some(src_ty),
            Some(clamped),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::FClamp as u32),
                Operand::IdRef(value),
                Operand::IdRef(lo),
                Operand::IdRef(hi),
            ],
        ),
        Instruction::new(op, Some(rty), Some(res), vec![Operand::IdRef(clamped)]),
    ]))
}

fn float_to_int_bounds(dst: char, width: u32) -> Result<(f32, f32), String> {
    if width == 0 || width >= 32 {
        return Err(format!("unsupported narrow float convert width {width}"));
    }
    match dst {
        'u' => Ok((0.0, ((1u64 << width) - 1) as f32)),
        's' => {
            let min = -(1i64 << (width - 1)) as f32;
            let max = ((1i64 << (width - 1)) - 1) as f32;
            Ok((min, max))
        }
        _ => Err(format!("unsupported float-to-int convert kind {dst}")),
    }
}

fn bitcast_to_integer_signedness(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    value: Word,
    signed: bool,
) -> Result<(Word, Word), String> {
    let ty = value_result_type(ctx, value).ok_or("air.convert integer source has no type")?;
    let target_ty = integer_type_like(ctx, ty, signed)?;
    if target_ty == ty {
        return Ok((value, ty));
    }
    let cast = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Bitcast,
        Some(target_ty),
        Some(cast),
        vec![Operand::IdRef(value)],
    ));
    Ok((cast, target_ty))
}

fn integer_type_like(ctx: &mut Ctx, ty: Word, signed: bool) -> Result<Word, String> {
    let def = type_def_of(ctx, ty).ok_or("air.convert integer source type is undefined")?;
    match def.class.opcode {
        Op::TypeInt => {
            let bits = literal_u32(def.operands.first())
                .ok_or("air.convert integer source int missing width")?;
            let current_signed = literal_u32(def.operands.get(1))
                .ok_or("air.convert integer source int missing signedness")?;
            if current_signed == u32::from(signed) {
                Ok(ty)
            } else {
                Ok(integer_type(ctx, bits, signed))
            }
        }
        Op::TypeVector => {
            let elem = id_ref(def.operands.first())
                .ok_or("air.convert integer source vector missing element type")?;
            let lanes = literal_u32(def.operands.get(1))
                .ok_or("air.convert integer source vector missing length")?;
            let target_elem = integer_type_like(ctx, elem, signed)?;
            if target_elem == elem {
                Ok(ty)
            } else {
                Ok(vector_type(ctx, target_elem, lanes))
            }
        }
        _ => Err("air.convert source is not an integer scalar/vector".into()),
    }
}

fn integer_type(ctx: &mut Ctx, bits: u32, signed: bool) -> Word {
    let key = SynthCacheKey::IntType { bits, signed };
    if let Some(&id) = ctx.synth_cache.get(&key) {
        return id;
    }
    let signedness = u32::from(signed);
    for inst in ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
    {
        if inst.class.opcode == Op::TypeInt
            && inst.operands.first() == Some(&Operand::LiteralBit32(bits))
            && inst.operands.get(1) == Some(&Operand::LiteralBit32(signedness))
        {
            if let Some(rid) = inst.result_id {
                ctx.synth_cache.insert(key, rid);
                return rid;
            }
        }
    }
    let id = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::TypeInt,
        None,
        Some(id),
        vec![
            Operand::LiteralBit32(bits),
            Operand::LiteralBit32(signedness),
        ],
    ));
    ctx.synth_cache.insert(key, id);
    id
}

fn vector_type(ctx: &mut Ctx, elem: Word, lanes: u32) -> Word {
    let key = SynthCacheKey::VecType { elem, lanes };
    if let Some(&id) = ctx.synth_cache.get(&key) {
        return id;
    }
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
            if let Some(rid) = inst.result_id {
                ctx.synth_cache.insert(key, rid);
                return rid;
            }
        }
    }
    let id = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::TypeVector,
        None,
        Some(id),
        vec![Operand::IdRef(elem), Operand::LiteralBit32(lanes)],
    ));
    ctx.synth_cache.insert(key, id);
    id
}

fn id_ref(op: Option<&Operand>) -> Option<Word> {
    match op {
        Some(Operand::IdRef(id)) => Some(*id),
        _ => None,
    }
}

fn literal_u32(op: Option<&Operand>) -> Option<u32> {
    match op {
        Some(Operand::LiteralBit32(v)) => Some(*v),
        _ => None,
    }
}

/// True if a convert type token denotes a bf16 (`bf16`, `v2bf16`, `v4bf16`, ...). bf16 is modeled as
/// `OpTypeInt 16` storage, so it needs widen/narrow around float conversions.
fn token_is_bf16(tok: &str) -> bool {
    tok.contains("bf16")
}

/// Lane count encoded in a convert type token: `v4f32` -> 4, `bf16` / `f32` -> 1.
fn token_lanes(tok: &str) -> u32 {
    tok.strip_prefix('v')
        .map(|rest| {
            rest.chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .unwrap_or(1)
        })
        .unwrap_or(1)
}

/// f32 (scalar or vN) type id for a lane count.
fn ty_f32_shaped(ctx: &mut Ctx, n: u32) -> Word {
    if n > 1 {
        ctx.ty_vecf(n)
    } else {
        ctx.ty_float()
    }
}

/// u32 (scalar or vN) type id for a lane count.
fn ty_u32_shaped(ctx: &mut Ctx, n: u32) -> Word {
    if n > 1 {
        ctx.ty_vec_uint(n)
    } else {
        ctx.ty_uint()
    }
}

fn ty_bool_shaped(ctx: &mut Ctx, n: u32) -> Word {
    if n > 1 {
        ctx.ty_vec_bool(n)
    } else {
        ctx.ty_bool()
    }
}

/// A shift amount of 16, shaped to match an `n`-lane operand (vector shifts need a vector amount).
fn shift_amount_16(ctx: &mut Ctx, n: u32) -> Word {
    let s = ctx.const_uint(16);
    if n > 1 {
        let vty = ctx.ty_vec_uint(n);
        splat(ctx, vty, s, n)
    } else {
        s
    }
}

fn shaped_u32_const(ctx: &mut Ctx, n: u32, value: u32) -> Word {
    let scalar = ctx.const_uint(value);
    if n > 1 {
        let vty = ctx.ty_vec_uint(n);
        splat(ctx, vty, scalar, n)
    } else {
        scalar
    }
}

/// Widen a bf16 bit pattern (modeled as int16, scalar or vN) to f32: bf16 is the top 16 bits of an
/// f32, so `f32 = bitcast<float>(zext_u32(bits) << 16)`.
fn widen_bf16_to_f32(ctx: &mut Ctx, out: &mut Vec<Instruction>, bits: Word, n: u32) -> Word {
    let u32_ty = ty_u32_shaped(ctx, n);
    let f32_ty = ty_f32_shaped(ctx, n);
    let widened = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::UConvert,
        Some(u32_ty),
        Some(widened),
        vec![Operand::IdRef(bits)],
    ));
    let shamt = shift_amount_16(ctx, n);
    let shifted = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ShiftLeftLogical,
        Some(u32_ty),
        Some(shifted),
        vec![Operand::IdRef(widened), Operand::IdRef(shamt)],
    ));
    let f = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Bitcast,
        Some(f32_ty),
        Some(f),
        vec![Operand::IdRef(shifted)],
    ));
    f
}

/// Narrow an f32 (scalar or vN) to a bf16 bit pattern, writing `res`/`rty` (the int16 storage type).
/// LLVM `fptrunc float to bfloat` rounds to nearest-even, so add the bf16 rounding bias before
/// taking the top 16 bits.
fn narrow_f32_to_bf16(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    f32val: Word,
    n: u32,
    rty: Word,
    res: Word,
) {
    let u32_ty = ty_u32_shaped(ctx, n);
    let as_u32 = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Bitcast,
        Some(u32_ty),
        Some(as_u32),
        vec![Operand::IdRef(f32val)],
    ));
    let shamt = shift_amount_16(ctx, n);
    let high = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ShiftRightLogical,
        Some(u32_ty),
        Some(high),
        vec![Operand::IdRef(as_u32), Operand::IdRef(shamt)],
    ));
    let one = shaped_u32_const(ctx, n, 1);
    let lsb = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::BitwiseAnd,
        Some(u32_ty),
        Some(lsb),
        vec![Operand::IdRef(high), Operand::IdRef(one)],
    ));
    let bias_base = shaped_u32_const(ctx, n, 0x7fff);
    let bias = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::IAdd,
        Some(u32_ty),
        Some(bias),
        vec![Operand::IdRef(bias_base), Operand::IdRef(lsb)],
    ));
    let rounded = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::IAdd,
        Some(u32_ty),
        Some(rounded),
        vec![Operand::IdRef(as_u32), Operand::IdRef(bias)],
    ));
    let shifted = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ShiftRightLogical,
        Some(u32_ty),
        Some(shifted),
        vec![Operand::IdRef(rounded), Operand::IdRef(shamt)],
    ));
    let shifted = select_canonical_bfloat_nan_bits(ctx, out, as_u32, shifted, n);
    out.push(Instruction::new(
        Op::UConvert,
        Some(rty),
        Some(res),
        vec![Operand::IdRef(shifted)],
    ));
}

fn select_canonical_bfloat_nan_bits(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    bits: Word,
    narrowed: Word,
    n: u32,
) -> Word {
    let u32_ty = ty_u32_shaped(ctx, n);
    let bool_ty = ty_bool_shaped(ctx, n);
    let exp_mask = shaped_u32_const(ctx, n, 0x7f80_0000);
    let mant_mask = shaped_u32_const(ctx, n, 0x007f_ffff);
    let zero = shaped_u32_const(ctx, n, 0);

    let exp_bits = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::BitwiseAnd,
        Some(u32_ty),
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
        Some(u32_ty),
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

    let canonical_nan = shaped_u32_const(ctx, n, 0x7fc0);
    let selected = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Select,
        Some(u32_ty),
        Some(selected),
        vec![
            Operand::IdRef(is_nan),
            Operand::IdRef(canonical_nan),
            Operand::IdRef(narrowed),
        ],
    ));
    selected
}

/// Lower an `air.convert` whose source and/or dest is bf16. The conversion is split around an f32
/// intermediate: the source is brought to f32 (widening bf16 bits, or an ordinary int/float->f32
/// convert), then f32 is taken to the dest (narrowing to bf16 bits, or an ordinary f32->int/float
/// convert). This keeps the bf16 leg honest — a real f32 value flows through the arithmetic convert.
#[allow(clippy::too_many_arguments)]
fn lower_convert_bf16(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    args: &[Word],
    dst_type: &str,
    src_type: &str,
    dst_kind: char,
    src_kind: char,
) -> Result<Vec<Instruction>, String> {
    let arg = *args
        .first()
        .ok_or("air.convert bf16: missing source operand")?;
    let src_is_bf16 = token_is_bf16(src_type);
    let dst_is_bf16 = token_is_bf16(dst_type);
    let n = if src_is_bf16 {
        token_lanes(src_type)
    } else {
        token_lanes(dst_type)
    };
    let mut out = Vec::new();

    // 1. Bring the source to f32.
    let f32_ty = ty_f32_shaped(ctx, n);
    let f32val = if src_is_bf16 {
        widen_bf16_to_f32(ctx, &mut out, arg, n)
    } else {
        let src_ty = value_result_type(ctx, arg).unwrap_or(rty);
        match src_kind {
            // f16/f32 source: widen/copy to f32 (FConvert rejects equal widths, so copy when already f32).
            'f' => {
                if scalar_bit_width(ctx, src_ty) == 32 {
                    arg
                } else {
                    let id = ctx.module.fresh_id();
                    out.push(Instruction::new(
                        Op::FConvert,
                        Some(f32_ty),
                        Some(id),
                        vec![Operand::IdRef(arg)],
                    ));
                    id
                }
            }
            's' => {
                let id = ctx.module.fresh_id();
                out.push(Instruction::new(
                    Op::ConvertSToF,
                    Some(f32_ty),
                    Some(id),
                    vec![Operand::IdRef(arg)],
                ));
                id
            }
            _ => {
                let id = ctx.module.fresh_id();
                out.push(Instruction::new(
                    Op::ConvertUToF,
                    Some(f32_ty),
                    Some(id),
                    vec![Operand::IdRef(arg)],
                ));
                id
            }
        }
    };

    // 2. Take the f32 intermediate to the dest.
    if dst_is_bf16 {
        narrow_f32_to_bf16(ctx, &mut out, f32val, n, rty, res);
    } else {
        let op = match dst_kind {
            's' => Op::ConvertFToS,
            'u' => Op::ConvertFToU,
            // float dest: f16 narrows via FConvert; f32 is identity (CopyObject, FConvert rejects it).
            _ => {
                if scalar_bit_width(ctx, rty) == 32 {
                    Op::CopyObject
                } else {
                    Op::FConvert
                }
            }
        };
        out.push(Instruction::new(
            op,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(f32val)],
        ));
    }
    Ok(out)
}
