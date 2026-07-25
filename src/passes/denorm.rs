//! Emulate AIR's denorm flush-to-zero mode without relying on optional Vulkan device support.

use super::*;
use crate::passes::access::const_composite_splat;

pub(in crate::passes) fn emulate_f32_denorm_flush_to_zero(ctx: &mut Ctx, entry_idx: usize) {
    if !ctx.denorm_flush_to_zero_f32 {
        return;
    }

    let mut block_idx = 0;
    while block_idx < ctx.module.functions[entry_idx].blocks.len() {
        let mut inst_idx = 0;
        while inst_idx
            < ctx.module.functions[entry_idx].blocks[block_idx]
                .instructions
                .len()
        {
            let snapshot =
                ctx.module.functions[entry_idx].blocks[block_idx].instructions[inst_idx].clone();
            if !denorm_sensitive_op(&snapshot) {
                inst_idx += 1;
                continue;
            }

            let mut prefix = Vec::new();
            let mut operands = snapshot.operands.clone();
            let first_float_operand = first_denorm_operand_index(&snapshot);
            for operand in operands.iter_mut().skip(first_float_operand) {
                let Operand::IdRef(id) = *operand else {
                    continue;
                };
                let Some(ty) = value_result_type(ctx, id) else {
                    continue;
                };
                if float_shape(ctx, ty).is_none() {
                    continue;
                }
                let flushed = build_float_ftz(ctx, id, ty, None, &mut prefix);
                let adjusted = if snapshot.class.opcode == Op::FConvert
                    && fconvert_narrows_to_f16(ctx, ty, snapshot.result_type)
                {
                    build_float_saturate_to_f16_range(ctx, flushed, ty, &mut prefix)
                } else {
                    flushed
                };
                *operand = Operand::IdRef(adjusted);
            }

            let prefix_len = prefix.len();
            if !prefix.is_empty() {
                ctx.module.functions[entry_idx].blocks[block_idx]
                    .instructions
                    .splice(inst_idx..inst_idx, prefix);
                inst_idx += prefix_len;
            }
            let Some(result) = snapshot.result_id else {
                ctx.module.functions[entry_idx].blocks[block_idx].instructions[inst_idx].operands =
                    operands;
                inst_idx += 1;
                continue;
            };
            let Some(result_ty) = snapshot.result_type else {
                ctx.module.functions[entry_idx].blocks[block_idx].instructions[inst_idx].operands =
                    operands;
                inst_idx += 1;
                continue;
            };
            if float_shape(ctx, result_ty).is_none() {
                ctx.module.functions[entry_idx].blocks[block_idx].instructions[inst_idx].operands =
                    operands;
                inst_idx += 1;
                continue;
            }

            let raw = ctx.module.fresh_id();
            ctx.module.functions[entry_idx].blocks[block_idx].instructions[inst_idx].operands =
                operands.clone();
            ctx.module.functions[entry_idx].blocks[block_idx].instructions[inst_idx].result_id =
                Some(raw);
            let mut suffix = Vec::new();
            let value = if snapshot.class.opcode == Op::FDiv {
                build_fdiv_zero_zero_nan(ctx, raw, result_ty, &operands, &mut suffix).unwrap_or(raw)
            } else {
                raw
            };
            let flushed = build_float_ftz(ctx, value, result_ty, Some(result), &mut suffix);
            debug_assert_eq!(flushed, result);
            let suffix_len = suffix.len();
            if !suffix.is_empty() {
                ctx.module.functions[entry_idx].blocks[block_idx]
                    .instructions
                    .splice((inst_idx + 1)..(inst_idx + 1), suffix);
            }
            inst_idx += 1 + suffix_len;
        }
        block_idx += 1;
    }
}

fn denorm_sensitive_op(inst: &Instruction) -> bool {
    matches!(
        inst.class.opcode,
        Op::FNegate
            | Op::ConvertFToU
            | Op::ConvertFToS
            | Op::FConvert
            | Op::QuantizeToF16
            | Op::FAdd
            | Op::FSub
            | Op::FMul
            | Op::FDiv
            | Op::FRem
            | Op::FMod
            | Op::VectorTimesScalar
            | Op::MatrixTimesScalar
            | Op::VectorTimesMatrix
            | Op::MatrixTimesVector
            | Op::MatrixTimesMatrix
            | Op::Dot
            | Op::FOrdEqual
            | Op::FUnordEqual
            | Op::FOrdNotEqual
            | Op::FUnordNotEqual
            | Op::FOrdLessThan
            | Op::FUnordLessThan
            | Op::FOrdGreaterThan
            | Op::FUnordGreaterThan
            | Op::FOrdLessThanEqual
            | Op::FUnordLessThanEqual
            | Op::FOrdGreaterThanEqual
            | Op::FUnordGreaterThanEqual
            | Op::ExtInst
    )
}

fn first_denorm_operand_index(inst: &Instruction) -> usize {
    if inst.class.opcode == Op::ExtInst {
        2
    } else {
        0
    }
}

fn fconvert_narrows_to_f16(ctx: &Ctx, src_ty: Word, dst_ty: Option<Word>) -> bool {
    let Some(src) = float_shape(ctx, src_ty) else {
        return false;
    };
    let Some(dst_ty) = dst_ty else {
        return false;
    };
    let Some(dst) = float_shape(ctx, dst_ty) else {
        return false;
    };
    src.width > 16 && dst.width == 16 && src.lanes == dst.lanes
}

fn build_float_saturate_to_f16_range(
    ctx: &mut Ctx,
    value: Word,
    ty: Word,
    out: &mut Vec<Instruction>,
) -> Word {
    let Some(shape) = float_shape(ctx, ty) else {
        return value;
    };
    if shape.width <= 16 {
        return value;
    }

    let bool_ty = shape_bool_type(ctx, shape);
    let max = shaped_float_const(ctx, ty, shape, 65_504.0);
    let above_max = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::FOrdGreaterThan,
        Some(bool_ty),
        Some(above_max),
        vec![Operand::IdRef(value), Operand::IdRef(max)],
    ));
    let high_clamped = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Select,
        Some(ty),
        Some(high_clamped),
        vec![
            Operand::IdRef(above_max),
            Operand::IdRef(max),
            Operand::IdRef(value),
        ],
    ));

    let min = shaped_float_const(ctx, ty, shape, -65_504.0);
    let below_min = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::FOrdLessThan,
        Some(bool_ty),
        Some(below_min),
        vec![Operand::IdRef(high_clamped), Operand::IdRef(min)],
    ));
    let clamped = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Select,
        Some(ty),
        Some(clamped),
        vec![
            Operand::IdRef(below_min),
            Operand::IdRef(min),
            Operand::IdRef(high_clamped),
        ],
    ));
    clamped
}

fn build_float_ftz(
    ctx: &mut Ctx,
    value: Word,
    ty: Word,
    result_id: Option<Word>,
    out: &mut Vec<Instruction>,
) -> Word {
    let Some(shape) = float_shape(ctx, ty) else {
        return value;
    };
    let int_ty = shape_int_type(ctx, shape);
    let bool_ty = shape_bool_type(ctx, shape);
    let bits = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Bitcast,
        Some(int_ty),
        Some(bits),
        vec![Operand::IdRef(value)],
    ));

    let mag = ctx.module.fresh_id();
    let magnitude_mask = shaped_int_const(ctx, int_ty, shape, magnitude_mask(shape.width));
    out.push(Instruction::new(
        Op::BitwiseAnd,
        Some(int_ty),
        Some(mag),
        vec![Operand::IdRef(bits), Operand::IdRef(magnitude_mask)],
    ));

    let below_min_normal = ctx.module.fresh_id();
    let min_normal = shaped_int_const(ctx, int_ty, shape, min_normal_bits(shape.width));
    out.push(Instruction::new(
        Op::ULessThan,
        Some(bool_ty),
        Some(below_min_normal),
        vec![Operand::IdRef(mag), Operand::IdRef(min_normal)],
    ));

    let nonzero = ctx.module.fresh_id();
    let zero = shaped_int_const(ctx, int_ty, shape, 0);
    out.push(Instruction::new(
        Op::INotEqual,
        Some(bool_ty),
        Some(nonzero),
        vec![Operand::IdRef(mag), Operand::IdRef(zero)],
    ));

    let should_flush = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::LogicalAnd,
        Some(bool_ty),
        Some(should_flush),
        vec![Operand::IdRef(below_min_normal), Operand::IdRef(nonzero)],
    ));

    let signed_zero_bits = ctx.module.fresh_id();
    let sign_mask = shaped_int_const(ctx, int_ty, shape, sign_mask(shape.width));
    out.push(Instruction::new(
        Op::BitwiseAnd,
        Some(int_ty),
        Some(signed_zero_bits),
        vec![Operand::IdRef(bits), Operand::IdRef(sign_mask)],
    ));

    let signed_zero = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Bitcast,
        Some(ty),
        Some(signed_zero),
        vec![Operand::IdRef(signed_zero_bits)],
    ));

    let result = result_id.unwrap_or_else(|| ctx.module.fresh_id());
    out.push(Instruction::new(
        Op::Select,
        Some(ty),
        Some(result),
        vec![
            Operand::IdRef(should_flush),
            Operand::IdRef(signed_zero),
            Operand::IdRef(value),
        ],
    ));
    result
}

#[derive(Clone, Copy)]
struct FloatShape {
    width: u32,
    lanes: Option<u32>,
}

fn float_shape(ctx: &Ctx, ty: Word) -> Option<FloatShape> {
    let def = type_def_of(ctx, ty)?;
    if def.class.opcode == Op::TypeFloat {
        let width = match def.operands.first() {
            Some(Operand::LiteralBit32(width @ (16 | 32 | 64))) => *width,
            _ => return None,
        };
        return Some(FloatShape { width, lanes: None });
    }
    if def.class.opcode != Op::TypeVector {
        return None;
    }
    let elem = match def.operands.first() {
        Some(Operand::IdRef(elem)) => *elem,
        _ => return None,
    };
    let lanes = match def.operands.get(1) {
        Some(Operand::LiteralBit32(lanes)) => *lanes,
        _ => return None,
    };
    let elem_def = type_def_of(ctx, elem)?;
    if elem_def.class.opcode != Op::TypeFloat {
        return None;
    }
    let width = match elem_def.operands.first() {
        Some(Operand::LiteralBit32(width @ (16 | 32 | 64))) => *width,
        _ => return None,
    };
    Some(FloatShape {
        width,
        lanes: Some(lanes),
    })
}

fn shape_int_type(ctx: &mut Ctx, shape: FloatShape) -> Word {
    match (shape.width, shape.lanes) {
        (16, Some(lanes)) => ctx.ty_vec_u16(lanes),
        (16, None) => ctx.ty_int16(),
        (32, Some(lanes)) => ctx.ty_vec_uint(lanes),
        (32, None) => ctx.ty_uint(),
        (64, Some(lanes)) => ctx.ty_vec_ulong(lanes),
        (64, None) => ctx.ty_ulong(),
        _ => unreachable!("unsupported float denorm width"),
    }
}

fn shape_scalar_int_type(ctx: &mut Ctx, width: u32) -> Word {
    match width {
        16 => ctx.ty_int16(),
        32 => ctx.ty_uint(),
        64 => ctx.ty_ulong(),
        _ => unreachable!("unsupported float denorm width"),
    }
}

fn shape_bool_type(ctx: &mut Ctx, shape: FloatShape) -> Word {
    match shape.lanes {
        Some(lanes) => ctx.ty_vec_bool(lanes),
        None => ctx.ty_bool(),
    }
}

fn shaped_int_const(ctx: &mut Ctx, ty: Word, shape: FloatShape, value: u64) -> Word {
    let scalar_ty = shape_scalar_int_type(ctx, shape.width);
    let scalar = ctx.const_int_of(scalar_ty, value as i64);
    match shape.lanes {
        Some(lanes) => const_composite_splat(ctx, ty, scalar, lanes),
        None => scalar,
    }
}

fn shaped_float_const(ctx: &mut Ctx, ty: Word, shape: FloatShape, value: f64) -> Word {
    let scalar_ty = match shape.width {
        32 => ctx.ty_float(),
        64 => ctx.get_or_create(Op::TypeFloat, None, vec![Operand::LiteralBit32(64)]),
        _ => unreachable!("unsupported float denorm width"),
    };
    let scalar = match shape.width {
        32 => ctx.const_float(value as f32),
        64 => ctx.get_or_create(
            Op::Constant,
            Some(scalar_ty),
            vec![Operand::LiteralBit64(value.to_bits())],
        ),
        _ => unreachable!("unsupported float denorm width"),
    };
    match shape.lanes {
        Some(lanes) => const_composite_splat(ctx, ty, scalar, lanes),
        None => scalar,
    }
}

fn magnitude_mask(width: u32) -> u64 {
    match width {
        16 => 0x7fff,
        32 => 0x7fff_ffff,
        64 => 0x7fff_ffff_ffff_ffff,
        _ => unreachable!("unsupported float denorm width"),
    }
}

fn min_normal_bits(width: u32) -> u64 {
    match width {
        16 => 0x0400,
        32 => 0x0080_0000,
        64 => 0x0010_0000_0000_0000,
        _ => unreachable!("unsupported float denorm width"),
    }
}

fn sign_mask(width: u32) -> u64 {
    match width {
        16 => 0x8000,
        32 => 0x8000_0000,
        64 => 0x8000_0000_0000_0000,
        _ => unreachable!("unsupported float denorm width"),
    }
}

fn qnan_bits(width: u32) -> u64 {
    match width {
        16 => 0x7e00,
        32 => 0x7fc0_0000,
        64 => 0x7ff8_0000_0000_0000,
        _ => unreachable!("unsupported float denorm width"),
    }
}

fn build_fdiv_zero_zero_nan(
    ctx: &mut Ctx,
    raw_div: Word,
    ty: Word,
    operands: &[Operand],
    out: &mut Vec<Instruction>,
) -> Option<Word> {
    let lhs = operand_id(operands.first()?)?;
    let rhs = operand_id(operands.get(1)?)?;
    let shape = float_shape(ctx, ty)?;
    let lhs_zero = build_float_zero_predicate(ctx, lhs, ty, out)?;
    let rhs_zero = build_float_zero_predicate(ctx, rhs, ty, out)?;
    let bool_ty = shape_bool_type(ctx, shape);
    let both_zero = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::LogicalAnd,
        Some(bool_ty),
        Some(both_zero),
        vec![Operand::IdRef(lhs_zero), Operand::IdRef(rhs_zero)],
    ));
    let qnan = qnan_float_value(ctx, ty, shape, out);
    let guarded = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Select,
        Some(ty),
        Some(guarded),
        vec![
            Operand::IdRef(both_zero),
            Operand::IdRef(qnan),
            Operand::IdRef(raw_div),
        ],
    ));
    Some(guarded)
}

fn build_float_zero_predicate(
    ctx: &mut Ctx,
    value: Word,
    ty: Word,
    out: &mut Vec<Instruction>,
) -> Option<Word> {
    let shape = float_shape(ctx, ty)?;
    let int_ty = shape_int_type(ctx, shape);
    let bool_ty = shape_bool_type(ctx, shape);
    let bits = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Bitcast,
        Some(int_ty),
        Some(bits),
        vec![Operand::IdRef(value)],
    ));
    let mag = ctx.module.fresh_id();
    let magnitude_mask = shaped_int_const(ctx, int_ty, shape, magnitude_mask(shape.width));
    out.push(Instruction::new(
        Op::BitwiseAnd,
        Some(int_ty),
        Some(mag),
        vec![Operand::IdRef(bits), Operand::IdRef(magnitude_mask)],
    ));
    let zero = shaped_int_const(ctx, int_ty, shape, 0);
    let is_zero = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::IEqual,
        Some(bool_ty),
        Some(is_zero),
        vec![Operand::IdRef(mag), Operand::IdRef(zero)],
    ));
    Some(is_zero)
}

fn qnan_float_value(
    ctx: &mut Ctx,
    ty: Word,
    shape: FloatShape,
    out: &mut Vec<Instruction>,
) -> Word {
    let int_ty = shape_int_type(ctx, shape);
    let qnan_bits = shaped_int_const(ctx, int_ty, shape, qnan_bits(shape.width));
    let qnan = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Bitcast,
        Some(ty),
        Some(qnan),
        vec![Operand::IdRef(qnan_bits)],
    ));
    qnan
}

fn operand_id(operand: &Operand) -> Option<Word> {
    match operand {
        Operand::IdRef(id) => Some(*id),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denorm_flush_to_zero_emulates_f64() {
        let mut module = Module::new();
        let double = module.fresh_id();
        let void = module.fresh_id();
        let fn_ty = module.fresh_id();
        let param = module.fresh_id();
        let div = module.fresh_id();
        let func = module.fresh_id();
        let label = module.fresh_id();
        module.types_global_values.extend([
            Instruction::new(
                Op::TypeFloat,
                None,
                Some(double),
                vec![Operand::LiteralBit32(64)],
            ),
            Instruction::new(Op::TypeVoid, None, Some(void), vec![]),
            Instruction::new(
                Op::TypeFunction,
                None,
                Some(fn_ty),
                vec![Operand::IdRef(void), Operand::IdRef(double)],
            ),
        ]);
        let mut block = Block::new();
        block.label = Some(Instruction::new(Op::Label, None, Some(label), vec![]));
        block.instructions.push(Instruction::new(
            Op::FDiv,
            Some(double),
            Some(div),
            vec![Operand::IdRef(param), Operand::IdRef(param)],
        ));
        block
            .instructions
            .push(Instruction::new(Op::Return, None, None, vec![]));
        let mut function = Function::new();
        function.def = Some(Instruction::new(
            Op::Function,
            Some(void),
            Some(func),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(fn_ty),
            ],
        ));
        function.parameters.push(Instruction::new(
            Op::FunctionParameter,
            Some(double),
            Some(param),
            vec![],
        ));
        function.blocks.push(block);
        function.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
        module.functions.push(function);

        let mut ctx = Ctx::with_options(
            module,
            Stage::Kernel,
            TransformOptions {
                denorm_flush_to_zero_f32: true,
                ..Default::default()
            },
        );
        emulate_f32_denorm_flush_to_zero(&mut ctx, 0);

        let globals = ctx
            .module
            .types_global_values
            .iter()
            .chain(ctx.new_globals.iter())
            .collect::<Vec<_>>();
        let instructions = &ctx.module.functions[0].blocks[0].instructions;
        assert!(globals.iter().any(|inst| {
            inst.class.opcode == Op::TypeInt
                && inst.operands == vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)]
        }));
        let constant_values = globals
            .iter()
            .filter(|inst| inst.class.opcode == Op::Constant)
            .filter_map(|inst| match inst.operands.as_slice() {
                [Operand::LiteralBit64(value)] => Some(*value),
                _ => None,
            })
            .collect::<Vec<_>>();
        for value in [
            0x7fff_ffff_ffff_ffff,
            0x0010_0000_0000_0000,
            0x8000_0000_0000_0000,
            0x7ff8_0000_0000_0000,
        ] {
            assert!(
                constant_values.contains(&value),
                "missing {value:#x} in {constant_values:#x?}"
            );
        }

        assert!(instructions
            .iter()
            .any(|inst| inst.class.opcode == Op::Bitcast && inst.result_type == Some(double)));
        assert!(instructions
            .iter()
            .any(|inst| inst.class.opcode == Op::BitwiseAnd));
        assert!(instructions
            .iter()
            .any(|inst| inst.class.opcode == Op::ULessThan));
        assert!(
            instructions
                .iter()
                .filter(|inst| inst.class.opcode == Op::Select)
                .count()
                >= 2
        );
        let rewritten_div = instructions
            .iter()
            .find(|inst| inst.class.opcode == Op::FDiv)
            .expect("rewritten fdiv");
        assert_ne!(rewritten_div.result_id, Some(div));
    }
}
