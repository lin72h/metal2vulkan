//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;
use crate::passes::stage_input::{layout_ty_size_align, round_up};

pub(in crate::passes) fn const_int_like(ctx: &mut Ctx, like: Word, value: u64) -> Word {
    let Some(ty) = value_result_type(ctx, like) else {
        return ctx.const_uint(value as u32);
    };
    let Some(def) = type_def_of(ctx, ty) else {
        return ctx.const_uint(value as u32);
    };
    if def.class.opcode != Op::TypeInt {
        return ctx.const_uint(value as u32);
    }
    match def.operands.first() {
        Some(Operand::LiteralBit32(64)) => {
            let id = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::Constant,
                Some(ty),
                Some(id),
                vec![Operand::LiteralBit64(value)],
            ));
            id
        }
        _ => ctx.const_uint(value as u32),
    }
}

/// Narrow 64-bit-integer INDEX operands of an `OpAccessChain`/`OpInBoundsAccessChain`/
/// `OpPtrAccessChain` in the entry function to 32-bit-integer equivalents WHERE VALUE-PRESERVING.
/// NVIDIA's SPIR-V->NVVM compiler SEGFAULTed when an access-chain index was a 64-bit (`%ulong`)
/// value (that rail is retired; the narrowing is kept because 32-bit indices remain the more
/// portable form):
///   * a 64-bit CONSTANT index that fits in u32 -> the equal-valued 32-bit `OpConstant uint`;
///   * a 64-bit CONSTANT index that does NOT fit -> left 64-bit (truncating it would silently WRAP
///     the address: a degenerate-but-real AIR shape, e.g. copyKernel's synthesized
///     `MTLCopyArgs{0x1_00000001,...}`, indexes with i64 values >= 2^32, and Apple hardware resolves
///     the full 64-bit address — truncation made metal2vulkan write a texel Apple leaves untouched);
///   * a 64-bit SSA index -> left 64-bit for the same reason: whether it exceeds u32 is a runtime
///     property, and MoltenVK/spirv-val accept 64-bit access-chain indices.
///     The base pointer operand (operand 0) is left untouched — only fitting constant indices are
///     narrowed.
pub(in crate::passes) fn narrow_access_chain_indices(ctx: &mut Ctx, entry_idx: usize) {
    let value_types = function_value_types(ctx, entry_idx);
    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let mut new_insts: Vec<Instruction> = vec![];
        let insts = ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .clone();
        for mut inst in insts {
            if !matches!(
                inst.class.opcode,
                Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
            ) {
                new_insts.push(inst);
                continue;
            }
            // Operand 0 is the base pointer; operands 1.. are the indices. Narrow each 64-bit index.
            let pre: Vec<Instruction> = vec![];
            for oi in 1..inst.operands.len() {
                let idx = match inst.operands[oi] {
                    Operand::IdRef(r) => r,
                    _ => continue,
                };
                let idx_ty = match value_types.get(&idx).copied() {
                    Some(t) => t,
                    None => continue,
                };
                let Some(signed) = int64_signedness(ctx, idx_ty) else {
                    continue; // already 32-bit (or narrower) -> leave it.
                };
                let _ = signed;
                if let Some(v) = const_i64_value(ctx, idx) {
                    // Constant index: reuse/synthesize the equal-valued 32-bit uint constant, but
                    // only when the value actually fits — truncating a >= 2^32 constant would
                    // silently wrap the address instead of keeping Apple's 64-bit resolution.
                    if u32::try_from(v).is_ok() {
                        let c32 = ctx.const_uint(v as u32);
                        inst.operands[oi] = Operand::IdRef(c32);
                    }
                }
                // Dynamic 64-bit indices stay 64-bit: whether they exceed u32 is a runtime
                // property, and truncation diverges from the 64-bit address math Apple hardware
                // performs for the same AIR.
            }
            new_insts.extend(pre);
            new_insts.push(inst);
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = new_insts;
    }
}

pub(in crate::passes) fn lower_scalar_i64_arithmetic_to_u32_halves(ctx: &mut Ctx) {
    ctx.module.sync_id_bound_from_instructions();
    if let Some(staged_bound) = ctx
        .new_globals
        .iter()
        .filter_map(|instruction| instruction.result_id)
        .max()
        .map(|id| id.saturating_add(1))
    {
        if ctx.module.id_bound() < staged_bound {
            ctx.module.set_id_bound(staged_bound);
        }
    }
    let mut int_types: HashMap<Word, (u32, u32)> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if inst.class.opcode != Op::TypeInt {
            continue;
        }
        let (Some(id), Some(Operand::LiteralBit32(width)), Some(Operand::LiteralBit32(signed))) =
            (inst.result_id, inst.operands.first(), inst.operands.get(1))
        else {
            continue;
        };
        int_types.insert(id, (*width, *signed));
    }

    let mut value_types: HashMap<Word, Word> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if let (Some(result), Some(ty)) = (inst.result_id, inst.result_type) {
            value_types.insert(result, ty);
        }
    }
    for function in &ctx.module.functions {
        for param in &function.parameters {
            if let (Some(result), Some(ty)) = (param.result_id, param.result_type) {
                value_types.insert(result, ty);
            }
        }
        for block in &function.blocks {
            for inst in &block.instructions {
                if let (Some(result), Some(ty)) = (inst.result_id, inst.result_type) {
                    value_types.insert(result, ty);
                }
            }
        }
    }

    let uint = ctx.ty_uint();
    let ulong = ctx.ty_ulong();
    let bool_ty = ctx.ty_bool();
    let shift32 = ctx.const_int_of(ulong, 32);
    let shift16 = ctx.const_uint(16);
    let mask16 = ctx.const_uint(0xffff);
    let zero = ctx.const_uint(0);
    let one = ctx.const_uint(1);

    for function_idx in 0..ctx.module.functions.len() {
        for block_idx in 0..ctx.module.functions[function_idx].blocks.len() {
            let insts = ctx.module.functions[function_idx].blocks[block_idx]
                .instructions
                .clone();
            let mut out = Vec::with_capacity(insts.len());
            for inst in insts {
                if !matches!(inst.class.opcode, Op::IAdd | Op::ISub | Op::IMul)
                    || inst.operands.len() != 2
                    || !matches!(
                        inst.result_type.and_then(|ty| int_types.get(&ty).copied()),
                        Some((64, _))
                    )
                {
                    out.push(inst);
                    continue;
                }
                let (Operand::IdRef(lhs), Operand::IdRef(rhs)) =
                    (inst.operands[0].clone(), inst.operands[1].clone())
                else {
                    out.push(inst);
                    continue;
                };
                let Some(result_ty) = inst.result_type else {
                    out.push(inst);
                    continue;
                };
                let Some(result_id) = inst.result_id else {
                    out.push(inst);
                    continue;
                };
                let Some(lhs_ty) = value_types.get(&lhs).copied() else {
                    out.push(inst);
                    continue;
                };
                let Some(rhs_ty) = value_types.get(&rhs).copied() else {
                    out.push(inst);
                    continue;
                };
                if !matches!(int_types.get(&lhs_ty).copied(), Some((64, _)))
                    || !matches!(int_types.get(&rhs_ty).copied(), Some((64, _)))
                {
                    out.push(inst);
                    continue;
                }

                let lhs_u64 = bitcast_i64_to_ulong(ctx, &mut out, lhs, lhs_ty, ulong);
                let rhs_u64 = bitcast_i64_to_ulong(ctx, &mut out, rhs, rhs_ty, ulong);
                let lowered = match inst.class.opcode {
                    Op::IAdd => scalar_i64_add_as_u32_halves(
                        ctx, &mut out, lhs_u64, rhs_u64, uint, ulong, bool_ty, shift32, zero, one,
                    ),
                    Op::ISub => scalar_i64_sub_as_u32_halves(
                        ctx, &mut out, lhs_u64, rhs_u64, uint, ulong, bool_ty, shift32, zero, one,
                    ),
                    Op::IMul => scalar_i64_mul_as_u32_halves(
                        ctx, &mut out, lhs_u64, rhs_u64, uint, ulong, shift32, shift16, mask16,
                    ),
                    _ => unreachable!(),
                };
                if result_ty == ulong {
                    out.last_mut()
                        .expect("lowering emits a final instruction")
                        .result_id = Some(result_id);
                    value_types.insert(result_id, ulong);
                } else {
                    out.push(Instruction::new(
                        Op::Bitcast,
                        Some(result_ty),
                        Some(result_id),
                        vec![Operand::IdRef(lowered)],
                    ));
                    value_types.insert(result_id, result_ty);
                }
            }
            ctx.module.functions[function_idx].blocks[block_idx].instructions = out;
        }
    }
}

fn bitcast_i64_to_ulong(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    value: Word,
    value_ty: Word,
    ulong: Word,
) -> Word {
    if value_ty == ulong {
        return value;
    }
    let cast = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Bitcast,
        Some(ulong),
        Some(cast),
        vec![Operand::IdRef(value)],
    ));
    cast
}

fn scalar_i64_add_as_u32_halves(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    lhs: Word,
    rhs: Word,
    uint: Word,
    ulong: Word,
    bool_ty: Word,
    shift32: Word,
    zero: Word,
    one: Word,
) -> Word {
    let (lhs_lo, lhs_hi) = u64_halves(ctx, out, lhs, uint, ulong, shift32);
    let (rhs_lo, rhs_hi) = u64_halves(ctx, out, rhs, uint, ulong, shift32);
    let low = iadd_u32(ctx, out, lhs_lo, rhs_lo, uint);
    let carry = ult_u32_select_bit(ctx, out, low, lhs_lo, uint, bool_ty, zero, one);
    let high_base = iadd_u32(ctx, out, lhs_hi, rhs_hi, uint);
    let high = iadd_u32(ctx, out, high_base, carry, uint);
    assemble_u64_halves(ctx, out, low, high, ulong, shift32)
}

fn scalar_i64_sub_as_u32_halves(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    lhs: Word,
    rhs: Word,
    uint: Word,
    ulong: Word,
    bool_ty: Word,
    shift32: Word,
    zero: Word,
    one: Word,
) -> Word {
    let (lhs_lo, lhs_hi) = u64_halves(ctx, out, lhs, uint, ulong, shift32);
    let (rhs_lo, rhs_hi) = u64_halves(ctx, out, rhs, uint, ulong, shift32);
    let borrow = ult_u32_select_bit(ctx, out, lhs_lo, rhs_lo, uint, bool_ty, zero, one);
    let low = isub_u32(ctx, out, lhs_lo, rhs_lo, uint);
    let high_base = isub_u32(ctx, out, lhs_hi, rhs_hi, uint);
    let high = isub_u32(ctx, out, high_base, borrow, uint);
    assemble_u64_halves(ctx, out, low, high, ulong, shift32)
}

fn scalar_i64_mul_as_u32_halves(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    lhs: Word,
    rhs: Word,
    uint: Word,
    ulong: Word,
    shift32: Word,
    shift16: Word,
    mask16: Word,
) -> Word {
    let lhs_lo = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::UConvert,
        Some(uint),
        Some(lhs_lo),
        vec![Operand::IdRef(lhs)],
    ));
    let rhs_lo = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::UConvert,
        Some(uint),
        Some(rhs_lo),
        vec![Operand::IdRef(rhs)],
    ));

    let lhs_shifted = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ShiftRightLogical,
        Some(ulong),
        Some(lhs_shifted),
        vec![Operand::IdRef(lhs), Operand::IdRef(shift32)],
    ));
    let lhs_hi = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::UConvert,
        Some(uint),
        Some(lhs_hi),
        vec![Operand::IdRef(lhs_shifted)],
    ));
    let rhs_shifted = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ShiftRightLogical,
        Some(ulong),
        Some(rhs_shifted),
        vec![Operand::IdRef(rhs), Operand::IdRef(shift32)],
    ));
    let rhs_hi = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::UConvert,
        Some(uint),
        Some(rhs_hi),
        vec![Operand::IdRef(rhs_shifted)],
    ));

    let (lo, carry) =
        mul_u32_to_u64_halves_via_u16(ctx, out, lhs_lo, rhs_lo, uint, shift16, mask16);

    let lhs_hi_rhs_lo = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::IMul,
        Some(uint),
        Some(lhs_hi_rhs_lo),
        vec![Operand::IdRef(lhs_hi), Operand::IdRef(rhs_lo)],
    ));
    let lhs_lo_rhs_hi = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::IMul,
        Some(uint),
        Some(lhs_lo_rhs_hi),
        vec![Operand::IdRef(lhs_lo), Operand::IdRef(rhs_hi)],
    ));
    let high_partial = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(high_partial),
        vec![Operand::IdRef(carry), Operand::IdRef(lhs_hi_rhs_lo)],
    ));
    let high = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(high),
        vec![Operand::IdRef(high_partial), Operand::IdRef(lhs_lo_rhs_hi)],
    ));

    let low64 = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::UConvert,
        Some(ulong),
        Some(low64),
        vec![Operand::IdRef(lo)],
    ));
    let high64 = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::UConvert,
        Some(ulong),
        Some(high64),
        vec![Operand::IdRef(high)],
    ));
    let high_shifted = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ShiftLeftLogical,
        Some(ulong),
        Some(high_shifted),
        vec![Operand::IdRef(high64), Operand::IdRef(shift32)],
    ));
    let product = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::BitwiseOr,
        Some(ulong),
        Some(product),
        vec![Operand::IdRef(high_shifted), Operand::IdRef(low64)],
    ));
    product
}

fn mul_u32_to_u64_halves_via_u16(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    lhs: Word,
    rhs: Word,
    uint: Word,
    shift16: Word,
    mask16: Word,
) -> (Word, Word) {
    let lhs_lo = and_u32(ctx, out, lhs, mask16, uint);
    let lhs_hi_shifted = shr_u32(ctx, out, lhs, shift16, uint);
    let lhs_hi = and_u32(ctx, out, lhs_hi_shifted, mask16, uint);
    let rhs_lo = and_u32(ctx, out, rhs, mask16, uint);
    let rhs_hi_shifted = shr_u32(ctx, out, rhs, shift16, uint);
    let rhs_hi = and_u32(ctx, out, rhs_hi_shifted, mask16, uint);

    let p0 = imul_u32(ctx, out, lhs_lo, rhs_lo, uint);
    let p1 = imul_u32(ctx, out, lhs_hi, rhs_lo, uint);
    let p2 = imul_u32(ctx, out, lhs_lo, rhs_hi, uint);
    let p3 = imul_u32(ctx, out, lhs_hi, rhs_hi, uint);

    let p0_lo = and_u32(ctx, out, p0, mask16, uint);
    let p0_hi = shr_u32(ctx, out, p0, shift16, uint);
    let p1_lo = and_u32(ctx, out, p1, mask16, uint);
    let p1_hi = shr_u32(ctx, out, p1, shift16, uint);
    let p2_lo = and_u32(ctx, out, p2, mask16, uint);
    let p2_hi = shr_u32(ctx, out, p2, shift16, uint);

    let middle_a = iadd_u32(ctx, out, p0_hi, p1_lo, uint);
    let middle = iadd_u32(ctx, out, middle_a, p2_lo, uint);
    let middle_lo = and_u32(ctx, out, middle, mask16, uint);
    let middle_carry = shr_u32(ctx, out, middle, shift16, uint);
    let middle_lo_shifted = shl_u32(ctx, out, middle_lo, shift16, uint);
    let low = or_u32(ctx, out, p0_lo, middle_lo_shifted, uint);

    let high_a = iadd_u32(ctx, out, p3, p1_hi, uint);
    let high_b = iadd_u32(ctx, out, high_a, p2_hi, uint);
    let high = iadd_u32(ctx, out, high_b, middle_carry, uint);
    (low, high)
}

fn u64_halves(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    value: Word,
    uint: Word,
    ulong: Word,
    shift32: Word,
) -> (Word, Word) {
    let lo = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::UConvert,
        Some(uint),
        Some(lo),
        vec![Operand::IdRef(value)],
    ));
    let shifted = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ShiftRightLogical,
        Some(ulong),
        Some(shifted),
        vec![Operand::IdRef(value), Operand::IdRef(shift32)],
    ));
    let hi = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::UConvert,
        Some(uint),
        Some(hi),
        vec![Operand::IdRef(shifted)],
    ));
    (lo, hi)
}

fn assemble_u64_halves(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    lo: Word,
    hi: Word,
    ulong: Word,
    shift32: Word,
) -> Word {
    let low64 = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::UConvert,
        Some(ulong),
        Some(low64),
        vec![Operand::IdRef(lo)],
    ));
    let high64 = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::UConvert,
        Some(ulong),
        Some(high64),
        vec![Operand::IdRef(hi)],
    ));
    let high_shifted = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ShiftLeftLogical,
        Some(ulong),
        Some(high_shifted),
        vec![Operand::IdRef(high64), Operand::IdRef(shift32)],
    ));
    let result = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::BitwiseOr,
        Some(ulong),
        Some(result),
        vec![Operand::IdRef(high_shifted), Operand::IdRef(low64)],
    ));
    result
}

fn imul_u32(ctx: &mut Ctx, out: &mut Vec<Instruction>, lhs: Word, rhs: Word, uint: Word) -> Word {
    let result = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::IMul,
        Some(uint),
        Some(result),
        vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
    ));
    result
}

fn isub_u32(ctx: &mut Ctx, out: &mut Vec<Instruction>, lhs: Word, rhs: Word, uint: Word) -> Word {
    let result = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ISub,
        Some(uint),
        Some(result),
        vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
    ));
    result
}

fn iadd_u32(ctx: &mut Ctx, out: &mut Vec<Instruction>, lhs: Word, rhs: Word, uint: Word) -> Word {
    let result = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(result),
        vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
    ));
    result
}

fn ult_u32_select_bit(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    lhs: Word,
    rhs: Word,
    uint: Word,
    bool_ty: Word,
    zero: Word,
    one: Word,
) -> Word {
    let pred = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ULessThan,
        Some(bool_ty),
        Some(pred),
        vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
    ));
    let result = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Select,
        Some(uint),
        Some(result),
        vec![
            Operand::IdRef(pred),
            Operand::IdRef(one),
            Operand::IdRef(zero),
        ],
    ));
    result
}

fn and_u32(ctx: &mut Ctx, out: &mut Vec<Instruction>, lhs: Word, rhs: Word, uint: Word) -> Word {
    let result = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(result),
        vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
    ));
    result
}

fn or_u32(ctx: &mut Ctx, out: &mut Vec<Instruction>, lhs: Word, rhs: Word, uint: Word) -> Word {
    let result = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::BitwiseOr,
        Some(uint),
        Some(result),
        vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
    ));
    result
}

fn shl_u32(ctx: &mut Ctx, out: &mut Vec<Instruction>, base: Word, shift: Word, uint: Word) -> Word {
    let result = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ShiftLeftLogical,
        Some(uint),
        Some(result),
        vec![Operand::IdRef(base), Operand::IdRef(shift)],
    ));
    result
}

fn shr_u32(ctx: &mut Ctx, out: &mut Vec<Instruction>, base: Word, shift: Word, uint: Word) -> Word {
    let result = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ShiftRightLogical,
        Some(uint),
        Some(result),
        vec![Operand::IdRef(base), Operand::IdRef(shift)],
    ));
    result
}

/// After helper inlining and access-chain composition, scalar pointer arithmetic can appear as
/// `OpInBoundsAccessChain %ptr_T %already_ptr_T %idx`. In Logical SPIR-V, that tries to index
/// through scalar `T`. Storage classes that Vulkan allows as `OpPtrAccessChain` bases use that form;
/// Private chains must have been composed back to an aggregate root first.
pub(in crate::passes) fn rewrite_scalar_pointer_arithmetic_access_chains(
    ctx: &mut Ctx,
    entry_idx: usize,
) {
    let mut pointer_storage = HashMap::new();
    let aggregate_types = ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .filter(|inst| {
            matches!(
                inst.class.opcode,
                Op::TypeStruct | Op::TypeArray | Op::TypeRuntimeArray | Op::TypeMatrix
            )
        })
        .filter_map(|inst| inst.result_id)
        .collect::<HashSet<_>>();
    let mut pointer_pointees = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if inst.class.opcode == Op::TypePointer {
            if let (
                Some(result),
                Some(Operand::StorageClass(storage)),
                Some(Operand::IdRef(pointee)),
            ) = (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                pointer_storage.insert(result, *storage);
                pointer_pointees.insert(result, *pointee);
            }
        }
    }

    let mut id_types = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if let (Some(result), Some(result_type)) = (inst.result_id, inst.result_type) {
            id_types.insert(result, result_type);
        }
    }
    for function in &ctx.module.functions {
        for inst in &function.parameters {
            if let (Some(result), Some(result_type)) = (inst.result_id, inst.result_type) {
                id_types.insert(result, result_type);
            }
        }
        for block in &function.blocks {
            for inst in &block.instructions {
                if let (Some(result), Some(result_type)) = (inst.result_id, inst.result_type) {
                    id_types.insert(result, result_type);
                }
            }
        }
    }

    for block in &mut ctx.module.functions[entry_idx].blocks {
        for inst in &mut block.instructions {
            if inst.class.opcode != Op::InBoundsAccessChain {
                continue;
            }
            let Some(result_type) = inst.result_type else {
                continue;
            };
            if !pointer_storage
                .get(&result_type)
                .is_some_and(|storage| ptr_access_chain_allowed_storage(*storage))
            {
                continue;
            }
            if pointer_pointees
                .get(&result_type)
                .is_some_and(|pointee| aggregate_types.contains(pointee))
            {
                continue;
            }
            let Some(Operand::IdRef(base)) = inst.operands.first() else {
                continue;
            };
            if id_types.get(base) != Some(&result_type) {
                continue;
            }
            *inst = Instruction::new(
                Op::PtrAccessChain,
                inst.result_type,
                inst.result_id,
                inst.operands.clone(),
            );
        }
    }
}

/// Replace a nullable pointer select/phi used as a memory base with its concrete arm.
///
/// Dereferencing the null arm is undefined in LLVM, so every defined execution that reaches the
/// memory operation necessarily selected the concrete arm. Removing the merge at that use exposes
/// the original pointer to ordinary memory lowering and lets the now-dead Logical pointer null and
/// merge disappear during final liveness collection.
pub(in crate::passes) fn expose_nullable_memory_bases(ctx: &mut Ctx, entry_idx: usize) {
    let null_ids: HashSet<Word> = ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .filter(|inst| inst.class.opcode == Op::ConstantNull)
        .filter_map(|inst| inst.result_id)
        .collect();
    let concrete_arm: HashMap<Word, Word> = ctx.module.functions[entry_idx]
        .blocks
        .iter()
        .flat_map(|block| block.instructions.iter())
        .filter_map(|inst| {
            let result = inst.result_id?;
            let arms = match inst.class.opcode {
                Op::Select => {
                    let [_, Operand::IdRef(on_true), Operand::IdRef(on_false)] =
                        inst.operands.as_slice()
                    else {
                        return None;
                    };
                    vec![*on_true, *on_false]
                }
                Op::Phi => inst
                    .operands
                    .chunks_exact(2)
                    .filter_map(|pair| match pair.first() {
                        Some(Operand::IdRef(value)) => Some(*value),
                        _ => None,
                    })
                    .collect(),
                _ => return None,
            };
            if !arms.iter().any(|arm| null_ids.contains(arm)) {
                return None;
            }
            let mut concrete = arms
                .into_iter()
                .filter(|arm| !null_ids.contains(arm))
                .collect::<HashSet<_>>();
            (concrete.len() == 1).then(|| (result, concrete.drain().next().unwrap()))
        })
        .collect();

    for block in &mut ctx.module.functions[entry_idx].blocks {
        for inst in &mut block.instructions {
            if !matches!(
                inst.class.opcode,
                Op::AccessChain
                    | Op::InBoundsAccessChain
                    | Op::PtrAccessChain
                    | Op::InBoundsPtrAccessChain
                    | Op::Load
                    | Op::Store
                    | Op::AtomicLoad
                    | Op::AtomicStore
                    | Op::AtomicExchange
                    | Op::AtomicCompareExchange
                    | Op::AtomicCompareExchangeWeak
                    | Op::AtomicIIncrement
                    | Op::AtomicIDecrement
                    | Op::AtomicIAdd
                    | Op::AtomicISub
                    | Op::AtomicSMin
                    | Op::AtomicUMin
                    | Op::AtomicSMax
                    | Op::AtomicUMax
                    | Op::AtomicAnd
                    | Op::AtomicOr
                    | Op::AtomicXor
                    | Op::AtomicFAddEXT
                    | Op::AtomicFMinEXT
                    | Op::AtomicFMaxEXT
            ) {
                continue;
            }
            let Some(Operand::IdRef(base)) = inst.operands.first_mut() else {
                continue;
            };
            if let Some(concrete) = concrete_arm.get(base) {
                *base = *concrete;
            }
        }
    }

    // Once all dereferences bypass a nullable merge, retire that merge if it has no remaining use.
    // This is part of the representation change, not general dead-code elimination: leaving the
    // unused OpPhi/OpSelect would keep its Logical pointer-null arm live at module scope.
    loop {
        let used = ctx.module.functions[entry_idx]
            .blocks
            .iter()
            .flat_map(|block| block.instructions.iter())
            .flat_map(|instruction| instruction.operands.iter())
            .filter_map(|operand| match operand {
                Operand::IdRef(id) | Operand::IdScope(id) | Operand::IdMemorySemantics(id) => {
                    Some(*id)
                }
                _ => None,
            })
            .collect::<HashSet<_>>();
        let mut changed = false;
        for block in &mut ctx.module.functions[entry_idx].blocks {
            block.instructions.retain(|instruction| {
                let dead_nullable_merge = instruction.result_id.is_some_and(|result| {
                    concrete_arm.contains_key(&result) && !used.contains(&result)
                });
                changed |= dead_nullable_merge;
                !dead_nullable_merge
            });
        }
        if !changed {
            break;
        }
    }
}

/// `OpPtrAccessChain` requires its Base pointer *type* to carry an `ArrayStride` decoration (the
/// element size it strides by), per VUID-StandaloneSpirv-None-10684. When the base points at a
/// scalar/vector element of a StorageBuffer (the scalar-pointer-arithmetic form produced above and
/// by the native emitter's `pointer_arithmetic_access_chain_op_for_storage` path), nothing in the
/// Block-layout pass decorates that pointer type — it is not an array/struct member — so spirv-val
/// rejects the chain. This walks every `OpPtrAccessChain` in the module and adds the missing
/// `ArrayStride = round_up(sizeof pointee)` to each distinct base pointer type, idempotently.
pub(in crate::passes) fn decorate_ptr_access_chain_base_strides(ctx: &mut Ctx) {
    // OpPtrAccessChain's result pointer must stay in the base pointer's storage class. Interface
    // substitution can expose an already-lowered PhysicalStorageBuffer base underneath a helper
    // chain whose provisional logical type was UniformConstant; the base is the exact address-domain
    // contract, so carry its storage through the chain before deriving layout decorations.
    let query_defs = ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .filter_map(|instruction| Some((instruction.result_id?, instruction)))
        .collect::<HashMap<_, _>>();
    let value_types = ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .chain(ctx.module.functions.iter().flat_map(|function| {
            function
                .parameters
                .iter()
                .chain(function.blocks.iter().flat_map(|block| &block.instructions))
        }))
        .filter_map(|instruction| Some((instruction.result_id?, instruction.result_type?)))
        .collect::<HashMap<_, _>>();
    let mut storage_plans = Vec::new();
    for (function_index, function) in ctx.module.functions.iter().enumerate() {
        for (block_index, block) in function.blocks.iter().enumerate() {
            for (instruction_index, inst) in block.instructions.iter().enumerate() {
                if inst.class.opcode != Op::PtrAccessChain {
                    continue;
                }
                let (Some(result_type), Some(Operand::IdRef(base))) =
                    (inst.result_type, inst.operands.first())
                else {
                    continue;
                };
                let Some(result_pointer) = query_defs.get(&result_type) else {
                    continue;
                };
                let (Some(Operand::StorageClass(result_storage)), Some(Operand::IdRef(pointee))) = (
                    result_pointer.operands.first(),
                    result_pointer.operands.get(1),
                ) else {
                    continue;
                };
                let Some((base_storage, base_pointee)) = value_types
                    .get(base)
                    .and_then(|ty| query_defs.get(ty))
                    .and_then(|ty| match (ty.operands.first(), ty.operands.get(1)) {
                        (Some(Operand::StorageClass(storage)), Some(Operand::IdRef(pointee))) => {
                            Some((*storage, *pointee))
                        }
                        _ => None,
                    })
                else {
                    continue;
                };
                // With only Base + Element operands, PtrAccessChain performs pointer arithmetic
                // over the base pointee and cannot change that pointee type. Additional operands
                // descend into it and retain the already selected result pointee.
                let selected_pointee = if inst.operands.len() == 2 {
                    base_pointee
                } else {
                    *pointee
                };
                if base_storage != *result_storage || selected_pointee != *pointee {
                    storage_plans.push((
                        function_index,
                        block_index,
                        instruction_index,
                        base_storage,
                        selected_pointee,
                    ));
                }
            }
        }
    }
    for (function_index, block_index, instruction_index, storage, pointee) in storage_plans {
        let pointer_type = ctx.ty_ptr(storage, pointee);
        ctx.module.functions[function_index].blocks[block_index].instructions[instruction_index]
            .result_type = Some(pointer_type);
    }

    let mut defs: HashMap<Word, Instruction> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if let Some(result) = inst.result_id {
            defs.insert(result, inst.clone());
        }
    }

    // Pointer types already carrying an ArrayStride (avoid emitting a duplicate decoration).
    let mut already: HashSet<Word> = HashSet::new();
    for ann in &ctx.module.annotations {
        if ann.class.opcode == Op::Decorate
            && ann.operands.get(1) == Some(&Operand::Decoration(Decoration::ArrayStride))
        {
            if let Some(Operand::IdRef(t)) = ann.operands.first() {
                already.insert(*t);
            }
        }
    }

    // Collect the pointer types that every OpPtrAccessChain strides through. The Base operand's
    // pointer type is the one the VUID requires ArrayStride on; for the scalar-pointer-arithmetic
    // (same-type) form base type == result type, but for a stride+descent chain (the
    // `rewrite_strided_descent_access_chains` output) the base points at a wider aggregate than the
    // result, so its type differs — decorate BOTH (a redundant ArrayStride on a non-base pointer type
    // is harmless; spirv-val enforces it only when the type is actually a PtrAccessChain Base).
    let mut base_ptr_types: Vec<Word> = Vec::new();
    for function in &ctx.module.functions {
        for block in &function.blocks {
            for inst in &block.instructions {
                if inst.class.opcode != Op::PtrAccessChain {
                    continue;
                }
                if let Some(t) = inst.result_type {
                    if !base_ptr_types.contains(&t) {
                        base_ptr_types.push(t);
                    }
                }
                if let Some(Operand::IdRef(base)) = inst.operands.first() {
                    if let Some(t) = value_types.get(base).copied() {
                        if !base_ptr_types.contains(&t) {
                            base_ptr_types.push(t);
                        }
                    }
                }
            }
        }
    }

    for ptr_ty in base_ptr_types {
        if already.contains(&ptr_ty) {
            continue;
        }
        let Some(def) = defs.get(&ptr_ty) else {
            continue;
        };
        if def.class.opcode != Op::TypePointer {
            continue;
        }
        // `ArrayStride` is an explicit layout decoration. Vulkan permits it on the logical
        // StorageBuffer/PhysicalStorageBuffer pointer view used by `OpPtrAccessChain`, but rejects
        // it on Function and Workgroup pointer types (VUID-StandaloneSpirv-None-10684).
        let Some(Operand::StorageClass(storage)) = def.operands.first() else {
            continue;
        };
        if !matches!(
            storage,
            StorageClass::StorageBuffer | StorageClass::PhysicalStorageBuffer
        ) {
            continue;
        }
        // OpTypePointer %storage %pointee — pointee is operand[1].
        let Some(Operand::IdRef(pointee)) = def.operands.get(1) else {
            continue;
        };
        let (size, align) = layout_ty_size_align(ctx, *pointee, &defs);
        let stride = round_up(size, align).max(1);
        ctx.module.annotations.push(Instruction::new(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(ptr_ty),
                Operand::Decoration(Decoration::ArrayStride),
                Operand::LiteralBit32(stride),
            ],
        ));
        already.insert(ptr_ty);
    }
}

pub(in crate::passes) fn ptr_access_chain_allowed_storage(storage: StorageClass) -> bool {
    matches!(
        storage,
        StorageClass::Workgroup | StorageClass::StorageBuffer | StorageClass::PhysicalStorageBuffer
    )
}

/// Walk a composite TYPE id through access-chain index operands (each an `IdRef` to an `OpConstant`),
/// returning the innermost reached type id, or `None` if a step indexes a non-composite. This is the
/// "index INTO the type" semantics (struct member by constant; array/runtime-array/vector/matrix
/// deref to element regardless of value) used to test InBounds validity and the PtrAccessChain
/// post-stride descent.
pub(in crate::passes) fn walk_into_type(
    ctx: &Ctx,
    mut cur: Word,
    indices: &[Operand],
) -> Option<Word> {
    for op in indices {
        let def = type_def_of(ctx, cur)?;
        cur = match def.class.opcode {
            Op::TypeStruct => {
                let Operand::IdRef(idx_id) = op else {
                    return None;
                };
                let cdef = type_def_of(ctx, *idx_id)?;
                if cdef.class.opcode != Op::Constant {
                    return None;
                }
                let member = match cdef.operands.first()? {
                    Operand::LiteralBit32(v) => *v as usize,
                    _ => return None,
                };
                match def.operands.get(member)? {
                    Operand::IdRef(m) => *m,
                    _ => return None,
                }
            }
            Op::TypeArray | Op::TypeRuntimeArray | Op::TypeVector | Op::TypeMatrix => {
                match def.operands.first()? {
                    Operand::IdRef(elem) => *elem,
                    _ => return None,
                }
            }
            _ => return None,
        };
    }
    Some(cur)
}

/// Walk a composite TYPE id through access-chain index operands AS FAR AS POSSIBLE, returning the
/// reached type id and how many indices were consumed. Unlike [`walk_into_type`] (which returns `None`
/// the moment a step indexes a non-composite), this stops gracefully at the first non-composite (or an
/// undecidable step) and reports the partial progress — used to detect a trailing over-index.
pub(in crate::passes) fn walk_into_type_partial(
    ctx: &Ctx,
    mut cur: Word,
    indices: &[Operand],
) -> (Word, usize) {
    for (n, op) in indices.iter().enumerate() {
        let Some(def) = type_def_of(ctx, cur) else {
            return (cur, n);
        };
        let next = match def.class.opcode {
            Op::TypeStruct => {
                let Operand::IdRef(idx_id) = op else {
                    return (cur, n);
                };
                let Some(cdef) = type_def_of(ctx, *idx_id) else {
                    return (cur, n);
                };
                if cdef.class.opcode != Op::Constant {
                    return (cur, n);
                }
                let member = match cdef.operands.first() {
                    Some(Operand::LiteralBit32(v)) => *v as usize,
                    _ => return (cur, n),
                };
                match def.operands.get(member) {
                    Some(Operand::IdRef(m)) => *m,
                    _ => return (cur, n),
                }
            }
            Op::TypeArray | Op::TypeRuntimeArray | Op::TypeVector | Op::TypeMatrix => {
                match def.operands.first() {
                    Some(Operand::IdRef(elem)) => *elem,
                    _ => return (cur, n),
                }
            }
            _ => return (cur, n),
        };
        cur = next;
    }
    (cur, indices.len())
}

/// Drop a TRAILING run of CONSTANT-ZERO over-indices from an INVALID member-access chain. The AIR
/// declares an MPS buffer's element struct from `air.struct_type_info`, which FLATTENS nested
/// single-member wrappers (`{{{uint}}}` → `uint`); the GEP keeps the full member-0 descent
/// (`base [0, N, 0, 0, 0]`), so under Logical addressing the chain reaches the flattened scalar and
/// then over-indexes it with the leftover `0`s — spirv-val "reached non-composite type while indexes
/// still remain". Because every dropped index is member-0 of a composite (byte offset 0), the leftover
/// descent lands at the SAME byte address as the reached scalar; dropping the trailing zeros is
/// byte-IDENTICAL.
///
/// Byte-safe / floor-safe by construction: only chains that are CURRENTLY INVALID (the partial walk
/// stops BEFORE consuming every index) are touched, and only when (1) every leftover index is an
/// `OpConstant 0`, (2) at least one index survives (a 0-index chain is not emitted), and (3) the result
/// pointee type EQUALS the scalar the surviving prefix reaches (so the truncated chain is valid and the
/// pointee is unchanged). If no index survives, the result pointer must have the exact base-pointer
/// type and the identity chain is removed. A banked/valid module (every chain fully walks) never
/// matches. Decides purely from IR structure (type walk + constant check), never a shader name.
pub(in crate::passes) fn drop_overindexed_zero_tail(ctx: &mut Ctx, entry_idx: usize) {
    let value_types = function_value_types(ctx, entry_idx);
    let mut ptr_info: HashMap<Word, (StorageClass, Word)> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if inst.class.opcode == Op::TypePointer {
            if let (Some(id), Some(Operand::StorageClass(s)), Some(Operand::IdRef(p))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*s, *p));
            }
        }
    }

    let mut edits: Vec<(usize, usize, usize)> = Vec::new();
    let mut reinterpret_edits: Vec<(usize, usize, usize, Word, StorageClass, Word)> = Vec::new();
    let mut identities: Vec<(Word, Word)> = Vec::new();
    let mut uses = HashMap::<Word, Vec<(usize, usize)>>::new();
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, instruction) in block.instructions.iter().enumerate() {
            for operand in &instruction.operands {
                if let Operand::IdRef(id) = operand {
                    uses.entry(*id).or_default().push((bi, ii));
                }
            }
        }
    }
    for (bi, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if !matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain) {
                continue;
            }
            let Some(result_type) = inst.result_type else {
                continue;
            };
            let Some(&(storage, result_pointee)) = ptr_info.get(&result_type) else {
                continue;
            };
            let Some(Operand::IdRef(base)) = inst.operands.first() else {
                continue;
            };
            let indices: Vec<Operand> = inst.operands[1..].to_vec();
            if indices.is_empty() {
                continue;
            }
            let Some(base_ptr_ty) = value_types.get(base).copied() else {
                continue;
            };
            let Some(&(_, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            let (reached, consumed) = walk_into_type_partial(ctx, base_pointee, &indices);
            // Only currently-invalid chains (a valid one walks every index).
            if consumed >= indices.len() {
                continue;
            }
            // Every leftover index must be a constant 0 (a member-0 / zero-stride descent).
            let all_zero_tail = indices[consumed..].iter().all(|op| match op {
                Operand::IdRef(id) => const_u32(ctx, *id) == Some(0),
                _ => false,
            });
            if !all_zero_tail {
                continue;
            }
            // The surviving prefix must reach exactly the declared result pointee, AND that pointee must
            // be a direct scalar — so the dropped indices were descending INTO a scalar (provably a
            // byte no-op), not shifting a struct member offset.
            let reached_width = direct_scalar_width(ctx, reached);
            let result_width = direct_scalar_width(ctx, result_pointee);
            if reached != result_pointee
                && reached_width == result_width
                && reached_width.is_some()
                && inst.result_id.is_some_and(|result| {
                    uses.get(&result).is_some_and(|sites| {
                        !sites.is_empty()
                            && sites.iter().all(|&(use_block, use_instruction)| {
                                let user = &ctx.module.functions[entry_idx].blocks[use_block]
                                    .instructions[use_instruction];
                                user.class.opcode == Op::Load
                                    && user.operands.first() == Some(&Operand::IdRef(result))
                                    && user.result_type == Some(result_pointee)
                            })
                    })
                })
            {
                reinterpret_edits.push((
                    bi,
                    ii,
                    consumed,
                    inst.result_id.expect("checked above"),
                    storage,
                    reached,
                ));
                continue;
            }
            if reached != result_pointee || reached_width.is_none() {
                continue;
            }
            if consumed == 0 {
                // A scalar base followed only by zero indices is the base pointer itself. SPIR-V
                // cannot spell the LLVM scalar GEP as an access chain, so remove the identity and
                // forward its uses to the same-typed base.
                if value_types.get(base).copied() == Some(result_type) {
                    if let Some(result) = inst.result_id {
                        identities.push((result, *base));
                    }
                }
            } else {
                edits.push((bi, ii, consumed));
            }
        }
    }
    for (bi, ii, consumed) in edits {
        ctx.module.functions[entry_idx].blocks[bi].instructions[ii]
            .operands
            .truncate(1 + consumed);
    }
    let reinterpret_loads = reinterpret_edits
        .iter()
        .map(|&(_, _, _, pointer, _, reached)| (pointer, reached))
        .collect::<HashMap<_, _>>();
    for &(bi, ii, consumed, _, storage, reached) in &reinterpret_edits {
        let pointer_type = ctx.ty_ptr(storage, reached);
        let instruction = &mut ctx.module.functions[entry_idx].blocks[bi].instructions[ii];
        instruction.result_type = Some(pointer_type);
        instruction.operands.truncate(consumed + 1);
    }
    let mut load_edits = Vec::new();
    for (block_index, block) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (instruction_index, current) in block.instructions.iter().enumerate() {
            if current.class.opcode != Op::Load {
                continue;
            }
            let Some(Operand::IdRef(pointer)) = current.operands.first() else {
                continue;
            };
            let (Some(&reached), Some(original_type), Some(original_result)) = (
                reinterpret_loads.get(pointer),
                current.result_type,
                current.result_id,
            ) else {
                continue;
            };
            load_edits.push((
                block_index,
                instruction_index,
                reached,
                original_type,
                original_result,
            ));
        }
    }
    load_edits.reverse();
    for (block_index, instruction_index, reached, original_type, original_result) in load_edits {
        let loaded = ctx.module.fresh_id();
        let block = &mut ctx.module.functions[entry_idx].blocks[block_index];
        let mut native_load = block.instructions[instruction_index].clone();
        native_load.result_type = Some(reached);
        native_load.result_id = Some(loaded);
        block.instructions[instruction_index] = native_load;
        block.instructions.insert(
            instruction_index + 1,
            Instruction::new(
                Op::Bitcast,
                Some(original_type),
                Some(original_result),
                vec![Operand::IdRef(loaded)],
            ),
        );
    }
    if !identities.is_empty() {
        let replacements = identities.iter().copied().collect::<HashMap<_, _>>();
        ctx.emit_sidecar.remap_ids(&replacements);
        let dead = replacements.keys().copied().collect::<HashSet<_>>();
        let function = &mut ctx.module.functions[entry_idx];
        for (from, to) in identities {
            replace_id_in_function(function, from, to);
        }
        for block in &mut function.blocks {
            block
                .instructions
                .retain(|instruction| instruction.result_id.is_none_or(|id| !dead.contains(&id)));
        }
    }
}

/// Collapse a widened Private scalar load when every observable consumer extracts only byte lane
/// zero. Pointer-selection lowering can temporarily normalize a one-byte private arm to the word
/// carried by a raw-buffer arm, yielding an unrepresentable `i32*` access chain over an `i8`
/// variable. If the word is immediately bitcast to a byte vector and only lane zero is observed,
/// load the declared byte directly and forward those extracts. No adjacent private storage is read.
pub(in crate::passes) fn lower_private_low_byte_word_load(ctx: &mut Ctx, function_idx: usize) {
    let value_types = function_value_types(ctx, function_idx);
    let mut ptr_info = HashMap::<Word, (StorageClass, Word)>::new();
    for instruction in ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
    {
        if instruction.class.opcode == Op::TypePointer {
            if let (Some(id), Some(Operand::StorageClass(storage)), Some(Operand::IdRef(pointee))) = (
                instruction.result_id,
                instruction.operands.first(),
                instruction.operands.get(1),
            ) {
                ptr_info.insert(id, (*storage, *pointee));
            }
        }
    }
    let function = &ctx.module.functions[function_idx];
    let mut definitions = HashMap::<Word, (usize, usize)>::new();
    let mut users = HashMap::<Word, Vec<(usize, usize)>>::new();
    for (block_index, block) in function.blocks.iter().enumerate() {
        for (instruction_index, instruction) in block.instructions.iter().enumerate() {
            if let Some(id) = instruction.result_id {
                definitions.insert(id, (block_index, instruction_index));
            }
            for operand in &instruction.operands {
                if let Operand::IdRef(id) = operand {
                    users
                        .entry(*id)
                        .or_default()
                        .push((block_index, instruction_index));
                }
            }
        }
    }

    struct Edit {
        block: usize,
        instruction: usize,
        base: Word,
        scalar_type: Word,
        replacement: Word,
        extracts: Vec<Word>,
        dead: HashSet<Word>,
    }
    let mut edits = Vec::new();
    for (block_index, block) in function.blocks.iter().enumerate() {
        for (instruction_index, chain) in block.instructions.iter().enumerate() {
            if !matches!(
                chain.class.opcode,
                Op::AccessChain | Op::InBoundsAccessChain
            ) {
                continue;
            }
            let (Some(chain_id), Some(chain_type), Some(Operand::IdRef(base))) =
                (chain.result_id, chain.result_type, chain.operands.first())
            else {
                continue;
            };
            if !chain.operands[1..].iter().all(
                |operand| matches!(operand, Operand::IdRef(id) if const_u32(ctx, *id) == Some(0)),
            ) {
                continue;
            }
            let Some(base_type) = value_types.get(base).copied() else {
                continue;
            };
            let (
                Some((StorageClass::Private, scalar_type)),
                Some((StorageClass::Private, wide_type)),
            ) = (
                ptr_info.get(&base_type).copied(),
                ptr_info.get(&chain_type).copied(),
            )
            else {
                continue;
            };
            let (Some(scalar_bits), Some(wide_bits)) = (
                direct_scalar_width(ctx, scalar_type),
                direct_scalar_width(ctx, wide_type),
            ) else {
                continue;
            };
            if scalar_bits != 8 || wide_bits <= scalar_bits {
                continue;
            }
            let Some(chain_users) = users.get(&chain_id) else {
                continue;
            };
            if chain_users.is_empty() {
                continue;
            }
            let mut extracts = Vec::new();
            let mut dead = HashSet::from([chain_id]);
            let mut valid = true;
            for &(load_block, load_index) in chain_users {
                let load = &function.blocks[load_block].instructions[load_index];
                let Some(load_id) = load.result_id else {
                    valid = false;
                    break;
                };
                if load.class.opcode != Op::Load || load.result_type != Some(wide_type) {
                    valid = false;
                    break;
                }
                dead.insert(load_id);
                let Some(load_users) = users.get(&load_id) else {
                    valid = false;
                    break;
                };
                for &(cast_block, cast_index) in load_users {
                    let cast = &function.blocks[cast_block].instructions[cast_index];
                    let Some(cast_id) = cast.result_id else {
                        valid = false;
                        break;
                    };
                    let vector_matches = cast.result_type.and_then(|ty| {
                        let definition = type_def_of(ctx, ty)?;
                        match definition.operands.as_slice() {
                            [Operand::IdRef(element), Operand::LiteralBit32(lanes)]
                                if definition.class.opcode == Op::TypeVector =>
                            {
                                Some((*element, *lanes))
                            }
                            _ => None,
                        }
                    }) == Some((scalar_type, wide_bits / scalar_bits));
                    if cast.class.opcode != Op::Bitcast || !vector_matches {
                        valid = false;
                        break;
                    }
                    dead.insert(cast_id);
                    let Some(cast_users) = users.get(&cast_id) else {
                        valid = false;
                        break;
                    };
                    for &(extract_block, extract_index) in cast_users {
                        let extract = &function.blocks[extract_block].instructions[extract_index];
                        if extract.class.opcode != Op::CompositeExtract
                            || extract.result_type != Some(scalar_type)
                            || extract.operands.get(1) != Some(&Operand::LiteralBit32(0))
                        {
                            valid = false;
                            break;
                        }
                        let Some(extract_id) = extract.result_id else {
                            valid = false;
                            break;
                        };
                        extracts.push(extract_id);
                        dead.insert(extract_id);
                    }
                }
            }
            if valid && !extracts.is_empty() {
                edits.push(Edit {
                    block: block_index,
                    instruction: instruction_index,
                    base: *base,
                    scalar_type,
                    replacement: 0,
                    extracts,
                    dead,
                });
            }
        }
    }
    for edit in &mut edits {
        edit.replacement = ctx.module.fresh_id();
    }
    for edit in &edits {
        ctx.module.functions[function_idx].blocks[edit.block]
            .instructions
            .insert(
                edit.instruction,
                Instruction::new(
                    Op::Load,
                    Some(edit.scalar_type),
                    Some(edit.replacement),
                    vec![Operand::IdRef(edit.base)],
                ),
            );
    }
    let replacements = edits
        .iter()
        .flat_map(|edit| {
            edit.extracts
                .iter()
                .map(move |extract| (*extract, edit.replacement))
        })
        .collect::<HashMap<_, _>>();
    if replacements.is_empty() {
        return;
    }
    ctx.emit_sidecar.remap_ids(&replacements);
    let dead = edits
        .into_iter()
        .flat_map(|edit| edit.dead)
        .collect::<HashSet<_>>();
    let function = &mut ctx.module.functions[function_idx];
    for (from, to) in replacements {
        replace_id_in_function(function, from, to);
    }
    for block in &mut function.blocks {
        block
            .instructions
            .retain(|instruction| instruction.result_id.is_none_or(|id| !dead.contains(&id)));
    }
}

/// Re-root an over-index of a demoted array-element-0 pointer back onto the array.
///
/// metal2vulkan sometimes lowers an AIR `getelementptr [K x T], ptr %arr, i64 0, i64 %i` (element `%i` of a
/// function/threadgroup/device array) in TWO steps: first an element-0 pointer `%p = AC %arr %uint_0`
/// (a `_ptr_SC_T` to element 0), then the dynamic part `%r = AC %p %i` — which OVER-indexes the scalar
/// `%p` points at ("OpInBoundsAccessChain reached non-composite type while indexes still remain"). The
/// element-0 pointer may also be merged through one or more `OpPhi`/`OpCopyObject` before the
/// over-index (e.g. a loop-carried accumulator pointer). Since `%p` — and every phi arm it flows from —
/// provably equals `&%arr[0]`, the address `&(&%arr[0])[%i]` is byte-IDENTICAL to `&%arr[%i]`; the pass
/// re-roots the over-indexing chain onto `%arr` with the SAME single dynamic index.
///
/// **Byte-EXACT by construction**: element 0 + offset `i` is element `i` — the SAME byte address, for
/// ANY storage class (the array is element-contiguous; for StorageBuffer the `ArrayStride` is already
/// on the array type). This recovers the array provenance the two-step lowering lost — the size `K` is
/// NOT lost (the `[K x T]` array variable is declared; this is the demotion the prior BVH/W-c note
/// flagged, here resolved for the case where the array IS still declared and provenance converges).
/// **Floor-SAFE by construction**: fires ONLY on a CURRENTLY-INVALID chain (a base pointing at a direct
/// SCALAR equal to the result pointee, with exactly one remaining index — a valid module never
/// over-indexes a scalar) whose base provenance, traced through `OpPhi` (every incoming must converge)
/// and `OpCopyObject` and element-0 `AC`s, resolves to element 0 of ONE declared `[K x T]` array
/// variable (same storage class) whose element type equals the result pointee. Any divergence, a
/// non-element-0 chain, or an unknown provenance leaf leaves the chain untouched. Decides purely from
/// IR structure (provenance trace + type compare), never a shader name.
pub(in crate::passes) fn reroot_demoted_array_element_overindex(ctx: &mut Ctx, entry_idx: usize) {
    let value_types = function_value_types(ctx, entry_idx);
    let mut ptr_info: HashMap<Word, (StorageClass, Word)> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if inst.class.opcode == Op::TypePointer {
            if let (Some(id), Some(Operand::StorageClass(s)), Some(Operand::IdRef(p))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*s, *p));
            }
        }
    }

    let func = ctx.module.functions[entry_idx].clone();
    let mut edits: Vec<(usize, usize, Word)> = Vec::new();
    for (bi, block) in func.blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if !matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain) {
                continue;
            }
            // Exactly one index (the dynamic element index) — the 1-D demoted-array shape.
            if inst.operands.len() != 2 {
                continue;
            }
            let Some(result_type) = inst.result_type else {
                continue;
            };
            let Some(&(_, result_pointee)) = ptr_info.get(&result_type) else {
                continue;
            };
            let Some(Operand::IdRef(base)) = inst.operands.first() else {
                continue;
            };
            // CURRENTLY INVALID: base points at a direct SCALAR equal to the result pointee (a pure
            // re-root, not a reinterpret) — indexing a scalar over-runs.
            let Some(base_ptr_ty) = value_types.get(base).copied() else {
                continue;
            };
            let Some(&(base_sc, base_pointee)) = ptr_info.get(&base_ptr_ty) else {
                continue;
            };
            if base_pointee != result_pointee || direct_scalar_width(ctx, base_pointee).is_none() {
                continue;
            }
            // Provenance must converge to element 0 of ONE declared `[K x base_pointee]` array.
            let Some(array_id) =
                trace_to_array_element_zero(ctx, &func, *base, base_pointee, base_sc, &ptr_info)
            else {
                continue;
            };
            edits.push((bi, ii, array_id));
        }
    }
    for (bi, ii, array_id) in edits {
        // Re-root: replace the base operand with the array variable; the single index now selects the
        // array element (byte-identical to element 0 + that offset). Opcode/index preserved.
        ctx.module.functions[entry_idx].blocks[bi].instructions[ii].operands[0] =
            Operand::IdRef(array_id);
    }
}
