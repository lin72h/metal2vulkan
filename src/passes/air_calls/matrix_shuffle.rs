//! Matrix and subgroup shuffle AIR call lowering.

use super::*;

/// True if `ty` is `OpTypeFloat 16`.
pub(in crate::passes) fn is_half_scalar(ctx: &Ctx, ty: Word) -> bool {
    type_def_of(ctx, ty)
        .map(|d| {
            d.class.opcode == Op::TypeFloat
                && d.operands.first() == Some(&Operand::LiteralBit32(16))
        })
        .unwrap_or(false)
}

pub(in crate::passes) fn is_half_scalar_or_vector(ctx: &Ctx, ty: Word) -> bool {
    is_half_scalar(ctx, ty) || is_half_vector(ctx, ty)
}

pub(in crate::passes) fn is_void_type(ctx: &Ctx, ty: Word) -> bool {
    type_def_of(ctx, ty)
        .map(|def| def.class.opcode == Op::TypeVoid)
        .unwrap_or(false)
}

pub(in crate::passes) fn is_bool_type(ctx: &Ctx, ty: Word) -> bool {
    type_def_of(ctx, ty)
        .map(|def| def.class.opcode == Op::TypeBool)
        .unwrap_or(false)
}

pub(in crate::passes) fn is_command_encoder_helper(name: &str) -> bool {
    name.starts_with("air.") && name.contains("_command")
}

// `air.simdgroup_matrix_8x8_multiply_accumulate(A, B, C) = C + A·B`, the documented row-major 8x8
// matmul. Result/A/C are v64f32; B is v64f32 OR v64f16 (the mixed-precision `.v64f16.` variant) — B
// half lanes are FConvert'd up to f32 before the multiply. Decided from operand types, not the mangled
// name.
pub(in crate::passes) fn lower_simdgroup_matrix_8x8_mac(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if args.len() != 3 {
        return Err(format!(
            "air.simdgroup_matrix_8x8_multiply_accumulate expects 3 operands, got {}",
            args.len()
        ));
    }
    let (elem, lanes) = composite_shape(ctx, rty)
        .ok_or_else(|| "air.simdgroup_matrix result is not a 64-lane composite".to_string())?;
    if lanes != 64 || !is_f32_scalar(ctx, elem) {
        return Err("air.simdgroup_matrix result is not v64f32".to_string());
    }
    // B operand element type: f32 directly, or f16 needing a widen.
    let b_ty = value_result_type(ctx, args[1])
        .ok_or_else(|| "air.simdgroup_matrix B operand has no type".to_string())?;
    let (b_elem, b_lanes) = composite_shape(ctx, b_ty)
        .ok_or_else(|| "air.simdgroup_matrix B operand is not a 64-lane composite".to_string())?;
    if b_lanes != 64 {
        return Err("air.simdgroup_matrix B operand is not 64-lane".to_string());
    }
    let b_is_half = is_half_scalar(ctx, b_elem);
    if !b_is_half && !is_f32_scalar(ctx, b_elem) {
        return Err("air.simdgroup_matrix B operand is neither f32 nor f16".to_string());
    }

    let mut insts = Vec::with_capacity(64 * (1 + 8 * 5) + 1);
    let mut result_lanes = Vec::with_capacity(64);
    for row in 0..8 {
        for col in 0..8 {
            let mut acc = composite_extract(ctx, &mut insts, elem, args[2], row * 8 + col);
            for k in 0..8 {
                let a = composite_extract(ctx, &mut insts, elem, args[0], row * 8 + k);
                let b_raw = composite_extract(ctx, &mut insts, b_elem, args[1], k * 8 + col);
                // Widen a half B lane to f32 so the multiply matches the f32 accumulator.
                let b = if b_is_half {
                    let widened = ctx.module.fresh_id();
                    insts.push(Instruction::new(
                        Op::FConvert,
                        Some(elem),
                        Some(widened),
                        vec![Operand::IdRef(b_raw)],
                    ));
                    widened
                } else {
                    b_raw
                };
                let product = ctx.module.fresh_id();
                insts.push(Instruction::new(
                    Op::FMul,
                    Some(elem),
                    Some(product),
                    vec![Operand::IdRef(a), Operand::IdRef(b)],
                ));
                let sum = ctx.module.fresh_id();
                insts.push(Instruction::new(
                    Op::FAdd,
                    Some(elem),
                    Some(sum),
                    vec![Operand::IdRef(acc), Operand::IdRef(product)],
                ));
                acc = sum;
            }
            result_lanes.push(Operand::IdRef(acc));
        }
    }
    insts.push(Instruction::new(
        Op::CompositeConstruct,
        Some(rty),
        Some(res),
        result_lanes,
    ));
    Ok(insts)
}

// `air.simdgroup_matrix_8x8_init_diag(v)` = an 8x8 matrix with `v` on the diagonal, 0 elsewhere.
// Modeled as a 64-lane row-major composite: lane r*8+c = (r==c) ? v : 0.
pub(in crate::passes) fn lower_simdgroup_matrix_8x8_init_diag(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if args.len() != 1 {
        return Err(format!(
            "air.simdgroup_matrix_8x8_init_diag expects 1 operand, got {}",
            args.len()
        ));
    }
    let (elem, lanes) = composite_shape(ctx, rty).ok_or_else(|| {
        "air.simdgroup_matrix init_diag result is not a 64-lane composite".to_string()
    })?;
    if lanes != 64 {
        return Err("air.simdgroup_matrix init_diag result is not 64-lane".to_string());
    }
    // Zero of the element type (f32 here).
    let zero = if is_half_scalar(ctx, elem) {
        ctx.const_half(0.0)
    } else {
        ctx.const_float(0.0)
    };
    let mut lanes_ops = Vec::with_capacity(64);
    for row in 0..8u32 {
        for col in 0..8u32 {
            if row == col {
                lanes_ops.push(Operand::IdRef(args[0]));
            } else {
                lanes_ops.push(Operand::IdRef(zero));
            }
        }
    }
    Ok(vec![Instruction::new(
        Op::CompositeConstruct,
        Some(rty),
        Some(res),
        lanes_ops,
    )])
}

// Element pointer for matrix cell (row, col) over a row-major device tile: `base[row*epr + col]`,
// where `base` is the tile-origin element pointer and `epr` the runtime leading dimension. Emits an
// `OpPtrAccessChain` (legal on a StorageBuffer element pointer under VariablePointersStorageBuffer,
// which finalize declares). Returns the element pointer id.
pub(in crate::passes) fn simdgroup_matrix_elem_ptr(
    ctx: &mut Ctx,
    insts: &mut Vec<Instruction>,
    ptr_ty: Word,
    base: Word,
    idx_ty: Word,
    epr: Word,
    row: u32,
    col: u32,
) -> Word {
    let row_c = ctx.const_int_of(idx_ty, row as i64);
    let row_off = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IMul,
        Some(idx_ty),
        Some(row_off),
        vec![Operand::IdRef(row_c), Operand::IdRef(epr)],
    ));
    let col_c = ctx.const_int_of(idx_ty, col as i64);
    let index = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(idx_ty),
        Some(index),
        vec![Operand::IdRef(row_off), Operand::IdRef(col_c)],
    ));
    let elem_ptr = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::PtrAccessChain,
        Some(ptr_ty),
        Some(elem_ptr),
        vec![Operand::IdRef(base), Operand::IdRef(index)],
    ));
    elem_ptr
}

// Leading dimension (`elements_per_row`) = component 0 of the first descriptor vector arg. See the
// AIR ABI decode in the simdgroup_matrix kb note.
pub(in crate::passes) fn simdgroup_matrix_leading_dim(
    ctx: &mut Ctx,
    insts: &mut Vec<Instruction>,
    desc_vec: Word,
) -> Result<(Word, Word), String> {
    let vec_ty = value_result_type(ctx, desc_vec)
        .ok_or_else(|| "simdgroup_matrix descriptor vector has no type".to_string())?;
    let (elem_ty, lanes) = composite_shape(ctx, vec_ty)
        .ok_or_else(|| "simdgroup_matrix descriptor is not a vector".to_string())?;
    if lanes < 1 {
        return Err("simdgroup_matrix descriptor vector is empty".to_string());
    }
    let epr = composite_extract(ctx, insts, elem_ty, desc_vec, 0);
    Ok((epr, elem_ty))
}

// `air.simdgroup_matrix_8x8_load(ptr, <epr,8>, <1,epr>, origin=0)` -> a 64-lane row-major composite
// gathered from device memory: lane r*8+c = base[r*epr + c].
pub(in crate::passes) fn lower_simdgroup_matrix_8x8_load(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if args.len() < 2 {
        return Err(format!(
            "air.simdgroup_matrix_8x8_load expects >=2 operands, got {}",
            args.len()
        ));
    }
    let (elem, lanes) = composite_shape(ctx, rty)
        .ok_or_else(|| "air.simdgroup_matrix load result is not a 64-lane composite".to_string())?;
    if lanes != 64 {
        return Err("air.simdgroup_matrix load result is not 64-lane".to_string());
    }
    let base = args[0];
    let ptr_ty = value_result_type(ctx, base)
        .ok_or_else(|| "air.simdgroup_matrix load pointer has no type".to_string())?;
    let pointee = pointer_pointee_type(ctx, ptr_ty)
        .ok_or_else(|| "air.simdgroup_matrix load pointer is not a pointer".to_string())?;
    if pointee != elem {
        return Err(
            "air.simdgroup_matrix load pointee does not match result element type".to_string(),
        );
    }
    let mut insts = Vec::new();
    let (epr, idx_ty) = simdgroup_matrix_leading_dim(ctx, &mut insts, args[1])?;
    let mut result_lanes = Vec::with_capacity(64);
    for row in 0..8u32 {
        for col in 0..8u32 {
            let elem_ptr =
                simdgroup_matrix_elem_ptr(ctx, &mut insts, ptr_ty, base, idx_ty, epr, row, col);
            let value = ctx.module.fresh_id();
            insts.push(Instruction::new(
                Op::Load,
                Some(elem),
                Some(value),
                vec![Operand::IdRef(elem_ptr)],
            ));
            result_lanes.push(Operand::IdRef(value));
        }
    }
    insts.push(Instruction::new(
        Op::CompositeConstruct,
        Some(rty),
        Some(res),
        result_lanes,
    ));
    Ok(insts)
}

// `air.simdgroup_matrix_8x8_store(M, ptr, <epr,8>, <1,epr>, origin=0)` -> scatter the 64-lane
// row-major matrix into device memory: base[r*epr + c] = M[r*8+c].
pub(in crate::passes) fn lower_simdgroup_matrix_8x8_store(
    ctx: &mut Ctx,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if args.len() < 3 {
        return Err(format!(
            "air.simdgroup_matrix_8x8_store expects >=3 operands, got {}",
            args.len()
        ));
    }
    let matrix = args[0];
    let base = args[1];
    let mat_ty = value_result_type(ctx, matrix)
        .ok_or_else(|| "air.simdgroup_matrix store value has no type".to_string())?;
    let (elem, lanes) = composite_shape(ctx, mat_ty)
        .ok_or_else(|| "air.simdgroup_matrix store value is not a 64-lane composite".to_string())?;
    if lanes != 64 {
        return Err("air.simdgroup_matrix store value is not 64-lane".to_string());
    }
    let ptr_ty = value_result_type(ctx, base)
        .ok_or_else(|| "air.simdgroup_matrix store pointer has no type".to_string())?;
    let pointee = pointer_pointee_type(ctx, ptr_ty)
        .ok_or_else(|| "air.simdgroup_matrix store pointer is not a pointer".to_string())?;
    if pointee != elem {
        return Err(
            "air.simdgroup_matrix store pointee does not match value element type".to_string(),
        );
    }
    let mut insts = Vec::new();
    let (epr, idx_ty) = simdgroup_matrix_leading_dim(ctx, &mut insts, args[2])?;
    for row in 0..8u32 {
        for col in 0..8u32 {
            let value = composite_extract(ctx, &mut insts, elem, matrix, row * 8 + col);
            let elem_ptr =
                simdgroup_matrix_elem_ptr(ctx, &mut insts, ptr_ty, base, idx_ty, epr, row, col);
            insts.push(Instruction::new(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(elem_ptr), Operand::IdRef(value)],
            ));
        }
    }
    Ok(insts)
}

pub(in crate::passes) fn composite_extract(
    ctx: &mut Ctx,
    insts: &mut Vec<Instruction>,
    elem: Word,
    value: Word,
    lane: u32,
) -> Word {
    let result = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::CompositeExtract,
        Some(elem),
        Some(result),
        vec![Operand::IdRef(value), Operand::LiteralBit32(lane)],
    ));
    result
}

pub(in crate::passes) fn subgroup_shuffle_index_u32(
    ctx: &mut Ctx,
    value: Word,
    insts: &mut Vec<Instruction>,
) -> Result<Word, String> {
    let Some(ty) = value_result_type(ctx, value) else {
        return Err("subgroup shuffle index has no type".to_string());
    };
    let Some(def) = type_def_of(ctx, ty) else {
        return Err("subgroup shuffle index type is undefined".to_string());
    };
    if def.class.opcode != Op::TypeInt {
        return Err("subgroup shuffle index is not an integer".to_string());
    }
    if def.operands.first() == Some(&Operand::LiteralBit32(32)) {
        return Ok(value);
    }
    let uint = ctx.ty_uint();
    let converted = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UConvert,
        Some(uint),
        Some(converted),
        vec![Operand::IdRef(value)],
    ));
    Ok(converted)
}

pub(in crate::passes) fn lower_quad_shuffle(
    ctx: &mut Ctx,
    result: Word,
    result_type: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let mut insts = Vec::new();
    let requested_lane = subgroup_shuffle_index_u32(ctx, args[1], &mut insts)?;
    let uint = ctx.ty_uint();
    let subgroup_lane = subgroup_lane_index_u32(ctx, &mut insts);
    let quad_base = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(quad_base),
        vec![
            Operand::IdRef(subgroup_lane),
            Operand::IdRef(ctx.const_uint(!3u32)),
        ],
    ));
    let quad_local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(quad_local),
        vec![
            Operand::IdRef(requested_lane),
            Operand::IdRef(ctx.const_uint(3)),
        ],
    ));
    let source_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(source_lane),
        vec![Operand::IdRef(quad_base), Operand::IdRef(quad_local)],
    ));
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(result),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[0]),
            Operand::IdRef(source_lane),
        ],
    ));
    Ok(insts)
}

/// `air.quad_shuffle_rotate_down(value, delta)` rotates values DOWN within the 4-lane quad, wrapping:
/// result on lane L = `value` from lane `(L + delta) % 4`. Verified bit-exact on Apple Metal (lane L,
/// delta d -> lane (L+d)%4 for all L,d in 0..3). Lowered as a masked `GroupNonUniformShuffle`:
/// `source_lane = (quad_base) + ((lane & 3) + delta) % 4`.
pub(in crate::passes) fn lower_quad_shuffle_rotate_down(
    ctx: &mut Ctx,
    result: Word,
    result_type: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let mut insts = Vec::new();
    let delta = subgroup_shuffle_index_u32(ctx, args[1], &mut insts)?;
    let uint = ctx.ty_uint();
    let subgroup_lane = subgroup_lane_index_u32(ctx, &mut insts);
    let quad_base = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(quad_base),
        vec![
            Operand::IdRef(subgroup_lane),
            Operand::IdRef(ctx.const_uint(!3u32)),
        ],
    ));
    let local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(local),
        vec![
            Operand::IdRef(subgroup_lane),
            Operand::IdRef(ctx.const_uint(3)),
        ],
    ));
    let local_plus_delta = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(local_plus_delta),
        vec![Operand::IdRef(local), Operand::IdRef(delta)],
    ));
    // (local + delta) % 4 — the quad size is a power of two, so a bitwise AND is the modulo.
    let source_local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(source_local),
        vec![
            Operand::IdRef(local_plus_delta),
            Operand::IdRef(ctx.const_uint(3)),
        ],
    ));
    let source_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(source_lane),
        vec![Operand::IdRef(quad_base), Operand::IdRef(source_local)],
    ));
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(result),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[0]),
            Operand::IdRef(source_lane),
        ],
    ));
    Ok(insts)
}

/// `air.quad_shuffle_up(value, delta)` reads `value` from lane `local - delta` within the 4-lane quad;
/// when `local - delta` underflows the quad, Apple Metal returns the lane's OWN value (verified: lane
/// 0 up(1) and lane 1 up(2) return their own value, not a cross-quad lane). Lowered quad-boundary-safe
/// via `quad_base = lane & ~3`: the source local index is clamped to the lane's own local when it would
/// underflow, so the shuffle reads a valid in-quad lane and yields the own value out of bounds.
pub(in crate::passes) fn lower_quad_shuffle_up(
    ctx: &mut Ctx,
    result: Word,
    result_type: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let mut insts = Vec::new();
    let delta = subgroup_shuffle_index_u32(ctx, args[1], &mut insts)?;
    let uint = ctx.ty_uint();
    let bool_ty = ctx.ty_bool();
    let subgroup_lane = subgroup_lane_index_u32(ctx, &mut insts);
    let quad_base = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(quad_base),
        vec![
            Operand::IdRef(subgroup_lane),
            Operand::IdRef(ctx.const_uint(!3u32)),
        ],
    ));
    let local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(local),
        vec![
            Operand::IdRef(subgroup_lane),
            Operand::IdRef(ctx.const_uint(3)),
        ],
    ));
    let in_bounds = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UGreaterThanEqual,
        Some(bool_ty),
        Some(in_bounds),
        vec![Operand::IdRef(local), Operand::IdRef(delta)],
    ));
    let lowered = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ISub,
        Some(uint),
        Some(lowered),
        vec![Operand::IdRef(local), Operand::IdRef(delta)],
    ));
    // Out of bounds -> read own local lane, which yields the lane's own value (Metal's behaviour).
    let source_local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::Select,
        Some(uint),
        Some(source_local),
        vec![
            Operand::IdRef(in_bounds),
            Operand::IdRef(lowered),
            Operand::IdRef(local),
        ],
    ));
    let source_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(source_lane),
        vec![Operand::IdRef(quad_base), Operand::IdRef(source_local)],
    ));
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(result),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[0]),
            Operand::IdRef(source_lane),
        ],
    ));
    Ok(insts)
}

pub(in crate::passes) fn lower_simd_shuffle_and_fill_down(
    ctx: &mut Ctx,
    result: Word,
    result_type: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let mut insts = Vec::new();
    let delta = subgroup_shuffle_index_u32(ctx, args[2], &mut insts)?;
    let modulo = subgroup_shuffle_index_u32(ctx, args[3], &mut insts)?;
    let uint = ctx.ty_uint();
    let modulo_is_zero = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IEqual,
        Some(ctx.ty_bool()),
        Some(modulo_is_zero),
        vec![Operand::IdRef(modulo), Operand::IdRef(ctx.const_uint(0))],
    ));
    let safe_modulo = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::Select,
        Some(uint),
        Some(safe_modulo),
        vec![
            Operand::IdRef(modulo_is_zero),
            Operand::IdRef(ctx.const_uint(1)),
            Operand::IdRef(modulo),
        ],
    ));
    let lane = subgroup_lane_index_u32(ctx, &mut insts);
    let local_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UMod,
        Some(uint),
        Some(local_lane),
        vec![Operand::IdRef(lane), Operand::IdRef(safe_modulo)],
    ));
    let cluster_base = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ISub,
        Some(uint),
        Some(cluster_base),
        vec![Operand::IdRef(lane), Operand::IdRef(local_lane)],
    ));
    let source_local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(source_local),
        vec![Operand::IdRef(local_lane), Operand::IdRef(delta)],
    ));
    let in_bounds = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ULessThan,
        Some(ctx.ty_bool()),
        Some(in_bounds),
        vec![Operand::IdRef(source_local), Operand::IdRef(safe_modulo)],
    ));
    let wrapped_local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UMod,
        Some(uint),
        Some(wrapped_local),
        vec![Operand::IdRef(source_local), Operand::IdRef(safe_modulo)],
    ));
    let data_local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::Select,
        Some(uint),
        Some(data_local),
        vec![
            Operand::IdRef(in_bounds),
            Operand::IdRef(source_local),
            Operand::IdRef(wrapped_local),
        ],
    ));
    let data_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(data_lane),
        vec![Operand::IdRef(cluster_base), Operand::IdRef(data_local)],
    ));
    let fill_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(fill_lane),
        vec![Operand::IdRef(cluster_base), Operand::IdRef(wrapped_local)],
    ));
    let data = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(data),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[0]),
            Operand::IdRef(data_lane),
        ],
    ));
    let fill = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(fill),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[1]),
            Operand::IdRef(fill_lane),
        ],
    ));
    insts.push(Instruction::new(
        Op::Select,
        Some(result_type),
        Some(result),
        vec![
            Operand::IdRef(in_bounds),
            Operand::IdRef(data),
            Operand::IdRef(fill),
        ],
    ));
    Ok(insts)
}

/// `simd_shuffle_and_fill_up(data, fill, delta, modulo)`: within each `modulo`-wide cluster, lane
/// `l` reads `data[l-delta]`; lanes with `l < delta` instead read `fill[l-delta+modulo]`. The
/// shuffle source index `cluster_base + ((l + modulo - (delta%modulo)) % modulo)` is identical for
/// both the data and fill reads (it wraps to the in-cluster position), and a final `Select` on
/// `l >= delta` picks data vs fill — mirroring `lower_simd_shuffle_and_fill_down` with the up
/// direction. (Empirically matched against Apple Metal across lane/delta/modulo combinations.)
pub(in crate::passes) fn lower_simd_shuffle_and_fill_up(
    ctx: &mut Ctx,
    result: Word,
    result_type: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let uint = ctx.ty_uint();
    let bool_ty = ctx.ty_bool();
    let mut insts = Vec::new();
    let delta = subgroup_shuffle_index_u32(ctx, args[2], &mut insts)?;
    let modulo = subgroup_shuffle_index_u32(ctx, args[3], &mut insts)?;
    // safe_modulo = modulo == 0 ? 1 : modulo (avoid UMod-by-zero).
    let modulo_is_zero = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IEqual,
        Some(bool_ty),
        Some(modulo_is_zero),
        vec![Operand::IdRef(modulo), Operand::IdRef(ctx.const_uint(0))],
    ));
    let safe_modulo = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::Select,
        Some(uint),
        Some(safe_modulo),
        vec![
            Operand::IdRef(modulo_is_zero),
            Operand::IdRef(ctx.const_uint(1)),
            Operand::IdRef(modulo),
        ],
    ));
    let lane = subgroup_lane_index_u32(ctx, &mut insts);
    let local_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UMod,
        Some(uint),
        Some(local_lane),
        vec![Operand::IdRef(lane), Operand::IdRef(safe_modulo)],
    ));
    let cluster_base = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ISub,
        Some(uint),
        Some(cluster_base),
        vec![Operand::IdRef(lane), Operand::IdRef(local_lane)],
    ));
    // delta_mod = delta % safe_modulo (bounds the arithmetic so the lift stays in [0, 2*modulo)).
    let delta_mod = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UMod,
        Some(uint),
        Some(delta_mod),
        vec![Operand::IdRef(delta), Operand::IdRef(safe_modulo)],
    ));
    // in_bounds = local_lane >= delta_mod (no wrap needed -> use data).
    let in_bounds = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UGreaterThanEqual,
        Some(bool_ty),
        Some(in_bounds),
        vec![Operand::IdRef(local_lane), Operand::IdRef(delta_mod)],
    ));
    // lifted = local_lane + safe_modulo - delta_mod, in [0, 2*modulo); wrapped = lifted % modulo is
    // the in-cluster source position for both data (when in bounds) and fill (when wrapped).
    let lifted_hi = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(lifted_hi),
        vec![Operand::IdRef(local_lane), Operand::IdRef(safe_modulo)],
    ));
    let lifted = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ISub,
        Some(uint),
        Some(lifted),
        vec![Operand::IdRef(lifted_hi), Operand::IdRef(delta_mod)],
    ));
    let wrapped_local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UMod,
        Some(uint),
        Some(wrapped_local),
        vec![Operand::IdRef(lifted), Operand::IdRef(safe_modulo)],
    ));
    let src_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(src_lane),
        vec![Operand::IdRef(cluster_base), Operand::IdRef(wrapped_local)],
    ));
    let data = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(data),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[0]),
            Operand::IdRef(src_lane),
        ],
    ));
    let fill = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(result_type),
        Some(fill),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(args[1]),
            Operand::IdRef(src_lane),
        ],
    ));
    insts.push(Instruction::new(
        Op::Select,
        Some(result_type),
        Some(result),
        vec![
            Operand::IdRef(in_bounds),
            Operand::IdRef(data),
            Operand::IdRef(fill),
        ],
    ));
    Ok(insts)
}

pub(in crate::passes) fn subgroup_lane_index_u32(
    ctx: &mut Ctx,
    insts: &mut Vec<Instruction>,
) -> Word {
    let uint = ctx.ty_uint();
    let var = local_invocation_index_input_var(ctx, uint);
    let local_index = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::Load,
        Some(uint),
        Some(local_index),
        vec![Operand::IdRef(var)],
    ));
    let mask = ctx.const_uint(31);
    let lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(lane),
        vec![Operand::IdRef(local_index), Operand::IdRef(mask)],
    ));
    lane
}

pub(in crate::passes) fn local_invocation_index_input_var(ctx: &mut Ctx, uint: Word) -> Word {
    let key = SynthCacheKey::LocalInvocationIndexInputVar;
    if let Some(&var) = ctx.synth_cache.get(&key) {
        return var;
    }
    if let Some(var) = existing_builtin_input_var(ctx, BuiltIn::LocalInvocationIndex, uint) {
        ctx.synth_cache.insert(key, var);
        return var;
    }
    let ptr_ty = ctx.ty_ptr(StorageClass::Input, uint);
    let var = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::Variable,
        Some(ptr_ty),
        Some(var),
        vec![Operand::StorageClass(StorageClass::Input)],
    ));
    ctx.module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(var),
            Operand::Decoration(Decoration::BuiltIn),
            Operand::BuiltIn(BuiltIn::LocalInvocationIndex),
        ],
    ));
    ctx.interface.push(var);
    ctx.synth_cache.insert(key, var);
    var
}

pub(in crate::passes) fn existing_builtin_input_var(
    ctx: &Ctx,
    builtin: BuiltIn,
    pointee: Word,
) -> Option<Word> {
    ctx.module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
        .find_map(|inst| {
            if inst.class.opcode != Op::Variable {
                return None;
            }
            if inst.operands.first() != Some(&Operand::StorageClass(StorageClass::Input)) {
                return None;
            }
            let var = inst.result_id?;
            let ptr_ty = inst.result_type?;
            let ptr_def = type_def_of(ctx, ptr_ty)?;
            if ptr_def.class.opcode != Op::TypePointer
                || ptr_def.operands.first() != Some(&Operand::StorageClass(StorageClass::Input))
                || ptr_def.operands.get(1) != Some(&Operand::IdRef(pointee))
            {
                return None;
            }
            has_builtin_decoration(ctx, var, builtin).then_some(var)
        })
}

pub(in crate::passes) fn has_builtin_decoration(ctx: &Ctx, var: Word, builtin: BuiltIn) -> bool {
    ctx.module.annotations.iter().any(|inst| {
        inst.class.opcode == Op::Decorate
            && inst.operands.first() == Some(&Operand::IdRef(var))
            && inst.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && inst.operands.get(2) == Some(&Operand::BuiltIn(builtin))
    })
}
