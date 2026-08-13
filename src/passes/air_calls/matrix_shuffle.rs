//! Matrix and subgroup shuffle AIR call lowering.

use super::*;
use crate::air_intrinsics::{matrix16_intrinsic, Matrix16Element};

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

// `air.simdgroup_matrix_8x8_multiply_accumulate(A, B, C) = C + A·B`, the documented row-major 8x8
// matmul. AIR permits f16/f32 independently at the ABI boundary: arithmetic uses f32 whenever the
// result or any operand is f32, then converts to the declared result element. This preserves mixed
// accumulation such as `(f32, f16, f32) -> f16` without trusting the mangled suffix.
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
    let (result_elem, lanes) = composite_shape(ctx, rty)
        .ok_or_else(|| "air.simdgroup_matrix result is not a 64-lane composite".to_string())?;
    if lanes != 64 || (!is_f32_scalar(ctx, result_elem) && !is_half_scalar(ctx, result_elem)) {
        return Err("air.simdgroup_matrix result is not v64f16 or v64f32".to_string());
    }
    let mut operand_elems = Vec::with_capacity(3);
    for (ordinal, arg) in args.iter().enumerate() {
        let ty = value_result_type(ctx, *arg)
            .ok_or_else(|| format!("air.simdgroup_matrix operand {ordinal} has no type"))?;
        let (elem, arg_lanes) = composite_shape(ctx, ty).ok_or_else(|| {
            format!("air.simdgroup_matrix operand {ordinal} is not a 64-lane composite")
        })?;
        if arg_lanes != 64 || (!is_f32_scalar(ctx, elem) && !is_half_scalar(ctx, elem)) {
            return Err(format!(
                "air.simdgroup_matrix operand {ordinal} is not v64f16 or v64f32"
            ));
        }
        operand_elems.push(elem);
    }
    let arithmetic_elem = if is_f32_scalar(ctx, result_elem)
        || operand_elems.iter().any(|elem| is_f32_scalar(ctx, *elem))
    {
        ctx.ty_float()
    } else {
        result_elem
    };

    let mut insts = Vec::with_capacity(64 * (1 + 8 * 5) + 1);
    let mut result_lanes = Vec::with_capacity(64);
    for row in 0..8 {
        for col in 0..8 {
            let acc_raw =
                composite_extract(ctx, &mut insts, operand_elems[2], args[2], row * 8 + col);
            let mut acc =
                convert_matrix_lane(ctx, &mut insts, acc_raw, operand_elems[2], arithmetic_elem);
            for k in 0..8 {
                let a_raw =
                    composite_extract(ctx, &mut insts, operand_elems[0], args[0], row * 8 + k);
                let a =
                    convert_matrix_lane(ctx, &mut insts, a_raw, operand_elems[0], arithmetic_elem);
                let b_raw =
                    composite_extract(ctx, &mut insts, operand_elems[1], args[1], k * 8 + col);
                let b =
                    convert_matrix_lane(ctx, &mut insts, b_raw, operand_elems[1], arithmetic_elem);
                let product = ctx.module.fresh_id();
                insts.push(Instruction::new(
                    Op::FMul,
                    Some(arithmetic_elem),
                    Some(product),
                    vec![Operand::IdRef(a), Operand::IdRef(b)],
                ));
                let sum = ctx.module.fresh_id();
                insts.push(Instruction::new(
                    Op::FAdd,
                    Some(arithmetic_elem),
                    Some(sum),
                    vec![Operand::IdRef(acc), Operand::IdRef(product)],
                ));
                acc = sum;
            }
            result_lanes.push(Operand::IdRef(convert_matrix_lane(
                ctx,
                &mut insts,
                acc,
                arithmetic_elem,
                result_elem,
            )));
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

fn convert_matrix_lane(
    ctx: &mut Ctx,
    insts: &mut Vec<Instruction>,
    value: Word,
    from: Word,
    to: Word,
) -> Word {
    if from == to {
        return value;
    }
    let converted = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::FConvert,
        Some(to),
        Some(converted),
        vec![Operand::IdRef(value)],
    ));
    converted
}

/// Recognize the four stable distributed-register AGX2 matrix-MAD ABI symbols.
///
/// These appear after Apple's AIR backend has converted the public logical `v64` matrix value into
/// the two cells physically owned by each lane. The scalar type and matrix dimension are both part
/// of the ABI symbol; the lowering still validates every operand and result structurally.
pub(in crate::passes) fn agx2_matmad_dimension(name: &str) -> Option<u32> {
    match name {
        "llvm.agx2.f16matmad4x4.v2f16" | "llvm.agx2.f32matmad4x4.v2f32" => Some(4),
        "llvm.agx2.f16matmad8x8.v2f16" | "llvm.agx2.f32matmad8x8.v2f32" => Some(8),
        _ => None,
    }
}

/// Lower Apple's distributed-register AGX2 matrix multiply-add ABI.
///
/// Apple-generated AGX2 code establishes this 32-lane storage mapping for an 8x8 register tile:
///
/// ```text
/// row       = ((lane >> 1) & 3) | ((lane >> 4) << 2)
/// first_col = ((lane >> 3) & 1) * 4 + (lane & 1) * 2
/// ```
///
/// Each lane owns `(row, first_col)` and `(row, first_col + 1)`. The 8x8 intrinsic multiplies the
/// full tile. The 4x4 intrinsic operates on the four independent 4x4 quadrants of that same tile,
/// which is why it also consumes and returns `v2` in every one of the 32 lanes. Absolute shuffle
/// indices include the enclosing 32-lane base, preserving Metal simdgroup partitions when Vulkan
/// exposes a wider subgroup.
pub(in crate::passes) fn lower_agx2_matmad(
    ctx: &mut Ctx,
    name: &str,
    res: Word,
    rty: Word,
    args: &[Word],
    dimension: u32,
) -> Result<Vec<Instruction>, String> {
    if args.len() != 3 {
        return Err(format!("{name} expects 3 operands, got {}", args.len()));
    }
    if !matches!(dimension, 4 | 8) {
        return Err(format!("{name} has unsupported dimension {dimension}"));
    }
    let (elem, lanes) = composite_shape(ctx, rty)
        .ok_or_else(|| format!("{name} result is not a two-lane float composite"))?;
    if lanes != 2 || (!is_f32_scalar(ctx, elem) && !is_half_scalar(ctx, elem)) {
        return Err(format!("{name} result is not v2f16 or v2f32"));
    }
    for (ordinal, arg) in args.iter().enumerate() {
        let arg_ty = value_result_type(ctx, *arg)
            .ok_or_else(|| format!("{name} operand {ordinal} has no type"))?;
        let (arg_elem, arg_lanes) = composite_shape(ctx, arg_ty)
            .ok_or_else(|| format!("{name} operand {ordinal} is not a two-lane composite"))?;
        if arg_lanes != 2 || arg_elem != elem {
            return Err(format!(
                "{name} operand {ordinal} does not match its v2 result element type"
            ));
        }
    }

    let mut out = Vec::with_capacity(96);
    let uint = ctx.ty_uint();
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let subgroup_lane = subgroup_lane_index_u32(ctx, &mut out);
    let simd_lane = metal_simd_lane_local_u32(ctx, subgroup_lane, &mut out);
    let simd_base = metal_simd_lane_base_u32(ctx, subgroup_lane, simd_lane, &mut out);

    // Lane bits 1,2,4 encode the row; bits 0,3 encode the two-column owner. A 4x4 operation keeps
    // the current quadrant (row bit 4 and column bit 3), while an 8x8 operation replaces both from k.
    let a_preserve_mask = if dimension == 4 { 0x1e } else { 0x16 };
    let b_preserve_mask = if dimension == 4 { 0x19 } else { 0x09 };
    let a_key = bitwise_with_const(
        ctx,
        &mut out,
        Op::BitwiseAnd,
        uint,
        simd_lane,
        a_preserve_mask,
    );
    let b_key = bitwise_with_const(
        ctx,
        &mut out,
        Op::BitwiseAnd,
        uint,
        simd_lane,
        b_preserve_mask,
    );

    let a_components = [
        composite_extract(ctx, &mut out, elem, args[0], 0),
        composite_extract(ctx, &mut out, elem, args[0], 1),
    ];
    let b_components = [
        composite_extract(ctx, &mut out, elem, args[1], 0),
        composite_extract(ctx, &mut out, elem, args[1], 1),
    ];
    let mut accumulators = [
        composite_extract(ctx, &mut out, elem, args[2], 0),
        composite_extract(ctx, &mut out, elem, args[2], 1),
    ];
    let ext = ctx.glsl();

    for k in 0..dimension {
        let a_varying_bits = if dimension == 4 {
            (k >> 1) & 1
        } else {
            ((k & 4) << 1) | ((k >> 1) & 1)
        };
        let b_varying_bits = if dimension == 4 {
            k << 1
        } else {
            ((k & 3) << 1) | ((k & 4) << 2)
        };
        let a_owner_local =
            bitwise_with_const(ctx, &mut out, Op::BitwiseOr, uint, a_key, a_varying_bits);
        let b_owner_local =
            bitwise_with_const(ctx, &mut out, Op::BitwiseOr, uint, b_key, b_varying_bits);
        let a_owner = binary_value(ctx, &mut out, Op::IAdd, uint, simd_base, a_owner_local);
        let b_owner = binary_value(ctx, &mut out, Op::IAdd, uint, simd_base, b_owner_local);
        let a = subgroup_shuffle(
            ctx,
            &mut out,
            elem,
            scope,
            a_components[(k & 1) as usize],
            a_owner,
        );
        for component in 0..2 {
            let b = subgroup_shuffle(ctx, &mut out, elem, scope, b_components[component], b_owner);
            let next = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::ExtInst,
                Some(elem),
                Some(next),
                vec![
                    Operand::IdRef(ext),
                    Operand::LiteralExtInstInteger(GLSLstd450::Fma as u32),
                    Operand::IdRef(a),
                    Operand::IdRef(b),
                    Operand::IdRef(accumulators[component]),
                ],
            ));
            accumulators[component] = next;
        }
    }
    out.push(Instruction::new(
        Op::CompositeConstruct,
        Some(rty),
        Some(res),
        accumulators.into_iter().map(Operand::IdRef).collect(),
    ));
    Ok(out)
}

#[derive(Clone, Copy)]
enum Matrix16Transpose {
    Constant(bool),
    Dynamic(Word),
}

#[derive(Clone, Copy)]
struct Matrix16Tile {
    components: [Word; 8],
    elem: Word,
    scope: Word,
    simd_base: Word,
}

#[derive(Clone, Copy)]
struct Matrix16OutputMapping {
    row_bits: Word,
    row_as_col: Word,
    col_bits: Word,
    col_as_row: Word,
    simd_lane: Word,
}

/// Lower the distributed 16x16x16 AIR matrix multiply-accumulate ABI.
///
/// A Metal simdgroup is 32 lanes. Each lane owns a 2x4 rectangle of the 16x16 tile, with the same
/// lane-bit permutation used by Apple's older AGX2 8x8 (1x2) register ABI:
///
/// ```text
/// row_base = 2 * (((lane >> 1) & 3) | ((lane >> 2) & 4))
/// col_base = 4 * (((lane >> 2) & 2) | (lane & 1))
/// component(row_offset, col_offset) = 4 * row_offset + col_offset
/// ```
///
/// The mapping is structural: the 32 lanes times eight scalar cells cover the tile exactly. Both
/// transpose operands are honored, including non-constant flags, by selecting between coordinates
/// before the multiply. Absolute shuffle indices retain the enclosing 32-lane base on Vulkan
/// implementations whose hardware subgroup is wider than Metal's fixed simdgroup.
pub(in crate::passes) fn lower_simdgroup_matrix_16x16_mac(
    ctx: &mut Ctx,
    name: &str,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let Some(signature) = matrix16_intrinsic(name) else {
        return Err(format!(
            "{name} has an unsupported 16x16x16 matrix ABI signature"
        ));
    };
    let a_kind = signature.lhs;
    let b_kind = signature.rhs;
    let integer = signature.integer;
    if args.len() != 5 {
        return Err(format!("{name} expects 5 operands, got {}", args.len()));
    }
    let (result_elem, result_lanes) = composite_shape(ctx, rty)
        .ok_or_else(|| format!("{name} result is not an eight-lane composite"))?;
    let result_ok = if integer {
        is_int_scalar_width(ctx, result_elem, 32)
    } else {
        is_f32_scalar(ctx, result_elem)
    };
    if result_lanes != 8 || !result_ok {
        return Err(format!(
            "{name} result must be {}",
            if integer { "v8i32" } else { "v8f32" }
        ));
    }
    let a_ty = validate_matrix16_fragment(ctx, name, args[0], a_kind, 0)?;
    let b_ty = validate_matrix16_fragment(ctx, name, args[2], b_kind, 2)?;
    let c_ty = value_result_type(ctx, args[4])
        .ok_or_else(|| format!("{name} accumulator has no result type"))?;
    if c_ty != rty {
        return Err(format!(
            "{name} accumulator type does not match its result type"
        ));
    }
    for (ordinal, flag) in [(1, args[1]), (3, args[3])] {
        let ty = value_result_type(ctx, flag)
            .ok_or_else(|| format!("{name} transpose operand {ordinal} has no type"))?;
        if !is_bool_type(ctx, ty) {
            return Err(format!("{name} transpose operand {ordinal} is not i1"));
        }
    }
    let transpose_a = matrix16_transpose(ctx, args[1]);
    let transpose_b = matrix16_transpose(ctx, args[3]);

    let mut out = Vec::with_capacity(640);
    let uint = ctx.ty_uint();
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let subgroup_lane = subgroup_lane_index_u32(ctx, &mut out);
    let simd_lane = metal_simd_lane_local_u32(ctx, subgroup_lane, &mut out);
    let simd_base = metal_simd_lane_base_u32(ctx, subgroup_lane, simd_lane, &mut out);
    let row_bits = bitwise_with_const(ctx, &mut out, Op::BitwiseAnd, uint, simd_lane, 0x16);
    let col_bits = bitwise_with_const(ctx, &mut out, Op::BitwiseAnd, uint, simd_lane, 0x09);

    // When a coordinate is transposed, the current output row becomes a column (lane bits 2/4 ->
    // owner bits 0/3), or the current output column becomes a row (bits 0/3 -> owner bits 2/4).
    let row_as_col_lo = shift_with_const(ctx, &mut out, Op::ShiftRightLogical, simd_lane, 2);
    let row_as_col_lo = bitwise_with_const(ctx, &mut out, Op::BitwiseAnd, uint, row_as_col_lo, 1);
    let row_as_col_hi = shift_with_const(ctx, &mut out, Op::ShiftRightLogical, simd_lane, 1);
    let row_as_col_hi = bitwise_with_const(ctx, &mut out, Op::BitwiseAnd, uint, row_as_col_hi, 8);
    let row_as_col = binary_value(
        ctx,
        &mut out,
        Op::BitwiseOr,
        uint,
        row_as_col_lo,
        row_as_col_hi,
    );
    let col_as_row_lo = shift_with_const(ctx, &mut out, Op::ShiftLeftLogical, simd_lane, 2);
    let col_as_row_lo = bitwise_with_const(ctx, &mut out, Op::BitwiseAnd, uint, col_as_row_lo, 4);
    let col_as_row_hi = shift_with_const(ctx, &mut out, Op::ShiftLeftLogical, simd_lane, 1);
    let col_as_row_hi = bitwise_with_const(ctx, &mut out, Op::BitwiseAnd, uint, col_as_row_hi, 16);
    let col_as_row = binary_value(
        ctx,
        &mut out,
        Op::BitwiseOr,
        uint,
        col_as_row_lo,
        col_as_row_hi,
    );

    let a_components = extract_matrix16_components(ctx, &mut out, a_ty, args[0]);
    let a_components =
        convert_matrix16_components(ctx, &mut out, a_components, a_kind, result_elem);
    let b_components = extract_matrix16_components(ctx, &mut out, b_ty, args[2]);
    let b_components =
        convert_matrix16_components(ctx, &mut out, b_components, b_kind, result_elem);
    let c_components = extract_matrix16_components(ctx, &mut out, result_elem, args[4]);
    let a_tile = Matrix16Tile {
        components: a_components,
        elem: result_elem,
        scope,
        simd_base,
    };
    let b_tile = Matrix16Tile {
        components: b_components,
        elem: result_elem,
        scope,
        simd_base,
    };
    let mapping = Matrix16OutputMapping {
        row_bits,
        row_as_col,
        col_bits,
        col_as_row,
        simd_lane,
    };
    let mut results = Vec::with_capacity(8);
    for component in 0..8u32 {
        let row_offset = component / 4;
        let col_offset = component % 4;
        let mut acc = c_components[component as usize];
        for k in 0..16u32 {
            let a = matrix16_a_cell(ctx, &mut out, &a_tile, &mapping, row_offset, k, transpose_a);
            let b = matrix16_b_cell(ctx, &mut out, &b_tile, &mapping, col_offset, k, transpose_b);
            if integer {
                let product = binary_value(ctx, &mut out, Op::IMul, result_elem, a, b);
                acc = binary_value(ctx, &mut out, Op::IAdd, result_elem, acc, product);
            } else {
                let next = ctx.module.fresh_id();
                let ext = ctx.glsl();
                out.push(Instruction::new(
                    Op::ExtInst,
                    Some(result_elem),
                    Some(next),
                    vec![
                        Operand::IdRef(ext),
                        Operand::LiteralExtInstInteger(GLSLstd450::Fma as u32),
                        Operand::IdRef(a),
                        Operand::IdRef(b),
                        Operand::IdRef(acc),
                    ],
                ));
                acc = next;
            }
        }
        results.push(Operand::IdRef(acc));
    }
    out.push(Instruction::new(
        Op::CompositeConstruct,
        Some(rty),
        Some(res),
        results,
    ));
    Ok(out)
}

fn is_int_scalar_width(ctx: &Ctx, ty: Word, width: u32) -> bool {
    type_def_of(ctx, ty)
        .map(|def| {
            def.class.opcode == Op::TypeInt
                && def.operands.first() == Some(&Operand::LiteralBit32(width))
        })
        .unwrap_or(false)
}

fn validate_matrix16_fragment(
    ctx: &Ctx,
    name: &str,
    value: Word,
    kind: Matrix16Element,
    ordinal: usize,
) -> Result<Word, String> {
    let ty = value_result_type(ctx, value)
        .ok_or_else(|| format!("{name} operand {ordinal} has no type"))?;
    let (elem, lanes) = composite_shape(ctx, ty)
        .ok_or_else(|| format!("{name} operand {ordinal} is not an eight-lane composite"))?;
    let valid = match kind {
        Matrix16Element::F32 => is_f32_scalar(ctx, elem),
        Matrix16Element::F16 => is_half_scalar(ctx, elem),
        Matrix16Element::Bf16 => is_int_scalar_width(ctx, elem, 16),
        Matrix16Element::F8E4M3
        | Matrix16Element::F8E4M3Fn
        | Matrix16Element::F8E5M2
        | Matrix16Element::I8 { .. } => is_int_scalar_width(ctx, elem, 8),
    };
    if lanes != 8 || !valid {
        return Err(format!(
            "{name} operand {ordinal} does not match its ABI element type"
        ));
    }
    Ok(elem)
}

fn matrix16_transpose(ctx: &Ctx, flag: Word) -> Matrix16Transpose {
    match const_bool_value(ctx, flag) {
        Some(value) => Matrix16Transpose::Constant(value),
        None => Matrix16Transpose::Dynamic(flag),
    }
}

fn extract_matrix16_components(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    elem: Word,
    vector: Word,
) -> [Word; 8] {
    std::array::from_fn(|index| composite_extract(ctx, out, elem, vector, index as u32))
}

fn convert_matrix16_components(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    components: [Word; 8],
    kind: Matrix16Element,
    accumulator_ty: Word,
) -> [Word; 8] {
    components.map(|value| matrix16_to_accumulator(ctx, out, value, kind, accumulator_ty))
}

fn matrix16_a_cell(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    tile: &Matrix16Tile,
    mapping: &Matrix16OutputMapping,
    row_offset: u32,
    k: u32,
    transpose: Matrix16Transpose,
) -> Word {
    let normal_owner = owner_with_bits(
        ctx,
        out,
        tile.simd_base,
        mapping.row_bits,
        encode_matrix16_col(k),
    );
    let normal = matrix16_shuffle_component(
        ctx,
        out,
        tile.components[(row_offset * 4 + k % 4) as usize],
        tile.elem,
        tile.scope,
        normal_owner,
    );
    if matches!(transpose, Matrix16Transpose::Constant(false)) {
        return normal;
    }
    let transposed_owner = owner_with_bits(
        ctx,
        out,
        tile.simd_base,
        mapping.row_as_col,
        encode_matrix16_row(k),
    );
    let uint = ctx.ty_uint();
    let row_low = bitwise_with_const(ctx, out, Op::BitwiseAnd, uint, mapping.simd_lane, 2);
    let component_offset = ctx.const_uint((k % 2) * 4 + row_offset);
    let component = binary_value(ctx, out, Op::IAdd, uint, row_low, component_offset);
    let transposed =
        matrix16_shuffle_dynamic_component(ctx, out, tile, transposed_owner, component);
    matrix16_select_transpose(ctx, out, tile.elem, transpose, normal, transposed)
}

fn matrix16_b_cell(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    tile: &Matrix16Tile,
    mapping: &Matrix16OutputMapping,
    col_offset: u32,
    k: u32,
    transpose: Matrix16Transpose,
) -> Word {
    let normal_owner = owner_with_bits(
        ctx,
        out,
        tile.simd_base,
        mapping.col_bits,
        encode_matrix16_row(k),
    );
    let normal = matrix16_shuffle_component(
        ctx,
        out,
        tile.components[((k % 2) * 4 + col_offset) as usize],
        tile.elem,
        tile.scope,
        normal_owner,
    );
    if matches!(transpose, Matrix16Transpose::Constant(false)) {
        return normal;
    }
    let row_offset_bit = (col_offset / 2) << 1;
    let transposed_owner = owner_with_bits(
        ctx,
        out,
        tile.simd_base,
        mapping.col_as_row,
        encode_matrix16_col(k) | row_offset_bit,
    );
    let transposed = matrix16_shuffle_component(
        ctx,
        out,
        tile.components[((col_offset % 2) * 4 + k % 4) as usize],
        tile.elem,
        tile.scope,
        transposed_owner,
    );
    matrix16_select_transpose(ctx, out, tile.elem, transpose, normal, transposed)
}

fn encode_matrix16_row(row: u32) -> u32 {
    let group = row / 2;
    ((group & 1) << 1) | ((group & 2) << 1) | ((group & 4) << 2)
}

fn encode_matrix16_col(col: u32) -> u32 {
    let group = col / 4;
    (group & 1) | ((group & 2) << 2)
}

fn owner_with_bits(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    simd_base: Word,
    varying: Word,
    fixed: u32,
) -> Word {
    let uint = ctx.ty_uint();
    let local = bitwise_with_const(ctx, out, Op::BitwiseOr, uint, varying, fixed);
    binary_value(ctx, out, Op::IAdd, uint, simd_base, local)
}

fn matrix16_shuffle_component(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    component: Word,
    elem: Word,
    scope: Word,
    owner: Word,
) -> Word {
    subgroup_shuffle(ctx, out, elem, scope, component, owner)
}

fn matrix16_shuffle_dynamic_component(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    tile: &Matrix16Tile,
    owner: Word,
    component: Word,
) -> Word {
    let shuffled: Vec<_> = tile
        .components
        .iter()
        .map(|value| subgroup_shuffle(ctx, out, tile.elem, tile.scope, *value, owner))
        .collect();
    let bool_ty = ctx.ty_bool();
    let mut result = shuffled[0];
    for index in 1..8u32 {
        let index_value = ctx.const_uint(index);
        let matches = binary_value(ctx, out, Op::IEqual, bool_ty, component, index_value);
        result = select_value(
            ctx,
            out,
            tile.elem,
            matches,
            shuffled[index as usize],
            result,
        );
    }
    result
}

fn matrix16_select_transpose(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    elem: Word,
    transpose: Matrix16Transpose,
    normal: Word,
    transposed: Word,
) -> Word {
    match transpose {
        Matrix16Transpose::Constant(true) => transposed,
        Matrix16Transpose::Constant(false) => normal,
        Matrix16Transpose::Dynamic(condition) => {
            let result = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::Select,
                Some(elem),
                Some(result),
                vec![
                    Operand::IdRef(condition),
                    Operand::IdRef(transposed),
                    Operand::IdRef(normal),
                ],
            ));
            result
        }
    }
}

fn matrix16_to_accumulator(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    value: Word,
    kind: Matrix16Element,
    accumulator_ty: Word,
) -> Word {
    match kind {
        Matrix16Element::F32 => value,
        Matrix16Element::F16 => {
            let result = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::FConvert,
                Some(accumulator_ty),
                Some(result),
                vec![Operand::IdRef(value)],
            ));
            result
        }
        Matrix16Element::Bf16 => bfloat_to_f32(ctx, out, value),
        Matrix16Element::F8E4M3 => matrix16_float8_to_f32(ctx, out, value, 4, 3, false),
        Matrix16Element::F8E4M3Fn => matrix16_float8_to_f32(ctx, out, value, 4, 3, true),
        Matrix16Element::F8E5M2 => matrix16_float8_to_f32(ctx, out, value, 5, 2, false),
        Matrix16Element::I8 { signed } => {
            let result = ctx.module.fresh_id();
            out.push(Instruction::new(
                if signed { Op::SConvert } else { Op::UConvert },
                Some(accumulator_ty),
                Some(result),
                vec![Operand::IdRef(value)],
            ));
            result
        }
    }
}

fn matrix16_float8_to_f32(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    value: Word,
    exponent_bits: u32,
    mantissa_bits: u32,
    finite_only: bool,
) -> Word {
    let uint = ctx.ty_uint();
    let int = ctx.ty_sint();
    let float = ctx.ty_float();
    let bool_ty = ctx.ty_bool();
    let bits = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::UConvert,
        Some(uint),
        Some(bits),
        vec![Operand::IdRef(value)],
    ));
    let mantissa_mask = (1u32 << mantissa_bits) - 1;
    let exponent_mask = (1u32 << exponent_bits) - 1;
    let mantissa = bitwise_with_const(ctx, out, Op::BitwiseAnd, uint, bits, mantissa_mask);
    let shifted_exp = shift_with_const(ctx, out, Op::ShiftRightLogical, bits, mantissa_bits);
    let exponent = bitwise_with_const(ctx, out, Op::BitwiseAnd, uint, shifted_exp, exponent_mask);
    let zero = ctx.const_uint(0);
    let exp_zero = binary_value(ctx, out, Op::IEqual, bool_ty, exponent, zero);
    let implicit_bit = ctx.const_uint(1 << mantissa_bits);
    let normal_significand = binary_value(ctx, out, Op::IAdd, uint, mantissa, implicit_bit);
    let significand = select_value(ctx, out, uint, exp_zero, mantissa, normal_significand);
    let significand_f = unary_value(ctx, out, Op::ConvertUToF, float, significand);
    let exponent_i = unary_value(ctx, out, Op::Bitcast, int, exponent);
    let bias = (1i32 << (exponent_bits - 1)) - 1;
    let normal_scale_offset = ctx.const_int_of(int, -(bias + mantissa_bits as i32) as i64);
    let normal_scale = binary_value(ctx, out, Op::IAdd, int, exponent_i, normal_scale_offset);
    let subnormal_scale = ctx.const_int_of(int, (1 - bias - mantissa_bits as i32) as i64);
    let scale = select_value(ctx, out, int, exp_zero, subnormal_scale, normal_scale);
    let magnitude = ctx.module.fresh_id();
    let ext = ctx.glsl();
    out.push(Instruction::new(
        Op::ExtInst,
        Some(float),
        Some(magnitude),
        vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(GLSLstd450::Ldexp as u32),
            Operand::IdRef(significand_f),
            Operand::IdRef(scale),
        ],
    ));
    let sign = bitwise_with_const(ctx, out, Op::BitwiseAnd, uint, bits, 0x80);
    let negative = binary_value(ctx, out, Op::INotEqual, bool_ty, sign, zero);
    let negated = unary_value(ctx, out, Op::FNegate, float, magnitude);
    let signed = select_value(ctx, out, float, negative, negated, magnitude);

    let exponent_mask_value = ctx.const_uint(exponent_mask);
    let exp_all_ones = binary_value(ctx, out, Op::IEqual, bool_ty, exponent, exponent_mask_value);
    let mantissa_mask_value = ctx.const_uint(mantissa_mask);
    let mant_all_ones = binary_value(ctx, out, Op::IEqual, bool_ty, mantissa, mantissa_mask_value);
    if finite_only {
        let is_nan = binary_value(
            ctx,
            out,
            Op::LogicalAnd,
            bool_ty,
            exp_all_ones,
            mant_all_ones,
        );
        let nan = ctx.const_float(f32::NAN);
        return select_value(ctx, out, float, is_nan, nan, signed);
    }
    let mant_zero = binary_value(ctx, out, Op::IEqual, bool_ty, mantissa, zero);
    let infinity = ctx.const_float(f32::INFINITY);
    let negative_infinity = ctx.const_float(f32::NEG_INFINITY);
    let signed_infinity = select_value(ctx, out, float, negative, negative_infinity, infinity);
    let nan = ctx.const_float(f32::NAN);
    let special = select_value(ctx, out, float, mant_zero, signed_infinity, nan);
    select_value(ctx, out, float, exp_all_ones, special, signed)
}

fn shift_with_const(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    op: Op,
    value: Word,
    amount: u32,
) -> Word {
    let uint = ctx.ty_uint();
    let amount = ctx.const_uint(amount);
    binary_value(ctx, out, op, uint, value, amount)
}

fn unary_value(ctx: &mut Ctx, out: &mut Vec<Instruction>, op: Op, ty: Word, value: Word) -> Word {
    let result = ctx.module.fresh_id();
    out.push(Instruction::new(
        op,
        Some(ty),
        Some(result),
        vec![Operand::IdRef(value)],
    ));
    result
}

fn select_value(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    ty: Word,
    condition: Word,
    when_true: Word,
    when_false: Word,
) -> Word {
    let result = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Select,
        Some(ty),
        Some(result),
        vec![
            Operand::IdRef(condition),
            Operand::IdRef(when_true),
            Operand::IdRef(when_false),
        ],
    ));
    result
}

fn bitwise_with_const(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    op: Op,
    ty: Word,
    lhs: Word,
    rhs: u32,
) -> Word {
    let rhs = ctx.const_uint(rhs);
    binary_value(ctx, out, op, ty, lhs, rhs)
}

fn binary_value(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    op: Op,
    ty: Word,
    lhs: Word,
    rhs: Word,
) -> Word {
    let result = ctx.module.fresh_id();
    out.push(Instruction::new(
        op,
        Some(ty),
        Some(result),
        vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
    ));
    result
}

fn subgroup_shuffle(
    ctx: &mut Ctx,
    out: &mut Vec<Instruction>,
    ty: Word,
    scope: Word,
    value: Word,
    lane: Word,
) -> Word {
    let result = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::GroupNonUniformShuffle,
        Some(ty),
        Some(result),
        vec![
            Operand::IdScope(scope),
            Operand::IdRef(value),
            Operand::IdRef(lane),
        ],
    ));
    result
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

/// `air.simd_shuffle_rotate_down(value, delta)` rotates values down within Metal's 32-lane simdgroup,
/// wrapping at the simdgroup boundary: source lane = `(lane + delta) & 31`.
pub(in crate::passes) fn lower_simd_shuffle_rotate_down(
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
    let simd_lane = metal_simd_lane_local_u32(ctx, subgroup_lane, &mut insts);
    let simd_base = metal_simd_lane_base_u32(ctx, subgroup_lane, simd_lane, &mut insts);
    let local_plus_delta = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(local_plus_delta),
        vec![Operand::IdRef(simd_lane), Operand::IdRef(delta)],
    ));
    let source_local = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(source_local),
        vec![
            Operand::IdRef(local_plus_delta),
            Operand::IdRef(ctx.const_uint(31)),
        ],
    ));
    let source_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(source_lane),
        vec![Operand::IdRef(simd_base), Operand::IdRef(source_local)],
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

/// `air.simd_shuffle_down(value, delta)` reads `value` from lane `lane + delta` within Metal's
/// 32-lane simdgroup. Lower via absolute shuffle instead of `ShuffleDown` so wider Vulkan subgroups
/// cannot cross the Metal simdgroup boundary. Out-of-range reads select the current lane to avoid
/// emitting an undefined SPIR-V source lane.
pub(in crate::passes) fn lower_simd_shuffle_down(
    ctx: &mut Ctx,
    result: Word,
    result_type: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let mut insts = Vec::new();
    let delta = subgroup_shuffle_index_u32(ctx, args[1], &mut insts)?;
    let uint = ctx.ty_uint();
    let lane = subgroup_lane_index_u32(ctx, &mut insts);
    let simd_lane = metal_simd_lane_local_u32(ctx, lane, &mut insts);
    let simd_base = metal_simd_lane_base_u32(ctx, lane, simd_lane, &mut insts);
    let remaining = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ISub,
        Some(uint),
        Some(remaining),
        vec![
            Operand::IdRef(ctx.const_uint(32)),
            Operand::IdRef(simd_lane),
        ],
    ));
    let in_bounds = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ULessThan,
        Some(ctx.ty_bool()),
        Some(in_bounds),
        vec![Operand::IdRef(delta), Operand::IdRef(remaining)],
    ));
    let shifted_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(shifted_lane),
        vec![Operand::IdRef(simd_lane), Operand::IdRef(delta)],
    ));
    let shifted_subgroup_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(shifted_subgroup_lane),
        vec![Operand::IdRef(simd_base), Operand::IdRef(shifted_lane)],
    ));
    let source_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::Select,
        Some(uint),
        Some(source_lane),
        vec![
            Operand::IdRef(in_bounds),
            Operand::IdRef(shifted_subgroup_lane),
            Operand::IdRef(lane),
        ],
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

/// `air.simd_shuffle_up(value, delta)` reads `value` from lane `lane - delta` within Metal's
/// 32-lane simdgroup. Use absolute shuffle to keep semantics independent of the Vulkan subgroup
/// width.
pub(in crate::passes) fn lower_simd_shuffle_up(
    ctx: &mut Ctx,
    result: Word,
    result_type: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let scope = ctx.const_uint(Scope::Subgroup as u32);
    let mut insts = Vec::new();
    let delta = subgroup_shuffle_index_u32(ctx, args[1], &mut insts)?;
    let uint = ctx.ty_uint();
    let lane = subgroup_lane_index_u32(ctx, &mut insts);
    let simd_lane = metal_simd_lane_local_u32(ctx, lane, &mut insts);
    let simd_base = metal_simd_lane_base_u32(ctx, lane, simd_lane, &mut insts);
    let in_bounds = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::UGreaterThanEqual,
        Some(ctx.ty_bool()),
        Some(in_bounds),
        vec![Operand::IdRef(simd_lane), Operand::IdRef(delta)],
    ));
    let shifted_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ISub,
        Some(uint),
        Some(shifted_lane),
        vec![Operand::IdRef(simd_lane), Operand::IdRef(delta)],
    ));
    let shifted_subgroup_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::IAdd,
        Some(uint),
        Some(shifted_subgroup_lane),
        vec![Operand::IdRef(simd_base), Operand::IdRef(shifted_lane)],
    ));
    let source_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::Select,
        Some(uint),
        Some(source_lane),
        vec![
            Operand::IdRef(in_bounds),
            Operand::IdRef(shifted_subgroup_lane),
            Operand::IdRef(lane),
        ],
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
    let var = subgroup_local_invocation_id_input_var(ctx, uint);
    let lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::Load,
        Some(uint),
        Some(lane),
        vec![Operand::IdRef(var)],
    ));
    lane
}

fn metal_simd_lane_local_u32(
    ctx: &mut Ctx,
    subgroup_lane: Word,
    insts: &mut Vec<Instruction>,
) -> Word {
    let uint = ctx.ty_uint();
    let simd_lane = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint),
        Some(simd_lane),
        vec![
            Operand::IdRef(subgroup_lane),
            Operand::IdRef(ctx.const_uint(31)),
        ],
    ));
    simd_lane
}

fn metal_simd_lane_base_u32(
    ctx: &mut Ctx,
    subgroup_lane: Word,
    simd_lane: Word,
    insts: &mut Vec<Instruction>,
) -> Word {
    let uint = ctx.ty_uint();
    let simd_base = ctx.module.fresh_id();
    insts.push(Instruction::new(
        Op::ISub,
        Some(uint),
        Some(simd_base),
        vec![Operand::IdRef(subgroup_lane), Operand::IdRef(simd_lane)],
    ));
    simd_base
}

pub(in crate::passes) fn subgroup_local_invocation_id_input_var(ctx: &mut Ctx, uint: Word) -> Word {
    let key = SynthCacheKey::SubgroupLocalInvocationIdInputVar;
    if let Some(&var) = ctx.synth_cache.get(&key) {
        return var;
    }
    if let Some(var) = existing_builtin_input_var(ctx, BuiltIn::SubgroupLocalInvocationId, uint) {
        decorate_fragment_integer_input_flat(ctx, var);
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
            Operand::BuiltIn(BuiltIn::SubgroupLocalInvocationId),
        ],
    ));
    decorate_fragment_integer_input_flat(ctx, var);
    ctx.interface.push(var);
    ctx.synth_cache.insert(key, var);
    var
}

fn decorate_fragment_integer_input_flat(ctx: &mut Ctx, var: Word) {
    if ctx.stage != Stage::Fragment
        || ctx.module.annotations.iter().any(|instruction| {
            instruction.class.opcode == Op::Decorate
                && instruction.operands.first() == Some(&Operand::IdRef(var))
                && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::Flat))
        })
    {
        return;
    }
    ctx.module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![Operand::IdRef(var), Operand::Decoration(Decoration::Flat)],
    ));
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

#[cfg(test)]
mod matrix16_mapping_tests {
    use super::{encode_matrix16_col, encode_matrix16_row};
    use std::collections::BTreeSet;

    fn coordinate(lane: u32, component: u32) -> (u32, u32) {
        let row_group = ((lane >> 1) & 3) | ((lane >> 2) & 4);
        let col_group = (lane & 1) | ((lane >> 2) & 2);
        (row_group * 2 + component / 4, col_group * 4 + component % 4)
    }

    #[test]
    fn distributed_fragments_cover_every_matrix_cell_once() {
        let cells: BTreeSet<_> = (0..32)
            .flat_map(|lane| (0..8).map(move |component| coordinate(lane, component)))
            .collect();
        assert_eq!(cells.len(), 16 * 16);
        assert_eq!(cells.first(), Some(&(0, 0)));
        assert_eq!(cells.last(), Some(&(15, 15)));
    }

    #[test]
    fn owner_formulas_cover_normal_and_transposed_operands() {
        for lane in 0..32u32 {
            let row_bits = lane & 0x16;
            let col_bits = lane & 0x09;
            let row_as_col = ((lane >> 2) & 1) | ((lane >> 1) & 8);
            let col_as_row = ((lane << 2) & 4) | ((lane << 1) & 16);
            for component in 0..8u32 {
                let (row, col) = coordinate(lane, component);
                let row_offset = component / 4;
                let col_offset = component % 4;
                for k in 0..16u32 {
                    let a = coordinate(row_bits | encode_matrix16_col(k), row_offset * 4 + k % 4);
                    assert_eq!(a, (row, k));
                    let at = coordinate(
                        row_as_col | encode_matrix16_row(k),
                        (k % 2) * 4 + (lane & 2) + row_offset,
                    );
                    assert_eq!(at, (k, row));

                    let b = coordinate(col_bits | encode_matrix16_row(k), (k % 2) * 4 + col_offset);
                    assert_eq!(b, (k, col));
                    let bt = coordinate(
                        col_as_row | encode_matrix16_col(k) | ((col_offset / 2) << 1),
                        (col_offset % 2) * 4 + k % 4,
                    );
                    assert_eq!(bt, (col, k));
                }
            }
        }
    }
}
