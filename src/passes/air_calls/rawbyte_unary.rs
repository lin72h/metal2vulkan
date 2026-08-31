//! Raw-byte, descriptor-alias, and unary AIR call lowering.

use super::*;

pub(in crate::passes) fn atomic_i32_pointer(
    ctx: &mut Ctx,
    ptr: Word,
    out: &mut Vec<Instruction>,
) -> Word {
    let Some(ptr_ty) = value_result_type(ctx, ptr) else {
        return ptr;
    };
    let Some(pointee) = pointer_pointee_type(ctx, ptr_ty) else {
        return ptr;
    };
    if is_uint_scalar_width(ctx, pointee, 32) {
        return ptr;
    }
    if !is_uint_scalar_width(ctx, pointee, 8) {
        return ptr;
    }
    let Some((root, byte_index)) = raw_byte_access_root_and_index(ctx, ptr) else {
        return ptr;
    };
    let Some(binding) = descriptor_binding(ctx, root) else {
        return ptr;
    };
    let alias = raw_uint_alias_buffer(ctx, binding);
    let Some(word_index) = raw_byte_index_to_word_index(ctx, byte_index, out) else {
        return ptr;
    };
    let uint = ctx.ty_uint();
    let ptr_ty = ctx.ty_ptr(StorageClass::StorageBuffer, uint);
    let fixed = ctx.module.fresh_id();
    let zero = ctx.const_uint(0);
    out.push(Instruction::new(
        Op::AccessChain,
        Some(ptr_ty),
        Some(fixed),
        vec![
            Operand::IdRef(alias),
            Operand::IdRef(zero),
            Operand::IdRef(word_index),
        ],
    ));
    fixed
}

pub(in crate::passes) fn raw_byte_access_root_and_index(
    ctx: &Ctx,
    ptr: Word,
) -> Option<(Word, Word)> {
    let inst = value_def_instruction(ctx, ptr)?;
    if !matches!(
        inst.class.opcode,
        Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
    ) {
        return None;
    }
    let Some(Operand::IdRef(root)) = inst.operands.first() else {
        return None;
    };
    let indices = inst.operands[1..]
        .iter()
        .map(|operand| match operand {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    if let Some(byte_index) = raw_byte_buffer_index(ctx, *root, &indices) {
        return Some((*root, byte_index));
    }
    let (root, mut base_indices) = raw_byte_access_path(ctx, *root)?;
    base_indices.extend(indices);
    raw_byte_buffer_index(ctx, root, &base_indices).map(|byte_index| (root, byte_index))
}

pub(in crate::passes) fn raw_byte_access_path(ctx: &Ctx, ptr: Word) -> Option<(Word, Vec<Word>)> {
    let inst = value_def_instruction(ctx, ptr)?;
    if !matches!(
        inst.class.opcode,
        Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
    ) {
        return None;
    }
    let Some(Operand::IdRef(base)) = inst.operands.first() else {
        return None;
    };
    let indices = inst.operands[1..]
        .iter()
        .map(|operand| match operand {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    if raw_byte_buffer_index(ctx, *base, &indices).is_some() {
        return Some((*base, indices));
    }
    let (root, mut base_indices) = raw_byte_access_path(ctx, *base)?;
    base_indices.extend(indices);
    Some((root, base_indices))
}

pub(in crate::passes) fn raw_byte_buffer_index(
    ctx: &Ctx,
    root: Word,
    indices: &[Word],
) -> Option<Word> {
    let [member, byte_index] = indices else {
        return None;
    };
    if constant_u32(ctx, *member) != Some(0) {
        return None;
    }
    let root_ty = value_result_type(ctx, root)?;
    let block_ty = pointer_pointee_type(ctx, root_ty)?;
    let block_def = type_def_of(ctx, block_ty)?;
    if block_def.class.opcode != Op::TypeStruct {
        return None;
    }
    let runtime = match block_def.operands.first() {
        Some(Operand::IdRef(runtime)) => *runtime,
        _ => return None,
    };
    let runtime_def = type_def_of(ctx, runtime)?;
    if runtime_def.class.opcode != Op::TypeRuntimeArray {
        return None;
    }
    let elem = match runtime_def.operands.first() {
        Some(Operand::IdRef(elem)) => *elem,
        _ => return None,
    };
    is_uint_scalar_width(ctx, elem, 8).then_some(*byte_index)
}

pub(in crate::passes) fn raw_byte_index_to_word_index(
    ctx: &mut Ctx,
    byte_index: Word,
    out: &mut Vec<Instruction>,
) -> Option<Word> {
    if let Some(byte_index) = constant_u32(ctx, byte_index) {
        return (byte_index % 4 == 0).then(|| ctx.const_uint(byte_index / 4));
    }
    let byte_ty = value_result_type(ctx, byte_index)?;
    let byte_index = if is_uint_scalar_width(ctx, byte_ty, 32) {
        byte_index
    } else if int_scalar_width(ctx, byte_ty) == Some(64) {
        let converted = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::UConvert,
            Some(ctx.ty_uint()),
            Some(converted),
            vec![Operand::IdRef(byte_index)],
        ));
        converted
    } else {
        return None;
    };
    let word = ctx.module.fresh_id();
    let divisor = ctx.const_uint(4);
    out.push(Instruction::new(
        Op::UDiv,
        Some(ctx.ty_uint()),
        Some(word),
        vec![Operand::IdRef(byte_index), Operand::IdRef(divisor)],
    ));
    Some(word)
}

pub(in crate::passes) fn raw_uint_alias_buffer(ctx: &mut Ctx, binding: u32) -> Word {
    if let Some(var) = find_raw_uint_alias_buffer(ctx, binding) {
        return var;
    }
    let uint = ctx.ty_uint();
    let runtime = ctx.ty_runtime_array(uint);
    let block = ctx.module.fresh_id();
    ctx.new_globals.push(type_inst(
        Op::TypeStruct,
        block,
        vec![Operand::IdRef(runtime)],
    ));
    decorate_raw_uint_block(ctx, block, runtime);

    let ptr_ty = ctx.ty_ptr(StorageClass::StorageBuffer, block);
    let var = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::Variable,
        Some(ptr_ty),
        Some(var),
        vec![Operand::StorageClass(StorageClass::StorageBuffer)],
    ));
    decorate_descriptor_binding(ctx, var, binding);
    ctx.interface.push(var);
    var
}

pub(in crate::passes) fn find_raw_uint_alias_buffer(ctx: &Ctx, binding: u32) -> Option<Word> {
    ctx.new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .filter(|inst| inst.class.opcode == Op::Variable)
        .filter_map(|inst| inst.result_id)
        .find(|var| {
            descriptor_binding(ctx, *var) == Some(binding) && is_raw_uint_buffer_var(ctx, *var)
        })
}

pub(in crate::passes) fn is_raw_uint_buffer_var(ctx: &Ctx, var: Word) -> bool {
    let Some(var_ty) = value_result_type(ctx, var) else {
        return false;
    };
    let Some(block_ty) = pointer_pointee_type(ctx, var_ty) else {
        return false;
    };
    let Some(block_def) = type_def_of(ctx, block_ty) else {
        return false;
    };
    if block_def.class.opcode != Op::TypeStruct {
        return false;
    }
    let Some(Operand::IdRef(runtime)) = block_def.operands.first() else {
        return false;
    };
    let Some(runtime_def) = type_def_of(ctx, *runtime) else {
        return false;
    };
    if runtime_def.class.opcode != Op::TypeRuntimeArray {
        return false;
    }
    let Some(Operand::IdRef(elem)) = runtime_def.operands.first() else {
        return false;
    };
    is_uint_scalar_width(ctx, *elem, 32)
}

pub(in crate::passes) fn decorate_raw_uint_block(ctx: &mut Ctx, block: Word, runtime: Word) {
    ctx.module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(block),
            Operand::Decoration(Decoration::Block),
        ],
    ));
    ctx.module.annotations.push(Instruction::new(
        Op::MemberDecorate,
        None,
        None,
        vec![
            Operand::IdRef(block),
            Operand::LiteralBit32(0),
            Operand::Decoration(Decoration::Offset),
            Operand::LiteralBit32(0),
        ],
    ));
    ctx.module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(runtime),
            Operand::Decoration(Decoration::ArrayStride),
            Operand::LiteralBit32(4),
        ],
    ));
}

pub(in crate::passes) fn decorate_descriptor_binding(ctx: &mut Ctx, var: Word, binding: u32) {
    let set = ctx.descriptor_layout.set;
    decorate_binding(&mut ctx.module, var, set, binding);
}

pub(in crate::passes) fn descriptor_binding(ctx: &Ctx, var: Word) -> Option<u32> {
    ctx.module.annotations.iter().find_map(|inst| {
        if inst.class.opcode == Op::Decorate
            && inst.operands.first() == Some(&Operand::IdRef(var))
            && inst.operands.get(1) == Some(&Operand::Decoration(Decoration::Binding))
        {
            match inst.operands.get(2) {
                Some(Operand::LiteralBit32(binding)) => Some(*binding),
                _ => None,
            }
        } else {
            None
        }
    })
}

pub(in crate::passes) fn vector_type_shape(ctx: &Ctx, ty: Word) -> Option<(Word, u32)> {
    let def = type_def_of(ctx, ty)?;
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
    Some((elem, lanes))
}

pub(in crate::passes) fn is_image_size_query(name: &str) -> bool {
    name.starts_with("air.get_width_texture")
        || name.starts_with("air.get_height_texture")
        || name.starts_with("air.get_depth_texture")
        || name.starts_with("air.get_array_size_texture")
        || name.starts_with("air.get_width_depth")
        || name.starts_with("air.get_height_depth")
        || name.starts_with("air.get_depth_depth")
}

/// Lower a derivative intrinsic (`OpDPdx`/`OpDPdy`/`OpFwidth`). On a 32-bit-float operand it is the
/// op directly; on a HALF operand (`rty` is half/half-vector) it round-trips through float, since
/// Vulkan requires these to operate on 32-bit floats. `arg` is the (half or float) operand.
pub(in crate::passes) fn half_deriv(
    ctx: &mut Ctx,
    op: Op,
    res: Word,
    rty: Word,
    arg: Word,
) -> Result<Vec<Instruction>, String> {
    if !is_f32_scalar_or_vector(ctx, rty) && !is_half_scalar_or_vector(ctx, rty) {
        return Err(format!(
            "{op:?} AIR result is not a half/float scalar or vector"
        ));
    }
    if value_result_type(ctx, arg) != Some(rty) {
        return Err(format!("{op:?} AIR operand does not match its result type"));
    }
    let float_ty = float_equivalent(ctx, rty);
    if float_ty == rty {
        // already a float type -> emit the op directly.
        return Ok(vec![Instruction::new(
            op,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(arg)],
        )]);
    }
    // half: FConvert up, derivative in float, FConvert down.
    let argf = ctx.module.fresh_id();
    let derivf = ctx.module.fresh_id();
    Ok(vec![
        Instruction::new(
            Op::FConvert,
            Some(float_ty),
            Some(argf),
            vec![Operand::IdRef(arg)],
        ),
        Instruction::new(op, Some(float_ty), Some(derivf), vec![Operand::IdRef(argf)]),
        Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(derivf)],
        ),
    ])
}

/// Lower `op(scale * x)` for a scalar float operand (e.g. `cospi` = `cos(pi*x)`, `exp10` =
/// `exp2(log2(10)*x)`). Computes in 32-bit float space (converting a half operand in and the result
/// back), so the scalar `scale` constant matches. Rejects non-scalar-float result types so vector
/// forms FALLBACK rather than emit a scalar-times-vector type error.
pub(in crate::passes) fn lower_premul_glsl_unary(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    arg: Word,
    scale: f32,
    op: GLSLstd450,
) -> Result<Vec<Instruction>, String> {
    let float_ty = float_equivalent(ctx, rty);
    match type_def_of(ctx, float_ty) {
        Some(def) if def.class.opcode == Op::TypeFloat => {}
        _ => return Err("premul transcendental currently supports scalar float only".to_string()),
    }
    let ext = ctx.glsl();
    let scale_c = ctx.const_float(scale);
    let mut out = Vec::new();
    // Convert a half operand up to f32 for the multiply/transcendental.
    let argf = if float_ty == rty {
        arg
    } else {
        let f = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FConvert,
            Some(float_ty),
            Some(f),
            vec![Operand::IdRef(arg)],
        ));
        f
    };
    let prod = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::FMul,
        Some(float_ty),
        Some(prod),
        vec![Operand::IdRef(argf), Operand::IdRef(scale_c)],
    ));
    // The transcendental writes the final result directly when no half round-trip is needed.
    let trans = if float_ty == rty {
        res
    } else {
        ctx.module.fresh_id()
    };
    out.push(Instruction::new(
        Op::ExtInst,
        Some(float_ty),
        Some(trans),
        vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(op as u32),
            Operand::IdRef(prod),
        ],
    ));
    if float_ty != rty {
        out.push(Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(trans)],
        ));
    }
    Ok(out)
}

/// Lower a unary GLSL.std.450 op. For half scalar/vector operands, round-trip through the matching
/// 32-bit float shape; SPIR-V Tools rejects at least Round/RoundEven directly on half vectors.
pub(in crate::passes) fn half_glsl_unary(
    ctx: &mut Ctx,
    op: GLSLstd450,
    res: Word,
    rty: Word,
    arg: Word,
) -> Vec<Instruction> {
    let float_ty = float_equivalent(ctx, rty);
    let ext = ctx.glsl();
    if float_ty == rty {
        return vec![Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(res),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(op as u32),
                Operand::IdRef(arg),
            ],
        )];
    }
    let argf = ctx.module.fresh_id();
    let outf = ctx.module.fresh_id();
    vec![
        Instruction::new(
            Op::FConvert,
            Some(float_ty),
            Some(argf),
            vec![Operand::IdRef(arg)],
        ),
        Instruction::new(
            Op::ExtInst,
            Some(float_ty),
            Some(outf),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(op as u32),
                Operand::IdRef(argf),
            ],
        ),
        Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(outf)],
        ),
    ]
}

pub(in crate::passes) fn half_abs_pow(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    base: Word,
    exponent: Word,
) -> Vec<Instruction> {
    let float_ty = float_equivalent(ctx, rty);
    let ext = ctx.glsl();
    let basef = ctx.module.fresh_id();
    let exponentf = ctx.module.fresh_id();
    let abs_base = ctx.module.fresh_id();
    let raw_powf = ctx.module.fresh_id();
    let powf = ctx.module.fresh_id();
    let base_is_zero = ctx.module.fresh_id();
    let exponent_is_zero = ctx.module.fresh_id();
    let zero_zero = ctx.module.fresh_id();
    let n = vector_len(ctx, rty);
    let bool_ty = if n > 1 {
        ctx.ty_vec_bool(n)
    } else {
        ctx.ty_bool()
    };
    let zero = splat_or_scalar(ctx, float_ty, 0.0, n);
    let one = splat_or_scalar(ctx, float_ty, 1.0, n);
    vec![
        Instruction::new(
            Op::FConvert,
            Some(float_ty),
            Some(basef),
            vec![Operand::IdRef(base)],
        ),
        Instruction::new(
            Op::FConvert,
            Some(float_ty),
            Some(exponentf),
            vec![Operand::IdRef(exponent)],
        ),
        Instruction::new(
            Op::ExtInst,
            Some(float_ty),
            Some(abs_base),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::FAbs as u32),
                Operand::IdRef(basef),
            ],
        ),
        Instruction::new(
            Op::ExtInst,
            Some(float_ty),
            Some(raw_powf),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::Pow as u32),
                Operand::IdRef(abs_base),
                Operand::IdRef(exponentf),
            ],
        ),
        Instruction::new(
            Op::FOrdEqual,
            Some(bool_ty),
            Some(base_is_zero),
            vec![Operand::IdRef(abs_base), Operand::IdRef(zero)],
        ),
        Instruction::new(
            Op::FOrdEqual,
            Some(bool_ty),
            Some(exponent_is_zero),
            vec![Operand::IdRef(exponentf), Operand::IdRef(zero)],
        ),
        Instruction::new(
            Op::LogicalAnd,
            Some(bool_ty),
            Some(zero_zero),
            vec![
                Operand::IdRef(base_is_zero),
                Operand::IdRef(exponent_is_zero),
            ],
        ),
        Instruction::new(
            Op::Select,
            Some(float_ty),
            Some(powf),
            vec![
                Operand::IdRef(zero_zero),
                Operand::IdRef(one),
                Operand::IdRef(raw_powf),
            ],
        ),
        Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(powf)],
        ),
    ]
}

/// f32 twin of [`half_abs_pow`], but SIGN-PRESERVING: `sign(base) * pow(|base|, exponent)` emitted
/// directly at the (scalar or vector) f32 result type, no half<->float widening. Byte-diffing Apple's
/// `air.fast_pow.f32` goldens against GLSL Pow shows the magnitude equals `pow(|base|, y)` exactly and
/// the result carries the SIGN of the base (a negative base yields a negative result), whereas GLSL
/// Pow leaves x < 0 undefined (NaN on the Apple GPU via MoltenVK). `FSign` reapplies that sign; a
/// non-negative base is unchanged (`sign>=0`, `|x|==x`), so no currently-passing case regresses.
/// AIR also treats `pow(0, 0)` as `1`, so guard that case around GLSL Pow's undefined edge.
pub(in crate::passes) fn f32_abs_pow(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    base: Word,
    exponent: Word,
) -> Vec<Instruction> {
    let ext = ctx.glsl();
    let abs_base = ctx.module.fresh_id();
    let sign = ctx.module.fresh_id();
    let raw_mag = ctx.module.fresh_id();
    let mag = ctx.module.fresh_id();
    let signed = ctx.module.fresh_id();
    let base_is_zero = ctx.module.fresh_id();
    let exponent_is_zero = ctx.module.fresh_id();
    let zero_zero = ctx.module.fresh_id();
    let n = vector_len(ctx, rty);
    let bool_ty = if n > 1 {
        ctx.ty_vec_bool(n)
    } else {
        ctx.ty_bool()
    };
    let zero = splat_or_scalar(ctx, rty, 0.0, n);
    let one = splat_or_scalar(ctx, rty, 1.0, n);
    vec![
        Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(abs_base),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::FAbs as u32),
                Operand::IdRef(base),
            ],
        ),
        Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(raw_mag),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::Pow as u32),
                Operand::IdRef(abs_base),
                Operand::IdRef(exponent),
            ],
        ),
        Instruction::new(
            Op::FOrdEqual,
            Some(bool_ty),
            Some(base_is_zero),
            vec![Operand::IdRef(abs_base), Operand::IdRef(zero)],
        ),
        Instruction::new(
            Op::FOrdEqual,
            Some(bool_ty),
            Some(exponent_is_zero),
            vec![Operand::IdRef(exponent), Operand::IdRef(zero)],
        ),
        Instruction::new(
            Op::LogicalAnd,
            Some(bool_ty),
            Some(zero_zero),
            vec![
                Operand::IdRef(base_is_zero),
                Operand::IdRef(exponent_is_zero),
            ],
        ),
        Instruction::new(
            Op::Select,
            Some(rty),
            Some(mag),
            vec![
                Operand::IdRef(zero_zero),
                Operand::IdRef(one),
                Operand::IdRef(raw_mag),
            ],
        ),
        Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(sign),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::FSign as u32),
                Operand::IdRef(base),
            ],
        ),
        Instruction::new(
            Op::FMul,
            Some(rty),
            Some(signed),
            vec![Operand::IdRef(sign), Operand::IdRef(mag)],
        ),
        Instruction::new(
            Op::Select,
            Some(rty),
            Some(res),
            vec![
                Operand::IdRef(zero_zero),
                Operand::IdRef(one),
                Operand::IdRef(signed),
            ],
        ),
    ]
}

pub(in crate::passes) fn lower_fast_ldexp(
    ctx: &mut Ctx,
    name: &str,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if !is_f32_scalar(ctx, rty) {
        return Err(format!("{name} currently supports scalar f32 results"));
    }
    let mantissa_ty =
        value_result_type(ctx, args[0]).ok_or_else(|| format!("{name} mantissa has no type"))?;
    if mantissa_ty != rty {
        return Err(format!("{name} mantissa/result type mismatch"));
    }
    let exponent_ty =
        value_result_type(ctx, args[1]).ok_or_else(|| format!("{name} exponent has no type"))?;
    if !is_int_scalar_width(ctx, exponent_ty, 32) {
        return Err(format!("{name} exponent is not scalar i32"));
    }
    let ext = ctx.glsl();
    let exp_as_float = ctx.module.fresh_id();
    let scale = ctx.module.fresh_id();
    Ok(vec![
        Instruction::new(
            Op::ConvertSToF,
            Some(rty),
            Some(exp_as_float),
            vec![Operand::IdRef(args[1])],
        ),
        Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(scale),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::Exp2 as u32),
                Operand::IdRef(exp_as_float),
            ],
        ),
        Instruction::new(
            Op::FMul,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(args[0]), Operand::IdRef(scale)],
        ),
    ])
}

/// The 32-bit-float type matching `ty`'s shape: `half` -> `float`, `<N x half>` -> `<N x float>`,
/// an already-float type -> itself.
pub(in crate::passes) fn float_equivalent(ctx: &mut Ctx, ty: Word) -> Word {
    if let Some(def) = type_def_of(ctx, ty) {
        match def.class.opcode {
            Op::TypeFloat => {
                if def.operands.first() == Some(&Operand::LiteralBit32(16)) {
                    return ctx.ty_float();
                }
                return ty; // float32 already
            }
            Op::TypeVector => {
                if let (Some(Operand::IdRef(elem)), Some(Operand::LiteralBit32(n))) =
                    (def.operands.first(), def.operands.get(1))
                {
                    let n = *n;
                    if is_half_scalar(ctx, *elem) {
                        return ctx.ty_vecf(n);
                    }
                }
                return ty;
            }
            _ => {}
        }
    }
    ty
}

/// Scalar 0/1 constants matching the ELEMENT type of `rty`: half element -> half 0/1, else float
/// 0/1. Used for `saturate`/`clamp` edges so FClamp's operand types match its half/float result.
/// See the `air.fast_tanh` arm in `lower_call`: (exp2(x*2log2e) - 1) / (exp2(x*2log2e) + 1),
/// byte-faithful to Metal's overflow-to-NaN fast tanh.
pub(in crate::passes) fn lower_fast_tanh(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    x: Word,
) -> Result<Vec<Instruction>, String> {
    let elem = element_type(ctx, rty);
    let (scale_scalar, one_scalar) = if is_half_scalar(ctx, elem) {
        (ctx.const_half(2.885_39), ctx.const_half(1.0))
    } else {
        (ctx.const_float(2.885_39), ctx.const_float(1.0))
    };
    let (scale, one) = clamp_edges(ctx, rty, scale_scalar, one_scalar);
    let ext = ctx.glsl();
    let scaled = ctx.module.fresh_id();
    let t = ctx.module.fresh_id();
    let num = ctx.module.fresh_id();
    let den = ctx.module.fresh_id();
    Ok(vec![
        Instruction::new(
            Op::FMul,
            Some(rty),
            Some(scaled),
            vec![Operand::IdRef(x), Operand::IdRef(scale)],
        ),
        Instruction::new(
            Op::ExtInst,
            Some(rty),
            Some(t),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::Exp2 as u32),
                Operand::IdRef(scaled),
            ],
        ),
        Instruction::new(
            Op::FSub,
            Some(rty),
            Some(num),
            vec![Operand::IdRef(t), Operand::IdRef(one)],
        ),
        Instruction::new(
            Op::FAdd,
            Some(rty),
            Some(den),
            vec![Operand::IdRef(t), Operand::IdRef(one)],
        ),
        Instruction::new(
            Op::FDiv,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(num), Operand::IdRef(den)],
        ),
    ])
}

pub(in crate::passes) fn scalar_zero_one(ctx: &mut Ctx, rty: Word) -> (Word, Word) {
    let elem = element_type(ctx, rty);
    if is_half_scalar(ctx, elem) {
        (ctx.const_half(0.0), ctx.const_half(1.0))
    } else {
        (ctx.const_float(0.0), ctx.const_float(1.0))
    }
}
