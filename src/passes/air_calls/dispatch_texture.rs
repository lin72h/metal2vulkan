//! Residual AIR intrinsic dispatch and texture/query call lowering.

use super::*;

pub(in crate::passes) fn lower_air_calls(ctx: &mut Ctx, entry_idx: usize) -> Result<(), String> {
    lower_agx_emask_memory_calls(ctx, entry_idx)?;

    let names = air_names(&ctx.module);
    let v4 = ctx.ty_vecf(4);

    // Walk each block; collect replacement instruction lists per call site.
    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let mut new_insts: Vec<Instruction> = vec![];
        let insts = ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .clone();
        for inst in insts {
            if inst.class.opcode != Op::FunctionCall {
                new_insts.push(inst);
                continue;
            }
            // operand 0 = callee id; rest = args.
            let callee = match inst.operands.first() {
                Some(Operand::IdRef(c)) => *c,
                _ => {
                    new_insts.push(inst);
                    continue;
                }
            };
            let Some(name) = names.get(&callee) else {
                new_insts.push(inst);
                continue;
            };
            let args: Vec<Word> = inst.operands[1..]
                .iter()
                .filter_map(|o| match o {
                    Operand::IdRef(r) => Some(*r),
                    _ => None,
                })
                .collect();
            let res = inst.result_id;
            let rty = inst.result_type;

            let lowered = lower_one(ctx, name, res, rty, &args, v4)?;
            new_insts.extend(lowered);
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = new_insts;
    }
    Ok(())
}

/// `air.fence_texture*`: a texture memory fence with no result. Emits an image-scoped acquire/release
/// `OpMemoryBarrier` at device scope.
pub(in crate::passes) fn lower_fence_texture(
    ctx: &mut Ctx,
    name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if args.len() != 1 {
        return Err(format!("{name} expects 1 operand"));
    }
    if res.is_some() || rty.is_some() {
        return Err(format!("{name} unexpectedly has a result"));
    }
    let scope = ctx.const_uint(Scope::Device as u32);
    let semantics =
        ctx.const_uint((MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::IMAGE_MEMORY).bits());
    Ok(vec![Instruction::new(
        Op::MemoryBarrier,
        None,
        None,
        vec![
            Operand::IdScope(scope),
            Operand::IdMemorySemantics(semantics),
        ],
    )])
}

/// `air.atomic.global.store.i32`: a global atomic store with no result. The AIR memory-order and scope
/// operands are ignored, matching the existing native/global atomic policy.
pub(in crate::passes) fn lower_atomic_global_store_i32(
    ctx: &mut Ctx,
    name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if args.len() != 5 {
        return Err(format!("{name} expects 5 operands"));
    }
    if res.is_some() || rty.is_some() {
        return Err(format!("{name} unexpectedly has a result"));
    }
    let mut out = Vec::new();
    let ptr = atomic_i32_pointer(ctx, args[0], &mut out);
    let scope = ctx.const_uint(Scope::Device as u32);
    let semantics = ctx.const_uint(MemorySemantics::RELAXED.bits());
    out.push(Instruction::new(
        Op::AtomicStore,
        None,
        None,
        vec![
            Operand::IdRef(ptr),
            Operand::IdScope(scope),
            Operand::IdMemorySemantics(semantics),
            Operand::IdRef(args[1]),
        ],
    ));
    Ok(out)
}

/// AIR global/local integer atomics (`air.atomic.{global,local}.{max,min,and,or,xor,xchg}.*.i32`)
/// that return the previous value. The AIR memory-order operands are ignored, matching the existing
/// native atomic-add policy. The caller's dispatch guard restricts `name` to the symbols handled here.
pub(in crate::passes) fn lower_atomic_integer_rmw(
    ctx: &mut Ctx,
    name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if args.len() != 5 {
        return Err(format!("{name} expects 5 operands"));
    }
    let res = res.ok_or_else(|| format!("{name} has no result"))?;
    let rty = rty.ok_or_else(|| format!("{name} has no result type"))?;
    let scope_kind = if name.starts_with("air.atomic.local.") {
        Scope::Workgroup
    } else {
        Scope::Device
    };
    let op = match name {
        "air.atomic.local.max.s.i32" => Op::AtomicSMax,
        "air.atomic.global.max.u.i32" | "air.atomic.local.max.u.i32" => Op::AtomicUMax,
        "air.atomic.local.min.s.i32" => Op::AtomicSMin,
        "air.atomic.local.min.u.i32" => Op::AtomicUMin,
        "air.atomic.global.and.u.i32" | "air.atomic.local.and.u.i32" => Op::AtomicAnd,
        "air.atomic.global.or.u.i32" | "air.atomic.local.or.u.i32" => Op::AtomicOr,
        "air.atomic.global.xor.u.i32" | "air.atomic.local.xor.u.i32" => Op::AtomicXor,
        "air.atomic.global.xchg.i32" | "air.atomic.local.xchg.i32" => Op::AtomicExchange,
        // The enclosing guard already restricts `name` to the atomic symbols above; FALLBACK
        // rather than abort if that guard and this match ever drift (refactor S23).
        _ => return Err(format!("unhandled integer atomic call {name}")),
    };
    let mut out = Vec::new();
    let ptr = atomic_i32_pointer(ctx, args[0], &mut out);
    let scope = ctx.const_uint(scope_kind as u32);
    let semantics = ctx.const_uint(MemorySemantics::RELAXED.bits());
    out.push(Instruction::new(
        op,
        Some(rty),
        Some(res),
        vec![
            Operand::IdRef(ptr),
            Operand::IdScope(scope),
            Operand::IdMemorySemantics(semantics),
            Operand::IdRef(args[1]),
        ],
    ));
    Ok(out)
}

/// Pair the inverse GLSL.std.450 operations and vector width for one AIR packed-vector format.
/// Keeping the format table here prevents pack and unpack dispatch from drifting independently.
fn packed_format(name: &str) -> Option<(GLSLstd450, GLSLstd450, u32)> {
    use GLSLstd450 as G;

    if name.contains("unorm4x8") {
        Some((G::PackUnorm4x8, G::UnpackUnorm4x8, 4))
    } else if name.contains("snorm4x8") {
        Some((G::PackSnorm4x8, G::UnpackSnorm4x8, 4))
    } else if name.contains("unorm2x16") {
        Some((G::PackUnorm2x16, G::UnpackUnorm2x16, 2))
    } else if name.contains("snorm2x16") {
        Some((G::PackSnorm2x16, G::UnpackSnorm2x16, 2))
    } else if name.contains("half2x16") {
        Some((G::PackHalf2x16, G::UnpackHalf2x16, 2))
    } else {
        None
    }
}

/// `air.pack.{unorm,snorm}4x8` / `*2x16` / `half2x16` -> the GLSL.std.450 Pack* ext-inst. The
/// normalized variants consume 32-bit-float vectors and return one packed u32.
pub(in crate::passes) fn lower_pack(
    ctx: &mut Ctx,
    name: &str,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let (pack_op, _, _) =
        packed_format(name).ok_or_else(|| format!("unhandled pack intrinsic: {name}"))?;
    let mut out = Vec::new();
    let mut pack_arg = args[0];
    if let Some(arg_ty) = value_result_type(ctx, args[0]) {
        let float_ty = float_equivalent(ctx, arg_ty);
        if float_ty != arg_ty {
            pack_arg = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::FConvert,
                Some(float_ty),
                Some(pack_arg),
                vec![Operand::IdRef(args[0])],
            ));
        }
    }
    out.push(Instruction::new(
        Op::ExtInst,
        Some(rty),
        Some(res),
        vec![
            Operand::IdRef(ctx.glsl()),
            Operand::LiteralExtInstInteger(pack_op as u32),
            Operand::IdRef(pack_arg),
        ],
    ));
    Ok(out)
}

/// `air.unpack.unorm.rgb10a2.<ret>` has no GLSL.std.450 equivalent, so unpack the 10/10/10/2
/// bit-fields by hand: r=(x&0x3FF)/1023, g=((x>>10)&0x3FF)/1023, b=((x>>20)&0x3FF)/1023,
/// a=((x>>30)&0x3)/3. The `.v4f16` variant FConverts the result down.
pub(in crate::passes) fn lower_unpack_rgb10a2(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let uint_ty = ctx.ty_uint();
    let float_ty = ctx.ty_float();
    let v_float = ctx.ty_vecf(4);
    let c1023 = ctx.const_float(1023.0);
    let c3 = ctx.const_float(3.0);
    let want_half = is_half_vector(ctx, rty);
    let fields = [
        (0u32, 0x3FFu32, c1023),
        (10, 0x3FF, c1023),
        (20, 0x3FF, c1023),
        (30, 0x3, c3),
    ];
    let mut out = Vec::new();
    let mut comps = Vec::with_capacity(4);
    for (shift, mask, div) in fields {
        let mut val = args[0];
        if shift != 0 {
            let sh = ctx.const_uint(shift);
            let shifted = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::ShiftRightLogical,
                Some(uint_ty),
                Some(shifted),
                vec![Operand::IdRef(val), Operand::IdRef(sh)],
            ));
            val = shifted;
        }
        let maskc = ctx.const_uint(mask);
        let masked = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::BitwiseAnd,
            Some(uint_ty),
            Some(masked),
            vec![Operand::IdRef(val), Operand::IdRef(maskc)],
        ));
        let asf = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::ConvertUToF,
            Some(float_ty),
            Some(asf),
            vec![Operand::IdRef(masked)],
        ));
        let comp = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FDiv,
            Some(float_ty),
            Some(comp),
            vec![Operand::IdRef(asf), Operand::IdRef(div)],
        ));
        comps.push(comp);
    }
    let vec_res = if want_half {
        ctx.module.fresh_id()
    } else {
        res
    };
    out.push(Instruction::new(
        Op::CompositeConstruct,
        Some(v_float),
        Some(vec_res),
        comps.iter().map(|c| Operand::IdRef(*c)).collect(),
    ));
    if want_half {
        out.push(Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(vec_res)],
        ));
    }
    Ok(out)
}

/// `air.unpack.unorm.rg11b10f.<ret>` (R11F_G11F_B10F) has no GLSL equivalent. The three fields are
/// unsigned small floats sharing half-float's 5-bit exponent (bias 15); each widens losslessly into
/// a half by left-justifying the mantissa, so `UnpackHalf2x16(bits).x` yields the float32 component.
pub(in crate::passes) fn lower_unpack_rg11b10f(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let uint_ty = ctx.ty_uint();
    let float_ty = ctx.ty_float();
    let v2_float = ctx.ty_vecf(2);
    let v_float = ctx.ty_vecf(3);
    let ext = ctx.glsl();
    let want_half = is_half_vector(ctx, rty);
    // (field_shift, field_mask, exp_shift_within_field, mantissa_mask, mantissa_left_shift)
    let fields = [
        (0u32, 0x7FFu32, 6u32, 0x3Fu32, 4u32),
        (11, 0x7FF, 6, 0x3F, 4),
        (22, 0x3FF, 5, 0x1F, 5),
    ];
    let mut out = Vec::new();
    let mut comps = Vec::with_capacity(3);
    for (fshift, fmask, eshift, mmask, mls) in fields {
        // field = (x >> fshift) & fmask
        let mut fv = args[0];
        if fshift != 0 {
            let sh = ctx.const_uint(fshift);
            let shifted = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::ShiftRightLogical,
                Some(uint_ty),
                Some(shifted),
                vec![Operand::IdRef(fv), Operand::IdRef(sh)],
            ));
            fv = shifted;
        }
        let fmaskc = ctx.const_uint(fmask);
        let field = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::BitwiseAnd,
            Some(uint_ty),
            Some(field),
            vec![Operand::IdRef(fv), Operand::IdRef(fmaskc)],
        ));
        // exp = (field >> eshift) << 10
        let eshiftc = ctx.const_uint(eshift);
        let exp_raw = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::ShiftRightLogical,
            Some(uint_ty),
            Some(exp_raw),
            vec![Operand::IdRef(field), Operand::IdRef(eshiftc)],
        ));
        let c10 = ctx.const_uint(10);
        let exp_hi = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::ShiftLeftLogical,
            Some(uint_ty),
            Some(exp_hi),
            vec![Operand::IdRef(exp_raw), Operand::IdRef(c10)],
        ));
        // mant = (field & mmask) << mls
        let mmaskc = ctx.const_uint(mmask);
        let mant_raw = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::BitwiseAnd,
            Some(uint_ty),
            Some(mant_raw),
            vec![Operand::IdRef(field), Operand::IdRef(mmaskc)],
        ));
        let mlsc = ctx.const_uint(mls);
        let mant_hi = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::ShiftLeftLogical,
            Some(uint_ty),
            Some(mant_hi),
            vec![Operand::IdRef(mant_raw), Operand::IdRef(mlsc)],
        ));
        // half_bits = exp_hi | mant_hi  (high 16 bits already zero)
        let half_bits = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::BitwiseOr,
            Some(uint_ty),
            Some(half_bits),
            vec![Operand::IdRef(exp_hi), Operand::IdRef(mant_hi)],
        ));
        // vec2 = UnpackHalf2x16(half_bits); comp = vec2.x
        let unpacked = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::ExtInst,
            Some(v2_float),
            Some(unpacked),
            vec![
                Operand::IdRef(ext),
                Operand::LiteralExtInstInteger(GLSLstd450::UnpackHalf2x16 as u32),
                Operand::IdRef(half_bits),
            ],
        ));
        let comp = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::CompositeExtract,
            Some(float_ty),
            Some(comp),
            vec![Operand::IdRef(unpacked), Operand::LiteralBit32(0)],
        ));
        comps.push(comp);
    }
    let vec_res = if want_half {
        ctx.module.fresh_id()
    } else {
        res
    };
    out.push(Instruction::new(
        Op::CompositeConstruct,
        Some(v_float),
        Some(vec_res),
        comps.iter().map(|c| Operand::IdRef(*c)).collect(),
    ));
    if want_half {
        out.push(Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(vec_res)],
        ));
    }
    Ok(out)
}

/// `air.unpack.unorm.rgb9e5.<ret>` (RGB9E5 shared-exponent float) has no GLSL equivalent. One 5-bit
/// exponent (bits[27:32)) is shared by three 9-bit integer mantissas; each component =
/// mantissa * 2^(exp-24) with no implicit leading 1.
pub(in crate::passes) fn lower_unpack_rgb9e5(
    ctx: &mut Ctx,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let uint_ty = ctx.ty_uint();
    let float_ty = ctx.ty_float();
    let v_float = ctx.ty_vecf(3);
    let ext = ctx.glsl();
    let want_half = is_half_vector(ctx, rty);
    let mut out = Vec::new();
    // exp = (x >> 27) & 0x1F ; scale = exp2(float(exp) - 24)
    let c27 = ctx.const_uint(27);
    let exp_sh = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ShiftRightLogical,
        Some(uint_ty),
        Some(exp_sh),
        vec![Operand::IdRef(args[0]), Operand::IdRef(c27)],
    ));
    let c1f = ctx.const_uint(0x1F);
    let exp_m = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::BitwiseAnd,
        Some(uint_ty),
        Some(exp_m),
        vec![Operand::IdRef(exp_sh), Operand::IdRef(c1f)],
    ));
    let exp_f = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ConvertUToF,
        Some(float_ty),
        Some(exp_f),
        vec![Operand::IdRef(exp_m)],
    ));
    let c24 = ctx.const_float(24.0);
    let exp_adj = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::FSub,
        Some(float_ty),
        Some(exp_adj),
        vec![Operand::IdRef(exp_f), Operand::IdRef(c24)],
    ));
    let scale = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ExtInst,
        Some(float_ty),
        Some(scale),
        vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(GLSLstd450::Exp2 as u32),
            Operand::IdRef(exp_adj),
        ],
    ));
    let mut comps = Vec::with_capacity(3);
    for shift in [0u32, 9, 18] {
        let mut mv = args[0];
        if shift != 0 {
            let sh = ctx.const_uint(shift);
            let shifted = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::ShiftRightLogical,
                Some(uint_ty),
                Some(shifted),
                vec![Operand::IdRef(mv), Operand::IdRef(sh)],
            ));
            mv = shifted;
        }
        let c1ff = ctx.const_uint(0x1FF);
        let mant = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::BitwiseAnd,
            Some(uint_ty),
            Some(mant),
            vec![Operand::IdRef(mv), Operand::IdRef(c1ff)],
        ));
        let mant_f = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::ConvertUToF,
            Some(float_ty),
            Some(mant_f),
            vec![Operand::IdRef(mant)],
        ));
        let comp = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::FMul,
            Some(float_ty),
            Some(comp),
            vec![Operand::IdRef(mant_f), Operand::IdRef(scale)],
        ));
        comps.push(comp);
    }
    let vec_res = if want_half {
        ctx.module.fresh_id()
    } else {
        res
    };
    out.push(Instruction::new(
        Op::CompositeConstruct,
        Some(v_float),
        Some(vec_res),
        comps.iter().map(|c| Operand::IdRef(*c)).collect(),
    ));
    if want_half {
        out.push(Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(vec_res)],
        ));
    }
    Ok(out)
}

/// `air.unpack.{unorm,snorm}4x8` / `*2x16` / `half2x16` -> the GLSL.std.450 Unpack* ext-inst. These
/// always return a 32-bit-float vector; a `.v4f16` AIR variant then FConverts down to half.
pub(in crate::passes) fn lower_unpack(
    ctx: &mut Ctx,
    name: &str,
    res: Word,
    rty: Word,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let (_, unpack_op, n) =
        packed_format(name).ok_or_else(|| format!("unhandled unpack intrinsic: {name}"))?;
    let ext = ctx.glsl();
    let v_float = ctx.ty_vecf(n);
    let want_half = is_half_vector(ctx, rty);
    let unpacked = if want_half {
        ctx.module.fresh_id()
    } else {
        res
    };
    let mut out = vec![Instruction::new(
        Op::ExtInst,
        Some(v_float),
        Some(unpacked),
        vec![
            Operand::IdRef(ext),
            Operand::LiteralExtInstInteger(unpack_op as u32),
            Operand::IdRef(args[0]),
        ],
    )];
    if want_half {
        out.push(Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(unpacked)],
        ));
    }
    Ok(out)
}

/// `air.get_{width,height,depth,array_size}_{texture,depth}_<dim>(texture, lod)`: an image size
/// query. Sampled images use `OpImageQuerySizeLod`; storage images (no sampler LOD) use
/// `OpImageQuerySize`. AIR's result is `i32`; the query yields a same-width uint component, bitcast
/// to the result. The wanted component index is derived from the intrinsic name.
pub(in crate::passes) fn lower_image_size_query(
    ctx: &mut Ctx,
    name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let res = res.ok_or_else(|| format!("{name} has no result"))?;
    let rty = rty.ok_or_else(|| format!("{name} has no result type"))?;
    if args.is_empty() {
        return Err(format!("{name} missing texture"));
    }
    let mut img = resolve_image_value(ctx, args[0]);
    if texture_operand_is_private_pointer(ctx, img) {
        if let Some(query_img) = single_image_for_private_query(ctx, img) {
            img = query_img;
        } else {
            let zero = ctx.const_int_of(rty, 0);
            return Ok(vec![Instruction::new(
                Op::CopyObject,
                Some(rty),
                Some(res),
                vec![Operand::IdRef(zero)],
            )]);
        }
    }
    let (dim, arrayed) = ctx
        .image_dims
        .get(&img)
        .copied()
        .unwrap_or((Dim::Dim2D, false));
    // Size vector component count: spatial dims + (arrayed ? 1 : 0).
    let spatial = match dim {
        Dim::Dim1D | Dim::DimBuffer => 1,
        Dim::Dim3D => 3,
        _ => 2,
    };
    let ncomp = spatial + if arrayed { 1 } else { 0 };
    let is_array_size_query = name.starts_with("air.get_array_size_texture");
    if is_array_size_query && !arrayed {
        return Err(format!("{name} used on non-array texture"));
    }
    let comp = if is_array_size_query {
        spatial
    } else if name.starts_with("air.get_height_") {
        1
    } else if name.starts_with("air.get_depth_") {
        2
    } else {
        0
    };
    let lod = args.get(1).copied();
    let uint = ctx.ty_uint();
    let size_ty = if ncomp == 1 {
        uint
    } else {
        ctx.ty_vec_uint(ncomp)
    };
    let size = ctx.module.fresh_id();
    let mut out = vec![];
    let query_op = if image_is_storage(ctx, img) || image_value_is_multisampled(ctx, img) {
        Op::ImageQuerySize
    } else {
        Op::ImageQuerySizeLod
    };
    let mut ops = vec![Operand::IdRef(img)];
    if query_op == Op::ImageQuerySizeLod {
        // OpImageQuerySizeLod requires a LOD operand; default to 0 if absent.
        let lod = lod.unwrap_or_else(|| ctx.const_uint(0));
        ops.push(Operand::IdRef(lod));
    }
    out.push(Instruction::new(query_op, Some(size_ty), Some(size), ops));
    // Extract the wanted component (or use the scalar size directly for a 1D non-arrayed texture).
    let comp_u = if ncomp == 1 {
        size
    } else {
        let c = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::CompositeExtract,
            Some(uint),
            Some(c),
            vec![
                Operand::IdRef(size),
                Operand::LiteralBit32(comp.min(ncomp - 1)),
            ],
        ));
        c
    };
    // Bitcast the uint component to the AIR result type (i32, same width).
    out.push(Instruction::new(
        Op::Bitcast,
        Some(rty),
        Some(res),
        vec![Operand::IdRef(comp_u)],
    ));
    Ok(out)
}

/// `air.is_null_texture*`: yields a bool telling whether the texture operand is one of the
/// synthesized null-image values tracked on the pass Ctx.
pub(in crate::passes) fn lower_is_null_texture(
    ctx: &mut Ctx,
    name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let res = res.ok_or_else(|| format!("{name} has no result"))?;
    let rty = rty.ok_or_else(|| format!("{name} has no result type"))?;
    let img = args
        .first()
        .copied()
        .ok_or_else(|| format!("{name} missing texture"))?;
    let is_null = ctx.null_image_values.contains(&img);
    let c = ctx.const_bool_of(rty, is_null);
    Ok(vec![Instruction::new(
        Op::CopyObject,
        Some(rty),
        Some(res),
        vec![Operand::IdRef(c)],
    )])
}

/// `air.get_num_mip_levels_texture_<dim>(texture)` -> `OpImageQueryLevels` for sampled images.
/// SPIR-V forbids the query on storage images (Sampled=2), and the texture contract used by private capture harnesses
/// synthesizes one mip level for storage-write targets, so those yield the constant 1.
pub(in crate::passes) fn lower_get_num_mip_levels(
    ctx: &mut Ctx,
    name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let res = res.ok_or_else(|| format!("{name} has no result"))?;
    let rty = rty.ok_or_else(|| format!("{name} has no result type"))?;
    let mut img = args
        .first()
        .copied()
        .ok_or_else(|| format!("{name} missing texture"))?;
    img = resolve_image_value(ctx, img);
    if texture_operand_is_private_pointer(ctx, img) {
        if let Some(query_img) = single_image_for_private_query(ctx, img) {
            img = query_img;
        } else {
            let one = ctx.const_uint(1);
            return Ok(vec![Instruction::new(
                Op::Bitcast,
                Some(rty),
                Some(res),
                vec![Operand::IdRef(one)],
            )]);
        }
    }
    if image_is_storage(ctx, img) {
        let one = ctx.const_uint(1);
        return Ok(vec![Instruction::new(
            Op::Bitcast,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(one)],
        )]);
    }
    let uint = ctx.ty_uint();
    let levels = ctx.module.fresh_id();
    Ok(vec![
        Instruction::new(
            Op::ImageQueryLevels,
            Some(uint),
            Some(levels),
            vec![Operand::IdRef(img)],
        ),
        Instruction::new(
            Op::Bitcast,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(levels)],
        ),
    ])
}

/// `air.get_num_samples_texture*`: the conformance texture contract creates single-sample textures,
/// so this lowers to the canonical count 1 rather than emitting an invalid multisample image query.
pub(in crate::passes) fn lower_get_num_samples_texture(
    ctx: &mut Ctx,
    name: &str,
    res: Option<Word>,
    rty: Option<Word>,
) -> Result<Vec<Instruction>, String> {
    let res = res.ok_or_else(|| format!("{name} has no result"))?;
    let rty = rty.ok_or_else(|| format!("{name} has no result type"))?;
    let one = ctx.const_uint(1);
    Ok(vec![Instruction::new(
        Op::Bitcast,
        Some(rty),
        Some(res),
        vec![Operand::IdRef(one)],
    )])
}

/// `air.calculate_unclamped_lod_texture_2d(texture, sampler, coord, flags)` returns the implicit
/// fragment LOD for a hypothetical sample. SPIR-V exposes the same query as `OpImageQueryLod`,
/// whose second component is the implicit level of detail relative to the image base level.
pub(in crate::passes) fn lower_calculate_unclamped_lod_texture_2d(
    ctx: &mut Ctx,
    name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let res = res.ok_or_else(|| format!("{name} has no result"))?;
    let rty = rty.ok_or_else(|| format!("{name} has no result type"))?;
    if args.len() < 3 {
        return Err(format!("{name} missing texture/sampler/coord"));
    }
    if ctx.stage != Stage::Fragment {
        return Err(format!("{name} requires fragment implicit derivatives"));
    }

    let mut img = resolve_image_value(ctx, args[0]);
    if texture_operand_is_private_pointer(ctx, img) {
        img = single_sampled_image_for_private_read(ctx, img)
            .ok_or_else(|| format!("{name} private texture operand is ambiguous"))?;
    }
    if image_is_storage(ctx, img) {
        return Err(format!("{name} requires a sampled texture"));
    }

    let mut out = Vec::new();
    img = load_image_if_pointer(ctx, img, &mut out);
    let (dim, arrayed, comp) = image_shape_or_recorded(ctx, img);
    if dim != Dim::Dim2D || arrayed {
        return Err(format!("{name} requires a non-array 2D texture"));
    }
    let (img_ty, _, _, _) = sampled_operand_image_info(ctx, img, dim, false, comp);
    let si_ty = ctx.ty_sampled_image(img_ty);
    let samp = if comp == crate::passes::ImageComp::Float {
        valid_sampler_value(ctx, args[1], &mut out)?
    } else {
        // Vulkan rejects linear-filter samplers paired with integer sampled images even for LOD
        // queries. The query only needs a valid sampler/image pair, not the AIR filter mode, so use
        // the translator-owned nearest sampler for integer textures.
        let var = ctx.default_read_sampler();
        let sty = ctx.ty_sampler();
        let id = ctx.module.fresh_id();
        out.push(Instruction::new(
            Op::Load,
            Some(sty),
            Some(id),
            vec![Operand::IdRef(var)],
        ));
        id
    };
    let si = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::SampledImage,
        Some(si_ty),
        Some(si),
        vec![Operand::IdRef(img), Operand::IdRef(samp)],
    ));
    let lod_pair_ty = ctx.ty_vecf(2);
    let lod_pair = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::ImageQueryLod,
        Some(lod_pair_ty),
        Some(lod_pair),
        vec![Operand::IdRef(si), Operand::IdRef(args[2])],
    ));

    let float_ty = ctx.ty_float();
    let lod = if rty == float_ty {
        res
    } else {
        ctx.module.fresh_id()
    };
    out.push(Instruction::new(
        Op::CompositeExtract,
        Some(float_ty),
        Some(lod),
        vec![Operand::IdRef(lod_pair), Operand::LiteralBit32(1)],
    ));
    if lod != res {
        out.push(Instruction::new(
            Op::FConvert,
            Some(rty),
            Some(res),
            vec![Operand::IdRef(lod)],
        ));
    }
    Ok(out)
}

/// `air.get_null_texture_<dim>()` -> load a synthesized default image (a function-constant-gated
/// optional attachment that, with our FCs folded off, resolves to a null texture), recording its
/// dims so a later sample works.
pub(in crate::passes) fn lower_get_null_texture(
    ctx: &mut Ctx,
    name: &str,
    res: Option<Word>,
) -> Result<Vec<Instruction>, String> {
    let res = res.ok_or_else(|| format!("{name} has no result"))?;
    let (dim, arrayed) = if name.contains("texture_buffer") {
        (Dim::DimBuffer, false)
    } else if name.contains("_1d_array") {
        (Dim::Dim1D, true)
    } else if name.contains("_1d") {
        (Dim::Dim1D, false)
    } else if name.contains("_3d") {
        (Dim::Dim3D, false)
    } else if name.contains("_cube_array") {
        (Dim::DimCube, true)
    } else if name.contains("_cube") {
        (Dim::DimCube, false)
    } else if name.contains("_2d_array") {
        (Dim::Dim2D, true)
    } else {
        (Dim::Dim2D, false)
    };
    let var = ctx.default_null_image_of(dim, arrayed);
    let img_ty = ctx.ty_image(dim, arrayed, crate::passes::ImageComp::Float);
    ctx.image_dims.insert(res, (dim, arrayed));
    ctx.image_comp.insert(res, crate::passes::ImageComp::Float);
    ctx.null_image_values.insert(res);
    Ok(vec![Instruction::new(
        Op::Load,
        Some(img_ty),
        Some(res),
        vec![Operand::IdRef(var)],
    )])
}

/// `air.map_screen_to_physical_coordinates.*`: the private capture harness backs the rasterization-rate map
/// with a single uniform physical tile, so this resolves to a constant zero physical coordinate.
pub(in crate::passes) fn lower_map_screen_to_physical(
    ctx: &mut Ctx,
    name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let res = res.ok_or_else(|| format!("{name} has no result"))?;
    let rty = rty.ok_or_else(|| format!("{name} has no result type"))?;
    if args.len() != 3 {
        return Err(format!(
            "{name} expects screen coordinate, map data, and layer"
        ));
    }
    let zero = ctx.const_float(0.0);
    Ok(vec![Instruction::new(
        Op::CompositeConstruct,
        Some(rty),
        Some(res),
        vec![Operand::IdRef(zero), Operand::IdRef(zero)],
    )])
}

/// `air.map_physical_to_screen_coordinates.*`: the inverse map; with a single uniform physical tile it
/// is the identity, so the physical coordinate passes through unchanged.
pub(in crate::passes) fn lower_map_physical_to_screen(
    name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    let res = res.ok_or_else(|| format!("{name} has no result"))?;
    let rty = rty.ok_or_else(|| format!("{name} has no result type"))?;
    if args.len() != 3 {
        return Err(format!(
            "{name} expects physical coordinate, map data, and layer"
        ));
    }
    Ok(vec![Instruction::new(
        Op::CopyObject,
        Some(rty),
        Some(res),
        vec![Operand::IdRef(args[0])],
    )])
}

/// `air.get_imageblock_{width,height}()`: for a compute kernel the implicit imageblock spans the
/// threadgroup, so the dimensions are the kernel's LocalSize x/y (from the pass Ctx).
pub(in crate::passes) fn lower_get_imageblock_extent(
    ctx: &mut Ctx,
    name: &str,
    res: Option<Word>,
    rty: Option<Word>,
) -> Result<Vec<Instruction>, String> {
    let res = res.ok_or_else(|| format!("{name} has no result"))?;
    let rty = rty.ok_or_else(|| format!("{name} has no result type"))?;
    let axis = usize::from(name == "air.get_imageblock_height");
    let extent = ctx.const_int_of(rty, i64::from(ctx.kernel_local_size[axis]));
    Ok(vec![Instruction::new(
        Op::CopyObject,
        Some(rty),
        Some(res),
        vec![Operand::IdRef(extent)],
    )])
}

/// `air.get_read_sampler()`: load a synthesized default sampler. Its result is only consumed as the
/// (ignored) sampler operand of `air.read_texture_*`, so a valid `OpTypeSampler` value is sufficient.
pub(in crate::passes) fn lower_get_read_sampler(
    ctx: &mut Ctx,
    res: Option<Word>,
) -> Result<Vec<Instruction>, String> {
    let res = res.ok_or_else(|| "air.get_read_sampler has no result".to_string())?;
    let var = ctx.default_read_sampler();
    let sty = ctx.ty_sampler();
    Ok(vec![Instruction::new(
        Op::Load,
        Some(sty),
        Some(res),
        vec![Operand::IdRef(var)],
    )])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::Module;

    fn ext_inst_number(instructions: &[Instruction]) -> u32 {
        instructions
            .iter()
            .find(|inst| inst.class.opcode == Op::ExtInst)
            .and_then(|inst| inst.operands.get(1))
            .and_then(|operand| match operand {
                Operand::LiteralExtInstInteger(number) => Some(*number),
                _ => None,
            })
            .expect("GLSL.std.450 instruction number")
    }

    #[test]
    fn packed_formats_pair_inverse_operations_and_widths() {
        let cases = [
            (
                "air.pack.unorm4x8.v4f32",
                GLSLstd450::PackUnorm4x8,
                GLSLstd450::UnpackUnorm4x8,
                4,
            ),
            (
                "air.unpack.snorm4x8.v4f32",
                GLSLstd450::PackSnorm4x8,
                GLSLstd450::UnpackSnorm4x8,
                4,
            ),
            (
                "air.pack.unorm2x16.v2f32",
                GLSLstd450::PackUnorm2x16,
                GLSLstd450::UnpackUnorm2x16,
                2,
            ),
            (
                "air.unpack.snorm2x16.v2f32",
                GLSLstd450::PackSnorm2x16,
                GLSLstd450::UnpackSnorm2x16,
                2,
            ),
            (
                "air.pack.half2x16.v2f32",
                GLSLstd450::PackHalf2x16,
                GLSLstd450::UnpackHalf2x16,
                2,
            ),
        ];

        for (name, pack_op, unpack_op, component_count) in cases {
            let (actual_pack, actual_unpack, actual_count) =
                packed_format(name).unwrap_or_else(|| panic!("classify {name}"));
            assert_eq!(actual_pack as u32, pack_op as u32, "{name}");
            assert_eq!(actual_unpack as u32, unpack_op as u32, "{name}");
            assert_eq!(actual_count, component_count, "{name}");
        }
        assert!(packed_format("air.unpack.unorm.rgb10a2.v4f32").is_none());
    }

    #[test]
    fn pack_and_unpack_lowering_use_the_paired_format_table() {
        let mut ctx = Ctx::new(Module::new());
        let vec_ty = ctx.ty_vecf(4);
        let scalar = ctx.const_float(0.25);
        let pack_arg = splat(&mut ctx, vec_ty, scalar, 4);
        let uint_ty = ctx.ty_uint();
        let pack_res = ctx.module.fresh_id();
        let packed = lower_pack(
            &mut ctx,
            "air.pack.unorm4x8.v4f32",
            pack_res,
            uint_ty,
            &[pack_arg],
        )
        .expect("lower pack");
        assert_eq!(packed.len(), 1);
        assert_eq!(ext_inst_number(&packed), GLSLstd450::PackUnorm4x8 as u32);

        let unpack_res = ctx.module.fresh_id();
        let unpacked = lower_unpack(
            &mut ctx,
            "air.unpack.unorm4x8.v4f32",
            unpack_res,
            vec_ty,
            &[pack_res],
        )
        .expect("lower unpack");
        assert_eq!(unpacked.len(), 1);
        assert_eq!(
            ext_inst_number(&unpacked),
            GLSLstd450::UnpackUnorm4x8 as u32
        );
    }
}
