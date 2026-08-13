//! Storage-image atomic lowering.

use super::*;

/// Lower the stable unsigned 2D texture fetch-max ABI to an image texel pointer plus `OpAtomicUMax`.
/// Metal exposes texture atomics through a four-lane ABI value, but the texture format is the scalar
/// `R32ui` atomic format: lane zero carries the operand/result and the remaining result lanes stay
/// undefined. The coordinate offset is applied before the 16-bit AIR coordinate is widened.
pub(in crate::passes) fn lower_atomic_texture_fetch_max(
    ctx: &mut Ctx,
    name: &str,
    res: Option<Word>,
    rty: Option<Word>,
    args: &[Word],
) -> Result<Vec<Instruction>, String> {
    if name != "air.atomic_fetch_max_explicit_texture_2d.i16.u.v4i32" {
        return Err(format!("unsupported texture atomic intrinsic: {name}"));
    }
    if args.len() != 6 {
        return Err(format!("{name} expects 6 operands"));
    }
    let res = res.ok_or_else(|| format!("{name} has no result"))?;
    let rty = rty.ok_or_else(|| format!("{name} has no result type"))?;
    let mut image = resolve_image_value(ctx, args[0]);
    if !image_is_storage(ctx, image) {
        image = single_storage_image_for_private_write(ctx, image)
            .ok_or_else(|| format!("{name} on non-storage image id {image}"))?;
    }
    let (_, _, comp) = image_shape_or_recorded(ctx, image);
    if comp != crate::passes::ImageComp::Uint {
        return Err(format!("{name} requires an unsigned integer storage image"));
    }
    let image_pointer = if value_is_pointer(ctx, image) {
        image
    } else {
        value_inst(ctx, image)
            .filter(|instruction| instruction.class.opcode == Op::Load)
            .and_then(|instruction| instruction.operands.first())
            .and_then(|operand| match operand {
                Operand::IdRef(pointer) => Some(*pointer),
                _ => None,
            })
            .ok_or_else(|| format!("{name} image has no descriptor pointer"))?
    };

    let mut out = Vec::new();
    let (coord, coord_ty) =
        coerce_image_coord32_typed(ctx, args[1], &mut out, "texture atomic coordinate")?;
    let (offset, offset_ty) =
        coerce_image_coord32_typed(ctx, args[2], &mut out, "texture atomic offset")?;
    if offset_ty != coord_ty {
        return Err(format!("{name} coordinate and offset shapes differ"));
    }
    let offset_coord = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::IAdd,
        Some(coord_ty),
        Some(offset_coord),
        vec![Operand::IdRef(coord), Operand::IdRef(offset)],
    ));

    let uint = ctx.ty_uint();
    let value_ty =
        value_result_type(ctx, args[3]).ok_or_else(|| format!("{name} value untyped"))?;
    if resolve::integer_shape(ctx, value_ty) != Some((32, 4)) {
        return Err(format!("{name} value must be a four-lane i32 vector"));
    }
    let value = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::CompositeExtract,
        Some(uint),
        Some(value),
        vec![Operand::IdRef(args[3]), Operand::LiteralBit32(0)],
    ));
    let pointer_ty = ctx.ty_ptr(StorageClass::Image, uint);
    let pointer = ctx.module.fresh_id();
    let sample = ctx.const_uint(0);
    out.push(Instruction::new(
        Op::ImageTexelPointer,
        Some(pointer_ty),
        Some(pointer),
        vec![
            Operand::IdRef(image_pointer),
            Operand::IdRef(offset_coord),
            Operand::IdRef(sample),
        ],
    ));
    let previous = ctx.module.fresh_id();
    let scope = ctx.const_uint(Scope::Device as u32);
    let semantics = ctx.const_uint(MemorySemantics::RELAXED.bits());
    out.push(Instruction::new(
        Op::AtomicUMax,
        Some(uint),
        Some(previous),
        vec![
            Operand::IdRef(pointer),
            Operand::IdScope(scope),
            Operand::IdMemorySemantics(semantics),
            Operand::IdRef(value),
        ],
    ));
    let undefined = ctx.module.fresh_id();
    out.push(Instruction::new(
        Op::Undef,
        Some(rty),
        Some(undefined),
        vec![],
    ));
    out.push(Instruction::new(
        Op::CompositeInsert,
        Some(rty),
        Some(res),
        vec![
            Operand::IdRef(previous),
            Operand::IdRef(undefined),
            Operand::LiteralBit32(0),
        ],
    ));
    Ok(out)
}
