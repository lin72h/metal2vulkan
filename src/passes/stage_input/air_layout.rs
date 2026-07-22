//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

pub(in crate::passes) fn texture_type_hints(
    params: &[(Word, Word)],
    stage: &Stage,
    frag: Option<&FragMeta>,
    vert: Option<&VertMeta>,
    kern: Option<&KernMeta>,
) -> HashMap<Word, (Dim, bool, ImageComp)> {
    params
        .iter()
        .enumerate()
        .filter_map(|(i, (pid, _))| {
            let idx = i as u32;
            let name = match stage {
                Stage::Fragment => frag.and_then(|m| m.texture_type_name(idx)),
                Stage::Vertex => vert.and_then(|m| m.texture_type_name(idx)),
                Stage::Kernel => kern.and_then(|m| m.texture_type_name(idx)),
            }?;
            let (dim, arrayed) = texture_arg_dim(name);
            Some((*pid, (dim, arrayed, texture_arg_comp(name))))
        })
        .collect()
}

pub(in crate::passes) fn array_type(
    defs: &HashMap<Word, Instruction>,
    ty: Word,
) -> Option<(Word, u32)> {
    let def = defs.get(&ty)?;
    if def.class.opcode != Op::TypeArray {
        return None;
    }
    let elem = match def.operands.first()? {
        Operand::IdRef(elem) => *elem,
        _ => return None,
    };
    let len_const = match def.operands.get(1)? {
        Operand::IdRef(len_const) => *len_const,
        _ => return None,
    };
    let len = defs
        .get(&len_const)
        .and_then(|constant| match constant.operands.first() {
            Some(Operand::LiteralBit32(len)) => Some(*len),
            _ => None,
        })?;
    Some((elem, len))
}

/// A fragment shader Input interface variable of integer (or 64-bit float) component type cannot be
/// interpolated and MUST carry a `Flat` decoration (VUID-StandaloneSpirv-Flat-04744). Returns true
/// for such a type, descending through vectors/matrices/arrays to the scalar component.
pub(in crate::passes) fn fragment_input_needs_flat(
    defs: &HashMap<Word, Instruction>,
    ty: Word,
) -> bool {
    let Some(def) = defs.get(&ty) else {
        return false;
    };
    match def.class.opcode {
        Op::TypeInt => true,
        Op::TypeFloat => type_float_width(defs, ty) == Some(64),
        Op::TypeVector | Op::TypeMatrix | Op::TypeArray => match def.operands.first() {
            Some(Operand::IdRef(elem)) => fragment_input_needs_flat(defs, *elem),
            _ => false,
        },
        _ => false,
    }
}

pub(in crate::passes) fn type_float_width(
    defs: &HashMap<Word, Instruction>,
    ty: Word,
) -> Option<u32> {
    let def = defs.get(&ty)?;
    (def.class.opcode == Op::TypeFloat).then(|| match def.operands.first() {
        Some(Operand::LiteralBit32(width)) => *width,
        _ => 32,
    })
}

pub(in crate::passes) fn type_int_width(
    defs: &HashMap<Word, Instruction>,
    ty: Word,
) -> Option<u32> {
    let def = defs.get(&ty)?;
    (def.class.opcode == Op::TypeInt).then(|| match def.operands.first() {
        Some(Operand::LiteralBit32(width)) => *width,
        _ => 32,
    })
}

pub(in crate::passes) fn is_backend_padding_array(
    defs: &HashMap<Word, Instruction>,
    ty: Word,
) -> bool {
    let Some((elem, _len)) = array_type(defs, ty) else {
        return false;
    };
    type_int_width(defs, elem) == Some(8)
}

pub(in crate::passes) fn bind_kernel_uint_builtin_once(
    ctx: &mut Ctx,
    var_slot: &mut Option<Word>,
    builtin: BuiltIn,
) -> Word {
    if let Some(var) = *var_slot {
        return var;
    }
    let uint_ty = ctx.ty_uint();
    let pptr = ctx.ty_ptr(StorageClass::Input, uint_ty);
    let var = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::Variable,
        Some(pptr),
        Some(var),
        vec![Operand::StorageClass(StorageClass::Input)],
    ));
    decorate_builtin(&mut ctx.module, var, builtin);
    ctx.interface.push(var);
    *var_slot = Some(var);
    var
}

pub(in crate::passes) fn bind_kernel_v3uint_builtin(ctx: &mut Ctx, builtin: BuiltIn) -> Word {
    let v3u = ctx.ty_vec_uint(3);
    let pptr = ctx.ty_ptr(StorageClass::Input, v3u);
    let var = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::Variable,
        Some(pptr),
        Some(var),
        vec![Operand::StorageClass(StorageClass::Input)],
    ));
    decorate_builtin(&mut ctx.module, var, builtin);
    ctx.interface.push(var);
    var
}

pub(in crate::passes) fn bind_kernel_v3uint_builtin_once(
    ctx: &mut Ctx,
    var_slot: &mut Option<Word>,
    builtin: BuiltIn,
) -> Word {
    if let Some(var) = *var_slot {
        return var;
    }
    let var = bind_kernel_v3uint_builtin(ctx, builtin);
    *var_slot = Some(var);
    var
}

pub(in crate::passes) fn bind_kernel_threads_per_grid(
    ctx: &mut Ctx,
    defs: &HashMap<Word, Instruction>,
    bindings: &mut Vec<(Word, ParamBinding)>,
    pid: Word,
    pty: Word,
    var: Word,
) {
    let v2u = ctx.ty_vec_uint(2);
    let v3u = ctx.ty_vec_uint(3);

    let vector_lanes = scalar_or_vector_component(defs, pty).and_then(|(_component, lanes)| lanes);
    let lanes = if pty == v3u || vector_lanes == Some(3) {
        3
    } else if pty == v2u || vector_lanes == Some(2) {
        2
    } else {
        1
    };
    let out_ty = if lanes == 2 && pty == v2u {
        v2u
    } else if lanes == 3 && pty == v3u {
        v3u
    } else {
        pty
    };
    bindings.push((
        pid,
        ParamBinding::LoadThreadsPerGrid {
            var,
            vec_ty: v3u,
            out_ty,
            lanes,
        },
    ));
}

pub(in crate::passes) fn color_input_index(
    stage: &Stage,
    frag: Option<&FragMeta>,
    idx: u32,
) -> Option<u32> {
    match stage {
        Stage::Fragment => match frag.and_then(|m| m.role_of(idx)) {
            Some(FragRole::ColorInput(index)) => Some(*index),
            _ => None,
        },
        _ => None,
    }
}

pub(in crate::passes) fn input_attachment_read_types(
    ctx: &mut Ctx,
    defs: &HashMap<Word, Instruction>,
    param_ty: Word,
) -> Result<(Word, Word), String> {
    let (component_ty, lanes) = scalar_or_vector_component(defs, param_ty)
        .ok_or_else(|| format!("color input param type {param_ty} is not scalar/vector"))?;
    if is_float_width(defs, component_ty, 16) {
        let sampled_ty = ctx.ty_float();
        let read_ty = lanes
            .map(|lane_count| ctx.ty_vecf(lane_count))
            .unwrap_or(sampled_ty);
        Ok((sampled_ty, read_ty))
    } else {
        Ok((component_ty, param_ty))
    }
}

pub(in crate::passes) fn scalar_or_vector_component(
    defs: &HashMap<Word, Instruction>,
    ty: Word,
) -> Option<(Word, Option<u32>)> {
    match defs.get(&ty).map(|inst| inst.class.opcode) {
        Some(Op::TypeVector) => {
            let inst = defs.get(&ty)?;
            let component = match inst.operands.first()? {
                Operand::IdRef(component) => *component,
                _ => return None,
            };
            let lanes = match inst.operands.get(1)? {
                Operand::LiteralBit32(lanes) => *lanes,
                _ => return None,
            };
            Some((component, Some(lanes)))
        }
        Some(Op::TypeFloat | Op::TypeInt | Op::TypeBool) => Some((ty, None)),
        _ => None,
    }
}

pub(in crate::passes) fn is_float_width(
    defs: &HashMap<Word, Instruction>,
    ty: Word,
    width: u32,
) -> bool {
    defs.get(&ty).is_some_and(|inst| {
        inst.class.opcode == Op::TypeFloat
            && inst.operands.first() == Some(&Operand::LiteralBit32(width))
    })
}
