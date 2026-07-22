//! Return-value lowering, output variables, and static-sampler materialization.

use super::*;
use crate::passes::stage_input::{decorate_builtin, decorate_location};

fn fragment_output_location(frag: Option<&FragMeta>, member_idx: usize) -> Option<u32> {
    match frag {
        Some(meta) => meta.render_target_location_for_member(member_idx as u32),
        None => Some(member_idx as u32),
    }
}

fn fragment_output_is_depth(frag: Option<&FragMeta>, member_idx: usize) -> bool {
    frag.map(|meta| meta.is_depth_member(member_idx as u32))
        .unwrap_or(false)
}

fn fragment_output_is_stencil(frag: Option<&FragMeta>, member_idx: usize) -> bool {
    frag.map(|meta| meta.is_stencil_member(member_idx as u32))
        .unwrap_or(false)
}

fn fragment_output_type(
    ctx: &mut Ctx,
    frag: Option<&FragMeta>,
    member_idx: usize,
    ty: Word,
    defs: &HashMap<Word, Instruction>,
) -> Word {
    let Some(name) = frag.and_then(|meta| meta.render_target_type_name(member_idx as u32)) else {
        return ty;
    };
    if !is_signed_int_air_type_name(name) {
        return ty;
    }
    signed_integer_type_like(ctx, ty, defs).unwrap_or(ty)
}

fn is_signed_int_air_type_name(name: &str) -> bool {
    let raw = name.trim();
    let raw = raw.strip_prefix("packed_").unwrap_or(raw);
    if raw == "int" {
        return true;
    }
    raw.strip_prefix("int")
        .and_then(|lanes| lanes.parse::<u32>().ok())
        .is_some()
}

fn signed_integer_type_like(
    ctx: &mut Ctx,
    ty: Word,
    defs: &HashMap<Word, Instruction>,
) -> Option<Word> {
    let def = defs.get(&ty)?;
    match def.class.opcode {
        Op::TypeInt => {
            if def.operands.first() != Some(&Operand::LiteralBit32(32)) {
                return None;
            }
            match def.operands.get(1) {
                Some(Operand::LiteralBit32(1)) => Some(ty),
                Some(Operand::LiteralBit32(0)) => Some(ctx.ty_sint()),
                _ => None,
            }
        }
        Op::TypeVector => {
            let elem = match def.operands.first()? {
                Operand::IdRef(elem) => *elem,
                _ => return None,
            };
            let lanes = match def.operands.get(1)? {
                Operand::LiteralBit32(lanes) => *lanes,
                _ => return None,
            };
            let elem_def = defs.get(&elem)?;
            if elem_def.class.opcode != Op::TypeInt
                || elem_def.operands.first() != Some(&Operand::LiteralBit32(32))
            {
                return None;
            }
            match elem_def.operands.get(1) {
                Some(Operand::LiteralBit32(1)) => Some(ty),
                Some(Operand::LiteralBit32(0)) => Some(ctx.ty_vec_sint(lanes)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn bitcast_for_store(
    ctx: &mut Ctx,
    stores: &mut Vec<Instruction>,
    value: Word,
    src_ty: Word,
    dst_ty: Word,
) -> Word {
    if src_ty == dst_ty {
        return value;
    }
    let cast = ctx.module.fresh_id();
    stores.push(Instruction::new(
        Op::Bitcast,
        Some(dst_ty),
        Some(cast),
        vec![Operand::IdRef(value)],
    ));
    cast
}

/// Rewrite the entry's OpReturnValue into stores to Output variable(s) + OpReturn. Fragment outputs
/// use the `air.render_target` location from metadata (including nonzero single-target coverage
/// passes). Vertex => member 0 of the output struct is gl_Position (BuiltIn), the remaining members
/// are Output varyings @ Location 0.. .
pub(in crate::passes) fn rewrite_return(
    ctx: &mut Ctx,
    entry_idx: usize,
    stage: &Stage,
    frag: Option<&FragMeta>,
    vert: Option<&VertMeta>,
    defs: &HashMap<Word, Instruction>,
) -> Result<(), String> {
    // Find the OpReturnValue (entry funcs have exactly one in these shaders).
    let mut ret_loc: Option<(usize, usize, Word)> = None; // (block, inst, value id)
    for (bi, blk) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in blk.instructions.iter().enumerate() {
            if inst.class.opcode == Op::ReturnValue {
                if let Some(Operand::IdRef(v)) = inst.operands.first() {
                    ret_loc = Some((bi, ii, *v));
                }
            }
        }
    }
    let Some((bi, ii, retval)) = ret_loc else {
        // void return (e.g. a discard-only shader): nothing to do.
        return Ok(());
    };

    // Determine the return value type.
    let ret_ty = ctx.module.functions[entry_idx]
        .def
        .as_ref()
        .and_then(|d| d.result_type)
        .ok_or_else(|| "entry function has no result type".to_string())?;

    let mut stores: Vec<Instruction> = vec![];
    let rdef = defs.get(&ret_ty).cloned();

    match stage {
        Stage::Fragment => {
            if let Some(def) = &rdef {
                if def.class.opcode == Op::TypeStruct {
                    // MRT/depth: one Output per modeled return member.
                    for (mi, op) in def.operands.clone().iter().enumerate() {
                        let Operand::IdRef(mty) = op else { continue };
                        enum OutKind {
                            Location(u32),
                            Depth,
                            Stencil,
                        }
                        let kind = if fragment_output_is_depth(frag, mi) {
                            Some(OutKind::Depth)
                        } else if fragment_output_is_stencil(frag, mi) {
                            Some(OutKind::Stencil)
                        } else {
                            fragment_output_location(frag, mi).map(OutKind::Location)
                        };
                        let Some(kind) = kind else { continue };
                        let output_ty = fragment_output_type(ctx, frag, mi, *mty, defs);
                        let var = make_output_var(ctx, output_ty);
                        match kind {
                            OutKind::Location(location) => {
                                decorate_location(&mut ctx.module, var, location)
                            }
                            OutKind::Depth => {
                                decorate_builtin(&mut ctx.module, var, BuiltIn::FragDepth);
                                ctx.writes_frag_depth = true;
                            }
                            OutKind::Stencil => {
                                decorate_builtin(&mut ctx.module, var, BuiltIn::FragStencilRefEXT)
                            }
                        }
                        ctx.interface.push(var);
                        let ext = ctx.module.fresh_id();
                        stores.push(Instruction::new(
                            Op::CompositeExtract,
                            Some(*mty),
                            Some(ext),
                            vec![Operand::IdRef(retval), Operand::LiteralBit32(mi as u32)],
                        ));
                        let ext = bitcast_for_store(ctx, &mut stores, ext, *mty, output_ty);
                        stores.push(Instruction::new(
                            Op::Store,
                            None,
                            None,
                            vec![Operand::IdRef(var), Operand::IdRef(ext)],
                        ));
                    }
                } else {
                    // single render target or bare depth scalar.
                    if fragment_output_is_depth(frag, 0) {
                        let var = make_output_var(ctx, ret_ty);
                        decorate_builtin(&mut ctx.module, var, BuiltIn::FragDepth);
                        ctx.writes_frag_depth = true;
                        ctx.interface.push(var);
                        stores.push(Instruction::new(
                            Op::Store,
                            None,
                            None,
                            vec![Operand::IdRef(var), Operand::IdRef(retval)],
                        ));
                    } else if fragment_output_is_stencil(frag, 0) {
                        let var = make_output_var(ctx, ret_ty);
                        decorate_builtin(&mut ctx.module, var, BuiltIn::FragStencilRefEXT);
                        ctx.interface.push(var);
                        stores.push(Instruction::new(
                            Op::Store,
                            None,
                            None,
                            vec![Operand::IdRef(var), Operand::IdRef(retval)],
                        ));
                    } else if let Some(location) = fragment_output_location(frag, 0) {
                        let output_ty = fragment_output_type(ctx, frag, 0, ret_ty, defs);
                        let var = make_output_var(ctx, output_ty);
                        decorate_location(&mut ctx.module, var, location);
                        ctx.interface.push(var);
                        let value = bitcast_for_store(ctx, &mut stores, retval, ret_ty, output_ty);
                        stores.push(Instruction::new(
                            Op::Store,
                            None,
                            None,
                            vec![Operand::IdRef(var), Operand::IdRef(value)],
                        ));
                    }
                }
            }
        }
        Stage::Vertex => {
            // Output struct roles come from `!air.vertex` metadata. Builtins such as point_size are
            // return members but do not consume user varying locations.
            if let Some(def) = &rdef {
                if def.class.opcode == Op::TypeStruct {
                    let mut fallback_loc = 0u32;
                    for (mi, op) in def.operands.clone().iter().enumerate() {
                        let Operand::IdRef(mty) = op else { continue };
                        enum OutKind {
                            Builtin(BuiltIn),
                            Location(u32),
                        }
                        let kind = match vert.and_then(|m| m.output_role_of(mi as u32)).cloned() {
                            Some(VertOutRole::Position) => OutKind::Builtin(BuiltIn::Position),
                            Some(VertOutRole::PointSize) => OutKind::Builtin(BuiltIn::PointSize),
                            Some(VertOutRole::ViewportArrayIndex) => {
                                OutKind::Builtin(BuiltIn::ViewportIndex)
                            }
                            Some(VertOutRole::Varying(l)) => {
                                fallback_loc = fallback_loc.max(l + 1);
                                OutKind::Location(l)
                            }
                            _ if mi == 0 => OutKind::Builtin(BuiltIn::Position),
                            _ => {
                                let loc = fallback_loc;
                                fallback_loc += 1;
                                OutKind::Location(loc)
                            }
                        };
                        let var = make_output_var(ctx, *mty);
                        match kind {
                            OutKind::Builtin(builtin) => {
                                decorate_builtin(&mut ctx.module, var, builtin)
                            }
                            OutKind::Location(loc) => decorate_location(&mut ctx.module, var, loc),
                        }
                        ctx.interface.push(var);
                        let ext = ctx.module.fresh_id();
                        stores.push(Instruction::new(
                            Op::CompositeExtract,
                            Some(*mty),
                            Some(ext),
                            vec![Operand::IdRef(retval), Operand::LiteralBit32(mi as u32)],
                        ));
                        stores.push(Instruction::new(
                            Op::Store,
                            None,
                            None,
                            vec![Operand::IdRef(var), Operand::IdRef(ext)],
                        ));
                    }
                } else {
                    // bare position vec4
                    let var = make_output_var(ctx, ret_ty);
                    decorate_builtin(&mut ctx.module, var, BuiltIn::Position);
                    ctx.interface.push(var);
                    stores.push(Instruction::new(
                        Op::Store,
                        None,
                        None,
                        vec![Operand::IdRef(var), Operand::IdRef(retval)],
                    ));
                }
            }
        }
        // A compute kernel returns void (it writes through its buffer pointers); there is no
        // OpReturnValue, so we never reach here for Stage::Kernel.
        Stage::Kernel => {}
    }

    // Replace the OpReturnValue with the stores + an OpReturn.
    let blk = &mut ctx.module.functions[entry_idx].blocks[bi];
    blk.instructions.remove(ii);
    let mut at = ii;
    for s in stores {
        blk.instructions.insert(at, s);
        at += 1;
    }
    blk.instructions
        .insert(at, Instruction::new(Op::Return, None, None, vec![]));
    Ok(())
}

/// Find the module-scope `__air_sampler_state` global (if any), replace it with a real sampler
/// resource variable (binding `*binding_ctr`, no initializer), drop the old global, and rewrite
/// uses in the entry body into an `OpLoad %sampler %var`.
pub(in crate::passes) fn handle_static_sampler(ctx: &mut Ctx, binding_ctr: &mut u32) {
    // A shader can embed SEVERAL static samplers (`__air_sampler_state`, `__air_sampler_state.31`,
    // ...) — one per distinct sampler-state the source used. Collect every such global by name prefix.
    let mut samp_globals: Vec<Word> = vec![];
    for inst in &ctx.module.debug_names {
        if inst.class.opcode == Op::Name {
            if let (Some(Operand::IdRef(id)), Some(Operand::LiteralString(s))) =
                (inst.operands.first(), inst.operands.get(1))
            {
                if s.trim_start_matches('@').starts_with("__air_sampler_state") {
                    samp_globals.push(*id);
                }
            }
        }
    }
    if samp_globals.is_empty() {
        return;
    }

    let sty = ctx.ty_sampler();
    let pptr = ctx.ty_ptr(StorageClass::UniformConstant, sty);

    for old_var in samp_globals {
        let sampler_state = static_sampler_word(ctx, old_var).map(AirStaticSamplerState::from_word);
        // One sampler resource per static sampler.
        let new_var = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::Variable,
            Some(pptr),
            Some(new_var),
            vec![Operand::StorageClass(StorageClass::UniformConstant)],
        ));
        decorate_binding(&mut ctx.module, new_var, *binding_ctr);
        *binding_ctr += 1;
        ctx.interface_buffer_var(new_var);

        // Drop the old global variable + its OpName/decorations (a dangling reference to its id would
        // be an undefined forward-reference at validation).
        ctx.module
            .types_global_values
            .retain(|i| i.result_id != Some(old_var));
        ctx.module
            .debug_names
            .retain(|i| i.operands.first() != Some(&Operand::IdRef(old_var)));
        ctx.module
            .annotations
            .retain(|i| i.operands.first() != Some(&Operand::IdRef(old_var)));

        // Rewrite each `OpBitcast ... %old_var` into `OpLoad %sampler %new_var`, preserving the
        // result id so the downstream sample call still finds its operand. The native AIR emitter can
        // also pass `%old_var` directly as the sample call's sampler operand; insert a load before
        // those uses and point the operand at the loaded sampler value.
        let load_count = ctx
            .module
            .functions
            .iter()
            .flat_map(|function| &function.blocks)
            .flat_map(|block| &block.instructions)
            .filter(|instruction| {
                !(instruction.class.opcode == Op::Bitcast
                    && instruction.operands.first() == Some(&Operand::IdRef(old_var)))
                    && instruction.operands.contains(&Operand::IdRef(old_var))
            })
            .count();
        let load_count = Word::try_from(load_count)
            .expect("SPIR-V module cannot reserve more than u32::MAX ids");
        let mut load_ids = ctx.module.reserve_ids(load_count);
        for func in &mut ctx.module.functions {
            for blk in &mut func.blocks {
                let mut ii = 0;
                while ii < blk.instructions.len() {
                    let inst = &mut blk.instructions[ii];
                    if inst.class.opcode == Op::Bitcast
                        && inst.operands.first() == Some(&Operand::IdRef(old_var))
                    {
                        let rid = inst.result_id;
                        *inst = Instruction::new(
                            Op::Load,
                            Some(sty),
                            rid,
                            vec![Operand::IdRef(new_var)],
                        );
                        if let (Some(rid), Some(state)) = (rid, sampler_state) {
                            ctx.sampler_states.insert(rid, state);
                        }
                        ii += 1;
                        continue;
                    }

                    let mut load_id = None;
                    for op in &mut inst.operands {
                        if *op == Operand::IdRef(old_var) {
                            let id = *load_id.get_or_insert_with(|| {
                                load_ids.next().expect(
                                    "reserved one sampler-load id per rewritten instruction",
                                )
                            });
                            *op = Operand::IdRef(id);
                        }
                    }
                    if let Some(id) = load_id {
                        blk.instructions.insert(
                            ii,
                            Instruction::new(
                                Op::Load,
                                Some(sty),
                                Some(id),
                                vec![Operand::IdRef(new_var)],
                            ),
                        );
                        if let Some(state) = sampler_state {
                            ctx.sampler_states.insert(id, state);
                        }
                        ii += 2;
                    } else {
                        ii += 1;
                    }
                }
            }
        }
        debug_assert!(load_ids.next().is_none());
    }
}

fn static_sampler_word(ctx: &Ctx, var: Word) -> Option<u64> {
    let initializer = ctx
        .module
        .types_global_values
        .iter()
        .find(|inst| inst.result_id == Some(var) && inst.class.opcode == Op::Variable)
        .and_then(|inst| inst.operands.get(1))
        .and_then(|operand| match operand {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        })?;
    static_sampler_word_from_const(ctx, initializer)
}

fn static_sampler_word_from_const(ctx: &Ctx, value: Word) -> Option<u64> {
    let inst = ctx
        .module
        .types_global_values
        .iter()
        .find(|inst| inst.result_id == Some(value))?;
    match inst.class.opcode {
        Op::Constant => match inst.operands.first()? {
            Operand::LiteralBit64(value) => Some(*value),
            Operand::LiteralBit32(value) => Some(*value as u64),
            _ => None,
        },
        Op::ConstantComposite => inst.operands.first().and_then(|operand| match operand {
            Operand::IdRef(id) => static_sampler_word_from_const(ctx, *id),
            _ => None,
        }),
        _ => None,
    }
}

/// Create an Output variable of element type `ty`.
fn make_output_var(ctx: &mut Ctx, ty: Word) -> Word {
    let pptr = ctx.ty_ptr(StorageClass::Output, ty);
    let var = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::Variable,
        Some(pptr),
        Some(var),
        vec![Operand::StorageClass(StorageClass::Output)],
    ));
    var
}
