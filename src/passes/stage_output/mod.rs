//! Return-value lowering, output variables, and static-sampler materialization.

use super::*;
use crate::passes::stage_input::{decorate_builtin, decorate_location, scalar_or_vector_component};
use crate::reflect::{COLOR_INPUT_BINDING_BASE, SAMPLER_BINDING_BASE};

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
    if let Some((signed, lanes)) = air_integer_render_target_shape(name) {
        return int32_render_target_interface_type(ctx, signed, lanes, ty, defs).unwrap_or(ty);
    }
    ty
}

fn air_integer_render_target_shape(name: &str) -> Option<(bool, u32)> {
    let raw = name.trim();
    let raw = raw.strip_prefix("packed_").unwrap_or(raw);
    for (prefix, signed) in [
        ("ushort", false),
        ("short", true),
        ("uint", false),
        ("int", true),
        ("uchar", false),
        ("char", true),
    ] {
        let Some(rest) = raw.strip_prefix(prefix) else {
            continue;
        };
        let lanes = if rest.is_empty() {
            1
        } else {
            rest.parse::<u32>().ok()?
        };
        if (1..=4).contains(&lanes) {
            return Some((signed, lanes));
        }
    }
    None
}

fn int32_render_target_interface_type(
    ctx: &mut Ctx,
    signed: bool,
    lanes: u32,
    ty: Word,
    defs: &HashMap<Word, Instruction>,
) -> Option<Word> {
    let (component, actual_lanes) = scalar_or_vector_component(defs, ty)?;
    let def = defs.get(&component)?;
    if def.class.opcode != Op::TypeInt {
        return None;
    }
    let actual_lanes = actual_lanes.unwrap_or(1);
    if actual_lanes != lanes {
        return None;
    }
    Some(match (signed, lanes) {
        (true, 1) => ctx.ty_sint(),
        (false, 1) => ctx.ty_uint(),
        (true, _) => ctx.ty_vec_sint(lanes),
        (false, _) => ctx.ty_vec_uint(lanes),
    })
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

fn value_for_store(
    ctx: &mut Ctx,
    stores: &mut Vec<Instruction>,
    value: Word,
    src_ty: Word,
    dst_ty: Word,
) -> Word {
    if src_ty == dst_ty {
        return value;
    }
    if let (Some((src_bits, _)), Some((dst_bits, dst_signed))) = (
        int_component_shape_live(ctx, src_ty),
        int_component_shape_live(ctx, dst_ty),
    ) {
        if src_bits != dst_bits {
            let converted = ctx.module.fresh_id();
            stores.push(Instruction::new(
                if dst_signed {
                    Op::SConvert
                } else {
                    Op::UConvert
                },
                Some(dst_ty),
                Some(converted),
                vec![Operand::IdRef(value)],
            ));
            return converted;
        }
    }
    bitcast_for_store(ctx, stores, value, src_ty, dst_ty)
}

fn int_component_shape_live(ctx: &Ctx, ty: Word) -> Option<(u32, bool)> {
    let mut scalar_ty = ty;
    let def = type_def_of(ctx, ty)?;
    if def.class.opcode == Op::TypeVector {
        scalar_ty = match def.operands.first()? {
            Operand::IdRef(component) => *component,
            _ => return None,
        };
    }
    let def = type_def_of(ctx, scalar_ty)?;
    if def.class.opcode != Op::TypeInt {
        return None;
    }
    let bits = match def.operands.first()? {
        Operand::LiteralBit32(bits) => *bits,
        _ => return None,
    };
    let signed = match def.operands.get(1)? {
        Operand::LiteralBit32(signed) => *signed != 0,
        _ => return None,
    };
    Some((bits, signed))
}

fn vertex_builtin_output_type(ctx: &mut Ctx, builtin: BuiltIn, member_ty: Word) -> Word {
    match builtin {
        BuiltIn::Layer | BuiltIn::ViewportIndex => ctx.ty_uint(),
        _ => member_ty,
    }
}

enum ClipDistanceOutputType {
    Scalar { array_ty: Word, elem_ty: Word },
    Array { array_ty: Word },
}

impl ClipDistanceOutputType {
    fn array_ty(&self) -> Word {
        match *self {
            ClipDistanceOutputType::Scalar { array_ty, .. }
            | ClipDistanceOutputType::Array { array_ty } => array_ty,
        }
    }
}

fn clip_distance_output_type(
    ctx: &mut Ctx,
    defs: &HashMap<Word, Instruction>,
    member_ty: Word,
) -> Result<ClipDistanceOutputType, String> {
    if let Some(def) = defs
        .get(&member_ty)
        .filter(|def| def.class.opcode == Op::TypeArray)
    {
        let elem = match def.operands.first() {
            Some(Operand::IdRef(elem)) => *elem,
            _ => {
                return Err(format!(
                    "clip_distance output type {member_ty} has no array element type"
                ));
            }
        };
        if !defs.get(&elem).is_some_and(|def| {
            def.class.opcode == Op::TypeFloat
                && def.operands.first() == Some(&Operand::LiteralBit32(32))
        }) {
            return Err(format!(
                "clip_distance output array type {member_ty} is not float[]"
            ));
        }
        return Ok(ClipDistanceOutputType::Array {
            array_ty: member_ty,
        });
    }

    let (component, lanes) = scalar_or_vector_component(defs, member_ty).ok_or_else(|| {
        format!("clip_distance output type {member_ty} is not scalar/vector/array")
    })?;
    if lanes.is_some() {
        return Err("vector clip_distance outputs are not yet supported".to_string());
    }
    if !defs.get(&component).is_some_and(|def| {
        def.class.opcode == Op::TypeFloat
            && def.operands.first() == Some(&Operand::LiteralBit32(32))
    }) {
        return Err(format!(
            "clip_distance output type {member_ty} is not scalar float"
        ));
    }
    Ok(ClipDistanceOutputType::Scalar {
        array_ty: ctx.ty_array(component, 1),
        elem_ty: component,
    })
}

fn value_is_statically_undef(ctx: &Ctx, value: Word) -> bool {
    value_def_instruction(ctx, value)
        .map(|def| def.class.opcode == Op::Undef)
        .unwrap_or(false)
}

fn composite_member_is_statically_undef(ctx: &Ctx, composite: Word, member: u32) -> bool {
    let Some(def) = value_def_instruction(ctx, composite) else {
        return false;
    };
    match def.class.opcode {
        Op::Undef => true,
        Op::CompositeConstruct => def
            .operands
            .get(member as usize)
            .and_then(|operand| match operand {
                Operand::IdRef(id) => Some(value_is_statically_undef(ctx, *id)),
                _ => None,
            })
            .unwrap_or(false),
        Op::CompositeInsert => {
            let Some(Operand::IdRef(inserted)) = def.operands.first() else {
                return false;
            };
            let Some(Operand::IdRef(base)) = def.operands.get(1) else {
                return false;
            };
            let inserted_member = def.operands.get(2).and_then(|operand| match operand {
                Operand::LiteralBit32(index) => Some(*index),
                _ => None,
            });
            match inserted_member {
                Some(index) if index == member => value_is_statically_undef(ctx, *inserted),
                Some(_) => composite_member_is_statically_undef(ctx, *base, member),
                None => false,
            }
        }
        _ => false,
    }
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
    // Find every OpReturnValue. Loop-budgeted validation candidates can add guarded early exits, so
    // the entry may have more than one return-value terminator.
    let mut ret_locs: Vec<(usize, usize, Word)> = Vec::new(); // (block, inst, value id)
    for (bi, blk) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
        for (ii, inst) in blk.instructions.iter().enumerate() {
            if inst.class.opcode == Op::ReturnValue {
                if let Some(Operand::IdRef(v)) = inst.operands.first() {
                    ret_locs.push((bi, ii, *v));
                }
            }
        }
    }
    if ret_locs.is_empty() {
        // void return (e.g. a discard-only shader): nothing to do.
        return Ok(());
    }

    // Determine the return value type.
    let ret_ty = ctx.module.functions[entry_idx]
        .def
        .as_ref()
        .and_then(|d| d.result_type)
        .ok_or_else(|| "entry function has no result type".to_string())?;

    enum OutputWrite {
        Direct {
            var: Word,
            src_ty: Word,
            dst_ty: Word,
        },
        Extract {
            var: Word,
            member: u32,
            src_ty: Word,
            dst_ty: Word,
        },
        ClipDistanceExtract {
            var: Word,
            member: u32,
            src_ty: Word,
            elem_ty: Word,
        },
    }

    impl OutputWrite {
        fn stores(&self, ctx: &mut Ctx, retval: Word) -> Vec<Instruction> {
            let mut stores = Vec::new();
            let (var, value, src_ty, dst_ty) = match *self {
                OutputWrite::Direct {
                    var,
                    src_ty,
                    dst_ty,
                } => {
                    if value_is_statically_undef(ctx, retval) {
                        return stores;
                    }
                    (var, retval, src_ty, dst_ty)
                }
                OutputWrite::Extract {
                    var,
                    member,
                    src_ty,
                    dst_ty,
                } => {
                    if composite_member_is_statically_undef(ctx, retval, member) {
                        return stores;
                    }
                    let ext = ctx.module.fresh_id();
                    stores.push(Instruction::new(
                        Op::CompositeExtract,
                        Some(src_ty),
                        Some(ext),
                        vec![Operand::IdRef(retval), Operand::LiteralBit32(member)],
                    ));
                    (var, ext, src_ty, dst_ty)
                }
                OutputWrite::ClipDistanceExtract {
                    var,
                    member,
                    src_ty,
                    elem_ty,
                } => {
                    if composite_member_is_statically_undef(ctx, retval, member) {
                        return stores;
                    }
                    let ext = ctx.module.fresh_id();
                    stores.push(Instruction::new(
                        Op::CompositeExtract,
                        Some(src_ty),
                        Some(ext),
                        vec![Operand::IdRef(retval), Operand::LiteralBit32(member)],
                    ));
                    let zero = ctx.const_uint(0);
                    let ptr_ty = ctx.ty_ptr(StorageClass::Output, elem_ty);
                    let ptr = ctx.module.fresh_id();
                    stores.push(Instruction::new(
                        Op::AccessChain,
                        Some(ptr_ty),
                        Some(ptr),
                        vec![Operand::IdRef(var), Operand::IdRef(zero)],
                    ));
                    let value = value_for_store(ctx, &mut stores, ext, src_ty, elem_ty);
                    stores.push(Instruction::new(
                        Op::Store,
                        None,
                        None,
                        vec![Operand::IdRef(ptr), Operand::IdRef(value)],
                    ));
                    return stores;
                }
            };
            let value = value_for_store(ctx, &mut stores, value, src_ty, dst_ty);
            stores.push(Instruction::new(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(var), Operand::IdRef(value)],
            ));
            stores
        }
    }

    let mut outputs: Vec<OutputWrite> = vec![];
    let rdef = defs.get(&ret_ty).cloned();

    match stage {
        Stage::Fragment => {
            if let Some(def) = &rdef {
                if def.class.opcode == Op::TypeStruct {
                    // MRT/depth: one Output per modeled return member.
                    for (mi, op) in def.operands.clone().iter().enumerate() {
                        let Operand::IdRef(mty) = op else { continue };
                        #[derive(Clone, Copy)]
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
                        outputs.push(OutputWrite::Extract {
                            var,
                            member: mi as u32,
                            src_ty: *mty,
                            dst_ty: output_ty,
                        });
                    }
                } else {
                    // single render target or bare depth scalar.
                    if fragment_output_is_depth(frag, 0) {
                        let var = make_output_var(ctx, ret_ty);
                        decorate_builtin(&mut ctx.module, var, BuiltIn::FragDepth);
                        ctx.writes_frag_depth = true;
                        ctx.interface.push(var);
                        outputs.push(OutputWrite::Direct {
                            var,
                            src_ty: ret_ty,
                            dst_ty: ret_ty,
                        });
                    } else if fragment_output_is_stencil(frag, 0) {
                        let var = make_output_var(ctx, ret_ty);
                        decorate_builtin(&mut ctx.module, var, BuiltIn::FragStencilRefEXT);
                        ctx.interface.push(var);
                        outputs.push(OutputWrite::Direct {
                            var,
                            src_ty: ret_ty,
                            dst_ty: ret_ty,
                        });
                    } else if let Some(location) = fragment_output_location(frag, 0) {
                        let output_ty = fragment_output_type(ctx, frag, 0, ret_ty, defs);
                        let var = make_output_var(ctx, output_ty);
                        decorate_location(&mut ctx.module, var, location);
                        ctx.interface.push(var);
                        outputs.push(OutputWrite::Direct {
                            var,
                            src_ty: ret_ty,
                            dst_ty: output_ty,
                        });
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
                        #[derive(Clone, Copy)]
                        enum OutKind {
                            Builtin(BuiltIn),
                            Location(u32),
                        }
                        let kind = match vert.and_then(|m| m.output_role_of(mi as u32)).cloned() {
                            Some(VertOutRole::Position) => OutKind::Builtin(BuiltIn::Position),
                            Some(VertOutRole::PointSize) => OutKind::Builtin(BuiltIn::PointSize),
                            Some(VertOutRole::ClipDistance) => {
                                OutKind::Builtin(BuiltIn::ClipDistance)
                            }
                            Some(VertOutRole::ViewportArrayIndex) => {
                                OutKind::Builtin(BuiltIn::ViewportIndex)
                            }
                            Some(VertOutRole::RenderTargetArrayIndex) => {
                                OutKind::Builtin(BuiltIn::Layer)
                            }
                            Some(VertOutRole::Varying(l)) => {
                                fallback_loc = fallback_loc.max(l + 1);
                                OutKind::Location(l)
                            }
                            Some(VertOutRole::FunctionConstantDisabled) => continue,
                            _ if mi == 0 => OutKind::Builtin(BuiltIn::Position),
                            _ => {
                                let loc = fallback_loc;
                                fallback_loc += 1;
                                OutKind::Location(loc)
                            }
                        };
                        let clip_distance_ty = match kind {
                            OutKind::Builtin(BuiltIn::ClipDistance) => {
                                Some(clip_distance_output_type(ctx, defs, *mty)?)
                            }
                            _ => None,
                        };
                        let output_ty = match (kind, clip_distance_ty.as_ref()) {
                            (OutKind::Builtin(BuiltIn::ClipDistance), Some(clip_distance_ty)) => {
                                clip_distance_ty.array_ty()
                            }
                            (OutKind::Builtin(builtin), _) => {
                                vertex_builtin_output_type(ctx, builtin, *mty)
                            }
                            (OutKind::Location(_), _) => *mty,
                        };
                        let var = make_output_var(ctx, output_ty);
                        match kind {
                            OutKind::Builtin(builtin) => {
                                decorate_builtin(&mut ctx.module, var, builtin)
                            }
                            OutKind::Location(loc) => decorate_location(&mut ctx.module, var, loc),
                        }
                        ctx.interface.push(var);
                        match clip_distance_ty {
                            Some(ClipDistanceOutputType::Scalar { elem_ty, .. }) => {
                                outputs.push(OutputWrite::ClipDistanceExtract {
                                    var,
                                    member: mi as u32,
                                    src_ty: *mty,
                                    elem_ty,
                                });
                            }
                            Some(ClipDistanceOutputType::Array { .. }) | None => {
                                outputs.push(OutputWrite::Extract {
                                    var,
                                    member: mi as u32,
                                    src_ty: *mty,
                                    dst_ty: output_ty,
                                });
                            }
                        }
                    }
                } else {
                    // bare position vec4
                    let var = make_output_var(ctx, ret_ty);
                    decorate_builtin(&mut ctx.module, var, BuiltIn::Position);
                    ctx.interface.push(var);
                    outputs.push(OutputWrite::Direct {
                        var,
                        src_ty: ret_ty,
                        dst_ty: ret_ty,
                    });
                }
            }
        }
        // A compute kernel returns void (it writes through its buffer pointers); there is no
        // OpReturnValue, so we never reach here for Stage::Kernel.
        Stage::Kernel => {}
    }

    // Replace every OpReturnValue with stores + OpReturn. Process in reverse block/index order so
    // earlier replacements do not invalidate later recorded instruction indices.
    ret_locs.sort_by_key(|(bi, ii, _)| (*bi, *ii));
    for (bi, ii, retval) in ret_locs.into_iter().rev() {
        let mut replacement = Vec::new();
        for output in &outputs {
            replacement.extend(output.stores(ctx, retval));
        }
        replacement.push(Instruction::new(Op::Return, None, None, vec![]));

        let blk = &mut ctx.module.functions[entry_idx].blocks[bi];
        blk.instructions.splice(ii..=ii, replacement);
    }
    Ok(())
}

/// Find the module-scope `__air_sampler_state` global (if any), replace it with a real sampler
/// resource variable in the first free sampler-band slot, drop the old global, and rewrite
/// uses in the entry body into an `OpLoad %sampler %var`.
pub(in crate::passes) fn handle_static_sampler(ctx: &mut Ctx) -> Result<(), String> {
    // A shader can embed SEVERAL static samplers (`__air_sampler_state`, `__air_sampler_state.31`,
    // ...) — one per distinct sampler-state the source used. Collect every such global by name prefix.
    let mut samp_globals: Vec<(Word, String)> = vec![];
    for inst in &ctx.module.debug_names {
        if inst.class.opcode == Op::Name {
            if let (Some(Operand::IdRef(id)), Some(Operand::LiteralString(s))) =
                (inst.operands.first(), inst.operands.get(1))
            {
                if s.trim_start_matches('@').starts_with("__air_sampler_state") {
                    samp_globals.push((*id, s.trim_start_matches('@').to_string()));
                }
            }
        }
    }
    if samp_globals.is_empty() {
        return Ok(());
    }
    samp_globals.sort_by_key(|(_, name)| static_sampler_name_order(name));

    let sty = ctx.ty_sampler();
    let pptr = ctx.ty_ptr(StorageClass::UniformConstant, sty);

    for (old_var, _) in samp_globals {
        let sampler_state = static_sampler_words(ctx, old_var)
            .map(StaticSamplerState::from_air_words)
            .transpose()?;
        // One sampler resource per static sampler.
        let new_var = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::Variable,
            Some(pptr),
            Some(new_var),
            vec![Operand::StorageClass(StorageClass::UniformConstant)],
        ));
        let binding = allocate_static_sampler_binding(&ctx.module).ok_or_else(|| {
            format!(
                "AIR constexpr sampler count exceeds descriptor band \
                 [{SAMPLER_BINDING_BASE},{COLOR_INPUT_BINDING_BASE})"
            )
        })?;
        decorate_binding(&mut ctx.module, new_var, binding);
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
    Ok(())
}

fn static_sampler_name_order(name: &str) -> u64 {
    name.strip_prefix("__air_sampler_state")
        .and_then(|suffix| suffix.strip_prefix('.'))
        .and_then(|suffix| suffix.parse::<u64>().ok())
        .unwrap_or(0)
}

fn static_sampler_words(ctx: &Ctx, var: Word) -> Option<[u64; 2]> {
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
    static_sampler_words_from_const(ctx, initializer)
}

fn static_sampler_words_from_const(ctx: &Ctx, value: Word) -> Option<[u64; 2]> {
    let inst = ctx
        .module
        .types_global_values
        .iter()
        .find(|inst| inst.result_id == Some(value))?;
    match inst.class.opcode {
        Op::Constant => match inst.operands.first()? {
            Operand::LiteralBit64(value) => Some([*value, 0]),
            Operand::LiteralBit32(value) => Some([*value as u64, 0]),
            _ => None,
        },
        Op::ConstantComposite => {
            let mut words = inst.operands.iter().filter_map(|operand| match operand {
                Operand::IdRef(id) => {
                    static_sampler_words_from_const(ctx, *id).map(|words| words[0])
                }
                _ => None,
            });
            Some([words.next()?, words.next().unwrap_or(0)])
        }
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
