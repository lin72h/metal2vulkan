//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

pub(in crate::passes) fn neutralize_null_access_chains(ctx: &mut Ctx, entry_idx: usize) {
    let mut null_ptrs: HashSet<Word> = ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
        .filter(|inst| inst.class.opcode == Op::ConstantNull)
        .filter_map(|inst| {
            let result = inst.result_id?;
            let result_type = inst.result_type?;
            pointer_pointee(ctx, result_type).map(|_| result)
        })
        .collect();

    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let insts = ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .clone();
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts {
            if matches!(
                inst.class.opcode,
                Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
            ) && inst
                .operands
                .first()
                .and_then(|operand| match operand {
                    Operand::IdRef(base) => Some(null_ptrs.contains(base)),
                    _ => None,
                })
                .unwrap_or(false)
            {
                if let (Some(result_type), Some(result)) = (inst.result_type, inst.result_id) {
                    null_ptrs.insert(result);
                    push_null_copy(ctx, result_type, result, &mut out);
                }
                continue;
            }

            if inst.class.opcode == Op::Load
                && inst
                    .operands
                    .first()
                    .and_then(|operand| match operand {
                        Operand::IdRef(ptr) => Some(null_ptrs.contains(ptr)),
                        _ => None,
                    })
                    .unwrap_or(false)
            {
                if let (Some(result_type), Some(result)) = (inst.result_type, inst.result_id) {
                    if pointer_pointee(ctx, result_type).is_some() {
                        null_ptrs.insert(result);
                    }
                    push_null_copy(ctx, result_type, result, &mut out);
                }
                continue;
            }

            if inst.class.opcode == Op::Store
                && inst
                    .operands
                    .first()
                    .and_then(|operand| match operand {
                        Operand::IdRef(ptr) => Some(null_ptrs.contains(ptr)),
                        _ => None,
                    })
                    .unwrap_or(false)
            {
                continue;
            }

            out.push(inst);
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = out;
    }
}

pub(in crate::passes) fn neutralize_private_placeholder_access_chains(
    ctx: &mut Ctx,
    entry_idx: usize,
) -> Result<(), String> {
    let named_ids: HashSet<Word> = ctx
        .module
        .debug_names
        .iter()
        .filter(|inst| inst.class.opcode == Op::Name)
        .filter_map(|inst| match inst.operands.first() {
            Some(Operand::IdRef(id)) => Some(*id),
            _ => None,
        })
        .collect();
    let mut roots: HashSet<Word> = ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
        .filter(|inst| inst.class.opcode == Op::Variable)
        .filter_map(|inst| {
            let result = inst.result_id?;
            if named_ids.contains(&result) {
                return None;
            }
            if !matches!(
                inst.operands.first(),
                Some(Operand::StorageClass(StorageClass::Private))
            ) {
                return None;
            }
            if !inst
                .operands
                .get(1)
                .and_then(|operand| match operand {
                    Operand::IdRef(init) => Some(is_constant_null_id(ctx, *init)),
                    _ => None,
                })
                .unwrap_or(false)
            {
                return None;
            }
            private_pointer_pointee(ctx, inst.result_type?).map(|_| result)
        })
        .collect();

    // Discover the complete placeholder-derived pointer component before rewriting it. In
    // particular, a loop phi is visited before its backedge access chain in serialized block order;
    // a one-pass walk therefore misses exactly the illegal `Private` pointer phi that joins two
    // otherwise-neutralized arms. Copies preserve placeholder identity, while a select/phi is a
    // placeholder only when every pointer arm is one. Use a dependency worklist so each edge is
    // visited once instead of repeatedly walking a very large module to reach the same fixed point.
    let mut dependencies = Vec::<(Word, Vec<Word>)>::new();
    for inst in ctx.module.functions[entry_idx]
        .blocks
        .iter()
        .flat_map(|block| block.instructions.iter())
    {
        let (Some(result_type), Some(result)) = (inst.result_type, inst.result_id) else {
            continue;
        };
        if private_pointer_pointee(ctx, result_type).is_none() {
            continue;
        }
        let deps = match inst.class.opcode {
            Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain | Op::CopyObject => {
                inst.operands.first().and_then(|operand| match operand {
                    Operand::IdRef(base) => Some(vec![*base]),
                    _ => None,
                })
            }
            Op::Select => match inst.operands.as_slice() {
                [_, Operand::IdRef(on_true), Operand::IdRef(on_false), ..] => {
                    Some(vec![*on_true, *on_false])
                }
                _ => None,
            },
            Op::Phi if !inst.operands.is_empty() && inst.operands.len().is_multiple_of(2) => inst
                .operands
                .chunks_exact(2)
                .map(|pair| match pair.first() {
                    Some(Operand::IdRef(value)) => Some(*value),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>(),
            _ => None,
        };
        if let Some(mut deps) = deps {
            deps.sort_unstable();
            deps.dedup();
            dependencies.push((result, deps));
        }
    }
    let mut waiting = HashMap::<Word, Vec<usize>>::new();
    let mut remaining = Vec::with_capacity(dependencies.len());
    let mut ready = Vec::new();
    for (candidate, (result, deps)) in dependencies.iter().enumerate() {
        let mut missing = 0usize;
        for dependency in deps {
            if !roots.contains(dependency) {
                missing += 1;
                waiting.entry(*dependency).or_default().push(candidate);
            }
        }
        remaining.push(missing);
        if missing == 0 {
            ready.push(*result);
        }
    }
    while let Some(root) = ready.pop() {
        if !roots.insert(root) {
            continue;
        }
        if let Some(dependents) = waiting.remove(&root) {
            for candidate in dependents {
                remaining[candidate] -= 1;
                if remaining[candidate] == 0 {
                    ready.push(dependencies[candidate].0);
                }
            }
        }
    }

    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let insts = ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .clone();
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts {
            let neutralize = match inst.class.opcode {
                Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain => {
                    matches!(inst.operands.first(), Some(Operand::IdRef(base)) if roots.contains(base))
                }
                Op::Select => {
                    matches!(
                        inst.operands.as_slice(),
                        [_, Operand::IdRef(on_true), Operand::IdRef(on_false), ..]
                            if roots.contains(on_true) && roots.contains(on_false)
                    )
                }
                Op::Phi => {
                    !inst.operands.is_empty()
                        && inst.operands.len().is_multiple_of(2)
                        && inst.operands.chunks_exact(2).all(|pair| {
                            matches!(pair.first(), Some(Operand::IdRef(value)) if roots.contains(value))
                        })
                }
                _ => false,
            };
            if neutralize {
                if let (Some(result_type), Some(result)) = (inst.result_type, inst.result_id) {
                    if private_pointer_pointee(ctx, result_type).is_some() {
                        let placeholder = private_zero_pointer_for_type(ctx, result_type)?;
                        out.push(Instruction::new(
                            Op::CopyObject,
                            Some(result_type),
                            Some(result),
                            vec![Operand::IdRef(placeholder)],
                        ));
                        continue;
                    }
                }
            }
            out.push(inst);
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = out;
    }
    Ok(())
}

pub(in crate::passes) fn lower_private_memory_atomics(ctx: &mut Ctx, entry_idx: usize) {
    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let insts = ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .clone();
        let mut out = Vec::with_capacity(insts.len());
        for inst in insts {
            let Some(bitwise_op) = private_atomic_bitwise_op(inst.class.opcode) else {
                out.push(inst);
                continue;
            };
            let (Some(result_type), Some(result)) = (inst.result_type, inst.result_id) else {
                out.push(inst);
                continue;
            };
            let [Operand::IdRef(ptr), _, _, Operand::IdRef(value), ..] = inst.operands.as_slice()
            else {
                out.push(inst);
                continue;
            };
            if !private_pointer_matches_pointee(ctx, *ptr, result_type) {
                out.push(inst);
                continue;
            }

            let updated = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::Load,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(*ptr)],
            ));
            out.push(Instruction::new(
                bitwise_op,
                Some(result_type),
                Some(updated),
                vec![Operand::IdRef(result), Operand::IdRef(*value)],
            ));
            out.push(Instruction::new(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(*ptr), Operand::IdRef(updated)],
            ));
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = out;
    }
}

pub(in crate::passes) fn private_atomic_bitwise_op(op: Op) -> Option<Op> {
    match op {
        Op::AtomicAnd => Some(Op::BitwiseAnd),
        Op::AtomicOr => Some(Op::BitwiseOr),
        _ => None,
    }
}

pub(in crate::passes) fn private_pointer_matches_pointee(
    ctx: &Ctx,
    ptr: Word,
    pointee: Word,
) -> bool {
    let Some(ptr_ty) = value_result_type(ctx, ptr) else {
        return false;
    };
    private_pointer_pointee(ctx, ptr_ty) == Some(pointee)
}

pub(in crate::passes) fn private_zero_pointer_for_type(
    ctx: &mut Ctx,
    ptr_ty: Word,
) -> Result<Word, String> {
    let pointee = private_pointer_pointee(ctx, ptr_ty).ok_or("private pointer type")?;
    let init = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::ConstantNull,
        Some(pointee),
        Some(init),
        vec![],
    ));
    let var = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::Variable,
        Some(ptr_ty),
        Some(var),
        vec![
            Operand::StorageClass(StorageClass::Private),
            Operand::IdRef(init),
        ],
    ));
    ctx.interface.push(var);
    Ok(var)
}

pub(in crate::passes) fn private_pointer_pointee(ctx: &Ctx, ptr_ty: Word) -> Option<Word> {
    let inst = type_def_of(ctx, ptr_ty)?;
    if inst.class.opcode != Op::TypePointer {
        return None;
    }
    if !matches!(
        inst.operands.first(),
        Some(Operand::StorageClass(StorageClass::Private))
    ) {
        return None;
    }
    match inst.operands.get(1) {
        Some(Operand::IdRef(pointee)) => Some(*pointee),
        _ => None,
    }
}

pub(in crate::passes) fn is_constant_null_id(ctx: &Ctx, id: Word) -> bool {
    ctx.new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .any(|inst| inst.result_id == Some(id) && inst.class.opcode == Op::ConstantNull)
}

pub(in crate::passes) fn push_null_copy(
    ctx: &mut Ctx,
    result_type: Word,
    result: Word,
    out: &mut Vec<Instruction>,
) {
    if pointer_pointee(ctx, result_type).is_some() {
        out.push(Instruction::new(
            Op::Undef,
            Some(result_type),
            Some(result),
            vec![],
        ));
        return;
    }
    let zero = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::ConstantNull,
        Some(result_type),
        Some(zero),
        vec![],
    ));
    out.push(Instruction::new(
        Op::CopyObject,
        Some(result_type),
        Some(result),
        vec![Operand::IdRef(zero)],
    ));
}

pub(in crate::passes) fn hoist_function_variables(ctx: &mut Ctx, entry_idx: usize) {
    let mut vars = Vec::new();
    for block in &mut ctx.module.functions[entry_idx].blocks {
        let mut kept = Vec::with_capacity(block.instructions.len());
        for inst in block.instructions.drain(..) {
            if is_function_variable(&inst) {
                vars.push(inst);
            } else {
                kept.push(inst);
            }
        }
        block.instructions = kept;
    }
    if let Some(first) = ctx.module.functions[entry_idx].blocks.first_mut() {
        vars.append(&mut first.instructions);
        first.instructions = vars;
    }
}

pub(in crate::passes) fn is_function_variable(inst: &Instruction) -> bool {
    inst.class.opcode == Op::Variable
        && matches!(
            inst.operands.first(),
            Some(Operand::StorageClass(StorageClass::Function))
        )
}

/// Lower every OpFunctionCall to a residual AIR/LLVM helper inside the entry function into native ops /
/// GLSL.std.450 ext-insts / image samples / derivatives. Unknown residual calls are a hard error
/// (caller falls back).
/// Replace any width-preserving `OpUConvert`/`OpSConvert`/`OpFConvert` in the entry with a legal
/// identity operation — these are illegal in SPIR-V (the converts REQUIRE differing bit widths). An
/// exactly matching type uses `OpCopyObject`; equal-width/equal-lane signedness changes use
/// `OpBitcast`, because `OpCopyObject` requires identical types. They arise when our
/// interface pass binds a narrower AIR param (`ushort` `[[vertex_id]]`, an `i16` index) to the 32-bit
/// Vulkan builtin: the body's original `zext i16 -> i32` then compiles to a same-width `OpUConvert
/// %uint %uint`. A no-op copy is the semantically-correct result. General (a true convert keeps its op).
pub(in crate::passes) fn fix_noop_width_converts(ctx: &mut Ctx, entry_idx: usize) {
    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let n_insts = ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .len();
        for ii in 0..n_insts {
            let (rty, src) = {
                let inst = &ctx.module.functions[entry_idx].blocks[bi].instructions[ii];
                if !matches!(
                    inst.class.opcode,
                    Op::UConvert | Op::SConvert | Op::FConvert
                ) {
                    continue;
                }
                let Some(rty) = inst.result_type else {
                    continue;
                };
                let Some(Operand::IdRef(src)) = inst.operands.first() else {
                    continue;
                };
                (rty, *src)
            };
            let Some(src_ty) = value_result_type(ctx, src) else {
                continue;
            };
            let replacement = if src_ty == rty {
                Some(Op::CopyObject)
            } else if scalar_bit_width(ctx, src_ty) == scalar_bit_width(ctx, rty)
                && type_lane_count(ctx, src_ty) == type_lane_count(ctx, rty)
            {
                Some(Op::Bitcast)
            } else {
                None
            };
            if let Some(replacement) = replacement {
                let inst = &mut ctx.module.functions[entry_idx].blocks[bi].instructions[ii];
                let res = inst.result_id;
                *inst = Instruction::new(replacement, Some(rty), res, vec![Operand::IdRef(src)]);
            }
        }
    }
}

fn type_lane_count(ctx: &Ctx, ty: Word) -> u32 {
    type_def_of(ctx, ty)
        .filter(|definition| definition.class.opcode == Op::TypeVector)
        .and_then(|definition| match definition.operands.get(1) {
            Some(Operand::LiteralBit32(lanes)) => Some(*lanes),
            _ => None,
        })
        .unwrap_or(1)
}

/// Integer division/remainder by zero is undefined in SPIR-V. Some AIR comes from guarded source
/// like `count == 0 ? sum : sum / count`, but the native IR can still contain an eager `OpUDiv`
/// feeding an `OpSelect`. Guard denominator zeros to one so the normal selected arm remains
/// unchanged while the otherwise-undefined arm becomes deterministic across Vulkan drivers.
pub(in crate::passes) fn guard_integer_division_by_zero(ctx: &mut Ctx, entry_idx: usize) {
    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let insts = ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .clone();
        let mut out = Vec::with_capacity(insts.len());
        for mut inst in insts {
            if !matches!(inst.class.opcode, Op::UDiv | Op::SDiv | Op::UMod | Op::SRem) {
                out.push(inst);
                continue;
            }
            let Some(result_ty) = inst.result_type else {
                out.push(inst);
                continue;
            };
            let Some(Operand::IdRef(denom)) = inst.operands.get(1) else {
                out.push(inst);
                continue;
            };
            let denom = *denom;
            let Some((zero, one, pred_ty)) = integer_division_guard_values(ctx, result_ty) else {
                out.push(inst);
                continue;
            };

            let is_zero = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::IEqual,
                Some(pred_ty),
                Some(is_zero),
                vec![Operand::IdRef(denom), Operand::IdRef(zero)],
            ));
            let safe_denom = ctx.module.fresh_id();
            out.push(Instruction::new(
                Op::Select,
                Some(result_ty),
                Some(safe_denom),
                vec![
                    Operand::IdRef(is_zero),
                    Operand::IdRef(one),
                    Operand::IdRef(denom),
                ],
            ));
            inst.operands[1] = Operand::IdRef(safe_denom);
            out.push(inst);
        }
        ctx.module.functions[entry_idx].blocks[bi].instructions = out;
    }
}

pub(in crate::passes) fn integer_division_guard_values(
    ctx: &mut Ctx,
    ty: Word,
) -> Option<(Word, Word, Word)> {
    let def = type_def_of(ctx, ty)?;
    match def.class.opcode {
        Op::TypeInt => {
            let zero = ctx.const_int_of(ty, 0);
            let one = ctx.const_int_of(ty, 1);
            let pred_ty = ctx.ty_bool();
            Some((zero, one, pred_ty))
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
            let elem_def = type_def_of(ctx, elem)?;
            if elem_def.class.opcode != Op::TypeInt {
                return None;
            }
            let zero_elem = ctx.const_int_of(elem, 0);
            let one_elem = ctx.const_int_of(elem, 1);
            let zero = const_composite_splat(ctx, ty, zero_elem, lanes);
            let one = const_composite_splat(ctx, ty, one_elem, lanes);
            let pred_ty = ctx.ty_vec_bool(lanes);
            Some((zero, one, pred_ty))
        }
        _ => None,
    }
}

pub(in crate::passes) fn const_composite_splat(
    ctx: &mut Ctx,
    ty: Word,
    value: Word,
    lanes: u32,
) -> Word {
    let key = SynthCacheKey::CompositeSplat { ty, value, lanes };
    if let Some(&id) = ctx.synth_cache.get(&key) {
        return id;
    }
    for inst in ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
    {
        if inst.class.opcode == Op::ConstantComposite
            && inst.result_type == Some(ty)
            && inst.operands.len() == lanes as usize
            && inst
                .operands
                .iter()
                .all(|operand| operand == &Operand::IdRef(value))
        {
            if let Some(id) = inst.result_id {
                ctx.synth_cache.insert(key, id);
                return id;
            }
        }
    }
    let id = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::ConstantComposite,
        Some(ty),
        Some(id),
        (0..lanes).map(|_| Operand::IdRef(value)).collect(),
    ));
    ctx.synth_cache.insert(key, id);
    id
}
