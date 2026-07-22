//! Push resource-handle selects through pure image operations to selectable result values.

use super::super::*;

#[derive(Clone, Copy)]
struct ResourceSelect {
    cond: Word,
    true_value: Word,
    false_value: Word,
}

/// Rewrite value-selects of RESOURCE handles (`%sel = OpSelect <UC-pointer-typed> %cond %imgA
/// %imgB`, the shape a dynamic `select(cond, texA, texB)` AIR argument leaves after param splicing)
/// by pushing the select DOWN through every use until the selected value has a selectable
/// (scalar/vector) type: each pure use (size query, OpSampledImage, sample, fetch) is duplicated
/// for the true/false resource and the RESULTS are selected. An `OpSelect` on an image or
/// sampled-image value is not valid SPIR-V, but sampling is pure, so
/// `sample(select(c, a, b), ...) == select(c, sample(a, ...), sample(b, ...))` holds exactly.
/// A seed select is rewritten only when its whole transitive use closure is duplicable; otherwise
/// it is left in place to fail validation VISIBLY rather than silently mis-emit.
pub(in crate::passes) fn rewrite_resource_query_selects(ctx: &mut Ctx) -> Result<(), String> {
    let value_types = value_result_types(&ctx.module);
    let pointer_types = pointer_type_storage_classes(&ctx.module);
    let opaque_types: HashSet<Word> = ctx
        .module
        .types_global_values
        .iter()
        .filter(|inst| {
            matches!(
                inst.class.opcode,
                Op::TypeImage | Op::TypeSampledImage | Op::TypeSampler
            )
        })
        .filter_map(|inst| inst.result_id)
        .collect();

    let mut seeds: Vec<(Word, ResourceSelect)> = vec![];
    for function in &ctx.module.functions {
        for block in &function.blocks {
            for inst in &block.instructions {
                if inst.class.opcode != Op::Select {
                    continue;
                }
                let (Some(result), Some(result_type)) = (inst.result_id, inst.result_type) else {
                    continue;
                };
                if pointer_types.get(&result_type) != Some(&StorageClass::UniformConstant) {
                    continue;
                }
                let (
                    Some(Operand::IdRef(cond)),
                    Some(Operand::IdRef(true_id)),
                    Some(Operand::IdRef(false_id)),
                ) = (
                    inst.operands.first(),
                    inst.operands.get(1),
                    inst.operands.get(2),
                )
                else {
                    continue;
                };
                let (Some(&true_type), Some(&false_type)) =
                    (value_types.get(true_id), value_types.get(false_id))
                else {
                    continue;
                };
                if true_type != false_type || pointer_types.contains_key(&true_type) {
                    continue;
                }
                seeds.push((
                    result,
                    ResourceSelect {
                        cond: *cond,
                        true_value: *true_id,
                        false_value: *false_id,
                    },
                ));
            }
        }
    }
    if seeds.is_empty() {
        return Ok(());
    }

    let mut candidates: HashMap<Word, ResourceSelect> = HashMap::new();
    'seed: for &(seed_id, sel) in &seeds {
        let mut closure: HashMap<Word, ResourceSelect> = HashMap::new();
        closure.insert(seed_id, sel);
        let mut work = vec![seed_id];
        while let Some(id) = work.pop() {
            let cond = closure[&id].cond;
            for function in &ctx.module.functions {
                for block in &function.blocks {
                    for inst in &block.instructions {
                        if inst.result_id == Some(id) {
                            continue;
                        }
                        if !inst.operands.iter().any(|o| o == &Operand::IdRef(id)) {
                            continue;
                        }
                        if !is_duplicable_resource_use(inst, id) {
                            continue 'seed;
                        }
                        let result = inst.result_id.ok_or(
                            "rewrite_resource_query_selects: duplicable use has no result",
                        )?;
                        let result_type = inst.result_type.ok_or(
                            "rewrite_resource_query_selects: duplicable use has no result type",
                        )?;
                        if opaque_types.contains(&result_type)
                            && closure
                                .insert(
                                    result,
                                    ResourceSelect {
                                        cond,
                                        true_value: 0,
                                        false_value: 0,
                                    },
                                )
                                .is_none()
                        {
                            work.push(result);
                        }
                    }
                }
            }
        }
        candidates.extend(closure);
    }
    if candidates.is_empty() {
        return Ok(());
    }

    for function_idx in 0..ctx.module.functions.len() {
        for block_idx in 0..ctx.module.functions[function_idx].blocks.len() {
            let old = std::mem::take(
                &mut ctx.module.functions[function_idx].blocks[block_idx].instructions,
            );
            let mut new_insts = Vec::with_capacity(old.len());
            for inst in old {
                let result = inst.result_id;
                if inst.class.opcode == Op::Select
                    && result.is_some_and(|id| candidates.contains_key(&id))
                {
                    continue;
                }
                let used = inst.operands.iter().find_map(|o| match o {
                    Operand::IdRef(r) if candidates.contains_key(r) => Some(*r),
                    _ => None,
                });
                let Some(used_id) = used else {
                    new_insts.push(inst);
                    continue;
                };
                let sel = candidates[&used_id];
                let result =
                    result.ok_or("rewrite_resource_query_selects: duplicable use has no result")?;
                let result_type = inst
                    .result_type
                    .ok_or("rewrite_resource_query_selects: duplicable use has no result type")?;
                let true_use = ctx.module.fresh_id();
                let false_use = ctx.module.fresh_id();
                let mut true_ops = inst.operands.clone();
                let mut false_ops = inst.operands.clone();
                for (t_op, f_op) in true_ops.iter_mut().zip(false_ops.iter_mut()) {
                    if *t_op == Operand::IdRef(used_id) {
                        *t_op = Operand::IdRef(sel.true_value);
                        *f_op = Operand::IdRef(sel.false_value);
                    }
                }
                new_insts.push(Instruction::new(
                    inst.class.opcode,
                    Some(result_type),
                    Some(true_use),
                    true_ops,
                ));
                new_insts.push(Instruction::new(
                    inst.class.opcode,
                    Some(result_type),
                    Some(false_use),
                    false_ops,
                ));
                if let Some(cascade) = candidates.get_mut(&result) {
                    cascade.true_value = true_use;
                    cascade.false_value = false_use;
                } else {
                    new_insts.push(Instruction::new(
                        Op::Select,
                        Some(result_type),
                        Some(result),
                        vec![
                            Operand::IdRef(sel.cond),
                            Operand::IdRef(true_use),
                            Operand::IdRef(false_use),
                        ],
                    ));
                }
            }
            ctx.module.functions[function_idx].blocks[block_idx].instructions = new_insts;
        }
    }
    Ok(())
}

fn is_duplicable_resource_use(inst: &Instruction, id: Word) -> bool {
    matches!(
        inst.class.opcode,
        Op::ImageQuerySize
            | Op::ImageQuerySizeLod
            | Op::ImageQueryLevels
            | Op::ImageQuerySamples
            | Op::SampledImage
            | Op::Image
            | Op::ImageFetch
            | Op::ImageGather
            | Op::ImageDrefGather
            | Op::ImageSampleImplicitLod
            | Op::ImageSampleExplicitLod
            | Op::ImageSampleDrefImplicitLod
            | Op::ImageSampleDrefExplicitLod
            | Op::ImageRead
    ) && inst.operands.first() == Some(&Operand::IdRef(id))
        && inst
            .operands
            .iter()
            .skip(1)
            .all(|o| o != &Operand::IdRef(id))
        && inst.result_id.is_some()
        && inst.result_type.is_some()
}

fn value_result_types(module: &Module) -> HashMap<Word, Word> {
    let mut types = HashMap::new();
    for inst in &module.types_global_values {
        if let (Some(result), Some(result_type)) = (inst.result_id, inst.result_type) {
            types.insert(result, result_type);
        }
    }
    for function in &module.functions {
        if let Some(def) = &function.def {
            if let (Some(result), Some(result_type)) = (def.result_id, def.result_type) {
                types.insert(result, result_type);
            }
        }
        for param in &function.parameters {
            if let (Some(result), Some(result_type)) = (param.result_id, param.result_type) {
                types.insert(result, result_type);
            }
        }
        for block in &function.blocks {
            if let Some(label) = &block.label {
                if let (Some(result), Some(result_type)) = (label.result_id, label.result_type) {
                    types.insert(result, result_type);
                }
            }
            for inst in &block.instructions {
                if let (Some(result), Some(result_type)) = (inst.result_id, inst.result_type) {
                    types.insert(result, result_type);
                }
            }
        }
    }
    types
}

fn pointer_type_storage_classes(module: &Module) -> HashMap<Word, StorageClass> {
    module
        .types_global_values
        .iter()
        .filter_map(|inst| {
            if inst.class.opcode != Op::TypePointer {
                return None;
            }
            let id = inst.result_id?;
            let storage = match inst.operands.first()? {
                Operand::StorageClass(storage) => *storage,
                _ => return None,
            };
            Some((id, storage))
        })
        .collect()
}
