//! Value-domain lowering for selected opaque image resources.
//!
//! Vulkan does not permit an `OpSelect` whose operands are images or sampled images under the
//! ordinary Logical capability set. AIR can nevertheless form a dynamic local texture table, and
//! the native emitter can preserve the table's stale pointer result type while the select operands
//! are loaded images. Re-typing that select to the image type only replaces one validation error
//! with the descriptor-indexing requirement; the portable form is to move the selection past the
//! pure image read and select the ordinary sampled VALUE instead.
//!
//! This pass deliberately accepts a narrow, structurally proven closure: an image-only select tree
//! whose every external consumer is `OpSampledImage`, whose sampled-image result is consumed only by
//! `OpImageSampleExplicitLod`, and whose values are available at the replay point. It clones the
//! pure sampled-image/read pair for every leaf and rebuilds the select tree over the non-opaque
//! result. Any query, write, sparse operation, non-dominating value, or opaque consumer declines the
//! rewrite. Values may come from the function entry block (which dominates every reachable block) or
//! earlier in the consuming block. The selected image is never selected in emitted SPIR-V, so no
//! bindless extension is required.

use crate::spirv_module::Instruction;
use crate::spirv_module::Module;
use crate::spirv_module::Operand;
use spirv::{Op, Word};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Site {
    function: usize,
    block: usize,
    instruction: usize,
}

#[derive(Clone, Copy, Debug)]
struct UseSite {
    site: Site,
    operand: usize,
}

struct TreePlan {
    select_ids: BTreeSet<Word>,
    sampled_image_ids: HashSet<Word>,
    replacements: HashMap<Site, Vec<Instruction>>,
}

/// Replace a selected opaque image consumed only by explicit-LOD sampling with a select over the
/// sampled value. Returns `true` only after the entire use closure has passed the strict gate.
pub(super) fn rewrite_opaque_image_selects(module: &mut Module) -> bool {
    let type_defs: HashMap<Word, Instruction> = module
        .types_global_values
        .iter()
        .filter_map(|inst| inst.result_id.map(|id| (id, inst.clone())))
        .collect();
    let mut value_defs = type_defs.clone();
    let mut value_types: HashMap<Word, Word> = module
        .types_global_values
        .iter()
        .filter_map(|inst| match (inst.result_id, inst.result_type) {
            (Some(id), Some(ty)) => Some((id, ty)),
            _ => None,
        })
        .collect();
    let global_ids: HashSet<Word> = module
        .types_global_values
        .iter()
        .filter_map(|inst| inst.result_id)
        .collect();
    let mut positions: HashMap<Word, Site> = HashMap::new();
    let mut parameter_function: HashMap<Word, usize> = HashMap::new();
    let mut uses: HashMap<Word, Vec<UseSite>> = HashMap::new();

    for (function, func) in module.functions.iter().enumerate() {
        for parameter in &func.parameters {
            if let Some(id) = parameter.result_id {
                parameter_function.insert(id, function);
                if let Some(ty) = parameter.result_type {
                    value_types.insert(id, ty);
                }
            }
        }
        for (block, blk) in func.blocks.iter().enumerate() {
            for (instruction, inst) in blk.instructions.iter().enumerate() {
                let site = Site {
                    function,
                    block,
                    instruction,
                };
                if let Some(id) = inst.result_id {
                    value_defs.insert(id, inst.clone());
                    positions.insert(id, site);
                    if let Some(ty) = inst.result_type {
                        value_types.insert(id, ty);
                    }
                }
                for (operand, op) in inst.operands.iter().enumerate() {
                    if let Operand::IdRef(id) = op {
                        uses.entry(*id).or_default().push(UseSite { site, operand });
                    }
                }
            }
        }
    }

    let mut candidates = Vec::new();
    for (function, func) in module.functions.iter().enumerate() {
        for (block, blk) in func.blocks.iter().enumerate() {
            for (instruction, inst) in blk.instructions.iter().enumerate() {
                if inst.class.opcode != Op::SampledImage {
                    continue;
                }
                let Some(Operand::IdRef(image)) = inst.operands.first() else {
                    continue;
                };
                if value_defs
                    .get(image)
                    .is_some_and(|def| def.class.opcode == Op::Select)
                {
                    candidates.push((
                        Site {
                            function,
                            block,
                            instruction,
                        },
                        *image,
                    ));
                }
            }
        }
    }

    let mut next_id = module
        .header
        .as_ref()
        .map(|header| header.bound)
        .unwrap_or(0);
    let mut processed_selects = HashSet::new();
    let mut select_ids_to_delete = HashSet::new();
    let mut sampled_image_ids_to_delete = HashSet::new();
    let mut replacements: HashMap<Site, Vec<Instruction>> = HashMap::new();

    for (_, root) in candidates {
        if processed_selects.contains(&root) {
            continue;
        }
        let Some(plan) = plan_tree(
            module,
            root,
            &type_defs,
            &value_defs,
            &value_types,
            &positions,
            &parameter_function,
            &global_ids,
            &uses,
            &mut next_id,
        ) else {
            continue;
        };
        if plan
            .select_ids
            .iter()
            .any(|id| processed_selects.contains(id))
        {
            continue;
        }
        processed_selects.extend(plan.select_ids.iter().copied());
        select_ids_to_delete.extend(plan.select_ids);
        sampled_image_ids_to_delete.extend(plan.sampled_image_ids);
        replacements.extend(plan.replacements);
    }

    if replacements.is_empty() {
        return false;
    }

    for (function, func) in module.functions.iter_mut().enumerate() {
        for (block, blk) in func.blocks.iter_mut().enumerate() {
            let old = std::mem::take(&mut blk.instructions);
            let mut rebuilt = Vec::with_capacity(old.len());
            for (instruction, inst) in old.into_iter().enumerate() {
                let site = Site {
                    function,
                    block,
                    instruction,
                };
                if let Some(replacement) = replacements.remove(&site) {
                    rebuilt.extend(replacement);
                    continue;
                }
                if inst.result_id.is_some_and(|id| {
                    select_ids_to_delete.contains(&id) || sampled_image_ids_to_delete.contains(&id)
                }) {
                    continue;
                }
                rebuilt.push(inst);
            }
            blk.instructions = rebuilt;
        }
    }

    let removed: HashSet<Word> = select_ids_to_delete
        .into_iter()
        .chain(sampled_image_ids_to_delete)
        .collect();
    let targets_removed = |inst: &Instruction| matches!(inst.operands.first(), Some(Operand::IdRef(id)) if removed.contains(id));
    module.debug_names.retain(|inst| !targets_removed(inst));
    module.annotations.retain(|inst| !targets_removed(inst));
    if let Some(header) = module.header.as_mut() {
        header.bound = next_id;
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn plan_tree(
    module: &Module,
    root: Word,
    type_defs: &HashMap<Word, Instruction>,
    value_defs: &HashMap<Word, Instruction>,
    value_types: &HashMap<Word, Word>,
    positions: &HashMap<Word, Site>,
    parameter_function: &HashMap<Word, usize>,
    global_ids: &HashSet<Word>,
    uses: &HashMap<Word, Vec<UseSite>>,
    next_id: &mut Word,
) -> Option<TreePlan> {
    let root_sampled_image = uses.get(&root)?.iter().find_map(|use_site| {
        let inst = instruction_at(module, use_site.site)?;
        (inst.class.opcode == Op::SampledImage && use_site.operand == 0).then_some(inst)
    })?;
    let sampled_image_ty = root_sampled_image.result_type?;
    let image_ty = sampled_image_image_type(type_defs, sampled_image_ty)?;
    if !is_image_type(type_defs, image_ty) {
        return None;
    }

    let mut select_ids = BTreeSet::new();
    let mut visiting = HashSet::new();
    collect_select_tree(
        root,
        image_ty,
        type_defs,
        value_defs,
        value_types,
        &mut select_ids,
        &mut visiting,
    )?;

    // Both loops below allocate fresh ids. Keep their traversal tied to the module's canonical id
    // order rather than RandomState iteration so identical modules produce identical bytes.
    let mut sampled_images: BTreeMap<Word, (Word, Word, Word)> = BTreeMap::new();
    for &select in &select_ids {
        for use_site in uses.get(&select).into_iter().flatten() {
            let inst = instruction_at(module, use_site.site)?;
            let is_tree_parent = inst.class.opcode == Op::Select
                && inst.result_id.is_some_and(|id| select_ids.contains(&id))
                && use_site.operand >= 1;
            if is_tree_parent {
                continue;
            }
            if inst.class.opcode != Op::SampledImage || use_site.operand != 0 {
                return None;
            }
            let sampled_id = inst.result_id?;
            let sampled_ty = inst.result_type?;
            if sampled_image_image_type(type_defs, sampled_ty)? != image_ty {
                return None;
            }
            let sampler = match inst.operands.get(1)? {
                Operand::IdRef(id) => *id,
                _ => return None,
            };
            sampled_images
                .entry(sampled_id)
                .or_insert((sampled_ty, sampler, select));
        }
    }
    if sampled_images.is_empty() {
        return None;
    }

    let mut replacements = HashMap::new();
    for (&sampled_id, &(sampled_ty, sampler, selected_image)) in &sampled_images {
        let sample_uses = uses.get(&sampled_id)?;
        if sample_uses.is_empty() {
            return None;
        }
        for use_site in sample_uses {
            let sample = instruction_at(module, use_site.site)?;
            if sample.class.opcode != Op::ImageSampleExplicitLod || use_site.operand != 0 {
                return None;
            }
            let sample_result_ty = sample.result_type?;
            let sample_result = sample.result_id?;
            if is_opaque_type(type_defs, sample_result_ty)
                || !tree_values_available_at(
                    &select_ids,
                    use_site.site,
                    value_defs,
                    positions,
                    parameter_function,
                    global_ids,
                )
                || !value_available_at(
                    sampler,
                    use_site.site,
                    positions,
                    parameter_function,
                    global_ids,
                )
            {
                return None;
            }
            let mut replacement = Vec::new();
            let mut replay_visiting = HashSet::new();
            replay_selected_image(
                selected_image,
                image_ty,
                sampled_ty,
                sampler,
                sample,
                Some(sample_result),
                type_defs,
                value_defs,
                value_types,
                &mut replay_visiting,
                next_id,
                &mut replacement,
            )?;
            if replacements.insert(use_site.site, replacement).is_some() {
                return None;
            }
        }
    }

    Some(TreePlan {
        select_ids,
        sampled_image_ids: sampled_images.into_keys().collect(),
        replacements,
    })
}

#[allow(clippy::too_many_arguments)]
fn collect_select_tree(
    value: Word,
    image_ty: Word,
    type_defs: &HashMap<Word, Instruction>,
    value_defs: &HashMap<Word, Instruction>,
    value_types: &HashMap<Word, Word>,
    select_ids: &mut BTreeSet<Word>,
    visiting: &mut HashSet<Word>,
) -> Option<()> {
    let def = value_defs.get(&value)?;
    if def.class.opcode != Op::Select {
        return (value_types.get(&value) == Some(&image_ty) && is_image_type(type_defs, image_ty))
            .then_some(());
    }
    // This is specifically the stale-pointer image-table form. A typed image select belongs to a
    // descriptor-indexing-capable module and must not be rewritten merely because an unrelated
    // validation error also classified as `Other`.
    if !def
        .result_type
        .is_some_and(|ty| is_pointer_type(type_defs, ty))
    {
        return None;
    }
    if !visiting.insert(value) {
        return None;
    }
    let true_value = match def.operands.get(1)? {
        Operand::IdRef(id) => *id,
        _ => return None,
    };
    let false_value = match def.operands.get(2)? {
        Operand::IdRef(id) => *id,
        _ => return None,
    };
    select_ids.insert(value);
    collect_select_tree(
        true_value,
        image_ty,
        type_defs,
        value_defs,
        value_types,
        select_ids,
        visiting,
    )?;
    collect_select_tree(
        false_value,
        image_ty,
        type_defs,
        value_defs,
        value_types,
        select_ids,
        visiting,
    )?;
    visiting.remove(&value);
    Some(())
}

#[allow(clippy::too_many_arguments)]
fn replay_selected_image(
    value: Word,
    image_ty: Word,
    sampled_image_ty: Word,
    sampler: Word,
    sample: &Instruction,
    final_result: Option<Word>,
    type_defs: &HashMap<Word, Instruction>,
    value_defs: &HashMap<Word, Instruction>,
    value_types: &HashMap<Word, Word>,
    visiting: &mut HashSet<Word>,
    next_id: &mut Word,
    out: &mut Vec<Instruction>,
) -> Option<Word> {
    let def = value_defs.get(&value)?;
    if def.class.opcode != Op::Select {
        if value_types.get(&value) != Some(&image_ty) || !is_image_type(type_defs, image_ty) {
            return None;
        }
        let sampled = fresh(next_id);
        out.push(Instruction::new(
            Op::SampledImage,
            Some(sampled_image_ty),
            Some(sampled),
            vec![Operand::IdRef(value), Operand::IdRef(sampler)],
        ));
        let result = fresh(next_id);
        let mut cloned_sample = sample.clone();
        cloned_sample.result_id = Some(result);
        *cloned_sample.operands.first_mut()? = Operand::IdRef(sampled);
        out.push(cloned_sample);
        return Some(result);
    }
    if !visiting.insert(value) {
        return None;
    }
    let condition = match def.operands.first()? {
        Operand::IdRef(id) => *id,
        _ => return None,
    };
    let true_image = match def.operands.get(1)? {
        Operand::IdRef(id) => *id,
        _ => return None,
    };
    let false_image = match def.operands.get(2)? {
        Operand::IdRef(id) => *id,
        _ => return None,
    };
    let true_value = replay_selected_image(
        true_image,
        image_ty,
        sampled_image_ty,
        sampler,
        sample,
        None,
        type_defs,
        value_defs,
        value_types,
        visiting,
        next_id,
        out,
    )?;
    let false_value = replay_selected_image(
        false_image,
        image_ty,
        sampled_image_ty,
        sampler,
        sample,
        None,
        type_defs,
        value_defs,
        value_types,
        visiting,
        next_id,
        out,
    )?;
    visiting.remove(&value);
    let result = final_result.unwrap_or_else(|| fresh(next_id));
    out.push(Instruction::new(
        Op::Select,
        sample.result_type,
        Some(result),
        vec![
            Operand::IdRef(condition),
            Operand::IdRef(true_value),
            Operand::IdRef(false_value),
        ],
    ));
    Some(result)
}

fn instruction_at(module: &Module, site: Site) -> Option<&Instruction> {
    module
        .functions
        .get(site.function)?
        .blocks
        .get(site.block)?
        .instructions
        .get(site.instruction)
}

fn sampled_image_image_type(
    type_defs: &HashMap<Word, Instruction>,
    sampled_image_ty: Word,
) -> Option<Word> {
    let def = type_defs.get(&sampled_image_ty)?;
    if def.class.opcode != Op::TypeSampledImage {
        return None;
    }
    match def.operands.first()? {
        Operand::IdRef(image_ty) => Some(*image_ty),
        _ => None,
    }
}

fn is_image_type(type_defs: &HashMap<Word, Instruction>, ty: Word) -> bool {
    type_defs
        .get(&ty)
        .is_some_and(|def| def.class.opcode == Op::TypeImage)
}

fn is_pointer_type(type_defs: &HashMap<Word, Instruction>, ty: Word) -> bool {
    type_defs
        .get(&ty)
        .is_some_and(|def| def.class.opcode == Op::TypePointer)
}

fn is_opaque_type(type_defs: &HashMap<Word, Instruction>, ty: Word) -> bool {
    type_defs.get(&ty).is_some_and(|def| {
        matches!(
            def.class.opcode,
            Op::TypeImage | Op::TypeSampler | Op::TypeSampledImage
        )
    })
}

fn tree_values_available_at(
    select_ids: &BTreeSet<Word>,
    site: Site,
    value_defs: &HashMap<Word, Instruction>,
    positions: &HashMap<Word, Site>,
    parameter_function: &HashMap<Word, usize>,
    global_ids: &HashSet<Word>,
) -> bool {
    select_ids.iter().all(|id| {
        let Some(def) = value_defs.get(id) else {
            return false;
        };
        def.operands.iter().all(|operand| match operand {
            Operand::IdRef(value) if !select_ids.contains(value) => {
                value_available_at(*value, site, positions, parameter_function, global_ids)
            }
            _ => true,
        })
    })
}

fn value_available_at(
    value: Word,
    site: Site,
    positions: &HashMap<Word, Site>,
    parameter_function: &HashMap<Word, usize>,
    global_ids: &HashSet<Word>,
) -> bool {
    global_ids.contains(&value)
        || parameter_function.get(&value) == Some(&site.function)
        || positions.get(&value).is_some_and(|position| {
            position.function == site.function
                // Every SSA definition in the entry block dominates a later reachable block. This
                // admits the common local texture-table shape while deliberately declining values
                // from arbitrary predecessor/sibling blocks, which would need full CFG proof.
                && (position.block == 0
                    || (position.block == site.block && position.instruction < site.instruction))
        })
}

fn fresh(next_id: &mut Word) -> Word {
    let id = *next_id;
    *next_id += 1;
    id
}
