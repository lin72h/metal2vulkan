//! Close `OpSampledImage` result types over their concrete image operands.
//!
//! Interface binding can refine an opaque pointer parameter to a concrete 2D/3D image after the
//! sampled-image instruction was first typed. SPIR-V requires the `OpTypeSampledImage` image operand
//! to exactly match that concrete image type. Retype only mismatches, which are validator-invalid by
//! definition; sampler and downstream image-operation values remain unchanged.

use crate::spirv_module::{Instruction, Module, Operand};
use spirv::Op;
use std::collections::{HashMap, HashSet};

pub(super) fn repair_sampled_image_result_types(module: &mut Module) -> bool {
    let image_types = module
        .types_global_values
        .iter()
        .filter_map(|instruction| {
            (instruction.class.opcode == Op::TypeImage).then_some(instruction.result_id?)
        })
        .collect::<HashSet<_>>();
    let mut sampled_by_image = module
        .types_global_values
        .iter()
        .filter_map(|instruction| {
            if instruction.class.opcode != Op::TypeSampledImage {
                return None;
            }
            let Operand::IdRef(image) = instruction.operands.first()? else {
                return None;
            };
            Some((*image, instruction.result_id?))
        })
        .collect::<HashMap<_, _>>();
    let value_types = module
        .all_inst_iter()
        .filter_map(|instruction| Some((instruction.result_id?, instruction.result_type?)))
        .collect::<HashMap<_, _>>();

    let mut required = HashSet::new();
    for function in &module.functions {
        for block in &function.blocks {
            for instruction in &block.instructions {
                if instruction.class.opcode != Op::SampledImage {
                    continue;
                }
                let Some(Operand::IdRef(image)) = instruction.operands.first() else {
                    continue;
                };
                let Some(image_ty) = value_types.get(image).copied() else {
                    continue;
                };
                if image_types.contains(&image_ty) {
                    required.insert(image_ty);
                }
            }
        }
    }
    if required.is_empty() {
        return false;
    }

    let mut next_id = module
        .header
        .as_ref()
        .map(|header| header.bound)
        .unwrap_or(1);
    let mut additions = Vec::new();
    for image_ty in required {
        sampled_by_image.entry(image_ty).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            additions.push((
                image_ty,
                Instruction::new(
                    Op::TypeSampledImage,
                    None,
                    Some(id),
                    vec![Operand::IdRef(image_ty)],
                ),
            ));
            id
        });
    }
    if !additions.is_empty() {
        let mut additions = additions.into_iter().collect::<HashMap<_, _>>();
        let old = std::mem::take(&mut module.types_global_values);
        let mut rebuilt = Vec::with_capacity(old.len() + additions.len());
        for instruction in old {
            let result = instruction.result_id;
            rebuilt.push(instruction);
            if let Some(sampled) = result.and_then(|id| additions.remove(&id)) {
                rebuilt.push(sampled);
            }
        }
        module.types_global_values = rebuilt;
    }

    let mut changed = false;
    for function in &mut module.functions {
        for block in &mut function.blocks {
            for instruction in &mut block.instructions {
                if instruction.class.opcode != Op::SampledImage {
                    continue;
                }
                let Some(Operand::IdRef(image)) = instruction.operands.first() else {
                    continue;
                };
                let Some(image_ty) = value_types.get(image).copied() else {
                    continue;
                };
                let Some(sampled_ty) = sampled_by_image.get(&image_ty).copied() else {
                    continue;
                };
                if instruction.result_type != Some(sampled_ty) {
                    instruction.result_type = Some(sampled_ty);
                    changed = true;
                }
            }
        }
    }
    if let Some(header) = module.header.as_mut() {
        header.bound = next_id;
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::{Block, Function, ModuleHeader};
    use spirv::Word;

    fn inst(op: Op, ty: Option<Word>, result: Option<Word>, operands: Vec<Operand>) -> Instruction {
        Instruction::new(op, ty, result, operands)
    }

    #[test]
    fn sampled_image_tracks_refined_image_dimension() {
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(32));
        module.types_global_values = vec![
            inst(Op::TypeImage, None, Some(1), vec![]),
            inst(Op::TypeImage, None, Some(2), vec![]),
            inst(Op::TypeSampledImage, None, Some(3), vec![Operand::IdRef(1)]),
            inst(Op::TypeSampledImage, None, Some(4), vec![Operand::IdRef(2)]),
        ];
        let mut block = Block::new();
        block.label = Some(inst(Op::Label, None, Some(10), vec![]));
        block.instructions = vec![
            inst(Op::Load, Some(2), Some(11), vec![Operand::IdRef(20)]),
            inst(Op::Load, Some(5), Some(12), vec![Operand::IdRef(21)]),
            inst(
                Op::SampledImage,
                Some(3),
                Some(13),
                vec![Operand::IdRef(11), Operand::IdRef(12)],
            ),
        ];
        let mut function = Function::new();
        function.blocks.push(block);
        module.functions.push(function);

        assert!(repair_sampled_image_result_types(&mut module));
        assert_eq!(
            module.functions[0].blocks[0].instructions[2].result_type,
            Some(4)
        );
    }
}
