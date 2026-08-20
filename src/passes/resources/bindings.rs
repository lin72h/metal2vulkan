//! Descriptor-set decoration and ABI binding-number allocation.

use super::*;
use crate::reflect::{RESOURCE_DESCRIPTOR_SET, SAMPLER_BINDING_RANGE, TEXTURE_BINDING_RANGE};

/// Add `DescriptorSet 0` and `Binding` decorations to one resource variable.
pub(in crate::passes) fn decorate_binding(module: &mut Module, id: Word, binding: u32) {
    module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(id),
            Operand::Decoration(Decoration::DescriptorSet),
            Operand::LiteralBit32(0),
        ],
    ));
    module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(id),
            Operand::Decoration(Decoration::Binding),
            Operand::LiteralBit32(binding),
        ],
    ));
}

pub(in crate::passes) fn decorate_input_attachment_index(
    module: &mut Module,
    id: Word,
    index: u32,
) {
    module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(id),
            Operand::Decoration(Decoration::InputAttachmentIndex),
            Operand::LiteralBit32(index),
        ],
    ));
}

/// Allocate one translator-owned constexpr sampler inside the sampler ABI band. Unlike the generic
/// high-water allocator, this cannot escape into the color-input/buffer ranges when a shader also
/// declares `[[color(n)]]`; it fills the first sampler slot not already claimed by a runtime
/// `[[sampler(n)]]`.
pub(in crate::passes) fn allocate_static_sampler_binding(module: &Module) -> Option<u32> {
    let occupied = module
        .annotations
        .iter()
        .filter(|instruction| {
            instruction.class.opcode == Op::Decorate
                && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::Binding))
        })
        .filter_map(|instruction| match instruction.operands.get(2) {
            Some(Operand::LiteralBit32(binding)) => Some(*binding),
            _ => None,
        })
        .collect::<std::collections::HashSet<_>>();
    (SAMPLER_BINDING_RANGE.start..SAMPLER_BINDING_RANGE.end)
        .find(|binding| !occupied.contains(binding))
}

/// Allocate one translator-owned null-image descriptor inside the texture ABI band. It must never
/// use a sampler/color-input binding merely because those bands contain the global high-water mark.
pub(in crate::passes) fn allocate_default_texture_binding(module: &Module) -> Option<u32> {
    let occupied = module
        .annotations
        .iter()
        .filter(|instruction| {
            instruction.class.opcode == Op::Decorate
                && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::Binding))
        })
        .filter_map(|instruction| match instruction.operands.get(2) {
            Some(Operand::LiteralBit32(binding)) => Some(*binding),
            _ => None,
        })
        .collect::<std::collections::HashSet<_>>();
    (TEXTURE_BINDING_RANGE.start..TEXTURE_BINDING_RANGE.end)
        .find(|binding| !occupied.contains(binding))
}

pub(in crate::passes) fn texture_resource_binding(index: u32) -> Result<u32, String> {
    crate::reflect::texture_resource_binding(index)
        .ok_or_else(|| format!("Metal texture index {index} exceeds the descriptor ABI band"))
}

pub(in crate::passes) fn storage_texture_resource_binding(index: u32) -> Result<u32, String> {
    crate::reflect::storage_texture_resource_binding(index).ok_or_else(|| {
        format!("Metal storage-texture index {index} exceeds the descriptor ABI band")
    })
}

pub(in crate::passes) fn buffer_resource_binding(index: u32) -> Result<u32, String> {
    crate::reflect::buffer_resource_binding(index)
        .ok_or_else(|| format!("Metal buffer index {index} exceeds the descriptor ABI band"))
}

pub(in crate::passes) fn sampler_resource_binding(index: u32) -> Result<u32, String> {
    crate::reflect::sampler_resource_binding(index)
        .ok_or_else(|| format!("Metal sampler index {index} exceeds the descriptor ABI band"))
}

pub(in crate::passes) fn color_input_resource_binding(index: u32) -> Result<u32, String> {
    crate::reflect::color_input_resource_binding(index)
        .ok_or_else(|| format!("Metal color-input index {index} exceeds the descriptor ABI band"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DescriptorClass {
    StorageBuffer,
    UniformBuffer,
    Sampler,
    SampledImage,
    CombinedImageSampler,
    StorageImage,
    InputAttachment,
    AccelerationStructure,
}

/// Reject a completed module whose descriptor decorations cannot describe one Vulkan set layout.
/// Multiple variables may alias one location only when their descriptor classes agree; this keeps
/// intentional same-buffer typed aliases legal while catching cross-class band collisions.
pub(in crate::passes) fn validate_descriptor_binding_classes(
    module: &Module,
) -> Result<(), String> {
    let names = module
        .debug_names
        .iter()
        .filter_map(|instruction| {
            if instruction.class.opcode != Op::Name {
                return None;
            }
            let (Some(Operand::IdRef(id)), Some(Operand::LiteralString(name))) =
                (instruction.operands.first(), instruction.operands.get(1))
            else {
                return None;
            };
            Some((*id, name.as_str()))
        })
        .collect::<HashMap<_, _>>();
    let label = |id: Word| match names.get(&id) {
        Some(name) => format!("%{id} ({name})"),
        None => format!("%{id}"),
    };
    let definitions = module
        .types_global_values
        .iter()
        .filter_map(|instruction| instruction.result_id.map(|result| (result, instruction)))
        .collect::<HashMap<_, _>>();
    let mut sets = HashMap::<Word, u32>::new();
    let mut bindings = HashMap::<Word, u32>::new();
    for annotation in &module.annotations {
        if annotation.class.opcode != Op::Decorate {
            continue;
        }
        let (
            Some(Operand::IdRef(target)),
            Some(Operand::Decoration(decoration)),
            Some(Operand::LiteralBit32(value)),
        ) = (
            annotation.operands.first(),
            annotation.operands.get(1),
            annotation.operands.get(2),
        )
        else {
            continue;
        };
        match decoration {
            Decoration::DescriptorSet => {
                if let Some(previous) = sets.insert(*target, *value) {
                    if previous != *value {
                        return Err(format!(
                            "descriptor variable {} has conflicting sets {previous} and {value}",
                            label(*target)
                        ));
                    }
                }
            }
            Decoration::Binding => {
                if let Some(previous) = bindings.insert(*target, *value) {
                    if previous != *value {
                        return Err(format!(
                            "descriptor variable {} has conflicting bindings {previous} and {value}",
                            label(*target)
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    fn classify(
        definitions: &HashMap<Word, &Instruction>,
        ty: Word,
        storage: Option<StorageClass>,
    ) -> Option<DescriptorClass> {
        let definition = definitions.get(&ty)?;
        match definition.class.opcode {
            Op::TypePointer => {
                let Operand::StorageClass(pointer_storage) = definition.operands.first()? else {
                    return None;
                };
                let Operand::IdRef(pointee) = definition.operands.get(1)? else {
                    return None;
                };
                classify(definitions, *pointee, Some(*pointer_storage))
            }
            Op::TypeArray | Op::TypeRuntimeArray => {
                let Operand::IdRef(element) = definition.operands.first()? else {
                    return None;
                };
                classify(definitions, *element, storage)
            }
            Op::TypeSampler => Some(DescriptorClass::Sampler),
            Op::TypeSampledImage => Some(DescriptorClass::CombinedImageSampler),
            Op::TypeImage => {
                if definition.operands.get(1) == Some(&Operand::Dim(spirv::Dim::DimSubpassData)) {
                    Some(DescriptorClass::InputAttachment)
                } else if definition.operands.get(5) == Some(&Operand::LiteralBit32(2)) {
                    Some(DescriptorClass::StorageImage)
                } else {
                    Some(DescriptorClass::SampledImage)
                }
            }
            Op::TypeAccelerationStructureKHR => Some(DescriptorClass::AccelerationStructure),
            _ if storage == Some(StorageClass::StorageBuffer) => {
                Some(DescriptorClass::StorageBuffer)
            }
            _ if storage == Some(StorageClass::Uniform) => Some(DescriptorClass::UniformBuffer),
            _ => None,
        }
    }

    let mut occupied = std::collections::BTreeMap::<(u32, u32), (Word, DescriptorClass)>::new();
    for variable in module
        .types_global_values
        .iter()
        .filter(|instruction| instruction.class.opcode == Op::Variable)
    {
        let Some(id) = variable.result_id else {
            continue;
        };
        let Some(binding) = bindings.get(&id).copied() else {
            continue;
        };
        let set = sets
            .get(&id)
            .copied()
            .ok_or_else(|| format!("descriptor variable {} has no descriptor set", label(id)))?;
        let class = variable
            .result_type
            .and_then(|ty| classify(&definitions, ty, None))
            .ok_or_else(|| {
                format!(
                    "descriptor variable {} has an unknown descriptor class",
                    label(id)
                )
            })?;
        let allowed = match class {
            DescriptorClass::StorageBuffer
            | DescriptorClass::UniformBuffer
            | DescriptorClass::AccelerationStructure => {
                crate::reflect::BUFFER_BINDING_RANGE.contains(binding)
                    || binding >= crate::reflect::SYNTHETIC_BINDING_BASE
            }
            DescriptorClass::Sampler => crate::reflect::SAMPLER_BINDING_RANGE.contains(binding),
            DescriptorClass::SampledImage | DescriptorClass::CombinedImageSampler => {
                crate::reflect::TEXTURE_BINDING_RANGE.contains(binding)
            }
            DescriptorClass::StorageImage => {
                crate::reflect::STORAGE_TEXTURE_BINDING_RANGE.contains(binding)
                    || crate::reflect::IMAGEBLOCK_BINDING_RANGE.contains(binding)
                    || crate::reflect::FRAGMENT_IMAGEBLOCK_BINDING_RANGE.contains(binding)
            }
            DescriptorClass::InputAttachment => {
                crate::reflect::COLOR_INPUT_BINDING_RANGE.contains(binding)
            }
        };
        if set != RESOURCE_DESCRIPTOR_SET || !allowed {
            return Err(format!(
                "descriptor variable {} has class {class:?} at set {set} binding {binding}, outside its ABI band",
                label(id)
            ));
        }
        if let Some((previous, previous_class)) = occupied.insert((set, binding), (id, class)) {
            if previous_class != class {
                return Err(format!(
                    "descriptor set {set} binding {binding} is shared by {} ({previous_class:?}) and {} ({class:?})",
                    label(previous), label(id)
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_sampler_allocator_is_bounded_by_its_descriptor_band() {
        let mut module = Module::new();
        decorate_binding(&mut module, 1, TEXTURE_BINDING_RANGE.end - 1);
        decorate_binding(
            &mut module,
            2,
            crate::reflect::COLOR_INPUT_BINDING_RANGE.start,
        );
        assert_eq!(
            allocate_static_sampler_binding(&module),
            Some(SAMPLER_BINDING_RANGE.start)
        );

        for (offset, binding) in
            (SAMPLER_BINDING_RANGE.start..SAMPLER_BINDING_RANGE.end).enumerate()
        {
            decorate_binding(&mut module, 100 + offset as u32, binding);
        }
        assert_eq!(allocate_static_sampler_binding(&module), None);
    }

    #[test]
    fn null_texture_allocator_is_bounded_by_its_descriptor_band() {
        let mut module = Module::new();
        decorate_binding(&mut module, 1, SAMPLER_BINDING_RANGE.start);
        assert_eq!(
            allocate_default_texture_binding(&module),
            Some(TEXTURE_BINDING_RANGE.start)
        );

        for (offset, binding) in
            (TEXTURE_BINDING_RANGE.start..TEXTURE_BINDING_RANGE.end).enumerate()
        {
            decorate_binding(&mut module, 100 + offset as u32, binding);
        }
        assert_eq!(allocate_default_texture_binding(&module), None);
    }
}
