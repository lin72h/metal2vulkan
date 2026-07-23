//! Descriptor-set decoration and ABI binding-number allocation.

use super::*;
use crate::reflect::{COLOR_INPUT_BINDING_BASE, SAMPLER_BINDING_BASE, TEXTURE_BINDING_BASE};

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

pub(in crate::passes) fn allocate_resource_binding(
    binding_ctr: &mut u32,
    fixed: Option<u32>,
) -> u32 {
    let binding = fixed.unwrap_or(*binding_ctr);
    *binding_ctr = (*binding_ctr).max(binding.saturating_add(1));
    binding
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
    (SAMPLER_BINDING_BASE..COLOR_INPUT_BINDING_BASE).find(|binding| !occupied.contains(binding))
}

pub(in crate::passes) fn texture_resource_binding(binding: u32) -> u32 {
    TEXTURE_BINDING_BASE.saturating_add(binding)
}

pub(in crate::passes) fn sampler_resource_binding(binding: u32) -> u32 {
    SAMPLER_BINDING_BASE.saturating_add(binding)
}

pub(in crate::passes) fn color_input_resource_binding(binding: u32) -> u32 {
    COLOR_INPUT_BINDING_BASE.saturating_add(binding)
}
