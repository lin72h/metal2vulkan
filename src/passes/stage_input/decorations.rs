//! Location and builtin decorations retained by interface binding.

use super::*;
use crate::meta::{VaryingInterpolation, VaryingSampling};

pub(in crate::passes) fn decorate_location(module: &mut Module, id: Word, loc: u32) {
    module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(id),
            Operand::Decoration(Decoration::Location),
            Operand::LiteralBit32(loc),
        ],
    ));
}

/// A fragment Input interface variable of integer / 64-bit-float component type cannot be
/// interpolated and MUST carry a `Flat` decoration (VUID-StandaloneSpirv-Flat-04744).
pub(in crate::passes) fn decorate_flat(module: &mut Module, id: Word) {
    module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![Operand::IdRef(id), Operand::Decoration(Decoration::Flat)],
    ));
}

/// Apply the interpolation attribute AIR declared for a fragment Input variable.
///
/// `must_flat` is the type-driven requirement (integer / 64-bit-float components cannot be
/// interpolated, VUID-StandaloneSpirv-Flat-04744) and wins over whatever AIR asked for, because a
/// module that decorates such an input `NoPerspective` or `Centroid` instead of `Flat` does not
/// validate. Otherwise every marker AIR set is carried across: `NoPerspective` for screen-space
/// linear interpolation, and `Centroid`/`Sample` for where in the pixel the value is taken. Vulkan's
/// defaults — perspective-correct, pixel center — are spelled by the absence of a decoration, so
/// `air.perspective` and `air.center` emit nothing.
pub(in crate::passes) fn decorate_interpolation(
    module: &mut Module,
    id: Word,
    interpolation: VaryingInterpolation,
    must_flat: bool,
) {
    if must_flat || interpolation.flat {
        decorate_flat(module, id);
        return;
    }
    if interpolation.no_perspective {
        decorate_with(module, id, Decoration::NoPerspective);
    }
    match interpolation.sampling {
        VaryingSampling::Center => {}
        VaryingSampling::Centroid => decorate_with(module, id, Decoration::Centroid),
        VaryingSampling::Sample => decorate_with(module, id, Decoration::Sample),
    }
}

/// Apply a decoration that takes no literal operands.
pub(in crate::passes) fn decorate_with(module: &mut Module, id: Word, decoration: Decoration) {
    module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![Operand::IdRef(id), Operand::Decoration(decoration)],
    ));
}

pub(in crate::passes) fn decorate_patch(module: &mut Module, id: Word) {
    module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![Operand::IdRef(id), Operand::Decoration(Decoration::Patch)],
    ));
}

pub(in crate::passes) fn decorate_builtin(module: &mut Module, id: Word, b: BuiltIn) {
    module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(id),
            Operand::Decoration(Decoration::BuiltIn),
            Operand::BuiltIn(b),
        ],
    ));
}
