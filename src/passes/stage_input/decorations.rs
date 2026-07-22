//! Location and builtin decorations retained by interface binding.

use super::*;

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
