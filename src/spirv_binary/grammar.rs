#![allow(dead_code)]

use spirv::Op;
use std::marker::PhantomData;

#[derive(Debug)]
pub(super) struct InstructionGrammar {
    pub(super) opcode: Op,
    pub(super) operands: &'static [LogicalOperand],
}

#[derive(Clone, Copy, Debug)]
pub(super) struct LogicalOperand {
    pub(super) kind: OperandKind,
    pub(super) quantifier: OperandQuantifier,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum OperandQuantifier {
    One,
    ZeroOrOne,
    ZeroOrMore,
}

macro_rules! inst {
    ($op:ident, [$($cap:ident),*], [$($ext:expr),*], [$(($kind:ident, $quant:ident)),*]) => {
        InstructionGrammar {
            opcode: Op::$op,
            operands: &[
                $(LogicalOperand {
                    kind: OperandKind::$kind,
                    quantifier: OperandQuantifier::$quant,
                }),*
            ],
        }
    };
}

pub(super) struct InstructionTable(&'static [InstructionGrammar], PhantomData<Op>);

impl InstructionTable {
    pub(super) fn lookup_opcode(&self, opcode: u32) -> Option<&'static InstructionGrammar> {
        self.0
            .iter()
            .find(|instruction| instruction.opcode as u32 == opcode)
    }
}

include!("grammar_generated.rs");
