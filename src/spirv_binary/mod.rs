//! Crate-owned parser for the exact locked SPIR-V 1.4.341 core grammar.

mod decoder;
mod error;
mod grammar;
mod parser;
mod type_tracker;

pub(crate) use grammar::{
    instruction_declaration_requirements, instruction_operands_match, instruction_result_shape,
    operand_declaration_requirements,
};
pub(crate) use parser::{parse_bytes, Error};
