//! Crate-owned parser for the exact locked SPIR-V 1.4.341 core grammar.

mod decoder;
mod error;
mod grammar;
mod parser;
mod type_tracker;

pub(crate) use parser::{parse_bytes, Error};
