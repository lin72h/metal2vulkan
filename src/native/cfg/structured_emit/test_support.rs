//! Shared block fixtures for the structurizer's unit tests.
//!
//! Every `#[cfg(test)] mod` under `structured_emit` builds its CFGs from the same two constructors;
//! they lived as six byte-equivalent private copies before this module. Keeping one copy means a new
//! test module costs a `use super::test_support::*;` line instead of another fixture builder, and a
//! change to how production populates a block carrier is made in one place rather than found in six.

use super::{BlockRole, BodyBlock};

/// A block whose typed carrier is lowered from `lines` exactly as production does at split time
/// (`lower_block_carrier` is the one place block instructions are lexed), with the default
/// [`BlockRole::Normal`] every AIR-sourced block carries.
pub(in crate::native) fn bb(name: &str, lines: &[&str]) -> BodyBlock {
    bb_role(name, BlockRole::Normal, lines)
}

/// [`bb`] with an explicit structural role, for the tests that exercise role-keyed behavior.
pub(in crate::native) fn bb_role(name: &str, role: BlockRole, lines: &[&str]) -> BodyBlock {
    let lines: Vec<String> = lines.iter().map(|line| line.to_string()).collect();
    BodyBlock {
        name: name.to_string(),
        role,
        typed: crate::native::tir::lower_block_carrier(
            name,
            &lines,
            &std::collections::HashMap::new(),
        )
        .map(Into::into),
    }
}
