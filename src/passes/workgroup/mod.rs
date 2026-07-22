//! Workgroup interface type construction and rooted pointer materialization.

use super::*;

mod pointers;
mod types;
mod zero_initialize;

pub(in crate::passes) use pointers::*;
pub(in crate::passes) use types::*;
pub(in crate::passes) use zero_initialize::zero_initialize_workgroup_memory;
