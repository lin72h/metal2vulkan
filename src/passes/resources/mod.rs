//! Resource and buffer-address discovery over the retained typed module.

use super::*;

mod bindings;
mod buffer_addresses;
mod collapse;
mod discovery;
mod query_select;
pub(in crate::passes) mod rewrites;
mod texture_array;

pub(in crate::passes) use bindings::*;
pub(in crate::passes) use buffer_addresses::*;
pub(in crate::passes) use collapse::*;
pub(in crate::passes) use discovery::*;
pub(in crate::passes) use query_select::rewrite_resource_query_selects;
pub(in crate::passes) use texture_array::materialize_texture_array_loads;
