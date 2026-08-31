//! Differential test support for the control-flow constructors.
//!
//! The two structurizers ([`crate::native::reloop_nest`] for reducible graphs, the state machine in
//! [`crate::native::relooper`] for everything else) rewrite a function's control flow wholesale.
//! Until now the tree could check that the result is *well formed* — `spirv-val`, the construct
//! ownership checks in [`crate::native::rewrites`], and the value-flow check in
//! [`crate::native::owned_cfg`] all do — and that it has the right *shape*, but nothing checked
//! that it still computes what it used to. A rewrite that drops an edge, selects the wrong `OpPhi`
//! incoming, or reaches a block along a path the original never took passes every one of those.
//!
//! This module closes that gap with the only check that can: author a function
//! ([`build::CfgBuilder`]), execute it ([`interp`]), construct it, execute the construction, and
//! require the same answer on every input. Because authoring and executing are both cheap, adding
//! a shape to the guarded set costs a few lines, which is the point — see [`shapes`] for the
//! generated family and `tests` for the named ones.

pub(in crate::native) mod build;
pub(in crate::native) mod interp;
pub(in crate::native) mod shapes;

#[cfg(test)]
mod tests;

use crate::spirv_module::Module;
use interp::Outcome;

/// The interpreter step budget for one run.
///
/// Generously above what any authored shape needs, so exhausting it means the function does not
/// terminate rather than that it is merely long.
pub(in crate::native) const STEP_LIMIT: usize = 2_000_000;

/// Run `module`'s function on `arguments` before and after control-flow construction and require
/// the same outcome.
///
/// Returns the constructed module so the caller can additionally assert its shape. Panics with the
/// disagreeing input on any semantic difference, which is the failure this exists to report.
pub(in crate::native) fn assert_construction_preserves_semantics(
    module: Module,
    arguments: &[&[u32]],
) -> Module {
    let before = arguments
        .iter()
        .map(|argument| {
            interp::run_module(&module, argument, STEP_LIMIT)
                .unwrap_or_else(|error| panic!("authored function on {argument:?}: {error}"))
        })
        .collect::<Vec<_>>();

    let mut constructed = module;
    crate::native::rewrites::construct_cfg_functions_module(
        &mut constructed,
        &std::collections::HashSet::new(),
    )
    .expect("construction");

    for (argument, expected) in arguments.iter().zip(&before) {
        let actual = interp::run_module(&constructed, argument, STEP_LIMIT)
            .unwrap_or_else(|error| panic!("constructed function on {argument:?}: {error}"));
        assert_eq!(
            &actual, expected,
            "construction changed the result on {argument:?}"
        );
    }
    constructed
}

/// Whether the nesting structurizer adopts `module`'s function, rather than leaving it to the
/// state-machine constructor.
///
/// Declining is always *safe* — the state machine can express any CFG — but it is not free: the
/// state machine is the whole-function dispatch shape that made a driver's shader compiler hang
/// (see `reducible_control_flow_is_not_constructed_as_a_whole_function_dispatch` in
/// [`crate::native::rewrites`]). A change that quietly turns reducible control flow back into
/// declines is a regression even though every semantic test still passes, so tests measure this
/// directly.
pub(in crate::native) fn nests(module: &Module) -> bool {
    let mut module = module.clone();
    let functions = module
        .functions
        .iter()
        .filter_map(|function| function.def.as_ref().and_then(|def| def.result_id))
        .collect::<std::collections::HashSet<_>>();
    !crate::native::reloop_nest::structure_selected_functions(&mut module, &functions).is_empty()
}

/// The outcome of running `module`'s function on each of `arguments`, for callers that want to
/// compare against something other than a construction of the same module.
pub(in crate::native) fn outcomes(module: &Module, arguments: &[&[u32]]) -> Vec<Outcome> {
    arguments
        .iter()
        .map(|argument| {
            interp::run_module(module, argument, STEP_LIMIT)
                .unwrap_or_else(|error| panic!("running on {argument:?}: {error}"))
        })
        .collect()
}
