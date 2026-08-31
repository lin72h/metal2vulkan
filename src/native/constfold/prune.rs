//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;
use crate::spirv_module::Instruction;
use crate::spirv_module::Module;
use crate::spirv_module::Operand;
use spirv::{Op, Word};
use std::collections::HashSet;

/// Entry: fold statically-constant branches and DCE the resulting dead code, to a fixpoint.
/// Returns `true` if anything changed.
pub(in crate::native) fn prune_constant_branches(module: &mut Module) -> bool {
    prune_constant_branches_impl(module, &HashSet::new(), true)
}

/// Fold constant-controlled CFG without sweeping unrelated unused values. This is the primary
/// construction form: it may run for every module without turning the failure-recovery DCE into a
/// byte-changing whole-module optimization.
pub(in crate::native) fn prune_constant_cfg(module: &mut Module) -> bool {
    prune_constant_branches_impl(module, &HashSet::new(), false)
}

fn prune_constant_branches_impl(
    module: &mut Module,
    preserved_global_ids: &HashSet<Word>,
    sweep_dead_values: bool,
) -> bool {
    let scalar_int_types = scalar_int_bool_types(module);
    let consts = module_scalar_constants(module, &scalar_int_types);
    let widths = value_int_widths(module);
    let composites = module_composite_constants(module, &scalar_int_types, &consts);
    let vec_globals = compute_vector_global_consts(module, &composites);
    let global_consts = compute_global_consts(module, &consts, &widths, &vec_globals);
    let numworkgroups = numworkgroups_vars(module);
    if consts.is_empty() && global_consts.is_empty() && vec_globals.is_empty() {
        return false;
    }

    let mut any = false;
    let mut cfg_was_pruned = false;
    // Module-level fixpoint: each pass may expose new constant conditions (a collapsed single-pred
    // phi whose arms are equal, a freshly-dead block) for the next.
    loop {
        let mut changed = false;
        for fi in 0..module.functions.len() {
            let mut vals = forward_eval(
                &module.functions[fi],
                &consts,
                &global_consts,
                &widths,
                &composites,
                &vec_globals,
            );
            // Grid-stride early-return guards: `if (stride > work + stride - 1) return;` folds to a
            // taken return once `work` (an FC-derived count) is a known 0, because `stride > stride-1`
            // holds for every dispatched invocation (`NumWorkgroups >= 1`). Fed in alongside the SCCP
            // constants so `fold_branches` prunes the guarded compute nest uniformly. Passed the full
            // folded `vals` (module constants PLUS folded loads of constant globals) so the guard's
            // affine offset threads through `n_requ_simd_groups` (an FC-derived 0 global).
            let guards = nonzero_self_minus_one_guards(
                &module.functions[fi],
                &vals,
                &widths,
                &numworkgroups,
            );
            for (g, v) in guards {
                vals.entry(g).or_insert(v);
            }
            if sweep_dead_values {
                changed |= fold_branches(&mut module.functions[fi], &vals);
                changed |= prune_unreachable(&mut module.functions[fi]);
                changed |= collapse_trivial_phis(&mut module.functions[fi]);
            } else {
                let folded = fold_branches(&mut module.functions[fi], &vals);
                let pruned = prune_unreachable(&mut module.functions[fi]);
                let collapsed =
                    (folded || pruned) && collapse_trivial_phis(&mut module.functions[fi]);
                changed |= folded || pruned || collapsed;
                if folded || pruned {
                    cfg_was_pruned = true;
                }
            }
        }
        // DCE is module-wide (uses cross blocks/functions); run once per outer iteration.
        if sweep_dead_values {
            changed |= dce_preserving(module, preserved_global_ids);
        }
        // Once branch folding removes every call to an unhandled-intrinsic function (igemm /
        // load.with.emask / ... — emitted as a BODYLESS OpFunction declaration), that declaration is
        // dead: a non-imported function with no basic blocks is invalid SPIR-V, so it must be swept
        // or the pruned module never validates. Fold-then-sweep, once per outer iteration.
        if sweep_dead_values || cfg_was_pruned {
            changed |= sweep_uncalled_functions(module);
        }
        any |= changed;
        if !changed {
            break;
        }
    }
    any
}

/// Remove functions unreachable from any entry point through the `OpFunctionCall` graph, plus the
/// debug names / decorations that target the removed ids. General and semantics-preserving: an
/// uncalled, non-entry function is dead by definition. This is what lets branch-fold clear the last
/// wall for an FC-dead cluster whose only remaining artifacts are the bodyless intrinsic
/// declarations (igemm / emask load-store) — invalid SPIR-V that the plain DCE leaves untouched
/// because it only sweeps pure result instructions and null/undef constants, never whole functions.
pub(in crate::native) fn sweep_uncalled_functions(module: &mut Module) -> bool {
    let fn_id = |f: &crate::spirv_module::Function| -> Option<Word> { f.def.as_ref()?.result_id };

    // Entry-point function ids are always live.
    let mut live: HashSet<Word> = HashSet::new();
    for ep in &module.entry_points {
        if let Some(Operand::IdRef(id)) = ep.operands.get(1) {
            live.insert(*id);
        }
    }
    // Transitive closure over the call graph: a live function keeps every function it calls live.
    loop {
        let mut added = false;
        for f in &module.functions {
            let Some(id) = fn_id(f) else { continue };
            if !live.contains(&id) {
                continue;
            }
            for b in &f.blocks {
                for inst in &b.instructions {
                    if inst.class.opcode == Op::FunctionCall {
                        if let Some(Operand::IdRef(callee)) = inst.operands.first() {
                            if live.insert(*callee) {
                                added = true;
                            }
                        }
                    }
                }
            }
        }
        if !added {
            break;
        }
    }

    // Collect every id defined by a dead function (its own id, its parameters, and all body result
    // ids) so we can drop dangling debug names / decorations that target them.
    let mut removed_ids: HashSet<Word> = HashSet::new();
    for f in &module.functions {
        let Some(id) = fn_id(f) else { continue };
        if live.contains(&id) {
            continue;
        }
        removed_ids.insert(id);
        for p in &f.parameters {
            if let Some(r) = p.result_id {
                removed_ids.insert(r);
            }
        }
        for b in &f.blocks {
            if let Some(r) = b.label.as_ref().and_then(|l| l.result_id) {
                removed_ids.insert(r);
            }
            for inst in &b.instructions {
                if let Some(r) = inst.result_id {
                    removed_ids.insert(r);
                }
            }
        }
    }
    if removed_ids.is_empty() {
        return false;
    }

    module.functions.retain(|f| match fn_id(f) {
        Some(id) => live.contains(&id),
        None => true,
    });
    // Drop debug names / decorations whose TARGET (operand 0) is a removed id, so no reference dangles.
    let targets_removed = |inst: &Instruction| -> bool {
        matches!(inst.operands.first(), Some(Operand::IdRef(t)) if removed_ids.contains(t))
    };
    module.debug_names.retain(|i| !targets_removed(i));
    module.annotations.retain(|i| !targets_removed(i));
    true
}
