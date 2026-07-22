//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;
use crate::spirv_module::Instruction;
use crate::spirv_module::Module;
use crate::spirv_module::Operand;
use spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

/// Scalar integer + boolean type ids (so we only treat an `OpConstant` over one of these as an
/// integer value, never a float bit pattern).
pub(in crate::native) fn scalar_int_bool_types(module: &Module) -> HashSet<Word> {
    module
        .types_global_values
        .iter()
        .filter(|i| matches!(i.class.opcode, Op::TypeInt | Op::TypeBool))
        .filter_map(|i| i.result_id)
        .collect()
}

/// Map each scalar int/bool constant id to its integer value (bool: false=0, true=1).
pub(in crate::native) fn module_scalar_constants(
    module: &Module,
    int_types: &HashSet<Word>,
) -> HashMap<Word, i128> {
    let mut out = HashMap::new();
    for inst in &module.types_global_values {
        let Some(rid) = inst.result_id else { continue };
        match inst.class.opcode {
            Op::ConstantTrue => {
                out.insert(rid, 1);
            }
            Op::ConstantFalse => {
                out.insert(rid, 0);
            }
            Op::ConstantNull => {
                if inst.result_type.is_some_and(|t| int_types.contains(&t)) {
                    out.insert(rid, 0);
                }
            }
            Op::Constant => {
                if !inst.result_type.is_some_and(|t| int_types.contains(&t)) {
                    continue;
                }
                // A scalar int constant we can fold has a single 32-bit literal operand. Wider
                // (64-bit, two-word) literals are left unknown — predicate folding never needs them.
                if let (Some(Operand::LiteralBit32(v)), None) =
                    (inst.operands.first(), inst.operands.get(1))
                {
                    out.insert(rid, *v as i128);
                }
            }
            _ => {}
        }
    }
    out
}

/// Map each result id to the bit width of its scalar-integer result type, so width-sensitive
/// arithmetic (IAdd/IMul/ISub) can be folded with the correct modular masking. Bool and non-integer
/// results are absent (their folds are width-independent or not tracked).
pub(in crate::native) fn value_int_widths(module: &Module) -> HashMap<Word, u32> {
    let mut type_width: HashMap<Word, u32> = HashMap::new();
    for inst in &module.types_global_values {
        if inst.class.opcode == Op::TypeInt {
            if let (Some(rid), Some(Operand::LiteralBit32(w))) =
                (inst.result_id, inst.operands.first())
            {
                type_width.insert(rid, *w);
            }
        }
    }
    let mut out: HashMap<Word, u32> = HashMap::new();
    for f in &module.functions {
        for b in &f.blocks {
            for inst in &b.instructions {
                if let (Some(rid), Some(t)) = (inst.result_id, inst.result_type) {
                    if let Some(w) = type_width.get(&t) {
                        out.insert(rid, *w);
                    }
                }
            }
        }
    }
    out
}

/// The direct (un-chained) global pointer a Load/Store addresses, if its first operand is a
/// module-scope variable id.
pub(in crate::native) fn direct_global(
    inst: &Instruction,
    global_vars: &HashSet<Word>,
) -> Option<Word> {
    match inst.operands.first() {
        Some(Operand::IdRef(p)) if global_vars.contains(p) => Some(*p),
        _ => None,
    }
}

/// Compute the set of module-scope scalar globals whose every load yields a known constant:
/// (a) never stored anywhere + a constant initializer, then (b) a fixpoint adding globals stored
/// exactly once — in the entry block, with no load preceding the store — by a now-known constant.
pub(in crate::native) fn compute_global_consts(
    module: &Module,
    consts: &HashMap<Word, i128>,
    widths: &HashMap<Word, u32>,
    vec_globals: &HashMap<Word, Vec<i128>>,
) -> HashMap<Word, i128> {
    // Module-scope variables and their (optional) initializer ids.
    let mut global_vars: HashSet<Word> = HashSet::new();
    let mut initializer: HashMap<Word, Word> = HashMap::new();
    for inst in &module.types_global_values {
        if inst.class.opcode != Op::Variable {
            continue;
        }
        let Some(rid) = inst.result_id else { continue };
        global_vars.insert(rid);
        if let Some(Operand::IdRef(init)) = inst.operands.get(1) {
            initializer.insert(rid, *init);
        }
    }

    // Count direct stores per global across the whole module, and remember the single store site.
    let mut store_count: HashMap<Word, usize> = HashMap::new();
    // (function index, block index, instruction index, value operand id)
    let mut single_store: HashMap<Word, (usize, usize, usize, Word)> = HashMap::new();
    for (fi, f) in module.functions.iter().enumerate() {
        for (bi, blk) in f.blocks.iter().enumerate() {
            for (ii, inst) in blk.instructions.iter().enumerate() {
                if inst.class.opcode != Op::Store {
                    continue;
                }
                if let Some(g) = direct_global(inst, &global_vars) {
                    *store_count.entry(g).or_default() += 1;
                    if let Some(Operand::IdRef(v)) = inst.operands.get(1) {
                        single_store.insert(g, (fi, bi, ii, *v));
                    }
                }
            }
        }
    }

    let mut gc: HashMap<Word, i128> = HashMap::new();
    // (a) never-stored globals with a constant initializer.
    for g in &global_vars {
        if store_count.get(g).copied().unwrap_or(0) == 0 {
            if let Some(init) = initializer.get(g) {
                if let Some(c) = consts.get(init) {
                    gc.insert(*g, *c);
                }
            }
        }
    }

    // (b) single-store forwarding fixpoint.
    loop {
        let mut changed = false;
        for (&g, &(fi, bi, ii, vid)) in &single_store {
            if gc.contains_key(&g) || store_count.get(&g).copied().unwrap_or(0) != 1 {
                continue;
            }
            let f = &module.functions[fi];
            // The single store must be in the function entry block, and no load of `g` may precede
            // it there — so the stored value dominates every load of `g`.
            if bi != 0 {
                continue;
            }
            let entry = &f.blocks[0];
            let load_before = entry.instructions[..ii]
                .iter()
                .any(|i| i.class.opcode == Op::Load && direct_global(i, &global_vars) == Some(g));
            if load_before {
                continue;
            }
            // The store's dominance only covers loads in its OWN function; if any other function
            // loads `g`, the forwarded constant is not guaranteed there. (Post-inline this is a
            // single function, but guard it so the rule stays sound for multi-function modules.)
            let loaded_elsewhere = module.functions.iter().enumerate().any(|(other, of)| {
                other != fi
                    && of.blocks.iter().flat_map(|b| &b.instructions).any(|i| {
                        i.class.opcode == Op::Load && direct_global(i, &global_vars) == Some(g)
                    })
            });
            if loaded_elsewhere {
                continue;
            }
            // Evaluate the stored value with the constants known so far.
            let vals = forward_eval(f, consts, &gc, widths, &HashMap::new(), vec_globals);
            if let Some(c) = vals.get(&vid) {
                gc.insert(g, *c);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    gc
}

/// Per-component constant values of module-scope COMPOSITE (vector) constants whose every element is
/// a known scalar integer: `OpConstantNull %vecInt` (all zeros) and `OpConstantComposite %vecInt`
/// with all-constant operands. This is what lets a vector function constant — the shape Metal emits
/// for a `[[function_constant]] ushort4 C` — participate in dead-arm folding: the disabled default is
/// a null vector, and a guard `C[0] == 0` only folds if we can extract element 0 as the constant 0.
pub(in crate::native) fn module_composite_constants(
    module: &Module,
    int_types: &HashSet<Word>,
    consts: &HashMap<Word, i128>,
) -> HashMap<Word, Vec<i128>> {
    // A vector type id whose element type is a tracked integer, plus its component count.
    let mut vec_len: HashMap<Word, u32> = HashMap::new();
    for inst in &module.types_global_values {
        if inst.class.opcode == Op::TypeVector {
            if let (Some(rid), Some(Operand::IdRef(elem)), Some(Operand::LiteralBit32(n))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                if int_types.contains(elem) {
                    vec_len.insert(rid, *n);
                }
            }
        }
    }
    let mut out: HashMap<Word, Vec<i128>> = HashMap::new();
    for inst in &module.types_global_values {
        let Some(rid) = inst.result_id else { continue };
        let Some(ty) = inst.result_type else { continue };
        let Some(&n) = vec_len.get(&ty) else { continue };
        match inst.class.opcode {
            Op::ConstantNull => {
                out.insert(rid, vec![0; n as usize]);
            }
            Op::ConstantComposite => {
                let comps: Option<Vec<i128>> = inst
                    .operands
                    .iter()
                    .map(|op| match op {
                        Operand::IdRef(c) => consts.get(c).copied(),
                        _ => None,
                    })
                    .collect();
                if let Some(comps) = comps {
                    if comps.len() == n as usize {
                        out.insert(rid, comps);
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Module-scope VECTOR globals whose every load yields a known constant vector, mirroring
/// `compute_global_consts` for the composite case: (a) never stored + a composite-constant
/// initializer, then (b) a fixpoint adding globals stored exactly once — in the entry block, before
/// any load — by a value that resolves to a known vector (a module composite constant, or a plain
/// load of an already-known vector global — the `%copy = load %fc_init; store %copy` shape Metal
/// emits when it stages a vector function constant into a local mirror global).
pub(in crate::native) fn compute_vector_global_consts(
    module: &Module,
    composites: &HashMap<Word, Vec<i128>>,
) -> HashMap<Word, Vec<i128>> {
    let mut global_vars: HashSet<Word> = HashSet::new();
    let mut initializer: HashMap<Word, Word> = HashMap::new();
    for inst in &module.types_global_values {
        if inst.class.opcode != Op::Variable {
            continue;
        }
        let Some(rid) = inst.result_id else { continue };
        global_vars.insert(rid);
        if let Some(Operand::IdRef(init)) = inst.operands.get(1) {
            initializer.insert(rid, *init);
        }
    }

    // Count direct stores per global; remember the single store site + its stored value id.
    let mut store_count: HashMap<Word, usize> = HashMap::new();
    let mut single_store: HashMap<Word, (usize, usize, usize, Word)> = HashMap::new();
    for (fi, f) in module.functions.iter().enumerate() {
        for (bi, blk) in f.blocks.iter().enumerate() {
            for (ii, inst) in blk.instructions.iter().enumerate() {
                if inst.class.opcode != Op::Store {
                    continue;
                }
                if let Some(g) = direct_global(inst, &global_vars) {
                    *store_count.entry(g).or_default() += 1;
                    if let Some(Operand::IdRef(v)) = inst.operands.get(1) {
                        single_store.insert(g, (fi, bi, ii, *v));
                    }
                }
            }
        }
    }

    let mut gc: HashMap<Word, Vec<i128>> = HashMap::new();
    // (a) never-stored globals with a composite-constant initializer.
    for g in &global_vars {
        if store_count.get(g).copied().unwrap_or(0) == 0 {
            if let Some(init) = initializer.get(g) {
                if let Some(c) = composites.get(init) {
                    gc.insert(*g, c.clone());
                }
            }
        }
    }

    // Index every result-bearing body instruction so a stored value id can be resolved to its
    // defining op (a load of another global, a composite constant reference, ...).
    let mut def: HashMap<Word, &Instruction> = HashMap::new();
    for f in &module.functions {
        for b in &f.blocks {
            for inst in &b.instructions {
                if let Some(r) = inst.result_id {
                    def.insert(r, inst);
                }
            }
        }
    }

    // Resolve a value id to a known constant vector, given what is known so far.
    let resolve = |vid: Word, gc: &HashMap<Word, Vec<i128>>| -> Option<Vec<i128>> {
        if let Some(c) = composites.get(&vid) {
            return Some(c.clone());
        }
        let inst = def.get(&vid)?;
        match inst.class.opcode {
            Op::Load | Op::CopyObject => match inst.operands.first() {
                Some(Operand::IdRef(p)) => gc.get(p).cloned(),
                _ => None,
            },
            _ => None,
        }
    };

    // (b) single-store forwarding fixpoint.
    loop {
        let mut changed = false;
        for (&g, &(fi, bi, ii, vid)) in &single_store {
            if gc.contains_key(&g) || store_count.get(&g).copied().unwrap_or(0) != 1 || bi != 0 {
                continue;
            }
            let entry = &module.functions[fi].blocks[0];
            let load_before = entry.instructions[..ii]
                .iter()
                .any(|i| i.class.opcode == Op::Load && direct_global(i, &global_vars) == Some(g));
            if load_before {
                continue;
            }
            let loaded_elsewhere = module.functions.iter().enumerate().any(|(other, of)| {
                other != fi
                    && of.blocks.iter().flat_map(|b| &b.instructions).any(|i| {
                        i.class.opcode == Op::Load && direct_global(i, &global_vars) == Some(g)
                    })
            });
            if loaded_elsewhere {
                continue;
            }
            if let Some(c) = resolve(vid, &gc) {
                gc.insert(g, c);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    gc
}

/// Module-scope `OpVariable`s decorated `BuiltIn NumWorkgroups`. Their loaded value is the compute
/// dispatch grid size, which is `>= 1` in every component for any invocation that actually executes
/// the shader (a zero-sized grid launches no invocation). That positivity is the seed for the
/// nonzero analysis that folds grid-stride early-return guards.
pub(in crate::native) fn numworkgroups_vars(module: &Module) -> HashSet<Word> {
    module
        .annotations
        .iter()
        .filter_map(|inst| {
            if inst.class.opcode != Op::Decorate {
                return None;
            }
            let target = match inst.operands.first() {
                Some(Operand::IdRef(t)) => *t,
                _ => return None,
            };
            match inst.operands.get(1) {
                Some(Operand::Decoration(spirv::Decoration::BuiltIn)) => {
                    match inst.operands.get(2) {
                        Some(Operand::BuiltIn(spirv::BuiltIn::NumWorkgroups)) => Some(target),
                        _ => None,
                    }
                }
                _ => None,
            }
        })
        .collect()
}
