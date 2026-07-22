//! Byte-neutral responsibility split of the former monolith; see the parent module.

use crate::spirv_module::Instruction;
use crate::spirv_module::Module;
use crate::spirv_module::Operand;
use spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

pub(in crate::native) fn block_id(blk: &crate::spirv_module::Block) -> Option<Word> {
    blk.label.as_ref().and_then(|l| l.result_id)
}

/// Fold every `OpBranchConditional` whose condition is a known constant into an `OpBranch` to the
/// taken target, dropping the preceding `OpSelectionMerge`. Loop-header conditionals (preceded by
/// `OpLoopMerge`) are left untouched — removing a loop merge would break the loop construct.
pub(in crate::native) fn fold_branches(
    f: &mut crate::spirv_module::Function,
    vals: &HashMap<Word, i128>,
) -> bool {
    let mut changed = false;
    for blk in &mut f.blocks {
        let n = blk.instructions.len();
        if n == 0 {
            continue;
        }
        let term = &blk.instructions[n - 1];
        if term.class.opcode != Op::BranchConditional {
            continue;
        }
        let Some(Operand::IdRef(cond)) = term.operands.first() else {
            continue;
        };
        let Some(c) = vals.get(cond).copied() else {
            continue;
        };
        let (Some(Operand::IdRef(t)), Some(Operand::IdRef(fl))) =
            (term.operands.get(1), term.operands.get(2))
        else {
            continue;
        };
        let taken = if c != 0 { *t } else { *fl };
        // Inspect the merge instruction preceding the terminator.
        let has_selection_merge =
            n >= 2 && blk.instructions[n - 2].class.opcode == Op::SelectionMerge;
        let has_loop_merge = n >= 2 && blk.instructions[n - 2].class.opcode == Op::LoopMerge;
        if has_loop_merge {
            continue; // never fold a loop header conditional
        }
        blk.instructions[n - 1] =
            Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(taken)]);
        if has_selection_merge {
            blk.instructions.remove(n - 2);
        }
        changed = true;
    }
    changed
}

/// Successor block ids of a terminator.
pub(in crate::native) fn successors(term: &Instruction) -> Vec<Word> {
    let id = |o: Option<&Operand>| match o {
        Some(Operand::IdRef(w)) => Some(*w),
        _ => None,
    };
    match term.class.opcode {
        Op::Branch => id(term.operands.first()).into_iter().collect(),
        Op::BranchConditional => id(term.operands.get(1))
            .into_iter()
            .chain(id(term.operands.get(2)))
            .collect(),
        Op::Switch => {
            // operands: selector, default, then (literal, label) pairs.
            let mut out: Vec<Word> = id(term.operands.get(1)).into_iter().collect();
            let mut i = 2;
            while i < term.operands.len() {
                if let Some(Operand::IdRef(w)) = term.operands.get(i + 1) {
                    out.push(*w);
                }
                i += 2;
            }
            out
        }
        _ => vec![],
    }
}

/// Remove blocks unreachable from the entry block, and fix phis in survivors to list exactly their
/// surviving predecessors.
pub(in crate::native) fn prune_unreachable(f: &mut crate::spirv_module::Function) -> bool {
    if f.blocks.is_empty() {
        return false;
    }
    let entry = match block_id(&f.blocks[0]) {
        Some(e) => e,
        None => return false,
    };
    let succ: HashMap<Word, Vec<Word>> = f
        .blocks
        .iter()
        .filter_map(|b| {
            let id = block_id(b)?;
            let term = b.instructions.last()?;
            Some((id, successors(term)))
        })
        .collect();
    // BFS reachability.
    let mut reach: HashSet<Word> = HashSet::new();
    let mut stack = vec![entry];
    while let Some(b) = stack.pop() {
        if !reach.insert(b) {
            continue;
        }
        if let Some(ss) = succ.get(&b) {
            for s in ss {
                if !reach.contains(s) {
                    stack.push(*s);
                }
            }
        }
    }
    let before = f.blocks.len();
    f.blocks
        .retain(|b| block_id(b).is_some_and(|id| reach.contains(&id)));
    let removed = f.blocks.len() != before;

    // Drop dangling structured-merge instructions: an `OpSelectionMerge`/`OpLoopMerge` whose merge
    // (or, for a loop, continue) target block was just pruned as unreachable is a forward reference
    // to a non-existent id — invalid SPIR-V that even the relooper cannot parse. Removing the merge
    // dissolves the (now incomplete) structured construct, leaving a well-formed — if unstructured —
    // module that the prune-then-relooper retry re-structures. This is what folding a function-
    // constant branch whose taken arm convergence was the merge produces (the merge block becomes
    // reachable only through the pruned not-taken path).
    let alive: HashSet<Word> = f.blocks.iter().filter_map(block_id).collect();
    let mut merge_fixed = false;
    for b in &mut f.blocks {
        let n = b.instructions.len();
        if n < 2 {
            continue;
        }
        let mi = n - 2;
        let op = b.instructions[mi].class.opcode;
        if op != Op::SelectionMerge && op != Op::LoopMerge {
            continue;
        }
        let merge_gone = matches!(
            b.instructions[mi].operands.first(),
            Some(Operand::IdRef(m)) if !alive.contains(m)
        );
        let cont_gone = op == Op::LoopMerge
            && matches!(
                b.instructions[mi].operands.get(1),
                Some(Operand::IdRef(c)) if !alive.contains(c)
            );
        if merge_gone || cont_gone {
            b.instructions.remove(mi);
            merge_fixed = true;
        }
    }

    // Recompute actual predecessors among survivors, then fix every phi to those preds.
    let mut preds: HashMap<Word, HashSet<Word>> = HashMap::new();
    for b in &f.blocks {
        let Some(id) = block_id(b) else { continue };
        if let Some(term) = b.instructions.last() {
            for s in successors(term) {
                preds.entry(s).or_default().insert(id);
            }
        }
    }
    let mut phi_fixed = false;
    for b in &mut f.blocks {
        let Some(id) = block_id(b) else { continue };
        let allowed = preds.get(&id).cloned().unwrap_or_default();
        for inst in &mut b.instructions {
            if inst.class.opcode != Op::Phi {
                continue;
            }
            // operands: (value, parent) pairs.
            let mut kept: Vec<Operand> = Vec::new();
            let mut i = 0;
            while i + 1 < inst.operands.len() {
                if let Operand::IdRef(parent) = inst.operands[i + 1] {
                    if allowed.contains(&parent) {
                        kept.push(inst.operands[i].clone());
                        kept.push(inst.operands[i + 1].clone());
                    } else {
                        phi_fixed = true;
                    }
                }
                i += 2;
            }
            inst.operands = kept;
        }
    }
    removed || phi_fixed || merge_fixed
}

/// Replace single-incoming phis with their value (substituting uses module-wide).
pub(in crate::native) fn collapse_trivial_phis(f: &mut crate::spirv_module::Function) -> bool {
    let mut repl: HashMap<Word, Word> = HashMap::new();
    for b in &f.blocks {
        for inst in &b.instructions {
            if inst.class.opcode == Op::Phi && inst.operands.len() == 2 {
                if let (Some(rid), Some(Operand::IdRef(v))) =
                    (inst.result_id, inst.operands.first())
                {
                    repl.insert(rid, *v);
                }
            }
        }
    }
    if repl.is_empty() {
        return false;
    }
    // Resolve chains (a phi whose value is itself a collapsed phi).
    let resolve = |mut x: Word| -> Word {
        let mut guard = 0;
        while let Some(&n) = repl.get(&x) {
            if n == x || guard > repl.len() {
                break;
            }
            x = n;
            guard += 1;
        }
        x
    };
    // Drop the collapsed phis.
    for b in &mut f.blocks {
        b.instructions.retain(|i| {
            !(i.class.opcode == Op::Phi && i.result_id.is_some_and(|r| repl.contains_key(&r)))
        });
    }
    // Substitute uses everywhere in this function.
    for b in &mut f.blocks {
        for inst in &mut b.instructions {
            for op in &mut inst.operands {
                if let Operand::IdRef(id) = op {
                    if repl.contains_key(id) {
                        *op = Operand::IdRef(resolve(*id));
                    }
                }
            }
        }
    }
    true
}

/// Whether an instruction is a pure value computation safe to delete when its result is unused.
pub(in crate::native) fn is_pure(op: Op) -> bool {
    matches!(
        op,
        Op::Undef
            | Op::AccessChain
            | Op::InBoundsAccessChain
            | Op::PtrAccessChain
            | Op::Load
            | Op::CopyObject
            | Op::Bitcast
            | Op::UConvert
            | Op::SConvert
            | Op::FConvert
            | Op::ConvertSToF
            | Op::ConvertUToF
            | Op::ConvertFToS
            | Op::ConvertFToU
            | Op::IAdd
            | Op::ISub
            | Op::IMul
            | Op::UDiv
            | Op::SDiv
            | Op::UMod
            | Op::SMod
            | Op::SRem
            | Op::FAdd
            | Op::FSub
            | Op::FMul
            | Op::FDiv
            | Op::FNegate
            | Op::SNegate
            | Op::ShiftLeftLogical
            | Op::ShiftRightLogical
            | Op::ShiftRightArithmetic
            | Op::BitwiseAnd
            | Op::BitwiseOr
            | Op::BitwiseXor
            | Op::Not
            | Op::IEqual
            | Op::INotEqual
            | Op::ULessThan
            | Op::SLessThan
            | Op::UGreaterThan
            | Op::SGreaterThan
            | Op::ULessThanEqual
            | Op::SLessThanEqual
            | Op::UGreaterThanEqual
            | Op::SGreaterThanEqual
            | Op::FOrdEqual
            | Op::FOrdNotEqual
            | Op::FOrdLessThan
            | Op::FOrdGreaterThan
            | Op::FOrdLessThanEqual
            | Op::FOrdGreaterThanEqual
            | Op::LogicalNot
            | Op::LogicalAnd
            | Op::LogicalOr
            | Op::LogicalEqual
            | Op::LogicalNotEqual
            | Op::Select
            | Op::Phi
            | Op::CompositeExtract
            | Op::CompositeConstruct
            | Op::CompositeInsert
            | Op::VectorShuffle
            | Op::VectorExtractDynamic
            | Op::VectorTimesScalar
            | Op::MatrixTimesVector
            | Op::MatrixTimesMatrix
            | Op::Transpose
            | Op::Dot
    )
}

/// Remove pure, result-bearing instructions whose result is dead. Liveness is computed
/// TRANSITIVELY from roots (non-pure sinks — terminators, stores, merge instructions, impure
/// result ops, plus decorations/debug-names/entry-points/types), not by "any operand reference =
/// used": a pure instruction's operands only count as a use when the instruction itself is live.
/// This is what lets a self-referential DEAD CYCLE be collected — e.g. a loop-carried pointer
/// phi whose only remaining reference is its own back-edge `OpPtrAccessChain` (the consumer load
/// having been pruned with its statically-dead FC arm). The naive mark "an id is used if it
/// appears in any operand" keeps such a cycle alive forever (the phi marks the access chain used,
/// the access chain marks the phi used), so the mistyped pointer phi survives and the module stays
/// invalid; transitive liveness from sinks drops the whole cycle in one pass.
#[cfg(test)]
pub(in crate::native) fn dce(module: &mut Module) -> bool {
    dce_preserving(module, &HashSet::new())
}

pub(in crate::native) fn dce_preserving(
    module: &mut Module,
    preserved_global_ids: &HashSet<Word>,
) -> bool {
    // Pure, result-bearing definitions: result id -> the operand ids it depends on. A live result
    // propagates liveness to these; a result reachable from no sink is dead.
    let mut pure_def: HashMap<Word, Vec<Word>> = HashMap::new();
    for f in &module.functions {
        for b in &f.blocks {
            for inst in &b.instructions {
                if is_pure(inst.class.opcode) {
                    if let Some(r) = inst.result_id {
                        let deps = inst
                            .operands
                            .iter()
                            .filter_map(|op| match op {
                                Operand::IdRef(id) => Some(*id),
                                _ => None,
                            })
                            .collect();
                        pure_def.insert(r, deps);
                    }
                }
            }
        }
    }

    // Seed the worklist from every reference that is NOT an operand of a pure result-bearing
    // instruction: module roots (decorations, names, entry points, exec modes, type/global
    // operands) and, in function bodies, the operands of sinks (terminators, stores, merges,
    // OpExtInst, impure result ops, labels).
    let mut work: Vec<Word> = Vec::new();
    let seed = |ops: &[Operand], work: &mut Vec<Word>| {
        for op in ops {
            if let Operand::IdRef(id) = op {
                work.push(*id);
            }
        }
    };
    for s in [
        &module.entry_points,
        &module.debug_names,
        &module.annotations,
        &module.execution_modes,
    ] {
        for inst in s {
            seed(&inst.operands, &mut work);
        }
    }
    work.extend(preserved_global_ids.iter().copied());
    for inst in &module.types_global_values {
        seed(&inst.operands, &mut work);
    }
    for f in &module.functions {
        for b in &f.blocks {
            for inst in &b.instructions {
                let is_pure_def = is_pure(inst.class.opcode) && inst.result_id.is_some();
                if !is_pure_def {
                    seed(&inst.operands, &mut work);
                }
            }
        }
    }

    // Transitive closure: a live pure result keeps its operands live.
    let mut live: HashSet<Word> = HashSet::new();
    while let Some(id) = work.pop() {
        if !live.insert(id) {
            continue;
        }
        if let Some(deps) = pure_def.get(&id) {
            for &d in deps {
                if !live.contains(&d) {
                    work.push(d);
                }
            }
        }
    }

    let mut any = false;
    for f in &mut module.functions {
        for b in &mut f.blocks {
            let before = b.instructions.len();
            b.instructions.retain(|inst| {
                let removable = is_pure(inst.class.opcode)
                    && inst.result_id.is_some_and(|r| !live.contains(&r));
                !removable
            });
            if b.instructions.len() != before {
                any = true;
            }
        }
    }
    // Module-scope dead-constant sweep, restricted to the value producers a dead arm leaves behind:
    // an `OpConstantNull`/`OpUndef` (e.g. the null base of a pruned pointer-induction walk). These
    // are pure and operand-free, so removing an unreferenced one is always safe — and necessary: a
    // pointer-typed `OpConstantNull %_ptr_UniformConstant_*` is itself invalid SPIR-V ("may only
    // return a logical pointer in StorageBuffer/Workgroup"), so leaving the orphan blocks the
    // relooper's structured output from validating even after the dead arm that used it is gone.
    // Types and global `OpVariable`s (interface/descriptors) are never swept.
    let before = module.types_global_values.len();
    module.types_global_values.retain(|inst| {
        let sweepable = matches!(inst.class.opcode, Op::ConstantNull | Op::Undef);
        let dead = inst.result_id.is_some_and(|r| !live.contains(&r));
        !(sweepable && dead)
    });
    if module.types_global_values.len() != before {
        any = true;
    }
    any
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::ModuleHeader;

    #[test]
    fn dce_keeps_only_typed_sidecar_rooted_dead_global() {
        let ulong = 1;
        let sentinel = 2;
        let dead = 3;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(4));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeInt,
                None,
                Some(ulong),
                vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
            ),
            Instruction::new(Op::ConstantNull, Some(ulong), Some(sentinel), vec![]),
            Instruction::new(Op::ConstantNull, Some(ulong), Some(dead), vec![]),
        ];

        assert!(dce_preserving(&mut module, &HashSet::from([sentinel])));

        let ids = module
            .types_global_values
            .iter()
            .filter_map(|inst| inst.result_id)
            .collect::<HashSet<_>>();
        assert!(ids.contains(&sentinel));
        assert!(!ids.contains(&dead));
    }
}
