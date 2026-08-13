//! Loop-closed-SSA repair on an EMITTED SPIR-V module: register-demote a value whose defining block
//! no longer dominates a use.
//!
//! The structurizer-by-construction reorders/restructures blocks to satisfy the structured-CFG rules.
//! A `MultipleExits` loop is funnelled through a synthesized dispatch merge (`synth_multi_exit_merge`),
//! and a loop-body value consumed after the loop (the classic loop-closed-SSA / LCSSA shape) can end up
//! referenced RAW in a post-loop block that the restructured CFG reaches by a path skipping the
//! defining block — so the def no longer dominates the use and spirv-val rejects: *"ID X defined in
//! block B does not dominate its use in block U"*. The AIR was strict SSA (the def dominated the use
//! through a proper LCSSA phi), and the emitted CFG is semantics-equivalent to the AIR — whenever
//! control reaches the use with that value live, the defining block ran first — so spilling the value
//! to a function-scope `OpVariable` (stored right after its definition, loaded before each
//! non-dominated use) preserves semantics while making the module structurally valid. This is exactly
//! the register-demotion the relooper retry already applies to these functions (via its big-switch
//! form); doing it SURGICALLY on the structured CFG keeps the structured emit and makes its PRIMARY
//! output valid, instead of shipping only via the relooper retry.
//!
//! Floor-safe by construction: a module that already validates has every def dominating its uses, so
//! the violation scan finds nothing and the module is returned byte-identical. Only a value whose type
//! is spillable to Function memory (not a pointer under Logical addressing, not an opaque
//! image/sampler) is demoted; any other non-dominating value is left in place (that function stays on
//! the floor, no regression). Decides purely from IR structure (block edges, dominance, value
//! def/use), never a shader name.

use super::graph::spirv_block_successors_by_label;
use super::EmittedDominators;
use crate::spirv_module::Operand;
use crate::spirv_module::{Function, Instruction, Module};
use spirv::{Op, StorageClass, Word};
use std::collections::{HashMap, HashSet};

/// Register-demote every value in `module` whose defining block does not dominate one of its uses.
/// Returns true if any value was demoted.
pub(in crate::native) fn demote_nondominating_values(module: &mut Module) -> bool {
    // Type kinds we can back with a Function `OpVariable` (a concrete data type). A pointer (illegal to
    // store under Logical addressing) or an opaque handle (image/sampler) is NOT spillable.
    let type_kind: HashMap<Word, Op> = module
        .types_global_values
        .iter()
        .filter_map(|i| i.result_id.map(|id| (id, i.class.opcode)))
        .collect();
    let spillable = |ty: Word| -> bool {
        matches!(
            type_kind.get(&ty),
            Some(
                Op::TypeInt
                    | Op::TypeFloat
                    | Op::TypeBool
                    | Op::TypeVector
                    | Op::TypeMatrix
                    | Op::TypeArray
                    | Op::TypeStruct
            )
        )
    };
    // Existing Function pointer types by pointee, so a demotion reuses one rather than emitting a
    // duplicate (SPIR-V forbids two identical OpTypePointer).
    let mut fn_ptr_type: HashMap<Word, Word> = HashMap::new();
    for i in &module.types_global_values {
        if i.class.opcode == Op::TypePointer {
            if let (
                Some(rid),
                Some(Operand::StorageClass(StorageClass::Function)),
                Some(Operand::IdRef(pointee)),
            ) = (i.result_id, i.operands.first(), i.operands.get(1))
            {
                fn_ptr_type.entry(*pointee).or_insert(rid);
            }
        }
    }

    let mut next_id = module.header.as_ref().map(|h| h.bound).unwrap_or(0);
    let mut fresh = || {
        let id = next_id;
        next_id += 1;
        id
    };

    let mut new_ptr_types: Vec<Instruction> = Vec::new();
    let mut any = false;
    for function in &mut module.functions {
        any |= demote_in_function(
            function,
            &spillable,
            &mut fn_ptr_type,
            &mut new_ptr_types,
            &mut fresh,
        );
    }

    if any {
        // Splice any freshly-created Function pointer types into the global type section (order among
        // type defs is not load-bearing; id-canonicalization re-sorts afterward).
        module.types_global_values.extend(new_ptr_types);
        if let Some(header) = module.header.as_mut() {
            header.bound = next_id;
        }
    }
    any
}

/// A single value to demote: its result id, its (value) type, and the block index of its definition.
struct Target {
    value: Word,
    ty: Word,
    def_block: usize,
    /// Blocks in which a load must be materialized (a non-dominated non-phi use lives here, or an
    /// incoming phi edge from here carries the value).
    load_blocks: Vec<usize>,
}

fn demote_in_function(
    function: &mut Function,
    spillable: &impl Fn(Word) -> bool,
    fn_ptr_type: &mut HashMap<Word, Word>,
    new_ptr_types: &mut Vec<Instruction>,
    fresh: &mut impl FnMut() -> Word,
) -> bool {
    if function.blocks.len() < 2 {
        return false;
    }
    let Some(entry) = function
        .blocks
        .first()
        .and_then(|b| b.label.as_ref()?.result_id)
    else {
        return false;
    };
    // block index -> label id, label id -> block index, and the reachable label list.
    let idx_to_label: Vec<Option<Word>> = function
        .blocks
        .iter()
        .map(|b| b.label.as_ref().and_then(|l| l.result_id))
        .collect();
    let mut label_to_idx: HashMap<Word, usize> = HashMap::new();
    let mut labels: Vec<Word> = Vec::new();
    for (bi, l) in idx_to_label.iter().enumerate() {
        if let Some(rid) = l {
            label_to_idx.insert(*rid, bi);
            labels.push(*rid);
        }
    }
    let successors = spirv_block_successors_by_label(&function.blocks);
    let dominators = EmittedDominators::new(entry, &labels, &successors);
    // A block whose dominators were not computed (unreachable) dominates nothing reachable and is
    // dominated by nothing — treat as non-dominating so its dead uses are left alone.
    let dominates = |a_label: Word, b_label: Word| dominators.dominates(a_label, b_label);

    // value id -> (def block index, result type), for every value defined in the function body.
    let mut def_block: HashMap<Word, usize> = HashMap::new();
    let mut value_type: HashMap<Word, Word> = HashMap::new();
    for (bi, block) in function.blocks.iter().enumerate() {
        for inst in &block.instructions {
            if let Some(rid) = inst.result_id {
                def_block.insert(rid, bi);
                if let Some(rty) = inst.result_type {
                    value_type.insert(rid, rty);
                }
            }
        }
    }

    // Plan phase (immutable scan): collect each value whose def block does not dominate a use, together
    // with the set of blocks that must load it.
    let mut targets: HashMap<Word, HashSet<usize>> = HashMap::new();
    for (bi, block) in function.blocks.iter().enumerate() {
        let this_label = idx_to_label[bi];
        for inst in &block.instructions {
            if inst.class.opcode == Op::Phi {
                // Operands are [v0, b0, v1, b1, ...]: value v_i flows in on the edge from predecessor
                // b_i, so v_i must dominate b_i (the predecessor's terminator), NOT the phi's block.
                let mut pos = 0;
                while pos + 1 < inst.operands.len() {
                    if let (Operand::IdRef(v), Operand::IdRef(pred)) =
                        (&inst.operands[pos], &inst.operands[pos + 1])
                    {
                        if let Some(&db) = def_block.get(v) {
                            if let (Some(db_label), Some(&pred_idx)) =
                                (idx_to_label[db], label_to_idx.get(pred))
                            {
                                if !dominates(db_label, *pred) {
                                    targets.entry(*v).or_default().insert(pred_idx);
                                }
                            }
                        }
                    }
                    pos += 2;
                }
            } else {
                // A non-phi instruction's operands must be dominated by this block.
                let Some(this) = this_label else { continue };
                for op in &inst.operands {
                    if let Operand::IdRef(v) = op {
                        if let Some(&db) = def_block.get(v) {
                            if let Some(db_label) = idx_to_label[db] {
                                if !dominates(db_label, this) {
                                    targets.entry(*v).or_default().insert(bi);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if targets.is_empty() {
        return false;
    }

    // Keep only spillable targets; a non-spillable non-dominating value (a pointer / opaque) is left in
    // place — the function simply stays invalid (relooper retry still rescues it), no regression.
    let mut plans: Vec<Target> = Vec::new();
    for (value, load_blocks) in targets {
        let Some(&ty) = value_type.get(&value) else {
            continue;
        };
        if !spillable(ty) {
            continue;
        }
        let Some(&def_b) = def_block.get(&value) else {
            continue;
        };
        let mut load_blocks: Vec<usize> = load_blocks.into_iter().collect();
        load_blocks.sort_unstable();
        plans.push(Target {
            value,
            ty,
            def_block: def_b,
            load_blocks,
        });
    }
    if plans.is_empty() {
        return false;
    }
    // Deterministic order (id ascending) so the pass is reproducible independent of HashMap iteration.
    plans.sort_by_key(|t| t.value);

    // Apply phase. Materialize a Function OpVariable per demoted value, store after its def, and load
    // (once per block) before each non-dominated use, repointing those uses to the load.
    for plan in &plans {
        let ptr_ty = *fn_ptr_type.entry(plan.ty).or_insert_with(|| {
            let id = fresh();
            new_ptr_types.push(Instruction::new(
                Op::TypePointer,
                None,
                Some(id),
                vec![
                    Operand::StorageClass(StorageClass::Function),
                    Operand::IdRef(plan.ty),
                ],
            ));
            id
        });
        let var_id = fresh();

        // 1) OpVariable at the top of the entry block (all Function variables must lead the first
        //    block). Insert after any existing leading variables so they stay contiguous.
        {
            let entry_block = &mut function.blocks[0];
            let after_vars = entry_block
                .instructions
                .iter()
                .position(|i| i.class.opcode != Op::Variable)
                .unwrap_or(0);
            entry_block.instructions.insert(
                after_vars,
                Instruction::new(
                    Op::Variable,
                    Some(ptr_ty),
                    Some(var_id),
                    vec![Operand::StorageClass(StorageClass::Function)],
                ),
            );
        }

        // 2) OpStore right after the definition (but after any leading phis, and before a trailing
        //    structured-merge instruction).
        {
            let block = &mut function.blocks[plan.def_block];
            let def_idx = block
                .instructions
                .iter()
                .position(|i| i.result_id == Some(plan.value))
                .expect("def instruction present");
            let first_non_phi = block
                .instructions
                .iter()
                .position(|i| i.class.opcode != Op::Phi)
                .unwrap_or(block.instructions.len());
            let mut at = (def_idx + 1).max(first_non_phi);
            let n = block.instructions.len();
            // Keep a trailing OpSelectionMerge/OpLoopMerge adjacent to its branch terminator.
            let safe_max = if n >= 2
                && matches!(
                    block.instructions[n - 2].class.opcode,
                    Op::SelectionMerge | Op::LoopMerge
                ) {
                n - 2
            } else {
                n - 1
            };
            at = at.min(safe_max);
            block.instructions.insert(
                at,
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(var_id), Operand::IdRef(plan.value)],
                ),
            );
        }

        // 3) One OpLoad at the top of each load block (after its phis), repointing that block's uses.
        for &lb in &plan.load_blocks {
            let load_id = fresh();
            let lb_label = idx_to_label[lb];
            {
                let block = &mut function.blocks[lb];
                let after_phis = block
                    .instructions
                    .iter()
                    .position(|i| i.class.opcode != Op::Phi && i.class.opcode != Op::Variable)
                    .unwrap_or(0);
                block.instructions.insert(
                    after_phis,
                    Instruction::new(
                        Op::Load,
                        Some(plan.ty),
                        Some(load_id),
                        vec![Operand::IdRef(var_id)],
                    ),
                );
                // Repoint non-phi uses of `value` in THIS block to the load (it precedes them). Never
                // touch the store we just inserted (its value operand must stay the original def), and
                // never the load itself.
                for inst in block.instructions.iter_mut() {
                    if inst.class.opcode == Op::Phi
                        || inst.class.opcode == Op::Variable
                        || inst.result_id == Some(load_id)
                    {
                        continue;
                    }
                    if inst.class.opcode == Op::Store
                        && inst.operands.first() == Some(&Operand::IdRef(var_id))
                    {
                        continue;
                    }
                    for op in inst.operands.iter_mut() {
                        if *op == Operand::IdRef(plan.value) {
                            *op = Operand::IdRef(load_id);
                        }
                    }
                }
            }
            // Repoint phi operands ANYWHERE whose predecessor label is this block: the load in this
            // predecessor dominates the outgoing edge, so the phi may read it.
            if let Some(lb_label) = lb_label {
                for other in function.blocks.iter_mut() {
                    for inst in other.instructions.iter_mut() {
                        if inst.class.opcode != Op::Phi {
                            continue;
                        }
                        let mut pos = 0;
                        while pos + 1 < inst.operands.len() {
                            if inst.operands[pos] == Operand::IdRef(plan.value)
                                && inst.operands[pos + 1] == Operand::IdRef(lb_label)
                            {
                                inst.operands[pos] = Operand::IdRef(load_id);
                            }
                            pos += 2;
                        }
                    }
                }
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::{Block, ModuleHeader};

    fn inst(op: Op, ty: Option<Word>, res: Option<Word>, ops: Vec<Operand>) -> Instruction {
        Instruction::new(op, ty, res, ops)
    }
    fn block(label: Word, instrs: Vec<Instruction>) -> Block {
        Block {
            label: Some(inst(Op::Label, None, Some(label), vec![])),
            instructions: instrs,
        }
    }

    // A value defined in a loop BODY (block 12) and used after the loop at the merge (block 14) — the
    // loop header can early-exit straight to 14, so 12 does not dominate 14. The value is
    // register-demoted: an OpVariable in the entry, an OpStore after the def, an OpLoad at the merge,
    // and the post-loop use is repointed to the load.
    #[test]
    fn demotes_loop_body_value_used_after_early_exit_loop() {
        // types: bool=1 float=2   (Function-float pointer type is synthesized by the pass)
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(60));
        m.types_global_values = vec![
            inst(Op::TypeBool, None, Some(1), vec![]),
            inst(
                Op::TypeFloat,
                None,
                Some(2),
                vec![Operand::LiteralBit32(32)],
            ),
        ];
        // entry 10 -> header 11
        let entry = block(
            10,
            vec![inst(Op::Branch, None, None, vec![Operand::IdRef(11)])],
        );
        // header 11: cond = undef; loopmerge 14 13; branchconditional cond -> body 12 / merge 14
        let header = block(
            11,
            vec![
                inst(Op::Undef, Some(1), Some(20), vec![]),
                inst(
                    Op::LoopMerge,
                    None,
                    None,
                    vec![
                        Operand::IdRef(14),
                        Operand::IdRef(13),
                        Operand::LoopControl(spirv::LoopControl::NONE),
                    ],
                ),
                inst(
                    Op::BranchConditional,
                    None,
                    None,
                    vec![Operand::IdRef(20), Operand::IdRef(12), Operand::IdRef(14)],
                ),
            ],
        );
        // body 12: %30 = fadd float (the loop-body value); branch to latch 13
        let body = block(
            12,
            vec![
                inst(
                    Op::FAdd,
                    Some(2),
                    Some(30),
                    vec![Operand::IdRef(2), Operand::IdRef(2)],
                ),
                inst(Op::Branch, None, None, vec![Operand::IdRef(13)]),
            ],
        );
        // latch 13 -> header 11 (back edge)
        let latch = block(
            13,
            vec![inst(Op::Branch, None, None, vec![Operand::IdRef(11)])],
        );
        // merge 14: %40 = fsub float %30 %30 (uses the loop-body value); return
        let merge = block(
            14,
            vec![
                inst(
                    Op::FSub,
                    Some(2),
                    Some(40),
                    vec![Operand::IdRef(30), Operand::IdRef(30)],
                ),
                inst(Op::Return, None, None, vec![]),
            ],
        );
        let mut func = Function::new();
        func.blocks = vec![entry, header, body, latch, merge];
        m.functions = vec![func];

        assert!(demote_nondominating_values(&mut m));

        // A Function pointer-to-float type was synthesized.
        let ptr_ty = m
            .types_global_values
            .iter()
            .find(|i| {
                i.class.opcode == Op::TypePointer
                    && i.operands.first() == Some(&Operand::StorageClass(StorageClass::Function))
                    && i.operands.get(1) == Some(&Operand::IdRef(2))
            })
            .and_then(|i| i.result_id)
            .expect("Function float pointer type synthesized");

        let f = &m.functions[0];
        // Entry block holds the demotion OpVariable of that pointer type.
        let var = f.blocks[0]
            .instructions
            .iter()
            .find(|i| i.class.opcode == Op::Variable)
            .expect("OpVariable in entry");
        assert_eq!(var.result_type, Some(ptr_ty));
        let var_id = var.result_id.unwrap();

        // The body (block index 2) stores the value after its def, before the terminator.
        let body = &f.blocks[2];
        let store_pos = body
            .instructions
            .iter()
            .position(|i| {
                i.class.opcode == Op::Store
                    && i.operands == vec![Operand::IdRef(var_id), Operand::IdRef(30)]
            })
            .expect("store of the loop-body value after its def");
        let def_pos = body
            .instructions
            .iter()
            .position(|i| i.result_id == Some(30))
            .unwrap();
        assert!(store_pos > def_pos);
        assert_eq!(body.instructions.last().unwrap().class.opcode, Op::Branch);

        // The merge (block index 4) loads from the var and the fsub now reads the load, not %30.
        let merge = &f.blocks[4];
        let load = merge
            .instructions
            .iter()
            .find(|i| i.class.opcode == Op::Load && i.operands == vec![Operand::IdRef(var_id)])
            .expect("load at the merge");
        let load_id = load.result_id.unwrap();
        let fsub = merge
            .instructions
            .iter()
            .find(|i| i.class.opcode == Op::FSub)
            .unwrap();
        assert_eq!(
            fsub.operands,
            vec![Operand::IdRef(load_id), Operand::IdRef(load_id)]
        );
    }

    // A module whose every def already dominates its uses is returned byte-identical — the floor-safety
    // property (no violation => no-op).
    #[test]
    fn valid_module_is_untouched() {
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(30));
        m.types_global_values = vec![inst(
            Op::TypeFloat,
            None,
            Some(2),
            vec![Operand::LiteralBit32(32)],
        )];
        // entry defines %10, single successor uses it (entry dominates all).
        let entry = block(
            1,
            vec![
                inst(
                    Op::FAdd,
                    Some(2),
                    Some(10),
                    vec![Operand::IdRef(2), Operand::IdRef(2)],
                ),
                inst(Op::Branch, None, None, vec![Operand::IdRef(2)]),
            ],
        );
        let tail = block(
            2,
            vec![
                inst(
                    Op::FSub,
                    Some(2),
                    Some(11),
                    vec![Operand::IdRef(10), Operand::IdRef(10)],
                ),
                inst(Op::Return, None, None, vec![]),
            ],
        );
        let mut func = Function::new();
        func.blocks = vec![entry, tail];
        m.functions = vec![func];
        let before = m.clone();

        assert!(!demote_nondominating_values(&mut m));
        assert_eq!(
            m.functions[0].blocks.len(),
            before.functions[0].blocks.len()
        );
        assert_eq!(
            m.functions[0].blocks[1].instructions[0].operands,
            vec![Operand::IdRef(10), Operand::IdRef(10)]
        );
    }
}
