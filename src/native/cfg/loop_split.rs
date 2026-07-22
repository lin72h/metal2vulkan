//! Post-emit multi-entry-loop split on an EMITTED SPIR-V module: node-split a loop whose header is
//! entered from two different selections' arms, so each entry gets its own single-entry loop.
//!
//! The structurizer-by-construction can ADMIT (its self-checks do not catch it) a function whose
//! emitted CFG has a loop header `R` reached by a forward edge from BOTH an inner selection's arm and
//! an enclosing selection's arm — a MULTI-ENTRY (irreducible) loop. SPIR-V structured control flow
//! requires each loop to be single-entry (the header dominates the whole loop), so spirv-val rejects
//! it, surfacing as *"block X exits the selection headed by Y, but not via a structured exit"* (the
//! inner arm's block escapes its selection into the shared loop). The mlx-steel `steel_attention`
//! kernel family is the canonical case: `if a { …; if b { work } else { →L } } else { →L }` where `L`
//! is a loop — the inner `else` and the outer `else` both enter `L`.
//!
//! The fix duplicates `L`'s loop region for the INNER arm's entry (the "broken" one — whose enclosing
//! selection's merge is NOT the loop's own exit, so its arm cannot reconverge at that merge), routing
//! the clone's exit to that selection's merge. The other entry keeps the original loop. Each loop is
//! then single-entry and each selection's arms reconverge at its own merge. This is exactly the
//! node-splitting the relooper retry already applies to these functions; doing it surgically on the
//! structured CFG makes its PRIMARY output valid instead of shipping only via the relooper retry.
//!
//! Applies ONLY when the whole loop region has NO SSA value used past its single exit boundary (the
//! `steel_attention` loops only store to Workgroup memory), so the clone is self-contained and needs no
//! exit-phi synthesis. Floor-safe by construction: a single-entry (valid) loop has < 2 forward header
//! predecessors so the scan finds nothing and the module is byte-identical; a matched region that is
//! not cleanly cloneable (multiple exits, escaping SSA, a boundary phi over region values) is left in
//! place (that function stays on the floor, no regression). Decides purely from IR structure (loop
//! merges, block edges, dominance), never a shader name.

use super::exit_check::dominator_sets;
use super::graph::{spirv_block_successors_by_label, spirv_predecessor_ids_by_label};
use crate::spirv_module::Module;
use crate::spirv_module::Operand;
use crate::spirv_module::{Block, Function, Instruction};
use spirv::{Op, Word};
use std::collections::{HashMap, HashSet};

/// Cap on cloned region size so a pathological loop cannot blow up emission.
const MAX_REGION_BLOCKS: usize = 96;

/// Split every multi-entry loop whose extra entries are broken selection arms. Returns true if any
/// clone was applied.
pub(in crate::native) fn split_multientry_loop_selection_exits(module: &mut Module) -> bool {
    let mut next_id = module.header.as_ref().map(|h| h.bound).unwrap_or(0);
    let mut fresh = || {
        let id = next_id;
        next_id += 1;
        id
    };
    let mut any = false;
    for function in &mut module.functions {
        any |= split_in_function(function, &mut fresh);
    }
    if any {
        if let Some(header) = module.header.as_mut() {
            header.bound = next_id;
        }
    }
    any
}

/// A planned clone: the loop entry (`arm`) to privatize, the region blocks (labels, in a stable order),
/// the loop's single exit boundary (`mo`), and the selection merge (`ms`) the clone must reconverge at.
struct Plan {
    arm: Word,
    region: Vec<Word>,
    mo: Word,
    ms: Word,
}

fn split_in_function(function: &mut Function, fresh: &mut impl FnMut() -> Word) -> bool {
    let Some(entry) = function
        .blocks
        .first()
        .and_then(|b| b.label.as_ref()?.result_id)
    else {
        return false;
    };
    let labels: Vec<Word> = function
        .blocks
        .iter()
        .filter_map(|b| b.label.as_ref()?.result_id)
        .collect();
    let label_set: HashSet<Word> = labels.iter().copied().collect();
    let successors = spirv_block_successors_by_label(&function.blocks);
    let preds = spirv_predecessor_ids_by_label(&successors);
    let doms = dominator_sets(entry, &labels, &successors);
    let dominates = |a: Word, b: Word| doms.get(&b).is_some_and(|s| s.contains(&a));
    let depth = |n: Word| doms.get(&n).map(|s| s.len()).unwrap_or(0);

    // Loop headers (block with OpLoopMerge) → its merge block.
    let mut loop_merge: HashMap<Word, Word> = HashMap::new();
    // Selection headers (block with OpSelectionMerge whose terminator is a conditional branch, not a
    // switch) → its merge block.
    let mut sel_merge: Vec<(Word, Word)> = Vec::new();
    for block in &function.blocks {
        let Some(h) = block.label.as_ref().and_then(|l| l.result_id) else {
            continue;
        };
        let is_switch = block
            .instructions
            .last()
            .is_some_and(|i| i.class.opcode == Op::Switch);
        for inst in &block.instructions {
            match inst.class.opcode {
                Op::LoopMerge => {
                    if let Some(Operand::IdRef(m)) = inst.operands.first() {
                        loop_merge.insert(h, *m);
                    }
                }
                Op::SelectionMerge if !is_switch => {
                    if let Some(Operand::IdRef(m)) = inst.operands.first() {
                        sel_merge.push((h, *m));
                    }
                }
                _ => {}
            }
        }
    }
    if loop_merge.is_empty() {
        return false;
    }

    // Innermost enclosing selection (header, merge) of a block P: the deepest selection header that
    // dominates P without its merge dominating P.
    let innermost_sel = |p: Word| -> Option<(Word, Word)> {
        sel_merge
            .iter()
            .filter(|(h, m)| *h != p && dominates(*h, p) && !dominates(*m, p))
            .max_by_key(|(h, _)| depth(*h))
            .copied()
    };

    // value id -> region membership is per-loop; collect ALL result ids once for the escaping-SSA check.
    let mut plans: Vec<Plan> = Vec::new();
    for (&r, &rm) in &loop_merge {
        // Forward predecessors of the header: preds NOT dominated by the header (a dominated pred is a
        // back-edge / in-loop latch).
        let fwd: Vec<Word> = preds
            .get(&r)
            .into_iter()
            .flatten()
            .copied()
            .filter(|&p| !dominates(r, p))
            .collect();
        if fwd.len() < 2 {
            continue; // single-entry loop — valid, nothing to split
        }
        // Region = the loop construct + its merge: blocks the header dominates, excluding anything the
        // merge dominates (post-loop tail), plus the merge itself.
        let region: Vec<Word> = labels
            .iter()
            .copied()
            .filter(|&b| dominates(r, b) && (b == rm || !dominates(rm, b)))
            .collect();
        if region.len() > MAX_REGION_BLOCKS {
            continue;
        }
        let region_set: HashSet<Word> = region.iter().copied().collect();
        // Single exit boundary: exactly one block outside the region that the region branches to.
        let mut boundary: HashSet<Word> = HashSet::new();
        for &b in &region {
            for &s in successors.get(&b).into_iter().flatten() {
                if !region_set.contains(&s) {
                    boundary.insert(s);
                }
            }
        }
        if boundary.len() != 1 {
            continue;
        }
        let mo = *boundary.iter().next().unwrap();

        // No SSA value defined in the region may be used outside it (else the clone would need exit-phi
        // synthesis); and the boundary block must have no OpPhi reading a region predecessor.
        if region_escapes(function, &region_set, mo) {
            continue;
        }

        // Partition forward entries: an entry is BROKEN (needs its own loop copy) when its innermost
        // enclosing selection's merge is not the loop's own exit `mo` and is outside the region — its
        // arm cannot reconverge at that merge because it dives into the shared loop instead. An entry
        // whose enclosing selection merges at `mo` (or has none) is already well-formed; it keeps the
        // original loop.
        let mut to_clone: Vec<(Word, Word)> = Vec::new(); // (arm, ms)
        let mut keepers = 0usize;
        for &p in &fwd {
            match innermost_sel(p) {
                Some((_hs, ms)) if ms != mo && !region_set.contains(&ms) => {
                    to_clone.push((p, ms));
                }
                _ => keepers += 1,
            }
        }
        // Need at least one keeper (retains the original loop) and at least one broken arm to fix.
        if keepers == 0 || to_clone.is_empty() {
            continue;
        }
        for (arm, ms) in to_clone {
            plans.push(Plan {
                arm,
                region: region.clone(),
                mo,
                ms,
            });
        }
    }
    if plans.is_empty() {
        return false;
    }

    let _ = label_set;
    let mut any = false;
    for plan in plans {
        if apply_plan(function, &plan, fresh) {
            any = true;
        }
    }
    any
}

/// Map each block's label id to its current index in `function.blocks` (recomputed per plan, since a
/// prior plan's clone insertion shifts indices).
fn index_of(function: &Function) -> HashMap<Word, usize> {
    function
        .blocks
        .iter()
        .enumerate()
        .filter_map(|(i, b)| b.label.as_ref()?.result_id.map(|l| (l, i)))
        .collect()
}

/// True if any SSA value defined in `region` is referenced from a block outside the region, or the
/// exit boundary block `mo` has an OpPhi with an incoming edge from a region block (which would need a
/// mirrored incoming for the clone). Either makes the self-contained clone unsound — bail.
fn region_escapes(function: &Function, region: &HashSet<Word>, mo: Word) -> bool {
    let mut region_defs: HashSet<Word> = HashSet::new();
    for block in &function.blocks {
        let Some(l) = block.label.as_ref().and_then(|i| i.result_id) else {
            continue;
        };
        if !region.contains(&l) {
            continue;
        }
        for inst in &block.instructions {
            if let Some(rid) = inst.result_id {
                region_defs.insert(rid);
            }
        }
    }
    for block in &function.blocks {
        let l = block.label.as_ref().and_then(|i| i.result_id);
        let in_region = l.is_some_and(|l| region.contains(&l));
        if in_region {
            continue;
        }
        // A use of a region-defined value in an out-of-region block escapes.
        for inst in &block.instructions {
            for op in &inst.operands {
                if let Operand::IdRef(v) = op {
                    if region_defs.contains(v) {
                        return true;
                    }
                }
            }
        }
    }
    // Boundary phi over a region predecessor?
    if let Some(block) = function
        .blocks
        .iter()
        .find(|b| b.label.as_ref().and_then(|i| i.result_id) == Some(mo))
    {
        for inst in &block.instructions {
            if inst.class.opcode != Op::Phi {
                continue;
            }
            let mut pos = 1;
            while pos < inst.operands.len() {
                if let Operand::IdRef(pred) = inst.operands[pos] {
                    if region.contains(&pred) {
                        return true;
                    }
                }
                pos += 2;
            }
        }
    }
    false
}

/// Clone `plan.region` for the `plan.arm` entry: fresh ids throughout, the cloned header's phi keeps
/// only the arm + cloned back-edge incomings, the original header's phi drops the arm incoming, the arm
/// is redirected to the clone header, and the clone's exit is routed to `plan.ms`.
fn apply_plan(function: &mut Function, plan: &Plan, fresh: &mut impl FnMut() -> Word) -> bool {
    let idx_of = index_of(function);
    let region_set: HashSet<Word> = plan.region.iter().copied().collect();
    // Re-derive the loop header: the region block carrying the OpLoopMerge.
    let Some(header) = plan.region.iter().copied().find(|&b| {
        function.blocks[idx_of[&b]]
            .instructions
            .iter()
            .any(|i| i.class.opcode == Op::LoopMerge)
    }) else {
        return false;
    };

    // Rename map: every region label + every value defined in the region → a fresh id.
    let mut rename: HashMap<Word, Word> = HashMap::new();
    for &b in &plan.region {
        rename.insert(b, fresh());
        for inst in &function.blocks[idx_of[&b]].instructions {
            if let Some(rid) = inst.result_id {
                rename.insert(rid, fresh());
            }
        }
    }
    let remap = |w: Word| -> Word { rename.get(&w).copied().unwrap_or(w) };

    // Materialize the cloned blocks.
    let mut cloned: Vec<Block> = Vec::with_capacity(plan.region.len());
    for &b in &plan.region {
        let src = &function.blocks[idx_of[&b]];
        let mut out = Block::new();
        out.label = Some(Instruction::new(Op::Label, None, Some(remap(b)), vec![]));
        for inst in &src.instructions {
            let mut ni = inst.clone();
            if let Some(rid) = ni.result_id {
                ni.result_id = Some(remap(rid));
            }
            if b == header && inst.class.opcode == Op::Phi {
                // Cloned header phi: keep only the arm incoming + cloned back-edge incomings.
                ni.operands = clone_header_phi(&inst.operands, plan.arm, &region_set, &remap);
            } else {
                for op in ni.operands.iter_mut() {
                    if let Operand::IdRef(v) = op {
                        *op = Operand::IdRef(remap(*v));
                    }
                }
            }
            out.instructions.push(ni);
        }
        // Route the cloned exit (the block that branched to `mo`, outside the region so un-renamed) to
        // the broken selection's merge `ms` instead.
        redirect_terminator(&mut out, plan.mo, plan.ms);
        cloned.push(out);
    }

    // Original header phi: drop the arm incoming (it now flows into the clone).
    for inst in function.blocks[idx_of[&header]].instructions.iter_mut() {
        if inst.class.opcode == Op::Phi {
            inst.operands = drop_phi_incoming(&inst.operands, plan.arm);
        }
    }
    // Redirect the arm's terminator: arm -> header becomes arm -> header_clone.
    redirect_terminator(
        &mut function.blocks[idx_of[&plan.arm]],
        header,
        remap(header),
    );

    // Splice the cloned blocks in right after the arm block, so they sit inside the arm's region.
    let insert_at = idx_of[&plan.arm] + 1;
    for (k, b) in cloned.into_iter().enumerate() {
        function.blocks.insert(insert_at + k, b);
    }
    true
}

/// Rebuild a header phi's operands for the CLONE: keep the `[val, arm]` incoming as-is (the arm still
/// branches here, into the clone), keep back-edge incomings (pred in region) with both val and pred
/// remapped, and drop every other incoming (the sibling forward entries that keep the original loop).
fn clone_header_phi(
    operands: &[Operand],
    arm: Word,
    region: &HashSet<Word>,
    remap: &impl Fn(Word) -> Word,
) -> Vec<Operand> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + 1 < operands.len() {
        if let (Operand::IdRef(v), Operand::IdRef(pred)) = (&operands[pos], &operands[pos + 1]) {
            if *pred == arm {
                out.push(Operand::IdRef(*v));
                out.push(Operand::IdRef(*pred));
            } else if region.contains(pred) {
                out.push(Operand::IdRef(remap(*v)));
                out.push(Operand::IdRef(remap(*pred)));
            }
        }
        pos += 2;
    }
    out
}

/// Rebuild a phi's operands dropping the incoming whose predecessor is `pred_to_drop`.
fn drop_phi_incoming(operands: &[Operand], pred_to_drop: Word) -> Vec<Operand> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos + 1 < operands.len() {
        if let (Operand::IdRef(_), Operand::IdRef(pred)) = (&operands[pos], &operands[pos + 1]) {
            if *pred != pred_to_drop {
                out.push(operands[pos].clone());
                out.push(operands[pos + 1].clone());
            }
        }
        pos += 2;
    }
    out
}

/// Replace every `IdRef(from)` in a block's TERMINATOR (last instruction) with `IdRef(to)`.
fn redirect_terminator(block: &mut Block, from: Word, to: Word) {
    if let Some(term) = block.instructions.last_mut() {
        for op in term.operands.iter_mut() {
            if *op == Operand::IdRef(from) {
                *op = Operand::IdRef(to);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::ModuleHeader;
    use spirv::LoopControl;

    fn i(op: Op, ty: Option<Word>, res: Option<Word>, ops: Vec<Operand>) -> Instruction {
        Instruction::new(op, ty, res, ops)
    }
    fn b(label: Word, instrs: Vec<Instruction>) -> Block {
        Block {
            label: Some(i(Op::Label, None, Some(label), vec![])),
            instructions: instrs,
        }
    }
    fn r(w: Word) -> Operand {
        Operand::IdRef(w)
    }
    fn base_types() -> Vec<Instruction> {
        vec![
            i(Op::TypeBool, None, Some(1), vec![]),
            i(
                Op::TypeInt,
                None,
                Some(2),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            i(
                Op::Constant,
                Some(2),
                Some(3),
                vec![Operand::LiteralBit32(0)],
            ),
        ]
    }
    fn term_targets(block: &Block) -> Vec<Word> {
        block
            .instructions
            .last()
            .map(|t| {
                t.operands
                    .iter()
                    .filter_map(|o| match o {
                        Operand::IdRef(w) => Some(*w),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
    fn find_block(f: &Function, label: Word) -> &Block {
        f.blocks
            .iter()
            .find(|b| b.label.as_ref().and_then(|l| l.result_id) == Some(label))
            .unwrap()
    }

    /// The `steel_attention` shape: outer selection H0(100, merge 109) then→inner selection H1(101,
    /// merge 105) else→106; H1's else(104) AND H0's else(106) both enter loop header 107 (a MULTI-ENTRY
    /// loop). The split clones the loop for the 104 entry and routes the clone's exit to H1's merge 105.
    fn attention_module() -> Module {
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(300));
        m.types_global_values = base_types();
        let entry = b(
            100,
            vec![
                i(Op::Undef, Some(1), Some(200), vec![]),
                i(Op::Undef, Some(1), Some(201), vec![]),
                i(Op::Undef, Some(1), Some(202), vec![]),
                i(
                    Op::SelectionMerge,
                    None,
                    None,
                    vec![
                        r(109),
                        Operand::SelectionControl(spirv::SelectionControl::NONE),
                    ],
                ),
                i(
                    Op::BranchConditional,
                    None,
                    None,
                    vec![r(200), r(101), r(106)],
                ),
            ],
        );
        let inner_h = b(
            101,
            vec![
                i(
                    Op::SelectionMerge,
                    None,
                    None,
                    vec![
                        r(105),
                        Operand::SelectionControl(spirv::SelectionControl::NONE),
                    ],
                ),
                i(
                    Op::BranchConditional,
                    None,
                    None,
                    vec![r(201), r(102), r(104)],
                ),
            ],
        );
        let inner_then = b(102, vec![i(Op::Branch, None, None, vec![r(105)])]);
        let inner_else = b(104, vec![i(Op::Branch, None, None, vec![r(107)])]);
        let inner_merge = b(105, vec![i(Op::Branch, None, None, vec![r(109)])]);
        let outer_else = b(106, vec![i(Op::Branch, None, None, vec![r(107)])]);
        let loop_h = b(
            107,
            vec![
                i(
                    Op::Phi,
                    Some(2),
                    Some(203),
                    vec![r(3), r(104), r(3), r(106), r(204), r(110)],
                ),
                i(Op::IAdd, Some(2), Some(204), vec![r(203), r(3)]),
                i(
                    Op::LoopMerge,
                    None,
                    None,
                    vec![r(108), r(110), Operand::LoopControl(LoopControl::NONE)],
                ),
                i(
                    Op::BranchConditional,
                    None,
                    None,
                    vec![r(202), r(108), r(110)],
                ),
            ],
        );
        let latch = b(110, vec![i(Op::Branch, None, None, vec![r(107)])]);
        let loop_merge = b(108, vec![i(Op::Branch, None, None, vec![r(109)])]);
        let outer_merge = b(109, vec![i(Op::Return, None, None, vec![])]);
        let mut f = Function::new();
        f.blocks = vec![
            entry,
            inner_h,
            inner_then,
            inner_else,
            inner_merge,
            outer_else,
            loop_h,
            latch,
            loop_merge,
            outer_merge,
        ];
        m.functions = vec![f];
        m
    }

    #[test]
    fn splits_multientry_loop_shared_by_two_selection_arms() {
        let mut m = attention_module();
        assert!(split_multientry_loop_selection_exits(&mut m));
        let f = &m.functions[0];
        // Three region blocks (107,110,108) were cloned.
        assert_eq!(f.blocks.len(), 13);
        // The original loop header 107 now has a single forward entry: its phi dropped the 104 incoming
        // (which now flows into the clone), keeping 106 and the 110 back-edge.
        let phi = find_block(f, 107)
            .instructions
            .iter()
            .find(|x| x.class.opcode == Op::Phi)
            .unwrap();
        let preds: Vec<Word> = phi
            .operands
            .iter()
            .enumerate()
            .filter_map(|(k, o)| match o {
                Operand::IdRef(w) if k % 2 == 1 => Some(*w),
                _ => None,
            })
            .collect();
        assert_eq!(
            preds,
            vec![106, 110],
            "original header keeps only 106 + back-edge"
        );
        // Arm 104 no longer branches to 107 (it targets the clone header instead).
        assert!(!term_targets(find_block(f, 104)).contains(&107));
        // H1's merge 105 now reconverges BOTH arms: the then-arm 102 (original) and the cloned
        // loop-merge (the 104 arm's private loop exit). The original loop-merge 108 still -> 109.
        let to_105 = f
            .blocks
            .iter()
            .filter(|b| term_targets(b).contains(&105))
            .count();
        assert_eq!(to_105, 2);
    }

    #[test]
    fn noop_on_single_entry_loop() {
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(200));
        m.types_global_values = base_types();
        let entry = b(100, vec![i(Op::Branch, None, None, vec![r(101)])]);
        let loop_h = b(
            101,
            vec![
                i(
                    Op::Phi,
                    Some(2),
                    Some(120),
                    vec![r(3), r(100), r(3), r(102)],
                ),
                i(
                    Op::LoopMerge,
                    None,
                    None,
                    vec![r(103), r(102), Operand::LoopControl(LoopControl::NONE)],
                ),
                i(
                    Op::BranchConditional,
                    None,
                    None,
                    vec![r(120), r(103), r(102)],
                ),
            ],
        );
        let latch = b(102, vec![i(Op::Branch, None, None, vec![r(101)])]);
        let merge = b(103, vec![i(Op::Return, None, None, vec![])]);
        let mut f = Function::new();
        f.blocks = vec![entry, loop_h, latch, merge];
        m.functions = vec![f];
        assert!(!split_multientry_loop_selection_exits(&mut m));
        assert_eq!(m.functions[0].blocks.len(), 4);
    }

    #[test]
    fn noop_when_selection_merge_is_the_loop_header() {
        // A VALID shape: both arms of a selection reconverge AT the loop header (Ms == loop header,
        // which is inside the loop region) — must NOT be split (the region-contains-ms guard).
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(200));
        m.types_global_values = base_types();
        let entry = b(
            100,
            vec![
                i(Op::Undef, Some(1), Some(200), vec![]),
                i(
                    Op::SelectionMerge,
                    None,
                    None,
                    vec![
                        r(101),
                        Operand::SelectionControl(spirv::SelectionControl::NONE),
                    ],
                ),
                i(
                    Op::BranchConditional,
                    None,
                    None,
                    vec![r(200), r(104), r(105)],
                ),
            ],
        );
        let arm_a = b(104, vec![i(Op::Branch, None, None, vec![r(101)])]);
        let arm_b = b(105, vec![i(Op::Branch, None, None, vec![r(101)])]);
        let loop_h = b(
            101,
            vec![
                i(
                    Op::Phi,
                    Some(2),
                    Some(120),
                    vec![r(3), r(104), r(3), r(105), r(3), r(102)],
                ),
                i(
                    Op::LoopMerge,
                    None,
                    None,
                    vec![r(103), r(102), Operand::LoopControl(LoopControl::NONE)],
                ),
                i(
                    Op::BranchConditional,
                    None,
                    None,
                    vec![r(120), r(103), r(102)],
                ),
            ],
        );
        let latch = b(102, vec![i(Op::Branch, None, None, vec![r(101)])]);
        let merge = b(103, vec![i(Op::Return, None, None, vec![])]);
        let mut f = Function::new();
        f.blocks = vec![entry, arm_a, arm_b, loop_h, latch, merge];
        m.functions = vec![f];
        assert!(!split_multientry_loop_selection_exits(&mut m));
        assert_eq!(m.functions[0].blocks.len(), 6);
    }
}
