//! Structured-CFG merge, continue, and phi repair over the retained SPIR-V module.

use super::*;
use crate::passes::spirv_cfg::{block_successors, block_successors_by_label};

/// Repair structured-CFG merge placement: `OpSelectionMerge`/`OpLoopMerge` must be the
/// second-to-last instruction in its block (immediately before the block's branch terminator). An
/// emitted block can contain a value computation (for example an `OpFNegate` for a select arm)
/// between the merge and conditional branch, leaving the merge mid-block (spirv-val rejects it).
/// The merge declaration has no result and only references label ids, so sliding it down to sit just
/// before the terminator is always semantics-preserving. We move it; the displaced instructions keep
/// their relative order and stay above it.
pub(in crate::passes) fn fix_merge_placement(ctx: &mut Ctx, entry_idx: usize) {
    let n_blocks = ctx.module.functions[entry_idx].blocks.len();
    for bi in 0..n_blocks {
        let insts = &mut ctx.module.functions[entry_idx].blocks[bi].instructions;
        let n = insts.len();
        if n < 2 {
            continue;
        }
        // Find a merge instruction that is NOT already at index n-2.
        let merge_pos = insts
            .iter()
            .position(|i| matches!(i.class.opcode, Op::SelectionMerge | Op::LoopMerge));
        if let Some(pos) = merge_pos {
            if pos != n - 2 {
                let merge = insts.remove(pos);
                // After removal the block has n-1 instructions; insert so the merge becomes the
                // second-to-last (just before the terminator at the new end).
                insts.insert(insts.len() - 1, merge);
            }
        }
    }
}

/// Give every structured header its own merge block when the emitter reused one natural
/// post-dominator for multiple constructs.
///
/// Rebuilding a multi-thousand-block function with the general relooper is unnecessary for this
/// local defect. Process inner (smaller dominated region) headers first. For each claim, redirect
/// only edges dominated by that header through a fresh pass-through merge. Phi nodes on the shared
/// target are split across the new edge, preserving their exact incoming values. Processing an
/// enclosing header afterward naturally redirects the inner pass-through into the enclosing one.
pub(in crate::passes) fn split_reused_merge_targets(ctx: &mut Ctx, entry_idx: usize) {
    let blocks = &ctx.module.functions[entry_idx].blocks;
    let dominators = CompactDominators::new(blocks);
    let mut claims = HashMap::<Word, Vec<Word>>::new();
    for block in blocks {
        let Some(header) = block.label.as_ref().and_then(|label| label.result_id) else {
            continue;
        };
        for instruction in &block.instructions {
            if !matches!(instruction.class.opcode, Op::SelectionMerge | Op::LoopMerge) {
                continue;
            }
            if let Some(target) = instruction.operands.first().and_then(id_ref_operand) {
                claims.entry(target).or_default().push(header);
            }
        }
    }

    for (target, mut headers) in claims {
        if headers.len() < 2 {
            continue;
        }
        // Dominated-region discovery walks the emitted CFG twice. Cache it once per header instead
        // of rebuilding successor/reachability maps for every comparison in the sort; large generated
        // functions can have hundreds of headers claiming one merge.
        headers.sort_by_cached_key(|header| dominators.dominated_count(*header));
        let mut synthetic_owners = HashMap::new();
        for header in headers {
            if let Some(synthetic) = split_reused_merge_claim_with_dominators(
                ctx,
                entry_idx,
                header,
                target,
                &dominators,
                &synthetic_owners,
            ) {
                synthetic_owners.insert(synthetic, header);
            }
        }
    }
}

/// Give an inner construct a private merge when it currently merges at an enclosing loop's
/// continue target.
///
/// A continue block belongs to the enclosing loop's continue construct, not its ordinary loop
/// construct. Therefore a nested header contained in the loop cannot use that block as its merge.
/// The same dominated-edge and phi-aware split used for duplicate merge claims creates the exact
/// pass-through boundary the inner construct needs while leaving the enclosing continue unchanged.
pub(in crate::passes) fn split_merges_that_are_enclosing_continues(
    ctx: &mut Ctx,
    entry_idx: usize,
) {
    let blocks = &ctx.module.functions[entry_idx].blocks;
    let continue_targets = blocks
        .iter()
        .filter_map(|block| {
            let header = block.label.as_ref()?.result_id?;
            let continue_target = block
                .instructions
                .iter()
                .find(|instruction| instruction.class.opcode == Op::LoopMerge)?
                .operands
                .get(1)
                .and_then(id_ref_operand)?;
            Some((continue_target, header))
        })
        .collect::<Vec<_>>();
    let conflicting_claims = blocks
        .iter()
        .filter_map(|block| {
            let header = block.label.as_ref()?.result_id?;
            let merge = block
                .instructions
                .iter()
                .find(|instruction| {
                    matches!(instruction.class.opcode, Op::SelectionMerge | Op::LoopMerge)
                })?
                .operands
                .first()
                .and_then(id_ref_operand)?;
            continue_targets
                .iter()
                .any(|(continue_target, owner)| *continue_target == merge && *owner != header)
                .then_some((header, merge))
        })
        .collect::<Vec<_>>();
    for (header, merge) in conflicting_claims {
        split_reused_merge_claim(ctx, entry_idx, header, merge);
    }
}

/// Give a loop or selection a private merge when its chosen exit is also reachable from outside the
/// construct. A structured header must dominate its declared merge; route only header-dominated
/// incoming edges through the same phi-aware pass-through used for overlapping merge ownership.
pub(in crate::passes) fn privatize_nondominated_construct_merges(ctx: &mut Ctx, entry_idx: usize) {
    loop {
        let blocks = &ctx.module.functions[entry_idx].blocks;
        let dominators = CompactDominators::new(blocks);
        let polluted = blocks.iter().find_map(|block| {
            let header = block.label.as_ref()?.result_id?;
            let merge = block
                .instructions
                .iter()
                .find(|instruction| {
                    matches!(instruction.class.opcode, Op::LoopMerge | Op::SelectionMerge)
                })?
                .operands
                .first()
                .and_then(id_ref_operand)?;
            (!dominators.dominates(header, merge)).then_some((header, merge))
        });
        let Some((header, merge)) = polluted else {
            break;
        };
        if split_reused_merge_claim_with_dominators(
            ctx,
            entry_idx,
            header,
            merge,
            &dominators,
            &HashMap::new(),
        )
        .is_none()
        {
            break;
        }
    }
}

/// Downgrade a stale loop declaration after earlier rewrites remove every natural back edge to its
/// header. SPIR-V requires exactly one back-edge block for each `OpLoopMerge`; retaining the marker on
/// an acyclic conditional is invalid even though its former merge target is still the correct
/// reconvergence. Preserve that target as `OpSelectionMerge` for a multi-way terminator, or remove the
/// declaration for an unconditional/terminal block. The decision is graph-structural: a predecessor is
/// a back edge exactly when the header dominates it.
pub(in crate::passes) fn downgrade_stale_loop_merges(ctx: &mut Ctx, entry_idx: usize) -> bool {
    let blocks = &ctx.module.functions[entry_idx].blocks;
    let predecessors = block_predecessors(blocks);
    let dominators = CompactDominators::new(blocks);
    let mut repairs = Vec::new();
    for (block_index, block) in blocks.iter().enumerate() {
        let Some(header) = block.label.as_ref().and_then(|label| label.result_id) else {
            continue;
        };
        let Some(merge_index) = block
            .instructions
            .iter()
            .position(|instruction| instruction.class.opcode == Op::LoopMerge)
        else {
            continue;
        };
        let has_back_edge = predecessors
            .get(&header)
            .into_iter()
            .flatten()
            .any(|predecessor| dominators.dominates(header, *predecessor));
        if has_back_edge {
            continue;
        }
        let multi_way = block.instructions.last().is_some_and(|terminator| {
            matches!(terminator.class.opcode, Op::BranchConditional | Op::Switch)
        });
        repairs.push((block_index, merge_index, multi_way));
    }

    for (block_index, merge_index, multi_way) in &repairs {
        let instructions = &mut ctx.module.functions[entry_idx].blocks[*block_index].instructions;
        if *multi_way {
            let merge_target = instructions[*merge_index]
                .operands
                .first()
                .cloned()
                .expect("OpLoopMerge has merge target");
            instructions[*merge_index] = Instruction::new(
                Op::SelectionMerge,
                None,
                None,
                vec![
                    merge_target,
                    Operand::SelectionControl(spirv::SelectionControl::NONE),
                ],
            );
        } else {
            instructions.remove(*merge_index);
        }
    }
    !repairs.is_empty()
}

/// Funnel a selection arm that bypasses the selection's declared merge into that merge when the
/// merge is already a direct pass-through to the arm's target.
///
/// This is the emitted form of a nested selection escaping straight to an enclosing selection merge:
/// `H -> { A -> M -> T, B -> T }`, where `H` declares `M`. SPIR-V requires `B` to leave through `M`.
/// When `M` and `T` carry phis, preserve the exact edge values by extending each leading phi in `M`
/// with `T`'s former `B` incoming, then remove the redundant `B` incoming at `T`. The rewrite is
/// transactional and declines unless every merge phi is represented by a corresponding `T` incoming;
/// no value is guessed and no non-pass-through merge is changed.
pub(in crate::passes) fn funnel_selection_merge_bypasses(ctx: &mut Ctx, entry_idx: usize) -> bool {
    let mut any = false;
    loop {
        let blocks = &ctx.module.functions[entry_idx].blocks;
        let predecessors = block_predecessors(blocks);
        let dominators = CompactDominators::new(blocks);
        let mut plan = None;

        for header in blocks {
            let Some(header_label) = header.label.as_ref().and_then(|label| label.result_id) else {
                continue;
            };
            let Some(merge_label) = header
                .instructions
                .iter()
                .find(|instruction| instruction.class.opcode == Op::SelectionMerge)
                .and_then(|instruction| instruction.operands.first())
                .and_then(id_ref_operand)
            else {
                continue;
            };
            let Some(merge_idx) = block_index_by_label(blocks, merge_label) else {
                continue;
            };
            let Some(target_label) = phi_prefix_branch_target(&blocks[merge_idx]) else {
                continue;
            };
            if !dominators.dominates(header_label, merge_label)
                || dominators.dominates(header_label, target_label)
            {
                continue;
            }
            let merge_predecessors = predecessors
                .get(&merge_label)
                .into_iter()
                .flatten()
                .copied()
                .collect::<HashSet<_>>();
            let bypasses = predecessors
                .get(&target_label)
                .into_iter()
                .flatten()
                .copied()
                .filter(|predecessor| {
                    *predecessor != merge_label
                        && !merge_predecessors.contains(predecessor)
                        && dominators.dominates(header_label, *predecessor)
                })
                .collect::<HashSet<_>>();
            if bypasses.is_empty() {
                continue;
            }
            let Some(target_idx) = block_index_by_label(blocks, target_label) else {
                continue;
            };
            let merge_phi_by_result = blocks[merge_idx]
                .instructions
                .iter()
                .enumerate()
                .take_while(|(_, instruction)| instruction.class.opcode == Op::Phi)
                .filter_map(|(index, instruction)| Some((instruction.result_id?, index)))
                .collect::<HashMap<_, _>>();
            let mut mapped_merge_phis = HashSet::new();
            let mut merge_phi_additions = Vec::new();
            let mut target_phi_updates = Vec::new();
            let mut valid = true;
            for (target_phi_idx, instruction) in blocks[target_idx]
                .instructions
                .iter()
                .enumerate()
                .take_while(|(_, instruction)| instruction.class.opcode == Op::Phi)
            {
                let incoming = phi_incoming(instruction);
                let Some((merge_value, _)) = incoming
                    .iter()
                    .find(|(_, predecessor)| *predecessor == merge_label)
                else {
                    valid = false;
                    break;
                };
                let bypass_incoming = incoming
                    .iter()
                    .filter(|(_, predecessor)| bypasses.contains(predecessor))
                    .cloned()
                    .collect::<Vec<_>>();
                if bypass_incoming.len() != bypasses.len() {
                    valid = false;
                    break;
                }
                if let Operand::IdRef(result) = merge_value {
                    if let Some(&merge_phi_idx) = merge_phi_by_result.get(result) {
                        if !mapped_merge_phis.insert(merge_phi_idx) {
                            valid = false;
                            break;
                        }
                        merge_phi_additions.push((merge_phi_idx, bypass_incoming.clone()));
                    } else if bypass_incoming
                        .iter()
                        .any(|(value, _)| value != merge_value)
                    {
                        valid = false;
                        break;
                    }
                } else if bypass_incoming
                    .iter()
                    .any(|(value, _)| value != merge_value)
                {
                    valid = false;
                    break;
                }
                target_phi_updates.push((
                    target_phi_idx,
                    instruction
                        .operands
                        .chunks(2)
                        .filter(|pair| {
                            !matches!(pair, [_, Operand::IdRef(pred)] if bypasses.contains(pred))
                        })
                        .flatten()
                        .cloned()
                        .collect::<Vec<_>>(),
                ));
            }
            if !valid || mapped_merge_phis.len() != merge_phi_by_result.len() {
                continue;
            }
            plan = Some((
                merge_idx,
                target_idx,
                merge_label,
                target_label,
                bypasses,
                merge_phi_additions,
                target_phi_updates,
            ));
            break;
        }

        let Some((
            merge_idx,
            target_idx,
            merge_label,
            target_label,
            bypasses,
            merge_phi_additions,
            target_phi_updates,
        )) = plan
        else {
            break;
        };
        let blocks = &mut ctx.module.functions[entry_idx].blocks;
        for predecessor in &bypasses {
            let Some(predecessor_idx) = block_index_by_label(blocks, *predecessor) else {
                continue;
            };
            redirect_terminator_target(&mut blocks[predecessor_idx], target_label, merge_label);
        }
        for (instruction_idx, additions) in merge_phi_additions {
            for (value, predecessor) in additions {
                blocks[merge_idx].instructions[instruction_idx]
                    .operands
                    .push(value);
                blocks[merge_idx].instructions[instruction_idx]
                    .operands
                    .push(Operand::IdRef(predecessor));
            }
        }
        for (instruction_idx, operands) in target_phi_updates {
            blocks[target_idx].instructions[instruction_idx].operands = operands;
        }
        any = true;
    }
    any
}

/// Privatize a direct arm shared by a nested selection and an enclosing sibling.
///
/// The short-circuit shape `outer -> shared | inner; inner -> shared | local` is not legal SPIR-V:
/// `shared` is an arm of `inner` but is not dominated by it. When `shared` is a single block and the
/// inner merge is a pass-through to the same outer continuation, clone `shared` for the nested edge
/// and route that clone through the inner merge. Values are carried by new phis at the inner merge,
/// using the exact former `shared`/inner-merge incoming pair from the outer continuation.
pub(in crate::passes) fn privatize_shared_direct_selection_arms(
    ctx: &mut Ctx,
    entry_idx: usize,
) -> bool {
    let mut any = false;
    loop {
        let blocks = &ctx.module.functions[entry_idx].blocks;
        let dominators = CompactDominators::new(blocks);
        let mut plan = None;
        'headers: for header in blocks {
            let Some(header_label) = header.label.as_ref().and_then(|label| label.result_id) else {
                continue;
            };
            let Some(merge) = header
                .instructions
                .iter()
                .find(|instruction| instruction.class.opcode == Op::SelectionMerge)
                .and_then(|instruction| instruction.operands.first())
                .and_then(id_ref_operand)
            else {
                continue;
            };
            let Some(terminator) = header.instructions.last() else {
                continue;
            };
            if terminator.class.opcode != Op::BranchConditional {
                continue;
            }
            for arm in terminator
                .operands
                .iter()
                .skip(1)
                .take(2)
                .filter_map(id_ref_operand)
            {
                if dominators.dominates(header_label, arm) {
                    continue;
                }
                let Some(arm_idx) = block_index_by_label(blocks, arm) else {
                    continue;
                };
                let arm_block = &blocks[arm_idx];
                let arm_successors = block_successors(arm_block);
                let [target] = arm_successors.as_slice() else {
                    continue;
                };
                let target = *target;
                let Some(merge_idx) = block_index_by_label(blocks, merge) else {
                    continue;
                };
                if phi_prefix_branch_target(&blocks[merge_idx]) != Some(target) {
                    continue;
                }
                if arm_block.instructions.iter().any(|instruction| {
                    matches!(
                        instruction.class.opcode,
                        Op::SelectionMerge | Op::LoopMerge | Op::Phi
                    )
                }) {
                    continue;
                }
                let defined = arm_block
                    .instructions
                    .iter()
                    .filter_map(|instruction| instruction.result_id)
                    .collect::<HashSet<_>>();
                let uses_are_local_or_merge_phi = blocks.iter().all(|block| {
                    let label = block.label.as_ref().and_then(|label| label.result_id);
                    block.instructions.iter().all(|instruction| {
                        instruction
                            .operands
                            .iter()
                            .enumerate()
                            .all(|(operand_idx, operand)| {
                                let Operand::IdRef(id) = operand else {
                                    return true;
                                };
                                if !defined.contains(id) || label == Some(arm) {
                                    return true;
                                }
                                label == Some(target)
                                    && instruction.class.opcode == Op::Phi
                                    && operand_idx % 2 == 0
                                    && instruction.operands.get(operand_idx + 1)
                                        == Some(&Operand::IdRef(arm))
                            })
                    })
                });
                if uses_are_local_or_merge_phi {
                    plan = Some((header_label, arm, arm_idx, merge, target, defined));
                    break 'headers;
                }
            }
        }
        let Some((header, arm, arm_idx, merge, target, defined)) = plan else {
            break;
        };

        let clone_label = ctx.module.fresh_id();
        let mut remap = HashMap::new();
        for id in defined {
            remap.insert(id, ctx.module.fresh_id());
        }
        let mut cloned = ctx.module.functions[entry_idx].blocks[arm_idx].clone();
        cloned.label = Some(Instruction::new(Op::Label, None, Some(clone_label), vec![]));
        for instruction in &mut cloned.instructions {
            if let Some(result) = instruction.result_id.as_mut() {
                if let Some(replacement) = remap.get(result) {
                    *result = *replacement;
                }
            }
            for operand in &mut instruction.operands {
                if let Operand::IdRef(id) = operand {
                    if let Some(replacement) = remap.get(id) {
                        *id = *replacement;
                    }
                }
            }
        }
        redirect_terminator_target(&mut cloned, target, merge);

        let (predecessors, carry_specs) = {
            let blocks = &ctx.module.functions[entry_idx].blocks;
            let predecessors = block_predecessors(blocks)
                .get(&merge)
                .cloned()
                .unwrap_or_default();
            let mut specs = Vec::new();
            if let Some(target_idx) = block_index_by_label(blocks, target) {
                for (phi_idx, instruction) in blocks[target_idx]
                    .instructions
                    .iter()
                    .take_while(|instruction| instruction.class.opcode == Op::Phi)
                    .enumerate()
                {
                    let mut shared_value = None;
                    let mut merge_value = None;
                    for pair in instruction.operands.chunks_exact(2) {
                        if pair[1] == Operand::IdRef(arm) {
                            shared_value = Some(pair[0].clone());
                        } else if pair[1] == Operand::IdRef(merge) {
                            merge_value = Some(pair[0].clone());
                        }
                    }
                    let (Some(mut shared_value), Some(merge_value)) = (shared_value, merge_value)
                    else {
                        continue;
                    };
                    if let Operand::IdRef(id) = &mut shared_value {
                        if let Some(replacement) = remap.get(id) {
                            *id = *replacement;
                        }
                    }
                    specs.push((phi_idx, instruction.result_type, shared_value, merge_value));
                }
            }
            (predecessors, specs)
        };
        let mut carried = Vec::new();
        for (phi_idx, result_type, shared_value, merge_value) in carry_specs {
            let result = ctx.module.fresh_id();
            let mut operands = Vec::with_capacity((predecessors.len() + 1) * 2);
            for predecessor in &predecessors {
                operands.push(merge_value.clone());
                operands.push(Operand::IdRef(*predecessor));
            }
            operands.push(shared_value);
            operands.push(Operand::IdRef(clone_label));
            carried.push((phi_idx, result_type, result, operands));
        }

        let blocks = &mut ctx.module.functions[entry_idx].blocks;
        let Some(header_idx) = block_index_by_label(blocks, header) else {
            break;
        };
        redirect_terminator_target(&mut blocks[header_idx], arm, clone_label);
        if let Some(merge_idx) = block_index_by_label(blocks, merge) {
            for (_, result_type, result, operands) in &carried {
                blocks[merge_idx].instructions.insert(
                    0,
                    Instruction::new(Op::Phi, *result_type, Some(*result), operands.clone()),
                );
            }
        }
        if let Some(target_idx) = block_index_by_label(blocks, target) {
            for (phi_idx, _, result, _) in &carried {
                let phi = &mut blocks[target_idx].instructions[*phi_idx];
                for pair in phi.operands.chunks_exact_mut(2) {
                    if pair[1] == Operand::IdRef(merge) {
                        pair[0] = Operand::IdRef(*result);
                    }
                }
            }
        }
        blocks.push(cloned);
        any = true;
    }
    any
}

/// Move reachable blocks after their CFG dominators when serialization placed them before one.
/// The CFG and SSA edges are unchanged; only block order moves.
///
/// Finish-time edge rewrites can make any previously serialized block, not only an `OpLoopMerge`
/// target, reachable through a later block. SPIR-V's ordering requirement is exact: every reachable
/// block must appear after its dominators. A stable topological order of the immediate-dominator tree
/// establishes the complete transitive property in one pass.
pub(in crate::passes) fn repair_dominator_block_order(ctx: &mut Ctx, entry_idx: usize) -> bool {
    let blocks = &ctx.module.functions[entry_idx].blocks;
    let dominators = CompactDominators::new(blocks);
    // The overwhelmingly common case is already serialized in dominator order. Checking each
    // immediate-dominator edge proves the complete transitive property in O(V); avoid the historical
    // all-blocks scan (O(V^2)) unless there is an actual ordering defect to repair. This matters for
    // generated functions with tens of thousands of blocks, which cross this shared repair boundary
    // more than once after independent late rewrites.
    if dominators.serialization_is_dominator_ordered(blocks) {
        return false;
    }
    // The immediate-dominator relation is a tree, so one stable topological ordering satisfies the
    // complete transitive rule. The former repair moved one block at a time and rescanned every
    // block/dominator pair after each move, becoming cubic on generated multi-thousand-block
    // kernels. Prefer the earliest original block whenever several tree nodes are ready so valid
    // sibling order remains as stable as the dependency permits.
    let order = dominators.stable_serialization_order(blocks);
    let blocks = &mut ctx.module.functions[entry_idx].blocks;
    let mut owned = std::mem::take(blocks)
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>();
    *blocks = order
        .into_iter()
        .map(|index| owned[index].take().expect("block order is a permutation"))
        .collect();
    true
}

/// Wrap a natural-loop target that has a CFG back-edge but no `OpLoopMerge` in a dedicated loop
/// header. This occurs when the original target is also a selection header: one SPIR-V block cannot
/// carry both merge instructions. The wrapper receives the target's leading phis, owns the loop
/// merge, and branches to the unchanged selection block.
pub(in crate::passes) fn repair_unmarked_natural_loops(ctx: &mut Ctx, entry_idx: usize) -> bool {
    let mut changed = false;
    loop {
        let blocks = &ctx.module.functions[entry_idx].blocks;
        let Some(entry) = blocks
            .first()
            .and_then(|block| block.label.as_ref())
            .and_then(|label| label.result_id)
        else {
            return changed;
        };
        let predecessors = block_predecessors(blocks);
        let successors = block_successors_by_label(blocks);
        let dominators = CompactDominators::new(blocks);
        let block_indices = blocks
            .iter()
            .enumerate()
            .filter_map(|(index, block)| Some((block.label.as_ref()?.result_id?, index)))
            .collect::<HashMap<_, _>>();
        let mut repair = None;
        for (&target, preds) in &predecessors {
            if target == entry {
                continue;
            }
            let Some(target_idx) = block_index_by_label(blocks, target) else {
                continue;
            };
            let has_loop_merge = blocks[target_idx]
                .instructions
                .iter()
                .any(|instruction| instruction.class.opcode == Op::LoopMerge);
            let backedges = preds
                .iter()
                .copied()
                .filter(|pred| dominators.dominates(target, *pred))
                .collect::<Vec<_>>();
            if has_loop_merge {
                continue;
            }
            // A single latch can itself be the SPIR-V continue target. Multiple latches require a
            // phi-aware continue funnel; leave those to the general structurizer until that local
            // composition is needed by a discovered source.
            let [continue_label] = backedges.as_slice() else {
                if !backedges.is_empty() && crate::env_vars::retry_debug() {
                    eprintln!(
                        "[retry-debug] unmarked-loop target={target} skipped: backedges={backedges:?}"
                    );
                }
                continue;
            };
            let continue_idx = block_indices[continue_label];
            let mut loop_nodes = HashSet::from([target]);
            let mut pending = vec![*continue_label];
            while let Some(label) = pending.pop() {
                let Some(&index) = block_indices.get(&label) else {
                    continue;
                };
                if !(target_idx..=continue_idx).contains(&index) || !loop_nodes.insert(label) {
                    continue;
                }
                pending.extend(predecessors.get(&label).into_iter().flatten().copied());
            }
            let mut exits = loop_nodes
                .iter()
                .flat_map(|label| successors.get(label).into_iter().flatten().copied())
                .filter(|successor| !loop_nodes.contains(successor))
                .collect::<HashSet<_>>();
            let nonterminating_exits = exits
                .iter()
                .copied()
                .filter(|exit| !exit_block_terminates(blocks, *exit))
                .collect::<HashSet<_>>();
            if !nonterminating_exits.is_empty() {
                exits = nonterminating_exits;
            }
            // A natural-loop core is defined by nodes that can reach its latch. Early-exit arms do
            // not satisfy that definition, even though SPIR-V includes them in the loop construct.
            // When every provisional exit necessarily reaches one common downstream label, extend
            // the construct through those arms and use that nearest common post-dominator as the
            // single legal loop merge. This preserves the arm computations and avoids a whole-CFG
            // state machine merely because a loop has several structured early exits.
            if exits.len() > 1 {
                if let Some(common_merge) = nearest_common_exit(blocks, &successors, &exits, target)
                {
                    loop_nodes.extend(reachable_before_target(blocks, target, common_merge));
                    exits = loop_nodes
                        .iter()
                        .flat_map(|label| successors.get(label).into_iter().flatten().copied())
                        .filter(|successor| !loop_nodes.contains(successor))
                        .collect();
                    let nonterminating_exits = exits
                        .iter()
                        .copied()
                        .filter(|exit| !exit_block_terminates(blocks, *exit))
                        .collect::<HashSet<_>>();
                    if !nonterminating_exits.is_empty() {
                        exits = nonterminating_exits;
                    }
                }
            }
            if exits.len() != 1 {
                if crate::env_vars::retry_debug() {
                    eprintln!(
                        "[retry-debug] unmarked-loop target={target} continue={} skipped: exits={exits:?}",
                        continue_label
                    );
                }
                continue;
            }
            let merge_label = *exits.iter().next().expect("one loop exit");
            repair = Some((
                target,
                target_idx,
                *continue_label,
                merge_label,
                preds.clone(),
            ));
            break;
        }
        let Some((target, target_idx, continue_label, merge_label, predecessors)) = repair else {
            break;
        };

        let header = ctx.module.fresh_id();
        let blocks = &mut ctx.module.functions[entry_idx].blocks;
        for predecessor in &predecessors {
            let Some(index) = block_index_by_label(blocks, *predecessor) else {
                continue;
            };
            redirect_terminator_target(&mut blocks[index], target, header);
        }
        let phi_count = blocks[target_idx]
            .instructions
            .iter()
            .take_while(|instruction| instruction.class.opcode == Op::Phi)
            .count();
        let mut header_instructions = blocks[target_idx]
            .instructions
            .drain(..phi_count)
            .collect::<Vec<_>>();
        header_instructions.push(Instruction::new(
            Op::LoopMerge,
            None,
            None,
            vec![
                Operand::IdRef(merge_label),
                Operand::IdRef(continue_label),
                Operand::LoopControl(spirv::LoopControl::NONE),
            ],
        ));
        header_instructions.push(Instruction::new(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(target)],
        ));
        blocks.insert(
            target_idx,
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(header), vec![])),
                instructions: header_instructions,
            },
        );
        changed = true;
    }
    changed
}

/// A direct terminating arm belongs to the loop construct but does not need to reach its merge. If
/// every exit terminates, the caller retains them so one terminating exit can still be the merge.
fn exit_block_terminates(blocks: &[Block], label: Word) -> bool {
    block_index_by_label(blocks, label).is_some_and(|index| {
        blocks[index]
            .instructions
            .last()
            .is_some_and(|instruction| {
                matches!(
                    instruction.class.opcode,
                    Op::Return
                        | Op::ReturnValue
                        | Op::Kill
                        | Op::TerminateInvocation
                        | Op::Unreachable
                )
            })
    })
}

/// Return the nearest exit label that every provisional loop exit must reach without re-entering
/// the loop header. Candidates are ranked by the total number of blocks traversed before them, which
/// selects the first common post-dominator rather than a later common tail.
fn nearest_common_exit(
    blocks: &[Block],
    successors: &HashMap<Word, Vec<Word>>,
    exits: &HashSet<Word>,
    loop_header: Word,
) -> Option<Word> {
    let mut reachable_from_every_exit: Option<HashSet<Word>> = None;
    for exit in exits {
        let reachable = reachable_labels_except(successors, *exit, Some(loop_header));
        reachable_from_every_exit = Some(match reachable_from_every_exit {
            Some(common) => common.intersection(&reachable).copied().collect(),
            None => reachable,
        });
    }
    reachable_from_every_exit?
        .iter()
        .copied()
        .filter(|candidate| {
            exits.iter().all(|start| {
                start == candidate
                    || all_paths_reach_target_without_labels(
                        blocks,
                        *start,
                        *candidate,
                        &[loop_header],
                    )
            })
        })
        .map(|candidate| {
            let distance = exits
                .iter()
                .map(|start| reachable_before_target(blocks, *start, candidate).len())
                .sum::<usize>();
            (distance, candidate)
        })
        .min()
        .map(|(_, candidate)| candidate)
}

fn split_reused_merge_claim(ctx: &mut Ctx, entry_idx: usize, header: Word, target: Word) {
    let dominators = CompactDominators::new(&ctx.module.functions[entry_idx].blocks);
    let _ = split_reused_merge_claim_with_dominators(
        ctx,
        entry_idx,
        header,
        target,
        &dominators,
        &HashMap::new(),
    );
}

fn split_reused_merge_claim_with_dominators(
    ctx: &mut Ctx,
    entry_idx: usize,
    header: Word,
    target: Word,
    dominators: &CompactDominators,
    synthetic_owners: &HashMap<Word, Word>,
) -> Option<Word> {
    if dominators.dominated_count(header) == 0 {
        return None;
    }
    let target_idx = block_index_by_label(&ctx.module.functions[entry_idx].blocks, target)?;
    // If the shared merge is itself a loop header, edges from blocks it dominates are back-edges.
    // They must continue to target that OpLoopMerge header; routing them through an ordinary
    // pass-through would turn the pass-through into the back-edge target, which SPIR-V forbids.
    let redirected = ctx.module.functions[entry_idx]
        .blocks
        .iter()
        .filter_map(|block| {
            let label = block.label.as_ref()?.result_id?;
            let owner = synthetic_owners.get(&label).copied().unwrap_or(label);
            (dominators.dominates(header, owner)
                && !dominators.dominates(target, owner)
                && block_successors(block).contains(&target))
            .then_some(label)
        })
        .collect::<HashSet<_>>();
    if redirected.is_empty() {
        return None;
    }

    let synthetic_label = ctx.module.fresh_id();
    let mut synthetic_instructions = Vec::new();
    let mut phi_updates = Vec::new();
    let target_phis = ctx.module.functions[entry_idx].blocks[target_idx]
        .instructions
        .iter()
        .take_while(|instruction| instruction.class.opcode == Op::Phi)
        .cloned()
        .enumerate()
        .collect::<Vec<_>>();
    for (instruction_idx, instruction) in target_phis {
        let mut kept = Vec::new();
        let mut routed = Vec::new();
        for pair in instruction.operands.chunks(2) {
            let [_, Operand::IdRef(predecessor)] = pair else {
                return None;
            };
            if redirected.contains(predecessor) {
                routed.extend_from_slice(pair);
            } else {
                kept.extend_from_slice(pair);
            }
        }
        if routed.is_empty() {
            continue;
        }
        let routed_value = if routed.len() == 2 {
            routed[0].clone()
        } else {
            let result = ctx.module.fresh_id();
            synthetic_instructions.push(Instruction::new(
                Op::Phi,
                instruction.result_type,
                Some(result),
                routed,
            ));
            Operand::IdRef(result)
        };
        kept.push(routed_value);
        kept.push(Operand::IdRef(synthetic_label));
        phi_updates.push((instruction_idx, kept));
    }
    synthetic_instructions.push(Instruction::new(
        Op::Branch,
        None,
        None,
        vec![Operand::IdRef(target)],
    ));

    let blocks = &mut ctx.module.functions[entry_idx].blocks;
    let header_idx = block_index_by_label(blocks, header)?;
    let merge = blocks[header_idx]
        .instructions
        .iter_mut()
        .find(|instruction| {
            matches!(instruction.class.opcode, Op::SelectionMerge | Op::LoopMerge)
                && instruction.operands.first().and_then(id_ref_operand) == Some(target)
        })?;
    if let Some(Operand::IdRef(merge_target)) = merge.operands.first_mut() {
        *merge_target = synthetic_label;
    }
    for block in blocks.iter_mut() {
        let label = block.label.as_ref().and_then(|label| label.result_id);
        if label.is_some_and(|label| redirected.contains(&label)) {
            redirect_terminator_target(block, target, synthetic_label);
        }
    }
    if let Some(target_block) = blocks.get_mut(target_idx) {
        for (instruction_idx, operands) in phi_updates {
            target_block.instructions[instruction_idx].operands = operands;
        }
    }
    blocks.insert(
        target_idx,
        Block {
            label: Some(Instruction::new(
                Op::Label,
                None,
                Some(synthetic_label),
                vec![],
            )),
            instructions: synthetic_instructions,
        },
    );
    Some(synthetic_label)
}

/// Compact immediate-dominator tree for repeated ownership queries in one repair round. The old
/// reachability-exclusion formulation rebuilt adjacency plus two `HashSet<Word>` traversals per query;
/// tree intervals represent the same entry-rooted dominance relation in O(V + E) storage total.
struct CompactDominators {
    labels: Vec<Word>,
    index: HashMap<Word, usize>,
    idom: Vec<Option<usize>>,
    preorder: Vec<usize>,
    postorder: Vec<usize>,
}

impl CompactDominators {
    fn new(blocks: &[Block]) -> Self {
        let labels = blocks
            .iter()
            .filter_map(|block| block.label.as_ref()?.result_id)
            .collect::<Vec<_>>();
        let index = labels
            .iter()
            .enumerate()
            .map(|(idx, label)| (*label, idx))
            .collect::<HashMap<_, _>>();
        let mut successors = vec![Vec::new(); labels.len()];
        let mut predecessors = vec![Vec::new(); labels.len()];
        for block in blocks {
            let Some(label) = block.label.as_ref().and_then(|label| label.result_id) else {
                continue;
            };
            let Some(&from) = index.get(&label) else {
                continue;
            };
            for successor in block_successors(block) {
                let Some(&to) = index.get(&successor) else {
                    continue;
                };
                successors[from].push(to);
                predecessors[to].push(from);
            }
        }

        let mut rpo = Vec::new();
        if !labels.is_empty() {
            let mut seen = vec![false; labels.len()];
            let mut stack = vec![(0usize, 0usize)];
            seen[0] = true;
            while let Some((node, next)) = stack.last_mut() {
                if *next < successors[*node].len() {
                    let successor = successors[*node][*next];
                    *next += 1;
                    if !seen[successor] {
                        seen[successor] = true;
                        stack.push((successor, 0));
                    }
                } else {
                    rpo.push(*node);
                    stack.pop();
                }
            }
            rpo.reverse();
        }
        let mut rpo_rank = vec![usize::MAX; labels.len()];
        for (rank, node) in rpo.iter().copied().enumerate() {
            rpo_rank[node] = rank;
        }
        let mut idom = vec![None; labels.len()];
        if let Some(&entry) = rpo.first() {
            idom[entry] = Some(entry);
            loop {
                let mut changed = false;
                for node in rpo.iter().copied().skip(1) {
                    let mut defined = predecessors[node]
                        .iter()
                        .copied()
                        .filter(|pred| idom[*pred].is_some());
                    let Some(mut next_idom) = defined.next() else {
                        continue;
                    };
                    for pred in defined {
                        next_idom = intersect_idom(pred, next_idom, &idom, &rpo_rank);
                    }
                    if idom[node] != Some(next_idom) {
                        idom[node] = Some(next_idom);
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        let mut children = vec![Vec::new(); labels.len()];
        for (node, parent) in idom.iter().copied().enumerate() {
            if let Some(parent) = parent.filter(|parent| *parent != node) {
                children[parent].push(node);
            }
        }
        let mut preorder = vec![usize::MAX; labels.len()];
        let mut postorder = vec![usize::MAX; labels.len()];
        if let Some(&entry) = rpo.first() {
            let mut clock = 0usize;
            let mut stack = vec![(entry, 0usize)];
            preorder[entry] = clock;
            clock += 1;
            while let Some((node, next)) = stack.last_mut() {
                if *next < children[*node].len() {
                    let child = children[*node][*next];
                    *next += 1;
                    preorder[child] = clock;
                    clock += 1;
                    stack.push((child, 0));
                } else {
                    postorder[*node] = clock;
                    stack.pop();
                }
            }
        }
        Self {
            labels,
            index,
            idom,
            preorder,
            postorder,
        }
    }

    fn dominates(&self, dominator: Word, node: Word) -> bool {
        let (Some(&dominator), Some(&node)) = (self.index.get(&dominator), self.index.get(&node))
        else {
            return false;
        };
        self.preorder[dominator] != usize::MAX
            && self.preorder[dominator] <= self.preorder[node]
            && self.preorder[node] < self.postorder[dominator]
    }

    fn dominated_count(&self, dominator: Word) -> usize {
        self.index
            .get(&dominator)
            .filter(|index| self.preorder[**index] != usize::MAX)
            .map_or(0, |index| self.postorder[*index] - self.preorder[*index])
    }

    fn serialization_is_dominator_ordered(&self, blocks: &[Block]) -> bool {
        let positions = blocks
            .iter()
            .enumerate()
            .filter_map(|(position, block)| Some((block.label.as_ref()?.result_id?, position)))
            .collect::<HashMap<_, _>>();
        self.idom.iter().enumerate().all(|(node, parent)| {
            let Some(parent) = parent.filter(|parent| *parent != node) else {
                return true;
            };
            match (
                positions.get(&self.labels[parent]),
                positions.get(&self.labels[node]),
            ) {
                (Some(parent_position), Some(node_position)) => parent_position < node_position,
                _ => true,
            }
        })
    }

    fn stable_serialization_order(&self, blocks: &[Block]) -> Vec<usize> {
        let positions = blocks
            .iter()
            .enumerate()
            .filter_map(|(position, block)| Some((block.label.as_ref()?.result_id?, position)))
            .collect::<HashMap<_, _>>();
        let mut children = vec![Vec::new(); blocks.len()];
        let mut has_parent = vec![false; blocks.len()];
        for (node, parent) in self.idom.iter().copied().enumerate() {
            let Some(parent) = parent.filter(|parent| *parent != node) else {
                continue;
            };
            let (Some(&node_position), Some(&parent_position)) = (
                positions.get(&self.labels[node]),
                positions.get(&self.labels[parent]),
            ) else {
                continue;
            };
            children[parent_position].push(node_position);
            has_parent[node_position] = true;
        }
        let mut ready = has_parent
            .iter()
            .enumerate()
            .filter_map(|(position, has_parent)| (!has_parent).then_some(position))
            .collect::<std::collections::BTreeSet<_>>();
        let mut order = Vec::with_capacity(blocks.len());
        while let Some(position) = ready.pop_first() {
            order.push(position);
            ready.extend(children[position].iter().copied());
        }
        debug_assert_eq!(order.len(), blocks.len());
        order
    }
}

fn intersect_idom(
    mut left: usize,
    mut right: usize,
    idom: &[Option<usize>],
    rpo_rank: &[usize],
) -> usize {
    while left != right {
        while rpo_rank[left] > rpo_rank[right] {
            left = idom[left].expect("reachable dominator predecessor");
        }
        while rpo_rank[right] > rpo_rank[left] {
            right = idom[right].expect("reachable dominator predecessor");
        }
    }
    left
}

fn reachable_labels_except(
    successors: &HashMap<Word, Vec<Word>>,
    entry: Word,
    excluded: Option<Word>,
) -> HashSet<Word> {
    let mut reachable = HashSet::new();
    let mut pending = vec![entry];
    while let Some(label) = pending.pop() {
        if Some(label) == excluded || !reachable.insert(label) {
            continue;
        }
        pending.extend(successors.get(&label).into_iter().flatten().copied());
    }
    reachable
}

pub(in crate::passes) fn repair_continue_selection_merge_targets(ctx: &mut Ctx, entry_idx: usize) {
    loop {
        if !repair_one_continue_selection_merge_target(ctx, entry_idx) {
            break;
        }
    }
}

pub(in crate::passes) fn repair_loop_continue_external_predecessors(
    ctx: &mut Ctx,
    entry_idx: usize,
) -> bool {
    let mut changed = false;
    loop {
        if !repair_one_loop_continue_external_predecessor(ctx, entry_idx) {
            break;
        }
        changed = true;
    }
    changed
}

pub(in crate::passes) fn repair_loop_continue_pass_through_targets(
    ctx: &mut Ctx,
    entry_idx: usize,
) {
    loop {
        if !repair_one_loop_continue_pass_through_target(ctx, entry_idx) {
            break;
        }
    }
}

pub(in crate::passes) fn repair_one_loop_continue_pass_through_target(
    ctx: &mut Ctx,
    entry_idx: usize,
) -> bool {
    let blocks = &mut ctx.module.functions[entry_idx].blocks;
    for header_idx in 0..blocks.len() {
        let Some(header_label) = blocks[header_idx]
            .label
            .as_ref()
            .and_then(|label| label.result_id)
        else {
            continue;
        };
        let Some(loop_merge_idx) = blocks[header_idx]
            .instructions
            .iter()
            .position(|inst| inst.class.opcode == Op::LoopMerge)
        else {
            continue;
        };
        let Some(loop_merge_label) = blocks[header_idx].instructions[loop_merge_idx]
            .operands
            .first()
            .and_then(id_ref_operand)
        else {
            continue;
        };
        let Some(continue_label) = blocks[header_idx].instructions[loop_merge_idx]
            .operands
            .get(1)
            .and_then(id_ref_operand)
        else {
            continue;
        };
        let Some(continue_idx) = block_index_by_label(blocks, continue_label) else {
            continue;
        };
        let Some(target) = branch_only_target(&blocks[continue_idx]) else {
            continue;
        };
        if target == header_label || target == loop_merge_label {
            continue;
        }
        if let Some(Operand::IdRef(continue_target)) = blocks[header_idx].instructions
            [loop_merge_idx]
            .operands
            .get_mut(1)
        {
            *continue_target = target;
            return true;
        }
    }
    false
}

pub(in crate::passes) fn repair_one_loop_continue_external_predecessor(
    ctx: &mut Ctx,
    entry_idx: usize,
) -> bool {
    let blocks = &ctx.module.functions[entry_idx].blocks;
    let predecessors = block_predecessors(blocks);
    let dominators = CompactDominators::new(blocks);

    for header_idx in 0..blocks.len() {
        let Some(header_label) = blocks[header_idx]
            .label
            .as_ref()
            .and_then(|label| label.result_id)
        else {
            continue;
        };
        let Some(loop_merge) = blocks[header_idx]
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == Op::LoopMerge)
        else {
            continue;
        };
        let Some(continue_label) = loop_merge.operands.get(1).and_then(id_ref_operand) else {
            continue;
        };
        if continue_label == header_label {
            continue;
        }
        let Some(continue_idx) = block_index_by_label(blocks, continue_label) else {
            continue;
        };
        if phi_prefix_branch_target(&blocks[continue_idx]) != Some(header_label) {
            continue;
        }
        let Some(continue_preds) = predecessors.get(&continue_label).cloned() else {
            continue;
        };
        let outside_preds = continue_preds
            .iter()
            .copied()
            .filter(|pred| !dominators.dominates(header_label, *pred))
            .collect::<HashSet<_>>();
        if outside_preds.is_empty() {
            continue;
        }

        let mut continue_phi_updates = Vec::new();
        let mut continue_phi_splits: HashMap<Word, PhiSplit> = HashMap::new();
        for (inst_idx, inst) in blocks[continue_idx].instructions.iter().enumerate() {
            if inst.class.opcode != Op::Phi {
                break;
            }
            let Some(result) = inst.result_id else {
                continue;
            };
            let (inside, outside) = split_phi_operands(&inst.operands, &outside_preds);
            if outside.is_empty() {
                continue;
            }
            if inside.is_empty() {
                continue_phi_updates.clear();
                continue_phi_splits.clear();
                break;
            }
            continue_phi_updates.push((inst_idx, inside.clone()));
            continue_phi_splits.insert(result, PhiSplit { inside, outside });
        }
        if !outside_preds.is_empty()
            && blocks[header_idx]
                .instructions
                .iter()
                .take_while(|inst| inst.class.opcode == Op::Phi)
                .any(|inst| {
                    !phi_predicates_after_continue_split(
                        inst,
                        continue_label,
                        &outside_preds,
                        &continue_phi_splits,
                    )
                })
        {
            continue;
        }

        let mut header_phi_updates = Vec::new();
        for (inst_idx, inst) in blocks[header_idx].instructions.iter().enumerate() {
            if inst.class.opcode != Op::Phi {
                break;
            }
            let Some(operands) =
                split_header_phi_continue_incoming(inst, continue_label, &continue_phi_splits)
            else {
                continue;
            };
            header_phi_updates.push((inst_idx, operands));
        }

        let blocks = &mut ctx.module.functions[entry_idx].blocks;
        for (inst_idx, operands) in continue_phi_updates {
            if let Some(inst) = blocks[continue_idx].instructions.get_mut(inst_idx) {
                inst.operands = operands;
            }
        }
        for (inst_idx, operands) in header_phi_updates {
            if let Some(inst) = blocks[header_idx].instructions.get_mut(inst_idx) {
                inst.operands = operands;
            }
        }
        for block in blocks.iter_mut() {
            let Some(label) = block.label.as_ref().and_then(|label| label.result_id) else {
                continue;
            };
            if outside_preds.contains(&label) {
                redirect_terminator_target(block, continue_label, header_label);
            }
        }
        return true;
    }
    false
}

pub(in crate::passes) fn repair_one_continue_selection_merge_target(
    ctx: &mut Ctx,
    entry_idx: usize,
) -> bool {
    let dominators = CompactDominators::new(&ctx.module.functions[entry_idx].blocks);
    let block_count = ctx.module.functions[entry_idx].blocks.len();
    for header_idx in 0..block_count {
        let Some(loop_header_label) = ctx.module.functions[entry_idx].blocks[header_idx]
            .label
            .as_ref()
            .and_then(|label| label.result_id)
        else {
            continue;
        };
        let Some(loop_merge) = ctx.module.functions[entry_idx].blocks[header_idx]
            .instructions
            .iter()
            .find(|inst| inst.class.opcode == Op::LoopMerge)
        else {
            continue;
        };
        let Some(loop_merge_label) = loop_merge.operands.first().and_then(id_ref_operand) else {
            continue;
        };
        let Some(continue_label) = loop_merge.operands.get(1).and_then(id_ref_operand) else {
            continue;
        };

        for block_idx in 0..block_count {
            let Some(block_label) = ctx.module.functions[entry_idx].blocks[block_idx]
                .label
                .as_ref()
                .and_then(|label| label.result_id)
            else {
                continue;
            };
            let Some(selection_merge_idx) = ctx.module.functions[entry_idx].blocks[block_idx]
                .instructions
                .iter()
                .position(|inst| inst.class.opcode == Op::SelectionMerge)
            else {
                continue;
            };
            let selection_merge = ctx.module.functions[entry_idx].blocks[block_idx].instructions
                [selection_merge_idx]
                .operands
                .first()
                .and_then(id_ref_operand);
            if selection_merge != Some(continue_label) {
                continue;
            }
            let Some(term) = ctx.module.functions[entry_idx].blocks[block_idx]
                .instructions
                .last()
            else {
                continue;
            };
            if term.class.opcode != Op::BranchConditional {
                continue;
            }
            let Some(true_target) = term.operands.get(1).and_then(id_ref_operand) else {
                continue;
            };
            let Some(false_target) = term.operands.get(2).and_then(id_ref_operand) else {
                continue;
            };
            // A selection outside a loop cannot reconverge in that loop's continue construct: the
            // back edge would enter the selection merge. If the loop exit necessarily reaches the
            // outside arm, that arm is their nearest shared boundary; otherwise the loop exit is
            // the valid boundary and a terminating outside arm need not reach it. Nested selections
            // are handled below by the existing phi-aware split.
            if dominators.dominates(block_label, loop_header_label) {
                let outside_arm = [true_target, false_target]
                    .into_iter()
                    .find(|target| !dominators.dominates(loop_header_label, *target));
                let boundary = outside_arm
                    .filter(|outside| {
                        all_paths_reach_target_without_labels(
                            &ctx.module.functions[entry_idx].blocks,
                            loop_merge_label,
                            *outside,
                            &[loop_header_label],
                        )
                    })
                    .unwrap_or(loop_merge_label);
                if let Some(Operand::IdRef(merge)) = ctx.module.functions[entry_idx].blocks
                    [block_idx]
                    .instructions[selection_merge_idx]
                    .operands
                    .first_mut()
                {
                    *merge = boundary;
                    return true;
                }
            }
            let branches_to_continue_and_merge = (true_target == continue_label
                && false_target == loop_merge_label)
                || (false_target == continue_label && true_target == loop_merge_label);
            if !branches_to_continue_and_merge {
                if all_paths_reach_target_without_labels(
                    &ctx.module.functions[entry_idx].blocks,
                    true_target,
                    continue_label,
                    &[loop_merge_label],
                ) && all_paths_reach_target_without_labels(
                    &ctx.module.functions[entry_idx].blocks,
                    false_target,
                    continue_label,
                    &[loop_merge_label],
                ) {
                    return split_selection_continue_merge(
                        ctx,
                        entry_idx,
                        block_idx,
                        selection_merge_idx,
                        continue_label,
                    );
                }
                continue;
            }
            let Some(loop_merge_idx) =
                block_index_by_label(&ctx.module.functions[entry_idx].blocks, loop_merge_label)
            else {
                continue;
            };

            let synthetic_label = ctx.module.fresh_id();
            let blocks = &mut ctx.module.functions[entry_idx].blocks;
            if let Some(Operand::IdRef(label)) = blocks[block_idx].instructions[selection_merge_idx]
                .operands
                .first_mut()
            {
                *label = synthetic_label;
            }
            redirect_terminator_target(&mut blocks[block_idx], loop_merge_label, synthetic_label);
            replace_phi_predecessor(blocks, loop_merge_label, block_label, synthetic_label);
            blocks.insert(
                loop_merge_idx,
                Block {
                    label: Some(Instruction::new(
                        Op::Label,
                        None,
                        Some(synthetic_label),
                        vec![],
                    )),
                    instructions: vec![Instruction::new(
                        Op::Branch,
                        None,
                        None,
                        vec![Operand::IdRef(loop_merge_label)],
                    )],
                },
            );
            return true;
        }
    }
    false
}

#[derive(Clone)]
pub(in crate::passes) struct PhiSplit {
    pub(in crate::passes) inside: Vec<Operand>,
    pub(in crate::passes) outside: Vec<Operand>,
}

pub(in crate::passes) fn split_phi_operands(
    operands: &[Operand],
    outside_preds: &HashSet<Word>,
) -> (Vec<Operand>, Vec<Operand>) {
    let mut inside = Vec::new();
    let mut outside = Vec::new();
    for pair in operands.chunks(2) {
        let [value, Operand::IdRef(pred)] = pair else {
            inside.extend_from_slice(pair);
            continue;
        };
        if outside_preds.contains(pred) {
            outside.push(value.clone());
            outside.push(Operand::IdRef(*pred));
        } else {
            inside.push(value.clone());
            inside.push(Operand::IdRef(*pred));
        }
    }
    (inside, outside)
}

pub(in crate::passes) fn phi_predicates_after_continue_split(
    inst: &Instruction,
    continue_label: Word,
    outside_preds: &HashSet<Word>,
    continue_phi_splits: &HashMap<Word, PhiSplit>,
) -> bool {
    let Some(operands) =
        split_header_phi_continue_incoming(inst, continue_label, continue_phi_splits)
    else {
        return false;
    };
    let preds = operands
        .chunks(2)
        .filter_map(|pair| match pair {
            [_, Operand::IdRef(pred)] => Some(*pred),
            _ => None,
        })
        .collect::<HashSet<_>>();
    outside_preds.iter().all(|pred| preds.contains(pred))
}

pub(in crate::passes) fn split_header_phi_continue_incoming(
    inst: &Instruction,
    continue_label: Word,
    continue_phi_splits: &HashMap<Word, PhiSplit>,
) -> Option<Vec<Operand>> {
    let mut changed = false;
    let mut operands = Vec::new();
    for pair in inst.operands.chunks(2) {
        let [value, Operand::IdRef(pred)] = pair else {
            operands.extend_from_slice(pair);
            continue;
        };
        if *pred != continue_label {
            operands.push(value.clone());
            operands.push(Operand::IdRef(*pred));
            continue;
        }
        let Operand::IdRef(value_id) = value else {
            operands.push(value.clone());
            operands.push(Operand::IdRef(*pred));
            continue;
        };
        let Some(split) = continue_phi_splits.get(value_id) else {
            operands.push(value.clone());
            operands.push(Operand::IdRef(*pred));
            continue;
        };
        changed = true;
        if !split.inside.is_empty() {
            operands.push(value.clone());
            operands.push(Operand::IdRef(*pred));
        }
        operands.extend(split.outside.clone());
    }
    changed.then_some(operands)
}

pub(in crate::passes) fn split_selection_continue_merge(
    ctx: &mut Ctx,
    entry_idx: usize,
    header_idx: usize,
    selection_merge_idx: usize,
    continue_label: Word,
) -> bool {
    let Some(header_label) = ctx.module.functions[entry_idx].blocks[header_idx]
        .label
        .as_ref()
        .and_then(|label| label.result_id)
    else {
        return false;
    };
    let Some(continue_idx) =
        block_index_by_label(&ctx.module.functions[entry_idx].blocks, continue_label)
    else {
        return false;
    };
    let construct_labels = reachable_before_target(
        &ctx.module.functions[entry_idx].blocks,
        header_label,
        continue_label,
    );
    let synthetic_label = ctx.module.fresh_id();

    let mut redirected_preds = HashSet::new();
    {
        let blocks = &mut ctx.module.functions[entry_idx].blocks;
        if let Some(Operand::IdRef(label)) = blocks[header_idx].instructions[selection_merge_idx]
            .operands
            .first_mut()
        {
            *label = synthetic_label;
        }
        for block in blocks.iter_mut() {
            let Some(pred) = block.label.as_ref().and_then(|label| label.result_id) else {
                continue;
            };
            if !construct_labels.contains(&pred) {
                continue;
            }
            if redirect_terminator_target(block, continue_label, synthetic_label) {
                redirected_preds.insert(pred);
            }
        }
    }
    if redirected_preds.is_empty() {
        return false;
    }

    let phi_splits = {
        let blocks = &ctx.module.functions[entry_idx].blocks;
        let Some(continue_block) = blocks.get(continue_idx) else {
            return false;
        };
        let mut splits = Vec::new();
        for (inst_idx, inst) in continue_block.instructions.iter().enumerate() {
            if inst.class.opcode != Op::Phi {
                break;
            }
            let mut kept = Vec::new();
            let mut redirected = Vec::new();
            for pair in inst.operands.chunks(2) {
                if pair.len() != 2 {
                    kept.extend_from_slice(pair);
                    continue;
                }
                let is_redirected =
                    matches!(pair[1], Operand::IdRef(pred) if redirected_preds.contains(&pred));
                if is_redirected {
                    redirected.extend_from_slice(pair);
                } else {
                    kept.extend_from_slice(pair);
                }
            }
            if !redirected.is_empty() {
                splits.push((inst_idx, inst.result_type, kept, redirected));
            }
        }
        splits
    };

    let mut synthetic_instructions = Vec::new();
    let mut phi_updates = Vec::new();
    for (inst_idx, result_type, mut kept, redirected) in phi_splits {
        let phi_id = ctx.module.fresh_id();
        synthetic_instructions.push(Instruction::new(
            Op::Phi,
            result_type,
            Some(phi_id),
            redirected,
        ));
        kept.push(Operand::IdRef(phi_id));
        kept.push(Operand::IdRef(synthetic_label));
        phi_updates.push((inst_idx, kept));
    }
    synthetic_instructions.push(Instruction::new(
        Op::Branch,
        None,
        None,
        vec![Operand::IdRef(continue_label)],
    ));

    let blocks = &mut ctx.module.functions[entry_idx].blocks;
    if let Some(continue_block) = blocks.get_mut(continue_idx) {
        for (inst_idx, operands) in phi_updates {
            if let Some(inst) = continue_block.instructions.get_mut(inst_idx) {
                inst.operands = operands;
            }
        }
    }
    blocks.insert(
        continue_idx,
        Block {
            label: Some(Instruction::new(
                Op::Label,
                None,
                Some(synthetic_label),
                vec![],
            )),
            instructions: synthetic_instructions,
        },
    );
    true
}

pub(in crate::passes) fn repair_phi_predecessor_edges(ctx: &mut Ctx, entry_idx: usize) -> bool {
    let mut changed = funnel_stale_phi_predecessors(ctx, entry_idx);
    let blocks = &ctx.module.functions[entry_idx].blocks;
    let predecessors = block_predecessors(blocks);
    let dominators = CompactDominators::new(blocks);
    let def_blocks = value_def_blocks(&ctx.module.functions[entry_idx]);

    let blocks = &mut ctx.module.functions[entry_idx].blocks;
    for block in blocks {
        let Some(label) = block.label.as_ref().and_then(|label| label.result_id) else {
            continue;
        };
        let Some(preds) = predecessors.get(&label) else {
            continue;
        };
        for inst in &mut block.instructions {
            if inst.class.opcode != Op::Phi {
                break;
            }
            let incoming = phi_incoming(inst);
            let incoming_preds = incoming
                .iter()
                .map(|(_, pred)| *pred)
                .collect::<HashSet<_>>();
            let actual_preds = preds.iter().copied().collect::<HashSet<_>>();
            let missing_preds = preds
                .iter()
                .copied()
                .filter(|pred| !incoming_preds.contains(pred))
                .collect::<Vec<_>>();
            for missing_pred in missing_preds {
                let mut replaced_stale = false;
                for pair in inst.operands.chunks_mut(2) {
                    let [value, Operand::IdRef(old_pred)] = pair else {
                        continue;
                    };
                    if actual_preds.contains(old_pred) {
                        continue;
                    }
                    if !phi_value_available_on_edge(value, missing_pred, &dominators, &def_blocks) {
                        continue;
                    }
                    *old_pred = missing_pred;
                    changed = true;
                    replaced_stale = true;
                    break;
                }
                if replaced_stale {
                    continue;
                }
                let Some((value, _)) = incoming.iter().find(|(value, _)| {
                    phi_value_available_on_edge(value, missing_pred, &dominators, &def_blocks)
                }) else {
                    continue;
                };
                inst.operands.push(value.clone());
                inst.operands.push(Operand::IdRef(missing_pred));
                changed = true;
            }
            let old_len = inst.operands.len();
            inst.operands = inst
                .operands
                .chunks(2)
                .filter(|pair| {
                    let [_, Operand::IdRef(pred)] = pair else {
                        return false;
                    };
                    actual_preds.contains(pred)
                })
                .flatten()
                .cloned()
                .collect();
            changed |= inst.operands.len() != old_len;
        }
    }
    changed
}

/// Move a phi's stale multi-edge merge into its sole pass-through predecessor. Structurization can
/// insert a loop header `F` between the original preheader/backedge set and a phi block `B`; `B` then
/// has only predecessor `F`, while its values still name the older edges. Rebuild that merge as a phi
/// in `F` and make `B` consume the single forwarded value.
fn funnel_stale_phi_predecessors(ctx: &mut Ctx, entry_idx: usize) -> bool {
    struct Plan {
        block_idx: usize,
        phi_result: Word,
        result_type: Word,
        funnel_idx: usize,
        funnel_label: Word,
        operands: Vec<Operand>,
    }

    let blocks = &ctx.module.functions[entry_idx].blocks;
    let predecessors = block_predecessors(blocks);
    let dominators = CompactDominators::new(blocks);
    let def_blocks = value_def_blocks(&ctx.module.functions[entry_idx]);
    let label_to_index = blocks
        .iter()
        .enumerate()
        .filter_map(|(idx, block)| Some((block.label.as_ref()?.result_id?, idx)))
        .collect::<HashMap<_, _>>();
    let mut plans = Vec::new();
    for (block_idx, block) in blocks.iter().enumerate() {
        let Some(label) = block.label.as_ref().and_then(|label| label.result_id) else {
            continue;
        };
        let Some([funnel_label]) = predecessors.get(&label).map(Vec::as_slice) else {
            continue;
        };
        let Some(funnel_preds) = predecessors.get(funnel_label) else {
            continue;
        };
        if funnel_preds.len() < 2 {
            continue;
        }
        let Some(&funnel_idx) = label_to_index.get(funnel_label) else {
            continue;
        };
        for phi in &block.instructions {
            if phi.class.opcode != Op::Phi {
                break;
            }
            let (Some(phi_result), Some(result_type)) = (phi.result_id, phi.result_type) else {
                continue;
            };
            let incoming = phi_incoming(phi);
            if incoming.len() < 2 || incoming.iter().any(|(_, pred)| pred == funnel_label) {
                continue;
            }
            let mut operands = Vec::with_capacity(funnel_preds.len() * 2);
            let mut complete = true;
            for pred in funnel_preds {
                let applicable = incoming
                    .iter()
                    .filter(|(value, old_pred)| {
                        dominators.dominates(*old_pred, *pred)
                            && phi_value_available_on_edge(value, *pred, &dominators, &def_blocks)
                    })
                    .collect::<Vec<_>>();
                let deepest = applicable.iter().filter(|(_, candidate_pred)| {
                    applicable
                        .iter()
                        .all(|(_, other_pred)| dominators.dominates(*other_pred, *candidate_pred))
                });
                let selected = deepest.copied().collect::<Vec<_>>();
                let [selected] = selected.as_slice() else {
                    complete = false;
                    break;
                };
                operands.push(selected.0.clone());
                operands.push(Operand::IdRef(*pred));
            }
            if complete {
                plans.push(Plan {
                    block_idx,
                    phi_result,
                    result_type,
                    funnel_idx,
                    funnel_label: *funnel_label,
                    operands,
                });
            }
        }
    }
    if plans.is_empty() {
        return false;
    }

    for plan in plans {
        let forwarded = ctx.module.fresh_id();
        let blocks = &mut ctx.module.functions[entry_idx].blocks;
        let funnel = &mut blocks[plan.funnel_idx];
        let insert_at = funnel
            .instructions
            .iter()
            .position(|instruction| instruction.class.opcode != Op::Phi)
            .unwrap_or(funnel.instructions.len());
        funnel.instructions.insert(
            insert_at,
            Instruction::new(
                Op::Phi,
                Some(plan.result_type),
                Some(forwarded),
                plan.operands,
            ),
        );
        if let Some(phi) = blocks[plan.block_idx]
            .instructions
            .iter_mut()
            .find(|instruction| instruction.result_id == Some(plan.phi_result))
        {
            phi.operands = vec![Operand::IdRef(forwarded), Operand::IdRef(plan.funnel_label)];
        }
    }
    true
}

pub(in crate::passes) fn block_index_by_label(blocks: &[Block], label_id: Word) -> Option<usize> {
    blocks
        .iter()
        .position(|block| block.label.as_ref().and_then(|label| label.result_id) == Some(label_id))
}

pub(in crate::passes) fn branch_only_target(block: &Block) -> Option<Word> {
    let [inst] = block.instructions.as_slice() else {
        return None;
    };
    (inst.class.opcode == Op::Branch)
        .then(|| inst.operands.first().and_then(id_ref_operand))
        .flatten()
}

pub(in crate::passes) fn phi_prefix_branch_target(block: &Block) -> Option<Word> {
    let mut iter = block.instructions.iter();
    let branch = loop {
        let inst = iter.next()?;
        if inst.class.opcode == Op::Phi {
            continue;
        }
        break inst;
    };
    if branch.class.opcode != Op::Branch || iter.next().is_some() {
        return None;
    }
    branch.operands.first().and_then(id_ref_operand)
}

pub(in crate::passes) fn block_predecessors(blocks: &[Block]) -> HashMap<Word, Vec<Word>> {
    let mut predecessors: HashMap<Word, Vec<Word>> = HashMap::new();
    for block in blocks {
        let Some(label) = block.label.as_ref().and_then(|label| label.result_id) else {
            continue;
        };
        for successor in block_successors(block) {
            let entry = predecessors.entry(successor).or_default();
            if !entry.contains(&label) {
                entry.push(label);
            }
        }
    }
    predecessors
}

pub(in crate::passes) fn all_paths_reach_target_without_labels(
    blocks: &[Block],
    start: Word,
    target: Word,
    forbidden: &[Word],
) -> bool {
    fn walk(
        blocks: &[Block],
        label: Word,
        target: Word,
        forbidden: &[Word],
        visiting: &mut HashSet<Word>,
        memo: &mut HashMap<Word, bool>,
    ) -> bool {
        if label == target {
            return true;
        }
        if forbidden.contains(&label) {
            return false;
        }
        if let Some(&cached) = memo.get(&label) {
            return cached;
        }
        if !visiting.insert(label) {
            return true;
        }
        let ok = block_index_by_label(blocks, label).is_some_and(|idx| {
            let successors = block_successors(&blocks[idx]);
            !successors.is_empty()
                && successors
                    .iter()
                    .all(|successor| walk(blocks, *successor, target, forbidden, visiting, memo))
        });
        visiting.remove(&label);
        memo.insert(label, ok);
        ok
    }

    walk(
        blocks,
        start,
        target,
        forbidden,
        &mut HashSet::new(),
        &mut HashMap::new(),
    )
}

pub(in crate::passes) fn reachable_before_target(
    blocks: &[Block],
    start: Word,
    target: Word,
) -> HashSet<Word> {
    let mut seen = HashSet::new();
    let mut stack = vec![start];
    while let Some(label) = stack.pop() {
        if label == target || !seen.insert(label) {
            continue;
        }
        let Some(idx) = block_index_by_label(blocks, label) else {
            continue;
        };
        stack.extend(block_successors(&blocks[idx]));
    }
    seen
}

pub(in crate::passes) fn id_ref_operand(operand: &Operand) -> Option<Word> {
    match operand {
        Operand::IdRef(id) => Some(*id),
        _ => None,
    }
}

pub(in crate::passes) fn redirect_terminator_target(
    block: &mut Block,
    from: Word,
    to: Word,
) -> bool {
    let Some(term) = block.instructions.last_mut() else {
        return false;
    };
    let mut changed = false;
    match term.class.opcode {
        Op::Branch => {
            if let Some(Operand::IdRef(label)) = term.operands.first_mut() {
                if *label == from {
                    *label = to;
                    changed = true;
                }
            }
        }
        Op::BranchConditional => {
            for operand in term.operands.iter_mut().skip(1).take(2) {
                if let Operand::IdRef(label) = operand {
                    if *label == from {
                        *label = to;
                        changed = true;
                    }
                }
            }
        }
        Op::Switch => {
            for index in std::iter::once(1).chain((3..term.operands.len()).step_by(2)) {
                if let Some(Operand::IdRef(label)) = term.operands.get_mut(index) {
                    if *label == from {
                        *label = to;
                        changed = true;
                    }
                }
            }
        }
        _ => {}
    }
    changed
}

pub(in crate::passes) fn replace_phi_predecessor(
    blocks: &mut [Block],
    target: Word,
    from: Word,
    to: Word,
) {
    let Some(block) = blocks
        .iter_mut()
        .find(|block| block.label.as_ref().and_then(|label| label.result_id) == Some(target))
    else {
        return;
    };
    for inst in &mut block.instructions {
        if inst.class.opcode != Op::Phi {
            break;
        }
        let mut idx = 1;
        while idx < inst.operands.len() {
            if inst.operands[idx] == Operand::IdRef(from) {
                inst.operands[idx] = Operand::IdRef(to);
            }
            idx += 2;
        }
    }
}

pub(in crate::passes) fn value_def_blocks(function: &Function) -> HashMap<Word, Word> {
    let mut defs = HashMap::new();
    for block in &function.blocks {
        let Some(label) = block.label.as_ref().and_then(|label| label.result_id) else {
            continue;
        };
        for inst in &block.instructions {
            if let Some(result) = inst.result_id {
                defs.insert(result, label);
            }
        }
    }
    defs
}

pub(in crate::passes) fn phi_incoming(inst: &Instruction) -> Vec<(Operand, Word)> {
    inst.operands
        .chunks(2)
        .filter_map(|pair| {
            let [value, Operand::IdRef(pred)] = pair else {
                return None;
            };
            Some((value.clone(), *pred))
        })
        .collect()
}

fn phi_value_available_on_edge(
    value: &Operand,
    pred: Word,
    dominators: &CompactDominators,
    def_blocks: &HashMap<Word, Word>,
) -> bool {
    let Operand::IdRef(value) = value else {
        return true;
    };
    let Some(def_block) = def_blocks.get(value).copied() else {
        return true;
    };
    dominators.dominates(def_block, pred)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::{Function, ModuleHeader};

    fn branch(label: Word, target: Word) -> Block {
        Block {
            label: Some(Instruction::new(Op::Label, None, Some(label), vec![])),
            instructions: vec![Instruction::new(
                Op::Branch,
                None,
                None,
                vec![Operand::IdRef(target)],
            )],
        }
    }

    fn selection(label: Word, condition: Word, yes: Word, no: Word, merge: Word) -> Block {
        Block {
            label: Some(Instruction::new(Op::Label, None, Some(label), vec![])),
            instructions: vec![
                Instruction::new(Op::SelectionMerge, None, None, vec![Operand::IdRef(merge)]),
                Instruction::new(
                    Op::BranchConditional,
                    None,
                    None,
                    vec![
                        Operand::IdRef(condition),
                        Operand::IdRef(yes),
                        Operand::IdRef(no),
                    ],
                ),
            ],
        }
    }

    #[test]
    fn stale_loop_merge_becomes_selection_without_back_edge() {
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(100));
        module.functions.push(Function {
            def: None,
            parameters: vec![],
            blocks: vec![
                branch(1, 2),
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(2), vec![])),
                    instructions: vec![
                        Instruction::new(
                            Op::LoopMerge,
                            None,
                            None,
                            vec![
                                Operand::IdRef(4),
                                Operand::IdRef(3),
                                Operand::LoopControl(spirv::LoopControl::NONE),
                            ],
                        ),
                        Instruction::new(
                            Op::BranchConditional,
                            None,
                            None,
                            vec![Operand::IdRef(90), Operand::IdRef(3), Operand::IdRef(4)],
                        ),
                    ],
                },
                branch(3, 4),
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(4), vec![])),
                    instructions: vec![Instruction::new(Op::Return, None, None, vec![])],
                },
            ],
            end: None,
        });
        let mut ctx = Ctx::new(module);

        assert!(downgrade_stale_loop_merges(&mut ctx, 0));
        let header = &ctx.module.functions[0].blocks[1];
        assert_eq!(header.instructions[0].class.opcode, Op::SelectionMerge);
        assert_eq!(header.instructions[0].operands[0], Operand::IdRef(4));
        assert!(!downgrade_stale_loop_merges(&mut ctx, 0));
    }

    #[test]
    fn phi_bearing_selection_bypass_is_funnelled_through_declared_merge() {
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(100));
        module.functions.push(Function {
            def: None,
            parameters: vec![],
            blocks: vec![
                selection(1, 90, 2, 3, 8),
                selection(2, 91, 4, 5, 6),
                branch(3, 8),
                branch(4, 6),
                branch(5, 8),
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(6), vec![])),
                    instructions: vec![
                        Instruction::new(
                            Op::Phi,
                            Some(50),
                            Some(60),
                            vec![Operand::IdRef(70), Operand::IdRef(4)],
                        ),
                        Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(8)]),
                    ],
                },
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(8), vec![])),
                    instructions: vec![
                        Instruction::new(
                            Op::Phi,
                            Some(50),
                            Some(61),
                            vec![
                                Operand::IdRef(60),
                                Operand::IdRef(6),
                                Operand::IdRef(71),
                                Operand::IdRef(5),
                                Operand::IdRef(72),
                                Operand::IdRef(3),
                            ],
                        ),
                        Instruction::new(Op::Return, None, None, vec![]),
                    ],
                },
            ],
            end: None,
        });
        let mut ctx = Ctx::new(module);

        assert!(funnel_selection_merge_bypasses(&mut ctx, 0));
        let blocks = &ctx.module.functions[0].blocks;
        assert_eq!(
            block_successors(&blocks[block_index_by_label(blocks, 5).unwrap()]),
            [6]
        );
        assert_eq!(
            phi_incoming(&blocks[block_index_by_label(blocks, 6).unwrap()].instructions[0]),
            [(Operand::IdRef(70), 4), (Operand::IdRef(71), 5)]
        );
        assert_eq!(
            phi_incoming(&blocks[block_index_by_label(blocks, 8).unwrap()].instructions[0]),
            [(Operand::IdRef(60), 6), (Operand::IdRef(72), 3)]
        );
        assert!(!funnel_selection_merge_bypasses(&mut ctx, 0));
    }

    #[test]
    fn shared_short_circuit_arm_is_cloned_through_inner_merge() {
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(100));
        module.functions.push(Function {
            def: None,
            parameters: vec![],
            blocks: vec![
                selection(1, 90, 4, 2, 6),
                selection(2, 91, 4, 3, 5),
                branch(3, 5),
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(4), vec![])),
                    instructions: vec![
                        Instruction::new(
                            Op::CopyObject,
                            Some(50),
                            Some(40),
                            vec![Operand::IdRef(92)],
                        ),
                        Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(6)]),
                    ],
                },
                branch(5, 6),
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(6), vec![])),
                    instructions: vec![
                        Instruction::new(
                            Op::Phi,
                            Some(50),
                            Some(60),
                            vec![
                                Operand::IdRef(40),
                                Operand::IdRef(4),
                                Operand::IdRef(93),
                                Operand::IdRef(5),
                            ],
                        ),
                        Instruction::new(Op::Return, None, None, vec![]),
                    ],
                },
            ],
            end: None,
        });
        let mut ctx = Ctx::new(module);

        assert!(privatize_shared_direct_selection_arms(&mut ctx, 0));
        let blocks = &ctx.module.functions[0].blocks;
        let inner = &blocks[block_index_by_label(blocks, 2).unwrap()];
        assert!(!block_successors(inner).contains(&4));
        let inner_merge = &blocks[block_index_by_label(blocks, 5).unwrap()];
        let carried = inner_merge
            .instructions
            .first()
            .filter(|instruction| instruction.class.opcode == Op::Phi)
            .expect("cloned arm value is carried through the inner merge");
        let carried_result = carried.result_id.unwrap();
        let outer_merge = &blocks[block_index_by_label(blocks, 6).unwrap()];
        let outer_phi = &outer_merge.instructions[0];
        assert!(outer_phi
            .operands
            .chunks_exact(2)
            .any(|pair| { pair == [Operand::IdRef(carried_result), Operand::IdRef(5)] }));
        assert!(blocks.iter().any(|block| {
            let label = block.label.as_ref().and_then(|label| label.result_id);
            label.is_some_and(|label| label >= 100) && block_successors(block).as_slice() == [5]
        }));
    }

    #[test]
    fn reused_nested_merge_targets_get_phi_aware_private_merges() {
        let shared = 6;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(1_000));
        module.functions.push(Function {
            def: None,
            parameters: vec![],
            blocks: vec![
                selection(1, 100, 2, 5, shared),
                selection(2, 101, 3, 4, shared),
                branch(3, shared),
                branch(4, shared),
                branch(5, shared),
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(shared), vec![])),
                    instructions: vec![
                        Instruction::new(
                            Op::Phi,
                            Some(200),
                            Some(201),
                            vec![
                                Operand::IdRef(301),
                                Operand::IdRef(3),
                                Operand::IdRef(302),
                                Operand::IdRef(4),
                                Operand::IdRef(303),
                                Operand::IdRef(5),
                            ],
                        ),
                        Instruction::new(Op::Return, None, None, vec![]),
                    ],
                },
            ],
            end: None,
        });
        let mut ctx = Ctx::new(module);

        split_reused_merge_targets(&mut ctx, 0);

        let blocks = &ctx.module.functions[0].blocks;
        let merge_targets = blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .filter(|instruction| instruction.class.opcode == Op::SelectionMerge)
            .filter_map(|instruction| instruction.operands.first().and_then(id_ref_operand))
            .collect::<Vec<_>>();
        assert_eq!(merge_targets.len(), 2);
        assert_ne!(merge_targets[0], merge_targets[1]);
        assert!(!merge_targets.contains(&shared));

        let predecessors = block_predecessors(blocks);
        for block in blocks {
            let Some(label) = block.label.as_ref().and_then(|label| label.result_id) else {
                continue;
            };
            let actual = predecessors
                .get(&label)
                .into_iter()
                .flatten()
                .copied()
                .collect::<HashSet<_>>();
            for phi in block
                .instructions
                .iter()
                .take_while(|instruction| instruction.class.opcode == Op::Phi)
            {
                let incoming = phi
                    .operands
                    .chunks(2)
                    .filter_map(|pair| match pair {
                        [_, Operand::IdRef(predecessor)] => Some(*predecessor),
                        _ => None,
                    })
                    .collect::<HashSet<_>>();
                assert_eq!(incoming, actual, "phi predecessor set for block {label}");
            }
        }
    }

    #[test]
    fn reused_merge_that_is_loop_header_keeps_its_back_edge() {
        let loop_header = 6;
        let loop_body = 7;
        let loop_exit = 8;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(1_000));
        module.functions.push(Function {
            def: None,
            parameters: vec![],
            blocks: vec![
                selection(1, 100, 2, 5, loop_header),
                selection(2, 101, 3, 4, loop_header),
                branch(3, loop_header),
                branch(4, loop_header),
                branch(5, loop_header),
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(loop_header), vec![])),
                    instructions: vec![
                        Instruction::new(
                            Op::LoopMerge,
                            None,
                            None,
                            vec![Operand::IdRef(loop_exit), Operand::IdRef(loop_body)],
                        ),
                        Instruction::new(
                            Op::BranchConditional,
                            None,
                            None,
                            vec![
                                Operand::IdRef(102),
                                Operand::IdRef(loop_body),
                                Operand::IdRef(loop_exit),
                            ],
                        ),
                    ],
                },
                branch(loop_body, loop_header),
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(loop_exit), vec![])),
                    instructions: vec![Instruction::new(Op::Return, None, None, vec![])],
                },
            ],
            end: None,
        });
        let mut ctx = Ctx::new(module);

        split_reused_merge_targets(&mut ctx, 0);

        let blocks = &ctx.module.functions[0].blocks;
        let body = &blocks[block_index_by_label(blocks, loop_body).unwrap()];
        assert_eq!(block_successors(body), [loop_header]);
        let selection_targets = blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .filter(|instruction| instruction.class.opcode == Op::SelectionMerge)
            .filter_map(|instruction| instruction.operands.first().and_then(id_ref_operand))
            .collect::<Vec<_>>();
        assert_eq!(selection_targets.len(), 2);
        assert_ne!(selection_targets[0], selection_targets[1]);
        assert!(!selection_targets.contains(&loop_header));
    }

    #[test]
    fn inner_merge_gets_private_boundary_before_enclosing_continue() {
        let entry = 1;
        let outer = 2;
        let inner = 3;
        let outer_continue = 4;
        let outer_exit = 5;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(100));
        module.functions.push(Function {
            def: None,
            parameters: vec![],
            blocks: vec![
                branch(entry, outer),
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(outer), vec![])),
                    instructions: vec![
                        Instruction::new(
                            Op::LoopMerge,
                            None,
                            None,
                            vec![Operand::IdRef(outer_exit), Operand::IdRef(outer_continue)],
                        ),
                        Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(inner)]),
                    ],
                },
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(inner), vec![])),
                    instructions: vec![
                        Instruction::new(
                            Op::LoopMerge,
                            None,
                            None,
                            vec![Operand::IdRef(outer_continue), Operand::IdRef(inner)],
                        ),
                        Instruction::new(
                            Op::BranchConditional,
                            None,
                            None,
                            vec![
                                Operand::IdRef(20),
                                Operand::IdRef(outer_continue),
                                Operand::IdRef(inner),
                            ],
                        ),
                    ],
                },
                Block {
                    label: Some(Instruction::new(
                        Op::Label,
                        None,
                        Some(outer_continue),
                        vec![],
                    )),
                    instructions: vec![Instruction::new(
                        Op::BranchConditional,
                        None,
                        None,
                        vec![
                            Operand::IdRef(21),
                            Operand::IdRef(outer),
                            Operand::IdRef(outer_exit),
                        ],
                    )],
                },
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(outer_exit), vec![])),
                    instructions: vec![Instruction::new(Op::Return, None, None, vec![])],
                },
            ],
            end: None,
        });
        let mut ctx = Ctx::new(module);

        split_merges_that_are_enclosing_continues(&mut ctx, 0);

        let blocks = &ctx.module.functions[0].blocks;
        let inner_block = &blocks[block_index_by_label(blocks, inner).unwrap()];
        let private_merge = inner_block
            .instructions
            .iter()
            .find(|instruction| instruction.class.opcode == Op::LoopMerge)
            .and_then(|instruction| instruction.operands.first())
            .and_then(id_ref_operand)
            .unwrap();
        assert_ne!(private_merge, outer_continue);
        assert_eq!(
            block_successors(&blocks[block_index_by_label(blocks, private_merge).unwrap()]),
            [outer_continue]
        );
        let outer_loop_merge = blocks[block_index_by_label(blocks, outer).unwrap()]
            .instructions
            .iter()
            .find(|instruction| instruction.class.opcode == Op::LoopMerge)
            .unwrap();
        assert_eq!(outer_loop_merge.operands[1], Operand::IdRef(outer_continue));
    }

    #[test]
    fn block_serialized_before_dominator_moves_after_it() {
        let entry = 1;
        let header = 2;
        let merge = 3;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(100));
        module.functions.push(Function {
            def: None,
            parameters: vec![],
            blocks: vec![
                branch(entry, header),
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(merge), vec![])),
                    instructions: vec![Instruction::new(Op::Return, None, None, vec![])],
                },
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(header), vec![])),
                    instructions: vec![
                        Instruction::new(
                            Op::LoopMerge,
                            None,
                            None,
                            vec![Operand::IdRef(merge), Operand::IdRef(header)],
                        ),
                        Instruction::new(
                            Op::BranchConditional,
                            None,
                            None,
                            vec![
                                Operand::IdRef(10),
                                Operand::IdRef(merge),
                                Operand::IdRef(header),
                            ],
                        ),
                    ],
                },
            ],
            end: None,
        });
        let mut ctx = Ctx::new(module);

        repair_dominator_block_order(&mut ctx, 0);

        let order = ctx.module.functions[0]
            .blocks
            .iter()
            .filter_map(|block| block.label.as_ref()?.result_id)
            .collect::<Vec<_>>();
        assert_eq!(order, [entry, header, merge]);
    }

    #[test]
    fn enclosing_selection_merges_after_loop_instead_of_at_continue() {
        let entry = 1;
        let loop_header = 2;
        let body = 3;
        let continue_label = 4;
        let loop_exit = 5;
        let function_exit = 6;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(100));
        module.functions.push(Function {
            def: None,
            parameters: vec![],
            blocks: vec![
                selection(entry, 10, loop_header, function_exit, continue_label),
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(loop_header), vec![])),
                    instructions: vec![
                        Instruction::new(
                            Op::LoopMerge,
                            None,
                            None,
                            vec![Operand::IdRef(loop_exit), Operand::IdRef(continue_label)],
                        ),
                        Instruction::new(
                            Op::BranchConditional,
                            None,
                            None,
                            vec![
                                Operand::IdRef(11),
                                Operand::IdRef(body),
                                Operand::IdRef(loop_exit),
                            ],
                        ),
                    ],
                },
                branch(body, continue_label),
                branch(continue_label, loop_header),
                branch(loop_exit, function_exit),
                Block {
                    label: Some(Instruction::new(
                        Op::Label,
                        None,
                        Some(function_exit),
                        vec![],
                    )),
                    instructions: vec![Instruction::new(Op::Return, None, None, vec![])],
                },
            ],
            end: None,
        });
        let mut ctx = Ctx::new(module);

        assert!(repair_one_continue_selection_merge_target(&mut ctx, 0));

        let selection_merge = ctx.module.functions[0].blocks[0]
            .instructions
            .iter()
            .find(|instruction| instruction.class.opcode == Op::SelectionMerge)
            .unwrap();
        assert_eq!(selection_merge.operands[0], Operand::IdRef(function_exit));
    }

    #[test]
    fn selection_target_with_backedge_gets_dedicated_loop_header() {
        let entry = 1;
        let target = 2;
        let body = 3;
        let exit = 4;
        let phi = 20;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(100));
        module.functions.push(Function {
            def: None,
            parameters: vec![],
            blocks: vec![
                branch(entry, target),
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(target), vec![])),
                    instructions: vec![
                        Instruction::new(
                            Op::Phi,
                            Some(30),
                            Some(phi),
                            vec![
                                Operand::IdRef(31),
                                Operand::IdRef(entry),
                                Operand::IdRef(32),
                                Operand::IdRef(body),
                            ],
                        ),
                        Instruction::new(
                            Op::SelectionMerge,
                            None,
                            None,
                            vec![Operand::IdRef(exit)],
                        ),
                        Instruction::new(
                            Op::BranchConditional,
                            None,
                            None,
                            vec![
                                Operand::IdRef(10),
                                Operand::IdRef(body),
                                Operand::IdRef(exit),
                            ],
                        ),
                    ],
                },
                branch(body, target),
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(exit), vec![])),
                    instructions: vec![Instruction::new(Op::Return, None, None, vec![])],
                },
            ],
            end: None,
        });
        let mut ctx = Ctx::new(module);

        repair_unmarked_natural_loops(&mut ctx, 0);

        let blocks = &ctx.module.functions[0].blocks;
        let wrapper = block_successors(&blocks[0])[0];
        assert_ne!(wrapper, target);
        let wrapper_block = &blocks[block_index_by_label(blocks, wrapper).unwrap()];
        assert_eq!(wrapper_block.instructions[0].class.opcode, Op::Phi);
        assert_eq!(wrapper_block.instructions[0].result_id, Some(phi));
        let loop_merge = wrapper_block
            .instructions
            .iter()
            .find(|instruction| instruction.class.opcode == Op::LoopMerge)
            .unwrap();
        assert_eq!(loop_merge.operands[0], Operand::IdRef(exit));
        assert_eq!(loop_merge.operands[1], Operand::IdRef(body));
        assert_eq!(block_successors(wrapper_block), [target]);
        assert_eq!(
            block_successors(&blocks[block_index_by_label(blocks, body).unwrap()]),
            [wrapper]
        );
        let target_block = &blocks[block_index_by_label(blocks, target).unwrap()];
        assert_ne!(target_block.instructions[0].class.opcode, Op::Phi);
        assert_eq!(
            target_block.instructions[0].class.opcode,
            Op::SelectionMerge
        );
    }

    #[test]
    fn serialized_backward_irreducible_edge_is_not_wrapped_as_natural_loop() {
        let entry = 1;
        let target = 2;
        let body = 3;
        let exit = 4;
        let conditional = |label, condition, yes, no| Block {
            label: Some(Instruction::new(Op::Label, None, Some(label), vec![])),
            instructions: vec![Instruction::new(
                Op::BranchConditional,
                None,
                None,
                vec![
                    Operand::IdRef(condition),
                    Operand::IdRef(yes),
                    Operand::IdRef(no),
                ],
            )],
        };
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(100));
        module.functions.push(Function {
            def: None,
            parameters: vec![],
            blocks: vec![
                conditional(entry, 10, target, body),
                branch(target, body),
                conditional(body, 11, target, exit),
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(exit), vec![])),
                    instructions: vec![Instruction::new(Op::Return, None, None, vec![])],
                },
            ],
            end: None,
        });
        let mut ctx = Ctx::new(module);

        assert!(!repair_unmarked_natural_loops(&mut ctx, 0));
        assert_eq!(ctx.module.functions[0].blocks.len(), 4);
        assert!(ctx.module.functions[0].blocks.iter().all(|block| block
            .instructions
            .iter()
            .all(|instruction| instruction.class.opcode != Op::LoopMerge)));
    }

    #[test]
    fn early_exit_arms_extend_to_their_common_loop_merge() {
        let entry = 1;
        let target = 2;
        let body = 3;
        let early_a = 4;
        let early_b = 5;
        let continue_label = 6;
        let merge = 7;
        let dead = 8;
        let conditional = |label, condition, yes, no| Block {
            label: Some(Instruction::new(Op::Label, None, Some(label), vec![])),
            instructions: vec![Instruction::new(
                Op::BranchConditional,
                None,
                None,
                vec![
                    Operand::IdRef(condition),
                    Operand::IdRef(yes),
                    Operand::IdRef(no),
                ],
            )],
        };
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(100));
        module.functions.push(Function {
            def: None,
            parameters: vec![],
            blocks: vec![
                branch(entry, target),
                conditional(target, 20, body, early_a),
                conditional(body, 21, continue_label, early_b),
                branch(early_a, merge),
                branch(early_b, merge),
                conditional(continue_label, 22, target, dead),
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(dead), vec![])),
                    instructions: vec![Instruction::new(Op::Unreachable, None, None, vec![])],
                },
                Block {
                    label: Some(Instruction::new(Op::Label, None, Some(merge), vec![])),
                    instructions: vec![Instruction::new(Op::Return, None, None, vec![])],
                },
            ],
            end: None,
        });
        let mut ctx = Ctx::new(module);

        repair_unmarked_natural_loops(&mut ctx, 0);

        let blocks = &ctx.module.functions[0].blocks;
        let wrapper = block_successors(&blocks[0])[0];
        assert_ne!(wrapper, target);
        let wrapper_block = &blocks[block_index_by_label(blocks, wrapper).unwrap()];
        let loop_merge = wrapper_block
            .instructions
            .iter()
            .find(|instruction| instruction.class.opcode == Op::LoopMerge)
            .unwrap();
        assert_eq!(loop_merge.operands[0], Operand::IdRef(merge));
        assert_eq!(loop_merge.operands[1], Operand::IdRef(continue_label));
        assert_eq!(
            block_successors(&blocks[block_index_by_label(blocks, continue_label).unwrap()]),
            [wrapper, dead]
        );
    }
}
