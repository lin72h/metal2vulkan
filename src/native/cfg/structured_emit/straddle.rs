//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

/// The base structured-plan derivation (loop merges + selection merges + order + self-checks), without
/// the trivial-cross-arm pre-pass. [`structured_plan`] runs it once as-is, then once on the privatized
/// graph if the first rejects.
/// Thin wrapper: the base ladder attempts run WITHOUT the multi-exit shared-exit region clone (so a
/// function that already admits stays byte-identical). Only the reject-triggered clone attempt calls
/// [`structured_plan_inner4`] with `multi_exit_clone = true`.
pub(in crate::native) fn structured_plan_inner(
    blocks: &[BodyBlock],
    converge_inloop: bool,
    break_aware: bool,
) -> Option<StructuredPlan> {
    structured_plan_inner4(blocks, converge_inloop, break_aware, false, false)
}

pub(in crate::native) fn structured_plan_inner4(
    blocks: &[BodyBlock],
    converge_inloop: bool,
    break_aware: bool,
    multi_exit_clone: bool,
    allow_bare_exit: bool,
) -> Option<StructuredPlan> {
    structured_plan_inner5(
        blocks,
        converge_inloop,
        break_aware,
        multi_exit_clone,
        allow_bare_exit,
        false,
    )
}

/// Variant of [`structured_plan_inner4`] used only by the reject-triggered loop-exit-selection tier.
/// `loop_exit_selection` lets selection planning retain a loop-local convergence block when every
/// alternative path is a checked break/continue out of the enclosing loop.
pub(in crate::native) fn structured_plan_inner5(
    blocks: &[BodyBlock],
    converge_inloop: bool,
    break_aware: bool,
    multi_exit_clone: bool,
    allow_bare_exit: bool,
    loop_exit_selection: bool,
) -> Option<StructuredPlan> {
    structured_plan_inner6(
        blocks,
        converge_inloop,
        break_aware,
        multi_exit_clone,
        allow_bare_exit,
        loop_exit_selection,
        false,
    )
}

/// Extended planner variant used only by the reject-triggered terminal-return attempt. It lets that
/// attempt supply explicit private selection merges for branches whose ordinary post-dominator is a
/// shared function return; every ordinary caller keeps the byte-identical `false` path above.
pub(in crate::native) fn structured_plan_inner6(
    blocks: &[BodyBlock],
    converge_inloop: bool,
    break_aware: bool,
    multi_exit_clone: bool,
    allow_bare_exit: bool,
    loop_exit_selection: bool,
    terminal_exit_selection: bool,
) -> Option<StructuredPlan> {
    structured_plan_inner7(
        blocks,
        converge_inloop,
        break_aware,
        multi_exit_clone,
        allow_bare_exit,
        loop_exit_selection,
        terminal_exit_selection,
        false,
    )
}

/// Complete a reject-only construct-tree candidate with the ordinary loop/selection maps while
/// retaining the ownership proof supplied by the tree. The tree has already separated the cross-arm
/// boundary that made the source CFG reject; source-CFG dominance can still report an enclosing loop
/// exit as `cross-arm-shared`, even though that exit is a preserved construct-tree route. Final
/// adoption remains module-level `spirv-val` gated.
pub(in crate::native) fn structured_plan_construct_tree(
    blocks: &[BodyBlock],
) -> Option<StructuredPlan> {
    // Prefer the immutable whole-CFG ownership proof when it already captures the graph without
    // loop-exit convergence. Besides avoiding needless synthetic roles, this keeps an outer
    // selection from being stretched across a loop that has entries in both of its arms.
    if let Some(plan) =
        structured_plan_inner7(blocks, false, false, false, true, false, false, true)
    {
        return Some(plan);
    }

    for (converge_inloop, break_aware) in [(true, false), (true, true), (false, false)] {
        if let Some(plan) = structured_plan_inner7(
            blocks,
            converge_inloop,
            break_aware,
            false,
            true,
            true,
            false,
            true,
        ) {
            return Some(plan);
        }
    }

    if let Some(cloned) = privatize_direct_construct_tree_cross_arm(blocks) {
        let shared_private =
            privatize_direct_construct_tree_shared_continuations(&cloned).unwrap_or(cloned);
        for &(converge_inloop, break_aware) in &[(false, false), (true, false), (true, true)] {
            if let Some(plan) = structured_plan_inner7(
                &shared_private,
                converge_inloop,
                break_aware,
                false,
                true,
                false,
                false,
                true,
            ) {
                return Some(plan);
            }
        }
    }
    None
}

fn privatize_direct_construct_tree_cross_arm(blocks: &[BodyBlock]) -> Option<Vec<BodyBlock>> {
    const DIRECT_CROSS_ARM_ROUNDS: usize = 24;
    const DIRECT_CROSS_ARM_GROWTH_CAP: usize = 3000;
    let mut cur = blocks.to_vec();
    let mut changed = false;
    let mut counter = 7_000_000usize;
    let max_blocks = blocks.len().saturating_add(DIRECT_CROSS_ARM_GROWTH_CAP);
    for round in 0..DIRECT_CROSS_ARM_ROUNDS {
        let Some((header, target)) = find_direct_construct_tree_cross_arm(&cur, round > 0) else {
            break;
        };
        let Some(next) =
            clone_crossarm::privatize_dominated_region(&cur, &header, &target, &mut counter)
        else {
            break;
        };
        if next.len() > max_blocks {
            break;
        }
        cur = next;
        changed = true;
    }
    changed.then_some(cur)
}

fn privatize_direct_construct_tree_shared_continuations(
    blocks: &[BodyBlock],
) -> Option<Vec<BodyBlock>> {
    const DIRECT_SHARED_ROUNDS: usize = 64;
    const DIRECT_SHARED_GROWTH_CAP: usize = 4_000;
    let mut cur = blocks.to_vec();
    let mut changed = false;
    let mut counter = 7_500_000usize;
    let max_blocks = blocks.len().saturating_add(DIRECT_SHARED_GROWTH_CAP);
    for _ in 0..DIRECT_SHARED_ROUNDS {
        let mut next = None;
        for (header, continuation) in clone_crossarm::find_deep_shared_continuations(&cur) {
            let Some(cloned) = clone_crossarm::privatize_dominated_region(
                &cur,
                &header,
                &continuation,
                &mut counter,
            ) else {
                continue;
            };
            if cloned.len() > max_blocks {
                continue;
            }
            next = Some(cloned);
            break;
        }
        let Some(cloned) = next else {
            break;
        };
        cur = cloned;
        changed = true;
    }
    changed.then_some(cur)
}

fn find_direct_construct_tree_cross_arm(
    blocks: &[BodyBlock],
    reverse: bool,
) -> Option<(String, String)> {
    let forest = analyze(blocks);
    let loop_headers: HashSet<&str> = forest
        .loops
        .iter()
        .map(|loop_info| loop_info.header.as_str())
        .collect();
    let by_name = blocks
        .iter()
        .map(|block| (block.name.as_str(), block))
        .collect::<HashMap<_, _>>();
    let names = blocks
        .iter()
        .map(|block| block.name.as_str())
        .collect::<HashSet<_>>();
    let indices: Box<dyn Iterator<Item = usize>> = if reverse {
        Box::new((0..blocks.len()).rev())
    } else {
        Box::new(0..blocks.len())
    };
    for index in indices {
        let block = &blocks[index];
        if loop_headers.contains(block.name.as_str()) {
            continue;
        }
        let Some((true_target, false_target)) = conditional_branch_targets(block) else {
            continue;
        };
        for target in [true_target, false_target] {
            if !names.contains(target.as_str()) || forest.dominates(&block.name, &target) {
                continue;
            }
            let mut child = block.name.as_str();
            while let Some(parent) = forest.idom(child) {
                if loop_headers.contains(parent) {
                    child = parent;
                    continue;
                }
                if let Some(parent_block) = by_name.get(parent) {
                    if let Some((left, right)) = conditional_branch_targets(parent_block) {
                        let sibling = if child == left {
                            Some(right)
                        } else if child == right {
                            Some(left)
                        } else {
                            None
                        };
                        if sibling.is_some_and(|sibling| forest.dominates(&sibling, &target)) {
                            return Some((block.name.clone(), target));
                        }
                    }
                }
                child = parent;
            }
        }
    }
    None
}

fn structured_plan_inner7(
    blocks: &[BodyBlock],
    converge_inloop: bool,
    break_aware: bool,
    multi_exit_clone: bool,
    allow_bare_exit: bool,
    loop_exit_selection: bool,
    terminal_exit_selection: bool,
    construct_tree_owned: bool,
) -> Option<StructuredPlan> {
    structured_plan_inner8(
        blocks,
        converge_inloop,
        break_aware,
        multi_exit_clone,
        allow_bare_exit,
        loop_exit_selection,
        terminal_exit_selection,
        construct_tree_owned,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(in crate::native) fn structured_plan_inner8(
    blocks: &[BodyBlock],
    converge_inloop: bool,
    break_aware: bool,
    multi_exit_clone: bool,
    allow_bare_exit: bool,
    loop_exit_selection: bool,
    terminal_exit_selection: bool,
    construct_tree_owned: bool,
    prepared_terminal: Option<&TerminalExitSelectionPlan>,
) -> Option<StructuredPlan> {
    let spi = crate::env_vars::spi_why();
    let tag = blocks.first().map(|b| b.name.clone()).unwrap_or_default();
    macro_rules! spi_reject {
        ($why:expr) => {
            if spi {
                eprintln!(
                    "[spi-why] fn0={} nblk={} converge={} break_aware={} loop_exit={} REJECT {}",
                    tag,
                    blocks.len(),
                    converge_inloop,
                    break_aware,
                    loop_exit_selection,
                    $why
                );
            }
        };
    }
    // A terminal-prefix guard may share its source return with the only exit of a later simple
    // loop. Give that loop a private return first: the prefix can then structure its own early
    // exits without claiming the loop's merge, and the loop retains a private, ordinary merge.
    // This is only a seed for the terminal retry; a seed with no terminal plan is discarded below.
    let terminal_seed = if terminal_exit_selection && prepared_terminal.is_none() {
        privatize_single_loop_return_exit(blocks)
    } else {
        None
    };
    let terminal_input = terminal_seed.as_deref().unwrap_or(blocks);
    let computed_terminal = if terminal_exit_selection && prepared_terminal.is_none() {
        terminal_exit_selection_merges(terminal_input)
    } else {
        None
    };
    let terminal = prepared_terminal.or(computed_terminal.as_ref());
    let terminal_blocks = terminal
        .map(|plan| plan.blocks.as_slice())
        .unwrap_or(terminal_input);
    let (base_lblocks, mut loop_merges) =
        forest_loop_merges(terminal_blocks, converge_inloop, multi_exit_clone);
    let terminal_dispatch = if terminal_exit_selection {
        terminal_unreachable_selection_merges(&base_lblocks)
    } else {
        None
    };
    if terminal_exit_selection && terminal.is_none() && terminal_dispatch.is_none() {
        spi_reject!("terminal-exit-no-candidate");
        return None;
    }
    let mut lblocks = terminal_dispatch
        .as_ref()
        .map(|plan| plan.blocks.clone())
        .unwrap_or(base_lblocks);
    let mut terminal_merges = HashMap::new();
    if let Some(plan) = terminal {
        terminal_merges.extend(plan.merges.clone());
    }
    if let Some(plan) = &terminal_dispatch {
        terminal_merges.extend(plan.merges.clone());
    }
    // The general terminal planner is intentionally bounded to modest CFGs, but a direct
    // `guard -> {continuation, ret}` is a local ownership relation. Compose those guards into the
    // construct-tree candidate with two edge splits each, so large generated functions do not fall
    // through to the enclosing-region repair and mistake the shared return for an ordinary merge.
    if construct_tree_owned {
        if let Some(plan) = direct_terminal_exit_selection_merges(&lblocks, &terminal_merges) {
            lblocks = plan.blocks;
            terminal_merges.extend(plan.merges);
        }
        coalesce_sibling_conditional_dispatches(&mut lblocks);
        // Regional ownership can add an enclosing predecessor to a loop's former exit after the
        // loop forest first selected it. Privatize that merge before selection synthesis so every
        // downstream role/collision map sees the final loop edge and cannot retain a stale key.
        privatize_nondominated_loop_merges(&mut lblocks, &mut loop_merges);
    }
    // Completeness: every natural loop must be covered by the forest loop-merge map (directly-
    // structurable, merge==continue split, or the narrow pure-self-latch NoExit split); every other
    // MultipleLatches/MultipleExits/NoExit shape still falls back.
    let lforest = analyze(&lblocks);
    for l in &lforest.loops {
        if !loop_merges.contains_key(&l.header) {
            if crate::env_vars::spi_why() {
                eprintln!(
                    "[spi-why]   uncovered-loop header={} latches={:?} exits={:?} parent={:?}",
                    l.header, l.latches, l.exits, l.parent,
                );
            }
            spi_reject!(format!("loop-uncovered header={}", l.header));
            return None;
        }
    }
    let (mut sblocks, mut branch, mut branch_merges_by_header, switch) = if construct_tree_owned {
        unique_selection_merges_with_construct_tree_ownership(
            &lblocks,
            &loop_merges,
            break_aware,
            loop_exit_selection,
            &terminal_merges,
        )
    } else {
        unique_selection_merges_with_loop_exit_and_forced(
            &lblocks,
            &loop_merges,
            break_aware,
            loop_exit_selection,
            &terminal_merges,
        )
    };
    if !construct_tree_owned {
        index_branch_merges_by_header(
            &sblocks,
            &loop_merges,
            &branch,
            &mut branch_merges_by_header,
        );
    }
    normalize_continue_selection_merge_targets(
        &mut sblocks,
        &loop_merges,
        &mut branch_merges_by_header,
    );
    if !construct_tree_owned {
        branch = sblocks
            .iter()
            .filter_map(|block| {
                let (true_target, false_target) = conditional_branch_targets(block)?;
                let merge = branch_merges_by_header.get(&block.name)?;
                Some(((true_target, false_target), merge.clone()))
            })
            .collect();
    }
    if selection_synth_growth_exceeds_ladder_cap(lblocks.len(), sblocks.len()) {
        spi_reject!(format!(
            "selection-synth-growth nblk={} from={}",
            sblocks.len(),
            lblocks.len()
        ));
        return None;
    }

    // Every conditional/switch header must have a unique merge (else module 2 skipped it → fall back).
    // Build header→merge alongside for the structured ordering.
    let forest = analyze(&sblocks);
    let loop_headers: HashSet<&str> = forest.loops.iter().map(|l| l.header.as_str()).collect();
    let names: HashSet<&str> = sblocks.iter().map(|b| b.name.as_str()).collect();
    let mut header_merge: HashMap<String, String> = HashMap::new();
    for (h, info) in &loop_merges {
        header_merge.insert(h.clone(), info.merge.clone());
    }
    // Set when the bare-loop-exit skip fires (see `bare_loop_exit_branch`): such a block is left out of
    // `header_merge`, so the completeness gate no longer forces it to reject. That skip can UNMASK a
    // latent unstructured exit elsewhere in the same function (a block that escapes its selection to a
    // non-merge, non-role block — which the base plan otherwise never reaches because it rejects at the
    // now-skipped block first). When any skip fired we re-verify the whole plan with
    // `bare_exit_escape_reason` before admitting.
    let mut skipped_bare_exit = false;
    for b in &sblocks {
        let is_switch = is_switch_block(b);
        if loop_headers.contains(b.name.as_str()) {
            // A loop header that also ends in a `switch` would need both OpLoopMerge AND
            // OpSelectionMerge on one block (illegal SPIR-V); the switch-merge gate below skips loop
            // headers, so `unique_selection_merges` never assigns it a switch merge and the switch emitter
            // aborts with "could not infer structured merge for switch". `split_loop_header_switch`
            // (in `forest_loop_merges`) already lifts the GENUINE in-loop case — every switch target an
            // in-loop block — into a fresh successor, so what reaches here is the residue whose switch
            // targets the loop's merge/continue (a `while`-style switch exit test, banked a private capture shard
            // `423ff479`); that residue has no split yet, so reject and fall back to the relooper retry.
            if is_switch {
                spi_reject!(format!("loop-header-switch header={}", b.name));
                return None;
            }
            continue;
        }
        if is_switch {
            match switch.get(&b.name) {
                Some(m) => header_merge.insert(b.name.clone(), m.clone()),
                None => {
                    spi_reject!(format!("switch-no-merge header={}", b.name));
                    return None;
                }
            };
            continue;
        }
        let succs = block_successors(b);
        let distinct: HashSet<&str> = succs
            .iter()
            .map(String::as_str)
            .filter(|t| names.contains(t))
            .collect();
        if distinct.len() < 2 {
            continue;
        }
        let Some((t, f)) = conditional_branch_targets(b) else {
            spi_reject!(format!("no-cond-targets header={}", b.name));
            return None;
        };
        if construct_tree_owned
            && allow_bare_exit
            && (bare_loop_exit_branch_with_passthroughs(
                &sblocks,
                &forest,
                &loop_merges,
                &b.name,
                &t,
                &f,
            ) || bare_enclosing_selection_region_escape(
                &sblocks,
                &forest,
                &branch_merges_by_header,
                &b.name,
                &t,
            ) || bare_enclosing_selection_region_escape(
                &sblocks,
                &forest,
                &branch_merges_by_header,
                &b.name,
                &f,
            ))
        {
            // The construct-tree candidate already owns inter-construct routes. A branch whose arm is
            // an enclosing loop role (merge/continue), or a sibling arm of an enclosing construct-tree
            // selection, is a legal bare structured exit/backedge inside that enclosing construct.
            // Keeping a synthesized OpSelectionMerge for it makes the branch exit the nested selection
            // through a non-merge target. Drop any pre-registered merge for this final arm pair so the
            // emitter leaves the branch bare.
            branch.remove(&(t.clone(), f.clone()));
            branch_merges_by_header.remove(&b.name);
            skipped_bare_exit = true;
            continue;
        }
        let merge = match branch_merges_by_header
            .get(&b.name)
            .or_else(|| branch.get(&(t.clone(), f.clone())))
        {
            Some(m) => m.clone(),
            None => {
                // Completeness fallback: a structural break/continue latch may have been missed by
                // the selection-merge map (no post-idom) yet still be a legal `{merge, continue}`
                // shape after multi-exit + do-while. Re-check against the final sblocks terminator.
                if let Some(cont) =
                    loop_break_continue_merge(&forest, &loop_merges, &b.name, &t, &f)
                {
                    cont
                } else if allow_bare_exit
                    && (bare_loop_exit_branch_with_passthroughs(
                        &sblocks,
                        &forest,
                        &loop_merges,
                        &b.name,
                        &t,
                        &f,
                    ) || bare_enclosing_selection_region_escape(
                        &sblocks,
                        &forest,
                        &branch_merges_by_header,
                        &b.name,
                        &t,
                    ) || bare_enclosing_selection_region_escape(
                        &sblocks,
                        &forest,
                        &branch_merges_by_header,
                        &b.name,
                        &f,
                    ))
                {
                    // A bare structured break/continue or enclosing-selection sibling exit: no
                    // OpSelectionMerge is needed. Skip the block (leave it out of `header_merge` → the
                    // emitter writes a bare OpBranchConditional; self-checks 2/3 also skip it).
                    skipped_bare_exit = true;
                    continue;
                } else if construct_tree_owned && allow_bare_exit {
                    // The construct tree is the ownership proof for inter-construct divergent
                    // routes. A conditional left without an ordinary post-dominator after all
                    // local merge/loop-role checks is a tree exit, not an incomplete local
                    // selection. Emit it bare; this retry-only candidate is adopted only after the
                    // finished module independently passes spirv-val.
                    skipped_bare_exit = true;
                    continue;
                } else {
                    spi_reject!(format!(
                        "branch-no-merge header={} arms=({},{})",
                        b.name, t, f
                    ));
                    if spi && sblocks.len() <= CROSS_ARM_EDGE_MAX_BLOCKS {
                        eprintln!("[spi-why]   --- sblocks skeleton (name -> terminator) ---");
                        for sb in &sblocks {
                            // Diagnostics-only skeleton print: source the phi count + terminator from the
                            // typed carrier (populated for every production block) instead of `.lines`.
                            let (phi, term) = sb
                                .typed
                                .as_ref()
                                .map(|t| {
                                    (
                                        t.insts.iter().filter(|i| i.is_phi()).count(),
                                        format!("{:?}", t.terminator),
                                    )
                                })
                                .unwrap_or((0, String::new()));
                            eprintln!("[spi-why]   {:24} phi={} | {}", sb.name, phi, term);
                        }
                        eprintln!(
                            "[spi-why]   branch keys: {:?}",
                            branch.keys().collect::<Vec<_>>()
                        );
                        eprintln!("[spi-why]   loop_merges: {loop_merges:?}");
                    }
                    return None;
                }
            }
        };
        header_merge.insert(b.name.clone(), merge);
    }

    // Module 1: structured order (dominator-tree preorder, merge-last). Dominance-respecting, so it
    // also keeps every definition before its uses (subsuming the forward-local-def reorder). This is
    // required for the repair-FREE path — without the repair, the block order must itself be valid.
    let order = if terminal_exit_selection {
        structured_order_terminal(&sblocks, &forest, |h| header_merge.get(h).cloned())
    } else {
        structured_order(&sblocks, &forest, |h| header_merge.get(h).cloned())
    };
    let rank: HashMap<&str, usize> = order
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    let mut ordered = sblocks;
    ordered.sort_by_key(|b| rank.get(b.name.as_str()).copied().unwrap_or(usize::MAX));

    // Plan self-checks: reject straddling loop merges + cross-arm shared-convergence so admission is
    // honest (see `plan_self_check_reason`). Shared with `structured_reject_reason` so the diagnostic
    // never drifts from the emission gate.
    if let Some(reason) = plan_self_check_reason(&ordered, &header_merge, &loop_merges) {
        let tree_owned_residual = construct_tree_owned
            && matches!(
                reason,
                "selection:cross-arm-shared" | "selection:straddle-loop-merge"
            );
        if !tree_owned_residual {
            spi_reject!(format!("self-check {reason}"));
            return None;
        }
    }

    // When a bare-loop-exit skip fired, re-verify that NO block escapes its construct to a non-merge,
    // non-role block (the residual unstructured exit self-checks 2/3 conservatively skip, unmasked by
    // the skip). Scoped to skip-affected functions so the primary emit stays byte-identical.
    if construct_tree_owned {
        if let Some(reason) = dominance_loop_exit_escape_reason(&ordered, &loop_merges) {
            spi_reject!(format!("self-check {reason}"));
            return None;
        }
    }
    if (skipped_bare_exit || loop_exit_selection) && !construct_tree_owned {
        if let Some(reason) = bare_exit_escape_reason(&ordered, &header_merge, &loop_merges) {
            spi_reject!(format!("self-check {reason}"));
            return None;
        }
    }
    if let Some(reason) = conflicting_phi_predecessor_reason(&ordered) {
        spi_reject!(format!("self-check {reason}"));
        return None;
    }

    Some(StructuredPlan {
        blocks: ordered,
        loop_merges,
        branch_merges: branch,
        branch_merges_by_header,
        switch_merges: switch,
    })
}

/// Reject conflicting incoming values after every edge split and merge synthesis. Repeated
/// byte-identical pairs canonicalize to one SPIR-V operand, but one predecessor cannot select two
/// different values.
fn conflicting_phi_predecessor_reason(blocks: &[BodyBlock]) -> Option<String> {
    for block in blocks {
        let Some(carrier) = &block.typed else {
            continue;
        };
        for inst in &carrier.insts {
            let Some((_, incoming)) = inst.phi_incoming() else {
                continue;
            };
            let mut value_by_predecessor = HashMap::new();
            for (value, predecessor) in incoming {
                if value_by_predecessor
                    .insert(predecessor.as_str(), value)
                    .is_some_and(|existing| existing != value)
                {
                    return Some(format!(
                        "phi-conflicting-predecessor block={} pred={predecessor}",
                        block.name
                    ));
                }
            }
        }
    }
    None
}

/// Restructure straddling loop merges into an admissible shape — the `selection:straddle-loop-merge`
/// reject class (`05/MPSRNNBreakUpToOutputVecs`: two enclosing early-return guards whose false arm is
/// the OpReturn block that also serves as the inner loop's exit merge, so the loop merge lands at/after
/// the guard's merge and the construct boundaries invert). Give each straddling loop its OWN merge by
/// inserting a pass-through block on its in-loop exit edges (loop → `SL` → shared merge `ML`): now `SL`
/// is the loop merge, dominated by the loop header (inside the construct), so `dominates(CM, SL)` is
/// false and the straddle self-check no longer fires. Returns the restructured blocks if at least one
/// straddle was split, else `None`.
///
/// Pure structural transform (adds pass-through blocks, redirects in-loop exit edges, preserves the
/// shared merge's phis via [`split_phi_overlap`]). It is one reject-only planner alternative and is
/// admitted through the same structural self-check as the ordinary plan.
pub(in crate::native) fn restructure_straddle_loop_merges(
    blocks: &[BodyBlock],
) -> Option<Vec<BodyBlock>> {
    restructure_straddle_loop_merges_with(blocks, crate::env_vars::converge_inloop(), false)
}

/// [`restructure_straddle_loop_merges`] with the straddle DETECTION driven under explicit
/// `converge_inloop`/`break_aware` flags (Keystone-2 M3). The default 6th attempt derives the
/// loop-merge / selection-merge maps on the ORIGINAL blocks (converge = the env override, break-aware
/// off), so a straddle that only MATERIALIZES once the converge/break-aware transforms split the
/// blocks (`04/7da629e9`) is invisible to it and it returns `None`. Passing `converge=true,
/// break_aware=true` derives those maps the same way `structured_plan_inner`'s 4th/5th attempts do, so
/// the derived straddle is detected. The pass-through split still lands on the ORIGINAL blocks (the
/// loop header + exit merge live there — the derived `sblocks` may have restructured the loop so its
/// exit no longer has the in-loop predecessors the split needs).
pub(in crate::native) fn restructure_straddle_loop_merges_with(
    blocks: &[BodyBlock],
    converge_inloop: bool,
    break_aware: bool,
) -> Option<Vec<BodyBlock>> {
    let (lblocks, loop_merges) = forest_loop_merges(blocks, converge_inloop, false);
    // Only act when every natural loop is forest-covered (the same completeness gate
    // `structured_plan_inner` uses); an uncovered loop needs a different restructure, leave it.
    let lforest = analyze(&lblocks);
    for l in &lforest.loops {
        if !loop_merges.contains_key(&l.header) {
            return None;
        }
    }
    let (sblocks, branch, switch) = unique_selection_merges(&lblocks, &loop_merges, break_aware);

    // Build header→declared-merge exactly as `structured_plan_inner` does, but best-effort (a header
    // with no assigned merge is simply skipped — it cannot participate in a straddle we can resolve).
    let forest = analyze(&sblocks);
    let loop_headers: HashSet<&str> = forest.loops.iter().map(|l| l.header.as_str()).collect();
    let names: HashSet<&str> = sblocks.iter().map(|b| b.name.as_str()).collect();
    let mut header_merge: HashMap<String, String> = HashMap::new();
    for (h, info) in &loop_merges {
        header_merge.insert(h.clone(), info.merge.clone());
    }
    for b in &sblocks {
        if loop_headers.contains(b.name.as_str()) {
            continue;
        }
        let is_switch = is_switch_block(b);
        if is_switch {
            if let Some(m) = switch.get(&b.name) {
                header_merge.insert(b.name.clone(), m.clone());
            }
            continue;
        }
        let succs = block_successors(b);
        let distinct: HashSet<&str> = succs
            .iter()
            .map(String::as_str)
            .filter(|t| names.contains(t))
            .collect();
        if distinct.len() < 2 {
            continue;
        }
        if let Some((t, f)) = conditional_branch_targets(b) {
            if let Some(m) = branch.get(&(t, f)) {
                header_merge.insert(b.name.clone(), m.clone());
            }
        }
    }

    // Detect straddling loops — the exact condition of `plan_self_check_reason` self-check 1.
    let mut straddles: Vec<(String, String)> = Vec::new();
    for l in &forest.loops {
        let Some(info) = loop_merges.get(&l.header) else {
            continue;
        };
        let ml = info.merge.clone();
        for (ch, cm) in &header_merge {
            if ch == &l.header {
                continue;
            }
            let inside = forest.dominates(ch, &l.header) && !forest.dominates(cm, &l.header);
            if inside && forest.dominates(cm, &ml) {
                straddles.push((l.header.clone(), ml));
                break;
            }
        }
    }
    if straddles.is_empty() {
        return None;
    }

    // Insert a pass-through loop-merge for each straddling loop, operating on the ORIGINAL blocks (the
    // loop header + its exit merge live there; the derived `sblocks` may have restructured the loop so
    // its exit no longer has the in-loop predecessors the split needs). Start the split counter past
    // the block count so the fresh `%metal2vulkan.lmerge.N` names cannot collide with anything.
    let mut out = blocks.to_vec();
    let mut counter = out.len();
    let mut changed = false;
    for (header, ml) in &straddles {
        let det_forest = analyze(&out);
        if det_forest.loop_for_header(header).is_none() || !out.iter().any(|b| &b.name == ml) {
            continue;
        }
        let split = if block_has_phi(&out, ml) {
            split_phi_overlap(&mut out, &det_forest, header, ml, &mut counter)
        } else {
            split_no_phi_overlap(&mut out, &det_forest, header, ml, &mut counter)
        };
        changed |= split.is_some();
    }
    changed.then_some(out)
}

/// Read-only diagnostic for `selection:straddle-loop-merge` rows. It mirrors the straddle self-check
/// across the same graph/mode variants the planner ladder uses, but never runs the expensive full emit
/// path. The important discriminator is whether the detected loop merge exists in the input graph the
/// existing splitter mutates: source AIR merge blocks can be split by
/// [`restructure_straddle_loop_merges_with`], while already-synthesized `%metal2vulkan.lmerge.*`
/// targets require a different ownership move.
pub(in crate::native) fn straddle_witness_lines(blocks: &[BodyBlock]) -> Vec<String> {
    let mut out = Vec::new();
    append_straddle_witness_modes("source", blocks, &mut out);
    if let Some(destraddled) = restructure_straddle_loop_merges(blocks) {
        let (region, region_witness) =
            clone_crossarm::privatize_region_cross_arm_with_witness(&destraddled);
        append_straddle_summary_line(
            "source-destraddled",
            &destraddled,
            Some(&region_witness),
            &mut out,
        );
        if blocks_changed(&destraddled, &region) {
            append_straddle_summary_line("source-destraddled-region", &region, None, &mut out);
        }
        append_straddle_witness_modes("source-destraddled", &destraddled, &mut out);
    }
    if let Some(destraddled) = restructure_straddle_loop_merges_with(blocks, true, true) {
        let (region, region_witness) =
            clone_crossarm::privatize_region_cross_arm_with_witness(&destraddled);
        append_straddle_summary_line(
            "source-derived-destraddled",
            &destraddled,
            Some(&region_witness),
            &mut out,
        );
        if blocks_changed(&destraddled, &region) {
            append_straddle_summary_line(
                "source-derived-destraddled-region",
                &region,
                None,
                &mut out,
            );
        }
        append_straddle_witness_modes("source-derived-destraddled", &destraddled, &mut out);
    }

    let deep_shared = privatize_shared_continuations_for_ladder(blocks);
    if blocks_changed(blocks, &deep_shared) {
        append_straddle_witness_modes("deep-shared", &deep_shared, &mut out);
    }

    let trivial = clone_crossarm::privatize_trivial_cross_arm(blocks);
    if blocks_changed(blocks, &trivial) {
        append_straddle_witness_modes("trivial", &trivial, &mut out);
    }

    let region = privatize_region_cross_arm_for_ladder(blocks);
    if blocks_changed(blocks, &region) {
        append_straddle_witness_modes("region", &region, &mut out);
    }

    let lowered_switches = blocks::lower_loop_exit_switches(blocks);
    if blocks_changed(blocks, &lowered_switches) {
        append_straddle_witness_modes("loop-switch-lowered", &lowered_switches, &mut out);
    }

    out.sort();
    out.dedup();
    out
}

fn append_straddle_summary_line(
    graph: &str,
    blocks: &[BodyBlock],
    region_fixpoint: Option<&clone_crossarm::RegionCrossArmFixpointWitness>,
    out: &mut Vec<String>,
) {
    let reason = super::reject::reject_reason_inner(blocks).unwrap_or_else(|| "ADMIT".to_string());
    let raw_clone = clone_crossarm::find_cross_arm(blocks)
        .map(|(header, arm)| clone_crossarm::dominated_region_clone_witness(blocks, &header, &arm));
    let synth_clone = first_synthesized_cross_arm_clone_witness(blocks);
    let raw_fields = clone_witness_fields("raw", raw_clone, None);
    let synth_fields = match synth_clone {
        Some((mode, witness)) => clone_witness_fields("synth", Some(witness), Some(mode)),
        None => clone_witness_fields("synth", None, None),
    };
    let region_fields = region_fixpoint
        .map(region_fixpoint_fields)
        .unwrap_or_default();
    out.push(format!(
        "graph={graph} blocks={} post_split_base_reason={} switch_gate={} {} {}{}",
        blocks.len(),
        reason,
        blocks_contain_multilevel_break_switch(blocks),
        raw_fields,
        synth_fields,
        region_fields,
    ));
}

fn first_synthesized_cross_arm_clone_witness(
    blocks: &[BodyBlock],
) -> Option<((bool, bool), clone_crossarm::DominatedRegionCloneWitness)> {
    for mode in [(false, false), (true, false), (true, true)] {
        let Some((header, arm)) = find_synthesized_cross_arm_shared(blocks, mode.0, mode.1) else {
            continue;
        };
        return Some((
            mode,
            clone_crossarm::dominated_region_clone_witness(blocks, &header, &arm),
        ));
    }
    None
}

fn clone_witness_fields(
    prefix: &str,
    witness: Option<clone_crossarm::DominatedRegionCloneWitness>,
    mode: Option<(bool, bool)>,
) -> String {
    let Some(witness) = witness else {
        return format!("{prefix}_cross_arm=none");
    };
    let mode = mode
        .map(|(converge, break_aware)| {
            format!(" {prefix}_mode=converge:{converge},break_aware:{break_aware}")
        })
        .unwrap_or_default();
    let missing_carrier = witness.first_missing_carrier.as_deref().unwrap_or("none");
    let empty_phi = witness.first_empty_phi_block.as_deref().unwrap_or("none");
    format!(
        "{prefix}_cross_arm={}->{}{} {prefix}_clone_reason={} {prefix}_region_blocks={} {prefix}_region_cap={} {prefix}_boundary_count={} {prefix}_boundary_cap={} {prefix}_boundary_sample=[{}] {prefix}_redirect_count={} {prefix}_external_pred_count={} {prefix}_arm_cycle_pred_count={} {prefix}_missing_carrier={} {prefix}_empty_phi_block={}",
        witness.header,
        witness.arm,
        mode,
        witness.reason,
        witness.region_blocks,
        witness.region_cap,
        witness.boundary_count,
        witness.boundary_cap,
        witness.boundary_sample.join(","),
        witness.redirect_count,
        witness.external_pred_count,
        witness.arm_cycle_pred_count,
        missing_carrier,
        empty_phi,
    )
}

fn region_fixpoint_fields(witness: &clone_crossarm::RegionCrossArmFixpointWitness) -> String {
    let next_blocks = witness
        .next_blocks
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string());
    let candidate = witness
        .stop_candidate
        .clone()
        .map(|candidate| clone_witness_fields("region_stop", Some(candidate), None))
        .unwrap_or_else(|| "region_stop_cross_arm=none".to_string());
    format!(
        " region_fixpoint_stop={} region_fixpoint_rounds={} region_fixpoint_input_blocks={} region_fixpoint_output_blocks={} region_fixpoint_max_blocks={} region_fixpoint_next_blocks={} {}",
        witness.stop_reason,
        witness.rounds,
        witness.input_blocks,
        witness.output_blocks,
        witness.max_blocks,
        next_blocks,
        candidate,
    )
}

fn append_straddle_witness_modes(graph: &str, blocks: &[BodyBlock], out: &mut Vec<String>) {
    for (converge, break_aware) in [(false, false), (true, false), (true, true)] {
        append_straddle_witness_lines(graph, blocks, converge, break_aware, out);
    }
}

fn append_straddle_witness_lines(
    graph: &str,
    blocks: &[BodyBlock],
    converge_inloop: bool,
    break_aware: bool,
    out: &mut Vec<String>,
) {
    let (lblocks, loop_merges) = forest_loop_merges(blocks, converge_inloop, false);
    let lforest = analyze(&lblocks);
    for loop_info in &lforest.loops {
        if !loop_merges.contains_key(&loop_info.header) {
            return;
        }
    }
    let (sblocks, branch, switch) = unique_selection_merges(&lblocks, &loop_merges, break_aware);
    let forest = analyze(&sblocks);
    let input_forest = analyze(blocks);
    let loop_headers: HashSet<&str> = forest.loops.iter().map(|l| l.header.as_str()).collect();
    let names: HashSet<&str> = sblocks.iter().map(|b| b.name.as_str()).collect();
    let source_names: HashSet<&str> = blocks.iter().map(|b| b.name.as_str()).collect();
    let sblocks_by_name: HashMap<&str, &BodyBlock> = sblocks
        .iter()
        .map(|block| (block.name.as_str(), block))
        .collect();
    let mut header_merge: HashMap<String, String> = HashMap::new();
    for (h, info) in &loop_merges {
        header_merge.insert(h.clone(), info.merge.clone());
    }
    for b in &sblocks {
        if loop_headers.contains(b.name.as_str()) {
            continue;
        }
        if is_switch_block(b) {
            if let Some(m) = switch.get(&b.name) {
                header_merge.insert(b.name.clone(), m.clone());
            }
            continue;
        }
        let succs = block_successors(b);
        let distinct: HashSet<&str> = succs
            .iter()
            .map(String::as_str)
            .filter(|t| names.contains(t))
            .collect();
        if distinct.len() < 2 {
            continue;
        }
        if let Some((t, f)) = conditional_branch_targets(b) {
            if let Some(m) = branch.get(&(t, f)) {
                header_merge.insert(b.name.clone(), m.clone());
            }
        }
    }

    let multilevel_switches = multilevel_break_switch_witnesses(blocks);
    let switch_gate = !multilevel_switches.is_empty();
    for l in &forest.loops {
        let Some(info) = loop_merges.get(&l.header) else {
            continue;
        };
        let ml = info.merge.as_str();
        for (ch, cm) in &header_merge {
            if ch == &l.header {
                continue;
            }
            let inside = forest.dominates(ch, &l.header) && !forest.dominates(cm, &l.header);
            if !(inside && forest.dominates(cm, ml)) {
                continue;
            }
            let in_loop_preds = sblocks
                .iter()
                .filter(|candidate| {
                    l.body.iter().any(|node| node == &candidate.name)
                        && block_successors(candidate)
                            .iter()
                            .any(|target| target == ml)
                })
                .map(|candidate| candidate.name.clone())
                .collect::<Vec<_>>();
            let mut pred_sample = in_loop_preds.iter().take(8).cloned().collect::<Vec<_>>();
            if in_loop_preds.len() > pred_sample.len() {
                pred_sample.push("…".to_string());
            }
            let input_preds = input_forest
                .loop_for_header(&l.header)
                .map(|input_loop| {
                    blocks
                        .iter()
                        .filter(|candidate| {
                            input_loop.body.iter().any(|node| node == &candidate.name)
                                && block_successors(candidate)
                                    .iter()
                                    .any(|target| target == ml)
                        })
                        .map(|candidate| candidate.name.clone())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let mut input_pred_sample = input_preds.iter().take(8).cloned().collect::<Vec<_>>();
            if input_preds.len() > input_pred_sample.len() {
                input_pred_sample.push("…".to_string());
            }
            let mut switch_sample = multilevel_switches
                .iter()
                .take(8)
                .cloned()
                .collect::<Vec<_>>();
            if multilevel_switches.len() > switch_sample.len() {
                switch_sample.push("…".to_string());
            }
            let closure = straddle_closure_stats(&sblocks, &forest, &l.body, ml, ch, cm);
            out.push(format!(
                "graph={graph} blocks={} converge={} break_aware={} switch_gate={} loop={} loop_merge={} loop_merge_role={} loop_merge_in_input={} loop_merge_phi={} loop_continue={} loop_body_blocks={} owner={} owner_kind={} owner_merge={} owner_merge_role={} in_loop_pred_count={} in_loop_preds=[{}] input_split_pred_count={} input_split_preds=[{}] multilevel_switch_count={} multilevel_switches=[{}] ct_owner_arm_blocks={} ct_merge_tail_blocks={} ct_closure_blocks={} ct_closure_edges={} ct_entry_count={} ct_entries=[{}] ct_exit_count={} ct_exits=[{}] ct_exit_phi_incoming_count={} ct_exit_phi_value_count={} ct_exit_phi_pointer_count={} ct_exit_phi_sample=[{}] ct_nonphi_escape_count={} ct_nonphi_pointer_escape_count={} ct_nonphi_escape_sample=[{}] ct_enclosing_selection_count={} ct_enclosing_selection_expanded_blocks={} ct_enclosing_selection_sample=[{}]",
                blocks.len(),
                converge_inloop,
                break_aware,
                switch_gate,
                l.header,
                ml,
                block_role_label(&sblocks_by_name, ml),
                source_names.contains(ml),
                block_has_phi(&sblocks, ml),
                info.continue_target,
                l.body.len(),
                ch,
                header_kind(&sblocks_by_name, &loop_headers, ch),
                cm,
                block_role_label(&sblocks_by_name, cm),
                in_loop_preds.len(),
                pred_sample.join(","),
                input_preds.len(),
                input_pred_sample.join(","),
                multilevel_switches.len(),
                switch_sample.join(","),
                closure.owner_arm_blocks,
                closure.merge_tail_blocks,
                closure.closure_blocks,
                closure.closure_edges,
                closure.entry_count,
                closure.entry_sample.join(","),
                closure.exit_count,
                closure.exit_sample.join(","),
                closure.exit_phi_incoming_count,
                closure.exit_phi_value_count,
                closure.exit_phi_pointer_count,
                closure.exit_phi_sample.join(","),
                closure.nonphi_escape_count,
                closure.nonphi_pointer_escape_count,
                closure.nonphi_escape_sample.join(","),
                closure.enclosing_selection_count,
                closure.enclosing_selection_expanded_blocks,
                closure.enclosing_selection_sample.join(","),
            ));
        }
    }
}

fn header_kind<'a>(
    blocks_by_name: &HashMap<&'a str, &'a BodyBlock>,
    loop_headers: &HashSet<&str>,
    name: &str,
) -> &'static str {
    if loop_headers.contains(name) {
        return "loop";
    }
    let Some(block) = blocks_by_name.get(name) else {
        return "missing";
    };
    if is_switch_block(block) {
        return "switch";
    }
    if conditional_branch_targets(block).is_some() {
        "cond"
    } else {
        "other"
    }
}

fn block_role_label<'a>(
    blocks_by_name: &HashMap<&'a str, &'a BodyBlock>,
    name: &str,
) -> &'static str {
    let Some(block) = blocks_by_name.get(name) else {
        return "missing";
    };
    match block.role {
        BlockRole::Normal => "normal",
        BlockRole::LMerge => "lmerge",
        BlockRole::TerminalExitReturn => "terminal-exit-return",
        BlockRole::SwitchBypass => "switch-bypass",
        BlockRole::ConstructTreeRoute => "construct-tree-route",
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StraddleClosureStats {
    owner_arm_blocks: usize,
    merge_tail_blocks: usize,
    closure_blocks: usize,
    closure_edges: usize,
    entry_count: usize,
    entry_sample: Vec<String>,
    exit_count: usize,
    exit_sample: Vec<String>,
    exit_phi_incoming_count: usize,
    exit_phi_value_count: usize,
    exit_phi_pointer_count: usize,
    exit_phi_sample: Vec<String>,
    nonphi_escape_count: usize,
    nonphi_pointer_escape_count: usize,
    nonphi_escape_sample: Vec<String>,
    enclosing_selection_count: usize,
    enclosing_selection_expanded_blocks: usize,
    enclosing_selection_sample: Vec<String>,
}

fn straddle_closure_stats(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    loop_body: &[String],
    loop_merge: &str,
    owner: &str,
    owner_merge: &str,
) -> StraddleClosureStats {
    let names: HashSet<&str> = blocks.iter().map(|block| block.name.as_str()).collect();
    let mut preds: HashMap<String, Vec<String>> = HashMap::new();
    for block in blocks {
        for succ in block_successors(block) {
            if names.contains(succ.as_str()) {
                preds.entry(succ).or_default().push(block.name.clone());
            }
        }
    }

    let mut reaches_loop_merge = HashSet::new();
    let mut stack = vec![loop_merge.to_string()];
    while let Some(node) = stack.pop() {
        if !reaches_loop_merge.insert(node.clone()) {
            continue;
        }
        if let Some(predecessors) = preds.get(&node) {
            stack.extend(predecessors.iter().cloned());
        }
    }

    let loop_body: HashSet<&str> = loop_body.iter().map(String::as_str).collect();
    let mut owner_arm_blocks = 0usize;
    let mut merge_tail_blocks = 0usize;
    let mut closure = HashSet::new();
    for block in blocks {
        let name = block.name.as_str();
        let in_owner_arm = forest.dominates(owner, name)
            && !forest.dominates(owner_merge, name)
            && reaches_loop_merge.contains(name);
        let in_merge_tail =
            forest.dominates(owner_merge, name) && reaches_loop_merge.contains(name);
        if in_owner_arm {
            owner_arm_blocks += 1;
        }
        if in_merge_tail {
            merge_tail_blocks += 1;
        }
        if in_owner_arm || in_merge_tail || loop_body.contains(name) || name == loop_merge {
            closure.insert(name.to_string());
        }
    }

    let selection_naturals = selection_merges(blocks, forest);
    let mut enclosing_selection_expanded = closure.clone();
    let mut enclosing_selection_sample = Vec::new();
    for block in blocks {
        if conditional_branch_targets(block).is_none() {
            continue;
        }
        let Some(natural) = selection_naturals.get(&block.name) else {
            continue;
        };
        let intersects = closure
            .iter()
            .any(|name| forest.dominates(&block.name, name) && !forest.dominates(natural, name));
        if !intersects {
            continue;
        }
        let before = enclosing_selection_expanded.len();
        for candidate in blocks {
            if forest.dominates(&block.name, &candidate.name)
                && !forest.dominates(natural, &candidate.name)
            {
                enclosing_selection_expanded.insert(candidate.name.clone());
            }
        }
        enclosing_selection_expanded.insert(natural.clone());
        let added = enclosing_selection_expanded.len().saturating_sub(before);
        enclosing_selection_sample.push(format!("{}->{}(+{})", block.name, natural, added));
    }
    enclosing_selection_sample.sort();
    enclosing_selection_sample.dedup();
    let enclosing_selection_count = enclosing_selection_sample.len();
    let mut enclosing_selection_sample = enclosing_selection_sample
        .iter()
        .take(8)
        .cloned()
        .collect::<Vec<_>>();
    if enclosing_selection_count > enclosing_selection_sample.len() {
        enclosing_selection_sample.push("…".to_string());
    }

    let mut closure_edges = 0usize;
    let mut entries = Vec::new();
    let mut exits = Vec::new();
    for block in blocks {
        for succ in block_successors(block) {
            if !names.contains(succ.as_str()) {
                continue;
            }
            let from_inside = closure.contains(&block.name);
            let to_inside = closure.contains(&succ);
            if from_inside && to_inside {
                closure_edges += 1;
            } else if from_inside {
                exits.push(format!("{}->{}", block.name, succ));
            } else if to_inside {
                entries.push(format!("{}->{}", block.name, succ));
            }
        }
    }
    entries.sort();
    entries.dedup();
    let mut entry_sample = entries.iter().take(8).cloned().collect::<Vec<_>>();
    if entries.len() > entry_sample.len() {
        entry_sample.push("…".to_string());
    }
    exits.sort();
    exits.dedup();
    let mut exit_sample = exits.iter().take(8).cloned().collect::<Vec<_>>();
    if exits.len() > exit_sample.len() {
        exit_sample.push("…".to_string());
    }

    let def_ty = blocks
        .iter()
        .filter(|block| closure.contains(&block.name))
        .flat_map(|block| {
            block
                .typed
                .as_ref()
                .into_iter()
                .flat_map(|carrier| &carrier.insts)
                .filter_map(|inst| Some((inst.result.clone()?, inst.result_ty.clone())))
        })
        .collect::<HashMap<_, _>>();
    let is_pointer_def = |name: &str| {
        def_ty
            .get(name)
            .and_then(|ty| ty.as_ref())
            .is_some_and(|ty| matches!(ty, crate::native::ir::LlType::Ptr(_)))
    };

    let mut exit_phi_incoming_count = 0usize;
    let mut exit_phi_value_count = 0usize;
    let mut exit_phi_pointer_count = 0usize;
    let mut exit_phi_sample = Vec::new();
    let mut nonphi_escapes = Vec::new();
    for block in blocks {
        if closure.contains(&block.name) {
            continue;
        }
        let Some(carrier) = &block.typed else {
            continue;
        };
        for inst in &carrier.insts {
            if let Some((_, incoming)) = &inst.phi_incoming() {
                for (value, predecessor) in incoming {
                    if !closure.contains(predecessor) {
                        continue;
                    }
                    exit_phi_incoming_count += 1;
                    let mut locals = Vec::new();
                    collect_llvalue_locals(value, &mut locals);
                    let value_locals = locals
                        .into_iter()
                        .filter(|name| def_ty.contains_key(name))
                        .collect::<Vec<_>>();
                    exit_phi_value_count += value_locals.len();
                    exit_phi_pointer_count += value_locals
                        .iter()
                        .filter(|name| is_pointer_def(name))
                        .count();
                    if exit_phi_sample.len() < 8 {
                        let result = inst.result.as_deref().unwrap_or("_");
                        let values = if value_locals.is_empty() {
                            "const".to_string()
                        } else {
                            value_locals.join("|")
                        };
                        exit_phi_sample.push(format!(
                            "{}:{}<-{}:{values}",
                            block.name, result, predecessor
                        ));
                    }
                }
                continue;
            }
            inst.visit_uses(|used| {
                if def_ty.contains_key(used) {
                    nonphi_escapes.push((used.to_string(), block.name.clone()));
                }
            });
        }
        for used in terminator_uses(carrier) {
            if def_ty.contains_key(&used) {
                nonphi_escapes.push((used, block.name.clone()));
            }
        }
    }
    nonphi_escapes.sort();
    nonphi_escapes.dedup();
    let nonphi_pointer_escape_count = nonphi_escapes
        .iter()
        .filter(|(name, _)| is_pointer_def(name))
        .count();
    let mut nonphi_escape_sample = nonphi_escapes
        .iter()
        .take(8)
        .map(|(name, block)| format!("{name}->{block}"))
        .collect::<Vec<_>>();
    if nonphi_escapes.len() > nonphi_escape_sample.len() {
        nonphi_escape_sample.push("…".to_string());
    }

    StraddleClosureStats {
        owner_arm_blocks,
        merge_tail_blocks,
        closure_blocks: closure.len(),
        closure_edges,
        entry_count: entries.len(),
        entry_sample,
        exit_count: exits.len(),
        exit_sample,
        exit_phi_incoming_count,
        exit_phi_value_count,
        exit_phi_pointer_count,
        exit_phi_sample,
        nonphi_escape_count: nonphi_escapes.len(),
        nonphi_pointer_escape_count,
        nonphi_escape_sample,
        enclosing_selection_count,
        enclosing_selection_expanded_blocks: enclosing_selection_expanded.len(),
        enclosing_selection_sample,
    }
}

fn collect_llvalue_locals(value: &crate::native::ir::LlValue, out: &mut Vec<String>) {
    use crate::native::ir::LlValue;
    match value {
        LlValue::Local(name) => out.push(name.clone()),
        LlValue::Vector(values) | LlValue::Array(values) | LlValue::Struct(values) => {
            for value in values {
                collect_llvalue_locals(&value.value, out);
            }
        }
        LlValue::Splat(value) => collect_llvalue_locals(&value.value, out),
        LlValue::Gep(gep) => {
            collect_llvalue_locals(&gep.base.value, out);
            for index in &gep.indices {
                collect_llvalue_locals(&index.value, out);
            }
        }
        LlValue::IntToPtr { source, .. } => collect_llvalue_locals(&source.value, out),
        LlValue::Global(_)
        | LlValue::Bool(_)
        | LlValue::Int(_)
        | LlValue::SignedInt(_)
        | LlValue::Hex(_)
        | LlValue::Float(_)
        | LlValue::Float32Bits(_)
        | LlValue::HalfBits(_)
        | LlValue::BFloatBits(_)
        | LlValue::Zero
        | LlValue::Undef => {}
    }
}

fn terminator_uses(carrier: &crate::native::tir::TirBlock) -> Vec<String> {
    match &carrier.terminator {
        crate::native::tir::TirTerminator::Br(_)
        | crate::native::tir::TirTerminator::Ret(None)
        | crate::native::tir::TirTerminator::Unreachable => Vec::new(),
        crate::native::tir::TirTerminator::BrCond { cond, .. }
        | crate::native::tir::TirTerminator::Switch { selector: cond, .. }
        | crate::native::tir::TirTerminator::Ret(Some(cond)) => vec![cond.clone()],
    }
}

fn multilevel_break_switch_witnesses(blocks: &[BodyBlock]) -> Vec<String> {
    let forest = analyze(blocks);
    let mut out = Vec::new();
    for loop_info in &forest.loops {
        let body: HashSet<&str> = loop_info.body.iter().map(String::as_str).collect();
        for block in blocks {
            if block.name == loop_info.header
                || !body.contains(block.name.as_str())
                || !is_switch_block(block)
            {
                continue;
            }
            let mut exits = block_successors(block)
                .into_iter()
                .filter(|target| !body.contains(target.as_str()))
                .collect::<Vec<_>>();
            if exits.is_empty() {
                continue;
            }
            exits.sort();
            exits.dedup();
            out.push(format!(
                "{}:{}=>{}",
                loop_info.header,
                block.name,
                exits.join("|")
            ));
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Keystone-2: locate a cross-arm-SHARED violation (self-check 2) that only MATERIALIZES post-synthesis —
/// the shared arm's declared merge is a synthesized `%…lmerge.sel*` block, so the raw-block detectors
/// can't characterize it. Derive the header→merge map under synthesis exactly as
/// [`structured_plan_inner4`] does, run self-check 2's dominance test, and return the RAW `(header, arm)`
/// of the first violation (both original blocks — synthesis only ADDS merge blocks). Returns `None` when
/// no violation, a loop is forest-uncovered, or the header/arm is not a raw block.
pub(in crate::native) fn find_synthesized_cross_arm_shared(
    blocks: &[BodyBlock],
    converge_inloop: bool,
    break_aware: bool,
) -> Option<(String, String)> {
    let (lblocks, loop_merges) = forest_loop_merges(blocks, converge_inloop, false);
    let lforest = analyze(&lblocks);
    for l in &lforest.loops {
        if !loop_merges.contains_key(&l.header) {
            return None;
        }
    }
    let (sblocks, branch, switch) = unique_selection_merges(&lblocks, &loop_merges, break_aware);
    let forest = analyze(&sblocks);
    let loop_headers: HashSet<&str> = forest.loops.iter().map(|l| l.header.as_str()).collect();
    let names: HashSet<&str> = sblocks.iter().map(|b| b.name.as_str()).collect();
    let mut header_merge: HashMap<String, String> = HashMap::new();
    for (h, info) in &loop_merges {
        header_merge.insert(h.clone(), info.merge.clone());
    }
    for b in &sblocks {
        if loop_headers.contains(b.name.as_str()) {
            continue;
        }
        let is_switch = is_switch_block(b);
        if is_switch {
            if let Some(m) = switch.get(&b.name) {
                header_merge.insert(b.name.clone(), m.clone());
            }
            continue;
        }
        let succs = block_successors(b);
        let distinct: HashSet<&str> = succs
            .iter()
            .map(String::as_str)
            .filter(|t| names.contains(t))
            .collect();
        if distinct.len() < 2 {
            continue;
        }
        if let Some((t, f)) = conditional_branch_targets(b) {
            if let Some(m) = branch.get(&(t, f)) {
                header_merge.insert(b.name.clone(), m.clone());
            }
        }
    }
    let is_enclosing_break = |b: &str, a: &str| -> bool {
        forest.loops.iter().any(|l| {
            l.body.iter().any(|n| n == b)
                && loop_merges
                    .get(&l.header)
                    .is_some_and(|i| i.merge == a || i.continue_target == a)
        })
    };
    let raw: HashSet<&str> = blocks.iter().map(|b| b.name.as_str()).collect();
    let raw_forest = analyze(blocks);
    let raw_loop_headers = raw_forest
        .loops
        .iter()
        .map(|natural_loop| natural_loop.header.as_str())
        .collect::<HashSet<_>>();
    let raw_loop_latches = raw_forest
        .loops
        .iter()
        .flat_map(|natural_loop| natural_loop.latches.iter().map(String::as_str))
        .collect::<HashSet<_>>();
    let raw_loop_exits = raw_forest
        .loops
        .iter()
        .flat_map(|natural_loop| natural_loop.exits.iter().map(String::as_str))
        .collect::<HashSet<_>>();
    let clone_is_loop_local = |owner: &str, continuation: &str| {
        super::clone_crossarm::shared_clone_is_loop_local(
            blocks,
            &raw_forest,
            owner,
            continuation,
            &raw_loop_headers,
            &raw_loop_latches,
            &raw_loop_exits,
        )
    };
    for b in &sblocks {
        if loop_headers.contains(b.name.as_str()) {
            continue;
        }
        let Some(m) = header_merge.get(&b.name) else {
            continue;
        };
        for a in block_successors(b) {
            if &a == m || is_enclosing_break(&b.name, &a) {
                continue;
            }
            if !forest.dominates(&b.name, &a)
                && raw.contains(b.name.as_str())
                && raw.contains(a.as_str())
                && clone_is_loop_local(&b.name, &a)
            {
                return Some((b.name.clone(), a));
            }
        }
    }
    // Self-check-3 (definitive cross-arm EDGE) under synthesis: an internal block `B` deep in one arm of an
    // ancestor 2-arm selection `H` (arm targets t0/t1, with an assigned merge) branching to `S` dominated
    // by the SIBLING arm. Return the RAW `(child, S)` — `child` = the near arm target dominating B (the
    // `privatize_dominated_region` header), `S` its arm — so the caller clones `S`'s region on the raw
    // blocks. Mirrors `plan_self_check_reason`'s self-check 3 but drives the clone (the 9th attempt's
    // `find_cross_arm_edge` runs on RAW blocks and can't see the synthesized-merge assignment).
    let targets: HashMap<&str, (String, String)> = sblocks
        .iter()
        .filter(|h| !loop_headers.contains(h.name.as_str()))
        .filter_map(|h| {
            let m = header_merge.get(&h.name)?;
            let (t0, t1) = conditional_branch_targets(h)?;
            if &t0 == m || &t1 == m || t0 == t1 {
                return None;
            }
            Some((h.name.as_str(), (t0, t1)))
        })
        .collect();
    for b in &sblocks {
        for s in block_successors(b) {
            if forest.dominates(&b.name, &s) {
                continue;
            }
            let mut child: &str = &b.name;
            while let Some(cur) = forest.idom(child) {
                if let Some((x, y)) = targets.get(cur) {
                    let sibling = if child == x {
                        Some(y)
                    } else if child == y {
                        Some(x)
                    } else {
                        None
                    };
                    if let Some(t1) = sibling {
                        if forest.dominates(t1, &s)
                            && raw.contains(child)
                            && raw.contains(s.as_str())
                            && clone_is_loop_local(child, &s)
                        {
                            return Some((child.to_string(), s));
                        }
                    }
                }
                child = cur;
            }
        }
    }
    None
}

/// Fixpoint driver for the synthesis-aware cross-arm-SHARED clone. Each
/// round DERIVES the synthesized merge map on the current (raw + prior-clone) blocks to find the next
/// `(header, arm)` violation, then clones `arm`'s dominated region on those RAW blocks. Clones are
/// raw-named so detection re-derives cleanly each round; pairs with the multi-boundary handling for
/// compound shapes. Floor-safe: reject-triggered, and the plan self-checks gate the cloned result.
pub(in crate::native) fn privatize_synthesized_cross_arm_shared(
    blocks: &[BodyBlock],
    converge_inloop: bool,
    break_aware: bool,
) -> Vec<BodyBlock> {
    let mut cur: Vec<BodyBlock> = blocks.to_vec();
    let mut counter = 2_000_000usize;
    const ROUNDS: usize = 16;
    // The ladder can only consume candidates within CROSS_ARM_EDGE_MAX_BLOCKS. Do not build (and,
    // critically, do not re-derive dominators/loop forests for) a larger candidate that the caller
    // will immediately discard. The former `blocks.len() + 512` private cap let a 20-block helper
    // grow beyond 500 blocks for another expensive detection round even though the public ladder's
    // 300-block gate made that work unobservable.
    let cap = (blocks.len() + 512).min(CROSS_ARM_EDGE_MAX_BLOCKS);
    // A block-count cap alone does not bound cloning cost: optimized AIR commonly has a few dozen
    // very large blocks, and duplicating their typed instruction payload several times can turn a
    // 25-block helper into hundreds of megabytes while every candidate still remains below 300
    // blocks. A successful dominated-region privatization needs at most a bounded number of copies
    // of the source payload; stop the reject-only attempt if compound cloning exceeds four source
    // payloads plus room for small synthesized blocks.
    let source_instructions = typed_instruction_count(blocks);
    let instruction_cap = source_instructions.saturating_mul(4).saturating_add(256);
    for _ in 0..ROUNDS {
        if cur.len() > cap {
            break;
        }
        let Some((header, arm)) =
            find_synthesized_cross_arm_shared(&cur, converge_inloop, break_aware)
        else {
            break;
        };
        let Some(next) =
            super::clone_crossarm::privatize_dominated_region(&cur, &header, &arm, &mut counter)
        else {
            break;
        };
        cur = next;
        if typed_instruction_count(&cur) > instruction_cap {
            // Return the source graph, not a partially cloned candidate: equality is the caller's
            // established signal that this optional ladder attempt made no consumable change.
            return blocks.to_vec();
        }
        if cur.len() > cap {
            // Preserve the oversized result so the caller's existing size check rejects it, but
            // avoid feeding it through another synthesis/forest round first.
            break;
        }
    }
    cur
}

fn typed_instruction_count(blocks: &[BodyBlock]) -> usize {
    blocks.iter().fold(0usize, |count, block| {
        count.saturating_add(
            block
                .typed
                .as_ref()
                .map_or(1, |typed| typed.insts.len().saturating_add(1)),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::super::test_support::bb;
    use super::*;

    #[test]
    fn straddle_closure_counts_owner_arm_and_merge_tail() {
        let blocks = vec![
            bb("%guard", &["br i1 %c0, label %loop, label %out"]),
            bb("%loop", &["br label %body"]),
            bb("%body", &["br i1 %c1, label %loop, label %selmerge"]),
            bb("%selmerge", &["br label %lmerge"]),
            bb("%lmerge", &["%v = add i32 1, 2", "br label %after"]),
            bb("%out", &["ret void"]),
            bb("%after", &["%x = phi i32 [ %v, %lmerge ]", "ret void"]),
        ];
        let forest = analyze(&blocks);
        let loop_info = forest
            .loop_for_header("%loop")
            .expect("synthetic loop is discovered");
        let stats = straddle_closure_stats(
            &blocks,
            &forest,
            &loop_info.body,
            "%lmerge",
            "%guard",
            "%selmerge",
        );

        assert_eq!(stats.owner_arm_blocks, 3);
        assert_eq!(stats.merge_tail_blocks, 2);
        assert_eq!(stats.closure_blocks, 5);
        assert_eq!(stats.closure_edges, 5);
        assert_eq!(stats.entry_count, 0);
        assert_eq!(stats.exit_count, 2);
        assert_eq!(stats.exit_phi_incoming_count, 1);
        assert_eq!(stats.exit_phi_value_count, 1);
        assert_eq!(stats.exit_phi_pointer_count, 0);
        assert_eq!(stats.nonphi_escape_count, 0);
        assert_eq!(stats.nonphi_pointer_escape_count, 0);
        assert!(stats.exit_sample.contains(&"%guard->%out".to_string()));
        assert!(stats.exit_sample.contains(&"%lmerge->%after".to_string()));
        assert!(stats
            .exit_phi_sample
            .contains(&"%after:%x<-%lmerge:%v".to_string()));
    }

    #[test]
    fn final_phi_contract_rejects_conflicting_predecessor_values() {
        let blocks = vec![
            bb("%entry", &["br label %merge"]),
            bb(
                "%merge",
                &["%v = phi i32 [ 1, %entry ], [ 2, %entry ]", "ret void"],
            ),
        ];
        assert_eq!(
            conflicting_phi_predecessor_reason(&blocks).as_deref(),
            Some("phi-conflicting-predecessor block=%merge pred=%entry")
        );
    }
}
