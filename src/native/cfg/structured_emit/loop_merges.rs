//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

fn next_selection_merge_suffix(blocks: &[BodyBlock]) -> usize {
    let prefix = format!("{SPLIT_PREFIX}{SEL_TOKEN}");
    blocks
        .iter()
        .filter_map(|block| block.name.strip_prefix(&prefix))
        .filter_map(|suffix| suffix.parse::<usize>().ok())
        .max()
        .map_or(0, |max| max + 1)
}

fn loop_role_targets_with_passthroughs(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
) -> HashSet<String> {
    let by_name = blocks
        .iter()
        .map(|block| (block.name.as_str(), block))
        .collect::<HashMap<_, _>>();
    let mut out = HashSet::new();
    for info in loop_merges.values() {
        add_loop_role_target_with_passthroughs(&by_name, &mut out, &info.merge);
        add_loop_role_target_with_passthroughs(&by_name, &mut out, &info.continue_target);
    }
    out
}

fn add_loop_role_target_with_passthroughs(
    by_name: &HashMap<&str, &BodyBlock>,
    out: &mut HashSet<String>,
    role: &str,
) {
    let mut current = role.to_string();
    for _ in 0..=by_name.len() {
        if !out.insert(current.clone()) {
            break;
        }
        let Some(block) = by_name.get(current.as_str()) else {
            break;
        };
        if !matches!(
            block.role,
            BlockRole::LMerge | BlockRole::ConstructTreeRoute
        ) {
            break;
        }
        let successors = block_successors(block);
        if successors.len() != 1 || successors[0] == current {
            break;
        }
        current = successors[0].clone();
    }
}

fn construct_tree_redirect_candidate(block: &BodyBlock) -> bool {
    matches!(block.role, BlockRole::Normal | BlockRole::LMerge)
}

/// Routes that still feed `natural` (directly). Used when a regional construct-tree wrapper has
/// rewritten a former Normal/LMerge reconvergence predecessor into a `ConstructTreeRoute` gateway:
/// the selection header no longer has a source-owned edge into `natural`, but the Normal/LMerge
/// blocks that *enter* those routes are still header-dominated and can be reclaimed onto a private
/// merge without rewriting the route edge itself.
fn construct_tree_routes_into_natural(blocks: &[BodyBlock], natural: &str) -> HashSet<String> {
    blocks
        .iter()
        .filter(|block| block.role == BlockRole::ConstructTreeRoute)
        .filter(|block| {
            block_successors(block)
                .iter()
                .any(|target| target == natural)
        })
        .map(|block| block.name.clone())
        .collect()
}

/// Header-dominated Normal/LMerge predecessors that either target `natural` directly or enter a
/// construct-tree route that targets `natural`. Empty when the ordinary direct sweep already covers
/// every reconvergence edge (routes are never claimed as ownership predecessors).
fn construct_tree_preds_reclaiming_routes(
    blocks: &[BodyBlock],
    forest: &LoopForest,
    header: &str,
    natural: &str,
) -> Vec<String> {
    let routes = construct_tree_routes_into_natural(blocks, natural);
    if routes.is_empty() {
        return Vec::new();
    }
    let mut preds = Vec::new();
    for block in blocks {
        if !construct_tree_redirect_candidate(block) || !forest.dominates(header, &block.name) {
            continue;
        }
        let hits = block_successors(block)
            .into_iter()
            .any(|target| target == natural || routes.iter().any(|route| route == &target));
        if hits {
            preds.push(block.name.clone());
        }
    }
    preds.sort();
    preds.dedup();
    preds
}

/// Compute forest-driven loop merges, splitting merge==continue overlaps and pure self-latching
/// no-exit loops in place.
///
/// Returns the (possibly block-augmented) body in program order and a `header -> LoopMergeInfo` map
/// covering the loops this increment handles. The map is a drop-in for the subset of
/// `infer_loop_merges` results it covers; callers may union it with the heuristic map for the rest.
pub(in crate::native) fn forest_loop_merges(
    blocks: &[BodyBlock],
    converge_inloop: bool,
    multi_exit_clone: bool,
) -> (Vec<BodyBlock>, HashMap<String, LoopMergeInfo>) {
    let forest = analyze(blocks);
    let plans = forest.structured_plan();
    let mut out_blocks = blocks.to_vec();
    let mut merges = HashMap::new();
    let mut split_counter = 0usize;
    let mut collision_cache: Option<(LoopForest, HashMap<String, String>)> = None;

    for plan in &plans {
        // A pure infinite loop with its header as the sole latch has no distinct continue target and
        // no source exit to name as its merge. Split the header's body away from its phi block, route
        // its old self-edge through a fresh continue, and add an unreachable merge. This preserves
        // non-termination while giving `OpLoopMerge` the distinct merge/continue blocks SPIR-V needs.
        // Kept deliberately narrow: a conditional/multi-block no-exit loop needs its own construction.
        if plan.restructure.as_slice() == [Restructure::NoExit]
            && plan.continue_block.as_deref() == Some(plan.header.as_str())
        {
            if let Some((merge, continue_target)) =
                synth_noexit_self_latch(&mut out_blocks, &plan.header, &mut split_counter)
            {
                collision_cache = None;
                merges.insert(
                    plan.header.clone(),
                    LoopMergeInfo {
                        merge,
                        continue_target,
                    },
                );
            }
            continue;
        }

        let (Some(continue_target), Some(merge_block)) =
            (plan.continue_block.clone(), plan.merge_block.clone())
        else {
            continue;
        };

        if plan.restructure.is_empty() {
            // Loop-merge ⇄ selection-merge collision: if the loop's merge block is ALSO the
            // post-dominator (natural selection merge) of a conditional OUTSIDE this loop, a single
            // block would have to be both a loop merge and a selection merge — illegal in SPIR-V (a
            // merge block belongs to exactly one construct). Give the LOOP a distinct merge: redirect
            // its in-loop exit edges to `merge_block` through a fresh pass-through (with phi surgery
            // when `merge_block` carries a phi), leaving the original block as the enclosing
            // selection's merge. This is the dominant `cond-phi-shared/loop-role` frontier shape.
            if collision_cache.is_none() {
                let det_forest = analyze(&out_blocks);
                let det_selection_merges = selection_merges(&out_blocks, &det_forest);
                collision_cache = Some((det_forest, det_selection_merges));
            }
            let (det_forest, det_selection_merges) = collision_cache.as_ref().unwrap();
            let collides = merge_collides_with_outer_selection_from(
                det_forest,
                det_selection_merges,
                &plan.header,
                &merge_block,
                converge_inloop,
            );
            if crate::env_vars::flm_why() {
                eprintln!(
                    "[flm-why] header={} restructure={:?} merge={} collides={} has_phi={}",
                    plan.header,
                    plan.restructure,
                    merge_block,
                    collides,
                    block_has_phi(&out_blocks, &merge_block),
                );
            }
            let mut mutated = false;
            let merge_block = if collides {
                let split = if block_has_phi(&out_blocks, &merge_block) {
                    split_phi_overlap(
                        &mut out_blocks,
                        det_forest,
                        &plan.header,
                        &merge_block,
                        &mut split_counter,
                    )
                } else {
                    split_no_phi_overlap(
                        &mut out_blocks,
                        det_forest,
                        &plan.header,
                        &merge_block,
                        &mut split_counter,
                    )
                };
                mutated |= split.is_some();
                split.unwrap_or(merge_block)
            } else {
                merge_block
            };
            // do-while normalization: when the latch (continue) block itself ends in a conditional
            // that exits to the loop merge (the exit test is at the loop bottom), split off a separate
            // unconditional continue block so the latch becomes an ordinary {continue, merge} break.
            let rotated_continue = synth_dowhile_continue(
                &mut out_blocks,
                &plan.header,
                &continue_target,
                &merge_block,
                &mut split_counter,
            );
            mutated |= rotated_continue.is_some();
            let continue_target = rotated_continue.unwrap_or(continue_target);
            if mutated {
                collision_cache = None;
            }
            merges.insert(
                plan.header.clone(),
                LoopMergeInfo {
                    merge: merge_block,
                    continue_target,
                },
            );
            continue;
        }

        // Multiple-exit loops: funnel both exits through one synthesized dispatch merge (k == 2,
        // no-phi exits). On success the loop is single-exit; otherwise left for the fallback path.
        if plan.restructure.as_slice() == [Restructure::MultipleExits] {
            let exits = forest
                .loop_for_header(&plan.header)
                .map(|l| l.exits.clone())
                .unwrap_or_default();
            if let Some(new_merge) = synth_multi_exit_merge(
                &mut out_blocks,
                &forest,
                &plan.header,
                &exits,
                &mut split_counter,
            ) {
                collision_cache = None;
                // A dispatch exit arm reached ALSO from outside the loop (a shared exit) is not
                // `M`-dominated; clone its dominated forward region so both arms reconverge at an
                // `M`-dominated merge (M2 residual fix). Only in the reject-triggered clone attempt
                // (`multi_exit_clone`) — running it on every attempt would corrupt a function that
                // already admits at the base attempt (turning its valid dispatch into a reject→repair).
                if multi_exit_clone {
                    privatize_dispatch_shared_exits(
                        &mut out_blocks,
                        &new_merge,
                        &mut split_counter,
                    );
                }
                // The funnelled loop may now be a do-while (its latch conditionally exits to the new
                // dispatch merge); rotate that latch into a separate continue block too.
                let continue_target = synth_dowhile_continue(
                    &mut out_blocks,
                    &plan.header,
                    &continue_target,
                    &new_merge,
                    &mut split_counter,
                )
                .unwrap_or(continue_target);
                merges.insert(
                    plan.header.clone(),
                    LoopMergeInfo {
                        merge: new_merge,
                        continue_target,
                    },
                );
            }
            continue;
        }

        // Multiple-latch loops: unify the back-edges into one synthesized latch so the loop is
        // single-latch and directly structurable. Mirror of the multi-exit funnel on the continue side.
        // Fires only for a pure-`[MultipleLatches]` loop, which currently rejects, so it can only ADD
        // admissions (never corrupt an already-admitting fn).
        if plan.restructure.as_slice() == [Restructure::MultipleLatches] {
            let latches = forest
                .loop_for_header(&plan.header)
                .map(|l| l.latches.clone())
                .unwrap_or_default();
            if let Some(new_latch) = synth_multi_latch_continue(
                &mut out_blocks,
                &plan.header,
                &latches,
                &mut split_counter,
            ) {
                collision_cache = None;
                // The unified loop may now be a do-while (its single latch conditionally exits to the
                // merge); rotate that latch into a separate unconditional continue block too.
                let continue_target = synth_dowhile_continue(
                    &mut out_blocks,
                    &plan.header,
                    &new_latch,
                    &merge_block,
                    &mut split_counter,
                )
                .unwrap_or(new_latch);
                merges.insert(
                    plan.header.clone(),
                    LoopMergeInfo {
                        merge: merge_block,
                        continue_target,
                    },
                );
            }
            continue;
        }

        // Only the lone merge==continue overlap is handled here; anything else is left for the
        // existing path (and the later increments of the consumer).
        if plan.restructure.as_slice() != [Restructure::MergeIsEnclosingContinue] {
            continue;
        }
        let split = if block_has_phi(&out_blocks, &merge_block) {
            // Phi-carrying shared block: the redirected predecessors' incomings are merged into a phi
            // in the synthesized pass-through block, and the shared block's phi takes that merged
            // value via the pass-through edge (preserving the enclosing continue's own incomings).
            split_phi_overlap(
                &mut out_blocks,
                &forest,
                &plan.header,
                &merge_block,
                &mut split_counter,
            )
        } else {
            split_no_phi_overlap(
                &mut out_blocks,
                &forest,
                &plan.header,
                &merge_block,
                &mut split_counter,
            )
        };
        if let Some(new_merge) = split {
            collision_cache = None;
            // The split redirected this loop's in-body predecessors of the shared merge (including its
            // latch) to `new_merge`; if the latch is now a do-while bottom test (conditionally branches
            // back to the header or out to `new_merge`), rotate it into a clean unconditional continue
            // too — exactly as the empty/MultipleExits branches do. Without this the latch stays both
            // the loop continue AND a conditional, which spirv-val rejects ("block exits the continue,
            // but not via a structured exit").
            let continue_target = synth_dowhile_continue(
                &mut out_blocks,
                &plan.header,
                &continue_target,
                &new_merge,
                &mut split_counter,
            )
            .unwrap_or(continue_target);
            merges.insert(
                plan.header.clone(),
                LoopMergeInfo {
                    merge: new_merge,
                    continue_target,
                },
            );
        }
    }

    // Loop-header-is-also-a-selection split: a header carrying both the loop's OpLoopMerge and a genuine
    // in-loop conditional/switch is illegal SPIR-V; give the conditional/switch its own block so the
    // header only branches unconditionally and the lifted block becomes a structurable selection/switch
    // header. Applied last (after all merge/continue are final) and sorted for determinism; each split is
    // local to one header. A header is at most one shape, so calling both is safe — each no-ops on the
    // other's terminator kind.
    let mut header_infos: Vec<(String, String, String)> = merges
        .iter()
        .map(|(h, i)| (h.clone(), i.merge.clone(), i.continue_target.clone()))
        .collect();
    header_infos.sort();
    for (h, m, c) in header_infos {
        split_loop_header_selection(&mut out_blocks, &h, &m, &c, &mut split_counter);
        split_loop_header_switch(&mut out_blocks, &h, &m, &c, &mut split_counter);
    }

    (out_blocks, merges)
}

/// Structure a pure self-latching infinite loop. LLVM permits a header which carries its own phis,
/// executes the whole body, and ends in `br label %header`; SPIR-V requires an `OpLoopMerge` to name
/// distinct continue and merge blocks. The transform makes that implicit layout explicit:
///
/// ```text
/// preheader -> H(phi, body, br H)    =>    preheader -> H(phi, br B)
///                                             B(body, br C)
///                                             C(br H)
///                                             M(unreachable)
/// ```
///
/// Header phi back-edge predecessors are retargeted from `H` to `C`. The unreachable merge is never
/// executed, exactly matching the source no-exit loop, but remains a real block for `OpLoopMerge`.
/// Returns the `(merge, continue)` pair only for an unconditional direct self branch; broader no-exit
/// shapes remain on the fallback path.
pub(in crate::native) fn synth_noexit_self_latch(
    blocks: &mut Vec<BodyBlock>,
    header: &str,
    counter: &mut usize,
) -> Option<(String, String)> {
    let header_idx = blocks.iter().position(|b| b.name == header)?;
    // Self-latch guard: the header ends in an unconditional `br` back to itself
    // (`TirTerminator::Br(header)`). The carrier is the sole substrate now.
    let src_typed = blocks[header_idx].typed.clone()?;
    if !matches!(&src_typed.terminator, crate::native::tir::TirTerminator::Br(l) if l == header) {
        return None;
    }

    let id = *counter;
    *counter += 1;
    let body = format!("%metal2vulkan.noexit.body.{id}");
    let continue_target = format!("%metal2vulkan.noexit.cont.{id}");
    let merge = format!("%metal2vulkan.noexit.merge.{id}");

    // Build both new carriers off the SOURCE header carrier (kb "STEP-1/STEP-2 decomposition"). The body
    // is the header's NON-PHI instruction suffix (`insts[phi_count..]`, real types already resolved) + a
    // terminator the self-latch guard fixes to `br label {continue_target}` (the header's
    // `br label {header}` redirected). The header is the leading phi PREFIX (`insts[..phi_count]`) with
    // its self back-edge predecessor rewritten to the continue target + a `br label {body}` terminator.
    // Both byte-identical to re-lowering the rewritten lines by construction (`from_suffix`/`prefix`
    // reuse already-typed insts; `rewrite_phi_predecessor` has a `== re-lower` test).
    let header_name = blocks[header_idx].name.clone();
    let phi_count = src_typed.insts.iter().take_while(|i| i.is_phi()).count();
    let body_typed = crate::native::tir::lower_block_carrier_from_suffix(
        &body,
        &src_typed,
        phi_count,
        &format!("br label {continue_target}"),
    );
    blocks[header_idx].typed = crate::native::tir::lower_block_carrier_prefix(
        &header_name,
        &src_typed,
        phi_count,
        &format!("br label {body}"),
    )
    .map(|mut h| {
        h.rewrite_phi_predecessor(header, &continue_target);
        h
    });
    blocks.insert(
        header_idx + 1,
        BodyBlock {
            name: body,
            role: BlockRole::Normal,
            typed: body_typed,
        },
    );
    let cont_typed = crate::native::tir::lower_block_carrier(
        &continue_target,
        &[format!("br label {header}")],
        &std::collections::HashMap::new(),
    );
    blocks.insert(
        header_idx + 2,
        BodyBlock {
            name: continue_target.clone(),
            role: BlockRole::Normal,
            typed: cont_typed,
        },
    );
    // Keep this after the reachable source blocks. `structured_order` appends unreachable blocks, so
    // the merge follows the loop body while the pruning pass retains it through the loop declaration.
    let merge_typed = crate::native::tir::lower_block_carrier(
        &merge,
        &["unreachable".to_string()],
        &std::collections::HashMap::new(),
    );
    blocks.push(BodyBlock {
        name: merge.clone(),
        role: BlockRole::Normal,
        typed: merge_typed,
    });

    Some((merge, continue_target))
}

/// Per-construct selection-merge synthesis (R2 relooper module 2).
///
/// Each conditional/switch header's natural merge is its immediate post-dominator
/// ([`selection_merges`]). A SPIR-V merge block belongs to exactly ONE construct, so where a
/// post-dominator is shared by two-or-more headers, or collides with a loop's merge/continue, this
/// gives the colliding header a fresh UNIQUE merge: it inserts an empty pass-through block, redirects
/// that header's region predecessors of the shared block to it, and branches it on to the shared
/// block (so other constructs still reach the original). Headers whose natural merge is unique are
/// left on it. Returns the augmented blocks + `branch_merges` (keyed `(true,false)`, the emitter's
/// `branch_merges` key) + `switch_merges` (keyed by header).
///
/// Scope: NO-PHI — a header whose shared merge carries a phi is skipped (omitted from the returned
/// maps), since rewiring it needs the phi-merge surgery [`split_phi_overlap`] does for loops; that is
/// the next sub-increment. Likewise this handles the *flat* collision (sibling/loop sharing); deeply
/// nested constructs that share a post-dominator are validated only by the wholesale floor A/B (module
/// 4), where the integration correctness this unit cannot prove is gated.
pub(in crate::native) fn unique_selection_merges(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
    break_aware: bool,
) -> (
    Vec<BodyBlock>,
    HashMap<(String, String), String>,
    HashMap<String, String>,
) {
    let (blocks, branch, _branch_by_header, switch) =
        unique_selection_merges_with_loop_exit(blocks, loop_merges, break_aware, false);
    (blocks, branch, switch)
}

/// [`unique_selection_merges`] with the reject-only loop-exit convergence extension enabled.
/// The extension is kept out of the public/default helper so all existing callers stay byte-identical;
/// only the final `structured_plan` retry tier asks it to replace a loop-merge post-dominator with a
/// proven in-loop convergence block.
pub(in crate::native) fn unique_selection_merges_with_loop_exit(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
    break_aware: bool,
    loop_exit_selection: bool,
) -> (
    Vec<BodyBlock>,
    HashMap<(String, String), String>,
    HashMap<String, String>,
    HashMap<String, String>,
) {
    unique_selection_merges_with_loop_exit_and_forced(
        blocks,
        loop_merges,
        break_aware,
        loop_exit_selection,
        &HashMap::new(),
    )
}

/// Variant of [`unique_selection_merges_with_loop_exit`] that accepts a small set of explicit
/// terminal-exit merges. A header in `forced_terminal_merges` has one arm that reaches a proved
/// function return, so ordinary post-dominance cannot name its non-return continuation. The caller
/// supplies the private continuation merge and this builder records it exactly like a synthesized
/// selection merge while leaving all other headers on the ordinary derivation.
pub(in crate::native) fn unique_selection_merges_with_loop_exit_and_forced(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
    break_aware: bool,
    loop_exit_selection: bool,
    forced_terminal_merges: &HashMap<String, String>,
) -> (
    Vec<BodyBlock>,
    HashMap<(String, String), String>,
    HashMap<String, String>,
    HashMap<String, String>,
) {
    unique_selection_merges_with_loop_exit_and_forced_inner(
        blocks,
        loop_merges,
        break_aware,
        loop_exit_selection,
        forced_terminal_merges,
        false,
    )
}

/// Construct-tree candidates may carry a stronger source ownership proof than the final synthesized
/// CFG dominance forest can express.  The regional wrapper is a flat state dispatcher: it preserves
/// the original edge semantics, but later merge synthesis can erase static dominance for a header
/// whose natural merge was owned in the pre-synthesis graph.  This variant keeps ordinary emission
/// byte-identical and only lets construct-tree-owned candidates retry the unique-merge split using
/// the immutable pre-synthesis forest when the current forest has no redirectable predecessor.
pub(in crate::native) fn unique_selection_merges_with_construct_tree_ownership(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
    break_aware: bool,
    loop_exit_selection: bool,
    forced_terminal_merges: &HashMap<String, String>,
) -> (
    Vec<BodyBlock>,
    HashMap<(String, String), String>,
    HashMap<String, String>,
    HashMap<String, String>,
) {
    unique_selection_merges_with_loop_exit_and_forced_inner(
        blocks,
        loop_merges,
        break_aware,
        loop_exit_selection,
        forced_terminal_merges,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn unique_selection_merges_with_loop_exit_and_forced_inner(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
    break_aware: bool,
    loop_exit_selection: bool,
    forced_terminal_merges: &HashMap<String, String>,
    construct_tree_owned: bool,
) -> (
    Vec<BodyBlock>,
    HashMap<(String, String), String>,
    HashMap<String, String>,
    HashMap<String, String>,
) {
    let forest = analyze(blocks);
    // Break-aware selection merges (the reject-triggered 5th `structured_plan` attempt) reconverge a
    // guarded-break selection at its non-break arm instead of the loop merge, so it never claims the
    // loop merge and the do-while latch is never redirected (the `merge-inloop` fix at its source).
    let mut sel = if break_aware {
        break_aware_selection_merges(blocks, &forest, loop_merges)
    } else {
        selection_merges(blocks, &forest)
    };
    if loop_exit_selection {
        refine_loop_exit_selection_merges(blocks, &forest, loop_merges, &mut sel);
    }
    let loop_headers: HashSet<&str> = forest.loops.iter().map(|l| l.header.as_str()).collect();
    let loop_roles = loop_role_targets_with_passthroughs(blocks, loop_merges);
    // How many selection headers claim each natural merge (>1 ⇒ shared ⇒ collision).
    let mut claims: HashMap<&str, usize> = HashMap::new();
    for m in sel.values() {
        *claims.entry(m.as_str()).or_default() += 1;
    }
    let mut construct_tree_selection_merges = sel.clone();
    construct_tree_selection_merges.extend(forced_terminal_merges.clone());

    let mut out = blocks.to_vec();
    let mut branch = HashMap::new();
    let mut switch = HashMap::new();
    // Keep assignments by header until every enclosing synth has finished. An outer pass-through may
    // redirect an already-processed inner header's arm, so a `(true, false)` key recorded eagerly can
    // become stale even though that header's declared merge is still correct.
    let mut header_merges: HashMap<String, String> = HashMap::new();
    let mut counter = if construct_tree_owned {
        next_selection_merge_suffix(&out)
    } else {
        0usize
    };

    // Pre-register EVERY structured break/continue latch — not only those with a post-idom selection
    // merge. After multi-exit + do-while rotation a latch has arms `{loop-merge, continue}` and no
    // in-function reconvergence (the arms diverge), so `selection_merges` omits it; without this pass
    // the completeness gate later rejects `branch-no-merge` on that latch (M2 residual / residual
    // `merge-inloop` after converge). Record the key from the CURRENT terminator before any selection
    // synth can rewrite it.
    let mut break_continue_blocks: HashSet<String> = HashSet::new();
    for b in blocks {
        if loop_headers.contains(b.name.as_str()) {
            continue;
        }
        let Some((t, f)) = conditional_branch_targets(b) else {
            continue;
        };
        if let Some(cont) = loop_break_continue_merge(&forest, loop_merges, &b.name, &t, &f) {
            header_merges.insert(b.name.clone(), cont.clone());
            if !loop_exit_selection {
                branch.insert((t, f), cont);
            }
            break_continue_blocks.insert(b.name.clone());
        }
    }

    // Process headers INNERMOST-FIRST (deepest dominator depth first): a nested construct must claim
    // its own arms before its enclosing construct does, else the outer header's region-predecessor
    // sweep grabs the inner arms.
    let depth = |name: &str| {
        let mut d = 0usize;
        let mut cur = name;
        while let Some(p) = forest.idom(cur) {
            d += 1;
            cur = p;
        }
        d
    };
    let mut headers: Vec<&BodyBlock> = blocks
        .iter()
        .filter(|b| {
            !loop_headers.contains(b.name.as_str())
                && (sel.contains_key(&b.name) || forced_terminal_merges.contains_key(&b.name))
                && !break_continue_blocks.contains(&b.name)
        })
        .collect();
    headers.sort_by_key(|b| std::cmp::Reverse(depth(&b.name)));

    // Non-construct retries need dominance over the CURRENT blocks, including pass-through merges
    // synthesized for already-processed (inner) headers. Construct-tree ownership cannot afford to
    // re-analyze the 1k+ block regional graph after every merge split; it derives dominance from the
    // immutable source forest and carries synthesized predecessor origins through `ct_owned_preds`
    // instead.
    let mut cur_forest = analyze(&out);
    #[derive(Clone, Debug)]
    struct CtOwnedPred {
        block: String,
        origins: Vec<String>,
    }
    let mut ct_owned_preds: HashMap<(String, String), Vec<CtOwnedPred>> = HashMap::new();
    let mut stopped_for_growth = false;
    for b in headers.iter().copied() {
        if let Some(merge) = forced_terminal_merges.get(&b.name).cloned() {
            header_merges.insert(b.name.clone(), merge.clone());
            if !loop_exit_selection {
                let cur_block = out.iter().find(|x| x.name == b.name).unwrap_or(b);
                if let Some((true_target, false_target)) = conditional_branch_targets(cur_block) {
                    branch.insert((true_target, false_target), merge);
                }
            }
            continue;
        }
        let Some(natural) = sel.get(&b.name).cloned() else {
            continue;
        };
        if construct_tree_owned {
            if let Some((true_target, false_target)) = conditional_branch_targets(b) {
                if bare_loop_exit_branch(&forest, loop_merges, &b.name, &true_target, &false_target)
                {
                    continue;
                }
                if let Some(exit_target) = enclosing_selection_region_exit_target(
                    blocks,
                    &forest,
                    &construct_tree_selection_merges,
                    &b.name,
                    &true_target,
                    &false_target,
                    Some(&natural),
                ) {
                    let synth = if block_has_phi(&out, &exit_target) {
                        synth_unique_selection_merge_phi(
                            &mut out,
                            &forest,
                            &b.name,
                            &exit_target,
                            &mut counter,
                        )
                    } else {
                        synth_unique_selection_merge(
                            &mut out,
                            &forest,
                            &b.name,
                            &exit_target,
                            &mut counter,
                        )
                    };
                    if let Some(merge) = synth {
                        header_merges.insert(b.name.clone(), merge);
                        continue;
                    }
                }
            }
        }
        // A shared merge is a collision when >1 selection claims it OR it doubles as a loop
        // merge/continue — the `claims`/`loop_roles` tests. But the `claims` counter only sees
        // OTHER SELECTION headers' post-idoms; when the block this header reconverges on is ALSO
        // branched to directly by an ENCLOSING construct's arm, no selection collision is recorded,
        // yet the header does not dominate it (an outer edge reaches it without passing the header) —
        // the `selection:merge-not-dominated` reject. That is the SAME shared-merge shape, so treat a
        // non-dominated natural merge as a collision and let the pass-through synth insert a
        // header-dominated merge (its predecessors are all header-dominated). `plan_self_check_reason`
        // re-checks dominance afterward, so if synth cannot resolve it the plan still honestly rejects.
        let natural_has_phi = block_has_phi(&out, &natural);
        let natural_is_synthetic_merge = construct_tree_owned
            && !natural_has_phi
            && out
                .iter()
                .find(|block| block.name == natural)
                .is_some_and(|block| block.role == BlockRole::LMerge);
        let dominance_forest = if construct_tree_owned {
            &forest
        } else {
            &cur_forest
        };
        let collides = claims.get(natural.as_str()).copied().unwrap_or(0) > 1
            || loop_roles.contains(&natural)
            || natural_is_synthetic_merge
            || !dominance_forest.dominates(&b.name, &natural);
        let merge = if collides {
            // A shared merge carrying a phi needs the phi-merge surgery (merged phi in the synthesized
            // pass-through, `natural`'s phi rebuilt); a no-phi shared merge takes the plain split.
            // NOTE: do NOT skip structural loop-exit predecessors here — that is the disproven
            // LOOP_BREAK_PROTECT shape (dead-end #19): leaving a latch pointing at a loop merge an
            // enclosing selection claimed over-admits into "branches to the selection construct, but
            // not to the selection header". Break-aware selection merges fix the redirect at its
            // source by not claiming the loop merge; multi-exit shared-exit trampolines fix the
            // residual cross-arm on the dispatch itself.
            let source_owned_preds = if construct_tree_owned {
                out.iter()
                    .filter(|candidate| {
                        construct_tree_redirect_candidate(candidate)
                            && forest.dominates(&b.name, &candidate.name)
                            && block_successors(candidate)
                                .iter()
                                .any(|target| target == &natural)
                    })
                    .map(|candidate| candidate.name.clone())
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            // Frontier straddle residual after regional wrap: the only edges into `natural` may be
            // ConstructTreeRoute gateways. Reclaim the Normal/LMerge predecessors of those routes so
            // multi-header shared merges still get private splits (exact-eight routes stay unclaimed).
            let route_reclaimed_preds = if construct_tree_owned && source_owned_preds.is_empty() {
                construct_tree_preds_reclaiming_routes(&out, &forest, &b.name, &natural)
            } else {
                Vec::new()
            };
            let carried_preds = if construct_tree_owned {
                ct_owned_preds
                    .remove(&(b.name.clone(), natural.clone()))
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            let current_owned_preds = Vec::new();
            let mut explicit_no_phi_preds = source_owned_preds.clone();
            explicit_no_phi_preds.extend(route_reclaimed_preds.iter().cloned());
            explicit_no_phi_preds.extend(current_owned_preds);
            explicit_no_phi_preds.extend(carried_preds.iter().map(|pred| pred.block.clone()));
            explicit_no_phi_preds.sort();
            explicit_no_phi_preds.dedup();
            let mut propagated_origins = source_owned_preds.clone();
            propagated_origins.extend(route_reclaimed_preds.iter().cloned());
            for pred in &carried_preds {
                propagated_origins.extend(pred.origins.iter().cloned());
            }
            propagated_origins.sort();
            propagated_origins.dedup();
            let routes_into_natural = if construct_tree_owned && !route_reclaimed_preds.is_empty() {
                construct_tree_routes_into_natural(&out, &natural)
            } else {
                HashSet::new()
            };
            let synth = if natural_has_phi {
                if construct_tree_owned {
                    synth_unique_selection_merge_phi(
                        &mut out,
                        &forest,
                        &b.name,
                        &natural,
                        &mut counter,
                    )
                } else {
                    synth_unique_selection_merge_phi(
                        &mut out,
                        &cur_forest,
                        &b.name,
                        &natural,
                        &mut counter,
                    )
                }
            } else if construct_tree_owned {
                synth_unique_selection_merge_no_phi_explicit(
                    &mut out,
                    &explicit_no_phi_preds,
                    &natural,
                    &routes_into_natural,
                    &mut counter,
                )
            } else {
                let synth = synth_unique_selection_merge(
                    &mut out,
                    &cur_forest,
                    &b.name,
                    &natural,
                    &mut counter,
                );
                if synth.is_none() && construct_tree_owned {
                    synth_unique_selection_merge(&mut out, &forest, &b.name, &natural, &mut counter)
                } else {
                    synth
                }
            };
            match synth {
                Some(s) => {
                    if selection_synth_growth_exceeds_ladder_cap(blocks.len(), out.len()) {
                        stopped_for_growth = true;
                    } else if !construct_tree_owned {
                        cur_forest = analyze(&out);
                    }
                    if construct_tree_owned && !propagated_origins.is_empty() {
                        for target in &headers {
                            if target.name == b.name {
                                continue;
                            }
                            if sel.get(&target.name) != Some(&natural) {
                                continue;
                            }
                            if propagated_origins
                                .iter()
                                .all(|origin| forest.dominates(&target.name, origin))
                            {
                                ct_owned_preds
                                    .entry((target.name.clone(), natural.clone()))
                                    .or_default()
                                    .push(CtOwnedPred {
                                        block: s.clone(),
                                        origins: propagated_origins.clone(),
                                    });
                            }
                        }
                    }
                    s
                }
                None => continue,
            }
        } else {
            natural
        };
        header_merges.insert(b.name.clone(), merge.clone());
        if !loop_exit_selection {
            // Keep the ordinary path byte-identical: it historically keyed each merge as soon as the
            // header was processed. The final re-key below is only needed by the reject-only
            // loop-exit tier, where an outer synth intentionally rewrites nested loop-exit arms.
            let cur_block = out.iter().find(|x| x.name == b.name).unwrap_or(b);
            if is_switch_block(cur_block) {
                switch.insert(b.name.clone(), merge);
            } else if let Some((true_target, false_target)) = conditional_branch_targets(cur_block)
            {
                branch.insert((true_target, false_target), merge);
            }
        }
        if stopped_for_growth {
            break;
        }
    }

    if construct_tree_owned && !stopped_for_growth {
        repair_construct_tree_nondominated_selection_merges(
            &mut out,
            loop_merges,
            &mut header_merges,
            &mut counter,
        );
        repair_construct_tree_enclosing_selection_region_escapes(
            &mut out,
            loop_merges,
            &mut header_merges,
            &mut counter,
        );
        repair_construct_tree_passthrough_selection_merges(&out, loop_merges, &mut header_merges);
    }

    // Re-key from final terminators when a later transform can rewrite an already-recorded header. This
    // is intentionally after the full innermost-first synthesis pass: an enclosing unique-merge split
    // can redirect a nested header's arm after that nested header was assigned, and the emitter keys
    // conditional merges by the final target pair rather than the header name.
    let rekey_all =
        loop_exit_selection || !forced_terminal_merges.is_empty() || construct_tree_owned;
    if rekey_all || !break_continue_blocks.is_empty() {
        for block in &out {
            if !rekey_all && !break_continue_blocks.contains(&block.name) {
                continue;
            }
            let Some(merge) = header_merges.get(&block.name).cloned() else {
                continue;
            };
            if is_switch_block(block) {
                switch.insert(block.name.clone(), merge);
            } else if let Some((true_target, false_target)) = conditional_branch_targets(block) {
                branch.insert((true_target, false_target), merge);
            }
        }
    }
    (out, branch, header_merges, switch)
}

pub(in crate::native) fn repair_construct_tree_nondominated_selection_merges(
    blocks: &mut Vec<BodyBlock>,
    loop_merges: &HashMap<String, LoopMergeInfo>,
    header_merges: &mut HashMap<String, String>,
    counter: &mut usize,
) {
    for _ in 0..8 {
        let forest = analyze(blocks);
        let mut merge_of = HashMap::new();
        for (header, info) in loop_merges {
            merge_of.insert(header.clone(), info.merge.clone());
        }
        merge_of.extend(
            header_merges
                .iter()
                .map(|(header, merge)| (header.clone(), merge.clone())),
        );
        let order = structured_order(blocks, &forest, |header| merge_of.get(header).cloned());
        let rank = order
            .iter()
            .enumerate()
            .map(|(index, name)| (name.as_str(), index))
            .collect::<HashMap<_, _>>();
        let mut headers = blocks
            .iter()
            .filter_map(|block| Some((block.name.clone(), header_merges.get(&block.name)?.clone())))
            .collect::<Vec<_>>();
        headers.sort_by_key(|(header, _)| rank.get(header.as_str()).copied().unwrap_or(usize::MAX));
        let mut repaired = false;
        for (header, merge) in headers {
            if forest.dominates(&header, &merge) {
                continue;
            }
            let Some(block) = blocks.iter().find(|block| block.name == header) else {
                continue;
            };
            let header_targets_merge = block_successors(block)
                .iter()
                .any(|target| target == &merge);
            let route_reclaim_available = !header_targets_merge
                && !construct_tree_preds_reclaiming_routes(blocks, &forest, &header, &merge)
                    .is_empty();
            // Ordinary polluted-merge repair needs a direct header edge into the merge. The
            // post-regional-wrapper residual has no such edge: reconvergence was rewritten into a
            // ConstructTreeRoute gateway, so reclaim Normal/LMerge predecessors of those routes.
            if !header_targets_merge && !route_reclaim_available {
                continue;
            }
            if let Some((t, f)) = conditional_branch_targets(block) {
                if bare_loop_exit_branch_with_passthroughs(
                    blocks,
                    &forest,
                    loop_merges,
                    &header,
                    &t,
                    &f,
                ) {
                    continue;
                }
            }
            if block_has_phi(blocks, &merge) {
                continue;
            }
            let mut repair_preds = blocks
                .iter()
                .filter(|candidate| {
                    construct_tree_redirect_candidate(candidate)
                        && forest.dominates(&header, &candidate.name)
                        && block_successors(candidate)
                            .iter()
                            .any(|target| target == &merge)
                })
                .map(|candidate| candidate.name.clone())
                .collect::<Vec<_>>();
            let routes_into_merge = if repair_preds.is_empty() {
                let reclaimed =
                    construct_tree_preds_reclaiming_routes(blocks, &forest, &header, &merge);
                repair_preds.extend(reclaimed);
                construct_tree_routes_into_natural(blocks, &merge)
            } else {
                HashSet::new()
            };
            repair_preds.sort();
            repair_preds.dedup();
            let split = synth_unique_selection_merge_no_phi_explicit(
                blocks,
                &repair_preds,
                &merge,
                &routes_into_merge,
                counter,
            );
            if let Some(private_merge) = split {
                header_merges.insert(header, private_merge);
                repaired = true;
                break;
            }
        }
        if !repaired {
            break;
        }
    }
}

pub(in crate::native) fn repair_construct_tree_passthrough_selection_merges(
    blocks: &[BodyBlock],
    loop_merges: &HashMap<String, LoopMergeInfo>,
    header_merges: &mut HashMap<String, String>,
) {
    let forest = analyze(blocks);
    let by_name = blocks
        .iter()
        .map(|block| (block.name.as_str(), block))
        .collect::<HashMap<_, _>>();
    let loop_roles = loop_role_targets_with_passthroughs(blocks, loop_merges);
    let mut claims = header_merges
        .values()
        .fold(HashMap::new(), |mut claims, merge| {
            *claims.entry(merge.clone()).or_insert(0usize) += 1;
            claims
        });
    for _ in 0..blocks.len() {
        let mut changed = false;
        let assignments = header_merges
            .iter()
            .map(|(header, merge)| (header.clone(), merge.clone()))
            .collect::<Vec<_>>();
        for (header, merge) in assignments {
            let Some(merge_block) = by_name.get(merge.as_str()) else {
                continue;
            };
            if merge_block.role != BlockRole::LMerge {
                continue;
            }
            let successors = block_successors(merge_block);
            let [successor] = successors.as_slice() else {
                continue;
            };
            // Promoting a private pass-through back onto a block already owned by another
            // selection (or by a loop role) recreates the exact collision the private merge was
            // synthesized to avoid. Reserve targets as promotions happen so two private merges in
            // the same pass cannot both collapse onto an initially-unclaimed successor either.
            if loop_roles.contains(successor) || claims.get(successor).copied().unwrap_or(0) != 0 {
                continue;
            }
            if !forest.dominates(&header, successor) {
                continue;
            }
            let bypasses_merge = blocks.iter().any(|candidate| {
                candidate.name != merge
                    && forest.dominates(&header, &candidate.name)
                    && block_successors(candidate)
                        .iter()
                        .any(|target| target == successor)
            });
            if bypasses_merge {
                header_merges.insert(header, successor.clone());
                if let Some(count) = claims.get_mut(&merge) {
                    *count = count.saturating_sub(1);
                }
                *claims.entry(successor.clone()).or_insert(0) += 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

fn synth_unique_selection_merge_no_phi_explicit(
    blocks: &mut Vec<BodyBlock>,
    preds: &[String],
    natural: &str,
    routes_into_natural: &HashSet<String>,
    counter: &mut usize,
) -> Option<String> {
    let pred_set: HashSet<&str> = preds.iter().map(String::as_str).collect();
    // (pred name, old successor to rewrite → private merge)
    let mut redirect: Vec<(String, String)> = Vec::new();
    for block in blocks.iter() {
        if !pred_set.contains(block.name.as_str()) {
            continue;
        }
        for target in block_successors(block) {
            if target == natural || routes_into_natural.contains(&target) {
                redirect.push((block.name.clone(), target));
            }
        }
    }
    if redirect.is_empty() {
        return None;
    }
    let new_name = format!("{SPLIT_PREFIX}{SEL_TOKEN}{counter}");
    *counter += 1;
    for (pred, old_target) in &redirect {
        if let Some(block) = blocks.iter_mut().find(|block| &block.name == pred) {
            if let Some(typed) = &mut block.typed {
                typed.redirect_successor(old_target, &new_name);
            }
        }
    }
    let at = blocks
        .iter()
        .position(|block| block.name == natural)
        .unwrap_or(blocks.len());
    blocks.insert(
        at,
        synthetic_block(
            new_name.clone(),
            vec![format!("br label {natural}")],
            role_for_name(&new_name),
        ),
    );
    Some(new_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(name: &str) -> BodyBlock {
        BodyBlock {
            name: name.to_string(),
            role: role_for_name(name),
            typed: crate::native::tir::lower_block_carrier(
                name,
                &["ret void".to_string()],
                &HashMap::new(),
            ),
        }
    }

    #[test]
    fn construct_tree_selection_counter_skips_existing_sel_suffixes() {
        let blocks = vec![
            block("%entry"),
            block("%metal2vulkan.lmerge.sel0"),
            block("%metal2vulkan.lmerge.sel12"),
            block("%metal2vulkan.lmerge.selx"),
            block("%metal2vulkan.lmerge.99"),
        ];
        assert_eq!(next_selection_merge_suffix(&blocks), 13);
    }
}
