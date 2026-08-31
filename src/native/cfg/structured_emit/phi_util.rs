//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

/// Privatize the shared (outer-reachable) dispatch arms of a multi-exit merge `M` produced by
/// [`synth_multi_exit_merge`]. After the funnel, `M` branches to the two real loop exits; an exit that
/// is ALSO reached from outside the loop body is not `M`-dominated, so `M`'s dispatch selection cannot
/// own it (`selection:cross-arm-shared`, the M2 residual). A bare trampoline pass-through does NOT fix
/// it — the shared exit block itself stays non-dominated, so the not-found arm still escapes and the
/// dominance self-check admits it anyway (dead-end #6). Instead clone that exit's dominated forward
/// region into an `M`-dominated private copy (reusing
/// [`super::clone_crossarm::privatize_dominated_region`]) so BOTH dispatch arms reconverge at a
/// private, `M`-dominated merge (inserted afterward by [`unique_selection_merges`]) while outer edges
/// keep flowing through the untouched original exit. A non-cloneable region (multiple reconvergence
/// boundaries, an in-region back-edge, or oversize) leaves the graph unchanged — the plan then honestly
/// rejects and falls to the relooper retry, so this is floor-safe.
pub(in crate::native) fn privatize_dispatch_shared_exits(
    blocks: &mut Vec<BodyBlock>,
    dispatch: &str,
    counter: &mut usize,
) {
    let Some((arm0, arm1)) = blocks
        .iter()
        .find(|b| b.name == dispatch)
        .and_then(conditional_branch_targets)
    else {
        return;
    };
    for arm in [arm0, arm1] {
        // Re-analyze each round (a prior clone changed the graph). An arm `M` already dominates is a
        // non-shared exit — already private, nothing to clone.
        if analyze(blocks).dominates(dispatch, &arm) {
            continue;
        }
        if let Some(next) =
            super::clone_crossarm::privatize_dominated_region(blocks, dispatch, &arm, counter)
        {
            *blocks = next;
        }
    }
}

/// Split a phi-carrying merge==continue overlap. Same structure as `split_no_phi_overlap`, but each
/// phi in the shared `exit` is rewritten: incomings from the redirected (inner-loop) predecessors are
/// merged into a fresh phi in the synthesized pass-through block, and the `exit` phi replaces those
/// incomings with a single `[merged, %passthrough]` pair while keeping the enclosing loop's incomings.
///
/// Declining leaves `blocks` byte-identical: the body runs against a staged copy that is
/// committed only on `Some` (see [`atomic_rewrite`]).
pub(in crate::native) fn split_phi_overlap(
    blocks: &mut Vec<BodyBlock>,
    forest: &LoopForest,
    header: &str,
    exit: &str,
    counter: &mut usize,
) -> Option<String> {
    atomic_rewrite(blocks, |blocks| {
        let body: Vec<String> = forest.loop_for_header(header)?.body.clone();
        let preds: Vec<String> = body
            .iter()
            .filter(|name| {
                blocks
                    .iter()
                    .find(|b| &b.name == *name)
                    .map(|b| block_successors(b).iter().any(|s| s == exit))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        if preds.is_empty() {
            return None;
        }
        let is_redirected = |pred: &str| preds.iter().any(|p| p == pred);

        let new_name = format!("{SPLIT_PREFIX}{counter}");
        *counter += 1;

        let exit_idx = blocks.iter().position(|b| b.name == exit)?;
        // Drive the exit-phi surgery off the exit block's TYPED carrier (line order preserved by `t.insts`).
        // For each phi with a redirected incoming, mint a merged pass-through phi over the redirected typed
        // incomings and rewrite the exit phi to `kept + [ merged, {new_name} ]`. Aggregate phis
        // (`phi_incoming: None`) carry no typed incoming list, so they are skipped here — such a phi fails
        // primary emit and routes to retry regardless (the carrier is the sole emission substrate), so the
        // primary path never emits this block. Each primitive has a `== re-lower` unit test.
        type TypedIncomings = Vec<(crate::native::ir::LlValue, String)>;
        // (merged pass-through phi name, phi type, redirected typed incomings).
        let mut passthrough_merges: Vec<(String, crate::native::ir::LlType, TypedIncomings)> =
            Vec::new();
        // (exit phi dst, kept typed incomings + the `[ merged, {new_name} ]` funnel).
        let mut exit_rewrites: Vec<(String, TypedIncomings)> = Vec::new();
        if let Some(t) = &blocks[exit_idx].typed {
            for inst in &t.insts {
                let (Some(dst), Some((ty, inc))) =
                    (inst.result.clone(), inst.phi_incoming().clone())
                else {
                    continue;
                };
                let (typed_red, mut kept_plus): (Vec<_>, Vec<_>) =
                    inc.into_iter().partition(|(_, pred)| is_redirected(pred));
                if typed_red.is_empty() {
                    continue;
                }
                let merged = format!("{new_name}.phi{}", passthrough_merges.len());
                kept_plus.push((
                    crate::native::ir::LlValue::Local(merged.clone()),
                    new_name.clone(),
                ));
                passthrough_merges.push((merged, ty, typed_red));
                exit_rewrites.push((dst, kept_plus));
            }
        }
        if let Some(t) = blocks[exit_idx].typed_mut() {
            for (dst, kept_plus) in &exit_rewrites {
                t.set_phi_incomings(dst, kept_plus);
            }
        }

        // Redirect the inner-loop predecessors' terminators from `exit` to the pass-through.
        for b in blocks.iter_mut() {
            if preds.iter().any(|p| p == &b.name) {
                if let Some(t) = b.typed_mut() {
                    t.redirect_successor(exit, &new_name);
                }
            }
        }

        // Insert the pass-through block: merged phis followed by the branch to the original exit. Build its
        // carrier by pushing the typed merged phis onto a fresh `br label {exit}` carrier (byte-identical to
        // lowering the merged phi lines + terminator — `push_value_phi`'s `== re-lower` test).
        let insert_at = blocks
            .iter()
            .position(|b| b.name == exit)
            .unwrap_or(blocks.len());
        let mut blk = crate::native::tir::lower_block_carrier(
            &new_name,
            &[format!("br label {exit}")],
            &std::collections::HashMap::new(),
        )?;
        for (merged, ty, typed_red) in &passthrough_merges {
            blk.push_value_phi(merged, ty, typed_red);
        }
        blocks.insert(
            insert_at,
            BodyBlock {
                name: new_name.clone(),
                role: role_for_name(&new_name),
                typed: Some(blk.into()),
            },
        );

        Some(new_name)
    })
}

/// True if `name`'s block has a phi node (an instruction of the form `%x = phi ...`). Reads the typed
/// carrier's instructions when populated (a `phi` line lowers to a `TirInst` with `opcode == "phi"`,
/// so this is identical to the line scan by construction — census 0), falling back to the line scan
/// only for a not-yet-populated block (pre-`populate_typed_carriers`).
pub(in crate::native) fn block_has_phi(blocks: &[BodyBlock], name: &str) -> bool {
    blocks
        .iter()
        .find(|b| b.name == name)
        .map(|b| {
            b.typed
                .as_ref()
                .is_some_and(|t| t.insts.iter().any(|i| i.is_phi()))
        })
        .unwrap_or(false)
}
