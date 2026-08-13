//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

/// Synthesize a single structured merge for a loop with exactly two exit targets (the
/// `Restructure::MultipleExits` class, k == 2). SPIR-V's `OpLoopMerge` names exactly one merge block,
/// so a loop that branches out to two distinct targets must be funnelled through one block that then
/// dispatches. This inserts a fresh merge `M`: every in-loop edge to either exit is redirected
/// straight to `M` (NOT via a trampoline — a trampoline would itself become a new loop exit), and `M`
/// carries an `i1` selector phi keyed by predecessor block plus a conditional branch back out to the
/// two real targets. The loop is then single-exit (`M`).
///
/// Scope: exactly two exits. A block that branches directly to BOTH exits is first split at those
/// critical exit edges, giving the dispatch selector one unambiguous predecessor per exit. A
/// phi-CARRYING exit is handled: the new `M`→exit edge would leave the
/// exit's phi an incoming short, so each such phi is funnelled through a fresh value phi in `M` (real
/// value for the predecessors targeting THIS exit, `undef` for those targeting the other — dynamically
/// unreachable, the selector routes them elsewhere), and the exit phi replaces its redirected incomings
/// with a single `[ merged, M ]`. Anything outside this scope returns `None`, leaving the loop
/// unstructured so `structured_plan` falls back.
///
/// The critical-edge split is structural and SSA-preserving: each new edge block only branches to its
/// original exit, and that exit's phi predecessor label is changed from the conditional source to its
/// corresponding edge block. Values defined by the source still dominate the new block and the phi
/// incoming, so no value phi is needed until the ordinary dispatch funnel below.
pub(in crate::native) fn split_multi_exit_critical_edges(
    blocks: &mut Vec<BodyBlock>,
    forest: &LoopForest,
    header: &str,
    exits: &[String],
    counter: &mut usize,
) -> Option<Vec<(String, usize)>> {
    if exits.len() != 2 {
        return None;
    }
    let body: HashSet<&str> = forest
        .loop_for_header(header)?
        .body
        .iter()
        .map(String::as_str)
        .collect();
    let mut splits: Vec<(String, Vec<usize>)> = Vec::new();

    for block in blocks.iter().filter(|b| body.contains(b.name.as_str())) {
        let successors = block_successors(block);
        let hit: Vec<usize> = exits
            .iter()
            .enumerate()
            .filter_map(|(index, exit)| {
                successors
                    .iter()
                    .any(|target| target == exit)
                    .then_some(index)
            })
            .collect();
        if hit.len() < 2 {
            continue;
        }
        splits.push((block.name.clone(), hit));
    }
    if splits.is_empty() {
        return Some(Vec::new());
    }

    let mut edge_blocks = Vec::new();
    let mut edge_preds = Vec::new();
    for (source, targets) in splits {
        let mut edges = Vec::with_capacity(targets.len());
        for index in targets {
            let edge = format!("{EXIT_EDGE_PREFIX}{counter}");
            *counter += 1;
            edges.push((index, edge));
        }

        let source_block = blocks.iter_mut().find(|b| b.name == source)?;
        // Redirect the terminator on the carrier (typed dual of the former string redirect).
        if let Some(t) = source_block.typed_mut() {
            for (index, edge) in &edges {
                t.redirect_successor(&exits[*index], edge);
            }
        }

        for (index, edge) in &edges {
            let exit = blocks.iter_mut().find(|b| b.name == exits[*index])?;
            if let Some(t) = exit.typed_mut() {
                t.rewrite_phi_predecessor(&source, edge);
            }
            let edge_lines = vec![format!("br label {}", exits[*index])];
            edge_blocks.push(BodyBlock {
                typed: crate::native::tir::lower_block_carrier(
                    edge,
                    &edge_lines,
                    &std::collections::HashMap::new(),
                )
                .map(Into::into),
                name: edge.clone(),
                role: role_for_name(edge),
            });
            edge_preds.push((edge.clone(), *index));
        }
    }
    blocks.extend(edge_blocks);
    Some(edge_preds)
}

pub(in crate::native) fn synth_multi_exit_merge(
    blocks: &mut Vec<BodyBlock>,
    forest: &LoopForest,
    header: &str,
    exits: &[String],
    counter: &mut usize,
) -> Option<String> {
    if exits.len() != 2 {
        return None;
    }
    // Deterministic exit ordering (the forest's exit list order is traversal-dependent).
    let mut exits = exits.to_vec();
    exits.sort();
    let exits = &exits[..];
    let critical_edge_preds =
        split_multi_exit_critical_edges(blocks, forest, header, exits, counter)?;
    let body: HashSet<&str> = forest
        .loop_for_header(header)?
        .body
        .iter()
        .map(String::as_str)
        .collect();

    // For each in-loop block, the single exit index it branches to (bail if a block targets both).
    let mut preds: Vec<(String, usize)> = Vec::new();
    let critical_edge_set: HashSet<&str> = critical_edge_preds
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    for b in blocks.iter() {
        if !body.contains(b.name.as_str()) && !critical_edge_set.contains(b.name.as_str()) {
            continue;
        }
        let mut hit: Option<usize> = None;
        for succ in block_successors(b) {
            if let Some(idx) = exits.iter().position(|e| e == &succ) {
                match hit {
                    Some(prev) if prev != idx => return None,
                    _ => hit = Some(idx),
                }
            }
        }
        if let Some(idx) = hit {
            preds.push((b.name.clone(), idx));
        }
    }
    if preds.is_empty() {
        return None;
    }

    let merge = format!("{SPLIT_PREFIX}{counter}");
    *counter += 1;
    let sel = format!("{EXIT_SEL_PREFIX}{counter}");
    *counter += 1;

    // Redirect every in-loop exit edge directly to the merge.
    for b in blocks.iter_mut() {
        if let Some((_, idx)) = preds.iter().find(|(n, _)| n == &b.name) {
            if let Some(t) = b.typed_mut() {
                t.redirect_successor(&exits[*idx], &merge);
            }
        }
    }

    // After the in-loop redirect, body blocks no longer target the exits — only outer preds remain on
    // each exit. `M` branches DIRECTLY to both real exits and funnels each exit's phis via the single
    // `M` edge (`exit_edge_pred[i]` = `M`). When an exit is ALSO reached from OUTSIDE the loop body (an
    // outer edge), `M` does not dominate it, so the dispatch selection cannot use it as a private arm —
    // the M2 `selection:cross-arm-shared` residual (`02/590c6cf2`). That shared-exit privatization is
    // done by the caller [`privatize_dispatch_shared_exits`]
    // by cloning the exit's dominated forward region into an `M`-dominated copy; a bare pass-through
    // trampoline is NOT enough because the shared exit block itself stays non-dominated (dead-end #6).
    let dispatch_targets = [exits[0].clone(), exits[1].clone()];
    let exit_edge_pred = [merge.clone(), merge.clone()];

    // For each exit that carries phis, funnel the redirected predecessors' incomings through a fresh
    // value phi in the merge `M` and rewrite the exit phi to take that merged value via the single
    // dispatch edge (from `M` directly, or from the private trampoline when the exit is shared).
    // `M` branches to BOTH dispatch targets, so each value phi must cover ALL of `M`'s predecessors:
    // the real incoming for the predecessors targeting THIS exit, and `undef` for those targeting
    // the other exit (dynamically unreachable here — the selector routes them out the other side).
    // STEP-1 (kb "STEP-1/STEP-2 decomposition"): drive the merge `M`'s value-phi carriers and each exit
    // phi's carrier rebuild from the exit blocks' TYPED incomings rather than re-lowering the rewritten
    // `.lines` text, so `M`'s carrier no longer depends on the `.lines` field. `carrier_ok` is all-or-
    // nothing over ALL funnelled value phis: any funnelled exit phi whose carrier `phi_incoming` did not
    // lower (aggregate → `None`, routes to retry) forces the full text fallback for `M`. Byte-identical
    // either way (each `phi_edit` primitive has a `== re-lower` unit test; BC drift NONE proves it
    // broadly).
    // Drive the funnel off each exit block's TYPED carrier (line order preserved by `t.insts`). For each
    // exit phi with an incoming from an in-loop predecessor rerouted to `M` for THIS exit, mint `M`'s
    // value phi (over ALL of `M`'s preds — the redirected value for a pred targeting THIS exit, `undef`
    // otherwise) and rewrite the exit phi to `kept + [ merged, edge_pred ]`. Aggregate phis
    // (`phi_incoming: None`) carry no typed incoming list, so they are skipped — such a phi fails primary
    // emit → retry regardless (the carrier is the sole emission substrate). Each primitive has a
    // `== re-lower` unit test.
    type TypedIncomings = Vec<(crate::native::ir::LlValue, String)>;
    // `M`'s value phis, in the order the exit phis are visited: (merged name, phi type, incomings/ALL preds).
    let mut typed_value_phis: Vec<(String, crate::native::ir::LlType, TypedIncomings)> = Vec::new();
    for (ei, e) in exits.iter().enumerate() {
        let Some(exit_idx) = blocks.iter().position(|b| &b.name == e) else {
            continue;
        };
        let edge_pred = exit_edge_pred[ei].clone();
        // (exit phi dst, kept + `[merged, edge_pred]` typed incomings).
        let mut exit_typed_merges: Vec<(String, TypedIncomings)> = Vec::new();
        if let Some(t) = &blocks[exit_idx].typed {
            for inst in &t.insts {
                let (Some(dst), Some((cty, inc))) =
                    (inst.result.clone(), inst.phi_incoming.clone())
                else {
                    continue;
                };
                // Redirected incomings: from an in-loop predecessor we rerouted to `M` for THIS exit.
                let is_this_exit_pred =
                    |p: &str| preds.iter().any(|(name, idx)| name == p && *idx == ei);
                let (typed_red, mut kept_plus): (Vec<_>, Vec<_>) =
                    inc.into_iter().partition(|(_, p)| is_this_exit_pred(p));
                if typed_red.is_empty() {
                    continue;
                }
                let merged = format!("%metal2vulkan.exitphi.{counter}");
                *counter += 1;
                // `M`'s value phi covers ALL of `M`'s preds: the redirected value for a pred targeting
                // THIS exit, `undef` for one targeting the other exit (dynamically unreachable).
                let merged_typed: TypedIncomings = preds
                    .iter()
                    .map(|(pname, pidx)| {
                        let v = if *pidx == ei {
                            typed_red
                                .iter()
                                .find(|(_, p)| p == pname)
                                .map(|(v, _)| v.clone())
                                .unwrap_or(crate::native::ir::LlValue::Undef)
                        } else {
                            crate::native::ir::LlValue::Undef
                        };
                        (v, pname.clone())
                    })
                    .collect();
                // Predecessor label is `edge_pred` (`M`, or the private trampoline); `merged` is defined
                // in `M` which dominates both. Outer incomings stay on `kept_plus`.
                kept_plus.push((
                    crate::native::ir::LlValue::Local(merged.clone()),
                    edge_pred.clone(),
                ));
                exit_typed_merges.push((dst, kept_plus));
                typed_value_phis.push((merged, cty, merged_typed));
            }
        }
        if let Some(t) = blocks[exit_idx].typed_mut() {
            for (dst, kept_plus) in &exit_typed_merges {
                t.set_phi_incomings(dst, kept_plus);
            }
        }
    }

    // Merge block: the funnelled value phis (if any), then an i1 selector phi (true = exit 0, false =
    // exit 1) over its redirected predecessors, then a conditional branch to the (possibly
    // trampoline-privatized) dispatch targets. Build `M`'s carrier by pushing the typed value phis then
    // the i1 selector phi onto a fresh conditional-branch carrier (value phis precede the selector;
    // each `push_value_phi` is byte-identical to lowering the printed phi line).
    let selector_typed: TypedIncomings = preds
        .iter()
        .map(|(name, idx)| (crate::native::ir::LlValue::Bool(*idx == 0), name.clone()))
        .collect();
    let branch_line = format!(
        "br i1 {sel}, label {}, label {}",
        dispatch_targets[0], dispatch_targets[1]
    );
    let mut blk = crate::native::tir::lower_block_carrier(
        &merge,
        &[branch_line],
        &std::collections::HashMap::new(),
    )?;
    for (mname, ty, merged_typed) in &typed_value_phis {
        blk.push_value_phi(mname, ty, merged_typed);
    }
    blk.push_value_phi(&sel, &crate::native::ir::LlType::Int(1), &selector_typed);
    let merge_block = BodyBlock {
        name: merge.clone(),
        role: role_for_name(&merge),
        typed: Some(blk.into()),
    };
    // Place it just before the first exit target so it sits in a natural position.
    let insert_at = blocks
        .iter()
        .position(|b| b.name == exits[0] || b.name == exits[1])
        .unwrap_or(blocks.len());
    blocks.insert(insert_at, merge_block);

    Some(merge)
}

/// Unify a loop's multiple back-edges (latches) into ONE synthesized latch so the loop can carry a
/// valid single `OpLoopMerge` continue target (`Restructure::MultipleLatches`). SPIR-V names exactly
/// one continue target, so a loop whose header has several back-edge predecessors must funnel them
/// through one block. This inserts a fresh latch `L`: every latch's edge to the header is redirected to
/// `L`, and `L` branches unconditionally back to the header. The header's phis carried a per-latch
/// incoming; those are funnelled through fresh value phis in `L` (one per header phi with a latch
/// incoming), and each header phi replaces its redirected latch incomings with a single `[ merged, L ]`,
/// keeping its non-latch (preheader) incomings. The loop is then single-latch (`L`).
///
/// Scope: the caller gates on a pure `[MultipleLatches]` loop. Returns `None` (leaving the loop
/// unstructured for the fallback) if a latch's terminator does not actually name the header among its
/// successors — so a well-formed redirect is always possible when it returns `Some`.
pub(in crate::native) fn synth_multi_latch_continue(
    blocks: &mut Vec<BodyBlock>,
    header: &str,
    latches: &[String],
    counter: &mut usize,
) -> Option<String> {
    if latches.len() < 2 {
        return None;
    }
    let mut latches = latches.to_vec();
    latches.sort();
    latches.dedup();
    let latch_set: HashSet<&str> = latches.iter().map(String::as_str).collect();

    // Every latch must currently branch to the header (a back-edge). If one does not, we cannot cleanly
    // isolate its back-edge, so bail and leave the loop for the fallback.
    for l in &latches {
        let b = blocks.iter().find(|b| &b.name == l)?;
        if !block_successors(b).iter().any(|s| s == header) {
            return None;
        }
    }

    let hidx = blocks.iter().position(|b| b.name == header)?;
    let new_latch = format!("{SPLIT_PREFIX}{counter}");
    *counter += 1;

    // Header phi surgery, driven off the header block's TYPED carrier (line order preserved by `t.insts`).
    // For each header phi with a latch (redirected) incoming, mint a merged value phi in `L` over ALL
    // latches (`undef` for a latch this phi does not carry) and rewrite the header phi to
    // `kept..., [ merged, L ]`. Aggregate phis (`phi_incoming: None`) carry no typed incoming list, so
    // they are skipped — such a phi fails primary emit → retry regardless (the carrier is the sole
    // emission substrate). Each primitive has a `== re-lower` unit test.
    type TypedIncomings = Vec<(crate::native::ir::LlValue, String)>;
    // `L`'s value phis, in the order the header phis are visited: (merged name, phi type, over ALL latches).
    let mut typed_value_phis: Vec<(String, crate::native::ir::LlType, TypedIncomings)> = Vec::new();
    // (header phi dst, kept + `[merged, L]` typed incomings).
    let mut header_typed_merges: Vec<(String, TypedIncomings)> = Vec::new();
    if let Some(t) = &blocks[hidx].typed {
        for inst in &t.insts {
            let (Some(dst), Some((cty, inc))) = (inst.result.clone(), inst.phi_incoming.clone())
            else {
                continue;
            };
            let (typed_red, mut kept_plus): (Vec<_>, Vec<_>) = inc
                .into_iter()
                .partition(|(_, p)| latch_set.contains(p.as_str()));
            if typed_red.is_empty() {
                continue;
            }
            let merged = format!("%metal2vulkan.latchphi.{counter}");
            *counter += 1;
            // Every latch now branches to `L`, so the merged phi covers ALL of them (a latch missing
            // from this phi's incomings gets `undef` — it does not carry this value on its back-edge).
            let merged_typed: TypedIncomings = latches
                .iter()
                .map(|lname| {
                    let v = typed_red
                        .iter()
                        .find(|(_, p)| p == lname)
                        .map(|(v, _)| v.clone())
                        .unwrap_or(crate::native::ir::LlValue::Undef);
                    (v, lname.clone())
                })
                .collect();
            kept_plus.push((
                crate::native::ir::LlValue::Local(merged.clone()),
                new_latch.clone(),
            ));
            header_typed_merges.push((dst, kept_plus));
            typed_value_phis.push((merged, cty, merged_typed));
        }
    }
    if let Some(t) = blocks[hidx].typed_mut() {
        for (dst, kept_plus) in &header_typed_merges {
            t.set_phi_incomings(dst, kept_plus);
        }
    }

    // Redirect each latch's header edge to the new single latch `L`.
    for l in &latches {
        if let Some(b) = blocks.iter_mut().find(|b| &b.name == l) {
            if let Some(t) = b.typed_mut() {
                t.redirect_successor(header, &new_latch);
            }
        }
    }

    // `L`: the funnelled value phis, then an unconditional branch back to the header. Insert it just
    // before the header (or after it when the header is the function entry, keeping `blocks[0]` intact);
    // `analyze` builds edges from terminators, so the vector position only matters for the entry. Build
    // `L`'s carrier by pushing the typed merged phis onto a fresh `br label {header}` carrier
    // (byte-identical to lowering the merged phi lines + terminator — `push_value_phi`'s `== re-lower`).
    let insert_at = hidx.max(1);
    let latch_block = {
        let mut blk = crate::native::tir::lower_block_carrier(
            &new_latch,
            &[format!("br label {header}")],
            &std::collections::HashMap::new(),
        )?;
        for (merged, ty, merged_typed) in &typed_value_phis {
            blk.push_value_phi(merged, ty, merged_typed);
        }
        BodyBlock {
            name: new_latch.clone(),
            role: role_for_name(&new_latch),
            typed: Some(blk.into()),
        }
    };
    blocks.insert(insert_at, latch_block);

    Some(new_latch)
}
