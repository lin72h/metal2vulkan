//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

/// The SSA defs of a block, carrier-first (`inst.result`), line fallback (`line_def`) for the
/// pre-`populate` window — the def half of the clone rename map (kb "STEP-1/STEP-2 decomposition").
/// Byte-identical: every non-terminator def line lowers to an `inst` with that `result` and a
/// terminator carries no def, so the carrier's inst results are exactly the `line_def`s of the block's
/// lines (BC drift NONE). Sourcing the rename map from the carrier keeps the clone carriers
/// (`c.rename(&rename)`) independent of the `.lines` field.
pub(in crate::native) fn block_defs(b: &BodyBlock) -> Vec<String> {
    b.typed
        .as_ref()
        .map(|t| t.insts.iter().filter_map(|i| i.result.clone()).collect())
        .unwrap_or_default()
}

/// Read-only explanation of whether [`privatize_dominated_region`] would clone `(header, arm)`, and
/// which guard would decline first. This keeps straddle/frontier diagnostics tied to the production
/// clone's actual bounds instead of duplicating those constants in the caller.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::native) struct DominatedRegionCloneWitness {
    pub header: String,
    pub arm: String,
    pub reason: &'static str,
    pub region_blocks: usize,
    pub region_cap: usize,
    pub boundary_count: usize,
    pub boundary_cap: usize,
    pub boundary_sample: Vec<String>,
    pub redirect_count: usize,
    pub external_pred_count: usize,
    pub arm_cycle_pred_count: usize,
    pub first_missing_carrier: Option<String>,
    pub first_empty_phi_block: Option<String>,
}

pub(in crate::native) fn dominated_region_clone_witness(
    blocks: &[BodyBlock],
    header: &str,
    arm: &str,
) -> DominatedRegionCloneWitness {
    let forest = analyze(blocks);
    let preds = predecessors(blocks);
    let by_name: HashMap<&str, &BodyBlock> = blocks.iter().map(|b| (b.name.as_str(), b)).collect();
    let names: HashSet<&str> = blocks.iter().map(|b| b.name.as_str()).collect();

    if !by_name.contains_key(arm) {
        return DominatedRegionCloneWitness {
            header: header.to_string(),
            arm: arm.to_string(),
            reason: "arm_missing",
            region_blocks: 0,
            region_cap: 0,
            boundary_count: 0,
            boundary_cap: MAX_REGION_BOUNDARIES,
            boundary_sample: Vec::new(),
            redirect_count: 0,
            external_pred_count: 0,
            arm_cycle_pred_count: 0,
            first_missing_carrier: None,
            first_empty_phi_block: None,
        };
    }

    let mut region: HashSet<String> = HashSet::new();
    let mut boundary: HashSet<String> = HashSet::new();
    let mut stack = vec![arm.to_string()];
    while let Some(n) = stack.pop() {
        if !region.insert(n.clone()) {
            continue;
        }
        let Some(b) = by_name.get(n.as_str()) else {
            continue;
        };
        for s in block_successors(b) {
            if !names.contains(s.as_str()) {
                continue;
            }
            if forest.dominates(arm, &s) {
                if !region.contains(&s) {
                    stack.push(s);
                }
            } else {
                boundary.insert(s);
            }
        }
    }

    let mut boundary_sample = boundary.iter().cloned().collect::<Vec<_>>();
    boundary_sample.sort();
    boundary_sample.truncate(8);

    let region_cap = if boundary.len() == 1 {
        MAX_SINGLE_BOUNDARY_REGION_BLOCKS
    } else {
        MAX_REGION_BLOCKS
    };
    let arm_cycle_pred_count = preds
        .get(arm)
        .into_iter()
        .flatten()
        .filter(|p| region.contains(*p))
        .count();
    let redirect: HashSet<String> = preds
        .get(arm)
        .into_iter()
        .flatten()
        .filter(|p| forest.dominates(header, p))
        .cloned()
        .collect();
    let external_pred_count = preds
        .get(arm)
        .into_iter()
        .flatten()
        .filter(|p| !redirect.contains(*p))
        .count();

    let mut first_missing_carrier = None;
    let mut first_empty_phi_block = None;
    if !boundary.is_empty()
        && boundary.len() <= MAX_REGION_BOUNDARIES
        && region.len() <= region_cap
        && arm_cycle_pred_count == 0
        && !redirect.is_empty()
        && external_pred_count > 0
    {
        for src in blocks.iter().filter(|block| region.contains(&block.name)) {
            let keep = |pred: &str| {
                if src.name == arm {
                    redirect.contains(pred)
                } else {
                    region.contains(pred)
                }
            };
            let Some(src_carrier) = &src.typed else {
                first_missing_carrier = Some(src.name.clone());
                break;
            };
            if src_carrier.insts.iter().any(|inst| {
                inst.phi_incoming.as_ref().is_some_and(|(_, incoming)| {
                    !incoming.is_empty() && !incoming.iter().any(|(_, p)| keep(p))
                })
            }) {
                first_empty_phi_block = Some(src.name.clone());
                break;
            }
        }
    }

    let reason = if boundary.is_empty() {
        "boundary_empty"
    } else if boundary.len() > MAX_REGION_BOUNDARIES {
        "boundary_over_cap"
    } else if region.len() > region_cap {
        "region_over_cap"
    } else if arm_cycle_pred_count > 0 {
        "arm_cycle"
    } else if redirect.is_empty() {
        "redirect_empty"
    } else if external_pred_count == 0 {
        "no_external_pred"
    } else if first_missing_carrier.is_some() {
        "missing_carrier"
    } else if first_empty_phi_block.is_some() {
        "phi_empty_after_partition"
    } else {
        "cloneable"
    };

    DominatedRegionCloneWitness {
        header: header.to_string(),
        arm: arm.to_string(),
        reason,
        region_blocks: region.len(),
        region_cap,
        boundary_count: boundary.len(),
        boundary_cap: MAX_REGION_BOUNDARIES,
        boundary_sample,
        redirect_count: redirect.len(),
        external_pred_count,
        arm_cycle_pred_count,
        first_missing_carrier,
        first_empty_phi_block,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::native) struct RegionCrossArmFixpointWitness {
    pub input_blocks: usize,
    pub output_blocks: usize,
    pub max_blocks: usize,
    pub rounds: usize,
    pub stop_reason: &'static str,
    pub next_blocks: Option<usize>,
    pub stop_candidate: Option<DominatedRegionCloneWitness>,
}

/// Privatize the forward-closed region rooted at `arm` for `header`'s in-construct entries. Returns
/// the new block list, or `None` if the region is not cleanly single-entry / loop-free-entry /
/// within budget.
pub(in crate::native) fn privatize_region(
    blocks: &[BodyBlock],
    header: &str,
    arm: &str,
    counter: &mut usize,
) -> Option<Vec<BodyBlock>> {
    let forest = analyze(blocks);
    let preds = predecessors(blocks);
    let by_name: HashMap<&str, &BodyBlock> = blocks.iter().map(|b| (b.name.as_str(), b)).collect();

    // Forward closure R from `arm`.
    let mut region: HashSet<String> = HashSet::new();
    let mut stack = vec![arm.to_string()];
    while let Some(n) = stack.pop() {
        if !region.insert(n.clone()) {
            continue;
        }
        let Some(b) = by_name.get(n.as_str()) else {
            continue;
        };
        for s in block_successors(b) {
            if by_name.contains_key(s.as_str()) && !region.contains(&s) {
                stack.push(s);
            }
        }
    }
    if region.len() > MAX_REGION_BLOCKS {
        return None;
    }

    // `arm` must not be inside a cycle within R (no predecessor of `arm` in R): the clone `arm_clone`
    // becomes the SOLE entry of the cloned region `R'`, so it must take only external (redirected)
    // incomings. An `arm` with an in-R back-edge would also need that edge mirrored into the clone —
    // the loop-aware variant, left to a future increment. (find_cross_arm already excludes loop
    // *headers*; this excludes the rarer reducible case where `arm` is a non-header cycle member.)
    if preds
        .get(arm)
        .into_iter()
        .flatten()
        .any(|p| region.contains(p))
    {
        return None;
    }

    // The predecessors of `arm` to redirect to the clone: those dominated by `header` (inside its
    // construct). The cross-arm guarantees `arm` also has a predecessor NOT dominated by `header`
    // (the external entry), so the original `arm` stays reachable.
    let redirect: HashSet<String> = preds
        .get(arm)
        .into_iter()
        .flatten()
        .filter(|p| forest.dominates(header, p))
        .cloned()
        .collect();
    if redirect.is_empty() {
        return None;
    }
    // `arm` must keep at least one non-redirected predecessor, else it was already private to
    // `header` (not a cross-arm). Defensive: bail rather than orphan it.
    let keeps_original = preds
        .get(arm)
        .into_iter()
        .flatten()
        .any(|p| !redirect.contains(p));
    if !keeps_original {
        return None;
    }

    // Build the rename map: every block label in R + every SSA value defined in R gets a fresh name.
    let id = *counter;
    *counter += 1;
    let mut rename: HashMap<String, String> = HashMap::new();
    let ordered_region: Vec<&BodyBlock> = blocks
        .iter()
        .filter(|block| region.contains(&block.name))
        .collect();
    for block in &ordered_region {
        rename.insert(block.name.clone(), fresh(&block.name, id));
        for def in block_defs(block) {
            rename.insert(def.clone(), fresh(&def, id));
        }
    }

    let mut out: Vec<BodyBlock> = blocks.to_vec();

    // 1. Original `arm`: drop the phi incomings coming from the redirected predecessors (they now
    //    flow into the clone instead) — on the carrier, the sole substrate.
    for b in out.iter_mut() {
        if b.name == arm {
            if let Some(t) = &mut b.typed {
                t.rebuild_phi_incomings(|pred| !redirect.contains(pred));
            }
        }
    }

    // 2. Redirect each redirected predecessor's terminator edge `arm -> arm_clone` (on the carrier).
    let arm_clone = rename.get(arm).cloned()?;
    for b in out.iter_mut() {
        if redirect.contains(&b.name) {
            if let Some(t) = &mut b.typed {
                t.redirect_successor(arm, &arm_clone);
            }
        }
    }

    // 3. Emit the cloned region blocks. Because the clone is the SOLE entry of `R'`, `arm_clone`
    //    dominates every cloned block, so external values (defined outside R, which dominate `arm` and
    //    thus dominate `arm_clone`) still dominate their cloned uses — no exit-phi synthesis needed.
    //    Each cloned block keeps only the phi incomings whose predecessor actually branches to its
    //    clone: for `arm`, the redirected external predecessors; for every other block, its in-region
    //    predecessors (the external predecessors of a shared block keep flowing into the ORIGINAL,
    //    which is untouched). Then all `%`-tokens are renamed to the clone namespace.
    for src in &ordered_region {
        let n = &src.name;
        let clone_name = rename.get(n)?.clone();
        // The partition predicate on this clone's phis: for `arm`, the redirected external predecessors;
        // for every other block, its in-region predecessors.
        let keep = |pred: &str| {
            if n == arm {
                redirect.contains(pred)
            } else {
                region.contains(pred)
            }
        };
        // Clone from the source's typed carrier (the sole substrate). Bail (rather than emit a malformed
        // clone) if any phi would lose ALL its incomings — the carrier dual of the text `rebuild_phi`
        // returning `None` on an empty keep-set. A source block without a carrier cannot be cloned.
        let Some(src_carrier) = &src.typed else {
            return None;
        };
        for inst in &src_carrier.insts {
            if let Some((_, incoming)) = &inst.phi_incoming {
                if !incoming.is_empty() && !incoming.iter().any(|(_, p)| keep(p)) {
                    return None;
                }
            }
        }
        // A rename-cloned block: its name is the carried identity, so the role tracks the (possibly
        // still-`texitret`-prefixed, possibly renamed-away) clone name.
        let role = role_for_name(&clone_name);
        let mut c = src_carrier.clone();
        c.rebuild_phi_incomings(keep);
        c.rename(&rename);
        out.push(BodyBlock {
            name: clone_name,
            role,
            typed: Some(c),
        });
    }

    Some(out)
}

/// Merge-PRESERVING cross-arm privatization for the default path. Clones only the region **dominated
/// by `arm`** — `arm`'s private forward subtree — stopping at the reconvergence boundary. Where
/// [`privatize_region`] clones the FULL forward closure (which sweeps in the shared merge and DESTROYS
/// reconvergence, the zero-win outcome the `--cfg-clone` funnel measured), this keeps every **boundary
/// block** (a successor of the region NOT dominated by `arm`, i.e. shared with an enclosing construct)
/// intact as the merge and only mirrors that boundary block's phi incomings for the cloned
/// predecessors. This generalizes [`privatize_trivial`] (whose region is the degenerate `{arm}` with a
/// single boundary `S`) to an arbitrary dominated subtree: multi-successor arms and def-carrying arms
/// whose values only reach the boundary through phis.
///
/// Soundness (no exit-phi synthesis needed): every block in the region is dominated by `arm`, hence by
/// the sole clone entry `arm_clone`, so external values (which dominate `arm`) still dominate their
/// cloned uses. A value DEFINED in the region cannot be used past the boundary except as a boundary
/// phi incoming — a post-boundary block is dominated by the boundary (a reconvergence with an external
/// predecessor), so a region value reaching it would have to dominate the boundary, which it cannot
/// (the boundary has a non-region predecessor). Mirroring the boundary phis therefore closes SSA.
/// Returns `None` if the region is over budget, `arm` sits in an in-region cycle, or there is no
/// dominated predecessor to redirect (or none to keep) — the same guards as [`privatize_region`].
pub(in crate::native) fn privatize_dominated_region(
    blocks: &[BodyBlock],
    header: &str,
    arm: &str,
    counter: &mut usize,
) -> Option<Vec<BodyBlock>> {
    let forest = analyze(blocks);
    let preds = predecessors(blocks);
    let by_name: HashMap<&str, &BodyBlock> = blocks.iter().map(|b| (b.name.as_str(), b)).collect();
    let names: HashSet<&str> = blocks.iter().map(|b| b.name.as_str()).collect();

    // Region = blocks dominated by `arm` reachable from `arm`. Descent stops at a non-dominated
    // successor: that block is a reconvergence boundary, recorded and kept intact as the shared merge.
    let mut region: HashSet<String> = HashSet::new();
    let mut boundary: HashSet<String> = HashSet::new();
    let mut stack = vec![arm.to_string()];
    while let Some(n) = stack.pop() {
        if !region.insert(n.clone()) {
            continue;
        }
        let Some(b) = by_name.get(n.as_str()) else {
            continue;
        };
        for s in block_successors(b) {
            if !names.contains(s.as_str()) {
                continue;
            }
            if forest.dominates(arm, &s) {
                if !region.contains(&s) {
                    stack.push(s);
                }
            } else {
                boundary.insert(s);
            }
        }
    }
    // Allow a small number of reconvergence boundaries (not just the single clean nested case). A region
    // that exits to several distinct outside blocks can have one boundary be the true merge while another
    // escapes an enclosing selection non-structurally (the "block exits the selection ... but not via a
    // structured exit" over-admission `structured_plan`'s self-checks do not fully catch); the
    // mirror/redirect/clone logic below already loops over ALL boundaries, and the plan self-checks +
    // `--primary-val-check` floor catch any dead-#6 straddle-escape over-admission.
    if boundary.is_empty() || boundary.len() > MAX_REGION_BOUNDARIES {
        return None;
    }
    let region_cap = if boundary.len() == 1 {
        MAX_SINGLE_BOUNDARY_REGION_BLOCKS
    } else {
        MAX_REGION_BLOCKS
    };
    if region.len() > region_cap {
        return None;
    }
    // `arm` must not sit in an in-region cycle (no in-region predecessor); such a back-edge would need
    // mirroring into the clone (the loop-aware variant). `find_cross_arm` already excludes loop headers.
    if preds
        .get(arm)
        .into_iter()
        .flatten()
        .any(|p| region.contains(p))
    {
        return None;
    }

    let redirect: HashSet<String> = preds
        .get(arm)
        .into_iter()
        .flatten()
        .filter(|p| forest.dominates(header, p))
        .cloned()
        .collect();
    if redirect.is_empty() {
        return None;
    }
    let keeps_original = preds
        .get(arm)
        .into_iter()
        .flatten()
        .any(|p| !redirect.contains(p));
    if !keeps_original {
        return None;
    }

    // Rename every region label + every SSA value defined in the region into a fresh clone namespace.
    let id = *counter;
    *counter += 1;
    let mut rename: HashMap<String, String> = HashMap::new();
    let ordered_region: Vec<&BodyBlock> = blocks
        .iter()
        .filter(|block| region.contains(&block.name))
        .collect();
    for block in &ordered_region {
        rename.insert(block.name.clone(), fresh(&block.name, id));
        for def in block_defs(block) {
            rename.insert(def.clone(), fresh(&def, id));
        }
    }
    let arm_clone = rename.get(arm).cloned()?;

    let mut out: Vec<BodyBlock> = blocks.to_vec();

    // 1. Original `arm`: drop the phi incomings from redirected preds (they now flow into the clone) —
    //    on the carrier, the sole substrate.
    for b in out.iter_mut() {
        if b.name == arm {
            if let Some(t) = &mut b.typed {
                t.rebuild_phi_incomings(|p| !redirect.contains(p));
            }
        }
    }
    // 2. Redirect each redirected predecessor's terminator edge `arm -> arm_clone` (on the carrier).
    for b in out.iter_mut() {
        if redirect.contains(&b.name) {
            if let Some(t) = &mut b.typed {
                t.redirect_successor(arm, &arm_clone);
            }
        }
    }
    // 3. Boundary blocks stay put but gain mirrored phi incomings for each cloned region predecessor:
    //    `[ v, %P ]` (P in region) gains `[ rename(v), %P_clone ]` — P's clone branches here too. Built
    //    directly on the carrier from its own phi incomings.
    for b in out.iter_mut() {
        if boundary.contains(&b.name) {
            if let Some(t) = &mut b.typed {
                t.mirror_region_incomings(&region, &rename);
            }
        }
    }
    // 4. Emit the cloned region blocks. Each keeps only in-region phi incomings (for `arm`, the
    //    redirected preds); all tokens are renamed. Boundary branch targets are NOT in `rename`, so the
    //    clone's terminators still target the original (shared) boundary blocks. Bail if a phi would lose
    //    ALL its incomings (the carrier dual of the text `rebuild_phi` returning `None`).
    for src in &ordered_region {
        let n = &src.name;
        let clone_name = rename.get(n)?.clone();
        let keep = |pred: &str| {
            if n == arm {
                redirect.contains(pred)
            } else {
                region.contains(pred)
            }
        };
        let Some(src_carrier) = &src.typed else {
            return None;
        };
        for inst in &src_carrier.insts {
            if let Some((_, incoming)) = &inst.phi_incoming {
                if !incoming.is_empty() && !incoming.iter().any(|(_, p)| keep(p)) {
                    return None;
                }
            }
        }
        let role = role_for_name(&clone_name);
        let mut c = src_carrier.clone();
        c.rebuild_phi_incomings(keep);
        c.rename(&rename);
        out.push(BodyBlock {
            name: clone_name,
            role,
            typed: Some(c),
        });
    }
    Some(out)
}

/// Append, to a boundary block's phi, a mirrored incoming for every existing incoming whose
/// predecessor is in the cloned `region`: `[ v, %P ]` gains `[ rename(v), rename(%P) ]` (the cloned
/// predecessor `P'` also branches to this boundary block, carrying the renamed region value or the
/// unchanged external value). Returns `None` (leave untouched) when no incoming comes from the region.
#[cfg(test)]
pub(in crate::native) fn mirror_region_incomings(
    line: &str,
    region: &HashSet<String>,
    rename: &HashMap<String, String>,
) -> Option<String> {
    let (head, body) = line.split_once("phi ")?;
    let ty_end = body.find('[')?;
    let ty = body[..ty_end].trim_end();
    let rest = &body[ty_end..];
    let mut incomings: Vec<String> = Vec::new();
    let mut added = false;
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, c) in rest.char_indices() {
        match c {
            '[' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            ']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let inc = rest[start..=i].trim().to_string();
                    let from_region = phi_incoming_pred(&inc)
                        .map(|p| region.contains(&p))
                        .unwrap_or(false);
                    incomings.push(inc.clone());
                    if from_region {
                        incomings.push(rename_tokens(&inc, rename));
                        added = true;
                    }
                }
            }
            _ => {}
        }
    }
    if !added {
        return None;
    }
    Some(format!("{head}phi {ty} {}", incomings.join(", ")))
}

/// Fixpoint driver for [`privatize_dominated_region`]: repeatedly find a cross-arm and merge-preserving-
/// clone its dominated region, up to [`MAX_ROUNDS`] times. Returns the (possibly grown) block list;
/// callers adopt it only when [`crate::native::cfg::structured_emit::structured_plan`] then admits (floor-safe — a
/// currently-admitting or trivial-privatized function never reaches this driver).
pub(in crate::native) fn privatize_region_cross_arm(blocks: &[BodyBlock]) -> Vec<BodyBlock> {
    let mut cur: Vec<BodyBlock> = blocks.to_vec();
    let mut counter = 0usize;
    let max_blocks = blocks.len().saturating_add(MAX_REGION_CLONE_GROWTH);
    for _ in 0..MAX_ROUNDS {
        if cur.len() >= max_blocks {
            break;
        }
        let Some((header, arm)) = find_cross_arm(&cur) else {
            break;
        };
        let Some(next) = privatize_dominated_region(&cur, &header, &arm, &mut counter) else {
            break;
        };
        if next.len() > max_blocks {
            break;
        }
        cur = next;
    }
    cur
}

pub(in crate::native) fn privatize_region_cross_arm_with_witness(
    blocks: &[BodyBlock],
) -> (Vec<BodyBlock>, RegionCrossArmFixpointWitness) {
    let mut cur: Vec<BodyBlock> = blocks.to_vec();
    let mut counter = 0usize;
    let max_blocks = blocks.len().saturating_add(MAX_REGION_CLONE_GROWTH);
    let mut rounds = 0usize;
    let mut stop_reason = "round_cap";
    let mut next_blocks = None;
    let mut stop_candidate = None;
    for _ in 0..MAX_ROUNDS {
        if cur.len() >= max_blocks {
            stop_reason = "growth_cap_reached";
            break;
        }
        let Some((header, arm)) = find_cross_arm(&cur) else {
            stop_reason = "no_cross_arm";
            break;
        };
        let witness = dominated_region_clone_witness(&cur, &header, &arm);
        if witness.reason != "cloneable" {
            stop_reason = "clone_declined";
            stop_candidate = Some(witness);
            break;
        }
        let Some(next) = privatize_dominated_region(&cur, &header, &arm, &mut counter) else {
            stop_reason = "clone_declined_unmatched";
            stop_candidate = Some(witness);
            break;
        };
        if next.len() > max_blocks {
            stop_reason = "next_over_growth_cap";
            next_blocks = Some(next.len());
            stop_candidate = Some(witness);
            break;
        }
        cur = next;
        rounds += 1;
    }
    let witness = RegionCrossArmFixpointWitness {
        input_blocks: blocks.len(),
        output_blocks: cur.len(),
        max_blocks,
        rounds,
        stop_reason,
        next_blocks,
        stop_candidate,
    };
    (cur, witness)
}

/// Privatize a continuation that is shared between a nested conditional and an enclosing arm.
///
/// A selection whose natural merge is reachable from outside its header needs a private synthesized
/// merge. [`crate::native::cfg::structured_emit::synth_unique_selection_merge`] redirects the header-dominated
/// *direct* predecessors of that natural merge, but an in-arm path can first enter a continuation
/// that is also reached from the enclosing arm. That continuation is not header-dominated, so the
/// direct-predecessor sweep cannot redirect its eventual edge to the natural merge. The emitted
/// selection then has an in-arm edge that exits through the enclosing construct rather than through
/// its declared merge (for example, `renderTestPattern/5f94cffd`).
///
/// Clone each such continuation's dominated forward region for the nested header's predecessors.
/// The existing [`privatize_dominated_region`] machinery preserves SSA by partitioning entry phis and
/// mirroring boundary-phi incomings; its single-boundary/cycle/budget guards keep this restricted to
/// the same safe shape as the established cross-arm clone. Repeat because one shared continuation can
/// feed another: the first clone makes its private predecessor visible, then the next round privatizes
/// the downstream shared continuation too. The caller only adopts the transformed graph when the
/// ordinary structured-plan ladder admits it.
pub(in crate::native) fn privatize_deep_shared_continuations(
    blocks: &[BodyBlock],
) -> Vec<BodyBlock> {
    if blocks.len() > MAX_DEEP_SHARED_CONTINUATION_BLOCKS {
        return blocks.to_vec();
    }
    let mut cur = blocks.to_vec();
    let mut counter = DEEP_SHARED_COUNTER_START;
    for _ in 0..MAX_ROUNDS {
        if cur.len() > MAX_DEEP_SHARED_CONTINUATION_BLOCKS {
            break;
        }
        let mut next = None;
        for (header, continuation) in find_deep_shared_continuations(&cur) {
            if let Some(cloned) =
                privatize_dominated_region(&cur, &header, &continuation, &mut counter)
            {
                next = Some(cloned);
                break;
            }
        }
        let Some(cloned) = next else {
            break;
        };
        cur = cloned;
    }
    cur
}

/// Separate a phi-carrying function exit shared by nested selections and enclosing paths.
///
/// Return unification gives divergent selections one common post-dominator, but an immediate
/// predecessor of that exit can itself be shared: some of its incoming edges are dominated by the
/// nested header while enclosing edges are not. The ordinary unique-merge synth cannot redirect such
/// a predecessor because the block is not header-dominated. Clone that predecessor's dominated region
/// for the nested entries, preserving the exit phi through [`privatize_dominated_region`]'s boundary
/// mirroring. The subsequent ordinary planner can then funnel the private clone predecessors through
/// a unique selection merge.
///
/// The driver is used only after the complete planner rejects and return unification has changed the
/// graph. It is bounded to the same modest CFG population and round cap as deep-continuation
/// privatization; an uncloneable candidate leaves the graph on the established fallback path.
pub(in crate::native) fn privatize_shared_phi_exit_predecessors(
    blocks: &[BodyBlock],
) -> Vec<BodyBlock> {
    if blocks.len() > MAX_DEEP_SHARED_CONTINUATION_BLOCKS {
        return blocks.to_vec();
    }
    let mut cur = blocks.to_vec();
    let mut counter = SHARED_EXIT_COUNTER_START;
    for _ in 0..MAX_ROUNDS {
        if cur.len() > MAX_DEEP_SHARED_CONTINUATION_BLOCKS {
            break;
        }
        let forest = analyze(&cur);
        let selection = selection_merges(&cur, &forest);
        let mut claims: HashMap<&str, usize> = HashMap::new();
        for merge in selection.values() {
            *claims.entry(merge.as_str()).or_default() += 1;
        }
        let preds = predecessors(&cur);
        let mut headers = selection.keys().collect::<Vec<_>>();
        headers.sort_by_key(|header| {
            let mut depth = 0usize;
            let mut node = header.as_str();
            while let Some(parent) = forest.idom(node) {
                depth += 1;
                node = parent;
            }
            std::cmp::Reverse(depth)
        });

        let mut next = None;
        'headers: for header in headers {
            let natural = &selection[header];
            if claims.get(natural.as_str()).copied().unwrap_or(0) < 2
                || forest.dominates(header, natural)
                || !crate::native::cfg::structured_emit::block_has_phi(&cur, natural)
            {
                continue;
            }
            for predecessor in preds.get(natural).into_iter().flatten() {
                if forest.dominates(header, predecessor) {
                    continue;
                }
                let mut incoming = preds.get(predecessor).into_iter().flatten();
                let has_nested = incoming.clone().any(|pred| forest.dominates(header, pred));
                let has_enclosing = incoming.any(|pred| !forest.dominates(header, pred));
                if !has_nested || !has_enclosing {
                    continue;
                }
                if let Some(cloned) =
                    privatize_dominated_region(&cur, header, predecessor, &mut counter)
                {
                    next = Some(cloned);
                    break 'headers;
                }
            }
        }
        let Some(cloned) = next else {
            break;
        };
        cur = cloned;
    }
    cur
}

/// Privatize a tail shared by two or more cases of a loop-free switch.
///
/// SPIR-V case constructs cannot enter a tail belonging to another case construct. A source switch can
/// express that compactly by letting cases fall into shared suffixes (`case 3 -> tail1`, `case 2 ->
/// tail1 -> tail2`, `case 1 -> tail2`). It can also branch through a conditional into the switch's
/// default case. When the switch's merge is externally reachable, the ordinary selection-merge synth
/// cannot make shared suffixes private; when a suffix is itself a direct case root, SPIR-V cannot let
/// another case enter it even if the merge is switch-dominated. This driver treats each case target as a
/// local root, splitting only a continuation it does not dominate that still reaches the switch's natural
/// merge. Repeating innermost-first gives each case a private copy of the shared tail while one canonical
/// case retains the original.
///
/// It uses the same [`privatize_dominated_region`] SSA and boundary guards as conditional deep-tail
/// privatization. Loop switches, direct merge exits, and large CFGs decline rather than inventing a
/// multi-level break representation.
pub(in crate::native) fn privatize_switch_case_continuations(
    blocks: &[BodyBlock],
) -> Vec<BodyBlock> {
    if blocks.len() > MAX_DEEP_SHARED_CONTINUATION_BLOCKS {
        return blocks.to_vec();
    }
    let mut cur = blocks.to_vec();
    let mut counter = SWITCH_CASE_COUNTER_START;
    for _ in 0..MAX_ROUNDS {
        if cur.len() > MAX_DEEP_SHARED_CONTINUATION_BLOCKS {
            break;
        }
        let mut next = None;
        let candidates = find_switch_case_shared_continuations(&cur);
        if crate::env_vars::switch_tail_why() {
            eprintln!("switch-tail candidates: {candidates:?}");
        }
        for (case_root, continuation) in candidates {
            if let Some(cloned) =
                privatize_dominated_region(&cur, &case_root, &continuation, &mut counter)
            {
                next = Some(cloned);
                break;
            }
        }
        let Some(cloned) = next else {
            break;
        };
        cur = cloned;
    }
    cur
}
