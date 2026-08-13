//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;

/// Fixpoint driver for the cross-arm-EDGE clone (the deeper analogue of [`privatize_region_cross_arm`]).
/// For each definitive cross-arm edge `B -> S` found by [`find_cross_arm_edge`], clone `S`'s dominated
/// forward region for the near-arm entries (`privatize_dominated_region(header = near, arm = S)`), so the
/// near-arm block reaches a PRIVATE copy that reconverges at the shared merge while the sibling arm keeps
/// the original. Returns the (possibly grown) block list; callers adopt it only when `structured_plan`
/// then admits (floor-safe — a currently-admitting function never reaches this, and the plan self-checks
/// still gate the cloned result).
pub(in crate::native) fn privatize_cross_arm_edge(blocks: &[BodyBlock]) -> Vec<BodyBlock> {
    let mut cur: Vec<BodyBlock> = blocks.to_vec();
    let mut counter = 1_000_000usize;
    // Round + growth caps: bound the cost. Each round re-runs `analyze` (dominator recompute, super-
    // linear in blocks) and clones up to `MAX_REGION_BLOCKS`, and this driver is re-invoked by
    // `structured_plan` per retry tier, so an unbounded loop on a big function grinds the cascade. Real
    // cross-arm-edge functions converge in 1-3 rounds (the largest landed win needs <=3); a function that
    // has not converged in `EDGE_ROUNDS` was not going to structure cleanly and falls to repair, exactly
    // as before this attempt existed. `structured_emit::CROSS_ARM_EDGE_MAX_BLOCKS` already gates the whole
    // attempt to modest functions, so `cap` is a secondary belt.
    const EDGE_ROUNDS: usize = 8;
    let cap = blocks.len() + MAX_REGION_BLOCKS * 4;
    for _ in 0..EDGE_ROUNDS {
        if cur.len() > cap {
            break;
        }
        let Some((header, arm)) = find_cross_arm_edge(&cur) else {
            break;
        };
        let Some(next) = privatize_dominated_region(&cur, &header, &arm, &mut counter) else {
            break;
        };
        cur = next;
    }
    cur
}

/// Privatize the TRIVIAL-pass-through cross-arm sub-case as a default-path pre-pass for
/// [`crate::native::cfg::structured_emit::structured_plan`] (distinct from [`clone_cross_arm_shared`], the gated
/// failure-retry that does full-closure tail duplication).
///
/// The dominant cross-arm shape (`selection:cross-arm-shared`, the frontier's largest reject class) is
/// a header `H` whose arm targets a block `A` shared with an ENCLOSING construct's arm, so `H` does not
/// dominate `A`. Full-closure tail duplication is the WRONG tool here — `A`'s forward closure reaches
/// the shared reconvergence, which duplication then DESTROYS (`--cfg-clone` yields zero wins). But when
/// `A` is a **trivial pass-through** — a single unconditional `br label %S`, no phi, no SSA
/// definition — the merge-PRESERVING fix is to clone ONLY `A`: give `H`'s (header-dominated) entries
/// their own verbatim copy `A'` (also `br label %S`) so `H` dominates its arm, while the external
/// entries keep flowing through the untouched original `A` into the shared reconvergence `S`. `S`
/// stays the merge; its phis gain an `A'` incoming mirroring the `A` one — valid because a value live
/// at the end of `A` dominates `A`, hence dominates `A'` (whose predecessors are a subset of `A`'s).
/// `A` defines nothing, so the clone needs no rename and introduces no duplicate SSA.
///
/// What remains after privatization is at most `selection:merge-not-dominated` on `H` (its declared
/// merge `S` is still reached from outside the construct), which the existing dominance-aware synth
/// (`unique_selection_merges`) resolves by inserting a header-dominated pass-through merge.
/// [`crate::native::cfg::structured_emit::structured_plan`] runs its self-checks AFTER this pre-pass, so an
/// unresolved residue still rejects honestly — the pre-pass cannot over-admit the structurizer gate
/// (though admission is not sufficient for valid SPIR-V; the frontier byte gate is the real check).
///
/// Fixpoint over up to [`MAX_ROUNDS`] privatizations (a switch/ladder can share several trivial arms);
/// returns the (possibly unchanged) block list. When no trivial cross-arm exists it returns a clone of
/// the input verbatim (same length), which the caller uses to skip the redundant retry.
pub(in crate::native) fn privatize_trivial_cross_arm(blocks: &[BodyBlock]) -> Vec<BodyBlock> {
    let mut cur: Vec<BodyBlock> = blocks.to_vec();
    let mut counter = 0usize;
    for _ in 0..MAX_ROUNDS {
        let Some((header, arm)) = find_trivial_cross_arm(&cur) else {
            break;
        };
        let Some(next) = privatize_trivial(&cur, &header, &arm, &mut counter) else {
            break;
        };
        cur = next;
    }
    cur
}

/// The trivial-pass-through variant of [`find_cross_arm`]: the first cross-arm `(H, A)` where `A` is a
/// single unconditional `br label %S` with no phi and no SSA definition — safe to clone verbatim (no
/// rename, the clone defines nothing). Reuses the same merge / enclosing-break / loop-header
/// exclusions as [`find_cross_arm`].
pub(in crate::native) fn find_trivial_cross_arm(blocks: &[BodyBlock]) -> Option<(String, String)> {
    let forest = analyze(blocks);
    let pidom = post_idom(blocks);
    let loop_headers: HashSet<&str> = forest.loops.iter().map(|l| l.header.as_str()).collect();
    let is_enclosing_break = |b: &str, a: &str| -> bool {
        forest.loops.iter().any(|l| {
            l.body.iter().any(|n| n == b) && (l.header == a || l.exits.iter().any(|e| e == a))
        })
    };
    let by_name: HashMap<&str, &BodyBlock> = blocks.iter().map(|b| (b.name.as_str(), b)).collect();
    let names: HashSet<&str> = blocks.iter().map(|b| b.name.as_str()).collect();
    for b in blocks {
        if loop_headers.contains(b.name.as_str()) {
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
        let merge = pidom.get(&b.name).map(String::as_str);
        // Deterministic arm order (distinct is a HashSet) so the fixpoint is reproducible.
        let mut arms: Vec<&str> = distinct.into_iter().collect();
        arms.sort_unstable();
        for a in arms {
            if Some(a) == merge || is_enclosing_break(&b.name, a) {
                continue;
            }
            if loop_headers.contains(a) {
                continue;
            }
            if forest.dominates(&b.name, a) {
                continue;
            }
            let Some(ablk) = by_name.get(a) else { continue };
            if !is_trivial_passthrough(ablk) || !arm_defs_safe_to_clone(a, ablk, blocks) {
                continue;
            }
            return Some((b.name.clone(), a.to_string()));
        }
    }
    None
}

/// A block that forwards to a SINGLE successor via `br label %S`. It MAY define values (the clone
/// renames each into a fresh namespace) and MAY carry phi(s) — a phi means the block is a merge of its
/// own predecessors, and [`privatize_trivial`] partitions its incomings between the original (external
/// preds) and the clone (redirected preds). Whether such an arm is safe to clone (its defs may only be
/// USED past the block as phi incomings, never in a body computation) is decided by the caller's
/// [`arm_defs_safe_to_clone`] guard.
pub(in crate::native) fn is_trivial_passthrough(b: &BodyBlock) -> bool {
    if block_successors(b).len() != 1 {
        return false;
    }
    // A trivial pass-through ends in an unconditional `br label %x` (`TirTerminator::Br`), read from the
    // carrier (the sole substrate) — a `br i1` lowers to `BrCond`, a metadata-tailed
    // `br label %x, !llvm.loop` still lowers to `Br`.
    b.typed
        .as_ref()
        .is_some_and(|t| matches!(t.terminator, crate::native::tir::TirTerminator::Br(_)))
}

/// The floor-safe guard for cloning a def-carrying arm: the arm MAY define values used past the arm,
/// PROVIDED every such outside use is a `phi` incoming (never a body computation). A phi incoming
/// `[ d, %A ]` only requires `d` to dominate the *predecessor* `%A`, not the merge block — so
/// splitting the arm into `A` + `A_clone` and mirroring `[ d_clone, %A_clone ]` (via
/// [`duplicate_phi_incoming`]) keeps SSA sound: each incoming's value is defined in its own arm copy.
/// A BODY use (like `%x = extractelement %d`) instead requires `d` to dominate that block; the split
/// removes `A`'s domination of the successor, leaving the use undefined on the clone edge (broken SSA —
/// the `000ca89f` structured-exit breaker). So body-uses stay excluded, phi-incoming-uses are admitted.
/// See [[metal2vulkan-native-emitter]] "S15 remaining reject classes" REAL MECHANISM.
pub(in crate::native) fn arm_defs_safe_to_clone(
    arm: &str,
    arm_block: &BodyBlock,
    blocks: &[BodyBlock],
) -> bool {
    // Read the arm's SSA defs and each other block's "mentions" from the typed carrier when populated
    // (the production state), else the line scan (pre-populate window). Measured byte-neutral: a PROBE
    // over frontier+banked found 0 divergences across 1,872 evaluations (all carriered). A def is
    // "mentioned" by a non-phi instruction's `uses` (a complete `%`-token scan) or by the terminator's
    // value/label operands ([`terminator_mentions`], the dual of `line_mentions_any` over a terminator).
    let defs: HashSet<String> = arm_block
        .typed
        .as_ref()
        .map(|t| t.insts.iter().filter_map(|i| i.result.clone()).collect())
        .unwrap_or_default();
    if defs.is_empty() {
        return true;
    }
    for b in blocks {
        if b.name == arm {
            continue;
        }
        let mentions = b.typed.as_ref().is_some_and(|t| {
            t.insts
                .iter()
                .any(|i| !i.is_phi() && i.uses.iter().any(|u| defs.contains(u)))
                || terminator_mentions(&t.terminator)
                    .iter()
                    .any(|u| defs.contains(u))
        });
        if mentions {
            return false;
        }
    }
    true
}

/// The `%`-token operands a terminator "mentions" — the dual of `line_mentions_any` over a terminator
/// line: its successor LABELS plus its `%`-value operand (branch cond / switch selector / ret value).
fn terminator_mentions(term: &crate::native::tir::TirTerminator) -> Vec<String> {
    use crate::native::tir::TirTerminator;
    let mut m: Vec<String> = term.successors().iter().map(|s| s.to_string()).collect();
    match term {
        TirTerminator::BrCond { cond, .. } => m.push(cond.clone()),
        TirTerminator::Switch { selector, .. } => m.push(selector.clone()),
        TirTerminator::Ret(Some(v)) if v.starts_with('%') => m.push(v.clone()),
        _ => {}
    }
    m
}

/// Clone the single-successor forwarding `arm` for `header`'s dominated entries: redirect them to a
/// fresh copy, patch the successor's phis to add the clone's incoming, and keep the original for the
/// external entries. When `arm` itself carries a phi, its incomings are partitioned — the original
/// keeps the external-pred incomings, the clone the redirected-pred ones. Returns `None` if there is no
/// dominated predecessor to redirect or no external one to keep (defensive — [`find_trivial_cross_arm`]
/// guarantees both).
pub(in crate::native) fn privatize_trivial(
    blocks: &[BodyBlock],
    header: &str,
    arm: &str,
    counter: &mut usize,
) -> Option<Vec<BodyBlock>> {
    let forest = analyze(blocks);
    let preds = predecessors(blocks);
    // Header-dominated predecessors of `arm` (inside the construct) get redirected to the clone.
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
    // `arm` must keep ≥1 external predecessor (else it was already private to `header`, not cross-arm).
    let keeps_original = preds
        .get(arm)
        .into_iter()
        .flatten()
        .any(|p| !redirect.contains(p));
    if !keeps_original {
        return None;
    }

    let by_name: HashMap<&str, &BodyBlock> = blocks.iter().map(|b| (b.name.as_str(), b)).collect();
    let arm_block = by_name.get(arm)?;
    // The single successor `S` — the shared reconvergence, kept as the merge.
    let succ = block_successors(arm_block);
    let s = succ.first()?.clone();

    // Rename map: the arm label + every value defined in the arm get a fresh clone name, so the clone
    // re-defines nothing that already exists. External values are absent, so `rename_tokens` leaves them
    // untouched. Per `arm_defs_safe_to_clone`, any arm def used past the arm is used ONLY as a phi
    // incoming — so the renamed clone value reaches `S`'s phi via `duplicate_phi_incoming` (step 2),
    // each incoming defined in its own arm copy; a def used in a body computation would be excluded.
    let id = *counter;
    *counter += 1;
    let mut rename: HashMap<String, String> = HashMap::new();
    rename.insert(arm.to_string(), fresh(arm, id));
    // Defs carrier-first (`inst.result` via `block_defs`), line fallback for the pre-`populate` window —
    // keeps the clone rename map independent of the `.lines` field (kb "STEP-1/STEP-2 decomposition").
    for def in block_defs(arm_block) {
        rename.insert(def.clone(), fresh(&def, id));
    }
    let arm_clone = rename.get(arm).cloned()?;

    let mut out: Vec<BodyBlock> = blocks.to_vec();

    // 1. Redirect each dominated predecessor's terminator edge `arm -> arm_clone` (on the carrier).
    for b in out.iter_mut() {
        if redirect.contains(&b.name) {
            if let Some(t) = &mut b.typed {
                let t = std::sync::Arc::make_mut(t);
                t.redirect_successor(arm, &arm_clone);
            }
        }
    }

    // 1b. If `arm` carries phi(s), it is a merge of its predecessors; the split partitions those
    //     incomings. The ORIGINAL keeps only the incomings from EXTERNAL (non-redirected) preds — the
    //     ones that still branch to it (`keeps_original` guarantees ≥1). The redirected preds' incomings
    //     move to the clone (step 3). Without this the original phi would list preds that no longer
    //     branch to it (invalid).
    for b in out.iter_mut() {
        if b.name == arm {
            if let Some(t) = &mut b.typed {
                let t = std::sync::Arc::make_mut(t);
                t.rebuild_phi_incomings(|p| !redirect.contains(p));
            }
        }
    }

    // 2. `S`'s phis gain an `arm_clone` incoming mirroring the `arm` one, with `rename` applied to the
    //    value. If the arm defines the value flowing into `S`'s phi, the mirrored incoming carries the
    //    RENAMED clone value (defined in `arm_clone`); otherwise the value is external and `rename` is a
    //    no-op. Either way each incoming's value is defined in its own predecessor — SSA-sound.
    for b in out.iter_mut() {
        if b.name == s {
            if let Some(t) = &mut b.typed {
                let t = std::sync::Arc::make_mut(t);
                t.duplicate_phi_incoming(arm, &arm_clone, &rename);
            }
        }
    }

    // 3. Emit the clone: `arm`'s carrier with all tokens renamed into the clone namespace (its defs
    //    become fresh names, so no SSA value is defined twice). A phi is rebuilt first to keep ONLY the
    //    redirected preds' incomings — the preds that now branch to the clone — mirroring the original's
    //    partition in step 1b (`rebuild_phi_incomings` leaves an all-dropped phi unchanged, matching the
    //    text `rebuild_phi`'s `None` fallback). `arm_block` is the pre-redirect copy, so its carrier still
    //    lists every pred. No module named-types needed — the source resolved them.
    let role = role_for_name(&arm_clone);
    let typed = arm_block.typed.as_ref().map(|src| {
        let mut c = (**src).clone();
        c.rebuild_phi_incomings(|p| redirect.contains(p));
        c.rename(&rename);
        c.into()
    });
    out.push(BodyBlock {
        name: arm_clone,
        role,
        typed,
    });
    Some(out)
}
