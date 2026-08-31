//! The typed phi-restructuring primitives the structurizer applies to a `TirBlock`'s phi instructions
//! (`rebuild_phi` / `duplicate_phi_incoming` / `mirror_region_incomings` / the incoming extend). Each
//! mutates a block's phi instructions when the structurizer reorders/clones a region — filtering,
//! duplicating, or mirroring the canonical `phi_incoming` pairs and keeping the derived def/use edges
//! consistent — so a mutation site keeps its typed carrier populated instead of
//! invalidating it (`typed = None`). Byte-identical to the retired text path's re-lowering of the
//! rewritten phi line by construction (verified per primitive by a `== re-lower` unit test + historical private byte-baseline
//! drift NONE): use traversal yields the deduped `%`-local names of the incoming values. Parsed phis do
//! not duplicate those values in `operands`; emission consumes the
//! canonical incoming carrier directly.

use super::*;
use crate::native::ir::LlValue;

/// Keep the derived-use views consistent after editing canonical `phi_incoming` values.
fn recompute_phi_uses(inst: &mut TirInst) {
    // Keep the parse-time-inference view `phi_incoming_values` (the incoming VALUES) in step with the
    // edited `phi_incoming`, so an edited phi stays byte-identical to re-lowering its rewritten line
    // (`resolve_phi_incoming_values` returns exactly these values; `phi_incoming` Some ⟹ that parse
    // succeeds too). No parse-time inference reads a structurized block, but the `== re-lower` invariant
    // must hold in every carried field.
    *inst.phi_incoming_values_mut() = None;
    inst.uses = None;
}

impl TirBlock {
    /// Filter every phi's incomings to those whose predecessor label satisfies `keep` — the typed dual
    /// of applying the text `rebuild_phi` to each phi line. A phi with a mix of kept and dropped
    /// incomings has its canonical pairs updated; use traversal derives from that carrier. A phi whose
    /// incomings are ALL kept (identity) or ALL dropped is left untouched — the
    /// text `rebuild_phi` returns `None` on an empty keep-set, leaving the line unchanged.
    pub(in crate::native) fn rebuild_phi_incomings(&mut self, keep: impl Fn(&str) -> bool) {
        for inst in &mut self.insts {
            let Some((_, incoming)) = &inst.phi_incoming() else {
                continue;
            };
            let keep_idx: Vec<usize> = incoming
                .iter()
                .enumerate()
                .filter(|(_, (_, pred))| keep(pred))
                .map(|(i, _)| i)
                .collect();
            if keep_idx.len() == incoming.len() || keep_idx.is_empty() {
                continue;
            }
            if let Some((_, incoming)) = inst.phi_incoming_mut() {
                *incoming = keep_idx.iter().map(|&i| incoming[i].clone()).collect();
            }
            recompute_phi_uses(inst);
        }
    }

    /// Append a value phi `<result> = phi <ty> [ v0, p0 ], ...` to this block's instructions, built
    /// directly from typed data — the carrier-direct dual of synthesizing a phi LINE, re-lowering it,
    /// and pushing the block. Byte-identical to lowering the equivalent `%result = phi <ty> [ v, p ], …`
    /// line by construction: `phi_incoming` holds the `(ty, incoming)` pairs `parse_phi` would produce,
    /// `result_ty` is the phi type, and use traversal derives local names from those values. Every
    /// non-phi field is `None`/empty, matching a re-lowered phi line. Used by structurizer sites to build
    /// a fresh phi from typed incomings without rendering + re-lexing a line (`render_value` cannot print
    /// the `undef`/aggregate incomings these sites funnel).
    pub(in crate::native) fn push_value_phi(
        &mut self,
        result: &str,
        ty: &LlType,
        incomings: &[(LlValue, String)],
    ) {
        let mut inst = TirInst {
            result: Some(result.to_string()),
            result_ty: Some(ty.clone()),
            uses: None,
            operands: Vec::new(),
            opcode: TirOpcode::Phi,
            // The lighter incoming-values view is filled by `recompute_phi_uses` below.
            data: Box::new(TirInstDetails {
                fast_math: false,
                payload: TirInstData::Phi {
                    parse_error: None,
                    incoming: Some((ty.clone(), incomings.to_vec())),
                    incoming_values: None,
                },
            }),
        };
        recompute_phi_uses(&mut inst);
        self.insts.push(inst);
    }

    /// Expand each phi's incomings by predecessor: for every incoming `[ v, %P ]` whose predecessor `%P`
    /// is a key in `rewrites`, replace it IN PLACE with one incoming `[ v, %Q ]` per new predecessor `%Q`
    /// in `rewrites[%P]` (same value, cloned per new predecessor; source order preserved). The typed dual
    /// of the switch-ladder string rewrite (`rewrite_lowered_switch_target_phis`): a switch predecessor
    /// that lowered to a comparison ladder fans out into the ladder's leaf blocks, so each target phi's
    /// incoming from the old switch block becomes one incoming per leaf. A phi with no matched predecessor
    /// is left untouched (the string path re-lowers only rewritten lines). A phi whose incomings did not
    /// lower (`phi_incoming: None`, aggregate) is skipped — re-lowering its rewritten line reproduces the
    /// same `None`/`Unresolved` carrier, so skipping is byte-identical. `phi_incoming`, the parallel value
    /// `operands`, and `uses` are recomputed from the expanded list (each duplicated incoming reuses the
    /// source operand — the operand carries the value only, not the predecessor).
    pub(in crate::native) fn expand_phi_predecessors(
        &mut self,
        rewrites: &HashMap<String, Vec<String>>,
    ) {
        for inst in &mut self.insts {
            let Some((_, incoming)) = &inst.phi_incoming() else {
                continue;
            };
            if !incoming.iter().any(|(_, pred)| rewrites.contains_key(pred)) {
                continue;
            }
            // The expanded plan as (source-index, value, predecessor) in source order.
            let plan: Vec<(usize, LlValue, String)> = incoming
                .iter()
                .enumerate()
                .flat_map(|(i, (value, pred))| match rewrites.get(pred) {
                    Some(new_preds) => new_preds
                        .iter()
                        .map(|new_pred| (i, value.clone(), new_pred.clone()))
                        .collect::<Vec<_>>(),
                    None => vec![(i, value.clone(), pred.clone())],
                })
                .collect();
            if let Some((_, inc)) = inst.phi_incoming_mut() {
                *inc = plan.into_iter().map(|(_, v, p)| (v, p)).collect();
            }
            recompute_phi_uses(inst);
        }
    }

    /// Append one incoming `[ value, pred ]` to this block's phi named `result` — the carrier-direct dual
    /// of rewriting that phi's LINE to add a trailing incoming and re-lowering it. The canonical carrier
    /// remains the sole owner of the values. A no-op if the block has no phi with that result.
    pub(in crate::native) fn append_phi_incoming(
        &mut self,
        result: &str,
        value: LlValue,
        pred: &str,
    ) {
        for inst in &mut self.insts {
            if inst.opcode == "phi" && inst.result.as_deref() == Some(result) {
                let Some((_, incoming)) = inst.phi_incoming_mut() else {
                    return;
                };
                incoming.push((value, pred.to_string()));
                recompute_phi_uses(inst);
                return;
            }
        }
    }

    /// Replace the incoming list of this block's phi named `result` with `incomings` — the carrier-direct
    /// dual of rewriting the phi line and re-lowering it. Keeps the phi's type/opcode/result untouched (an incoming edit
    /// never changes them). A no-op if the block has no phi with that result, or if that phi did not lower
    /// its incomings (`phi_incoming: None`, the degenerate aggregate class that routes to retry — such a
    /// phi carries no typed incoming list to rebuild, so the caller never targets it).
    pub(in crate::native) fn set_phi_incomings(
        &mut self,
        result: &str,
        incomings: &[(LlValue, String)],
    ) {
        for inst in &mut self.insts {
            if inst.opcode == "phi" && inst.result.as_deref() == Some(result) {
                let Some((_, existing)) = inst.phi_incoming_mut() else {
                    return;
                };
                *existing = incomings.to_vec();
                recompute_phi_uses(inst);
                return;
            }
        }
    }

    /// Append, to each of this block's phis carrying an incoming from `from`, a mirrored incoming
    /// `[ rename(value), to ]` — the `from`-incoming's value with the clone rename applied, for the new
    /// predecessor `to`. The carrier-direct dual of the string `duplicate_phi_incoming`: a phi with no
    /// `from` incoming (or `phi_incoming: None`, aggregate) is left untouched, and the appended incoming's
    /// value/operand/uses are recomputed exactly as a re-lower of the extended line would
    /// (`rename::renamed_llvalue` moves the same `%`-tokens the string `rename_tokens` moves).
    pub(in crate::native) fn duplicate_phi_incoming(
        &mut self,
        from: &str,
        to: &str,
        rename: &HashMap<String, String>,
    ) {
        for inst in &mut self.insts {
            let Some((_, incoming)) = inst.phi_incoming().as_ref() else {
                continue;
            };
            let Some(new_value) = incoming
                .iter()
                .find(|(_, p)| p == from)
                .map(|(v, _)| super::rename::renamed_llvalue(v, rename))
            else {
                continue;
            };
            if let Some((_, inc)) = inst.phi_incoming_mut() {
                inc.push((new_value.clone(), to.to_string()));
            }
            recompute_phi_uses(inst);
        }
    }

    /// Mirror region-clone phi incomings onto this block's carrier — the typed dual of the string
    /// `mirror_region_incomings` privatize applies to each boundary phi LINE. For every phi with a
    /// parseable incoming list, each incoming `[ v, %P ]` whose predecessor `%P` is in `region` gains an
    /// interleaved renamed clone `[ rename(v), rename(%P) ]` immediately after it (the cloned region
    /// predecessor `P'` branches to this boundary block too, carrying the renamed region value or the
    /// unchanged external value). The canonical incoming carrier is replaced by the mirrored list. A phi
    /// whose incomings did not lower (`phi_incoming: None` — an
    /// aggregate incoming) is left untouched: it carries no typed incoming list to mirror, and it fails
    /// primary emit → retry regardless (the boundary carrier is what emits), so the string mirror on it
    /// is emit-irrelevant. Byte-identical to re-lowering the string-mirrored phi lines by construction
    /// (`==re-lower` unit test + byte-baseline drift NONE in historical private gates): the same in-region incomings are interleaved in
    /// the same order, and `rename::{renamed_llvalue,renamed_label}` move the same `%`-tokens the string
    /// `rename_tokens` moves.
    pub(in crate::native) fn mirror_region_incomings(
        &mut self,
        region: &HashSet<String>,
        rename: &HashMap<String, String>,
    ) {
        for inst in &mut self.insts {
            let Some((ty, incoming)) = inst.phi_incoming().as_ref() else {
                continue;
            };
            let ty = ty.clone();
            let mut mirrored: Vec<(LlValue, String)> = Vec::with_capacity(incoming.len());
            let mut added = false;
            for (value, pred) in incoming {
                mirrored.push((value.clone(), pred.clone()));
                if region.contains(pred) {
                    mirrored.push((
                        super::rename::renamed_llvalue(value, rename),
                        super::rename::renamed_label(pred, rename),
                    ));
                    added = true;
                }
            }
            if !added {
                continue;
            }
            *inst.phi_incoming_mut() = Some((ty, mirrored));
            recompute_phi_uses(inst);
        }
    }

    /// Replace this block's phi named by `phi_line`'s result with one re-derived from `phi_line` — the
    /// typed dual of a synthesis site that rebuilds a phi's incoming list (drop redirected + append a
    /// `[merged, M]` funnel incoming, mirror a cloned predecessor, or extend a boundary phi) and writes
    /// the new line. Recomputes the canonical incoming field with the same `phi_incoming_parse` parser a
    /// re-lower runs (retaining unresolved operands only when that canonical parse fails), so the
    /// carrier's phi is byte-identical to re-lowering the rewritten line — the phi TYPE (hence
    /// `result_ty`, `opcode`, and every non-incoming field) is unchanged by an incoming edit. A no-op if
    /// the block has no phi with that result (the caller's line did not name one of this block's phis).
    #[cfg(test)]
    pub(in crate::native) fn relower_phi(&mut self, phi_line: &str) {
        let Some(result) = result_name(phi_line) else {
            return;
        };
        for inst in &mut self.insts {
            if inst.opcode == "phi" && inst.result.as_deref() == Some(result.as_str()) {
                let incoming = phi_incoming_parse(phi_line).0;
                let fallback = incoming
                    .is_none()
                    .then(|| resolve_phi_incoming_values(phi_line, "phi"))
                    .flatten();
                let parsed = incoming.is_some();
                *inst.phi_incoming_mut() = incoming;
                *inst.phi_incoming_values_mut() = fallback;
                inst.operands = if parsed {
                    Vec::new()
                } else {
                    resolve_operands(phi_line)
                };
                inst.uses = (!parsed).then(|| instruction_uses(phi_line, Some(result.as_str())));
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::cfg::clone_crossarm::rebuild_phi;
    use crate::native::tir::lower_block_carrier;

    fn types() -> HashMap<String, LlType> {
        HashMap::new()
    }

    /// The typed `rebuild_phi_incomings` must equal re-lowering the string-`rebuild_phi`ed lines, for a
    /// spread of phi shapes (single/multi phi, kept/dropped predecessors, aggregate + constant incoming
    /// values, non-phi neighbours).
    #[test]
    fn rebuild_phi_incomings_matches_relowered_lines() {
        let cases: &[(&[&str], &[&str])] = &[
            // Drop `%p2`, keep `%p1`.
            (
                &["%r = phi i32 [ %a, %p1 ], [ %b, %p2 ]", "br label %exit"],
                &["%p1"],
            ),
            // Two phis, mixed values (local + constant incoming), drop `%p3`.
            (
                &[
                    "%r = phi i32 [ %a, %p1 ], [ 0, %p3 ]",
                    "%s = phi float [ %c, %p1 ], [ %d, %p3 ]",
                    "ret void",
                ],
                &["%p1"],
            ),
            // All kept — identity (line untouched).
            (
                &["%r = phi i32 [ %a, %p1 ], [ %b, %p2 ]", "br label %x"],
                &["%p1", "%p2"],
            ),
            // Non-phi neighbours are untouched.
            (
                &[
                    "%t = add i32 %a, %b",
                    "%r = phi i32 [ %t, %p1 ], [ %b, %p2 ]",
                    "br label %x",
                ],
                &["%p1"],
            ),
        ];
        for (lines, keep) in cases {
            let keep_set: HashSet<&str> = keep.iter().copied().collect();
            let src: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
            let mut carrier = lower_block_carrier("%blk", &src, &types()).unwrap();
            carrier.rebuild_phi_incomings(|pred| keep_set.contains(pred));

            let rewritten: Vec<String> = src
                .iter()
                .map(|l| {
                    rebuild_phi(l, |pred| keep_set.contains(pred)).unwrap_or_else(|| l.clone())
                })
                .collect();
            let expected = lower_block_carrier("%blk", &rewritten, &types()).unwrap();
            assert_eq!(
                format!("{carrier:?}"),
                format!("{expected:?}"),
                "typed rebuild_phi_incomings diverged from re-lower for {lines:?} keep {keep:?}"
            );
        }
    }

    /// The typed `relower_phi` must equal re-lowering the block with the rewritten phi line, for the
    /// synthesis-site edit shapes: drop-and-append a funnel incoming, mirror a cloned predecessor, and
    /// extend a boundary phi with new incomings — including aggregate/constant/`undef` incoming values.
    #[test]
    fn relower_phi_matches_relowered_lines() {
        // (original block lines, index of the phi line, its rewritten replacement).
        let cases: &[(&[&str], usize, &str)] = &[
            // Drop `%p2`, append a merged funnel incoming `[ %m, %M ]` (multi_exit / phi_util shape).
            (
                &["%r = phi i32 [ %a, %p1 ], [ %b, %p2 ]", "br label %x"],
                0,
                "%r = phi i32 [ %a, %p1 ], [ %m, %M ]",
            ),
            // Mirror a cloned predecessor: append `[ %a.c, %arm.c ]` (privatize / cross_arm shape).
            (
                &["%r = phi i32 [ %a, %arm ], [ %b, %o ]", "ret void"],
                0,
                "%r = phi i32 [ %a, %arm ], [ %b, %o ], [ %a.c, %arm.c ]",
            ),
            // Extend a boundary phi with an undef funnel incoming, aggregate incoming value present.
            (
                &[
                    "%r = phi <2 x i32> [ <2 x i32> <i32 %a, i32 %b>, %p1 ]",
                    "br label %x",
                ],
                0,
                "%r = phi <2 x i32> [ <2 x i32> <i32 %a, i32 %b>, %p1 ], [ undef, %M ]",
            ),
        ];
        for (lines, phi_idx, rewritten_phi) in cases {
            let src: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
            let mut carrier = lower_block_carrier("%blk", &src, &types()).unwrap();
            carrier.relower_phi(rewritten_phi);

            let mut rewritten = src.clone();
            rewritten[*phi_idx] = rewritten_phi.to_string();
            let expected = lower_block_carrier("%blk", &rewritten, &types()).unwrap();
            assert_eq!(
                format!("{carrier:?}"),
                format!("{expected:?}"),
                "typed relower_phi diverged from re-lower for {rewritten_phi:?}"
            );
        }
    }

    /// The typed `mirror_region_incomings` must equal re-lowering the string-`mirror_region_incomings`ed
    /// phi lines — the privatize boundary-mirror shape: each in-region incoming gains an interleaved
    /// renamed clone, external incomings and value renames follow the same rename map.
    #[test]
    fn mirror_region_incomings_matches_relowered_lines() {
        use crate::native::cfg::clone_crossarm::mirror_region_incomings as string_mirror;
        // (block lines, region labels, rename pairs).
        let cases: &[(&[&str], &[&str], &[(&str, &str)])] = &[
            // One in-region incoming (%arm renames, its value %a renames too), one external (%o stays).
            (
                &["%r = phi i32 [ %a, %arm ], [ %b, %o ]", "br label %x"],
                &["%arm"],
                &[("%arm", "%arm.c"), ("%a", "%a.c")],
            ),
            // Two phis; both boundary preds in region; a constant incoming (no value rename).
            (
                &[
                    "%r = phi i32 [ %a, %p1 ], [ 0, %p2 ]",
                    "%s = phi float [ %c, %p1 ], [ %d, %p2 ]",
                    "ret void",
                ],
                &["%p1", "%p2"],
                &[
                    ("%p1", "%p1.c"),
                    ("%p2", "%p2.c"),
                    ("%a", "%a.c"),
                    ("%c", "%c.c"),
                    ("%d", "%d.c"),
                ],
            ),
            // No in-region incoming — untouched (both preds external).
            (
                &["%r = phi i32 [ %a, %o1 ], [ %b, %o2 ]", "br label %x"],
                &["%arm"],
                &[("%arm", "%arm.c")],
            ),
        ];
        for (lines, region, rename_pairs) in cases {
            let region_set: HashSet<String> = region.iter().map(|s| s.to_string()).collect();
            let rename: HashMap<String, String> = rename_pairs
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect();
            let src: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
            let mut carrier = lower_block_carrier("%blk", &src, &types()).unwrap();
            carrier.mirror_region_incomings(&region_set, &rename);

            let rewritten: Vec<String> = src
                .iter()
                .map(|l| string_mirror(l, &region_set, &rename).unwrap_or_else(|| l.clone()))
                .collect();
            let expected = lower_block_carrier("%blk", &rewritten, &types()).unwrap();
            assert_eq!(
                format!("{carrier:?}"),
                format!("{expected:?}"),
                "typed mirror_region_incomings diverged from re-lower for {lines:?} region {region:?}"
            );
        }
    }

    /// `push_value_phi` must equal re-lowering a block whose first line is the equivalent phi line, for
    /// the incoming shapes it is REACHABLE for: local, integer constant, and the `undef` funnel. The
    /// caller only ever passes typed incomings it extracted from a carrier's `phi_incoming`, which is
    /// `Some` exactly for these parseable forms — a phi with an aggregate incoming lowers to
    /// `phi_incoming: None` (`parse_phi` rejects it) and routes to retry, so no typed incoming list is
    /// available to feed this helper for that class (see [`aggregate_phi_incoming_is_none`]). The test
    /// authors BOTH the phi line text and the parallel typed incoming list (production has only the typed
    /// list — the carrier's already-parsed incomings — so `render` is never needed there).
    #[test]
    fn push_value_phi_matches_relowered_line() {
        // (phi line, phi type, typed incoming (value, pred) list).
        let cases: &[(&str, LlType, &[(LlValue, &str)])] = &[
            (
                "%r = phi i32 [ %a, %p1 ], [ %m, %M ]",
                LlType::Int(32),
                &[
                    (LlValue::Local("%a".to_string()), "%p1"),
                    (LlValue::Local("%m".to_string()), "%M"),
                ],
            ),
            (
                "%r = phi i32 [ %a, %p1 ], [ 0, %p2 ]",
                LlType::Int(32),
                &[
                    (LlValue::Local("%a".to_string()), "%p1"),
                    (LlValue::Int(0), "%p2"),
                ],
            ),
            (
                "%r = phi i32 [ %a, %p1 ], [ undef, %M ]",
                LlType::Int(32),
                &[
                    (LlValue::Local("%a".to_string()), "%p1"),
                    (LlValue::Undef, "%M"),
                ],
            ),
            // The multi-exit merge SELECTOR phi: `i1` with `true`/`false` constant incomings (the
            // synth-side value the merge dispatch is built from). Locks that `push_value_phi` builds an
            // i1 `Bool` incoming byte-identically to re-lowering the printed `[ true/false, %p ]` line.
            (
                "%r = phi i1 [ true, %p1 ], [ false, %p2 ]",
                LlType::Int(1),
                &[(LlValue::Bool(true), "%p1"), (LlValue::Bool(false), "%p2")],
            ),
        ];
        for (phi_line, ty, incomings) in cases {
            let expected = lower_block_carrier(
                "%blk",
                &[phi_line.to_string(), "br label %x".to_string()],
                &types(),
            )
            .unwrap();
            let mut carrier =
                lower_block_carrier("%blk", &["br label %x".to_string()], &types()).unwrap();
            let owned: Vec<(LlValue, String)> = incomings
                .iter()
                .map(|(v, p)| (v.clone(), p.to_string()))
                .collect();
            carrier.push_value_phi("%r", ty, &owned);
            assert_eq!(
                format!("{carrier:?}"),
                format!("{expected:?}"),
                "push_value_phi diverged from re-lower for {phi_line:?}"
            );
        }
    }

    /// `append_phi_incoming` must equal re-lowering a block whose phi line carries the extra incoming, for
    /// local/constant/aggregate/`undef` appended values.
    #[test]
    fn append_phi_incoming_matches_relowered_line() {
        // (original phi line, extended phi line, appended (value, pred)).
        let cases: &[(&str, &str, LlValue, &str)] = &[
            (
                "%r = phi i32 [ %a, %p1 ]",
                "%r = phi i32 [ %a, %p1 ], [ %m, %M ]",
                LlValue::Local("%m".to_string()),
                "%M",
            ),
            (
                "%r = phi i32 [ %a, %p1 ]",
                "%r = phi i32 [ %a, %p1 ], [ 0, %M ]",
                LlValue::Int(0),
                "%M",
            ),
            (
                "%r = phi i32 [ %a, %p1 ]",
                "%r = phi i32 [ %a, %p1 ], [ undef, %M ]",
                LlValue::Undef,
                "%M",
            ),
        ];
        for (orig, extended, value, pred) in cases {
            let expected = lower_block_carrier(
                "%blk",
                &[extended.to_string(), "br label %x".to_string()],
                &types(),
            )
            .unwrap();
            let mut carrier = lower_block_carrier(
                "%blk",
                &[orig.to_string(), "br label %x".to_string()],
                &types(),
            )
            .unwrap();
            carrier.append_phi_incoming("%r", value.clone(), pred);
            assert_eq!(
                format!("{carrier:?}"),
                format!("{expected:?}"),
                "append_phi_incoming diverged from re-lower for {extended:?}"
            );
        }
    }

    /// `set_phi_incomings` must equal re-lowering a block whose phi line carries the replacement incoming
    /// list — the split-phi-overlap exit-phi rewrite (kept incomings + a single `[merged, %passthrough]`
    /// funnel), including the all-redirected case (every original incoming replaced by the funnel).
    #[test]
    fn set_phi_incomings_matches_relowered_line() {
        // (original phi line, rewritten phi line, replacement incomings).
        let cases: &[(&str, &str, &[(LlValue, &str)])] = &[
            // Keep `%a` from `%p1`, replace the `%p2` incoming with the merged funnel.
            (
                "%r = phi i32 [ %a, %p1 ], [ %b, %p2 ]",
                "%r = phi i32 [ %a, %p1 ], [ %m, %M ]",
                &[
                    (LlValue::Local("%a".to_string()), "%p1"),
                    (LlValue::Local("%m".to_string()), "%M"),
                ],
            ),
            // All original incomings redirected → single merged funnel incoming.
            (
                "%r = phi i32 [ %a, %p1 ], [ %b, %p2 ]",
                "%r = phi i32 [ %m, %M ]",
                &[(LlValue::Local("%m".to_string()), "%M")],
            ),
        ];
        for (orig, rewritten, incomings) in cases {
            let expected = lower_block_carrier(
                "%blk",
                &[rewritten.to_string(), "br label %x".to_string()],
                &types(),
            )
            .unwrap();
            let mut carrier = lower_block_carrier(
                "%blk",
                &[orig.to_string(), "br label %x".to_string()],
                &types(),
            )
            .unwrap();
            let owned: Vec<(LlValue, String)> = incomings
                .iter()
                .map(|(v, p)| (v.clone(), p.to_string()))
                .collect();
            carrier.set_phi_incomings("%r", &owned);
            assert_eq!(
                format!("{carrier:?}"),
                format!("{expected:?}"),
                "set_phi_incomings diverged from re-lower for {rewritten:?}"
            );
        }
    }

    /// The reachability boundary for [`TirBlock::push_value_phi`] / [`TirBlock::append_phi_incoming`]: a
    /// phi with an AGGREGATE incoming lowers to `phi_incoming: None` (`parse_phi` rejects the
    /// `<N x T> <...>` incoming form) with an `Unresolved` operand, so it fails primary emit and routes
    /// to retry. A structurizer site therefore never extracts a typed incoming list from such a phi —
    /// the carrier-direct helpers are only ever fed the parseable (`Some`) incomings the tests above
    /// cover. This locks that invariant so the helpers are not "fixed" to fake-parse aggregates (which
    /// would emit where the line path retried — a byte change).
    #[test]
    fn malformed_phi_incoming_is_none() {
        let carrier = lower_block_carrier(
            "%blk",
            &[
                "%r = phi <2 x i32> [ <2 x i32> <i32 %a, i32 %b> %p1 ], [ undef, %M ]".to_string(),
                "br label %x".to_string(),
            ],
            &types(),
        )
        .unwrap();
        let phi = &carrier.insts[0];
        assert_eq!(phi.opcode, "phi");
        assert!(
            phi.phi_incoming().is_none(),
            "a malformed phi incoming must not parse to a typed incoming list"
        );
    }

    /// A vector-constant phi incoming (`<2 x i32> <i32 %a, i32 %b>`) carries an inline type prefix.
    /// It parses to a typed incoming list, so the phi is editable and the CFG editors may rewrite it.
    #[test]
    fn aggregate_phi_incoming_parses() {
        let carrier = lower_block_carrier(
            "%blk",
            &[
                "%r = phi <2 x i32> [ <2 x i32> <i32 %a, i32 %b>, %p1 ], [ undef, %M ]".to_string(),
                "br label %x".to_string(),
            ],
            &types(),
        )
        .unwrap();
        let phi = &carrier.insts[0];
        assert_eq!(phi.opcode, "phi");
        let (_, incoming) = phi
            .phi_incoming()
            .as_ref()
            .expect("a typed vector-constant phi incoming parses");
        assert_eq!(incoming.len(), 2);
        assert_eq!(incoming[0].1, "%p1");
        assert_eq!(incoming[1].1, "%M");
    }

    /// `duplicate_phi_incoming` must equal re-lowering a block whose phis carry a mirrored
    /// `[ rename(v), to ]` incoming appended for each incoming from `from` — the cross-arm clone shape
    /// (value renamed, external/no-`from` incomings and constant values handled).
    #[test]
    fn duplicate_phi_incoming_matches_relowered_line() {
        // (block lines, from, to, rename pairs, expected rewritten lines).
        let cases: &[(&[&str], &str, &str, &[(&str, &str)], &[&str])] = &[
            // One in-`from` incoming: its value %a renames, the new pred is `to`.
            (
                &["%r = phi i32 [ %a, %arm ], [ %b, %o ]", "br label %x"],
                "%arm",
                "%arm.c",
                &[("%a", "%a.c")],
                &[
                    "%r = phi i32 [ %a, %arm ], [ %b, %o ], [ %a.c, %arm.c ]",
                    "br label %x",
                ],
            ),
            // No incoming from `from` — untouched.
            (
                &["%r = phi i32 [ %a, %o1 ], [ %b, %o2 ]", "ret void"],
                "%arm",
                "%arm.c",
                &[("%arm", "%arm.c")],
                &["%r = phi i32 [ %a, %o1 ], [ %b, %o2 ]", "ret void"],
            ),
            // Constant incoming from `from` (no value rename).
            (
                &["%r = phi i32 [ 0, %arm ], [ %b, %o ]", "br label %x"],
                "%arm",
                "%arm.c",
                &[("%arm", "%arm.c")],
                &[
                    "%r = phi i32 [ 0, %arm ], [ %b, %o ], [ 0, %arm.c ]",
                    "br label %x",
                ],
            ),
        ];
        for (lines, from, to, rename_pairs, expected_lines) in cases {
            let rename: HashMap<String, String> = rename_pairs
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect();
            let src: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
            let mut carrier = lower_block_carrier("%blk", &src, &types()).unwrap();
            carrier.duplicate_phi_incoming(from, to, &rename);
            let expected_src: Vec<String> = expected_lines.iter().map(|s| s.to_string()).collect();
            let expected = lower_block_carrier("%blk", &expected_src, &types()).unwrap();
            assert_eq!(
                format!("{carrier:?}"),
                format!("{expected:?}"),
                "duplicate_phi_incoming diverged from re-lower for {lines:?}"
            );
        }
    }

    /// `expand_phi_predecessors` must equal re-lowering the string-expanded phi lines — the switch-ladder
    /// shape: a matched predecessor fans out into N leaves (same value duplicated per leaf), unmatched
    /// predecessors untouched. Covers the single-leaf (1:1 rename) and multi-phi/constant cases.
    #[test]
    fn expand_phi_predecessors_matches_relowered_lines() {
        // (block lines, rewrites old->new preds, expected rewritten lines).
        let cases: &[(&[&str], &[(&str, &[&str])], &[&str])] = &[
            // %p1 fans out to two leaves (value %a duplicated); %p2 untouched.
            (
                &["%r = phi i32 [ %a, %p1 ], [ %b, %p2 ]", "br label %x"],
                &[("%p1", &["%l0", "%l1"][..])],
                &[
                    "%r = phi i32 [ %a, %l0 ], [ %a, %l1 ], [ %b, %p2 ]",
                    "br label %x",
                ],
            ),
            // No matched predecessor — untouched.
            (
                &["%r = phi i32 [ %a, %p1 ]", "ret void"],
                &[("%zzz", &["%q"][..])],
                &["%r = phi i32 [ %a, %p1 ]", "ret void"],
            ),
            // Two phis; single-leaf remap of %p2 (1:1), a constant incoming present.
            (
                &[
                    "%r = phi i32 [ %a, %p1 ], [ 0, %p2 ]",
                    "%s = phi float [ %c, %p1 ], [ %d, %p2 ]",
                    "ret void",
                ],
                &[("%p2", &["%m"][..])],
                &[
                    "%r = phi i32 [ %a, %p1 ], [ 0, %m ]",
                    "%s = phi float [ %c, %p1 ], [ %d, %m ]",
                    "ret void",
                ],
            ),
        ];
        for (lines, rewrites_pairs, expected_lines) in cases {
            let rewrites: HashMap<String, Vec<String>> = rewrites_pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.iter().map(|s| s.to_string()).collect()))
                .collect();
            let src: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
            let mut carrier = lower_block_carrier("%blk", &src, &types()).unwrap();
            carrier.expand_phi_predecessors(&rewrites);
            let expected_src: Vec<String> = expected_lines.iter().map(|s| s.to_string()).collect();
            let expected = lower_block_carrier("%blk", &expected_src, &types()).unwrap();
            assert_eq!(
                format!("{carrier:?}"),
                format!("{expected:?}"),
                "expand_phi_predecessors diverged from re-lower for {lines:?}"
            );
        }
    }
}
