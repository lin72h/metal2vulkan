//! R2 cross-arm node-splitting (tail duplication) for the structured-by-construction path.
//!
//! The structurizer ([`super::structured_emit::structured_plan`]) REJECTS a selection/switch whose
//! arm jumps to a block the header does not dominate — a block shared with an enclosing construct's
//! arm (`selection:cross-arm-shared` / `cond-shared-arm` / `cond-other`). SPIR-V structured control
//! flow requires each arm target to be private to that arm, so the shared block must be CLONED so the
//! cross-arm entry gets its own copy.
//!
//! This module implements the simplest *correct* clone: **single-entry forward-closed tail
//! duplication**. For a violating edge (header `H`, arm target `A`) it clones the entire set of blocks
//! reachable from `A` (`R`) and redirects `H`'s in-construct predecessor edges of `A` to the clone.
//! Because `R` is the FULL forward closure of `A`, every use of a value defined inside `R` is itself
//! inside `R` (uses are reachable from their def) — so the clone needs **no exit-phi synthesis**: each
//! copy is self-contained. The transform fires only when `R` is **single-entry** (every block in
//! `R\{A}` has all its predecessors inside `R`); otherwise a deeper block is shared too and a single
//! clone would just move the sharing (the cascade the prior session proved necessary — left to a
//! future increment).
//!
//! It is NOT wired into the default emission path. It is reached only via the failure-triggered
//! `inline_sroa_raw_cfg_restructure` retry tier (adopt-if-validates, mirroring the R4 raw retry), so a
//! banked/passing case — which `structured_plan` admits on the default path — never reaches it and the
//! floor is byte-identical by construction.

use super::blocks::{block_successors, conditional_branch_targets};
use super::loopforest::{analyze, post_idom, selection_merges};
use super::structured_emit::role_for_name;
use super::BodyBlock;
use std::collections::{HashMap, HashSet};

mod detect;
pub(in crate::native) use detect::*;
mod privatize;
pub(in crate::native) use privatize::*;
mod finders;
pub(in crate::native) use finders::*;
mod cross_arm;
pub(in crate::native) use cross_arm::*;
mod normalize;
pub(in crate::native) use normalize::*;

/// Legacy cap on duplication for a region with multiple reconvergence boundaries. Those shapes can
/// expose more than one enclosing construct, so retain the established conservative budget.
const MAX_REGION_BLOCKS: usize = 96;
/// Maximum number of reconvergence boundaries the merge-preserving region clone is willing to
/// mirror in one privatization.
const MAX_REGION_BOUNDARIES: usize = 4;
/// A single-boundary dominated region has one explicit reconvergence point: the clone mirrors only
/// that boundary's phi incomings and leaves the boundary itself intact. It is therefore the clean
/// large-tail form. The broad fixpoint driver still has an independent total-growth cap; this per-clone
/// cap also lets the construct-tree direct-cross-arm path perform one large, structurally clean clone.
const MAX_SINGLE_BOUNDARY_REGION_BLOCKS: usize = 1152;
/// Total block growth allowed to the merge-preserving cross-arm fixpoint for one function. This
/// permits the established eight legacy-sized regions in the broad fixpoint driver, but stops a chain
/// of independently valid clones from turning a rejected large CFG into a pathological planner input.
/// Single-shot callers can still use the larger single-boundary per-clone cap above.
const MAX_REGION_CLONE_GROWTH: usize = MAX_SINGLE_BOUNDARY_REGION_BLOCKS * 2;
/// Cap on the number of privatization rounds — a runaway backstop only. The driver breaks the round
/// loop as soon as a round finds no cross-arm (`find_cross_arm` → `None`) or cannot clone the one it
/// found (`privatize_dominated_region` → `None`), so a function that resolves in `k` rounds stops at
/// `k` and is unaffected by a higher cap; the cap can bite ONLY a function still finding cloneable
/// violations after this many rounds. A function that inlines the same short-circuit-boolean shape
/// many times (`MatrixNeuronGradient`/06 inlines `applyNeuronGradient` into 8 replicated diamonds,
/// each needing several rounds to privatize its shared merge) needs more rounds than one diamond — 8
/// starved such functions into a reject/repair fall-back. 64 clears replicated-diamond shapes with
/// margin while still bounding the pathological case. The region-growth budget additionally bounds
/// the accumulated cloned CFG. Raising it is floor-safe: the clone only ever runs on a base-rejecting
/// function, and adoption still requires `structured_plan` to admit the cloned result.
const MAX_ROUNDS: usize = 64;
/// The deep-continuation detector derives a reverse reachability set for each candidate selection.
/// Keep that pre-admission probe on the same modest-CFG population as the existing cross-arm-edge
/// machinery: the malformed shared-tail wins are small, while running another quadratic graph walk on
/// multi-thousand-block dispatch kernels adds cost without a demonstrated structurization result.
const MAX_DEEP_SHARED_CONTINUATION_BLOCKS: usize = 300;
/// Keep this clone namespace disjoint from the ordinary cross-arm retries, whose counters begin at
/// zero after the deep pre-pass has already added blocks to the graph.
const DEEP_SHARED_COUNTER_START: usize = 4_000_000;
/// Namespace for copies made while separating overlapping switch case tails.
const SWITCH_CASE_COUNTER_START: usize = 5_000_000;
/// Namespace for terminal predecessors cloned while separating a shared phi-carrying function exit.
const SHARED_EXIT_COUNTER_START: usize = 6_000_000;

#[cfg(test)]
mod tests {
    use super::*;

    fn blk(name: &str, lines: &[&str]) -> BodyBlock {
        // Populate the typed carrier the way the production split+populate does (the sole substrate),
        // so the cross-arm passes that read the carrier (return normalization) exercise their real path.
        let name = format!("%{name}");
        let lines: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
        let typed = crate::native::tir::lower_block_carrier(&name, &lines, &HashMap::new());
        BodyBlock {
            name,
            role: crate::native::cfg::BlockRole::Normal,
            typed: typed.map(Into::into),
        }
    }

    #[test]
    fn rename_is_boundary_aware() {
        let mut map = HashMap::new();
        map.insert("%1".to_string(), "%xa0_1".to_string());
        // %1 renamed, %10 untouched, %11 untouched.
        assert_eq!(
            rename_tokens("  %r = add i32 %1, %10", &map),
            "  %r = add i32 %xa0_1, %10"
        );
    }

    #[test]
    fn rebuild_phi_keeps_selected_preds() {
        let line = "  %d = phi i32 [ %a, %p1 ], [ %b, %p2 ]";
        let kept = rebuild_phi(line, |p| p == "%p1").unwrap();
        assert_eq!(kept, "  %d = phi i32 [ %a, %p1 ]");
        let dropped = rebuild_phi(line, |p| p == "%nope");
        assert!(dropped.is_none());
    }

    #[test]
    fn line_def_extracts_result() {
        assert_eq!(line_def("  %5 = add i32 %1, %2"), Some("%5".to_string()));
        assert_eq!(line_def("  store i32 %1, ptr %2"), None);
        assert_eq!(line_def("  br label %3"), None);
    }

    /// The real cross-arm shape: a short-circuit `a || b` ladder. `%entry` (outer header) and
    /// `%checkb` (inner header) BOTH have an arm into the shared block `%taken`. `%taken` is reached
    /// from `%entry`'s arm without passing through `%checkb`, so `%checkb` does not dominate its arm
    /// `%taken` — the `selection:cross-arm-shared` reject. Here `%taken` returns privately (no shared
    /// merge below it), so its single-entry forward closure is `{%taken}` and the clone is clean.
    #[test]
    fn detects_and_clones_inner_header_cross_arm() {
        let blocks = vec![
            blk("entry", &["br i1 %ca, label %taken, label %checkb"]),
            blk("checkb", &["br i1 %cb, label %taken, label %elseb"]),
            blk("taken", &["ret void"]),
            blk("elseb", &["ret void"]),
        ];
        // The inner header %checkb shares its arm %taken with the outer header's arm.
        assert_eq!(
            find_cross_arm(&blocks),
            Some(("%checkb".to_string(), "%taken".to_string()))
        );
        // The complete planner now admits the fully-terminal source directly, so exercise the
        // cross-arm primitive itself rather than its reject-only retry driver.
        let mut counter = 0;
        let out = privatize_region(&blocks, "%checkb", "%taken", &mut counter)
            .expect("should apply a clone");
        // A clone of %taken exists; %checkb now branches to it; original %taken keeps the %entry edge.
        let clone = out
            .iter()
            .find(|b| b.name.starts_with("%xa"))
            .expect("clone block present");
        assert_eq!(clone.lines(), vec!["ret void".to_string()]);
        let checkb = out.iter().find(|b| b.name == "%checkb").unwrap();
        assert!(checkb.lines().last().unwrap().contains(&clone.name));
        assert!(
            !checkb.lines().last().unwrap().contains("%taken,")
                || checkb.lines().last().unwrap().contains(&clone.name)
        );
        let entry = out.iter().find(|b| b.name == "%entry").unwrap();
        assert!(entry.lines().last().unwrap().contains("%taken"));
    }

    #[test]
    fn find_cross_arm_preserves_source_terminator_arm_order() {
        let blocks = vec![
            blk("entry", &["br i1 %outer, label %shared_a, label %gate"]),
            blk("gate", &["br i1 %middle, label %shared_b, label %inner"]),
            blk(
                "inner",
                &["br i1 %nested, label %shared_b, label %shared_a"],
            ),
            blk("shared_a", &["ret void"]),
            blk("shared_b", &["ret void"]),
        ];

        assert_eq!(
            find_cross_arm(&blocks),
            Some(("%inner".to_string(), "%shared_b".to_string())),
            "when both arms are shared, choose the first source terminator arm"
        );
    }

    /// A cross-arm whose shared region reconverges at a block reached from OUTSIDE the region (`%end`
    /// is reached from `%elseb` too). The full forward closure `R = {%taken, %end}` is cloned: the
    /// shared `%end` is duplicated, the ORIGINAL `%end` keeps its `%elseb` (and original-`%taken`)
    /// edges, the clone `%end'` is reached only via `%taken'`. The clone applies (privatize succeeds),
    /// demonstrating the full-closure clone handles a non-single-entry shared merge — unlike the prior
    /// `dom(A)`-only / single-entry variants which moved the sharing or bailed.
    #[test]
    fn clones_shared_merge_via_full_closure() {
        let blocks = vec![
            blk("entry", &["br i1 %ca, label %taken, label %checkb"]),
            blk("checkb", &["br i1 %cb, label %taken, label %elseb"]),
            blk("taken", &["br label %end"]),
            blk("elseb", &["br label %end"]),
            blk("end", &["ret void"]),
        ];
        assert_eq!(
            find_cross_arm(&blocks),
            Some(("%checkb".to_string(), "%taken".to_string()))
        );
        let mut counter = 0;
        let out = privatize_region(&blocks, "%checkb", "%taken", &mut counter)
            .expect("full-closure clone applies");
        // Both %taken and %end were cloned (the shared merge is duplicated).
        assert_eq!(out.iter().filter(|b| b.name.starts_with("%xa")).count(), 2);
        // %checkb's arm now points at the cloned %taken, not the original.
        let checkb = out.iter().find(|b| b.name == "%checkb").unwrap();
        let term = checkb.lines().last().unwrap().clone();
        assert!(term.contains("%xa"));
        // Original %end is untouched (still reached from %elseb + original %taken).
        let end = out.iter().find(|b| b.name == "%end").unwrap();
        assert_eq!(end.lines(), vec!["ret void".to_string()]);
    }

    #[test]
    fn dominated_region_clone_witness_reports_cloneable_shape() {
        let blocks = vec![
            blk("entry", &["br i1 %ca, label %taken, label %checkb"]),
            blk("checkb", &["br i1 %cb, label %taken, label %elseb"]),
            blk("taken", &["br label %end"]),
            blk("elseb", &["br label %end"]),
            blk("end", &["ret void"]),
        ];

        let witness = dominated_region_clone_witness(&blocks, "%checkb", "%taken");
        assert_eq!(witness.reason, "cloneable");
        assert_eq!(witness.region_blocks, 1);
        assert_eq!(witness.region_cap, MAX_SINGLE_BOUNDARY_REGION_BLOCKS);
        assert_eq!(witness.boundary_count, 1);
        assert_eq!(witness.boundary_cap, MAX_REGION_BOUNDARIES);
        assert_eq!(witness.redirect_count, 1);
        assert_eq!(witness.external_pred_count, 1);
        assert_eq!(witness.arm_cycle_pred_count, 0);
    }

    #[test]
    fn region_fixpoint_witness_reports_stop_after_successful_clone() {
        let blocks = vec![
            blk("entry", &["br i1 %ca, label %taken, label %checkb"]),
            blk("checkb", &["br i1 %cb, label %taken, label %elseb"]),
            blk("taken", &["br label %end"]),
            blk("elseb", &["br label %end"]),
            blk("end", &["ret void"]),
        ];

        let (out, witness) = privatize_region_cross_arm_with_witness(&blocks);
        assert_eq!(out.len(), 6);
        assert_eq!(witness.input_blocks, 5);
        assert_eq!(witness.output_blocks, 6);
        assert_eq!(witness.rounds, 1);
        assert_eq!(witness.stop_reason, "no_cross_arm");
        assert!(witness.next_blocks.is_none());
        assert!(witness.stop_candidate.is_none());
    }

    /// The trivial-cross-arm pre-pass clones ONLY the shared single-`br` arm: `%inner`'s arm `%shared`
    /// is also `%entry`'s arm (cross-arm). `%shared` is `br label %merge` (trivial), so it is cloned —
    /// `%inner` branches to the clone, the original keeps `%entry`'s edge, and the whole function then
    /// structures. Contrast with full-closure duplication, which would clone `%merge` too and destroy
    /// the reconvergence.
    #[test]
    fn privatize_trivial_clones_shared_passthrough_arm() {
        let blocks = vec![
            blk("entry", &["br i1 %ca, label %shared, label %inner"]),
            blk("inner", &["br i1 %cb, label %shared, label %elseb"]),
            blk("shared", &["br label %merge"]),
            blk("elseb", &["br label %merge"]),
            blk("merge", &["ret void"]),
        ];
        assert_eq!(
            find_trivial_cross_arm(&blocks),
            Some(("%inner".to_string(), "%shared".to_string()))
        );
        let out = privatize_trivial_cross_arm(&blocks);
        // Exactly one clone of %shared (a verbatim `br label %merge`), no clone of the reconvergence.
        let clones: Vec<&BodyBlock> = out.iter().filter(|b| b.name.starts_with("%xa")).collect();
        assert_eq!(clones.len(), 1);
        assert_eq!(clones[0].lines(), vec!["br label %merge".to_string()]);
        // %inner now branches to the clone; original %shared kept (still reached from %entry).
        let inner = out.iter().find(|b| b.name == "%inner").unwrap();
        assert!(inner.lines().last().unwrap().contains(&clones[0].name));
        assert!(out.iter().any(|b| b.name == "%shared"));
        // The whole function now structures (via privatization + merge-not-dominated synth).
        assert!(super::super::structured_emit::structured_plan(&blocks).is_some());
    }

    /// The successor's phi gains an incoming for the clone, mirroring the original arm's value (valid
    /// because the value dominates the arm and hence the clone).
    #[test]
    fn privatize_trivial_mirrors_successor_phi_incoming() {
        let blocks = vec![
            blk("entry", &["br i1 %ca, label %shared, label %inner"]),
            blk("inner", &["br i1 %cb, label %shared, label %elseb"]),
            blk("shared", &["br label %merge"]),
            blk("elseb", &["br label %merge"]),
            blk(
                "merge",
                &["%p = phi i32 [ %v, %shared ], [ %w, %elseb ]", "ret void"],
            ),
        ];
        let out = privatize_trivial_cross_arm(&blocks);
        let clone_name = out
            .iter()
            .find(|b| b.name.starts_with("%xa"))
            .unwrap()
            .name
            .clone();
        let merge = out.iter().find(|b| b.name == "%merge").unwrap();
        let phi = merge.lines()[0].clone();
        // Original %shared incoming kept, and a mirrored incoming for the clone added (same value %v).
        assert!(phi.contains("[ %v, %shared ]"), "phi = {phi}");
        assert!(
            phi.contains(&format!("[ %v, {clone_name} ]")),
            "phi = {phi}"
        );
        assert!(phi.contains("[ %w, %elseb ]"), "phi = {phi}");
    }

    /// A shared arm that defines a value DEAD past the arm (`%d` used only inside `%shared`) IS cloned:
    /// the def is renamed into a fresh namespace and, because it never reaches `%merge`'s phi, the
    /// successor-phi patch carries only external values (the safe def-free pattern).
    #[test]
    fn privatize_trivial_clones_arm_with_dead_def() {
        let blocks = vec![
            blk("entry", &["br i1 %ca, label %shared, label %inner"]),
            blk("inner", &["br i1 %cb, label %shared, label %elseb"]),
            // %d and %u are used ONLY within %shared (%u feeds a store) — dead past the arm.
            blk(
                "shared",
                &[
                    "%d = add i32 %x, 1",
                    "%u = mul i32 %d, 2",
                    "store i32 %u, ptr %p",
                    "br label %merge",
                ],
            ),
            blk("elseb", &["br label %merge"]),
            blk("merge", &["ret void"]),
        ];
        assert_eq!(
            find_trivial_cross_arm(&blocks),
            Some(("%inner".to_string(), "%shared".to_string()))
        );
        let out = privatize_trivial_cross_arm(&blocks);
        let clone = out.iter().find(|b| b.name.starts_with("%xa")).unwrap();
        // The clone re-defines %d/%u under fresh names (no duplicate SSA), preserving the store.
        assert!(clone
            .lines()
            .iter()
            .any(|l| l.contains("%xa") && l.contains("add")));
        assert!(clone.lines().iter().any(|l| l.contains("store")));
        assert!(!clone
            .lines()
            .iter()
            .any(|l| line_def(l) == Some("%d".to_string())));
    }

    /// A shared arm whose def flows ONLY into the successor's PHI (`%d` in `%merge`'s phi incoming) is
    /// now CAPTURED: a phi incoming needs its value to dominate only the predecessor, so cloning the arm
    /// and mirroring `[ renamed, clone ]` keeps SSA sound. The clone renames `%d`; `%merge`'s phi keeps
    /// the original `[ %d, %shared ]` and gains the renamed clone incoming.
    #[test]
    fn privatize_trivial_clones_arm_def_used_only_in_successor_phi() {
        let blocks = vec![
            blk("entry", &["br i1 %ca, label %shared, label %inner"]),
            blk("inner", &["br i1 %cb, label %shared, label %elseb"]),
            blk("shared", &["%d = add i32 %x, 1", "br label %merge"]),
            blk("elseb", &["br label %merge"]),
            blk(
                "merge",
                &["%p = phi i32 [ %d, %shared ], [ %w, %elseb ]", "ret void"],
            ),
        ];
        assert_eq!(
            find_trivial_cross_arm(&blocks),
            Some(("%inner".to_string(), "%shared".to_string()))
        );
        let out = privatize_trivial_cross_arm(&blocks);
        let clone = out.iter().find(|b| b.name.starts_with("%xa")).unwrap();
        let clone_name = clone.name.clone();
        // Clone renames %d to a fresh name (no duplicate SSA).
        let renamed = clone.lines().iter().find_map(|l| line_def(l)).unwrap();
        assert_ne!(renamed, "%d");
        // %merge's phi keeps the original [ %d, %shared ] and gains [ renamed, clone ].
        let merge = out.iter().find(|b| b.name == "%merge").unwrap();
        let phi = merge.lines()[0].clone();
        assert!(phi.contains("[ %d, %shared ]"), "phi = {phi}");
        assert!(
            phi.contains(&format!("[ {renamed}, {clone_name} ]")),
            "phi = {phi}"
        );
    }

    /// A shared arm whose def is used in a BODY computation past the arm (`%d` in `%merge`'s
    /// `mul`, not a phi incoming) is still EXCLUDED: the split removes the arm's domination of the
    /// successor, so the body use is undefined on the clone edge (broken SSA — the `000ca89f`
    /// structured-exit breaker). Left for SSA reconstruction (phi insertion + use rewrite in `S`).
    #[test]
    fn privatize_trivial_skips_arm_def_used_in_successor_body() {
        let blocks = vec![
            blk("entry", &["br i1 %ca, label %shared, label %inner"]),
            blk("inner", &["br i1 %cb, label %shared, label %elseb"]),
            blk("shared", &["%d = add i32 %x, 1", "br label %merge"]),
            blk("elseb", &["br label %merge"]),
            blk("merge", &["%q = mul i32 %d, 3", "ret void"]),
        ];
        assert_eq!(find_trivial_cross_arm(&blocks), None);
        assert_eq!(privatize_trivial_cross_arm(&blocks).len(), blocks.len());
    }

    /// A single-successor cross-arm that ITSELF carries a phi is cloned with its incomings PARTITIONED:
    /// the original keeps the external-pred incoming, the clone the redirected-pred incoming (each renamed
    /// into the clone namespace), and `%merge`'s phi gains the renamed clone result.
    #[test]
    fn privatize_trivial_clones_arm_with_phi_partitions_incomings() {
        let blocks = vec![
            blk("entry", &["br i1 %ca, label %shared, label %inner"]),
            blk("inner", &["br i1 %cb, label %shared, label %elseb"]),
            blk(
                "shared",
                &[
                    "%p = phi i32 [ %a, %entry ], [ %b, %inner ]",
                    "br label %merge",
                ],
            ),
            blk("elseb", &["br label %merge"]),
            blk(
                "merge",
                &["%m = phi i32 [ %p, %shared ], [ %w, %elseb ]", "ret void"],
            ),
        ];
        assert_eq!(
            find_trivial_cross_arm(&blocks),
            Some(("%inner".to_string(), "%shared".to_string()))
        );
        let out = privatize_trivial_cross_arm(&blocks);
        // Original %shared's phi keeps only the external (%entry) incoming.
        let shared = out.iter().find(|b| b.name == "%shared").unwrap();
        assert!(
            shared.lines()[0].contains("[ %a, %entry ]"),
            "{}",
            shared.lines()[0]
        );
        assert!(
            !shared.lines()[0].contains("%inner"),
            "{}",
            shared.lines()[0]
        );
        // Clone keeps only the redirected (%inner) incoming, with the phi result renamed.
        let clone = out.iter().find(|b| b.name.starts_with("%xa")).unwrap();
        let renamed = clone.lines().iter().find_map(|l| line_def(l)).unwrap();
        assert_ne!(renamed, "%p");
        assert!(
            clone.lines()[0].contains("[ %b, %inner ]"),
            "{}",
            clone.lines()[0]
        );
        assert!(!clone.lines()[0].contains("%entry"), "{}", clone.lines()[0]);
        // %merge's phi keeps the original and gains [ renamed, clone ].
        let merge = out.iter().find(|b| b.name == "%merge").unwrap();
        assert!(
            merge.lines()[0].contains("[ %p, %shared ]"),
            "{}",
            merge.lines()[0]
        );
        assert!(
            merge.lines()[0].contains(&format!("[ {renamed}, {} ]", clone.name)),
            "{}",
            merge.lines()[0]
        );
    }

    /// Return unification remains a valid normalization for a divergent void selection even though
    /// the planner can now represent two terminal arms with a disconnected unreachable merge.
    #[test]
    fn unify_returns_void_makes_divergent_selection_structurable() {
        let blocks = vec![
            blk("h", &["br i1 %c, label %a, label %b"]),
            blk("a", &["ret void"]),
            blk("b", &["ret void"]),
        ];
        assert!(super::super::structured_emit::structured_plan_ladder(&blocks, false).is_some());
        let out = unify_returns(&blocks).expect("two rets to unify");
        // Both arms now branch to the same exit.
        let a = out.iter().find(|x| x.name == "%a").unwrap();
        let b = out.iter().find(|x| x.name == "%b").unwrap();
        assert_eq!(a.lines().last(), b.lines().last());
        assert!(a
            .lines()
            .last()
            .unwrap()
            .starts_with(format!("br label {URET_PREFIX}").as_str()));
        assert!(super::super::structured_emit::structured_plan(&out).is_some());
        let plan = super::super::structured_emit::structured_plan(&out)
            .expect("the explicitly unified graph must admit");
        assert!(
            plan.blocks
                .iter()
                .any(|block| block.name.starts_with(URET_PREFIX)),
            "admitted plan must contain the synthesized shared exit"
        );
    }

    /// An `unreachable` arm is return-like for structurization: lowering UB to the function's modeled
    /// return and then unifying exits gives the enclosing selection one shared merge.
    #[test]
    fn divergent_unreachable_and_return_gain_shared_exit() {
        let blocks = vec![
            blk("h", &["br i1 %c, label %a, label %b"]),
            blk("a", &["unreachable"]),
            blk("b", &["ret void"]),
        ];
        let candidate = separate_divergent_selection_exits(&blocks)
            .expect("return-like divergent exits must normalize");
        assert_eq!(
            candidate
                .iter()
                .filter(|block| block.name.starts_with(URET_PREFIX))
                .count(),
            1
        );
        assert!(super::super::structured_emit::structured_plan(&candidate).is_some());
    }

    /// A nested selection shares its value-return block with an enclosing path. Return unification
    /// supplies a common phi exit, but the nested header still cannot own that merge until the shared
    /// return predecessor is cloned for its dominated incoming edge.
    #[test]
    fn shared_phi_exit_predecessor_is_privatized_for_nested_selection() {
        let blocks = vec![
            blk("entry", &["br i1 %outer, label %ret, label %h"]),
            blk("h", &["br i1 %c, label %a, label %b"]),
            blk("a", &["br label %ret"]),
            blk("b", &["unreachable"]),
            blk(
                "ret",
                &["%rv = phi i32 [ 1, %entry ], [ 2, %a ]", "ret i32 %rv"],
            ),
        ];
        assert!(super::super::structured_emit::structured_plan_ladder(&blocks, false).is_some());
        let unified =
            separate_divergent_selection_exits(&blocks).expect("return-like exits must unify");
        let private = privatize_shared_phi_exit_predecessors(&unified);
        assert!(
            private.len() > unified.len(),
            "the shared return predecessor must gain a private clone"
        );
        assert!(
            super::super::structured_emit::structured_plan_ladder(&private, false).is_some(),
            "the ordinary ladder must admit after shared-exit separation"
        );
        assert!(
            super::super::structured_emit::structured_plan(&blocks).is_some(),
            "the reject-only C1 construction must compose both transforms"
        );
    }

    /// Value returns unify through a phi over the returned values.
    #[test]
    fn unify_returns_value_builds_phi() {
        let blocks = vec![
            blk("h", &["br i1 %c, label %a, label %b"]),
            blk("a", &["ret i32 %va"]),
            blk("b", &["ret i32 %vb"]),
        ];
        let out = unify_returns(&blocks).expect("two rets to unify");
        let exit = out
            .iter()
            .find(|x| x.name.starts_with(URET_PREFIX))
            .unwrap();
        assert_eq!(
            exit.lines()[0],
            format!("{URET_PREFIX}.v = phi i32 [ %va, %a ], [ %vb, %b ]").as_str()
        );
        assert_eq!(exit.lines()[1], format!("ret i32 {URET_PREFIX}.v").as_str());
        assert!(super::super::structured_emit::structured_plan(&out).is_some());
    }

    /// The merge-PRESERVING dominated-region clone on a MULTI-SUCCESSOR shared arm (the residual
    /// `cross-arm-shared` shape the trivial single-`br` privatizer leaves). `%shared` is a cross-arm of
    /// `%inner` (also reached from `%entry`) and is a CONDITIONAL (`br %cc, %left, %right`); its dominated
    /// region is `{%shared, %left, %right}`, reconverging at boundary `%merge`. The clone duplicates the
    /// whole region for `%inner`'s entry, keeps `%merge` intact (NOT cloned — reconvergence preserved),
    /// and the function then structures. Full-closure duplication would clone `%merge` too and destroy it.
    #[test]
    fn privatize_dominated_region_clones_multi_successor_arm() {
        let blocks = vec![
            blk("entry", &["br i1 %ca, label %shared, label %inner"]),
            blk("inner", &["br i1 %cb, label %shared, label %elseb"]),
            blk("shared", &["br i1 %cc, label %left, label %right"]),
            blk("left", &["br label %merge"]),
            blk("right", &["br label %merge"]),
            blk("elseb", &["br label %merge"]),
            blk("merge", &["ret void"]),
        ];
        let out = privatize_region_cross_arm(&blocks);
        // The region {shared,left,right} is cloned (3 fresh blocks); %merge is NOT cloned.
        let clones: Vec<&BodyBlock> = out.iter().filter(|b| b.name.starts_with("%xa")).collect();
        assert_eq!(clones.len(), 3, "shared+left+right cloned, merge preserved");
        assert_eq!(
            clones
                .iter()
                .map(|block| block.name.as_str())
                .collect::<Vec<_>>(),
            ["%xa0_shared", "%xa0_left", "%xa0_right"],
            "cloned blocks preserve their source block order"
        );
        assert_eq!(out.iter().filter(|b| b.name == "%merge").count(), 1);
        // The privatized graph now structures (the clone gives %inner a dominated arm). The clone is a
        // spirv-val-gated retry candidate, not on the default admission path, so we check the OUTPUT.
        assert!(super::super::structured_emit::structured_plan(&out).is_some());
    }

    /// The boundary block's phi gains a mirrored incoming for each cloned region predecessor, carrying
    /// the renamed region value — so SSA stays closed across the clone. Here `%merge`'s phi takes `%vl`
    /// from `%left` and `%vr` from `%right`; after cloning it must ALSO list `%left_clone`/`%right_clone`.
    #[test]
    fn privatize_dominated_region_mirrors_boundary_phi() {
        let blocks = vec![
            blk("entry", &["br i1 %ca, label %shared, label %inner"]),
            blk("inner", &["br i1 %cb, label %shared, label %elseb"]),
            blk("shared", &["br i1 %cc, label %left, label %right"]),
            blk("left", &["%vl = add i32 1, 1", "br label %merge"]),
            blk("right", &["%vr = add i32 2, 2", "br label %merge"]),
            blk("elseb", &["br label %merge"]),
            blk(
                "merge",
                &[
                    "%p = phi i32 [ %vl, %left ], [ %vr, %right ], [ 0, %elseb ]",
                    "ret void",
                ],
            ),
        ];
        let out = privatize_region_cross_arm(&blocks);
        let merge = out.iter().find(|b| b.name == "%merge").unwrap();
        let phi = merge
            .lines()
            .iter()
            .find(|l| l.contains("phi"))
            .unwrap()
            .clone();
        // Original three incomings preserved, plus one per cloned region predecessor (left', right').
        assert_eq!(
            phi.matches('[').count(),
            5,
            "3 original + 2 mirrored incomings"
        );
        assert!(phi.contains("%elseb")); // external incoming untouched
        assert!(super::super::structured_emit::structured_plan(&out).is_some());
    }

    /// An inner conditional can share two consecutive continuations with an enclosing arm. Neither
    /// continuation is the inner conditional's natural merge, so ordinary unique-merge synthesis only
    /// redirects the direct early-exit edge and leaves the deep path escaping the selection. The deep
    /// continuation pre-pass clones `%tail` first, then `%shared` for `%inner`, then `%shared` again for
    /// the nested `%inner_else` selection, so every nested arm reaches a private copy before
    /// reconverging at `%outer_merge`.
    #[test]
    fn privatize_deep_shared_continuations_clones_nested_shared_tail() {
        let blocks = vec![
            blk("entry", &["br i1 %co, label %inner, label %outer"]),
            blk("inner", &["br i1 %ci, label %direct, label %inner_else"]),
            blk("direct", &["br label %shared"]),
            blk(
                "inner_else",
                &["br i1 %ce, label %outer_merge, label %tail"],
            ),
            blk("outer", &["br label %tail"]),
            blk("tail", &["br label %shared"]),
            blk("shared", &["br label %outer_merge"]),
            blk("outer_merge", &["ret void"]),
        ];
        assert!(
            find_deep_shared_continuations(&blocks)
                .contains(&("%inner".to_string(), "%tail".to_string())),
            "the outer nested header sees its externally reachable continuation"
        );
        let out = privatize_deep_shared_continuations(&blocks);
        let clones: Vec<&BodyBlock> = out.iter().filter(|b| b.name.starts_with("%xa")).collect();
        assert_eq!(
            clones.len(),
            3,
            "the two nested selections each receive a private shared continuation"
        );
        let direct = out.iter().find(|b| b.name == "%direct").unwrap();
        assert!(
            direct.lines().last().unwrap().contains("%xa"),
            "inner true arm must enter the private downstream continuation"
        );
        let outer = out.iter().find(|b| b.name == "%outer").unwrap();
        assert_eq!(outer.lines().last(), Some(&"br label %tail".to_string()));
        assert!(
            super::super::structured_emit::structured_plan(&out).is_some(),
            "the cloned graph remains structurable"
        );
    }

    /// A switch may encode a compact fallthrough tail: cases `%one`/`%two`/`%three` share `%tail1`
    /// and `%tail2`, while the enclosing entry can branch directly to `%merge`. The switch-case
    /// transform makes the nested suffixes private per case without cloning the enclosing merge.
    #[test]
    fn privatize_switch_case_continuations_clones_shared_case_tails() {
        let blocks = vec![
            blk("entry", &["br i1 %co, label %sw, label %merge"]),
            blk(
                "sw",
                &[
                    "switch i32 %x, label %merge [ i32 4, label %four i32 3, label %three i32 2, label %two i32 1, label %one ]",
                ],
            ),
            blk("one", &["br label %tail2"]),
            blk("two", &["br label %tail1"]),
            blk("three", &["br label %tail1"]),
            blk("four", &["br label %merge"]),
            blk("tail1", &["br label %tail2"]),
            blk("tail2", &["br label %merge"]),
            blk("merge", &["ret void"]),
        ];
        assert!(
            find_switch_case_shared_continuations(&blocks)
                .contains(&("%one".to_string(), "%tail2".to_string())),
            "the first case sees its shared tail"
        );
        let out = privatize_switch_case_continuations(&blocks);
        let clones: Vec<&BodyBlock> = out.iter().filter(|b| b.name.starts_with("%xa")).collect();
        assert_eq!(clones.len(), 3, "each overlapping case tail is privatized");
        assert_eq!(out.iter().filter(|b| b.name == "%merge").count(), 1);
        assert!(
            super::super::structured_emit::structured_plan(&out).is_some(),
            "the privatized switch graph remains structurable"
        );
    }

    /// A case can also enter the switch's DEFAULT CASE through a conditional tail. The switch merge is
    /// ordinarily dominated in this shape, so generic tail sharing is left alone; the continuation is
    /// still a direct case root, though, and SPIR-V forbids one case construct from entering another.
    #[test]
    fn privatize_switch_case_continuations_splits_case_to_case_entry() {
        let blocks = vec![
            blk("entry", &["br label %sw"]),
            blk(
                "sw",
                &["switch i32 %x, label %default [ i32 0, label %a i32 1, label %b i32 2, label %c ]"],
            ),
            blk("a", &["br i1 %ca, label %default, label %a_tail"]),
            blk("a_tail", &["br label %default"]),
            blk("b", &["br label %default"]),
            blk("c", &["br label %merge"]),
            blk("default", &["br label %merge"]),
            blk("merge", &["ret void"]),
        ];
        assert!(
            find_switch_case_shared_continuations(&blocks)
                .contains(&("%a".to_string(), "%default".to_string())),
            "a case-to-case edge is recognized even with a switch-dominated merge"
        );
        let out = privatize_switch_case_continuations(&blocks);
        assert!(
            out.len() > blocks.len(),
            "case entry received a private copy"
        );
        assert_eq!(out.iter().filter(|b| b.name == "%default").count(), 1);
        assert!(
            super::super::structured_emit::structured_plan(&out).is_some(),
            "the split case-entry graph remains structurable"
        );
    }

    /// A switch inside a natural loop is deliberately outside the case-tail transform: its shared
    /// continuation can be a loop break/continue boundary, which needs a real multi-level-break model
    /// rather than tail duplication.
    #[test]
    fn privatize_switch_case_continuations_skips_loop_contained_switch() {
        let blocks = vec![
            blk("entry", &["br label %loop"]),
            blk("loop", &["br i1 %again, label %sw, label %exit"]),
            blk(
                "sw",
                &["switch i32 %x, label %exit [ i32 0, label %a i32 1, label %b ]"],
            ),
            blk("a", &["br label %shared"]),
            blk("b", &["br label %shared"]),
            blk("shared", &["br label %loop"]),
            blk("exit", &["ret void"]),
        ];
        assert_eq!(
            privatize_switch_case_continuations(&blocks).len(),
            blocks.len(),
            "loop-contained switch is left to the loop-aware structurizer"
        );
    }

    /// The cross-arm-EDGE shape self-check 2 misses: a block DEEP in one arm branches to a SIBLING arm
    /// of an ANCESTOR selection (not the header's direct arm). Here `entry` selects `%A` (then) / `%B`
    /// (else, merge `%M`); `%C` in `%B`'s subtree branches to `%A` — escaping `%B`'s arm into the then
    /// arm. `find_cross_arm_edge` reports `(%B, %A)`; `privatize_cross_arm_edge` clones `%A`'s region for
    /// `%C`'s entry so `%C` reaches a private `%A'` (dominated by `%B`) while the then-arm keeps `%A`,
    /// removing the cross-arm edge.
    #[test]
    fn privatize_cross_arm_edge_clones_ancestor_sibling_escape() {
        let blocks = vec![
            blk("entry", &["br i1 %c0, label %A, label %B"]),
            blk("A", &["br label %M"]),
            blk("B", &["br i1 %c1, label %C, label %D"]),
            blk("C", &["br label %A"]),
            blk("D", &["br label %M"]),
            blk("M", &["ret void"]),
        ];
        assert!(
            find_cross_arm_edge(&blocks).is_some(),
            "the %C->%A sibling-arm escape is detected"
        );
        let out = privatize_cross_arm_edge(&blocks);
        assert!(out.len() > blocks.len(), "a private clone of %A was added");
        assert!(
            find_cross_arm_edge(&out).is_none(),
            "the cross-arm edge is gone after privatization"
        );
    }
}
