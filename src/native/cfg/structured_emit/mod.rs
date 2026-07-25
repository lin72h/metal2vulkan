//! Forest-driven structured loop-merge computation — the first increment of the R2 emission
//! consumer (the wholesale replacement for the order-heuristic `infer_loop_merges` + post-hoc
//! `repair_structured_merges`, see [[metal2vulkan-native-emitter]]).
//!
//! `infer_loop_merges` guesses each loop's merge/continue from block order and then the emitter
//! leans on a bounded post-hoc repair to fix the guesses — which does not converge for the nested
//! merge==continue overlap (a previously observed module shape: an inner loop's single exit is also the enclosing
//! loop's continue/latch). Here we instead derive the merges from the dominator/natural-loop forest
//! ([`super::loopforest`]) so they are correct by construction, and split the merge==continue overlap
//! up front by inserting a pass-through merge block.
//!
//! Scope so far: it returns merges for loops that are **directly structurable** (single latch, single
//! non-shared exit) or that need only the **merge==continue split** — both the no-phi case
//! (`split_no_phi_overlap`) and the **phi-carrying case** (`split_phi_overlap`, which merges the
//! redirected predecessors' phi incomings into a phi in the synthesized pass-through and rewrites the
//! shared block's phi to take that merged value via the pass-through edge), and the **two-exit
//! `MultipleExits` case** (`synth_multi_exit_merge`: funnel both exits through one synthesized dispatch
//! merge with an `i1` selector phi + conditional branch back out, no-phi exits), and **do-while
//! rotation** (`synth_dowhile_continue`: split a bottom-test latch into a fresh continue block + a
//! `{continue, merge}` break, rewriting the header phi's back-edge predecessor). Loops needing
//! `MultipleLatches`, `MultipleExits` with >2 exits / phi-carrying exit targets, or a do-while
//! combined with multi-exit, are still left to the existing path. A narrow `NoExit` subset is also
//! structured: a header whose sole latch is its unconditional direct self branch is split into
//! header/body/continue blocks and given an unreachable merge.
//!
//! The planner is wired into the default structured-emission path. A former failure-triggered retry
//! (re-emitting with forest merges only when the heuristic path failed) was empirically disproven to
//! move the CFG frontier: its reachable triggers either could not be fixed by that partial consumer or
//! never fired, because spirv-val's back-edge notion is not pure dominance. Remaining classes still
//! need their own complete restructuring before they can replace the fallback path.

use super::blocks::{block_successors, conditional_branch_targets, synthetic_block};
use super::loopforest::{
    analyze, break_aware_selection_merges, selection_merges, LoopForest, Restructure,
};
use super::structured_order::{structured_order, structured_order_terminal};
// Re-bind the sibling module names so the responsibility bands can reach them via `super::`
// (they were direct `super::` siblings before this file became a directory module).
use super::{blocks, clone_crossarm, loopforest};
use super::{BlockRole, BodyBlock, LoopMergeInfo};
use std::collections::{HashMap, HashSet};

mod plan;
pub(in crate::native) use plan::*;
mod straddle;
pub(in crate::native) use straddle::*;
mod reject;
pub(in crate::native) use reject::*;
mod loop_merges;
pub(in crate::native) use loop_merges::*;
mod terminal_exit;
pub(in crate::native) use terminal_exit::*;
mod selection_merge;
pub(in crate::native) use selection_merge::*;
mod multi_exit;
pub(in crate::native) use multi_exit::*;
mod phi_util;
pub(in crate::native) use phi_util::*;
// The own-arm retry consumes the construct-tree planner core. The generic R1 route/re-nest
// materializers remain fixture-only until a later class needs the full regional dispatcher.
#[allow(dead_code)]
mod construct_tree;
mod own_arm;
pub(in crate::native) use own_arm::*;
mod straddle_region;
pub(in crate::native) use straddle_region::*;

#[cfg(test)]
mod construct_tree_fixtures;

#[cfg(test)]
mod tests {
    use super::*;

    fn bb(name: &str, lines: &[&str]) -> BodyBlock {
        // Populate the typed carrier exactly as production does at split time (the sole substrate).
        let lines: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
        let typed = crate::native::tir::lower_block_carrier(
            name,
            &lines,
            &std::collections::HashMap::new(),
        );
        BodyBlock {
            name: name.to_string(),
            role: BlockRole::Normal,
            typed,
        }
    }

    fn bb_role(name: &str, role: BlockRole, lines: &[&str]) -> BodyBlock {
        BodyBlock {
            role,
            ..bb(name, lines)
        }
    }

    /// A phi incoming VALUE rendered to a comparable token for the carrier-substrate test assertions
    /// (the cases these tests use: locals, integer constants, `true`/`false`, `undef`).
    fn val_str(v: &crate::native::ir::LlValue) -> String {
        use crate::native::ir::LlValue::*;
        match v {
            Local(n) => n.clone(),
            Int(i) => i.to_string(),
            Bool(true) => "true".to_string(),
            Bool(false) => "false".to_string(),
            Undef => "undef".to_string(),
            other => format!("{other:?}"),
        }
    }

    /// The `(value, predecessor)` incomings of the block-carrier phi named `dst` (panics if absent) — the
    /// carrier substitute for scanning a phi LINE in the block's `.lines()`.
    fn phi_incomings<'a>(
        block: &'a BodyBlock,
        dst: &str,
    ) -> &'a [(crate::native::ir::LlValue, String)] {
        let carrier = block.typed.as_ref().expect("carrier");
        carrier
            .insts
            .iter()
            .find(|i| i.is_phi() && i.result.as_deref() == Some(dst))
            .and_then(|i| i.phi_incoming.as_ref())
            .map(|(_, inc)| inc.as_slice())
            .unwrap_or_else(|| panic!("phi {dst} not found in {}", block.name))
    }

    /// Every phi in the block's carrier as `(result, incomings)` — for tests that scan for "a phi with
    /// these incomings" without knowing its result name.
    fn carrier_phis(block: &BodyBlock) -> Vec<(String, Vec<(crate::native::ir::LlValue, String)>)> {
        let carrier = block.typed.as_ref().expect("carrier");
        carrier
            .insts
            .iter()
            .filter(|i| i.is_phi())
            .filter_map(|i| {
                Some((
                    i.result.clone()?,
                    i.phi_incoming.as_ref().map(|(_, inc)| inc.clone())?,
                ))
            })
            .collect()
    }

    /// A nested guard may share its non-returning side-effect block with an enclosing arm while the
    /// other arm jumps to a common function return. The terminal-return planner gives the nested
    /// guard a private pass-through merge, leaving the shared side-effect block after the construct.
    #[test]
    fn terminal_exit_selection_routes_shared_continuation_after_merge() {
        let blocks = vec![
            bb("%entry", &["br label %outer"]),
            bb("%outer", &["br i1 %co, label %shared, label %inner"]),
            bb("%inner", &["br i1 %ci, label %shared, label %ret"]),
            bb("%shared", &["%local = add i32 %x, %y", "br label %guard"]),
            bb("%guard", &["br i1 %g, label %cont, label %tail"]),
            bb("%tail", &["call void @side_effect()", "br label %ret"]),
            bb("%cont", &["br i1 %more, label %ret, label %done"]),
            bb("%done", &["ret void"]),
            bb("%ret", &["ret void"]),
        ];
        let terminal = terminal_exit_selection_merges(&blocks).expect("terminal guard applies");
        let merge = terminal.merges.get("%inner").expect("inner private merge");
        assert!(merge.starts_with(format!("{SPLIT_PREFIX}{SEL_TOKEN}").as_str()));
        assert!(terminal
            .blocks
            .iter()
            .find(|block| block.name == *merge)
            .is_some_and(|block| block
                .lines()
                .last()
                .is_some_and(|line| line == "br label %shared")));
        assert!(terminal.blocks.iter().any(|block| {
            block.role == BlockRole::TerminalExitReturn
                && block.lines().len() == 1
                && block.lines().first().is_some_and(|line| line == "ret void")
        }));
        assert_eq!(terminal.merges.get("%outer"), Some(&"%shared".to_string()));
        assert!(terminal.merges.contains_key("%guard"));
    }

    /// A private linear return tail can safely receive a private merge and return clone. The retry
    /// itself remains reject-triggered, so already-admitting early-return functions do not change.
    #[test]
    fn terminal_exit_selection_clones_private_return_tail() {
        let blocks = vec![
            bb("%entry", &["br label %guard"]),
            bb("%guard", &["br i1 %c, label %shared, label %tail"]),
            bb("%tail", &["call void @side_effect()", "br label %ret"]),
            bb("%shared", &["%value = add i32 %x, %y", "br label %cont"]),
            bb("%cont", &["br i1 %more, label %ret, label %done"]),
            bb("%done", &["ret void"]),
            bb("%ret", &["ret void"]),
        ];
        let terminal = terminal_exit_selection_merges(&blocks).expect("terminal guard applies");
        assert!(terminal.merges.contains_key("%guard"));
        assert!(terminal.blocks.iter().any(|block| {
            block.role == BlockRole::TerminalExitReturn && block.lines() == ["ret void"]
        }));
    }

    /// A return shared by an early guard and a later loop cannot serve both structural roles. The
    /// loop-only edge is split to an empty private return, while the early guard keeps its source
    /// return edge intact.
    #[test]
    fn terminal_loop_return_privatization_redirects_only_loop_predecessor() {
        let blocks = vec![
            bb("%entry", &["br label %guard"]),
            bb("%guard", &["br i1 %early, label %ret, label %loop.header"]),
            bb("%loop.header", &["br label %loop.body"]),
            bb("%loop.body", &["br label %loop.latch"]),
            bb(
                "%loop.latch",
                &["br i1 %again, label %loop.header, label %ret"],
            ),
            bb("%ret", &["ret void"]),
        ];
        let private =
            privatize_single_loop_return_exit(&blocks).expect("simple shared return splits");
        let private_return = private
            .iter()
            .find(|block| {
                block
                    .name
                    .starts_with(format!("{SPLIT_PREFIX}{TLOOPRET_TOKEN}").as_str())
            })
            .expect("private return")
            .name
            .clone();
        let guard = private
            .iter()
            .find(|block| block.name == "%guard")
            .expect("guard");
        let latch = private
            .iter()
            .find(|block| block.name == "%loop.latch")
            .expect("latch");
        // The guard keeps its source-return edge (only the loop predecessor is redirected); the latch's
        // `%ret` edge is redirected to the private return — both read from the typed carrier, the
        // production substrate (the redirect no longer touches `.lines()`).
        assert!(block_successors(guard).iter().any(|s| s == "%ret"));
        assert!(block_successors(guard).iter().any(|s| s == "%loop.header"));
        assert!(block_successors(latch).contains(&private_return));
    }

    /// M1b switch-guard narrowing (`blocks_contain_multilevel_break_switch`): a switch is a dead-end #6
    /// multi-level break ONLY when it lives inside a loop and an arm leaves that loop. A loop-free switch
    /// and an in-loop switch whose arms all stay in the loop reconverge normally and are safe to admit via
    /// the break-aware/straddle/region-converge attempts.
    #[test]
    fn multilevel_break_switch_detects_only_loop_exiting_arms() {
        // (1) loop-free switch: no enclosing loop → not a multi-level break.
        let loop_free = vec![
            bb("%entry", &["br label %sw"]),
            bb(
                "%sw",
                &["switch i32 %v, label %m [ i32 0, label %a i32 1, label %b ]"],
            ),
            bb("%a", &["br label %m"]),
            bb("%b", &["br label %m"]),
            bb("%m", &["ret void"]),
        ];
        assert!(!blocks_contain_multilevel_break_switch(&loop_free));

        // (2) in-loop switch, every arm stays inside the loop → safe.
        let contained = vec![
            bb("%entry", &["br label %lh"]),
            bb("%lh", &["br i1 %cond, label %sw, label %exit"]),
            bb(
                "%sw",
                &["switch i32 %v, label %latch [ i32 0, label %a i32 1, label %b ]"],
            ),
            bb("%a", &["br label %latch"]),
            bb("%b", &["br label %latch"]),
            bb("%latch", &["br label %lh"]),
            bb("%exit", &["ret void"]),
        ];
        assert!(!blocks_contain_multilevel_break_switch(&contained));

        // (3) in-loop switch with an arm targeting a block OUTSIDE the loop → multi-level break.
        let breaking = vec![
            bb("%entry", &["br label %lh"]),
            bb("%lh", &["br i1 %cond, label %sw, label %exit"]),
            bb(
                "%sw",
                &["switch i32 %v, label %latch [ i32 0, label %a i32 1, label %exit ]"],
            ),
            bb("%a", &["br label %latch"]),
            bb("%latch", &["br label %lh"]),
            bb("%exit", &["ret void"]),
        ];
        assert!(blocks_contain_multilevel_break_switch(&breaking));
    }

    /// Ordinary post-dominance assigns an in-loop guarded-break selection the loop merge, even though
    /// its non-breaking paths converge at the latch. The loop-exit refinement must retain that latch as
    /// the selection convergence; the later unique-merge pass gives any shared use a private block.
    #[test]
    fn loop_exit_selection_refines_guarded_breaks_to_latch() {
        let blocks = vec![
            bb("%entry", &["br label %head"]),
            bb("%head", &["br label %top"]),
            bb("%top", &["br i1 %outer, label %a, label %latch"]),
            bb("%a", &["br i1 %break_now, label %exit, label %latch"]),
            bb("%latch", &["br i1 %again, label %cont, label %exit"]),
            bb("%cont", &["br label %head"]),
            bb("%exit", &["ret void"]),
        ];
        let forest = analyze(&blocks);
        let mut merges = selection_merges(&blocks, &forest);
        assert_eq!(merges.get("%a"), Some(&"%exit".to_string()));
        let loop_merges = HashMap::from([(
            "%head".to_string(),
            LoopMergeInfo {
                merge: "%exit".to_string(),
                continue_target: "%cont".to_string(),
            },
        )]);
        refine_loop_exit_selection_merges(&blocks, &forest, &loop_merges, &mut merges);
        assert!(
            matches!(merges.get("%a"), Some(merge) if merge == "%latch"),
            "guarded break should retain the latch as convergence: {merges:?}"
        );
    }

    /// `find_synthesized_cross_arm_shared` locates a `selection:cross-arm-shared` violation under the
    /// synthesized merge map and returns the RAW `(header, arm)` so the caller clones on raw blocks. The
    /// short-circuit `a || b` ladder is the canonical shape: `%entry` and `%checkb` both branch to the
    /// shared `%taken`, so `%checkb` does not dominate its arm `%taken`. Both are raw blocks → detected.
    #[test]
    fn synth_cross_arm_detects_shared_ladder_arm() {
        let blocks = vec![
            bb("%entry", &["br i1 %ca, label %taken, label %checkb"]),
            bb("%checkb", &["br i1 %cb, label %taken, label %elseb"]),
            bb("%taken", &["br label %end"]),
            bb("%elseb", &["br label %end"]),
            bb("%end", &["ret void"]),
        ];
        assert_eq!(
            find_synthesized_cross_arm_shared(&blocks, false, false),
            Some(("%checkb".to_string(), "%taken".to_string())),
        );
    }

    /// A clean diamond has no shared arm — every arm is privately dominated by its header — so the
    /// detector must NOT fire (guards against over-admission of a spirv-val-legal selection).
    #[test]
    fn synth_cross_arm_ignores_clean_diamond() {
        let blocks = vec![
            bb("%entry", &["br i1 %c, label %a, label %b"]),
            bb("%a", &["br label %m"]),
            bb("%b", &["br label %m"]),
            bb("%m", &["ret void"]),
        ];
        assert_eq!(
            find_synthesized_cross_arm_shared(&blocks, false, false),
            None,
        );
    }

    #[test]
    fn large_region_cross_arm_ladder_input_skips_clone_retry() {
        let mut blocks = vec![
            bb("%entry", &["br i1 %outer, label %shared, label %h"]),
            bb("%h", &["br i1 %c, label %shared, label %private"]),
            bb("%private", &["br label %merge"]),
            bb("%shared", &["br label %merge"]),
            bb("%merge", &["ret void"]),
        ];
        assert!(
            clone_crossarm::privatize_region_cross_arm(&blocks).len() > blocks.len(),
            "fixture should exercise the region-cross-arm clone below the ladder cap"
        );

        while blocks.len() <= REGION_CROSS_ARM_LADDER_MAX_BLOCKS {
            let name = format!("%pad{}", blocks.len());
            blocks.push(bb(&name, &["ret void"]));
        }

        let capped = privatize_region_cross_arm_for_ladder(&blocks);
        assert_eq!(
            capped.len(),
            blocks.len(),
            "large CFGs should skip the retry clone and keep the fallback path bounded"
        );

        let mut grows_past_cap = vec![
            bb("%entry", &["br i1 %outer, label %shared, label %h"]),
            bb("%h", &["br i1 %c, label %shared, label %private"]),
            bb("%private", &["br label %merge"]),
            bb("%shared", &["br label %merge"]),
            bb("%merge", &["ret void"]),
        ];
        while grows_past_cap.len() < REGION_CROSS_ARM_LADDER_MAX_BLOCKS {
            let name = format!("%grow_pad{}", grows_past_cap.len());
            grows_past_cap.push(bb(&name, &["ret void"]));
        }
        let raw = clone_crossarm::privatize_region_cross_arm(&grows_past_cap);
        assert!(
            raw.len() > REGION_CROSS_ARM_LADDER_MAX_BLOCKS,
            "fixture should prove the raw region clone can cross the ladder cap"
        );
        let capped = privatize_region_cross_arm_for_ladder(&grows_past_cap);
        assert_eq!(
            capped.len(),
            grows_past_cap.len(),
            "the ladder should skip region clones whose output crosses the cap"
        );
    }

    fn deep_shared_tail_fixture() -> Vec<BodyBlock> {
        vec![
            bb("%entry", &["br i1 %co, label %inner, label %outer"]),
            bb("%inner", &["br i1 %ci, label %direct, label %inner_else"]),
            bb("%direct", &["br label %shared"]),
            bb(
                "%inner_else",
                &["br i1 %ce, label %outer_merge, label %tail"],
            ),
            bb("%outer", &["br label %tail"]),
            bb("%tail", &["br label %shared"]),
            bb("%shared", &["br label %outer_merge"]),
            bb("%outer_merge", &["ret void"]),
        ]
    }

    #[test]
    fn shared_continuation_ladder_input_preserves_small_clone_and_skips_oversized_growth() {
        let small = deep_shared_tail_fixture();
        let small_out = privatize_shared_continuations_for_ladder(&small);
        assert!(
            small_out.len() > small.len(),
            "small shared-continuation fixtures should still run the pre-ladder clone"
        );

        let mut large = deep_shared_tail_fixture();
        while large.len() < SHARED_CONTINUATION_LADDER_MAX_BLOCKS - 2 {
            let name = format!("%pad{}", large.len());
            large.push(bb(&name, &["ret void"]));
        }
        let direct = clone_crossarm::privatize_deep_shared_continuations(&large);
        assert!(
            direct.len() > SHARED_CONTINUATION_LADDER_MAX_BLOCKS,
            "fixture should prove the raw clone can cross the ladder cap"
        );

        let capped = privatize_shared_continuations_for_ladder(&large);
        assert_eq!(
            capped.len(),
            large.len(),
            "the ladder should skip an oversized shared-continuation clone"
        );
    }

    #[test]
    fn selection_synth_growth_cap_only_rejects_oversized_growth() {
        assert!(!selection_synth_growth_exceeds_ladder_cap(
            SELECTION_SYNTH_GROWTH_MAX_BLOCKS,
            SELECTION_SYNTH_GROWTH_MAX_BLOCKS
        ));
        assert!(!selection_synth_growth_exceeds_ladder_cap(
            SELECTION_SYNTH_GROWTH_MAX_BLOCKS + 50,
            SELECTION_SYNTH_GROWTH_MAX_BLOCKS + 50
        ));
        assert!(selection_synth_growth_exceeds_ladder_cap(
            63,
            SELECTION_SYNTH_GROWTH_MAX_BLOCKS + 1
        ));
    }

    #[test]
    fn massive_structured_plan_input_declines_to_retry_path() {
        let mut blocks = vec![bb("%entry", &["br label %pad0"])];
        for idx in 0..=STRUCTURED_PLAN_MAX_BLOCKS {
            let name = format!("%pad{idx}");
            let next = format!("%pad{}", idx + 1);
            blocks.push(bb(&name, &[&format!("br label {next}")]));
        }
        blocks.push(bb(
            &format!("%pad{}", STRUCTURED_PLAN_MAX_BLOCKS + 1),
            &["ret void"],
        ));

        assert!(structured_plan(&blocks).is_none());
    }

    /// `synth_multi_latch_continue` funnels a loop's two back-edges through one synthesized latch `L`:
    /// `%l1`/`%l2` (which both branch to header `%h`) are redirected to `L`, `L` branches back to `%h`,
    /// the header's per-latch phi incomings are merged into a value phi in `L`, and the header phi keeps
    /// only its preheader incoming plus `[ merged, L ]`. The loop becomes single-latch.
    #[test]
    fn synth_multi_latch_unifies_two_back_edges() {
        let mut blocks = vec![
            bb("%entry", &["br label %h"]),
            bb(
                "%h",
                &[
                    "%x = phi i32 [ 0, %entry ], [ %a, %l1 ], [ %b, %l2 ]",
                    "br i1 %c, label %body, label %exit",
                ],
            ),
            bb("%body", &["br i1 %d, label %l1, label %l2"]),
            bb("%l1", &["%a = add i32 %x, 1", "br label %h"]),
            bb("%l2", &["%b = add i32 %x, 2", "br label %h"]),
            bb("%exit", &["ret void"]),
        ];
        let latches = vec!["%l1".to_string(), "%l2".to_string()];
        let mut counter = 0usize;
        let l = synth_multi_latch_continue(&mut blocks, "%h", &latches, &mut counter)
            .expect("two-latch loop unifies");
        // The original latches no longer branch to the header (redirected to L); only L and the
        // preheader `%entry` reach the header now — so L is the single back-edge (latch).
        let to_header: Vec<&str> = blocks
            .iter()
            .filter(|b| block_successors(b).iter().any(|s| s == "%h"))
            .map(|b| b.name.as_str())
            .collect();
        assert_eq!(to_header, vec!["%entry", l.as_str()]);
        for latch in ["%l1", "%l2"] {
            let b = blocks.iter().find(|b| b.name == latch).unwrap();
            assert!(block_successors(b).iter().any(|s| s == &l));
            assert!(!block_successors(b).iter().any(|s| s == "%h"));
        }
        // The header phi keeps its preheader incoming + a single [ merged, L ] (2 incomings, no %l1/%l2).
        let h = blocks.iter().find(|b| b.name == "%h").unwrap();
        let hphi = phi_incomings(h, "%x");
        assert_eq!(hphi.len(), 2);
        assert!(hphi.iter().any(|(_, p)| p == "%entry"));
        assert!(hphi.iter().any(|(_, p)| *p == l));
        assert!(!hphi.iter().any(|(_, p)| p == "%l1" || p == "%l2"));
        // L carries the merged value phi over both original latches.
        let lb = blocks.iter().find(|b| b.name == l).unwrap();
        let lphi = &carrier_phis(lb)[0].1;
        assert_eq!(lphi.len(), 2);
        assert!(lphi.iter().any(|(_, p)| p == "%l1"));
        assert!(lphi.iter().any(|(_, p)| p == "%l2"));
    }

    /// A pure self-latching infinite loop has no source exit and uses its header as the back-edge.
    /// `synth_noexit_self_latch` makes both loop roles explicit: header phis flow from a distinct
    /// continue block, and a never-executed `unreachable` merge gives `OpLoopMerge` a legal target.
    #[test]
    fn self_latching_noexit_loop_gets_distinct_continue_and_unreachable_merge() {
        let blocks = vec![
            bb("%entry", &["br label %loop"]),
            bb(
                "%loop",
                &[
                    "%x = phi i32 [ 0, %entry ], [ %next, %loop ]",
                    "%next = add i32 %x, 1",
                    "br label %loop",
                ],
            ),
        ];

        let plan = structured_plan(&blocks).expect("pure self loop must structure");
        let info = plan
            .loop_merges
            .get("%loop")
            .expect("loop roles must be synthesized");
        assert_ne!(info.merge, "%loop");
        assert_ne!(info.continue_target, "%loop");
        assert_ne!(info.merge, info.continue_target);

        let header = plan
            .blocks
            .iter()
            .find(|b| b.name == "%loop")
            .expect("original header remains");
        assert!(header.lines()[0].contains(&format!(", {} ]", info.continue_target)));
        let body = header
            .lines()
            .last()
            .and_then(|line| line.strip_prefix("br label ").map(str::to_string))
            .expect("header branches to split body");
        let body = plan
            .blocks
            .iter()
            .find(|b| b.name == body)
            .expect("split body exists");
        assert_eq!(
            body.lines().last().map(String::as_str),
            Some(format!("br label {}", info.continue_target).as_str())
        );
        let cont = plan
            .blocks
            .iter()
            .find(|b| b.name == info.continue_target)
            .expect("continue exists");
        assert_eq!(cont.lines(), vec!["br label %loop"]);
        let merge = plan
            .blocks
            .iter()
            .find(|b| b.name == info.merge)
            .expect("unreachable merge exists");
        assert_eq!(merge.lines(), vec!["unreachable"]);
    }

    /// BC-neutrality invariant for the M-B1 straddle restructure (05/`b00a8a8d`). The genuine
    /// straddle-loop-merge shape only arises in `MPSRNNBreakUpToOutputVecs`'s full 54-block function
    /// (its two nested `if(!c) return` guards over a loop whose exit merge is the shared OpReturn block);
    /// the positive direction — that `restructure_straddle_loop_merges` gives that loop its own merge and
    /// clears the last NO_REPAIR blocker — is proven by the `NO_REPAIR --list-fail` = EMPTY integration
    /// battery. The load-bearing UNIT guarantee is the other side: the pre-pass runs ONLY on the
    /// cfg-restructure retry emit, and must be a strict NO-OP (`None`) on any function that already
    /// admits `structured_plan` — otherwise it could perturb a currently-admitting or banked case and
    /// break byte-baseline neutrality. This locks that contract on a spread of admitting loop shapes,
    /// including ones whose loop exits directly to an enclosing guard/selection merge (the family the
    /// self-check is closest to firing on, all resolved by the in-plan collision splitter instead).
    #[test]
    fn straddle_restructure_is_noop_on_admitting_shapes() {
        let admitting: Vec<(&str, Vec<BodyBlock>)> = vec![
            (
                "nested guards over a loop, shared OpReturn merge",
                vec![
                    bb("%entry", &["br label %g1"]),
                    bb("%g1", &["br i1 %c1, label %g2, label %ret"]),
                    bb("%g2", &["br i1 %c2, label %lh, label %ret"]),
                    bb("%lh", &["br i1 %cond, label %latch, label %ret"]),
                    bb("%latch", &["br label %lh"]),
                    bb("%ret", &["ret void"]),
                ],
            ),
            (
                "loop exits to enclosing selection merge",
                vec![
                    bb("%entry", &["br label %sel"]),
                    bb("%sel", &["br i1 %c, label %pre, label %m"]),
                    bb("%pre", &["br label %lh"]),
                    bb("%lh", &["br i1 %cond, label %latch, label %m"]),
                    bb("%latch", &["br label %lh"]),
                    bb("%m", &["br label %post"]),
                    bb("%post", &["ret void"]),
                ],
            ),
            (
                "loop exits via intermediate to guard merge",
                vec![
                    bb("%entry", &["br i1 %c1, label %body, label %gm"]),
                    bb("%body", &["br label %lh"]),
                    bb("%lh", &["br i1 %cond, label %latch, label %lx"]),
                    bb("%latch", &["br label %lh"]),
                    bb("%lx", &["br label %gm"]),
                    bb("%gm", &["ret void"]),
                ],
            ),
        ];
        for (name, blocks) in &admitting {
            assert!(
                structured_plan(blocks).is_some(),
                "precondition: {name} must admit structured_plan"
            );
            assert!(
                restructure_straddle_loop_merges(blocks).is_none(),
                "BC-neutrality: restructure must be a no-op on the admitting shape {name}"
            );
        }
    }

    /// A short-circuit `a || b` ladder funnels every condition's taken-arm into one shared block
    /// (`%shared`, reached from both headers) that the inner header `%h1` does not dominate — the
    /// `selection:cross-arm-shared` reject (banked `06502644`). Because `%shared` is a TRIVIAL
    /// pass-through (`br label %merge`, no phi/def), the S15 pre-pass clones it: `%h1`'s entry gets a
    /// private copy so `%h1` dominates its arm, `%entry`'s edge keeps the original `%shared`, and the
    /// residual `merge-not-dominated` is synthesized. `structured_plan` now ADMITS instead of falling
    /// back to the repair path — and the reject diagnostic agrees (mirror consistency).
    #[test]
    fn cross_arm_shared_convergence_is_structured_by_trivial_privatization() {
        let blocks = vec![
            bb("%entry", &["br label %h0"]),
            bb("%h0", &["br i1 %a, label %shared, label %h1"]),
            bb("%h1", &["br i1 %b, label %shared, label %elseb"]),
            bb("%shared", &["br label %merge"]),
            bb("%elseb", &["br label %merge"]),
            bb("%merge", &["ret void"]),
        ];
        let plan = structured_plan(&blocks)
            .expect("trivial cross-arm privatization structures the short-circuit ladder");
        assert!(
            plan.blocks.iter().any(|b| b.name.starts_with("%xa")),
            "a private clone of the shared pass-through must be present"
        );
        assert!(
            structured_reject_reason(&blocks).is_none(),
            "the reject diagnostic must mirror admission after privatization"
        );
    }

    /// A `switch` nested inside an outer selection whose default targets a SIBLING arm (`%sib`) of that
    /// outer selection (banked `6ceaffd7`/`b8eba912`/`e848bc87`). `%sib` is a trivial pass-through
    /// (`br label %merge`), so the S15 pre-pass clones it for the switch's own entry — the switch then
    /// dominates its arm and the residual is synthesized. `structured_plan` now ADMITS.
    #[test]
    fn switch_arm_to_enclosing_sibling_is_structured_by_trivial_privatization() {
        let blocks = vec![
            bb("%entry", &["br label %h0"]),
            bb("%h0", &["br i1 %a, label %sw, label %sib"]),
            bb(
                "%sw",
                &["switch i32 %v, label %sib [ i32 0, label %c0 i32 1, label %swm ]"],
            ),
            bb("%c0", &["br label %swm"]),
            bb("%swm", &["br label %merge"]),
            bb("%sib", &["br label %merge"]),
            bb("%merge", &["ret void"]),
        ];
        let plan = structured_plan(&blocks)
            .expect("trivial cross-arm privatization structures the switch-to-sibling shape");
        assert!(
            plan.blocks.iter().any(|b| b.name.starts_with("%xa")),
            "a private clone of the shared sibling pass-through must be present"
        );
    }

    /// A nested switch whose arms reconverge at a block that an ENCLOSING selection also branches to as
    /// one of its arms: the switch's NATURAL merge is then not dominated by the switch header (an outer
    /// edge reaches it directly). Rather than reject as `selection:merge-not-dominated` (the old
    /// behavior — this shape, banked `e848bc87`/`72cbab44`, was punted to the repair path), the
    /// non-dominated natural merge is now treated as a collision and `unique_selection_merges` inserts a
    /// header-dominated pass-through merge: the switch's own reconvergence is redirected to a fresh block
    /// whose predecessors are all switch-dominated, and that fresh block flows on to the shared block. So
    /// the switch gains a merge it dominates and the plan admits with a valid structured CFG.
    #[test]
    fn switch_merge_shared_with_enclosing_arm_is_synthesized() {
        let blocks = vec![
            bb("%entry", &["br i1 %c0, label %b, label %x"]),
            bb("%b", &["br i1 %c1, label %sw, label %bm"]),
            bb(
                "%sw",
                &["switch i32 %v, label %x [ i32 0, label %k0 i32 1, label %k1 ]"],
            ),
            bb("%k0", &["br label %x"]),
            bb("%k1", &["br label %x"]),
            // %bm bypasses %x straight to %exit, so %x is the switch's UNIQUE natural merge (no
            // collision detected by merge-claims) — yet %entry also branches to %x as an arm, so %sw
            // does not dominate it. The dominance-aware collision test now catches this and synthesizes.
            bb("%bm", &["br label %exit"]),
            bb("%x", &["br label %exit"]),
            bb("%exit", &["ret void"]),
        ];
        let plan =
            structured_plan(&blocks).expect("dominance-aware synth admits the shared-merge switch");
        // The switch's declared merge must now be dominated by the switch header (the whole point).
        let forest = analyze(&plan.blocks);
        let merge = plan
            .switch_merges
            .get("%sw")
            .expect("switch header has a structured merge");
        assert!(
            forest.dominates("%sw", merge),
            "the synthesized switch merge {merge} must be dominated by the switch header %sw"
        );
    }

    /// A switch whose arm (`%sib`) and default (`%bm`) both target SIBLING arms of enclosing
    /// selections. Both are trivial pass-throughs (`br label %exit`), so the S15 pre-pass privatizes
    /// each in turn (fixpoint) — the switch gets private copies it dominates, the external entries keep
    /// the originals, and the residual merges synthesize. `structured_plan` now ADMITS.
    #[test]
    fn switch_arm_targets_enclosing_sibling_is_structured_by_trivial_privatization() {
        let blocks = vec![
            bb("%entry", &["br i1 %c0, label %b, label %sib"]),
            bb("%b", &["br i1 %c1, label %sw, label %bm"]),
            bb(
                "%sw",
                &["switch i32 %v, label %bm [ i32 0, label %sib i32 1, label %k1 ]"],
            ),
            bb("%sib", &["br label %exit"]),
            bb("%k1", &["br label %exit"]),
            bb("%bm", &["br label %exit"]),
            bb("%exit", &["ret void"]),
        ];
        let plan = structured_plan(&blocks)
            .expect("fixpoint trivial privatization structures the multi-sibling switch");
        // Two shared arms privatized → at least one private clone present.
        assert!(
            plan.blocks.iter().any(|b| b.name.starts_with("%xa")),
            "a private clone of a shared sibling pass-through must be present"
        );
    }

    /// A loop header ending in a `switch` whose targets include the loop's merge (a `while`-style switch
    /// exit test) is NOT splittable — `split_loop_header_switch` requires every target be a genuine
    /// in-loop block — and would need both OpLoopMerge and OpSelectionMerge on one block. `structured_plan`
    /// must still reject this residue so it falls back to the repair path (banked `423ff479`).
    #[test]
    fn loop_header_switch_is_rejected() {
        let blocks = vec![
            bb("%entry", &["br label %h"]),
            bb(
                "%h",
                &["switch i32 %sel, label %merge [ i32 0, label %body i32 1, label %merge ]"],
            ),
            bb("%body", &["br label %h"]),
            bb("%merge", &["ret void"]),
        ];
        assert!(
            structured_plan(&blocks).is_none(),
            "a loop header that is also a switch must not be admitted"
        );
    }

    /// A loop header whose `switch` targets are all GENUINE in-loop blocks (none is the loop's
    /// merge/continue/header) is split by `split_loop_header_switch`: the switch is lifted into a fresh
    /// successor so the header branches unconditionally and the lifted block becomes an ordinary switch
    /// header. `structured_plan` then ADMITS it (this shape was previously rejected as
    /// `loop:loop-header-switch` and forced to the repair fixpoint).
    #[test]
    fn loop_header_in_loop_switch_is_split_and_admitted() {
        let blocks = vec![
            bb("%entry", &["br label %h"]),
            bb(
                "%h",
                &[
                    "%i = phi i32 [ 0, %entry ], [ %n, %latch ]",
                    "switch i32 %sel, label %a [ i32 0, label %b ]",
                ],
            ),
            bb("%a", &["br label %latch"]),
            bb("%b", &["br label %latch"]),
            bb(
                "%latch",
                &[
                    "%n = add i32 %i, 1",
                    "%done = icmp eq i32 %n, 4",
                    "br i1 %done, label %exit, label %h",
                ],
            ),
            bb("%exit", &["ret void"]),
        ];
        assert!(
            structured_plan(&blocks).is_some(),
            "an in-loop switch at a loop header must be split and admitted"
        );
    }

    #[test]
    fn redirect_label_respects_identifier_boundary() {
        let line = "br i1 %c, label %bb1, label %bb10";
        let out = redirect_label(line, "%bb1", "%new");
        assert_eq!(out, "br i1 %c, label %new, label %bb10");
    }

    #[test]
    fn redirect_label_unconditional() {
        assert_eq!(
            redirect_label("br label %exit", "%exit", "%m"),
            "br label %m"
        );
    }

    #[test]
    fn is_phi_line_detects_phi() {
        assert!(is_phi_line("  %x = phi i32 [ 0, %a ], [ %y, %b ]"));
        assert!(!is_phi_line("  %x = add i32 %a, %b"));
        assert!(!is_phi_line("br label %x"));
    }

    /// A single if/else with a unique merge: no synthesis, the header keeps its natural merge.
    #[test]
    fn unique_selection_merge_keeps_unshared_merge() {
        let blocks = vec![
            bb("%entry", &["br i1 %c, label %a, label %b"]),
            bb("%a", &["br label %m"]),
            bb("%b", &["br label %m"]),
            bb("%m", &["ret void"]),
        ];
        let (out, branch, switch) = unique_selection_merges(&blocks, &HashMap::new(), false);
        assert_eq!(out.len(), 4, "no block synthesized for a unique merge");
        assert!(switch.is_empty());
        assert_eq!(
            branch.get(&("%a".to_string(), "%b".to_string())),
            Some(&"%m".to_string())
        );
    }

    /// A nested if whose inner and outer constructs share the SAME post-dominator merge `%m`
    /// (`entry?(a,b)`, `a?(c,d)`, with `c,d,b` all branching to `%m`). Both `entry` and `a`
    /// post-dominate at `%m`; each must get a DISTINCT merge so neither emits an `OpSelectionMerge`
    /// onto a block another header already claims.
    #[test]
    fn unique_selection_merge_splits_shared_merge() {
        let blocks = vec![
            bb("%entry", &["br i1 %c0, label %a, label %b"]),
            bb("%a", &["br i1 %c1, label %c, label %d"]),
            bb("%c", &["br label %m"]),
            bb("%d", &["br label %m"]),
            bb("%b", &["br label %m"]),
            bb("%m", &["ret void"]),
        ];
        let (out, branch, _switch) = unique_selection_merges(&blocks, &HashMap::new(), false);
        let entry_merge = branch.get(&("%a".to_string(), "%b".to_string())).unwrap();
        let inner_merge = branch.get(&("%c".to_string(), "%d".to_string())).unwrap();
        assert_ne!(
            entry_merge, inner_merge,
            "nested constructs sharing %m get distinct merges"
        );
        // At least one synthesized pass-through block was inserted (prefix-tagged), branching onward.
        let synth: Vec<&BodyBlock> = out
            .iter()
            .filter(|b| b.name.starts_with(SPLIT_PREFIX))
            .collect();
        assert!(!synth.is_empty(), "a unique merge was synthesized");
        for s in &synth {
            assert!(s.lines().last().unwrap().starts_with("br label "));
        }
    }

    /// Nested selections that share the SAME natural merge `%r`: the inner construct synthesizes its
    /// own pass-through merge, and the enclosing header's sweep must redirect that synthesized block
    /// too (it is a construct-internal predecessor of `%r`). The synth dominance must therefore be
    /// recomputed after each insertion — using the pre-synthesis forest leaves the inner synth on `%r`,
    /// so `%r` keeps two predecessors and the outer merge branches back into its own construct
    /// (spirv-val: "branches to the selection construct, but not to the selection header").
    #[test]
    fn unique_selection_merge_redirects_inner_synth_for_enclosing_header() {
        let blocks = vec![
            bb("%entry", &["br i1 %c0, label %a, label %s"]),
            bb("%a", &["br i1 %c1, label %x, label %y"]),
            bb("%x", &["br label %r"]),
            bb("%y", &["br label %r"]),
            bb("%s", &["br label %r"]),
            bb("%r", &["ret void"]),
        ];
        let (out, _branch, _switch) = unique_selection_merges(&blocks, &HashMap::new(), false);
        // After both constructs claim unique merges, the shared `%r` must have exactly ONE predecessor
        // (the outermost synthesized merge): everything else funnels through the nested merges.
        let preds = out
            .iter()
            .filter(|b| block_successors(b).iter().any(|s| s == "%r"))
            .count();
        assert_eq!(
            preds, 1,
            "the shared natural merge %r must be reached through a single (outer) merge block, \
             not directly from an inner construct's synthesized merge"
        );
    }

    /// A fully-structurable if/else yields a plan with a unique branch merge.
    #[test]
    fn structured_plan_some_for_simple_if() {
        let blocks = vec![
            bb("%entry", &["br i1 %c, label %a, label %b"]),
            bb("%a", &["br label %m"]),
            bb("%b", &["br label %m"]),
            bb("%m", &["ret void"]),
        ];
        let plan = structured_plan(&blocks).expect("simple if is fully structurable");
        assert_eq!(
            plan.branch_merges
                .get(&("%a".to_string(), "%b".to_string())),
            Some(&"%m".to_string())
        );
    }

    /// A loop header that also ends in an in-loop conditional (an `if` whose arms reconverge inside the
    /// loop, not at the loop merge/continue) is split: the header branches unconditionally to a fresh
    /// selection block that hosts the conditional, which then gets a normal branch merge. Without the
    /// split the header would carry both OpLoopMerge and a bare conditional → "Selection must be
    /// structured".
    #[test]
    fn loop_header_with_in_loop_selection_is_split() {
        // entry -> H ; H: if(c) A else B ; A,B -> J ; J -> latch ; latch: back to H / exit.
        let blocks = vec![
            bb("%entry", &["br label %H"]),
            bb("%H", &["br i1 %c, label %A, label %B"]),
            bb("%A", &["br label %J"]),
            bb("%B", &["br label %J"]),
            bb("%J", &["br label %latch"]),
            bb("%latch", &["br i1 %d, label %H, label %exit"]),
            bb("%exit", &["ret void"]),
        ];
        let plan =
            structured_plan(&blocks).expect("loop-header selection is structurable after split");
        // The conditional now lives in a synthesized selection block, not the loop header. The header
        // (still the OpLoopMerge carrier) must end in an unconditional branch.
        let header = plan
            .blocks
            .iter()
            .find(|b| b.name == "%H")
            .expect("header retained");
        assert!(
            header
                .lines()
                .last()
                .is_some_and(|l| l.trim_start().starts_with("br label ")),
            "loop header must branch unconditionally after the split, got {:?}",
            header.lines().last()
        );
        // The lifted conditional (arms %A/%B) is recorded with %J as its branch merge.
        assert_eq!(
            plan.branch_merges
                .get(&("%A".to_string(), "%B".to_string())),
            Some(&"%J".to_string()),
            "branch_merges: {:?}",
            plan.branch_merges
        );
    }

    /// A simple conditional whose arms both return has no shared enclosing predecessor to separate, so
    /// it stays on the established fallback path. C1 admits only the narrower shared-exit class.
    #[test]
    fn structured_plan_none_for_unshared_exit_reconvergence() {
        let blocks = vec![
            bb("%entry", &["br i1 %c, label %a, label %b"]),
            bb("%a", &["ret void"]),
            bb("%b", &["ret void"]),
        ];
        assert!(
            structured_plan(&blocks).is_none(),
            "unshared exit reconvergence must retain its established fallback"
        );
    }

    /// A selection whose natural merge collides with a loop's continue target gets a synthesized merge
    /// rather than reusing the loop role.
    #[test]
    fn unique_selection_merge_avoids_loop_role_collision() {
        // header loop; inside, an if reconverges at the latch (== loop continue).
        let blocks = vec![
            bb("%entry", &["br label %h"]),
            bb("%h", &["br i1 %c, label %sel, label %exit"]),
            bb("%sel", &["br i1 %c2, label %t, label %f"]),
            bb("%t", &["br label %latch"]),
            bb("%f", &["br label %latch"]),
            bb("%latch", &["br label %h"]),
            bb("%exit", &["ret void"]),
        ];
        let loop_merges = HashMap::from([(
            "%h".to_string(),
            LoopMergeInfo {
                merge: "%exit".to_string(),
                continue_target: "%latch".to_string(),
            },
        )]);
        let (out, branch, _switch) = unique_selection_merges(&blocks, &loop_merges, false);
        if let Some(sel_merge) = branch.get(&("%t".to_string(), "%f".to_string())) {
            assert_ne!(
                sel_merge, "%latch",
                "selection merge must not reuse the loop continue"
            );
            assert!(out.iter().any(|b| &b.name == sel_merge));
        }
    }

    /// Simple single loop: header branches into body/exit, latch branches back. Directly
    /// structurable -> merge=exit, continue=latch, no new blocks.
    #[test]
    fn simple_loop_directly_structurable() {
        let blocks = vec![
            bb("%entry", &["br label %head"]),
            bb("%head", &["br i1 %c, label %body, label %exit"]),
            bb("%body", &["br label %head"]),
            bb("%exit", &["ret void"]),
        ];
        let (out, merges) = forest_loop_merges(&blocks, false, false);
        assert_eq!(
            out.len(),
            4,
            "no blocks added for a directly-structurable loop"
        );
        let info = merges.get("%head").expect("head loop merge recorded");
        assert_eq!(info.merge, "%exit");
        assert_eq!(info.continue_target, "%body");
    }

    /// Nested merge==continue overlap (the 2647a6f3 shape) with NO phi on the shared block: the
    /// inner loop's exit is the outer loop's latch/continue. Expect a synthesized pass-through merge
    /// for the inner loop, the outer continue preserved, and the inner predecessor redirected.
    #[test]
    fn merge_is_enclosing_continue_no_phi_splits() {
        // outer: %outer -> ... -> %latch -> %outer ; inner: %inner -> %ibody -> %latch(exit==outer latch)
        let blocks = vec![
            bb("%entry", &["br label %outer"]),
            bb("%outer", &["br i1 %co, label %inner, label %done"]),
            bb("%inner", &["br i1 %ci, label %ibody, label %latch"]),
            bb("%ibody", &["br label %inner"]),
            bb("%latch", &["br label %outer"]),
            bb("%done", &["ret void"]),
        ];
        let (out, merges) = forest_loop_merges(&blocks, false, false);
        // A pass-through merge block was synthesized.
        let synth: Vec<&BodyBlock> = out
            .iter()
            .filter(|b| b.name.starts_with(SPLIT_PREFIX))
            .collect();
        assert_eq!(synth.len(), 1, "one pass-through merge inserted");
        let new_merge = synth[0].name.clone();
        assert_eq!(synth[0].lines(), vec!["br label %latch".to_string()]);

        // Inner loop merge points at the synthesized block, continue at its latch (%inner is its own
        // latch via %ibody).
        let inner = merges.get("%inner").expect("inner loop merge recorded");
        assert_eq!(inner.merge, new_merge);

        // The inner predecessor that used to branch to %latch now branches to the new merge.
        let inner_blk = out.iter().find(|b| b.name == "%inner").unwrap();
        assert!(
            inner_blk.lines().last().unwrap().contains(&new_merge),
            "inner predecessor redirected to synthesized merge"
        );
        // The outer latch still branches back to the outer header (continue preserved).
        let latch = out.iter().find(|b| b.name == "%latch").unwrap();
        assert_eq!(latch.lines().last().unwrap(), "br label %outer");
    }

    /// A shared phi-carrying selection merge (an if-without-else chain whose every else-arm targets
    /// the same merge block, which carries a phi over all the arms). `unique_selection_merges` must
    /// give the colliding header a fresh pass-through merge AND fold the redirected arms' phi incomings
    /// into a merged phi in that pass-through, rebuilding the shared block's phi — so `structured_plan`
    /// admits.
    #[test]
    fn shared_phi_selection_merge_gets_phi_aware_split() {
        let blocks = vec![
            bb("%entry", &["br i1 %ca, label %outer_then, label %m"]),
            bb("%outer_then", &["br i1 %cb, label %inner_then, label %m"]),
            bb("%inner_then", &["br label %m"]),
            bb(
                "%m",
                &[
                    "%p = phi i32 [ 0, %entry ], [ 1, %outer_then ], [ 2, %inner_then ]",
                    "ret void",
                ],
            ),
        ];
        let plan = structured_plan(&blocks).expect("shared phi merge fully structured");
        // A pass-through carrying a merged phi over the redirected arms was synthesized.
        let synth: Vec<&BodyBlock> = plan
            .blocks
            .iter()
            .filter(|b| b.name.starts_with(SPLIT_PREFIX))
            .collect();
        assert!(!synth.is_empty(), "a pass-through merge was synthesized");
        assert!(
            synth
                .iter()
                .any(|b| b.lines().iter().any(|l| l.contains("= phi "))),
            "the pass-through carries a merged phi"
        );
        // The original merge's phi was rebuilt to take a merged value via a pass-through (it no longer
        // references all three original predecessors directly).
        let m = plan.blocks.iter().find(|b| b.name == "%m").unwrap();
        let phi = m
            .lines()
            .iter()
            .find(|l| l.contains("= phi "))
            .unwrap()
            .clone();
        assert!(
            phi.contains(SPLIT_PREFIX),
            "merge phi takes a value via the synthesized pass-through: {phi}"
        );
    }

    /// A do-while loop: the latch block itself ends in the exit test (`br header / merge`).
    /// `structured_plan` must split off a fresh unconditional continue block, rewrite the header's
    /// phi back-edge to come from it, and fully admit the function.
    #[test]
    fn do_while_latch_is_rotated_into_a_continue_block() {
        let blocks = vec![
            bb("%entry", &["br label %head"]),
            bb(
                "%head",
                &[
                    "%i = phi i32 [ 0, %entry ], [ %inc, %latch ]",
                    "br label %latch",
                ],
            ),
            bb(
                "%latch",
                &[
                    "%inc = add i32 %i, 1",
                    "br i1 %c, label %head, label %merge",
                ],
            ),
            bb("%merge", &["ret void"]),
        ];
        let plan = structured_plan(&blocks).expect("do-while fully structured");
        let info = plan.loop_merges.get("%head").expect("loop merge");
        assert_eq!(info.merge, "%merge");
        // Continue is a freshly synthesized block, NOT the original latch.
        assert!(
            info.continue_target.starts_with(CONT_PREFIX),
            "continue rotated into a synth block: {}",
            info.continue_target
        );
        let cont = &info.continue_target;
        // The synth continue block branches back to the header.
        let cb = plan
            .blocks
            .iter()
            .find(|b| &b.name == cont)
            .expect("continue block");
        assert_eq!(cb.lines(), vec!["br label %head".to_string()]);
        // The header phi now takes its back-edge incoming from the continue block, not the latch.
        let head = plan.blocks.iter().find(|b| b.name == "%head").unwrap();
        let phi = head
            .lines()
            .iter()
            .find(|l| l.contains("= phi "))
            .unwrap()
            .clone();
        assert!(
            phi.contains(&format!("{cont} ]")),
            "phi back-edge from continue: {phi}"
        );
        assert!(
            !phi.contains("%latch ]"),
            "phi no longer references the old latch: {phi}"
        );
    }

    /// An outer do-while loop whose latch (`%cont`) is the bottom exit-test (branches to the outer
    /// merge or back to the outer header) AND which sits after a nested inner loop. Historical private regression sets
    /// `cond-phi-shared/loop-role`-adjacent "block exits the continue, but not via a structured exit"
    /// failure: the latch is both the loop's continue target and a conditional, so it must be rotated
    /// (do-while normalization) into a clean unconditional continue before the selection layer treats
    /// it as a selection header. `structured_plan` must fully admit and rotate the continue.
    #[test]
    fn nested_do_while_latch_is_rotated_not_a_selection() {
        let blocks = vec![
            bb("%entry", &["br label %head"]),
            bb(
                "%head",
                &[
                    "%i = phi i32 [ 0, %entry ], [ %inc, %cont ]",
                    "br i1 %c0, label %body, label %cont",
                ],
            ),
            bb("%body", &["br label %inner"]),
            bb(
                "%inner",
                &[
                    "%j = phi i32 [ 0, %body ], [ %jinc, %inner ]",
                    "%jinc = add i32 %j, 1",
                    "%ic = icmp slt i32 %j, 5",
                    "br i1 %ic, label %inner, label %iexit",
                ],
            ),
            bb("%iexit", &["br label %cont"]),
            bb(
                "%cont",
                &[
                    "%inc = add i32 %i, 1",
                    "%ec = icmp eq i32 %inc, 3",
                    "br i1 %ec, label %merge, label %head",
                ],
            ),
            bb("%merge", &["ret void"]),
        ];
        let plan = structured_plan(&blocks).expect("nested do-while fully structured");
        let info = plan.loop_merges.get("%head").expect("outer loop merge");
        assert_eq!(info.merge, "%merge");
        // The conditional latch was rotated into a fresh unconditional continue block (not left as
        // both the continue and a selection header, which spirv-val rejects).
        assert!(
            info.continue_target.starts_with(CONT_PREFIX),
            "continue rotated: {}",
            info.continue_target
        );
    }

    /// An inner loop whose merge block IS the outer loop's continue target (the
    /// `MergeIsEnclosingContinue` restructure class) AND whose latch is a do-while bottom test
    /// (conditionally branches back to the inner header or out to that shared merge). The split that
    /// gives the inner loop a distinct merge must ALSO rotate the do-while latch into a clean
    /// unconditional continue — otherwise the latch stays both the continue and a conditional, which
    /// spirv-val rejects ("block exits the continue, but not via a structured exit"). This was the
    /// dominant structured-emit failure class; the rotation was missing from this branch.
    #[test]
    fn merge_is_enclosing_continue_do_while_latch_is_rotated() {
        let blocks = vec![
            bb("%entry", &["br label %outer"]),
            bb(
                "%outer",
                &[
                    "%i = phi i32 [ 0, %entry ], [ %ni, %ocont ]",
                    "br i1 %oc, label %ihead, label %odone",
                ],
            ),
            bb(
                "%ihead",
                &[
                    "%j = phi i32 [ 0, %outer ], [ %nj, %ilatch ]",
                    "br label %ilatch",
                ],
            ),
            bb(
                "%ilatch",
                &[
                    "%nj = add i32 %j, 1",
                    "%ic = icmp slt i32 %nj, 5",
                    "br i1 %ic, label %ihead, label %ocont",
                ],
            ),
            bb("%ocont", &["%ni = add i32 %i, 1", "br label %outer"]),
            bb("%odone", &["ret void"]),
        ];
        let plan =
            structured_plan(&blocks).expect("nested MergeIsEnclosingContinue do-while structured");
        // The inner loop got a distinct merge (split off the shared %ocont) AND its do-while latch was
        // rotated into a fresh unconditional continue (not left as the conditional %ilatch).
        let inner = plan.loop_merges.get("%ihead").expect("inner loop merge");
        assert_ne!(
            inner.merge, "%ocont",
            "inner merge split off the shared block"
        );
        assert!(
            inner.continue_target.starts_with(CONT_PREFIX),
            "inner do-while latch rotated to a clean continue: {}",
            inner.continue_target
        );
    }

    /// A single-exit loop with a mid-body conditional break: `body` branches to the loop merge
    /// (break) or the continue block. The break conditional is NOT a selection needing a post-dom
    /// merge — its OpSelectionMerge is the continue block — so `structured_plan` must fully admit the
    /// function (loop directly structurable + break recognized), recording the break in branch_merges.
    #[test]
    fn mid_body_break_is_recognized_not_a_selection() {
        let blocks = vec![
            bb("%entry", &["br label %head"]),
            bb(
                "%head",
                &[
                    "%i = phi i32 [ 0, %entry ], [ %inc, %cont ]",
                    "br i1 %c0, label %body, label %merge",
                ],
            ),
            bb("%body", &["br i1 %c1, label %merge, label %cont"]),
            bb("%cont", &["%inc = add i32 %i, 1", "br label %head"]),
            bb("%merge", &["ret void"]),
        ];
        let plan = structured_plan(&blocks).expect("function fully structured");
        // Loop directly structurable: merge=%merge, continue=%cont.
        let info = plan.loop_merges.get("%head").expect("loop merge");
        assert_eq!(info.merge, "%merge");
        assert_eq!(info.continue_target, "%cont");
        // The body break (arms = loop merge + continue) is recorded with the continue as its merge,
        // NOT synthesized as a fresh selection merge or rejected as cond-phi-shared.
        assert_eq!(
            plan.branch_merges
                .get(&("%merge".to_string(), "%cont".to_string())),
            Some(&"%cont".to_string())
        );
    }

    /// A loop whose merge block collides with an enclosing (out-of-loop) selection's post-dominator:
    /// `%entry` is an `if` whose two arms reconverge at `%merge`, and `%merge` is ALSO the single-exit
    /// loop `%head`'s merge (and carries a phi). One block cannot be both a loop merge and a selection
    /// merge, so the loop must be given a DISTINCT synthesized merge; then `%entry` keeps `%merge` and
    /// the function fully structures. This is the dominant `cond-phi-shared/loop-role` frontier shape.
    #[test]
    fn loop_merge_colliding_with_enclosing_selection_gets_distinct_merge() {
        let blocks = vec![
            bb("%entry", &["br i1 %c0, label %pre, label %merge"]),
            bb("%pre", &["br label %head"]),
            bb(
                "%head",
                &[
                    "%i = phi i32 [ 0, %pre ], [ %inc, %cont ]",
                    "br i1 %c1, label %body, label %merge",
                ],
            ),
            bb("%body", &["br i1 %c2, label %merge, label %cont"]),
            bb("%cont", &["%inc = add i32 %i, 1", "br label %head"]),
            bb(
                "%merge",
                &[
                    "%r = phi i32 [ 7, %entry ], [ %i, %head ], [ 9, %body ]",
                    "ret void",
                ],
            ),
        ];
        let plan =
            structured_plan(&blocks).expect("loop-merge/selection-merge collision structured");
        // The loop got a distinct synthesized merge, NOT %merge (which stays the selection's).
        let info = plan.loop_merges.get("%head").expect("loop merge");
        assert_ne!(
            info.merge, "%merge",
            "loop merge split off from the shared block"
        );
        assert!(
            info.merge.starts_with(SPLIT_PREFIX),
            "loop merge is synthesized: {}",
            info.merge
        );
        assert_eq!(info.continue_target, "%cont");
    }

    /// A loop that is BOTH multi-exit AND do-while (the header exits one way, the self-latch body
    /// exits the other and also back-edges): the two transforms must compose — multi-exit funnels the
    /// exits into a dispatch merge, then do-while rotation splits the body's back-edge into a continue
    /// block — so `structured_plan` fully admits the function.
    #[test]
    fn multi_exit_plus_do_while_compose_to_full_structure() {
        let blocks = vec![
            bb("%entry", &["br label %head"]),
            bb(
                "%head",
                &[
                    "%i = phi i32 [ 0, %entry ], [ %inc, %body ]",
                    "br i1 %c0, label %body, label %exitA",
                ],
            ),
            bb(
                "%body",
                &[
                    "%inc = add i32 %i, 1",
                    "br i1 %c1, label %exitB, label %head",
                ],
            ),
            bb("%exitA", &["br label %after"]),
            bb("%exitB", &["br label %after"]),
            bb("%after", &["ret void"]),
        ];
        let plan = structured_plan(&blocks).expect("multi-exit + do-while fully structured");
        let info = plan.loop_merges.get("%head").expect("loop merge");
        // Merge is the synthesized multi-exit dispatch; continue is the synthesized do-while block.
        assert!(
            info.merge.starts_with(SPLIT_PREFIX),
            "merge: {}",
            info.merge
        );
        assert!(
            info.continue_target.starts_with(CONT_PREFIX),
            "continue: {}",
            info.continue_target
        );
    }

    /// Residual merge-inloop / M2 shape: do-while with a mid-body guarded break that claims the loop
    /// merge. Converge gives the loop a distinct merge + rotates the latch to `{merge, continue}`;
    /// without structural-exit protection the subsequent selection synth would redirect the latch's
    /// break arm through a pass-through and reject `branch-no-merge`. With the latch pre-registered
    /// and protected, the plan admits (and the latch keeps its `{merge, continue}` arms).
    #[test]
    fn do_while_latch_survives_selection_merge_synth_claiming_loop_merge() {
        // entry -> h -> S; S -> body / m (guarded break); body -> latch; latch -> h / m (do-while).
        let blocks = vec![
            bb("%entry", &["br label %h"]),
            bb(
                "%h",
                &["%i = phi i32 [ 0, %entry ], [ %n, %latch ]", "br label %S"],
            ),
            bb("%S", &["br i1 %c, label %body, label %m"]),
            bb("%body", &["%n = add i32 %i, 1", "br label %latch"]),
            bb("%latch", &["br i1 %d, label %h, label %m"]),
            bb("%m", &["ret void"]),
        ];
        let plan = structured_plan(&blocks)
            .expect("do-while + guarded-break must structure with latch protection");
        let info = plan.loop_merges.get("%h").expect("loop merge for %h");
        // Latch was rotated: continue is a fresh cont block, not the original latch.
        assert!(
            info.continue_target.starts_with(CONT_PREFIX) || info.continue_target == "%latch",
            "continue: {}",
            info.continue_target
        );
        // The rotated latch (or original if not rotated) keeps a branch_merges entry keyed by its
        // CURRENT arms — never orphaned as (selN, cont).
        let latch = plan
            .blocks
            .iter()
            .find(|b| {
                conditional_branch_targets(b).is_some_and(|(t, f)| {
                    let arms = [t.as_str(), f.as_str()];
                    arms.contains(&info.merge.as_str())
                        && arms.contains(&info.continue_target.as_str())
                })
            })
            .expect("a break/continue latch with arms {{merge, continue}} must remain");
        let (t, f) = conditional_branch_targets(latch).unwrap();
        assert!(
            plan.branch_merges.contains_key(&(t.clone(), f.clone()))
                || plan.branch_merges.contains_key(&(f.clone(), t.clone())),
            "latch {} arms ({},{}) must be in branch_merges: {:?}",
            latch.name,
            t,
            f,
            plan.branch_merges.keys().collect::<Vec<_>>()
        );
    }

    /// A construct-tree candidate is already a reject-triggered re-nesting attempt. Its bottom-test
    /// latch may be rotated to `{loop-merge, synthetic-continue}`; keeping that as an OpSelectionMerge
    /// can assign the loop continue trampoline as the selection merge and fail dominance later. The
    /// construct-tree path must leave this loop-role branch bare instead.
    #[test]
    fn construct_tree_bottom_test_latch_uses_bare_loop_role_branch() {
        let blocks = vec![
            bb("%entry", &["br label %h"]),
            bb(
                "%h",
                &[
                    "%i = phi i32 [ 0, %entry ], [ %n, %latch ]",
                    "br label %latch",
                ],
            ),
            bb(
                "%latch",
                &["%n = add i32 %i, 1", "br i1 %d, label %h, label %m"],
            ),
            bb("%m", &["ret void"]),
        ];

        let plan = structured_plan_construct_tree(&blocks)
            .expect("construct-tree do-while latch should structure");
        let info = plan.loop_merges.get("%h").expect("loop merge for %h");
        assert!(
            info.continue_target.starts_with(CONT_PREFIX),
            "do-while latch should rotate to a synthetic continue: {}",
            info.continue_target
        );
        let latch = plan
            .blocks
            .iter()
            .find(|b| b.name == "%latch")
            .expect("rotated latch retained");
        let (t, f) = conditional_branch_targets(latch).expect("conditional latch");
        assert!(
            [t.as_str(), f.as_str()].contains(&info.merge.as_str())
                && [t.as_str(), f.as_str()].contains(&info.continue_target.as_str()),
            "latch arms should be loop roles, got ({t},{f}) with {info:?}"
        );
        assert!(
            !plan.branch_merges.contains_key(&(t.clone(), f.clone()))
                && !plan.branch_merges.contains_key(&(f.clone(), t.clone())),
            "construct-tree loop-role latch must be emitted bare, got branch merges {:?}",
            plan.branch_merges
        );
    }

    /// A loop header that carries a genuine in-loop conditional is split into an `lhsel` carrier so
    /// the loop header owns only `OpLoopMerge`. That carrier is an ordinary structured selection and
    /// must not be mistaken for a construct-tree bare enclosing-selection exit.
    #[test]
    fn construct_tree_loop_header_selection_carrier_keeps_selection_merge() {
        let blocks = vec![
            bb("%entry", &["br label %h"]),
            bb(
                "%h",
                &[
                    "%i = phi i32 [ 0, %entry ], [ %n, %cont ]",
                    "br i1 %c, label %a, label %b",
                ],
            ),
            bb("%a", &["br label %latch"]),
            bb("%b", &["br label %latch"]),
            bb(
                "%latch",
                &["%n = add i32 %i, 1", "br i1 %d, label %m, label %cont"],
            ),
            bb("%cont", &["br label %h"]),
            bb("%m", &["ret void"]),
        ];

        let plan = structured_plan_construct_tree(&blocks)
            .expect("construct-tree loop-header selection should structure");
        let lifted = plan
            .blocks
            .iter()
            .find(|block| block.name.starts_with("%metal2vulkan.lhsel."))
            .expect("loop-header conditional should be lifted");
        let (t, f) = conditional_branch_targets(lifted).expect("lifted conditional");
        assert!(
            plan.branch_merges_by_header.contains_key(&lifted.name)
                || plan.branch_merges.contains_key(&(t.clone(), f.clone())),
            "lifted loop-header selection must keep a merge, got header={:?} pair={:?}",
            plan.branch_merges_by_header.get(&lifted.name),
            plan.branch_merges.get(&(t, f))
        );
    }

    /// A construct-tree regional candidate can expose a mid-loop break where one arm leaves through the
    /// loop merge while the other continues into ordinary loop body work. The branch is already a legal
    /// SPIR-V structured loop exit; synthesizing a private selection merge for the break arm creates a
    /// nested selection whose body arm does not reconverge through that merge.
    #[test]
    fn construct_tree_mid_loop_break_to_body_stays_bare() {
        let blocks = vec![
            bb("%entry", &["br label %h"]),
            bb(
                "%h",
                &[
                    "%i = phi i32 [ 0, %entry ], [ %n, %cont ]",
                    "br label %break",
                ],
            ),
            bb("%break", &["br i1 %done, label %m, label %body"]),
            bb("%body", &["%n = add i32 %i, 1", "br label %cont"]),
            bb("%cont", &["br label %h"]),
            bb("%m", &["ret void"]),
        ];

        let plan = structured_plan_construct_tree(&blocks)
            .expect("construct-tree mid-loop break should structure");
        let info = plan.loop_merges.get("%h").expect("loop merge for %h");
        assert_eq!(info.merge, "%m");
        let branch = plan
            .blocks
            .iter()
            .find(|b| b.name == "%break")
            .expect("break block retained");
        let (t, f) = conditional_branch_targets(branch).expect("conditional break");
        assert_eq!((t.as_str(), f.as_str()), ("%m", "%body"));
        assert!(
            !plan.branch_merges.contains_key(&(t.clone(), f.clone()))
                && !plan.branch_merges.contains_key(&(f.clone(), t.clone())),
            "construct-tree mid-loop break must be emitted bare, got branch merges {:?}",
            plan.branch_merges
        );
    }

    /// If a loop merge is a synthetic pass-through to a function return, later finish/prune stages can
    /// collapse the loop merge target onto that return. A construct-tree selection must therefore treat
    /// the pass-through successor as a loop-role alias and give an enclosing guard a private merge, while
    /// leaving the loop merge pass-through itself on the original return.
    #[test]
    fn construct_tree_selection_splits_loop_merge_passthrough_alias() {
        let blocks = vec![
            bb("%entry", &["br i1 %skip, label %ret, label %h"]),
            bb("%h", &["br label %body"]),
            bb("%body", &["br label %cont"]),
            bb("%cont", &["br label %h"]),
            bb_role("%lm", BlockRole::LMerge, &["br label %ret"]),
            bb("%ret", &["ret void"]),
        ];
        let loop_merges = HashMap::from([(
            "%h".to_string(),
            LoopMergeInfo {
                merge: "%lm".to_string(),
                continue_target: "%cont".to_string(),
            },
        )]);

        let (out, branch, _branch_by_header, _) =
            unique_selection_merges_with_construct_tree_ownership(
                &blocks,
                &loop_merges,
                false,
                false,
                &HashMap::new(),
            );

        let entry = out.iter().find(|block| block.name == "%entry").unwrap();
        let (t, f) = conditional_branch_targets(entry).expect("entry remains conditional");
        let private = branch
            .get(&(t.clone(), f.clone()))
            .expect("entry guard receives a private selection merge");
        assert!(
            private.starts_with(format!("{SPLIT_PREFIX}{SEL_TOKEN}").as_str()),
            "private merge should be a selection split, got {private}"
        );
        let loop_merge = out.iter().find(|block| block.name == "%lm").unwrap();
        assert_eq!(
            block_successors(loop_merge),
            vec!["%ret".to_string()],
            "loop merge pass-through must not be redirected into the selection merge"
        );
    }

    /// A pre-existing synthetic merge block may be the natural postdominator for an enclosing
    /// construct-tree guard. It cannot be reused as that guard's selection merge, because it may already
    /// be owned by the transform that synthesized it and may later collapse onto another structured role.
    #[test]
    fn construct_tree_selection_splits_preexisting_synthetic_merge_natural() {
        let blocks = vec![
            bb("%entry", &["br i1 %c, label %merge, label %body"]),
            bb("%body", &["br label %merge"]),
            bb_role("%merge", BlockRole::LMerge, &["br label %ret"]),
            bb("%ret", &["ret void"]),
        ];

        let (out, branch, _branch_by_header, _) =
            unique_selection_merges_with_construct_tree_ownership(
                &blocks,
                &HashMap::new(),
                false,
                false,
                &HashMap::new(),
            );

        let entry = out.iter().find(|block| block.name == "%entry").unwrap();
        let (t, f) = conditional_branch_targets(entry).expect("entry remains conditional");
        let private = branch
            .get(&(t.clone(), f.clone()))
            .expect("entry guard receives a private selection merge");
        assert!(
            private.starts_with(format!("{SPLIT_PREFIX}{SEL_TOKEN}").as_str()),
            "private merge should be a selection split, got {private}"
        );
        let merge = out.iter().find(|block| block.name == "%merge").unwrap();
        assert_eq!(block_successors(merge), vec!["%ret".to_string()]);
    }

    /// A phi-carrying synthetic merge is a real data reconvergence point. Unlike a bare pass-through
    /// synthetic merge, construct-tree ownership must not split it solely because it is synthetic: doing
    /// so can pick one arm's pass-through as the nested merge while another arm reaches the real phi join.
    #[test]
    fn construct_tree_keeps_phi_carrying_synthetic_merge_natural() {
        let blocks = vec![
            bb("%entry", &["br i1 %c, label %a, label %b"]),
            bb("%a", &["br label %merge"]),
            bb("%b", &["br label %merge"]),
            bb_role(
                "%merge",
                BlockRole::LMerge,
                &["%x = phi i32 [ 0, %a ], [ 1, %b ]", "ret void"],
            ),
        ];

        let (_out, branch, branch_by_header, _) =
            unique_selection_merges_with_construct_tree_ownership(
                &blocks,
                &HashMap::new(),
                false,
                false,
                &HashMap::new(),
            );

        assert_eq!(
            branch_by_header.get("%entry").map(String::as_str),
            Some("%merge")
        );
        assert_eq!(
            branch
                .get(&("%a".to_string(), "%b".to_string()))
                .map(String::as_str),
            Some("%merge")
        );
    }

    /// If the selected merge is only a synthetic pass-through to a successor that another
    /// header-owned path reaches directly, the pass-through is not the real reconvergence point. Promote
    /// the header's merge to the shared successor.
    #[test]
    fn construct_tree_promotes_bypassed_passthrough_merge() {
        let blocks = vec![
            bb("%header", &["br i1 %c, label %pass, label %body"]),
            bb_role(
                "%pass",
                BlockRole::LMerge,
                &["%p = phi i32 [ 0, %header ]", "br label %join"],
            ),
            bb("%body", &["br label %route"]),
            bb_role("%route", BlockRole::ConstructTreeRoute, &["br label %join"]),
            bb_role(
                "%join",
                BlockRole::LMerge,
                &["%j = phi i32 [ %p, %pass ], [ 1, %route ]", "ret void"],
            ),
        ];
        let mut header_merges = HashMap::from([("%header".to_string(), "%pass".to_string())]);

        repair_construct_tree_passthrough_selection_merges(&blocks, &mut header_merges);

        assert_eq!(
            header_merges.get("%header").map(String::as_str),
            Some("%join")
        );
    }

    /// Construct-tree merge repair can leave only synthetic pass-throughs as the current predecessors
    /// of a shared no-phi merge. Those pass-throughs are still owned by the source selection and must be
    /// redirectable; construct-tree route gateways remain excluded by role.
    #[test]
    fn construct_tree_selection_splits_lmerge_predecessors() {
        let blocks = vec![
            bb("%entry", &["br i1 %outside, label %h, label %old"]),
            bb("%h", &["br i1 %c, label %a, label %b"]),
            bb_role("%a", BlockRole::LMerge, &["br label %old"]),
            bb_role("%b", BlockRole::LMerge, &["br label %old"]),
            bb("%old", &["ret void"]),
        ];

        let (out, branch, branch_by_header, _) =
            unique_selection_merges_with_construct_tree_ownership(
                &blocks,
                &HashMap::new(),
                false,
                false,
                &HashMap::new(),
            );

        let private = branch_by_header
            .get("%h")
            .expect("header receives a private merge through LMerge predecessors");
        assert!(
            private.starts_with(format!("{SPLIT_PREFIX}{SEL_TOKEN}").as_str()),
            "private merge should be a selection split, got {private}"
        );
        let h = out.iter().find(|block| block.name == "%h").unwrap();
        let (t, f) = conditional_branch_targets(h).expect("header remains conditional");
        assert_eq!(branch.get(&(t, f)), Some(private));
        for pred in ["%a", "%b"] {
            let block = out.iter().find(|block| block.name == pred).unwrap();
            assert_eq!(block_successors(block), vec![private.clone()]);
        }
        let private_block = out.iter().find(|block| &block.name == private).unwrap();
        let tail = branch_by_header
            .get("%entry")
            .cloned()
            .unwrap_or_else(|| "%old".to_string());
        assert_eq!(block_successors(private_block), vec![tail.clone()]);
        let tail_block = out.iter().find(|block| block.name == tail).unwrap();
        assert_eq!(block_successors(tail_block), vec!["%old".to_string()]);
    }

    /// A nested construct-tree branch may jump into the sibling arm of an enclosing selection. Vulkan
    /// still requires the nested conditional itself to be structured, so the construct-tree path gives
    /// it a private merge instead of emitting the branch bare.
    #[test]
    fn construct_tree_enclosing_selection_sibling_exit_gets_private_merge() {
        let blocks = vec![
            bb("%entry", &["br label %outer"]),
            bb("%outer", &["br i1 %a, label %sibling, label %inner"]),
            bb("%inner", &["br i1 %b, label %work, label %sibling"]),
            bb("%work", &["br label %inner_tail"]),
            bb("%inner_tail", &["br label %outer_merge"]),
            bb("%sibling", &["br label %outer_merge"]),
            bb("%outer_merge", &["ret void"]),
        ];

        let plan = structured_plan_construct_tree(&blocks)
            .expect("enclosing sibling-arm exit should structure");
        let inner = plan
            .blocks
            .iter()
            .find(|block| block.name == "%inner")
            .expect("inner block retained");
        let (t, f) = conditional_branch_targets(inner).expect("inner conditional");
        assert_eq!(t.as_str(), "%work");
        assert!(
            f == "%sibling" || f.ends_with("_sibling"),
            "sibling arm should be the original or a private clone, got {f}"
        );
        let private = plan
            .branch_merges_by_header
            .get("%inner")
            .expect("inner sibling-arm exit receives a private merge");
        assert!(
            private.starts_with(format!("{SPLIT_PREFIX}{SEL_TOKEN}").as_str()),
            "private merge should be a selection split, got {private}"
        );
        assert_eq!(
            plan.branch_merges.get(&(t, f)),
            Some(private),
            "pair map should mirror the header-specific merge"
        );
        assert!(
            plan.branch_merges_by_header.contains_key("%outer"),
            "enclosing selection still owns the region"
        );
    }

    /// A nested construct-tree branch can also exit through a dominated arm block into an enclosing
    /// selection continuation that is shared with the sibling arm. The branch target is not itself the
    /// sibling arm, so the construct-tree path must split the shared continuation and use that private
    /// split as the nested selection merge.
    #[test]
    fn construct_tree_deep_enclosing_selection_region_exit_gets_private_merge() {
        let blocks = vec![
            bb("%entry", &["br label %outer"]),
            bb("%outer", &["br i1 %a, label %sibling, label %gate"]),
            bb("%gate", &["br i1 %b, label %sibling, label %inner"]),
            bb("%inner", &["br i1 %c, label %exit, label %work"]),
            bb("%work", &["br label %inner_tail"]),
            bb("%inner_tail", &["br label %outer_merge"]),
            bb("%exit", &["br label %join"]),
            bb("%sibling", &["br label %join"]),
            bb("%join", &["br label %outer_merge"]),
            bb("%outer_merge", &["ret void"]),
        ];

        let plan = structured_plan_construct_tree(&blocks)
            .expect("deep enclosing-selection region exit should structure");
        let inner = plan
            .blocks
            .iter()
            .find(|block| block.name == "%inner")
            .expect("inner block retained");
        let (t, f) = conditional_branch_targets(inner).expect("inner conditional");
        assert_eq!((t.as_str(), f.as_str()), ("%exit", "%work"));
        let private = plan
            .branch_merges_by_header
            .get("%inner")
            .expect("deep enclosing-region exit receives a private merge");
        assert!(
            private.starts_with(format!("{SPLIT_PREFIX}{SEL_TOKEN}").as_str()),
            "private merge should be a selection split, got {private}"
        );
        assert_eq!(
            plan.branch_merges.get(&(t, f)),
            Some(private),
            "pair map should mirror the header-specific merge"
        );
        assert!(
            plan.branch_merges_by_header.contains_key("%outer"),
            "enclosing selection still owns the region"
        );
    }

    /// Construct-tree synthesis can temporarily assign a private selection merge, then a later route
    /// gateway or enclosing merge split can leave that merge with an outside predecessor. The final
    /// construct-tree-only repair must split the header-dominated edge(s) to a new private merge while
    /// preserving the outside predecessor's direct edge to the old merge.
    #[test]
    fn construct_tree_final_repair_splits_polluted_selection_merge() {
        let mut blocks = vec![
            bb("%entry", &["br i1 %c0, label %h, label %ext"]),
            bb("%h", &["br i1 %c1, label %body, label %old"]),
            bb("%body", &["br label %old"]),
            bb("%ext", &["br label %old"]),
            bb("%old", &["ret void"]),
        ];
        let mut header_merges = HashMap::from([("%h".to_string(), "%old".to_string())]);
        let mut counter = 0usize;

        repair_construct_tree_nondominated_selection_merges(
            &mut blocks,
            &HashMap::new(),
            &mut header_merges,
            &mut counter,
        );

        let private_merge = header_merges
            .get("%h")
            .expect("header merge retained after repair");
        assert!(
            private_merge.starts_with(SPLIT_PREFIX),
            "new private merge: {private_merge}"
        );
        let h = blocks.iter().find(|block| block.name == "%h").unwrap();
        let body = blocks.iter().find(|block| block.name == "%body").unwrap();
        let ext = blocks.iter().find(|block| block.name == "%ext").unwrap();
        assert!(
            block_successors(h)
                .iter()
                .any(|target| target == private_merge),
            "header direct old-merge arm is redirected to the private merge"
        );
        assert_eq!(
            block_successors(body),
            vec![private_merge.clone()],
            "dominated predecessor must route through the private merge"
        );
        assert_eq!(
            block_successors(ext),
            vec!["%old".to_string()],
            "outside predecessor must remain on the old merge"
        );
        let private = blocks
            .iter()
            .find(|block| &block.name == private_merge)
            .expect("private merge block inserted");
        assert_eq!(block_successors(private), vec!["%old".to_string()]);
        assert!(analyze(&blocks).dominates("%h", private_merge));
    }

    /// A construct-tree route gateway preserves an original inter-construct edge. The final repair
    /// splits ordinary header-owned predecessors, but must not drag that route into the header's private
    /// selection merge merely because the wrapper CFG makes it statically dominated by the header.
    #[test]
    fn construct_tree_final_repair_preserves_route_gateway_predecessor() {
        let mut blocks = vec![
            bb("%entry", &["br i1 %c0, label %h, label %ext"]),
            bb("%h", &["br i1 %c1, label %body, label %old"]),
            bb("%body", &["br label %route"]),
            bb_role("%route", BlockRole::ConstructTreeRoute, &["br label %old"]),
            bb("%ext", &["br label %old"]),
            bb("%old", &["ret void"]),
        ];
        let mut header_merges = HashMap::from([("%h".to_string(), "%old".to_string())]);
        let mut counter = 0usize;

        repair_construct_tree_nondominated_selection_merges(
            &mut blocks,
            &HashMap::new(),
            &mut header_merges,
            &mut counter,
        );

        let private_merge = header_merges
            .get("%h")
            .expect("header merge retained after repair");
        assert!(
            private_merge.starts_with(SPLIT_PREFIX),
            "new private merge: {private_merge}"
        );
        let h = blocks.iter().find(|block| block.name == "%h").unwrap();
        let route = blocks.iter().find(|block| block.name == "%route").unwrap();
        assert!(
            block_successors(h)
                .iter()
                .any(|target| target == private_merge),
            "header direct old-merge arm is redirected to the private merge"
        );
        assert_eq!(
            block_successors(route),
            vec!["%old".to_string()],
            "construct-tree route gateway must preserve its original edge"
        );
        assert!(analyze(&blocks).dominates("%h", private_merge));
    }

    /// Frontier straddle residual after regional wrap (01/1716c0e9 shape): two selections share a
    /// natural merge whose only header-dominated predecessors were rewritten into ConstructTreeRoute
    /// gateways. Arms reconverge at the natural merge *through different routes*, so the post-dominator
    /// is the shared natural (not a single route). Construct-tree ownership must reclaim the Normal
    /// edges into those routes onto private merges while leaving each route edge intact.
    #[test]
    fn construct_tree_reclaims_route_entry_preds_for_shared_natural_merge() {
        let blocks = vec![
            bb("%entry", &["br i1 %c0, label %h0, label %h1"]),
            bb("%h0", &["br i1 %c1, label %a0, label %b0"]),
            bb("%a0", &["br label %r0"]),
            bb("%b0", &["br label %r1"]),
            bb_role("%r0", BlockRole::ConstructTreeRoute, &["br label %old"]),
            bb_role("%r1", BlockRole::ConstructTreeRoute, &["br label %old"]),
            bb("%h1", &["br i1 %c2, label %a1, label %b1"]),
            bb("%a1", &["br label %r2"]),
            bb("%b1", &["br label %r3"]),
            bb_role("%r2", BlockRole::ConstructTreeRoute, &["br label %old"]),
            bb_role("%r3", BlockRole::ConstructTreeRoute, &["br label %old"]),
            bb("%old", &["ret void"]),
        ];

        let plan = structured_plan_construct_tree(&blocks)
            .expect("route-reclaim ownership should structure the post-wrapper residual");
        let h0_merge = plan
            .branch_merges_by_header
            .get("%h0")
            .expect("h0 selection merge");
        let h1_merge = plan
            .branch_merges_by_header
            .get("%h1")
            .expect("h1 selection merge");
        assert_ne!(h0_merge, "%old", "h0 must not share the raw natural merge");
        assert_ne!(h1_merge, "%old", "h1 must not share the raw natural merge");
        assert_ne!(h0_merge, h1_merge, "each header gets a private merge");
        assert!(
            h0_merge.starts_with(SPLIT_PREFIX) && h1_merge.starts_with(SPLIT_PREFIX),
            "private merges are synthesized: {h0_merge}, {h1_merge}"
        );

        for (pred, private) in [
            ("%a0", h0_merge),
            ("%b0", h0_merge),
            ("%a1", h1_merge),
            ("%b1", h1_merge),
        ] {
            let block = plan
                .blocks
                .iter()
                .find(|block| block.name == pred)
                .unwrap_or_else(|| panic!("missing {pred}"));
            assert_eq!(
                block_successors(block),
                vec![private.clone()],
                "{pred} reclaimed onto private merge {private}"
            );
        }
        for route in ["%r0", "%r1", "%r2", "%r3"] {
            let block = plan
                .blocks
                .iter()
                .find(|block| block.name == route)
                .unwrap_or_else(|| panic!("missing {route}"));
            assert_eq!(
                block_successors(block),
                vec!["%old".to_string()],
                "route gateway {route} must keep its original edge to the natural merge"
            );
        }
    }

    /// A pre-registered break/continue latch can be rewritten later by an enclosing selection's
    /// unique-merge synthesis: the break arm (`%m`) becomes a synthetic LMerge pass-through while the
    /// continue arm stays the loop continue. The branch merge map must be keyed by those FINAL arms, not
    /// the stale pre-synthesis `{%m,%cont}` pair.
    #[test]
    fn break_continue_latch_rekeys_after_enclosing_selection_synth() {
        let blocks = vec![
            bb("%entry", &["br label %h"]),
            bb(
                "%h",
                &["%i = phi i32 [ 0, %entry ], [ %n, %cont ]", "br label %S"],
            ),
            bb("%S", &["br i1 %c, label %body, label %m"]),
            bb("%body", &["br label %latch"]),
            bb("%latch", &["br i1 %d, label %m, label %cont"]),
            bb("%cont", &["%n = add i32 %i, 1", "br label %h"]),
            bb("%m", &["ret void"]),
        ];

        let plan = structured_plan(&blocks)
            .expect("rewritten break/continue latch must retain a final branch merge");
        let info = plan.loop_merges.get("%h").expect("loop merge for %h");
        let latch = plan
            .blocks
            .iter()
            .find(|b| b.name == "%latch")
            .expect("latch retained");
        let (t, f) = conditional_branch_targets(latch).expect("conditional latch");

        assert!(
            t.starts_with(SPLIT_PREFIX) || f.starts_with(SPLIT_PREFIX),
            "one rewritten latch arm must be a synthesized merge, got ({t},{f})"
        );
        assert!(
            [t.as_str(), f.as_str()].contains(&info.continue_target.as_str()),
            "one rewritten latch arm must remain the loop continue, got ({t},{f}), continue={}",
            info.continue_target
        );
        assert!(
            plan.branch_merges.contains_key(&(t.clone(), f.clone())),
            "branch_merges must use the final latch arms ({t},{f}), got {:?}",
            plan.branch_merges.keys().collect::<Vec<_>>()
        );
    }

    /// Multi-exit whose one real exit is ALSO reached from outside the loop (M2 shape): the dispatch
    /// CLONES that exit's dominated forward region so the
    /// dispatch dominates a private arm, not the shared exit — otherwise `selection:cross-arm-shared` on
    /// the dispatch merge (a bare trampoline pass-through does not fix it: dead-end #6).
    #[test]
    fn multi_exit_shared_outer_target_gets_region_clone() {
        // entry -> guard; guard -> head / sharedExit (outer edge); head -> body / exitA;
        // body -> sharedExit / latch; latch -> head. sharedExit is both a loop exit and the outer arm.
        let blocks = vec![
            bb("%entry", &["br label %guard"]),
            bb("%guard", &["br i1 %g, label %head, label %sharedExit"]),
            bb("%head", &["br i1 %c0, label %body, label %exitA"]),
            bb("%body", &["br i1 %c1, label %sharedExit, label %latch"]),
            bb("%latch", &["br label %head"]),
            bb("%exitA", &["br label %after"]),
            bb("%sharedExit", &["br label %after"]),
            bb("%after", &["ret void"]),
        ];
        // The clone is now a plan PARAMETER (not the env flag): pass multi_exit_clone = true.
        let (out, merges) = forest_loop_merges(&blocks, false, true);
        let info = merges.get("%head").expect("loop merge");
        assert!(
            info.merge.starts_with(SPLIT_PREFIX),
            "merge: {}",
            info.merge
        );
        // A private CLONE of the shared exit exists (distinct name, same `br label %after` body) and
        // the dispatch targets it, not the shared exit directly.
        let clone = out
            .iter()
            .find(|b| b.name != "%sharedExit" && b.name.contains("sharedExit"))
            .expect("shared exit must be region-cloned into a private arm under the flag");
        assert_eq!(block_successors(clone), vec!["%after".to_string()]);
        // The original shared exit stays, still reached from the outer `guard` edge.
        assert!(out.iter().any(|b| b.name == "%sharedExit"));
        let dispatch = out
            .iter()
            .find(|b| b.name == info.merge)
            .expect("dispatch merge block");
        let succ = block_successors(dispatch);
        assert!(
            succ.contains(&clone.name),
            "dispatch must branch to the private clone, not bare shared exit: {succ:?}"
        );
        assert!(
            !succ.iter().any(|s| s == "%sharedExit"),
            "dispatch must not arm-target the shared exit directly: {succ:?}"
        );
    }

    /// A loop with two distinct exit targets (the `MultipleExits` class). Expect a single
    /// synthesized dispatch merge: an `i1` selector phi over the redirected exit predecessors plus a
    /// conditional branch back out to the two real targets, with the loop's merge pointing at it.
    #[test]
    fn multiple_exits_funnel_into_one_dispatch_merge() {
        let blocks = vec![
            bb("%entry", &["br label %head"]),
            bb("%head", &["br i1 %c0, label %body, label %exitA"]),
            bb("%body", &["br i1 %c1, label %exitB, label %latch"]),
            bb("%latch", &["br label %head"]),
            bb("%exitA", &["ret void"]),
            bb("%exitB", &["ret void"]),
        ];
        let (out, merges) = forest_loop_merges(&blocks, false, false);

        let synth: Vec<&BodyBlock> = out
            .iter()
            .filter(|b| b.name.starts_with(SPLIT_PREFIX))
            .collect();
        assert_eq!(synth.len(), 1, "one dispatch merge synthesized");
        let m = synth[0];
        // i1 selector phi over the two redirected exit predecessors (sorted: exitA<exitB => head=true).
        let selector = &carrier_phis(m)[0].1;
        assert!(
            selector
                .iter()
                .any(|(v, p)| val_str(v) == "true" && p == "%head"),
            "head exits to exitA (idx 0 = true): {selector:?}"
        );
        assert!(
            selector
                .iter()
                .any(|(v, p)| val_str(v) == "false" && p == "%body"),
            "body exits to exitB (idx 1 = false): {selector:?}"
        );
        // Conditional branch back out to the two real exit targets.
        match &m.typed.as_ref().unwrap().terminator {
            crate::native::tir::TirTerminator::BrCond { cond, t, f } => {
                assert!(cond.starts_with(EXIT_SEL_PREFIX));
                assert_eq!(t, "%exitA");
                assert_eq!(f, "%exitB");
            }
            other => panic!("dispatch must end in a conditional branch: {other:?}"),
        }

        // The loop is now single-exit: merge = M, continue = the unconditional latch.
        let info = merges.get("%head").expect("head loop merge recorded");
        assert_eq!(info.merge, m.name);
        assert_eq!(info.continue_target, "%latch");

        // The in-loop exit edges were redirected to the dispatch merge (no longer to exitA/exitB).
        let head = out.iter().find(|b| b.name == "%head").unwrap();
        let body = out.iter().find(|b| b.name == "%body").unwrap();
        assert!(
            block_successors(head).contains(&m.name)
                && !block_successors(head).iter().any(|s| s == "%exitA")
        );
        assert!(
            block_successors(body).contains(&m.name)
                && !block_successors(body).iter().any(|s| s == "%exitB")
        );
    }

    /// A multi-exit loop whose BOTH exit targets carry a phi fed by the in-loop exit predecessor
    /// (`loop:MultipleExits[k=2,phi=1]`). The dispatch merge `M` must funnel each exit phi through a
    /// fresh value phi covering ALL of `M`'s predecessors — the real incoming for the predecessor
    /// targeting that exit, `undef` for the one targeting the other exit — and rewrite each exit phi to
    /// take the merged value via the single `M` edge.
    #[test]
    fn multiple_exits_with_phi_exits_funnel_through_value_phis() {
        let blocks = vec![
            bb("%entry", &["br label %head"]),
            bb("%head", &["br i1 %c0, label %body, label %exitA"]),
            bb("%body", &["br i1 %c1, label %exitB, label %latch"]),
            bb("%latch", &["br label %head"]),
            bb("%exitA", &["%pa = phi i32 [ 10, %head ]", "ret void"]),
            bb("%exitB", &["%pb = phi i32 [ 20, %body ]", "ret void"]),
        ];
        let (out, merges) = forest_loop_merges(&blocks, false, false);

        let synth: Vec<&BodyBlock> = out
            .iter()
            .filter(|b| b.name.starts_with(SPLIT_PREFIX))
            .collect();
        assert_eq!(synth.len(), 1, "one dispatch merge synthesized");
        let m = synth[0];
        // The loop is now single-exit (merge = M), so MultipleExits is resolved.
        assert_eq!(merges.get("%head").expect("head merge").merge, m.name);

        // Two value phis in M: exitA's (real from %head, undef from %body) and exitB's (undef from
        // %head, real from %body) — each covering BOTH of M's predecessors. Value phis carry an `undef`
        // slot; the selector phi carries only true/false, so `undef` distinguishes them.
        let phis = carrier_phis(m);
        let value_phi_count = phis
            .iter()
            .filter(|(_, inc)| inc.iter().any(|(v, _)| val_str(v) == "undef"))
            .count();
        assert_eq!(
            value_phi_count, 2,
            "one value phi per phi-carrying exit: {phis:?}"
        );
        assert!(
            phis.iter().any(
                |(_, inc)| inc.iter().any(|(v, p)| val_str(v) == "10" && p == "%head")
                    && inc
                        .iter()
                        .any(|(v, p)| val_str(v) == "undef" && p == "%body")
            ),
            "exitA value phi funnels head's 10, undef for the body edge: {phis:?}"
        );
        assert!(
            phis.iter().any(
                |(_, inc)| inc.iter().any(|(v, p)| val_str(v) == "20" && p == "%body")
                    && inc
                        .iter()
                        .any(|(v, p)| val_str(v) == "undef" && p == "%head")
            ),
            "exitB value phi funnels body's 20, undef for the head edge: {phis:?}"
        );

        // Each exit phi now takes a single merged incoming via the M edge, no longer the in-loop pred.
        let exit_a = out.iter().find(|b| b.name == "%exitA").unwrap();
        let pa = phi_incomings(exit_a, "%pa");
        assert!(
            pa.iter().any(|(_, p)| *p == m.name) && !pa.iter().any(|(_, p)| p == "%head"),
            "exitA phi rebuilt via the merge: {pa:?}"
        );
        let exit_b = out.iter().find(|b| b.name == "%exitB").unwrap();
        let pb = phi_incomings(exit_b, "%pb");
        assert!(
            pb.iter().any(|(_, p)| *p == m.name) && !pb.iter().any(|(_, p)| p == "%body"),
            "exitB phi rebuilt via the merge: {pb:?}"
        );
    }

    /// A two-exit loop may leave through two distinct exit edges of one switch/branch block. Its
    /// predecessor cannot be used directly as the dispatch selector's phi source because it would need
    /// two selector values. Split those critical edges first, preserving each exit phi's predecessor,
    /// then funnel the now-unambiguous edge blocks through the ordinary multi-exit dispatch merge.
    #[test]
    fn multiple_exits_split_a_direct_two_exit_critical_edge() {
        let blocks = vec![
            bb("%entry", &["br label %head"]),
            bb("%head", &["br label %body"]),
            bb(
                "%body",
                &[
                    "%v = add i32 %x, 1",
                    "switch i32 %which, label %exitB [ i32 0, label %latch i32 1, label %exitA ]",
                ],
            ),
            bb("%latch", &["br label %head"]),
            bb("%exitA", &["%pa = phi i32 [ %v, %body ]", "ret void"]),
            bb("%exitB", &["%pb = phi i32 [ %v, %body ]", "ret void"]),
        ];

        let (out, merges) = forest_loop_merges(&blocks, false, false);
        let info = merges.get("%head").expect("multi-exit loop was funnelled");
        assert!(
            info.merge.starts_with(SPLIT_PREFIX),
            "dispatch merge: {}",
            info.merge
        );

        let body = out.iter().find(|b| b.name == "%body").unwrap();
        let body_succ = block_successors(body);
        assert!(
            body_succ.iter().any(|s| s.starts_with(EXIT_EDGE_PREFIX))
                && !body_succ.iter().any(|s| s == "%exitA" || s == "%exitB"),
            "body must branch through distinct critical-edge blocks: {body_succ:?}"
        );
        let edge_blocks: Vec<&BodyBlock> = out
            .iter()
            .filter(|b| b.name.starts_with(EXIT_EDGE_PREFIX))
            .collect();
        assert_eq!(edge_blocks.len(), 2, "one edge block per direct exit arm");

        let dispatch = out.iter().find(|b| b.name == info.merge).unwrap();
        // The i1 selector phi carries only true/false; its preds are the split edge blocks.
        let selector = carrier_phis(dispatch)
            .into_iter()
            .find(|(_, inc)| {
                inc.iter()
                    .all(|(v, _)| matches!(val_str(v).as_str(), "true" | "false"))
            })
            .expect("dispatch selector phi");
        for edge in &edge_blocks {
            assert!(
                selector.1.iter().any(|(_, p)| *p == edge.name),
                "selector must distinguish edge {}: {:?}",
                edge.name,
                selector.1
            );
        }
        for exit in ["%exitA", "%exitB"] {
            let block = out.iter().find(|b| b.name == exit).unwrap();
            let phi = &carrier_phis(block)[0].1;
            assert!(
                phi.iter().any(|(_, p)| *p == info.merge) && !phi.iter().any(|(_, p)| p == "%body"),
                "exit phi must be funnelled through the dispatch: {phi:?}"
            );
        }
    }

    /// A phi-carrying merge==continue overlap: the shared block `%latch` (inner exit == outer
    /// continue) carries a phi fed by both inner predecessors (`%inner`, `%ibody`). Expect a
    /// synthesized pass-through carrying a merged phi over those two incomings, and `%latch`'s phi
    /// rewritten to take the merged value via the pass-through edge.
    #[test]
    fn merge_is_enclosing_continue_with_phi_merges() {
        let blocks = vec![
            bb("%entry", &["br label %outer"]),
            bb("%outer", &["br i1 %co, label %inner, label %done"]),
            bb("%inner", &["br i1 %ci, label %ibody, label %latch"]),
            bb("%ibody", &["br i1 %ci2, label %inner, label %latch"]),
            bb(
                "%latch",
                &[
                    "%p = phi i32 [ 0, %inner ], [ 1, %ibody ]",
                    "br label %outer",
                ],
            ),
            bb("%done", &["ret void"]),
        ];
        let (out, merges) = forest_loop_merges(&blocks, false, false);
        let synth: Vec<&BodyBlock> = out
            .iter()
            .filter(|b| b.name.starts_with(SPLIT_PREFIX))
            .collect();
        assert_eq!(synth.len(), 1, "one pass-through merge inserted");
        let pt = synth[0];
        // The pass-through carries a merged phi over both inner predecessors, then branches to %latch —
        // read from the typed carrier (the emission substrate; carrier-first, ahead of the writer migration).
        let ptc = pt.typed.as_ref().expect("pass-through carrier");
        let merged_phi = ptc
            .insts
            .iter()
            .find(|i| i.is_phi())
            .and_then(|i| i.phi_incoming.as_ref())
            .expect("merged phi in pass-through");
        assert!(
            merged_phi.1.iter().any(|(_, p)| p == "%inner"),
            "merged phi keeps %inner incoming"
        );
        assert!(
            merged_phi.1.iter().any(|(_, p)| p == "%ibody"),
            "merged phi keeps %ibody incoming"
        );
        assert!(block_successors(pt).iter().any(|s| s == "%latch"));

        // %latch's phi now takes the merged value via the pass-through, not the inner preds directly.
        let latch = out.iter().find(|b| b.name == "%latch").unwrap();
        let latch_phi = latch
            .typed
            .as_ref()
            .unwrap()
            .insts
            .iter()
            .find(|i| i.is_phi())
            .and_then(|i| i.phi_incoming.as_ref())
            .unwrap();
        assert!(
            latch_phi.1.iter().any(|(_, p)| *p == pt.name),
            "latch phi takes the pass-through edge: {:?}",
            latch_phi.1
        );
        assert!(
            !latch_phi
                .1
                .iter()
                .any(|(_, p)| p == "%inner" || p == "%ibody"),
            "latch phi no longer references the redirected inner preds: {:?}",
            latch_phi.1
        );
        // The inner loop's merge is the pass-through; both inner preds were redirected to it.
        assert_eq!(
            merges.get("%inner").map(|m| m.merge.as_str()),
            Some(pt.name.as_str())
        );
        for p in ["%inner", "%ibody"] {
            let blk = out.iter().find(|b| b.name == p).unwrap();
            assert!(
                block_successors(blk).contains(&pt.name),
                "{p} redirected to pass-through"
            );
        }
        // Outer continue preserved.
        assert!(block_successors(latch).iter().any(|s| s == "%outer"));
    }
}
