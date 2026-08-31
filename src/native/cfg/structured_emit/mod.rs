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
//! The planner is wired into the primary structured-emission path. Its complete structural
//! alternatives are attempted before the owned pipeline selects the raw-CFG representation;
//! validator output does not participate in that choice.

use super::blocks::{
    block_successors, conditional_branch_targets, index_branch_merges_by_header, synthetic_block,
};
use super::graph::block_dominators;
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
mod rewrite;
pub(in crate::native) use rewrite::*;
// The own-arm and loop-exit-sibling paths consume the construct-tree planner core. The bounded
// dispatcher materializers are also the typed source-level fallback for modest rejected CFGs.
#[allow(dead_code)]
mod construct_tree;
pub(in crate::native) use construct_tree::{
    renest_whole_cfg_dispatch, requires_loop_exit_sibling_dispatch,
};
mod own_arm;
pub(in crate::native) use own_arm::*;
mod straddle_region;
pub(in crate::native) use straddle_region::*;

#[cfg(test)]
mod construct_tree_fixtures;
#[cfg(test)]
mod test_support;

#[cfg(test)]
mod tests {
    use super::test_support::{bb, bb_role};
    use super::*;

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
            .and_then(|i| i.phi_incoming().as_ref())
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
                    i.phi_incoming().as_ref().map(|(_, inc)| inc.clone())?,
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
        assert!(
            terminal.merges.contains_key("%outer"),
            "the enclosing terminal owner must also reach the fixed point"
        );
        assert!(terminal.merges.contains_key("%guard"));
    }

    /// A private linear return tail can safely use a disconnected unreachable merge when every path
    /// below the selection terminates. No source return needs to be redirected or cloned.
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
        let merge = terminal.merges.get("%guard").expect("guard merge");
        assert!(terminal
            .blocks
            .iter()
            .find(|block| block.name == *merge)
            .is_some_and(is_bare_unreachable));
    }

    /// The exact shared-return edge split is linear and must not disappear merely because the two
    /// arms contain more blocks than the general terminal-search retry budget.
    #[test]
    fn large_two_arm_shared_return_gets_a_private_terminal_merge() {
        let mut blocks = vec![
            bb("%entry", &["br i1 %outside, label %ret, label %header"]),
            bb("%header", &["br i1 %choose, label %left0, label %right0"]),
        ];
        for arm in ["left", "right"] {
            for index in 0..160 {
                let name = format!("%{arm}{index}");
                let target = if index == 159 {
                    "%ret".to_string()
                } else {
                    format!("%{arm}{}", index + 1)
                };
                blocks.push(bb(&name, &[&format!("br label {target}")]));
            }
        }
        blocks.push(bb("%ret", &["ret void"]));
        assert!(blocks.len() > TERMINAL_EXIT_SELECTION_MAX_BLOCKS);

        let terminal = terminal_exit_selection_merges(&blocks).expect("large shared return splits");
        let private_merge = terminal
            .merges
            .get("%header")
            .expect("header receives a private terminal merge");
        assert!(terminal.blocks.iter().any(|block| {
            block.name == *private_merge
                && block.role == BlockRole::LMerge
                && block.lines() == ["unreachable"]
        }));
        assert!(
            terminal.blocks.iter().any(|block| block.name == "%ret"),
            "the source return remains available while each structured owner gets a private exit"
        );
        for tail in ["%left159", "%right159"] {
            assert_eq!(
                block_successors(
                    terminal
                        .blocks
                        .iter()
                        .find(|block| block.name == tail)
                        .unwrap()
                ),
                vec!["%ret".to_string()]
            );
        }

        let plan = structured_plan_inner6(&blocks, false, false, false, false, false, true)
            .expect("large shared return enters the terminal planner");
        let planned_header = plan
            .blocks
            .iter()
            .find(|block| block.name == "%header")
            .unwrap();
        let planned_arms = conditional_branch_targets(planned_header).unwrap();
        assert!(
            plan.branch_merges.contains_key(&planned_arms),
            "the large-CFG caller must not size-gate the exact terminal split: {:?}",
            plan.branch_merges
        );
    }

    /// An inner guard reaches the same shared return as its enclosing sibling. Splitting the inner
    /// guard first would make the outer arms disagree about their return target, so the shared-return
    /// phase must establish the outer owner before refining the nested owner.
    #[test]
    fn nested_shared_returns_are_claimed_outermost_first() {
        let blocks = vec![
            bb("%entry", &["br i1 %outside, label %ret, label %outer"]),
            bb("%outer", &["br i1 %co, label %left, label %inner"]),
            bb("%left", &["br label %ret"]),
            bb(
                "%inner",
                &["br i1 %ci, label %inner_left, label %inner_right"],
            ),
            bb("%inner_left", &["br label %ret"]),
            bb("%inner_right", &["br label %ret"]),
            bb("%ret", &["ret void"]),
        ];

        let terminal = terminal_exit_selection_merges(&blocks).expect("nested returns split");
        let outer_return = terminal.merges.get("%outer").expect("outer return owner");
        let inner_return = terminal.merges.get("%inner").expect("inner return owner");
        assert_ne!(outer_return, inner_return);
        assert!(terminal
            .blocks
            .iter()
            .any(|block| { block.name == *outer_return && block.role == BlockRole::LMerge }));
        assert!(terminal
            .blocks
            .iter()
            .any(|block| { block.name == *inner_return && block.role == BlockRole::LMerge }));
    }

    /// A previously structured owner may have replaced one arm's source return with an equivalent
    /// private return. A containing selection can still funnel both bare returns into its own exit;
    /// return-block identity is not a semantic distinction for `ret void`.
    #[test]
    fn distinct_void_returns_share_one_private_selection_exit() {
        let blocks = vec![
            bb("%entry", &["br i1 %outside, label %ret_a, label %header"]),
            bb("%header", &["br i1 %choose, label %left, label %right"]),
            bb("%left", &["br label %ret_a"]),
            bb("%right", &["br label %ret_b"]),
            bb("%ret_a", &["ret void"]),
            bb("%ret_b", &["ret void"]),
            bb("%dead", &["br label %ret_a"]),
        ];

        let terminal = terminal_exit_selection_merges(&blocks).expect("void returns split");
        let private_merge = terminal
            .merges
            .get("%header")
            .expect("header receives one private merge");
        assert!(terminal.blocks.iter().any(|block| {
            block.name == *private_merge
                && block.role == BlockRole::LMerge
                && block.lines() == ["unreachable"]
        }));
        assert_eq!(
            block_successors(
                terminal
                    .blocks
                    .iter()
                    .find(|block| block.name == "%left")
                    .unwrap()
            ),
            vec!["%ret_a".to_string()]
        );
        assert_eq!(
            block_successors(
                terminal
                    .blocks
                    .iter()
                    .find(|block| block.name == "%right")
                    .unwrap()
            ),
            vec!["%ret_b".to_string()]
        );
        assert_eq!(
            block_successors(
                terminal
                    .blocks
                    .iter()
                    .find(|block| block.name == "%dead")
                    .unwrap()
            )
            .first()
            .map(String::as_str),
            Some("%ret_a"),
            "the unrelated predecessor keeps the source return"
        );
    }

    /// Generated shaders routinely contain more than a small hand-chosen number of terminal guards.
    /// The planner reaches a structural fixed point, so depth must not be constrained by a retry cap.
    #[test]
    fn terminal_selection_planning_has_no_arbitrary_guard_limit() {
        let mut blocks = vec![
            bb("%entry", &["br i1 %outside, label %ret, label %guard0"]),
            bb("%ret", &["ret void"]),
        ];
        for index in 0..24 {
            let header = format!("%guard{index}");
            let continuation = if index == 23 {
                "%tail".to_string()
            } else {
                format!("%guard{}", index + 1)
            };
            blocks.push(bb(
                &header,
                &[&format!(
                    "br i1 %condition{index}, label %ret, label {continuation}"
                )],
            ));
        }
        blocks.push(bb("%tail", &["ret void"]));

        let terminal = terminal_exit_selection_merges(&blocks).expect("terminal chain splits");
        for index in 0..24 {
            assert!(
                terminal.merges.contains_key(&format!("%guard{index}")),
                "guard {index} must not be dropped by a numeric retry cap"
            );
        }
    }

    /// A terminal subtree can be paired with a continuation that has an outside predecessor. The
    /// enclosing header owns only its edge into that continuation and therefore needs a private
    /// pass-through merge rather than ownership of the shared block itself.
    #[test]
    fn owned_terminal_arm_splits_shared_sibling_continuation() {
        let blocks = vec![
            bb("%entry", &["br i1 %outside, label %shared, label %outer"]),
            bb("%outer", &["br i1 %co, label %shared, label %inner"]),
            bb(
                "%inner",
                &["br i1 %ci, label %inner_left, label %inner_right"],
            ),
            bb("%inner_left", &["br label %ret"]),
            bb("%inner_right", &["br label %ret"]),
            bb("%shared", &["call void @side_effect()", "br label %ret"]),
            bb("%ret", &["ret void"]),
        ];

        let terminal = terminal_exit_selection_merges(&blocks).expect("terminal ownership closes");
        let outer_merge = terminal.merges.get("%outer").expect("outer merge");
        let merge_block = terminal
            .blocks
            .iter()
            .find(|block| block.name == *outer_merge)
            .expect("private continuation split");
        assert_eq!(block_successors(merge_block), vec!["%shared".to_string()]);
        assert_eq!(
            block_successors(
                terminal
                    .blocks
                    .iter()
                    .find(|block| block.name == "%entry")
                    .unwrap()
            )
            .first()
            .map(String::as_str),
            Some("%shared"),
            "the outside predecessor retains the shared continuation"
        );
    }

    /// A loop exit target can differ from the final synthesized merge label. It is still outside the
    /// natural loop and must not be lifted into an `lhsel` selection inside that loop.
    #[test]
    fn loop_header_exit_arm_is_not_lifted_as_an_in_loop_selection() {
        let mut blocks = vec![
            bb(
                "%header",
                &["br i1 %condition, label %body, label %exit_work"],
            ),
            bb("%body", &["br label %continue"]),
            bb("%continue", &["br label %header"]),
            bb(
                "%exit_work",
                &["call void @side_effect()", "br label %merge"],
            ),
            bb("%merge", &["ret void"]),
        ];
        let loop_body = ["%header", "%body", "%continue"]
            .into_iter()
            .map(str::to_string)
            .collect::<HashSet<_>>();
        assert_eq!(
            split_loop_header_selection(
                &mut blocks,
                "%header",
                "%merge",
                "%continue",
                &loop_body,
                &mut 0,
            ),
            None
        );
        assert_eq!(
            conditional_branch_targets(
                blocks.iter().find(|block| block.name == "%header").unwrap()
            ),
            Some(("%body".to_string(), "%exit_work".to_string()))
        );
    }

    /// A lifted loop-entry conditional still has a real merge when one nested path returns before
    /// reaching it. The return is a legal structured exit and must not erase the non-terminal merge.
    #[test]
    fn loop_entry_selection_ignores_terminal_path_for_merge_proof() {
        let blocks = vec![
            bb("%entry", &["br label %header"]),
            bb("%header", &["br label %selection"]),
            bb(
                "%selection",
                &["br i1 %choose, label %body, label %converge"],
            ),
            bb("%body", &["br i1 %stop, label %ret, label %converge"]),
            bb("%converge", &["br label %continue"]),
            bb("%continue", &["br label %header"]),
            bb("%ret", &["ret void"]),
            bb("%exit", &["unreachable"]),
        ];
        let forest = analyze(&blocks);
        let loop_merges = HashMap::from([(
            "%header".to_string(),
            LoopMergeInfo {
                merge: "%exit".to_string(),
                continue_target: "%continue".to_string(),
            },
        )]);
        let mut selections = HashMap::new();
        refine_loop_entry_terminal_selection_merges(
            &blocks,
            &forest,
            &loop_merges,
            &mut selections,
        );
        assert_eq!(selections.get("%selection"), Some(&"%converge".to_string()));
    }

    /// The live convergence may be below both direct arms; terminal paths from either arm do not
    /// erase that loop-local merge.
    #[test]
    fn loop_entry_selection_finds_internal_merge_past_terminal_paths() {
        let blocks = vec![
            bb("%entry", &["br label %header"]),
            bb("%header", &["br label %selection"]),
            bb("%selection", &["br i1 %choose, label %left, label %right"]),
            bb("%left", &["br i1 %left_stop, label %ret, label %join"]),
            bb("%right", &["br i1 %right_stop, label %ret, label %join"]),
            bb("%join", &["br label %continue"]),
            bb("%continue", &["br label %header"]),
            bb("%ret", &["ret void"]),
            bb("%exit", &["unreachable"]),
        ];
        let forest = analyze(&blocks);
        let loop_merges = HashMap::from([(
            "%header".to_string(),
            LoopMergeInfo {
                merge: "%exit".to_string(),
                continue_target: "%continue".to_string(),
            },
        )]);
        let mut selections = HashMap::new();
        refine_loop_entry_terminal_selection_merges(
            &blocks,
            &forest,
            &loop_merges,
            &mut selections,
        );
        assert_eq!(selections.get("%selection"), Some(&"%join".to_string()));
    }

    /// The same terminal-aware convergence applies below a loop's entry block, not only to the
    /// conditional lifted directly off the loop header.
    #[test]
    fn nested_loop_selection_finds_merge_past_terminal_path() {
        let blocks = vec![
            bb("%entry", &["br label %header"]),
            bb("%header", &["br label %prefix"]),
            bb("%prefix", &["br label %selection"]),
            bb("%selection", &["br i1 %choose, label %left, label %right"]),
            bb("%left", &["br i1 %left_stop, label %ret, label %join"]),
            bb("%right", &["br label %join"]),
            bb("%join", &["br label %continue"]),
            bb("%continue", &["br label %header"]),
            bb("%ret", &["ret void"]),
            bb("%exit", &["unreachable"]),
        ];
        let forest = analyze(&blocks);
        let loop_merges = HashMap::from([(
            "%header".to_string(),
            LoopMergeInfo {
                merge: "%exit".to_string(),
                continue_target: "%continue".to_string(),
            },
        )]);
        let mut selections = HashMap::new();
        refine_nested_terminal_selection_merges(
            &blocks,
            &forest,
            &loop_merges,
            &HashMap::new(),
            &mut selections,
        );
        assert_eq!(selections.get("%selection"), Some(&"%join".to_string()));
    }

    /// Prefer the direct arm that all live paths first reach over a later common block.
    #[test]
    fn nested_loop_selection_prefers_direct_arm_convergence() {
        let blocks = vec![
            bb("%entry", &["br label %header"]),
            bb("%header", &["br label %prefix"]),
            bb("%prefix", &["br label %selection"]),
            bb("%selection", &["br i1 %choose, label %work, label %join"]),
            bb("%work", &["br label %join"]),
            bb("%join", &["br label %later"]),
            bb("%later", &["br label %continue"]),
            bb("%continue", &["br label %header"]),
            bb("%exit", &["unreachable"]),
        ];
        let forest = analyze(&blocks);
        let loop_merges = HashMap::from([(
            "%header".to_string(),
            LoopMergeInfo {
                merge: "%exit".to_string(),
                continue_target: "%continue".to_string(),
            },
        )]);
        let mut selections = HashMap::new();
        refine_nested_terminal_selection_merges(
            &blocks,
            &forest,
            &loop_merges,
            &HashMap::new(),
            &mut selections,
        );
        assert_eq!(selections.get("%selection"), Some(&"%join".to_string()));
    }

    /// A conditional nested in more than one loop may directly break the outer loop. The live arm
    /// is its merge; requiring the exit to be a role of only the innermost loop loses that legal
    /// multi-level structured exit.
    #[test]
    fn nested_selection_uses_live_arm_when_other_arm_exits_outer_loop() {
        let blocks = vec![
            bb("%entry", &["br label %outer_header"]),
            bb("%outer_header", &["br label %inner_header"]),
            bb("%inner_header", &["br label %selection"]),
            bb(
                "%selection",
                &["br i1 %choose, label %work, label %outer_exit"],
            ),
            bb(
                "%work",
                &["br i1 %again, label %inner_continue, label %outer_continue"],
            ),
            bb("%inner_continue", &["br label %inner_header"]),
            bb("%outer_continue", &["br label %outer_header"]),
            bb("%outer_exit", &["ret void"]),
        ];
        let forest = analyze(&blocks);
        let loop_merges = HashMap::from([
            (
                "%inner_header".to_string(),
                LoopMergeInfo {
                    merge: "%outer_continue".to_string(),
                    continue_target: "%inner_continue".to_string(),
                },
            ),
            (
                "%outer_header".to_string(),
                LoopMergeInfo {
                    merge: "%outer_exit".to_string(),
                    continue_target: "%outer_continue".to_string(),
                },
            ),
        ]);
        let mut selections = HashMap::new();
        refine_nested_terminal_selection_merges(
            &blocks,
            &forest,
            &loop_merges,
            &HashMap::new(),
            &mut selections,
        );
        assert_eq!(selections.get("%selection"), Some(&"%work".to_string()));
    }

    /// A loop-local switch keeps its live convergence even when one case returns from the function.
    #[test]
    fn nested_loop_switch_finds_merge_past_terminal_case() {
        let blocks = vec![
            bb("%entry", &["br label %header"]),
            bb("%header", &["br label %prefix"]),
            bb("%prefix", &["br label %switch"]),
            bb(
                "%switch",
                &["switch i32 %selector, label %default [ i32 0, label %case0 i32 1, label %ret ]"],
            ),
            bb("%case0", &["br label %join"]),
            bb("%default", &["br label %join"]),
            bb("%join", &["br label %continue"]),
            bb("%continue", &["br label %header"]),
            bb("%ret", &["ret void"]),
            bb("%exit", &["unreachable"]),
        ];
        let forest = analyze(&blocks);
        let loop_merges = HashMap::from([(
            "%header".to_string(),
            LoopMergeInfo {
                merge: "%exit".to_string(),
                continue_target: "%continue".to_string(),
            },
        )]);
        let mut selections = HashMap::new();
        refine_nested_terminal_selection_merges(
            &blocks,
            &forest,
            &loop_merges,
            &HashMap::new(),
            &mut selections,
        );
        assert_eq!(selections.get("%switch"), Some(&"%join".to_string()));
    }

    /// Bare unreachable arms hide an otherwise ordinary loop-free convergence from virtual-exit
    /// post-dominance. The terminal-aware refinement retains the first shared live continuation.
    #[test]
    fn loop_free_selection_finds_merge_past_unreachable_paths() {
        let blocks = vec![
            bb("%entry", &["br i1 %run, label %selection, label %join"]),
            bb("%selection", &["br i1 %choose, label %left, label %right"]),
            bb(
                "%left",
                &["br i1 %live_left, label %left_work, label %trap"],
            ),
            bb(
                "%right",
                &["br i1 %live_right, label %right_work, label %trap"],
            ),
            bb("%left_work", &["br label %join"]),
            bb("%right_work", &["br label %join"]),
            bb("%join", &["br label %after"]),
            bb("%after", &["ret void"]),
            bb("%trap", &["unreachable"]),
        ];
        let forest = analyze(&blocks);
        let mut selections = HashMap::new();
        refine_nested_terminal_selection_merges(
            &blocks,
            &forest,
            &HashMap::new(),
            &HashMap::new(),
            &mut selections,
        );
        assert_eq!(selections.get("%selection"), Some(&"%join".to_string()));
    }

    /// A loop-free guard and a later loop cannot both own their shared return directly. The generic
    /// convergence refinement must leave this case to the terminal planner, which splits the loop
    /// exit and gives the guard a private merge before the loop entry.
    #[test]
    fn loop_free_guard_defers_shared_loop_return_to_terminal_ownership() {
        let blocks = vec![
            bb("%entry", &["br label %guard"]),
            bb("%guard", &["br i1 %early, label %ret, label %preheader"]),
            bb("%preheader", &["br label %loop"]),
            bb("%loop", &["br label %latch"]),
            bb("%latch", &["br i1 %again, label %loop, label %ret"]),
            bb("%ret", &["ret void"]),
        ];
        let forest = analyze(&blocks);
        let mut selections = HashMap::new();
        refine_nested_terminal_selection_merges(
            &blocks,
            &forest,
            &HashMap::new(),
            &HashMap::new(),
            &mut selections,
        );
        assert!(!selections.contains_key("%guard"));

        let terminal = prepare_terminal_exit_selection(&blocks)
            .expect("shared loop return has explicit terminal ownership");
        let guard_merge = terminal.merges.get("%guard").expect("guard merge");
        assert_ne!(guard_merge, "%preheader");
        assert_ne!(guard_merge, "%ret");
        assert!(terminal
            .blocks
            .iter()
            .any(|block| { block.name == *guard_merge && block.role == BlockRole::LMerge }));
    }

    /// A shared continuation with an outside predecessor is not header-dominated, but it remains the
    /// selection merge when every arm reaches it. The convergence proof must not require a terminal arm.
    #[test]
    fn loop_free_selection_finds_externally_entered_shared_continuation() {
        let blocks = vec![
            bb(
                "%entry",
                &["br i1 %enter, label %selection, label %continuation"],
            ),
            bb(
                "%selection",
                &["br i1 %choose, label %work, label %continuation"],
            ),
            bb("%work", &["br label %continuation"]),
            bb("%continuation", &["br label %exit"]),
            bb("%exit", &["ret void"]),
        ];
        let forest = analyze(&blocks);
        assert_eq!(
            terminal_exit_convergence(&blocks, &forest, "%selection"),
            Some("%continuation".to_string())
        );
    }

    #[test]
    fn loop_free_switch_finds_merge_past_unreachable_default() {
        let blocks = vec![
            bb("%entry", &["br label %switch"]),
            bb(
                "%switch",
                &["switch i32 %selector, label %trap [ i32 0, label %case0 i32 1, label %case1 ]"],
            ),
            bb("%case0", &["br label %join"]),
            bb("%case1", &["br label %join"]),
            bb("%join", &["br label %ret"]),
            bb("%ret", &["ret void"]),
            bb("%trap", &["unreachable"]),
        ];
        let forest = analyze(&blocks);
        let mut selections = HashMap::new();
        refine_nested_terminal_selection_merges(
            &blocks,
            &forest,
            &HashMap::new(),
            &HashMap::new(),
            &mut selections,
        );
        assert_eq!(selections.get("%switch"), Some(&"%join".to_string()));
    }

    #[test]
    fn loop_free_selection_finds_merge_beyond_nested_loop_and_return() {
        let blocks = vec![
            bb("%entry", &["br label %selection"]),
            bb("%selection", &["br i1 %choose, label %loop, label %join"]),
            bb("%loop", &["br i1 %work, label %body, label %join"]),
            bb("%body", &["br i1 %stop, label %ret, label %latch"]),
            bb("%latch", &["br label %loop"]),
            bb("%join", &["br label %ret"]),
            bb("%ret", &["ret void"]),
        ];
        let forest = analyze(&blocks);
        let mut selections = HashMap::new();
        refine_nested_terminal_selection_merges(
            &blocks,
            &forest,
            &HashMap::new(),
            &HashMap::new(),
            &mut selections,
        );
        assert_eq!(selections.get("%selection"), Some(&"%join".to_string()));
    }

    /// The exact loop-entry proof remains active above the general loop-exit search's size budget.
    #[test]
    fn large_loop_entry_terminal_selection_reaches_the_exact_retry() {
        let mut blocks = vec![
            bb("%entry", &["br label %header"]),
            bb("%header", &["br label %selection"]),
            bb(
                "%selection",
                &["br i1 %choose, label %body, label %converge"],
            ),
            bb("%body", &["br i1 %stop, label %ret, label %converge"]),
            bb("%converge", &["br label %continue"]),
            bb("%continue", &["br label %header"]),
            bb("%ret", &["ret void"]),
            bb("%exit", &["unreachable"]),
        ];
        for index in 0..LOOP_EXIT_SELECTION_MAX_BLOCKS {
            blocks.push(bb(&format!("%dead{index}"), &["unreachable"]));
        }
        assert!(blocks.len() > LOOP_EXIT_SELECTION_MAX_BLOCKS);
        let forest = analyze(&blocks);
        let loop_merges = HashMap::from([(
            "%header".to_string(),
            LoopMergeInfo {
                merge: "%exit".to_string(),
                continue_target: "%continue".to_string(),
            },
        )]);
        let mut selections = HashMap::new();
        refine_loop_entry_terminal_selection_merges(
            &blocks,
            &forest,
            &loop_merges,
            &mut selections,
        );
        assert_eq!(selections.get("%selection"), Some(&"%converge".to_string()));
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
        assert!(!selection_synth_growth_exceeds_ladder_cap(
            SELECTION_SYNTH_GROWTH_MAX_BLOCKS * 3,
            SELECTION_SYNTH_GROWTH_MAX_BLOCKS * 3 + 1
        ));
        assert!(selection_synth_growth_exceeds_ladder_cap(
            63,
            63 + SELECTION_SYNTH_GROWTH_MAX_BLOCKS + 1
        ));
        assert!(!selection_synth_growth_exceeds_ladder_cap(400, 800));
        assert!(selection_synth_growth_exceeds_ladder_cap(400, 801));
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

    #[test]
    fn dense_cfg_declines_repeated_local_planning() {
        let mut blocks = Vec::new();
        for index in 0..100 {
            let left = format!("%b{}", (index + 1).min(100));
            let right = format!("%b{}", (index + 2).min(100));
            blocks.push(bb(
                &format!("%b{index}"),
                &[&format!("br i1 %condition, label {left}, label {right}")],
            ));
        }
        blocks.push(bb("%b100", &["ret void"]));

        assert!(exceeds_local_structured_plan_budget(&blocks));
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

    /// The reusable two-target dispatch owns the complete SSA edge contract independently of loop
    /// discovery: predecessor edges converge at one selector block, and destination phis retain
    /// unrelated incoming values while replacing routed predecessors with that block.
    #[test]
    fn two_target_dispatch_funnels_typed_destination_phis() {
        let mut blocks = vec![
            bb("%entry", &["br i1 %c, label %p0, label %p1"]),
            bb("%p0", &["br label %x"]),
            bb("%p1", &["br label %y"]),
            bb("%outside.x", &["br label %x"]),
            bb("%outside.y", &["br label %y"]),
            bb(
                "%x",
                &["%vx = phi i32 [ 1, %p0 ], [ 2, %outside.x ]", "ret void"],
            ),
            bb(
                "%y",
                &["%vy = phi i32 [ 3, %p1 ], [ 4, %outside.y ]", "ret void"],
            ),
        ];
        let exits = vec!["%x".to_string(), "%y".to_string()];
        let predecessors = vec![("%p0".to_string(), 0), ("%p1".to_string(), 1)];
        let mut counter = 0;

        let dispatch = synth_two_target_dispatch(&mut blocks, &exits, &predecessors, &mut counter)
            .expect("typed dispatch");

        for predecessor in ["%p0", "%p1"] {
            assert_eq!(
                block_successors(
                    blocks
                        .iter()
                        .find(|block| block.name == predecessor)
                        .unwrap()
                ),
                std::slice::from_ref(&dispatch)
            );
        }
        assert_eq!(
            block_successors(blocks.iter().find(|block| block.name == dispatch).unwrap()),
            exits
        );
        let dispatch_phis =
            carrier_phis(blocks.iter().find(|block| block.name == dispatch).unwrap());
        assert_eq!(dispatch_phis.len(), 3);
        for (_, incoming) in dispatch_phis {
            assert_eq!(incoming.len(), 2);
            assert!(incoming.iter().any(|(_, predecessor)| predecessor == "%p0"));
            assert!(incoming.iter().any(|(_, predecessor)| predecessor == "%p1"));
        }
        for (target, phi, outside) in [("%x", "%vx", "%outside.x"), ("%y", "%vy", "%outside.y")] {
            let incoming = phi_incomings(
                blocks.iter().find(|block| block.name == target).unwrap(),
                phi,
            );
            assert_eq!(incoming.len(), 2);
            assert!(incoming
                .iter()
                .any(|(_, predecessor)| *predecessor == dispatch));
            assert!(incoming
                .iter()
                .any(|(_, predecessor)| predecessor == outside));
        }
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
    /// back to the relooper retry — and the reject diagnostic agrees (mirror consistency).
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
    /// behavior — this shape, banked `e848bc87`/`72cbab44`, was punted to the relooper retry), the
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

    /// A loop-header switch with an exit arm is lowered to a comparison ladder before admission. The
    /// header owns the loop merge and first conditional; later comparisons are ordinary selection
    /// blocks, so no block needs both `OpLoopMerge` and `OpSelectionMerge`.
    #[test]
    fn loop_header_exit_switch_is_lowered_and_admitted() {
        let blocks = vec![
            bb("%entry", &["br label %h"]),
            bb(
                "%h",
                &["switch i32 %sel, label %merge [ i32 0, label %body i32 1, label %merge ]"],
            ),
            bb("%body", &["br label %h"]),
            bb("%merge", &["ret void"]),
        ];
        let plan = structured_plan(&blocks).expect("loop-header exit switch must structure");
        assert!(plan.blocks.iter().all(|block| {
            !block.typed.as_ref().is_some_and(|typed| {
                matches!(
                    typed.terminator,
                    crate::native::tir::TirTerminator::Switch { .. }
                )
            })
        }));
    }

    #[test]
    fn multi_exit_loop_header_switch_is_lowered_before_planning() {
        let blocks = vec![
            bb("%entry", &["br label %h"]),
            bb(
                "%h",
                &["switch i32 %sel, label %left [ i32 0, label %body i32 1, label %right ]"],
            ),
            bb("%body", &["br label %h"]),
            bb("%left", &["ret void"]),
            bb("%right", &["ret void"]),
        ];
        let lowered = super::blocks::lower_loop_exit_switches(&blocks);
        assert!(lowered.iter().all(|block| {
            !block.typed.as_ref().is_some_and(|typed| {
                matches!(
                    typed.terminator,
                    crate::native::tir::TirTerminator::Switch { .. }
                )
            })
        }));
    }

    /// A non-header switch inside a loop may target both the loop latch and merge. Those targets are
    /// enclosing loop roles, not case constructs owned by the switch, even when source dominance
    /// makes the raw CFG appear admissible. Planning must select the branch-ladder form first.
    #[test]
    fn loop_role_switch_targets_are_lowered_before_plan_admission() {
        let blocks = vec![
            bb("%entry", &["br label %head"]),
            bb("%head", &["br label %switch"]),
            bb(
                "%switch",
                &["switch i32 %tag, label %exit [ i32 0, label %latch i32 1, label %latch ]"],
            ),
            bb("%latch", &["br label %head"]),
            bb("%exit", &["ret void"]),
        ];
        let plan =
            structured_plan(&blocks).expect("loop-exiting switch must structure as a ladder");
        assert!(
            plan.blocks.iter().all(|block| {
                !block.typed.as_ref().is_some_and(|typed| {
                    matches!(
                        typed.terminator,
                        crate::native::tir::TirTerminator::Switch { .. }
                    )
                })
            }),
            "the raw switch must not survive admission"
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

    /// A simple conditional whose arms both return is represented with a disconnected unreachable
    /// merge; both source returns remain legal exits from the selection construct.
    #[test]
    fn structured_plan_none_for_unshared_exit_reconvergence() {
        let blocks = vec![
            bb("%entry", &["br i1 %c, label %a, label %b"]),
            bb("%a", &["ret void"]),
            bb("%b", &["ret void"]),
        ];
        let plan = structured_plan(&blocks).expect("terminal selection must admit");
        let merge = plan
            .branch_merges
            .get(&("%a".to_string(), "%b".to_string()))
            .expect("terminal merge");
        assert!(plan
            .blocks
            .iter()
            .find(|block| block.name == *merge)
            .is_some_and(is_bare_unreachable));
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

    #[test]
    fn finalized_emission_plan_keeps_external_entry_out_of_loop_continue() {
        let mut blocks = vec![
            bb("%entry", &["br label %preheader"]),
            bb("%preheader", &["br label %continue"]),
            bb(
                "%continue",
                &[
                    "%next = phi i32 [ 0, %preheader ], [ 1, %body ]",
                    "br label %header",
                ],
            ),
            bb(
                "%header",
                &[
                    "%value = phi i32 [ %next, %continue ]",
                    "br i1 %condition, label %body, label %exit",
                ],
            ),
            bb("%body", &["br label %continue"]),
            bb("%exit", &["ret void"]),
        ];
        let loop_merges = HashMap::from([(
            "%header".to_string(),
            LoopMergeInfo {
                merge: "%exit".to_string(),
                continue_target: "%continue".to_string(),
            },
        )]);

        assert!(normalize_loop_continue_external_predecessors(
            &mut blocks,
            &loop_merges,
        ));
        assert!(
            blocks
                .iter()
                .position(|block| block.name == "%header")
                .expect("header position")
                < blocks
                    .iter()
                    .position(|block| block.name == "%continue")
                    .expect("continue position"),
            "new dominance must be reflected in serialization",
        );
        assert_eq!(
            block_successors(
                blocks
                    .iter()
                    .find(|block| block.name == "%preheader")
                    .expect("preheader"),
            ),
            ["%header"],
        );
        assert_eq!(
            phi_incomings(
                blocks
                    .iter()
                    .find(|block| block.name == "%continue")
                    .expect("continue"),
                "%next",
            )
            .iter()
            .map(|(value, predecessor)| (val_str(value), predecessor.as_str()))
            .collect::<Vec<_>>(),
            [("1".to_string(), "%body")],
        );
        assert_eq!(
            phi_incomings(
                blocks
                    .iter()
                    .find(|block| block.name == "%header")
                    .expect("header"),
                "%value",
            )
            .iter()
            .map(|(value, predecessor)| (val_str(value), predecessor.as_str()))
            .collect::<Vec<_>>(),
            [
                ("%next".to_string(), "%continue"),
                ("0".to_string(), "%preheader"),
            ],
        );
    }

    #[test]
    fn finalized_emission_plan_moves_enclosing_selection_off_loop_continue() {
        let mut blocks = vec![
            bb("%entry", &["br i1 %outer, label %header, label %done"]),
            bb("%header", &["br i1 %loop, label %body, label %loop_exit"]),
            bb("%body", &["br label %continue"]),
            bb("%continue", &["br label %header"]),
            bb("%loop_exit", &["br label %done"]),
            bb("%done", &["ret void"]),
        ];
        let loops = HashMap::from([(
            "%header".to_string(),
            LoopMergeInfo {
                merge: "%loop_exit".to_string(),
                continue_target: "%continue".to_string(),
            },
        )]);
        let mut merges = HashMap::from([("%entry".to_string(), "%continue".to_string())]);

        assert!(normalize_continue_selection_merge_targets(
            &mut blocks,
            &loops,
            &mut merges,
        ));
        assert_eq!(merges["%entry"], "%done");
    }

    #[test]
    fn finalized_emission_plan_declines_identical_loop_boundaries_without_spinning() {
        let mut blocks = vec![
            bb("%entry", &["br i1 %outer, label %header, label %done"]),
            bb("%header", &["br label %boundary"]),
            bb("%boundary", &["br label %header"]),
            bb("%done", &["ret void"]),
        ];
        let loops = HashMap::from([(
            "%header".to_string(),
            LoopMergeInfo {
                merge: "%boundary".to_string(),
                continue_target: "%boundary".to_string(),
            },
        )]);
        let mut merges = HashMap::from([("%entry".to_string(), "%boundary".to_string())]);

        assert!(!normalize_continue_selection_merge_targets(
            &mut blocks,
            &loops,
            &mut merges,
        ));
        assert_eq!(merges["%entry"], "%boundary");
    }

    #[test]
    fn finalized_emission_plan_splits_direct_break_from_loop_continue_merge() {
        let mut blocks = vec![
            bb("%header", &["br label %selection"]),
            bb(
                "%selection",
                &["br i1 %condition, label %continue, label %exit"],
            ),
            bb("%continue", &["br label %header"]),
            bb("%exit", &["ret void"]),
        ];
        let loops = HashMap::from([(
            "%header".to_string(),
            LoopMergeInfo {
                merge: "%exit".to_string(),
                continue_target: "%continue".to_string(),
            },
        )]);
        let mut merges = HashMap::from([("%selection".to_string(), "%continue".to_string())]);

        assert!(normalize_continue_selection_merge_targets(
            &mut blocks,
            &loops,
            &mut merges,
        ));
        let private = &merges["%selection"];
        assert_ne!(private, "%continue");
        assert_eq!(
            block_successors(
                blocks
                    .iter()
                    .find(|block| block.name == "%selection")
                    .expect("selection"),
            ),
            ["%continue", private.as_str()],
        );
        assert_eq!(
            block_successors(
                blocks
                    .iter()
                    .find(|block| block.name == *private)
                    .expect("private merge"),
            ),
            ["%exit"],
        );
    }

    #[test]
    fn structured_plan_owns_continue_selection_boundary_before_emission() {
        let blocks = vec![
            bb("%header", &["br label %selection"]),
            bb(
                "%selection",
                &["br i1 %condition, label %continue, label %exit"],
            ),
            bb("%continue", &["br label %header"]),
            bb("%exit", &["ret void"]),
        ];

        let plan = structured_plan(&blocks).expect("structured plan");
        let private = plan
            .branch_merges_by_header
            .get("%selection")
            .expect("selection ownership");
        assert_ne!(private, "%continue");
        assert_eq!(
            block_successors(
                plan.blocks
                    .iter()
                    .find(|block| block.name == "%selection")
                    .expect("selection"),
            ),
            ["%continue", private.as_str()],
        );
        assert_eq!(
            block_successors(
                plan.blocks
                    .iter()
                    .find(|block| block.name == *private)
                    .expect("private merge"),
            ),
            ["%exit"],
        );
    }

    #[test]
    fn finalized_emission_plan_splits_phi_continue_reconvergence() {
        let mut blocks = vec![
            bb("%header", &["br i1 %loop, label %selection, label %exit"]),
            bb(
                "%selection",
                &["br i1 %condition, label %left, label %right"],
            ),
            bb("%left", &["br label %right"]),
            bb("%right", &["br label %continue"]),
            bb(
                "%continue",
                &["%value = phi i32 [ 7, %right ]", "br label %header"],
            ),
            bb("%exit", &["ret void"]),
        ];
        let loops = HashMap::from([(
            "%header".to_string(),
            LoopMergeInfo {
                merge: "%exit".to_string(),
                continue_target: "%continue".to_string(),
            },
        )]);
        let mut merges = HashMap::from([("%selection".to_string(), "%continue".to_string())]);

        assert!(normalize_continue_selection_merge_targets(
            &mut blocks,
            &loops,
            &mut merges,
        ));
        let private = &merges["%selection"];
        assert_eq!(
            block_successors(
                blocks
                    .iter()
                    .find(|block| block.name == "%right")
                    .expect("right arm"),
            ),
            [private.as_str()],
        );
        let continue_incoming = phi_incomings(
            blocks
                .iter()
                .find(|block| block.name == "%continue")
                .expect("continue"),
            "%value",
        );
        assert_eq!(continue_incoming.len(), 1);
        assert_eq!(continue_incoming[0].1, *private);
    }

    #[test]
    fn finalized_emission_plan_privatizes_reused_phi_merge_targets() {
        let mut blocks = vec![
            bb("%outer", &["br i1 %c0, label %inner, label %outer_arm"]),
            bb("%inner", &["br i1 %c1, label %inner_t, label %inner_f"]),
            bb("%inner_t", &["br label %join"]),
            bb("%inner_f", &["br label %join"]),
            bb("%outer_arm", &["br label %join"]),
            bb(
                "%join",
                &[
                    "%value = phi i32 [ 1, %inner_t ], [ 2, %inner_f ], [ 3, %outer_arm ]",
                    "ret void",
                ],
            ),
        ];
        let mut loop_merges = HashMap::new();
        let branch_merges = HashMap::new();
        let mut branch_merges_by_header = HashMap::from([
            ("%outer".to_string(), "%join".to_string()),
            ("%inner".to_string(), "%join".to_string()),
        ]);
        let mut switch_merges = HashMap::new();

        assert!(privatize_reused_emitted_merge_targets(
            &mut blocks,
            &mut loop_merges,
            &branch_merges,
            &mut branch_merges_by_header,
            true,
            &mut switch_merges,
        ));

        let outer_merge = &branch_merges_by_header["%outer"];
        let inner_merge = &branch_merges_by_header["%inner"];
        assert_ne!(outer_merge, inner_merge);
        assert_ne!(outer_merge, "%join");
        assert_ne!(inner_merge, "%join");
        let inner_block = blocks
            .iter()
            .find(|block| block.name == *inner_merge)
            .expect("inner private merge");
        assert_eq!(block_successors(inner_block), [outer_merge.as_str()]);
        let outer_block = blocks
            .iter()
            .find(|block| block.name == *outer_merge)
            .expect("outer private merge");
        assert_eq!(block_successors(outer_block), ["%join"]);
        assert_eq!(
            phi_incomings(
                blocks
                    .iter()
                    .find(|block| block.name == "%join")
                    .expect("join block"),
                "%value",
            )
            .iter()
            .map(|(_, predecessor)| predecessor.as_str())
            .collect::<Vec<_>>(),
            [outer_merge.as_str()],
        );
    }

    #[test]
    fn finalized_emission_plan_does_not_redirect_loop_header_backedges() {
        let mut blocks = vec![
            bb("%outer", &["br i1 %c0, label %inner, label %outer_arm"]),
            bb("%inner", &["br i1 %c1, label %inner_t, label %inner_f"]),
            bb("%inner_t", &["br label %loop"]),
            bb("%inner_f", &["br label %loop"]),
            bb("%outer_arm", &["br label %loop"]),
            bb("%loop", &["br i1 %c2, label %body, label %exit"]),
            bb("%body", &["br label %loop"]),
            bb("%exit", &["ret void"]),
        ];
        let mut loop_merges = HashMap::from([(
            "%loop".to_string(),
            LoopMergeInfo {
                merge: "%exit".to_string(),
                continue_target: "%body".to_string(),
            },
        )]);
        let mut branch_merges_by_header = HashMap::from([
            ("%outer".to_string(), "%loop".to_string()),
            ("%inner".to_string(), "%loop".to_string()),
        ]);

        assert!(privatize_reused_emitted_merge_targets(
            &mut blocks,
            &mut loop_merges,
            &HashMap::new(),
            &mut branch_merges_by_header,
            true,
            &mut HashMap::new(),
        ));

        assert_ne!(branch_merges_by_header["%outer"], "%loop");
        assert_ne!(branch_merges_by_header["%inner"], "%loop");
        assert_eq!(
            block_successors(
                blocks
                    .iter()
                    .find(|block| block.name == "%body")
                    .expect("loop body"),
            ),
            ["%loop"],
            "the loop-header backedge must not enter an ordinary private merge",
        );
    }

    #[test]
    fn finalized_emission_plan_privatizes_merge_from_enclosing_continue() {
        let mut blocks = vec![
            bb("%outer", &["br label %inner"]),
            bb(
                "%inner",
                &["br i1 %inner_loop, label %outer_continue, label %inner"],
            ),
            bb(
                "%outer_continue",
                &["br i1 %outer_loop, label %outer, label %exit"],
            ),
            bb("%exit", &["ret void"]),
        ];
        let mut loops = HashMap::from([
            (
                "%outer".to_string(),
                LoopMergeInfo {
                    merge: "%exit".to_string(),
                    continue_target: "%outer_continue".to_string(),
                },
            ),
            (
                "%inner".to_string(),
                LoopMergeInfo {
                    merge: "%outer_continue".to_string(),
                    continue_target: "%inner".to_string(),
                },
            ),
        ]);

        assert!(privatize_reused_emitted_merge_targets(
            &mut blocks,
            &mut loops,
            &HashMap::new(),
            &mut HashMap::new(),
            true,
            &mut HashMap::new(),
        ));
        let private = &loops["%inner"].merge;
        assert_ne!(private, "%outer_continue");
        assert_eq!(loops["%outer"].continue_target, "%outer_continue");
        assert_eq!(
            block_successors(
                blocks
                    .iter()
                    .find(|block| block.name == *private)
                    .expect("private inner merge"),
            ),
            ["%outer_continue"],
        );
    }

    #[test]
    fn finalized_emission_plan_drops_stale_unconditional_loop_claim() {
        let blocks = vec![
            bb("%entry", &["br label %header"]),
            bb("%header", &["br label %exit"]),
            bb("%exit", &["ret void"]),
        ];
        let mut loops = HashMap::from([(
            "%header".to_string(),
            LoopMergeInfo {
                merge: "%exit".to_string(),
                continue_target: "%header".to_string(),
            },
        )]);

        assert!(normalize_stale_emitted_loop_claims(
            &blocks,
            &mut loops,
            &mut HashMap::new(),
            &mut HashMap::new(),
        ));
        assert!(loops.is_empty());
    }

    #[test]
    fn finalized_emission_plan_reclassifies_stale_conditional_loop_as_selection() {
        let blocks = vec![
            bb("%entry", &["br label %header"]),
            bb("%header", &["br i1 %choose, label %body, label %exit"]),
            bb("%body", &["br label %exit"]),
            bb("%exit", &["ret void"]),
        ];
        let mut loops = HashMap::from([(
            "%header".to_string(),
            LoopMergeInfo {
                merge: "%exit".to_string(),
                continue_target: "%header".to_string(),
            },
        )]);
        let mut branch_merges = HashMap::new();

        assert!(normalize_stale_emitted_loop_claims(
            &blocks,
            &mut loops,
            &mut branch_merges,
            &mut HashMap::new(),
        ));
        assert!(loops.is_empty());
        assert_eq!(branch_merges["%header"], "%exit");
    }

    #[test]
    fn finalized_emission_plan_privatizes_nondominated_selection_merge() {
        let mut blocks = vec![
            bb(
                "%entry",
                &["br i1 %outside, label %header, label %external"],
            ),
            bb("%header", &["br i1 %choose, label %body, label %exit"]),
            bb("%body", &["br label %exit"]),
            bb("%external", &["br label %exit"]),
            bb("%exit", &["ret void"]),
        ];
        let mut branch_merges_by_header =
            HashMap::from([("%header".to_string(), "%exit".to_string())]);

        assert!(privatize_reused_emitted_merge_targets(
            &mut blocks,
            &mut HashMap::new(),
            &HashMap::new(),
            &mut branch_merges_by_header,
            true,
            &mut HashMap::new(),
        ));
        let private = &branch_merges_by_header["%header"];
        assert_ne!(private, "%exit");
        let forest = analyze(&blocks);
        assert!(forest.dominates("%header", private));
        assert_eq!(
            block_successors(blocks.iter().find(|block| block.name == "%body").unwrap()).as_slice(),
            std::slice::from_ref(private)
        );
        assert_eq!(
            block_successors(
                blocks
                    .iter()
                    .find(|block| block.name == "%external")
                    .unwrap(),
            ),
            ["%exit"]
        );
    }

    #[test]
    fn finalized_emission_plan_funnels_phi_bypass_through_declared_merge() {
        let mut blocks = vec![
            bb(
                "%entry",
                &["br i1 %outside, label %header, label %external"],
            ),
            bb("%header", &["br i1 %c, label %merge, label %body"]),
            bb(
                "%merge",
                &["%merged = phi i32 [ 1, %header ]", "br label %join"],
            ),
            bb("%body", &["br label %join"]),
            bb("%external", &["br label %join"]),
            bb(
                "%join",
                &[
                    "%value = phi i32 [ %merged, %merge ], [ 2, %body ], [ 3, %external ]",
                    "ret void",
                ],
            ),
        ];
        let branch_merges_by_header =
            HashMap::from([("%header".to_string(), "%merge".to_string())]);

        assert!(funnel_emitted_selection_merge_bypasses(
            &mut blocks,
            &HashMap::new(),
            &HashMap::new(),
            &branch_merges_by_header,
            true,
            &HashMap::new(),
        ));

        assert_eq!(
            block_successors(blocks.iter().find(|block| block.name == "%body").unwrap(),),
            ["%merge"],
        );
        assert_eq!(
            phi_incomings(
                blocks.iter().find(|block| block.name == "%merge").unwrap(),
                "%merged",
            )
            .iter()
            .map(|(_, predecessor)| predecessor.as_str())
            .collect::<Vec<_>>(),
            ["%header", "%body"],
        );
        assert_eq!(
            phi_incomings(
                blocks.iter().find(|block| block.name == "%join").unwrap(),
                "%value",
            )
            .iter()
            .map(|(_, predecessor)| predecessor.as_str())
            .collect::<Vec<_>>(),
            ["%merge", "%external"],
        );
        assert!(!funnel_emitted_selection_merge_bypasses(
            &mut blocks,
            &HashMap::new(),
            &HashMap::new(),
            &branch_merges_by_header,
            true,
            &HashMap::new(),
        ));
    }

    #[test]
    fn phi_bypass_value_equality_preserves_nan_payload_bits() {
        use crate::native::ir::LlValue;

        let payload = LlValue::Float(f64::from_bits(0x7ff8_0000_0000_0042));
        let same_payload = LlValue::Float(f64::from_bits(0x7ff8_0000_0000_0042));
        let other_payload = LlValue::Float(f64::from_bits(0x7ff8_0000_0000_0043));
        assert!(phi_value_exact_eq(&payload, &same_payload));
        assert!(!phi_value_exact_eq(&payload, &other_payload));
    }

    #[test]
    fn finalized_emission_plan_privatizes_shared_direct_arm_through_merge() {
        let mut blocks = vec![
            bb("%entry", &["br i1 %outer, label %shared, label %inner"]),
            bb("%inner", &["br i1 %nested, label %shared, label %local"]),
            bb("%shared", &["%defined = add i32 %x, 1", "br label %target"]),
            bb("%local", &["br label %merge"]),
            bb("%merge", &["br label %target"]),
            bb(
                "%target",
                &[
                    "%value = phi i32 [ %defined, %shared ], [ 9, %merge ]",
                    "ret void",
                ],
            ),
        ];
        let mut branch_merges_by_header =
            HashMap::from([("%inner".to_string(), "%merge".to_string())]);

        assert!(privatize_emitted_shared_direct_selection_arms(
            &mut blocks,
            &HashMap::new(),
            &HashMap::new(),
            &mut branch_merges_by_header,
            true,
            &HashMap::new(),
        ));
        let clone = blocks
            .iter()
            .find(|block| block.name.starts_with("%xa"))
            .expect("private shared-arm clone")
            .name
            .clone();
        assert!(
            block_successors(blocks.iter().find(|block| block.name == "%inner").unwrap())
                .contains(&clone)
        );

        assert!(!funnel_emitted_selection_merge_bypasses(
            &mut blocks,
            &HashMap::new(),
            &HashMap::new(),
            &branch_merges_by_header,
            true,
            &HashMap::new(),
        ));
        assert_eq!(
            block_successors(blocks.iter().find(|block| block.name == clone).unwrap()),
            ["%merge"],
        );
        let merge_phis = carrier_phis(blocks.iter().find(|block| block.name == "%merge").unwrap());
        assert_eq!(merge_phis.len(), 1);
        assert!(merge_phis[0]
            .1
            .iter()
            .any(|(_, predecessor)| predecessor == "%local"));
        assert!(merge_phis[0]
            .1
            .iter()
            .any(|(_, predecessor)| predecessor == &clone));
        assert_eq!(
            phi_incomings(
                blocks.iter().find(|block| block.name == "%target").unwrap(),
                "%value",
            )
            .iter()
            .map(|(_, predecessor)| predecessor.as_str())
            .collect::<Vec<_>>(),
            ["%shared", "%merge"],
        );
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
    /// (break) or the continue block. Planning gives the conditional a private selection boundary
    /// which then exits to the loop merge, keeping the selection inside the loop construct.
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
        let private = plan
            .branch_merges_by_header
            .get("%body")
            .expect("private body selection merge");
        assert_ne!(private, "%cont");
        assert_eq!(
            block_successors(
                plan.blocks
                    .iter()
                    .find(|block| block.name == "%body")
                    .expect("body"),
            ),
            [private.as_str(), "%cont"],
        );
        assert_eq!(
            block_successors(
                plan.blocks
                    .iter()
                    .find(|block| block.name == *private)
                    .expect("private merge"),
            ),
            ["%merge"],
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

    /// Residual merge-inloop / M2 shape: do-while with a mid-body guarded break that initially claims
    /// the loop continue. Planning must finalize that collision into a private selection merge which
    /// then exits to the loop merge; no post-admission emitter rewrite may own this relationship.
    #[test]
    fn do_while_latch_finalizes_private_selection_boundary() {
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
        let (latch, continue_arm, private) = plan
            .blocks
            .iter()
            .find_map(|block| {
                let (true_target, false_target) = conditional_branch_targets(block)?;
                if true_target == info.continue_target {
                    Some((block, true_target, false_target))
                } else if false_target == info.continue_target {
                    Some((block, false_target, true_target))
                } else {
                    None
                }
            })
            .expect("continue selection");
        assert_eq!(continue_arm, info.continue_target);
        assert_eq!(plan.branch_merges_by_header[&latch.name], private);
        assert_eq!(
            block_successors(
                plan.blocks
                    .iter()
                    .find(|block| block.name == private)
                    .expect("private selection merge"),
            ),
            [info.merge.as_str()],
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

    #[test]
    fn final_natural_loop_exit_remains_bare_after_header_rekey() {
        let blocks = vec![
            bb("%entry", &["br label %header"]),
            bb("%header", &["br label %test"]),
            bb("%test", &["br i1 %done, label %merge, label %continue"]),
            bb("%continue", &["br label %header"]),
            bb("%merge", &["ret void"]),
        ];
        let forest = analyze(&blocks);
        assert!(bare_natural_loop_exit_branch(
            &forest,
            "%test",
            "%merge",
            "%continue"
        ));
    }

    /// An exit-only arm dominated by a loop header belongs to the SPIR-V loop construct even though
    /// it is not part of the natural-loop SCC. It may not bypass the loop merge by jumping directly
    /// to an enclosing selection merge shared with an outside arm.
    #[test]
    fn dominance_owned_loop_arm_cannot_exit_to_outer_selection_merge() {
        let blocks = vec![
            bb("%entry", &["br label %outer"]),
            bb(
                "%outer",
                &["br i1 %choose, label %preheader, label %outside"],
            ),
            bb("%preheader", &["br label %head"]),
            bb(
                "%head",
                &[
                    "%i = phi i32 [ 0, %preheader ], [ %n, %latch ]",
                    "br label %body",
                ],
            ),
            bb("%body", &["br i1 %escape, label %exit_only, label %latch"]),
            bb("%latch", &["%n = add i32 %i, 1", "br label %head"]),
            bb("%exit_only", &["br label %outer_merge"]),
            bb("%loop_merge", &["br label %outer_merge"]),
            bb("%outside", &["br label %outer_merge"]),
            bb("%outer_merge", &["ret void"]),
        ];
        let loop_merges = HashMap::from([(
            "%head".to_string(),
            LoopMergeInfo {
                merge: "%loop_merge".to_string(),
                continue_target: "%latch".to_string(),
            },
        )]);
        let forest = analyze(&blocks);
        let natural_loop = forest
            .loops
            .iter()
            .find(|natural_loop| natural_loop.header == "%head")
            .expect("natural loop");
        assert!(!natural_loop.body.iter().any(|node| node == "%exit_only"));
        assert!(forest.dominates("%head", "%exit_only"));
        assert_eq!(
            dominance_loop_exit_escape_reason(&blocks, &loop_merges),
            Some("loop-exit:dominance-owned-bypass")
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

        repair_construct_tree_passthrough_selection_merges(
            &blocks,
            &HashMap::new(),
            &mut header_merges,
        );

        assert_eq!(
            header_merges.get("%header").map(String::as_str),
            Some("%join")
        );
    }

    #[test]
    fn construct_tree_does_not_promote_private_merge_onto_claimed_outer_merge() {
        let blocks = vec![
            bb("%outer", &["br i1 %a, label %inner, label %join"]),
            bb("%inner", &["br i1 %b, label %pass, label %work"]),
            bb("%work", &["br label %join"]),
            bb_role("%pass", BlockRole::LMerge, &["br label %join"]),
            bb("%join", &["ret void"]),
        ];
        let mut header_merges = HashMap::from([
            ("%outer".to_string(), "%join".to_string()),
            ("%inner".to_string(), "%pass".to_string()),
        ]);

        repair_construct_tree_passthrough_selection_merges(
            &blocks,
            &HashMap::new(),
            &mut header_merges,
        );

        assert_eq!(
            header_merges.get("%inner").map(String::as_str),
            Some("%pass"),
            "the outer selection already owns %join"
        );
    }

    #[test]
    fn construct_tree_direct_terminal_guards_compose_innermost_first() {
        let blocks = vec![
            bb("%entry", &["br label %outer"]),
            bb("%outer", &["br i1 %a, label %inner, label %ret"]),
            bb("%inner", &["br i1 %b, label %continuation, label %ret"]),
            bb(
                "%continuation",
                &["%value = phi i32 [ 7, %inner ]", "br label %tail"],
            ),
            bb("%tail", &["ret void"]),
            bb("%ret", &["ret void"]),
        ];

        let plan = direct_terminal_exit_selection_merges(&blocks, &HashMap::new())
            .expect("both direct terminal guards should be owned");
        for (header, successor) in [("%outer", "%inner"), ("%inner", "%continuation")] {
            let merge = plan
                .merges
                .get(header)
                .expect("terminal guard receives a private merge");
            let merge_block = plan
                .blocks
                .iter()
                .find(|block| block.name == *merge)
                .expect("private merge materialized");
            assert_eq!(block_successors(merge_block), vec![successor.to_string()]);
            let header_block = plan
                .blocks
                .iter()
                .find(|block| block.name == header)
                .expect("header retained");
            let arms = block_successors(header_block);
            assert!(arms.contains(merge));
            assert!(arms.iter().any(|arm| {
                plan.blocks
                    .iter()
                    .any(|block| block.name == *arm && block.role == BlockRole::TerminalExitReturn)
            }));
        }
        let continuation = plan
            .blocks
            .iter()
            .find(|block| block.name == "%continuation")
            .expect("continuation retained");
        let incoming = continuation
            .typed
            .as_ref()
            .and_then(|typed| typed.insts.first())
            .and_then(|instruction| instruction.phi_incoming().as_ref())
            .expect("continuation phi retained");
        assert_eq!(incoming.1[0].1, plan.merges["%inner"]);
    }

    #[test]
    fn construct_tree_direct_terminal_owner_accepts_a_proved_linear_tail() {
        let blocks = vec![
            bb(
                "%entry",
                &["br i1 %outside_path, label %header, label %outside"],
            ),
            bb("%header", &["br i1 %guard, label %live, label %exit"]),
            bb("%live", &["br i1 %work, label %left, label %right"]),
            bb("%left", &["ret void"]),
            bb("%right", &["ret void"]),
            bb("%exit", &["br label %exit_tail"]),
            bb("%exit_tail", &["br label %shared_return"]),
            bb("%outside", &["br label %shared_return"]),
            bb("%shared_return", &["ret void"]),
        ];

        let already_owned = HashMap::from([("%entry".to_string(), "%outside".to_string())]);
        let plan = direct_terminal_exit_selection_merges(&blocks, &already_owned)
            .expect("the proved linear terminal tail is owned before selection construction");
        let merge = plan.merges.get("%header").expect("header owns a merge");
        assert_eq!(
            block_successors(
                plan.blocks
                    .iter()
                    .find(|block| block.name == *merge)
                    .expect("private merge materialized")
            ),
            vec!["%live".to_string()]
        );
        let exit_tail = plan
            .blocks
            .iter()
            .find(|block| block.name == "%exit_tail")
            .expect("terminal predecessor retained");
        let exit_tail_successors = block_successors(exit_tail);
        let [private_return] = exit_tail_successors.as_slice() else {
            panic!("terminal predecessor must have one successor");
        };
        assert!(plan.blocks.iter().any(|block| {
            block.name == *private_return && block.role == BlockRole::TerminalExitReturn
        }));
        assert_eq!(
            block_successors(
                plan.blocks
                    .iter()
                    .find(|block| block.name == "%outside")
                    .unwrap()
            ),
            vec!["%shared_return".to_string()]
        );
    }

    #[test]
    fn construct_tree_direct_terminal_owner_preserves_work_in_return_blocks() {
        let blocks = vec![
            bb("%header", &["br i1 %guard, label %live, label %exit"]),
            bb("%live", &["br i1 %work, label %left, label %right"]),
            bb("%left", &["ret void"]),
            bb("%right", &["ret void"]),
            bb("%exit", &["%value = add i32 1, 2", "ret void"]),
        ];

        assert!(direct_terminal_exit_selection_merges(&blocks, &HashMap::new()).is_none());
    }

    #[test]
    fn construct_tree_generic_repairs_preserve_direct_terminal_guard_ownership() {
        let mut blocks = vec![
            bb("%header", &["br i1 %condition, label %return, label %live"]),
            bb_role("%return", BlockRole::TerminalExitReturn, &["ret void"]),
            bb("%live", &["br i1 %work, label %left, label %right"]),
            bb("%left", &["br label %join"]),
            bb("%right", &["br label %join"]),
            bb_role("%local", BlockRole::LMerge, &["br label %join"]),
            bb("%join", &["ret void"]),
        ];
        let mut merges = HashMap::from([("%header".to_string(), "%local".to_string())]);
        let mut counter = 0usize;

        repair_construct_tree_nondominated_selection_merges(
            &mut blocks,
            &HashMap::new(),
            &mut merges,
            &mut counter,
        );
        repair_construct_tree_passthrough_selection_merges(&blocks, &HashMap::new(), &mut merges);
        assert!(!repair_construct_tree_bypassed_passthrough_merges(
            &mut blocks,
            &mut merges,
            &mut counter,
        ));

        assert_eq!(merges.get("%header").map(String::as_str), Some("%local"));
        assert_eq!(counter, 0);
        assert_eq!(
            block_successors(blocks.iter().find(|block| block.name == "%left").unwrap()),
            vec!["%join".to_string()]
        );
    }

    #[test]
    fn construct_tree_materializes_pure_enclosing_selection_route() {
        let mut blocks = vec![
            bb("%outer", &["br i1 %a, label %route, label %sibling"]),
            bb("%route", &["br i1 %b, label %shared, label %private"]),
            bb_role(
                "%private",
                BlockRole::LMerge,
                &["%x = phi i32 [ 1, %route ]", "br label %outer_merge"],
            ),
            bb("%sibling", &["br label %shared"]),
            bb("%shared", &["br label %outer_merge"]),
            bb("%outer_merge", &["ret void"]),
        ];
        let mut merges = HashMap::from([
            ("%outer".to_string(), "%outer_merge".to_string()),
            ("%route".to_string(), "%private".to_string()),
        ]);
        let mut counter = 0;
        let indexed = pure_enclosing_selection_owners(&blocks, &analyze(&blocks), &merges);

        assert!(materialize_pure_enclosing_selection_routes_for_owner(
            &mut blocks,
            &HashMap::new(),
            &mut merges,
            &indexed,
            "%outer",
            &mut counter,
        ));
        let route_merge = merges
            .get("%route")
            .expect("conditional ownership is preserved");
        assert!(route_merge.starts_with(format!("{SPLIT_PREFIX}{SEL_TOKEN}").as_str()));
        assert_eq!(
            merges.get("%outer").map(String::as_str),
            Some("%outer_merge")
        );
        let route = blocks
            .iter()
            .find(|block| block.name == "%route")
            .expect("route header retained");
        assert!(!block_successors(route).contains(&"%shared".to_string()));
        let private = blocks
            .iter()
            .find(|block| block.name == "%private")
            .expect("phi pass-through retained");
        assert_eq!(block_successors(private), vec![route_merge.clone()]);
        let merge_block = blocks
            .iter()
            .find(|block| block.name == *route_merge)
            .expect("private selection merge materialized");
        assert_eq!(
            block_successors(merge_block),
            vec!["%outer_merge".to_string()]
        );
    }

    #[test]
    fn construct_tree_region_clone_carries_nested_selection_ownership() {
        let mut blocks = vec![
            bb("%outer", &["br i1 %a, label %route, label %sibling"]),
            bb("%route", &["br i1 %b, label %shared, label %private"]),
            bb_role("%private", BlockRole::LMerge, &["br label %outer_merge"]),
            bb("%sibling", &["br label %shared"]),
            bb("%shared", &["br label %nested"]),
            bb("%nested", &["br i1 %c, label %left, label %right"]),
            bb("%left", &["br label %nested_merge"]),
            bb("%right", &["br label %nested_merge"]),
            bb("%nested_merge", &["br label %outer_merge"]),
            bb("%outer_merge", &["ret void"]),
        ];
        let mut merges = HashMap::from([
            ("%outer".to_string(), "%outer_merge".to_string()),
            ("%route".to_string(), "%private".to_string()),
            ("%nested".to_string(), "%nested_merge".to_string()),
        ]);
        let mut counter = 0;
        let indexed = pure_enclosing_selection_owners(&blocks, &analyze(&blocks), &merges);

        assert!(materialize_pure_enclosing_selection_routes_for_owner(
            &mut blocks,
            &HashMap::new(),
            &mut merges,
            &indexed,
            "%outer",
            &mut counter,
        ));

        let (cloned_header, cloned_merge) = merges
            .iter()
            .find(|(header, _)| header.ends_with("_nested"))
            .expect("cloned nested header carries an assignment");
        assert!(cloned_merge.ends_with("_nested_merge"));
        assert!(blocks.iter().any(|block| block.name == *cloned_header));
        assert!(blocks.iter().any(|block| block.name == *cloned_merge));
        assert_eq!(
            merges.get("%nested").map(String::as_str),
            Some("%nested_merge")
        );
    }

    #[test]
    fn construct_tree_privatizes_phi_loop_merge_with_external_predecessor() {
        let mut blocks = vec![
            bb("%entry", &["br i1 %outside, label %external, label %pre"]),
            bb("%pre", &["br label %header"]),
            bb(
                "%header",
                &[
                    "%i = phi i32 [ 0, %pre ], [ %next, %continue ]",
                    "br label %body",
                ],
            ),
            bb(
                "%body",
                &[
                    "%next = add i32 %i, 1",
                    "br i1 %done, label %exit, label %continue",
                ],
            ),
            bb("%continue", &["br label %header"]),
            bb("%external", &["br label %exit"]),
            bb(
                "%exit",
                &[
                    "%value = phi i32 [ %next, %body ], [ 9, %external ]",
                    "ret void",
                ],
            ),
        ];
        let mut loops = HashMap::from([(
            "%header".to_string(),
            LoopMergeInfo {
                merge: "%exit".to_string(),
                continue_target: "%continue".to_string(),
            },
        )]);

        assert!(privatize_nondominated_loop_merges(&mut blocks, &mut loops));
        let private = &loops["%header"].merge;
        assert_ne!(private, "%exit");
        let forest = analyze(&blocks);
        assert!(forest.dominates("%header", private));
        assert_eq!(
            block_successors(blocks.iter().find(|block| block.name == "%body").unwrap()),
            vec![private.clone(), "%continue".to_string()]
        );
        let private_block = blocks
            .iter()
            .find(|block| block.name == *private)
            .expect("private loop merge materialized");
        assert_eq!(block_successors(private_block), vec!["%exit".to_string()]);
        let exit_phi = blocks
            .iter()
            .find(|block| block.name == "%exit")
            .and_then(|block| block.typed.as_ref())
            .and_then(|typed| typed.insts.first())
            .and_then(|instruction| instruction.phi_incoming().as_ref())
            .expect("exit phi retained");
        assert!(exit_phi
            .1
            .iter()
            .any(|(_, predecessor)| predecessor == private));
        assert!(exit_phi
            .1
            .iter()
            .any(|(_, predecessor)| predecessor == "%external"));
    }

    #[test]
    fn construct_tree_assigns_fully_terminal_merges_with_the_other_headers() {
        let blocks = vec![
            bb("%header", &["br i1 %c, label %left, label %right"]),
            bb("%left", &["br i1 %a, label %left_ret, label %left_tail"]),
            bb("%left_tail", &["br label %left_ret"]),
            bb("%left_ret", &["ret void"]),
            bb("%right", &["br i1 %b, label %right_ret, label %right_tail"]),
            bb("%right_tail", &["br label %right_ret"]),
            bb("%right_ret", &["ret void"]),
        ];
        let (planned, _, header_merges, _) = unique_selection_merges_with_construct_tree_ownership(
            &blocks,
            &HashMap::new(),
            false,
            false,
            &HashMap::new(),
        );

        for header in ["%header", "%left", "%right"] {
            assert!(
                header_merges.contains_key(header),
                "{header} was assigned during header construction"
            );
        }
        let terminal_merge = &header_merges["%header"];
        assert!(
            planned
                .iter()
                .find(|block| block.name == *terminal_merge)
                .is_some_and(is_bare_unreachable),
            "%header -> {terminal_merge}"
        );
    }

    #[test]
    fn construct_tree_completes_outer_terminal_guard_continuation() {
        let blocks = vec![
            bb(
                "%header",
                &["br i1 %outer, label %guard, label %continuation"],
            ),
            bb(
                "%guard",
                &["br i1 %inner, label %private_merge, label %private_return"],
            ),
            bb_role(
                "%private_merge",
                BlockRole::LMerge,
                &["br label %continuation"],
            ),
            bb_role(
                "%private_return",
                BlockRole::TerminalExitReturn,
                &["ret void"],
            ),
            bb("%continuation", &["br label %return"]),
            bb("%return", &["ret void"]),
        ];
        let (planned, _, merges, _) = unique_selection_merges_with_construct_tree_ownership(
            &blocks,
            &HashMap::new(),
            false,
            false,
            &HashMap::new(),
        );
        let outer_merge = merges
            .get("%header")
            .expect("outer terminal continuation is owned during construction");
        assert!(outer_merge.starts_with(format!("{SPLIT_PREFIX}{SEL_TOKEN}").as_str()));
        assert_eq!(
            block_successors(
                planned
                    .iter()
                    .find(|block| block.name == *outer_merge)
                    .expect("outer merge is materialized")
            ),
            vec!["%continuation".to_string()]
        );
        let guard_merge = merges
            .get("%guard")
            .expect("nested terminal continuation is owned during construction");
        assert!(guard_merge.starts_with(format!("{SPLIT_PREFIX}{SEL_TOKEN}").as_str()));
        assert_eq!(
            block_successors(
                planned
                    .iter()
                    .find(|block| block.name == *guard_merge)
                    .expect("nested merge is materialized")
            ),
            vec![outer_merge.clone()]
        );
        assert!(planned.iter().any(|block| block.name == "%continuation"));
    }

    #[test]
    fn terminal_convergence_rejects_region_reentry_after_early_merge() {
        let blocks = vec![
            bb("%header", &["br i1 %c, label %left, label %right"]),
            bb("%left", &["br i1 %a, label %early, label %late"]),
            bb("%right", &["br i1 %b, label %early, label %late"]),
            bb("%early", &["br label %work"]),
            bb("%work", &["br label %late"]),
            bb("%late", &["br label %return"]),
            bb("%return", &["ret void"]),
        ];
        let forest = analyze(&blocks);

        assert_eq!(
            terminal_exit_convergence(&blocks, &forest, "%header").as_deref(),
            Some("%late")
        );
    }

    #[test]
    fn late_terminal_convergence_privatizes_its_shared_return() {
        let mut blocks = vec![
            bb("%header", &["br i1 %c, label %merge, label %terminal"]),
            bb("%terminal", &["br label %shared_return"]),
            bb("%merge", &["br label %downstream"]),
            bb("%downstream", &["br label %shared_return"]),
            bb("%shared_return", &["ret void"]),
        ];
        let mut merges = HashMap::new();
        let mut counter = 0usize;

        assert!(complete_construct_tree_terminal_convergences(
            &blocks,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &mut merges,
        ));
        assert_eq!(merges.get("%header").map(String::as_str), Some("%merge"));
        assert!(privatize_direct_arm_terminal_return(
            &mut blocks,
            "%header",
            &merges["%header"],
            &mut counter,
        ));

        let terminal_target = block_successors(
            blocks
                .iter()
                .find(|block| block.name == "%terminal")
                .unwrap(),
        );
        assert_eq!(terminal_target.len(), 1);
        assert_ne!(terminal_target[0], "%shared_return");
        assert!(blocks.iter().any(|block| {
            block.name == terminal_target[0] && block.role == BlockRole::TerminalExitReturn
        }));
        assert_eq!(
            block_successors(
                blocks
                    .iter()
                    .find(|block| block.name == "%downstream")
                    .unwrap()
            ),
            vec!["%shared_return".to_string()]
        );
    }

    #[test]
    fn construct_tree_coalesces_sibling_conditional_dispatches() {
        let mut blocks = vec![
            bb("%header", &["br i1 %outer, label %left, label %right"]),
            bb("%left", &["br i1 %a, label %x, label %y"]),
            bb("%right", &["br i1 %b, label %x, label %y"]),
            bb(
                "%x",
                &["%vx = phi i32 [ 1, %left ], [ 2, %right ]", "ret void"],
            ),
            bb(
                "%y",
                &["%vy = phi i32 [ 3, %left ], [ 4, %right ]", "ret void"],
            ),
        ];

        assert!(coalesce_sibling_conditional_dispatches(&mut blocks));
        let forest = analyze(&blocks);
        let merge = selection_merges(&blocks, &forest)
            .remove("%header")
            .expect("outer sibling choice now has a natural merge");
        for arm in ["%left", "%right"] {
            assert_eq!(
                block_successors(blocks.iter().find(|block| block.name == arm).unwrap()),
                vec![merge.clone()]
            );
        }
        let merge_block = blocks
            .iter()
            .find(|block| block.name == merge)
            .expect("coalesced dispatch block exists");
        assert_eq!(
            block_successors(merge_block),
            vec!["%x".to_string(), "%y".to_string()]
        );
        let merge_phis = merge_block
            .typed
            .as_ref()
            .expect("coalesced dispatch is typed")
            .insts
            .iter()
            .filter(|instruction| instruction.is_phi())
            .count();
        assert_eq!(merge_phis, 3);
        for target in ["%x", "%y"] {
            let incoming = blocks
                .iter()
                .find(|block| block.name == target)
                .and_then(|block| block.typed.as_ref())
                .and_then(|typed| typed.insts.first())
                .and_then(|instruction| instruction.phi_incoming().as_ref())
                .expect("target phi retained");
            assert_eq!(incoming.1.len(), 1);
            assert_eq!(incoming.1[0].1, merge);
        }
    }

    #[test]
    fn terminal_parent_composes_after_nested_private_merge() {
        let mut blocks = vec![
            bb(
                "%parent",
                &["br i1 %outer, label %child, label %parent_return"],
            ),
            bb(
                "%child",
                &["br i1 %inner, label %child_merge, label %child_return"],
            ),
            bb_role("%child_merge", BlockRole::LMerge, &["br label %final"]),
            bb_role(
                "%child_return",
                BlockRole::TerminalExitReturn,
                &["ret void"],
            ),
            bb_role(
                "%parent_return",
                BlockRole::TerminalExitReturn,
                &["ret void"],
            ),
            bb("%final", &["ret void"]),
        ];
        let mut merges = HashMap::from([
            ("%parent".to_string(), "%parent_return".to_string()),
            ("%child".to_string(), "%child_merge".to_string()),
        ]);
        let mut counter = 0usize;

        assert!(compose_terminal_parent_nested_merges(
            &mut blocks,
            &mut merges,
            &mut counter,
        ));
        let parent_merge = merges.get("%parent").unwrap();
        assert_ne!(parent_merge, "%parent_return");
        assert_ne!(parent_merge, "%child_merge");
        assert_eq!(
            block_successors(
                blocks
                    .iter()
                    .find(|block| block.name == "%child_merge")
                    .unwrap()
            ),
            vec![parent_merge.clone()]
        );
        assert_eq!(
            block_successors(
                blocks
                    .iter()
                    .find(|block| block.name == *parent_merge)
                    .unwrap()
            ),
            vec!["%final".to_string()]
        );
    }

    #[test]
    fn final_terminal_guard_merge_returns_to_the_direct_live_edge() {
        let mut blocks = vec![
            bb(
                "%header",
                &["br i1 %condition, label %terminal_tail, label %work"],
            ),
            bb("%terminal_tail", &["br label %return"]),
            bb("%work", &["br i1 %more, label %left, label %right"]),
            bb("%left", &["br label %shared"]),
            bb("%right", &["br label %shared"]),
            bb_role("%stale", BlockRole::LMerge, &["br label %shared"]),
            bb("%shared", &["ret void"]),
            bb("%return", &["ret void"]),
        ];
        let mut merges = HashMap::from([("%header".to_string(), "%stale".to_string())]);
        let mut counter = 0usize;

        assert!(finalize_direct_terminal_guard_merges(
            &mut blocks,
            &mut merges,
            &mut counter,
        ));
        let merge = merges
            .get("%header")
            .expect("header receives a fresh merge");
        assert_ne!(merge, "%stale");
        assert_eq!(
            block_successors(blocks.iter().find(|block| block.name == "%header").unwrap()),
            vec!["%terminal_tail".to_string(), merge.clone()]
        );
        assert_eq!(
            block_successors(blocks.iter().find(|block| block.name == *merge).unwrap()),
            vec!["%work".to_string()]
        );
    }

    #[test]
    fn final_terminal_switch_removes_shared_return_passthroughs() {
        let mut blocks = vec![
            bb(
                "%entry",
                &["br i1 %outside, label %switch, label %external"],
            ),
            bb(
                "%switch",
                &[
                    "switch i32 %selector, label %default [ i32 0, label %left i32 1, label %right ]",
                ],
            ),
            bb("%left", &["br label %shared"]),
            bb("%right", &["br label %shared"]),
            bb("%default", &["unreachable"]),
            bb("%external", &["br label %shared"]),
            bb_role("%shared", BlockRole::LMerge, &["br label %return"]),
            bb("%return", &["ret void"]),
            bb_role("%stale", BlockRole::LMerge, &["br label %shared"]),
        ];
        let mut merges = HashMap::from([("%switch".to_string(), "%stale".to_string())]);
        let mut counter = 0usize;

        assert!(finalize_fully_terminal_switches(
            &mut blocks,
            &mut merges,
            &mut counter,
        ));
        for case in ["%left", "%right"] {
            assert_eq!(
                blocks
                    .iter()
                    .find(|block| block.name == case)
                    .unwrap()
                    .lines(),
                ["ret void"]
            );
        }
        assert_eq!(
            block_successors(
                blocks
                    .iter()
                    .find(|block| block.name == "%external")
                    .unwrap()
            ),
            vec!["%shared".to_string()]
        );
        let merge = merges.get("%switch").expect("switch receives a merge");
        assert!(blocks
            .iter()
            .find(|block| block.name == *merge)
            .is_some_and(is_bare_unreachable));
    }

    #[test]
    fn construct_tree_privatizes_shared_return_below_direct_arm_merge() {
        let blocks = vec![
            bb(
                "%entry",
                &["br i1 %outside, label %header, label %external"],
            ),
            bb("%header", &["br i1 %c, label %work, label %merge"]),
            bb(
                "%work",
                &["br i1 %nested, label %shared_return, label %private_return"],
            ),
            bb_role(
                "%private_return",
                BlockRole::TerminalExitReturn,
                &["ret void"],
            ),
            bb("%merge", &["br label %shared_return"]),
            bb("%external", &["br label %shared_return"]),
            bb("%shared_return", &["ret void"]),
        ];
        let forced = HashMap::from([("%header".to_string(), "%merge".to_string())]);
        let (blocks, _, merges, _) = unique_selection_merges_with_construct_tree_ownership(
            &blocks,
            &HashMap::new(),
            false,
            false,
            &forced,
        );
        assert_eq!(merges.get("%header").map(String::as_str), Some("%merge"));
        let work_target =
            block_successors(blocks.iter().find(|block| block.name == "%work").unwrap());
        assert_eq!(work_target.len(), 2);
        assert!(work_target.contains(&"%private_return".to_string()));
        let private_return = work_target
            .iter()
            .find(|target| target.as_str() != "%private_return")
            .expect("shared return edge receives a new private return");
        assert_ne!(private_return, "%shared_return");
        assert!(blocks.iter().any(|block| {
            block.name == *private_return && block.role == BlockRole::TerminalExitReturn
        }));
        for predecessor in ["%merge", "%external"] {
            assert_eq!(
                block_successors(
                    blocks
                        .iter()
                        .find(|block| block.name == predecessor)
                        .unwrap()
                ),
                vec!["%shared_return".to_string()]
            );
        }
    }

    #[test]
    fn construct_tree_refunnels_bypassed_phi_passthrough_merge() {
        let mut blocks = vec![
            bb(
                "%entry",
                &["br i1 %outside, label %header, label %external"],
            ),
            bb("%header", &["br i1 %c, label %old, label %body"]),
            bb_role("%old", BlockRole::LMerge, &["br label %join"]),
            bb("%body", &["br label %join"]),
            bb("%external", &["br label %join"]),
            bb(
                "%join",
                &[
                    "%value = phi i32 [ 1, %old ], [ 2, %body ], [ 3, %external ]",
                    "ret void",
                ],
            ),
        ];
        let mut merges = HashMap::from([("%header".to_string(), "%old".to_string())]);
        let mut counter = 0usize;

        assert!(repair_construct_tree_bypassed_passthrough_merges(
            &mut blocks,
            &mut merges,
            &mut counter,
        ));
        let private = merges
            .get("%header")
            .expect("header receives the refunnelled private merge");
        assert_ne!(private, "%old");
        for predecessor in ["%old", "%body"] {
            assert_eq!(
                block_successors(
                    blocks
                        .iter()
                        .find(|block| block.name == predecessor)
                        .unwrap()
                ),
                vec![private.clone()]
            );
        }
        assert_eq!(
            block_successors(
                blocks
                    .iter()
                    .find(|block| block.name == "%external")
                    .unwrap()
            ),
            vec!["%join".to_string()]
        );
        let join_phi = blocks
            .iter()
            .find(|block| block.name == "%join")
            .and_then(|block| block.typed.as_ref())
            .and_then(|typed| typed.insts.first())
            .and_then(|instruction| instruction.phi_incoming().as_ref())
            .expect("join phi retained");
        assert!(join_phi
            .1
            .iter()
            .any(|(_, predecessor)| predecessor == private));
        assert!(join_phi
            .1
            .iter()
            .any(|(_, predecessor)| predecessor == "%external"));
    }

    #[test]
    fn construct_tree_refunnels_bypassed_passthrough_chain() {
        let mut blocks = vec![
            bb(
                "%entry",
                &["br i1 %outside, label %header, label %external"],
            ),
            bb("%header", &["br i1 %c, label %old, label %body"]),
            bb_role("%old", BlockRole::LMerge, &["br label %middle"]),
            bb_role("%middle", BlockRole::LMerge, &["br label %join"]),
            bb("%body", &["br label %join"]),
            bb("%external", &["br label %join"]),
            bb(
                "%join",
                &[
                    "%value = phi i32 [ 1, %middle ], [ 2, %body ], [ 3, %external ]",
                    "ret void",
                ],
            ),
        ];
        let mut merges = HashMap::from([("%header".to_string(), "%old".to_string())]);
        let mut counter = 0usize;

        assert!(repair_construct_tree_bypassed_passthrough_merges(
            &mut blocks,
            &mut merges,
            &mut counter,
        ));
        let private = merges.get("%header").unwrap();
        for predecessor in ["%middle", "%body"] {
            assert_eq!(
                block_successors(
                    blocks
                        .iter()
                        .find(|block| block.name == predecessor)
                        .unwrap()
                ),
                vec![private.clone()]
            );
        }
        assert_eq!(
            block_successors(blocks.iter().find(|block| block.name == "%old").unwrap()),
            vec!["%middle".to_string()]
        );
    }

    #[test]
    fn construct_tree_phi_split_reclaims_route_predecessors() {
        let mut blocks = vec![
            bb(
                "%entry",
                &["br i1 %outside, label %header, label %external"],
            ),
            bb("%header", &["br i1 %c, label %left, label %right"]),
            bb("%left", &["br label %left_route"]),
            bb("%right", &["br label %right_route"]),
            bb("%external", &["br label %left_route"]),
            bb_role(
                "%left_route",
                BlockRole::ConstructTreeRoute,
                &["br label %join"],
            ),
            bb_role(
                "%right_route",
                BlockRole::ConstructTreeRoute,
                &["br label %join"],
            ),
            bb(
                "%join",
                &[
                    "%value = phi i32 [ 1, %left_route ], [ 2, %right_route ]",
                    "ret void",
                ],
            ),
        ];
        let mut counter = 0usize;

        let private = synth_unique_selection_merge_phi_explicit(
            &mut blocks,
            &["%left".to_string(), "%right".to_string()],
            "%join",
            &HashSet::from(["%left_route".to_string(), "%right_route".to_string()]),
            &mut counter,
        )
        .expect("owned route predecessors receive a phi-aware private merge");

        for predecessor in ["%left", "%right"] {
            assert_eq!(
                block_successors(
                    blocks
                        .iter()
                        .find(|block| block.name == predecessor)
                        .unwrap()
                ),
                vec![private.clone()]
            );
        }
        assert_eq!(
            block_successors(
                blocks
                    .iter()
                    .find(|block| block.name == "%external")
                    .unwrap()
            ),
            vec!["%left_route".to_string()]
        );
        let private_phi = blocks
            .iter()
            .find(|block| block.name == private)
            .and_then(|block| block.typed.as_ref())
            .and_then(|typed| typed.insts.first())
            .and_then(|instruction| instruction.phi_incoming().as_ref())
            .expect("private merge carries the routed values");
        assert_eq!(
            private_phi
                .1
                .iter()
                .map(|(_, predecessor)| predecessor.as_str())
                .collect::<HashSet<_>>(),
            HashSet::from(["%left", "%right"])
        );
        let join_phi = blocks
            .iter()
            .find(|block| block.name == "%join")
            .and_then(|block| block.typed.as_ref())
            .and_then(|typed| typed.insts.first())
            .and_then(|instruction| instruction.phi_incoming().as_ref())
            .expect("join phi retained");
        assert_eq!(
            join_phi
                .1
                .iter()
                .map(|(_, predecessor)| predecessor.as_str())
                .collect::<HashSet<_>>(),
            HashSet::from(["%left_route", private.as_str()])
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
        let mut cursor = private.clone();
        let mut seen = HashSet::new();
        while cursor != "%old" {
            assert!(
                seen.insert(cursor.clone()),
                "merge chain cycles at {cursor}"
            );
            assert!(
                cursor.starts_with(format!("{SPLIT_PREFIX}{SEL_TOKEN}").as_str()),
                "unexpected non-selection block in merge chain: {cursor}"
            );
            let block = out.iter().find(|block| block.name == cursor).unwrap();
            let successors = block_successors(block);
            let [next] = successors.as_slice() else {
                panic!("merge {cursor} must have one successor");
            };
            cursor = next.clone();
        }
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
        let private = plan
            .branch_merges_by_header
            .get("%inner")
            .expect("inner sibling-arm exit receives a private merge");
        assert!(
            private.starts_with(format!("{SPLIT_PREFIX}{SEL_TOKEN}").as_str()),
            "private merge should be a selection split, got {private}"
        );
        assert_eq!(f, *private);
        assert_eq!(
            block_successors(
                plan.blocks
                    .iter()
                    .find(|block| block.name == *private)
                    .expect("private merge retained")
            ),
            vec!["%sibling".to_string()]
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
        assert_eq!(f.as_str(), "%work");
        let private = plan
            .branch_merges_by_header
            .get("%inner")
            .expect("deep enclosing-region exit receives a private merge");
        assert!(
            private.starts_with(format!("{SPLIT_PREFIX}{SEL_TOKEN}").as_str()),
            "private merge should be a selection split, got {private}"
        );
        assert_eq!(t, *private);
        assert_eq!(
            block_successors(
                plan.blocks
                    .iter()
                    .find(|block| block.name == *private)
                    .expect("private merge retained")
            ),
            vec!["%exit".to_string()]
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

    #[test]
    fn ordinary_planner_constructs_deep_enclosing_selection_boundary_once() {
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

        let (out, branch, by_header, _) =
            unique_selection_merges_with_loop_exit(&blocks, &HashMap::new(), false, false);
        let private = by_header
            .get("%inner")
            .expect("deep enclosing-region exit receives its boundary during construction");
        assert!(private.starts_with(format!("{SPLIT_PREFIX}{SEL_TOKEN}").as_str()));
        let inner = out.iter().find(|block| block.name == "%inner").unwrap();
        let arms = conditional_branch_targets(inner).expect("inner remains conditional");
        assert_eq!(branch.get(&arms), Some(private));
        assert_eq!(
            block_successors(out.iter().find(|block| block.name == "%exit").unwrap()),
            vec![private.clone()]
        );
        let private_successors =
            block_successors(out.iter().find(|block| block.name == *private).unwrap());
        let [enclosing_boundary] = private_successors.as_slice() else {
            panic!("private boundary must have one enclosing successor");
        };
        assert!(enclosing_boundary.starts_with(format!("{SPLIT_PREFIX}{SEL_TOKEN}").as_str()));
        assert_ne!(enclosing_boundary, private);
    }

    #[test]
    fn ordinary_enclosing_boundary_declines_multiple_distinct_escapes() {
        let blocks = vec![
            bb("%outer", &["br i1 %a, label %sibling, label %inner"]),
            bb("%inner", &["br i1 %b, label %left, label %right"]),
            bb("%left", &["br label %left_join"]),
            bb("%right", &["br label %right_join"]),
            bb(
                "%sibling",
                &["br i1 %c, label %left_join, label %right_join"],
            ),
            bb("%left_join", &["br label %outer_merge"]),
            bb("%right_join", &["br label %outer_merge"]),
            bb("%outer_merge", &["ret void"]),
        ];
        let forest = analyze(&blocks);
        let source_merges = HashMap::from([
            ("%outer".to_string(), "%outer_merge".to_string()),
            ("%inner".to_string(), "%outer_merge".to_string()),
            ("%sibling".to_string(), "%outer_merge".to_string()),
        ]);

        assert!(enclosing_selection_region_exit_target(
            &blocks,
            &forest,
            &HashMap::new(),
            &source_merges,
            "%inner",
            "%left",
            "%right",
            Some("%outer_merge"),
        )
        .is_some());
        assert_eq!(
            ordinary_selection_enclosing_boundary_target(
                &blocks,
                &forest,
                &HashMap::new(),
                &source_merges,
                "%inner",
                "%outer_merge",
            ),
            None,
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

    /// A value defined only on one loop-exit path no longer dominates its original target after the
    /// two exits are funnelled through a common dispatch. The constructor must carry that live-in in
    /// the dispatch itself and retarget the ordinary consumer, rather than relying on a later SSA
    /// repair or whole-function relooper.
    #[test]
    fn multiple_exits_carry_path_local_values_through_dispatch() {
        let blocks = vec![
            bb("%entry", &["br label %head"]),
            bb("%head", &["br i1 %c0, label %body, label %exitA"]),
            bb(
                "%body",
                &[
                    "%path = sext i32 %index to i64",
                    "br i1 %c1, label %exitB, label %latch",
                ],
            ),
            bb("%latch", &["br label %head"]),
            bb("%exitA", &["ret void"]),
            bb(
                "%exitB",
                &[
                    "%address = getelementptr i32, ptr %base, i64 %path",
                    "ret void",
                ],
            ),
        ];

        let (out, merges) = forest_loop_merges(&blocks, false, false);
        let dispatch = out
            .iter()
            .find(|block| block.name == merges["%head"].merge)
            .expect("dispatch merge");
        let live_phi = carrier_phis(dispatch)
            .into_iter()
            .find(|(result, incoming)| {
                result.starts_with("%metal2vulkan.exitlive.")
                    && incoming.iter().any(|(value, predecessor)| {
                        val_str(value) == "%path" && predecessor == "%body"
                    })
            })
            .expect("path-local value phi");
        assert!(live_phi
            .1
            .iter()
            .any(|(value, predecessor)| val_str(value) == "undef" && predecessor == "%head"));

        let exit = out.iter().find(|block| block.name == "%exitB").unwrap();
        let mut uses = Vec::new();
        exit.typed.as_ref().unwrap().insts[0].visit_uses(|name| uses.push(name.to_string()));
        assert!(uses.contains(&live_phi.0.to_string()));
        assert!(!uses.contains(&"%path".to_string()));
    }

    /// Bare trap arms terminate inside the loop construct; they are not live continuation targets
    /// and therefore must not turn a one-merge loop into an artificial multi-exit loop.
    #[test]
    fn unreachable_trap_arms_do_not_compete_for_the_loop_merge() {
        let blocks = vec![
            bb("%entry", &["br label %head"]),
            bb("%head", &["br i1 %c0, label %body, label %exit"]),
            bb("%body", &["br i1 %c1, label %work, label %trap0"]),
            bb("%work", &["br i1 %c2, label %latch, label %trap1"]),
            bb("%latch", &["br label %head"]),
            bb("%trap0", &["unreachable"]),
            bb("%trap1", &["unreachable"]),
            bb("%exit", &["ret void"]),
        ];

        let raw = analyze(&blocks);
        let loop_info = raw.loop_for_header("%head").unwrap();
        assert_eq!(loop_info.exits.len(), 3, "raw CFG inventory stays complete");

        let (out, merges) = forest_loop_merges(&blocks, false, false);
        assert_eq!(out.len(), blocks.len(), "no dispatch funnel is needed");
        let merge = merges.get("%head").expect("loop is directly covered");
        assert_eq!(merge.merge, "%exit");
        assert_eq!(merge.continue_target, "%latch");
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

    /// An inner multi-exit loop can synthesize a dispatch edge that is itself an exit of its
    /// enclosing loop. Loop planning must recompute the outer body after materializing the inner
    /// dispatch, so that new edge is routed through the outer merge rather than bypassing it.
    #[test]
    fn nested_multi_exit_dispatch_is_owned_by_recomputed_outer_loop() {
        let blocks = vec![
            bb("%entry", &["br label %outer"]),
            bb(
                "%outer",
                &["br i1 %enter, label %outer.latch, label %inner"],
            ),
            bb(
                "%inner",
                &["br i1 %leave.inner, label %exit.a, label %inner.body"],
            ),
            bb(
                "%inner.body",
                &["br i1 %finish.inner, label %outer.latch, label %inner"],
            ),
            bb(
                "%outer.latch",
                &["br i1 %leave.outer, label %exit.b, label %outer"],
            ),
            bb("%exit.a", &["ret void"]),
            bb("%exit.b", &["ret void"]),
        ];

        let (planned, merges) = forest_loop_merges(&blocks, false, false);
        let inner = merges.get("%inner").expect("inner loop");
        let outer = merges.get("%outer").expect("outer loop");
        let inner_dispatch = planned
            .iter()
            .find(|block| block.name == inner.merge)
            .expect("inner dispatch");
        let outer_dispatch = planned
            .iter()
            .find(|block| block.name == outer.merge)
            .expect("outer dispatch");

        let inner_targets = block_successors(inner_dispatch);
        assert!(inner_targets.contains(&outer.merge));
        assert!(inner_targets.contains(&"%outer.latch".to_string()));
        assert!(!inner_targets.contains(&"%exit.a".to_string()));
        let outer_targets = block_successors(outer_dispatch);
        assert_eq!(
            outer_targets.into_iter().collect::<HashSet<_>>(),
            HashSet::from(["%exit.a".to_string(), "%exit.b".to_string()])
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
            .and_then(|i| i.phi_incoming().as_ref())
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
            .and_then(|i| i.phi_incoming().as_ref())
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
