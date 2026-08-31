//! A translation must not re-derive the whole function once per construct.
//!
//! The two hard per-attempt budgets in `AGENTS.md` -- 20 seconds and 500 MiB -- are broken by
//! quadratic terms, not by slow loop bodies, and the two quadratics an analysis can be need
//! different instruments to see. A quadratic *representation* shows up in peak memory, which is what
//! `tests/bounded_growth.rs` watches. A quadratic *repetition* does not: each table is linear and
//! short-lived, so the footprint stays flat while the wall clock goes up. What it does move is the
//! number of times the whole function gets analyzed, and that count is exact, deterministic, and the
//! same on every machine -- so this measures it directly rather than through a proxy.
//!
//! Two of these have shipped. The merge-ownership pass re-derived the function's dominance after
//! every split it made, and the selection structurizer re-derived dominance and the loop forest
//! after each pass-through it inserted. Both now record the one block they added
//! ([`crate::native::cfg::graph::Dominators::record_pass_through`]) instead. The kernel below
//! reaches the structurizer's; the merge-ownership pass's is the one
//! `tests/bounded_growth.rs` catches, in its total-allocation bound.

use crate::native::cfg::graph::cfg_builds_during;
use crate::passes::Stage;

/// A kernel with `groups` selections inside one loop, each able to break out to the loop's exit.
///
/// The exit is therefore the natural merge that all `groups` selection headers claim at once, and
/// the structurizer gives each of them a private merge by splicing a pass-through in front of it.
/// That is the shape whose per-split re-derivation this file exists to keep out: the splits happen
/// one after another on the same block, so the cost of re-deriving is multiplied by the number of
/// selections. Each group loads a distinct buffer word so nothing folds away.
fn loop_exit_selection_chain(groups: usize) -> String {
    let mut out = String::from(
        r#"target triple = "air64_v28-apple-macosx26.5.0"

%Words = type { [1024 x i32] }

define void @k(ptr addrspace(1) %out) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inext, %latch ]
  br label %h0
"#,
    );
    for group in 0..groups {
        let word = group % 1024;
        out.push_str(&format!(
            "h{group}:
  %p{group} = getelementptr inbounds %Words, ptr addrspace(1) %out, i64 0, i32 0, i64 {word}
  %v{group} = load i32, ptr addrspace(1) %p{group}
  %c{group} = icmp sgt i32 %v{group}, 0
  br i1 %c{group}, label %t{group}, label %h{next}
t{group}:
  %tv{group} = add i32 %v{group}, 1
  %tc{group} = icmp sgt i32 %tv{group}, 3
  br i1 %tc{group}, label %exit, label %h{next}
",
            next = group + 1
        ));
    }
    out.push_str(&format!(
        r#"h{groups}:
  br label %latch

latch:
  %inext = add i32 %i, 1
  %done = icmp slt i32 %inext, 16
  br i1 %done, label %loop, label %exit

exit:
  ret void
}}

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3}}
!3 = !{{i32 0, !"air.buffer", !"air.buffer_size", i32 4096, !"air.struct_type_info", !4, !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.arg_type_name", !"Words", !"air.arg_name", !"out"}}
!4 = !{{i32 0, i32 4096, i32 0, !"uint", !"v0"}}
"#
    ));
    out
}

/// Selections in the measured kernel. Two blocks each, plus the entry, loop header, the block that
/// falls out of the chain, the latch and the exit -- seventeen blocks at six.
const GROUPS: usize = 6;

/// The most source CFGs this kernel may be worth.
///
/// Measured at 867. Restoring the structurizer's per-split re-derivation puts it at 2209; this sits
/// between them with room on both sides. The count is deterministic -- the same six numbers come back on every
/// run -- so a failure here is a real change in how much whole-function analysis a translation does,
/// not noise. That is a decision worth making deliberately: raising this bound is fine when a new
/// analysis genuinely earns its place, and is the wrong answer when a maintained result was replaced
/// by a re-derived one.
const MAX_CFG_BUILDS: usize = 1400;

#[test]
fn a_shared_loop_exit_is_not_re_analyzed_once_per_split() {
    let scratch = std::env::temp_dir().join(format!(
        "m2v_bounded_work_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    std::fs::create_dir_all(&scratch).expect("scratch directory");
    let source = loop_exit_selection_chain(GROUPS);
    let (translated, builds) =
        cfg_builds_during(|| crate::translate_sanitized_native(&source, Stage::Kernel, &scratch));
    let _ = std::fs::remove_dir_all(&scratch);

    translated.expect("the generated kernel translates");
    assert!(
        builds <= MAX_CFG_BUILDS,
        "translating a {GROUPS}-selection kernel built {builds} source CFGs (bound {MAX_CFG_BUILDS}); \
         some pass is deriving the whole function again after a graph edit instead of maintaining \
         what it already had, which is how the 20-second per-attempt ceiling gets broken"
    );
}
