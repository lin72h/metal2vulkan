//! What a translation costs must grow with the size of a function, not with its square.
//!
//! 500 MiB and 20 seconds per translation attempt are hard budgets (`AGENTS.md`), and neither gets
//! broken by a leak or a slow loop body. Both get broken by a quadratic term that a small graph
//! cannot show: every individual translation is correct, and then a large shader dies at the worker
//! boundary. Two have shipped, and they are the two different quadratics an analysis can be:
//!
//! * **The representation.** The nesting structurizer decided reducibility from a dominator *set*
//!   per block — on a 6301-block function, about 40 million set entries live at once.
//! * **The repetition.** The merge-ownership pass re-derived the whole function's dominance after
//!   each split it made, once per construct. Each table is linear and short-lived, so the footprint
//!   never looks wrong; the wall clock does.
//!
//! So this measures the shape of the growth rather than a byte count, and it measures it twice:
//! against the high-water mark, which sees the first kind, and against the total bytes handed out,
//! which sees the second. Both ratios are properties of the code, not of the machine — the
//! allocator below counts bytes the translation asked for, which is the same number on any host.
//! Each defect is unmistakable in its own number: the dominator-set representation measures 3.85x
//! per doubling against 2.00x for the tree, and the per-split re-derivation measures 3.71x of total
//! allocation per doubling against 1.95x once it is recorded instead.
//!
//! The single `#[test]` is deliberate. The counters are process-wide, so a second test running
//! concurrently in this binary would be measured into the first one's.

use metal2vulkan::passes::Stage;
use metal2vulkan::translate_sanitized_native;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Live bytes, the high-water mark since it was last reset, and every byte ever handed out.
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static HANDED_OUT: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
            HANDED_OUT.fetch_add(layout.size(), Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let grown = unsafe { System.realloc(pointer, layout, new_size) };
        if !grown.is_null() {
            let live = LIVE
                .fetch_add(new_size, Ordering::Relaxed)
                .saturating_add(new_size)
                .saturating_sub(layout.size());
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            PEAK.fetch_max(live, Ordering::Relaxed);
            HANDED_OUT.fetch_add(new_size.saturating_sub(layout.size()), Ordering::Relaxed);
        }
        grown
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// What the allocator saw while `work` ran.
struct Cost {
    /// Peak bytes live at any point, above what was already live when `work` started.
    peak: usize,
    /// Bytes of fresh memory handed out over the whole run, whether or not they were held at once.
    handed_out: usize,
}

fn cost_of<T>(work: impl FnOnce() -> T) -> (T, Cost) {
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let handed_out_before = HANDED_OUT.load(Ordering::Relaxed);
    let value = work();
    let cost = Cost {
        peak: PEAK.load(Ordering::Relaxed).saturating_sub(before),
        handed_out: HANDED_OUT
            .load(Ordering::Relaxed)
            .saturating_sub(handed_out_before),
    };
    (value, cost)
}

/// A kernel of `groups` chained irreducible pairs: `%a` and `%b` each branch into the other, so
/// neither heads a natural loop and no shape tree can nest them.
///
/// Irreducible on purpose. It is the shape that reaches the nesting structurizer's reducibility
/// question — the one that used to carry the quadratic table — and it also forces the whole
/// function onto the CFG-construction path rather than the ordinary structured planner, so the
/// analyses this test is about actually run. Each group loads a distinct buffer word so the
/// conditions cannot be folded away and the CFG survives to the planner intact.
fn irreducible_chain(groups: usize) -> String {
    let mut out = String::from(
        r#"target triple = "air64_v28-apple-macosx26.5.0"

%Words = type { [1024 x i32] }

define void @k(ptr addrspace(1) %out) {
entry:
  br label %g0
"#,
    );
    for group in 0..groups {
        let word = group % 1024;
        out.push_str(&format!(
            "g{group}:
  %p{group} = getelementptr inbounds %Words, ptr addrspace(1) %out, i64 0, i32 0, i64 {word}
  %v{group} = load i32, ptr addrspace(1) %p{group}
  %c{group} = icmp sgt i32 %v{group}, 0
  br i1 %c{group}, label %a{group}, label %b{group}
a{group}:
  %av{group} = add i32 %v{group}, 1
  %ac{group} = icmp sgt i32 %av{group}, 3
  br i1 %ac{group}, label %b{group}, label %g{next}
b{group}:
  %bv{group} = add i32 %v{group}, 2
  %bc{group} = icmp sgt i32 %bv{group}, 5
  br i1 %bc{group}, label %a{group}, label %g{next}
",
            next = group + 1
        ));
    }
    out.push_str(&format!(
        r#"g{groups}:
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

/// Groups in the smaller of the two functions. Three blocks per group plus the entry and the exit,
/// so 100 groups is a 302-block function and 200 is a 602-block one — large enough that a quadratic
/// term separates cleanly from the linear one, small enough to translate in a debug build.
const SMALL: usize = 100;

/// The most the peak may grow for a doubled function before the growth stops being linear.
///
/// Linear growth doubles, and the measurement is 2.00x. Putting the dominator-set representation
/// back measures 3.85x on the same two functions. This sits between them with room on both sides.
const MAX_PEAK_GROWTH: f64 = 2.6;

/// The most the total allocation may grow for a doubled function.
///
/// A bounded peak is not a bounded translation. An analysis that allocates a table proportional to
/// the whole function, runs, and frees it holds a linear peak however many times it repeats -- and
/// repeating it once per construct is exactly how the 20-second ceiling gets broken. Total bytes
/// handed out sees that where a high-water mark cannot: it is the work, not the footprint.
///
/// Linear growth doubles, and the measurement is 1.95x. Restoring the per-split re-derivation
/// measures 3.71x on the same two functions. This sits between them with room on both sides.
const MAX_WORK_GROWTH: f64 = 2.6;

#[test]
fn translation_cost_grows_linearly_with_the_block_count() {
    let scratch = std::env::temp_dir().join(format!("m2v_bounded_growth_{}", std::process::id()));
    std::fs::create_dir_all(&scratch).expect("scratch directory");

    let small = irreducible_chain(SMALL);
    let large = irreducible_chain(SMALL * 2);
    // Translate each once before measuring: the first translation in the process pays for lazily
    // built one-off tables, and that constant would otherwise land on whichever ran first.
    for source in [&small, &large] {
        translate_sanitized_native(source, Stage::Kernel, &scratch)
            .expect("the generated kernel translates");
    }

    let (_, small_cost) = cost_of(|| {
        translate_sanitized_native(&small, Stage::Kernel, &scratch)
            .expect("the smaller kernel translates")
    });
    let (_, large_cost) = cost_of(|| {
        translate_sanitized_native(&large, Stage::Kernel, &scratch)
            .expect("the doubled kernel translates")
    });
    let _ = std::fs::remove_dir_all(&scratch);

    assert!(
        small_cost.peak > 0 && large_cost.peak > 0,
        "the allocator measured nothing: {} and {} bytes",
        small_cost.peak,
        large_cost.peak
    );
    let peak_growth = large_cost.peak as f64 / small_cost.peak as f64;
    assert!(
        peak_growth <= MAX_PEAK_GROWTH,
        "doubling the function multiplied peak translation memory by {peak_growth:.2} \
         ({} bytes at {SMALL} groups, {} bytes at {}); some analysis is \
         quadratic in the block count, which is how the 500 MiB per-translation budget gets broken",
        small_cost.peak,
        large_cost.peak,
        SMALL * 2
    );

    let work_growth = large_cost.handed_out as f64 / small_cost.handed_out as f64;
    assert!(
        work_growth <= MAX_WORK_GROWTH,
        "doubling the function multiplied total translation allocation by {work_growth:.2} \
         ({} bytes at {SMALL} groups, {} bytes at {}); an analysis of the whole function is being \
         re-run once per construct, which is how the 20-second per-translation ceiling gets broken",
        small_cost.handed_out,
        large_cost.handed_out,
        SMALL * 2
    );
}
