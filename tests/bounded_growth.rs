//! Translation memory must grow with the size of a function, not with its square.
//!
//! 500 MiB per translation attempt is a hard budget (`AGENTS.md`), and the way that budget gets
//! broken is never a leak — it is an analysis whose representation is quadratic in the block count.
//! One such representation shipped: the nesting structurizer decided reducibility from a dominator
//! *set* per block, which on a 6301-block function was about 40 million set entries. Nothing about
//! that is visible on a small graph, and every individual translation is correct; the only symptom
//! is that a large shader dies at the worker boundary.
//!
//! So this measures the shape of the growth rather than a byte count. Doubling the function must
//! not much more than double the memory. That ratio is a property of the code, not of the machine:
//! the allocator below counts bytes the translation asked for, which is the same number on any
//! host. A quadratic term is unmistakable in it: restoring the dominator-set version behind this
//! graph measures 3.85x per doubling where the tree version measures 2.00x.
//!
//! The single `#[test]` is deliberate. The counter is process-wide, so a second test running
//! concurrently in this binary would be measured into the first one's peak.

use metal2vulkan::passes::Stage;
use metal2vulkan::translate_sanitized_native;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Live bytes, and the high-water mark since it was last reset.
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
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
        }
        grown
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Peak bytes live at any point during `work`, above what was already live when it started.
fn peak_bytes<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let value = work();
    let peak = PEAK.load(Ordering::Relaxed);
    (value, peak.saturating_sub(before))
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
const MAX_GROWTH: f64 = 2.6;

#[test]
fn translation_memory_grows_linearly_with_the_block_count() {
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

    let (_, small_peak) = peak_bytes(|| {
        translate_sanitized_native(&small, Stage::Kernel, &scratch)
            .expect("the smaller kernel translates")
    });
    let (_, large_peak) = peak_bytes(|| {
        translate_sanitized_native(&large, Stage::Kernel, &scratch)
            .expect("the doubled kernel translates")
    });
    let _ = std::fs::remove_dir_all(&scratch);

    assert!(
        small_peak > 0 && large_peak > 0,
        "the allocator measured nothing: {small_peak} and {large_peak} bytes"
    );
    let growth = large_peak as f64 / small_peak as f64;
    assert!(
        growth <= MAX_GROWTH,
        "doubling the function multiplied peak translation memory by {growth:.2} \
         ({small_peak} bytes at {SMALL} groups, {large_peak} bytes at {}); some analysis is \
         quadratic in the block count, which is how the 500 MiB per-translation budget gets broken",
        SMALL * 2
    );
}
