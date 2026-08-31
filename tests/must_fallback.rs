//! Negative / must-FALLBACK suite (refactor safety net T4).
//!
//! The crate's floor-safety argument is that unsupported inputs FALLBACK *cleanly* — a
//! `translate_sanitized_native` `Err`, never wrong-but-valid SPIR-V that a downstream gate would
//! wave through. These tests pin that behaviour for the known-unsupported classes so a refactor
//! (especially S23, panic→Result) can't silently turn a clean FALLBACK into a translate that
//! "succeeds" with garbage — or into a process abort.
//!
//! Covers known-unsupported classes: `air.intersect.*` raytracing, `llvm.agx3.*` emask
//! intrinsics, texture atomics — plus structural malformations (no definitions, truncated).
//!
//! Two of these are the classes that actually turn up. Measured over a 2880-source local corpus
//! sample, 204 sources do not translate, and 191 of them are one of two shapes: a call through a
//! Metal visible function table, or an intersection whose custom intersection functions come from a
//! function buffer. Both are function pointers, which Logical SPIR-V has none of — no lowering is
//! coming, so what matters is that the rejection stays a rejection. Each is pinned below on the
//! exact diagnostic those corpus sources produce.

use metal2vulkan::passes::Stage;
use metal2vulkan::translate_sanitized_native;
use std::env;
use std::path::PathBuf;

fn tmp() -> PathBuf {
    let d = env::temp_dir().join(format!("m2v_must_fallback_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

/// Assert the input FALLBACKs cleanly (Err) and the diagnostic mentions `needle` — i.e. it is the
/// *intended* rejection, not an unrelated parse failure.
fn assert_fallback(ll: &str, needle: &str) {
    match translate_sanitized_native(ll, Stage::Kernel, &tmp()) {
        Ok(spv) => panic!(
            "expected a clean FALLBACK (Err) but translate succeeded ({} bytes); \
             wrong-but-valid SPIR-V defeats the floor-safety guarantee",
            spv.len()
        ),
        Err(e) => assert!(
            e.contains(needle),
            "FALLBACK diagnostic should mention {needle:?}; got: {e}"
        ),
    }
}

// A minimal, genuinely-translatable kernel; each negative case injects exactly one unsupported
// construct into `%OP` so the FALLBACK is attributable to that construct and nothing else.
const HEAD: &str = r#"
target triple = "air64_v28-apple-macosx26.5.0"

%Input = type { [4 x i32] }
%Output = type { [4 x i32] }

define void @k(ptr addrspace(2) %in, ptr addrspace(1) %out) {
entry:
  %a0p = getelementptr inbounds %Input, ptr addrspace(2) %in, i64 0, i32 0, i64 0
  %a0 = load i32, ptr addrspace(2) %a0p
"#;

const TAIL: &str = r#"
  %o0 = getelementptr inbounds %Output, ptr addrspace(1) %out, i64 0, i32 0, i64 0
  store i32 %r, ptr addrspace(1) %o0
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 16, !"air.struct_type_info", !5, !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"Input", !"air.arg_name", !"in"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 16, !"air.struct_type_info", !5, !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.arg_type_name", !"Output", !"air.arg_name", !"out"}
!5 = !{i32 0, i32 16, i32 0, !"uint", !"v0", i32 4, i32 4, i32 0, !"uint", !"v1", i32 8, i32 4, i32 0, !"uint", !"v2", i32 12, i32 4, i32 0, !"uint", !"v3"}
"#;

fn kernel_with(op: &str) -> String {
    format!("{HEAD}{op}{TAIL}")
}

/// As [`kernel_with`], for a construct that also needs its callee declared.
fn kernel_with_declarations(op: &str, declarations: &str) -> String {
    format!("{HEAD}{op}{TAIL}{declarations}")
}

/// Sanity anchor: the base kernel (op = a plain add) DOES translate, so each negative below is
/// attributable to its injected construct rather than a broken template.
#[test]
fn base_kernel_translates() {
    let base = kernel_with("  %r = add i32 %a0, %a0\n");
    assert!(
        translate_sanitized_native(&base, Stage::Kernel, &tmp()).is_ok(),
        "the negative-suite base template must itself translate"
    );
}

#[test]
fn raytracing_intersect_intrinsic_fallbacks() {
    let ll = kernel_with("  %r = call i32 @air.intersect.f32.i32(i32 %a0, i32 %a0, i32 %a0)\n");
    assert_fallback(&ll, "@air.intersect.");
}

#[test]
fn agx3_emask_intrinsic_fallbacks() {
    let ll = kernel_with("  %r = call i32 @llvm.agx3.emask.i32(i32 %a0)\n");
    assert_fallback(&ll, "@llvm.agx3.");
}

#[test]
fn texture_atomic_fallbacks() {
    let ll = kernel_with(
        "  %r = call i32 @air.atomic_fetch_add.explicit.texture.2d.i32(i32 %a0, i32 %a0)\n",
    );
    assert_fallback(&ll, "@air.atomic_fetch_add.explicit.texture");
}

/// A call through a Metal visible function table: the callee is an SSA value, not a symbol.
///
/// The largest unsupported class in the corpus sample, 136 sources of the 204. Logical SPIR-V has no
/// function pointers, so this cannot become a lowering; it can only become a wrong one. The
/// diagnostic names the pointer so the author can find the call.
#[test]
fn visible_function_table_call_fallbacks() {
    let ll = kernel_with_declarations(
        "  %fp = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %out, i32 0)\n\
         \x20 %r = call i32 %fp(ptr addrspace(2) %in)\n",
        "declare ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1), i32)\n",
    );
    assert_fallback(
        &ll,
        "unsupported indirect call through function pointer %fp",
    );
}

/// An intersection whose intersection functions come from a function buffer.
///
/// The second largest class, about 55 sources. `air.intersect.*` with a null function table lowers
/// (`ray_intersection.rs`); the `intersection_function_buffer` variant is a dispatch through custom
/// intersection functions, so it is the same function-pointer wall as the case above wearing a
/// raytracing hat. The tag suffix is one of a combinatorial family -- `instancing`, `triangle_data`,
/// `world_space_data`, `user_data`, `primitive_motion`, `instance_motion`,
/// `multi_level_instancing` -- and the rejection is on the `intersection_function_buffer` stem, not
/// on any one combination, which is why one of them stands for all of them here.
#[test]
fn intersection_function_buffer_fallbacks() {
    let signature = "{ i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 }";
    let ll = kernel_with_declarations(
        &format!(
            "  %hit = call {signature} @air.intersect.intersection_function_buffer.triangle_data(\
             <3 x float> zeroinitializer, <3 x float> zeroinitializer, float 0.0, float 1.0, \
             ptr addrspace(1) %out, ptr addrspace(1) %out, i64 0, i64 1, ptr null, i64 0, i32 0, \
             i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 -1, i32 0, i1 false, i1 false)\n\
             \x20 %r = extractvalue {signature} %hit, 0\n"
        ),
        "declare { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } \
         @air.intersect.intersection_function_buffer.triangle_data(<3 x float>, <3 x float>, float, \
         float, ptr addrspace(1), ptr addrspace(1), i64, i64, ptr, i64, i32, i32, i32, i32, i32, \
         i32, i32, i32, i32, i32, i1, i1)\n",
    );
    assert_fallback(&ll, "air.intersect.intersection_function_buffer");
}

#[test]
fn no_function_definitions_fallbacks() {
    assert_fallback(
        "target triple = \"air64_v28-apple-macosx26.5.0\"\n",
        "no function definitions found",
    );
}

#[test]
fn truncated_module_fallbacks() {
    // a `define` block that never closes / never returns
    let ll = "target triple = \"air64_v28-apple-macosx26.5.0\"\n\
              define void @k(ptr %x) {\n\
              entry:\n\
              \x20 %a = load i32, ptr %x\n";
    assert_fallback(ll, "unterminated function");
}
