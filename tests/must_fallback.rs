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
//! One of them is a GAP rather than a limit: the byte-view vector store below has a lowering, it
//! just has not been written, and the reason it is pinned is that the obvious ways to write it are
//! wrong. Closing it must be a deliberate change to this test, not a silent pass.
//!
//! One case here is deliberately POSITIVE. A rejection is only pinned from one side: a test that
//! asserts a class FALLBACKs cannot notice a change that starts rejecting inputs that used to
//! translate. The function-constant-gated pair below asserts both directions over a single
//! template, so the boundary itself is what is pinned, not just the far side of it.
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

/// The same visible-function-table call, reached only when a `[[function_constant]]` predicate is
/// true — which pins the *edge* of the rejection above rather than its interior.
///
/// Nothing supplies function-constant values at translate time (`fc_air_specialize` bakes any
/// caller-supplied ones into the AIR before parsing, and the emitted modules carry no
/// `OpSpecConstant` for them), so `air.is_function_constant_defined` folds to `false` and the
/// module's static initializer stores `0` into the gating global. The gated region is then
/// statically dead, gets pruned, and the indirect call never reaches the emitter.
///
/// That fold is load-bearing for real shaders: corpus sources that call through a visible function
/// table inside an off-by-default region translate today *because of it*. It is also fragile —
/// it spans `fold_static_initializer_constants` and `prune_unreachable_function_bodies`
/// (`src/native/ir/static_init.rs`) and depends on the initializer being recognised as one. So the
/// two directions are pinned as a pair over one template that differs in exactly one token, the
/// order of the branch arms:
///
/// - dead side taken → the call is folded away and the kernel translates;
/// - live side taken → the same call FALLBACKs with the same diagnostic as above.
///
/// A regression that stops recognising the initializer turns the first into a FALLBACK; one that
/// prunes too eagerly turns the second into a silent success. Neither is caught by either half
/// alone.
const FC_GATED_VFT: &str = r#"target triple = "air64_v28-apple-macosx26.5.0"

@enabled.MTL_FC_INIT_0_b = internal addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@kEnabled = internal unnamed_addr addrspace(2) global i8 0, align 1

declare i1 @air.is_function_constant_defined(ptr addrspace(2))
declare ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1), i32)

define internal void @_GLOBAL__sub_I_fc() section "air.static_init" {
  %1 = load i8, ptr addrspace(2) @enabled.MTL_FC_INIT_0_b, align 1
  %2 = call i1 @air.is_function_constant_defined(ptr addrspace(2) @enabled.MTL_FC_INIT_0_b)
  %3 = icmp ne i8 %1, 0
  %4 = select i1 %2, i1 %3, i1 false
  %5 = zext i1 %4 to i8
  store i8 %5, ptr addrspace(2) @kEnabled, align 1
  ret void
}

define internal fastcc float @fetch(ptr addrspace(1) %table, ptr addrspace(1) %data) {
  %fp = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 0)
  %r = call float %fp(ptr addrspace(1) %data)
  ret float %r
}

define void @k(ptr addrspace(1) %out, ptr addrspace(1) %table) {
entry:
  %e = load i8, ptr addrspace(2) @kEnabled, align 1
  %c = icmp eq i8 %e, 0
  br i1 %c, label %ARMS

use:
  %v = call fastcc float @fetch(ptr addrspace(1) %table, ptr addrspace(1) %out)
  br label %done

done:
  %r = phi float [ 0.000000e+00, %entry ], [ %v, %use ]
  store float %r, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!air.function_constants = !{!6}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.function_constant", !6, !"air.visible_function_table", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"visible_function_table", !"air.arg_name", !"table"}
!6 = !{ptr addrspace(2) @enabled.MTL_FC_INIT_0_b, !"bool", !"enabled", i32 0, i1 false}
"#;

/// `FC_GATED_VFT` with the branch arms in `arms` order — the sole difference between the two cases.
fn fc_gated_vft(arms: &str) -> String {
    assert!(
        FC_GATED_VFT.contains("label %ARMS"),
        "the branch placeholder must survive edits to the template"
    );
    FC_GATED_VFT.replace("label %ARMS", arms)
}

#[test]
fn function_constant_gated_visible_function_table_call_is_folded_away() {
    let ll = fc_gated_vft("label %done, label %use");
    let spv = translate_sanitized_native(&ll, Stage::Kernel, &tmp()).expect(
        "an off-by-default function constant makes the visible-function-table region dead; \
         folding it is what lets such shaders translate at all",
    );
    assert!(
        !spv.is_empty(),
        "the folded kernel must still emit a module"
    );
}

/// The initializer is recognised by its `air.static_init` section, not by its Itanium name.
///
/// `_GLOBAL__sub_I…` is how clang mangles a translation-unit initializer; `section
/// "air.static_init"` is how AIR declares what the function *is*. Over the corpus the two always
/// travel together, so only an authored pair can tell which one the readers act on — and the answer
/// decides real translations, since the fold above is what makes the gated call disappear. Renaming
/// the function must change nothing; removing the section must stop the fold, leaving the same
/// module with the same dead-side branch to reject on the call it can no longer prune.
#[test]
fn a_static_initializer_is_recognised_by_its_air_section_not_its_name() {
    let dead_side = fc_gated_vft("label %done, label %use");

    let renamed = dead_side.replace("_GLOBAL__sub_I_fc", "air_static_ctor");
    assert_ne!(
        renamed, dead_side,
        "the initializer name must be substituted"
    );
    translate_sanitized_native(&renamed, Stage::Kernel, &tmp())
        .expect("an initializer keeps its meaning when only its name changes");

    let unsectioned = dead_side.replace(" section \"air.static_init\"", "");
    assert_ne!(unsectioned, dead_side, "the section must be removed");
    assert_fallback(
        &unsectioned,
        "unsupported indirect call through function pointer %fp",
    );
}

#[test]
fn live_visible_function_table_call_still_fallbacks_under_a_function_constant() {
    let ll = fc_gated_vft("label %use, label %done");
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

/// A `<3 x float>` store through a byte view of a raw word buffer, reached across a call.
///
/// This one is a GAP, not a limit — unlike the function-pointer classes above, the lowering is
/// expressible; nothing has written it. It is the largest remaining non-function-pointer rejection
/// class in the corpus sample (8 of the 12), and it is pinned here because it is the class most
/// likely to be closed WRONGLY. A `device uchar*` view of a `{ RuntimeArray<uint> }` block writing
/// four, eight or twelve bytes at a runtime offset has two tempting lowerings that are not the AIR's
/// semantics: a non-atomic read-modify-write of the surrounding words, which races another thread
/// writing the neighbouring bytes (`emit_scalar_narrowing_store` restricts exactly that to
/// thread-local slots, and `emit_raw_byte_store_from_u32` uses atomics on shared storage); and a
/// plain word store, which is only byte-exact if the offset is four-byte aligned — the AIR asserts
/// that with `align`, but the alignment does not reach the SPIR-V pass that would need it. Until one
/// of those is resolved honestly, this must stay a clean `Err`.
///
/// Reduced from a corpus source with `llvm-reduce` (708 lines to 44) and then renamed, so it carries
/// no third-party identifiers. Every remaining part is load-bearing: dropping the `i32` load or the
/// byte `getelementptr` in `@k` (which together fix the block's element width at 32 bits and open
/// the byte view), inlining `@store_vec` into `@k`, flattening its pass-through blocks, or shrinking
/// the argument list each make the module translate.
#[test]
fn byte_view_vector_store_into_a_word_block_fallbacks() {
    let ll = r#"target triple = "air64_v28-apple-macosx26.5.0"

%struct.view = type { ptr addrspace(2), ptr addrspace(1) }

define void @k(ptr addrspace(1) %0) {
  %2 = alloca %struct.view, align 8
  %3 = getelementptr %struct.view, ptr %2, i64 0, i32 1
  store ptr addrspace(1) %0, ptr %3, align 8
  %4 = getelementptr i8, ptr addrspace(1) %0, i64 0
  %5 = load i32, ptr addrspace(1) %0, align 16
  call fastcc void @store_vec(ptr %2)
  ret void
}

define fastcc void @store_vec(ptr %0) {
  br label %2

2:                                                ; preds = %1
  br label %4

4:                                                ; preds = %2
  %5 = getelementptr %struct.view, ptr %0, i64 0, i32 1
  %6 = load ptr addrspace(1), ptr %5, align 8
  %7 = zext i32 0 to i64
  %8 = getelementptr i8, ptr addrspace(1) %6, i64 %7
  store <3 x float> zeroinitializer, ptr addrspace(1) %8, align 16
  br label %9

9:                                                ; preds = %4
  ret void
}

!air.kernel = !{!0}

!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !7, !8, !9}
!3 = !{i32 0, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"index"}
!4 = !{i32 1, !"air.buffer", !"air.buffer_size", i32 280, !"air.location_index", i32 4, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 280, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"hdr", !"air.arg_name", !"header"}
!5 = !{!"air.struct_type_info", !6, i32 0, i32 8, i32 35, !"hdr_entry", !"entries"}
!6 = !{i32 0, i32 4, i32 0, !"int", !"offset", i32 4, i32 2, i32 0, !"short", !"type", i32 6, i32 2, i32 0, !"short", !"stride"}
!7 = !{i32 2, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"data"}
!8 = !{i32 3, !"air.buffer", !"air.buffer_size", i32 280, !"air.location_index", i32 6, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !5, !"air.arg_type_size", i32 280, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"hdr", !"air.arg_name", !"header"}
!9 = !{i32 4, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"data"}"#;
    assert_fallback(ll, "no dynamic-struct-index rewrite repaired");
}
