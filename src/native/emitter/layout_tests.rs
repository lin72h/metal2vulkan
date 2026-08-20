//! Emitter-side layout/width table tests (refactor T3, emitter half). Companion to the
//! `native::ir::layout_abi_tests` table (which pins the `LlModule` size/align calculators). The
//! emitter carries three more calculators that the `LlModule` table cannot reach — they need an
//! `Emitter` (module context / `resolve_type`) rather than an `LlModule`:
//!
//! - `Emitter::raw_type_size_align` → the "Raw" `LayoutRule` (LLVM vector allocation size and
//!   source ABI alignment, arrays floor align at 4; a `Result` that ERRORS — not `None` — on
//!   odd-width ints / `Void`).
//! - `bitcast_width` / `Emitter::vector_total_bits` → the bit-width helpers used to decide when
//!   two types are a byte-identical `OpBitcast` reinterpret. These are NOT size/align calculators
//!   (they return bit counts, not `(size, align)`), so they are pinned here but are not folded
//!   into the crate-level `layout` oracle.
//!
//! S4 migrates `raw_type_size_align`'s body into `crate::layout`; these tests target the emitter
//! method so they survive that move as its delegation contract.

use super::super::ir::{LlModule, LlType};
use super::helpers::bitcast_width;
use super::Emitter;

fn emitter() -> Emitter {
    let ir = LlModule::parse("define void @k() {\nentry:\n  ret void\n}\n")
        .expect("minimal module parses");
    Emitter::new(ir)
}

fn vec(elem: LlType, lanes: u32) -> LlType {
    LlType::Vector(Box::new(elem), lanes)
}

#[test]
fn raw_rule_pads_vec3_and_floors_array_align() {
    // "Raw" (`raw_type_size_align`): same padded shape as the Memcpy rule (vec3 = 16/16), a
    // `Result` return, and explicit standard-width int buckets.
    let e = emitter();
    let raw = |ty: &LlType| e.raw_type_size_align(ty).expect("covered type");
    // scalars
    assert_eq!(raw(&LlType::Bool), (1, 1));
    assert_eq!(raw(&LlType::Int(8)), (1, 1));
    assert_eq!(raw(&LlType::Int(16)), (2, 2));
    assert_eq!(raw(&LlType::Half), (2, 2));
    assert_eq!(raw(&LlType::BFloat), (2, 2));
    assert_eq!(raw(&LlType::Int(32)), (4, 4));
    assert_eq!(raw(&LlType::Float), (4, 4));
    assert_eq!(raw(&LlType::Int(64)), (8, 8));
    assert_eq!(raw(&LlType::Ptr(1)), (8, 8));
    // vectors: vec3 pads to 4 lanes and self-aligns
    assert_eq!(raw(&vec(LlType::Float, 2)), (8, 8));
    assert_eq!(raw(&vec(LlType::Float, 3)), (16, 16));
    assert_eq!(raw(&vec(LlType::Float, 4)), (16, 16));
    // array of scalars floors align at 4; array of vec3 strides at the padded 16
    assert_eq!(raw(&LlType::Array(Box::new(LlType::Float), 3)), (12, 4));
    assert_eq!(
        raw(&LlType::Array(Box::new(vec(LlType::Float, 3)), 2)),
        (32, 16)
    );
    // struct { i8, float }: i8@0, float@4 (align 4), total 8
    assert_eq!(
        raw(&LlType::Struct(vec![LlType::Int(8), LlType::Float])),
        (8, 4)
    );
}

#[test]
fn raw_rule_errors_on_uncovered_types() {
    // The Raw rule is strict: odd-width ints and `Void` are an `Err`, not a silent default. (The
    // Memcpy rule, by contrast, gives a generic answer for odd ints — the intentional divergence.)
    let e = emitter();
    assert!(e.raw_type_size_align(&LlType::Int(24)).is_err());
    assert!(e.raw_type_size_align(&LlType::Void).is_err());
}

#[test]
fn raw_rule_uses_source_vector_abi_alignment() {
    let mut e = emitter();
    e.air_data_layout = Some(
        crate::layout::AirDataLayout::parse("e-v24:64:64").expect("parse custom vector alignment"),
    );

    assert_eq!(
        e.raw_type_size_align(&vec(LlType::Int(8), 3))
            .expect("covered vector"),
        (8, 8)
    );
    assert_eq!(
        e.raw_type_size_align(&LlType::Struct(vec![
            vec(LlType::Int(8), 3),
            LlType::Int(8),
        ]))
        .expect("covered struct"),
        (16, 8)
    );
}

#[test]
fn bitcast_width_is_total_scalar_bits() {
    // `bitcast_width`: total bit count for bitcast-equivalence. Does NOT resolve named types and
    // does NOT fold `Int(1)`→`Bool` (that is a `resolve_type` behaviour, upstream of this helper).
    assert_eq!(bitcast_width(&LlType::Float), Some(32));
    assert_eq!(bitcast_width(&LlType::Int(32)), Some(32));
    assert_eq!(bitcast_width(&LlType::Half), Some(16));
    assert_eq!(bitcast_width(&LlType::BFloat), Some(16));
    assert_eq!(bitcast_width(&LlType::Int(16)), Some(16));
    assert_eq!(bitcast_width(&LlType::Int(8)), Some(8));
    assert_eq!(bitcast_width(&LlType::Int(1)), Some(1));
    assert_eq!(bitcast_width(&LlType::Int(64)), Some(64));
    // vectors multiply through
    assert_eq!(bitcast_width(&vec(LlType::Float, 4)), Some(128));
    assert_eq!(bitcast_width(&vec(LlType::Half, 2)), Some(32));
    // non-bitcastable leaves
    assert_eq!(bitcast_width(&LlType::Bool), None);
    assert_eq!(bitcast_width(&LlType::Ptr(1)), None);
    assert_eq!(
        bitcast_width(&LlType::Array(Box::new(LlType::Float), 4)),
        None
    );
    assert_eq!(
        bitcast_width(&LlType::Struct(vec![LlType::Float, LlType::Float])),
        None
    );
}

#[test]
fn vector_total_bits_is_vectors_only() {
    // `vector_total_bits`: resolves the type, then returns total bits for a VECTOR only (`None`
    // otherwise). A `Vector(_, 1)` collapses to its scalar during resolution, so it is not a vector.
    let e = emitter();
    assert_eq!(e.vector_total_bits(&vec(LlType::Float, 4)), Some(128));
    assert_eq!(e.vector_total_bits(&vec(LlType::Half, 3)), Some(48));
    assert_eq!(e.vector_total_bits(&LlType::Float), None);
    assert_eq!(e.vector_total_bits(&vec(LlType::Float, 1)), None);
}
