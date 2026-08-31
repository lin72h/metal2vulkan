//! An argument-buffer member is reported at the size it occupies, not the size of what it names.
//!
//! Metal spells a handle member with the type it points AT — `!"char"` for a `device char *`,
//! `!"float4x3"` for a `device float4x3 *`, an opaque class name for a `device Foo *` — while the
//! member itself stores an eight-byte handle. The member's `air.indirect_argument` node is what
//! distinguishes the two: every role but `air.indirect_constant` names a resource the buffer
//! references rather than a value it stores inline.
//!
//! Reading the name as the member's storage is silent data corruption, and uniquely hard to
//! notice. AIR states the argument's total size and its member layout as two independent facts, so
//! neither contradicts the other locally; a consumer packs its upload at the reported offsets and
//! the shader reads somewhere else. The emitted SPIR-V does not settle it either, because a buffer
//! whose reconstruction is that far off is represented as raw bytes and declares no struct type to
//! compare against. It shipped: `MTLGenericBVHData` reported a 120-byte layout for its 72-byte
//! argument, putting a 64-byte matrix where the buffer holds an address and shifting the meaning of
//! every member after it.
//!
//! The two ends have to agree, so the sizing test here is that the reported layout reaches exactly
//! as far as `air.arg_type_size` says the argument does.

use metal2vulkan::meta::{AirMember, AirScalar, AirType};
use metal2vulkan::passes::{Stage, TransformOptions};
use metal2vulkan::reflect::ShaderReflection;
use metal2vulkan::translate_sanitized_native_reflected;
use std::path::PathBuf;

/// A kernel whose first argument is a 36-byte `air.indirect_buffer` mixing both member classes.
///
/// Three members are handles named after what they point at — a `device char *`, a
/// `device float4x3 *`, and a sampled texture — and two are stored inline. Every handle names a
/// type whose own footprint (1, 64, and 4 bytes) differs from the eight bytes the member occupies,
/// which is the shape that used to run the layout past the end of the argument.
const ARGUMENT_BUFFER_OF_HANDLES: &str = r#"target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(2) %args, ptr addrspace(1) %out) {
entry:
  %field = getelementptr inbounds i8, ptr addrspace(2) %args, i64 32
  %value = load i32, ptr addrspace(2) %field, align 4
  store i32 %value, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !9}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 36, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 36, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Args", !"air.arg_name", !"args"}
!4 = !{i32 0, i32 8, i32 0, !"char", !"header", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"float4x3", !"transforms", !"air.indirect_argument", !6, i32 16, i32 8, i32 0, !"texture2d<float, sample>", !"albedo", !"air.indirect_argument", !7, i32 24, i32 8, i32 0, !"ulong", !"identifier", !"air.indirect_argument", !8, i32 32, i32 4, i32 0, !"uint", !"count", !"air.indirect_argument", !8}
!5 = !{i32 0, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"header"}
!6 = !{i32 1, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 64, !"air.arg_type_name", !"float4x3", !"air.arg_name", !"transforms"}
!7 = !{i32 2, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"albedo"}
!8 = !{i32 3, !"air.indirect_constant", !"air.location_index", i32 3, i32 1, !"air.arg_type_name", !"ulong", !"air.arg_name", !"identifier"}
!9 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;

/// The same shape one level down: a nested `air.struct_type_info` member sitting between two
/// handles. AIR gave that member a layout of its own, so the layout is what it stores — the
/// `air.indirect_argument` suffix it also carries must not turn it into an address.
const ARGUMENT_BUFFER_WITH_NESTED_STRUCT: &str = r#"target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(2) %args, ptr addrspace(1) %out) {
entry:
  %field = getelementptr inbounds i8, ptr addrspace(2) %args, i64 16
  %value = load i32, ptr addrspace(2) %field, align 4
  store i32 %value, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !9}
!3 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 20, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 20, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Args", !"air.arg_name", !"args"}
!4 = !{i32 0, i32 8, i32 0, !"char", !"data", !"air.indirect_argument", !5, !"air.struct_type_info", !6, i32 8, i32 8, i32 0, !"Bounds", !"bounds", !"air.indirect_argument", !7, i32 16, i32 4, i32 0, !"uint", !"count", !"air.indirect_argument", !7}
!5 = !{i32 0, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_name", !"char", !"air.arg_name", !"data"}
!6 = !{i32 0, i32 4, i32 0, !"float", !"lo", i32 4, i32 4, i32 0, !"float", !"hi"}
!7 = !{i32 2, !"air.indirect_constant", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"count"}
!9 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;

/// An ordinary device buffer whose struct is entirely inline. No member carries an
/// `air.indirect_argument` suffix, so every one of them keeps the type AIR named for it — including
/// the `float3` that occupies a four-lane slot and the two-byte tail that leaves the argument's
/// declared size unrounded.
const INLINE_STRUCT_BUFFER: &str = r#"target triple = "spirv-unknown-vulkan1.2"

define void @k(ptr addrspace(2) %params, ptr addrspace(1) %out) {
entry:
  %field = getelementptr inbounds i8, ptr addrspace(2) %params, i64 16
  %value = load i32, ptr addrspace(2) %field, align 4
  store i32 %value, ptr addrspace(1) %out, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !5}
!3 = !{i32 0, !"air.buffer", !"air.buffer_size", i32 22, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 22, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"Params", !"air.arg_name", !"params"}
!4 = !{i32 0, i32 16, i32 0, !"float3", !"origin", i32 16, i32 4, i32 0, !"uint", !"count", i32 20, i32 2, i32 0, !"ushort", !"flags"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
"#;

/// Scratch for one subject. Tool invocations write fixed file names inside it and these tests run
/// concurrently, so each subject gets its own directory.
fn scratch(label: &str) -> PathBuf {
    let directory =
        std::env::temp_dir().join(format!("m2v_handle_sizes_{}_{}", std::process::id(), label));
    std::fs::create_dir_all(&directory).expect("scratch directory");
    directory
}

fn reflect(source: &str, label: &str) -> ShaderReflection {
    let (_spirv, reflection) = translate_sanitized_native_reflected(
        source,
        Stage::Kernel,
        &scratch(label),
        TransformOptions::default(),
    )
    .unwrap_or_else(|error| panic!("{label} translates: {error}"));
    reflection
}

/// The reported layout of the argument declaring `declared_size` bytes, with that size.
fn layout_of(reflection: &ShaderReflection, declared_size: u32) -> AirType {
    reflection
        .bindings
        .iter()
        .find(|resource| resource.declared_size == Some(declared_size))
        .and_then(|resource| resource.type_layout.clone())
        .unwrap_or_else(|| panic!("no {declared_size}-byte argument reports a member layout"))
}

/// How far a reported layout reaches: offset zero to the last byte any member touches, with no tail
/// padding, which is the quantity `air.arg_type_size` states. Derived here rather than borrowed
/// from the translator, so this file is an independent opinion on the same question. A three-lane
/// vector occupies a four-lane slot unless it is `packed_`.
fn extent(ty: &AirType) -> u32 {
    fn width(scalar: AirScalar) -> u32 {
        match scalar {
            AirScalar::ULong | AirScalar::SLong => 8,
            AirScalar::Float | AirScalar::UInt | AirScalar::SInt => 4,
            AirScalar::Half | AirScalar::UShort | AirScalar::SShort => 2,
            AirScalar::UChar | AirScalar::Bool => 1,
        }
    }
    let slots = |lanes: u32| if lanes == 3 { 4 } else { lanes };
    match ty {
        AirType::Scalar(scalar) => width(*scalar),
        AirType::Vec { scalar, lanes } => width(*scalar) * slots(*lanes),
        AirType::PackedVec { scalar, lanes } => width(*scalar) * lanes,
        AirType::Matrix { scalar, cols, rows } => width(*scalar) * slots(*rows) * cols,
        AirType::Array { elem, len } => extent(elem) * len,
        AirType::Struct(members) => members
            .iter()
            .map(|member| member.offset + extent(&member.ty))
            .max()
            .unwrap_or(0),
    }
}

#[test]
fn a_handle_member_is_reported_at_the_handle_size_not_the_pointee_size() {
    let reflection = reflect(ARGUMENT_BUFFER_OF_HANDLES, "handles");

    // The three handles are eight bytes each whatever they point at; the two inline members keep
    // the type AIR named for them.
    assert_eq!(
        layout_of(&reflection, 36),
        AirType::Struct(vec![
            AirMember {
                offset: 0,
                ty: AirType::Scalar(AirScalar::ULong),
            },
            AirMember {
                offset: 8,
                ty: AirType::Scalar(AirScalar::ULong),
            },
            AirMember {
                offset: 16,
                ty: AirType::Scalar(AirScalar::ULong),
            },
            AirMember {
                offset: 24,
                ty: AirType::Scalar(AirScalar::ULong),
            },
            AirMember {
                offset: 32,
                ty: AirType::Scalar(AirScalar::UInt),
            },
        ]),
    );
}

#[test]
fn a_member_with_its_own_layout_keeps_it() {
    assert_eq!(
        layout_of(&reflect(ARGUMENT_BUFFER_WITH_NESTED_STRUCT, "nested"), 20),
        AirType::Struct(vec![
            AirMember {
                offset: 0,
                ty: AirType::Scalar(AirScalar::ULong),
            },
            AirMember {
                offset: 8,
                ty: AirType::Struct(vec![
                    AirMember {
                        offset: 0,
                        ty: AirType::Scalar(AirScalar::Float),
                    },
                    AirMember {
                        offset: 4,
                        ty: AirType::Scalar(AirScalar::Float),
                    },
                ]),
            },
            AirMember {
                offset: 16,
                ty: AirType::Scalar(AirScalar::UInt),
            },
        ]),
    );
}

#[test]
fn an_inline_struct_keeps_every_type_air_named() {
    assert_eq!(
        layout_of(&reflect(INLINE_STRUCT_BUFFER, "inline"), 22),
        AirType::Struct(vec![
            AirMember {
                offset: 0,
                ty: AirType::Vec {
                    scalar: AirScalar::Float,
                    lanes: 3,
                },
            },
            AirMember {
                offset: 16,
                ty: AirType::Scalar(AirScalar::UInt),
            },
            AirMember {
                offset: 20,
                ty: AirType::Scalar(AirScalar::UShort),
            },
        ]),
    );
}

/// The property the three cases above are instances of: AIR states the argument's size and its
/// member layout independently, and a consumer needs them to describe the same bytes. A layout that
/// reaches past the argument means some member's storage was mistaken; one that stops short means a
/// member was reported narrower than it is.
#[test]
fn a_reported_layout_reaches_exactly_as_far_as_the_argument_it_describes() {
    for (label, source, declared) in [
        ("handles", ARGUMENT_BUFFER_OF_HANDLES, 36),
        ("nested", ARGUMENT_BUFFER_WITH_NESTED_STRUCT, 20),
        ("inline", INLINE_STRUCT_BUFFER, 22),
    ] {
        let layout = layout_of(&reflect(source, &format!("extent_{label}")), declared);
        assert_eq!(
            extent(&layout),
            declared,
            "{label}: the reported layout does not describe the {declared} bytes AIR declares, {layout:?}",
        );
    }
}
