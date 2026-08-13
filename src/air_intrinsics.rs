//! Canonical recognition contract for Metal AIR intrinsic symbols.
//!
//! Translation validates operand and result shapes in the lowering that owns each family. This
//! module answers the earlier, narrower question: whether a symbol belongs to an AIR ABI family for
//! which the product has an intentional lowering or static-linkage contract. Validation uses the
//! same inventory, so a newly harvested `air.*` family cannot disappear behind an unrelated clean
//! authored-resource audit.

use std::collections::BTreeMap;

/// Where the product intentionally handles an AIR intrinsic family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AirIntrinsicDisposition {
    /// The native emitter or retained-SPIR-V passes lower this family directly.
    Lowered,
    /// Translation consumes this family only after the caller supplies exact authored linkage.
    StaticLinkage,
    /// Some exact ABI shapes lower directly while callback-bearing shapes require static linkage.
    LoweredOrStaticLinkage,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Matrix16Element {
    F32,
    F16,
    Bf16,
    F8E4M3,
    F8E4M3Fn,
    F8E5M2,
    I8 { signed: bool },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Matrix16Intrinsic {
    pub lhs: Matrix16Element,
    pub rhs: Matrix16Element,
    pub integer: bool,
}

/// Parse the complete stable 16x16x16 matrix ABI grammar shared by inventory and lowering.
pub(crate) fn matrix16_intrinsic(name: &str) -> Option<Matrix16Intrinsic> {
    let fields: Vec<_> = name.split('.').collect();
    if fields.len() != 8 || fields[0] != "air" {
        return None;
    }
    match fields[1] {
        "simdgroup_matrix_16x16x16_multiply_accumulate"
            if fields[2] == "f"
                && fields[3] == "f"
                && fields[4] == "v8f32"
                && fields[7] == "v8f32" =>
        {
            Some(Matrix16Intrinsic {
                lhs: matrix16_float_element(fields[5])?,
                rhs: matrix16_float_element(fields[6])?,
                integer: false,
            })
        }
        "simdgroup_matrix_16x16x16_widening_multiply_accumulate"
            if matches!(fields[2], "s" | "u")
                && matches!(fields[3], "s" | "u")
                && fields[4] == "v8i32"
                && fields[5] == "v8i8"
                && fields[6] == "v8i8"
                && fields[7] == "v8i32" =>
        {
            Some(Matrix16Intrinsic {
                lhs: Matrix16Element::I8 {
                    signed: fields[2] == "s",
                },
                rhs: Matrix16Element::I8 {
                    signed: fields[3] == "s",
                },
                integer: true,
            })
        }
        _ => None,
    }
}

fn matrix16_float_element(token: &str) -> Option<Matrix16Element> {
    Some(match token {
        "v8f32" => Matrix16Element::F32,
        "v8f16" => Matrix16Element::F16,
        "v8bf16" => Matrix16Element::Bf16,
        "v8f8e4m3" => Matrix16Element::F8E4M3,
        "v8f8e4m3fn" => Matrix16Element::F8E4M3Fn,
        "v8f8e5m2" => Matrix16Element::F8E5M2,
        _ => return None,
    })
}

/// Count AIR calls in sanitized LLVM IR by exact ABI symbol.
///
/// Declarations are not uses. The scan accepts ordinary, tail, musttail, and notail calls and counts
/// every `@air.*` callee on the instruction line, matching the translator's call-oriented contract.
pub fn air_call_counts(ll: &str) -> BTreeMap<String, usize> {
    let mut calls = BTreeMap::new();
    for line in ll.lines() {
        for symbol in direct_air_callees(line) {
            *calls.entry(symbol).or_default() += 1;
        }
    }
    calls
}

fn direct_air_callees(line: &str) -> Vec<String> {
    if ![
        "call ", "call\t", "invoke ", "invoke\t", "callbr ", "callbr\t",
    ]
    .iter()
    .any(|opcode| line.contains(opcode))
    {
        return Vec::new();
    }
    let line = if line.contains(';') {
        strip_llvm_comment(line)
    } else {
        line
    }
    .trim();
    call_opcode_ends(line)
        .into_iter()
        .filter_map(|offset| direct_air_callee_after(&line[offset..]))
        .collect()
}

fn call_opcode_ends(line: &str) -> Vec<usize> {
    const OPCODES: [&str; 6] = [
        "call ", "call\t", "invoke ", "invoke\t", "callbr ", "callbr\t",
    ];
    let mut offsets = Vec::new();
    let mut cursor = 0usize;
    while cursor < line.len() {
        let Some((start, opcode)) = OPCODES
            .iter()
            .filter_map(|opcode| line[cursor..].find(opcode).map(|at| (cursor + at, *opcode)))
            .min_by_key(|(at, _)| *at)
        else {
            break;
        };
        let before = line[..start].chars().next_back();
        let opcode_context =
            before.is_none_or(|ch| ch.is_whitespace() || matches!(ch, '=' | '{' | '}' | ';'));
        if opcode_context && !quoted_at(line, start) {
            offsets.push(start + opcode.len());
        }
        cursor = start + 1;
    }
    offsets
}

fn quoted_at(line: &str, offset: usize) -> bool {
    if !line[..offset].contains('"') {
        return false;
    }
    let mut quoted = false;
    let mut escaped = false;
    for ch in line[..offset].chars() {
        if quoted && ch == '\\' && !escaped {
            escaped = true;
            continue;
        }
        if ch == '"' && !escaped {
            quoted = !quoted;
        }
        escaped = false;
    }
    quoted
}

fn direct_air_callee_after(call: &str) -> Option<String> {
    let mut depth = 0usize;
    let mut quoted = false;
    let mut escaped = false;
    let mut at = None;
    for (offset, ch) in call.char_indices() {
        if quoted && ch == '\\' && !escaped {
            escaped = true;
            continue;
        }
        if ch == '"' && !escaped {
            quoted = !quoted;
        } else if !quoted {
            match ch {
                '(' => depth += 1,
                ')' => depth = depth.saturating_sub(1),
                '@' if depth == 0 => {
                    at = Some(offset);
                    break;
                }
                _ => {}
            }
        }
        escaped = false;
    }
    let symbol = &call[at? + 1..];
    if let Some(quoted) = symbol.strip_prefix('"') {
        let end = quoted.find('"')?;
        return quoted[..end]
            .starts_with("air.")
            .then(|| quoted[..end].to_string());
    }
    let end = symbol
        .find(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '_' | '.')))
        .unwrap_or(symbol.len());
    (end > "air.".len() && symbol.starts_with("air.")).then(|| symbol[..end].to_string())
}

fn strip_llvm_comment(line: &str) -> &str {
    let mut quoted = false;
    let mut escaped = false;
    for (offset, ch) in line.char_indices() {
        if quoted && ch == '\\' && !escaped {
            escaped = true;
            continue;
        }
        if ch == '"' && !escaped {
            quoted = !quoted;
        } else if ch == ';' && !quoted {
            return &line[..offset];
        }
        escaped = false;
    }
    line
}

/// Count every called AIR symbol outside the product's intentional family contract.
pub fn unrecognized_air_intrinsics(ll: &str) -> BTreeMap<String, usize> {
    unrecognized_air_intrinsics_from_counts(&air_call_counts(ll))
}

/// Filter an existing exact call inventory through the product's intentional family contract.
///
/// Classification uses this form so inventory reporting and unknown-family reporting consume the
/// same parse rather than independently scanning a potentially large AIR module.
pub fn unrecognized_air_intrinsics_from_counts(
    calls: &BTreeMap<String, usize>,
) -> BTreeMap<String, usize> {
    calls
        .iter()
        .filter(|(name, _)| air_intrinsic_disposition(name).is_none())
        .map(|(name, count)| (name.clone(), *count))
        .collect()
}

fn typed(name: &str, stem: &str) -> bool {
    name == stem
        || name
            .strip_prefix(stem)
            .is_some_and(|tail| tail.starts_with('.'))
}

fn any_typed(name: &str, stems: &[&str]) -> bool {
    stems.iter().any(|stem| typed(name, stem))
}

fn any_prefix(name: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| name.starts_with(prefix))
}

pub(crate) fn is_command_encoder_helper(name: &str) -> bool {
    matches!(
        name,
        "air.concurrent_dispatch_threadgroups_compute_command"
            | "air.concurrent_dispatch_threads_compute_command"
            | "air.draw_primitives_render_command"
            | "air.set_barrier_compute_command"
            | "air.set_pipeline_state_compute_command"
            | "air.set_threadgroup_memory_length_compute_command"
    ) || any_typed(
        name,
        &[
            "air.set_fragment_buffer_render_command",
            "air.set_kernel_buffer_compute_command",
            "air.set_object_buffer_render_command",
            "air.set_vertex_buffer_render_command",
        ],
    )
}

/// Return the product contract for one complete `air.*` ABI symbol.
///
/// Broad namespaces such as `air.simdgroup_matrix_` are deliberately not accepted. Every admitted
/// prefix corresponds to a concrete dispatch arm, so a new sibling family remains visible until
/// its implementation and this contract are added together.
pub fn air_intrinsic_disposition(name: &str) -> Option<AirIntrinsicDisposition> {
    use AirIntrinsicDisposition::{Lowered, LoweredOrStaticLinkage, StaticLinkage};

    if !name.starts_with("air.") {
        return None;
    }

    if name.starts_with("air.intersect.") {
        return Some(LoweredOrStaticLinkage);
    }

    if any_prefix(
        name,
        &[
            "air.set_buffer_intersection_function_table.",
            "air.set_intersection_function_table.",
        ],
    ) || matches!(
        name,
        "air.get_null_intersection_function_table"
            | "air.get_size_visible_function_table"
            | "air.get_function_pointer_visible_function_table"
            | "air.is_null_visible_function_table"
            // Inline tensor descriptors are opaque Apple ABI objects consumed by externally-defined
            // tensor-ops helpers. They must be supplied in the same exact authored linkage unit;
            // lowering only these calls would invent a layout the linked consumer does not share.
            | "air.get_descriptor_size_tensor"
            | "air.get_extent_private_tensor.i32"
            | "air.init_strided_private_tensor.i32.global"
            | "air.init_strided_private_tensor.i32.local"
            | "air.slice_private_tensor_private_tensor.s.i32"
    ) {
        return Some(StaticLinkage);
    }

    if matches!(
        name,
        "air.simdgroup.barrier"
            | "air.wg.barrier"
            | "air.atomic.fence"
            | "air.atomic.local.store.i32"
            | "air.atomic.global.store.i32"
            | "air.atomic.local.load.i32"
            | "air.atomic.global.load.i32"
            | "air.atomic.local.cmpxchg.weak.i32"
            | "air.atomic.global.cmpxchg.weak.i32"
            | "air.atomic.global.add.f32"
            | "air.atomic.global.sub.f32"
            | "air.atomic.local.add.s.i32"
            | "air.atomic.local.add.u.i32"
            | "air.atomic.local.sub.s.i32"
            | "air.atomic.local.sub.u.i32"
            | "air.atomic.local.max.s.i32"
            | "air.atomic.local.max.u.i32"
            | "air.atomic.local.min.s.i32"
            | "air.atomic.local.min.u.i32"
            | "air.atomic.local.and.u.i32"
            | "air.atomic.local.or.u.i32"
            | "air.atomic.local.xor.u.i32"
            | "air.atomic.local.xchg.i32"
            | "air.atomic.global.add.s.i32"
            | "air.atomic.global.add.u.i32"
            | "air.atomic.global.and.u.i32"
            | "air.atomic.global.or.u.i32"
            | "air.atomic.global.xchg.i32"
            | "air.atomic.global.max.s.i32"
            | "air.atomic.global.max.u.i32"
            | "air.atomic.global.min.s.i32"
            | "air.atomic.global.min.u.i32"
            | "air.atomic.global.sub.s.i32"
            | "air.atomic.global.sub.u.i32"
            | "air.atomic.global.xor.u.i32"
            | "air.simd_any"
            | "air.simd_all"
            | "air.simd_ballot.i64"
            | "air.quad_all"
            | "air.quad_any"
            | "air.quad_active_threads_mask"
            | "air.quad_is_first"
            | "air.simd_is_first"
            | "air.is_function_constant_defined"
            | "air.get_num_samples.i32"
            | "air.get_imageblock_width"
            | "air.get_imageblock_height"
            | "air.get_read_sampler"
            | "air.get_instance_count_instance_acceleration_structure"
            | "air.get_primitive_acceleration_structure_instance_acceleration_structure"
            | "air.get_data_pointer_instance_acceleration_structure"
            | "air.calculate_unclamped_lod_texture_2d"
            | "air.calculate_clamped_lod_texture_2d"
            | "air.imageblock_data"
            | "air.rhadd.u.i16"
    ) {
        return Some(Lowered);
    }

    if any_prefix(
        name,
        &[
            "air.load.device_coherent.",
            "air.load.system_coherent.",
            "air.store.device_coherent.",
            "air.store.system_coherent.",
            "air.fence_texture",
            "air.sample_texture",
            "air.sample_depth",
            "air.sample_compare_depth",
            "air.gather_texture",
            "air.gather_depth",
            "air.read_texture",
            "air.read_depth",
            "air.write_texture",
            "air.atomic_fetch_max_explicit_texture_",
            "air.write_imageblock_slice_to_texture",
            "air.load.implicit_imageblock.",
            "air.store.implicit_imageblock.",
            "air.discard_fragment",
            "air.get_width_texture",
            "air.get_height_texture",
            "air.get_depth_texture",
            "air.get_array_size_texture",
            "air.get_width_depth",
            "air.get_height_depth",
            "air.get_depth_depth",
            "air.get_num_mip_levels_texture",
            "air.get_num_mip_levels_depth",
            "air.get_num_samples_texture",
            "air.is_null_texture",
            "air.get_null_texture",
            "air.map_screen_to_physical_coordinates.",
            "air.map_physical_to_screen_coordinates.",
            "air.simdgroup_matrix_8x8_load.",
            "air.simdgroup_matrix_8x8_store.",
            "air.simdgroup_matrix_8x8_init_diag.",
            "air.simdgroup_matrix_8x8_multiply_accumulate.",
            "air.simdgroup_async_copy_2d.",
            "air.get_null_simdgroup_event",
            "air.is_null_simdgroup_event",
            "air.wait_simdgroup_events",
            "air.function_constant_predicate",
            "air.normalize_function_constant_predicate.",
        ],
    ) {
        return Some(Lowered);
    }

    if matrix16_intrinsic(name).is_some() {
        return Some(Lowered);
    }

    // Command helpers are an intentional observed-output contract: translation consumes these exact
    // stable ABI families while the validation executor reports the indirect-command-buffer tooling
    // requirement separately. Do not admit the broad `_command` suffix: a new sibling must remain
    // visible until both this contract and the lowering intentionally support it.
    if is_command_encoder_helper(name) {
        return Some(Lowered);
    }

    if any_typed(
        name,
        &[
            "air.abs_diff",
            "air.abs",
            "air.reverse_bits",
            "air.bswap",
            "air.rotate",
            "air.extract_bits.u",
            "air.extract_bits.s",
            "air.insert_bits",
            "air.popcount",
            "air.ctz",
            "air.clz",
            "air.mul_hi.u",
            "air.mad_sat.s",
            "air.add_sat.u",
            "air.sub_sat.u",
            "air.convert",
            "air.dfdx",
            "air.fast_dfdx",
            "air.dfdy",
            "air.fast_dfdy",
            "air.fwidth",
            "air.dot",
            "air.is_uniform",
            "air.get_simdgroup_size",
            "air.simd_broadcast_first",
            "air.simd_broadcast",
            "air.simd_shuffle",
            "air.simd_shuffle_down",
            "air.simd_shuffle_rotate_down",
            "air.simd_shuffle_and_fill_down",
            "air.simd_shuffle_and_fill_up",
            "air.simd_shuffle_up",
            "air.simd_shuffle_xor",
            "air.quad_shuffle",
            "air.quad_broadcast",
            "air.quad_sum",
            "air.quad_min",
            "air.quad_max",
            "air.quad_shuffle_xor",
            "air.quad_shuffle_rotate_down",
            "air.quad_shuffle_up",
            "air.quad_shuffle_down",
            "air.simd_prefix_exclusive_sum",
            "air.simd_prefix_inclusive_sum",
            "air.simd_sum",
            "air.simd_or",
            "air.simd_xor",
            "air.simd_and",
            "air.simd_min",
            "air.simd_max",
            "air.all",
            "air.any",
            "air.pack",
            "air.unpack",
            "air.sincos",
            "air.cospi",
            "air.sinpi",
            "air.tanpi",
            "air.exp10",
            "air.fmod",
            "air.log10",
            "air.min.s",
            "air.min.u",
            "air.max.s",
            "air.max.u",
            "air.clamp.s",
            "air.clamp.u",
            "air.max3.u",
            "air.min3.u",
            "air.max3.s",
            "air.min3.s",
            "air.fmax3",
            "air.fmin3",
            "air.fmedian3",
            "air.ldexp",
            "air.atan2",
            "air.asin",
            "air.acos",
            "air.atan",
            "air.fmax",
            "air.max",
            "air.fmin",
            "air.min",
            "air.sqrt",
            "air.rsqrt",
            "air.fabs",
            "air.pow",
            "air.powr",
            "air.sign",
            "air.mix",
            "air.floor",
            "air.ceil",
            "air.round",
            "air.rint",
            "air.trunc",
            "air.tan",
            "air.sinh",
            "air.cosh",
            "air.tanh",
            "air.asinh",
            "air.acosh",
            "air.atanh",
            "air.sin",
            "air.cos",
            "air.exp2",
            "air.exp",
            "air.log2",
            "air.log",
            "air.fract",
            "air.fma",
            "air.clamp",
            "air.saturate",
        ],
    ) || name.strip_prefix("air.fast_").is_some_and(|tail| {
        [
            "sincos", "cospi", "sinpi", "tanpi", "exp10", "fmod", "log10", "fmax3", "fmin3",
            "fmedian3", "ldexp", "atan2", "asin", "acos", "atan", "fmax", "max", "fmin", "min",
            "sqrt", "rsqrt", "fabs", "pow", "powr", "sign", "mix", "floor", "ceil", "round",
            "rint", "trunc", "tan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "sin",
            "cos", "exp2", "exp", "log2", "log", "fract", "fma", "clamp", "saturate",
        ]
        .iter()
        .any(|stem| {
            tail.strip_prefix(stem)
                .is_some_and(|rest| rest.starts_with('.'))
        })
    }) {
        return Some(Lowered);
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognition_is_family_bounded() {
        assert_eq!(
            air_intrinsic_disposition("air.any.v4i1"),
            Some(AirIntrinsicDisposition::Lowered)
        );
        assert_eq!(air_intrinsic_disposition("air.anything.v4i1"), None);
        assert_eq!(air_intrinsic_disposition("air.tensor_future.i32"), None);
        assert_eq!(
            air_intrinsic_disposition("air.init_strided_private_tensor.i32.local"),
            Some(AirIntrinsicDisposition::StaticLinkage)
        );
        assert_eq!(air_intrinsic_disposition("air.future_command"), None);
        assert_eq!(
            air_intrinsic_disposition("air.set_kernel_buffer_compute_command.p1i8"),
            Some(AirIntrinsicDisposition::Lowered)
        );
    }

    #[test]
    fn linkage_is_distinct_from_direct_lowering() {
        assert_eq!(
            air_intrinsic_disposition("air.intersect.instancing.triangle_data"),
            Some(AirIntrinsicDisposition::LoweredOrStaticLinkage)
        );
        assert_eq!(
            air_intrinsic_disposition("air.simdgroup_matrix_8x8_load.v64f32.p1f32"),
            Some(AirIntrinsicDisposition::Lowered)
        );
        assert_eq!(
            air_intrinsic_disposition(
                "air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f16.v8f16.v8f32"
            ),
            Some(AirIntrinsicDisposition::Lowered)
        );
    }

    #[test]
    fn matrix16_recognition_requires_the_complete_known_abi_shape() {
        let float =
            "air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8bf16.v8f8e4m3.v8f32";
        assert_eq!(
            matrix16_intrinsic(float),
            Some(Matrix16Intrinsic {
                lhs: Matrix16Element::Bf16,
                rhs: Matrix16Element::F8E4M3,
                integer: false,
            })
        );
        assert!(matrix16_intrinsic(
            "air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8future.v8f16.v8f32"
        )
        .is_none());
        assert!(matrix16_intrinsic(
            "air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f16.v8f16.v8f16.v8f16"
        )
        .is_none());
        assert!(matrix16_intrinsic(
            "air.simdgroup_matrix_16x16x16_widening_multiply_accumulate.s.x.v8i32.v8i8.v8i8.v8i32"
        )
        .is_none());
    }

    #[test]
    fn call_inventory_ignores_non_callees_and_counts_all_call_forms() {
        let ll = "define void @main() {\nentry:\n %a = call i32 @air.abs.s.i32(i32 -1)\n %b = tail call i32 @air.future_tensor.i32(i32 1)\n %c = musttail\tcall i32 @air.future_tensor.i32(i32 2)\n %d = invoke i32 @\"air.quoted_future.i32\"(i32 3) to label %ok unwind label %err\n callbr void @air.branch_future() to label %ok [label %err]\n call void @ordinary(ptr @air.declaration_only.i32) ; @air.comment_only\n ret void\n}\ndefine void @compact() { %x = call i32 @air.compact_future(i32 0) ret void }\ndeclare i32 @air.declaration_only.i32(i32)";
        assert_eq!(
            air_call_counts(ll),
            BTreeMap::from([
                ("air.abs.s.i32".into(), 1),
                ("air.branch_future".into(), 1),
                ("air.compact_future".into(), 1),
                ("air.future_tensor.i32".into(), 2),
                ("air.quoted_future.i32".into(), 1),
            ])
        );
        assert_eq!(
            unrecognized_air_intrinsics(ll),
            BTreeMap::from([
                ("air.branch_future".into(), 1),
                ("air.compact_future".into(), 1),
                ("air.future_tensor.i32".into(), 2),
                ("air.quoted_future.i32".into(), 1),
            ])
        );
    }
}
