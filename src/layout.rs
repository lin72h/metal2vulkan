//! One Metal/SPIR-V layout oracle: size/align for native `LlType`, decoded AIR `AirType`, and
//! completed SPIR-V type graphs under explicit named rules.
//!
//! Each rule is a pure function modulo a `resolve` closure that expands `LlType::Named` to its
//! concrete form via the owning module's named-type table — the only `&self` dependency the original
//! `LlModule` methods carried. The `LlModule` methods now delegate here, so the recursion (and the
//! vec3 padding / struct alignment rules that differ per rule) lives in exactly one place.
//!
//! The SPIR-V rule accepts either natural struct packing or the exact AIR member offsets carried by
//! `EmitSidecar`. Both native emission and final passes use this seam-neutral module; neither
//! ownership domain imports the other.

use crate::meta::AirType;
use crate::native::ir::{ll_type_from_air_type, round_up_u64, LlType};
use crate::spirv_module::Instruction;
use crate::spirv_module::Operand;
use spirv::{Op, Word};
use std::collections::HashMap;

/// Byte size of a scalar/pointer `LlType` (the "scalar storage" rule). `None` for aggregates and
/// unknown types. Was `LlModule::scalar_storage_size`.
pub(super) fn scalar_storage_size(
    ty: &LlType,
    resolve: &impl Fn(&LlType) -> LlType,
) -> Option<u64> {
    match resolve(ty) {
        LlType::Bool | LlType::Int(1) | LlType::Int(8) => Some(1),
        LlType::Half | LlType::BFloat | LlType::Int(16) => Some(2),
        LlType::Float | LlType::Int(32) => Some(4),
        LlType::Int(64) | LlType::Ptr(_) => Some(8),
        _ => None,
    }
}

/// The "Native" rule: tight packing, alignment = element alignment (a vec3 is 12/4 — no vector-size
/// growth). Was `LlModule::type_storage_size_align`.
pub(super) fn native_size_align(
    ty: &LlType,
    resolve: &impl Fn(&LlType) -> LlType,
) -> Option<(u64, u64)> {
    match resolve(ty) {
        LlType::Bool => Some((1, 1)),
        LlType::Int(bits) => {
            let bytes = u64::from(bits).div_ceil(8).max(1);
            Some((bytes, bytes.next_power_of_two().min(8)))
        }
        LlType::Half | LlType::BFloat => Some((2, 2)),
        LlType::Float => Some((4, 4)),
        LlType::Ptr(_) => Some((8, 8)),
        LlType::Vector(elem, lanes) => {
            let (elem_size, elem_align) = native_size_align(&elem, resolve)?;
            Some((elem_size * lanes as u64, elem_align))
        }
        LlType::Array(elem, len) => {
            let (elem_size, elem_align) = native_size_align(&elem, resolve)?;
            Some((round_up_u64(elem_size, elem_align) * len as u64, elem_align))
        }
        LlType::Struct(fields) => {
            let mut offset = 0u64;
            let mut max_align = 1u64;
            for field in fields {
                let (size, align) = native_size_align(&field, resolve)?;
                max_align = max_align.max(align);
                offset = round_up_u64(offset, align);
                offset += size;
            }
            Some((round_up_u64(offset, max_align), max_align))
        }
        LlType::Void | LlType::Named(_) => None,
    }
}

/// The "Memcpy" rule: storage layout for a `memcpy` (a vec3 is padded to 4 lanes = 16/16, its
/// size==align; arrays align to at least 4). Was `LlModule::native_memcpy_type_size_align`.
pub(super) fn memcpy_size_align(
    ty: &LlType,
    resolve: &impl Fn(&LlType) -> LlType,
) -> Option<(u64, u64)> {
    match resolve(ty) {
        LlType::Bool => Some((1, 1)),
        LlType::Int(bits) => {
            let bytes = u64::from(bits).div_ceil(8).max(1);
            Some((bytes, bytes.next_power_of_two().min(8)))
        }
        LlType::Half | LlType::BFloat => Some((2, 2)),
        LlType::Float => Some((4, 4)),
        LlType::Ptr(_) => Some((8, 8)),
        LlType::Vector(elem, lanes) => {
            let (elem_size, _elem_align) = memcpy_size_align(&elem, resolve)?;
            let storage_lanes = if lanes == 3 { 4 } else { lanes };
            let size = elem_size * storage_lanes as u64;
            Some((size, size))
        }
        LlType::Array(elem, len) => {
            let (elem_size, elem_align) = memcpy_size_align(&elem, resolve)?;
            Some((
                round_up_u64(elem_size, elem_align) * len as u64,
                elem_align.max(4),
            ))
        }
        LlType::Struct(fields) => {
            let mut offset = 0u64;
            let mut max_align = 1u64;
            for field in fields {
                let (size, align) = memcpy_size_align(&field, resolve)?;
                max_align = max_align.max(align);
                offset = round_up_u64(offset, align);
                offset += size;
            }
            Some((round_up_u64(offset, max_align), max_align))
        }
        LlType::Void | LlType::Named(_) => None,
    }
}

/// The "Raw" rule (emitter): the same padded shape as the Memcpy rule (a vec3 occupies 4 lanes and
/// self-aligns; arrays floor align at 4), but with two differences that make it its own variant:
///   * `resolve` is FALLIBLE — a `Named` type the module can't expand is an error, not a fallthrough.
///   * uncovered types (odd-width ints, `Void`) are an `Err`, not `None` — the Raw rule only accepts
///     the standard scalar widths it knows how to lay out.
///     Returns `Result` accordingly. Was `Emitter::raw_type_size_align` (`resolve` = `Emitter::resolve_type`).
pub(super) fn raw_size_align(
    ty: &LlType,
    resolve: &impl Fn(&LlType) -> Result<LlType, String>,
) -> Result<(u64, u64), String> {
    match resolve(ty)? {
        LlType::Bool | LlType::Int(1) | LlType::Int(8) => Ok((1, 1)),
        LlType::Half | LlType::BFloat | LlType::Int(16) => Ok((2, 2)),
        LlType::Float | LlType::Int(32) => Ok((4, 4)),
        LlType::Int(64) => Ok((8, 8)),
        LlType::Ptr(_) => Ok((8, 8)),
        LlType::Vector(elem, lanes) => {
            let (elem_size, _) = raw_size_align(&elem, resolve)?;
            let storage_lanes = if lanes == 3 { 4 } else { lanes };
            let size = elem_size * storage_lanes as u64;
            Ok((size, size))
        }
        LlType::Array(elem, len) => {
            let (elem_size, elem_align) = raw_size_align(&elem, resolve)?;
            let stride = round_up_u64(elem_size, elem_align);
            Ok((stride * len as u64, elem_align.max(4)))
        }
        LlType::Struct(fields) => {
            let mut off = 0;
            let mut max_align = 1;
            for field in fields {
                let (size, align) = raw_size_align(&field, resolve)?;
                off = round_up_u64(off, align) + size;
                max_align = max_align.max(align);
            }
            Ok((round_up_u64(off, max_align), max_align))
        }
        other => Err(format!(
            "native emitter: raw size/align for {other:?} is not covered yet"
        )),
    }
}

/// The "AirMetadata" rule: size/align for an AIR-metadata `AirType`, honoring explicit member offsets
/// on structs (the compiler-declared ABI layout) and falling through to the Memcpy rule for
/// non-structs. Was `LlModule::air_metadata_type_size_align`.
pub(super) fn air_metadata_size_align(
    ty: &AirType,
    resolve: &impl Fn(&LlType) -> LlType,
) -> Option<(u64, u64)> {
    match ty {
        AirType::Struct(members) => {
            let mut end = 0u64;
            let mut max_align = 1u64;
            for member in members {
                let (size, align) = air_metadata_size_align(&member.ty, resolve)?;
                max_align = max_align.max(align);
                end = end.max(u64::from(member.offset) + size);
            }
            Some((round_up_u64(end, max_align), max_align))
        }
        _ => memcpy_size_align(&ll_type_from_air_type(ty), resolve),
    }
}

#[derive(Clone, Copy)]
pub(crate) enum SpirvLayout<'a> {
    /// Natural MSL/std430-style packing over the emitted type graph.
    Natural,
    /// Exact AIR offsets for mapped structs, natural packing for every other struct.
    AirOffsets(&'a HashMap<Word, Vec<u32>>),
}

impl<'a> SpirvLayout<'a> {
    fn struct_offsets(self, ty: Word) -> Option<&'a [u32]> {
        match self {
            Self::Natural => None,
            Self::AirOffsets(offsets) => offsets.get(&ty).map(Vec::as_slice),
        }
    }

    fn supports_runtime_array(self) -> bool {
        matches!(self, Self::AirOffsets(_))
    }
}

/// Size/alignment of an emitted SPIR-V type under the descriptor-block rule used by interface
/// decoration and producer-side AIR-layout matching.
///
/// Scalars use their declared byte width. Vectors keep tight size but align vec2 to two components
/// and vec3/vec4 to four. Arrays use element alignment with no four-byte floor. Natural structs pack
/// each member at its alignment; `AirOffsets` substitutes exact decoded offsets where the sidecar
/// provides them. A runtime array has one-element extent only for `AirOffsets`, matching the
/// final-pass sizing behavior; the natural producer matcher never treats one as a sized aggregate.
pub(crate) fn spirv_size_align(
    ty: Word,
    defs: &HashMap<Word, Instruction>,
    rule: SpirvLayout<'_>,
) -> (u32, u32) {
    let Some(def) = defs.get(&ty) else {
        return (4, 4);
    };
    match def.class.opcode {
        Op::TypeFloat | Op::TypeInt => {
            let width = match def.operands.first() {
                Some(Operand::LiteralBit32(bits)) => *bits / 8,
                _ => 4,
            };
            (width, width)
        }
        Op::TypeVector => {
            let elem = match def.operands.first() {
                Some(Operand::IdRef(elem)) => *elem,
                _ => return (16, 16),
            };
            let lanes = match def.operands.get(1) {
                Some(Operand::LiteralBit32(lanes)) => *lanes,
                _ => 4,
            };
            let (elem_size, _) = spirv_size_align(elem, defs, rule);
            let align = if lanes == 2 {
                elem_size * 2
            } else {
                elem_size * 4
            };
            (elem_size * lanes, align)
        }
        Op::TypeArray | Op::TypeRuntimeArray if rule.supports_runtime_array() => {
            let elem = match def.operands.first() {
                Some(Operand::IdRef(elem)) => *elem,
                _ => return (16, 16),
            };
            let len = if def.class.opcode == Op::TypeArray {
                array_len(defs, def).unwrap_or(1)
            } else {
                1
            };
            let (elem_size, elem_align) = spirv_size_align(elem, defs, rule);
            (round_up_u32(elem_size, elem_align) * len, elem_align)
        }
        Op::TypeArray => {
            let elem = match def.operands.first() {
                Some(Operand::IdRef(elem)) => *elem,
                _ => return (16, 16),
            };
            let len = array_len(defs, def).unwrap_or(1);
            let (elem_size, elem_align) = spirv_size_align(elem, defs, rule);
            (round_up_u32(elem_size, elem_align) * len, elem_align)
        }
        Op::TypeStruct => {
            let mut cursor = 0u32;
            let mut end = 0u32;
            let mut max_align = 4u32;
            let explicit_offsets = rule.struct_offsets(ty);
            for (index, operand) in def.operands.iter().enumerate() {
                let Operand::IdRef(member) = operand else {
                    continue;
                };
                let (size, align) = spirv_size_align(*member, defs, rule);
                // A member consumes its allocation size — `alignTo(storeSize, abiAlign)`, the same
                // rule LLVM's `StructLayout` applies. Only three-lane vectors differ (store 3/6/12
                // bytes, allocate 4/8/16); every other emitted type already has a size that is a
                // multiple of its alignment.
                let alloc = round_up_u32(size, align);
                let member_offset = explicit_offsets
                    .and_then(|offsets| offsets.get(index).copied())
                    .unwrap_or_else(|| round_up_u32(cursor, align));
                end = end.max(member_offset + alloc);
                cursor = member_offset + alloc;
                max_align = max_align.max(align);
            }
            (round_up_u32(end, max_align), max_align)
        }
        _ => (4, 4),
    }
}

fn array_len(defs: &HashMap<Word, Instruction>, def: &Instruction) -> Option<u32> {
    let Operand::IdRef(constant) = def.operands.get(1)? else {
        return None;
    };
    match defs.get(constant)?.operands.first()? {
        Operand::LiteralBit32(len) => Some(*len),
        _ => None,
    }
}

pub(crate) fn round_up_u32(value: u32, align: u32) -> u32 {
    if align == 0 {
        value
    } else {
        value.div_ceil(align) * align
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spirv_air_offsets_set_struct_and_runtime_array_extent() {
        let uint_ty = 1;
        let struct_ty = 2;
        let runtime_array_ty = 3;
        let mut defs = HashMap::new();
        defs.insert(
            uint_ty,
            Instruction::new(
                Op::TypeInt,
                None,
                Some(uint_ty),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
        );
        defs.insert(
            struct_ty,
            Instruction::new(
                Op::TypeStruct,
                None,
                Some(struct_ty),
                vec![Operand::IdRef(uint_ty), Operand::IdRef(uint_ty)],
            ),
        );
        defs.insert(
            runtime_array_ty,
            Instruction::new(
                Op::TypeRuntimeArray,
                None,
                Some(runtime_array_ty),
                vec![Operand::IdRef(struct_ty)],
            ),
        );

        let offsets = HashMap::from([(struct_ty, vec![0, 16])]);
        let rule = SpirvLayout::AirOffsets(&offsets);

        assert_eq!(spirv_size_align(struct_ty, &defs, rule), (20, 4));
        assert_eq!(spirv_size_align(runtime_array_ty, &defs, rule), (20, 4));
        assert_eq!(
            spirv_size_align(runtime_array_ty, &defs, SpirvLayout::Natural),
            (4, 4)
        );
    }

    #[test]
    fn spirv_natural_vector_and_array_rules() {
        let float_ty = 1u32;
        let vec2_ty = 2u32;
        let vec3_ty = 3u32;
        let vec4_ty = 4u32;
        let len3 = 5u32;
        let float_array_ty = 6u32;
        let vec3_array_ty = 7u32;
        let len2 = 8u32;
        let uint_ty = 9u32;
        let mut defs = HashMap::new();
        defs.insert(
            float_ty,
            Instruction::new(
                Op::TypeFloat,
                None,
                Some(float_ty),
                vec![Operand::LiteralBit32(32)],
            ),
        );
        for (id, lanes) in [(vec2_ty, 2u32), (vec3_ty, 3), (vec4_ty, 4)] {
            defs.insert(
                id,
                Instruction::new(
                    Op::TypeVector,
                    None,
                    Some(id),
                    vec![Operand::IdRef(float_ty), Operand::LiteralBit32(lanes)],
                ),
            );
        }
        defs.insert(
            uint_ty,
            Instruction::new(
                Op::TypeInt,
                None,
                Some(uint_ty),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
        );
        defs.insert(
            len3,
            Instruction::new(
                Op::Constant,
                Some(uint_ty),
                Some(len3),
                vec![Operand::LiteralBit32(3)],
            ),
        );
        defs.insert(
            len2,
            Instruction::new(
                Op::Constant,
                Some(uint_ty),
                Some(len2),
                vec![Operand::LiteralBit32(2)],
            ),
        );
        defs.insert(
            float_array_ty,
            Instruction::new(
                Op::TypeArray,
                None,
                Some(float_array_ty),
                vec![Operand::IdRef(float_ty), Operand::IdRef(len3)],
            ),
        );
        defs.insert(
            vec3_array_ty,
            Instruction::new(
                Op::TypeArray,
                None,
                Some(vec3_array_ty),
                vec![Operand::IdRef(vec3_ty), Operand::IdRef(len2)],
            ),
        );

        let rule = SpirvLayout::Natural;
        assert_eq!(spirv_size_align(float_ty, &defs, rule), (4, 4));
        assert_eq!(spirv_size_align(vec2_ty, &defs, rule), (8, 8));
        assert_eq!(spirv_size_align(vec3_ty, &defs, rule), (12, 16));
        assert_eq!(spirv_size_align(vec4_ty, &defs, rule), (16, 16));
        assert_eq!(spirv_size_align(float_array_ty, &defs, rule), (12, 4));
        assert_eq!(spirv_size_align(vec3_array_ty, &defs, rule), (32, 16));
        assert_eq!(spirv_size_align(999, &defs, rule), (4, 4));
    }

    #[test]
    fn spirv_natural_struct_consumes_vector_allocation_size() {
        // A three-lane vector stores 3/6/12 bytes but allocates 4/8/16 under the AIR datalayout
        // (`v24:32:32`, `v48:64:64`, `v96:128:128`), so the member after it starts at the
        // allocation boundary and the struct extends past the vector's store size.
        let uchar_ty = 1u32;
        let v3uchar_ty = 2u32;
        let struct_ty = 3u32;
        let mut defs = HashMap::new();
        defs.insert(
            uchar_ty,
            Instruction::new(
                Op::TypeInt,
                None,
                Some(uchar_ty),
                vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
            ),
        );
        defs.insert(
            v3uchar_ty,
            Instruction::new(
                Op::TypeVector,
                None,
                Some(v3uchar_ty),
                vec![Operand::IdRef(uchar_ty), Operand::LiteralBit32(3)],
            ),
        );
        defs.insert(
            struct_ty,
            Instruction::new(
                Op::TypeStruct,
                None,
                Some(struct_ty),
                vec![
                    Operand::IdRef(uchar_ty),
                    Operand::IdRef(v3uchar_ty),
                    Operand::IdRef(uchar_ty),
                ],
            ),
        );

        let rule = SpirvLayout::Natural;
        assert_eq!(spirv_size_align(v3uchar_ty, &defs, rule), (3, 4));
        // Members at 0, 4, 8: the trailing byte follows the vector's four-byte allocation, not its
        // three-byte store size, so the struct is 12 bytes rather than 8.
        assert_eq!(spirv_size_align(struct_ty, &defs, rule), (12, 4));
    }
}
