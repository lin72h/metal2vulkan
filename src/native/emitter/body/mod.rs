use super::*;

mod calls;
mod core_inst;
mod gep_provenance;
mod memcpy_memset;
mod reinterpret_pointer;
mod vector_load;
mod vector_store;
mod word_access;

/// The already-computed predicates a GEP's access-chain-opcode decision reads (refactor S7). Grouped
/// into a struct so [`decide_ptr_access_chain`] stays a pure, unit-testable function of its inputs;
/// `is_indexed_container_root` is threaded in as a bool because computing it needs `&self`.
pub(super) struct PtrAccessChainInputs {
    pub pointee_points_to_aggregate: bool,
    pub base_storage: StorageClass,
    pub is_indexed_container_root: bool,
    pub struct_base_stride: bool,
    pub array_base_stride: bool,
    /// The GEP's leading index is not a constant `0` (`const_index(indices[0]) != Some(0)`).
    pub first_index_nonzero: bool,
    pub was_composed: bool,
    pub base_is_param: bool,
    pub base_points_to_aggregate: bool,
    pub base_is_pointer_phi: bool,
}

/// Decide `OpPtrAccessChain` (`true`) vs `OpInBoundsAccessChain` (`false`) for a getelementptr.
///
/// `scalar_ptr_access_chain` is the "this GEP indexes a scalar/byte root with a real stride, on a
/// storage class that permits `OpPtrAccessChain`" predicate; the outer conjunction then only lets a
/// composed / entry-param / aggregate base use `OpPtrAccessChain` when that scalar predicate holds
/// (otherwise those must stay on `OpInBoundsAccessChain`), and a leading constant-`0` index always
/// falls back to `OpInBoundsAccessChain`. Extracted verbatim from `emit_gep_result` so the
/// floor-tuned reasoning (the array-stride ≥2-index gate, the storage allow-list) lives in one
/// unit-tested place.
///
/// Kept as three explicit `<unsafe-property> || scalar_ptr_access_chain` guards (one per base
/// property that OpPtrAccessChain would otherwise forbid) rather than the factored
/// `scalar_ptr_access_chain || (…&&…&&…)` — the per-property form documents which base shapes the
/// scalar predicate excuses, which is the load-bearing reasoning here.
#[allow(clippy::nonminimal_bool)]
pub(super) fn decide_ptr_access_chain(i: PtrAccessChainInputs) -> bool {
    let scalar_ptr_access_chain = !i.pointee_points_to_aggregate
        && ptr_access_chain_allowed_storage(i.base_storage)
        && (!i.is_indexed_container_root || i.struct_base_stride || i.array_base_stride)
        && i.first_index_nonzero;
    (!i.base_is_pointer_phi || ptr_access_chain_allowed_storage(i.base_storage))
        && (!i.was_composed || scalar_ptr_access_chain)
        && (!i.base_is_param || scalar_ptr_access_chain)
        && (!i.base_points_to_aggregate || scalar_ptr_access_chain)
        && i.first_index_nonzero
}

/// Which resolved binary emitter an opcode routes to under the graph walk (`emit_body_inst`). The three
/// variants distinguish logical/unsigned integer ops, IEEE float ops, and the SIGNED integer ops that
/// emit through a signed-typed operand pair.
#[derive(Clone, Copy)]
enum BinaryKind {
    Int,
    Float,
    Signed,
}

/// The `(SPIR-V Op, emitter kind)` a two-operand arithmetic/bitwise opcode dispatches to, or `None` for
/// any non-binary opcode. This is the opcode→emitter table the graph walk consults
/// (`"add" => (Op::IAdd, Int)`, etc.). `fneg`/`freeze` (unary) and
/// `icmp`/`fcmp` (predicate-carrying) are deliberately excluded — they are not part of this M-A4 family.
fn binary_op_dispatch(opcode: &str) -> Option<(Op, BinaryKind)> {
    let entry = match opcode {
        "fmul" => (Op::FMul, BinaryKind::Float),
        "fadd" => (Op::FAdd, BinaryKind::Float),
        "fsub" => (Op::FSub, BinaryKind::Float),
        "fdiv" => (Op::FDiv, BinaryKind::Float),
        "add" => (Op::IAdd, BinaryKind::Int),
        "sub" => (Op::ISub, BinaryKind::Int),
        "mul" => (Op::IMul, BinaryKind::Int),
        "udiv" => (Op::UDiv, BinaryKind::Int),
        "urem" => (Op::UMod, BinaryKind::Int),
        "shl" => (Op::ShiftLeftLogical, BinaryKind::Int),
        "lshr" => (Op::ShiftRightLogical, BinaryKind::Int),
        "ashr" => (Op::ShiftRightArithmetic, BinaryKind::Int),
        "and" => (Op::BitwiseAnd, BinaryKind::Int),
        "or" => (Op::BitwiseOr, BinaryKind::Int),
        "xor" => (Op::BitwiseXor, BinaryKind::Int),
        "sdiv" => (Op::SDiv, BinaryKind::Signed),
        "srem" => (Op::SRem, BinaryKind::Signed),
        _ => return None,
    };
    Some(entry)
}

/// Which resolved single-operand emitter a `fneg`/`freeze` opcode routes to under the graph walk.
/// `Fneg` emits through `Op::FNegate`; `Freeze` is a pure copy.
#[derive(Clone, Copy)]
enum UnaryKind {
    Fneg,
    Freeze,
}

/// The single-operand emitter a `fneg`/`freeze` opcode dispatches to, or `None` for any other opcode.
/// Deliberately narrow: the type CONVERSIONS (`zext`/`sext`/`trunc`/`fptrunc`/`fpext`/`sitofp`/`uitofp`)
/// are also single-operand but split their `rest` into `<src> to <dst>` and source the dest type
/// separately, so they are a separate dispatch family (`convert_op_dispatch`), not part of this one.
fn unary_op_dispatch(opcode: &str) -> Option<UnaryKind> {
    match opcode {
        "fneg" => Some(UnaryKind::Fneg),
        "freeze" => Some(UnaryKind::Freeze),
        _ => None,
    }
}

/// Which resolved conversion emitter a single-operand type-conversion opcode routes to under the graph
/// walk. The variants distinguish integer resize (`zext`/`sext`/`trunc`, carrying the SPIR-V `Op`),
/// float resize (`fptrunc`/`fpext`), and integer→float (`sitofp`/`uitofp`, carrying the `Op`).
#[derive(Clone, Copy)]
enum ConvertKind {
    Int(Op),
    Float,
    IntToFloat(Op),
}

/// The conversion emitter a `zext`/`sext`/`trunc`/`fptrunc`/`fpext`/`sitofp`/`uitofp` opcode dispatches
/// to, or `None` for any other opcode (`"zext" => Int(Op::UConvert)`, etc.).
fn convert_op_dispatch(opcode: &str) -> Option<ConvertKind> {
    let kind = match opcode {
        "zext" => ConvertKind::Int(Op::UConvert),
        "sext" => ConvertKind::Int(Op::SConvert),
        "trunc" => ConvertKind::Int(Op::UConvert),
        "fptrunc" | "fpext" => ConvertKind::Float,
        "sitofp" => ConvertKind::IntToFloat(Op::ConvertSToF),
        "uitofp" => ConvertKind::IntToFloat(Op::ConvertUToF),
        _ => return None,
    };
    Some(kind)
}

fn typed_value_is_zero(value: &TypedValue) -> bool {
    matches!(
        value.value,
        LlValue::Int(0) | LlValue::SignedInt(0) | LlValue::Hex(0) | LlValue::Zero
    )
}

fn typed_value_u64(value: &TypedValue) -> Option<u64> {
    match value.value {
        LlValue::Int(value) | LlValue::Hex(value) => Some(value),
        LlValue::SignedInt(value) if value >= 0 => Some(value as u64),
        _ => None,
    }
}

fn is_copy_memory_aggregate(ty: &LlType) -> bool {
    matches!(ty, LlType::Array(_, _) | LlType::Struct(_))
}

fn is_interface_backed_copy_storage(storage: StorageClass) -> bool {
    matches!(
        storage,
        StorageClass::UniformConstant | StorageClass::StorageBuffer
    )
}

fn align_reinterpreted_workgroup_type(
    base: &mut LlType,
    source: &LlType,
    indices: &mut Vec<TypedValue>,
) -> bool {
    loop {
        if types_compatible(base, source) {
            return true;
        }
        match (&*base, source) {
            (LlType::Array(_, base_len), LlType::Array(_, source_len)) => {
                return base_len == source_len;
            }
            (LlType::Vector(_, base_lanes), LlType::Vector(_, source_lanes)) => {
                return base_lanes == source_lanes;
            }
            (LlType::Struct(base_fields), LlType::Struct(source_fields)) => {
                return base_fields.len() == source_fields.len();
            }
            (LlType::Struct(fields), _) => {
                let Some(first) = fields.first().cloned() else {
                    return false;
                };
                indices.push(TypedValue {
                    ty: LlType::Int(32),
                    value: LlValue::Int(0),
                });
                *base = first;
            }
            _ => return false,
        }
    }
}

fn aggregate_member0_wraps_source(base: &LlType, source: &LlType) -> bool {
    match base {
        LlType::Array(elem, _) => types_compatible(elem, source),
        LlType::Struct(fields) => fields
            .first()
            .is_some_and(|field| types_compatible(field, source)),
        _ => false,
    }
}

fn aggregate_member0_array_element_wraps_source(base: &LlType, source: &LlType) -> bool {
    match base {
        LlType::Struct(fields) => fields.first().is_some_and(
            |field| matches!(field, LlType::Array(elem, _) if types_compatible(elem, source)),
        ),
        _ => false,
    }
}

fn is_i32_pair_struct(ty: &LlType) -> bool {
    matches!(
        ty,
        LlType::Struct(fields)
            if fields.len() == 2
                && fields[0] == LlType::Int(32)
                && fields[1] == LlType::Int(32)
    )
}

fn is_i8_array_type(ty: &LlType) -> bool {
    matches!(ty, LlType::Array(elem, _) if elem.as_ref() == &LlType::Int(8))
}

fn is_scalar_storage_type(ty: &LlType) -> bool {
    matches!(
        ty,
        LlType::Bool | LlType::Half | LlType::BFloat | LlType::Float | LlType::Int(_)
    )
}

fn type_contains_pointer(ty: &LlType) -> bool {
    match ty {
        LlType::Ptr(_) => true,
        LlType::Vector(elem, _) | LlType::Array(elem, _) => type_contains_pointer(elem),
        LlType::Struct(fields) => fields.iter().any(type_contains_pointer),
        _ => false,
    }
}

fn first_pointer_access_path(ty: &LlType) -> Option<(LlType, Vec<u32>)> {
    // Pointer leaves are never carried inside a vector here, so do not descend vectors (refactor S5).
    first_aggregate_leaf(ty, &|t| matches!(t, LlType::Ptr(_)), false)
}

fn narrowing_vector_store_target(
    pointee: &LlType,
    object_ty: &LlType,
) -> Option<(LlType, Vec<u32>)> {
    let (target, indices) = leading_vector_store_target(pointee)?;
    let LlType::Vector(target_elem, target_lanes) = &target else {
        return None;
    };
    let LlType::Vector(object_elem, object_lanes) = object_ty else {
        return None;
    };
    (*object_lanes > *target_lanes && types_compatible(target_elem, object_elem))
        .then_some((target, indices))
}

fn leading_vector_store_target(pointee: &LlType) -> Option<(LlType, Vec<u32>)> {
    // The vector itself is the leaf, so vector-descent never triggers (refactor S5).
    first_aggregate_leaf(pointee, &|t| matches!(t, LlType::Vector(_, _)), false)
}

#[cfg(test)]
mod binary_dispatch_tests {
    //! Guard the structured binary dispatch table (`binary_op_dispatch`) against drift: every mnemonic
    //! must map to the SAME SPIR-V `Op` and the same emitter kind, or the graph walk would silently emit
    //! wrong bytes for that arithmetic/bitwise op.
    use super::{
        binary_op_dispatch, convert_op_dispatch, unary_op_dispatch, BinaryKind, ConvertKind,
        UnaryKind,
    };
    use spirv::Op;

    fn kind_tag(k: BinaryKind) -> &'static str {
        match k {
            BinaryKind::Int => "int",
            BinaryKind::Float => "float",
            BinaryKind::Signed => "signed",
        }
    }

    #[test]
    fn every_binary_mnemonic_maps_as_the_text_path_does() {
        // (opcode, expected Op, expected kind) — mirrors the `match opcode` arms verbatim.
        let table: &[(&str, Op, &str)] = &[
            ("fmul", Op::FMul, "float"),
            ("fadd", Op::FAdd, "float"),
            ("fsub", Op::FSub, "float"),
            ("fdiv", Op::FDiv, "float"),
            ("add", Op::IAdd, "int"),
            ("sub", Op::ISub, "int"),
            ("mul", Op::IMul, "int"),
            ("udiv", Op::UDiv, "int"),
            ("urem", Op::UMod, "int"),
            ("shl", Op::ShiftLeftLogical, "int"),
            ("lshr", Op::ShiftRightLogical, "int"),
            ("ashr", Op::ShiftRightArithmetic, "int"),
            ("and", Op::BitwiseAnd, "int"),
            ("or", Op::BitwiseOr, "int"),
            ("xor", Op::BitwiseXor, "int"),
            ("sdiv", Op::SDiv, "signed"),
            ("srem", Op::SRem, "signed"),
        ];
        for (opcode, op, kind) in table {
            let (got_op, got_kind) = binary_op_dispatch(opcode)
                .unwrap_or_else(|| panic!("binary_op_dispatch({opcode}) returned None"));
            assert_eq!(got_op, *op, "Op mismatch for {opcode}");
            assert_eq!(kind_tag(got_kind), *kind, "kind mismatch for {opcode}");
        }
    }

    #[test]
    fn non_binary_opcodes_are_not_dispatched() {
        // Unary / predicate / memory / control opcodes are NOT part of this family and must fall through
        // to the text substrate (so their own migration gates stay in effect).
        for opcode in [
            "fneg",
            "freeze",
            "icmp",
            "fcmp",
            "select",
            "load",
            "store",
            "getelementptr",
            "phi",
            "bitcast",
            "zext",
            "call",
            "alloca",
            "",
        ] {
            assert!(
                binary_op_dispatch(opcode).is_none(),
                "{opcode} must not route through the binary dispatch"
            );
        }
    }

    #[test]
    fn unary_dispatch_covers_only_fneg_and_freeze() {
        assert!(matches!(unary_op_dispatch("fneg"), Some(UnaryKind::Fneg)));
        assert!(matches!(
            unary_op_dispatch("freeze"),
            Some(UnaryKind::Freeze)
        ));
        // Conversions are single-operand but NOT part of this family (they split src/dst); everything
        // else must also decline so it stays on the text substrate.
        for opcode in [
            "zext", "sext", "trunc", "fptrunc", "sitofp", "add", "load", "select", "",
        ] {
            assert!(
                unary_op_dispatch(opcode).is_none(),
                "{opcode} must not route through the unary dispatch"
            );
        }
    }

    #[test]
    fn convert_dispatch_maps_as_the_text_path_does() {
        assert!(matches!(
            convert_op_dispatch("zext"),
            Some(ConvertKind::Int(Op::UConvert))
        ));
        assert!(matches!(
            convert_op_dispatch("sext"),
            Some(ConvertKind::Int(Op::SConvert))
        ));
        assert!(matches!(
            convert_op_dispatch("trunc"),
            Some(ConvertKind::Int(Op::UConvert))
        ));
        assert!(matches!(
            convert_op_dispatch("fptrunc"),
            Some(ConvertKind::Float)
        ));
        assert!(matches!(
            convert_op_dispatch("fpext"),
            Some(ConvertKind::Float)
        ));
        assert!(matches!(
            convert_op_dispatch("sitofp"),
            Some(ConvertKind::IntToFloat(Op::ConvertSToF))
        ));
        assert!(matches!(
            convert_op_dispatch("uitofp"),
            Some(ConvertKind::IntToFloat(Op::ConvertUToF))
        ));
        // `fptoui`/`fptosi`/`ptrtoint`/`inttoptr` are conversions the text path handles elsewhere (or
        // not at all) — they must NOT route through this family.
        for opcode in [
            "fptoui", "fptosi", "ptrtoint", "inttoptr", "add", "load", "",
        ] {
            assert!(
                convert_op_dispatch(opcode).is_none(),
                "{opcode} must not route through the conversion dispatch"
            );
        }
    }
}

#[cfg(test)]
mod ptr_access_chain_tests {
    //! Unit tests for the GEP access-chain-opcode decision (refactor S7). `true` => OpPtrAccessChain,
    //! `false` => OpInBoundsAccessChain.
    use super::{decide_ptr_access_chain, PtrAccessChainInputs};
    use spirv::StorageClass;

    /// A base case that DOES take OpPtrAccessChain: a scalar StorageBuffer root, non-zero leading
    /// index, not an indexed-container root, no composed/param/aggregate constraint.
    fn scalar_ptr_chain() -> PtrAccessChainInputs {
        PtrAccessChainInputs {
            pointee_points_to_aggregate: false,
            base_storage: StorageClass::StorageBuffer,
            is_indexed_container_root: false,
            struct_base_stride: false,
            array_base_stride: false,
            first_index_nonzero: true,
            was_composed: false,
            base_is_param: false,
            base_points_to_aggregate: false,
            base_is_pointer_phi: false,
        }
    }

    #[test]
    fn scalar_root_with_nonzero_index_uses_ptr_access_chain() {
        assert!(decide_ptr_access_chain(scalar_ptr_chain()));
    }

    #[test]
    fn leading_zero_index_always_falls_back_to_inbounds() {
        // A constant-0 leading index is never a stride -> OpInBoundsAccessChain, regardless.
        let i = PtrAccessChainInputs {
            first_index_nonzero: false,
            ..scalar_ptr_chain()
        };
        assert!(!decide_ptr_access_chain(i));
    }

    #[test]
    fn aggregate_pointee_forces_inbounds() {
        // pointee_points_to_aggregate kills scalar_ptr_access_chain; with no composed/param/aggregate
        // base constraint the outer conjunction still needs first_index_nonzero (true here) but the
        // scalar predicate is false, so a plain aggregate-pointee scalar-base still returns true only
        // via the (!was_composed && !base_is_param && !base_points_to_aggregate) path.
        let i = PtrAccessChainInputs {
            pointee_points_to_aggregate: true,
            ..scalar_ptr_chain()
        };
        // was_composed/base_is_param/base_points_to_aggregate all false => outer conjunction is just
        // first_index_nonzero (true). So this is true even though scalar_ptr_access_chain is false.
        assert!(decide_ptr_access_chain(i));
    }

    #[test]
    fn composed_base_needs_scalar_predicate() {
        // A composed base may only use OpPtrAccessChain when scalar_ptr_access_chain holds.
        let composed_not_scalar = PtrAccessChainInputs {
            was_composed: true,
            pointee_points_to_aggregate: true, // kills scalar predicate
            ..scalar_ptr_chain()
        };
        assert!(!decide_ptr_access_chain(composed_not_scalar));
        let composed_scalar = PtrAccessChainInputs {
            was_composed: true,
            ..scalar_ptr_chain() // scalar predicate holds
        };
        assert!(decide_ptr_access_chain(composed_scalar));
    }

    #[test]
    fn disallowed_storage_class_blocks_scalar_predicate() {
        // Private is not in ptr_access_chain_allowed_storage, so scalar_ptr_access_chain is false;
        // with an entry-param base that means OpInBoundsAccessChain.
        let i = PtrAccessChainInputs {
            base_storage: StorageClass::Private,
            base_is_param: true,
            ..scalar_ptr_chain()
        };
        assert!(!decide_ptr_access_chain(i));
    }

    #[test]
    fn pointer_phi_base_needs_ptr_access_chain_storage() {
        let i = PtrAccessChainInputs {
            base_storage: StorageClass::Private,
            base_is_pointer_phi: true,
            ..scalar_ptr_chain()
        };
        assert!(!decide_ptr_access_chain(i));
    }

    #[test]
    fn indexed_container_root_needs_a_stride() {
        // An indexed-container root defeats the scalar predicate UNLESS a struct/array stride is set.
        let root_no_stride = PtrAccessChainInputs {
            is_indexed_container_root: true,
            base_is_param: true,
            ..scalar_ptr_chain()
        };
        assert!(!decide_ptr_access_chain(root_no_stride));
        let root_with_struct_stride = PtrAccessChainInputs {
            is_indexed_container_root: true,
            struct_base_stride: true,
            base_is_param: true,
            ..scalar_ptr_chain()
        };
        assert!(decide_ptr_access_chain(root_with_struct_stride));
    }
}
