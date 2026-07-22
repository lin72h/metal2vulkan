use super::*;
mod gep_linearize;
mod phi_compare;
mod phi_merge;
mod phi_provenance;
mod raw_select;
mod select;
mod select_gep;

/// Whether `ty` is a plain scalar (the "part" side of the whole-vs-part upgrade): an integer, a
/// float of any width, or a bool. Composites (vector/array/struct/pointer) and unresolved aliases
/// are not scalars.
pub(super) fn is_scalar_pointee(ty: &LlType) -> bool {
    matches!(
        ty,
        LlType::Int(_) | LlType::Float | LlType::Half | LlType::BFloat | LlType::Bool
    )
}

/// Bit width of a scalar `LlType`, or `None` for a non-scalar / width-less type. Used by the
/// M-A2(a) reinterpret upgrade to require the carrier be the SAME width as the recorded pointee
/// (a legal bit-reinterpret). `Bool` has no defined storage width here, so it is excluded.
fn scalar_bit_width(ty: &LlType) -> Option<u32> {
    match ty {
        LlType::Float => Some(32),
        LlType::Half | LlType::BFloat => Some(16),
        LlType::Int(w) => Some(*w),
        _ => None,
    }
}

/// M-A2(b) whole-vs-part shape test: whether the (already-resolved) `carrier` is the WHOLE composite
/// `Vector(S,N)` / `[N x S]` whose element is exactly the recorded `scalar_pointee` `S` — a pure
/// part→whole granularity widening. Any other carrier shape (different element, struct, nested, scalar)
/// is out of the family.
fn whole_part_widens(carrier: &LlType, scalar_pointee: &LlType) -> bool {
    match carrier {
        LlType::Vector(elem, _) | LlType::Array(elem, _) => elem.as_ref() == scalar_pointee,
        _ => false,
    }
}

/// M-A2(a) reinterpret shape test: whether the (already-resolved) `carrier` is a DIFFERENT scalar of
/// the SAME bit width as the recorded `scalar_pointee` (a legal bit-reinterpret, e.g.
/// `Float(32)`↔`Int(32)`). NOTE this family is proven UNSOUND to prefer (dead-end #14) — the test only
/// gates the default-off diagnostic flag.
fn reinterp_compatible(carrier: &LlType, scalar_pointee: &LlType) -> bool {
    is_scalar_pointee(carrier)
        && carrier != scalar_pointee
        && match (scalar_bit_width(carrier), scalar_bit_width(scalar_pointee)) {
            (Some(cw), Some(pw)) => cw == pw,
            _ => false,
        }
}

fn pointer_arithmetic_access_chain_op_for_storage(
    storage: StorageClass,
    base_is_indexed_container: bool,
    pointee: &LlType,
    indices: &[TypedValue],
) -> Op {
    let scalar_pointee = !matches!(pointee, LlType::Array(_, _) | LlType::Struct(_));
    if scalar_pointee
        && ptr_access_chain_allowed_storage(storage)
        && !base_is_indexed_container
        && indices.first().and_then(|idx| const_index(Some(idx))) != Some(0)
    {
        Op::PtrAccessChain
    } else {
        Op::InBoundsAccessChain
    }
}

#[cfg(test)]
mod pointee_upgrade_tests {
    //! Coverage for the M-A2 pointee-upgrade shape predicates (`is_scalar_pointee`,
    //! `scalar_bit_width`, `whole_part_widens`, `reinterp_compatible`). These decide whether the
    //! default-off byte→real / whole-vs-part / reinterpret flags upgrade a pointer's pointee; getting
    //! the shape test wrong would silently mis-upgrade (or fail to upgrade) at the `pointer_pointee_for_value`
    //! seam. Pure functions on already-resolved `LlType`s, so no emitter state is needed.
    use super::{is_scalar_pointee, reinterp_compatible, scalar_bit_width, whole_part_widens};
    use crate::native::ir::LlType;

    fn vec(elem: LlType, n: u32) -> LlType {
        LlType::Vector(Box::new(elem), n)
    }
    fn arr(elem: LlType, n: u32) -> LlType {
        LlType::Array(Box::new(elem), n)
    }

    #[test]
    fn scalars_are_scalar_composites_are_not() {
        for s in [
            LlType::Float,
            LlType::Half,
            LlType::BFloat,
            LlType::Bool,
            LlType::Int(32),
            LlType::Int(16),
            LlType::Int(8),
        ] {
            assert!(is_scalar_pointee(&s), "{s:?} should be scalar");
        }
        for c in [
            vec(LlType::Float, 4),
            arr(LlType::Float, 8),
            LlType::Struct(vec![LlType::Half, LlType::Half]),
            LlType::Ptr(1),
            LlType::Void,
            LlType::Named("x".into()),
        ] {
            assert!(!is_scalar_pointee(&c), "{c:?} should not be scalar");
        }
    }

    #[test]
    fn bit_widths() {
        assert_eq!(scalar_bit_width(&LlType::Float), Some(32));
        assert_eq!(scalar_bit_width(&LlType::Int(32)), Some(32));
        assert_eq!(scalar_bit_width(&LlType::Half), Some(16));
        assert_eq!(scalar_bit_width(&LlType::BFloat), Some(16));
        assert_eq!(scalar_bit_width(&LlType::Int(16)), Some(16));
        assert_eq!(scalar_bit_width(&LlType::Int(8)), Some(8));
        // Bool has no defined storage width here; composites are width-less.
        assert_eq!(scalar_bit_width(&LlType::Bool), None);
        assert_eq!(scalar_bit_width(&vec(LlType::Float, 4)), None);
        assert_eq!(scalar_bit_width(&LlType::Ptr(1)), None);
    }

    #[test]
    fn whole_part_upgrades_only_matching_element_composite() {
        // Vector/array WHOSE ELEMENT is the recorded scalar → widen.
        assert!(whole_part_widens(&vec(LlType::Float, 4), &LlType::Float));
        assert!(whole_part_widens(&arr(LlType::Half, 8), &LlType::Half));
        assert!(whole_part_widens(
            &vec(LlType::Int(32), 2),
            &LlType::Int(32)
        ));
        // Element mismatch, scalar carrier, or struct → do NOT widen.
        assert!(!whole_part_widens(&vec(LlType::Int(32), 4), &LlType::Float));
        assert!(!whole_part_widens(&LlType::Float, &LlType::Float));
        assert!(!whole_part_widens(
            &LlType::Struct(vec![LlType::Float]),
            &LlType::Float
        ));
    }

    #[test]
    fn reinterp_requires_same_width_different_scalar() {
        // Same width, different kind → reinterpret-compatible.
        assert!(reinterp_compatible(&LlType::Int(32), &LlType::Float));
        assert!(reinterp_compatible(&LlType::Float, &LlType::Int(32)));
        assert!(reinterp_compatible(&LlType::Int(16), &LlType::Half));
        assert!(reinterp_compatible(&LlType::Half, &LlType::Int(16)));
        // Same type → nothing to reinterpret.
        assert!(!reinterp_compatible(&LlType::Float, &LlType::Float));
        // Different width → not a legal reinterpret.
        assert!(!reinterp_compatible(&LlType::Int(16), &LlType::Float));
        assert!(!reinterp_compatible(&LlType::Half, &LlType::Int(32)));
        // Composite carrier / Bool → excluded (that is whole-vs-part / width-less).
        assert!(!reinterp_compatible(&vec(LlType::Float, 4), &LlType::Float));
        assert!(!reinterp_compatible(&LlType::Bool, &LlType::Int(8)));
    }
}
