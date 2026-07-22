//! Exact-inverse text rendering of the typed IR (`LlType` / `LlValue`) — the reverse of
//! `parse::parse_type` / `parse::parse_value`. It lets a structurizer synthesizer rebuild a `ret` /
//! `phi` LINE from the typed carrier, so the cross-arm return-normalization passes source their
//! terminator model from `TirBlock` (`ret`) instead of the retiring `.lines` text substrate.
//!
//! Totality is deliberately partial. `render_type` is total EXCEPT `Struct` (parse maps both `{..}`
//! and `<{..}>` to `Struct`, so the packed/non-packed distinction is lost — non-injective) and `Bool`
//! (never produced by `parse_type`, which yields `Int(1)` for `i1`). `render_value` renders only the
//! injective value shapes (`Local` / `Global` / `Bool` / `Int` / `SignedInt`); every literal whose
//! parse is lossy or ambiguous (`Undef`↔`poison`, `Zero`↔`null`/`zeroinitializer`, `Hex`, floats,
//! aggregates, `Splat`, `Gep`) returns `None`, so a caller that cannot faithfully rebuild the text
//! DECLINES the refinement rather than emitting divergent bytes.

use crate::native::ir::{LlType, LlValue};

/// Render an `LlType` back to its LLVM-IR text — the exact inverse of `parse::parse_type`. `None` for
/// the non-injective / never-parsed shapes (`Struct`, `Bool`), which propagates through the aggregate
/// arms so a `Vector`/`Array` over an unrenderable element also yields `None`.
pub(in crate::native) fn render_type(ty: &LlType) -> Option<String> {
    Some(match ty {
        LlType::Void => "void".to_string(),
        LlType::Float => "float".to_string(),
        LlType::Half => "half".to_string(),
        LlType::BFloat => "bfloat".to_string(),
        LlType::Int(n) => format!("i{n}"),
        LlType::Ptr(0) => "ptr".to_string(),
        LlType::Ptr(n) => format!("ptr addrspace({n})"),
        LlType::Vector(elem, n) => format!("<{n} x {}>", render_type(elem)?),
        LlType::Array(elem, n) => format!("[{n} x {}]", render_type(elem)?),
        LlType::Named(name) => name.clone(),
        LlType::Struct(_) | LlType::Bool => return None,
    })
}

/// Render an `LlValue` back to its LLVM-IR text — the exact inverse of the injective cases of
/// `parse::parse_value`. `None` for every lossy / ambiguous shape (see the module docs). `Local` /
/// `Global` carry their leading `%` / `@` verbatim (that is how `parse_value` stores them), so the
/// render is the identity on the stored string.
pub(in crate::native) fn render_value(value: &LlValue) -> Option<String> {
    Some(match value {
        LlValue::Local(name) => name.clone(),
        LlValue::Global(name) => name.clone(),
        LlValue::Bool(true) => "true".to_string(),
        LlValue::Bool(false) => "false".to_string(),
        LlValue::Int(v) => v.to_string(),
        LlValue::SignedInt(v) => v.to_string(),
        LlValue::Hex(_)
        | LlValue::Float(_)
        | LlValue::HalfBits(_)
        | LlValue::BFloatBits(_)
        | LlValue::Vector(_)
        | LlValue::Array(_)
        | LlValue::Struct(_)
        | LlValue::Splat(_)
        | LlValue::Gep(_)
        | LlValue::Zero
        | LlValue::Undef => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::parse::{parse_type, parse_value};

    /// `text -> parse_type -> render_type` is the identity on the renderable type grammar.
    #[test]
    fn render_type_round_trips_parse_type() {
        for text in [
            "void",
            "float",
            "half",
            "bfloat",
            "i1",
            "i8",
            "i32",
            "i64",
            "ptr",
            "ptr addrspace(1)",
            "ptr addrspace(3)",
            "<4 x i32>",
            "<2 x float>",
            "[8 x i8]",
            "[4 x <2 x half>]",
            "%struct.Foo",
            "%\"quoted.name\"",
        ] {
            let ty = parse_type(text).unwrap_or_else(|e| panic!("parse {text}: {e}"));
            let rendered =
                render_type(&ty).unwrap_or_else(|| panic!("render {text} returned None"));
            assert_eq!(rendered, text, "round-trip mismatch for {text}");
        }
    }

    /// Aggregate / packed-struct types are not injectively renderable.
    #[test]
    fn render_type_declines_struct() {
        let ty = parse_type("{ i32, float }").unwrap();
        assert_eq!(render_type(&ty), None);
    }

    /// `text -> parse_value -> render_value` is the identity on the injective value grammar.
    #[test]
    fn render_value_round_trips_parse_value() {
        for text in ["%x", "%0", "@g", "true", "false", "-3", "7"] {
            let v = parse_value(text).unwrap_or_else(|e| panic!("parse {text}: {e}"));
            let rendered =
                render_value(&v).unwrap_or_else(|| panic!("render {text} returned None"));
            assert_eq!(rendered, text, "round-trip mismatch for {text}");
        }
    }

    /// Lossy / ambiguous value shapes decline rendering.
    #[test]
    fn render_value_declines_ambiguous() {
        for text in ["undef", "poison", "null", "zeroinitializer", "1.5"] {
            let v = parse_value(text).unwrap();
            assert_eq!(render_value(&v), None, "expected None for {text}");
        }
    }
}
