//! The single definition of what marks an AIR static initializer.
//!
//! Metal compiles a translation unit's namespace-scope initialization — most importantly the
//! `[[function_constant]]` default-state stores that gate whole regions of a shader — into one
//! parameterless function that AIR places in `section "air.static_init"`. Recognising that function
//! is load-bearing: `native::ir::static_init` folds its stores into the globals they target,
//! several inliners hold it live as an implicit emitter root because its call is injected after
//! function emission, and `meta::globals` reads it to recover a function constant's default value.
//! Miss it and an off-by-default region stays live, which for a real class of shaders turns a
//! translate into a FALLBACK (`tests/must_fallback.rs` pins both directions).
//!
//! Every one of those readers used to key on the function being *named* `_GLOBAL__sub_I…`. That is
//! the Itanium C++ ABI's mangling for a translation-unit initializer — a frontend detail standing in
//! for an ABI contract, and exactly the name-keyed branch AGENTS.md ("structure and semantics over
//! names") says to replace when a structural test exists. One does: the `air.static_init` section
//! is the AIR ABI's own statement of what the function is, and `air.*` is a stable ABI family the
//! same rule permits dispatching on. Over a 2880-source corpus sample the two markers are exactly
//! co-extensive — 972 sources carry both on every static initializer, none carries either alone —
//! so the section is not a narrower test, it is the same test stated honestly. Reading it also
//! drops the false positive the name test carried, where an ordinary function that happened to be
//! named `_GLOBAL__sub_I…` had its stores folded into globals as if it initialized them.
//!
//! Both forms below answer the same question, so a reader working on IR text and a reader working
//! on parsed functions cannot drift apart.

/// The section AIR places a translation unit's static initializer in.
pub(crate) const AIR_STATIC_INIT_SECTION: &str = "air.static_init";

/// Whether a `define` header line declares an AIR static initializer.
///
/// For readers that still work on IR text. The section attribute follows the parameter list, so the
/// search starts after it: a parameter default or a quoted symbol name is not an attribute, and
/// scanning the whole line would let one masquerade as the marker.
pub(crate) fn define_line_declares_static_initializer(line: &str) -> bool {
    attribute_tail(line).is_some_and(tail_declares_static_init_section)
}

/// Whether the attribute tail of a `define` header — everything after the parameter list — carries
/// the `air.static_init` section attribute.
pub(crate) fn tail_declares_static_init_section(tail: &str) -> bool {
    let mut rest = tail;
    while let Some(at) = rest.find("section ") {
        let after = rest[at + "section ".len()..].trim_start();
        if let Some(quoted) = after.strip_prefix('"') {
            if let Some(end) = quoted.find('"') {
                if &quoted[..end] == AIR_STATIC_INIT_SECTION {
                    return true;
                }
            }
        }
        rest = &rest[at + "section ".len()..];
    }
    false
}

/// The part of a `define` header line that follows the parameter list, or `None` if the line has no
/// balanced parameter list. Quoted spans are skipped so a `(` or `)` inside a quoted symbol name
/// cannot unbalance the scan.
fn attribute_tail(line: &str) -> Option<&str> {
    let mut depth = 0usize;
    let mut in_quotes = false;
    let mut escaped = false;
    for (offset, ch) in line.char_indices() {
        if in_quotes {
            match ch {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_quotes = false,
                _ => {}
            }
            continue;
        }
        match ch {
            '"' => in_quotes = true,
            '(' => depth += 1,
            ')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(&line[offset + 1..]);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_air_section_is_what_marks_an_initializer() {
        assert!(define_line_declares_static_initializer(
            "define internal void @_GLOBAL__sub_I_x() #0 section \"air.static_init\" {"
        ));
        assert!(
            define_line_declares_static_initializer(
                "define internal void @ctor() section \"air.static_init\" {"
            ),
            "the section, not the Itanium name, is the marker"
        );
        assert!(
            !define_line_declares_static_initializer(
                "define internal void @_GLOBAL__sub_I_x() #0 {"
            ),
            "the Itanium name alone must not stand in for the section"
        );
        assert!(!define_line_declares_static_initializer(
            "define internal void @f() section \"air.fc_initializer\" {"
        ));
    }

    #[test]
    fn the_marker_is_only_read_from_the_attribute_tail() {
        assert!(
            !define_line_declares_static_initializer(
                "define void @\"section \\\"air.static_init\\\"\"(i32 %x) {"
            ),
            "a quoted symbol name is not an attribute"
        );
        assert!(
            !define_line_declares_static_initializer(
                "define void @f(ptr %p, ptr %section \"air.static_init\") {"
            ),
            "the parameter list is not the attribute tail"
        );
        assert!(
            define_line_declares_static_initializer(
                "define void @\"odd(name)\"() section \"air.static_init\" {"
            ),
            "parentheses inside a quoted symbol must not unbalance the scan"
        );
    }

    #[test]
    fn a_line_without_a_balanced_parameter_list_declares_nothing() {
        assert!(!define_line_declares_static_initializer(
            "define void @f( section \"air.static_init\""
        ));
        assert!(!define_line_declares_static_initializer(""));
    }
}
