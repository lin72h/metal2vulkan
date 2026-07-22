//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;
use crate::native::parse::{parse_memory_alignment, split_top_level};

/// `label:` (a block header) -> the `%`-prefixed label name. AIR numeric labels appear bare (`12:`),
/// matching the CFG layer's `%12` convention.
#[cfg(test)]
pub(in crate::native) fn parse_block_label(line: &str) -> Option<String> {
    let head = line.split_whitespace().next()?;
    let name = head.strip_suffix(':')?;
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
    {
        return None;
    }
    if name.starts_with('%') {
        Some(name.to_string())
    } else {
        Some(format!("%{name}"))
    }
}

/// Parse a structured terminator from a line, or `None` if it is not a terminator.
pub(in crate::native) fn parse_terminator(line: &str) -> Option<TirTerminator> {
    let line = line.trim();
    if line == "unreachable" {
        return Some(TirTerminator::Unreachable);
    }
    if let Some(rest) = line.strip_prefix("ret ") {
        // Drop any trailing `, !dbg !N` metadata, then the value is the last token (`void` -> None).
        let rest = rest.split(", !").next().unwrap_or(rest).trim();
        if rest == "void" {
            return Some(TirTerminator::Ret(None));
        }
        return Some(TirTerminator::Ret(
            rest.split_whitespace().last().map(str::to_string),
        ));
    }
    if let Some(rest) = line.strip_prefix("br ") {
        // The branch labels are every `label %X`; the optional cond is the token before the first
        // `label`. This tolerates trailing `, !llvm.loop !N` / `, !dbg !N` metadata that a strict
        // operand count would reject (the cause of most build errors on real modules).
        let labels = collect_labels(rest);
        match labels.len() {
            1 => return Some(TirTerminator::Br(labels[0].clone())),
            2 => {
                let head = &rest[..rest.find("label ").unwrap_or(rest.len())];
                let cond = head
                    .trim()
                    .trim_end_matches(',')
                    .split_whitespace()
                    .last()?
                    .to_string();
                return Some(TirTerminator::BrCond {
                    cond,
                    t: labels[0].clone(),
                    f: labels[1].clone(),
                });
            }
            _ => return None,
        }
    }
    if let Some(rest) = line.strip_prefix("switch ") {
        return parse_switch(rest);
    }
    None
}

/// Every `label %X` target in a terminator's operand text, in order. Tolerates trailing metadata.
pub(in crate::native) fn collect_labels(s: &str) -> Vec<String> {
    s.split("label ")
        .skip(1)
        .filter_map(|chunk| chunk.split([',', ' ', '\t']).next())
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

/// `switch <ty> %sel, label %default [ <ty> C, label %L  <ty> C2, label %L2 ... ]`
pub(in crate::native) fn parse_switch(rest: &str) -> Option<TirTerminator> {
    let open = rest.find('[')?;
    let close = rest.rfind(']')?;
    let head = &rest[..open];
    let head_parts = split_top_level(head, ',');
    if head_parts.len() < 2 {
        return None;
    }
    let selector = head_parts[0].split_whitespace().last()?.to_string();
    let default = head_parts[1]
        .trim()
        .strip_prefix("label ")?
        .trim()
        .to_string();
    // Case list: `<ty> <const>, label %target` entries (whitespace-separated). Walk them like the
    // canonical `parse::parse_switch`, capturing each case's constant token + target label.
    let mut cases = Vec::new();
    let mut body = rest[open + 1..close].trim();
    while !body.is_empty() {
        let (value_text, after_value) = body.split_once(',')?;
        let constant = value_text.split_whitespace().last()?.to_string();
        let after_label = after_value.trim().strip_prefix("label ")?;
        let label_end = after_label
            .find(char::is_whitespace)
            .unwrap_or(after_label.len());
        let label = after_label[..label_end].to_string();
        body = after_label[label_end..].trim();
        cases.push((constant, label));
    }
    Some(TirTerminator::Switch {
        selector,
        default,
        cases,
    })
}

/// LLVM fast-math / wrap / exact flag tokens that appear between an opcode and its first typed
/// operand (`add nsw`, `fmul fast`, `udiv exact`, `getelementptr inbounds`, …). Skipped when locating
/// where the typed operands begin.
pub(in crate::native) const OPERAND_FLAG_TOKENS: &[&str] = &[
    "nsw", "nuw", "exact", "fast", "nnan", "ninf", "nsz", "arcp", "contract", "afn", "reassoc",
    "disjoint", "volatile",
];

/// Resolve an instruction's value operands to typed form for the opcode shapes tir lowers. Returns one
/// `TirOperand` per value operand the instruction reads, in source order (an SSA `Value` carrying its
/// use-site type, a typed `Const`, or `Unresolved`). Operand layout is opcode-specific, so only the
/// regular high-frequency shapes are lowered; an opcode whose layout is not yet handled yields a single
/// `Unresolved` marker (when it has any operands at all) so operand coverage is reported honestly. This
/// is additive R3 graph data — not yet consumed by emission — proven for soundness by `tir_self_check`.
/// The comparison PREDICATE token of an `icmp`/`fcmp` line (`eq`/`ne`/`slt`/`oeq`/`olt`/...), or
/// `None` for any other opcode. The predicate is the first whitespace token after the opcode and any
/// fast-math flags (LLVM syntax: `icmp <pred> <ty> ...` / `fcmp <flags>* <pred> <ty> ...`), exactly the
/// token `icmp_predicate`/`fcmp_predicate` match against — so the emitter's `*_predicate(token)` over
/// this stored token yields the same `Op` the text path derives, byte-identically. `skip_flag_tokens`
/// strips the fast-math flags, leaving the predicate as the leading token.
pub(in crate::native) fn resolve_cmp_predicate(line: &str) -> Option<String> {
    let rhs = rhs_of(line);
    let opcode = rhs.split_whitespace().next()?;
    if opcode != "icmp" && opcode != "fcmp" {
        return None;
    }
    let after_opcode = rhs[opcode.len()..].trim_start();
    skip_flag_tokens(after_opcode)
        .split_whitespace()
        .next()
        .map(|t| t.to_string())
}

/// The explicit memory ALIGNMENT (`align N`) of a `load`/`store` line, or `None` for any other opcode
/// or when no `align` is written. Splits the after-opcode region on top-level commas and scans the
/// trailing fields (`parts[2..]`) with the same `parse_memory_alignment` the emitter calls on its own
/// `parts[2..]` (load: `load <ty>, <ptrty> <p>[, align N]`; store: `store <ty> <v>, <ptrty> <p>[,
/// align N]`), so the stored value is byte-identical to the text path's.
pub(in crate::native) fn resolve_mem_align(line: &str) -> Option<u64> {
    let rhs = rhs_of(line);
    let opcode = rhs.split_whitespace().next()?;
    if opcode != "load" && opcode != "store" {
        return None;
    }
    let after_opcode = rhs[opcode.len()..].trim_start();
    let parts = split_top_level(after_opcode, ',');
    parse_memory_alignment(parts.get(2..).unwrap_or(&[]))
        .ok()
        .flatten()
}
