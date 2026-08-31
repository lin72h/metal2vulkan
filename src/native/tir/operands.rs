//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;
use crate::native::ir::{LlGep, LlType, LlValue, TypedValue};
use crate::native::lex::{matching_paren, strip_comment};
use crate::native::parse::{
    parse_call, parse_constant_vector, parse_identity_ptr_bitcast, parse_load,
    parse_phi_incoming_values, parse_type, parse_typed_value, parse_value, parse_vector_i32_values,
    split_top_level, strip_call_prefix, LlCall, LlLoad,
};
use std::collections::HashMap;

/// Precomputed `parse_identity_ptr_bitcast` on the strip-commented line — `(result, base)` for an
/// identity pointer bitcast (`%r = bitcast T %b to T`, src-type TEXT == dst-type TEXT). Carried so the
/// pointer alias/pointee inferences read it from the typed carrier instead of re-lexing the body text.
/// Byte-identical to the inferences' `parse_identity_ptr_bitcast(strip_comment(line).trim())` by
/// construction. `None` for every other line.
pub(in crate::native) fn resolve_identity_ptr_bitcast(line: &str) -> Option<(String, String)> {
    parse_identity_ptr_bitcast(strip_comment(line).trim())
}

/// Precomputed `parse_phi_incoming_values` for a `phi` — the incoming VALUES only (no phi-type parse,
/// unlike `phi_incoming` / `parse_phi`), matching the alias inferences' lighter `parse_phi_incoming_values`
/// on the post-`phi ` rest. Carried so those inferences read it from the carrier. `None` for a non-phi
/// opcode or an unparseable incoming list.
pub(in crate::native) fn resolve_phi_incoming_values(
    line: &str,
    opcode: &str,
) -> Option<Vec<LlValue>> {
    if opcode != "phi" {
        return None;
    }
    let cleaned = strip_comment(line).trim();
    let rhs = cleaned.split_once(" = ")?.1.trim();
    let rest = rhs.strip_prefix("phi ")?;
    parse_phi_incoming_values(rest).ok()
}

/// Precomputed select arms (`%r = select cond, T a, T b`) — the parsed true/false `TypedValue` arms
/// when the line is a 3-operand select whose two arms both parse. Carried so the alias inferences read
/// them from the carrier and apply their own Ptr/Local filters. `None` otherwise (non-select, wrong
/// operand count, or a non-parsing arm).
pub(in crate::native) fn resolve_select_arms(
    line: &str,
    opcode: &str,
) -> Option<(TypedValue, TypedValue)> {
    if opcode != "select" {
        return None;
    }
    let cleaned = strip_comment(line).trim();
    let rhs = cleaned.split_once(" = ")?.1.trim();
    let rest = rhs.strip_prefix("select ")?;
    let parts = split_top_level(rest, ',');
    if parts.len() != 3 {
        return None;
    }
    Some((
        parse_typed_value(parts[1]).ok()?,
        parse_typed_value(parts[2]).ok()?,
    ))
}

/// Precomputed `parse_load` for a `load` — the full parsed load (`ptr` + `result_ty`). Carried so the
/// pointee/raw-buffer inferences read it from the carrier instead of re-lexing the line. `None` for a
/// non-load opcode or a malformed load.
pub(in crate::native) fn resolve_load_inst(line: &str, opcode: &str) -> Option<LlLoad> {
    if opcode != "load" {
        return None;
    }
    let cleaned = strip_comment(line).trim();
    let rhs = cleaned.split_once(" = ")?.1.trim();
    let rest = rhs.strip_prefix("load ")?;
    parse_load(rest).ok()
}

/// Precomputed store operands (`store T obj, T* ptr`) — the parsed `(object, ptr)` `TypedValue`s when the
/// line is a `store ` with at least two comma-parts that both parse. Carried so the raw-buffer / local-
/// pointer-table inferences read them from the carrier. `None` otherwise.
pub(in crate::native) fn resolve_store(
    line: &str,
    opcode: &str,
) -> Option<(TypedValue, TypedValue)> {
    if opcode != "store" {
        return None;
    }
    let cleaned = strip_comment(line).trim();
    let rest = cleaned.strip_prefix("store ")?;
    let parts = split_top_level(rest, ',');
    if parts.len() < 2 {
        return None;
    }
    Some((
        parse_typed_value(parts[0]).ok()?,
        parse_typed_value(parts[1]).ok()?,
    ))
}

/// Precomputed alias-call parse: the `strip_call_prefix` chain (value-call rhs after `%r = `, ELSE the
/// whole-line void call) fed to `parse_call`, swallow-on-fail. This matches the ir/ alias & call-edge
/// scans' detection, which is NARROWER than `call` / `resolve_call` (the latter also accepts
/// `musttail`/`notail`, which `strip_call_prefix` does not) — so those scans read this field, not `call`.
/// `None` for a non-call line or an indirect call.
pub(in crate::native) fn resolve_alias_call(line: &str) -> Option<LlCall> {
    let cleaned = strip_comment(line).trim();
    let call_text = cleaned
        .split_once(" = ")
        .and_then(|(_, rhs)| strip_call_prefix(rhs.trim()))
        .or_else(|| strip_call_prefix(cleaned))?;
    parse_call(call_text).ok()
}

/// Precomputed emitter call-scan parse for `infer_function_param_pointees` / `_nonnull`, PRESERVING their
/// error PROPAGATION (unlike `alias_call`'s swallow): the exact preamble `is_ignored_call_line` gate →
/// `strip_call_prefix` chain → `contains('@')` gate → `parse_call`. `None` means the line is skipped
/// (ignored void intrinsic, not a call, or an indirect `@`-less call); `Some(Ok/Err)` is the `parse_call`
/// result the scan propagates with `?`. Byte-identical to the scans' inline preamble by construction —
/// the same functions on the same `strip_comment(line).trim()`.
pub(in crate::native) fn resolve_emit_scan_call(line: &str) -> Option<Result<LlCall, String>> {
    let cleaned = strip_comment(line).trim();
    if crate::native::parse::is_ignored_call_line(cleaned) {
        return None;
    }
    let call_text = if let Some((_, rhs)) = cleaned.split_once(" = ") {
        strip_call_prefix(rhs.trim())
    } else {
        strip_call_prefix(cleaned)
    }?;
    if !call_text.contains('@') {
        return None;
    }
    Some(parse_call(call_text))
}

/// The FULL parsed `LlGep` of a `getelementptr` line (`%r = getelementptr [inbounds] <srcty>, ...`),
/// or `None` for any other opcode. Parsed via the SAME `parse_gep` the emitter and `collect_forward_geps`
/// re-lexed it with, so the stored `LlGep` (and the `gep_source_ty` sliced from it) is byte-identical to
/// the text path's; consumers read the parsed GEP from the typed graph instead of re-parsing the line.
pub(in crate::native) fn resolve_gep(line: &str) -> Option<LlGep> {
    let rhs = rhs_of(line);
    let opcode = rhs.split_whitespace().next()?;
    if opcode != "getelementptr" {
        return None;
    }
    let after_opcode = rhs[opcode.len()..].trim_start();
    crate::native::parse::parse_gep(after_opcode).ok()
}

/// The parsed `LlCall` for a direct `call`/`[must|no]tail call` line — mirrors the `resolve_call_operands`
/// opcode dispatch (drop the `call` keyword, then hand the `<ret> @callee(args)` remainder to
/// `parse_call`). `None` for a non-call line or an indirect call (`parse_call` rejects the missing
/// `@callee`); byte-identical to the emitter's own `parse_call` by construction.
pub(in crate::native) fn resolve_call(line: &str) -> Option<LlCall> {
    let rhs = rhs_of(line);
    let opcode = rhs.split_whitespace().next()?;
    let after_opcode = rhs[opcode.len()..].trim_start();
    let after_call = match opcode {
        "call" => after_opcode,
        "tail" | "musttail" | "notail" => after_opcode.strip_prefix("call ")?.trim_start(),
        _ => return None,
    };
    parse_call(after_call).ok()
}

/// The trailing constant INDEX literals of an `extractvalue`/`insertvalue` line — the integer fields
/// after the aggregate (`extractvalue`) or aggregate+element (`insertvalue`) value operands. They are
/// plain literals in the opcode text, not SSA value operands the graph lowers. Parsed here via the SAME
/// derivation (`split_once(" = ")` + strip opcode) + `split_top_level` + `parse_u32`, so the carried
/// `Vec<u32>` matches what the resolved core needs; the emitter reads it from the typed graph. `None` for
/// any other opcode, or a malformed/unparsable index list.
pub(in crate::native) fn resolve_aggregate_indices(line: &str, opcode: &str) -> Option<Vec<u32>> {
    let skip = match opcode {
        "extractvalue" => 1,
        "insertvalue" => 2,
        _ => return None,
    };
    let rest = line.split_once(" = ")?.1.trim();
    let after_opcode = rest[opcode.len()..].trim_start();
    let parts = split_top_level(after_opcode, ',');
    if parts.len() <= skip {
        return None;
    }
    parts[skip..]
        .iter()
        .map(|idx| crate::native::lex::parse_u32(idx.trim()).ok())
        .collect()
}

/// The parsed constant MASK of a `shufflevector` line — `(declared_lane_count, index_values)`. The mask
/// (`parts[2]`, a `<N x i32>` constant vector) is a text literal, not an SSA value operand the graph
/// lowers. Parsed here via the SAME derivation (`split_once(" = ")` + strip opcode) +
/// `split_top_level` shape check + the shuffle mask parse (`parse_constant_vector` /
/// `parse_vector_i32_values`), so the carried pair matches what the resolved core needs; the emitter reads
/// it from the typed graph instead of re-lexing the mask. `None` for a non-three-operand shape or a mask
/// that does not parse as `<N x i32>`. The a-operand vector check is intentionally NOT done here — it needs the resolved operand
/// type and stays on the emit side (byte-identical: it embeds the type, not the raw line).
pub(in crate::native) fn resolve_shuffle_mask(line: &str) -> Option<(u32, Vec<u32>)> {
    let rest = line.split_once(" = ")?.1.trim();
    let opcode = rest.split_whitespace().next().unwrap_or("");
    let after_opcode = rest[opcode.len()..].trim_start();
    let parts = split_top_level(after_opcode, ',');
    if parts.len() != 3 {
        return None;
    }
    let mask = parts[2];
    let TypedValue { ty: mask_ty, .. } = parse_constant_vector(mask).ok()?;
    let LlType::Vector(mask_elem, lanes) = mask_ty else {
        return None;
    };
    if *mask_elem != LlType::Int(32) {
        return None;
    }
    let indexes = parse_vector_i32_values(mask).ok()?;
    Some((lanes, indexes))
}

/// The source typed value + destination-type TEXT of a `bitcast` line (`%r = bitcast <ty> <v> to <ty2>`).
/// Parsed here via the SAME `strip_comment(...).trim()` + (`split_once(" = ")` rhs, opcode
/// token dropped) + `split_once(" to ")` + `parse_typed_value` derivation, so the carried pair matches what
/// the bitcast core needs; the emitter reads it from the typed graph instead of re-lexing. The destination
/// stays TEXT because `convert_dst_type` is a `&mut self` emit-time
/// method (it may register/convert types); the typed core runs it unchanged. The pointer copy-prop
/// side-tables stay keyed on the source operand's name (from `src.value`) exactly as before, so no
/// side-table ownership move is needed. `None` for a non-`bitcast` opcode or a malformed line makes
/// the emitter return the fail-visible unmigrated-opcode `Err`.
pub(in crate::native) fn resolve_bitcast(line: &str, opcode: &str) -> Option<(TypedValue, String)> {
    if opcode != "bitcast" {
        return None;
    }
    let cleaned = crate::native::lex::strip_comment(line).trim();
    let rhs = cleaned.split_once(" = ")?.1.trim();
    let after_opcode = rhs[opcode.len()..].trim_start();
    let (src_text, dst_text) = after_opcode.split_once(" to ")?;
    let src = parse_typed_value(src_text).ok()?;
    Some((src, dst_text.to_string()))
}

/// The operand TEXT of an `icmp` line — everything after the `icmp` mnemonic (`<pred> <ty> <a>, <b>`).
/// Carried so the POINTER-form icmp emitter can reproduce its two unsupported-form error diagnostics
/// (`ordered pointer icmp is not supported` / `pointer icmp is only supported against null`)
/// byte-identically: those embed the raw operand `rest`, which BC fingerprints as an `err:` string. Parsed
/// via `strip_comment(...).trim()` + (`split_once(" = ")` rhs, opcode token dropped), so
/// the carried string matches the diagnostic exactly; the compared VALUES come from the typed graph
/// (`operands`), never this text. `None` for a non-`icmp` opcode or a line without ` = ` (the pointer-icmp
/// emitter then has no `rest` and reports its shape error without the embedded operands).
pub(in crate::native) fn resolve_icmp_rest(line: &str, opcode: &str) -> Option<String> {
    if opcode != "icmp" {
        return None;
    }
    let cleaned = crate::native::lex::strip_comment(line).trim();
    let rhs = cleaned.split_once(" = ")?.1.trim();
    let after_opcode = rhs[opcode.len()..].trim_start();
    Some(after_opcode.to_string())
}

pub(in crate::native) fn resolve_operands(line: &str) -> Vec<TirOperand> {
    let rhs = rhs_of(line);
    let mut words = rhs.split_whitespace();
    let Some(opcode) = words.next() else {
        return Vec::new();
    };
    let after_opcode = rhs[opcode.len()..].trim_start();
    match opcode {
        // Binary arithmetic / bitwise: `<op> <flags>* <ty> <a>, <b>` — `<b>` shares `<a>`'s type.
        "add" | "sub" | "mul" | "udiv" | "sdiv" | "urem" | "srem" | "and" | "or" | "xor"
        | "shl" | "lshr" | "ashr" | "fadd" | "fsub" | "fmul" | "fdiv" | "frem" => {
            two_operands_shared_type(skip_flag_tokens(after_opcode))
        }
        // Compare: `icmp <pred> <ty> <a>, <b>` / `fcmp <flags>* <pred> <ty> <a>, <b>`. After flags the
        // next token is the predicate; the rest is the shared-type binary form.
        "icmp" | "fcmp" => {
            let rest = skip_flag_tokens(after_opcode);
            let rest = rest
                .split_once(char::is_whitespace)
                .map(|(_, r)| r)
                .unwrap_or("");
            two_operands_shared_type(rest.trim())
        }
        // `select <cty> <c>, <ty> <a>, <ty> <b>` — every field is independently typed.
        "select" => split_top_level(after_opcode, ',')
            .iter()
            .map(|c| operand_from_chunk(c.trim()))
            .collect(),
        // Conversions: `<op> <ty> <a> to <ty2>` — one value operand (`<ty> <a>`).
        "trunc" | "zext" | "sext" | "fptrunc" | "fpext" | "fptoui" | "fptosi" | "uitofp"
        | "sitofp" | "ptrtoint" | "inttoptr" | "bitcast" | "addrspacecast" => {
            let value = after_opcode
                .split(" to ")
                .next()
                .unwrap_or(after_opcode)
                .trim();
            vec![operand_from_chunk(value)]
        }
        // `freeze <ty> <a>` / `fneg <flags>* <ty> <a>` — one value operand.
        "freeze" | "fneg" => vec![operand_from_chunk(skip_flag_tokens(after_opcode))],
        // `load <flags>* <ty>, <ptrty> <p>[, align N]` — the pointer is the 2nd comma field.
        "load" => {
            let fields = split_top_level(skip_flag_tokens(after_opcode), ',');
            match fields.get(1) {
                Some(ptr) => vec![operand_from_chunk(ptr.trim())],
                None => vec![TirOperand::Unresolved],
            }
        }
        // `store <flags>* <ty> <v>, <ptrty> <p>[, align N]` — value then pointer.
        "store" => {
            let fields = split_top_level(skip_flag_tokens(after_opcode), ',');
            match (fields.first(), fields.get(1)) {
                (Some(v), Some(p)) => {
                    vec![operand_from_chunk(v.trim()), operand_from_chunk(p.trim())]
                }
                _ => vec![TirOperand::Unresolved],
            }
        }
        // `phi <ty> [ <v0>, <l0> ], [ <v1>, <l1> ]` — each incoming value shares `<ty>`.
        "phi" => resolve_phi_operands(after_opcode),
        // Vector element ops — every comma field is an independently typed `<ty> <val>`:
        //   `extractelement <ty> <vec>, <idxty> <idx>`        (2 operands)
        //   `insertelement  <ty> <vec>, <elty> <elt>, <idxty> <idx>` (3 operands)
        //   `shufflevector  <ty> <v1>, <ty> <v2>, <maskty> <mask>`   (2 vectors + the mask constant;
        //     `split_top_level` keeps the mask's inner `<...>` commas grouped at depth>0)
        "extractelement" | "insertelement" | "shufflevector" => split_top_level(after_opcode, ',')
            .iter()
            .map(|c| operand_from_chunk(c.trim()))
            .collect(),
        // `extractvalue <aggty> <agg>, <idx>...` — only the aggregate is a value operand; the trailing
        // indices are bare integer literals (no type prefix), not typed SSA values.
        "extractvalue" => match split_top_level(after_opcode, ',').first() {
            Some(agg) => vec![operand_from_chunk(agg.trim())],
            None => vec![TirOperand::Unresolved],
        },
        // `insertvalue <aggty> <agg>, <elty> <elt>, <idx>...` — aggregate + inserted element are the
        // value operands; trailing indices are bare literals.
        "insertvalue" => {
            let chunks = split_top_level(after_opcode, ',');
            match (chunks.first(), chunks.get(1)) {
                (Some(agg), Some(elt)) => {
                    vec![
                        operand_from_chunk(agg.trim()),
                        operand_from_chunk(elt.trim()),
                    ]
                }
                _ => vec![TirOperand::Unresolved],
            }
        }
        // `getelementptr [inbounds] <sourcety>, <ptrty> <ptr>, <idxty> <idx>...` — the first comma
        // field is the source element type (a *type*, not an operand); the rest are the base pointer
        // followed by the index values, each an independently typed `<ty> <val>`.
        "getelementptr" => {
            let chunks = split_top_level(after_opcode, ',');
            if chunks.len() < 2 {
                return vec![TirOperand::Unresolved];
            }
            chunks[1..]
                .iter()
                .map(|c| operand_from_chunk(c.trim()))
                .collect()
        }
        // `call <ret> @callee(<argty> <arg>, ...)` — the arguments are the value operands. An indirect
        // call (`%fnptr(...)`) is left Unresolved (the emitter rejects it as unsupported).
        "call" => resolve_call_operands(after_opcode),
        // `[must|no]tail call <ret> @callee(<args>)` — drop the `call` keyword, then resolve as a call.
        "tail" | "musttail" | "notail" => match after_opcode.strip_prefix("call ") {
            Some(rest) => resolve_call_operands(rest.trim_start()),
            None => vec![TirOperand::Unresolved],
        },
        // `alloca <ty>[, <cntty> <cnt>][, align N]` — the allocated type and `align` are not operands; a
        // dynamic element count (the 2nd comma field, when it is a typed value rather than `align`) is.
        "alloca" => match split_top_level(after_opcode, ',').get(1) {
            Some(field) if !field.trim_start().starts_with("align") => {
                vec![operand_from_chunk(field.trim())]
            }
            _ => Vec::new(),
        },
        // Opcodes whose operand layout is not yet lowered.
        _ => {
            if after_opcode.is_empty() {
                Vec::new()
            } else {
                vec![TirOperand::Unresolved]
            }
        }
    }
}

/// The value operands of a direct call: each argument in `... @callee(<argty> <arg>, ...)`, parsed as
/// an independently typed `<ty> <val>` (param attributes like `nonnull`/`signext` are tolerated by
/// `parse_typed_value`, exactly as `parse_call` does). An indirect call (no `@callee`) yields a single
/// `Unresolved` marker — the emitter rejects it as unsupported, so its operands are never consumed.
/// `after_call` is the text after the `call` keyword (`<ret> @callee(<args>)`).
pub(in crate::native) fn resolve_call_operands(after_call: &str) -> Vec<TirOperand> {
    let Some(at) = after_call.find('@') else {
        return vec![TirOperand::Unresolved];
    };
    let Some(open) = after_call[at..].find('(').map(|p| p + at) else {
        return vec![TirOperand::Unresolved];
    };
    let Some(close) = matching_paren(after_call, open) else {
        return vec![TirOperand::Unresolved];
    };
    let args_text = after_call[open + 1..close].trim();
    if args_text.is_empty() {
        return Vec::new();
    }
    split_top_level(args_text, ',')
        .iter()
        .map(|c| operand_from_chunk(c.trim()))
        .collect()
}

/// Drop leading whitespace-separated `OPERAND_FLAG_TOKENS` from `s`, returning the remainder (which
/// then starts at the first typed operand).
pub(in crate::native) fn skip_flag_tokens(s: &str) -> &str {
    let mut s = s.trim_start();
    loop {
        let Some((head, rest)) = s.split_once(char::is_whitespace) else {
            return s;
        };
        if OPERAND_FLAG_TOKENS.contains(&head) {
            s = rest.trim_start();
        } else {
            return s;
        }
    }
}

/// `<ty> <a>, <b>` where `<b>` shares `<a>`'s declared type (the LLVM binary-operator form).
pub(in crate::native) fn two_operands_shared_type(region: &str) -> Vec<TirOperand> {
    let chunks = split_top_level(region, ',');
    if chunks.len() != 2 {
        return vec![TirOperand::Unresolved];
    }
    let first = chunks[0].trim();
    let Ok(tv) = parse_typed_value(first) else {
        return vec![TirOperand::Unresolved, TirOperand::Unresolved];
    };
    let ty = tv.ty.clone();
    vec![
        operand_from_typed_value(&tv),
        operand_from_bare(chunks[1].trim(), ty),
    ]
}

/// `phi <ty> [ <v>, <pred> ], ...` — one `Value`/`Const` operand per incoming, all sharing `<ty>`.
pub(in crate::native) fn resolve_phi_operands(after_opcode: &str) -> Vec<TirOperand> {
    let Some(open) = after_opcode.find('[') else {
        return vec![TirOperand::Unresolved];
    };
    let Ok(ty) = parse_type(after_opcode[..open].trim()) else {
        return vec![TirOperand::Unresolved];
    };
    split_top_level(&after_opcode[open..], ',')
        .iter()
        .filter_map(|pair| {
            let inner = pair.trim().trim_start_matches('[').trim_end_matches(']');
            split_top_level(inner, ',')
                .first()
                .map(|val| operand_from_bare(val.trim(), ty.clone()))
        })
        .collect()
}

/// Parse a `<ty> <val>` chunk into a typed operand, or `Unresolved` if it does not parse.
pub(in crate::native) fn operand_from_chunk(chunk: &str) -> TirOperand {
    match parse_typed_value(chunk) {
        Ok(tv) => operand_from_typed_value(&tv),
        Err(_) => TirOperand::Unresolved,
    }
}

/// A bare operand token (no inline type) carrying a type derived from a sibling operand (the LLVM
/// binary-operator form, where the second operand omits its type). An SSA `%name` becomes `Value`; a
/// literal token is parsed to its `LlValue` (`Unresolved` if it does not parse).
pub(in crate::native) fn operand_from_bare(token: &str, ty: LlType) -> TirOperand {
    if token.starts_with('%') {
        return TirOperand::Value {
            name: token.to_string(),
            ty,
        };
    }
    // A phi incoming value may carry an inline type prefix (e.g.
    // `<4 x float> <float 0x3F800000, …>`), so try parse_typed_value first —
    // which splits the type from the value — and fall back to parse_value
    // for bare constants like `zeroinitializer` or `0x3F800000`.
    if let Ok(tv) = parse_typed_value(token) {
        return operand_from_typed_value(&tv);
    }
    match parse_value(token) {
        Ok(value) => TirOperand::Const { value, ty },
        Err(_) => TirOperand::Unresolved,
    }
}

/// Map a parsed `TypedValue` to a `TirOperand`: a local `%name` becomes `Value`, anything else `Const`.
pub(in crate::native) fn operand_from_typed_value(tv: &TypedValue) -> TirOperand {
    match &tv.value {
        LlValue::Local(name) => TirOperand::Value {
            name: name.clone(),
            ty: tv.ty.clone(),
        },
        value => TirOperand::Const {
            value: value.clone(),
            ty: tv.ty.clone(),
        },
    }
}

/// The SSA value operands an instruction reads (def/use edges): the `%name` tokens it references,
/// excluding its own result and — for `phi` — the predecessor *labels* (control-flow edges, not value
/// uses). A `%name` here is a value reference; block labels never appear in a non-phi instruction's
/// operand list (terminators are parsed separately), so a generic `%name` scan is correct for them.
pub(in crate::native) fn instruction_uses(line: &str, result: Option<&str>) -> Vec<String> {
    let rhs = line.split_once('=').map(|(_, r)| r.trim()).unwrap_or(line);
    let mut uses = Vec::new();
    if rhs.starts_with("phi ") {
        // `phi <ty> [ <val>, %pred ], ...` — keep each incoming's VALUE (first field), drop the label.
        if let Some(open) = rhs.find('[') {
            for pair in split_top_level(&rhs[open..], ',') {
                let inner = pair.trim().trim_start_matches('[').trim_end_matches(']');
                if let Some(val) = split_top_level(inner, ',').first() {
                    collect_value_names(val, &mut uses);
                }
            }
        }
        return dedup_keep_order(uses, result);
    }
    collect_value_names(rhs, &mut uses);
    dedup_keep_order(uses, result)
}

/// Push every `%name` value token in `s` (an identifier starting `%`, alnum/`_`/`.`) into `out`.
pub(in crate::native) fn collect_value_names(s: &str, out: &mut Vec<String>) {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let start = i;
            i += 1;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'.')
            {
                i += 1;
            }
            if i > start + 1 {
                out.push(s[start..i].to_string());
            }
        } else {
            i += 1;
        }
    }
}

pub(in crate::native) fn dedup_keep_order(names: Vec<String>, result: Option<&str>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    names
        .into_iter()
        .filter(|n| Some(n.as_str()) != result && seen.insert(n.clone()))
        .collect()
}

/// The `%r` of an `%r = ...` line, if any.
pub(in crate::native) fn result_name(line: &str) -> Option<String> {
    let (lhs, _) = line.split_once('=')?;
    let lhs = lhs.trim();
    if lhs.starts_with('%') {
        Some(lhs.to_string())
    } else {
        None
    }
}

/// Resolve `(result_name, result_type)` for the instruction forms whose result type is syntactically
/// local. Returns `None` for non-defining lines and for defining forms whose type is not yet
/// inferable here (e.g. `getelementptr`), which the caller records with `result_ty = None`.
pub(in crate::native) fn resolve_result(
    line: &str,
    named_types: &HashMap<String, LlType>,
) -> Option<(String, Option<LlType>)> {
    let name = result_name(line)?;
    let (_, rhs) = line.split_once('=')?;
    let rhs = rhs.trim();
    let opcode = rhs.split_whitespace().next()?;

    let ty = match opcode {
        // `%r = <binop> [flags] <ty> <a>, <b>` — result type is the first type token after the op.
        "add" | "sub" | "mul" | "and" | "or" | "xor" | "shl" | "lshr" | "ashr" | "udiv"
        | "sdiv" | "urem" | "srem" | "fadd" | "fsub" | "fmul" | "fdiv" | "frem" => {
            first_type_after_op(rhs)
        }
        // `%r = load <ty>, ptr ...` — result type is the loaded type.
        "load" => {
            let after = rhs.strip_prefix("load ")?;
            let ty_text = split_top_level(after, ',').into_iter().next()?;
            parse_type(strip_keywords(ty_text).trim()).ok()
        }
        // `%r = <conv> <srcty> <v> to <dstty>` — result type is the destination type.
        "bitcast" | "trunc" | "zext" | "sext" | "fptrunc" | "fpext" | "sitofp" | "uitofp"
        | "fptosi" | "fptoui" | "ptrtoint" | "inttoptr" | "addrspacecast" => {
            let (_, dst) = rhs.rsplit_once(" to ")?;
            parse_type(dst.trim()).ok()
        }
        // `%r = icmp <pred> <ty> ...` / `fcmp` — bool, vectorized over a vector operand. Scan from
        // right after the opcode: `first_type_after` skips the predicate + any flags to the type.
        "icmp" | "fcmp" => match first_type_after(rhs, 1) {
            Some(LlType::Vector(_, n)) => Some(LlType::Vector(Box::new(LlType::Bool), n)),
            Some(_) => Some(LlType::Bool),
            None => None,
        },
        // `%r = select <cond>, <ty> <a>, <ty> <b>` — result type is the arm type.
        "select" => {
            let parts = split_top_level(rhs.strip_prefix("select ")?, ',');
            parts.get(1).and_then(|arm| first_type_token(arm.trim()))
        }
        // `%r = phi <ty> [ ... ]` — result type is the phi type.
        "phi" => first_type_after_op(rhs),
        // `%r = [tail] call [flags] <retty> <callee>(<args>)` — the return type is the first type, and
        // it precedes the callee whether that callee is `@name` (direct) or `%reg` (an indirect call),
        // so scanning from after `call ` resolves both.
        "call" | "tail" => {
            let after = rhs.strip_prefix("tail ").unwrap_or(rhs);
            let after = after.strip_prefix("call ")?;
            first_type_after(after, 0)
        }
        // `%r = extractelement <n x ty> %v, ...` — element type.
        "extractelement" => match first_type_after_op(rhs) {
            Some(LlType::Vector(elem, _)) => Some(*elem),
            other => other,
        },
        // `%r = extractvalue <aggty> %v, <i0>, <i1>...` — walk the constant index path into the
        // aggregate (struct member / array element). A `Named` struct can't be indexed without its
        // definition, so it resolves to `None` (a later increment with the type table).
        "extractvalue" => {
            let parts = split_top_level(rhs, ',');
            let agg = first_type_after_op(rhs)?;
            // `extractvalue` indices are always constant integers (struct member / array element).
            let indices: Vec<Option<usize>> = parts[1..]
                .iter()
                .filter_map(|p| p.split_whitespace().last().map(|t| t.parse().ok()))
                .collect();
            extract_aggregate_member(agg, &indices, named_types)
        }
        // `%r = insertelement <ty> ...` / `insertvalue <aggty> ...` — the (aggregate/vector) type.
        "insertelement" | "insertvalue" => first_type_after_op(rhs),
        // `%r = fneg [fast] <ty> %v` / `freeze <ty> %v` — the operand type.
        "fneg" | "freeze" => first_type_after_op(rhs),
        // `%r = alloca <ty> [, align N] [, addrspace(N)]` — a pointer in the alloca's address space
        // (default 0). `LlType::Ptr` is addrspace-only, so fully resolvable.
        "alloca" => Some(LlType::Ptr(operand_addrspace(rhs))),
        // `%r = shufflevector <n x ty> %a, <n x ty> %b, <m x i32> %mask` — result is a vector of the
        // input element type with the mask's length.
        "shufflevector" => {
            let parts = split_top_level(rhs, ',');
            let elem = match first_type_after_op(rhs) {
                Some(LlType::Vector(e, _)) => *e,
                _ => return Some((name, None)),
            };
            let mask_len = parts.last().and_then(|m| match first_type_token(m.trim()) {
                Some(LlType::Vector(_, n)) => Some(n),
                _ => None,
            });
            mask_len.map(|n| LlType::Vector(Box::new(elem), n))
        }
        // `%r = getelementptr [inbounds] <srcty>, ptr addrspace(N) %base, <indices>` — the result is
        // a pointer in the base's address space. `LlType::Ptr` is addrspace-only (the pointee is
        // tracked separately by inference), so a GEP's result type IS fully resolvable here: it is the
        // base pointer operand's address space.
        "getelementptr" => {
            let parts = split_top_level(rhs, ',');
            parts
                .get(1)
                .map(|base| LlType::Ptr(operand_addrspace(base)))
        }
        // Anything else: defining, but type not resolved in this increment.
        _ => None,
    };
    Some((name, ty))
}

/// The first parseable type token starting right after the opcode (token index 1).
pub(in crate::native) fn first_type_after_op(rhs: &str) -> Option<LlType> {
    first_type_after(rhs, 1)
}

/// The first parseable type at or after whitespace-token index `start`, SCANNING forward token by
/// token (a type may be multi-token like `<4 x float>`). Scanning — rather than requiring the type at
/// a fixed offset — tolerates the variable run of leading non-type tokens: fast-math flags, and the
/// `fcmp`/`icmp` predicate (`olt`/`eq`/...) that sits between the opcode and the operand type.
pub(in crate::native) fn first_type_after(rhs: &str, start: usize) -> Option<LlType> {
    let toks: Vec<&str> = rhs.split_whitespace().collect();
    for begin in start..toks.len() {
        for end in (begin + 1)..=toks.len() {
            let candidate = toks[begin..end].join(" ");
            let candidate = candidate.trim_end_matches(',');
            if let Ok(ty) = parse_type(candidate) {
                return Some(ty);
            }
        }
    }
    None
}

/// The first parseable type token at the start of `s` (for a select arm `<ty> <v>`).
pub(in crate::native) fn first_type_token(s: &str) -> Option<LlType> {
    let toks: Vec<&str> = s.split_whitespace().collect();
    for end in 1..=toks.len() {
        let candidate = toks[..end].join(" ");
        if let Ok(ty) = parse_type(candidate.trim_end_matches(',')) {
            return Some(ty);
        }
    }
    None
}

/// Walk an index path into an aggregate type (the `extractvalue`/`insertvalue` result type, or a
/// `getelementptr` pointee). Each index is `Some(n)` for a constant or `None` for a dynamic
/// (non-constant) index. A struct member step REQUIRES a constant (`None` → give up); array/vector
/// element steps are index-value-independent, so a dynamic index still resolves the element type.
/// `None` overall if a step indexes a non-aggregate, an out-of-range struct member, or a struct with a
/// dynamic index (or an opaque `Named` struct absent from `named_types`).
pub(in crate::native) fn extract_aggregate_member(
    mut ty: LlType,
    indices: &[Option<usize>],
    named_types: &HashMap<String, LlType>,
) -> Option<LlType> {
    for &idx in indices {
        // Resolve a named struct to its definition before indexing.
        if let LlType::Named(name) = &ty {
            ty = named_types.get(name).cloned()?;
        }
        ty = match ty {
            LlType::Struct(members) => members.into_iter().nth(idx?)?,
            LlType::Array(elem, _) => *elem,
            LlType::Vector(elem, _) => *elem,
            _ => return None,
        };
    }
    Some(ty)
}

/// The address space of a pointer operand text like `ptr addrspace(1) %base` or `ptr %base` (-> 0).
pub(in crate::native) fn operand_addrspace(operand: &str) -> u32 {
    operand
        .find("addrspace(")
        .and_then(|p| {
            let after = &operand[p + "addrspace(".len()..];
            after.find(')').and_then(|e| after[..e].trim().parse().ok())
        })
        .unwrap_or(0)
}

pub(in crate::native) fn is_flag_keyword(tok: &str) -> bool {
    matches!(
        tok,
        "nuw"
            | "nsw"
            | "exact"
            | "fast"
            | "nnan"
            | "ninf"
            | "nsz"
            | "arcp"
            | "contract"
            | "afn"
            | "reassoc"
            | "volatile"
    )
}

/// Strip leading LLVM keyword/attribute tokens that precede a type (`fast`, `volatile`, alignment
/// noise) so `parse_type` sees the type cleanly.
pub(in crate::native) fn strip_keywords(s: &str) -> String {
    s.split_whitespace()
        .filter(|t| !is_flag_keyword(t))
        .collect::<Vec<_>>()
        .join(" ")
}
