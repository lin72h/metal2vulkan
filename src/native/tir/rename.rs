//! Typed dual of the string `rename_tokens` clone: deep-rename every `%`-token in a `TirBlock` per a
//! rename map, on the source's ALREADY-TYPED carrier — so a clone site can produce the clone's carrier
//! without the module named-type table (the source resolved every type already) and without re-lexing.
//! Byte-identical to re-lowering `rename_tokens(lines)` by construction (verified by byte-baseline drift NONE in historical private gates): the
//! same `%`-tokens move, types (never in the rename map) are untouched, and the diagnostics-only raw-line
//! fields are renamed with the SAME `cfg::rename_tokens` the string clone applies.

use super::*;
use crate::native::ir::{LlGep, LlValue, TypedValue};
use crate::native::parse::{LlCall, LlSwitch};
use std::collections::HashMap;

type Map = HashMap<String, String>;

/// Rename one `%name` token per the map (identity if absent) — labels and SSA values share the `%`
/// namespace, so one map covers both. Non-`%` tokens (constants, `undef`, `@global`) are never in the
/// map, so this is the identity on them, matching the string `rename_tokens`.
fn rn(name: &str, map: &Map) -> String {
    map.get(name).cloned().unwrap_or_else(|| name.to_string())
}

fn rn_in_place(name: &mut String, map: &Map) {
    if let Some(repl) = map.get(name.as_str()) {
        *name = repl.clone();
    }
}

fn rename_llvalue(v: &mut LlValue, map: &Map) {
    match v {
        LlValue::Local(n) => rn_in_place(n, map),
        LlValue::Vector(vs) | LlValue::Array(vs) | LlValue::Struct(vs) => {
            for tv in vs {
                rename_typed_value(tv, map);
            }
        }
        LlValue::Splat(b) => rename_typed_value(b, map),
        LlValue::Gep(g) => rename_gep(g, map),
        LlValue::IntToPtr { source, .. } => rename_typed_value(source, map),
        // Global(@name) is not a `%`-token; scalar constants carry no names.
        LlValue::Global(_)
        | LlValue::Bool(_)
        | LlValue::Int(_)
        | LlValue::SignedInt(_)
        | LlValue::Hex(_)
        | LlValue::Float(_)
        | LlValue::HalfBits(_)
        | LlValue::BFloatBits(_)
        | LlValue::Zero
        | LlValue::Undef => {}
    }
}

/// Clone-and-rename an `LlValue` per the map — the by-value form of the in-place `rename_llvalue`,
/// used by the carrier-direct `mirror_region_incomings` to build a renamed incoming value without
/// mutating the source phi (which is still needed unrenamed for the original incoming).
pub(in crate::native) fn renamed_llvalue(v: &LlValue, map: &Map) -> LlValue {
    let mut out = v.clone();
    rename_llvalue(&mut out, map);
    out
}

/// Rename one `%label` token per the map (identity if absent) — the predecessor half of a mirrored
/// incoming (`region` labels are in the map, external labels are not).
pub(in crate::native) fn renamed_label(name: &str, map: &Map) -> String {
    rn(name, map)
}

fn rename_typed_value(tv: &mut TypedValue, map: &Map) {
    // `ty` never carries a rename-map token (the map holds region block labels + SSA defs, never a
    // type name), so only the value moves — matching the string rewrite over the operand text.
    rename_llvalue(&mut tv.value, map);
}

fn rename_gep(g: &mut LlGep, map: &Map) {
    rename_typed_value(&mut g.base, map);
    for idx in &mut g.indices {
        rename_typed_value(idx, map);
    }
}

fn rename_call(c: &mut LlCall, map: &Map) {
    // `callee` is an `@global`, never a `%`-token; `ret`/`arg_aligns` carry no names.
    for arg in &mut c.args {
        rename_typed_value(arg, map);
    }
}

fn rename_operand(op: &mut TirOperand, map: &Map) {
    match op {
        TirOperand::Value { name, .. } => rn_in_place(name, map),
        TirOperand::Const { value, .. } => rename_llvalue(value, map),
        TirOperand::Unresolved => {}
    }
}

fn rename_terminator(t: &mut TirTerminator, map: &Map) {
    match t {
        TirTerminator::Br(target) => rn_in_place(target, map),
        TirTerminator::BrCond { cond, t, f } => {
            rn_in_place(cond, map);
            rn_in_place(t, map);
            rn_in_place(f, map);
        }
        TirTerminator::Switch {
            selector,
            default,
            cases,
        } => {
            rn_in_place(selector, map);
            rn_in_place(default, map);
            for (_, label) in cases {
                rn_in_place(label, map);
            }
        }
        // The `ret` value token is renamed via `RetEmit` below; the structured `Ret(Option<String>)`
        // string is the same value token and is renamed here for consistency.
        TirTerminator::Ret(Some(v)) => rn_in_place(v, map),
        TirTerminator::Ret(None) | TirTerminator::Unreachable => {}
    }
}

fn rename_ret_emit(r: &mut RetEmit, map: &Map) {
    if let RetEmit::Value(tv) = r {
        rename_typed_value(tv, map);
    }
}

fn rename_switch(sw: &mut LlSwitch, map: &Map) {
    rename_typed_value(&mut sw.selector, map);
    rn_in_place(&mut sw.default_label, map);
    for (value, label) in &mut sw.cases {
        rename_llvalue(value, map);
        rn_in_place(label, map);
    }
}

fn rename_inst(inst: &mut TirInst, map: &Map) {
    if let Some(r) = &mut inst.result {
        rn_in_place(r, map);
    }
    for u in &mut inst.uses {
        rn_in_place(u, map);
    }
    for op in &mut inst.operands {
        rename_operand(op, map);
    }
    if let Some(g) = &mut inst.gep {
        rename_gep(g, map);
    }
    if let Some(c) = &mut inst.call {
        rename_call(c, map);
    }
    if let Some((_, incoming)) = &mut inst.phi_incoming {
        for (value, pred) in incoming {
            rename_llvalue(value, map);
            rn_in_place(pred, map);
        }
    }
    if let Some((tv, _dst)) = inst.bitcast.as_deref_mut() {
        rename_typed_value(tv, map);
    }
    // Parse-time inference views also carry `%`-tokens; rename them so a cloned/renamed carrier stays
    // byte-identical to re-lowering the renamed line (the `== re-lower` invariant). No parse-time
    // inference reads a structurized block, but every carried field must match re-lower.
    if let Some((result, base)) = &mut inst.identity_ptr_bitcast {
        rn_in_place(result, map);
        rn_in_place(base, map);
    }
    if let Some(values) = &mut inst.phi_incoming_values {
        for value in values {
            rename_llvalue(value, map);
        }
    }
    if let Some((true_value, false_value)) = inst.select_arms.as_deref_mut() {
        rename_typed_value(true_value, map);
        rename_typed_value(false_value, map);
    }
    if let Some(load) = &mut inst.load {
        rename_typed_value(&mut load.ptr, map);
    }
    if let Some((object, ptr)) = inst.store.as_deref_mut() {
        rename_typed_value(object, map);
        rename_typed_value(ptr, map);
    }
    if let Some(call) = &mut inst.alias_call {
        rename_call(call, map);
    }
    match inst.emit_scan_call.as_deref_mut() {
        Some(Ok(call)) => rename_call(call, map),
        Some(Err(msg)) => *msg = crate::native::cfg::rename_tokens(msg, map),
        None => {}
    }
    // Diagnostics-only raw-line carriers: rename with the SAME string `rename_tokens` the clone applies
    // to the whole line, so a cloned block's error bytes (and a fresh-re-lower Debug) match the
    // re-lower of the renamed line exactly.
    for s in [
        &mut inst.diag_line,
        &mut inst.void_call_line,
        &mut inst.value_call_error,
        &mut inst.icmp_rest,
    ]
    .into_iter()
    .flatten()
    {
        *s = crate::native::cfg::rename_tokens(s, map);
    }
    if let Some((_, dst)) = inst.bitcast.as_deref_mut() {
        *dst = crate::native::cfg::rename_tokens(dst, map);
    }
    // `result_ty`/`gep_source_ty`/`alloca_ty`/`cmp_predicate`/`mem_align`/`opcode`/`aggregate_indices`/
    // `shuffle_mask` never carry a rename-map token (types/opcodes/constants), so they are untouched.
}

impl TirBlock {
    /// Deep-rename every `%`-token in this block per `map` (labels + SSA values; types untouched). The
    /// typed dual of `rename_tokens(line, map)` applied to every line, used by the clone sites to build a
    /// renamed carrier from the source's carrier. See the module doc.
    pub(in crate::native) fn rename(&mut self, map: &Map) {
        rn_in_place(&mut self.label, map);
        for inst in &mut self.insts {
            rename_inst(inst, map);
        }
        rename_terminator(&mut self.terminator, map);
        rename_ret_emit(&mut self.ret, map);
        if let Some(sw) = &mut self.switch {
            rename_switch(sw, map);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::cfg::rename_tokens;

    fn map(pairs: &[(&str, &str)]) -> Map {
        pairs
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    /// The typed `rename` must equal re-lowering the string-`rename_tokens`ed lines, for a spread of
    /// instruction shapes (results, operands, gep, call, phi, terminators, switch, raw-line carriers).
    #[test]
    fn rename_matches_relowered_lines() {
        let types = HashMap::new();
        let m = map(&[
            ("%a", "%a.c"),
            ("%b", "%b.c"),
            ("%r", "%r.c"),
            ("%p", "%p.c"),
            ("%blk", "%blk.c"),
            ("%pred1", "%pred1.c"),
            ("%pred2", "%pred2.c"),
        ]);
        let cases: &[&[&str]] = &[
            &["%r = add i32 %a, %b", "br label %blk"],
            &["%r = icmp eq i32 %a, %b", "br i1 %r, label %blk, label %a"],
            &["%r = getelementptr i8, ptr %a, i32 %b", "ret ptr %r"],
            &[
                "%r = phi i32 [ %a, %pred1 ], [ %b, %pred2 ]",
                "switch i32 %r, label %blk [ i32 0, label %pred1 ]",
            ],
            &["%r = call i32 @f(i32 %a, i32 %b)", "ret i32 %r"],
            &[
                "%r = extractelement <4 x float> %a, i32 %b",
                "br label %blk",
            ],
            &["%r = bitcast i32 %a to float", "ret void"],
        ];
        for case in cases {
            let lines: Vec<String> = case.iter().map(|s| s.to_string()).collect();
            let mut carrier = lower_block_carrier("%blk", &lines, &types).unwrap();
            carrier.rename(&m);
            let renamed: Vec<String> = lines.iter().map(|l| rename_tokens(l, &m)).collect();
            // The block label `%blk` renames to `%blk.c` too.
            let expected = lower_block_carrier("%blk.c", &renamed, &types).unwrap();
            assert_eq!(
                format!("{carrier:?}"),
                format!("{expected:?}"),
                "typed rename diverged from re-lower for {case:?}"
            );
        }
    }
}
