//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;
use crate::native::tir::RetTerm;

/// Synthesized unified-return exit-block name for [`unify_returns`] (`%metal2vulkan.uret*`). A
/// same-memory structurizer name that never crosses the emit->SPIR-V reparse seam; centralized here so
/// the base label, its collision-avoidance variants, and the merged-return phi value cannot drift apart.
pub(in crate::native) const URET_PREFIX: &str = "%metal2vulkan.uret";

/// Lower every `unreachable` terminator to a `ret` (transform #1.5 of the cross-arm cascade, applied
/// before [`unify_returns`]). An arm ending in `unreachable` never reconverges with its sibling, so
/// the enclosing selection has no natural merge (`cond-no-natural`) even after return unification.
/// Executing `unreachable` is UB, so replacing it with a function return is a legal refinement; the
/// return's type/shape is copied from the function's first real `ret` (`ret void`, or `ret T undef`).
/// Returns `None` when there is no `unreachable` or no `ret` to model the replacement on.
pub(in crate::native) fn lower_unreachable_to_ret(blocks: &[BodyBlock]) -> Option<Vec<BodyBlock>> {
    // Model the replacement `ret` on the function's first real `ret`, read from the typed carrier (the
    // dual of scanning the trailing `ret` LINE). `Unrenderable` — a value `ret` whose type does not
    // render injectively (e.g. an anonymous struct return) — declines the transform, mirroring the line
    // path's `?`-bail on an un-splittable `ret`; a non-`ret` terminator is skipped.
    let mut model: Option<String> = None;
    for b in blocks {
        let Some(t) = &b.typed else { continue };
        match t.ret_term() {
            RetTerm::Void => {
                model = Some("ret void".to_string());
                break;
            }
            RetTerm::Value { ty, .. } => {
                model = Some(format!("ret {ty} undef"));
                break;
            }
            RetTerm::Unrenderable => return None,
            RetTerm::NotRet => {}
        }
    }
    let model = model?;
    let mut hit = false;
    let mut out = blocks.to_vec();
    for b in out.iter_mut() {
        // Detect an `unreachable` terminator from the carrier (the sole substrate).
        let is_unreachable = b.typed.as_ref().is_some_and(|t| {
            matches!(t.terminator, crate::native::tir::TirTerminator::Unreachable)
        });
        if is_unreachable {
            // Rewrite the `unreachable` -> `ret` terminator on the carrier.
            if let Some(t) = &mut b.typed {
                t.set_terminator_line(&model);
            }
            hit = true;
        }
    }
    hit.then_some(out)
}

/// Single-exit transform (transform #2 of the cross-arm cascade): route every `ret` through one
/// synthesized exit block, so a divergent selection (arms that `ret` rather than reconverge) gains a
/// natural merge — the common exit — instead of rejecting `selection:cond-no-natural`. Returns the
/// rewritten blocks, or `None` if there are fewer than two `ret` blocks (nothing to unify) or the
/// returns are not uniformly typed.
///
/// For a void function the exit block is `ret void`; for a value-returning function it is a phi over
/// the returned values (`%uret.v = phi T [v_i, %b_i]...`) followed by `ret T %uret.v`. Each original
/// `ret` block's terminator becomes `br label %exit`. Floor-safe: invoked only behind the
/// `inline_sroa_raw_cfg_restructure` retry, adopted only if `structured_plan` then admits.
pub(in crate::native) fn unify_returns(blocks: &[BodyBlock]) -> Option<Vec<BodyBlock>> {
    unify_return_like_exits(blocks, false)
}

/// Variant of [`unify_returns`] used by reject-only divergent-exit separation. `unreachable` is a
/// return-like exit under LLVM's UB semantics; for a value-returning function its synthesized
/// incoming is explicitly `undef`, while the real returns still provide the modeled result type.
fn unify_returns_and_unreachable(blocks: &[BodyBlock]) -> Option<Vec<BodyBlock>> {
    unify_return_like_exits(blocks, true)
}

#[derive(Clone)]
enum ReturnLike {
    Return(Option<(String, String)>),
    Unreachable,
}

fn unify_return_like_exits(
    blocks: &[BodyBlock],
    include_unreachable: bool,
) -> Option<Vec<BodyBlock>> {
    // Collect the `ret` blocks and their (type, value) read from the typed carrier; `None` value =
    // `ret void`. `Unrenderable` — a value `ret` whose type/value does not render injectively —
    // declines the whole unification (mirrors the line path's `?`-bail on an un-splittable `ret`),
    // never a silent skip that would miscount the returns; a non-`ret` terminator is skipped.
    let mut exits: Vec<(usize, ReturnLike)> = Vec::new();
    for (i, b) in blocks.iter().enumerate() {
        let Some(t) = &b.typed else { continue };
        match t.ret_term() {
            RetTerm::Void => exits.push((i, ReturnLike::Return(None))),
            RetTerm::Value { ty, val } => exits.push((i, ReturnLike::Return(Some((ty, val))))),
            RetTerm::Unrenderable => return None,
            RetTerm::NotRet
                if include_unreachable
                    && matches!(t.terminator, crate::native::tir::TirTerminator::Unreachable) =>
            {
                exits.push((i, ReturnLike::Unreachable));
            }
            RetTerm::NotRet => {}
        }
    }
    if exits.len() < 2 {
        return None;
    }
    let real_returns = exits.iter().filter_map(|(_, exit)| match exit {
        ReturnLike::Return(value) => Some(value),
        ReturnLike::Unreachable => None,
    });
    let first_return = real_returns.clone().next()?;
    // All returns must agree on void-ness and (for value returns) on type.
    let is_void = first_return.is_none();
    let ret_ty = first_return.as_ref().map(|(ty, _)| ty.clone());
    for value in real_returns {
        if value.is_none() != is_void {
            return None;
        }
        if let (Some(want), Some((ty, _))) = (&ret_ty, value) {
            if ty != want {
                return None;
            }
        }
    }

    // A fresh exit label that does not collide with an existing block.
    let names: HashSet<&str> = blocks.iter().map(|b| b.name.as_str()).collect();
    let mut exit = URET_PREFIX.to_string();
    let mut n = 0usize;
    while names.contains(exit.as_str()) {
        exit = format!("{URET_PREFIX}.{n}");
        n += 1;
    }

    let mut out: Vec<BodyBlock> = blocks.to_vec();
    let mut incomings: Vec<String> = Vec::new();
    for (idx, return_like) in &exits {
        match return_like {
            ReturnLike::Return(Some((_, val))) => {
                incomings.push(format!("[ {val}, {} ]", out[*idx].name));
            }
            ReturnLike::Unreachable if !is_void => {
                incomings.push(format!("[ undef, {} ]", out[*idx].name));
            }
            ReturnLike::Return(None) | ReturnLike::Unreachable => {}
        }
        // Rewrite the return-like terminator to a branch to the unified exit (on the carrier, the sole
        // substrate).
        if let Some(t) = &mut out[*idx].typed {
            t.set_unconditional_branch(&exit);
        }
    }

    let exit_lines = if is_void {
        vec!["ret void".to_string()]
    } else {
        let ty = ret_ty.as_ref()?;
        vec![
            format!("{URET_PREFIX}.v = phi {ty} {}", incomings.join(", ")),
            format!("ret {ty} {URET_PREFIX}.v"),
        ]
    };
    let role = role_for_name(&exit);
    // Purely synthetic unified-exit block (`ret void`, or a phi over the returned values + `ret`) — lower
    // its carrier directly. Empty named-types is exact: the phi type parses without resolution and
    // `ret_emit` never needs the module type table.
    let typed = crate::native::tir::lower_block_carrier(&exit, &exit_lines, &HashMap::new());
    out.push(BodyBlock {
        name: exit,
        role,
        typed,
    });
    Some(out)
}

/// Reject-only shared-merge separation for conditionals whose arms leave through different function
/// exits. A conditional with no immediate post-dominator cannot own an `OpSelectionMerge`; route its
/// return-like exits through one synthesized block so the planner can derive their shared continuation.
///
/// `unreachable` is a return-like exit under LLVM's UB semantics. The caller invokes this only after the
/// complete ordinary planner ladder rejects and adopts it only when the unchanged ladder then admits,
/// keeping the construction on the closed structural class “rejected CFG made structurable by one shared
/// function exit.”
pub(in crate::native) fn separate_divergent_selection_exits(
    blocks: &[BodyBlock],
) -> Option<Vec<BodyBlock>> {
    unify_returns_and_unreachable(blocks)
}

/// Fresh, collision-resistant name for a cloned label/value. `orig` includes the leading `%`.
pub(in crate::native) fn fresh(orig: &str, id: usize) -> String {
    let stripped = orig.strip_prefix('%').unwrap_or(orig);
    format!("%xa{id}_{stripped}")
}

/// The SSA value a line defines (`%x = ...`), including the `%`, or `None` for a non-defining line.
#[cfg(test)]
pub(in crate::native) fn line_def(line: &str) -> Option<String> {
    let t = line.trim_start();
    if !t.starts_with('%') {
        return None;
    }
    let eq = t.find('=')?;
    let lhs = t[..eq].trim();
    if lhs.contains(char::is_whitespace) || !lhs.starts_with('%') {
        return None;
    }
    Some(lhs.to_string())
}

/// Rebuild a phi line keeping only the incoming `[ value, %pred ]` entries whose predecessor label
/// satisfies `keep`. Returns `None` if the line is not a parseable phi (left untouched by caller).
#[cfg(test)]
pub(in crate::native) fn rebuild_phi(line: &str, keep: impl Fn(&str) -> bool) -> Option<String> {
    let (head, body) = line.split_once("phi ")?;
    // body = "<ty> [ v, %p ], [ v, %p ], ... <maybe ;comment>"
    let ty_end = body.find('[')?;
    let ty = body[..ty_end].trim_end();
    let rest = &body[ty_end..];
    // Split into bracketed incomings.
    let mut kept: Vec<String> = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, c) in rest.char_indices() {
        match c {
            '[' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            ']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let inc = &rest[start..=i]; // "[ v, %p ]"
                    if let Some(pred) = phi_incoming_pred(inc) {
                        if keep(&pred) {
                            kept.push(inc.trim().to_string());
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if kept.is_empty() {
        return None;
    }
    Some(format!("{head}phi {ty} {}", kept.join(", ")))
}

/// The predecessor label (`%p`) of a single phi incoming `[ value, %p ]`.
#[cfg(test)]
pub(in crate::native) fn phi_incoming_pred(inc: &str) -> Option<String> {
    let inner = inc.trim().strip_prefix('[')?.strip_suffix(']')?;
    let comma = inner.rfind(',')?;
    Some(inner[comma + 1..].trim().to_string())
}

/// Replace every whole `%token` in `line` per `map`, boundary-aware (so `%1` never matches inside
/// `%10`). Labels and SSA values share the `%` namespace, so one map covers both.
pub(in crate::native) fn rename_tokens(line: &str, map: &HashMap<String, String>) -> String {
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let mut j = i + 1;
            while j < bytes.len() && is_ident_byte(bytes[j]) {
                j += 1;
            }
            let token = &line[i..j];
            match map.get(token) {
                Some(repl) => out.push_str(repl),
                None => out.push_str(token),
            }
            i = j;
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

pub(in crate::native) fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'.'
}
