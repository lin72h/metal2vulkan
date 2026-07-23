//! Pre-submission loop-budget instrumentation for the Metal oracle.
//!
//! A committed Metal command buffer cannot be cancelled — once a compute kernel enters an
//! unbounded loop it pins the GPU until the machine is rebooted. No choice of input data can
//! prevent this (halting problem), so the only structural guarantee is to bound GPU-side work
//! **before** `command_buffer.commit()`.
//!
//! This module takes the Apple-dialect textual LLVM IR that `metal-objdump --disassemble-all`
//! emits (and `metal-as` re-assembles) and classifies each corpus case:
//!
//! - [`GuardPlan::LoopFree`] — no loops anywhere; provably bounded; run unchanged.
//! - [`GuardPlan::Instrumented`] — inject a per-thread back-edge budget so every loop is forced to
//!   exit after [`LOOP_BUDGET_BACKEDGES`] iterations; the module is byte-for-byte equivalent for
//!   any run that stays under budget (the guard only touches private budget memory and a
//!   never-taken branch), so provably-bounded goldens are unchanged.
//! - [`GuardPlan::Quarantine`] — anything we cannot instrument *and verify* (switch back-edges,
//!   unparseable control flow, loop-calls-loopy-callee composition). The caller must not dispatch.
//!
//! The transform is intentionally conservative: when in doubt it quarantines. The caller also
//! round-trips the instrumented text through `metal-as`/`metallib`; if that rejects it, the case
//! is quarantined too. So a bug in this pass can never wedge the GPU — worst case it over-skips.
//!
//! Everything here is a pure string transform with no Metal/objc2 dependency, so it is unit-tested
//! on any OS.

use std::collections::{HashMap, HashSet};

/// Per-thread, per-function back-edge budget. Every loop iteration traverses a back-edge, which
/// decrements this counter; when it reaches zero control jumps to the function's exit. Under the
/// harness's bounded seeds (dims ≤ 16, buffers ≤ 4 KB) legitimate trip counts are ≤ ~10⁶, so
/// ~16.7M gives ample headroom while a genuine runaway (≥10⁸) is caught in tens of ms of GPU time.
pub const LOOP_BUDGET_BACKEDGES: i32 = 1 << 24;

/// Result of classifying one module (entry + all defined functions).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardPlan {
    /// No loops in any defined function — safe to run as-is.
    LoopFree,
    /// Instrumented module text, ready for `metal-as`.
    Instrumented(String),
    /// Cannot instrument safely; the reason (for the ledger). Do not dispatch.
    Quarantine(String),
}

/// Classify a module and, if it contains loops, return an instrumented copy that cannot run the
/// GPU unbounded. `entry` is accepted for symmetry with the caller / future entry-specific rules;
/// classification covers every defined function because linked/visible callees also execute.
pub fn classify_and_instrument(module_text: &str, entry: &str) -> GuardPlan {
    let _ = entry;
    let lines: Vec<&str> = module_text.lines().collect();
    let funcs = find_functions(&lines);
    if funcs.is_empty() {
        return GuardPlan::LoopFree;
    }

    let opaque_ptr = uses_opaque_pointers(module_text);

    // Parse every function. Any unparseable control flow → quarantine (we cannot prove it halts).
    let mut parsed: Vec<ParsedFunc> = Vec::with_capacity(funcs.len());
    for f in &funcs {
        match parse_func(&lines, f) {
            Ok(pf) => parsed.push(pf),
            Err(reason) => return GuardPlan::Quarantine(reason),
        }
    }

    // Transitive "contains a loop" over the static call graph (Metal has no recursion → DAG).
    let direct_loopy: HashMap<&str, bool> = parsed
        .iter()
        .map(|f| (f.name.as_str(), !f.back_edges.is_empty()))
        .collect();
    let trans_loopy = transitive_loopy(&parsed, &direct_loopy);

    // Compose gate: a loop that calls a transitively-loopy function is bounded only by
    // CAP_caller × CAP_callee. With a per-function budget that product can be huge, so refuse it.
    for f in &parsed {
        if f.back_edges.is_empty() {
            continue;
        }
        for callee in &f.calls {
            if *trans_loopy.get(callee.as_str()).unwrap_or(&false) {
                return GuardPlan::Quarantine(format!(
                    "loop in {:?} calls loopy callee {:?} (unbounded composition)",
                    f.name, callee
                ));
            }
        }
    }

    if parsed.iter().all(|f| f.back_edges.is_empty()) {
        return GuardPlan::LoopFree;
    }

    // Instrument: transform loopy functions, copy the rest verbatim, stitch the module back.
    let mut out: Vec<String> = Vec::with_capacity(lines.len() + 64);
    let mut cursor = 0usize;
    for (f, pf) in funcs.iter().zip(parsed.iter()) {
        // Emit untouched lines before this function.
        while cursor < f.define_idx {
            out.push(lines[cursor].to_string());
            cursor += 1;
        }
        if pf.back_edges.is_empty() {
            for line in &lines[f.define_idx..=f.close_idx] {
                out.push(line.to_string());
            }
        } else {
            match transform_func(&lines, f, pf, opaque_ptr) {
                Ok(func_lines) => out.extend(func_lines),
                Err(reason) => return GuardPlan::Quarantine(reason),
            }
        }
        cursor = f.close_idx + 1;
    }
    while cursor < lines.len() {
        out.push(lines[cursor].to_string());
        cursor += 1;
    }

    let mut text = out.join("\n");
    // Preserve a trailing newline if the input had one (metal-as is not picky, but keep it clean).
    if module_text.ends_with('\n') {
        text.push('\n');
    }
    GuardPlan::Instrumented(text)
}

// --- module structure ----------------------------------------------------------------------------

struct FuncSpan {
    define_idx: usize,
    close_idx: usize,
    name: String,
    ret_ty: String,
}

/// Locate every `define … { … }` (bodies only; `declare`s have no body). LLVM emits `{` at the end
/// of the define line and a lone `}` at column 0.
fn find_functions(lines: &[&str]) -> Vec<FuncSpan> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].starts_with("define ") {
            let mut j = i + 1;
            let mut close = None;
            while j < lines.len() {
                if lines[j] == "}" || lines[j].trim() == "}" {
                    close = Some(j);
                    break;
                }
                if lines[j].starts_with("define ") {
                    break; // malformed; stop to avoid swallowing the next function
                }
                j += 1;
            }
            if let (Some(close), Some((name, ret_ty))) = (close, parse_define(lines[i])) {
                out.push(FuncSpan {
                    define_idx: i,
                    close_idx: close,
                    name,
                    ret_ty,
                });
                i = close + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Extract `(function name, return type)` from a `define` line. Return type is what remains after
/// stripping leading attribute keywords and before `@name`.
fn parse_define(line: &str) -> Option<(String, String)> {
    let rest = line.strip_prefix("define ")?;
    let at = rest.find('@')?;
    let name = symbol_after_at(&rest[at..])?;
    let ret_seg = rest[..at].trim();
    let ret_ty = strip_leading_attrs(ret_seg);
    if ret_ty.is_empty() {
        return None;
    }
    Some((name, ret_ty.to_string()))
}

/// Return-position attribute keywords that can precede the return type on a `define` line.
const RET_ATTRS: &[&str] = &[
    "private",
    "internal",
    "available_externally",
    "linkonce",
    "linkonce_odr",
    "weak",
    "weak_odr",
    "common",
    "appending",
    "extern_weak",
    "external",
    "default",
    "hidden",
    "protected",
    "dllimport",
    "dllexport",
    "dso_local",
    "dso_preemptable",
    "local_unnamed_addr",
    "unnamed_addr",
    "spir_func",
    "spir_kernel",
    "fastcc",
    "coldcc",
    "signext",
    "zeroext",
    "noundef",
    "nonnull",
    "inreg",
];

fn strip_leading_attrs(seg: &str) -> &str {
    let mut s = seg.trim_start();
    loop {
        let tok = s.split_whitespace().next().unwrap_or("");
        if tok.is_empty() {
            return s;
        }
        // Calling conventions like `cc10`.
        let is_cc_num = tok.starts_with("cc") && tok[2..].chars().all(|c| c.is_ascii_digit());
        if RET_ATTRS.contains(&tok) || (is_cc_num && tok.len() > 2) {
            s = s[tok.len()..].trim_start();
        } else {
            return s;
        }
    }
}

fn symbol_after_at(s: &str) -> Option<String> {
    let rest = s.strip_prefix('@')?;
    if let Some(rest) = rest.strip_prefix('"') {
        let mut out = String::from("\"");
        let mut escaped = false;
        for ch in rest.chars() {
            out.push(ch);
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => return Some(out),
                _ => {}
            }
        }
        return None;
    }
    let name: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.' || *c == '$')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// Metal's textual IR uses opaque `ptr` in modern toolchains; detect it so injected loads/stores
/// use the matching pointer syntax (`ptr %p` vs `i32* %p`).
fn uses_opaque_pointers(module_text: &str) -> bool {
    module_text.contains("ptr addrspace(")
        || module_text.contains("(ptr ")
        || module_text.contains(", ptr ")
        || module_text.contains(" ptr,")
        || module_text.contains(" ptr)")
        || module_text.contains(" ptr ")
}

// --- per-function parse --------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum TermKind {
    Ret,
    Unreachable,
    Br,
    Switch,
}

struct Block {
    label: Option<String>,
    /// Body-relative inclusive line span (includes the label line, if any).
    start: usize,
    end: usize,
    /// Body-relative index of the terminator's first line.
    term_idx: usize,
    kind: TermKind,
    succ: Vec<String>,
}

struct ParsedFunc {
    name: String,
    ret_ty: String,
    blocks: Vec<Block>,
    /// (source block index, target block index) back-edges, over a DFS of *all* blocks.
    back_edges: Vec<(usize, usize)>,
    /// Directly-called function symbols (for the compose gate).
    calls: Vec<String>,
}

fn parse_func(lines: &[&str], f: &FuncSpan) -> Result<ParsedFunc, String> {
    let body = &lines[f.define_idx + 1..f.close_idx];
    let blocks = parse_blocks(body, &f.name)?;

    let mut label_to_idx: HashMap<&str, usize> = HashMap::new();
    for (i, b) in blocks.iter().enumerate() {
        if let Some(label) = &b.label {
            if label_to_idx.insert(label.as_str(), i).is_some() {
                return Err(format!("{:?}: duplicate block label {label:?}", f.name));
            }
        }
    }

    // Successor indices; a target that names no known block is malformed control flow.
    let mut succ_idx: Vec<Vec<usize>> = Vec::with_capacity(blocks.len());
    for b in &blocks {
        let mut v = Vec::with_capacity(b.succ.len());
        for s in &b.succ {
            match label_to_idx.get(s.as_str()) {
                Some(&idx) => v.push(idx),
                None => return Err(format!("{:?}: branch to unknown block {s:?}", f.name)),
            }
        }
        succ_idx.push(v);
    }

    let back_edges = back_edges(&succ_idx);

    let mut calls = Vec::new();
    for line in body {
        collect_calls(line, &mut calls);
    }

    Ok(ParsedFunc {
        name: f.name.clone(),
        ret_ty: f.ret_ty.clone(),
        blocks,
        back_edges,
        calls,
    })
}

fn parse_blocks(body: &[&str], fname: &str) -> Result<Vec<Block>, String> {
    let mut blocks: Vec<Block> = Vec::new();
    let mut cur_label: Option<String> = None;
    let mut cur_start = 0usize;
    let mut i = 0usize;
    let mut pending = false; // have we seen any line for the current block?

    while i < body.len() {
        let line = body[i];
        if let Some(label) = parse_label(line) {
            if pending {
                blocks.push(finish_block(
                    body,
                    cur_label.take(),
                    cur_start,
                    i - 1,
                    fname,
                )?);
            }
            cur_label = Some(label);
            cur_start = i;
            pending = true;
            i += 1;
            continue;
        }
        pending = true;
        i += 1;
    }
    if pending {
        blocks.push(finish_block(
            body,
            cur_label.take(),
            cur_start,
            body.len() - 1,
            fname,
        )?);
    }
    if blocks.is_empty() {
        return Err(format!("{fname:?}: empty function body"));
    }
    Ok(blocks)
}

/// Build a [`Block`] from its line span, locating and classifying the terminator.
fn finish_block(
    body: &[&str],
    label: Option<String>,
    start: usize,
    end: usize,
    fname: &str,
) -> Result<Block, String> {
    // Find the terminator instruction within [start, end]. A well-formed block has exactly one and
    // it is last; scan forward and keep the last recognised terminator.
    let mut term: Option<(usize, TermKind, Vec<String>)> = None;
    let mut k = start;
    while k <= end {
        let t = body[k].trim_start();
        let kind = if t.starts_with("ret ") || t == "ret" || t == "ret void" {
            Some(TermKind::Ret)
        } else if t == "unreachable" || t.starts_with("unreachable ") {
            Some(TermKind::Unreachable)
        } else if t.starts_with("br ") {
            Some(TermKind::Br)
        } else if t.starts_with("switch ") {
            Some(TermKind::Switch)
        } else if t.starts_with("indirectbr") || t.starts_with("callbr") || t.starts_with("invoke ")
        {
            return Err(format!(
                "{fname:?}: unsupported terminator (indirectbr/callbr/invoke)"
            ));
        } else {
            None
        };
        if let Some(kind) = kind {
            let (succ, last) = match kind {
                TermKind::Ret | TermKind::Unreachable => (Vec::new(), k),
                TermKind::Br => (extract_label_targets(body[k]), k),
                TermKind::Switch => {
                    // A switch spans lines until the closing `]`.
                    let mut last = k;
                    while last <= end && !body[last].contains(']') {
                        last += 1;
                    }
                    if last > end {
                        return Err(format!("{fname:?}: unterminated switch"));
                    }
                    let mut succ = Vec::new();
                    for row in &body[k..=last] {
                        succ.extend(extract_label_targets(row));
                    }
                    (succ, last)
                }
            };
            term = Some((k, kind, succ));
            k = last + 1;
            continue;
        }
        k += 1;
    }
    let (term_idx, kind, succ) =
        term.ok_or_else(|| format!("{fname:?}: block with no terminator"))?;
    Ok(Block {
        label,
        start,
        end,
        term_idx,
        kind,
        succ,
    })
}

/// A block label line: column 0, ends with `:` (before any comment). Handles quoted labels.
fn parse_label(line: &str) -> Option<String> {
    if line.starts_with([' ', '\t', ';']) || line.is_empty() {
        return None;
    }
    if let Some(rest) = line.strip_prefix('"') {
        // "quoted":  → label is `"quoted"`.
        let mut escaped = false;
        for (i, ch) in rest.char_indices() {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => {
                    let after = rest[i + 1..].trim_start();
                    return after
                        .starts_with(':')
                        .then(|| format!("\"{}\"", &rest[..i]));
                }
                _ => {}
            }
        }
        return None;
    }
    let colon = line.find(':')?;
    let name = line[..colon].trim_end();
    if name.is_empty() || name.contains(char::is_whitespace) {
        return None;
    }
    // Guard against `define`/globals slipping through (they never appear inside a body, but be safe).
    if name.starts_with('@') || name.starts_with('!') {
        return None;
    }
    Some(name.to_string())
}

/// All `label %<name>` targets on a line (branch/switch destinations).
fn extract_label_targets(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(pos) = rest.find("label %") {
        let after = &rest[pos + "label %".len()..];
        if let Some(name) = read_value_name(after) {
            out.push(name);
        }
        rest = after;
    }
    out
}

/// Read an SSA/label name starting right after a `%` (quoted or bare).
fn read_value_name(s: &str) -> Option<String> {
    if let Some(rest) = s.strip_prefix('"') {
        let mut escaped = false;
        for (i, ch) in rest.char_indices() {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => return Some(format!("\"{}\"", &rest[..i])),
                _ => {}
            }
        }
        return None;
    }
    let name: String = s
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.' || *c == '$')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// Collect directly-called function symbols from an instruction line.
fn collect_calls(line: &str, out: &mut Vec<String>) {
    let t = line.trim_start();
    if !(t.contains("call ") || t.starts_with("call ")) {
        return;
    }
    // Callee is the `@sym` immediately followed by `(`.
    let mut rest = line;
    while let Some(pos) = rest.find('@') {
        let after = &rest[pos..];
        if let Some(name) = symbol_after_at(after) {
            let tail = &after[1 + raw_sym_len(&after[1..])..];
            if tail.starts_with('(') && !out.iter().any(|s| s == &name) {
                out.push(name);
            }
        }
        rest = &rest[pos + 1..];
    }
}

/// Length of the raw symbol text after `@` (including quotes), to find what follows it.
fn raw_sym_len(s: &str) -> usize {
    if let Some(rest) = s.strip_prefix('"') {
        let mut escaped = false;
        for (i, ch) in rest.char_indices() {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => return i + 2, // opening + closing quote
                _ => {}
            }
        }
        return s.len();
    }
    s.chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.' || *c == '$')
        .count()
}

/// Back-edges over a DFS forest covering *all* blocks (so loops in blocks not reachable from the
/// entry — e.g. a disconnected component the disassembler kept — are still bounded). Any cycle has
/// at least one back-edge in any DFS, so guarding all of them makes every cycle bounded.
fn back_edges(succ: &[Vec<usize>]) -> Vec<(usize, usize)> {
    #[derive(Clone, Copy, PartialEq)]
    enum Color {
        White,
        Grey,
        Black,
    }
    let n = succ.len();
    let mut color = vec![Color::White; n];
    let mut edges = Vec::new();
    // Iterative DFS; stack holds (node, next-successor-cursor).
    for root in 0..n {
        if color[root] != Color::White {
            continue;
        }
        let mut stack: Vec<(usize, usize)> = vec![(root, 0)];
        color[root] = Color::Grey;
        while let Some(&mut (u, ref mut ci)) = stack.last_mut() {
            if *ci < succ[u].len() {
                let v = succ[u][*ci];
                *ci += 1;
                match color[v] {
                    Color::White => {
                        color[v] = Color::Grey;
                        stack.push((v, 0));
                    }
                    Color::Grey => edges.push((u, v)),
                    Color::Black => {}
                }
            } else {
                color[u] = Color::Black;
                stack.pop();
            }
        }
    }
    edges
}

fn transitive_loopy<'a>(
    parsed: &'a [ParsedFunc],
    direct: &HashMap<&'a str, bool>,
) -> HashMap<&'a str, bool> {
    let names: HashSet<&str> = parsed.iter().map(|f| f.name.as_str()).collect();
    let mut loopy: HashMap<&str, bool> = direct.clone();
    // Fixpoint: a function is loopy if it loops directly or calls a loopy (known) function.
    loop {
        let mut changed = false;
        for f in parsed {
            if *loopy.get(f.name.as_str()).unwrap_or(&false) {
                continue;
            }
            if f.calls
                .iter()
                .any(|c| names.contains(c.as_str()) && *loopy.get(c.as_str()).unwrap_or(&false))
            {
                loopy.insert(f.name.as_str(), true);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    loopy
}

// --- instrumentation -----------------------------------------------------------------------------

/// Transform one loopy function: add a per-thread budget in the entry block, split every back-edge
/// through a decrement-and-check guard, and append a single exit block. Returns the whole function
/// (`define … { … }` inclusive) as lines. Errors → the caller quarantines the case.
fn transform_func(
    lines: &[&str],
    f: &FuncSpan,
    pf: &ParsedFunc,
    opaque_ptr: bool,
) -> Result<Vec<String>, String> {
    let body_src = &lines[f.define_idx + 1..f.close_idx];
    let mut body: Vec<String> = body_src.iter().map(|s| s.to_string()).collect();

    let ptr = if opaque_ptr { "ptr" } else { "i32*" };
    let budget = "%m2v.bd";

    // 1) In-place rewrites (before any insertion, so body indices stay valid).
    let mut guards: Vec<String> = Vec::new();
    for (k, &(u, v)) in pf.back_edges.iter().enumerate() {
        let ub = &pf.blocks[u];
        let vb = &pf.blocks[v];
        if ub.kind != TermKind::Br {
            return Err(format!(
                "{:?}: back-edge from non-branch terminator (switch/other) — not split",
                pf.name
            ));
        }
        let v_label = vb
            .label
            .as_deref()
            .ok_or_else(|| format!("{:?}: back-edge into entry block", pf.name))?;
        let guard_label = format!("m2v.g.{k}");

        // Redirect u's back-edge target to the guard (exactly one occurrence expected).
        let term = &body[ub.term_idx];
        let rewritten = replace_branch_target(term, v_label, &guard_label)
            .ok_or_else(|| format!("{:?}: ambiguous back-edge target {v_label:?}", pf.name))?;
        body[ub.term_idx] = rewritten;

        // Rename v's phi predecessor u → guard (u's label, since v's phis name their predecessors).
        if let Some(u_label) = ub.label.as_deref() {
            for line in &mut body[vb.start..=vb.end] {
                if line.contains(" = phi ") || line.trim_start().starts_with("phi ") {
                    *line = rename_phi_pred(line, u_label, &guard_label);
                }
            }
        }

        guards.push(format!("{guard_label}:"));
        guards.push(format!("  %m2v.{k}.a = load i32, {ptr} {budget}, align 4"));
        guards.push(format!("  %m2v.{k}.b = sub i32 %m2v.{k}.a, 1"));
        guards.push(format!("  store i32 %m2v.{k}.b, {ptr} {budget}, align 4"));
        guards.push(format!("  %m2v.{k}.c = icmp sle i32 %m2v.{k}.b, 0"));
        guards.push(format!(
            "  br i1 %m2v.{k}.c, label %m2v.exit, label %{v_label}"
        ));
    }

    // 2) Prepend the budget alloca + init to the entry block (index 0).
    let entry = &pf.blocks[0];
    let insert_at = if entry.label.is_some() {
        entry.start + 1
    } else {
        entry.start
    };
    let init = vec![
        format!("  {budget} = alloca i32, align 4"),
        format!("  store i32 {LOOP_BUDGET_BACKEDGES}, {ptr} {budget}, align 4"),
    ];
    body.splice(insert_at..insert_at, init);

    // 3) Append the guard blocks and the single exit block.
    body.extend(guards);
    body.push("m2v.exit:".to_string());
    if pf.ret_ty == "void" {
        body.push("  ret void".to_string());
    } else {
        // `undef` is a valid value of any type; the exit is only reached on a runaway.
        body.push(format!("  ret {} undef", pf.ret_ty));
    }

    // 4) Reassemble the function.
    let mut out = Vec::with_capacity(body.len() + 2);
    out.push(lines[f.define_idx].to_string());
    out.extend(body);
    out.push(lines[f.close_idx].to_string());
    Ok(out)
}

/// Replace a single `label %<from>` branch target with `label %<to>`. Returns `None` if `<from>`
/// is not present exactly once as a target (ambiguous / degenerate → caller quarantines).
fn replace_branch_target(line: &str, from: &str, to: &str) -> Option<String> {
    let needle = format!("label %{from}");
    let mut matches = Vec::new();
    let mut search_from = 0;
    while let Some(pos) = line[search_from..].find(&needle) {
        let abs = search_from + pos;
        let after = abs + needle.len();
        // Word boundary: the char after the label name must not continue an identifier.
        let boundary = line[after..]
            .chars()
            .next()
            .map(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '$' || c == '"'))
            .unwrap_or(true);
        if boundary {
            matches.push(abs);
        }
        search_from = after;
    }
    if matches.len() != 1 {
        return None;
    }
    let abs = matches[0];
    let mut out = String::with_capacity(line.len() + to.len());
    out.push_str(&line[..abs]);
    out.push_str(&format!("label %{to}"));
    out.push_str(&line[abs + needle.len()..]);
    Some(out)
}

/// Rename a phi's predecessor block `%<from>` → `%<to>`. The predecessor is the token immediately
/// before `]` in each `[ value, %pred ]` incoming, so only predecessor slots are touched.
fn rename_phi_pred(line: &str, from: &str, to: &str) -> String {
    let needle = format!("%{from}");
    let mut out = String::with_capacity(line.len() + to.len());
    let mut rest = line;
    while let Some(pos) = rest.find(&needle) {
        let after = pos + needle.len();
        // Boundary after the name, then (spaces) `]` → this is a predecessor slot.
        let tail = &rest[after..];
        let next = tail.chars().next();
        let is_boundary = next
            .map(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '$' || c == '"'))
            .unwrap_or(true);
        let is_pred = is_boundary && tail.trim_start().starts_with(']');
        out.push_str(&rest[..pos]);
        if is_pred {
            out.push('%');
            out.push_str(to);
        } else {
            out.push_str(&needle);
        }
        rest = &rest[after..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instrumented(text: &str, entry: &str) -> String {
        match classify_and_instrument(text, entry) {
            GuardPlan::Instrumented(s) => s,
            other => panic!("expected Instrumented, got {other:?}"),
        }
    }

    #[test]
    fn loop_free_is_unchanged() {
        let ll = "\
define void @k(ptr addrspace(1) %0) {
  %2 = load i32, ptr addrspace(1) %0, align 4
  ret void
}
";
        assert_eq!(classify_and_instrument(ll, "k"), GuardPlan::LoopFree);
    }

    #[test]
    fn self_loop_is_instrumented_and_bounded() {
        let ll = "\
define void @spin(ptr addrspace(1) %0) {
  br label %1
1:
  br label %1
}
";
        let out = instrumented(ll, "spin");
        assert!(
            out.contains("%m2v.bd = alloca i32, align 4"),
            "budget alloca missing:\n{out}"
        );
        assert!(
            out.contains(&format!("store i32 {LOOP_BUDGET_BACKEDGES}, ptr %m2v.bd")),
            "budget init missing:\n{out}"
        );
        assert!(out.contains("m2v.g.0:"), "guard block missing:\n{out}");
        assert!(
            out.contains("icmp sle i32 %m2v.0.b, 0"),
            "budget check missing:\n{out}"
        );
        assert!(out.contains("m2v.exit:"), "exit block missing:\n{out}");
        assert!(out.contains("ret void"), "void exit missing:\n{out}");
        // The self-branch now targets the guard, and the guard returns to the loop head.
        assert!(
            out.contains("br label %m2v.g.0"),
            "back-edge not redirected:\n{out}"
        );
        assert!(
            out.contains("br i1 %m2v.0.c, label %m2v.exit, label %1"),
            "guard exit branch missing:\n{out}"
        );
    }

    #[test]
    fn phi_predecessor_is_renamed_on_split_edge() {
        // Loop header `%loop` has a phi over {entry, latch}; the latch back-edge must have its phi
        // predecessor rewritten to the guard, while the entry incoming is untouched.
        let ll = "\
define void @counted(ptr addrspace(1) %0) {
  br label %loop
loop:
  %i = phi i32 [ 0, %1 ], [ %next, %latch ]
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, 100
  br i1 %done, label %exit, label %latch
latch:
  br label %loop
exit:
  ret void
}
";
        let out = instrumented(ll, "counted");
        // Entry incoming preserved…
        assert!(
            out.contains("[ 0, %1 ]"),
            "entry phi incoming changed:\n{out}"
        );
        // …latch incoming redirected to the guard block.
        assert!(
            out.contains("[ %next, %m2v.g.0 ]"),
            "latch phi pred not renamed:\n{out}"
        );
        assert!(
            !out.contains("[ %next, %latch ]"),
            "stale latch phi incoming:\n{out}"
        );
        assert!(
            out.contains("br label %m2v.g.0"),
            "latch back-edge not redirected:\n{out}"
        );
    }

    #[test]
    fn nested_loops_guard_every_back_edge() {
        let ll = "\
define void @nested(ptr addrspace(1) %0) {
  br label %outer
outer:
  br label %inner
inner:
  %c = icmp eq i32 0, 0
  br i1 %c, label %inner, label %outer
}
";
        let out = instrumented(ll, "nested");
        assert!(out.contains("m2v.g.0:"), "first guard missing:\n{out}");
        assert!(out.contains("m2v.g.1:"), "second guard missing:\n{out}");
    }

    #[test]
    fn switch_back_edge_is_quarantined() {
        let ll = "\
define void @sw(ptr addrspace(1) %0) {
  br label %head
head:
  switch i32 0, label %exit [
    i32 0, label %head
  ]
exit:
  ret void
}
";
        match classify_and_instrument(ll, "sw") {
            GuardPlan::Quarantine(_) => {}
            other => panic!("expected Quarantine for switch back-edge, got {other:?}"),
        }
    }

    #[test]
    fn loop_calling_loopy_callee_is_quarantined() {
        let ll = "\
define void @helper(ptr addrspace(1) %0) {
  br label %1
1:
  br label %1
}

define void @entry(ptr addrspace(1) %0) {
  br label %loop
loop:
  call void @helper(ptr addrspace(1) %0)
  br label %loop
}
";
        match classify_and_instrument(ll, "entry") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("loopy callee"), "{msg}"),
            other => panic!("expected Quarantine for loop→loopy callee, got {other:?}"),
        }
    }

    #[test]
    fn loop_calling_loop_free_callee_is_instrumented() {
        let ll = "\
define void @leaf(ptr addrspace(1) %0) {
  ret void
}

define void @entry(ptr addrspace(1) %0) {
  br label %loop
loop:
  call void @leaf(ptr addrspace(1) %0)
  br label %loop
}
";
        let out = instrumented(ll, "entry");
        assert!(out.contains("m2v.exit:"), "entry not instrumented:\n{out}");
        // The loop-free leaf is copied verbatim (no budget added).
        assert!(!out.contains("define void @leaf(ptr addrspace(1) %0) {\n  %m2v.bd"));
    }

    #[test]
    fn non_void_exit_returns_undef() {
        let ll = "\
define <4 x float> @frag(ptr addrspace(1) %0) {
  br label %1
1:
  br label %1
}
";
        let out = instrumented(ll, "frag");
        assert!(
            out.contains("ret <4 x float> undef"),
            "typed undef exit missing:\n{out}"
        );
    }

    #[test]
    fn typed_pointer_dialect_uses_typed_loads() {
        // No opaque `ptr` anywhere → injected loads/stores must use `i32*`.
        let ll = "\
define void @spin(i32 addrspace(1)* %0) {
  br label %1
1:
  br label %1
}
";
        let out = instrumented(ll, "spin");
        assert!(
            out.contains("load i32, i32* %m2v.bd"),
            "expected typed load:\n{out}"
        );
        assert!(
            !out.contains("load i32, ptr %m2v.bd"),
            "unexpected opaque load:\n{out}"
        );
    }

    #[test]
    fn mps_like_k_loop_is_instrumented() {
        // The MPS-style GEMM shape whose K-loop (~1.8e9) wedged the GPU: a phi-carried counter loop
        // that must now be budget-bounded.
        let ll = "\
define void @gemm(ptr addrspace(1) %0, ptr addrspace(2) %1) {
  %3 = load i32, ptr addrspace(2) %1, align 4
  br label %loop
loop:
  %i = phi i32 [ 0, %2 ], [ %n, %loop ]
  %n = add nuw i32 %i, 1
  %c = icmp eq i32 %n, %3
  br i1 %c, label %done, label %loop
done:
  ret void
}
";
        let out = instrumented(ll, "gemm");
        assert!(out.contains("m2v.g.0:"), "K-loop guard missing:\n{out}");
        assert!(
            out.contains("[ %n, %m2v.g.0 ]"),
            "K-loop phi pred not renamed:\n{out}"
        );
    }
}
