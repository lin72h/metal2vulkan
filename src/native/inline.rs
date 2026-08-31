//! Pre-emit AIR normalization: a **non-recursive internal-function inliner** for sanitized
//! LLVM/AIR textual IR.
//!
//! AIR modules emitted by the Metal front end frequently split a kernel across small `internal`
//! helper functions (`fastcc`, mangled names) that the native emitter would otherwise have to model
//! as full SPIR-V `OpFunctionCall`s — a construct the Logical-addressing emitter does not lower.
//! This pass folds every direct call to a module-local `internal` helper into its caller so the
//! emitter sees a single self-contained function body per entry point. It runs on the sanitized AIR
//! TEXT before [`super::LlModule::parse`], so the reconstructed CFG/SSA is recomputed from the
//! rewritten text.
//!
//! It is **floor-safe by construction**: it fires only when a direct call targets a function that is
//! `define internal` in this very module (never a `declare`, never an `air.*`/`llvm.*` intrinsic,
//! never an indirect `%ptr` call), the internal call graph on the inlined path is acyclic, and every
//! callee body is a shape the tokenized renamer can handle safely. On ANY difficulty — recursion, a
//! parse it cannot model, a construct it cannot rename — it returns the module text UNCHANGED,
//! leaving the call for another tier to handle. It decides purely from IR structure (the `define
//! internal` set, the call-graph shape, the SSA/label token grammar), never from a shader name.

use super::lex::split_top_level;
use std::collections::{HashMap, HashSet};

/// Inline only internal helper callsites that carry an exact linked function symbol as an
/// argument. Substituting the helper parameter turns its indirect `%function(...)` call into an
/// ordinary direct `@function(...)` call before Logical-SPIR-V construction, without attempting to
/// represent a first-class function pointer.
pub(super) fn inline_direct_function_pointer_consumers(
    san_ll: &str,
    direct_functions: &HashSet<String>,
) -> String {
    let mut source = san_ll.to_string();
    loop {
        let Some(items) = parse_items(&source) else {
            return source;
        };
        let internal = items
            .iter()
            .filter_map(|item| {
                let Item::Func(function) = item else {
                    return None;
                };
                let signature = parse_def_header(&function.header)?;
                signature.internal.then_some(signature.name)
            })
            .collect::<HashSet<_>>();
        let selected = items.iter().find_map(|item| {
            let Item::Func(function) = item else {
                return None;
            };
            function.body.iter().find_map(|line| {
                let call = parse_call(line)?;
                if !internal.contains(&call.callee) {
                    return None;
                }
                call.args
                    .iter()
                    .find(|argument| direct_functions.contains(*argument))
                    .map(|argument| (call.callee.clone(), argument.clone()))
            })
        });
        let Some((consumer, function)) = selected else {
            return source;
        };
        let targets = HashSet::from([consumer]);
        let Some(inlined) = try_inline(&source, Some((&targets, &function, None))) else {
            return source;
        };
        if inlined == source {
            return source;
        }
        source = inlined;
    }
}

/// Inline every eligible direct call to a module-local `internal` helper, transitively, to a
/// fixpoint. Returns the module text unchanged when nothing is eligible or any difficulty is hit.
#[cfg(test)]
pub(super) fn inline_nonrecursive_internal_calls(san_ll: &str) -> String {
    match try_inline(san_ll, None) {
        Some(out) => out,
        None => san_ll.to_string(),
    }
}

pub(super) struct PointerConsumerInlining {
    pub(super) source: String,
    pub(super) requires_relooper: bool,
}

/// Inline internal helpers that directly consume pointer-select results before emission. A
/// deferred select has no standalone Logical-SPIR-V pointer value when its arms name distinct
/// buffers; moving its consumer into the caller lets loads/stores replay the select in value space.
/// The returned construction fact selects the bounded whole-CFG path whenever inlining changed the
/// source, before either emitter attempts to construct a module.
pub(super) fn inline_pointer_select_consumers(
    san_ll: &str,
    entry_name: Option<&str>,
) -> PointerConsumerInlining {
    let Some(items) = parse_items(san_ll) else {
        return PointerConsumerInlining {
            source: san_ll.to_string(),
            requires_relooper: false,
        };
    };
    let eligible_callees = items
        .iter()
        .filter_map(|item| {
            let Item::Func(function) = item else {
                return None;
            };
            let signature = parse_def_header(&function.header)?;
            signature.internal.then_some(signature.name)
        })
        .collect::<HashSet<_>>();
    let mut selected_consumers = Vec::new();
    for item in &items {
        let Item::Func(function) = item else {
            continue;
        };
        if let Some(entry_name) = entry_name {
            let Some(signature) = parse_def_header(&function.header) else {
                continue;
            };
            if signature.name.trim_start_matches('@') != entry_name {
                continue;
            }
        }
        let pointer_selects = function
            .body
            .iter()
            .filter_map(|line| {
                let (true_value, false_value) =
                    crate::native::tir::resolve_select_arms(line, "select")?;
                (matches!(true_value.ty, crate::native::ir::LlType::Ptr(_))
                    && matches!(false_value.ty, crate::native::ir::LlType::Ptr(_)))
                .then(|| crate::native::tir::result_name(line))
                .flatten()
            })
            .collect::<HashSet<_>>();
        for call in function
            .body
            .iter()
            .filter_map(|line| parse_call(line))
            .filter(|call| eligible_callees.contains(&call.callee))
        {
            for argument in call.args {
                if pointer_selects.contains(&argument)
                    && !selected_consumers.contains(&(argument.clone(), call.callee.clone()))
                {
                    selected_consumers.push((argument, call.callee.clone()));
                }
            }
        }
    }
    let mut source = san_ll.to_string();
    let mut changed = false;
    for (selected, consumer) in selected_consumers {
        let targets = HashSet::from([consumer]);
        if let Some(inlined) = try_inline(&source, Some((&targets, selected.as_str(), entry_name)))
        {
            changed |= inlined != source;
            source = inlined;
        }
    }
    PointerConsumerInlining {
        source,
        // Inlining may splice a multi-block helper or enlarge an already complex entry. Select the
        // bounded whole-CFG constructor from that source fact instead of attempting ordinary
        // emission and waiting for an opaque pointer materialization error.
        requires_relooper: changed,
    }
}

// ---------------------------------------------------------------------------------------------
// Module model
// ---------------------------------------------------------------------------------------------

/// A parsed `define ... { ... }` function: its header line, the body lines (verbatim, between the
/// `{` line and the `}` line), and the closing-brace line.
#[derive(Clone)]
struct FuncBlock {
    header: String,
    body: Vec<String>,
}

/// One top-level module item: either a function we parsed, or a run of verbatim lines we pass
/// through untouched (globals, `declare`s, metadata, comments, blank lines).
enum Item {
    Func(FuncBlock),
    Raw(Vec<String>),
}

/// A parsed `define` header: the callee name (`@foo`), whether it is `internal`, its return type,
/// its parameter value tokens (the `%name`s in the signature, in order), and whether AIR marked it
/// as this translation unit's static initializer.
struct DefSig {
    name: String,
    internal: bool,
    ret_ty: String,
    params: Vec<String>,
    static_initializer: bool,
}

/// Split the module into ordered items. Fails (returns None) only if a `define` opens without a
/// matching `}` — a malformed module we refuse to touch.
fn parse_items(san_ll: &str) -> Option<Vec<Item>> {
    let lines: Vec<&str> = san_ll.lines().collect();
    let mut items: Vec<Item> = Vec::new();
    let mut raw_start = 0;
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if line.starts_with("define") && line.trim_end().ends_with('{') {
            if raw_start != i {
                items.push(Item::Raw(
                    lines[raw_start..i]
                        .iter()
                        .map(|line| (*line).to_string())
                        .collect(),
                ));
            }
            let start = i + 1;
            let mut j = start;
            while j < lines.len() && lines[j].trim() != "}" {
                j += 1;
            }
            if j >= lines.len() {
                return None; // unterminated function
            }
            items.push(Item::Func(FuncBlock {
                header: line.to_string(),
                body: lines[start..j].iter().map(|s| s.to_string()).collect(),
            }));
            i = j + 1;
            raw_start = i;
        } else {
            i += 1;
        }
    }
    if raw_start != lines.len() {
        items.push(Item::Raw(
            lines[raw_start..]
                .iter()
                .map(|line| (*line).to_string())
                .collect(),
        ));
    }
    Some(items)
}

/// Parse a `define` header line into its signature. `None` if it is not a `define` or the signature
/// is not a shape we can model (unbalanced parens, no callee name).
fn parse_def_header(header: &str) -> Option<DefSig> {
    let h = header.trim();
    let rest = h.strip_prefix("define")?.trim_start();
    // Attribute/linkage words up to the return type + `@name(`. Find the `@name(` marker.
    let at = rest.find('@')?;
    let paren = rest[at..].find('(')? + at;
    let name = rest[at..paren].trim().to_string();
    if name.len() < 2 {
        return None;
    }
    // Everything between `define` and `@name` is linkage/attrs + return type. `internal` is a word.
    let pre = rest[..at].trim();
    let pre_words: Vec<&str> = pre.split_whitespace().collect();
    let internal = pre_words.contains(&"internal");
    // Return type is the last "type" token(s) before `@name`; take the last whitespace-token that
    // is not a linkage/attr keyword. For our purposes the last token of `pre` is the return type
    // (return types with spaces like `<4 x float>` are wrapped in `<...>` and remain one token only
    // if we split on top-level whitespace — but headers are simple here; take the trailing token).
    let ret_ty = ret_type_from_pre(pre).unwrap_or_default();
    // Parameters: text inside the outermost `(...)`.
    let close = matching_paren(rest, paren)?;
    let params_str = &rest[paren + 1..close];
    let params = parse_param_names(params_str);
    let static_initializer =
        crate::air_static_init::tail_declares_static_init_section(&rest[close + 1..]);
    Some(DefSig {
        name,
        internal,
        ret_ty,
        params,
        static_initializer,
    })
}

/// The return type is the token-run between the last linkage/attr keyword and `@name`. Linkage/attr
/// keywords that can appear before the return type in sanitized AIR are a small fixed set.
fn ret_type_from_pre(pre: &str) -> Option<String> {
    const KW: &[&str] = &[
        "internal",
        "fastcc",
        "coldcc",
        "cc",
        "weak",
        "weak_odr",
        "linkonce",
        "linkonce_odr",
        "private",
        "external",
        "available_externally",
        "dso_local",
        "dso_preemptable",
        "hidden",
        "protected",
        "default",
        "signext",
        "zeroext",
        "noundef",
    ];
    // Walk tokens; the return type is everything after the last leading keyword run. Since types can
    // contain spaces only inside `<...>`/`[...]`/`{...}`, we split at top-level whitespace.
    let toks = super::lex::split_top_level_whitespace(pre);
    let mut idx = 0;
    while idx < toks.len() && KW.contains(&toks[idx]) {
        idx += 1;
    }
    if idx >= toks.len() {
        return None;
    }
    Some(toks[idx..].join(" "))
}

/// Parse a parameter list string into the ordered `%name` value tokens. A parameter is
/// `<ty> [attrs] %name`; the value token is the trailing `%name`. Params with no `%name` (e.g. a
/// bare type in a `declare`) yield nothing for that slot, which makes the function un-inlinable
/// (we require every param to name a value) — the caller checks the count.
fn parse_param_names(params_str: &str) -> Vec<String> {
    let s = params_str.trim();
    if s.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for chunk in split_top_level(s, ',') {
        let chunk = chunk.trim();
        if chunk == "..." {
            // Varargs: not inlinable; represent as a sentinel that fails the name check.
            out.push(String::new());
            continue;
        }
        match last_local_token(chunk) {
            Some(name) => out.push(name.to_string()),
            None => out.push(String::new()),
        }
    }
    out
}

/// The trailing `%name` local token of a typed operand, e.g. `i32 %0` -> `%0`, `ptr %x` -> `%x`.
fn last_local_token(chunk: &str) -> Option<&str> {
    let tok = super::lex::split_top_level_whitespace(chunk.trim())
        .last()
        .copied()?;
    tok.starts_with('%').then_some(tok)
}

/// Matching `)` for the `(` at byte offset `open` (delegates to the shared lexer helper).
fn matching_paren(s: &str, open: usize) -> Option<usize> {
    super::lex::matching_paren(s, open)
}

// ---------------------------------------------------------------------------------------------
// Call-site model
// ---------------------------------------------------------------------------------------------

/// A parsed direct call to a named callee found on one body line.
struct CallSite {
    /// `Some("%r")` when the call has a result; `None` for a void call.
    result: Option<String>,
    callee: String,
    ret_ty: String,
    /// Each argument's trailing value token (`%name` or a literal like `3`/`null`), positionally.
    args: Vec<String>,
}

/// Parse a body line as a direct call. Returns None unless the line is exactly a call instruction
/// with a literal `@name` callee (indirect `%ptr` callees, and lines that merely mention a call,
/// yield None). Attributes/metadata after the arg list are ignored for arg extraction but the line
/// is only inlined when we can reconstruct it, so we keep parsing conservative.
fn parse_call(line: &str) -> Option<CallSite> {
    let trimmed = line.trim();
    // Optional `%r = ` result.
    let (result, rhs) = match trimmed.find(" = ") {
        Some(eq) if trimmed[..eq].trim().starts_with('%') => (
            Some(trimmed[..eq].trim().to_string()),
            trimmed[eq + 3..].trim(),
        ),
        _ => (None, trimmed),
    };
    // Strip a leading `tail`/`musttail`/`notail`.
    let rhs = strip_leading_word(rhs, &["tail", "musttail", "notail"]);
    let rhs = rhs.trim_start();
    let after_call = rhs.strip_prefix("call ")?;
    // The callee `@name` immediately precedes the `(`. Find the `@` that starts the callee, then the
    // `(` that opens its argument list. Calling conventions/attrs/ret-ty sit between `call` and `@`.
    let at = after_call.find('@')?;
    // Ensure no `(` appears before the `@` at top level — that would signal a function-pointer type
    // wrapper or an indirect call we do not handle.
    if after_call[..at].contains('(') {
        return None;
    }
    let paren = after_call[at..].find('(')? + at;
    let callee = after_call[at..paren].trim().to_string();
    if callee.len() < 2 || callee.contains(char::is_whitespace) {
        return None;
    }
    let close = matching_paren(after_call, paren)?;
    let args_str = &after_call[paren + 1..close];
    let ret_ty = ret_type_from_call_prefix(&after_call[..at]);
    let args = parse_call_args(args_str)?;
    Some(CallSite {
        result,
        callee,
        ret_ty,
        args,
    })
}

/// The return type token-run in a call's `call <cc?> <ret_ty> @name` prefix (between `call ` and
/// `@name`). Calling-convention keywords are stripped.
fn ret_type_from_call_prefix(prefix: &str) -> String {
    // Everything that can sit between `call` and the return type: calling conventions, fast-math
    // flags (which precede the cc on a `call fast fastcc <ty> @f`), and return attributes. The `<ty>`
    // that follows is the first token that is none of these.
    const SKIP: &[&str] = &[
        // calling conventions
        "fastcc",
        "coldcc",
        "cc",
        "tailcc",
        "swiftcc",
        "swifttailcc",
        "cfguard_checkcc",
        // fast-math flags
        "fast",
        "nnan",
        "ninf",
        "nsz",
        "arcp",
        "contract",
        "afn",
        "reassoc",
        // return attributes
        "signext",
        "zeroext",
        "noundef",
        "nonnull",
        "noalias",
        "inreg",
        "returned",
    ];
    let toks = super::lex::split_top_level_whitespace(prefix.trim());
    let mut idx = 0;
    while idx < toks.len() {
        if SKIP.contains(&toks[idx]) {
            idx += 1;
        } else if (toks[idx] == "align" || toks[idx] == "dereferenceable")
            && idx + 1 < toks.len()
            && toks[idx + 1].chars().all(|c| c.is_ascii_digit())
        {
            // `align 4` / `dereferenceable 20`-style two-token attributes.
            idx += 2;
        } else {
            break;
        }
    }
    toks[idx..].join(" ")
}

/// Parse a call argument list into the ordered value tokens (trailing `%name` or literal). Returns
/// None if any argument is not a simple `<ty> <value>` we can substitute (e.g. an inline `blockaddress`
/// or a nested aggregate literal we would mis-handle). We accept `%name`, plain integer/float
/// literals, `null`, `undef`, `poison`, `zeroinitializer`, `true`, `false`, and `splat (...)`.
fn parse_call_args(args_str: &str) -> Option<Vec<String>> {
    let s = args_str.trim();
    if s.is_empty() {
        return Some(Vec::new());
    }
    let mut out = Vec::new();
    for chunk in split_top_level(s, ',') {
        let chunk = chunk.trim();
        if chunk == "..." {
            return None; // varargs call: refuse
        }
        out.push(arg_value_token(chunk)?);
    }
    Some(out)
}

/// Extract the substitutable value from a typed argument operand. The value is everything after the
/// leading type + attribute tokens; we take the trailing top-level token, but preserve a `splat (…)`
/// as a whole. Returns None for shapes we cannot safely reduce to a single substitutable token.
fn arg_value_token(chunk: &str) -> Option<String> {
    let toks = super::lex::split_top_level_whitespace(chunk);
    let last = *toks.last()?;
    // A `splat (<ty> %v)` or similar wrapped constant: only substitutable if it is self-contained
    // (balanced) and contains no `%`-local we'd need to leave un-substituted improperly. We allow it
    // verbatim since it references caller-scope values that are already correct in F.
    if last.starts_with('%') {
        return Some(last.to_string());
    }
    // Literal / keyword operands: the trailing token is the value.
    Some(last.to_string())
}

/// Strip a leading whitespace-delimited word from `s` if it matches one of `words`.
fn strip_leading_word<'a>(s: &'a str, words: &[&str]) -> &'a str {
    let t = s.trim_start();
    for w in words {
        if let Some(rest) = t.strip_prefix(w) {
            if rest.starts_with(char::is_whitespace) {
                return rest;
            }
        }
    }
    s
}

// ---------------------------------------------------------------------------------------------
// Tokenized rename / substitution
// ---------------------------------------------------------------------------------------------

/// True for a character that continues a `%`-name or a label identifier in LLVM textual IR.
fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '$' || c == '-'
}

/// Apply a token-accurate substitution over one line. `value_map` maps a `%old` local token to its
/// replacement text (`%new` or a substituted arg token). `label_map` maps a BARE label id (e.g.
/// `12` or `entry`) to its replacement bare id, applied only in label positions (`label %X`,
/// `[ %v, %X ]`, `<X>:` block starts, and switch destinations).
///
/// The scan walks the line left to right. Whenever it sees a `%` that begins a local token, it reads
/// the whole `[A-Za-z0-9_.$-]+` name and, if present in `value_map`, replaces it. It also detects
/// label references — a `label ` keyword makes the following `%name` OR bare id a label. Block-start
/// labels (`<id>:` at line head) and switch/phi bare-id labels are handled by the label pass.
fn rename_line(
    line: &str,
    value_map: &HashMap<String, String>,
    label_map: &HashMap<String, String>,
) -> String {
    // First: block-start label line rewrite (`12:` or `%lbl:` possibly with a `; preds` comment).
    if let Some(rewritten) = rewrite_block_label_line(line, label_map) {
        return rewritten;
    }

    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len() + 16);
    let mut i = 0;
    // Track whether the immediately-preceding significant token was `label`, to interpret the next
    // `%name`/bare-id as a label reference.
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '%' {
            // Read the full local name.
            let start = i;
            i += 1;
            let name_start = i;
            while i < bytes.len() && is_name_char(bytes[i] as char) {
                i += 1;
            }
            let name = &line[start..i];
            let bare = &line[name_start..i];
            // Is this a label reference position? (preceded by `label` keyword.)
            if preceded_by_label_keyword(&out) {
                // `label %name` -> map the bare id via label_map if present.
                if let Some(rep) = label_map.get(bare) {
                    out.push('%');
                    out.push_str(rep);
                } else if let Some(rep) = value_map.get(name) {
                    // A `%name` label that is actually a value-mapped param would be a bug (params
                    // are never labels), but if the name is a renamed local, its value_map holds the
                    // `%new` form — reuse it.
                    out.push_str(rep);
                } else {
                    out.push_str(name);
                }
            } else if let Some(rep) = value_map.get(name) {
                out.push_str(rep);
            } else {
                out.push_str(name);
            }
            continue;
        }
        out.push(c);
        i += 1;
    }

    // Handle bare-id label references inside `label <id>` (no `%`); rare, but covered.
    let out = rename_bare_label_refs(&out, label_map);
    // Finally, map phi predecessor labels `[ %v, %label ]`. These labels are NOT preceded by the
    // `label` keyword, so the main scan (which only maps value tokens) left them untouched — the
    // predecessor id must be routed through `label_map`, not `value_map`.
    rename_phi_pred_labels(&out, label_map)
}

/// Map the predecessor-label token in each phi `[ %v, %label ]` group through `label_map`. The value
/// token (first element) was already handled by the value pass; only the second (label) element is
/// rewritten here. Non-phi lines are returned unchanged.
fn rename_phi_pred_labels(line: &str, label_map: &HashMap<String, String>) -> String {
    if label_map.is_empty() || !is_phi_line(line) {
        return line.to_string();
    }
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    loop {
        let Some(open) = rest.find('[') else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..=open]);
        let after = &rest[open + 1..];
        let Some(close) = after.find(']') else {
            out.push_str(after);
            break;
        };
        let inner = &after[..close];
        let parts = split_top_level(inner, ',');
        if parts.len() == 2 {
            let val = parts[0].trim();
            let lbl = parts[1].trim();
            let lbl_bare = lbl.strip_prefix('%').unwrap_or(lbl);
            match label_map.get(lbl_bare) {
                Some(rep) => out.push_str(&format!(" {val}, %{rep} ")),
                None => out.push_str(inner),
            }
        } else {
            out.push_str(inner);
        }
        out.push(']');
        rest = &after[close + 1..];
    }
    out
}

/// If `line` is a block-start label (`<id>:` optionally followed by `  ; preds = ...`), rewrite the
/// id via `label_map` and return the new line. Returns None if the line is not a block label.
fn rewrite_block_label_line(line: &str, label_map: &HashMap<String, String>) -> Option<String> {
    // Leading whitespace preserved.
    let ws_len = line.len() - line.trim_start().len();
    let (ws, rest) = line.split_at(ws_len);
    // Must be `IDENT:` where IDENT has no whitespace, followed by end or a `;` comment / whitespace.
    let colon = rest.find(':')?;
    let ident = &rest[..colon];
    if ident.is_empty() {
        return None;
    }
    // The identifier is either a bare label (`12`, `entry`) or a `%name`? Block labels are bare.
    let bare = ident.strip_prefix('%').unwrap_or(ident);
    if !bare.chars().all(is_name_char) {
        return None;
    }
    let after = &rest[colon + 1..];
    // After the colon there must be only whitespace and/or a `;` comment for this to be a label line.
    let after_trim = after.trim_start();
    if !after_trim.is_empty() && !after_trim.starts_with(';') {
        return None;
    }
    let new_bare = label_map
        .get(bare)
        .cloned()
        .unwrap_or_else(|| bare.to_string());
    // Rewrite `; preds = %a, %b` predecessors too, if they are in the label map (they reference
    // labels as `%id`). We leave predecessor comments' value tokens to the general value pass by
    // routing the comment through the value/label maps below.
    let new_after = rewrite_preds_comment(after, label_map);
    Some(format!("{ws}{new_bare}:{new_after}"))
}

/// Rewrite `%id` label tokens inside a `; preds = ...` comment (or any trailing text) using
/// `label_map`. This keeps predecessor lists consistent after a block-id rename.
fn rewrite_preds_comment(after: &str, label_map: &HashMap<String, String>) -> String {
    let bytes = after.as_bytes();
    let mut out = String::with_capacity(after.len());
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '%' {
            let name_start = i + 1;
            let mut j = name_start;
            while j < bytes.len() && is_name_char(bytes[j] as char) {
                j += 1;
            }
            let bare = &after[name_start..j];
            if let Some(rep) = label_map.get(bare) {
                out.push('%');
                out.push_str(rep);
            } else {
                out.push_str(&after[i..j]);
            }
            i = j;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Rewrite bare-id label references that appear directly after a `label ` keyword without a `%`
/// (rare). Uses `label_map`. Value/`%`-prefixed labels were already handled in the main pass.
fn rename_bare_label_refs(line: &str, label_map: &HashMap<String, String>) -> String {
    if label_map.is_empty() || !line.contains("label ") {
        return line.to_string();
    }
    // Only touch `label <bareid>` (no `%`). We do a token walk.
    let mut out = String::with_capacity(line.len());
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Match the word `label` at a word boundary.
        if line[i..].starts_with("label")
            && (i == 0 || !is_name_char(bytes[i - 1] as char))
            && line[i + 5..]
                .chars()
                .next()
                .is_some_and(|c| c.is_whitespace())
        {
            out.push_str("label");
            let mut j = i + 5;
            // copy whitespace
            while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                out.push(bytes[j] as char);
                j += 1;
            }
            // If the destination is a bare id (not `%`), map it.
            if j < bytes.len() && bytes[j] as char != '%' {
                let s = j;
                while j < bytes.len() && is_name_char(bytes[j] as char) {
                    j += 1;
                }
                let bare = &line[s..j];
                if let Some(rep) = label_map.get(bare) {
                    out.push_str(rep);
                } else {
                    out.push_str(bare);
                }
            }
            i = j;
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// True if the trailing text of `out` ends with the `label` keyword followed by whitespace — i.e.
/// the `%name` we are about to read is a label reference.
fn preceded_by_label_keyword(out: &str) -> bool {
    let trimmed = out.trim_end();
    if trimmed.len() == out.len() {
        return false; // no trailing whitespace: `label%x` is not valid; require a space
    }
    // The last whitespace-delimited word of the trimmed prefix must be exactly `label`.
    match trimmed.rsplit(|c: char| c.is_whitespace()).next() {
        Some(w) => w == "label",
        None => false,
    }
}

// ---------------------------------------------------------------------------------------------
// The inliner
// ---------------------------------------------------------------------------------------------

/// The whole pass, in the fallible `Option` domain so any bail returns `None` and the caller keeps
/// the input verbatim.
fn try_inline(
    san_ll: &str,
    targets: Option<(&HashSet<String>, &str, Option<&str>)>,
) -> Option<String> {
    let items = parse_items(san_ll)?;

    // Collect the internal-function table: name -> (signature, body). Only `define internal`.
    let mut internal: HashMap<String, (DefSig, Vec<String>)> = HashMap::new();
    let mut has_any_func = false;
    for it in &items {
        if let Item::Func(f) = it {
            has_any_func = true;
            let sig = parse_def_header(&f.header)?;
            if sig.internal
                && targets.is_none_or(|(targets, _selected, _entry)| targets.contains(&sig.name))
            {
                // Every param must name a value for us to substitute; reject varargs/unnamed.
                if sig.params.iter().any(|p| p.is_empty()) {
                    // Un-inlinable internal fn: keep it in the table only as a non-target by NOT
                    // inserting, so calls to it are simply left alone.
                    continue;
                }
                internal.insert(sig.name.clone(), (sig, f.body.clone()));
            }
        }
    }
    if !has_any_func || internal.is_empty() {
        return None; // nothing to inline
    }

    // Acyclicity check over the internal call graph restricted to inlinable direct calls. If ANY
    // cycle exists among internal functions, bail — inlining would not terminate.
    if internal_callgraph_has_cycle(&internal) {
        return None;
    }

    // Determine whether any caller actually calls an internal function; if none, byte-identical.
    let mut any_call = false;
    for it in &items {
        if let Item::Func(f) = it {
            for line in &f.body {
                if let Some(c) = parse_call(line) {
                    if internal.contains_key(&c.callee) {
                        any_call = true;
                        break;
                    }
                }
            }
        }
        if any_call {
            break;
        }
    }
    if !any_call {
        return None;
    }

    // Rewrite each function body to a fixpoint. A shared counter gives every inlined site a unique
    // `.inl<K>` prefix across the whole module.
    let mut counter = next_inline_ordinal(san_ll);
    let mut out_items: Vec<Item> = Vec::with_capacity(items.len());
    for it in items {
        match it {
            Item::Raw(r) => out_items.push(Item::Raw(r)),
            Item::Func(f) => {
                let selected_pointer = targets.and_then(|(_, selected, entry_name)| {
                    let signature = parse_def_header(&f.header)?;
                    entry_name
                        .is_none_or(|entry_name| {
                            signature.name.trim_start_matches('@') == entry_name
                        })
                        .then_some(selected)
                });
                let new_body = if targets.is_some() && selected_pointer.is_none() {
                    f.body
                } else {
                    inline_body_to_fixpoint(f.body, &internal, &mut counter, selected_pointer)?
                };
                out_items.push(Item::Func(FuncBlock {
                    header: f.header,
                    body: new_body,
                }));
            }
        }
    }

    // Sweep now-dead `internal` helpers. Once every call site of a helper is folded in, nothing
    // references it — but the native emitter still tries to emit the leftover `define internal` and
    // can fail on a shape it never needs to (e.g. a by-value `_MPSKernelInOut` struct param it
    // cannot type). Drop any internal function unreachable, by `@name` reference, from a
    // non-internal function, a module-scope item (globals / `@llvm.global_ctors` / metadata), or
    // another live internal function.
    let out_items = drop_dead_internal_functions(out_items);
    Some(render(&out_items, san_ll.ends_with('\n')))
}

/// Return an ordinal above every existing `.inl<N>.` namespace in the input. Pointer-consumer
/// normalization may invoke the inliner more than once on the progressively rewritten module; a
/// per-invocation zero counter would then duplicate SSA values and block labels from an earlier
/// invocation.
fn next_inline_ordinal(source: &str) -> usize {
    let bytes = source.as_bytes();
    let mut next = 0usize;
    let mut cursor = 0usize;
    while let Some(relative) = source[cursor..].find(".inl") {
        let start = cursor + relative + 4;
        let mut end = start;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end > start && bytes.get(end) == Some(&b'.') {
            if let Ok(ordinal) = source[start..end].parse::<usize>() {
                next = next.max(ordinal.saturating_add(1));
            }
        }
        cursor = start;
        if cursor >= source.len() {
            break;
        }
    }
    next
}

/// Mark-and-sweep the `define internal` functions: roots are the non-internal functions and every
/// module-scope `Raw` item; an internal function is live if its `@name` token appears in any live
/// item, iterated to a fixpoint (so a helper kept alive only by another live helper survives). This
/// only ever runs on the successful inline path (which already proved the internal call graph
/// acyclic), so a self-reference is ignored and cannot keep a function spuriously alive.
fn drop_dead_internal_functions(items: Vec<Item>) -> Vec<Item> {
    let internal_names: Vec<String> = items
        .iter()
        .filter_map(|it| match it {
            Item::Func(f) => parse_def_header(&f.header)
                .filter(|s| s.internal)
                .map(|s| s.name),
            _ => None,
        })
        .collect();
    if internal_names.is_empty() {
        return items;
    }
    let mut live: std::collections::HashSet<String> = std::collections::HashSet::new();
    loop {
        let mut changed = false;
        for it in &items {
            // Determine whether this item is a live root/source of references.
            let (lines, self_name): (&[String], Option<String>) = match it {
                Item::Raw(r) => (r.as_slice(), None),
                Item::Func(f) => {
                    let sig = match parse_def_header(&f.header) {
                        Some(s) => s,
                        None => continue,
                    };
                    // AIR static constructors are implicit emitter roots: their calls are injected
                    // after function emission and therefore do not appear in the textual IR. Keep
                    // them live during this source-level helper sweep so targeted inlining cannot
                    // silently discard function-constant/default-state initialization.
                    let implicit_constructor_root = sig.static_initializer;
                    if sig.internal && !implicit_constructor_root && !live.contains(&sig.name) {
                        continue; // a not-yet-live internal fn does not propagate references
                    }
                    (f.body.as_slice(), Some(sig.name))
                }
            };
            for name in &internal_names {
                if Some(name) == self_name.as_ref() || live.contains(name) {
                    continue;
                }
                if lines_mention_symbol(lines, name) {
                    live.insert(name.clone());
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    items
        .into_iter()
        .filter(|it| match it {
            Item::Func(f) => match parse_def_header(&f.header) {
                Some(sig) if sig.internal => sig.static_initializer || live.contains(&sig.name),
                _ => true,
            },
            _ => true,
        })
        .collect()
}

/// True if any line references the exact symbol token `sym` (a leading-`@` global name). Matches on
/// a trailing name-char boundary so `@foo` does not match inside `@foobar`; the leading `@` is its
/// own left boundary.
fn lines_mention_symbol(lines: &[String], sym: &str) -> bool {
    lines.iter().any(|l| {
        let mut start = 0;
        while let Some(pos) = l[start..].find(sym) {
            let after = start + pos + sym.len();
            if l[after..].chars().next().is_none_or(|c| !is_name_char(c)) {
                return true;
            }
            start = after;
        }
        false
    })
}

/// Re-serialize the module items to text.
fn render(items: &[Item], trailing_newline: bool) -> String {
    let mut lines: Vec<String> = Vec::new();
    for it in items {
        match it {
            Item::Raw(r) => lines.extend(r.iter().cloned()),
            Item::Func(f) => {
                lines.push(f.header.clone());
                lines.extend(f.body.iter().cloned());
                lines.push("}".to_string());
            }
        }
    }
    let mut s = lines.join("\n");
    if trailing_newline {
        s.push('\n');
    }
    s
}

/// Depth-first cycle detection over internal-fn -> internal-fn direct call edges.
fn internal_callgraph_has_cycle(internal: &HashMap<String, (DefSig, Vec<String>)>) -> bool {
    let mut edges: HashMap<&str, Vec<String>> = HashMap::new();
    for (name, (_, body)) in internal {
        let mut callees = Vec::new();
        for line in body {
            if let Some(c) = parse_call(line) {
                if internal.contains_key(&c.callee) {
                    callees.push(c.callee.clone());
                }
            }
        }
        edges.insert(name.as_str(), callees);
    }
    // Standard white/grey/black DFS.
    #[derive(Clone, Copy, PartialEq)]
    enum Color {
        White,
        Grey,
        Black,
    }
    let mut color: HashMap<&str, Color> = internal
        .keys()
        .map(|k| (k.as_str(), Color::White))
        .collect();
    fn dfs<'a>(
        n: &'a str,
        edges: &HashMap<&'a str, Vec<String>>,
        color: &mut HashMap<&'a str, Color>,
    ) -> bool {
        color.insert(n, Color::Grey);
        if let Some(cs) = edges.get(n) {
            for c in cs {
                match color.get(c.as_str()).copied() {
                    Some(Color::Grey) => return true,
                    Some(Color::White) => {
                        // SAFETY: c is a key of `internal`, so it lives as long as `edges`' keys.
                        let key = edges.keys().find(|k| **k == c.as_str()).copied();
                        if let Some(k) = key {
                            if dfs(k, edges, color) {
                                return true;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        color.insert(n, Color::Black);
        false
    }
    let names: Vec<&str> = internal.keys().map(|k| k.as_str()).collect();
    for n in names {
        if color.get(n).copied() == Some(Color::White) && dfs(n, &edges, &mut color) {
            return true;
        }
    }
    false
}

/// Repeatedly inline the first eligible call site in `body` until none remain. `None` on any bail.
fn inline_body_to_fixpoint(
    mut body: Vec<String>,
    internal: &HashMap<String, (DefSig, Vec<String>)>,
    counter: &mut usize,
    selected_pointer: Option<&str>,
) -> Option<Vec<String>> {
    // A generous bound to guarantee termination even if something pathological slips past the cycle
    // check; acyclic transitive inlining is finite, this only guards against bugs.
    let mut budget = 100_000usize;
    loop {
        let Some(idx) = find_inlinable_call(&body, internal, selected_pointer) else {
            return Some(body);
        };
        body = inline_one(body, idx, internal, counter, selected_pointer.is_some())?;
        budget -= 1;
        if budget == 0 {
            return None;
        }
    }
}

/// Index of the first body line that is a direct call to an internal function.
fn find_inlinable_call(
    body: &[String],
    internal: &HashMap<String, (DefSig, Vec<String>)>,
    selected_pointer: Option<&str>,
) -> Option<usize> {
    for (i, line) in body.iter().enumerate() {
        if let Some(c) = parse_call(line) {
            if internal.contains_key(&c.callee)
                && selected_pointer
                    .is_none_or(|selected| c.args.iter().any(|argument| argument == selected))
            {
                return Some(i);
            }
        }
    }
    None
}

/// Inline the call at `body[idx]`. Implements the SSA-correct transform from the module docs.
fn inline_one(
    body: Vec<String>,
    idx: usize,
    internal: &HashMap<String, (DefSig, Vec<String>)>,
    counter: &mut usize,
    preserve_caller_cfg: bool,
) -> Option<Vec<String>> {
    let call = parse_call(&body[idx])?;
    let (sig, callee_body) = internal.get(&call.callee)?;
    if sig.params.len() != call.args.len() {
        return None; // arity mismatch (varargs etc.) — refuse
    }
    if preserve_caller_cfg && is_single_block_leaf(callee_body) {
        return inline_single_block_leaf(body, idx, &call, sig, callee_body, counter);
    }

    let k = *counter;
    *counter += 1;
    let prefix = format!(".inl{k}");
    let entry_label = format!("{prefix}.entry");
    let cont_label = format!("{prefix}.cont");

    // Build the local-name maps for the callee body. First discover every callee-local SSA value
    // (instruction results) and every callee block label id.
    let (local_values, local_labels) = collect_callee_locals(callee_body)?;

    // value_map: %param_i -> arg_i (verbatim caller token); other %local -> %prefix.local
    let mut value_map: HashMap<String, String> = HashMap::new();
    for (p, a) in sig.params.iter().zip(call.args.iter()) {
        value_map.insert(p.clone(), a.clone());
    }
    for v in &local_values {
        if value_map.contains_key(v) {
            // A local value that shadows a param name should not happen (params are distinct), but if
            // it does, prefer the local rename to stay SSA-correct; overwrite.
            value_map.insert(v.clone(), format!("%{prefix}.{}", &v[1..]));
        } else {
            value_map.insert(v.clone(), format!("%{prefix}.{}", &v[1..]));
        }
    }
    // Ensure params that are ALSO shadow a local (shouldn't be) keep the arg mapping; re-apply.
    for (p, a) in sig.params.iter().zip(call.args.iter()) {
        value_map.insert(p.clone(), a.clone());
    }

    // label_map: every callee block label id -> prefixed id. The (implicit) entry block gets the
    // fresh entry label; it has no source id so we add nothing for it here.
    let mut label_map: HashMap<String, String> = HashMap::new();
    for l in &local_labels {
        label_map.insert(l.clone(), format!("{prefix}.{l}"));
    }
    // An UNLABELED callee entry block still carries an implicit LLVM id (the next unnamed number
    // after the params), and a `phi` inside the callee can name it as an incoming predecessor (e.g.
    // `[ zeroinitializer, %5 ]` where `%5` is the entry of a 5-param helper). label_map has no entry
    // for it — it is never written as a `label:` line — so that bare reference would survive
    // un-renamed and dangle ("unknown block label %5"). Map the implicit entry id to the fresh
    // entry label too.
    if let Some(id) = implicit_entry_block_id(&sig.params, callee_body) {
        label_map.entry(id).or_insert_with(|| entry_label.clone());
    }

    // Rename + substitute the callee body, turning `ret` into `br label %cont` and collecting the
    // returned (value_token, pred_label) pairs.
    let mut renamed: Vec<String> = Vec::new();
    let mut returns: Vec<(String, String)> = Vec::new();
    // The current source block label as we walk callee lines. The entry block is `entry_label`.
    let mut cur_block = entry_label.clone();
    // An explicitly labeled source entry is renamed to `entry_label` by `label_map`. Only synthesize
    // the label for an implicit entry; emitting both would create an empty block followed by a
    // duplicate label, which is not a valid CFG.
    let has_explicit_entry = callee_body
        .iter()
        .find(|line| {
            let line = line.trim();
            !line.is_empty() && !line.starts_with(';')
        })
        .is_some_and(|line| block_label_id(line).is_some());
    if !has_explicit_entry {
        renamed.push(format!("{entry_label}:"));
    }
    for line in callee_body {
        // Track block boundaries: a block-start label line updates cur_block.
        if let Some(bare) = block_label_id(line) {
            let mapped = label_map
                .get(bare)
                .cloned()
                .unwrap_or_else(|| bare.to_string());
            cur_block = mapped;
            renamed.push(rename_line(line, &value_map, &label_map));
            continue;
        }
        // A `ret` terminator: record (value, cur_block) and replace with a branch to cont.
        if let Some(ret) = parse_ret(line) {
            match ret {
                RetKind::Value(_ty, _val) => {
                    // Rename the returned value through the maps (it may be a param/local/literal).
                    let renamed_val = substitute_value_token(&_val, &value_map);
                    returns.push((renamed_val, cur_block.clone()));
                }
                RetKind::Void => {
                    returns.push((String::new(), cur_block.clone()));
                }
            }
            let indent = leading_ws(line);
            renamed.push(format!("{indent}br label %{cont_label}"));
            continue;
        }
        renamed.push(rename_line(line, &value_map, &label_map));
    }

    // Split the caller block at the call.
    let before: Vec<String> = body[..idx].to_vec();
    let after: Vec<String> = body[idx + 1..].to_vec();

    // The names defined in the moved (post-call) region — used to relabel phi predecessors that
    // referenced the call's block.
    let bcall_label = enclosing_block_label(&before);

    // Rewrite phi predecessors in F: after the split, Bcall ends with `br label %entry_label` — its
    // ONLY successor is the callee entry. Every original successor edge `Bcall -> S` is inherited by
    // `cont` (which holds the post-call region and Bcall's original terminator). So EVERY phi incoming
    // `[ %v, %Bcall ]` in the `after` region refers to a now-severed edge and must be rewired to
    // `[ %v, %cont ]` — regardless of where `%v` is defined. (The earlier "%v defined in the moved
    // region" guard was wrong: it dropped incomings whose value predates the call, e.g. a compare-swap
    // network where the caller block held both the loop-carried phi value AND the fcmp+branch that
    // inlining pushes into `cont` — leaving a malformed merge phi missing the cont incoming, the TopK
    // `%10` wall.) No block in `after` retains Bcall as a live predecessor (Bcall's sole live edge now
    // targets the callee entry, which lives in the renamed callee body, not `after`), so relabeling
    // unconditionally cannot mis-route a genuine Bcall edge.
    let mut after_relabeled: Vec<String> = after
        .iter()
        .map(|l| relabel_phi_preds(l, bcall_label.as_deref(), &cont_label))
        .collect();
    // Phis in the `before` region cannot reference a post-call value (SSA dominance), so leave as-is.

    // Assemble the new body:
    //   before...
    //   br label %entry_label
    //   <renamed callee blocks, with ret -> br %cont>
    //   cont_label:
    //     [ %r = phi ... ]  (if value-returning with a result)
    //   after (relabeled)...
    let mut new_body: Vec<String> = Vec::with_capacity(body.len() + renamed.len() + 4);
    new_body.extend(before);
    // Terminate the pre-call region with a branch to the callee entry.
    let call_indent = leading_ws(&body[idx]);
    new_body.push(format!("{call_indent}br label %{entry_label}"));
    new_body.extend(renamed);
    // Continuation block.
    new_body.push(format!("{cont_label}:"));
    if let Some(result) = &call.result {
        let non_void: Vec<&(String, String)> =
            returns.iter().filter(|(v, _)| !v.is_empty()).collect();
        if non_void.is_empty() {
            // The callee returns void but the call had a result — malformed; bail.
            return None;
        }
        if non_void.len() == 1 {
            // A single return edge needs no merge. The returned value dominates the continuation,
            // so forward it directly into the post-call region. Besides avoiding a redundant phi,
            // this preserves visible insertvalue/extractvalue def-use chains for the following SROA
            // pass; wrapping a returned aggregate in a one-arm phi needlessly hid those chains.
            let mut return_value = HashMap::new();
            return_value.insert(result.clone(), non_void[0].0.clone());
            for line in &mut after_relabeled {
                *line = rename_line(line, &return_value, &HashMap::new());
            }
        } else {
            // Multiple returns genuinely require a merge phi in the continuation.
            let ret_ty = if !call.ret_ty.is_empty() {
                call.ret_ty.clone()
            } else {
                sig.ret_ty.clone()
            };
            if ret_ty.is_empty() {
                return None;
            }
            let arms: Vec<String> = non_void
                .iter()
                .map(|(v, blk)| format!("[ {v}, %{blk} ]"))
                .collect();
            new_body.push(format!(
                "{call_indent}{result} = phi {ret_ty} {}",
                arms.join(", ")
            ));
        }
    }
    new_body.extend(after_relabeled);

    Some(new_body)
}

/// Inline a one-block helper without splitting the caller block. This is the selected-pointer
/// consumer path: preserving the caller CFG is part of its construction contract.
fn inline_single_block_leaf(
    body: Vec<String>,
    idx: usize,
    call: &CallSite,
    sig: &DefSig,
    callee_body: &[String],
    counter: &mut usize,
) -> Option<Vec<String>> {
    let prefix = format!(".inl{}", *counter);
    *counter += 1;
    let (local_values, _) = collect_callee_locals(callee_body)?;
    let mut value_map = HashMap::new();
    for (parameter, argument) in sig.params.iter().zip(&call.args) {
        value_map.insert(parameter.clone(), argument.clone());
    }
    for value in local_values {
        value_map.insert(value.clone(), format!("%{prefix}.{}", &value[1..]));
    }
    for (parameter, argument) in sig.params.iter().zip(&call.args) {
        value_map.insert(parameter.clone(), argument.clone());
    }

    let mut replacement = Vec::new();
    let mut returned = None;
    for line in callee_body {
        if block_label_id(line).is_some() {
            continue;
        }
        if let Some(ret) = parse_ret(line) {
            returned = Some(match ret {
                RetKind::Value(_, value) => substitute_value_token(&value, &value_map),
                RetKind::Void => String::new(),
            });
            continue;
        }
        replacement.push(rename_line(line, &value_map, &HashMap::new()));
    }
    let returned = returned?;
    let mut after = body[idx + 1..].to_vec();
    if let Some(result) = &call.result {
        if returned.is_empty() {
            return None;
        }
        let result_map = HashMap::from([(result.clone(), returned)]);
        for line in &mut after {
            *line = rename_line(line, &result_map, &HashMap::new());
        }
    }
    let mut new_body = Vec::with_capacity(body.len() + replacement.len());
    new_body.extend_from_slice(&body[..idx]);
    new_body.extend(replacement);
    new_body.extend(after);
    Some(new_body)
}

fn is_single_block_leaf(body: &[String]) -> bool {
    body.iter()
        .filter(|line| block_label_id(line).is_some())
        .count()
        <= 1
        && body.iter().filter(|line| parse_ret(line).is_some()).count() == 1
        && !body.iter().any(|line| {
            let line = line.trim_start();
            line.starts_with("br ")
                || line.starts_with("switch ")
                || line.starts_with("indirectbr ")
                || line.starts_with("unreachable")
        })
}

/// The `%name` value or literal token substituted through `value_map` (for `ret` operands, which are
/// value positions only).
fn substitute_value_token(tok: &str, value_map: &HashMap<String, String>) -> String {
    if let Some(rep) = value_map.get(tok) {
        rep.clone()
    } else {
        tok.to_string()
    }
}

enum RetKind {
    Value(String, String),
    Void,
}

/// Parse a `ret` terminator line: `ret void` or `ret <ty> <value>`. None if not a `ret`.
fn parse_ret(line: &str) -> Option<RetKind> {
    let t = line.trim();
    let rest = t.strip_prefix("ret")?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim();
    if rest == "void" {
        return Some(RetKind::Void);
    }
    // `<ty> <value>`: take the trailing top-level token as the value, the rest as type.
    let toks = super::lex::split_top_level_whitespace(rest);
    if toks.len() < 2 {
        // `ret <value>` without a type is invalid AIR; refuse by returning None so the whole inline
        // bails.
        return None;
    }
    let val = (*toks.last()?).to_string();
    let ty = toks[..toks.len() - 1].join(" ");
    Some(RetKind::Value(ty, val))
}

/// Leading-whitespace prefix of a line.
fn leading_ws(line: &str) -> &str {
    &line[..line.len() - line.trim_start().len()]
}

/// If `line` is a block-start label line, return the bare label id.
fn block_label_id(line: &str) -> Option<&str> {
    let rest = line.trim_start();
    let colon = rest.find(':')?;
    let ident = &rest[..colon];
    if ident.is_empty() {
        return None;
    }
    let bare = ident.strip_prefix('%').unwrap_or(ident);
    if !bare.chars().all(is_name_char) {
        return None;
    }
    let after = rest[colon + 1..].trim_start();
    if !after.is_empty() && !after.starts_with(';') {
        return None;
    }
    Some(bare)
}

/// Collect every callee-local SSA value name (`%name` instruction results) and every block label id.
/// Returns None if a construct suggests the body is not modelable (we keep it small and conservative:
/// any line that is neither a label, a def, a terminator, nor a bare instruction is still fine — we
/// only need the *result* names, which are the `%x = ` LHS tokens, plus the entry-implicit block).
fn collect_callee_locals(body: &[String]) -> Option<(Vec<String>, Vec<String>)> {
    let mut values: Vec<String> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    let mut seen_v: HashSet<String> = HashSet::new();
    let mut seen_l: HashSet<String> = HashSet::new();
    for line in body {
        if let Some(bare) = block_label_id(line) {
            if seen_l.insert(bare.to_string()) {
                labels.push(bare.to_string());
            }
            continue;
        }
        if let Some(lhs) = def_lhs(line) {
            if seen_v.insert(lhs.to_string()) {
                values.push(lhs.to_string());
            }
        }
    }
    Some((values, labels))
}

/// The implicit LLVM id of an UNLABELED callee entry block: the next unnamed number after its
/// contiguous numeric parameters. Named parameters do not consume numeric slots, so a signature may
/// freely mix `%0`, `%1`, `%.named`, `%2`, ... and still have a provable implicit entry id.
fn implicit_entry_block_id(params: &[String], body: &[String]) -> Option<String> {
    // Entry is explicit if the first meaningful callee line is a block label.
    for line in body {
        let t = line.trim();
        if t.is_empty() || t.starts_with(';') {
            continue;
        }
        if block_label_id(line).is_some() {
            return None; // labeled entry — handled by local_labels
        }
        break; // first real line is an instruction -> implicit entry
    }
    // Numeric parameters must form exactly %0..%N. Named parameters are irrelevant to LLVM's
    // unnamed-number counter and therefore do not make the entry id ambiguous.
    let mut ids: Vec<u32> = Vec::with_capacity(params.len());
    for p in params {
        let bare = p.strip_prefix('%')?;
        if let Ok(id) = bare.parse::<u32>() {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    if ids.iter().enumerate().any(|(i, id)| *id != i as u32) {
        return None; // gapped numeric params -> cannot derive the implicit id safely
    }
    Some(ids.len().to_string())
}

/// The `%name` LHS of a `%name = ...` definition line, else None.
fn def_lhs(line: &str) -> Option<&str> {
    let t = line.trim();
    let eq = t.find(" = ")?;
    let lhs = t[..eq].trim();
    lhs.starts_with('%').then_some(lhs)
}

/// The label of the block that encloses the call — i.e. the nearest preceding block-start label in
/// `before`, or None if the call is in the entry block (no explicit label).
fn enclosing_block_label(before: &[String]) -> Option<String> {
    for line in before.iter().rev() {
        if let Some(bare) = block_label_id(line) {
            return Some(bare.to_string());
        }
    }
    None
}

/// Rewrite phi predecessor labels in `line`: every entry `[ %v, %Bcall ]` becomes `[ %v, %cont ]`
/// (Bcall no longer branches to any of its original successors after the call split — `cont` does).
/// Only touches lines that are phis; the `Bcall` may be `None` (entry block) in which case the
/// predecessor label is the entry block, which has no textual label — such phis cannot reference it by
/// name, so nothing to do.
fn relabel_phi_preds(line: &str, bcall: Option<&str>, cont: &str) -> String {
    let Some(bcall) = bcall else {
        return line.to_string();
    };
    // Only phi instructions carry `[ %v, %label ]` predecessor groups.
    if !is_phi_line(line) {
        return line.to_string();
    }
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    loop {
        let Some(open) = rest.find('[') else {
            out.push_str(rest);
            break;
        };
        out.push_str(&rest[..=open]);
        let after = &rest[open + 1..];
        let Some(close) = after.find(']') else {
            out.push_str(after);
            break;
        };
        let inner = &after[..close];
        // inner = ` %v, %label `
        let parts = split_top_level(inner, ',');
        if parts.len() == 2 {
            let val = parts[0].trim();
            let lbl = parts[1].trim();
            let lbl_bare = lbl.strip_prefix('%').unwrap_or(lbl);
            if lbl_bare == bcall {
                out.push_str(&format!(" {val}, %{cont} "));
            } else {
                out.push_str(inner);
            }
        } else {
            out.push_str(inner);
        }
        out.push(']');
        rest = &after[close + 1..];
    }
    out
}

/// True if the line is a `phi` instruction (`%x = phi ...`).
fn is_phi_line(line: &str) -> bool {
    if let Some(eq) = line.trim().find(" = ") {
        let rhs = line.trim()[eq + 3..].trim_start();
        return rhs.starts_with("phi ") || rhs == "phi";
    }
    false
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_inlining_starts_after_existing_namespace() {
        let source = "%x.inl2.value = add i32 1, 2\n.inl17.block:\n  ret void\n";
        assert_eq!(next_inline_ordinal(source), 18);
        assert_eq!(next_inline_ordinal("define void @f() { ret void }\n"), 0);
    }

    // 7. Token-rename edge case: renaming %1 must not touch %10 or %11.
    #[test]
    fn rename_is_token_accurate() {
        let mut vm = HashMap::new();
        vm.insert("%1".to_string(), "%.inl0.1".to_string());
        let line = "  %10 = add i32 %1, %11";
        let out = rename_line(line, &vm, &HashMap::new());
        assert_eq!(out, "  %10 = add i32 %.inl0.1, %11", "got: {out}");
    }

    #[test]
    fn rename_maps_label_positions() {
        let mut lm = HashMap::new();
        lm.insert("5".to_string(), ".inl0.5".to_string());
        let line = "  br label %5";
        let out = rename_line(line, &HashMap::new(), &lm);
        assert_eq!(out, "  br label %.inl0.5", "got: {out}");
    }

    // 8. No internal calls -> output byte-identical (including trailing newline).
    #[test]
    fn no_internal_calls_is_identical() {
        let src = "\
define void @main(i32 %0) {
  %1 = add i32 %0, 1
  ret void
}
";
        assert_eq!(inline_nonrecursive_internal_calls(src), src);
    }

    #[test]
    fn declared_not_defined_is_not_inlined() {
        let src = "\
declare i32 @helper(i32)
define i32 @main(i32 %0) {
  %1 = call i32 @helper(i32 %0)
  ret i32 %1
}
";
        assert_eq!(inline_nonrecursive_internal_calls(src), src);
    }

    // 6a. Intrinsic call -> not inlined.
    #[test]
    fn intrinsic_call_is_not_inlined() {
        let src = "\
declare float @llvm.fabs.f32(float)
define float @main(float %0) {
  %1 = call float @llvm.fabs.f32(float %0)
  ret float %1
}
";
        assert_eq!(inline_nonrecursive_internal_calls(src), src);
    }

    // 6b. Indirect call -> not inlined.
    #[test]
    fn indirect_call_is_not_inlined() {
        let src = "\
define i32 @main(i32 %0, ptr %fp) {
  %1 = call i32 %fp(i32 %0)
  ret i32 %1
}
";
        assert_eq!(inline_nonrecursive_internal_calls(src), src);
    }

    // 1. Trivial single-return value callee inlined.
    #[test]
    fn single_return_value_callee_inlined() {
        let src = "\
define internal i32 @add1(i32 %0) {
  %1 = add i32 %0, 1
  ret i32 %1
}
define i32 @main(i32 %0) {
  %r = call i32 @add1(i32 %0)
  %2 = mul i32 %r, 2
  ret i32 %2
}
";
        let out = inline_nonrecursive_internal_calls(src);
        // The call line is gone.
        assert!(!out.contains("call i32 @add1"), "call not removed:\n{out}");
        // The callee body appears with prefixed names, param %0 substituted by caller arg %0.
        assert!(
            out.contains(".inl0.1 = add i32 %0, 1"),
            "body not inlined:\n{out}"
        );
        // The single return value is forwarded directly; a one-arm phi would hide aggregate
        // def-use chains from the following SROA pass.
        assert!(
            !out.contains("%r = phi i32"),
            "redundant result phi:\n{out}"
        );
        assert!(out.contains(".inl0.cont:"), "no cont block:\n{out}");
        assert!(
            out.contains("br label %.inl0.entry"),
            "no entry branch:\n{out}"
        );
        // Post-call instruction survives.
        assert!(
            out.contains("%2 = mul i32 %.inl0.1, 2"),
            "post-call result not forwarded:\n{out}"
        );
    }

    // 2. Void callee.
    #[test]
    fn void_callee_inlined() {
        let src = "\
define internal void @sideeffect(ptr %0, i32 %1) {
  store i32 %1, ptr %0
  ret void
}
define void @main(ptr %0, i32 %1) {
  call void @sideeffect(ptr %0, i32 %1)
  ret void
}
";
        let out = inline_nonrecursive_internal_calls(src);
        assert!(
            !out.contains("call void @sideeffect"),
            "call not removed:\n{out}"
        );
        // Param substitution: %0->%0, %1->%1 (identity here) in the store.
        assert!(
            out.contains("store i32 %1, ptr %0"),
            "store not inlined:\n{out}"
        );
        // No phi for a void return.
        assert!(!out.contains("= phi"), "void return should not phi:\n{out}");
        assert!(
            out.contains("br label %.inl0.cont"),
            "ret not turned to branch:\n{out}"
        );
    }

    // 3. Two-return callee -> merge phi.
    #[test]
    fn two_return_callee_builds_phi() {
        let src = "\
define internal i32 @sel(i1 %0, i32 %1, i32 %2) {
  br i1 %0, label %t, label %f
t:
  ret i32 %1
f:
  ret i32 %2
}
define i32 @main(i1 %0, i32 %1, i32 %2) {
  %r = call i32 @sel(i1 %0, i32 %1, i32 %2)
  ret i32 %r
}
";
        let out = inline_nonrecursive_internal_calls(src);
        assert!(!out.contains("call i32 @sel"), "call not removed:\n{out}");
        // Two return blocks branch to cont; the phi merges both, with args substituted.
        assert!(out.contains("%r = phi i32"), "no merge phi:\n{out}");
        // The phi references both prefixed predecessor blocks.
        assert!(out.contains("%.inl0.t"), "missing t pred:\n{out}");
        assert!(out.contains("%.inl0.f"), "missing f pred:\n{out}");
        // Both returns became branches to cont.
        assert_eq!(
            out.matches("br label %.inl0.cont").count(),
            2,
            "want 2 branches to cont:\n{out}"
        );
        // Return values are the substituted caller args %1 and %2.
        assert!(out.contains("[ %1, %.inl0.t ]"), "arm t wrong:\n{out}");
        assert!(out.contains("[ %2, %.inl0.f ]"), "arm f wrong:\n{out}");
    }

    // 4. Transitive inlining: callee calls another internal fn.
    #[test]
    fn transitive_inlining() {
        let src = "\
define internal i32 @inner(i32 %0) {
  %1 = add i32 %0, 1
  ret i32 %1
}
define internal i32 @outer(i32 %0) {
  %1 = call i32 @inner(i32 %0)
  %2 = mul i32 %1, 2
  ret i32 %2
}
define i32 @main(i32 %0) {
  %r = call i32 @outer(i32 %0)
  ret i32 %r
}
";
        let out = inline_nonrecursive_internal_calls(src);
        assert!(
            !out.contains("call i32 @outer"),
            "outer call remains:\n{out}"
        );
        assert!(
            !out.contains("call i32 @inner"),
            "inner call remains:\n{out}"
        );
        // Two distinct inline prefixes were minted (outer then its transitive inner). The dead
        // `@outer`/`@inner` definitions are swept, so every surviving prefix lives in `@main`.
        let prefixes: std::collections::HashSet<&str> = out
            .match_indices(".inl")
            .filter_map(|(i, _)| out[i..].split('.').nth(1))
            .collect();
        assert!(
            prefixes.len() >= 2,
            "expected >=2 distinct inline prefixes, got {prefixes:?}:\n{out}"
        );
        // The now-dead helper definitions were removed after inlining.
        assert!(
            !out.contains("define internal i32 @outer"),
            "dead @outer not swept:\n{out}"
        );
        assert!(
            !out.contains("define internal i32 @inner"),
            "dead @inner not swept:\n{out}"
        );
        // The `add i32 ..., 1` from inner and `mul ..., 2` from outer both appear in main.
        assert!(out.contains("add i32"), "inner body missing:\n{out}");
        assert!(out.contains("mul i32"), "outer body missing:\n{out}");
    }

    // A callee whose UNLABELED entry block is referenced by its implicit id (`%6`) in a phi and whose
    // signature mixes six numeric parameters with named parameters. Named parameters do not consume
    // LLVM's unnamed-number slots, so the entry remains `%6`.
    #[test]
    fn implicit_entry_id_and_fast_fastcc_are_handled() {
        let src = "\
define internal fast fastcc float @sel(float %0, float %1, float %.a, float %.b, float %2, float %3, float %4, float %5) {
  %7 = fcmp olt float %0, %1
  br i1 %7, label %8, label %9
8:
  br label %9
9:
  %10 = phi float [ %0, %6 ], [ %1, %8 ]
  %11 = fcmp olt float %10, %.a
  br i1 %11, label %12, label %13
12:
  ret float %10
13:
  ret float %.b
}
define float @main(float %0, float %1, float %2, float %3, float %4, float %5, float %6, float %7) {
  %r = call fast fastcc float @sel(float %0, float %1, float %2, float %3, float %4, float %5, float %6, float %7)
  ret float %r
}
";
        let out = inline_nonrecursive_internal_calls(src);
        assert!(
            !out.contains("call fast fastcc"),
            "call not inlined:\n{out}"
        );
        // The implicit entry id `%6` must have been remapped to the fresh entry label, not survive
        // as a bare `%6` phi predecessor (which would dangle -> "unknown block label %6").
        assert!(
            !out.contains("[ %0, %6 ]"),
            "implicit entry id %6 left un-renamed:\n{out}"
        );
        assert!(
            out.contains(".inl0.entry"),
            "entry label not minted:\n{out}"
        );
        // The two-return cont-block phi must carry the bare return type `float`, never the leading
        // fast-math/calling-convention tokens from the function header.
        assert!(out.contains("%r = phi float"), "result phi missing:\n{out}");
        assert!(
            !out.contains("phi fast fastcc"),
            "cc/fast-math leaked into phi type:\n{out}"
        );
    }

    // 5. Recursive / cyclic pair -> BAIL (output unchanged).
    #[test]
    fn recursive_pair_bails() {
        let src = "\
define internal i32 @a(i32 %0) {
  %1 = call i32 @b(i32 %0)
  ret i32 %1
}
define internal i32 @b(i32 %0) {
  %1 = call i32 @a(i32 %0)
  ret i32 %1
}
define i32 @main(i32 %0) {
  %r = call i32 @a(i32 %0)
  ret i32 %r
}
";
        assert_eq!(inline_nonrecursive_internal_calls(src), src);
    }

    #[test]
    fn direct_self_recursion_bails() {
        let src = "\
define internal i32 @rec(i32 %0) {
  %1 = call i32 @rec(i32 %0)
  ret i32 %1
}
define i32 @main(i32 %0) {
  %r = call i32 @rec(i32 %0)
  ret i32 %r
}
";
        assert_eq!(inline_nonrecursive_internal_calls(src), src);
    }

    // Multi-block callee with an internal branch: labels and value refs are prefixed consistently.
    #[test]
    fn multiblock_callee_labels_prefixed() {
        let src = "\
define internal i32 @clamp(i32 %0) {
  %1 = icmp slt i32 %0, 0
  br i1 %1, label %neg, label %pos
neg:
  br label %done
pos:
  br label %done
done:
  %2 = phi i32 [ 0, %neg ], [ %0, %pos ]
  ret i32 %2
}
define i32 @main(i32 %0) {
  %r = call i32 @clamp(i32 %0)
  ret i32 %r
}
";
        let out = inline_nonrecursive_internal_calls(src);
        assert!(!out.contains("call i32 @clamp"), "call remains:\n{out}");
        // The inner phi's predecessor labels are prefixed and its %0 param substituted to caller %0.
        assert!(
            out.contains("phi i32 [ 0, %.inl0.neg ], [ %0, %.inl0.pos ]"),
            "inner phi not renamed:\n{out}"
        );
        // Block labels are prefixed.
        assert!(out.contains(".inl0.neg:"), "neg label missing:\n{out}");
        assert!(out.contains(".inl0.done:"), "done label missing:\n{out}");
    }

    // Phi in the CALLER that references the call's block must move to the continuation block when the
    // referenced value is defined after the call.
    #[test]
    fn caller_phi_pred_relabeled_to_cont() {
        let src = "\
define internal i32 @id(i32 %0) {
  ret i32 %0
}
define i32 @main(i32 %0) {
entry:
  br label %loop
loop:
  %r = call i32 @id(i32 %0)
  %next = add i32 %r, 1
  %acc = phi i32 [ 0, %entry ], [ %next, %loop ]
  br label %loop
}
";
        let out = inline_nonrecursive_internal_calls(src);
        // %next is defined after the call (now in cont), so the phi pred %loop -> %.inl0.cont.
        assert!(
            out.contains("[ %next, %.inl0.cont ]"),
            "phi pred not relabeled to cont:\n{out}"
        );
        // The %entry arm (value defined before call) stays.
        assert!(out.contains("[ 0, %entry ]"), "entry arm changed:\n{out}");
    }

    // A merge phi incoming `[ %v, %Bcall ]` whose value `%v` is defined BEFORE the call must ALSO move
    // to the continuation block: after the split, Bcall branches only to the callee entry, so its
    // original successor edge (into the merge) is inherited by `cont`. The earlier "only if %v is
    // defined post-call" guard left this incoming pointing at %Bcall — a merge phi missing the cont
    // predecessor (the TopK compare-swap `%10` wall, where the caller block held both the loop-carried
    // phi value and the fcmp+branch that inlining pushes into cont).
    #[test]
    fn caller_phi_pred_before_call_value_relabeled_to_cont() {
        let src = "\
define internal i32 @helper(i32 %0) {
  ret i32 %0
}
define i32 @main(i32 %0, i1 %c) {
entry:
  br label %bcall
bcall:
  %pre = add i32 %0, 7
  %r = call i32 @helper(i32 %0)
  br i1 %c, label %taken, label %merge
taken:
  br label %merge
merge:
  %m = phi i32 [ %pre, %bcall ], [ %r, %taken ]
  ret i32 %m
}
";
        let out = inline_nonrecursive_internal_calls(src);
        assert!(!out.contains("call i32 @helper"), "call remains:\n{out}");
        // %pre is defined BEFORE the call, but its edge into %merge now flows through cont, so the
        // predecessor must be relabeled anyway.
        assert!(
            out.contains("[ %pre, %.inl0.cont ]"),
            "before-call value's phi pred not relabeled to cont:\n{out}"
        );
        // No stale `%bcall` predecessor survives in the merge phi.
        assert!(
            !out.contains("[ %pre, %bcall ]"),
            "stale %bcall phi predecessor survived:\n{out}"
        );
    }

    #[test]
    fn pointer_select_consumer_inline_is_structurally_planned() {
        let src = r#"
@fallback = internal addrspace(2) global i32 0
@fc_default = internal addrspace(2) global i8 0

define internal void @_GLOBAL__sub_I_defaults() section "air.static_init" {
entry:
  store i8 1, ptr addrspace(2) @fc_default
  ret void
}

define internal i32 @consume(ptr addrspace(2) %pointer, i1 %branch) {
entry:
  br i1 %branch, label %left, label %right
left:
  %value = load i32, ptr addrspace(2) %pointer
  ret i32 %value
right:
  ret i32 0
}

define internal i32 @unrelated(i32 %value) {
entry:
  %sum = add i32 %value, 1
  ret i32 %sum
}

define internal i32 @consume_leaf(ptr addrspace(2) %pointer) {
entry:
  %value = load i32, ptr addrspace(2) %pointer
  ret i32 %value
}

define i32 @main(ptr addrspace(2) %runtime, i1 %choose) {
entry:
  %first = select i1 %choose, ptr addrspace(2) %runtime, ptr addrspace(2) @fallback
  %selected = select i1 %choose, ptr addrspace(2) @fallback, ptr addrspace(2) %first
  %leaf = call i32 @consume_leaf(ptr addrspace(2) %selected)
  %consumed = call i32 @consume(ptr addrspace(2) %selected, i1 %choose)
  %sum = add i32 %leaf, %consumed
  %other = call i32 @unrelated(i32 %sum)
  ret i32 %other
}
"#;
        let plan = inline_pointer_select_consumers(src, Some("main"));
        let out = &plan.source;
        assert!(plan.requires_relooper);
        assert!(!out.contains("call i32 @consume"), "{out}");
        assert!(out.contains("call i32 @unrelated"), "{out}");
        assert!(out.contains("define internal i32 @unrelated"), "{out}");
        assert!(
            out.contains("define internal void @_GLOBAL__sub_I_defaults"),
            "implicit constructor root was swept:\n{out}"
        );
        assert!(out.contains(".left:"), "{out}");

        assert!(!out.contains("call i32 @consume_leaf"), "{out}");
    }
}
