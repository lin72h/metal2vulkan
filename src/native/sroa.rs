//! Pre-emit AIR normalization: promote entry-stored, non-escaping `Function` allocas and fold
//! `insertvalue`/`extractvalue` aggregate round-trips.
//!
//! The MPS NDArray multi-destination kernels (TopK/pooling) build a small `alloca [N x ptr
//! addrspace(1)]` of their device buffer args, take its element-0 pointer, store that pointer into a
//! Function-local `MPSNDArrays` staging struct, read the struct back field-by-field, rebuild it as a
//! by-value aggregate, and pass it to an inlined helper that reads the array_ref at a STATIC index.
//! In Logical SPIR-V a pointer cannot live in memory-as-a-value (the emitter models a memory-stored
//! pointer as its i64 address) and a struct cannot carry a pointer member — so `store ptr %14, ptr
//! %slot` (a Function pointer that has no device address) bails `missing pointer storage`.
//!
//! Once the helper chain is inlined (see [`super::inline`]) every array_ref access is at a constant
//! index inside ONE function, so the whole staging is removable by classic scalar-replacement:
//!   1. store-to-load forwarding for a non-escaping alloca whose slot is stored exactly once in the
//!      entry block (that store dominates every load, so the load resolves to the stored value);
//!   2. `extractvalue (insertvalue AGG, V, PATH), PATH` -> `V` (aggregate round-trip fold);
//!      iterated to a fixpoint (forwarding the struct slot feeds the insertvalue, whose extractvalue then
//!      folds to the array pointer, which makes the array-slot load forwardable). The result is the
//!      "device pointer used directly" shape the emitter accepts and spirv-val validates.
//!
//! This runs on the sanitized AIR TEXT before [`super::LlModule::parse`]. It is **byte-neutral by
//! construction**: it only forwards a load to the UNIQUE value that provably reaches it (single
//! entry-block store, no aliasing store to the same slot) and only folds an aggregate extract to the
//! value that was inserted at exactly that path. It is **floor-safe**: it removes only stores/allocas
//! it has fully promoted, and bails (leaves the alloca untouched) on any escape, non-constant index,
//! type mismatch, or multi-store slot. It decides purely from IR structure, never a name.

use super::lex::split_top_level;
use std::collections::HashMap;

/// Promote entry-stored non-escaping Function allocas and fold aggregate round-trips in every
/// function of `san_ll`, to a fixpoint. Returns the text unchanged when nothing is promotable.
pub(super) fn promote_entry_allocas_and_fold_aggregates(san_ll: &str) -> String {
    let lines: Vec<&str> = san_ll.lines().collect();
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if line.starts_with("define") && line.trim_end().ends_with('{') {
            let start = i + 1;
            let mut j = start;
            while j < lines.len() && lines[j] != "}" {
                j += 1;
            }
            out.push(line.to_string());
            let body: Vec<String> = lines[start..j].iter().map(|s| s.to_string()).collect();
            out.extend(rewrite_body_to_fixpoint(body));
            if j < lines.len() {
                out.push(lines[j].to_string());
            }
            i = j + 1;
        } else {
            out.push(line.to_string());
            i += 1;
        }
    }
    let mut result = out.join("\n");
    if san_ll.ends_with('\n') {
        result.push('\n');
    }
    result
}

fn rewrite_body_to_fixpoint(mut body: Vec<String>) -> Vec<String> {
    // Bounded fixpoint: each round is a strict simplification (removes >=1 instruction or folds a
    // value); cap iterations defensively so a pathological body can never loop forever.
    for _ in 0..64 {
        let (next, changed_a) = fold_aggregates_once(body.clone());
        let (next, changed_c) = dce_pure_once(next);
        let (next, changed_b) = promote_allocas_once(next);
        let (next, changed_d) = lower_pointer_array_index_once(next);
        body = next;
        if !changed_a && !changed_b && !changed_c && !changed_d {
            break;
        }
    }
    body
}

/// A parsed body line that defines an SSA value: `%name = opcode rest`.
struct Def {
    name: String,
    opcode: String,
    rest: String,
}

fn parse_def(line: &str) -> Option<Def> {
    let t = line.trim();
    let eq = t.find(" = ")?;
    let name = t[..eq].trim();
    if !name.starts_with('%') {
        return None;
    }
    let rhs = t[eq + 3..].trim();
    let opcode = rhs.split_whitespace().next()?;
    let rest = rhs[opcode.len()..].trim();
    Some(Def {
        name: name.to_string(),
        opcode: opcode.to_string(),
        rest: rest.to_string(),
    })
}

/// Substitute a single `%old` token with `new_tok` across a line, token-accurately (word boundary on
/// `[A-Za-z0-9_.]`, so `%1` never matches inside `%10`).
fn subst_value_token(line: &str, old: &str, new_tok: &str) -> String {
    debug_assert!(old.starts_with('%'));
    let bytes = line.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < line.len() {
        if line[i..].starts_with(old) {
            let after = i + old.len();
            let next_ok = after >= line.len() || {
                let c = bytes[after] as char;
                !(c.is_ascii_alphanumeric() || c == '_' || c == '.')
            };
            // also require the char BEFORE not be part of an identifier / not a '%' continuation
            let prev_ok = i == 0 || {
                let c = bytes[i - 1] as char;
                !(c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '%')
            };
            if next_ok && prev_ok {
                out.push_str(new_tok);
                i = after;
                continue;
            }
        }
        let ch = line[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Replace every occurrence of each `%old -> new` across all lines (used to retire a folded/forwarded
/// SSA value: its every use becomes the resolved token).
fn apply_substitutions(body: &mut [String], subs: &HashMap<String, String>) {
    if subs.is_empty() {
        return;
    }
    for line in body.iter_mut() {
        for (old, new_tok) in subs {
            if line.contains(old.as_str()) {
                *line = subst_value_token(line, old, new_tok);
            }
        }
    }
}

// ---- aggregate round-trip fold: extractvalue (insertvalue AGG, V, PATH), PATH -> V ----

/// Parse `insertvalue <ty> <agg>, <ty> <val>, <idx0>, <idx1>, ...` -> (agg_tok, val_tok, path).
fn parse_insertvalue(rest: &str) -> Option<(String, String, Vec<u32>)> {
    let parts = split_top_level(rest, ',');
    if parts.len() < 3 {
        return None;
    }
    let agg = last_token(parts[0].trim())?;
    let val = last_token(parts[1].trim())?;
    let mut path = Vec::new();
    for p in &parts[2..] {
        path.push(p.trim().parse::<u32>().ok()?);
    }
    Some((agg, val, path))
}

/// Parse `extractvalue <ty> <agg>, <idx0>, <idx1>, ...` -> (agg_tok, path).
fn parse_extractvalue(rest: &str) -> Option<(String, Vec<u32>)> {
    let parts = split_top_level(rest, ',');
    if parts.len() < 2 {
        return None;
    }
    let agg = last_token(parts[0].trim())?;
    let mut path = Vec::new();
    for p in &parts[1..] {
        path.push(p.trim().parse::<u32>().ok()?);
    }
    Some((agg, path))
}

/// The final whitespace-separated token of `s` (the operand value: `<ty> %v` -> `%v`, or a literal).
fn last_token(s: &str) -> Option<String> {
    s.split_whitespace().last().map(|t| t.to_string())
}

/// One pass: for each `%r = extractvalue %agg, PATH` where `%agg` traces (through a chain of
/// `insertvalue`) to an `insertvalue ... V, PATH` at the SAME path, replace uses of `%r` with `V`.
/// Only fires when the walk reaches a definite inserted value at exactly PATH (never a partial /
/// aliasing overwrite ambiguity: a disjoint-path insertvalue is transparently skipped, a same-path
/// insert is the answer, and anything else — a phi/select/poison base with no matching insert — bails
/// for that extract).
fn fold_aggregates_once(mut body: Vec<String>) -> (Vec<String>, bool) {
    // Map agg SSA name -> its defining insertvalue (val, path, parent-agg).
    let mut inserts: HashMap<String, (String, Vec<u32>, String)> = HashMap::new();
    // Map extracted SSA name -> (source aggregate, path). Chaining these paths lets an extraction
    // through an intermediate sub-aggregate reach the original insertvalue, e.g. outer[0][0].
    let mut extracts: HashMap<String, (String, Vec<u32>)> = HashMap::new();
    for line in &body {
        if let Some(def) = parse_def(line) {
            if def.opcode == "insertvalue" {
                if let Some((agg, val, path)) = parse_insertvalue(&def.rest) {
                    inserts.insert(def.name.clone(), (val, path, agg));
                }
            } else if def.opcode == "extractvalue" {
                if let Some((agg, path)) = parse_extractvalue(&def.rest) {
                    extracts.insert(def.name.clone(), (agg, path));
                }
            }
        }
    }
    let mut subs: HashMap<String, String> = HashMap::new();
    for line in &body {
        let Some(def) = parse_def(line) else { continue };
        if def.opcode != "extractvalue" {
            continue;
        }
        let Some((agg, path)) = parse_extractvalue(&def.rest) else {
            continue;
        };
        let (agg, path) = resolve_extract_root(agg, path, &extracts);
        if let Some(val) = resolve_aggregate_path(&agg, &path, &inserts) {
            // Only substitute a concrete value token (a `%name` or literal), never poison/undef.
            if val != "poison" && val != "undef" && val != "zeroinitializer" {
                subs.insert(def.name.clone(), val);
            }
        }
    }
    if subs.is_empty() {
        return (body, false);
    }
    apply_substitutions(&mut body, &subs);
    // Drop the now-dead extractvalue defs whose result was fully substituted away.
    body.retain(|line| {
        if let Some(def) = parse_def(line) {
            if def.opcode == "extractvalue" && subs.contains_key(&def.name) {
                return false;
            }
        }
        true
    });
    (body, true)
}

/// Collapse `%inner = extractvalue ROOT, A; extractvalue %inner, B` into the structural lookup
/// `ROOT, A+B`. No IR is rewritten here; this only exposes the full constant path to the existing
/// exact insertvalue resolver. The bounded walk is defensive against malformed cyclic SSA.
fn resolve_extract_root(
    mut agg: String,
    mut path: Vec<u32>,
    extracts: &HashMap<String, (String, Vec<u32>)>,
) -> (String, Vec<u32>) {
    for _ in 0..256 {
        let Some((parent, prefix)) = extracts.get(&agg) else {
            break;
        };
        let mut combined = Vec::with_capacity(prefix.len() + path.len());
        combined.extend_from_slice(prefix);
        combined.extend_from_slice(&path);
        agg = parent.clone();
        path = combined;
    }
    (agg, path)
}

/// Walk the insertvalue chain from `agg` looking for an insert at exactly `path`. Returns the
/// inserted value token, or `None` if the chain hits a base with no matching insert (poison/undef/a
/// non-insertvalue root) — in which case folding is not proven and we leave the extract alone.
fn resolve_aggregate_path(
    agg: &str,
    path: &[u32],
    inserts: &HashMap<String, (String, Vec<u32>, String)>,
) -> Option<String> {
    let mut cur = agg.to_string();
    for _ in 0..256 {
        let (val, ipath, parent) = inserts.get(&cur)?;
        if ipath == path {
            return Some(val.clone());
        }
        // A prefix relationship makes the fold ambiguous (the whole sub-aggregate was replaced);
        // bail conservatively rather than fold wrongly.
        if is_prefix(ipath, path) || is_prefix(path, ipath) {
            return None;
        }
        cur = parent.clone();
    }
    None
}

fn is_prefix(a: &[u32], b: &[u32]) -> bool {
    a.len() < b.len() && b[..a.len()] == *a
}

/// Remove unused PURE defs (`insertvalue`/`extractvalue`/`getelementptr`/`bitcast`) so a dead
/// staging chain (e.g. a now-consumed aggregate) can no longer hold a family pointer live and block
/// alloca promotion. Only these four opcodes are swept — they have no side effects and their removal
/// is byte-neutral; loads/stores/calls are never touched. A fully-UNUSED `alloca` (its name appears
/// in no operand — every store/gep/load of it was already forwarded away) is also swept: an
/// unreferenced stack slot is dead and its removal changes no observable byte.
fn dce_pure_once(mut body: Vec<String>) -> (Vec<String>, bool) {
    const PURE: [&str; 4] = ["insertvalue", "extractvalue", "getelementptr", "bitcast"];
    // Count how many times each `%name` token is used as an OPERAND (i.e. anywhere except as the LHS
    // result of its own def). A def is dead iff its name is used zero times as an operand.
    let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
    for line in &body {
        let operand_text = match line.find(" = ") {
            Some(eq) => &line[eq + 3..],
            None => line.as_str(),
        };
        collect_value_tokens(operand_text, &mut used);
    }
    let before = body.len();
    body.retain(|line| {
        if let Some(def) = parse_def(line) {
            let dead = !used.contains(&def.name);
            if dead && (PURE.contains(&def.opcode.as_str()) || def.opcode == "alloca") {
                return false;
            }
        }
        true
    });
    let changed = body.len() != before;
    (body, changed)
}

/// Collect every `%name` token appearing in `s`.
fn collect_value_tokens(s: &str, out: &mut std::collections::HashSet<String>) {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < s.len() {
        if bytes[i] == b'%' {
            let start = i;
            i += 1;
            while i < s.len() {
                let c = bytes[i] as char;
                if c.is_ascii_alphanumeric() || c == '_' || c == '.' {
                    i += 1;
                } else {
                    break;
                }
            }
            if i > start + 1 {
                out.insert(s[start..i].to_string());
            }
        } else {
            i += 1;
        }
    }
}

// ---- fully-initialized value-array with dynamic index -> select cascade ----

/// The element index a family pointer addresses within the array.
#[derive(Clone)]
enum ElemIdx {
    Const(u32),
    Dyn(String, String), // (index value token, index integer type e.g. `i64`)
}

/// One pass: for a non-escaping entry alloca `[N x ELEM]` fully initialized with known element values
/// V_0..V_{N-1} (each a unique entry store), rewrite every LOAD of an element to the known value —
/// directly for a constant index, or an `ELEM` `select` cascade `select(idx==0, V_0, select(idx==1,
/// V_1, ... V_{N-1}))` for a dynamic index. This is the honest lowering of a Function array of device
/// buffer pointers that a helper reads by a runtime source index (the MPS multi-source readers): the
/// select cascade over the known buffer args is exactly the `VariablePointersStorageBuffer` shape the
/// emitter accepts for a runtime buffer selection. Byte-exact (the cascade returns V_idx for every
/// in-range idx; an out-of-range idx is UB in the original OOB load, so the last-element default is
/// conformant). Returns whether anything changed.
fn lower_pointer_array_index_once(body: Vec<String>) -> (Vec<String>, bool) {
    let entry_end = body
        .iter()
        .position(|l| is_block_label(l))
        .unwrap_or(body.len());
    // Collect entry allocas of array type `[N x ELEM]`.
    let arrays: Vec<(String, u32, String)> = body[..entry_end]
        .iter()
        .filter_map(|line| {
            let def = parse_def(line)?;
            if def.opcode != "alloca" {
                return None;
            }
            let ty = def.rest.split(',').next().unwrap_or("").trim();
            let (n, elem) = parse_array_type(ty)?;
            Some((def.name, n, elem))
        })
        .collect();
    for (aname, n, elem) in &arrays {
        if let Some(nb) = try_lower_array(&body, entry_end, aname, *n, elem) {
            return (nb, true);
        }
    }
    (body, false)
}

/// Parse `[N x ELEM]` -> `(N, ELEM)`.
fn parse_array_type(ty: &str) -> Option<(u32, String)> {
    let t = ty.trim();
    let inner = t.strip_prefix('[')?.strip_suffix(']')?.trim();
    let (n, rest) = inner.split_once(" x ")?;
    Some((n.trim().parse::<u32>().ok()?, rest.trim().to_string()))
}

fn try_lower_array(
    body: &[String],
    entry_end: usize,
    aname: &str,
    n: u32,
    elem: &str,
) -> Option<Vec<String>> {
    // family pointer -> the element index it addresses.
    let mut elem_idx: HashMap<String, ElemIdx> = HashMap::new();
    // Discover element pointers to a fixpoint.
    let mut progressed = true;
    while progressed {
        progressed = false;
        for line in body {
            let Some(def) = parse_def(line) else { continue };
            if def.opcode != "getelementptr" || elem_idx.contains_key(&def.name) {
                continue;
            }
            let parts = split_top_level(&def.rest, ',');
            if parts.len() < 2 {
                continue;
            }
            let base = last_token(parts[1].trim())?;
            // Two shapes:
            //  A) `gep [N x ELEM], ptr %A, i64 0, i64 <idx>`  -> element <idx>
            //  B) `gep ELEM, ptr <elemptr>, i64 <idx>`         -> base_elem + <idx>
            if base == aname {
                // shape A: parts[2] = "i64 0", parts[3] = "i64 <idx>"
                if parts.len() == 4 && last_token(parts[2].trim())? == "0" {
                    let idx = parse_index_operand(parts[3].trim())?;
                    elem_idx.insert(def.name.clone(), idx);
                    progressed = true;
                }
            } else if let Some(base_idx) = elem_idx.get(&base).cloned() {
                // shape B: single index added to the base element.
                if parts.len() == 3 {
                    let add = parse_index_operand(parts[2].trim())?;
                    let combined = add_elem_idx(&base_idx, &add)?;
                    elem_idx.insert(def.name.clone(), combined);
                    progressed = true;
                }
            }
        }
    }
    // The alloca and every element pointer form the family. Now classify all uses.
    let mut family: std::collections::HashSet<String> = elem_idx.keys().cloned().collect();
    family.insert(aname.to_string());

    let mut vals: Vec<Option<String>> = vec![None; n as usize];
    let mut store_count: Vec<usize> = vec![0; n as usize];
    let mut load_rewrites: Vec<(usize, String, ElemIdx)> = Vec::new(); // (line idx, result, index)
    let mut dead_ptrs: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (li, line) in body.iter().enumerate() {
        let t = line.trim();
        // A block-label line (`.inl0.21:  ; preds = %13`) is control-flow structure, never a data
        // use. The inliner can leave a STALE `; preds =` comment naming a since-renamed value that
        // collides with a family element-pointer name (here the entry gep `%13`), so scanning the
        // label yields a false-positive escape. Labels never use an SSA value as data — skip them.
        if is_block_label(line) {
            continue;
        }
        if let Some(def) = parse_def(line) {
            if family.contains(&def.name)
                && (def.opcode == "alloca" || def.opcode == "getelementptr")
            {
                continue; // family's own defs
            }
            if def.opcode == "load" {
                let parts = split_top_level(&def.rest, ',');
                if let Some(ptr) = parts.get(1).and_then(|p| last_token(p.trim())) {
                    if let Some(idx) = elem_idx.get(&ptr) {
                        load_rewrites.push((li, def.name.clone(), idx.clone()));
                        continue;
                    }
                    if ptr == aname {
                        // load directly from the array base = element 0
                        load_rewrites.push((li, def.name.clone(), ElemIdx::Const(0)));
                        continue;
                    }
                }
            }
        }
        if let Some(inner) = t.strip_prefix("store ") {
            let parts = split_top_level(inner, ',');
            if parts.len() >= 2 {
                let val = last_token(parts[0].trim())?;
                let ptr = last_token(parts[1].trim())?;
                if let Some(ElemIdx::Const(c)) = elem_idx.get(&ptr) {
                    if (*c as usize) < vals.len() {
                        // A store in a non-entry block does not dominate all loads; a second store to
                        // the same element means the value isn't a compile-time constant — bail both.
                        if li >= entry_end {
                            return None;
                        }
                        store_count[*c as usize] += 1;
                        if store_count[*c as usize] > 1 {
                            return None;
                        }
                        vals[*c as usize] = Some(val.clone());
                        dead_ptrs.insert(ptr.clone());
                        continue;
                    }
                }
                if family.contains(&val) {
                    return None; // a family pointer stored as a value = escape
                }
                if family.contains(&ptr) {
                    return None; // store to a non-constant element = unresolved
                }
            }
            continue;
        }
        if (t.starts_with("call ") || t.starts_with("tail call ")) && is_ignorable_intrinsic_call(t)
        {
            continue;
        }
        // any other mention of a family pointer = escape (ignore trailing `; ...` comments, which
        // are not data uses and may carry stale value names from the inliner). For a `phi`, the
        // block-label operand of each `[ val, block ]` pair is control-flow, not a data use — and a
        // family element-pointer name (`%13`) can collide with an incoming BLOCK name; strip those
        // block operands so only the value operands are checked for a genuine pointer escape.
        let code = line.split(';').next().unwrap_or(line);
        let code = strip_phi_block_operands(code);
        if mentions_family(&code, &family) {
            return None;
        }
    }

    // Must be fully initialized and actually have loads to rewrite.
    if vals.iter().any(|v| v.is_none()) || load_rewrites.is_empty() {
        return None;
    }
    let vals: Vec<String> = vals.into_iter().map(|v| v.unwrap()).collect();

    // Reserve a fresh-name prefix; bail if it already appears (keeps names unique).
    let prefix = ".psa";
    if body.iter().any(|l| l.contains(prefix)) {
        return None;
    }

    // Build the new body: replace each load line with the resolved value / select cascade; drop the
    // element geps, the init stores, the alloca, and lifetime markers on the family.
    let mut counter = 0u32;
    let rewrite_map: HashMap<usize, (String, ElemIdx)> = load_rewrites
        .into_iter()
        .map(|(li, res, idx)| (li, (res, idx)))
        .collect();
    let mut out: Vec<String> = Vec::with_capacity(body.len());
    let mut load_subs: HashMap<String, String> = HashMap::new();
    for (li, line) in body.iter().enumerate() {
        if let Some((res, idx)) = rewrite_map.get(&li) {
            match idx {
                ElemIdx::Const(c) => {
                    // direct value substitution (record; applied after).
                    let v = vals.get(*c as usize)?;
                    load_subs.insert(res.clone(), v.clone());
                }
                ElemIdx::Dyn(tok, ity) => {
                    if vals.len() == 1 {
                        // a 1-element array: the index is always 0.
                        load_subs.insert(res.clone(), vals[0].clone());
                    } else {
                        // emit a select cascade defining `res`.
                        let lines =
                            emit_select_cascade(res, tok, ity, &vals, elem, prefix, &mut counter);
                        out.extend(lines);
                    }
                }
            }
            continue; // drop the original load line
        }
        // drop family element geps and the alloca
        if let Some(def) = parse_def(line) {
            if family.contains(&def.name)
                && (def.opcode == "alloca" || def.opcode == "getelementptr")
            {
                continue;
            }
        }
        // drop the init stores
        let t = line.trim();
        if let Some(inner) = t.strip_prefix("store ") {
            let parts = split_top_level(inner, ',');
            if let Some(ptr) = parts.get(1).and_then(|p| last_token(p.trim())) {
                if dead_ptrs.contains(&ptr) {
                    continue;
                }
            }
        }
        // drop lifetime markers referencing the family
        if (t.starts_with("call ") || t.starts_with("tail call "))
            && is_ignorable_intrinsic_call(t)
            && mentions_family(line, &family)
        {
            continue;
        }
        out.push(line.clone());
    }
    apply_substitutions(&mut out, &load_subs);
    Some(out)
}

/// Parse a gep index operand `i64 <idx>` / `i32 <idx>` -> its element index (const or a value token).
fn parse_index_operand(op: &str) -> Option<ElemIdx> {
    // `op` is `<intty> <idx>` (e.g. `i64 0` or `i64 %x`).
    let toks: Vec<&str> = op.split_whitespace().collect();
    let tok = toks.last()?.to_string();
    let ty = toks
        .get(toks.len().wrapping_sub(2))
        .unwrap_or(&"i64")
        .to_string();
    if let Ok(c) = tok.parse::<u32>() {
        Some(ElemIdx::Const(c))
    } else if tok.starts_with('%') {
        Some(ElemIdx::Dyn(tok, ty))
    } else {
        None
    }
}

fn add_elem_idx(base: &ElemIdx, add: &ElemIdx) -> Option<ElemIdx> {
    match (base, add) {
        (ElemIdx::Const(0), a) => Some(a.clone()),
        (ElemIdx::Const(b), ElemIdx::Const(a)) => Some(ElemIdx::Const(b + a)),
        _ => None, // dynamic base + offset: unsupported (never occurs for the elem-0 base shape)
    }
}

/// Emit the instructions for `res = select(idx==0, V_0, select(idx==1, V_1, ... V_{N-1}))`. Returns
/// the emitted lines and the final result token (== `res`).
fn emit_select_cascade(
    res: &str,
    idx_tok: &str,
    idx_ty: &str,
    vals: &[String],
    elem: &str,
    prefix: &str,
    counter: &mut u32,
) -> Vec<String> {
    let n = vals.len();
    // acc starts as the last value (the default for the largest index).
    let mut acc = vals[n - 1].clone();
    let mut lines: Vec<String> = Vec::new();
    // Build from second-to-last down to 0; the outermost select defines `res`.
    for i in (0..n - 1).rev() {
        let cmp = format!("{prefix}.c{}", *counter);
        *counter += 1;
        lines.push(format!("  {cmp} = icmp eq {idx_ty} {idx_tok}, {i}"));
        let sel_name = if i == 0 {
            res.to_string()
        } else {
            let s = format!("{prefix}.s{}", *counter);
            *counter += 1;
            s
        };
        lines.push(format!(
            "  {sel_name} = select i1 {cmp}, {elem} {}, {elem} {acc}",
            vals[i]
        ));
        acc = sel_name;
    }
    lines
}

// ---- entry-single-store alloca promotion (mem2reg for straight-line staging) ----

/// Per-line block dominance for a function body. Store-to-load forwarding is byte-exact only when the
/// unique store DOMINATES the load; the entry block trivially dominates everything, but an inlined
/// helper's alloca+store land in the helper's own entry block (after a label), which still dominates
/// all of the inlined body because that block has a single predecessor. This computes real block
/// dominators so those cases forward too, without ever forwarding a store that does not dominate.
struct BlockDoms {
    /// block index for each body line.
    line_block: Vec<usize>,
    /// `dom[b]` = the set of blocks that dominate block `b` (including `b`).
    dom: Vec<std::collections::HashSet<usize>>,
}

impl BlockDoms {
    fn build(body: &[String]) -> BlockDoms {
        // Blocks: the entry block runs from line 0 to the first label; each label begins a block.
        let mut block_start: Vec<usize> = vec![0];
        let mut label_to_block: HashMap<String, usize> = HashMap::new();
        for (i, line) in body.iter().enumerate() {
            if is_block_label(line) {
                let name = line.trim();
                let name = name.split(':').next().unwrap_or("").trim();
                // Branch targets are written `label %name`; store the map key WITH the `%` so lookups
                // against those tokens match (labels appear as `name:` but are referenced as `%name`).
                label_to_block.insert(format!("%{name}"), block_start.len());
                block_start.push(i);
            }
        }
        let nblocks = block_start.len();
        // Map each line to its block.
        let mut line_block = vec![0usize; body.len()];
        for b in 0..nblocks {
            let start = block_start[b];
            let end = if b + 1 < nblocks {
                block_start[b + 1]
            } else {
                body.len()
            };
            line_block[start..end].fill(b);
        }
        // Successors from each block's terminator (the last `br`/`switch` in the block).
        let mut succ: Vec<Vec<usize>> = vec![Vec::new(); nblocks];
        for b in 0..nblocks {
            let start = block_start[b];
            let end = if b + 1 < nblocks {
                block_start[b + 1]
            } else {
                body.len()
            };
            for l in (start..end).rev() {
                let t = body[l].trim();
                if t.starts_with("br ") || t.starts_with("switch ") {
                    // Collect `label %X` targets: a `%name` token immediately after a `label` keyword.
                    let toks: Vec<&str> = t.split_whitespace().collect();
                    for w in 0..toks.len() {
                        if toks[w].trim_end_matches(',') == "label" {
                            if let Some(name) = toks.get(w + 1) {
                                let name = name.trim_end_matches(',');
                                if let Some(&tb) = label_to_block.get(name) {
                                    if !succ[b].contains(&tb) {
                                        succ[b].push(tb);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }
                if t.starts_with("ret") || t.starts_with("unreachable") {
                    break;
                }
            }
        }
        // Predecessors.
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); nblocks];
        for (b, succ_b) in succ.iter().enumerate() {
            for &s in succ_b {
                preds[s].push(b);
            }
        }
        // Reachability from the entry (block 0) over the succ graph. Unreachable blocks must be
        // EXCLUDED from the dominator intersection: a naive `∩ over all preds` lets an unreachable
        // pred (whose dom set is just {itself}) poison a genuinely-reachable block, cascading the
        // whole graph to "unreachable" (the Cooper/Harvey/Kennedy caveat). Loads in unreachable
        // blocks are dead, so treating them conservatively costs nothing.
        let mut reachable = vec![false; nblocks];
        let mut stack = vec![0usize];
        reachable[0] = true;
        while let Some(b) = stack.pop() {
            for &s in &succ[b] {
                if !reachable[s] {
                    reachable[s] = true;
                    stack.push(s);
                }
            }
        }
        // Iterative dominators over REACHABLE blocks only: dom[0] = {0}; dom[b] = {b} ∪ (∩ dom[p]
        // for reachable preds p). Init each reachable non-entry block to the full reachable set.
        let reachable_set: std::collections::HashSet<usize> =
            (0..nblocks).filter(|&b| reachable[b]).collect();
        let mut dom: Vec<std::collections::HashSet<usize>> = vec![Default::default(); nblocks];
        for b in 0..nblocks {
            if reachable[b] {
                dom[b] = reachable_set.clone();
            }
        }
        dom[0] = std::collections::HashSet::from([0]);
        let mut changed = true;
        while changed {
            changed = false;
            for b in 1..nblocks {
                if !reachable[b] {
                    continue;
                }
                let mut new_dom: Option<std::collections::HashSet<usize>> = None;
                for &p in &preds[b] {
                    if !reachable[p] {
                        continue;
                    }
                    new_dom = Some(match new_dom {
                        None => dom[p].clone(),
                        Some(acc) => acc.intersection(&dom[p]).copied().collect(),
                    });
                }
                let mut new_dom = new_dom.unwrap_or_default();
                new_dom.insert(b);
                if new_dom != dom[b] {
                    dom[b] = new_dom;
                    changed = true;
                }
            }
        }
        BlockDoms { line_block, dom }
    }

    /// True if the store at `store_line` dominates the load at `load_line` (byte-exact forward).
    fn store_dominates_load(&self, store_line: usize, load_line: usize) -> bool {
        let sb = self.line_block[store_line];
        let lb = self.line_block[load_line];
        if sb == lb {
            return store_line < load_line;
        }
        // The function entry block (block 0) dominates every reachable block by definition; a load in
        // an unreachable block is dead, so forwarding it is harmless. This is the original entry-store
        // behavior and is always byte-safe, independent of any edge-parsing completeness.
        if sb == 0 {
            return true;
        }
        self.dom[lb].contains(&sb)
    }
}

/// One pass over the body: promote every non-escaping alloca whose accessed slots are each stored
/// exactly once by a store that DOMINATES its loads, forwarding matching loads and deleting the
/// promoted stores + the alloca. Returns whether anything changed. Allocas anywhere in the body are
/// considered (an inlined helper's entry alloca sits after a label), gated by real block dominance so
/// the forward stays byte-exact.
fn promote_allocas_once(body: Vec<String>) -> (Vec<String>, bool) {
    let doms = BlockDoms::build(&body);

    // Collect allocas anywhere in the body: `%A = alloca <ty>`.
    let mut allocas: Vec<(String, String)> = Vec::new(); // (name, pointee-type text)
    for line in &body {
        if let Some(def) = parse_def(line) {
            if def.opcode == "alloca" {
                let ty = def.rest.split(',').next().unwrap_or("").trim().to_string();
                allocas.push((def.name.clone(), ty));
            }
        }
    }
    if allocas.is_empty() {
        return (body, false);
    }

    // For each alloca, map its derived pointer geps -> normalized index-path. A gep whose base is the
    // alloca (or a gep-of-the-alloca) contributes to the slot key.
    for (aname, _ty) in &allocas {
        if let Some((subs, dead_defs)) = try_promote_alloca(&body, &doms, aname) {
            let mut nb = body.clone();
            // Remove the promoted dead lines (alloca, family geps, promoted stores, and the forwarded
            // load defs) BEFORE substituting — so a forwarded load's own def line is gone and the
            // substitution only rewrites its remaining USES (never renaming a surviving def LHS).
            nb.retain(|line| {
                if let Some(def) = parse_def(line) {
                    if dead_defs.contains(&def.name) {
                        return false;
                    }
                }
                // Drop the promoted stores (recorded by their pointer token key `store@<ptr>`) ...
                if is_dead_store(line, &dead_defs) {
                    return false;
                }
                // ... and any liveness/debug intrinsic call on a now-removed family pointer (else it
                // would reference an undefined id after the alloca/gep is gone).
                let t = line.trim();
                if (t.starts_with("call ") || t.starts_with("tail call "))
                    && is_ignorable_intrinsic_call(t)
                    && dead_defs
                        .iter()
                        .any(|d| !d.starts_with("store@") && mentions_name(line, d))
                {
                    return false;
                }
                true
            });
            apply_substitutions(&mut nb, &subs);
            return (nb, true);
        }
    }
    (body, false)
}

/// Attempt to promote one alloca. Returns (value substitutions, set of dead def/marker names) on
/// success, or None if the alloca escapes / has an unpromotable access.
fn try_promote_alloca(
    body: &[String],
    doms: &BlockDoms,
    aname: &str,
) -> Option<(HashMap<String, String>, std::collections::HashSet<String>)> {
    // gep result name -> normalized index-path text (relative to the alloca base).
    let mut gep_path: HashMap<String, String> = HashMap::new();
    // pointers that ARE the alloca or a gep of it (the "in-family" pointer set).
    let mut family: std::collections::HashSet<String> = std::collections::HashSet::new();
    family.insert(aname.to_string());

    // First, discover geps in family (iterate to a fixpoint over the whole body: a gep may derive
    // from an earlier gep).
    let mut progressed = true;
    while progressed {
        progressed = false;
        for line in body {
            let Some(def) = parse_def(line) else { continue };
            if def.opcode != "getelementptr" {
                continue;
            }
            if gep_path.contains_key(&def.name) {
                continue;
            }
            if let Some((base, idxpath)) = parse_gep_base_and_path(&def.rest) {
                if family.contains(&base) {
                    let base_path = if base == aname {
                        String::new()
                    } else {
                        gep_path.get(&base)?.clone()
                    };
                    let full = join_path(&base_path, &idxpath);
                    gep_path.insert(def.name.clone(), full);
                    family.insert(def.name.clone());
                    progressed = true;
                }
            }
        }
    }

    // Now classify EVERY use of a family pointer. Allowed: `store <ty> <val>, ptr <fam>` and
    // `%r = load <ty>, ptr <fam>`, and being the base of a family gep (already captured). Anything
    // else = escape -> bail.
    // slot index-path key -> (stored value, store count, store line-index, store ptr token). The
    // store line-index lets the forwarding gate check that the store DOMINATES each load (byte-exact).
    let mut slot_store: HashMap<String, (String, usize, usize, String)> = HashMap::new();
    let mut loads: Vec<(String, String, usize)> = Vec::new(); // (result, slotkey, line-index)
    let mut dead: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (li, line) in body.iter().enumerate() {
        let t = line.trim();
        // Skip the family pointers' own DEFINING lines (the alloca and its geps): the family name
        // appears there only as the LHS result, which is not a use/escape.
        if let Some(def) = parse_def(line) {
            if family.contains(&def.name)
                && (def.opcode == "alloca" || def.opcode == "getelementptr")
            {
                continue;
            }
        }
        // stores: `store <ty> <val>, ptr <ptrtok>[, align N]`
        if let Some(inner) = t.strip_prefix("store ") {
            let parts = split_top_level(inner, ',');
            if parts.len() < 2 {
                // malformed for us; if it mentions a family ptr, bail
                if mentions_family(line, &family) {
                    return None;
                }
                continue;
            }
            let val = last_token(parts[0].trim())?;
            let ptr = last_token(parts[1].trim())?;
            if family.contains(&ptr) {
                let key = slot_key(&ptr, aname, &gep_path);
                let entry =
                    slot_store
                        .entry(key.clone())
                        .or_insert((val.clone(), 0, li, ptr.clone()));
                entry.0 = val.clone();
                entry.1 += 1;
                entry.2 = li;
                entry.3 = ptr.clone();
            } else if family.contains(&val) {
                // storing a family POINTER as a value into other memory = escape
                return None;
            }
            continue;
        }
        // loads: `%r = load <ty>, ptr <ptrtok>[, align N]`
        if let Some(def) = parse_def(line) {
            if def.opcode == "load" {
                let parts = split_top_level(&def.rest, ',');
                if let Some(pp) = parts.get(1) {
                    if let Some(ptr) = last_token(pp.trim()) {
                        if family.contains(&ptr) {
                            let key = slot_key(&ptr, aname, &gep_path);
                            loads.push((def.name.clone(), key, li));
                            continue;
                        }
                    }
                }
                // a load whose pointer isn't family is fine
            }
            if def.opcode == "getelementptr" && family.contains(&def.name) {
                continue; // already accounted
            }
        }
        // Calls to liveness/debug intrinsics (`llvm.lifetime.start/end`, `llvm.dbg.*`,
        // `llvm.assume`, `llvm.invariant.*`) that take a family pointer do NOT escape it — they are
        // pure markers with no data effect. Mark such a call line dead (removed with the alloca).
        if (t.starts_with("call ") || t.starts_with("tail call ")) && is_ignorable_intrinsic_call(t)
        {
            // non-escaping marker; its line is dropped in retain (it mentions the dead family ptr).
            continue;
        }
        // any OTHER mention of a family pointer (call arg, ret, bitcast, ptrtoint, icmp, phi, select,
        // insertvalue of the pointer, etc.) = escape
        if mentions_family_as_operand(line, &family, aname) {
            return None;
        }
    }

    // PER-SLOT store-to-load forwarding: for each load whose slot has a UNIQUE entry-block store of
    // a type-matching value, forward the load to the stored value and (once every load of that slot
    // is forwarded) remove the store. Slots with a multi/non-entry store, or no store, are left
    // intact — we do NOT require the WHOLE alloca to be promotable (the MPS `MPSNDArrays` staging
    // struct has fields that are read uninitialized or stored via non-matching paths). The alloca and
    // its geps stay; `dce_pure_once` sweeps any that become unused. Because we already bailed on any
    // escape, no external code can have mutated a slot between its unique entry store and a load, so
    // the forward is byte-exact.
    let mut subs: HashMap<String, String> = HashMap::new();
    // Track, per slot, how many loads exist vs how many we forwarded (to know if the store is dead).
    let mut slot_loads: HashMap<String, usize> = HashMap::new();
    let mut slot_forwarded: HashMap<String, usize> = HashMap::new();
    for (_res, key, _li) in &loads {
        *slot_loads.entry(key.clone()).or_insert(0) += 1;
    }
    for (res, key, load_li) in &loads {
        if let Some((val, count, store_li, _ptr)) = slot_store.get(key) {
            if *count == 1 && doms.store_dominates_load(*store_li, *load_li) {
                subs.insert(res.clone(), val.clone());
                dead.insert(res.clone());
                *slot_forwarded.entry(key.clone()).or_insert(0) += 1;
            }
        }
    }
    if subs.is_empty() {
        return None; // nothing forwardable in this alloca
    }
    // A store is dead iff every load of its slot was forwarded (no remaining reader).
    for (key, (_val, count, _store_li, ptr)) in &slot_store {
        if *count == 1
            && slot_forwarded.get(key).copied().unwrap_or(0)
                == slot_loads.get(key).copied().unwrap_or(0)
            && slot_loads.get(key).copied().unwrap_or(0) > 0
        {
            dead.insert(store_marker(ptr));
        }
    }
    Some((subs, dead))
}

fn store_marker(ptr: &str) -> String {
    format!("store@{ptr}")
}

/// True if the call line is a pure liveness/debug marker intrinsic whose pointer argument does not
/// escape the pointee (removing it is byte-neutral once the pointee alloca is gone).
fn is_ignorable_intrinsic_call(t: &str) -> bool {
    t.contains("@llvm.lifetime.start")
        || t.contains("@llvm.lifetime.end")
        || t.contains("@llvm.dbg.")
        || t.contains("@llvm.assume")
        || t.contains("@llvm.invariant.start")
        || t.contains("@llvm.invariant.end")
}

/// True if `name` (a `%token`) appears as a token in `line`.
fn mentions_name(line: &str, name: &str) -> bool {
    subst_value_token(line, name, "\0") != *line
}

fn is_dead_store(line: &str, dead: &std::collections::HashSet<String>) -> bool {
    let t = line.trim();
    if !t.starts_with("store ") {
        return false;
    }
    let inner = &t[6..];
    let parts = split_top_level(inner, ',');
    if parts.len() < 2 {
        return false;
    }
    if let Some(ptr) = last_token(parts[1].trim()) {
        return dead.contains(&store_marker(&ptr));
    }
    false
}

fn slot_key(ptr: &str, aname: &str, gep_path: &HashMap<String, String>) -> String {
    if ptr == aname {
        String::new()
    } else {
        gep_path
            .get(ptr)
            .cloned()
            .unwrap_or_else(|| ptr.to_string())
    }
}

/// For a `phi` line, drop the block-label operand of every `[ val, block ]` pair (keeping `val`), so
/// a mention scan sees only the data operands. Non-phi lines are returned unchanged. The block token
/// is the second comma-separated operand inside each bracket pair and is never a data use.
fn strip_phi_block_operands(line: &str) -> String {
    let t = line.trim_start();
    let is_phi = t
        .split_once('=')
        .map(|(_, r)| r.trim_start().starts_with("phi "))
        .unwrap_or(false);
    if !is_phi {
        return line.to_string();
    }
    let mut out = String::with_capacity(line.len());
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            // find matching ']'
            let start = i;
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] != b']' {
                j += 1;
            }
            let inner = &line[start + 1..j.min(line.len())];
            // keep only the value operand (before the top-level comma).
            let val = split_top_level(inner, ',')
                .into_iter()
                .next()
                .unwrap_or_default();
            out.push('[');
            out.push_str(val.trim());
            out.push(']');
            i = (j + 1).min(bytes.len());
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

fn mentions_family(line: &str, family: &std::collections::HashSet<String>) -> bool {
    family
        .iter()
        .any(|f| subst_value_token(line, f, "\0") != *line)
}

/// True if `line` uses a family pointer in a position OTHER than a store-ptr / load-ptr / gep-base
/// (those are handled by the caller before this is consulted).
fn mentions_family_as_operand(
    line: &str,
    family: &std::collections::HashSet<String>,
    _aname: &str,
) -> bool {
    // The caller already `continue`d for store/load/gep of family pointers, so any remaining mention
    // is an escape.
    mentions_family(line, family)
}

fn is_block_label(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() || t.starts_with(';') {
        return false;
    }
    // `N:` or `%name:` optionally followed by a comment.
    let head = t.split_whitespace().next().unwrap_or("");
    head.ends_with(':') && !head.starts_with('"')
}

/// Parse `getelementptr [inbounds] <ty>, ptr <base>, <idx...>` -> (base_tok, index-path text).
/// The index-path text is the normalized comma-joined index operands (verbatim), used as a slot key.
fn parse_gep_base_and_path(rest: &str) -> Option<(String, String)> {
    // rest: `inbounds <ty>, ptr <base>, <idx0>, <idx1>, ...`
    let parts = split_top_level(rest, ',');
    if parts.len() < 2 {
        return None;
    }
    // parts[0] = `[inbounds] <ty>` (the aggregate type), parts[1] = `ptr <base>`.
    let base = last_token(parts[1].trim())?;
    let idxs: Vec<String> = parts[2..].iter().map(|p| p.trim().to_string()).collect();
    Some((base, idxs.join(", ")))
}

fn join_path(base: &str, add: &str) -> String {
    if base.is_empty() {
        add.to_string()
    } else if add.is_empty() {
        base.to_string()
    } else {
        format!("{base} | {add}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &str) -> String {
        promote_entry_allocas_and_fold_aggregates(src)
    }

    #[test]
    fn no_op_module_is_byte_identical() {
        let src = "define void @k() {\n  ret void\n}\n";
        assert_eq!(run(src), src);
    }

    #[test]
    fn extractvalue_of_insertvalue_folds() {
        let src = "\
define void @k(ptr addrspace(1) %0) {
  %a = insertvalue { ptr } poison, ptr %0, 0
  %b = extractvalue { ptr } %a, 0
  store ptr %b, ptr %0, align 8
  ret void
}
";
        let out = run(src);
        assert!(
            !out.contains("extractvalue"),
            "extract should fold away:\n{out}"
        );
        assert!(
            out.contains("store ptr %0, ptr %0"),
            "use of %b -> %0:\n{out}"
        );
    }

    #[test]
    fn nested_extractvalue_reaches_nested_insertvalue() {
        // Mirrors an opaque resource wrapper returned by a builder and destructured after its
        // consumer is inlined: the first extract produces the wrapper, the second its handle.
        let src = "\
define void @k(ptr addrspace(1) %texture, ptr addrspace(1) %out) {
  %bundle = insertvalue { { ptr addrspace(1) }, i32 } poison, ptr addrspace(1) %texture, 0, 0
  %wrapper = extractvalue { { ptr addrspace(1) }, i32 } %bundle, 0
  %handle = extractvalue { ptr addrspace(1) } %wrapper, 0
  store ptr addrspace(1) %handle, ptr addrspace(1) %out, align 8
  ret void
}
";
        let out = run(src);
        assert!(
            !out.contains("insertvalue"),
            "nested insert remains:\n{out}"
        );
        assert!(
            !out.contains("extractvalue"),
            "nested extract remains:\n{out}"
        );
        assert!(
            out.contains("store ptr addrspace(1) %texture, ptr addrspace(1) %out"),
            "handle was not forwarded:\n{out}"
        );
    }

    #[test]
    fn entry_single_store_load_forwards_and_removes_alloca() {
        let src = "\
define void @k(ptr addrspace(1) %0, ptr addrspace(1) %out) {
  %s = alloca [1 x ptr addrspace(1)], align 8
  %p = getelementptr inbounds [1 x ptr addrspace(1)], ptr %s, i64 0, i64 0
  store ptr addrspace(1) %0, ptr %p, align 8
  %q = getelementptr inbounds [1 x ptr addrspace(1)], ptr %s, i64 0, i64 0
  %v = load ptr addrspace(1), ptr %q, align 8
  store ptr addrspace(1) %v, ptr %out, align 8
  ret void
}
";
        let out = run(src);
        assert!(!out.contains("alloca"), "alloca promoted away:\n{out}");
        assert!(!out.contains("= load"), "load forwarded:\n{out}");
        assert!(
            out.contains("store ptr addrspace(1) %0, ptr %out"),
            "forwarded %0:\n{out}"
        );
    }

    #[test]
    fn struct_then_array_fixpoint_resolves_pointer() {
        // Mirrors the TopK post-inline shape: array-of-ptr staged through a struct + by-value agg.
        let src = "\
define void @k(ptr addrspace(1) %0, ptr addrspace(1) %out) {
  %arr = alloca [1 x ptr addrspace(1)], align 8
  %e0 = getelementptr inbounds [1 x ptr addrspace(1)], ptr %arr, i64 0, i64 0
  store ptr addrspace(1) %0, ptr %e0, align 8
  %ap = getelementptr inbounds [1 x ptr addrspace(1)], ptr %arr, i64 0, i64 0
  %st = alloca { ptr }, align 8
  %sp = getelementptr inbounds { ptr }, ptr %st, i64 0, i32 0
  store ptr %ap, ptr %sp, align 8
  %spl = getelementptr inbounds { ptr }, ptr %st, i64 0, i32 0
  %ld = load ptr, ptr %spl, align 8
  %agg = insertvalue { ptr } poison, ptr %ld, 0
  %m = extractvalue { ptr } %agg, 0
  %dev = load ptr addrspace(1), ptr %m, align 8
  %ev = getelementptr inbounds float, ptr addrspace(1) %dev, i64 0
  %val = load float, ptr addrspace(1) %ev, align 4
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}
";
        let out = run(src);
        // After the fixpoint, %dev must resolve to %0 (the device buffer), no alloca/insertvalue left.
        assert!(!out.contains("alloca"), "all allocas promoted:\n{out}");
        assert!(!out.contains("insertvalue"), "aggregate folded:\n{out}");
        assert!(!out.contains("extractvalue"), "extract folded:\n{out}");
        assert!(
            out.contains("getelementptr inbounds float, ptr addrspace(1) %0"),
            "device ptr resolved to %0:\n{out}"
        );
    }

    #[test]
    fn escaping_alloca_is_left_untouched() {
        // The array pointer is stored into another buffer (escape) -> must NOT promote.
        let src = "\
define void @k(ptr addrspace(1) %0, ptr %esc) {
  %arr = alloca [1 x ptr addrspace(1)], align 8
  %e0 = getelementptr inbounds [1 x ptr addrspace(1)], ptr %arr, i64 0, i64 0
  store ptr addrspace(1) %0, ptr %e0, align 8
  store ptr %e0, ptr %esc, align 8
  ret void
}
";
        let out = run(src);
        assert!(out.contains("alloca"), "escaping alloca kept:\n{out}");
    }

    #[test]
    fn multi_store_slot_is_not_forwarded() {
        let src = "\
define void @k(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %out) {
  %s = alloca [1 x ptr addrspace(1)], align 8
  %p = getelementptr inbounds [1 x ptr addrspace(1)], ptr %s, i64 0, i64 0
  store ptr addrspace(1) %0, ptr %p, align 8
  store ptr addrspace(1) %1, ptr %p, align 8
  %v = load ptr addrspace(1), ptr %p, align 8
  store ptr addrspace(1) %v, ptr %out, align 8
  ret void
}
";
        let out = run(src);
        assert!(
            out.contains("alloca"),
            "multi-store alloca not promoted:\n{out}"
        );
    }

    #[test]
    fn subst_token_respects_word_boundary() {
        let l = "%10 = add i32 %1, %11";
        assert_eq!(subst_value_token(l, "%1", "%99"), "%10 = add i32 %99, %11");
    }

    #[test]
    fn non_entry_dominating_store_forwards() {
        // The cf9ad06f shape: an inlined helper's alloca+store land in the helper's OWN entry block
        // (after a label, single predecessor), and the load is in a further-dominated block. The
        // store dominates the load, so it must forward even though it is not in the function entry.
        let src = "\
define void @k(ptr addrspace(1) %tex, ptr addrspace(1) %out) {
  br label %h
h:
  %s = alloca { ptr addrspace(1) }, align 8
  %p = getelementptr inbounds { ptr addrspace(1) }, ptr %s, i64 0, i32 0
  store ptr addrspace(1) %tex, ptr %p, align 8
  br label %use
use:
  %pp = getelementptr inbounds { ptr addrspace(1) }, ptr %s, i64 0, i32 0
  %dev = load ptr addrspace(1), ptr %pp, align 8
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %dev, i32 0)
  ret void
}
";
        let out = run(src);
        assert!(
            !out.contains("= alloca") || !out.contains("store ptr addrspace(1) %tex"),
            "non-entry dominating store should forward + drop:\n{out}"
        );
        assert!(
            out.contains("@air.get_width_texture_2d(ptr addrspace(1) %tex"),
            "load must resolve to %tex:\n{out}"
        );
    }

    #[test]
    fn non_dominating_store_is_not_forwarded() {
        // The unique store is guarded by a branch, so it does NOT dominate the merge-block load
        // (the load can execute on the path that skips the store). Forwarding would be byte-WRONG,
        // so the dominance gate must leave the alloca/store/load intact.
        let src = "\
define void @k(ptr addrspace(1) %tex, i1 %c, ptr addrspace(1) %out) {
  %s = alloca { ptr addrspace(1) }, align 8
  br i1 %c, label %set, label %merge
set:
  %p = getelementptr inbounds { ptr addrspace(1) }, ptr %s, i64 0, i32 0
  store ptr addrspace(1) %tex, ptr %p, align 8
  br label %merge
merge:
  %pp = getelementptr inbounds { ptr addrspace(1) }, ptr %s, i64 0, i32 0
  %dev = load ptr addrspace(1), ptr %pp, align 8
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %dev, i32 0)
  ret void
}
";
        let out = run(src);
        assert!(
            out.contains("load ptr addrspace(1), ptr %pp"),
            "non-dominating store must NOT be forwarded (byte-safety):\n{out}"
        );
    }
}
