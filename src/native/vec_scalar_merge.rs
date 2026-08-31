//! Pre-emit AIR normalization: scalarize a pointer-merge that mixes a `<N x T>*` arm with a `T*`
//! arm of the SAME element type T.
//!
//! A handful of MPS kernels (`MPSRNNBreakUpToOutputVecs`) walk one device buffer two ways inside a
//! single loop nest: a vectorized body that loads/stores `<4 x float>` at a time and a scalar
//! remainder that loads/stores one `float` at a time. The two walks are joined by `phi`/`select`
//! over `ptr addrspace(1)`, so the merged Logical pointer has no single pointee width — the native
//! emitter's `pointer_merge_meta` rejects it with `pointer merge pointee mismatch Float vs
//! Vector(Float, 4)` and produces NO module (so no post-emit byte pass can reach it). The raw
//! all-buffers-raw retry does not help either: the conflicting merge is over DERIVED pointers, not
//! buffer roots.
//!
//! The byte-correct fix is to pick the SCALAR representation for the whole merge component: a
//! `<N x T>` load is decomposed into N consecutive scalar `T` loads + a vector rebuild (same N*4
//! bytes, same values), a `<N x T>` store into N scalar stores, and every `<N x T>`-strided
//! `getelementptr` in the component into the equivalent scalar `getelementptr T` with the index
//! multiplied by N (the same byte address). After the rewrite every arm of every merge is `T*`, so
//! the merge is a legal Logical pointer select.
//!
//! This runs on the sanitized AIR TEXT before [`super::LlModule::parse`], so every derived pointee
//! map is recomputed from the rewritten text. It is **floor-safe by construction**: it fires only on
//! a function that contains a `phi`/`select` mixing a scalar and a same-element vector pointee — a
//! shape the emitter currently rejects outright, so no module that emits today is altered. It
//! decides purely from IR structure (pointer flow + gep/load/store element shapes), never a name.

use super::lex::split_top_level;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};

/// Rewrite every function in `san_ll` that carries a scalar/vector pointer-merge into its scalarized
/// form. Returns the text unchanged when no such merge exists (the common case).
///
/// KEYSTONE-1: the scalarization is ALSO seeded from whole-vs-part USE WIDTH — a pointer network
/// dereferenced at both scalar `S` and vector `<N x S>` of one element S — not only from the def-flow
/// scalar/vector merge shape. That seeding touches components that emit today, so it is byte-changing:
/// its output is baked into the BC byte baseline (the ~180 golden cases the 2026-07-09 MoltenVK A/B
/// validated), i.e. on the primary path this pass is BC-COVERED. The def-flow path (a shape the
/// emitter rejects) is always on and floor-safe. The `widen` seeding is threaded to `lower_impl` as an
/// explicit parameter (always `true` in production) so unit tests can exercise both modes.
pub(super) fn lower_vector_scalar_pointer_merge(san_ll: &str) -> Cow<'_, str> {
    lower_impl(san_ll, true)
}

/// Core rewrite with the whole-vs-part use-width seeding threaded as an explicit `widen` parameter so
/// unit tests exercise both modes without racing on the process-global env var.
fn lower_impl(san_ll: &str, widen: bool) -> Cow<'_, str> {
    let lines: Vec<&str> = san_ll.lines().collect();
    // Stay allocation-free at the source-text level until the first function that actually needs a
    // rewrite. Large generated modules commonly contain none of these pointer networks; eagerly
    // cloning every line in that case leaves hundreds of thousands of freed small allocations in the
    // system allocator immediately before the typed parser reaches its own peak.
    let mut out: Option<Vec<String>> = None;
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if line.starts_with("define") && line.trim_end().ends_with('{') {
            let start = i + 1;
            let mut j = start;
            while j < lines.len() && lines[j] != "}" {
                j += 1;
            }
            if let Some(rewritten) = rewrite_function_body(&lines[start..j], widen) {
                let output = out.get_or_insert_with(|| {
                    let mut prefix = Vec::with_capacity(lines.len());
                    prefix.extend(lines[..i].iter().map(|line| (*line).to_string()));
                    prefix
                });
                output.push(line.to_string());
                output.extend(rewritten);
                if j < lines.len() {
                    output.push(lines[j].to_string());
                }
            } else if let Some(output) = &mut out {
                output.extend(
                    lines[i..j.min(lines.len() - 1) + 1]
                        .iter()
                        .map(|line| (*line).to_string()),
                );
            }
            i = j + 1;
        } else {
            if let Some(output) = &mut out {
                output.push(line.to_string());
            }
            i += 1;
        }
    }
    let Some(out) = out else {
        return Cow::Borrowed(san_ll);
    };
    let mut result = out.join("\n");
    if san_ll.ends_with('\n') {
        result.push('\n');
    }
    Cow::Owned(result)
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum Pointee {
    Scalar(String),
    Vec(String, usize),
    Unknown,
}

/// A parsed body line that defines an SSA value: `%name = opcode rest`.
struct Def<'a> {
    name: &'a str,
    opcode: &'a str,
    rest: &'a str,
}

fn parse_def(line: &str) -> Option<Def<'_>> {
    let line = line.trim();
    let eq = line.find(" = ")?;
    let name = line[..eq].trim();
    if !name.starts_with('%') {
        return None;
    }
    let rhs = line[eq + 3..].trim();
    let opcode = rhs.split_whitespace().next()?;
    let rest = rhs[opcode.len()..].trim();
    Some(Def { name, opcode, rest })
}

/// Parse `<N x ELEM>` into `(ELEM, N)`; `None` for any non-vector type.
fn vec_elem_lanes(ty: &str) -> Option<(String, usize)> {
    let ty = ty.trim();
    let inner = ty.strip_prefix('<')?.strip_suffix('>')?.trim();
    let (n, elem) = inner.split_once(" x ")?;
    let lanes: usize = n.trim().parse().ok()?;
    let elem = elem.trim();
    if !is_simple_scalar(elem) {
        return None;
    }
    Some((elem.to_string(), lanes))
}

/// A simple 32-bit-ish scalar type token (`float`, `half`, `i32`, …) — no pointers, vectors, or
/// aggregates. Splitting a vector load/store of these into element ops is a pure byte-decomposition.
fn is_simple_scalar(ty: &str) -> bool {
    !ty.is_empty()
        && ty.chars().all(|c| c.is_ascii_alphanumeric())
        && ty.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
}

/// Split `"ptr addrspace(1) %144"` into the type prefix (`"ptr addrspace(1)"`) and the value
/// (`"%144"`). The prefix is reused verbatim so the rewritten gep/load/store keeps the exact storage
/// class the source used.
fn split_ptr_operand(chunk: &str) -> Option<(&str, &str)> {
    let chunk = chunk.trim();
    let name_start = chunk.rfind(char::is_whitespace)? + 1;
    let name = chunk[name_start..].trim();
    if !name.starts_with('%') {
        return None;
    }
    Some((chunk[..name_start].trim(), name))
}

/// `%name` value token at the end of a typed operand, e.g. `"<4 x float> %147"` → `"%147"`.
fn trailing_local(chunk: &str) -> Option<&str> {
    let tok = chunk.trim().rsplit(char::is_whitespace).next()?.trim();
    tok.starts_with('%').then_some(tok)
}

struct GepInfo<'a> {
    inbounds: bool,
    src_ty: &'a str,
    /// The base pointer operand chunk verbatim, e.g. `"ptr addrspace(1) %144"`.
    base_chunk: &'a str,
    base_name: &'a str,
    /// Each index operand verbatim, e.g. `"i64 %125"`.
    indices: Vec<&'a str>,
}

fn parse_gep(rest: &str) -> Option<GepInfo<'_>> {
    let (inbounds, rest) = match rest.strip_prefix("inbounds ") {
        Some(r) => (true, r.trim()),
        None => (false, rest.trim()),
    };
    let chunks = split_top_level(rest, ',');
    if chunks.len() < 2 {
        return None;
    }
    let (_, base_name) = split_ptr_operand(chunks[1])?;
    Some(GepInfo {
        inbounds,
        src_ty: chunks[0].trim(),
        base_chunk: chunks[1].trim(),
        base_name,
        indices: chunks[2..].to_vec(),
    })
}

/// The pointer-flow neighbours of a def (arms / bases / bitcast source), for component discovery.
fn pointer_neighbours(def: &Def) -> Vec<String> {
    match def.opcode {
        "getelementptr" => parse_gep(def.rest)
            .map(|g| vec![g.base_name.to_string()])
            .unwrap_or_default(),
        "bitcast" => bitcast_source(def.rest)
            .map(|s| vec![s.to_string()])
            .unwrap_or_default(),
        "phi" => phi_arm_values(def.rest),
        "select" => select_arm_values(def.rest),
        _ => Vec::new(),
    }
}

fn bitcast_source(rest: &str) -> Option<&str> {
    let (lhs, rhs) = rest.split_once(" to ")?;
    if !lhs.trim_start().starts_with("ptr") || !rhs.trim_start().starts_with("ptr") {
        return None;
    }
    trailing_local(lhs)
}

/// `[ %v, %label ]` groups → the value tokens (`%v`), local-only.
fn phi_arm_values(rest: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut search = rest;
    while let Some(open) = search.find('[') {
        let after = &search[open + 1..];
        let Some(close) = after.find(']') else { break };
        let inner = &after[..close];
        if let Some(val) = inner.split(',').next() {
            let val = val.trim();
            if val.starts_with('%') {
                out.push(val.to_string());
            }
        }
        search = &after[close + 1..];
    }
    out
}

fn select_arm_values(rest: &str) -> Vec<String> {
    let parts = split_top_level(rest, ',');
    if parts.len() != 3 {
        return Vec::new();
    }
    parts[1..]
        .iter()
        .filter(|p| p.trim_start().starts_with("ptr"))
        .filter_map(|p| trailing_local(p))
        .map(str::to_string)
        .collect()
}

/// Resolve every value's pointee shape to a fixpoint and collect the names of `phi`/`select` results
/// whose arms mix a scalar and a same-element vector pointee.
fn classify(defs: &HashMap<String, Def>) -> (HashMap<String, Pointee>, HashSet<String>) {
    let mut pointees: HashMap<String, Pointee> = HashMap::new();
    let mut mismatches: HashSet<String> = HashSet::new();
    // Bounded fixpoint: each pass can only refine Unknown → concrete, so it converges in at most one
    // pass per phi-chain depth; cap defensively at the def count.
    for _ in 0..=defs.len() {
        let mut changed = false;
        let mut round_mismatch = HashSet::new();
        for (name, def) in defs {
            let p = compute_pointee(def, &pointees, &mut round_mismatch, name);
            if pointees.get(name) != Some(&p) {
                pointees.insert(name.clone(), p);
                changed = true;
            }
        }
        mismatches = round_mismatch;
        if !changed {
            break;
        }
    }
    (pointees, mismatches)
}

fn compute_pointee(
    def: &Def,
    pointees: &HashMap<String, Pointee>,
    mismatches: &mut HashSet<String>,
    name: &str,
) -> Pointee {
    match def.opcode {
        "getelementptr" => {
            let Some(g) = parse_gep(def.rest) else {
                return Pointee::Unknown;
            };
            if let Some((elem, lanes)) = vec_elem_lanes(g.src_ty) {
                if g.indices.len() == 1 {
                    Pointee::Vec(elem, lanes)
                } else {
                    Pointee::Scalar(elem)
                }
            } else if is_simple_scalar(g.src_ty) {
                Pointee::Scalar(g.src_ty.to_string())
            } else {
                Pointee::Unknown
            }
        }
        "bitcast" => bitcast_source(def.rest)
            .and_then(|s| pointees.get(s).cloned())
            .unwrap_or(Pointee::Unknown),
        "phi" => merge_arms(&phi_arm_values(def.rest), pointees, mismatches, name),
        "select" => merge_arms(&select_arm_values(def.rest), pointees, mismatches, name),
        _ => Pointee::Unknown,
    }
}

/// Merge the pointees of merge arms. If a single element type appears as BOTH a scalar and a vector,
/// the merge is a scalar/vector mismatch — record it and unify to the scalar (the byte-narrowest
/// view). Mixed element types or inconsistent lanes resolve to Unknown (not our shape).
fn merge_arms(
    arms: &[String],
    pointees: &HashMap<String, Pointee>,
    mismatches: &mut HashSet<String>,
    name: &str,
) -> Pointee {
    let mut elems: HashSet<String> = HashSet::new();
    let mut saw_scalar = false;
    let mut vec_lanes: Option<usize> = None;
    let mut consistent_vec = true;
    for arm in arms {
        match pointees.get(arm) {
            Some(Pointee::Scalar(e)) => {
                elems.insert(e.clone());
                saw_scalar = true;
            }
            Some(Pointee::Vec(e, n)) => {
                elems.insert(e.clone());
                match vec_lanes {
                    Some(prev) if prev != *n => consistent_vec = false,
                    _ => vec_lanes = Some(*n),
                }
            }
            _ => {}
        }
    }
    if elems.len() != 1 || !consistent_vec {
        return Pointee::Unknown;
    }
    let elem = elems.into_iter().next().unwrap();
    match (saw_scalar, vec_lanes) {
        (true, Some(_)) => {
            mismatches.insert(name.to_string());
            Pointee::Scalar(elem)
        }
        (true, None) => Pointee::Scalar(elem),
        (false, Some(n)) => Pointee::Vec(elem, n),
        (false, None) => Pointee::Unknown,
    }
}

/// Undirected pointer-flow adjacency: every def linked to its arm/base/bitcast-source neighbours.
fn pointer_adjacency(defs: &HashMap<String, Def>) -> HashMap<String, Vec<String>> {
    let mut adj: HashMap<String, Vec<String>> = HashMap::new();
    for (name, def) in defs {
        for nb in pointer_neighbours(def) {
            adj.entry(name.clone()).or_default().push(nb.clone());
            adj.entry(nb).or_default().push(name.clone());
        }
    }
    adj
}

/// Undirected pointer-flow component reachable from the mismatch merges.
fn component(defs: &HashMap<String, Def>, seeds: &HashSet<String>) -> HashSet<String> {
    let adj = pointer_adjacency(defs);
    let mut seen: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<String> = seeds.iter().cloned().collect();
    for s in seeds {
        seen.insert(s.clone());
    }
    while let Some(cur) = queue.pop_front() {
        if let Some(nbs) = adj.get(&cur) {
            for nb in nbs {
                if seen.insert(nb.clone()) {
                    queue.push_back(nb.clone());
                }
            }
        }
    }
    seen
}

/// Per pointer, the distinct primitive-scalar deref widths observed on it across the body: which
/// element scalars it is loaded/stored/stepped at, and separately at scalar vs whole-vector width.
/// `other` flags a deref at a shape whole-vs-part widening does not model (a struct/array/nested-ptr
/// load or aggregate gep). `logical` flags a use through a NON-word-addressable pointer (any address
/// space other than device `addrspace(1)` / constant `addrspace(2)`) — a logical Workgroup/thread
/// pointer cannot be re-viewed at a finer scalar stride without an illegal logical-pointer bitcast
/// (the native emitter rejects "cannot reinterpret workgroup pointer arg … to raw word view"). Either
/// flag excludes the whole component (floor-safe: never widened).
#[derive(Default)]
struct UseWidths {
    scalars: HashSet<String>,
    vectors: HashSet<String>,
    other: bool,
    logical: bool,
}

/// Names of SSA values defined by a pointer-typed `load` — a pointer RELOADED from memory (typically
/// an `[N x ptr]` alloca the source spills derived pointers into and reads back). Such a value's
/// pointee type is pinned by whatever pointer values were STORED to that memory, which live in a
/// DIFFERENT SSA component the pointer-flow adjacency (gep/bitcast/phi/select only) cannot reach. A
/// whole-vs-part component that includes one of these is only HALF the real network — scalarizing its
/// SSA-visible accesses while the memory-stored siblings keep their vector typing is a partial retype
/// (`OpPtrAccessChain %float` on a `vec4*` base), exactly the dead-end #14/#15 defect. So any component
/// touching a memory-reloaded pointer is excluded from widening. This is the `MPSLSTMMultiInputKernelFloat`
/// shape that miscompiled the KEYSTONE-1 default flip.
fn memory_reloaded_pointers(body: &[&str]) -> HashSet<String> {
    let mut out = HashSet::new();
    for line in body {
        if let Some(def) = parse_def(line) {
            if def.opcode == "load" {
                let loaded_ty = split_top_level(def.rest, ',')
                    .first()
                    .map(|c| c.trim().to_string())
                    .unwrap_or_default();
                if loaded_ty.starts_with("ptr") {
                    out.insert(def.name.to_string());
                }
            }
        }
    }
    out
}

/// True iff a pointer-operand type prefix (`"ptr addrspace(1)"`, `"ptr"`, …) names a WORD-ADDRESSABLE
/// storage class — device (`addrspace(1)`) or constant (`addrspace(2)`), the only spaces the emitter
/// can re-view at a finer scalar stride via the raw byte-GEP model. Every other space (Workgroup
/// `addrspace(3)`, thread/private default, …) is logical and cannot be scalarized soundly.
fn is_word_addressable_ptr(prefix: &str) -> bool {
    prefix.contains("addrspace(1)") || prefix.contains("addrspace(2)")
}

/// Byte width of a primitive scalar LLVM/AIR type token (`i8`→1, `half`→2, `float`/`i32`→4, `i64`→8),
/// or `None` for anything not a recognized primitive scalar. Used by [`widen_targets`] to gate the
/// whole-vs-part scalarization on the target scalar being a native word (≥ 4 bytes): a sub-word target
/// forces a native 8/16-bit StorageBuffer descriptor view whose capability the executors never enable.
fn scalar_byte_width(elem: &str) -> Option<usize> {
    match elem {
        "i8" => Some(1),
        "i16" | "half" | "bfloat" => Some(2),
        "i32" | "float" => Some(4),
        "i64" | "double" => Some(8),
        other => other
            .strip_prefix('i')
            .and_then(|b| b.parse::<usize>().ok())
            .map(|bits| bits.div_ceil(8)),
    }
}

/// Census the primitive-scalar deref width of every pointer USE in the body (`load`/`store` element
/// type and `getelementptr` source element). This is the whole-vs-part signal the def-flow `classify`
/// misses: a component whose members are dereferenced at BOTH scalar `S` and vector `<N x S>` of one
/// element S — even when no single phi/select arm mixes the two — is a whole-vs-part network.
fn use_widths(body: &[&str]) -> HashMap<String, UseWidths> {
    let mut map: HashMap<String, UseWidths> = HashMap::new();
    let mut observe = |name: &str, ty: &str, ptr_prefix: &str| {
        let entry = map.entry(name.to_string()).or_default();
        if let Some((elem, _)) = vec_elem_lanes(ty) {
            entry.vectors.insert(elem);
        } else if is_simple_scalar(ty.trim()) {
            entry.scalars.insert(ty.trim().to_string());
        } else {
            entry.other = true;
        }
        entry.logical |= !is_word_addressable_ptr(ptr_prefix);
    };
    for line in body {
        let trimmed = line.trim();
        if let Some(def) = parse_def(line) {
            match def.opcode {
                "load" => {
                    let chunks = split_top_level(def.rest, ',');
                    if chunks.len() >= 2 {
                        if let Some((prefix, ptr)) = split_ptr_operand(chunks[1]) {
                            observe(ptr, chunks[0].trim(), prefix);
                        }
                    }
                }
                "getelementptr" => {
                    if let Some(g) = parse_gep(def.rest) {
                        let prefix = split_ptr_operand(g.base_chunk).map_or("", |(p, _)| p);
                        observe(g.base_name, g.src_ty, prefix);
                    }
                }
                _ => {}
            }
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("store ") {
            let chunks = split_top_level(rest, ',');
            if chunks.len() >= 2 {
                if let (Some((ty, _)), Some((prefix, ptr))) =
                    (split_ptr_operand(chunks[0]), split_ptr_operand(chunks[1]))
                {
                    observe(ptr, ty, prefix);
                }
            }
        }
    }
    map
}

/// Enumerate the whole-vs-part pointer networks (KEYSTONE-1): connected pointer-flow components whose
/// members are dereferenced at exactly one primitive scalar S but at BOTH scalar and whole-vector
/// width. Each returned pair is `(component members, S)` — every vector access on a member is then
/// scalarized to S, so the merged Logical pointer is a consistent `S*`. ReinterpretMix (>=2 distinct
/// scalars), any component with an unmodeled deref (`other`), any component touching a logical
/// non-word-addressable pointer (`logical` — Workgroup/thread, unsound to re-view at a finer stride),
/// and any component touching a memory-reloaded pointer (`memory_ptr` — a `load ptr` whose pointee is
/// pinned by memory-stored siblings the SSA adjacency cannot reach, so scalarizing only the visible
/// half is a partial retype; see [`memory_reloaded_pointers`]) are excluded.
fn widen_targets(body: &[&str], defs: &HashMap<String, Def>) -> Vec<(HashSet<String>, String)> {
    let widths = use_widths(body);
    let reloaded = memory_reloaded_pointers(body);
    let adj = pointer_adjacency(defs);
    let mut nodes: HashSet<String> = adj.keys().cloned().collect();
    nodes.extend(widths.keys().cloned());

    let mut visited: HashSet<String> = HashSet::new();
    let mut targets: Vec<(HashSet<String>, String)> = Vec::new();
    for start in &nodes {
        if !visited.insert(start.clone()) {
            continue;
        }
        let mut members: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(start.clone());
        members.insert(start.clone());
        while let Some(cur) = queue.pop_front() {
            if let Some(nbs) = adj.get(&cur) {
                for nb in nbs {
                    if visited.insert(nb.clone()) {
                        members.insert(nb.clone());
                        queue.push_back(nb.clone());
                    }
                }
            }
        }
        // Census this component's use widths.
        let mut elems: HashSet<String> = HashSet::new();
        let (mut has_scalar, mut has_vector, mut other, mut logical) = (false, false, false, false);
        let mut memory_ptr = false;
        for m in &members {
            memory_ptr |= reloaded.contains(m);
            if let Some(w) = widths.get(m) {
                for s in &w.scalars {
                    elems.insert(s.clone());
                    has_scalar = true;
                }
                for v in &w.vectors {
                    elems.insert(v.clone());
                    has_vector = true;
                }
                other |= w.other;
                logical |= w.logical;
            }
        }
        if !other && !logical && !memory_ptr && has_scalar && has_vector && elems.len() == 1 {
            let elem = elems.into_iter().next().unwrap();
            // Sub-word (< 32-bit) exclusion: scalarizing a whole-vs-part network to an 8/16-bit scalar
            // collapses the device buffer's SPIR-V descriptor from the MoltenVK-safe word view
            // (`{RuntimeArray<uint>}`, byte-extract per access) to a NATIVE `{RuntimeArray<S>}` view
            // (ArrayStride 1/2) — an 8/16-bit StorageBuffer access that needs `StorageBuffer8BitAccess`
            // / `StorageBuffer16BitAccess`, capabilities the executors never enable. That is
            // spirv-val-VALID but byte-WRONG on MoltenVK (dead-end #14: validity != correctness), the
            // `MPSRNNGRURecursion*char` miscompiles the serial-MoltenVK A/B caught. 32-bit-and-wider
            // scalars are the native word (no narrow-storage capability needed) and stay widened.
            // Keyed purely on the element type token's byte width — never a name. Excluded components
            // fall back to the currently-shipping WIDEN-off word-view emit, which conforms.
            if scalar_byte_width(&elem).is_some_and(|w| w < 4) {
                continue;
            }
            targets.push((members, elem));
        }
    }
    targets
}

/// A gep index operand parsed as a constant or a dynamic value with its integer type.
enum Index {
    Const(i128),
    Dyn { value: String, ity: String },
}

fn parse_index(operand: &str) -> Option<Index> {
    let mut it = operand.split_whitespace();
    let ity = it.next()?.to_string();
    let val = it.next()?.trim();
    if let Ok(c) = val.parse::<i128>() {
        Some(Index::Const(c))
    } else if val.starts_with('%') {
        Some(Index::Dyn {
            value: val.to_string(),
            ity,
        })
    } else {
        None
    }
}

/// Reserved prefix for the fresh SSA value names this pass synthesizes (`%.vsm0`, `%.vsm1`, …).
/// Named once so the generator ([`Rewriter::fresh`]) and the collision guard that bails when the
/// input already carries it (in [`vec_scalar_merge`]) can never drift apart — both must agree on the
/// exact literal for the reservation to be sound.
const VSM_NAME_PREFIX: &str = ".vsm";

struct Rewriter {
    counter: usize,
}

impl Rewriter {
    fn fresh(&mut self) -> String {
        let n = self.counter;
        self.counter += 1;
        format!("%{VSM_NAME_PREFIX}{n}")
    }

    /// Rewrite a `<N x ELEM>`-typed gep in the component into the scalar `getelementptr ELEM` whose
    /// index is the original index(es) scaled to element units. Byte-identical address.
    fn rewrite_gep(&mut self, name: &str, g: &GepInfo, elem: &str, lanes: usize) -> Vec<String> {
        let inbounds = if g.inbounds { "inbounds " } else { "" };
        let mut lines = Vec::new();
        // total element index = idx0 * lanes (+ idx1 for the 2-index vector-element form).
        let i0 = parse_index(g.indices[0]).unwrap();
        let (mut acc_const, mut acc_dyn): (i128, Option<(String, String)>) = match i0 {
            Index::Const(c) => (c * lanes as i128, None),
            Index::Dyn { value, ity } => {
                let m = self.fresh();
                lines.push(format!("  {m} = mul {ity} {value}, {lanes}"));
                (0, Some((m, ity)))
            }
        };
        if g.indices.len() >= 2 {
            if let Some(i1) = parse_index(g.indices[1]) {
                match i1 {
                    Index::Const(c) => acc_const += c,
                    Index::Dyn { value, ity } => {
                        acc_dyn = Some(match acc_dyn {
                            Some((acc, _)) => {
                                let a = self.fresh();
                                lines.push(format!("  {a} = add {ity} {acc}, {value}"));
                                (a, ity)
                            }
                            None => {
                                if acc_const == 0 {
                                    (value, ity)
                                } else {
                                    let a = self.fresh();
                                    lines.push(format!("  {a} = add {ity} {value}, {acc_const}"));
                                    acc_const = 0;
                                    (a, ity)
                                }
                            }
                        });
                    }
                }
            }
        }
        let (idx_ty, idx_val) = match acc_dyn {
            Some((v, ity)) if acc_const == 0 => (ity, v),
            Some((v, ity)) => {
                let a = self.fresh();
                lines.push(format!("  {a} = add {ity} {v}, {acc_const}"));
                (ity, a)
            }
            None => ("i64".to_string(), acc_const.to_string()),
        };
        let _ = elem;
        lines.push(format!(
            "  {name} = getelementptr {inbounds}{elem}, {base}, {idx_ty} {idx_val}",
            base = g.base_chunk
        ));
        lines
    }

    /// Decompose `%name = load <N x ELEM>, <ptr>` into N scalar loads + a vector rebuild.
    fn split_load(
        &mut self,
        name: &str,
        elem: &str,
        lanes: usize,
        ptr_prefix: &str,
        ptr_name: &str,
    ) -> Vec<String> {
        let mut lines = Vec::new();
        let mut elem_ids = Vec::with_capacity(lanes);
        for k in 0..lanes {
            let p = self.fresh();
            let l = self.fresh();
            lines.push(format!(
                "  {p} = getelementptr inbounds {elem}, {ptr_prefix} {ptr_name}, i64 {k}"
            ));
            lines.push(format!("  {l} = load {elem}, {ptr_prefix} {p}, align 4"));
            elem_ids.push(l);
        }
        let vty = format!("<{lanes} x {elem}>");
        let mut prev = "undef".to_string();
        for (k, elem_id) in elem_ids.iter().enumerate() {
            let out = if k + 1 == lanes {
                name.to_string()
            } else {
                self.fresh()
            };
            lines.push(format!(
                "  {out} = insertelement {vty} {prev}, {elem} {elem_id}, i32 {k}"
            ));
            prev = out;
        }
        lines
    }

    /// Decompose `store <N x ELEM> %v, <ptr>` into N scalar extract+store pairs.
    fn split_store(
        &mut self,
        value: &str,
        elem: &str,
        lanes: usize,
        ptr_prefix: &str,
        ptr_name: &str,
    ) -> Vec<String> {
        let vty = format!("<{lanes} x {elem}>");
        let mut lines = Vec::new();
        for k in 0..lanes {
            let x = self.fresh();
            let p = self.fresh();
            lines.push(format!("  {x} = extractelement {vty} {value}, i32 {k}"));
            lines.push(format!(
                "  {p} = getelementptr inbounds {elem}, {ptr_prefix} {ptr_name}, i64 {k}"
            ));
            lines.push(format!("  store {elem} {x}, {ptr_prefix} {p}, align 4"));
        }
        lines
    }
}

fn rewrite_function_body(body: &[&str], widen: bool) -> Option<Vec<String>> {
    let defs: HashMap<String, Def> = body
        .iter()
        .filter_map(|line| parse_def(line))
        .map(|d| (d.name.to_string(), d))
        .collect();

    // `comp_elem` maps every pointer whose vector accesses must be scalarized to the scalar element
    // its component narrows to. Two independent seeders feed it: the always-on def-flow scalar/vector
    // MERGE shape (floor-safe, the emitter rejects it), and — when `widen` is set (KEYSTONE-1) — the
    // use-width whole-vs-part networks (byte-changing, they emit today).
    let mut comp_elem: HashMap<String, String> = HashMap::new();

    let (pointees, mismatches) = classify(&defs);
    if let Some(elem) = unify_mismatch_elem(&mismatches, &pointees) {
        for m in component(&defs, &mismatches) {
            comp_elem.entry(m).or_insert_with(|| elem.clone());
        }
    }
    if widen {
        for (members, elem) in widen_targets(body, &defs) {
            for m in members {
                comp_elem.entry(m).or_insert_with(|| elem.clone());
            }
        }
    }

    if comp_elem.is_empty() {
        return None;
    }
    // Reserved fresh-name prefix must not already appear (it never does in llvm-dis output).
    if body.iter().any(|l| l.contains(VSM_NAME_PREFIX)) {
        return None;
    }
    let mut rw = Rewriter { counter: 0 };
    let mut out: Vec<String> = Vec::with_capacity(body.len());
    for line in body {
        if let Some(rewritten) = rewrite_line(line, &comp_elem, &mut rw) {
            out.extend(rewritten);
        } else {
            out.push(line.to_string());
        }
    }
    Some(out)
}

/// The single scalar element the def-flow mismatch merges narrow to, or `None` when there are no
/// mismatches or they disagree (either way the def-flow path scalarizes nothing — floor-safe).
fn unify_mismatch_elem(
    mismatches: &HashSet<String>,
    pointees: &HashMap<String, Pointee>,
) -> Option<String> {
    let mut elem: Option<String> = None;
    for m in mismatches {
        if let Some(Pointee::Scalar(e)) = pointees.get(m) {
            match &elem {
                Some(prev) if prev != e => return None,
                _ => elem = Some(e.clone()),
            }
        }
    }
    elem
}

/// Rewrite one body line if it is a vector gep / vector load / vector store whose pointer is a member
/// of a scalarized component (looked up in `comp_elem`, which also carries that component's target
/// scalar); `None` keeps the line verbatim.
fn rewrite_line(
    line: &str,
    comp_elem: &HashMap<String, String>,
    rw: &mut Rewriter,
) -> Option<Vec<String>> {
    let trimmed = line.trim();
    if let Some(def) = parse_def(line) {
        match def.opcode {
            "getelementptr" => {
                let elem = comp_elem.get(def.name)?;
                let g = parse_gep(def.rest)?;
                let (e, lanes) = vec_elem_lanes(g.src_ty)?;
                if &e != elem {
                    return None;
                }
                return Some(rw.rewrite_gep(def.name, &g, &e, lanes));
            }
            "load" => {
                let chunks = split_top_level(def.rest, ',');
                if chunks.len() < 2 {
                    return None;
                }
                let (e, lanes) = vec_elem_lanes(chunks[0].trim())?;
                let (ptr_prefix, ptr_name) = split_ptr_operand(chunks[1])?;
                let elem = comp_elem.get(ptr_name)?;
                if &e != elem {
                    return None;
                }
                return Some(rw.split_load(def.name, &e, lanes, ptr_prefix, ptr_name));
            }
            _ => return None,
        }
    }
    if let Some(rest) = trimmed.strip_prefix("store ") {
        let chunks = split_top_level(rest, ',');
        if chunks.len() < 2 {
            return None;
        }
        let (ty, value) = split_ptr_operand(chunks[0])?;
        let (e, lanes) = vec_elem_lanes(ty)?;
        let (ptr_prefix, ptr_name) = split_ptr_operand(chunks[1])?;
        let elem = comp_elem.get(ptr_name)?;
        if &e != elem {
            return None;
        }
        return Some(rw.split_store(value, &e, lanes, ptr_prefix, ptr_name));
    }
    None
}

/// Test-only entry that runs the whole-vs-part scalarization (KEYSTONE-1) with `widen` explicitly on,
/// so a spirv-val primary-emit test can scalarize a module and feed the result to the (non-widening)
/// `emit_vulkan_spirv` path.
#[cfg(test)]
pub(in crate::native) fn lower_with_widen_for_test(san_ll: &str) -> String {
    lower_impl(san_ll, true).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `MPSRNNBreakUpToOutputVecs` shape: a vectorized `<4 x float>` walk merged with a scalar
    /// `float` walk via `phi`. The pass must scalarize the vector loads/stores/geps so the merge
    /// becomes a legal `float*` Logical pointer.
    #[test]
    fn scalar_vector_pointer_merge_is_scalarized() {
        let src = "\
define void @k(ptr addrspace(1) %src, ptr addrspace(1) %dst, i64 %n) {
entry:
  %s0 = getelementptr inbounds float, ptr addrspace(1) %src, i64 %n
  %d0 = getelementptr inbounds float, ptr addrspace(1) %dst, i64 %n
  br label %loop
loop:
  %p = phi ptr addrspace(1) [ %pn, %loop ], [ %s0, %entry ]
  %q = phi ptr addrspace(1) [ %qn, %loop ], [ %d0, %entry ]
  %v = load <4 x float>, ptr addrspace(1) %p, align 16
  store <4 x float> %v, ptr addrspace(1) %q, align 16
  %pn = getelementptr inbounds <4 x float>, ptr addrspace(1) %p, i64 %n
  %qn = getelementptr inbounds <4 x float>, ptr addrspace(1) %q, i64 %n
  br label %loop
}
";
        let out = lower_vector_scalar_pointer_merge(src);
        // No vector load/store/gep survives.
        assert!(
            !out.contains("load <4 x float>"),
            "vector load not split:\n{out}"
        );
        assert!(
            !out.contains("store <4 x float>"),
            "vector store not split:\n{out}"
        );
        assert!(
            !out.contains("getelementptr inbounds <4 x float>"),
            "vector gep not scalarized:\n{out}"
        );
        // The vector-stride gep becomes a scalar gep with the index multiplied by the lane count.
        assert!(out.contains("mul i64 %n, 4"), "stride not scaled:\n{out}");
        // Four scalar loads rebuild the vector; the last insert defines the original %v.
        assert_eq!(
            out.matches("load float,").count(),
            4,
            "want 4 loads:\n{out}"
        );
        assert!(
            out.contains("%v = insertelement <4 x float>"),
            "vector not rebuilt into %v:\n{out}"
        );
        assert_eq!(
            out.matches("extractelement <4 x float>").count(),
            4,
            "want 4 extracts:\n{out}"
        );
    }

    /// A module with no scalar/vector pointer merge is returned byte-for-byte unchanged.
    #[test]
    fn unrelated_module_is_untouched() {
        let src = "\
define void @k(ptr addrspace(1) %src) {
entry:
  %p = getelementptr inbounds <4 x float>, ptr addrspace(1) %src, i64 0
  %v = load <4 x float>, ptr addrspace(1) %p, align 16
  ret void
}
";
        assert_eq!(lower_vector_scalar_pointer_merge(src), src);
    }

    /// KEYSTONE-1 use-width whole-vs-part: a phi whose arms are BOTH scalar-typed by def flow (so the
    /// always-on merge path does NOT fire) yet whose result is loaded WHOLE-vector. Only the use-width
    /// census sees the `<4 x float>` load, so with `widen` the vector load is scalarized to four `float`
    /// loads and the merged pointer becomes a consistent `float*`. With `widen` OFF the module is
    /// returned byte-for-byte unchanged (widen is DEFAULT ON in production, so this OFF mode is the
    /// `=0` opt-out, not the default).
    #[test]
    fn use_width_whole_vs_part_phi_is_scalarized_only_when_widen() {
        let src = "\
define void @k(ptr addrspace(1) %m, ptr addrspace(1) %n, i1 %c, i64 %i) {
entry:
  %a = getelementptr inbounds float, ptr addrspace(1) %m, i64 %i
  %b = getelementptr inbounds float, ptr addrspace(1) %n, i64 %i
  br label %loop
loop:
  %p = phi ptr addrspace(1) [ %a, %entry ], [ %b, %loop ]
  %v = load <4 x float>, ptr addrspace(1) %p, align 16
  br label %loop
}
";
        // widen=false (the `=0` opt-out) leaves the whole-vs-part shape untouched.
        assert_eq!(lower_impl(src, false), src);

        let out = lower_impl(src, true);
        assert!(
            !out.contains("load <4 x float>"),
            "vector load not split under widen:\n{out}"
        );
        assert_eq!(
            out.matches("load float,").count(),
            4,
            "want 4 scalar loads:\n{out}"
        );
        assert!(
            out.contains("%v = insertelement <4 x float>"),
            "vector not rebuilt into %v:\n{out}"
        );
        // The scalar-typed phi arms are already `float*` — untouched.
        assert!(out.contains("%a = getelementptr inbounds float"), "{out}");
        assert!(out.contains("%b = getelementptr inbounds float"), "{out}");
    }

    /// A single pointer dereferenced at BOTH `<4 x float>` and `float` (no phi/select at all) is a
    /// whole-vs-part network the use-width census catches: `widen` scalarizes the vector load, the
    /// scalar load is left verbatim.
    #[test]
    fn use_width_single_pointer_mixed_width_is_scalarized() {
        let src = "\
define void @k(ptr addrspace(1) %buf) {
entry:
  %v = load <4 x float>, ptr addrspace(1) %buf, align 16
  %s = load float, ptr addrspace(1) %buf, align 4
  ret void
}
";
        assert_eq!(lower_impl(src, false), src);
        let out = lower_impl(src, true);
        assert!(!out.contains("load <4 x float>"), "{out}");
        assert_eq!(out.matches("load float,").count(), 5, "{out}"); // 4 split + the original scalar
    }

    /// A genuine reinterpret (one pointer read as `<4 x float>` and as `i32`) is NOT whole-vs-part —
    /// its component has TWO distinct scalars — so `widen` must leave it untouched (dead-end #14: a
    /// reinterpret cannot be narrowed to one scalar soundly).
    #[test]
    fn use_width_reinterpret_mix_is_not_widened() {
        let src = "\
define void @k(ptr addrspace(1) %buf) {
entry:
  %v = load <4 x float>, ptr addrspace(1) %buf, align 16
  %w = load i32, ptr addrspace(1) %buf, align 4
  ret void
}
";
        assert_eq!(lower_impl(src, true), src);
    }

    /// A whole-vs-part network on a LOGICAL (non-word-addressable) pointer — here a threadgroup
    /// `addrspace(3)` scratch buffer, the `computeLSTMGate2` shape — is NOT widened: scalarizing a
    /// logical pointer to a finer stride needs an illegal logical-pointer bitcast (the emitter rejects
    /// "cannot reinterpret workgroup pointer arg … to raw word view"). Only device/constant
    /// (`addrspace(1)`/`(2)`) whole-vs-part networks are word-addressable enough to scalarize soundly.
    #[test]
    fn use_width_logical_workgroup_pointer_is_not_widened() {
        let src = "\
define void @k(ptr addrspace(3) %tg) {
entry:
  %v = load <4 x float>, ptr addrspace(3) %tg, align 16
  %s = load float, ptr addrspace(3) %tg, align 4
  ret void
}
";
        assert_eq!(lower_impl(src, true), src);
    }

    /// A pointer stepped through an AGGREGATE (`[16 x float]`) is an unmodeled deref (`other`) — the
    /// component is excluded even under `widen`, so no aggregate re-declaration is attempted.
    #[test]
    fn use_width_aggregate_gep_component_is_excluded() {
        let src = "\
define void @k(ptr addrspace(1) %buf) {
entry:
  %p = getelementptr inbounds [16 x float], ptr addrspace(1) %buf, i64 0, i64 0
  %v = load <4 x float>, ptr addrspace(1) %p, align 16
  %q = getelementptr inbounds [16 x float], ptr addrspace(1) %buf, i64 0, i64 4
  %s = load float, ptr addrspace(1) %q, align 4
  ret void
}
";
        assert_eq!(lower_impl(src, true), src);
    }

    /// The `MPSLSTMMultiInputKernelFloat` shape: a whole-vs-part network whose pointers are spilled to
    /// an `[N x ptr]` alloca and RELOADED (`%q = load ptr, ptr %slot`), then read one way scalar and
    /// another way vector. `pointer_adjacency` (gep/bitcast/phi/select) cannot follow the store→load
    /// memory edge, so the component is only HALF the real network — the memory-stored siblings keep
    /// their `vec4*` typing while the reloaded half would be scalarized, a partial retype that emits
    /// spirv-INVALID (`OpPtrAccessChain %float` on a `vec4*` base) and byte-miscompiles on MoltenVK.
    /// `widen` must leave such a component untouched even though it censuses as whole-vs-part.
    #[test]
    fn use_width_memory_reloaded_pointer_component_is_excluded() {
        // `%p` is reloaded from the alloca slot and dereferenced BOTH scalar (`load float`) and vector
        // (`gep <4 x float>` + `load <4 x float>`) — one connected component, whole-vs-part by use
        // width — but it is memory-reloaded, so the fix must leave it verbatim.
        let src = "\
define void @k(ptr addrspace(1) %slot, i64 %i) {
entry:
  %p = load ptr addrspace(1), ptr addrspace(1) %slot, align 8
  %s = load float, ptr addrspace(1) %p, align 4
  %g = getelementptr inbounds <4 x float>, ptr addrspace(1) %p, i64 %i
  %v = load <4 x float>, ptr addrspace(1) %g, align 16
  ret void
}
";
        assert_eq!(lower_impl(src, true), src);
        // Sanity: the SAME shape with `%p` an SSA buffer PARAMETER (not memory-reloaded) IS widened —
        // proving the exclusion keys on the reload, not on the whole-vs-part shape.
        let ssa = src.replace(
            "  %p = load ptr addrspace(1), ptr addrspace(1) %slot, align 8\n",
            "",
        );
        let ssa = ssa.replace("%slot, i64 %i", "%p, i64 %i");
        assert!(
            !lower_impl(&ssa, true).contains("load <4 x float>"),
            "SSA-visible whole-vs-part should still widen:\n{}",
            lower_impl(&ssa, true)
        );
    }

    #[test]
    fn use_width_subword_component_is_excluded() {
        // `%b` is a device buffer dereferenced BOTH scalar (`load i8`) and vector (`load <4 x i8>`) —
        // a whole-vs-part i8 network. Scalarizing it to native i8 forces an 8-bit StorageBuffer view
        // whose capability the executors never enable (spirv-val-valid, MoltenVK byte-wrong — the
        // MPSRNNGRURecursion*char miscompiles). It must be left verbatim; the width gate keys on the
        // element type token, never a name.
        let i8_src = "\
define void @k(ptr addrspace(1) %b, i64 %i) {
entry:
  %v = load <4 x i8>, ptr addrspace(1) %b, align 4
  %p = getelementptr inbounds i8, ptr addrspace(1) %b, i64 4
  %s = load i8, ptr addrspace(1) %p, align 1
  ret void
}
";
        assert_eq!(lower_impl(i8_src, true), i8_src);
        // The SAME shape at `float` (a native 32-bit word) IS widened — proving the gate keys on the
        // scalar width, not on the whole-vs-part shape.
        let f32_src = i8_src
            .replace("<4 x i8>", "<4 x float>")
            .replace("inbounds i8", "inbounds float")
            .replace("load i8", "load float")
            .replace("align 4", "align 16");
        assert!(
            !lower_impl(&f32_src, true).contains("load <4 x float>"),
            "float whole-vs-part should still widen:\n{}",
            lower_impl(&f32_src, true)
        );
    }

    #[test]
    fn scalar_byte_width_recognizes_primitive_tokens() {
        assert_eq!(scalar_byte_width("i8"), Some(1));
        assert_eq!(scalar_byte_width("half"), Some(2));
        assert_eq!(scalar_byte_width("i16"), Some(2));
        assert_eq!(scalar_byte_width("float"), Some(4));
        assert_eq!(scalar_byte_width("i32"), Some(4));
        assert_eq!(scalar_byte_width("i64"), Some(8));
        assert_eq!(scalar_byte_width("i24"), Some(3)); // arbitrary-width int, rounded up
        assert_eq!(scalar_byte_width("ptr"), None);
    }

    /// A pure-vector merge (both arms `<4 x float>*`) is legal as-is and must NOT be scalarized.
    #[test]
    fn pure_vector_merge_is_untouched() {
        let src = "\
define void @k(ptr addrspace(1) %a, ptr addrspace(1) %b, i1 %c) {
entry:
  %pa = getelementptr inbounds <4 x float>, ptr addrspace(1) %a, i64 0
  %pb = getelementptr inbounds <4 x float>, ptr addrspace(1) %b, i64 0
  %m = select i1 %c, ptr addrspace(1) %pa, ptr addrspace(1) %pb
  %v = load <4 x float>, ptr addrspace(1) %m, align 16
  ret void
}
";
        assert!(matches!(
            lower_vector_scalar_pointer_merge(src),
            Cow::Borrowed(value) if std::ptr::eq(value, src)
        ));
    }
}
