//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;
use crate::native::ir::{LlType, LlValue};
use crate::native::lex::strip_comment;
use crate::native::parse::{parse_type, split_top_level};
use std::collections::{HashMap, HashSet};

/// The ALLOCATED type of an `alloca` line — the first comma field of the rhs after the opcode,
/// parsed to an (unresolved) `LlType`. Resolved once at parse onto the typed carrier; byte-identical to
/// the retired text-path `alloca` handler by construction: it `strip_comment`/`trim`s the line, takes
/// the rhs after `%r = ` with the opcode token dropped, splits the top-level
/// commas, and `parse_type`s the first field. `None` when the line has no `%r = ` prefix or the type
/// does not parse — the emitter then reaches the fail-visible unmigrated-opcode `Err`, and a retry
/// re-parses the sanitized AIR from scratch.
pub(in crate::native) fn resolve_alloca_ty(line: &str) -> Option<LlType> {
    let line = strip_comment(line).trim();
    let rhs = line.split_once(" = ")?.1.trim();
    let opcode = rhs.split_whitespace().next().unwrap_or("");
    let rest = rhs[opcode.len()..].trim_start();
    let parts = split_top_level(rest, ',');
    parse_type(parts.first()?.trim()).ok()
}

/// The right-hand side of a `%r = <rhs>` instruction (trimmed), or the whole line if it has no `=`.
pub(in crate::native) fn rhs_of(line: &str) -> &str {
    line.split_once('=')
        .map(|(_, rhs)| rhs.trim())
        .unwrap_or(line)
}

/// Infer the pointee of a `getelementptr` result by walking its source aggregate along the constant
/// index path. AIR form: `getelementptr [inbounds] <srcty>, ptr [addrspace(N)] %base, <ty> <i0>,
/// <ty> <i1>, ...` — `i0` is the pointer-stride index (LLVM GEP semantics; does not enter the
/// aggregate), so the aggregate walk uses `i1..`. Returns `None` for a non-GEP rhs, a dynamic
/// (non-constant) index, or an index path that leaves a non-aggregate before it ends.
pub(in crate::native) fn resolve_gep_pointee(
    rhs: &str,
    named_types: &HashMap<String, LlType>,
) -> Option<LlType> {
    let after = rhs.strip_prefix("getelementptr ")?;
    let after = after.strip_prefix("inbounds ").unwrap_or(after);
    let parts = split_top_level(after, ',');
    // parts: [0]=source type, [1]=base pointer operand, [2]=i0 (stride, skipped), [3..]=i1.. walk.
    if parts.len() < 3 {
        return None;
    }
    let source_ty = parse_type(parts[0].trim()).ok()?;
    // Each index is the operand's last token: a constant integer, or a `%reg`/non-numeric (dynamic).
    // Dynamic indices resolve through array/vector steps (element type is index-independent) but not
    // struct steps, so carry them as `None` rather than bailing the whole walk.
    let indices: Vec<Option<usize>> = parts[3..]
        .iter()
        .map(|p| p.split_whitespace().last().and_then(|t| t.parse().ok()))
        .collect();
    extract_aggregate_member(source_ty, &indices, named_types)
}

/// Build the USE-based pointee map (see [`TirFunction::use_pointees`]). Scans every instruction for a
/// dereference that pins down the pointee of one of its pointer OPERANDS, and records it:
///   * `%r = load <ty>, ptr %p`        -> pointee(%p) = `<ty>`   (the loaded type)
///   * `store <ty> <v>, ptr %p`        -> pointee(%p) = `<ty>`   (the stored value's type)
///   * `%r = getelementptr <srcty>, ptr %p, ...` -> pointee(%p) = `<srcty>` (GEP source element type)
///     Returns the map plus the count of pointer values whose uses disagreed (a non-zero count is a real
///     signal — a pointer reinterpreted across types — surfaced by the self-check, not an error). On a
///     disagreement the richer pointee wins (`pointee_richness`): an aggregate/vector view subsumes a
///     scalar view, which subsumes a byte (`i8`) view of the same storage.
pub(in crate::native) fn infer_use_pointees<B: AsRef<TirBlock>>(
    blocks: &[B],
) -> (HashMap<String, LlType>, usize, HashSet<String>) {
    let mut map: HashMap<String, LlType> = HashMap::new();
    let mut conflicts = 0usize;
    // Pointers dereferenced at least once through a BYTE (`i8`) view — a `getelementptr inbounds i8`
    // byte cursor, an `i8` load/store, or a byte atomic. When such a pointer ALSO has a wider deref,
    // its carrier resolves to the wider type (richness), but the emitter still emits the byte-cursor
    // access as an `OpPtrAccessChain` with a `uchar` result — which is only well-typed against a
    // `uchar`-pointee base. Upgrading the pointee to the wider type (the M2 byte→real flip) strands
    // that byte cursor and produces globally-invalid SPIR-V (`OpPtrAccessChain result %uchar does not
    // match indexing into base %<wide>`). This set lets the byte→real upgrade EXCLUDE the mixed
    // byte/wide subset while still upgrading the pure-widening subset (a pointee that is only ever
    // dereferenced as the wider type, no byte cursor). See `pointer_pointee_for_value`.
    let mut byte_viewed: HashSet<String> = HashSet::new();
    for tb in blocks {
        let tb = tb.as_ref();
        for inst in &tb.insts {
            if let Some((ptr, pointee)) = deref_implied_pointee(inst) {
                if pointee == LlType::Int(8) {
                    byte_viewed.insert(ptr.to_string());
                    // A `getelementptr i8` RESULT is itself a byte cursor — it addresses a byte OFFSET
                    // into `ptr`, and the emitter lowers it to a `uchar`-result `OpPtrAccessChain`. So
                    // the result, not just the GEP base, is unsafe to widen (see the taint fixpoint
                    // below, which flows this through the `bitcast` that typically follows it).
                    if inst.opcode == "getelementptr" {
                        if let Some(result) = &inst.result {
                            byte_viewed.insert(result.clone());
                        }
                    }
                }
                record_use_pointee(&mut map, &mut conflicts, ptr, pointee);
            }
            for (ptr, pointee) in atomic_call_pointees(inst) {
                if pointee == LlType::Int(8) {
                    byte_viewed.insert(ptr.clone());
                }
                record_use_pointee(&mut map, &mut conflicts, &ptr, pointee);
            }
        }
    }
    // Propagate the byte-cursor taint FORWARD along pointer data flow: a `bitcast`/`select`/`phi`/
    // `freeze` whose pointer operand is a byte cursor produces another byte cursor (same byte-offset
    // address, re-typed). AIR emits a byte cursor as `%c = getelementptr i8, ptr %b, N` immediately
    // followed by `%p = bitcast %c to ptr` and then dereferences `%p` as the wide element — so without
    // this flow the taint would sit on `%c` while the byte→real upgrade fires on `%p`. Fixpoint because
    // these chain. Same aliasing opcodes as the pointee propagation below.
    //
    // `getelementptr` is ALSO a taint carrier here (unlike in the pointee-merge fixpoint below, where a
    // GEP result has a distinct pointee from its base): a TYPED GEP whose BASE is a byte cursor —
    // `%fp = getelementptr float, ptr %alias, N` where `%alias` re-typed a `getelementptr i8` cursor —
    // still addresses byte-granular storage at a byte offset, so its result must NOT be widened either.
    // AIR emits exactly this after a bitcast-of-i8-cursor (`%byte`→`%alias`→`%fp`/`%vp`) and reads the
    // result as the wide element; without GEP taint the byte→real upgrade fires on `%fp`/`%vp` and emits
    // a misaligned direct typed load instead of the byte assembly the offset requires (the
    // `native_byte_view_multiroot_phi…` miscompile). Tainting is conservative — it only ever EXCLUDES a
    // pointer from the upgrade, never miscompiles; a pure-widening pointee (no byte cursor in its chain)
    // has no tainted base and is unaffected.
    let mut changed = true;
    while changed {
        changed = false;
        for tb in blocks {
            let tb = tb.as_ref();
            for inst in &tb.insts {
                let Some(result) = &inst.result else { continue };
                if byte_viewed.contains(result) || !matches!(inst.result_ty, Some(LlType::Ptr(_))) {
                    continue;
                }
                let op = inst.opcode.as_str();
                if !matches!(
                    op,
                    "bitcast" | "select" | "phi" | "freeze" | "getelementptr"
                ) {
                    continue;
                }
                let tainted_operand = inst.operands.iter().any(|operand| match operand {
                    TirOperand::Value { name, ty } => {
                        matches!(ty, LlType::Ptr(_)) && byte_viewed.contains(name)
                    }
                    _ => false,
                }) || (op == "phi"
                    && inst.phi_values().is_some_and(|values| {
                        values.into_iter().any(|value| {
                            matches!(value, LlValue::Local(name) if byte_viewed.contains(name))
                        })
                    }));
                if tainted_operand {
                    byte_viewed.insert(result.clone());
                    changed = true;
                }
            }
        }
    }
    // Propagate pointees across pointer MERGES. A `select`/`phi`/`freeze` whose result is a pointer
    // aliases the SAME memory as its pointer operands, so the result and those operands share one
    // pointee. Deref sites pin only some members (e.g. a pointer merged from a typed arm but
    // dereferenced only on the OTHER arm, or passed to a call without a local deref); this flows the
    // known pointee to the rest. `getelementptr`/`load`/`call` pointer RESULTS are NOT merges (their
    // pointee differs from any operand), so only the three aliasing opcodes participate. Iterated to a
    // fixpoint because merges chain (a phi of a select of a phi). The richer view wins (same order as
    // `record_use_pointee`); this fills coverage, it does not re-count conflicts (deref-site
    // disagreements are already tallied above).
    let mut changed = true;
    while changed {
        changed = false;
        for tb in blocks {
            let tb = tb.as_ref();
            for inst in &tb.insts {
                let Some(result) = &inst.result else { continue };
                if !matches!(inst.result_ty, Some(LlType::Ptr(_))) {
                    continue;
                }
                let op = inst.opcode.as_str();
                if !matches!(op, "select" | "phi" | "freeze") {
                    continue;
                }
                let mut members: Vec<&str> = vec![result.as_str()];
                if op == "phi" {
                    if let Some(values) = inst.phi_values() {
                        for value in values {
                            if let LlValue::Local(name) = value {
                                members.push(name.as_str());
                            }
                        }
                    }
                } else {
                    for operand in &inst.operands {
                        if let TirOperand::Value { name, ty } = operand {
                            if matches!(ty, LlType::Ptr(_)) {
                                members.push(name.as_str());
                            }
                        }
                    }
                }
                let best = members
                    .iter()
                    .filter_map(|m| map.get(*m))
                    .max_by_key(|t| pointee_richness(t))
                    .cloned();
                let Some(best) = best else { continue };
                let best_rank = pointee_richness(&best);
                for m in &members {
                    if map.get(*m).map(pointee_richness) != Some(best_rank) {
                        map.insert(m.to_string(), best.clone());
                        changed = true;
                    }
                }
            }
        }
    }
    (map, conflicts, byte_viewed)
}

/// The buffer-pointer dereferences an `air.atomic.{global,local}.*` intrinsic call pins down. AIR
/// lowers Metal buffer atomics to these intrinsics — the DOTTED prefix, which the underscore
/// `air.atomic_*_texture_*` texture-atomic family (whose first operand is a texture, not a buffer
/// pointer) does not match, so textures are excluded structurally. Every POINTER argument is the
/// address of one atomic element, so its pointee is the element type: the call's result type for a
/// value-returning atomic (`load`/`add`/`xchg`/`cmpxchg`/`min`/...), or the first non-pointer
/// argument's type for a void-returning `store`. (`cmpxchg` passes the target AND an expected-value
/// pointer — both point at the element, so both are typed.) Dispatching on the stable `air.atomic.*`
/// intrinsic name is the AIR/LLVM-ABI exception the project allows, not name-keyed special-casing. Returns
/// `(SSA %name, element type)` for each `%`-local pointer argument; a constant/global pointer operand
/// (e.g. an `@`-symbol threadgroup buffer) is skipped — its pointee is already known from its decl.
pub(in crate::native) fn atomic_call_pointees(inst: &TirInst) -> Vec<(String, LlType)> {
    let Some(call) = &inst.call() else {
        return Vec::new();
    };
    if !(call.callee.starts_with("air.atomic.global.")
        || call.callee.starts_with("air.atomic.local."))
    {
        return Vec::new();
    }
    // Element type: the call's result for a value-returning atomic; else (void `store`) the first
    // non-pointer argument's type — the stored value.
    let element = match &inst.result_ty {
        Some(t) => t.clone(),
        None => match call.args.iter().find(|tv| !matches!(tv.ty, LlType::Ptr(_))) {
            Some(tv) => tv.ty.clone(),
            None => return Vec::new(),
        },
    };
    call.args
        .iter()
        .filter_map(|tv| match (&tv.ty, &tv.value) {
            (LlType::Ptr(_), LlValue::Local(name)) => Some((name.clone(), element.clone())),
            _ => None,
        })
        .collect()
}

/// USE-pointee coverage for the self-check: `(resolved, beyond_gep, conflicts)` — the number of
/// dereferenced pointers with a use-implied pointee, the subset whose pointee is net-new over the
/// GEP-result `pointer_pointees` map, and the number of pointers whose dereferences disagree.
pub(in crate::native) fn use_pointee_coverage(tir: &TirFunction) -> (usize, usize, usize) {
    let conflicts = infer_use_pointees(&tir.blocks).1;
    let resolved = tir.use_pointees.len();
    let beyond_gep = tir
        .use_pointees
        .keys()
        .filter(|k| !tir.pointer_pointees.contains_key(*k))
        .count();
    (resolved, beyond_gep, conflicts)
}
