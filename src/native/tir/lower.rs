//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;
use crate::native::ir::{LlGep, LlType, LlValue};
use crate::native::parse::{
    parse_call, parse_phi, parse_switch, parse_typed_value, strip_call_prefix, strip_comment,
    LlSwitch,
};
use std::collections::{HashMap, HashSet};

/// Compute the emit-ready `ret` decision for a block's terminator line (see [`RetEmit`]). Uses
/// `strip_comment(...).trim()` + `strip_prefix("ret ")` + `rest.trim() == "void"` void test +
/// `parse_typed_value(rest)` on the (metadata-including) rest string, so the value/void decision is exact
/// (including the `ret void, !dbg` case `Ret(None)` would misclassify — see [`RetEmit`]). A non-`ret`
/// line, or a `ret` value that does not parse, yields `FromText` — a fail-visible emit error (the raw
/// terminator line is no longer stored), measured dead broadly.
pub(in crate::native) fn ret_emit(terminator_text: &str) -> RetEmit {
    let cleaned = strip_comment(terminator_text).trim();
    let Some(rest) = cleaned.strip_prefix("ret ") else {
        return RetEmit::FromText;
    };
    if rest.trim() == "void" {
        return RetEmit::Void;
    }
    match parse_typed_value(rest) {
        Ok(tv) => RetEmit::Value(tv),
        Err(_) => RetEmit::FromText,
    }
}

/// Compute the emit-ready `switch` operands for a block's terminator line, or `None` when it is not a
/// `switch` (or a `switch` whose operands do not parse). Gates on
/// `strip_comment(...).trim().starts_with("switch ")` then runs `parse_switch` on the cleaned line, so
/// `emit_switch_resolved` emits from the typed `LlSwitch`. A parse failure yields `None` — a fail-visible
/// emit error (the raw line is no longer stored), measured dead broadly.
pub(in crate::native) fn switch_emit(terminator_text: &str) -> Option<LlSwitch> {
    let cleaned = strip_comment(terminator_text).trim();
    if !cleaned.starts_with("switch ") {
        return None;
    }
    parse_switch(cleaned).ok()
}

/// Compute the `phi` incoming carrier (parsed result type + `(value, predecessor)` pairs) for an
/// instruction line, together with the exact `parse_phi` refusal when that carrier is absent.
/// Computes the post-opcode rest from `strip_comment(line).trim()` (the rhs after `%r = `, opcode
/// token dropped) and hands it to `parse_phi` — the same derivation `emit_phi_resolved` consumes
/// from the carrier. The refusal is diagnostics-only: it never changes parsing or lowering.
pub(in crate::native) fn phi_incoming_parse(
    line: &str,
) -> (Option<(LlType, Vec<(LlValue, String)>)>, Option<String>) {
    let Some(rhs) = strip_comment(line)
        .trim()
        .split_once(" = ")
        .map(|s| s.1.trim())
    else {
        return (None, None);
    };
    let opcode = rhs.split_whitespace().next().unwrap_or("");
    let rest = rhs[opcode.len()..].trim_start();
    match parse_phi(rest) {
        Ok(parsed) => (Some(parsed), None),
        Err(error) => (None, Some(error)),
    }
}

/// Collect every `getelementptr` result (`%r = getelementptr …`) keyed by SSA name → parsed `LlGep`.
/// Sourced entirely from the typed graph: a `getelementptr` instruction carries its full parsed
/// `LlGep` on `TirInst.gep()` (set at build via the same `parse_gep` this walk used to re-lex from text)
/// and its result name on `TirInst.result`. Byte-identical to the retired text-walk by construction —
/// same instructions, same `parse_gep` output, same `%name` keys — with no `inst.text` read.
pub(in crate::native) fn collect_forward_geps<B: AsRef<TirBlock>>(
    blocks: &[B],
) -> HashMap<String, LlGep> {
    let mut geps = HashMap::new();
    for block in blocks {
        let block = block.as_ref();
        for inst in &block.insts {
            if let (Some(name), Some(gep)) = (&inst.result, &inst.gep()) {
                geps.insert(name.clone(), (**gep).clone());
            }
        }
    }
    geps
}

/// Collect the pointer-`phi` membership sets from a built block list: the result names of every
/// `%r = phi ptr ...` and the `%name` incoming values those phis merge. Sourced entirely from the
/// typed graph: a `phi ptr` is `opcode == "phi"` with a pointer `result_ty` (both the plain `ptr` and
/// the `ptr addrspace(N)` forms resolve to `LlType::Ptr`, and a vector-of-ptr phi resolves to
/// `Vector`, so this is exactly the old `phi ptr ` / `phi ptr addrspace(` text predicate), and the
/// incoming VALUES come from the canonical phi carrier (labels remain alongside them as control-flow
/// edges). Byte-identical to the retired `parse_phi_incoming_values` + `LlValue::Local` filter.
pub(in crate::native) fn collect_pointer_phi_sets<B: AsRef<TirBlock>>(
    blocks: &[B],
) -> (HashSet<String>, HashSet<String>) {
    let mut results = HashSet::new();
    let mut incoming = HashSet::new();
    for block in blocks {
        let block = block.as_ref();
        for inst in &block.insts {
            if inst.opcode != "phi" || !matches!(inst.result_ty, Some(LlType::Ptr(_))) {
                continue;
            }
            let Some(name) = &inst.result else { continue };
            results.insert(name.clone());
            if let Some(values) = inst.phi_values() {
                for value in values {
                    if let LlValue::Local(name) = value {
                        incoming.insert(name.clone());
                    }
                }
            } else {
                for operand in &inst.operands {
                    if let TirOperand::Value { name, .. } = operand {
                        incoming.insert(name.clone());
                    }
                }
            }
        }
    }
    (results, incoming)
}

/// Parse a function body into the typed SSA IR. `entry_label` is the synthetic name for the implicit
/// entry block (the leading instructions before the first explicit `label:`), matching the CFG layer.
/// `named_types` is the module's `%struct.name = type {...}` table, used to resolve `extractvalue`
/// into a named-struct aggregate (pass `&HashMap::new()` when unavailable — those values stay `None`).
///
/// TEST-ONLY: production builds the typed IR via [`build_from_blocks`] from the `split_body_blocks`
/// carriers (the sole substrate), which also carry the `pointer_pointee` this flat walk accumulates —
/// so `build_from_blocks` yields the same `TirFunction`. This flat-from-`body`-lines entry survives only
/// as the unit-test constructor (hand-written body lines → `TirFunction`); no production path re-lexes
/// body text through it (`LlFunction` no longer even carries a `Vec<String>` body).
#[cfg(test)]
pub(in crate::native) fn build(
    body: &[String],
    entry_label: &str,
    named_types: &HashMap<String, LlType>,
) -> Result<TirFunction, String> {
    let mut blocks: Vec<TirBlock> = Vec::new();
    let mut value_types: HashMap<String, LlType> = HashMap::new();
    let mut pointer_pointees: HashMap<String, LlType> = HashMap::new();

    let mut cur_label = entry_label.to_string();
    let mut cur_insts: Vec<TirInst> = Vec::new();
    let mut cur_term: Option<TirTerminator> = None;
    let mut cur_term_text = String::new();

    for raw in body {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        // A `label:` header opens a new block (the previous one must already be terminated).
        if let Some(label) = parse_block_label(line) {
            if cur_term.is_some() || !cur_insts.is_empty() {
                blocks.push(finish_flat_block(
                    cur_label,
                    cur_insts,
                    cur_term,
                    cur_term_text,
                )?);
                cur_insts = Vec::new();
                cur_term = None;
                cur_term_text = String::new();
            }
            cur_label = label;
            continue;
        }
        if let Some(term) = parse_terminator(line) {
            cur_term = Some(term);
            cur_term_text = line.to_string();
            continue;
        }
        push_inst_line(
            line,
            named_types,
            &mut value_types,
            &mut pointer_pointees,
            &mut cur_insts,
        );
    }
    if !cur_insts.is_empty() || cur_term.is_some() || !blocks.is_empty() {
        blocks.push(finish_flat_block(
            cur_label,
            cur_insts,
            cur_term,
            cur_term_text,
        )?);
    }

    let (use_pointees, _, byte_view_pointers) = infer_use_pointees(&blocks);
    let (pointer_phi_results, pointer_phi_incoming) = collect_pointer_phi_sets(&blocks);
    let forward_geps = collect_forward_geps(&blocks);
    Ok(TirFunction {
        blocks: blocks.into_iter().map(std::sync::Arc::new).collect(),
        value_types,
        pointer_pointees,
        use_pointees,
        byte_view_pointers,
        pointer_phi_results,
        pointer_phi_incoming,
        forward_geps,
    })
}

#[cfg(test)]
fn finish_flat_block(
    label: String,
    insts: Vec<TirInst>,
    term: Option<TirTerminator>,
    terminator_text: String,
) -> Result<TirBlock, String> {
    let terminator = term.ok_or_else(|| format!("native tir: block {label} has no terminator"))?;
    Ok(TirBlock {
        label,
        insts,
        terminator,
        ret: ret_emit(&terminator_text),
        switch: switch_emit(&terminator_text),
    })
}

/// Build the typed SSA IR from a list of already-split basic blocks (each a label plus its typed
/// carrier). This is the entry point for emission: emission walks the *structurized* CFG
/// (`cfg::structured_plan` reorders blocks and inserts synthetic `%metal2vulkan.lmerge.*` merge blocks +
/// phis between parse and emit), so a typed graph that matches what emission actually emits must be built
/// from those post-structurization `BodyBlock`s — NOT from the un-structurized parse-time `f.blocks`,
/// whose phis/predecessors the structurizer has since rewritten. Each `BodyBlock` carries
/// its label (`name`) and its typed carrier (`BodyBlock.typed`) — the SOLE substrate, built at split
/// time and dual-updated at every synthesis/mutation site. This consumes those carriers directly; a
/// block with no carrier is a fail-visible `Err` (a synthesis site that produced a block without
/// lowering its carrier), not a re-lower fallback. This is the sole emission substrate: `emit_function`
/// walks the returned `TirFunction.blocks`.
pub(in crate::native) fn build_from_blocks(
    blocks: &[crate::native::cfg::BodyBlock],
) -> Result<TirFunction, String> {
    let mut tir_blocks: Vec<std::sync::Arc<TirBlock>> = Vec::new();
    let mut value_types: HashMap<String, LlType> = HashMap::new();
    // Diagnostic-only accumulator, rebuilt from each carrier inst's `pointer_pointee` field (stamped at
    // lower time by the same `resolve_gep_pointee` the flat `build` ran) — byte-identical to `build`'s
    // own map, so this is a complete stand-in. Emission never reads it.
    let mut pointer_pointees: HashMap<String, LlType> = HashMap::new();

    // The typed carrier is the SOLE emission substrate: emission walks each block's carrier. A block is
    // carriered from birth (`split_body_blocks`) and every synthesis/mutation site dual-updates it, so a
    // `None` carrier is a fail-visible defect (a synthesis site that produced a block without lowering
    // its carrier), not a re-lower fallback. `value_types` (diagnostic-only accumulator, unread on the
    // emission path — production reads only the block-derived aggregations below) is re-derived from the
    // carrier's typed results. `pointer_pointees` is likewise diagnostic-only, rebuilt from each inst's
    // `pointer_pointee` field below (so the carrier is a complete stand-in for the flat `build`).
    for bb in blocks {
        let Some(carrier) = &bb.typed else {
            return Err(format!(
                "native tir: block {} role={:?} has no typed carrier (unpopulated synthesis site)",
                bb.name, bb.role
            ));
        };
        for inst in &carrier.insts {
            if let (Some(name), Some(ty)) = (&inst.result, &inst.result_ty) {
                value_types.insert(name.clone(), ty.clone());
            }
            if let (Some(name), Some(pointee)) = (&inst.result, &inst.pointer_pointee()) {
                pointer_pointees.insert(name.clone(), pointee.clone());
            }
        }
        tir_blocks.push(std::sync::Arc::clone(carrier));
    }

    let (use_pointees, _, byte_view_pointers) = infer_use_pointees(&tir_blocks);
    let (pointer_phi_results, pointer_phi_incoming) = collect_pointer_phi_sets(&tir_blocks);
    let forward_geps = collect_forward_geps(&tir_blocks);
    Ok(TirFunction {
        blocks: tir_blocks,
        value_types,
        pointer_pointees,
        use_pointees,
        byte_view_pointers,
        pointer_phi_results,
        pointer_phi_incoming,
        forward_geps,
    })
}

/// Lower one already-split basic block (label `name` + its instruction lines, terminator included)
/// into a typed [`TirBlock`], threading each result's resolved type into `value_types` and any inferred
/// GEP pointee into `pointer_pointees` (the function-level accumulators [`build_from_blocks`] carries
/// across all blocks). Errs on a block with no terminator. Shared by [`build_from_blocks`] and the
/// `BodyBlock.typed` carrier population, so a carrier lowered here is byte-identical to the re-parse by
/// construction — the same `parse_terminator` / `push_inst_line` / `ret_emit` / `switch_emit`.
pub(in crate::native) fn lower_block<S: AsRef<str>>(
    name: &str,
    lines: &[S],
    named_types: &HashMap<String, LlType>,
    value_types: &mut HashMap<String, LlType>,
    pointer_pointees: &mut HashMap<String, LlType>,
) -> Result<TirBlock, String> {
    let mut cur_insts: Vec<TirInst> = Vec::new();
    let mut cur_term: Option<TirTerminator> = None;
    let mut cur_term_text = String::new();
    for raw in lines {
        let line = raw.as_ref().trim();
        if line.is_empty() {
            continue;
        }
        if let Some(term) = parse_terminator(line) {
            cur_term = Some(term);
            cur_term_text = line.to_string();
            continue;
        }
        push_inst_line(
            line,
            named_types,
            value_types,
            pointer_pointees,
            &mut cur_insts,
        );
    }
    let terminator =
        cur_term.ok_or_else(|| format!("native tir: block {name} has no terminator"))?;
    Ok(TirBlock {
        label: name.to_string(),
        insts: cur_insts,
        terminator,
        ret: ret_emit(&cur_term_text),
        switch: switch_emit(&cur_term_text),
    })
}

/// Build a [`BodyBlock::typed`] carrier for one block from its lines, discarding the function-level
/// accumulators (those are rebuilt at emit from the whole block list). The `Option` form the synthesis
/// sites store — `None` if the lines do not lower (no terminator). Byte-identical to the emit-time
/// re-parse by construction (the shared [`lower_block`]). Pass the module `named_types` when the lines
/// can carry an `extractvalue` into a named struct; `&HashMap::new()` is exact for purely synthetic
/// `br`/`ret`/`icmp`/`phi` blocks (which never do).
pub(in crate::native) fn lower_block_carrier<S: AsRef<str>>(
    name: &str,
    lines: &[S],
    named_types: &HashMap<String, LlType>,
) -> Option<TirBlock> {
    let mut value_types = HashMap::new();
    let mut pointer_pointees = HashMap::new();
    lower_block(
        name,
        lines,
        named_types,
        &mut value_types,
        &mut pointer_pointees,
    )
    .ok()
}

/// Build a carrier for a constructed block that COPIES an existing block's instruction PREFIX (its
/// `lines[..terminator]`, whose types the emitter cannot re-resolve in the cfg layer without the module
/// `named_types`) and then appends a synthetic `tail_lines` (new `icmp`/`br`/`ret` scaffolding + the
/// terminator, which never carry named-struct types). Reuses `prefix`'s already-typed carrier insts for
/// the copied portion and lowers only the synthetic tail with empty named-types. Byte-identical to
/// re-lowering the full `prefix-lines ++ tail_lines` with the real module types by construction: the
/// prefix insts come from a carrier the emitter already lowered with the real types, and the tail is
/// type-resolution-independent. `None` if the tail does not lower (no terminator).
pub(in crate::native) fn lower_block_carrier_with_prefix(
    name: &str,
    prefix: &TirBlock,
    tail_lines: &[String],
) -> Option<TirBlock> {
    let tail = lower_block_carrier(name, tail_lines, &HashMap::new())?;
    let mut insts = prefix.insts.clone();
    insts.extend(tail.insts);
    Some(TirBlock {
        label: name.to_string(),
        insts,
        terminator: tail.terminator,
        ret: tail.ret,
        switch: tail.switch,
    })
}

/// Build a carrier for a block that COPIES an existing block's instruction SUFFIX (`source.insts[skip..]`
/// — e.g. a loop body extracted from its header, dropping the leading `skip` phis) and gives it
/// `terminator_line` (a synthetic/label-redirected `br`, type-resolution-independent). Reuses the source
/// carrier's suffix insts (real types already resolved) and lowers only the new terminator with empty
/// named-types. Byte-identical to re-lowering the copied suffix lines + terminator with the real module
/// types by construction. `None` if the terminator does not lower or `skip` exceeds the source insts.
pub(in crate::native) fn lower_block_carrier_from_suffix(
    name: &str,
    source: &TirBlock,
    skip: usize,
    terminator_line: &str,
) -> Option<TirBlock> {
    let insts = source.insts.get(skip..)?.to_vec();
    let term = lower_block_carrier(
        name,
        std::slice::from_ref(&terminator_line.to_string()),
        &HashMap::new(),
    )?;
    Some(TirBlock {
        label: name.to_string(),
        insts,
        terminator: term.terminator,
        ret: term.ret,
        switch: term.switch,
    })
}

/// Build a carrier for a block that COPIES an existing block's instruction PREFIX (`source.insts[..keep]`
/// — e.g. a loop header's leading phis, dropping the body instruction suffix) and gives it
/// `terminator_line` (a synthetic `br`, type-resolution-independent). Symmetric to
/// [`lower_block_carrier_from_suffix`]: reuses the source carrier's prefix insts (real types already
/// resolved) and lowers only the new terminator with empty named-types. Byte-identical to re-lowering
/// the copied prefix lines + terminator with the real module types by construction. `None` if the
/// terminator does not lower or `keep` exceeds the source insts.
pub(in crate::native) fn lower_block_carrier_prefix(
    name: &str,
    source: &TirBlock,
    keep: usize,
    terminator_line: &str,
) -> Option<TirBlock> {
    let insts = source.insts.get(..keep)?.to_vec();
    let term = lower_block_carrier(
        name,
        std::slice::from_ref(&terminator_line.to_string()),
        &HashMap::new(),
    )?;
    Some(TirBlock {
        label: name.to_string(),
        insts,
        terminator: term.terminator,
        ret: term.ret,
        switch: term.switch,
    })
}

/// Process one non-terminator, non-label instruction line into a `TirInst`, threading the result's
/// resolved type into `value_types` and any inferred GEP pointee into `pointer_pointees`. Shared by
/// `build` (flat body-line stream, test-only) and `build_from_blocks` (pre-split blocks).
pub(in crate::native) fn push_inst_line(
    line: &str,
    named_types: &HashMap<String, LlType>,
    value_types: &mut HashMap<String, LlType>,
    pointer_pointees: &mut HashMap<String, LlType>,
    cur_insts: &mut Vec<TirInst>,
) {
    let mut pointer_pointee: Option<LlType> = None;
    let (result, result_ty) = match resolve_result(line, named_types) {
        Some((name, ty)) => {
            if let Some(ty) = &ty {
                value_types.insert(name.clone(), ty.clone());
            }
            // Pointer-typed result: try to infer its pointee. A `getelementptr` walks its
            // source aggregate by the constant index path; record the resulting pointee so a
            // later use (e.g. a reinterpret load) can see it. The value is both accumulated into
            // the function-level `pointer_pointees` map (the flat `build`'s output) AND stamped on
            // the inst below, so the carrier alone can rebuild the map in `build_from_blocks`.
            if matches!(ty, Some(LlType::Ptr(_))) {
                if let Some(pointee) = resolve_gep_pointee(rhs_of(line), named_types) {
                    pointer_pointees.insert(name.clone(), pointee.clone());
                    pointer_pointee = Some(pointee);
                }
            }
            (Some(name), ty)
        }
        None => (result_name(line), None),
    };
    let mut operands = resolve_operands(line);
    let cmp_predicate = resolve_cmp_predicate(line);
    let mem_align = resolve_mem_align(line);
    let gep = resolve_gep(line).map(Box::new);
    let call = resolve_call(line).map(Box::new);
    let opcode = rhs_of(line)
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_string();
    let fast_math = rhs_of(line).split_whitespace().nth(1) == Some("fast");
    let alloca_ty = if opcode == "alloca" {
        resolve_alloca_ty(line)
    } else {
        None
    };
    let (phi_incoming, phi_parse_error) = if opcode == "phi" {
        phi_incoming_parse(line)
    } else {
        (None, None)
    };
    // A parsed phi's canonical `(value, predecessor)` carrier already contains the typed incoming
    // values consumed by emission and CFG edits. Keeping the parallel `TirOperand` copies would retain
    // every (often aggregate) value twice on phi-heavy modules. Preserve the unresolved placeholder
    // only for malformed forms whose canonical parse failed, so they still fall back honestly.
    if phi_incoming.is_some() {
        operands.clear();
    }
    // Typed operands are the canonical def/use carrier whenever the complete operand shape resolved.
    // Retain the textual scan only for an unresolved shape, where dropping it could hide an edge from
    // structurization or diagnostics. Parsed phis derive uses from their canonical incoming values.
    let uses = (phi_incoming.is_none()
        && operands
            .iter()
            .any(|operand| matches!(operand, TirOperand::Unresolved)))
    .then(|| instruction_uses(line, result.as_deref()));
    let aggregate_indices = resolve_aggregate_indices(line, &opcode);
    // Diagnostics-only strip-commented line for the element-op families whose resolved cores embed the
    // raw line in post-resolution SEMANTIC errors; read only by that error formatting (never re-parsed).
    let diag_line = if matches!(
        opcode.as_str(),
        "extractelement" | "insertelement" | "shufflevector"
    ) {
        Some(crate::native::lex::strip_comment(line).trim().to_string())
    } else {
        None
    };
    let shuffle_mask = if opcode == "shufflevector" {
        resolve_shuffle_mask(line)
    } else {
        None
    };
    let bitcast = resolve_bitcast(line, &opcode).map(Box::new);
    let icmp_rest = resolve_icmp_rest(line, &opcode);
    // Precomputed parse-time views the parse-time pointer-alias / raw-buffer / pointee inferences read
    // off the carrier instead of re-lexing the body text (F-track / T5). Each mirrors the exact expression
    // the inference used, computed once here at lower time; byte-identical by construction, BC-refereed.
    let identity_ptr_bitcast = resolve_identity_ptr_bitcast(line);
    let phi_incoming_values = if opcode == "phi" && phi_incoming.is_none() {
        resolve_phi_incoming_values(line, &opcode)
    } else {
        None
    };
    let select_arms = resolve_select_arms(line, &opcode).map(Box::new);
    let load = resolve_load_inst(line, &opcode).map(Box::new);
    let store = resolve_store(line, &opcode).map(Box::new);
    let alias_call = resolve_alias_call(line).map(Box::new);
    let emit_scan_call = resolve_emit_scan_call(line);
    // For a result-LESS call/tail (a VOID call), the strip-commented line the void-call emitter needs for
    // the is_ignored gate + the non-void diagnostic (a value call carries a result and rides `call`).
    let void_call_line = if matches!(opcode.as_str(), "call" | "tail") && result.is_none() {
        Some(crate::native::lex::strip_comment(line).trim().to_string())
    } else {
        None
    };
    let value_call_error =
        if matches!(opcode.as_str(), "call" | "tail") && result.is_some() && call.is_none() {
            let cleaned = crate::native::lex::strip_comment(line).trim();
            let call_text = cleaned
                .split_once(" = ")
                .and_then(|(_, rhs)| strip_call_prefix(rhs.trim()));
            call_text.and_then(|text| parse_call(text).err())
        } else {
            None
        };
    let data = match opcode.as_str() {
        "icmp" | "fcmp" => TirInstData::Compare {
            predicate: cmp_predicate,
            rest: icmp_rest,
        },
        "load" | "store" => TirInstData::Memory {
            align: mem_align,
            load,
            store,
        },
        "getelementptr" => TirInstData::Gep {
            parsed: gep,
            pointee: pointer_pointee,
        },
        "call" | "tail" | "musttail" | "notail" => {
            let alias_from_parsed = alias_call.is_some() && call.is_some();
            let alias_override = (!alias_from_parsed).then_some(alias_call).flatten();
            let emit_scan = match emit_scan_call {
                Some(Ok(_)) if call.is_some() => EmitScanData::Parsed,
                Some(result) => EmitScanData::Owned(Box::new(result)),
                None => EmitScanData::None,
            };
            TirInstData::Call {
                parsed: call,
                void_line: void_call_line,
                value_error: value_call_error,
                alias_from_parsed,
                alias_override,
                emit_scan,
            }
        }
        "alloca" => TirInstData::Alloca(alloca_ty),
        "phi" => TirInstData::Phi {
            incoming: phi_incoming,
            incoming_values: phi_incoming_values,
            parse_error: phi_parse_error,
        },
        "extractvalue" | "insertvalue" => TirInstData::Aggregate(aggregate_indices),
        "extractelement" | "insertelement" | "shufflevector" => TirInstData::Element {
            diag_line,
            shuffle_mask,
        },
        "bitcast" => TirInstData::Bitcast {
            destination: bitcast.map(|parsed| parsed.1),
            identity: identity_ptr_bitcast.is_some(),
        },
        "select" => TirInstData::Select(select_arms),
        _ => TirInstData::Plain,
    };
    cur_insts.push(TirInst {
        result,
        result_ty,
        uses,
        operands,
        opcode: TirOpcode::new(opcode),
        data: Box::new(TirInstDetails {
            fast_math,
            payload: data,
        }),
    });
}
