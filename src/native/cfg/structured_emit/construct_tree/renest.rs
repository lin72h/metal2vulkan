//! Whole-region structured role materialization for the R1 fixture gate.
//!
//! The immutable ownership plan is lowered once to a regional dispatcher construct: one loop owns
//! one switch, every original body is one switch case, and original edges select the next case.
//! Original phi values are carried as loop-header state and updated from their exact source edges.
//! This is not the post-SPIR-V whole-function relooper: it consumes typed source blocks before
//! emission, uses the bounded reject closure supplied by the caller, and does not spill values.

use super::*;
use crate::native::ir::{LlType, LlValue, TypedValue};
use crate::native::tir;
use std::collections::{BTreeSet, HashMap, HashSet};

const PREFIX: &str = "%metal2vulkan.ct.";
const PRE: &str = "%metal2vulkan.ct.pre";
const HEADER: &str = "%metal2vulkan.ct.header";
const TEST: &str = "%metal2vulkan.ct.test";
const DISPATCH: &str = "%metal2vulkan.ct.dispatch";
const INVALID: &str = "%metal2vulkan.ct.invalid";
const MERGE: &str = "%metal2vulkan.ct.merge";
const CONTINUE: &str = "%metal2vulkan.ct.continue";
const EXIT: &str = "%metal2vulkan.ct.exit";
const PC: &str = "%metal2vulkan.ct.pc";
const NEXT_PC: &str = "%metal2vulkan.ct.nextpc";
const DONE: &str = "%metal2vulkan.ct.done";
const ROUTE: &str = "%metal2vulkan.ct.route";

#[derive(Clone, Debug)]
struct PhiSlot {
    ty: LlType,
    host: usize,
    incoming: HashMap<usize, LlValue>,
    current: String,
    next: String,
}

#[derive(Clone, Debug)]
struct ValueSlot {
    ty: LlType,
    owner: usize,
    original: String,
    current: String,
    next: String,
}

fn passthrough(name: &str, target: &str) -> BodyBlock {
    synthetic_block(
        name.to_string(),
        vec![format!("br label {target}")],
        BlockRole::Normal,
    )
}

fn route_passthrough(name: &str, target: &str) -> BodyBlock {
    synthetic_block(
        name.to_string(),
        vec![format!("br label {target}")],
        BlockRole::ConstructTreeRoute,
    )
}

fn carrier_with_phis(
    name: &str,
    target: &str,
    phis: &[(String, LlType, Vec<(LlValue, String)>)],
    role: BlockRole,
) -> Result<BodyBlock, String> {
    let mut carrier =
        tir::lower_block_carrier(name, &[format!("br label {target}")], &HashMap::new())
            .ok_or_else(|| format!("construct-tree:gateway-carrier name={name}"))?;
    for (result, ty, incoming) in phis {
        carrier.push_value_phi(result, ty, incoming);
    }
    Ok(BodyBlock {
        name: name.to_string(),
        role,
        typed: Some(carrier.into()),
    })
}

fn renamed_value(value: &LlValue, map: &HashMap<String, String>) -> LlValue {
    tir::renamed_llvalue(value, map)
}

fn collect_value_locals(value: &LlValue, out: &mut Vec<String>) {
    match value {
        LlValue::Local(name) => out.push(name.clone()),
        LlValue::Vector(values) | LlValue::Array(values) | LlValue::Struct(values) => {
            for value in values {
                collect_value_locals(&value.value, out);
            }
        }
        LlValue::Splat(value) => collect_value_locals(&value.value, out),
        LlValue::Gep(gep) => {
            collect_value_locals(&gep.base.value, out);
            for index in &gep.indices {
                collect_value_locals(&index.value, out);
            }
        }
        LlValue::IntToPtr { source, .. } => collect_value_locals(&source.value, out),
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

fn substitute_cross_slot_value(
    value: &mut LlValue,
    value_slots: &HashMap<&str, &ValueSlot>,
    source: usize,
) {
    match value {
        LlValue::Local(name) => {
            let Some(slot) = value_slots.get(name.as_str()) else {
                return;
            };
            if slot.owner != source {
                *value = LlValue::Local(slot.current.clone());
            }
        }
        LlValue::Vector(values) | LlValue::Array(values) | LlValue::Struct(values) => {
            for value in values {
                substitute_cross_slot_value(&mut value.value, value_slots, source);
            }
        }
        LlValue::Splat(value) => {
            substitute_cross_slot_value(&mut value.value, value_slots, source);
        }
        LlValue::Gep(gep) => {
            substitute_cross_slot_value(&mut gep.base.value, value_slots, source);
            for index in &mut gep.indices {
                substitute_cross_slot_value(&mut index.value, value_slots, source);
            }
        }
        LlValue::IntToPtr {
            source: operand, ..
        } => substitute_cross_slot_value(&mut operand.value, value_slots, source),
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

fn terminator_value_uses(terminator: &tir::TirTerminator) -> Vec<String> {
    match terminator {
        tir::TirTerminator::Br(_)
        | tir::TirTerminator::Ret(None)
        | tir::TirTerminator::Unreachable => Vec::new(),
        tir::TirTerminator::BrCond { cond, .. }
        | tir::TirTerminator::Switch { selector: cond, .. }
        | tir::TirTerminator::Ret(Some(cond)) => vec![cond.clone()],
    }
}

/// Materialize the whole explicit reject closure into distinct loop/switch entry, merge, and continue
/// roles. The result is proved admissible by the ordinary structured planner before it is returned,
/// but remains pre-plan so the ordinary emitter can consume it exactly once. This R1 API is
/// intentionally unwired from admission; R2+ supplies live class-derived claims behind the
/// reject gate.
pub(in crate::native) fn materialize_construct_tree_roles(
    blocks: &[BodyBlock],
    plan: &ConstructTreePlan,
) -> Result<Vec<BodyBlock>, String> {
    if blocks.is_empty() {
        return Err("construct-tree:empty-region".to_string());
    }
    if blocks.len() != plan.owners.len() {
        return Err(format!(
            "construct-tree:block-plan-mismatch blocks={} owners={}",
            blocks.len(),
            plan.owners.len()
        ));
    }
    if blocks.iter().any(|block| block.name.starts_with(PREFIX)) {
        return Err("construct-tree:reserved-block-prefix".to_string());
    }

    let names: HashMap<&str, usize> = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.name.as_str(), index))
        .collect();
    if names.len() != blocks.len() {
        return Err("construct-tree:duplicate-block-name".to_string());
    }
    let planned_edges: HashSet<(usize, usize)> = plan
        .routes
        .iter()
        .map(|route| (route.edge.from, route.edge.to))
        .collect();
    if planned_edges.len() != plan.routes.len() {
        return Err("construct-tree:duplicate-edge".to_string());
    }
    let routes: HashMap<(usize, usize), &ConstructRoute> = plan
        .routes
        .iter()
        .map(|route| ((route.edge.from, route.edge.to), route))
        .collect();

    let mut successors = Vec::with_capacity(blocks.len());
    for (source, block) in blocks.iter().enumerate() {
        let carrier = block
            .typed
            .as_ref()
            .ok_or_else(|| format!("construct-tree:missing-carrier block={}", block.name))?;
        if carrier
            .insts
            .iter()
            .filter_map(|inst| inst.result.as_deref())
            .any(|result| result.starts_with(PREFIX))
        {
            return Err(format!(
                "construct-tree:reserved-value-prefix block={}",
                block.name
            ));
        }
        if matches!(carrier.terminator, tir::TirTerminator::Ret(Some(_))) {
            return Err(format!(
                "construct-tree:nonvoid-return block={}",
                block.name
            ));
        }
        let mut block_successors = Vec::new();
        for target in carrier.terminator.successors() {
            let Some(&target_index) = names.get(target) else {
                return Err(format!(
                    "construct-tree:unknown-successor from={} to={target}",
                    block.name
                ));
            };
            if !planned_edges.contains(&(source, target_index)) {
                return Err(format!(
                    "construct-tree:unplanned-source-edge from={} to={target}",
                    block.name
                ));
            }
            if !block_successors.contains(&target_index) {
                block_successors.push(target_index);
            }
        }
        successors.push(block_successors);
    }
    let actual_edges: HashSet<(usize, usize)> = successors
        .iter()
        .enumerate()
        .flat_map(|(source, targets)| targets.iter().map(move |target| (source, *target)))
        .collect();
    if actual_edges != planned_edges {
        return Err(format!(
            "construct-tree:edge-set-mismatch actual={} planned={}",
            actual_edges.len(),
            planned_edges.len()
        ));
    }

    // Collect source phis before mutating carriers. Every incoming predecessor is an original edge.
    let mut slots = Vec::new();
    let mut rename = HashMap::new();
    for (host, block) in blocks.iter().enumerate() {
        let carrier = block.typed.as_ref().expect("checked above");
        for inst in &carrier.insts {
            let Some((ty, incoming)) = &inst.phi_incoming else {
                continue;
            };
            let result = inst
                .result
                .as_ref()
                .ok_or_else(|| format!("construct-tree:phi-without-result block={}", block.name))?;
            let slot_index = slots.len();
            let current = format!("{PREFIX}slot.{slot_index}");
            let next = format!("{PREFIX}nextslot.{slot_index}");
            rename.insert(result.clone(), current.clone());
            let mut routed = HashMap::new();
            for (value, predecessor) in incoming {
                let Some(&source) = names.get(predecessor.as_str()) else {
                    return Err(format!(
                        "construct-tree:phi-unknown-predecessor block={} pred={predecessor}",
                        block.name
                    ));
                };
                if !planned_edges.contains(&(source, host)) {
                    return Err(format!(
                        "construct-tree:phi-nonedge block={} pred={predecessor}",
                        block.name
                    ));
                }
                if routed.insert(source, value.clone()).is_some() {
                    return Err(format!(
                        "construct-tree:phi-duplicate-predecessor block={} pred={predecessor}",
                        block.name
                    ));
                }
            }
            slots.push(PhiSlot {
                ty: ty.clone(),
                host,
                incoming: routed,
                current,
                next,
            });
        }
    }
    for slot in &mut slots {
        for value in slot.incoming.values_mut() {
            *value = renamed_value(value, &rename);
        }
    }

    // Non-phi values crossing between original cases are carried explicitly through typed state slots.
    // Pointer values remain excluded: a raw `ptr` slot would discard provenance/pointee information that
    // the later pointer path still depends on.
    let mut def_block = HashMap::new();
    for (index, block) in blocks.iter().enumerate() {
        for inst in &block.typed.as_ref().expect("checked above").insts {
            if !inst.is_phi() {
                if let Some(result) = &inst.result {
                    def_block.insert(result.clone(), (index, inst.result_ty.clone()));
                }
            }
        }
    }
    let mut cross_values = BTreeSet::new();
    for (index, block) in blocks.iter().enumerate() {
        let carrier = block.typed.as_ref().expect("checked above");
        for inst in &block.typed.as_ref().expect("checked above").insts {
            if inst.is_phi() {
                continue;
            }
            for name in &inst.uses {
                if def_block
                    .get(name.as_str())
                    .is_some_and(|(owner, _)| *owner != index)
                {
                    cross_values.insert(name.clone());
                }
            }
        }
        for name in terminator_value_uses(&carrier.terminator) {
            if def_block
                .get(name.as_str())
                .is_some_and(|(owner, _)| *owner != index)
            {
                cross_values.insert(name);
            }
        }
    }
    for slot in &slots {
        for (&source, value) in &slot.incoming {
            let mut names = Vec::new();
            collect_value_locals(value, &mut names);
            for name in names {
                if def_block
                    .get(name.as_str())
                    .is_some_and(|(owner, _)| *owner != source)
                    && !rename.values().any(|slot| slot.as_str() == name.as_str())
                {
                    cross_values.insert(name);
                }
            }
        }
    }
    let mut value_slots = Vec::new();
    for original in cross_values {
        let Some((owner, ty)) = def_block.get(&original) else {
            continue;
        };
        let Some(ty) = ty else {
            return Err(format!(
                "construct-tree:cross-case-untyped-value owner={} value={original}",
                blocks[*owner].name
            ));
        };
        if matches!(ty, LlType::Ptr(_)) {
            return Err(format!(
                "construct-tree:cross-case-pointer-value owner={} value={original}",
                blocks[*owner].name
            ));
        }
        let slot_index = value_slots.len();
        value_slots.push(ValueSlot {
            ty: ty.clone(),
            owner: *owner,
            original,
            current: format!("{PREFIX}vslot.{slot_index}"),
            next: format!("{PREFIX}nextvslot.{slot_index}"),
        });
    }
    let value_slot_by_name: HashMap<&str, &ValueSlot> = value_slots
        .iter()
        .map(|slot| (slot.original.as_str(), slot))
        .collect();
    for slot in &mut slots {
        for (&source, value) in &mut slot.incoming {
            substitute_cross_slot_value(value, &value_slot_by_name, source);
        }
    }

    let done_state = blocks.len() as u64;
    let mut out = vec![passthrough(PRE, HEADER)];
    let mut header_phis = vec![(
        PC.to_string(),
        LlType::Int(32),
        vec![
            (LlValue::Int(0), PRE.to_string()),
            (LlValue::Local(NEXT_PC.to_string()), CONTINUE.to_string()),
        ],
    )];
    for slot in &slots {
        header_phis.push((
            slot.current.clone(),
            slot.ty.clone(),
            vec![
                (LlValue::Undef, PRE.to_string()),
                (LlValue::Local(slot.next.clone()), CONTINUE.to_string()),
            ],
        ));
    }
    for slot in &value_slots {
        header_phis.push((
            slot.current.clone(),
            slot.ty.clone(),
            vec![
                (LlValue::Undef, PRE.to_string()),
                (LlValue::Local(slot.next.clone()), CONTINUE.to_string()),
            ],
        ));
    }
    out.push(carrier_with_phis(
        HEADER,
        TEST,
        &header_phis,
        BlockRole::Normal,
    )?);
    out.push(synthetic_block(
        TEST.to_string(),
        vec![
            format!("{DONE} = icmp eq i32 {PC}, {done_state}"),
            format!("br i1 {DONE}, label {EXIT}, label {DISPATCH}"),
        ],
        BlockRole::Normal,
    ));
    let switch_cases = blocks
        .iter()
        .enumerate()
        .map(|(state, block)| format!("i32 {state}, label {}", block.name))
        .collect::<Vec<_>>()
        .join(" ");
    out.push(synthetic_block(
        DISPATCH.to_string(),
        vec![format!(
            "switch i32 {PC}, label {INVALID} [ {switch_cases} ]"
        )],
        BlockRole::ConstructTreeRoute,
    ));

    let mut pc_updates = Vec::with_capacity(blocks.len());
    let mut phi_updates = slots
        .iter()
        .map(|_| Vec::with_capacity(blocks.len()))
        .collect::<Vec<_>>();
    let mut value_update_incoming = value_slots
        .iter()
        .map(|_| Vec::with_capacity(blocks.len()))
        .collect::<Vec<_>>();
    for (case_index, original) in blocks.iter().enumerate() {
        let mut case = original.clone();
        let carrier = std::sync::Arc::make_mut(case.typed.as_mut().expect("checked above"));
        carrier.rename(&rename);
        // Most cross-case values are irrelevant to any one case. Building the full value-slot map
        // for every block made this bounded transform allocate O(cases * slots) cloned names and
        // types before it touched a single operand. Restrict the substitution map to actual uses in
        // this carrier; the deep typed substitution below remains the sole mutation primitive.
        let mut used_names = carrier
            .insts
            .iter()
            .filter(|inst| !inst.is_phi())
            .flat_map(|inst| inst.uses.iter().cloned())
            .collect::<HashSet<_>>();
        used_names.extend(terminator_value_uses(&carrier.terminator));
        let mut substitutions = HashMap::new();
        for name in used_names {
            let Some(slot) = value_slot_by_name.get(name.as_str()) else {
                continue;
            };
            if slot.owner == case_index {
                continue;
            }
            substitutions.insert(
                slot.original.clone(),
                TypedValue {
                    ty: slot.ty.clone(),
                    value: LlValue::Local(slot.current.clone()),
                },
            );
        }
        carrier.substitute_values(&substitutions);
        carrier.insts.retain(|inst| !inst.is_phi());
        let targets = &successors[case_index];
        let join = format!("{PREFIX}casejoin.{case_index}");
        let mut route_tails = Vec::with_capacity(targets.len());
        let mut route_blocks = Vec::new();
        if targets.is_empty() {
            carrier.set_unconditional_branch(&join);
        }
        for (edge_index, target) in targets.iter().enumerate() {
            let route = routes.get(&(case_index, *target)).ok_or_else(|| {
                format!(
                    "construct-tree:missing-route from={} to={}",
                    original.name, blocks[*target].name
                )
            })?;
            let mut route_names = Vec::with_capacity(route.exits.len() + route.enters.len() + 1);
            route_names.extend(route.exits.iter().enumerate().map(|(step, construct)| {
                format!("{ROUTE}.{case_index}.{edge_index}.exit.{step}.{construct}")
            }));
            route_names.extend(route.enters.iter().enumerate().map(|(step, construct)| {
                format!("{ROUTE}.{case_index}.{edge_index}.enter.{step}.{construct}")
            }));
            route_names.push(format!("{ROUTE}.{case_index}.{edge_index}.select"));
            carrier.redirect_successor(&blocks[*target].name, &route_names[0]);
            for (step, name) in route_names.iter().enumerate() {
                let next = route_names
                    .get(step + 1)
                    .map(String::as_str)
                    .unwrap_or(&join);
                route_blocks.push(route_passthrough(name, next));
            }
            route_tails.push(route_names.last().expect("terminal route step").clone());
        }
        out.push(case);
        out.extend(route_blocks);

        let pc_update = match targets.as_slice() {
            [] => LlValue::Int(done_state),
            [target] => LlValue::Int(*target as u64),
            _ => LlValue::Local(format!("{PREFIX}casepc.{case_index}")),
        };
        let mut join_phis = Vec::new();
        if targets.len() > 1 {
            join_phis.push((
                format!("{PREFIX}casepc.{case_index}"),
                LlType::Int(32),
                targets
                    .iter()
                    .enumerate()
                    .map(|(edge_index, target)| {
                        (
                            LlValue::Int(*target as u64),
                            route_tails[edge_index].clone(),
                        )
                    })
                    .collect(),
            ));
        }
        let mut slot_updates = Vec::with_capacity(slots.len());
        for (slot_index, slot) in slots.iter().enumerate() {
            let edge_value = |target: usize| -> Result<LlValue, String> {
                if target != slot.host {
                    return Ok(LlValue::Local(slot.current.clone()));
                }
                slot.incoming.get(&case_index).cloned().ok_or_else(|| {
                    format!(
                        "construct-tree:phi-missing-source block={} pred={}",
                        blocks[slot.host].name, original.name
                    )
                })
            };
            let update = match targets.as_slice() {
                [] => LlValue::Local(slot.current.clone()),
                [target] => edge_value(*target)?,
                _ if targets.contains(&slot.host) => {
                    let result = format!("{PREFIX}caseslot.{case_index}.{slot_index}");
                    join_phis.push((
                        result.clone(),
                        slot.ty.clone(),
                        targets
                            .iter()
                            .enumerate()
                            .map(|(edge_index, target)| {
                                Ok((edge_value(*target)?, route_tails[edge_index].clone()))
                            })
                            .collect::<Result<Vec<_>, String>>()?,
                    ));
                    LlValue::Local(result)
                }
                _ => LlValue::Local(slot.current.clone()),
            };
            slot_updates.push(update);
        }
        let case_value_updates: Vec<_> = value_slots
            .iter()
            .map(|slot| {
                if slot.owner == case_index {
                    LlValue::Local(slot.original.clone())
                } else {
                    LlValue::Local(slot.current.clone())
                }
            })
            .collect();
        out.push(carrier_with_phis(
            &join,
            MERGE,
            &join_phis,
            BlockRole::Normal,
        )?);
        pc_updates.push((pc_update, join.clone()));
        for (incoming, value) in phi_updates.iter_mut().zip(slot_updates) {
            incoming.push((value, join.clone()));
        }
        for (incoming, value) in value_update_incoming.iter_mut().zip(case_value_updates) {
            incoming.push((value, join.clone()));
        }
    }

    out.push(passthrough(INVALID, MERGE));
    let mut merge_phis = vec![(
        NEXT_PC.to_string(),
        LlType::Int(32),
        pc_updates
            .into_iter()
            .chain(std::iter::once((
                LlValue::Int(done_state),
                INVALID.to_string(),
            )))
            .collect(),
    )];
    for (slot, incoming) in slots.iter().zip(phi_updates) {
        merge_phis.push((
            slot.next.clone(),
            slot.ty.clone(),
            incoming
                .into_iter()
                .chain(std::iter::once((
                    LlValue::Local(slot.current.clone()),
                    INVALID.to_string(),
                )))
                .collect(),
        ));
    }
    for (slot, incoming) in value_slots.iter().zip(value_update_incoming) {
        merge_phis.push((
            slot.next.clone(),
            slot.ty.clone(),
            incoming
                .into_iter()
                .chain(std::iter::once((
                    LlValue::Local(slot.current.clone()),
                    INVALID.to_string(),
                )))
                .collect(),
        ));
    }
    out.push(carrier_with_phis(
        MERGE,
        CONTINUE,
        &merge_phis,
        BlockRole::LMerge,
    )?);
    out.push(passthrough(CONTINUE, HEADER));
    out.push(synthetic_block(
        EXIT.to_string(),
        vec!["ret void".to_string()],
        BlockRole::Normal,
    ));

    if out.len() > plan.block_bound {
        return Err(format!(
            "construct-tree:materialization-bound blocks={} bound={}",
            out.len(),
            plan.block_bound
        ));
    }
    if crate::env_vars::why() {
        let instructions = out
            .iter()
            .filter_map(|block| block.typed.as_ref())
            .map(|block| block.insts.len())
            .sum::<usize>();
        let phis = out
            .iter()
            .filter_map(|block| block.typed.as_ref())
            .flat_map(|block| &block.insts)
            .filter(|inst| inst.is_phi())
            .count();
        let phi_incoming = out
            .iter()
            .filter_map(|block| block.typed.as_ref())
            .flat_map(|block| &block.insts)
            .filter_map(|inst| inst.phi_incoming.as_ref())
            .map(|(_, incoming)| incoming.len())
            .sum::<usize>();
        eprintln!(
            "WHY-CONSTRUCT-TREE materialized blocks={} instructions={instructions} phis={phis} phi-incoming={phi_incoming} source-phis={} cross-values={}",
            out.len(),
            slots.len(),
            value_slots.len()
        );
    }
    if super::super::structured_plan(&out).is_none() {
        let reason =
            super::super::structured_reject_reason(&out).unwrap_or_else(|| "unknown".to_string());
        return Err(format!("construct-tree:role-plan-reject reason={reason}"));
    }
    Ok(out)
}

/// Materialize the bounded construct tree and return the ordinary planner's finalized result. Kept as
/// the direct R1 proof API; production wiring can consume the returned maps without re-planning.
pub(in crate::native) fn renest_construct_tree(
    blocks: &[BodyBlock],
    plan: &ConstructTreePlan,
) -> Result<super::super::StructuredPlan, String> {
    let out = materialize_construct_tree_roles(blocks, plan)?;
    super::super::structured_plan(&out)
        .ok_or_else(|| "construct-tree:role-plan-nondeterministic".to_string())
}
