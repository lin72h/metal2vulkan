//! Deep typed-value substitution for pre-emit inlining.
//!
//! [`TirBlock::rename`] is the right primitive when every replacement remains an SSA `%name`.
//! Ordinary helper calls may bind parameters to literals, however, and a value-returning helper may
//! return either an SSA value or a literal. This module performs that more general edit directly on
//! every typed carrier. It never decides which helper is eligible; it only applies a caller-supplied
//! `%local -> TypedValue` map.

use super::*;
use crate::native::ir::{LlGep, LlValue, TypedValue};
use crate::native::parse::{LlCall, LlSwitch};
use std::collections::HashMap;

type Substitutions = HashMap<String, TypedValue>;

fn substitute_typed_value(value: &mut TypedValue, substitutions: &Substitutions) {
    if let LlValue::Local(name) = &value.value {
        if let Some(replacement) = substitutions.get(name) {
            *value = replacement.clone();
            return;
        }
    }
    substitute_value(&mut value.value, substitutions);
}

fn substitute_value(value: &mut LlValue, substitutions: &Substitutions) {
    match value {
        LlValue::Local(name) => {
            if let Some(replacement) = substitutions.get(name) {
                *value = replacement.value.clone();
            }
        }
        LlValue::Vector(values) | LlValue::Array(values) | LlValue::Struct(values) => {
            for value in values {
                substitute_typed_value(value, substitutions);
            }
        }
        LlValue::Splat(value) => substitute_typed_value(value, substitutions),
        LlValue::Gep(gep) => substitute_gep(gep, substitutions),
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

fn substitute_gep(gep: &mut LlGep, substitutions: &Substitutions) {
    substitute_typed_value(&mut gep.base, substitutions);
    for index in &mut gep.indices {
        substitute_typed_value(index, substitutions);
    }
}

fn substitute_call(call: &mut LlCall, substitutions: &Substitutions) {
    for argument in &mut call.args {
        substitute_typed_value(argument, substitutions);
    }
}

fn substitute_operand(operand: &mut TirOperand, substitutions: &Substitutions) {
    let Some(mut value) = operand.as_typed_value() else {
        return;
    };
    substitute_typed_value(&mut value, substitutions);
    *operand = operand_from_typed_value(&value);
}

fn local_uses(value: &LlValue, uses: &mut Vec<String>) {
    match value {
        LlValue::Local(name) => uses.push(name.clone()),
        LlValue::Vector(values) | LlValue::Array(values) | LlValue::Struct(values) => {
            for value in values {
                local_uses(&value.value, uses);
            }
        }
        LlValue::Splat(value) => local_uses(&value.value, uses),
        LlValue::Gep(gep) => {
            local_uses(&gep.base.value, uses);
            for index in &gep.indices {
                local_uses(&index.value, uses);
            }
        }
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

fn replacement_token(value: &TypedValue) -> String {
    if let Some(rendered) = crate::native::render::render_value(&value.value) {
        return rendered;
    }
    // These strings back only diagnostics and the redundant string half of typed terminators.
    // Emission consumes the structured value above. Keep the token non-local so def/use analyses
    // never mistake a replaced literal for an SSA edge.
    match &value.value {
        LlValue::Hex(bits) => format!("0x{bits:016X}"),
        LlValue::Float(number) => format!("{number:e}"),
        LlValue::HalfBits(bits) => format!("0xH{bits:04X}"),
        LlValue::BFloatBits(bits) => format!("0xR{bits:04X}"),
        LlValue::Zero if matches!(value.ty, LlType::Ptr(_)) => "null".to_string(),
        LlValue::Zero => "zeroinitializer".to_string(),
        LlValue::Undef => "undef".to_string(),
        LlValue::Vector(_) | LlValue::Array(_) | LlValue::Struct(_) | LlValue::Splat(_) => {
            "zeroinitializer".to_string()
        }
        LlValue::Gep(_) => "null".to_string(),
        LlValue::Local(_)
        | LlValue::Global(_)
        | LlValue::Bool(_)
        | LlValue::Int(_)
        | LlValue::SignedInt(_) => unreachable!("injectively rendered above"),
    }
}

fn token_map(substitutions: &Substitutions) -> HashMap<String, String> {
    substitutions
        .iter()
        .map(|(name, value)| (name.clone(), replacement_token(value)))
        .collect()
}

fn substitute_inst(
    inst: &mut TirInst,
    substitutions: &Substitutions,
    tokens: &HashMap<String, String>,
) {
    let mut uses = Vec::new();
    for name in &inst.uses {
        match substitutions.get(name) {
            Some(replacement) => local_uses(&replacement.value, &mut uses),
            None => uses.push(name.clone()),
        }
    }
    uses.dedup();
    inst.uses = uses;

    for operand in &mut inst.operands {
        substitute_operand(operand, substitutions);
    }
    if let Some(gep) = &mut inst.gep {
        substitute_gep(gep, substitutions);
    }
    if let Some(call) = &mut inst.call {
        substitute_call(call, substitutions);
    }
    if let Some((_, incoming)) = &mut inst.phi_incoming {
        for (value, _) in incoming {
            substitute_value(value, substitutions);
        }
    }
    if let Some((value, _)) = &mut inst.bitcast {
        substitute_typed_value(value, substitutions);
    }
    if let Some((result, base)) = &mut inst.identity_ptr_bitcast {
        if let Some(replacement) = tokens.get(result) {
            *result = replacement.clone();
        }
        if let Some(replacement) = tokens.get(base) {
            *base = replacement.clone();
        }
    }
    if let Some(values) = &mut inst.phi_incoming_values {
        for value in values {
            substitute_value(value, substitutions);
        }
    }
    if let Some((true_value, false_value)) = &mut inst.select_arms {
        substitute_typed_value(true_value, substitutions);
        substitute_typed_value(false_value, substitutions);
    }
    if let Some(load) = &mut inst.load {
        substitute_typed_value(&mut load.ptr, substitutions);
    }
    if let Some((object, pointer)) = &mut inst.store {
        substitute_typed_value(object, substitutions);
        substitute_typed_value(pointer, substitutions);
    }
    if let Some(call) = &mut inst.alias_call {
        substitute_call(call, substitutions);
    }
    if let Some(Ok(call)) = &mut inst.emit_scan_call {
        substitute_call(call, substitutions);
    }
    for text in [
        &mut inst.diag_line,
        &mut inst.void_call_line,
        &mut inst.icmp_rest,
    ]
    .into_iter()
    .flatten()
    {
        *text = crate::native::cfg::rename_tokens(text, tokens);
    }
}

fn substitute_switch(switch: &mut LlSwitch, substitutions: &Substitutions) {
    substitute_typed_value(&mut switch.selector, substitutions);
    for (value, _) in &mut switch.cases {
        substitute_value(value, substitutions);
    }
}

impl TirBlock {
    /// Replace every use of a mapped SSA local with its typed caller value.
    pub(in crate::native) fn substitute_values(&mut self, substitutions: &Substitutions) {
        if substitutions.is_empty() {
            return;
        }
        let tokens = token_map(substitutions);
        for inst in &mut self.insts {
            substitute_inst(inst, substitutions, &tokens);
        }
        match &mut self.terminator {
            TirTerminator::Br(_) | TirTerminator::Ret(None) | TirTerminator::Unreachable => {}
            TirTerminator::BrCond { cond, .. } => {
                if let Some(replacement) = tokens.get(cond) {
                    *cond = replacement.clone();
                }
            }
            TirTerminator::Switch { selector, .. } => {
                if let Some(replacement) = tokens.get(selector) {
                    *selector = replacement.clone();
                }
            }
            TirTerminator::Ret(Some(value)) => {
                if let Some(replacement) = tokens.get(value) {
                    *value = replacement.clone();
                }
            }
        }
        if let RetEmit::Value(value) = &mut self.ret {
            substitute_typed_value(value, substitutions);
        }
        if let Some(switch) = &mut self.switch {
            substitute_switch(switch, substitutions);
        }
    }
}
