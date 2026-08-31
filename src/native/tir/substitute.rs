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
        LlValue::IntToPtr { source, .. } => substitute_typed_value(source, substitutions),
        LlValue::Global(_)
        | LlValue::Bool(_)
        | LlValue::Int(_)
        | LlValue::SignedInt(_)
        | LlValue::Hex(_)
        | LlValue::Float(_)
        | LlValue::Float32Bits(_)
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
        LlValue::IntToPtr { source, .. } => local_uses(&source.value, uses),
        LlValue::Global(_)
        | LlValue::Bool(_)
        | LlValue::Int(_)
        | LlValue::SignedInt(_)
        | LlValue::Hex(_)
        | LlValue::Float(_)
        | LlValue::Float32Bits(_)
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
        LlValue::Float32Bits(bits) => format!("f0x{bits:08X}"),
        LlValue::HalfBits(bits) => format!("0xH{bits:04X}"),
        LlValue::BFloatBits(bits) => format!("0xR{bits:04X}"),
        LlValue::Zero if matches!(value.ty, LlType::Ptr(_)) => "null".to_string(),
        LlValue::Zero => "zeroinitializer".to_string(),
        LlValue::Undef => "undef".to_string(),
        LlValue::Vector(_) | LlValue::Array(_) | LlValue::Struct(_) | LlValue::Splat(_) => {
            "zeroinitializer".to_string()
        }
        LlValue::Gep(_) => "null".to_string(),
        LlValue::IntToPtr { .. } => "null".to_string(),
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
    if let Some(stored_uses) = &mut inst.uses {
        let mut uses = Vec::new();
        for name in &*stored_uses {
            match substitutions.get(name) {
                Some(replacement) => local_uses(&replacement.value, &mut uses),
                None => uses.push(name.clone()),
            }
        }
        uses.dedup();
        *stored_uses = uses;
    }

    for operand in &mut inst.operands {
        substitute_operand(operand, substitutions);
    }
    match &mut inst.data.payload {
        TirInstData::Compare { rest, .. } => {
            if let Some(rest) = rest {
                *rest = crate::native::cfg::rename_tokens(rest, tokens);
            }
        }
        TirInstData::Memory { load, store, .. } => {
            if let Some(load) = load {
                substitute_typed_value(&mut load.ptr, substitutions);
            }
            if let Some((object, pointer)) = store.as_deref_mut() {
                substitute_typed_value(object, substitutions);
                substitute_typed_value(pointer, substitutions);
            }
        }
        TirInstData::Gep { parsed, .. } => {
            if let Some(gep) = parsed {
                substitute_gep(gep, substitutions);
            }
        }
        TirInstData::Call {
            parsed,
            void_line,
            value_error,
            alias_override,
            emit_scan,
            ..
        } => {
            if let Some(call) = parsed {
                substitute_call(call, substitutions);
            }
            if let Some(call) = alias_override {
                substitute_call(call, substitutions);
            }
            if let EmitScanData::Owned(result) = emit_scan {
                if let Ok(call) = result.as_mut() {
                    substitute_call(call, substitutions);
                }
            }
            for text in [void_line, value_error].into_iter().flatten() {
                *text = crate::native::cfg::rename_tokens(text, tokens);
            }
        }
        TirInstData::Phi {
            incoming,
            incoming_values,
            ..
        } => {
            if let Some((_, incoming)) = incoming {
                for (value, _) in incoming {
                    substitute_value(value, substitutions);
                }
            }
            if let Some(values) = incoming_values {
                for value in values {
                    substitute_value(value, substitutions);
                }
            }
        }
        TirInstData::Element { diag_line, .. } => {
            if let Some(line) = diag_line {
                *line = crate::native::cfg::rename_tokens(line, tokens);
            }
        }
        TirInstData::Bitcast { .. } => {}
        TirInstData::Select(arms) => {
            if let Some((true_value, false_value)) = arms.as_deref_mut() {
                substitute_typed_value(true_value, substitutions);
                substitute_typed_value(false_value, substitutions);
            }
        }
        TirInstData::Plain | TirInstData::Alloca(_) | TirInstData::Aggregate(_) => {}
    }
}

fn substitute_switch(switch: &mut LlSwitch, substitutions: &Substitutions) {
    substitute_typed_value(&mut switch.selector, substitutions);
    for (value, _) in &mut switch.cases {
        substitute_value(value, substitutions);
    }
}

impl TirBlock {
    fn substitute_values_impl(&mut self, substitutions: &Substitutions, include_phis: bool) {
        if substitutions.is_empty() {
            return;
        }
        let tokens = token_map(substitutions);
        for inst in &mut self.insts {
            if include_phis || !inst.is_phi() {
                substitute_inst(inst, substitutions, &tokens);
            }
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

    /// Replace every use of a mapped SSA local with its typed caller value.
    pub(in crate::native) fn substitute_values(&mut self, substitutions: &Substitutions) {
        self.substitute_values_impl(substitutions, true);
    }

    /// Replace ordinary and terminator uses while leaving phi incoming edges for the caller's
    /// dedicated edge rewrite. Structural dispatch construction uses this when a live value crosses
    /// the new dispatch merge: destination phis are already funnelled by their exact predecessor
    /// contract, while non-phi consumers need the newly constructed dominating value.
    pub(in crate::native) fn substitute_non_phi_values(&mut self, substitutions: &Substitutions) {
        self.substitute_values_impl(substitutions, false);
    }
}
