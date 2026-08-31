use super::location_index;
use std::collections::{HashMap, HashSet};

pub(super) fn location_index_with_static(
    body: &str,
    fallback: u32,
    static_int_globals: &HashMap<String, u32>,
) -> u32 {
    location_index_global(body)
        .and_then(|global| static_int_globals.get(&global).copied())
        .unwrap_or_else(|| location_index(body, fallback))
}

fn location_index_global(body: &str) -> Option<String> {
    global_after_marker(body, "air.location_index")
}

fn global_after_marker(body: &str, marker: &str) -> Option<String> {
    let marker = format!("!\"{marker}\"");
    let pos = body.find(&marker)?;
    let after = &body[pos + marker.len()..];
    parse_global_name(after)
}

fn parse_global_name(s: &str) -> Option<String> {
    let at = s.find('@')?;
    let name = s[at..]
        .chars()
        .take_while(|c| !c.is_whitespace() && !matches!(*c, ',' | ')' | '(' | '[' | ']'))
        .collect::<String>();
    if name.len() > 1 {
        Some(name)
    } else {
        None
    }
}

#[derive(Clone, Debug)]
enum StaticValue {
    Bool(bool),
    Int(u64),
    Vector(Vec<u64>),
}

#[derive(Clone, Debug)]
pub(crate) enum StaticIntValue {
    Scalar(u32),
    Vector(Vec<u32>),
}

/// Best-effort evaluator for AIR static initializers that materialize default function-constant
/// integer globals. Unknown expressions are ignored so dynamic metadata falls back to its immediate
/// location index rather than guessing.
pub(super) fn static_init_int_global_values(ll: &str) -> HashMap<String, u32> {
    static_init_global_values(ll)
        .into_iter()
        .filter_map(|(global, value)| match value {
            StaticValue::Bool(value) => Some((global, u32::from(value))),
            StaticValue::Int(value) => Some((global, value as u32)),
            StaticValue::Vector(_) => None,
        })
        .collect()
}

fn static_init_global_values(ll: &str) -> HashMap<String, StaticValue> {
    let mut globals = parse_static_global_initializers(ll);
    let mut unknown_stores = HashSet::new();
    let mut env: HashMap<String, StaticValue> = HashMap::new();
    let mut in_static_init = false;

    for raw in ll.lines() {
        let line = raw.split(';').next().unwrap_or(raw).trim();
        if line.starts_with("define ") {
            in_static_init = line.contains("@_GLOBAL__sub_I");
            env.clear();
            continue;
        }
        if !in_static_init {
            continue;
        }
        if line == "}" {
            in_static_init = false;
            continue;
        }
        if line.is_empty() || line.ends_with(':') || line.starts_with("switch ") {
            continue;
        }
        if let Some(rest) = line.strip_prefix("store ") {
            let mut parts = rest.splitn(2, ',');
            let Some(value) = parts.next() else { continue };
            let Some(ptr) = parts.next() else { continue };
            let Some(global) = parse_global_name(ptr) else {
                continue;
            };
            if let Some(value) = eval_value_token(value_token(value), &env, &globals) {
                unknown_stores.remove(&global);
                globals.insert(
                    global,
                    match value {
                        StaticValue::Int(value) => StaticValue::Int(value & u64::from(u32::MAX)),
                        value => value,
                    },
                );
            } else {
                globals.remove(&global);
                unknown_stores.insert(global);
            }
            continue;
        }
        let Some((name, rhs)) = line.split_once(" = ") else {
            continue;
        };
        let name = name.trim();
        if !name.starts_with('%') {
            continue;
        }
        if let Some(value) = eval_static_rhs(rhs.trim(), &env, &globals) {
            env.insert(name.to_string(), value);
        }
    }

    globals.retain(|global, _| !unknown_stores.contains(global));
    globals
}

/// Integer scalar/vector mirrors derived from an AIR function-constant initializer and only read
/// afterward. This
/// is the product-safe subset of the metadata evaluator above: ordinary constructor state is left
/// intact, and any non-load use outside a constructor may mutate or escape the cell, so it is
/// excluded.
pub(crate) fn static_init_foldable_global_values(ll: &str) -> HashMap<String, StaticIntValue> {
    let mut values = static_init_global_values(ll);
    let mut derived_globals = ll
        .lines()
        .filter_map(|raw| {
            let line = raw.split(';').next().unwrap_or(raw).trim();
            (line.starts_with('@') && line.contains("air.fc_initializer"))
                .then(|| line.split_once(" = ").map(|(name, _)| name.to_string()))
                .flatten()
        })
        .collect::<HashSet<_>>();
    let initializer_globals = derived_globals.clone();
    let mut derived_locals = HashSet::<String>::new();
    let mut in_constructor = false;
    for raw in ll.lines() {
        let line = raw.split(';').next().unwrap_or(raw).trim();
        if line.starts_with("define ") {
            in_constructor = line.contains("@_GLOBAL__sub_I");
            derived_locals.clear();
            continue;
        }
        if line == "}" {
            in_constructor = false;
            continue;
        }
        if !in_constructor {
            continue;
        }
        if let Some(rest) = line.strip_prefix("store ") {
            let mut parts = rest.splitn(2, ',');
            let value = parts.next().map(value_token);
            let target = parts.next().and_then(parse_global_name);
            if let (Some(value), Some(target)) = (value, target) {
                if derived_locals.contains(value) || derived_globals.contains(value) {
                    derived_globals.insert(target);
                } else {
                    derived_globals.remove(&target);
                }
            }
            continue;
        }
        let Some((result, rhs)) = line.split_once(" = ") else {
            continue;
        };
        if result.starts_with('%')
            && derived_globals
                .iter()
                .chain(&derived_locals)
                .any(|symbol| references_symbol(rhs, symbol))
        {
            derived_locals.insert(result.to_string());
        }
    }
    values.retain(|global, _| derived_globals.contains(global));

    let mut in_static_init = false;
    let mut in_function = false;
    for raw in ll.lines() {
        let line = raw.split(';').next().unwrap_or(raw).trim();
        if line.starts_with("define ") {
            in_function = true;
            in_static_init = line.contains("@_GLOBAL__sub_I");
            continue;
        }
        if line == "}" {
            in_function = false;
            in_static_init = false;
            continue;
        }
        if !in_function || in_static_init || line.is_empty() {
            continue;
        }
        values.retain(|global, _| {
            if !references_symbol(line, global) {
                return true;
            }
            let Some(load) = line.split_once(" = load ").map(|(_, load)| load) else {
                return false;
            };
            parse_global_name(load).as_ref() == Some(global)
        });
    }
    // Keep the ABI initializer cells and their loads in generic SPIR-V so the public post-emit
    // specialization helper can still override direct function-constant uses. Only constructor-
    // derived immutable mirrors are folded under the generic translation's zero/default model;
    // structure-changing nonzero values use the AIR-level specialization API.
    values.retain(|global, _| !initializer_globals.contains(global));
    values
        .into_iter()
        .map(|(global, value)| {
            let value = match value {
                StaticValue::Bool(value) => StaticIntValue::Scalar(u32::from(value)),
                StaticValue::Int(value) => StaticIntValue::Scalar(value as u32),
                StaticValue::Vector(values) => {
                    StaticIntValue::Vector(values.into_iter().map(|value| value as u32).collect())
                }
            };
            (global, value)
        })
        .collect()
}

fn references_symbol(text: &str, symbol: &str) -> bool {
    text.match_indices(symbol).any(|(start, _)| {
        let end = start + symbol.len();
        let boundary = |byte: u8| !matches!(byte, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'$');
        (start == 0 || boundary(text.as_bytes()[start - 1]))
            && (end == text.len() || boundary(text.as_bytes()[end]))
    })
}

fn parse_static_global_initializers(ll: &str) -> HashMap<String, StaticValue> {
    let mut globals = HashMap::new();
    for raw in ll.lines() {
        let line = raw.split(';').next().unwrap_or(raw).trim();
        if !line.starts_with('@') || !(line.contains(" global ") || line.contains(" constant ")) {
            continue;
        }
        let Some((name, rest)) = line.split_once(" = ") else {
            continue;
        };
        let Some(value) = integer_initializer(rest) else {
            continue;
        };
        globals.insert(name.trim().to_string(), value);
    }
    globals
}

fn integer_initializer(rest: &str) -> Option<StaticValue> {
    let typed_init = rest
        .split_once(" global ")
        .map(|(_, init)| init)
        .or_else(|| rest.split_once(" constant ").map(|(_, init)| init))?;
    if typed_init.starts_with('<') {
        let type_end = typed_init.find('>')?;
        let vector_ty = &typed_init[1..type_end];
        let (lanes, element_ty) = vector_ty.split_once(" x ")?;
        let lanes = lanes.parse::<usize>().ok()?;
        let width = element_ty.strip_prefix('i')?.parse::<u32>().ok()?;
        if width == 0 || width > 32 {
            return None;
        }
        let vector = typed_init[type_end + 1..].trim_start();
        let vector = if vector.starts_with('<') {
            &vector[..=vector.find('>')?]
        } else {
            vector.split(',').next().unwrap_or(vector).trim()
        };
        if vector == "undef" || vector == "zeroinitializer" {
            return rest
                .contains("air.fc_initializer")
                .then(|| StaticValue::Vector(vec![0; lanes]));
        }
        let values = vector.strip_prefix('<')?.strip_suffix('>')?;
        let mask = if width == 32 {
            u64::from(u32::MAX)
        } else {
            (1_u64 << width) - 1
        };
        let values = values
            .split(',')
            .map(|lane| {
                let (ty, value) = lane.trim().split_once(' ')?;
                (ty == element_ty)
                    .then(|| {
                        value
                            .parse::<u64>()
                            .or_else(|_| value.parse::<i64>().map(|value| value as u64))
                            .ok()
                    })
                    .flatten()
                    .map(|value| value & mask)
            })
            .collect::<Option<Vec<_>>>()?;
        return (values.len() == lanes).then_some(StaticValue::Vector(values));
    }
    let mut tokens = typed_init.split_whitespace();
    let ty = tokens.next()?;
    let width = ty.strip_prefix('i')?.parse::<u32>().ok()?;
    if !matches!(width, 8 | 16 | 32) {
        return None;
    }
    let mask = (1_u64 << width) - 1;
    let value = tokens.next()?.trim_end_matches(',');
    if value == "undef" && rest.contains("air.fc_initializer") {
        return Some(StaticValue::Int(0));
    }
    value
        .parse::<u64>()
        .or_else(|_| value.parse::<i64>().map(|value| value as u64))
        .ok()
        .map(|value| StaticValue::Int(value & mask))
}

fn eval_static_rhs(
    rhs: &str,
    env: &HashMap<String, StaticValue>,
    globals: &HashMap<String, StaticValue>,
) -> Option<StaticValue> {
    if rhs.contains("@air.is_function_constant_defined(") {
        return Some(StaticValue::Bool(false));
    }
    if rhs.contains("@air.normalize_function_constant_predicate.") {
        let arguments = rhs.split_once('(')?.1.rsplit_once(')')?.0;
        let value = eval_int_operand(arguments, env, globals)?;
        return Some(StaticValue::Int(u64::from(value != 0)));
    }
    if rhs.starts_with("load ") {
        let global = parse_global_name(rhs)?;
        return globals.get(&global).cloned();
    }
    if let Some(rest) = rhs.strip_prefix("extractelement ") {
        let parts = rest.split(',').collect::<Vec<_>>();
        let vector = eval_value_token(parts.first()?.split_whitespace().last()?, env, globals)?;
        let idx = eval_int_operand(parts.get(1)?, env, globals)? as usize;
        let StaticValue::Vector(values) = vector else {
            return None;
        };
        return values.get(idx).copied().map(StaticValue::Int);
    }
    if let Some((opcode, rest)) = ["add", "mul", "and", "or", "xor", "shl", "lshr"]
        .into_iter()
        .find_map(|opcode| {
            rhs.strip_prefix(opcode)
                .and_then(|rest| rest.strip_prefix(' '))
                .map(|rest| (opcode, rest))
        })
    {
        let (lhs, rhs) = eval_binary_int(rest, env, globals)?;
        let (width, mask) = integer_result_width_and_mask(rest)?;
        let lhs = lhs & mask;
        let value = match opcode {
            "add" => lhs.wrapping_add(rhs & mask),
            "mul" => lhs.wrapping_mul(rhs & mask),
            "and" => lhs & (rhs & mask),
            "or" => lhs | (rhs & mask),
            "xor" => lhs ^ (rhs & mask),
            "shl" if rhs < u64::from(width) => lhs.checked_shl(rhs.try_into().ok()?)?,
            "lshr" if rhs < u64::from(width) => lhs.checked_shr(rhs.try_into().ok()?)?,
            "shl" | "lshr" => return None,
            _ => unreachable!("matched integer opcode"),
        };
        return Some(StaticValue::Int(value & mask));
    }
    if let Some(rest) = rhs.strip_prefix("icmp ") {
        let mut fields = rest.splitn(2, ' ');
        let pred = fields.next()?;
        let (lhs, rhs) = eval_binary_int(fields.next()?, env, globals)?;
        let value = match pred {
            "eq" => lhs == rhs,
            "ne" => lhs != rhs,
            _ => return None,
        };
        return Some(StaticValue::Bool(value));
    }
    if let Some(rest) = rhs.strip_prefix("select ") {
        let parts = rest.split(',').collect::<Vec<_>>();
        let cond = eval_bool_operand(parts.first()?, env, globals)?;
        let chosen = if cond { parts.get(1)? } else { parts.get(2)? };
        return eval_value_token(value_token(chosen), env, globals);
    }
    if let Some(rest) = rhs
        .strip_prefix("trunc ")
        .or_else(|| rhs.strip_prefix("zext "))
    {
        let (value, to_ty) = rest.split_once(" to ")?;
        let mut int = eval_int_operand(value, env, globals)?;
        if to_ty.trim_start().starts_with("i8") {
            int &= 0xff;
        } else if to_ty.trim_start().starts_with("i16") {
            int &= 0xffff;
        }
        return Some(StaticValue::Int(int));
    }
    if rhs.contains("function_constant_predicate") {
        let open = rhs.rfind('(')?;
        let close = rhs.rfind(')')?;
        return Some(StaticValue::Int(eval_int_operand(
            &rhs[open + 1..close],
            env,
            globals,
        )?));
    }
    None
}

fn eval_binary_int(
    rest: &str,
    env: &HashMap<String, StaticValue>,
    globals: &HashMap<String, StaticValue>,
) -> Option<(u64, u64)> {
    let parts = rest.split(',').collect::<Vec<_>>();
    Some((
        eval_int_operand(parts.first()?, env, globals)?,
        eval_int_operand(parts.get(1)?, env, globals)?,
    ))
}

fn integer_result_width_and_mask(text: &str) -> Option<(u32, u64)> {
    let width = text
        .split(|c: char| c.is_whitespace() || c == ',')
        .find_map(|token| token.strip_prefix('i')?.parse::<u32>().ok())?;
    match width {
        1..=63 => Some((width, (1_u64 << width) - 1)),
        64 => Some((width, u64::MAX)),
        _ => None,
    }
}

fn eval_int_operand(
    text: &str,
    env: &HashMap<String, StaticValue>,
    globals: &HashMap<String, StaticValue>,
) -> Option<u64> {
    match eval_value_token(value_token(text), env, globals)? {
        StaticValue::Bool(value) => Some(u64::from(value)),
        StaticValue::Int(value) => Some(value),
        StaticValue::Vector(_) => None,
    }
}

fn eval_bool_operand(
    text: &str,
    env: &HashMap<String, StaticValue>,
    globals: &HashMap<String, StaticValue>,
) -> Option<bool> {
    match eval_value_token(value_token(text), env, globals)? {
        StaticValue::Bool(value) => Some(value),
        StaticValue::Int(value) => Some(value != 0),
        StaticValue::Vector(_) => None,
    }
}

fn eval_value_token(
    token: &str,
    env: &HashMap<String, StaticValue>,
    globals: &HashMap<String, StaticValue>,
) -> Option<StaticValue> {
    let token = token.trim().trim_end_matches(',');
    if token == "true" {
        return Some(StaticValue::Bool(true));
    }
    if token == "false" {
        return Some(StaticValue::Bool(false));
    }
    if token.starts_with('%') {
        return env.get(token).cloned();
    }
    if token.starts_with('@') {
        return globals.get(token).cloned();
    }
    token
        .parse::<u64>()
        .or_else(|_| token.parse::<i64>().map(|value| value as u64))
        .ok()
        .map(StaticValue::Int)
}

fn value_token(text: &str) -> &str {
    text.split_whitespace()
        .last()
        .unwrap_or_else(|| text.trim())
        .trim_end_matches(',')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_initializer_is_not_misclassified_as_a_scalar() {
        let ll = r#"
@fc.MTL_FC_INIT_3_Dv4_j = internal addrspace(2) externally_initialized constant <4 x i32> <i32 1, i32 2, i32 3, i32 4>, section "air.fc_initializer", align 16
@mirror = internal addrspace(2) global i32 undef, align 4

define internal void @_GLOBAL__sub_I_vector_fc() {
entry:
  %value = load <4 x i32>, ptr addrspace(2) @fc.MTL_FC_INIT_3_Dv4_j
  %lane = extractelement <4 x i32> %value, i64 2
  store i32 %lane, ptr addrspace(2) @mirror
  ret void
}

define i32 @use() {
entry:
  %value = load i32, ptr addrspace(2) @mirror
  ret i32 %value
}
"#;

        let values = static_init_int_global_values(ll);
        assert_eq!(values.get("@mirror"), Some(&3));
        assert!(!values.contains_key("@fc.MTL_FC_INIT_3_Dv4_j"));
        let foldable = static_init_foldable_global_values(ll);
        assert!(matches!(
            foldable.get("@mirror"),
            Some(StaticIntValue::Scalar(3))
        ));
        assert!(!foldable.contains_key("@fc.MTL_FC_INIT_3_Dv4_j"));
    }

    #[test]
    fn signed_masked_function_constant_initializer_is_foldable() {
        let ll = r#"
@fc.MTL_FC_INIT_2_t = internal addrspace(2) externally_initialized constant i16 undef, section "air.fc_initializer", align 2
@rounded = internal addrspace(2) global i16 undef, align 2
@negative = internal addrspace(2) global i16 -1, align 2

define internal void @_GLOBAL__sub_I_fc() {
entry:
  %value = load i16, ptr addrspace(2) @fc.MTL_FC_INIT_2_t
  %biased = add i16 %value, 15
  %masked = and i16 %biased, -16
  store i16 %masked, ptr addrspace(2) @rounded
  ret void
}

define i16 @use() {
entry:
  %value = load i16, ptr addrspace(2) @rounded
  ret i16 %value
}
"#;

        let values = static_init_foldable_global_values(ll);
        assert!(matches!(
            values.get("@rounded"),
            Some(StaticIntValue::Scalar(0))
        ));
        assert_eq!(
            static_init_int_global_values(ll).get("@negative"),
            Some(&65535)
        );
    }

    #[test]
    fn out_of_width_shift_is_not_folded() {
        let ll = r#"
@fc.MTL_FC_INIT_2_t = internal addrspace(2) externally_initialized constant i16 undef, section "air.fc_initializer", align 2
@shifted = internal addrspace(2) global i16 undef, align 2

define internal void @_GLOBAL__sub_I_fc() {
entry:
  %value = load i16, ptr addrspace(2) @fc.MTL_FC_INIT_2_t
  %value.shifted = shl i16 %value, 16
  store i16 %value.shifted, ptr addrspace(2) @shifted
  ret void
}

define i16 @use() {
entry:
  %value = load i16, ptr addrspace(2) @shifted
  ret i16 %value
}
"#;

        assert!(!static_init_foldable_global_values(ll).contains_key("@shifted"));
    }

    #[test]
    fn function_constant_predicate_normalization_preserves_an_enabled_default() {
        let ll = r#"
@fc.MTL_FC_INIT_1_b = internal addrspace(2) externally_initialized constant i8 1, section "air.fc_initializer", align 1
@predicate = internal addrspace(2) global i8 0, align 1

define internal void @_GLOBAL__sub_I_fc() {
entry:
  %value = load i8, ptr addrspace(2) @fc.MTL_FC_INIT_1_b
  %normalized = tail call i8 @air.normalize_function_constant_predicate.i8(i8 %value)
  store i8 %normalized, ptr addrspace(2) @predicate
  ret void
}
"#;

        assert_eq!(
            static_init_int_global_values(ll).get("@predicate"),
            Some(&1)
        );
    }
}
