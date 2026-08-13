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
    Vec4([u64; 4]),
}

/// Best-effort evaluator for AIR static initializers that materialize default function-constant
/// integer globals. Unknown expressions are ignored so dynamic metadata falls back to its immediate
/// location index rather than guessing.
pub(super) fn static_init_int_global_values(ll: &str) -> HashMap<String, u32> {
    let mut globals = parse_integer_global_initializers(ll);
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
            if let Some(value) = eval_int_operand(value, &env, &globals) {
                let value = value as u32;
                unknown_stores.remove(&global);
                globals.insert(global, value);
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

    globals
        .into_iter()
        .filter(|(global, _)| !unknown_stores.contains(global))
        .collect()
}

/// Integer mirrors derived from an AIR function-constant initializer and only read afterward. This
/// is the product-safe subset of the metadata evaluator above: ordinary constructor state is left
/// intact, and any non-load use outside a constructor may mutate or escape the cell, so it is
/// excluded.
pub(crate) fn static_init_foldable_int_global_values(ll: &str) -> HashMap<String, u32> {
    let mut values = static_init_int_global_values(ll);
    let mut derived_globals = ll
        .lines()
        .filter_map(|raw| {
            let line = raw.split(';').next().unwrap_or(raw).trim();
            (line.starts_with('@') && line.contains("air.fc_initializer"))
                .then(|| line.split_once(" = ").map(|(name, _)| name.to_string()))
                .flatten()
        })
        .collect::<HashSet<_>>();
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
    values
}

fn references_symbol(text: &str, symbol: &str) -> bool {
    text.match_indices(symbol).any(|(start, _)| {
        let end = start + symbol.len();
        let boundary = |byte: u8| !matches!(byte, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'$');
        (start == 0 || boundary(text.as_bytes()[start - 1]))
            && (end == text.len() || boundary(text.as_bytes()[end]))
    })
}

fn parse_integer_global_initializers(ll: &str) -> HashMap<String, u32> {
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

fn integer_initializer(rest: &str) -> Option<u32> {
    let typed_init = rest
        .split_once(" global ")
        .map(|(_, init)| init)
        .or_else(|| rest.split_once(" constant ").map(|(_, init)| init))?;
    let tokens = typed_init.split_whitespace().collect::<Vec<_>>();
    let ty_pos = tokens
        .iter()
        .position(|tok| matches!(*tok, "i8" | "i16" | "i32"))?;
    let value = tokens.get(ty_pos + 1)?.trim_end_matches(',');
    if value == "undef" && rest.contains("air.fc_initializer") {
        return Some(0);
    }
    value.parse::<u32>().ok()
}

fn eval_static_rhs(
    rhs: &str,
    env: &HashMap<String, StaticValue>,
    globals: &HashMap<String, u32>,
) -> Option<StaticValue> {
    if rhs.contains("@air.is_function_constant_defined(") {
        return Some(StaticValue::Bool(false));
    }
    if rhs.starts_with("load <4 x i32>") {
        return Some(StaticValue::Vec4([0; 4]));
    }
    if rhs.starts_with("load i") {
        let global = parse_global_name(rhs)?;
        return globals
            .get(&global)
            .map(|value| StaticValue::Int(*value as u64));
    }
    if let Some(rest) = rhs.strip_prefix("extractelement ") {
        let parts = rest.split(',').collect::<Vec<_>>();
        let vector = eval_value_token(parts.first()?.split_whitespace().last()?, env, globals)?;
        let idx = eval_int_operand(parts.get(1)?, env, globals)? as usize;
        let StaticValue::Vec4(values) = vector else {
            return None;
        };
        return values.get(idx).copied().map(StaticValue::Int);
    }
    if let Some(rest) = rhs.strip_prefix("lshr ") {
        let (lhs, rhs) = eval_binary_int(rest, env, globals)?;
        return Some(StaticValue::Int(lhs >> rhs));
    }
    if let Some(rest) = rhs.strip_prefix("and ") {
        let (lhs, rhs) = eval_binary_int(rest, env, globals)?;
        return Some(StaticValue::Int(lhs & rhs));
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
    globals: &HashMap<String, u32>,
) -> Option<(u64, u64)> {
    let parts = rest.split(',').collect::<Vec<_>>();
    Some((
        eval_int_operand(parts.first()?, env, globals)?,
        eval_int_operand(parts.get(1)?, env, globals)?,
    ))
}

fn eval_int_operand(
    text: &str,
    env: &HashMap<String, StaticValue>,
    globals: &HashMap<String, u32>,
) -> Option<u64> {
    match eval_value_token(value_token(text), env, globals)? {
        StaticValue::Bool(value) => Some(u64::from(value)),
        StaticValue::Int(value) => Some(value),
        StaticValue::Vec4(_) => None,
    }
}

fn eval_bool_operand(
    text: &str,
    env: &HashMap<String, StaticValue>,
    globals: &HashMap<String, u32>,
) -> Option<bool> {
    match eval_value_token(value_token(text), env, globals)? {
        StaticValue::Bool(value) => Some(value),
        StaticValue::Int(value) => Some(value != 0),
        StaticValue::Vec4(_) => None,
    }
}

fn eval_value_token(
    token: &str,
    env: &HashMap<String, StaticValue>,
    globals: &HashMap<String, u32>,
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
        return globals
            .get(token)
            .map(|value| StaticValue::Int(*value as u64));
    }
    token.parse::<u64>().ok().map(StaticValue::Int)
}

fn value_token(text: &str) -> &str {
    text.split_whitespace()
        .last()
        .unwrap_or_else(|| text.trim())
        .trim_end_matches(',')
}
