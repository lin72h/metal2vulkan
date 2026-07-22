use super::location_index;
use std::collections::HashMap;

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
    let mut stores = HashMap::new();
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
                globals.insert(global.clone(), value);
                stores.insert(global, value);
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

    stores
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
    tokens
        .get(ty_pos + 1)?
        .trim_end_matches(',')
        .parse::<u32>()
        .ok()
}

fn eval_static_rhs(
    rhs: &str,
    env: &HashMap<String, StaticValue>,
    globals: &HashMap<String, u32>,
) -> Option<StaticValue> {
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
