//! Static specialization of authored Metal function tables.
//!
//! Logical SPIR-V has no portable runtime function-pointer call. A caller that knows the exact
//! Metal table population can nevertheless preserve the program by resolving each table lookup to
//! a directly linked AIR function before native parsing. The linkage contract is structural: entry
//! parameter index, table slot, exact LLVM symbol, and sanitized dependency module.

use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LinkedFunctionLinkage {
    /// Direct `.MTL_VISIBLE_FN_REF` dependencies named by AIR's
    /// `!air.visible_function_references` metadata.
    pub visible_references: Vec<LinkedFunctionReference>,
    pub visible_tables: Vec<LinkedFunctionTable>,
    pub intersection_tables: Vec<IntersectionFunctionTable>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkedFunctionReference {
    /// Logical Metal function name recorded in the AIR reference metadata.
    pub symbol: String,
    /// Sanitized AIR LLVM module that defines `symbol` and its dependencies.
    pub module_ll: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkedFunctionTable {
    /// Zero-based LLVM parameter index of the AIR function-table argument.
    pub parameter_index: u32,
    /// Total authored capacity, including null slots.
    pub size: u32,
    pub entries: Vec<LinkedFunction>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkedFunction {
    pub index: u32,
    pub symbol: String,
    /// Sanitized AIR LLVM module that defines `symbol` and its dependencies.
    pub module_ll: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntersectionFunctionTable {
    pub source: IntersectionFunctionTableSource,
    pub size: u32,
    pub entries: Vec<IntersectionFunctionEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntersectionFunctionTableSource {
    Parameter {
        parameter_index: u32,
    },
    ArgumentBuffer {
        buffer_parameter_index: u32,
        field_ordinal: u32,
        field_offset: u32,
    },
}

impl IntersectionFunctionTableSource {
    fn parameter_index(self) -> u32 {
        match self {
            Self::Parameter { parameter_index } => parameter_index,
            Self::ArgumentBuffer {
                buffer_parameter_index,
                ..
            } => buffer_parameter_index,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IntersectionFunctionEntry {
    Linked(LinkedFunction),
    OpaqueTriangle {
        index: u32,
        signature: Vec<IntersectionFunctionSignature>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntersectionFunctionSignature {
    Instancing,
    TriangleData,
    WorldSpaceData,
    InstanceMotion,
    PrimitiveMotion,
    ExtendedLimits,
    MaxLevels,
    IntersectionFunctionBuffer,
    UserData,
}

#[derive(Clone)]
struct PointerTrace<'a> {
    table: &'a LinkedFunctionTable,
    index: String,
}

#[derive(Default)]
struct LinkedFlow<'a> {
    table_parameters: HashMap<(String, usize), &'a LinkedFunctionTable>,
    pointer_parameters: HashMap<(String, usize), &'a LinkedFunctionTable>,
}

#[derive(Clone, Debug)]
struct FunctionSignature {
    parameters: Vec<String>,
}

impl LinkedFunctionLinkage {
    pub fn is_empty(&self) -> bool {
        self.visible_references.is_empty()
            && self.visible_tables.is_empty()
            && self.intersection_tables.is_empty()
    }
}

/// Replace callback-bearing AIR ray queries with the corresponding callback-free ABI only when
/// the authored table is completely populated by Metal's opaque-triangle sentinel and every slot
/// declares the exact signature required by the query family. Linked callbacks and null slots are
/// deliberately left untouched.
pub fn specialize_opaque_triangle_intersection_tables(
    entry_ll: &str,
    entry_name: &str,
    linkage: &LinkedFunctionLinkage,
) -> Result<String, String> {
    if linkage.intersection_tables.is_empty() {
        return Ok(entry_ll.to_string());
    }
    let signatures = function_signatures(entry_ll)?;
    let entry_global = llvm_global(entry_name)?;
    let entry_signature = signatures.get(&entry_global).ok_or_else(|| {
        format!("intersection-table entry function {entry_name:?} is not defined")
    })?;
    let mut flow = HashMap::<(String, usize), &IntersectionFunctionTable>::new();
    let mut embedded_roots = HashMap::<String, Vec<&IntersectionFunctionTable>>::new();
    for table in &linkage.intersection_tables {
        let parameter_index = table.source.parameter_index();
        if table.size == 0 {
            return Err(format!(
                "intersection function-table parameter {} has zero size",
                parameter_index
            ));
        }
        entry_signature
            .parameters
            .get(parameter_index as usize)
            .ok_or_else(|| {
                format!(
                    "intersection function-table parameter {} exceeds entry {:?} arity {}",
                    parameter_index,
                    entry_name,
                    entry_signature.parameters.len()
                )
            })?;
        match table.source {
            IntersectionFunctionTableSource::Parameter { parameter_index } => {
                if flow
                    .insert((entry_global.clone(), parameter_index as usize), table)
                    .is_some()
                {
                    return Err(format!(
                        "duplicate intersection function-table parameter {parameter_index}"
                    ));
                }
            }
            IntersectionFunctionTableSource::ArgumentBuffer { .. } => {
                embedded_roots
                    .entry(entry_signature.parameters[parameter_index as usize].clone())
                    .or_default()
                    .push(table);
            }
        }
    }
    propagate_intersection_table_flow(
        entry_ll,
        &entry_global,
        &signatures,
        &embedded_roots,
        &mut flow,
    )?;

    let mut output = String::with_capacity(entry_ll.len());
    let mut current = None::<String>;
    let mut tables = HashMap::<String, &IntersectionFunctionTable>::new();
    let mut embedded_pointers = HashMap::<String, &IntersectionFunctionTable>::new();
    for line in entry_ll.lines() {
        let trimmed = line.trim_start();
        if let Some(global) = definition_global(trimmed) {
            current = Some(global.clone());
            tables.clear();
            embedded_pointers.clear();
            let signature = signatures
                .get(&global)
                .ok_or_else(|| format!("missing parsed signature for {global}"))?;
            for (ordinal, parameter) in signature.parameters.iter().enumerate() {
                if let Some(table) = flow.get(&(global.clone(), ordinal)) {
                    tables.insert(parameter.clone(), *table);
                }
            }
            output.push_str(line);
            output.push('\n');
            continue;
        }
        if trimmed == "}" {
            current = None;
            tables.clear();
            embedded_pointers.clear();
            output.push_str(line);
            output.push('\n');
            continue;
        }
        if current.is_some() {
            if current.as_deref() == Some(entry_global.as_str()) {
                trace_embedded_intersection_table(
                    trimmed,
                    &embedded_roots,
                    &mut embedded_pointers,
                    &mut tables,
                );
            }
            if let Some((result, instruction)) = trimmed.split_once(" = ") {
                if instruction.starts_with("bitcast ") || instruction.starts_with("addrspacecast ")
                {
                    if let Some(source) = cast_source_value(instruction) {
                        if let Some(table) = tables.get(source) {
                            tables.insert(result.trim().to_string(), *table);
                        }
                    }
                }
            }
            if let Some((callee, open, close, arguments)) = named_call(trimmed) {
                let symbol = callee.trim_start_matches('@').trim_matches('"');
                if symbol.starts_with("air.set_buffer_intersection_function_table.") {
                    if arguments.len() != 3 {
                        return Err(format!(
                            "AIR intersection function-table setter {symbol} has {} operands, expected 3",
                            arguments.len()
                        ));
                    }
                    // Logical SPIR-V has no mutable function-table object. The authored linkage is
                    // the exact table population presented to the translated shader, so a setter
                    // targeting that proven table has already been applied by construction. Do not
                    // consume an untraced destination: without authored contents, dropping the call
                    // would silently erase dynamic table semantics.
                    if tables.contains_key(value_operand(arguments[0])) {
                        continue;
                    }
                }
                if let Some(family) = crate::meta::AirIntersectionFamily::parse(symbol)? {
                    if family.intersection_function_buffer {
                        let table_ordinal = family.intersection_table_argument_index();
                        if let Some(table) = arguments
                            .get(table_ordinal)
                            .and_then(|argument| tables.get(value_operand(argument)))
                        {
                            if opaque_table_matches_family(table, &family) {
                                let removed_arguments = family
                                    .opaque_triangle_removed_argument_indices()
                                    .expect("callback family has callback operands");
                                let expected_arguments = family.argument_count();
                                if arguments.len() != expected_arguments {
                                    return Err(format!(
                                        "AIR intersection call {symbol} has {} operands, expected {expected_arguments}",
                                        arguments.len(),
                                    ));
                                }
                                let kept = arguments
                                    .iter()
                                    .enumerate()
                                    .filter(|(ordinal, _)| !removed_arguments.contains(ordinal))
                                    .map(|(_, argument)| *argument)
                                    .collect::<Vec<_>>();
                                let callback_free = symbol
                                    .split('.')
                                    .filter(|token| {
                                        !matches!(
                                            *token,
                                            "intersection_function_buffer" | "user_data"
                                        )
                                    })
                                    .collect::<Vec<_>>()
                                    .join(".");
                                let leading = line.len() - trimmed.len();
                                output.push_str(&line[..leading]);
                                output.push_str(&trimmed[..open - callee.len()]);
                                output.push('@');
                                output.push_str(&callback_free);
                                output.push('(');
                                output.push_str(&kept.join(", "));
                                output.push_str(&trimmed[close..]);
                                output.push('\n');
                                continue;
                            }
                        }
                    }
                }
            }
        }
        output.push_str(line);
        output.push('\n');
    }
    Ok(output)
}

fn opaque_table_matches_family(
    table: &IntersectionFunctionTable,
    family: &crate::meta::AirIntersectionFamily,
) -> bool {
    if table.entries.len() != table.size as usize {
        return false;
    }
    let expected = opaque_triangle_signature(family);
    let Some(expected) = expected else {
        return false;
    };
    table.entries.iter().enumerate().all(|(index, entry)| {
        let IntersectionFunctionEntry::OpaqueTriangle {
            index: entry_index,
            signature,
        } = entry
        else {
            return false;
        };
        let mut signature = signature.clone();
        signature.sort_unstable();
        *entry_index == index as u32 && signature == expected
    })
}

/// Exact sorted Metal signature flags for an authored opaque-triangle slot serving this AIR
/// callback family. `None` means the family does not consume an intersection-function table.
pub fn opaque_triangle_signature(
    family: &crate::meta::AirIntersectionFamily,
) -> Option<Vec<IntersectionFunctionSignature>> {
    if !family.intersection_function_buffer {
        return None;
    }
    let mut expected = Vec::new();
    use crate::meta::AirIntersectionInstancing;
    if family.instancing != AirIntersectionInstancing::None {
        expected.push(IntersectionFunctionSignature::Instancing);
    }
    if family.triangle_data {
        expected.push(IntersectionFunctionSignature::TriangleData);
    }
    if family.world_space_data {
        expected.push(IntersectionFunctionSignature::WorldSpaceData);
    }
    if family.instance_motion {
        expected.push(IntersectionFunctionSignature::InstanceMotion);
    }
    if family.primitive_motion {
        expected.push(IntersectionFunctionSignature::PrimitiveMotion);
    }
    if family.extended_limits {
        expected.push(IntersectionFunctionSignature::ExtendedLimits);
    }
    if family.instancing == AirIntersectionInstancing::MultiLevel {
        expected.push(IntersectionFunctionSignature::MaxLevels);
    }
    expected.push(IntersectionFunctionSignature::IntersectionFunctionBuffer);
    if family.user_data {
        expected.push(IntersectionFunctionSignature::UserData);
    }
    expected.sort_unstable();
    Some(expected)
}

fn trace_embedded_intersection_table<'a>(
    line: &str,
    roots: &HashMap<String, Vec<&'a IntersectionFunctionTable>>,
    pointers: &mut HashMap<String, &'a IntersectionFunctionTable>,
    tables: &mut HashMap<String, &'a IntersectionFunctionTable>,
) {
    let Some((result, instruction)) = line.split_once(" = ") else {
        return;
    };
    let result = result.trim();
    if instruction.starts_with("getelementptr ")
        || instruction.starts_with("getelementptr inbounds ")
    {
        let operands = split_top_level(instruction, ',');
        let Some(base) = operands.get(1).map(|operand| value_operand(operand)) else {
            return;
        };
        let Some(field_ordinal) = operands.last().and_then(|operand| integer_operand(operand))
        else {
            return;
        };
        if let Some(table) = roots.get(base).and_then(|tables| {
            tables.iter().copied().find(|table| {
                matches!(
                    table.source,
                    IntersectionFunctionTableSource::ArgumentBuffer {
                        field_ordinal: authored,
                        ..
                    } if authored == field_ordinal
                )
            })
        }) {
            pointers.insert(result.to_string(), table);
        }
        return;
    }
    if instruction.starts_with("load ") {
        let operands = split_top_level(instruction, ',');
        if let Some(pointer) = operands.get(1).map(|operand| value_operand(operand)) {
            if let Some(table) = pointers.get(pointer) {
                tables.insert(result.to_string(), *table);
            }
        }
    }
}

fn propagate_intersection_table_flow<'a>(
    ll: &str,
    entry_global: &str,
    signatures: &HashMap<String, FunctionSignature>,
    embedded_roots: &HashMap<String, Vec<&'a IntersectionFunctionTable>>,
    flow: &mut HashMap<(String, usize), &'a IntersectionFunctionTable>,
) -> Result<(), String> {
    loop {
        let mut changed = false;
        let mut current = None::<String>;
        let mut tables = HashMap::<String, &'a IntersectionFunctionTable>::new();
        let mut embedded_pointers = HashMap::<String, &'a IntersectionFunctionTable>::new();
        for line in ll.lines() {
            let trimmed = line.trim_start();
            if let Some(global) = definition_global(trimmed) {
                current = Some(global.clone());
                tables.clear();
                embedded_pointers.clear();
                let signature = signatures
                    .get(&global)
                    .ok_or_else(|| format!("missing parsed signature for {global}"))?;
                for (ordinal, parameter) in signature.parameters.iter().enumerate() {
                    if let Some(table) = flow.get(&(global.clone(), ordinal)) {
                        tables.insert(parameter.clone(), *table);
                    }
                }
                continue;
            }
            if trimmed == "}" {
                current = None;
                continue;
            }
            if current.is_none() {
                continue;
            }
            if current.as_deref() == Some(entry_global) {
                trace_embedded_intersection_table(
                    trimmed,
                    embedded_roots,
                    &mut embedded_pointers,
                    &mut tables,
                );
            }
            if let Some((result, instruction)) = trimmed.split_once(" = ") {
                if instruction.starts_with("bitcast ") || instruction.starts_with("addrspacecast ")
                {
                    if let Some(source) = cast_source_value(instruction) {
                        if let Some(table) = tables.get(source) {
                            tables.insert(result.trim().to_string(), *table);
                        }
                    }
                }
            }
            let Some((callee, _, _, arguments)) = named_call(trimmed) else {
                continue;
            };
            let Some(callee_signature) = signatures.get(callee) else {
                continue;
            };
            for (ordinal, argument) in arguments
                .iter()
                .take(callee_signature.parameters.len())
                .enumerate()
            {
                if let Some(table) = tables.get(value_operand(argument)) {
                    let key = (callee.to_string(), ordinal);
                    match flow.get(&key) {
                        Some(previous) if !std::ptr::eq(*previous, *table) => {
                            return Err(format!(
                                "function parameter {ordinal} of {callee} receives multiple intersection tables"
                            ));
                        }
                        Some(_) => {}
                        None => {
                            flow.insert(key, *table);
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            return Ok(());
        }
    }
}

/// Resolve AIR's direct Metal visible-function references to exact authored dependencies.
///
/// Metal records these as calls to module-local `.MTL_VISIBLE_FN_REF` stubs plus metadata mapping
/// each stub to its logical linked symbol. Logical SPIR-V cannot retain that dynamic linker
/// contract, so the authored dependency graph is closed here and every stub is rewritten to the
/// directly appended definition. Dependency modules may themselves contain visible references;
/// those are resolved recursively from the same exact authored set.
pub fn specialize_visible_function_references(
    entry_ll: &str,
    linkage: &LinkedFunctionLinkage,
) -> Result<String, String> {
    validate_linkage(linkage)?;
    let authored = linkage
        .visible_references
        .iter()
        .map(|reference| (reference.symbol.as_str(), reference))
        .collect::<HashMap<_, _>>();
    let mut used = HashSet::<&str>::new();
    let mut appended_modules = HashSet::<&str>::new();
    let mut pending = Vec::<&LinkedFunctionReference>::new();
    let mut output = rewrite_visible_reference_stubs(entry_ll, &authored, &mut used, &mut pending)?;

    while let Some(reference) = pending.pop() {
        if !appended_modules.insert(reference.module_ll.as_str()) {
            continue;
        }
        let rewritten = rewrite_visible_reference_stubs(
            &reference.module_ll,
            &authored,
            &mut used,
            &mut pending,
        )?;
        append_dependency_module(&mut output, &rewritten);
    }

    Ok(output)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AirVisibleFunctionReference {
    stub_global: String,
    symbol: String,
}

fn rewrite_visible_reference_stubs<'a>(
    module: &str,
    authored: &HashMap<&str, &'a LinkedFunctionReference>,
    used: &mut HashSet<&'a str>,
    pending: &mut Vec<&'a LinkedFunctionReference>,
) -> Result<String, String> {
    let references = air_visible_function_references(module)?;
    let mut replacements = Vec::new();
    for reference in references {
        let dependency = authored.get(reference.symbol.as_str()).ok_or_else(|| {
            format!(
                "AIR visible function reference {:?} has no authored linked module",
                reference.symbol
            )
        })?;
        used.insert(dependency.symbol.as_str());
        pending.push(dependency);
        replacements.push((reference.stub_global, llvm_global(&reference.symbol)?));
    }
    let mut output = String::with_capacity(module.len());
    for line in module.lines() {
        if line
            .trim_start()
            .starts_with("!air.visible_function_references =")
        {
            continue;
        }
        let mut line = line.to_string();
        for (stub, target) in &replacements {
            line = line.replace(stub, target);
        }
        output.push_str(&line);
        output.push('\n');
    }
    Ok(output)
}

fn air_visible_function_references(
    module: &str,
) -> Result<Vec<AirVisibleFunctionReference>, String> {
    const MARKER: &str = "!\"air.visible_function_reference\", ptr ";
    let mut references = Vec::new();
    let mut stubs = HashMap::<String, String>::new();
    for line in module.lines() {
        let Some(marker) = line.find(MARKER) else {
            continue;
        };
        let body = &line[marker + MARKER.len()..];
        let at = body
            .find('@')
            .ok_or_else(|| format!("AIR visible-function reference has no stub global: {line}"))?;
        let stub_end = llvm_metadata_global_end(body, at).ok_or_else(|| {
            format!("AIR visible-function reference has malformed stub global: {line}")
        })?;
        let stub_global = body[at..stub_end].to_string();
        let rest = &body[stub_end..];
        let name_start = rest.find(", !\"").ok_or_else(|| {
            format!("AIR visible-function reference has no logical symbol: {line}")
        })? + 4;
        let (encoded, _) = llvm_quoted_string(&rest[name_start..]).ok_or_else(|| {
            format!("AIR visible-function reference has malformed logical symbol: {line}")
        })?;
        let symbol = decode_llvm_string(encoded)?;
        if let Some(previous) = stubs.insert(stub_global.clone(), symbol.clone()) {
            if previous != symbol {
                return Err(format!(
                    "AIR visible-function stub {stub_global} maps to both {previous:?} and {symbol:?}"
                ));
            }
            continue;
        }
        references.push(AirVisibleFunctionReference {
            stub_global,
            symbol,
        });
    }
    Ok(references)
}

fn llvm_metadata_global_end(text: &str, at: usize) -> Option<usize> {
    if text.as_bytes().get(at) != Some(&b'@') {
        return None;
    }
    if text.as_bytes().get(at + 1) != Some(&b'\"') {
        return text[at..]
            .find(|character: char| character == ',' || character.is_ascii_whitespace())
            .map(|relative| at + relative);
    }
    let (_, consumed) = llvm_quoted_string(&text[at + 2..])?;
    Some(at + 2 + consumed)
}

/// Return the encoded body and bytes consumed including the terminating quote.
fn llvm_quoted_string(text: &str) -> Option<(&str, usize)> {
    let mut escaped = false;
    for (index, byte) in text.bytes().enumerate() {
        if byte == b'\"' && !escaped {
            return Some((&text[..index], index + 1));
        }
        escaped = byte == b'\\' && !escaped;
        if byte != b'\\' {
            escaped = false;
        }
    }
    None
}

fn decode_llvm_string(encoded: &str) -> Result<String, String> {
    let bytes = encoded.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'\\' {
            let pair = bytes
                .get(index + 1..index + 3)
                .ok_or_else(|| format!("unterminated LLVM string escape in {encoded:?}"))?;
            let hex = std::str::from_utf8(pair)
                .ok()
                .and_then(|pair| u8::from_str_radix(pair, 16).ok())
                .ok_or_else(|| format!("invalid LLVM string escape in {encoded:?}"))?;
            decoded.push(hex);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).map_err(|_| format!("non-UTF-8 LLVM symbol in {encoded:?}"))
}

/// Resolve visible-function-table lookups and append their exact dependency modules.
///
/// Literal slots become direct calls. Dynamic slots become a direct call to a generated switch
/// dispatcher containing one arm per authored table entry. Calls not derived from an authored table
/// are left for the native emitter's ordinary indirect-call error.
pub fn specialize_visible_function_tables(
    entry_ll: &str,
    entry_name: &str,
    linkage: &LinkedFunctionLinkage,
) -> Result<String, String> {
    if linkage.visible_tables.is_empty() {
        return Ok(entry_ll.to_string());
    }
    validate_linkage(linkage)?;
    let signatures = function_signatures(entry_ll)?;
    let entry_global = llvm_global(entry_name)?;
    let entry_signature = signatures.get(&entry_global).ok_or_else(|| {
        format!("linked function-table entry function {entry_name:?} is not defined")
    })?;
    let mut flow = LinkedFlow::default();
    for table in &linkage.visible_tables {
        let parameter = entry_signature
            .parameters
            .get(table.parameter_index as usize)
            .ok_or_else(|| {
                format!(
                    "linked function table parameter {} exceeds entry {:?} arity {}",
                    table.parameter_index,
                    entry_name,
                    entry_signature.parameters.len()
                )
            })?
            .clone();
        if flow
            .table_parameters
            .insert(
                (entry_global.clone(), table.parameter_index as usize),
                table,
            )
            .is_some()
        {
            return Err(format!(
                "multiple linked function tables target entry parameter {} ({parameter})",
                table.parameter_index
            ));
        }
    }
    propagate_linked_flow(entry_ll, &signatures, &mut flow)?;

    struct Dispatcher<'a> {
        name: String,
        entries: Vec<&'a LinkedFunction>,
        return_type: String,
        argument_types: Vec<String>,
    }

    let mut tables_by_value = HashMap::<String, &LinkedFunctionTable>::new();
    let mut pointers = HashMap::<String, PointerTrace<'_>>::new();
    let mut sentinel_integers = HashMap::<String, PointerTrace<'_>>::new();
    let mut output = String::with_capacity(entry_ll.len());
    let mut linked_modules = Vec::<&str>::new();
    let mut linked_module_set = HashSet::<&str>::new();
    let mut dispatchers = Vec::<Dispatcher<'_>>::new();
    let mut dispatcher_names = HashMap::<String, String>::new();
    let mut presence_counter = 0usize;
    let mut current_function = None::<String>;
    for line in entry_ll.lines() {
        let mut line = line.to_string();
        let mut trimmed = line.trim_start();
        if let Some(global) = definition_global(trimmed) {
            let closes_inline = trimmed.ends_with('}');
            current_function = Some(global.clone());
            tables_by_value.clear();
            pointers.clear();
            sentinel_integers.clear();
            seed_function_values(
                &global,
                &signatures,
                &flow,
                &mut tables_by_value,
                &mut pointers,
            )?;
            line = append_pointer_slot_parameters(&line, &global, &signatures, &flow)?;
            output.push_str(&line);
            output.push('\n');
            if closes_inline {
                current_function = None;
            }
            continue;
        }
        if trimmed == "}" {
            output.push_str(&line);
            output.push('\n');
            current_function = None;
            tables_by_value.clear();
            pointers.clear();
            sentinel_integers.clear();
            continue;
        }
        let Some(current_global) = current_function.as_deref() else {
            output.push_str(&line);
            output.push('\n');
            continue;
        };
        if let Some(rewritten) = append_pointer_slot_call_arguments(
            &line,
            current_global,
            &signatures,
            &flow,
            &pointers,
        )? {
            output.push_str(&rewritten);
            output.push('\n');
            continue;
        }
        trimmed = line.trim_start();
        let mut table_query_replacement = None;
        if let Some((result, instruction)) = trimmed.split_once(" = ") {
            let result = result.trim();
            if instruction.contains("@air.get_function_pointer_visible_function_table(") {
                let arguments = call_arguments(instruction)?;
                if arguments.len() < 2 {
                    return Err(format!(
                        "visible function-table lookup has {} arguments: {trimmed}",
                        arguments.len()
                    ));
                }
                let table_value = value_operand(arguments[0]);
                if let Some(table) = tables_by_value.get(table_value) {
                    pointers.insert(
                        result.to_string(),
                        PointerTrace {
                            table,
                            index: arguments[1].to_string(),
                        },
                    );
                }
            } else if instruction.contains("@air.get_size_visible_function_table(") {
                let arguments = call_arguments(instruction)?;
                if let Some(table) = arguments
                    .first()
                    .and_then(|argument| tables_by_value.get(value_operand(argument)))
                {
                    let size = table.size;
                    table_query_replacement = Some(format!("{result} = add i32 0, {size}"));
                }
            } else if instruction.contains("@air.is_null_visible_function_table(") {
                let arguments = call_arguments(instruction)?;
                if arguments
                    .first()
                    .is_some_and(|argument| tables_by_value.contains_key(value_operand(argument)))
                {
                    table_query_replacement = Some(format!("{result} = or i1 false, false"));
                }
            } else if let Some((pointer_value, equal_to_null)) =
                null_pointer_comparison(instruction)
            {
                if let Some(pointer) = pointers.get(pointer_value) {
                    table_query_replacement = Some(authored_null_comparison(
                        result,
                        pointer.table,
                        &pointer.index,
                        equal_to_null,
                        &mut presence_counter,
                        entry_ll,
                    )?);
                }
            } else if instruction.starts_with("ptrtoint ") {
                if let Some(source) = cast_source_value(instruction) {
                    if let Some(pointer) = pointers.get(source).cloned() {
                        sentinel_integers.insert(result.to_string(), pointer);
                    }
                }
            } else if instruction.starts_with("trunc ") {
                if let Some(source) = cast_source_value(instruction) {
                    if let Some(pointer) = sentinel_integers.get(source).cloned() {
                        sentinel_integers.insert(result.to_string(), pointer);
                    }
                }
            } else if let Some((integer, equal_to_sentinel)) =
                opaque_sentinel_comparison(instruction)
            {
                if sentinel_integers.contains_key(integer) {
                    // Authored visible tables contain only null slots or linked functions. AIR's
                    // integer value 1 is Metal's reserved opaque-intersection sentinel, which this
                    // table kind cannot author. Fold only that exact ABI probe; arbitrary pointer
                    // integer observations remain unsupported.
                    table_query_replacement =
                        Some(format!("{result} = or i1 false, {}", !equal_to_sentinel));
                }
            } else if instruction.starts_with("bitcast ")
                || instruction.starts_with("addrspacecast ")
            {
                if let Some(source) = cast_source_value(instruction) {
                    if let Some(pointer) = pointers.get(source).cloned() {
                        pointers.insert(result.to_string(), pointer);
                    }
                }
            }
        }
        if let Some(replacement) = table_query_replacement {
            let leading = line.len() - trimmed.len();
            output.push_str(&line[..leading]);
            output.push_str(&replacement);
            output.push('\n');
            continue;
        }

        let Some((callee_start, callee_end, callee)) = indirect_call_callee(trimmed) else {
            output.push_str(&line);
            output.push('\n');
            continue;
        };
        let Some(pointer) = pointers.get(callee).cloned() else {
            output.push_str(&line);
            output.push('\n');
            continue;
        };
        let call = indirect_call_shape(trimmed, callee_start, callee_end)?;
        let (global, prepend_index) = if let Some(index) = integer_operand(&pointer.index) {
            let function = pointer
                .table
                .entries
                .iter()
                .find(|entry| entry.index == index)
                .ok_or_else(|| {
                    format!(
                        "linked visible function table parameter {} has no entry at slot {index}",
                        pointer.table.parameter_index
                    )
                })?;
            if !linked_function_matches_call(function, &call)? {
                return Err(format!(
                    "linked visible function {:?} at table parameter {} slot {index} does not match indirect call type",
                    function.symbol, pointer.table.parameter_index
                ));
            }
            if linked_module_set.insert(function.module_ll.as_str()) {
                linked_modules.push(&function.module_ll);
            }
            (llvm_global(&function.symbol)?, false)
        } else {
            if !pointer.index.trim_start().starts_with("i32 ") {
                return Err(format!(
                    "linked visible function table parameter {} has a non-i32 slot operand {:?}",
                    pointer.table.parameter_index, pointer.index
                ));
            }
            let key = format!(
                "{}|{}|{}",
                pointer.table.parameter_index,
                call.return_type,
                call.argument_types.join(",")
            );
            let name = if let Some(name) = dispatcher_names.get(&key) {
                name.clone()
            } else {
                let name = format!(
                    "metal2vulkan.linked.table.p{}.dispatch.{}",
                    pointer.table.parameter_index,
                    dispatchers.len()
                );
                dispatcher_names.insert(key, name.clone());
                let mut entries = Vec::new();
                for function in &pointer.table.entries {
                    if linked_function_matches_call(function, &call)? {
                        entries.push(function);
                    }
                }
                if entries.is_empty() {
                    return Err(format!(
                        "linked visible function table parameter {} has no function matching indirect call type",
                        pointer.table.parameter_index
                    ));
                }
                dispatchers.push(Dispatcher {
                    name: name.clone(),
                    entries: entries.clone(),
                    return_type: call.return_type,
                    argument_types: call.argument_types,
                });
                for function in entries {
                    if linked_module_set.insert(function.module_ll.as_str()) {
                        linked_modules.push(&function.module_ll);
                    }
                }
                name
            };
            (llvm_global(&name)?, true)
        };
        let leading = line.len() - trimmed.len();
        output.push_str(&line[..leading + callee_start]);
        output.push_str(&global);
        output.push_str(&trimmed[callee_end..=callee_end]);
        if prepend_index {
            output.push_str(&pointer.index);
            if trimmed.as_bytes().get(callee_end + 1) != Some(&b')') {
                output.push_str(", ");
            }
        }
        output.push_str(&trimmed[callee_end + 1..]);
        output.push('\n');
    }
    for dispatcher in dispatchers {
        output.push('\n');
        output.push_str(&dispatcher_definition(
            &dispatcher.name,
            &dispatcher.entries,
            &dispatcher.return_type,
            &dispatcher.argument_types,
        )?);
    }
    for module in linked_modules {
        output.push('\n');
        let resolved = specialize_visible_function_references(module, linkage)?;
        append_dependency_module(&mut output, &resolved);
    }
    Ok(output)
}

fn null_pointer_comparison(instruction: &str) -> Option<(&str, bool)> {
    let (equal_to_null, operands) = if let Some(operands) = instruction.strip_prefix("icmp eq ") {
        (true, operands)
    } else {
        (false, instruction.strip_prefix("icmp ne ")?)
    };
    let operands = split_top_level(operands, ',');
    if operands.len() != 2 {
        return None;
    }
    let left = value_operand(operands[0]);
    let right = value_operand(operands[1]);
    match (left, right) {
        (pointer, "null") if pointer.starts_with('%') => Some((pointer, equal_to_null)),
        ("null", pointer) if pointer.starts_with('%') => Some((pointer, equal_to_null)),
        _ => None,
    }
}

fn opaque_sentinel_comparison(instruction: &str) -> Option<(&str, bool)> {
    let (equal_to_sentinel, operands) = if let Some(operands) = instruction.strip_prefix("icmp eq ")
    {
        (true, operands)
    } else {
        (false, instruction.strip_prefix("icmp ne ")?)
    };
    let operands = split_top_level(operands, ',');
    if operands.len() != 2 {
        return None;
    }
    let left = value_operand(operands[0]);
    let right = value_operand(operands[1]);
    match (left, right) {
        (integer, "1") if integer.starts_with('%') => Some((integer, equal_to_sentinel)),
        ("1", integer) if integer.starts_with('%') => Some((integer, equal_to_sentinel)),
        _ => None,
    }
}

fn authored_null_comparison(
    result: &str,
    table: &LinkedFunctionTable,
    index: &str,
    equal_to_null: bool,
    counter: &mut usize,
    module: &str,
) -> Result<String, String> {
    if let Some(index) = integer_operand(index) {
        let populated = table.entries.iter().any(|entry| entry.index == index);
        return Ok(format!(
            "{result} = or i1 false, {}",
            populated != equal_to_null
        ));
    }
    if !index.trim_start().starts_with("i32 ") {
        return Err(format!(
            "linked visible function table parameter {} has a non-i32 slot operand {index:?}",
            table.parameter_index
        ));
    }
    let slot = value_operand(index);
    let mut lines = Vec::new();
    let mut present = None::<String>;
    for entry in &table.entries {
        let comparison = fresh_presence_value(module, counter);
        lines.push(format!(
            "{comparison} = icmp eq i32 {slot}, {}",
            entry.index
        ));
        present = Some(if let Some(previous) = present {
            let combined = fresh_presence_value(module, counter);
            lines.push(format!("{combined} = or i1 {previous}, {comparison}"));
            combined
        } else {
            comparison
        });
    }
    let present = present.unwrap_or_else(|| "false".into());
    if equal_to_null {
        lines.push(format!("{result} = xor i1 {present}, true"));
    } else {
        lines.push(format!("{result} = or i1 {present}, false"));
    }
    Ok(lines.join("\n"))
}

fn fresh_presence_value(module: &str, counter: &mut usize) -> String {
    loop {
        let value = format!("%metal2vulkan.table.present.{}", *counter);
        *counter += 1;
        if !contains_llvm_value(module, &value) {
            return value;
        }
    }
}

fn contains_llvm_value(text: &str, value: &str) -> bool {
    text.match_indices(value).any(|(start, _)| {
        let end = start + value.len();
        let is_boundary = |byte: Option<&u8>| {
            byte.is_none_or(|byte| {
                !byte.is_ascii_alphanumeric() && !matches!(byte, b'_' | b'.' | b'$' | b'-')
            })
        };
        is_boundary(
            start
                .checked_sub(1)
                .and_then(|index| text.as_bytes().get(index)),
        ) && is_boundary(text.as_bytes().get(end))
    })
}

/// Append executable linkage content without importing a second module's numeric metadata arena.
///
/// LLVM metadata ids and attribute-group ids are module-local. Concatenating their definitions
/// would allow a helper's `!0`/`#0` to replace the entry module's stage metadata or attributes.
/// Native parsing consumes instruction semantics directly and does not need those definition
/// tables, so dependency modules contribute types, globals, declarations, and function bodies only.
fn append_dependency_module(output: &mut String, module: &str) {
    for line in module.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("; ModuleID =")
            || trimmed.starts_with("source_filename =")
            || trimmed.starts_with("target datalayout =")
            || trimmed.starts_with("target triple =")
            || trimmed.starts_with("attributes #")
            || trimmed.starts_with('!')
        {
            continue;
        }
        output.push_str(line);
        output.push('\n');
    }
}

fn validate_linkage(linkage: &LinkedFunctionLinkage) -> Result<(), String> {
    let mut parameters = HashSet::new();
    let mut symbols = HashMap::<&str, &str>::new();
    for reference in &linkage.visible_references {
        if !module_defines(&reference.module_ll, &reference.symbol) {
            return Err(format!(
                "linked module does not define authored visible function reference {:?}",
                reference.symbol
            ));
        }
        if symbols
            .insert(&reference.symbol, &reference.module_ll)
            .is_some()
        {
            return Err(format!(
                "duplicate authored visible function reference {:?}",
                reference.symbol
            ));
        }
    }
    for table in &linkage.visible_tables {
        if !parameters.insert(table.parameter_index) {
            return Err(format!(
                "duplicate linked visible function-table parameter {}",
                table.parameter_index
            ));
        }
        if table.size == 0 {
            return Err(format!(
                "linked visible function-table parameter {} has zero size",
                table.parameter_index
            ));
        }
        let mut previous = None;
        for entry in &table.entries {
            if entry.index >= table.size {
                return Err(format!(
                    "linked visible function-table parameter {} entry {} exceeds size {}",
                    table.parameter_index, entry.index, table.size
                ));
            }
            if previous.is_some_and(|index| index >= entry.index) {
                return Err(format!(
                    "linked visible function-table parameter {} entries must be sorted and unique",
                    table.parameter_index
                ));
            }
            previous = Some(entry.index);
            if !module_defines(&entry.module_ll, &entry.symbol) {
                return Err(format!(
                    "linked module does not define authored function {:?}",
                    entry.symbol
                ));
            }
            if let Some(previous_module) = symbols.insert(&entry.symbol, &entry.module_ll) {
                if previous_module != entry.module_ll {
                    return Err(format!(
                        "linked function symbol {:?} is defined by multiple modules",
                        entry.symbol
                    ));
                }
            }
        }
    }
    Ok(())
}

fn function_signatures(ll: &str) -> Result<HashMap<String, FunctionSignature>, String> {
    let mut functions = HashMap::new();
    let mut signature = String::new();
    let mut collecting = false;
    for line in ll.lines() {
        let trimmed = line.trim_start();
        if !collecting && trimmed.starts_with("define ") {
            collecting = true;
            signature.clear();
        }
        if collecting {
            signature.push_str(trimmed);
            signature.push(' ');
            if trimmed.contains('{') {
                let global = definition_global(&signature).ok_or_else(|| {
                    format!("linked function definition has no global: {signature}")
                })?;
                let open = signature
                    .find(&format!("{global}("))
                    .map(|index| index + global.len())
                    .ok_or_else(|| format!("linked function {global} has no parameter list"))?;
                let close = matching_paren(&signature, open).ok_or_else(|| {
                    format!("linked function {global} has an unterminated parameter list")
                })?;
                let parameters = parameter_values(&signature[open + 1..close], &global)?;
                if functions
                    .insert(global.clone(), FunctionSignature { parameters })
                    .is_some()
                {
                    return Err("linked module contains duplicate function definitions".into());
                }
                collecting = false;
            }
        }
    }
    if collecting {
        return Err("linked module has an unterminated function definition header".into());
    }
    Ok(functions)
}

fn parameter_values(body: &str, global: &str) -> Result<Vec<String>, String> {
    let body = body.trim();
    if body.is_empty() {
        return Ok(Vec::new());
    }
    split_top_level(body, ',')
        .into_iter()
        .map(|parameter| {
            parameter
                .split_whitespace()
                .last()
                .filter(|value| value.starts_with('%'))
                .map(str::to_string)
                .ok_or_else(|| format!("function {global} parameter has no SSA name: {parameter}"))
        })
        .collect()
}

fn definition_global(line: &str) -> Option<String> {
    let line = line.trim_start();
    line.starts_with("define ")
        .then(|| line.find('@'))
        .flatten()
        .and_then(|at| global_token(line, at).map(|(global, _)| global.to_string()))
}

fn global_token(text: &str, at: usize) -> Option<(&str, usize)> {
    if text.as_bytes().get(at) != Some(&b'@') {
        return None;
    }
    if text.as_bytes().get(at + 1) == Some(&b'"') {
        let mut escaped = false;
        for (relative, byte) in text.as_bytes()[at + 2..].iter().enumerate() {
            if *byte == b'"' && !escaped {
                let end = at + 2 + relative + 1;
                let open = text[end..].find('(')? + end;
                return Some((&text[at..end], open));
            }
            escaped = *byte == b'\\' && !escaped;
            if *byte != b'\\' {
                escaped = false;
            }
        }
        None
    } else {
        let open = text[at..].find('(')? + at;
        Some((text[at..open].trim_end(), open))
    }
}

fn named_call(line: &str) -> Option<(&str, usize, usize, Vec<&str>)> {
    let call = line.find("call ")?;
    let at = line[call + 5..].find('@')? + call + 5;
    let (global, open) = global_token(line, at)?;
    let close = matching_paren(line, open)?;
    let arguments = split_top_level(&line[open + 1..close], ',');
    Some((global, open, close, arguments))
}

fn seed_function_values<'a>(
    global: &str,
    signatures: &HashMap<String, FunctionSignature>,
    flow: &LinkedFlow<'a>,
    tables: &mut HashMap<String, &'a LinkedFunctionTable>,
    pointers: &mut HashMap<String, PointerTrace<'a>>,
) -> Result<(), String> {
    let signature = signatures
        .get(global)
        .ok_or_else(|| format!("missing parsed signature for {global}"))?;
    for (ordinal, parameter) in signature.parameters.iter().enumerate() {
        if let Some(table) = flow.table_parameters.get(&(global.to_string(), ordinal)) {
            tables.insert(parameter.clone(), *table);
        }
        if let Some(table) = flow.pointer_parameters.get(&(global.to_string(), ordinal)) {
            pointers.insert(
                parameter.clone(),
                PointerTrace {
                    table,
                    index: format!("i32 {}", pointer_slot_parameter(global, ordinal, signature)),
                },
            );
        }
    }
    Ok(())
}

fn propagate_linked_flow<'a>(
    ll: &str,
    signatures: &HashMap<String, FunctionSignature>,
    flow: &mut LinkedFlow<'a>,
) -> Result<(), String> {
    loop {
        let mut changed = false;
        let mut current = None::<String>;
        let mut tables = HashMap::<String, &'a LinkedFunctionTable>::new();
        let mut pointers = HashMap::<String, &'a LinkedFunctionTable>::new();
        for line in ll.lines() {
            let trimmed = line.trim_start();
            if let Some(global) = definition_global(trimmed) {
                current = Some(global.clone());
                tables.clear();
                pointers.clear();
                let signature = signatures
                    .get(&global)
                    .ok_or_else(|| format!("missing parsed signature for {global}"))?;
                for (ordinal, parameter) in signature.parameters.iter().enumerate() {
                    if let Some(table) = flow.table_parameters.get(&(global.clone(), ordinal)) {
                        tables.insert(parameter.clone(), *table);
                    }
                    if let Some(table) = flow.pointer_parameters.get(&(global.clone(), ordinal)) {
                        pointers.insert(parameter.clone(), *table);
                    }
                }
                continue;
            }
            if trimmed == "}" {
                current = None;
                continue;
            }
            let Some(current_global) = current.as_deref() else {
                continue;
            };
            if let Some((result, instruction)) = trimmed.split_once(" = ") {
                let result = result.trim();
                if instruction.contains("@air.get_function_pointer_visible_function_table(") {
                    let arguments = call_arguments(instruction)?;
                    if let Some(table) = arguments
                        .first()
                        .and_then(|argument| tables.get(value_operand(argument)))
                    {
                        pointers.insert(result.to_string(), *table);
                    }
                } else if instruction.starts_with("bitcast ")
                    || instruction.starts_with("addrspacecast ")
                {
                    if let Some(table) = audit_cast_pointer(instruction, &pointers) {
                        pointers.insert(result.to_string(), table);
                    }
                } else if instruction.starts_with("phi ") || instruction.starts_with("select ") {
                    let used = pointers
                        .iter()
                        .filter(|(value, _)| contains_llvm_value(instruction, value))
                        .map(|(_, table)| *table)
                        .collect::<Vec<_>>();
                    if let Some(table) = same_table(&used) {
                        pointers.insert(result.to_string(), table);
                    }
                }
            }
            let Some((callee, _, _, arguments)) = named_call(trimmed) else {
                continue;
            };
            let Some(callee_signature) = signatures.get(callee) else {
                continue;
            };
            for (ordinal, argument) in arguments
                .iter()
                .take(callee_signature.parameters.len())
                .enumerate()
            {
                let value = value_operand(argument);
                if let Some(table) = tables.get(value) {
                    changed |= insert_flow_parameter(
                        &mut flow.table_parameters,
                        (callee.to_string(), ordinal),
                        table,
                        "function table",
                    )?;
                }
                if let Some(table) = pointers.get(value) {
                    changed |= insert_flow_parameter(
                        &mut flow.pointer_parameters,
                        (callee.to_string(), ordinal),
                        table,
                        "function pointer",
                    )?;
                }
            }
            let _ = current_global;
        }
        if !changed {
            return Ok(());
        }
    }
}

fn audit_cast_pointer<'a>(
    instruction: &str,
    pointers: &HashMap<String, &'a LinkedFunctionTable>,
) -> Option<&'a LinkedFunctionTable> {
    cast_source_value(instruction).and_then(|source| pointers.get(source).copied())
}

fn same_table<'a>(tables: &[&'a LinkedFunctionTable]) -> Option<&'a LinkedFunctionTable> {
    let first = *tables.first()?;
    tables
        .iter()
        .all(|table| std::ptr::eq(*table, first))
        .then_some(first)
}

fn insert_flow_parameter<'a>(
    parameters: &mut HashMap<(String, usize), &'a LinkedFunctionTable>,
    key: (String, usize),
    table: &'a LinkedFunctionTable,
    kind: &str,
) -> Result<bool, String> {
    if let Some(previous) = parameters.get(&key) {
        if !std::ptr::eq(*previous, table) {
            return Err(format!(
                "linked {kind} parameter {} of {} receives multiple authored tables",
                key.1, key.0
            ));
        }
        Ok(false)
    } else {
        parameters.insert(key, table);
        Ok(true)
    }
}

fn pointer_slot_parameter(_global: &str, ordinal: usize, signature: &FunctionSignature) -> String {
    let base = format!("%metal2vulkan.table.param{ordinal}.slot");
    if !signature.parameters.contains(&base) {
        return base;
    }
    for suffix in 1usize.. {
        let candidate = format!("{base}.{suffix}");
        if !signature.parameters.contains(&candidate) {
            return candidate;
        }
    }
    unreachable!()
}

fn pointer_parameter_ordinals(flow: &LinkedFlow<'_>, global: &str) -> Vec<usize> {
    let mut ordinals = flow
        .pointer_parameters
        .keys()
        .filter_map(|(function, ordinal)| (function == global).then_some(*ordinal))
        .collect::<Vec<_>>();
    ordinals.sort_unstable();
    ordinals
}

fn append_pointer_slot_parameters(
    line: &str,
    global: &str,
    signatures: &HashMap<String, FunctionSignature>,
    flow: &LinkedFlow<'_>,
) -> Result<String, String> {
    let ordinals = pointer_parameter_ordinals(flow, global);
    if ordinals.is_empty() {
        return Ok(line.to_string());
    }
    let signature = signatures
        .get(global)
        .ok_or_else(|| format!("missing parsed signature for {global}"))?;
    let at = line
        .find(global)
        .ok_or_else(|| format!("definition line does not contain {global}"))?;
    let open = line[at + global.len()..]
        .find('(')
        .map(|offset| offset + at + global.len())
        .ok_or_else(|| format!("definition {global} has no parameter list"))?;
    let close = matching_paren(line, open)
        .ok_or_else(|| format!("multiline definition parameters for {global} are unsupported"))?;
    let additions = ordinals
        .into_iter()
        .map(|ordinal| format!("i32 {}", pointer_slot_parameter(global, ordinal, signature)))
        .collect::<Vec<_>>()
        .join(", ");
    let separator = if line[open + 1..close].trim().is_empty() {
        ""
    } else {
        ", "
    };
    Ok(format!(
        "{}{}{}{}",
        &line[..close],
        separator,
        additions,
        &line[close..]
    ))
}

fn append_pointer_slot_call_arguments(
    line: &str,
    caller: &str,
    signatures: &HashMap<String, FunctionSignature>,
    flow: &LinkedFlow<'_>,
    pointers: &HashMap<String, PointerTrace<'_>>,
) -> Result<Option<String>, String> {
    let Some((callee, open, close, arguments)) = named_call(line) else {
        return Ok(None);
    };
    let ordinals = pointer_parameter_ordinals(flow, callee);
    if ordinals.is_empty() || !signatures.contains_key(callee) {
        return Ok(None);
    }
    let mut additions = Vec::new();
    for ordinal in ordinals {
        let argument = arguments.get(ordinal).ok_or_else(|| {
            format!("call from {caller} to {callee} omits linked pointer parameter {ordinal}")
        })?;
        let value = value_operand(argument);
        let pointer = pointers.get(value).ok_or_else(|| {
            format!(
                "call from {caller} to {callee} passes untraced linked pointer parameter {ordinal}: {value}"
            )
        })?;
        additions.push(pointer.index.clone());
    }
    let separator = if line[open + 1..close].trim().is_empty() {
        ""
    } else {
        ", "
    };
    Ok(Some(format!(
        "{}{}{}{}",
        &line[..close],
        separator,
        additions.join(", "),
        &line[close..]
    )))
}

fn call_arguments(instruction: &str) -> Result<Vec<&str>, String> {
    let open = instruction
        .find('(')
        .ok_or_else(|| format!("call has no argument list: {instruction}"))?;
    let close = matching_paren(instruction, open)
        .ok_or_else(|| format!("call has an unterminated argument list: {instruction}"))?;
    Ok(split_top_level(&instruction[open + 1..close], ','))
}

fn value_operand(argument: &str) -> &str {
    argument.split_whitespace().last().unwrap_or_default()
}

fn integer_operand(argument: &str) -> Option<u32> {
    value_operand(argument).parse().ok()
}

fn cast_source_value(instruction: &str) -> Option<&str> {
    let (_, source_and_destination) = instruction.split_once(' ')?;
    let (source, _) = source_and_destination.rsplit_once(" to ")?;
    source
        .split_whitespace()
        .last()
        .filter(|value| value.starts_with('%'))
}

fn indirect_call_callee(line: &str) -> Option<(usize, usize, &str)> {
    let call = line.find("call ")?;
    let after_call = &line[call + 5..];
    let open = after_call.find('(')? + call + 5;
    let head = &line[..open];
    let end = head.len();
    let start = head.rfind(char::is_whitespace).map_or(0, |index| index + 1);
    let callee = &line[start..end];
    callee.starts_with('%').then_some((start, end, callee))
}

struct IndirectCallShape {
    return_type: String,
    argument_types: Vec<String>,
}

fn indirect_call_shape(
    line: &str,
    callee_start: usize,
    callee_end: usize,
) -> Result<IndirectCallShape, String> {
    let call = line[..callee_start]
        .rfind("call ")
        .ok_or_else(|| format!("indirect call has no call opcode: {line}"))?;
    let mut return_type = line[call + 5..callee_start].trim();
    while let Some((first, rest)) = return_type.split_once(' ') {
        if matches!(
            first,
            "fast" | "nnan" | "ninf" | "nsz" | "arcp" | "contract" | "afn" | "reassoc"
        ) {
            return_type = rest.trim_start();
        } else {
            break;
        }
    }
    if return_type.is_empty() {
        return Err(format!("indirect call has no return type: {line}"));
    }
    let close = matching_paren(line, callee_end)
        .ok_or_else(|| format!("indirect call has an unterminated argument list: {line}"))?;
    let arguments = &line[callee_end + 1..close];
    let argument_types = if arguments.trim().is_empty() {
        Vec::new()
    } else {
        split_top_level(arguments, ',')
            .into_iter()
            .map(argument_type)
            .collect::<Result<Vec<_>, _>>()?
    };
    Ok(IndirectCallShape {
        return_type: return_type.to_string(),
        argument_types,
    })
}

fn argument_type(argument: &str) -> Result<String, String> {
    let value = value_operand(argument);
    let type_end = argument
        .rfind(value)
        .filter(|index| *index > 0)
        .ok_or_else(|| format!("linked indirect-call argument has no typed value: {argument}"))?;
    let ty = argument[..type_end].trim_end();
    if ty.is_empty() {
        return Err(format!(
            "linked indirect-call argument has no type: {argument}"
        ));
    }
    Ok(ty.to_string())
}

fn linked_function_matches_call(
    function: &LinkedFunction,
    call: &IndirectCallShape,
) -> Result<bool, String> {
    let (return_type, argument_types) = linked_function_type(function)?;
    let call_return = crate::native::parse_llvm_type_prefix(&call.return_type)?;
    let call_arguments = call
        .argument_types
        .iter()
        .map(|argument| crate::native::parse_llvm_type_prefix(argument))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(return_type == call_return && argument_types == call_arguments)
}

fn linked_function_type(
    function: &LinkedFunction,
) -> Result<(crate::native::ir::LlType, Vec<crate::native::ir::LlType>), String> {
    let global = llvm_global(&function.symbol)?;
    let mut signature = String::new();
    let mut collecting = false;
    for line in function.module_ll.lines() {
        let trimmed = line.trim_start();
        if !collecting && trimmed.starts_with("define ") && trimmed.contains(&format!("{global}("))
        {
            collecting = true;
        }
        if collecting {
            signature.push_str(trimmed);
            signature.push(' ');
            if trimmed.contains('{') {
                break;
            }
        }
    }
    if signature.is_empty() {
        return Err(format!(
            "linked module does not define authored function {:?}",
            function.symbol
        ));
    }
    let at = signature
        .find(&global)
        .ok_or_else(|| format!("linked function {:?} has no global", function.symbol))?;
    let open = signature[at + global.len()..]
        .find('(')
        .map(|offset| offset + at + global.len())
        .ok_or_else(|| format!("linked function {:?} has no parameters", function.symbol))?;
    let close = matching_paren(&signature, open).ok_or_else(|| {
        format!(
            "linked function {:?} has unterminated parameters",
            function.symbol
        )
    })?;
    let return_type = crate::native::parse_llvm_return_type(&signature[..at])?;
    let argument_types = if signature[open + 1..close].trim().is_empty() {
        Vec::new()
    } else {
        split_top_level(&signature[open + 1..close], ',')
            .into_iter()
            .map(crate::native::parse_llvm_type_prefix)
            .collect::<Result<Vec<_>, _>>()?
    };
    Ok((return_type, argument_types))
}

fn dispatcher_definition(
    name: &str,
    entries: &[&LinkedFunction],
    return_type: &str,
    argument_types: &[String],
) -> Result<String, String> {
    let global = llvm_global(name)?;
    let parameters = argument_types
        .iter()
        .enumerate()
        .map(|(index, ty)| format!("{ty} %arg{index}"))
        .collect::<Vec<_>>();
    let forwarded = argument_types
        .iter()
        .enumerate()
        .map(|(index, ty)| format!("{ty} %arg{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let parameter_separator = if parameters.is_empty() { "" } else { ", " };
    let mut output = format!(
        "define internal {return_type} {global}(i32 %metal2vulkan_slot{}{}) {{\nentry:\n  switch i32 %metal2vulkan_slot, label %invalid [",
        parameter_separator,
        parameters.join(", ")
    );
    for entry in entries {
        output.push_str(&format!(
            " i32 {}, label %case_{}",
            entry.index, entry.index
        ));
    }
    output.push_str(" ]\n\n");
    let is_void = return_type == "void";
    for entry in entries {
        output.push_str(&format!("case_{}:\n", entry.index));
        let target = llvm_global(&entry.symbol)?;
        if is_void {
            output.push_str(&format!(
                "  call void {target}({forwarded})\n  ret void\n\n"
            ));
        } else {
            output.push_str(&format!(
                "  %result_{} = call {return_type} {target}({forwarded})\n  br label %exit\n\n",
                entry.index
            ));
        }
    }
    output.push_str("invalid:\n  unreachable\n");
    if !is_void {
        output.push_str("\nexit:\n  %result = phi ");
        output.push_str(return_type);
        output.push(' ');
        for (ordinal, entry) in entries.iter().enumerate() {
            if ordinal != 0 {
                output.push_str(", ");
            }
            output.push_str(&format!(
                "[ %result_{}, %case_{} ]",
                entry.index, entry.index
            ));
        }
        output.push_str(&format!("\n  ret {return_type} %result\n"));
    }
    output.push_str("}\n");
    Ok(output)
}

fn llvm_global(symbol: &str) -> Result<String, String> {
    if symbol.is_empty() || symbol.contains(['\n', '\r', '\0']) {
        return Err(format!("invalid linked LLVM function symbol {symbol:?}"));
    }
    if symbol
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'$' | b'-'))
    {
        Ok(format!("@{symbol}"))
    } else {
        Ok(format!(
            "@\"{}\"",
            symbol.replace('\\', "\\5C").replace('"', "\\22")
        ))
    }
}

fn module_defines(module: &str, symbol: &str) -> bool {
    llvm_global(symbol).is_ok_and(|global| {
        module.lines().any(|line| {
            let line = line.trim_start();
            line.starts_with("define ") && line.contains(&format!("{global}("))
        })
    })
}

fn matching_paren(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (relative, byte) in text.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'(' => depth += 1,
            b')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(open + relative);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_top_level(text: &str, delimiter: char) -> Vec<&str> {
    let mut depth = 0usize;
    let mut start = 0usize;
    let mut fields = Vec::new();
    for (index, ch) in text.char_indices() {
        match ch {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' | '>' if depth > 0 => depth -= 1,
            _ if ch == delimiter && depth == 0 => {
                fields.push(text[start..index].trim());
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    fields.push(text[start..].trim());
    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_visible_references_close_the_authored_dependency_graph() {
        let entry = r#"define void @main() {
entry:
  %value = call i32 @leaf.MTL_VISIBLE_FN_REF(i32 7)
  ret void
}
declare i32 @leaf.MTL_VISIBLE_FN_REF(i32) section "air.externally_defined"
!air.visible_function_references = !{!0}
!0 = !{!"air.visible_function_reference", ptr @leaf.MTL_VISIBLE_FN_REF, !"leaf"}
"#;
        let leaf = r#"define i32 @leaf(i32 %value) {
entry:
  %result = call i32 @base.MTL_VISIBLE_FN_REF(i32 %value)
  ret i32 %result
}
declare i32 @base.MTL_VISIBLE_FN_REF(i32) section "air.externally_defined"
!air.visible_function_references = !{!7}
!7 = !{!"air.visible_function_reference", ptr @base.MTL_VISIBLE_FN_REF, !"base"}
"#;
        let base = "define i32 @base(i32 %value) {\nentry:\n  ret i32 %value\n}\n";
        let linkage = LinkedFunctionLinkage {
            visible_references: vec![
                LinkedFunctionReference {
                    symbol: "base".into(),
                    module_ll: base.into(),
                },
                LinkedFunctionReference {
                    symbol: "leaf".into(),
                    module_ll: leaf.into(),
                },
            ],
            visible_tables: vec![],
            intersection_tables: vec![],
        };

        let specialized = specialize_visible_function_references(entry, &linkage).unwrap();
        assert!(specialized.contains("call i32 @leaf(i32 7)"));
        assert!(specialized.contains("call i32 @base(i32 %value)"));
        assert!(specialized.contains("define i32 @leaf("));
        assert!(specialized.contains("define i32 @base("));
        assert!(!specialized.contains("MTL_VISIBLE_FN_REF"));
        assert!(!specialized.contains("!air.visible_function_references"));
    }

    #[test]
    fn direct_visible_reference_requires_an_exact_authored_symbol() {
        let entry = r#"define void @main() { ret void }
!air.visible_function_references = !{!0}
!0 = !{!"air.visible_function_reference", ptr @missing.MTL_VISIBLE_FN_REF, !"missing"}
"#;
        let error = specialize_visible_function_references(
            entry,
            &LinkedFunctionLinkage {
                visible_references: vec![],
                visible_tables: vec![],
                intersection_tables: vec![],
            },
        )
        .unwrap_err();
        assert!(error.contains("has no authored linked module"), "{error}");
    }

    const ENTRY: &str = r#"
define void @main(ptr addrspace(1) %output, ptr addrspace(1) %table) {
entry:
  %fp = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 0)
  %cast = bitcast ptr %fp to ptr
  %value = call i32 %cast(i32 41)
  store i32 %value, ptr addrspace(1) %output, align 4
  ret void
}
"#;

    #[test]
    fn linkage_retains_intersection_tables_even_without_visible_tables() {
        let linkage = LinkedFunctionLinkage {
            visible_references: vec![],
            visible_tables: vec![],
            intersection_tables: vec![IntersectionFunctionTable {
                source: IntersectionFunctionTableSource::Parameter { parameter_index: 2 },
                size: 1,
                entries: vec![IntersectionFunctionEntry::Linked(LinkedFunction {
                    index: 0,
                    symbol: "intersection".into(),
                    module_ll: "define i1 @intersection() { ret i1 true }".into(),
                })],
            }],
        };
        assert!(!linkage.is_empty());
        assert_eq!(
            linkage.intersection_tables[0].source,
            IntersectionFunctionTableSource::Parameter { parameter_index: 2 }
        );
        // Visible specialization must not discard or try to reinterpret intersection-table
        // entries. Ray-query specialization consumes them in its own ABI-aware pass.
        assert_eq!(
            specialize_visible_function_tables(ENTRY, "main", &linkage).unwrap(),
            ENTRY
        );
    }

    #[test]
    fn authored_intersection_table_setter_is_consumed_only_for_traced_destination() {
        let entry = r#"
define void @main(ptr addrspace(1) %destination, ptr addrspace(1) %source, i32 %index) {
entry:
  call void @air.set_buffer_intersection_function_table.p1i8(ptr addrspace(1) %destination, ptr addrspace(1) %source, i32 %index)
  ret void
}
declare void @air.set_buffer_intersection_function_table.p1i8(ptr addrspace(1), ptr addrspace(1), i32)
"#;
        let table = |parameter_index| IntersectionFunctionTable {
            source: IntersectionFunctionTableSource::Parameter { parameter_index },
            size: 1,
            entries: vec![],
        };
        let linkage = |parameter_index| LinkedFunctionLinkage {
            visible_references: vec![],
            visible_tables: vec![],
            intersection_tables: vec![table(parameter_index)],
        };

        let specialized =
            specialize_opaque_triangle_intersection_tables(entry, "main", &linkage(0)).unwrap();
        assert_eq!(
            specialized
                .lines()
                .filter(|line| line.contains("air.set_buffer_intersection_function_table"))
                .count(),
            1,
            "only the declaration remains after authored destination specialization"
        );

        let untraced =
            specialize_opaque_triangle_intersection_tables(entry, "main", &linkage(1)).unwrap();
        assert!(untraced.lines().any(|line| {
            line.contains("call void @air.set_buffer_intersection_function_table")
        }));
    }

    #[test]
    fn fully_opaque_triangle_table_specializes_callback_query() {
        let entry = r#"
define void @main(ptr addrspace(1) %output, ptr addrspace(1) %table, ptr addrspace(1) %as) {
entry:
  %hit = call { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } @air.intersect.intersection_function_buffer.triangle_data(<3 x float> zeroinitializer, <3 x float> zeroinitializer, float 0.0, float 1.0, ptr addrspace(1) %as, ptr addrspace(1) %table, i64 0, i64 1, ptr null, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 -1, i32 0, i1 false, i1 false)
  ret void
}
"#;
        let linkage = LinkedFunctionLinkage {
            visible_references: vec![],
            visible_tables: vec![],
            intersection_tables: vec![IntersectionFunctionTable {
                source: IntersectionFunctionTableSource::Parameter { parameter_index: 1 },
                size: 1,
                entries: vec![IntersectionFunctionEntry::OpaqueTriangle {
                    index: 0,
                    signature: vec![
                        IntersectionFunctionSignature::TriangleData,
                        IntersectionFunctionSignature::IntersectionFunctionBuffer,
                    ],
                }],
            }],
        };
        let specialized =
            specialize_opaque_triangle_intersection_tables(entry, "main", &linkage).unwrap();
        assert!(specialized.contains("@air.intersect.triangle_data("));
        assert!(!specialized.contains("@air.intersect.intersection_function_buffer"));
        let call = specialized
            .lines()
            .find(|line| line.contains("%hit = call"))
            .unwrap();
        assert_eq!(named_call(call).unwrap().3.len(), 18);
    }

    #[test]
    fn null_or_signature_mismatched_slots_do_not_erase_callbacks() {
        let entry = r#"
define void @main(ptr addrspace(1) %table, ptr addrspace(1) %as) {
entry:
  %hit = call { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } @air.intersect.intersection_function_buffer.triangle_data(<3 x float> zeroinitializer, <3 x float> zeroinitializer, float 0.0, float 1.0, ptr addrspace(1) %as, ptr addrspace(1) %table, i64 0, i64 1, ptr null, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 -1, i32 0, i1 false, i1 false)
  ret void
}
"#;
        for entries in [
            vec![],
            vec![IntersectionFunctionEntry::OpaqueTriangle {
                index: 0,
                signature: vec![IntersectionFunctionSignature::IntersectionFunctionBuffer],
            }],
        ] {
            let linkage = LinkedFunctionLinkage {
                visible_references: vec![],
                visible_tables: vec![],
                intersection_tables: vec![IntersectionFunctionTable {
                    source: IntersectionFunctionTableSource::Parameter { parameter_index: 0 },
                    size: 1,
                    entries,
                }],
            };
            assert_eq!(
                specialize_opaque_triangle_intersection_tables(entry, "main", &linkage).unwrap(),
                entry
            );
        }
    }

    #[test]
    fn opaque_triangle_user_data_family_removes_the_fifth_callback_operand() {
        let entry = r#"
define void @main(ptr addrspace(1) %table, ptr addrspace(1) %as) {
entry:
  %hit = call { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } @air.intersect.intersection_function_buffer.triangle_data.user_data(<3 x float> zeroinitializer, <3 x float> zeroinitializer, float 0.0, float 1.0, ptr addrspace(1) %as, ptr addrspace(1) %table, i64 1, i64 8, ptr addrspace(1) null, ptr null, i64 0, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i1 false, i32 91, i32 92)
  ret void
}
"#;
        let linkage = LinkedFunctionLinkage {
            visible_references: vec![],
            visible_tables: vec![],
            intersection_tables: vec![IntersectionFunctionTable {
                source: IntersectionFunctionTableSource::Parameter { parameter_index: 0 },
                size: 1,
                entries: vec![IntersectionFunctionEntry::OpaqueTriangle {
                    index: 0,
                    signature: vec![
                        IntersectionFunctionSignature::TriangleData,
                        IntersectionFunctionSignature::IntersectionFunctionBuffer,
                        IntersectionFunctionSignature::UserData,
                    ],
                }],
            }],
        };
        let specialized =
            specialize_opaque_triangle_intersection_tables(entry, "main", &linkage).unwrap();
        let call = specialized
            .lines()
            .find(|line| line.contains("%hit = call"))
            .unwrap();
        assert!(call.contains("@air.intersect.triangle_data("));
        assert_eq!(named_call(call).unwrap().3.len(), 18);
        assert!(call.contains("ptr null, i64 0, i32 10"), "{call}");
        assert!(!call.contains("i32 91"), "{call}");
        assert!(!call.contains("i32 92"), "{call}");
    }

    #[test]
    fn opaque_table_loaded_from_authored_argument_buffer_field_is_specialized() {
        let entry = r#"
%struct.Args = type { ptr addrspace(1), i64 }
define void @main(ptr addrspace(1) %args, ptr addrspace(1) %as) {
entry:
  %slot = getelementptr inbounds %struct.Args, ptr addrspace(1) %args, i64 0, i32 0
  %table = load ptr addrspace(1), ptr addrspace(1) %slot, align 8
  %hit = call { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } @air.intersect.intersection_function_buffer.triangle_data(<3 x float> zeroinitializer, <3 x float> zeroinitializer, float 0.0, float 1.0, ptr addrspace(1) %as, ptr addrspace(1) %table, i64 1, i64 8, ptr null, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 -1, i32 0, i1 false, i1 false)
  ret void
}
"#;
        let linkage = LinkedFunctionLinkage {
            visible_references: vec![],
            visible_tables: vec![],
            intersection_tables: vec![IntersectionFunctionTable {
                source: IntersectionFunctionTableSource::ArgumentBuffer {
                    buffer_parameter_index: 0,
                    field_ordinal: 0,
                    field_offset: 0,
                },
                size: 1,
                entries: vec![IntersectionFunctionEntry::OpaqueTriangle {
                    index: 0,
                    signature: vec![
                        IntersectionFunctionSignature::TriangleData,
                        IntersectionFunctionSignature::IntersectionFunctionBuffer,
                    ],
                }],
            }],
        };
        let specialized =
            specialize_opaque_triangle_intersection_tables(entry, "main", &linkage).unwrap();
        assert!(specialized.contains("@air.intersect.triangle_data("));
        assert!(!specialized.contains("@air.intersect.intersection_function_buffer"));
    }

    #[test]
    fn constant_slot_becomes_a_direct_linked_call() {
        let linked = LinkedFunctionLinkage {
            visible_references: vec![],
            visible_tables: vec![LinkedFunctionTable {
                parameter_index: 1,
                size: 1,
                entries: vec![LinkedFunction {
                    index: 0,
                    symbol: "add_one".into(),
                    module_ll: "define i32 @add_one(i32 %x) #0 {\nentry:\n  %y = add i32 %x, 1, !range !0\n  ret i32 %y\n}\nattributes #0 = { nounwind }\n!0 = !{i32 0, i32 2}\n".into(),
                }],
            }],
            intersection_tables: vec![],
        };
        let specialized = specialize_visible_function_tables(ENTRY, "main", &linked).unwrap();
        assert!(specialized.contains("%value = call i32 @add_one(i32 41)"));
        assert!(specialized.contains("define i32 @add_one"));
        assert!(!specialized.contains("attributes #0 ="));
        assert!(!specialized.contains("!0 = !{i32 0, i32 2}"));
    }

    #[test]
    fn dynamic_slot_gets_an_authored_switch_dispatcher() {
        let entry = ENTRY
            .replace(
                "ptr addrspace(1) %table)",
                "ptr addrspace(1) %table, i32 %slot)",
            )
            .replace("i32 0)", "i32 %slot)");
        let linked = LinkedFunctionLinkage {
            visible_references: vec![],
            visible_tables: vec![LinkedFunctionTable {
                parameter_index: 1,
                size: 1,
                entries: vec![LinkedFunction {
                    index: 0,
                    symbol: "add_one".into(),
                    module_ll: "define i32 @add_one(i32 %x) { ret i32 %x }".into(),
                }],
            }],
            intersection_tables: vec![],
        };
        let specialized = specialize_visible_function_tables(&entry, "main", &linked).unwrap();
        assert!(specialized
            .contains("call i32 @metal2vulkan.linked.table.p1.dispatch.0(i32 %slot, i32 41)"));
        assert!(specialized.contains("switch i32 %metal2vulkan_slot"));
        assert!(specialized.contains("call i32 @add_one(i32 %arg0)"));
    }

    #[test]
    fn dynamic_dispatcher_contains_only_type_compatible_table_entries() {
        let entry = ENTRY
            .replace(
                "ptr addrspace(1) %table)",
                "ptr addrspace(1) %table, i32 %slot)",
            )
            .replace("i32 0)", "i32 %slot)");
        let linked = LinkedFunctionLinkage {
            visible_references: vec![],
            visible_tables: vec![LinkedFunctionTable {
                parameter_index: 1,
                size: 2,
                entries: vec![
                    LinkedFunction {
                        index: 0,
                        symbol: "integer_function".into(),
                        module_ll: "define i32 @integer_function(i32 %x) { ret i32 %x }".into(),
                    },
                    LinkedFunction {
                        index: 1,
                        symbol: "float_function".into(),
                        module_ll: "define float @float_function(float %x) { ret float %x }".into(),
                    },
                ],
            }],
            intersection_tables: vec![],
        };
        let specialized = specialize_visible_function_tables(&entry, "main", &linked).unwrap();
        assert!(specialized.contains("i32 0, label %case_0"));
        assert!(!specialized.contains("label %case_1"));
        assert!(!specialized.contains("define float @float_function"));
    }

    #[test]
    fn authored_table_size_and_nullness_are_not_placeholders() {
        let entry = ENTRY.replace(
            "%fp = call ptr",
            "%size = call i32 @air.get_size_visible_function_table(ptr addrspace(1) %table)\n  %is_null = call i1 @air.is_null_visible_function_table(ptr addrspace(1) %table)\n  %fp = call ptr",
        );
        let linked = LinkedFunctionLinkage {
            visible_references: vec![],
            visible_tables: vec![LinkedFunctionTable {
                parameter_index: 1,
                size: 4,
                entries: vec![LinkedFunction {
                    index: 3,
                    symbol: "add_one".into(),
                    module_ll: "define i32 @add_one(i32 %x) { ret i32 %x }".into(),
                }],
            }],
            intersection_tables: vec![],
        };
        let entry = entry.replace("i32 0)", "i32 3)");
        let specialized = specialize_visible_function_tables(&entry, "main", &linked).unwrap();
        assert!(specialized.contains("%size = add i32 0, 4"));
        assert!(specialized.contains("%is_null = or i1 false, false"));
    }

    #[test]
    fn all_null_table_preserves_authored_capacity_and_null_slots() {
        let entry = r#"
define void @main(ptr addrspace(1) %table, i32 %slot) {
entry:
  %size = call i32 @air.get_size_visible_function_table(ptr addrspace(1) %table)
  %fp = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 %slot)
  %is_null = icmp eq ptr %fp, null
  ret void
}
"#;
        let linked = LinkedFunctionLinkage {
            visible_references: vec![],
            visible_tables: vec![LinkedFunctionTable {
                parameter_index: 0,
                size: 6,
                entries: vec![],
            }],
            intersection_tables: vec![],
        };
        let specialized = specialize_visible_function_tables(entry, "main", &linked).unwrap();
        assert!(specialized.contains("%size = add i32 0, 6"));
        assert!(specialized.contains("%is_null = xor i1 false, true"));
    }

    #[test]
    fn dynamic_lookup_nullness_is_authored_slot_membership() {
        let entry = ENTRY
            .replace(
                "ptr addrspace(1) %table)",
                "ptr addrspace(1) %table, i32 %slot)",
            )
            .replace("i32 0)", "i32 %slot)")
            .replace(
                "%cast = bitcast ptr %fp to ptr",
                "%is_null = icmp eq ptr %fp, null\n  %cast = bitcast ptr %fp to ptr",
            );
        let linked = LinkedFunctionLinkage {
            visible_references: vec![],
            visible_tables: vec![LinkedFunctionTable {
                parameter_index: 1,
                size: 8,
                entries: vec![
                    LinkedFunction {
                        index: 2,
                        symbol: "add_one".into(),
                        module_ll: "define i32 @add_one(i32 %x) { ret i32 %x }".into(),
                    },
                    LinkedFunction {
                        index: 7,
                        symbol: "add_two".into(),
                        module_ll: "define i32 @add_two(i32 %x) { ret i32 %x }".into(),
                    },
                ],
            }],
            intersection_tables: vec![],
        };
        let specialized = specialize_visible_function_tables(&entry, "main", &linked).unwrap();
        assert!(specialized.contains("icmp eq i32 %slot, 2"));
        assert!(specialized.contains("icmp eq i32 %slot, 7"));
        assert!(specialized.contains("%is_null = xor i1 %metal2vulkan.table.present."));
        assert!(!specialized.contains("%is_null = icmp eq ptr %fp, null"));
    }

    #[test]
    fn visible_function_pointer_opaque_sentinel_probe_is_folded() {
        let entry = ENTRY
            .replace(
                "ptr addrspace(1) %table)",
                "ptr addrspace(1) %table, i32 %slot)",
            )
            .replace("i32 0)", "i32 %slot)")
            .replace(
                "%cast = bitcast ptr %fp to ptr",
                "%wide = ptrtoint ptr %fp to i64\n  %low = trunc i64 %wide to i32\n  %is_opaque = icmp eq i32 %low, 1\n  %cast = bitcast ptr %fp to ptr",
            );
        let linked = LinkedFunctionLinkage {
            visible_references: vec![],
            visible_tables: vec![LinkedFunctionTable {
                parameter_index: 1,
                size: 4,
                entries: vec![LinkedFunction {
                    index: 3,
                    symbol: "add_one".into(),
                    module_ll: "define i32 @add_one(i32 %x) { ret i32 %x }".into(),
                }],
            }],
            intersection_tables: vec![],
        };
        let specialized = specialize_visible_function_tables(&entry, "main", &linked).unwrap();
        assert!(specialized.contains("%is_opaque = or i1 false, false"));
        assert!(!specialized.contains("%is_opaque = icmp eq i32 %low, 1"));
        assert!(specialized.contains("@metal2vulkan.linked.table.p1.dispatch.0"));
    }

    #[test]
    fn dynamic_slot_is_threaded_through_an_internal_helper_parameter() {
        let entry = r#"
define void @main(ptr addrspace(1) %output, ptr addrspace(1) %table, i32 %slot) {
entry:
  %fp = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 %slot)
  %typed = bitcast ptr %fp to ptr
  %value = call i32 @invoke(ptr %typed, i32 41)
  store i32 %value, ptr addrspace(1) %output, align 4
  ret void
}

define internal i32 @invoke(ptr %callback, i32 %value) {
entry:
  %result = call i32 %callback(i32 %value)
  ret i32 %result
}
"#;
        let linked = LinkedFunctionLinkage {
            visible_references: vec![],
            visible_tables: vec![LinkedFunctionTable {
                parameter_index: 1,
                size: 4,
                entries: vec![LinkedFunction {
                    index: 3,
                    symbol: "add_one".into(),
                    module_ll: "define i32 @add_one(i32 %x) { ret i32 %x }".into(),
                }],
            }],
            intersection_tables: vec![],
        };
        let specialized = specialize_visible_function_tables(entry, "main", &linked).unwrap();
        assert!(specialized.contains("call i32 @invoke(ptr %typed, i32 41, i32 %slot)"));
        assert!(specialized.contains(
            "define internal i32 @invoke(ptr %callback, i32 %value, i32 %metal2vulkan.table.param0.slot)"
        ));
        assert!(specialized.contains(
            "@metal2vulkan.linked.table.p1.dispatch.0(i32 %metal2vulkan.table.param0.slot, i32 %value)"
        ));
        assert!(!specialized.contains("call i32 %callback("));
    }

    #[test]
    fn linked_translation_emits_the_direct_function_and_no_table_descriptor() {
        let entry = format!(
            r#"{ENTRY}
declare ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1), i32)
!air.kernel = !{{!0}}
!0 = !{{ptr @main, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"uint"}}
!4 = !{{i32 1, !"air.visible_function_table", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"visible_function_table"}}
"#
        );
        let linked = LinkedFunctionLinkage {
            visible_references: vec![],
            visible_tables: vec![LinkedFunctionTable {
                parameter_index: 1,
                size: 1,
                entries: vec![LinkedFunction {
                    index: 0,
                    symbol: "add_one".into(),
                    module_ll: "define i32 @add_one(i32 %x) {\nentry:\n  %y = add i32 %x, 1\n  ret i32 %y\n}\n".into(),
                }],
            }],
            intersection_tables: vec![],
        };
        struct Scratch(std::path::PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let scratch = Scratch(std::env::temp_dir().join(format!(
            "metal2vulkan-linked-function-{}",
            std::process::id()
        )));
        let _ = std::fs::remove_dir_all(&scratch.0);
        std::fs::create_dir(&scratch.0).unwrap();
        let spv = crate::translate_sanitized_native_linked_with_options(
            &entry,
            crate::passes::Stage::Kernel,
            &scratch.0,
            crate::passes::TransformOptions::default(),
            &linked,
        )
        .unwrap();
        let asm = crate::disassemble(&spv).unwrap();
        assert!(asm.contains("OpIAdd"), "{asm}");
        assert!(!asm.contains("visible_function_table"), "{asm}");
        assert!(!asm.contains("Binding 1"), "{asm}");
    }

    #[test]
    fn direct_reference_linked_translation_emits_the_authored_definition() {
        let entry = r#"
define void @main(ptr addrspace(1) %output) {
entry:
  %value = call i32 @add_one.MTL_VISIBLE_FN_REF(i32 41)
  store i32 %value, ptr addrspace(1) %output, align 4
  ret void
}
declare i32 @add_one.MTL_VISIBLE_FN_REF(i32) section "air.externally_defined"
!air.kernel = !{!0}
!air.visible_function_references = !{!4}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"uint"}
!4 = !{!"air.visible_function_reference", ptr @add_one.MTL_VISIBLE_FN_REF, !"add_one"}
"#;
        let linked = LinkedFunctionLinkage {
            visible_references: vec![LinkedFunctionReference {
                symbol: "add_one".into(),
                module_ll:
                    "define i32 @add_one(i32 %x) {\nentry:\n  %y = add i32 %x, 1\n  ret i32 %y\n}\n"
                        .into(),
            }],
            visible_tables: vec![],
            intersection_tables: vec![],
        };
        struct Scratch(std::path::PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let scratch = Scratch(std::env::temp_dir().join(format!(
            "metal2vulkan-direct-linked-function-{}",
            std::process::id()
        )));
        let _ = std::fs::remove_dir_all(&scratch.0);
        std::fs::create_dir(&scratch.0).unwrap();
        let spv = crate::translate_sanitized_native_linked_with_options(
            entry,
            crate::passes::Stage::Kernel,
            &scratch.0,
            crate::passes::TransformOptions::default(),
            &linked,
        )
        .unwrap();
        let asm = crate::disassemble(&spv).unwrap();
        assert!(asm.contains("OpIAdd"), "{asm}");
    }

    #[test]
    fn dynamic_linked_translation_emits_authored_slot_dispatch() {
        let entry = r#"
define void @main(ptr addrspace(1) %output, ptr addrspace(1) %table, ptr addrspace(2) %slot_buffer) {
entry:
  %slot = load i32, ptr addrspace(2) %slot_buffer, align 4
  %fp = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 %slot)
  %value = call i32 %fp(i32 40)
  store i32 %value, ptr addrspace(1) %output, align 4
  ret void
}
declare ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1), i32)
!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"uint"}
!4 = !{i32 1, !"air.visible_function_table", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"visible_function_table"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 2, !"air.arg_type_name", !"uint"}
"#;
        let linked = LinkedFunctionLinkage {
            visible_references: vec![],
            visible_tables: vec![LinkedFunctionTable {
                parameter_index: 1,
                size: 2,
                entries: vec![
                    LinkedFunction {
                        index: 0,
                        symbol: "add_one".into(),
                        module_ll: "define i32 @add_one(i32 %x) {\nentry:\n  %y = add i32 %x, 1\n  ret i32 %y\n}\n".into(),
                    },
                    LinkedFunction {
                        index: 1,
                        symbol: "add_two".into(),
                        module_ll: "define i32 @add_two(i32 %x) {\nentry:\n  %y = add i32 %x, 2\n  ret i32 %y\n}\n".into(),
                    },
                ],
            }],
            intersection_tables: vec![],
        };
        struct Scratch(std::path::PathBuf);
        impl Drop for Scratch {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let scratch = Scratch(std::env::temp_dir().join(format!(
            "metal2vulkan-linked-function-dynamic-{}",
            std::process::id()
        )));
        let _ = std::fs::remove_dir_all(&scratch.0);
        std::fs::create_dir(&scratch.0).unwrap();
        let spv = crate::translate_sanitized_native_linked_with_options(
            entry,
            crate::passes::Stage::Kernel,
            &scratch.0,
            crate::passes::TransformOptions::default(),
            &linked,
        )
        .unwrap();
        let asm = crate::disassemble(&spv).unwrap();
        assert!(asm.contains("OpSwitch"), "{asm}");
        assert!(!asm.contains("Binding 1"), "{asm}");
    }
}
