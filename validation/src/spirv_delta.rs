//! Mechanical SPIR-V delta classification for typed-pass migration referees.
//!
//! The classifier is intentionally conservative. `DC1` accepts only logical identity after the
//! translator's shipped id canonicalization and SPIRV-Tools' declaration-aware mapping. `DC2`
//! accepts a deletion-only logical diff whose removed result-producing instructions are a closed,
//! side-effect-free scaffolding class and have no references from the surviving module.

use rspirv::binary::Assemble;
use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{MemoryAccess, Op, StorageClass, Word};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// The migration referee verdict for one pre/post SPIR-V pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpirvDelta {
    /// The input byte streams are identical.
    Dc0,
    /// The modules differ only in ids or declaration order.
    Dc1,
    /// The post module only removes provably dead, side-effect-free scaffolding.
    Dc2,
    /// The first line or structural fact that makes the delta semantics-visible or unproved.
    Other { first_offending_line: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InstructionSite {
    Global,
    Function,
}

#[derive(Clone, Debug)]
struct Definition {
    instruction: Instruction,
    site: InstructionSite,
}

#[derive(Debug)]
struct DiffFiles {
    before: PathBuf,
    after: PathBuf,
}

impl DiffFiles {
    fn create(before: &[u8], after: &[u8]) -> Result<Self, String> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nonce = NEXT.fetch_add(1, Ordering::Relaxed);
        let stem = format!("metal2vulkan-spirv-delta-{}-{nonce}", std::process::id());
        let before_path = std::env::temp_dir().join(format!("{stem}-before.spv"));
        let after_path = std::env::temp_dir().join(format!("{stem}-after.spv"));
        fs::write(&before_path, before)
            .map_err(|error| format!("write {}: {error}", before_path.display()))?;
        if let Err(error) = fs::write(&after_path, after) {
            let _ = fs::remove_file(&before_path);
            return Err(format!("write {}: {error}", after_path.display()));
        }
        Ok(Self {
            before: before_path,
            after: after_path,
        })
    }
}

impl Drop for DiffFiles {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.before);
        let _ = fs::remove_file(&self.after);
    }
}

/// Classify one pre/post SPIR-V pair as `DC0`, `DC1`, `DC2`, or an unapproved delta.
///
/// This calls the locally installed `spirv-diff` through metal2vulkan's bounded tool runner.
/// `METAL2VULKAN_SPIRV_DIFF` can override the binary path through the existing dynamic tool-path
/// family. Invalid SPIR-V or an unavailable referee tool is an infrastructure error, not `Other`.
pub fn classify_spirv_delta(before: &[u8], after: &[u8]) -> Result<SpirvDelta, String> {
    let canonical_before = metal2vulkan::canonicalize_spirv_bytes(before)?;
    let canonical_after = metal2vulkan::canonicalize_spirv_bytes(after)?;
    let before_module = load_module(&canonical_before, "before")?;
    let after_module = load_module(&canonical_after, "after")?;

    if let Some(reason) = incompatible_header(&before_module, &after_module) {
        return Ok(other(reason));
    }
    if before == after {
        return Ok(SpirvDelta::Dc0);
    }

    let before_opcodes = opcode_multiset(&before_module);
    let after_opcodes = opcode_multiset(&after_module);
    if canonical_before == canonical_after {
        return Ok(if before_opcodes == after_opcodes {
            SpirvDelta::Dc1
        } else {
            other("canonical bytes match but opcode multisets differ")
        });
    }

    if scaffolding_normalizes_to_identity(&before_module, &after_module)? {
        return Ok(SpirvDelta::Dc2);
    }

    if let Some((opcode, after_count)) = after_opcodes.iter().find(|(opcode, after_count)| {
        **after_count > before_opcodes.get(opcode).copied().unwrap_or(0)
    }) {
        let before_count = before_opcodes.get(opcode).copied().unwrap_or(0);
        return Ok(other(format!(
            "post opcode {opcode} count increases from {before_count} to {after_count}"
        )));
    }

    let diff_files = DiffFiles::create(&canonical_before, &canonical_after)?;
    let before_arg = path_arg(&diff_files.before)?;
    let after_arg = path_arg(&diff_files.after)?;
    let (stdout, _) = metal2vulkan::tools::run(
        "spirv-diff",
        &["--no-color", "--no-header", before_arg, after_arg],
    )?;
    let diff = String::from_utf8(stdout)
        .map_err(|error| format!("spirv-diff emitted non-UTF-8 output: {error}"))?;
    let changed = changed_lines(&diff);

    if changed.is_empty() {
        return Ok(if before_opcodes == after_opcodes {
            SpirvDelta::Dc1
        } else {
            other("spirv-diff mapped the modules but opcode multisets differ")
        });
    }

    if before_opcodes == after_opcodes
        && is_global_declaration_order_delta(&before_module, &diff, &changed)
    {
        return Ok(SpirvDelta::Dc1);
    }

    if let Some(line) = changed.iter().find(|line| line.starts_with('+')) {
        return Ok(other((*line).to_string()));
    }

    classify_removals(&before_module, &diff, &changed)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ScaffoldingStats {
    private_variables: usize,
    loads: usize,
    copies: usize,
}

fn scaffolding_normalizes_to_identity(before: &Module, after: &Module) -> Result<bool, String> {
    let mut normalized_before = before.clone();
    let mut normalized_after = after.clone();
    let before_stats = normalize_scaffolding(&mut normalized_before);
    let after_stats = normalize_scaffolding(&mut normalized_after);
    if before_stats == after_stats {
        return Ok(false);
    }

    let before_bytes = canonical_assembled_bytes(&normalized_before)?;
    let after_bytes = canonical_assembled_bytes(&normalized_after)?;
    if before_bytes == after_bytes {
        return Ok(true);
    }

    let before_module = load_module(&before_bytes, "normalized before")?;
    let after_module = load_module(&after_bytes, "normalized after")?;
    if opcode_multiset(&before_module) != opcode_multiset(&after_module) {
        return Ok(false);
    }
    let files = DiffFiles::create(&before_bytes, &after_bytes)?;
    let before_arg = path_arg(&files.before)?;
    let after_arg = path_arg(&files.after)?;
    let Ok((stdout, _)) = metal2vulkan::tools::run(
        "spirv-diff",
        &["--no-color", "--no-header", before_arg, after_arg],
    ) else {
        return Ok(false);
    };
    let diff = String::from_utf8(stdout)
        .map_err(|error| format!("normalized spirv-diff emitted non-UTF-8 output: {error}"))?;
    let changed = changed_lines(&diff);
    Ok(changed.is_empty() || is_global_declaration_order_delta(&before_module, &diff, &changed))
}

fn canonical_assembled_bytes(module: &Module) -> Result<Vec<u8>, String> {
    let bytes = module
        .assemble()
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect::<Vec<_>>();
    metal2vulkan::canonicalize_spirv_bytes(&bytes)
}

fn normalize_scaffolding(module: &mut Module) -> ScaffoldingStats {
    let definitions = definitions(module);
    let mut private_initializers = HashMap::<Word, Word>::new();
    for instruction in &module.types_global_values {
        let (Some(variable), Some(Operand::StorageClass(StorageClass::Private))) =
            (instruction.result_id, instruction.operands.first())
        else {
            continue;
        };
        let Some(Operand::IdRef(initializer)) = instruction.operands.get(1) else {
            continue;
        };
        let Some(initializer_definition) = definitions.get(initializer) else {
            continue;
        };
        let opname = initializer_definition.instruction.class.opname;
        if initializer_definition.site == InstructionSite::Global
            && (opname.starts_with("Constant")
                || opname.starts_with("SpecConstant")
                || initializer_definition.instruction.class.opcode == Op::Undef)
        {
            private_initializers.insert(variable, *initializer);
        }
    }

    let mut alias_root = private_initializers
        .keys()
        .copied()
        .map(|variable| (variable, variable))
        .collect::<HashMap<_, _>>();
    loop {
        let mut changed = false;
        for function in &module.functions {
            for instruction in function.all_inst_iter() {
                if instruction.class.opcode != Op::CopyObject {
                    continue;
                }
                let (Some(result), Some(Operand::IdRef(source))) =
                    (instruction.result_id, instruction.operands.first())
                else {
                    continue;
                };
                if let Some(root) = alias_root.get(source).copied() {
                    changed |= alias_root.insert(result, root) != Some(root);
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut unsafe_roots = HashSet::new();
    for entry_point in &module.entry_points {
        for (index, operand) in entry_point.operands.iter().enumerate() {
            let Operand::IdRef(id) = operand else {
                continue;
            };
            let Some(root) = alias_root.get(id).copied() else {
                continue;
            };
            if index < 3 || *id != root {
                unsafe_roots.insert(root);
            }
        }
    }
    for function in &module.functions {
        for instruction in function.all_inst_iter() {
            for (index, operand) in instruction.operands.iter().enumerate() {
                let Operand::IdRef(id) = operand else {
                    continue;
                };
                let Some(root) = alias_root.get(id).copied() else {
                    continue;
                };
                let allowed = match instruction.class.opcode {
                    Op::CopyObject => index == 0,
                    Op::Load => {
                        let initializer = private_initializers[&root];
                        index == 0
                            && instruction.result_type
                                == definitions
                                    .get(&initializer)
                                    .and_then(|definition| definition.instruction.result_type)
                            && !instruction.operands.iter().any(|operand| {
                                matches!(
                                    operand,
                                    Operand::MemoryAccess(access)
                                        if access.contains(MemoryAccess::VOLATILE)
                                )
                            })
                    }
                    _ => false,
                };
                if !allowed {
                    unsafe_roots.insert(root);
                }
            }
        }
    }

    let safe_variables = private_initializers
        .keys()
        .copied()
        .filter(|variable| !unsafe_roots.contains(variable))
        .collect::<HashSet<_>>();
    let safe_aliases = alias_root
        .into_iter()
        .filter(|(_, root)| safe_variables.contains(root))
        .collect::<HashMap<_, _>>();
    let mut stats = ScaffoldingStats {
        private_variables: safe_variables.len(),
        ..ScaffoldingStats::default()
    };

    for entry_point in &mut module.entry_points {
        entry_point.operands.retain(
            |operand| !matches!(operand, Operand::IdRef(id) if safe_variables.contains(id)),
        );
    }
    module.types_global_values.retain(|instruction| {
        !instruction
            .result_id
            .is_some_and(|id| safe_variables.contains(&id))
    });

    let mut substitutions = HashMap::<Word, Word>::new();
    for function in &mut module.functions {
        for block in &mut function.blocks {
            let mut retained = Vec::with_capacity(block.instructions.len());
            for mut instruction in std::mem::take(&mut block.instructions) {
                rewrite_substituted_operands(&mut instruction, &substitutions);
                if instruction.class.opcode == Op::CopyObject {
                    let (Some(result), Some(Operand::IdRef(source))) =
                        (instruction.result_id, instruction.operands.first())
                    else {
                        retained.push(instruction);
                        continue;
                    };
                    substitutions.insert(result, resolve_substitution(*source, &substitutions));
                    stats.copies += 1;
                    continue;
                }
                if instruction.class.opcode == Op::Load {
                    let Some(Operand::IdRef(pointer)) = instruction.operands.first() else {
                        retained.push(instruction);
                        continue;
                    };
                    let Some(root) = safe_aliases.get(pointer).copied() else {
                        retained.push(instruction);
                        continue;
                    };
                    let Some(result) = instruction.result_id else {
                        retained.push(instruction);
                        continue;
                    };
                    substitutions.insert(result, private_initializers[&root]);
                    stats.loads += 1;
                    continue;
                }
                retained.push(instruction);
            }
            block.instructions = retained;
        }
    }
    for function in &mut module.functions {
        for instruction in function.all_inst_iter_mut() {
            rewrite_substituted_operands(instruction, &substitutions);
        }
    }

    // Names carry no semantics and may target removed scaffolding. Decorations remain and are
    // retained only when their target survives the global reachability closure below.
    module.debug_names.clear();
    prune_unreachable_globals(module);
    stats
}

fn resolve_substitution(id: Word, substitutions: &HashMap<Word, Word>) -> Word {
    let mut resolved = id;
    let mut seen = HashSet::new();
    while seen.insert(resolved) {
        let Some(next) = substitutions.get(&resolved).copied() else {
            break;
        };
        resolved = next;
    }
    resolved
}

fn rewrite_substituted_operands(
    instruction: &mut Instruction,
    substitutions: &HashMap<Word, Word>,
) {
    for operand in &mut instruction.operands {
        if let Operand::IdRef(id) = operand {
            *id = resolve_substitution(*id, substitutions);
        }
    }
}

fn prune_unreachable_globals(module: &mut Module) {
    let global_definitions = module
        .types_global_values
        .iter()
        .filter_map(|instruction| instruction.result_id.map(|id| (id, instruction.clone())))
        .collect::<HashMap<_, _>>();
    let mut live = HashSet::<Word>::new();
    let mut queue = VecDeque::<Word>::new();
    let mut add_instruction_refs = |instruction: &Instruction| {
        if let Some(result_type) = instruction.result_type {
            if live.insert(result_type) {
                queue.push_back(result_type);
            }
        }
        for operand in &instruction.operands {
            if let Operand::IdRef(id) = operand {
                if live.insert(*id) {
                    queue.push_back(*id);
                }
            }
        }
    };
    for instruction in &module.entry_points {
        add_instruction_refs(instruction);
    }
    for instruction in &module.execution_modes {
        add_instruction_refs(instruction);
    }
    for function in &module.functions {
        for instruction in function.all_inst_iter() {
            add_instruction_refs(instruction);
        }
    }
    close_global_references(&global_definitions, &mut live, &mut queue);

    module.annotations.retain(|instruction| {
        let target = instruction
            .operands
            .iter()
            .find_map(|operand| match operand {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            });
        target.is_none_or(|id| !global_definitions.contains_key(&id) || live.contains(&id))
    });
    for instruction in &module.annotations {
        for operand in &instruction.operands {
            if let Operand::IdRef(id) = operand {
                if live.insert(*id) {
                    queue.push_back(*id);
                }
            }
        }
    }
    close_global_references(&global_definitions, &mut live, &mut queue);
    module
        .types_global_values
        .retain(|instruction| instruction.result_id.is_none_or(|id| live.contains(&id)));
}

fn close_global_references(
    definitions: &HashMap<Word, Instruction>,
    live: &mut HashSet<Word>,
    queue: &mut VecDeque<Word>,
) {
    while let Some(id) = queue.pop_front() {
        let Some(instruction) = definitions.get(&id) else {
            continue;
        };
        if let Some(result_type) = instruction.result_type {
            if live.insert(result_type) {
                queue.push_back(result_type);
            }
        }
        for operand in &instruction.operands {
            if let Operand::IdRef(dependency) = operand {
                if live.insert(*dependency) {
                    queue.push_back(*dependency);
                }
            }
        }
    }
}

fn load_module(bytes: &[u8], side: &str) -> Result<Module, String> {
    rspirv::dr::load_bytes(bytes).map_err(|error| format!("{side} rspirv load: {error:?}"))
}

fn path_arg(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| format!("non-UTF-8 temporary path: {}", path.display()))
}

fn other(first_offending_line: impl Into<String>) -> SpirvDelta {
    SpirvDelta::Other {
        first_offending_line: first_offending_line.into(),
    }
}

fn incompatible_header(before: &Module, after: &Module) -> Option<String> {
    let before = before.header.as_ref()?;
    let after = after.header.as_ref()?;
    [
        ("magic", before.magic_number, after.magic_number),
        ("version", before.version, after.version),
        ("generator", before.generator, after.generator),
        ("reserved", before.reserved_word, after.reserved_word),
    ]
    .into_iter()
    .find_map(|(field, lhs, rhs)| {
        (lhs != rhs).then(|| format!("SPIR-V header {field} differs: {lhs:#x} != {rhs:#x}"))
    })
}

fn opcode_multiset(module: &Module) -> BTreeMap<u32, usize> {
    let mut opcodes = BTreeMap::new();
    for instruction in module.all_inst_iter() {
        *opcodes
            .entry(u32::from(instruction.class.opcode))
            .or_insert(0) += 1;
    }
    opcodes
}

fn changed_lines(diff: &str) -> Vec<&str> {
    diff.lines()
        .filter(|line| line.starts_with('+') || line.starts_with('-'))
        .collect()
}

fn classify_removals(
    before: &Module,
    diff: &str,
    removed_lines: &[&str],
) -> Result<SpirvDelta, String> {
    let definitions = definitions(before);
    let mut removed_ids = HashSet::new();

    for line in removed_lines {
        let Some((id, opname)) = removed_definition(line) else {
            return Ok(other((*line).to_string()));
        };
        let Some(definition) = definitions.get(&id) else {
            return Ok(other(format!(
                "{line} (removed result %{id} is absent from the canonical before module)"
            )));
        };
        let actual_opname = format!("Op{}", definition.instruction.class.opname);
        if actual_opname != opname {
            return Ok(other(format!(
                "{line} (diff opcode {opname} does not match {actual_opname})"
            )));
        }
        if !is_dead_scaffolding(definition) {
            return Ok(other((*line).to_string()));
        }
        removed_ids.insert(id);
    }

    for line in diff.lines().filter(|line| !line.starts_with('-')) {
        if let Some(id) = id_tokens(line)
            .into_iter()
            .find(|id| removed_ids.contains(id))
        {
            return Ok(other(format!(
                "{line} (surviving instruction still uses removed result %{id})"
            )));
        }
    }

    Ok(SpirvDelta::Dc2)
}

fn definitions(module: &Module) -> HashMap<Word, Definition> {
    let mut definitions = HashMap::new();
    for instruction in module.global_inst_iter() {
        if let Some(id) = instruction.result_id {
            definitions.insert(
                id,
                Definition {
                    instruction: instruction.clone(),
                    site: InstructionSite::Global,
                },
            );
        }
    }
    for function in &module.functions {
        for instruction in function.all_inst_iter() {
            if let Some(id) = instruction.result_id {
                definitions.insert(
                    id,
                    Definition {
                        instruction: instruction.clone(),
                        site: InstructionSite::Function,
                    },
                );
            }
        }
    }
    definitions
}

fn removed_definition(line: &str) -> Option<(Word, String)> {
    changed_definition(line, '-').map(|definition| (definition.id, definition.opname))
}

#[derive(Clone, Debug)]
struct ChangedDefinition {
    id: Word,
    opname: String,
    operand_shape: String,
}

fn changed_definition(line: &str, prefix: char) -> Option<ChangedDefinition> {
    let body = line.strip_prefix(prefix)?.trim_start();
    let (result, instruction) = body.split_once(" = ")?;
    let id = result.trim().strip_prefix('%')?.parse().ok()?;
    let opname = instruction.split_whitespace().next()?.to_string();
    Some(ChangedDefinition {
        id,
        opname,
        // Keep referenced ids in the seed shape. spirv-diff preserves ids for declarations it
        // maps, and retaining them prevents same-opcode declarations of different types (notably
        // OpUndef) from being paired arbitrarily.
        operand_shape: instruction.to_string(),
    })
}

fn is_global_declaration_order_delta(before: &Module, diff: &str, changed: &[&str]) -> bool {
    let before_definitions = definitions(before);
    let mut removed = Vec::<ChangedDefinition>::new();
    let mut added = Vec::<ChangedDefinition>::new();

    for line in changed {
        let (prefix, declarations) = if line.starts_with('-') {
            ('-', &mut removed)
        } else if line.starts_with('+') {
            ('+', &mut added)
        } else {
            continue;
        };
        let Some(changed_definition) = changed_definition(line, prefix) else {
            continue;
        };
        if prefix == '-' {
            let Some(definition) = before_definitions.get(&changed_definition.id) else {
                return false;
            };
            if !is_reorderable_global_declaration(definition)
                || format!("Op{}", definition.instruction.class.opname) != changed_definition.opname
            {
                continue;
            }
        } else if !is_reorderable_global_opname(&changed_definition.opname) {
            continue;
        }
        declarations.push(changed_definition);
    }

    if removed.is_empty() || added.is_empty() {
        return false;
    }

    let mut id_mapping = HashMap::new();
    let mut matched_removed = HashSet::new();
    let mut matched_added = HashSet::new();
    loop {
        let mut removed_by_shape = BTreeMap::<(String, String), Vec<usize>>::new();
        let mut added_by_shape = BTreeMap::<(String, String), Vec<usize>>::new();
        for (index, declaration) in removed.iter().enumerate() {
            if matched_removed.contains(&index) {
                continue;
            }
            let operand_shape = replace_id_tokens(&declaration.operand_shape, |id| {
                id_mapping
                    .get(&id)
                    .map_or_else(|| format!("%{id}"), |new| format!("%{new}"))
            });
            removed_by_shape
                .entry((declaration.opname.clone(), operand_shape))
                .or_default()
                .push(index);
        }
        for (index, declaration) in added.iter().enumerate() {
            if !matched_added.contains(&index) {
                added_by_shape
                    .entry((
                        declaration.opname.clone(),
                        declaration.operand_shape.clone(),
                    ))
                    .or_default()
                    .push(index);
            }
        }

        let mut paired = 0;
        for (shape, removed_indices) in &mut removed_by_shape {
            let Some(added_indices) = added_by_shape.get_mut(shape) else {
                continue;
            };
            removed_indices.sort_by_key(|index| removed[*index].id);
            added_indices.sort_by_key(|index| added[*index].id);
            for (removed_index, added_index) in removed_indices.iter().zip(added_indices.iter()) {
                matched_removed.insert(*removed_index);
                matched_added.insert(*added_index);
                id_mapping.insert(removed[*removed_index].id, added[*added_index].id);
                paired += 1;
            }
        }
        if paired == 0 {
            break;
        }
    }

    // A context line containing one of the remapped ids would mean spirv-diff did not expose the
    // complete substitution. Refuse to infer identity in that case.
    if diff.lines().any(|line| {
        !line.starts_with('+')
            && !line.starts_with('-')
            && id_tokens(line)
                .iter()
                .any(|id| id_mapping.contains_key(id) || id_mapping.values().any(|new| new == id))
    }) {
        return false;
    }

    let mut removed = changed
        .iter()
        .filter_map(|line| line.strip_prefix('-'))
        .map(str::trim_start)
        .map(|line| {
            replace_id_tokens(line, |id| {
                id_mapping
                    .get(&id)
                    .map_or_else(|| format!("%{id}"), |new| format!("%{new}"))
            })
        })
        .collect::<Vec<_>>();
    let mut added = changed
        .iter()
        .filter_map(|line| line.strip_prefix('+'))
        .map(str::trim_start)
        .map(str::to_string)
        .collect::<Vec<_>>();
    removed.sort_unstable();
    added.sort_unstable();
    removed == added
}

fn is_reorderable_global_declaration(definition: &Definition) -> bool {
    if definition.site != InstructionSite::Global {
        return false;
    }
    let opname = definition.instruction.class.opname;
    is_reorderable_global_opname(&format!("Op{opname}"))
}

fn is_reorderable_global_opname(opname: &str) -> bool {
    opname.starts_with("OpType")
        || opname.starts_with("OpConstant")
        || opname.starts_with("OpSpecConstant")
        || opname == "OpUndef"
        || opname == "OpVariable"
}

fn replace_id_tokens(line: &str, mut replacement: impl FnMut(Word) -> String) -> String {
    let bytes = line.as_bytes();
    let mut replaced = String::with_capacity(line.len());
    let mut copied_until = 0;
    let mut offset = 0;
    while offset < bytes.len() {
        if bytes[offset] != b'%' {
            offset += 1;
            continue;
        }
        let start = offset + 1;
        let mut end = start;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end == start {
            offset += 1;
            continue;
        }
        let Ok(id) = line[start..end].parse() else {
            offset = end;
            continue;
        };
        replaced.push_str(&line[copied_until..offset]);
        replaced.push_str(&replacement(id));
        copied_until = end;
        offset = end;
    }
    replaced.push_str(&line[copied_until..]);
    replaced
}

fn id_tokens(line: &str) -> Vec<Word> {
    let bytes = line.as_bytes();
    let mut ids = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        if bytes[offset] != b'%' {
            offset += 1;
            continue;
        }
        let start = offset + 1;
        let mut end = start;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end > start {
            if let Ok(id) = line[start..end].parse() {
                ids.push(id);
            }
        }
        offset = end.max(offset + 1);
    }
    ids
}

fn is_dead_scaffolding(definition: &Definition) -> bool {
    let instruction = &definition.instruction;
    match definition.site {
        InstructionSite::Global => {
            let opname = instruction.class.opname;
            opname.starts_with("Type")
                || opname.starts_with("Constant")
                || opname.starts_with("SpecConstant")
                || instruction.class.opcode == Op::Undef
                || is_unused_private_global(instruction)
        }
        InstructionSite::Function => is_pure_scaffolding_instruction(instruction),
    }
}

fn is_unused_private_global(instruction: &Instruction) -> bool {
    instruction.class.opcode == Op::Variable
        && matches!(
            instruction.operands.first(),
            Some(Operand::StorageClass(StorageClass::Private))
        )
}

fn is_pure_scaffolding_instruction(instruction: &Instruction) -> bool {
    match instruction.class.opcode {
        Op::CopyObject
        | Op::AccessChain
        | Op::InBoundsAccessChain
        | Op::PtrAccessChain
        | Op::InBoundsPtrAccessChain
        | Op::SampledImage
        | Op::Image
        | Op::ImageTexelPointer
        | Op::CompositeConstruct
        | Op::CompositeExtract
        | Op::CompositeInsert
        | Op::VectorExtractDynamic
        | Op::VectorInsertDynamic
        | Op::VectorShuffle => true,
        Op::Load => !instruction.operands.iter().any(|operand| {
            matches!(
                operand,
                Operand::MemoryAccess(access) if access.contains(MemoryAccess::VOLATILE)
            )
        }),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rspirv::binary::Assemble;
    use rspirv::dr::{Block, Function, ModuleHeader};
    use rspirv::spirv::{
        AddressingModel, Capability, ExecutionMode, ExecutionModel, FunctionControl, MemoryModel,
    };

    fn fixture(id_base: Word, dead_scaffolding: bool, changed_return: bool) -> Vec<u8> {
        let id = |offset: Word| id_base + offset;
        let void = id(1);
        let uint = id(2);
        let private_uint = id(3);
        let zero = id(4);
        let dead_private = id(5);
        let function_type = id(6);
        let function = id(7);
        let label = id(8);
        let dead_copy = id(9);

        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(id(10)));
        module.capabilities.push(Instruction::new(
            Op::Capability,
            None,
            None,
            vec![Operand::Capability(Capability::Shader)],
        ));
        module.memory_model = Some(Instruction::new(
            Op::MemoryModel,
            None,
            None,
            vec![
                Operand::AddressingModel(AddressingModel::Logical),
                Operand::MemoryModel(MemoryModel::GLSL450),
            ],
        ));
        module.entry_points.push(Instruction::new(
            Op::EntryPoint,
            None,
            None,
            vec![
                Operand::ExecutionModel(ExecutionModel::GLCompute),
                Operand::IdRef(function),
                Operand::LiteralString("main".into()),
            ],
        ));
        module.execution_modes.push(Instruction::new(
            Op::ExecutionMode,
            None,
            None,
            vec![
                Operand::IdRef(function),
                Operand::ExecutionMode(ExecutionMode::LocalSize),
                Operand::LiteralBit32(1),
                Operand::LiteralBit32(1),
                Operand::LiteralBit32(1),
            ],
        ));
        module.types_global_values = vec![
            Instruction::new(Op::TypeVoid, None, Some(void), vec![]),
            Instruction::new(
                Op::TypeInt,
                None,
                Some(uint),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            Instruction::new(
                Op::TypePointer,
                None,
                Some(private_uint),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(uint),
                ],
            ),
            Instruction::new(
                Op::Constant,
                Some(uint),
                Some(zero),
                vec![Operand::LiteralBit32(0)],
            ),
        ];
        if dead_scaffolding {
            module.types_global_values.push(Instruction::new(
                Op::Variable,
                Some(private_uint),
                Some(dead_private),
                vec![Operand::StorageClass(StorageClass::Private)],
            ));
        }
        module.types_global_values.push(Instruction::new(
            Op::TypeFunction,
            None,
            Some(function_type),
            vec![Operand::IdRef(void)],
        ));

        let mut instructions = Vec::new();
        if dead_scaffolding {
            instructions.push(Instruction::new(
                Op::CopyObject,
                Some(uint),
                Some(dead_copy),
                vec![Operand::IdRef(zero)],
            ));
        }
        instructions.push(Instruction::new(
            if changed_return {
                Op::Unreachable
            } else {
                Op::Return
            },
            None,
            None,
            vec![],
        ));
        module.functions.push(Function {
            def: Some(Instruction::new(
                Op::Function,
                Some(void),
                Some(function),
                vec![
                    Operand::FunctionControl(FunctionControl::NONE),
                    Operand::IdRef(function_type),
                ],
            )),
            end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
            parameters: vec![],
            blocks: vec![Block {
                label: Some(Instruction::new(Op::Label, None, Some(label), vec![])),
                instructions,
            }],
        });

        module
            .assemble()
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect()
    }

    fn reorder_independent_declarations(bytes: &[u8]) -> Vec<u8> {
        let mut module = rspirv::dr::load_bytes(bytes).unwrap();
        module.types_global_values.swap(3, 4);
        module
            .assemble()
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect()
    }

    fn reorder_dependent_undef(bytes: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let mut before = rspirv::dr::load_bytes(bytes).unwrap();
        let uint = 2;
        let vector = 10;
        let undef = 11;
        let copy = 12;
        before.header.as_mut().unwrap().bound = 13;
        before.types_global_values.push(Instruction::new(
            Op::TypeVector,
            None,
            Some(vector),
            vec![Operand::IdRef(uint), Operand::LiteralBit32(2)],
        ));
        before.types_global_values.push(Instruction::new(
            Op::Undef,
            Some(vector),
            Some(undef),
            vec![],
        ));
        before.functions[0].blocks[0].instructions.insert(
            0,
            Instruction::new(
                Op::CopyObject,
                Some(vector),
                Some(copy),
                vec![Operand::IdRef(undef)],
            ),
        );
        let mut after = before.clone();
        after.types_global_values.swap(5, 6);
        let assemble = |module: Module| {
            module
                .assemble()
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect()
        };
        (assemble(before), assemble(after))
    }

    fn reorder_private_globals(bytes: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let mut before = rspirv::dr::load_bytes(bytes).unwrap();
        let private_uint = 3;
        let first_variable = 5;
        let second_variable = 10;
        before.header.as_mut().unwrap().bound = 11;
        before.types_global_values.push(Instruction::new(
            Op::Variable,
            Some(private_uint),
            Some(second_variable),
            vec![Operand::StorageClass(StorageClass::Private)],
        ));
        before.entry_points[0].operands.extend([
            Operand::IdRef(first_variable),
            Operand::IdRef(second_variable),
        ]);
        let mut after = before.clone();
        after.types_global_values.swap(4, 6);
        let assemble = |module: Module| {
            module
                .assemble()
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect()
        };
        (assemble(before), assemble(after))
    }

    fn private_load_scaffold(scaffold: bool, store_through_pointer: bool) -> Vec<u8> {
        let mut module = rspirv::dr::load_bytes(fixture(0, false, false)).unwrap();
        let uint = 2;
        let private_uint = 3;
        let zero = 4;
        let variable = 5;
        let pointer_copy = 9;
        let loaded = 10;
        let value_copy = 11;
        let sum = 12;
        module.header.as_mut().unwrap().bound = 13;

        let mut instructions = Vec::new();
        if scaffold {
            module.types_global_values.insert(
                4,
                Instruction::new(
                    Op::Variable,
                    Some(private_uint),
                    Some(variable),
                    vec![
                        Operand::StorageClass(StorageClass::Private),
                        Operand::IdRef(zero),
                    ],
                ),
            );
            module.entry_points[0]
                .operands
                .push(Operand::IdRef(variable));
            if store_through_pointer {
                instructions.push(Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(variable), Operand::IdRef(zero)],
                ));
            }
            instructions.extend([
                Instruction::new(
                    Op::CopyObject,
                    Some(private_uint),
                    Some(pointer_copy),
                    vec![Operand::IdRef(variable)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(uint),
                    Some(loaded),
                    vec![Operand::IdRef(pointer_copy)],
                ),
                Instruction::new(
                    Op::CopyObject,
                    Some(uint),
                    Some(value_copy),
                    vec![Operand::IdRef(loaded)],
                ),
                Instruction::new(
                    Op::IAdd,
                    Some(uint),
                    Some(sum),
                    vec![Operand::IdRef(value_copy), Operand::IdRef(zero)],
                ),
            ]);
        } else {
            instructions.push(Instruction::new(
                Op::IAdd,
                Some(uint),
                Some(sum),
                vec![Operand::IdRef(zero), Operand::IdRef(zero)],
            ));
        }
        instructions.append(&mut module.functions[0].blocks[0].instructions);
        module.functions[0].blocks[0].instructions = instructions;
        module
            .assemble()
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect()
    }

    #[test]
    fn classifies_byte_identity_as_dc0() {
        let module = fixture(0, true, false);
        assert_eq!(
            classify_spirv_delta(&module, &module).unwrap(),
            SpirvDelta::Dc0
        );
    }

    #[test]
    fn classifies_id_only_delta_as_dc1() {
        assert_eq!(
            classify_spirv_delta(&fixture(0, true, false), &fixture(40, true, false)).unwrap(),
            SpirvDelta::Dc1
        );
    }

    #[test]
    fn classifies_declaration_order_only_delta_as_dc1() {
        let before = fixture(0, false, false);
        let after = reorder_independent_declarations(&before);
        assert_eq!(
            classify_spirv_delta(&before, &after).unwrap(),
            SpirvDelta::Dc1
        );
    }

    #[test]
    fn classifies_dependent_declaration_order_delta_as_dc1() {
        let (before, after) = reorder_dependent_undef(&fixture(0, false, false));
        assert_eq!(
            classify_spirv_delta(&before, &after).unwrap(),
            SpirvDelta::Dc1
        );
    }

    #[test]
    fn classifies_private_global_order_delta_as_dc1() {
        let (before, _) = reorder_private_globals(&fixture(0, true, false));
        let before = load_module(&before, "private-global-order fixture").unwrap();
        let diff = "\
-               OpEntryPoint GLCompute %7 \"main\" %5 %10
+               OpEntryPoint GLCompute %7 \"main\" %11 %12
-          %5 = OpVariable %3 Private
-         %10 = OpVariable %3 Private
+         %11 = OpVariable %3 Private
+         %12 = OpVariable %3 Private
";
        let changed = changed_lines(diff);
        assert!(is_global_declaration_order_delta(&before, diff, &changed));
    }

    #[test]
    fn classifies_dead_private_and_copy_removal_as_dc2() {
        assert_eq!(
            classify_spirv_delta(&fixture(0, true, false), &fixture(0, false, false)).unwrap(),
            SpirvDelta::Dc2
        );
    }

    #[test]
    fn classifies_immutable_private_load_scaffolding_as_dc2() {
        assert_eq!(
            classify_spirv_delta(
                &private_load_scaffold(true, false),
                &private_load_scaffold(false, false),
            )
            .unwrap(),
            SpirvDelta::Dc2
        );
    }

    #[test]
    fn refuses_private_scaffolding_with_a_store() {
        assert!(matches!(
            classify_spirv_delta(
                &private_load_scaffold(true, true),
                &private_load_scaffold(false, false),
            )
            .unwrap(),
            SpirvDelta::Other { .. }
        ));
    }

    #[test]
    fn classifies_executable_delta_as_other_with_first_line() {
        let verdict =
            classify_spirv_delta(&fixture(0, false, false), &fixture(0, false, true)).unwrap();
        let SpirvDelta::Other {
            first_offending_line,
        } = verdict
        else {
            panic!("expected Other, got {verdict:?}");
        };
        assert!(
            first_offending_line.contains("OpReturn")
                || first_offending_line.contains("OpUnreachable")
                || first_offending_line.contains("post opcode"),
            "unexpected first offending line: {first_offending_line}"
        );
    }
}
