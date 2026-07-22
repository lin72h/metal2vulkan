//! Metal function-constant specialization: bake explicit `[[function_constant(N)]]` values into
//! already-assembled SPIR-V. Used by the byte-conformance harness so translated FC kernels take the
//! same specialized path as the Apple oracle. Not part of the primary emit path.

use crate::spirv_module::load_bytes;

/// Parse the FC index `N` out of an `air.fc_initializer` global's mangled name — the stable
/// `...MTL_FC_INIT_<N>_<suffix>` shape the AIR/LLVM backend emits for a `[[function_constant(N)]]`.
/// Returns `None` for any name lacking that ABI marker (working copies, ordinary globals), so this
/// keys only on the documented Metal function-constant machinery, never on a shader-specific name.
fn fc_init_index(name: &str) -> Option<u32> {
    let rest = name.split("MTL_FC_INIT_").nth(1)?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<u32>().ok()
}

/// Bake explicit values into a module's Metal function constants, in place on assembled SPIR-V.
///
/// The AIR/LLVM backend lowers each `[[function_constant(N)]]` to a module-scope **Private**
/// `OpVariable` named `<mangled>.MTL_FC_INIT_<N>_<suffix>`, initialized to `OpConstantNull` (the
/// disabled/zero default), and copies it into a working-copy global at entry which the kernel body
/// reads. Repointing that INIT variable's initializer at an `OpConstant <ty> value` bakes the chosen
/// function-constant value into the module — the exact analogue of what `MTLFunctionConstantValues`
/// does at Metal pipeline creation, applied here at translation time. The byte-conformance harness
/// pairs this with the same values on the Apple oracle so both sides take the same specialized code
/// path (many function-constant kernels otherwise fold every FC to 0 → `udiv`-by-zero / unbounded loop → no
/// derivable oracle). `values` maps FC index → value; unlisted indices keep their zero default. Only
/// scalar-integer function constants are supported (the ones that gate loop bounds / divisors);
/// a listed index whose global or scalar-int pointee type is not found is a hard error, so a stale
/// override can never silently no-op.
pub fn specialize_function_constants(spv: &[u8], values: &[(u32, u64)]) -> Result<Vec<u8>, String> {
    use crate::spirv_module::Instruction;
    use crate::spirv_module::Operand;
    use spirv::Op;
    if values.is_empty() {
        return Ok(spv.to_vec());
    }
    let want: std::collections::HashMap<u32, u64> = values.iter().copied().collect();
    let mut module = load_bytes(spv).map_err(|e| format!("SPIR-V load: {e:?}"))?;

    // OpName: var id -> FC index, restricted to `MTL_FC_INIT_<N>` globals.
    let mut var_index: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for inst in &module.debug_names {
        if inst.class.opcode != Op::Name {
            continue;
        }
        if let (Some(Operand::IdRef(id)), Some(Operand::LiteralString(s))) =
            (inst.operands.first(), inst.operands.get(1))
        {
            if let Some(idx) = fc_init_index(s) {
                var_index.insert(*id, idx);
            }
        }
    }

    // Type tables: pointer id -> pointee id; int type id -> bit width.
    let mut pointee: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut int_width: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for inst in &module.types_global_values {
        match inst.class.opcode {
            Op::TypePointer => {
                if let (Some(rid), Some(Operand::IdRef(p))) = (inst.result_id, inst.operands.get(1))
                {
                    pointee.insert(rid, *p);
                }
            }
            Op::TypeInt => {
                if let (Some(rid), Some(Operand::LiteralBit32(w))) =
                    (inst.result_id, inst.operands.first())
                {
                    int_width.insert(rid, *w);
                }
            }
            _ => {}
        }
    }

    // Synthesize one OpConstant per (scalar-int type, value) and repoint each targeted variable's
    // initializer at it. Allocate fresh ids above the current bound.
    let mut next_id = module.header.as_ref().map(|h| h.bound).unwrap_or(1);
    let mut const_for: std::collections::HashMap<(u32, u64), u32> =
        std::collections::HashMap::new();
    let mut new_consts: Vec<Instruction> = vec![];
    // Collect the edits first (immutable borrow of the table), then apply.
    let mut edits: Vec<(usize, u32)> = vec![]; // (var instruction index, const id)
    for (vi, inst) in module.types_global_values.iter().enumerate() {
        if inst.class.opcode != Op::Variable {
            continue;
        }
        let Some(vid) = inst.result_id else { continue };
        let Some(&idx) = var_index.get(&vid) else {
            continue;
        };
        let Some(&val) = want.get(&idx) else { continue };
        let ptr_ty = inst
            .result_type
            .ok_or_else(|| format!("FC var %{vid} has no result type"))?;
        let scalar_ty = *pointee
            .get(&ptr_ty)
            .ok_or_else(|| format!("FC var %{vid}: pointer type %{ptr_ty} has no pointee"))?;
        let width = *int_width.get(&scalar_ty).ok_or_else(|| {
            format!("FC index {idx}: pointee %{scalar_ty} is not a scalar integer type")
        })?;
        let cid = *const_for.entry((scalar_ty, val)).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            let operand = if width > 32 {
                Operand::LiteralBit64(val)
            } else {
                Operand::LiteralBit32(val as u32)
            };
            new_consts.push(Instruction::new(
                Op::Constant,
                Some(scalar_ty),
                Some(id),
                vec![operand],
            ));
            id
        });
        edits.push((vi, cid));
    }

    let requested: std::collections::HashSet<u32> = want.keys().copied().collect();
    let applied: std::collections::HashSet<u32> = edits
        .iter()
        .filter_map(|(vi, _)| module.types_global_values[*vi].result_id)
        .filter_map(|vid| var_index.get(&vid).copied())
        .collect();
    let missing: Vec<u32> = requested.difference(&applied).copied().collect();
    if !missing.is_empty() {
        return Err(format!(
            "specialize_function_constants: no MTL_FC_INIT global for FC index(es) {missing:?} \
             (module has indices {:?})",
            var_index.values().collect::<std::collections::HashSet<_>>()
        ));
    }

    // Map: FC variable id -> its constant id, and const id -> the OpConstant instruction.
    let var_to_const: std::collections::HashMap<u32, u32> = edits
        .iter()
        .filter_map(|(vi, cid)| module.types_global_values[*vi].result_id.map(|v| (v, *cid)))
        .collect();
    let mut const_inst: std::collections::HashMap<u32, Instruction> = new_consts
        .into_iter()
        .filter_map(|c| c.result_id.map(|id| (id, c)))
        .collect();

    // Rebuild the type/global section. An OpConstant must follow its result-type definition but
    // precede the OpVariable that references it as an initializer — and this SPIR-V section
    // INTERLEAVES types and variables (e.g. a `%ushort` type defined after an earlier `%uchar`
    // variable), so a single "insert before the first variable" is wrong (forward type ref). Instead
    // emit each constant immediately before the FIRST FC variable that uses it: that variable already
    // references its pointer-to-scalar type, so the scalar type is guaranteed defined earlier. Set the
    // variable's initializer to the constant as we go.
    let mut rebuilt: Vec<Instruction> = Vec::with_capacity(module.types_global_values.len() + 1);
    let mut emitted: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for mut inst in module.types_global_values.drain(..) {
        if inst.class.opcode == Op::Variable {
            if let Some(&cid) = inst.result_id.as_ref().and_then(|v| var_to_const.get(v)) {
                if emitted.insert(cid) {
                    if let Some(c) = const_inst.remove(&cid) {
                        rebuilt.push(c);
                    }
                }
                // Repoint (or append) the initializer operand.
                if inst.operands.len() >= 2 {
                    inst.operands[1] = Operand::IdRef(cid);
                } else {
                    inst.operands.push(Operand::IdRef(cid));
                }
            }
        }
        rebuilt.push(inst);
    }
    module.types_global_values = rebuilt;
    if let Some(h) = module.header.as_mut() {
        h.bound = next_id;
    }

    // If the baked values make AIR function-constant branches static, prune the dead arms and then
    // rebuild the entry interface from the variables still referenced by function bodies. This keeps
    // mutually-exclusive FC-gated resources honest: a Metal function may present a texture2d and a
    // texture2d_array at the same `[[texture(N)]]` slot under different FC predicates, but Vulkan
    // cannot bind two image view types to one descriptor in one specialized module. The pruning pass
    // is structural and already used by native retry tiers; this helper is opt-in because it only runs
    // for explicit harness/user-provided FC values.
    if !module.entry_points.is_empty() {
        let before_prune = module.clone();
        if crate::native::prune_constant_branches_module(&mut module).is_ok() {
            restore_loop_merges_removed_by_fc_prune(&before_prune, &mut module);
            drop_unreferenced_entry_interface_globals(&mut module);
        }
    }

    Ok(module
        .assemble()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect())
}

fn restore_loop_merges_removed_by_fc_prune(
    before: &crate::spirv_module::Module,
    after: &mut crate::spirv_module::Module,
) {
    use crate::spirv_module::{Block, Function, Instruction, Operand};
    use spirv::Op;
    use std::collections::{HashMap, HashSet};

    fn function_id(function: &Function) -> Option<u32> {
        function.def.as_ref().and_then(|def| def.result_id)
    }

    fn block_id(block: &Block) -> Option<u32> {
        block.label.as_ref().and_then(|label| label.result_id)
    }

    fn loop_merge(block: &Block) -> Option<&Instruction> {
        let n = block.instructions.len();
        if n >= 2 && block.instructions[n - 2].class.opcode == Op::LoopMerge {
            Some(&block.instructions[n - 2])
        } else {
            None
        }
    }

    let before_functions = before
        .functions
        .iter()
        .filter_map(|function| function_id(function).map(|id| (id, function)))
        .collect::<HashMap<_, _>>();

    for function in &mut after.functions {
        let Some(fid) = function_id(function) else {
            continue;
        };
        let Some(before_function) = before_functions.get(&fid) else {
            continue;
        };
        let before_blocks = before_function
            .blocks
            .iter()
            .filter_map(|block| block_id(block).map(|id| (id, block)))
            .collect::<HashMap<_, _>>();
        let mut alive = function
            .blocks
            .iter()
            .filter_map(block_id)
            .collect::<HashSet<_>>();
        let mut synthesized_merges = HashSet::new();
        for block in &mut function.blocks {
            if loop_merge(block).is_some() {
                continue;
            }
            let Some(id) = block_id(block) else { continue };
            let Some(before_block) = before_blocks.get(&id) else {
                continue;
            };
            let Some(original_merge) = loop_merge(before_block).cloned() else {
                continue;
            };
            let (Some(Operand::IdRef(merge)), Some(Operand::IdRef(cont))) = (
                original_merge.operands.first(),
                original_merge.operands.get(1),
            ) else {
                continue;
            };
            let merge = *merge;
            let cont = *cont;
            if !alive.contains(&cont) {
                continue;
            }
            if block
                .instructions
                .iter()
                .any(|inst| inst.class.opcode == Op::SelectionMerge)
            {
                continue;
            }
            let insert_at = block.instructions.len().saturating_sub(1);
            block.instructions.insert(insert_at, original_merge);
            if !alive.contains(&merge) && synthesized_merges.insert(merge) {
                alive.insert(merge);
            }
        }
        if synthesized_merges.is_empty() {
            continue;
        }
        let mut new_blocks = synthesized_merges.into_iter().collect::<Vec<_>>();
        new_blocks.sort_unstable();
        for merge in new_blocks {
            function.blocks.push(Block {
                label: Some(Instruction::new(Op::Label, None, Some(merge), vec![])),
                instructions: vec![Instruction::new(Op::Unreachable, None, None, vec![])],
            });
        }
    }
}

fn drop_unreferenced_entry_interface_globals(module: &mut crate::spirv_module::Module) {
    use crate::spirv_module::Instruction;
    use crate::spirv_module::Operand;
    use spirv::Op;
    let mut function_refs = std::collections::HashSet::new();
    for function in &module.functions {
        for block in &function.blocks {
            for inst in &block.instructions {
                for op in &inst.operands {
                    if let Operand::IdRef(id)
                    | Operand::IdScope(id)
                    | Operand::IdMemorySemantics(id) = op
                    {
                        function_refs.insert(*id);
                    }
                }
            }
        }
    }

    let original_interface_ids = module
        .entry_points
        .iter()
        .flat_map(|entry| entry.operands.iter().skip(3))
        .filter_map(|op| match op {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        })
        .collect::<std::collections::HashSet<_>>();
    let mut interface_ids = std::collections::HashSet::new();
    for entry in &mut module.entry_points {
        let mut rebuilt = Vec::with_capacity(entry.operands.len());
        for (idx, op) in entry.operands.iter().cloned().enumerate() {
            if idx < 3 {
                rebuilt.push(op);
                continue;
            }
            match op {
                Operand::IdRef(id) if function_refs.contains(&id) => {
                    interface_ids.insert(id);
                    rebuilt.push(Operand::IdRef(id));
                }
                Operand::IdRef(_) => {}
                other => rebuilt.push(other),
            }
        }
        entry.operands = rebuilt;
    }

    module.types_global_values.retain(|inst| {
        if inst.class.opcode != Op::Variable {
            return true;
        }
        let Some(id) = inst.result_id else {
            return true;
        };
        if !original_interface_ids.contains(&id) {
            return true;
        }
        function_refs.contains(&id) || interface_ids.contains(&id)
    });

    let defined = defined_ids(module);
    let keep = |inst: &Instruction| {
        !matches!(
            inst.operands.first(),
            Some(Operand::IdRef(id)) if !defined.contains(id)
        )
    };
    module.debug_names.retain(keep);
    module.annotations.retain(keep);
}

fn defined_ids(module: &crate::spirv_module::Module) -> std::collections::HashSet<spirv::Word> {
    let mut out = std::collections::HashSet::new();
    for inst in &module.types_global_values {
        if let Some(id) = inst.result_id {
            out.insert(id);
        }
    }
    for inst in &module.ext_inst_imports {
        if let Some(id) = inst.result_id {
            out.insert(id);
        }
    }
    for function in &module.functions {
        if let Some(id) = function.def.as_ref().and_then(|def| def.result_id) {
            out.insert(id);
        }
        for param in &function.parameters {
            if let Some(id) = param.result_id {
                out.insert(id);
            }
        }
        for block in &function.blocks {
            if let Some(id) = block.label.as_ref().and_then(|label| label.result_id) {
                out.insert(id);
            }
            for inst in &block.instructions {
                if let Some(id) = inst.result_id {
                    out.insert(id);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::{Block, Function, Instruction, Module, ModuleHeader, Operand};
    use spirv::{
        AddressingModel, Capability, ExecutionModel, FunctionControl, MemoryModel, Op,
        SelectionControl, StorageClass,
    };

    fn fixture_bytes(bound: u32, globals: Vec<Instruction>, names: Vec<Instruction>) -> Vec<u8> {
        let mut module = Module::new();
        let mut header = ModuleHeader::new(bound);
        header.set_version(1, 4);
        module.header = Some(header);
        module.capabilities.push(Instruction::new(
            Op::Capability,
            None,
            None,
            vec![Operand::Capability(Capability::Shader)],
        ));
        module.types_global_values = globals;
        module.debug_names = names;
        module
            .assemble()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    }

    fn name(id: u32, value: &str) -> Instruction {
        Instruction::new(
            Op::Name,
            None,
            None,
            vec![Operand::IdRef(id), Operand::from(value)],
        )
    }

    fn block(label: u32, instructions: Vec<Instruction>) -> Block {
        Block {
            label: Some(Instruction::new(Op::Label, None, Some(label), vec![])),
            instructions,
        }
    }

    #[test]
    fn fc_prune_restores_loop_merge_when_merge_block_was_pruned() {
        let header = 10;
        let body = 11;
        let cont = 12;
        let merge = 13;
        let mut before = Module::new();
        before.functions.push(Function {
            def: Some(Instruction::new(Op::Function, Some(1), Some(50), vec![])),
            end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
            parameters: vec![],
            blocks: vec![
                block(
                    header,
                    vec![
                        Instruction::new(
                            Op::LoopMerge,
                            None,
                            None,
                            vec![Operand::IdRef(merge), Operand::IdRef(cont)],
                        ),
                        Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(body)]),
                    ],
                ),
                block(
                    body,
                    vec![Instruction::new(
                        Op::Branch,
                        None,
                        None,
                        vec![Operand::IdRef(cont)],
                    )],
                ),
                block(
                    cont,
                    vec![Instruction::new(
                        Op::Branch,
                        None,
                        None,
                        vec![Operand::IdRef(header)],
                    )],
                ),
                block(
                    merge,
                    vec![Instruction::new(Op::Unreachable, None, None, vec![])],
                ),
            ],
        });
        let mut after = before.clone();
        after.functions[0].blocks[0].instructions.remove(0);
        after.functions[0].blocks.retain(|b| {
            b.label
                .as_ref()
                .and_then(|label| label.result_id)
                .is_some_and(|id| id != merge)
        });

        restore_loop_merges_removed_by_fc_prune(&before, &mut after);

        let fixed_header = &after.functions[0].blocks[0];
        assert_eq!(fixed_header.instructions[0].class.opcode, Op::LoopMerge);
        assert_eq!(
            fixed_header.instructions[0].operands,
            vec![Operand::IdRef(merge), Operand::IdRef(cont)]
        );
        let labels = after.functions[0]
            .blocks
            .iter()
            .filter_map(|b| b.label.as_ref().and_then(|label| label.result_id))
            .collect::<Vec<_>>();
        assert_eq!(labels, vec![header, body, cont, merge]);
    }

    #[test]
    fn specialize_prunes_dead_fc_interface_globals() {
        let mut module = Module::new();
        let mut header = ModuleHeader::new(22);
        header.set_version(1, 4);
        module.header = Some(header);
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
                Operand::IdRef(12),
                Operand::LiteralString("main".into()),
                Operand::IdRef(9),
                Operand::IdRef(10),
            ],
        ));
        module.types_global_values = vec![
            Instruction::new(Op::TypeVoid, None, Some(1), vec![]),
            Instruction::new(Op::TypeBool, None, Some(2), vec![]),
            Instruction::new(
                Op::TypeInt,
                None,
                Some(3),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            Instruction::new(
                Op::TypePointer,
                None,
                Some(4),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(3),
                ],
            ),
            Instruction::new(
                Op::TypePointer,
                None,
                Some(5),
                vec![
                    Operand::StorageClass(StorageClass::Function),
                    Operand::IdRef(3),
                ],
            ),
            Instruction::new(Op::ConstantNull, Some(3), Some(6), vec![]),
            Instruction::new(
                Op::Variable,
                Some(4),
                Some(8),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(6),
                ],
            ),
            Instruction::new(
                Op::Variable,
                Some(4),
                Some(9),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(6),
                ],
            ),
            Instruction::new(
                Op::Variable,
                Some(4),
                Some(10),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(6),
                ],
            ),
            Instruction::new(Op::TypeFunction, None, Some(11), vec![Operand::IdRef(1)]),
        ];
        module.debug_names = vec![
            name(8, "_Z1x.MTL_FC_INIT_0_b"),
            name(9, "live_global"),
            name(10, "dead_global"),
        ];

        let mut function = Function::new();
        function.def = Some(Instruction::new(
            Op::Function,
            Some(1),
            Some(12),
            vec![
                Operand::FunctionControl(FunctionControl::NONE),
                Operand::IdRef(11),
            ],
        ));
        function.blocks = vec![
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(13), vec![])),
                instructions: vec![
                    Instruction::new(
                        Op::Variable,
                        Some(5),
                        Some(19),
                        vec![Operand::StorageClass(StorageClass::Function)],
                    ),
                    Instruction::new(Op::Load, Some(3), Some(14), vec![Operand::IdRef(8)]),
                    Instruction::new(
                        Op::IEqual,
                        Some(2),
                        Some(15),
                        vec![Operand::IdRef(14), Operand::IdRef(6)],
                    ),
                    Instruction::new(
                        Op::SelectionMerge,
                        None,
                        None,
                        vec![
                            Operand::IdRef(18),
                            Operand::SelectionControl(SelectionControl::NONE),
                        ],
                    ),
                    Instruction::new(
                        Op::BranchConditional,
                        None,
                        None,
                        vec![Operand::IdRef(15), Operand::IdRef(16), Operand::IdRef(17)],
                    ),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(16), vec![])),
                instructions: vec![
                    Instruction::new(Op::Load, Some(3), Some(20), vec![Operand::IdRef(9)]),
                    Instruction::new(
                        Op::Store,
                        None,
                        None,
                        vec![Operand::IdRef(19), Operand::IdRef(20)],
                    ),
                    Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(18)]),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(17), vec![])),
                instructions: vec![
                    Instruction::new(Op::Load, Some(3), Some(21), vec![Operand::IdRef(10)]),
                    Instruction::new(
                        Op::Store,
                        None,
                        None,
                        vec![Operand::IdRef(19), Operand::IdRef(21)],
                    ),
                    Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(18)]),
                ],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(18), vec![])),
                instructions: vec![Instruction::new(Op::Return, None, None, vec![])],
            },
        ];
        function.end = Some(Instruction::new(Op::FunctionEnd, None, None, vec![]));
        module.functions.push(function);

        let bytes = module
            .assemble()
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        let out = specialize_function_constants(&bytes, &[(0, 0)]).expect("specialize");
        let m = load_bytes(&out).expect("reload");
        let entry_interface = m.entry_points[0]
            .operands
            .iter()
            .skip(3)
            .filter_map(|op| match op {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            })
            .collect::<std::collections::HashSet<_>>();
        let variables = m
            .types_global_values
            .iter()
            .filter(|inst| inst.class.opcode == Op::Variable)
            .filter_map(|inst| inst.result_id)
            .collect::<std::collections::HashSet<_>>();

        assert!(
            entry_interface.contains(&9),
            "live global stays in interface"
        );
        assert!(
            !entry_interface.contains(&10),
            "dead FC-arm global leaves interface"
        );
        assert!(variables.contains(&9), "live global variable stays");
        assert!(!variables.contains(&10), "dead FC-arm global is dropped");
    }

    /// Build a minimal module with one Private `MTL_FC_INIT_0` uint variable initialized to
    /// OpConstantNull, plus a decoy working-copy variable (no ABI marker), and confirm
    /// `specialize_function_constants` repoints only the INIT variable's initializer to a fresh
    /// `OpConstant uint 7` while leaving the decoy untouched.
    #[test]
    fn specialize_repoints_init_initializer() {
        let bytes = fixture_bytes(
            6,
            vec![
                Instruction::new(
                    Op::TypeInt,
                    None,
                    Some(1),
                    vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
                ),
                Instruction::new(
                    Op::TypePointer,
                    None,
                    Some(2),
                    vec![
                        Operand::StorageClass(StorageClass::Private),
                        Operand::IdRef(1),
                    ],
                ),
                Instruction::new(Op::ConstantNull, Some(1), Some(3), vec![]),
                Instruction::new(
                    Op::Variable,
                    Some(2),
                    Some(4),
                    vec![
                        Operand::StorageClass(StorageClass::Private),
                        Operand::IdRef(3),
                    ],
                ),
                Instruction::new(
                    Op::Variable,
                    Some(2),
                    Some(5),
                    vec![
                        Operand::StorageClass(StorageClass::Private),
                        Operand::IdRef(3),
                    ],
                ),
            ],
            vec![
                name(4, "_ZN3app11fc_channelsE.MTL_FC_INIT_0_j"),
                name(5, "_ZN3app11fc_channelsE.13"),
            ],
        );

        let out = specialize_function_constants(&bytes, &[(0, 7)]).expect("specialize");
        let m = load_bytes(&out).expect("reload");

        // Find the INIT variable and read its initializer id.
        let init_name = m
            .debug_names
            .iter()
            .find(|i| {
                matches!(i.operands.get(1), Some(Operand::LiteralString(s)) if s.contains("MTL_FC_INIT_0"))
            })
            .and_then(|i| match i.operands.first() {
                Some(Operand::IdRef(id)) => Some(*id),
                _ => None,
            })
            .expect("init name");
        let init_var = m
            .types_global_values
            .iter()
            .find(|i| i.class.opcode == Op::Variable && i.result_id == Some(init_name))
            .expect("init var");
        let init_id = match init_var.operands.get(1) {
            Some(Operand::IdRef(id)) => *id,
            other => panic!("init var has no initializer: {other:?}"),
        };
        let init_const = m
            .types_global_values
            .iter()
            .find(|i| i.class.opcode == Op::Constant && i.result_id == Some(init_id))
            .expect("init constant def");
        assert_eq!(
            init_const.operands.first(),
            Some(&Operand::LiteralBit32(7)),
            "INIT initializer should be OpConstant uint 7"
        );

        // Unknown index must error rather than silently no-op.
        assert!(specialize_function_constants(&bytes, &[(9, 1)]).is_err());
    }

    /// Regression: real modules INTERLEAVE types and variables (a `%ushort` type defined AFTER an
    /// earlier `%uchar` variable). Each synthesized OpConstant must be emitted after its scalar type
    /// and before the variable that uses it, or spirv-val rejects a forward type reference
    /// ("Type Id N is not a type"). Build an interleaved module with two FCs of different widths and
    /// assert every constant's result-type and initializer-use ordering holds.
    #[test]
    fn specialize_handles_interleaved_types_and_vars() {
        let private_pointer = |id, pointee| {
            Instruction::new(
                Op::TypePointer,
                None,
                Some(id),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(pointee),
                ],
            )
        };
        let variable = |id, pointer, initializer| {
            Instruction::new(
                Op::Variable,
                Some(pointer),
                Some(id),
                vec![
                    Operand::StorageClass(StorageClass::Private),
                    Operand::IdRef(initializer),
                ],
            )
        };
        let bytes = fixture_bytes(
            9,
            vec![
                Instruction::new(
                    Op::TypeInt,
                    None,
                    Some(1),
                    vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
                ),
                private_pointer(2, 1),
                Instruction::new(Op::ConstantNull, Some(1), Some(3), vec![]),
                variable(4, 2, 3),
                Instruction::new(
                    Op::TypeInt,
                    None,
                    Some(5),
                    vec![Operand::LiteralBit32(16), Operand::LiteralBit32(0)],
                ),
                private_pointer(6, 5),
                Instruction::new(Op::ConstantNull, Some(5), Some(7), vec![]),
                variable(8, 6, 7),
            ],
            vec![
                name(4, "_Z3fooE.MTL_FC_INIT_9_b"),
                name(8, "_Z3barE.MTL_FC_INIT_8_t"),
            ],
        );

        let out = specialize_function_constants(&bytes, &[(8, 32), (9, 1)]).expect("specialize");
        let m = load_bytes(&out).expect("reload");
        use spirv::Op;
        // position of each id in the type/global section
        let pos = |id: u32| {
            m.types_global_values
                .iter()
                .position(|i| i.result_id == Some(id))
        };
        for var in m
            .types_global_values
            .iter()
            .filter(|i| i.class.opcode == Op::Variable)
        {
            // skip non-FC (none here) — every var is an FC var
            let init = match var.operands.get(1) {
                Some(Operand::IdRef(id)) => *id,
                other => panic!("var missing initializer: {other:?}"),
            };
            let cinst = m
                .types_global_values
                .iter()
                .find(|i| i.class.opcode == Op::Constant && i.result_id == Some(init))
                .expect("constant def present");
            let cty = cinst.result_type.expect("const has type");
            let (pc, pv, pt) = (
                pos(init).unwrap(),
                pos(var.result_id.unwrap()).unwrap(),
                pos(cty).unwrap(),
            );
            assert!(pt < pc, "constant type must precede the constant");
            assert!(
                pc < pv,
                "constant must precede the variable that initializes with it"
            );
        }
        // And the values landed.
        let val_of = |marker: &str| -> u64 {
            let vid = m
                .debug_names
                .iter()
                .find(|i| matches!(i.operands.get(1), Some(Operand::LiteralString(s)) if s.contains(marker)))
                .and_then(|i| match i.operands.first() { Some(Operand::IdRef(id)) => Some(*id), _ => None })
                .unwrap();
            let init = match m
                .types_global_values
                .iter()
                .find(|i| i.result_id == Some(vid))
                .unwrap()
                .operands
                .get(1)
            {
                Some(Operand::IdRef(id)) => *id,
                _ => panic!(),
            };
            match m
                .types_global_values
                .iter()
                .find(|i| i.result_id == Some(init))
                .unwrap()
                .operands
                .first()
            {
                Some(Operand::LiteralBit32(v)) => *v as u64,
                Some(Operand::LiteralBit64(v)) => *v,
                other => panic!("{other:?}"),
            }
        };
        assert_eq!(val_of("MTL_FC_INIT_8"), 32);
        assert_eq!(val_of("MTL_FC_INIT_9"), 1);
    }
}
