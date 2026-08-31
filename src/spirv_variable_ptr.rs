//! Shared SPIR-V cleanups for variable-pointer portability.
//!
//! Native emission and the final interface pass both close the `VariablePointers*` capability set.
//! Keep the structural cleanups that can make those capabilities unnecessary in one place so the
//! final emitted module cannot reintroduce a driver-fragile variable-pointer path after native
//! cleanup already removed it.

use crate::spirv_module::{Instruction, Module, Operand};
use spirv::{Op, StorageClass, Word};
use std::collections::{HashMap, HashSet};

#[derive(Clone)]
struct AccessChainDef {
    root: Word,
    indices: Vec<Word>,
    parent_before_last: Option<Word>,
}

/// Lower the simple StorageBuffer shape:
///
/// ```text
/// %base = OpAccessChain    %_ptr_StorageBuffer_T %root ... %base_index
/// %ptr  = OpPtrAccessChain %_ptr_StorageBuffer_U %base %dynamic ...
/// ```
///
/// to:
///
/// ```text
/// %sum = OpIAdd %idx_ty %base_index %dynamic  ; omitted when base_index is zero
/// %ptr = OpAccessChain %_ptr_StorageBuffer_U %root ... %sum ...
/// ```
///
/// but only when `%base_index` indexes an array/runtime-array element, the dynamic offset has the same
/// integer type when addition is needed, and the recomputed access-chain pointee matches the original
/// result type. This preserves byte address semantics while avoiding `VariablePointersStorageBuffer`
/// for address math that is expressible as a normal logical access chain.
pub(crate) fn lower_zero_base_storage_buffer_ptr_access_chains(module: &mut Module) -> usize {
    if !module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .any(|inst| {
            matches!(
                inst.class.opcode,
                Op::PtrAccessChain | Op::InBoundsPtrAccessChain
            )
        })
    {
        return 0;
    }
    let defs = collect_defs(module);
    let zero_constants = zero_integer_constants(&defs);
    let access_chains = collect_access_chain_defs(module, &defs, &zero_constants);
    let direct_demotions =
        collect_zero_stride_ptr_access_chain_demotions(module, &defs, &zero_constants);
    if access_chains.is_empty() && direct_demotions.is_empty() {
        return 0;
    }

    struct Rewrite {
        operands: Vec<Operand>,
        add: Option<(Word, Word, Word, Word)>, // result, type, lhs, rhs
    }

    let mut next_id = module.header.as_ref().map(|h| h.bound).unwrap_or(0);
    let mut rewrites: HashMap<Word, Rewrite> = HashMap::new();
    for inst in module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
    {
        if !matches!(
            inst.class.opcode,
            Op::PtrAccessChain | Op::InBoundsPtrAccessChain
        ) {
            continue;
        }
        let (Some(result_id), Some(result_type)) = (inst.result_id, inst.result_type) else {
            continue;
        };
        let Some((StorageClass::StorageBuffer, result_pointee)) = ptr_info(&defs, result_type)
        else {
            continue;
        };
        let Some(Operand::IdRef(base)) = inst.operands.first() else {
            continue;
        };
        let Some(base_chain) = access_chains.get(base) else {
            continue;
        };
        let Some((StorageClass::StorageBuffer, root_pointee)) =
            value_pointer_info(&defs, base_chain.root)
        else {
            continue;
        };
        if base_chain.indices.is_empty() {
            continue;
        }
        let Some(parent) = base_chain.parent_before_last else {
            continue;
        };
        if !is_array_indexable_type(&defs, parent) {
            continue;
        }
        let Some(ptr_indices) = id_ref_operands(&inst.operands[1..]) else {
            continue;
        };
        if ptr_indices.is_empty() {
            continue;
        }

        let mut new_indices = Vec::with_capacity(base_chain.indices.len() + ptr_indices.len() - 1);
        new_indices.extend_from_slice(&base_chain.indices[..base_chain.indices.len() - 1]);
        let Some(last_base_index) = base_chain.indices.last() else {
            continue;
        };
        let add = if zero_constants.contains(last_base_index) {
            new_indices.push(ptr_indices[0]);
            None
        } else {
            let Some(index_ty) = integer_value_type(&defs, *last_base_index) else {
                continue;
            };
            if integer_value_type(&defs, ptr_indices[0]) != Some(index_ty) {
                continue;
            }
            let sum = next_id;
            next_id += 1;
            new_indices.push(sum);
            Some((sum, index_ty, *last_base_index, ptr_indices[0]))
        };
        new_indices.extend_from_slice(&ptr_indices[1..]);
        if walk_access_chain_pointee(&defs, root_pointee, &new_indices) != Some(result_pointee) {
            continue;
        }

        let mut operands = Vec::with_capacity(1 + new_indices.len());
        operands.push(Operand::IdRef(base_chain.root));
        operands.extend(new_indices.into_iter().map(Operand::IdRef));
        rewrites.insert(result_id, Rewrite { operands, add });
    }

    let mut changed = 0;
    for block in module
        .functions
        .iter_mut()
        .flat_map(|function| &mut function.blocks)
    {
        let mut pos = 0;
        while pos < block.instructions.len() {
            let Some(result_id) = block.instructions[pos].result_id else {
                pos += 1;
                continue;
            };
            if let Some(rewrite) = rewrites.remove(&result_id) {
                if let Some((sum, ty, lhs, rhs)) = rewrite.add {
                    block.instructions.insert(
                        pos,
                        Instruction::new(
                            Op::IAdd,
                            Some(ty),
                            Some(sum),
                            vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
                        ),
                    );
                    pos += 1;
                }
                block.instructions[pos].class.opcode = Op::AccessChain;
                block.instructions[pos].operands = rewrite.operands;
                changed += 1;
                pos += 1;
                continue;
            }
            if direct_demotions.contains(&result_id) {
                block.instructions[pos].class.opcode = Op::AccessChain;
                changed += 1;
            }
            pos += 1;
        }
    }

    if changed != 0 {
        if let Some(header) = module.header.as_mut() {
            header.bound = next_id;
        }
    }
    changed
}

pub(crate) fn rewrite_storage_buffer_atomic_scopes(module: &mut Module) -> usize {
    if !module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .any(|inst| atomic_scope_semantics_indices(inst).is_some())
    {
        return 0;
    }
    let mut defs = collect_defs(module);
    let mut rewrites = Vec::new();
    for (fi, function) in module.functions.iter().enumerate() {
        for (bi, block) in function.blocks.iter().enumerate() {
            for (ii, inst) in block.instructions.iter().enumerate() {
                let Some((scope_idx, semantics_indices)) = atomic_scope_semantics_indices(inst)
                else {
                    continue;
                };
                let Some(Operand::IdRef(ptr)) = inst.operands.first() else {
                    continue;
                };
                let Some((StorageClass::StorageBuffer, _)) = value_pointer_info(&defs, *ptr) else {
                    continue;
                };
                let Some(scope_operand) = inst.operands.get(scope_idx) else {
                    continue;
                };
                let Some(scope_id) = operand_id(scope_operand) else {
                    continue;
                };
                if constant_u32(&defs, scope_id) != Some(spirv::Scope::Workgroup as u32) {
                    continue;
                }
                let Some(scope_ty) = integer_value_type(&defs, scope_id) else {
                    continue;
                };
                let mut semantics_tys = Vec::new();
                for &semantics_idx in semantics_indices {
                    let Some(semantics_operand) = inst.operands.get(semantics_idx) else {
                        semantics_tys.clear();
                        break;
                    };
                    let Some(semantics_id) = operand_id(semantics_operand) else {
                        semantics_tys.clear();
                        break;
                    };
                    let Some(semantics_ty) = integer_value_type(&defs, semantics_id) else {
                        semantics_tys.clear();
                        break;
                    };
                    semantics_tys.push((semantics_idx, semantics_ty));
                }
                if semantics_tys.len() != semantics_indices.len() {
                    continue;
                }
                rewrites.push((fi, bi, ii, scope_idx, scope_ty, semantics_tys));
            }
        }
    }
    if rewrites.is_empty() {
        return 0;
    }

    let mut next_id = module.header.as_ref().map(|h| h.bound).unwrap_or(0);
    let mut constants = HashMap::new();
    for (_, _, _, _, scope_ty, semantics_tys) in &rewrites {
        let device = get_or_create_integer_constant(module, &mut defs, &mut next_id, *scope_ty, 1);
        constants.insert((*scope_ty, 1), device);
        for (_, semantics_ty) in semantics_tys {
            let relaxed =
                get_or_create_integer_constant(module, &mut defs, &mut next_id, *semantics_ty, 0);
            constants.insert((*semantics_ty, 0), relaxed);
        }
    }

    for (fi, bi, ii, scope_idx, scope_ty, semantics_tys) in rewrites {
        let inst = &mut module.functions[fi].blocks[bi].instructions[ii];
        let device = constants[&(scope_ty, 1)];
        inst.operands[scope_idx] = Operand::IdScope(device);
        for (semantics_idx, semantics_ty) in semantics_tys {
            let relaxed = constants[&(semantics_ty, 0)];
            inst.operands[semantics_idx] = Operand::IdMemorySemantics(relaxed);
        }
    }

    if let Some(header) = module.header.as_mut() {
        header.bound = next_id;
    }
    constants.len()
}

fn collect_zero_stride_ptr_access_chain_demotions(
    module: &Module,
    defs: &HashMap<Word, Instruction>,
    zero_constants: &HashSet<Word>,
) -> HashSet<Word> {
    module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter_map(|inst| {
            if !matches!(
                inst.class.opcode,
                Op::PtrAccessChain | Op::InBoundsPtrAccessChain
            ) {
                return None;
            }
            let result_id = inst.result_id?;
            let result_type = inst.result_type?;
            let Some((StorageClass::StorageBuffer, result_pointee)) = ptr_info(defs, result_type)
            else {
                return None;
            };
            let Some(Operand::IdRef(base)) = inst.operands.first() else {
                return None;
            };
            let (StorageClass::StorageBuffer, base_pointee) = value_pointer_info(defs, *base)?
            else {
                return None;
            };
            let ptr_indices = id_ref_operands(&inst.operands[1..])?;
            let first = ptr_indices.first()?;
            if !zero_constants.contains(first) {
                return None;
            }
            (walk_access_chain_pointee(defs, base_pointee, &ptr_indices) == Some(result_pointee))
                .then_some(result_id)
        })
        .collect()
}

fn atomic_scope_semantics_indices(inst: &Instruction) -> Option<(usize, &'static [usize])> {
    match inst.class.opcode {
        Op::AtomicLoad => Some((1, &[2])),
        Op::AtomicStore => Some((1, &[2])),
        Op::AtomicExchange
        | Op::AtomicIIncrement
        | Op::AtomicIDecrement
        | Op::AtomicIAdd
        | Op::AtomicISub
        | Op::AtomicSMin
        | Op::AtomicUMin
        | Op::AtomicSMax
        | Op::AtomicUMax
        | Op::AtomicAnd
        | Op::AtomicOr
        | Op::AtomicXor
        | Op::AtomicFAddEXT => Some((1, &[2])),
        Op::AtomicCompareExchange | Op::AtomicCompareExchangeWeak => Some((1, &[2, 3])),
        _ => None,
    }
}

fn operand_id(operand: &Operand) -> Option<Word> {
    match operand {
        Operand::IdRef(id) | Operand::IdScope(id) | Operand::IdMemorySemantics(id) => Some(*id),
        _ => None,
    }
}

fn get_or_create_integer_constant(
    module: &mut Module,
    defs: &mut HashMap<Word, Instruction>,
    next_id: &mut Word,
    ty: Word,
    value: u32,
) -> Word {
    if let Some(id) = defs.iter().find_map(|(&id, inst)| {
        (inst.class.opcode == Op::Constant
            && inst.result_type == Some(ty)
            && constant_u32(defs, id) == Some(value))
        .then_some(id)
    }) {
        return id;
    }
    let id = *next_id;
    *next_id += 1;
    let inst = Instruction::new(
        Op::Constant,
        Some(ty),
        Some(id),
        vec![Operand::LiteralBit32(value)],
    );
    module.types_global_values.push(inst.clone());
    defs.insert(id, inst);
    id
}

pub(crate) fn variable_pointer_requirement(module: &Module) -> (bool, bool) {
    if !module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .any(|inst| {
            matches!(
                inst.class.opcode,
                Op::Phi | Op::Select | Op::PtrAccessChain | Op::InBoundsPtrAccessChain
            )
        })
    {
        return (false, false);
    }
    let pointer_storage = module
        .types_global_values
        .iter()
        .filter_map(|inst| {
            if inst.class.opcode != Op::TypePointer {
                return None;
            }
            let Operand::StorageClass(storage) = inst.operands.first()? else {
                return None;
            };
            Some((inst.result_id?, *storage))
        })
        .collect::<HashMap<_, _>>();
    let mut has_storage_buffer_pointer_merge = false;
    let mut has_other_pointer_merge = false;
    for inst in module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
    {
        if !matches!(
            inst.class.opcode,
            Op::Phi | Op::Select | Op::PtrAccessChain | Op::InBoundsPtrAccessChain
        ) {
            continue;
        }
        let Some(result_type) = inst.result_type else {
            continue;
        };
        let Some(&storage) = pointer_storage.get(&result_type) else {
            continue;
        };
        if storage == StorageClass::StorageBuffer {
            has_storage_buffer_pointer_merge = true;
        } else {
            has_other_pointer_merge = true;
        }
    }
    (has_storage_buffer_pointer_merge, has_other_pointer_merge)
}

fn collect_defs(module: &Module) -> HashMap<Word, Instruction> {
    let needed_values = module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter(|inst| {
            matches!(
                inst.class.opcode,
                Op::AccessChain
                    | Op::InBoundsAccessChain
                    | Op::PtrAccessChain
                    | Op::InBoundsPtrAccessChain
            ) || atomic_scope_semantics_indices(inst).is_some()
        })
        .flat_map(|inst| &inst.operands)
        .filter_map(operand_id)
        .collect::<HashSet<_>>();
    let mut definitions = module
        .types_global_values
        .iter()
        .filter_map(|inst| inst.result_id.map(|id| (id, inst.clone())))
        .collect::<HashMap<_, _>>();
    for function in &module.functions {
        for parameter in &function.parameters {
            if let Some(id) = parameter.result_id.filter(|id| needed_values.contains(id)) {
                definitions.insert(
                    id,
                    Instruction::new(
                        parameter.class.opcode,
                        parameter.result_type,
                        Some(id),
                        Vec::new(),
                    ),
                );
            }
        }
        for inst in function.blocks.iter().flat_map(|block| &block.instructions) {
            if let Some(id) = inst.result_id.filter(|id| needed_values.contains(id)) {
                definitions.insert(
                    id,
                    Instruction::new(inst.class.opcode, inst.result_type, Some(id), Vec::new()),
                );
            }
        }
    }
    definitions
}

fn collect_access_chain_defs(
    module: &Module,
    defs: &HashMap<Word, Instruction>,
    zero_constants: &HashSet<Word>,
) -> HashMap<Word, AccessChainDef> {
    module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .filter_map(|inst| {
            if !matches!(inst.class.opcode, Op::AccessChain | Op::InBoundsAccessChain) {
                return None;
            }
            let result_id = inst.result_id?;
            let result_type = inst.result_type?;
            if ptr_info(defs, result_type)?.0 != StorageClass::StorageBuffer {
                return None;
            }
            let Operand::IdRef(root) = inst.operands.first()? else {
                return None;
            };
            let indices = id_ref_operands(&inst.operands[1..])?;
            let (root_storage, root_pointee) = value_pointer_info(defs, *root)?;
            if root_storage != StorageClass::StorageBuffer {
                return None;
            }
            let (parent_before_last, selected_pointee) =
                walk_access_chain_with_parent(defs, root_pointee, &indices, zero_constants)?;
            if ptr_info(defs, result_type)?.1 != selected_pointee {
                return None;
            }
            Some((
                result_id,
                AccessChainDef {
                    root: *root,
                    indices,
                    parent_before_last,
                },
            ))
        })
        .collect()
}

fn id_ref_operands(operands: &[Operand]) -> Option<Vec<Word>> {
    operands
        .iter()
        .map(|operand| match operand {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        })
        .collect()
}

fn value_pointer_info(
    defs: &HashMap<Word, Instruction>,
    value: Word,
) -> Option<(StorageClass, Word)> {
    let ptr_ty = defs.get(&value)?.result_type?;
    ptr_info(defs, ptr_ty)
}

fn ptr_info(defs: &HashMap<Word, Instruction>, ptr_ty: Word) -> Option<(StorageClass, Word)> {
    let inst = defs.get(&ptr_ty)?;
    if inst.class.opcode != Op::TypePointer {
        return None;
    }
    match (inst.operands.first()?, inst.operands.get(1)?) {
        (Operand::StorageClass(storage), Operand::IdRef(pointee)) => Some((*storage, *pointee)),
        _ => None,
    }
}

fn zero_integer_constants(defs: &HashMap<Word, Instruction>) -> HashSet<Word> {
    defs.iter()
        .filter_map(|(&id, inst)| is_zero_integer_constant(defs, inst).then_some(id))
        .collect()
}

fn is_zero_integer_constant(defs: &HashMap<Word, Instruction>, inst: &Instruction) -> bool {
    if inst
        .result_type
        .and_then(|ty| defs.get(&ty))
        .is_none_or(|ty| ty.class.opcode != Op::TypeInt)
    {
        return false;
    }
    matches!(
        (inst.class.opcode, inst.operands.as_slice()),
        (Op::ConstantNull, [])
            | (Op::Constant, [Operand::LiteralBit32(0)])
            | (Op::Constant, [Operand::LiteralBit64(0)])
    )
}

fn integer_value_type(defs: &HashMap<Word, Instruction>, id: Word) -> Option<Word> {
    let ty = defs.get(&id)?.result_type?;
    (defs.get(&ty)?.class.opcode == Op::TypeInt).then_some(ty)
}

fn constant_u32(defs: &HashMap<Word, Instruction>, id: Word) -> Option<u32> {
    let inst = defs.get(&id)?;
    if inst
        .result_type
        .and_then(|ty| defs.get(&ty))
        .is_none_or(|ty| ty.class.opcode != Op::TypeInt)
    {
        return None;
    }
    match (inst.class.opcode, inst.operands.as_slice()) {
        (Op::ConstantNull, []) => Some(0),
        (Op::Constant, [Operand::LiteralBit32(value)]) => Some(*value),
        (Op::Constant, [Operand::LiteralBit64(value)]) => u32::try_from(*value).ok(),
        _ => None,
    }
}

fn is_array_indexable_type(defs: &HashMap<Word, Instruction>, ty: Word) -> bool {
    defs.get(&ty)
        .is_some_and(|inst| matches!(inst.class.opcode, Op::TypeArray | Op::TypeRuntimeArray))
}

fn walk_access_chain_pointee(
    defs: &HashMap<Word, Instruction>,
    root_pointee: Word,
    indices: &[Word],
) -> Option<Word> {
    walk_access_chain_with_parent(defs, root_pointee, indices, &HashSet::new())
        .map(|(_, pointee)| pointee)
}

fn walk_access_chain_with_parent(
    defs: &HashMap<Word, Instruction>,
    root_pointee: Word,
    indices: &[Word],
    zero_constants: &HashSet<Word>,
) -> Option<(Option<Word>, Word)> {
    let mut cur = root_pointee;
    let mut parent_before_last = None;
    for (index_pos, &index_id) in indices.iter().enumerate() {
        if index_pos + 1 == indices.len() {
            parent_before_last = Some(cur);
        }
        cur = walk_member(defs, cur, index_id, zero_constants)?;
    }
    Some((parent_before_last, cur))
}

fn walk_member(
    defs: &HashMap<Word, Instruction>,
    aggregate: Word,
    index: Word,
    zero_constants: &HashSet<Word>,
) -> Option<Word> {
    let inst = defs.get(&aggregate)?;
    match inst.class.opcode {
        Op::TypeVector | Op::TypeMatrix | Op::TypeArray | Op::TypeRuntimeArray => {
            match inst.operands.first()? {
                Operand::IdRef(elem) => Some(*elem),
                _ => None,
            }
        }
        Op::TypeStruct => {
            let member_idx = if zero_constants.contains(&index) {
                0
            } else {
                constant_u32(defs, index)? as usize
            };
            match inst.operands.get(member_idx)? {
                Operand::IdRef(member) => Some(*member),
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::{Block, Function, ModuleHeader};

    fn inst(
        op: Op,
        result_type: Option<Word>,
        result_id: Option<Word>,
        operands: Vec<Operand>,
    ) -> Instruction {
        Instruction::new(op, result_type, result_id, operands)
    }

    #[test]
    fn zero_stride_storage_buffer_ptr_access_chain_demotes_to_access_chain() {
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(20));
        module.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(2),
                vec![Operand::LiteralBit32(0)],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(3),
                vec![Operand::LiteralBit32(1)],
            ),
            inst(Op::TypeRuntimeArray, None, Some(4), vec![Operand::IdRef(1)]),
            inst(Op::TypeStruct, None, Some(5), vec![Operand::IdRef(4)]),
            inst(
                Op::TypePointer,
                None,
                Some(6),
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(5),
                ],
            ),
            inst(
                Op::Variable,
                Some(6),
                Some(7),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(8),
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(1),
                ],
            ),
        ];
        let mut function = Function::new();
        let mut block = Block::new();
        block.instructions.push(inst(
            Op::PtrAccessChain,
            Some(8),
            Some(9),
            vec![Operand::IdRef(7), Operand::IdRef(2), Operand::IdRef(3)],
        ));
        function.blocks.push(block);
        module.functions.push(function);

        assert_eq!(
            lower_zero_base_storage_buffer_ptr_access_chains(&mut module),
            1
        );
        assert_eq!(
            module.functions[0].blocks[0].instructions[0].class.opcode,
            Op::AccessChain
        );
    }

    #[test]
    fn storage_buffer_atomic_workgroup_scope_rewrites_to_device_relaxed() {
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(20));
        module.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(2),
                vec![Operand::LiteralBit32(0)],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(3),
                vec![Operand::LiteralBit32(1)],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(4),
                vec![Operand::LiteralBit32(2)],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(5),
                vec![Operand::LiteralBit32(264)],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(6),
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(1),
                ],
            ),
            inst(
                Op::Variable,
                Some(6),
                Some(7),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
        ];
        let mut function = Function::new();
        let mut block = Block::new();
        block.instructions.push(inst(
            Op::AtomicOr,
            Some(1),
            Some(8),
            vec![
                Operand::IdRef(7),
                Operand::IdScope(4),
                Operand::IdMemorySemantics(5),
                Operand::IdRef(3),
            ],
        ));
        function.blocks.push(block);
        module.functions.push(function);

        assert_eq!(rewrite_storage_buffer_atomic_scopes(&mut module), 2);
        let operands = &module.functions[0].blocks[0].instructions[0].operands;
        assert_eq!(operands[1], Operand::IdScope(3));
        assert_eq!(operands[2], Operand::IdMemorySemantics(2));
    }
}
