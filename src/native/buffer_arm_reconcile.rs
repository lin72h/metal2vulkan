//! Whole-buffer scalar-arm reconciliation for a cross-binding pointer merge.
//!
//! An FC-multiplexed kernel (e.g. the MPS Winograd weight transform) accesses a device buffer at
//! CONFLICTING element types across its function-constant-dead template variants (a `device float*`
//! view in the live variant, `half*`/`bfloat*`/`uchar*` views in the dead ones). The emitter cannot
//! pick one pointee, so it models the buffer as a raw `{ RuntimeArray<uchar> }` block and forms the
//! whole-buffer base as an `OpAccessChain %_ptr_StorageBuffer_uchar %buf %uint_0 %uint_0` (byte 0).
//! After the FC dead arms are pruned, that byte-0 base is the FALLBACK arm of `OpSelect`/`OpPhi`
//! merges whose result type is `%_ptr_StorageBuffer_float` and whose loads are `OpLoad %float` — so
//! spirv-val rejects the merge: *"Expected both objects to be of Result Type"* (the float arm vs the
//! uchar arm).
//!
//! This pass reconciles the mismatch: a StorageBuffer variable whose block is `{ RuntimeArray<E> }`,
//! used ONLY through byte-0 whole-buffer base chains `[buf, 0, 0]` that feed ONLY pointer merges of a
//! single DIFFERENT scalar pointee `T`, is retyped so its block is `{ RuntimeArray<T> }` and its base
//! chains yield `%_ptr_StorageBuffer_T`. Byte-EXACT by construction: only element 0 (offset 0) is ever
//! accessed, and `E`-element-0 and `T`-element-0 both start at byte 0. After this, the merge arms
//! share the pointee type `T`; the remaining wall is that the two arms point into DISTINCT bindings —
//! a cross-binding merge the PSB pass ([`super::psb`]) then lowers to PhysicalStorageBuffer64.
//!
//! Applied only in `lib.rs`'s adopt-if-VALIDATES `fc_promote_psb` retry (feeding PSB), so floor-safe
//! by construction. Decides purely from IR structure (storage class, block shape, the byte-0 base
//! chain, the uniform merge pointee) — never a shader name. New types are synthesized fresh; no
//! existing (possibly-shared) `RuntimeArray`/struct/pointer type is mutated in place.

use crate::spirv_module::Instruction;
use crate::spirv_module::Module;
use crate::spirv_module::Operand;
use spirv::{Decoration, Op, StorageClass, Word};
use std::collections::HashMap;

/// A buffer to retype: its variable, byte-0 base chains, target scalar pointee `T`, and `T`'s size in
/// bytes (precomputed during discovery so the rewrite stage needs no type lookups).
struct Retype {
    var: Word,
    base_chains: Vec<Word>,
    target: Word, // pointee scalar T
    target_bytes: u32,
}

/// The two synthesized pointer types a plan's rewrite installs: the struct pointer for the variable
/// and the `T` pointer for its byte-0 base chains.
struct PlanTypes {
    ptr_struct: Word,
    ptr_t: Word,
}

/// Reconcile whole-buffer scalar fallback arms as described above. Returns true if any buffer was
/// retyped. Three stages: discover the retype plans (pure), then for each plan synthesize its fresh
/// types and repoint the module.
pub(super) fn reconcile_whole_buffer_scalar_arms(module: &mut Module) -> bool {
    let plans = discover_reconcile_plans(module);
    if plans.is_empty() {
        return false;
    }

    let mut next_id = module.header.as_ref().map(|h| h.bound).unwrap_or(1);
    for plan in &plans {
        let types = synthesize_plan_types(module, plan, &mut next_id);
        repoint_plan(module, plan, &types);
    }

    // The new struct/pointer types are appended at the END of `types_global_values`, but the repointed
    // `OpVariable`s appear earlier — an illegal forward type reference. A module-scope type/constant
    // never depends on an `OpVariable`, so stably moving every `OpVariable` to the end (relative order
    // preserved) guarantees each variable's (pointer) type precedes it, with no other ordering change.
    module
        .types_global_values
        .sort_by_key(|i| i.class.opcode == Op::Variable);

    if let Some(h) = module.header.as_mut() {
        h.bound = next_id;
    }
    true
}

/// Discovery + validation (pure): every StorageBuffer variable safe to retype and the scalar pointee
/// to retype it to, per the structural criteria in the module doc. No mutation.
fn discover_reconcile_plans(module: &Module) -> Vec<Retype> {
    let type_defs: HashMap<Word, Instruction> = module
        .types_global_values
        .iter()
        .filter_map(|i| i.result_id.map(|id| (id, i.clone())))
        .collect();

    // (storage class, pointee) of a pointer type.
    let ptr_info = |ty: Word| -> Option<(StorageClass, Word)> {
        let inst = type_defs.get(&ty)?;
        if inst.class.opcode != Op::TypePointer {
            return None;
        }
        match (inst.operands.first()?, inst.operands.get(1)?) {
            (Operand::StorageClass(s), Operand::IdRef(p)) => Some((*s, *p)),
            _ => None,
        }
    };
    // The single runtimearray-element type of a `{ RuntimeArray<E> }` struct pointee, else None.
    let single_runtime_array_elem = |struct_ty: Word| -> Option<Word> {
        let s = type_defs.get(&struct_ty)?;
        if s.class.opcode != Op::TypeStruct || s.operands.len() != 1 {
            return None;
        }
        let Operand::IdRef(member) = s.operands.first()? else {
            return None;
        };
        let ra = type_defs.get(member)?;
        if ra.class.opcode != Op::TypeRuntimeArray {
            return None;
        }
        match ra.operands.first()? {
            Operand::IdRef(elem) => Some(*elem),
            _ => None,
        }
    };
    let is_scalar = |ty: Word| -> bool {
        type_defs
            .get(&ty)
            .is_some_and(|i| matches!(i.class.opcode, Op::TypeInt | Op::TypeFloat))
    };
    let scalar_bytes = |ty: Word| -> Option<u32> {
        match type_defs.get(&ty)?.operands.first()? {
            Operand::LiteralBit32(bits) => Some(bits / 8),
            _ => None,
        }
    };

    // Constant-zero uint ids (for the byte-0 base chain `[buf, 0, 0]`).
    let zero_consts: std::collections::HashSet<Word> = module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::Constant)
        .filter_map(|i| match i.operands.first()? {
            Operand::LiteralBit32(0) => i.result_id,
            _ => None,
        })
        .collect();

    // Result-type of every merge (OpSelect/OpPhi) result id, and the merge's pointee if a pointer.
    // Also: for each value id used as a merge arm, the merge result pointee (for uniformity check).
    let mut merge_arm_pointee: HashMap<Word, Vec<Word>> = HashMap::new();
    for f in &module.functions {
        for b in &f.blocks {
            for inst in &b.instructions {
                if !matches!(inst.class.opcode, Op::Select | Op::Phi) {
                    continue;
                }
                let Some(rt) = inst.result_type else { continue };
                let Some((_, pointee)) = ptr_info(rt) else {
                    continue;
                };
                // Operand ids that are pointer arms: for Select they are operands 1,2; for Phi the
                // even-indexed operands (value, parent, value, parent, …). Record the merge pointee
                // against each arm id.
                match inst.class.opcode {
                    Op::Select => {
                        for op in inst.operands.iter().skip(1) {
                            if let Operand::IdRef(a) = op {
                                merge_arm_pointee.entry(*a).or_default().push(pointee);
                            }
                        }
                    }
                    Op::Phi => {
                        for (k, op) in inst.operands.iter().enumerate() {
                            if k % 2 == 0 {
                                if let Operand::IdRef(a) = op {
                                    merge_arm_pointee.entry(*a).or_default().push(pointee);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Every operand USE of a value id across all function bodies (to prove a base chain is used ONLY as
    // a merge arm, and a buffer var ONLY as byte-0 base chains).
    let mut all_uses: HashMap<Word, usize> = HashMap::new();
    let mut base_chains_of: HashMap<Word, Vec<Word>> = HashMap::new(); // buffer var -> its base chain ids
    for f in &module.functions {
        for b in &f.blocks {
            for inst in &b.instructions {
                for op in &inst.operands {
                    if let Operand::IdRef(id) = op {
                        *all_uses.entry(*id).or_default() += 1;
                    }
                }
                if matches!(inst.class.opcode, Op::AccessChain | Op::InBoundsAccessChain) {
                    // base chain shape: [buf, zero, zero]
                    if inst.operands.len() == 3 {
                        if let (
                            Some(Operand::IdRef(buf)),
                            Some(Operand::IdRef(i0)),
                            Some(Operand::IdRef(i1)),
                        ) = (
                            inst.operands.first(),
                            inst.operands.get(1),
                            inst.operands.get(2),
                        ) {
                            if zero_consts.contains(i0) && zero_consts.contains(i1) {
                                if let Some(rid) = inst.result_id {
                                    base_chains_of.entry(*buf).or_default().push(rid);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Candidate buffers: module-scope StorageBuffer variables with a `{ RuntimeArray<scalar E> }`
    // pointee, whose EVERY function-body use is a byte-0 base chain, and whose base chains feed ONLY
    // pointer merges of a single scalar pointee T != E.
    let mut plans: Vec<Retype> = vec![];
    for inst in &module.types_global_values {
        if inst.class.opcode != Op::Variable {
            continue;
        }
        let Some(var) = inst.result_id else { continue };
        let Some(var_ty) = inst.result_type else {
            continue;
        };
        let Some((StorageClass::StorageBuffer, struct_ty)) = ptr_info(var_ty) else {
            continue;
        };
        let Some(elem) = single_runtime_array_elem(struct_ty) else {
            continue;
        };
        if !is_scalar(elem) {
            continue;
        }
        let Some(base_chains) = base_chains_of.get(&var) else {
            continue;
        };
        if base_chains.is_empty() {
            continue;
        }
        // Every function-body use of the var must be one of its byte-0 base chains (one operand use per
        // chain access-chain instruction).
        let var_uses = *all_uses.get(&var).unwrap_or(&0);
        if var_uses != base_chains.len() {
            continue;
        }
        // Every base chain must be used ONLY as merge arms, and all those merges must share ONE scalar
        // pointee T != E.
        let mut target: Option<Word> = None;
        let mut ok = true;
        for &bc in base_chains {
            let uses = *all_uses.get(&bc).unwrap_or(&0);
            let Some(pointees) = merge_arm_pointee.get(&bc) else {
                ok = false;
                break;
            };
            // chain used somewhere other than as a merge arm → bail
            if uses != pointees.len() {
                ok = false;
                break;
            }
            for &p in pointees {
                if p == elem || !is_scalar(p) {
                    ok = false;
                    break;
                }
                match target {
                    None => target = Some(p),
                    Some(t) if t == p => {}
                    Some(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                break;
            }
        }
        let (Some(target), true) = (target, ok) else {
            continue;
        };
        // Both E and T must be sized scalars (they are, checked is_scalar) and the base chain's declared
        // result pointee must be E (sanity).
        if scalar_bytes(target).is_none() || scalar_bytes(elem).is_none() {
            continue;
        }
        plans.push(Retype {
            var,
            base_chains: base_chains.clone(),
            target,
            // `scalar_bytes(target)` is Some here (checked just above); precompute the stride so the
            // rewrite stage carries no type lookups.
            target_bytes: scalar_bytes(target).unwrap(),
        });
    }

    plans
}

/// Find an existing `%_ptr_StorageBuffer_<pointee>` type, if the module already declares one.
fn find_storage_buffer_ptr(module: &Module, pointee: Word) -> Option<Word> {
    module.types_global_values.iter().find_map(|i| {
        if i.class.opcode == Op::TypePointer
            && i.operands.first() == Some(&Operand::StorageClass(StorageClass::StorageBuffer))
            && i.operands.get(1) == Some(&Operand::IdRef(pointee))
        {
            i.result_id
        } else {
            None
        }
    })
}

/// Type synthesis (mutating): append the `RuntimeArray<T>` (reused iff it already carries the matching
/// ArrayStride), the fresh `{ RuntimeArray<T> }` Block struct, and the two pointer types this plan
/// installs. `next_id` threads the module's fresh-id counter across plans.
fn synthesize_plan_types(module: &mut Module, plan: &Retype, next_id: &mut u32) -> PlanTypes {
    let t = plan.target;
    let stride = plan.target_bytes;
    let mut fresh = || {
        let id = *next_id;
        *next_id += 1;
        id
    };

    // RuntimeArray<T> with ArrayStride = size(T). Reuse an existing one iff it already carries the
    // matching ArrayStride; else synthesize fresh (never mutate a possibly-shared existing RA).
    let ra_id = module
        .types_global_values
        .iter()
        .find_map(|i| {
            if i.class.opcode == Op::TypeRuntimeArray
                && i.operands.first() == Some(&Operand::IdRef(t))
            {
                let rid = i.result_id?;
                let has_stride = module.annotations.iter().any(|a| {
                    a.class.opcode == Op::Decorate
                        && a.operands.first() == Some(&Operand::IdRef(rid))
                        && a.operands.get(1) == Some(&Operand::Decoration(Decoration::ArrayStride))
                        && a.operands.get(2) == Some(&Operand::LiteralBit32(stride))
                });
                if has_stride {
                    Some(rid)
                } else {
                    None
                }
            } else {
                None
            }
        })
        .unwrap_or_else(|| {
            let id = fresh();
            module.types_global_values.push(Instruction::new(
                Op::TypeRuntimeArray,
                None,
                Some(id),
                vec![Operand::IdRef(t)],
            ));
            module.annotations.push(Instruction::new(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(id),
                    Operand::Decoration(Decoration::ArrayStride),
                    Operand::LiteralBit32(stride),
                ],
            ));
            id
        });

    // Fresh `{ RuntimeArray<T> }` Block struct (member 0 at Offset 0).
    let struct_id = fresh();
    module.types_global_values.push(Instruction::new(
        Op::TypeStruct,
        None,
        Some(struct_id),
        vec![Operand::IdRef(ra_id)],
    ));
    module.annotations.push(Instruction::new(
        Op::MemberDecorate,
        None,
        None,
        vec![
            Operand::IdRef(struct_id),
            Operand::LiteralBit32(0),
            Operand::Decoration(Decoration::Offset),
            Operand::LiteralBit32(0),
        ],
    ));
    module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(struct_id),
            Operand::Decoration(Decoration::Block),
        ],
    ));

    // Pointer to the new struct (for the variable), and pointer to T (for the base chains).
    let ptr_struct = fresh();
    module.types_global_values.push(Instruction::new(
        Op::TypePointer,
        None,
        Some(ptr_struct),
        vec![
            Operand::StorageClass(StorageClass::StorageBuffer),
            Operand::IdRef(struct_id),
        ],
    ));
    let ptr_t = find_storage_buffer_ptr(module, t).unwrap_or_else(|| {
        let id = fresh();
        module.types_global_values.push(Instruction::new(
            Op::TypePointer,
            None,
            Some(id),
            vec![
                Operand::StorageClass(StorageClass::StorageBuffer),
                Operand::IdRef(t),
            ],
        ));
        id
    });

    PlanTypes { ptr_struct, ptr_t }
}

/// Rewrite (mutating): repoint the buffer variable to the new struct pointer and each byte-0 base
/// chain to the new `T` pointer.
fn repoint_plan(module: &mut Module, plan: &Retype, types: &PlanTypes) {
    for inst in &mut module.types_global_values {
        if inst.result_id == Some(plan.var) {
            inst.result_type = Some(types.ptr_struct);
        }
    }
    for f in &mut module.functions {
        for b in &mut f.blocks {
            for inst in &mut b.instructions {
                if let Some(rid) = inst.result_id {
                    if plan.base_chains.contains(&rid) {
                        inst.result_type = Some(types.ptr_t);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::{Block, Function, ModuleHeader};

    fn ti(op: Op, rt: Option<Word>, rid: Option<Word>, ops: Vec<Operand>) -> Instruction {
        Instruction::new(op, rt, rid, ops)
    }

    /// A StorageBuffer buffer whose block is `{ RuntimeArray<uchar> }`, used only through a byte-0
    /// whole-buffer base chain that feeds an `OpSelect` of pointee `float`, is retyped to a
    /// `{ RuntimeArray<float> }` block with a `float*` base chain — the merge arms then share `float*`.
    #[test]
    fn reconciles_byte0_uchar_arm_to_float_merge_pointee() {
        // ids: 1 uint, 2 float, 3 uchar, 4 uint_0, 5 ra_uchar, 6 struct_v, 7 ptr_sb_struct_v,
        // 8 ra_float, 9 struct_w, 10 ptr_sb_struct_w, 11 ptr_sb_uchar, 12 ptr_sb_float, 13 bool,
        // 14 true, 15 V, 16 W, 20 base_v, 21 elem_w, 22 sel, 23 val
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(30));
        let sb = || Operand::StorageClass(StorageClass::StorageBuffer);
        m.types_global_values = vec![
            ti(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            ti(
                Op::TypeFloat,
                None,
                Some(2),
                vec![Operand::LiteralBit32(32)],
            ),
            ti(
                Op::TypeInt,
                None,
                Some(3),
                vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
            ),
            ti(
                Op::Constant,
                Some(1),
                Some(4),
                vec![Operand::LiteralBit32(0)],
            ),
            ti(Op::TypeRuntimeArray, None, Some(5), vec![Operand::IdRef(3)]),
            ti(Op::TypeStruct, None, Some(6), vec![Operand::IdRef(5)]),
            ti(
                Op::TypePointer,
                None,
                Some(7),
                vec![sb(), Operand::IdRef(6)],
            ),
            ti(Op::TypeRuntimeArray, None, Some(8), vec![Operand::IdRef(2)]),
            ti(Op::TypeStruct, None, Some(9), vec![Operand::IdRef(8)]),
            ti(
                Op::TypePointer,
                None,
                Some(10),
                vec![sb(), Operand::IdRef(9)],
            ),
            ti(
                Op::TypePointer,
                None,
                Some(11),
                vec![sb(), Operand::IdRef(3)],
            ),
            ti(
                Op::TypePointer,
                None,
                Some(12),
                vec![sb(), Operand::IdRef(2)],
            ),
            ti(Op::TypeBool, None, Some(13), vec![]),
            ti(Op::ConstantTrue, Some(13), Some(14), vec![]),
            ti(Op::Variable, Some(7), Some(15), vec![sb()]),
            ti(Op::Variable, Some(10), Some(16), vec![sb()]),
        ];
        // ArrayStride 4 on the existing float runtimearray, so reconcile reuses it.
        m.annotations = vec![ti(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(8),
                Operand::Decoration(Decoration::ArrayStride),
                Operand::LiteralBit32(4),
            ],
        )];
        let mut block = Block::new();
        block.label = Some(ti(Op::Label, None, Some(19), vec![]));
        block.instructions = vec![
            ti(
                Op::AccessChain,
                Some(11),
                Some(20),
                vec![Operand::IdRef(15), Operand::IdRef(4), Operand::IdRef(4)],
            ),
            ti(
                Op::AccessChain,
                Some(12),
                Some(21),
                vec![Operand::IdRef(16), Operand::IdRef(4), Operand::IdRef(4)],
            ),
            ti(
                Op::Select,
                Some(12),
                Some(22),
                vec![Operand::IdRef(14), Operand::IdRef(21), Operand::IdRef(20)],
            ),
            ti(Op::Load, Some(2), Some(23), vec![Operand::IdRef(22)]),
            ti(Op::Return, None, None, vec![]),
        ];
        let mut f = Function::new();
        f.blocks = vec![block];
        m.functions = vec![f];

        assert!(reconcile_whole_buffer_scalar_arms(&mut m));

        // V (%15) now points at a struct whose single runtimearray element is float (%2).
        let var_ty = m
            .types_global_values
            .iter()
            .find(|i| i.result_id == Some(15))
            .and_then(|i| i.result_type)
            .expect("V typed");
        let elem = single_runtime_array_elem_in(&m, ptr_pointee_in(&m, var_ty).expect("ptr"))
            .expect("runtime array");
        assert_eq!(elem, 2, "V block runtimearray must be retyped to float");
        // The base chain (%20) now yields `float*` (%12), matching the merge's `float*` arm.
        let base_ty = m.functions[0].blocks[0]
            .instructions
            .iter()
            .find(|i| i.result_id == Some(20))
            .and_then(|i| i.result_type)
            .expect("base chain typed");
        assert_eq!(base_ty, 12, "byte-0 base chain must be retyped to float*");
    }

    fn ptr_pointee_in(m: &Module, ty: Word) -> Option<Word> {
        m.types_global_values.iter().find_map(|i| {
            if i.result_id == Some(ty) && i.class.opcode == Op::TypePointer {
                match i.operands.get(1) {
                    Some(Operand::IdRef(p)) => Some(*p),
                    _ => None,
                }
            } else {
                None
            }
        })
    }
    fn single_runtime_array_elem_in(m: &Module, struct_ty: Word) -> Option<Word> {
        let s = m
            .types_global_values
            .iter()
            .find(|i| i.result_id == Some(struct_ty))?;
        let Operand::IdRef(member) = s.operands.first()? else {
            return None;
        };
        let ra = m
            .types_global_values
            .iter()
            .find(|i| i.result_id == Some(*member))?;
        match ra.operands.first()? {
            Operand::IdRef(elem) => Some(*elem),
            _ => None,
        }
    }
}
