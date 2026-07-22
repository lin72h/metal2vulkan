//! In-memory Function sub-word packed-scalar reinterpret remodel.
//!
//! Metal's `as_type<uint>(half2(a, b))` pack idiom (and its `uchar4`/`ushort2` siblings) compiles to a
//! scalar `i32` alloca whose bytes are written through SMALLER-element geps — e.g.
//! `getelementptr inbounds half, ptr %slot, i64 0/1` + `store half` — and then read back whole as
//! `load i32`. The native emitter declares the alloca's `OpVariable` from its `i32` pointee but takes the
//! access-chain result pointer type straight from the gep element type, emitting
//! `OpInBoundsAccessChain %_ptr_Function_half %slot %uint_0` — indexing a SCALAR variable, which
//! spirv-val rejects: *"reached non-composite type while indexes still remain to be traversed"*. The
//! existing same-size alloca reinterpret (`ir::infer_local_alloca_pointees`) does not cover a sub-word
//! element access (the element is narrower than the alloca scalar), so the module ships invalid.
//!
//! The honest fix is to RETYPE the variable's pointee from the scalar integer `T` (width `W`) to a vector
//! `<N x E>` of the sub-word element `E` (width `w`, `N = W / w`): the sub-element access chains then
//! index a vector component (legal in Function storage) unchanged, and the whole-word `OpLoad %T` /
//! `OpStore` reinterpret via a VALUE `OpBitcast` between `<N x E>` and `T` (a legal numeric reinterpret,
//! both `W` bits wide).
//!
//! This is byte-SAFE by construction: a Function alloca is per-invocation scratch (never device-visible,
//! never the golden's output), and `<N x E>` shares `T`'s little-endian byte layout exactly — component 0
//! occupies the lowest-address `w` bits, identical to gep element index 0, so every store/load lands on
//! the same bytes the packed scalar did. The retry clones the canonical pre-primary `Module`, mutates
//! it in place, and adopts it only if it validates, so it is floor-safe by construction. It decides
//! purely from IR structure (a Function scalar-integer variable whose only access chains use one
//! smaller scalar element type) — never a shader name.

use crate::spirv_module::Instruction;
use crate::spirv_module::Module;
use crate::spirv_module::Operand;
use spirv::{Op, StorageClass, Word};
use std::collections::HashMap;

/// Retype every Function scalar-integer variable that is accessed ONLY as the sub-word packed-scalar
/// idiom so its pointee becomes a `<N x E>` vector of the sub-word element, dropping the illegal
/// scalar-indexing access chains' invalidity and value-bitcasting its whole-word loads/stores. Returns
/// true if any variable was remodeled.
pub(super) fn rewrite_subword_packed_scalars(module: &mut Module) -> bool {
    // id -> (storage class, pointee) for every pointer type; id -> type def for every type.
    let mut ptr_info: HashMap<Word, (StorageClass, Word)> = HashMap::new();
    let mut type_defs: HashMap<Word, Instruction> = HashMap::new();
    for inst in &module.types_global_values {
        if let Some(result) = inst.result_id {
            type_defs.insert(result, inst.clone());
        }
        if inst.class.opcode == Op::TypePointer {
            if let (Some(id), Some(Operand::StorageClass(s)), Some(Operand::IdRef(p))) =
                (inst.result_id, inst.operands.first(), inst.operands.get(1))
            {
                ptr_info.insert(id, (*s, *p));
            }
        }
    }

    // Candidate Function variables whose pointee is a scalar integer type (the packed whole word).
    // Function-storage `OpVariable`s live in the function's entry block, NOT the module-scope
    // types_global_values (only Private/Workgroup/UniformConstant vars are module scope).
    let mut cands: Vec<(Word, Word, u32)> = Vec::new(); // (var, scalar_int_ty, width)
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                if inst.class.opcode != Op::Variable {
                    continue;
                }
                // An initializer (operand beyond the storage class) would not survive the retype — skip.
                if inst.operands.len() != 1
                    || inst.operands.first() != Some(&Operand::StorageClass(StorageClass::Function))
                {
                    continue;
                }
                let (Some(var), Some(ptr_ty)) = (inst.result_id, inst.result_type) else {
                    continue;
                };
                let Some(&(StorageClass::Function, pointee)) = ptr_info.get(&ptr_ty) else {
                    continue;
                };
                if let Some(width) = scalar_int_width(&type_defs, pointee) {
                    cands.push((var, pointee, width));
                }
            }
        }
    }
    if cands.is_empty() {
        return false;
    }

    let mut next_id = module.header.as_ref().map(|h| h.bound).unwrap_or(0);
    let mut changed = false;
    for (var, scalar_ty, width) in cands {
        if remodel_one(module, var, scalar_ty, width, &mut next_id) {
            changed = true;
        }
    }
    if changed {
        if let Some(header) = module.header.as_mut() {
            header.bound = next_id;
        }
    }
    changed
}

/// Width in bits of a scalar integer type id, or None if `ty` is not an `OpTypeInt`.
fn scalar_int_width(defs: &HashMap<Word, Instruction>, ty: Word) -> Option<u32> {
    let def = defs.get(&ty)?;
    if def.class.opcode != Op::TypeInt {
        return None;
    }
    match def.operands.first()? {
        Operand::LiteralBit32(bits) => Some(*bits),
        _ => None,
    }
}

/// Width in bits of a scalar numeric (int or float) type id, or None otherwise.
fn scalar_numeric_width(defs: &HashMap<Word, Instruction>, ty: Word) -> Option<u32> {
    let def = defs.get(&ty)?;
    match def.class.opcode {
        Op::TypeInt | Op::TypeFloat => match def.operands.first()? {
            Operand::LiteralBit32(bits) => Some(*bits),
            _ => None,
        },
        _ => None,
    }
}

struct Plan {
    /// The sub-word element type `E` every access chain agrees on.
    elem_ty: Word,
    /// Number of `E` lanes packed in the scalar word (`W / w`).
    lanes: u32,
    /// Whole-word `OpLoad %T %var` result ids (rewritten: load the vector, value-bitcast to `T`).
    word_loads: Vec<Word>,
    /// Whole-word `OpStore %var %val` — pointer is operand 0 (rewritten: bitcast `val` to vec, store).
    /// Identified at rewrite time by `operand 0 == var`; nothing to collect here beyond the var.
    has_word_store: bool,
}

/// Validate `var` (scalar word type `scalar_ty`, width `width`) is reached only as the sub-word pack
/// idiom and, if so, retype it to a `<lanes x elem>` vector in place. Returns true if remodeled. On any
/// unmodeled use the variable is left untouched.
fn remodel_one(
    module: &mut Module,
    var: Word,
    scalar_ty: Word,
    width: u32,
    next_id: &mut Word,
) -> bool {
    let type_defs: HashMap<Word, Instruction> = module
        .types_global_values
        .iter()
        .filter_map(|i| i.result_id.map(|id| (id, i.clone())))
        .collect();
    let ptr_info: HashMap<Word, (StorageClass, Word)> = module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::TypePointer)
        .filter_map(|i| {
            let id = i.result_id?;
            match (i.operands.first()?, i.operands.get(1)?) {
                (Operand::StorageClass(s), Operand::IdRef(p)) => Some((id, (*s, *p))),
                _ => None,
            }
        })
        .collect();

    let Some(plan) = validate(module, var, scalar_ty, width, &type_defs, &ptr_info) else {
        return false;
    };

    let mut fresh = || {
        let id = *next_id;
        *next_id += 1;
        id
    };

    // The `<lanes x elem>` vector type, and a Function pointer to it. Reuse existing defs when present,
    // else synthesize and append to the module-scope type section. Appending at the end is define-
    // before-use safe: the vector references the pre-existing element scalar, the pointer references the
    // vector just before it, and the only user (the function-body variable) follows all types.
    let vec_ty = find_vector(module, plan.elem_ty, plan.lanes).unwrap_or_else(|| {
        let id = fresh();
        module.types_global_values.push(Instruction::new(
            Op::TypeVector,
            None,
            Some(id),
            vec![
                Operand::IdRef(plan.elem_ty),
                Operand::LiteralBit32(plan.lanes),
            ],
        ));
        id
    });
    let vec_ptr = find_ptr(module, StorageClass::Function, vec_ty).unwrap_or_else(|| {
        let id = fresh();
        module.types_global_values.push(Instruction::new(
            Op::TypePointer,
            None,
            Some(id),
            vec![
                Operand::StorageClass(StorageClass::Function),
                Operand::IdRef(vec_ty),
            ],
        ));
        id
    });

    // Repoint the function-body variable instruction to the vector pointer.
    let mut found = false;
    for func in module.functions.iter_mut() {
        for block in func.blocks.iter_mut() {
            for inst in block.instructions.iter_mut() {
                if inst.class.opcode == Op::Variable && inst.result_id == Some(var) {
                    inst.result_type = Some(vec_ptr);
                    found = true;
                }
            }
        }
    }
    if !found {
        return false;
    }

    // Rewrite the function bodies: value-bitcast every whole-word load/store of the variable between the
    // vector view and the scalar word. The sub-element access chains stay as-is (now valid, since they
    // index a vector component instead of a scalar).
    rewrite_bodies(module, var, scalar_ty, vec_ty, &plan, next_id);
    true
}

fn validate(
    module: &Module,
    var: Word,
    scalar_ty: Word,
    width: u32,
    type_defs: &HashMap<Word, Instruction>,
    ptr_info: &HashMap<Word, (StorageClass, Word)>,
) -> Option<Plan> {
    let mut elem_ty: Option<Word> = None;
    let mut word_loads: Vec<Word> = Vec::new();
    let mut has_word_store = false;
    let mut saw_subword_chain = false;

    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                // Examine every use of the variable as an id operand.
                for (oi, op) in inst.operands.iter().enumerate() {
                    if !matches!(op, Operand::IdRef(id) if *id == var) {
                        continue;
                    }
                    match inst.class.opcode {
                        // A sub-word access chain: base (operand 0) is the variable, exactly one index,
                        // result a Function pointer to a smaller scalar element. All such chains must
                        // agree on the same element type.
                        Op::InBoundsAccessChain | Op::AccessChain if oi == 0 => {
                            // Exactly one index operand (a scalar word has no nested structure).
                            if inst.operands.len() != 2 {
                                return None;
                            }
                            let result_ty = inst.result_type?;
                            let &(sc, pointee) = ptr_info.get(&result_ty)?;
                            if sc != StorageClass::Function {
                                return None;
                            }
                            let ew = scalar_numeric_width(type_defs, pointee)?;
                            if ew >= width || !width.is_multiple_of(ew) {
                                return None;
                            }
                            match elem_ty {
                                Some(t) if t != pointee => return None,
                                _ => elem_ty = Some(pointee),
                            }
                            saw_subword_chain = true;
                        }
                        // Whole-word load: the variable is the pointer (operand 0), result is the word.
                        Op::Load if oi == 0 => {
                            if inst.result_type != Some(scalar_ty) {
                                return None;
                            }
                            word_loads.push(inst.result_id?);
                        }
                        // Whole-word store: the variable is the pointer (operand 0).
                        Op::Store if oi == 0 => {
                            has_word_store = true;
                        }
                        // Any other mention of the variable disqualifies it (do not miscompile).
                        _ => return None,
                    }
                }
            }
        }
    }

    let elem_ty = elem_ty?;
    if !saw_subword_chain {
        return None;
    }
    let ew = scalar_numeric_width(type_defs, elem_ty)?;
    let lanes = width / ew;
    // Vulkan vectors are 2..=4 components; a wider pack would need Vector16 — leave it to the
    // adopt-if-validates gate by simply declining here.
    if !(2..=4).contains(&lanes) {
        return None;
    }
    Some(Plan {
        elem_ty,
        lanes,
        word_loads,
        has_word_store,
    })
}

fn rewrite_bodies(
    module: &mut Module,
    var: Word,
    scalar_ty: Word,
    vec_ty: Word,
    plan: &Plan,
    next_id: &mut Word,
) {
    for func in module.functions.iter_mut() {
        for block in func.blocks.iter_mut() {
            let insts = std::mem::take(&mut block.instructions);
            let mut out = Vec::with_capacity(insts.len());
            for inst in insts {
                // Whole-word load: load the vector, then bitcast the value to the scalar word (keeping the
                // original load's result id for downstream uses).
                if inst.class.opcode == Op::Load
                    && inst
                        .result_id
                        .map(|r| plan.word_loads.contains(&r))
                        .unwrap_or(false)
                    && operand_id(&inst, 0) == Some(var)
                {
                    let rid = inst.result_id.unwrap();
                    let tmp = *next_id;
                    *next_id += 1;
                    out.push(Instruction::new(
                        Op::Load,
                        Some(vec_ty),
                        Some(tmp),
                        vec![Operand::IdRef(var)],
                    ));
                    out.push(Instruction::new(
                        Op::Bitcast,
                        Some(scalar_ty),
                        Some(rid),
                        vec![Operand::IdRef(tmp)],
                    ));
                    continue;
                }
                // Whole-word store: bitcast the scalar object to the vector view, then store.
                if plan.has_word_store
                    && inst.class.opcode == Op::Store
                    && operand_id(&inst, 0) == Some(var)
                {
                    let val = operand_id(&inst, 1).unwrap();
                    let tmp = *next_id;
                    *next_id += 1;
                    out.push(Instruction::new(
                        Op::Bitcast,
                        Some(vec_ty),
                        Some(tmp),
                        vec![Operand::IdRef(val)],
                    ));
                    out.push(Instruction::new(
                        Op::Store,
                        None,
                        None,
                        vec![Operand::IdRef(var), Operand::IdRef(tmp)],
                    ));
                    continue;
                }
                out.push(inst);
            }
            block.instructions = out;
        }
    }
}

fn find_vector(module: &Module, component: Word, count: u32) -> Option<Word> {
    module.types_global_values.iter().find_map(|i| {
        (i.class.opcode == Op::TypeVector
            && i.operands.first() == Some(&Operand::IdRef(component))
            && i.operands.get(1) == Some(&Operand::LiteralBit32(count)))
        .then_some(i.result_id)
        .flatten()
    })
}

fn find_ptr(module: &Module, sc: StorageClass, pointee: Word) -> Option<Word> {
    module.types_global_values.iter().find_map(|i| {
        (i.class.opcode == Op::TypePointer
            && i.operands.first() == Some(&Operand::StorageClass(sc))
            && i.operands.get(1) == Some(&Operand::IdRef(pointee)))
        .then_some(i.result_id)
        .flatten()
    })
}

fn operand_id(inst: &Instruction, idx: usize) -> Option<Word> {
    match inst.operands.get(idx) {
        Some(Operand::IdRef(id)) => Some(*id),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::{Block, Function, ModuleHeader};

    fn i(op: Op, ty: Option<Word>, res: Option<Word>, ops: Vec<Operand>) -> Instruction {
        Instruction::new(op, ty, res, ops)
    }

    // A `uint` Function alloca written as two packed `half` lanes (index 0/1) then read whole as `uint`
    // (the `as_type<uint>(half2)` idiom). The remodel must retype the variable to `<2 x half>`, leave the
    // half access chains untouched, and rewrite the whole-word `OpLoad %uint` into `OpLoad %v2half` +
    // `OpBitcast %uint`. Byte-safe (Function scratch, 32-bit reinterpret, same byte offsets).
    #[test]
    fn half2_packed_into_uint_remodels_to_vector() {
        // ids: uint=1 half=2 v2half=3(synth) | uint_0=10 uint_1=11 | ptrF_uint=4 ptrF_half=5
        //      var=20 | entry=30 g0=31 g1=32 hv0=33 hv1=34 load=35
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(40));
        m.types_global_values = vec![
            i(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            i(
                Op::TypeFloat,
                None,
                Some(2),
                vec![Operand::LiteralBit32(16)],
            ),
            i(
                Op::TypePointer,
                None,
                Some(4),
                vec![
                    Operand::StorageClass(StorageClass::Function),
                    Operand::IdRef(1),
                ],
            ),
            i(
                Op::TypePointer,
                None,
                Some(5),
                vec![
                    Operand::StorageClass(StorageClass::Function),
                    Operand::IdRef(2),
                ],
            ),
            i(
                Op::Constant,
                Some(1),
                Some(10),
                vec![Operand::LiteralBit32(0)],
            ),
            i(
                Op::Constant,
                Some(1),
                Some(11),
                vec![Operand::LiteralBit32(1)],
            ),
        ];
        let mut block = Block::new();
        block.label = Some(i(Op::Label, None, Some(30), vec![]));
        block.instructions = vec![
            // The Function-storage variable lives in the function's entry block.
            i(
                Op::Variable,
                Some(4),
                Some(20),
                vec![Operand::StorageClass(StorageClass::Function)],
            ),
            i(
                Op::InBoundsAccessChain,
                Some(5),
                Some(31),
                vec![Operand::IdRef(20), Operand::IdRef(10)],
            ),
            i(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(31), Operand::IdRef(33)],
            ),
            i(
                Op::InBoundsAccessChain,
                Some(5),
                Some(32),
                vec![Operand::IdRef(20), Operand::IdRef(11)],
            ),
            i(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(32), Operand::IdRef(34)],
            ),
            i(Op::Load, Some(1), Some(35), vec![Operand::IdRef(20)]),
            i(Op::Return, None, None, vec![]),
        ];
        // %33/%34 are half values produced upstream (modeled as undefs here for the test body).
        let mut func = Function::new();
        func.blocks = vec![block];
        m.functions = vec![func];

        assert!(rewrite_subword_packed_scalars(&mut m));

        // The variable now points at a `<2 x half>` Function pointer (not the original uint pointer %4).
        let var = m.functions[0].blocks[0]
            .instructions
            .iter()
            .find(|x| x.result_id == Some(20) && x.class.opcode == Op::Variable)
            .unwrap();
        let var_ptr = var.result_type.unwrap();
        assert_ne!(var_ptr, 4);
        let pointee = m
            .types_global_values
            .iter()
            .find(|x| x.result_id == Some(var_ptr))
            .and_then(|x| x.operands.get(1))
            .cloned();
        let vec_id = match pointee {
            Some(Operand::IdRef(v)) => v,
            _ => panic!("var ptr has no pointee"),
        };
        let vecdef = m
            .types_global_values
            .iter()
            .find(|x| x.result_id == Some(vec_id))
            .unwrap();
        assert_eq!(vecdef.class.opcode, Op::TypeVector);
        assert_eq!(vecdef.operands.first(), Some(&Operand::IdRef(2))); // component = half
        assert_eq!(vecdef.operands.get(1), Some(&Operand::LiteralBit32(2))); // 2 lanes

        // The whole-word load %35 became OpBitcast %uint (its result id is preserved), fed by a fresh
        // OpLoad of the vector type.
        let body = &m.functions[0].blocks[0].instructions;
        let bc = body.iter().find(|x| x.result_id == Some(35)).unwrap();
        assert_eq!(bc.class.opcode, Op::Bitcast);
        assert_eq!(bc.result_type, Some(1)); // -> uint
        let src = match bc.operands.first() {
            Some(Operand::IdRef(s)) => *s,
            _ => panic!("bitcast has no source"),
        };
        let vload = body.iter().find(|x| x.result_id == Some(src)).unwrap();
        assert_eq!(vload.class.opcode, Op::Load);
        assert_eq!(vload.result_type, Some(vec_id));
        assert_eq!(vload.operands.first(), Some(&Operand::IdRef(20)));

        // The half access chains are untouched (still index the variable producing %_ptr_Function_half).
        let g0 = body.iter().find(|x| x.result_id == Some(31)).unwrap();
        assert_eq!(g0.class.opcode, Op::InBoundsAccessChain);
        assert_eq!(g0.result_type, Some(5));
    }

    // A `uint` Function alloca whose bytes are taken by a foreign op (here passed as a function-call
    // argument, not a load/store/sub-word-chain) must be left untouched — the strict all-uses gate bails.
    #[test]
    fn uint_alloca_with_foreign_use_is_left_untouched() {
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(40));
        m.types_global_values = vec![
            i(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            i(
                Op::TypeFloat,
                None,
                Some(2),
                vec![Operand::LiteralBit32(16)],
            ),
            i(
                Op::TypePointer,
                None,
                Some(4),
                vec![
                    Operand::StorageClass(StorageClass::Function),
                    Operand::IdRef(1),
                ],
            ),
            i(
                Op::TypePointer,
                None,
                Some(5),
                vec![
                    Operand::StorageClass(StorageClass::Function),
                    Operand::IdRef(2),
                ],
            ),
            i(
                Op::Constant,
                Some(1),
                Some(10),
                vec![Operand::LiteralBit32(0)],
            ),
        ];
        let mut block = Block::new();
        block.label = Some(i(Op::Label, None, Some(30), vec![]));
        block.instructions = vec![
            // The Function-storage variable lives in the function's entry block.
            i(
                Op::Variable,
                Some(4),
                Some(20),
                vec![Operand::StorageClass(StorageClass::Function)],
            ),
            // A legitimate sub-word chain...
            i(
                Op::InBoundsAccessChain,
                Some(5),
                Some(31),
                vec![Operand::IdRef(20), Operand::IdRef(10)],
            ),
            // ...but the variable is ALSO handed whole to a foreign op (the address escapes) -> bail.
            i(
                Op::FunctionCall,
                Some(1),
                Some(32),
                vec![Operand::IdRef(99), Operand::IdRef(20)],
            ),
            i(Op::Return, None, None, vec![]),
        ];
        let mut func = Function::new();
        func.blocks = vec![block];
        m.functions = vec![func];

        assert!(!rewrite_subword_packed_scalars(&mut m));
    }
}
