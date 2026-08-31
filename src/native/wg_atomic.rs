//! In-memory Workgroup float-as-int atomic remodel.
//!
//! Metal's `atomic_fetch_min/max` over a *threadgroup* `float` (the float-as-signed-int min/max idiom)
//! lowers to an `OpBitcast %_ptr_Workgroup_<int32> %chain` followed by `OpAtomicSMin/SMax`. The pointer
//! `OpBitcast` is illegal under Logical addressing — spirv-val: *"Instruction may not have a logical
//! pointer operand"*. The honest fix is to RETYPE the Workgroup variable's pointee from float to the
//! 32-bit int the atomics use: the atomic then operates on a native int pointer (no bitcast), and the
//! plain float load/stores of the same variable reinterpret via a VALUE `OpBitcast` (a legal numeric
//! reinterpret, not a pointer one).
//!
//! This is byte-SAFE by construction: Workgroup is shader-internal scratch (never device-visible, never
//! the golden's output), and float↔int32 is a bit-identical 32-bit reinterpret. The retype preserves the
//! pointee TREE shape exactly (every array length and struct field order is kept; only the float LEAVES
//! become int), so every access-chain index is unchanged — the data sits at the same byte offsets.
//!
//! The ordinary lowering pipeline runs this after interface and memory construction, when every
//! Workgroup pointer use and its final pointee tree are available. Retry variants share that same
//! construction boundary; no validator result is needed to select it. It decides purely from IR
//! structure (a Workgroup variable whose pointee tree is reached only as float/int leaves used as the
//! atomic-bitcast idiom) — never a shader name.

use crate::spirv_module::Instruction;
use crate::spirv_module::Module;
use crate::spirv_module::Operand;
use spirv::{Op, StorageClass, Word};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Retype every Workgroup variable that is accessed ONLY as the float-as-int atomic idiom so its float
/// leaves become the 32-bit int the atomics use, dropping the illegal pointer `OpBitcast`s. Returns true
/// if any variable was remodeled.
pub(super) fn construct_workgroup_atomic_floats(module: &mut Module) -> bool {
    let float_ty = match scalar_type(module, Op::TypeFloat, 32) {
        Some(t) => t,
        None => return false,
    };
    let int32_types: HashSet<Word> = module
        .types_global_values
        .iter()
        .filter(|i| {
            i.class.opcode == Op::TypeInt && i.operands.first() == Some(&Operand::LiteralBit32(32))
        })
        .filter_map(|i| i.result_id)
        .collect();
    if int32_types.is_empty() {
        return false;
    }

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

    // Candidate Workgroup variables whose pointee tree CONTAINS a float leaf (a flat `array<float,K>`,
    // a nested struct/array tree, or a bare float scalar).
    let mut cands: Vec<(Word, Word)> = Vec::new(); // (var, pointee)
    for inst in &module.types_global_values {
        if inst.class.opcode != Op::Variable {
            continue;
        }
        let (Some(var), Some(ptr_ty)) = (inst.result_id, inst.result_type) else {
            continue;
        };
        let Some(&(StorageClass::Workgroup, pointee)) = ptr_info.get(&ptr_ty) else {
            continue;
        };
        if tree_contains_float(&type_defs, pointee, float_ty, &mut HashSet::new()) {
            cands.push((var, pointee));
        }
    }
    if cands.is_empty() {
        return false;
    }

    let mut next_id = module.header.as_ref().map(|h| h.bound).unwrap_or(0);
    let mut changed = false;
    for (var, pointee) in cands {
        if remodel_one(module, var, pointee, float_ty, &int32_types, &mut next_id) {
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

/// Whether `ty` is a float leaf, or an array/struct tree that bottoms out (somewhere) on a float leaf.
fn tree_contains_float(
    defs: &HashMap<Word, Instruction>,
    ty: Word,
    float_ty: Word,
    seen: &mut HashSet<Word>,
) -> bool {
    if ty == float_ty {
        return true;
    }
    if !seen.insert(ty) {
        return false;
    }
    let Some(def) = defs.get(&ty) else {
        return false;
    };
    match def.class.opcode {
        Op::TypeArray => operand_id(def, 0)
            .map(|elem| tree_contains_float(defs, elem, float_ty, seen))
            .unwrap_or(false),
        Op::TypeStruct => def.operands.iter().any(
            |o| matches!(o, Operand::IdRef(f) if tree_contains_float(defs, *f, float_ty, seen)),
        ),
        _ => false,
    }
}

struct RemodelPlan {
    /// int32 type the atomic bitcasts agree on.
    int_ty: Word,
    /// Every access chain (transitively) rooted at the variable -> its ORIGINAL result pointee type.
    /// Each is retyped to point at the float→int clone of that pointee (intermediate aggregate chains
    /// included), so the indices traverse the cloned tree unchanged.
    chain_pointee: BTreeMap<Word, Word>,
    /// Result ids of the chains whose leaf pointee is float (retyped to the int element pointer; their
    /// plain load/stores reinterpret via a value bitcast).
    float_chain_ids: HashSet<Word>,
    /// `OpBitcast %_ptr_Workgroup_int %chain` result ids to drop.
    bitcast_ids: HashSet<Word>,
    /// Dropped-bitcast id -> the underlying chain id it should be replaced with.
    bitcast_to_chain: HashMap<Word, Word>,
    /// The variable being remodeled.
    var: Word,
    /// True if a whole-variable zero-init `OpStore %var %pointee_null` is present; its value operand is
    /// retyped to a null of the cloned int tree in `rewrite_bodies`.
    null_init: bool,
}

/// Validate `var` (pointee `pointee`) is reached only as the float-as-int atomic idiom and, if so,
/// retype it in place. Returns true if remodeled. On any unmodeled use the variable is left untouched.
fn remodel_one(
    module: &mut Module,
    var: Word,
    pointee: Word,
    float_ty: Word,
    int32_types: &HashSet<Word>,
    next_id: &mut Word,
) -> bool {
    // The remodel only operates on the entry point's function(s); a banked case never reaches here. Find
    // every function and validate the var's uses across all of them (a Workgroup var is module scope).
    let Some(plan) = validate(module, var, pointee, float_ty, int32_types) else {
        return false;
    };

    // Recursively clone the pointee tree float -> int_ty (fresh, undecorated type ids so the clone can
    // never alias a device Block type that dedups to the same shape). Children are emitted before
    // parents (define-before-use), then the whole block is spliced in front of the variable def.
    let mut fresh = || {
        let id = *next_id;
        *next_id += 1;
        id
    };
    let mut memo: HashMap<Word, Word> = HashMap::new();
    let mut new_types: Vec<Instruction> = Vec::new();
    let type_defs: HashMap<Word, Instruction> = module
        .types_global_values
        .iter()
        .filter_map(|i| i.result_id.map(|id| (id, i.clone())))
        .collect();
    let Some(new_pointee) = clone_f2i(
        &type_defs,
        pointee,
        float_ty,
        plan.int_ty,
        &mut memo,
        &mut new_types,
        &mut fresh,
    ) else {
        return false;
    };

    // Whole-variable zero-init store: mint an `OpConstantNull` of the cloned (int) tree so the store's
    // value matches the retyped variable. All-zero bits are float↔int-identical, so this writes the same
    // bytes. Emitted after `new_pointee` (define-before-use) and repointed onto the store in the body.
    let new_null = plan.null_init.then(|| {
        let id = fresh();
        new_types.push(Instruction::new(
            Op::ConstantNull,
            Some(new_pointee),
            Some(id),
            vec![],
        ));
        id
    });

    // The cloned counterpart of an original pointee type: int_ty for a float leaf, the memoized clone of
    // a float-bearing aggregate, or the type itself for a subtree with no float.
    let cloned_of = |p: Word| -> Word {
        if p == float_ty {
            plan.int_ty
        } else {
            *memo.get(&p).unwrap_or(&p)
        }
    };

    // A Workgroup pointer to each cloned pointee a rooted chain (or the variable) needs as its result
    // type. Reuse an existing pointer type when present, else synthesize one.
    let mut wg_ptr_cache: HashMap<Word, Word> = HashMap::new();
    let mut wg_ptr_to = |pointee: Word,
                         module: &Module,
                         new_types: &mut Vec<Instruction>,
                         fresh: &mut dyn FnMut() -> Word|
     -> Word {
        if let Some(&id) = wg_ptr_cache.get(&pointee) {
            return id;
        }
        let id = find_ptr(module, StorageClass::Workgroup, pointee).unwrap_or_else(|| {
            let id = fresh();
            new_types.push(Instruction::new(
                Op::TypePointer,
                None,
                Some(id),
                vec![
                    Operand::StorageClass(StorageClass::Workgroup),
                    Operand::IdRef(pointee),
                ],
            ));
            id
        });
        wg_ptr_cache.insert(pointee, id);
        id
    };

    // The variable's new pointer type points at the cloned full tree.
    let new_var_ptr = wg_ptr_to(new_pointee, module, &mut new_types, &mut fresh);
    // The new result-type pointer for each rooted chain, keyed by its result id.
    let mut chain_new_ptr: HashMap<Word, Word> = HashMap::new();
    for (&chain_id, &orig_pointee) in &plan.chain_pointee {
        let new_pointee = cloned_of(orig_pointee);
        let ptr = wg_ptr_to(new_pointee, module, &mut new_types, &mut fresh);
        chain_new_ptr.insert(chain_id, ptr);
    }

    // Splice the synthesized types immediately before the variable def and repoint the variable to the
    // cloned-tree pointer (define-before-use: the cloned types reference only pre-existing scalars + each
    // other in dependency order, all now ahead of the variable).
    let Some(var_pos) = module
        .types_global_values
        .iter()
        .position(|i| i.class.opcode == Op::Variable && i.result_id == Some(var))
    else {
        return false;
    };
    module.types_global_values[var_pos].result_type = Some(new_var_ptr);
    let tail = module.types_global_values.split_off(var_pos);
    module.types_global_values.extend(new_types);
    module.types_global_values.extend(tail);

    // Retype each rooted chain result to its cloned-pointee pointer (result ids are globally unique).
    for func in module.functions.iter_mut() {
        for block in func.blocks.iter_mut() {
            for inst in block.instructions.iter_mut() {
                if let Some(&ptr) = chain_new_ptr.get(&inst.result_id.unwrap_or(0)) {
                    inst.result_type = Some(ptr);
                }
            }
        }
    }

    // Rewrite each function body: drop the pointer bitcasts, repoint atomics straight at the int chain,
    // and value-bitcast the plain float loads/stores through the retyped (now int) leaf chains.
    rewrite_bodies(module, &plan, new_null, next_id);
    true
}

fn validate(
    module: &Module,
    var: Word,
    pointee: Word,
    float_ty: Word,
    int32_types: &HashSet<Word>,
) -> Option<RemodelPlan> {
    // Whole-variable zero-init: `OpStore %var %null` where `%null` is `OpConstantNull` of the variable's
    // pointee tree (the threadgroup array cleared to zero). All-bits-zero is identical in float and int
    // representation, so this is retypeable to a null of the cloned (int) tree — byte-identical.
    let pointee_null_ids: HashSet<Word> = module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::ConstantNull && i.result_type == Some(pointee))
        .filter_map(|i| i.result_id)
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

    // Pass 1: collect every access chain TRANSITIVELY rooted at the variable (a chain whose base —
    // operand 0 — is the variable or another rooted chain). Intermediate chains may stop at a sub-
    // aggregate; the leaves must reach a float or 32-bit int scalar. Iterated to a fixpoint so chain
    // order within/across blocks does not matter. Every such chain's result must be a Workgroup pointer.
    let mut chain_pointee: BTreeMap<Word, Word> = BTreeMap::new();
    let mut float_chain_ids: HashSet<Word> = HashSet::new();
    let mut roots: HashSet<Word> = HashSet::new();
    roots.insert(var);
    loop {
        let mut added = false;
        for func in &module.functions {
            for block in &func.blocks {
                for inst in &block.instructions {
                    let Some(result_id) = inst.result_id else {
                        continue;
                    };
                    if chain_pointee.contains_key(&result_id) {
                        continue;
                    }
                    if !matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain) {
                        continue;
                    }
                    let Some(base) = operand_id(inst, 0) else {
                        continue;
                    };
                    if !roots.contains(&base) {
                        continue;
                    }
                    let result_ty = inst.result_type?;
                    let &(sc, pointee) = ptr_info.get(&result_ty)?;
                    if sc != StorageClass::Workgroup {
                        return None;
                    }
                    chain_pointee.insert(result_id, pointee);
                    roots.insert(result_id);
                    if pointee == float_ty {
                        float_chain_ids.insert(result_id);
                    }
                    added = true;
                }
            }
        }
        if !added {
            break;
        }
    }
    let all_chain_ids: HashSet<Word> = chain_pointee.keys().copied().collect();
    if float_chain_ids.is_empty() {
        return None;
    }

    // The variable itself may appear ONLY as the base of a rooted access chain (operand 0), or as the
    // pointer of a whole-variable zero-init `OpStore %var %pointee_null`. Any other mention disqualifies
    // it. The zero-init store is retyped to a null of the cloned int tree in `rewrite_bodies` (all-zero
    // bits are float↔int-identical).
    let mut null_init = false;
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                let is_chain_base =
                    matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain)
                        && operand_id(inst, 0) == Some(var);
                if is_chain_base {
                    continue;
                }
                if inst.class.opcode == Op::Store
                    && operand_id(inst, 0) == Some(var)
                    && operand_id(inst, 1).is_some_and(|v| pointee_null_ids.contains(&v))
                {
                    null_init = true;
                    continue;
                }
                if inst
                    .operands
                    .iter()
                    .any(|o| matches!(o, Operand::IdRef(id) if *id == var))
                {
                    return None;
                }
            }
        }
    }

    // Pass 2: collect the chain -> `_ptr_Workgroup_<int>` pointer bitcasts and PIN the single int32 type
    // they reinterpret to. Every such bitcast must agree on int_ty and feed an OpAtomic* as its pointer.
    let mut bitcast_ids: HashSet<Word> = HashSet::new();
    let mut bitcast_to_chain: HashMap<Word, Word> = HashMap::new();
    let mut int_ty: Option<Word> = None;
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                if inst.class.opcode != Op::Bitcast {
                    continue;
                }
                let Some(src) = operand_id(inst, 0) else {
                    continue;
                };
                if !all_chain_ids.contains(&src) {
                    continue;
                }
                let result_ty = inst.result_type?;
                let &(sc, leaf) = ptr_info.get(&result_ty)?;
                if sc != StorageClass::Workgroup || !int32_types.contains(&leaf) {
                    return None;
                }
                match int_ty {
                    Some(t) if t != leaf => return None,
                    _ => int_ty = Some(leaf),
                }
                bitcast_ids.insert(inst.result_id?);
                bitcast_to_chain.insert(inst.result_id?, src);
            }
        }
    }
    let int_ty = match int_ty {
        Some(t) => t,
        None => {
            return None;
        }
    };

    // Pass 3: strict all-uses gate. Every use of a chain id must be either a dropped bitcast source, an
    // OpLoad/OpStore pointer, or an OpAtomic* pointer (operand 0). Every use of a bitcast id must be an
    // OpAtomic* pointer. Any other use disqualifies the variable (bail, do not miscompile).
    for func in &module.functions {
        for block in &func.blocks {
            for inst in &block.instructions {
                for (oi, op) in inst.operands.iter().enumerate() {
                    let Operand::IdRef(id) = op else { continue };
                    if all_chain_ids.contains(id) {
                        let ok = match inst.class.opcode {
                            // A rooted chain may base a deeper rooted chain (operand 0).
                            Op::InBoundsAccessChain | Op::AccessChain => oi == 0,
                            Op::Bitcast => bitcast_ids.contains(&inst.result_id.unwrap_or(0)),
                            Op::Load => oi == 0,
                            Op::Store => oi == 0,
                            op if is_atomic(op) => oi == 0,
                            _ => false,
                        };
                        if !ok {
                            return None;
                        }
                    }
                    if bitcast_ids.contains(id) && !(is_atomic(inst.class.opcode) && oi == 0) {
                        return None;
                    }
                }
            }
        }
    }

    Some(RemodelPlan {
        int_ty,
        chain_pointee,
        float_chain_ids,
        bitcast_ids,
        bitcast_to_chain,
        var,
        null_init,
    })
}

fn rewrite_bodies(
    module: &mut Module,
    plan: &RemodelPlan,
    new_null: Option<Word>,
    next_id: &mut Word,
) {
    let int_ty = plan.int_ty;
    for func in module.functions.iter_mut() {
        for block in func.blocks.iter_mut() {
            let insts = block.instructions.clone();
            let mut out = Vec::with_capacity(insts.len());
            for mut inst in insts {
                // Whole-variable zero-init store: repoint its value to the cloned int tree's null so it
                // matches the retyped variable (byte-identical all-zero write).
                if let (Op::Store, Some(new_null)) = (inst.class.opcode, new_null) {
                    if operand_id(&inst, 0) == Some(plan.var) {
                        inst.operands[1] = Operand::IdRef(new_null);
                        out.push(inst);
                        continue;
                    }
                }
                // Drop the now-redundant `OpBitcast %_ptr_Workgroup_int %chain`.
                if inst.class.opcode == Op::Bitcast
                    && inst
                        .result_id
                        .map(|r| plan.bitcast_ids.contains(&r))
                        .unwrap_or(false)
                {
                    continue;
                }
                // Plain FLOAT load through a retyped (now int) leaf chain: load int, bitcast value to the
                // original float result type.
                if inst.class.opcode == Op::Load {
                    if let Some(ptr) = operand_id(&inst, 0) {
                        if plan.float_chain_ids.contains(&ptr) && inst.result_type != Some(int_ty) {
                            let (rt, rid) = (inst.result_type.unwrap(), inst.result_id.unwrap());
                            let tmp = *next_id;
                            *next_id += 1;
                            out.push(Instruction::new(
                                Op::Load,
                                Some(int_ty),
                                Some(tmp),
                                vec![Operand::IdRef(ptr)],
                            ));
                            out.push(Instruction::new(
                                Op::Bitcast,
                                Some(rt),
                                Some(rid),
                                vec![Operand::IdRef(tmp)],
                            ));
                            continue;
                        }
                    }
                }
                // Plain FLOAT store through a retyped leaf chain: bitcast the float object to int, store.
                if inst.class.opcode == Op::Store {
                    if let Some(ptr) = operand_id(&inst, 0) {
                        if plan.float_chain_ids.contains(&ptr) {
                            let fval = operand_id(&inst, 1).unwrap();
                            let tmp = *next_id;
                            *next_id += 1;
                            out.push(Instruction::new(
                                Op::Bitcast,
                                Some(int_ty),
                                Some(tmp),
                                vec![Operand::IdRef(fval)],
                            ));
                            out.push(Instruction::new(
                                Op::Store,
                                None,
                                None,
                                vec![Operand::IdRef(ptr), Operand::IdRef(tmp)],
                            ));
                            continue;
                        }
                    }
                }
                // Repoint any operand referencing a dropped bitcast onto the underlying int chain (the
                // atomic's pointer operand).
                for op in inst.operands.iter_mut() {
                    if let Operand::IdRef(id) = op {
                        if let Some(&c) = plan.bitcast_to_chain.get(id) {
                            *op = Operand::IdRef(c);
                        }
                    }
                }
                out.push(inst);
            }
            block.instructions = out;
        }
    }
}

/// Recursively clone `ty` replacing every float leaf with `int_ty`, minting fresh undecorated type ids.
/// Children are pushed to `new_types` before parents (define-before-use). Returns the cloned type id, or
/// the original id for a subtree with no float (unchanged), or None on an unsupported composite.
fn clone_f2i(
    defs: &HashMap<Word, Instruction>,
    ty: Word,
    float_ty: Word,
    int_ty: Word,
    memo: &mut HashMap<Word, Word>,
    new_types: &mut Vec<Instruction>,
    fresh: &mut dyn FnMut() -> Word,
) -> Option<Word> {
    if ty == float_ty {
        return Some(int_ty);
    }
    if let Some(&m) = memo.get(&ty) {
        return Some(m);
    }
    let def = defs.get(&ty)?;
    match def.class.opcode {
        Op::TypeArray => {
            let elem = operand_id(def, 0)?;
            let len = match def.operands.get(1)? {
                Operand::IdRef(c) => *c,
                _ => return None,
            };
            let new_elem = clone_f2i(defs, elem, float_ty, int_ty, memo, new_types, fresh)?;
            if new_elem == elem {
                memo.insert(ty, ty);
                return Some(ty); // no float inside; reuse the original array type.
            }
            let id = fresh();
            new_types.push(Instruction::new(
                Op::TypeArray,
                None,
                Some(id),
                vec![Operand::IdRef(new_elem), Operand::IdRef(len)],
            ));
            memo.insert(ty, id);
            Some(id)
        }
        Op::TypeStruct => {
            let mut fields: Vec<Word> = Vec::new();
            let mut any_changed = false;
            for o in &def.operands {
                let Operand::IdRef(f) = o else { return None };
                let nf = clone_f2i(defs, *f, float_ty, int_ty, memo, new_types, fresh)?;
                any_changed |= nf != *f;
                fields.push(nf);
            }
            if !any_changed {
                memo.insert(ty, ty);
                return Some(ty);
            }
            let id = fresh();
            new_types.push(Instruction::new(
                Op::TypeStruct,
                None,
                Some(id),
                fields.into_iter().map(Operand::IdRef).collect(),
            ));
            memo.insert(ty, id);
            Some(id)
        }
        // A non-float scalar / vector / other composite with no float subtree stays as-is.
        _ => {
            memo.insert(ty, ty);
            Some(ty)
        }
    }
}

fn scalar_type(module: &Module, op: Op, bits: u32) -> Option<Word> {
    module.types_global_values.iter().find_map(|i| {
        (i.class.opcode == op && i.operands.first() == Some(&Operand::LiteralBit32(bits)))
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

fn is_atomic(op: Op) -> bool {
    matches!(
        op,
        Op::AtomicSMin
            | Op::AtomicSMax
            | Op::AtomicUMin
            | Op::AtomicUMax
            | Op::AtomicIAdd
            | Op::AtomicISub
            | Op::AtomicAnd
            | Op::AtomicOr
            | Op::AtomicXor
            | Op::AtomicExchange
            | Op::AtomicCompareExchange
            | Op::AtomicLoad
            | Op::AtomicStore
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::{Block, Function, ModuleHeader};

    fn i(op: Op, ty: Option<Word>, res: Option<Word>, ops: Vec<Operand>) -> Instruction {
        Instruction::new(op, ty, res, ops)
    }

    // A nested-aggregate Workgroup variable `[2 x {float}]` reached through a TWO-LEVEL access chain to a
    // float leaf, then `OpBitcast %_ptr_Workgroup_uint` → `OpAtomicSMin` (the float-as-signed-int atomic
    // idiom). The remodel must: clone the pointee tree float→uint, retype the variable AND both chain
    // levels (the intermediate struct chain included), drop the pointer bitcast, and repoint the atomic at
    // the now-uint leaf chain. Byte-safe by construction (Workgroup scratch, 32-bit reinterpret).
    #[test]
    fn nested_workgroup_float_atomic_remodels_to_uint() {
        // ids: float=1 uint=2 uint_2=3 struct{float}=5 arr=6 ptrWgArr=7 ptrWgStruct=8 ptrWgFloat=9
        //      ptrWgUint=10 | uint_0=11 scope=12 val=13 | var=20 | entry=30 chain1=31 chain2=32 bitcast=33
        //      atomic=34
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(40));
        m.types_global_values = vec![
            i(
                Op::TypeFloat,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32)],
            ),
            i(
                Op::TypeInt,
                None,
                Some(2),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            i(
                Op::Constant,
                Some(2),
                Some(3),
                vec![Operand::LiteralBit32(2)],
            ),
            i(Op::TypeStruct, None, Some(5), vec![Operand::IdRef(1)]),
            i(
                Op::TypeArray,
                None,
                Some(6),
                vec![Operand::IdRef(5), Operand::IdRef(3)],
            ),
            i(
                Op::TypePointer,
                None,
                Some(7),
                vec![
                    Operand::StorageClass(StorageClass::Workgroup),
                    Operand::IdRef(6),
                ],
            ),
            i(
                Op::TypePointer,
                None,
                Some(8),
                vec![
                    Operand::StorageClass(StorageClass::Workgroup),
                    Operand::IdRef(5),
                ],
            ),
            i(
                Op::TypePointer,
                None,
                Some(9),
                vec![
                    Operand::StorageClass(StorageClass::Workgroup),
                    Operand::IdRef(1),
                ],
            ),
            i(
                Op::TypePointer,
                None,
                Some(10),
                vec![
                    Operand::StorageClass(StorageClass::Workgroup),
                    Operand::IdRef(2),
                ],
            ),
            i(
                Op::Constant,
                Some(2),
                Some(11),
                vec![Operand::LiteralBit32(0)],
            ),
            i(
                Op::Constant,
                Some(2),
                Some(12),
                vec![Operand::LiteralBit32(1)],
            ),
            i(
                Op::Constant,
                Some(2),
                Some(13),
                vec![Operand::LiteralBit32(7)],
            ),
            i(
                Op::Variable,
                Some(7),
                Some(20),
                vec![Operand::StorageClass(StorageClass::Workgroup)],
            ),
            i(Op::Undef, Some(9), Some(14), vec![]),
        ];
        let mut block = Block::new();
        block.label = Some(i(Op::Label, None, Some(30), vec![]));
        block.instructions = vec![
            // var[0] -> struct ptr (intermediate)
            i(
                Op::InBoundsAccessChain,
                Some(8),
                Some(31),
                vec![Operand::IdRef(20), Operand::IdRef(11)],
            ),
            // struct.field0 -> float ptr (leaf)
            i(
                Op::InBoundsAccessChain,
                Some(9),
                Some(32),
                vec![Operand::IdRef(31), Operand::IdRef(11)],
            ),
            i(Op::Bitcast, Some(10), Some(33), vec![Operand::IdRef(32)]),
            i(
                Op::AtomicSMin,
                Some(2),
                Some(34),
                vec![
                    Operand::IdRef(33),
                    Operand::IdRef(12),
                    Operand::IdRef(11),
                    Operand::IdRef(13),
                ],
            ),
            // A structurizer may leave a dead pointer phi and its rooted chain after removing every
            // observable consumer. Liveness closure at the module-construction boundary must remove
            // this graph before the strict all-uses storage check runs.
            i(
                Op::InBoundsAccessChain,
                Some(9),
                Some(35),
                vec![Operand::IdRef(20), Operand::IdRef(11), Operand::IdRef(11)],
            ),
            i(
                Op::Phi,
                Some(9),
                Some(36),
                vec![
                    Operand::IdRef(14),
                    Operand::IdRef(30),
                    Operand::IdRef(35),
                    Operand::IdRef(30),
                ],
            ),
            i(Op::Return, None, None, vec![]),
        ];
        let mut func = Function::new();
        func.blocks = vec![block];
        m.functions = vec![func];

        assert!(
            !construct_workgroup_atomic_floats(&mut m),
            "a dead pointer escape must not weaken the constructor's all-uses gate"
        );
        assert!(crate::native::eliminate_dead_pointer_values_module(
            &mut m,
            &HashSet::new()
        ));
        assert!(construct_workgroup_atomic_floats(&mut m));
        assert!(
            !construct_workgroup_atomic_floats(&mut m),
            "construction must close the complete Workgroup float-as-int atomic graph"
        );

        let body = &m.functions[0].blocks[0].instructions;
        // The pointer bitcast (%33) is gone.
        assert!(!body.iter().any(|x| x.result_id == Some(33)));
        // The atomic's pointer operand is now the leaf chain (%32) directly.
        let atomic = body.iter().find(|x| x.result_id == Some(34)).unwrap();
        assert_eq!(atomic.operands.first(), Some(&Operand::IdRef(32)));
        // The leaf chain (%32) result type now points at uint (Workgroup int element pointer).
        let leaf = body.iter().find(|x| x.result_id == Some(32)).unwrap();
        let leaf_ptr = leaf.result_type.unwrap();
        let pointee = m
            .types_global_values
            .iter()
            .find(|x| x.result_id == Some(leaf_ptr))
            .and_then(|x| x.operands.get(1));
        assert_eq!(pointee, Some(&Operand::IdRef(2))); // -> uint
                                                       // The variable (%20) was retyped to a fresh cloned-tree pointer (not the original %7).
        let var = m
            .types_global_values
            .iter()
            .find(|x| x.result_id == Some(20))
            .unwrap();
        assert_ne!(var.result_type, Some(7));
        // No OpBitcast to a Workgroup pointer survives anywhere (the illegal logical-pointer bitcast).
        assert!(!m.functions[0].blocks[0]
            .instructions
            .iter()
            .any(|x| x.class.opcode == Op::Bitcast));
    }

    // A Workgroup variable whose float leaf is reached but ALSO used outside the atomic idiom (here passed
    // to a non-idiom op) must be left untouched — the strict all-uses gate bails rather than miscompile.
    #[test]
    fn workgroup_float_with_foreign_use_is_left_untouched() {
        // Minimal: var[float leaf] loaded normally (no atomic bitcast at all) -> no int_ty pinned -> bail.
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(30));
        m.types_global_values = vec![
            i(
                Op::TypeFloat,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32)],
            ),
            i(
                Op::TypeInt,
                None,
                Some(2),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            i(
                Op::TypePointer,
                None,
                Some(7),
                vec![
                    Operand::StorageClass(StorageClass::Workgroup),
                    Operand::IdRef(1),
                ],
            ),
            i(
                Op::Variable,
                Some(7),
                Some(20),
                vec![Operand::StorageClass(StorageClass::Workgroup)],
            ),
        ];
        let mut block = Block::new();
        block.label = Some(i(Op::Label, None, Some(25), vec![]));
        block.instructions = vec![
            i(Op::Load, Some(1), Some(26), vec![Operand::IdRef(20)]),
            i(Op::Return, None, None, vec![]),
        ];
        let mut func = Function::new();
        func.blocks = vec![block];
        m.functions = vec![func];

        // No access chain, no atomic bitcast -> the variable is not in the idiom -> no remodel.
        assert!(!construct_workgroup_atomic_floats(&mut m));
    }
}
