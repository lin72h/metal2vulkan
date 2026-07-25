//! Deterministic threadgroup memory: zero-initialize every Workgroup variable at kernel entry.
//!
//! Metal leaves threadgroup memory contents UNDEFINED at dispatch start; a kernel that reads a
//! threadgroup slot it never wrote observes leftover GPU memory, so its output is nondeterministic
//! run-to-run. The validation harness byte-compares an Apple Metal golden against this translator's
//! Vulkan run, which is only meaningful if both sides refine that undefined behavior the same way:
//! reads of never-written threadgroup memory return ZERO. The macOS oracle injects an equivalent
//! zero-fill prologue into the AIR module; this pass is the candidate half.
//!
//! Mechanism: at the top of the kernel entry block (after the leading `OpVariable`s), every
//! invocation stores `OpConstantNull` into each Workgroup variable, then one `OpControlBarrier`
//! (Workgroup/Workgroup, AcquireRelease|WorkgroupMemory) orders the fill before the body. All
//! invocations write the same zero bytes, so the racy fill is value-deterministic, and the barrier
//! orders it against the kernel's own threadgroup writes. Purely structural: it keys on the
//! Workgroup storage class only (module threadgroup globals and the interface's synthesized
//! threadgroup-argument arrays alike) and is a no-op for modules with no Workgroup variables.
//!
//! When the shader explicitly stores to an atomic Workgroup object before the first workgroup
//! barrier, that store defines the object's initial value. The harness prologue is redundant for
//! that shape, so those variables are left to the shader's own initialization.

use super::*;

pub(in crate::passes) fn zero_initialize_workgroup_memory(ctx: &mut Ctx, entry_idx: usize) {
    // Every Workgroup OpVariable: module globals (emitter-declared threadgroup globals) plus the
    // not-yet-drained synthesized globals (the interface's threadgroup-argument arrays).
    let vars: Vec<(Word, Word)> = ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
        .filter(|inst| {
            inst.class.opcode == Op::Variable
                && inst.operands.first() == Some(&Operand::StorageClass(StorageClass::Workgroup))
        })
        .filter_map(|inst| Some((inst.result_id?, inst.result_type?)))
        .collect();
    if vars.is_empty() {
        return;
    }

    let shader_initialized_atomic_vars =
        shader_initialized_atomic_workgroup_vars(ctx, entry_idx, &vars);
    let mut null_by_pointee: HashMap<Word, Word> = HashMap::new();
    let mut prologue = Vec::new();
    for (var, ptr_ty) in vars {
        if shader_initialized_atomic_vars.contains(&var) {
            continue;
        }
        let Some(ptr_def) = type_def_of(ctx, ptr_ty) else {
            continue;
        };
        if ptr_def.class.opcode != Op::TypePointer {
            continue;
        }
        let Some(&Operand::IdRef(pointee)) = ptr_def.operands.get(1) else {
            continue;
        };
        let null_id = *null_by_pointee.entry(pointee).or_insert_with(|| {
            let id = ctx.module.fresh_id();
            ctx.new_globals.push(Instruction::new(
                Op::ConstantNull,
                Some(pointee),
                Some(id),
                vec![],
            ));
            id
        });
        prologue.push(Instruction::new(
            Op::Store,
            None,
            None,
            vec![Operand::IdRef(var), Operand::IdRef(null_id)],
        ));
    }
    if prologue.is_empty() {
        return;
    }
    let scope = ctx.const_uint(Scope::Workgroup as u32);
    let semantics = ctx
        .const_uint((MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY).bits());
    prologue.push(Instruction::new(
        Op::ControlBarrier,
        None,
        None,
        vec![
            Operand::IdScope(scope),
            Operand::IdScope(scope),
            Operand::IdMemorySemantics(semantics),
        ],
    ));

    // Insert after the entry block's leading OpVariables (which must stay first in the block).
    let Some(block) = ctx.module.functions[entry_idx].blocks.first_mut() else {
        return;
    };
    let insert_at = block
        .instructions
        .iter()
        .position(|inst| inst.class.opcode != Op::Variable)
        .unwrap_or(block.instructions.len());
    block.instructions.splice(insert_at..insert_at, prologue);
}

fn shader_initialized_atomic_workgroup_vars(
    ctx: &Ctx,
    entry_idx: usize,
    vars: &[(Word, Word)],
) -> HashSet<Word> {
    let workgroup_vars: HashSet<Word> = vars.iter().map(|(var, _)| *var).collect();
    let pointer_sources = pointer_source_map(ctx, entry_idx);
    let mut prebarrier_stores = HashSet::new();
    let mut atomic_uses = HashSet::new();
    let mut before_first_barrier = true;

    for block in &ctx.module.functions[entry_idx].blocks {
        for inst in &block.instructions {
            if inst.class.opcode == Op::ControlBarrier {
                before_first_barrier = false;
            }
            if before_first_barrier && inst.class.opcode == Op::Store {
                if let Some(root) =
                    id_ref_at(inst, 0).and_then(|ptr| pointer_root(ptr, &pointer_sources))
                {
                    if workgroup_vars.contains(&root) {
                        prebarrier_stores.insert(root);
                    }
                }
            }
            if is_atomic_op(inst.class.opcode) {
                if let Some(root) =
                    id_ref_at(inst, 0).and_then(|ptr| pointer_root(ptr, &pointer_sources))
                {
                    if workgroup_vars.contains(&root) {
                        atomic_uses.insert(root);
                    }
                }
            }
        }
    }

    prebarrier_stores
        .intersection(&atomic_uses)
        .copied()
        .collect()
}

fn pointer_source_map(ctx: &Ctx, entry_idx: usize) -> HashMap<Word, Word> {
    let mut sources = HashMap::new();
    for block in &ctx.module.functions[entry_idx].blocks {
        for inst in &block.instructions {
            if matches!(
                inst.class.opcode,
                Op::AccessChain
                    | Op::InBoundsAccessChain
                    | Op::PtrAccessChain
                    | Op::Bitcast
                    | Op::CopyObject
            ) {
                if let (Some(result), Some(source)) = (inst.result_id, id_ref_at(inst, 0)) {
                    sources.insert(result, source);
                }
            }
        }
    }
    sources
}

fn pointer_root(ptr: Word, sources: &HashMap<Word, Word>) -> Option<Word> {
    let mut cur = ptr;
    let mut seen = HashSet::new();
    while seen.insert(cur) {
        match sources.get(&cur).copied() {
            Some(next) => cur = next,
            None => return Some(cur),
        }
    }
    None
}

fn id_ref_at(inst: &Instruction, index: usize) -> Option<Word> {
    match inst.operands.get(index) {
        Some(Operand::IdRef(id)) => Some(*id),
        _ => None,
    }
}

fn is_atomic_op(op: Op) -> bool {
    matches!(
        op,
        Op::AtomicLoad
            | Op::AtomicStore
            | Op::AtomicExchange
            | Op::AtomicCompareExchange
            | Op::AtomicCompareExchangeWeak
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
    )
}
