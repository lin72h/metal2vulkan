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

    let mut null_by_pointee: HashMap<Word, Word> = HashMap::new();
    let mut prologue = Vec::new();
    for (var, ptr_ty) in vars {
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
