//! Lower the stable AGX2 physical-cluster lane-number ABI to Vulkan compute builtins.

use super::*;

/// AIR generated for AGX2 acceleration-structure traversal uses physical cluster rows padded to
/// 16 lanes in X: `(x, y, z)` maps to `x + 16*ceil(local_x/16)*(y + local_y*z)`. The authored
/// dispatch already supplies the logical local dimensions through `TransformOptions`, so this
/// projection introduces no corpus- or shader-name keyed behavior.
pub(in crate::passes) fn lower_agx2_cluster_numbers(
    ctx: &mut Ctx,
    entry_idx: usize,
    stage: &Stage,
) -> Result<(), String> {
    if ctx.emit_sidecar.agx2_cluster_numbers.is_empty() {
        return Ok(());
    }
    if !matches!(stage, Stage::Kernel) {
        return Err("llvm.agx2.cluster.num is only valid in a compute kernel".to_string());
    }

    let sidecar_ids = ctx
        .emit_sidecar
        .agx2_cluster_numbers
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    // Producer-side inlining clones a fact for each call site, while the original helper fact can
    // outlive the helper body until final cleanup. Only sentinels now resident in the entry need an
    // interface projection; a fact whose defining helper was pruned is intentionally stale.
    let entry_results = ctx.module.functions[entry_idx]
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .filter_map(|instruction| instruction.result_id)
        .collect::<HashSet<_>>();
    let marker_ids = sidecar_ids
        .intersection(&entry_results)
        .copied()
        .collect::<HashSet<_>>();
    if marker_ids.is_empty() {
        ctx.emit_sidecar.agx2_cluster_numbers.clear();
        return Ok(());
    }
    let uint = ctx.ty_uint();
    let v3uint = ctx.ty_vec_uint(3);
    let local_id_var = stage_input::bind_kernel_v3uint_builtin(ctx, BuiltIn::LocalInvocationId);
    let padded_x = ctx.kernel_local_size[0].div_ceil(16) * 16;
    let stride_x = ctx.const_uint(padded_x);
    let stride_y = ctx.const_uint(ctx.kernel_local_size[1]);

    let mut found = HashSet::new();
    let mut blocks = std::mem::take(&mut ctx.module.functions[entry_idx].blocks);
    for block in &mut blocks {
        let old = std::mem::take(&mut block.instructions);
        let mut rewritten = Vec::with_capacity(old.len() + marker_ids.len() * 7);
        for instruction in old {
            let Some(result) = instruction.result_id.filter(|id| marker_ids.contains(id)) else {
                rewritten.push(instruction);
                continue;
            };
            if instruction.class.opcode != Op::CopyObject || instruction.result_type != Some(uint) {
                return Err(format!(
                    "llvm.agx2.cluster.num sentinel %{result} changed before interface lowering"
                ));
            }
            found.insert(result);

            let local_id = ctx.module.fresh_id();
            let x = ctx.module.fresh_id();
            let y = ctx.module.fresh_id();
            let z = ctx.module.fresh_id();
            let z_rows = ctx.module.fresh_id();
            let row = ctx.module.fresh_id();
            let row_offset = ctx.module.fresh_id();
            rewritten.extend([
                Instruction::new(
                    Op::Load,
                    Some(v3uint),
                    Some(local_id),
                    vec![Operand::IdRef(local_id_var)],
                ),
                composite_extract(uint, x, local_id, 0),
                composite_extract(uint, y, local_id, 1),
                composite_extract(uint, z, local_id, 2),
                binary(Op::IMul, uint, z_rows, z, stride_y),
                binary(Op::IAdd, uint, row, y, z_rows),
                binary(Op::IMul, uint, row_offset, row, stride_x),
                binary(Op::IAdd, uint, result, x, row_offset),
            ]);
        }
        block.instructions = rewritten;
    }
    ctx.module.functions[entry_idx].blocks = blocks;

    debug_assert_eq!(found, marker_ids);
    ctx.emit_sidecar.agx2_cluster_numbers.clear();
    Ok(())
}

fn composite_extract(ty: Word, result: Word, composite: Word, index: u32) -> Instruction {
    Instruction::new(
        Op::CompositeExtract,
        Some(ty),
        Some(result),
        vec![Operand::IdRef(composite), Operand::LiteralBit32(index)],
    )
}

fn binary(op: Op, ty: Word, result: Word, lhs: Word, rhs: Word) -> Instruction {
    Instruction::new(
        op,
        Some(ty),
        Some(result),
        vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
    )
}
