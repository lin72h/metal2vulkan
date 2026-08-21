//! One kernel-grid source shared by AIR `[[threads_per_grid]]` and dispatch-tail culling.

use super::*;

pub(crate) const UNSAFE_DISPATCH_BARRIER_ERROR: &str =
    "non-uniform dispatchThreads grids with source control barriers are unsupported because surplus Vulkan invocations cannot return before a barrier";

pub(crate) fn kernel_dispatch_guard_required(
    dispatch: crate::reflect::KernelDispatch,
    local_size: [u32; 3],
) -> bool {
    match dispatch {
        crate::reflect::KernelDispatch::Workgroups => false,
        crate::reflect::KernelDispatch::ThreadsFixed { threads_per_grid } => threads_per_grid
            .into_iter()
            .zip(local_size)
            .any(|(grid, local)| grid % local != 0),
        crate::reflect::KernelDispatch::ThreadsPushConstant { .. } => true,
    }
}

/// Declare the three-scalar push-constant block once. Keeping the dimensions as separate `u32`
/// members gives the public ABI an exact 12-byte range without inheriting a `uvec3`'s 16-byte
/// aggregate alignment.
pub(in crate::passes) fn bind_kernel_grid_push_constant_once(
    ctx: &mut Ctx,
    slot: &mut Option<Word>,
    offset: u32,
) -> Word {
    if let Some(var) = *slot {
        return var;
    }
    let uint_ty = ctx.ty_uint();
    let block_ty = ctx.module.fresh_id();
    ctx.new_globals.push(type_inst(
        Op::TypeStruct,
        block_ty,
        vec![
            Operand::IdRef(uint_ty),
            Operand::IdRef(uint_ty),
            Operand::IdRef(uint_ty),
        ],
    ));
    ctx.module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(block_ty),
            Operand::Decoration(Decoration::Block),
        ],
    ));
    for member in 0..3 {
        ctx.module.annotations.push(Instruction::new(
            Op::MemberDecorate,
            None,
            None,
            vec![
                Operand::IdRef(block_ty),
                Operand::LiteralBit32(member),
                Operand::Decoration(Decoration::Offset),
                Operand::LiteralBit32(offset + member * 4),
            ],
        ));
    }
    let pointer_ty = ctx.ty_ptr(StorageClass::PushConstant, block_ty);
    let var = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::Variable,
        Some(pointer_ty),
        Some(var),
        vec![Operand::StorageClass(StorageClass::PushConstant)],
    ));
    ctx.module.debug_names.push(Instruction::new(
        Op::Name,
        None,
        None,
        vec![
            Operand::IdRef(var),
            Operand::LiteralString("metal2vulkan_threads_per_grid".into()),
        ],
    ));
    ctx.interface.push(var);
    *slot = Some(var);
    var
}

fn load_push_constant_component(
    ctx: &mut Ctx,
    instructions: &mut Vec<Instruction>,
    var: Word,
    member: u32,
) -> Word {
    let uint_ty = ctx.ty_uint();
    let pointer_ty = ctx.ty_ptr(StorageClass::PushConstant, uint_ty);
    let member_index = ctx.const_uint(member);
    let pointer = ctx.module.fresh_id();
    instructions.push(Instruction::new(
        Op::AccessChain,
        Some(pointer_ty),
        Some(pointer),
        vec![Operand::IdRef(var), Operand::IdRef(member_index)],
    ));
    let value = ctx.module.fresh_id();
    instructions.push(Instruction::new(
        Op::Load,
        Some(uint_ty),
        Some(value),
        vec![Operand::IdRef(pointer)],
    ));
    value
}

/// Materialize the selected prefix of the dynamic grid in the AIR parameter's integer type.
pub(in crate::passes) fn materialize_kernel_grid_push_constant(
    ctx: &mut Ctx,
    instructions: &mut Vec<Instruction>,
    var: Word,
    out_ty: Word,
    lanes: u32,
) -> Result<Word, String> {
    if !(1..=3).contains(&lanes) {
        return Err(format!(
            "kernel threads_per_grid parameter has unsupported lane count {lanes}"
        ));
    }
    let uint_ty = ctx.ty_uint();
    let components = (0..lanes)
        .map(|member| load_push_constant_component(ctx, instructions, var, member))
        .collect::<Vec<_>>();
    let value = if lanes == 1 {
        components[0]
    } else {
        let vector_ty = ctx.ty_vec_uint(lanes);
        let vector = ctx.module.fresh_id();
        instructions.push(Instruction::new(
            Op::CompositeConstruct,
            Some(vector_ty),
            Some(vector),
            components.into_iter().map(Operand::IdRef).collect(),
        ));
        vector
    };
    let value_ty = if lanes == 1 {
        uint_ty
    } else {
        ctx.ty_vec_uint(lanes)
    };
    if value_ty == out_ty {
        return Ok(value);
    }
    let converted = ctx.module.fresh_id();
    instructions.push(Instruction::new(
        Op::UConvert,
        Some(out_ty),
        Some(converted),
        vec![Operand::IdRef(value)],
    ));
    Ok(converted)
}

/// Insert a structured early-return cull for Metal `dispatchThreads` tail invocations.
///
/// This runs after AIR calls have lowered, so every source workgroup barrier is visible. Returning
/// only the surplus Vulkan lanes before such a barrier would violate its uniform participation
/// contract; that shape remains an honest fallback unless a fixed grid proves no cull is needed.
pub(in crate::passes) fn insert_kernel_dispatch_guard(
    ctx: &mut Ctx,
    entry_idx: usize,
) -> Result<(), String> {
    let compared_dimensions = match ctx.kernel_dispatch {
        crate::reflect::KernelDispatch::Workgroups => return Ok(()),
        crate::reflect::KernelDispatch::ThreadsFixed { threads_per_grid } => threads_per_grid
            .into_iter()
            .zip(ctx.kernel_local_size)
            .enumerate()
            .filter_map(|(dimension, (grid, local))| {
                (grid % local != 0).then_some((dimension as u32, Some(grid)))
            })
            .collect::<Vec<_>>(),
        crate::reflect::KernelDispatch::ThreadsPushConstant { .. } => {
            (0..3).map(|dimension| (dimension, None)).collect()
        }
    };
    if compared_dimensions.is_empty() {
        return Ok(());
    }
    if ctx.module.functions[entry_idx]
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .any(|instruction| instruction.class.opcode == Op::ControlBarrier)
    {
        return Err(UNSAFE_DISPATCH_BARRIER_ERROR.to_string());
    }

    let global_id_var = ctx
        .kernel_dispatch_global_invocation_id_var
        .ok_or_else(|| {
            "dispatchThreads culling requires a GlobalInvocationId interface variable".to_string()
        })?;
    let push_constant_var = ctx.kernel_grid_push_constant_var;
    let uint_ty = ctx.ty_uint();
    let bool_ty = ctx.ty_bool();
    let v3uint_ty = ctx.ty_vec_uint(3);
    let global_id = ctx.module.fresh_id();
    let mut guard = vec![Instruction::new(
        Op::Load,
        Some(v3uint_ty),
        Some(global_id),
        vec![Operand::IdRef(global_id_var)],
    )];
    let mut outside = None;
    for (dimension, fixed_bound) in compared_dimensions {
        let invocation = ctx.module.fresh_id();
        guard.push(Instruction::new(
            Op::CompositeExtract,
            Some(uint_ty),
            Some(invocation),
            vec![Operand::IdRef(global_id), Operand::LiteralBit32(dimension)],
        ));
        let bound = match fixed_bound {
            Some(bound) => ctx.const_uint(bound),
            None => load_push_constant_component(
                ctx,
                &mut guard,
                push_constant_var.ok_or_else(|| {
                    "dynamic dispatchThreads culling has no push-constant grid variable".to_string()
                })?,
                dimension,
            ),
        };
        let comparison = ctx.module.fresh_id();
        guard.push(Instruction::new(
            Op::UGreaterThanEqual,
            Some(bool_ty),
            Some(comparison),
            vec![Operand::IdRef(invocation), Operand::IdRef(bound)],
        ));
        outside = Some(match outside {
            None => comparison,
            Some(previous) => {
                let combined = ctx.module.fresh_id();
                guard.push(Instruction::new(
                    Op::LogicalOr,
                    Some(bool_ty),
                    Some(combined),
                    vec![Operand::IdRef(previous), Operand::IdRef(comparison)],
                ));
                combined
            }
        });
    }
    let outside = outside.ok_or_else(|| "dispatchThreads guard has no comparisons".to_string())?;

    let prologue_label = ctx.module.fresh_id();
    let return_label = ctx.module.fresh_id();
    let function = &mut ctx.module.functions[entry_idx];
    let original = function
        .blocks
        .first_mut()
        .ok_or_else(|| "kernel entry has no block for dispatchThreads culling".to_string())?;
    let original_label = original
        .label
        .as_ref()
        .and_then(|label| label.result_id)
        .ok_or_else(|| "kernel entry block has no label".to_string())?;
    let variable_count = original
        .instructions
        .iter()
        .take_while(|instruction| instruction.class.opcode == Op::Variable)
        .count();
    let variables = original
        .instructions
        .drain(..variable_count)
        .collect::<Vec<_>>();

    let mut prologue_instructions = variables;
    prologue_instructions.append(&mut guard);
    prologue_instructions.push(Instruction::new(
        Op::SelectionMerge,
        None,
        None,
        vec![
            Operand::IdRef(original_label),
            Operand::SelectionControl(spirv::SelectionControl::NONE),
        ],
    ));
    prologue_instructions.push(Instruction::new(
        Op::BranchConditional,
        None,
        None,
        vec![
            Operand::IdRef(outside),
            Operand::IdRef(return_label),
            Operand::IdRef(original_label),
        ],
    ));
    function.blocks.insert(
        0,
        Block {
            label: Some(Instruction::new(
                Op::Label,
                None,
                Some(prologue_label),
                vec![],
            )),
            instructions: prologue_instructions,
        },
    );
    function.blocks.insert(
        1,
        Block {
            label: Some(Instruction::new(
                Op::Label,
                None,
                Some(return_label),
                vec![],
            )),
            instructions: vec![Instruction::new(Op::Return, None, None, vec![])],
        },
    );
    Ok(())
}
