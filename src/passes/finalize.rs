//! Entry-point and execution-mode synthesis.

use super::*;
use module_cleanup::{
    add_needed_capabilities, drop_dangling_debug, drop_dead_unreferenced_variables,
    drop_unused_int64_capability, drop_unused_variable_pointer_capabilities,
    function_referenced_ids, gc_dead_globals,
};

pub(in crate::passes) fn finalize(
    ctx: &mut Ctx,
    entry_idx: usize,
    stage: &Stage,
) -> Result<(), String> {
    // The entry must be `void main()` with FunctionControl None. The backend gave it the AIR return
    // type + `Const` control; retype it now that the return value goes to Output variables. Create the
    // void types BEFORE we drain new_globals into the module.
    let void = ctx.ty_void();
    let fn_void = ctx.ty_fn_void(void);
    {
        let def = ctx.module.functions[entry_idx]
            .def
            .as_mut()
            .ok_or("finalize: entry function has no OpFunction def")?;
        def.result_type = Some(void);
        // OpFunction operands: [FunctionControl, IdRef(function_type)].
        if let Some(Operand::FunctionControl(_)) = def.operands.first() {
            def.operands[0] = Operand::FunctionControl(FunctionControl::NONE);
        }
        if def.operands.get(1).is_some() {
            def.operands[1] = Operand::IdRef(fn_void);
        }
    }

    // Append synthesized globals. Every builder emits its dependencies before its users.
    let globals = std::mem::take(&mut ctx.new_globals);
    ctx.module.types_global_values.extend(globals);
    rewrite_resource_query_selects(ctx)?;
    let globals = std::mem::take(&mut ctx.new_globals);
    ctx.module.types_global_values.extend(globals);

    let entry_id = ctx.module.functions[entry_idx]
        .def
        .as_ref()
        .and_then(|d| d.result_id)
        .ok_or("finalize: entry function def has no result id")?;
    let exec_model = match stage {
        Stage::Vertex => spirv::ExecutionModel::Vertex,
        Stage::Fragment => spirv::ExecutionModel::Fragment,
        Stage::Kernel => spirv::ExecutionModel::GLCompute,
    };
    let mut ep_operands = vec![
        Operand::ExecutionModel(exec_model),
        Operand::IdRef(entry_id),
        Operand::LiteralString("main".into()),
    ];
    let referenced_from_functions = function_referenced_ids(&ctx.module);
    let mut interface = Vec::new();
    let mut interface_ids = HashSet::new();
    for &id in &ctx.interface {
        if referenced_from_functions.contains(&id) && interface_ids.insert(id) {
            interface.push(id);
        }
    }
    for inst in &ctx.module.types_global_values {
        if inst.class.opcode == Op::Variable
            && inst.operands.first() == Some(&Operand::StorageClass(StorageClass::Workgroup))
        {
            if let Some(id) = inst.result_id {
                if referenced_from_functions.contains(&id) && interface_ids.insert(id) {
                    interface.push(id);
                }
            }
        }
    }
    for id in interface {
        ep_operands.push(Operand::IdRef(id));
    }
    ctx.module
        .entry_points
        .push(Instruction::new(Op::EntryPoint, None, None, ep_operands));

    if matches!(stage, Stage::Fragment) {
        ctx.module.execution_modes.push(Instruction::new(
            Op::ExecutionMode,
            None,
            None,
            vec![
                Operand::IdRef(entry_id),
                Operand::ExecutionMode(spirv::ExecutionMode::OriginUpperLeft),
            ],
        ));
        if ctx.writes_frag_depth {
            ctx.module.execution_modes.push(Instruction::new(
                Op::ExecutionMode,
                None,
                None,
                vec![
                    Operand::IdRef(entry_id),
                    Operand::ExecutionMode(spirv::ExecutionMode::DepthReplacing),
                ],
            ));
        }
    }
    if matches!(stage, Stage::Kernel) {
        let [x, y, z] = ctx.kernel_local_size;
        ctx.module.execution_modes.push(Instruction::new(
            Op::ExecutionMode,
            None,
            None,
            vec![
                Operand::IdRef(entry_id),
                Operand::ExecutionMode(spirv::ExecutionMode::LocalSize),
                Operand::LiteralBit32(x),
                Operand::LiteralBit32(y),
                Operand::LiteralBit32(z),
            ],
        ));
    }
    let air_ids: HashSet<Word> = air_names(&ctx.module).keys().copied().collect();
    ctx.module.functions.retain(|function| {
        let is_decl = function.blocks.is_empty();
        let id = function
            .def
            .as_ref()
            .and_then(|definition| definition.result_id);
        !(is_decl && id.is_some_and(|id| air_ids.contains(&id)))
    });
    ctx.module.functions.retain(|function| {
        !function.blocks.is_empty()
            || function
                .def
                .as_ref()
                .and_then(|definition| definition.result_id)
                .is_none_or(|id| !air_ids.contains(&id))
    });
    ctx.module.debug_names.retain(|instruction| {
        !matches!(
            (instruction.class.opcode, instruction.operands.first()),
            (Op::Name, Some(Operand::IdRef(id))) if air_ids.contains(id)
        )
    });

    drop_dead_unreferenced_variables(ctx, &referenced_from_functions, &interface_ids);
    drop_dangling_debug(ctx);
    gc_dead_globals(ctx);
    drop_unused_int64_capability(ctx);
    drop_unused_variable_pointer_capabilities(ctx);
    add_needed_capabilities(ctx);

    Ok(())
}
