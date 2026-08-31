//! Entry-point and execution-mode synthesis.

use super::*;
use module_cleanup::{
    add_needed_capabilities, drop_dead_unreferenced_variables, drop_unused_int64_capability,
    drop_unused_variable_pointer_capabilities, function_referenced_ids, gc_dead_globals,
};

pub(in crate::passes) fn finalize(
    ctx: &mut Ctx,
    entry_idx: usize,
    stage: &Stage,
    vert: Option<&VertMeta>,
) -> Result<(), String> {
    // The entry must be `void main()` with FunctionControl None. The backend gave it the AIR return
    // type + `Const` control; retype it now that the return value goes to Output variables. Create the
    // void types BEFORE we drain new_globals into the module.
    let void = ctx.ty_void();
    let fn_void = ctx.ty_fn_void(void);
    if matches!(stage, Stage::Kernel)
        && !matches!(
            ctx.kernel_dispatch,
            crate::reflect::KernelDispatch::Workgroups
        )
    {
        ctx.kernel_workgroup_size_id();
    }
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
    ctx.module.types_global_values.append(&mut ctx.new_globals);
    rewrite_resource_query_selects(ctx)?;
    ctx.module.types_global_values.append(&mut ctx.new_globals);
    // The two producer streams are now one owned graph. Recompute the allocator floor before the
    // native closure constructors append pointer/image types so none can reuse an id that was
    // reserved while still staged in `new_globals`.
    ctx.module.sync_id_bound_from_instructions();
    // Resource-query construction is the last producer of descriptor-backed pointer-select
    // closures. Close them in the Logical value domain, construct any remaining address-domain
    // closure, then construct opaque image selections that value replay exposes. This is the final
    // resource-value production boundary: later work computes liveness and layout but cannot create
    // another descriptor-backed pointer or image selection.
    let _ = crate::native::construct_interface_cross_binding_pointer_values_module(&mut ctx.module);
    if !ctx.emit_sidecar.all_device_buffers_raw
        || ctx.emit_sidecar.construct_cross_binding_addresses
    {
        if let Some(address_table) =
            crate::native::construct_interface_cross_binding_pointer_merges_module(
                &mut ctx.module,
                ctx.descriptor_layout,
            )
        {
            ctx.interface_buffer_var(address_table);
        }
    }
    crate::native::construct_opaque_image_selects_module(&mut ctx.module);

    let entry_id = ctx.module.functions[entry_idx]
        .def
        .as_ref()
        .and_then(|d| d.result_id)
        .ok_or("finalize: entry function def has no result id")?;
    let tessellation = matches!(stage, Stage::Vertex)
        .then(|| vert.and_then(|meta| meta.tessellation.as_ref()))
        .flatten();
    let exec_model = match (stage, tessellation) {
        (Stage::Vertex, Some(_)) => spirv::ExecutionModel::TessellationEvaluation,
        (Stage::Vertex, None) => spirv::ExecutionModel::Vertex,
        (Stage::Fragment, _) => spirv::ExecutionModel::Fragment,
        (Stage::Kernel, _) => spirv::ExecutionModel::GLCompute,
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

    if let Some(tessellation) = tessellation {
        use crate::meta::PatchDomain;
        let domain = match tessellation.domain {
            PatchDomain::Triangle => spirv::ExecutionMode::Triangles,
            PatchDomain::Quad => spirv::ExecutionMode::Quads,
            PatchDomain::Isoline => spirv::ExecutionMode::Isolines,
        };
        for mode in [domain, spirv::ExecutionMode::SpacingEqual] {
            ctx.module.execution_modes.push(Instruction::new(
                Op::ExecutionMode,
                None,
                None,
                vec![Operand::IdRef(entry_id), Operand::ExecutionMode(mode)],
            ));
        }
        if !matches!(tessellation.domain, PatchDomain::Isoline) {
            ctx.module.execution_modes.push(Instruction::new(
                Op::ExecutionMode,
                None,
                None,
                vec![
                    Operand::IdRef(entry_id),
                    Operand::ExecutionMode(spirv::ExecutionMode::VertexOrderCcw),
                ],
            ));
        }
    }

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
        if ctx.uses_fragment_imageblock {
            let capability = spirv::Capability::FragmentShaderPixelInterlockEXT;
            if !ctx.module.capabilities.iter().any(|instruction| {
                instruction.operands.as_slice() == [Operand::Capability(capability)]
            }) {
                ctx.module.capabilities.push(Instruction::new(
                    Op::Capability,
                    None,
                    None,
                    vec![Operand::Capability(capability)],
                ));
            }
            if !ctx.module.extensions.iter().any(|instruction| {
                instruction.operands.first()
                    == Some(&Operand::LiteralString(
                        "SPV_EXT_fragment_shader_interlock".to_string(),
                    ))
            }) {
                ctx.module.extensions.push(Instruction::new(
                    Op::Extension,
                    None,
                    None,
                    vec![Operand::LiteralString(
                        "SPV_EXT_fragment_shader_interlock".to_string(),
                    )],
                ));
            }
            ctx.module.execution_modes.push(Instruction::new(
                Op::ExecutionMode,
                None,
                None,
                vec![
                    Operand::IdRef(entry_id),
                    Operand::ExecutionMode(spirv::ExecutionMode::PixelInterlockOrderedEXT),
                ],
            ));
        }
    }
    if matches!(stage, Stage::Kernel)
        && matches!(
            ctx.kernel_dispatch,
            crate::reflect::KernelDispatch::Workgroups
        )
    {
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
    ctx.module.functions.retain(|function| {
        !function.blocks.is_empty()
            || function
                .def
                .as_ref()
                .and_then(|definition| definition.result_id)
                .is_some_and(|id| referenced_from_functions.contains(&id))
    });
    ctx.module.debug_names.retain(|instruction| {
        !matches!(
            (instruction.class.opcode, instruction.operands.first()),
            (Op::Name, Some(Operand::IdRef(id))) if air_ids.contains(id)
        )
    });

    drop_dead_unreferenced_variables(ctx, &referenced_from_functions, &interface_ids);
    gc_dead_globals(ctx);
    drop_unused_int64_capability(ctx);
    let variable_pointer_requirements = drop_unused_variable_pointer_capabilities(ctx);
    add_needed_capabilities(ctx, variable_pointer_requirements);
    order_module_scope_dependencies(&mut ctx.module)?;

    Ok(())
}

/// Put every module-scope definition before its users while preserving source order wherever no
/// dependency requires movement. Rewrites may point an existing aggregate or variable at a type
/// synthesized later in `new_globals`; finalization owns merging those two construction streams.
fn order_module_scope_dependencies(module: &mut Module) -> Result<(), String> {
    let definitions = module
        .types_global_values
        .iter()
        .enumerate()
        .filter_map(|(index, instruction)| instruction.result_id.map(|id| (id, index)))
        .collect::<HashMap<_, _>>();
    let forward_pointers = module
        .types_global_values
        .iter()
        .filter(|instruction| instruction.class.opcode == Op::TypeForwardPointer)
        .filter_map(|instruction| match instruction.operands.first() {
            Some(Operand::IdRef(id)) => Some(*id),
            _ => None,
        })
        .collect::<HashSet<_>>();
    let mut dependencies = vec![Vec::new(); module.types_global_values.len()];
    for (index, instruction) in module.types_global_values.iter().enumerate() {
        let ids =
            instruction
                .result_type
                .into_iter()
                .chain(
                    instruction
                        .operands
                        .iter()
                        .filter_map(|operand| match operand {
                            Operand::IdRef(id)
                            | Operand::IdMemorySemantics(id)
                            | Operand::IdScope(id) => Some(*id),
                            _ => None,
                        }),
                );
        for id in ids {
            if forward_pointers.contains(&id) {
                continue;
            }
            if let Some(&dependency) = definitions.get(&id) {
                if dependency != index {
                    dependencies[index].push(dependency);
                }
            }
        }
        dependencies[index].sort_unstable();
        dependencies[index].dedup();
    }

    fn visit(
        index: usize,
        dependencies: &[Vec<usize>],
        state: &mut [u8],
        order: &mut Vec<usize>,
    ) -> Result<(), String> {
        match state[index] {
            2 => return Ok(()),
            1 => {
                return Err(format!(
                    "module-scope definitions contain a dependency cycle at instruction {index}"
                ));
            }
            _ => {}
        }
        state[index] = 1;
        for &dependency in &dependencies[index] {
            visit(dependency, dependencies, state, order)?;
        }
        state[index] = 2;
        order.push(index);
        Ok(())
    }

    let mut state = vec![0u8; dependencies.len()];
    let mut order = Vec::with_capacity(dependencies.len());
    for index in 0..dependencies.len() {
        visit(index, &dependencies, &mut state, &mut order)?;
    }
    let mut ranks = vec![0usize; order.len()];
    for (rank, index) in order.into_iter().enumerate() {
        ranks[index] = rank;
    }
    let original_indices = module
        .types_global_values
        .iter()
        .enumerate()
        .filter_map(|(index, instruction)| instruction.result_id.map(|id| (id, index)))
        .collect::<HashMap<_, _>>();
    let forward_ranks = module
        .types_global_values
        .iter()
        .enumerate()
        .filter_map(|(index, instruction)| {
            (instruction.class.opcode == Op::TypeForwardPointer)
                .then(|| match instruction.operands.first() {
                    Some(Operand::IdRef(id)) => Some((*id, ranks[index])),
                    _ => None,
                })
                .flatten()
        })
        .collect::<HashMap<_, _>>();
    module.types_global_values.sort_by_key(|instruction| {
        instruction
            .result_id
            .and_then(|id| original_indices.get(&id).copied())
            .map(|index| ranks[index])
            .or_else(|| {
                (instruction.class.opcode == Op::TypeForwardPointer)
                    .then(|| match instruction.operands.first() {
                        Some(Operand::IdRef(id)) => forward_ranks.get(id).copied(),
                        _ => None,
                    })
                    .flatten()
            })
            .unwrap_or(usize::MAX)
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finalization_orders_late_aggregate_type_dependencies_before_existing_users() {
        let mut module = Module::new();
        module.types_global_values = vec![
            Instruction::new(Op::TypeStruct, None, Some(1), vec![Operand::IdRef(4)]),
            Instruction::new(
                Op::TypeInt,
                None,
                Some(2),
                vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
            ),
            Instruction::new(
                Op::Constant,
                Some(2),
                Some(3),
                vec![Operand::LiteralBit32(11)],
            ),
            Instruction::new(
                Op::TypeArray,
                None,
                Some(4),
                vec![Operand::IdRef(2), Operand::IdRef(3)],
            ),
        ];

        order_module_scope_dependencies(&mut module).unwrap();

        assert_eq!(
            module
                .types_global_values
                .iter()
                .filter_map(|instruction| instruction.result_id)
                .collect::<Vec<_>>(),
            vec![2, 3, 4, 1]
        );
    }
}
