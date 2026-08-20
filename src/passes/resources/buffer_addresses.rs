use super::*;

/// Replace native-emitter buffer-address facts with loads from one reflected address-table
/// StorageBuffer. The runtime fills table[Metal buffer location] with vkGetBufferDeviceAddress.
pub(in crate::passes) fn lower_buffer_address_facts(
    ctx: &mut Ctx,
    entry_idx: usize,
    kern: Option<&KernMeta>,
) -> Result<(), String> {
    let address_words = ctx
        .emit_sidecar
        .buffer_address_words
        .iter()
        .map(|fact| (fact.id, (fact.param_index, fact.component)))
        .collect::<HashMap<_, _>>();
    if address_words.is_empty() {
        return Ok(());
    }
    let kern = kern.ok_or("buffer-address facts require kernel metadata")?;
    let mut locations = HashMap::new();
    for (id, (param_index, component)) in &address_words {
        let Some(
            KernRole::Buffer(location)
            | KernRole::AccelerationStructureShadow(location)
            | KernRole::PrimitiveAccelerationStructureShadow(location),
        ) = kern.role_of(*param_index)
        else {
            return Err(format!(
                "buffer-address fact {id} references a kernel parameter without a buffer-backed resource {param_index}"
            ));
        };
        locations.insert(*id, (*location, *component));
    }

    let uint_ty = ctx.ty_uint();
    let vec_ty = ctx.ty_vec_uint(2);
    let array_ty = ctx.module.fresh_id();
    ctx.new_globals.push(type_inst(
        Op::TypeRuntimeArray,
        array_ty,
        vec![Operand::IdRef(vec_ty)],
    ));
    ctx.module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(array_ty),
            Operand::Decoration(Decoration::ArrayStride),
            Operand::LiteralBit32(std::mem::size_of::<u64>() as u32),
        ],
    ));
    let block_ty = ctx.module.fresh_id();
    ctx.new_globals.push(type_inst(
        Op::TypeStruct,
        block_ty,
        vec![Operand::IdRef(array_ty)],
    ));
    ctx.module.annotations.push(Instruction::new(
        Op::MemberDecorate,
        None,
        None,
        vec![
            Operand::IdRef(block_ty),
            Operand::LiteralBit32(0),
            Operand::Decoration(Decoration::Offset),
            Operand::LiteralBit32(0),
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
    let block_ptr_ty = ctx.ty_ptr(StorageClass::StorageBuffer, block_ty);
    let uint_ptr_ty = ctx.ty_ptr(StorageClass::StorageBuffer, uint_ty);
    let table = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::Variable,
        Some(block_ptr_ty),
        Some(table),
        vec![Operand::StorageClass(StorageClass::StorageBuffer)],
    ));
    let layout = ctx.descriptor_layout;
    let occupied = ctx
        .module
        .annotations
        .iter()
        .filter(|inst| {
            inst.class.opcode == Op::Decorate
                && inst.operands.get(1) == Some(&Operand::Decoration(Decoration::Binding))
        })
        .filter_map(|inst| match inst.operands.get(2) {
            Some(Operand::LiteralBit32(binding)) => Some(*binding),
            _ => None,
        })
        .collect::<HashSet<_>>();
    let binding = (layout.synthetic.start..layout.synthetic.end)
        .find(|binding| !occupied.contains(binding))
        .ok_or_else(|| "descriptor binding space exhausted for buffer-address table".to_string())?;
    decorate_binding(&mut ctx.module, table, layout.set, binding);
    ctx.interface_buffer_var(table);

    let zero = ctx.const_uint(0);
    let location_constants = locations
        .values()
        .map(|(location, _)| *location)
        .collect::<HashSet<_>>()
        .into_iter()
        .map(|location| (location, ctx.const_uint(location)))
        .collect::<HashMap<_, _>>();
    let component_constants = [ctx.const_uint(0), ctx.const_uint(1)];
    let block_count = ctx.module.functions[entry_idx].blocks.len();
    for block_index in 0..block_count {
        let instructions =
            std::mem::take(&mut ctx.module.functions[entry_idx].blocks[block_index].instructions);
        let mut rewritten = Vec::with_capacity(instructions.len());
        for inst in instructions {
            let Some(result) = inst.result_id else {
                rewritten.push(inst);
                continue;
            };
            let Some((location, component)) = locations.get(&result).copied() else {
                rewritten.push(inst);
                continue;
            };
            let pointer = ctx.module.fresh_id();
            rewritten.push(Instruction::new(
                Op::AccessChain,
                Some(uint_ptr_ty),
                Some(pointer),
                vec![
                    Operand::IdRef(table),
                    Operand::IdRef(zero),
                    Operand::IdRef(location_constants[&location]),
                    Operand::IdRef(component_constants[component as usize]),
                ],
            ));
            rewritten.push(Instruction::new(
                Op::Load,
                Some(uint_ty),
                Some(result),
                vec![Operand::IdRef(pointer)],
            ));
        }
        ctx.module.functions[entry_idx].blocks[block_index].instructions = rewritten;
    }
    ctx.emit_sidecar.buffer_address_words.clear();
    Ok(())
}
