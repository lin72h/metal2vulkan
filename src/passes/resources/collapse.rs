//! Apply interface bindings and collapse their resource wrappers into final resource values.

use super::rewrites::*;
use super::*;
use crate::passes::stage_input::{const_ivec, BufWrap, ParamBinding};

/// Apply the per-param bindings to the entry function body:
///  - remove all OpFunctionParameter,
///  - for LoadVar/Sampler params: insert an OpLoad at function entry, splice its id for the param id,
///  - for Image params: insert an OpLoad of the image at entry, splice for the param id,
///  - for Buffer params: splice the variable id for the param id, then rewrite buffer-derived
///    pointer storage classes (UniformConstant -> StorageBuffer),
///  - for WorkgroupMemory params: splice the variable id and rewrite derived pointers to Workgroup,
///  - for ZeroValue params: splice the undef id.
pub(in crate::passes) fn apply_bindings(
    ctx: &mut Ctx,
    entry_idx: usize,
    bindings: Vec<(Word, ParamBinding)>,
    buffer_structs: &[(Word, Word)],
    defs: &HashMap<Word, Instruction>,
) -> Result<(), String> {
    // Prepare loads to inject at the top of the entry block, and id splices to perform.
    let mut loads: Vec<Instruction> = vec![];
    let mut splices: Vec<(Word, Word)> = vec![];
    let mut resource_values: HashSet<Word> = HashSet::new();
    let mut buffer_param_ids: Vec<Word> = vec![];
    // Collapsed buffers (RuntimeArray-wrapped arrays + reconstructed structs): (param id, var,
    // block type, prepend_member0). Rewritten after the generic splices, since their uses need
    // per-use handling (re-root chains vs route direct loads through the offset-0 leaf).
    let mut collapsed_buffers: Vec<(Word, Word, Word, bool)> = vec![];
    // Struct buffers with implicit record indexing need per-chain handling because some chains are
    // record-0 member paths while others carry a real record index as their first operand.
    let mut record_array_buffers: Vec<(Word, Word, Word, Word)> = vec![];
    // Direct record-member GEPs can bridge a padded AIR struct pointer to its compact metadata
    // layout. Descendant helper GEPs still carry AIR member ordinals and need one offset remap.
    let mut nested_air_ordinal_roots = HashSet::new();
    // Unmodeled pointer params bound to Private zero vars: their derived access chains keep the
    // backend's (UniformConstant) element-pointer storage class and must be re-classed to Private.
    let mut zero_pointer_vars: Vec<Word> = vec![];
    // Threadgroup memory params are true Workgroup variables, not resource descriptors.
    let mut workgroup_vars: Vec<Word> = vec![];

    for (pid, b) in bindings {
        match b {
            ParamBinding::LoadVar { var, ty } => {
                let lid = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::Load,
                    Some(ty),
                    Some(lid),
                    vec![Operand::IdRef(var)],
                ));
                splices.push((pid, lid));
            }
            ParamBinding::LoadVarBoolFromUint { var, bool_ty } => {
                let uint_ty = ctx.ty_uint();
                let lid = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::Load,
                    Some(uint_ty),
                    Some(lid),
                    vec![Operand::IdRef(var)],
                ));
                let converted = ctx.module.fresh_id();
                let zero = ctx.const_uint(0);
                loads.push(Instruction::new(
                    Op::INotEqual,
                    Some(bool_ty),
                    Some(converted),
                    vec![Operand::IdRef(lid), Operand::IdRef(zero)],
                ));
                splices.push((pid, converted));
            }
            ParamBinding::LoadVarConverted {
                var,
                load_ty,
                param_ty,
            } => {
                // load the 32-bit builtin, then UConvert down to the narrower param int type.
                let lid = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::Load,
                    Some(load_ty),
                    Some(lid),
                    vec![Operand::IdRef(var)],
                ));
                let cid = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::UConvert,
                    Some(param_ty),
                    Some(cid),
                    vec![Operand::IdRef(lid)],
                ));
                splices.push((pid, cid));
            }
            ParamBinding::LoadVarBitcast {
                var,
                load_ty,
                param_ty,
            } => {
                let lid = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::Load,
                    Some(load_ty),
                    Some(lid),
                    vec![Operand::IdRef(var)],
                ));
                let bid = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::Bitcast,
                    Some(param_ty),
                    Some(bid),
                    vec![Operand::IdRef(lid)],
                ));
                splices.push((pid, bid));
            }
            ParamBinding::LoadVarBitAnd {
                var,
                load_ty,
                param_ty,
                mask,
            } => {
                let lid = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::Load,
                    Some(load_ty),
                    Some(lid),
                    vec![Operand::IdRef(var)],
                ));
                let masked = ctx.module.fresh_id();
                let mask = ctx.const_uint(mask);
                loads.push(Instruction::new(
                    Op::BitwiseAnd,
                    Some(load_ty),
                    Some(masked),
                    vec![Operand::IdRef(lid), Operand::IdRef(mask)],
                ));
                if param_ty == load_ty {
                    splices.push((pid, masked));
                } else {
                    let cid = ctx.module.fresh_id();
                    loads.push(Instruction::new(
                        Op::UConvert,
                        Some(param_ty),
                        Some(cid),
                        vec![Operand::IdRef(masked)],
                    ));
                    splices.push((pid, cid));
                }
            }
            ParamBinding::LoadVarShiftRight {
                var,
                load_ty,
                param_ty,
                shift,
            } => {
                let lid = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::Load,
                    Some(load_ty),
                    Some(lid),
                    vec![Operand::IdRef(var)],
                ));
                let shifted = ctx.module.fresh_id();
                let shift = ctx.const_uint(shift);
                loads.push(Instruction::new(
                    Op::ShiftRightLogical,
                    Some(load_ty),
                    Some(shifted),
                    vec![Operand::IdRef(lid), Operand::IdRef(shift)],
                ));
                if param_ty == load_ty {
                    splices.push((pid, shifted));
                } else {
                    let cid = ctx.module.fresh_id();
                    loads.push(Instruction::new(
                        Op::UConvert,
                        Some(param_ty),
                        Some(cid),
                        vec![Operand::IdRef(shifted)],
                    ));
                    splices.push((pid, cid));
                }
            }
            ParamBinding::LoadVarComponent {
                var,
                vec_ty,
                scalar_ty,
                out_ty,
                comp,
            } => {
                // load the builtin vector, then extract the wanted scalar component.
                let vid = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::Load,
                    Some(vec_ty),
                    Some(vid),
                    vec![Operand::IdRef(var)],
                ));
                let cid = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::CompositeExtract,
                    Some(scalar_ty),
                    Some(cid),
                    vec![Operand::IdRef(vid), Operand::LiteralBit32(comp)],
                ));
                if scalar_ty == out_ty {
                    splices.push((pid, cid));
                } else {
                    let converted = ctx.module.fresh_id();
                    loads.push(Instruction::new(
                        Op::UConvert,
                        Some(out_ty),
                        Some(converted),
                        vec![Operand::IdRef(cid)],
                    ));
                    splices.push((pid, converted));
                }
            }
            ParamBinding::LoadVarVectorPrefix {
                var,
                vec_ty,
                out_ty,
                lanes,
            } => {
                let vid = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::Load,
                    Some(vec_ty),
                    Some(vid),
                    vec![Operand::IdRef(var)],
                ));
                let scalar_ty = ctx.ty_uint();
                let prefix_ty = ctx.ty_vec_uint(lanes);
                let mut components = Vec::with_capacity(lanes as usize);
                for lane in 0..lanes {
                    let cid = ctx.module.fresh_id();
                    loads.push(Instruction::new(
                        Op::CompositeExtract,
                        Some(scalar_ty),
                        Some(cid),
                        vec![Operand::IdRef(vid), Operand::LiteralBit32(lane)],
                    ));
                    components.push(Operand::IdRef(cid));
                }
                let vector = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::CompositeConstruct,
                    Some(prefix_ty),
                    Some(vector),
                    components,
                ));
                if prefix_ty == out_ty {
                    splices.push((pid, vector));
                } else {
                    let converted = ctx.module.fresh_id();
                    loads.push(Instruction::new(
                        Op::UConvert,
                        Some(out_ty),
                        Some(converted),
                        vec![Operand::IdRef(vector)],
                    ));
                    splices.push((pid, converted));
                }
            }
            ParamBinding::LoadThreadsPerGrid {
                var,
                vec_ty,
                out_ty,
                lanes,
            } => {
                let vid = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::Load,
                    Some(vec_ty),
                    Some(vid),
                    vec![Operand::IdRef(var)],
                ));
                let uint_ty = ctx.ty_uint();
                let local_size = ctx.kernel_local_size;
                let mut components = Vec::with_capacity(lanes as usize);
                for lane in 0..lanes {
                    let workgroups = ctx.module.fresh_id();
                    loads.push(Instruction::new(
                        Op::CompositeExtract,
                        Some(uint_ty),
                        Some(workgroups),
                        vec![Operand::IdRef(vid), Operand::LiteralBit32(lane)],
                    ));
                    let product = ctx.module.fresh_id();
                    let local = ctx.const_uint(local_size[lane as usize]);
                    loads.push(Instruction::new(
                        Op::IMul,
                        Some(uint_ty),
                        Some(product),
                        vec![Operand::IdRef(workgroups), Operand::IdRef(local)],
                    ));
                    components.push(Operand::IdRef(product));
                }
                if lanes == 1 {
                    let product = match &components[0] {
                        Operand::IdRef(product) => *product,
                        _ => {
                            return Err(
                                "apply_bindings: dispatch-index component is not an id".to_string()
                            )
                        }
                    };
                    if out_ty == uint_ty {
                        splices.push((pid, product));
                    } else {
                        let converted = ctx.module.fresh_id();
                        loads.push(Instruction::new(
                            Op::UConvert,
                            Some(out_ty),
                            Some(converted),
                            vec![Operand::IdRef(product)],
                        ));
                        splices.push((pid, converted));
                    }
                } else {
                    let prefix_ty = ctx.ty_vec_uint(lanes);
                    let vector = ctx.module.fresh_id();
                    loads.push(Instruction::new(
                        Op::CompositeConstruct,
                        Some(prefix_ty),
                        Some(vector),
                        components,
                    ));
                    if prefix_ty == out_ty {
                        splices.push((pid, vector));
                    } else {
                        let converted = ctx.module.fresh_id();
                        loads.push(Instruction::new(
                            Op::UConvert,
                            Some(out_ty),
                            Some(converted),
                            vec![Operand::IdRef(vector)],
                        ));
                        splices.push((pid, converted));
                    }
                }
            }
            ParamBinding::Sampler { var } => {
                let sty = ctx.ty_sampler();
                let lid = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::Load,
                    Some(sty),
                    Some(lid),
                    vec![Operand::IdRef(var)],
                ));
                resource_values.insert(lid);
                splices.push((pid, lid));
            }
            ParamBinding::Image {
                var,
                image_ty,
                dim,
                comp,
                multisampled,
            } => {
                let lid = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::Load,
                    Some(image_ty),
                    Some(lid),
                    vec![Operand::IdRef(var)],
                ));
                ctx.image_dims.insert(lid, dim);
                ctx.image_comp.insert(lid, comp);
                if multisampled {
                    ctx.image_multisampled.insert(lid);
                }
                resource_values.insert(lid);
                splices.push((pid, lid));
            }
            ParamBinding::ImageArray {
                var,
                elem_image_ty,
                dim,
                comp,
                multisampled,
            } => {
                // Splice the param to the array VARIABLE (a pointer to `array<image>`); the per-element
                // image is materialized at each use by `materialize_texture_array_loads` reading
                // `image_array_vars`. No function-top load: the array as a whole is never an operand.
                ctx.image_array_vars
                    .insert(var, (elem_image_ty, dim, comp, multisampled));
                resource_values.insert(var);
                splices.push((pid, var));
            }
            ParamBinding::StorageImage {
                var,
                image_ty,
                dim,
                comp,
            } => {
                let lid = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::Load,
                    Some(image_ty),
                    Some(lid),
                    vec![Operand::IdRef(var)],
                ));
                ctx.image_dims.insert(lid, dim);
                ctx.image_comp.insert(lid, comp);
                ctx.image_storage.insert(lid);
                resource_values.insert(lid);
                splices.push((pid, lid));
            }
            ParamBinding::InputAttachment {
                var,
                image_ty,
                read_ty,
                param_ty,
            } => {
                let image = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::Load,
                    Some(image_ty),
                    Some(image),
                    vec![Operand::IdRef(var)],
                ));
                let coord_ty = ctx.ty_vec_sint(2);
                let coord = const_ivec(ctx, coord_ty, &[0, 0]);
                let read = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::ImageRead,
                    Some(read_ty),
                    Some(read),
                    vec![Operand::IdRef(image), Operand::IdRef(coord)],
                ));
                let value = adapt_input_attachment_read(ctx, &mut loads, read, read_ty, param_ty)?;
                splices.push((pid, value));
            }
            ParamBinding::Buffer { var, wrap } => match wrap {
                BufWrap::Direct => {
                    // struct buffer: the body's access chains index into the struct off the var.
                    splices.push((pid, var));
                    buffer_param_ids.push(var);
                }
                BufWrap::Collapsed {
                    block_ty,
                    prepend_member0,
                } => {
                    // re-rooted / leaf-routed after splices (see below).
                    collapsed_buffers.push((pid, var, block_ty, prepend_member0));
                }
                BufWrap::RecordArray { block_ty, elem_ty } => {
                    record_array_buffers.push((pid, var, block_ty, elem_ty));
                }
            },
            ParamBinding::StageInput {
                var,
                value_ty,
                index_var,
            } => {
                let v3u = ctx.ty_vec_uint(3);
                let uint_ty = ctx.ty_uint();
                let gid = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::Load,
                    Some(v3u),
                    Some(gid),
                    vec![Operand::IdRef(index_var)],
                ));
                let x = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::CompositeExtract,
                    Some(uint_ty),
                    Some(x),
                    vec![Operand::IdRef(gid), Operand::LiteralBit32(0)],
                ));
                let zero = ctx.const_uint(0);
                let ptr_ty = ctx.ty_ptr(StorageClass::StorageBuffer, value_ty);
                let ptr = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::AccessChain,
                    Some(ptr_ty),
                    Some(ptr),
                    vec![Operand::IdRef(var), Operand::IdRef(zero), Operand::IdRef(x)],
                ));
                let value = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::Load,
                    Some(value_ty),
                    Some(value),
                    vec![Operand::IdRef(ptr)],
                ));
                splices.push((pid, value));
            }
            ParamBinding::WorkgroupMemory { var } => {
                splices.push((pid, var));
                workgroup_vars.push(var);
            }
            ParamBinding::Value { val } => {
                splices.push((pid, val));
            }
            ParamBinding::ZeroValue { val } => {
                splices.push((pid, val));
            }
            ParamBinding::ZeroPointer { var } => {
                splices.push((pid, var));
                zero_pointer_vars.push(var);
            }
        }
    }

    // Remove parameters.
    ctx.module.functions[entry_idx].parameters.clear();

    // Inject loads at the top of the first block, but AFTER any leading function-local OpVariables:
    // SPIR-V requires every OpVariable in a function to be among the first instructions of the entry
    // block, so the load (a non-Variable instruction) must follow them. Order among the loads is
    // preserved.
    if !loads.is_empty() {
        let func = &mut ctx.module.functions[entry_idx];
        if let Some(first) = func.blocks.first_mut() {
            let at = first
                .instructions
                .iter()
                .position(|i| i.class.opcode != Op::Variable)
                .unwrap_or(first.instructions.len());
            for (k, ld) in loads.into_iter().enumerate() {
                first.instructions.insert(at + k, ld);
            }
        }
    }

    // Keep typed pointer-field provenance in step with the interface splice so cross-function helper
    // recovery can replay texture handles after entry parameters become loaded image ids or descriptor
    // array variables.
    ctx.emit_sidecar
        .remap_ids(&splices.iter().copied().collect());

    // Splice param ids -> their replacement ids.
    {
        let func = &mut ctx.module.functions[entry_idx];
        for (from, to) in splices {
            replace_id_in_function(func, from, to);
        }
    }
    collapse_resource_wrappers(ctx, entry_idx, &resource_values);

    // Rewrite collapsed-buffer uses. Done after the generic splices (these param ids were deliberately
    // NOT spliced) so each use can be handled by kind.
    for (pid, var, block_ty, prepend_member0) in collapsed_buffers {
        rewrite_collapsed_buffer(ctx, entry_idx, pid, var, block_ty, prepend_member0, defs);
    }
    for (pid, var, block_ty, elem_ty) in record_array_buffers {
        nested_air_ordinal_roots.extend(rewrite_record_array_buffer(
            ctx, entry_idx, pid, var, block_ty, elem_ty, defs,
        ));
    }
    if !nested_air_ordinal_roots.is_empty() {
        remap_nested_air_struct_accesses(ctx, entry_idx, &nested_air_ordinal_roots, defs);
    }

    // Rewrite buffer-derived pointer storage classes to StorageBuffer. The buffer var is already
    // StorageBuffer-typed; the OpAccessChain results that walk into it keep the backend's
    // UniformConstant element-pointer types, which Vulkan rejects. Clone those pointer types into
    // StorageBuffer and remap. Root from EVERY buffer variable (struct + synthesized-wrapper) so
    // both direct uses and the spliced member-0 chains are covered.
    let _ = &buffer_param_ids;
    if !buffer_structs.is_empty() {
        let roots: Vec<Word> = buffer_structs.iter().map(|(v, _)| *v).collect();
        rewrite_pointer_storage(ctx, entry_idx, &roots, StorageClass::StorageBuffer, defs)?;
        rewrite_raw_word_alias_chains(ctx, entry_idx, buffer_structs, defs)?;
    }
    // Re-class the access chains derived from unmodeled-pointer Private zero vars (same problem as the
    // buffer chains, but the var/leaf type is Private not StorageBuffer).
    if !zero_pointer_vars.is_empty() {
        rewrite_pointer_storage(
            ctx,
            entry_idx,
            &zero_pointer_vars,
            StorageClass::Private,
            defs,
        )?;
        // A function-constant-gated atomic buffer (e.g. an MPS reduce `groupid_counter`) becomes one
        // of these absent Private zero vars; SPIR-V forbids OpAtomic* on Private storage. Private
        // memory is per-invocation, so the atomic is semantically a plain load/op/store — rewrite it.
        rewrite_private_pointer_atomics(ctx, entry_idx);
    }
    if !workgroup_vars.is_empty() {
        for var in &workgroup_vars {
            rewrite_workgroup_root_access(ctx, entry_idx, *var, defs);
        }
        rewrite_pointer_storage(
            ctx,
            entry_idx,
            &workgroup_vars,
            StorageClass::Workgroup,
            defs,
        )?;
        rewrite_flattened_workgroup_leaf_accesses(ctx, entry_idx, &workgroup_vars, defs);
    }
    rewrite_structural_load_result_types(ctx, entry_idx, defs);
    rewrite_ulong_uint2_memory_reinterprets(ctx, entry_idx, defs);
    Ok(())
}

fn adapt_input_attachment_read(
    ctx: &mut Ctx,
    loads: &mut Vec<Instruction>,
    read: Word,
    read_ty: Word,
    param_ty: Word,
) -> Result<Word, String> {
    if read_ty == param_ty {
        return Ok(read);
    }

    let Some((read_component, Some(read_lanes))) = scalar_or_vector_component_live(ctx, read_ty)
    else {
        return Err(format!(
            "input attachment read type {read_ty} is not a vector"
        ));
    };
    let (param_component, param_lanes) = scalar_or_vector_component_live(ctx, param_ty)
        .ok_or_else(|| format!("input attachment param type {param_ty} is not scalar/vector"))?;
    let needed_lanes = param_lanes.unwrap_or(1);
    if needed_lanes > read_lanes {
        return Err(format!(
            "input attachment param type {param_ty} needs {needed_lanes} components, read type {read_ty} has {read_lanes}"
        ));
    }

    let (shaped, shaped_ty) = match param_lanes {
        None => {
            let extracted = ctx.module.fresh_id();
            loads.push(Instruction::new(
                Op::CompositeExtract,
                Some(read_component),
                Some(extracted),
                vec![Operand::IdRef(read), Operand::LiteralBit32(0)],
            ));
            (extracted, read_component)
        }
        Some(lanes) if lanes == read_lanes => (read, read_ty),
        Some(lanes) => {
            let prefix_ty = input_attachment_vector_type(ctx, read_component, lanes);
            let mut components = Vec::with_capacity(lanes as usize);
            for lane in 0..lanes {
                let extracted = ctx.module.fresh_id();
                loads.push(Instruction::new(
                    Op::CompositeExtract,
                    Some(read_component),
                    Some(extracted),
                    vec![Operand::IdRef(read), Operand::LiteralBit32(lane)],
                ));
                components.push(Operand::IdRef(extracted));
            }
            let vector = ctx.module.fresh_id();
            loads.push(Instruction::new(
                Op::CompositeConstruct,
                Some(prefix_ty),
                Some(vector),
                components,
            ));
            (vector, prefix_ty)
        }
    };
    if shaped_ty == param_ty {
        return Ok(shaped);
    }

    if !is_float_type_live(ctx, read_component) || !is_float_type_live(ctx, param_component) {
        return Err(format!(
            "input attachment read type {read_ty} cannot be converted to param type {param_ty}"
        ));
    }

    let converted = ctx.module.fresh_id();
    loads.push(Instruction::new(
        Op::FConvert,
        Some(param_ty),
        Some(converted),
        vec![Operand::IdRef(shaped)],
    ));
    Ok(converted)
}

fn scalar_or_vector_component_live(ctx: &Ctx, ty: Word) -> Option<(Word, Option<u32>)> {
    let def = type_def_of(ctx, ty)?;
    match def.class.opcode {
        Op::TypeVector => {
            let component = match def.operands.first()? {
                Operand::IdRef(component) => *component,
                _ => return None,
            };
            let lanes = match def.operands.get(1)? {
                Operand::LiteralBit32(lanes) => *lanes,
                _ => return None,
            };
            Some((component, Some(lanes)))
        }
        Op::TypeFloat | Op::TypeInt | Op::TypeBool => Some((ty, None)),
        _ => None,
    }
}

fn is_float_type_live(ctx: &Ctx, ty: Word) -> bool {
    type_def_of(ctx, ty).is_some_and(|def| def.class.opcode == Op::TypeFloat)
}

fn input_attachment_vector_type(ctx: &mut Ctx, component_ty: Word, lanes: u32) -> Word {
    ctx.get_or_create(
        Op::TypeVector,
        None,
        vec![Operand::IdRef(component_ty), Operand::LiteralBit32(lanes)],
    )
}

pub(in crate::passes) fn rewrite_ulong_uint2_memory_reinterprets(
    ctx: &mut Ctx,
    entry_idx: usize,
    defs: &HashMap<Word, Instruction>,
) {
    let types = combined_type_defs(ctx, defs);
    let mut value_types = combined_value_types(ctx, entry_idx);
    let uint = ctx.ty_uint();
    let ulong = ctx.ty_ulong();
    let v2uint = ctx.ty_vec_uint(2);
    let shift = ctx.const_uint(32);

    let block_count = ctx.module.functions[entry_idx].blocks.len();
    for block_idx in 0..block_count {
        let instructions =
            std::mem::take(&mut ctx.module.functions[entry_idx].blocks[block_idx].instructions);
        let mut rewritten = Vec::with_capacity(instructions.len());
        for mut inst in instructions {
            match inst.class.opcode {
                Op::Load if inst.result_type == Some(v2uint) => {
                    let Some(result) = inst.result_id else {
                        rewritten.push(inst);
                        continue;
                    };
                    let Some(Operand::IdRef(ptr)) = inst.operands.first() else {
                        rewritten.push(inst);
                        continue;
                    };
                    if pointer_value_pointee(&types, &value_types, *ptr) != Some(ulong) {
                        rewritten.push(inst);
                        continue;
                    }
                    let loaded = ctx.module.fresh_id();
                    inst.result_type = Some(ulong);
                    inst.result_id = Some(loaded);
                    rewritten.push(inst);
                    let low = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::UConvert,
                        Some(uint),
                        Some(low),
                        vec![Operand::IdRef(loaded)],
                    ));
                    let shifted = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::ShiftRightLogical,
                        Some(ulong),
                        Some(shifted),
                        vec![Operand::IdRef(loaded), Operand::IdRef(shift)],
                    ));
                    let high = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::UConvert,
                        Some(uint),
                        Some(high),
                        vec![Operand::IdRef(shifted)],
                    ));
                    rewritten.push(Instruction::new(
                        Op::CompositeConstruct,
                        Some(v2uint),
                        Some(result),
                        vec![Operand::IdRef(low), Operand::IdRef(high)],
                    ));
                    value_types.insert(loaded, ulong);
                    value_types.insert(low, uint);
                    value_types.insert(shifted, ulong);
                    value_types.insert(high, uint);
                    value_types.insert(result, v2uint);
                }
                Op::Store => {
                    let (Some(Operand::IdRef(ptr)), Some(Operand::IdRef(object))) =
                        (inst.operands.first(), inst.operands.get(1))
                    else {
                        rewritten.push(inst);
                        continue;
                    };
                    if pointer_value_pointee(&types, &value_types, *ptr) != Some(ulong)
                        || value_types.get(object).copied() != Some(v2uint)
                    {
                        rewritten.push(inst);
                        continue;
                    }
                    let low = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::CompositeExtract,
                        Some(uint),
                        Some(low),
                        vec![Operand::IdRef(*object), Operand::LiteralBit32(0)],
                    ));
                    let high = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::CompositeExtract,
                        Some(uint),
                        Some(high),
                        vec![Operand::IdRef(*object), Operand::LiteralBit32(1)],
                    ));
                    let low64 = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::UConvert,
                        Some(ulong),
                        Some(low64),
                        vec![Operand::IdRef(low)],
                    ));
                    let high64 = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::UConvert,
                        Some(ulong),
                        Some(high64),
                        vec![Operand::IdRef(high)],
                    ));
                    let shifted = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::ShiftLeftLogical,
                        Some(ulong),
                        Some(shifted),
                        vec![Operand::IdRef(high64), Operand::IdRef(shift)],
                    ));
                    let joined = ctx.module.fresh_id();
                    rewritten.push(Instruction::new(
                        Op::BitwiseOr,
                        Some(ulong),
                        Some(joined),
                        vec![Operand::IdRef(low64), Operand::IdRef(shifted)],
                    ));
                    inst.operands[1] = Operand::IdRef(joined);
                    rewritten.push(inst);
                    value_types.insert(low, uint);
                    value_types.insert(high, uint);
                    value_types.insert(low64, ulong);
                    value_types.insert(high64, ulong);
                    value_types.insert(shifted, ulong);
                    value_types.insert(joined, ulong);
                }
                _ => rewritten.push(inst),
            }
        }
        ctx.module.functions[entry_idx].blocks[block_idx].instructions = rewritten;
    }
}

pub(in crate::passes) fn pointer_value_pointee(
    types: &HashMap<Word, Instruction>,
    value_types: &HashMap<Word, Word>,
    pointer: Word,
) -> Option<Word> {
    value_types
        .get(&pointer)
        .and_then(|ptr_ty| ptr_pointee(types, *ptr_ty))
}

pub(in crate::passes) fn collapse_resource_wrappers(
    ctx: &mut Ctx,
    entry_idx: usize,
    resource_values: &HashSet<Word>,
) {
    if resource_values.is_empty() {
        return;
    }

    let func = &ctx.module.functions[entry_idx];
    let mut paths: HashMap<Word, HashMap<Vec<u32>, Word>> = HashMap::new();
    let mut replacements: Vec<(Word, Word)> = vec![];

    for blk in &func.blocks {
        for inst in &blk.instructions {
            let Some(result) = inst.result_id else {
                continue;
            };
            match inst.class.opcode {
                Op::CompositeInsert => {
                    let Some(Operand::IdRef(object)) = inst.operands.first() else {
                        continue;
                    };
                    let Some(Operand::IdRef(base)) = inst.operands.get(1) else {
                        continue;
                    };
                    let path = literal_path(&inst.operands[2..]);
                    let mut result_paths = paths.get(base).cloned().unwrap_or_default();
                    insert_resource_path(
                        &mut result_paths,
                        &path,
                        *object,
                        &paths,
                        resource_values,
                    );
                    if !result_paths.is_empty() {
                        paths.insert(result, result_paths);
                    }
                }
                Op::CompositeConstruct => {
                    let mut result_paths = HashMap::new();
                    for (idx, operand) in inst.operands.iter().enumerate() {
                        let Operand::IdRef(object) = operand else {
                            continue;
                        };
                        insert_resource_path(
                            &mut result_paths,
                            &[idx as u32],
                            *object,
                            &paths,
                            resource_values,
                        );
                    }
                    if !result_paths.is_empty() {
                        paths.insert(result, result_paths);
                    }
                }
                Op::CopyObject => {
                    let Some(Operand::IdRef(source)) = inst.operands.first() else {
                        continue;
                    };
                    if let Some(source_paths) = paths.get(source).cloned() {
                        paths.insert(result, source_paths);
                    } else if resource_values.contains(source) {
                        let mut result_paths = HashMap::new();
                        result_paths.insert(vec![], *source);
                        paths.insert(result, result_paths);
                    }
                }
                Op::CompositeExtract => {
                    let Some(Operand::IdRef(composite)) = inst.operands.first() else {
                        continue;
                    };
                    let path = literal_path(&inst.operands[1..]);
                    let Some(composite_paths) = paths.get(composite) else {
                        continue;
                    };
                    if let Some(resource) = composite_paths.get(&path).copied() {
                        replacements.push((result, resource));
                        continue;
                    }
                    let mut result_paths = HashMap::new();
                    for (resource_path, resource) in composite_paths {
                        if let Some(suffix) = path_suffix(resource_path, &path) {
                            result_paths.insert(suffix.to_vec(), *resource);
                        }
                    }
                    if !result_paths.is_empty() {
                        paths.insert(result, result_paths);
                    }
                }
                _ => {}
            }
        }
    }

    if replacements.is_empty() {
        return;
    }

    {
        let func = &mut ctx.module.functions[entry_idx];
        for (from, to) in replacements {
            replace_id_in_function(func, from, to);
        }
    }
    remove_dead_resource_wrapper_ops(ctx, entry_idx);
}

pub(in crate::passes) fn insert_resource_path(
    out: &mut HashMap<Vec<u32>, Word>,
    prefix: &[u32],
    value: Word,
    paths: &HashMap<Word, HashMap<Vec<u32>, Word>>,
    resource_values: &HashSet<Word>,
) {
    if resource_values.contains(&value) {
        out.insert(prefix.to_vec(), value);
    }
    if let Some(value_paths) = paths.get(&value) {
        for (suffix, resource) in value_paths {
            let mut full = prefix.to_vec();
            full.extend(suffix);
            out.insert(full, *resource);
        }
    }
}

pub(in crate::passes) fn literal_path(operands: &[Operand]) -> Vec<u32> {
    operands
        .iter()
        .filter_map(|operand| match operand {
            Operand::LiteralBit32(value) => Some(*value),
            _ => None,
        })
        .collect()
}

pub(in crate::passes) fn path_suffix<'a>(path: &'a [u32], prefix: &[u32]) -> Option<&'a [u32]> {
    if path.len() < prefix.len() || &path[..prefix.len()] != prefix {
        return None;
    }
    Some(&path[prefix.len()..])
}

pub(in crate::passes) fn remove_dead_resource_wrapper_ops(ctx: &mut Ctx, entry_idx: usize) {
    loop {
        let used = function_used_ids(&ctx.module.functions[entry_idx]);
        let mut changed = false;
        for blk in &mut ctx.module.functions[entry_idx].blocks {
            let before = blk.instructions.len();
            blk.instructions.retain(|inst| {
                let dead = inst
                    .result_id
                    .map(|id| !used.contains(&id))
                    .unwrap_or(false);
                !(dead
                    && matches!(
                        inst.class.opcode,
                        Op::CompositeInsert
                            | Op::CompositeExtract
                            | Op::CompositeConstruct
                            | Op::CopyObject
                            | Op::Undef
                    ))
            });
            changed |= blk.instructions.len() != before;
        }
        if !changed {
            break;
        }
    }
}

pub(in crate::passes) fn function_used_ids(func: &Function) -> HashSet<Word> {
    let mut used = HashSet::new();
    for param in &func.parameters {
        for operand in &param.operands {
            collect_operand_id_refs(operand, &mut used);
        }
    }
    for blk in &func.blocks {
        if let Some(label) = &blk.label {
            for operand in &label.operands {
                collect_operand_id_refs(operand, &mut used);
            }
        }
        for inst in &blk.instructions {
            for operand in &inst.operands {
                collect_operand_id_refs(operand, &mut used);
            }
        }
    }
    used
}

pub(in crate::passes) fn collect_operand_id_refs(operand: &Operand, used: &mut HashSet<Word>) {
    if let Operand::IdRef(id) = operand {
        used.insert(*id);
    }
}
