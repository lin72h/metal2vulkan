use super::*;
use crate::native::cfg::BodyBlock;
use crate::native::tir::{TirFunction, TirOpcode, TirOperand, TirTerminator};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FastSumLeaf {
    Addend,
    Positive,
    Negative,
    Difference,
}

fn ordinary_cfg_shape_key(blocks: &[BodyBlock]) -> String {
    // `structured_plan` rejection is a control-flow property. Keep value opcodes out of this key so
    // type-specialized helper bodies with the same block graph share the rejection fact; each body
    // still receives its own construct-tree plan and typed-SSA closure check.
    let labels = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut key = String::new();
    for block in blocks {
        key.push_str(&format!("{:?}:", block.role));
        let Some(typed) = block.typed.as_deref() else {
            key.push_str("untyped;");
            continue;
        };
        let tag = match &typed.terminator {
            TirTerminator::Br(_) => "b",
            TirTerminator::BrCond { .. } => "c",
            TirTerminator::Switch { cases, .. } => {
                key.push_str(&format!("s{},", cases.len()));
                "s"
            }
            TirTerminator::Ret(Some(_)) => "rv",
            TirTerminator::Ret(None) => "r",
            TirTerminator::Unreachable => "u",
        };
        key.push_str(tag);
        for successor in typed.terminator.successors() {
            key.push(',');
            key.push_str(
                &labels
                    .get(successor)
                    .map(|index| index.to_string())
                    .unwrap_or_else(|| "?".to_string()),
            );
        }
        key.push(';');
    }
    key
}

fn sixteen_leaf_cancellation_order(shape: &[FastSumLeaf]) -> Option<&'static [usize; 16]> {
    use FastSumLeaf::{Addend as A, Difference as D, Negative as N, Positive as P};

    match shape {
        [P, P, A, P, N, N, N, N, N, D, N, D, P, D, D, A] => {
            Some(&[13, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 14, 15])
        }
        [A, P, P, N, N, N, N, D, N, D, N, D, P, D, P, D] => {
            Some(&[12, 1, 2, 3, 0, 5, 6, 7, 8, 9, 10, 11, 4, 13, 14, 15])
        }
        [P, P, A, N, N, N, P, P, N, D, N, D, N, D, D, A] => {
            Some(&[0, 15, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1])
        }
        [A, P, N, N, N, P, P, D, N, D, N, D, N, D, P, D] => {
            Some(&[0, 12, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 13, 14, 15])
        }
        [P, N, A, N, P, P, N, N, P, D, P, D, N, D, D, A] => {
            Some(&[3, 15, 2, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1])
        }
        [A, N, N, P, P, N, N, D, P, D, P, D, N, D, N, D] => {
            Some(&[14, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 3, 15])
        }
        [P, N, A, N, P, N, P, P, N, D, P, D, N, D, D, A] => {
            Some(&[3, 1, 2, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        }
        [A, N, N, P, N, P, P, D, N, D, P, D, N, D, N, D] => {
            Some(&[1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        }
        [A, N, N, P, N, P, P, N, D, P, D, N, D, N, D, D] => {
            Some(&[1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        }
        [P, N, A, P, N, P, N, N, P, D, N, D, P, D, D, A] => {
            Some(&[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1])
        }
        [A, N, P, N, P, N, N, D, P, D, N, D, P, D, N, D] => {
            Some(&[0, 12, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 13, 14, 15])
        }
        _ => None,
    }
}

fn bda_forward_address_values(
    tir: &TirFunction,
    seeds: &HashSet<String>,
    opaque: &HashSet<String>,
) -> HashSet<String> {
    struct Node {
        dependencies: Vec<String>,
        literal_anchor: bool,
        supported: bool,
    }

    let value_dependencies = |values: Vec<&LlValue>| {
        let mut dependencies = Vec::new();
        let mut literal_anchor = false;
        let mut supported = true;
        for value in values {
            match value {
                LlValue::Local(name) => dependencies.push(name.clone()),
                LlValue::Zero | LlValue::Undef => literal_anchor = true,
                _ => supported = false,
            }
        }
        (dependencies, literal_anchor, supported)
    };

    let mut nodes = HashMap::<String, Node>::new();
    for inst in tir.blocks.iter().flat_map(|block| &block.insts) {
        let Some(result) = &inst.result else {
            continue;
        };
        let result_is_device_pointer = match inst.opcode {
            TirOpcode::GetElementPtr => inst
                .gep()
                .as_ref()
                .is_some_and(|gep| matches!(gep.base.ty, LlType::Ptr(1))),
            TirOpcode::Bitcast => inst.bitcast().is_some_and(|(_, destination)| {
                matches!(
                    crate::native::parse::parse_type(destination),
                    Ok(LlType::Ptr(1))
                )
            }),
            TirOpcode::Phi => inst
                .phi_incoming()
                .as_ref()
                .is_some_and(|(ty, _)| matches!(ty, LlType::Ptr(1))),
            TirOpcode::Select => {
                inst.select_arms()
                    .as_deref()
                    .is_some_and(|(true_value, false_value)| {
                        matches!(true_value.ty, LlType::Ptr(1))
                            && matches!(false_value.ty, LlType::Ptr(1))
                    })
            }
            TirOpcode::Freeze | TirOpcode::Metal2VulkanInlineParameter => inst
                .operands
                .first()
                .and_then(TirOperand::as_typed_value)
                .is_some_and(|value| matches!(value.ty, LlType::Ptr(1))),
            TirOpcode::Load => inst
                .load()
                .as_deref()
                .is_some_and(|load| matches!(load.result_ty, LlType::Ptr(1))),
            _ => matches!(inst.result_ty, Some(LlType::Ptr(1))),
        };
        if !result_is_device_pointer {
            continue;
        }
        if opaque.contains(result) {
            continue;
        }
        let (dependencies, literal_anchor, supported) = match inst.opcode {
            TirOpcode::GetElementPtr => inst
                .gep()
                .as_ref()
                .map(|gep| value_dependencies(vec![&gep.base.value]))
                .unwrap_or_default(),
            TirOpcode::Bitcast
            | TirOpcode::AddrSpaceCast
            | TirOpcode::Freeze
            | TirOpcode::Metal2VulkanInlineParameter => inst
                .operands
                .first()
                .and_then(TirOperand::as_typed_value)
                .map(|value| value_dependencies(vec![&value.value]))
                .unwrap_or_default(),
            // A pointer-valued load produces the address payload stored in memory; it does not
            // forward the address of the slot being loaded. Opaque resource loads were excluded
            // above, so every remaining device-pointer load is a concrete address-domain anchor.
            TirOpcode::Load => (Vec::new(), true, true),
            TirOpcode::Phi => inst
                .phi_values()
                .map(|values| value_dependencies(values.collect()))
                .unwrap_or_default(),
            TirOpcode::Select => inst
                .select_arms()
                .as_deref()
                .map(|(true_value, false_value)| {
                    value_dependencies(vec![&true_value.value, &false_value.value])
                })
                .unwrap_or_default(),
            _ => (Vec::new(), false, false),
        };
        nodes.insert(
            result.clone(),
            Node {
                dependencies,
                literal_anchor,
                supported,
            },
        );
    }

    // Greatest fixed point: retain cyclic state only when every concrete dependency remains inside
    // the address-domain graph. Then require a least-fixed-point path to a real address/zero/undef
    // anchor, excluding an ungrounded phi cycle.
    let mut valid = nodes
        .iter()
        .filter_map(|(name, node)| node.supported.then_some(name.clone()))
        .chain(seeds.iter().cloned())
        .collect::<HashSet<_>>();
    loop {
        let rejected = nodes
            .iter()
            .filter(|(name, node)| {
                valid.contains(*name)
                    && node
                        .dependencies
                        .iter()
                        .any(|dependency| !valid.contains(dependency))
            })
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        if rejected.is_empty() {
            break;
        }
        for name in rejected {
            valid.remove(&name);
        }
    }
    let mut anchored = seeds.clone();
    loop {
        let additions = nodes
            .iter()
            .filter(|(name, node)| {
                valid.contains(*name)
                    && !anchored.contains(*name)
                    && (node.literal_anchor
                        || node
                            .dependencies
                            .iter()
                            .any(|dependency| anchored.contains(dependency)))
            })
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        if additions.is_empty() {
            break;
        }
        anchored.extend(additions);
    }
    anchored.retain(|name| valid.contains(name));
    anchored
}

fn is_air_texture_operation(callee: &str) -> bool {
    callee.starts_with("air.sample_texture")
        || callee.starts_with("air.sample_depth")
        || callee.starts_with("air.sample_compare_depth")
        || callee.starts_with("air.gather_texture")
        || callee.starts_with("air.gather_depth")
        || callee.starts_with("air.read_texture")
        || callee.starts_with("air.read_depth")
        || callee.starts_with("air.write_texture")
        || callee.starts_with("air.write_imageblock_slice_to_texture")
        || callee.starts_with("air.atomic_fetch_max_explicit_texture_")
        || callee.starts_with("air.get_width_texture")
        || callee.starts_with("air.get_height_texture")
        || callee.starts_with("air.get_depth_texture")
        || callee.starts_with("air.get_array_size_texture")
        || callee.starts_with("air.get_width_depth")
        || callee.starts_with("air.get_height_depth")
        || callee.starts_with("air.get_depth_depth")
        || callee.starts_with("air.get_num_mip_levels_texture")
        || callee.starts_with("air.get_num_mip_levels_depth")
        || callee.starts_with("air.get_num_samples_texture")
        || callee.starts_with("air.calculate_unclamped_lod_texture")
        || callee.starts_with("air.calculate_clamped_lod_texture")
        || callee.starts_with("air.is_null_texture")
        || callee.starts_with("air.fence_texture")
}

pub(in crate::native) fn opaque_resource_pointer_values_by_function(
    ir: &LlModule,
) -> HashMap<String, HashSet<String>> {
    let functions = ir
        .functions
        .iter()
        .map(|function| (function.name.as_str(), function))
        .collect::<HashMap<_, _>>();
    let mut opaque_params = HashSet::<(String, usize)>::new();

    for function in &ir.functions {
        let params = function
            .params
            .iter()
            .enumerate()
            .map(|(index, (name, _))| (name.as_str(), index))
            .collect::<HashMap<_, _>>();
        for instruction in function.carrier_insts() {
            let Some(call) = instruction.call().as_deref() else {
                continue;
            };
            if !is_air_texture_operation(&call.callee) {
                continue;
            }
            for argument in &call.args {
                let LlValue::Local(name) = &argument.value else {
                    continue;
                };
                if matches!(argument.ty, LlType::Ptr(1)) {
                    if let Some(index) = params.get(name.as_str()) {
                        opaque_params.insert((function.name.clone(), *index));
                    }
                }
            }
        }
    }

    let mut changed = true;
    while changed {
        changed = false;
        for function in &ir.functions {
            let params = function
                .params
                .iter()
                .enumerate()
                .map(|(index, (name, _))| (name.as_str(), index))
                .collect::<HashMap<_, _>>();
            for instruction in function.carrier_insts() {
                let Some(call) = instruction.call().as_deref() else {
                    continue;
                };
                if !functions.contains_key(call.callee.as_str()) {
                    continue;
                }
                for (callee_index, argument) in call.args.iter().enumerate() {
                    if !opaque_params.contains(&(call.callee.clone(), callee_index)) {
                        continue;
                    }
                    let LlValue::Local(name) = &argument.value else {
                        continue;
                    };
                    if let Some(caller_index) = params.get(name.as_str()) {
                        changed |= opaque_params.insert((function.name.clone(), *caller_index));
                    }
                }
            }
        }
    }

    ir.functions
        .iter()
        .map(|function| {
            let mut opaque = HashSet::new();
            for instruction in function.carrier_insts() {
                let Some(call) = instruction.call().as_deref() else {
                    continue;
                };
                for (index, argument) in call.args.iter().enumerate() {
                    let direct_texture = is_air_texture_operation(&call.callee)
                        && matches!(argument.ty, LlType::Ptr(1));
                    let helper_texture = opaque_params.contains(&(call.callee.clone(), index));
                    if direct_texture || helper_texture {
                        if let LlValue::Local(name) = &argument.value {
                            opaque.insert(name.clone());
                        }
                    }
                }
            }
            let mut changed = true;
            while changed {
                changed = false;
                for instruction in function.carrier_insts() {
                    let Some(result) = instruction.result.as_ref() else {
                        continue;
                    };
                    if !matches!(instruction.result_ty, Some(LlType::Ptr(1)))
                        || !matches!(
                            instruction.opcode.as_str(),
                            "bitcast" | "freeze" | "phi" | "select"
                        )
                    {
                        continue;
                    }
                    let mut operands = Vec::new();
                    instruction.visit_uses(|name| operands.push(name.to_owned()));
                    if opaque.contains(result) {
                        for operand in operands {
                            changed |= opaque.insert(operand);
                        }
                    } else if operands.iter().any(|operand| opaque.contains(operand)) {
                        changed |= opaque.insert(result.clone());
                    }
                }
            }
            (function.name.clone(), opaque)
        })
        .collect()
}

impl Emitter {
    pub(super) fn emit_function(&mut self, f: &LlFunction) -> Result<(), String> {
        self.values.clear();
        self.fast_float_products.clear();
        self.fast_grouped_sums.clear();
        self.fast_partitioned_sums.clear();
        self.fast_grouped_sum_boundaries.clear();
        self.fast_contract_adds.clear();
        self.fast_uncontracted_sums.clear();
        self.construct_tree_active = false;
        // Graph-driven emission: the typed graph is built below from the STRUCTURIZED block list
        // (`body_blocks`), once it is finalized — see the `tir::build_from_blocks` call after
        // structurization. Building it there (not from the parse-time blocks here) sources operands from exactly the
        // IR emission walks, the prerequisite the phi/store migration needed (the structurizer rewrites
        // those operands between parse and emit). It is the SOLE emission substrate — there is no string
        // fall-back path; a build failure (a block with no terminator) is a fail-visible error.
        self.tir_result_types.clear();
        self.tir_predicates.clear();
        self.tir_aligns.clear();
        self.tir_gep_source_types.clear();
        self.tir_use_pointees.clear();
        self.tir_direct_load_pointers.clear();
        self.byte_view_pointers.clear();
        self.network_pointees.clear();
        self.null_rooted_pointer_values.clear();
        self.null_rooted_pointer_peers.clear();
        self.gep_provenance.clear();
        self.workgroup_padding_byte_pointers.clear();
        self.selected_pointers.clear();
        self.selected_load_pointers.clear();
        self.selected_access_trees.clear();
        self.vector_word_roots.clear();
        self.vector_word_pointers.clear();
        self.local_pointer_fields.clear();
        self.pointer_forward_values.clear();
        self.raw_memcpy_shadows.clear();
        self.dynamic_pointer_tables.clear();
        self.forward_geps.clear();
        self.forward_pointer_selects.clear();
        self.forward_pointer_select_conditions.clear();
        self.pointer_storage.clear();
        self.pointer_pointees.clear();
        self.local_alloca_pointees = self
            .ir
            .local_alloca_pointees
            .iter()
            .filter(|&((func, _name), _pointee)| func == &f.name)
            .map(|((_func, name), pointee)| (name.clone(), pointee.clone()))
            .collect();
        self.pointer_nullness.clear();
        self.bda_inttoptr_sources.clear();
        self.opaque_resource_payload_loads.clear();
        self.bda_forward_sources.clear();
        self.bda_address_loads.clear();
        self.bda_forward_addresses.clear();
        self.bda_direct_addresses.clear();
        self.bda_aggregate_addresses.clear();
        self.aggregate_pointer_values.clear();
        self.opaque_resource_pointers = self
            .opaque_resource_pointer_values
            .get(&f.name)
            .cloned()
            .unwrap_or_default();
        self.pointer_payload_words.clear();
        self.pointer_payload_values.clear();
        self.pointer_phi_values.clear();
        self.pointer_phi_incoming_values.clear();
        self.tir_phi_incomings.clear();
        self.phi_edge_instructions.clear();
        self.phi_result_instructions.clear();
        self.direct_param_values.clear();
        self.direct_param_indices.clear();
        self.param_values.clear();
        self.inline_parameter_substitutions.clear();
        self.raw_offsets.clear();
        self.int_alignments.clear();
        self.unmodeled_pointers.clear();
        for global in &self.ir.globals.clone() {
            let pointee = self.global_declared_pointee(global)?;
            self.pointer_pointees.insert(global.name.clone(), pointee);
            self.pointer_storage.insert(
                global.name.clone(),
                if global.addrspace == 3 {
                    StorageClass::Workgroup
                } else {
                    StorageClass::Private
                },
            );
        }
        // Build the typed block carrier before parameter types are chosen. A
        // reinterpret-mixed pointer network has no single legal Logical-SPIR-V pointee. When every
        // leaf traces to function-constant alternatives at one buffer location, model only those
        // roots as byte-addressed from the outset. This preserves every per-use typed view without
        // the all-buffer raw retry. Replan CFG ownership only when this analysis actually adds a raw
        // root; rediscovering an existing raw model must leave its already-valid plan untouched.
        // Construct-tree owns source switches directly. Other emission paths lower them to branch
        // ladders first so every intermediate block is serializable; the relooper feed preserves
        // the same edge semantics and later replaces the complete emitted CFG.
        let source_labels = f
            .blocks
            .iter()
            .map(|block| block.name.clone())
            .collect::<HashSet<_>>();
        let bda_label_value_overlap = self.bda_device_pointers
            && crate::native::cfg::clone_crossarm::cloned_labels_overlap_ssa_values(
                &f.blocks,
                &source_labels,
            );
        if bda_label_value_overlap {
            self.emit_sidecar
                .ordinary_plan_rejected_functions
                .insert(f.name.clone());
            self.emit_sidecar
                .ownership_plan_rejected_functions
                .insert(f.name.clone());
        }
        let construct_tree_enabled = !self.relooper_feed
            && !bda_label_value_overlap
            && !self.known_ownership_plan_rejections.contains(&f.name);
        let source_switches_require_ladder =
            crate::native::cfg::switch_default_is_inferred_merge(&f.blocks);
        let switch_ready_blocks = if construct_tree_enabled && !source_switches_require_ladder {
            f.blocks.clone()
        } else {
            lower_unstructured_switches(&f.blocks)
        };
        let mut body_blocks =
            crate::native::tir::prune_literal_branch_dead_blocks(switch_ready_blocks);
        body_blocks = crate::native::tir::prune_unused_geps(body_blocks);
        let buffer_params = f
            .params
            .iter()
            .filter_map(|(name, _)| {
                self.ir
                    .metadata_fc_buffer_locations
                    .get(&(f.name.clone(), name.clone()))
                    .map(|location| (name.clone(), *location))
            })
            .collect();
        let reinterpret_mix_roots =
            crate::native::emitter::pointer_network::reinterpret_mix_buffer_params(
                &body_blocks,
                &buffer_params,
            );
        let mut canonical_roots = BTreeMap::new();
        let mut shared_raw_roots = HashMap::new();
        for (name, _) in &f.params {
            if !reinterpret_mix_roots.contains(name) {
                continue;
            }
            let location = buffer_params[name];
            let canonical = canonical_roots
                .entry(location)
                .or_insert_with(|| name.clone());
            shared_raw_roots.insert(name.clone(), canonical.clone());
        }
        for name in reinterpret_mix_roots {
            self.ir.raw_buffer_params.insert((f.name.clone(), name));
        }
        let buffer_pointer_params = f
            .params
            .iter()
            .filter(|(_, ty)| matches!(ty, LlType::Ptr(1 | 2)))
            .map(|(name, _)| name.clone())
            .collect();
        for name in crate::native::emitter::pointer_network::cross_member_widening_load_roots(
            &body_blocks,
            &buffer_pointer_params,
            &self.ir.types,
        ) {
            self.ir.raw_buffer_params.insert((f.name.clone(), name));
        }
        self.raw_buffer_params = self
            .ir
            .raw_buffer_params
            .iter()
            .filter(|(function, _)| function == &f.name)
            .map(|(_, name)| name.clone())
            .collect();
        self.data_buffer_params = self
            .ir
            .metadata_data_buffer_params
            .iter()
            .filter(|(function, _)| function == &f.name)
            .map(|(_, name)| name.clone())
            .collect();
        let ret_ty = self.resolve_type(&f.ret)?;
        let ret_id = self.type_id(&ret_ty)?;
        let param_types: Vec<Word> = f
            .params
            .iter()
            .enumerate()
            .map(|(index, (name, ty))| self.param_type_id(&f.name, index, name, ty))
            .collect::<Result<Vec<_>, _>>()?;
        let nullness_params = f
            .params
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                self.function_param_nullness
                    .contains(&(f.name.clone(), *index))
            })
            .map(|(index, (name, _))| (index, name.clone()))
            .collect::<Vec<_>>();
        let bool_type = (!nullness_params.is_empty())
            .then(|| self.type_id(&LlType::Bool))
            .transpose()?;
        let mut function_param_types = param_types.clone();
        if let Some(bool_type) = bool_type {
            function_param_types.extend(std::iter::repeat_n(bool_type, nullness_params.len()));
        }
        let fn_ty = self.function_type_id(ret_id, &function_param_types);

        let func_id = *self
            .function_ids
            .get(&f.name)
            .ok_or_else(|| format!("native emitter: missing function id for {}", f.name))?;
        self.module.debug_names.push(Self::inst(
            Op::Name,
            None,
            None,
            vec![
                Operand::IdRef(func_id),
                Operand::LiteralString(f.name.clone()),
            ],
        ));
        let mut params = Vec::with_capacity(function_param_types.len());
        for (param_index, ((name, ty), type_id)) in
            f.params.iter().zip(param_types.iter()).enumerate()
        {
            let id = self.fresh();
            params.push(Self::inst(
                Op::FunctionParameter,
                Some(*type_id),
                Some(id),
                vec![],
            ));
            let resolved_ty = self.resolve_type(ty)?;
            self.values.insert(name.clone(), (id, resolved_ty.clone()));
            if let LlType::Ptr(addrspace) = resolved_ty {
                let bda_data_buffer_parameter = self.bda_device_pointers
                    && addrspace == 0
                    && self.data_buffer_params.contains(name)
                    && !self.ir.entry_functions.contains(&f.name);
                let storage = if addrspace == 4
                    && (self.ir.imageblock_dimensions.is_some() || self.ir.imageblock_shared_cells)
                {
                    StorageClass::Workgroup
                } else if bda_data_buffer_parameter {
                    StorageClass::PhysicalStorageBuffer
                } else {
                    llvm_pointer_storage(addrspace)?
                };
                self.pointer_storage.insert(name.clone(), storage);
                if bda_data_buffer_parameter {
                    self.bda_direct_addresses.insert(name.clone(), id);
                    self.used_device_address = true;
                }
                if self.ir.entry_functions.contains(&f.name)
                    || self
                        .function_param_nonnull
                        .contains(&(f.name.clone(), param_index))
                {
                    let is_null = self.const_bool(false)?;
                    self.record_pointer_nullness(name.clone(), is_null);
                }
                let concrete_workgroup_raw_param = addrspace == 3
                    && self
                        .concrete_vector_workgroup_raw_param_pointee(&f.name, param_index, name)
                        .is_some();
                if self.raw_buffer_params.contains(name) {
                    let pointee = if addrspace == 3 {
                        self.concrete_vector_workgroup_raw_param_pointee(&f.name, param_index, name)
                            .unwrap_or_else(raw_workgroup_array_type)
                    } else {
                        raw_buffer_block_type()
                    };
                    self.pointer_pointees.insert(name.clone(), pointee);
                } else if let Some(pointee) = self
                    .function_param_pointees
                    .get(&(f.name.clone(), param_index))
                    .cloned()
                {
                    self.pointer_pointees.insert(name.clone(), pointee);
                } else if let Some(pointee) = self
                    .ir
                    .ptr_pointees
                    .get(&(f.name.clone(), name.clone()))
                    .cloned()
                {
                    self.pointer_pointees.insert(name.clone(), pointee);
                }
                if self.raw_buffer_params.contains(name) && !concrete_workgroup_raw_param {
                    let mut raw = self
                        .raw_call_param_offsets
                        .get(&(f.name.clone(), name.clone()))
                        .cloned()
                        .unwrap_or_else(|| RawBufferOffset::root(name.clone(), addrspace));
                    raw.root = shared_raw_roots
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| name.clone());
                    if self.bda_device_pointers
                        && addrspace == 1
                        && !self.ir.entry_functions.contains(&f.name)
                    {
                        raw.device_addr_base = Some(id);
                        self.used_device_address = true;
                        self.pointer_storage
                            .insert(name.clone(), StorageClass::PhysicalStorageBuffer);
                    }
                    self.raw_offsets.insert(name.clone(), raw);
                }
            }
            self.direct_param_values.insert(name.clone());
            if self.ir.entry_functions.contains(&f.name) {
                self.direct_param_indices
                    .insert(name.clone(), param_index as u32);
            }
            self.param_values.insert(name.clone());
        }
        for (_, name) in nullness_params {
            let id = self.fresh();
            params.push(Self::inst(
                Op::FunctionParameter,
                bool_type,
                Some(id),
                vec![],
            ));
            self.record_pointer_nullness(name, id);
        }

        // T5/T8 keystone: seed every block's typed carrier from its lines at SPLIT time — BEFORE
        // `lower_unstructured_switches` — so no window exists in which a structurizer reader sees an
        // unpopulated (`None`) carrier. Switch lowering then preserves carriers on pass-through blocks
        // and constructs them on the ladder blocks it synthesizes, so every block is carriered from
        // birth — the invariant that let `BodyBlock.lines` retire (readers never need a `.lines`
        // fallback). BC drift NONE proves the carriers stay byte-identical to a fresh re-lower.
        // The switch lowerer already returns an owned copy and never mutates its input. Feed it the
        // parse-time carriers directly: cloning the complete source CFG here retained two owned
        // copies of very large functions before planning had even started.
        let reorder_defuse = ReorderDefUse::from_blocks(&body_blocks)?;
        reorder_forward_local_def_blocks(&mut body_blocks, &reorder_defuse)?;
        // R2 module 4: structured-by-construction emission is the DEFAULT (module 4). For a fully-
        // structurable function, take the forest-derived plan (reordered blocks + per-construct unique
        // merges) and skip the post-hoc pre-phi fixup. A rejection is recorded as a construction fact
        // so the owned pipeline selects the raw-CFG representation. The structured path is a strict improvement on
        // the unseeded frontier (cfg 479→110, total 929→574, zero over-admission across all 16 shards).
        // A relooper-feed emission deliberately preserves the original CFG for the raw-CFG
        // switch-dispatch structurizer.  Running the full structured-plan ladder first cannot
        // improve that feed, and on a large rejected graph it repeatedly recomputes dominators and
        // selection merges before the feed gets a chance to run.  Keep the normal diagnostic and
        // planning paths intact for every primary emission; only this internal construction
        // intermediate bypasses them.
        let relooper_feed = self.relooper_feed;
        if crate::env_vars::why() && !relooper_feed {
            if body_blocks.len() <= crate::native::cfg::CROSS_ARM_EDGE_MAX_BLOCKS {
                match crate::native::cfg::structured_reject_reason(&body_blocks) {
                    None => eprintln!("WHY ADMIT"),
                    Some(r) => eprintln!("WHY REJECT {r}"),
                }
            } else {
                eprintln!(
                    "WHY LARGE-CFG blocks={} local-reject-replay=skipped",
                    body_blocks.len()
                );
            }
        }
        let mut structured_active = false;
        let retry_debug = crate::env_vars::retry_debug();
        let planner_started = retry_debug.then(std::time::Instant::now);
        // A natural loop with forward entries owned by different nested selections has no valid
        // private merge assignment in the original region forest. Build its single-owner region
        // tree before ordinary merge planning; repairing the already-emitted module would have to
        // rediscover and clone source constructs after their ownership information was erased.
        let skip_source_cfg_construction =
            crate::native::cfg::exceeds_local_structured_plan_budget(&body_blocks);
        let shared_loop_entry_ownership = construct_tree_enabled
            && !skip_source_cfg_construction
            && crate::native::cfg::requires_shared_loop_entry_ownership(&body_blocks);
        let loop_exit_sibling_ownership = construct_tree_enabled
            && !skip_source_cfg_construction
            && crate::native::cfg::requires_loop_exit_sibling_dispatch(&body_blocks);
        let mut construct_tree_plan = if loop_exit_sibling_ownership {
            match crate::native::cfg::renest_loop_exit_sibling(&body_blocks) {
                Ok(Some(renested)) => {
                    if let Some(plan) =
                        crate::native::cfg::structured_plan_construct_tree(&renested)
                    {
                        body_blocks = renested;
                        Some(plan)
                    } else {
                        None
                    }
                }
                Ok(None) => None,
                Err(regional_error) => {
                    if crate::env_vars::why() {
                        eprintln!(
                            "WHY-CONSTRUCT-TREE function={} loop-exit-sibling decline {regional_error}",
                            f.name
                        );
                    }
                    None
                }
            }
        } else {
            shared_loop_entry_ownership
                .then(|| crate::native::cfg::structured_plan_construct_tree(&body_blocks))
                .flatten()
        };
        // Planning is pure for a fixed `body_blocks` graph and can dominate translation time on
        // generated functions. Carry the result through the decision tree and invalidate it only
        // when a retry actually rewrites the CFG; do not re-run the full ladder merely to ask the
        // same admission question at a later branch.
        let ownership_preplanned = shared_loop_entry_ownership
            || (loop_exit_sibling_ownership && construct_tree_plan.is_some());
        let ordinary_shape_key = ordinary_cfg_shape_key(&body_blocks);
        let ordinary_shape_rejected = self
            .ordinary_plan_rejected_shapes
            .contains(&ordinary_shape_key);
        let ordinary_plan_attempted = construct_tree_enabled
            && !ownership_preplanned
            && !skip_source_cfg_construction
            && !ordinary_shape_rejected
            && !self.known_ordinary_plan_rejections.contains(&f.name);
        let mut ordinary_plan = if ordinary_plan_attempted {
            let plan = crate::native::cfg::structured_plan(&body_blocks);
            if plan.is_none() {
                self.ordinary_plan_rejected_shapes
                    .insert(ordinary_shape_key);
            }
            plan
        } else {
            None
        };
        if ordinary_plan
            .as_ref()
            .is_some_and(|plan| !typed_ssa_is_closed(&plan.blocks, &f.params))
        {
            ordinary_plan = None;
        }
        if (ordinary_plan_attempted || ordinary_shape_rejected) && ordinary_plan.is_none() {
            self.emit_sidecar
                .ordinary_plan_rejected_functions
                .insert(f.name.clone());
        }
        if let Some(planner_started) = planner_started {
            let (mode, result) = if loop_exit_sibling_ownership && construct_tree_plan.is_some() {
                (
                    "regional-dispatch",
                    if construct_tree_plan.is_some() {
                        "admit"
                    } else {
                        "reject"
                    },
                )
            } else if shared_loop_entry_ownership {
                (
                    "shared-loop-owner",
                    if construct_tree_plan.is_some() {
                        "admit"
                    } else {
                        "reject"
                    },
                )
            } else {
                (
                    "ordinary",
                    if ordinary_plan.is_some() {
                        "admit"
                    } else {
                        "reject"
                    },
                )
            };
            eprintln!(
                "[retry-debug] planner: function={} blocks={} mode={} result={} phase_ms={}",
                f.name,
                body_blocks.len(),
                mode,
                result,
                planner_started.elapsed().as_millis(),
            );
        }
        if construct_tree_enabled {
            // Source-level ownership construction runs before emission on every ordinary-plan
            // rejection. Local recognizers preserve unaffected CFG/SSA; when they decline, the bounded
            // whole-CFG dispatcher carries scalar state through explicit typed slots.
            if ordinary_plan.is_none() && construct_tree_plan.is_none() {
                let mut candidate = if skip_source_cfg_construction {
                    Vec::new()
                } else {
                    body_blocks.clone()
                };
                let mut candidate_changed = false;
                let mut construct_tree_applied = false;
                // The own-arm/straddle recognizers are local clone prepasses. Each begins by running
                // the full source-CFG reject classifier and can retain rewritten candidates; on a
                // large graph that duplicates the very planner work the global ownership tier exists
                // to replace. Keep those bounded like the other local clone machinery and send large
                // functions straight to the raw construct tree below.
                if construct_tree_enabled
                    && !skip_source_cfg_construction
                    && body_blocks.len() <= crate::native::cfg::CROSS_ARM_EDGE_MAX_BLOCKS
                {
                    // Both recognizers dispatch on the same classification of the same graph, and
                    // deriving it replays the whole planner ladder. `candidate` only changes in the
                    // arm that sets `construct_tree_applied`, which skips the second recognizer, so
                    // one derivation is all either of them can ever see.
                    let reject_reason = crate::native::cfg::structured_reject_reason(&candidate);
                    match crate::native::cfg::renest_cond_phi_shared_own_arm(
                        &candidate,
                        reject_reason.as_deref(),
                    ) {
                        Ok(Some(renested)) => {
                            candidate = renested;
                            candidate_changed = true;
                            construct_tree_applied = true;
                            construct_tree_plan =
                                crate::native::cfg::structured_plan_construct_tree(&candidate);
                            if crate::env_vars::why() && construct_tree_plan.is_none() {
                                eprintln!("WHY-CONSTRUCT-TREE own-arm plan-decline");
                            }
                        }
                        Ok(None) => {}
                        Err(error) => {
                            if crate::env_vars::why() {
                                eprintln!("WHY-CONSTRUCT-TREE own-arm decline {error}");
                            }
                        }
                    }
                    if !construct_tree_applied {
                        match crate::native::cfg::renest_straddle_loop_merge(
                            &candidate,
                            reject_reason.as_deref(),
                        ) {
                            Ok(Some(renested)) => {
                                candidate = renested;
                                candidate_changed = true;
                                construct_tree_applied = true;
                                construct_tree_plan =
                                    crate::native::cfg::structured_plan_construct_tree(&candidate);
                                if crate::env_vars::why() && construct_tree_plan.is_none() {
                                    eprintln!("WHY-CONSTRUCT-TREE straddle plan-decline");
                                }
                            }
                            Ok(None) => {}
                            Err(error) => {
                                if crate::env_vars::why() {
                                    eprintln!("WHY-CONSTRUCT-TREE straddle decline {error}");
                                }
                            }
                        }
                    }
                }
                if construct_tree_enabled && !construct_tree_applied {
                    // The global ownership planner is useful even when neither optional pre-renesting
                    // transform changes the graph, and it is the primary path for large CFGs whose
                    // local recognizers are deliberately skipped above.
                    let construct_tree_started = retry_debug.then(std::time::Instant::now);
                    construct_tree_plan = if skip_source_cfg_construction {
                        None
                    } else {
                        crate::native::cfg::structured_plan_construct_tree(&candidate)
                    };
                    if construct_tree_plan.is_none() && !skip_source_cfg_construction {
                        match crate::native::cfg::renest_whole_cfg_dispatch(&candidate) {
                            Ok(plan) => construct_tree_plan = Some(plan),
                            Err(error) => {
                                if crate::env_vars::why() {
                                    eprintln!(
                                        "WHY-CONSTRUCT-TREE function={} whole-cfg decline {error}",
                                        f.name
                                    );
                                }
                            }
                        }
                    }
                    if let Some(construct_tree_started) = construct_tree_started {
                        eprintln!(
                            "[retry-debug] planner: function={} blocks={} mode=construct-tree result={} phase_ms={}",
                            f.name,
                            body_blocks.len(),
                            if construct_tree_plan.is_some() {
                                "admit"
                            } else {
                                "reject"
                            },
                            construct_tree_started.elapsed().as_millis(),
                        );
                    }
                    if !skip_source_cfg_construction
                        && crate::env_vars::why()
                        && construct_tree_plan.is_none()
                    {
                        // The detailed witness deliberately replays several planner variants. Keep it
                        // for the small graphs it was built to explain; on a global CFG the SPI
                        // breadcrumbs already identify the exact rejecting header.
                        if candidate.len() <= crate::native::cfg::CROSS_ARM_EDGE_MAX_BLOCKS {
                            for witness in
                                crate::native::cfg::construct_tree_gate_witness_lines(&candidate)
                            {
                                eprintln!("WHY-CONSTRUCT-TREE {witness}");
                            }
                        }
                    }
                }
                if !skip_source_cfg_construction
                    && crate::env_vars::why()
                    && candidate.len() <= crate::native::cfg::CROSS_ARM_EDGE_MAX_BLOCKS
                {
                    match crate::native::cfg::structured_reject_reason(&candidate) {
                        None => eprintln!("WHY-CANDIDATE ADMIT"),
                        Some(r) => eprintln!("WHY-CANDIDATE REJECT {r}"),
                    }
                }
                let candidate_plan = (construct_tree_plan.is_none() && candidate_changed)
                    .then(|| crate::native::cfg::structured_plan(&candidate))
                    .flatten();
                if construct_tree_plan.is_some() || candidate_plan.is_some() {
                    if !skip_source_cfg_construction {
                        body_blocks = candidate;
                    }
                    ordinary_plan = candidate_plan;
                }
            }
            if ordinary_plan
                .as_ref()
                .is_some_and(|plan| !typed_ssa_is_closed(&plan.blocks, &f.params))
            {
                ordinary_plan = None;
                self.emit_sidecar
                    .ordinary_plan_rejected_functions
                    .insert(f.name.clone());
            }
            if construct_tree_plan
                .as_ref()
                .is_some_and(|plan| !typed_ssa_is_closed(&plan.blocks, &f.params))
            {
                construct_tree_plan = None;
                self.emit_sidecar
                    .ownership_plan_rejected_functions
                    .insert(f.name.clone());
            }
            let construct_tree_active = construct_tree_plan.is_some();
            if let Some(plan) = construct_tree_plan.or(ordinary_plan) {
                self.construct_tree_active = construct_tree_active;
                body_blocks = plan.blocks;
                self.block_labels.clear();
                self.branch_merges = plan.branch_merges;
                self.branch_merges_by_header = plan.branch_merges_by_header;
                self.branch_merges_header_only = construct_tree_active;
                self.loop_merges = plan.loop_merges;
                self.switch_merges = plan.switch_merges;
                structured_active = true;
            }
            if !structured_active {
                self.emit_sidecar
                    .ownership_plan_rejected_functions
                    .insert(f.name.clone());
            }
        }
        if !structured_active {
            self.branch_merges_header_only = false;
            self.block_labels.clear();
            self.branch_merges_by_header.clear();
            if relooper_feed {
                self.branch_merges = infer_direct_branch_merges(&body_blocks);
                self.loop_merges.clear();
                self.switch_merges = infer_switch_merges(&body_blocks);
            } else if skip_source_cfg_construction {
                // The bounded relooper owns this complete CFG. Keep the primary carrier linear and
                // local: it exists only to preserve typed instructions and source edges until that
                // selected constructor runs, so cloning/refunneling it here would duplicate the
                // over-budget ownership work without changing the chosen representation.
                self.branch_merges_by_header = infer_bounded_branch_merges_by_header(&body_blocks);
                self.branch_merges.clear();
                self.loop_merges.clear();
                self.switch_merges = infer_direct_switch_merges(&body_blocks);
            } else if body_blocks.len() > crate::native::cfg::CROSS_ARM_EDGE_MAX_BLOCKS {
                // The ordinary and construct-tree planners have both rejected this large CFG. Building
                // heuristic branch/loop transitive-closure maps here can exceed the memory budget before
                // owned construction selects the relooper. Preserve only branch
                // and switch ownership proved by direct local reconvergence; these linear subsets
                // handle ordinary case blocks and lowered switch ladders without restoring the
                // transitive-closure analysis. The intermediate remains owned and must satisfy the
                // complete construction checks before serialization.
                body_blocks = funnel_shared_branch_dispatches(&body_blocks);
                let mut refunnel_counter = body_blocks.len();
                if let Some(refunnelled) =
                    refunnel_one_deep_shared_arm(&body_blocks, &mut refunnel_counter)
                {
                    let ordinary_refunnelled = crate::native::cfg::structured_plan(&refunnelled);
                    let construct_tree_refunnelled = ordinary_refunnelled.is_none();
                    let plan = ordinary_refunnelled.or_else(|| {
                        crate::native::cfg::structured_plan_construct_tree(&refunnelled)
                    });
                    if let Some(plan) = plan {
                        self.construct_tree_active = construct_tree_refunnelled;
                        body_blocks = plan.blocks;
                        self.block_labels.clear();
                        self.branch_merges = plan.branch_merges;
                        self.branch_merges
                            .extend(infer_direct_branch_merges(&body_blocks));
                        self.branch_merges_by_header = plan.branch_merges_by_header;
                        self.branch_merges_header_only = construct_tree_refunnelled;
                        self.loop_merges = plan.loop_merges;
                        self.switch_merges = plan.switch_merges;
                        structured_active = true;
                    }
                }
                if !structured_active {
                    self.branch_merges_by_header =
                        infer_bounded_branch_merges_by_header(&body_blocks);
                    self.branch_merges.clear();
                    self.loop_merges.clear();
                    self.switch_merges = infer_direct_switch_merges(&body_blocks);
                }
            } else {
                self.branch_merges = infer_branch_merges(&body_blocks);
                self.loop_merges = infer_loop_merges(&body_blocks);
                self.switch_merges = infer_switch_merges(&body_blocks);
            }
            if !self.branch_merges_header_only {
                index_branch_merges_by_header(
                    &body_blocks,
                    &self.loop_merges,
                    &self.branch_merges,
                    &mut self.branch_merges_by_header,
                );
            }
            // Rejected CFGs use heuristic ownership rather than an admitted `StructuredPlan`.
            // Complete that fallback constructor here so its output observes the same one-header-
            // per-merge invariant the structured planner self-checks before admission. Structured
            // plans never pass through this normalization after admission.
            privatize_reused_emitted_merge_targets(
                &mut body_blocks,
                &mut self.loop_merges,
                &self.branch_merges,
                &mut self.branch_merges_by_header,
                self.branch_merges_header_only,
                &mut self.switch_merges,
            );
        }
        // Structured region cloning can introduce a definition block after a new use even when the
        // source order was already normalized. Re-establish def-before-use order on the finalized
        // plan itself; block order is not control-flow semantics, while the single-pass emitter
        // requires every ordinary non-phi SSA operand to have reserved its result id first.
        let finalized_defuse = ReorderDefUse::from_blocks(&body_blocks)?;
        reorder_forward_local_def_blocks(&mut body_blocks, &finalized_defuse)?;
        // R3: index resolved typed operands by result name, built from the now-finalized structurized
        // block list — the exact IR the emission loop below walks. Straight-line instruction text is
        // never rewritten by structurization, so migrated consumers (binary/unary/convert/compare/
        // select/load/extract+insert/shuffle/gep) read byte-identical operands; the added synthetic
        // `%metal2vulkan.lmerge.*` phi entries are inert until phi emission consumes the graph. `store` and
        // the void-`call` (both result-LESS) drive straight off `inst.operands`/`inst.mem_align()`/`inst.call()`
        // in the graph walk — the former per-block store/call text-keyed queues are retired.
        // The typed graph is built from the finalized structurized blocks and is the SOLE emission
        // substrate. A build failure (a block with no terminator) is a fail-visible error, not a
        // text-walk fallback. Measured dead broadly (0 build failures / 16942 frontier + 0 / 15,336 banked).
        crate::native::tir::canonicalize_single_predecessor_phis(&mut body_blocks);
        let tir = crate::native::tir::build_from_blocks(&body_blocks)?;
        {
            let tir = &tir;
            let mut fast_products = HashMap::new();
            for inst in tir.blocks.iter().flat_map(|block| &block.insts) {
                if !inst.fast_math() || inst.opcode != "fmul" {
                    continue;
                }
                let (Some(result), Some(operands)) =
                    (inst.result.as_ref(), self.tir_inst_typed_operands(inst))
                else {
                    continue;
                };
                if let [lhs, rhs] = operands.as_slice() {
                    fast_products.insert(result.clone(), (lhs.clone(), rhs.clone()));
                }
            }
            let mut product_users = HashMap::<String, usize>::new();
            for inst in tir.blocks.iter().flat_map(|block| &block.insts) {
                inst.visit_uses(|name| {
                    if fast_products.contains_key(name) {
                        *product_users.entry(name.to_string()).or_default() += 1;
                    }
                });
            }
            self.fast_float_products = fast_products
                .iter()
                .filter(|(name, _)| product_users.get(*name) == Some(&1))
                .map(|(name, operands)| (name.clone(), operands.clone()))
                .collect();
            let fast_add_operands = tir
                .blocks
                .iter()
                .flat_map(|block| &block.insts)
                .filter(|inst| inst.fast_math() && inst.opcode == "fadd")
                .filter_map(|inst| {
                    let result = inst.result.as_ref()?;
                    let operands = self.tir_inst_typed_operands(inst)?;
                    let [lhs, rhs] = operands.as_slice() else {
                        return None;
                    };
                    let local = |value: &LlValue| match value {
                        LlValue::Local(name) => Some(name.clone()),
                        _ => None,
                    };
                    Some((result.clone(), (local(&lhs.value), local(&rhs.value))))
                })
                .collect::<HashMap<_, _>>();
            let fast_add_values = tir
                .blocks
                .iter()
                .flat_map(|block| &block.insts)
                .filter(|inst| inst.fast_math() && inst.opcode == "fadd")
                .filter_map(|inst| {
                    let result = inst.result.as_ref()?;
                    let operands = self.tir_inst_typed_operands(inst)?;
                    let [lhs, rhs] = operands.as_slice() else {
                        return None;
                    };
                    Some((result.clone(), (lhs.clone(), rhs.clone())))
                })
                .collect::<HashMap<_, _>>();
            let fast_sum_values = tir
                .blocks
                .iter()
                .flat_map(|block| &block.insts)
                .filter(|inst| inst.fast_math() && matches!(inst.opcode.as_str(), "fadd" | "fsub"))
                .filter_map(|inst| {
                    let result = inst.result.as_ref()?;
                    let operands = self.tir_inst_typed_operands(inst)?;
                    let [lhs, rhs] = operands.as_slice() else {
                        return None;
                    };
                    Some((
                        result.clone(),
                        (inst.opcode == "fsub", lhs.clone(), rhs.clone()),
                    ))
                })
                .collect::<HashMap<_, _>>();
            let nested_fast_adds = fast_sum_values
                .values()
                .flat_map(|(_, lhs, rhs)| [lhs, rhs])
                .filter_map(|value| match &value.value {
                    LlValue::Local(name) if fast_add_values.contains_key(name) => {
                        Some(name.clone())
                    }
                    _ => None,
                })
                .collect::<HashSet<_>>();
            let mut stored_fast_adds = HashSet::new();
            for inst in tir
                .blocks
                .iter()
                .flat_map(|block| &block.insts)
                .filter(|inst| inst.opcode == "store")
            {
                inst.visit_uses(|name| {
                    if fast_add_values.contains_key(name) {
                        stored_fast_adds.insert(name.to_string());
                    }
                });
            }
            fn collect_fast_sum_terms(
                value: &TypedValue,
                negate: bool,
                sums: &HashMap<String, (bool, TypedValue, TypedValue)>,
                visiting: &mut HashSet<String>,
                terms: &mut Vec<(TypedValue, bool)>,
            ) -> bool {
                let LlValue::Local(name) = &value.value else {
                    terms.push((value.clone(), negate));
                    return true;
                };
                let Some((subtract, lhs, rhs)) = sums.get(name) else {
                    terms.push((value.clone(), negate));
                    return true;
                };
                if *subtract {
                    terms.push((value.clone(), negate));
                    return true;
                }
                if !visiting.insert(name.clone()) {
                    return false;
                }
                let valid = collect_fast_sum_terms(lhs, negate, sums, visiting, terms)
                    && collect_fast_sum_terms(rhs, negate ^ subtract, sums, visiting, terms);
                visiting.remove(name);
                valid
            }
            fn literal_is_negative(value: &TypedValue) -> Option<bool> {
                match (&value.ty, &value.value) {
                    (LlType::Float, LlValue::Float(value)) => Some(value.is_sign_negative()),
                    (LlType::Float, LlValue::Hex(bits)) => Some(bits >> 63 != 0),
                    (LlType::Half, LlValue::HalfBits(bits)) => Some(bits >> 15 != 0),
                    (LlType::BFloat, LlValue::BFloatBits(bits)) => Some(bits >> 15 != 0),
                    (_, LlValue::SignedInt(value)) => Some(*value < 0),
                    (_, LlValue::Int(_) | LlValue::Zero) => Some(false),
                    _ => None,
                }
            }
            let leaf_shape = |term: &TypedValue, negate: bool| {
                let LlValue::Local(name) = &term.value else {
                    return Some(FastSumLeaf::Addend);
                };
                let Some((lhs, rhs)) = fast_products.get(name) else {
                    return Some(FastSumLeaf::Addend);
                };
                if [lhs, rhs].into_iter().any(|operand| {
                    matches!(&operand.value, LlValue::Local(name) if fast_sum_values.get(name).is_some_and(|(subtract, _, _)| *subtract))
                }) {
                    return Some(FastSumLeaf::Difference);
                }
                let coefficient_negative =
                    [lhs, rhs].into_iter().find_map(literal_is_negative)? ^ negate;
                Some(if coefficient_negative {
                    FastSumLeaf::Negative
                } else {
                    FastSumLeaf::Positive
                })
            };
            for (root, (lhs, rhs)) in &fast_add_values {
                if nested_fast_adds.contains(root) {
                    continue;
                }
                let mut terms = Vec::new();
                let mut visiting = HashSet::new();
                if !collect_fast_sum_terms(lhs, false, &fast_sum_values, &mut visiting, &mut terms)
                    || !collect_fast_sum_terms(
                        rhs,
                        false,
                        &fast_sum_values,
                        &mut visiting,
                        &mut terms,
                    )
                    || !(5..10).contains(&terms.len())
                {
                    continue;
                }
                let shape = terms
                    .iter()
                    .map(|(term, negate)| leaf_shape(term, *negate))
                    .collect::<Option<Vec<_>>>();
                let stored_source_tree = stored_fast_adds.contains(root)
                    && shape.as_deref().is_some_and(|shape| {
                        shape.len() == 7
                            && shape
                                .iter()
                                .filter(|leaf| matches!(leaf, FastSumLeaf::Addend))
                                .count()
                                == 1
                            && shape
                                .iter()
                                .filter(|leaf| matches!(leaf, FastSumLeaf::Difference))
                                .count()
                                == 2
                    });
                if stored_source_tree {
                    continue;
                }
                let mut positive = Vec::new();
                let mut negative = Vec::new();
                let mut supported = true;
                for (term, negate) in terms {
                    let product = match &term.value {
                        LlValue::Local(name) => fast_products.get(name),
                        _ => None,
                    };
                    let sign = if let Some((a, b)) = product {
                        let literals = [a, b]
                            .into_iter()
                            .filter_map(literal_is_negative)
                            .collect::<Vec<_>>();
                        if literals.len() != 1 {
                            supported = false;
                            break;
                        }
                        negate ^ literals[0]
                    } else {
                        negate
                    };
                    if sign {
                        negative.push((term, negate));
                    } else {
                        positive.push((term, negate));
                    }
                }
                let mut moved_accumulator = false;
                if let Some(position) = positive.iter().position(|(term, _)| match &term.value {
                    LlValue::Local(name) => {
                        !fast_products.contains_key(name) && !fast_sum_values.contains_key(name)
                    }
                    _ => true,
                }) {
                    if position > 1 {
                        let accumulator = positive.remove(position);
                        positive.insert(1, accumulator);
                        moved_accumulator = true;
                    }
                }
                if supported && positive.len() >= 2 && negative.len() >= 2 {
                    for (term, _) in positive.iter().chain(&negative) {
                        let LlValue::Local(name) = &term.value else {
                            continue;
                        };
                        if fast_sum_values
                            .get(name)
                            .is_some_and(|(subtract, _, _)| *subtract)
                        {
                            self.fast_grouped_sum_boundaries.insert(name.clone());
                        }
                    }
                    self.fast_grouped_sums
                        .insert(root.clone(), (positive, negative, moved_accumulator));
                }
            }
            for (root, (lhs, rhs)) in &fast_add_values {
                if nested_fast_adds.contains(root) {
                    continue;
                }
                let mut terms = Vec::new();
                let mut visiting = HashSet::new();
                if !collect_fast_sum_terms(lhs, false, &fast_sum_values, &mut visiting, &mut terms)
                    || !collect_fast_sum_terms(
                        rhs,
                        false,
                        &fast_sum_values,
                        &mut visiting,
                        &mut terms,
                    )
                    || !matches!(terms.len(), 10 | 16)
                {
                    continue;
                }
                if terms.len() == 16 {
                    let Some(shape) = terms
                        .iter()
                        .map(|(term, negate)| leaf_shape(term, *negate))
                        .collect::<Option<Vec<_>>>()
                    else {
                        continue;
                    };
                    let Some(order) = sixteen_leaf_cancellation_order(&shape) else {
                        continue;
                    };
                    let ordered = order
                        .iter()
                        .map(|index| terms[*index].0.clone())
                        .collect::<Vec<_>>();
                    self.fast_partitioned_sums
                        .insert(root.clone(), vec![ordered]);
                    continue;
                }
                let signs = terms
                    .iter()
                    .map(|(term, negate)| {
                        let LlValue::Local(name) = &term.value else {
                            return None;
                        };
                        let (lhs, rhs) = fast_products.get(name)?;
                        [lhs, rhs]
                            .into_iter()
                            .find_map(literal_is_negative)
                            .map(|negative| negative ^ negate)
                    })
                    .collect::<Vec<_>>();
                let difference_products = terms
                    .iter()
                    .map(|(term, _)| {
                        let LlValue::Local(name) = &term.value else {
                            return false;
                        };
                        fast_products.get(name).is_some_and(|(lhs, rhs)| {
                            [lhs, rhs].into_iter().any(|operand| {
                                matches!(&operand.value, LlValue::Local(name) if fast_sum_values.get(name).is_some_and(|(subtract, _, _)| *subtract))
                            })
                        })
                    })
                    .collect::<Vec<_>>();
                let accumulator_real_tree = difference_products.as_slice()
                    == [
                        false, false, false, false, false, false, true, false, true, true,
                    ];
                // Preserve the partitions selected by each typed cancellation topology. These
                // predicates use only leaf operation kind and coefficient sign: direct products,
                // products of source differences, and product differences retain distinct rounding
                // boundaries independently of symbols and exact coefficient values.
                let partitions: &[usize] = match signs.as_slice() {
                    [Some(false), Some(false), _, Some(true), ..] => &[2, 6],
                    [Some(false), Some(true), _, Some(true), ..] => &[1, 4],
                    [None, Some(true), Some(true), Some(false), ..] if accumulator_real_tree => {
                        &[2]
                    }
                    [None, Some(false), Some(true), Some(true), ..] if accumulator_real_tree => {
                        &[3]
                    }
                    _ => continue,
                };
                let mut groups = Vec::with_capacity(partitions.len() + 1);
                let mut start = 0;
                for end in partitions
                    .iter()
                    .copied()
                    .chain(std::iter::once(terms.len()))
                {
                    groups.push(
                        terms[start..end]
                            .iter()
                            .map(|(term, _)| term.clone())
                            .collect(),
                    );
                    start = end;
                }
                self.fast_partitioned_sums.insert(root.clone(), groups);
            }
            const EXPLICIT_FMA_CHAIN_MIN_TERMS: usize = 10;
            for result in fast_add_operands.keys() {
                let mut chain = Vec::new();
                let mut current = result.as_str();
                let mut visited = HashSet::new();
                while visited.insert(current.to_string()) {
                    let Some((lhs, rhs)) = fast_add_operands.get(current) else {
                        break;
                    };
                    if ![lhs.as_ref(), rhs.as_ref()]
                        .into_iter()
                        .flatten()
                        .any(|name| fast_products.contains_key(name))
                    {
                        break;
                    }
                    chain.push(current.to_string());
                    let Some(previous) = [lhs.as_ref(), rhs.as_ref()]
                        .into_iter()
                        .flatten()
                        .find(|name| fast_add_operands.contains_key(*name))
                    else {
                        break;
                    };
                    current = previous;
                }
                if chain.len() >= EXPLICIT_FMA_CHAIN_MIN_TERMS {
                    self.fast_contract_adds.extend(chain);
                }
            }
            let mut changed = true;
            while changed {
                changed = false;
                for inst in tir.blocks.iter().flat_map(|block| &block.insts) {
                    if !inst.fast_math() || inst.opcode != "fadd" {
                        continue;
                    }
                    let (Some(result), Some(operands)) =
                        (inst.result.as_ref(), self.tir_inst_typed_operands(inst))
                    else {
                        continue;
                    };
                    let [lhs, rhs] = operands.as_slice() else {
                        continue;
                    };
                    let lhs_name = match &lhs.value {
                        LlValue::Local(name) => Some(name.as_str()),
                        _ => None,
                    };
                    let rhs_name = match &rhs.value {
                        LlValue::Local(name) => Some(name.as_str()),
                        _ => None,
                    };
                    let product_pair = lhs_name
                        .is_some_and(|name| fast_products.contains_key(name))
                        && rhs_name.is_some_and(|name| fast_products.contains_key(name));
                    let extends_sum = lhs_name
                        .is_some_and(|name| self.fast_uncontracted_sums.contains(name))
                        || rhs_name.is_some_and(|name| self.fast_uncontracted_sums.contains(name));
                    if (product_pair || extends_sum)
                        && self.fast_uncontracted_sums.insert(result.clone())
                    {
                        changed = true;
                    }
                }
            }
            // M1 (pointer-typing rewrite): carry the USE-based pointee of every pointer SSA value onto
            // the value, sourced from the SAME structurized graph the operand map below is built from.
            // This is the whole-function `use_pointees` map (keyed by value name, propagated across
            // select/phi/freeze pointer merges to a fixpoint), not a per-block projection. Available to
            // emission as the pointee carrier the side-table-retiring rewrite (M2+) consumes; unused here,
            // so byte-neutral by construction (proven via the BC byte-drift gate).
            self.tir_use_pointees = tir.use_pointees.clone();
            // A pointer select whose arms have incompatible provisional pointees cannot become an
            // SPIR-V pointer OpSelect. When every use is a direct load, it is nevertheless exactly
            // representable by loading each arm and selecting the values. Record that closed use
            // class from the typed graph; a store/GEP/call or any other consumer excludes the value.
            let mut direct_load_pointers = HashSet::new();
            let mut other_pointer_uses = HashSet::new();
            for block in &tir.blocks {
                for inst in &block.insts {
                    let load_pointer =
                        inst.load()
                            .as_deref()
                            .and_then(|load| match &load.ptr.value {
                                LlValue::Local(name) => Some(name.as_str()),
                                _ => None,
                            });
                    if let Some(name) = load_pointer {
                        direct_load_pointers.insert(name.to_string());
                    }
                    inst.visit_uses(|name| {
                        if load_pointer != Some(name) {
                            other_pointer_uses.insert(name.to_string());
                        }
                    });
                }
            }
            direct_load_pointers.retain(|name| !other_pointer_uses.contains(name));
            self.tir_direct_load_pointers = direct_load_pointers;
            // The mixed byte/wide subset of the carrier: the byte→real upgrade in
            // `pointer_pointee_for_value` must skip these (upgrading strands their `uchar` byte cursor).
            self.byte_view_pointers = tir.byte_view_pointers.clone();
            // M3 (pointer-typing rewrite): the pointer-`phi` membership side-tables are now carried on
            // the tir graph (computed once during the build), retiring the emitter's separate
            // `pointer_phi_result_names` / `pointer_phi_incoming_value_names` `body_blocks` text-walks.
            // Byte-identical by construction (same source lines + same `phi ptr` predicate), proven
            // byte-neutral via the BC byte-drift gate.
            self.pointer_phi_values = tir.pointer_phi_results.clone();
            self.pointer_phi_incoming_values = tir.pointer_phi_incoming.clone();
            // M3: the `getelementptr`-result `forward_geps` side-table is likewise carried on the tir
            // graph now, retiring the standalone `forward_gep_results` `body_blocks` walk from the primary
            // path. Byte-identical by construction (same lines + `parse_gep`).
            self.forward_geps = tir.forward_geps.clone();
            for block in &tir.blocks {
                for inst in &block.insts {
                    if self.bda_device_pointers && inst.opcode == "inttoptr" {
                        if let (Some(result), Some(source)) = (
                            inst.result.as_ref(),
                            inst.operands
                                .first()
                                .and_then(|operand| operand.as_typed_value()),
                        ) {
                            self.bda_inttoptr_sources.insert(result.clone(), source);
                        }
                    }
                    if self.bda_device_pointers
                        && matches!(
                            inst.opcode,
                            TirOpcode::Bitcast
                                | TirOpcode::AddrSpaceCast
                                | TirOpcode::Freeze
                                | TirOpcode::Metal2VulkanInlineParameter
                        )
                        && matches!(inst.result_ty, Some(LlType::Ptr(1)))
                    {
                        if let (Some(result), Some(source)) = (
                            inst.result.as_ref(),
                            inst.operands
                                .first()
                                .and_then(|operand| operand.as_typed_value()),
                        ) {
                            self.bda_forward_sources.insert(result.clone(), source);
                        }
                    }
                    if self.bda_device_pointers
                        && inst.opcode == TirOpcode::Load
                        && matches!(inst.result_ty, Some(LlType::Ptr(1)))
                    {
                        if let Some(result) = inst.result.as_ref() {
                            self.bda_address_loads.insert(result.clone());
                        }
                    }
                    if matches!(inst.cmp_predicate().as_deref(), Some("eq" | "ne")) {
                        if let [crate::native::tir::TirOperand::Value {
                            name: lhs_name,
                            ty: LlType::Ptr(_),
                        }, crate::native::tir::TirOperand::Value {
                            name: rhs_name,
                            ty: LlType::Ptr(_),
                        }] = inst.operands.as_slice()
                        {
                            self.pointer_payload_values.insert(lhs_name.clone());
                            self.pointer_payload_values.insert(rhs_name.clone());
                        }
                    }
                    if let Some(result) = &inst.result {
                        if matches!(inst.result_ty, Some(LlType::Ptr(_))) {
                            if let Some((true_value, false_value)) = inst.select_arms().as_deref() {
                                self.forward_pointer_selects.insert(
                                    result.clone(),
                                    (true_value.clone(), false_value.clone()),
                                );
                                if let Some(condition) = inst
                                    .operands
                                    .first()
                                    .and_then(|operand| operand.as_typed_value())
                                {
                                    self.forward_pointer_select_conditions
                                        .insert(result.clone(), condition);
                                }
                            }
                        }
                        if let Some((_, incoming)) = &inst.phi_incoming() {
                            self.tir_phi_incomings
                                .insert(result.clone(), incoming.clone());
                        }
                        if let Some(result_ty) = &inst.result_ty {
                            self.tir_result_types
                                .insert(result.clone(), result_ty.clone());
                        }
                        if let Some(pred) = &inst.cmp_predicate() {
                            self.tir_predicates.insert(result.clone(), pred.clone());
                        }
                        // `mem_align` is `Some`/None only for load/store; store is result-LESS, so the
                        // only result-keyed entries that ever carry an alignment are loads. The inert
                        // `None` entries for non-load results are never read (`mem_align_of` is called
                        // solely on the load path).
                        self.tir_aligns.insert(result.clone(), inst.mem_align());
                        // `gep_source_ty` is `Some` only for getelementptr results; the inert `None`
                        // entries for other results are never read (the gep emitter is the only consumer).
                        if let Some(src) = inst.gep_source_ty() {
                            self.tir_gep_source_types
                                .insert(result.clone(), src.clone());
                        }
                    }
                    // Historical note: the result-LESS store/void-call operand queues once built here are
                    // retired. The graph walk drives store and void-call straight off the carriers
                    // (`inst.operands`/`inst.call()`/`inst.mem_align()`), so no per-instruction operand queue
                    // survives.
                }
            }
        }
        if self.bda_device_pointers {
            let seeds = self
                .raw_buffer_params
                .iter()
                .cloned()
                .chain(self.bda_inttoptr_sources.keys().cloned())
                .chain(self.bda_direct_addresses.keys().cloned())
                .collect::<HashSet<_>>();
            self.bda_forward_addresses
                .extend(bda_forward_address_values(
                    &tir,
                    &seeds,
                    &self.opaque_resource_pointers,
                ));
        }
        let mut use_counts = HashMap::<String, usize>::new();
        for inst in tir.blocks.iter().flat_map(|block| &block.insts) {
            inst.visit_uses(|name| *use_counts.entry(name.to_string()).or_default() += 1);
        }
        self.opaque_resource_payload_loads
            .extend(
                self.bda_inttoptr_sources
                    .iter()
                    .filter_map(|(pointer, source)| {
                        let LlValue::Local(integer) = &source.value else {
                            return None;
                        };
                        (self.opaque_resource_pointers.contains(pointer)
                            && use_counts.get(integer) == Some(&1))
                        .then(|| (integer.clone(), pointer.clone()))
                    }),
            );
        self.seed_network_storage(&tir, &body_blocks, &f.params);
        // M-A3: the tir carrier is now the SOLE source of the pointer-`phi` membership sets
        // (`pointer_phi_values`/`pointer_phi_incoming_values`) and the `forward_geps` map — the legacy
        // standalone `body_blocks` text-walks that mirrored `collect_pointer_phi_sets`/
        // `collect_forward_geps` are retired. The tir graph is always built (a build failure returned
        // `Err` above), so these sets always reflect the graph.
        for network in
            crate::native::emitter::pointer_network::null_rooted_pointer_networks(&body_blocks)
        {
            for member in &network {
                self.null_rooted_pointer_values.insert(member.clone());
                self.null_rooted_pointer_peers
                    .insert(member.clone(), network.clone());
            }
        }
        self.seed_network_pointees(&body_blocks);
        // R3 STRUCTURAL: drive the emission walk from the typed-IR graph's per-block instruction list
        // (`tir.blocks[i].insts`) — the emission substrate is the typed graph, not a raw line stream
        // (`LlFunction.body` is deleted; text is read once, at parse). Emission sources
        // every opcode's OPERANDS from this same graph, and now the INSTRUCTION STREAM itself. The graph
        // was built from `body_blocks` above, so `tir.blocks[i]` aligns with `body_blocks[i]`. Each
        // straight-line instruction emits from its typed carriers; the block terminator is emitted after
        // the straight-line stream (terminators are not in `insts`) entirely from typed state — the
        // structured `TirTerminator` (`br`/`unreachable`) plus the `ret`/`switch` operand carriers
        // (`TirBlock.ret`/`switch`), no raw terminator line. There is no raw-line fallback: a tir-build
        // failure already returned `Err` above.
        for block in &body_blocks {
            let id = self.fresh();
            self.block_labels.insert(block.name.clone(), id);
        }
        let mut blocks = Vec::with_capacity(body_blocks.len());
        for (block_idx, body_block) in body_blocks.iter().enumerate() {
            self.current_block = Some(body_block.name.clone());
            let label = *self
                .block_labels
                .get(&body_block.name)
                .ok_or_else(|| format!("native emitter: missing block {}", body_block.name))?;
            if crate::env_vars::spi_why() {
                self.module.debug_names.push(Self::inst(
                    Op::Name,
                    None,
                    None,
                    vec![
                        Operand::IdRef(label),
                        Operand::LiteralString(format!("block {}", body_block.name)),
                    ],
                ));
            }
            let mut instructions = Vec::new();
            if self.bda_device_pointers && block_idx == 0 {
                let mut direct_buffers = self
                    .raw_offsets
                    .iter()
                    .filter_map(|(name, raw)| {
                        let &param_index = self.direct_param_indices.get(name)?;
                        (raw.root == *name
                            && raw.const_off == 0
                            && raw.dyn_terms.is_empty()
                            && (self.data_buffer_params.contains(name)
                                || self
                                    .ir
                                    .metadata_fc_buffer_locations
                                    .contains_key(&(f.name.clone(), name.clone()))))
                        .then(|| (param_index, name.clone()))
                    })
                    .collect::<Vec<_>>();
                // `raw_offsets` is a hash map, and this loop decides the order the entry prologue
                // materializes buffer addresses in -- which fixes their positions in the emitted
                // module and, downstream, which address-table slot each position reads. Taking that
                // order from the map makes the whole translation differ from run to run for the
                // same input. Order by the entry parameter ordinal instead: it is stable, and it is
                // the order a reader expects the prologue to be in.
                direct_buffers.sort_unstable();
                for (_, name) in direct_buffers {
                    let (low, high) =
                        self.emit_direct_buffer_address_payload(&name, &mut instructions)?;
                    let address =
                        self.combine_pointer_payload_words(low, high, &mut instructions)?;
                    self.pointer_payload_words.insert(name.clone(), (low, high));
                    self.bda_direct_addresses.insert(name, address);
                }
            }
            let tir_block = &tir.blocks[block_idx];
            for inst in &tir_block.insts {
                // M-A4: dispatch each instruction through the graph-driven `emit_body_inst`, which
                // drives every opcode family straight off the typed `TirInst`. An unmigrated opcode or
                // absent carrier (unreachable in well-formed AIR) is a fail-visible `Err`, not a text fallback.
                self.emit_body_inst(inst, &mut instructions)?;
            }
            let phi_count = instructions
                .iter()
                .take_while(|instruction| instruction.class.opcode == Op::Phi)
                .count();
            instructions.splice(phi_count..phi_count, self.phi_result_instructions.drain(..));
            self.emit_terminator(
                &tir_block.terminator,
                &tir_block.ret,
                &tir_block.switch,
                &mut instructions,
            )?;
            blocks.push(Block {
                label: Some(Self::inst(Op::Label, None, Some(label), vec![])),
                instructions,
            });
        }
        // Phi incoming values are edge uses. Their materializations therefore belong to the named
        // predecessor, even when the phi's destination block is visited first. Attach the explicitly
        // owned instructions now that every block exists; this is construction, not a module scan or a
        // dominance-based relocation guess.
        self.attach_phi_edge_instructions(&mut blocks)?;
        let inline_parameter_substitutions = self
            .inline_parameter_substitutions
            .iter()
            .copied()
            .collect::<HashMap<_, _>>();
        self.lower_inline_bda_access_chains(&mut blocks, &inline_parameter_substitutions)?;
        apply_inline_parameter_substitutions(&mut blocks, &inline_parameter_substitutions)?;
        self.emit_sidecar.remap_ids(&inline_parameter_substitutions);
        blocks = self.finalize_emitted_block_structure(blocks)?;
        self.lower_inline_bda_access_chains(&mut blocks, &inline_parameter_substitutions)?;
        let mut local_pointer_table_roots = self
            .local_pointer_fields
            .keys()
            .map(|field| field.root)
            .collect::<HashSet<_>>();
        local_pointer_table_roots.extend(
            self.emit_sidecar
                .local_pointer_field_stores
                .iter()
                .map(|fact| fact.root),
        );
        local_pointer_table_roots.extend(
            self.emit_sidecar
                .local_pointer_field_loads
                .iter()
                .map(|fact| fact.root),
        );
        local_pointer_table_roots.extend(
            self.emit_sidecar
                .local_pointer_dynamic_field_loads
                .iter()
                .map(|fact| fact.root),
        );
        retire_dead_local_pointer_table_projections(
            &mut blocks,
            &local_pointer_table_roots,
            &self.emit_sidecar.referenced_ids(),
            &mut self.module.debug_names,
            &mut self.module.annotations,
        );
        // Final block splitting can synthesize a conditional after source-plan admission. Enforce
        // the SPIR-V header contract on the exact owned instruction graph: every conditional/switch
        // terminator must be immediately preceded by SelectionMerge or LoopMerge. A missing owner
        // selects bounded construction before this module can be serialized or validated.
        let unowned_headers = crate::native::unowned_selection_header_labels(&blocks);
        if !unowned_headers.is_empty() {
            self.emit_sidecar
                .post_lowering_cfg_construction_functions
                .insert(f.name.clone());
        }
        // A construct-tree rewrite can carry a symbolic raw byte cursor through synthesized merge
        // phis. The cursor's concrete descriptor root lives in `raw_offsets`, not in its Private
        // placeholder value, so that rewritten pointer phi is not a complete Logical-addressing
        // representation. Select the raw relooper feed while the source ownership fact is still
        // available; that feed preserves the original raw-offset use graph for final construction.
        if self.construct_tree_active && !self.raw_offsets.is_empty() {
            self.emit_sidecar
                .ownership_plan_rejected_functions
                .insert(f.name.clone());
        }
        self.current_block = None;

        if crate::env_vars::ptr_network_why() {
            self.report_pointer_networks(&f.name, &body_blocks);
        }

        self.module.functions.push(Function {
            def: Some(Self::inst(
                Op::Function,
                Some(ret_id),
                Some(func_id),
                vec![
                    Operand::FunctionControl(FunctionControl::NONE),
                    Operand::IdRef(fn_ty),
                ],
            )),
            end: Some(Self::inst(Op::FunctionEnd, None, None, vec![])),
            parameters: params,
            blocks,
        });
        if self.capture_storage {
            // M1 storage-carrier measurement: record this function's final storage derivation. On a
            // structurally raw-selected function this is the sole emission, so the snapshot reflects
            // exactly the storage model chosen by typed AIR analysis.
            self.storage_snapshots
                .push((f.name.clone(), self.pointer_storage.clone()));
        }
        if self.capture_pointees {
            // M2 pointee-carrier measurement: record this function's final per-value pointee
            // derivation from that same sole emission. Compared against the from-tir `use_pointees`
            // carrier in `tir_pointee_check`.
            self.pointee_snapshots
                .push((f.name.clone(), self.pointer_pointees.clone()));
        }
        self.bda_address_values
            .extend(self.bda_direct_addresses.values().copied());
        for (name, address) in &self.bda_direct_addresses {
            if let Some((pointer, LlType::Ptr(_))) = self.values.get(name) {
                self.bda_pointer_addresses.insert(*pointer, *address);
            }
        }
        for (name, raw) in &self.raw_offsets {
            let Some(address) = raw.device_addr_base else {
                continue;
            };
            if let Some((pointer, LlType::Ptr(_))) = self.values.get(name) {
                self.bda_pointer_addresses.insert(*pointer, address);
            }
        }
        self.opaque_resource_ids.extend(
            self.opaque_resource_pointers
                .iter()
                .filter_map(|name| self.values.get(name).map(|(id, _)| *id)),
        );
        Ok(())
    }

    /// Preseed one storage class across each pointer phi/select component when the typed graph and
    /// already-established parameter/global facts agree. This removes instruction-order dependence
    /// from cyclic construct-tree state phis without coercing a genuinely mixed-storage merge.
    fn seed_network_storage(
        &mut self,
        tir: &crate::native::tir::TirFunction,
        body_blocks: &[BodyBlock],
        params: &[(String, LlType)],
    ) {
        use crate::native::emitter::pointer_network::analyze_pointer_networks;

        let derived = crate::native::tir::derive_pointer_storage_from(
            tir,
            params,
            &self.ir.types,
            &self.pointer_storage,
        );
        let empty_pointees = HashMap::new();
        for network in analyze_pointer_networks(body_blocks, &empty_pointees) {
            let storages = network
                .members
                .iter()
                .filter_map(|member| derived.get(member).copied())
                .collect::<BTreeSet<_>>();
            let mut storages = storages.into_iter();
            let Some(storage) = storages.next() else {
                continue;
            };
            if storages.next().is_some() {
                continue;
            }
            for member in network.members {
                self.pointer_storage.entry(member).or_insert(storage);
            }
        }
    }

    fn lower_inline_bda_access_chains(
        &mut self,
        blocks: &mut [Block],
        substitutions: &HashMap<Word, Word>,
    ) -> Result<(), String> {
        if substitutions.is_empty() || !self.bda_device_pointers {
            return Ok(());
        }

        let mut value_types = HashMap::new();
        for instruction in self
            .module
            .types_global_values
            .iter()
            .chain(self.module.functions.iter().flat_map(|function| {
                function.parameters.iter().chain(
                    function
                        .blocks
                        .iter()
                        .flat_map(|block| block.instructions.iter()),
                )
            }))
            .chain(blocks.iter().flat_map(|block| block.instructions.iter()))
        {
            if let (Some(result), Some(result_type)) =
                (instruction.result_id, instruction.result_type)
            {
                value_types.insert(result, result_type);
            }
        }
        let pointer_pointees = self
            .interner
            .ptr_types
            .iter()
            .map(|((_, pointee), id)| (*id, pointee.clone()))
            .collect::<HashMap<_, _>>();
        let int64_types = self
            .interner
            .types
            .iter()
            .filter_map(|(ty, id)| matches!(ty, LlType::Int(64)).then_some(*id))
            .chain(
                self.interner
                    .signed_int_types
                    .iter()
                    .filter_map(|(ty, id)| matches!(ty, LlType::Int(64)).then_some(*id)),
            )
            .collect::<HashSet<_>>();
        let zero = self.const_uint(0)?;

        for block in blocks {
            let mut instruction_index = 0;
            while instruction_index < block.instructions.len() {
                let instruction = block.instructions[instruction_index].clone();
                if !matches!(
                    instruction.class.opcode,
                    Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
                ) {
                    instruction_index += 1;
                    continue;
                }
                let Some(Operand::IdRef(base)) = instruction.operands.first() else {
                    instruction_index += 1;
                    continue;
                };
                let replacement = resolve_inline_substitution(*base, substitutions)?;
                let replacement_is_address = self.bda_address_values.contains(&replacement)
                    || self
                        .bda_direct_addresses
                        .values()
                        .any(|id| *id == replacement)
                    || self
                        .raw_offsets
                        .values()
                        .any(|raw| raw.device_addr_base == Some(replacement));
                let pointer_base_pointee = value_types
                    .get(base)
                    .and_then(|base_type| pointer_pointees.get(base_type))
                    .cloned();
                let direct_address_base = value_types
                    .get(base)
                    .is_some_and(|base_type| int64_types.contains(base_type));
                let direct_linear_index = match instruction.operands.as_slice() {
                    [Operand::IdRef(_), Operand::IdRef(index)]
                        if instruction.class.opcode == Op::PtrAccessChain =>
                    {
                        Some(*index)
                    }
                    [Operand::IdRef(_), Operand::IdRef(first), Operand::IdRef(index)]
                        if *first == zero =>
                    {
                        Some(*index)
                    }
                    _ => None,
                };
                if !((replacement != *base
                    && replacement_is_address
                    && pointer_base_pointee.is_some())
                    || (direct_address_base && direct_linear_index.is_some()))
                {
                    instruction_index += 1;
                    continue;
                }
                let Some(result_pointee) = instruction
                    .result_type
                    .and_then(|result_type| pointer_pointees.get(&result_type))
                    .cloned()
                else {
                    instruction_index += 1;
                    continue;
                };
                let base_pointee = pointer_base_pointee
                    .clone()
                    .unwrap_or_else(|| result_pointee.clone());

                let physical_base_type =
                    self.ptr_type_id(StorageClass::PhysicalStorageBuffer, &base_pointee)?;
                let physical_result_type =
                    self.ptr_type_id(StorageClass::PhysicalStorageBuffer, &result_pointee)?;
                let physical_base = self.fresh();
                block.instructions.insert(
                    instruction_index,
                    Self::inst(
                        Op::ConvertUToPtr,
                        Some(physical_base_type),
                        Some(physical_base),
                        vec![Operand::IdRef(replacement)],
                    ),
                );
                let access = &mut block.instructions[instruction_index + 1];
                if let Some(index) = direct_linear_index.filter(|_| direct_address_base) {
                    *access = Self::inst(
                        Op::PtrAccessChain,
                        Some(physical_result_type),
                        access.result_id,
                        vec![Operand::IdRef(physical_base), Operand::IdRef(index)],
                    );
                } else {
                    access.result_type = Some(physical_result_type);
                    access.operands[0] = Operand::IdRef(physical_base);
                }
                self.used_device_address = true;
                instruction_index += 2;
            }
        }
        Ok(())
    }

    pub(in crate::native) fn attach_phi_edge_instructions(
        &mut self,
        blocks: &mut [Block],
    ) -> Result<(), String> {
        for block in blocks {
            let Some(label) = block.label.as_ref().and_then(|label| label.result_id) else {
                continue;
            };
            let Some(edge_instructions) = self.phi_edge_instructions.remove(&label) else {
                continue;
            };
            let terminator = block.instructions.len().saturating_sub(1);
            let insert_at = if terminator > 0
                && matches!(
                    block.instructions[terminator - 1].class.opcode,
                    Op::SelectionMerge | Op::LoopMerge
                ) {
                terminator - 1
            } else {
                terminator
            };
            block
                .instructions
                .splice(insert_at..insert_at, edge_instructions);
        }
        if self.phi_edge_instructions.is_empty() {
            return Ok(());
        }
        let mut labels = self
            .phi_edge_instructions
            .keys()
            .copied()
            .collect::<Vec<_>>();
        labels.sort();
        Err(format!(
            "native emitter: phi incoming materializations reference missing predecessor ids: {labels:?}"
        ))
    }

    /// Materialize instruction-local control flow as real blocks before the function leaves the
    /// emitter. When that lowering splits a source loop header, retain its phis and loop ownership
    /// on a dedicated header whose sole successor is the lowered body; backedges continue to target
    /// that source header.
    fn finalize_emitted_block_structure(
        &mut self,
        source_blocks: Vec<Block>,
    ) -> Result<Vec<Block>, String> {
        let mut finalized = Vec::new();
        let mut selections_sharing_loop_merge = Vec::new();
        let source_labels = self.block_labels.values().copied().collect::<HashSet<_>>();
        let mut source_exit_labels = HashMap::new();
        for source in source_blocks {
            let has_embedded_label = source
                .instructions
                .iter()
                .any(|instruction| instruction.class.opcode == Op::Label);
            let plain_loop_merge_target = source
                .instructions
                .iter()
                .find(|instruction| instruction.class.opcode == Op::LoopMerge)
                .and_then(|instruction| instruction.operands.first())
                .and_then(|operand| match operand {
                    Operand::IdRef(target) => Some(*target),
                    _ => None,
                });
            let plain_terminator_targets = source
                .instructions
                .last()
                .map(|instruction| match instruction.class.opcode {
                    Op::BranchConditional => instruction.operands.get(1..3).unwrap_or_default(),
                    Op::Switch => instruction.operands.get(1..).unwrap_or_default(),
                    _ => &[],
                })
                .unwrap_or_default();
            let plain_header_targets_loop_merge = plain_terminator_targets.iter().any(|operand| {
                matches!(
                    operand,
                    Operand::IdRef(target) if Some(*target) == plain_loop_merge_target
                )
            });
            let plain_header_has_store = source
                .instructions
                .iter()
                .any(|instruction| instruction.class.opcode == Op::Store);
            // A pure exit-test header already has loop ownership from OpLoopMerge. Keep that compact
            // shape so fixed-loop recognition still sees the source latch. A header that performs
            // stores before the exit test needs a dedicated body: its side effects belong to the
            // loop region rather than the structural header.
            let needs_plain_loop_header_split = plain_loop_merge_target.is_some()
                && !plain_terminator_targets.is_empty()
                && source
                    .instructions
                    .iter()
                    .all(|instruction| instruction.class.opcode != Op::SelectionMerge)
                && (!plain_header_targets_loop_merge || plain_header_has_store);
            if !has_embedded_label && !needs_plain_loop_header_split {
                finalized.push(source);
                continue;
            }

            let source_label = source
                .label
                .clone()
                .ok_or_else(|| "native emitter: split source block has no label".to_string())?;
            let mut segments = Vec::new();
            let mut current = Block {
                label: Some(source_label.clone()),
                instructions: Vec::new(),
            };
            for instruction in source.instructions {
                if instruction.class.opcode == Op::Label {
                    if !current
                        .instructions
                        .last()
                        .is_some_and(|candidate| is_block_terminator(candidate.class.opcode))
                    {
                        return Err(
                            "native emitter: embedded label does not follow a terminator"
                                .to_string(),
                        );
                    }
                    segments.push(current);
                    current = Block {
                        label: Some(instruction),
                        instructions: Vec::new(),
                    };
                } else {
                    current.instructions.push(instruction);
                }
            }
            segments.push(current);

            let source_label_id = source_label
                .result_id
                .ok_or_else(|| "native emitter: split source label has no result id".to_string())?;
            let exit_label = segments
                .last()
                .and_then(|block| block.label.as_ref())
                .and_then(|label| label.result_id)
                .ok_or_else(|| "native emitter: split source exit has no label".to_string())?;
            if source_label_id != exit_label {
                source_exit_labels.insert(source_label_id, exit_label);
            }

            let loop_claims = segments
                .iter()
                .enumerate()
                .flat_map(|(block_index, block)| {
                    block
                        .instructions
                        .iter()
                        .enumerate()
                        .filter(|(_, instruction)| instruction.class.opcode == Op::LoopMerge)
                        .map(move |(instruction_index, _)| (block_index, instruction_index))
                })
                .collect::<Vec<_>>();
            if let [(claim_block, claim_instruction)] = loop_claims.as_slice() {
                let loop_merge = segments[*claim_block]
                    .instructions
                    .remove(*claim_instruction);
                let merge_target = loop_merge
                    .operands
                    .first()
                    .and_then(|operand| match operand {
                        Operand::IdRef(target) => Some(*target),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        "native emitter: emitted OpLoopMerge has no id merge target".to_string()
                    })?;
                let phi_count = segments[0]
                    .instructions
                    .iter()
                    .take_while(|instruction| instruction.class.opcode == Op::Phi)
                    .count();
                let leading_phis = segments[0]
                    .instructions
                    .drain(..phi_count)
                    .collect::<Vec<_>>();
                let body_label = self.fresh();
                segments[0].label = Some(Self::inst(Op::Label, None, Some(body_label), vec![]));

                let claim_header = segments[*claim_block]
                    .label
                    .as_ref()
                    .and_then(|label| label.result_id)
                    .ok_or_else(|| "native emitter: split loop body has no label".to_string())?;
                if !has_embedded_label {
                    // The original loop header's conditional now leaves `claim_header`; successor
                    // phis must name that real predecessor rather than the dedicated loop header,
                    // whose only edge is the new unconditional entry edge.
                    source_exit_labels.insert(source_label_id, claim_header);
                }
                let claim_instructions = &mut segments[*claim_block].instructions;
                let terminator_index =
                    claim_instructions.len().checked_sub(1).ok_or_else(|| {
                        "native emitter: split loop body has no terminator".to_string()
                    })?;
                if matches!(
                    claim_instructions[terminator_index].class.opcode,
                    Op::BranchConditional | Op::Switch
                ) && !claim_instructions
                    .iter()
                    .any(|instruction| instruction.class.opcode == Op::SelectionMerge)
                {
                    claim_instructions.insert(
                        terminator_index,
                        Self::inst(
                            Op::SelectionMerge,
                            None,
                            None,
                            vec![
                                Operand::IdRef(merge_target),
                                Operand::SelectionControl(SelectionControl::NONE),
                            ],
                        ),
                    );
                    selections_sharing_loop_merge.push((claim_header, merge_target));
                }

                let mut header_instructions = leading_phis;
                header_instructions.push(loop_merge);
                header_instructions.push(Self::inst(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(body_label)],
                ));
                finalized.push(Block {
                    label: Some(source_label),
                    instructions: header_instructions,
                });
            } else if !loop_claims.is_empty() {
                return Err(
                    "native emitter: one source block emitted multiple loop claims".to_string(),
                );
            }
            finalized.extend(segments);
        }
        // A source TIR block can expand into several real SPIR-V blocks when an instruction owns
        // local control flow. Its outgoing source-CFG edge then leaves the final segment, not the
        // source entry label initially allocated in `block_labels`. Rewrite only phis belonging to
        // source blocks: instruction-local phis live on fresh labels and retain their exact internal
        // predecessors.
        for block in &mut finalized {
            let Some(label) = block.label.as_ref().and_then(|label| label.result_id) else {
                continue;
            };
            if !source_labels.contains(&label) {
                continue;
            }
            for phi in block
                .instructions
                .iter_mut()
                .take_while(|instruction| instruction.class.opcode == Op::Phi)
            {
                for pair in phi.operands.chunks_mut(2) {
                    let [_, Operand::IdRef(predecessor)] = pair else {
                        return Err("native emitter: malformed source phi".to_string());
                    };
                    if let Some(exit_label) = source_exit_labels.get(predecessor) {
                        *predecessor = *exit_label;
                    }
                }
            }
        }
        for (header, target) in selections_sharing_loop_merge {
            self.privatize_emitted_selection_merge(&mut finalized, header, target)?;
        }
        Ok(finalized)
    }

    /// Give a selection introduced inside a split loop header a private pass-through merge. The
    /// outer loop retains its original merge, while all selection-owned edges and target phis are
    /// routed through the new boundary.
    fn privatize_emitted_selection_merge(
        &mut self,
        blocks: &mut Vec<Block>,
        header: Word,
        target: Word,
    ) -> Result<(), String> {
        let labels = blocks
            .iter()
            .filter_map(|block| block.label.as_ref()?.result_id)
            .collect::<Vec<_>>();
        let successors = blocks
            .iter()
            .filter_map(|block| {
                let label = block.label.as_ref()?.result_id?;
                Some((label, emitted_block_successors(block)))
            })
            .collect::<HashMap<_, _>>();
        if labels.is_empty() {
            return Err("native emitter: split function has no entry block".to_string());
        }
        let dominators = crate::native::cfg::EmittedDominators::new(&labels, &successors);
        let redirected = blocks
            .iter()
            .filter_map(|block| {
                let label = block.label.as_ref()?.result_id?;
                (dominators.dominates(header, label)
                    && !dominators.dominates(target, label)
                    && emitted_block_successors(block).contains(&target))
                .then_some(label)
            })
            .collect::<HashSet<_>>();
        if redirected.is_empty() {
            return Err("native emitter: split selection has no owned merge edge".to_string());
        }

        let target_index = blocks
            .iter()
            .position(|block| {
                block.label.as_ref().and_then(|label| label.result_id) == Some(target)
            })
            .ok_or_else(|| "native emitter: split selection merge block is missing".to_string())?;
        let synthetic_label = self.fresh();
        let mut synthetic_instructions = Vec::new();
        let mut phi_updates = Vec::new();
        for (instruction_index, instruction) in blocks[target_index]
            .instructions
            .iter()
            .take_while(|instruction| instruction.class.opcode == Op::Phi)
            .enumerate()
        {
            let mut kept = Vec::new();
            let mut routed = Vec::new();
            for pair in instruction.operands.chunks(2) {
                let [_, Operand::IdRef(predecessor)] = pair else {
                    return Err("native emitter: malformed phi at split loop merge".to_string());
                };
                if redirected.contains(predecessor) {
                    routed.extend_from_slice(pair);
                } else {
                    kept.extend_from_slice(pair);
                }
            }
            if routed.is_empty() {
                continue;
            }
            let routed_value = if routed.len() == 2 {
                routed[0].clone()
            } else {
                let result = self.fresh();
                synthetic_instructions.push(Self::inst(
                    Op::Phi,
                    instruction.result_type,
                    Some(result),
                    routed,
                ));
                Operand::IdRef(result)
            };
            kept.push(routed_value);
            kept.push(Operand::IdRef(synthetic_label));
            phi_updates.push((instruction_index, kept));
        }
        synthetic_instructions.push(Self::inst(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(target)],
        ));

        let header_block = blocks
            .iter_mut()
            .find(|block| block.label.as_ref().and_then(|label| label.result_id) == Some(header))
            .ok_or_else(|| "native emitter: split selection header is missing".to_string())?;
        let selection_merge = header_block
            .instructions
            .iter_mut()
            .find(|instruction| {
                instruction.class.opcode == Op::SelectionMerge
                    && instruction.operands.first() == Some(&Operand::IdRef(target))
            })
            .ok_or_else(|| "native emitter: split selection claim is missing".to_string())?;
        selection_merge.operands[0] = Operand::IdRef(synthetic_label);
        for block in blocks.iter_mut() {
            let label = block.label.as_ref().and_then(|label| label.result_id);
            if label.is_some_and(|label| redirected.contains(&label)) {
                redirect_emitted_terminator_target(block, target, synthetic_label);
            }
        }
        for (instruction_index, operands) in phi_updates {
            blocks[target_index].instructions[instruction_index].operands = operands;
        }
        blocks.insert(
            target_index,
            Block {
                label: Some(Self::inst(Op::Label, None, Some(synthetic_label), vec![])),
                instructions: synthetic_instructions,
            },
        );
        Ok(())
    }

    /// Insert calls to AIR module static initializers at the selected entry's first executable
    /// instruction. This is the emitter-side form of the retired post-serialization SPIR-V pass:
    /// source function order is preserved, calls follow leading Function `OpVariable`s, and ids are
    /// allocated after ordinary emission/rewrite ids so canonical output stays byte-identical.
    pub(super) fn inject_static_initializer_calls(
        &mut self,
        entry_name: Option<&str>,
        initializer_names: &[String],
    ) -> Result<(), String> {
        let Some(entry_name) = entry_name else {
            return Ok(());
        };
        let entry_id = *self
            .function_ids
            .get(entry_name)
            .ok_or_else(|| format!("native emitter: missing function id for {entry_name}"))?;
        let initializer_ids = initializer_names
            .iter()
            .map(|initializer_name| {
                self.function_ids
                    .get(initializer_name)
                    .copied()
                    .ok_or_else(|| {
                        format!("native emitter: missing function id for {initializer_name}")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if initializer_ids.is_empty() {
            return Ok(());
        }

        let void = self.type_id(&LlType::Void)?;
        let calls = initializer_ids
            .into_iter()
            .map(|callee| {
                Self::inst(
                    Op::FunctionCall,
                    Some(void),
                    Some(self.fresh()),
                    vec![Operand::IdRef(callee)],
                )
            })
            .collect::<Vec<_>>();
        let entry = self
            .module
            .functions
            .iter_mut()
            .find(|function| {
                function.def.as_ref().and_then(|def| def.result_id) == Some(entry_id)
                    && !function.blocks.is_empty()
            })
            .ok_or_else(|| {
                format!(
                    "native emitter: emitted entry function {} not found",
                    entry_name
                )
            })?;
        let first_block = entry
            .blocks
            .first_mut()
            .ok_or_else(|| format!("native emitter: entry function {} has no block", entry_id))?;
        let insert_at = first_block
            .instructions
            .iter()
            .take_while(|instruction| instruction.class.opcode == Op::Variable)
            .count();
        first_block.instructions.splice(insert_at..insert_at, calls);
        Ok(())
    }

    /// The IR use-implied pointee per pointer value — the granularity BEFORE the byte-view flattening
    /// that records `Int(8)` at def time. Built from the tir `use_pointees` carrier (resolved through
    /// named aliases), falling back to the recorded `pointer_pointees` for values the carrier does not
    /// cover. This is the source the M-A2 network fix must record from: `pointer_pointees` alone
    /// byte-flattens the byte-addressed arm of a whole-vs-part network to `Int(8)`, disguising it as a
    /// reinterpret-mix.
    fn use_implied_pointees(&self) -> HashMap<String, LlType> {
        let mut m = self.pointer_pointees.clone();
        for (name, ty) in &self.tir_use_pointees {
            if let Ok(resolved) = self.resolve_type(ty) {
                m.insert(name.clone(), resolved);
            }
        }
        m
    }

    /// M-A2 def-site recording: seed `network_pointees` with the uniform ACCESS pointee
    /// for every pointer network whose TRUE IR access granularity is CONSISTENT across the whole
    /// component (census class `Uniform`, one concrete non-byte pointee). `pointer_meta_for_value` then
    /// reports that type for every member, so `pointer_merge_meta` reconciles the phi/select on it
    /// instead of erroring on the byte-view `Int(8)` the raw recording flattens the byte-addressed arm
    /// to. Restricted to the access-uniform case: every member's real access already matches the seeded
    /// type, so no load/store retyping (scalarization) and no GEP re-striding is needed — the
    /// mixed-granularity (whole-vs-part) minority is left alone. Default-off; byte-changing when set, so
    /// flip is G7/G8-gated.
    fn seed_network_pointees(&mut self, body_blocks: &[BodyBlock]) {
        use crate::native::emitter::pointer_network::{
            analyze_networks_by_access, array_indexed_scalar_bases, NetworkClass,
        };
        // Pointers stepped as an ARRAY of their bare scalar element (a non-identity scalar-stride GEP):
        // recording the scalar pointee mis-declares the object as a scalar `OpVariable`, so any network
        // touching one is excluded here (it needs the object re-declared as an array + indices
        // re-strided — the unbuilt M-A2(c) #2/#3 keystone). Excluding keeps the seed a STRICT SUBSET of
        // the sound access-uniform set, so it can only reduce fails, never add.
        let array_indexed = array_indexed_scalar_bases(body_blocks);
        // Census by the TRUE IR ACCESS WIDTH (loads/stores/gep-steps), NOT the element-scalar carrier.
        // The carrier (`analyze_pointer_networks` + `use_implied_pointees`) reports the innermost scalar,
        // so it flattens a real whole-vs-part component (`{Vector(Float,4), Float}` accesses) to
        // `Uniform [Float]` and seeds the scalar — but the GEP/load def sites still derive
        // `Vector(Float,4)` fresh from the LLVM source type (`gep_pointee`), so recording the scalar
        // leaves the pointer PARTIALLY retyped and emits invalid SPIR-V (measured: 11 frontier
        // regressions). The access census classifies that same component `WholeVsPart(Float)` (non-
        // Uniform), so it is SKIPPED here; only a genuinely access-uniform network — every member
        // dereferenced/stepped at ONE type — is seeded, where recording that whole type cannot disagree
        // with any def site. Sound consistent widening of the whole-vs-part networks needs coordinated
        // GEP re-striding (the remaining M-A2(c) keystone work), not a read-side seed.
        for net in analyze_networks_by_access(body_blocks) {
            if matches!(net.class, NetworkClass::WholeVsPart(LlType::Half))
                && net
                    .members
                    .iter()
                    .all(|member| self.null_rooted_pointer_values.contains(member))
            {
                // A closed null-rooted half/half-vector network cannot use a native scalar-half
                // carrier: that leaves vector accesses wider than their pointer type. Record the
                // byte-view carrier uniformly at every def site until the guarded dead paths are
                // removed. Concrete-rooted networks are deliberately excluded.
                for member in &net.members {
                    self.network_pointees.insert(member.clone(), LlType::Int(8));
                }
                continue;
            }
            if !matches!(net.class, NetworkClass::Uniform) {
                continue;
            }
            // Uniform means 0 or 1 distinct access pointee. Seed only when there IS one concrete,
            // non-byte-view pointee to record (an empty or `Int(8)`-only network carries no widening).
            let [pointee] = net.pointees.as_slice() else {
                continue;
            };
            if matches!(pointee, LlType::Int(8)) {
                continue;
            }
            // Skip a network any of whose members is stepped as an array of its bare scalar element —
            // seeding the scalar there mis-declares the object (see `array_indexed_scalar_bases`).
            if net.members.iter().any(|m| array_indexed.contains(m)) {
                continue;
            }
            // Skip a network a member of which is a LOGICAL (non-word-addressable) pointer that already
            // carries a CONCRETE (non-byte-view) def-site pointee DISAGREEING with the uniform ACCESS
            // pointee. The access census is body-local, so it misses a whole-vs-part split whose narrow
            // evidence lives at a def site the body never re-derives — the canonical case is a Workgroup/
            // Private pointer PARAMETER (its pointee comes from the callsite/arg metadata into
            // `pointer_pointees` before this seed runs) that the body then dereferences at a wider
            // granularity (an `addrspace(3)` scalar `float*` scratch arg loaded as `<4 x float>` in a
            // helper). Recording the wide access pointee there mis-declares the arg, and a LOGICAL pointer
            // (Workgroup/Private) has NO raw-word view — re-viewing it to the wide type needs an illegal
            // `OpBitcast` on a logical pointer (`body.rs` "cannot reinterpret workgroup pointer arg"). A
            // word-addressable device pointer (`UniformConstant`/StorageBuffer) CAN be reinterpreted via
            // the raw byte-GEP model, so it is NOT excluded — that is the whole-vs-part population RECORD
            // exists to seed. Gating on logical storage (not on any disagreement) keeps the exclusion a
            // STRICT SUBSET that removes only the un-reinterpretable logical members, so it can only
            // reduce fails, never add, without gutting the device-buffer wins.
            if net.members.iter().any(|m| {
                self.pointer_pointees
                    .get(m)
                    .is_some_and(|p| !matches!(p, LlType::Int(8)) && p != pointee)
                    && matches!(
                        self.pointer_storage.get(m),
                        Some(StorageClass::Workgroup | StorageClass::Private)
                    )
            }) {
                continue;
            }
            for member in &net.members {
                self.network_pointees
                    .insert(member.clone(), pointee.clone());
            }
        }
    }

    /// `METAL2VULKAN_PTR_NETWORK_WHY` diagnostic: print each pointer network classified by the TRUE IR
    /// ACCESS WIDTH (`analyze_networks_by_access` — loads/stores/geps) as the PRIMARY tag, with the
    /// element-scalar carrier class in `carrier=` when it disagrees. The access census is the honest one:
    /// the carrier reports the innermost scalar, so it flattens a real whole-vs-part component to
    /// `Uniform [Float]` (measured: seeding that flattened scalar regresses 11 frontier cases), whereas
    /// the access census sees the `<4 x float>` load and reports `WholeVsPart(Float)`. Prints any network
    /// the access census finds non-uniform OR where the two censuses disagree. Read-only — feeds no
    /// emission. Covers functions whose emission completed (a function that errors mid-body never reaches
    /// here; those shapes are exercised by `pointer_network`'s unit tests instead).
    fn report_pointer_networks(&self, fn_name: &str, body_blocks: &[BodyBlock]) {
        use crate::native::emitter::pointer_network::{
            analyze_networks_by_access, analyze_pointer_networks, NetworkClass,
        };
        let use_implied = self.use_implied_pointees();
        let carrier_class: std::collections::HashMap<Vec<String>, NetworkClass> =
            analyze_pointer_networks(body_blocks, &use_implied)
                .into_iter()
                .map(|n| (n.members, n.class))
                .collect();
        for net in analyze_networks_by_access(body_blocks) {
            let carrier = carrier_class.get(&net.members);
            if matches!(net.class, NetworkClass::Uniform)
                && matches!(carrier, Some(NetworkClass::Uniform) | None)
            {
                continue;
            }
            let carrier_tag = match carrier {
                Some(c) if c != &net.class => format!(" carrier={c:?}"),
                _ => String::new(),
            };
            eprintln!(
                "PTR-NETWORK {fn_name} {:?}{carrier_tag} members={} pointees={:?}",
                net.class,
                net.members.len(),
                net.pointees,
            );
        }
    }

    /// Find module globals that an integer atomic (`air.atomic.*.i32`) dereferences directly, so
    /// `emit_global` can declare them with an `i32` pointee instead of their float type (the
    /// atomic-min/max bit-pattern idiom over a threadgroup scratch slot). Reasoned purely from the
    /// `air.atomic.*` ABI symbol family — the allowed structural exception (AGENTS.md) — and the
    /// operand being a bare `LlValue::Global`, never a shader name. A global also seen under a float
    /// atomic (`air.atomic.*.f32`) is excluded: retyping it to `i32` would only move the illegal
    /// pointer bitcast to the float-atomic site.
    pub(super) fn scan_int_atomic_reinterpret_globals(
        globals: &[LlGlobal],
        functions: &[LlFunction],
    ) -> HashSet<String> {
        let global_names: HashSet<&str> = globals.iter().map(|g| g.name.as_str()).collect();
        let mut int_atomic = HashSet::new();
        let mut float_atomic = HashSet::new();
        for function in functions {
            // Read the parsed call off the typed carrier (`inst.call()`) instead of re-lexing `body`. The
            // carrier's `resolve_call` is broader than the old `strip_call_prefix` (it also parses
            // `musttail`/`notail`), so gate on `opcode ∈ {call, tail}` to reproduce the old acceptance
            // exactly (`call …` / `tail call …` only) — byte-identical.
            for block in &function.blocks {
                let Some(carrier) = &block.typed else {
                    continue;
                };
                for inst in &carrier.insts {
                    if !matches!(inst.opcode.as_str(), "call" | "tail") {
                        continue;
                    }
                    let Some(call) = &inst.call() else { continue };
                    if !call.callee.starts_with("air.atomic.") {
                        continue;
                    }
                    let Some(LlValue::Global(name)) = call.args.first().map(|a| &a.value) else {
                        continue;
                    };
                    if !global_names.contains(name.as_str()) {
                        continue;
                    }
                    if call.callee.ends_with(".i32") {
                        int_atomic.insert(name.clone());
                    } else if call.callee.ends_with(".f32") {
                        float_atomic.insert(name.clone());
                    }
                }
            }
        }
        int_atomic
            .into_iter()
            .filter(|name| !float_atomic.contains(name))
            .collect()
    }

    /// The pointee type a module global is *declared* with in SPIR-V. Normally its LLVM type, except
    /// an integer atomic on a float-typed threadgroup global needs an `i32`-typed pointer to that exact
    /// memory; under Logical addressing that pointer only exists if the variable itself is declared
    /// `i32`. Retype those (Workgroup scratch only, so there is no initializer to reconcile); the float
    /// load/store value accesses then reinterpret through the existing 32-bit scalar
    /// `OpBitcast`-on-value load/store paths, and the atomic gets a clean `i32` pointer with no illegal
    /// logical-pointer bitcast. Used by `emit_global` (declaration) and the per-function pointee reset
    /// (`emit_function` clears `pointer_pointees` and reseeds globals) so both agree.
    pub(super) fn global_declared_pointee(&mut self, global: &LlGlobal) -> Result<LlType, String> {
        let ty = self.resolve_type(&global.ty)?;
        if global.addrspace == 3
            && ty == LlType::Float
            && self.int_atomic_reinterpret_globals.contains(&global.name)
        {
            return Ok(LlType::Int(32));
        }
        // A constant table accessed through a GEP whose source type is NOT the declared type (a
        // reinterpret view — e.g. a packed byte-table struct addressed as `[16 x [32 x i8]]` with a
        // dynamic row index, which is invalid as a structural chain since struct indices must be
        // constants). When every leaf of the declared type is `i8` the byte image is exact (i8
        // fields/arrays have alignment 1, so there is no padding), so declare the variable as the
        // flat byte array; every view then lowers through the byte-array raw paths.
        if global.addrspace != 3 && self.byte_view_reinterpret_globals.contains(&global.name) {
            if let Some(size) = i8_leaf_byte_size(&ty) {
                return Ok(LlType::Array(Box::new(LlType::Int(8)), size));
            }
        }
        if global.addrspace != 3 {
            if let Some(view) = self.flat_scalar_reinterpret_globals.get(&global.name) {
                return Ok(view.clone());
            }
        }
        Ok(ty)
    }

    /// Scan for globals used as the base of a `getelementptr` whose SOURCE type differs from the
    /// global's declared type — the byte-table reinterpret-view shape `global_declared_pointee`
    /// remodels to a flat byte array. Textual companion to `scan_int_atomic_reinterpret_globals`.
    pub(super) fn scan_byte_view_reinterpret_globals(&mut self) -> Result<HashSet<String>, String> {
        let globals = self.ir.globals.clone();
        let functions = self.ir.functions.clone();
        let mut reinterpreted = HashSet::new();
        let declared: HashMap<&str, &LlType> =
            globals.iter().map(|g| (g.name.as_str(), &g.ty)).collect();
        for function in &functions {
            // Read the parsed GEP off the typed carrier (`inst.gep()`, set by `resolve_gep` = the same
            // `parse_gep` on the same `after "getelementptr "` text) instead of re-lexing `body` —
            // byte-identical.
            for block in &function.blocks {
                let Some(carrier) = &block.typed else {
                    continue;
                };
                for inst in &carrier.insts {
                    let Some(gep) = &inst.gep() else { continue };
                    let LlValue::Global(base) = &gep.base.value else {
                        continue;
                    };
                    let Some(declared_ty) = declared.get(base.as_str()) else {
                        continue;
                    };
                    let declared_ty = self.resolve_type(declared_ty)?;
                    let source_ty = self.resolve_type(&gep.source_ty)?;
                    if source_ty != declared_ty && i8_leaf_byte_size(&declared_ty).is_some() {
                        reinterpreted.insert(base.clone());
                    }
                }
            }
        }
        Ok(reinterpreted)
    }

    pub(super) fn scan_flat_scalar_reinterpret_globals(
        &mut self,
    ) -> Result<HashMap<String, LlType>, String> {
        let globals = self.ir.globals.clone();
        let functions = self.ir.functions.clone();
        let mut reinterpreted = HashMap::new();
        let declared: HashMap<&str, &LlType> =
            globals.iter().map(|g| (g.name.as_str(), &g.ty)).collect();
        let functions_by_name: HashMap<&str, &LlFunction> =
            functions.iter().map(|f| (f.name.as_str(), f)).collect();
        for function in &functions {
            for block in &function.blocks {
                let Some(carrier) = &block.typed else {
                    continue;
                };
                for inst in &carrier.insts {
                    if let Some(gep) = &inst.gep() {
                        let LlValue::Global(base) = &gep.base.value else {
                            continue;
                        };
                        let Some(declared_ty) = declared.get(base.as_str()) else {
                            continue;
                        };
                        let declared_ty = self.resolve_type(declared_ty)?;
                        let source_ty = self.resolve_type(&gep.source_ty)?;
                        if source_ty != declared_ty {
                            if let Some(view) =
                                flat_scalar_reinterpret_view(&declared_ty, &source_ty)
                            {
                                reinterpreted.entry(base.clone()).or_insert(view);
                            }
                        }
                    }
                    let Some(call_result) = inst.emit_scan_call() else {
                        continue;
                    };
                    let call = call_result?;
                    let Some(callee) = functions_by_name.get(call.callee.as_str()) else {
                        continue;
                    };
                    for (index, arg) in call.args.iter().enumerate() {
                        let LlValue::Global(base) = &arg.value else {
                            continue;
                        };
                        let Some(declared_ty) = declared.get(base.as_str()) else {
                            continue;
                        };
                        let Some((param_name, _)) = callee.params.get(index) else {
                            continue;
                        };
                        let Some(expected) =
                            self.function_param_concrete_pointee(&callee.name, index, param_name)
                        else {
                            continue;
                        };
                        let declared_ty = self.resolve_type(declared_ty)?;
                        let expected = self.resolve_type(&expected)?;
                        if expected != declared_ty {
                            if let Some(view) =
                                flat_scalar_reinterpret_view(&declared_ty, &expected)
                            {
                                reinterpreted.entry(base.clone()).or_insert(view);
                            }
                        }
                    }
                }
            }
        }
        Ok(reinterpreted)
    }

    pub(super) fn emit_global(&mut self, global: &LlGlobal) -> Result<(), String> {
        let ty = self.global_declared_pointee(global)?;
        let storage = if global.addrspace == 3 {
            StorageClass::Workgroup
        } else {
            StorageClass::Private
        };
        let ptr_ty = self.ptr_type_id(storage, &ty)?;
        let initializer = if storage == StorageClass::Private {
            Some(match &global.initializer {
                Some(initializer) => {
                    let initializer_ty = self.resolve_type(&initializer.ty)?;
                    if initializer_ty != ty {
                        // Reinterpret remodels declare this global as a flat scalar array; flatten the
                        // initializer to the same byte/scalar image.
                        let LlType::Array(elem, len) = &ty else {
                            return Err(format!(
                                "native emitter: global {} initializer type {:?} does not match {:?}",
                                global.name, initializer_ty, ty
                            ));
                        };
                        if elem.as_ref() == &LlType::Int(8)
                            && self.byte_view_reinterpret_globals.contains(&global.name)
                        {
                            let mut bytes = Vec::new();
                            self.append_i8_initializer_bytes(
                                &initializer.value,
                                &initializer_ty,
                                &mut bytes,
                            )?;
                            if bytes.len() != *len as usize {
                                return Err(format!(
                                    "native emitter: global {} byte-flattened initializer is {} bytes, declared {}",
                                    global.name,
                                    bytes.len(),
                                    len
                                ));
                            }
                            let flat = LlValue::Array(
                                bytes
                                    .into_iter()
                                    .map(|byte| TypedValue {
                                        ty: LlType::Int(8),
                                        value: LlValue::Int(u64::from(byte)),
                                    })
                                    .collect(),
                            );
                            self.const_initializer_id(&flat, &ty)?
                        } else if let Some(view_ty) = self
                            .flat_scalar_reinterpret_globals
                            .get(&global.name)
                            .cloned()
                        {
                            let Some((leaf, count)) = flat_scalar_leaf_count(&initializer_ty)
                            else {
                                return Err(format!(
                                    "native emitter: global {} initializer type {:?} does not match {:?}",
                                    global.name, initializer_ty, ty
                                ));
                            };
                            let Some((view_leaf, view_count)) = flat_scalar_leaf_count(&view_ty)
                            else {
                                return Err(format!(
                                    "native emitter: global {} initializer type {:?} does not match {:?}",
                                    global.name, initializer_ty, ty
                                ));
                            };
                            if !types_compatible(&view_leaf, &leaf) || view_count != count {
                                return Err(format!(
                                    "native emitter: global {} initializer type {:?} does not match {:?}",
                                    global.name, initializer_ty, ty
                                ));
                            }
                            let mut values = Vec::new();
                            self.append_flat_scalar_initializer(
                                &initializer.value,
                                &initializer_ty,
                                &leaf,
                                &mut values,
                            )?;
                            let mut values = values.into_iter();
                            let flat = self.scalar_initializer_from_flat(&view_ty, &mut values)?;
                            self.const_initializer_id(&flat, &view_ty)?
                        } else {
                            return Err(format!(
                                "native emitter: global {} initializer type {:?} does not match {:?}",
                                global.name, initializer_ty, ty
                            ));
                        }
                    } else {
                        self.const_initializer_id(&initializer.value, &initializer.ty)?
                    }
                }
                None => self.const_null(&ty)?,
            })
        } else {
            None
        };
        let id = self.fresh();
        let mut operands = vec![Operand::StorageClass(storage)];
        if let Some(initializer) = initializer {
            operands.push(Operand::IdRef(initializer));
        }
        self.module.types_global_values.push(Self::inst(
            Op::Variable,
            Some(ptr_ty),
            Some(id),
            operands,
        ));
        self.module.debug_names.push(Self::inst(
            Op::Name,
            None,
            None,
            vec![
                Operand::IdRef(id),
                Operand::LiteralString(global.name.trim_start_matches('@').to_string()),
            ],
        ));
        self.global_values
            .insert(global.name.clone(), (id, LlType::Ptr(global.addrspace)));
        self.pointer_pointees.insert(global.name.clone(), ty);
        Ok(())
    }

    /// Serialize an all-i8-leaf constant initializer to its byte image (the flat-byte-array remodel
    /// of `global_declared_pointee`). `Zero`/`Undef` fill their type's byte size with zeros.
    fn append_i8_initializer_bytes(
        &mut self,
        value: &LlValue,
        ty: &LlType,
        out: &mut Vec<u8>,
    ) -> Result<(), String> {
        let ty = self.resolve_type(ty)?;
        match value {
            LlValue::Zero | LlValue::Undef => {
                let size = i8_leaf_byte_size(&ty)
                    .ok_or_else(|| format!("native emitter: cannot byte-flatten zero of {ty:?}"))?;
                out.extend(std::iter::repeat_n(0u8, size as usize));
                Ok(())
            }
            LlValue::Int(v) if ty == LlType::Int(8) => {
                out.push(*v as u8);
                Ok(())
            }
            LlValue::SignedInt(v) if ty == LlType::Int(8) => {
                out.push(*v as u8);
                Ok(())
            }
            LlValue::Hex(v) if ty == LlType::Int(8) => {
                out.push(*v as u8);
                Ok(())
            }
            LlValue::Array(elems) | LlValue::Struct(elems) => {
                for elem in elems {
                    self.append_i8_initializer_bytes(&elem.value, &elem.ty, out)?;
                }
                Ok(())
            }
            other => Err(format!(
                "native emitter: cannot byte-flatten initializer {other:?} of {ty:?}"
            )),
        }
    }

    fn append_flat_scalar_initializer(
        &mut self,
        value: &LlValue,
        ty: &LlType,
        leaf: &LlType,
        out: &mut Vec<TypedValue>,
    ) -> Result<(), String> {
        let ty = self.resolve_type(ty)?;
        match value {
            LlValue::Zero | LlValue::Undef => {
                let (actual_leaf, count) = flat_scalar_leaf_count(&ty).ok_or_else(|| {
                    format!("native emitter: cannot scalar-flatten zero of {ty:?}")
                })?;
                if !types_compatible(&actual_leaf, leaf) {
                    return Err(format!(
                        "native emitter: cannot scalar-flatten {ty:?} as {leaf:?}"
                    ));
                }
                out.extend(std::iter::repeat_n(
                    TypedValue {
                        ty: leaf.clone(),
                        value: LlValue::Zero,
                    },
                    count as usize,
                ));
                Ok(())
            }
            LlValue::Array(elems) | LlValue::Struct(elems) => {
                for elem in elems {
                    self.append_flat_scalar_initializer(&elem.value, &elem.ty, leaf, out)?;
                }
                Ok(())
            }
            scalar if types_compatible(&ty, leaf) => {
                out.push(TypedValue {
                    ty,
                    value: scalar.clone(),
                });
                Ok(())
            }
            other => Err(format!(
                "native emitter: cannot scalar-flatten initializer {other:?} of {ty:?} as {leaf:?}"
            )),
        }
    }

    fn scalar_initializer_from_flat<I>(
        &mut self,
        ty: &LlType,
        values: &mut I,
    ) -> Result<LlValue, String>
    where
        I: Iterator<Item = TypedValue>,
    {
        let ty = self.resolve_type(ty)?;
        match ty {
            LlType::Array(elem, len) => {
                let elem = self.resolve_type(&elem)?;
                let mut elems = Vec::with_capacity(len as usize);
                for _ in 0..len {
                    elems.push(TypedValue {
                        ty: elem.clone(),
                        value: self.scalar_initializer_from_flat(&elem, values)?,
                    });
                }
                Ok(LlValue::Array(elems))
            }
            LlType::Struct(fields) => {
                let mut elems = Vec::with_capacity(fields.len());
                for field in fields {
                    let field = self.resolve_type(&field)?;
                    elems.push(TypedValue {
                        ty: field.clone(),
                        value: self.scalar_initializer_from_flat(&field, values)?,
                    });
                }
                Ok(LlValue::Struct(elems))
            }
            scalar => {
                let value = values
                    .next()
                    .ok_or_else(|| format!("native emitter: missing scalar for {scalar:?}"))?;
                if !types_compatible(&value.ty, &scalar) {
                    return Err(format!(
                        "native emitter: scalar initializer {:?} does not match {scalar:?}",
                        value.ty
                    ));
                }
                Ok(value.value)
            }
        }
    }

    pub(super) fn function_param_concrete_pointee(
        &self,
        func: &str,
        index: usize,
        name: &str,
    ) -> Option<LlType> {
        self.function_param_pointees
            .get(&(func.to_string(), index))
            .cloned()
            .or_else(|| {
                self.ir
                    .ptr_pointees
                    .get(&(func.to_string(), name.to_string()))
                    .cloned()
            })
    }

    pub(super) fn concrete_vector_workgroup_raw_param_pointee(
        &self,
        func: &str,
        index: usize,
        name: &str,
    ) -> Option<LlType> {
        let key = (func.to_string(), name.to_string());
        if !self.ir.raw_buffer_params.contains(&key)
            || self.ir.param_connected_raw_params.contains(&key)
        {
            return None;
        }
        self.function_param_concrete_pointee(func, index, name)
            .filter(vector_backed_workgroup_raw_pointee)
    }

    pub(super) fn param_type_id(
        &mut self,
        func: &str,
        index: usize,
        name: &str,
        ty: &LlType,
    ) -> Result<Word, String> {
        if let LlType::Ptr(addrspace) = ty {
            if self.bda_device_pointers
                && *addrspace == 0
                && !self.ir.entry_functions.contains(func)
                && self
                    .ir
                    .metadata_data_buffer_params
                    .contains(&(func.to_string(), name.to_string()))
            {
                return self.type_id(&LlType::Int(64));
            }
            if self
                .ir
                .raw_buffer_params
                .contains(&(func.to_string(), name.to_string()))
            {
                if self.bda_device_pointers
                    && *addrspace == 1
                    && !self.ir.entry_functions.contains(func)
                {
                    return self.type_id(&LlType::Int(64));
                }
                if *addrspace == 3 {
                    if let Some(pointee) =
                        self.concrete_vector_workgroup_raw_param_pointee(func, index, name)
                    {
                        return self.ptr_type_id(StorageClass::Workgroup, &pointee);
                    }
                    return self.ptr_type_id(StorageClass::Workgroup, &raw_workgroup_array_type());
                }
                return self.ptr_type_id(StorageClass::UniformConstant, &raw_buffer_block_type());
            }
            let storage = if *addrspace == 4
                && (self.ir.imageblock_dimensions.is_some() || self.ir.imageblock_shared_cells)
            {
                StorageClass::Workgroup
            } else {
                llvm_pointer_storage(*addrspace)?
            };
            if let Some(pointee) = self
                .function_param_pointees
                .get(&(func.to_string(), index))
                .cloned()
            {
                return self.ptr_type_id(storage, &pointee);
            }
            if let Some(pointee) = self
                .ir
                .ptr_pointees
                .get(&(func.to_string(), name.to_string()))
                .cloned()
            {
                return self.ptr_type_id(storage, &pointee);
            }
        }
        self.type_id(ty)
    }

    pub(super) fn emit_declaration(&mut self, decl: &LlDeclaration) -> Result<(), String> {
        let ret_ty = self.resolve_type(&decl.ret)?;
        let ret_id = self.type_id(&ret_ty)?;
        let param_types: Vec<Word> = decl
            .params
            .iter()
            .map(|ty| self.type_id(ty))
            .collect::<Result<Vec<_>, _>>()?;
        let fn_ty = self.function_type_id(ret_id, &param_types);

        let func_id = *self
            .function_ids
            .get(&decl.name)
            .ok_or_else(|| format!("native emitter: missing declaration id for {}", decl.name))?;
        self.module.debug_names.push(Self::inst(
            Op::Name,
            None,
            None,
            vec![
                Operand::IdRef(func_id),
                Operand::LiteralString(decl.name.clone()),
            ],
        ));
        let mut params = Vec::with_capacity(param_types.len());
        for type_id in &param_types {
            let id = self.fresh();
            params.push(Self::inst(
                Op::FunctionParameter,
                Some(*type_id),
                Some(id),
                vec![],
            ));
        }
        self.module.functions.push(Function {
            def: Some(Self::inst(
                Op::Function,
                Some(ret_id),
                Some(func_id),
                vec![
                    Operand::FunctionControl(FunctionControl::NONE),
                    Operand::IdRef(fn_ty),
                ],
            )),
            end: Some(Self::inst(Op::FunctionEnd, None, None, vec![])),
            parameters: params,
            blocks: vec![],
        });
        Ok(())
    }

    fn function_type_id(&mut self, ret_id: Word, param_types: &[Word]) -> Word {
        let mut key = Vec::with_capacity(param_types.len() + 1);
        key.push(ret_id);
        key.extend_from_slice(param_types);
        if let Some(id) = self.interner.function_types.get(&key) {
            return *id;
        }
        let id = self.fresh();
        let mut operands = vec![Operand::IdRef(ret_id)];
        operands.extend(param_types.iter().map(|id| Operand::IdRef(*id)));
        self.module.types_global_values.push(Self::inst(
            Op::TypeFunction,
            None,
            Some(id),
            operands,
        ));
        self.interner.function_types.insert(key, id);
        id
    }
}

/// Complete local pointer-table lowering by omitting dead source-layout projections rooted at the
/// exact Function/Private tables whose pointer fields were reconstructed above. These access paths
/// are pure and have no SPIR-V consumer after the table load/store transaction becomes a typed
/// resource selection.
fn retire_dead_local_pointer_table_projections(
    blocks: &mut [Block],
    roots: &HashSet<Word>,
    protected: &HashSet<Word>,
    debug_names: &mut Vec<Instruction>,
    annotations: &mut Vec<Instruction>,
) {
    if roots.is_empty() {
        return;
    }
    let is_projection = |instruction: &Instruction| {
        matches!(
            instruction.class.opcode,
            Op::AccessChain
                | Op::InBoundsAccessChain
                | Op::PtrAccessChain
                | Op::InBoundsPtrAccessChain
                | Op::Bitcast
                | Op::CopyObject
        )
    };
    let mut candidates = roots.clone();
    loop {
        let additions = blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .filter(|instruction| is_projection(instruction))
            .filter(|instruction| {
                instruction
                    .operands
                    .iter()
                    .any(|operand| matches!(operand, Operand::IdRef(id) if candidates.contains(id)))
            })
            .filter_map(|instruction| instruction.result_id)
            .filter(|result| !candidates.contains(result))
            .collect::<Vec<_>>();
        if additions.is_empty() {
            break;
        }
        candidates.extend(additions);
    }

    let mut removed = HashSet::new();
    loop {
        let mut used = blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .flat_map(|instruction| &instruction.operands)
            .filter_map(|operand| match operand {
                Operand::IdRef(id) => Some(*id),
                _ => None,
            })
            .collect::<HashSet<_>>();
        used.extend(protected.iter().copied());
        let mut changed = false;
        for block in blocks.iter_mut() {
            block.instructions.retain(|instruction| {
                let discard = is_projection(instruction)
                    && instruction.result_id.is_some_and(|result| {
                        candidates.contains(&result) && !used.contains(&result)
                    });
                if discard {
                    if let Some(result) = instruction.result_id {
                        removed.insert(result);
                    }
                    changed = true;
                }
                !discard
            });
        }
        if !changed {
            break;
        }
    }
    let references_removed = |instruction: &Instruction| {
        instruction
            .operands
            .iter()
            .any(|operand| matches!(operand, Operand::IdRef(id) if removed.contains(id)))
    };
    debug_names.retain(|instruction| !references_removed(instruction));
    annotations.retain(|instruction| !references_removed(instruction));
}

fn flat_scalar_reinterpret_view(declared: &LlType, source: &LlType) -> Option<LlType> {
    let (declared_leaf, declared_count) = flat_scalar_leaf_count(declared)?;
    let (source_leaf, source_count) = flat_scalar_leaf_count(source)?;
    if source_count == 0
        || !types_compatible(&declared_leaf, &source_leaf)
        || declared_count % source_count != 0
    {
        return None;
    }
    let len = declared_count / source_count;
    if len == 1 {
        Some(source.clone())
    } else {
        Some(LlType::Array(Box::new(source.clone()), len))
    }
}

fn apply_inline_parameter_substitutions(
    blocks: &mut [Block],
    substitutions: &HashMap<Word, Word>,
) -> Result<(), String> {
    if substitutions.is_empty() {
        return Ok(());
    }
    for instruction in blocks
        .iter_mut()
        .flat_map(|block| block.instructions.iter_mut())
    {
        for operand in &mut instruction.operands {
            if let Operand::IdRef(id) = operand {
                *id = resolve_inline_substitution(*id, substitutions)?;
            }
        }
    }
    Ok(())
}

fn resolve_inline_substitution(
    id: Word,
    substitutions: &HashMap<Word, Word>,
) -> Result<Word, String> {
    let mut replacement = id;
    let mut visited = HashSet::new();
    while let Some(next) = substitutions.get(&replacement) {
        if !visited.insert(replacement) {
            return Err("native emitter: inline parameter substitution cycle".to_string());
        }
        replacement = *next;
    }
    Ok(replacement)
}

/// A block TERMINATOR is one of `br` / `switch` / `ret` / `unreachable`. `TirBlock` carries its
/// terminator separately from `insts`, so the graph walk emits the straight-line `insts` from the typed
/// graph and then the terminator entirely from typed state. The keywords are reserved LLVM terminators,
/// so no value-defining (`%r = ...`) line matches — there is exactly one terminator per block.
fn reorder_forward_local_def_blocks(
    body_blocks: &mut Vec<BodyBlock>,
    defuse: &ReorderDefUse,
) -> Result<(), String> {
    if body_blocks.len() <= 2 {
        return Ok(());
    }

    let mut seen_orders = HashSet::new();
    let max_moves = body_blocks.len() * body_blocks.len();
    let mut moves = 0;
    loop {
        let order = body_blocks
            .iter()
            .map(|block| block.name.clone())
            .collect::<Vec<_>>();
        if !seen_orders.insert(order) {
            return Err(format!(
                "native emitter: cyclic forward local block dependencies while reordering blocks after {moves} moves"
            ));
        }

        // Index local uses by current block index once per move. We still scan definitions in current
        // block/instruction order below, so the selected move is byte-for-byte equivalent to the old
        // nested scan: earliest use block wins, and within that block the earliest forward definition
        // in current order wins.
        let mut use_indices_by_name: HashMap<&str, Vec<usize>> = HashMap::new();
        for (idx, block) in body_blocks.iter().enumerate().skip(1) {
            if let Some(uses) = defuse.uses_by_block.get(&block.name) {
                for name in uses {
                    use_indices_by_name
                        .entry(name.as_str())
                        .or_default()
                        .push(idx);
                }
            }
        }
        let mut first_forward_def_by_use_idx: Vec<Option<usize>> = vec![None; body_blocks.len()];
        for (def_idx, block) in body_blocks.iter().enumerate() {
            let Some(names) = defuse.defs_by_block.get(&block.name) else {
                continue;
            };
            for name in names {
                let Some(use_indices) = use_indices_by_name.get(name.as_str()) else {
                    continue;
                };
                for &use_idx in use_indices {
                    if def_idx > use_idx && first_forward_def_by_use_idx[use_idx].is_none() {
                        first_forward_def_by_use_idx[use_idx] = Some(def_idx);
                    }
                }
            }
        }
        let mut moved = false;
        if let Some((idx, def_idx)) = first_forward_def_by_use_idx
            .iter()
            .enumerate()
            .skip(1)
            .filter_map(|(idx, def_idx)| def_idx.map(|def_idx| (idx, def_idx)))
            .next()
        {
            let block = body_blocks.remove(def_idx);
            body_blocks.insert(idx, block);
            moves += 1;
            if moves > max_moves {
                return Err(format!(
                    "native emitter: forward local block reorder budget exceeded after {moves} moves"
                ));
            }
            moved = true;
        }
        if !moved {
            break;
        }
    }
    Ok(())
}

/// The forward-reorder def/use facts, sourced ONCE from the typed per-block graph
/// (`tir::build_from_blocks`) instead of re-lexing `BodyBlock.lines` on every reorder iteration. Keyed
/// by block NAME (labels are unique and reorder never renames/creates blocks), so the maps stay valid as
/// reorder permutes the block Vec. Reproduces the retired text scan exactly:
/// - `defs_by_block`: every result-defining instruction's `%name`, in instruction order, EXCLUDING a
///   scalar-pointer `phi` (`phi ptr` / `phi ptr addrspace(...)`) — those are not reorder candidates.
/// - `uses_by_block`: the local `%value` uses that can bind a forward def — every NON-`phi`
///   instruction's value operands (a `phi` contributes ZERO uses, matching the text scan) plus the
///   terminator's condition / selector / return value.
struct ReorderDefUse {
    defs_by_block: HashMap<String, Vec<String>>,
    uses_by_block: HashMap<String, HashSet<String>>,
}

impl ReorderDefUse {
    fn from_blocks(body_blocks: &[BodyBlock]) -> Result<Self, String> {
        let tir = crate::native::tir::build_from_blocks(body_blocks)?;
        let mut defs_by_block = HashMap::new();
        let mut uses_by_block = HashMap::new();
        for block in &tir.blocks {
            let mut defs = Vec::new();
            let mut uses = HashSet::new();
            for inst in &block.insts {
                if let Some(result) = &inst.result {
                    // The one non-reorderable defining form: a scalar-pointer phi (the text scan
                    // skipped `phi ptr` / `phi ptr addrspace(`).
                    let scalar_ptr_phi =
                        inst.opcode == "phi" && matches!(inst.result_ty, Some(LlType::Ptr(_)));
                    if !scalar_ptr_phi {
                        defs.push(result.clone());
                    }
                }
                // A `phi` contributes no forward-binding uses (its incoming values arrive along
                // predecessor edges, not within this block); the text scan returned an empty set for it.
                if inst.opcode != "phi" {
                    inst.visit_uses(|name| {
                        uses.insert(name.to_string());
                    });
                }
            }
            terminator_local_uses(&block.terminator, &mut uses);
            defs_by_block.insert(block.label.clone(), defs);
            uses_by_block.insert(block.label.clone(), uses);
        }
        Ok(Self {
            defs_by_block,
            uses_by_block,
        })
    }
}

/// The local `%value` uses a terminator reads: a conditional branch's condition, a switch's selector, or
/// a return value. Unconditional `br label`, `ret void`, and `unreachable` read no value.
fn terminator_local_uses(term: &crate::native::tir::TirTerminator, uses: &mut HashSet<String>) {
    use crate::native::tir::TirTerminator;
    let operand = match term {
        TirTerminator::BrCond { cond, .. } => Some(cond.as_str()),
        TirTerminator::Switch { selector, .. } => Some(selector.as_str()),
        TirTerminator::Ret(Some(value)) => Some(value.as_str()),
        TirTerminator::Br(_) | TirTerminator::Ret(None) | TirTerminator::Unreachable => None,
    };
    if let Some(operand) = operand {
        let mut names = Vec::new();
        crate::native::tir::collect_value_names(operand, &mut names);
        uses.extend(names);
    }
}

fn vector_backed_workgroup_raw_pointee(pointee: &LlType) -> bool {
    match pointee {
        LlType::Vector(_, _) => true,
        LlType::Array(elem, _) => matches!(elem.as_ref(), LlType::Vector(_, _)),
        _ => false,
    }
}

fn emitted_block_successors(block: &Block) -> Vec<Word> {
    let Some(terminator) = block.instructions.last() else {
        return Vec::new();
    };
    match terminator.class.opcode {
        Op::Branch => terminator
            .operands
            .first()
            .and_then(|operand| match operand {
                Operand::IdRef(target) => Some(vec![*target]),
                _ => None,
            })
            .unwrap_or_default(),
        Op::BranchConditional => terminator
            .operands
            .iter()
            .skip(1)
            .filter_map(|operand| match operand {
                Operand::IdRef(target) => Some(*target),
                _ => None,
            })
            .collect(),
        Op::Switch => terminator
            .operands
            .iter()
            .enumerate()
            .filter_map(|(index, operand)| {
                (index == 1 || index >= 3 && index % 2 == 1)
                    .then_some(match operand {
                        Operand::IdRef(target) => Some(*target),
                        _ => None,
                    })
                    .flatten()
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn redirect_emitted_terminator_target(block: &mut Block, from: Word, to: Word) {
    let Some(terminator) = block.instructions.last_mut() else {
        return;
    };
    let opcode = terminator.class.opcode;
    let operand_count = terminator.operands.len();
    let mut rewrite = |index: usize| {
        if terminator.operands.get(index) == Some(&Operand::IdRef(from)) {
            terminator.operands[index] = Operand::IdRef(to);
        }
    };
    match opcode {
        Op::Branch => rewrite(0),
        Op::BranchConditional => {
            rewrite(1);
            rewrite(2);
        }
        Op::Switch => {
            rewrite(1);
            for index in (3..operand_count).step_by(2) {
                rewrite(index);
            }
        }
        _ => {}
    }
}

/// A structured plan is admissible only when every local value use still has a function parameter
/// or instruction definition. Region cloning renames SSA together with blocks; checking that closed
/// contract before emission prevents a partially renamed plan from becoming a late unknown-value
/// failure and lets the existing ownership constructor select another complete CFG representation.
fn typed_ssa_is_closed(
    blocks: &[crate::native::cfg::BodyBlock],
    params: &[(String, LlType)],
) -> bool {
    let mut defined = params
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<HashSet<_>>();
    let mut definition_sites = HashMap::<String, (usize, usize)>::new();
    for (block_index, block) in blocks.iter().enumerate() {
        let Some(carrier) = block.typed.as_ref() else {
            return false;
        };
        for (instruction_index, instruction) in carrier.insts.iter().enumerate() {
            if let Some(result) = &instruction.result {
                defined.insert(result.clone());
                definition_sites.insert(result.clone(), (block_index, instruction_index));
            }
        }
    }
    let block_indices = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut predecessors = vec![Vec::new(); blocks.len()];
    for (index, block) in blocks.iter().enumerate() {
        let carrier = block.typed.as_ref().expect("typed blocks checked above");
        for successor in carrier.terminator.successors() {
            if let Some(target) = block_indices.get(successor) {
                predecessors[*target].push(index);
            }
        }
    }
    let all_blocks = (0..blocks.len()).collect::<HashSet<_>>();
    let mut dominators = vec![all_blocks; blocks.len()];
    if !dominators.is_empty() {
        dominators[0] = HashSet::from([0]);
    }
    loop {
        let mut changed = false;
        for block in 1..blocks.len() {
            if predecessors[block].is_empty() {
                continue;
            }
            let mut next = dominators[predecessors[block][0]].clone();
            for predecessor in predecessors[block].iter().skip(1) {
                next.retain(|candidate| dominators[*predecessor].contains(candidate));
            }
            next.insert(block);
            if next != dominators[block] {
                dominators[block] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let dominates_use = |name: &str, block: usize, instruction: usize| {
        if params.iter().any(|(param, _)| param == name) {
            return true;
        }
        definition_sites
            .get(name)
            .is_some_and(|&(owner, position)| {
                if owner == block {
                    position < instruction
                } else {
                    dominators[block].contains(&owner)
                }
            })
    };
    blocks.iter().enumerate().all(|(block_index, block)| {
        let carrier = block.typed.as_ref().expect("typed blocks checked above");
        let instructions_closed =
            carrier
                .insts
                .iter()
                .enumerate()
                .all(|(instruction_index, instruction)| {
                    if let Some((_, incoming)) = instruction.phi_incoming() {
                        return incoming.iter().all(|(value, predecessor)| {
                            let LlValue::Local(name) = value else {
                                return true;
                            };
                            let Some(&predecessor) = block_indices.get(predecessor.as_str()) else {
                                return false;
                            };
                            definition_sites.get(name).is_some_and(|&(owner, _)| {
                                owner == predecessor || dominators[predecessor].contains(&owner)
                            }) || params.iter().any(|(param, _)| param == name)
                        });
                    }
                    let mut closed = true;
                    instruction.visit_uses(|name| {
                        closed &= defined.contains(name)
                            && dominates_use(name, block_index, instruction_index)
                    });
                    closed
                });
        let terminator_position = carrier.insts.len();
        let terminator_closed = match &carrier.terminator {
            crate::native::tir::TirTerminator::Br(_)
            | crate::native::tir::TirTerminator::Ret(None)
            | crate::native::tir::TirTerminator::Unreachable => true,
            crate::native::tir::TirTerminator::BrCond { cond, .. } => {
                defined.contains(cond) && dominates_use(cond, block_index, terminator_position)
            }
            crate::native::tir::TirTerminator::Switch { selector, .. } => {
                defined.contains(selector)
                    && dominates_use(selector, block_index, terminator_position)
            }
            crate::native::tir::TirTerminator::Ret(Some(value)) => {
                defined.contains(value) && dominates_use(value, block_index, terminator_position)
            }
        };
        instructions_closed && terminator_closed
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(name: &str, lines: &[&str]) -> BodyBlock {
        let lines: Vec<String> = lines.iter().map(|line| line.to_string()).collect();
        let typed = crate::native::tir::lower_block_carrier(name, &lines, &HashMap::new());
        BodyBlock {
            name: name.to_string(),
            role: crate::native::cfg::BlockRole::Normal,
            typed: typed.map(Into::into),
        }
    }

    fn reorder(blocks: &mut Vec<BodyBlock>) -> Result<(), String> {
        let defuse = ReorderDefUse::from_blocks(blocks)?;
        reorder_forward_local_def_blocks(blocks, &defuse)
    }

    #[test]
    fn bda_forward_addresses_cross_grounded_pointer_state_cycles() {
        let lines = [
            "%slot = phi ptr addrspace(1) [ undef, %pre ], [ %next, %continue ]",
            "%next = phi ptr addrspace(1) [ %slot, %case0 ], [ %root, %case1 ]",
            "%child = load ptr addrspace(1), ptr addrspace(1) %slot, align 8",
            "%orphan = phi ptr addrspace(1) [ %orphan.next, %a ]",
            "%orphan.next = phi ptr addrspace(1) [ %orphan, %b ]",
            "ret void",
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
        let tir = crate::native::tir::build(&lines, "%entry", &HashMap::new()).expect("tir");
        let addresses = bda_forward_address_values(
            &tir,
            &HashSet::from(["%root".to_string()]),
            &HashSet::new(),
        );

        for expected in ["%root", "%slot", "%next", "%child"] {
            assert!(
                addresses.contains(expected),
                "missing {expected}: {addresses:?}"
            );
        }
        assert!(!addresses.contains("%orphan"), "{addresses:?}");
        assert!(!addresses.contains("%orphan.next"), "{addresses:?}");
    }

    #[test]
    fn structured_plan_ssa_closure_rejects_renamed_use_without_definition() {
        let complete = vec![block(
            "%entry",
            &[
                "%source = add i32 %arg, 1",
                "%use = add i32 %source, 2",
                "ret void",
            ],
        )];
        assert!(typed_ssa_is_closed(
            &complete,
            &[("%arg".to_string(), LlType::Int(32))]
        ));

        let incomplete = vec![block(
            "%entry",
            &["%use = add i32 %renamed_missing, 2", "ret void"],
        )];
        assert!(!typed_ssa_is_closed(&incomplete, &[]));
    }

    #[test]
    fn forward_local_reorder_moves_def_before_later_use() {
        let mut blocks = vec![
            block("%entry", &["br label %use"]),
            block("%use", &["%use.value = fadd float %later, 1.0", "ret void"]),
            block("%def", &["%later = fadd float 1.0, 2.0", "ret void"]),
        ];

        reorder(&mut blocks).unwrap();

        let names = blocks
            .iter()
            .map(|block| block.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["%entry", "%def", "%use"]);
    }

    #[test]
    fn forward_local_reorder_reports_cycles_instead_of_looping() {
        let mut blocks = vec![
            block("%entry", &["br label %a"]),
            block("%a", &["%a.value = fadd float %b.value, 1.0", "ret void"]),
            block("%b", &["%b.value = fadd float %a.value, 1.0", "ret void"]),
        ];

        let error = reorder(&mut blocks).unwrap_err();

        assert!(error.contains("cyclic forward local block dependencies"));
    }

    #[test]
    fn reorder_defuse_ignores_definition_lhs_and_phi_incoming_values() {
        let blocks = vec![block(
            "%body",
            &[
                "%lhs = fadd float %rhs, 1.0",
                "%phi = phi i32 [ %from.a, %a ], [ %from.b, %b ]",
                "br i1 %cond, label %then, label %else",
            ],
        )];

        let defuse = ReorderDefUse::from_blocks(&blocks).unwrap();
        let uses = &defuse.uses_by_block["%body"];

        assert!(uses.contains("%rhs"));
        assert!(uses.contains("%cond"));
        assert!(!uses.contains("%lhs"));
        assert!(!uses.contains("%phi"));
        assert!(!uses.contains("%from.a"));
        assert!(!uses.contains("%then"));
    }

    #[test]
    fn deferred_inline_parameter_substitution_rewrites_uses_not_definitions() {
        let mut blocks = vec![Block {
            label: Some(Emitter::inst(Op::Label, None, Some(1), vec![])),
            instructions: vec![
                Emitter::inst(Op::CopyObject, Some(2), Some(20), vec![Operand::IdRef(10)]),
                Emitter::inst(
                    Op::IAdd,
                    Some(2),
                    Some(21),
                    vec![Operand::IdRef(10), Operand::IdRef(20)],
                ),
            ],
        }];

        apply_inline_parameter_substitutions(&mut blocks, &HashMap::from([(10, 8), (8, 7)]))
            .unwrap();

        assert_eq!(blocks[0].instructions[0].result_id, Some(20));
        assert_eq!(blocks[0].instructions[0].operands, vec![Operand::IdRef(7)]);
        assert_eq!(
            blocks[0].instructions[1].operands,
            vec![Operand::IdRef(7), Operand::IdRef(20)]
        );
    }

    #[test]
    fn local_pointer_table_transaction_omits_only_dead_rooted_projections() {
        let mut blocks = vec![Block {
            label: Some(Emitter::inst(Op::Label, None, Some(1), vec![])),
            instructions: vec![
                Emitter::inst(
                    Op::Variable,
                    Some(2),
                    Some(10),
                    vec![Operand::StorageClass(StorageClass::Function)],
                ),
                Emitter::inst(
                    Op::InBoundsAccessChain,
                    Some(3),
                    Some(20),
                    vec![Operand::IdRef(10), Operand::IdRef(4)],
                ),
                Emitter::inst(
                    Op::InBoundsAccessChain,
                    Some(3),
                    Some(21),
                    vec![Operand::IdRef(20), Operand::IdRef(4)],
                ),
                Emitter::inst(
                    Op::AccessChain,
                    Some(3),
                    Some(22),
                    vec![Operand::IdRef(10), Operand::IdRef(4)],
                ),
                Emitter::inst(Op::Load, Some(5), Some(23), vec![Operand::IdRef(22)]),
            ],
        }];
        let mut debug_names = vec![Emitter::inst(
            Op::Name,
            None,
            None,
            vec![Operand::IdRef(20), Operand::LiteralString("dead".into())],
        )];
        let mut annotations = vec![Emitter::inst(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(21),
                Operand::Decoration(spirv::Decoration::Alignment),
                Operand::LiteralBit32(4),
            ],
        )];

        retire_dead_local_pointer_table_projections(
            &mut blocks,
            &HashSet::from([10]),
            &HashSet::new(),
            &mut debug_names,
            &mut annotations,
        );

        let results = blocks[0]
            .instructions
            .iter()
            .filter_map(|instruction| instruction.result_id)
            .collect::<HashSet<_>>();
        assert!(!results.contains(&20));
        assert!(!results.contains(&21));
        assert!(results.contains(&22));
        assert!(results.contains(&23));
        assert!(debug_names.is_empty());
        assert!(annotations.is_empty());
    }
}
