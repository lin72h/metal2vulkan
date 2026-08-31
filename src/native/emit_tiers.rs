//! Emitter entry points for the primary representation and structurally selected alternatives
//! (inline+SROA, raw buffers, address-domain lowering, and relooper feed). Each composes pre-parse
//! AIR lowering, typed parsing, and [`Emitter`] with a different buffer or CFG model. `lib.rs`
//! selects among them from emitter errors and owned structural facts before validation.

use super::ir::{LlType, LlValue, TypedValue};
use super::*;
use crate::meta;

pub fn emit_vulkan_spirv(san_ll: &str) -> Result<Vec<u8>, String> {
    let kern = meta::parse_air_kernel_meta(san_ll);
    let entry_name = meta::entry_name(san_ll, "kernel");
    Ok(emit_vulkan_spirv_with_sidecar(
        san_ll,
        kern.as_ref(),
        entry_name.as_deref(),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
    )?
    .into_bytes())
}

/// Test adapter for CFG synthesis: parse the function shell so parameters/module state follow the
/// ordinary path, replace its body with the finalized typed carriers, then run the unchanged emitter.
/// This avoids serializing a carrier through the deliberately partial debug-text renderer and proves
/// synthetic blocks are consumable by the real graph-driven emission substrate.
#[cfg(test)]
pub(in crate::native) fn emit_vulkan_spirv_from_typed_blocks(
    function_shell: &str,
    blocks: Vec<crate::native::cfg::BodyBlock>,
) -> Result<Vec<u8>, String> {
    let mut parsed = LlModule::parse(function_shell)?;
    let [function] = parsed.functions.as_mut_slice() else {
        return Err(format!(
            "native emitter: typed-block test adapter needs exactly one function, got {}",
            parsed.functions.len()
        ));
    };
    function.blocks = blocks;
    Ok(finalize_emission(Emitter::new(parsed), None, function_shell)?.into_bytes())
}

pub(crate) fn emit_vulkan_spirv_with_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    emit_vulkan_spirv_with_outcome(san_ll, kern, entry_name, buffer_layouts)
        .map_err(|failure| failure.error)
}

pub(crate) fn emit_vulkan_spirv_with_outcome(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, crate::emit_sidecar::EmissionFailure> {
    emit_vulkan_spirv_inner(san_ll, false, kern, entry_name, buffer_layouts)
}

/// Re-emit with the narrowly scoped primitive metadata inference for a cross-buffer pointer phi.
/// This is a diagnostic emitter mode. Production uses owned type facts to select its representation.
pub fn emit_vulkan_spirv_with_primitive_phi_metadata(san_ll: &str) -> Result<Vec<u8>, String> {
    let kern = meta::parse_air_kernel_meta(san_ll);
    let entry_name = meta::entry_name(san_ll, "kernel");
    Ok(emit_vulkan_spirv_with_primitive_phi_metadata_sidecar(
        san_ll,
        kern.as_ref(),
        entry_name.as_deref(),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
    )?
    .into_bytes())
}

pub(crate) fn emit_vulkan_spirv_with_primitive_phi_metadata_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    emit_vulkan_spirv_inner(san_ll, true, kern, entry_name, buffer_layouts)
        .map_err(|failure| failure.error)
}

fn emit_vulkan_spirv_inner(
    san_ll: &str,
    primitive_phi_metadata: bool,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, crate::emit_sidecar::EmissionFailure> {
    // Lower `air.simdgroup_async_copy_2d` (+ its event/wait pair) to an explicit strided tile copy
    // before parse. Alternate constructions see this lowering because the production entry applies
    // it before re-emission. This
    // entry's copy is retained for direct callers; already-lowered text is a no-op guard. See
    // `async_copy` and its structural regression tests. Floor-safe: only fires on async-copy modules,
    // which fail the emitter outright otherwise.
    let retry_debug = crate::env_vars::retry_debug();
    if retry_debug {
        eprintln!("[retry-debug] native emit: AIR normalizations start");
    }
    let san_ll = async_copy::lower_simdgroup_async_copy(san_ll);
    // Scalarize any scalar/vector pointer-merge before parse (floor-safe: a no-op unless the module
    // carries a `<N x T>*`/`T*` merge the emitter rejects outright). See `vec_scalar_merge`.
    let san_ll = vec_scalar_merge::lower_vector_scalar_pointer_merge(&san_ll);
    // A pointer select over distinct buffers is deliberately deferred into value-domain load/store
    // replay. When that value reaches an internal call, inline only the consuming helper before the
    // first emission so no unrepresentable standalone pointer crosses a SPIR-V function boundary.
    let pointer_consumers = inline::inline_pointer_select_consumers(&san_ll, entry_name);
    let san_ll = pointer_consumers.source;
    if retry_debug {
        eprintln!("[retry-debug] native emit: AIR normalizations complete; typed parse start");
    }
    let mut parsed = if primitive_phi_metadata {
        LlModule::parse_with_primitive_phi_metadata_and_stage_meta(&san_ll, kern, entry_name)
    } else {
        LlModule::parse_with_stage_meta(&san_ll, kern, entry_name)
    }
    .map_err(|error| crate::emit_sidecar::EmissionFailure {
        error,
        ordinary_plan_rejected_functions: HashSet::new(),
        ownership_plan_rejected_functions: HashSet::new(),
    })?;
    let requires_device_addresses = requires_device_address_model(&parsed);
    if retry_debug {
        eprintln!(
            "[retry-debug] native emit: requires_device_addresses={requires_device_addresses}"
        );
    }
    // A metadata-owned aggregate cannot replace a scalar/vector pointee while preserving the source
    // GEP paths. Select the raw word model from the typed source graph before the first emission.
    // Aggregate aliases stay typed so exact source ordinals and AIR offsets can reconcile them.
    let raw_layout_params = if requires_device_addresses {
        HashSet::new()
    } else {
        parsed
            .functions
            .iter()
            .find(|function| entry_name.is_some_and(|entry| function.name == entry))
            .into_iter()
            .flat_map(|function| 0..function.params.len())
            .filter_map(|index| {
                parsed
                    .entry_param_requires_raw_layout(entry_name, index as u32)
                    .then_some(index as u32)
            })
            .collect::<HashSet<_>>()
    };
    let flat_raw_interface_params = parsed
        .functions
        .iter()
        .find(|function| entry_name.is_some_and(|entry| function.name == entry))
        .into_iter()
        .flat_map(|function| function.params.iter().enumerate())
        .filter_map(|(index, (name, _))| {
            parsed
                .call_connected_raw_params
                .contains(&(entry_name?.to_owned(), name.clone()))
                .then_some(index as u32)
        })
        .chain(raw_layout_params.iter().copied())
        .collect::<HashSet<_>>();
    if !raw_layout_params.is_empty() {
        mark_entry_buffer_params_raw(&mut parsed, entry_name, &raw_layout_params);
        parsed.propagate_raw_buffer_params();
    }
    // Physical-address construction changes pointer representation module-wide and can therefore
    // fail before a live indirect call reports its missing linked-function input. Preserve that
    // structurally prior dependency diagnostic only on modules that actually select BDA; logical
    // modules retain their ordinary function-constant pruning and retry behavior.
    if requires_device_addresses {
        if let Some(error) = parsed
            .functions
            .iter()
            .flat_map(|function| function.carrier_insts())
            .filter_map(|inst| inst.value_call_error().as_deref())
            .find(|error| error.contains("unsupported indirect call through function pointer"))
        {
            return Err(crate::emit_sidecar::EmissionFailure {
                error: error.to_string(),
                ordinary_plan_rejected_functions: HashSet::new(),
                ownership_plan_rejected_functions: HashSet::new(),
            });
        }
    }
    if requires_device_addresses {
        mark_all_device_buffers_raw(&mut parsed, false);
    }
    if retry_debug {
        eprintln!("[retry-debug] native emit: typed parse complete; emission start");
    }
    let mut emitter = Emitter::new(parsed);
    if requires_device_addresses {
        emitter = emitter.with_bda_device_pointers();
    }
    if pointer_consumers.requires_relooper {
        emitter = emitter.with_relooper_feed();
    }
    let mut emitted = finalize_emission_outcome(emitter, buffer_layouts, &san_ll)?;
    emitted
        .sidecar
        .flat_raw_buffer_params
        .extend(flat_raw_interface_params);
    Ok(emitted)
}

fn pointer_parameter_roots(function: &ir::LlFunction, value: &str) -> HashSet<usize> {
    let params = function
        .params
        .iter()
        .enumerate()
        .filter(|(_, (_, ty))| matches!(ty, LlType::Ptr(1)))
        .map(|(index, (name, _))| (name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut roots = HashSet::new();
    let mut pending = vec![value.to_string()];
    let mut seen = HashSet::new();
    while let Some(name) = pending.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        if let Some(index) = params.get(name.as_str()) {
            roots.insert(*index);
            continue;
        }
        let Some(inst) = function
            .carrier_insts()
            .find(|inst| inst.result.as_deref() == Some(name.as_str()))
        else {
            continue;
        };
        if let Some(gep) = inst.gep().as_deref() {
            if let LlValue::Local(base) = &gep.base.value {
                pending.push(base.clone());
            }
        } else if let Some((_, base)) = inst.identity_ptr_bitcast() {
            pending.push(base.to_string());
        } else if inst.opcode == "freeze"
            || inst.phi_incoming().is_some()
            || inst.select_arms().is_some()
        {
            inst.visit_uses(|source| pending.push(source.to_string()));
        }
    }
    roots
}

/// Whether the typed AIR graph uses a reconstructed device pointer as a byte address.
///
/// Pointer-shaped values also carry opaque texture and acceleration-structure handles in AIR. Their
/// type alone must not select PhysicalStorageBuffer64. Start from runtime pointer-producing loads or
/// `inttoptr`, propagate through pointer-preserving SSA nodes, and require the address model only at a
/// memory/address consumer (or a stable ABI operation whose contract consumes a device address).
pub(super) fn requires_device_address_model(parsed: &LlModule) -> bool {
    fn local_is(value: &LlValue, values: &HashSet<String>) -> bool {
        matches!(value, LlValue::Local(name) if values.contains(name))
    }

    fn is_constant_index(value: &LlValue) -> bool {
        matches!(
            value,
            LlValue::Zero | LlValue::Int(_) | LlValue::SignedInt(_) | LlValue::Hex(_)
        )
    }

    fn gep_reaches_float(source: &LlType, indices: &[TypedValue]) -> bool {
        let mut current = source.clone();
        for index in indices.iter().skip(1) {
            current = match current {
                LlType::Array(element, _) | LlType::Vector(element, _) => *element,
                LlType::Struct(fields) => {
                    let member = match index.value {
                        LlValue::Int(value) | LlValue::Hex(value) => usize::try_from(value).ok(),
                        LlValue::SignedInt(value) => usize::try_from(value).ok(),
                        LlValue::Zero => Some(0),
                        _ => None,
                    };
                    let Some(field) = member.and_then(|member| fields.get(member)).cloned() else {
                        return false;
                    };
                    field
                }
                _ => return false,
            };
        }
        current == LlType::Float
    }

    let functions_by_name = parsed
        .functions
        .iter()
        .map(|function| (function.name.as_str(), function))
        .collect::<HashMap<_, _>>();
    let has_device_buffer_array =
        parsed
            .metadata_data_buffer_params
            .iter()
            .any(|(function_name, parameter_name)| {
                parsed.functions.iter().any(|function| {
                    function.name == *function_name
                        && function.params.iter().any(|(name, ty)| {
                            name == parameter_name && matches!(ty, LlType::Ptr(0))
                        })
                })
            });

    // An integer atomic over a float-typed device slot is a bit-pattern operation. Logical SPIR-V
    // cannot form the required integer pointer view (`OpBitcast` may not consume that logical
    // pointer), while the physical-address model can express the same address with either pointee.
    // Trace through opaque-pointer identity aliases to the typed GEP/load fact before choosing the
    // representation; local/workgroup atomics are deliberately excluded.
    for function in &parsed.functions {
        let aliases = function
            .carrier_insts()
            .filter_map(|inst| inst.identity_ptr_bitcast())
            .map(|(result, base)| (result.to_string(), base.to_string()))
            .collect::<HashMap<_, _>>();
        for inst in function.carrier_insts() {
            let Some(call) = inst.call().as_deref() else {
                continue;
            };
            if !call.callee.starts_with("air.atomic.global") || !call.callee.ends_with(".i32") {
                continue;
            }
            let Some(LlValue::Local(argument)) = call.args.first().map(|arg| &arg.value) else {
                continue;
            };
            let mut root = argument;
            let mut seen = HashSet::new();
            while seen.insert(root.clone()) {
                let Some(base) = aliases.get(root) else {
                    break;
                };
                root = base;
            }
            let inferred_float = parsed
                .ptr_pointees
                .get(&(function.name.clone(), root.clone()))
                .or_else(|| {
                    parsed
                        .metadata_primitive_buffer_pointees
                        .get(&(function.name.clone(), root.clone()))
                })
                .is_some_and(|pointee| matches!(pointee, LlType::Float));
            let gep_float = function.carrier_insts().any(|inst| {
                inst.result.as_ref() == Some(root)
                    && inst
                        .gep()
                        .as_deref()
                        .is_some_and(|gep| gep_reaches_float(&gep.source_ty, &gep.indices))
            });
            if inferred_float || gep_float {
                return true;
            }
        }
    }

    let mut address_consuming_params = HashSet::<(String, usize)>::new();
    for function in &parsed.functions {
        for (index, (param, ty)) in function.params.iter().enumerate() {
            if !matches!(ty, LlType::Ptr(1)) {
                continue;
            }
            let consumed = function.carrier_insts().any(|inst| {
                inst.gep().as_deref().is_some_and(
                    |gep| matches!(&gep.base.value, LlValue::Local(name) if name == param),
                ) || inst.load().as_deref().is_some_and(
                    |load| matches!(&load.ptr.value, LlValue::Local(name) if name == param),
                ) || inst.store().as_deref().is_some_and(
                    |(_, pointer)| matches!(&pointer.value, LlValue::Local(name) if name == param),
                ) || inst.call().as_deref().is_some_and(|call| {
                    call.callee.starts_with("air.atomic.global")
                        && call
                            .args
                            .iter()
                            .any(|arg| matches!(&arg.value, LlValue::Local(name) if name == param))
                })
            });
            if consumed {
                address_consuming_params.insert((function.name.clone(), index));
            }
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for function in &parsed.functions {
            let params = function
                .params
                .iter()
                .enumerate()
                .map(|(index, (name, _))| (name.as_str(), index))
                .collect::<HashMap<_, _>>();
            for inst in function.carrier_insts() {
                let Some(call) = inst.call().as_deref() else {
                    continue;
                };
                if !functions_by_name.contains_key(call.callee.as_str()) {
                    continue;
                }
                for (callee_index, arg) in call.args.iter().enumerate() {
                    if !address_consuming_params.contains(&(call.callee.clone(), callee_index)) {
                        continue;
                    }
                    let LlValue::Local(argument) = &arg.value else {
                        continue;
                    };
                    let Some(caller_index) = params.get(argument.as_str()).copied() else {
                        continue;
                    };
                    if address_consuming_params.insert((function.name.clone(), caller_index)) {
                        changed = true;
                    }
                }
            }
        }
    }

    // Track the narrower helper contract that specifically reaches a global atomic. Dynamic raw
    // cursors only require a physical representation when they cross such a boundary; ordinary
    // helper loads and stores are already representable by the logical raw-buffer lowering.
    let mut atomic_consuming_params = HashSet::<(String, usize)>::new();
    for function in &parsed.functions {
        for inst in function.carrier_insts() {
            let Some(call) = inst.call().as_deref() else {
                continue;
            };
            if !call.callee.starts_with("air.atomic.global") {
                continue;
            }
            for arg in &call.args {
                let LlValue::Local(argument) = &arg.value else {
                    continue;
                };
                atomic_consuming_params.extend(
                    pointer_parameter_roots(function, argument)
                        .into_iter()
                        .map(|index| (function.name.clone(), index)),
                );
            }
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for function in &parsed.functions {
            for inst in function.carrier_insts() {
                let Some(call) = inst.call().as_deref() else {
                    continue;
                };
                if !functions_by_name.contains_key(call.callee.as_str()) {
                    continue;
                }
                for (callee_index, arg) in call.args.iter().enumerate() {
                    if !atomic_consuming_params.contains(&(call.callee.clone(), callee_index)) {
                        continue;
                    }
                    let LlValue::Local(argument) = &arg.value else {
                        continue;
                    };
                    for caller_index in pointer_parameter_roots(function, argument) {
                        if atomic_consuming_params.insert((function.name.clone(), caller_index)) {
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    // The integer atomic may live in a helper whose opaque pointer parameter has already acquired
    // the atomic's i32 use type. Trace that helper contract back to entry parameters before choosing
    // an address model: if an atomic-consuming parameter is structurally rooted in a float buffer,
    // its integer access is a same-address bit-pattern view and requires physical addressing just
    // like the direct in-function case above.
    if atomic_consuming_params
        .iter()
        .any(|(function_name, index)| {
            functions_by_name
                .get(function_name.as_str())
                .and_then(|function| function.params.get(*index))
                .is_some_and(|(name, ty)| {
                    matches!(ty, LlType::Ptr(1))
                        && parsed
                            .metadata_primitive_buffer_pointees
                            .get(&(function_name.clone(), name.clone()))
                            .is_some_and(|pointee| matches!(pointee, LlType::Float))
                })
        })
    {
        return true;
    }

    for function in &parsed.functions {
        // Visible callbacks receive device-buffer-array state through an opaque generic pointer.
        // A load from that state is nevertheless a device address when the loaded value is used as
        // the base of a GEP whose declared source element is a device pointer. Preserve that typed
        // GEP fact instead of treating the opaque `ptr` spelling as a resource handle.
        let generic_device_address_bases = function
            .carrier_insts()
            .filter_map(|inst| inst.gep().as_deref())
            .filter(|gep| matches!(gep.source_ty, LlType::Ptr(1)))
            .filter_map(|gep| match &gep.base.value {
                LlValue::Local(name) => Some(name.clone()),
                _ => None,
            })
            .collect::<HashSet<_>>();
        // Pointer payloads loaded from a constant argument buffer remain byte values unless they
        // cross a helper/atomic ABI boundary that consumes them as device addresses. Keep that
        // distinction explicit: direct null/payload comparisons retain their logical raw lowering,
        // while helper-connected memory pointers select BDA before either function is emitted.
        let mut argument_buffer_addresses = function
            .carrier_insts()
            .filter(|inst| matches!(inst.result_ty.as_ref(), Some(LlType::Ptr(1))))
            .filter(|inst| {
                inst.load()
                    .as_deref()
                    .is_some_and(|load| matches!(load.ptr.ty, LlType::Ptr(2)))
            })
            .filter_map(|inst| inst.result.clone())
            .collect::<HashSet<_>>();
        let mut changed = true;
        while changed {
            changed = false;
            for inst in function.carrier_insts() {
                let Some(result) = inst.result.as_ref() else {
                    continue;
                };
                if argument_buffer_addresses.contains(result)
                    || !matches!(inst.result_ty.as_ref(), Some(LlType::Ptr(1)))
                {
                    continue;
                }
                if matches!(
                    inst.opcode.as_str(),
                    "bitcast" | "freeze" | "phi" | "select"
                ) {
                    let mut source_is_address = false;
                    inst.visit_uses(|name| {
                        source_is_address |= argument_buffer_addresses.contains(name)
                    });
                    if source_is_address {
                        argument_buffer_addresses.insert(result.clone());
                        changed = true;
                    }
                }
            }
        }
        if function.carrier_insts().any(|inst| {
            inst.gep()
                .as_deref()
                .is_some_and(|gep| local_is(&gep.base.value, &argument_buffer_addresses))
                || inst
                    .load()
                    .as_deref()
                    .is_some_and(|load| local_is(&load.ptr.value, &argument_buffer_addresses))
                || inst.store().as_deref().is_some_and(|(_, pointer)| {
                    local_is(&pointer.value, &argument_buffer_addresses)
                })
                || inst.call().as_deref().is_some_and(|call| {
                    call.args.iter().enumerate().any(|(index, arg)| {
                        local_is(&arg.value, &argument_buffer_addresses)
                            && (call.callee.starts_with("air.atomic.global")
                                || address_consuming_params.contains(&(call.callee.clone(), index)))
                    })
                })
        }) {
            return true;
        }

        // A logical raw-buffer helper parameter can carry a constant descriptor-relative cursor,
        // but a runtime cursor would require passing a pointer between incompatible logical pointer
        // types. Select the physical-address representation before emission when such a cursor
        // crosses an ordinary helper boundary. This is derived from the same raw-buffer facts and
        // typed GEP graph that emission consumes; source inlining is not needed to expose it.
        let mut raw_param_offsets = function
            .params
            .iter()
            .filter(|(name, ty)| {
                matches!(ty, LlType::Ptr(1))
                    && parsed
                        .raw_buffer_params
                        .contains(&(function.name.clone(), name.clone()))
            })
            .map(|(name, _)| (name.clone(), false))
            .collect::<HashMap<_, _>>();
        let mut changed = true;
        while changed {
            changed = false;
            for inst in function.carrier_insts() {
                let Some(result) = inst.result.as_ref() else {
                    continue;
                };
                if raw_param_offsets.contains_key(result) {
                    continue;
                }
                let derived = if let Some(gep) = inst.gep().as_deref() {
                    let LlValue::Local(base) = &gep.base.value else {
                        continue;
                    };
                    raw_param_offsets.get(base).copied().map(|dynamic| {
                        dynamic
                            || gep
                                .indices
                                .iter()
                                .any(|index| !is_constant_index(&index.value))
                    })
                } else if let Some((_, base)) = inst.identity_ptr_bitcast() {
                    raw_param_offsets.get(base).copied()
                } else {
                    None
                };
                if let Some(dynamic) = derived {
                    raw_param_offsets.insert(result.clone(), dynamic);
                    changed = true;
                }
            }
        }
        if function.carrier_insts().any(|inst| {
            inst.call().as_deref().is_some_and(|call| {
                !call.callee.starts_with("air.")
                    && !call.callee.starts_with("llvm.")
                    && call.args.iter().enumerate().any(|(index, arg)| {
                        atomic_consuming_params.contains(&(call.callee.clone(), index))
                            && matches!(
                                &arg.value,
                                LlValue::Local(name)
                                    if raw_param_offsets.get(name) == Some(&true)
                            )
                    })
            })
        }) {
            return true;
        }

        let mut addresses = function
            .carrier_insts()
            .filter(|inst| {
                matches!(inst.result_ty.as_ref(), Some(LlType::Ptr(1)))
                    || (matches!(inst.result_ty.as_ref(), Some(LlType::Ptr(0)))
                        && inst
                            .result
                            .as_ref()
                            .is_some_and(|name| generic_device_address_bases.contains(name)))
            })
            .filter(|inst| {
                if inst.opcode == "load" {
                    // AIR device-buffer arrays store addresses in a private aggregate, then load a
                    // `ptr addrspace(1)` through a generic field cursor. That generic form is a BDA
                    // seed only when the entry metadata declares the array contract; ordinary
                    // local pointer staging must remain in the logical address model.
                    return inst.load().as_deref().is_some_and(|load| {
                        matches!(load.ptr.ty, LlType::Ptr(1)) || has_device_buffer_array
                    });
                }
                if inst.opcode != "inttoptr" {
                    return false;
                }
                inst.operands
                    .first()
                    .and_then(|operand| operand.as_typed_value())
                    .is_some_and(|source| {
                        !matches!(
                            source.value,
                            LlValue::Zero
                                | LlValue::Int(0)
                                | LlValue::SignedInt(0)
                                | LlValue::Hex(0)
                        )
                    })
            })
            .filter_map(|inst| inst.result.clone())
            .collect::<HashSet<_>>();
        if addresses.is_empty() {
            continue;
        }

        let mut changed = true;
        while changed {
            changed = false;
            for inst in function.carrier_insts() {
                let Some(result) = inst.result.as_ref() else {
                    continue;
                };
                if addresses.contains(result)
                    || !matches!(
                        inst.result_ty.as_ref(),
                        Some(LlType::Ptr(0) | LlType::Ptr(1))
                    )
                {
                    continue;
                }
                let forwards_address = inst.opcode == "bitcast"
                    || inst.opcode == "freeze"
                    || inst.opcode == "phi"
                    || inst.opcode == "select";
                if forwards_address {
                    let mut source_is_address = false;
                    inst.visit_uses(|name| source_is_address |= addresses.contains(name));
                    if source_is_address {
                        addresses.insert(result.clone());
                        changed = true;
                    }
                }
            }
        }

        for inst in function.carrier_insts() {
            if inst
                .gep()
                .as_deref()
                .is_some_and(|gep| local_is(&gep.base.value, &addresses))
                || inst
                    .load()
                    .as_deref()
                    .is_some_and(|load| local_is(&load.ptr.value, &addresses))
                || inst.store().as_deref().is_some_and(|(value, pointer)| {
                    local_is(&value.value, &addresses) || local_is(&pointer.value, &addresses)
                })
            {
                return true;
            }

            let Some(call) = inst.call().as_deref() else {
                continue;
            };
            let consumes_address = call.callee == "mtl.force_not_checked.load.i64.p1"
                || call.callee == "air.get_data_pointer_instance_acceleration_structure"
                || call.callee.starts_with("air.atomic.global")
                || (!call.callee.starts_with("air.") && !call.callee.starts_with("llvm."));
            if consumes_address && call.args.iter().any(|arg| local_is(&arg.value, &addresses)) {
                return true;
            }
        }
    }
    false
}

fn finalize_emission(
    emitter: Emitter,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
    san_ll: &str,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    finalize_emission_outcome(emitter, buffer_layouts, san_ll).map_err(|failure| failure.error)
}

fn finalize_all_buffers_raw_emission(
    emitter: Emitter,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
    san_ll: &str,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    let mut emitted = finalize_emission(emitter, buffer_layouts, san_ll)?;
    emitted.sidecar.all_device_buffers_raw = true;
    Ok(emitted)
}

fn finalize_emission_outcome(
    emitter: Emitter,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
    san_ll: &str,
) -> Result<crate::emit_sidecar::EmittedSpirv, crate::emit_sidecar::EmissionFailure> {
    let air_data_layout = crate::layout::AirDataLayout::from_ir(san_ll).map_err(|error| {
        crate::emit_sidecar::EmissionFailure {
            error,
            ordinary_plan_rejected_functions: HashSet::new(),
            ownership_plan_rejected_functions: HashSet::new(),
        }
    })?;
    let (mut module, sidecar) =
        emitter.emit_with_sidecar(buffer_layouts, air_data_layout.as_ref())?;
    add_native_module_capabilities(&mut module);
    Ok(crate::emit_sidecar::EmittedSpirv { module, sidecar })
}

/// Emit with every device/constant (`addrspace(1)`/`addrspace(2)`) buffer pointer param modeled raw
/// (byte-offset access on a `RuntimeArray<uint>` backing, which is view-agnostic). The R4 ground-truth
/// raw retry (`translate`'s pipeline) falls back to this for a module whose default typed emission
/// produces a structurally-valid-but-mistyped buffer access — the dominant pointer-merge frontier
/// class, where the buffer's declared SPIR-V block (its Metal argument-metadata layout) is a
/// divergent type tree from the AIR `getelementptr` view (a field split across two declared members,
/// a scalar buried in a declared sub-struct, or a view that traverses past a declared scalar leaf).
pub fn emit_vulkan_spirv_all_buffers_raw(san_ll: &str) -> Result<Vec<u8>, String> {
    let kern = meta::parse_air_kernel_meta(san_ll);
    let entry_name = meta::entry_name(san_ll, "kernel");
    Ok(emit_vulkan_spirv_all_buffers_raw_with_sidecar(
        san_ll,
        kern.as_ref(),
        entry_name.as_deref(),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
        &HashSet::new(),
        &HashSet::new(),
    )?
    .into_bytes())
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_with_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
    known_ordinary_plan_rejections: &HashSet<String>,
    known_ownership_plan_rejections: &HashSet<String>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    let san_ll = vec_scalar_merge::lower_vector_scalar_pointer_merge(san_ll);
    let mut parsed = LlModule::parse_with_stage_meta(&san_ll, kern, entry_name)?;
    mark_all_device_buffers_raw(&mut parsed, false);
    finalize_all_buffers_raw_emission(
        Emitter::new(parsed).with_known_plan_rejections(
            known_ordinary_plan_rejections,
            known_ownership_plan_rejections,
        ),
        buffer_layouts,
        &san_ll,
    )
}

/// Like [`emit_vulkan_spirv_all_buffers_raw`], but additionally models every threadgroup
/// (`addrspace(3)`) buffer pointer param raw (`RuntimeArray`/concrete-vector byte-offset access). The
/// explicit broad form is retained for the raw-tier diagnostic probe. Production does not escalate
/// to it after a construction or validation failure: typed AIR analysis marks only Workgroup
/// parameters whose connected pointee views require raw storage before the selected raw emission.
/// Marking every Workgroup parameter raw can make otherwise representable Logical pointer traffic
/// inexpressible, so this broad diagnostic form must not become a product fallback.
pub fn emit_vulkan_spirv_all_buffers_raw_with_workgroup(san_ll: &str) -> Result<Vec<u8>, String> {
    let kern = meta::parse_air_kernel_meta(san_ll);
    let entry_name = meta::entry_name(san_ll, "kernel");
    Ok(emit_vulkan_spirv_all_buffers_raw_with_workgroup_sidecar(
        san_ll,
        kern.as_ref(),
        entry_name.as_deref(),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
        &HashSet::new(),
        &HashSet::new(),
    )?
    .into_bytes())
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_with_workgroup_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
    known_ordinary_plan_rejections: &HashSet<String>,
    known_ownership_plan_rejections: &HashSet<String>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    let san_ll = vec_scalar_merge::lower_vector_scalar_pointer_merge(san_ll);
    let mut parsed = LlModule::parse_with_stage_meta(&san_ll, kern, entry_name)?;
    mark_all_device_buffers_raw(&mut parsed, true);
    finalize_all_buffers_raw_emission(
        Emitter::new(parsed).with_known_plan_rejections(
            known_ordinary_plan_rejections,
            known_ownership_plan_rejections,
        ),
        buffer_layouts,
        &san_ll,
    )
}

/// Emit the all-device/constant-buffer raw view with the structured-plan attempt forced off (the
/// `relooper_feed` path) so a caller that immediately runs the relooper can rebuild the CFG directly
/// from a guaranteed-unstructured complete module. This is an intermediate-only form:
/// it intentionally omits branch/loop structured merge hints, and it must never be adopted before
/// the relooper (and any required pointer rewrite) independently validates the result.
pub fn emit_vulkan_spirv_all_buffers_raw_relooper_feed(san_ll: &str) -> Result<Vec<u8>, String> {
    let kern = meta::parse_air_kernel_meta(san_ll);
    let entry_name = meta::entry_name(san_ll, "kernel");
    Ok(
        emit_vulkan_spirv_all_buffers_raw_relooper_feed_with_sidecar(
            san_ll,
            kern.as_ref(),
            entry_name.as_deref(),
            kern.as_ref().map(|meta| &meta.buffer_layouts),
        )?
        .into_bytes(),
    )
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_relooper_feed_with_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    let san_ll = vec_scalar_merge::lower_vector_scalar_pointer_merge(san_ll);
    let mut parsed = LlModule::parse_with_stage_meta(&san_ll, kern, entry_name)?;
    let requires_device_addresses = requires_device_address_model(&parsed);
    mark_all_device_buffers_raw(&mut parsed, false);
    let mut emitter = Emitter::new(parsed).with_relooper_feed();
    if requires_device_addresses {
        // CFG ownership does not erase the source's address-domain requirement. The raw-relooper
        // construction must carry the same physical pointer representation selected by the primary
        // graph, otherwise integer atomics over float slots regress to illegal Logical bitcasts.
        emitter = emitter.with_bda_device_pointers();
    }
    finalize_all_buffers_raw_emission(emitter, buffer_layouts, &san_ll)
}

/// Emit with every device/constant buffer modeled raw AND device-pointer (BDA) modeling enabled: a
/// device pointer (`addrspace(1)`) LOADED from a buffer word is its real 64-bit address (an
/// `OpConvertUToPtr` PhysicalStorageBuffer64 pointer), so the kernel can STORE it (a verbatim 8-byte
/// copy) and DEREFERENCE it (`address + struct/array offset`). This is the honest lowering of the
/// "BDA" frontier class — the Apple BVH builders that load a device pointer from one buffer, store it
/// into another, and walk it as a `MTLSWBVH*` struct (the `raw store for Ptr(1) is not covered yet`
/// emit gap). Byte-correct by construction: the stored bytes are the exact loaded address, the deref
/// is `address + offset` with no tag-bit manipulation (verified across the cluster), and the address
/// is a real Vulkan device address under `buffer_device_address`. The default Logical emit is never
/// altered unless construction selects this representation. The internal emitter's
/// `with_bda_device_pointers` mode owns the address conversion.
pub fn emit_vulkan_spirv_all_buffers_raw_bda(san_ll: &str) -> Result<Vec<u8>, String> {
    let kern = meta::parse_air_kernel_meta(san_ll);
    let entry_name = meta::entry_name(san_ll, "kernel");
    Ok(emit_vulkan_spirv_all_buffers_raw_bda_with_sidecar(
        san_ll,
        kern.as_ref(),
        entry_name.as_deref(),
        kern.as_ref().map(|meta| &meta.buffer_layouts),
        &HashSet::new(),
        &HashSet::new(),
    )?
    .into_bytes())
}

pub(crate) fn emit_vulkan_spirv_all_buffers_raw_bda_with_sidecar(
    san_ll: &str,
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    buffer_layouts: Option<&HashMap<u32, meta::AirType>>,
    known_ordinary_plan_rejections: &HashSet<String>,
    known_ownership_plan_rejections: &HashSet<String>,
) -> Result<crate::emit_sidecar::EmittedSpirv, String> {
    let san_ll = vec_scalar_merge::lower_vector_scalar_pointer_merge(san_ll);
    let mut parsed = LlModule::parse_with_stage_meta(&san_ll, kern, entry_name)?;
    mark_all_device_buffers_raw(&mut parsed, false);
    finalize_all_buffers_raw_emission(
        Emitter::new(parsed)
            .with_known_plan_rejections(
                known_ordinary_plan_rejections,
                known_ownership_plan_rejections,
            )
            .with_bda_device_pointers(),
        buffer_layouts,
        &san_ll,
    )
}

/// Mark every device/constant (`addrspace(1)`/`addrspace(2)`) buffer pointer param of every function
/// raw in `parsed.raw_buffer_params`. With `include_workgroup`, also marks threadgroup
/// (`addrspace(3)`) buffer params (the explicit diagnostic form — see
/// [`emit_vulkan_spirv_all_buffers_raw_with_workgroup`]).
fn mark_all_device_buffers_raw(parsed: &mut LlModule, include_workgroup: bool) {
    let metadata_buffers = parsed.metadata_data_buffer_params.clone();
    let mut keys = Vec::new();
    for function in &parsed.functions {
        for (name, ty) in &function.params {
            let key = (function.name.clone(), name.clone());
            let raw = matches!(ty, ir::LlType::Ptr(1 | 2))
                || metadata_buffers.contains(&key)
                || (include_workgroup && matches!(ty, ir::LlType::Ptr(3)));
            if raw {
                keys.push(key);
            }
        }
    }
    for key in keys {
        parsed.raw_buffer_params.insert(key);
    }
}

fn mark_entry_buffer_params_raw(
    parsed: &mut LlModule,
    entry_name: Option<&str>,
    params: &HashSet<u32>,
) {
    let Some(entry_name) = entry_name else {
        return;
    };
    let keys = parsed
        .functions
        .iter()
        .find(|function| function.name == entry_name)
        .into_iter()
        .flat_map(|function| {
            function
                .params
                .iter()
                .enumerate()
                .filter_map(move |(index, (name, ty))| {
                    if params.contains(&(index as u32)) && matches!(ty, ir::LlType::Ptr(1 | 2)) {
                        Some((function.name.clone(), name.clone()))
                    } else {
                        None
                    }
                })
        })
        .collect::<Vec<_>>();
    parsed.raw_buffer_params.extend(keys);
}
