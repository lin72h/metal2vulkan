//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    pub(in crate::native::emitter) fn emit_void_air_call(
        &mut self,
        call: &LlCall,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        match call.callee.as_str() {
            "air.simdgroup.barrier" => {
                if call.args.len() != 2 {
                    return Err(
                        "native emitter: air.simdgroup.barrier expects 2 operands".to_string()
                    );
                }
                let scope = self.const_uint(Scope::Subgroup as u32)?;
                let memory_scope =
                    self.const_uint(air_barrier_memory_scope(call, Scope::Subgroup) as u32)?;
                let semantics = self.const_uint(air_barrier_memory_semantics(call).bits())?;
                instructions.push(Self::inst(
                    Op::ControlBarrier,
                    None,
                    None,
                    vec![
                        Operand::IdScope(scope),
                        Operand::IdScope(memory_scope),
                        Operand::IdMemorySemantics(semantics),
                    ],
                ));
                Ok(true)
            }
            "air.wg.barrier" => {
                if call.args.len() != 2 {
                    return Err("native emitter: air.wg.barrier expects 2 operands".to_string());
                }
                let scope = self.const_uint(Scope::Workgroup as u32)?;
                let memory_scope =
                    self.const_uint(air_barrier_memory_scope(call, Scope::Workgroup) as u32)?;
                let semantics = self.const_uint(air_barrier_memory_semantics(call).bits())?;
                instructions.push(Self::inst(
                    Op::ControlBarrier,
                    None,
                    None,
                    vec![
                        Operand::IdScope(scope),
                        Operand::IdScope(memory_scope),
                        Operand::IdMemorySemantics(semantics),
                    ],
                ));
                Ok(true)
            }
            "air.atomic.fence" => {
                if call.args.len() != 3 {
                    return Err("native emitter: air.atomic.fence expects 3 operands".to_string());
                }
                let scope_kind = match air_i32_literal(&call.args[2].value) {
                    Some(1) => Scope::Workgroup,
                    _ => Scope::Device,
                };
                let semantics_kind = if scope_kind == Scope::Workgroup {
                    MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY
                } else {
                    MemorySemantics::ACQUIRE_RELEASE
                        | MemorySemantics::UNIFORM_MEMORY
                        | MemorySemantics::CROSS_WORKGROUP_MEMORY
                };
                let scope = self.const_uint(scope_kind as u32)?;
                let semantics = self.const_uint(semantics_kind.bits())?;
                instructions.push(Self::inst(
                    Op::MemoryBarrier,
                    None,
                    None,
                    vec![
                        Operand::IdScope(scope),
                        Operand::IdMemorySemantics(semantics),
                    ],
                ));
                Ok(true)
            }
            callee if callee.starts_with("air.fence_texture") => {
                if call.args.len() != 1 {
                    return Err(format!("native emitter: {} expects 1 operand", call.callee));
                }
                let scope = self.const_uint(Scope::Device as u32)?;
                let semantics = self.const_uint(
                    (MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::IMAGE_MEMORY).bits(),
                )?;
                instructions.push(Self::inst(
                    Op::MemoryBarrier,
                    None,
                    None,
                    vec![
                        Operand::IdScope(scope),
                        Operand::IdMemorySemantics(semantics),
                    ],
                ));
                Ok(true)
            }
            callee if is_coherent_air_store(callee) => {
                if call.args.len() != 2 {
                    return Err(format!(
                        "native emitter: {} expects 2 operands",
                        call.callee
                    ));
                }
                let value =
                    self.value_id_in(&call.args[0].value, &call.args[0].ty, instructions)?;
                let ptr_arg = &call.args[1];
                if let LlValue::Local(name) = &ptr_arg.value {
                    if let Some(raw) = self.raw_offsets.get(name).cloned() {
                        self.emit_raw_store(&call.args[0].ty, value, &raw, None, instructions)?;
                        return Ok(true);
                    }
                    if self.unmodeled_pointers.contains(name) {
                        return Ok(true);
                    }
                }
                let ptr = self.value_id_in(&ptr_arg.value, &ptr_arg.ty, instructions)?;
                instructions.push(Self::inst(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(ptr), Operand::IdRef(value)],
                ));
                Ok(true)
            }
            "air.atomic.local.store.i32" | "air.atomic.global.store.i32" => {
                if call.args.len() != 5 {
                    return Err(format!(
                        "native emitter: {} expects 5 operands",
                        call.callee
                    ));
                }
                let ptr = self.atomic_i32_pointer_id(&call.args[0], instructions)?;
                let value =
                    self.value_id_in(&call.args[1].value, &call.args[1].ty, instructions)?;
                let scope_kind = self.atomic_i32_scope_for_arg(&call.args[0])?;
                let scope = self.const_uint(scope_kind as u32)?;
                let semantics_kind =
                    Self::atomic_i32_memory_semantics(scope_kind, MemorySemantics::RELEASE);
                let semantics = self.const_uint(semantics_kind.bits())?;
                instructions.push(Self::inst(
                    Op::AtomicStore,
                    None,
                    None,
                    vec![
                        Operand::IdRef(ptr),
                        Operand::IdScope(scope),
                        Operand::IdMemorySemantics(semantics),
                        Operand::IdRef(value),
                    ],
                ));
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub(in crate::native::emitter) fn emit_value_air_call(
        &mut self,
        call: &LlCall,
        name: &str,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        match call.callee.as_str() {
            // `air.is_null_texture_<dim>(%tex)` asks whether a texture handle is the null texture.
            // We consume it HERE only when `%tex` is a value we ourselves synthesized from
            // `air.get_null_texture_*` (tracked in `null_texture_values`): that value never crosses
            // the emitter->reparse seam as a recognizable image, so the passes-layer null tracking
            // (`ctx.null_image_values`) can't see it and would answer FALSE. A real bound texture is
            // NOT in the set, so it falls through to the default emission and the passes layer lowers
            // it exactly as before — keeping every regression case byte-identical. Dispatches on a stable
            // `air.*` ABI symbol plus our own data-flow set, never a shader name.
            callee if callee.starts_with("air.is_null_texture") => {
                if call.args.len() == 1 {
                    if let LlValue::Local(arg_name) = &call.args[0].value {
                        if self.null_texture_values.contains(arg_name) {
                            let result_ty = self.resolve_type(&call.ret)?;
                            let result_type = self.type_id(&result_ty)?;
                            let result = self.result_id(name, &result_ty)?;
                            let c = self.const_bool(true)?;
                            instructions.push(Self::inst(
                                Op::CopyObject,
                                Some(result_type),
                                Some(result),
                                vec![Operand::IdRef(c)],
                            ));
                            return Ok(true);
                        }
                    }
                }
                Ok(false)
            }
            "air.get_instance_count_instance_acceleration_structure" => {
                if call.args.len() != 1 {
                    return Err(format!("native emitter: {} expects 1 operand", call.callee));
                }
                let LlValue::Local(shadow_name) = &call.args[0].value else {
                    return Err(format!(
                        "native emitter: {} shadow operand is not SSA",
                        call.callee
                    ));
                };
                let Some(mut raw) = self.raw_offsets.get(shadow_name).cloned() else {
                    return Ok(false);
                };
                raw.const_off += crate::as_shadow::INSTANCE_COUNT_BYTE_OFFSET as i64;
                let result_ty = self.resolve_type(&call.ret)?;
                if result_ty != LlType::Int(32) {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}, expected i32",
                        call.callee
                    ));
                }
                let result = self.result_id(name, &result_ty)?;
                self.emit_raw_load(result, &result_ty, &raw, Some(4), instructions)?;
                Ok(true)
            }
            "air.get_primitive_acceleration_structure_instance_acceleration_structure" => {
                if call.args.len() != 2 {
                    return Err(format!(
                        "native emitter: {} expects 2 operands",
                        call.callee
                    ));
                }
                let LlValue::Local(shadow_name) = &call.args[0].value else {
                    return Err(format!(
                        "native emitter: {} shadow operand is not SSA",
                        call.callee
                    ));
                };
                let Some(mut raw) = self.raw_offsets.get(shadow_name).cloned() else {
                    return Ok(false);
                };
                let result_ty = self.resolve_type(&call.ret)?;
                let LlType::Ptr(addrspace) = result_ty else {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}, expected pointer",
                        call.callee
                    ));
                };
                raw.const_off += crate::as_shadow::CHILD_REFERENCES_BYTE_OFFSET as i64;
                raw.dyn_terms.push((
                    call.args[1].clone(),
                    crate::as_shadow::CHILD_REFERENCE_BYTE_STRIDE as i64,
                ));
                self.define_unmodeled_byte_pointer_value(name, addrspace)?;
                let (payload, is_null) =
                    self.emit_raw_pointer_payload(&raw, 0, Some(8), instructions)?;
                self.pointer_payload_words.insert(name.to_string(), payload);
                self.record_pointer_nullness(name.to_string(), is_null);
                Ok(true)
            }
            // `air.get_data_pointer_instance_acceleration_structure(%p)` returns the data pointer of an
            // instance acceleration structure. In the paravirt AS ABI (a design decision this stack
            // co-designs — see the AS-pointer-passthrough note) the instance AS's data pointer IS its
            // device address, so the intrinsic is an IDENTITY passthrough of `%p`. The argument is a
            // device pointer loaded from the instances buffer (BDA-eligible); when BDA mode has rooted
            // it at a device address, alias the result to that same device-address offset. A later
            // store of the result then copies the address verbatim and a field-offset GEP/deref reads
            // through it — exactly the plain-BDA path the 12 `store ptr addrspace(1)` cases use, with no
            // intervening tag-bit arithmetic. Byte-correct GIVEN the host rail lays out the instance AS
            // at this device address (the contract). Dispatches on a stable `air.*` ABI symbol, never a
            // shader name; outside BDA mode it returns `Ok(false)` so the default emit is untouched.
            "air.get_data_pointer_instance_acceleration_structure" => {
                if self.bda_device_pointers && call.args.len() == 1 {
                    if let LlValue::Local(arg_name) = &call.args[0].value {
                        if let Some(raw) = self.raw_offsets.get(arg_name).cloned() {
                            if raw.device_addr_base.is_some() {
                                self.used_device_address = true;
                                self.raw_offsets.insert(name.to_string(), raw);
                                self.pointer_storage
                                    .insert(name.to_string(), StorageClass::PhysicalStorageBuffer);
                                return Ok(true);
                            }
                        }
                    }
                }

                Ok(false)
            }
            callee if is_coherent_air_load(callee) => {
                if call.args.len() != 1 {
                    return Err(format!("native emitter: {} expects 1 operand", call.callee));
                }
                let result_ty = self.resolve_type(&call.ret)?;
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(name, &result_ty)?;
                let ptr_arg = &call.args[0];
                if let LlValue::Local(ptr_name) = &ptr_arg.value {
                    if let Some(raw) = self.raw_offsets.get(ptr_name).cloned() {
                        self.emit_raw_load(result, &result_ty, &raw, None, instructions)?;
                        return Ok(true);
                    }
                    if self.unmodeled_pointers.contains(ptr_name) {
                        let zero = self.const_null(&result_ty)?;
                        instructions.push(Self::inst(
                            Op::CopyObject,
                            Some(result_type),
                            Some(result),
                            vec![Operand::IdRef(zero)],
                        ));
                        return Ok(true);
                    }
                }
                let ptr = self.value_id_in(&ptr_arg.value, &ptr_arg.ty, instructions)?;
                instructions.push(Self::inst(
                    Op::Load,
                    Some(result_type),
                    Some(result),
                    vec![Operand::IdRef(ptr)],
                ));
                Ok(true)
            }
            "air.simd_any" | "air.simd_all" => {
                if call.args.len() != 1 {
                    return Err(format!("native emitter: {} expects 1 operand", call.callee));
                }
                let result_ty = self.resolve_type(&call.ret)?;
                if !is_bool_type(&result_ty) {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}",
                        call.callee
                    ));
                }
                let value_ty = self.resolve_type(&call.args[0].ty)?;
                if !is_bool_type(&value_ty) {
                    return Err(format!(
                        "native emitter: {} value is {value_ty:?}",
                        call.callee
                    ));
                }
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(name, &result_ty)?;
                let value =
                    self.value_id_in(&call.args[0].value, &call.args[0].ty, instructions)?;
                let scope = self.const_uint(Scope::Subgroup as u32)?;
                let op = if call.callee == "air.simd_any" {
                    Op::GroupNonUniformAny
                } else {
                    Op::GroupNonUniformAll
                };
                instructions.push(Self::inst(
                    op,
                    Some(result_type),
                    Some(result),
                    vec![Operand::IdScope(scope), Operand::IdRef(value)],
                ));
                Ok(true)
            }
            "air.simd_ballot.i64" => {
                if call.args.len() != 1 {
                    return Err("native emitter: air.simd_ballot.i64 expects 1 operand".to_string());
                }
                let result_ty = self.resolve_type(&call.ret)?;
                if result_ty != LlType::Int(64) {
                    return Err(format!(
                        "native emitter: air.simd_ballot.i64 returned {result_ty:?}"
                    ));
                }
                let predicate_ty = self.resolve_type(&call.args[0].ty)?;
                if !is_bool_type(&predicate_ty) {
                    return Err(format!(
                        "native emitter: air.simd_ballot.i64 predicate is {predicate_ty:?}"
                    ));
                }

                let uint = self.type_id(&LlType::Int(32))?;
                let ulong = self.type_id(&LlType::Int(64))?;
                let ballot_ty = LlType::Vector(Box::new(LlType::Int(32)), 4);
                let ballot_type = self.type_id(&ballot_ty)?;
                let result = self.result_id(name, &result_ty)?;
                let predicate =
                    self.value_id_in(&call.args[0].value, &call.args[0].ty, instructions)?;
                let scope = self.const_uint(Scope::Subgroup as u32)?;

                let ballot = self.fresh();
                instructions.push(Self::inst(
                    Op::GroupNonUniformBallot,
                    Some(ballot_type),
                    Some(ballot),
                    vec![Operand::IdScope(scope), Operand::IdRef(predicate)],
                ));

                let lo32 = self.fresh();
                instructions.push(Self::inst(
                    Op::CompositeExtract,
                    Some(uint),
                    Some(lo32),
                    vec![Operand::IdRef(ballot), Operand::LiteralBit32(0)],
                ));
                let hi32 = self.fresh();
                instructions.push(Self::inst(
                    Op::CompositeExtract,
                    Some(uint),
                    Some(hi32),
                    vec![Operand::IdRef(ballot), Operand::LiteralBit32(1)],
                ));

                let lo64 = self.fresh();
                instructions.push(Self::inst(
                    Op::UConvert,
                    Some(ulong),
                    Some(lo64),
                    vec![Operand::IdRef(lo32)],
                ));
                let hi64 = self.fresh();
                instructions.push(Self::inst(
                    Op::UConvert,
                    Some(ulong),
                    Some(hi64),
                    vec![Operand::IdRef(hi32)],
                ));

                let shift = self.const_int(64, 32)?;
                let shifted_hi = self.fresh();
                instructions.push(Self::inst(
                    Op::ShiftLeftLogical,
                    Some(ulong),
                    Some(shifted_hi),
                    vec![Operand::IdRef(hi64), Operand::IdRef(shift)],
                ));
                instructions.push(Self::inst(
                    Op::BitwiseOr,
                    Some(ulong),
                    Some(result),
                    vec![Operand::IdRef(shifted_hi), Operand::IdRef(lo64)],
                ));
                Ok(true)
            }
            "air.simd_shuffle.u.i32" | "air.simd_shuffle.s.i32" => {
                if call.args.len() != 2 {
                    return Err(format!(
                        "native emitter: {} expects 2 operands",
                        call.callee
                    ));
                }
                let result_ty = self.resolve_type(&call.ret)?;
                if result_ty != LlType::Int(32) {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}",
                        call.callee
                    ));
                }
                let value_ty = self.resolve_type(&call.args[0].ty)?;
                if value_ty != LlType::Int(32) {
                    return Err(format!(
                        "native emitter: {} value is {value_ty:?}",
                        call.callee
                    ));
                }
                let lane_ty = self.resolve_type(&call.args[1].ty)?;
                if !matches!(lane_ty, LlType::Int(_)) {
                    return Err(format!(
                        "native emitter: {} lane is {lane_ty:?}",
                        call.callee
                    ));
                }
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(name, &result_ty)?;
                let value =
                    self.value_id_in(&call.args[0].value, &call.args[0].ty, instructions)?;
                let lane = self.value_id_in(&call.args[1].value, &call.args[1].ty, instructions)?;
                let uint = self.type_id(&LlType::Int(32))?;
                let invocation = if lane_ty == LlType::Int(32) {
                    lane
                } else {
                    let converted = self.fresh();
                    instructions.push(Self::inst(
                        Op::UConvert,
                        Some(uint),
                        Some(converted),
                        vec![Operand::IdRef(lane)],
                    ));
                    converted
                };
                let scope = self.const_uint(Scope::Subgroup as u32)?;
                instructions.push(Self::inst(
                    Op::GroupNonUniformShuffle,
                    Some(result_type),
                    Some(result),
                    vec![
                        Operand::IdScope(scope),
                        Operand::IdRef(value),
                        Operand::IdRef(invocation),
                    ],
                ));
                Ok(true)
            }
            "air.atomic.local.load.i32" | "air.atomic.global.load.i32" => {
                if call.args.len() != 4 {
                    return Err(format!(
                        "native emitter: {} expects 4 operands",
                        call.callee
                    ));
                }
                let result_ty = self.resolve_type(&call.ret)?;
                if result_ty != LlType::Int(32) {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}",
                        call.callee
                    ));
                }
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(name, &result_ty)?;
                let ptr = self.atomic_i32_pointer_id(&call.args[0], instructions)?;
                let scope_kind = self.atomic_i32_scope_for_arg(&call.args[0])?;
                let scope = self.const_uint(scope_kind as u32)?;
                let semantics_kind =
                    Self::atomic_i32_memory_semantics(scope_kind, MemorySemantics::ACQUIRE);
                let semantics = self.const_uint(semantics_kind.bits())?;
                instructions.push(Self::inst(
                    Op::AtomicLoad,
                    Some(result_type),
                    Some(result),
                    vec![
                        Operand::IdRef(ptr),
                        Operand::IdScope(scope),
                        Operand::IdMemorySemantics(semantics),
                    ],
                ));
                Ok(true)
            }
            "air.atomic.global.add.f32" => {
                if call.args.len() != 5 {
                    return Err(format!(
                        "native emitter: {} expects 5 operands",
                        call.callee
                    ));
                }
                let result_ty = self.resolve_type(&call.ret)?;
                if result_ty != LlType::Float {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}",
                        call.callee
                    ));
                }
                self.require_capability(Capability::AtomicFloat32AddEXT);
                self.require_extension("SPV_EXT_shader_atomic_float_add");
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(name, &result_ty)?;
                let ptr = self.atomic_f32_pointer_id(&call.args[0], instructions)?;
                let value =
                    self.value_id_in(&call.args[1].value, &call.args[1].ty, instructions)?;
                let scope = self.const_uint(Scope::Device as u32)?;
                let semantics = self.const_uint(MemorySemantics::RELAXED.bits())?;
                instructions.push(Self::inst(
                    Op::AtomicFAddEXT,
                    Some(result_type),
                    Some(result),
                    vec![
                        Operand::IdRef(ptr),
                        Operand::IdScope(scope),
                        Operand::IdMemorySemantics(semantics),
                        Operand::IdRef(value),
                    ],
                ));
                Ok(true)
            }
            "air.atomic.global.sub.f32" => {
                // SPIR-V has no atomic float subtract; an atomic fetch-sub is exactly an atomic
                // fetch-add of the negated operand (both return the prior value), so negate then
                // AtomicFAddEXT.
                if call.args.len() != 5 {
                    return Err(format!(
                        "native emitter: {} expects 5 operands",
                        call.callee
                    ));
                }
                let result_ty = self.resolve_type(&call.ret)?;
                if result_ty != LlType::Float {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}",
                        call.callee
                    ));
                }
                self.require_capability(Capability::AtomicFloat32AddEXT);
                self.require_extension("SPV_EXT_shader_atomic_float_add");
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(name, &result_ty)?;
                let ptr = self.atomic_f32_pointer_id(&call.args[0], instructions)?;
                let value =
                    self.value_id_in(&call.args[1].value, &call.args[1].ty, instructions)?;
                let negated = self.fresh();
                instructions.push(Self::inst(
                    Op::FNegate,
                    Some(result_type),
                    Some(negated),
                    vec![Operand::IdRef(value)],
                ));
                let scope = self.const_uint(Scope::Device as u32)?;
                let semantics = self.const_uint(MemorySemantics::RELAXED.bits())?;
                instructions.push(Self::inst(
                    Op::AtomicFAddEXT,
                    Some(result_type),
                    Some(result),
                    vec![
                        Operand::IdRef(ptr),
                        Operand::IdScope(scope),
                        Operand::IdMemorySemantics(semantics),
                        Operand::IdRef(negated),
                    ],
                ));
                Ok(true)
            }
            "air.atomic.local.cmpxchg.weak.i32" | "air.atomic.global.cmpxchg.weak.i32" => {
                if call.args.len() != 7 {
                    return Err(format!(
                        "native emitter: {} expects 7 operands",
                        call.callee
                    ));
                }
                let result_ty = self.resolve_type(&call.ret)?;
                if result_ty != LlType::Int(32) {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}",
                        call.callee
                    ));
                }
                let compare_ptr_ty = self.resolve_type(&call.args[1].ty)?;
                if !matches!(compare_ptr_ty, LlType::Ptr(_)) {
                    return Err(format!(
                        "native emitter: {} compare operand is {compare_ptr_ty:?}",
                        call.callee
                    ));
                }
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(name, &result_ty)?;
                let ptr = self.atomic_i32_pointer_id(&call.args[0], instructions)?;
                let compare_ptr = self.value_id(&call.args[1].value, &call.args[1].ty)?;
                let compare = self.fresh();
                instructions.push(Self::inst(
                    Op::Load,
                    Some(result_type),
                    Some(compare),
                    vec![Operand::IdRef(compare_ptr)],
                ));
                let value =
                    self.value_id_in(&call.args[2].value, &call.args[2].ty, instructions)?;
                let scope_kind = self.atomic_i32_scope_for_arg(&call.args[0])?;
                let scope = self.const_uint(scope_kind as u32)?;
                let success_semantics_kind =
                    Self::atomic_i32_memory_semantics(scope_kind, MemorySemantics::ACQUIRE_RELEASE);
                let failure_semantics_kind =
                    Self::atomic_i32_memory_semantics(scope_kind, MemorySemantics::ACQUIRE);
                let success_semantics = self.const_uint(success_semantics_kind.bits())?;
                let failure_semantics = self.const_uint(failure_semantics_kind.bits())?;
                instructions.push(Self::inst(
                    Op::AtomicCompareExchange,
                    Some(result_type),
                    Some(result),
                    vec![
                        Operand::IdRef(ptr),
                        Operand::IdScope(scope),
                        Operand::IdMemorySemantics(success_semantics),
                        Operand::IdMemorySemantics(failure_semantics),
                        Operand::IdRef(value),
                        Operand::IdRef(compare),
                    ],
                ));
                instructions.push(Self::inst(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(compare_ptr), Operand::IdRef(result)],
                ));
                Ok(true)
            }
            "air.atomic.local.add.s.i32"
            | "air.atomic.local.add.u.i32"
            | "air.atomic.local.max.s.i32"
            | "air.atomic.local.max.u.i32"
            | "air.atomic.local.min.s.i32"
            | "air.atomic.local.min.u.i32"
            | "air.atomic.local.and.u.i32"
            | "air.atomic.local.or.u.i32"
            | "air.atomic.local.xchg.i32"
            | "air.atomic.global.add.s.i32"
            | "air.atomic.global.add.u.i32"
            | "air.atomic.global.and.u.i32"
            | "air.atomic.global.or.u.i32"
            | "air.atomic.global.xchg.i32"
            | "air.atomic.global.max.s.i32"
            | "air.atomic.global.max.u.i32"
            | "air.atomic.global.min.s.i32"
            | "air.atomic.global.min.u.i32"
            | "air.atomic.global.sub.u.i32" => {
                if call.args.len() != 5 {
                    return Err(format!(
                        "native emitter: {} expects 5 operands",
                        call.callee
                    ));
                }
                let result_ty = self.resolve_type(&call.ret)?;
                if result_ty != LlType::Int(32) {
                    return Err(format!(
                        "native emitter: {} returned {result_ty:?}",
                        call.callee
                    ));
                }
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(name, &result_ty)?;
                let ptr = self.atomic_i32_pointer_id(&call.args[0], instructions)?;
                let value =
                    self.value_id_in(&call.args[1].value, &call.args[1].ty, instructions)?;
                let op = match call.callee.as_str() {
                    "air.atomic.local.add.s.i32" => Op::AtomicIAdd,
                    "air.atomic.local.add.u.i32" => Op::AtomicIAdd,
                    "air.atomic.local.max.s.i32" => Op::AtomicSMax,
                    "air.atomic.local.max.u.i32" => Op::AtomicUMax,
                    "air.atomic.local.min.s.i32" => Op::AtomicSMin,
                    "air.atomic.local.min.u.i32" => Op::AtomicUMin,
                    "air.atomic.local.and.u.i32" => Op::AtomicAnd,
                    "air.atomic.global.add.s.i32" => Op::AtomicIAdd,
                    "air.atomic.global.add.u.i32" => Op::AtomicIAdd,
                    "air.atomic.global.and.u.i32" => Op::AtomicAnd,
                    "air.atomic.global.xchg.i32" => Op::AtomicExchange,
                    "air.atomic.global.max.s.i32" => Op::AtomicSMax,
                    "air.atomic.global.max.u.i32" => Op::AtomicUMax,
                    "air.atomic.global.min.s.i32" => Op::AtomicSMin,
                    "air.atomic.global.min.u.i32" => Op::AtomicUMin,
                    "air.atomic.global.or.u.i32" => Op::AtomicOr,
                    "air.atomic.global.sub.u.i32" => Op::AtomicISub,
                    "air.atomic.local.or.u.i32" => Op::AtomicOr,
                    "air.atomic.local.xchg.i32" => Op::AtomicExchange,
                    _ => Op::AtomicIAdd,
                };
                let scope_kind = self.atomic_i32_scope_for_arg(&call.args[0])?;
                let scope = self.const_uint(scope_kind as u32)?;
                let semantics_kind =
                    Self::atomic_i32_memory_semantics(scope_kind, MemorySemantics::ACQUIRE_RELEASE);
                let semantics = self.const_uint(semantics_kind.bits())?;
                instructions.push(Self::inst(
                    op,
                    Some(result_type),
                    Some(result),
                    vec![
                        Operand::IdRef(ptr),
                        Operand::IdScope(scope),
                        Operand::IdMemorySemantics(semantics),
                        Operand::IdRef(value),
                    ],
                ));
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

fn air_barrier_memory_semantics(call: &LlCall) -> MemorySemantics {
    let flags = call
        .args
        .first()
        .and_then(|arg| air_i32_literal(&arg.value))
        .unwrap_or(0);
    let mut semantics = MemorySemantics::ACQUIRE_RELEASE;
    if flags & 1 != 0 {
        semantics |= MemorySemantics::UNIFORM_MEMORY | MemorySemantics::CROSS_WORKGROUP_MEMORY;
    }
    if flags & 2 != 0 {
        semantics |= MemorySemantics::WORKGROUP_MEMORY;
    }
    if flags & 4 != 0 {
        semantics |= MemorySemantics::IMAGE_MEMORY;
    }
    if semantics == MemorySemantics::ACQUIRE_RELEASE {
        semantics |= MemorySemantics::WORKGROUP_MEMORY;
    }
    semantics
}

fn air_barrier_memory_scope(call: &LlCall, default_scope: Scope) -> Scope {
    let flags = call
        .args
        .first()
        .and_then(|arg| air_i32_literal(&arg.value))
        .unwrap_or(0);
    if flags & (1 | 4) != 0 {
        Scope::Device
    } else {
        default_scope
    }
}
