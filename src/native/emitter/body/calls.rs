//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    pub(in crate::native::emitter) fn emit_mtl_force_not_checked_load_call(
        &mut self,
        call: &LlCall,
        name: &str,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if call.callee != "mtl.force_not_checked.load.i64.p1" {
            return Ok(false);
        }
        if call.args.len() != 1 {
            return Err(format!(
                "native emitter: {} expected one pointer argument, got {}",
                call.callee,
                call.args.len()
            ));
        }
        let result_ty = self.resolve_type(&call.ret)?;
        if result_ty != LlType::Int(64) {
            return Err(format!(
                "native emitter: {} returned unsupported type {:?}",
                call.callee, result_ty
            ));
        }
        let arg = &call.args[0];
        let LlType::Ptr(addrspace) = self.resolve_type(&arg.ty)? else {
            return Err(format!(
                "native emitter: {} argument is not a pointer: {:?}",
                call.callee, arg.ty
            ));
        };
        if addrspace != 1 {
            return Err(format!(
                "native emitter: {} expected ptr addrspace(1), got addrspace({addrspace})",
                call.callee
            ));
        }
        let LlValue::Local(arg_name) = &arg.value else {
            return Err(format!(
                "native emitter: {} requires a local device-address pointer",
                call.callee
            ));
        };
        let Some(raw) = self.raw_offsets.get(arg_name).cloned() else {
            return Err(format!(
                "native emitter: {} requires BDA device-address lowering",
                call.callee
            ));
        };
        if raw.device_addr_base.is_none() {
            return Err(format!(
                "native emitter: {} pointer is not backed by a device address",
                call.callee
            ));
        }
        let result = self.result_id(name, &result_ty)?;
        self.emit_device_addr_load(result, &result_ty, &raw, instructions)?;
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_visible_function_table_placeholder_call(
        &mut self,
        call: &LlCall,
        name: &str,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if call.callee == "air.get_size_visible_function_table" {
            self.validate_call_args(call, instructions)?;
            let result_ty = self.resolve_type(&call.ret)?;
            if !matches!(result_ty, LlType::Int(_)) {
                return Err(format!(
                    "native emitter: visible function table size returned {result_ty:?}"
                ));
            }
            let zero = self.const_null(&result_ty)?;
            let result_type = self.type_id(&result_ty)?;
            let result = self.result_id(name, &result_ty)?;
            instructions.push(Self::inst(
                Op::CopyObject,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(zero)],
            ));
            return Ok(true);
        }
        if call.callee == "air.get_function_pointer_visible_function_table" {
            self.validate_call_args(call, instructions)?;
            let result_ty = self.resolve_type(&call.ret)?;
            let LlType::Ptr(addrspace) = result_ty else {
                return Err(format!(
                    "native emitter: visible function table pointer returned {result_ty:?}"
                ));
            };
            self.define_unmodeled_byte_pointer_value(name, addrspace)?;
            return Ok(true);
        }
        Ok(false)
    }

    pub(in crate::native::emitter) fn emit_imageblock_data_call(
        &mut self,
        call: &LlCall,
        name: &str,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if call.callee != "air.imageblock_data" {
            return Ok(false);
        }
        self.validate_call_args(call, instructions)?;
        let result_ty = self.resolve_type(&call.ret)?;
        let LlType::Ptr(addrspace) = result_ty else {
            return Err(format!(
                "native emitter: imageblock_data returned non-pointer {result_ty:?}"
            ));
        };
        if addrspace != 4 {
            return Err(format!(
                "native emitter: imageblock_data returned ptr addrspace({addrspace})"
            ));
        }
        let pointee = self
            .ir
            .imageblock_data_pointee
            .clone()
            .unwrap_or_else(|| LlType::Vector(Box::new(LlType::Half), 4));
        if self.ir.imageblock_dimensions.is_none() && !self.ir.imageblock_shared_cells {
            // A single structural coordinate is per-invocation slice staging: model it as a flat
            // scalar-slot scratch and hand back element 0 as the base. AIR byte-addresses each slice
            // off this base (bitcast for member 0, `getelementptr i8` for the rest), and the
            // `root_is_indexed_container` provenance drives those to scratch element access chains.
            let array_ty = LlType::Array(Box::new(pointee.clone()), 2);
            let slot_ptr_ty = self.ptr_type_id(StorageClass::Private, &pointee)?;
            let storage = if let Some((storage, _)) = self.imageblock_data_scratch.clone() {
                storage
            } else {
                let array_ptr_ty = self.ptr_type_id(StorageClass::Private, &array_ty)?;
                let storage = self.fresh();
                self.module.types_global_values.push(Self::inst(
                    Op::Variable,
                    Some(array_ptr_ty),
                    Some(storage),
                    vec![Operand::StorageClass(StorageClass::Private)],
                ));
                self.imageblock_data_scratch = Some((storage, array_ty.clone()));
                storage
            };
            let result = self.result_id(name, &LlType::Ptr(addrspace))?;
            let zero = self.const_uint(0)?;
            instructions.push(Self::inst(
                Op::InBoundsAccessChain,
                Some(slot_ptr_ty),
                Some(result),
                vec![Operand::IdRef(storage), Operand::IdRef(zero)],
            ));
            self.pointer_storage
                .insert(name.to_string(), StorageClass::Private);
            self.pointer_pointees
                .insert(name.to_string(), self.resolve_type(&pointee)?);
            self.gep_provenance.insert(
                name.to_string(),
                GepProvenance {
                    root: storage,
                    addrspace,
                    source_ty: array_ty,
                    indices: vec![TypedValue {
                        ty: LlType::Int(32),
                        value: LlValue::Int(0),
                    }],
                    root_indices: None,
                    root_is_indexed_container: true,
                },
            );
            if !self.pointer_phi_values.is_empty() && !self.pointer_nullness.contains_key(name) {
                let is_null = self.const_bool(false)?;
                self.record_pointer_nullness(name.to_string(), is_null);
            }
            return Ok(true);
        }
        // A cross-coordinate AIR imageblock is shared tile memory. APV supplies an explicit extent;
        // ordinary compute AIR supplies its row stride through `[[threads_per_threadgroup]]`. Both
        // forms allocate one complete metadata-typed cell per coordinate.
        let (cell_count, width_id) = if let Some([width, height]) = self.ir.imageblock_dimensions {
            let cell_count = width
                .checked_mul(height)
                .ok_or_else(|| "native emitter: imageblock cell count overflows".to_string())?;
            (cell_count, self.const_uint(width)?)
        } else {
            (
                crate::native::imageblock::CELL_CAPACITY,
                self.emit_imageblock_threadgroup_width(instructions)?,
            )
        };
        let array = LlType::Array(Box::new(pointee.clone()), cell_count);
        let storage = if let Some((storage, _)) = self.imageblock_data_scratch.clone() {
            storage
        } else {
            let array_ptr_ty = self.ptr_type_id(StorageClass::Workgroup, &array)?;
            let storage = self.fresh();
            self.module.types_global_values.push(Self::inst(
                Op::Variable,
                Some(array_ptr_ty),
                Some(storage),
                vec![Operand::StorageClass(StorageClass::Workgroup)],
            ));
            self.imageblock_data_scratch = Some((storage, pointee.clone()));
            storage
        };

        let coordinate = call
            .args
            .first()
            .ok_or_else(|| "native emitter: imageblock_data has no coordinate".to_string())?;
        let coordinate_ty = self.resolve_type(&coordinate.ty)?;
        let LlType::Vector(component, lanes) = coordinate_ty else {
            return Err("native emitter: imageblock_data coordinate is not a vector".into());
        };
        let component = self.resolve_type(&component)?;
        let LlType::Int(component_bits) = component else {
            return Err("native emitter: imageblock_data coordinate is not integer".into());
        };
        if lanes < 2 {
            return Err(
                "native emitter: imageblock_data coordinate has fewer than two lanes".into(),
            );
        }
        let coordinate_id = self.value_id_in(&coordinate.value, &coordinate.ty, instructions)?;
        let component_ty = self.type_id(&LlType::Int(component_bits))?;
        let mut components = [0; 2];
        for (lane, component_id) in components.iter_mut().enumerate() {
            let extracted = self.fresh();
            instructions.push(Self::inst(
                Op::CompositeExtract,
                Some(component_ty),
                Some(extracted),
                vec![
                    Operand::IdRef(coordinate_id),
                    Operand::LiteralBit32(lane as u32),
                ],
            ));
            *component_id = if component_bits == 32 {
                extracted
            } else {
                let converted = self.fresh();
                let uint_ty = self.type_id(&LlType::Int(32))?;
                instructions.push(Self::inst(
                    Op::UConvert,
                    Some(uint_ty),
                    Some(converted),
                    vec![Operand::IdRef(extracted)],
                ));
                converted
            };
        }
        let uint_ty = self.type_id(&LlType::Int(32))?;
        let row = self.fresh();
        instructions.push(Self::inst(
            Op::IMul,
            Some(uint_ty),
            Some(row),
            vec![Operand::IdRef(components[1]), Operand::IdRef(width_id)],
        ));
        let index = self.fresh();
        instructions.push(Self::inst(
            Op::IAdd,
            Some(uint_ty),
            Some(index),
            vec![Operand::IdRef(row), Operand::IdRef(components[0])],
        ));
        let slot_ptr_ty = self.ptr_type_id(StorageClass::Workgroup, &pointee)?;
        let pointer = self.fresh();
        instructions.push(Self::inst(
            Op::InBoundsAccessChain,
            Some(slot_ptr_ty),
            Some(pointer),
            vec![Operand::IdRef(storage), Operand::IdRef(index)],
        ));
        self.values
            .insert(name.to_string(), (pointer, LlType::Ptr(addrspace)));
        self.pointer_storage
            .insert(name.to_string(), StorageClass::Workgroup);
        self.pointer_pointees
            .insert(name.to_string(), self.resolve_type(&pointee)?);
        self.gep_provenance.insert(
            name.to_string(),
            GepProvenance {
                root: pointer,
                addrspace,
                source_ty: pointee,
                indices: vec![],
                root_indices: None,
                root_is_indexed_container: false,
            },
        );
        if !self.pointer_phi_values.is_empty() && !self.pointer_nullness.contains_key(name) {
            let is_null = self.const_bool(false)?;
            self.record_pointer_nullness(name.to_string(), is_null);
        }
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_imageblock_threadgroup_width(
        &mut self,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let param_name = self
            .ir
            .imageblock_threads_per_threadgroup_param
            .clone()
            .ok_or_else(|| {
                "native emitter: cross-coordinate imageblock has no threads_per_threadgroup parameter"
                    .to_string()
            })?;
        let (param_id, param_ty) = self.values.get(&param_name).cloned().ok_or_else(|| {
            format!(
                "native emitter: imageblock threads_per_threadgroup value {param_name} is unavailable"
            )
        })?;
        let component = match self.resolve_type(&param_ty)? {
            LlType::Vector(component, lanes) if lanes > 0 => {
                let component = self.resolve_type(&component)?;
                let component_ty = self.type_id(&component)?;
                let extracted = self.fresh();
                instructions.push(Self::inst(
                    Op::CompositeExtract,
                    Some(component_ty),
                    Some(extracted),
                    vec![Operand::IdRef(param_id), Operand::LiteralBit32(0)],
                ));
                (extracted, component)
            }
            scalar @ LlType::Int(_) => (param_id, scalar),
            other => {
                return Err(format!(
                    "native emitter: threads_per_threadgroup has unsupported imageblock width type {other:?}"
                ));
            }
        };
        let LlType::Int(bits) = component.1 else {
            return Err("native emitter: imageblock row width is not integer".to_string());
        };
        if bits == 32 {
            return Ok(component.0);
        }
        let uint_ty = self.type_id(&LlType::Int(32))?;
        let converted = self.fresh();
        instructions.push(Self::inst(
            Op::UConvert,
            Some(uint_ty),
            Some(converted),
            vec![Operand::IdRef(component.0)],
        ));
        Ok(converted)
    }

    pub(in crate::native::emitter) fn validate_call_args(
        &mut self,
        call: &LlCall,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let callee_params = self
            .ir
            .functions
            .iter()
            .find(|function| function.name == call.callee)
            .map(|function| function.params.clone());
        if let Some(callee_params) = callee_params {
            let args = self.function_call_args_for_params(call, &callee_params)?;
            for (index, arg, (param_name, param_ty)) in args {
                let _ = self.value_id_in(&arg.value, &arg.ty, instructions)?;
                if !matches!(self.resolve_type(&param_ty)?, LlType::Ptr(_)) {
                    continue;
                }
                let Some(pointee) = self.pointer_pointee_for_value(&arg.value)? else {
                    continue;
                };
                let key = (call.callee.clone(), index);
                match self.function_param_pointees.get(&key) {
                    Some(existing) if !types_compatible(existing, &pointee) => {
                        if self.callee_param_accepts_call_pointee(
                            &call.callee,
                            &param_name,
                            &param_ty,
                            &pointee,
                        ) {
                            self.function_param_pointees.insert(key, pointee);
                        }
                    }
                    Some(_) => {}
                    None => {
                        if self.callee_param_accepts_call_pointee(
                            &call.callee,
                            &param_name,
                            &param_ty,
                            &pointee,
                        ) {
                            self.function_param_pointees.insert(key, pointee);
                        }
                    }
                }
            }
        } else {
            for arg in &call.args {
                let _ = self.value_id_in(&arg.value, &arg.ty, instructions)?;
            }
        }
        Ok(())
    }

    pub(in crate::native::emitter) fn function_call_arg_ids(
        &mut self,
        call: &LlCall,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Vec<Word>, String> {
        let callee_params = self
            .ir
            .functions
            .iter()
            .find(|function| function.name == call.callee)
            .map(|function| function.params.clone())
            .unwrap_or_default();
        if callee_params.is_empty() {
            let mut ids = Vec::with_capacity(call.args.len());
            for arg in &call.args {
                ids.push(self.value_id_in(&arg.value, &arg.ty, instructions)?);
            }
            return Ok(ids);
        }
        let args = self.function_call_args_for_params(call, &callee_params)?;
        let mut ids = Vec::with_capacity(args.len());
        for (index, arg, (param_name, param_ty)) in args {
            if let Some(id) = self.raw_workgroup_call_arg_id(
                &call.callee,
                index,
                &param_name,
                &param_ty,
                &arg,
                instructions,
            )? {
                ids.push(id);
                continue;
            }
            if let Some(id) = self.decayed_global_call_arg_id(
                &call.callee,
                index,
                &param_name,
                &param_ty,
                &arg,
                instructions,
            )? {
                ids.push(id);
                continue;
            }
            if let Some(id) = self.raw_device_call_arg_id(
                &call.callee,
                &param_name,
                &param_ty,
                &arg,
                instructions,
            )? {
                ids.push(id);
                continue;
            }
            ids.push(self.value_id_in(&arg.value, &arg.ty, instructions)?);
        }
        Ok(ids)
    }

    fn function_call_args_for_params(
        &self,
        call: &LlCall,
        callee_params: &[(String, LlType)],
    ) -> Result<Vec<(usize, TypedValue, (String, LlType))>, String> {
        if call.args.len() == callee_params.len() {
            return Ok(call
                .args
                .iter()
                .cloned()
                .zip(callee_params.iter().cloned())
                .enumerate()
                .map(|(index, (arg, param))| (index, arg, param))
                .collect());
        }
        if call.args.len() < callee_params.len() {
            return Err(format!(
                "native emitter: call @{} has {} args for {} params",
                call.callee,
                call.args.len(),
                callee_params.len()
            ));
        }

        let mut out = Vec::with_capacity(callee_params.len());
        let mut arg_index = 0usize;
        for (param_index, param) in callee_params.iter().cloned().enumerate() {
            let mut matched = None;
            for (offset, arg) in call.args[arg_index..].iter().enumerate() {
                if self.call_arg_matches_param(arg, &param.1)? {
                    matched = Some((arg_index + offset, arg.clone()));
                    break;
                }
            }
            let Some((matched_index, arg)) = matched else {
                return Err(format!(
                    "native emitter: call @{} could not align arg {} to param {} type {:?}",
                    call.callee, arg_index, param_index, param.1
                ));
            };
            arg_index = matched_index + 1;
            out.push((param_index, arg, param));
        }
        Ok(out)
    }

    fn call_arg_matches_param(&self, arg: &TypedValue, param_ty: &LlType) -> Result<bool, String> {
        let arg_ty = self.resolve_type(&arg.ty)?;
        let param_ty = self.resolve_type(param_ty)?;
        Ok(types_compatible(&arg_ty, &param_ty))
    }

    pub(in crate::native::emitter) fn decayed_global_call_arg_id(
        &mut self,
        callee: &str,
        param_index: usize,
        param_name: &str,
        param_ty: &LlType,
        arg: &TypedValue,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Option<Word>, String> {
        let LlValue::Global(global_name) = &arg.value else {
            return Ok(None);
        };
        let LlType::Ptr(param_addrspace) = self.resolve_type(param_ty)? else {
            return Ok(None);
        };
        let LlType::Ptr(arg_addrspace) = self.resolve_type(&arg.ty)? else {
            return Ok(None);
        };
        if param_addrspace != arg_addrspace {
            return Ok(None);
        }
        let Some(expected) = self.function_param_concrete_pointee(callee, param_index, param_name)
        else {
            return Ok(None);
        };
        let Some(global_pointee) = self.pointer_pointees.get(global_name).cloned() else {
            return Ok(None);
        };
        let expected = self.resolve_type(&expected)?;
        let global_pointee = self.resolve_type(&global_pointee)?;
        let Some(decay_indices) = decayed_global_call_arg_indices(&global_pointee, &expected)
        else {
            return Ok(None);
        };
        let base = self.value_id(&arg.value, &arg.ty)?;
        if self
            .flat_scalar_reinterpret_globals
            .contains_key(global_name)
            && matches!(&global_pointee, LlType::Array(elem, _) if types_compatible(elem, &expected))
        {
            let ptr_ty = self.ptr_type_id(StorageClass::Private, &expected)?;
            let zero = self.const_int(32, 0)?;
            let result = self.fresh();
            instructions.push(Self::inst(
                Op::InBoundsAccessChain,
                Some(ptr_ty),
                Some(result),
                vec![Operand::IdRef(base), Operand::IdRef(zero)],
            ));
            return Ok(Some(result));
        }
        let storage = if arg_addrspace == 3 {
            StorageClass::Workgroup
        } else {
            StorageClass::Private
        };
        let ptr_ty = self.ptr_type_id(storage, &expected)?;
        let result = self.fresh();
        let mut ops = Vec::with_capacity(decay_indices.len() + 1);
        ops.push(Operand::IdRef(base));
        for index in &decay_indices {
            ops.push(Operand::IdRef(self.value_id(&index.value, &index.ty)?));
        }
        instructions.push(Self::inst(
            Op::InBoundsAccessChain,
            Some(ptr_ty),
            Some(result),
            ops,
        ));
        Ok(Some(result))
    }

    /// When a raw, descriptor-backed DEVICE buffer pointer is passed to a callee param that is itself a
    /// raw device buffer (and will be helper-inlined), pass the buffer's ROOT id rather than the value the
    /// arg normally resolves to. A `void*` device buffer reaches a helper through an IDENTITY
    /// `bitcast ptr addrspace(1) %buf to ptr addrspace(1)`; the bitcast result carries a Private byte
    /// placeholder VALUE for any DIRECT in-function access, but if that placeholder is passed as the call
    /// argument, the emitted-graph inliner roots the callee's accesses (loads/stores AND atomics) on
    /// the Private var — demoting the buffer to reads-zero / writes-discarded and dropping its
    /// StorageBuffer binding. Passing the descriptor-backed root instead keeps the real device buffer
    /// live.
    ///
    /// Gated structurally and tightly so it only fires for the genuine case: the arg must be a Local with
    /// a modelable descriptor-backed `raw_offsets` entry whose root is a real device-buffer param. A
    /// constant non-zero cursor is recorded for that callee parameter and reapplied while the helper is
    /// emitted; conflicting call-site cursors and dynamic offsets fail visibly rather than silently
    /// discarding an offset. This lives ONLY on the call-argument path; direct accesses and pointer merges
    /// keep their existing behavior.
    pub(in crate::native::emitter) fn raw_device_call_arg_id(
        &mut self,
        callee: &str,
        param_name: &str,
        param_ty: &LlType,
        arg: &TypedValue,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Option<Word>, String> {
        if !self
            .ir
            .raw_buffer_params
            .contains(&(callee.to_string(), param_name.to_string()))
        {
            return Ok(None);
        }
        let LlType::Ptr(param_addrspace) = self.resolve_type(param_ty)? else {
            return Ok(None);
        };
        if param_addrspace != 1 {
            return Ok(None); // device buffers only; the workgroup case is handled separately
        }
        let LlValue::Local(arg_name) = &arg.value else {
            return Ok(None);
        };
        let Some(raw) = self.raw_offsets.get(arg_name).cloned() else {
            return Ok(None);
        };
        if raw.addrspace != 1
            || !raw.dyn_terms.is_empty()
            || raw.unmodelable
            || raw.device_addr_base.is_some()
        {
            return Ok(None);
        }
        let root = raw.root.clone();
        if !self.param_values.contains(&root) || !self.raw_buffer_params.contains(&root) {
            return Ok(None);
        }
        let key = (callee.to_string(), param_name.to_string());
        if let Some(previous) = self.raw_call_param_offsets.get(&key) {
            if previous.const_off != raw.const_off || previous.addrspace != raw.addrspace {
                return Err(format!(
                    "native emitter: raw helper parameter @{callee} {param_name} is called with \
                     conflicting constant byte offsets {} and {}",
                    previous.const_off, raw.const_off
                ));
            }
        } else {
            let mut parameter_raw = raw;
            parameter_raw.root = param_name.to_string();
            self.raw_call_param_offsets.insert(key, parameter_raw);
        }
        let id = self.value_id_in(&LlValue::Local(root), &arg.ty, instructions)?;
        Ok(Some(id))
    }

    pub(in crate::native::emitter) fn raw_workgroup_call_arg_id(
        &mut self,
        callee: &str,
        param_index: usize,
        param_name: &str,
        param_ty: &LlType,
        arg: &TypedValue,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Option<Word>, String> {
        if !self
            .ir
            .raw_buffer_params
            .contains(&(callee.to_string(), param_name.to_string()))
        {
            return Ok(None);
        }
        let LlType::Ptr(addrspace) = self.resolve_type(param_ty)? else {
            return Ok(None);
        };
        if addrspace != 3 {
            return Ok(None);
        }
        if self
            .concrete_vector_workgroup_raw_param_pointee(callee, param_index, param_name)
            .is_some()
        {
            return Ok(None);
        }
        let LlType::Ptr(arg_addrspace) = self.resolve_type(&arg.ty)? else {
            return Ok(None);
        };
        if arg_addrspace != 3 {
            return Ok(None);
        }
        let storage = self.pointer_storage_for(&arg.value, arg_addrspace)?;
        if storage != StorageClass::Workgroup {
            return Ok(None);
        }
        let raw_ty = raw_workgroup_array_type();
        let arg_id = self.value_id_in(&arg.value, &arg.ty, instructions)?;
        if self
            .pointer_pointee_for_value(&arg.value)?
            .is_some_and(|pointee| types_compatible(&pointee, &raw_ty))
        {
            return Ok(Some(arg_id));
        }
        // Reinterpreting the typed Workgroup argument pointer to the raw word-array view would require
        // an `OpBitcast` on a logical pointer — illegal under Logical addressing (the module emits no
        // VariablePointers/PhysicalStorageBuffer for Workgroup). Such a bitcast is never part of a valid
        // module, so rather than emit it (a guaranteed spirv-val reject), surface a pointer-typing emit
        // error that routes to the failure-triggered raw retry (`is_pointer_typing_emit_error` ->
        // all-buffers-raw-with-workgroup), which models the whole callee buffer raw without a pointer
        // bitcast. Floor-safe: a banked module never contains this illegal bitcast, so it never reaches
        // here; a module that did was already failing.
        Err(format!(
            "native emitter: cannot reinterpret workgroup pointer arg {param_name} to raw word view \
             without a logical-pointer bitcast (callee {callee})"
        ))
    }

    pub(in crate::native::emitter) fn shuffled_lane_id(
        &mut self,
        a: &TypedValue,
        b: &TypedValue,
        lane: u32,
        elem: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        if lane == u32::MAX {
            return self.undef_id(elem);
        }
        let a_lanes = self.vector_lane_count(&a.ty)?;
        let (source, source_lane) = if lane < a_lanes {
            (a, lane)
        } else {
            (b, lane - a_lanes)
        };
        if self.one_lane_vector_elem(&source.ty)?.is_some() {
            if source_lane != 0 {
                return Err(format!(
                    "native emitter: one-lane shufflevector source index {source_lane} is out of range"
                ));
            }
            return self.value_id_in(&source.value, &source.ty, instructions);
        }
        let source_id = self.value_id_in(&source.value, &source.ty, instructions)?;
        let result_type = self.type_id(elem)?;
        let result = self.fresh();
        instructions.push(Self::inst(
            Op::CompositeExtract,
            Some(result_type),
            Some(result),
            vec![
                Operand::IdRef(source_id),
                Operand::LiteralBit32(source_lane),
            ],
        ));
        Ok(result)
    }
}

fn decayed_global_call_arg_indices(
    global_pointee: &LlType,
    expected: &LlType,
) -> Option<Vec<TypedValue>> {
    let zero = TypedValue {
        ty: LlType::Int(32),
        value: LlValue::Int(0),
    };
    match global_pointee {
        LlType::Array(elem, _) if types_compatible(elem, expected) => Some(vec![zero]),
        LlType::Struct(fields) => {
            let first = fields.first()?;
            if types_compatible(first, expected) {
                Some(vec![zero])
            } else if matches!(first, LlType::Array(elem, _) if types_compatible(elem, expected)) {
                Some(vec![zero.clone(), zero])
            } else {
                None
            }
        }
        _ => None,
    }
}
