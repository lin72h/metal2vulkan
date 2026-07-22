//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    pub(in crate::native::emitter) fn materialize_reserved_raw_word_index(
        &mut self,
        name: &str,
        raw: &RawBufferOffset,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        self.materialize_raw_word_index(name, raw, false, instructions)
    }

    pub(in crate::native::emitter) fn materialize_raw_word_index(
        &mut self,
        name: &str,
        raw: &RawBufferOffset,
        force: bool,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let word_name = raw_word_index_name(name);
        if !force && !self.values.contains_key(&word_name) {
            return Ok(());
        }
        if !self.raw_pointer_word_aligned(raw) {
            return Err(format!(
                "native emitter: raw pointer phi offset for {name} is not word-aligned: {raw:?}"
            ));
        }
        let value = self.emit_raw_word_index(raw, 0, instructions)?;
        let result_ty = self.type_id(&LlType::Int(32))?;
        let result = self.result_id(&word_name, &LlType::Int(32))?;
        if value != result {
            instructions.push(Self::inst(
                Op::CopyObject,
                Some(result_ty),
                Some(result),
                vec![Operand::IdRef(value)],
            ));
        }
        Ok(())
    }

    pub(in crate::native::emitter) fn materialize_reserved_raw_byte_index(
        &mut self,
        name: &str,
        raw: &RawBufferOffset,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        self.materialize_raw_byte_index(name, raw, false, instructions)
    }

    pub(in crate::native::emitter) fn materialize_raw_byte_index(
        &mut self,
        name: &str,
        raw: &RawBufferOffset,
        force: bool,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let byte_name = raw_byte_index_name(name);
        if !force && !self.values.contains_key(&byte_name) {
            return Ok(());
        }
        let value = self.emit_raw_byte_index(raw, 0, instructions)?;
        let result_ty = self.type_id(&LlType::Int(32))?;
        let result = self.result_id(&byte_name, &LlType::Int(32))?;
        if value != result {
            instructions.push(Self::inst(
                Op::CopyObject,
                Some(result_ty),
                Some(result),
                vec![Operand::IdRef(value)],
            ));
        }
        Ok(())
    }

    pub(in crate::native::emitter) fn raw_index_u32(
        &mut self,
        index: &TypedValue,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let ty = self.resolve_type(&index.ty)?;
        let id = self.value_id(&index.value, &index.ty)?;
        match ty {
            LlType::Int(32) => Ok(id),
            LlType::Int(_) => {
                let uint_ty = self.type_id(&LlType::Int(32))?;
                let out = self.fresh();
                instructions.push(Self::inst(
                    Op::UConvert,
                    Some(uint_ty),
                    Some(out),
                    vec![Operand::IdRef(id)],
                ));
                Ok(out)
            }
            other => Err(format!(
                "native emitter: raw dynamic index must be integer, got {other:?}"
            )),
        }
    }

    /// Synthesize a null Private stand-in for a pointer computation the emitter could not model, and
    /// record the callsite that gave up (`site`) for the `METAL2VULKAN_UNMODELED_WHY` diagnostic (see
    /// [`crate::env_vars`]). The `site` label is descriptive only — no behavior depends on it.
    pub(in crate::native::emitter) fn emit_private_zero_pointer_value_at(
        &mut self,
        result: Word,
        pointee: &LlType,
        site: &str,
    ) -> Result<(), String> {
        if crate::env_vars::unmodeled_why() {
            eprintln!(
                "[unmodeled-why] result=%{result} site={site} pointee={:?}",
                self.resolve_type(pointee)
                    .unwrap_or_else(|_| pointee.clone())
            );
        }
        let pointee = function_storage_local_type(&self.resolve_type(pointee)?);
        let ptr_ty = self.ptr_type_id(StorageClass::Private, &pointee)?;
        let init = self.const_null(&pointee)?;
        self.module.types_global_values.push(Self::inst(
            Op::Variable,
            Some(ptr_ty),
            Some(result),
            vec![
                Operand::StorageClass(StorageClass::Private),
                Operand::IdRef(init),
            ],
        ));
        Ok(())
    }

    pub(in crate::native::emitter) fn define_unmodeled_pointer_value(
        &mut self,
        name: &str,
        addrspace: u32,
        pointee: &LlType,
    ) -> Result<(), String> {
        let result = self.result_id(name, &LlType::Ptr(addrspace))?;
        self.emit_private_zero_pointer_value_at(
            result,
            pointee,
            &format!("define_unmodeled_pointer name={name}"),
        )?;
        self.pointer_storage
            .insert(name.to_string(), StorageClass::Private);
        self.pointer_pointees
            .insert(name.to_string(), self.resolve_type(pointee)?);
        self.unmodeled_pointers.insert(name.to_string());
        if !self.pointer_phi_values.is_empty() && !self.pointer_nullness.contains_key(name) {
            let is_null = self.const_bool(false)?;
            self.record_pointer_nullness(name.to_string(), is_null);
        }
        Ok(())
    }

    pub(in crate::native::emitter) fn define_unmodeled_byte_pointer_value(
        &mut self,
        name: &str,
        addrspace: u32,
    ) -> Result<(), String> {
        let result = self.result_id(name, &LlType::Ptr(addrspace))?;
        self.emit_private_zero_pointer_value_at(
            result,
            &LlType::Int(8),
            &format!("define_unmodeled_byte_pointer name={name}"),
        )?;
        self.pointer_storage
            .insert(name.to_string(), StorageClass::Private);
        self.unmodeled_pointers.insert(name.to_string());
        if !self.pointer_phi_values.is_empty() && !self.pointer_nullness.contains_key(name) {
            let is_null = self.const_bool(false)?;
            self.record_pointer_nullness(name.to_string(), is_null);
        }
        Ok(())
    }

    pub(in crate::native::emitter) fn emit_unmodeled_byte_pointer_copy(
        &mut self,
        name: &str,
        result: Word,
        addrspace: u32,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let placeholder_name = format!("{name}.metal2vulkan.unmodeled_byte_pointer");
        self.define_unmodeled_byte_pointer_value(&placeholder_name, addrspace)?;
        let placeholder = self.value_id(
            &LlValue::Local(placeholder_name.clone()),
            &LlType::Ptr(addrspace),
        )?;
        let result_type = self.ptr_type_id(StorageClass::Private, &LlType::Int(8))?;
        instructions.push(Self::inst(
            Op::CopyObject,
            Some(result_type),
            Some(result),
            vec![Operand::IdRef(placeholder)],
        ));
        self.pointer_storage
            .insert(name.to_string(), StorageClass::Private);
        self.unmodeled_pointers.insert(name.to_string());
        if let Some(is_null) = self.pointer_nullness.get(&placeholder_name).copied() {
            self.record_pointer_nullness(name.to_string(), is_null);
        }
        Ok(())
    }

    /// The "Raw" native layout rule; delegates to the shared oracle (`crate::layout`, refactor S4),
    /// threading `resolve_type` (this emitter's fallible named-type expansion) as its `resolve`.
    pub(in crate::native::emitter) fn raw_type_size_align(
        &self,
        ty: &LlType,
    ) -> Result<(u64, u64), String> {
        crate::layout::raw_size_align(ty, &|t| self.resolve_type(t))
    }

    /// Assemble a bitcastable scalar `T` (16/32/64-bit) from a concrete `uchar` (byte) pointer that
    /// addresses the low byte of `T`. Used when a `getelementptr T, ptr %bytebase, %i` re-typed a byte
    /// pointer to a wider scalar (see [`Self::emit_byte_view_scalar_gep`]): SPIR-V logical addressing
    /// cannot retype the pointer, so read `sizeof(T)` consecutive bytes (`OpPtrAccessChain` element k),
    /// zero-extend each, and pack them little-endian into a `uint`/`ulong`, then `OpBitcast` to `T`.
    /// Structural: keyed purely on the pointer's `Int(8)` pointee and a bitcastable NON-vector scalar
    /// result, never a name. Returns `false` for result types it does not assemble so the caller keeps
    /// its existing fallback.
    pub(in crate::native::emitter) fn emit_scalar_load_from_byte_pointer(
        &mut self,
        result: Word,
        result_ty: &LlType,
        storage: StorageClass,
        base_ptr: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        // Element pointers are `OpPtrAccessChain`s, illegal on a Private/Function stack alias. Decline
        // there so the caller keeps its existing lowering rather than emit invalid SPIR-V. (Device
        // buffer roots may carry a stale `UniformConstant` class that the interface passes canonicalize
        // to StorageBuffer, so those are kept — matching `emit_byte_view_scalar_gep`'s own guard.)
        if matches!(storage, StorageClass::Private | StorageClass::Function) {
            return Ok(false);
        }
        let ty = self.resolve_type(result_ty)?;
        // A vector result assembles each lane from consecutive `sizeof(elem)`-byte spans of the byte
        // pointer, then `OpCompositeConstruct`s the vector (same little-endian byte order as a native
        // vector load). Structural: a byte-pointer load of an `N x scalar`.
        if let LlType::Vector(elem, lanes) = &ty {
            let elem = self.resolve_type(elem)?;
            let (elem_size, _) = self.raw_type_size_align(&elem)?;
            let ptr_ty = self.ptr_type_id(storage, &LlType::Int(8))?;
            let mut lane_ids = Vec::with_capacity(*lanes as usize);
            for lane in 0..*lanes {
                let lane_ptr = if lane == 0 {
                    base_ptr
                } else {
                    let p = self.fresh();
                    let idx = self.const_uint(lane * elem_size as u32)?;
                    instructions.push(Self::inst(
                        Op::PtrAccessChain,
                        Some(ptr_ty),
                        Some(p),
                        vec![Operand::IdRef(base_ptr), Operand::IdRef(idx)],
                    ));
                    p
                };
                let lane_id = self.fresh();
                if !self.emit_scalar_load_from_byte_pointer(
                    lane_id,
                    &elem,
                    storage,
                    lane_ptr,
                    instructions,
                )? {
                    return Ok(false);
                }
                lane_ids.push(lane_id);
            }
            let result_type = self.type_id(&ty)?;
            instructions.push(Self::inst(
                Op::CompositeConstruct,
                Some(result_type),
                Some(result),
                lane_ids.into_iter().map(Operand::IdRef).collect(),
            ));
            return Ok(true);
        }
        if !matches!(
            ty,
            LlType::Float | LlType::Half | LlType::BFloat | LlType::Int(_)
        ) {
            return Ok(false);
        }
        let Some(bits) = bitcast_width(&ty) else {
            return Ok(false);
        };
        if !(bits == 16 || bits == 32 || bits == 64) {
            return Ok(false);
        }
        let nbytes = bits / 8;
        let uchar_ty = self.type_id(&LlType::Int(8))?;
        let ptr_ty = self.ptr_type_id(storage, &LlType::Int(8))?;
        // Accumulate in the integer width we bitcast from (u32 for <=32-bit results, u64 for 64-bit).
        let acc_ty_ll = if bits <= 32 {
            LlType::Int(32)
        } else {
            LlType::Int(64)
        };
        let acc_ty = self.type_id(&acc_ty_ll)?;
        let mut acc: Option<Word> = None;
        for k in 0..nbytes {
            let byte_ptr = if k == 0 {
                base_ptr
            } else {
                let p = self.fresh();
                let idx = self.const_uint(k)?;
                instructions.push(Self::inst(
                    Op::PtrAccessChain,
                    Some(ptr_ty),
                    Some(p),
                    vec![Operand::IdRef(base_ptr), Operand::IdRef(idx)],
                ));
                p
            };
            let byte = self.fresh();
            instructions.push(Self::inst(
                Op::Load,
                Some(uchar_ty),
                Some(byte),
                vec![Operand::IdRef(byte_ptr)],
            ));
            let widened = self.fresh();
            instructions.push(Self::inst(
                Op::UConvert,
                Some(acc_ty),
                Some(widened),
                vec![Operand::IdRef(byte)],
            ));
            let shifted = if k == 0 {
                widened
            } else {
                let s = self.const_uint(k * 8)?;
                let sh = self.fresh();
                instructions.push(Self::inst(
                    Op::ShiftLeftLogical,
                    Some(acc_ty),
                    Some(sh),
                    vec![Operand::IdRef(widened), Operand::IdRef(s)],
                ));
                sh
            };
            acc = Some(match acc {
                None => shifted,
                Some(a) => {
                    let o = self.fresh();
                    instructions.push(Self::inst(
                        Op::BitwiseOr,
                        Some(acc_ty),
                        Some(o),
                        vec![Operand::IdRef(a), Operand::IdRef(shifted)],
                    ));
                    o
                }
            });
        }
        let word = acc.ok_or_else(|| {
            "native emitter: byte-pointer scalar load produced no accumulator \
             (nbytes must be >= 1 for a 16/32/64-bit scalar)"
                .to_string()
        })?;
        let result_type = self.type_id(&ty)?;
        let op = if ty == acc_ty_ll {
            Op::CopyObject
        } else {
            Op::Bitcast
        };
        instructions.push(Self::inst(
            op,
            Some(result_type),
            Some(result),
            vec![Operand::IdRef(word)],
        ));
        Ok(true)
    }

    /// Load a scalar through a byte-offset view of Private scalar scratch without retyping the
    /// logical pointer. The recorded provenance roots the byte view at a pointer whose pointee is
    /// already the requested scalar type; convert the constant byte offset to a scalar element index
    /// and keep `OpPtrAccessChain` typed as that scalar. This is the Private analogue of byte
    /// assembly for buffer/workgroup storage, where loading individual `uchar`s is legal.
    pub(in crate::native::emitter) fn emit_private_scalar_load_from_byte_pointer(
        &mut self,
        result: Word,
        result_ty: &LlType,
        pointer: &LlValue,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let ty = self.resolve_type(result_ty)?;
        if !matches!(
            ty,
            LlType::Float | LlType::Half | LlType::BFloat | LlType::Int(_)
        ) {
            return Ok(false);
        }
        let LlValue::Local(pointer_name) = pointer else {
            return Ok(false);
        };
        let Some(provenance) = self.gep_provenance.get(pointer_name).cloned() else {
            return Ok(false);
        };
        if provenance.source_ty != LlType::Int(8)
            || !provenance.root_is_indexed_container
            || provenance.indices.len() != 1
        {
            return Ok(false);
        }
        let Some(byte_offset) = const_index_i64(&provenance.indices[0]) else {
            return Ok(false);
        };
        if byte_offset < 0 {
            return Ok(false);
        }
        let (size, _) = self.raw_type_size_align(&ty)?;
        if size == 0 || !(byte_offset as u64).is_multiple_of(size) {
            return Ok(false);
        }
        let root_has_matching_pointee = self.values.iter().any(|(name, (id, value_ty))| {
            *id == provenance.root
                && matches!(value_ty, LlType::Ptr(_))
                && self
                    .pointer_pointees
                    .get(name)
                    .is_some_and(|pointee| self.resolve_type(pointee).is_ok_and(|p| p == ty))
        });
        if !root_has_matching_pointee {
            return Ok(false);
        }
        let element_index = byte_offset as u64 / size;
        let ptr = if element_index == 0 {
            provenance.root
        } else {
            let ptr_ty = self.ptr_type_id(StorageClass::Private, &ty)?;
            let ptr = self.fresh();
            let index = self.const_uint(u32::try_from(element_index).map_err(|_| {
                format!(
                    "native emitter: private byte-view element index {element_index} overflows u32"
                )
            })?)?;
            instructions.push(Self::inst(
                Op::PtrAccessChain,
                Some(ptr_ty),
                Some(ptr),
                vec![Operand::IdRef(provenance.root), Operand::IdRef(index)],
            ));
            ptr
        };
        let result_type = self.type_id(&ty)?;
        instructions.push(Self::inst(
            Op::Load,
            Some(result_type),
            Some(result),
            vec![Operand::IdRef(ptr)],
        ));
        Ok(true)
    }

    /// Materialize the two words of a direct buffer parameter's runtime device address as marker
    /// values. The interface pass replaces these markers with loads from its reflected address-table
    /// sidecar once AIR metadata maps the entry-parameter ordinal to a Metal buffer location.
    pub(in crate::native::emitter) fn emit_direct_buffer_address_payload(
        &mut self,
        name: &str,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(Word, Word), String> {
        let param_index = self.direct_param_indices.get(name).copied().ok_or_else(|| {
            format!("native emitter: direct pointer payload source {name} is not an entry parameter")
        })?;
        let uint_ty = self.type_id(&LlType::Int(32))?;
        let zero = self.const_uint(0)?;
        let mut words = [0; 2];
        for (component, word) in words.iter_mut().enumerate() {
            let id = self.fresh();
            instructions.push(Self::inst(
                Op::CopyObject,
                Some(uint_ty),
                Some(id),
                vec![Operand::IdRef(zero)],
            ));
            self.emit_sidecar
                .buffer_address_words
                .push(crate::emit_sidecar::BufferAddressWord {
                    id,
                    param_index,
                    component: component as u32,
                });
            *word = id;
        }
        Ok((words[0], words[1]))
    }

    pub(in crate::native::emitter) fn raw_struct_member(
        &self,
        fields: &[LlType],
        index: u64,
    ) -> Result<(u64, LlType), String> {
        let mut off = 0;
        for (field_index, field) in fields.iter().enumerate() {
            let (size, align) = self.raw_type_size_align(field)?;
            off = round_up_u64(off, align);
            if field_index as u64 == index {
                return Ok((off, self.resolve_type(field)?));
            }
            off += size;
        }
        Err(format!(
            "native emitter: raw struct member index {index} out of range"
        ))
    }
}
