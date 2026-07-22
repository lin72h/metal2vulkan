//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    pub(in crate::native::emitter) fn emit_i32_pair_struct_to_i64_load(
        &mut self,
        result: Word,
        pointee: &LlType,
        result_ty: &LlType,
        ptr: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if !is_i32_pair_struct(pointee) || *result_ty != LlType::Int(64) {
            return Ok(false);
        }
        let struct_type = self.type_id(pointee)?;
        let loaded = self.fresh();
        instructions.push(Self::inst(
            Op::Load,
            Some(struct_type),
            Some(loaded),
            vec![Operand::IdRef(ptr)],
        ));
        let uint = self.type_id(&LlType::Int(32))?;
        let low_word = self.fresh();
        instructions.push(Self::inst(
            Op::CompositeExtract,
            Some(uint),
            Some(low_word),
            vec![Operand::IdRef(loaded), Operand::LiteralBit32(0)],
        ));
        let high_word = self.fresh();
        instructions.push(Self::inst(
            Op::CompositeExtract,
            Some(uint),
            Some(high_word),
            vec![Operand::IdRef(loaded), Operand::LiteralBit32(1)],
        ));
        self.emit_join_i32_words_as_i64(result, low_word, high_word, instructions)?;
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_aggregate_prefix_integer_reinterpret_load(
        &mut self,
        result: Word,
        pointee: &LlType,
        result_ty: &LlType,
        ptr_value: &TypedValue,
        ptr: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if !matches!(
            pointee,
            LlType::Array(_, _) | LlType::Struct(_) | LlType::Vector(_, _)
        ) {
            return Ok(false);
        }
        let LlType::Int(bits) = result_ty else {
            return Ok(false);
        };
        if *bits == 0 || bits % 32 != 0 {
            return Ok(false);
        }
        let byte_count = u64::from(bits / 8);
        let Some(fields) = self.leading_i32_scalar_accesses(pointee, byte_count)? else {
            return Ok(false);
        };

        let storage = match self.resolve_type(&ptr_value.ty)? {
            LlType::Ptr(addrspace) => self.pointer_storage_for(&ptr_value.value, addrspace)?,
            other => {
                return Err(format!(
                    "native emitter: load pointer is not a pointer: {other:?}"
                ))
            }
        };
        let result_type = self.type_id(result_ty)?;
        let mut acc = None;
        for (word_index, (access_path, field_ty)) in fields.into_iter().enumerate() {
            let word = self.emit_i32_scalar_field_load(
                ptr,
                storage,
                &access_path,
                &field_ty,
                instructions,
            )?;
            let term = if *bits == 32 {
                word
            } else {
                let widened = self.fresh();
                instructions.push(Self::inst(
                    Op::UConvert,
                    Some(result_type),
                    Some(widened),
                    vec![Operand::IdRef(word)],
                ));
                if word_index == 0 {
                    widened
                } else {
                    let shifted = self.fresh();
                    let shift = self.const_signed_int(*bits, (word_index as i64) * 32)?;
                    instructions.push(Self::inst(
                        Op::ShiftLeftLogical,
                        Some(result_type),
                        Some(shifted),
                        vec![Operand::IdRef(widened), Operand::IdRef(shift)],
                    ));
                    shifted
                }
            };
            acc = Some(if let Some(prev) = acc {
                let combined = if word_index + 1 == (*bits / 32) as usize {
                    result
                } else {
                    self.fresh()
                };
                instructions.push(Self::inst(
                    Op::BitwiseOr,
                    Some(result_type),
                    Some(combined),
                    vec![Operand::IdRef(prev), Operand::IdRef(term)],
                ));
                combined
            } else {
                term
            });
        }
        if acc != Some(result) {
            instructions.push(Self::inst(
                Op::CopyObject,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(acc.ok_or_else(|| {
                    "native emitter: aggregate-prefix reinterpret load produced no accumulator"
                        .to_string()
                })?)],
            ));
        }
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_pointer_to_local_field_store(
        &mut self,
        object: &TypedValue,
        ptr: &TypedValue,
        pointee: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if *pointee != LlType::Int(64) || !matches!(self.resolve_type(&object.ty)?, LlType::Ptr(_))
        {
            return Ok(false);
        }
        let Some(key) = self.local_pointer_field_key(ptr)? else {
            return Ok(false);
        };

        let ptr_id = self.value_id(&ptr.value, &ptr.ty)?;
        let object_id = self.value_id(&object.value, &object.ty)?;
        let zero_ty = self.type_id(&LlType::Int(64))?;
        let zero = self.fresh();
        self.module.types_global_values.push(Self::inst(
            Op::ConstantNull,
            Some(zero_ty),
            Some(zero),
            vec![],
        ));
        self.emit_sidecar.local_pointer_field_stores.push(
            crate::emit_sidecar::LocalPointerFieldStore {
                id: zero,
                source: object_id,
            },
        );
        instructions.push(Self::inst(
            Op::Store,
            None,
            None,
            vec![Operand::IdRef(ptr_id), Operand::IdRef(zero)],
        ));
        self.local_pointer_fields.insert(key, object.clone());
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_pointer_from_local_field_load(
        &mut self,
        result_name: &str,
        result: Word,
        result_ty: &LlType,
        ptr: &TypedValue,
        pointee: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let LlType::Ptr(_addrspace) = result_ty else {
            return Ok(false);
        };
        if *pointee != LlType::Int(64) {
            return Ok(false);
        }
        let Some(key) = self.local_pointer_field_key(ptr)? else {
            return Ok(false);
        };
        self.emit_pointer_from_local_field_key(result_name, result, result_ty, &key, instructions)?;
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_pointer_from_local_field_key(
        &mut self,
        result_name: &str,
        result: Word,
        result_ty: &LlType,
        key: &LocalPointerField,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let LlType::Ptr(addrspace) = result_ty else {
            return Err(format!(
                "native emitter: local pointer field loaded as non-pointer {result_ty:?}"
            ));
        };
        let stored = (!self
            .ir
            .preinlined_helper_pointer_loads
            .contains(result_name))
        .then(|| self.local_pointer_fields.get(key).cloned())
        .flatten();
        let Some(stored) = stored else {
            self.emit_unmodeled_byte_pointer_copy(result_name, result, *addrspace, instructions)?;
            self.emit_sidecar.local_pointer_field_loads.push(
                crate::emit_sidecar::LocalPointerFieldLoad {
                    id: result,
                    root: key.root,
                    indices: key.indices.clone(),
                },
            );
            return Ok(());
        };

        let storage = self.pointer_storage_for(&stored.value, *addrspace)?;
        let pointee = self
            .pointer_pointee_for_value(&stored.value)?
            .unwrap_or(LlType::Int(8));
        let result_type = self.ptr_type_id(storage, &pointee)?;
        let source = self.value_id_in(&stored.value, &stored.ty, instructions)?;
        instructions.push(Self::inst(
            Op::CopyObject,
            Some(result_type),
            Some(result),
            vec![Operand::IdRef(source)],
        ));
        self.pointer_storage
            .insert(result_name.to_string(), storage);
        self.pointer_pointees
            .insert(result_name.to_string(), pointee);
        if let LlValue::Local(source_name) = &stored.value {
            if let Some(raw) = self.raw_offsets.get(source_name).cloned() {
                self.raw_offsets.insert(result_name.to_string(), raw);
            }
            if let Some(nullness) = self.pointer_nullness.get(source_name).copied() {
                self.record_pointer_nullness(result_name.to_string(), nullness);
            }
        }
        Ok(())
    }

    pub(in crate::native::emitter) fn emit_pointer_from_local_dynamic_field_load(
        &mut self,
        result_name: &str,
        result: Word,
        result_ty: &LlType,
        ptr: &TypedValue,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let LlType::Ptr(_) = result_ty else {
            return Ok(false);
        };
        let Some((root, prefix, index, suffix)) = self.local_pointer_dynamic_field_index(ptr)?
        else {
            return Ok(false);
        };
        let values = self
            .local_pointer_fields
            .iter()
            .filter_map(|(key, value)| {
                let suffix_start = prefix.len() + 1;
                (key.root == root
                    && key.indices.len() == suffix_start + suffix.len()
                    && key.indices.starts_with(&prefix)
                    && &key.indices[suffix_start..] == suffix.as_slice())
                .then(|| (key.indices[prefix.len()], value.clone()))
            })
            .collect::<Vec<_>>();
        if values.is_empty() {
            let LlType::Ptr(addrspace) = result_ty else {
                return Ok(false);
            };
            self.emit_unmodeled_byte_pointer_copy(result_name, result, *addrspace, instructions)?;
            self.emit_sidecar.local_pointer_dynamic_field_loads.push(
                crate::emit_sidecar::LocalPointerDynamicFieldLoad {
                    id: result,
                    root,
                    prefix,
                    suffix,
                },
            );
            return Ok(true);
        }
        let mut values = values;
        values.sort_by_key(|(idx, _)| *idx);

        let value_refs = values
            .iter()
            .map(|(_, value)| &value.value)
            .collect::<Vec<_>>();
        let Some(pointer_meta) = self.pointer_merge_meta(&value_refs, result_ty)? else {
            return Ok(false);
        };
        if !matches!(
            pointer_meta.storage,
            StorageClass::StorageBuffer | StorageClass::UniformConstant | StorageClass::Workgroup
        ) {
            return Ok(false);
        }

        let index_ty = self.resolve_type(&index.ty)?;
        let LlType::Int(index_bits) = index_ty else {
            return Ok(false);
        };
        let index_id = self.value_id_in(&index.value, &index.ty, instructions)?;
        let result_type = self.pointer_aware_type_id(result_ty, Some(&pointer_meta))?;
        let bool_type = self.type_id(&LlType::Bool)?;

        let mut current = self.value_id_in(&values[0].1.value, &values[0].1.ty, instructions)?;
        if values.len() == 1 {
            instructions.push(Self::inst(
                Op::CopyObject,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(current)],
            ));
        } else {
            for (entry_idx, (stored_index, stored_value)) in values.iter().enumerate().skip(1) {
                let stored_id =
                    self.value_id_in(&stored_value.value, &stored_value.ty, instructions)?;
                let index_const = self.const_int(index_bits, *stored_index as u64)?;
                let is_entry = self.fresh();
                instructions.push(Self::inst(
                    Op::IEqual,
                    Some(bool_type),
                    Some(is_entry),
                    vec![Operand::IdRef(index_id), Operand::IdRef(index_const)],
                ));
                let selected = if entry_idx + 1 == values.len() {
                    result
                } else {
                    self.fresh()
                };
                instructions.push(Self::inst(
                    Op::Select,
                    Some(result_type),
                    Some(selected),
                    vec![
                        Operand::IdRef(is_entry),
                        Operand::IdRef(stored_id),
                        Operand::IdRef(current),
                    ],
                ));
                current = selected;
            }
        }

        self.record_pointer_meta(result_name.to_string(), pointer_meta);
        self.dynamic_pointer_tables.insert(
            result_name.to_string(),
            DynamicPointerTable {
                index: index.clone(),
                entries: values,
            },
        );
        if !self.pointer_phi_values.is_empty() && !self.pointer_nullness.contains_key(result_name) {
            let is_null = self.const_bool(false)?;
            self.record_pointer_nullness(result_name.to_string(), is_null);
        }
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_dynamic_pointer_table_gep(
        &mut self,
        name: &str,
        addrspace: u32,
        table: &DynamicPointerTable,
        source_ty: &LlType,
        indices: &[TypedValue],
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if table.entries.is_empty() {
            return Ok(false);
        }
        let pointee = gep_pointee(source_ty, indices)?;
        let mut storage = None;
        for (_, entry) in &table.entries {
            let LlValue::Local(entry_name) = &entry.value else {
                return Ok(false);
            };
            if !self.param_values.contains(entry_name) {
                return Ok(false);
            }
            let entry_storage = self.pointer_storage_for(&entry.value, addrspace)?;
            if !matches!(
                entry_storage,
                StorageClass::StorageBuffer
                    | StorageClass::UniformConstant
                    | StorageClass::Workgroup
            ) {
                return Ok(false);
            }
            match storage {
                Some(existing) if existing != entry_storage => return Ok(false),
                Some(_) => {}
                None => storage = Some(entry_storage),
            }
            let Some(entry_pointee) = self.pointer_pointee_for_value(&entry.value)? else {
                return Ok(false);
            };
            if !types_compatible(&self.resolve_type(&entry_pointee)?, source_ty) {
                return Ok(false);
            }
        }
        let Some(storage) = storage else {
            return Ok(false);
        };

        let index_ty = self.resolve_type(&table.index.ty)?;
        let LlType::Int(index_bits) = index_ty else {
            return Ok(false);
        };
        let index_id = self.value_id_in(&table.index.value, &table.index.ty, instructions)?;
        let result_type = self.ptr_type_id(storage, &pointee)?;
        let bool_type = self.type_id(&LlType::Bool)?;
        let result = self.result_id(name, &LlType::Ptr(addrspace))?;

        let mut current = self.emit_dynamic_pointer_table_entry_gep(
            result_type,
            &table.entries[0].1,
            indices,
            instructions,
        )?;
        if table.entries.len() == 1 {
            instructions.push(Self::inst(
                Op::CopyObject,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(current)],
            ));
        } else {
            for (entry_idx, (stored_index, entry)) in table.entries.iter().enumerate().skip(1) {
                let entry_ptr = self.emit_dynamic_pointer_table_entry_gep(
                    result_type,
                    entry,
                    indices,
                    instructions,
                )?;
                let index_const = self.const_int(index_bits, *stored_index as u64)?;
                let is_entry = self.fresh();
                instructions.push(Self::inst(
                    Op::IEqual,
                    Some(bool_type),
                    Some(is_entry),
                    vec![Operand::IdRef(index_id), Operand::IdRef(index_const)],
                ));
                let selected = if entry_idx + 1 == table.entries.len() {
                    result
                } else {
                    self.fresh()
                };
                instructions.push(Self::inst(
                    Op::Select,
                    Some(result_type),
                    Some(selected),
                    vec![
                        Operand::IdRef(is_entry),
                        Operand::IdRef(entry_ptr),
                        Operand::IdRef(current),
                    ],
                ));
                current = selected;
            }
        }

        self.pointer_storage.insert(name.to_string(), storage);
        self.pointer_pointees.insert(name.to_string(), pointee);
        if !self.pointer_phi_values.is_empty() {
            let is_null = self.const_bool(false)?;
            self.record_pointer_nullness(name.to_string(), is_null);
        }
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_dynamic_pointer_table_entry_gep(
        &mut self,
        ptr_type: Word,
        entry: &TypedValue,
        indices: &[TypedValue],
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let base = self.value_id_in(&entry.value, &entry.ty, instructions)?;
        let mut ops = vec![Operand::IdRef(base)];
        for idx in gep_spirv_indices(indices)? {
            ops.push(Operand::IdRef(self.value_id(&idx.value, &idx.ty)?));
        }
        let result = self.fresh();
        instructions.push(Self::inst(
            Op::AccessChain,
            Some(ptr_type),
            Some(result),
            ops,
        ));
        Ok(result)
    }

    pub(in crate::native::emitter) fn emit_i64_to_i32_pair_struct_store(
        &mut self,
        object: &TypedValue,
        ptr: &TypedValue,
        pointee: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let object_ty = self.resolve_type(&object.ty)?;
        if !is_i32_pair_struct(pointee) || object_ty != LlType::Int(64) {
            return Ok(false);
        }
        let ptr_id = self.value_id(&ptr.value, &ptr.ty)?;
        let value = self.value_id_in(&object.value, &object.ty, instructions)?;
        let uint = self.type_id(&LlType::Int(32))?;
        let low_word = self.fresh();
        instructions.push(Self::inst(
            Op::UConvert,
            Some(uint),
            Some(low_word),
            vec![Operand::IdRef(value)],
        ));
        let uint64 = self.type_id(&LlType::Int(64))?;
        let shift = self.const_signed_int(64, 32)?;
        let shifted_high = self.fresh();
        instructions.push(Self::inst(
            Op::ShiftRightLogical,
            Some(uint64),
            Some(shifted_high),
            vec![Operand::IdRef(value), Operand::IdRef(shift)],
        ));
        let high_word = self.fresh();
        instructions.push(Self::inst(
            Op::UConvert,
            Some(uint),
            Some(high_word),
            vec![Operand::IdRef(shifted_high)],
        ));

        let storage = match self.resolve_type(&ptr.ty)? {
            LlType::Ptr(addrspace) => self.pointer_storage_for(&ptr.value, addrspace)?,
            other => {
                return Err(format!(
                    "native emitter: store pointer is not a pointer: {other:?}"
                ))
            }
        };
        self.emit_i32_pair_struct_field_store(ptr_id, storage, 0, low_word, instructions)?;
        self.emit_i32_pair_struct_field_store(ptr_id, storage, 1, high_word, instructions)?;
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_aggregate_prefix_integer_reinterpret_store(
        &mut self,
        object: &TypedValue,
        ptr: &TypedValue,
        object_ty: &LlType,
        pointee: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if !matches!(
            pointee,
            LlType::Array(_, _) | LlType::Struct(_) | LlType::Vector(_, _)
        ) {
            return Ok(false);
        }
        let LlType::Int(bits) = object_ty else {
            return Ok(false);
        };
        if *bits == 0 || bits % 32 != 0 {
            return Ok(false);
        }
        let byte_count = u64::from(bits / 8);
        let Some(fields) = self.leading_i32_scalar_accesses(pointee, byte_count)? else {
            return Ok(false);
        };

        let ptr_id = self.value_id(&ptr.value, &ptr.ty)?;
        let object_id = self.value_id_in(&object.value, &object.ty, instructions)?;
        let storage = match self.resolve_type(&ptr.ty)? {
            LlType::Ptr(addrspace) => self.pointer_storage_for(&ptr.value, addrspace)?,
            other => {
                return Err(format!(
                    "native emitter: store pointer is not a pointer: {other:?}"
                ))
            }
        };
        for (word_index, (access_path, field_ty)) in fields.into_iter().enumerate() {
            let word = if *bits == 32 {
                object_id
            } else {
                let shifted = if word_index == 0 {
                    object_id
                } else {
                    let shifted = self.fresh();
                    let shift = self.const_signed_int(*bits, (word_index as i64) * 32)?;
                    instructions.push(Self::inst(
                        Op::ShiftRightLogical,
                        Some(self.type_id(object_ty)?),
                        Some(shifted),
                        vec![Operand::IdRef(object_id), Operand::IdRef(shift)],
                    ));
                    shifted
                };
                let uint = self.type_id(&LlType::Int(32))?;
                let word = self.fresh();
                instructions.push(Self::inst(
                    Op::UConvert,
                    Some(uint),
                    Some(word),
                    vec![Operand::IdRef(shifted)],
                ));
                word
            };
            self.emit_i32_scalar_field_store(
                ptr_id,
                storage,
                &access_path,
                &field_ty,
                word,
                instructions,
            )?;
        }
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_byte_array_integer_reinterpret_load(
        &mut self,
        result: Word,
        pointee: &LlType,
        result_ty: &LlType,
        ptr_value: &TypedValue,
        ptr: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let LlType::Array(elem, len) = pointee else {
            return Ok(false);
        };
        if elem.as_ref() != &LlType::Int(8) {
            return Ok(false);
        }
        let LlType::Int(bits) = result_ty else {
            return Ok(false);
        };
        if bits % 8 != 0 {
            return Ok(false);
        }
        let byte_count = bits / 8;
        if byte_count == 0 || byte_count > *len {
            return Ok(false);
        }

        let storage = match self.resolve_type(&ptr_value.ty)? {
            LlType::Ptr(addrspace) => self.pointer_storage_for(&ptr_value.value, addrspace)?,
            other => {
                return Err(format!(
                    "native emitter: load pointer is not a pointer: {other:?}"
                ))
            }
        };
        let byte_ty = self.type_id(&LlType::Int(8))?;
        let byte_ptr_ty = self.ptr_type_id(storage, &LlType::Int(8))?;
        let result_ty_id = self.type_id(result_ty)?;
        let mut acc = None;
        for byte in 0..byte_count {
            let byte_ptr = self.fresh();
            instructions.push(Self::inst(
                Op::InBoundsAccessChain,
                Some(byte_ptr_ty),
                Some(byte_ptr),
                vec![Operand::IdRef(ptr), Operand::IdRef(self.const_uint(byte)?)],
            ));
            let loaded = self.fresh();
            instructions.push(Self::inst(
                Op::Load,
                Some(byte_ty),
                Some(loaded),
                vec![Operand::IdRef(byte_ptr)],
            ));
            let widened = if *bits == 8 {
                loaded
            } else {
                let widened = self.fresh();
                instructions.push(Self::inst(
                    Op::UConvert,
                    Some(result_ty_id),
                    Some(widened),
                    vec![Operand::IdRef(loaded)],
                ));
                widened
            };
            let term = if byte == 0 {
                widened
            } else {
                let shift = self.const_signed_int(*bits, i64::from(byte * 8))?;
                let shifted = self.fresh();
                instructions.push(Self::inst(
                    Op::ShiftLeftLogical,
                    Some(result_ty_id),
                    Some(shifted),
                    vec![Operand::IdRef(widened), Operand::IdRef(shift)],
                ));
                shifted
            };
            acc = Some(if let Some(prev) = acc {
                let combined = if byte + 1 == byte_count {
                    result
                } else {
                    self.fresh()
                };
                instructions.push(Self::inst(
                    Op::BitwiseOr,
                    Some(result_ty_id),
                    Some(combined),
                    vec![Operand::IdRef(prev), Operand::IdRef(term)],
                ));
                combined
            } else {
                term
            });
        }
        if acc != Some(result) {
            instructions.push(Self::inst(
                Op::CopyObject,
                Some(result_ty_id),
                Some(result),
                vec![Operand::IdRef(acc.ok_or_else(|| {
                    "native emitter: byte-array reinterpret load produced no accumulator"
                        .to_string()
                })?)],
            ));
        }
        Ok(true)
    }

    pub(in crate::native::emitter) fn local_pointer_field_key(
        &self,
        ptr: &TypedValue,
    ) -> Result<Option<LocalPointerField>, String> {
        let LlValue::Local(name) = &ptr.value else {
            return Ok(None);
        };
        let Some(provenance) = self.gep_provenance.get(name) else {
            return Ok(None);
        };
        let LlType::Ptr(addrspace) = self.resolve_type(&ptr.ty)? else {
            return Err(format!(
                "native emitter: local pointer field key for non-pointer {:?}",
                ptr.ty
            ));
        };
        let storage = self.pointer_storage_for(&ptr.value, addrspace)?;
        if !matches!(storage, StorageClass::Function | StorageClass::Private) {
            return Ok(None);
        }
        let Some(mut indices) = provenance
            .indices
            .iter()
            .map(|idx| const_index(Some(idx)))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        if indices.len() > 1 && indices.first() == Some(&0) {
            indices.remove(0);
        }
        Ok(Some(LocalPointerField {
            root: provenance.root,
            indices,
        }))
    }

    pub(in crate::native::emitter) fn local_pointer_dynamic_field_index(
        &self,
        ptr: &TypedValue,
    ) -> Result<Option<(Word, Vec<u32>, TypedValue, Vec<u32>)>, String> {
        let LlValue::Local(name) = &ptr.value else {
            return Ok(None);
        };
        let Some(provenance) = self.gep_provenance.get(name) else {
            return Ok(None);
        };
        let LlType::Ptr(addrspace) = self.resolve_type(&ptr.ty)? else {
            return Err(format!(
                "native emitter: local pointer dynamic field key for non-pointer {:?}",
                ptr.ty
            ));
        };
        if self.pointer_storage_for(&ptr.value, addrspace)? != StorageClass::Function {
            return Ok(None);
        }
        let mut indices = provenance.indices.clone();
        if indices.len() > 1 && const_index(indices.first()) == Some(0) {
            indices.remove(0);
        }
        let dynamic_positions = indices
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| const_index(Some(value)).is_none().then_some(idx))
            .collect::<Vec<_>>();
        let [dynamic_pos] = dynamic_positions.as_slice() else {
            return Ok(None);
        };
        let prefix = indices[..*dynamic_pos]
            .iter()
            .map(|idx| const_index(Some(idx)))
            .collect::<Option<Vec<_>>>();
        let suffix = indices[*dynamic_pos + 1..]
            .iter()
            .map(|idx| const_index(Some(idx)))
            .collect::<Option<Vec<_>>>();
        let (Some(prefix), Some(suffix)) = (prefix, suffix) else {
            return Ok(None);
        };
        Ok(Some((
            provenance.root,
            prefix,
            indices[*dynamic_pos].clone(),
            suffix,
        )))
    }

    pub(in crate::native::emitter) fn local_alloca_storage_compatible(
        &self,
        original: &LlType,
        candidate: &LlType,
    ) -> bool {
        let original = function_storage_local_type(original);
        let candidate = function_storage_local_type(candidate);
        match (
            self.raw_type_size_align(&original),
            self.raw_type_size_align(&candidate),
        ) {
            (Ok((original_size, _)), Ok((candidate_size, _))) => original_size == candidate_size,
            _ => false,
        }
    }
}
