//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    pub(in crate::native::emitter) fn atomic_i32_scope_for_arg(
        &self,
        ptr_arg: &TypedValue,
    ) -> Result<Scope, String> {
        let storage = if let LlValue::Local(name) = &ptr_arg.value {
            if self.unmodeled_pointers.contains(name) {
                StorageClass::Workgroup
            } else if let Some(raw) = self.raw_offsets.get(name) {
                self.raw_access_storage(raw)?
            } else {
                let LlType::Ptr(addrspace) = self.resolve_type(&ptr_arg.ty)? else {
                    return Err(format!(
                        "native emitter: atomic i32 pointer argument has type {:?}",
                        ptr_arg.ty
                    ));
                };
                self.pointer_storage_for(&ptr_arg.value, addrspace)?
            }
        } else {
            let LlType::Ptr(addrspace) = self.resolve_type(&ptr_arg.ty)? else {
                return Err(format!(
                    "native emitter: atomic i32 pointer argument has type {:?}",
                    ptr_arg.ty
                ));
            };
            self.pointer_storage_for(&ptr_arg.value, addrspace)?
        };
        Ok(if storage == StorageClass::Workgroup {
            Scope::Workgroup
        } else {
            Scope::Device
        })
    }

    pub(in crate::native::emitter) fn atomic_i32_memory_semantics(
        scope: Scope,
        workgroup_semantics: MemorySemantics,
    ) -> MemorySemantics {
        if scope == Scope::Workgroup {
            workgroup_semantics | MemorySemantics::WORKGROUP_MEMORY
        } else {
            MemorySemantics::RELAXED
        }
    }

    pub(in crate::native::emitter) fn atomic_i32_pointer_id(
        &mut self,
        ptr_arg: &TypedValue,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        if let LlValue::IntToPtr {
            source,
            destination,
        } = &ptr_arg.value
        {
            if self.resolve_type(destination)? != LlType::Ptr(3) {
                return Err(format!(
                    "native emitter: atomic inttoptr constant expression targets {destination:?}, expected Workgroup pointer"
                ));
            }
            let address = match source.value {
                LlValue::Int(value) | LlValue::Hex(value) => value,
                LlValue::SignedInt(value) if value >= 0 => value as u64,
                _ => {
                    return Err(
                        "native emitter: Workgroup atomic inttoptr address is not a nonnegative integer literal"
                            .into(),
                    )
                }
            };
            return self.workgroup_i32_pointer_for_address(address);
        }
        if let LlValue::Local(name) = &ptr_arg.value {
            if let Some(raw) = self.raw_offsets.get(name).cloned() {
                return self.emit_raw_word_pointer_for_access(&raw, 0, Some(4), instructions);
            }
        }
        if matches!(&ptr_arg.value, LlValue::Local(name) if self.unmodeled_pointers.contains(name))
        {
            return self.unmodeled_atomic_i32_pointer();
        }
        let ptr = self.value_id_in(&ptr_arg.value, &ptr_arg.ty, instructions)?;
        let Some(pointee) = self.pointer_pointee_for_value(&ptr_arg.value)? else {
            return Ok(ptr);
        };
        let pointee = self.resolve_type(&pointee)?;
        if pointee == LlType::Int(32) {
            return Ok(ptr);
        }
        if pointee == LlType::Float {
            return self.bitcast_pointer_to_atomic_i32(ptr_arg, ptr, instructions);
        }

        let LlType::Struct(fields) = pointee else {
            return Err(format!(
                "native emitter: atomic i32 pointer targets {pointee:?}"
            ));
        };
        let mut field = LlType::Struct(fields);
        let mut depth = 0usize;
        loop {
            field = self.resolve_type(&field)?;
            match field {
                LlType::Int(32) => break,
                LlType::Struct(ref fields) if !fields.is_empty() => {
                    field = fields[0].clone();
                    depth += 1;
                }
                LlType::Array(ref element, count) if count > 0 => {
                    field = element.as_ref().clone();
                    depth += 1;
                }
                _ => {
                    return Err(format!(
                        "native emitter: atomic i32 pointer first-field chain targets {field:?}"
                    ));
                }
            }
        }
        let LlType::Ptr(addrspace) = self.resolve_type(&ptr_arg.ty)? else {
            return Err(format!(
                "native emitter: atomic i32 pointer argument has type {:?}",
                ptr_arg.ty
            ));
        };
        let ptr_type = self.ptr_type_id(llvm_pointer_storage(addrspace)?, &LlType::Int(32))?;
        let zero = self.const_uint(0)?;
        let result = self.fresh();
        instructions.push(Self::inst(
            Op::InBoundsAccessChain,
            Some(ptr_type),
            Some(result),
            std::iter::once(Operand::IdRef(ptr))
                .chain(std::iter::repeat_n(Operand::IdRef(zero), depth))
                .collect(),
        ));
        Ok(result)
    }

    pub(in crate::native::emitter) fn bitcast_pointer_to_atomic_i32(
        &mut self,
        ptr_arg: &TypedValue,
        ptr: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let LlType::Ptr(addrspace) = self.resolve_type(&ptr_arg.ty)? else {
            return Err(format!(
                "native emitter: atomic i32 pointer argument has type {:?}",
                ptr_arg.ty
            ));
        };
        let storage = self.pointer_storage_for(&ptr_arg.value, addrspace)?;
        let ptr_type = self.ptr_type_id(storage, &LlType::Int(32))?;
        let result = self.fresh();
        instructions.push(Self::inst(
            Op::Bitcast,
            Some(ptr_type),
            Some(result),
            vec![Operand::IdRef(ptr)],
        ));
        Ok(result)
    }

    pub(in crate::native::emitter) fn atomic_f32_pointer_id(
        &mut self,
        ptr_arg: &TypedValue,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        if matches!(&ptr_arg.value, LlValue::Local(name) if self.unmodeled_pointers.contains(name))
        {
            return self.unmodeled_atomic_f32_pointer();
        }
        let ptr = self.value_id_in(&ptr_arg.value, &ptr_arg.ty, instructions)?;
        let Some(pointee) = self.pointer_pointee_for_value(&ptr_arg.value)? else {
            return Ok(ptr);
        };
        let pointee = self.resolve_type(&pointee)?;
        if pointee == LlType::Float {
            return Ok(ptr);
        }

        let LlType::Struct(fields) = pointee else {
            return Err(format!(
                "native emitter: atomic f32 pointer targets {pointee:?}"
            ));
        };
        let [field] = fields.as_slice() else {
            return Err(format!(
                "native emitter: atomic f32 pointer targets struct with {} fields",
                fields.len()
            ));
        };
        let field = self.resolve_type(field)?;
        if field != LlType::Float {
            return Err(format!(
                "native emitter: atomic f32 pointer targets struct field {field:?}"
            ));
        }
        let LlType::Ptr(addrspace) = self.resolve_type(&ptr_arg.ty)? else {
            return Err(format!(
                "native emitter: atomic f32 pointer argument has type {:?}",
                ptr_arg.ty
            ));
        };
        let ptr_type = self.ptr_type_id(llvm_pointer_storage(addrspace)?, &LlType::Float)?;
        let zero = self.const_uint(0)?;
        let result = self.fresh();
        instructions.push(Self::inst(
            Op::InBoundsAccessChain,
            Some(ptr_type),
            Some(result),
            vec![Operand::IdRef(ptr), Operand::IdRef(zero)],
        ));
        Ok(result)
    }

    pub(in crate::native::emitter) fn unmodeled_atomic_i32_pointer(
        &mut self,
    ) -> Result<Word, String> {
        let ptr_type = self.ptr_type_id(StorageClass::Workgroup, &LlType::Int(32))?;
        let result = self.fresh();
        self.module.types_global_values.push(Self::inst(
            Op::Variable,
            Some(ptr_type),
            Some(result),
            vec![Operand::StorageClass(StorageClass::Workgroup)],
        ));
        Ok(result)
    }

    fn workgroup_i32_pointer_for_address(&mut self, address: u64) -> Result<Word, String> {
        if let Some(pointer) = self.workgroup_i32_addresses.get(&address) {
            return Ok(*pointer);
        }
        let ptr_type = self.ptr_type_id(StorageClass::Workgroup, &LlType::Int(32))?;
        let pointer = self.fresh();
        self.module.types_global_values.push(Self::inst(
            Op::Variable,
            Some(ptr_type),
            Some(pointer),
            vec![Operand::StorageClass(StorageClass::Workgroup)],
        ));
        self.workgroup_i32_addresses.insert(address, pointer);
        Ok(pointer)
    }

    pub(in crate::native::emitter) fn unmodeled_atomic_f32_pointer(
        &mut self,
    ) -> Result<Word, String> {
        let ptr_type = self.ptr_type_id(StorageClass::Workgroup, &LlType::Float)?;
        let result = self.fresh();
        self.module.types_global_values.push(Self::inst(
            Op::Variable,
            Some(ptr_type),
            Some(result),
            vec![Operand::StorageClass(StorageClass::Workgroup)],
        ));
        Ok(result)
    }
}
