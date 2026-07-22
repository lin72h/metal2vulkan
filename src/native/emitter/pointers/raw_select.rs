//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    pub(in crate::native::emitter) fn emit_selected_pointer_direct_load(
        &mut self,
        result: Word,
        result_ty: &LlType,
        selected: &SelectedPointer,
        _access_align: Option<u64>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let LlType::Ptr(addrspace) = self.resolve_type(&selected.ty)? else {
            return Err("native emitter: selected load source is not a pointer".to_string());
        };
        let true_is_null = matches!(selected.true_value, LlValue::Zero);
        let false_is_null = matches!(selected.false_value, LlValue::Zero);
        let true_storage = (!true_is_null)
            .then(|| self.pointer_storage_for(&selected.true_value, addrspace))
            .transpose()?;
        let false_storage = (!false_is_null)
            .then(|| self.pointer_storage_for(&selected.false_value, addrspace))
            .transpose()?;
        let Some(storage) = true_storage.or(false_storage) else {
            return Err("native emitter: selected load has no concrete pointer arm".to_string());
        };
        if !matches!(
            storage,
            StorageClass::StorageBuffer | StorageClass::UniformConstant
        ) || true_storage.is_some_and(|true_storage| true_storage != storage)
            || false_storage.is_some_and(|false_storage| false_storage != storage)
        {
            return Err(
                "native emitter: selected direct load requires uniform storage-class arms"
                    .to_string(),
            );
        }
        let pointee = if true_is_null {
            self.pointer_pointee_for_value(&selected.false_value)?
        } else {
            self.pointer_pointee_for_value(&selected.true_value)?
        }
        .ok_or_else(|| "native emitter: selected direct load missing pointee type".to_string())?;
        let result_ty = self.resolve_type(result_ty)?;
        if !types_compatible(&self.resolve_type(&pointee)?, &result_ty) {
            return Err(format!(
                "native emitter: selected direct load type mismatch {pointee:?} vs {result_ty:?}"
            ));
        }
        let result_type = self.type_id(&result_ty)?;
        let true_value = if true_is_null {
            self.const_null(&result_ty)?
        } else {
            let ptr = self.value_id(&selected.true_value, &selected.ty)?;
            let value = self.fresh();
            instructions.push(Self::inst(
                Op::Load,
                Some(result_type),
                Some(value),
                vec![Operand::IdRef(ptr)],
            ));
            value
        };
        let false_value = if false_is_null {
            self.const_null(&result_ty)?
        } else {
            let ptr = self.value_id(&selected.false_value, &selected.ty)?;
            let value = self.fresh();
            instructions.push(Self::inst(
                Op::Load,
                Some(result_type),
                Some(value),
                vec![Operand::IdRef(ptr)],
            ));
            value
        };
        instructions.push(Self::inst(
            Op::Select,
            Some(result_type),
            Some(result),
            vec![
                Operand::IdRef(selected.cond),
                Operand::IdRef(true_value),
                Operand::IdRef(false_value),
            ],
        ));
        Ok(())
    }

    /// Store `object` THROUGH a direct pointer select (`selected_pointers`), the write-side analog of
    /// [`emit_selected_pointer_direct_load`]. A pointer `OpSelect`-then-store is illegal under Logical
    /// addressing when the arms can point into distinct buffers, and there is no value-level "store to
    /// both" — so each arm is updated by a read-modify-write conditional store: load the arm's current
    /// value, `OpSelect` between the new object and that old value under the select condition (so the
    /// SELECTED arm receives the new value and the other arm is written back unchanged), then store. This
    /// is branch-free and valid (each arm is loaded/stored through its own typed pointer). Returns Ok
    /// only when both arms are uniform StorageBuffer/UniformConstant with a pointee matching the stored
    /// type; otherwise errors so the caller routes to the raw retry.
    pub(in crate::native::emitter) fn emit_selected_pointer_direct_store(
        &mut self,
        object_ty: &LlType,
        object_id: Word,
        selected: &SelectedPointer,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let LlType::Ptr(addrspace) = self.resolve_type(&selected.ty)? else {
            return Err("native emitter: selected store target is not a pointer".to_string());
        };
        let true_is_null = matches!(selected.true_value, LlValue::Zero);
        let false_is_null = matches!(selected.false_value, LlValue::Zero);
        let true_storage = (!true_is_null)
            .then(|| self.pointer_storage_for(&selected.true_value, addrspace))
            .transpose()?;
        let false_storage = (!false_is_null)
            .then(|| self.pointer_storage_for(&selected.false_value, addrspace))
            .transpose()?;
        let Some(storage) = true_storage.or(false_storage) else {
            return Err("native emitter: selected store has no concrete pointer arm".to_string());
        };
        if !matches!(
            storage,
            StorageClass::StorageBuffer | StorageClass::UniformConstant
        ) || true_storage.is_some_and(|s| s != storage)
            || false_storage.is_some_and(|s| s != storage)
        {
            return Err(
                "native emitter: cannot store through reinterpreted pointer select (non-uniform arms)"
                    .to_string(),
            );
        }
        let pointee = if true_is_null {
            self.pointer_pointee_for_value(&selected.false_value)?
        } else {
            self.pointer_pointee_for_value(&selected.true_value)?
        }
        .ok_or_else(|| {
            "native emitter: cannot store through reinterpreted pointer select (missing pointee)"
                .to_string()
        })?;
        let object_ty = self.resolve_type(object_ty)?;
        if !types_compatible(&self.resolve_type(&pointee)?, &object_ty) {
            return Err(format!(
                "native emitter: selected store type mismatch {pointee:?} vs {object_ty:?}"
            ));
        }
        let object_type = self.type_id(&object_ty)?;
        // The SELECTED arm receives the new value; the other arm is written back unchanged. `cond` true
        // selects the true arm, so the true arm stores `select(cond, object, old)` and the false arm
        // stores `select(cond, old, object)`.
        for (value, take_object_when_true) in
            [(&selected.true_value, true), (&selected.false_value, false)]
        {
            if matches!(value, LlValue::Zero) {
                continue; // a null arm is UB to store through in the source; skip it.
            }
            let ptr = self.value_id(value, &selected.ty)?;
            let old = self.fresh();
            instructions.push(Self::inst(
                Op::Load,
                Some(object_type),
                Some(old),
                vec![Operand::IdRef(ptr)],
            ));
            let merged = self.fresh();
            let (t, f) = if take_object_when_true {
                (object_id, old)
            } else {
                (old, object_id)
            };
            instructions.push(Self::inst(
                Op::Select,
                Some(object_type),
                Some(merged),
                vec![
                    Operand::IdRef(selected.cond),
                    Operand::IdRef(t),
                    Operand::IdRef(f),
                ],
            ));
            instructions.push(Self::inst(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(ptr), Operand::IdRef(merged)],
            ));
        }
        Ok(())
    }

    pub(in crate::native::emitter) fn materialize_selected_pointer_value(
        &mut self,
        name: &str,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let selected = self
            .selected_pointers
            .get(name)
            .cloned()
            .ok_or_else(|| format!("native emitter: unknown selected pointer {name}"))?;
        let LlType::Ptr(addrspace) = self.resolve_type(&selected.ty)? else {
            return Err(format!(
                "native emitter: selected pointer {name} has non-pointer type {:?}",
                selected.ty
            ));
        };
        let meta = self
            .pointer_meta_for_value(&LlValue::Local(name.to_string()), addrspace)?
            .ok_or_else(|| format!("native emitter: selected pointer {name} missing metadata"))?;
        let result_type = self.pointer_aware_type_id(&selected.ty, Some(&meta))?;
        let result = self.result_id(name, &selected.ty)?;
        let true_id =
            self.pointer_select_arm_id(&selected.true_value, &selected.ty, &meta, instructions)?;
        let false_id =
            self.pointer_select_arm_id(&selected.false_value, &selected.ty, &meta, instructions)?;
        instructions.push(Self::inst(
            Op::Select,
            Some(result_type),
            Some(result),
            vec![
                Operand::IdRef(selected.cond),
                Operand::IdRef(true_id),
                Operand::IdRef(false_id),
            ],
        ));
        Ok(())
    }

    pub(in crate::native::emitter) fn emit_selected_pointer_load(
        &mut self,
        result: Word,
        result_ty: &LlType,
        selected: &SelectedLoadPointer,
        access_align: Option<u64>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let result_ty = self.resolve_type(result_ty)?;
        if selected.load_typed {
            // The arms are already-typed pointers; load each with the load's value type and select.
            let (Some(true_ptr), Some(false_ptr)) = (selected.true_ptr, selected.false_ptr) else {
                return Err(
                    "native emitter: load-typed selected pointer missing an arm".to_string()
                );
            };
            let result_type = self.type_id(&result_ty)?;
            let true_value = self.fresh();
            let false_value = self.fresh();
            instructions.push(Self::inst(
                Op::Load,
                Some(result_type),
                Some(true_value),
                vec![Operand::IdRef(true_ptr)],
            ));
            instructions.push(Self::inst(
                Op::Load,
                Some(result_type),
                Some(false_value),
                vec![Operand::IdRef(false_ptr)],
            ));
            instructions.push(Self::inst(
                Op::Select,
                Some(result_type),
                Some(result),
                vec![
                    Operand::IdRef(selected.cond),
                    Operand::IdRef(true_value),
                    Operand::IdRef(false_value),
                ],
            ));
            return Ok(());
        }
        let pointee = self.resolve_type(&selected.pointee)?;
        if pointee == LlType::Int(8) && selected.true_ptr.is_none() && selected.false_ptr.is_none()
        {
            if let (Some(true_raw), Some(false_raw)) = (&selected.true_raw, &selected.false_raw) {
                let result_type = self.type_id(&result_ty)?;
                let true_value = self.fresh();
                let false_value = self.fresh();
                self.emit_raw_load(true_value, &result_ty, true_raw, access_align, instructions)?;
                self.emit_raw_load(
                    false_value,
                    &result_ty,
                    false_raw,
                    access_align,
                    instructions,
                )?;
                instructions.push(Self::inst(
                    Op::Select,
                    Some(result_type),
                    Some(result),
                    vec![
                        Operand::IdRef(selected.cond),
                        Operand::IdRef(true_value),
                        Operand::IdRef(false_value),
                    ],
                ));
                return Ok(());
            }
            return Err(
                "native emitter: selected i8 pointer load requires raw offsets".to_string(),
            );
        }
        if pointee == LlType::Int(8) && pointee != result_ty {
            let pointee_bits = bitcast_width(&pointee).ok_or_else(|| {
                format!("native emitter: cannot reinterpret selected load from {pointee:?}")
            })?;
            let result_bits = bitcast_width(&result_ty).ok_or_else(|| {
                format!("native emitter: cannot reinterpret selected load to {result_ty:?}")
            })?;
            if pointee_bits > result_bits || result_bits % pointee_bits != 0 {
                return Err(format!(
                    "native emitter: selected pointer load bit width mismatch {pointee:?} ({pointee_bits}) vs {result_ty:?} ({result_bits})"
                ));
            }
            if let (Some(true_raw), Some(false_raw)) = (&selected.true_raw, &selected.false_raw) {
                let result_type = self.type_id(&result_ty)?;
                let true_value = self.fresh();
                let false_value = self.fresh();
                self.emit_raw_load(true_value, &result_ty, true_raw, access_align, instructions)?;
                self.emit_raw_load(
                    false_value,
                    &result_ty,
                    false_raw,
                    access_align,
                    instructions,
                )?;
                instructions.push(Self::inst(
                    Op::Select,
                    Some(result_type),
                    Some(result),
                    vec![
                        Operand::IdRef(selected.cond),
                        Operand::IdRef(true_value),
                        Operand::IdRef(false_value),
                    ],
                ));
                return Ok(());
            }
            return Err(
                "native emitter: selected i8 pointer reinterpret load requires raw offsets"
                    .to_string(),
            );
        }
        if !types_compatible(&pointee, &result_ty) {
            return Err(format!(
                "native emitter: selected pointer load type mismatch {pointee:?} vs {result_ty:?}"
            ));
        }
        let result_type = self.type_id(&result_ty)?;
        let true_value = self.fresh();
        let false_value = self.fresh();
        let true_ptr = selected
            .true_ptr
            .ok_or_else(|| "native emitter: selected pointer load missing true arm".to_string())?;
        let false_ptr = selected
            .false_ptr
            .ok_or_else(|| "native emitter: selected pointer load missing false arm".to_string())?;
        instructions.push(Self::inst(
            Op::Load,
            Some(result_type),
            Some(true_value),
            vec![Operand::IdRef(true_ptr)],
        ));
        instructions.push(Self::inst(
            Op::Load,
            Some(result_type),
            Some(false_value),
            vec![Operand::IdRef(false_ptr)],
        ));
        instructions.push(Self::inst(
            Op::Select,
            Some(result_type),
            Some(result),
            vec![
                Operand::IdRef(selected.cond),
                Operand::IdRef(true_value),
                Operand::IdRef(false_value),
            ],
        ));
        Ok(())
    }

    pub(in crate::native::emitter) fn emit_selected_pointer_store(
        &mut self,
        object_ty: &LlType,
        object: Word,
        selected: &SelectedLoadPointer,
        access_align: Option<u64>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let object_ty = self.resolve_type(object_ty)?;
        let pointee = self.resolve_type(&selected.pointee)?;
        if let (Some(true_raw), Some(false_raw)) = (&selected.true_raw, &selected.false_raw) {
            return self.emit_selected_raw_store(
                &object_ty,
                object,
                true_raw,
                false_raw,
                selected.cond,
                access_align,
                instructions,
            );
        }
        if !types_compatible(&pointee, &object_ty) {
            return Err(format!(
                "native emitter: selected pointer store type mismatch {pointee:?} vs {object_ty:?}"
            ));
        }
        // `SelectedLoadPointer` keeps the two access-chain arms separate precisely because they can
        // resolve to distinct descriptor bindings. Selecting those pointer ids here would reintroduce
        // the illegal Logical-SPIR-V cross-binding pointer merge. Replay the ordinary store in the
        // value domain instead: each arm gets a read-modify-write value selection, matching the
        // direct selected-pointer store path above while keeping both stores rooted in their own
        // concrete pointer type.
        let true_ptr = selected
            .true_ptr
            .ok_or_else(|| "native emitter: selected pointer store missing true arm".to_string())?;
        let false_ptr = selected.false_ptr.ok_or_else(|| {
            "native emitter: selected pointer store missing false arm".to_string()
        })?;
        let object_type = self.type_id(&object_ty)?;
        let true_old = self.fresh();
        instructions.push(Self::inst(
            Op::Load,
            Some(object_type),
            Some(true_old),
            vec![Operand::IdRef(true_ptr)],
        ));
        let false_old = self.fresh();
        instructions.push(Self::inst(
            Op::Load,
            Some(object_type),
            Some(false_old),
            vec![Operand::IdRef(false_ptr)],
        ));
        let true_value = self.fresh();
        instructions.push(Self::inst(
            Op::Select,
            Some(object_type),
            Some(true_value),
            vec![
                Operand::IdRef(selected.cond),
                Operand::IdRef(object),
                Operand::IdRef(true_old),
            ],
        ));
        let false_value = self.fresh();
        instructions.push(Self::inst(
            Op::Select,
            Some(object_type),
            Some(false_value),
            vec![
                Operand::IdRef(selected.cond),
                Operand::IdRef(false_old),
                Operand::IdRef(object),
            ],
        ));
        instructions.push(Self::inst(
            Op::Store,
            None,
            None,
            vec![Operand::IdRef(true_ptr), Operand::IdRef(true_value)],
        ));
        instructions.push(Self::inst(
            Op::Store,
            None,
            None,
            vec![Operand::IdRef(false_ptr), Operand::IdRef(false_value)],
        ));
        Ok(())
    }

    pub(in crate::native::emitter) fn emit_raw_pointer_select_index(
        &mut self,
        name: &str,
        result_ty: &LlType,
        true_value: &LlValue,
        false_value: &LlValue,
        cond_id: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let LlType::Ptr(addrspace) = result_ty else {
            return Ok(false);
        };
        let (LlValue::Local(true_name), LlValue::Local(false_name)) = (true_value, false_value)
        else {
            return Ok(false);
        };
        let (Some(true_raw), Some(false_raw)) = (
            self.raw_offsets.get(true_name).cloned(),
            self.raw_offsets.get(false_name).cloned(),
        ) else {
            return Ok(false);
        };
        if true_raw.root != false_raw.root || true_raw.addrspace != false_raw.addrspace {
            return Ok(false);
        }
        if !self.raw_pointer_word_aligned(&true_raw) || !self.raw_pointer_word_aligned(&false_raw) {
            // Byte-granular fallback: when either arm is not word-aligned (e.g. a 2-byte `half`
            // element pointer merged with a struct-base pointer — the dominant aggregate-vs-element
            // pointer-merge class), select between the two full byte offsets and model the result as
            // a byte-stride raw offset. Downstream raw GEPs/loads add their own byte offsets, and the
            // sub-word raw load/store path already handles non-word-aligned half/i16 access.
            return self.emit_raw_pointer_select_byte(
                name,
                *addrspace,
                &true_raw,
                &false_raw,
                cond_id,
                instructions,
            );
        }

        let uint_ty = self.type_id(&LlType::Int(32))?;
        let true_word = self.emit_raw_word_index(&true_raw, 0, instructions)?;
        let false_word = self.emit_raw_word_index(&false_raw, 0, instructions)?;
        let word_name = raw_word_index_name(name);
        let result = self.result_id(&word_name, &LlType::Int(32))?;
        instructions.push(Self::inst(
            Op::Select,
            Some(uint_ty),
            Some(result),
            vec![
                Operand::IdRef(cond_id),
                Operand::IdRef(true_word),
                Operand::IdRef(false_word),
            ],
        ));
        self.pointer_storage
            .insert(name.to_string(), llvm_pointer_storage(*addrspace)?);
        self.pointer_pointees
            .insert(name.to_string(), raw_buffer_block_type());
        self.raw_offsets.insert(
            name.to_string(),
            RawBufferOffset {
                const_off: 0,
                dyn_terms: vec![(
                    TypedValue {
                        ty: LlType::Int(32),
                        value: LlValue::Local(word_name),
                    },
                    4,
                )],
                root: true_raw.root,
                addrspace: true_raw.addrspace,
                unmodelable: false,
                device_addr_base: true_raw.device_addr_base,
            },
        );
        Ok(true)
    }

    /// Byte-granular twin of the word-granular raw pointer-select tail in
    /// [`Self::emit_raw_pointer_select_index`]: select between the two arms' full byte offsets and
    /// model the merged pointer as a raw byte-stride (stride 1) offset.
    pub(in crate::native::emitter) fn emit_raw_pointer_select_byte(
        &mut self,
        name: &str,
        addrspace: u32,
        true_raw: &RawBufferOffset,
        false_raw: &RawBufferOffset,
        cond_id: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let uint_ty = self.type_id(&LlType::Int(32))?;
        let true_byte = self.emit_raw_byte_index(true_raw, 0, instructions)?;
        let false_byte = self.emit_raw_byte_index(false_raw, 0, instructions)?;
        let byte_name = raw_byte_index_name(name);
        let result = self.result_id(&byte_name, &LlType::Int(32))?;
        instructions.push(Self::inst(
            Op::Select,
            Some(uint_ty),
            Some(result),
            vec![
                Operand::IdRef(cond_id),
                Operand::IdRef(true_byte),
                Operand::IdRef(false_byte),
            ],
        ));
        self.pointer_storage
            .insert(name.to_string(), llvm_pointer_storage(addrspace)?);
        self.pointer_pointees
            .insert(name.to_string(), raw_buffer_block_type());
        self.raw_offsets.insert(
            name.to_string(),
            RawBufferOffset {
                const_off: 0,
                dyn_terms: vec![(
                    TypedValue {
                        ty: LlType::Int(32),
                        value: LlValue::Local(byte_name),
                    },
                    1,
                )],
                root: true_raw.root.clone(),
                addrspace: true_raw.addrspace,
                unmodelable: false,
                device_addr_base: true_raw.device_addr_base,
            },
        );
        Ok(true)
    }
}
