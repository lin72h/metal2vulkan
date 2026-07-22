//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    pub(in crate::native::emitter) fn drop_unmodeled_memcpy(&self, call: &LlCall) -> bool {
        call.callee.starts_with("llvm.memcpy.")
            && call.args.iter().take(2).any(|arg| match &arg.value {
                LlValue::Local(name) => self.unmodeled_pointers.contains(name),
                _ => false,
            })
    }

    pub(in crate::native::emitter) fn emit_raw_memcpy(
        &mut self,
        call: &LlCall,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if !call.callee.starts_with("llvm.memcpy.") {
            return Ok(false);
        }
        if call.args.len() != 4 || !matches!(call.args[3].value, LlValue::Bool(false)) {
            return Ok(false);
        }
        let Some(len) = typed_value_u64(&call.args[2]) else {
            return Ok(false);
        };
        if len == 0 {
            return Ok(false);
        }
        let (LlValue::Local(dst_name), LlValue::Local(src_name)) =
            (&call.args[0].value, &call.args[1].value)
        else {
            return Ok(false);
        };
        let dst_raw = self.raw_offsets.get(dst_name).cloned();
        let src_raw = self.raw_offsets.get(src_name).cloned();
        if let (Some(dst), Some(src)) = (dst_raw.as_ref(), src_raw.as_ref()) {
            if self.raw_pointer_word_aligned(dst)
                && self.raw_pointer_word_aligned(src)
                && len % 4 == 0
            {
                for byte in (0..len).step_by(4) {
                    let word = self.emit_raw_word_load(src, byte, instructions)?;
                    self.emit_raw_word_store(dst, byte, word, instructions)?;
                }
            } else {
                for byte in 0..len {
                    let value = self.emit_raw_byte_load_as_u32(src, byte, instructions)?;
                    self.emit_raw_byte_store_from_u32(value, dst, byte, instructions)?;
                }
            }
            return Ok(true);
        }

        if let Some(dst) = dst_raw {
            let dst_align = call.arg_aligns.first().copied().flatten();
            return self.emit_typed_to_raw_memcpy(
                &dst,
                dst_align,
                &call.args[1],
                len,
                instructions,
            );
        }

        if let Some(src) = src_raw {
            if !self.raw_pointer_word_aligned(&src) {
                return Ok(false);
            }
            if let Some((dst_root, dst_addrspace, dst_base)) =
                self.byte_gep_root_and_const_offset(&call.args[0])?
            {
                if dst_base < 0 || dst_base as u64 > u32::MAX as u64 - len {
                    return Ok(false);
                }
                let dst_ptr_ty =
                    self.ptr_type_id(llvm_pointer_storage(dst_addrspace)?, &LlType::Int(8))?;
                for byte in (0..len).step_by(4) {
                    let word = self.emit_raw_word_load(&src, byte, instructions)?;
                    let dst_ptr = self.fresh();
                    let byte_offset = self.const_uint(dst_base as u32 + byte as u32)?;
                    instructions.push(Self::inst(
                        Op::InBoundsAccessChain,
                        Some(dst_ptr_ty),
                        Some(dst_ptr),
                        vec![Operand::IdRef(dst_root), Operand::IdRef(byte_offset)],
                    ));
                    instructions.push(Self::inst(
                        Op::Store,
                        None,
                        None,
                        vec![Operand::IdRef(dst_ptr), Operand::IdRef(word)],
                    ));
                }
                return Ok(true);
            }
            return self.emit_raw_to_typed_struct_memcpy(&call.args[0], &src, len, instructions);
        }
        Ok(false)
    }

    pub(in crate::native::emitter) fn emit_typed_to_raw_memcpy(
        &mut self,
        dst: &RawBufferOffset,
        dst_align: Option<u64>,
        src: &TypedValue,
        len: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let Some(src_pointee) = self.pointer_pointee_for_value(&src.value)? else {
            return Ok(false);
        };
        let src_pointee = function_storage_local_type(&self.resolve_type(&src_pointee)?);
        let (src_size, _) = self.raw_type_size_align(&src_pointee)?;
        if len > src_size {
            return Ok(false);
        }
        let src_ptr = self.value_id(&src.value, &src.ty)?;
        let src_ty = self.type_id(&src_pointee)?;
        let src_value = self.fresh();
        instructions.push(Self::inst(
            Op::Load,
            Some(src_ty),
            Some(src_value),
            vec![Operand::IdRef(src_ptr)],
        ));

        let mut words = Vec::new();
        if !self.collect_typed_memcpy_words(
            src_value,
            &src_pointee,
            0,
            len,
            &mut words,
            instructions,
        )? {
            return Ok(false);
        }
        words.sort_by_key(|(offset, _)| *offset);
        let expected_words = (len / 4) as usize;
        if words.len() != expected_words {
            let Some(words_with_padding) =
                self.typed_memcpy_words_with_raw_padding(src, len, words)?
            else {
                return Ok(false);
            };
            words = words_with_padding;
        }
        for (index, (offset, word)) in words.into_iter().enumerate() {
            if offset != (index as u64) * 4 {
                return Ok(false);
            }
            self.emit_raw_word_store_for_access(dst, offset, word, dst_align, instructions)?;
        }
        Ok(true)
    }

    pub(in crate::native::emitter) fn typed_memcpy_words_with_raw_padding(
        &self,
        src: &TypedValue,
        len: u64,
        words: Vec<(u64, Word)>,
    ) -> Result<Option<Vec<(u64, Word)>>, String> {
        if !len.is_multiple_of(4) {
            return Ok(None);
        }
        let LlValue::Local(src_name) = &src.value else {
            return Ok(None);
        };
        let Some(shadow) = self.raw_memcpy_shadows.get(src_name) else {
            return Ok(None);
        };
        let mut by_offset = HashMap::new();
        for (offset, word) in words {
            if offset % 4 != 0 || offset >= len {
                return Ok(None);
            }
            by_offset.insert(offset, word);
        }
        for (offset, word) in shadow {
            if *offset % 4 != 0 || *offset >= len {
                continue;
            }
            by_offset.entry(*offset).or_insert(*word);
        }
        let expected_words = (len / 4) as usize;
        if by_offset.len() != expected_words {
            return Ok(None);
        }
        let mut words = Vec::with_capacity(expected_words);
        for index in 0..expected_words {
            let offset = (index as u64) * 4;
            let Some(word) = by_offset.get(&offset).copied() else {
                return Ok(None);
            };
            words.push((offset, word));
        }
        Ok(Some(words))
    }

    pub(in crate::native::emitter) fn collect_typed_memcpy_words(
        &mut self,
        value: Word,
        ty: &LlType,
        base_offset: u64,
        len: u64,
        words: &mut Vec<(u64, Word)>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if base_offset >= len {
            return Ok(true);
        }
        match self.resolve_type(ty)? {
            LlType::Int(32) | LlType::Float => {
                if base_offset + 4 > len {
                    return Ok(false);
                }
                let word = self.typed_memcpy_word(value, ty, instructions)?;
                words.push((base_offset, word));
                Ok(true)
            }
            LlType::Vector(elem, lanes) => {
                if !matches!(self.resolve_type(&elem)?, LlType::Int(32) | LlType::Float) {
                    return Ok(false);
                }
                for lane in 0..lanes {
                    let offset = base_offset + (lane as u64) * 4;
                    if offset >= len {
                        break;
                    }
                    if offset + 4 > len {
                        return Ok(false);
                    }
                    let elem_ty = self.resolve_type(&elem)?;
                    let elem_type = self.type_id(&elem_ty)?;
                    let elem_value = self.fresh();
                    instructions.push(Self::inst(
                        Op::CompositeExtract,
                        Some(elem_type),
                        Some(elem_value),
                        vec![Operand::IdRef(value), Operand::LiteralBit32(lane)],
                    ));
                    let word = self.typed_memcpy_word(elem_value, &elem_ty, instructions)?;
                    words.push((offset, word));
                }
                Ok(true)
            }
            LlType::Array(elem, count) => {
                let elem = self.resolve_type(&elem)?;
                let (elem_size, elem_align) = self.raw_type_size_align(&elem)?;
                let stride = round_up_u64(elem_size, elem_align);
                let elem_type = self.type_id(&elem)?;
                for index in 0..count {
                    let offset = base_offset + stride * index as u64;
                    if offset >= len {
                        break;
                    }
                    let elem_value = self.fresh();
                    instructions.push(Self::inst(
                        Op::CompositeExtract,
                        Some(elem_type),
                        Some(elem_value),
                        vec![Operand::IdRef(value), Operand::LiteralBit32(index)],
                    ));
                    if !self.collect_typed_memcpy_words(
                        elem_value,
                        &elem,
                        offset,
                        len,
                        words,
                        instructions,
                    )? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            LlType::Struct(fields) => {
                for index in 0..fields.len() {
                    let (offset, field) = self.raw_struct_member(&fields, index as u64)?;
                    let offset = base_offset + offset;
                    if offset >= len {
                        break;
                    }
                    let field_type = self.type_id(&field)?;
                    let field_value = self.fresh();
                    instructions.push(Self::inst(
                        Op::CompositeExtract,
                        Some(field_type),
                        Some(field_value),
                        vec![Operand::IdRef(value), Operand::LiteralBit32(index as u32)],
                    ));
                    if !self.collect_typed_memcpy_words(
                        field_value,
                        &field,
                        offset,
                        len,
                        words,
                        instructions,
                    )? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub(in crate::native::emitter) fn typed_memcpy_word(
        &mut self,
        value: Word,
        ty: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        match self.resolve_type(ty)? {
            LlType::Int(32) => Ok(value),
            LlType::Float => {
                let uint_ty = self.type_id(&LlType::Int(32))?;
                let word = self.fresh();
                instructions.push(Self::inst(
                    Op::Bitcast,
                    Some(uint_ty),
                    Some(word),
                    vec![Operand::IdRef(value)],
                ));
                Ok(word)
            }
            other => Err(format!(
                "native emitter: memcpy word from typed source {other:?} is not supported"
            )),
        }
    }

    pub(in crate::native::emitter) fn emit_raw_to_typed_struct_memcpy(
        &mut self,
        dst: &TypedValue,
        src: &RawBufferOffset,
        len: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let Some(dst_pointee) = self.pointer_pointee_for_value(&dst.value)? else {
            return Ok(false);
        };
        let dst_pointee = function_storage_local_type(&self.resolve_type(&dst_pointee)?);
        if !matches!(&dst_pointee, LlType::Struct(_)) {
            return Ok(false);
        }
        let LlType::Ptr(dst_addrspace) = self.resolve_type(&dst.ty)? else {
            return Ok(false);
        };
        let dst_storage = self.pointer_storage_for(&dst.value, dst_addrspace)?;
        let dst_id = self.value_id(&dst.value, &dst.ty)?;

        // Emit the whole copy into a scratch buffer first: a nested aggregate that turns out to carry
        // an unsupported leaf (e.g. a sub-word field) bails to `Ok(false)` and the caller falls back to
        // the typed memcpy path, so nothing partial may be committed to the real stream.
        let mut scratch = Vec::new();
        let word_shadow = if len.is_multiple_of(4) {
            let mut words = Vec::new();
            for byte in (0..len).step_by(4) {
                words.push((byte, self.emit_raw_word_load(src, byte, &mut scratch)?));
            }
            Some(words)
        } else {
            None
        };

        let mut copied_any = false;
        let mut index_path = Vec::new();
        let ok = self.emit_raw_to_typed_aggregate_fields(
            src,
            dst_id,
            dst_storage,
            &dst_pointee,
            0,
            &mut index_path,
            len,
            &mut scratch,
            &mut copied_any,
        )?;
        if !ok || !copied_any {
            return Ok(false);
        }
        instructions.extend(scratch);
        if let (LlValue::Local(dst_name), Some(words)) = (&dst.value, word_shadow) {
            self.raw_memcpy_shadows.insert(dst_name.clone(), words);
        }
        Ok(true)
    }

    /// Recursively store a raw device byte range into a typed (Function/Private) aggregate, one scalar
    /// or vector leaf at a time. Structs and arrays recurse, accumulating the access-chain index path
    /// and the running byte offset; scalar/vector leaves are read from `src` at their byte offset and
    /// stored through the full chain. This is the nesting generalization of the flat top-level struct
    /// copy — byte-faithful because every leaf lands at exactly the layout offset `raw_type_size_align`
    /// dictates, and the destination is per-invocation scratch (never the host-visible golden output).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::native::emitter) fn emit_raw_to_typed_aggregate_fields(
        &mut self,
        src: &RawBufferOffset,
        dst_id: Word,
        dst_storage: StorageClass,
        ty: &LlType,
        base_offset: u64,
        index_path: &mut Vec<Word>,
        len: u64,
        out: &mut Vec<Instruction>,
        copied_any: &mut bool,
    ) -> Result<bool, String> {
        match self.resolve_type(ty)? {
            LlType::Struct(fields) => {
                for index in 0..fields.len() {
                    let (member_off, field) = self.raw_struct_member(&fields, index as u64)?;
                    let offset = base_offset + member_off;
                    if offset >= len {
                        break;
                    }
                    let field = function_storage_local_type(&field);
                    let (field_size, _) = self.raw_type_size_align(&field)?;
                    if field_size == 0 {
                        continue;
                    }
                    if offset + field_size > len {
                        return Ok(false);
                    }
                    let idx = self.const_uint(index as u32)?;
                    index_path.push(idx);
                    let ok = self.emit_raw_to_typed_aggregate_fields(
                        src,
                        dst_id,
                        dst_storage,
                        &field,
                        offset,
                        index_path,
                        len,
                        out,
                        copied_any,
                    )?;
                    index_path.pop();
                    if !ok {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            LlType::Array(elem, count) => {
                let elem = function_storage_local_type(&self.resolve_type(&elem)?);
                let (elem_size, elem_align) = self.raw_type_size_align(&elem)?;
                if elem_size == 0 {
                    return Ok(true);
                }
                let stride = round_up_u64(elem_size, elem_align);
                for i in 0..count as u64 {
                    let offset = base_offset + i * stride;
                    if offset >= len {
                        break;
                    }
                    if offset + elem_size > len {
                        return Ok(false);
                    }
                    let idx = self.const_uint(i as u32)?;
                    index_path.push(idx);
                    let ok = self.emit_raw_to_typed_aggregate_fields(
                        src,
                        dst_id,
                        dst_storage,
                        &elem,
                        offset,
                        index_path,
                        len,
                        out,
                        copied_any,
                    )?;
                    index_path.pop();
                    if !ok {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            leaf @ (LlType::Int(32) | LlType::Float | LlType::Vector(_, _)) => {
                let Some(value) =
                    self.emit_raw_word_as_typed_memcpy_field(src, base_offset, &leaf, out)?
                else {
                    return Ok(false);
                };
                let dst_ptr_ty = self.ptr_type_id(dst_storage, &leaf)?;
                let dst_field = self.fresh();
                let mut operands = Vec::with_capacity(index_path.len() + 1);
                operands.push(Operand::IdRef(dst_id));
                for idx in index_path.iter() {
                    operands.push(Operand::IdRef(*idx));
                }
                out.push(Self::inst(
                    Op::InBoundsAccessChain,
                    Some(dst_ptr_ty),
                    Some(dst_field),
                    operands,
                ));
                out.push(Self::inst(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(dst_field), Operand::IdRef(value)],
                ));
                *copied_any = true;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    pub(in crate::native::emitter) fn emit_raw_word_as_typed_memcpy_field(
        &mut self,
        src: &RawBufferOffset,
        offset: u64,
        field: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Option<Word>, String> {
        if !offset.is_multiple_of(4) {
            return Ok(None);
        }
        match self.resolve_type(field)? {
            LlType::Int(32) | LlType::Float => {
                let (field_size, _) = self.raw_type_size_align(field)?;
                if field_size != 4 {
                    return Ok(None);
                }
                let word = self.emit_raw_word_load(src, offset, instructions)?;
                self.emit_raw_word_as_typed_scalar(word, field, instructions)
                    .map(Some)
            }
            LlType::Vector(elem, lanes) => {
                let elem = self.resolve_type(&elem)?;
                if !matches!(elem, LlType::Int(32) | LlType::Float) {
                    return Ok(None);
                }
                let (elem_size, _) = self.raw_type_size_align(&elem)?;
                if elem_size != 4 {
                    return Ok(None);
                }
                let elem_ty = self.type_id(&elem)?;
                let vector_ty = self.type_id(field)?;
                let mut values = Vec::with_capacity(lanes as usize);
                for lane in 0..lanes {
                    let word =
                        self.emit_raw_word_load(src, offset + u64::from(lane) * 4, instructions)?;
                    let value = self.emit_raw_word_as_typed_scalar(word, &elem, instructions)?;
                    let value_ty = self.type_id(&elem)?;
                    if value_ty != elem_ty {
                        return Ok(None);
                    }
                    values.push(Operand::IdRef(value));
                }
                let vector = self.fresh();
                instructions.push(Self::inst(
                    Op::CompositeConstruct,
                    Some(vector_ty),
                    Some(vector),
                    values,
                ));
                Ok(Some(vector))
            }
            _ => Ok(None),
        }
    }

    pub(in crate::native::emitter) fn emit_raw_word_as_typed_scalar(
        &mut self,
        word: Word,
        ty: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        match self.resolve_type(ty)? {
            LlType::Int(32) => Ok(word),
            LlType::Float => {
                let float_ty = self.type_id(&LlType::Float)?;
                let value = self.fresh();
                instructions.push(Self::inst(
                    Op::Bitcast,
                    Some(float_ty),
                    Some(value),
                    vec![Operand::IdRef(word)],
                ));
                Ok(value)
            }
            other => Err(format!(
                "native emitter: raw memcpy scalar field {other:?} is not supported"
            )),
        }
    }

    pub(in crate::native::emitter) fn byte_gep_root_and_const_offset(
        &self,
        ptr: &TypedValue,
    ) -> Result<Option<(Word, u32, i64)>, String> {
        let LlValue::Local(name) = &ptr.value else {
            return Ok(None);
        };
        let Some(provenance) = self.gep_provenance.get(name) else {
            return Ok(None);
        };
        if self.resolve_type(&provenance.source_ty)? != LlType::Int(8) {
            return Ok(None);
        }
        if provenance.indices.len() != 1 {
            return Ok(None);
        }
        let Some(base) = const_index_i64(&provenance.indices[0]) else {
            return Ok(None);
        };
        Ok(Some((provenance.root, provenance.addrspace, base)))
    }

    pub(in crate::native::emitter) fn emit_zero_memset(
        &mut self,
        call: &LlCall,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if !call.callee.starts_with("llvm.memset.") {
            return Ok(false);
        }
        if call.args.len() != 4
            || !typed_value_is_zero(&call.args[1])
            || typed_value_u64(&call.args[2]).is_none_or(|len| len == 0)
            || !matches!(call.args[3].value, LlValue::Bool(false))
        {
            return Ok(false);
        }
        let len = typed_value_u64(&call.args[2]).ok_or_else(|| {
            "native emitter: zero-memset length is not a positive u64 constant".to_string()
        })?;
        if let LlValue::Local(name) = &call.args[0].value {
            if let Some(raw) = self.raw_offsets.get(name).cloned() {
                let align = call.arg_aligns.first().copied().flatten();
                self.emit_raw_zero_memset(&raw, len, align, instructions)?;
                return Ok(true);
            }
        }
        if self.emit_byte_pointer_zero_memset(&call.args[0], len, instructions)? {
            return Ok(true);
        }
        let Some(pointee) = self.pointer_pointee_for_value(&call.args[0].value)? else {
            return Ok(false);
        };
        let storage_pointee = function_storage_local_type(&pointee);
        let (size, _) = self.raw_type_size_align(&storage_pointee)?;
        if len < size {
            // Partial zero-memset: clearing the first `len` bytes of a LARGER object. When the pointee
            // is an array and `len` is a whole number of leading elements, clear those elements with one
            // `OpStore null` each (the byte range [0,len) maps to element indices [0, len/elem_size)) —
            // a typed lowering that stays valid under Logical addressing, where the alternative (a
            // generic call to the byte `llvm.memset` declaration whose pointer param is `uchar`) is an
            // invalid-SPIR-V dead end. A sub-element or struct-spanning partial memset has no clean typed
            // form, so it still falls through.
            if let LlType::Array(elem, _) = self.resolve_type(&storage_pointee)? {
                let elem_ty = self.resolve_type(&elem)?;
                let (elem_size, _) = self.raw_type_size_align(&elem_ty)?;
                if elem_size > 0 && len % elem_size == 0 {
                    let LlType::Ptr(addrspace) = self.resolve_type(&call.args[0].ty)? else {
                        return Ok(false);
                    };
                    let storage = self.pointer_storage_for(&call.args[0].value, addrspace)?;
                    let elem_ptr_ty = self.ptr_type_id(storage, &elem_ty)?;
                    let base = self.value_id(&call.args[0].value, &call.args[0].ty)?;
                    let zero = self.const_null(&elem_ty)?;
                    for i in 0..(len / elem_size) {
                        let index = self.const_uint(u32::try_from(i).map_err(|_| {
                            format!("native emitter: memset element index {i} exceeds u32")
                        })?)?;
                        let elem_ptr = self.fresh();
                        instructions.push(Self::inst(
                            Op::InBoundsAccessChain,
                            Some(elem_ptr_ty),
                            Some(elem_ptr),
                            vec![Operand::IdRef(base), Operand::IdRef(index)],
                        ));
                        instructions.push(Self::inst(
                            Op::Store,
                            None,
                            None,
                            vec![Operand::IdRef(elem_ptr), Operand::IdRef(zero)],
                        ));
                    }
                    return Ok(true);
                }
            }
            return Ok(false);
        }
        let ptr = self.value_id(&call.args[0].value, &call.args[0].ty)?;
        let zero = self.const_null(&storage_pointee)?;
        instructions.push(Self::inst(
            Op::Store,
            None,
            None,
            vec![Operand::IdRef(ptr), Operand::IdRef(zero)],
        ));
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_raw_zero_memset(
        &mut self,
        raw: &RawBufferOffset,
        len: u64,
        access_align: Option<u64>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let zero = self.const_uint(0)?;
        if self.raw_pointer_word_aligned(raw) {
            let word_bytes = len / 4 * 4;
            for byte in (0..word_bytes).step_by(4) {
                self.emit_raw_word_store_for_access(raw, byte, zero, access_align, instructions)?;
            }
            for byte in word_bytes..len {
                self.emit_raw_byte_store_from_u32(zero, raw, byte, instructions)?;
            }
            return Ok(());
        }
        for byte in 0..len {
            self.emit_raw_byte_store_from_u32(zero, raw, byte, instructions)?;
        }
        Ok(())
    }

    pub(in crate::native::emitter) fn emit_byte_pointer_zero_memset(
        &mut self,
        dst: &TypedValue,
        len: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let Some(pointee) = self.pointer_pointee_for_value(&dst.value)? else {
            return Ok(false);
        };
        if self.resolve_type(&pointee)? != LlType::Int(8) {
            return Ok(false);
        }
        let LlType::Ptr(addrspace) = self.resolve_type(&dst.ty)? else {
            return Ok(false);
        };
        let storage = self.pointer_storage_for(&dst.value, addrspace)?;
        let access_op = if ptr_access_chain_allowed_storage(storage) {
            Op::PtrAccessChain
        } else {
            Op::InBoundsAccessChain
        };
        let ptr_ty = self.ptr_type_id(storage, &LlType::Int(8))?;
        let ptr = self.value_id(&dst.value, &dst.ty)?;
        let zero = self.const_null(&LlType::Int(8))?;
        for byte in 0..len {
            let store_ptr = if byte == 0 {
                ptr
            } else {
                let index = self.const_uint(u32::try_from(byte).map_err(|_| {
                    format!("native emitter: memset byte offset {byte} exceeds u32")
                })?)?;
                let byte_ptr = self.fresh();
                instructions.push(Self::inst(
                    access_op,
                    Some(ptr_ty),
                    Some(byte_ptr),
                    vec![Operand::IdRef(ptr), Operand::IdRef(index)],
                ));
                byte_ptr
            };
            instructions.push(Self::inst(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(store_ptr), Operand::IdRef(zero)],
            ));
        }
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_typed_memcpy(
        &mut self,
        call: &LlCall,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if !call.callee.starts_with("llvm.memcpy.") {
            return Ok(false);
        }
        if call.args.len() != 4 || !matches!(call.args[3].value, LlValue::Bool(false)) {
            return Ok(false);
        }
        let Some(len) = typed_value_u64(&call.args[2]) else {
            return Ok(false);
        };
        if len == 0 {
            return Ok(true);
        }
        let Some(dst_pointee) = self.pointer_pointee_for_value(&call.args[0].value)? else {
            return Ok(false);
        };
        let Some(src_pointee) = self.pointer_pointee_for_value(&call.args[1].value)? else {
            return Ok(false);
        };
        let copy_pointee = function_storage_local_type(&src_pointee);
        let (copy_size, _) = self.raw_type_size_align(&copy_pointee)?;
        if len < copy_size {
            if !types_compatible(&dst_pointee, &src_pointee) {
                return Ok(false);
            }
            let LlType::Struct(fields) = &copy_pointee else {
                return Ok(false);
            };
            let LlType::Ptr(dst_addrspace) = self.resolve_type(&call.args[0].ty)? else {
                return Ok(false);
            };
            let LlType::Ptr(src_addrspace) = self.resolve_type(&call.args[1].ty)? else {
                return Ok(false);
            };
            let dst_storage = self.pointer_storage_for(&call.args[0].value, dst_addrspace)?;
            let src_storage = self.pointer_storage_for(&call.args[1].value, src_addrspace)?;
            let dst = self.value_id(&call.args[0].value, &call.args[0].ty)?;
            let src = self.value_id(&call.args[1].value, &call.args[1].ty)?;
            return self.emit_prefix_struct_memcpy(
                dst,
                src,
                dst_storage,
                src_storage,
                fields,
                fields,
                len,
                instructions,
            );
        }

        let LlType::Ptr(dst_addrspace) = self.resolve_type(&call.args[0].ty)? else {
            return Ok(false);
        };
        let LlType::Ptr(src_addrspace) = self.resolve_type(&call.args[1].ty)? else {
            return Ok(false);
        };
        let dst_storage = self.pointer_storage_for(&call.args[0].value, dst_addrspace)?;
        let src_storage = self.pointer_storage_for(&call.args[1].value, src_addrspace)?;
        let src = self.value_id(&call.args[1].value, &call.args[1].ty)?;
        if self.can_emit_whole_copy_memory(&dst_pointee, &src_pointee, dst_storage, src_storage)? {
            let dst = self.value_id(&call.args[0].value, &call.args[0].ty)?;
            self.emit_copy_memory(dst, src, instructions);
            return Ok(true);
        }

        if let (LlType::Struct(dst_fields), LlType::Struct(src_fields)) =
            (&dst_pointee, &src_pointee)
        {
            if self.struct_prefix_fields_compatible(dst_fields, src_fields)? {
                let dst = self.value_id(&call.args[0].value, &call.args[0].ty)?;
                return self.emit_prefix_struct_memcpy(
                    dst,
                    src,
                    dst_storage,
                    src_storage,
                    dst_fields,
                    src_fields,
                    len,
                    instructions,
                );
            }
        }

        if let (LlType::Array(dst_elem, dst_len), LlType::Array(src_elem, src_len)) =
            (&dst_pointee, &src_pointee)
        {
            if dst_len != src_len || !types_compatible(dst_elem, src_elem) {
                return Ok(false);
            }
            let dst = self.value_id(&call.args[0].value, &call.args[0].ty)?;
            return self.emit_prefix_array_memcpy(
                dst,
                src,
                dst_storage,
                src_storage,
                dst_elem,
                src_elem,
                *src_len,
                len,
                instructions,
            );
        }

        if let LlType::Struct(fields) = &dst_pointee {
            let Some(field) = fields.first() else {
                return Ok(false);
            };
            if !types_compatible(field, &src_pointee) {
                return Ok(false);
            }
            let dst_field_pointee = function_storage_local_type(field);
            let field_ptr_ty = self.ptr_type_id(dst_storage, &dst_field_pointee)?;
            let base = self.value_id(&call.args[0].value, &call.args[0].ty)?;
            let zero = self.const_uint(0)?;
            let field_ptr = self.fresh();
            instructions.push(Self::inst(
                Op::InBoundsAccessChain,
                Some(field_ptr_ty),
                Some(field_ptr),
                vec![Operand::IdRef(base), Operand::IdRef(zero)],
            ));
            return self.emit_aggregate_memcpy(
                field_ptr,
                src,
                dst_storage,
                src_storage,
                &dst_field_pointee,
                &src_pointee,
                len,
                instructions,
            );
        }

        Ok(false)
    }

    pub(in crate::native::emitter) fn struct_prefix_fields_compatible(
        &self,
        dst_fields: &[LlType],
        src_fields: &[LlType],
    ) -> Result<bool, String> {
        if src_fields.len() > dst_fields.len() {
            return Ok(false);
        }
        for index in 0..src_fields.len() {
            let (dst_offset, dst_field) = self.raw_struct_member(dst_fields, index as u64)?;
            let (src_offset, src_field) = self.raw_struct_member(src_fields, index as u64)?;
            if dst_offset != src_offset {
                return Ok(false);
            }
            let dst_field = function_storage_local_type(&dst_field);
            let src_field = function_storage_local_type(&src_field);
            if !types_compatible(&dst_field, &src_field) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(in crate::native::emitter) fn can_emit_whole_copy_memory(
        &mut self,
        dst_pointee: &LlType,
        src_pointee: &LlType,
        dst_storage: StorageClass,
        src_storage: StorageClass,
    ) -> Result<bool, String> {
        let dst_ty = function_storage_local_type(dst_pointee);
        let src_ty = function_storage_local_type(src_pointee);
        if self.type_id(&dst_ty)? != self.type_id(&src_ty)? {
            return Ok(false);
        }
        if (is_copy_memory_aggregate(&dst_ty) || is_copy_memory_aggregate(&src_ty))
            && (is_interface_backed_copy_storage(dst_storage)
                || is_interface_backed_copy_storage(src_storage))
        {
            return Ok(false);
        }
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_copy_memory(
        &self,
        dst: Word,
        src: Word,
        instructions: &mut Vec<Instruction>,
    ) {
        instructions.push(Self::inst(
            Op::CopyMemory,
            None,
            None,
            vec![Operand::IdRef(dst), Operand::IdRef(src)],
        ));
    }

    pub(in crate::native::emitter) fn emit_aggregate_memcpy(
        &mut self,
        dst: Word,
        src: Word,
        dst_storage: StorageClass,
        src_storage: StorageClass,
        dst_pointee: &LlType,
        src_pointee: &LlType,
        len: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if len == 0 {
            return Ok(true);
        }
        if self.can_emit_whole_copy_memory(dst_pointee, src_pointee, dst_storage, src_storage)? {
            self.emit_copy_memory(dst, src, instructions);
            return Ok(true);
        }

        let dst_pointee = function_storage_local_type(dst_pointee);
        let src_pointee = function_storage_local_type(src_pointee);
        match (&dst_pointee, &src_pointee) {
            (LlType::Struct(dst_fields), LlType::Struct(src_fields)) => {
                if !self.struct_prefix_fields_compatible(dst_fields, src_fields)? {
                    return Ok(false);
                }
                self.emit_prefix_struct_memcpy(
                    dst,
                    src,
                    dst_storage,
                    src_storage,
                    dst_fields,
                    src_fields,
                    len,
                    instructions,
                )
            }
            (LlType::Array(dst_elem, dst_len), LlType::Array(src_elem, src_len)) => {
                if dst_len != src_len || !types_compatible(dst_elem, src_elem) {
                    return Ok(false);
                }
                self.emit_prefix_array_memcpy(
                    dst,
                    src,
                    dst_storage,
                    src_storage,
                    dst_elem,
                    src_elem,
                    *src_len,
                    len,
                    instructions,
                )
            }
            _ => Ok(false),
        }
    }

    pub(in crate::native::emitter) fn emit_prefix_struct_memcpy(
        &mut self,
        dst: Word,
        src: Word,
        dst_storage: StorageClass,
        src_storage: StorageClass,
        dst_fields: &[LlType],
        src_fields: &[LlType],
        len: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let mut copied_any = false;
        for index in 0..src_fields.len() {
            let (dst_offset, dst_field) = self.raw_struct_member(dst_fields, index as u64)?;
            let (src_offset, src_field) = self.raw_struct_member(src_fields, index as u64)?;
            if dst_offset != src_offset {
                return Ok(false);
            }
            let offset = src_offset;
            if offset >= len {
                break;
            }
            let dst_field = function_storage_local_type(&dst_field);
            let src_field = function_storage_local_type(&src_field);
            let (field_size, _) = self.raw_type_size_align(&src_field)?;
            if field_size == 0 {
                continue;
            }
            if offset + field_size > len {
                return Ok(false);
            }

            let index_id = self.const_uint(index as u32)?;
            let dst_ptr_ty = self.ptr_type_id(dst_storage, &dst_field)?;
            let dst_field = self.fresh();
            instructions.push(Self::inst(
                Op::InBoundsAccessChain,
                Some(dst_ptr_ty),
                Some(dst_field),
                vec![Operand::IdRef(dst), Operand::IdRef(index_id)],
            ));
            let src_ptr_ty = self.ptr_type_id(src_storage, &src_field)?;
            let src_field = self.fresh();
            instructions.push(Self::inst(
                Op::InBoundsAccessChain,
                Some(src_ptr_ty),
                Some(src_field),
                vec![Operand::IdRef(src), Operand::IdRef(index_id)],
            ));
            if !self.emit_aggregate_memcpy(
                dst_field,
                src_field,
                dst_storage,
                src_storage,
                &dst_fields[index],
                &src_fields[index],
                field_size,
                instructions,
            )? {
                return Ok(false);
            }
            copied_any = true;
        }
        Ok(copied_any)
    }

    pub(in crate::native::emitter) fn emit_prefix_array_memcpy(
        &mut self,
        dst: Word,
        src: Word,
        dst_storage: StorageClass,
        src_storage: StorageClass,
        dst_elem: &LlType,
        src_elem: &LlType,
        count: u32,
        len: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let dst_elem = function_storage_local_type(dst_elem);
        let src_elem = function_storage_local_type(src_elem);
        let (elem_size, elem_align) = self.raw_type_size_align(&src_elem)?;
        let stride = round_up_u64(elem_size, elem_align);
        let mut copied_any = false;
        for index in 0..count {
            let offset = stride * u64::from(index);
            if offset >= len {
                break;
            }
            if offset + elem_size > len {
                return Ok(false);
            }

            let index_id = self.const_uint(index)?;
            let dst_ptr_ty = self.ptr_type_id(dst_storage, &dst_elem)?;
            let dst_element = self.fresh();
            instructions.push(Self::inst(
                Op::InBoundsAccessChain,
                Some(dst_ptr_ty),
                Some(dst_element),
                vec![Operand::IdRef(dst), Operand::IdRef(index_id)],
            ));
            let src_ptr_ty = self.ptr_type_id(src_storage, &src_elem)?;
            let src_element = self.fresh();
            instructions.push(Self::inst(
                Op::InBoundsAccessChain,
                Some(src_ptr_ty),
                Some(src_element),
                vec![Operand::IdRef(src), Operand::IdRef(index_id)],
            ));
            if !self.emit_aggregate_memcpy(
                dst_element,
                src_element,
                dst_storage,
                src_storage,
                &dst_elem,
                &src_elem,
                elem_size,
                instructions,
            )? {
                return Ok(false);
            }
            copied_any = true;
        }
        Ok(copied_any)
    }
}
