//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    pub(in crate::native::emitter) fn emit_raw_pointer_payload(
        &mut self,
        raw: &RawBufferOffset,
        extra_byte: u64,
        access_align: Option<u64>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<((Word, Word), Word), String> {
        let low_word =
            self.emit_raw_word_load_for_access(raw, extra_byte, access_align, instructions)?;
        let high_word =
            self.emit_raw_word_load_for_access(raw, extra_byte + 4, access_align, instructions)?;
        let zero = self.const_uint(0)?;
        let bool_ty = self.type_id(&LlType::Bool)?;
        let low_is_zero = self.fresh();
        instructions.push(Self::inst(
            Op::IEqual,
            Some(bool_ty),
            Some(low_is_zero),
            vec![Operand::IdRef(low_word), Operand::IdRef(zero)],
        ));
        let high_is_zero = self.fresh();
        instructions.push(Self::inst(
            Op::IEqual,
            Some(bool_ty),
            Some(high_is_zero),
            vec![Operand::IdRef(high_word), Operand::IdRef(zero)],
        ));
        let is_null = self.fresh();
        instructions.push(Self::inst(
            Op::LogicalAnd,
            Some(bool_ty),
            Some(is_null),
            vec![Operand::IdRef(low_is_zero), Operand::IdRef(high_is_zero)],
        ));
        Ok(((low_word, high_word), is_null))
    }

    pub(in crate::native::emitter) fn emit_raw_word_load(
        &mut self,
        raw: &RawBufferOffset,
        extra_byte: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let word_index = self.emit_raw_word_index(raw, extra_byte, instructions)?;
        self.emit_raw_word_load_at_index(raw, word_index, instructions)
    }

    pub(in crate::native::emitter) fn emit_raw_word_load_for_access(
        &mut self,
        raw: &RawBufferOffset,
        extra_byte: u64,
        access_align: Option<u64>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        if !self.raw_word_access_can_use_word_index(raw, extra_byte, access_align) {
            return self.emit_raw_u32_from_unaligned_bytes(raw, extra_byte, instructions);
        }
        let word_index =
            self.emit_raw_word_index_for_access(raw, extra_byte, access_align, instructions)?;
        self.emit_raw_word_load_at_index(raw, word_index, instructions)
    }

    pub(in crate::native::emitter) fn emit_raw_word_load_at_index(
        &mut self,
        raw: &RawBufferOffset,
        word_index: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let ptr = self.emit_raw_word_pointer_at_index(raw, word_index, instructions)?;
        let word_ty = self.type_id(&LlType::Int(32))?;
        let word = self.fresh();
        instructions.push(Self::inst(
            Op::Load,
            Some(word_ty),
            Some(word),
            vec![Operand::IdRef(ptr)],
        ));
        Ok(word)
    }

    pub(in crate::native::emitter) fn emit_raw_word_pointer_at_index(
        &mut self,
        raw: &RawBufferOffset,
        word_index: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let (ptr, _) =
            self.emit_raw_word_pointer_at_index_with_storage(raw, word_index, instructions)?;
        Ok(ptr)
    }

    pub(in crate::native::emitter) fn emit_raw_word_pointer_for_access(
        &mut self,
        raw: &RawBufferOffset,
        extra_byte: u64,
        access_align: Option<u64>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let word_index =
            self.emit_raw_word_index_for_access(raw, extra_byte, access_align, instructions)?;
        self.emit_raw_word_pointer_at_index(raw, word_index, instructions)
    }

    pub(in crate::native::emitter) fn emit_raw_word_pointer_at_index_with_storage(
        &mut self,
        raw: &RawBufferOffset,
        word_index: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(Word, StorageClass), String> {
        let root_id = self.raw_root_value_id(raw)?;
        let storage = self.raw_access_storage(raw)?;
        let ptr_ty = self.ptr_type_id(storage, &LlType::Int(32))?;
        let ptr = self.fresh();
        if storage == StorageClass::Private
            && self
                .pointer_pointees
                .get(&raw.root)
                .is_some_and(|pointee| pointee == &LlType::Int(32))
        {
            instructions.push(Self::inst(
                Op::CopyObject,
                Some(ptr_ty),
                Some(ptr),
                vec![Operand::IdRef(root_id)],
            ));
            return Ok((ptr, storage));
        }
        let mut ops = vec![Operand::IdRef(root_id)];
        if storage != StorageClass::Workgroup {
            ops.push(Operand::IdRef(self.const_uint(0)?));
        }
        ops.push(Operand::IdRef(word_index));
        instructions.push(Self::inst(
            Op::InBoundsAccessChain,
            Some(ptr_ty),
            Some(ptr),
            ops,
        ));
        Ok((ptr, storage))
    }

    pub(in crate::native::emitter) fn raw_access_storage(
        &self,
        raw: &RawBufferOffset,
    ) -> Result<StorageClass, String> {
        self.pointer_storage
            .get(&raw.root)
            .copied()
            .map(Ok)
            .unwrap_or_else(|| llvm_pointer_storage(raw.addrspace))
    }

    pub(in crate::native::emitter) fn emit_raw_word_store(
        &mut self,
        raw: &RawBufferOffset,
        extra_byte: u64,
        value: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let word_index = self.emit_raw_word_index(raw, extra_byte, instructions)?;
        self.emit_raw_word_store_at_index(raw, word_index, value, instructions)
    }

    pub(in crate::native::emitter) fn emit_raw_word_store_for_access(
        &mut self,
        raw: &RawBufferOffset,
        extra_byte: u64,
        value: Word,
        access_align: Option<u64>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        if !self.raw_word_access_can_use_word_index(raw, extra_byte, access_align) {
            return self.emit_raw_word_store_unaligned_bytes(raw, extra_byte, value, instructions);
        }
        let word_index =
            self.emit_raw_word_index_for_access(raw, extra_byte, access_align, instructions)?;
        self.emit_raw_word_store_at_index(raw, word_index, value, instructions)
    }

    pub(in crate::native::emitter) fn emit_raw_word_store_at_index(
        &mut self,
        raw: &RawBufferOffset,
        word_index: Word,
        value: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let ptr = self.emit_raw_word_pointer_at_index(raw, word_index, instructions)?;
        instructions.push(Self::inst(
            Op::Store,
            None,
            None,
            vec![Operand::IdRef(ptr), Operand::IdRef(value)],
        ));
        Ok(())
    }

    pub(in crate::native::emitter) fn emit_raw_subword_source(
        &mut self,
        raw: &RawBufferOffset,
        extra_byte: u64,
        element_size: u32,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(Word, RawSubwordLane), String> {
        let (_, word, lane) =
            self.emit_raw_subword_target(raw, extra_byte, element_size, instructions)?;
        Ok((word, lane))
    }

    pub(in crate::native::emitter) fn emit_raw_subword_target(
        &mut self,
        raw: &RawBufferOffset,
        extra_byte: u64,
        element_size: u32,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(Word, Word, RawSubwordLane), String> {
        let off = raw.const_off + extra_byte as i64;
        let needs_dynamic_lane = raw
            .dyn_terms
            .iter()
            .any(|(index, stride)| !self.raw_dynamic_term_word_aligned(index, *stride));
        if !self.raw_offset_aligned_to(raw, extra_byte, element_size as u64) {
            return Err(format!(
                "native emitter: raw byte offset is not {element_size}-byte aligned"
            ));
        }
        if !needs_dynamic_lane {
            let word_index = self.emit_raw_word_index(raw, extra_byte, instructions)?;
            let word = self.emit_raw_word_load_at_index(raw, word_index, instructions)?;
            return Ok((
                word_index,
                word,
                RawSubwordLane::Static(off.rem_euclid(4) as u32 / element_size),
            ));
        }
        let byte_index = self.emit_raw_byte_index(raw, extra_byte, instructions)?;
        let uint_ty = self.type_id(&LlType::Int(32))?;
        let word_index = self.fresh();
        let word_divisor = self.const_uint(4)?;
        instructions.push(Self::inst(
            Op::UDiv,
            Some(uint_ty),
            Some(word_index),
            vec![Operand::IdRef(byte_index), Operand::IdRef(word_divisor)],
        ));
        let word = self.emit_raw_word_load_at_index(raw, word_index, instructions)?;
        let masked = self.fresh();
        let lane_mask = self.const_uint(3)?;
        instructions.push(Self::inst(
            Op::BitwiseAnd,
            Some(uint_ty),
            Some(masked),
            vec![Operand::IdRef(byte_index), Operand::IdRef(lane_mask)],
        ));
        let lane = if element_size == 1 {
            masked
        } else {
            let lane = self.fresh();
            let shift = self.const_uint(element_size.trailing_zeros())?;
            instructions.push(Self::inst(
                Op::ShiftRightLogical,
                Some(uint_ty),
                Some(lane),
                vec![Operand::IdRef(masked), Operand::IdRef(shift)],
            ));
            lane
        };
        Ok((word_index, word, RawSubwordLane::Dynamic(lane)))
    }

    pub(in crate::native::emitter) fn emit_raw_u32_from_unaligned_bytes(
        &mut self,
        raw: &RawBufferOffset,
        extra_byte: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let low16 = self.emit_raw_u16_from_unaligned_bytes(raw, extra_byte, instructions)?;
        let high16 = self.emit_raw_u16_from_unaligned_bytes(raw, extra_byte + 2, instructions)?;
        let uint_ty = self.type_id(&LlType::Int(32))?;
        let low = self.fresh();
        instructions.push(Self::inst(
            Op::UConvert,
            Some(uint_ty),
            Some(low),
            vec![Operand::IdRef(low16)],
        ));
        let high = self.fresh();
        instructions.push(Self::inst(
            Op::UConvert,
            Some(uint_ty),
            Some(high),
            vec![Operand::IdRef(high16)],
        ));
        let shift = self.const_uint(16)?;
        let shifted_high = self.fresh();
        instructions.push(Self::inst(
            Op::ShiftLeftLogical,
            Some(uint_ty),
            Some(shifted_high),
            vec![Operand::IdRef(high), Operand::IdRef(shift)],
        ));
        let assembled = self.fresh();
        instructions.push(Self::inst(
            Op::BitwiseOr,
            Some(uint_ty),
            Some(assembled),
            vec![Operand::IdRef(low), Operand::IdRef(shifted_high)],
        ));
        Ok(assembled)
    }

    pub(in crate::native::emitter) fn emit_raw_word_store_unaligned_bytes(
        &mut self,
        raw: &RawBufferOffset,
        extra_byte: u64,
        value: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        self.emit_raw_byte_store_from_u32(value, raw, extra_byte, instructions)?;
        let uint_ty = self.type_id(&LlType::Int(32))?;
        for byte in 1..4 {
            let shifted = self.fresh();
            let shift = self.const_uint(byte * 8)?;
            instructions.push(Self::inst(
                Op::ShiftRightLogical,
                Some(uint_ty),
                Some(shifted),
                vec![Operand::IdRef(value), Operand::IdRef(shift)],
            ));
            self.emit_raw_byte_store_from_u32(
                shifted,
                raw,
                extra_byte + byte as u64,
                instructions,
            )?;
        }
        Ok(())
    }

    pub(in crate::native::emitter) fn emit_raw_byte_index(
        &mut self,
        raw: &RawBufferOffset,
        extra_byte: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        if raw.unmodelable {
            return Err("native emitter: raw buffer offset is not modelable".into());
        }
        let off = raw.const_off + extra_byte as i64;
        if off < 0 || off > u32::MAX as i64 {
            return Err(format!(
                "native emitter: raw buffer byte offset {off} is out of range"
            ));
        }
        let mut acc = self.const_uint(off as u32)?;
        let uint_ty = self.type_id(&LlType::Int(32))?;
        for (index, stride) in &raw.dyn_terms {
            if *stride < 0 {
                return Err(format!(
                    "native emitter: raw dynamic byte stride {stride} is not supported"
                ));
            }
            let mut term = self.raw_index_u32(index, instructions)?;
            if crate::env_vars::dbg_rawbyte() {
                eprintln!(
                    "[rawbyte] term_id={} ty={:?} value={:?} stride={}",
                    term, index.ty, index.value, stride
                );
            }
            if *stride != 1 {
                let mul = self.fresh();
                let factor = self.const_uint(*stride as u32)?;
                instructions.push(Self::inst(
                    Op::IMul,
                    Some(uint_ty),
                    Some(mul),
                    vec![Operand::IdRef(term), Operand::IdRef(factor)],
                ));
                term = mul;
            }
            let sum = self.fresh();
            instructions.push(Self::inst(
                Op::IAdd,
                Some(uint_ty),
                Some(sum),
                vec![Operand::IdRef(acc), Operand::IdRef(term)],
            ));
            acc = sum;
        }
        Ok(acc)
    }

    pub(in crate::native::emitter) fn emit_raw_word_index(
        &mut self,
        raw: &RawBufferOffset,
        extra_byte: u64,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let off = raw.const_off + extra_byte as i64;
        let const_word = off.div_euclid(4);
        if const_word < 0 || const_word > u32::MAX as i64 {
            return Err(format!(
                "native emitter: raw buffer word offset {const_word} is out of range"
            ));
        }
        let mut acc = self.const_uint(const_word as u32)?;
        let uint_ty = self.type_id(&LlType::Int(32))?;
        for (index, stride) in &raw.dyn_terms {
            if *stride < 0 {
                return Err(format!(
                    "native emitter: raw dynamic byte stride {stride} is not supported"
                ));
            }
            if !self.raw_dynamic_term_word_aligned(index, *stride) {
                return Err(format!(
                    "native emitter: raw dynamic byte stride {stride} is not word-aligned"
                ));
            }
            let mut term = self.raw_index_u32(index, instructions)?;
            if stride % 4 == 0 {
                let words = stride / 4;
                if words != 1 {
                    let mul = self.fresh();
                    let factor = self.const_uint(words as u32)?;
                    instructions.push(Self::inst(
                        Op::IMul,
                        Some(uint_ty),
                        Some(mul),
                        vec![Operand::IdRef(term), Operand::IdRef(factor)],
                    ));
                    term = mul;
                }
            } else {
                if *stride != 1 {
                    let mul = self.fresh();
                    let factor = self.const_uint(*stride as u32)?;
                    instructions.push(Self::inst(
                        Op::IMul,
                        Some(uint_ty),
                        Some(mul),
                        vec![Operand::IdRef(term), Operand::IdRef(factor)],
                    ));
                    term = mul;
                }
                let div = self.fresh();
                let divisor = self.const_uint(4)?;
                instructions.push(Self::inst(
                    Op::UDiv,
                    Some(uint_ty),
                    Some(div),
                    vec![Operand::IdRef(term), Operand::IdRef(divisor)],
                ));
                term = div;
            }
            let sum = self.fresh();
            instructions.push(Self::inst(
                Op::IAdd,
                Some(uint_ty),
                Some(sum),
                vec![Operand::IdRef(acc), Operand::IdRef(term)],
            ));
            acc = sum;
        }
        Ok(acc)
    }

    pub(in crate::native::emitter) fn emit_raw_word_index_for_access(
        &mut self,
        raw: &RawBufferOffset,
        extra_byte: u64,
        access_align: Option<u64>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        if self.raw_offset_aligned_to(raw, extra_byte, 4) {
            return self.emit_raw_word_index(raw, extra_byte, instructions);
        }
        if !self.raw_word_access_can_use_word_index(raw, extra_byte, access_align) {
            return Err("native emitter: raw word access is not word-aligned".into());
        }
        let byte_index = self.emit_raw_byte_index(raw, extra_byte, instructions)?;
        let uint_ty = self.type_id(&LlType::Int(32))?;
        let word_index = self.fresh();
        let divisor = self.const_uint(4)?;
        instructions.push(Self::inst(
            Op::UDiv,
            Some(uint_ty),
            Some(word_index),
            vec![Operand::IdRef(byte_index), Operand::IdRef(divisor)],
        ));
        Ok(word_index)
    }

    pub(in crate::native::emitter) fn raw_word_access_can_use_word_index(
        &self,
        raw: &RawBufferOffset,
        extra_byte: u64,
        access_align: Option<u64>,
    ) -> bool {
        self.raw_offset_aligned_to(raw, extra_byte, 4)
            || (access_align.unwrap_or(1) >= 4 && extra_byte.is_multiple_of(4))
    }
}
