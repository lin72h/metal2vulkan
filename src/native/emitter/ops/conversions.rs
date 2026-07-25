//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    /// The operand-resolved core of the int-convert handler — the M-A4 graph walk drives it from
    /// `TirInst.operands[0]` (source) + `TirInst.result_ty` (dest), see `emit_binary_float_op_resolved`.
    pub(in crate::native::emitter) fn emit_int_convert_resolved(
        &mut self,
        op: Op,
        src: TypedValue,
        dst_ty: LlType,
        name: String,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let src_ty = self.resolve_type(&src.ty)?;
        if is_bool_type(&src_ty) && is_integer_type(&dst_ty) {
            let result_type = self.type_id(&dst_ty)?;
            let result = self.result_id(&name, &dst_ty)?;
            let src_id = self.value_id_in(&src.value, &src.ty, instructions)?;
            let true_value = if op == Op::SConvert { -1 } else { 1 };
            let one = self.int_constant_like(&dst_ty, true_value)?;
            let zero = self.int_constant_like(&dst_ty, 0)?;
            instructions.push(Self::inst(
                Op::Select,
                Some(result_type),
                Some(result),
                vec![
                    Operand::IdRef(src_id),
                    Operand::IdRef(one),
                    Operand::IdRef(zero),
                ],
            ));
            self.record_int_alignment(&name, &dst_ty, 1);
            return Ok(());
        }
        if is_integer_type(&src_ty) && is_bool_type(&dst_ty) {
            let result_type = self.type_id(&dst_ty)?;
            let result = self.result_id(&name, &dst_ty)?;
            let src_id = self.value_id_in(&src.value, &src.ty, instructions)?;
            let zero = self.int_constant_like(&src_ty, 0)?;
            instructions.push(Self::inst(
                Op::INotEqual,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(src_id), Operand::IdRef(zero)],
            ));
            return Ok(());
        }
        if !int_convert_supported(&src_ty, &dst_ty) {
            return Err(format!(
                "native emitter: integer conversion needs matching integer shapes, got {src_ty:?} to {dst_ty:?}"
            ));
        }
        let result_type = self.type_id(&dst_ty)?;
        let result = self.result_id(&name, &dst_ty)?;
        let src_id = self.value_id_in(&src.value, &src.ty, instructions)?;
        // Nonstandard dest widths (`trunc i16 to i2`) emit as the legal container type (i8) via
        // `type_id`. A bare `OpUConvert` to i8 keeps the low *8* bits, but LLVM `i2` only has the
        // low 2 — mask so a subsequent `switch i2` with cases `i2 -1/-2/1` (encoded as 3/2/1)
        // matches. Unsigned convert (zext/trunc) only; signed narrow/widen of nonstandard widths
        // is a residual (not seen on the SkyLight reduction kernels).
        let dst_resolved = self.resolve_type(&dst_ty)?;
        if let Some(bits) = nonstandard_scalar_int_bits(&dst_resolved) {
            if op == Op::UConvert {
                let tmp = self.fresh();
                instructions.push(Self::inst(
                    op,
                    Some(result_type),
                    Some(tmp),
                    vec![Operand::IdRef(src_id)],
                ));
                let legal = spirv_int_width(bits)?;
                let mask = self.const_int(legal, (1u64 << bits) - 1)?;
                instructions.push(Self::inst(
                    Op::BitwiseAnd,
                    Some(result_type),
                    Some(result),
                    vec![Operand::IdRef(tmp), Operand::IdRef(mask)],
                ));
                self.record_int_alignment(&name, &dst_ty, self.int_value_alignment(&src.value));
                return Ok(());
            }
        }
        instructions.push(Self::inst(
            op,
            Some(result_type),
            Some(result),
            vec![Operand::IdRef(src_id)],
        ));
        self.record_int_alignment(&name, &dst_ty, self.int_value_alignment(&src.value));
        Ok(())
    }

    /// The operand-resolved core of `emit_float_convert` — the M-A4 graph walk drives it from
    /// `TirInst.operands[0]` (source) + `TirInst.result_ty` (dest).
    pub(in crate::native::emitter) fn emit_float_convert_resolved(
        &mut self,
        src: TypedValue,
        dst_ty: LlType,
        name: String,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let src_ty = self.resolve_type(&src.ty)?;
        if let (Some(src_lanes), Some(dst_lanes)) = (bfloat_lanes(&src_ty), float_lanes(&dst_ty)) {
            if src_lanes == dst_lanes {
                let src_id = self.value_id_in(&src.value, &src.ty, instructions)?;
                let result = self.result_id(&name, &dst_ty)?;
                return self.bfloat_bits_to_float_shaped(src_id, result, src_lanes, instructions);
            }
        }
        if let (Some(src_lanes), Some(dst_lanes)) = (float_lanes(&src_ty), bfloat_lanes(&dst_ty)) {
            if src_lanes == dst_lanes {
                let src_id = self.value_id_in(&src.value, &src.ty, instructions)?;
                let result = self.result_id(&name, &dst_ty)?;
                return self.emit_float_to_bfloat_bits_shaped(
                    src_id,
                    result,
                    src_lanes,
                    instructions,
                );
            }
        }
        if !float_convert_supported(&src_ty, &dst_ty) {
            return Err(format!(
                "native emitter: float conversion needs matching float/half shapes, got {src_ty:?} to {dst_ty:?}"
            ));
        }
        let result_type = self.type_id(&dst_ty)?;
        let result = self.result_id(&name, &dst_ty)?;
        let src_id = self.value_id_in(&src.value, &src.ty, instructions)?;
        instructions.push(Self::inst(
            Op::FConvert,
            Some(result_type),
            Some(result),
            vec![Operand::IdRef(src_id)],
        ));
        Ok(())
    }

    /// The operand-resolved core of `emit_int_to_float_convert` — the M-A4 graph walk drives it from
    /// `TirInst.operands[0]` (source) + `TirInst.result_ty` (dest). Note the text entry above sources
    /// its operand via `parse_typed_value` (not the tir carrier), so the graph walk is the FIRST tir
    /// source for `sitofp`/`uitofp` operands; BC confirms it is byte-identical (tir operand soundness).
    pub(in crate::native::emitter) fn emit_int_to_float_convert_resolved(
        &mut self,
        op: Op,
        src: TypedValue,
        dst_ty: LlType,
        name: String,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let src_ty = self.resolve_type(&src.ty)?;
        if !int_to_float_convert_supported(&src_ty, &dst_ty) {
            return Err(format!(
                "native emitter: integer-to-float conversion needs matching integer-to-float shapes, got {src_ty:?} to {dst_ty:?}"
            ));
        }
        let result_type = self.type_id(&dst_ty)?;
        let result = self.result_id(&name, &dst_ty)?;
        let src_id = self.value_id_in(&src.value, &src.ty, instructions)?;
        instructions.push(Self::inst(
            op,
            Some(result_type),
            Some(result),
            vec![Operand::IdRef(src_id)],
        ));
        Ok(())
    }

    /// Like [`Self::bfloat_bits_to_float_shaped`] but allocating the result id.
    pub(in crate::native::emitter) fn bfloat_bits_to_float_shaped_id(
        &mut self,
        src_id: Word,
        n: u32,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let result = self.fresh();
        self.bfloat_bits_to_float_shaped(src_id, result, n, instructions)?;
        Ok(result)
    }

    /// Widen a bf16 bit pattern (its `Int(16)` storage, scalar or `n`-lane vector) to f32: bf16 is the
    /// top 16 bits of an f32, so zero-extend to i32, shift left 16, then reinterpret as float. Shaped by
    /// `n` (1 = scalar), so `Vector(BFloat, n)` arithmetic can round-trip through `Vector(Float, n)`.
    pub(in crate::native::emitter) fn bfloat_bits_to_float_shaped(
        &mut self,
        src_id: Word,
        result: Word,
        n: u32,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let i32_ty = self.type_id(&shaped_type(LlType::Int(32), n))?;
        let float_ty = self.type_id(&shaped_type(LlType::Float, n))?;
        let widened = self.fresh();
        instructions.push(Self::inst(
            Op::UConvert,
            Some(i32_ty),
            Some(widened),
            vec![Operand::IdRef(src_id)],
        ));
        let shift = self.const_uint_shaped(16, n)?;
        let shifted = self.fresh();
        instructions.push(Self::inst(
            Op::ShiftLeftLogical,
            Some(i32_ty),
            Some(shifted),
            vec![Operand::IdRef(widened), Operand::IdRef(shift)],
        ));
        instructions.push(Self::inst(
            Op::Bitcast,
            Some(float_ty),
            Some(result),
            vec![Operand::IdRef(shifted)],
        ));
        Ok(())
    }

    /// Narrow an f32 (scalar or `n`-lane vector) to a bf16 bit pattern (its `Int(16)` storage).
    /// LLVM `fptrunc float to bfloat` rounds to nearest-even, so add the bf16 rounding bias before
    /// taking the top 16 bits. Shaped by `n` (1 = scalar) so `Vector(Float, n)` results can be
    /// re-narrowed to `Vector(BFloat, n)` storage.
    pub(in crate::native::emitter) fn emit_float_to_bfloat_bits_shaped(
        &mut self,
        src_id: Word,
        result: Word,
        n: u32,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let i32_ty = self.type_id(&shaped_type(LlType::Int(32), n))?;
        let result_type = self.type_id(&shaped_type(LlType::BFloat, n))?;
        let bits = self.fresh();
        instructions.push(Self::inst(
            Op::Bitcast,
            Some(i32_ty),
            Some(bits),
            vec![Operand::IdRef(src_id)],
        ));
        let shift = self.const_uint_shaped(16, n)?;
        let high = self.fresh();
        instructions.push(Self::inst(
            Op::ShiftRightLogical,
            Some(i32_ty),
            Some(high),
            vec![Operand::IdRef(bits), Operand::IdRef(shift)],
        ));
        let one = self.const_uint_shaped(1, n)?;
        let lsb = self.fresh();
        instructions.push(Self::inst(
            Op::BitwiseAnd,
            Some(i32_ty),
            Some(lsb),
            vec![Operand::IdRef(high), Operand::IdRef(one)],
        ));
        let bias_base = self.const_uint_shaped(0x7fff, n)?;
        let bias = self.fresh();
        instructions.push(Self::inst(
            Op::IAdd,
            Some(i32_ty),
            Some(bias),
            vec![Operand::IdRef(bias_base), Operand::IdRef(lsb)],
        ));
        let rounded = self.fresh();
        instructions.push(Self::inst(
            Op::IAdd,
            Some(i32_ty),
            Some(rounded),
            vec![Operand::IdRef(bits), Operand::IdRef(bias)],
        ));
        let shifted = self.fresh();
        instructions.push(Self::inst(
            Op::ShiftRightLogical,
            Some(i32_ty),
            Some(shifted),
            vec![Operand::IdRef(rounded), Operand::IdRef(shift)],
        ));
        let narrowed = self.select_canonical_bfloat_nan_bits(bits, shifted, n, instructions)?;
        instructions.push(Self::inst(
            Op::UConvert,
            Some(result_type),
            Some(result),
            vec![Operand::IdRef(narrowed)],
        ));
        Ok(())
    }

    fn select_canonical_bfloat_nan_bits(
        &mut self,
        bits: Word,
        narrowed: Word,
        n: u32,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let i32_ty = self.type_id(&shaped_type(LlType::Int(32), n))?;
        let bool_ty = self.type_id(&shaped_type(LlType::Bool, n))?;
        let exp_mask = self.const_uint_shaped(0x7f80_0000, n)?;
        let mant_mask = self.const_uint_shaped(0x007f_ffff, n)?;
        let zero = self.const_uint_shaped(0, n)?;

        let exp_bits = self.fresh();
        instructions.push(Self::inst(
            Op::BitwiseAnd,
            Some(i32_ty),
            Some(exp_bits),
            vec![Operand::IdRef(bits), Operand::IdRef(exp_mask)],
        ));
        let exp_all_ones = self.fresh();
        instructions.push(Self::inst(
            Op::IEqual,
            Some(bool_ty),
            Some(exp_all_ones),
            vec![Operand::IdRef(exp_bits), Operand::IdRef(exp_mask)],
        ));

        let mant_bits = self.fresh();
        instructions.push(Self::inst(
            Op::BitwiseAnd,
            Some(i32_ty),
            Some(mant_bits),
            vec![Operand::IdRef(bits), Operand::IdRef(mant_mask)],
        ));
        let mant_nonzero = self.fresh();
        instructions.push(Self::inst(
            Op::INotEqual,
            Some(bool_ty),
            Some(mant_nonzero),
            vec![Operand::IdRef(mant_bits), Operand::IdRef(zero)],
        ));

        let is_nan = self.fresh();
        instructions.push(Self::inst(
            Op::LogicalAnd,
            Some(bool_ty),
            Some(is_nan),
            vec![Operand::IdRef(exp_all_ones), Operand::IdRef(mant_nonzero)],
        ));

        let canonical_nan = self.const_uint_shaped(0x7fc0, n)?;
        let selected = self.fresh();
        instructions.push(Self::inst(
            Op::Select,
            Some(i32_ty),
            Some(selected),
            vec![
                Operand::IdRef(is_nan),
                Operand::IdRef(canonical_nan),
                Operand::IdRef(narrowed),
            ],
        ));
        Ok(selected)
    }

    pub(in crate::native::emitter) fn emit_i32_to_v4i8(
        &mut self,
        src: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let i32_ty = self.type_id(&LlType::Int(32))?;
        let i8_ty = self.type_id(&LlType::Int(8))?;
        let v4i8_ty = self.type_id(&LlType::Vector(Box::new(LlType::Int(8)), 4))?;
        let mask = self.const_uint(0xff)?;
        let mut lanes = Vec::with_capacity(4);
        for lane in 0..4u32 {
            let shifted = if lane == 0 {
                src
            } else {
                let shift = self.const_uint(lane * 8)?;
                let id = self.fresh();
                instructions.push(Self::inst(
                    Op::ShiftRightLogical,
                    Some(i32_ty),
                    Some(id),
                    vec![Operand::IdRef(src), Operand::IdRef(shift)],
                ));
                id
            };
            let masked = self.fresh();
            instructions.push(Self::inst(
                Op::BitwiseAnd,
                Some(i32_ty),
                Some(masked),
                vec![Operand::IdRef(shifted), Operand::IdRef(mask)],
            ));
            let narrow = self.fresh();
            instructions.push(Self::inst(
                Op::UConvert,
                Some(i8_ty),
                Some(narrow),
                vec![Operand::IdRef(masked)],
            ));
            lanes.push(narrow);
        }
        let result = self.fresh();
        instructions.push(Self::inst(
            Op::CompositeConstruct,
            Some(v4i8_ty),
            Some(result),
            lanes.into_iter().map(Operand::IdRef).collect(),
        ));
        Ok(result)
    }
}
