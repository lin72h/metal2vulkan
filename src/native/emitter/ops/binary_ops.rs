//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    /// The operand-resolved core of `emit_binary_float_op`: emits the float op given its already-typed
    /// `(lhs, rhs)`. Split out so the R3 STRUCTURAL / M-A4 graph walk (`emit_body_inst`) can drive it
    /// straight from `TirInst.operands` without going through the text-sourcing `binary_operands` — the
    /// text entry above and the structured caller produce byte-identical output (same operands, same Op).
    pub(in crate::native::emitter) fn emit_binary_float_op_resolved(
        &mut self,
        op: Op,
        lhs: TypedValue,
        rhs: TypedValue,
        name: String,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let result_ty = self.resolve_type(&lhs.ty)?;
        // A vector wider than 4 lanes is typed as an OpTypeArray (SPIR-V/Vulkan Logical vectors cap at
        // 4 components), and SPIR-V arithmetic ops reject an array Result Type ("Expected floating
        // scalar or vector type"). Scalarize elementwise: extract each lane, apply the op per lane
        // (delegating a bf16 element to the same widen/op/narrow path used for the scalar case), then
        // rebuild the array. Structural (keyed on lane count > 4), never name-based; floor-safe — a
        // wide-vector arithmetic op emits an invalid module today, so this only turns invalid into valid.
        if let LlType::Vector(elem, n) = &result_ty {
            if *n > 4 {
                let elem_ty = (**elem).clone();
                let elem_type_id = self.type_id(&elem_ty)?;
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(&name, &result_ty)?;
                let lhs_id = self.value_id_in(&lhs.value, &lhs.ty, instructions)?;
                let rhs_id = self.value_id_in(&rhs.value, &rhs.ty, instructions)?;
                let is_bf = bfloat_lanes(&elem_ty).is_some();
                let mut lanes = Vec::with_capacity(*n as usize);
                for i in 0..*n {
                    let a = self.fresh();
                    instructions.push(Self::inst(
                        Op::CompositeExtract,
                        Some(elem_type_id),
                        Some(a),
                        vec![Operand::IdRef(lhs_id), Operand::LiteralBit32(i)],
                    ));
                    let b = self.fresh();
                    instructions.push(Self::inst(
                        Op::CompositeExtract,
                        Some(elem_type_id),
                        Some(b),
                        vec![Operand::IdRef(rhs_id), Operand::LiteralBit32(i)],
                    ));
                    let r = if is_bf {
                        // bf16 element: widen both lanes to f32, op in float, narrow back to bf16 bits.
                        let a_f = self.bfloat_bits_to_float_shaped_id(a, 1, instructions)?;
                        let b_f = self.bfloat_bits_to_float_shaped_id(b, 1, instructions)?;
                        let float_ty = self.type_id(&LlType::Float)?;
                        let r_f = self.fresh();
                        instructions.push(Self::inst(
                            op,
                            Some(float_ty),
                            Some(r_f),
                            vec![Operand::IdRef(a_f), Operand::IdRef(b_f)],
                        ));
                        let r_bits = self.fresh();
                        self.emit_float_to_bfloat_bits_shaped(r_f, r_bits, 1, instructions)?;
                        r_bits
                    } else {
                        let r = self.fresh();
                        instructions.push(Self::inst(
                            op,
                            Some(elem_type_id),
                            Some(r),
                            vec![Operand::IdRef(a), Operand::IdRef(b)],
                        ));
                        r
                    };
                    lanes.push(Operand::IdRef(r));
                }
                instructions.push(Self::inst(
                    Op::CompositeConstruct,
                    Some(result_type),
                    Some(result),
                    lanes,
                ));
                return Ok(());
            }
        }
        // bf16 arithmetic (scalar OR `Vector(BFloat, n)`) has no native float op — its operands are u16
        // storage. Widen each to f32 (shaped by lane count), do the op in float, narrow the result back
        // to bf16 bits. The scalar-only guard here previously fell through for a bf16 vector, emitting a
        // type-invalid `OpFAdd` on the u16 vector storage.
        if let Some(n) = bfloat_lanes(&result_ty) {
            let result = self.result_id(&name, &result_ty)?;
            let lhs_id = self.value_id_in(&lhs.value, &lhs.ty, instructions)?;
            let rhs_id = self.value_id_in(&rhs.value, &rhs.ty, instructions)?;
            let lhs_f32 = self.bfloat_bits_to_float_shaped_id(lhs_id, n, instructions)?;
            let rhs_f32 = self.bfloat_bits_to_float_shaped_id(rhs_id, n, instructions)?;
            let float_ty = self.type_id(&shaped_type(LlType::Float, n))?;
            let result_f32 = self.fresh();
            instructions.push(Self::inst(
                op,
                Some(float_ty),
                Some(result_f32),
                vec![Operand::IdRef(lhs_f32), Operand::IdRef(rhs_f32)],
            ));
            return self.emit_float_to_bfloat_bits_shaped(result_f32, result, n, instructions);
        }
        let result_type = self.type_id(&result_ty)?;
        let result = self.result_id(&name, &result_ty)?;
        let lhs_id = self.value_id_in(&lhs.value, &lhs.ty, instructions)?;
        let rhs_id = self.value_id_in(&rhs.value, &rhs.ty, instructions)?;
        instructions.push(Self::inst(
            op,
            Some(result_type),
            Some(result),
            vec![Operand::IdRef(lhs_id), Operand::IdRef(rhs_id)],
        ));
        Ok(())
    }

    /// The operand-resolved core of the binary-int-op handler (see `emit_binary_float_op_resolved` for why
    /// this is split out — the M-A4 graph walk drives it straight from `TirInst.operands`).
    pub(in crate::native::emitter) fn emit_binary_int_op_resolved(
        &mut self,
        op: Op,
        lhs: TypedValue,
        rhs: TypedValue,
        name: String,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let result_ty = self.resolve_type(&lhs.ty)?;
        let int_alignment = match op {
            Op::IAdd | Op::ISub => add_int_alignment(
                self.int_value_alignment(&lhs.value),
                self.int_value_alignment(&rhs.value),
            ),
            Op::IMul => mul_int_alignment(
                self.int_value_alignment(&lhs.value),
                self.int_value_alignment(&rhs.value),
            ),
            Op::ShiftLeftLogical => {
                shift_left_int_alignment(self.int_value_alignment(&lhs.value), &rhs.value)
            }
            _ => 1,
        };
        if let Some(logical_op) = logical_op_for_bitwise(op, &result_ty) {
            let result_type = self.type_id(&result_ty)?;
            let result = self.result_id(&name, &result_ty)?;
            let lhs_id = self.value_id_in(&lhs.value, &lhs.ty, instructions)?;
            let rhs_id = self.value_id_in(&rhs.value, &rhs.ty, instructions)?;
            instructions.push(Self::inst(
                logical_op,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(lhs_id), Operand::IdRef(rhs_id)],
            ));
            return Ok(());
        }
        if !is_integer_type(&result_ty) {
            return Err(format!(
                "native emitter: {op:?} currently supports scalar/vector integer types, got {result_ty:?}"
            ));
        }
        // Vulkan vectors have at most four components. AIR retains wider LLVM integer vectors,
        // which `type_id` represents as arrays; scalarize their elementwise operation instead of
        // applying an integer opcode to an aggregate result type.
        if let LlType::Vector(element, lanes) = &result_ty {
            if *lanes > 4 {
                let element_ty = self.resolve_type(element)?;
                let element_type = self.type_id(&element_ty)?;
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(&name, &result_ty)?;
                let lhs_id = self.value_id_in(&lhs.value, &lhs.ty, instructions)?;
                let rhs_id = self.value_id_in(&rhs.value, &rhs.ty, instructions)?;
                let mut values = Vec::with_capacity(*lanes as usize);
                for lane in 0..*lanes {
                    let left = self.fresh();
                    instructions.push(Self::inst(
                        Op::CompositeExtract,
                        Some(element_type),
                        Some(left),
                        vec![Operand::IdRef(lhs_id), Operand::LiteralBit32(lane)],
                    ));
                    let right = self.fresh();
                    instructions.push(Self::inst(
                        Op::CompositeExtract,
                        Some(element_type),
                        Some(right),
                        vec![Operand::IdRef(rhs_id), Operand::LiteralBit32(lane)],
                    ));
                    let value = self.fresh();
                    instructions.push(Self::inst(
                        op,
                        Some(element_type),
                        Some(value),
                        vec![Operand::IdRef(left), Operand::IdRef(right)],
                    ));
                    values.push(Operand::IdRef(value));
                }
                instructions.push(Self::inst(
                    Op::CompositeConstruct,
                    Some(result_type),
                    Some(result),
                    values,
                ));
                self.record_int_alignment(&name, &result_ty, int_alignment);
                return Ok(());
            }
        }
        let result_type = self.type_id(&result_ty)?;
        let result = self.result_id(&name, &result_ty)?;
        let lhs_id = self.value_id_in(&lhs.value, &lhs.ty, instructions)?;
        let rhs_id = self.value_id_in(&rhs.value, &rhs.ty, instructions)?;
        // Wrap nonstandard-width arithmetic at the logical bit width (LLVM `add i2` wraps mod 4).
        if let Some(bits) = nonstandard_scalar_int_bits(&result_ty) {
            let tmp = self.fresh();
            instructions.push(Self::inst(
                op,
                Some(result_type),
                Some(tmp),
                vec![Operand::IdRef(lhs_id), Operand::IdRef(rhs_id)],
            ));
            let legal = spirv_int_width(bits)?;
            let mask = self.const_int(legal, (1u64 << bits) - 1)?;
            instructions.push(Self::inst(
                Op::BitwiseAnd,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(tmp), Operand::IdRef(mask)],
            ));
        } else {
            instructions.push(Self::inst(
                op,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(lhs_id), Operand::IdRef(rhs_id)],
            ));
        }
        self.record_int_alignment(&name, &result_ty, int_alignment);
        Ok(())
    }

    /// The operand-resolved core of `emit_signed_binary_int_op` (M-A4 graph-walk entry; see
    /// `emit_binary_float_op_resolved`).
    pub(in crate::native::emitter) fn emit_signed_binary_int_op_resolved(
        &mut self,
        op: Op,
        lhs: TypedValue,
        rhs: TypedValue,
        name: String,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let result_ty = self.resolve_type(&lhs.ty)?;
        if !is_integer_type(&result_ty) {
            return Err(format!(
                "native emitter: {op:?} currently supports scalar/vector integer types, got {result_ty:?}"
            ));
        }
        let signed_type = self.signed_int_type_id(&result_ty)?;
        let canonical_type = self.type_id(&result_ty)?;
        let result = self.result_id(&name, &result_ty)?;
        let lhs_id = self.value_id_in(&lhs.value, &lhs.ty, instructions)?;
        let rhs_id = self.value_id_in(&rhs.value, &rhs.ty, instructions)?;
        let lhs_signed = self.fresh();
        instructions.push(Self::inst(
            Op::Bitcast,
            Some(signed_type),
            Some(lhs_signed),
            vec![Operand::IdRef(lhs_id)],
        ));
        let rhs_signed = self.fresh();
        instructions.push(Self::inst(
            Op::Bitcast,
            Some(signed_type),
            Some(rhs_signed),
            vec![Operand::IdRef(rhs_id)],
        ));
        let signed_result = if op == Op::SRem {
            // NVIDIA's Vulkan stack has produced non-LLVM results for OpSRem on negative operands.
            // Emit the LLVM remainder identity directly: x - (x / y) * y.
            let quotient = self.fresh();
            instructions.push(Self::inst(
                Op::SDiv,
                Some(signed_type),
                Some(quotient),
                vec![Operand::IdRef(lhs_signed), Operand::IdRef(rhs_signed)],
            ));
            let product = self.fresh();
            instructions.push(Self::inst(
                Op::IMul,
                Some(signed_type),
                Some(product),
                vec![Operand::IdRef(quotient), Operand::IdRef(rhs_signed)],
            ));
            let remainder = self.fresh();
            instructions.push(Self::inst(
                Op::ISub,
                Some(signed_type),
                Some(remainder),
                vec![Operand::IdRef(lhs_signed), Operand::IdRef(product)],
            ));
            remainder
        } else {
            let signed_result = self.fresh();
            instructions.push(Self::inst(
                op,
                Some(signed_type),
                Some(signed_result),
                vec![Operand::IdRef(lhs_signed), Operand::IdRef(rhs_signed)],
            ));
            signed_result
        };
        instructions.push(Self::inst(
            Op::Bitcast,
            Some(canonical_type),
            Some(result),
            vec![Operand::IdRef(signed_result)],
        ));
        self.record_int_alignment(&name, &result_ty, 1);
        Ok(())
    }
}
