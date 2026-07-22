//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    /// When a select arm is already a pointer to scalar `pointee`, fold `indices` through
    /// `source_ty` into one element offset and emit `OpPtrAccessChain %ptr_S %base %flat`.
    /// Returns `None` if linearization is unsupported (caller keeps the structured path).
    pub(in crate::native::emitter) fn emit_flattened_scalar_arm_access_chain(
        &mut self,
        ptr_type: Word,
        base: Word,
        source_ty: &LlType,
        pointee: &LlType,
        indices: &[TypedValue],
        instructions: &mut Vec<Instruction>,
    ) -> Result<Option<Word>, String> {
        let Some(flat) =
            self.linearize_gep_indices_in_scalar_units(source_ty, indices, pointee, instructions)?
        else {
            return Ok(None);
        };
        let flat_id = self.value_id(&flat.value, &flat.ty)?;
        let result = self.fresh();
        instructions.push(Self::inst(
            Op::PtrAccessChain,
            Some(ptr_type),
            Some(result),
            vec![Operand::IdRef(base), Operand::IdRef(flat_id)],
        ));
        Ok(Some(result))
    }

    /// Look up the SPIR-V result type of a previously emitted SSA id (pending block instructions
    /// first, then the module). Used to detect a select arm already typed as the destination
    /// scalar pointer — `pointer_pointees` can disagree after network unify.
    pub(in crate::native::emitter) fn word_result_type(
        &self,
        id: Word,
        pending: &[Instruction],
    ) -> Option<Word> {
        for inst in pending.iter().rev() {
            if inst.result_id == Some(id) {
                return inst.result_type;
            }
        }
        for function in &self.module.functions {
            for inst in &function.parameters {
                if inst.result_id == Some(id) {
                    return inst.result_type;
                }
            }
            for block in &function.blocks {
                for inst in &block.instructions {
                    if inst.result_id == Some(id) {
                        return inst.result_type;
                    }
                }
            }
        }
        for inst in &self.module.types_global_values {
            if inst.result_id == Some(id) {
                return inst.result_type;
            }
        }
        // Globals / params may live only in the emitter side-tables until the module is finalized.
        for (gid, _) in self.global_values.values() {
            if *gid == id {
                // Global's SPIR-V pointer type is not stored separately; fall through.
                break;
            }
        }
        None
    }

    /// Whether a pointer SSA id already points at a scalar (so a multi-index structured GEP would
    /// over-index). Ground truth is the Word's SPIR-V `OpTypePointer` pointee when findable; falls
    /// back to modeled `pointer_pointees` / `base_pointee` when the defining instruction is not
    /// visible yet.
    pub(in crate::native::emitter) fn pointer_id_already_at_scalar(
        &self,
        base_id: Word,
        base_value: &LlValue,
        want_scalar: &LlType,
        want_ptr_type: Word,
        pending: &[Instruction],
        base_pointee: Option<&LlType>,
    ) -> bool {
        if !is_scalar_pointee(want_scalar) {
            return false;
        }
        if let Some(base_ptr_ty) = self.word_result_type(base_id, pending) {
            if base_ptr_ty == want_ptr_type {
                return true;
            }
            // SPIR-V ground truth: if the base pointer's pointee is already a scalar type, any
            // multi-index structured chain over-indexes — flatten regardless of TypePointer id
            // equality (interning can leave distinct Words for the same shape in edge cases).
            if let Some(pointee_ty) = self.type_pointer_pointee_id(base_ptr_ty) {
                if self.spirv_type_is_scalar(pointee_ty) {
                    return true;
                }
            }
        }
        if let Some(bp) = base_pointee {
            if let Ok(bp) = self.resolve_type(bp) {
                if types_compatible(&bp, want_scalar) {
                    return true;
                }
            }
        }
        if let LlValue::Local(name) = base_value {
            if let Some(pp) = self.pointer_pointees.get(name) {
                if let Ok(pp) = self.resolve_type(pp) {
                    if types_compatible(&pp, want_scalar) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Pointee type id of an `OpTypePointer` Word, or `None` if `ptr_ty` is not a pointer type.
    pub(in crate::native::emitter) fn type_pointer_pointee_id(&self, ptr_ty: Word) -> Option<Word> {
        for inst in &self.module.types_global_values {
            if inst.class.opcode == Op::TypePointer && inst.result_id == Some(ptr_ty) {
                if let Some(Operand::IdRef(p)) = inst.operands.get(1) {
                    return Some(*p);
                }
            }
        }
        None
    }

    /// Whether a SPIR-V type id is a scalar (float/int/bool), not a composite.
    pub(in crate::native::emitter) fn spirv_type_is_scalar(&self, ty: Word) -> bool {
        for inst in &self.module.types_global_values {
            if inst.result_id == Some(ty) {
                return matches!(
                    inst.class.opcode,
                    Op::TypeFloat | Op::TypeInt | Op::TypeBool
                );
            }
        }
        false
    }

    /// Fold structured GEP `indices` through `source_ty` into a single offset in units of `scalar`.
    /// Returns `None` when the layout cannot be linearized (dynamic struct index, size mismatch,
    /// non-divisible stride) so the caller can leave the chain alone rather than miscompile.
    ///
    /// Index list is the AIR/LLVM GEP list (leading const-0 dropped via [`gep_spirv_indices`], same
    /// as the structured emit path). First remaining index strides by `sizeof(source_ty)`; further
    /// indices walk members/elements.
    pub(in crate::native::emitter) fn linearize_gep_indices_in_scalar_units(
        &mut self,
        source_ty: &LlType,
        indices: &[TypedValue],
        scalar: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Option<TypedValue>, String> {
        let indices = match gep_spirv_indices(indices) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        if indices.is_empty() {
            return Ok(None);
        }
        let source = self.resolve_type(source_ty)?;
        let scalar = self.resolve_type(scalar)?;
        if !is_scalar_pointee(&scalar) {
            return Ok(None);
        }
        let (scalar_size, _) = match self.raw_type_size_align(&scalar) {
            Ok(sa) if sa.0 > 0 => sa,
            _ => return Ok(None),
        };

        let mut total: Option<TypedValue> = None;
        let mut cur = source;
        for (i, idx) in indices.iter().enumerate() {
            let contrib = if i == 0 {
                // Leading index: element stride over the whole source type.
                let (stride_bytes, _) = match self.raw_type_size_align(&cur) {
                    Ok(sa) => sa,
                    Err(_) => return Ok(None),
                };
                if stride_bytes % scalar_size != 0 {
                    return Ok(None);
                }
                let stride_units = (stride_bytes / scalar_size) as u32;
                self.scale_gep_index(idx, stride_units, "sel.flat0", instructions)?
            } else {
                match &cur {
                    LlType::Struct(fields) => {
                        let Some(member_i) = const_index(Some(idx)) else {
                            return Ok(None);
                        };
                        let mut offset = 0u64;
                        let mut member_ty = None;
                        for (fi, field) in fields.iter().enumerate() {
                            let field = match self.resolve_type(field) {
                                Ok(t) => t,
                                Err(_) => return Ok(None),
                            };
                            let (fsz, falign) = match self.raw_type_size_align(&field) {
                                Ok(sa) => sa,
                                Err(_) => return Ok(None),
                            };
                            let align = falign.max(1);
                            offset = offset.div_ceil(align) * align;
                            if fi as u32 == member_i {
                                member_ty = Some(field);
                                break;
                            }
                            offset += fsz;
                        }
                        let Some(member_ty) = member_ty else {
                            return Ok(None);
                        };
                        if !offset.is_multiple_of(scalar_size) {
                            return Ok(None);
                        }
                        let units = offset / scalar_size;
                        cur = member_ty;
                        TypedValue {
                            ty: LlType::Int(32),
                            value: LlValue::Int(units),
                        }
                    }
                    LlType::Array(elem, _) | LlType::Vector(elem, _) => {
                        let elem = match self.resolve_type(elem) {
                            Ok(t) => t,
                            Err(_) => return Ok(None),
                        };
                        let (stride_bytes, _) = match self.raw_type_size_align(&elem) {
                            Ok(sa) => sa,
                            Err(_) => return Ok(None),
                        };
                        if stride_bytes % scalar_size != 0 {
                            return Ok(None);
                        }
                        let stride_units = (stride_bytes / scalar_size) as u32;
                        let scaled =
                            self.scale_gep_index(idx, stride_units, "sel.flati", instructions)?;
                        cur = elem;
                        scaled
                    }
                    _ => return Ok(None),
                }
            };
            total = Some(match total {
                None => contrib,
                Some(acc) => self.combine_gep_indices(&acc, &contrib, instructions)?,
            });
        }
        // After member walks (`indices.len() > 1`), the walked type must be the result scalar.
        // A leading-only index list leaves `cur == source` (element stride only) — still a valid
        // flat offset when the arm is already at scalar depth for a same-scalar source (rare).
        if indices.len() > 1 {
            let landed = match self.resolve_type(&cur) {
                Ok(t) => t,
                Err(_) => return Ok(None),
            };
            if !types_compatible(&landed, &scalar) {
                return Ok(None);
            }
        }
        Ok(total)
    }
}
