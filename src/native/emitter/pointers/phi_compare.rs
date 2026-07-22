//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    /// The operand-resolved core of the `phi` handler. Driven from the parsed (unresolved) phi type + the
    /// `(value, predecessor-label)` pairs — either re-parsed by the text entry above or carried typed on
    /// `TirInst.phi_incoming`. Byte-identical either way: same `parse_phi` output, and the incoming VALUES
    /// are then overlaid from the typed graph (`phi_incoming_values`), labels kept from the pairs.
    pub(in crate::native::emitter) fn emit_phi_resolved(
        &mut self,
        name: String,
        phi_ty: &LlType,
        parsed_incoming: Vec<(LlValue, String)>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let result_ty = self.resolve_type(phi_ty)?;
        // R3: source the incoming VALUES from the typed graph (labels stay from the parsed pairs).
        // Byte-identical to the parsed values by tir's phi-operand soundness; falls back on any mismatch.
        let incoming = self.phi_incoming_values(&name, parsed_incoming);
        if self.emit_raw_pointer_phi(&name, &incoming, &result_ty, instructions)? {
            return Ok(());
        }
        if self.emit_unmodeled_pointer_phi(&name, &incoming, &result_ty, instructions)? {
            return Ok(());
        }
        let pointer_meta = self.pointer_merge_meta(
            &incoming.iter().map(|(value, _)| value).collect::<Vec<_>>(),
            &result_ty,
        )?;
        let int_alignment = matches!(result_ty, LlType::Int(_))
            .then(|| self.merged_int_alignment(incoming.iter().map(|(value, _)| value.clone())));
        let pointer_provenance =
            self.emit_pointer_phi_provenance(&name, &incoming, instructions)?;
        self.emit_pointer_nullness_phi(&name, &incoming, &result_ty, instructions)?;
        let result_type = self.pointer_aware_type_id(&result_ty, pointer_meta.as_ref())?;
        let result = self.result_id(&name, &result_ty)?;
        let mut ops = Vec::new();
        let mut seen_incoming: HashMap<Word, Word> = HashMap::new();
        for (value, label) in incoming {
            let value_id = if let Some(meta) = pointer_meta.as_ref() {
                self.pointer_phi_value_id(
                    &value,
                    &result_ty,
                    meta,
                    pointer_provenance.as_ref(),
                    instructions,
                )?
            } else {
                self.phi_value_id(&value, &result_ty, instructions)?
            };
            let label_id = self.label_id(&label)?;
            if let Some(existing) = seen_incoming.insert(label_id, value_id) {
                if existing != value_id {
                    return Err(format!(
                        "native emitter: phi has multiple values from predecessor {}",
                        label
                    ));
                }
                continue;
            }
            ops.push(Operand::IdRef(value_id));
            ops.push(Operand::IdRef(label_id));
        }
        instructions.push(Self::inst(Op::Phi, Some(result_type), Some(result), ops));
        if let Some(meta) = pointer_meta {
            self.record_pointer_meta(name.clone(), meta);
        }
        if let Some(provenance) = pointer_provenance {
            self.gep_provenance.insert(name.clone(), provenance);
        }
        if let Some(alignment) = int_alignment {
            self.record_int_alignment(&name, &result_ty, alignment);
        }
        Ok(())
    }

    pub(in crate::native::emitter) fn pointer_phi_value_id(
        &mut self,
        value: &LlValue,
        result_ty: &LlType,
        meta: &PointerMeta,
        result_provenance: Option<&GepProvenance>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let Some(pointee) = meta.pointee.as_ref() else {
            return self.phi_value_id(value, result_ty, instructions);
        };
        // A `null`/`undef` incoming must carry the phi's resolved pointer type (storage + pointee), the
        // same the select path already does — otherwise it is emitted with the generic default type and
        // mismatches the phi result (the cross-storage `_ptr_UniformConstant_uchar` validation reject).
        if let Some(id) = self.typed_null_or_undef_pointer_id(value, meta.storage, pointee)? {
            return Ok(id);
        }
        let Some(template) = result_provenance else {
            if let LlValue::Local(name) = value {
                if meta.storage == StorageClass::Workgroup && self.param_values.contains(name) {
                    let ptr_ty = self.ptr_type_id(meta.storage, pointee)?;
                    let base = self.value_id(value, result_ty)?;
                    let zero = self.const_uint(0)?;
                    let result = self.fresh();
                    instructions.push(Self::inst(
                        Op::InBoundsAccessChain,
                        Some(ptr_ty),
                        Some(result),
                        vec![Operand::IdRef(base), Operand::IdRef(zero)],
                    ));
                    return Ok(result);
                }
            }
            return self.phi_value_id(value, result_ty, instructions);
        };
        let LlValue::Local(name) = value else {
            return self.phi_value_id(value, result_ty, instructions);
        };
        if self
            .values
            .get(name)
            .is_none_or(|(id, _)| *id != template.root)
        {
            return self.phi_value_id(value, result_ty, instructions);
        }
        let Some(index_ty) = template.indices.first().map(|index| index.ty.clone()) else {
            return self.phi_value_id(value, result_ty, instructions);
        };
        let Some(provenance) =
            self.provenance_for_pointer_value(value, Some(template), Some(&index_ty))?
        else {
            return self.phi_value_id(value, result_ty, instructions);
        };
        if !compatible_pointer_provenance(template, &provenance) {
            return self.phi_value_id(value, result_ty, instructions);
        }
        let ptr_ty = self.ptr_type_id(meta.storage, pointee)?;
        let result = self.fresh();
        let mut ops = vec![Operand::IdRef(provenance.root)];
        let mut indices = gep_spirv_indices(&provenance.indices)?;
        if indices.is_empty() && !provenance.indices.is_empty() {
            indices = provenance.indices.clone();
        }
        for idx in indices {
            ops.push(Operand::IdRef(self.value_id(&idx.value, &idx.ty)?));
        }
        instructions.push(Self::inst(
            Op::InBoundsAccessChain,
            Some(ptr_ty),
            Some(result),
            ops,
        ));
        Ok(result)
    }

    /// The predicate-and-operand-resolved core of the `fcmp` handler — the M-A4 graph walk drives it from
    /// `TirInst.cmp_predicate` (mapped through `fcmp_predicate`) + `TirInst.operands`, byte-identical to
    /// the text path (which sources the same predicate token + operands from the tir carrier).
    pub(in crate::native::emitter) fn emit_fcmp_resolved(
        &mut self,
        pred: Op,
        lhs: TypedValue,
        rhs: TypedValue,
        name: String,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let operand_ty = self.resolve_type(&lhs.ty)?;
        let result_ty = float_compare_result_type(&operand_ty)?;
        let result_type = self.type_id(&result_ty)?;
        let result = self.result_id(&name, &result_ty)?;
        let lhs_id = self.value_id_in(&lhs.value, &lhs.ty, instructions)?;
        let rhs_id = self.value_id_in(&rhs.value, &rhs.ty, instructions)?;
        instructions.push(Self::inst(
            pred,
            Some(result_type),
            Some(result),
            vec![Operand::IdRef(lhs_id), Operand::IdRef(rhs_id)],
        ));
        Ok(())
    }

    /// The POINTER core of `emit_icmp`: emit a pointer equality/inequality (direct-param constant fold,
    /// provenance compare, payload-word compare, or null compare) given the resolved predicate + typed
    /// operands. Split out so the M-A4 graph walk drives it straight from `TirInst.cmp_predicate` +
    /// `TirInst.operands` (the pointer form was the last opcode still falling through to the text
    /// substrate). `rest` is the operand TEXT, needed ONLY for the two unsupported-form error diagnostics
    /// (which BC fingerprints); the graph walk passes it from the diagnostics-only `TirInst.icmp_rest`
    /// carrier (byte-identical to the text path's `rest`), so no `inst.text` re-lex.
    pub(in crate::native::emitter) fn emit_icmp_ptr_resolved(
        &mut self,
        pred: Op,
        lhs: TypedValue,
        rhs: TypedValue,
        name: String,
        rest: &str,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let is_equal = match pred {
            Op::IEqual => true,
            Op::INotEqual => false,
            _ => {
                return Err(format!(
                    "native emitter: ordered pointer icmp is not supported: {rest}"
                ))
            }
        };
        if matches!((&lhs.value, &rhs.value), (LlValue::Zero, LlValue::Zero)) {
            let value = self.const_bool(is_equal)?;
            let result_ty = LlType::Bool;
            let result_type = self.type_id(&result_ty)?;
            let result = self.result_id(&name, &result_ty)?;
            instructions.push(Self::inst(
                Op::CopyObject,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(value)],
            ));
            return Ok(());
        }
        if let (LlValue::Local(lhs_name), LlValue::Local(rhs_name)) = (&lhs.value, &rhs.value) {
            if self.direct_param_values.contains(lhs_name)
                && self.direct_param_values.contains(rhs_name)
            {
                let equal = lhs_name == rhs_name;
                let value = self.const_bool(if is_equal { equal } else { !equal })?;
                let result_ty = LlType::Bool;
                let result_type = self.type_id(&result_ty)?;
                let result = self.result_id(&name, &result_ty)?;
                instructions.push(Self::inst(
                    Op::CopyObject,
                    Some(result_type),
                    Some(result),
                    vec![Operand::IdRef(value)],
                ));
                return Ok(());
            }
        }
        if self.emit_provenance_pointer_icmp(pred, &name, &lhs.value, &rhs.value, instructions)? {
            return Ok(());
        }
        if let (LlValue::Local(lhs_name), LlValue::Local(rhs_name)) = (&lhs.value, &rhs.value) {
            if let (Some((lhs_low, lhs_high)), Some((rhs_low, rhs_high))) = (
                self.pointer_payload_words.get(lhs_name).copied(),
                self.pointer_payload_words.get(rhs_name).copied(),
            ) {
                let result_ty = LlType::Bool;
                let result_type = self.type_id(&result_ty)?;
                let low_equal = self.fresh();
                instructions.push(Self::inst(
                    Op::IEqual,
                    Some(result_type),
                    Some(low_equal),
                    vec![Operand::IdRef(lhs_low), Operand::IdRef(rhs_low)],
                ));
                let high_equal = self.fresh();
                instructions.push(Self::inst(
                    Op::IEqual,
                    Some(result_type),
                    Some(high_equal),
                    vec![Operand::IdRef(lhs_high), Operand::IdRef(rhs_high)],
                ));
                let result = self.result_id(&name, &result_ty)?;
                if is_equal {
                    instructions.push(Self::inst(
                        Op::LogicalAnd,
                        Some(result_type),
                        Some(result),
                        vec![Operand::IdRef(low_equal), Operand::IdRef(high_equal)],
                    ));
                } else {
                    let equal = self.fresh();
                    instructions.push(Self::inst(
                        Op::LogicalAnd,
                        Some(result_type),
                        Some(equal),
                        vec![Operand::IdRef(low_equal), Operand::IdRef(high_equal)],
                    ));
                    instructions.push(Self::inst(
                        Op::LogicalNot,
                        Some(result_type),
                        Some(result),
                        vec![Operand::IdRef(equal)],
                    ));
                }
                return Ok(());
            }
        }
        let nullness = match (&lhs.value, &rhs.value) {
            (LlValue::Zero, value) | (value, LlValue::Zero) => {
                self.pointer_nullness_for_compare(value)?
            }
            _ => {
                return Err(format!(
                    "native emitter: pointer icmp is only supported against null: {rest}"
                ))
            }
        };
        let result_ty = LlType::Bool;
        let result_type = self.type_id(&result_ty)?;
        let result = self.result_id(&name, &result_ty)?;
        if is_equal {
            instructions.push(Self::inst(
                Op::CopyObject,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(nullness)],
            ));
        } else {
            instructions.push(Self::inst(
                Op::LogicalNot,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(nullness)],
            ));
        }
        Ok(())
    }

    /// The NON-pointer core of `emit_icmp`: emit a scalar/vector integer compare given its resolved
    /// predicate `Op`, typed operands, and (already-resolved) operand type. Extracted so the M-A4 graph
    /// walk can drive it straight from `TirInst.cmp_predicate` + `TirInst.operands` for the common
    /// (non-pointer) case — byte-identical to the text path. The POINTER icmp forms are driven the same way
    /// via `emit_icmp_ptr_resolved`, with the operand `rest` supplied from the `TirInst.icmp_rest` carrier
    /// for their error diagnostics.
    pub(in crate::native::emitter) fn emit_icmp_int_resolved(
        &mut self,
        pred: Op,
        lhs: TypedValue,
        rhs: TypedValue,
        operand_ty: LlType,
        name: String,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let result_ty = int_compare_result_type(&operand_ty)?;
        let result_type = self.type_id(&result_ty)?;
        let result = self.result_id(&name, &result_ty)?;
        let lhs_id = self.value_id_in(&lhs.value, &lhs.ty, instructions)?;
        let rhs_id = self.value_id_in(&rhs.value, &rhs.ty, instructions)?;
        instructions.push(Self::inst(
            pred,
            Some(result_type),
            Some(result),
            vec![Operand::IdRef(lhs_id), Operand::IdRef(rhs_id)],
        ));
        Ok(())
    }

    pub(in crate::native::emitter) fn emit_provenance_pointer_icmp(
        &mut self,
        pred: Op,
        name: &str,
        lhs: &LlValue,
        rhs: &LlValue,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let Some(lhs_provenance) =
            self.normalized_pointer_icmp_provenance(lhs, name, instructions)?
        else {
            return Ok(false);
        };
        let Some(rhs_provenance) =
            self.normalized_pointer_icmp_provenance(rhs, name, instructions)?
        else {
            return Ok(false);
        };
        if !compatible_pointer_provenance(&lhs_provenance, &rhs_provenance) {
            return Ok(false);
        }
        let Some(equal) = self.emit_pointer_index_equality(
            &lhs_provenance.indices,
            &rhs_provenance.indices,
            instructions,
        )?
        else {
            return Ok(false);
        };

        let result_ty = LlType::Bool;
        let result_type = self.type_id(&result_ty)?;
        let result = self.result_id(name, &result_ty)?;
        match pred {
            Op::IEqual => instructions.push(Self::inst(
                Op::CopyObject,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(equal)],
            )),
            Op::INotEqual => instructions.push(Self::inst(
                Op::LogicalNot,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(equal)],
            )),
            _ => return Ok(false),
        }
        Ok(true)
    }

    pub(in crate::native::emitter) fn normalized_pointer_icmp_provenance(
        &mut self,
        value: &LlValue,
        name: &str,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Option<GepProvenance>, String> {
        let LlValue::Local(local) = value else {
            return Ok(None);
        };
        let Some(provenance) = self.gep_provenance.get(local).cloned() else {
            return Ok(None);
        };
        self.flatten_pointer_icmp_provenance(provenance, name, instructions, 0)
            .map(Some)
    }

    pub(in crate::native::emitter) fn flatten_pointer_icmp_provenance(
        &mut self,
        provenance: GepProvenance,
        name: &str,
        instructions: &mut Vec<Instruction>,
        depth: usize,
    ) -> Result<GepProvenance, String> {
        if depth >= 8 {
            return Ok(provenance);
        }
        let Some(root_name) = self.provenance_root_local_name(provenance.root) else {
            return Ok(provenance);
        };
        let Some(root_provenance) = self.gep_provenance.get(&root_name).cloned() else {
            return Ok(provenance);
        };
        let root_provenance =
            self.flatten_pointer_icmp_provenance(root_provenance, name, instructions, depth + 1)?;
        let Some(indices) = self.compose_followup_gep(
            name,
            &root_provenance,
            &provenance.source_ty,
            &provenance.indices,
            instructions,
        )?
        else {
            return Ok(provenance);
        };
        Ok(GepProvenance {
            root: root_provenance.root,
            addrspace: root_provenance.addrspace,
            source_ty: root_provenance.source_ty,
            indices,
            root_is_indexed_container: root_provenance.root_is_indexed_container,
        })
    }

    pub(in crate::native::emitter) fn provenance_root_local_name(
        &self,
        root: Word,
    ) -> Option<String> {
        self.values.iter().find_map(|(name, (id, _))| {
            (*id == root && self.gep_provenance.contains_key(name)).then(|| name.clone())
        })
    }

    pub(in crate::native::emitter) fn emit_pointer_index_equality(
        &mut self,
        lhs: &[TypedValue],
        rhs: &[TypedValue],
        instructions: &mut Vec<Instruction>,
    ) -> Result<Option<Word>, String> {
        if lhs.len() != rhs.len() {
            return Ok(None);
        }
        let result_type = self.type_id(&LlType::Bool)?;
        let mut equal = None;
        for (lhs, rhs) in lhs.iter().zip(rhs) {
            let lhs_ty = self.resolve_type(&lhs.ty)?;
            let rhs_ty = self.resolve_type(&rhs.ty)?;
            if lhs_ty != rhs_ty || !matches!(lhs_ty, LlType::Int(_)) {
                return Ok(None);
            }
            let lhs_id = self.value_id_in(&lhs.value, &lhs.ty, instructions)?;
            let rhs_id = self.value_id_in(&rhs.value, &rhs.ty, instructions)?;
            let index_equal = if lhs_id == rhs_id {
                self.const_bool(true)?
            } else {
                let index_equal = self.fresh();
                instructions.push(Self::inst(
                    Op::IEqual,
                    Some(result_type),
                    Some(index_equal),
                    vec![Operand::IdRef(lhs_id), Operand::IdRef(rhs_id)],
                ));
                index_equal
            };
            equal = Some(if let Some(prev_equal) = equal {
                if index_equal == self.const_bool(true)? {
                    prev_equal
                } else {
                    let combined = self.fresh();
                    instructions.push(Self::inst(
                        Op::LogicalAnd,
                        Some(result_type),
                        Some(combined),
                        vec![Operand::IdRef(prev_equal), Operand::IdRef(index_equal)],
                    ));
                    combined
                }
            } else {
                index_equal
            });
        }
        Ok(Some(equal.unwrap_or(self.const_bool(true)?)))
    }
}
