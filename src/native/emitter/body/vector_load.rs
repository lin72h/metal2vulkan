//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    pub(in crate::native::emitter) fn scale_gep_index(
        &mut self,
        index: &TypedValue,
        scale: u32,
        name: &str,
        instructions: &mut Vec<Instruction>,
    ) -> Result<TypedValue, String> {
        if scale == 1 {
            return Ok(index.clone());
        }
        if let Some(value) = const_index(Some(index)) {
            let mut out = index.clone();
            out.value = LlValue::Int((value * scale) as u64);
            return Ok(out);
        }
        let index_ty = self.resolve_type(&index.ty)?;
        let LlType::Int(bits) = index_ty else {
            return Err(format!(
                "native emitter: vector GEP index is not an integer: {index_ty:?}"
            ));
        };
        let result_type = self.type_id(&index_ty)?;
        let index_id = self.value_id(&index.value, &index.ty)?;
        let scale_id = self.const_int(bits, scale as u64)?;
        let result = self.fresh();
        instructions.push(Self::inst(
            Op::IMul,
            Some(result_type),
            Some(result),
            vec![Operand::IdRef(index_id), Operand::IdRef(scale_id)],
        ));
        let scaled_name = format!("%air.vecidx.{}", name.trim_start_matches('%'));
        self.values
            .insert(scaled_name.clone(), (result, index_ty.clone()));
        self.record_int_alignment(
            &scaled_name,
            &index_ty,
            self.int_value_alignment(&index.value),
        );
        Ok(TypedValue {
            ty: index_ty,
            value: LlValue::Local(scaled_name),
        })
    }

    pub(in crate::native::emitter) fn vector_lane_count(&self, ty: &LlType) -> Result<u32, String> {
        match ty {
            LlType::Named(name) => {
                let aliased = self
                    .ir
                    .types
                    .get(name)
                    .ok_or_else(|| format!("native emitter: unknown named type {name}"))?;
                self.vector_lane_count(aliased)
            }
            LlType::Vector(_, lanes) => Ok(*lanes),
            other => Err(format!(
                "native emitter: expected vector type for shufflevector, got {other:?}"
            )),
        }
    }

    pub(in crate::native::emitter) fn emit_widening_vector_load(
        &mut self,
        result: Word,
        pointee: &LlType,
        result_ty: &LlType,
        ptr: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let (LlType::Vector(src_elem, src_lanes), LlType::Vector(dst_elem, dst_lanes)) =
            (pointee, result_ty)
        else {
            return Ok(false);
        };
        if src_lanes >= dst_lanes || !types_compatible(src_elem, dst_elem) {
            return Ok(false);
        }

        let pointee_type = self.type_id(pointee)?;
        let loaded = self.fresh();
        instructions.push(Self::inst(
            Op::Load,
            Some(pointee_type),
            Some(loaded),
            vec![Operand::IdRef(ptr)],
        ));

        let elem_type = self.type_id(src_elem)?;
        let mut lanes = Vec::with_capacity(*dst_lanes as usize);
        for lane in 0..*src_lanes {
            let extracted = self.fresh();
            instructions.push(Self::inst(
                Op::CompositeExtract,
                Some(elem_type),
                Some(extracted),
                vec![Operand::IdRef(loaded), Operand::LiteralBit32(lane)],
            ));
            lanes.push(Operand::IdRef(extracted));
        }
        for _ in *src_lanes..*dst_lanes {
            lanes.push(Operand::IdRef(self.undef_id(src_elem)?));
        }

        let result_type = self.type_id(result_ty)?;
        instructions.push(Self::inst(
            Op::CompositeConstruct,
            Some(result_type),
            Some(result),
            lanes,
        ));
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_narrowing_vector_load(
        &mut self,
        result: Word,
        pointee: &LlType,
        result_ty: &LlType,
        ptr: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let (LlType::Vector(src_elem, src_lanes), LlType::Vector(dst_elem, dst_lanes)) =
            (pointee, result_ty)
        else {
            return Ok(false);
        };
        if src_lanes <= dst_lanes || !types_compatible(src_elem, dst_elem) {
            return Ok(false);
        }

        let pointee_type = self.type_id(pointee)?;
        let loaded = self.fresh();
        instructions.push(Self::inst(
            Op::Load,
            Some(pointee_type),
            Some(loaded),
            vec![Operand::IdRef(ptr)],
        ));

        let elem_type = self.type_id(dst_elem)?;
        let mut lanes = Vec::with_capacity(*dst_lanes as usize);
        for lane in 0..*dst_lanes {
            let extracted = self.fresh();
            instructions.push(Self::inst(
                Op::CompositeExtract,
                Some(elem_type),
                Some(extracted),
                vec![Operand::IdRef(loaded), Operand::LiteralBit32(lane)],
            ));
            lanes.push(Operand::IdRef(extracted));
        }

        let result_type = self.type_id(result_ty)?;
        instructions.push(Self::inst(
            Op::CompositeConstruct,
            Some(result_type),
            Some(result),
            lanes,
        ));
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_scalar_to_vector_load(
        &mut self,
        result: Word,
        pointee: &LlType,
        result_ty: &LlType,
        ptr_value: &TypedValue,
        ptr: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let LlType::Vector(elem, lanes) = result_ty else {
            return Ok(false);
        };
        let elem = self.resolve_type(elem)?;
        if !types_compatible(pointee, &elem) || *lanes == 0 {
            return Ok(false);
        }
        if *lanes > 1 && !self.can_emit_gep_provenance_lane_ptrs(ptr_value, &elem)? {
            return self.emit_scalar_pointer_vector_load(
                result,
                &elem,
                *lanes,
                ptr_value,
                ptr,
                instructions,
            );
        }

        let mut lane_ptrs = Vec::with_capacity(*lanes as usize);
        for lane in 0..*lanes {
            let lane_ptr = if lane == 0 {
                ptr
            } else {
                self.emit_gep_provenance_lane_ptr(ptr_value, &elem, lane, instructions)?
                    .ok_or_else(|| {
                        "native emitter: scalar-to-vector load lost pointer provenance".to_string()
                    })?
            };
            lane_ptrs.push(lane_ptr);
        }

        let elem_type = self.type_id(&elem)?;
        let mut lane_ids = Vec::with_capacity(*lanes as usize);
        for lane_ptr in lane_ptrs {
            let lane_id = self.fresh();
            instructions.push(Self::inst(
                Op::Load,
                Some(elem_type),
                Some(lane_id),
                vec![Operand::IdRef(lane_ptr)],
            ));
            lane_ids.push(Operand::IdRef(lane_id));
        }

        let result_type = self.type_id(result_ty)?;
        instructions.push(Self::inst(
            Op::CompositeConstruct,
            Some(result_type),
            Some(result),
            lane_ids,
        ));
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_scalar_word_to_subword_vector_load(
        &mut self,
        result: Word,
        pointee: &LlType,
        result_ty: &LlType,
        ptr_value: &TypedValue,
        ptr: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let LlType::Vector(elem, lanes) = result_ty else {
            return Ok(false);
        };
        let elem = self.resolve_type(elem)?;
        let Some(pointee_bits) = bitcast_width(pointee) else {
            return Ok(false);
        };
        let Some(elem_bits) = bitcast_width(&elem) else {
            return Ok(false);
        };
        if elem_bits == 0 || pointee_bits <= elem_bits || pointee_bits % elem_bits != 0 {
            return Ok(false);
        }
        let lanes_per_word = pointee_bits / elem_bits;
        if *lanes < lanes_per_word || *lanes % lanes_per_word != 0 {
            return Ok(false);
        }
        let word_count = *lanes / lanes_per_word;
        if word_count > 1 && !self.can_emit_gep_provenance_lane_ptrs(ptr_value, pointee)? {
            return Ok(false);
        }

        let chunk_ty = LlType::Vector(Box::new(elem.clone()), lanes_per_word);
        if bitcast_width(&chunk_ty) != Some(pointee_bits) {
            return Ok(false);
        }
        let pointee_type = self.type_id(pointee)?;
        let chunk_type = self.type_id(&chunk_ty)?;
        let elem_type = self.type_id(&elem)?;
        let mut lane_ids = Vec::with_capacity(*lanes as usize);
        for word in 0..word_count {
            let word_ptr = if word == 0 {
                ptr
            } else {
                self.emit_gep_provenance_lane_ptr(ptr_value, pointee, word, instructions)?
                    .ok_or_else(|| {
                        "native emitter: subword-vector load lost pointer provenance".to_string()
                    })?
            };
            let loaded = self.fresh();
            instructions.push(Self::inst(
                Op::Load,
                Some(pointee_type),
                Some(loaded),
                vec![Operand::IdRef(word_ptr)],
            ));
            let chunk = self.fresh();
            instructions.push(Self::inst(
                Op::Bitcast,
                Some(chunk_type),
                Some(chunk),
                vec![Operand::IdRef(loaded)],
            ));
            for lane in 0..lanes_per_word {
                let lane_id = self.fresh();
                instructions.push(Self::inst(
                    Op::CompositeExtract,
                    Some(elem_type),
                    Some(lane_id),
                    vec![Operand::IdRef(chunk), Operand::LiteralBit32(lane)],
                ));
                lane_ids.push(Operand::IdRef(lane_id));
            }
        }

        let result_type = self.type_id(result_ty)?;
        instructions.push(Self::inst(
            Op::CompositeConstruct,
            Some(result_type),
            Some(result),
            lane_ids,
        ));
        Ok(true)
    }

    /// Reinterpret-load a WIDER VECTOR result from a narrower SCALAR pointee by reading the contiguous
    /// scalar slots that span the result's bytes and bit-reinterpreting them. The dominant case is an
    /// `OpLoad %v4float` (128 bits) through a `half` element pointer into a `device half*` runtime array
    /// (an MPS half-buffer read as a float vector): 8 contiguous halfs are the 4 floats' bytes. For each
    /// result lane this packs `slots_per_elem = result_elem_bits / pointee_bits` consecutive pointee
    /// scalars (lane 0 = the chain pointer, the rest via gep-provenance sibling pointers) little-endian
    /// into the result element's same-width unsigned int, builds a `Vector(uint, lanes)`, and bitcasts
    /// it to the result vector. Packing into a `uint` vector (never an N-wide pointee vector) keeps the
    /// component count in {2,3,4} — no `Vector16` capability is needed.
    ///
    /// Byte-EXACT on a little-endian target (the assembled bits are the bytes the load reads at the
    /// chain address). Byte-SAFE by construction: only fires when the gep provenance strides a
    /// contiguous element (a single leading pointer-stride index, or a multi-index chain whose last
    /// index descends an ARRAY/vector — NOT a struct member, whose sibling indices would walk into a
    /// differently-typed member). Floor-SAFE by construction: only REACHED on a width mismatch a valid
    /// module never has, and returns false unless the provenance proves contiguity, so a banked module
    /// is provably untouched.
    pub(in crate::native::emitter) fn emit_scalar_to_wider_vector_load(
        &mut self,
        result: Word,
        pointee: &LlType,
        result_ty: &LlType,
        ptr_value: &TypedValue,
        ptr: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if matches!(pointee, LlType::Vector(..)) {
            return Ok(false);
        }
        let LlType::Vector(result_elem, lanes) = result_ty else {
            return Ok(false);
        };
        let result_elem = self.resolve_type(result_elem)?;
        let (Some(pointee_bits), Some(result_elem_bits)) =
            (bitcast_width(pointee), bitcast_width(&result_elem))
        else {
            return Ok(false);
        };
        if pointee_bits == 0
            || *lanes == 0
            || result_elem_bits <= pointee_bits
            || result_elem_bits % pointee_bits != 0
        {
            return Ok(false);
        }
        let slots_per_elem = result_elem_bits / pointee_bits;
        let total_slots = slots_per_elem * *lanes;
        // Byte-safety gate: the sibling slots must be a contiguous array/vector stride, and the
        // provenance must be able to form per-slot sibling pointers.
        if !self.gep_provenance_strides_contiguous(ptr_value, pointee)?
            || !self.can_emit_gep_provenance_lane_ptrs(ptr_value, pointee)?
        {
            return Ok(false);
        }

        let pointee_type = self.type_id(pointee)?;
        let pointee_uint = LlType::Int(pointee_bits);
        let pointee_uint_type = self.type_id(&pointee_uint)?;
        let elem_uint = LlType::Int(result_elem_bits);
        let elem_uint_type = self.type_id(&elem_uint)?;

        // Load every contiguous scalar slot once.
        let mut slot_uints = Vec::with_capacity(total_slots as usize);
        for slot in 0..total_slots {
            let slot_ptr = if slot == 0 {
                ptr
            } else {
                self.emit_gep_provenance_lane_ptr(ptr_value, pointee, slot, instructions)?
                    .ok_or_else(|| {
                        "native emitter: scalar-to-wider-vector load lost pointer provenance"
                            .to_string()
                    })?
            };
            let loaded = self.fresh();
            instructions.push(Self::inst(
                Op::Load,
                Some(pointee_type),
                Some(loaded),
                vec![Operand::IdRef(slot_ptr)],
            ));
            // Reinterpret the scalar's bits as its same-width unsigned int.
            let as_uint = if *pointee == pointee_uint {
                loaded
            } else {
                let id = self.fresh();
                instructions.push(Self::inst(
                    Op::Bitcast,
                    Some(pointee_uint_type),
                    Some(id),
                    vec![Operand::IdRef(loaded)],
                ));
                id
            };
            slot_uints.push(as_uint);
        }

        // Pack `slots_per_elem` consecutive slots, little-endian, into each result element's uint.
        let mut elem_uints = Vec::with_capacity(*lanes as usize);
        for lane in 0..*lanes {
            let mut acc: Option<Word> = None;
            for j in 0..slots_per_elem {
                let slot_uint = slot_uints[(lane * slots_per_elem + j) as usize];
                // Widen the slot's bits to the element-uint width.
                let widened = if pointee_bits == result_elem_bits {
                    slot_uint
                } else {
                    let id = self.fresh();
                    instructions.push(Self::inst(
                        Op::UConvert,
                        Some(elem_uint_type),
                        Some(id),
                        vec![Operand::IdRef(slot_uint)],
                    ));
                    id
                };
                let shifted = if j == 0 {
                    widened
                } else {
                    let shift = self.const_uint(j * pointee_bits)?;
                    let id = self.fresh();
                    instructions.push(Self::inst(
                        Op::ShiftLeftLogical,
                        Some(elem_uint_type),
                        Some(id),
                        vec![Operand::IdRef(widened), Operand::IdRef(shift)],
                    ));
                    id
                };
                acc = Some(match acc {
                    None => shifted,
                    Some(prev) => {
                        let id = self.fresh();
                        instructions.push(Self::inst(
                            Op::BitwiseOr,
                            Some(elem_uint_type),
                            Some(id),
                            vec![Operand::IdRef(prev), Operand::IdRef(shifted)],
                        ));
                        id
                    }
                });
            }
            elem_uints.push(Operand::IdRef(acc.ok_or_else(|| {
                "native emitter: vector integer load produced no lane accumulator \
                 (slots_per_elem must be >= 1)"
                    .to_string()
            })?));
        }

        // Build the uint vector, then bitcast to the result vector type if it isn't already uint.
        let uint_vec_ty = LlType::Vector(Box::new(elem_uint.clone()), *lanes);
        let bitcast_needed = !types_compatible(&result_elem, &elem_uint);
        let uint_vec_type = self.type_id(&uint_vec_ty)?;
        let uint_vec = if bitcast_needed { self.fresh() } else { result };
        instructions.push(Self::inst(
            Op::CompositeConstruct,
            Some(uint_vec_type),
            Some(uint_vec),
            elem_uints,
        ));
        if bitcast_needed {
            let result_type = self.type_id(result_ty)?;
            instructions.push(Self::inst(
                Op::Bitcast,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(uint_vec)],
            ));
        }
        Ok(true)
    }

    /// True when striding the last gep-provenance index yields contiguous same-type storage — a single
    /// leading pointer-stride index (LLVM `getelementptr T, ptr, i64 N`, contiguous by construction) or
    /// a multi-index chain whose last index descends an array/vector element. A struct-member last
    /// index is rejected: striding it would walk into a different, possibly differently-typed member.
    pub(in crate::native::emitter) fn gep_provenance_strides_contiguous(
        &self,
        ptr: &TypedValue,
        _elem: &LlType,
    ) -> Result<bool, String> {
        let LlValue::Local(name) = &ptr.value else {
            return Ok(false);
        };
        let Some(provenance) = self.gep_provenance.get(name) else {
            return Ok(false);
        };
        if provenance.indices.len() <= 1 {
            return Ok(true);
        }
        Ok(matches!(
            gep_parent_before_last(&provenance.source_ty, &provenance.indices),
            Some(LlType::Array(..)) | Some(LlType::Vector(..))
        ))
    }

    /// Reinterpret-load a SCALAR result from a VECTOR pointee by reading component 0 (the value at the
    /// access chain's byte address) and reinterpreting it to the scalar result. The dominant case is a
    /// `Half` (16-bit) load through a `<4 x float>` member pointer — a sub-component reinterpret an MPS
    /// heterogeneous-struct buffer expresses: the half is the low 16 bits of float component 0, so it
    /// is byte-EXACT on a little-endian target (extract component 0, then take the low bits via the
    /// element's same-width unsigned int and bitcast to the result scalar). Only fires for a result no
    /// WIDER than the vector element (a sub-element read at byte offset 0); a wider scalar that would
    /// span multiple lanes is left to the other handlers / the raw retry.
    ///
    /// Byte-safe by construction (the bits live at the chain's address) and floor-safe by construction:
    /// only REACHED when the load width already mismatches the declared pointee (a valid module's loads
    /// match their pointee and never enter this branch), and it returns `false` for any non-vector
    /// pointee / non-scalar or wider result, so a banked module is provably untouched.
    pub(in crate::native::emitter) fn emit_scalar_from_vector_load(
        &mut self,
        result: Word,
        pointee: &LlType,
        result_ty: &LlType,
        ptr: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let LlType::Vector(elem, _lanes) = pointee else {
            return Ok(false);
        };
        let elem = self.resolve_type(elem)?;
        if matches!(result_ty, LlType::Vector(..)) {
            return Ok(false);
        }
        let (Some(elem_bits), Some(res_bits)) = (bitcast_width(&elem), bitcast_width(result_ty))
        else {
            return Ok(false);
        };
        // A wider result spans multiple lanes — not a single-component sub-read.
        if elem_bits == 0 || res_bits == 0 || res_bits > elem_bits {
            return Ok(false);
        }

        let pointee_type = self.type_id(pointee)?;
        let loaded = self.fresh();
        instructions.push(Self::inst(
            Op::Load,
            Some(pointee_type),
            Some(loaded),
            vec![Operand::IdRef(ptr)],
        ));
        let elem_type = self.type_id(&elem)?;
        // Component 0 is the scalar at the access chain's byte address.
        if elem == *result_ty {
            // Same scalar type — the extract IS the result (no reinterpret needed).
            instructions.push(Self::inst(
                Op::CompositeExtract,
                Some(self.type_id(result_ty)?),
                Some(result),
                vec![Operand::IdRef(loaded), Operand::LiteralBit32(0)],
            ));
            return Ok(true);
        }
        let comp0 = self.fresh();
        instructions.push(Self::inst(
            Op::CompositeExtract,
            Some(elem_type),
            Some(comp0),
            vec![Operand::IdRef(loaded), Operand::LiteralBit32(0)],
        ));

        if res_bits == elem_bits {
            // Same-width bit reinterpret of component 0 (e.g. uint from float).
            let result_type = self.type_id(result_ty)?;
            instructions.push(Self::inst(
                Op::Bitcast,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(comp0)],
            ));
            return Ok(true);
        }

        // Narrower result: take the low `res_bits` via the element's same-width unsigned int, then
        // bitcast to the result scalar (little-endian: the low bits are the bytes at the address).
        let elem_uint = LlType::Int(elem_bits);
        let as_uint = if elem == elem_uint {
            comp0
        } else {
            let elem_uint_type = self.type_id(&elem_uint)?;
            let id = self.fresh();
            instructions.push(Self::inst(
                Op::Bitcast,
                Some(elem_uint_type),
                Some(id),
                vec![Operand::IdRef(comp0)],
            ));
            id
        };
        let res_uint = LlType::Int(res_bits);
        let res_is_uint = *result_ty == res_uint;
        let res_uint_type = self.type_id(&res_uint)?;
        let truncated = if res_is_uint { result } else { self.fresh() };
        instructions.push(Self::inst(
            Op::UConvert,
            Some(res_uint_type),
            Some(truncated),
            vec![Operand::IdRef(as_uint)],
        ));
        if !res_is_uint {
            let result_type = self.type_id(result_ty)?;
            instructions.push(Self::inst(
                Op::Bitcast,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(truncated)],
            ));
        }
        Ok(true)
    }

    /// Reinterpret a scalar `OpLoad` whose declared pointee is a WIDER scalar than the result (e.g.
    /// `OpLoad %float` (32) through an `Int(64)` slot): load the pointee, reinterpret to its same-width
    /// unsigned int, truncate to the low `result_bits` via `OpUConvert`, then bitcast to the result
    /// scalar. Little-endian, so the low bits are EXACTLY the bytes at the access chain's address —
    /// byte-EXACT and confined WITHIN the pointee slot (a narrowing read never crosses into a sibling
    /// slot or a struct-member boundary, unlike a widening read which the access-chain rewrite must
    /// model). This is the scalar-pointee analogue of the narrowing branch of
    /// `emit_scalar_from_vector_load` (which narrows a vector pointee's component 0).
    ///
    /// Floor-safe / byte-safe by construction: only fires on a scalar→scalar width mismatch
    /// (`result_bits < pointee_bits`) a valid module never has — every other handler in the
    /// reinterpret-load chain already declined, and a banked/valid module's loads all match their
    /// pointee, so it never reaches here. Returns `Ok(false)` (fall through to the raw retry) for a
    /// vector operand, a non-bitcastable type, or a widening/same-width load.
    pub(in crate::native::emitter) fn emit_scalar_narrowing_load(
        &mut self,
        result: Word,
        pointee: &LlType,
        result_ty: &LlType,
        ptr: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if matches!(pointee, LlType::Vector(..)) || matches!(result_ty, LlType::Vector(..)) {
            return Ok(false);
        }
        let (Some(pointee_bits), Some(res_bits)) =
            (bitcast_width(pointee), bitcast_width(result_ty))
        else {
            return Ok(false);
        };
        // Strict narrowing only: a wider/same-width result is not this handler's job (a wider read
        // would cross into a sibling slot the access-chain rewrite owns; same-width is the bitcast
        // path below the mismatch block).
        if pointee_bits == 0 || res_bits == 0 || res_bits >= pointee_bits {
            return Ok(false);
        }

        let pointee_type = self.type_id(pointee)?;
        let loaded = self.fresh();
        instructions.push(Self::inst(
            Op::Load,
            Some(pointee_type),
            Some(loaded),
            vec![Operand::IdRef(ptr)],
        ));
        // Reinterpret the loaded value as its same-width unsigned int (a no-op when the pointee
        // already is that int), truncate to the low `res_bits`, then bitcast to the result scalar.
        let pointee_uint = LlType::Int(pointee_bits);
        let as_uint = if *pointee == pointee_uint {
            loaded
        } else {
            let pointee_uint_type = self.type_id(&pointee_uint)?;
            let id = self.fresh();
            instructions.push(Self::inst(
                Op::Bitcast,
                Some(pointee_uint_type),
                Some(id),
                vec![Operand::IdRef(loaded)],
            ));
            id
        };
        let res_uint = LlType::Int(res_bits);
        let res_is_uint = *result_ty == res_uint;
        let res_uint_type = self.type_id(&res_uint)?;
        let truncated = if res_is_uint { result } else { self.fresh() };
        instructions.push(Self::inst(
            Op::UConvert,
            Some(res_uint_type),
            Some(truncated),
            vec![Operand::IdRef(as_uint)],
        ));
        if !res_is_uint {
            let result_type = self.type_id(result_ty)?;
            instructions.push(Self::inst(
                Op::Bitcast,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(truncated)],
            ));
        }
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_scalar_pointer_vector_load(
        &mut self,
        result: Word,
        elem: &LlType,
        lanes: u32,
        ptr_value: &TypedValue,
        ptr: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let LlType::Ptr(addrspace) = self.resolve_type(&ptr_value.ty)? else {
            return Ok(false);
        };
        let storage = self.pointer_storage_for(&ptr_value.value, addrspace)?;
        if storage != StorageClass::Workgroup {
            return Ok(false);
        }

        let ptr_type = self.ptr_type_id(storage, elem)?;
        let elem_type = self.type_id(elem)?;
        let mut lane_ids = Vec::with_capacity(lanes as usize);
        for lane in 0..lanes {
            let lane_ptr = if lane == 0 {
                ptr
            } else {
                let lane_index = self.const_uint(lane)?;
                let lane_ptr = self.fresh();
                instructions.push(Self::inst(
                    Op::PtrAccessChain,
                    Some(ptr_type),
                    Some(lane_ptr),
                    vec![Operand::IdRef(ptr), Operand::IdRef(lane_index)],
                ));
                lane_ptr
            };
            let lane_id = self.fresh();
            instructions.push(Self::inst(
                Op::Load,
                Some(elem_type),
                Some(lane_id),
                vec![Operand::IdRef(lane_ptr)],
            ));
            lane_ids.push(Operand::IdRef(lane_id));
        }

        let result_type = self.type_id(&LlType::Vector(Box::new(elem.clone()), lanes))?;
        instructions.push(Self::inst(
            Op::CompositeConstruct,
            Some(result_type),
            Some(result),
            lane_ids,
        ));
        Ok(true)
    }

    pub(in crate::native::emitter) fn can_emit_gep_provenance_lane_ptrs(
        &self,
        ptr: &TypedValue,
        elem: &LlType,
    ) -> Result<bool, String> {
        let LlValue::Local(name) = &ptr.value else {
            return Ok(false);
        };
        let Some(provenance) = self.gep_provenance.get(name) else {
            return Ok(false);
        };
        if provenance.indices.is_empty() {
            return Ok(false);
        }
        Ok(types_compatible(
            &gep_pointee(&provenance.source_ty, &provenance.indices)?,
            elem,
        ) && self.gep_provenance_strides_contiguous(ptr, elem)?)
    }

    pub(in crate::native::emitter) fn emit_gep_provenance_lane_ptr(
        &mut self,
        ptr: &TypedValue,
        elem: &LlType,
        lane: u32,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Option<Word>, String> {
        let LlValue::Local(name) = &ptr.value else {
            return Ok(None);
        };
        let Some(provenance) = self.gep_provenance.get(name).cloned() else {
            return Ok(None);
        };
        if !types_compatible(
            &gep_pointee(&provenance.source_ty, &provenance.indices)?,
            elem,
        ) {
            return Ok(None);
        }
        let Some(last_index) = provenance.indices.last() else {
            return Ok(None);
        };
        let mut indices = provenance.indices.clone();
        let lane_index = TypedValue {
            ty: last_index.ty.clone(),
            value: LlValue::Int(lane as u64),
        };
        if let Some(last) = indices.last_mut() {
            *last = self.combine_gep_indices(last, &lane_index, instructions)?;
        }
        let pointee = gep_pointee(&provenance.source_ty, &indices)?;
        let storage = self.pointer_storage_for(&ptr.value, provenance.addrspace)?;
        let ptr_ty = self.ptr_type_id(storage, &pointee)?;
        let result = self.fresh();
        let mut ops = vec![Operand::IdRef(provenance.root)];
        for idx in gep_spirv_indices(&indices)? {
            ops.push(Operand::IdRef(self.value_id(&idx.value, &idx.ty)?));
        }
        let root_is_param = self.param_values.iter().any(|param| {
            self.values
                .get(param)
                .is_some_and(|(id, _)| *id == provenance.root)
        });
        instructions.push(Self::inst(
            if root_is_param
                || provenance.root_is_indexed_container
                || self.is_indexed_container_root(provenance.root, None)
                || !ptr_access_chain_allowed_storage(storage)
            {
                Op::InBoundsAccessChain
            } else {
                Op::PtrAccessChain
            },
            Some(ptr_ty),
            Some(result),
            ops,
        ));
        Ok(Some(result))
    }

    pub(in crate::native::emitter) fn drop_indirect_function_group_call(
        &mut self,
        rest: &str,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        if !rest.contains("!air.function_groups") {
            return Ok(false);
        }
        let Some(open) = rest.find('(') else {
            return Ok(false);
        };
        let head = rest[..open].trim();
        let head_parts = split_top_level_whitespace(head);
        let Some(callee_text) = head_parts.last().copied() else {
            return Ok(false);
        };
        if !callee_text.starts_with('%') {
            return Ok(false);
        }
        let ret_text = head
            .strip_suffix(callee_text)
            .map(str::trim)
            .unwrap_or(head);
        let ret_ty = self.resolve_type(&parse_type(ret_text)?)?;
        if ret_ty != LlType::Void {
            return Err(format!(
                "native emitter: indirect function-group call returned {ret_ty:?}"
            ));
        }
        let callee = parse_value(callee_text)?;
        let _ = self.value_id(&callee, &LlType::Ptr(0))?;
        let close = matching_paren(rest, open)
            .ok_or_else(|| format!("native emitter: unmatched indirect call parens: {rest}"))?;
        let args_text = &rest[open + 1..close];
        if !args_text.trim().is_empty() {
            for arg in split_top_level(args_text, ',') {
                let arg = parse_typed_value(arg)?;
                let _ = self.value_id_in(&arg.value, &arg.ty, instructions)?;
            }
        }
        Ok(true)
    }
}
