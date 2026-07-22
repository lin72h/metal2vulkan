use super::*;

impl Emitter {
    /// Canonicalize a type to its SPIR-V *storage* form for type-table interning. `BFloat` shares
    /// `Int(16)`'s storage type id everywhere (the u16 holding the bf16 bit pattern — see the
    /// scalar contract in `memory.rs`), so `Vector(BFloat, N)`/`Array(BFloat, N)`/`Struct{…BFloat…}`
    /// must collapse onto their `Int(16)` equivalents. Without this, a bf16 vector and a u16 vector
    /// mint two distinct `OpTypeVector %ushort N` declarations, which spirv-val rejects as a
    /// "Duplicate non-aggregate type declaration".
    ///
    /// Non-SPIR-V-legal integer widths (`i2`…`i7`, `i9`…`i15`, `i17`…`i31`, `i33`…`i63`) are also
    /// widened here to the next legal container (`i8`/`i16`/`i32`/`i64`) so the type table never
    /// emits `OpTypeInt N` for an N-bit type the owned loader/`spirv-val` reject. Logical
    /// `LlType::Int(N)` is still
    /// preserved by `resolve_type` for value-level masking/const encoding; this only normalizes the
    /// type-id key/build basis.
    fn storage_type(ty: &LlType) -> LlType {
        match ty {
            LlType::BFloat => LlType::Int(16),
            LlType::Int(bits) => match spirv_int_width(*bits) {
                Ok(legal) if legal != *bits => LlType::Int(legal),
                _ => ty.clone(),
            },
            LlType::Vector(elem, lanes) => {
                LlType::Vector(Box::new(Self::storage_type(elem)), *lanes)
            }
            LlType::Array(elem, len) => LlType::Array(Box::new(Self::storage_type(elem)), *len),
            LlType::Struct(fields) => {
                LlType::Struct(fields.iter().map(Self::storage_type).collect())
            }
            other => other.clone(),
        }
    }

    pub(super) fn type_id(&mut self, ty: &LlType) -> Result<Word, String> {
        let ty = Self::storage_type(&self.resolve_type(ty)?);
        if let Some(id) = self.interner.types.get(&ty) {
            return Ok(*id);
        }
        if let LlType::Ptr(addrspace) = &ty {
            let storage = llvm_pointer_storage(*addrspace)?;
            let key = (storage, LlType::Int(8));
            if let Some(id) = self.interner.ptr_types.get(&key).copied() {
                self.interner.types.insert(ty, id);
                return Ok(id);
            }
        }
        let id = self.fresh();
        let inst = match &ty {
            LlType::Void => Self::inst(Op::TypeVoid, None, Some(id), vec![]),
            LlType::Bool => Self::inst(Op::TypeBool, None, Some(id), vec![]),
            LlType::Float => Self::inst(
                Op::TypeFloat,
                None,
                Some(id),
                vec![Operand::LiteralBit32(32)],
            ),
            LlType::Half => {
                self.require_capability(Capability::Float16);
                Self::inst(
                    Op::TypeFloat,
                    None,
                    Some(id),
                    vec![Operand::LiteralBit32(16)],
                )
            }
            LlType::BFloat => {
                return Err(
                    "native emitter: BFloat must be normalized to its u16 storage type before \
                     type_id"
                        .into(),
                )
            }
            LlType::Int(bits) => {
                match *bits {
                    8 => self.require_capability(Capability::Int8),
                    16 => self.require_capability(Capability::Int16),
                    64 => self.require_capability(Capability::Int64),
                    _ => {}
                }
                Self::inst(
                    Op::TypeInt,
                    None,
                    Some(id),
                    vec![Operand::LiteralBit32(*bits), Operand::LiteralBit32(0)],
                )
            }
            LlType::Ptr(addrspace) => {
                let storage = llvm_pointer_storage(*addrspace)?;
                let uchar = self.type_id(&LlType::Int(8))?;
                self.interner
                    .ptr_types
                    .insert((storage, LlType::Int(8)), id);
                Self::inst(
                    Op::TypePointer,
                    None,
                    Some(id),
                    vec![Operand::StorageClass(storage), Operand::IdRef(uchar)],
                )
            }
            LlType::Vector(elem, lanes) if *lanes > 4 => {
                let elem = self.type_id(elem)?;
                let len = self.const_uint(*lanes)?;
                Self::inst(
                    Op::TypeArray,
                    None,
                    Some(id),
                    vec![Operand::IdRef(elem), Operand::IdRef(len)],
                )
            }
            LlType::Vector(elem, lanes) => {
                let elem = self.type_id(elem)?;
                Self::inst(
                    Op::TypeVector,
                    None,
                    Some(id),
                    vec![Operand::IdRef(elem), Operand::LiteralBit32(*lanes)],
                )
            }
            LlType::Array(elem, len) => {
                let elem = self.type_id(elem)?;
                if *len == 0 {
                    Self::inst(
                        Op::TypeRuntimeArray,
                        None,
                        Some(id),
                        vec![Operand::IdRef(elem)],
                    )
                } else {
                    let len = self.const_uint(*len)?;
                    Self::inst(
                        Op::TypeArray,
                        None,
                        Some(id),
                        vec![Operand::IdRef(elem), Operand::IdRef(len)],
                    )
                }
            }
            LlType::Struct(fields) => {
                let mut operands = Vec::with_capacity(fields.len());
                for field in fields {
                    operands.push(Operand::IdRef(self.type_id(field)?));
                }
                Self::inst(Op::TypeStruct, None, Some(id), operands)
            }
            LlType::Named(name) => {
                return Err(format!("native emitter: unresolved named type {name}"));
            }
        };
        self.module.types_global_values.push(inst);
        self.interner.types.insert(ty, id);
        Ok(id)
    }

    pub(super) fn signed_int_type_id(&mut self, ty: &LlType) -> Result<Word, String> {
        // Legalize nonstandard int widths the same way `type_id`/`storage_type` do, so a signed
        // `i2` shares the `OpTypeInt 8 1` declaration with a signed `i8` rather than minting a
        // duplicate illegal-width type.
        let ty = Self::storage_type(&self.resolve_type(ty)?);
        if let Some(id) = self.interner.signed_int_types.get(&ty) {
            return Ok(*id);
        }
        let id = self.fresh();
        let inst = match &ty {
            LlType::Int(bits) => {
                match *bits {
                    8 => self.require_capability(Capability::Int8),
                    16 => self.require_capability(Capability::Int16),
                    64 => self.require_capability(Capability::Int64),
                    _ => {}
                }
                Self::inst(
                    Op::TypeInt,
                    None,
                    Some(id),
                    vec![Operand::LiteralBit32(*bits), Operand::LiteralBit32(1)],
                )
            }
            LlType::Vector(elem, lanes) if is_integer_type(elem) => {
                let elem = self.signed_int_type_id(elem)?;
                Self::inst(
                    Op::TypeVector,
                    None,
                    Some(id),
                    vec![Operand::IdRef(elem), Operand::LiteralBit32(*lanes)],
                )
            }
            _ => {
                return Err(format!(
                    "native emitter: signed integer type requested for {ty:?}"
                ))
            }
        };
        self.module.types_global_values.push(inst);
        self.interner.signed_int_types.insert(ty, id);
        Ok(id)
    }

    pub(super) fn ptr_type_id(
        &mut self,
        storage: StorageClass,
        pointee: &LlType,
    ) -> Result<Word, String> {
        let pointee = self.resolve_type(pointee)?;
        let key = (storage, pointee.clone());
        if let Some(id) = self.interner.ptr_types.get(&key) {
            return Ok(*id);
        }
        if pointee == LlType::Int(8) {
            if let Some((_, id)) = self.interner.types.iter().find(|(ty, _)| {
                matches!(
                    ty,
                    LlType::Ptr(addrspace)
                        if llvm_pointer_storage(*addrspace).ok() == Some(storage)
                )
            }) {
                self.interner.ptr_types.insert(key, *id);
                return Ok(*id);
            }
        }
        let pointee_id = self.type_id(&pointee)?;
        let id = self.fresh();
        self.module.types_global_values.push(Self::inst(
            Op::TypePointer,
            None,
            Some(id),
            vec![Operand::StorageClass(storage), Operand::IdRef(pointee_id)],
        ));
        self.interner.ptr_types.insert(key, id);
        Ok(id)
    }

    pub(super) fn const_uint(&mut self, value: u32) -> Result<Word, String> {
        if let Some(id) = self.interner.uint_constants.get(&value) {
            return Ok(*id);
        }
        let uint = self.type_id(&LlType::Int(32))?;
        let id = self.fresh();
        self.module.types_global_values.push(Self::inst(
            Op::Constant,
            Some(uint),
            Some(id),
            vec![Operand::LiteralBit32(value)],
        ));
        self.interner.uint_constants.insert(value, id);
        Ok(id)
    }

    /// A `uint` constant shaped to `n` lanes: the scalar constant for `n <= 1`, else a `Vector(uint, n)`
    /// splat (`OpConstantComposite`). Used to build the shift amount for shaped bf16 widen/narrow, where
    /// a vector shift requires a same-component-count shift operand.
    pub(super) fn const_uint_shaped(&mut self, value: u32, n: u32) -> Result<Word, String> {
        let scalar = self.const_uint(value)?;
        if n <= 1 {
            return Ok(scalar);
        }
        let vec_ty = LlType::Vector(Box::new(LlType::Int(32)), n);
        self.const_composite_with_constituents(&vec_ty, vec![scalar; n as usize])
    }

    pub(super) fn resolve_type(&self, ty: &LlType) -> Result<LlType, String> {
        match ty {
            LlType::Int(1) => Ok(LlType::Bool),
            LlType::Named(name) => {
                let aliased = self
                    .ir
                    .types
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("native emitter: unknown named type {name}"))?;
                self.resolve_type(&aliased)
            }
            LlType::Vector(elem, 1) => self.resolve_type(elem),
            LlType::Vector(elem, lanes) => {
                Ok(LlType::Vector(Box::new(self.resolve_type(elem)?), *lanes))
            }
            LlType::Array(elem, len) => Ok(LlType::Array(Box::new(self.resolve_type(elem)?), *len)),
            LlType::Struct(fields) => fields
                .iter()
                .map(|f| self.resolve_type(f))
                .collect::<Result<Vec<_>, _>>()
                .map(LlType::Struct),
            _ => Ok(ty.clone()),
        }
    }

    pub(super) fn undef_id(&mut self, ty: &LlType) -> Result<Word, String> {
        let ty = self.resolve_type(ty)?;
        if let Some(id) = self.interner.undefs.get(&ty) {
            return Ok(*id);
        }
        let type_id = self.type_id(&ty)?;
        let id = self.fresh();
        self.module.types_global_values.push(Self::inst(
            Op::Undef,
            Some(type_id),
            Some(id),
            vec![],
        ));
        self.interner.undefs.insert(ty, id);
        Ok(id)
    }

    pub(super) fn result_id(&mut self, name: &str, ty: &LlType) -> Result<Word, String> {
        let ty = self.resolve_type(ty)?;
        if let Some((id, have_ty)) = self.values.get(name).cloned() {
            let have = self.resolve_type(&have_ty)?;
            if !types_compatible(&have, &ty) {
                return Err(format!(
                    "native emitter: SSA value {name} was reserved as {have:?}, defined as {ty:?}"
                ));
            }
            return Ok(id);
        }
        let id = self.fresh();
        self.values.insert(name.to_string(), (id, ty));
        Ok(id)
    }

    pub(super) fn phi_value_id(
        &mut self,
        value: &LlValue,
        ty: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        if let LlValue::Local(name) = value {
            if !self.values.contains_key(name) {
                return self.result_id(name, ty);
            }
        }
        self.value_id_in(value, ty, instructions)
    }

    pub(super) fn value_id(&mut self, value: &LlValue, ty: &LlType) -> Result<Word, String> {
        match value {
            LlValue::Local(name) => {
                let (id, have_ty) = if let Some((id, have_ty)) = self.values.get(name).cloned() {
                    (id, have_ty)
                } else if self.construct_tree {
                    // The construct-tree retry can route a loop latch before the header phi it
                    // back-edges to.  That is still a dominance-valid SSA use, and `OpPhi` already
                    // allocates ids for future incoming values via `phi_value_id`; mirror that behavior
                    // for non-phi uses, but only inside this reject-triggered retry tier.  The finished
                    // module remains adopted only after `spirv-val`, so an actually non-dominating use
                    // still falls through.
                    let Some(have_ty) = self.tir_result_types.get(name).cloned() else {
                        return Err(format!("native emitter: unknown SSA value {name}"));
                    };
                    let want = self.resolve_type(ty)?;
                    let have = self.resolve_type(&have_ty)?;
                    if !types_compatible(&have, &want) {
                        return Err(format!(
                            "native emitter: SSA value {name} has type {have:?}, used as {want:?}"
                        ));
                    }
                    (self.result_id(name, &have_ty)?, have_ty)
                } else {
                    return Err(format!("native emitter: unknown SSA value {name}"));
                };
                let want = self.resolve_type(ty)?;
                let have = self.resolve_type(&have_ty)?;
                if !types_compatible(&have, &want) {
                    return Err(format!(
                        "native emitter: SSA value {name} has type {have:?}, used as {want:?}"
                    ));
                }
                Ok(id)
            }
            LlValue::Global(name) => {
                let (id, _have_ty) = self
                    .global_values
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("native emitter: unknown global value {name}"))?;
                Ok(id)
            }
            LlValue::Gep(_) => Err(
                "native emitter: getelementptr value requires instruction-local materialization"
                    .to_string(),
            ),
            LlValue::Bool(value) => match self.resolve_type(ty)? {
                LlType::Bool | LlType::Int(1) => self.const_bool(*value),
                other => Err(format!(
                    "native emitter: bool literal {value} used as non-bool type {other:?}"
                )),
            },
            LlValue::Int(value) => match self.resolve_type(ty)? {
                LlType::Int(bits) => self.const_int(bits, *value),
                other => Err(format!(
                    "native emitter: integer literal {value} used as non-int type {other:?}"
                )),
            },
            LlValue::SignedInt(value) => match self.resolve_type(ty)? {
                LlType::Int(bits) => self.const_signed_int(bits, *value),
                other => Err(format!(
                    "native emitter: integer literal {value} used as non-int type {other:?}"
                )),
            },
            LlValue::Hex(bits) => match self.resolve_type(ty)? {
                LlType::Int(width) => self.const_int(width, *bits),
                LlType::Float => self.const_float32(f64::from_bits(*bits) as f32),
                other => Err(format!(
                    "native emitter: hex literal 0x{bits:x} used as unsupported type {other:?}"
                )),
            },
            LlValue::Float(value) => match self.resolve_type(ty)? {
                LlType::Float => self.const_float32(*value as f32),
                other => Err(format!(
                    "native emitter: float literal {value} used as non-float type {other:?}"
                )),
            },
            LlValue::HalfBits(bits) => match self.resolve_type(ty)? {
                LlType::Half => self.const_float16_bits(*bits),
                other => Err(format!(
                    "native emitter: half literal 0xH{bits:04x} used as non-half type {other:?}"
                )),
            },
            LlValue::BFloatBits(bits) => match self.resolve_type(ty)? {
                LlType::BFloat => self.const_int(16, *bits as u64),
                other => Err(format!(
                    "native emitter: bfloat literal 0xR{bits:04x} used as non-bfloat type {other:?}"
                )),
            },
            LlValue::Vector(_) | LlValue::Array(_) | LlValue::Struct(_) | LlValue::Splat(_) => Err(
                "native emitter: aggregate literal requires instruction-local materialization"
                    .into(),
            ),
            LlValue::Zero => self.const_null(ty),
            LlValue::Undef => self.undef_id(ty),
        }
    }

    pub(super) fn value_id_in(
        &mut self,
        value: &LlValue,
        ty: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let result_ty = self.resolve_type(ty)?;
        if matches!(result_ty, LlType::Ptr(_)) {
            if let LlValue::Local(name) = value {
                if self.selected_pointers.contains_key(name) && !self.values.contains_key(name) {
                    self.materialize_selected_pointer_value(name, instructions)?;
                }
            }
        }
        if let LlValue::Gep(gep) = value {
            let name = format!("%air.constgep.{}", self.module.id_bound());
            if let Some(id) = self.emit_gep_result(&name, gep, instructions)? {
                return Ok(id);
            }
            // A constant GEP whose base is a module global (e.g. an `addrspace(3)` threadgroup
            // array) is materialized by the Workgroup aggregate handler, which stores the access
            // chain in `self.values` and returns `Ok(None)` (the convention the named-GEP caller
            // reads). Recover that stored id for the inline-constgep caller. Restricted to a Global
            // base: Local-base inline GEPs that return `Ok(None)` set up DEFERRED machinery (vector-
            // word / streaming pointers) that is not usable as an immediate inline value.
            if matches!(gep.base.value, LlValue::Global(_)) {
                if let Some((id, _)) = self.values.get(&name) {
                    return Ok(*id);
                }
            }
            return Err(format!(
                "native emitter: getelementptr value `{name}` did not materialize a pointer"
            ));
        }
        if let Some(elem) = self.one_lane_vector_elem(ty)? {
            match value {
                LlValue::Splat(lane) => {
                    let lane_ty = self.resolve_type(&lane.ty)?;
                    if lane_ty != elem {
                        return Err(format!(
                            "native emitter: splat lane type {lane_ty:?} does not match {elem:?}"
                        ));
                    }
                    return self.value_id(&lane.value, &lane.ty);
                }
                LlValue::Vector(lanes) => {
                    let [lane] = lanes.as_slice() else {
                        return Err(format!(
                            "native emitter: vector literal has {} lanes, expected 1",
                            lanes.len()
                        ));
                    };
                    let lane_ty = self.resolve_type(&lane.ty)?;
                    if lane_ty != elem {
                        return Err(format!(
                            "native emitter: vector lane type {lane_ty:?} does not match {elem:?}"
                        ));
                    }
                    return self.value_id(&lane.value, &lane.ty);
                }
                _ => return self.value_id(value, ty),
            }
        }
        if let Some(id) = self.const_composite_id(value, &result_ty)? {
            return Ok(id);
        }
        let LlType::Vector(elem, count) = &result_ty else {
            if let LlValue::Splat(lane) = value {
                let lane_ty = self.resolve_type(&lane.ty)?;
                if lane_ty == result_ty {
                    return self.value_id(&lane.value, &lane.ty);
                }
            } else if let LlValue::Vector(lanes) = value {
                if let [lane] = lanes.as_slice() {
                    let lane_ty = self.resolve_type(&lane.ty)?;
                    if lane_ty == result_ty {
                        return self.value_id(&lane.value, &lane.ty);
                    }
                }
                return Err(format!(
                    "native emitter: vector literal used as non-vector type {result_ty:?}"
                ));
            }
            return self.value_id(value, ty);
        };
        if let Some(id) = self.const_composite_id(value, &result_ty)? {
            return Ok(id);
        }
        if let LlValue::Splat(lane) = value {
            let lane_ty = self.resolve_type(&lane.ty)?;
            if lane_ty != **elem {
                return Err(format!(
                    "native emitter: splat lane type {lane_ty:?} does not match {elem:?}"
                ));
            }
            let lane_id = self.value_id(&lane.value, &lane.ty)?;
            let result_type = self.type_id(&result_ty)?;
            let result = self.fresh();
            instructions.push(Self::inst(
                Op::CompositeConstruct,
                Some(result_type),
                Some(result),
                (0..*count).map(|_| Operand::IdRef(lane_id)).collect(),
            ));
            return Ok(result);
        }
        let LlValue::Vector(lanes) = value else {
            return self.value_id(value, ty);
        };
        if lanes.len() != *count as usize {
            return Err(format!(
                "native emitter: vector literal has {} lanes, expected {count}",
                lanes.len()
            ));
        }
        let result_type = self.type_id(&result_ty)?;
        let mut ops = Vec::with_capacity(lanes.len());
        for lane in lanes {
            let lane_ty = self.resolve_type(&lane.ty)?;
            if lane_ty != **elem {
                return Err(format!(
                    "native emitter: vector lane type {lane_ty:?} does not match {elem:?}"
                ));
            }
            ops.push(Operand::IdRef(self.value_id(&lane.value, &lane.ty)?));
        }
        let result = self.fresh();
        instructions.push(Self::inst(
            Op::CompositeConstruct,
            Some(result_type),
            Some(result),
            ops,
        ));
        Ok(result)
    }

    pub(super) fn one_lane_vector_elem(&self, ty: &LlType) -> Result<Option<LlType>, String> {
        match ty {
            LlType::Named(name) => {
                let aliased = self
                    .ir
                    .types
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("native emitter: unknown named type {name}"))?;
                self.one_lane_vector_elem(&aliased)
            }
            LlType::Vector(elem, 1) => self.resolve_type(elem).map(Some),
            _ => Ok(None),
        }
    }

    pub(super) fn const_composite_id(
        &mut self,
        value: &LlValue,
        ty: &LlType,
    ) -> Result<Option<Word>, String> {
        let ty = self.resolve_type(ty)?;
        let (elem, count, is_array) = match &ty {
            LlType::Vector(elem, count) => (elem, count, false),
            LlType::Array(elem, count) => (elem, count, true),
            LlType::Struct(fields) => {
                let LlValue::Struct(values) = value else {
                    if matches!(value, LlValue::Vector(_) | LlValue::Array(_)) {
                        return Err(
                            "native emitter: non-struct aggregate literal used as struct type"
                                .into(),
                        );
                    }
                    return Ok(None);
                };
                if values.len() != fields.len() {
                    return Err(format!(
                        "native emitter: struct literal has {} fields, expected {}",
                        values.len(),
                        fields.len()
                    ));
                }
                let mut constituents = Vec::with_capacity(values.len());
                for (index, (field, expected_ty)) in values.iter().zip(fields).enumerate() {
                    let field_ty = self.resolve_type(&field.ty)?;
                    let expected_ty = self.resolve_type(expected_ty)?;
                    if field_ty != expected_ty {
                        return Err(format!(
                            "native emitter: struct field {index} type {field_ty:?} does not match {expected_ty:?}"
                        ));
                    }
                    let Some(field_id) =
                        self.const_scalar_or_composite_id(&field.value, &field.ty)?
                    else {
                        return Ok(None);
                    };
                    constituents.push(field_id);
                }
                return self
                    .const_composite_with_constituents(&ty, constituents)
                    .map(Some);
            }
            _ => return Ok(None),
        };
        let type_name = if is_array { "array" } else { "vector" };
        let lane_name = if is_array { "element" } else { "lane" };
        let lanes = match value {
            LlValue::Splat(lane) => {
                let lane_ty = self.resolve_type(&lane.ty)?;
                if lane_ty != **elem {
                    return Err(format!(
                        "native emitter: splat {lane_name} type {lane_ty:?} does not match {elem:?}"
                    ));
                }
                let Some(lane_id) = self.const_scalar_or_composite_id(&lane.value, &lane.ty)?
                else {
                    return Ok(None);
                };
                return self
                    .const_composite_with_constituents(&ty, (0..*count).map(|_| lane_id).collect())
                    .map(Some);
            }
            LlValue::Vector(lanes) if !is_array => lanes,
            LlValue::Array(lanes) if is_array => lanes,
            LlValue::Vector(_) => {
                return Err("native emitter: vector literal used as array type".into());
            }
            LlValue::Array(_) => {
                return Err("native emitter: array literal used as vector type".into());
            }
            LlValue::Struct(_) => {
                return Err("native emitter: struct literal used as vector/array type".into());
            }
            _ => return Ok(None),
        };
        if lanes.len() != *count as usize {
            return Err(format!(
                "native emitter: {type_name} literal has {} {lane_name}s, expected {count}",
                lanes.len()
            ));
        }
        let mut constituents = Vec::with_capacity(*count as usize);
        for lane in lanes {
            let lane_ty = self.resolve_type(&lane.ty)?;
            if lane_ty != **elem {
                return Err(format!(
                    "native emitter: {type_name} {lane_name} type {lane_ty:?} does not match {elem:?}"
                ));
            }
            let Some(lane_id) = self.const_scalar_or_composite_id(&lane.value, &lane.ty)? else {
                return Ok(None);
            };
            constituents.push(lane_id);
        }
        self.const_composite_with_constituents(&ty, constituents)
            .map(Some)
    }

    fn const_composite_with_constituents(
        &mut self,
        ty: &LlType,
        constituents: Vec<Word>,
    ) -> Result<Word, String> {
        let ty = self.resolve_type(ty)?;
        let key = (ty.clone(), constituents.clone());
        if let Some(id) = self.interner.composite_constants.get(&key) {
            return Ok(*id);
        }
        let result_type = self.type_id(&ty)?;
        let result = self.fresh();
        self.module.types_global_values.push(Self::inst(
            Op::ConstantComposite,
            Some(result_type),
            Some(result),
            constituents.into_iter().map(Operand::IdRef).collect(),
        ));
        self.interner.composite_constants.insert(key, result);
        Ok(result)
    }

    fn const_scalar_or_composite_id(
        &mut self,
        value: &LlValue,
        ty: &LlType,
    ) -> Result<Option<Word>, String> {
        if let Some(id) = self.const_scalar_id(value, ty)? {
            return Ok(Some(id));
        }
        self.const_composite_id(value, ty)
    }

    pub(super) fn const_initializer_id(
        &mut self,
        value: &LlValue,
        ty: &LlType,
    ) -> Result<Word, String> {
        if let Some(id) = self.const_scalar_or_composite_id(value, ty)? {
            return Ok(id);
        }
        Err(format!(
            "native emitter: unsupported global initializer {value:?} for {ty:?}"
        ))
    }

    pub(super) fn const_scalar_id(
        &mut self,
        value: &LlValue,
        ty: &LlType,
    ) -> Result<Option<Word>, String> {
        match value {
            LlValue::Local(_)
            | LlValue::Global(_)
            | LlValue::Vector(_)
            | LlValue::Array(_)
            | LlValue::Struct(_)
            | LlValue::Splat(_) => Ok(None),
            LlValue::Undef => self.const_null(ty).map(Some),
            _ => self.value_id(value, ty).map(Some),
        }
    }

    pub(super) fn const_int(&mut self, bits: u32, value: u64) -> Result<Word, String> {
        // Mask to the *logical* width (so `i2 -1` → low bits `0b11`), then emit under the
        // SPIR-V-legal container width. Key the interner on the legal width so `const_int(2, 3)`
        // and `const_int(8, 3)` share one `OpConstant` (same SPIR-V type + value).
        let legal = spirv_int_width(bits)?;
        let encoded = if bits >= 64 {
            value
        } else {
            value & ((1u64 << bits) - 1)
        };
        if legal == 32 {
            return self.const_uint(encoded as u32);
        }
        if let Some(id) = self.interner.int_constants.get(&(legal, encoded)) {
            return Ok(*id);
        }
        let ty = self.type_id(&LlType::Int(legal))?;
        let id = self.fresh();
        let operands = if legal <= 32 {
            vec![Operand::LiteralBit32(encoded as u32)]
        } else if legal == 64 {
            vec![Operand::LiteralBit64(encoded)]
        } else {
            return Err(format!(
                "native emitter: unsupported integer constant width i{bits}"
            ));
        };
        self.module.types_global_values.push(Self::inst(
            Op::Constant,
            Some(ty),
            Some(id),
            operands,
        ));
        self.interner.int_constants.insert((legal, encoded), id);
        Ok(id)
    }

    pub(super) fn const_bool(&mut self, value: bool) -> Result<Word, String> {
        if let Some(id) = self.interner.bool_constants.get(&value) {
            return Ok(*id);
        }
        let ty = self.type_id(&LlType::Bool)?;
        let id = self.fresh();
        let op = if value {
            Op::ConstantTrue
        } else {
            Op::ConstantFalse
        };
        self.module
            .types_global_values
            .push(Self::inst(op, Some(ty), Some(id), vec![]));
        self.interner.bool_constants.insert(value, id);
        Ok(id)
    }

    pub(super) fn const_null(&mut self, ty: &LlType) -> Result<Word, String> {
        let ty = self.resolve_type(ty)?;
        if let Some(id) = self.interner.null_constants.get(&ty) {
            return Ok(*id);
        }
        let type_id = self.type_id(&ty)?;
        let id = self.fresh();
        self.module.types_global_values.push(Self::inst(
            Op::ConstantNull,
            Some(type_id),
            Some(id),
            vec![],
        ));
        self.interner.null_constants.insert(ty, id);
        Ok(id)
    }

    pub(super) fn const_signed_int(&mut self, bits: u32, value: i64) -> Result<Word, String> {
        let encoded = if bits >= 64 {
            value as u64
        } else {
            (value as u64) & ((1u64 << bits) - 1)
        };
        self.const_int(bits, encoded)
    }

    pub(super) fn int_constant_like(&mut self, ty: &LlType, value: i64) -> Result<Word, String> {
        match ty {
            LlType::Int(bits) => self.const_signed_int(*bits, value),
            LlType::Vector(elem, lanes) => {
                let LlType::Int(bits) = elem.as_ref() else {
                    return Err(format!(
                        "native emitter: integer constant requested for non-int vector {ty:?}"
                    ));
                };
                let scalar = self.const_signed_int(*bits, value)?;
                let result_type = self.type_id(ty)?;
                let result = self.fresh();
                self.module.types_global_values.push(Self::inst(
                    Op::ConstantComposite,
                    Some(result_type),
                    Some(result),
                    (0..*lanes).map(|_| Operand::IdRef(scalar)).collect(),
                ));
                Ok(result)
            }
            other => Err(format!(
                "native emitter: integer constant requested for non-int type {other:?}"
            )),
        }
    }

    pub(super) fn const_float32(&mut self, value: f32) -> Result<Word, String> {
        let bits = value.to_bits();
        if let Some(id) = self.interner.float32_constants.get(&bits) {
            return Ok(*id);
        }
        let ty = self.type_id(&LlType::Float)?;
        let id = self.fresh();
        self.module.types_global_values.push(Self::inst(
            Op::Constant,
            Some(ty),
            Some(id),
            vec![Operand::LiteralBit32(bits)],
        ));
        self.interner.float32_constants.insert(bits, id);
        Ok(id)
    }

    pub(super) fn const_float16_bits(&mut self, bits: u16) -> Result<Word, String> {
        if let Some(id) = self.interner.float16_constants.get(&bits) {
            return Ok(*id);
        }
        let ty = self.type_id(&LlType::Half)?;
        let id = self.fresh();
        self.module.types_global_values.push(Self::inst(
            Op::Constant,
            Some(ty),
            Some(id),
            vec![Operand::LiteralBit32(bits as u32)],
        ));
        self.interner.float16_constants.insert(bits, id);
        Ok(id)
    }
}
