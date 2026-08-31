//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    pub(in crate::native::emitter) fn emit_raw_load(
        &mut self,
        result: Word,
        ty: &LlType,
        raw: &RawBufferOffset,
        access_align: Option<u64>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        if raw.unmodelable {
            return Err("native emitter: raw buffer offset is not modelable".into());
        }
        if raw.device_addr_base.is_some() {
            let ty = self.resolve_type(ty)?;
            return self.emit_device_addr_load(result, &ty, raw, instructions);
        }
        match self.resolve_type(ty)? {
            LlType::Vector(elem, lanes) => {
                let elem = self.resolve_type(&elem)?;
                let (elem_size, _) = self.raw_type_size_align(&elem)?;
                let mut lane_ids = Vec::with_capacity(lanes as usize);
                for lane in 0..lanes {
                    let lane_id = self.fresh();
                    self.emit_raw_scalar_load(
                        lane_id,
                        &elem,
                        raw,
                        lane as u64 * elem_size,
                        access_align,
                        instructions,
                    )?;
                    lane_ids.push(lane_id);
                }
                let result_type = self.type_id(&LlType::Vector(Box::new(elem), lanes))?;
                instructions.push(Self::inst(
                    Op::CompositeConstruct,
                    Some(result_type),
                    Some(result),
                    lane_ids.into_iter().map(Operand::IdRef).collect(),
                ));
            }
            scalar => {
                self.emit_raw_scalar_load(result, &scalar, raw, 0, access_align, instructions)?
            }
        }
        Ok(())
    }

    /// Materialize the 64-bit device byte ADDRESS of a `device_addr_base`-rooted offset:
    /// `base + const_off + Σ(index * stride)`, all in `Int(64)`. `base` is the address value the kernel
    /// loaded from device memory; `const_off`/`dyn_terms` are the byte offsets folded in by GEP. Used by
    /// the BDA leaf load/store below to feed `OpConvertUToPtr`. Byte-exact: the same address arithmetic
    /// the descriptor path applies as a logical access chain, computed here as an explicit integer sum.
    pub(in crate::native::emitter) fn materialize_device_address(
        &mut self,
        raw: &RawBufferOffset,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Word, String> {
        let base = raw
            .device_addr_base
            .ok_or("native emitter: device address offset has no base")?;
        let i64_ty = self.type_id(&LlType::Int(64))?;
        let mut addr = base;
        if raw.const_off != 0 {
            let c = self.const_signed_int(64, raw.const_off)?;
            let sum = self.fresh();
            instructions.push(Self::inst(
                Op::IAdd,
                Some(i64_ty),
                Some(sum),
                vec![Operand::IdRef(addr), Operand::IdRef(c)],
            ));
            addr = sum;
        }
        // Clone the dyn-term descriptors so the value materialization can borrow `self` mutably.
        let terms = raw.dyn_terms.clone();
        for (tv, stride) in &terms {
            let idx = self.value_id_in(&tv.value, &tv.ty, instructions)?;
            let idx64 = match self.resolve_type(&tv.ty)? {
                LlType::Int(64) => idx,
                _ => {
                    // GEP indices are signed; sign-extend to 64-bit for the address sum.
                    let w = self.fresh();
                    instructions.push(Self::inst(
                        Op::SConvert,
                        Some(i64_ty),
                        Some(w),
                        vec![Operand::IdRef(idx)],
                    ));
                    w
                }
            };
            let term = if *stride == 1 {
                idx64
            } else {
                let s = self.const_signed_int(64, *stride)?;
                let m = self.fresh();
                instructions.push(Self::inst(
                    Op::IMul,
                    Some(i64_ty),
                    Some(m),
                    vec![Operand::IdRef(idx64), Operand::IdRef(s)],
                ));
                m
            };
            let sum = self.fresh();
            instructions.push(Self::inst(
                Op::IAdd,
                Some(i64_ty),
                Some(sum),
                vec![Operand::IdRef(addr), Operand::IdRef(term)],
            ));
            addr = sum;
        }
        Ok(addr)
    }

    pub(in crate::native::emitter) fn materialize_reserved_bda_address(
        &mut self,
        name: &str,
        raw: &RawBufferOffset,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let address_name = bda_address_name(name);
        let Some((reserved, _)) = self.values.get(&address_name).cloned() else {
            return Ok(());
        };
        let address = self.materialize_device_address(raw, instructions)?;
        if address != reserved {
            let address_ty = self.type_id(&LlType::Int(64))?;
            instructions.push(Self::inst(
                Op::CopyObject,
                Some(address_ty),
                Some(reserved),
                vec![Operand::IdRef(address)],
            ));
        }
        self.bda_address_values.insert(reserved);
        Ok(())
    }

    /// The `Aligned` memory-operand value for a PhysicalStorageBuffer load/store of `ty`: the component
    /// (scalar) size in bytes, a power of two (`VUID-StandaloneSpirv-PhysicalStorageBuffer64-06314`
    /// requires the operand to be a power of two and present).
    pub(in crate::native::emitter) fn device_addr_align(&mut self, ty: &LlType) -> u32 {
        let scalar = match ty {
            LlType::Vector(elem, _) => elem.as_ref().clone(),
            other => other.clone(),
        };
        match scalar {
            LlType::Int(64) | LlType::Ptr(_) => 8,
            LlType::Int(32) | LlType::Float => 4,
            LlType::Int(16) | LlType::Half | LlType::BFloat => 2,
            LlType::Int(8) => 1,
            _ => 4,
        }
    }

    /// BDA leaf load: read `ty` from the device address of a `device_addr_base`-rooted offset via
    /// `OpConvertUToPtr` to a `PhysicalStorageBuffer` pointer + an `Aligned` `OpLoad`. Byte-exact (a
    /// typed read at the exact loaded address + folded offset).
    pub(in crate::native::emitter) fn emit_device_addr_load(
        &mut self,
        result: Word,
        ty: &LlType,
        raw: &RawBufferOffset,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        self.used_device_address = true;
        let addr = self.materialize_device_address(raw, instructions)?;
        let result_ty = self.type_id(ty)?;
        let ptr_ty = self.ptr_type_id(StorageClass::PhysicalStorageBuffer, ty)?;
        let p = self.fresh();
        instructions.push(Self::inst(
            Op::ConvertUToPtr,
            Some(ptr_ty),
            Some(p),
            vec![Operand::IdRef(addr)],
        ));
        let align = self.device_addr_align(ty);
        instructions.push(Self::inst(
            Op::Load,
            Some(result_ty),
            Some(result),
            vec![
                Operand::IdRef(p),
                Operand::MemoryAccess(spirv::MemoryAccess::ALIGNED),
                Operand::LiteralBit32(align),
            ],
        ));
        Ok(())
    }

    /// BDA leaf store: write `value: ty` to the device address of a `device_addr_base`-rooted offset via
    /// `OpConvertUToPtr` + an `Aligned` `OpStore`. Byte-exact (writes the exact bytes at the loaded
    /// address + folded offset).
    pub(in crate::native::emitter) fn emit_device_addr_store(
        &mut self,
        ty: &LlType,
        value: Word,
        raw: &RawBufferOffset,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        self.used_device_address = true;
        let addr = self.materialize_device_address(raw, instructions)?;
        let ptr_ty = self.ptr_type_id(StorageClass::PhysicalStorageBuffer, ty)?;
        let p = self.fresh();
        instructions.push(Self::inst(
            Op::ConvertUToPtr,
            Some(ptr_ty),
            Some(p),
            vec![Operand::IdRef(addr)],
        ));
        let align = self.device_addr_align(ty);
        instructions.push(Self::inst(
            Op::Store,
            None,
            None,
            vec![
                Operand::IdRef(p),
                Operand::IdRef(value),
                Operand::MemoryAccess(spirv::MemoryAccess::ALIGNED),
                Operand::LiteralBit32(align),
            ],
        ));
        Ok(())
    }
}
