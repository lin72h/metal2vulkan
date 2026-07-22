//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    pub(in crate::native::emitter) fn pointer_aware_type_id(
        &mut self,
        ty: &LlType,
        meta: Option<&PointerMeta>,
    ) -> Result<Word, String> {
        if let Some(PointerMeta {
            storage,
            pointee: Some(pointee),
        }) = meta
        {
            self.ptr_type_id(*storage, pointee)
        } else {
            self.type_id(ty)
        }
    }

    pub(in crate::native::emitter) fn pointer_merge_meta(
        &self,
        values: &[&LlValue],
        ty: &LlType,
    ) -> Result<Option<PointerMeta>, String> {
        let LlType::Ptr(addrspace) = ty else {
            return Ok(None);
        };
        let mut merged = PointerMeta {
            storage: llvm_pointer_storage(*addrspace)?,
            pointee: None,
        };
        let mut saw_meta = false;
        for value in values {
            let Some(meta) = self.pointer_meta_for_value(value, *addrspace)? else {
                continue;
            };
            if saw_meta && merged.storage != meta.storage {
                return Err(format!(
                    "native emitter: pointer merge storage mismatch {:?} vs {:?} at {value:?}",
                    merged.storage, meta.storage
                ));
            }
            saw_meta = true;
            merged.storage = meta.storage;
            if let Some(pointee) = meta.pointee {
                match &merged.pointee {
                    Some(existing) if existing != &pointee => {
                        return Err(format!(
                            "native emitter: pointer merge pointee mismatch {existing:?} vs {pointee:?}"
                        ));
                    }
                    None => merged.pointee = Some(pointee),
                    _ => {}
                }
            }
        }
        Ok(Some(merged))
    }

    pub(in crate::native::emitter) fn pointer_meta_for_value(
        &self,
        value: &LlValue,
        addrspace: u32,
    ) -> Result<Option<PointerMeta>, String> {
        match value {
            LlValue::Local(name) => {
                // M-A2 def-site recording: for a member of a carrier-uniform pointer
                // network, report the network's recorded pointee so a phi/select merge reconciles on it
                // instead of the byte-view `Int(8)` the raw recording flattened one arm to. Empty for a
                // pointer that is not a member of a carrier-uniform network, so this is inert by default.
                let network_pointee = self.network_pointees.get(name);
                Ok(self
                    .pointer_storage
                    .get(name)
                    .copied()
                    .map(|storage| PointerMeta {
                        storage,
                        pointee: network_pointee
                            .or_else(|| self.pointer_pointees.get(name))
                            .cloned(),
                    }))
            }
            LlValue::Global(name) => {
                // A global's storage class is intrinsic to its declared address space: an
                // `addrspace(3)` global is a threadgroup variable (Workgroup), not Private. Classifying
                // it as Workgroup here — matching `pointer_storage_for`'s `Global if addrspace == 3`
                // rule — lets a pointer merge over two Workgroup pointers (one a `%local`, one a
                // threadgroup global) agree on storage instead of failing `pointer merge storage
                // mismatch Workgroup vs Private`. Only addrspace(3) is special-cased: addrspace(0/2)
                // globals keep the historical Private classification here (reclassifying addrspace(2)
                // constants to UniformConstant surfaces a separate pointee-type mismatch in
                // constant-array selects — a regression, out of scope for this storage fix).
                let storage = match self.global_values.get(name) {
                    Some((_, LlType::Ptr(3))) => StorageClass::Workgroup,
                    _ => StorageClass::Private,
                };
                Ok(Some(PointerMeta {
                    storage,
                    pointee: None,
                }))
            }
            LlValue::Gep(gep) => {
                let LlType::Ptr(base_addrspace) = self.resolve_type(&gep.base.ty)? else {
                    return Err(format!(
                        "native emitter: getelementptr base is not a pointer: {:?}",
                        gep.base.ty
                    ));
                };
                Ok(Some(PointerMeta {
                    storage: self.pointer_storage_for(&gep.base.value, base_addrspace)?,
                    pointee: Some(gep_pointee(
                        &self.resolve_type(&gep.source_ty)?,
                        &gep.indices,
                    )?),
                }))
            }
            LlValue::Undef | LlValue::Zero => Ok(None),
            _ => Ok(Some(PointerMeta {
                storage: llvm_pointer_storage(addrspace)?,
                pointee: None,
            })),
        }
    }

    /// Whether a local pointer `name` participates in a POINTER MERGE — the reconciliation points
    /// (phi, select) that read the un-upgraded `pointer_pointees` and so must not see a pointee that
    /// the whole-vs-part upgrade (M-A2(b)) widened on one arm only. Covers phi results, phi incoming
    /// values, and both the result and the arm operands of a pointer-select.
    pub(in crate::native::emitter) fn pointer_in_pointer_merge(&self, name: &str) -> bool {
        if self.pointer_phi_values.contains(name)
            || self.pointer_phi_incoming_values.contains(name)
            || self.selected_pointers.contains_key(name)
        {
            return true;
        }
        self.selected_pointers.values().any(|sp| {
            matches!(&sp.true_value, LlValue::Local(n) if n == name)
                || matches!(&sp.false_value, LlValue::Local(n) if n == name)
        })
    }

    pub(in crate::native::emitter) fn pointer_pointee_for_value(
        &self,
        value: &LlValue,
    ) -> Result<Option<LlType>, String> {
        match value {
            LlValue::Local(name) => {
                if let Some(pointee) = self.pointer_pointees.get(name) {
                    // M2 (S20) byte→real upgrade: the emitter recorded the `Int(8)` BYTE PLACEHOLDER
                    // for this local pointer (a def-time typing gap, not a real `i8*`), yet the
                    // use-implied carrier knows the concrete element type the pointer is actually
                    // dereferenced as. Prefer the carrier's real type — EXCEPT where the byte view is
                    // deliberate: a `raw_offsets` (byte-cursor addressed) or `unmodeled_pointers`
                    // pointer is intentionally byte-typed and entangled with the raw addressing model,
                    // so its `Int(8)` stays authoritative. This retires part of the byte-placeholder
                    // fallback the emitter doc names as debt, without touching the raw-addressing path.
                    //
                    // Also EXCLUDE any pointer that carries a byte (`i8`) view in its uses
                    // (`tir_byte_view_pointers`): a pointer dereferenced BOTH as a byte cursor
                    // (`getelementptr i8` → `uchar`-result `OpPtrAccessChain`) AND as a wider type. Its
                    // carrier resolves to the wider type, but the byte cursor still expects a
                    // `uchar`-pointee base, so upgrading the pointee strands it and emits globally-invalid
                    // SPIR-V (`OpPtrAccessChain result %uchar does not match indexing into base %<wide>` —
                    // observed on 01/9c30da00). `raw_offsets`/`unmodeled_pointers` cover the emitter's
                    // OWN byte-addressing models; this covers the plain-`i8`-GEP byte cursor that lands in
                    // NEITHER table. Only the pure-widening subset (no byte view) stays upgradeable.
                    if *pointee == LlType::Int(8)
                        && !self.raw_offsets.contains_key(name)
                        && !self.unmodeled_pointers.contains(name)
                        && !self.tir_byte_view_pointers.contains(name)
                    {
                        if let Some(carrier) = self.tir_use_pointees.get(name) {
                            let carrier = self.resolve_type(carrier)?;
                            if carrier != LlType::Int(8) {
                                return Ok(Some(carrier));
                            }
                        }
                    }
                    // M-A2(b) whole-vs-part upgrade: the emitter recorded a plain SCALAR pointee `S`
                    // for this local pointer (the def site only saw the element type), yet the
                    // use-implied carrier proves the pointer is dereferenced as the WHOLE composite
                    // `Vector(S,N)` / `[N x S]` of that same scalar (a load/store of the whole vector,
                    // a whole-aggregate GEP source). This is a pure granularity widening — same base
                    // type, part→whole — so prefer the carrier's composite.
                    //
                    // The same byte-view exclusions as the M-A1 subset apply (a deliberately
                    // byte/raw-typed pointer stays authoritative). ADDITIONALLY exclude any pointer
                    // that PARTICIPATES IN A POINTER MERGE (a phi result / phi incoming / select arm
                    // or result): a merge reconciles its arms' pointees through `pointer_meta_for_value`
                    // / the select direct-load-store, both of which read the UN-upgraded
                    // `pointer_pointees` (scalar). Upgrading one merged arm's access-chain pointee to
                    // the wider whole while the merge result stays scalar strands the merge
                    // (`pointer merge pointee mismatch` / `selected … type mismatch`) — dead-end #8:
                    // the merge is pointee-authoritative, the carrier is a def-site-only channel.
                    // Only merge-free pure-whole-view pointers are upgradeable.
                    if crate::env_vars::whole_part()
                        && is_scalar_pointee(pointee)
                        && !self.raw_offsets.contains_key(name)
                        && !self.unmodeled_pointers.contains(name)
                        && !self.tir_byte_view_pointers.contains(name)
                        && !self.pointer_in_pointer_merge(name)
                    {
                        if let Some(carrier) = self.tir_use_pointees.get(name) {
                            // `resolve_type` already recurses into vector/array elements, so the
                            // resolved carrier's element is fully resolved — the shape test is pure.
                            let carrier = self.resolve_type(carrier)?;
                            if whole_part_widens(&carrier, pointee) {
                                return Ok(Some(carrier));
                            }
                        }
                    }
                    // M-A2(a) Float<->Int reinterpret upgrade — DIAGNOSTIC/UNSOUND, default-off, do NOT
                    // flip: the emitter recorded a SCALAR pointee `S`, but the use-implied carrier proves
                    // the pointer is dereferenced as a DIFFERENT scalar of the SAME bit width (e.g.
                    // `Int(32)` recorded, loaded as `Float`; or the reverse). Unlike the Int(8) placeholder
                    // (the Int(8) definitionally-wrong stand-in, always upgraded) or whole-vs-part (WHOLE_PART, unambiguous
                    // widening), BOTH scalars here are load-bearing and a pointer can be legitimately
                    // reinterpreted (loaded as int in one place, float in another) — so preferring the
                    // carrier is NOT sound: the carrier propagation collapsed the reinterpret to one arm.
                    // G7 confirms it MISCOMPILES the topk_common_matrix_float family. Kept only as the
                    // enumeration substrate (`--reinterp-real-check`) for the eventual DEF-SITE version
                    // that knows a pointer's unambiguous deref type; never flip in this read-side form.
                    // Same byte-view + pointer-MERGE-participant exclusions apply (dead-end #8).
                    if crate::env_vars::reinterp_real()
                        && is_scalar_pointee(pointee)
                        && !self.raw_offsets.contains_key(name)
                        && !self.unmodeled_pointers.contains(name)
                        && !self.tir_byte_view_pointers.contains(name)
                        && !self.pointer_in_pointer_merge(name)
                    {
                        if let Some(carrier) = self.tir_use_pointees.get(name) {
                            let carrier = self.resolve_type(carrier)?;
                            if reinterp_compatible(&carrier, pointee) {
                                return Ok(Some(carrier));
                            }
                        }
                    }
                    return Ok(Some(pointee.clone()));
                }
                // M2 (S20): where the emitter recorded no pointee for a local pointer, consult the
                // use-implied pointee carrier (`tir_use_pointees`, the type the pointer is actually
                // dereferenced as — load/store/GEP-source/atomic element, propagated across
                // select/phi/freeze). This fills exactly the gaps the def-time side-tables left empty
                // (the census's carrier-fills-emitter-None set), the first production consumer of the
                // carrier the whole pointer-typing rewrite is built on. Every emitter-recorded pointee
                // is still authoritative (checked first), so this only ADDS answers, never overrides.
                Ok(self.tir_use_pointees.get(name).cloned())
            }
            LlValue::Global(name) => {
                if let Some(pointee) = self.pointer_pointees.get(name) {
                    return Ok(Some(pointee.clone()));
                }
                self.global_values
                    .get(name)
                    .map(|(_, ty)| match ty {
                        LlType::Ptr(_) => None,
                        other => Some(other.clone()),
                    })
                    .ok_or_else(|| format!("native emitter: unknown global value {name}"))
            }
            LlValue::Gep(gep) => Ok(Some(gep_pointee(
                &self.resolve_type(&gep.source_ty)?,
                &gep.indices,
            )?)),
            _ => Ok(None),
        }
    }

    /// Whether a pointer phi is a fully-RAW single-root induction: every arm is either a pointer
    /// already raw-modeled under the template's root (same root + addrspace, modelable) or the
    /// loop-carried step — a forward GEP whose base is the phi itself. Such a phi is exactly the
    /// shape `emit_raw_pointer_phi` models (a byte/word index phi over one raw root), and the ONLY
    /// shape where the `network_pointees` defer must not steal the claim: the typed merge path has
    /// no way to express the network pointee against the root's raw block declaration.
    pub(in crate::native::emitter) fn raw_only_induction_phi(
        &self,
        name: &str,
        incoming: &[(LlValue, String)],
        template: &RawBufferOffset,
    ) -> bool {
        incoming.iter().all(|(value, _)| {
            let LlValue::Local(incoming_name) = value else {
                return false;
            };
            if let Some(raw) = self.raw_offsets.get(incoming_name) {
                return raw.root == template.root
                    && raw.addrspace == template.addrspace
                    && !raw.unmodelable;
            }
            self.forward_geps
                .get(incoming_name)
                .is_some_and(|gep| matches!(&gep.base.value, LlValue::Local(base) if base == name))
        })
    }

    pub(in crate::native::emitter) fn emit_raw_pointer_phi(
        &mut self,
        name: &str,
        incoming: &[(LlValue, String)],
        result_ty: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let LlType::Ptr(addrspace) = result_ty else {
            return Ok(false);
        };
        let Some(template) = incoming.iter().find_map(|(value, _)| match value {
            LlValue::Local(name) => self.raw_offsets.get(name).cloned(),
            _ => None,
        }) else {
            return Ok(false);
        };
        if template.unmodelable {
            return Ok(false);
        }
        // M-A2 def-site recording: a phi whose network the carrier types uniformly
        // (seeded in `network_pointees`) must reconcile on that real type via `pointer_merge_meta`, not
        // be flattened to a byte/word index here — the raw path is what strands `05/b00a8a8d`'s
        // `float*` incoming under a `uchar*` result. Defer so the typed merge path claims it.
        // EXCEPT for a fully-RAW single-root induction: when every arm is raw-modeled under the
        // template's root (or the loop-carried forward GEP off the phi itself), the typed merge path
        // CANNOT express the phi — its arms materialize against the root's RAW declaration
        // (`{ [0 x i32] }` block), so the seeded network pointee yields a mistyped element chain on one
        // arm and the raw runtime-array base value on the other (`OpPhi` result/incoming type mismatch
        // — the 4-row MPSRNNLSTMRecursionCombined banked floor family). The raw byte/word-index phi
        // models the shape exactly, so it keeps the claim.
        if self.network_pointees.contains_key(name)
            && !self.raw_only_induction_phi(name, incoming, &template)
        {
            return Ok(false);
        }
        let all_incoming_raw = incoming.iter().all(|(value, _)| match value {
            LlValue::Local(name) => self.raw_offsets.contains_key(name),
            _ => false,
        });
        // A raw phi can use word indices only when EVERY arm is word-aligned. Choosing the
        // template's representation alone is insufficient: a prior byte-indexed phi/GEP arm may
        // carry the same raw root at an offset whose alignment is not statically proven. Merge
        // that shape as byte offsets; `emit_raw_word_pointer_for_access` converts the byte cursor
        // only at a later access whose alignment proves it legal.
        let word_indexed = all_incoming_raw
            && incoming.iter().all(|(value, _)| {
                matches!(value, LlValue::Local(name) if self
                    .raw_offsets
                    .get(name)
                    .is_some_and(|raw| self.raw_pointer_word_aligned(raw)))
            });

        let index_ty = LlType::Int(32);
        let index_name = if word_indexed {
            raw_word_index_name(name)
        } else {
            raw_byte_index_name(name)
        };
        let result = self.result_id(&index_name, &index_ty)?;
        let result_type = self.type_id(&index_ty)?;
        let mut ops = Vec::new();
        let mut seen_incoming: HashMap<Word, Word> = HashMap::new();
        for (value, label) in incoming {
            let value_id = match value {
                LlValue::Local(incoming_name) => {
                    if let Some(raw) = self.raw_offsets.get(incoming_name).cloned() {
                        if raw.root != template.root
                            || raw.addrspace != template.addrspace
                            || raw.unmodelable
                            || (word_indexed && !self.raw_pointer_word_aligned(&raw))
                        {
                            return Ok(false);
                        }
                        let incoming_index_name = if word_indexed {
                            raw_word_index_name(incoming_name)
                        } else {
                            raw_byte_index_name(incoming_name)
                        };
                        if self.values.contains_key(&incoming_index_name) {
                            self.value_id(&LlValue::Local(incoming_index_name), &LlType::Int(32))?
                        } else if word_indexed && raw.dyn_terms.is_empty() {
                            self.emit_raw_word_index(&raw, 0, instructions)?
                        } else if !word_indexed {
                            self.emit_raw_byte_index(&raw, 0, instructions)?
                        } else {
                            return Ok(false);
                        }
                    } else if self.values.contains_key(incoming_name) {
                        return Ok(false);
                    } else {
                        self.phi_value_id(
                            &LlValue::Local(if word_indexed {
                                raw_word_index_name(incoming_name)
                            } else {
                                raw_byte_index_name(incoming_name)
                            }),
                            &index_ty,
                            instructions,
                        )?
                    }
                }
                _ => return Ok(false),
            };
            let label_id = self.label_id(label)?;
            if let Some(existing) = seen_incoming.insert(label_id, value_id) {
                if existing != value_id {
                    return Err(format!(
                        "native emitter: raw pointer phi has multiple offsets from predecessor {label}"
                    ));
                }
                continue;
            }
            ops.push(Operand::IdRef(value_id));
            ops.push(Operand::IdRef(label_id));
        }
        instructions.push(Self::inst(Op::Phi, Some(result_type), Some(result), ops));
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
                        ty: index_ty,
                        value: LlValue::Local(index_name),
                    },
                    if word_indexed { 4 } else { 1 },
                )],
                root: template.root.clone(),
                addrspace: template.addrspace,
                unmodelable: false,
                device_addr_base: template.device_addr_base,
            },
        );
        self.define_unmodeled_pointer_value(name, *addrspace, &LlType::Int(8))?;
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_unmodeled_pointer_phi(
        &mut self,
        name: &str,
        incoming: &[(LlValue, String)],
        result_ty: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<bool, String> {
        let LlType::Ptr(addrspace) = result_ty else {
            return Ok(false);
        };
        let has_unmodeled = incoming.iter().any(|(value, _)| match value {
            LlValue::Local(name) => {
                self.unmodeled_pointers.contains(name)
                    || (self.pointer_phi_values.contains(name) && !self.values.contains_key(name))
            }
            _ => false,
        });
        if !has_unmodeled {
            return Ok(false);
        }
        self.emit_pointer_nullness_phi(name, incoming, result_ty, instructions)?;
        self.define_unmodeled_pointer_value(name, *addrspace, &LlType::Int(8))?;
        Ok(true)
    }

    pub(in crate::native::emitter) fn emit_pointer_phi_provenance(
        &mut self,
        name: &str,
        incoming: &[(LlValue, String)],
        instructions: &mut Vec<Instruction>,
    ) -> Result<Option<GepProvenance>, String> {
        let Some(template) = self.pointer_phi_template_provenance(name, incoming)? else {
            return Ok(None);
        };
        if template.indices.len() != 1 {
            return Ok(None);
        }
        let index_ty = template.indices[0].ty.clone();
        let index_name = pointer_index_name(name);
        let result = self.result_id(&index_name, &index_ty)?;
        let result_type = self.type_id(&index_ty)?;
        let mut ops = Vec::new();
        let mut seen_incoming: HashMap<Word, Word> = HashMap::new();
        for (value, label) in incoming {
            let Some(provenance) =
                self.provenance_for_pointer_value(value, Some(&template), Some(&index_ty))?
            else {
                return Ok(None);
            };
            if !compatible_pointer_provenance(&template, &provenance)
                || provenance.indices.len() != 1
            {
                return Ok(None);
            }
            let index = &provenance.indices[0];
            let value_id = self.phi_value_id(&index.value, &index.ty, instructions)?;
            let label_id = self.label_id(label)?;
            if let Some(existing) = seen_incoming.insert(label_id, value_id) {
                if existing != value_id {
                    return Err(format!(
                        "native emitter: pointer index phi has multiple values from predecessor {label}"
                    ));
                }
                continue;
            }
            ops.push(Operand::IdRef(value_id));
            ops.push(Operand::IdRef(label_id));
        }
        instructions.push(Self::inst(Op::Phi, Some(result_type), Some(result), ops));
        Ok(Some(GepProvenance {
            root: template.root,
            addrspace: template.addrspace,
            source_ty: template.source_ty,
            indices: vec![TypedValue {
                ty: index_ty,
                value: LlValue::Local(index_name),
            }],
            root_is_indexed_container: template.root_is_indexed_container,
        }))
    }
}
