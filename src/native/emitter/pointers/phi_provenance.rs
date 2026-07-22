//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    pub(in crate::native::emitter) fn emit_pointer_select_provenance(
        &mut self,
        name: &str,
        result_ty: &LlType,
        true_value: &LlValue,
        false_value: &LlValue,
        cond_id: Word,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Option<GepProvenance>, String> {
        if !matches!(result_ty, LlType::Ptr(_)) {
            return Ok(None);
        }
        let Some(template) = self.first_merge_provenance([true_value, false_value].into_iter())
        else {
            return Ok(None);
        };
        if template.indices.len() != 1 {
            return Ok(None);
        }
        let index_ty = template.indices[0].ty.clone();
        let true_is_null = matches!(true_value, LlValue::Zero);
        let false_is_null = matches!(false_value, LlValue::Zero);
        let true_provenance = if true_is_null {
            GepProvenance {
                root: template.root,
                addrspace: template.addrspace,
                source_ty: template.source_ty.clone(),
                indices: vec![TypedValue {
                    ty: index_ty.clone(),
                    value: LlValue::Int(0),
                }],
                root_is_indexed_container: template.root_is_indexed_container,
            }
        } else {
            let Some(provenance) =
                self.provenance_for_pointer_value(true_value, Some(&template), Some(&index_ty))?
            else {
                return Ok(None);
            };
            provenance
        };
        let false_provenance = if false_is_null {
            GepProvenance {
                root: template.root,
                addrspace: template.addrspace,
                source_ty: template.source_ty.clone(),
                indices: vec![TypedValue {
                    ty: index_ty.clone(),
                    value: LlValue::Int(0),
                }],
                root_is_indexed_container: template.root_is_indexed_container,
            }
        } else {
            let Some(provenance) =
                self.provenance_for_pointer_value(false_value, Some(&template), Some(&index_ty))?
            else {
                return Ok(None);
            };
            provenance
        };
        if !compatible_pointer_provenance(&template, &true_provenance)
            || !compatible_pointer_provenance(&template, &false_provenance)
            || true_provenance.indices.len() != 1
            || false_provenance.indices.len() != 1
        {
            return Ok(None);
        }

        let result_type = self.type_id(&index_ty)?;
        let index_name = pointer_index_name(name);
        let result = self.result_id(&index_name, &index_ty)?;
        let true_index = &true_provenance.indices[0];
        let false_index = &false_provenance.indices[0];
        let true_id = self.value_id_in(&true_index.value, &true_index.ty, instructions)?;
        let false_id = self.value_id_in(&false_index.value, &false_index.ty, instructions)?;
        instructions.push(Self::inst(
            Op::Select,
            Some(result_type),
            Some(result),
            vec![
                Operand::IdRef(cond_id),
                Operand::IdRef(true_id),
                Operand::IdRef(false_id),
            ],
        ));
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

    pub(in crate::native::emitter) fn first_merge_provenance<'a>(
        &self,
        mut values: impl Iterator<Item = &'a LlValue>,
    ) -> Option<GepProvenance> {
        values.find_map(|value| match value {
            LlValue::Local(name) => self.gep_provenance.get(name).cloned(),
            _ => None,
        })
    }

    pub(in crate::native::emitter) fn pointer_phi_template_provenance(
        &mut self,
        name: &str,
        incoming: &[(LlValue, String)],
    ) -> Result<Option<GepProvenance>, String> {
        if let Some(template) = self.first_merge_provenance(incoming.iter().map(|(value, _)| value))
        {
            if template.indices.len() != 1 {
                if let Some(forward_template) =
                    self.pointer_phi_forward_template_provenance(name, incoming)?
                {
                    return Ok(Some(forward_template));
                }
            }
            return Ok(Some(template));
        }
        self.pointer_phi_forward_template_provenance(name, incoming)
    }

    pub(in crate::native::emitter) fn pointer_phi_forward_template_provenance(
        &mut self,
        name: &str,
        incoming: &[(LlValue, String)],
    ) -> Result<Option<GepProvenance>, String> {
        let Some(gep) = incoming.iter().find_map(|(value, _)| {
            let LlValue::Local(incoming_name) = value else {
                return None;
            };
            let gep = self.forward_geps.get(incoming_name)?;
            matches!(&gep.base.value, LlValue::Local(base_name) if base_name == name)
                .then(|| gep.clone())
        }) else {
            return Ok(None);
        };
        let LlValue::Local(base_name) = &gep.base.value else {
            return Ok(None);
        };
        if base_name != name || gep.indices.len() != 1 {
            return Ok(None);
        }
        let LlType::Ptr(addrspace) = self.resolve_type(&gep.base.ty)? else {
            return Ok(None);
        };
        let root_candidates = incoming
            .iter()
            .filter_map(|(value, _)| {
                let is_self_forward_gep = match value {
                    LlValue::Local(incoming_name) => {
                        self.forward_geps.get(incoming_name).is_some_and(|gep| {
                            matches!(&gep.base.value, LlValue::Local(base_name) if base_name == name)
                        })
                    }
                    _ => false,
                };
                (!is_self_forward_gep).then(|| value.clone())
            })
            .collect::<Vec<_>>();
        let root = root_candidates
            .iter()
            .find_map(|value| self.value_id(value, &gep.base.ty).ok());
        let Some(root) = root else {
            return Ok(None);
        };
        Ok(Some(GepProvenance {
            root,
            addrspace,
            source_ty: self.resolve_type(&gep.source_ty)?,
            indices: vec![TypedValue {
                ty: gep.indices[0].ty.clone(),
                value: LlValue::Int(0),
            }],
            root_is_indexed_container: self.is_indexed_container_root(root, None),
        }))
    }

    pub(in crate::native::emitter) fn provenance_for_pointer_value(
        &mut self,
        value: &LlValue,
        template: Option<&GepProvenance>,
        index_ty: Option<&LlType>,
    ) -> Result<Option<GepProvenance>, String> {
        match value {
            LlValue::Local(name) => {
                let had_existing_provenance = self.gep_provenance.contains_key(name);
                if let Some(provenance) = self.gep_provenance.get(name).cloned() {
                    if let Some(template) = template {
                        if compatible_pointer_provenance(template, &provenance) {
                            return Ok(Some(provenance));
                        }
                    } else {
                        return Ok(Some(provenance));
                    }
                }
                let (Some(template), Some(index_ty)) = (template, index_ty) else {
                    if let Some(provenance) = self.gep_provenance.get(name).cloned() {
                        return Ok(Some(provenance));
                    }
                    return Ok(None);
                };
                if self
                    .values
                    .get(name)
                    .is_some_and(|(id, _)| *id == template.root)
                {
                    let provenance = GepProvenance {
                        root: template.root,
                        addrspace: template.addrspace,
                        source_ty: template.source_ty.clone(),
                        indices: vec![TypedValue {
                            ty: index_ty.clone(),
                            value: LlValue::Int(0),
                        }],
                        root_is_indexed_container: template.root_is_indexed_container,
                    };
                    if !had_existing_provenance {
                        self.gep_provenance.insert(name.clone(), provenance.clone());
                    }
                    return Ok(Some(provenance));
                }
                if let Some(provenance) =
                    self.forward_gep_pointer_provenance(name, template, index_ty)?
                {
                    return Ok(Some(provenance));
                }
                // The reserved-index promise below is fulfilled only by a LATER defining GEP of
                // `name` (the `materialize_reserved_pointer_index` call sites). A name that already
                // has a value id is already defined — a param or an emitted pointer — so no future
                // definition will materialize the index, and any consumer of the synthesized
                // provenance would reference a dangling id. Bail instead.
                if self.values.contains_key(name) {
                    return Ok(None);
                }
                let index_name = pointer_index_name(name);
                self.result_id(&index_name, index_ty)?;
                let provenance = GepProvenance {
                    root: template.root,
                    addrspace: template.addrspace,
                    source_ty: template.source_ty.clone(),
                    indices: vec![TypedValue {
                        ty: index_ty.clone(),
                        value: LlValue::Local(index_name),
                    }],
                    root_is_indexed_container: template.root_is_indexed_container,
                };
                self.gep_provenance.insert(name.clone(), provenance.clone());
                Ok(Some(provenance))
            }
            _ => Ok(None),
        }
    }

    pub(in crate::native::emitter) fn forward_gep_pointer_provenance(
        &mut self,
        name: &str,
        template: &GepProvenance,
        index_ty: &LlType,
    ) -> Result<Option<GepProvenance>, String> {
        let Some(gep) = self.forward_geps.get(name).cloned() else {
            return Ok(None);
        };
        let LlValue::Local(base_name) = &gep.base.value else {
            return Ok(None);
        };
        let base_index_name = pointer_index_name(base_name);
        if !self.values.contains_key(&base_index_name) || gep.indices.len() != 1 {
            return Ok(None);
        }
        let LlType::Ptr(addrspace) = self.resolve_type(&gep.base.ty)? else {
            return Ok(None);
        };
        let source_ty = self.resolve_type(&gep.source_ty)?;
        if addrspace != template.addrspace || source_ty != template.source_ty {
            return Ok(None);
        }
        let index_name = pointer_index_name(name);
        self.result_id(&index_name, index_ty)?;
        let provenance = GepProvenance {
            root: template.root,
            addrspace: template.addrspace,
            source_ty: template.source_ty.clone(),
            indices: vec![TypedValue {
                ty: index_ty.clone(),
                value: LlValue::Local(index_name),
            }],
            root_is_indexed_container: template.root_is_indexed_container,
        };
        self.gep_provenance
            .insert(name.to_string(), provenance.clone());
        Ok(Some(provenance))
    }

    pub(in crate::native::emitter) fn materialize_reserved_pointer_index(
        &mut self,
        name: &str,
        index: &TypedValue,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let index_name = pointer_index_name(name);
        if !self.values.contains_key(&index_name) {
            return Ok(());
        }
        let index_ty = self.resolve_type(&index.ty)?;
        let value = self.value_id(&index.value, &index_ty)?;
        let result_type = self.type_id(&index_ty)?;
        let result = self.result_id(&index_name, &index_ty)?;
        if value != result {
            instructions.push(Self::inst(
                Op::CopyObject,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(value)],
            ));
        }
        Ok(())
    }

    pub(in crate::native::emitter) fn record_pointer_meta(
        &mut self,
        name: String,
        meta: PointerMeta,
    ) {
        self.pointer_storage.insert(name.clone(), meta.storage);
        if let Some(pointee) = meta.pointee {
            self.pointer_pointees.insert(name, pointee);
        }
    }

    pub(in crate::native::emitter) fn record_pointer_nullness(
        &mut self,
        name: String,
        is_null: Word,
    ) {
        self.pointer_nullness.insert(name, is_null);
    }

    pub(in crate::native::emitter) fn emit_pointer_nullness_phi(
        &mut self,
        name: &str,
        incoming: &[(LlValue, String)],
        result_ty: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        if !matches!(result_ty, LlType::Ptr(_)) {
            return Ok(());
        }
        let bool_ty = LlType::Bool;
        let result_type = self.type_id(&bool_ty)?;
        let result = self.result_id(&pointer_null_name(name), &bool_ty)?;
        let mut ops = Vec::new();
        let mut seen_incoming: HashMap<Word, Word> = HashMap::new();
        for (value, label) in incoming {
            let value_id = self.pointer_nullness_phi_value_id(value)?;
            let label_id = self.label_id(label)?;
            if let Some(existing) = seen_incoming.insert(label_id, value_id) {
                if existing != value_id {
                    return Err(format!(
                        "native emitter: pointer nullness phi has multiple values from predecessor {label}"
                    ));
                }
                continue;
            }
            ops.push(Operand::IdRef(value_id));
            ops.push(Operand::IdRef(label_id));
        }
        instructions.push(Self::inst(Op::Phi, Some(result_type), Some(result), ops));
        self.pointer_nullness.insert(name.to_string(), result);
        Ok(())
    }

    pub(in crate::native::emitter) fn pointer_nullness_phi_value_id(
        &mut self,
        value: &LlValue,
    ) -> Result<Word, String> {
        match value {
            LlValue::Zero => self.const_bool(true),
            LlValue::Local(name) => {
                if let Some(id) = self.pointer_nullness.get(name).copied() {
                    return Ok(id);
                }
                if !self.values.contains_key(name) && self.pointer_phi_values.contains(name) {
                    let id = self.result_id(&pointer_null_name(name), &LlType::Bool)?;
                    self.pointer_nullness.insert(name.clone(), id);
                    return Ok(id);
                }
                self.const_bool(false)
            }
            _ => self.const_bool(false),
        }
    }

    pub(in crate::native::emitter) fn pointer_nullness_for_compare(
        &self,
        value: &LlValue,
    ) -> Result<Word, String> {
        match value {
            LlValue::Zero => Err("native emitter: degenerate null pointer comparison".to_string()),
            LlValue::Local(name) => self.pointer_nullness.get(name).copied().ok_or_else(|| {
                format!("native emitter: pointer nullness is not tracked for {name}")
            }),
            _ => Err("native emitter: pointer nullness comparison requires SSA value".to_string()),
        }
    }

    pub(in crate::native::emitter) fn label_id(&self, label: &str) -> Result<Word, String> {
        self.block_labels
            .get(label)
            .copied()
            .ok_or_else(|| format!("native emitter: unknown block label {label}"))
    }

    pub(in crate::native::emitter) fn pointer_storage_for(
        &self,
        value: &LlValue,
        addrspace: u32,
    ) -> Result<StorageClass, String> {
        match value {
            LlValue::Local(name) => self
                .pointer_storage
                .get(name)
                .copied()
                .ok_or_else(|| format!("native emitter: missing pointer storage for {name}")),
            LlValue::Global(_) if addrspace == 3 => Ok(StorageClass::Workgroup),
            LlValue::Global(_) => Ok(StorageClass::Private),
            LlValue::Gep(gep) => {
                let LlType::Ptr(base_addrspace) = self.resolve_type(&gep.base.ty)? else {
                    return Err(format!(
                        "native emitter: getelementptr base is not a pointer: {:?}",
                        gep.base.ty
                    ));
                };
                self.pointer_storage_for(&gep.base.value, base_addrspace)
            }
            _ => llvm_pointer_storage(addrspace),
        }
    }

    pub(in crate::native::emitter) fn is_workgroup_global_root(&self, root: Word) -> bool {
        self.global_values
            .values()
            .any(|(id, ty)| *id == root && matches!(ty, LlType::Ptr(3)))
            || self.values.iter().any(|(name, (id, _))| {
                *id == root
                    && self.pointer_storage.get(name) == Some(&StorageClass::Workgroup)
                    && self.pointer_pointees.get(name).is_some_and(|pointee| {
                        matches!(
                            pointee,
                            LlType::Array(_, _) | LlType::Struct(_) | LlType::Vector(_, _)
                        )
                    })
            })
    }

    pub(in crate::native::emitter) fn is_workgroup_indexed_container_root(
        &self,
        root: Word,
    ) -> bool {
        self.is_indexed_container_root(root, Some(StorageClass::Workgroup))
    }

    pub(in crate::native::emitter) fn is_indexed_container_root(
        &self,
        root: Word,
        storage: Option<StorageClass>,
    ) -> bool {
        self.global_values.iter().any(|(name, (id, ty))| {
            let matches_storage = match (storage, ty) {
                (Some(storage), LlType::Ptr(addrspace)) => {
                    llvm_pointer_storage(*addrspace) == Ok(storage)
                }
                (Some(_), _) => false,
                (None, _) => true,
            };
            *id == root
                && matches_storage
                && self.pointer_pointees.get(name).is_some_and(|pointee| {
                    matches!(pointee, LlType::Array(_, _) | LlType::Struct(_))
                })
        }) || self.values.iter().any(|(name, (id, _))| {
            *id == root
                && storage
                    .map(|storage| self.pointer_storage.get(name) == Some(&storage))
                    .unwrap_or(true)
                && self.pointer_pointees.get(name).is_some_and(|pointee| {
                    matches!(pointee, LlType::Array(_, _) | LlType::Struct(_))
                })
        })
    }
}
