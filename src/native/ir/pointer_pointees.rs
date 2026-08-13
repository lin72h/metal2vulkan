//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl LlModule {
    pub(in crate::native) fn infer_pointer_pointees(&mut self) {
        for f in &self.functions {
            for ((name, ty), pointee) in f.params.iter().zip(&f.byval_param_pointees) {
                if matches!(ty, LlType::Ptr(_)) {
                    if let Some(pointee) = pointee {
                        self.ptr_pointees
                            .insert((f.name.clone(), name.clone()), pointee.clone());
                    }
                }
            }
            let params = f
                .params
                .iter()
                .map(|(param, _)| param.clone())
                .collect::<HashSet<_>>();
            let pointer_params = f
                .params
                .iter()
                .filter(|&(_param, ty)| matches!(ty, LlType::Ptr(_)))
                .map(|(param, _ty)| param.clone())
                .collect::<HashSet<_>>();
            let mut pointer_select_arms: HashMap<String, Vec<String>> = HashMap::new();
            for inst in f.carrier_insts() {
                let Some(result) = &inst.result else {
                    continue;
                };
                let Some((true_value, false_value)) = inst.select_arms.as_deref() else {
                    continue;
                };
                if !matches!(true_value.ty, LlType::Ptr(_))
                    || !matches!(false_value.ty, LlType::Ptr(_))
                {
                    continue;
                }
                let (LlValue::Local(true_name), LlValue::Local(false_name)) =
                    (&true_value.value, &false_value.value)
                else {
                    continue;
                };
                pointer_select_arms
                    .insert(result.clone(), vec![true_name.clone(), false_name.clone()]);
            }

            for inst in f.carrier_insts() {
                if let Some(gep) = &inst.gep {
                    if let LlValue::Local(name) = &gep.base.value {
                        if params.contains(name) {
                            let key = (f.name.clone(), name.clone());
                            if self.gep_source_should_override(&key, &gep.source_ty) {
                                self.metadata_pointee_params.remove(&key);
                                self.ptr_pointees.insert(key, gep.source_ty.clone());
                            } else {
                                self.ptr_pointees
                                    .entry(key)
                                    .or_insert_with(|| gep.source_ty.clone());
                            }
                        }
                        if let Some(arms) = pointer_select_arms.get(name) {
                            for arm in arms {
                                if params.contains(arm) {
                                    let key = (f.name.clone(), arm.clone());
                                    if self.gep_source_should_override(&key, &gep.source_ty) {
                                        self.metadata_pointee_params.remove(&key);
                                        self.ptr_pointees.insert(key, gep.source_ty.clone());
                                    } else {
                                        self.ptr_pointees
                                            .entry(key)
                                            .or_insert_with(|| gep.source_ty.clone());
                                    }
                                }
                            }
                        }
                    }
                    continue;
                }
                if let Some(load) = &inst.load {
                    if let LlValue::Local(name) = &load.ptr.value {
                        if params.contains(name) {
                            self.ptr_pointees
                                .entry((f.name.clone(), name.clone()))
                                .or_insert_with(|| load.result_ty.clone());
                        }
                    }
                }
            }

            let mut sources = self.infer_local_pointer_table_param_pointees(f, &pointer_params);
            let mut roots = pointer_params
                .iter()
                .map(|param| (param.clone(), param.clone()))
                .collect::<HashMap<_, _>>();
            let mut changed = true;
            while changed {
                changed = false;
                for inst in f.carrier_insts() {
                    if let Some((res, base)) = &inst.identity_ptr_bitcast {
                        if let Some(root) = roots.get(base).cloned() {
                            if roots.insert(res.clone(), root).is_none() {
                                changed = true;
                            }
                        }
                        continue;
                    }

                    let Some(res) = &inst.result else {
                        continue;
                    };
                    if let Some(gep) = &inst.gep {
                        let LlValue::Local(base) = &gep.base.value else {
                            continue;
                        };
                        let Some(root) = roots.get(base).cloned() else {
                            continue;
                        };
                        sources
                            .entry(root.clone())
                            .or_default()
                            .insert(self.resolve_known_type(&gep.source_ty));
                        if roots.insert(res.clone(), root).is_none() {
                            changed = true;
                        }
                        continue;
                    }
                    if let Some(incoming) = &inst.phi_incoming_values {
                        let mut root: Option<String> = None;
                        for value in incoming {
                            let LlValue::Local(name) = value else {
                                continue;
                            };
                            let Some(candidate) = roots.get(name).cloned() else {
                                continue;
                            };
                            match &root {
                                Some(existing) if existing != &candidate => {
                                    root = None;
                                    break;
                                }
                                None => root = Some(candidate),
                                _ => {}
                            }
                        }
                        if let Some(root) = root {
                            if roots.insert(res.clone(), root).is_none() {
                                changed = true;
                            }
                        }
                        continue;
                    }
                    if let Some((true_value, false_value)) = inst.select_arms.as_deref() {
                        let (LlValue::Local(true_name), LlValue::Local(false_name)) =
                            (&true_value.value, &false_value.value)
                        else {
                            continue;
                        };
                        let (Some(true_root), Some(false_root)) = (
                            roots.get(true_name).cloned(),
                            roots.get(false_name).cloned(),
                        ) else {
                            continue;
                        };
                        if true_root == false_root && roots.insert(res.clone(), true_root).is_none()
                        {
                            changed = true;
                        }
                    }
                }
            }

            for (param, seen) in sources {
                let mut seen = seen.into_iter().collect::<Vec<_>>();
                seen.sort_by_key(|ty| format!("{ty:?}"));
                seen.dedup();
                let [pointee] = seen.as_slice() else {
                    continue;
                };
                self.ptr_pointees
                    .entry((f.name.clone(), param))
                    .or_insert_with(|| pointee.clone());
            }
        }

        // Propagate pointees through calls: if caller param `%p` is passed to callee param `%q`, and
        // `%q` is typed by a GEP in the callee, `%p` has the same pointee.
        let call_edges = self.param_call_edges();
        let mut changed = true;
        while changed {
            changed = false;
            let mut candidates: HashMap<(String, String), HashSet<LlType>> = HashMap::new();
            for edge in &call_edges {
                let Some(pointee) = self
                    .ptr_pointees
                    .get(&(edge.callee_func.clone(), edge.callee_param.clone()))
                    .cloned()
                else {
                    continue;
                };
                let key = (edge.caller_func.clone(), edge.caller_param.clone());
                candidates.entry(key).or_default().insert(pointee);
            }
            for (key, pointees) in candidates {
                let pointees = pointees.into_iter().collect::<Vec<_>>();
                let [pointee] = pointees.as_slice() else {
                    continue;
                };
                match self.ptr_pointees.get(&key).cloned() {
                    Some(existing)
                        if self.metadata_pointee_params.contains(&key)
                            && self.metadata_pointee_can_yield_to_call(&existing, pointee) =>
                    {
                        self.ptr_pointees.insert(key.clone(), pointee.clone());
                        self.metadata_pointee_params.remove(&key);
                        changed = true;
                    }
                    Some(_) => {}
                    None => {
                        self.ptr_pointees.insert(key, pointee.clone());
                        changed = true;
                    }
                }
            }
        }
    }

    /// Record a buffer param's pointee from a GEP `source_ty`. If the param currently holds a
    /// metadata-seeded pointee (from `air.struct_type_info`) whose member count/ordering can
    /// diverge from the LLVM aggregate the GEP indices actually address — e.g. a union/bitfield
    /// tail described as overlapping members — the concrete same-size LLVM aggregate must win:
    /// GEP struct indices are emitted verbatim and walked by ordinal against this type, so it has
    /// to be member-isomorphic to the SPIR-V element struct. Guarded by the same byte-size
    /// equality as `metadata_pointee_can_yield_to_call`, so buffer stride/extent never changes.
    /// Purely structural (IR shape + size), never keyed on a name.
    pub(in crate::native) fn gep_source_should_override(
        &self,
        key: &(String, String),
        source_ty: &LlType,
    ) -> bool {
        if !self.metadata_pointee_params.contains(key) {
            return false;
        }
        let resolved = self.resolve_known_type(source_ty);
        if !matches!(resolved, LlType::Struct(_) | LlType::Array(_, _))
            || self.type_contains_pointer(&resolved)
        {
            return false;
        }
        // Byte extent must match the buffer's declared element size (offset-aware), NOT the
        // re-derived metadata `LlType::Struct` size, which overcounts when the metadata described
        // a union/bitfield tail as overlapping members. Same bytes => same stride/layout, only the
        // member ordinals get corrected to what the verbatim GEP indices address.
        let Some(&declared) = self.metadata_pointee_sizes.get(key) else {
            return false;
        };
        self.native_memcpy_type_size_align(&resolved)
            .is_some_and(|(size, _)| size == declared)
    }

    pub(in crate::native) fn metadata_pointee_can_yield_to_call(
        &self,
        metadata_pointee: &LlType,
        candidate: &LlType,
    ) -> bool {
        let metadata_pointee = self.resolve_known_type(metadata_pointee);
        let candidate = self.resolve_known_type(candidate);
        if self.type_contains_pointer(&metadata_pointee) || self.type_contains_pointer(&candidate) {
            return false;
        }
        if !matches!(metadata_pointee, LlType::Array(_, _) | LlType::Struct(_))
            || !matches!(candidate, LlType::Array(_, _) | LlType::Struct(_))
        {
            return false;
        }
        let Some((metadata_size, _)) = self.native_memcpy_type_size_align(&metadata_pointee) else {
            return false;
        };
        let Some((candidate_size, _)) = self.native_memcpy_type_size_align(&candidate) else {
            return false;
        };
        metadata_size == candidate_size
    }

    pub(in crate::native) fn infer_local_pointer_table_param_pointees(
        &self,
        f: &LlFunction,
        pointer_params: &HashSet<String>,
    ) -> HashMap<String, HashSet<LlType>> {
        let mut table_roots: HashMap<String, String> = HashMap::new();
        let mut table_params: HashMap<String, HashSet<String>> = HashMap::new();
        let mut loaded_tables: HashMap<String, String> = HashMap::new();
        let mut sources: HashMap<String, HashSet<LlType>> = HashMap::new();

        for inst in f.carrier_insts() {
            if let Some(res) = &inst.result {
                if let Some(ty) = &inst.alloca_ty {
                    if self.type_contains_pointer(ty) {
                        table_roots.insert(res.clone(), res.clone());
                    }
                    continue;
                }
                if let Some((bres, base)) = &inst.identity_ptr_bitcast {
                    if let Some(root) = table_roots.get(base).cloned() {
                        table_roots.insert(bres.clone(), root);
                    }
                    continue;
                }
                if let Some(gep) = &inst.gep {
                    let LlValue::Local(base) = &gep.base.value else {
                        continue;
                    };
                    if let Some(root) = table_roots.get(base).cloned() {
                        table_roots.insert(res.clone(), root);
                    } else if let Some(root) = loaded_tables.get(base) {
                        if let Some(params) = table_params.get(root) {
                            for param in params {
                                sources
                                    .entry(param.clone())
                                    .or_default()
                                    .insert(self.resolve_known_type(&gep.source_ty));
                            }
                        }
                    }
                    continue;
                }
                if let Some(load) = &inst.load {
                    if !matches!(load.result_ty, LlType::Ptr(_)) {
                        continue;
                    }
                    let LlValue::Local(ptr_name) = &load.ptr.value else {
                        continue;
                    };
                    if let Some(root) = table_roots.get(ptr_name).cloned() {
                        loaded_tables.insert(res.clone(), root);
                    }
                }
                continue;
            }

            let Some((object, ptr)) = inst.store.as_deref() else {
                continue;
            };
            let LlValue::Local(param) = &object.value else {
                continue;
            };
            if !pointer_params.contains(param) {
                continue;
            }
            let LlValue::Local(ptr_name) = &ptr.value else {
                continue;
            };
            if let Some(root) = table_roots.get(ptr_name) {
                table_params
                    .entry(root.clone())
                    .or_default()
                    .insert(param.clone());
            }
        }

        sources
    }
}
