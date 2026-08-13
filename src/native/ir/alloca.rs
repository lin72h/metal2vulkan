//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl LlModule {
    pub(in crate::native) fn infer_local_alloca_pointees(&mut self) {
        for f in &self.functions {
            let allocas = self.local_allocas(f);
            if allocas.is_empty() {
                continue;
            }

            let mut roots: HashMap<String, String> = allocas
                .keys()
                .map(|name| (name.clone(), name.clone()))
                .collect();
            let mut sources: HashMap<String, HashSet<LlType>> = HashMap::new();
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
                            .insert(gep.source_ty.clone());
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
                }
            }

            for (name, seen) in sources {
                let Some(original) = allocas.get(&name) else {
                    continue;
                };
                // Opaque-pointer AIR commonly packs subword lanes into scalar scratch by bitcasting
                // the alloca and indexing it as `i8` (for example, two half lanes in one float).
                // Logical SPIR-V cannot form an `uchar*` PtrAccessChain from a scalar pointer. Model
                // the allocation by its byte image instead; the existing raw-byte load/store path
                // then preserves each typed access at its exact byte offset. This is structural and
                // bounded to the allocation's declared size, never keyed to a source identifier.
                if seen
                    .iter()
                    .any(|ty| self.resolve_known_type(ty) == LlType::Int(8))
                    && !self.type_contains_pointer(original)
                {
                    if let Some((size, _)) = self.native_memcpy_type_size_align(original) {
                        if let Ok(size) = u32::try_from(size) {
                            if size > 1 {
                                self.local_alloca_pointees.insert(
                                    (f.name.clone(), name.clone()),
                                    LlType::Array(Box::new(LlType::Int(8)), size),
                                );
                                continue;
                            }
                        }
                    }
                }
                let candidates = seen
                    .into_iter()
                    .filter(|ty| ty != original)
                    .filter(|ty| self.is_local_alloca_reinterpret_candidate(original, ty))
                    .collect::<Vec<_>>();
                if let [candidate] = candidates.as_slice() {
                    self.local_alloca_pointees
                        .insert((f.name.clone(), name), candidate.clone());
                } else if candidates.len() > 1 {
                    // Conflicting same-size views of one alloca (e.g. a union staged through both an
                    // `{ i64 }` and a `{ {i32}, {i32} }` reinterpret). A single struct view can't
                    // receive the others, but a byte array is the universal receiver: every
                    // incompatible typed view lowers through the byte-reinterpret raw GEP + byte-
                    // assembled load/store path. Only force it when the views themselves include an
                    // i8-array (the byte-fill idiom), so exotic multi-struct sets that validate
                    // structurally today keep their current model.
                    let byte_view = candidates.iter().find(|ty| {
                        matches!(self.resolve_known_type(ty),
                                 LlType::Array(elem, _) if *elem == LlType::Int(8))
                    });
                    if let Some(byte_view) = byte_view {
                        self.local_alloca_pointees
                            .insert((f.name.clone(), name), byte_view.clone());
                    }
                }
            }
        }
    }

    pub(in crate::native) fn local_allocas(&self, f: &LlFunction) -> HashMap<String, LlType> {
        let mut allocas = HashMap::new();
        for inst in f.carrier_insts() {
            // `alloca_ty` is `resolve_alloca_ty` — the exact `parse_type(parts[0])` on the post-`alloca`
            // comma-split the reader ran, computed only for `opcode == "alloca"`, so a `Some` here is
            // precisely the reader's parsed alloca type. The result name is the reader's LHS.
            let (Some(name), Some(ty)) = (&inst.result, &inst.alloca_ty) else {
                continue;
            };
            allocas.insert(name.clone(), ty.clone());
        }
        allocas
    }

    pub(in crate::native) fn is_local_alloca_reinterpret_candidate(
        &self,
        original: &LlType,
        candidate: &LlType,
    ) -> bool {
        let original = self.resolve_known_type(original);
        let candidate = self.resolve_known_type(candidate);
        if self.type_contains_pointer(&original) || self.type_contains_pointer(&candidate) {
            return false;
        }
        let Some((original_size, _)) = self.native_memcpy_type_size_align(&original) else {
            return false;
        };
        let Some((candidate_size, _)) = self.native_memcpy_type_size_align(&candidate) else {
            return false;
        };
        original_size == candidate_size
    }

    pub(crate) fn resolve_known_type(&self, ty: &LlType) -> LlType {
        match ty {
            LlType::Int(1) => LlType::Bool,
            LlType::Named(name) => self
                .types
                .get(name)
                .map(|ty| self.resolve_known_type(ty))
                .unwrap_or_else(|| ty.clone()),
            LlType::Vector(elem, 1) => self.resolve_known_type(elem),
            LlType::Vector(elem, lanes) => {
                LlType::Vector(Box::new(self.resolve_known_type(elem)), *lanes)
            }
            LlType::Array(elem, len) => {
                LlType::Array(Box::new(self.resolve_known_type(elem)), *len)
            }
            LlType::Struct(fields) => LlType::Struct(
                fields
                    .iter()
                    .map(|field| self.resolve_known_type(field))
                    .collect(),
            ),
            _ => ty.clone(),
        }
    }
}
