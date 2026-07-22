//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl LlModule {
    pub(in crate::native) fn infer_raw_buffer_params(&mut self) {
        for f in &self.functions {
            let pointer_params: HashSet<String> = f
                .params
                .iter()
                .filter_map(|(name, ty)| match ty {
                    LlType::Ptr(1..=3) => Some(name.clone()),
                    _ => None,
                })
                .collect();
            if pointer_params.is_empty() {
                continue;
            }

            let mut roots: HashMap<String, String> = pointer_params
                .iter()
                .map(|name| (name.clone(), name.clone()))
                .collect();
            let mut direct_root_aliases = pointer_params.clone();
            let mut sources: HashMap<String, HashSet<LlType>> = HashMap::new();
            let mut byte_gep_aliases: HashMap<String, String> = HashMap::new();
            let mut selected_roots: HashMap<String, HashSet<String>> = HashMap::new();
            let mut byte_gep_selected_aliases: HashMap<String, HashSet<String>> = HashMap::new();
            let mut byte_reinterpret_roots: HashSet<String> = HashSet::new();
            let mut memcpy_source_roots: HashMap<String, HashSet<u64>> = HashMap::new();
            let mut memcpy_struct_copy_pairs: Vec<(String, String)> = Vec::new();
            let mut vector_load_roots: HashMap<String, HashSet<LlType>> = HashMap::new();
            let mut acceleration_structure_shadow_roots: HashSet<String> = HashSet::new();
            let mut changed = true;
            while changed {
                changed = false;
                for inst in f.carrier_insts() {
                    if let Some((res, base)) = &inst.identity_ptr_bitcast {
                        if let Some(root) = byte_gep_aliases.get(base).cloned() {
                            if byte_gep_aliases.insert(res.clone(), root).is_none() {
                                changed = true;
                            }
                        }
                        if let Some(roots) = byte_gep_selected_aliases.get(base).cloned() {
                            if byte_gep_selected_aliases.get(res) != Some(&roots) {
                                byte_gep_selected_aliases.insert(res.clone(), roots);
                                changed = true;
                            }
                        }
                        if let Some(root) = roots.get(base).cloned() {
                            let direct = direct_root_aliases.contains(base);
                            if direct && direct_root_aliases.insert(res.clone()) {
                                changed = true;
                            }
                            if roots.insert(res.clone(), root).is_none() {
                                changed = true;
                            }
                        }
                        if let Some(arm_roots) = selected_roots.get(base).cloned() {
                            if selected_roots.get(res) != Some(&arm_roots) {
                                selected_roots.insert(res.clone(), arm_roots);
                                changed = true;
                            }
                        }
                        continue;
                    }

                    if let Some(call) = &inst.alias_call {
                        if matches!(
                            call.callee.as_str(),
                            "air.get_instance_count_instance_acceleration_structure"
                                | "air.get_primitive_acceleration_structure_instance_acceleration_structure"
                        ) {
                            if let Some(LlValue::Local(arg_name)) =
                                call.args.first().map(|arg| &arg.value)
                            {
                                if let Some(root) = roots.get(arg_name) {
                                    acceleration_structure_shadow_roots.insert(root.clone());
                                }
                            }
                        }
                        if call.callee.starts_with("llvm.memcpy.")
                            && call.args.len() == 4
                            && matches!(call.args[3].value, LlValue::Bool(false))
                        {
                            let Some(len) = typed_value_u64(&call.args[2]) else {
                                continue;
                            };
                            if len == 0 {
                                continue;
                            }
                            let LlValue::Local(src) = &call.args[1].value else {
                                continue;
                            };
                            let LlValue::Local(dst) = &call.args[0].value else {
                                continue;
                            };
                            let Some(src_root) = roots.get(src).cloned() else {
                                continue;
                            };
                            let Some(dst_root) = roots.get(dst) else {
                                continue;
                            };
                            if self.root_has_struct_pointee(&f.name, dst_root) {
                                memcpy_struct_copy_pairs.push((dst_root.clone(), src_root.clone()));
                                memcpy_source_roots.entry(src_root).or_default().insert(len);
                            }
                        }
                    }

                    if let Some((object, ptr)) = &inst.store {
                        if let LlValue::Local(base) = &ptr.value {
                            if self.resolve_known_type(&object.ty) != LlType::Int(8) {
                                if let Some(root) = byte_gep_aliases.get(base).cloned() {
                                    byte_reinterpret_roots.insert(root);
                                }
                                if let Some(roots) = byte_gep_selected_aliases.get(base).cloned() {
                                    byte_reinterpret_roots.extend(roots);
                                }
                            }
                        }
                    }

                    let Some(res) = &inst.result else {
                        continue;
                    };
                    if let Some(load) = &inst.load {
                        let LlValue::Local(base) = &load.ptr.value else {
                            continue;
                        };
                        if let Some(root) = byte_gep_aliases.get(base).cloned() {
                            if self.resolve_known_type(&load.result_ty) != LlType::Int(8) {
                                byte_reinterpret_roots.insert(root);
                            }
                        }
                        if let Some(roots) = byte_gep_selected_aliases.get(base).cloned() {
                            if self.resolve_known_type(&load.result_ty) != LlType::Int(8) {
                                byte_reinterpret_roots.extend(roots);
                            }
                        }
                        if let Some(root) = roots.get(base).cloned() {
                            if let LlType::Vector(elem, lanes) =
                                self.resolve_known_type(&load.result_ty)
                            {
                                if lanes > 1 {
                                    vector_load_roots.entry(root).or_default().insert(*elem);
                                }
                            }
                        }
                        if !direct_root_aliases.contains(base) {
                            continue;
                        }
                        let Some(root) = roots.get(base).cloned() else {
                            continue;
                        };
                        sources
                            .entry(root)
                            .or_default()
                            .insert(load.result_ty.clone());
                        continue;
                    }
                    if let Some(gep) = &inst.gep {
                        let LlValue::Local(base) = &gep.base.value else {
                            continue;
                        };
                        if let Some(arm_roots) = selected_roots.get(base).cloned() {
                            if self.resolve_known_type(&gep.source_ty) == LlType::Int(8) {
                                for root in &arm_roots {
                                    sources
                                        .entry(root.clone())
                                        .or_default()
                                        .insert(gep.source_ty.clone());
                                }
                                if byte_gep_selected_aliases.get(res) != Some(&arm_roots) {
                                    byte_gep_selected_aliases.insert(res.clone(), arm_roots);
                                    changed = true;
                                }
                            }
                            continue;
                        }
                        let Some(root) = roots.get(base).cloned() else {
                            continue;
                        };
                        sources
                            .entry(root.clone())
                            .or_default()
                            .insert(gep.source_ty.clone());
                        if self.resolve_known_type(&gep.source_ty) == LlType::Int(8)
                            && byte_gep_aliases.insert(res.clone(), root.clone()).is_none()
                        {
                            changed = true;
                        }
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
                            let res_name = res.clone();
                            if roots.insert(res_name.clone(), root.clone()).is_none() {
                                changed = true;
                            }
                            let all_byte_gep_aliases = incoming.iter().all(|value| match value {
                                LlValue::Local(name) => byte_gep_aliases
                                    .get(name)
                                    .is_some_and(|candidate| candidate == &root),
                                _ => false,
                            });
                            if all_byte_gep_aliases
                                && byte_gep_aliases.get(&res_name) != Some(&root)
                            {
                                byte_gep_aliases.insert(res_name, root);
                                changed = true;
                            }
                        }
                        continue;
                    }
                    if let Some((true_value, false_value)) = &inst.select_arms {
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
                        let mut arm_roots = HashSet::new();
                        if let Some(root) = roots.get(true_name) {
                            arm_roots.insert(root.clone());
                        }
                        if let Some(root) = roots.get(false_name) {
                            arm_roots.insert(root.clone());
                        }
                        if let Some(roots) = selected_roots.get(true_name) {
                            arm_roots.extend(roots.iter().cloned());
                        }
                        if let Some(roots) = selected_roots.get(false_name) {
                            arm_roots.extend(roots.iter().cloned());
                        }
                        if !arm_roots.is_empty() && selected_roots.get(res) != Some(&arm_roots) {
                            selected_roots.insert(res.clone(), arm_roots);
                            changed = true;
                        }
                    }
                }
            }

            for param in byte_reinterpret_roots {
                self.raw_buffer_params.insert((f.name.clone(), param));
            }
            for param in acceleration_structure_shadow_roots {
                self.raw_buffer_params.insert((f.name.clone(), param));
            }
            let mut memcpy_raw_sources = HashSet::new();
            for (param, lengths) in &memcpy_source_roots {
                if self.memcpy_source_implies_raw_bytes(sources.get(param), lengths) {
                    memcpy_raw_sources.insert(param.clone());
                    self.raw_buffer_params
                        .insert((f.name.clone(), param.clone()));
                }
            }
            for (dst, src) in memcpy_struct_copy_pairs {
                if memcpy_raw_sources.contains(&src)
                    || self.raw_buffer_params.contains(&(f.name.clone(), src))
                {
                    self.raw_buffer_params.insert((f.name.clone(), dst));
                }
            }
            for (param, elems) in &vector_load_roots {
                if sources.get(param).is_some_and(|seen| {
                    elems.iter().any(|elem| {
                        seen.iter()
                            .any(|source| self.buffer_source_matches_vector_load_elem(source, elem))
                    })
                }) {
                    self.raw_buffer_params
                        .insert((f.name.clone(), param.clone()));
                }
            }
            for (param, seen) in sources {
                let metadata_byte_buffer = self
                    .metadata_byte_buffer_params
                    .contains(&(f.name.clone(), param.clone()));
                if metadata_byte_buffer
                    && seen
                        .iter()
                        .any(|ty| self.is_struct_or_array_buffer_source(ty))
                {
                    self.raw_buffer_params.insert((f.name.clone(), param));
                    continue;
                }
                let non_wrapper_sources = seen
                    .iter()
                    .filter(|ty| !self.is_zero_wrapper_type(ty))
                    .count();
                if non_wrapper_sources > 1 || seen.iter().any(|ty| self.type_contains_pointer(ty)) {
                    self.raw_buffer_params.insert((f.name.clone(), param));
                }
            }
        }
    }

    pub(in crate::native) fn memcpy_source_implies_raw_bytes(
        &self,
        seen: Option<&HashSet<LlType>>,
        lengths: &HashSet<u64>,
    ) -> bool {
        let Some(seen) = seen else {
            return true;
        };
        let mut largest_scalar = 0;
        for ty in seen.iter().filter(|ty| !self.is_zero_wrapper_type(ty)) {
            match self.resolve_known_type(ty) {
                LlType::Struct(_) | LlType::Array(_, _) => {
                    let Some((size, _align)) = self.type_storage_size_align(ty) else {
                        return false;
                    };
                    if lengths.iter().any(|len| *len >= size) {
                        return true;
                    }
                }
                _ => {
                    let Some(size) = self.scalar_storage_size(ty) else {
                        return false;
                    };
                    largest_scalar = largest_scalar.max(size);
                }
            }
        }
        lengths.iter().any(|len| *len > largest_scalar)
    }

    pub(in crate::native) fn buffer_source_matches_vector_load_elem(
        &self,
        source: &LlType,
        elem: &LlType,
    ) -> bool {
        let elem = self.resolve_known_type(elem);
        match self.resolve_known_type(source) {
            LlType::Vector(source_elem, lanes) if lanes > 1 => {
                self.resolve_known_type(&source_elem) == elem
            }
            source => source == elem,
        }
    }

    pub(in crate::native) fn root_has_struct_pointee(&self, func: &str, root: &str) -> bool {
        self.ptr_pointees
            .get(&(func.to_string(), root.to_string()))
            .is_some_and(|ty| matches!(self.resolve_known_type(ty), LlType::Struct(_)))
    }

    pub(in crate::native) fn is_struct_or_array_buffer_source(&self, ty: &LlType) -> bool {
        !self.is_zero_wrapper_type(ty)
            && matches!(
                self.resolve_known_type(ty),
                LlType::Struct(_) | LlType::Array(_, _)
            )
    }

    // These four size/align methods delegate to the shared layout oracle (`crate::layout`,
    // refactor S4). Each threads `resolve_known_type` (the module's named-type table) as the
    // oracle's `resolve` closure — the only `&self` dependency the calculators ever had.
    pub(in crate::native) fn scalar_storage_size(&self, ty: &LlType) -> Option<u64> {
        crate::layout::scalar_storage_size(ty, &|t| self.resolve_known_type(t))
    }

    pub(in crate::native) fn type_storage_size_align(&self, ty: &LlType) -> Option<(u64, u64)> {
        crate::layout::native_size_align(ty, &|t| self.resolve_known_type(t))
    }

    pub(in crate::native) fn native_memcpy_type_size_align(
        &self,
        ty: &LlType,
    ) -> Option<(u64, u64)> {
        crate::layout::memcpy_size_align(ty, &|t| self.resolve_known_type(t))
    }

    pub(in crate::native) fn air_metadata_type_size_align(
        &self,
        ty: &AirType,
    ) -> Option<(u64, u64)> {
        crate::layout::air_metadata_size_align(ty, &|t| self.resolve_known_type(t))
    }

    pub(in crate::native) fn propagate_raw_buffer_params(&mut self) {
        // Raw byte addressing is a property of the buffer object, not of the particular function
        // parameter currently naming it. Follow the call edge in BOTH directions so a byte-view root
        // passed through a GEP/identity-bitcast reaches a helper, and a helper's byte-view requirement
        // reaches its entry buffer. The latter is required for dynamic byte offsets after inlining:
        // keeping the entry typed leaves an AccessChain rooted on a scalar raw byte. The interface then
        // has one raw descriptor for that reflected Metal binding rather than a typed/raw alias pair.
        // `raw_param_call_edges` admits only same-address-space aliases of one parameter, never a
        // cross-buffer merge.
        let call_edges = self.raw_param_call_edges();
        let mut changed = true;
        while changed {
            changed = false;
            let mut additions = vec![];
            for edge in &call_edges {
                let caller = (edge.caller_func.clone(), edge.caller_param.clone());
                let callee = (edge.callee_func.clone(), edge.callee_param.clone());
                if self.raw_buffer_params.contains(&caller)
                    && !self.raw_buffer_params.contains(&callee)
                {
                    additions.push(callee.clone());
                }
                if self.raw_buffer_params.contains(&callee)
                    && !self.raw_buffer_params.contains(&caller)
                {
                    additions.push(caller);
                }
            }
            additions.sort();
            additions.dedup();
            for key in additions {
                if self.raw_buffer_params.insert(key) {
                    changed = true;
                }
            }
        }
    }

    /// Raw-buffer propagation needs the address-preserving alias of a caller parameter, not merely a
    /// direct parameter operand. A helper often receives `%p + byte_offset` after the caller's
    /// `getelementptr i8` / identity-bitcast chain; treating that argument as unrelated loses the raw
    /// byte model precisely when helper inlining exposes its typed accesses at the raw root.
    pub(in crate::native) fn raw_param_call_edges(&self) -> Vec<ParamCallEdge> {
        let funcs: HashMap<String, &LlFunction> =
            self.functions.iter().map(|f| (f.name.clone(), f)).collect();
        let mut seen = HashSet::new();
        let mut edges = Vec::new();
        for f in &self.functions {
            let aliases = pointer_param_alias_roots(f);
            let param_types: HashMap<&str, &LlType> = f
                .params
                .iter()
                .map(|(name, ty)| (name.as_str(), ty))
                .collect();
            for inst in f.carrier_insts() {
                let Some(call) = &inst.alias_call else {
                    continue;
                };
                let Some(callee) = funcs.get(&call.callee) else {
                    continue;
                };
                for (arg, (callee_param, callee_ty)) in call.args.iter().zip(&callee.params) {
                    let LlValue::Local(arg_name) = &arg.value else {
                        continue;
                    };
                    let Some(caller_param) = aliases.get(arg_name) else {
                        continue;
                    };
                    let Some(caller_ty) = param_types.get(caller_param.as_str()) else {
                        continue;
                    };
                    let (LlType::Ptr(caller_as), LlType::Ptr(arg_as), LlType::Ptr(callee_as)) =
                        (caller_ty, &arg.ty, callee_ty)
                    else {
                        continue;
                    };
                    if caller_as != arg_as || arg_as != callee_as {
                        continue;
                    }
                    let edge = ParamCallEdge {
                        caller_func: f.name.clone(),
                        caller_param: caller_param.clone(),
                        callee_func: callee.name.clone(),
                        callee_param: callee_param.clone(),
                    };
                    if seen.insert(edge.clone()) {
                        edges.push(edge);
                    }
                }
            }
        }
        edges
    }

    pub(in crate::native) fn param_call_edges(&self) -> Vec<ParamCallEdge> {
        let funcs: HashMap<String, &LlFunction> =
            self.functions.iter().map(|f| (f.name.clone(), f)).collect();
        let mut seen = HashSet::new();
        let mut edges = Vec::new();
        for f in &self.functions {
            let caller_params = f
                .params
                .iter()
                .map(|(param, _)| param.as_str())
                .collect::<HashSet<_>>();
            for inst in f.carrier_insts() {
                let Some(call) = &inst.alias_call else {
                    continue;
                };
                let Some(callee) = funcs.get(&call.callee) else {
                    continue;
                };
                for (arg, (callee_param, _)) in call.args.iter().zip(&callee.params) {
                    let LlValue::Local(caller_param) = &arg.value else {
                        continue;
                    };
                    if !caller_params.contains(caller_param.as_str()) {
                        continue;
                    }
                    let edge = ParamCallEdge {
                        caller_func: f.name.clone(),
                        caller_param: caller_param.clone(),
                        callee_func: callee.name.clone(),
                        callee_param: callee_param.clone(),
                    };
                    if seen.insert(edge.clone()) {
                        edges.push(edge);
                    }
                }
            }
        }
        edges
    }

    pub(in crate::native) fn is_zero_wrapper_type(&self, ty: &LlType) -> bool {
        match ty {
            LlType::Named(name) => self
                .types
                .get(name)
                .map(|ty| self.is_zero_wrapper_type(ty))
                .unwrap_or(false),
            LlType::Struct(fields) => {
                matches!(fields.as_slice(), [LlType::Array(_, 0)])
            }
            _ => false,
        }
    }

    pub(in crate::native) fn type_contains_pointer(&self, ty: &LlType) -> bool {
        match ty {
            LlType::Ptr(_) => true,
            LlType::Named(name) => self
                .types
                .get(name)
                .map(|ty| self.type_contains_pointer(ty))
                .unwrap_or(false),
            LlType::Vector(elem, _) | LlType::Array(elem, _) => self.type_contains_pointer(elem),
            LlType::Struct(fields) => fields.iter().any(|field| self.type_contains_pointer(field)),
            _ => false,
        }
    }
}
