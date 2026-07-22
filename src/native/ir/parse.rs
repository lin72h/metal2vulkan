//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl LlModule {
    pub(in crate::native) fn parse(ll: &str) -> Result<Self, String> {
        let kern = meta::parse_air_kernel_meta(ll);
        let entry_name = meta::entry_name(ll, "kernel");
        Self::parse_inner(ll, false, kern.as_ref(), entry_name.as_deref())
    }

    /// Parse using stage metadata already owned by the translate path. Retry tiers re-emit the same
    /// AIR metadata with different buffer/CFG models, so reparsing that metadata inside every tier is
    /// pure waste and risks divergent inference. Direct native-emitter callers retain [`Self::parse`].
    pub(in crate::native) fn parse_with_stage_meta(
        ll: &str,
        kern: Option<&meta::KernMeta>,
        entry_name: Option<&str>,
    ) -> Result<Self, String> {
        Self::parse_inner(ll, false, kern, entry_name)
    }

    /// Parse with the exact primitive-metadata fallback enabled for a validation-gated re-emission.
    /// The normal emitter deliberately leaves this off: metadata alone is insufficient authority to
    /// change a primary module whose raw form has not first proved the relevant pointer-typing gap.
    #[cfg(test)]
    pub(in crate::native) fn parse_with_primitive_phi_metadata(ll: &str) -> Result<Self, String> {
        let kern = meta::parse_air_kernel_meta(ll);
        let entry_name = meta::entry_name(ll, "kernel");
        Self::parse_inner(ll, true, kern.as_ref(), entry_name.as_deref())
    }

    pub(in crate::native) fn parse_with_primitive_phi_metadata_and_stage_meta(
        ll: &str,
        kern: Option<&meta::KernMeta>,
        entry_name: Option<&str>,
    ) -> Result<Self, String> {
        Self::parse_inner(ll, true, kern, entry_name)
    }

    pub(in crate::native) fn parse_inner(
        ll: &str,
        primitive_phi_metadata: bool,
        kern: Option<&meta::KernMeta>,
        entry_name: Option<&str>,
    ) -> Result<Self, String> {
        let mut types = HashMap::new();
        let lines: Vec<&str> = ll.lines().collect();
        let mut functions = Vec::new();
        // Each function's raw body lines, parallel to `functions`, held only until the carriers are
        // lowered below; `LlFunction` itself has no `Vec<String>` body (T5 — text is read once, here).
        let mut function_bodies: Vec<Vec<String>> = Vec::new();
        let mut declarations = Vec::new();
        let mut globals = Vec::new();
        let mut i = 0;
        while i < lines.len() {
            let line = strip_comment(lines[i]).trim();
            if line.starts_with('%') && line.contains(" = type ") {
                let (name, body) = line
                    .split_once(" = type ")
                    .ok_or_else(|| format!("native emitter: malformed type alias: {line}"))?;
                types.insert(name.trim().to_string(), parse_type(body.trim())?);
                i += 1;
                continue;
            }
            if line.starts_with("define ") {
                let (func, body, next) = parse_function(&lines, i)?;
                functions.push(func);
                function_bodies.push(body);
                i = next;
                continue;
            }
            if line.starts_with("declare ") {
                let decl = parse_declaration(line)?;
                if !is_ignored_intrinsic(&decl.name) {
                    declarations.push(decl);
                }
                i += 1;
                continue;
            }
            if is_ignored_global(line) {
                i += 1;
                continue;
            }
            if line.starts_with('@') && (line.contains(" constant ") || line.contains(" global ")) {
                globals.push(parse_global(line)?);
                i += 1;
                continue;
            }
            i += 1;
        }
        if functions.is_empty() {
            return Err("native emitter: no function definitions found".into());
        }
        // Parse-once typed IR: lower every function's blocks to carriers now that the module type
        // table is complete, BEFORE any inference reads a body. This is the SAME `split_body_blocks`
        // call the emitter runs per function (same immutable `body`, same module `types`); the
        // carriers do not depend on any inference output, so hoisting the population to the top of
        // `parse_inner` is byte-neutral. Every body reader — the free-function imageblock scans
        // below, the parse-time pointee/raw-buffer inferences, the CFG diagnostics, and the emitter's
        // global-reinterpret scans — reads `f.blocks` (the F-track / T5): `LlFunction.body` is deleted
        // and the LLVM-IR text is read exactly once (here, at parse).
        let block_lists: Vec<Vec<crate::native::cfg::BodyBlock>> = functions
            .iter()
            .zip(&function_bodies)
            .map(|(f, body)| {
                let entry = crate::native::cfg::implicit_entry_block_name(f);
                crate::native::cfg::split_body_blocks(body, entry, &types)
            })
            .collect();
        for (f, blocks) in functions.iter_mut().zip(block_lists) {
            f.blocks = blocks;
        }
        let entry_functions = entry_name
            .map(|name| HashSet::from([name.to_string()]))
            .unwrap_or_else(|| infer_entry_functions(ll));
        let metadata_byte_buffer_params =
            infer_metadata_byte_buffer_params(kern, entry_name, &functions);
        let metadata_data_buffer_params =
            infer_metadata_data_buffer_params(kern, entry_name, &functions);
        let imageblock_dimensions = infer_apv_imageblock_dimensions(ll);
        let cross_coordinate_imageblock =
            infer_cross_coordinate_imageblock(&functions, &entry_functions);
        let imageblock_threads_per_threadgroup_param = kern
            .and_then(|meta| {
                meta.roles.iter().find_map(|(index, role)| {
                    matches!(role, KernRole::ThreadsPerThreadgroup).then_some(*index as usize)
                })
            })
            .and_then(|index| {
                functions
                    .iter()
                    .find(|function| entry_functions.contains(&function.name))
                    .and_then(|function| function.params.get(index))
                    .map(|(name, _)| name.clone())
            });
        let imageblock_shared_cells =
            cross_coordinate_imageblock && imageblock_threads_per_threadgroup_param.is_some();
        // Private imageblock scratch normally keeps only its first metadata member, because a
        // single-coordinate slice never addresses another field. A byte GEP rooted at
        // `air.imageblock_data` proves that this module does address a later field, though; retain
        // the complete cell so the emitter can form that field through a typed access chain rather
        // than an illegal Logical-pointer reinterpret.
        let imageblock_nonzero_byte_field =
            infer_imageblock_nonzero_byte_field(&functions, &entry_functions);
        let imageblock_data_pointee = infer_imageblock_data_pointee(
            kern,
            imageblock_dimensions.is_some()
                || imageblock_shared_cells
                || imageblock_nonzero_byte_field,
        );
        let mut module = Self {
            types,
            functions,
            declarations,
            globals,
            entry_name: entry_name.map(str::to_string),
            preinlined_static_initializers: HashSet::new(),
            preinlined_helper_pointer_loads: HashSet::new(),
            preinlined_helper_type_capabilities: HashSet::new(),
            entry_functions,
            ptr_pointees: HashMap::new(),
            local_alloca_pointees: HashMap::new(),
            imageblock_data_pointee,
            imageblock_dimensions,
            imageblock_shared_cells,
            imageblock_threads_per_threadgroup_param,
            metadata_pointee_params: HashSet::new(),
            metadata_pointee_sizes: HashMap::new(),
            metadata_byte_buffer_params,
            metadata_data_buffer_params,
            raw_buffer_params: HashSet::new(),
        };
        module.infer_metadata_buffer_pointees(kern, entry_name);
        module.infer_pointer_pointees();
        if primitive_phi_metadata {
            module.infer_metadata_primitive_buffer_pointees(kern, entry_name);
        }
        module.infer_local_alloca_pointees();
        module.infer_raw_buffer_params();
        module.propagate_raw_buffer_params();
        Ok(module)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threaded_stage_meta_preserves_kernel_inference() {
        let ll = r#"
define void @k(ptr addrspace(1) %input) {
entry:
  %value = load float, ptr addrspace(1) %input, align 4
  ret void
}

!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"input"}
"#;
        let reparsed = LlModule::parse(ll).expect("ordinary parse");
        let kern = meta::parse_air_kernel_meta(ll);
        let threaded =
            LlModule::parse_with_stage_meta(ll, kern.as_ref(), Some("k")).expect("threaded parse");

        assert_eq!(threaded.entry_functions, reparsed.entry_functions);
        assert_eq!(threaded.ptr_pointees, reparsed.ptr_pointees);
        assert_eq!(
            threaded.metadata_pointee_params,
            reparsed.metadata_pointee_params
        );
        assert_eq!(
            threaded.metadata_pointee_sizes,
            reparsed.metadata_pointee_sizes
        );
        assert_eq!(
            threaded.metadata_byte_buffer_params,
            reparsed.metadata_byte_buffer_params
        );
        assert_eq!(
            threaded.metadata_data_buffer_params,
            reparsed.metadata_data_buffer_params
        );
        assert_eq!(
            threaded.imageblock_data_pointee,
            reparsed.imageblock_data_pointee
        );
        assert_eq!(
            threaded.imageblock_threads_per_threadgroup_param,
            reparsed.imageblock_threads_per_threadgroup_param
        );
    }
}
