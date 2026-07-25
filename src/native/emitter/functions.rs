use super::*;
use crate::native::cfg::BodyBlock;
use std::collections::{HashMap, HashSet};

impl Emitter {
    pub(super) fn emit_function(&mut self, f: &LlFunction) -> Result<(), String> {
        self.values.clear();
        // Graph-driven emission: the typed graph is built below from the STRUCTURIZED block list
        // (`body_blocks`), once it is finalized — see the `tir::build_from_blocks` call after
        // structurization. Building it there (not from the parse-time blocks here) sources operands from exactly the
        // IR emission walks, the prerequisite the phi/store migration needed (the structurizer rewrites
        // those operands between parse and emit). It is the SOLE emission substrate — there is no string
        // fall-back path; a build failure (a block with no terminator) is a fail-visible error that
        // routes the function to the retry cascade (measured dead broadly).
        self.tir_operands.clear();
        self.tir_result_types.clear();
        self.tir_predicates.clear();
        self.tir_aligns.clear();
        self.tir_gep_source_types.clear();
        self.tir_use_pointees.clear();
        self.tir_byte_view_pointers.clear();
        self.network_pointees.clear();
        self.gep_provenance.clear();
        self.selected_pointers.clear();
        self.selected_load_pointers.clear();
        self.vector_word_roots.clear();
        self.vector_word_pointers.clear();
        self.local_pointer_fields.clear();
        self.raw_memcpy_shadows.clear();
        self.dynamic_pointer_tables.clear();
        self.forward_geps.clear();
        self.pointer_storage.clear();
        self.pointer_pointees.clear();
        self.local_alloca_pointees = self
            .ir
            .local_alloca_pointees
            .iter()
            .filter(|&((func, _name), _pointee)| func == &f.name)
            .map(|((_func, name), pointee)| (name.clone(), pointee.clone()))
            .collect();
        self.pointer_nullness.clear();
        self.pointer_payload_words.clear();
        self.pointer_payload_values.clear();
        self.pointer_phi_values.clear();
        self.pointer_phi_incoming_values.clear();
        self.tir_phi_incomings.clear();
        self.direct_param_values.clear();
        self.direct_param_indices.clear();
        self.param_values.clear();
        self.inline_parameter_substitutions.clear();
        self.raw_offsets.clear();
        self.int_alignments.clear();
        self.unmodeled_pointers.clear();
        for global in &self.ir.globals.clone() {
            let pointee = self.global_declared_pointee(global)?;
            self.pointer_pointees.insert(global.name.clone(), pointee);
            self.pointer_storage.insert(
                global.name.clone(),
                if global.addrspace == 3 {
                    StorageClass::Workgroup
                } else {
                    StorageClass::Private
                },
            );
        }
        self.raw_buffer_params = self
            .ir
            .raw_buffer_params
            .iter()
            .filter(|(function, _)| function == &f.name)
            .map(|(_, name)| name.clone())
            .collect();
        self.data_buffer_params = self
            .ir
            .metadata_data_buffer_params
            .iter()
            .filter(|(function, _)| function == &f.name)
            .map(|(_, name)| name.clone())
            .collect();
        let ret_ty = self.resolve_type(&f.ret)?;
        let ret_id = self.type_id(&ret_ty)?;
        let param_types: Vec<Word> = f
            .params
            .iter()
            .enumerate()
            .map(|(index, (name, ty))| self.param_type_id(&f.name, index, name, ty))
            .collect::<Result<Vec<_>, _>>()?;
        let fn_ty = self.function_type_id(ret_id, &param_types);

        let func_id = *self
            .function_ids
            .get(&f.name)
            .ok_or_else(|| format!("native emitter: missing function id for {}", f.name))?;
        self.module.debug_names.push(Self::inst(
            Op::Name,
            None,
            None,
            vec![
                Operand::IdRef(func_id),
                Operand::LiteralString(f.name.clone()),
            ],
        ));
        let mut params = Vec::with_capacity(f.params.len());
        for (param_index, ((name, ty), type_id)) in
            f.params.iter().zip(param_types.iter()).enumerate()
        {
            let id = self.fresh();
            params.push(Self::inst(
                Op::FunctionParameter,
                Some(*type_id),
                Some(id),
                vec![],
            ));
            let resolved_ty = self.resolve_type(ty)?;
            self.values.insert(name.clone(), (id, resolved_ty.clone()));
            if let LlType::Ptr(addrspace) = resolved_ty {
                let storage = if addrspace == 4
                    && (self.ir.imageblock_dimensions.is_some() || self.ir.imageblock_shared_cells)
                {
                    StorageClass::Workgroup
                } else {
                    llvm_pointer_storage(addrspace)?
                };
                self.pointer_storage.insert(name.clone(), storage);
                if self.ir.entry_functions.contains(&f.name)
                    || self
                        .function_param_nonnull
                        .contains(&(f.name.clone(), param_index))
                {
                    let is_null = self.const_bool(false)?;
                    self.record_pointer_nullness(name.clone(), is_null);
                }
                let concrete_workgroup_raw_param = addrspace == 3
                    && self
                        .concrete_vector_workgroup_raw_param_pointee(&f.name, param_index, name)
                        .is_some();
                if self.raw_buffer_params.contains(name) {
                    let pointee = if addrspace == 3 {
                        self.concrete_vector_workgroup_raw_param_pointee(&f.name, param_index, name)
                            .unwrap_or_else(raw_workgroup_array_type)
                    } else {
                        raw_buffer_block_type()
                    };
                    self.pointer_pointees.insert(name.clone(), pointee);
                } else if let Some(pointee) = self
                    .function_param_pointees
                    .get(&(f.name.clone(), param_index))
                    .cloned()
                {
                    self.pointer_pointees.insert(name.clone(), pointee);
                } else if let Some(pointee) = self
                    .ir
                    .ptr_pointees
                    .get(&(f.name.clone(), name.clone()))
                    .cloned()
                {
                    self.pointer_pointees.insert(name.clone(), pointee);
                }
                if self.raw_buffer_params.contains(name) && !concrete_workgroup_raw_param {
                    self.raw_offsets
                        .insert(name.clone(), RawBufferOffset::root(name.clone(), addrspace));
                }
            }
            self.direct_param_values.insert(name.clone());
            if self.ir.entry_functions.contains(&f.name) {
                self.direct_param_indices
                    .insert(name.clone(), param_index as u32);
            }
            self.param_values.insert(name.clone());
        }

        // T5/T8 keystone: seed every block's typed carrier from its lines at SPLIT time — BEFORE
        // `lower_unstructured_switches` — so no window exists in which a structurizer reader sees an
        // unpopulated (`None`) carrier. Switch lowering then preserves carriers on pass-through blocks
        // and constructs them on the ladder blocks it synthesizes, so every block is carriered from
        // birth — the invariant that let `BodyBlock.lines` retire (readers never need a `.lines`
        // fallback). BC drift NONE proves the carriers stay byte-identical to a fresh re-lower.
        // The function's blocks were lowered to carriers once at parse time (`f.blocks`); reuse them
        // instead of re-splitting the body text. Byte-identical (same `split_body_blocks` call), and the
        // clone keeps the parse-time carriers pristine while this emit mutates its copy.
        let split = f.blocks.clone();
        let mut body_blocks = lower_unstructured_switches(&split);
        let reorder_defuse = ReorderDefUse::from_blocks(&body_blocks)?;
        reorder_forward_local_def_blocks(&mut body_blocks, &reorder_defuse)?;
        // R2 module 4: structured-by-construction emission is the DEFAULT (module 4). For a fully-
        // structurable function, take the forest-derived plan (reordered blocks + per-construct unique
        // merges) and skip the post-hoc pre-phi fixup; functions `structured_plan` rejects emit their
        // inferred merges unrepaired and fall to the retry cascade's relooper tiers (the W4 roster
        // deletion — see the reject-path comment below). The structured path is a strict improvement on
        // the unseeded frontier (cfg 479→110, total 929→574, zero over-admission across all 16 shards).
        // A relooper-feed emission deliberately preserves the original CFG for the retry's
        // switch-dispatch structurizer.  Running the full structured-plan ladder first cannot
        // improve that feed, and on a large rejected graph it repeatedly recomputes dominators and
        // selection merges before the feed gets a chance to run.  Keep the normal diagnostic and
        // planning paths intact for every production/default emission; only this internal retry
        // intermediate bypasses them.
        let relooper_feed = self.relooper_feed;
        if crate::env_vars::why() && !relooper_feed {
            match crate::native::cfg::structured_reject_reason(&body_blocks) {
                None => eprintln!("WHY ADMIT"),
                Some(r) => eprintln!("WHY REJECT {r}"),
            }
        }
        let mut structured_active = false;
        let mut construct_tree_plan = None;
        if !relooper_feed {
            // R2 cross-arm restructure — produces a CANDIDATE, adopted at the translate level ONLY if
            // the whole module then passes spirv-val (`self.cfg_restructure`, set by the
            // `inline_sroa_raw_cfg_restructure` emit variant that retry tier drives). When
            // `structured_plan` rejects, restructure with two coordinated transforms — privatize
            // cross-arm shared regions via full-closure tail duplication, then unify all returns into a
            // single exit so divergent selections gain a natural merge — and take the result if it then
            // ADMITS a structured plan. Admission is NOT sufficient: every currently-admitting case still
            // emits INVALID structured SPIR-V ("branches to the selection construct, but not to the
            // header" — a nesting violation the admission check does not catch on the transformed graph),
            // which is exactly why adoption is gated on spirv-val at the caller, not on admission here.
            // The DEFAULT emit leaves `cfg_restructure` false, so a reject emits its inferred merges
            // unrepaired (post-W4) and an admitting case can never be hijacked.
            if (self.cfg_restructure || self.construct_tree)
                && body_blocks.len() <= crate::native::cfg::CROSS_ARM_EDGE_MAX_BLOCKS
                && crate::native::cfg::structured_plan(&body_blocks).is_none()
            {
                let mut candidate = body_blocks.clone();
                let mut construct_tree_applied = false;
                if self.construct_tree {
                    match crate::native::cfg::renest_cond_phi_shared_own_arm(&candidate) {
                        Ok(Some(renested)) => {
                            candidate = renested;
                            construct_tree_applied = true;
                            construct_tree_plan =
                                crate::native::cfg::structured_plan_construct_tree(&candidate);
                            if crate::env_vars::why() && construct_tree_plan.is_none() {
                                eprintln!("WHY-CONSTRUCT-TREE own-arm plan-decline");
                            }
                        }
                        Ok(None) => {}
                        Err(error) => {
                            if crate::env_vars::why() {
                                eprintln!("WHY-CONSTRUCT-TREE own-arm decline {error}");
                            }
                        }
                    }
                    if !construct_tree_applied {
                        match crate::native::cfg::renest_straddle_loop_merge(&candidate) {
                            Ok(Some(renested)) => {
                                candidate = renested;
                                construct_tree_applied = true;
                                construct_tree_plan =
                                    crate::native::cfg::structured_plan_construct_tree(&candidate);
                                if crate::env_vars::why() && construct_tree_plan.is_none() {
                                    eprintln!("WHY-CONSTRUCT-TREE straddle plan-decline");
                                }
                            }
                            Ok(None) => {}
                            Err(error) => {
                                if crate::env_vars::why() {
                                    eprintln!("WHY-CONSTRUCT-TREE straddle decline {error}");
                                }
                            }
                        }
                    }
                }
                // Straddle-loop-merge restructure first: give a loop whose exit merge straddles an
                // enclosing early-return guard its OWN pass-through merge, so the construct boundaries
                // no longer invert (05/MPSRNNBreakUpToOutputVecs). Runs before the cross-arm passes so
                // they operate on the de-straddled CFG. Adopt-if-validates at the caller (floor-safe).
                if self.cfg_restructure && !construct_tree_applied {
                    if let Some(destraddled) =
                        crate::native::cfg::restructure_straddle_loop_merges(&candidate)
                    {
                        candidate = destraddled;
                    }
                }
                // Merge-PRESERVING dominated-region privatization first: clones each cross-arm's
                // dominated subtree up to its reconvergence boundary (keeping the shared merge), the
                // correct tool for the dominant cross-arm-shared shape — where full-closure tail
                // duplication (`clone_cross_arm_shared`, applied next for the shapes this leaves) sweeps
                // in and destroys the merge. Both are candidates gated on spirv-val at the lib.rs caller
                // (adopt-if-VALIDATES), so an admitting-but-invalid restructure is discarded, not adopted.
                if self.cfg_restructure && !construct_tree_applied {
                    let region_privatized =
                        crate::native::cfg::privatize_region_cross_arm(&candidate);
                    if region_privatized.len() != candidate.len() {
                        candidate = region_privatized;
                    }
                    if let Some(cloned) = crate::native::cfg::clone_cross_arm_shared(&candidate) {
                        candidate = cloned;
                    }
                    if let Some(lowered) = crate::native::cfg::lower_unreachable_to_ret(&candidate)
                    {
                        candidate = lowered;
                    }
                    if let Some(unified) = crate::native::cfg::unify_returns(&candidate) {
                        candidate = unified;
                    }
                }
                if crate::env_vars::why() {
                    match crate::native::cfg::structured_reject_reason(&candidate) {
                        None => eprintln!("WHY-CANDIDATE ADMIT"),
                        Some(r) => eprintln!("WHY-CANDIDATE REJECT {r}"),
                    }
                }
                if construct_tree_plan.is_some()
                    || crate::native::cfg::structured_plan(&candidate).is_some()
                {
                    body_blocks = candidate;
                }
            }
            if let Some(plan) =
                construct_tree_plan.or_else(|| crate::native::cfg::structured_plan(&body_blocks))
            {
                body_blocks = plan.blocks;
                self.block_labels.clear();
                self.branch_merges = plan.branch_merges;
                self.branch_merges_by_header = plan.branch_merges_by_header;
                self.loop_merges = plan.loop_merges;
                self.switch_merges = plan.switch_merges;
                structured_active = true;
            }
        }
        if !structured_active {
            self.block_labels.clear();
            self.branch_merges_by_header.clear();
            if relooper_feed {
                self.branch_merges.clear();
                self.loop_merges.clear();
                self.switch_merges = infer_switch_merges(&body_blocks);
            } else {
                self.branch_merges = infer_branch_merges(&body_blocks);
                self.loop_merges = infer_loop_merges(&body_blocks);
                self.switch_merges = infer_switch_merges(&body_blocks);
            }
        }
        // R3: index resolved typed operands by result name, built from the now-finalized structurized
        // block list — the exact IR the emission loop below walks. Straight-line instruction text is
        // never rewritten by structurization, so migrated consumers (binary/unary/convert/compare/
        // select/load/extract+insert/shuffle/gep) read byte-identical operands; the added synthetic
        // `%metal2vulkan.lmerge.*` phi entries are inert until phi emission consumes the graph. `store` and
        // the void-`call` (both result-LESS) drive straight off `inst.operands`/`inst.mem_align`/`inst.call`
        // in the graph walk — the former per-block store/call text-keyed queues are retired.
        // The typed graph is built from the finalized structurized blocks and is the SOLE emission
        // substrate. A build failure (a block with no terminator) is a fail-visible error that routes
        // the function to the retry cascade (which rebuilds the CFG via the relooper) — NOT a text-walk
        // fallback. Measured dead broadly (0 build failures / 16942 frontier + 0 / 15,336 banked).
        let tir = crate::native::tir::build_from_blocks(&body_blocks)?;
        {
            let tir = &tir;
            // M1 (pointer-typing rewrite): carry the USE-based pointee of every pointer SSA value onto
            // the value, sourced from the SAME structurized graph the operand map below is built from.
            // This is the whole-function `use_pointees` map (keyed by value name, propagated across
            // select/phi/freeze pointer merges to a fixpoint), not a per-block projection. Available to
            // emission as the pointee carrier the side-table-retiring rewrite (M2+) consumes; unused here,
            // so byte-neutral by construction (proven via the BC byte-drift gate).
            self.tir_use_pointees = tir.use_pointees.clone();
            // The mixed byte/wide subset of the carrier: the byte→real upgrade in
            // `pointer_pointee_for_value` must skip these (upgrading strands their `uchar` byte cursor).
            self.tir_byte_view_pointers = tir.byte_view_pointers.clone();
            // M3 (pointer-typing rewrite): the pointer-`phi` membership side-tables are now carried on
            // the tir graph (computed once during the build), retiring the emitter's separate
            // `pointer_phi_result_names` / `pointer_phi_incoming_value_names` `body_blocks` text-walks.
            // Byte-identical by construction (same source lines + same `phi ptr` predicate), proven
            // byte-neutral via the BC byte-drift gate.
            self.pointer_phi_values = tir.pointer_phi_results.clone();
            self.pointer_phi_incoming_values = tir.pointer_phi_incoming.clone();
            // M3: the `getelementptr`-result `forward_geps` side-table is likewise carried on the tir
            // graph now, retiring the standalone `forward_gep_results` `body_blocks` walk from the primary
            // path. Byte-identical by construction (same lines + `parse_gep`).
            self.forward_geps = tir.forward_geps.clone();
            for block in &tir.blocks {
                for inst in &block.insts {
                    if matches!(inst.cmp_predicate.as_deref(), Some("eq" | "ne")) {
                        if let [crate::native::tir::TirOperand::Value {
                            name: lhs_name,
                            ty: LlType::Ptr(_),
                        }, crate::native::tir::TirOperand::Value {
                            name: rhs_name,
                            ty: LlType::Ptr(_),
                        }] = inst.operands.as_slice()
                        {
                            self.pointer_payload_values.insert(lhs_name.clone());
                            self.pointer_payload_values.insert(rhs_name.clone());
                        }
                    }
                    if let Some(result) = &inst.result {
                        if let Some((_, incoming)) = &inst.phi_incoming {
                            let mut incoming = incoming.clone();
                            let operands = inst
                                .operands
                                .iter()
                                .map(crate::native::tir::TirOperand::as_typed_value)
                                .collect::<Option<Vec<_>>>();
                            if let Some(operands) = operands {
                                if operands.len() == incoming.len() {
                                    for (entry, operand) in incoming.iter_mut().zip(operands) {
                                        entry.0 = operand.value;
                                    }
                                }
                            }
                            self.tir_phi_incomings.insert(result.clone(), incoming);
                        }
                        self.tir_operands
                            .insert(result.clone(), inst.operands.clone());
                        if let Some(result_ty) = &inst.result_ty {
                            self.tir_result_types
                                .insert(result.clone(), result_ty.clone());
                        }
                        if let Some(pred) = &inst.cmp_predicate {
                            self.tir_predicates.insert(result.clone(), pred.clone());
                        }
                        // `mem_align` is `Some`/None only for load/store; store is result-LESS, so the
                        // only result-keyed entries that ever carry an alignment are loads. The inert
                        // `None` entries for non-load results are never read (`mem_align_of` is called
                        // solely on the load path).
                        self.tir_aligns.insert(result.clone(), inst.mem_align);
                        // `gep_source_ty` is `Some` only for getelementptr results; the inert `None`
                        // entries for other results are never read (the gep emitter is the only consumer).
                        if let Some(src) = &inst.gep_source_ty {
                            self.tir_gep_source_types
                                .insert(result.clone(), src.clone());
                        }
                    }
                    // Historical note: the result-LESS store/void-call operand queues once built here are
                    // retired. The graph walk drives store and void-call straight off the carriers
                    // (`inst.operands`/`inst.call`/`inst.mem_align`), so no per-instruction operand queue
                    // survives.
                }
            }
        }
        // M-A3: the tir carrier is now the SOLE source of the pointer-`phi` membership sets
        // (`pointer_phi_values`/`pointer_phi_incoming_values`) and the `forward_geps` map — the legacy
        // standalone `body_blocks` text-walks that mirrored `collect_pointer_phi_sets`/
        // `collect_forward_geps` are retired. The tir graph is always built (a build failure returned
        // `Err` above), so these sets always reflect the graph.
        self.seed_network_pointees(&body_blocks);
        // R3 STRUCTURAL: drive the emission walk from the typed-IR graph's per-block instruction list
        // (`tir.blocks[i].insts`) — the emission substrate is the typed graph, not a raw line stream
        // (`LlFunction.body` is deleted; text is read once, at parse). Emission sources
        // every opcode's OPERANDS from this same graph, and now the INSTRUCTION STREAM itself. The graph
        // was built from `body_blocks` above, so `tir.blocks[i]` aligns with `body_blocks[i]`. Each
        // straight-line instruction emits from its typed carriers; the block terminator is emitted after
        // the straight-line stream (terminators are not in `insts`) entirely from typed state — the
        // structured `TirTerminator` (`br`/`unreachable`) plus the `ret`/`switch` operand carriers
        // (`TirBlock.ret`/`switch`), no raw terminator line. There is no raw-line fallback: a tir-build
        // failure already returned `Err` above (routing to the retry cascade).
        for block in &body_blocks {
            let id = self.fresh();
            self.block_labels.insert(block.name.clone(), id);
        }
        let mut blocks = Vec::with_capacity(body_blocks.len());
        for (block_idx, body_block) in body_blocks.iter().enumerate() {
            self.current_block = Some(body_block.name.clone());
            let label = *self
                .block_labels
                .get(&body_block.name)
                .ok_or_else(|| format!("native emitter: missing block {}", body_block.name))?;
            let mut instructions = Vec::new();
            let tir_block = &tir.blocks[block_idx];
            for inst in &tir_block.insts {
                // M-A4: dispatch each instruction through the graph-driven `emit_body_inst`, which
                // drives every opcode family straight off the typed `TirInst`. An unmigrated opcode or
                // absent carrier (unreachable in well-formed AIR) is a fail-visible `Err`, not a text fallback.
                self.emit_body_inst(inst, &mut instructions)?;
            }
            self.emit_terminator(
                &tir_block.terminator,
                &tir_block.ret,
                &tir_block.switch,
                &mut instructions,
            )?;
            blocks.push(Block {
                label: Some(Self::inst(Op::Label, None, Some(label), vec![])),
                instructions,
            });
        }
        // W4 (2026-07-16): the post-hoc repair roster is DELETED. `METAL2VULKAN_NO_REPAIR` proved the
        // retry cascade ships 100% of that private capture set spirv-val-valid without it (frontier `--list-fail`
        // EMPTY, banked TOTAL-FAIL 0/15,336; only 3 banked primaries go invalid and all ship via
        // retry-rescue). A reject function (structured_plan did not admit) now emits its inferred
        // (possibly structurally-invalid) merges unrepaired and falls to the retry cascade's relooper
        // tiers, which strip every merge and rebuild the CFG from scratch — the same behavior the old
        // `relooper_feed` / `NO_REPAIR` branch had, now unconditional. This is a byte-CHANGING landing
        // on 48 primary-drift rows (all PSB/BDA `byte_gate_skip` or no-golden — gated by spirv-val +
        // banked status per the plan, not byte-conformance); see kb "NO_REPAIR REFRAME" + the W4
        // terminal record. The structured (admit) path keeps `repair_pre_phi_incoming_materializations`
        // — a phi-incoming access-chain relocation that is CFG-structure-agnostic (relocates a
        // materialization to its unique phi-incoming predecessor edge under dominance guards; never
        // reorders blocks or rewrites merges), load-bearing for the banked phi-materialization rows
        // (f2eeff34/e91bba5c/36f701c5/a4440d75/0691c869/f5d57f02) independent of the deleted roster.
        if structured_active {
            // The structurizer produces correct CFG structure but, like the default path, can still
            // materialize a pointer phi's incoming access-chain inside the phi's OWN block (between phi
            // nodes) — spirv-val rejects it ("OpPhi must appear within a non-entry block before all
            // non-OpPhi instructions"; banked `f2eeff34`/`e91bba5c`/`36f701c5`/`a4440d75`/`0691c869`/
            // `f5d57f02`). This fixup is CFG-structure-agnostic: it only relocates such a materialization
            // to its UNIQUE phi-incoming predecessor edge under dominance guards, never reorders blocks
            // or rewrites merges, so it is safe to run on structured output without reintroducing any of
            // the CFG surgery the W4-deleted repair roster performed.
            self.repair_pre_phi_incoming_materializations(&mut blocks);
        }
        let inline_parameter_substitutions = self
            .inline_parameter_substitutions
            .iter()
            .copied()
            .collect::<HashMap<_, _>>();
        apply_inline_parameter_substitutions(&mut blocks, &inline_parameter_substitutions);
        self.emit_sidecar.remap_ids(&inline_parameter_substitutions);
        self.current_block = None;

        if crate::env_vars::ptr_network_why() {
            self.report_pointer_networks(&f.name, &body_blocks);
        }

        self.module.functions.push(Function {
            def: Some(Self::inst(
                Op::Function,
                Some(ret_id),
                Some(func_id),
                vec![
                    Operand::FunctionControl(FunctionControl::NONE),
                    Operand::IdRef(fn_ty),
                ],
            )),
            end: Some(Self::inst(Op::FunctionEnd, None, None, vec![])),
            parameters: params,
            blocks,
        });
        if self.capture_storage {
            // M1 storage-carrier measurement: record this function's final storage derivation. On a
            // raw retry `emit_function` runs twice; only the succeeding call reaches here, so the
            // snapshot reflects the storage of the adopted emission.
            self.storage_snapshots
                .push((f.name.clone(), self.pointer_storage.clone()));
        }
        if self.capture_pointees {
            // M2 pointee-carrier measurement: record this function's final per-value pointee
            // derivation (the same raw-retry rule as storage above: only the adopted emission reaches
            // here). Compared against the from-tir `use_pointees` carrier in `tir_pointee_check`.
            self.pointee_snapshots
                .push((f.name.clone(), self.pointer_pointees.clone()));
        }
        Ok(())
    }

    /// Insert calls to AIR module static initializers at the selected entry's first executable
    /// instruction. This is the emitter-side form of the retired post-serialization SPIR-V pass:
    /// source function order is preserved, calls follow leading Function `OpVariable`s, and ids are
    /// allocated after ordinary emission/rewrite ids so canonical output stays byte-identical.
    pub(super) fn inject_static_initializer_calls(
        &mut self,
        functions: &[LlFunction],
    ) -> Result<(), String> {
        let entry = match self.ir.entry_name.as_deref() {
            Some(name) => functions
                .iter()
                .find(|function| function.name == name)
                .ok_or_else(|| format!("native emitter: entry function {name} not found"))?,
            None => match functions.first() {
                Some(function) => function,
                None => return Ok(()),
            },
        };
        let entry_id = *self
            .function_ids
            .get(&entry.name)
            .ok_or_else(|| format!("native emitter: missing function id for {}", entry.name))?;
        let initializer_ids = functions
            .iter()
            .filter(|function| {
                function.name != entry.name
                    && function.name.starts_with("_GLOBAL__sub_I")
                    && !self
                        .ir
                        .preinlined_static_initializers
                        .contains(&function.name)
            })
            .map(|function| {
                self.function_ids
                    .get(&function.name)
                    .copied()
                    .ok_or_else(|| {
                        format!("native emitter: missing function id for {}", function.name)
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        if initializer_ids.is_empty() {
            return Ok(());
        }

        let void = self.type_id(&LlType::Void)?;
        let calls = initializer_ids
            .into_iter()
            .map(|callee| {
                Self::inst(
                    Op::FunctionCall,
                    Some(void),
                    Some(self.fresh()),
                    vec![Operand::IdRef(callee)],
                )
            })
            .collect::<Vec<_>>();
        let entry = self
            .module
            .functions
            .iter_mut()
            .find(|function| {
                function.def.as_ref().and_then(|def| def.result_id) == Some(entry_id)
                    && !function.blocks.is_empty()
            })
            .ok_or_else(|| {
                format!(
                    "native emitter: emitted entry function {} not found",
                    entry.name
                )
            })?;
        let first_block = entry
            .blocks
            .first_mut()
            .ok_or_else(|| format!("native emitter: entry function {} has no block", entry_id))?;
        let insert_at = first_block
            .instructions
            .iter()
            .take_while(|instruction| instruction.class.opcode == Op::Variable)
            .count();
        first_block.instructions.splice(insert_at..insert_at, calls);
        Ok(())
    }

    /// The IR use-implied pointee per pointer value — the granularity BEFORE the byte-view flattening
    /// that records `Int(8)` at def time. Built from the tir `use_pointees` carrier (resolved through
    /// named aliases), falling back to the recorded `pointer_pointees` for values the carrier does not
    /// cover. This is the source the M-A2 network fix must record from: `pointer_pointees` alone
    /// byte-flattens the byte-addressed arm of a whole-vs-part network to `Int(8)`, disguising it as a
    /// reinterpret-mix.
    fn use_implied_pointees(&self) -> HashMap<String, LlType> {
        let mut m = self.pointer_pointees.clone();
        for (name, ty) in &self.tir_use_pointees {
            if let Ok(resolved) = self.resolve_type(ty) {
                m.insert(name.clone(), resolved);
            }
        }
        m
    }

    /// M-A2 def-site recording: seed `network_pointees` with the uniform ACCESS pointee
    /// for every pointer network whose TRUE IR access granularity is CONSISTENT across the whole
    /// component (census class `Uniform`, one concrete non-byte pointee). `pointer_meta_for_value` then
    /// reports that type for every member, so `pointer_merge_meta` reconciles the phi/select on it
    /// instead of erroring on the byte-view `Int(8)` the raw recording flattens the byte-addressed arm
    /// to. Restricted to the access-uniform case: every member's real access already matches the seeded
    /// type, so no load/store retyping (scalarization) and no GEP re-striding is needed — the
    /// mixed-granularity (whole-vs-part) minority is left alone. Default-off; byte-changing when set, so
    /// flip is G7/G8-gated.
    fn seed_network_pointees(&mut self, body_blocks: &[BodyBlock]) {
        use crate::native::emitter::pointer_network::{
            analyze_networks_by_access, array_indexed_scalar_bases, NetworkClass,
        };
        // Pointers stepped as an ARRAY of their bare scalar element (a non-identity scalar-stride GEP):
        // recording the scalar pointee mis-declares the object as a scalar `OpVariable`, so any network
        // touching one is excluded here (it needs the object re-declared as an array + indices
        // re-strided — the unbuilt M-A2(c) #2/#3 keystone). Excluding keeps the seed a STRICT SUBSET of
        // the sound access-uniform set, so it can only reduce fails, never add.
        let array_indexed = array_indexed_scalar_bases(body_blocks);
        // Census by the TRUE IR ACCESS WIDTH (loads/stores/gep-steps), NOT the element-scalar carrier.
        // The carrier (`analyze_pointer_networks` + `use_implied_pointees`) reports the innermost scalar,
        // so it flattens a real whole-vs-part component (`{Vector(Float,4), Float}` accesses) to
        // `Uniform [Float]` and seeds the scalar — but the GEP/load def sites still derive
        // `Vector(Float,4)` fresh from the LLVM source type (`gep_pointee`), so recording the scalar
        // leaves the pointer PARTIALLY retyped and emits invalid SPIR-V (measured: 11 frontier
        // regressions). The access census classifies that same component `WholeVsPart(Float)` (non-
        // Uniform), so it is SKIPPED here; only a genuinely access-uniform network — every member
        // dereferenced/stepped at ONE type — is seeded, where recording that whole type cannot disagree
        // with any def site. Sound consistent widening of the whole-vs-part networks needs coordinated
        // GEP re-striding (the remaining M-A2(c) keystone work), not a read-side seed.
        for net in analyze_networks_by_access(body_blocks) {
            if !matches!(net.class, NetworkClass::Uniform) {
                continue;
            }
            // Uniform means 0 or 1 distinct access pointee. Seed only when there IS one concrete,
            // non-byte-view pointee to record (an empty or `Int(8)`-only network carries no widening).
            let [pointee] = net.pointees.as_slice() else {
                continue;
            };
            if matches!(pointee, LlType::Int(8)) {
                continue;
            }
            // Skip a network any of whose members is stepped as an array of its bare scalar element —
            // seeding the scalar there mis-declares the object (see `array_indexed_scalar_bases`).
            if net.members.iter().any(|m| array_indexed.contains(m)) {
                continue;
            }
            // Skip a network a member of which is a LOGICAL (non-word-addressable) pointer that already
            // carries a CONCRETE (non-byte-view) def-site pointee DISAGREEING with the uniform ACCESS
            // pointee. The access census is body-local, so it misses a whole-vs-part split whose narrow
            // evidence lives at a def site the body never re-derives — the canonical case is a Workgroup/
            // Private pointer PARAMETER (its pointee comes from the callsite/arg metadata into
            // `pointer_pointees` before this seed runs) that the body then dereferences at a wider
            // granularity (an `addrspace(3)` scalar `float*` scratch arg loaded as `<4 x float>` in a
            // helper). Recording the wide access pointee there mis-declares the arg, and a LOGICAL pointer
            // (Workgroup/Private) has NO raw-word view — re-viewing it to the wide type needs an illegal
            // `OpBitcast` on a logical pointer (`body.rs` "cannot reinterpret workgroup pointer arg"). A
            // word-addressable device pointer (`UniformConstant`/StorageBuffer) CAN be reinterpreted via
            // the raw byte-GEP model, so it is NOT excluded — that is the whole-vs-part population RECORD
            // exists to seed. Gating on logical storage (not on any disagreement) keeps the exclusion a
            // STRICT SUBSET that removes only the un-reinterpretable logical members, so it can only
            // reduce fails, never add, without gutting the device-buffer wins.
            if net.members.iter().any(|m| {
                self.pointer_pointees
                    .get(m)
                    .is_some_and(|p| !matches!(p, LlType::Int(8)) && p != pointee)
                    && matches!(
                        self.pointer_storage.get(m),
                        Some(StorageClass::Workgroup | StorageClass::Private)
                    )
            }) {
                continue;
            }
            for member in &net.members {
                self.network_pointees
                    .insert(member.clone(), pointee.clone());
            }
        }
    }

    /// `METAL2VULKAN_PTR_NETWORK_WHY` diagnostic: print each pointer network classified by the TRUE IR
    /// ACCESS WIDTH (`analyze_networks_by_access` — loads/stores/geps) as the PRIMARY tag, with the
    /// element-scalar carrier class in `carrier=` when it disagrees. The access census is the honest one:
    /// the carrier reports the innermost scalar, so it flattens a real whole-vs-part component to
    /// `Uniform [Float]` (measured: seeding that flattened scalar regresses 11 frontier cases), whereas
    /// the access census sees the `<4 x float>` load and reports `WholeVsPart(Float)`. Prints any network
    /// the access census finds non-uniform OR where the two censuses disagree. Read-only — feeds no
    /// emission. Covers functions whose emission completed (a function that errors mid-body never reaches
    /// here; those shapes are exercised by `pointer_network`'s unit tests instead).
    fn report_pointer_networks(&self, fn_name: &str, body_blocks: &[BodyBlock]) {
        use crate::native::emitter::pointer_network::{
            analyze_networks_by_access, analyze_pointer_networks, NetworkClass,
        };
        let use_implied = self.use_implied_pointees();
        let carrier_class: std::collections::HashMap<Vec<String>, NetworkClass> =
            analyze_pointer_networks(body_blocks, &use_implied)
                .into_iter()
                .map(|n| (n.members, n.class))
                .collect();
        for net in analyze_networks_by_access(body_blocks) {
            let carrier = carrier_class.get(&net.members);
            if matches!(net.class, NetworkClass::Uniform)
                && matches!(carrier, Some(NetworkClass::Uniform) | None)
            {
                continue;
            }
            let carrier_tag = match carrier {
                Some(c) if c != &net.class => format!(" carrier={c:?}"),
                _ => String::new(),
            };
            eprintln!(
                "PTR-NETWORK {fn_name} {:?}{carrier_tag} members={} pointees={:?}",
                net.class,
                net.members.len(),
                net.pointees,
            );
        }
    }

    /// Find module globals that an integer atomic (`air.atomic.*.i32`) dereferences directly, so
    /// `emit_global` can declare them with an `i32` pointee instead of their float type (the
    /// atomic-min/max bit-pattern idiom over a threadgroup scratch slot). Reasoned purely from the
    /// `air.atomic.*` ABI symbol family — the allowed structural exception (AGENTS.md) — and the
    /// operand being a bare `LlValue::Global`, never a shader name. A global also seen under a float
    /// atomic (`air.atomic.*.f32`) is excluded: retyping it to `i32` would only move the illegal
    /// pointer bitcast to the float-atomic site.
    pub(super) fn scan_int_atomic_reinterpret_globals(
        globals: &[LlGlobal],
        functions: &[LlFunction],
    ) -> HashSet<String> {
        let global_names: HashSet<&str> = globals.iter().map(|g| g.name.as_str()).collect();
        let mut int_atomic = HashSet::new();
        let mut float_atomic = HashSet::new();
        for function in functions {
            // Read the parsed call off the typed carrier (`inst.call`) instead of re-lexing `body`. The
            // carrier's `resolve_call` is broader than the old `strip_call_prefix` (it also parses
            // `musttail`/`notail`), so gate on `opcode ∈ {call, tail}` to reproduce the old acceptance
            // exactly (`call …` / `tail call …` only) — byte-identical.
            for block in &function.blocks {
                let Some(carrier) = &block.typed else {
                    continue;
                };
                for inst in &carrier.insts {
                    if !matches!(inst.opcode.as_str(), "call" | "tail") {
                        continue;
                    }
                    let Some(call) = &inst.call else { continue };
                    if !call.callee.starts_with("air.atomic.") {
                        continue;
                    }
                    let Some(LlValue::Global(name)) = call.args.first().map(|a| &a.value) else {
                        continue;
                    };
                    if !global_names.contains(name.as_str()) {
                        continue;
                    }
                    if call.callee.ends_with(".i32") {
                        int_atomic.insert(name.clone());
                    } else if call.callee.ends_with(".f32") {
                        float_atomic.insert(name.clone());
                    }
                }
            }
        }
        int_atomic
            .into_iter()
            .filter(|name| !float_atomic.contains(name))
            .collect()
    }

    /// The pointee type a module global is *declared* with in SPIR-V. Normally its LLVM type, except
    /// an integer atomic on a float-typed threadgroup global needs an `i32`-typed pointer to that exact
    /// memory; under Logical addressing that pointer only exists if the variable itself is declared
    /// `i32`. Retype those (Workgroup scratch only, so there is no initializer to reconcile); the float
    /// load/store value accesses then reinterpret through the existing 32-bit scalar
    /// `OpBitcast`-on-value load/store paths, and the atomic gets a clean `i32` pointer with no illegal
    /// logical-pointer bitcast. Used by `emit_global` (declaration) and the per-function pointee reset
    /// (`emit_function` clears `pointer_pointees` and reseeds globals) so both agree.
    pub(super) fn global_declared_pointee(&mut self, global: &LlGlobal) -> Result<LlType, String> {
        let ty = self.resolve_type(&global.ty)?;
        if global.addrspace == 3
            && ty == LlType::Float
            && self.int_atomic_reinterpret_globals.contains(&global.name)
        {
            return Ok(LlType::Int(32));
        }
        // A constant table accessed through a GEP whose source type is NOT the declared type (a
        // reinterpret view — e.g. a packed byte-table struct addressed as `[16 x [32 x i8]]` with a
        // dynamic row index, which is invalid as a structural chain since struct indices must be
        // constants). When every leaf of the declared type is `i8` the byte image is exact (i8
        // fields/arrays have alignment 1, so there is no padding), so declare the variable as the
        // flat byte array; every view then lowers through the byte-array raw paths.
        if global.addrspace != 3 && self.byte_view_reinterpret_globals.contains(&global.name) {
            if let Some(size) = i8_leaf_byte_size(&ty) {
                return Ok(LlType::Array(Box::new(LlType::Int(8)), size));
            }
        }
        Ok(ty)
    }

    /// Scan for globals used as the base of a `getelementptr` whose SOURCE type differs from the
    /// global's declared type — the byte-table reinterpret-view shape `global_declared_pointee`
    /// remodels to a flat byte array. Textual companion to `scan_int_atomic_reinterpret_globals`.
    pub(super) fn scan_byte_view_reinterpret_globals(&mut self) -> Result<HashSet<String>, String> {
        let globals = self.ir.globals.clone();
        let functions = self.ir.functions.clone();
        let mut reinterpreted = HashSet::new();
        let declared: HashMap<&str, &LlType> =
            globals.iter().map(|g| (g.name.as_str(), &g.ty)).collect();
        for function in &functions {
            // Read the parsed GEP off the typed carrier (`inst.gep`, set by `resolve_gep` = the same
            // `parse_gep` on the same `after "getelementptr "` text) instead of re-lexing `body` —
            // byte-identical.
            for block in &function.blocks {
                let Some(carrier) = &block.typed else {
                    continue;
                };
                for inst in &carrier.insts {
                    let Some(gep) = &inst.gep else { continue };
                    let LlValue::Global(base) = &gep.base.value else {
                        continue;
                    };
                    let Some(declared_ty) = declared.get(base.as_str()) else {
                        continue;
                    };
                    let declared_ty = self.resolve_type(declared_ty)?;
                    let source_ty = self.resolve_type(&gep.source_ty)?;
                    if source_ty != declared_ty && i8_leaf_byte_size(&declared_ty).is_some() {
                        reinterpreted.insert(base.clone());
                    }
                }
            }
        }
        Ok(reinterpreted)
    }

    pub(super) fn emit_global(&mut self, global: &LlGlobal) -> Result<(), String> {
        let ty = self.global_declared_pointee(global)?;
        let storage = if global.addrspace == 3 {
            StorageClass::Workgroup
        } else {
            StorageClass::Private
        };
        let ptr_ty = self.ptr_type_id(storage, &ty)?;
        let initializer = if storage == StorageClass::Private {
            Some(match &global.initializer {
                Some(initializer) => {
                    let initializer_ty = self.resolve_type(&initializer.ty)?;
                    if initializer_ty != ty {
                        // The byte-view remodel declared this global as a flat `[N x i8]`
                        // (`global_declared_pointee`); flatten the all-i8-leaf initializer to the
                        // same byte image.
                        let LlType::Array(elem, len) = &ty else {
                            return Err(format!(
                                "native emitter: global {} initializer type {:?} does not match {:?}",
                                global.name, initializer_ty, ty
                            ));
                        };
                        if elem.as_ref() != &LlType::Int(8)
                            || !self.byte_view_reinterpret_globals.contains(&global.name)
                        {
                            return Err(format!(
                                "native emitter: global {} initializer type {:?} does not match {:?}",
                                global.name, initializer_ty, ty
                            ));
                        }
                        let mut bytes = Vec::new();
                        self.append_i8_initializer_bytes(
                            &initializer.value,
                            &initializer_ty,
                            &mut bytes,
                        )?;
                        if bytes.len() != *len as usize {
                            return Err(format!(
                                "native emitter: global {} byte-flattened initializer is {} bytes, declared {}",
                                global.name,
                                bytes.len(),
                                len
                            ));
                        }
                        let flat = LlValue::Array(
                            bytes
                                .into_iter()
                                .map(|byte| TypedValue {
                                    ty: LlType::Int(8),
                                    value: LlValue::Int(u64::from(byte)),
                                })
                                .collect(),
                        );
                        self.const_initializer_id(&flat, &ty)?
                    } else {
                        self.const_initializer_id(&initializer.value, &initializer.ty)?
                    }
                }
                None => self.const_null(&ty)?,
            })
        } else {
            None
        };
        let id = self.fresh();
        let mut operands = vec![Operand::StorageClass(storage)];
        if let Some(initializer) = initializer {
            operands.push(Operand::IdRef(initializer));
        }
        self.module.types_global_values.push(Self::inst(
            Op::Variable,
            Some(ptr_ty),
            Some(id),
            operands,
        ));
        self.module.debug_names.push(Self::inst(
            Op::Name,
            None,
            None,
            vec![
                Operand::IdRef(id),
                Operand::LiteralString(global.name.trim_start_matches('@').to_string()),
            ],
        ));
        self.global_values
            .insert(global.name.clone(), (id, LlType::Ptr(global.addrspace)));
        self.pointer_pointees.insert(global.name.clone(), ty);
        Ok(())
    }

    /// Serialize an all-i8-leaf constant initializer to its byte image (the flat-byte-array remodel
    /// of `global_declared_pointee`). `Zero`/`Undef` fill their type's byte size with zeros.
    fn append_i8_initializer_bytes(
        &mut self,
        value: &LlValue,
        ty: &LlType,
        out: &mut Vec<u8>,
    ) -> Result<(), String> {
        let ty = self.resolve_type(ty)?;
        match value {
            LlValue::Zero | LlValue::Undef => {
                let size = i8_leaf_byte_size(&ty)
                    .ok_or_else(|| format!("native emitter: cannot byte-flatten zero of {ty:?}"))?;
                out.extend(std::iter::repeat_n(0u8, size as usize));
                Ok(())
            }
            LlValue::Int(v) if ty == LlType::Int(8) => {
                out.push(*v as u8);
                Ok(())
            }
            LlValue::SignedInt(v) if ty == LlType::Int(8) => {
                out.push(*v as u8);
                Ok(())
            }
            LlValue::Hex(v) if ty == LlType::Int(8) => {
                out.push(*v as u8);
                Ok(())
            }
            LlValue::Array(elems) | LlValue::Struct(elems) => {
                for elem in elems {
                    self.append_i8_initializer_bytes(&elem.value, &elem.ty, out)?;
                }
                Ok(())
            }
            other => Err(format!(
                "native emitter: cannot byte-flatten initializer {other:?} of {ty:?}"
            )),
        }
    }

    pub(super) fn function_param_concrete_pointee(
        &self,
        func: &str,
        index: usize,
        name: &str,
    ) -> Option<LlType> {
        self.function_param_pointees
            .get(&(func.to_string(), index))
            .cloned()
            .or_else(|| {
                self.ir
                    .ptr_pointees
                    .get(&(func.to_string(), name.to_string()))
                    .cloned()
            })
    }

    pub(super) fn concrete_vector_workgroup_raw_param_pointee(
        &self,
        func: &str,
        index: usize,
        name: &str,
    ) -> Option<LlType> {
        if !self
            .ir
            .raw_buffer_params
            .contains(&(func.to_string(), name.to_string()))
        {
            return None;
        }
        self.function_param_concrete_pointee(func, index, name)
            .filter(vector_backed_workgroup_raw_pointee)
    }

    pub(super) fn param_type_id(
        &mut self,
        func: &str,
        index: usize,
        name: &str,
        ty: &LlType,
    ) -> Result<Word, String> {
        if let LlType::Ptr(addrspace) = ty {
            if self
                .ir
                .raw_buffer_params
                .contains(&(func.to_string(), name.to_string()))
            {
                if *addrspace == 3 {
                    if let Some(pointee) =
                        self.concrete_vector_workgroup_raw_param_pointee(func, index, name)
                    {
                        return self.ptr_type_id(StorageClass::Workgroup, &pointee);
                    }
                    return self.ptr_type_id(StorageClass::Workgroup, &raw_workgroup_array_type());
                }
                return self.ptr_type_id(StorageClass::UniformConstant, &raw_buffer_block_type());
            }
            let storage = if *addrspace == 4
                && (self.ir.imageblock_dimensions.is_some() || self.ir.imageblock_shared_cells)
            {
                StorageClass::Workgroup
            } else {
                llvm_pointer_storage(*addrspace)?
            };
            if let Some(pointee) = self
                .function_param_pointees
                .get(&(func.to_string(), index))
                .cloned()
            {
                return self.ptr_type_id(storage, &pointee);
            }
            if let Some(pointee) = self
                .ir
                .ptr_pointees
                .get(&(func.to_string(), name.to_string()))
                .cloned()
            {
                return self.ptr_type_id(storage, &pointee);
            }
        }
        self.type_id(ty)
    }

    pub(super) fn emit_declaration(&mut self, decl: &LlDeclaration) -> Result<(), String> {
        let ret_ty = self.resolve_type(&decl.ret)?;
        let ret_id = self.type_id(&ret_ty)?;
        let param_types: Vec<Word> = decl
            .params
            .iter()
            .map(|ty| self.type_id(ty))
            .collect::<Result<Vec<_>, _>>()?;
        let fn_ty = self.function_type_id(ret_id, &param_types);

        let func_id = *self
            .function_ids
            .get(&decl.name)
            .ok_or_else(|| format!("native emitter: missing declaration id for {}", decl.name))?;
        self.module.debug_names.push(Self::inst(
            Op::Name,
            None,
            None,
            vec![
                Operand::IdRef(func_id),
                Operand::LiteralString(decl.name.clone()),
            ],
        ));
        let mut params = Vec::with_capacity(param_types.len());
        for type_id in &param_types {
            let id = self.fresh();
            params.push(Self::inst(
                Op::FunctionParameter,
                Some(*type_id),
                Some(id),
                vec![],
            ));
        }
        self.module.functions.push(Function {
            def: Some(Self::inst(
                Op::Function,
                Some(ret_id),
                Some(func_id),
                vec![
                    Operand::FunctionControl(FunctionControl::NONE),
                    Operand::IdRef(fn_ty),
                ],
            )),
            end: Some(Self::inst(Op::FunctionEnd, None, None, vec![])),
            parameters: params,
            blocks: vec![],
        });
        Ok(())
    }

    fn function_type_id(&mut self, ret_id: Word, param_types: &[Word]) -> Word {
        let mut key = Vec::with_capacity(param_types.len() + 1);
        key.push(ret_id);
        key.extend_from_slice(param_types);
        if let Some(id) = self.interner.function_types.get(&key) {
            return *id;
        }
        let id = self.fresh();
        let mut operands = vec![Operand::IdRef(ret_id)];
        operands.extend(param_types.iter().map(|id| Operand::IdRef(*id)));
        self.module.types_global_values.push(Self::inst(
            Op::TypeFunction,
            None,
            Some(id),
            operands,
        ));
        self.interner.function_types.insert(key, id);
        id
    }
}

fn apply_inline_parameter_substitutions(blocks: &mut [Block], substitutions: &HashMap<Word, Word>) {
    if substitutions.is_empty() {
        return;
    }
    for instruction in blocks
        .iter_mut()
        .flat_map(|block| block.instructions.iter_mut())
    {
        for operand in &mut instruction.operands {
            if let Operand::IdRef(id) = operand {
                if let Some(replacement) = substitutions.get(id) {
                    *id = *replacement;
                }
            }
        }
    }
}

/// A block TERMINATOR is one of `br` / `switch` / `ret` / `unreachable`. `TirBlock` carries its
/// terminator separately from `insts`, so the graph walk emits the straight-line `insts` from the typed
/// graph and then the terminator entirely from typed state. The keywords are reserved LLVM terminators,
/// so no value-defining (`%r = ...`) line matches — there is exactly one terminator per block.
fn reorder_forward_local_def_blocks(
    body_blocks: &mut Vec<BodyBlock>,
    defuse: &ReorderDefUse,
) -> Result<(), String> {
    if body_blocks.len() <= 2 {
        return Ok(());
    }

    let mut seen_orders = HashSet::new();
    let max_moves = body_blocks.len() * body_blocks.len();
    let mut moves = 0;
    loop {
        let order = body_blocks
            .iter()
            .map(|block| block.name.clone())
            .collect::<Vec<_>>();
        if !seen_orders.insert(order) {
            return Err(format!(
                "native emitter: cyclic forward local block dependencies while reordering blocks after {moves} moves"
            ));
        }

        // Index local uses by current block index once per move. We still scan definitions in current
        // block/instruction order below, so the selected move is byte-for-byte equivalent to the old
        // nested scan: earliest use block wins, and within that block the earliest forward definition
        // in current order wins.
        let mut use_indices_by_name: HashMap<&str, Vec<usize>> = HashMap::new();
        for (idx, block) in body_blocks.iter().enumerate().skip(1) {
            if let Some(uses) = defuse.uses_by_block.get(&block.name) {
                for name in uses {
                    use_indices_by_name
                        .entry(name.as_str())
                        .or_default()
                        .push(idx);
                }
            }
        }
        let mut first_forward_def_by_use_idx: Vec<Option<usize>> = vec![None; body_blocks.len()];
        for (def_idx, block) in body_blocks.iter().enumerate() {
            let Some(names) = defuse.defs_by_block.get(&block.name) else {
                continue;
            };
            for name in names {
                let Some(use_indices) = use_indices_by_name.get(name.as_str()) else {
                    continue;
                };
                for &use_idx in use_indices {
                    if def_idx > use_idx && first_forward_def_by_use_idx[use_idx].is_none() {
                        first_forward_def_by_use_idx[use_idx] = Some(def_idx);
                    }
                }
            }
        }
        let mut moved = false;
        if let Some((idx, def_idx)) = first_forward_def_by_use_idx
            .iter()
            .enumerate()
            .skip(1)
            .filter_map(|(idx, def_idx)| def_idx.map(|def_idx| (idx, def_idx)))
            .next()
        {
            let block = body_blocks.remove(def_idx);
            body_blocks.insert(idx, block);
            moves += 1;
            if moves > max_moves {
                return Err(format!(
                    "native emitter: forward local block reorder budget exceeded after {moves} moves"
                ));
            }
            moved = true;
        }
        if !moved {
            break;
        }
    }
    Ok(())
}

/// The forward-reorder def/use facts, sourced ONCE from the typed per-block graph
/// (`tir::build_from_blocks`) instead of re-lexing `BodyBlock.lines` on every reorder iteration. Keyed
/// by block NAME (labels are unique and reorder never renames/creates blocks), so the maps stay valid as
/// reorder permutes the block Vec. Reproduces the retired text scan exactly:
/// - `defs_by_block`: every result-defining instruction's `%name`, in instruction order, EXCLUDING a
///   scalar-pointer `phi` (`phi ptr` / `phi ptr addrspace(...)`) — those are not reorder candidates.
/// - `uses_by_block`: the local `%value` uses that can bind a forward def — every NON-`phi`
///   instruction's value operands (a `phi` contributes ZERO uses, matching the text scan) plus the
///   terminator's condition / selector / return value.
struct ReorderDefUse {
    defs_by_block: HashMap<String, Vec<String>>,
    uses_by_block: HashMap<String, HashSet<String>>,
}

impl ReorderDefUse {
    fn from_blocks(body_blocks: &[BodyBlock]) -> Result<Self, String> {
        let tir = crate::native::tir::build_from_blocks(body_blocks)?;
        let mut defs_by_block = HashMap::new();
        let mut uses_by_block = HashMap::new();
        for block in &tir.blocks {
            let mut defs = Vec::new();
            let mut uses = HashSet::new();
            for inst in &block.insts {
                if let Some(result) = &inst.result {
                    // The one non-reorderable defining form: a scalar-pointer phi (the text scan
                    // skipped `phi ptr` / `phi ptr addrspace(`).
                    let scalar_ptr_phi =
                        inst.opcode == "phi" && matches!(inst.result_ty, Some(LlType::Ptr(_)));
                    if !scalar_ptr_phi {
                        defs.push(result.clone());
                    }
                }
                // A `phi` contributes no forward-binding uses (its incoming values arrive along
                // predecessor edges, not within this block); the text scan returned an empty set for it.
                if inst.opcode != "phi" {
                    uses.extend(inst.uses.iter().cloned());
                }
            }
            terminator_local_uses(&block.terminator, &mut uses);
            defs_by_block.insert(block.label.clone(), defs);
            uses_by_block.insert(block.label.clone(), uses);
        }
        Ok(Self {
            defs_by_block,
            uses_by_block,
        })
    }
}

/// The local `%value` uses a terminator reads: a conditional branch's condition, a switch's selector, or
/// a return value. Unconditional `br label`, `ret void`, and `unreachable` read no value.
fn terminator_local_uses(term: &crate::native::tir::TirTerminator, uses: &mut HashSet<String>) {
    use crate::native::tir::TirTerminator;
    let operand = match term {
        TirTerminator::BrCond { cond, .. } => Some(cond.as_str()),
        TirTerminator::Switch { selector, .. } => Some(selector.as_str()),
        TirTerminator::Ret(Some(value)) => Some(value.as_str()),
        TirTerminator::Br(_) | TirTerminator::Ret(None) | TirTerminator::Unreachable => None,
    };
    if let Some(operand) = operand {
        let mut names = Vec::new();
        crate::native::tir::collect_value_names(operand, &mut names);
        uses.extend(names);
    }
}

fn vector_backed_workgroup_raw_pointee(pointee: &LlType) -> bool {
    match pointee {
        LlType::Vector(_, _) => true,
        LlType::Array(elem, _) => matches!(elem.as_ref(), LlType::Vector(_, _)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(name: &str, lines: &[&str]) -> BodyBlock {
        let lines: Vec<String> = lines.iter().map(|line| line.to_string()).collect();
        let typed = crate::native::tir::lower_block_carrier(name, &lines, &HashMap::new());
        BodyBlock {
            name: name.to_string(),
            role: crate::native::cfg::BlockRole::Normal,
            typed,
        }
    }

    fn reorder(blocks: &mut Vec<BodyBlock>) -> Result<(), String> {
        let defuse = ReorderDefUse::from_blocks(blocks)?;
        reorder_forward_local_def_blocks(blocks, &defuse)
    }

    #[test]
    fn forward_local_reorder_moves_def_before_later_use() {
        let mut blocks = vec![
            block("%entry", &["br label %use"]),
            block("%use", &["%use.value = fadd float %later, 1.0", "ret void"]),
            block("%def", &["%later = fadd float 1.0, 2.0", "ret void"]),
        ];

        reorder(&mut blocks).unwrap();

        let names = blocks
            .iter()
            .map(|block| block.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["%entry", "%def", "%use"]);
    }

    #[test]
    fn forward_local_reorder_reports_cycles_instead_of_looping() {
        let mut blocks = vec![
            block("%entry", &["br label %a"]),
            block("%a", &["%a.value = fadd float %b.value, 1.0", "ret void"]),
            block("%b", &["%b.value = fadd float %a.value, 1.0", "ret void"]),
        ];

        let error = reorder(&mut blocks).unwrap_err();

        assert!(error.contains("cyclic forward local block dependencies"));
    }

    #[test]
    fn reorder_defuse_ignores_definition_lhs_and_phi_incoming_values() {
        let blocks = vec![block(
            "%body",
            &[
                "%lhs = fadd float %rhs, 1.0",
                "%phi = phi i32 [ %from.a, %a ], [ %from.b, %b ]",
                "br i1 %cond, label %then, label %else",
            ],
        )];

        let defuse = ReorderDefUse::from_blocks(&blocks).unwrap();
        let uses = &defuse.uses_by_block["%body"];

        assert!(uses.contains("%rhs"));
        assert!(uses.contains("%cond"));
        assert!(!uses.contains("%lhs"));
        assert!(!uses.contains("%phi"));
        assert!(!uses.contains("%from.a"));
        assert!(!uses.contains("%then"));
    }

    #[test]
    fn deferred_inline_parameter_substitution_rewrites_uses_not_definitions() {
        let mut blocks = vec![Block {
            label: Some(Emitter::inst(Op::Label, None, Some(1), vec![])),
            instructions: vec![
                Emitter::inst(Op::CopyObject, Some(2), Some(20), vec![Operand::IdRef(10)]),
                Emitter::inst(
                    Op::IAdd,
                    Some(2),
                    Some(21),
                    vec![Operand::IdRef(10), Operand::IdRef(20)],
                ),
            ],
        }];

        apply_inline_parameter_substitutions(&mut blocks, &HashMap::from([(10, 7)]));

        assert_eq!(blocks[0].instructions[0].result_id, Some(20));
        assert_eq!(blocks[0].instructions[0].operands, vec![Operand::IdRef(7)]);
        assert_eq!(
            blocks[0].instructions[1].operands,
            vec![Operand::IdRef(7), Operand::IdRef(20)]
        );
    }
}
