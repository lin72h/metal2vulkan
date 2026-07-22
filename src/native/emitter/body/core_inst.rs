//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    /// R3 STRUCTURAL / M-A4: the graph-driven emission entry, and the SOLE straight-line dispatcher on
    /// the graph walk (`emit_function`, when tir built and its block count matches). Dispatches EVERY
    /// opcode family straight off the typed `TirInst` (`opcode` + `operands`): the two-operand
    /// arithmetic/bitwise ops (`binary_op_dispatch`), single-operand `fneg`/`freeze`
    /// (`unary_op_dispatch`), the conversions (`convert_op_dispatch`), `select`, `fcmp`/`icmp`,
    /// `inttoptr`/`ptrtoint`, `getelementptr`, `load`, `store`, the vector/aggregate element ops
    /// (`extractelement`/`insertelement`/`shufflevector`/`extractvalue`/`insertvalue`), `alloca`, `phi`,
    /// `bitcast`, and `call` (void + value). Byte-identical to the retired text path by construction:
    /// the operands come from the same `inst.operands` that path re-keyed through `tir_operands`, so the
    /// same emitter runs with the same `Op` and the same operand values.
    ///
    /// When a migrated op's operands do not resolve to the expected typed arity (an `Unresolved`
    /// operand), or a per-op carrier is absent, the instruction reaches the fail-visible `Err`
    /// (`reason=graph_walk_unmigrated_opcode`) at the end of this function — the retired-substrate
    /// discipline: there is no per-instruction re-lex fallback any more (`inst.text` is deleted). That
    /// `Err` is measured DEAD on private capture sets historically used for gating (every family resolves) and routes the function to the retry
    /// cascade (which rebuilds the CFG via the relooper) — the text-walk emission path is retired.
    pub(in crate::native::emitter) fn emit_body_inst(
        &mut self,
        inst: &crate::native::tir::TirInst,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        if inst.opcode == "metal2vulkan.inline_parameter" {
            let (Some(name), Some(argument)) = (
                inst.result.as_deref(),
                inst.operands
                    .first()
                    .and_then(crate::native::tir::TirOperand::as_typed_value),
            ) else {
                return Err("native emitter: malformed typed inline parameter \
                     (reason=inline_parameter_missing_operand)"
                    .to_string());
            };
            return self.bind_inline_parameter(name, argument, instructions);
        }
        if let Some(name) = &inst.result {
            if let Some((op, kind)) = binary_op_dispatch(&inst.opcode) {
                if let Some(operands) = self.tir_inst_typed_operands(inst) {
                    if let [lhs, rhs] = operands.as_slice() {
                        let (lhs, rhs, name) = (lhs.clone(), rhs.clone(), name.clone());
                        return match kind {
                            BinaryKind::Int => {
                                self.emit_binary_int_op_resolved(op, lhs, rhs, name, instructions)
                            }
                            BinaryKind::Float => {
                                self.emit_binary_float_op_resolved(op, lhs, rhs, name, instructions)
                            }
                            BinaryKind::Signed => self.emit_signed_binary_int_op_resolved(
                                op,
                                lhs,
                                rhs,
                                name,
                                instructions,
                            ),
                        };
                    }
                }
            } else if let Some(kind) = unary_op_dispatch(&inst.opcode) {
                if let Some(operands) = self.tir_inst_typed_operands(inst) {
                    if let [value] = operands.as_slice() {
                        let (value, name) = (value.clone(), name.clone());
                        return match kind {
                            UnaryKind::Fneg => self.emit_unary_float_op_resolved(
                                Op::FNegate,
                                value,
                                name,
                                instructions,
                            ),
                            UnaryKind::Freeze => {
                                self.emit_freeze_resolved(value, name, instructions)
                            }
                        };
                    }
                }
            } else if let Some(kind) = convert_op_dispatch(&inst.opcode) {
                // Conversions need the source operand AND the dest type. The source is
                // `inst.operands[0]`; the dest is `inst.result_ty` resolved — byte-identical to the text
                // path's `convert_dst_type`, which reads the same `tir_result_types` entry. Both must be
                // present, else it reaches the fail-visible unmigrated-opcode `Err` (the retry cascade
                // owns the raw `.lines` text walk).
                if let (Some(operands), Some(result_ty)) =
                    (self.tir_inst_typed_operands(inst), inst.result_ty.as_ref())
                {
                    if let [src] = operands.as_slice() {
                        let dst_ty = self.resolve_type(result_ty)?;
                        let (src, name) = (src.clone(), name.clone());
                        return match kind {
                            ConvertKind::Int(op) => {
                                self.emit_int_convert_resolved(op, src, dst_ty, name, instructions)
                            }
                            ConvertKind::Float => {
                                self.emit_float_convert_resolved(src, dst_ty, name, instructions)
                            }
                            ConvertKind::IntToFloat(op) => self.emit_int_to_float_convert_resolved(
                                op,
                                src,
                                dst_ty,
                                name,
                                instructions,
                            ),
                        };
                    }
                }
            } else if inst.opcode == "select" {
                if let Some(operands) = self.tir_inst_typed_operands(inst) {
                    if let [cond, t, f] = operands.as_slice() {
                        let (cond, t, f, name) = (cond.clone(), t.clone(), f.clone(), name.clone());
                        return self.emit_select_resolved(cond, t, f, name, instructions);
                    }
                }
            } else if inst.opcode == "fcmp" {
                // `fcmp` sources its predicate from `inst.cmp_predicate` (mapped to an `Op` by the same
                // `fcmp_predicate` the text path uses over the tir token) and its two operands from
                // `inst.operands`. Reaches the fail-visible unmigrated-opcode `Err` if either is absent,
                // or if the predicate token maps to no `Op` (the retry cascade owns the raw `.lines`
                // text walk).
                if let (Some(tok), Some(operands)) = (
                    inst.cmp_predicate.as_deref(),
                    self.tir_inst_typed_operands(inst),
                ) {
                    if let [lhs, rhs] = operands.as_slice() {
                        if let Some(pred) = fcmp_predicate(tok) {
                            let (lhs, rhs, name) = (lhs.clone(), rhs.clone(), name.clone());
                            return self.emit_fcmp_resolved(pred, lhs, rhs, name, instructions);
                        }
                    }
                }
            } else if inst.opcode == "icmp" {
                // `icmp` mirrors `fcmp`. BOTH the non-pointer and pointer forms are graph-driven: operands
                // and predicate ride the typed graph, and the pointer form's two unsupported-shape error
                // diagnostics (which embed the raw operand `rest` — BC fingerprints error strings) read that
                // `rest` from the diagnostics-only `TirInst.icmp_rest` carrier (byte-identical to the text
                // path's `rest`), not `inst.text`. `resolve_type` is the same resolution the text path
                // applies to `lhs.ty` before the pointer test. Falls through only if the carrier is absent.
                if let (Some(tok), Some(operands)) = (
                    inst.cmp_predicate.as_deref(),
                    self.tir_inst_typed_operands(inst),
                ) {
                    if let [lhs, rhs] = operands.as_slice() {
                        if let Some(pred) = icmp_predicate(tok) {
                            let operand_ty = self.resolve_type(&lhs.ty)?;
                            if !matches!(operand_ty, LlType::Ptr(_)) {
                                let (lhs, rhs, name) = (lhs.clone(), rhs.clone(), name.clone());
                                return self.emit_icmp_int_resolved(
                                    pred,
                                    lhs,
                                    rhs,
                                    operand_ty,
                                    name,
                                    instructions,
                                );
                            } else if let Some(rest) = &inst.icmp_rest {
                                let (lhs, rhs, name, rest) =
                                    (lhs.clone(), rhs.clone(), name.clone(), rest.clone());
                                return self.emit_icmp_ptr_resolved(
                                    pred,
                                    lhs,
                                    rhs,
                                    name,
                                    &rest,
                                    instructions,
                                );
                            }
                        }
                    }
                }
            } else if inst.opcode == "inttoptr" || inst.opcode == "ptrtoint" {
                // Both are single-operand pointer/int address casts: source from `inst.operands[0]`, dest
                // type from `inst.result_ty` (== the text path's `convert_dst_type`). Their type-shape
                // errors embed the resolved types (not `rest`), so migration is BC-safe.
                if let (Some(operands), Some(result_ty)) =
                    (self.tir_inst_typed_operands(inst), inst.result_ty.as_ref())
                {
                    if let [src] = operands.as_slice() {
                        let dst_ty = self.resolve_type(result_ty)?;
                        let (src, name) = (src.clone(), name.clone());
                        return if inst.opcode == "inttoptr" {
                            self.emit_inttoptr_resolved(src, dst_ty, name, instructions)
                        } else {
                            self.emit_ptrtoint_resolved(src, dst_ty, name, instructions)
                        };
                    }
                }
            } else if inst.opcode == "getelementptr" {
                // gep is already fully graph-driven on the default path: the graph-driven gep lowering builds the whole
                // `LlGep` from `tir_gep_source_types` (== `inst.gep_source_ty`) + `tir_typed_operands`
                // (== `inst.operands`; base = [0], indices = [1..]). Mirror that branch here off the inst;
                // when source_ty/operands aren't both present it reaches the fail-visible unmigrated-opcode
                // `Err` (the retry cascade owns the raw `.lines` text walk).
                if let (Some(source_ty), Some(ops)) = (
                    inst.gep_source_ty.as_ref(),
                    self.tir_inst_typed_operands(inst),
                ) {
                    if !ops.is_empty() {
                        let gep = LlGep {
                            source_ty: source_ty.clone(),
                            base: ops[0].clone(),
                            indices: ops[1..].to_vec(),
                        };
                        let name = name.clone();
                        self.emit_gep_result(&name, &gep, instructions)?;
                        return Ok(());
                    }
                }
            } else if inst.opcode == "load" {
                // `load` needs the pointer operand (`operands[0]`), the loaded type (`result_ty`), and the
                // alignment (`mem_align`) — the same three the text path sources from the carrier in the
                // graph-walk context. Proceed only when the operand resolves and `result_ty` is present,
                // else reach the fail-visible unmigrated-opcode `Err` (the retry cascade owns the raw
                // `.lines` text walk).
                if let (Some(operands), Some(result_ty)) =
                    (self.tir_inst_typed_operands(inst), inst.result_ty.as_ref())
                {
                    if let [ptr] = operands.as_slice() {
                        let result_ty = self.resolve_type(result_ty)?;
                        let load = LlLoad {
                            ptr: ptr.clone(),
                            result_ty: result_ty.clone(),
                            align: inst.mem_align,
                        };
                        let name = name.clone();
                        return self.emit_load_resolved(name, load, result_ty, instructions);
                    }
                }
            } else if inst.opcode == "extractelement" {
                // Both value operands (`[vector, index]`) are lowered by the graph; only the two
                // post-resolution SEMANTIC errors need the raw line, so read it from the diagnostics-only
                // `TirInst.diag_line` carrier (the same strip-commented/trimmed line the text path formats)
                // instead of re-lexing `text`. Falls through when the graph left an operand unresolved.
                if let (Some(operands), Some(line)) =
                    (self.tir_inst_typed_operands(inst), &inst.diag_line)
                {
                    if let [vector, idx] = operands.as_slice() {
                        let (vector, idx, name, line) =
                            (vector.clone(), idx.clone(), name.clone(), line.clone());
                        return self.emit_extractelement_resolved(
                            vector,
                            idx,
                            name,
                            &line,
                            instructions,
                        );
                    }
                }
            } else if inst.opcode == "insertelement" {
                // `[vector, object, index]` all come from the graph; the one-lane error embeds the line,
                // read from the diagnostics-only `TirInst.diag_line` carrier (same strip-commented line)
                // instead of re-lexing `text`. Falls through when the graph left an operand unresolved.
                if let (Some(operands), Some(line)) =
                    (self.tir_inst_typed_operands(inst), &inst.diag_line)
                {
                    if let [composite, object, idx] = operands.as_slice() {
                        let (composite, object, idx, name, line) = (
                            composite.clone(),
                            object.clone(),
                            idx.clone(),
                            name.clone(),
                            line.clone(),
                        );
                        return self.emit_insertelement_resolved(
                            composite,
                            object,
                            idx,
                            name,
                            &line,
                            instructions,
                        );
                    }
                }
            } else if inst.opcode == "shufflevector" {
                // The two LEADING operands (source vectors) are graph-lowered; the constant mask rides the
                // `TirInst.shuffle_mask` carrier (declared lane count + index values, the same parse), and
                // the residual `empty one-lane shuffle` diagnostic reads the strip-commented line from
                // `diag_line`. So the typed core re-lexes neither the mask nor the text. Reaches the
                // fail-visible unmigrated-opcode `Err` when an operand, the mask, or the diag line is
                // absent (the retry cascade owns the raw `.lines` text walk).
                if let (Some(a), Some(b), Some((declared, lanes)), Some(line)) = (
                    inst.operands
                        .first()
                        .and_then(crate::native::tir::TirOperand::as_typed_value),
                    inst.operands
                        .get(1)
                        .and_then(crate::native::tir::TirOperand::as_typed_value),
                    &inst.shuffle_mask,
                    &inst.diag_line,
                ) {
                    let (name, line, lanes) = (name.clone(), line.clone(), lanes.clone());
                    return self.emit_shufflevector_from_mask(
                        a,
                        b,
                        *declared,
                        lanes,
                        &line,
                        name,
                        instructions,
                    );
                }
            } else if inst.opcode == "extractvalue" {
                // The aggregate is a graph operand; the trailing constant indices ride the
                // `TirInst.aggregate_indices` carrier (parsed once at build: rhs after `%r = `, opcode
                // token dropped, then `split_top_level` + `parse_u32`), so the typed core needs no `line`.
                // Reaches the fail-visible unmigrated-opcode `Err` when either the operand or the index
                // list is unresolved (the retry cascade owns the raw `.lines` text walk).
                if let (Some(operands), Some(indices)) =
                    (self.tir_inst_typed_operands(inst), &inst.aggregate_indices)
                {
                    if let [composite] = operands.as_slice() {
                        let (composite, name, indices) =
                            (composite.clone(), name.clone(), indices.clone());
                        return self.emit_extractvalue_typed(
                            composite,
                            &indices,
                            name,
                            instructions,
                        );
                    }
                }
            } else if inst.opcode == "insertvalue" {
                // `[composite, object]` come from the graph; the trailing constant indices ride the
                // `TirInst.aggregate_indices` carrier (same parse: rhs after `%r = `, opcode token
                // dropped, then `split_top_level` + `parse_u32`), so the typed core needs no `line`.
                // Reaches the fail-visible unmigrated-opcode `Err` when either is unresolved (the retry
                // cascade owns the raw `.lines` text walk).
                if let (Some(operands), Some(indices)) =
                    (self.tir_inst_typed_operands(inst), &inst.aggregate_indices)
                {
                    if let [composite, object] = operands.as_slice() {
                        let (composite, object, name, indices) = (
                            composite.clone(),
                            object.clone(),
                            name.clone(),
                            indices.clone(),
                        );
                        return self.emit_insertvalue_typed(
                            composite,
                            object,
                            &indices,
                            name,
                            instructions,
                        );
                    }
                }
            } else if inst.opcode == "alloca" {
                // No graph value operands — the allocated type rides the tir as `inst.alloca_ty`
                // (parsed at build: rhs after `%r = `, opcode token dropped, then `parse_type`), so
                // dispatch straight on it, no `inst.text`. When the type did not parse at build
                // (unreachable in well-formed AIR), it reaches the fail-visible unmigrated-opcode `Err` below.
                if let Some(alloca_ty) = &inst.alloca_ty {
                    let name = name.clone();
                    return self.emit_alloca_typed(name, alloca_ty, instructions);
                }
            } else if inst.opcode == "phi" {
                // Fully graph-driven: the phi's parsed result type + `(value, predecessor)` pairs ride the
                // `TirInst.phi_incoming` carrier (built via the same `parse_phi` the text path runs), and
                // the incoming VALUES are re-sourced from the graph inside `emit_phi_resolved`. No
                // `inst.text` re-lex. Reaches the fail-visible unmigrated-opcode `Err` below only when
                // the carrier is absent (the phi operands did not parse at build — unreachable in well-formed AIR).
                if let Some((phi_ty, parsed_incoming)) = &inst.phi_incoming {
                    let (name, phi_ty, parsed_incoming) =
                        (name.clone(), phi_ty.clone(), parsed_incoming.clone());
                    return self.emit_phi_resolved(name, &phi_ty, parsed_incoming, instructions);
                }
            } else if inst.opcode == "bitcast" {
                // Fully graph-driven: the parsed source typed value + destination-type text ride the
                // `TirInst.bitcast` carrier (built via the same strip_comment + rhs after `%r = ` with
                // the opcode token dropped + split_once(" to ") + parse_typed_value the `bitcast` handler
                // ran), so no `inst.text` re-lex.
                // The pointer copy-prop side-tables (pointer_pointees/raw_offsets/gep_provenance/
                // selected_load_pointers) stay keyed on the source operand's name (from `src.value`)
                // exactly as before — no side-table ownership move needed. Reaches the fail-visible
                // unmigrated-opcode `Err` below only when the carrier is absent (a malformed bitcast —
                // unreachable in well-formed AIR).
                if let Some((src, dst_text)) = &inst.bitcast {
                    let (name, src, dst_text) = (name.clone(), src.clone(), dst_text.clone());
                    return self.emit_bitcast_resolved(src, &dst_text, name, instructions);
                }
            } else if matches!(inst.opcode.as_str(), "call" | "tail") {
                // Value-producing call. Sourced from the build-time `TirInst.call` carrier — the SAME
                // `parse_call(<ret> @callee(args))` the text entry runs (`resolve_call` strips the
                // `[tail ]call` keyword identically to `strip_call_prefix`), so no `inst.text` re-lex.
                // The argument VALUES are overlaid from the typed graph inside `emit_value_call_resolved`.
                // Reaches the fail-visible unmigrated-opcode `Err` below only when the carrier is absent
                // (an indirect call, which `parse_call` rejects — unreachable in well-formed AIR for a value call).
                if let Some(call) = &inst.call {
                    let (name, call) = (name.clone(), call.clone());
                    return self.emit_value_call_resolved(name, call, instructions);
                }
            }
        } else if inst.opcode == "store" {
            // The one result-LESS migrated family. Source `[value, pointer]` from `inst.operands` and
            // the alignment from `inst.mem_align` (the same alignment the retired text path parsed); when
            // the operands don't resolve it reaches the fail-visible unmigrated-opcode `Err` below.
            if let Some(operands) = self.tir_inst_typed_operands(inst) {
                if let [object, ptr] = operands.as_slice() {
                    let (object, ptr) = (object.clone(), ptr.clone());
                    return self.emit_store_resolved(object, ptr, inst.mem_align, instructions);
                }
            }
        } else if let Some(line) = &inst.void_call_line {
            // Result-LESS (void) call, or an ignored debug/lifetime marker. All off the
            // `TirInst.void_call_line` carrier (the strip-commented/trimmed line): drop the ignored markers
            // FIRST, then the INDIRECT function-group call (an indirect callee `%fp`, so `inst.call ==
            // None`) which is dropped as a no-op after materializing its callee/args — from
            // `strip_call_prefix(line)`, then finally drive a
            // DIRECT call off the `TirInst.call` carrier with its argument VALUES overlaid straight from
            // `inst.operands` (byte-identical to the `tir_call_queue` the text path pops).
            if is_ignored_call_line(line) {
                return Ok(());
            }
            if let Some(rest) = strip_call_prefix(line) {
                if self.drop_indirect_function_group_call(rest, instructions)? {
                    return Ok(());
                }
            }
            if let Some(call) = &inst.call {
                let (mut call, line) = (call.clone(), line.clone());
                self.apply_tir_inst_void_call_args(inst, &mut call);
                return self.emit_void_call_body(call, &line, instructions);
            }
        }
        // Every opcode reachable in the graph walk is now typed: this fall-through is measured DEAD
        // (0 hits / 16942 byte-baseline + 0 / 15336 banked, `APV_FALLTHROUGH_PROBE`, removed). Rather than
        // re-lex `inst.text` here, fail visibly — the same retired-substrate discipline as the ret/switch
        // `FromText` deletion: a hit (an opcode the typed dispatch left unmigrated, or a carrier absent on a
        // unreachable in well-formed AIR malformed line) routes to the retry cascade, which still owns the raw
        // `body_block.lines` text walk. This removes the last graph-walk `inst.text` reader.
        Err(format!(
            "native emitter: instruction not handled by the typed graph walk \
             (reason=graph_walk_unmigrated_opcode, opcode={})",
            inst.opcode
        ))
    }

    fn bind_inline_parameter(
        &mut self,
        name: &str,
        argument: TypedValue,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let resolved_ty = self.resolve_type(&argument.ty)?;
        let pointer_facts = if let LlType::Ptr(addrspace) = resolved_ty {
            let storage = self.pointer_storage_for(&argument.value, addrspace)?;
            let pointee = self.pointer_pointee_for_value(&argument.value)?;
            let nullness = match &argument.value {
                LlValue::Local(source) => self.pointer_nullness.get(source).copied(),
                LlValue::Global(_) | LlValue::Gep(_) => Some(self.const_bool(false)?),
                _ => None,
            };
            Some((addrspace, storage, pointee, nullness))
        } else {
            None
        };
        let argument_id = self.value_id_in(&argument.value, &argument.ty, instructions)?;
        let placeholder_id = self.fresh();
        self.values
            .insert(name.to_string(), (placeholder_id, resolved_ty.clone()));
        self.inline_parameter_substitutions
            .push((placeholder_id, argument_id));
        self.direct_param_values.insert(name.to_string());
        self.param_values.insert(name.to_string());

        if let Some((addrspace, storage, pointee, nullness)) = pointer_facts {
            self.pointer_storage.insert(name.to_string(), storage);
            if let Some(pointee) = pointee {
                self.pointer_pointees.insert(name.to_string(), pointee);
            }
            if let Some(nullness) = nullness {
                self.record_pointer_nullness(name.to_string(), nullness);
            }
            if self.raw_buffer_params.contains(name) {
                self.raw_offsets.insert(
                    name.to_string(),
                    RawBufferOffset::root(name.to_string(), addrspace),
                );
            }
        }
        Ok(())
    }

    /// The typed core of the `alloca` handler. `alloca` has no graph VALUE operands (its allocated type
    /// is a type, not an operand); the M-A5 graph walk passes the parsed allocated type from
    /// `TirInst.alloca_ty`, the text entry parses it from the line — either way this core resolves it
    /// against the module and applies the pointee/storage overrides keyed off the SSA result `name`.
    pub(in crate::native::emitter) fn emit_alloca_typed(
        &mut self,
        name: String,
        alloca_ty: &LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let pointee = self.resolve_type(alloca_ty)?;
        let effective_pointee = self
            .local_alloca_pointees
            .get(&name)
            .cloned()
            .map(|ty| self.resolve_type(&ty))
            .transpose()?
            .filter(|candidate| self.local_alloca_storage_compatible(&pointee, candidate))
            .unwrap_or_else(|| pointee.clone());
        let storage_pointee = function_storage_local_type(&effective_pointee);
        let ptr_type = self.ptr_type_id(StorageClass::Function, &storage_pointee)?;
        let result = self.result_id(&name, &LlType::Ptr(0))?;
        instructions.push(Self::inst(
            Op::Variable,
            Some(ptr_type),
            Some(result),
            vec![Operand::StorageClass(StorageClass::Function)],
        ));
        self.pointer_storage
            .insert(name.clone(), StorageClass::Function);
        self.pointer_pointees
            .insert(name.clone(), effective_pointee);
        // LLVM `alloca` always produces a non-null local pointer. Record that fact unconditionally:
        // an early typed helper splice can move a callee's `icmp eq ptr %parameter, null` onto the
        // caller's alloca name before the former function-parameter inference runs.
        let is_null = self.const_bool(false)?;
        self.record_pointer_nullness(name, is_null);
        Ok(())
    }

    /// The mask-resolved core: computes the result type from the a-operand + the carried mask lane count
    /// (replicating the shufflevector result-type computation's a-operand check + compose — a-operand
    /// element type × declared mask lane count; the mask-side parse already ran at
    /// build), then delegates to the shared tail. Byte-identical: the a-not-vector error embeds the operand
    /// type (not `line`), and `(result_ty, lanes)` are the exact shufflevector result-type outputs.
    pub(in crate::native::emitter) fn emit_shufflevector_from_mask(
        &mut self,
        a: TypedValue,
        b: TypedValue,
        declared_lanes: u32,
        lanes: Vec<u32>,
        line: &str,
        name: String,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let LlType::Vector(elem, _) = &a.ty else {
            return Err(format!(
                "native emitter: shufflevector first operand is not a vector: {:?}",
                a.ty
            ));
        };
        let result_ty = LlType::Vector(elem.clone(), declared_lanes);
        self.emit_shufflevector_with(a, b, result_ty, lanes, line, name, instructions)
    }

    /// The shared tail of the `shufflevector` handler: one-lane copy / >4-lane composite-construct /
    /// generic `OpVectorShuffle`, given the already-computed result type + lane indices. `line` is read
    /// only by the `empty one-lane shuffle` diagnostic (fed from `diag_line` on the typed path).
    pub(in crate::native::emitter) fn emit_shufflevector_with(
        &mut self,
        a: TypedValue,
        b: TypedValue,
        result_ty: LlType,
        lanes: Vec<u32>,
        line: &str,
        name: String,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        if let Some(elem) = self.one_lane_vector_elem(&result_ty)? {
            let Some(lane) = lanes.first().copied() else {
                return Err(format!("native emitter: empty one-lane shuffle: {line}"));
            };
            let result_type = self.type_id(&elem)?;
            let result = self.result_id(&name, &result_ty)?;
            let value = self.shuffled_lane_id(&a, &b, lane, &elem, instructions)?;
            instructions.push(Self::inst(
                Op::CopyObject,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(value)],
            ));
            return Ok(());
        }
        let result_type = self.type_id(&result_ty)?;
        let result = self.result_id(&name, &result_ty)?;
        let a_lanes = self.vector_lane_count(&a.ty)?;
        let b_lanes = self.vector_lane_count(&b.ty)?;
        if lanes.len() > 4 || a_lanes > 4 || b_lanes > 4 {
            let LlType::Vector(elem, _) = self.resolve_type(&result_ty)? else {
                return Err(format!(
                    "native emitter: shufflevector result is not a vector: {result_ty:?}"
                ));
            };
            let mut ops = Vec::with_capacity(lanes.len());
            for lane in lanes {
                ops.push(Operand::IdRef(self.shuffled_lane_id(
                    &a,
                    &b,
                    lane,
                    &elem,
                    instructions,
                )?));
            }
            instructions.push(Self::inst(
                Op::CompositeConstruct,
                Some(result_type),
                Some(result),
                ops,
            ));
            return Ok(());
        }
        let mut ops = vec![
            Operand::IdRef(self.value_id_in(&a.value, &a.ty, instructions)?),
            Operand::IdRef(self.value_id_in(&b.value, &b.ty, instructions)?),
        ];
        ops.extend(lanes.into_iter().map(Operand::LiteralBit32));
        instructions.push(Self::inst(
            Op::VectorShuffle,
            Some(result_type),
            Some(result),
            ops,
        ));
        Ok(())
    }

    /// The operand-resolved core of the `insertvalue` handler. Driven from `TirInst.operands`
    /// (`[composite, object]`) + the parsed trailing constant `indices` (from the
    /// `TirInst.aggregate_indices` carrier on the typed path, or the text entry's parse). No `line`.
    pub(in crate::native::emitter) fn emit_insertvalue_typed(
        &mut self,
        composite: TypedValue,
        object: TypedValue,
        indices: &[u32],
        name: String,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let result_ty = self.resolve_type(&composite.ty)?;
        let result_type = self.type_id(&result_ty)?;
        let result = self.result_id(&name, &result_ty)?;
        let mut ops = vec![
            Operand::IdRef(self.value_id_in(&object.value, &object.ty, instructions)?),
            Operand::IdRef(self.value_id_in(&composite.value, &composite.ty, instructions)?),
        ];
        ops.extend(indices.iter().copied().map(Operand::LiteralBit32));
        instructions.push(Self::inst(
            Op::CompositeInsert,
            Some(result_type),
            Some(result),
            ops,
        ));
        Ok(())
    }

    /// The operand-resolved core of the `extractvalue` handler. Driven from `TirInst.operands`
    /// (`[composite]`) + the parsed trailing constant `indices` (from the `TirInst.aggregate_indices`
    /// carrier on the typed path, or the text entry's parse). No `line`: `extract_value_type` embeds the
    /// resolved types, not the raw text, so the error bytes are text-independent.
    pub(in crate::native::emitter) fn emit_extractvalue_typed(
        &mut self,
        composite: TypedValue,
        indices: &[u32],
        name: String,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let result_ty = extract_value_type(&self.resolve_type(&composite.ty)?, indices)?;
        let result_type = self.type_id(&result_ty)?;
        let result = self.result_id(&name, &result_ty)?;
        let composite_id = self.value_id_in(&composite.value, &composite.ty, instructions)?;
        let mut ops = vec![Operand::IdRef(composite_id)];
        ops.extend(indices.iter().copied().map(Operand::LiteralBit32));
        instructions.push(Self::inst(
            Op::CompositeExtract,
            Some(result_type),
            Some(result),
            ops,
        ));
        Ok(())
    }

    /// The operand-resolved core of the `extractelement` handler. The graph walk drives it from
    /// `TirInst.operands` (`[vector, index]`, both typed operands the graph lowers). The two error strings
    /// embed the raw `line`, so the caller passes the strip-commented/trimmed instruction line (from
    /// `inst.diag_line`) unchanged.
    pub(in crate::native::emitter) fn emit_extractelement_resolved(
        &mut self,
        vector: TypedValue,
        idx: TypedValue,
        name: String,
        line: &str,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let LlType::Vector(elem, _) = self.resolve_type(&vector.ty)? else {
            if let Some(elem) = self.one_lane_vector_elem(&vector.ty)? {
                if const_index(Some(&idx)).is_some_and(|idx| idx != 0) {
                    return Err(format!(
                        "native emitter: one-lane extractelement index is not zero: {line}"
                    ));
                }
                let result_type = self.type_id(&elem)?;
                let result = self.result_id(&name, &elem)?;
                let vector_id = self.value_id_in(&vector.value, &vector.ty, instructions)?;
                instructions.push(Self::inst(
                    Op::CopyObject,
                    Some(result_type),
                    Some(result),
                    vec![Operand::IdRef(vector_id)],
                ));
                return Ok(());
            } else {
                return Err(format!(
                    "native emitter: extractelement from non-vector: {line}"
                ));
            }
        };
        let result_ty = *elem;
        let result_type = self.type_id(&result_ty)?;
        let result = self.result_id(&name, &result_ty)?;
        let vector_id = self.value_id_in(&vector.value, &vector.ty, instructions)?;
        if let Some(idx) = const_index(Some(&idx)) {
            instructions.push(Self::inst(
                Op::CompositeExtract,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(vector_id), Operand::LiteralBit32(idx)],
            ));
        } else {
            let idx_id = self.vector_index_id(&idx, instructions)?;
            instructions.push(Self::inst(
                Op::VectorExtractDynamic,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(vector_id), Operand::IdRef(idx_id)],
            ));
        }
        Ok(())
    }

    /// The operand-resolved core of the `insertelement` handler (see `emit_extractelement_resolved`).
    /// Driven from `TirInst.operands` (`[vector, object, index]`); the one-lane error embeds `line`.
    pub(in crate::native::emitter) fn emit_insertelement_resolved(
        &mut self,
        composite: TypedValue,
        object: TypedValue,
        idx: TypedValue,
        name: String,
        line: &str,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let result_ty = self.resolve_type(&composite.ty)?;
        let result_type = self.type_id(&result_ty)?;
        let result = self.result_id(&name, &result_ty)?;
        if self.one_lane_vector_elem(&composite.ty)?.is_some() {
            if const_index(Some(&idx)).is_some_and(|idx| idx != 0) {
                return Err(format!(
                    "native emitter: one-lane insertelement index is not zero: {line}"
                ));
            }
            let object_id = self.value_id_in(&object.value, &object.ty, instructions)?;
            instructions.push(Self::inst(
                Op::CopyObject,
                Some(result_type),
                Some(result),
                vec![Operand::IdRef(object_id)],
            ));
            return Ok(());
        }
        let vector_id = self.value_id_in(&composite.value, &composite.ty, instructions)?;
        let object_id = self.value_id_in(&object.value, &object.ty, instructions)?;
        if let Some(idx) = const_index(Some(&idx)) {
            instructions.push(Self::inst(
                Op::CompositeInsert,
                Some(result_type),
                Some(result),
                vec![
                    Operand::IdRef(object_id),
                    Operand::IdRef(vector_id),
                    Operand::LiteralBit32(idx),
                ],
            ));
        } else {
            let idx_id = self.vector_index_id(&idx, instructions)?;
            instructions.push(Self::inst(
                Op::VectorInsertDynamic,
                Some(result_type),
                Some(result),
                vec![
                    Operand::IdRef(vector_id),
                    Operand::IdRef(object_id),
                    Operand::IdRef(idx_id),
                ],
            ));
        }
        Ok(())
    }

    /// The operand-resolved core of the `inttoptr` handler. The graph walk drives it from
    /// `TirInst.operands[0]` (int source) + `TirInst.result_ty` (pointer dest). The type-shape errors
    /// embed the resolved `src_ty`/`dst_ty`, not any raw text.
    pub(in crate::native::emitter) fn emit_inttoptr_resolved(
        &mut self,
        src: TypedValue,
        dst_ty: LlType,
        name: String,
        _instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let src_ty = self.resolve_type(&src.ty)?;
        let LlType::Int(_) = src_ty else {
            return Err(format!(
                "native emitter: inttoptr source is not integer: {src_ty:?}"
            ));
        };
        let LlType::Ptr(addrspace) = dst_ty else {
            return Err(format!(
                "native emitter: inttoptr destination is not pointer: {dst_ty:?}"
            ));
        };
        let _ = self.value_id(&src.value, &src.ty)?;
        // Logical SPIR-V cannot materialize an integer as an address. Preserve a valid SSA pointer so
        // function-constant-dead command-buffer paths translate without claiming active GPU-address
        // semantics.
        self.define_unmodeled_pointer_value(&name, addrspace, &LlType::Int(8))?;
        Ok(())
    }

    /// The operand-resolved core of the `ptrtoint` handler (see `emit_inttoptr_resolved`). Driven from
    /// `TirInst.operands[0]` (pointer source) + `TirInst.result_ty` (integer dest).
    pub(in crate::native::emitter) fn emit_ptrtoint_resolved(
        &mut self,
        src: TypedValue,
        dst_ty: LlType,
        name: String,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let src_ty = self.resolve_type(&src.ty)?;
        let LlType::Ptr(_) = src_ty else {
            return Err(format!(
                "native emitter: ptrtoint source is not pointer: {src_ty:?}"
            ));
        };
        let LlType::Int(_) = dst_ty else {
            return Err(format!(
                "native emitter: ptrtoint destination is not integer: {dst_ty:?}"
            ));
        };
        if let LlValue::Local(src_name) = &src.value {
            if let Some((low, high)) = self.pointer_payload_words.get(src_name).copied() {
                let LlType::Int(bits) = dst_ty else {
                    unreachable!()
                };
                let result_type = self.type_id(&dst_ty)?;
                let result = self.result_id(&name, &dst_ty)?;
                if bits == 32 {
                    instructions.push(Self::inst(
                        Op::CopyObject,
                        Some(result_type),
                        Some(result),
                        vec![Operand::IdRef(low)],
                    ));
                    return Ok(());
                }
                if bits == 64 {
                    let low64 = self.fresh();
                    instructions.push(Self::inst(
                        Op::UConvert,
                        Some(result_type),
                        Some(low64),
                        vec![Operand::IdRef(low)],
                    ));
                    let high64 = self.fresh();
                    instructions.push(Self::inst(
                        Op::UConvert,
                        Some(result_type),
                        Some(high64),
                        vec![Operand::IdRef(high)],
                    ));
                    let shifted_high = self.fresh();
                    let shift = self.const_signed_int(64, 32)?;
                    instructions.push(Self::inst(
                        Op::ShiftLeftLogical,
                        Some(result_type),
                        Some(shifted_high),
                        vec![Operand::IdRef(high64), Operand::IdRef(shift)],
                    ));
                    instructions.push(Self::inst(
                        Op::BitwiseOr,
                        Some(result_type),
                        Some(result),
                        vec![Operand::IdRef(low64), Operand::IdRef(shifted_high)],
                    ));
                    return Ok(());
                }
                return Err(format!(
                    "native emitter: serialized pointer payload cannot convert to i{bits}"
                ));
            }
        }
        let _ = self.value_id(&src.value, &src.ty)?;
        // Logical SPIR-V has no portable pointer address value. Keep GPU-address arithmetic paths
        // structurally valid without claiming physical pointer semantics.
        let zero = self.const_null(&dst_ty)?;
        let result = self.result_id(&name, &dst_ty)?;
        instructions.push(Self::inst(
            Op::CopyObject,
            Some(self.type_id(&dst_ty)?),
            Some(result),
            vec![Operand::IdRef(zero)],
        ));
        Ok(())
    }

    /// The operand-resolved core of the `store` handler. `store` is result-LESS, so the graph walk reads its
    /// `(object, ptr)` straight off `TirInst.operands` (`[value, pointer]`) and its `align` off
    /// `inst.mem_align`, and calls here; the text-walk fallback parses the same from the line. Byte-
    /// identical by construction — `inst.operands`/`inst.mem_align` are lowered from the same store line.
    pub(in crate::native::emitter) fn emit_store_resolved(
        &mut self,
        object: TypedValue,
        ptr: TypedValue,
        align: Option<u64>,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        if let LlValue::Local(ptr_name) = &ptr.value {
            if let Some(vector_word) = self.vector_word_pointers.get(ptr_name).cloned() {
                let object_id = self.value_id_in(&object.value, &object.ty, instructions)?;
                self.emit_vector_word_store(&vector_word, &object.ty, object_id, instructions)?;
                return Ok(());
            }
            if let Some(selected) = self.selected_load_pointers.get(ptr_name).cloned() {
                let object_id = self.value_id_in(&object.value, &object.ty, instructions)?;
                self.emit_selected_pointer_store(
                    &object.ty,
                    object_id,
                    &selected,
                    align,
                    instructions,
                )?;
                return Ok(());
            }
            if let Some(raw) = self.raw_offsets.get(ptr_name).cloned() {
                if matches!(object.ty, LlType::Ptr(_)) {
                    if let LlValue::Local(obj_name) = &object.value {
                        if !self.pointer_payload_words.contains_key(obj_name)
                            && self.direct_param_indices.contains_key(obj_name)
                        {
                            let payload =
                                self.emit_direct_buffer_address_payload(obj_name, instructions)?;
                            self.pointer_payload_words.insert(obj_name.clone(), payload);
                        }
                        if let Some((low, high)) = self.pointer_payload_words.get(obj_name).copied()
                        {
                            self.emit_raw_word_store_for_access(&raw, 0, low, align, instructions)?;
                            self.emit_raw_word_store_for_access(
                                &raw,
                                4,
                                high,
                                align,
                                instructions,
                            )?;
                            return Ok(());
                        }
                    }
                }
                // BDA mode: storing a DEVICE pointer VALUE (`store ptr addrspace(1) %p, %dst`) is a
                // verbatim 8-byte copy of the loaded address. `%p` is tracked as a device-address
                // offset (no plain value id); materialize its 64-bit address and store it as an
                // `Int(64)` word at the target offset. Byte-exact (the exact loaded address bits).
                if self.bda_device_pointers && matches!(object.ty, LlType::Ptr(1)) {
                    if let LlValue::Local(obj_name) = &object.value {
                        if let Some(src) = self.raw_offsets.get(obj_name).cloned() {
                            if src.device_addr_base.is_some() {
                                let addr = self.materialize_device_address(&src, instructions)?;
                                self.emit_raw_store(
                                    &LlType::Int(64),
                                    addr,
                                    &raw,
                                    align,
                                    instructions,
                                )?;
                                return Ok(());
                            }
                        }
                    }
                }
                let object_id = self.value_id_in(&object.value, &object.ty, instructions)?;
                self.emit_raw_store(&object.ty, object_id, &raw, align, instructions)?;
                return Ok(());
            }
            // A store THROUGH a direct pointer select (`selected_pointers`). The load path lowers
            // this by loading both arms and selecting the loaded VALUES
            // (`emit_selected_pointer_direct_load`); the store analog is a per-arm read-modify-write
            // conditional store (`emit_selected_pointer_direct_store`). The default path otherwise
            // falls through to `value_id(%sel)` and dies with "unknown SSA value" (the select was
            // deferred into the side-table, never materialized as a plain value). When the arms are
            // not uniform-storage (the RMW form is invalid), the helper errors and the
            // "cannot store through reinterpreted pointer select" message routes to the raw retry.
            if let Some(selected) = self.selected_pointers.get(ptr_name).cloned() {
                let object_id = self.value_id_in(&object.value, &object.ty, instructions)?;
                self.emit_selected_pointer_direct_store(
                    &object.ty,
                    object_id,
                    &selected,
                    instructions,
                )?;
                return Ok(());
            }
        }
        if let Some(raw) = self.byte_array_reinterpret_raw_pointer(&ptr.value)? {
            let object_id = self.value_id_in(&object.value, &object.ty, instructions)?;
            self.emit_raw_store(&object.ty, object_id, &raw, align, instructions)?;
            return Ok(());
        }
        if self.emit_vector_root_store(&ptr, &object, instructions)? {
            return Ok(());
        }
        if matches!(&ptr.value, LlValue::Local(name) if self.unmodeled_pointers.contains(name)) {
            return Ok(());
        }
        if let Some(pointee) = self.pointer_pointee_for_value(&ptr.value)? {
            let pointee = self.resolve_type(&pointee)?;
            let object_ty = self.resolve_type(&object.ty)?;
            if !types_compatible(&pointee, &object_ty)
                && self.emit_pointer_to_local_field_store(&object, &ptr, &pointee, instructions)?
            {
                return Ok(());
            }
            if !types_compatible(&pointee, &object_ty)
                && self.emit_i64_to_i32_pair_struct_store(&object, &ptr, &pointee, instructions)?
            {
                return Ok(());
            }
            if !types_compatible(&pointee, &object_ty)
                && self.emit_aggregate_prefix_integer_reinterpret_store(
                    &object,
                    &ptr,
                    &object_ty,
                    &pointee,
                    instructions,
                )?
            {
                return Ok(());
            }
            if !types_compatible(&pointee, &object_ty)
                && self.emit_first_vector_aggregate_reinterpret_store(
                    &object,
                    &ptr,
                    &object_ty,
                    &pointee,
                    instructions,
                )?
            {
                return Ok(());
            }
            if !types_compatible(&pointee, &object_ty)
                && self.emit_first_scalar_aggregate_reinterpret_store(
                    &object,
                    &ptr,
                    &object_ty,
                    &pointee,
                    instructions,
                )?
            {
                return Ok(());
            }
            if !types_compatible(&pointee, &object_ty)
                && self.emit_zero_scalar_to_aggregate_store(
                    &object,
                    &ptr,
                    &object_ty,
                    &pointee,
                    instructions,
                )?
            {
                return Ok(());
            }
            if !types_compatible(&pointee, &object_ty)
                && self.emit_vector_to_scalar_stores(&object, &ptr, &pointee, instructions)?
            {
                return Ok(());
            }
            if !types_compatible(&pointee, &object_ty)
                && self.emit_workgroup_vector_chunk_stores(
                    &object,
                    &ptr,
                    &object_ty,
                    &pointee,
                    instructions,
                )?
            {
                return Ok(());
            }
            if !types_compatible(&pointee, &object_ty)
                && self.emit_widening_vector_store(&object, &ptr, &pointee, instructions)?
            {
                return Ok(());
            }
            if !types_compatible(&pointee, &object_ty)
                && self.emit_narrowing_vector_store(&object, &ptr, &pointee, instructions)?
            {
                return Ok(());
            }
            if !types_compatible(&pointee, &object_ty)
                && self.emit_same_width_scalar_store(
                    &object,
                    &ptr,
                    &object_ty,
                    &pointee,
                    instructions,
                )?
            {
                return Ok(());
            }
            if !types_compatible(&pointee, &object_ty)
                && self.emit_scalar_narrowing_store(
                    &object,
                    &ptr,
                    &object_ty,
                    &pointee,
                    instructions,
                )?
            {
                return Ok(());
            }
            if !types_compatible(&pointee, &object_ty)
                && self.emit_same_width_vector_reinterpret_store(
                    &object,
                    &ptr,
                    &object_ty,
                    &pointee,
                    instructions,
                )?
            {
                return Ok(());
            }
        }
        let ptr_id = self.value_id_in(&ptr.value, &ptr.ty, instructions)?;
        let object_id = self.value_id_in(&object.value, &object.ty, instructions)?;
        instructions.push(Self::inst(
            Op::Store,
            None,
            None,
            vec![Operand::IdRef(ptr_id), Operand::IdRef(object_id)],
        ));
        Ok(())
    }

    /// The resolved core of the void-call handler. Driven by the graph walk from the `TirInst.call`
    /// carrier with args overlaid from `inst.operands` (`apply_tir_inst_void_call_args`). `line` is read
    /// only by the `non-void call without result` diagnostic (fed from `TirInst.void_call_line`).
    pub(in crate::native::emitter) fn emit_void_call_body(
        &mut self,
        call: LlCall,
        line: &str,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        if self.emit_zero_memset(&call, instructions)? {
            return Ok(());
        }
        if self.emit_raw_memcpy(&call, instructions)? {
            return Ok(());
        }
        if self.emit_typed_memcpy(&call, instructions)? {
            return Ok(());
        }
        if self.drop_unmodeled_memcpy(&call) {
            return Ok(());
        }
        let result_ty = self.resolve_type(&call.ret)?;
        if result_ty != LlType::Void {
            return Err(format!(
                "native emitter: non-void call without result is not covered yet: {line}"
            ));
        }
        if self.emit_void_air_call(&call, instructions)? {
            return Ok(());
        }
        self.validate_call_args(&call, instructions)?;
        let result_type = self.type_id(&result_ty)?;
        let result = self.fresh();
        let callee = *self
            .function_ids
            .get(&call.callee)
            .ok_or_else(|| format!("native emitter: unknown callee @{}", call.callee))?;
        let mut ops = vec![Operand::IdRef(callee)];
        for arg in self.function_call_arg_ids(&call, instructions)? {
            ops.push(Operand::IdRef(arg));
        }
        instructions.push(Self::inst(
            Op::FunctionCall,
            Some(result_type),
            Some(result),
            ops,
        ));
        Ok(())
    }

    /// The operand-resolved core of the value-producing call handler. Driven either from the text entry's
    /// `parse_call` or, on the typed path, from `TirInst.call` — byte-identical, both are the same
    /// `parse_call` on the same `<ret> @callee(args)` remainder. The argument values are overlaid from the
    /// typed graph keyed by the SSA result `name`, then the special-case intrinsic/AIR emitters run before
    /// the generic call.
    pub(in crate::native::emitter) fn emit_value_call_resolved(
        &mut self,
        name: String,
        mut call: LlCall,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        // R3 graph-driven: source the call's ARGUMENT values from the typed graph (keyed by the
        // SSA result `name`; callee + return type stay from text). Done before the special-case
        // emitters so they consume the graph operands too. Byte-identical by tir's operand soundness.
        self.apply_tir_call_args(&name, &mut call);
        if self.emit_visible_function_table_placeholder_call(&call, &name, instructions)? {
            return Ok(());
        }
        if self.emit_llvm_fshl_i32_call(&call, &name, instructions)? {
            return Ok(());
        }
        if self.emit_llvm_cttz_i32_call(&call, &name, instructions)? {
            return Ok(());
        }
        if self.emit_llvm_abs_call(&call, &name, instructions)? {
            return Ok(());
        }
        if self.emit_llvm_usub_sat_call(&call, &name, instructions)? {
            return Ok(());
        }
        if self.emit_air_unsigned_sat_call(&call, &name, instructions)? {
            return Ok(());
        }
        if self.emit_air_unsigned_rhadd_call(&call, &name, instructions)? {
            return Ok(());
        }
        if self.emit_llvm_int_minmax_call(&call, &name, instructions)? {
            return Ok(());
        }
        if self.emit_imageblock_data_call(&call, &name, instructions)? {
            return Ok(());
        }
        if self.emit_value_air_call(&call, &name, instructions)? {
            return Ok(());
        }
        let result_ty = self.resolve_type(&call.ret)?;
        self.validate_call_args(&call, instructions)?;
        let result_type = self.type_id(&result_ty)?;
        let result = self.result_id(&name, &result_ty)?;
        let callee = *self
            .function_ids
            .get(&call.callee)
            .ok_or_else(|| format!("native emitter: unknown callee @{}", call.callee))?;
        let mut ops = vec![Operand::IdRef(callee)];
        for arg in self.function_call_arg_ids(&call, instructions)? {
            ops.push(Operand::IdRef(arg));
        }
        instructions.push(Self::inst(
            Op::FunctionCall,
            Some(result_type),
            Some(result),
            ops,
        ));
        Ok(())
    }

    /// The typed core of `bitcast`: everything past the `<src> to <dst>` split, driven by the parsed
    /// source typed value + destination-type TEXT. Driven straight off the `TirInst.bitcast` carrier
    /// by the graph walk. The pointer
    /// copy-prop side-tables are keyed on `src.value`'s local name, so this needs no `inst.text`.
    pub(in crate::native::emitter) fn emit_bitcast_resolved(
        &mut self,
        src: TypedValue,
        dst_text: &str,
        name: String,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let dst_ty = self.convert_dst_type(&name, dst_text)?;
        let src_ty = self.resolve_type(&src.ty)?;
        if matches!((&src_ty, &dst_ty), (LlType::Ptr(_), LlType::Ptr(_))) {
            if let LlValue::Local(src_name) = &src.value {
                if let Some(selected) = self.selected_load_pointers.get(src_name).cloned() {
                    self.selected_load_pointers.insert(name.clone(), selected);
                    if let Some(storage) = self.pointer_storage.get(src_name).copied() {
                        self.pointer_storage.insert(name.clone(), storage);
                    }
                    if let Some(is_null) = self.pointer_nullness.get(src_name).copied() {
                        self.record_pointer_nullness(name.clone(), is_null);
                    }
                    if let Some(pointee) = self.pointer_pointees.get(src_name).cloned() {
                        self.pointer_pointees.insert(name.clone(), pointee);
                    }
                    if self.param_values.contains(src_name) {
                        self.param_values.insert(name.clone());
                    }
                    return Ok(());
                }
                if let Some(raw) = self.raw_offsets.get(src_name).cloned() {
                    if let Some(storage) = self.pointer_storage.get(src_name).copied() {
                        self.pointer_storage.insert(name.clone(), storage);
                    }
                    if let Some(is_null) = self.pointer_nullness.get(src_name).copied() {
                        self.record_pointer_nullness(name.clone(), is_null);
                    }
                    if let Some(pointee) = self.pointer_pointees.get(src_name).cloned() {
                        self.pointer_pointees.insert(name.clone(), pointee);
                    }
                    if !self.pointer_phi_values.is_empty() {
                        self.materialize_raw_byte_index(&name, &raw, true, instructions)?;
                        if self.raw_pointer_word_aligned(&raw) {
                            self.materialize_raw_word_index(&name, &raw, true, instructions)?;
                        }
                    } else {
                        self.materialize_reserved_raw_byte_index(&name, &raw, instructions)?;
                        if self.raw_pointer_word_aligned(&raw) {
                            self.materialize_reserved_raw_word_index(&name, &raw, instructions)?;
                        }
                    }
                    self.raw_offsets.insert(name.clone(), raw);
                    let addrspace = match dst_ty {
                        LlType::Ptr(addrspace) => addrspace,
                        _ => {
                            return Err(
                                "native emitter: unmodeled-byte bitcast destination is not a \
                                 pointer"
                                    .into(),
                            )
                        }
                    };
                    self.define_unmodeled_byte_pointer_value(&name, addrspace)?;
                    if self.param_values.contains(src_name) {
                        self.param_values.insert(name.clone());
                    }
                    return Ok(());
                }
            }
        }
        let src_id = self.value_id(&src.value, &src.ty)?;
        let result = match (&src_ty, &dst_ty) {
            (LlType::Int(32), LlType::Vector(elem, 4)) if **elem == LlType::Int(8) => {
                self.emit_i32_to_v4i8(src_id, instructions)?
            }
            (a, b) if a == b && !self.values.contains_key(&name) => src_id,
            (a, b) if a == b => {
                let result_type = self.type_id(&dst_ty)?;
                let result = self.result_id(&name, &dst_ty)?;
                instructions.push(Self::inst(
                    Op::CopyObject,
                    Some(result_type),
                    Some(result),
                    vec![Operand::IdRef(src_id)],
                ));
                result
            }
            // A pointer→pointer bitcast that is not the same logical type (here, a cross-address-space
            // reinterpret — `LlType::Ptr` carries only the address space, so same-space pointers took
            // the `a == b` CopyObject arm above) would be an `OpBitcast` on a logical pointer, illegal
            // under Logical addressing. Never part of a valid module, so route to the failure-triggered
            // raw retry instead of emitting the rejected instruction. Floor-safe: a banked module never
            // contains this bitcast.
            (LlType::Ptr(_), LlType::Ptr(_)) => {
                return Err(format!(
                    "native emitter: cannot reinterpret pointer {name} across address spaces \
                     without a logical-pointer bitcast"
                ));
            }
            _ => {
                let result_type = self.type_id(&dst_ty)?;
                let result = self.result_id(&name, &dst_ty)?;
                instructions.push(Self::inst(
                    Op::Bitcast,
                    Some(result_type),
                    Some(result),
                    vec![Operand::IdRef(src_id)],
                ));
                result
            }
        };
        if matches!(dst_ty, LlType::Ptr(_)) {
            if let LlValue::Local(src_name) = &src.value {
                if let Some(storage) = self.pointer_storage.get(src_name).copied() {
                    self.pointer_storage.insert(name.clone(), storage);
                }
                if let Some(is_null) = self.pointer_nullness.get(src_name).copied() {
                    self.record_pointer_nullness(name.clone(), is_null);
                }
                if let Some(pointee) = self.pointer_pointees.get(src_name).cloned() {
                    self.pointer_pointees.insert(name.clone(), pointee);
                }
                if let Some(raw) = self.raw_offsets.get(src_name).cloned() {
                    self.raw_offsets.insert(name.clone(), raw);
                }
                if let Some(provenance) = self.gep_provenance.get(src_name).cloned() {
                    self.gep_provenance.insert(name.clone(), provenance);
                }
                if self.unmodeled_pointers.contains(src_name) {
                    self.unmodeled_pointers.insert(name.clone());
                }
                if self.param_values.contains(src_name) {
                    self.param_values.insert(name.clone());
                }
            }
        }
        self.values.insert(name, (result, dst_ty));
        Ok(())
    }

    /// The operand-resolved core of the `load` handler. The M-A4 graph walk sources the pointer from
    /// `TirInst.operands[0]`, the loaded type from `TirInst.result_ty`, and the alignment from
    /// `TirInst.mem_align` — the same three values the text path derives (its
    /// `load_pointer_operand`/`result_type_of`/`mem_align_of` read them from the tir carrier), so the two
    /// entries are byte-identical. `load` carries the resolved `ptr`+`align`; its
    /// `result_ty` field is unused here (the resolved `result_ty` is passed separately).
    pub(in crate::native::emitter) fn emit_load_resolved(
        &mut self,
        name: String,
        load: LlLoad,
        result_ty: LlType,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), String> {
        let result_type = self.type_id(&result_ty)?;
        let result = self.result_id(&name, &result_ty)?;
        if let LlValue::Local(ptr_name) = &load.ptr.value {
            if let Some(selected) = self.selected_load_pointers.get(ptr_name).cloned() {
                self.emit_selected_pointer_load(
                    result,
                    &result_ty,
                    &selected,
                    load.align,
                    instructions,
                )?;
                return Ok(());
            }
            if let Some(selected) = self.selected_pointers.get(ptr_name).cloned() {
                self.emit_selected_pointer_direct_load(
                    result,
                    &result_ty,
                    &selected,
                    load.align,
                    instructions,
                )?;
                return Ok(());
            }
            if let Some(vector_word) = self.vector_word_pointers.get(ptr_name).cloned() {
                self.emit_vector_word_load(result, &result_ty, &vector_word, instructions)?;
                return Ok(());
            }
            if let Some(raw) = self.raw_offsets.get(ptr_name).cloned() {
                // BDA mode: a DEVICE pointer (`addrspace(1)`) loaded from a buffer word is its real
                // 64-bit address, not a Private null placeholder. Load the 8 address bytes and
                // register `name` as a device-address-rooted offset, so a later GEP folds field/
                // element offsets into it and a store of `name` copies the 8 bytes verbatim (see
                // the store dispatch + `RawBufferOffset::device_addr_base`). The result VALUE id is
                // intentionally left undefined — a device pointer is only ever used AS a pointer
                // (GEP/store/deref), all routed through `raw_offsets`, never as a plain value.
                if self.bda_device_pointers {
                    if let LlType::Ptr(1) = result_ty {
                        let addr = self.fresh();
                        self.emit_raw_load(addr, &LlType::Int(64), &raw, load.align, instructions)?;
                        self.used_device_address = true;
                        let mut dev = RawBufferOffset::root(format!(".bda_{addr}"), 1);
                        dev.device_addr_base = Some(addr);
                        self.raw_offsets.insert(name.clone(), dev);
                        // A GEP through this pointer (`emit_gep_result`) reads `pointer_storage` for
                        // the base; a device-address pointer is a PhysicalStorageBuffer pointer. The
                        // leaf load/store routes through the device-address path regardless, so this
                        // only satisfies the storage lookup and propagates to GEP results harmlessly.
                        self.pointer_storage
                            .insert(name.clone(), StorageClass::PhysicalStorageBuffer);
                        return Ok(());
                    }
                }
                self.emit_raw_load(result, &result_ty, &raw, load.align, instructions)?;
                if let LlType::Ptr(_) = result_ty {
                    self.pointer_storage
                        .insert(name.clone(), StorageClass::Private);
                    self.pointer_pointees.insert(name.clone(), LlType::Int(8));
                    self.unmodeled_pointers.insert(name.clone());
                    let needs_payload = self.pointer_payload_values.contains(&name);
                    let is_null = if self.raw_pointer_word_aligned(&raw) || needs_payload {
                        let (payload, is_null) =
                            self.emit_raw_pointer_payload(&raw, 0, load.align, instructions)?;
                        self.pointer_payload_words.insert(name.clone(), payload);
                        is_null
                    } else {
                        self.const_bool(false)?
                    };
                    self.record_pointer_nullness(name.clone(), is_null);
                }
                return Ok(());
            }
            if let Some(raw) = self.byte_array_reinterpret_raw_pointer(&load.ptr.value)? {
                self.emit_raw_load(result, &result_ty, &raw, load.align, instructions)?;
                return Ok(());
            }
            if self.unmodeled_pointers.contains(ptr_name) {
                if let LlType::Ptr(addrspace) = result_ty {
                    self.define_unmodeled_byte_pointer_value(&name, addrspace)?;
                } else {
                    let zero = self.const_null(&result_ty)?;
                    instructions.push(Self::inst(
                        Op::CopyObject,
                        Some(result_type),
                        Some(result),
                        vec![Operand::IdRef(zero)],
                    ));
                }
                return Ok(());
            }
        }
        if self.emit_pointer_from_local_dynamic_field_load(
            &name,
            result,
            &result_ty,
            &load.ptr,
            instructions,
        )? {
            return Ok(());
        }
        if let Some(pointee) = self.pointer_pointee_for_value(&load.ptr.value)? {
            let pointee = self.resolve_type(&pointee)?;
            if self.emit_pointer_from_local_field_load(
                &name,
                result,
                &result_ty,
                &load.ptr,
                &pointee,
                instructions,
            )? {
                return Ok(());
            }
        }
        if self.emit_vector_root_load(result, &result_ty, &load.ptr, instructions)? {
            return Ok(());
        }
        let ptr = self.value_id_in(&load.ptr.value, &load.ptr.ty, instructions)?;
        if let Some(pointee) = self.pointer_pointee_for_value(&load.ptr.value)? {
            let pointee = self.resolve_type(&pointee)?;
            if !types_compatible(&pointee, &result_ty) {
                if self.emit_i32_pair_struct_to_i64_load(
                    result,
                    &pointee,
                    &result_ty,
                    ptr,
                    instructions,
                )? {
                    return Ok(());
                }
                if self.emit_aggregate_prefix_integer_reinterpret_load(
                    result,
                    &pointee,
                    &result_ty,
                    &load.ptr,
                    ptr,
                    instructions,
                )? {
                    return Ok(());
                }
                if self.emit_byte_array_integer_reinterpret_load(
                    result,
                    &pointee,
                    &result_ty,
                    &load.ptr,
                    ptr,
                    instructions,
                )? {
                    return Ok(());
                }
                if self.emit_first_pointer_aggregate_reinterpret_load(
                    &name,
                    result,
                    &pointee,
                    &result_ty,
                    &load.ptr,
                    ptr,
                    instructions,
                )? {
                    return Ok(());
                }
                if self.emit_first_vector_aggregate_reinterpret_load(
                    result,
                    &pointee,
                    &result_ty,
                    &load.ptr,
                    ptr,
                    instructions,
                )? {
                    return Ok(());
                }
                if self.emit_first_scalar_aggregate_reinterpret_load(
                    result,
                    &pointee,
                    &result_ty,
                    &load.ptr,
                    ptr,
                    instructions,
                )? {
                    return Ok(());
                }
                let pointee_bits = bitcast_width(&pointee).ok_or_else(|| {
                    format!(
                        "native emitter: cannot reinterpret load `{name}` from {:?} with non-bitcastable pointee {pointee:?} to {result_ty:?}",
                        load.ptr.value
                    )
                })?;
                let result_bits = bitcast_width(&result_ty).ok_or_else(|| {
                    format!(
                        "native emitter: cannot reinterpret load to non-bitcastable result {result_ty:?}"
                    )
                })?;
                // A load through a `uchar` (byte-view) pointer that was re-typed to a wider scalar
                // or vector by `emit_byte_view_scalar_gep`: the emitted pointer is a legal `uchar`
                // StorageBuffer/Workgroup pointer at the low byte of the value, so we cannot
                // `OpLoad` the wider type directly. Assemble it from the byte-addressed data
                // (structural: pointee `Int(8)` + bitcastable scalar/vector result).
                if pointee == LlType::Int(8) {
                    let asp = match self.resolve_type(&load.ptr.ty)? {
                        LlType::Ptr(asp) => asp,
                        _ => 1,
                    };
                    let storage = self.pointer_storage_for(&load.ptr.value, asp)?;
                    if storage == StorageClass::Private
                        && self.emit_private_scalar_load_from_byte_pointer(
                            result,
                            &result_ty,
                            &load.ptr.value,
                            instructions,
                        )?
                    {
                        return Ok(());
                    }
                    if self.emit_scalar_load_from_byte_pointer(
                        result,
                        &result_ty,
                        storage,
                        ptr,
                        instructions,
                    )? {
                        return Ok(());
                    }
                }
                if pointee_bits != result_bits {
                    if self.emit_narrowing_vector_load(
                        result,
                        &pointee,
                        &result_ty,
                        ptr,
                        instructions,
                    )? {
                        return Ok(());
                    }
                    if self.emit_widening_vector_load(
                        result,
                        &pointee,
                        &result_ty,
                        ptr,
                        instructions,
                    )? {
                        return Ok(());
                    }
                    if self.emit_scalar_to_vector_load(
                        result,
                        &pointee,
                        &result_ty,
                        &load.ptr,
                        ptr,
                        instructions,
                    )? {
                        return Ok(());
                    }
                    if self.emit_scalar_word_to_subword_vector_load(
                        result,
                        &pointee,
                        &result_ty,
                        &load.ptr,
                        ptr,
                        instructions,
                    )? {
                        return Ok(());
                    }
                    if self.emit_scalar_to_wider_vector_load(
                        result,
                        &pointee,
                        &result_ty,
                        &load.ptr,
                        ptr,
                        instructions,
                    )? {
                        return Ok(());
                    }
                    if self.emit_scalar_from_vector_load(
                        result,
                        &pointee,
                        &result_ty,
                        ptr,
                        instructions,
                    )? {
                        return Ok(());
                    }
                    if self.emit_scalar_narrowing_load(
                        result,
                        &pointee,
                        &result_ty,
                        ptr,
                        instructions,
                    )? {
                        return Ok(());
                    }
                    if pointee == LlType::Int(8) {
                        // Reinterpreting the i8 (byte) pointer to the wider result type would need an
                        // `OpBitcast` on a logical pointer (retyping the pointee), illegal under Logical
                        // addressing. Such a module never validates, so route to the failure-triggered
                        // raw retry — which models the buffer as a byte/word RuntimeArray and forms the
                        // load by byte offset, no pointer bitcast. Floor-safe: a banked module never
                        // contains this bitcast, so it never reaches here.
                        return Err(format!(
                            "native emitter: cannot reinterpret load of byte pointer to {result_ty:?} without a logical-pointer bitcast"
                        ));
                    }
                    return Err(format!(
                        "native emitter: reinterpret load bit width mismatch {pointee:?} ({pointee_bits}) vs {result_ty:?} ({result_bits})"
                    ));
                }
                let pointee_type = self.type_id(&pointee)?;
                let loaded = self.fresh();
                instructions.push(Self::inst(
                    Op::Load,
                    Some(pointee_type),
                    Some(loaded),
                    vec![Operand::IdRef(ptr)],
                ));
                instructions.push(Self::inst(
                    Op::Bitcast,
                    Some(result_type),
                    Some(result),
                    vec![Operand::IdRef(loaded)],
                ));
                return Ok(());
            }
        }
        instructions.push(Self::inst(
            Op::Load,
            Some(result_type),
            Some(result),
            vec![Operand::IdRef(ptr)],
        ));
        if let LlType::Ptr(addrspace) = result_ty {
            self.pointer_storage
                .insert(name.clone(), llvm_pointer_storage(addrspace)?);
        }
        Ok(())
    }
}
