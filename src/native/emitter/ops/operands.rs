//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    /// The instruction's operands as typed values resolved by the typed SSA IR (`tir`), or `None` if the
    /// graph did not resolve every operand (the caller then falls back to string parsing). This is the
    /// single entry point for R3 graph-driven emission: an opcode is "migrated" once its emitter reads
    /// its operands from here instead of re-lexing the instruction text.
    pub(in crate::native::emitter) fn tir_typed_operands(
        &self,
        name: &str,
    ) -> Option<Vec<TypedValue>> {
        self.tir_operands
            .get(name)?
            .iter()
            .map(crate::native::tir::TirOperand::as_typed_value)
            .collect()
    }

    /// The typed operands of a `TirInst` read STRAIGHT off the instruction (not via the `tir_operands`
    /// side-table keyed by result name), or `None` if any operand is `Unresolved`. This is the M-A4
    /// graph-walk operand source — byte-identical to `tir_typed_operands(name)` for a result-bearing
    /// instruction, since `emit_function` populates that map from this same `inst.operands`.
    pub(in crate::native::emitter) fn tir_inst_typed_operands(
        &self,
        inst: &crate::native::tir::TirInst,
    ) -> Option<Vec<TypedValue>> {
        inst.operands
            .iter()
            .map(crate::native::tir::TirOperand::as_typed_value)
            .collect()
    }

    /// The phi's incoming `(value, predecessor-label)` pairs, with the VALUES sourced from the typed
    /// graph. `phi` carries an SSA result, so its value operands live in `tir_operands` (one per
    /// incoming, in source order — `resolve_phi_operands` keeps the first field of each `[ val, pred ]`
    /// chunk and drops the predecessor label). The predecessor LABELS are control-flow edges that exist
    /// only in the instruction text, so they always come from `parsed`. When the graph resolved exactly
    /// one operand per incoming, each parsed value is replaced by the graph's; on any count mismatch or
    /// unresolved operand the parsed values stand (and the `METAL2VULKAN_TIR_ONLY` gate fires). tir's phi
    /// operands are proven sound broadly — including the synthetic `%metal2vulkan.lmerge.*` merge phis the
    /// structurizer adds — so this is byte-identical to parsing; it routes the value sourcing through
    /// the typed graph rather than the parsed text.
    pub(in crate::native::emitter) fn phi_incoming_values(
        &self,
        name: &str,
        parsed: Vec<(LlValue, String)>,
    ) -> Vec<(LlValue, String)> {
        if let Some(operands) = self.tir_typed_operands(name) {
            if operands.len() == parsed.len() {
                return operands
                    .into_iter()
                    .zip(parsed)
                    .map(|(op, (_, label))| (op.value, label))
                    .collect();
            }
        }
        Self::tir_only_gate(name, "phi");
        parsed
    }

    /// Overlay typed-graph `operands` onto a direct call's parsed argument values, returning whether it
    /// applied. The callee and return type are NOT operands (the graph does not lower them), so the
    /// caller keeps them from the instruction text; only the value arguments are sourced from the
    /// graph. tir resolves a direct call's args with the same `parse_typed_value` per `<ty> <val>`
    /// chunk `parse_call` uses (`resolve_call_operands`), so the overlaid values are byte-identical to
    /// the parsed ones — this only routes the value sourcing through the graph. Applies only when the
    /// graph resolved exactly one operand per arg (a count mismatch leaves the parsed args untouched).
    pub(in crate::native::emitter) fn overlay_call_args(
        args: &mut [TypedValue],
        operands: &[TypedValue],
    ) -> bool {
        if operands.len() != args.len() {
            return false;
        }
        for (arg, op) in args.iter_mut().zip(operands) {
            *arg = op.clone();
        }
        true
    }

    /// Source a result-bearing direct call's argument values from the typed graph (keyed by the SSA
    /// result `name`, as the straight-line ops are), falling back to the parsed args on any count
    /// mismatch or unresolved operand. Byte-identical to parsing by tir's call-operand soundness.
    pub(in crate::native::emitter) fn apply_tir_call_args(&self, name: &str, call: &mut LlCall) {
        match self.tir_typed_operands(name) {
            Some(ops) if Self::overlay_call_args(&mut call.args, &ops) => {}
            _ => Self::tir_only_gate(name, "call"),
        }
    }

    /// Source a result-LESS (void) direct call's argument values STRAIGHT off the `TirInst.operands` in the
    /// graph walk (a void call has no SSA result, so it cannot be keyed in `tir_operands`). Byte-identical
    /// to parsing by tir's call-operand soundness: the same `overlay_call_args` runs on the same operands.
    /// Falls back to the parsed args on any count mismatch or unresolved operand (the `METAL2VULKAN_TIR_ONLY`
    /// gate fires). The former text-walk analogue (`apply_tir_void_call_args`, popping a per-block
    /// `tir_call_queue`) is retired — the text-walk fallback just uses the parsed args.
    pub(in crate::native::emitter) fn apply_tir_inst_void_call_args(
        &self,
        inst: &crate::native::tir::TirInst,
        call: &mut LlCall,
    ) {
        match self.tir_inst_typed_operands(inst) {
            Some(ops) if Self::overlay_call_args(&mut call.args, &ops) => {}
            _ => Self::tir_only_gate("void", "call"),
        }
    }

    /// R3 migration gate: `METAL2VULKAN_TIR_ONLY=1` asserts the typed graph drove an instruction (never set in
    /// production). A migrated emitter calls this on its string-parse fallback so a future session can
    /// prove each opcode class is actually graph-driven — the migrated classes currently hit 0 fallbacks
    /// across broad regression sets and the lib test suite.
    pub(in crate::native::emitter) fn tir_only_gate(name: &str, opcode_class: &str) {
        if crate::env_vars::tir_only() {
            panic!("METAL2VULKAN_TIR_ONLY: {opcode_class} {name} fell back to string parse");
        }
    }

    /// The resolved DESTINATION type of a `<conv> <srcty> <v> to <dstty>` named `name`: from the typed
    /// SSA graph (tir's resolved result type) when available, else parsed from `dst_text`. tir stores
    /// `parse_type(dst)` for the conversion opcodes (the same the text path parses), so
    /// `resolve_type` of either is byte-identical — this retires the destination-type re-lex for the
    /// conversion family. Only the type *sourcing* moves; the conversion dispatch is unchanged.
    pub(in crate::native::emitter) fn convert_dst_type(
        &self,
        name: &str,
        dst_text: &str,
    ) -> Result<LlType, String> {
        if let Some(ty) = self.tir_result_types.get(name) {
            return self.resolve_type(ty);
        }
        // No `tir_only_gate` here: the destination is a re-derivable TYPE, not an operand value, so the
        // text fallback is provably byte-identical (`resolve_type(parse_type(dst))` either way) for the
        // rare conversions tir leaves untyped (`result_ty == None`). Gating those would crash them under
        // METAL2VULKAN_TIR_ONLY without any correctness benefit; a silent, identical fallback is correct.
        self.resolve_type(&parse_type(dst_text)?)
    }
}
