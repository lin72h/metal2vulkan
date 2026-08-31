//! Byte-neutral responsibility split of the former monolith impl; see the parent module.

use super::*;

impl Emitter {
    /// The typed operands of a `TirInst` read straight off the instruction, or `None` if any operand is
    /// `Unresolved`. This direct carrier is the sole M-A4 graph-walk operand source; emission does not
    /// retain a parallel result-keyed clone.
    pub(in crate::native::emitter) fn tir_inst_typed_operands(
        &self,
        inst: &crate::native::tir::TirInst,
    ) -> Option<Vec<TypedValue>> {
        inst.operands
            .iter()
            .map(crate::native::tir::TirOperand::as_typed_value)
            .collect()
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

    /// Source a direct call's argument values straight off `TirInst.operands` in the graph walk.
    /// Byte-identical to parsing by tir's call-operand soundness: the same `overlay_call_args` runs on
    /// the same operands for result-bearing and result-less calls.
    /// Falls back to the parsed args on any count mismatch or unresolved operand (the `METAL2VULKAN_TIR_ONLY`
    /// gate fires). The former text-walk analogue (`apply_tir_void_call_args`, popping a per-block
    /// `tir_call_queue`) is retired — the text-walk fallback just uses the parsed args.
    pub(in crate::native::emitter) fn apply_tir_inst_call_args(
        &self,
        inst: &crate::native::tir::TirInst,
        diagnostic_name: &str,
        call: &mut LlCall,
    ) {
        match self.tir_inst_typed_operands(inst) {
            Some(ops) if Self::overlay_call_args(&mut call.args, &ops) => {}
            _ => Self::tir_only_gate(diagnostic_name, "call"),
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
