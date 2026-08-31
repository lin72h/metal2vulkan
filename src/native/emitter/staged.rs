//! The emitter's decline discipline.

use crate::spirv_module::Instruction;

/// Run an emit step that is allowed to decline, guaranteeing that a decline emits nothing.
///
/// The emitter dispatches most lowerings as a chain of responsibility: a handler takes the caller's
/// instruction stream, returns `Ok(true)` for "I emitted it" and `Ok(false)` for "not mine, try the
/// next one". `Ok(false)` is therefore a claim about the stream as much as about the handler, and
/// the next handler in the chain — or [`drop_unmodeled_memcpy`](super::Emitter::drop_unmodeled_memcpy),
/// which discards the call entirely — acts on it.
///
/// A handler that walks an aggregate cannot know it will decline until it reaches the member that
/// defeats it, by which point it has emitted the members before it. `emit_prefix_struct_memcpy` did
/// exactly that: it copies field 0, finds field 1 running past the requested byte count, and reports
/// that it did nothing — leaving the destination's first field written and the rest not, in a module
/// that then validates cleanly. `emit_raw_to_typed_struct_memcpy` already avoided this by building
/// into a scratch vector and splicing it in only on success ("nothing partial may be committed to the
/// real stream"); this is that pattern, named, so it is applied rather than remembered.
///
/// Ids minted by a declined attempt stay minted. They are never referenced, and skipping a few keeps
/// every id unique across attempts.
pub(in crate::native::emitter) fn staged_emit(
    instructions: &mut Vec<Instruction>,
    emit: impl FnOnce(&mut Vec<Instruction>) -> Result<bool, String>,
) -> Result<bool, String> {
    let mut staged = Vec::new();
    if emit(&mut staged)? {
        instructions.append(&mut staged);
        Ok(true)
    } else {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::Operand;
    use spirv::Op;

    fn nop() -> Instruction {
        Instruction::new(Op::Nop, None, None, vec![Operand::LiteralBit32(0)])
    }

    /// The whole point: a handler that emitted part of an aggregate before finding the member that
    /// defeats it must not leave that part in the caller's stream when it reports `Ok(false)`.
    #[test]
    fn a_declined_step_leaves_the_stream_it_was_given() {
        let mut instructions = vec![nop()];
        let declined = staged_emit(&mut instructions, |staged| {
            staged.push(nop());
            staged.push(nop());
            Ok(false)
        });
        assert_eq!(declined, Ok(false));
        assert_eq!(instructions.len(), 1, "a decline must emit nothing");
    }

    #[test]
    fn a_handled_step_appends_in_order() {
        let mut instructions = vec![nop()];
        let handled = staged_emit(&mut instructions, |staged| {
            staged.push(nop());
            staged.push(nop());
            Ok(true)
        });
        assert_eq!(handled, Ok(true));
        assert_eq!(instructions.len(), 3);
    }

    /// An `Err` aborts the whole translation attempt, so the stream is never read again; staging it
    /// either way would only cost a copy on a path that is already dead.
    #[test]
    fn an_erroring_step_reports_the_error() {
        let mut instructions = Vec::new();
        let failed = staged_emit(&mut instructions, |staged| {
            staged.push(nop());
            Err("no".to_string())
        });
        assert_eq!(failed, Err("no".to_string()));
    }
}
