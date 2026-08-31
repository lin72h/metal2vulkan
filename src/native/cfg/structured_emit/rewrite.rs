//! The structurizer's CFG-rewrite discipline.

use super::BodyBlock;

/// Run a CFG rewrite that is allowed to decline, guaranteeing the caller's block list is untouched
/// when it does.
///
/// Every `synth_*`/`split_*` primitive in this module advertises the same contract — "anything
/// outside this scope returns `None`, leaving the loop unstructured so `structured_plan` falls
/// back" — and callers depend on it. [`forest_loop_merges`](super::forest_loop_merges) only
/// invalidates its cached [`LoopForest`](super::LoopForest) when a primitive reports `Some`, and a
/// declined loop's blocks are handed straight to the next construction attempt.
///
/// Honoring that by hand means proving, for every `?` in a rewrite body, that it cannot fire after
/// the first edit. That is not a property a reviewer can check by reading, and not one a later edit
/// preserves. Three primitives here had it wrong: `split_two_target_critical_edges` could redirect a
/// source terminator and then bail, leaving an edge pointing at a block it never inserted, and
/// `synth_multi_latch_continue` could rewrite a header's phis onto a latch it then failed to build.
///
/// So the rewrite runs against a private copy and the caller's list is replaced only on `Some`.
/// [`BodyBlock`] holds its instruction payload behind an `Arc`, so that copy is a vector of
/// (name, role, pointer) and only the blocks the rewrite actually edits are deep-copied — the same
/// copy-on-write `typed_mut` already performs on every edit.
///
/// A name counter borrowed by `rewrite` is deliberately *not* rolled back: names a declined attempt
/// consumed stay consumed, so every synthesized name remains unique across attempts.
pub(in crate::native) fn atomic_rewrite<T>(
    blocks: &mut Vec<BodyBlock>,
    rewrite: impl FnOnce(&mut Vec<BodyBlock>) -> Option<T>,
) -> Option<T> {
    let mut staged = blocks.clone();
    let outcome = rewrite(&mut staged)?;
    *blocks = staged;
    Some(outcome)
}
