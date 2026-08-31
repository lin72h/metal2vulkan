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

/// A block list that counts the rewrites which have actually landed on it, so an analysis derived
/// from it can say whether it still describes the graph.
///
/// The structurizer's transform loops interleave whole-function analyses (a [`LoopForest`], a
/// selection-merge map) with rewrites that may or may not fire. Caching an analysis across those
/// rewrites is the difference between one analysis per loop header and one per graph edit, but the
/// invalidation was hand-written: `forest_loop_merges` carried eight separate `cache = None`
/// assignments, one per mutation site, each of which a later edit has to remember to add. A missing
/// one is invisible — the stale forest still answers every query, just about the previous graph.
///
/// Here the revision is the cache key and only this type can advance it, so an analysis is reused
/// exactly when the block list has not changed since it was derived and never otherwise.
///
/// [`LoopForest`]: super::LoopForest
pub(in crate::native) struct RewriteBlocks {
    blocks: Vec<BodyBlock>,
    revision: u64,
}

impl RewriteBlocks {
    pub(in crate::native) fn new(blocks: Vec<BodyBlock>) -> Self {
        Self {
            blocks,
            revision: 0,
        }
    }

    pub(in crate::native) fn get(&self) -> &[BodyBlock] {
        &self.blocks
    }

    /// The number of rewrites that have landed. Only ever compared for equality.
    pub(in crate::native) fn revision(&self) -> u64 {
        self.revision
    }

    /// Run a rewrite that may decline, advancing the revision only when it commits.
    ///
    /// A decline must leave the list untouched — that is the contract [`atomic_rewrite`] gives every
    /// primitive here — so holding the revision across one keeps a cached analysis valid, which is
    /// the whole point of distinguishing a decline from a no-op. Debug builds check it rather than
    /// assume it, so a primitive that stops honoring the contract fails the test suite here instead
    /// of silently feeding a stale forest to the next loop header.
    pub(in crate::native) fn rewrite<T>(
        &mut self,
        rewrite: impl FnOnce(&mut Vec<BodyBlock>) -> Option<T>,
    ) -> Option<T> {
        #[cfg(debug_assertions)]
        let before = decline_witness(&self.blocks);
        match rewrite(&mut self.blocks) {
            Some(outcome) => {
                self.revision += 1;
                Some(outcome)
            }
            None => {
                #[cfg(debug_assertions)]
                assert!(
                    before == decline_witness(&self.blocks),
                    "a declined rewrite edited the block list; wrap it in atomic_rewrite"
                );
                None
            }
        }
    }

    /// Mutable access for a rewrite that does not report whether it fired. The revision advances
    /// unconditionally, so a cached analysis is dropped whether or not the list actually changed.
    pub(in crate::native) fn edit(&mut self) -> &mut Vec<BodyBlock> {
        self.revision += 1;
        &mut self.blocks
    }

    pub(in crate::native) fn into_inner(self) -> Vec<BodyBlock> {
        self.blocks
    }
}

/// What [`RewriteBlocks::rewrite`] compares to catch a decline that edited the list: a hash of the
/// block names and successor labels, in order. That is the substrate the corrupting cases here
/// damage — an edge redirected to a block that was never inserted, or a split block left behind —
/// and it is cheap enough to take on every declined rewrite in a debug build, which a full carrier
/// comparison is not.
#[cfg(debug_assertions)]
fn decline_witness(blocks: &[BodyBlock]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    blocks.len().hash(&mut hasher);
    for block in blocks {
        block.name.hash(&mut hasher);
        super::block_successors(block).hash(&mut hasher);
    }
    hasher.finish()
}

/// A whole-function analysis remembered alongside the [`RewriteBlocks`] revision it was derived
/// from, so the derivation runs once per graph state instead of once per query.
pub(in crate::native) struct AnalysisCache<T> {
    entry: Option<(u64, T)>,
}

impl<T> Default for AnalysisCache<T> {
    fn default() -> Self {
        Self { entry: None }
    }
}

impl<T> AnalysisCache<T> {
    /// The analysis for `blocks` as it stands now, deriving it only if the cached one predates the
    /// current revision.
    pub(in crate::native) fn get<'cache>(
        &'cache mut self,
        blocks: &RewriteBlocks,
        derive: impl FnOnce(&[BodyBlock]) -> T,
    ) -> &'cache T {
        let revision = blocks.revision();
        if !matches!(&self.entry, Some((cached, _)) if *cached == revision) {
            self.entry = Some((revision, derive(blocks.get())));
        }
        &self.entry.as_ref().expect("just derived").1
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::bb;
    use super::*;

    fn one_block() -> Vec<BodyBlock> {
        vec![bb("%entry", &["ret void"])]
    }

    /// The property the whole caching scheme rests on: a decline is not a graph state. A rewrite
    /// that reports `None` leaves the revision alone, so an analysis derived before it is still
    /// the analysis of the current graph; a rewrite that reports `Some` moves it.
    #[test]
    fn only_a_committed_rewrite_advances_the_revision() {
        let mut blocks = RewriteBlocks::new(one_block());
        let start = blocks.revision();

        assert_eq!(blocks.rewrite(|_| None::<()>), None);
        assert_eq!(blocks.revision(), start, "a decline is not a change");

        assert_eq!(
            blocks.rewrite(|blocks| {
                blocks.push(bb("%tail", &["ret void"]));
                Some(())
            }),
            Some(())
        );
        assert_ne!(blocks.revision(), start, "a commit is a change");
        assert_eq!(blocks.get().len(), 2);
    }

    /// A rewrite that does not report whether it fired cannot be trusted to have done nothing, so
    /// handing out `&mut` counts as a change either way.
    #[test]
    fn unreported_edit_access_advances_the_revision() {
        let mut blocks = RewriteBlocks::new(one_block());
        let start = blocks.revision();
        let _ = blocks.edit();
        assert_ne!(blocks.revision(), start);
    }

    /// The cache derives once per graph state and not once per query — the reuse that turns one
    /// analysis per graph edit back into one per loop header — and never serves a result derived
    /// from a graph that has since changed.
    #[test]
    fn an_analysis_is_derived_once_per_revision() {
        let mut blocks = RewriteBlocks::new(one_block());
        let mut cache = AnalysisCache::default();
        let mut derivations = 0usize;

        for _ in 0..3 {
            assert_eq!(
                *cache.get(&blocks, |b| {
                    derivations += 1;
                    b.len()
                }),
                1
            );
        }
        assert_eq!(derivations, 1, "an unchanged graph is analyzed once");

        blocks.edit().push(bb("%tail", &["ret void"]));
        assert_eq!(
            *cache.get(&blocks, |b| {
                derivations += 1;
                b.len()
            }),
            2
        );
        assert_eq!(derivations, 2, "a changed graph is analyzed again");
    }

    /// A declined rewrite that edited the list anyway is what silently poisons every cache keyed on
    /// the revision, so debug builds catch it at the rewrite rather than at some later wrong answer.
    #[test]
    #[should_panic(expected = "a declined rewrite edited the block list")]
    #[cfg(debug_assertions)]
    fn a_declined_rewrite_that_edited_the_list_is_caught() {
        let mut blocks = RewriteBlocks::new(one_block());
        blocks.rewrite(|blocks| {
            blocks.push(bb("%tail", &["ret void"]));
            None::<()>
        });
    }
}
