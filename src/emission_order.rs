//! Encounter-ordered deduplication for collections that decide what the module looks like.
//!
//! Any decision the emitter makes while walking a set — which id to mint first, which decoration to
//! push next — becomes part of the output bytes. A `HashSet`/`HashMap` has no order to walk: Rust
//! seeds every map's hasher separately, so the same input yields a different permutation on every
//! run and the translator silently produces a semantically identical module with different bytes.
//! That breaks content-hash caching, the validation ladder's SHA-256 comparison, and the ability to
//! reproduce a bug report from the module that caused it.
//!
//! The fix is always the same shape: gather in the order the module's own instruction stream
//! presents the elements, and drop repeats. That order is a property of the input, so it is the same
//! on every run and on every machine. [`dedup_in_encounter_order`] is that idiom in one place, so
//! the correct spelling is shorter than the incorrect one.
//!
//! Use this — not a `HashSet` — whenever the collected elements go on to mint ids, push
//! instructions, or push annotations. A `HashSet` is still right for pure membership questions,
//! where nothing is ever iterated.

use std::collections::HashSet;
use std::hash::Hash;

/// The distinct elements of `items`, in the order they were first produced.
///
/// The `HashSet` here only answers "have I seen this?", which is order-free; the returned `Vec`
/// carries the encounter order the caller must emit in.
pub(crate) fn dedup_in_encounter_order<T: Copy + Eq + Hash>(
    items: impl IntoIterator<Item = T>,
) -> Vec<T> {
    let mut seen = HashSet::new();
    items
        .into_iter()
        .filter(|item| seen.insert(*item))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_occurrence_wins_and_repeats_are_dropped() {
        assert_eq!(
            dedup_in_encounter_order([7u32, 3, 7, 9, 3, 3, 1]),
            vec![7, 3, 9, 1]
        );
    }

    #[test]
    fn an_empty_input_collects_to_nothing() {
        assert!(dedup_in_encounter_order(std::iter::empty::<u32>()).is_empty());
    }

    /// The point of the helper: the result must not depend on how the elements hash. Building the
    /// same elements through many independently seeded `HashSet`s and re-deduping must reproduce the
    /// input order every time, which a `HashSet`-collected pipeline would not.
    #[test]
    fn the_order_is_the_inputs_order_not_a_hash_order() {
        let items: Vec<u32> = (0..64).map(|i| i * 7 + 1).collect();
        for _ in 0..64 {
            assert_eq!(dedup_in_encounter_order(items.iter().copied()), items);
        }
    }
}
