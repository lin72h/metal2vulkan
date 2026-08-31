//! The one dominator computation in this crate.
//!
//! Cooper-Harvey-Kennedy iterative immediate dominators over reverse postorder, numbered into
//! Euler intervals so a dominance query is two integer comparisons rather than a walk up the tree.
//!
//! It works on dense block indices, `0` being the entry, because every caller already has (or can
//! cheaply build) that shape: the owned-CFG check indexes blocks positionally, the relooper indexes
//! them positionally, and the nesting structurizer maps its labels through a table it builds anyway.
//! Keeping one implementation is the point. The obvious alternative -- a dominator *set* per block --
//! is quadratic in the block count, and generated shaders reach block counts where that alone costs
//! hundreds of megabytes; the structurizer shipped that form once and it cost more memory than the
//! whole rest of the translation.

/// The reverse adjacency of a dense successor table.
pub(in crate::native) fn build_predecessors(successors: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut predecessors = vec![Vec::new(); successors.len()];
    for (source, targets) in successors.iter().enumerate() {
        for target in targets {
            predecessors[*target].push(source);
        }
    }
    predecessors
}

/// Reachability, dominance intervals, and immediate dominators of a dense block graph.
///
/// Returns `(reachable, intervals, idom)`, each indexed by block. `reachable` is what a depth-first
/// walk from block `0` reaches; `intervals` and `idom` are `None` for everything it does not.
pub(in crate::native) fn dominance(
    successors: &[Vec<usize>],
    predecessors: &[Vec<usize>],
) -> (Vec<bool>, Vec<Option<(usize, usize)>>, Vec<Option<usize>>) {
    let count = successors.len();
    if count == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let mut visited = vec![false; count];
    let mut postorder = Vec::with_capacity(count);
    let mut stack = vec![(0usize, 0usize)];
    visited[0] = true;
    while let Some((node, cursor)) = stack.last_mut() {
        if *cursor < successors[*node].len() {
            let child = successors[*node][*cursor];
            *cursor += 1;
            if !visited[child] {
                visited[child] = true;
                stack.push((child, 0));
            }
        } else {
            postorder.push(*node);
            stack.pop();
        }
    }
    let mut rpo = postorder;
    rpo.reverse();
    let mut rpo_rank = vec![usize::MAX; count];
    for (rank, block) in rpo.iter().enumerate() {
        rpo_rank[*block] = rank;
    }
    let mut idom = vec![None; count];
    idom[0] = Some(0);
    loop {
        let mut changed = false;
        for &node in rpo.iter().skip(1) {
            let mut defined = predecessors[node]
                .iter()
                .copied()
                .filter(|predecessor| idom[*predecessor].is_some());
            let Some(mut candidate) = defined.next() else {
                continue;
            };
            for predecessor in defined {
                candidate = intersect(candidate, predecessor, &idom, &rpo_rank);
            }
            if idom[node] != Some(candidate) {
                idom[node] = Some(candidate);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut children = vec![Vec::new(); count];
    for (node, parent) in idom.iter().enumerate() {
        if let Some(parent) = parent.filter(|parent| *parent != node) {
            children[parent].push(node);
        }
    }
    let mut intervals: Vec<Option<(usize, usize)>> = vec![None; count];
    let mut clock = 0usize;
    let mut stack = vec![(0usize, false)];
    while let Some((node, exiting)) = stack.pop() {
        if exiting {
            let start = intervals[node].expect("reachable dominator entered").0;
            intervals[node] = Some((start, clock));
            clock += 1;
        } else {
            intervals[node] = Some((clock, clock));
            clock += 1;
            stack.push((node, true));
            stack.extend(children[node].iter().rev().map(|child| (*child, false)));
        }
    }
    (visited, intervals, idom)
}

/// The meet of two nodes in the partially built dominator tree (the CHK `intersect`).
///
/// Both fingers only ever move to a strictly earlier reverse-postorder position, so the walk
/// terminates.
fn intersect(
    mut left: usize,
    mut right: usize,
    idom: &[Option<usize>],
    rpo_rank: &[usize],
) -> usize {
    while left != right {
        while rpo_rank[left] > rpo_rank[right] {
            left = idom[left].expect("defined dominator");
        }
        while rpo_rank[right] > rpo_rank[left] {
            right = idom[right].expect("defined dominator");
        }
    }
    left
}

/// Whether `dominator` dominates `node`, given the Euler intervals from [`dominance`].
///
/// Reflexive, and false for either block being unreachable -- an unreachable block dominates
/// nothing and is dominated by nothing.
pub(in crate::native) fn dominates_interval(
    dominance: &[Option<(usize, usize)>],
    dominator: usize,
    node: usize,
) -> bool {
    let (Some((dom_in, dom_out)), Some((node_in, node_out))) =
        (dominance[dominator], dominance[node])
    else {
        return false;
    };
    dom_in <= node_in && node_out <= dom_out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The textbook dominator sets, by fixpoint. This is the definition [`dominance`] has to agree
    /// with; it lives here rather than in the graph because it is quadratic in the block count and
    /// no production path can afford it.
    fn dense_dominators(successors: &[Vec<usize>]) -> Vec<Option<BTreeSet<usize>>> {
        let predecessors = build_predecessors(successors);
        let mut reachable = vec![false; successors.len()];
        let mut stack = vec![0usize];
        if successors.is_empty() {
            return Vec::new();
        }
        reachable[0] = true;
        while let Some(node) = stack.pop() {
            for target in &successors[node] {
                if !reachable[*target] {
                    reachable[*target] = true;
                    stack.push(*target);
                }
            }
        }
        let all = (0..successors.len())
            .filter(|node| reachable[*node])
            .collect::<BTreeSet<_>>();
        let mut dominators = (0..successors.len())
            .map(|node| reachable[node].then(|| all.clone()))
            .collect::<Vec<_>>();
        dominators[0] = Some(BTreeSet::from([0]));
        let mut changed = true;
        while changed {
            changed = false;
            for node in all.iter().copied().filter(|node| *node != 0) {
                let mut incoming = predecessors[node]
                    .iter()
                    .filter(|predecessor| reachable[**predecessor])
                    .filter_map(|predecessor| dominators[*predecessor].as_ref());
                let Some(first) = incoming.next() else {
                    continue;
                };
                let mut next = first.clone();
                for other in incoming {
                    next.retain(|candidate| other.contains(candidate));
                }
                next.insert(node);
                if dominators[node].as_ref() != Some(&next) {
                    dominators[node] = Some(next);
                    changed = true;
                }
            }
        }
        dominators
    }

    /// Every graph shape the translator hands this: straight line, diamond, loops, a self loop, an
    /// irreducible pair, unreachable blocks, and a block with no successors at all.
    fn shapes() -> Vec<(&'static str, Vec<Vec<usize>>)> {
        vec![
            ("single block", vec![vec![]]),
            ("straight line", vec![vec![1], vec![2], vec![]]),
            ("diamond", vec![vec![1, 2], vec![3], vec![3], vec![]]),
            (
                "loop with an exit",
                vec![vec![1], vec![2, 3], vec![1], vec![]],
            ),
            (
                "nested loops",
                vec![vec![1], vec![2], vec![2, 3], vec![1, 4], vec![]],
            ),
            ("self loop", vec![vec![0, 1], vec![]]),
            ("irreducible pair", vec![vec![1, 2], vec![2], vec![1]]),
            ("unreachable block", vec![vec![1], vec![], vec![1]]),
            (
                "two paths rejoining under a loop",
                vec![vec![1, 2], vec![3], vec![3], vec![4, 5], vec![3], vec![]],
            ),
        ]
    }

    #[test]
    fn the_intervals_answer_exactly_what_the_dominator_sets_answer() {
        for (what, successors) in shapes() {
            let dense = dense_dominators(&successors);
            let (reachable, intervals, _) =
                dominance(&successors, &build_predecessors(&successors));
            for node in 0..successors.len() {
                assert_eq!(
                    intervals[node].is_some(),
                    reachable[node],
                    "{what}: block {node} is numbered iff it is reachable"
                );
                for dominator in 0..successors.len() {
                    let expected = dense[node]
                        .as_ref()
                        .is_some_and(|set| set.contains(&dominator));
                    assert_eq!(
                        dominates_interval(&intervals, dominator, node),
                        expected,
                        "{what}: disagreement on whether {dominator} dominates {node}"
                    );
                }
            }
        }
    }

    /// The representation is one number per block, three ways -- not a set per block. A dominator
    /// set per block is quadratic, and shaders reach block counts where that alone costs more than
    /// the whole per-translation memory budget.
    #[test]
    fn the_tables_hold_one_entry_per_block() {
        for (what, successors) in shapes() {
            let (reachable, intervals, idom) =
                dominance(&successors, &build_predecessors(&successors));
            assert_eq!(reachable.len(), successors.len(), "{what}");
            assert_eq!(intervals.len(), successors.len(), "{what}");
            assert_eq!(idom.len(), successors.len(), "{what}");
        }
    }

    #[test]
    fn an_empty_graph_has_no_dominators() {
        let (reachable, intervals, idom) = dominance(&[], &[]);
        assert!(reachable.is_empty() && intervals.is_empty() && idom.is_empty());
    }
}
