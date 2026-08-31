//! Shape computation for the nesting structurizer.
//!
//! Given a control-flow graph this derives the classic relooper shape tree — `Simple`, `Loop`,
//! `Multiple` — which is the nesting the emitter turns into SPIR-V loop and selection constructs.
//! It is pure graph analysis: no SPIR-V types, no module state, no ids beyond the block labels it
//! is handed, so it is unit-testable on synthetic graphs.
//!
//! The derivation is the standard one. For a set of `blocks` reachable through a set of `entries`:
//!
//! - one entry with no edge into it from inside `blocks` is a **Simple** shape: that block runs,
//!   then whatever its successors start;
//! - an entry that IS re-entered from inside `blocks` is a **Loop**: the blocks that can reach an
//!   entry form the body, the rest follows the loop;
//! - several entries that own disjoint block sets form a **Multiple**: each entry's private region
//!   becomes one arm, and everything shared follows the dispatch.
//!
//! Every rule strictly shrinks `blocks`, so the recursion terminates on any graph.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Edges the current recursion treats as absent: the back edges of an enclosing loop, which the
/// emitter renders as `continue` rather than as an ordinary branch into the body.
type Cut = BTreeSet<(Label, Label)>;

/// A block label. The emitter uses SPIR-V result ids; the shape layer only needs a total order.
pub(super) type Label = u32;

/// The control-flow graph the shape derivation reads.
pub(super) struct Graph {
    pub(super) entry: Label,
    /// Successors in terminator order, with duplicates removed.
    pub(super) successors: BTreeMap<Label, Vec<Label>>,
    pub(super) predecessors: BTreeMap<Label, Vec<Label>>,
}

impl Graph {
    pub(super) fn new(entry: Label, successors: BTreeMap<Label, Vec<Label>>) -> Self {
        let mut predecessors: BTreeMap<Label, Vec<Label>> = BTreeMap::new();
        for label in successors.keys() {
            predecessors.entry(*label).or_default();
        }
        for (label, targets) in &successors {
            for target in targets {
                let list = predecessors.entry(*target).or_default();
                if !list.contains(label) {
                    list.push(*label);
                }
            }
        }
        Self {
            entry,
            successors,
            predecessors,
        }
    }

    pub(super) fn succ(&self, label: Label) -> &[Label] {
        self.successors.get(&label).map_or(&[], Vec::as_slice)
    }

    fn pred(&self, label: Label) -> &[Label] {
        self.predecessors.get(&label).map_or(&[], Vec::as_slice)
    }

    /// Blocks reachable from `label` without leaving `within`, ignoring `cut` edges.
    fn reachable_within(
        &self,
        label: Label,
        within: &BTreeSet<Label>,
        cut: &Cut,
    ) -> BTreeSet<Label> {
        let mut seen = BTreeSet::new();
        let mut queue = VecDeque::new();
        if within.contains(&label) {
            seen.insert(label);
            queue.push_back(label);
        }
        while let Some(current) = queue.pop_front() {
            for target in self.succ(current) {
                if within.contains(target)
                    && !cut.contains(&(current, *target))
                    && seen.insert(*target)
                {
                    queue.push_back(*target);
                }
            }
        }
        seen
    }

    /// The blocks a depth-first walk from the entry can reach.
    pub(super) fn reachable(&self) -> BTreeSet<Label> {
        let all = self.successors.keys().copied().collect::<BTreeSet<_>>();
        self.reachable_within(self.entry, &all, &Cut::new())
    }

    /// Whether every retreating edge of a depth-first walk targets a dominator of its source. A
    /// graph that fails this is irreducible: no shape tree preserves its entries without cloning,
    /// so the caller keeps such a function on the state-machine constructor.
    pub(super) fn is_reducible(&self) -> bool {
        let dominators = self.dominators();
        let mut state: BTreeMap<Label, u8> = BTreeMap::new();
        let mut stack = vec![(self.entry, 0usize)];
        state.insert(self.entry, 1);
        while let Some((node, index)) = stack.pop() {
            let targets = self.succ(node);
            if index < targets.len() {
                stack.push((node, index + 1));
                let target = targets[index];
                match state.get(&target).copied().unwrap_or(0) {
                    0 => {
                        state.insert(target, 1);
                        stack.push((target, 0));
                    }
                    // A retreating edge: legal only when the target dominates the source.
                    1 if !dominators
                        .get(&node)
                        .is_some_and(|doms| doms.contains(&target)) =>
                    {
                        return false;
                    }
                    _ => {}
                }
            } else {
                state.insert(node, 2);
            }
        }
        true
    }

    /// Iterative dominator sets over the reachable subgraph.
    fn dominators(&self) -> BTreeMap<Label, BTreeSet<Label>> {
        let reachable = self.reachable();
        let mut dominators: BTreeMap<Label, BTreeSet<Label>> = reachable
            .iter()
            .map(|label| (*label, reachable.clone()))
            .collect();
        if let Some(entry) = dominators.get_mut(&self.entry) {
            *entry = BTreeSet::from([self.entry]);
        }
        let mut changed = true;
        while changed {
            changed = false;
            for label in &reachable {
                if *label == self.entry {
                    continue;
                }
                let mut incoming = self
                    .pred(*label)
                    .iter()
                    .filter(|predecessor| reachable.contains(predecessor))
                    .filter_map(|predecessor| dominators.get(predecessor));
                let Some(first) = incoming.next() else {
                    continue;
                };
                let mut next = first.clone();
                for other in incoming {
                    next.retain(|candidate| other.contains(candidate));
                }
                next.insert(*label);
                if dominators.get(label) != Some(&next) {
                    dominators.insert(*label, next);
                    changed = true;
                }
            }
        }
        dominators
    }
}

/// One node of the relooper shape tree.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Shape {
    /// One original block, followed by whatever its successors begin.
    Simple {
        id: usize,
        label: Label,
        next: Option<Box<Shape>>,
    },
    /// A construct re-entered from within itself. `inner` starts at `entries`; `next` is everything
    /// the body branches out to.
    Loop {
        id: usize,
        entries: BTreeSet<Label>,
        inner: Box<Shape>,
        next: Option<Box<Shape>>,
    },
    /// Several entries owning disjoint regions. Each arm is entered by its own label; anything the
    /// arms share follows in `next`.
    Multiple {
        id: usize,
        handled: Vec<(Label, Shape)>,
        next: Option<Box<Shape>>,
    },
}

impl Shape {
    pub(super) fn id(&self) -> usize {
        match self {
            Shape::Simple { id, .. } | Shape::Loop { id, .. } | Shape::Multiple { id, .. } => *id,
        }
    }

    /// The labels this shape can be entered at. A branch whose target is in this set is a branch
    /// *into* the shape rather than past it.
    pub(super) fn entry_labels(&self) -> BTreeSet<Label> {
        match self {
            Shape::Simple { label, .. } => BTreeSet::from([*label]),
            Shape::Loop { entries, .. } => entries.clone(),
            Shape::Multiple { handled, next, .. } => {
                let mut labels = handled
                    .iter()
                    .map(|(label, _)| *label)
                    .collect::<BTreeSet<_>>();
                if let Some(next) = next {
                    labels.extend(next.entry_labels());
                }
                labels
            }
        }
    }
}

/// Derive the shape tree for `graph`, covering exactly its reachable blocks.
pub(super) fn calculate(graph: &Graph) -> Option<Shape> {
    let mut counter = 0usize;
    let blocks = graph.reachable();
    let entries = BTreeSet::from([graph.entry]);
    let mut cut = Cut::new();
    calculate_region(graph, blocks, entries, &mut cut, &mut counter)
}

fn calculate_region(
    graph: &Graph,
    mut blocks: BTreeSet<Label>,
    entries: BTreeSet<Label>,
    cut: &mut Cut,
    counter: &mut usize,
) -> Option<Shape> {
    let entries = entries
        .into_iter()
        .filter(|entry| blocks.contains(entry))
        .collect::<BTreeSet<_>>();
    if entries.is_empty() || blocks.is_empty() {
        return None;
    }
    if entries.len() == 1 {
        let label = *entries.iter().next()?;
        let reentered = graph.pred(label).iter().any(|predecessor| {
            blocks.contains(predecessor) && !cut.contains(&(*predecessor, label))
        });
        if !reentered {
            let id = fresh(counter);
            blocks.remove(&label);
            let next_entries = graph
                .succ(label)
                .iter()
                .copied()
                .filter(|target| blocks.contains(target) && !cut.contains(&(label, *target)))
                .collect::<BTreeSet<_>>();
            let next = calculate_region(graph, blocks, next_entries, cut, counter);
            return Some(Shape::Simple {
                id,
                label,
                next: next.map(Box::new),
            });
        }
        return Some(make_loop(graph, blocks, entries, cut, counter));
    }
    match independent_groups(graph, &blocks, &entries, cut) {
        Some(groups) => Some(make_multiple(graph, blocks, entries, groups, cut, counter)),
        None => Some(make_loop(graph, blocks, entries, cut, counter)),
    }
}

fn make_loop(
    graph: &Graph,
    mut blocks: BTreeSet<Label>,
    entries: BTreeSet<Label>,
    cut: &mut Cut,
    counter: &mut usize,
) -> Shape {
    let id = fresh(counter);
    // The body is every block that can still reach an entry: walking predecessors backwards from
    // the entries, staying inside the region, collects exactly the blocks whose control can loop.
    let mut inner = BTreeSet::new();
    let mut queue = entries.iter().copied().collect::<VecDeque<_>>();
    while let Some(current) = queue.pop_front() {
        if !blocks.contains(&current) || !inner.insert(current) {
            continue;
        }
        for predecessor in graph.pred(current) {
            if blocks.contains(predecessor)
                && !cut.contains(&(*predecessor, current))
                && !inner.contains(predecessor)
            {
                queue.push_back(*predecessor);
            }
        }
    }
    blocks.retain(|label| !inner.contains(label));
    let next_entries = inner
        .iter()
        .flat_map(|label| {
            graph
                .succ(*label)
                .iter()
                .map(move |target| (*label, *target))
        })
        .filter(|edge| blocks.contains(&edge.1) && !cut.contains(edge))
        .map(|edge| edge.1)
        .collect::<BTreeSet<_>>();
    // Inside the body the edges back to the entries are `continue`, not branches into the body.
    // Removing them is what lets the recursion see an acyclic region and terminate.
    let back_edges = inner
        .iter()
        .flat_map(|label| {
            entries
                .iter()
                .filter(|entry| graph.succ(*label).contains(entry))
                .map(move |entry| (*label, *entry))
        })
        .collect::<Vec<_>>();
    let restore = back_edges
        .iter()
        .filter(|edge| cut.insert(**edge))
        .copied()
        .collect::<Vec<_>>();
    let body = calculate_region(graph, inner, entries.clone(), cut, counter);
    for edge in restore {
        cut.remove(&edge);
    }
    let next = calculate_region(graph, blocks, next_entries, cut, counter);
    Shape::Loop {
        id,
        entries,
        // A loop body always covers at least its entries, so the recursion cannot decline.
        inner: Box::new(body.unwrap_or(Shape::Multiple {
            id: fresh(counter),
            handled: Vec::new(),
            next: None,
        })),
        next: next.map(Box::new),
    }
}

fn make_multiple(
    graph: &Graph,
    mut blocks: BTreeSet<Label>,
    entries: BTreeSet<Label>,
    groups: BTreeMap<Label, BTreeSet<Label>>,
    cut: &mut Cut,
    counter: &mut usize,
) -> Shape {
    let id = fresh(counter);
    let owned = groups
        .values()
        .flat_map(|group| group.iter().copied())
        .collect::<BTreeSet<_>>();
    blocks.retain(|label| !owned.contains(label));
    let mut handled = Vec::new();
    let mut next_entries = entries
        .iter()
        .copied()
        .filter(|entry| !groups.contains_key(entry))
        .collect::<BTreeSet<_>>();
    for (entry, group) in &groups {
        next_entries.extend(
            group
                .iter()
                .flat_map(|label| {
                    graph
                        .succ(*label)
                        .iter()
                        .map(move |target| (*label, *target))
                })
                .filter(|edge| blocks.contains(&edge.1) && !cut.contains(edge))
                .map(|edge| edge.1),
        );
        if let Some(shape) =
            calculate_region(graph, group.clone(), BTreeSet::from([*entry]), cut, counter)
        {
            handled.push((*entry, shape));
        }
    }
    let next = calculate_region(graph, blocks, next_entries, cut, counter);
    Shape::Multiple {
        id,
        handled,
        next: next.map(Box::new),
    }
}

/// The blocks each entry owns exclusively: reachable from that entry and from no other, and not an
/// entry itself. Returns `None` when no entry owns anything, which is the signal that the region is
/// a shared-continuation loop rather than a dispatch.
fn independent_groups(
    graph: &Graph,
    blocks: &BTreeSet<Label>,
    entries: &BTreeSet<Label>,
    cut: &Cut,
) -> Option<BTreeMap<Label, BTreeSet<Label>>> {
    let reach = entries
        .iter()
        .map(|entry| (*entry, graph.reachable_within(*entry, blocks, cut)))
        .collect::<BTreeMap<_, _>>();
    let mut groups = BTreeMap::new();
    for entry in entries {
        let Some(own) = reach.get(entry) else {
            continue;
        };
        let exclusive = own
            .iter()
            .copied()
            .filter(|label| {
                if entries.contains(label) && label != entry {
                    return false;
                }
                reach
                    .iter()
                    .all(|(other, seen)| other == entry || !seen.contains(label))
            })
            .collect::<BTreeSet<_>>();
        if !exclusive.is_empty() {
            groups.insert(*entry, exclusive);
        }
    }
    (!groups.is_empty()).then_some(groups)
}

fn fresh(counter: &mut usize) -> usize {
    let id = *counter;
    *counter += 1;
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(entry: Label, edges: &[(Label, &[Label])]) -> Graph {
        Graph::new(
            entry,
            edges
                .iter()
                .map(|(label, targets)| (*label, targets.to_vec()))
                .collect(),
        )
    }

    /// Every label the shape tree renders, in tree order.
    fn rendered(shape: &Shape, out: &mut Vec<Label>) {
        match shape {
            Shape::Simple { label, next, .. } => {
                out.push(*label);
                if let Some(next) = next {
                    rendered(next, out);
                }
            }
            Shape::Loop { inner, next, .. } => {
                rendered(inner, out);
                if let Some(next) = next {
                    rendered(next, out);
                }
            }
            Shape::Multiple { handled, next, .. } => {
                for (_, arm) in handled {
                    rendered(arm, out);
                }
                if let Some(next) = next {
                    rendered(next, out);
                }
            }
        }
    }

    #[test]
    fn straight_line_is_a_chain_of_simple_shapes() {
        let g = graph(1, &[(1, &[2]), (2, &[3]), (3, &[])]);
        let shape = calculate(&g).expect("shape");
        let mut order = Vec::new();
        rendered(&shape, &mut order);
        assert_eq!(order, vec![1, 2, 3]);
        assert!(matches!(shape, Shape::Simple { label: 1, .. }));
    }

    #[test]
    fn diamond_becomes_a_multiple_with_two_arms() {
        let g = graph(1, &[(1, &[2, 3]), (2, &[4]), (3, &[4]), (4, &[])]);
        let shape = calculate(&g).expect("shape");
        let Shape::Simple { label: 1, next, .. } = shape else {
            panic!("expected the header to be simple");
        };
        let next = next.expect("diamond has a continuation");
        let Shape::Multiple { handled, next, .. } = *next else {
            panic!("expected a dispatch over the two arms");
        };
        assert_eq!(
            handled.iter().map(|(label, _)| *label).collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert!(matches!(
            next.as_deref(),
            Some(Shape::Simple { label: 4, .. })
        ));
    }

    #[test]
    fn if_without_else_puts_the_single_arm_in_the_multiple() {
        let g = graph(1, &[(1, &[2, 3]), (2, &[3]), (3, &[])]);
        let shape = calculate(&g).expect("shape");
        let Shape::Simple { next, .. } = shape else {
            panic!("expected simple header");
        };
        let Some(box_next) = next else {
            panic!("expected a continuation")
        };
        let Shape::Multiple { handled, next, .. } = *box_next else {
            panic!("expected a dispatch");
        };
        assert_eq!(
            handled.iter().map(|(label, _)| *label).collect::<Vec<_>>(),
            vec![2]
        );
        assert!(matches!(
            next.as_deref(),
            Some(Shape::Simple { label: 3, .. })
        ));
    }

    #[test]
    fn self_loop_becomes_a_loop_shape() {
        let g = graph(1, &[(1, &[2]), (2, &[2, 3]), (3, &[])]);
        let shape = calculate(&g).expect("shape");
        let Shape::Simple { label: 1, next, .. } = shape else {
            panic!("expected simple header");
        };
        let Shape::Loop { entries, next, .. } = *next.expect("loop follows") else {
            panic!("expected a loop");
        };
        assert_eq!(entries, BTreeSet::from([2]));
        assert!(matches!(
            next.as_deref(),
            Some(Shape::Simple { label: 3, .. })
        ));
    }

    #[test]
    fn nested_loops_nest_in_the_shape_tree() {
        // 1 -> 2 -> 3 -> 3|4 ; 4 -> 2|5
        let g = graph(
            1,
            &[(1, &[2]), (2, &[3]), (3, &[3, 4]), (4, &[2, 5]), (5, &[])],
        );
        let shape = calculate(&g).expect("shape");
        let Shape::Simple { next, .. } = shape else {
            panic!("expected simple header");
        };
        let Shape::Loop { inner, next, .. } = *next.expect("outer loop") else {
            panic!("expected the outer loop");
        };
        assert!(matches!(
            next.as_deref(),
            Some(Shape::Simple { label: 5, .. })
        ));
        // The inner self-loop on 3 is a Loop nested inside the outer body.
        let mut order = Vec::new();
        rendered(&inner, &mut order);
        assert_eq!(order, vec![2, 3, 4]);
    }

    #[test]
    fn every_reachable_block_is_rendered_exactly_once() {
        let g = graph(
            1,
            &[
                (1, &[2, 3]),
                (2, &[4]),
                (3, &[4, 5]),
                (4, &[6]),
                (5, &[6]),
                (6, &[7, 2]),
                (7, &[]),
            ],
        );
        let shape = calculate(&g).expect("shape");
        let mut order = Vec::new();
        rendered(&shape, &mut order);
        let unique = order.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(order.len(), unique.len(), "rendered twice: {order:?}");
        assert_eq!(unique, g.reachable());
    }

    #[test]
    fn unreachable_blocks_are_not_rendered() {
        let g = graph(1, &[(1, &[2]), (2, &[]), (9, &[2])]);
        let shape = calculate(&g).expect("shape");
        let mut order = Vec::new();
        rendered(&shape, &mut order);
        assert_eq!(order, vec![1, 2]);
    }

    #[test]
    fn reducible_graphs_are_recognized() {
        let reducible = graph(1, &[(1, &[2]), (2, &[3, 4]), (3, &[2]), (4, &[])]);
        assert!(reducible.is_reducible());
        // Two blocks entering each other's "loop" from outside: no single header dominates.
        let irreducible = graph(1, &[(1, &[2, 3]), (2, &[3]), (3, &[2])]);
        assert!(!irreducible.is_reducible());
    }
}
