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
                    1 if !dominators.dominates(target, node) => {
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

    /// The reachable subgraph in reverse postorder, entry first.
    ///
    /// Reverse postorder is what makes the dominator iteration below converge in a couple of passes
    /// instead of one pass per graph depth: every node's predecessors along a forward edge are
    /// already final when it is visited.
    fn reverse_postorder(&self) -> Vec<Label> {
        let mut order = Vec::new();
        let mut seen = BTreeSet::new();
        if !self.successors.contains_key(&self.entry) {
            return order;
        }
        seen.insert(self.entry);
        let mut stack = vec![(self.entry, 0usize)];
        while let Some((node, index)) = stack.pop() {
            let targets = self.succ(node);
            if index < targets.len() {
                stack.push((node, index + 1));
                let target = targets[index];
                if self.successors.contains_key(&target) && seen.insert(target) {
                    stack.push((target, 0));
                }
            } else {
                order.push(node);
            }
        }
        order.reverse();
        order
    }

    /// The dominator tree of the reachable subgraph, by the Cooper-Harvey-Kennedy iteration.
    fn dominators(&self) -> Dominators {
        let order = self.reverse_postorder();
        if order.is_empty() {
            return Dominators::default();
        }
        let position = order
            .iter()
            .enumerate()
            .map(|(index, label)| (*label, index))
            .collect::<BTreeMap<_, _>>();
        // Indexed by reverse-postorder position, so a dominator always sits at a lower index than
        // the node it dominates -- which is what lets `intersect` walk two fingers to their meet.
        let mut idom: Vec<Option<usize>> = vec![None; order.len()];
        idom[0] = Some(0);
        let mut changed = true;
        while changed {
            changed = false;
            for index in 1..order.len() {
                let mut candidate: Option<usize> = None;
                for predecessor in self.pred(order[index]) {
                    let Some(&predecessor) = position.get(predecessor) else {
                        continue;
                    };
                    if idom[predecessor].is_none() {
                        continue;
                    }
                    candidate = Some(match candidate {
                        None => predecessor,
                        Some(current) => intersect(current, predecessor, &idom),
                    });
                }
                if candidate.is_some() && idom[index] != candidate {
                    idom[index] = candidate;
                    changed = true;
                }
            }
        }
        Dominators::from_tree(&order, &idom)
    }
}

/// The meet of two nodes in the partially built dominator tree: their deepest common dominator.
///
/// Both fingers only ever move to a strictly lower reverse-postorder position, so the walk
/// terminates. An index whose parent link is missing or does not decrease cannot occur once the
/// iteration is seeded from the entry; treating it as the entry keeps the walk finite and only ever
/// weakens the dominance the caller concludes, which is the safe direction.
fn intersect(mut left: usize, mut right: usize, idom: &[Option<usize>]) -> usize {
    while left != right {
        while left > right {
            match idom[left] {
                Some(parent) if parent < left => left = parent,
                _ => return 0,
            }
        }
        while right > left {
            match idom[right] {
                Some(parent) if parent < right => right = parent,
                _ => return 0,
            }
        }
    }
    left
}

/// Which blocks dominate which, over a graph's reachable subgraph.
///
/// One pre-order interval per block, not a dominator *set* per block. The set form is quadratic in
/// the block count: a 6301-block function -- an ordinary size for a generated convolution kernel --
/// materializes about 40 million set entries, roughly 400 MiB, to answer the single question
/// [`Graph::is_reducible`] asks of it. That alone exceeds the whole per-translation memory budget.
/// The interval form is linear, and answers the same question with two integer comparisons.
#[derive(Default)]
pub(super) struct Dominators {
    /// `(enter, exit)` of each block in a pre-order walk of the dominator tree. `a` dominates `b`
    /// exactly when `b`'s interval nests inside `a`'s.
    interval: BTreeMap<Label, (u32, u32)>,
}

impl Dominators {
    /// Number the dominator tree described by `idom` (parent links by reverse-postorder position).
    fn from_tree(order: &[Label], idom: &[Option<usize>]) -> Self {
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); order.len()];
        for (index, parent) in idom.iter().enumerate() {
            match parent {
                Some(parent) if *parent != index => children[*parent].push(index),
                _ => {}
            }
        }
        let mut interval = BTreeMap::new();
        let mut clock = 0u32;
        // `Descend` numbers a node on the way in, `Close` on the way back out; an explicit stack
        // rather than recursion because the tree is as deep as the function is long.
        enum Step {
            Descend(usize),
            Close(usize),
        }
        let mut stack = vec![Step::Descend(0)];
        let mut enter = vec![0u32; order.len()];
        while let Some(step) = stack.pop() {
            match step {
                Step::Descend(index) => {
                    enter[index] = clock;
                    clock += 1;
                    stack.push(Step::Close(index));
                    for child in children[index].iter().rev() {
                        stack.push(Step::Descend(*child));
                    }
                }
                Step::Close(index) => {
                    interval.insert(order[index], (enter[index], clock));
                    clock += 1;
                }
            }
        }
        Self { interval }
    }

    /// Whether `ancestor` dominates `node`. Reflexive, and false for any block the reachable
    /// subgraph does not contain -- which is what the retreating-edge check wants: an edge whose
    /// source is unreachable was never on the walk that asks.
    pub(super) fn dominates(&self, ancestor: Label, node: Label) -> bool {
        let (Some(ancestor), Some(node)) = (self.interval.get(&ancestor), self.interval.get(&node))
        else {
            return false;
        };
        ancestor.0 <= node.0 && node.1 <= ancestor.1
    }

    /// How many numbers the tree holds, which is what the linear bound is about.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.interval.len()
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
        /// The labels control can arrive at, which is not the arm list: an entry whose region is
        /// shared with another entry owns no arm and is reached through `next`.
        entries: BTreeSet<Label>,
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

    /// Exactly the labels control can arrive at from outside this shape. A branch whose target is
    /// in this set enters the shape; anything else passes it by, which is what tells the emitter a
    /// branch has to leave more than one construct.
    pub(super) fn entry_labels(&self) -> BTreeSet<Label> {
        match self {
            Shape::Simple { label, .. } => BTreeSet::from([*label]),
            Shape::Loop { entries, .. } | Shape::Multiple { entries, .. } => entries.clone(),
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
    // A loop body always covers at least its entries, so the recursion cannot decline; the empty
    // dispatch is an unconstructible placeholder the emitter rejects rather than a real shape.
    let inner = body.unwrap_or_else(|| Shape::Multiple {
        id: fresh(counter),
        entries: entries.clone(),
        handled: Vec::new(),
        next: None,
    });
    Shape::Loop {
        id,
        entries,
        inner: Box::new(inner),
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
        entries,
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

    /// The textbook dominator sets, by fixpoint over the reachable subgraph.
    ///
    /// This is the definition the tree form has to agree with, kept here rather than in the graph
    /// because it is quadratic in the block count and no production path can afford it.
    fn dense_dominators(graph: &Graph) -> BTreeMap<Label, BTreeSet<Label>> {
        let reachable = graph.reachable();
        let mut dominators: BTreeMap<Label, BTreeSet<Label>> = reachable
            .iter()
            .map(|label| (*label, reachable.clone()))
            .collect();
        if let Some(entry) = dominators.get_mut(&graph.entry) {
            *entry = BTreeSet::from([graph.entry]);
        }
        let mut changed = true;
        while changed {
            changed = false;
            for label in &reachable {
                if *label == graph.entry {
                    continue;
                }
                let mut incoming = graph
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

    fn assert_matches_dense(graph: &Graph, what: &str) {
        let dense = dense_dominators(graph);
        let tree = graph.dominators();
        let reachable = graph.reachable();
        for node in &reachable {
            for ancestor in &reachable {
                assert_eq!(
                    tree.dominates(*ancestor, *node),
                    dense[node].contains(ancestor),
                    "{what}: disagreement on whether %{ancestor} dominates %{node}"
                );
            }
        }
    }

    /// The dominator tree has to answer exactly what the dominator sets it replaced answered, on
    /// every shape the structurizer can be handed -- including the irreducible ones, where the
    /// reducibility verdict is the whole point of asking.
    #[test]
    fn the_dominator_tree_agrees_with_the_dominator_sets() {
        for (what, graph) in [
            ("straight line", graph(1, &[(1, &[2]), (2, &[3]), (3, &[])])),
            (
                "diamond",
                graph(1, &[(1, &[2, 3]), (2, &[4]), (3, &[4]), (4, &[])]),
            ),
            (
                "loop with an exit",
                graph(1, &[(1, &[2]), (2, &[3, 4]), (3, &[2]), (4, &[])]),
            ),
            (
                "nested loops",
                graph(
                    1,
                    &[(1, &[2]), (2, &[3]), (3, &[3, 4]), (4, &[2, 5]), (5, &[])],
                ),
            ),
            ("self loop", graph(1, &[(1, &[1, 2]), (2, &[])])),
            (
                "irreducible pair",
                graph(1, &[(1, &[2, 3]), (2, &[3]), (3, &[2])]),
            ),
            (
                "unreachable block",
                graph(1, &[(1, &[2]), (2, &[]), (9, &[2])]),
            ),
            (
                "two paths rejoining under a loop",
                graph(
                    1,
                    &[
                        (1, &[2, 3]),
                        (2, &[4]),
                        (3, &[4]),
                        (4, &[5, 6]),
                        (5, &[4]),
                        (6, &[]),
                    ],
                ),
            ),
        ] {
            assert_matches_dense(&graph, what);
        }
    }

    /// The bound this representation exists for: the dominator structure holds one interval per
    /// reachable block, not one dominator set per reachable block.
    ///
    /// A dominator *set* per block is quadratic, and generated shaders reach block counts where
    /// that alone costs hundreds of megabytes -- past the whole per-translation memory budget for
    /// one reducibility question. Asserting the linear size keeps that from creeping back: the
    /// dense form would report `n * (n + 1) / 2` numbers here, not `n`.
    #[test]
    fn the_dominator_structure_stays_linear_in_the_block_count() {
        // A long chain of two-way diamonds: deep enough that a quadratic representation is
        // unmistakable, and reducible, so the walk covers every block.
        let mut edges: Vec<(Label, Vec<Label>)> = Vec::new();
        let diamonds = 2_000;
        for index in 0..diamonds {
            let head = index * 4 + 1;
            edges.push((head, vec![head + 1, head + 2]));
            edges.push((head + 1, vec![head + 3]));
            edges.push((head + 2, vec![head + 3]));
            edges.push((head + 3, vec![head + 4]));
        }
        let last = diamonds * 4 + 1;
        edges.push((last, vec![]));
        let graph = Graph::new(
            1,
            edges.into_iter().collect::<BTreeMap<Label, Vec<Label>>>(),
        );
        let blocks = graph.reachable().len();
        assert_eq!(blocks, (diamonds * 4 + 1) as usize);
        assert!(graph.is_reducible());
        assert_eq!(graph.dominators().len(), blocks);
    }
}
