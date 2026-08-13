//! Compact dominance over an emitted SPIR-V function CFG.
//!
//! Shared by late native rewrites and the emitter's phi-materialization repair. The immediate-
//! dominator tree uses O(V + E) storage and answers dominance with DFS intervals; the former
//! `label -> HashSet<all dominators>` representation was O(V²) and drove large translations far
//! beyond their resident-memory budget even after those temporary sets were freed.

use spirv::Word;
use std::collections::HashMap;

pub(in crate::native) struct EmittedDominators {
    index: HashMap<Word, usize>,
    preorder: Vec<usize>,
    postorder: Vec<usize>,
    depth: Vec<usize>,
}

impl EmittedDominators {
    pub(in crate::native) fn new(
        entry: Word,
        labels: &[Word],
        successors_by_label: &HashMap<Word, Vec<Word>>,
    ) -> Self {
        let index = labels
            .iter()
            .enumerate()
            .map(|(idx, label)| (*label, idx))
            .collect::<HashMap<_, _>>();
        let mut successors = vec![Vec::new(); labels.len()];
        let mut predecessors = vec![Vec::new(); labels.len()];
        for (&label, targets) in successors_by_label {
            let Some(&from) = index.get(&label) else {
                continue;
            };
            for target in targets {
                let Some(&to) = index.get(target) else {
                    continue;
                };
                successors[from].push(to);
                predecessors[to].push(from);
            }
        }

        let mut rpo = Vec::new();
        if let Some(&entry) = index.get(&entry) {
            let mut seen = vec![false; labels.len()];
            let mut stack = vec![(entry, 0usize)];
            seen[entry] = true;
            while let Some((node, next)) = stack.last_mut() {
                if *next < successors[*node].len() {
                    let successor = successors[*node][*next];
                    *next += 1;
                    if !seen[successor] {
                        seen[successor] = true;
                        stack.push((successor, 0));
                    }
                } else {
                    rpo.push(*node);
                    stack.pop();
                }
            }
            rpo.reverse();
        }
        let mut rpo_rank = vec![usize::MAX; labels.len()];
        for (rank, node) in rpo.iter().copied().enumerate() {
            rpo_rank[node] = rank;
        }
        let mut idom = vec![None; labels.len()];
        if let Some(&entry) = rpo.first() {
            idom[entry] = Some(entry);
            loop {
                let mut changed = false;
                for node in rpo.iter().copied().skip(1) {
                    let mut defined = predecessors[node]
                        .iter()
                        .copied()
                        .filter(|pred| idom[*pred].is_some());
                    let Some(mut next_idom) = defined.next() else {
                        continue;
                    };
                    for pred in defined {
                        next_idom = intersect(pred, next_idom, &idom, &rpo_rank);
                    }
                    if idom[node] != Some(next_idom) {
                        idom[node] = Some(next_idom);
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        let mut children = vec![Vec::new(); labels.len()];
        for (node, parent) in idom.iter().copied().enumerate() {
            if let Some(parent) = parent.filter(|parent| *parent != node) {
                children[parent].push(node);
            }
        }
        let mut preorder = vec![usize::MAX; labels.len()];
        let mut postorder = vec![usize::MAX; labels.len()];
        let mut depth = vec![0; labels.len()];
        if let Some(&entry) = rpo.first() {
            let mut clock = 0usize;
            let mut stack = vec![(entry, 0usize)];
            preorder[entry] = clock;
            depth[entry] = 1;
            clock += 1;
            while let Some((node, next)) = stack.last_mut() {
                if *next < children[*node].len() {
                    let child = children[*node][*next];
                    *next += 1;
                    preorder[child] = clock;
                    depth[child] = depth[*node] + 1;
                    clock += 1;
                    stack.push((child, 0));
                } else {
                    postorder[*node] = clock;
                    stack.pop();
                }
            }
        }
        Self {
            index,
            preorder,
            postorder,
            depth,
        }
    }

    pub(in crate::native) fn dominates(&self, dominator: Word, node: Word) -> bool {
        let (Some(&dominator), Some(&node)) = (self.index.get(&dominator), self.index.get(&node))
        else {
            return false;
        };
        self.preorder[dominator] != usize::MAX
            && self.preorder[dominator] <= self.preorder[node]
            && self.preorder[node] < self.postorder[dominator]
    }

    pub(in crate::native) fn depth(&self, label: Word) -> usize {
        self.index.get(&label).map_or(0, |index| self.depth[*index])
    }
}

fn intersect(
    mut left: usize,
    mut right: usize,
    idom: &[Option<usize>],
    rpo_rank: &[usize],
) -> usize {
    while left != right {
        while rpo_rank[left] > rpo_rank[right] {
            left = idom[left].expect("defined dominator chain");
        }
        while rpo_rank[right] > rpo_rank[left] {
            right = idom[right].expect("defined dominator chain");
        }
    }
    left
}
