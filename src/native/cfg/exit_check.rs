//! Iterative dominator-set analysis over an emitted SPIR-V function's CFG.
//!
//! Shared by [`super::ssa_demote`] and [`super::loop_split`], which both need emitted-CFG dominance
//! (`dominates(a, b)` == `doms[b].contains(a)`) over the reachable subgraph.

use super::graph::spirv_predecessor_ids_by_label;
use spirv::Word;
use std::collections::{HashMap, HashSet};

/// Iterative dominator-set computation (`node -> set of blocks that dominate it`) over the reachable
/// subgraph. Standard fixpoint intersection of predecessors' dominators; unreachable blocks are left
/// out (they carry no structured-exit meaning). Shared with [`super::ssa_demote`], which needs the same
/// emitted-CFG dominance to find defs that no longer dominate their uses.
pub(super) fn dominator_sets(
    entry: Word,
    labels: &[Word],
    successors: &HashMap<Word, Vec<Word>>,
) -> HashMap<Word, HashSet<Word>> {
    let preds = spirv_predecessor_ids_by_label(successors);
    let reachable = reachable_from(entry, successors);
    let all: HashSet<Word> = labels
        .iter()
        .copied()
        .filter(|l| reachable.contains(l))
        .collect();

    let mut dom: HashMap<Word, HashSet<Word>> = HashMap::new();
    for &l in labels {
        if !reachable.contains(&l) {
            continue;
        }
        if l == entry {
            dom.insert(l, HashSet::from([l]));
        } else {
            dom.insert(l, all.clone());
        }
    }
    loop {
        let mut changed = false;
        for &l in labels {
            if l == entry || !reachable.contains(&l) {
                continue;
            }
            let mut acc: Option<HashSet<Word>> = None;
            for &p in preds.get(&l).into_iter().flatten() {
                if !reachable.contains(&p) {
                    continue;
                }
                match &mut acc {
                    None => acc = Some(dom[&p].clone()),
                    Some(a) => a.retain(|x| dom[&p].contains(x)),
                }
            }
            let mut next = acc.unwrap_or_default();
            next.insert(l);
            if dom.get(&l) != Some(&next) {
                dom.insert(l, next);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    dom
}

fn reachable_from(entry: Word, successors: &HashMap<Word, Vec<Word>>) -> HashSet<Word> {
    let mut seen = HashSet::new();
    let mut stack = vec![entry];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        for &s in successors.get(&n).into_iter().flatten() {
            stack.push(s);
        }
    }
    seen
}
