//! Hold an emitted nesting to the contract its flow variable depends on.
//!
//! An edge that has to leave more than one construct is staged: it records its destination in one
//! function-scope flow variable, leaves the innermost construct, and every dispatch it passes
//! through reads that variable to decide whether to enter its own continuation or forward one more
//! level. The whole scheme rests on a single property — **every path that reaches a dispatch has
//! already written the flow variable on that path.**
//!
//! Nothing else catches a violation. An `OpVariable` with no initializer reads an undefined value,
//! which is legal SPIR-V: `spirv-val` accepts it, the construct-ownership checks accept it, and the
//! value-flow check in [`crate::native::owned_cfg`] accepts it too, because SSA dominance says
//! nothing about what is in memory. Nor does the result look wrong in shape — it is a properly
//! nested loop with a properly declared selection. The failure appears only at run time, as a
//! dispatch that jumps to whatever a previous iteration left in the slot, and on the first
//! iteration to whatever was already there.
//!
//! So it is checked here, directly, as a definite-assignment analysis over the emitted blocks. A
//! nesting that fails is declined and the function stays on the state-machine constructor, which
//! writes the selector for every one of its edges on that edge.
//!
//! The phi slots this emitter also introduces are deliberately *not* checked this way. A phi slot
//! is written on every original edge into its block, but the emitted paths to the load run through
//! dispatch switches whose arms the flow variable selects, so a path-insensitive analysis cannot
//! tell an unwritten read from routing that never takes that arm — it reports the second as the
//! first. What guards those is the differential execution in [`crate::native::cfg_testkit`], which
//! compares what the function computes rather than which stores it can prove.

use super::super::relooper::block_label;
use crate::spirv_module::{Block, Operand};
use spirv::{Op, Word};
use std::collections::HashMap;

/// Why the emitted nesting reads its flow variable on a path that never wrote it, or `None` if
/// every read is preceded by a write on every path.
pub(super) fn reads_the_flow_variable_unwritten(
    blocks: &[Block],
    flow_variable: Option<Word>,
) -> Option<String> {
    let flow_variable = flow_variable?;
    let graph = FlowGraph::of(blocks, flow_variable)?;
    let written_on_entry = graph.definitely_written_on_entry();
    for (index, block) in blocks.iter().enumerate() {
        let mut written = written_on_entry[index];
        for instruction in &block.instructions {
            if instruction.operands.first() != Some(&Operand::IdRef(flow_variable)) {
                continue;
            }
            match instruction.class.opcode {
                Op::Load if !written => {
                    let label = graph.labels[index];
                    return Some(format!(
                        "nesting dispatches in %{label} on a flow value it may not have written"
                    ));
                }
                Op::Store => written = true,
                _ => {}
            }
        }
    }
    None
}

/// The emitted blocks indexed for the analysis: labels, predecessors, and whether each block writes
/// the flow variable. Built once, so the fixpoint below never rescans instructions or searches by
/// label.
struct FlowGraph {
    labels: Vec<Word>,
    predecessors: Vec<Vec<usize>>,
    stores: Vec<bool>,
}

impl FlowGraph {
    /// `None` when a block has no label — the construction checks reject that anyway, and there is
    /// nothing to verify about blocks that cannot be connected up.
    fn of(blocks: &[Block], flow_variable: Word) -> Option<Self> {
        let labels = blocks.iter().map(block_label).collect::<Option<Vec<_>>>()?;
        let index = labels
            .iter()
            .enumerate()
            .map(|(index, label)| (*label, index))
            .collect::<HashMap<_, _>>();
        let mut predecessors = vec![Vec::new(); blocks.len()];
        for (source, block) in blocks.iter().enumerate() {
            for target in successors(block) {
                if let Some(target) = index.get(&target) {
                    predecessors[*target].push(source);
                }
            }
        }
        let stores = blocks
            .iter()
            .map(|block| {
                block.instructions.iter().any(|instruction| {
                    instruction.class.opcode == Op::Store
                        && instruction.operands.first() == Some(&Operand::IdRef(flow_variable))
                })
            })
            .collect();
        Some(Self {
            labels,
            predecessors,
            stores,
        })
    }

    /// For each block, whether the flow variable is written on *every* path from the entry to it.
    ///
    /// A must-analysis, so the fixpoint starts optimistic — every block but the entry assumed
    /// written — and shrinks. A back edge whose latch does not write is exactly what has to take it
    /// back out.
    fn definitely_written_on_entry(&self) -> Vec<bool> {
        let mut on_entry = (0..self.labels.len())
            .map(|index| index != 0)
            .collect::<Vec<_>>();
        // Each round can only turn entries off, so this terminates.
        let mut changed = true;
        while changed {
            changed = false;
            // The entry block stays unwritten: nothing precedes the first arrival there, so a back
            // edge into it must not be credited with a write that arrival never made.
            for index in 1..self.labels.len() {
                if self.predecessors[index].is_empty() {
                    continue;
                }
                let merged = self.predecessors[index]
                    .iter()
                    .all(|source| on_entry[*source] || self.stores[*source]);
                if on_entry[index] != merged {
                    on_entry[index] = merged;
                    changed = true;
                }
            }
        }
        on_entry
    }
}

fn successors(block: &Block) -> Vec<Word> {
    let Some(terminator) = block.instructions.last() else {
        return Vec::new();
    };
    let operands = match terminator.class.opcode {
        Op::Branch => &terminator.operands[..],
        Op::BranchConditional => &terminator.operands[1..3.min(terminator.operands.len())],
        Op::Switch => &terminator.operands[1..],
        _ => return Vec::new(),
    };
    operands
        .iter()
        .filter_map(|operand| match operand {
            Operand::IdRef(id) => Some(*id),
            _ => None,
        })
        .collect()
}
