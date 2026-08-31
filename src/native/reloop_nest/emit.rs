//! SPIR-V emission for the nesting structurizer.
//!
//! Walks the [`Shape`](super::shape::Shape) tree and writes the nested loop and selection
//! constructs it describes. The point of the nesting is what it does NOT do: it keeps every value
//! in SSA, because the shape tree preserves the original CFG's paths and therefore its dominance.
//! Only `OpPhi` results are register-demoted, because a phi's predecessor identity is exactly what
//! nesting rewrites.
//!
//! # Leaving more than one construct
//!
//! SPIR-V lets a block leave the construct it is in, and no further: a branch out has to target
//! that construct's own merge block, the innermost enclosing loop's continue target, or a
//! `return`-like terminator. Real control flow needs more than that — `while (c) { if (d) { p();
//! if (e) break; } q(); }` leaves a selection AND a loop on one edge.
//!
//! Such an edge is staged. It records its destination in one function-scope flow variable and
//! leaves the innermost construct. Every construct it has to pass through carries a dispatch at its
//! merge that reads the variable, enters its own continuation when the destination belongs to it,
//! and otherwise forwards through the dispatch's merge block — which sits directly inside the next
//! construct out, so forwarding one more level is an ordinary branch to that construct's merge. The
//! analysis pass decides which constructs need a dispatch; a construct no staged edge crosses, and
//! whose body leaves for a single destination, keeps a plain merge and never touches the variable.
//!
//! Contrast the state-machine constructor in `super::super::relooper`, which makes every block a
//! sibling switch case and so must demote every value that crosses a block boundary. On a large
//! function that yields one loop containing the whole program plus thousands of function-scope
//! variables — valid SPIR-V, and a shape a driver's compiler can take unbounded time on.
//!
//! Anything this emitter cannot express is returned as `Err`, and the caller keeps the function on
//! the state-machine constructor. It never emits a guess.

use super::super::relooper::{block_label, decode_term, Term, TypeCtx};
use super::shape::{Graph, Shape};
use crate::spirv_module::{Block, Function, Instruction, Operand};
use spirv::{LoopControl, Op, SelectionControl, StorageClass, Word};
use std::collections::{BTreeSet, HashMap, HashSet};

/// What a branch to some original block becomes in the emitted nesting.
struct Action {
    /// The label to branch to: an arm entry, the continuation's start, a construct merge, or a
    /// loop's continue target.
    label: Word,
    /// The destination to record in the flow variable first, when the construct that handles this
    /// branch reads it.
    flow: Option<u64>,
}

/// One enclosing construct, as the emitter sees it.
struct Frame {
    /// The shape this construct came from, so resolution can consult its facts.
    id: usize,
    /// The original loop entry labels, for a loop frame; empty for a selection frame.
    loop_entries: BTreeSet<Word>,
    continue_label: Option<Word>,
    /// Where a branch leaving this construct goes.
    merge_label: Word,
    /// The original labels this construct's continuation can be entered at.
    next_entries: BTreeSet<Word>,
    /// Whether the merge carries a flow dispatch, so edges leaving here name a destination.
    dispatches: bool,
}

/// What the analysis pass learned about one construct.
#[derive(Default)]
struct ConstructFacts {
    /// The destinations branches inside this construct leave it for.
    destinations: BTreeSet<Word>,
    /// Whether a staged edge bound for an outer construct passes through this one.
    forwards: bool,
    /// Whether some block nested in a further construct leaves this one. SPIR-V lets a nested
    /// construct break to an enclosing loop or switch, never to an enclosing selection, so a
    /// dispatch marked here has to be written as `OpSwitch` even when two arms would read as an
    /// ordinary conditional.
    needs_switch: bool,
}

impl ConstructFacts {
    fn needs_dispatch(&self) -> bool {
        self.forwards || self.destinations.len() > 1
    }
}

/// A construct as the analysis pass sees it: the same stack, without any emitted labels.
struct AnalysisFrame {
    id: usize,
    loop_entries: BTreeSet<Word>,
    next_entries: BTreeSet<Word>,
    is_loop: bool,
}

pub(super) struct Emitter<'a, 'b> {
    tc: &'a mut TypeCtx<'b>,
    blocks: HashMap<Word, Block>,
    terms: HashMap<Word, Term>,
    /// Original block -> the phi results it must load on entry, as (result, variable, type).
    phi_loads: HashMap<Word, Vec<(Word, Word, Word)>>,
    /// Every demoted phi slot, as (block, result, variable), for the promotion pass.
    phi_slots: Vec<(Word, Word, Word)>,
    /// (predecessor, block) -> the values that edge carries into the block's phis.
    phi_stores: HashMap<(Word, Word), Vec<(Word, Word)>>,
    flow_var: Option<Word>,
    flow_ty: Word,
    flow_id: HashMap<Word, u64>,
    facts: HashMap<usize, ConstructFacts>,
    /// Shape id -> the label a branch enters that shape at, for shapes with a single entry.
    head: HashMap<usize, Word>,
    /// Construct shape id -> its merge label.
    merge: HashMap<usize, Word>,
    /// Constructs whose plain merge is a dispatch header, so edges leaving them record their
    /// destination even though the construct itself does not dispatch.
    merge_flow: HashSet<usize>,
    /// Multiple shapes entered by one terminator, so no dispatch block selects their arms.
    fused: HashSet<usize>,
    /// Constructs whose merge must be a block of its own even when one destination would do. A
    /// dispatch header's merge has to be dominated by that header, so it cannot be a label the
    /// enclosing construct also branches straight at.
    forced_dispatch: HashSet<usize>,
    variables: Vec<Instruction>,
    out: Vec<Block>,
}

/// Rebuild `function`'s blocks as nested structured control flow.
pub(super) fn structure_function(
    function: &Function,
    graph: &Graph,
    shape: &Shape,
    tc: &mut TypeCtx<'_>,
) -> Result<Vec<Block>, String> {
    let flow_ty = tc.i32_ty();
    let mut emitter = Emitter {
        tc,
        blocks: HashMap::new(),
        terms: HashMap::new(),
        phi_loads: HashMap::new(),
        phi_slots: Vec::new(),
        phi_stores: HashMap::new(),
        flow_var: None,
        flow_ty,
        flow_id: HashMap::new(),
        facts: HashMap::new(),
        head: HashMap::new(),
        merge: HashMap::new(),
        merge_flow: HashSet::new(),
        fused: HashSet::new(),
        forced_dispatch: HashSet::new(),
        variables: Vec::new(),
        out: Vec::new(),
    };
    emitter.index(function)?;
    emitter.plan(shape, true)?;
    emitter.analyze(shape, graph, &mut Vec::new())?;
    emitter.assign_merges(shape)?;
    emitter.demote_phis(graph)?;

    let entry_label = emitter.head_of(shape)?;
    emitter.emit(shape, &mut Vec::new())?;
    emitter.promote_phi_slots();

    // OpVariable is only legal in a function's first block, and the root shape may be a loop
    // header, so the hoisted declarations get a prologue of their own.
    let prologue_label = emitter.tc.fresh();
    let mut prologue = Block::new();
    prologue.label = Some(Instruction::new(
        Op::Label,
        None,
        Some(prologue_label),
        vec![],
    ));
    prologue.instructions = std::mem::take(&mut emitter.variables);
    prologue.instructions.push(Instruction::new(
        Op::Branch,
        None,
        None,
        vec![Operand::IdRef(entry_label)],
    ));
    let mut blocks = vec![prologue];
    blocks.append(&mut emitter.out);
    Ok(blocks)
}

impl Emitter<'_, '_> {
    // ---------------------------------------------------------------- indexing

    fn index(&mut self, function: &Function) -> Result<(), String> {
        for block in &function.blocks {
            let label = block_label(block).ok_or_else(|| "block without a label".to_string())?;
            let term = block
                .instructions
                .last()
                .and_then(decode_term)
                .ok_or_else(|| "unhandled terminator".to_string())?;
            self.terms.insert(label, term);
            let mut block = block.clone();
            // OpVariable is only legal in a function's first block, and nesting can put any block
            // inside a loop. Hoist the declarations into the prologue this emitter always writes.
            let mut kept = Vec::with_capacity(block.instructions.len());
            for instruction in std::mem::take(&mut block.instructions) {
                match instruction.class.opcode {
                    Op::Variable => self.variables.push(instruction),
                    // The incoming module already carries merge declarations for whatever structure
                    // it had. This pass derives the structure again, and a stale declaration would
                    // no longer sit immediately before the branch it belongs to.
                    Op::SelectionMerge | Op::LoopMerge => {}
                    _ => kept.push(instruction),
                }
            }
            block.instructions = kept;
            self.blocks.insert(label, block);
        }
        Ok(())
    }

    /// Assign the label a branch enters each shape at, and record which dispatches fuse away.
    fn plan(&mut self, shape: &Shape, single_predecessor: bool) -> Result<(), String> {
        match shape {
            Shape::Simple { id, label, next } => {
                self.head.insert(*id, *label);
                if let Some(next) = next {
                    self.plan(next, true)?;
                }
            }
            Shape::Loop {
                id,
                entries,
                inner,
                next,
            } => {
                if entries.len() != 1 {
                    return Err("loop with several entries".to_string());
                }
                let header = self.tc.fresh();
                self.head.insert(*id, header);
                self.plan(inner, true)?;
                if let Some(next) = next {
                    self.plan(next, false)?;
                }
            }
            Shape::Multiple {
                id, handled, next, ..
            } => {
                if handled.is_empty() {
                    return Err("dispatch with no arms".to_string());
                }
                // After a `Simple` the single preceding terminator selects the arm itself, so the
                // dispatch fuses into it. After a construct the arms are selected by that
                // construct's own merge dispatch, and this shape occupies no block of its own.
                if single_predecessor {
                    self.fused.insert(*id);
                } else {
                    // Entered from an enclosing construct's dispatch, so it needs a header of its
                    // own: that header is what makes it a switch construct with a merge block
                    // nested constructs inside its arms are allowed to break to.
                    let header = self.tc.fresh();
                    self.head.insert(*id, header);
                    self.forced_dispatch.insert(*id);
                }
                for (_, arm) in handled {
                    self.plan(arm, true)?;
                }
                if let Some(next) = next {
                    self.plan(next, false)?;
                }
            }
        }
        Ok(())
    }

    fn head_of(&self, shape: &Shape) -> Result<Word, String> {
        self.head
            .get(&shape.id())
            .copied()
            .ok_or_else(|| "shape has no single entry".to_string())
    }

    // ---------------------------------------------------------------- analysis

    /// Classify every branch against the constructs enclosing it, recording per construct which
    /// destinations its body leaves for and whether staged edges pass through it.
    fn analyze(
        &mut self,
        shape: &Shape,
        graph: &Graph,
        stack: &mut Vec<AnalysisFrame>,
    ) -> Result<(), String> {
        match shape {
            Shape::Simple { label, next, .. } => {
                let local = next.as_ref().map(|next| next.entry_labels());
                let targets = graph.succ(*label).to_vec();
                let divergent = targets.len() > 1;
                for target in targets {
                    if local.as_ref().is_some_and(|local| local.contains(&target)) {
                        if let Some(next) = next {
                            self.route(next, target);
                        }
                        continue;
                    }
                    self.classify(target, divergent, stack)?;
                }
                if let Some(next) = next {
                    self.analyze(next, graph, stack)?;
                }
            }
            Shape::Loop {
                id,
                entries,
                inner,
                next,
            } => {
                stack.push(AnalysisFrame {
                    id: *id,
                    loop_entries: entries.clone(),
                    next_entries: next
                        .as_ref()
                        .map(|next| next.entry_labels())
                        .unwrap_or_default(),
                    is_loop: true,
                });
                let result = self.analyze(inner, graph, stack);
                stack.pop();
                result?;
                if let Some(next) = next {
                    self.route_destinations(*id, next);
                    self.analyze(next, graph, stack)?;
                }
            }
            Shape::Multiple {
                id, handled, next, ..
            } => {
                stack.push(AnalysisFrame {
                    id: *id,
                    loop_entries: BTreeSet::new(),
                    next_entries: next
                        .as_ref()
                        .map(|next| next.entry_labels())
                        .unwrap_or_default(),
                    is_loop: false,
                });
                let mut result = Ok(());
                for (_, arm) in handled {
                    result = self.analyze(arm, graph, stack);
                    if result.is_err() {
                        break;
                    }
                }
                stack.pop();
                result?;
                if let Some(next) = next {
                    self.route_destinations(*id, next);
                    self.analyze(next, graph, stack)?;
                }
            }
        }
        Ok(())
    }

    fn route_destinations(&mut self, id: usize, next: &Shape) {
        let destinations = self
            .facts
            .get(&id)
            .map(|facts| facts.destinations.iter().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        for destination in destinations {
            self.route(next, destination);
        }
    }

    /// Record `target` on every dispatch it has to pass through to reach where it is rendered. A
    /// dispatch owns each of its entries, but an entry whose region is shared with another owns no
    /// arm: reaching it means leaving the dispatch again through its merge, which therefore has to
    /// know about it.
    fn route(&mut self, shape: &Shape, target: Word) {
        let Shape::Multiple {
            id, handled, next, ..
        } = shape
        else {
            return;
        };
        if handled.iter().any(|(label, _)| *label == target) {
            return;
        }
        self.facts
            .entry(*id)
            .or_default()
            .destinations
            .insert(target);
        if let Some(next) = next {
            self.route(next, target);
        }
    }

    fn classify(
        &mut self,
        target: Word,
        divergent: bool,
        stack: &[AnalysisFrame],
    ) -> Result<(), String> {
        // A continue may cross selections, but only reaches the innermost enclosing loop;
        // continuing an outer loop is not expressible as one structured branch.
        if let Some(innermost_loop) = stack.iter().rev().find(|frame| frame.is_loop) {
            if innermost_loop.loop_entries.contains(&target) {
                return Ok(());
            }
        }
        if stack
            .iter()
            .any(|frame| frame.loop_entries.contains(&target))
        {
            return Err("branch continues an outer loop".to_string());
        }
        let Some(depth) = stack
            .iter()
            .rposition(|frame| frame.next_entries.contains(&target))
        else {
            return Err("branch target is outside every enclosing construct".to_string());
        };
        self.facts
            .entry(stack[depth].id)
            .or_default()
            .destinations
            .insert(target);
        for frame in &stack[depth + 1..] {
            self.facts.entry(frame.id).or_default().forwards = true;
        }
        if divergent {
            // A divergent terminator declares a selection of its own, so this edge leaves that
            // selection as well as the construct around it. Only a loop or a switch may be broken
            // out of from inside a nested construct.
            if let Some(innermost) = stack.last() {
                self.facts.entry(innermost.id).or_default().needs_switch = true;
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------ merge labels

    /// Give every construct its merge label, now that the analysis says whether it dispatches.
    fn assign_merges(&mut self, shape: &Shape) -> Result<(), String> {
        match shape {
            Shape::Simple { next, .. } => {
                if let Some(next) = next {
                    self.assign_merges(next)?;
                }
            }
            Shape::Loop {
                id, inner, next, ..
            } => {
                // The continuation is assigned first: a plain merge names a label inside it.
                if let Some(next) = next {
                    self.assign_merges(next)?;
                }
                self.assign_construct_merge(*id, next.as_deref())?;
                self.assign_merges(inner)?;
            }
            Shape::Multiple {
                id, handled, next, ..
            } => {
                if let Some(next) = next {
                    self.assign_merges(next)?;
                }
                self.assign_construct_merge(*id, next.as_deref())?;
                for (_, arm) in handled {
                    self.assign_merges(arm)?;
                }
            }
        }
        Ok(())
    }

    fn assign_construct_merge(&mut self, id: usize, next: Option<&Shape>) -> Result<(), String> {
        let dispatches = self.dispatches(id);
        let destination = self
            .facts
            .entry(id)
            .or_default()
            .destinations
            .iter()
            .copied()
            .next();
        let label = if dispatches {
            self.tc.fresh()
        } else {
            match (destination, next) {
                // One way out: the merge is simply where that destination is rendered, so the
                // construct's body branches straight at it.
                (Some(destination), Some(next)) => {
                    let (label, needs_flow) = self.entry_label(next, destination)?;
                    if needs_flow {
                        self.merge_flow.insert(id);
                    }
                    label
                }
                // Nothing leaves this construct normally; its declared merge is unreachable.
                _ => self.fresh_unreachable(),
            }
        };
        self.merge.insert(id, label);
        Ok(())
    }

    /// Where a branch enters `shape` at `target`, which must be one of its entries, and whether
    /// the edge has to record `target` in the flow variable first — which it does exactly when the
    /// label is a dispatch header that reads the variable to pick its arm.
    fn entry_label(&self, shape: &Shape, target: Word) -> Result<(Word, bool), String> {
        match shape {
            Shape::Simple { label, .. } if *label == target => Ok((*label, false)),
            Shape::Loop { entries, id, .. } if entries.contains(&target) => self
                .head
                .get(id)
                .copied()
                .map(|header| (header, false))
                .ok_or_else(|| "loop without a header".to_string()),
            Shape::Multiple {
                id, handled, next, ..
            } => {
                if !self.fused.contains(id) {
                    // Every entry goes through the header: an arm is one of its cases, and anything
                    // further on leaves again through the dispatch at this construct's merge. Making
                    // it the single way in is what keeps the header dominating its own merge.
                    return self
                        .head
                        .get(id)
                        .copied()
                        .map(|header| (header, true))
                        .ok_or_else(|| "dispatch without a header".to_string());
                }
                if let Some((_, arm)) = handled.iter().find(|(label, _)| *label == target) {
                    // The arm is a shape, not a block: when it is a loop, control enters at the
                    // loop header this emitter synthesized, never at the original block.
                    return Ok((self.head_of(arm)?, false));
                }
                match next {
                    Some(next) => self.entry_label(next, target),
                    None => Err("destination is not an entry of the continuation".to_string()),
                }
            }
            _ => Err("destination is not an entry of the continuation".to_string()),
        }
    }

    fn merge_of(&self, id: usize) -> Result<Word, String> {
        self.merge
            .get(&id)
            .copied()
            .ok_or_else(|| "construct without a merge".to_string())
    }

    fn dispatches(&self, id: usize) -> bool {
        self.forced_dispatch.contains(&id)
            || self
                .facts
                .get(&id)
                .is_some_and(ConstructFacts::needs_dispatch)
    }

    /// A block that exists only to be a declared merge no path reaches.
    fn fresh_unreachable(&mut self) -> Word {
        let label = self.tc.fresh();
        let mut block = Block::new();
        block.label = Some(Instruction::new(Op::Label, None, Some(label), vec![]));
        block
            .instructions
            .push(Instruction::new(Op::Unreachable, None, None, vec![]));
        self.out.push(block);
        label
    }

    // ------------------------------------------------------------------- phis

    /// Replace every `OpPhi` with a function-scope variable: the incoming edges store, the block
    /// loads. The nesting rewrites which block physically branches into a merge, so a phi's
    /// predecessor list is the one thing it cannot carry through unchanged.
    fn demote_phis(&mut self, graph: &Graph) -> Result<(), String> {
        let labels = self.blocks.keys().copied().collect::<Vec<_>>();
        for label in labels {
            let phis = self
                .blocks
                .get(&label)
                .map(|block| {
                    block
                        .instructions
                        .iter()
                        .filter(|inst| inst.class.opcode == Op::Phi)
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if phis.is_empty() {
                continue;
            }
            for phi in &phis {
                let result = phi
                    .result_id
                    .ok_or_else(|| "phi without a result".to_string())?;
                let ty = phi
                    .result_type
                    .ok_or_else(|| "phi without a result type".to_string())?;
                let pointer = self.tc.ptr_function(ty);
                let var = self.tc.fresh();
                self.variables.push(Instruction::new(
                    Op::Variable,
                    Some(pointer),
                    Some(var),
                    vec![Operand::StorageClass(StorageClass::Function)],
                ));
                self.phi_loads
                    .entry(label)
                    .or_default()
                    .push((result, var, ty));
                self.phi_slots.push((label, result, var));
                let mut index = 0;
                while index + 1 < phi.operands.len() {
                    let (Operand::IdRef(value), Operand::IdRef(predecessor)) =
                        (&phi.operands[index], &phi.operands[index + 1])
                    else {
                        return Err("phi operand shape".to_string());
                    };
                    // A predecessor no path reaches keeps no edge in the emitted nesting.
                    if graph.successors.contains_key(predecessor) {
                        self.phi_stores
                            .entry((*predecessor, label))
                            .or_default()
                            .push((var, *value));
                    }
                    index += 2;
                }
            }
            if let Some(block) = self.blocks.get_mut(&label) {
                block
                    .instructions
                    .retain(|inst| inst.class.opcode != Op::Phi);
            }
        }
        Ok(())
    }

    // -------------------------------------------------------------- resolution

    fn flow_variable(&mut self) -> Word {
        if let Some(var) = self.flow_var {
            return var;
        }
        let pointer = self.tc.ptr_function(self.flow_ty);
        let var = self.tc.fresh();
        self.variables.push(Instruction::new(
            Op::Variable,
            Some(pointer),
            Some(var),
            vec![Operand::StorageClass(StorageClass::Function)],
        ));
        self.flow_var = Some(var);
        var
    }

    fn flow_id_of(&mut self, label: Word) -> u64 {
        let next = self.flow_id.len() as u64 + 1;
        *self.flow_id.entry(label).or_insert(next)
    }

    /// Where a branch to `target` goes. `local` is the shape that follows the block being emitted:
    /// its entries are reached by falling through, without leaving any construct.
    fn resolve(
        &mut self,
        target: Word,
        local: Option<&Shape>,
        stack: &[Frame],
    ) -> Result<Action, String> {
        if let Some(local) = local {
            if local.entry_labels().contains(&target) {
                return self.enter(local, target);
            }
        }
        if let Some(frame) = stack
            .iter()
            .rev()
            .find(|frame| frame.continue_label.is_some())
        {
            if frame.loop_entries.contains(&target) {
                let label = frame
                    .continue_label
                    .ok_or_else(|| "loop frame without a continue target".to_string())?;
                return Ok(Action { label, flow: None });
            }
        }
        let depth = stack
            .iter()
            .rposition(|frame| frame.next_entries.contains(&target))
            .ok_or_else(|| "branch target is outside every enclosing construct".to_string())?;
        let records_flow = stack[depth].dispatches || self.merge_flow.contains(&stack[depth].id);
        let flow = records_flow.then(|| self.flow_id_of(target));
        let label = stack
            .last()
            .ok_or_else(|| "branch leaves a construct at the top level".to_string())?
            .merge_label;
        Ok(Action { label, flow })
    }

    /// How to enter `shape` at `target`, which is one of its entries.
    fn enter(&mut self, shape: &Shape, target: Word) -> Result<Action, String> {
        match shape {
            Shape::Simple { .. } | Shape::Loop { .. } => {
                let (label, _) = self.entry_label(shape, target)?;
                Ok(Action { label, flow: None })
            }
            Shape::Multiple { id, handled, .. } => {
                if handled.iter().any(|(label, _)| *label == target) {
                    let (label, needs_flow) = self.entry_label(shape, target)?;
                    let flow = needs_flow.then(|| self.flow_id_of(target));
                    return Ok(Action { label, flow });
                }
                // The destination is past this dispatch: leave it through its merge, saying which
                // continuation was meant when that merge reads the flow variable.
                let label = self.merge_of(*id)?;
                let flow = self.dispatches(*id).then(|| self.flow_id_of(target));
                Ok(Action { label, flow })
            }
        }
    }

    // ---------------------------------------------------------------- emission

    fn emit(&mut self, shape: &Shape, stack: &mut Vec<Frame>) -> Result<(), String> {
        match shape {
            Shape::Simple { label, next, .. } => self.emit_simple(*label, next.as_deref(), stack),
            Shape::Loop {
                id,
                entries,
                inner,
                next,
            } => self.emit_loop(*id, entries, inner, next.as_deref(), stack),
            Shape::Multiple {
                id, handled, next, ..
            } => self.emit_multiple(*id, handled, next.as_deref(), stack),
        }
    }

    fn emit_simple(
        &mut self,
        label: Word,
        next: Option<&Shape>,
        stack: &mut Vec<Frame>,
    ) -> Result<(), String> {
        let source = self
            .blocks
            .get(&label)
            .cloned()
            .ok_or_else(|| "missing block".to_string())?;
        let term = self
            .terms
            .get(&label)
            .cloned()
            .ok_or_else(|| "missing terminator".to_string())?;
        let mut block = Block::new();
        block.label = source.label.clone();
        for (result, var, ty) in self.phi_loads.get(&label).cloned().unwrap_or_default() {
            block.instructions.push(Instruction::new(
                Op::Load,
                Some(ty),
                Some(result),
                vec![Operand::IdRef(var)],
            ));
        }
        let body = source.instructions.len().saturating_sub(1);
        block
            .instructions
            .extend(source.instructions.iter().take(body).cloned());
        self.emit_terminator(label, term, &mut block, next, stack)?;
        self.out.push(block);
        if let Some(next) = next {
            self.emit(next, stack)?;
        }
        Ok(())
    }

    fn emit_terminator(
        &mut self,
        label: Word,
        term: Term,
        block: &mut Block,
        next: Option<&Shape>,
        stack: &[Frame],
    ) -> Result<(), String> {
        match term {
            Term::Return => {
                block
                    .instructions
                    .push(Instruction::new(Op::Return, None, None, vec![]));
                Ok(())
            }
            Term::ReturnValue(value) => {
                block.instructions.push(Instruction::new(
                    Op::ReturnValue,
                    None,
                    None,
                    vec![Operand::IdRef(value)],
                ));
                Ok(())
            }
            Term::Unreachable => {
                block
                    .instructions
                    .push(Instruction::new(Op::Unreachable, None, None, vec![]));
                Ok(())
            }
            Term::Kill(instruction) => {
                block.instructions.push(instruction);
                Ok(())
            }
            Term::Branch(target) => {
                let action = self.resolve(target, next, stack)?;
                self.finish_single_edge(label, target, action, block);
                Ok(())
            }
            Term::BranchCond(condition, on_true, on_false) => {
                if on_true == on_false {
                    let action = self.resolve(on_true, next, stack)?;
                    self.finish_single_edge(label, on_true, action, block);
                    return Ok(());
                }
                let true_label = self.edge_label(label, on_true, next, stack)?;
                let false_label = self.edge_label(label, on_false, next, stack)?;
                let merge = self.selection_merge(next)?;
                block.instructions.push(Instruction::new(
                    Op::SelectionMerge,
                    None,
                    None,
                    vec![
                        Operand::IdRef(merge),
                        Operand::SelectionControl(SelectionControl::NONE),
                    ],
                ));
                if self.needs_switch(next) {
                    // Something nested in one of these arms breaks out past this construct, and
                    // SPIR-V allows that only for a loop or a switch. Widen the condition to a
                    // selector so the two-way choice is a switch construct.
                    let selector = self.widen_condition(condition, block);
                    block.instructions.push(Instruction::new(
                        Op::Switch,
                        None,
                        None,
                        vec![
                            Operand::IdRef(selector),
                            Operand::IdRef(false_label),
                            Operand::LiteralBit32(1),
                            Operand::IdRef(true_label),
                        ],
                    ));
                    return Ok(());
                }
                block.instructions.push(Instruction::new(
                    Op::BranchConditional,
                    None,
                    None,
                    vec![
                        Operand::IdRef(condition),
                        Operand::IdRef(true_label),
                        Operand::IdRef(false_label),
                    ],
                ));
                Ok(())
            }
            Term::Switch(selector, default, cases) => {
                let default_label = self.edge_label(label, default, next, stack)?;
                let mut operands = vec![Operand::IdRef(selector), Operand::IdRef(default_label)];
                for (literal, target) in &cases {
                    let case_label = self.edge_label(label, *target, next, stack)?;
                    operands.push(if *literal > u64::from(u32::MAX) {
                        Operand::LiteralBit64(*literal)
                    } else {
                        Operand::LiteralBit32(*literal as u32)
                    });
                    operands.push(Operand::IdRef(case_label));
                }
                let merge = self.selection_merge(next)?;
                block.instructions.push(Instruction::new(
                    Op::SelectionMerge,
                    None,
                    None,
                    vec![
                        Operand::IdRef(merge),
                        Operand::SelectionControl(SelectionControl::NONE),
                    ],
                ));
                block
                    .instructions
                    .push(Instruction::new(Op::Switch, None, None, operands));
                Ok(())
            }
        }
    }

    /// Whether the construct a divergent terminator heads has to be written as a switch.
    fn needs_switch(&self, next: Option<&Shape>) -> bool {
        matches!(next, Some(Shape::Multiple { id, .. })
            if self.facts.get(id).is_some_and(|facts| facts.needs_switch))
    }

    /// A `0`/`1` selector for a boolean condition, so a two-way choice can head a switch construct.
    fn widen_condition(&mut self, condition: Word, block: &mut Block) -> Word {
        let one = self.tc.int_const(self.flow_ty, 1);
        let zero = self.tc.int_const(self.flow_ty, 0);
        let selector = self.tc.fresh();
        let selection = Instruction::new(
            Op::Select,
            Some(self.flow_ty),
            Some(selector),
            vec![
                Operand::IdRef(condition),
                Operand::IdRef(one),
                Operand::IdRef(zero),
            ],
        );
        // OpSelectionMerge must stay immediately before the branch, so the widening goes in front.
        let position = block.instructions.len().saturating_sub(1);
        block.instructions.insert(position, selection);
        selector
    }

    /// The declared merge for a divergent terminator: where the shape that follows converges. A
    /// terminator whose arms all leave the enclosing construct declares a merge nothing reaches.
    fn selection_merge(&mut self, next: Option<&Shape>) -> Result<Word, String> {
        match next {
            Some(Shape::Multiple { id, .. }) => self.merge_of(*id),
            Some(shape) => self.head_of(shape),
            None => Ok(self.fresh_unreachable()),
        }
    }

    /// The label one outgoing edge branches to, creating a helper block when the edge has to store
    /// phi values or a flow destination before branching.
    fn edge_label(
        &mut self,
        from: Word,
        target: Word,
        next: Option<&Shape>,
        stack: &[Frame],
    ) -> Result<Word, String> {
        let action = self.resolve(target, next, stack)?;
        let stores = self
            .phi_stores
            .get(&(from, target))
            .cloned()
            .unwrap_or_default();
        if stores.is_empty() && action.flow.is_none() {
            return Ok(action.label);
        }
        let helper = self.tc.fresh();
        let mut block = Block::new();
        block.label = Some(Instruction::new(Op::Label, None, Some(helper), vec![]));
        self.write_edge_stores(&stores, action.flow, &mut block);
        block.instructions.push(Instruction::new(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(action.label)],
        ));
        self.out.push(block);
        Ok(helper)
    }

    /// Finish a block whose terminator has one outgoing edge: its stores go inline.
    fn finish_single_edge(&mut self, from: Word, target: Word, action: Action, block: &mut Block) {
        let stores = self
            .phi_stores
            .get(&(from, target))
            .cloned()
            .unwrap_or_default();
        self.write_edge_stores(&stores, action.flow, block);
        block.instructions.push(Instruction::new(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(action.label)],
        ));
    }

    fn write_edge_stores(&mut self, stores: &[(Word, Word)], flow: Option<u64>, block: &mut Block) {
        for (var, value) in stores {
            block.instructions.push(Instruction::new(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(*var), Operand::IdRef(*value)],
            ));
        }
        if let Some(flow) = flow {
            let variable = self.flow_variable();
            let constant = self.tc.int_const(self.flow_ty, flow);
            block.instructions.push(Instruction::new(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(variable), Operand::IdRef(constant)],
            ));
        }
    }

    fn emit_loop(
        &mut self,
        id: usize,
        entries: &BTreeSet<Word>,
        inner: &Shape,
        next: Option<&Shape>,
        stack: &mut Vec<Frame>,
    ) -> Result<(), String> {
        let header = self
            .head
            .get(&id)
            .copied()
            .ok_or_else(|| "loop without a header".to_string())?;
        let continue_label = self.tc.fresh();
        let merge_label = self.merge_of(id)?;
        let body_start = self.head_of(inner)?;
        let mut head = Block::new();
        head.label = Some(Instruction::new(Op::Label, None, Some(header), vec![]));
        head.instructions.push(Instruction::new(
            Op::LoopMerge,
            None,
            None,
            vec![
                Operand::IdRef(merge_label),
                Operand::IdRef(continue_label),
                Operand::LoopControl(LoopControl::NONE),
            ],
        ));
        head.instructions.push(Instruction::new(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(body_start)],
        ));
        self.out.push(head);

        stack.push(Frame {
            id,
            loop_entries: entries.clone(),
            continue_label: Some(continue_label),
            merge_label,
            next_entries: next.map(Shape::entry_labels).unwrap_or_default(),
            dispatches: self.dispatches(id),
        });
        let result = self.emit(inner, stack);
        stack.pop();
        result?;

        let mut latch = Block::new();
        latch.label = Some(Instruction::new(
            Op::Label,
            None,
            Some(continue_label),
            vec![],
        ));
        latch.instructions.push(Instruction::new(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(header)],
        ));
        self.out.push(latch);

        self.emit_construct_tail(id, next, stack)
    }

    /// A dispatch occupies no block of its own: it is entered either by the terminator it fused
    /// into or by the enclosing construct's merge dispatch, both of which branch straight at the
    /// arm they selected. Emitting it is therefore emitting the arms, then the continuation.
    fn emit_multiple(
        &mut self,
        id: usize,
        handled: &[(Word, Shape)],
        next: Option<&Shape>,
        stack: &mut Vec<Frame>,
    ) -> Result<(), String> {
        let merge_label = self.merge_of(id)?;
        if !self.fused.contains(&id) {
            self.emit_dispatch_header(id, handled, merge_label)?;
        }
        stack.push(Frame {
            id,
            loop_entries: BTreeSet::new(),
            continue_label: None,
            merge_label,
            next_entries: next.map(Shape::entry_labels).unwrap_or_default(),
            dispatches: self.dispatches(id),
        });
        let mut result = Ok(());
        for (_, arm) in handled {
            result = self.emit(arm, stack);
            if result.is_err() {
                break;
            }
        }
        stack.pop();
        result?;

        self.emit_construct_tail(id, next, stack)
    }

    /// The header of a dispatch entered from an enclosing construct rather than fused into one
    /// terminator. Making it a real `OpSwitch` header is what gives the dispatch a merge block that
    /// constructs nested inside its arms are allowed to break to.
    fn emit_dispatch_header(
        &mut self,
        id: usize,
        handled: &[(Word, Shape)],
        merge_label: Word,
    ) -> Result<(), String> {
        let header = self
            .head
            .get(&id)
            .copied()
            .ok_or_else(|| "dispatch without a header".to_string())?;
        let variable = self.flow_variable();
        let loaded = self.tc.fresh();
        let mut block = Block::new();
        block.label = Some(Instruction::new(Op::Label, None, Some(header), vec![]));
        block.instructions.push(Instruction::new(
            Op::Load,
            Some(self.flow_ty),
            Some(loaded),
            vec![Operand::IdRef(variable)],
        ));
        // The default is this construct's merge: a destination past the arms leaves again there,
        // where the dispatch that routes it on already has a case for it.
        let mut operands = vec![Operand::IdRef(loaded), Operand::IdRef(merge_label)];
        for (arm, shape) in handled {
            let flow = self.flow_id_of(*arm);
            let head = self.head_of(shape)?;
            operands.push(Operand::LiteralBit32(flow as u32));
            operands.push(Operand::IdRef(head));
        }
        block.instructions.push(Instruction::new(
            Op::SelectionMerge,
            None,
            None,
            vec![
                Operand::IdRef(merge_label),
                Operand::SelectionControl(SelectionControl::NONE),
            ],
        ));
        block
            .instructions
            .push(Instruction::new(Op::Switch, None, None, operands));
        self.out.push(block);
        Ok(())
    }

    /// Put back every demoted phi the nesting did not actually disturb.
    ///
    /// A phi has to be demoted while the structure is being derived, because the nesting is free to
    /// rewrite which block physically branches into a merge. Most merges come through unchanged:
    /// each incoming edge still ends in a block that branches straight at the merge, and the merge
    /// has no other predecessor. That is exactly the shape of an `OpPhi`, so the slot is folded
    /// back — the variable, its stores and its load all disappear.
    ///
    /// The check is on the emitted blocks, so it needs no model of how they were produced: an edge
    /// that got a helper block of its own is as good a phi predecessor as the original block.
    fn promote_phi_slots(&mut self) {
        let mut predecessors: HashMap<Word, Vec<Word>> = HashMap::new();
        for block in &self.out {
            let Some(label) = block_label(block) else {
                continue;
            };
            let Some(term) = block.instructions.last().and_then(decode_term) else {
                continue;
            };
            let targets = match term {
                Term::Branch(target) => vec![target],
                Term::BranchCond(_, on_true, on_false) => vec![on_true, on_false],
                Term::Switch(_, default, cases) => {
                    let mut targets = vec![default];
                    targets.extend(cases.into_iter().map(|(_, label)| label));
                    targets
                }
                Term::Return | Term::ReturnValue(_) | Term::Unreachable | Term::Kill(_) => vec![],
            };
            for target in targets {
                let list = predecessors.entry(target).or_default();
                if !list.contains(&label) {
                    list.push(label);
                }
            }
        }
        let index = self
            .out
            .iter()
            .enumerate()
            .filter_map(|(position, block)| Some((block_label(block)?, position)))
            .collect::<HashMap<_, _>>();

        let slots = std::mem::take(&mut self.phi_slots);
        let mut promoted = HashSet::new();
        for (block, result, var) in slots {
            let Some(incoming) = predecessors.get(&block) else {
                continue;
            };
            // Every predecessor has to reach this block by an unconditional branch, so the value it
            // carries is unambiguous, and it has to store this slot exactly once.
            let mut operands = Vec::with_capacity(incoming.len() * 2);
            let mut carriers = Vec::with_capacity(incoming.len());
            let mut usable = !incoming.is_empty();
            for predecessor in incoming {
                let Some(source) = index.get(predecessor).map(|position| &self.out[*position])
                else {
                    usable = false;
                    break;
                };
                if !matches!(
                    source.instructions.last().and_then(decode_term),
                    Some(Term::Branch(_))
                ) {
                    usable = false;
                    break;
                }
                // The edge's store may sit one or more pass-through blocks back — a loop's continue
                // target, for instance, is written by the latch that branches into it. Walking a
                // chain of single-predecessor branch-only blocks is safe: each dominates the next,
                // so the value is still available where the phi needs it.
                let mut carrier = *predecessor;
                let value = loop {
                    let Some(&position) = index.get(&carrier) else {
                        break None;
                    };
                    let mut stored =
                        self.out[position]
                            .instructions
                            .iter()
                            .filter_map(|instruction| {
                                (instruction.class.opcode == Op::Store
                                    && instruction.operands.first() == Some(&Operand::IdRef(var)))
                                .then(|| instruction.operands.get(1).cloned())
                                .flatten()
                            });
                    match (stored.next(), stored.next()) {
                        (Some(value), None) => break Some(value),
                        (Some(_), Some(_)) => break None,
                        (None, _) => {}
                    }
                    let ancestors = predecessors.get(&carrier).map(Vec::as_slice).unwrap_or(&[]);
                    let [only] = ancestors else {
                        break None;
                    };
                    if !matches!(
                        index
                            .get(only)
                            .and_then(|position| self.out[*position].instructions.last())
                            .and_then(decode_term),
                        Some(Term::Branch(_))
                    ) {
                        break None;
                    }
                    carrier = *only;
                };
                let Some(value) = value else {
                    usable = false;
                    break;
                };
                operands.push(value);
                operands.push(Operand::IdRef(*predecessor));
                carriers.push(carrier);
            }
            if !usable {
                continue;
            }
            let Some(&position) = index.get(&block) else {
                continue;
            };
            let Some(load) = self.out[position]
                .instructions
                .iter()
                .position(|instruction| {
                    instruction.result_id == Some(result) && instruction.class.opcode == Op::Load
                })
            else {
                continue;
            };
            let ty = self.out[position].instructions[load].result_type;
            self.out[position].instructions.remove(load);
            self.out[position]
                .instructions
                .insert(0, Instruction::new(Op::Phi, ty, Some(result), operands));
            for carrier in carriers {
                if let Some(&source) = index.get(&carrier) {
                    self.out[source].instructions.retain(|instruction| {
                        instruction.class.opcode != Op::Store
                            || instruction.operands.first() != Some(&Operand::IdRef(var))
                    });
                }
            }
            promoted.insert(var);
        }
        self.variables
            .retain(|variable| !variable.result_id.is_some_and(|id| promoted.contains(&id)));
    }

    /// Write whatever follows a construct.
    ///
    /// Without a dispatch that is just the continuation, which already starts at the construct's
    /// merge. With one, the merge reads the flow variable and either enters the continuation at the
    /// destination it names or forwards outward through its own merge block — the one block that is
    /// directly inside the next construct out, and so may branch at that construct's merge.
    fn emit_construct_tail(
        &mut self,
        id: usize,
        next: Option<&Shape>,
        stack: &mut Vec<Frame>,
    ) -> Result<(), String> {
        if !self.dispatches(id) {
            return match next {
                Some(next) => self.emit(next, stack),
                None => Ok(()),
            };
        }
        let dispatch = self.merge_of(id)?;
        let destinations = self
            .facts
            .get(&id)
            .map(|facts| facts.destinations.iter().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        let Some(next) = next else {
            // Nothing follows this construct, so every edge arriving at its merge is a staged one
            // still on its way out. Forward without reading the variable: there is no destination
            // here to compare against.
            let mut block = Block::new();
            block.label = Some(Instruction::new(Op::Label, None, Some(dispatch), vec![]));
            match stack.last().map(|frame| frame.merge_label) {
                Some(outer) => block.instructions.push(Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(outer)],
                )),
                None => {
                    block
                        .instructions
                        .push(Instruction::new(Op::Unreachable, None, None, vec![]))
                }
            }
            self.out.push(block);
            return Ok(());
        };
        let after = self.tc.fresh();
        let mut cases: Vec<(u64, Word)> = Vec::with_capacity(destinations.len());
        for destination in destinations {
            let (label, _) = self.entry_label(next, destination)?;
            cases.push((self.flow_id_of(destination), label));
        }
        let variable = self.flow_variable();
        let loaded = self.tc.fresh();
        let mut block = Block::new();
        block.label = Some(Instruction::new(Op::Label, None, Some(dispatch), vec![]));
        block.instructions.push(Instruction::new(
            Op::Load,
            Some(self.flow_ty),
            Some(loaded),
            vec![Operand::IdRef(variable)],
        ));
        // The default forwards: a destination this construct does not own belongs further out, and
        // `after` is where the dispatch converges, directly inside the next construct.
        let mut operands = vec![Operand::IdRef(loaded), Operand::IdRef(after)];
        for (flow, label) in &cases {
            operands.push(Operand::LiteralBit32(*flow as u32));
            operands.push(Operand::IdRef(*label));
        }
        block.instructions.push(Instruction::new(
            Op::SelectionMerge,
            None,
            None,
            vec![
                Operand::IdRef(after),
                Operand::SelectionControl(SelectionControl::NONE),
            ],
        ));
        block
            .instructions
            .push(Instruction::new(Op::Switch, None, None, operands));
        self.out.push(block);

        // The continuation is written inside the dispatch's selection, so anything in it that
        // leaves this nesting level converges at `after` instead of branching past it.
        let restore = stack.last().map(|frame| frame.merge_label);
        if let Some(frame) = stack.last_mut() {
            frame.merge_label = after;
        }
        let result = self.emit(next, stack);
        if let (Some(frame), Some(restore)) = (stack.last_mut(), restore) {
            frame.merge_label = restore;
        }
        result?;

        let mut tail = Block::new();
        tail.label = Some(Instruction::new(Op::Label, None, Some(after), vec![]));
        match restore {
            Some(outer) => tail.instructions.push(Instruction::new(
                Op::Branch,
                None,
                None,
                vec![Operand::IdRef(outer)],
            )),
            // Nothing encloses this construct, so no staged edge can be pending here.
            None => tail
                .instructions
                .push(Instruction::new(Op::Unreachable, None, None, vec![])),
        }
        self.out.push(tail);
        Ok(())
    }
}
