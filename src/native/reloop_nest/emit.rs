//! SPIR-V emission for the nesting structurizer.
//!
//! Walks the [`Shape`](super::shape::Shape) tree and writes the nested loop / selection constructs
//! it describes. The point of the nesting is what it does NOT do: it keeps every value in SSA,
//! because the shape tree preserves the original CFG's paths and therefore its dominance. Only
//! `OpPhi` results are register-demoted, because a phi's predecessor identity is what the nesting
//! rewrites.
//!
//! Contrast with the state-machine constructor in `super::super::relooper`, which makes every block
//! a sibling switch case and so has to demote every value that crosses a block boundary. On a large
//! function that produces one loop containing the whole program plus thousands of function-scope
//! variables — a shape that is valid, but that drives a driver's SSA reconstruction and register
//! allocator superlinear.
//!
//! Anything this emitter cannot express is returned as `Err`, and the caller keeps the function on
//! the state-machine constructor. It never emits a guess.

use super::super::relooper::{block_label, decode_term, Term, TypeCtx};
use super::shape::{Graph, Shape};
use crate::spirv_module::{Block, Function, Instruction, Operand};
use spirv::{LoopControl, Op, SelectionControl, StorageClass, Word};
use std::collections::{BTreeSet, HashMap, HashSet};

/// What a branch to some original block becomes in the emitted nesting.
enum Action {
    /// Branch straight at a label: an arm entry, the next shape's start, a construct merge, or a
    /// loop's continue target.
    Goto(Word),
    /// Same, but the destination is a dispatch that reads the flow variable, so the edge has to say
    /// which entry it meant first.
    Dispatch { label: Word, flow: u64 },
}

/// One enclosing SPIR-V construct. Resolution walks this stack outward to classify a branch.
struct Frame {
    /// Continue targets: the original loop entry labels. Empty for a selection frame.
    loop_entries: BTreeSet<Word>,
    continue_label: Option<Word>,
    merge_label: Word,
    /// The original labels reachable by leaving this construct through its merge.
    next_entries: BTreeSet<Word>,
    /// Whether the merge block is a flow dispatch rather than a single continuation.
    merge_dispatches: bool,
}

pub(super) struct Emitter<'a, 'b> {
    tc: &'a mut TypeCtx<'b>,
    blocks: HashMap<Word, Block>,
    terms: HashMap<Word, Term>,
    /// Phi result id -> (variable, result type).
    phi_var: HashMap<Word, (Word, Word)>,
    /// Original block -> the phi entries it must load on entry.
    phi_loads: HashMap<Word, Vec<(Word, Word, Word)>>,
    /// (predecessor, block) -> the values that edge carries into the block's phis.
    phi_stores: HashMap<(Word, Word), Vec<(Word, Word)>>,
    flow_var: Option<Word>,
    flow_ty: Word,
    flow_id: HashMap<Word, u64>,
    /// Shape id -> the label its emission starts at.
    start: HashMap<usize, Word>,
    /// Multiple shapes entered from exactly one terminator, so no dispatch block is needed.
    fused: HashSet<usize>,
    variables: Vec<Instruction>,
    out: Vec<Block>,
}

/// Rebuild `function`'s blocks as nested structured control flow. Returns `Err` with a short reason
/// when the shape tree contains something this emitter does not express; the function is left
/// untouched in that case.
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
        phi_var: HashMap::new(),
        phi_loads: HashMap::new(),
        phi_stores: HashMap::new(),
        flow_var: None,
        flow_ty,
        flow_id: HashMap::new(),
        start: HashMap::new(),
        fused: HashSet::new(),
        variables: Vec::new(),
        out: Vec::new(),
    };
    emitter.index(function)?;
    emitter.plan(shape, true);
    emitter.demote_phis(graph)?;

    let entry_label = emitter.start_of(shape);
    let mut stack = Vec::new();
    emitter.emit(shape, &mut stack)?;

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
            for instruction in block.instructions.drain(..) {
                if instruction.class.opcode == Op::Variable {
                    self.variables.push(instruction);
                } else {
                    kept.push(instruction);
                }
            }
            block.instructions = kept;
            self.blocks.insert(label, block);
        }
        Ok(())
    }

    /// Assign a start label to every shape and decide which dispatches can be fused away.
    ///
    /// A `Multiple` needs a dispatch block only when more than one terminator can enter it. That is
    /// exactly when it follows a `Loop` or another `Multiple`: after a `Simple`, the single
    /// preceding block's own terminator selects the arm, so the branch goes straight to the arm and
    /// the flow variable is never involved.
    fn plan(&mut self, shape: &Shape, single_predecessor: bool) {
        match shape {
            Shape::Simple { id, label, next } => {
                self.start.insert(*id, *label);
                if let Some(next) = next {
                    self.plan(next, true);
                }
            }
            Shape::Loop {
                id, inner, next, ..
            } => {
                let header = self.tc.fresh();
                self.start.insert(*id, header);
                self.plan(inner, true);
                if let Some(next) = next {
                    self.plan(next, false);
                }
            }
            Shape::Multiple { id, handled, next } => {
                if single_predecessor && !handled.is_empty() {
                    self.fused.insert(*id);
                    // A fused dispatch starts wherever its continuation does; the arms are branched
                    // to directly, so the shape itself never occupies a block.
                    let label = match next {
                        Some(next) => {
                            self.plan(next, false);
                            self.start_of(next)
                        }
                        None => self.fresh_unreachable(),
                    };
                    self.start.insert(*id, label);
                } else {
                    let dispatch = self.tc.fresh();
                    self.start.insert(*id, dispatch);
                    if let Some(next) = next {
                        self.plan(next, false);
                    }
                }
                for (_, arm) in handled {
                    self.plan(arm, true);
                }
            }
        }
    }

    fn start_of(&self, shape: &Shape) -> Word {
        self.start
            .get(&shape.id())
            .copied()
            .expect("every shape was planned")
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

    /// Replace every `OpPhi` with a function-scope variable: the incoming edges store, the block
    /// loads. The nesting rewrites which block physically branches into a merge, so a phi's
    /// predecessor list is the one thing it cannot carry through unchanged.
    fn demote_phis(&mut self, graph: &Graph) -> Result<(), String> {
        let labels = self.blocks.keys().copied().collect::<Vec<_>>();
        for label in labels {
            let Some(block) = self.blocks.get(&label) else {
                continue;
            };
            let phis = block
                .instructions
                .iter()
                .filter(|inst| inst.class.opcode == Op::Phi)
                .cloned()
                .collect::<Vec<_>>();
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
                self.phi_var.insert(result, (var, ty));
                self.phi_loads
                    .entry(label)
                    .or_default()
                    .push((result, var, ty));
                let mut index = 0;
                while index + 1 < phi.operands.len() {
                    let (Operand::IdRef(value), Operand::IdRef(predecessor)) =
                        (&phi.operands[index], &phi.operands[index + 1])
                    else {
                        return Err("phi operand shape".to_string());
                    };
                    // Unreachable predecessors keep no edge in the emitted nesting.
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

    /// Where a branch to `target` goes, given the constructs currently open. `local` is the shape
    /// that follows the block being emitted; its entries are reached by falling through rather than
    /// by leaving a construct.
    fn resolve(
        &mut self,
        target: Word,
        local: Option<&Shape>,
        stack: &[Frame],
    ) -> Result<Action, String> {
        if let Some(local) = local {
            if let Some(action) = self.enter(target, local)? {
                return Ok(action);
            }
        }
        for (depth, frame) in stack.iter().enumerate().rev() {
            let innermost = depth + 1 == stack.len();
            if frame.loop_entries.contains(&target) {
                if !innermost {
                    return Err("branch continues an outer loop".to_string());
                }
                let label = frame
                    .continue_label
                    .ok_or_else(|| "loop frame without a continue target".to_string())?;
                return Ok(Action::Goto(label));
            }
            if frame.next_entries.contains(&target) {
                if !innermost {
                    return Err("branch leaves more than one construct".to_string());
                }
                if frame.merge_dispatches {
                    let flow = self.flow_id_of(target);
                    return Ok(Action::Dispatch {
                        label: frame.merge_label,
                        flow,
                    });
                }
                return Ok(Action::Goto(frame.merge_label));
            }
        }
        Err("branch target is not reachable from the enclosing constructs".to_string())
    }

    /// How to enter `shape` at `target`, if `shape` starts there.
    fn enter(&mut self, target: Word, shape: &Shape) -> Result<Option<Action>, String> {
        match shape {
            Shape::Simple { label, .. } => Ok((*label == target).then_some(Action::Goto(*label))),
            Shape::Loop { entries, .. } => {
                if !entries.contains(&target) {
                    return Ok(None);
                }
                if entries.len() > 1 {
                    return Err("loop with several entries".to_string());
                }
                Ok(Some(Action::Goto(self.start_of(shape))))
            }
            Shape::Multiple { id, handled, next } => {
                if handled.iter().any(|(label, _)| *label == target) {
                    if self.fused.contains(id) {
                        return Ok(Some(Action::Goto(target)));
                    }
                    let flow = self.flow_id_of(target);
                    return Ok(Some(Action::Dispatch {
                        label: self.start_of(shape),
                        flow,
                    }));
                }
                match next {
                    Some(next) => self.enter(target, next),
                    None => Ok(None),
                }
            }
        }
    }

    fn emit(&mut self, shape: &Shape, stack: &mut Vec<Frame>) -> Result<(), String> {
        match shape {
            Shape::Simple { label, next, .. } => self.emit_simple(*label, next.as_deref(), stack),
            Shape::Loop {
                entries,
                inner,
                next,
                ..
            } => self.emit_loop(shape, entries, inner, next.as_deref(), stack),
            Shape::Multiple { id, handled, next } => {
                self.emit_multiple(*id, shape, handled, next.as_deref(), stack)
            }
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
        block.instructions.extend(
            source
                .instructions
                .iter()
                .take(source.instructions.len().saturating_sub(1))
                .cloned(),
        );
        self.emit_terminator(label, term.clone(), &mut block, next, stack)?;
        self.out.push(block);
        if let Some(next) = next {
            self.emit(next, stack)?;
        }
        Ok(())
    }

    /// Write the branch structure for `label`'s terminator, appending any per-edge helper blocks.
    fn emit_terminator(
        &mut self,
        label: Word,
        term: Term,
        block: &mut Block,
        next: Option<&Shape>,
        stack: &[Frame],
    ) -> Result<(), String> {
        match term {
            Term::Return | Term::Unreachable | Term::ReturnValue(_) | Term::Kill(_) => {
                let instruction = match term {
                    Term::Return => Instruction::new(Op::Return, None, None, vec![]),
                    Term::ReturnValue(value) => {
                        Instruction::new(Op::ReturnValue, None, None, vec![Operand::IdRef(value)])
                    }
                    Term::Unreachable => Instruction::new(Op::Unreachable, None, None, vec![]),
                    Term::Kill(instruction) => instruction,
                    _ => unreachable!("matched above"),
                };
                block.instructions.push(instruction);
                Ok(())
            }
            Term::Branch(target) => {
                let action = self.resolve(target, next, stack)?;
                self.apply(label, target, action, block);
                Ok(())
            }
            Term::BranchCond(condition, on_true, on_false) => {
                if on_true == on_false {
                    let action = self.resolve(on_true, next, stack)?;
                    self.apply(label, on_true, action, block);
                    return Ok(());
                }
                let true_label = self.edge_label(label, on_true, next, stack)?;
                let false_label = self.edge_label(label, on_false, next, stack)?;
                let merge = self.selection_merge(next);
                block.instructions.push(Instruction::new(
                    Op::SelectionMerge,
                    None,
                    None,
                    vec![
                        Operand::IdRef(merge),
                        Operand::SelectionControl(SelectionControl::NONE),
                    ],
                ));
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
                let merge = self.selection_merge(next);
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

    /// The declared merge for a divergent terminator: wherever the shape that follows begins. A
    /// terminator whose arms all leave the enclosing construct has no continuation, so it declares
    /// a merge block nothing reaches.
    fn selection_merge(&mut self, next: Option<&Shape>) -> Word {
        match next {
            Some(next) => self.start_of(next),
            None => self.fresh_unreachable(),
        }
    }

    /// The label a single outgoing edge branches to, creating a helper block when that edge has to
    /// store phi values or a flow id before it can branch.
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
        let flow = match action {
            Action::Goto(label) if stores.is_empty() => return Ok(label),
            Action::Goto(_) => None,
            Action::Dispatch { flow, .. } => Some(flow),
        };
        let destination = match action {
            Action::Goto(label) | Action::Dispatch { label, .. } => label,
        };
        let helper = self.tc.fresh();
        let mut block = Block::new();
        block.label = Some(Instruction::new(Op::Label, None, Some(helper), vec![]));
        for (var, value) in stores {
            block.instructions.push(Instruction::new(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(var), Operand::IdRef(value)],
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
        block.instructions.push(Instruction::new(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(destination)],
        ));
        self.out.push(block);
        Ok(helper)
    }

    /// Finish a block whose terminator has one outgoing edge: the stores can go inline.
    fn apply(&mut self, from: Word, target: Word, action: Action, block: &mut Block) {
        for (var, value) in self
            .phi_stores
            .get(&(from, target))
            .cloned()
            .unwrap_or_default()
        {
            block.instructions.push(Instruction::new(
                Op::Store,
                None,
                None,
                vec![Operand::IdRef(var), Operand::IdRef(value)],
            ));
        }
        let destination = match action {
            Action::Goto(label) => label,
            Action::Dispatch { label, flow } => {
                let variable = self.flow_variable();
                let constant = self.tc.int_const(self.flow_ty, flow);
                block.instructions.push(Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(variable), Operand::IdRef(constant)],
                ));
                label
            }
        };
        block.instructions.push(Instruction::new(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(destination)],
        ));
    }

    fn emit_loop(
        &mut self,
        shape: &Shape,
        entries: &BTreeSet<Word>,
        inner: &Shape,
        next: Option<&Shape>,
        stack: &mut Vec<Frame>,
    ) -> Result<(), String> {
        if entries.len() != 1 {
            return Err("loop with several entries".to_string());
        }
        let header = self.start_of(shape);
        let continue_label = self.tc.fresh();
        let merge_label = match next {
            Some(next) => self.start_of(next),
            None => self.fresh_unreachable(),
        };
        let body_start = self.start_of(inner);
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
            loop_entries: entries.clone(),
            continue_label: Some(continue_label),
            merge_label,
            next_entries: next.map(Shape::entry_labels).unwrap_or_default(),
            merge_dispatches: next.is_some_and(|next| self.needs_dispatch(next)),
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

        if let Some(next) = next {
            self.emit(next, stack)?;
        }
        Ok(())
    }

    fn needs_dispatch(&self, shape: &Shape) -> bool {
        matches!(shape, Shape::Multiple { id, handled, .. }
            if !handled.is_empty() && !self.fused.contains(id))
    }

    fn emit_multiple(
        &mut self,
        id: usize,
        shape: &Shape,
        handled: &[(Word, Shape)],
        next: Option<&Shape>,
        stack: &mut Vec<Frame>,
    ) -> Result<(), String> {
        let merge_label = match next {
            Some(next) => self.start_of(next),
            None => self.fresh_unreachable(),
        };
        if !self.fused.contains(&id) && !handled.is_empty() {
            // Several terminators reach this dispatch, so the flow variable says which arm was
            // meant. The dispatch block IS this shape's start, which is also the enclosing
            // construct's merge, so control arrives here by an ordinary structured exit.
            let dispatch = self.start_of(shape);
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
            let mut operands = vec![Operand::IdRef(loaded), Operand::IdRef(merge_label)];
            for (entry, _) in handled {
                let flow = self.flow_id_of(*entry);
                operands.push(Operand::LiteralBit32(flow as u32));
                operands.push(Operand::IdRef(*entry));
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
        }

        stack.push(Frame {
            loop_entries: BTreeSet::new(),
            continue_label: None,
            merge_label,
            next_entries: next.map(Shape::entry_labels).unwrap_or_default(),
            merge_dispatches: next.is_some_and(|next| self.needs_dispatch(next)),
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

        if let Some(next) = next {
            self.emit(next, stack)?;
        }
        Ok(())
    }
}
