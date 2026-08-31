//! Typed SSA IR — the emission substrate.
//!
//! A function body is parsed ONCE into typed basic blocks — structured terminators plus a `value_types`
//! map resolving each SSA result's type up front, on the value, instead of re-deriving it at each use.
//! This is the SOLE substrate emission walks: `build_from_blocks` lowers the structurized `BodyBlock`s
//! by consuming each block's `typed` carrier (populated at split time and dual-updated at every
//! synthesis/mutation site); a block that reaches it without a carrier is a fail-visible `Err`, never a
//! re-lex fallback. `emit_function` emits from the returned `TirFunction.blocks`. `BodyBlock.lines` and
//! `LlFunction.body` are deleted (T5): LLVM-IR text is read exactly once, at parse — no mid-pipeline
//! re-lexing survives.
//!
//! Scope: block splitting + structured terminators (the complete control-flow set), and result-type
//! resolution for arithmetic, compares (→ bool/bool-vector), conversions, `load`, `select`, `phi`,
//! direct+indirect `call`, `fneg`/`freeze`, element/value extract+insert (incl. `extractvalue`'s
//! constant index walk into struct/array aggregates), `alloca`/`getelementptr` (→ addrspace-only
//! `Ptr`), and `shufflevector`. Measured **100.0% of all defining instructions across the 16-shard
//! private capture** resolve (the residual ~0.03% is `extractvalue` into an opaque `Named` struct, which needs
//! the module type table). The `--tir-check` gate cross-validates terminators against the proven string
//! lexer (0 mismatches).
//!
//! Each `TirInst` also carries its **resolved typed operands** (`TirOperand`): for the
//! binary/compare/select/convert/load/store/phi/freeze/fneg shapes AND the vector/aggregate element ops
//! (`extractelement`/`insertelement`/`shufflevector`/`extractvalue`/`insertvalue`), every value operand
//! is lowered to an SSA `Value { name, ty }` or typed `Const { ty }` carrying its use-site type (opcodes
//! whose operand layout is not yet lowered — getelementptr/call — contribute one `Unresolved` marker,
//! with the parsed whole carried on `gep`/`call`). `tir_self_check` proves these sound: **0 / 1.79M
//! checked operand type mismatches** broadly (every `Value` operand's use-site type is compatible
//! with the type its def recorded, under `i1`≡`Bool` and the opaque-pointer addrspace-only rule).

use super::ir::{LlGep, LlType, LlValue, TypedValue};
use super::parse::{LlCall, LlLoad};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;

mod lower;
pub(in crate::native) use lower::*;
mod pointee;
pub(in crate::native) use pointee::*;
mod storage;
pub(in crate::native) use storage::*;
mod terminator;
pub(in crate::native) use terminator::*;
mod operands;
pub(in crate::native) use operands::*;
mod phi_edit;
mod rename;
pub(in crate::native) use rename::renamed_llvalue;
mod substitute;

/// Canonicalize an SSA identity created by lowering LLVM control flow to SPIR-V's block-predecessor
/// model. LLVM switch edges may repeat a destination, while SPIR-V permits exactly one `OpPhi` pair
/// per predecessor block. When the finalized CFG has one actual predecessor and every incoming pair
/// names that predecessor with the same value, the phi is the value itself. Substitute every such
/// result in the typed carrier before SPIR-V values or pointer-representation sidecars are built.
///
/// This runs on the finalized structurized carriers, where the predecessor relation is authoritative.
/// It deliberately leaves partial, conflicting, and multi-predecessor phis intact and fail-visible.
pub(in crate::native) fn canonicalize_single_predecessor_phis(
    blocks: &mut [crate::native::cfg::BodyBlock],
) {
    let block_names = blocks
        .iter()
        .map(|block| block.name.as_str())
        .collect::<HashSet<_>>();
    let mut predecessors: HashMap<String, HashSet<String>> = HashMap::new();
    for block in blocks.iter() {
        let Some(typed) = block.typed.as_ref() else {
            continue;
        };
        for successor in typed.terminator.successors() {
            if block_names.contains(successor) {
                predecessors
                    .entry(successor.to_string())
                    .or_default()
                    .insert(block.name.clone());
            }
        }
    }
    loop {
        let identity = blocks.iter().enumerate().find_map(|(block_index, block)| {
            let actual = predecessors.get(&block.name)?;
            if actual.len() != 1 {
                return None;
            }
            let predecessor = actual.iter().next()?;
            let typed = block.typed.as_ref()?;
            typed
                .insts
                .iter()
                .enumerate()
                .find_map(|(inst_index, inst)| {
                    let result = inst.result.as_ref()?;
                    let (ty, incoming) = inst.phi_incoming().as_ref()?;
                    let (value, _) = incoming.first()?;
                    let is_identity = incoming
                        .iter()
                        .all(|(candidate, parent)| parent == predecessor && candidate == value);
                    let is_self_reference = matches!(value, LlValue::Local(name) if name == result);
                    (is_identity && !is_self_reference).then(|| {
                        (
                            block_index,
                            inst_index,
                            result.clone(),
                            TypedValue {
                                ty: ty.clone(),
                                value: value.clone(),
                            },
                        )
                    })
                })
        });
        let Some((block_index, inst_index, result, replacement)) = identity else {
            break;
        };
        blocks[block_index]
            .typed_mut()
            .expect("identity phi came from a typed carrier")
            .insts
            .remove(inst_index);
        let substitutions = HashMap::from([(result, replacement)]);
        for block in blocks.iter_mut() {
            if let Some(typed) = block.typed_mut() {
                typed.substitute_values(&substitutions);
            }
        }
    }
}

/// Fold literal conditional branches on the owned typed CFG, remove blocks made unreachable by
/// those branches, and repair the surviving phi predecessor sets before merge planning. This is the
/// source-graph counterpart of late SPIR-V constant-branch pruning: planners and the emitter never
/// see the dead arm, so a pointer phi whose only other incoming is a dead null stays out of pointer
/// SSA by construction.
///
/// Aggregate phis whose incoming carrier did not lower cannot be edited faithfully. Decline only
/// the literal edge whose reachability transaction would change such a phi; independent literal
/// edges remain safe to fold.
pub(in crate::native) fn prune_literal_branch_dead_blocks(
    mut blocks: Vec<crate::native::cfg::BodyBlock>,
) -> Vec<crate::native::cfg::BodyBlock> {
    if blocks.is_empty() || blocks.iter().any(|block| block.typed.is_none()) {
        return blocks;
    }

    let mut declined = HashSet::new();
    loop {
        let candidate = blocks.iter().find_map(|block| {
            if declined.contains(&block.name) {
                return None;
            }
            let typed = block.typed.as_ref()?;
            let TirTerminator::BrCond { cond, t, f } = &typed.terminator else {
                return None;
            };
            let target = match cond.as_str() {
                "true" => t,
                "false" => f,
                _ => return None,
            };
            Some((block.name.clone(), target.clone()))
        });
        let Some((source, target)) = candidate else {
            break;
        };
        if prune_one_literal_edge(&mut blocks, &source, &target) {
            declined.clear();
        } else {
            declined.insert(source);
        }
    }
    blocks
}

/// Omit unused `getelementptr` definitions before SPIR-V pointer construction.
///
/// A source GEP is a pure address calculation. If no typed instruction or terminator consumes its
/// SSA result, emitting an `OpAccessChain` for it can only preserve dead source representation; in
/// particular, LLVM permits dead typed-pointer paths that have no legal Logical-SPIR-V equivalent.
/// Peeling to a fixpoint also omits a parent GEP whose sole consumer was another omitted GEP.
pub(in crate::native) fn prune_unused_geps(
    mut blocks: Vec<crate::native::cfg::BodyBlock>,
) -> Vec<crate::native::cfg::BodyBlock> {
    loop {
        let mut used = HashSet::<String>::new();
        for block in &blocks {
            let Some(typed) = block.typed.as_ref() else {
                return blocks;
            };
            for instruction in &typed.insts {
                if instruction.is_ignored_void_call() {
                    continue;
                }
                instruction.visit_uses(|name| {
                    used.insert(name.to_string());
                });
            }
            match &typed.terminator {
                TirTerminator::Br(_) | TirTerminator::Unreachable => {}
                TirTerminator::BrCond { cond, .. } => {
                    used.insert(cond.clone());
                }
                TirTerminator::Switch { selector, .. } => {
                    used.insert(selector.clone());
                }
                TirTerminator::Ret(Some(value)) => {
                    used.insert(value.clone());
                }
                TirTerminator::Ret(None) => {}
            }
        }

        let mut changed = false;
        for block in &mut blocks {
            let typed = block.typed_mut().expect("all typed carriers checked above");
            let before = typed.insts.len();
            typed.insts.retain(|instruction| {
                instruction.opcode != TirOpcode::GetElementPtr
                    || instruction
                        .result
                        .as_ref()
                        .is_none_or(|result| used.contains(result))
            });
            changed |= typed.insts.len() != before;
        }
        if !changed {
            return blocks;
        }
    }
}

fn prune_one_literal_edge(
    blocks: &mut Vec<crate::native::cfg::BodyBlock>,
    source: &str,
    target: &str,
) -> bool {
    let Some(old_cfg) = crate::native::cfg::graph::Cfg::from_blocks(blocks) else {
        return false;
    };
    let old_reachable = old_cfg.reachable_from(&old_cfg.entry);
    let block_indices = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| (block.name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut reachable = HashSet::new();
    let mut pending = vec![0usize];
    while let Some(index) = pending.pop() {
        let block = &blocks[index];
        if !reachable.insert(block.name.clone()) {
            continue;
        }
        let successors = if block.name == source {
            vec![target]
        } else {
            block
                .typed
                .as_ref()
                .expect("typed carrier checked above")
                .terminator
                .successors()
        };
        pending.extend(
            successors
                .into_iter()
                .filter_map(|successor| block_indices.get(successor).copied()),
        );
    }
    let mut predecessors = HashMap::<String, HashSet<String>>::new();
    for block in blocks
        .iter()
        .filter(|block| reachable.contains(&block.name))
    {
        let successors = if block.name == source {
            vec![target]
        } else {
            block
                .typed
                .as_ref()
                .expect("typed carrier checked above")
                .terminator
                .successors()
        };
        for successor in successors {
            if reachable.contains(successor) {
                predecessors
                    .entry(successor.to_string())
                    .or_default()
                    .insert(block.name.clone());
            }
        }
    }
    drop(block_indices);

    // Prove the edge/phi transaction before mutating the graph. A phi carrier blocks only this
    // candidate when this candidate would alter that exact block's reachable predecessor set.
    for block in blocks
        .iter()
        .filter(|block| reachable.contains(&block.name))
    {
        let old_predecessors = old_cfg
            .predecessors
            .get(&block.name)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter(|predecessor| old_reachable.contains(predecessor))
            .collect::<HashSet<_>>();
        let new_predecessors = predecessors.get(&block.name).cloned().unwrap_or_default();
        if old_predecessors != new_predecessors
            && block
                .typed
                .as_ref()
                .expect("typed carrier checked above")
                .insts
                .iter()
                .any(|inst| inst.opcode == "phi" && inst.phi_incoming().is_none())
        {
            return false;
        }
    }

    blocks
        .iter_mut()
        .find(|block| block.name == source)
        .and_then(|block| block.typed_mut())
        .expect("literal edge source came from a typed block")
        .set_unconditional_branch(target);
    for block in blocks.iter_mut() {
        if reachable.contains(&block.name) {
            let incoming = predecessors.get(&block.name).cloned().unwrap_or_default();
            block
                .typed_mut()
                .expect("typed carrier checked above")
                .rebuild_phi_incomings(|predecessor| incoming.contains(predecessor));
        }
    }
    blocks.retain(|block| reachable.contains(&block.name));
    canonicalize_single_predecessor_phis(blocks);
    true
}

/// A block's terminator, parsed once instead of re-lexed from the trailing line on every CFG pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum TirTerminator {
    /// `br label %t`
    Br(String),
    /// `br i1 %c, label %t, label %f`
    BrCond { cond: String, t: String, f: String },
    /// `switch <ty> %sel, label %default [ <ty> C, label %L ... ]`
    Switch {
        selector: String,
        default: String,
        cases: Vec<(String, String)>,
    },
    /// `ret void` / `ret <ty> <v>`
    Ret(Option<String>),
    /// `unreachable`
    Unreachable,
}

impl TirTerminator {
    /// The block labels this terminator can transfer control to, in order.
    pub(super) fn successors(&self) -> Vec<&str> {
        match self {
            TirTerminator::Br(t) => vec![t.as_str()],
            TirTerminator::BrCond { t, f, .. } => vec![t.as_str(), f.as_str()],
            TirTerminator::Switch { default, cases, .. } => {
                let mut s = vec![default.as_str()];
                s.extend(cases.iter().map(|(_, l)| l.as_str()));
                s
            }
            TirTerminator::Ret(_) | TirTerminator::Unreachable => vec![],
        }
    }

    /// Redirect every successor label equal to `from` to `to` (the typed dual of the text
    /// `redirect_label` applied to a terminator line). Only `label` targets move — a `ret` value is
    /// never a label — so this reproduces the string rewrite exactly for the terminator half.
    pub(super) fn redirect_successor(&mut self, from: &str, to: &str) {
        match self {
            TirTerminator::Br(t) => {
                if t == from {
                    *t = to.to_string();
                }
            }
            TirTerminator::BrCond { t, f, .. } => {
                if t == from {
                    *t = to.to_string();
                }
                if f == from {
                    *f = to.to_string();
                }
            }
            TirTerminator::Switch { default, cases, .. } => {
                if default == from {
                    *default = to.to_string();
                }
                for (_, l) in cases {
                    if l == from {
                        *l = to.to_string();
                    }
                }
            }
            TirTerminator::Ret(_) | TirTerminator::Unreachable => {}
        }
    }
}

impl TirBlock {
    /// Redirect every terminator successor label `from` -> `to`, on both the structured terminator and
    /// the `switch` operand carrier — the typed dual of applying the text `redirect_label` to this
    /// block's terminator line (the `ret` carrier is untouched: a redirect never rewrites a return
    /// value). Byte-identical to re-lowering the redirected line by construction, so a mutation site can
    /// keep the carrier in step instead of invalidating it (`typed = None`).
    pub(super) fn redirect_successor(&mut self, from: &str, to: &str) {
        self.terminator.redirect_successor(from, to);
        if let Some(sw) = &mut self.switch {
            if sw.default_label == from {
                sw.default_label = to.to_string();
            }
            for (_, l) in &mut sw.cases {
                if l == from {
                    *l = to.to_string();
                }
            }
        }
    }

    /// Rewrite every `phi` predecessor label `from` -> `to` across this block's instructions — the typed
    /// dual of applying the text `rewrite_phi_predecessor` to each phi line. Only the incoming
    /// predecessor labels move; the incoming VALUES and the phi result/type are untouched (matching the
    /// string rewrite). Byte-identical to re-lowering the rewritten phi lines by construction, so a
    /// mutation site can keep the carrier in step instead of invalidating it.
    pub(super) fn rewrite_phi_predecessor(&mut self, from: &str, to: &str) {
        for inst in &mut self.insts {
            if inst.opcode != "phi" {
                continue;
            }
            if let Some((_, incoming)) = inst.phi_incoming_mut() {
                for (_, pred) in incoming {
                    if pred == from {
                        *pred = to.to_string();
                    }
                }
            }
        }
    }

    /// Replace this block's terminator with an unconditional `br label <target>` — the typed dual of
    /// popping a block's terminator line and pushing `br label {target}`. Resets the `ret`/`switch`
    /// operand carriers (the new terminator is neither a `ret` nor a `switch`), matching a re-lower of
    /// the rewritten block; the straight-line instructions are untouched.
    pub(super) fn set_unconditional_branch(&mut self, target: &str) {
        self.terminator = TirTerminator::Br(target.to_string());
        self.ret = RetEmit::FromText;
        self.switch = None;
    }

    /// A fresh block named `name` carrying ONLY this block's terminator (and its typed `ret`/`switch`
    /// operands) with no instructions — the carrier-direct dual of lifting a header's conditional/switch
    /// terminator LINE into a new successor block (the former `synthetic_block(sel, vec![term])`).
    /// Byte-identical to re-lowering a block whose single line is that terminator by construction: the
    /// terminator carries no straight-line instruction, so the block has empty `insts` and the same
    /// `terminator`/`ret`/`switch` this block already lowered from the one terminator line.
    pub(super) fn terminator_only_block(&self, name: &str) -> TirBlock {
        TirBlock {
            label: name.to_string(),
            insts: Vec::new(),
            terminator: self.terminator.clone(),
            ret: self.ret.clone(),
            switch: self.switch.clone(),
        }
    }

    /// Replace this block's terminator with the one parsed from `term_line` — the typed dual of
    /// overwriting a block's trailing terminator line (e.g. `unreachable` -> `ret void` / `ret <ty>
    /// undef`). Recomputes all three terminator carriers (`terminator` / `ret` / `switch`) with the
    /// SAME `parse_terminator` / `ret_emit` / `switch_emit` the block lowering runs, so the result is
    /// byte-identical to re-lowering the rewritten block; the straight-line instructions are untouched.
    /// A no-op if `term_line` does not parse as a terminator (the caller only ever passes one).
    pub(super) fn set_terminator_line(&mut self, term_line: &str) {
        if let Some(term) = parse_terminator(term_line) {
            self.terminator = term;
            self.ret = ret_emit(term_line);
            self.switch = switch_emit(term_line);
        }
    }

    /// Classify this block's `ret` terminator for the cross-arm return-normalization passes
    /// (divergent-exit unification), rendered from the typed `ret` carrier — the
    /// carrier substitute for those passes re-lexing the trailing terminator line. `RetEmit::FromText`
    /// maps to `NotRet`: it is a non-`ret` terminator (a `ret` whose value did not parse is measured
    /// dead broadly). See [`RetTerm`] for the value/unrenderable distinction that lets the caller
    /// mirror the line path's `?`-bail on an un-modellable `ret`.
    pub(in crate::native) fn ret_term(&self) -> RetTerm {
        match &self.ret {
            RetEmit::Void => RetTerm::Void,
            RetEmit::FromText => RetTerm::NotRet,
            RetEmit::Value(tv) => {
                match (
                    crate::native::render::render_type(&tv.ty),
                    crate::native::render::render_value(&tv.value),
                ) {
                    (Some(ty), Some(val)) => RetTerm::Value { ty, val },
                    _ => RetTerm::Unrenderable,
                }
            }
        }
    }
}

/// The shape of a block's `ret` terminator as seen by the cross-arm return-normalization passes,
/// rendered from the typed carrier (see [`TirBlock::ret_term`]).
pub(in crate::native) enum RetTerm {
    /// `ret void`.
    Void,
    /// `ret <ty> <val>` whose type + value both render to text (the line path's `ret <ty> <val>` split).
    Value { ty: String, val: String },
    /// A value `ret` whose type/value is not injectively renderable — the caller must DECLINE the
    /// transform (mirrors the line path bailing via `?`), never silently skip the block.
    Unrenderable,
    /// Any non-`ret` terminator (`br` / `switch` / `unreachable`).
    NotRet,
}

/// A resolved instruction operand: enough typed structure for emission to consume the instruction
/// without re-lexing the line. A `Value` operand carries the type it is *used as* at this site, so the
/// typed graph can be checked for use/def type agreement (the `Value.ty` at a use must equal the
/// `value_types` the def recorded). `Const` is a typed literal; `Unresolved` is an operand tir does not
/// yet lower (kept so operand coverage is reported honestly rather than silently dropped).
#[derive(Clone, Debug)]
pub(super) enum TirOperand {
    /// `%name` SSA reference, carrying its use-site declared type.
    Value { name: String, ty: LlType },
    /// A typed literal/constant operand (`i32 7`, `float 1.0`), carrying its parsed value so emission
    /// can materialize the constant without re-lexing.
    Const { value: LlValue, ty: LlType },
    /// An operand tir does not yet lower to a typed form.
    Unresolved,
}

impl TirOperand {
    /// The operand as a `TypedValue` (`Value` -> a `Local`, `Const` -> its literal), or `None` if the
    /// operand is `Unresolved`. This is what graph-driven emission consumes in place of re-parsing the
    /// instruction text.
    pub(super) fn as_typed_value(&self) -> Option<TypedValue> {
        match self {
            TirOperand::Value { name, ty } => Some(TypedValue {
                ty: ty.clone(),
                value: LlValue::Local(name.clone()),
            }),
            TirOperand::Const { value, ty } => Some(TypedValue {
                ty: ty.clone(),
                value: value.clone(),
            }),
            TirOperand::Unresolved => None,
        }
    }
}

/// One non-terminator instruction with its (optional) SSA result, resolved result type, the SSA
/// values it uses (the def/use edges — operand `%names`, excluding the result and any phi predecessor
/// labels, which are control-flow edges not value uses), and its resolved typed operands.
#[derive(Clone, Debug)]
pub(super) struct TirInst {
    /// `%r` for `%r = ...`; `None` for an effect-only instruction (`store`, a void `call`).
    pub(super) result: Option<String>,
    /// The resolved result type, or `None` when not yet inferable from the line alone (e.g. GEP).
    pub(super) result_ty: Option<LlType>,
    /// Lossless def/use fallback for an operand shape tir could not resolve. Resolved instructions
    /// derive these names from `operands` (and parsed phis from their canonical incoming carrier), so
    /// the common path does not retain a second allocation and copy of every SSA operand name.
    pub(super) uses: Option<Vec<String>>,
    /// The instruction's operands resolved to typed form, in source order, for the opcode shapes tir
    /// lowers (binary/compare/select/convert/load/store/phi/freeze/fneg + the vector/aggregate element
    /// ops extractelement/insertelement/shufflevector/extractvalue/insertvalue). Opcodes whose operand
    /// layout is not yet lowered (getelementptr/call) contribute a single `Unresolved` so the operand
    /// list is never empty for an instruction that has operands.
    pub(super) operands: Vec<TirOperand>,
    /// The instruction's OPCODE mnemonic (`add`/`load`/`getelementptr`/...), the first whitespace token
    /// of the rhs — computed once at build time so structured emission (`emit_body_inst`) can DISPATCH on
    /// it. Every opcode family routes by this field into its graph-driven emitter; an unmigrated opcode is
    /// a fail-visible `Err` (there is no text fallback). Effect-only lines (`store`, void `call`) carry
    /// their leading token here too (`store`/`call`/`tail`). Empty string for a blank/comment/label line.
    pub(super) opcode: TirOpcode,
    /// Opcode-family data. Mutually exclusive carrier shapes share this one tagged storage slot rather
    /// than making every instruction reserve space for every opcode family.
    data: Box<TirInstDetails>,
}

/// An LLVM instruction mnemonic interned into the typed carrier. AIR modules repeat a small opcode
/// vocabulary hundreds of thousands of times; retaining a separately allocated `String` per
/// instruction needlessly fragments the worker heap. Known structural mnemonics occupy only the enum
/// discriminant. An unfamiliar mnemonic remains lossless in `Other`, so unsupported input still
/// reaches the same fail-visible diagnostic rather than being guessed or collapsed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum TirOpcode {
    Add,
    AddrSpaceCast,
    Alloca,
    And,
    Ashr,
    AtomicRmw,
    Bitcast,
    Call,
    CmpXchg,
    ExtractElement,
    ExtractValue,
    FAdd,
    FCmp,
    FDiv,
    FMul,
    FNeg,
    FPToSI,
    FPToUI,
    FPExt,
    FPTrunc,
    FRem,
    FSub,
    Freeze,
    GetElementPtr,
    ICmp,
    InsertElement,
    InsertValue,
    IntToPtr,
    Load,
    LShr,
    Metal2VulkanInlineParameter,
    Mul,
    MustTail,
    NoTail,
    Or,
    Phi,
    PtrToInt,
    SDiv,
    SExt,
    Shl,
    ShuffleVector,
    SIToFP,
    SRem,
    Select,
    Store,
    Sub,
    Tail,
    Trunc,
    UDiv,
    UIToFP,
    URem,
    Xor,
    ZExt,
    Other(Box<UnknownOpcode>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct UnknownOpcode(String);

impl TirOpcode {
    pub(super) fn new(opcode: String) -> Self {
        match opcode.as_str() {
            "add" => Self::Add,
            "addrspacecast" => Self::AddrSpaceCast,
            "alloca" => Self::Alloca,
            "and" => Self::And,
            "ashr" => Self::Ashr,
            "atomicrmw" => Self::AtomicRmw,
            "bitcast" => Self::Bitcast,
            "call" => Self::Call,
            "cmpxchg" => Self::CmpXchg,
            "extractelement" => Self::ExtractElement,
            "extractvalue" => Self::ExtractValue,
            "fadd" => Self::FAdd,
            "fcmp" => Self::FCmp,
            "fdiv" => Self::FDiv,
            "fmul" => Self::FMul,
            "fneg" => Self::FNeg,
            "fptosi" => Self::FPToSI,
            "fptoui" => Self::FPToUI,
            "fpext" => Self::FPExt,
            "fptrunc" => Self::FPTrunc,
            "frem" => Self::FRem,
            "fsub" => Self::FSub,
            "freeze" => Self::Freeze,
            "getelementptr" => Self::GetElementPtr,
            "icmp" => Self::ICmp,
            "insertelement" => Self::InsertElement,
            "insertvalue" => Self::InsertValue,
            "inttoptr" => Self::IntToPtr,
            "load" => Self::Load,
            "lshr" => Self::LShr,
            "metal2vulkan.inline_parameter" => Self::Metal2VulkanInlineParameter,
            "mul" => Self::Mul,
            "musttail" => Self::MustTail,
            "notail" => Self::NoTail,
            "or" => Self::Or,
            "phi" => Self::Phi,
            "ptrtoint" => Self::PtrToInt,
            "sdiv" => Self::SDiv,
            "sext" => Self::SExt,
            "shl" => Self::Shl,
            "shufflevector" => Self::ShuffleVector,
            "sitofp" => Self::SIToFP,
            "srem" => Self::SRem,
            "select" => Self::Select,
            "store" => Self::Store,
            "sub" => Self::Sub,
            "tail" => Self::Tail,
            "trunc" => Self::Trunc,
            "udiv" => Self::UDiv,
            "uitofp" => Self::UIToFP,
            "urem" => Self::URem,
            "xor" => Self::Xor,
            "zext" => Self::ZExt,
            _ => Self::Other(Box::new(UnknownOpcode(opcode))),
        }
    }

    pub(super) fn as_str(&self) -> &str {
        match self {
            Self::Add => "add",
            Self::AddrSpaceCast => "addrspacecast",
            Self::Alloca => "alloca",
            Self::And => "and",
            Self::Ashr => "ashr",
            Self::AtomicRmw => "atomicrmw",
            Self::Bitcast => "bitcast",
            Self::Call => "call",
            Self::CmpXchg => "cmpxchg",
            Self::ExtractElement => "extractelement",
            Self::ExtractValue => "extractvalue",
            Self::FAdd => "fadd",
            Self::FCmp => "fcmp",
            Self::FDiv => "fdiv",
            Self::FMul => "fmul",
            Self::FNeg => "fneg",
            Self::FPToSI => "fptosi",
            Self::FPToUI => "fptoui",
            Self::FPExt => "fpext",
            Self::FPTrunc => "fptrunc",
            Self::FRem => "frem",
            Self::FSub => "fsub",
            Self::Freeze => "freeze",
            Self::GetElementPtr => "getelementptr",
            Self::ICmp => "icmp",
            Self::InsertElement => "insertelement",
            Self::InsertValue => "insertvalue",
            Self::IntToPtr => "inttoptr",
            Self::Load => "load",
            Self::LShr => "lshr",
            Self::Metal2VulkanInlineParameter => "metal2vulkan.inline_parameter",
            Self::Mul => "mul",
            Self::MustTail => "musttail",
            Self::NoTail => "notail",
            Self::Or => "or",
            Self::Phi => "phi",
            Self::PtrToInt => "ptrtoint",
            Self::SDiv => "sdiv",
            Self::SExt => "sext",
            Self::Shl => "shl",
            Self::ShuffleVector => "shufflevector",
            Self::SIToFP => "sitofp",
            Self::SRem => "srem",
            Self::Select => "select",
            Self::Store => "store",
            Self::Sub => "sub",
            Self::Tail => "tail",
            Self::Trunc => "trunc",
            Self::UDiv => "udiv",
            Self::UIToFP => "uitofp",
            Self::URem => "urem",
            Self::Xor => "xor",
            Self::ZExt => "zext",
            Self::Other(opcode) => &opcode.0,
        }
    }
}

impl Deref for TirOpcode {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl PartialEq<str> for TirOpcode {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for TirOpcode {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl fmt::Display for TirOpcode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[derive(Clone, Debug)]
enum TirInstData {
    Plain,
    Compare {
        predicate: Option<String>,
        rest: Option<String>,
    },
    Memory {
        align: Option<u64>,
        load: Option<Box<LlLoad>>,
        store: Option<Box<(TypedValue, TypedValue)>>,
    },
    Gep {
        parsed: Option<Box<LlGep>>,
        pointee: Option<LlType>,
    },
    Call {
        parsed: Option<Box<LlCall>>,
        void_line: Option<String>,
        value_error: Option<String>,
        alias_from_parsed: bool,
        alias_override: Option<Box<LlCall>>,
        emit_scan: EmitScanData,
    },
    Alloca(Option<LlType>),
    Phi {
        incoming: Option<(LlType, Vec<(LlValue, String)>)>,
        incoming_values: Option<Vec<LlValue>>,
        /// Exact `parse_phi` refusal captured while building a phi whose `incoming` carrier is
        /// absent. Diagnostics-only: emission surfaces it through the fail-visible graph-walk
        /// error; it never changes parsing or lowering. `None` for valid phis.
        parse_error: Option<String>,
    },
    Aggregate(Option<Vec<u32>>),
    Element {
        diag_line: Option<String>,
        shuffle_mask: Option<(u32, Vec<u32>)>,
    },
    Bitcast {
        destination: Option<String>,
        identity: bool,
    },
    Select(Option<Box<(TypedValue, TypedValue)>>),
}

#[derive(Clone, Debug)]
struct TirInstDetails {
    /// LLVM's aggregate `fast` floating-point flag on this instruction. Keeping it in the existing
    /// opcode-detail allocation avoids enlarging every instruction carrier.
    fast_math: bool,
    payload: TirInstData,
}

#[derive(Clone, Debug)]
enum EmitScanData {
    None,
    Parsed,
    Owned(Box<Result<LlCall, String>>),
}

impl TirInst {
    pub(super) fn fast_math(&self) -> bool {
        self.data.fast_math
    }

    /// Visit the instruction's def/use edges without retaining a second copy of resolved operand values.
    /// Parsed phis derive local names from `(value, predecessor)` pairs; other resolved instructions use
    /// their typed operands. Unresolved shapes use the lossless scan captured at lowering. The borrowed
    /// scratch exists only for the duration of the query and preserves source order, de-duplication, and
    /// exclusion of the instruction's own result.
    pub(super) fn visit_uses(&self, mut visit: impl FnMut(&str)) {
        fn collect<'a>(value: &'a LlValue, names: &mut Vec<&'a str>) {
            match value {
                LlValue::Local(name) => names.push(name),
                LlValue::Vector(values) | LlValue::Array(values) | LlValue::Struct(values) => {
                    for value in values {
                        collect(&value.value, names);
                    }
                }
                LlValue::Splat(value) => collect(&value.value, names),
                LlValue::Gep(gep) => {
                    collect(&gep.base.value, names);
                    for index in &gep.indices {
                        collect(&index.value, names);
                    }
                }
                LlValue::IntToPtr { source, .. } => collect(&source.value, names),
                LlValue::Global(_)
                | LlValue::Bool(_)
                | LlValue::Int(_)
                | LlValue::SignedInt(_)
                | LlValue::Hex(_)
                | LlValue::Float(_)
                | LlValue::Float32Bits(_)
                | LlValue::HalfBits(_)
                | LlValue::BFloatBits(_)
                | LlValue::Zero
                | LlValue::Undef => {}
            }
        }

        let mut names = Vec::new();
        if let Some(values) = self.phi_values() {
            for value in values {
                collect(value, &mut names);
            }
        } else if let Some(uses) = &self.uses {
            for name in uses {
                visit(name);
            }
            return;
        } else {
            for operand in &self.operands {
                match operand {
                    TirOperand::Value { name, .. } => names.push(name),
                    TirOperand::Const { value, .. } => collect(value, &mut names),
                    TirOperand::Unresolved => {
                        debug_assert!(false, "unresolved operand without def/use fallback")
                    }
                }
            }
        }

        let mut unique = Vec::with_capacity(names.len());
        for name in names {
            if self.result.as_deref() != Some(name) && !unique.contains(&name) {
                unique.push(name);
                visit(name);
            }
        }
    }

    pub(super) fn uses_any(&self, mut predicate: impl FnMut(&str) -> bool) -> bool {
        let mut matched = false;
        self.visit_uses(|name| matched |= predicate(name));
        matched
    }

    fn is_ignored_void_call(&self) -> bool {
        matches!(
            &self.data.payload,
            TirInstData::Call {
                void_line: Some(line),
                ..
            } if crate::native::parse::is_ignored_call_line(line)
        )
    }

    pub(super) fn cmp_predicate(&self) -> &Option<String> {
        match &self.data.payload {
            TirInstData::Compare { predicate, .. } => predicate,
            _ => &None,
        }
    }

    pub(super) fn mem_align(&self) -> Option<u64> {
        match &self.data.payload {
            TirInstData::Memory { align, .. } => *align,
            _ => None,
        }
    }

    pub(super) fn gep_source_ty(&self) -> Option<&LlType> {
        match &self.data.payload {
            TirInstData::Gep { parsed, .. } => parsed.as_ref().map(|gep| &gep.source_ty),
            _ => None,
        }
    }

    pub(super) fn gep(&self) -> &Option<Box<LlGep>> {
        match &self.data.payload {
            TirInstData::Gep { parsed, .. } => parsed,
            _ => &None,
        }
    }

    pub(super) fn call(&self) -> &Option<Box<LlCall>> {
        match &self.data.payload {
            TirInstData::Call { parsed, .. } => parsed,
            _ => &None,
        }
    }

    pub(super) fn alloca_ty(&self) -> &Option<LlType> {
        match &self.data.payload {
            TirInstData::Alloca(ty) => ty,
            _ => &None,
        }
    }

    pub(super) fn phi_incoming(&self) -> &Option<(LlType, Vec<(LlValue, String)>)> {
        match &self.data.payload {
            TirInstData::Phi { incoming, .. } => incoming,
            _ => &None,
        }
    }

    pub(super) fn phi_parse_error(&self) -> Option<&str> {
        match &self.data.payload {
            TirInstData::Phi { parse_error, .. } => parse_error.as_deref(),
            _ => None,
        }
    }

    pub(super) fn phi_incoming_mut(&mut self) -> &mut Option<(LlType, Vec<(LlValue, String)>)> {
        match &mut self.data.payload {
            TirInstData::Phi { incoming, .. } => incoming,
            _ => panic!("phi incoming mutation on non-phi instruction"),
        }
    }

    pub(super) fn aggregate_indices(&self) -> &Option<Vec<u32>> {
        match &self.data.payload {
            TirInstData::Aggregate(indices) => indices,
            _ => &None,
        }
    }

    pub(super) fn diag_line(&self) -> &Option<String> {
        match &self.data.payload {
            TirInstData::Element { diag_line, .. } => diag_line,
            _ => &None,
        }
    }

    pub(super) fn shuffle_mask(&self) -> &Option<(u32, Vec<u32>)> {
        match &self.data.payload {
            TirInstData::Element { shuffle_mask, .. } => shuffle_mask,
            _ => &None,
        }
    }

    pub(super) fn void_call_line(&self) -> &Option<String> {
        match &self.data.payload {
            TirInstData::Call { void_line, .. } => void_line,
            _ => &None,
        }
    }

    pub(super) fn value_call_error(&self) -> &Option<String> {
        match &self.data.payload {
            TirInstData::Call { value_error, .. } => value_error,
            _ => &None,
        }
    }

    pub(super) fn bitcast(&self) -> Option<(TypedValue, &str)> {
        match &self.data.payload {
            TirInstData::Bitcast { destination, .. } => self
                .operands
                .first()
                .and_then(TirOperand::as_typed_value)
                .zip(destination.as_deref()),
            _ => None,
        }
    }

    pub(super) fn icmp_rest(&self) -> &Option<String> {
        match &self.data.payload {
            TirInstData::Compare { rest, .. } => rest,
            _ => &None,
        }
    }

    pub(super) fn pointer_pointee(&self) -> &Option<LlType> {
        match &self.data.payload {
            TirInstData::Gep { pointee, .. } => pointee,
            _ => &None,
        }
    }

    pub(super) fn identity_ptr_bitcast(&self) -> Option<(&str, &str)> {
        match &self.data.payload {
            TirInstData::Bitcast { identity: true, .. } => {
                let result = self.result.as_deref()?;
                let base = self.operands.first().and_then(|operand| match operand {
                    TirOperand::Value { name, .. } => Some(name.as_str()),
                    _ => None,
                })?;
                Some((result, base))
            }
            _ => None,
        }
    }

    /// Incoming phi values from the full `(value, predecessor)` carrier when available, falling back
    /// to the lighter parser only for phi forms the full parser cannot represent. The two owned views
    /// are mutually exclusive, so ordinary phis do not retain every value twice.
    pub(super) fn phi_values(&self) -> Option<impl Iterator<Item = &LlValue> + Clone> {
        let (incoming, fallback) = match &self.data.payload {
            TirInstData::Phi {
                incoming,
                incoming_values,
                ..
            } => (incoming.as_ref(), incoming_values.as_ref()),
            _ => (None, None),
        };
        (incoming.is_some() || fallback.is_some()).then(|| {
            incoming
                .into_iter()
                .flat_map(|(_, values)| values.iter().map(|(value, _)| value))
                .chain(fallback.into_iter().flat_map(|values| values.iter()))
        })
    }

    pub(super) fn phi_incoming_values_mut(&mut self) -> &mut Option<Vec<LlValue>> {
        match &mut self.data.payload {
            TirInstData::Phi {
                incoming_values, ..
            } => incoming_values,
            _ => panic!("phi value mutation on non-phi instruction"),
        }
    }

    pub(super) fn select_arms(&self) -> &Option<Box<(TypedValue, TypedValue)>> {
        match &self.data.payload {
            TirInstData::Select(arms) => arms,
            _ => &None,
        }
    }

    pub(super) fn load(&self) -> &Option<Box<LlLoad>> {
        match &self.data.payload {
            TirInstData::Memory { load, .. } => load,
            _ => &None,
        }
    }

    pub(super) fn store(&self) -> &Option<Box<(TypedValue, TypedValue)>> {
        match &self.data.payload {
            TirInstData::Memory { store, .. } => store,
            _ => &None,
        }
    }

    pub(super) fn alias_call(&self) -> Option<&LlCall> {
        match &self.data.payload {
            TirInstData::Call {
                parsed,
                alias_from_parsed,
                alias_override,
                ..
            } => alias_override
                .as_deref()
                .or_else(|| alias_from_parsed.then(|| parsed.as_deref()).flatten()),
            _ => None,
        }
    }

    pub(super) fn emit_scan_call(&self) -> Option<Result<&LlCall, String>> {
        match &self.data.payload {
            TirInstData::Call {
                parsed, emit_scan, ..
            } => match emit_scan {
                EmitScanData::None => None,
                EmitScanData::Parsed => parsed.as_deref().map(Ok),
                EmitScanData::Owned(result) => Some(
                    result
                        .as_ref()
                        .as_ref()
                        .map_err(std::string::ToString::to_string),
                ),
            },
            _ => None,
        }
    }

    /// A helper-parameter boundary introduced by the typed inliner.
    ///
    /// The emitter gives this value an opaque temporary id while lowering the cloned helper body,
    /// then substitutes the caller argument id after the whole function has emitted. This preserves
    /// the residual SPIR-V inliner's ordering without serializing a synthetic instruction.
    pub(in crate::native) fn inline_parameter(result: String, argument: TypedValue) -> Self {
        Self {
            result: Some(result),
            result_ty: Some(argument.ty.clone()),
            uses: None,
            operands: vec![operand_from_typed_value(&argument)],
            opcode: TirOpcode::Metal2VulkanInlineParameter,
            data: Box::new(TirInstDetails {
                fast_math: false,
                payload: TirInstData::Plain,
            }),
        }
    }

    /// Whether this instruction is a `phi` — the structural dual of the string `is_phi_line` (a phi line
    /// lowers to an inst whose `opcode` is `"phi"`), for CFG analysis that reads the typed carrier.
    pub(in crate::native) fn is_phi(&self) -> bool {
        self.opcode == "phi"
    }
}

/// Render a block's typed carrier back to canonical LLVM-IR text lines — instruction lines followed by
/// the terminator, matching what `split_body_blocks` fed to the retired `.lines` substrate. TEST-ONLY:
/// the CFG-restructuring unit tests were written against `.lines`; this reproduces those lines from the
/// carrier (the sole substrate) so they read structured output as text. `phi`/`br`/`ret`/`unreachable`
/// render exactly (canonical spacing); other instructions render best-effort (`[<res> = ]<opcode>
/// <values>`), enough for the opcode/def/operand-substring assertions those tests make. An incoming or
/// operand value that is not injectively renderable falls back to its `Debug` form.
#[cfg(test)]
pub(in crate::native) fn render_block_lines(block: &TirBlock) -> Vec<String> {
    use crate::native::render::{render_type, render_value};
    fn val(v: &LlValue) -> String {
        render_value(v).unwrap_or_else(|| format!("{v:?}"))
    }
    let mut lines = Vec::with_capacity(block.insts.len() + 1);
    for inst in &block.insts {
        if inst.opcode == "phi" {
            if let Some((ty, incoming)) = &inst.phi_incoming() {
                let ty = render_type(ty).unwrap_or_else(|| format!("{ty:?}"));
                let incoming = incoming
                    .iter()
                    .map(|(value, pred)| format!("[ {}, {pred} ]", val(value)))
                    .collect::<Vec<_>>()
                    .join(", ");
                let result = inst.result.clone().unwrap_or_default();
                lines.push(format!("{result} = phi {ty} {incoming}"));
                continue;
            }
        }
        let mut line = String::new();
        if let Some(result) = &inst.result {
            line.push_str(result);
            line.push_str(" = ");
        }
        line.push_str(&inst.opcode);
        let ops = inst
            .operands
            .iter()
            .filter_map(|op| op.as_typed_value())
            .map(|tv| val(&tv.value))
            .collect::<Vec<_>>();
        if !ops.is_empty() {
            line.push(' ');
            line.push_str(&ops.join(", "));
        }
        lines.push(line);
    }
    lines.push(render_terminator_line(block));
    lines
}

/// Render a block's terminator to canonical LLVM-IR text. TEST-ONLY helper for [`render_block_lines`];
/// `ret`/`switch` render from the emit-ready `ret`/`switch` carriers when present.
#[cfg(test)]
fn render_terminator_line(block: &TirBlock) -> String {
    use crate::native::render::{render_type, render_value};
    match &block.terminator {
        TirTerminator::Br(t) => format!("br label {t}"),
        TirTerminator::BrCond { cond, t, f } => {
            format!("br i1 {cond}, label {t}, label {f}")
        }
        TirTerminator::Ret(_) => match &block.ret {
            RetEmit::Void => "ret void".to_string(),
            RetEmit::Value(tv) => {
                let ty = render_type(&tv.ty).unwrap_or_else(|| format!("{:?}", tv.ty));
                let value = render_value(&tv.value).unwrap_or_else(|| format!("{:?}", tv.value));
                format!("ret {ty} {value}")
            }
            RetEmit::FromText => "ret".to_string(),
        },
        TirTerminator::Unreachable => "unreachable".to_string(),
        TirTerminator::Switch {
            selector,
            default,
            cases,
        } => {
            let sel_ty = block
                .switch
                .as_ref()
                .and_then(|sw| render_type(&sw.selector.ty))
                .unwrap_or_else(|| "i32".to_string());
            let arms = cases
                .iter()
                .map(|(c, l)| format!("{sel_ty} {c}, label {l}"))
                .collect::<Vec<_>>()
                .join(" ");
            format!("switch {sel_ty} {selector}, label {default} [ {arms} ]")
        }
    }
}

/// How a block's `ret` terminator emits. The structured `TirTerminator::Ret` carries only the value's
/// SSA NAME (or `None` for the `Ret(None)` shape), which is not enough to emit: `ReturnValue` needs the
/// operand's TYPE, and the `void` decision must use the exact `rest.trim() == "void"` test on the
/// (metadata-including) `ret ` rest — NOT the structured `Ret(None)`, which strips trailing
/// `, !dbg` metadata differently and so mis-classifies `ret void, !dbg !N`. This carrier records the
/// emit-ready decision computed once at build time via `strip_comment` + `strip_prefix("ret ")` +
/// `parse_typed_value`, so the ret emitter reads it
/// from the typed graph — the raw terminator line is no longer stored. `TirTerminator` derives `Eq`,
/// which a `TypedValue` (float constants) cannot, so this lives on `TirBlock` (no `Eq`) beside it.
#[derive(Clone, Debug)]
pub(super) enum RetEmit {
    /// The terminator is not a `ret`, or is a `ret` whose value did not `parse_typed_value` at build. The
    /// emitter treats this as a fail-visible error (the raw-line emission substrate is retired); it is
    /// measured dead broadly (0 / 16942 frontier + 0 / 15,336 banked), so a hit routes to retry.
    FromText,
    /// `ret void` (the `rest.trim() == "void"` case) — emit `Op::Return`.
    Void,
    /// `ret <ty> <v>` with the operand parsed at build time — emit `Op::ReturnValue` from it.
    Value(TypedValue),
}

/// A typed basic block: a label, its straight-line instructions, and a structured terminator.
#[derive(Clone, Debug)]
pub(super) struct TirBlock {
    pub(super) label: String,
    pub(super) insts: Vec<TirInst>,
    pub(super) terminator: TirTerminator,
    /// The emit-ready `ret` decision (see [`RetEmit`]): the `ret` terminator emits from this typed
    /// carrier, so the walk never re-lexes the value/void classification from a raw terminator line.
    pub(super) ret: RetEmit,
    /// The emit-ready `switch` operands: the parsed `LlSwitch` (typed selector + typed case constants +
    /// target labels) for a `switch` terminator, or `None` for any other terminator OR a `switch` whose
    /// operands did not strict-`parse_switch` at build (a fail-visible emit error — measured dead). Built
    /// once via the byte-identical text-path parse (`tir::switch_emit`), so `switch` emits from the typed
    /// graph instead of re-lexing the line — the structured `TirTerminator::Switch` carries only labels +
    /// case constant TOKENS, not the selector/constant TYPES emission needs.
    pub(super) switch: Option<crate::native::parse::LlSwitch>,
}

impl AsRef<TirBlock> for TirBlock {
    fn as_ref(&self) -> &TirBlock {
        self
    }
}

/// A function parsed once into typed blocks, with every resolvable SSA result's type carried on the
/// value (`value_types`) rather than re-derived at each use.
#[derive(Clone, Debug)]
pub(super) struct TirFunction {
    /// Final structurized carriers shared with the CFG plan. Emission is read-only; retaining the
    /// `Arc`s avoids deep-cloning every instruction in the current function beside the parse-time
    /// module immediately before the emitter walks it.
    pub(super) blocks: Vec<Arc<TirBlock>>,
    pub(super) value_types: HashMap<String, LlType>,
    /// For pointer-typed SSA results, the inferred pointee type. `LlType::Ptr` is addrspace-only, so
    /// the pointee lives here rather than in `value_types`. Populated for `getelementptr` results by
    /// walking the source aggregate along the index path; constant indices resolve struct members,
    /// and dynamic (non-constant) indices still resolve through array/vector steps (element type is
    /// index-independent) — only a dynamic STRUCT-member index leaves the result unresolved.
    pub(super) pointer_pointees: HashMap<String, LlType>,
    /// USE-based pointee map: for a pointer-typed SSA value, the pointee implied by how it is
    /// DEREFERENCED at its use sites — the type a `load` reads through it, a `store` writes through it,
    /// or the source element type of a `getelementptr` rooted at it. This is the dual of
    /// `pointer_pointees` (which carries a GEP RESULT's pointee from its source aggregate): there the
    /// key is the GEP result, here the key is the pointer OPERAND a deref consumes. It is the type the
    /// emitter needs to stop defaulting derived/loaded/aggregate pointers to a byte (`uchar`) pointer
    /// (the R4 pointer-typing foundation). When a pointer's uses disagree, the richer (aggregate/vector
    /// over scalar over byte) pointee is kept — a byte/scalar view of the same storage is the
    /// less-informative one — and the disagreement is reported by the self-check. Not yet consumed by
    /// emission; the byte-conformance gate guards consumption.
    pub(super) use_pointees: HashMap<String, LlType>,
    /// Pointers dereferenced at least once through a BYTE (`i8`) view (a `getelementptr inbounds i8`
    /// byte cursor, an `i8` load/store, or a byte atomic). When such a pointer also has a wider
    /// dereference, its `use_pointees` carrier resolves to the wider type, but the emitter still emits
    /// the byte cursor as a `uchar`-result `OpPtrAccessChain`, which is only well-typed against a
    /// `uchar`-pointee base. This set marks the pointers the M2 byte→real pointee upgrade must NOT flip
    /// (flipping them strands the byte cursor → invalid SPIR-V); the pure-widening subset (no `i8` view)
    /// is absent here and stays upgradeable. Consumed by `pointer_pointee_for_value`.
    pub(super) byte_view_pointers: HashSet<String>,
    /// The SSA result names of every pointer-typed `phi` (`%r = phi ptr ...`) in the function. The M3
    /// (pointer-typing rewrite) migration of the emitter's `pointer_phi_values` side-table onto the
    /// carrier: computed once here during the build instead of by a separate `body_blocks` text-walk in
    /// the emitter. Byte-identical to that walk by construction (same source lines, same `phi ptr`
    /// predicate — see [`collect_pointer_phi_sets`]).
    pub(super) pointer_phi_results: HashSet<String>,
    /// The `%name` incoming VALUES of every pointer-typed `phi` (the values merged by a `phi ptr`,
    /// excluding the block labels). The M3 carrier home of the emitter's `pointer_phi_incoming_values`
    /// side-table; byte-identical to the retired text-walk by construction.
    pub(super) pointer_phi_incoming: HashSet<String>,
    /// Every `getelementptr` result keyed by its SSA name → the parsed `LlGep`. The M3 carrier home of
    /// the emitter's `forward_geps` side-table (formerly the standalone `forward_gep_results`
    /// `body_blocks` text-walk); byte-identical by construction (same lines, same
    /// `strip_prefix("getelementptr ")` + `parse_gep`). See [`collect_forward_geps`].
    pub(super) forward_geps: HashMap<String, LlGep>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_carrier_keeps_sparse_opcode_payloads_compact() {
        assert!(
            std::mem::size_of::<TirInst>() <= 128,
            "TirInst grew to {} bytes",
            std::mem::size_of::<TirInst>()
        );
        assert_eq!(TirOpcode::new("phi".to_string()), TirOpcode::Phi);
        assert_eq!(
            TirOpcode::new("future.op".to_string()).as_str(),
            "future.op"
        );
    }

    #[test]
    fn build_from_blocks_shares_the_existing_carrier() {
        let carrier = Arc::new(
            lower_block_carrier("%entry", &["ret void"], &HashMap::new()).expect("carrier"),
        );
        let blocks = [crate::native::cfg::BodyBlock {
            name: "%entry".to_string(),
            role: crate::native::cfg::BlockRole::Normal,
            typed: Some(Arc::clone(&carrier)),
        }];
        let function = build_from_blocks(&blocks).expect("function");
        assert!(Arc::ptr_eq(&function.blocks[0], &carrier));
    }

    #[test]
    fn unused_geps_are_omitted_to_a_fixpoint_before_emission() {
        let types = HashMap::new();
        let mut value_types = HashMap::new();
        let mut pointees = HashMap::new();
        let block = crate::native::cfg::BodyBlock {
            name: "%entry".to_string(),
            role: crate::native::cfg::BlockRole::Normal,
            typed: Some(Arc::new(
                lower_block(
                    "%entry",
                    &[
                        "%root = alloca { i32 }, align 4",
                        "%dead.parent = getelementptr { i32 }, ptr %root, i32 0, i32 0",
                        "%dead.child = getelementptr i32, ptr %dead.parent, i32 0",
                        "call void @llvm.lifetime.start.p0(ptr %dead.child)",
                        "%live = getelementptr { i32 }, ptr %root, i32 0, i32 0",
                        "%value = load i32, ptr %live, align 4",
                        "ret i32 %value",
                    ],
                    &types,
                    &mut value_types,
                    &mut pointees,
                )
                .expect("typed block"),
            )),
        };

        let blocks = prune_unused_geps(vec![block]);
        let results = blocks[0]
            .typed
            .as_ref()
            .expect("entry")
            .insts
            .iter()
            .filter_map(|instruction| instruction.result.as_deref())
            .collect::<HashSet<_>>();
        assert!(!results.contains("%dead.parent"));
        assert!(!results.contains("%dead.child"));
        assert!(results.contains("%live"));
    }

    #[test]
    fn finalized_pointer_identity_is_substituted_before_emission() {
        let types = HashMap::new();
        let mut value_types = HashMap::new();
        let mut pointees = HashMap::new();
        let mut block = |name: &str, lines: &[&str]| crate::native::cfg::BodyBlock {
            name: name.to_string(),
            role: crate::native::cfg::BlockRole::Normal,
            typed: Some(Arc::new(
                lower_block(name, lines, &types, &mut value_types, &mut pointees)
                    .expect("typed block"),
            )),
        };
        let mut blocks = vec![
            block(
                "%entry",
                &["switch i32 %selector, label %merge [ i32 0, label %merge ]"],
            ),
            block(
                "%merge",
                &[
                    "%p = phi ptr addrspace(2) [ %source, %entry ], [ %source, %entry ]",
                    "%value = load float, ptr addrspace(2) %p",
                    "ret void",
                ],
            ),
        ];

        canonicalize_single_predecessor_phis(&mut blocks);

        let merge = blocks[1].typed.as_ref().expect("merge carrier");
        assert!(!merge.insts.iter().any(|inst| inst.opcode == "phi"));
        let load = merge
            .insts
            .iter()
            .find_map(|inst| inst.load().as_deref())
            .expect("load");
        assert!(matches!(&load.ptr.value, LlValue::Local(name) if name == "%source"));
    }

    #[test]
    fn literal_branch_pruning_removes_dead_blocks_and_canonicalizes_phis() {
        let types = HashMap::new();
        let mut value_types = HashMap::new();
        let mut pointees = HashMap::new();
        let mut block = |name: &str, lines: &[&str]| crate::native::cfg::BodyBlock {
            name: name.to_string(),
            role: crate::native::cfg::BlockRole::Normal,
            typed: Some(Arc::new(
                lower_block(name, lines, &types, &mut value_types, &mut pointees)
                    .expect("typed block"),
            )),
        };
        let blocks = vec![
            block("%entry", &["br i1 false, label %dead, label %live"]),
            block("%dead", &["br label %merge"]),
            block("%live", &["br label %merge"]),
            block(
                "%merge",
                &[
                    "%p = phi ptr addrspace(1) [ null, %dead ], [ %source, %live ]",
                    "%value = load i32, ptr addrspace(1) %p",
                    "ret void",
                ],
            ),
        ];

        let blocks = prune_literal_branch_dead_blocks(blocks);

        assert_eq!(
            blocks
                .iter()
                .map(|block| block.name.as_str())
                .collect::<Vec<_>>(),
            ["%entry", "%live", "%merge"]
        );
        assert_eq!(
            blocks[0].typed.as_ref().expect("entry").terminator,
            TirTerminator::Br("%live".to_string())
        );
        let merge = blocks[2].typed.as_ref().expect("merge");
        assert!(!merge.insts.iter().any(|inst| inst.opcode == "phi"));
        let load = merge
            .insts
            .iter()
            .find_map(|inst| inst.load().as_deref())
            .expect("load");
        assert!(matches!(&load.ptr.value, LlValue::Local(name) if name == "%source"));
    }

    #[test]
    fn literal_branch_pruning_declines_uneditable_phi_carriers() {
        let types = HashMap::new();
        let mut value_types = HashMap::new();
        let mut pointees = HashMap::new();
        let mut block = |name: &str, lines: &[&str]| crate::native::cfg::BodyBlock {
            name: name.to_string(),
            role: crate::native::cfg::BlockRole::Normal,
            typed: Some(Arc::new(
                lower_block(name, lines, &types, &mut value_types, &mut pointees)
                    .expect("typed block"),
            )),
        };
        let blocks = vec![
            block("%entry", &["br i1 false, label %dead, label %live"]),
            block("%dead", &["br label %merge"]),
            block("%live", &["br label %merge"]),
            block(
                "%merge",
                &[
                    "%value = phi <2 x i32> [ <2 x i32> <i32 %a, i32 %b> %dead ], [ undef, %live ]",
                    "ret void",
                ],
            ),
        ];

        let blocks = prune_literal_branch_dead_blocks(blocks);

        assert_eq!(blocks.len(), 4);
        assert!(matches!(
            blocks[0].typed.as_ref().expect("entry").terminator,
            TirTerminator::BrCond { .. }
        ));
    }

    #[test]
    fn literal_branch_pruning_ignores_unaffected_aggregate_phi() {
        let types = HashMap::new();
        let mut value_types = HashMap::new();
        let mut pointees = HashMap::new();
        let mut block = |name: &str, lines: &[&str]| crate::native::cfg::BodyBlock {
            name: name.to_string(),
            role: crate::native::cfg::BlockRole::Normal,
            typed: Some(Arc::new(
                lower_block(name, lines, &types, &mut value_types, &mut pointees)
                    .expect("typed block"),
            )),
        };
        let blocks = vec![
            block("%entry", &["br i1 false, label %dead, label %live"]),
            block("%dead", &["br label %join"]),
            block("%live", &["br label %join"]),
            block("%join", &["br label %aggregate"]),
            block(
                "%aggregate",
                &[
                    "%value = phi <2 x i32> [ <2 x i32> <i32 %a, i32 %b>, %join ]",
                    "ret void",
                ],
            ),
        ];

        let blocks = prune_literal_branch_dead_blocks(blocks);

        assert_eq!(
            blocks
                .iter()
                .map(|block| block.name.as_str())
                .collect::<Vec<_>>(),
            ["%entry", "%live", "%join", "%aggregate"]
        );
        assert_eq!(
            blocks[0].typed.as_ref().expect("entry").terminator,
            TirTerminator::Br("%live".to_string())
        );
    }

    #[test]
    fn an_uneditable_phi_blocks_only_its_own_literal_edge() {
        let types = HashMap::new();
        let mut value_types = HashMap::new();
        let mut pointees = HashMap::new();
        let mut block = |name: &str, lines: &[&str]| crate::native::cfg::BodyBlock {
            name: name.to_string(),
            role: crate::native::cfg::BlockRole::Normal,
            typed: Some(Arc::new(
                lower_block(name, lines, &types, &mut value_types, &mut pointees)
                    .expect("typed block"),
            )),
        };
        let blocks = vec![
            block("%entry", &["br label %unsafe"]),
            block(
                "%unsafe",
                &["br i1 false, label %aggregate, label %unsafe_live"],
            ),
            block("%unsafe_live", &["br label %aggregate"]),
            block(
                "%aggregate",
                &[
                    "%value = phi <2 x i32> [ <2 x i32> <i32 %a, i32 %b> %unsafe ], [ zeroinitializer, %unsafe_live ]",
                    "br label %safe",
                ],
            ),
            block("%safe", &["br i1 false, label %dead, label %live"]),
            block("%dead", &["br label %exit"]),
            block("%live", &["br label %exit"]),
            block("%exit", &["ret void"]),
        ];

        let blocks = prune_literal_branch_dead_blocks(blocks);

        assert!(blocks.iter().all(|block| block.name != "%dead"));
        let unsafe_block = blocks
            .iter()
            .find(|block| block.name == "%unsafe")
            .and_then(|block| block.typed.as_ref())
            .expect("unsafe block");
        assert!(matches!(
            unsafe_block.terminator,
            TirTerminator::BrCond { .. }
        ));
        let safe = blocks
            .iter()
            .find(|block| block.name == "%safe")
            .and_then(|block| block.typed.as_ref())
            .expect("safe block");
        assert_eq!(safe.terminator, TirTerminator::Br("%live".into()));
    }

    #[test]
    fn literal_branch_pruning_repairs_a_still_reachable_target_phi() {
        let types = HashMap::new();
        let mut value_types = HashMap::new();
        let mut pointees = HashMap::new();
        let mut block = |name: &str, lines: &[&str]| crate::native::cfg::BodyBlock {
            name: name.to_string(),
            role: crate::native::cfg::BlockRole::Normal,
            typed: Some(Arc::new(
                lower_block(name, lines, &types, &mut value_types, &mut pointees)
                    .expect("typed block"),
            )),
        };
        let blocks = vec![
            block("%entry", &["br i1 false, label %merge, label %live"]),
            block("%live", &["br label %merge"]),
            block(
                "%merge",
                &[
                    "%value = phi i32 [ 1, %entry ], [ 2, %live ]",
                    "%sum = add i32 %value, 1",
                    "ret void",
                ],
            ),
        ];

        let blocks = prune_literal_branch_dead_blocks(blocks);

        let merge = blocks[2].typed.as_ref().expect("merge");
        assert!(!merge.insts.iter().any(|inst| inst.opcode == "phi"));
        let add = merge
            .insts
            .iter()
            .find(|inst| inst.opcode == "add")
            .expect("add");
        assert!(add
            .operands
            .iter()
            .filter_map(|operand| operand.as_typed_value())
            .any(|value| value.value == LlValue::Int(2)));
    }

    #[test]
    fn phi_canonicalization_rejects_partial_and_conflicting_shapes() {
        let types = HashMap::new();
        let mut value_types = HashMap::new();
        let mut pointees = HashMap::new();
        let mut block = |name: &str, lines: &[&str]| crate::native::cfg::BodyBlock {
            name: name.to_string(),
            role: crate::native::cfg::BlockRole::Normal,
            typed: Some(Arc::new(
                lower_block(name, lines, &types, &mut value_types, &mut pointees)
                    .expect("typed block"),
            )),
        };
        let mut blocks = vec![
            block("%entry", &["br i1 %condition, label %merge, label %other"]),
            block("%other", &["br label %merge"]),
            block(
                "%merge",
                &[
                    "%partial = phi i32 [ 1, %entry ]",
                    "%conflict = phi i32 [ 1, %entry ], [ 2, %entry ]",
                    "ret void",
                ],
            ),
        ];

        canonicalize_single_predecessor_phis(&mut blocks);

        let merge = blocks[2].typed.as_ref().expect("merge carrier");
        assert_eq!(
            merge
                .insts
                .iter()
                .filter(|inst| inst.opcode == "phi")
                .count(),
            2
        );

        let mut conflicting = vec![
            block(
                "%source",
                &["switch i32 %selector, label %conflict [ i32 0, label %conflict ]"],
            ),
            block(
                "%conflict",
                &[
                    "%value = phi float [ -0.0, %source ], [ 0.0, %source ]",
                    "ret void",
                ],
            ),
        ];
        assert!(conflicting[1]
            .typed
            .as_ref()
            .expect("conflict carrier")
            .insts[0]
            .phi_incoming()
            .is_some());
        canonicalize_single_predecessor_phis(&mut conflicting);
        assert!(conflicting[1]
            .typed
            .as_ref()
            .expect("conflict carrier")
            .insts
            .iter()
            .any(|inst| inst.opcode == "phi"));
    }

    /// Build the raw body lines the test-only flat [`build`] consumes (`LlFunction` no longer carries a
    /// `Vec<String>` body — production lowers carriers directly).
    fn func(body: &[&str]) -> Vec<String> {
        body.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn pointer_storage_reaches_alloca_anchor_through_cyclic_phis() {
        use spirv::StorageClass;

        let function = func(&[
            "%slot = phi ptr [ undef, %pre ], [ %next, %continue ]",
            "%next = phi ptr [ %slot, %case0 ], [ %root, %case1 ]",
            "%root = alloca i32, align 4",
            "ret void",
        ]);
        let tir = build(&function, "%entry", &HashMap::new()).expect("build");
        let storage = derive_pointer_storage(&tir, &[], &HashMap::new());

        assert_eq!(storage.get("%root"), Some(&StorageClass::Function));
        assert_eq!(storage.get("%next"), Some(&StorageClass::Function));
        assert_eq!(storage.get("%slot"), Some(&StorageClass::Function));
    }

    #[test]
    fn pointer_storage_reaches_global_anchor_through_gep_and_cyclic_phis() {
        use spirv::StorageClass;

        let function = func(&[
            "%slot = phi ptr addrspace(2) [ undef, %pre ], [ %next, %continue ]",
            "%next = phi ptr addrspace(2) [ %slot, %case0 ], [ %element, %case1 ]",
            "%element = getelementptr [4 x i32], ptr addrspace(2) @table, i64 0, i64 1",
            "ret void",
        ]);
        let tir = build(&function, "%entry", &HashMap::new()).expect("build");
        let seeds = HashMap::from([("@table".to_string(), StorageClass::Private)]);
        let storage = derive_pointer_storage_from(&tir, &[], &HashMap::new(), &seeds);

        assert_eq!(storage.get("%element"), Some(&StorageClass::Private));
        assert_eq!(storage.get("%next"), Some(&StorageClass::Private));
        assert_eq!(storage.get("%slot"), Some(&StorageClass::Private));
    }

    #[test]
    fn redirect_successor_matches_relowered_lines() {
        // The typed `redirect_successor` edit must be byte-identical to re-lowering the
        // string-redirected terminator line, for every terminator shape (this is what keeps a mutated
        // carrier in step with its lines, so the flip stays byte-neutral).
        let types = HashMap::new();
        let cases: &[&[&str]] = &[
            &["br label %old"],
            &["br i1 %c, label %old, label %keep"],
            &["br i1 %c, label %keep, label %old"],
            &["br i1 %c, label %old, label %old"],
            &["switch i32 %s, label %old [ i32 0, label %keep i32 1, label %old ]"],
            &["switch i32 %s, label %keep [ i32 0, label %old ]"],
        ];
        for case in cases {
            let lines: Vec<String> = case.iter().map(|s| s.to_string()).collect();
            let mut carrier = lower_block_carrier("%b", &lines, &types).unwrap();
            carrier.redirect_successor("%old", "%new");
            let redirected: Vec<String> = lines.iter().map(|l| l.replace("%old", "%new")).collect();
            let expected = lower_block_carrier("%b", &redirected, &types).unwrap();
            assert_eq!(
                format!("{carrier:?}"),
                format!("{expected:?}"),
                "redirect diverged from re-lower for {case:?}"
            );
        }
    }

    #[test]
    fn rewrite_phi_predecessor_matches_relowered_lines() {
        // The typed `rewrite_phi_predecessor` edit must be byte-identical to re-lowering the
        // string-rewritten phi lines (predecessor label renamed, values untouched).
        let types = HashMap::new();
        let cases: &[&[&str]] = &[
            &["%r = phi i32 [ %a, %old ], [ %b, %keep ]", "br label %x"],
            &["%r = phi i32 [ %a, %keep ], [ %b, %old ]", "br label %x"],
            &[
                "%r = phi i32 [ %a, %old ], [ %b, %keep ]",
                "%s = phi float [ 0.0, %old ], [ %c, %other ]",
                "br label %x",
            ],
        ];
        for case in cases {
            let lines: Vec<String> = case.iter().map(|s| s.to_string()).collect();
            let mut carrier = lower_block_carrier("%b", &lines, &types).unwrap();
            carrier.rewrite_phi_predecessor("%old", "%new");
            let rewritten: Vec<String> = lines.iter().map(|l| l.replace("%old", "%new")).collect();
            let expected = lower_block_carrier("%b", &rewritten, &types).unwrap();
            assert_eq!(
                format!("{carrier:?}"),
                format!("{expected:?}"),
                "phi-pred rewrite diverged from re-lower for {case:?}"
            );
        }
    }

    #[test]
    fn set_unconditional_branch_matches_relowered_lines() {
        // Replacing the terminator with `br label %sel` on the carrier must equal re-lowering the block
        // with its terminator line swapped for `br label %sel`, for every prior terminator shape.
        let types = HashMap::new();
        let cases: &[&[&str]] = &[
            &["%c = icmp eq i32 %a, %b", "br i1 %c, label %t, label %f"],
            &["ret i32 %v"],
            &["ret void"],
            &["switch i32 %s, label %d [ i32 0, label %k ]"],
            &["br label %z"],
        ];
        for case in cases {
            let lines: Vec<String> = case.iter().map(|s| s.to_string()).collect();
            let mut carrier = lower_block_carrier("%b", &lines, &types).unwrap();
            carrier.set_unconditional_branch("%sel");
            let mut relowered: Vec<String> = lines[..lines.len() - 1].to_vec();
            relowered.push("br label %sel".to_string());
            let expected = lower_block_carrier("%b", &relowered, &types).unwrap();
            assert_eq!(
                format!("{carrier:?}"),
                format!("{expected:?}"),
                "set_unconditional_branch diverged from re-lower for {case:?}"
            );
        }
    }

    #[test]
    fn set_terminator_line_matches_relowered_lines() {
        // Overwriting the terminator line on the carrier (e.g. `unreachable` -> a `ret`) must equal
        // re-lowering the block with its terminator line swapped, across prior + replacement shapes.
        let types = HashMap::new();
        // (original block lines, replacement terminator line).
        let cases: &[(&[&str], &str)] = &[
            (&["%a = add i32 %x, %y", "unreachable"], "ret void"),
            (&["%a = add i32 %x, %y", "unreachable"], "ret i32 undef"),
            (&["unreachable"], "ret <2 x float> undef"),
            (&["br label %z"], "ret void"),
            (&["ret void"], "br label %z"),
            (&["ret void"], "switch i32 %s, label %d [ i32 0, label %k ]"),
        ];
        for (lines, replacement) in cases {
            let src: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
            let mut carrier = lower_block_carrier("%b", &src, &types).unwrap();
            carrier.set_terminator_line(replacement);
            let mut relowered: Vec<String> = src[..src.len() - 1].to_vec();
            relowered.push(replacement.to_string());
            let expected = lower_block_carrier("%b", &relowered, &types).unwrap();
            assert_eq!(
                format!("{carrier:?}"),
                format!("{expected:?}"),
                "set_terminator_line diverged from re-lower for {lines:?} -> {replacement:?}"
            );
        }
    }

    #[test]
    fn terminator_only_block_matches_relowered_line() {
        // Lifting a block's terminator into a fresh terminator-only block must equal re-lowering a block
        // whose single line is that terminator, for every terminator shape (this is what keeps the
        // split_loop_header lift byte-neutral).
        let types = HashMap::new();
        // (source block lines, the terminator line alone).
        let cases: &[(&[&str], &str)] = &[
            (
                &["%c = icmp eq i32 %a, %b", "br i1 %c, label %t, label %f"],
                "br i1 %c, label %t, label %f",
            ),
            (&["br label %z"], "br label %z"),
            (&["ret void"], "ret void"),
            (&["%v = add i32 0, 1", "ret i32 %v"], "ret i32 %v"),
            (
                &["switch i32 %s, label %d [ i32 0, label %k ]"],
                "switch i32 %s, label %d [ i32 0, label %k ]",
            ),
            (&["unreachable"], "unreachable"),
        ];
        for (lines, term_line) in cases {
            let src: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
            let carrier = lower_block_carrier("%src", &src, &types).unwrap();
            let lifted = carrier.terminator_only_block("%sel");
            let expected = lower_block_carrier("%sel", &[term_line.to_string()], &types).unwrap();
            assert_eq!(
                format!("{lifted:?}"),
                format!("{expected:?}"),
                "terminator_only_block diverged from re-lower for {term_line:?}"
            );
        }
    }

    #[test]
    fn terminators_parse_to_structured_forms() {
        assert_eq!(
            parse_terminator("br label %head"),
            Some(TirTerminator::Br("%head".into()))
        );
        assert_eq!(
            parse_terminator("br i1 %c, label %t, label %f"),
            Some(TirTerminator::BrCond {
                cond: "%c".into(),
                t: "%t".into(),
                f: "%f".into()
            })
        );
        assert_eq!(parse_terminator("ret void"), Some(TirTerminator::Ret(None)));
        assert_eq!(
            parse_terminator("ret i32 %v"),
            Some(TirTerminator::Ret(Some("%v".into())))
        );
        assert_eq!(
            parse_terminator("unreachable"),
            Some(TirTerminator::Unreachable)
        );
        assert!(parse_terminator("%r = add i32 %a, %b").is_none());
    }

    #[test]
    fn terminators_tolerate_trailing_metadata() {
        // Loop/debug metadata on a terminator must not break parsing (the common back-edge form).
        assert_eq!(
            parse_terminator("br label %head, !llvm.loop !5"),
            Some(TirTerminator::Br("%head".into()))
        );
        assert_eq!(
            parse_terminator("br i1 %c, label %t, label %f, !llvm.loop !5"),
            Some(TirTerminator::BrCond {
                cond: "%c".into(),
                t: "%t".into(),
                f: "%f".into()
            })
        );
        assert_eq!(
            parse_terminator("ret i32 %v, !dbg !9"),
            Some(TirTerminator::Ret(Some("%v".into())))
        );
    }

    #[test]
    fn switch_terminator_lists_successors() {
        let t = parse_terminator("switch i32 %s, label %def [ i32 1, label %a i32 2, label %b ]")
            .expect("switch parses");
        assert_eq!(t.successors(), vec!["%def", "%a", "%b"]);
        // Case constants are now captured alongside their targets (emission-ready).
        match t {
            TirTerminator::Switch {
                selector,
                default,
                cases,
            } => {
                assert_eq!(selector, "%s");
                assert_eq!(default, "%def");
                assert_eq!(
                    cases,
                    vec![
                        ("1".to_string(), "%a".to_string()),
                        ("2".to_string(), "%b".to_string())
                    ]
                );
            }
            other => panic!("expected switch, got {other:?}"),
        }
    }

    #[test]
    fn result_types_resolved_per_form() {
        let f = func(&[
            "%a = add i32 %x, %y",
            "%b = load <4 x float>, ptr addrspace(1) %p, align 4",
            "%c = icmp slt i32 %a, %a",
            "%d = bitcast i32 %a to float",
            "%e = select i1 %c, i32 %a, i32 %a",
            "%f = fadd fast <2 x float> %b, %b",
            "%g = extractelement <4 x float> %b, i32 0",
            "ret void",
        ]);
        let tir = build(&f, "%entry", &HashMap::new()).expect("build");
        let vt = &tir.value_types;
        assert_eq!(vt.get("%a"), Some(&LlType::Int(32)));
        assert_eq!(
            vt.get("%b"),
            Some(&LlType::Vector(Box::new(LlType::Float), 4))
        );
        assert_eq!(vt.get("%c"), Some(&LlType::Bool));
        assert_eq!(vt.get("%d"), Some(&LlType::Float));
        assert_eq!(vt.get("%e"), Some(&LlType::Int(32)));
        assert_eq!(
            vt.get("%f"),
            Some(&LlType::Vector(Box::new(LlType::Float), 2))
        );
        assert_eq!(vt.get("%g"), Some(&LlType::Float));
    }

    #[test]
    fn more_instruction_forms_resolve() {
        let f = func(&[
            "%n = fneg fast float %x",
            "%z = freeze i32 %x",
            "%a = alloca <4 x float>, align 16",
            "%s = shufflevector <4 x float> %v, <4 x float> %v, <2 x i32> <i32 0, i32 1>",
            "%c = tail call fast <4 x float> %fp(ptr %p)",
            "%d = call i32 @air.foo(i32 %x)",
            "ret void",
        ]);
        let tir = build(&f, "%entry", &HashMap::new()).unwrap();
        let vt = &tir.value_types;
        assert_eq!(vt.get("%n"), Some(&LlType::Float));
        assert_eq!(vt.get("%z"), Some(&LlType::Int(32)));
        assert_eq!(vt.get("%a"), Some(&LlType::Ptr(0)));
        // shufflevector: input element type, mask length.
        assert_eq!(
            vt.get("%s"),
            Some(&LlType::Vector(Box::new(LlType::Float), 2))
        );
        // indirect call return type resolves (callee is %reg, not @name).
        assert_eq!(
            vt.get("%c"),
            Some(&LlType::Vector(Box::new(LlType::Float), 4))
        );
        assert_eq!(vt.get("%d"), Some(&LlType::Int(32)));
    }

    #[test]
    fn extractvalue_walks_aggregate_index_path() {
        let f = func(&[
            "%a = extractvalue { float, i32 } %v, 1",
            "%b = extractvalue [4 x float] %arr, 2",
            "%c = extractvalue { i32, { float, i32 } } %n, 1, 0",
            "ret void",
        ]);
        let tir = build(&f, "%entry", &HashMap::new()).unwrap();
        let vt = &tir.value_types;
        assert_eq!(vt.get("%a"), Some(&LlType::Int(32)));
        assert_eq!(vt.get("%b"), Some(&LlType::Float));
        assert_eq!(vt.get("%c"), Some(&LlType::Float)); // nested struct member
    }

    #[test]
    fn extractvalue_resolves_named_struct_via_type_table() {
        // `extractvalue` into an opaque named struct needs the module's `%T = type {...}` table.
        let f = func(&["%a = extractvalue %struct.Foo %v, 1", "ret void"]);
        let mut types = HashMap::new();
        types.insert(
            "%struct.Foo".to_string(),
            LlType::Struct(vec![LlType::Float, LlType::Int(32)]),
        );
        assert_eq!(
            build(&f, "%entry", &types).unwrap().value_types.get("%a"),
            Some(&LlType::Int(32))
        );
        // Without the table the same extract is unresolved.
        assert_eq!(
            build(&f, "%entry", &HashMap::new())
                .unwrap()
                .value_types
                .get("%a"),
            None
        );
    }

    #[test]
    fn fcmp_with_flags_resolves_to_bool() {
        // The fast-math flag + predicate must not hide the operand type.
        let f = func(&["%c = fcmp fast olt float %a, %b", "ret void"]);
        let tir = build(&f, "%entry", &HashMap::new()).unwrap();
        assert_eq!(tir.value_types.get("%c"), Some(&LlType::Bool));
    }

    #[test]
    fn icmp_on_vector_yields_bool_vector() {
        let f = func(&["%c = icmp eq <4 x i32> %a, %a", "ret void"]);
        let tir = build(&f, "%entry", &HashMap::new()).unwrap();
        assert_eq!(
            tir.value_types.get("%c"),
            Some(&LlType::Vector(Box::new(LlType::Bool), 4))
        );
    }

    #[test]
    fn instruction_uses_capture_value_operands_not_labels() {
        let f = func(&[
            "%a = add i32 %x, %y",
            "%p = phi i32 [ %a, %entry ], [ %z, %loop ]",
            "store i32 %a, ptr %dst",
            "%old = atomicrmw add ptr %dst, i32 %a seq_cst",
            "ret void",
        ]);
        let tir = build(&f, "%entry", &HashMap::new()).unwrap();
        let inst = |r: &str| {
            tir.blocks[0]
                .insts
                .iter()
                .find(|i| i.result.as_deref() == Some(r))
                .unwrap()
        };
        let mut add_uses = Vec::new();
        inst("%a").visit_uses(|name| add_uses.push(name.to_string()));
        assert_eq!(add_uses, vec!["%x", "%y"]);
        assert!(inst("%a").uses.is_none());
        // Phi use traversal reads the canonical incoming VALUES (%a, %z), not the predecessor labels
        // (%entry, %loop), without retaining a parallel vector of names.
        let mut phi_uses = Vec::new();
        inst("%p").visit_uses(|name| phi_uses.push(name.to_string()));
        assert_eq!(phi_uses, vec!["%a", "%z"]);
        assert!(inst("%p").uses.is_none());
        let store = tir.blocks[0]
            .insts
            .iter()
            .find(|i| i.opcode == "store")
            .unwrap();
        assert_eq!(store.result, None);
        let mut store_uses = Vec::new();
        store.visit_uses(|name| store_uses.push(name.to_string()));
        assert_eq!(store_uses, vec!["%a", "%dst"]);
        assert!(store.uses.is_none());

        // Unsupported operand layouts retain their lossless scan until a typed carrier exists.
        let atomic = inst("%old");
        let mut atomic_uses = Vec::new();
        atomic.visit_uses(|name| atomic_uses.push(name.to_string()));
        assert_eq!(atomic_uses, vec!["%dst", "%a"]);
        assert_eq!(atomic.uses.as_ref().map(Vec::len), Some(2));
    }

    #[test]
    fn resolve_call_parses_direct_and_rejects_non_calls() {
        // A direct call carries its callee + typed args; a `tail call` variant drops the keyword first.
        let c = resolve_call(
            "%r = call float @air.atomic.global.add.f.f32(float addrspace(1)* %p, float %v, i32 0)",
        )
        .expect("direct call resolves");
        assert_eq!(c.callee, "air.atomic.global.add.f.f32");
        assert_eq!(c.args.len(), 3);
        assert_eq!(c.ret, LlType::Float);
        let t = resolve_call("%r = tail call i32 @foo(i32 %x)").expect("tail call resolves");
        assert_eq!(t.callee, "foo");
        // Non-call opcodes and indirect calls (no `@callee`) yield no carrier.
        assert!(resolve_call("%a = add i32 %x, %y").is_none());
        assert!(resolve_call("%r = call void %fnptr(i32 %x)").is_none());
    }

    #[test]
    fn call_views_share_the_primary_parse_and_preserve_narrower_contracts() {
        let direct = build(
            &func(&["%r = call i32 @foo(i32 %x)", "ret void"]),
            "%entry",
            &HashMap::new(),
        )
        .expect("direct call");
        let inst = &direct.blocks[0].insts[0];
        let primary = inst.call().as_deref().expect("primary parse");
        assert!(std::ptr::eq(
            primary,
            inst.alias_call().expect("alias view")
        ));
        assert!(std::ptr::eq(
            primary,
            inst.emit_scan_call()
                .expect("emit scan")
                .expect("valid emit scan")
        ));

        let musttail = build(
            &func(&["%r = musttail call i32 @foo(i32 %x)", "ret void"]),
            "%entry",
            &HashMap::new(),
        )
        .expect("musttail call");
        let inst = &musttail.blocks[0].insts[0];
        assert!(inst.call().is_some());
        assert!(inst.alias_call().is_none());
        assert!(inst.emit_scan_call().is_none());

        let malformed = build(
            &func(&["%r = call i32 @foo(", "ret void"]),
            "%entry",
            &HashMap::new(),
        )
        .expect("malformed call carrier");
        let inst = &malformed.blocks[0].insts[0];
        assert!(inst.call().is_none());
        assert!(inst.emit_scan_call().is_some_and(|result| result.is_err()));
    }

    #[test]
    fn atomic_call_pointees_read_the_call_carrier() {
        // `atomic_call_pointees` sources the pointer + element type from the `TirInst.call()` carrier
        // (no `inst.text` re-lex): a value-returning atomic types its pointer arg with the CALL RESULT.
        let f = func(&[
            "%r = call float @air.atomic.global.add.f.f32(ptr addrspace(1) %p, float %v, i32 0)",
            "ret void",
        ]);
        let tir = build(&f, "%entry", &HashMap::new()).expect("build");
        let inst = tir.blocks[0]
            .insts
            .iter()
            .find(|i| i.result.as_deref() == Some("%r"))
            .expect("atomic call inst");
        assert!(
            inst.call().is_some(),
            "the CALL carrier is populated at build time"
        );
        assert_eq!(
            atomic_call_pointees(inst),
            vec![("%p".to_string(), LlType::Float)]
        );
        // A void atomic store types its pointer from the first non-pointer (stored-value) arg instead.
        let f2 = func(&[
            "call void @air.atomic.global.store.f.f32(ptr addrspace(1) %q, float %w, i32 0)",
            "ret void",
        ]);
        let tir2 = build(&f2, "%entry", &HashMap::new()).expect("build");
        let store = tir2.blocks[0]
            .insts
            .iter()
            .find(|i| i.opcode == "call")
            .expect("void atomic call inst");
        assert_eq!(
            atomic_call_pointees(store),
            vec![("%q".to_string(), LlType::Float)]
        );
        // A non-atomic call contributes no use-pointees even though its carrier is populated.
        let f3 = func(&["%s = call i32 @ordinary(ptr addrspace(1) %z)", "ret void"]);
        let tir3 = build(&f3, "%entry", &HashMap::new()).expect("build");
        let other = tir3.blocks[0]
            .insts
            .iter()
            .find(|i| i.result.as_deref() == Some("%s"))
            .expect("ordinary call inst");
        assert!(other.call().is_some());
        assert!(atomic_call_pointees(other).is_empty());
    }

    #[test]
    fn ret_emit_classifies_like_the_text_path() {
        // `ret void` -> Void; a typed value -> Value carrying its type; a non-ret -> FromText.
        assert!(matches!(ret_emit("ret void"), RetEmit::Void));
        assert!(matches!(
            ret_emit("ret i32 %v"),
            RetEmit::Value(TypedValue {
                ty: LlType::Int(32),
                ..
            })
        ));
        assert!(matches!(
            ret_emit("ret i32 7"),
            RetEmit::Value(TypedValue {
                ty: LlType::Int(32),
                ..
            })
        ));
        assert!(matches!(ret_emit("br label %x"), RetEmit::FromText));
        // The edge case the carrier exists for: `ret void, !dbg !N`. The text path strips the `ret `
        // prefix but NOT the trailing metadata, so `rest.trim()` is `"void, !dbg !9"` (not `"void"`) and
        // the value parse then fails — the text path errors. `ret_emit` mirrors this exactly (FromText,
        // routed to the text path to reproduce that error), rather than the structured `Ret(None)` which
        // strips the metadata and would wrongly classify it as void.
        assert!(matches!(ret_emit("ret void, !dbg !9"), RetEmit::FromText));
    }

    #[test]
    fn phi_incoming_carrier_parses_type_and_labels() {
        // The carrier holds the phi's parsed (unresolved) type and its (value, predecessor) pairs — the
        // labels the graph `operands` drop. A non-phi line yields `None`.
        let (ty, incoming) = phi_incoming_parse("%p = phi i32 [ %a, %entry ], [ 0, %loop ]")
            .0
            .expect("phi parses");
        assert_eq!(ty, LlType::Int(32));
        assert_eq!(incoming.len(), 2);
        assert_eq!(incoming[0].1, "%entry");
        assert_eq!(incoming[1].1, "%loop");
        assert!(phi_incoming_parse("%a = add i32 %x, %y").0.is_none());
    }

    #[test]
    fn array_phi_incoming_does_not_confuse_the_type_bracket_for_an_operand() {
        let (ty, incoming) = phi_incoming_parse("%p = phi [14 x i8] [ %a, %entry ], [ %b, %loop ]")
            .0
            .expect("array phi parses");
        assert_eq!(ty, LlType::Array(Box::new(LlType::Int(8)), 14));
        assert_eq!(incoming.len(), 2);
        assert_eq!(incoming[0].1, "%entry");
        assert_eq!(incoming[1].1, "%loop");
    }

    #[test]
    fn body_carries_phi_incoming_on_the_block() {
        let f = func(&[
            "%i = phi i32 [ 0, %entry ], [ %i2, %loop ]",
            "%i2 = add i32 %i, 1",
            "ret void",
        ]);
        let tir = build(&f, "%entry", &HashMap::new()).expect("build");
        let phi = tir.blocks[0]
            .insts
            .iter()
            .find(|i| i.result.as_deref() == Some("%i"))
            .unwrap();
        let (ty, incoming) = phi.phi_incoming().as_ref().expect("phi carrier present");
        assert_eq!(*ty, LlType::Int(32));
        assert_eq!(incoming[1].1, "%loop");
        // A non-phi inst carries None.
        let add = tir.blocks[0]
            .insts
            .iter()
            .find(|i| i.result.as_deref() == Some("%i2"))
            .unwrap();
        assert!(add.phi_incoming().is_none());
    }

    #[test]
    fn switch_emit_parses_like_the_text_path() {
        // A `switch` line parses to the typed `LlSwitch`; a non-switch yields `None`.
        let sw = switch_emit("switch i32 %s, label %def [ i32 1, label %a i32 2, label %b ]")
            .expect("switch parses");
        assert_eq!(sw.selector.ty, LlType::Int(32));
        assert_eq!(sw.default_label, "%def");
        assert_eq!(sw.cases.len(), 2);
        assert!(switch_emit("ret void").is_none());
        assert!(switch_emit("br label %x").is_none());
    }

    #[test]
    fn body_carries_switch_on_the_block() {
        let f = func(&[
            "switch i32 %s, label %def [ i32 0, label %a ]",
            "def:",
            "ret void",
            "a:",
            "ret void",
        ]);
        let tir = build(&f, "%entry", &HashMap::new()).expect("build");
        let sw = tir.blocks[0]
            .switch
            .as_ref()
            .expect("switch carrier present");
        assert_eq!(sw.default_label, "%def");
        // A ret block carries no switch.
        assert!(tir.blocks[1].switch.is_none());
    }

    #[test]
    fn body_carries_ret_emit_on_the_block() {
        let f = func(&["%a = add i32 0, 1", "ret i32 %a"]);
        let tir = build(&f, "%entry", &HashMap::new()).expect("build");
        assert!(matches!(
            tir.blocks[0].ret,
            RetEmit::Value(TypedValue {
                ty: LlType::Int(32),
                ..
            })
        ));
        let v = func(&["ret void"]);
        let tir = build(&v, "%entry", &HashMap::new()).expect("build");
        assert!(matches!(tir.blocks[0].ret, RetEmit::Void));
    }

    #[test]
    fn body_splits_into_typed_blocks_with_terminators() {
        let f = func(&[
            "%a = add i32 0, 1",
            "br label %loop",
            "loop:",
            "%i = phi i32 [ 0, %entry ], [ %i2, %loop ]",
            "%i2 = add i32 %i, 1",
            "br i1 %c, label %loop, label %done",
            "done:",
            "ret void",
        ]);
        let tir = build(&f, "%entry", &HashMap::new()).expect("build");
        assert_eq!(tir.blocks.len(), 3);
        assert_eq!(tir.blocks[0].label, "%entry");
        assert_eq!(tir.blocks[0].terminator, TirTerminator::Br("%loop".into()));
        assert_eq!(tir.blocks[1].label, "%loop");
        assert_eq!(
            tir.blocks[1].terminator,
            TirTerminator::BrCond {
                cond: "%c".into(),
                t: "%loop".into(),
                f: "%done".into()
            }
        );
        // phi result type carried on the value.
        assert_eq!(tir.value_types.get("%i"), Some(&LlType::Int(32)));
        assert_eq!(tir.blocks[2].label, "%done");
        assert_eq!(tir.blocks[2].terminator, TirTerminator::Ret(None));
    }

    #[test]
    fn getelementptr_resolves_to_base_address_space() {
        let f = func(&[
            "%p = getelementptr inbounds float, ptr addrspace(1) %base, i64 4",
            "%q = getelementptr inbounds i8, ptr %local, i64 2",
            "ret void",
        ]);
        let tir = build(&f, "%entry", &HashMap::new()).unwrap();
        // The GEP result is a pointer in the base operand's address space (Ptr is addrspace-only).
        assert_eq!(tir.value_types.get("%p"), Some(&LlType::Ptr(1)));
        assert_eq!(tir.value_types.get("%q"), Some(&LlType::Ptr(0)));
    }

    #[test]
    fn getelementptr_resolves_nested_struct_pointee() {
        // The bb1f5672 shape: a GEP walks struct -> field -> sub-element; the pointee is recorded so
        // a later reinterpret load can see it (the string emitter's per-site map under-tracks this).
        let mut named = HashMap::new();
        // %struct.S = { float, { half, i8 } }
        named.insert(
            "%struct.S".to_string(),
            LlType::Struct(vec![
                LlType::Float,
                LlType::Struct(vec![LlType::Half, LlType::Int(8)]),
            ]),
        );
        let f = func(&[
            // i64 0 = pointer-stride (skipped); i32 1 -> field 1 (inner struct); i32 0 -> half.
            "%p = getelementptr inbounds %struct.S, ptr addrspace(1) %base, i64 0, i32 1, i32 0",
            // A flat field access: i32 0 -> float.
            "%q = getelementptr inbounds %struct.S, ptr addrspace(1) %base, i64 0, i32 0",
            // Stride-only pointer arithmetic (no aggregate walk): pointee is the element type.
            "%s = getelementptr inbounds float, ptr addrspace(1) %base, i64 %idx",
            // A DYNAMIC array index still resolves: the array element type is index-independent.
            "%a = getelementptr inbounds [8 x half], ptr addrspace(1) %base, i64 0, i64 %idx",
            // A dynamic (non-constant) STRUCT-field index can't be walked -> no pointee.
            "%d = getelementptr inbounds %struct.S, ptr addrspace(1) %base, i64 0, i32 %fld",
            "ret void",
        ]);
        let tir = build(&f, "%entry", &named).unwrap();
        assert_eq!(tir.pointer_pointees.get("%p"), Some(&LlType::Half));
        assert_eq!(tir.pointer_pointees.get("%q"), Some(&LlType::Float));
        assert_eq!(tir.pointer_pointees.get("%s"), Some(&LlType::Float));
        assert_eq!(tir.pointer_pointees.get("%a"), Some(&LlType::Half));
        assert_eq!(tir.pointer_pointees.get("%d"), None);
        // The result types are still the addrspace-only pointers.
        assert_eq!(tir.value_types.get("%p"), Some(&LlType::Ptr(1)));
    }

    /// A compact `(kind, type)` summary of a resolved operand for assertions.
    fn op(o: &TirOperand) -> (&'static str, Option<LlType>) {
        match o {
            TirOperand::Value { ty, .. } => ("val", Some(ty.clone())),
            TirOperand::Const { ty, .. } => ("const", Some(ty.clone())),
            TirOperand::Unresolved => ("unres", None),
        }
    }

    #[test]
    fn resolve_operands_typed_shapes() {
        // Binary: both operands carry the shared declared type; the bare second operand too.
        let ops = resolve_operands("%r = add nsw i32 %a, %b");
        assert_eq!(op(&ops[0]), ("val", Some(LlType::Int(32))));
        assert_eq!(op(&ops[1]), ("val", Some(LlType::Int(32))));
        // A literal second operand becomes a typed Const.
        let ops = resolve_operands("%r = add i32 %a, 7");
        assert_eq!(op(&ops[1]), ("const", Some(LlType::Int(32))));
        // Compare: predicate skipped; both operands share the compared type; result is bool elsewhere.
        let ops = resolve_operands("%r = icmp slt i32 %a, %b");
        assert_eq!(op(&ops[0]), ("val", Some(LlType::Int(32))));
        assert_eq!(op(&ops[1]), ("val", Some(LlType::Int(32))));
        // Select: each field independently typed.
        let ops = resolve_operands("%r = select i1 %c, float %a, float %b");
        assert_eq!(op(&ops[0]), ("val", Some(LlType::Int(1))));
        assert_eq!(op(&ops[1]), ("val", Some(LlType::Float)));
        assert_eq!(op(&ops[2]), ("val", Some(LlType::Float)));
        // Conversion: one value operand (the `to <ty2>` target is not an operand).
        let ops = resolve_operands("%r = fptrunc float %a to half");
        assert_eq!(ops.len(), 1);
        assert_eq!(op(&ops[0]), ("val", Some(LlType::Float)));
        // Load: the pointer is the operand (the loaded type field is not).
        let ops = resolve_operands("%r = load i32, ptr addrspace(1) %p, align 4");
        assert_eq!(ops.len(), 1);
        assert_eq!(op(&ops[0]), ("val", Some(LlType::Ptr(1))));
        // Store: value then pointer; align is not an operand.
        let ops = resolve_operands("store i32 %v, ptr addrspace(1) %p, align 4");
        assert_eq!(op(&ops[0]), ("val", Some(LlType::Int(32))));
        assert_eq!(op(&ops[1]), ("val", Some(LlType::Ptr(1))));
        // Phi: one operand per incoming, all sharing the phi type.
        let ops = resolve_operands("%r = phi i32 [ %a, %l0 ], [ 0, %l1 ]");
        assert_eq!(op(&ops[0]), ("val", Some(LlType::Int(32))));
        assert_eq!(op(&ops[1]), ("const", Some(LlType::Int(32))));
        // extractelement: vector then index, each independently typed.
        let ops = resolve_operands("%r = extractelement <4 x float> %v, i32 2");
        assert_eq!(ops.len(), 2);
        assert_eq!(
            op(&ops[0]),
            ("val", Some(LlType::Vector(Box::new(LlType::Float), 4)))
        );
        assert_eq!(op(&ops[1]), ("const", Some(LlType::Int(32))));
        // insertelement: vector, inserted element, index.
        let ops = resolve_operands("%r = insertelement <4 x float> %v, float %e, i32 %i");
        assert_eq!(ops.len(), 3);
        assert_eq!(
            op(&ops[0]),
            ("val", Some(LlType::Vector(Box::new(LlType::Float), 4)))
        );
        assert_eq!(op(&ops[1]), ("val", Some(LlType::Float)));
        assert_eq!(op(&ops[2]), ("val", Some(LlType::Int(32))));
        // shufflevector: the two source vectors are operands; the mask's inner commas stay grouped.
        let ops = resolve_operands(
            "%r = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 3>",
        );
        assert_eq!(ops.len(), 3);
        assert_eq!(
            op(&ops[0]),
            ("val", Some(LlType::Vector(Box::new(LlType::Float), 4)))
        );
        assert_eq!(
            op(&ops[1]),
            ("val", Some(LlType::Vector(Box::new(LlType::Float), 4)))
        );
        // extractvalue: only the aggregate is an operand; trailing index literals are not. The
        // aggregate's struct type is carried as-is (kind checked; struct payload not over-asserted).
        let ops = resolve_operands("%r = extractvalue { i32, float } %s, 1");
        assert_eq!(ops.len(), 1);
        assert_eq!(op(&ops[0]).0, "val");
        // insertvalue: aggregate + inserted element; trailing index literals are not operands.
        let ops = resolve_operands("%r = insertvalue { i32, float } %s, float %e, 1");
        assert_eq!(ops.len(), 2);
        assert_eq!(op(&ops[0]).0, "val");
        assert_eq!(op(&ops[1]), ("val", Some(LlType::Float)));
        // getelementptr: the source element type (first field) is not an operand; the base pointer
        // and each index are. `ptr` is addrspace-only so the base carries `Ptr(0)`.
        let ops = resolve_operands("%r = getelementptr i8, ptr %p, i64 %i");
        assert_eq!(ops.len(), 2);
        assert_eq!(op(&ops[0]), ("val", Some(LlType::Ptr(0))));
        assert_eq!(op(&ops[1]), ("val", Some(LlType::Int(64))));
        // GEP with a constant index and an addrspaced base + struct walk.
        let ops = resolve_operands(
            "%r = getelementptr inbounds %struct.X, ptr addrspace(1) %p, i32 0, i32 2",
        );
        assert_eq!(ops.len(), 3);
        assert_eq!(op(&ops[0]), ("val", Some(LlType::Ptr(1))));
        assert_eq!(op(&ops[1]), ("const", Some(LlType::Int(32))));
        assert_eq!(op(&ops[2]), ("const", Some(LlType::Int(32))));
        // call: each argument is a value operand; the callee and return type are not.
        let ops = resolve_operands("%r = call i32 @f(i32 %a, float %b)");
        assert_eq!(ops.len(), 2);
        assert_eq!(op(&ops[0]), ("val", Some(LlType::Int(32))));
        assert_eq!(op(&ops[1]), ("val", Some(LlType::Float)));
        // tail call: the `call` keyword is dropped, then resolved as a call.
        let ops = resolve_operands("%r = tail call float @g(float %x)");
        assert_eq!(ops.len(), 1);
        assert_eq!(op(&ops[0]), ("val", Some(LlType::Float)));
        // A direct call with no arguments has no value operands.
        let ops = resolve_operands("%r = call i32 @h()");
        assert!(ops.is_empty());
        // An indirect call (no `@callee`) is left Unresolved — the emitter rejects it.
        let ops = resolve_operands("%r = call i32 %fnptr(i32 %a)");
        assert_eq!(ops.len(), 1);
        assert_eq!(op(&ops[0]), ("unres", None));
        // alloca with no dynamic count has no value operands (the type + align are not operands).
        let ops = resolve_operands("%r = alloca i32, align 4");
        assert!(ops.is_empty());
        // alloca with a dynamic element count: the count is the one value operand.
        let ops = resolve_operands("%r = alloca i32, i32 %n, align 4");
        assert_eq!(ops.len(), 1);
        assert_eq!(op(&ops[0]), ("val", Some(LlType::Int(32))));
        // An opcode still not lowered (atomicrmw) yields a single Unresolved marker.
        let ops = resolve_operands("%r = atomicrmw add ptr %p, i32 1 seq_cst");
        assert_eq!(ops.len(), 1);
        assert_eq!(op(&ops[0]), ("unres", None));
    }

    #[test]
    fn use_pointees_inferred_from_load_store_gep() {
        // `%p` is a bare pointer param-like value; its pointee is pinned by how it is dereferenced:
        // a load reads f32 through it, a GEP indexes a `[4 x float]` through `%q`, a store writes i32
        // through `%s`.
        let tir = build(
            &func(&[
                "%a = load float, ptr %p",
                "%e = getelementptr inbounds [4 x float], ptr %q, i64 0, i64 %i",
                "store i32 %v, ptr %s",
                "ret void",
            ]),
            "%entry",
            &HashMap::new(),
        )
        .expect("build");
        assert_eq!(tir.use_pointees.get("%p"), Some(&LlType::Float));
        assert_eq!(
            tir.use_pointees.get("%q"),
            Some(&LlType::Array(Box::new(LlType::Float), 4))
        );
        assert_eq!(tir.use_pointees.get("%s"), Some(&LlType::Int(32)));
        // GEP-result vs use-based are distinct maps: `%q` is keyed in use_pointees (it is a GEP BASE),
        // while the GEP RESULT `%e` is keyed in pointer_pointees, not use_pointees.
        assert!(!tir.use_pointees.contains_key("%e"));
    }

    #[test]
    fn use_pointee_conflict_prefers_richer_view_and_is_counted() {
        // `%p` is read once as a raw byte (i8) and once as a `<4 x float>` — a genuine reinterpret.
        // The richer (vector) view wins, and the disagreement is counted.
        let (map, conflicts, byte_viewed) = infer_use_pointees(
            &build(
                &func(&[
                    "%a = load i8, ptr %p",
                    "%b = load <4 x float>, ptr %p",
                    "ret void",
                ]),
                "%entry",
                &HashMap::new(),
            )
            .expect("build")
            .blocks,
        );
        assert_eq!(
            map.get("%p"),
            Some(&LlType::Vector(Box::new(LlType::Float), 4))
        );
        assert_eq!(conflicts, 1);
        // `%p` also carries a byte (i8) view, so it is flagged NOT-upgradeable: emitting its byte cursor
        // as `uchar` and then widening its pointee would strand the cursor (invalid SPIR-V).
        assert!(byte_viewed.contains("%p"));
    }

    #[test]
    fn pure_widening_pointer_is_not_byte_viewed() {
        // `%p` is only ever dereferenced as the wider type (no `i8` view). It is the pure-widening
        // subset the byte→real upgrade IS allowed to flip, so it must be absent from `byte_view_pointers`.
        let (map, _conflicts, byte_viewed) = infer_use_pointees(
            &build(
                &func(&["%b = load half, ptr %p", "ret void"]),
                "%entry",
                &HashMap::new(),
            )
            .expect("build")
            .blocks,
        );
        assert_eq!(map.get("%p"), Some(&LlType::Half));
        assert!(!byte_viewed.contains("%p"));
    }

    #[test]
    fn typed_gep_off_byte_cursor_result_is_byte_viewed() {
        // The `native_byte_view_multiroot_phi…` shape: a byte cursor (`getelementptr i8`) is bitcast and
        // then a TYPED gep (`getelementptr float`) is taken off the alias and dereferenced wide. The
        // typed-gep result addresses byte-granular storage at a byte offset, so it must be flagged
        // byte-viewed — otherwise the byte→real upgrade fires on it and emits a misaligned direct typed
        // load instead of the required byte assembly. The taint must flow i8-cursor → bitcast → typed gep.
        let (_map, _conflicts, byte_viewed) = infer_use_pointees(
            &build(
                &func(&[
                    "%byte = getelementptr i8, ptr %p, i64 %o",
                    "%alias = bitcast ptr %byte to ptr",
                    "%fp = getelementptr float, ptr %alias, i64 %o",
                    "%v = load float, ptr %fp",
                    "ret void",
                ]),
                "%entry",
                &HashMap::new(),
            )
            .expect("build")
            .blocks,
        );
        // The i8 cursor and its bitcast alias are byte-viewed (pre-existing taint) …
        assert!(byte_viewed.contains("%p"));
        assert!(byte_viewed.contains("%byte"));
        assert!(byte_viewed.contains("%alias"));
        // … and the fix: the typed gep taken off the byte-cursor alias is byte-viewed too.
        assert!(byte_viewed.contains("%fp"));
    }

    #[test]
    fn use_pointees_inferred_from_air_atomic_buffer_calls() {
        // A device pointer reached ONLY through atomics: `%p` via a value-returning add (pointee =
        // result type i32), `%q` + `%exp` via a cmpxchg (target AND expected-value pointer both point
        // at the element, i32), `%s` via a void store (pointee = first non-pointer arg, i32). A texture
        // atomic (`air.atomic_*_texture_*`, UNDERSCORE prefix) must NOT contribute a buffer pointee.
        let tir = build(
            &func(&[
                "%a = tail call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) captures(none) %p, i32 %v, i32 0, i32 2, i1 true)",
                "%c = tail call i32 @air.atomic.global.cmpxchg.weak.i32(ptr addrspace(1) captures(none) %q, ptr nonnull captures(none) %exp, i32 %d, i32 0, i32 0, i32 2, i1 true)",
                "tail call void @air.atomic.local.store.i32(ptr addrspace(3) captures(none) %s, i32 %v, i32 0, i32 2, i1 true)",
                "%t = tail call <4 x i32> @air.atomic_exchange_explicit_texture_2d.u.v4i32(ptr addrspace(1) %tex, <2 x i32> %coord, <4 x i32> %val)",
                "ret void",
            ]),
            "%entry",
            &HashMap::new(),
        )
        .expect("build");
        assert_eq!(tir.use_pointees.get("%p"), Some(&LlType::Int(32)));
        assert_eq!(tir.use_pointees.get("%q"), Some(&LlType::Int(32)));
        assert_eq!(tir.use_pointees.get("%exp"), Some(&LlType::Int(32)));
        assert_eq!(tir.use_pointees.get("%s"), Some(&LlType::Int(32)));
        // The texture-atomic operand is excluded structurally (dotted prefix only).
        assert!(!tir.use_pointees.contains_key("%tex"));
    }

    #[test]
    fn use_pointees_propagate_across_pointer_merges() {
        // `%sel` selects between `%a` and `%b`; only `%a` is dereferenced (load f32). The merge result
        // `%sel` and the un-dereferenced arm `%b` alias the same memory, so propagation flows the f32
        // pointee to both. `%phi` then merges `%sel` with `%c` and is itself only passed to a call
        // (never dereferenced) — propagation still types it and `%c` through the chain.
        let tir = build(
            &func(&[
                "%v = load float, ptr %a",
                "%sel = select i1 %cond, ptr %a, ptr %b",
                "%phi = phi ptr [ %sel, %x ], [ %c, %y ]",
                "%r = call i32 @use(ptr %phi)",
                "ret void",
            ]),
            "%entry",
            &HashMap::new(),
        )
        .expect("build");
        assert_eq!(tir.use_pointees.get("%a"), Some(&LlType::Float));
        // Propagated to the merge result and the un-dereferenced arm.
        assert_eq!(tir.use_pointees.get("%sel"), Some(&LlType::Float));
        assert_eq!(tir.use_pointees.get("%b"), Some(&LlType::Float));
        // And through the chained phi to its result + other incoming.
        assert_eq!(tir.use_pointees.get("%phi"), Some(&LlType::Float));
        assert_eq!(tir.use_pointees.get("%c"), Some(&LlType::Float));
    }
}
