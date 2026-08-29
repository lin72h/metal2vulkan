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
            if let Some((_, incoming)) = &mut inst.phi_incoming {
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
    /// (`lower_unreachable_to_ret` / `unify_returns`), rendered from the typed `ret` carrier — the
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
    /// The SSA value operands this instruction reads (def/use edges; `%name`s only, no labels/literals).
    pub(super) uses: Vec<String>,
    /// The instruction's operands resolved to typed form, in source order, for the opcode shapes tir
    /// lowers (binary/compare/select/convert/load/store/phi/freeze/fneg + the vector/aggregate element
    /// ops extractelement/insertelement/shufflevector/extractvalue/insertvalue). Opcodes whose operand
    /// layout is not yet lowered (getelementptr/call) contribute a single `Unresolved` so the operand
    /// list is never empty for an instruction that has operands.
    pub(super) operands: Vec<TirOperand>,
    /// For an `icmp`/`fcmp`, the comparison PREDICATE token (`eq`/`ne`/`slt`/`oeq`/`olt`/...) — the
    /// structural literal that selects the SPIR-V compare op (`Op::IEqual`, `Op::FOrdLessThan`, ...).
    /// `None` for every other opcode. Carried so the emitter reads the predicate from the typed graph
    /// instead of re-lexing it from the line (the R3 STRUCTURAL retirement, predicate increment).
    pub(super) cmp_predicate: Option<String>,
    /// For a `load`/`store`, the explicit memory ALIGNMENT (`align N`), or `None` when none is written
    /// (and `None` for every other opcode). The structural literal the memory-op emitters re-lex from
    /// the trailing `, align N` comma field; carried so they read it from the typed graph instead (the
    /// R3 STRUCTURAL retirement, alignment increment). Computed by the same `parse_memory_alignment` the
    /// emitter uses, so it is byte-identical.
    pub(super) mem_align: Option<u64>,
    /// For a `getelementptr`, the SOURCE element type (`getelementptr <srcty>, ...`) — the structural
    /// TYPE that selects the access-chain walk, parsed once here via the same `parse_gep` the emitter
    /// used to re-lex it. `None` for every other opcode. Carried so the `getelementptr` emitter builds
    /// its `LlGep` entirely from the typed graph (this `source_ty` + the base/index operands already in
    /// `operands`) instead of re-parsing the line, retiring the emit-time `parse_gep` (the R3 STRUCTURAL
    /// endgame on the R4-critical pointer path). Byte-identical by construction: same `parse_gep`.
    pub(super) gep_source_ty: Option<LlType>,
    /// For a `getelementptr`, the FULL parsed `LlGep` (`source_ty` + base + indices) — the same
    /// `parse_gep` result `gep_source_ty` above is sliced from, retained whole so the
    /// `collect_forward_geps` carrier reads the parsed GEP from the typed graph instead of re-lexing
    /// `text`. `None` for every other opcode (and for a `getelementptr` whose operands do not parse).
    /// Byte-identical to the retired text-walk by construction: the exact same `parse_gep` on the exact
    /// same rhs. Distinct from `operands` (which carries `getelementptr` as a single `Unresolved`).
    pub(super) gep: Option<Box<LlGep>>,
    /// For a direct `call`/`[must|no]tail call`, the FULL parsed `LlCall` (`ret` + `callee` + typed
    /// `args`), parsed once here via the same `parse_call` the emitter uses. Retained so use-pointee
    /// inference (`atomic_call_pointees`) reads the callee name + typed args from the typed graph instead
    /// of re-lexing `text`. `None` for a non-call line or an indirect call (no `@callee`, which
    /// `parse_call` rejects) — matching the text reader's own early-out. Byte-identical by construction:
    /// the same `parse_call` on the same `<ret> @callee(args)` rhs.
    pub(super) call: Option<Box<LlCall>>,
    /// The instruction's OPCODE mnemonic (`add`/`load`/`getelementptr`/...), the first whitespace token
    /// of the rhs — computed once at build time so structured emission (`emit_body_inst`) can DISPATCH on
    /// it. Every opcode family routes by this field into its graph-driven emitter; an unmigrated opcode is
    /// a fail-visible `Err` (there is no text fallback). Effect-only lines (`store`, void `call`) carry
    /// their leading token here too (`store`/`call`/`tail`). Empty string for a blank/comment/label line.
    pub(super) opcode: String,
    /// For an `alloca`, the ALLOCATED type (`alloca <ty>[, <count>][, align N]` → `<ty>`), parsed once
    /// here via the same `split_top_level` + `parse_type` the emitter re-lexed from the line. `None` for
    /// every other opcode (and for an `alloca` whose type does not parse — the emitter then reaches the
    /// fail-visible unmigrated-opcode `Err`, and the retry cascade owns the raw `.lines` text walk). Carried so the `alloca`
    /// emitter reads its allocated type from the typed graph instead of `text` (the M-A5 text retirement,
    /// alloca increment). Unresolved (module `resolve_type` runs at emit time); byte-identical by parse.
    pub(super) alloca_ty: Option<LlType>,
    /// For a `phi`, its parsed (unresolved) result type + `(value, predecessor-label)` incoming pairs
    /// (`parse_phi` on the operand text after the opcode), or `None` for every other opcode (and for a
    /// `phi` whose operands do not parse — the emitter then reaches the fail-visible unmigrated-opcode
    /// `Err`, and the retry cascade owns the raw `.lines` text walk). The
    /// incoming VALUES are re-sourced from `operands` at emit (`phi_incoming_values`); this carrier exists
    /// for the phi's predecessor LABELS (control-flow edges, absent from `operands`) and its result type,
    /// so the `phi` emitter reads them from the typed graph instead of re-lexing `text`. Byte-identical by
    /// construction: the SAME `parse_phi` on the SAME post-opcode rest the emitter computes.
    pub(super) phi_incoming: Option<(LlType, Vec<(LlValue, String)>)>,
    /// Exact `parse_phi` refusal captured while building a phi whose [`Self::phi_incoming`] carrier
    /// is absent. Diagnostics-only: emission returns it through the existing fail-visible graph-walk
    /// error; it never changes parsing or lowering. `None` for valid phis and non-phi instructions.
    pub(super) phi_parse_error: Option<String>,
    /// For an `extractvalue`/`insertvalue`, the trailing constant INDEX literals — the fields after the
    /// aggregate (`extractvalue`) or aggregate+element (`insertvalue`) value operands. These are plain
    /// integer literals in the opcode text, not SSA value operands the graph lowers, so the emitter used
    /// to re-lex them from `text`; this carrier holds them, parsed once at build (rhs after `%r = `,
    /// opcode token dropped, then `split_top_level` + `parse_u32`) the resolved core ran. `None` for every other
    /// opcode (and for a malformed/unparsable index list — the emitter then reaches the fail-visible
    /// unmigrated-opcode `Err`, and the retry cascade owns the raw `.lines` text walk). Byte-identical by construction.
    pub(super) aggregate_indices: Option<Vec<u32>>,
    /// A DIAGNOSTICS-ONLY strip-commented/trimmed copy of the instruction line, populated at build for the
    /// element-op opcodes (`extractelement`/`insertelement`/`shufflevector`) whose resolved cores embed the
    /// raw `{line}` in SEMANTIC error strings that fire post-type-resolution (`extractelement from
    /// non-vector`, `one-lane … index is not zero`, `empty one-lane shuffle`). Those errors cannot move to
    /// a build-time parse (they need the resolved module type), and the BC gate fingerprints error text, so
    /// this carrier lets the typed dispatch feed the exact same `line` to the error formatting WITHOUT
    /// re-lexing `text` — byte-identical by construction. Read ONLY by error formatting (the T1 §"diagnostics-
    /// only raw-line field"); never re-parsed for operands/data. `None` for every other opcode.
    pub(super) diag_line: Option<String>,
    /// For a `shufflevector`, the parsed constant MASK — `(declared_lane_count, index_values)`. The mask
    /// is a `<N x i32>` constant vector in the opcode text, not an SSA value operand the graph lowers, so
    /// the emitter used to re-lex it (the mask half of the shufflevector result-type computation —
    /// a-operand element type × declared mask lane count). This carrier holds the mask-only
    /// parse (declared lane count from the mask type + the `parse_vector_i32_values` indices), computed at
    /// build via the SAME `parse_constant_vector`/`parse_vector_i32_values` the emitter ran. `None` for
    /// every other opcode, and for a `shufflevector` whose mask does not parse or whose operand list is not
    /// three-wide (the emitter then reaches the fail-visible unmigrated-opcode `Err`, and the retry cascade
    /// owns the raw `.lines` text walk — the a-operand vector check + `empty one-lane` error stay on the emit side, the
    /// former off the resolved type, the latter off `diag_line`). Byte-identical by construction.
    pub(super) shuffle_mask: Option<(u32, Vec<u32>)>,
    /// For a result-LESS `call`/`tail call` (a VOID call — a value call carries a result and is driven off
    /// [`Self::call`] directly), the strip-commented/trimmed instruction line. The void-call emitter needs
    /// it for two things the typed graph does not carry: the `is_ignored_call_line` gate (debug/lifetime
    /// markers dropped as no-ops — matched by callee name / all-`metadata` operands) and the `non-void call
    /// without result` diagnostic. Populated at build (`strip_comment(line).trim()`) so the graph walk
    /// reads it here instead of re-lexing `text`; the direct call's callee/return/args ride [`Self::call`]
    /// and its argument VALUES come straight from [`Self::operands`] (byte-identical to the
    /// `tir_call_queue` the text path pops — the queue holds the same operands). `None` for every other
    /// opcode and for a result-bearing (value) call.
    pub(super) void_call_line: Option<String>,
    /// For a result-bearing `call`/`tail call` whose direct-call parse failed, the exact parse diagnostic
    /// computed at lower time. This is diagnostics-only: the emitter reads it to return the same unsupported
    /// indirect-call error the text parser would have returned, instead of falling through to the generic
    /// unmigrated-opcode bucket. `None` for non-calls and successfully parsed direct value calls.
    pub(super) value_call_error: Option<String>,
    /// For a `bitcast`: the parsed source typed value + the destination-type TEXT (`resolve_bitcast` —
    /// the SAME `strip_comment` + rhs after `%r = ` with the opcode token dropped + `split_once(" to ")` +
    /// `parse_typed_value` the `bitcast` handler re-lexed). The bitcast emitter reads it off the graph instead of
    /// `text`; the destination stays TEXT because `convert_dst_type` is a `&mut self` emit-time method.
    /// `None` for every other opcode or a malformed line (the emitter then reaches the fail-visible
    /// unmigrated-opcode `Err`, and the retry cascade owns the raw `.lines` text walk).
    pub(super) bitcast: Option<Box<(TypedValue, String)>>,
    /// For an `icmp`: the operand TEXT after the mnemonic (`resolve_icmp_rest`), read ONLY by the
    /// POINTER-form icmp emitter to reproduce its two unsupported-form error diagnostics byte-identically
    /// (they embed the raw `rest`, which BC fingerprints). The compared values come from `operands`; this
    /// carrier is diagnostics-only, never re-parsed. `None` for every other opcode.
    pub(super) icmp_rest: Option<String>,
    /// For a pointer-typed result whose defining rhs is a `getelementptr` walkable to a concrete member,
    /// the resolved POINTEE type (`resolve_gep_pointee`) — the exact value the flat `build` inserts into
    /// its `pointer_pointees` accumulator. Carried on the inst so [`build_from_blocks`] can rebuild that
    /// map from the carriers (the sole substrate) instead of re-lexing the body text, making the carrier
    /// self-describing for pointees. `None` for every other opcode and for a GEP whose walk does not
    /// resolve (dynamic struct index / aggregate-walk gap). Emission never reads it (pointees are
    /// diagnostic-only); it exists so the carrier is a complete stand-in for the flat build.
    pub(super) pointer_pointee: Option<LlType>,
    /// Precomputed `parse_identity_ptr_bitcast` (`resolve_identity_ptr_bitcast`) — `(result, base)` for
    /// an identity pointer bitcast. Read by the parse-time pointer alias/pointee inferences off the
    /// carrier (F-track / T5) instead of re-lexing the body text. `None` for every other line.
    pub(super) identity_ptr_bitcast: Option<(String, String)>,
    /// Precomputed `parse_phi_incoming_values` (`resolve_phi_incoming_values`) — the incoming VALUES of a
    /// `phi`, matching the alias inferences' lighter parser (no phi-type parse, unlike `phi_incoming`).
    /// `None` for a non-phi or an unparseable incoming list.
    pub(super) phi_incoming_values: Option<Vec<LlValue>>,
    /// Precomputed select arms (`resolve_select_arms`) — the parsed true/false `TypedValue` arms of a
    /// 3-operand `select`. Read by the alias inferences (which apply their own Ptr/Local filters).
    /// `None` otherwise.
    pub(super) select_arms: Option<Box<(TypedValue, TypedValue)>>,
    /// Precomputed `parse_load` (`resolve_load_inst`) — the parsed load (`ptr` + `result_ty`) of a
    /// `load`. Read by the pointee/raw-buffer inferences. `None` otherwise.
    pub(super) load: Option<Box<LlLoad>>,
    /// Precomputed store operands (`resolve_store`) — the parsed `(object, ptr)` `TypedValue`s of a
    /// `store`. Read by the raw-buffer / local-pointer-table inferences. `None` otherwise.
    pub(super) store: Option<Box<(TypedValue, TypedValue)>>,
    /// Precomputed alias-call parse (`resolve_alias_call`) — the `strip_call_prefix` chain fed to
    /// `parse_call`, NARROWER than `call`/`resolve_call`. Read by the ir/ alias & call-edge scans.
    /// `None` for a non-call line or an indirect call.
    pub(super) alias_call: Option<Box<LlCall>>,
    /// Precomputed emitter call-scan parse (`resolve_emit_scan_call`) — the `is_ignored`/`@`-gated
    /// `strip_call_prefix` + `parse_call`, PRESERVING error propagation. Read by
    /// `infer_function_param_pointees`/`_nonnull` (which propagate with `?`). `None` = the line is
    /// skipped; `Some(Ok/Err)` = the propagatable `parse_call` result.
    pub(super) emit_scan_call: Option<Box<Result<LlCall, String>>>,
}

impl TirInst {
    /// A helper-parameter boundary introduced by the typed inliner.
    ///
    /// The emitter gives this value an opaque temporary id while lowering the cloned helper body,
    /// then substitutes the caller argument id after the whole function has emitted. This preserves
    /// the residual SPIR-V inliner's ordering without serializing a synthetic instruction.
    pub(in crate::native) fn inline_parameter(result: String, argument: TypedValue) -> Self {
        let uses = match &argument.value {
            LlValue::Local(name) => vec![name.clone()],
            _ => Vec::new(),
        };
        Self {
            result: Some(result),
            result_ty: Some(argument.ty.clone()),
            uses,
            operands: vec![operand_from_typed_value(&argument)],
            cmp_predicate: None,
            mem_align: None,
            gep_source_ty: None,
            gep: None,
            call: None,
            opcode: "metal2vulkan.inline_parameter".to_string(),
            alloca_ty: None,
            phi_incoming: None,
            phi_parse_error: None,
            aggregate_indices: None,
            diag_line: None,
            shuffle_mask: None,
            void_call_line: None,
            value_call_error: None,
            bitcast: None,
            icmp_rest: None,
            pointer_pointee: None,
            identity_ptr_bitcast: None,
            phi_incoming_values: None,
            select_arms: None,
            load: None,
            store: None,
            alias_call: None,
            emit_scan_call: None,
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
            if let Some((ty, incoming)) = &inst.phi_incoming {
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

/// A function parsed once into typed blocks, with every resolvable SSA result's type carried on the
/// value (`value_types`) rather than re-derived at each use.
#[derive(Clone, Debug)]
pub(super) struct TirFunction {
    pub(super) blocks: Vec<TirBlock>,
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
            std::mem::size_of::<TirInst>() <= 640,
            "TirInst grew to {} bytes",
            std::mem::size_of::<TirInst>()
        );
    }

    /// Build the raw body lines the test-only flat [`build`] consumes (`LlFunction` no longer carries a
    /// `Vec<String>` body — production lowers carriers directly).
    fn func(body: &[&str]) -> Vec<String> {
        body.iter().map(|s| s.to_string()).collect()
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
        assert_eq!(inst("%a").uses, vec!["%x", "%y"]);
        // phi keeps the incoming VALUES (%a, %z), not the predecessor labels (%entry, %loop).
        assert_eq!(inst("%p").uses, vec!["%a", "%z"]);
        let store = tir.blocks[0]
            .insts
            .iter()
            .find(|i| i.opcode == "store")
            .unwrap();
        assert_eq!(store.result, None);
        assert_eq!(store.uses, vec!["%a", "%dst"]);
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
    fn atomic_call_pointees_read_the_call_carrier() {
        // `atomic_call_pointees` sources the pointer + element type from the `TirInst.call` carrier
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
            inst.call.is_some(),
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
        assert!(other.call.is_some());
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
        let (parsed, error) = phi_incoming_of("%p = phi i32 [ %a, %entry ], [ 0, %loop ]");
        let (ty, incoming) = parsed.expect("phi parses");
        assert!(error.is_none());
        assert_eq!(ty, LlType::Int(32));
        assert_eq!(incoming.len(), 2);
        assert_eq!(incoming[0].1, "%entry");
        assert_eq!(incoming[1].1, "%loop");
        let (incoming, error) = phi_incoming_of("%a = add i32 %x, %y");
        assert!(incoming.is_none());
        assert!(error.is_none());
    }

    #[test]
    fn malformed_phi_carrier_retains_the_parser_error() {
        let (incoming, error) = phi_incoming_of("%p = phi i32 [ nope ]");
        assert!(incoming.is_none());
        assert_eq!(
            error.as_deref(),
            Some("native emitter: malformed phi incoming fields:  nope ")
        );
    }

    #[test]
    fn array_phi_incoming_does_not_confuse_the_type_bracket_for_an_operand() {
        let (parsed, error) = phi_incoming_of("%p = phi [14 x i8] [ %a, %entry ], [ %b, %loop ]");
        let (ty, incoming) = parsed.expect("array phi parses");
        assert!(error.is_none());
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
        let (ty, incoming) = phi.phi_incoming.as_ref().expect("phi carrier present");
        assert_eq!(*ty, LlType::Int(32));
        assert_eq!(incoming[1].1, "%loop");
        // A non-phi inst carries None.
        let add = tir.blocks[0]
            .insts
            .iter()
            .find(|i| i.result.as_deref() == Some("%i2"))
            .unwrap();
        assert!(add.phi_incoming.is_none());
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
