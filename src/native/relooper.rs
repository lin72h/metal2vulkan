//! W2 (frontier-walls plan): a general relooper for the cfg frontier class.
//!
//! The structurizer-by-construction (`native/cfg`) handles the reducible CFGs it can merge by
//! construction, but a residue (switch-in-loop, multi-exit loops with phi-carrying exits, irregular
//! selection merges — spirv-val: "Selection must be structured", "does not structurally dominate the
//! merge block", "Multiple case constructs have branches …") defeats it. This pass lowers ANY CFG to
//! structured SPIR-V mechanically, the classic relooper / "big switch" form: every NON-entry block
//! becomes a case of a single `OpSwitch` driven by an `i32` state variable, wrapped in one
//! `OpLoopMerge` loop. Control transfers become `OpStore` to the state variable; the loop+switch is
//! trivially structured, so the output always satisfies the structured-CFG rules.
//!
//! The original ENTRY block is kept as a pure pre-loop PROLOGUE (the synthetic first block): it runs
//! once before the dispatch loop, so its SSA values — crucially including access-chain POINTERS, which
//! cannot be spilled to memory — dominate every case and need no demotion. Its terminator sets the
//! INITIAL state and branches to the loop header.
//!
//! Because every non-entry block becomes a sibling switch case, SSA values defined in one such block
//! and used in another no longer dominate their uses. The pass therefore register-demotes those
//! cross-block values (and every `OpPhi` result) to a function-scope `OpVariable` (stored after its
//! definition / on the incoming edge, loaded before each use). A phi incoming value that is computed
//! in the phi's own header block but carried back on an edge from a different predecessor (the classic
//! loop-induction phi) is likewise demoted — the edge-store lands in that predecessor case, where the
//! value no longer dominates. Entry-block values, original `OpVariable`s, and function parameters
//! dominate the whole function unchanged and are not demoted.
//!
//! Pointers cannot be spilled to memory under Logical addressing. A demoted pointer is instead
//! REMATERIALIZED per use-case (re-emitted) when it is an ACCESS CHAIN, a pointer `OpSelect`, or a
//! pointer `OpPhi` whose POINTER operands are themselves dominating-or-rematerializable and whose
//! SCALAR operands (chain indices, select condition) dominate or are spillable scalars
//! (additionally demoted+loaded). A chain re-emits its base+indices; a select re-emits both arms +
//! condition; a phi spills a small i32 TAG on each incoming edge (which arm fired) and rebuilds
//! `select(tag==i, remat(arm_i), …)` at the use, re-emitting each arm (not-taken arms are computed
//! but discarded by the selects — pure address arithmetic, no dereference, so it is
//! semantics-preserving). Rematerialization is a fixpoint over `ptr_def` and is acyclic (phi arms are
//! never themselves phis). Any other demoted pointer (a function-call result, a non-rematerializable
//! base) still bails.
//!
//! Applied in `lib.rs`'s failure-triggered cfg retry (adopt-if-VALIDATES), so it is floor-safe by
//! construction: a module that validates on the default path never reaches it, and a relooped module
//! that does not independently `spirv-val` is discarded. The transform decides purely from IR
//! structure (block edges, phi membership, value def/use), never a shader name.
//!
//! Conservative bails (leave a function unchanged, so the retry simply yields no win for it): >512
//! blocks (pathological), a cross-block NON-rematerializable pointer or an opaque-typed value defined
//! in a NON-entry block (can't be spilled to memory), a 64-bit `OpSwitch` selector, an entry block
//! that is itself a branch target or has no branch successor, a phi on the entry block, or a
//! terminator outside the handled set.

use crate::spirv_module::Module;
use crate::spirv_module::Operand;
use crate::spirv_module::{Block, Function, Instruction};
use spirv::{Op, StorageClass, Word};
use std::collections::{HashMap, HashSet};

/// Rewrite every eligible multi-block function of `module` into relooper form. Returns true if any
/// function was rewritten.
pub(super) fn rewrite_to_relooper(module: &mut Module, max_blocks: usize) -> bool {
    // Snapshot the global type/const tables we read while choosing/creating types and constants.
    let mut next_id = module.header.as_ref().map(|h| h.bound).unwrap_or(1);
    let mut tc = TypeCtx::new(module, &mut next_id);

    let mut any = false;
    // Borrow functions separately from the type table being mutated: collect the new
    // types/constants tc accumulates and append them after the function rewrites.
    let mut functions = std::mem::take(&mut module.functions);
    if crate::env_vars::reloop_why() {
        eprintln!("RELOOP-ENTER functions={}", functions.len());
    }
    for function in &mut functions {
        if rewrite_function(function, &mut tc, max_blocks) {
            any = true;
        }
    }
    module.functions = functions;
    tc.flush(module);
    if let Some(h) = module.header.as_mut() {
        h.bound = next_id;
    }
    any
}

/// Find-or-create cache for the integer / bool / pointer types and the integer constants the relooper
/// synthesizes. Accumulates new global instructions in `pending`, flushed into the module at the end.
struct TypeCtx<'a> {
    next_id: &'a mut Word,
    /// Existing + pending type/const defs, keyed for lookup.
    int_types: HashMap<(u32, u32), Word>, // (width, signedness) -> id
    bool_type: Option<Word>,
    ptr_func: HashMap<Word, Word>, // pointee -> ptr Function pointee
    int_consts: HashMap<(Word, u64), Word>, // (int type, value) -> const id
    /// Snapshot of pointee for each pointer type id, and the opcode of each type id (to classify).
    type_op: HashMap<Word, Op>,
    pending: Vec<Instruction>,
}

impl<'a> TypeCtx<'a> {
    fn new(module: &Module, next_id: &'a mut Word) -> Self {
        let mut int_types = HashMap::new();
        let mut bool_type = None;
        let mut ptr_func = HashMap::new();
        let mut int_consts = HashMap::new();
        let mut type_op = HashMap::new();
        for inst in &module.types_global_values {
            let Some(rid) = inst.result_id else { continue };
            type_op.insert(rid, inst.class.opcode);
            match inst.class.opcode {
                Op::TypeInt => {
                    if let (Some(Operand::LiteralBit32(w)), Some(Operand::LiteralBit32(s))) =
                        (inst.operands.first(), inst.operands.get(1))
                    {
                        int_types.insert((*w, *s), rid);
                    }
                }
                Op::TypeBool => bool_type = Some(rid),
                Op::TypePointer => {
                    if let (
                        Some(Operand::StorageClass(StorageClass::Function)),
                        Some(Operand::IdRef(p)),
                    ) = (inst.operands.first(), inst.operands.get(1))
                    {
                        ptr_func.insert(*p, rid);
                    }
                }
                Op::Constant => {
                    if let (Some(ty), Some(Operand::LiteralBit32(v))) =
                        (inst.result_type, inst.operands.first())
                    {
                        int_consts.insert((ty, *v as u64), rid);
                    }
                }
                _ => {}
            }
        }
        TypeCtx {
            next_id,
            int_types,
            bool_type,
            ptr_func,
            int_consts,
            type_op,
            pending: Vec::new(),
        }
    }

    fn fresh(&mut self) -> Word {
        let id = *self.next_id;
        *self.next_id += 1;
        id
    }

    fn type_opcode(&self, ty: Word) -> Option<Op> {
        self.type_op.get(&ty).copied()
    }

    fn int_ty(&mut self, width: u32, signed: u32) -> Word {
        if let Some(&id) = self.int_types.get(&(width, signed)) {
            return id;
        }
        let id = self.fresh();
        self.pending.push(Instruction::new(
            Op::TypeInt,
            None,
            Some(id),
            vec![Operand::LiteralBit32(width), Operand::LiteralBit32(signed)],
        ));
        self.type_op.insert(id, Op::TypeInt);
        self.int_types.insert((width, signed), id);
        id
    }

    fn i32_ty(&mut self) -> Word {
        self.int_ty(32, 0)
    }

    fn bool_ty(&mut self) -> Word {
        if let Some(id) = self.bool_type {
            return id;
        }
        let id = self.fresh();
        self.pending
            .push(Instruction::new(Op::TypeBool, None, Some(id), vec![]));
        self.type_op.insert(id, Op::TypeBool);
        self.bool_type = Some(id);
        id
    }

    fn ptr_function(&mut self, pointee: Word) -> Word {
        if let Some(&id) = self.ptr_func.get(&pointee) {
            return id;
        }
        let id = self.fresh();
        self.pending.push(Instruction::new(
            Op::TypePointer,
            None,
            Some(id),
            vec![
                Operand::StorageClass(StorageClass::Function),
                Operand::IdRef(pointee),
            ],
        ));
        self.type_op.insert(id, Op::TypePointer);
        self.ptr_func.insert(pointee, id);
        id
    }

    fn int_const(&mut self, ty: Word, value: u64) -> Word {
        if let Some(&id) = self.int_consts.get(&(ty, value)) {
            return id;
        }
        let id = self.fresh();
        self.pending.push(Instruction::new(
            Op::Constant,
            Some(ty),
            Some(id),
            vec![Operand::LiteralBit32(value as u32)],
        ));
        self.int_consts.insert((ty, value), id);
        id
    }

    fn flush(self, module: &mut Module) {
        module.types_global_values.extend(self.pending);
    }
}

/// A handled block terminator, decoded from the original block's last instruction.
enum Term {
    Branch(Word),
    BranchCond(Word, Word, Word),         // cond, true, false
    Switch(Word, Word, Vec<(u32, Word)>), // selector, default, (literal, label)
    Return,
    ReturnValue(Word),
    Unreachable,
    Kill(Instruction), // OpKill / OpTerminateInvocation / OpDemoteToHelperInvocation-as-terminator
}

fn decode_term(inst: &Instruction) -> Option<Term> {
    match inst.class.opcode {
        Op::Branch => match inst.operands.first()? {
            Operand::IdRef(t) => Some(Term::Branch(*t)),
            _ => None,
        },
        Op::BranchConditional => {
            let (Operand::IdRef(c), Operand::IdRef(t), Operand::IdRef(f)) = (
                inst.operands.first()?,
                inst.operands.get(1)?,
                inst.operands.get(2)?,
            ) else {
                return None;
            };
            Some(Term::BranchCond(*c, *t, *f))
        }
        Op::Switch => {
            let Operand::IdRef(sel) = inst.operands.first()? else {
                return None;
            };
            let Operand::IdRef(def) = inst.operands.get(1)? else {
                return None;
            };
            let mut cases = Vec::new();
            let mut i = 2;
            while i + 1 < inst.operands.len() {
                let lit = match &inst.operands[i] {
                    Operand::LiteralBit32(v) => *v,
                    // 64-bit selector literal -> bail (None) so the whole function is left unchanged.
                    _ => return None,
                };
                let Operand::IdRef(lbl) = &inst.operands[i + 1] else {
                    return None;
                };
                cases.push((lit, *lbl));
                i += 2;
            }
            Some(Term::Switch(*sel, *def, cases))
        }
        Op::Return => Some(Term::Return),
        Op::ReturnValue => match inst.operands.first()? {
            Operand::IdRef(v) => Some(Term::ReturnValue(*v)),
            _ => None,
        },
        Op::Unreachable => Some(Term::Unreachable),
        Op::Kill | Op::TerminateInvocation => Some(Term::Kill(inst.clone())),
        _ => None,
    }
}

fn block_label(block: &Block) -> Option<Word> {
    block.label.as_ref().and_then(|l| l.result_id)
}

/// Above this block count the relooper bails: a giant function would produce an enormous switch +
/// register-demotion module that gives pathological spirv-val / downstream compile time for no payoff.
/// 1024 clears the mid-size emitted-then-rejected cfg family (542–953 blocks, ~1–2s each to reloop +
/// validate) while still bailing the >1024-block cost-budget cluster (1304–4630 blocks), which is
/// mostly emit-fail anyway (no module reaches the relooper) and would be pathological to lower.
const MAX_RELOOPER_BLOCKS: usize = 1024;

/// The effective default block cap. Defaults to `MAX_RELOOPER_BLOCKS` but can be raised via
/// `METAL2VULKAN_RELOOPER_MAX_BLOCKS` for experiments. The prune-then-relooper retry passes its own (higher)
/// cap explicitly. Behaviour is adopt-if-validates regardless, so a raised cap can only add wins,
/// never regress the floor.
pub(super) fn default_max_relooper_blocks() -> usize {
    crate::env_vars::relooper_max_blocks(MAX_RELOOPER_BLOCKS)
}

fn rewrite_function(function: &mut Function, tc: &mut TypeCtx, max_blocks: usize) -> bool {
    // Bail with a reason string kept at each call site as documentation of the residual cfg limits.
    let bail = |_r: &str| -> bool {
        if crate::env_vars::reloop_why() {
            eprintln!("RELOOP-BAIL {_r} (blocks={})", function.blocks.len());
        }
        false
    };

    if crate::env_vars::reloop_why() {
        eprintln!(
            "RELOOP-FN blocks={} max={}",
            function.blocks.len(),
            max_blocks
        );
    }
    if function.blocks.len() < 2 {
        return bail("too-few-blocks");
    }
    if function.blocks.len() > max_blocks {
        return bail("too-many-blocks");
    }

    // Decode every block's terminator; bail if any is unhandled.
    let mut terms: Vec<Term> = Vec::with_capacity(function.blocks.len());
    for block in &function.blocks {
        let Some(last) = block.instructions.last() else {
            return bail("empty-block");
        };
        match decode_term(last) {
            Some(t) => terms.push(t),
            None => return bail("unhandled-terminator"),
        }
    }

    let labels: Vec<Word> = match function
        .blocks
        .iter()
        .map(block_label)
        .collect::<Option<_>>()
    {
        Some(v) => v,
        None => return false,
    };
    let label_index: HashMap<Word, usize> =
        labels.iter().enumerate().map(|(i, l)| (*l, i)).collect();

    // The entry block (block 0) is kept as a pure pre-loop PROLOGUE: its instructions run once before
    // the dispatch loop, so its SSA values (including access-chain pointers) dominate every case and
    // need no register demotion. That requires the entry to have NO predecessor (never a back-edge
    // target) and to end in a branch (a successor to enter the loop). Bail otherwise. Also bail if it
    // carries a phi (LLVM entry never does; we could not source its incoming).
    let entry_term = &terms[0];
    let entry_has_successor = matches!(
        entry_term,
        Term::Branch(_) | Term::BranchCond(..) | Term::Switch(..)
    );
    let mut all_targets: HashSet<Word> = HashSet::new();
    for t in &terms {
        match t {
            Term::Branch(x) => {
                all_targets.insert(*x);
            }
            Term::BranchCond(_, a, b) => {
                all_targets.insert(*a);
                all_targets.insert(*b);
            }
            Term::Switch(_, d, cs) => {
                all_targets.insert(*d);
                for (_, l) in cs {
                    all_targets.insert(*l);
                }
            }
            _ => {}
        }
    }
    if !entry_has_successor {
        return bail("entry-no-successor");
    }
    if !all_targets
        .iter()
        .all(|target| label_index.contains_key(target))
    {
        return bail("missing-target");
    }
    if all_targets.contains(&labels[0]) {
        return bail("entry-is-branch-target");
    }
    if function.blocks[0]
        .instructions
        .iter()
        .any(|i| i.class.opcode == Op::Phi)
    {
        return bail("entry-has-phi");
    }

    // result id -> defining block index (for non-phi, non-variable instructions and phi results).
    let mut def_block: HashMap<Word, usize> = HashMap::new();
    // phi result -> (result type, Vec<(value, predecessor label)>)
    let mut phis: HashMap<Word, (Word, Vec<(Word, Word)>)> = HashMap::new();
    // value id -> result type, for every result-bearing instruction.
    let mut value_type: HashMap<Word, Word> = HashMap::new();
    // OpVariable instructions (Function storage) to hoist into the synthetic entry, with their ids.
    let mut variables: Vec<Instruction> = Vec::new();
    let mut variable_ids: HashSet<Word> = HashSet::new();
    // pointer-result id -> its defining instruction (for per-case rematerialization of demoted
    // pointers, which cannot spill to memory). Holds the two rematerializable shapes: access chains
    // and pointer `OpSelect`s (a select over two access chains arises from a `cond ? &a[i] : &b[j]`).
    let mut ptr_def: HashMap<Word, Instruction> = HashMap::new();

    for (bi, block) in function.blocks.iter().enumerate() {
        for inst in &block.instructions {
            if let Some(rid) = inst.result_id {
                if let Some(rty) = inst.result_type {
                    value_type.insert(rid, rty);
                }
                def_block.insert(rid, bi);
                if matches!(
                    inst.class.opcode,
                    Op::AccessChain
                        | Op::InBoundsAccessChain
                        | Op::PtrAccessChain
                        | Op::Select
                        | Op::CopyObject
                ) {
                    ptr_def.insert(rid, inst.clone());
                }
            }
            match inst.class.opcode {
                Op::Phi => {
                    let rid = inst.result_id.unwrap();
                    let rty = match inst.result_type {
                        Some(t) => t,
                        None => return false,
                    };
                    let mut incoming = Vec::new();
                    let mut k = 0;
                    while k + 1 < inst.operands.len() {
                        let (Operand::IdRef(v), Operand::IdRef(p)) =
                            (&inst.operands[k], &inst.operands[k + 1])
                        else {
                            return false;
                        };
                        incoming.push((*v, *p));
                        k += 2;
                    }
                    phis.insert(rid, (rty, incoming));
                }
                Op::Variable => {
                    variables.push(inst.clone());
                    variable_ids.insert(inst.result_id.unwrap_or(0));
                }
                _ => {}
            }
        }
    }

    // Per block index, the phi result ids that block hosts (so the terminator lowering can store the
    // right incoming on each edge without re-borrowing `function.blocks`).
    let block_phis: Vec<Vec<Word>> = function
        .blocks
        .iter()
        .map(|b| {
            b.instructions
                .iter()
                .filter(|i| i.class.opcode == Op::Phi)
                .filter_map(|i| i.result_id)
                .collect()
        })
        .collect();

    // Function parameters dominate every block and are never demoted.
    let param_ids: HashSet<Word> = function
        .parameters
        .iter()
        .filter_map(|p| p.result_id)
        .collect();

    // A value is "local" to a block if defined there. Determine which values are used in a block
    // other than the one that defines them (cross-block) — those plus all phi results are demoted.
    let mut demote: HashSet<Word> = phis.keys().copied().collect();
    for (bi, block) in function.blocks.iter().enumerate() {
        for inst in &block.instructions {
            for op in &inst.operands {
                if let Operand::IdRef(id) = op {
                    if param_ids.contains(id) || variable_ids.contains(id) {
                        continue;
                    }
                    if let Some(&db) = def_block.get(id) {
                        // Entry (block 0) is the prologue and dominates every case, so its values are
                        // never demoted even when used cross-block.
                        if db != bi && db != 0 {
                            demote.insert(*id);
                        }
                    }
                }
            }
        }
    }

    // A phi incoming value is referenced by the edge-store the relooper synthesizes in the
    // PREDECESSOR block (`store_phi_edges`), not in the phi's own block. The instruction-operand scan
    // above runs in the phi's block, so it MISSES a value that is defined in the phi's own block yet
    // consumed on an incoming edge from a DIFFERENT predecessor — the classic loop-header phi
    // `%p = OpPhi %next %back …` where `%next` is computed later in the same header block and carried
    // back through `%back`. In relooper form every non-entry block is a sibling switch case, so such a
    // value does not dominate the predecessor case where its edge-store lands; it must be demoted (and
    // loaded in that predecessor) like any other cross-case value. Demote any incoming value whose def
    // block is neither the predecessor case nor the entry/prologue (and is not a param/hoisted var).
    for (_, incoming) in phis.values() {
        for (val, pred) in incoming {
            if param_ids.contains(val) || variable_ids.contains(val) {
                continue;
            }
            let Some(&pred_bi) = label_index.get(pred) else {
                continue;
            };
            if let Some(&db) = def_block.get(val) {
                if db != pred_bi && db != 0 {
                    demote.insert(*val);
                }
            }
        }
    }

    // A value dominates every switch case iff it is available unconditionally in the relooper form:
    // a module global / constant (no def block), a function parameter, a hoisted Function OpVariable,
    // or an entry-block (prologue) value.
    let dominates_all_cases = |id: &Word| -> bool {
        !def_block.contains_key(id)
            || def_block.get(id) == Some(&0)
            || param_ids.contains(id)
            || variable_ids.contains(id)
    };

    // Pointers cannot spill to memory, but a pointer that is an ACCESS CHAIN or a pointer `OpSelect`
    // can instead be REMATERIALIZED in each case that uses it (re-emit it), provided its POINTER
    // operands are themselves dominating-or-rematerializable and its SCALAR operands (chain indices,
    // select condition) are either dominating (a constant) or a by-value scalar we additionally
    // demote+spill. A select's arm is often a chain used ONLY by the select (so not itself cross-block
    // demoted), yet must be re-emitted alongside — so compute the rematerializable set over ALL
    // pointer chains/selects (`ptr_def`), not only the demoted ones, as a fixpoint (an arm chain must
    // be admitted before the select that consumes it). This generalizes the per-case access-chain
    // rematerialization lever to `cond ? &a[i] : &b[j]` pointer selects.
    let mut remat: HashMap<Word, Instruction> = HashMap::new();
    let mut remat_scalar_demote: HashSet<Word> = HashSet::new();
    loop {
        let mut newly: Vec<(Word, Instruction, Vec<Word>)> = Vec::new();
        for (&v, inst) in &ptr_def {
            if remat.contains_key(&v) {
                continue;
            }
            let Some(&ty) = value_type.get(&v) else {
                continue;
            };
            if tc.type_opcode(ty) != Some(Op::TypePointer) {
                continue;
            }
            // All operands must be id refs (a literal-bearing op is not rematerializable here).
            let ids: Option<Vec<Word>> = inst
                .operands
                .iter()
                .map(|o| match o {
                    Operand::IdRef(i) => Some(*i),
                    _ => None,
                })
                .collect();
            let Some(ids) = ids else { continue };
            // Partition operands into pointer operands and scalar operands by opcode.
            let (ptr_ops, scalar_ops): (&[Word], &[Word]) = match inst.class.opcode {
                Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain
                    if ids.len() >= 2 =>
                {
                    (&ids[..1], &ids[1..]) // base pointer, then index scalars
                }
                Op::Select if ids.len() == 3 => (&ids[1..3], &ids[..1]), // two arm pointers, condition
                Op::CopyObject if ids.len() == 1 => (&ids[..1], &ids[..0]), // pure pointer alias
                _ => continue,
            };
            // Every pointer operand must dominate every case or already be rematerializable.
            if !ptr_ops
                .iter()
                .all(|p| dominates_all_cases(p) || remat.contains_key(p))
            {
                continue;
            }
            // Every non-dominating scalar operand must be a by-value scalar/vector/bool we can spill.
            let mut pending: Vec<Word> = Vec::new();
            let mut ok = true;
            for s in scalar_ops {
                if dominates_all_cases(s) {
                    continue;
                }
                match value_type.get(s).and_then(|t| tc.type_opcode(*t)) {
                    Some(Op::TypeInt | Op::TypeFloat | Op::TypeVector | Op::TypeBool) => {
                        pending.push(*s)
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            newly.push((v, inst.clone(), pending));
        }
        if newly.is_empty() {
            break;
        }
        for (v, inst, pending) in newly {
            remat.insert(v, inst);
            remat_scalar_demote.extend(pending);
        }
    }
    // Rematerialized pointers are NOT spilled; their dynamic scalar operands ARE.
    for v in remat.keys() {
        demote.remove(v);
    }
    demote.extend(remat_scalar_demote);

    // A POINTER PHI cannot spill to memory either, but if every incoming pointer is dominating or
    // rematerializable, the merge is expressible as a TAG-SELECT: spill a small i32 tag (which arm
    // fired) on each incoming edge instead of the pointer, and a use rebuilds the pointer by
    // re-emitting each arm and selecting on the tag. This is the phi analogue of the chain/select
    // rematerialization above; it covers `p = phi[cond?&a:&b, &c]`-style cross-block pointer merges
    // the demotion otherwise bails on. Arms must be dominating-or-`remat` (never another phi), so the
    // rematerialization is acyclic.
    // Fixpoint: a pointer phi is rematerializable if every incoming pointer is dominating, an already
    // rematerializable chain/select (`remat`), or ANOTHER rematerializable phi (`remat_phi`) — i.e. a
    // phi-of-phi (e.g. a degenerate `phi[%p,%p]` whose `%p` is itself `phi[&a[k],&a[k]]`, the
    // loop-invariant accumulator-pointer shape). Iterating to a fixpoint lets the inner phi qualify in
    // an earlier round and the outer in a later one. A self-referential (genuinely loop-carried)
    // pointer phi is REJECTED: its back-edge arm derives from the phi itself, so that arm is never
    // dominating/`remat`/`remat_phi` and the phi is added in no round — keeping the rematerialization
    // strictly acyclic (so the recursive `rematerialize` terminates). The `*v != pid` guard drops a
    // direct self-edge as well.
    let mut remat_phi: HashMap<Word, (Word, Vec<(Word, Word)>)> = HashMap::new();
    loop {
        let mut added = false;
        for (&pid, (pty, incoming)) in &phis {
            if remat_phi.contains_key(&pid)
                || !demote.contains(&pid)
                || tc.type_opcode(*pty) != Some(Op::TypePointer)
            {
                continue;
            }
            if incoming.iter().all(|(v, _)| {
                *v != pid
                    && (dominates_all_cases(v)
                        || remat.contains_key(v)
                        || remat_phi.contains_key(v))
            }) {
                remat_phi.insert(pid, (*pty, incoming.clone()));
                added = true;
            }
        }
        if !added {
            break;
        }
    }
    for pid in remat_phi.keys() {
        demote.remove(pid);
    }

    // A remat_phi whose incoming arms ALL denote the same address — either the same id, or
    // structurally identical rematerializable access chains (same opcode + operands, e.g. two
    // `AC %arr %k` from different predecessors, the loop-invariant accumulator-pointer shape) — is
    // INVARIANT: it is rematerialized as a SINGLE re-emitted arm with NO `OpSelect`. This is essential
    // for Function/Private pointers, where an `OpSelect`/phi OF pointers is ILLEGAL (variable pointers
    // cover only StorageBuffer/Workgroup); the no-select invariant form is the only sound
    // rematerialization there. (A non-invariant pointer phi still uses the tag-select, which validates
    // only for SB/Workgroup pointees — adopt-if-validates discards an illegal Function-pointer select.)
    let mut remat_phi_invariant: HashSet<Word> = HashSet::new();
    for (&pid, (_, incoming)) in &remat_phi {
        let key_of = |v: Word| -> (u32, Vec<Operand>) {
            match remat.get(&v) {
                Some(inst) => (inst.class.opcode as u32, inst.operands.clone()),
                None => (u32::MAX, vec![Operand::IdRef(v)]),
            }
        };
        let mut keys = incoming.iter().map(|(v, _)| key_of(*v));
        if let Some(first) = keys.next() {
            if keys.all(|k| k == first) {
                remat_phi_invariant.insert(pid);
            }
        }
    }

    // Every demoted value must be spillable: a scalar/vector/aggregate by value, never a pointer or an
    // opaque (image/sampler/accel-structure) type. Bail the whole function otherwise.
    for v in &demote {
        let Some(&ty) = value_type.get(v) else {
            return bail("non-spillable-demote");
        };
        match tc.type_opcode(ty) {
            Some(Op::TypePointer)
            | Some(Op::TypeImage)
            | Some(Op::TypeSampler)
            | Some(Op::TypeSampledImage)
            | Some(Op::TypeAccelerationStructureKHR)
            | Some(Op::TypeRuntimeArray)
            | None => {
                return bail("non-spillable-demote");
            }
            _ => {}
        }
    }

    // ---- Eligible. Synthesize the relooper form. ----
    let i32_ty = tc.i32_ty();
    let ptr_i32 = tc.ptr_function(i32_ty);
    let bool_ty = tc.bool_ty();

    // A spill variable per demoted value. Sorted so spill-var id assignment (and thus the emitted
    // module bytes) is deterministic run-to-run rather than following HashSet iteration order.
    let mut demote_order: Vec<Word> = demote.iter().copied().collect();
    demote_order.sort_unstable();
    let mut spill: HashMap<Word, (Word, Word)> = HashMap::new(); // value -> (var id, value type)
    let mut spill_vars: Vec<Instruction> = Vec::new();
    for &v in &demote_order {
        let ty = value_type[&v];
        let ptr_ty = tc.ptr_function(ty);
        let var = tc.fresh();
        spill_vars.push(Instruction::new(
            Op::Variable,
            Some(ptr_ty),
            Some(var),
            vec![Operand::StorageClass(StorageClass::Function)],
        ));
        spill.insert(v, (var, ty));
    }

    // A tag spill var (i32) per rematerialized pointer phi (sorted for deterministic ids).
    let mut remat_phi_order: Vec<Word> = remat_phi.keys().copied().collect();
    remat_phi_order.sort_unstable();
    let mut phi_tag: HashMap<Word, Word> = HashMap::new();
    for &pid in &remat_phi_order {
        let var = tc.fresh();
        spill_vars.push(Instruction::new(
            Op::Variable,
            Some(ptr_i32),
            Some(var),
            vec![Operand::StorageClass(StorageClass::Function)],
        ));
        phi_tag.insert(pid, var);
    }

    // The state variable.
    let state_var = tc.fresh();
    spill_vars.push(Instruction::new(
        Op::Variable,
        Some(ptr_i32),
        Some(state_var),
        vec![Operand::StorageClass(StorageClass::Function)],
    ));

    // Case id per original block = its index. State constants per case id.
    let case_const: Vec<Word> = (0..function.blocks.len())
        .map(|i| tc.int_const(i32_ty, i as u64))
        .collect();

    // New ids for the structural blocks.
    let new_entry = tc.fresh();
    let loop_header = tc.fresh();
    let dispatch = tc.fresh();
    let switch_default_break = tc.fresh();
    let sel_merge = tc.fresh();
    let loop_continue = tc.fresh();
    let loop_merge = tc.fresh();

    // Phi results are always loaded from their spill (the phi instruction is stripped), even by users
    // in the phi's own block — unlike a plain SSA value, which stays in scope within its def block.
    // A rematerialized pointer phi is NOT spilled (it has no value slot), so it is excluded here.
    let phi_ids: HashSet<Word> = phis
        .keys()
        .copied()
        .filter(|p| !remat_phi.contains_key(p))
        .collect();

    // Read-only demotion context threaded into the free helpers (avoids nested-closure aliasing).
    let demo = Demo {
        spill: &spill,
        demote: &demote,
        def_block: &def_block,
        phi: &phi_ids,
        remat: &remat,
        remat_phi: &remat_phi,
        phi_tag: &phi_tag,
        remat_phi_invariant: &remat_phi_invariant,
    };

    // Build the rewritten case blocks. Each NON-entry original block becomes a switch case (phis
    // stripped, demoted values loaded at the top / stored after definition, terminator lowered to a
    // state store + branch to the selection merge). Block 0 (entry) is instead lowered into the
    // synthetic prologue: its body runs once before the loop and its terminator sets the INITIAL state
    // and branches to the loop header.
    let mut case_blocks: Vec<Block> = Vec::with_capacity(function.blocks.len());
    let mut entry_processed: Vec<Instruction> = Vec::new();

    for (bi, block) in function.blocks.iter().enumerate() {
        let this_label = labels[bi];
        let exit_target = if bi == 0 { loop_header } else { sel_merge };
        let mut prelude: Vec<Instruction> = Vec::new();
        // Per-block load substitution for demoted values referenced here (incl. phi results read).
        let mut local_load: HashMap<Word, Word> = HashMap::new();
        // Per-block rematerialized access chains for demoted pointers referenced here.
        let mut local_remat: HashMap<Word, Word> = HashMap::new();

        let mut body: Vec<Instruction> = Vec::new();
        // Emit the block body (skip phis, hoisted variables, and the terminator); rewrite operands
        // referencing cross-block demoted values to a load, and spill results that are demoted.
        let n = block.instructions.len();
        for (ii, inst) in block.instructions.iter().enumerate() {
            // Skip phis (demoted), hoisted variables, the merge hints of the OLD structure (we build
            // a fresh one), and the terminator (lowered below).
            if matches!(
                inst.class.opcode,
                Op::Phi | Op::Variable | Op::LoopMerge | Op::SelectionMerge
            ) || ii == n - 1
            {
                continue;
            }
            let mut inst = inst.clone();
            for op in &mut inst.operands {
                if let Operand::IdRef(id) = op {
                    // A rematerialized pointer phi is never in scope (the phi is stripped), so it is
                    // rematerialized in every using block; a rematerialized chain/select only needs
                    // re-emitting where it is NOT already in scope (cross-block).
                    if remat_phi.contains_key(id)
                        || (remat.contains_key(id) && def_block.get(id) != Some(&bi))
                    {
                        let r = rematerialize(
                            tc,
                            &mut prelude,
                            &mut local_remat,
                            &mut local_load,
                            &demo,
                            bi,
                            *id,
                        );
                        *op = Operand::IdRef(r);
                    } else if demote.contains(id)
                        && (phi_ids.contains(id) || def_block.get(id) != Some(&bi))
                    {
                        let l = load_demoted(tc, &mut prelude, &mut local_load, &spill, *id);
                        *op = Operand::IdRef(l);
                    }
                }
            }
            // A demoted value defined earlier in THIS block stays SSA for in-block readers, but is
            // spilled right after definition so cross-block readers see it.
            let spill_after = inst.result_id.filter(|r| demote.contains(r));
            body.push(inst);
            if let Some(r) = spill_after {
                let (var, _) = spill[&r];
                body.push(Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(var), Operand::IdRef(r)],
                ));
            }
        }

        let mut tail: Vec<Instruction> = Vec::new();
        match &terms[bi] {
            Term::Branch(t) => {
                store_phi_edges(
                    tc,
                    &mut prelude,
                    &mut tail,
                    &mut local_load,
                    &demo,
                    bi,
                    this_label,
                    *t,
                    &label_index,
                    &block_phis,
                    &phis,
                );
                let next = case_const[label_index[t]];
                tail.push(store_state(state_var, next));
                tail.push(branch(exit_target));
            }
            Term::BranchCond(c, t, f) => {
                store_phi_edges(
                    tc,
                    &mut prelude,
                    &mut tail,
                    &mut local_load,
                    &demo,
                    bi,
                    this_label,
                    *t,
                    &label_index,
                    &block_phis,
                    &phis,
                );
                store_phi_edges(
                    tc,
                    &mut prelude,
                    &mut tail,
                    &mut local_load,
                    &demo,
                    bi,
                    this_label,
                    *f,
                    &label_index,
                    &block_phis,
                    &phis,
                );
                if t == f {
                    let next = case_const[label_index[t]];
                    tail.push(store_state(state_var, next));
                } else {
                    let cond = resolve(tc, &mut prelude, &mut local_load, &demo, bi, *c);
                    let tc_const = case_const[label_index[t]];
                    let fc_const = case_const[label_index[f]];
                    let sel = tc.fresh();
                    tail.push(Instruction::new(
                        Op::Select,
                        Some(i32_ty),
                        Some(sel),
                        vec![
                            Operand::IdRef(cond),
                            Operand::IdRef(tc_const),
                            Operand::IdRef(fc_const),
                        ],
                    ));
                    tail.push(store_state(state_var, sel));
                }
                tail.push(branch(exit_target));
            }
            Term::Switch(selv, def, cases) => {
                store_phi_edges(
                    tc,
                    &mut prelude,
                    &mut tail,
                    &mut local_load,
                    &demo,
                    bi,
                    this_label,
                    *def,
                    &label_index,
                    &block_phis,
                    &phis,
                );
                for (_, lbl) in cases {
                    store_phi_edges(
                        tc,
                        &mut prelude,
                        &mut tail,
                        &mut local_load,
                        &demo,
                        bi,
                        this_label,
                        *lbl,
                        &label_index,
                        &block_phis,
                        &phis,
                    );
                }
                let selector = resolve(tc, &mut prelude, &mut local_load, &demo, bi, *selv);
                let sel_ty = value_type.get(selv).copied().unwrap_or(i32_ty);
                let mut cur = case_const[label_index[def]];
                for (lit, lbl) in cases {
                    let lit_const = tc.int_const(sel_ty, *lit as u64);
                    let eq = tc.fresh();
                    tail.push(Instruction::new(
                        Op::IEqual,
                        Some(bool_ty),
                        Some(eq),
                        vec![Operand::IdRef(selector), Operand::IdRef(lit_const)],
                    ));
                    let picked = case_const[label_index[lbl]];
                    let nxt = tc.fresh();
                    tail.push(Instruction::new(
                        Op::Select,
                        Some(i32_ty),
                        Some(nxt),
                        vec![
                            Operand::IdRef(eq),
                            Operand::IdRef(picked),
                            Operand::IdRef(cur),
                        ],
                    ));
                    cur = nxt;
                }
                tail.push(store_state(state_var, cur));
                tail.push(branch(exit_target));
            }
            Term::Return => tail.push(Instruction::new(Op::Return, None, None, vec![])),
            Term::ReturnValue(v) => {
                let rv = resolve(tc, &mut prelude, &mut local_load, &demo, bi, *v);
                tail.push(Instruction::new(
                    Op::ReturnValue,
                    None,
                    None,
                    vec![Operand::IdRef(rv)],
                ));
            }
            Term::Unreachable => tail.push(Instruction::new(Op::Unreachable, None, None, vec![])),
            Term::Kill(inst) => tail.push(inst.clone()),
        }

        let mut instructions = Vec::new();
        instructions.extend(prelude);
        instructions.extend(body);
        instructions.extend(tail);
        if bi == 0 {
            // The entry's lowered body+terminator becomes the prologue (prepended with the hoisted
            // variables below); it ends in the initial state store + branch to the loop header.
            entry_processed = instructions;
        } else {
            case_blocks.push(Block {
                label: Some(Instruction::new(Op::Label, None, Some(this_label), vec![])),
                instructions,
            });
        }
    }

    // Synthetic entry (prologue): hoist all OpVariables (original + spills + state) FIRST, then the
    // entry block's lowered body, which ends in the initial state store + branch to the loop header.
    let mut entry_insts: Vec<Instruction> = Vec::new();
    entry_insts.extend(variables.iter().cloned());
    entry_insts.extend(spill_vars);
    entry_insts.extend(entry_processed);
    let entry_block = Block {
        label: Some(Instruction::new(Op::Label, None, Some(new_entry), vec![])),
        instructions: entry_insts,
    };

    // Loop header.
    let header_block = Block {
        label: Some(Instruction::new(Op::Label, None, Some(loop_header), vec![])),
        instructions: vec![
            Instruction::new(
                Op::LoopMerge,
                None,
                None,
                vec![
                    Operand::IdRef(loop_merge),
                    Operand::IdRef(loop_continue),
                    Operand::LoopControl(spirv::LoopControl::NONE),
                ],
            ),
            Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(dispatch)]),
        ],
    };

    // Dispatch: load state, selection-merge to continue, switch to the cases. The switch default goes
    // to a synthetic default case dominated by the switch; that case branches to the loop merge (a
    // break). It is never taken at runtime (state is always a valid case id) but keeps the loop merge
    // statically reachable. The default cannot target the loop merge directly because SPIR-V requires
    // a switch header to dominate every case/default construct.
    let state_load = tc.fresh();
    let mut switch_ops = vec![
        Operand::IdRef(state_load),
        Operand::IdRef(switch_default_break),
    ];
    // Cases are the NON-entry blocks (1..N); the entry is the prologue, never a switch target.
    for (i, lbl) in labels.iter().enumerate().skip(1) {
        switch_ops.push(Operand::LiteralBit32(i as u32));
        switch_ops.push(Operand::IdRef(*lbl));
    }
    let dispatch_block = Block {
        label: Some(Instruction::new(Op::Label, None, Some(dispatch), vec![])),
        instructions: vec![
            Instruction::new(
                Op::Load,
                Some(i32_ty),
                Some(state_load),
                vec![Operand::IdRef(state_var)],
            ),
            Instruction::new(
                Op::SelectionMerge,
                None,
                None,
                vec![
                    Operand::IdRef(sel_merge),
                    Operand::SelectionControl(spirv::SelectionControl::NONE),
                ],
            ),
            Instruction::new(Op::Switch, None, None, switch_ops),
        ],
    };
    let switch_default_block = Block {
        label: Some(Instruction::new(
            Op::Label,
            None,
            Some(switch_default_break),
            vec![],
        )),
        instructions: vec![Instruction::new(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(loop_merge)],
        )],
    };

    // Selection merge: where the switch cases converge, then on to the loop continue (back-edge).
    let sel_merge_block = Block {
        label: Some(Instruction::new(Op::Label, None, Some(sel_merge), vec![])),
        instructions: vec![Instruction::new(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(loop_continue)],
        )],
    };
    let continue_block = Block {
        label: Some(Instruction::new(
            Op::Label,
            None,
            Some(loop_continue),
            vec![],
        )),
        instructions: vec![Instruction::new(
            Op::Branch,
            None,
            None,
            vec![Operand::IdRef(loop_header)],
        )],
    };
    let merge_terminal = if function
        .def
        .as_ref()
        .and_then(|def| def.result_type)
        .is_some_and(|ty| tc.type_opcode(ty) == Some(Op::TypeVoid))
    {
        Instruction::new(Op::Return, None, None, vec![])
    } else {
        Instruction::new(Op::Unreachable, None, None, vec![])
    };
    let merge_block = Block {
        label: Some(Instruction::new(Op::Label, None, Some(loop_merge), vec![])),
        instructions: vec![merge_terminal],
    };

    // Assemble: synthetic entry, loop header, dispatch, default-break case, all cases, selection
    // merge, loop continue, loop merge.
    let mut new_blocks = Vec::with_capacity(case_blocks.len() + 7);
    new_blocks.push(entry_block);
    new_blocks.push(header_block);
    new_blocks.push(dispatch_block);
    new_blocks.push(switch_default_block);
    new_blocks.extend(case_blocks);
    new_blocks.push(sel_merge_block);
    new_blocks.push(continue_block);
    new_blocks.push(merge_block);
    function.blocks = new_blocks;
    true
}

/// Read-only demotion context threaded into the terminator-lowering helpers.
struct Demo<'a> {
    spill: &'a HashMap<Word, (Word, Word)>,
    demote: &'a HashSet<Word>,
    def_block: &'a HashMap<Word, usize>,
    phi: &'a HashSet<Word>,
    /// Demoted pointers that are rematerialized (re-emitted) per use-case instead of spilled.
    remat: &'a HashMap<Word, Instruction>,
    /// Pointer phis rematerialized as a tag-select: result type + incomings (value, predecessor).
    /// A small i32 TAG is spilled per incoming edge (which arm fired) instead of the pointer, and a
    /// use rebuilds `select(tag==i, remat(arm_i), …)` by re-emitting each arm.
    remat_phi: &'a HashMap<Word, (Word, Vec<(Word, Word)>)>,
    /// Tag spill var (i32, Function) per `remat_phi`.
    phi_tag: &'a HashMap<Word, Word>,
    /// `remat_phi` whose arms all denote the same address (loop-invariant) — rematerialized as a
    /// single re-emitted arm with no `OpSelect` (the only sound form for Function/Private pointers).
    remat_phi_invariant: &'a HashSet<Word>,
}

fn store_state(state_var: Word, value: Word) -> Instruction {
    Instruction::new(
        Op::Store,
        None,
        None,
        vec![Operand::IdRef(state_var), Operand::IdRef(value)],
    )
}

fn branch(target: Word) -> Instruction {
    Instruction::new(Op::Branch, None, None, vec![Operand::IdRef(target)])
}

/// Load a demoted value's spill slot once per block, caching the loaded SSA id in `local_load`.
fn load_demoted(
    tc: &mut TypeCtx,
    prelude: &mut Vec<Instruction>,
    local_load: &mut HashMap<Word, Word>,
    spill: &HashMap<Word, (Word, Word)>,
    v: Word,
) -> Word {
    if let Some(&l) = local_load.get(&v) {
        return l;
    }
    let (var, ty) = spill[&v];
    let l = tc.fresh();
    prelude.push(Instruction::new(
        Op::Load,
        Some(ty),
        Some(l),
        vec![Operand::IdRef(var)],
    ));
    local_load.insert(v, l);
    l
}

/// Resolve a value operand for a terminator: load it from its spill slot if it is demoted and defined
/// in another block; otherwise it is in scope here (a param, an in-block SSA value, or non-demoted).
fn resolve(
    tc: &mut TypeCtx,
    prelude: &mut Vec<Instruction>,
    local_load: &mut HashMap<Word, Word>,
    demo: &Demo,
    bi: usize,
    v: Word,
) -> Word {
    if demo.demote.contains(&v) && (demo.phi.contains(&v) || demo.def_block.get(&v) != Some(&bi)) {
        load_demoted(tc, prelude, local_load, demo.spill, v)
    } else {
        v
    }
}

/// Rematerialize a demoted ACCESS-CHAIN pointer in block `bi`: re-emit the identical chain (its base
/// dominates every case) with each dynamic index resolved to its spill load, returning the fresh
/// result id. Cached per block via `local_remat` so repeated uses share one chain.
fn rematerialize(
    tc: &mut TypeCtx,
    prelude: &mut Vec<Instruction>,
    local_remat: &mut HashMap<Word, Word>,
    local_load: &mut HashMap<Word, Word>,
    demo: &Demo,
    bi: usize,
    v: Word,
) -> Word {
    if let Some(&r) = local_remat.get(&v) {
        return r;
    }
    // A rematerialized pointer phi: load its tag and rebuild `select(tag==i, remat(arm_i), …)`. Each
    // arm is re-emitted (recursively); the not-taken arms are computed but discarded by the selects
    // (pure address arithmetic, no dereference), so this is semantics-preserving.
    if let Some((pty, incoming)) = demo.remat_phi.get(&v) {
        // Loop-invariant pointer phi: every arm denotes the same address, so re-emit just ONE arm —
        // no tag load, no `OpSelect` (which would be illegal for a Function/Private pointer).
        if demo.remat_phi_invariant.contains(&v) {
            let first = incoming.first().expect("phi has >=1 incoming").0;
            let r = if demo.remat.contains_key(&first) || demo.remat_phi.contains_key(&first) {
                rematerialize(tc, prelude, local_remat, local_load, demo, bi, first)
            } else {
                resolve(tc, prelude, local_load, demo, bi, first)
            };
            local_remat.insert(v, r);
            return r;
        }
        let i32_ty = tc.i32_ty();
        let bool_ty = tc.bool_ty();
        let tag_var = demo.phi_tag[&v];
        let tag = tc.fresh();
        prelude.push(Instruction::new(
            Op::Load,
            Some(i32_ty),
            Some(tag),
            vec![Operand::IdRef(tag_var)],
        ));
        let arm = |tc: &mut TypeCtx,
                   prelude: &mut Vec<Instruction>,
                   local_remat: &mut HashMap<Word, Word>,
                   local_load: &mut HashMap<Word, Word>,
                   id: Word|
         -> Word {
            if demo.remat.contains_key(&id) || demo.remat_phi.contains_key(&id) {
                rematerialize(tc, prelude, local_remat, local_load, demo, bi, id)
            } else {
                resolve(tc, prelude, local_load, demo, bi, id)
            }
        };
        // Fold from the last incoming backward: acc starts as the last arm, then each earlier arm i is
        // selected when tag == i.
        let last = incoming.last().expect("phi has at least one incoming").0;
        let mut acc = arm(tc, prelude, local_remat, local_load, last);
        for (i, (val, _)) in incoming.iter().enumerate().rev().skip(1) {
            let armv = arm(tc, prelude, local_remat, local_load, *val);
            let tagc = tc.int_const(i32_ty, i as u64);
            let cmp = tc.fresh();
            prelude.push(Instruction::new(
                Op::IEqual,
                Some(bool_ty),
                Some(cmp),
                vec![Operand::IdRef(tag), Operand::IdRef(tagc)],
            ));
            let sel = tc.fresh();
            prelude.push(Instruction::new(
                Op::Select,
                Some(*pty),
                Some(sel),
                vec![
                    Operand::IdRef(cmp),
                    Operand::IdRef(armv),
                    Operand::IdRef(acc),
                ],
            ));
            acc = sel;
        }
        local_remat.insert(v, acc);
        return acc;
    }
    let inst = demo.remat[&v].clone();
    let mut new_ops = Vec::with_capacity(inst.operands.len());
    for op in inst.operands.iter() {
        match op {
            Operand::IdRef(id) => {
                // A pointer operand that is itself rematerializable is re-emitted recursively (a
                // select's arm chain, or a chain whose base is another rematerializable chain). Any
                // other id is a scalar (chain index / select condition) or a dominating base pointer:
                // resolve it (spill-load if it is a demoted cross-block value, else use it in scope).
                let r = if demo.remat.contains_key(id) {
                    rematerialize(tc, prelude, local_remat, local_load, demo, bi, *id)
                } else {
                    resolve(tc, prelude, local_load, demo, bi, *id)
                };
                new_ops.push(Operand::IdRef(r));
            }
            _ => new_ops.push(op.clone()),
        }
    }
    let result = tc.fresh();
    prelude.push(Instruction::new(
        inst.class.opcode,
        inst.result_type,
        Some(result),
        new_ops,
    ));
    local_remat.insert(v, result);
    result
}

/// For an edge from block `bi` (label `this_label`) to `target`, store each of `target`'s phi
/// incomings that come from this edge into the phi's spill slot (so the loaded phi reads the right
/// value when `target` runs).
#[allow(clippy::too_many_arguments)]
fn store_phi_edges(
    tc: &mut TypeCtx,
    prelude: &mut Vec<Instruction>,
    tail: &mut Vec<Instruction>,
    local_load: &mut HashMap<Word, Word>,
    demo: &Demo,
    bi: usize,
    this_label: Word,
    target: Word,
    label_index: &HashMap<Word, usize>,
    block_phis: &[Vec<Word>],
    phis: &HashMap<Word, (Word, Vec<(Word, Word)>)>,
) {
    let Some(&ti) = label_index.get(&target) else {
        return;
    };
    for &rid in &block_phis[ti] {
        let (_, incoming) = &phis[&rid];
        // A rematerialized pointer phi spills a small i32 TAG (which incoming fired on this edge), not
        // the pointer value — the use site rebuilds the pointer by re-emitting the tagged arm.
        if let Some(&tag_var) = demo.phi_tag.get(&rid) {
            for (idx, (_, pred)) in incoming.iter().enumerate() {
                if *pred == this_label {
                    let i32_ty = tc.i32_ty();
                    let tagc = tc.int_const(i32_ty, idx as u64);
                    tail.push(Instruction::new(
                        Op::Store,
                        None,
                        None,
                        vec![Operand::IdRef(tag_var), Operand::IdRef(tagc)],
                    ));
                }
            }
            continue;
        }
        for (val, pred) in incoming {
            if *pred == this_label {
                let (var, _) = demo.spill[&rid];
                let rv = resolve(tc, prelude, local_load, demo, bi, *val);
                tail.push(Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(var), Operand::IdRef(rv)],
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A temp dir unique to this call (pid + a process-wide counter) so parallel tests never share a
    /// scratch file.
    fn scratch() -> std::path::PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "metal2vulkan_relooper_{}_{}",
            std::process::id(),
            n
        ));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    /// Assemble spvasm to a SPIR-V byte module via the local `spirv-as` (Vulkan 1.3). Returns None if
    /// the tool is unavailable, so the test no-ops in toolchain-less environments.
    fn assemble(spvasm: &str) -> Option<Vec<u8>> {
        if std::process::Command::new("spirv-as")
            .arg("--version")
            .output()
            .is_err()
        {
            return None;
        }
        let dir = scratch();
        let src = dir.join("in.spvasm");
        let out = dir.join("in.spv");
        std::fs::write(&src, spvasm).unwrap();
        let st = std::process::Command::new("spirv-as")
            .args(["--target-env", "vulkan1.3"])
            .arg(&src)
            .arg("-o")
            .arg(&out)
            .output()
            .unwrap();
        assert!(
            st.status.success(),
            "spirv-as: {}",
            String::from_utf8_lossy(&st.stderr)
        );
        Some(std::fs::read(&out).unwrap())
    }

    fn validates(spv: &[u8]) -> bool {
        let dir = scratch();
        let p = dir.join("m.spv");
        std::fs::write(&p, spv).unwrap();
        let st = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&p)
            .output()
            .unwrap();
        if !st.status.success() {
            eprintln!("spirv-val: {}", String::from_utf8_lossy(&st.stderr));
        }
        st.status.success()
    }

    fn relooper_bytes(spv: &[u8]) -> Vec<u8> {
        let mut module = crate::spirv_module::load_bytes(spv).expect("load");
        assert!(
            rewrite_to_relooper(&mut module, MAX_RELOOPER_BLOCKS),
            "expected a rewrite"
        );
        module
            .assemble()
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect()
    }

    fn block_id(block: &Block) -> Word {
        block
            .label
            .as_ref()
            .and_then(|label| label.result_id)
            .expect("block label id")
    }

    fn id_ref(operand: &Operand) -> Word {
        let Operand::IdRef(id) = operand else {
            panic!("expected IdRef operand, got {operand:?}");
        };
        *id
    }

    fn single_loop_dispatch_default(spv: &[u8]) -> (Word, Word, Word, Op) {
        let module = crate::spirv_module::load_bytes(spv).expect("load relooped module");
        let function = module
            .functions
            .iter()
            .find(|function| {
                function.blocks.iter().any(|block| {
                    block
                        .instructions
                        .iter()
                        .any(|instruction| instruction.class.opcode == Op::LoopMerge)
                })
            })
            .expect("function with a relooped dispatcher");
        let loop_header = function
            .blocks
            .iter()
            .find(|block| {
                block
                    .instructions
                    .iter()
                    .any(|instruction| instruction.class.opcode == Op::LoopMerge)
            })
            .expect("loop header");
        let loop_merge = loop_header
            .instructions
            .iter()
            .find(|instruction| instruction.class.opcode == Op::LoopMerge)
            .and_then(|instruction| instruction.operands.first())
            .map(id_ref)
            .expect("loop merge target");
        let dispatch = loop_header
            .instructions
            .iter()
            .find(|instruction| instruction.class.opcode == Op::Branch)
            .and_then(|instruction| instruction.operands.first())
            .map(id_ref)
            .expect("dispatch branch");
        let dispatch_block = function
            .blocks
            .iter()
            .find(|block| block_id(block) == dispatch)
            .expect("dispatch block");
        let switch_default = dispatch_block
            .instructions
            .iter()
            .find(|instruction| instruction.class.opcode == Op::Switch)
            .and_then(|instruction| instruction.operands.get(1))
            .map(id_ref)
            .expect("switch default target");
        let default_block = function
            .blocks
            .iter()
            .find(|block| block_id(block) == switch_default)
            .expect("switch default block");
        let default_branch = default_block
            .instructions
            .iter()
            .find(|instruction| instruction.class.opcode == Op::Branch)
            .and_then(|instruction| instruction.operands.first())
            .map(id_ref)
            .expect("default block branch");
        let merge_terminal = function
            .blocks
            .iter()
            .find(|block| block_id(block) == loop_merge)
            .and_then(|block| block.instructions.last())
            .map(|instruction| instruction.class.opcode)
            .expect("loop merge terminal");
        (loop_merge, switch_default, default_branch, merge_terminal)
    }

    #[test]
    fn relooper_bails_instead_of_panicking_on_missing_target() {
        let mut module = Module::new();
        let mut function = Function::new();
        function.blocks = vec![
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(1), vec![])),
                instructions: vec![Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(2)],
                )],
            },
            Block {
                label: Some(Instruction::new(Op::Label, None, Some(2), vec![])),
                instructions: vec![Instruction::new(
                    Op::Branch,
                    None,
                    None,
                    vec![Operand::IdRef(99)],
                )],
            },
        ];
        module.functions.push(function);

        assert!(
            !rewrite_to_relooper(&mut module, MAX_RELOOPER_BLOCKS),
            "missing branch target should decline, not panic"
        );
    }

    #[test]
    fn relooper_dispatch_default_breaks_to_loop_merge() {
        let spvasm = r#"
                       OpCapability Shader
                       OpMemoryModel Logical GLSL450
                       OpEntryPoint GLCompute %main "main"
                       OpExecutionMode %main LocalSize 1 1 1
               %void = OpTypeVoid
                 %fn = OpTypeFunction %void
               %bool = OpTypeBool
               %true = OpConstantTrue %bool
               %main = OpFunction %void None %fn
              %entry = OpLabel
                       OpBranch %head
               %head = OpLabel
                       OpLoopMerge %merge %cont None
                       OpBranchConditional %true %body %merge
               %body = OpLabel
                       OpBranch %cont
               %cont = OpLabel
                       OpBranch %head
              %merge = OpLabel
                       OpReturn
                       OpFunctionEnd
        "#;
        let Some(spv) = assemble(spvasm) else { return };
        assert!(validates(&spv), "input must validate");
        let out = relooper_bytes(&spv);
        assert!(validates(&out), "relooper output must validate");
        let (loop_merge, switch_default, default_branch, merge_terminal) =
            single_loop_dispatch_default(&out);
        assert_ne!(
            switch_default, loop_merge,
            "SPIR-V requires a dominated switch default block, not a direct merge target"
        );
        assert_eq!(
            default_branch, loop_merge,
            "relooper switch default block must statically break to the loop merge"
        );
        assert_eq!(
            merge_terminal,
            Op::Return,
            "void-function dispatch merge represents normal function exit"
        );
    }

    // A counting loop with a phi-carried induction variable and a cross-block value. Already valid;
    // the relooper must keep it valid after rebuilding the CFG as a switch-dispatch loop.
    #[test]
    fn relooper_loop_with_phi_validates() {
        let spvasm = r#"
                       OpCapability Shader
                       OpMemoryModel Logical GLSL450
                       OpEntryPoint GLCompute %main "main"
                       OpExecutionMode %main LocalSize 1 1 1
               %void = OpTypeVoid
                 %fn = OpTypeFunction %void
               %uint = OpTypeInt 32 0
               %bool = OpTypeBool
                 %u0 = OpConstant %uint 0
                 %u1 = OpConstant %uint 1
                %u10 = OpConstant %uint 10
              %ptr_u = OpTypePointer Function %uint
               %main = OpFunction %void None %fn
              %entry = OpLabel
                %acc = OpVariable %ptr_u Function
                       OpStore %acc %u0
                       OpBranch %head
               %head = OpLabel
                  %i = OpPhi %uint %u0 %entry %inext %body
                %cmp = OpULessThan %bool %i %u10
                       OpLoopMerge %merge %body None
                       OpBranchConditional %cmp %body %merge
               %body = OpLabel
              %inext = OpIAdd %uint %i %u1
                       OpStore %acc %inext
                       OpBranch %head
              %merge = OpLabel
                       OpReturn
                       OpFunctionEnd
        "#;
        let Some(spv) = assemble(spvasm) else { return };
        assert!(validates(&spv), "input must validate");
        let out = relooper_bytes(&spv);
        assert!(validates(&out), "relooper output must validate");
    }

    // A switch nested in a loop (the cfg class the structurizer-by-construction struggles with) with a
    // value defined before the switch and used in multiple cases.
    #[test]
    fn relooper_switch_in_loop_validates() {
        let spvasm = r#"
                       OpCapability Shader
                       OpMemoryModel Logical GLSL450
                       OpEntryPoint GLCompute %main "main"
                       OpExecutionMode %main LocalSize 1 1 1
               %void = OpTypeVoid
                 %fn = OpTypeFunction %void
               %uint = OpTypeInt 32 0
               %bool = OpTypeBool
                 %u0 = OpConstant %uint 0
                 %u1 = OpConstant %uint 1
                 %u2 = OpConstant %uint 2
                 %u5 = OpConstant %uint 5
              %ptr_u = OpTypePointer Function %uint
               %main = OpFunction %void None %fn
              %entry = OpLabel
                %acc = OpVariable %ptr_u Function
                       OpStore %acc %u0
                       OpBranch %head
               %head = OpLabel
                  %i = OpPhi %uint %u0 %entry %inext %cont
                %cmp = OpULessThan %bool %i %u5
                       OpLoopMerge %merge %cont None
                       OpBranchConditional %cmp %sw %merge
                 %sw = OpLabel
                %dbl = OpIAdd %uint %i %i
                       OpSelectionMerge %swm None
                       OpSwitch %i %def 0 %c0 1 %c1
                 %c0 = OpLabel
                       OpStore %acc %dbl
                       OpBranch %swm
                 %c1 = OpLabel
                       OpStore %acc %i
                       OpBranch %swm
                %def = OpLabel
                       OpStore %acc %u2
                       OpBranch %swm
                %swm = OpLabel
                       OpBranch %cont
               %cont = OpLabel
              %inext = OpIAdd %uint %i %u1
                       OpBranch %head
              %merge = OpLabel
                       OpReturn
                       OpFunctionEnd
        "#;
        let Some(spv) = assemble(spvasm) else { return };
        assert!(validates(&spv), "input must validate");
        let out = relooper_bytes(&spv);
        assert!(validates(&out), "relooper output must validate");
    }

    // An access-chain POINTER defined in the ENTRY block and used inside the loop body. The entry is
    // kept as a dominating prologue, so this entry-defined pointer is NOT register-demoted (which would
    // bail, as pointers can't spill to memory) — the relooper must still rewrite and validate.
    #[test]
    fn relooper_entry_pointer_used_in_loop_validates() {
        let spvasm = r#"
                       OpCapability Shader
                       OpMemoryModel Logical GLSL450
                       OpEntryPoint GLCompute %main "main"
                       OpExecutionMode %main LocalSize 1 1 1
               %void = OpTypeVoid
                 %fn = OpTypeFunction %void
               %uint = OpTypeInt 32 0
               %bool = OpTypeBool
                 %u0 = OpConstant %uint 0
                 %u1 = OpConstant %uint 1
                %u10 = OpConstant %uint 10
              %ptr_u = OpTypePointer Function %uint
               %main = OpFunction %void None %fn
              %entry = OpLabel
                %acc = OpVariable %ptr_u Function
                  %p = OpAccessChain %ptr_u %acc
                       OpBranch %head
               %head = OpLabel
                  %i = OpPhi %uint %u0 %entry %inext %body
                %cmp = OpULessThan %bool %i %u10
                       OpLoopMerge %merge %body None
                       OpBranchConditional %cmp %body %merge
               %body = OpLabel
                       OpStore %p %i
              %inext = OpIAdd %uint %i %u1
                       OpBranch %head
              %merge = OpLabel
                       OpReturn
                       OpFunctionEnd
        "#;
        let Some(spv) = assemble(spvasm) else { return };
        assert!(validates(&spv), "input must validate");
        let out = relooper_bytes(&spv);
        assert!(validates(&out), "relooper output must validate");
    }

    #[test]
    fn relooper_rematerializes_nonentry_pointer_access_chain() {
        // A pointer access chain DEFINED in a non-entry block (%body) and USED in a different block
        // (%use). After relooping both become sibling switch cases, so %p does not dominate its use;
        // pointers cannot spill, so the relooper must REMATERIALIZE the chain in %use (its base %arr is
        // a hoisted Function variable, its index a constant). Before the rematerialization lever this
        // bailed ("unspillable demoted value"); now it must validate.
        let spvasm = r#"
                       OpCapability Shader
                       OpMemoryModel Logical GLSL450
                       OpEntryPoint GLCompute %main "main"
                       OpExecutionMode %main LocalSize 1 1 1
               %void = OpTypeVoid
                 %fn = OpTypeFunction %void
               %uint = OpTypeInt 32 0
               %bool = OpTypeBool
                 %u0 = OpConstant %uint 0
                 %u1 = OpConstant %uint 1
                %u10 = OpConstant %uint 10
                %u4 = OpConstant %uint 4
              %ptr_u = OpTypePointer Function %uint
                %arr_t = OpTypeArray %uint %u4
            %ptr_arr = OpTypePointer Function %arr_t
               %main = OpFunction %void None %fn
              %entry = OpLabel
                %arr = OpVariable %ptr_arr Function
                       OpBranch %head
               %head = OpLabel
                  %i = OpPhi %uint %u0 %entry %inext %use
                %cmp = OpULessThan %bool %i %u10
                       OpLoopMerge %merge %use None
                       OpBranchConditional %cmp %body %merge
               %body = OpLabel
                  %p = OpAccessChain %ptr_u %arr %u0
                       OpBranch %use
                %use = OpLabel
                       OpStore %p %i
              %inext = OpIAdd %uint %i %u1
                       OpBranch %head
              %merge = OpLabel
                       OpReturn
                       OpFunctionEnd
        "#;
        let Some(spv) = assemble(spvasm) else { return };
        assert!(validates(&spv), "input must validate");
        let out = relooper_bytes(&spv);
        assert!(
            validates(&out),
            "relooper output must validate (rematerialized pointer)"
        );
    }

    #[test]
    fn relooper_rematerializes_pointer_select_and_phi() {
        // A pointer OpSelect (%psel) and a pointer OpPhi (%p, one arm the select) over distinct
        // StorageBuffer element pointers, defined in non-entry blocks and used cross-block. Neither can
        // spill to memory; the relooper must rematerialize the select (re-emit it) and the phi (as a
        // tag-select that re-emits each arm — including the nested select) and still validate. Before
        // this lever the relooper bailed "unspillable demoted value … TypePointer (Select/Phi)".
        let spvasm = r#"
                       OpCapability Shader
                       OpCapability VariablePointersStorageBuffer
                       OpExtension "SPV_KHR_variable_pointers"
                       OpMemoryModel Logical GLSL450
                       OpEntryPoint GLCompute %main "main" %buf
                       OpExecutionMode %main LocalSize 1 1 1
                       OpDecorate %arr_t ArrayStride 4
                       OpMemberDecorate %buf_t 0 Offset 0
                       OpDecorate %buf_t Block
                       OpDecorate %buf DescriptorSet 0
                       OpDecorate %buf Binding 0
               %void = OpTypeVoid
                 %fn = OpTypeFunction %void
               %uint = OpTypeInt 32 0
               %bool = OpTypeBool
                 %u0 = OpConstant %uint 0
                 %u1 = OpConstant %uint 1
                 %u2 = OpConstant %uint 2
                 %u4 = OpConstant %uint 4
                %u10 = OpConstant %uint 10
              %arr_t = OpTypeArray %uint %u4
              %buf_t = OpTypeStruct %arr_t
            %ptr_buf = OpTypePointer StorageBuffer %buf_t
                %buf = OpVariable %ptr_buf StorageBuffer
              %ptr_u = OpTypePointer StorageBuffer %uint
               %main = OpFunction %void None %fn
              %entry = OpLabel
                       OpBranch %head
               %head = OpLabel
                  %i = OpPhi %uint %u0 %entry %inext %cont
                %cmp = OpULessThan %bool %i %u10
                       OpLoopMerge %merge %cont None
                       OpBranchConditional %cmp %body %merge
               %body = OpLabel
                  %c = OpULessThan %bool %i %u2
                 %pa = OpAccessChain %ptr_u %buf %u0 %u0
                 %pb = OpAccessChain %ptr_u %buf %u0 %u1
               %psel = OpSelect %ptr_u %c %pa %pb
                       OpSelectionMerge %join None
                       OpBranchConditional %c %t %f
                 %t = OpLabel
                 %pt = OpAccessChain %ptr_u %buf %u0 %u2
                       OpBranch %join
                 %f = OpLabel
                       OpBranch %join
               %join = OpLabel
                  %p = OpPhi %ptr_u %pt %t %psel %f
                       OpStore %p %i
                       OpBranch %cont
               %cont = OpLabel
              %inext = OpIAdd %uint %i %u1
                       OpBranch %head
              %merge = OpLabel
                       OpReturn
                       OpFunctionEnd
        "#;
        let Some(spv) = assemble(spvasm) else { return };
        assert!(validates(&spv), "input must validate");
        let out = relooper_bytes(&spv);
        assert!(
            validates(&out),
            "relooper output must validate (rematerialized pointer select + phi)"
        );
    }

    #[test]
    fn relooper_rematerializes_pointer_copy_object() {
        // A pointer OpCopyObject (%pc, a pure alias of an access chain) DEFINED in a non-entry block
        // (%body) and USED cross-block (%use). After relooping both become sibling switch cases, so
        // %pc does not dominate its use; a pointer cannot spill to memory, so the relooper must
        // REMATERIALIZE the CopyObject in %use by re-emitting it over its (dominating-or-rematerializable)
        // source. A CopyObject is a pure alias, so re-emission is byte-neutral. Before the CopyObject
        // rematerialization lever this bailed "unspillable demoted value … TypePointer (CopyObject)".
        let spvasm = r#"
                       OpCapability Shader
                       OpMemoryModel Logical GLSL450
                       OpEntryPoint GLCompute %main "main"
                       OpExecutionMode %main LocalSize 1 1 1
               %void = OpTypeVoid
                 %fn = OpTypeFunction %void
               %uint = OpTypeInt 32 0
               %bool = OpTypeBool
                 %u0 = OpConstant %uint 0
                 %u1 = OpConstant %uint 1
                %u10 = OpConstant %uint 10
                 %u4 = OpConstant %uint 4
              %ptr_u = OpTypePointer Function %uint
              %arr_t = OpTypeArray %uint %u4
            %ptr_arr = OpTypePointer Function %arr_t
               %main = OpFunction %void None %fn
              %entry = OpLabel
                %arr = OpVariable %ptr_arr Function
                       OpBranch %head
               %head = OpLabel
                  %i = OpPhi %uint %u0 %entry %inext %use
                %cmp = OpULessThan %bool %i %u10
                       OpLoopMerge %merge %use None
                       OpBranchConditional %cmp %body %merge
               %body = OpLabel
                  %p = OpAccessChain %ptr_u %arr %u0
                 %pc = OpCopyObject %ptr_u %p
                       OpBranch %use
                %use = OpLabel
                       OpStore %pc %i
              %inext = OpIAdd %uint %i %u1
                       OpBranch %head
              %merge = OpLabel
                       OpReturn
                       OpFunctionEnd
        "#;
        let Some(spv) = assemble(spvasm) else { return };
        assert!(validates(&spv), "input must validate");
        let out = relooper_bytes(&spv);
        assert!(
            validates(&out),
            "relooper output must validate (rematerialized pointer CopyObject)"
        );
    }

    #[test]
    fn relooper_rematerializes_invariant_function_pointer_phi() {
        // A FUNCTION-storage pointer phi whose two arms are STRUCTURALLY IDENTICAL access chains
        // (`%pt`/%pf both `AC %arr %u2`) inside a loop. metal2vulkan emits this loop-invariant
        // accumulator-pointer shape (the demoted `[K x float]` element pointer carried through a phi),
        // which is INVALID input (a Function pointer phi has no variable-pointers form — that capability
        // covers only StorageBuffer/Workgroup), so the input is not asserted to validate. The relooper
        // must rematerialize the phi as a SINGLE re-emitted arm with NO `OpSelect` (the select form
        // would be an illegal Function-pointer select); the no-select invariant form validates. Before
        // the fixpoint+invariant lever the relooper bailed "unspillable demoted value … TypePointer
        // (Phi)" or emitted an illegal `OpSelect %_ptr_Function`.
        let spvasm = r#"
                       OpCapability Shader
                       OpMemoryModel Logical GLSL450
                       OpEntryPoint GLCompute %main "main"
                       OpExecutionMode %main LocalSize 1 1 1
               %void = OpTypeVoid
                 %fn = OpTypeFunction %void
               %uint = OpTypeInt 32 0
               %bool = OpTypeBool
                 %u0 = OpConstant %uint 0
                 %u1 = OpConstant %uint 1
                 %u2 = OpConstant %uint 2
                 %u4 = OpConstant %uint 4
                %u10 = OpConstant %uint 10
              %arr_t = OpTypeArray %uint %u4
            %ptr_arr = OpTypePointer Function %arr_t
              %ptr_u = OpTypePointer Function %uint
               %main = OpFunction %void None %fn
              %entry = OpLabel
                %arr = OpVariable %ptr_arr Function
                       OpBranch %head
               %head = OpLabel
                  %i = OpPhi %uint %u0 %entry %inext %cont
                %cmp = OpULessThan %bool %i %u10
                       OpLoopMerge %merge %cont None
                       OpBranchConditional %cmp %body %merge
               %body = OpLabel
                  %c = OpULessThan %bool %i %u2
                       OpSelectionMerge %join None
                       OpBranchConditional %c %t %f
                 %t = OpLabel
                 %pt = OpAccessChain %ptr_u %arr %u2
                       OpBranch %join
                 %f = OpLabel
                 %pf = OpAccessChain %ptr_u %arr %u2
                       OpBranch %join
               %join = OpLabel
                  %p = OpPhi %ptr_u %pt %t %pf %f
                       OpStore %p %i
                       OpBranch %cont
               %cont = OpLabel
              %inext = OpIAdd %uint %i %u1
                       OpBranch %head
              %merge = OpLabel
                       OpReturn
                       OpFunctionEnd
        "#;
        let Some(spv) = assemble(spvasm) else { return };
        let out = relooper_bytes(&spv);
        assert!(
            validates(&out),
            "relooper output must validate (invariant Function pointer phi rematerialized without a select)"
        );
    }
}
