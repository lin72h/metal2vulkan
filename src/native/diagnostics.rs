//! Read-only diagnostic / measurement passes over parsed AIR: the typed-SSA (`tir`) soundness
//! self-checks, the storage/pointee/param-pointee carrier comparisons against the emitter's stateful
//! ground truth, and the structured-reject / irreducible-region reports. None of these touch the
//! emission path; they drive the validation binary's census subcommands. Kept out of `mod.rs` so the
//! emitter entry points and byte-rewrite adapters read cleanly.

use super::*;

/// R2 diagnostic: the per-function structured-plan reject reason (`None` = the function would be
/// admitted by `structured_plan`, i.e. emitted structured-by-construction; a reject emits its
/// inferred merges unrepaired and ships via the relooper retry). Mirrors the gate the emitter
/// consults by default (R2 module 4), for aggregating the frontier `cfg` bucket by which restructure
/// class must be extended next. Returns one entry per parsed function.
pub fn structured_reject_reasons(san_ll: &str) -> Result<Vec<Option<String>>, String> {
    let parsed = LlModule::parse(san_ll)?;
    Ok(parsed
        .functions
        .iter()
        .map(|f| {
            let blocks = cfg::lower_unstructured_switches(&f.blocks);
            cfg::structured_reject_reason(&blocks)
        })
        .collect())
}

/// R2 diagnostic: compact structural witnesses for the `selection:cond-other` reject class. This is
/// intentionally read-only and mirrors the CFG lowering used by `structured_reject_reason`; it helps
/// derive the smaller switch/loop construct-tree region without using the huge `SPI_WHY` skeleton dump
/// as the primary evidence.
pub fn cond_other_witness_report(san_ll: &str) -> Result<Vec<String>, String> {
    let parsed = LlModule::parse(san_ll)?;
    let mut out = Vec::new();
    for (index, f) in parsed.functions.iter().enumerate() {
        let blocks = cfg::lower_unstructured_switches(&f.blocks);
        let witnesses = cfg::cond_other_witness_lines(&blocks);
        for witness in witnesses {
            out.push(format!("fn={index} name={} {witness}", f.name));
        }
    }
    Ok(out)
}

/// R2 diagnostic: compact structural witnesses for `selection:straddle-loop-merge` rows. This is
/// intentionally read-only and avoids full module emission; it reports the exact loop/enclosing-construct
/// pair that triggers the self-check and whether the existing straddle splitter can target that loop
/// merge in the graph it mutates.
pub fn straddle_witness_report(san_ll: &str) -> Result<Vec<String>, String> {
    let parsed = LlModule::parse(san_ll)?;
    let mut out = Vec::new();
    for (index, f) in parsed.functions.iter().enumerate() {
        let blocks = cfg::lower_unstructured_switches(&f.blocks);
        let witnesses = cfg::straddle_witness_lines(&blocks);
        for witness in witnesses {
            out.push(format!("fn={index} name={} {witness}", f.name));
        }
    }
    Ok(out)
}

/// R2 diagnostic: run the bounded straddle regionalizer without emitting SPIR-V. This checks whether
/// the source-derived single-entry/two-exit wrapper can be built for each function and whether the
/// resulting block graph is accepted by the structured planner. It is read-only; production adoption
/// must wait until the representative follow-on blockers admit and remain whole-module `spirv-val`
/// gated.
pub fn straddle_region_report(san_ll: &str) -> Result<Vec<String>, String> {
    let parsed = LlModule::parse(san_ll)?;
    let mut out = Vec::new();
    for (index, f) in parsed.functions.iter().enumerate() {
        let blocks = cfg::lower_unstructured_switches(&f.blocks);
        let reject_reason = cfg::structured_reject_reason(&blocks);
        let source_reason = reject_reason.clone().unwrap_or_else(|| "ADMIT".to_string());
        match cfg::renest_straddle_loop_merge(&blocks, reject_reason.as_deref()) {
            Ok(Some(candidate)) => {
                let candidate_status = cfg::construct_tree_reject_reason(&candidate)
                    .unwrap_or_else(|| "ADMIT".to_string());
                out.push(format!(
                    "fn={index} name={} source={} candidate=some blocks={} status={}",
                    f.name,
                    source_reason,
                    candidate.len(),
                    candidate_status
                ));
                for witness in cfg::construct_tree_gate_witness_lines(&candidate) {
                    out.push(format!(
                        "fn={index} name={} candidate-gate {witness}",
                        f.name
                    ));
                }
                for witness in cfg::cond_other_witness_lines(&candidate) {
                    out.push(format!(
                        "fn={index} name={} candidate-followup {witness}",
                        f.name
                    ));
                }
                for witness in cfg::cond_phi_shared_witness_lines(&candidate) {
                    out.push(format!(
                        "fn={index} name={} candidate-followup {witness}",
                        f.name
                    ));
                }
            }
            Ok(None) => out.push(format!(
                "fn={index} name={} source={} candidate=none",
                f.name, source_reason
            )),
            Err(error) => out.push(format!(
                "fn={index} name={} source={} candidate=decline reason={error}",
                f.name, source_reason
            )),
        }
    }
    Ok(out)
}

/// Resume-lever diagnostic (`relooper-primary` M-C4 safety census): for each function that
/// `structured_plan` REJECTS (i.e. emits unrepaired and ships via the relooper retry), classify the M-C4 relooper
/// hazard from its source loop-nesting forest. Returns `Some((reject_reason, category))` for a
/// rejecting function and `None` for an admitting one, where `category` is:
///   - `loop-free`    — no natural loop: the relooper emits a pure `OpSwitch` state machine with no
///     loop nesting, so the M-C4 "nested loop through the shared outer merge runs
///     zero-trip" miscompile cannot arise. Safe to route relooper-primary.
///   - `flat-loops`   — one or more loops, none nested inside another (`parent == None` for all).
///   - `nested-loops` — at least one loop is nested (`parent.is_some()`): the M-C4 hazard surface.
///     Tallied over private capture sets (`historical reject-class probes`) this measures how much
///     of the reject census the `relooper-primary` lever could absorb WITHOUT the Phase-5 re-nesting
///     rewrite — the un-measured fact gating that decision. Read-only: touches no emission path.
pub fn structured_reject_loop_classes(
    san_ll: &str,
) -> Result<Vec<Option<(String, &'static str)>>, String> {
    let parsed = LlModule::parse(san_ll)?;
    Ok(parsed
        .functions
        .iter()
        .map(|f| {
            let blocks = cfg::lower_unstructured_switches(&f.blocks);
            let reason = cfg::structured_reject_reason(&blocks)?;
            let forest = cfg::loopforest::analyze(&blocks);
            // Max loop-nesting depth = longest parent chain over the forest (0 = loop-free,
            // 1 = flat loops, >=2 = nested). This scopes the M-C4 relooper fix: depth-2 needs only
            // "wrap the outer loop as a real OpLoopMerge" while depth>=3 needs full recursive nesting.
            // (header, parent) pairs snapshotted so the depth walk does not re-borrow `forest.loops`.
            let parent_of: HashMap<&str, Option<&str>> = forest
                .loops
                .iter()
                .map(|l| (l.header.as_str(), l.parent.as_deref()))
                .collect();
            let max_depth = parent_of
                .keys()
                .map(|h| {
                    let mut d = 1usize;
                    let mut cur = *h;
                    while let Some(Some(p)) = parent_of.get(cur) {
                        d += 1;
                        cur = p;
                    }
                    d
                })
                .max()
                .unwrap_or(0);
            let category = match max_depth {
                0 => "loop-free",
                1 => "flat-loops",
                2 => "nested-loops-d2",
                _ => "nested-loops-d3+",
            };
            Some((reason, category))
        })
        .collect())
}

/// R3 validation: parse every function in `san_ll` with the typed SSA IR (`tir`) and check it against
/// the shipped string path. Returns per-module tallies `(functions, values_typed, term_mismatches,
/// build_errors)`: `values_typed` counts SSA results whose type `tir` resolved up front;
/// `term_mismatches` counts blocks where `tir`'s structured-terminator successors disagree with the
/// proven string-based `block_successors` (must be 0 — that is the correctness gate); `build_errors`
/// counts functions `build_from_blocks` could not lower (a block without a typed carrier). Aggregated
/// over private capture sets this proves the typed IR parses all real AIR and resolves types, without touching
/// emission.
/// Per-GEP pointee-resolution report for one module: `(function, gep_result_value,
/// resolved_pointee_debug)` where the pointee is `None` when `tir` could not infer it (dynamic index
/// or an aggregate-walk gap). Diagnostic for the R3 emission-wiring work — lets the validation binary
/// show, for a specific regression case, which `getelementptr` results `tir` resolves a pointee for.
pub fn tir_gep_pointee_report(
    san_ll: &str,
) -> Result<Vec<(String, String, Option<String>)>, String> {
    let parsed = LlModule::parse(san_ll)?;
    let mut out = Vec::new();
    for f in &parsed.functions {
        let split = f.blocks.clone();
        let Ok(tir) = tir::build_from_blocks(&split) else {
            continue;
        };
        for tb in &tir.blocks {
            for inst in &tb.insts {
                let Some(result) = &inst.result else { continue };
                let is_gep = inst.opcode == "getelementptr";
                if is_gep {
                    let pointee = tir.pointer_pointees.get(result).map(|p| format!("{p:?}"));
                    out.push((f.name.clone(), result.clone(), pointee));
                }
            }
        }
    }
    Ok(out)
}

/// R2 diagnostic: per-function irreducible (multi-entry) regions of the source CFG — the cycles the
/// dominance loop forest is blind to (no single dominating header), which would need node-splitting to
/// structure. Returns, for each function that has at least one such region, `(function name, [(region
/// node count, entry count) per region])`. A function with a fully reducible CFG contributes nothing.
/// Built on the same `split_body_blocks` + `lower_unstructured_switches` lowering the structurizer
/// consumes, so the regions match what emission would face. Drives `--irreducible`, which MEASURED THE
/// POPULATION EMPTY: 0 irreducible regions over all 16,071 frontier + banked rows, establishing that the
/// residual cfg frontier is REDUCIBLE-CFG selection/cost-budget failures, not node-split territory.
pub fn irreducible_region_report(
    san_ll: &str,
) -> Result<Vec<(String, Vec<(usize, usize)>)>, String> {
    let parsed = LlModule::parse(san_ll)?;
    let mut out = Vec::new();
    for f in &parsed.functions {
        let blocks = cfg::lower_unstructured_switches(&f.blocks);
        let regions = cfg::loopforest::irreducible_regions(&blocks);
        if !regions.is_empty() {
            out.push((
                f.name.clone(),
                regions
                    .iter()
                    .map(|r| (r.nodes.len(), r.entries.len()))
                    .collect(),
            ));
        }
    }
    Ok(out)
}

pub fn tir_self_check(san_ll: &str) -> Result<TirCheckStats, String> {
    let parsed = LlModule::parse(san_ll)?;
    let mut stats = TirCheckStats::default();
    for f in &parsed.functions {
        stats.functions += 1;
        let split = f.blocks.clone();
        let tir = match tir::build_from_blocks(&split) {
            Ok(t) => t,
            Err(_) => {
                stats.build_errors += 1;
                continue;
            }
        };
        accumulate_tir_soundness(&tir, f, &parsed.types, &mut stats);
    }
    Ok(stats)
}

/// Whether an operand's use-site type is compatible with the type its def recorded, under the same
/// type-aliasing the emitter tolerates plus the opaque-pointer reality: `i1` and `Bool` are the same
/// LLVM type, and `LlType::Ptr` is addrspace-only so a def and a use may legitimately carry different
/// addrspaces (LLVM opaque `ptr` has no pointee and many use sites omit the addrspace). Vectors recurse
/// elementwise at equal lane count.
fn operand_type_compatible(def: &ir::LlType, used: &ir::LlType) -> bool {
    use ir::LlType::{Bool, Int, Ptr, Vector};
    match (def, used) {
        _ if def == used => true,
        (Bool, Int(1)) | (Int(1), Bool) => true,
        (Ptr(_), Ptr(_)) => true,
        (Vector(d, dn), Vector(u, un)) if dn == un => operand_type_compatible(d, u),
        _ => false,
    }
}

/// Tally one function's typed-graph soundness into `stats`: the def/use SSA-closure check
/// (`dangling_uses`), resolved-operand coverage + use/def type agreement (`operands_*`,
/// `operand_type_mismatches`), and result-type / GEP-pointee coverage (`values_*`, `gep_*`). Shared by
/// `tir_self_check` (graph built from the parse-time `f.blocks`) and `tir_structured_self_check` (graph
/// built from the structurized block list) so both populations get the identical soundness verdict.
/// Note: `defined` derives from `tir`'s own blocks, so for the structurized graph the dangling-use check
/// correctly validates the synthetic `%metal2vulkan.lmerge.*` phis against the structurized SSA universe
/// (their incomings reference values defined in the structurized blocks, not the parse-time `f.blocks`).
fn accumulate_tir_soundness(
    tir: &tir::TirFunction,
    f: &ir::LlFunction,
    parsed_types: &HashMap<String, ir::LlType>,
    stats: &mut TirCheckStats,
) {
    stats.values_typed += tir.value_types.len();
    let (use_resolved, use_beyond_gep, use_conflicts) = tir::use_pointee_coverage(tir);
    stats.use_pointees_resolved += use_resolved;
    stats.use_pointee_beyond_gep += use_beyond_gep;
    stats.use_pointee_conflicts += use_conflicts;
    // SSA-closure: the set of `%name`s that ARE defined — every result, every function param, and
    // every module named-type (a `%struct.X` token appears in type positions, not as a value def).
    let mut defined: HashSet<&str> = parsed_types.keys().map(String::as_str).collect();
    for (p, _) in &f.params {
        defined.insert(p.as_str());
    }
    for tb in &tir.blocks {
        for inst in &tb.insts {
            if let Some(r) = &inst.result {
                defined.insert(r.as_str());
            }
        }
    }
    for tb in &tir.blocks {
        for inst in &tb.insts {
            inst.visit_uses(|u| {
                if !defined.contains(u) {
                    stats.dangling_uses += 1;
                }
            });
        }
    }
    // Resolved-operand coverage + typed-graph soundness: every `Value` operand's use-site type must
    // equal the type its def recorded (param type or `value_types`).
    let param_types: HashMap<&str, &ir::LlType> =
        f.params.iter().map(|(n, t)| (n.as_str(), t)).collect();
    for tb in &tir.blocks {
        for inst in &tb.insts {
            for op in &inst.operands {
                stats.operands_total += 1;
                match op {
                    tir::TirOperand::Unresolved => {
                        if crate::env_vars::tir_dbg() {
                            eprintln!("TIR-UNRESOLVED-OP {}", inst.opcode);
                        }
                    }
                    tir::TirOperand::Const { .. } => stats.operands_resolved += 1,
                    tir::TirOperand::Value { name, ty } => {
                        stats.operands_resolved += 1;
                        let def_ty = param_types
                            .get(name.as_str())
                            .copied()
                            .or_else(|| tir.value_types.get(name.as_str()));
                        if let Some(def_ty) = def_ty {
                            stats.operand_value_defs_checked += 1;
                            if !operand_type_compatible(def_ty, ty) {
                                stats.operand_type_mismatches += 1;
                                if crate::env_vars::tir_dbg() {
                                    eprintln!("TIR-OPTYPE {name} def={def_ty:?} use={ty:?}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // Coverage: count defining instructions, and tally the opcodes whose result type tir did NOT
    // resolve (the next type-resolution increment's target).
    for tb in &tir.blocks {
        for inst in &tb.insts {
            if let Some(result) = &inst.result {
                stats.values_total += 1;
                if inst.result_ty.is_none() && crate::env_vars::tir_dbg() {
                    eprintln!("TIR-UNTYPED {}", inst.opcode);
                }
                // GEP pointee coverage: count GEP defining instrs and how many got a pointee.
                let is_gep = inst.opcode == "getelementptr";
                if is_gep {
                    stats.gep_results += 1;
                    if tir.pointer_pointees.contains_key(result) {
                        stats.gep_pointees_resolved += 1;
                    }
                }
            }
        }
    }
}

/// R3 endgame validation: build the typed SSA graph from the **structurized** CFG — the same
/// post-structurization block list emission walks (`cfg::structured_plan`'s reordered blocks + the
/// synthetic `%metal2vulkan.lmerge.*` merge blocks/phis it inserts) — and prove it sound, the way
/// `tir_self_check` proves the parse-time `f.blocks` graph. This is the prerequisite the HARD BOUND
/// identified: phi/store emission cannot be migrated from the parse-time `f.blocks` graph because the
/// structurizer rewrites those between parse and emit; a graph built from the structurized blocks
/// carries the *post*-rewrite
/// operands, so its `Value` operands and synthetic-phi incomings reference the SSA universe emission
/// actually sees. Built and validated alongside the string path; NOT yet consumed by emission (the
/// byte gate guards consumption). Mirrors `emit_function`'s pre-loop exactly: split → lower
/// unstructured switches → `structured_plan` (and on a structurizer reject, validate the lowered
/// blocks retained for raw-CFG construction).
pub fn tir_structured_self_check(san_ll: &str) -> Result<TirCheckStats, String> {
    let parsed = LlModule::parse(san_ll)?;
    let mut stats = TirCheckStats::default();
    for f in &parsed.functions {
        stats.functions += 1;
        let split = f.blocks.clone();
        let mut body_blocks = cfg::lower_unstructured_switches(&split);
        if let Some(plan) = cfg::structured_plan(&body_blocks) {
            body_blocks = plan.blocks;
        }
        let tir = match tir::build_from_blocks(&body_blocks) {
            Ok(t) => t,
            Err(_) => {
                stats.build_errors += 1;
                continue;
            }
        };
        accumulate_tir_soundness(&tir, f, &parsed.types, &mut stats);
    }
    Ok(stats)
}

/// Aggregated `tir_self_check` tallies (see that function).
#[derive(Debug, Default, Clone, Copy)]
pub struct TirCheckStats {
    pub functions: usize,
    pub values_typed: usize,
    pub values_total: usize,
    pub term_mismatches: usize,
    pub build_errors: usize,
    /// def/use SSA-closure check: operand `%name` uses not defined by any result/param/global. A
    /// sound typed graph has 0 (every value used is defined somewhere); a non-zero count flags either
    /// a use-extraction bug or an AIR shape the operand scan mis-reads.
    pub dangling_uses: usize,
    /// Pointer-typed SSA results whose pointee `tir` inferred (currently `getelementptr` results with
    /// a fully-constant index path). `gep_results` is the denominator (all GEP defining instrs).
    pub gep_results: usize,
    pub gep_pointees_resolved: usize,
    /// Resolved-operand coverage: total operands across all instructions vs. how many tir lowered to a
    /// typed `Value`/`Const` (not `Unresolved`). The gap is the opcodes whose operand layout is not yet
    /// lowered (gep/call/aggregate ops).
    pub operands_total: usize,
    pub operands_resolved: usize,
    /// Typed-graph soundness: `Value` operands whose use-site type disagrees with the type the def
    /// recorded (`value_types`/param type). A sound graph has 0 — a non-zero count is either a tir
    /// operand-parse bug or a genuine use/def type drift in the input. Only operands whose def type is
    /// known are checked (the denominator is `operand_value_defs_checked`).
    pub operand_type_mismatches: usize,
    pub operand_value_defs_checked: usize,
    /// USE-based pointee coverage (the R4 pointer-typing foundation): `use_pointees_resolved` counts
    /// distinct dereferenced pointer values tir gave a use-implied pointee; `use_pointee_beyond_gep` is
    /// the subset whose pointee is NOT already a GEP-result pointee (`pointer_pointees`) — the NET-NEW
    /// type information over the existing map, i.e. the loaded / phi / select / param / bitcast pointers
    /// that today default to a byte (`uchar`) pointer. `use_pointee_conflicts` is the number of pointers
    /// whose dereferences disagree on a pointee (a real reinterpret signal, not an error).
    pub use_pointees_resolved: usize,
    pub use_pointee_beyond_gep: usize,
    pub use_pointee_conflicts: usize,
}

/// M1 storage-carrier measurement (the pointer-typing rewrite). Compares the from-tir storage
/// derivation (`tir::derive_pointer_storage`) against the emitter's actual per-value `pointer_storage`
/// — the stateful ground truth — for every function in the module, so the next increment knows exactly
/// which pointers the structural derivation gets right and which still need the emitter's stateful
/// rules (buffer-modeling → `StorageBuffer`, raw byte-load → `Private`, merge-meta storage). The
/// pointee half of the carrier is already proven (`use_pointees`, 0 conflicts); this is the storage
/// half, NOT yet consumed by emission.
#[derive(Debug, Default, Clone, Copy)]
pub struct StorageCheckStats {
    /// Functions whose emitter snapshot and tir derivation were both produced and compared.
    pub functions: usize,
    /// Emitter pointer values (the denominator): every `%name` the emitter assigned a storage class.
    pub emitter_values: usize,
    /// Emitter values the tir derivation also resolved AND agreed on.
    pub agree: usize,
    /// Emitter values the tir derivation resolved to a DIFFERENT class (the residual to close).
    pub diverge: usize,
    /// Emitter values the tir derivation left unmapped (no rule produced a class).
    pub tir_missing: usize,
    /// Emitter values in a LOGICAL storage class (`UniformConstant`/`Function`/`Workgroup`/
    /// `StorageBuffer`) — the real per-value pointer storage, excluding the `Private` byte-placeholder
    /// fallback. This is the population a typed-storage carrier must reproduce.
    pub logical_values: usize,
    /// Logical-class emitter values the tir derivation agreed on (the true carrier-faithfulness
    /// numerator).
    pub logical_agree: usize,
    /// Emitter values lowered to `Private` — dominated by the unmodeled/raw byte-placeholder fallback
    /// for derived device pointers (the surface M2's typed access-chain lowering converts), plus genuine
    /// `addrspace(0)` locals. NOT a carrier target.
    pub private_values: usize,
}

/// Run [`StorageCheckStats`] over one module's AIR. Emits once with storage capture to get the
/// emitter's ground-truth `pointer_storage` per function, then for each function builds the
/// structurized typed graph (the same block list emission walks — see `tir_structured_self_check`),
/// derives storage from it, and tallies agreement. A function the emitter could not emit (it never
/// reaches the snapshot) is simply absent from the comparison.
pub fn tir_storage_check(san_ll: &str) -> Result<StorageCheckStats, String> {
    let parsed = LlModule::parse(san_ll)?;
    let snapshots = Emitter::new(parsed.clone()).emit_collecting_storage()?;
    let snap_by_fn: HashMap<&str, &HashMap<String, StorageClass>> =
        snapshots.iter().map(|(n, m)| (n.as_str(), m)).collect();
    let mut stats = StorageCheckStats::default();
    for f in &parsed.functions {
        let Some(emitter_map) = snap_by_fn.get(f.name.as_str()) else {
            continue;
        };
        let split = f.blocks.clone();
        let mut body_blocks = cfg::lower_unstructured_switches(&split);
        if let Some(plan) = cfg::structured_plan(&body_blocks) {
            body_blocks = plan.blocks;
        }
        let tir = match tir::build_from_blocks(&body_blocks) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let derived = tir::derive_pointer_storage(&tir, &f.params, &parsed.types);
        stats.functions += 1;
        for (name, emitter_storage) in emitter_map.iter() {
            stats.emitter_values += 1;
            let is_logical = *emitter_storage != StorageClass::Private;
            if is_logical {
                stats.logical_values += 1;
            } else {
                stats.private_values += 1;
            }
            match derived.get(name) {
                Some(d) if d == emitter_storage => {
                    stats.agree += 1;
                    if is_logical {
                        stats.logical_agree += 1;
                    }
                }
                Some(d) => {
                    stats.diverge += 1;
                    if crate::env_vars::storage_dbg() {
                        eprintln!(
                            "STORAGE-DIVERGE {} {name} emitter={emitter_storage:?} tir={d:?}",
                            f.name
                        );
                    }
                }
                None => {
                    stats.tir_missing += 1;
                    if crate::env_vars::storage_dbg() {
                        eprintln!(
                            "STORAGE-MISSING {} {name} emitter={emitter_storage:?}",
                            f.name
                        );
                    }
                }
            }
        }
    }
    Ok(stats)
}

/// M2 pointee-carrier measurement (the pointer-typing rewrite, the pointee half). The storage half has
/// [`tir_storage_check`]; this is its exact analogue for the pointee. It compares the from-tir
/// `use_pointees` carrier (`tir::TirFunction::use_pointees`, the use-implied pointee propagated to a
/// fixpoint across select/phi/freeze) against the emitter's actual per-value `pointer_pointees` — the
/// stateful ground truth emission uses today, populated across GEP/load/phi/select/bitcast/buffer sites —
/// for every function, so the M2 increment knows exactly which pointer defs the carrier already reproduces
/// (safe to flip) and which still need the emitter's stateful rules (the reconciliation set). Byte-invisible:
/// a standalone measurement, exactly as `tir_storage_check`/`param_pointee_check` are.
#[derive(Debug, Default, Clone, Copy)]
pub struct PointeeCheckStats {
    /// Functions whose emitter snapshot and tir graph were both produced and compared.
    pub functions: usize,
    /// Emitter pointer values (the denominator): every `%name` the emitter assigned a pointee.
    pub emitter_values: usize,
    /// Emitter values the carrier also resolved AND agreed on (under the emitter's own type aliasing:
    /// `Ptr`≡`Ptr` addrspace-only, `i1`≡`Bool`, vectors elementwise). Safe-to-flip today.
    pub agree: usize,
    /// Emitter values the carrier resolved to a DIFFERENT pointee — the reconciliation set an M2
    /// consumer must settle before trusting the carrier over the side-table (`METAL2VULKAN_POINTEE_DBG=1`
    /// enumerates them).
    pub diverge: usize,
    /// Emitter values the carrier left unmapped (the value is never dereferenced in-body, so only the
    /// emitter's stateful derivation typed it — e.g. globals typed at declaration, dead pointers).
    pub carrier_missing: usize,
    /// Emitter values lowered to the `uchar` (`Int(8)`) byte placeholder — the raw/unmodeled fallback the
    /// carrier's real use-type replaces. The surface M2's typed pointer rewrite converts.
    pub byte_placeholder: usize,
    /// Byte-placeholder emitter values the carrier UPGRADES to a concrete non-`Int(8)` pointee — the
    /// direct net-new type win M2 lands (a `float`/vector/struct pointee where the emitter defaulted to a
    /// byte pointer). A subset of both `byte_placeholder` and `diverge`.
    pub carrier_upgrades: usize,
}

/// Run [`PointeeCheckStats`] over one module's AIR. Emits once with pointee capture to get the emitter's
/// ground-truth `pointer_pointees` per function, then for each function builds the structurized typed
/// graph (the same block list emission walks — mirroring [`tir_storage_check`]) and tallies agreement of
/// its `use_pointees` carrier. A function the emitter could not emit (never reaches the snapshot) is
/// simply absent from the comparison.
pub fn tir_pointee_check(san_ll: &str) -> Result<PointeeCheckStats, String> {
    let parsed = LlModule::parse(san_ll)?;
    let snapshots = Emitter::new(parsed.clone()).emit_collecting_pointees()?;
    let snap_by_fn: HashMap<&str, &HashMap<String, ir::LlType>> =
        snapshots.iter().map(|(n, m)| (n.as_str(), m)).collect();
    let mut stats = PointeeCheckStats::default();
    for f in &parsed.functions {
        let Some(emitter_map) = snap_by_fn.get(f.name.as_str()) else {
            continue;
        };
        let split = f.blocks.clone();
        let mut body_blocks = cfg::lower_unstructured_switches(&split);
        if let Some(plan) = cfg::structured_plan(&body_blocks) {
            body_blocks = plan.blocks;
        }
        let tir = match tir::build_from_blocks(&body_blocks) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let carrier = &tir.use_pointees;
        stats.functions += 1;
        for (name, emitter_pointee) in emitter_map.iter() {
            stats.emitter_values += 1;
            // Canonicalize BOTH pointees before comparison. The emitter's `pointer_pointees` stores
            // already-resolved structural types, but the carrier's `use_pointees` holds whatever the
            // AIR text names — including unresolved `Named("%struct._half8")` aliases whose definition
            // is the emitter's structural form (`Struct([Array(Half,8)])`). Comparing raw counts those
            // as divergence though they are the SAME type; resolving both via the module's type table
            // (`resolve_known_type`: Named → definition, i1 → Bool, 1-lane vector → scalar) reports the
            // TRUE reconciliation set an M2 consumer must settle. Emitter-side resolve is idempotent.
            let emitter_pointee = parsed.resolve_known_type(emitter_pointee);
            let is_byte = emitter_pointee == ir::LlType::Int(8);
            if is_byte {
                stats.byte_placeholder += 1;
            }
            match carrier.get(name).map(|c| parsed.resolve_known_type(c)) {
                Some(c) if operand_type_compatible(&emitter_pointee, &c) => stats.agree += 1,
                Some(c) => {
                    stats.diverge += 1;
                    if is_byte && c != ir::LlType::Int(8) {
                        stats.carrier_upgrades += 1;
                    }
                    if crate::env_vars::pointee_dbg() {
                        eprintln!(
                            "POINTEE-DIVERGE {} {name} emitter={emitter_pointee:?} carrier={c:?}",
                            f.name
                        );
                    }
                }
                None => {
                    stats.carrier_missing += 1;
                    if crate::env_vars::pointee_dbg() {
                        eprintln!(
                            "POINTEE-MISSING {} {name} emitter={emitter_pointee:?}",
                            f.name
                        );
                    }
                }
            }
        }
    }
    Ok(stats)
}

/// S18 advisory-sidecar measurement (`--param-pointee-check`): quantify the information content of
/// the emitter's `function_param_pointees` sidecar — the per-`(function, param-index)` pointer
/// pointee it infers from CALL SITES — against the INDEPENDENT use-site pointee the module's own
/// `infer_pointer_pointees` derives from how each function's body dereferences (GEPs) that parameter.
/// The two inferences are keyed the same way (function name + parameter SSA name), so agreement means
/// the sidecar is redundant with body inference; `use_missing` counts params the sidecar types but the
/// body never dereferences (the knowledge the sidecar UNIQUELY carries — a passes layer relying only on
/// body inference would lack it, the value an S19 consumer would gain). Byte-invisible: a standalone
/// measurement, exactly as `tir_storage_check` is. It carries nothing across the seam yet — that is the
/// next S18 increment; this first step establishes whether the sidecar is worth carrying.
#[derive(Default)]
pub struct ParamPointeeStats {
    /// Functions with at least one call-site-inferred pointer parameter.
    pub functions: usize,
    /// Sidecar entries compared (the denominator): every `(function, param-index)` the emitter
    /// call-site-inferred a pointee for and whose parameter still exists.
    pub sidecar_values: usize,
    /// Sidecar entries the use-site (body-GEP) inference also resolved AND agreed on.
    pub agree: usize,
    /// Sidecar entries the use-site inference resolved to a DIFFERENT pointee (a real divergence to
    /// reconcile before an S19 consumer trusts one over the other).
    pub diverge: usize,
    /// Sidecar entries the use-site inference left unresolved — the parameter is never GEP'd in-body,
    /// so ONLY the call-site sidecar knows its pointee. The population that justifies the sidecar.
    pub use_missing: usize,
}

/// Run [`ParamPointeeStats`] over one module's AIR. See the struct docs for the measurement.
pub fn param_pointee_check(san_ll: &str) -> Result<ParamPointeeStats, String> {
    let parsed = LlModule::parse(san_ll)?;
    // The module's own use-site pointee inference is populated during parse.
    let use_site = &parsed.ptr_pointees;
    // The emitter's call-site sidecar (the production inference).
    let sidecar = Emitter::new(parsed.clone()).emit_collecting_param_pointees()?;
    let mut stats = ParamPointeeStats::default();
    let mut fns_counted: HashSet<&str> = HashSet::new();
    for ((fn_name, idx), call_pointee) in &sidecar {
        let Some(func) = parsed.functions.iter().find(|f| &f.name == fn_name) else {
            continue;
        };
        let Some((param_name, _)) = func.params.get(*idx) else {
            continue;
        };
        stats.sidecar_values += 1;
        fns_counted.insert(fn_name.as_str());
        match use_site.get(&(fn_name.clone(), param_name.clone())) {
            Some(use_pointee) if use_pointee == call_pointee => stats.agree += 1,
            Some(use_pointee) => {
                stats.diverge += 1;
                if crate::env_vars::param_pointee_dbg() {
                    eprintln!(
                        "PARAM-POINTEE-DIVERGE {fn_name} #{idx} {param_name} call={call_pointee:?} use={use_pointee:?}"
                    );
                }
            }
            None => {
                stats.use_missing += 1;
                if crate::env_vars::param_pointee_dbg() {
                    eprintln!(
                        "PARAM-POINTEE-USE-MISSING {fn_name} #{idx} {param_name} call={call_pointee:?}"
                    );
                }
            }
        }
    }
    stats.functions = fns_counted.len();
    Ok(stats)
}
