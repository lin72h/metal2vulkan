//! Pre-submission loop-budget instrumentation for the Metal oracle.
//!
//! A committed Metal command buffer cannot be cancelled — once a compute kernel enters an
//! unbounded loop it pins the GPU until the machine is rebooted. No choice of input data can
//! prevent this (halting problem), so the only structural guarantee is to bound GPU-side work
//! **before** `command_buffer.commit()`.
//!
//! This module takes the Apple-dialect textual LLVM IR that `metal-objdump --disassemble-all`
//! emits (and `metal-as` re-assembles) and classifies each corpus case:
//!
//! - [`GuardPlan::LoopFree`] — no loops anywhere; provably bounded; run unchanged.
//! - [`GuardPlan::Instrumented`] — inject a per-thread back-edge budget so every loop is forced to
//!   exit after [`LOOP_BUDGET_BACKEDGES`] iterations; the module is byte-for-byte equivalent for
//!   any run that stays under budget (the guard only touches private budget memory and a
//!   never-taken branch), so provably-bounded goldens are unchanged.
//! - [`GuardPlan::Quarantine`] — anything we cannot instrument *and verify* (switch back-edges,
//!   unparseable control flow, loop-calls-loopy-callee composition). The caller must not dispatch.
//!
//! The transform is intentionally conservative: when in doubt it quarantines. The caller also
//! round-trips the instrumented text through `metal-as`/`metallib`; if that rejects it, the case
//! is quarantined too. So a bug in this pass can never wedge the GPU — worst case it over-skips.
//!
//! Everything here is a pure string transform with no Metal/objc2 dependency, so it is unit-tested
//! on any OS.

use std::collections::{HashMap, HashSet};

/// Per-thread, per-function back-edge budget. Every loop iteration traverses a back-edge, which
/// decrements this counter; when it reaches zero control jumps to the function's exit. The oracle
/// marks instrumented kernels as `compare=none`; this cap is therefore a validation liveness guard,
/// not a product semantic contract. Keep it large enough for bounded harness work but small enough
/// that runaway SIMT reductions do not run into the corpus worker's wall timeout.
pub const LOOP_BUDGET_BACKEDGES: i32 = 1 << 12;
/// Cap used only when reproducing an already-banked semantic `plan_version < 3` Metal golden.
/// Historic `compare=none` rows are non-semantic and always use the safer cap above before they are
/// quarantined; their smoke bytes do not justify replaying the older, much larger budget.
pub const LEGACY_LOOP_BUDGET_BACKEDGES: i32 = 1 << 18;

/// Result of classifying one module (entry + all defined functions).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GuardPlan {
    /// No loops in any defined function — safe to run as-is.
    LoopFree,
    /// Instrumented module text, ready for `metal-as`.
    Instrumented(String),
    /// Cannot instrument safely; the reason (for the ledger). Do not dispatch.
    Quarantine(String),
}

/// Validation-only facts about the concrete inputs the oracle will bind for one run.
#[derive(Default)]
pub struct LoopInputFacts<'a> {
    pub fc_values: &'a [(usize, u64)],
    pub arg_values: &'a [(String, i128)],
    pub arg_float_values: &'a [(String, f64)],
    pub arg_upper_bounds: &'a [(String, i128)],
    pub arg_field_values: &'a [(String, Vec<i32>, i128)],
    /// Exact integer values loaded through constant byte (`getelementptr i8`) offsets.
    pub arg_byte_values: &'a [(String, usize, i128)],
    pub arg_vector_values: &'a [(String, usize, i128)],
    pub arg_vector_upper_bounds: &'a [(String, usize, i128)],
    pub texture_extents: &'a [(String, [i128; 3])],
    pub imageblock_extent: Option<[i128; 2]>,
}

/// Classify a module and, if it contains loops, return an instrumented copy that cannot run the
/// GPU unbounded. Classification starts from `entry` when it names a defined function; unknown
/// entries keep the historical whole-module behavior.
pub fn classify_and_instrument(module_text: &str, entry: &str) -> GuardPlan {
    classify_and_instrument_with_function_constants(module_text, entry, &[])
}

/// Like [`classify_and_instrument`], but treats AIR function constants listed in `fc_values`
/// as explicit integer values while proving small fixed-trip loops. The caller is responsible for
/// executing Metal with the same FC values; this is validation-only input modeling, not product
/// translation behavior.
pub fn classify_and_instrument_with_function_constants(
    module_text: &str,
    entry: &str,
    fc_values: &[(usize, u64)],
) -> GuardPlan {
    classify_and_instrument_with_input_facts(module_text, entry, fc_values, &[])
}

/// Like [`classify_and_instrument_with_function_constants`], plus exact integer values for pointer
/// arguments. Each `(arg, value)` entry names an LLVM pointer argument without `%`; loads directly
/// from that pointer are treated as the given value while proving reachability and small loops.
pub fn classify_and_instrument_with_input_facts(
    module_text: &str,
    entry: &str,
    fc_values: &[(usize, u64)],
    arg_values: &[(String, i128)],
) -> GuardPlan {
    classify_and_instrument_with_input_facts_and_fields(
        module_text,
        entry,
        fc_values,
        arg_values,
        &[],
    )
}

/// Like [`classify_and_instrument_with_input_facts`], plus exact integer values for constant GEP
/// paths rooted at pointer arguments. Each `(arg, path, value)` entry names an LLVM pointer
/// argument without `%` and a GEP index path after the leading zero root index; loads from a
/// matching derived pointer are treated as the given value.
pub fn classify_and_instrument_with_input_facts_and_fields(
    module_text: &str,
    entry: &str,
    fc_values: &[(usize, u64)],
    arg_values: &[(String, i128)],
    arg_field_values: &[(String, Vec<i32>, i128)],
) -> GuardPlan {
    classify_and_instrument_with_input_facts_bounds_and_fields(
        module_text,
        entry,
        fc_values,
        arg_values,
        &[],
        arg_field_values,
    )
}

/// Like [`classify_and_instrument_with_input_facts_and_fields`], plus upper bounds for integer
/// arguments whose exact value varies by lane (for example `thread_position_in_threadgroup`).
pub fn classify_and_instrument_with_input_facts_bounds_and_fields(
    module_text: &str,
    entry: &str,
    fc_values: &[(usize, u64)],
    arg_values: &[(String, i128)],
    arg_upper_bounds: &[(String, i128)],
    arg_field_values: &[(String, Vec<i32>, i128)],
) -> GuardPlan {
    classify_and_instrument_with_loop_input_facts(
        module_text,
        entry,
        LoopInputFacts {
            fc_values,
            arg_values,
            arg_upper_bounds,
            arg_field_values,
            ..LoopInputFacts::default()
        },
    )
}

/// Like [`classify_and_instrument_with_input_facts_bounds_and_fields`], plus exact / upper-bound
/// facts for integer vector argument lanes and exact dimensions for bound texture arguments.
pub fn classify_and_instrument_with_loop_input_facts(
    module_text: &str,
    entry: &str,
    input_facts: LoopInputFacts<'_>,
) -> GuardPlan {
    classify_and_instrument_with_loop_input_facts_and_budget(
        module_text,
        entry,
        input_facts,
        LOOP_BUDGET_BACKEDGES,
    )
}

/// Return whether the entry's reachable static call graph contains any reachable CFG cycle.
///
/// Unlike [`classify_and_instrument_with_loop_input_facts`], this does not erase loops whose trip
/// count appears small under the supplied facts. It is the conservative semantic-oracle gate:
/// proving termination is not the same as proving an aggregate GPU-time bound, especially for
/// nested loops and large loop bodies.
pub fn reachable_module_has_cfg_cycle_with_loop_input_facts(
    module_text: &str,
    entry: &str,
    input_facts: LoopInputFacts<'_>,
) -> Result<bool, String> {
    let lines = module_text.lines().collect::<Vec<_>>();
    let funcs = find_functions(&lines);
    if funcs.is_empty() {
        return Ok(false);
    }
    let facts = LoopFacts::from_module(&lines, &input_facts);
    let parsed = funcs
        .iter()
        .map(|function| parse_func_preserving_loops(&lines, function, &facts))
        .collect::<Result<Vec<_>, _>>()?;
    let reachable = reachable_functions_from_entry(&parsed, entry);
    Ok(parsed
        .iter()
        .filter(|function| reachable.contains(function.name.as_str()))
        .any(|function| !function.back_edges.is_empty()))
}

/// Classify and instrument using an explicit back-edge cap. This exists so candidate runners can
/// reproduce older banked Metal goldens while new oracle executions use the safer current cap.
pub fn classify_and_instrument_with_loop_input_facts_and_budget(
    module_text: &str,
    entry: &str,
    input_facts: LoopInputFacts<'_>,
    loop_budget_backedges: i32,
) -> GuardPlan {
    if loop_budget_backedges <= 0 {
        return GuardPlan::Quarantine("loop back-edge budget must be positive".into());
    }
    let _ = entry;
    let lines: Vec<&str> = module_text.lines().collect();
    let funcs = find_functions(&lines);
    if funcs.is_empty() {
        return GuardPlan::LoopFree;
    }

    let opaque_ptr = uses_opaque_pointers(module_text);
    let facts = LoopFacts::from_module(&lines, &input_facts);
    let func_by_name = funcs
        .iter()
        .map(|f| (f.name.as_str(), f))
        .collect::<HashMap<_, _>>();

    // Parse every function. Any unparseable control flow → quarantine (we cannot prove it halts).
    let mut parsed: Vec<ParsedFunc> = Vec::with_capacity(funcs.len());
    for f in &funcs {
        match parse_func(&lines, f, &facts) {
            Ok(pf) => parsed.push(pf),
            Err(reason) => return GuardPlan::Quarantine(reason),
        }
    }

    let reachable_funcs = reachable_functions_from_entry(&parsed, entry);

    // Transitive "contains a loop" over the static call graph (Metal has no recursion → DAG).
    let direct_loopy: HashMap<&str, bool> = parsed
        .iter()
        .map(|f| (f.name.as_str(), !f.back_edges.is_empty()))
        .collect();
    let trans_loopy = transitive_loopy(&parsed, &direct_loopy);

    // Compose gate: a loop that calls a transitively-loopy function is bounded only by
    // CAP_caller × CAP_callee. With a per-function budget that product can be huge, so refuse it.
    for f in parsed
        .iter()
        .filter(|f| reachable_funcs.contains(f.name.as_str()))
    {
        if f.back_edges.is_empty() {
            continue;
        }
        if f.loop_has_workgroup_barrier {
            return GuardPlan::Quarantine(format!(
                "loop in {:?} contains air.wg.barrier (cannot preserve uniform barrier semantics)",
                f.name
            ));
        }
        for call in &f.loop_calls {
            if *trans_loopy.get(call.callee.as_str()).unwrap_or(&false)
                && !loop_call_callee_is_bounded_with_call_facts(
                    &lines,
                    &func_by_name,
                    &facts,
                    &trans_loopy,
                    f,
                    call,
                )
            {
                return GuardPlan::Quarantine(format!(
                    "loop in {:?} calls loopy callee {:?} (unbounded composition)",
                    f.name, call.callee
                ));
            }
        }
    }

    if parsed
        .iter()
        .filter(|f| reachable_funcs.contains(f.name.as_str()))
        .all(|f| f.back_edges.is_empty())
    {
        return GuardPlan::LoopFree;
    }

    // Instrument: transform loopy functions, copy the rest verbatim, stitch the module back.
    let mut out: Vec<String> = Vec::with_capacity(lines.len() + 64);
    let mut cursor = 0usize;
    for (f, pf) in funcs.iter().zip(parsed.iter()) {
        // Emit untouched lines before this function.
        while cursor < f.define_idx {
            out.push(lines[cursor].to_string());
            cursor += 1;
        }
        if pf.back_edges.is_empty() || !reachable_funcs.contains(pf.name.as_str()) {
            for line in &lines[f.define_idx..=f.close_idx] {
                out.push(line.to_string());
            }
        } else {
            match transform_func(&lines, f, pf, opaque_ptr, loop_budget_backedges) {
                Ok(func_lines) => out.extend(func_lines),
                Err(reason) => return GuardPlan::Quarantine(reason),
            }
        }
        cursor = f.close_idx + 1;
    }
    while cursor < lines.len() {
        out.push(lines[cursor].to_string());
        cursor += 1;
    }

    let mut text = out.join("\n");
    // Preserve a trailing newline if the input had one (metal-as is not picky, but keep it clean).
    if module_text.ends_with('\n') {
        text.push('\n');
    }
    GuardPlan::Instrumented(text)
}

fn reachable_functions_from_entry(parsed: &[ParsedFunc], entry: &str) -> HashSet<String> {
    let known = parsed
        .iter()
        .map(|f| f.name.as_str())
        .collect::<HashSet<_>>();
    let Some(entry_name) = parsed
        .iter()
        .find(|f| function_symbol_matches_entry(&f.name, entry))
        .map(|f| f.name.clone())
    else {
        return known.into_iter().map(ToOwned::to_owned).collect();
    };

    let calls = parsed
        .iter()
        .map(|f| (f.name.as_str(), f.calls.as_slice()))
        .collect::<HashMap<_, _>>();
    let mut reachable = HashSet::new();
    let mut stack = vec![entry_name];
    while let Some(name) = stack.pop() {
        if !reachable.insert(name.clone()) {
            continue;
        }
        if let Some(callees) = calls.get(name.as_str()) {
            for callee in *callees {
                if known.contains(callee.as_str()) && !reachable.contains(callee.as_str()) {
                    stack.push(callee.clone());
                }
            }
        }
    }
    reachable
}

fn function_symbol_matches_entry(symbol: &str, entry: &str) -> bool {
    if symbol == entry {
        return true;
    }
    if let Some(unquoted) = symbol.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        return unquoted == entry;
    }
    false
}

// --- module structure ----------------------------------------------------------------------------

struct FuncSpan {
    define_idx: usize,
    close_idx: usize,
    name: String,
    ret_ty: String,
}

/// Locate every `define … { … }` (bodies only; `declare`s have no body). LLVM emits `{` at the end
/// of the define line and a lone `}` at column 0.
fn find_functions(lines: &[&str]) -> Vec<FuncSpan> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if lines[i].starts_with("define ") {
            let mut j = i + 1;
            let mut close = None;
            while j < lines.len() {
                if lines[j] == "}" || lines[j].trim() == "}" {
                    close = Some(j);
                    break;
                }
                if lines[j].starts_with("define ") {
                    break; // malformed; stop to avoid swallowing the next function
                }
                j += 1;
            }
            if let (Some(close), Some((name, ret_ty))) = (close, parse_define(lines[i])) {
                out.push(FuncSpan {
                    define_idx: i,
                    close_idx: close,
                    name,
                    ret_ty,
                });
                i = close + 1;
                continue;
            }
        }
        i += 1;
    }
    out
}

/// Extract `(function name, return type)` from a `define` line. Return type is what remains after
/// stripping leading attribute keywords and before `@name`.
fn parse_define(line: &str) -> Option<(String, String)> {
    let rest = line.strip_prefix("define ")?;
    let at = rest.find('@')?;
    let name = symbol_after_at(&rest[at..])?;
    let ret_seg = rest[..at].trim();
    let ret_ty = strip_leading_attrs(ret_seg);
    if ret_ty.is_empty() {
        return None;
    }
    Some((name, ret_ty.to_string()))
}

/// Return-position attribute keywords that can precede the return type on a `define` line.
const RET_ATTRS: &[&str] = &[
    "private",
    "internal",
    "available_externally",
    "linkonce",
    "linkonce_odr",
    "weak",
    "weak_odr",
    "common",
    "appending",
    "extern_weak",
    "external",
    "default",
    "hidden",
    "protected",
    "dllimport",
    "dllexport",
    "dso_local",
    "dso_preemptable",
    "local_unnamed_addr",
    "unnamed_addr",
    "spir_func",
    "spir_kernel",
    "fastcc",
    "coldcc",
    "signext",
    "zeroext",
    "noundef",
    "nonnull",
    "inreg",
];

fn strip_leading_attrs(seg: &str) -> &str {
    let mut s = seg.trim_start();
    loop {
        let tok = s.split_whitespace().next().unwrap_or("");
        if tok.is_empty() {
            return s;
        }
        // Calling conventions like `cc10`.
        let is_cc_num = tok.starts_with("cc") && tok[2..].chars().all(|c| c.is_ascii_digit());
        if RET_ATTRS.contains(&tok) || (is_cc_num && tok.len() > 2) {
            s = s[tok.len()..].trim_start();
        } else {
            return s;
        }
    }
}

#[derive(Clone, Default)]
struct LoopFacts {
    global_ints: HashMap<String, i128>,
    arg_ints: HashMap<String, i128>,
    arg_floats: HashMap<String, f64>,
    arg_field_ints: HashMap<(String, Vec<i32>), i128>,
    arg_byte_ints: HashMap<(String, usize), i128>,
    arg_upper_bounds: HashMap<String, i128>,
    arg_vector_ints: HashMap<(String, usize), i128>,
    arg_vector_upper_bounds: HashMap<(String, usize), i128>,
    texture_extents: HashMap<String, [i128; 3]>,
    imageblock_extent: Option<[i128; 2]>,
}

impl LoopFacts {
    fn from_module(lines: &[&str], input_facts: &LoopInputFacts<'_>) -> Self {
        let arg_ints = input_facts
            .arg_values
            .iter()
            .map(|(name, value)| (name.trim_start_matches('%').to_string(), *value))
            .collect::<HashMap<_, _>>();
        let arg_floats = input_facts
            .arg_float_values
            .iter()
            .map(|(name, value)| (name.trim_start_matches('%').to_string(), *value))
            .collect::<HashMap<_, _>>();
        let arg_upper_bounds = input_facts
            .arg_upper_bounds
            .iter()
            .map(|(name, value)| (name.trim_start_matches('%').to_string(), *value))
            .collect::<HashMap<_, _>>();
        let arg_field_ints = input_facts
            .arg_field_values
            .iter()
            .map(|(name, path, value)| {
                (
                    (name.trim_start_matches('%').to_string(), path.clone()),
                    *value,
                )
            })
            .collect::<HashMap<_, _>>();
        let arg_byte_ints = input_facts
            .arg_byte_values
            .iter()
            .map(|(name, offset, value)| {
                ((name.trim_start_matches('%').to_string(), *offset), *value)
            })
            .collect::<HashMap<_, _>>();
        let arg_vector_ints = input_facts
            .arg_vector_values
            .iter()
            .map(|(name, lane, value)| ((name.trim_start_matches('%').to_string(), *lane), *value))
            .collect::<HashMap<_, _>>();
        let arg_vector_upper_bounds = input_facts
            .arg_vector_upper_bounds
            .iter()
            .map(|(name, lane, value)| ((name.trim_start_matches('%').to_string(), *lane), *value))
            .collect::<HashMap<_, _>>();
        let texture_extents = input_facts
            .texture_extents
            .iter()
            .map(|(name, extent)| (name.trim_start_matches('%').to_string(), *extent))
            .collect::<HashMap<_, _>>();
        if input_facts.fc_values.is_empty() {
            let mut facts = Self {
                global_ints: HashMap::new(),
                arg_ints,
                arg_floats,
                arg_field_ints,
                arg_byte_ints,
                arg_upper_bounds,
                arg_vector_ints,
                arg_vector_upper_bounds,
                texture_extents,
                imageblock_extent: input_facts.imageblock_extent,
            };
            facts.propagate_direct_call_arg_facts(lines);
            return facts;
        }
        let fc_values = input_facts
            .fc_values
            .iter()
            .map(|(index, value)| (*index, i128::from(*value)))
            .collect::<HashMap<_, _>>();
        let mut fc_globals = HashMap::new();
        for line in lines {
            if !line.contains(".MTL_FC_INIT_") || !line.contains("section \"air.fc_initializer\"") {
                continue;
            }
            let Some(global) = global_definition_name(line) else {
                continue;
            };
            let Some(index) = fc_index_from_global(&global) else {
                continue;
            };
            if let Some(&value) = fc_values.get(&index) {
                fc_globals.insert(global, value);
            }
        }

        let mut global_ints = fc_globals.clone();
        let mut current_static_init = false;
        let mut regs = HashMap::new();
        for line in lines {
            if line.trim_start().starts_with("define ") {
                current_static_init = line.contains("section \"air.static_init\"");
                regs.clear();
                continue;
            }
            if current_static_init && line.trim_start() == "}" {
                current_static_init = false;
                regs.clear();
                continue;
            }
            if !current_static_init {
                continue;
            }
            if let Some((result, global)) = load_result_and_global(line) {
                if let Some(&value) = global_ints.get(&global) {
                    regs.insert(result, value);
                }
                continue;
            }
            if let Some(result) = result_name(line) {
                if let Some(value) = consteval_icmp_int(line, &regs)
                    .or_else(|| consteval_select_int(line, &regs))
                    .or_else(|| consteval_cast(line, &regs))
                    .or_else(|| consteval_binary(line, &regs))
                    .or_else(|| consteval_same_value_phi(line, &regs))
                {
                    regs.insert(result, value);
                    continue;
                }
            }
            if let Some((value, dest)) = store_value_and_dest_global(line) {
                if let Some(value) = value_as_const(&value, &regs) {
                    global_ints.insert(dest, value);
                }
            }
        }

        let mut facts = Self {
            global_ints,
            arg_ints,
            arg_floats,
            arg_field_ints,
            arg_byte_ints,
            arg_upper_bounds,
            arg_vector_ints,
            arg_vector_upper_bounds,
            texture_extents,
            imageblock_extent: input_facts.imageblock_extent,
        };
        facts.propagate_direct_call_arg_facts(lines);
        facts
    }

    fn is_empty(&self) -> bool {
        self.global_ints.is_empty()
            && self.arg_ints.is_empty()
            && self.arg_floats.is_empty()
            && self.arg_field_ints.is_empty()
            && self.arg_byte_ints.is_empty()
            && self.arg_upper_bounds.is_empty()
            && self.arg_vector_ints.is_empty()
            && self.arg_vector_upper_bounds.is_empty()
            && self.texture_extents.is_empty()
            && self.imageblock_extent.is_none()
    }

    fn field_value(&self, arg: &str, path: &[i32]) -> Option<i128> {
        self.arg_field_ints
            .get(&(arg.trim_start_matches('%').to_string(), path.to_vec()))
            .copied()
    }

    fn with_arg_upper_bounds(&self, arg_upper_bounds: HashMap<String, i128>) -> Self {
        Self {
            global_ints: self.global_ints.clone(),
            arg_ints: self.arg_ints.clone(),
            arg_floats: self.arg_floats.clone(),
            arg_field_ints: self.arg_field_ints.clone(),
            arg_byte_ints: self.arg_byte_ints.clone(),
            arg_upper_bounds,
            arg_vector_ints: self.arg_vector_ints.clone(),
            arg_vector_upper_bounds: self.arg_vector_upper_bounds.clone(),
            texture_extents: self.texture_extents.clone(),
            imageblock_extent: self.imageblock_extent,
        }
    }

    fn propagate_direct_call_arg_facts(&mut self, lines: &[&str]) {
        let funcs = find_functions(lines);
        if funcs.is_empty() {
            return;
        }
        let formals = funcs
            .iter()
            .map(|f| (f.name.as_str(), formal_arg_names(lines[f.define_idx])))
            .collect::<HashMap<_, _>>();
        let mut changed = true;
        while changed {
            changed = false;
            for caller in &funcs {
                let body = &lines[caller.define_idx + 1..caller.close_idx];
                for line in body.iter().copied().filter(|line| line.contains(" call ")) {
                    for callee in &funcs {
                        let Some(formals) = formals.get(callee.name.as_str()) else {
                            continue;
                        };
                        let Some(actuals) = call_args_for_callee(line, &callee.name) else {
                            continue;
                        };
                        for (index, formal) in formals.iter().enumerate() {
                            let Some(actual) = actuals.get(index) else {
                                continue;
                            };
                            if self.copy_arg_facts(body, actual, formal) {
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    }

    fn copy_arg_facts(&mut self, caller_body: &[&str], actual: &str, formal: &str) -> bool {
        let actual = actual.trim_start_matches('%');
        let formal = formal.trim_start_matches('%');
        if actual.is_empty() || formal.is_empty() {
            return false;
        }
        let mut changed = false;
        if let Ok(value) = actual.parse::<i128>() {
            changed |= insert_if_absent(&mut self.arg_ints, formal.to_string(), value);
            changed |= insert_if_absent(&mut self.arg_upper_bounds, formal.to_string(), value);
        }
        if let Some(&value) = self.arg_ints.get(actual) {
            changed |= insert_if_absent(&mut self.arg_ints, formal.to_string(), value);
        }
        if let Some(&value) = self.arg_upper_bounds.get(actual) {
            changed |= insert_if_absent(&mut self.arg_upper_bounds, formal.to_string(), value);
        }
        if let Some(&extent) = self.texture_extents.get(actual) {
            changed |= insert_if_absent(&mut self.texture_extents, formal.to_string(), extent);
        }
        let field_copies = self
            .arg_field_ints
            .iter()
            .filter(|((arg, _), _)| arg == actual)
            .map(|((_, path), &value)| ((formal.to_string(), path.clone()), value))
            .collect::<Vec<_>>();
        for (key, value) in field_copies {
            changed |= insert_if_absent(&mut self.arg_field_ints, key, value);
        }
        let vector_copies = self
            .arg_vector_ints
            .iter()
            .filter(|((arg, _), _)| arg == actual)
            .map(|((_, lane), &value)| ((formal.to_string(), *lane), value))
            .collect::<Vec<_>>();
        for (key, value) in vector_copies {
            changed |= insert_if_absent(&mut self.arg_vector_ints, key, value);
        }
        let vector_upper_copies = self
            .arg_vector_upper_bounds
            .iter()
            .filter(|((arg, _), _)| arg == actual)
            .map(|((_, lane), &value)| ((formal.to_string(), *lane), value))
            .collect::<Vec<_>>();
        for (key, value) in vector_upper_copies {
            changed |= insert_if_absent(&mut self.arg_vector_upper_bounds, key, value);
        }
        for lane in 0..4 {
            if self
                .arg_vector_upper_bounds
                .contains_key(&(formal.to_string(), lane))
            {
                continue;
            }
            if let Some(upper) = vector_lane_upper_bound(caller_body, self, actual, lane, 32) {
                changed |= insert_if_absent(
                    &mut self.arg_vector_upper_bounds,
                    (formal.to_string(), lane),
                    upper,
                );
            }
        }
        changed
    }
}

fn insert_if_absent<K, V>(map: &mut HashMap<K, V>, key: K, value: V) -> bool
where
    K: Eq + std::hash::Hash,
{
    if let std::collections::hash_map::Entry::Vacant(entry) = map.entry(key) {
        entry.insert(value);
        true
    } else {
        false
    }
}

fn formal_arg_names(define_line: &str) -> Vec<String> {
    let Some(args) = paren_contents_after_at(define_line, None) else {
        return Vec::new();
    };
    split_top_level_commas(args)
        .into_iter()
        .map(|arg| {
            arg.rfind('%')
                .and_then(|percent| read_value_name(&arg[percent + 1..]))
                .unwrap_or_default()
        })
        .collect()
}

fn global_definition_name(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix('@')?;
    let end = rest
        .find(|ch: char| ch == '=' || ch.is_whitespace())
        .unwrap_or(rest.len());
    Some(format!("@{}", &rest[..end]))
}

fn fc_index_from_global(global: &str) -> Option<usize> {
    let rest = global.split_once(".MTL_FC_INIT_")?.1;
    let digits = rest
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn load_result_and_global(line: &str) -> Option<(String, String)> {
    if !line.contains(" = load ") {
        return None;
    }
    let result = result_name(line)?;
    let global = global_name_after(line, "ptr addrspace(")
        .or_else(|| global_name_after(line, "addrspace("))?;
    Some((result, global))
}

fn load_result_and_pointer_arg(line: &str) -> Option<(String, String)> {
    if !line.contains(" = load ") {
        return None;
    }
    let result = result_name(line)?;
    let pointer = pointer_operand_after(line, "ptr addrspace(")
        .or_else(|| pointer_operand_after(line, "addrspace("))?;
    Some((result, pointer))
}

fn store_value_and_dest_global(line: &str) -> Option<(String, String)> {
    let store = line.trim_start().strip_prefix("store ")?;
    let (value_part, dest_part) = store.split_once(',')?;
    let stored_value = value_part.split_whitespace().last()?.to_string();
    let dest_global = global_name_after(dest_part, "ptr addrspace(")
        .or_else(|| global_name_after(dest_part, "addrspace("))?;
    Some((stored_value, dest_global))
}

fn store_dest_pointer(line: &str) -> Option<String> {
    let store = line.trim_start().strip_prefix("store ")?;
    let (_, dest_part) = store.split_once(',')?;
    pointer_operand_after(dest_part, "ptr addrspace(")
        .or_else(|| pointer_operand_after(dest_part, "addrspace("))
}

fn pointer_operand_after(line: &str, marker: &str) -> Option<String> {
    pointer_operand_and_tail_after(line, marker).map(|(name, _)| name)
}

fn pointer_operand_and_tail_after<'a>(line: &'a str, marker: &str) -> Option<(String, &'a str)> {
    let (_, tail) = line.split_once(marker)?;
    let percent = tail.find('%')?;
    let name = &tail[percent + 1..];
    let end = name
        .find(|ch: char| ch == ',' || ch.is_whitespace())
        .unwrap_or(name.len());
    Some((name[..end].to_string(), &name[end..]))
}

fn global_name_after(line: &str, marker: &str) -> Option<String> {
    let (_, tail) = line.split_once(marker)?;
    let at = tail.find('@')?;
    let global = &tail[at..];
    let end = global
        .find(|ch: char| ch == ',' || ch.is_whitespace())
        .unwrap_or(global.len());
    Some(global[..end].to_string())
}

fn symbol_after_at(s: &str) -> Option<String> {
    let rest = s.strip_prefix('@')?;
    if let Some(rest) = rest.strip_prefix('"') {
        let mut out = String::from("\"");
        let mut escaped = false;
        for ch in rest.chars() {
            out.push(ch);
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => return Some(out),
                _ => {}
            }
        }
        return None;
    }
    let name: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.' || *c == '$')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// Metal's textual IR uses opaque `ptr` in modern toolchains; detect it so injected loads/stores
/// use the matching pointer syntax (`ptr %p` vs `i32* %p`).
fn uses_opaque_pointers(module_text: &str) -> bool {
    if module_text.contains("ptr addrspace(")
        || module_text.contains("(ptr ")
        || module_text.contains(", ptr ")
        || module_text.contains(" ptr,")
        || module_text.contains(" ptr)")
        || module_text.contains(" ptr ")
    {
        return true;
    }
    if uses_typed_pointer_spelling(module_text) {
        return false;
    }
    module_text.contains("target triple = \"air64_v")
}

fn uses_typed_pointer_spelling(module_text: &str) -> bool {
    module_text.lines().any(|line| {
        let line = line.trim();
        !line.starts_with(';')
            && (line.contains(")*") || line.contains("* ") || line.contains("*,"))
    })
}

// --- per-function parse --------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum TermKind {
    Ret,
    Unreachable,
    Br,
    Switch,
}

struct Block {
    label: Option<String>,
    /// Body-relative inclusive line span (includes the label line, if any).
    start: usize,
    end: usize,
    /// Body-relative index of the terminator's first line.
    term_idx: usize,
    kind: TermKind,
    succ: Vec<String>,
}

struct ParsedFunc {
    name: String,
    ret_ty: String,
    blocks: Vec<Block>,
    /// (source block index, target block index) back-edges, over a DFS of *all* blocks.
    back_edges: Vec<(usize, usize)>,
    /// True when a natural loop body contains a workgroup barrier call.
    loop_has_workgroup_barrier: bool,
    /// Directly-called function symbols anywhere in this function body.
    calls: Vec<String>,
    /// Direct call sites inside remaining loop bodies.
    loop_calls: Vec<CallSite>,
}

#[derive(Clone)]
struct CallSite {
    callee: String,
    line: String,
}

fn parse_func(lines: &[&str], f: &FuncSpan, facts: &LoopFacts) -> Result<ParsedFunc, String> {
    parse_func_with_loop_elision(lines, f, facts, true)
}

fn parse_func_preserving_loops(
    lines: &[&str],
    f: &FuncSpan,
    facts: &LoopFacts,
) -> Result<ParsedFunc, String> {
    parse_func_with_loop_elision(lines, f, facts, false)
}

fn parse_func_with_loop_elision(
    lines: &[&str],
    f: &FuncSpan,
    facts: &LoopFacts,
    elide_small_loops: bool,
) -> Result<ParsedFunc, String> {
    let body = &lines[f.define_idx + 1..f.close_idx];
    let blocks = parse_blocks(body, &f.name)?;

    let mut label_to_idx: HashMap<&str, usize> = HashMap::new();
    for (i, b) in blocks.iter().enumerate() {
        if let Some(label) = &b.label {
            if label_to_idx.insert(label.as_str(), i).is_some() {
                return Err(format!("{:?}: duplicate block label {label:?}", f.name));
            }
        }
    }

    // Successor indices; a target that names no known block is malformed control flow.
    let mut succ_idx: Vec<Vec<usize>> = Vec::with_capacity(blocks.len());
    for b in &blocks {
        let mut v = Vec::with_capacity(b.succ.len());
        for s in &b.succ {
            match label_to_idx.get(s.as_str()) {
                Some(&idx) => v.push(idx),
                None => return Err(format!("{:?}: branch to unknown block {s:?}", f.name)),
            }
        }
        succ_idx.push(v);
    }

    let reachable = reachable_blocks_under_facts(body, &blocks, facts);
    let back_edges = back_edges(&succ_idx)
        .into_iter()
        .filter(|&(src, dst)| reachable[src] && reachable[dst])
        .filter(|&(src, dst)| {
            !elide_small_loops || !small_fixed_trip_loop(body, &blocks, facts, src, dst)
        })
        .collect::<Vec<_>>();
    let loop_has_workgroup_barrier = back_edges.iter().any(|&(src, dst)| {
        natural_loop_nodes(&succ_idx, src, dst)
            .into_iter()
            .any(|idx| block_contains_workgroup_barrier(body, &blocks[idx]))
    });
    let loop_calls = calls_in_remaining_loops(body, &blocks, &succ_idx, &back_edges);

    let mut calls = Vec::new();
    for (idx, block) in blocks.iter().enumerate() {
        if !reachable[idx] {
            continue;
        }
        for line in block_lines(body, block) {
            collect_calls(line, &mut calls);
        }
    }

    Ok(ParsedFunc {
        name: f.name.clone(),
        ret_ty: f.ret_ty.clone(),
        blocks,
        back_edges,
        loop_has_workgroup_barrier,
        calls,
        loop_calls,
    })
}

fn calls_in_remaining_loops(
    body: &[&str],
    blocks: &[Block],
    succ_idx: &[Vec<usize>],
    back_edges: &[(usize, usize)],
) -> Vec<CallSite> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for &(src, dst) in back_edges {
        for idx in natural_loop_nodes(succ_idx, src, dst) {
            for line in block_lines(body, &blocks[idx]) {
                let mut line_calls = Vec::new();
                collect_calls(line, &mut line_calls);
                for call in line_calls {
                    if seen.insert((call.clone(), line.to_string())) {
                        out.push(CallSite {
                            callee: call,
                            line: line.to_string(),
                        });
                    }
                }
            }
        }
    }
    out
}

fn parse_blocks(body: &[&str], fname: &str) -> Result<Vec<Block>, String> {
    let mut blocks: Vec<Block> = Vec::new();
    let mut cur_label: Option<String> = None;
    let mut cur_start = 0usize;
    let mut i = 0usize;
    let mut pending = false; // have we seen any line for the current block?

    while i < body.len() {
        let line = body[i];
        if let Some(label) = parse_label(line) {
            if pending {
                blocks.push(finish_block(
                    body,
                    cur_label.take(),
                    cur_start,
                    i - 1,
                    fname,
                )?);
            }
            cur_label = Some(label);
            cur_start = i;
            pending = true;
            i += 1;
            continue;
        }
        pending = true;
        i += 1;
    }
    if pending {
        blocks.push(finish_block(
            body,
            cur_label.take(),
            cur_start,
            body.len() - 1,
            fname,
        )?);
    }
    if blocks.is_empty() {
        return Err(format!("{fname:?}: empty function body"));
    }
    Ok(blocks)
}

/// Build a [`Block`] from its line span, locating and classifying the terminator.
fn finish_block(
    body: &[&str],
    label: Option<String>,
    start: usize,
    end: usize,
    fname: &str,
) -> Result<Block, String> {
    // Find the terminator instruction within [start, end]. A well-formed block has exactly one and
    // it is last; scan forward and keep the last recognised terminator.
    let mut term: Option<(usize, TermKind, Vec<String>)> = None;
    let mut k = start;
    while k <= end {
        let t = body[k].trim_start();
        let kind = if t.starts_with("ret ") || t == "ret" || t == "ret void" {
            Some(TermKind::Ret)
        } else if t == "unreachable" || t.starts_with("unreachable ") {
            Some(TermKind::Unreachable)
        } else if t.starts_with("br ") {
            Some(TermKind::Br)
        } else if t.starts_with("switch ") {
            Some(TermKind::Switch)
        } else if t.starts_with("indirectbr") || t.starts_with("callbr") || t.starts_with("invoke ")
        {
            return Err(format!(
                "{fname:?}: unsupported terminator (indirectbr/callbr/invoke)"
            ));
        } else {
            None
        };
        if let Some(kind) = kind {
            let (succ, last) = match kind {
                TermKind::Ret | TermKind::Unreachable => (Vec::new(), k),
                TermKind::Br => (extract_label_targets(body[k]), k),
                TermKind::Switch => {
                    // A switch spans lines until the closing `]`.
                    let mut last = k;
                    while last <= end && !body[last].contains(']') {
                        last += 1;
                    }
                    if last > end {
                        return Err(format!("{fname:?}: unterminated switch"));
                    }
                    let mut succ = Vec::new();
                    for row in &body[k..=last] {
                        succ.extend(extract_label_targets(row));
                    }
                    (succ, last)
                }
            };
            term = Some((k, kind, succ));
            k = last + 1;
            continue;
        }
        k += 1;
    }
    let (term_idx, kind, succ) =
        term.ok_or_else(|| format!("{fname:?}: block with no terminator"))?;
    Ok(Block {
        label,
        start,
        end,
        term_idx,
        kind,
        succ,
    })
}

/// A block label line: column 0, ends with `:` (before any comment). Handles quoted labels.
fn parse_label(line: &str) -> Option<String> {
    if line.starts_with([' ', '\t', ';']) || line.is_empty() {
        return None;
    }
    if let Some(rest) = line.strip_prefix('"') {
        // "quoted":  → label is `"quoted"`.
        let mut escaped = false;
        for (i, ch) in rest.char_indices() {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => {
                    let after = rest[i + 1..].trim_start();
                    return after
                        .starts_with(':')
                        .then(|| format!("\"{}\"", &rest[..i]));
                }
                _ => {}
            }
        }
        return None;
    }
    let colon = line.find(':')?;
    let name = line[..colon].trim_end();
    if name.is_empty() || name.contains(char::is_whitespace) {
        return None;
    }
    // Guard against `define`/globals slipping through (they never appear inside a body, but be safe).
    if name.starts_with('@') || name.starts_with('!') {
        return None;
    }
    Some(name.to_string())
}

/// All `label %<name>` targets on a line (branch/switch destinations).
fn extract_label_targets(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(pos) = rest.find("label %") {
        let after = &rest[pos + "label %".len()..];
        if let Some(name) = read_value_name(after) {
            out.push(name);
        }
        rest = after;
    }
    out
}

/// Read an SSA/label name starting right after a `%` (quoted or bare).
fn read_value_name(s: &str) -> Option<String> {
    if let Some(rest) = s.strip_prefix('"') {
        let mut escaped = false;
        for (i, ch) in rest.char_indices() {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => return Some(format!("\"{}\"", &rest[..i])),
                _ => {}
            }
        }
        return None;
    }
    let name: String = s
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.' || *c == '$')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// Collect directly-called function symbols from an instruction line.
fn collect_calls(line: &str, out: &mut Vec<String>) {
    let t = line.trim_start();
    if !(t.contains("call ") || t.starts_with("call ")) {
        return;
    }
    // Callee is the `@sym` immediately followed by `(`.
    let mut rest = line;
    while let Some(pos) = rest.find('@') {
        let after = &rest[pos..];
        if let Some(name) = symbol_after_at(after) {
            let tail = &after[1 + raw_sym_len(&after[1..])..];
            if tail.starts_with('(') && !out.iter().any(|s| s == &name) {
                out.push(name);
            }
        }
        rest = &rest[pos + 1..];
    }
}

/// Length of the raw symbol text after `@` (including quotes), to find what follows it.
fn raw_sym_len(s: &str) -> usize {
    if let Some(rest) = s.strip_prefix('"') {
        let mut escaped = false;
        for (i, ch) in rest.char_indices() {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => return i + 2, // opening + closing quote
                _ => {}
            }
        }
        return s.len();
    }
    s.chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.' || *c == '$')
        .count()
}

/// Back-edges over a DFS forest covering *all* blocks (so loops in blocks not reachable from the
/// entry — e.g. a disconnected component the disassembler kept — are still bounded). Any cycle has
/// at least one back-edge in any DFS, so guarding all of them makes every cycle bounded.
fn back_edges(succ: &[Vec<usize>]) -> Vec<(usize, usize)> {
    #[derive(Clone, Copy, PartialEq)]
    enum Color {
        White,
        Grey,
        Black,
    }
    let n = succ.len();
    let mut color = vec![Color::White; n];
    let mut edges = Vec::new();
    // Iterative DFS; stack holds (node, next-successor-cursor).
    for root in 0..n {
        if color[root] != Color::White {
            continue;
        }
        let mut stack: Vec<(usize, usize)> = vec![(root, 0)];
        color[root] = Color::Grey;
        while let Some(&mut (u, ref mut ci)) = stack.last_mut() {
            if *ci < succ[u].len() {
                let v = succ[u][*ci];
                *ci += 1;
                match color[v] {
                    Color::White => {
                        color[v] = Color::Grey;
                        stack.push((v, 0));
                    }
                    Color::Grey => edges.push((u, v)),
                    Color::Black => {}
                }
            } else {
                color[u] = Color::Black;
                stack.pop();
            }
        }
    }
    edges
}

fn natural_loop_nodes(succ: &[Vec<usize>], src: usize, dst: usize) -> Vec<usize> {
    let mut pred = vec![Vec::<usize>::new(); succ.len()];
    for (from, targets) in succ.iter().enumerate() {
        for &to in targets {
            pred[to].push(from);
        }
    }

    let mut seen = vec![false; succ.len()];
    let mut stack = vec![src];
    seen[dst] = true;
    while let Some(node) = stack.pop() {
        if seen[node] {
            continue;
        }
        seen[node] = true;
        for &p in &pred[node] {
            stack.push(p);
        }
    }
    seen.iter()
        .enumerate()
        .filter_map(|(idx, &is_loop)| is_loop.then_some(idx))
        .collect()
}

fn block_contains_workgroup_barrier(body: &[&str], block: &Block) -> bool {
    body[block.start..=block.end]
        .iter()
        .any(|line| line_calls_workgroup_barrier(line))
}

fn line_calls_workgroup_barrier(line: &str) -> bool {
    line.contains("@air.wg.barrier(") || line.contains("@\"air.wg.barrier\"(")
}

fn transitive_loopy<'a>(
    parsed: &'a [ParsedFunc],
    direct: &HashMap<&'a str, bool>,
) -> HashMap<&'a str, bool> {
    let names: HashSet<&str> = parsed.iter().map(|f| f.name.as_str()).collect();
    let mut loopy: HashMap<&str, bool> = direct.clone();
    // Fixpoint: a function is loopy if it loops directly or calls a loopy (known) function.
    loop {
        let mut changed = false;
        for f in parsed {
            if *loopy.get(f.name.as_str()).unwrap_or(&false) {
                continue;
            }
            if f.calls
                .iter()
                .any(|c| names.contains(c.as_str()) && *loopy.get(c.as_str()).unwrap_or(&false))
            {
                loopy.insert(f.name.as_str(), true);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    loopy
}

fn loop_call_callee_is_bounded_with_call_facts(
    lines: &[&str],
    funcs: &HashMap<&str, &FuncSpan>,
    base_facts: &LoopFacts,
    trans_loopy: &HashMap<&str, bool>,
    caller: &ParsedFunc,
    call: &CallSite,
) -> bool {
    let mut stack = Vec::new();
    loop_call_callee_is_bounded_recursive(
        lines,
        funcs,
        base_facts,
        trans_loopy,
        caller,
        call,
        &mut stack,
    )
}

fn loop_call_callee_is_bounded_recursive(
    lines: &[&str],
    funcs: &HashMap<&str, &FuncSpan>,
    base_facts: &LoopFacts,
    trans_loopy: &HashMap<&str, bool>,
    caller: &ParsedFunc,
    call: &CallSite,
    stack: &mut Vec<String>,
) -> bool {
    let Some(callee) = funcs.get(call.callee.as_str()) else {
        return false;
    };
    let Some(caller_span) = funcs.get(caller.name.as_str()) else {
        return false;
    };
    if stack.iter().any(|name| name == &callee.name) {
        return false;
    }
    let Some(call_facts) = facts_for_bounded_call(lines, base_facts, caller_span, callee, call)
    else {
        return false;
    };
    let Ok(parsed) = parse_func(lines, callee, &call_facts) else {
        return false;
    };
    if !parsed.back_edges.is_empty() {
        return false;
    }

    stack.push(callee.name.clone());
    let callee_body = &lines[callee.define_idx + 1..callee.close_idx];
    for nested in call_sites_in_body(callee_body) {
        if !funcs.contains_key(nested.callee.as_str())
            || !*trans_loopy.get(nested.callee.as_str()).unwrap_or(&false)
        {
            continue;
        }
        if !loop_call_callee_is_bounded_recursive(
            lines,
            funcs,
            &call_facts,
            trans_loopy,
            &parsed,
            &nested,
            stack,
        ) {
            stack.pop();
            return false;
        }
    }
    stack.pop();
    true
}

fn facts_for_bounded_call(
    lines: &[&str],
    base_facts: &LoopFacts,
    caller_span: &FuncSpan,
    callee: &FuncSpan,
    call: &CallSite,
) -> Option<LoopFacts> {
    let formals = integer_formals(lines[callee.define_idx]);
    if formals.is_empty() {
        return Some(base_facts.clone());
    }
    let actuals = call_args_for_callee(&call.line, &call.callee)?;
    let caller_body = &lines[caller_span.define_idx + 1..caller_span.close_idx];
    let mut arg_upper_bounds = base_facts.arg_upper_bounds.clone();
    let mut bounded_any_formal = false;
    for formal in formals {
        let Some(actual) = actuals.get(formal.index) else {
            continue;
        };
        let Some(upper) =
            small_integer_upper_bound_with_facts(caller_body, base_facts, actual, formal.bit_width)
        else {
            continue;
        };
        if !(0..=256).contains(&upper) {
            continue;
        }
        arg_upper_bounds.insert(formal.name, upper);
        bounded_any_formal = true;
    }
    bounded_any_formal.then(|| base_facts.with_arg_upper_bounds(arg_upper_bounds))
}

struct IntegerFormal {
    index: usize,
    name: String,
    bit_width: u32,
}

fn integer_formals(define_line: &str) -> Vec<IntegerFormal> {
    let Some(args) = paren_contents_after_at(define_line, None) else {
        return Vec::new();
    };
    split_top_level_commas(args)
        .into_iter()
        .enumerate()
        .filter_map(|(index, arg)| {
            let ty = arg.split_whitespace().find_map(int_type_width)?;
            let percent = arg.rfind('%')?;
            let name = read_value_name(&arg[percent + 1..])?;
            Some(IntegerFormal {
                index,
                name,
                bit_width: ty,
            })
        })
        .collect()
}

fn call_args_for_callee(line: &str, callee: &str) -> Option<Vec<String>> {
    let args = paren_contents_after_at(line, Some(callee))?;
    Some(
        split_top_level_commas(args)
            .into_iter()
            .map(|arg| arg_value_name_or_const(arg).unwrap_or_default())
            .collect(),
    )
}

fn call_sites_in_body(body: &[&str]) -> Vec<CallSite> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for line in body {
        let mut line_calls = Vec::new();
        collect_calls(line, &mut line_calls);
        for call in line_calls {
            if seen.insert((call.clone(), (*line).to_string())) {
                out.push(CallSite {
                    callee: call,
                    line: (*line).to_string(),
                });
            }
        }
    }
    out
}

fn paren_contents_after_at<'a>(line: &'a str, symbol: Option<&str>) -> Option<&'a str> {
    let mut rest = line;
    while let Some(pos) = rest.find('@') {
        let after = &rest[pos..];
        if let Some(name) = symbol_after_at(after) {
            let raw_len = raw_sym_len(&after[1..]);
            let tail = &after[1 + raw_len..];
            if symbol.is_none_or(|wanted| wanted == name) && tail.starts_with('(') {
                return balanced_paren_contents(tail);
            }
        }
        rest = &rest[pos + 1..];
    }
    None
}

fn balanced_paren_contents(s: &str) -> Option<&str> {
    let mut depth = 0usize;
    let mut start = None;
    for (idx, ch) in s.char_indices() {
        match ch {
            '(' => {
                if depth == 0 {
                    start = Some(idx + ch.len_utf8());
                }
                depth += 1;
            }
            ')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(&s[start?..idx]);
                }
            }
            _ => {}
        }
    }
    None
}

fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut paren = 0usize;
    let mut bracket = 0usize;
    let mut angle = 0usize;
    for (idx, ch) in s.char_indices() {
        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            '<' => angle += 1,
            '>' => angle = angle.saturating_sub(1),
            ',' if paren == 0 && bracket == 0 && angle == 0 => {
                out.push(s[start..idx].trim());
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }
    let tail = s[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

fn arg_value_name_or_const(arg: &str) -> Option<String> {
    if let Some(percent) = arg.rfind('%') {
        return read_value_name(&arg[percent + 1..]);
    }
    arg.split_whitespace()
        .rev()
        .map(|part| part.trim_end_matches(','))
        .find(|part| part.parse::<i128>().is_ok())
        .map(ToOwned::to_owned)
}

// --- instrumentation -----------------------------------------------------------------------------

/// Transform one loopy function: add a per-thread budget in the entry block, split every back-edge
/// through a decrement-and-check guard, and append a single exit block. Returns the whole function
/// (`define … { … }` inclusive) as lines. Errors → the caller quarantines the case.
fn transform_func(
    lines: &[&str],
    f: &FuncSpan,
    pf: &ParsedFunc,
    opaque_ptr: bool,
    loop_budget_backedges: i32,
) -> Result<Vec<String>, String> {
    let body_src = &lines[f.define_idx + 1..f.close_idx];
    let mut body: Vec<String> = body_src.iter().map(|s| s.to_string()).collect();

    let ptr = if opaque_ptr { "ptr" } else { "i32*" };
    let budget = "%m2v.bd";

    // 1) In-place rewrites (before any insertion, so body indices stay valid).
    let mut guards_after: HashMap<usize, Vec<String>> = HashMap::new();
    let mut budget_before: HashMap<usize, Vec<String>> = HashMap::new();
    let mut metadata_rewritten_sources = HashSet::new();
    for (k, &(u, v)) in pf.back_edges.iter().enumerate() {
        let ub = &pf.blocks[u];
        let vb = &pf.blocks[v];
        if ub.kind != TermKind::Br {
            return Err(format!(
                "{:?}: back-edge from non-branch terminator (switch/other) — not split",
                pf.name
            ));
        }
        let v_label = vb
            .label
            .as_deref()
            .ok_or_else(|| format!("{:?}: back-edge into entry block", pf.name))?;
        let guard_label = format!("m2v.g.{k}");

        // Redirect u's back-edge target to the guard (exactly one occurrence expected).
        let term = &body[ub.term_idx];
        let (term_without_loop_metadata, loop_metadata) = split_loop_metadata(term);
        if !loop_metadata.is_empty() {
            if !metadata_rewritten_sources.insert(u) {
                return Err(format!(
                    "{:?}: multiple metadata back-edges from one branch terminator are not split",
                    pf.name
                ));
            }
            let targets = extract_label_targets(&term_without_loop_metadata);
            match targets.as_slice() {
                [only] if only == v_label => {
                    budget_before.entry(ub.term_idx).or_default().extend([
                        format!("  %m2v.{k}.a = load i32, {ptr} {budget}, align 4"),
                        format!("  %m2v.{k}.b = sub i32 %m2v.{k}.a, 1"),
                        format!("  store i32 %m2v.{k}.b, {ptr} {budget}, align 4"),
                        format!("  %m2v.{k}.ex = icmp sle i32 %m2v.{k}.b, 0"),
                    ]);
                    body[ub.term_idx] = format!(
                        "  br i1 %m2v.{k}.ex, label %m2v.exit, label %{v_label}{loop_metadata}"
                    );
                    continue;
                }
                [true_target, false_target] => {
                    let cond =
                        branch_condition_operand(&term_without_loop_metadata).ok_or_else(|| {
                            format!(
                                "{:?}: conditional back-edge branch has no condition",
                                pf.name
                            )
                        })?;
                    if true_target == v_label {
                        budget_before.entry(ub.term_idx).or_default().extend([
                            format!("  %m2v.{k}.a = load i32, {ptr} {budget}, align 4"),
                            format!("  %m2v.{k}.b = sub i32 %m2v.{k}.a, 1"),
                            format!("  store i32 %m2v.{k}.b, {ptr} {budget}, align 4"),
                            format!("  %m2v.{k}.ok = icmp sgt i32 %m2v.{k}.b, 0"),
                            format!("  %m2v.{k}.keep = and i1 {cond}, %m2v.{k}.ok"),
                        ]);
                        body[ub.term_idx] = format!(
                            "  br i1 %m2v.{k}.keep, label %{true_target}, label %{false_target}{loop_metadata}"
                        );
                        continue;
                    }
                    if false_target == v_label {
                        budget_before.entry(ub.term_idx).or_default().extend([
                            format!("  %m2v.{k}.a = load i32, {ptr} {budget}, align 4"),
                            format!("  %m2v.{k}.b = sub i32 %m2v.{k}.a, 1"),
                            format!("  store i32 %m2v.{k}.b, {ptr} {budget}, align 4"),
                            format!("  %m2v.{k}.ex = icmp sle i32 %m2v.{k}.b, 0"),
                            format!("  %m2v.{k}.leave = or i1 {cond}, %m2v.{k}.ex"),
                        ]);
                        body[ub.term_idx] = format!(
                            "  br i1 %m2v.{k}.leave, label %{true_target}, label %{false_target}{loop_metadata}"
                        );
                        continue;
                    }
                    return Err(format!(
                        "{:?}: loop metadata branch does not target loop header {v_label:?}",
                        pf.name
                    ));
                }
                _ => {
                    return Err(format!(
                        "{:?}: unsupported loop metadata branch shape",
                        pf.name
                    ));
                }
            }
        }
        let rewritten =
            replace_branch_target(&term_without_loop_metadata, v_label, &guard_label)
                .ok_or_else(|| format!("{:?}: ambiguous back-edge target {v_label:?}", pf.name))?;
        body[ub.term_idx] = rewritten;

        // Rename v's phi predecessor u → guard (u's label, since v's phis name their predecessors).
        if let Some(u_label) = ub.label.as_deref() {
            for line in &mut body[vb.start..=vb.end] {
                if line.contains(" = phi ") || line.trim_start().starts_with("phi ") {
                    *line = rename_phi_pred(line, u_label, &guard_label);
                }
            }
        }

        guards_after.entry(ub.end).or_default().extend([
            format!("{guard_label}:"),
            format!("  %m2v.{k}.a = load i32, {ptr} {budget}, align 4"),
            format!("  %m2v.{k}.b = sub i32 %m2v.{k}.a, 1"),
            format!("  store i32 %m2v.{k}.b, {ptr} {budget}, align 4"),
            format!("  %m2v.{k}.c = icmp sle i32 %m2v.{k}.b, 0"),
            format!("  br i1 %m2v.{k}.c, label %m2v.exit, label %{v_label}{loop_metadata}"),
        ]);
    }

    // 2) Prepend the budget alloca + init to the entry block (index 0), and emit each guard block
    // immediately after the latch block whose back-edge it split. Keeping the synthetic back-edge
    // block next to the source loop makes the downstream native structurizer see an ordinary loop
    // instead of a post-return block that branches back into earlier code.
    let entry = &pf.blocks[0];
    let insert_at = if entry.label.is_some() {
        entry.start + 1
    } else {
        entry.start
    };
    let init = vec![
        format!("  {budget} = alloca i32, align 4"),
        format!("  store i32 {loop_budget_backedges}, {ptr} {budget}, align 4"),
    ];
    let mut emitted_body = Vec::with_capacity(
        body.len() + init.len() + guards_after.len() * 6 + budget_before.len() * 5 + 2,
    );
    for (idx, line) in body.into_iter().enumerate() {
        if idx == insert_at {
            emitted_body.extend(init.iter().cloned());
        }
        if let Some(budget_lines) = budget_before.remove(&idx) {
            emitted_body.extend(budget_lines);
        }
        emitted_body.push(line);
        if let Some(guard_lines) = guards_after.remove(&idx) {
            emitted_body.extend(guard_lines);
        }
    }
    if insert_at == body_src.len() {
        emitted_body.extend(init);
    }
    if !guards_after.is_empty() {
        return Err(format!("{:?}: guard insertion point went stale", pf.name));
    }
    if !budget_before.is_empty() {
        return Err(format!("{:?}: budget insertion point went stale", pf.name));
    }

    // 3) Append the single budget-exhausted exit block.
    emitted_body.push("m2v.exit:".to_string());
    if pf.ret_ty == "void" {
        emitted_body.push("  ret void".to_string());
    } else {
        // `undef` is a valid value of any type; the exit is only reached on a runaway.
        emitted_body.push(format!("  ret {} undef", pf.ret_ty));
    }

    // 4) Reassemble the function.
    let mut out = Vec::with_capacity(emitted_body.len() + 2);
    out.push(lines[f.define_idx].to_string());
    out.extend(emitted_body);
    out.push(lines[f.close_idx].to_string());
    Ok(out)
}

fn small_fixed_trip_loop(
    body: &[&str],
    blocks: &[Block],
    facts: &LoopFacts,
    src: usize,
    dst: usize,
) -> bool {
    let bool_toggle = bool_toggle_loop(body, blocks, src, dst);
    let const_loop = counted_const_loop(body, blocks, src, dst);
    let symbolic = counted_small_symbolic_loop(body, blocks, facts, src, dst);
    let consteval = counted_consteval_loop(body, blocks, facts, src, dst);
    let min_chunk = bounded_min_chunk_descending_loop(body, blocks, facts, src, dst);
    bool_toggle || const_loop || symbolic || consteval || min_chunk
}

/// Recognize `remaining -= min(remaining, positive_chunk)` loops. This is a common AIR memcpy /
/// tiled-copy shape. With a concrete small initial value and a positive chunk it reaches zero in a
/// bounded number of iterations without needing validation-only IR instrumentation.
fn bounded_min_chunk_descending_loop(
    body: &[&str],
    blocks: &[Block],
    facts: &LoopFacts,
    src: usize,
    dst: usize,
) -> bool {
    let header = &blocks[dst];
    let latch = &blocks[src];
    let Some(src_label) = latch.label.as_deref() else {
        return false;
    };
    let Some(dst_label) = header.label.as_deref() else {
        return false;
    };
    let latch_lines = block_lines(body, latch).collect::<Vec<_>>();
    let Some(branch) = latch_lines
        .iter()
        .rev()
        .copied()
        .find(|line| line.trim_start().starts_with("br i1 "))
    else {
        return false;
    };
    let Some(backedge_on_true) = branch_backedge_on_true(branch, dst_label) else {
        return false;
    };
    let Some(condition) = branch_condition(branch) else {
        return false;
    };
    let Some(condition_line) = latch_lines
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(condition.as_str()))
    else {
        return false;
    };
    let Some((predicate, condition_lhs, condition_rhs)) = parse_icmp_value_operands(condition_line)
    else {
        return false;
    };
    let continues_while_nonzero =
        (predicate == "ne" && backedge_on_true) || (predicate == "eq" && !backedge_on_true);
    if !continues_while_nonzero {
        return false;
    }

    let lines = body.to_vec();
    for ty in ["i32", "i64"] {
        let bit_width = ty.trim_start_matches('i').parse::<u32>().unwrap_or(64);
        for phi_line in
            block_lines(body, header).filter(|line| line.contains(&format!(" = phi {ty} ")))
        {
            let Some(phi) = parse_integer_phi(phi_line, src_label, ty) else {
                continue;
            };
            let condition_lhs = condition_lhs.trim_start_matches('%');
            let condition_rhs = condition_rhs.trim_start_matches('%');
            if !((condition_lhs == phi.recur && condition_rhs == "0")
                || (condition_rhs == phi.recur && condition_lhs == "0"))
            {
                continue;
            }
            let Some(recur_line) = lines
                .iter()
                .copied()
                .find(|line| result_name(line).as_deref() == Some(phi.recur.as_str()))
            else {
                continue;
            };
            let Some((_, sub_rhs)) = recur_line.split_once(" = sub ") else {
                continue;
            };
            let operands = sub_rhs
                .split([',', ' '])
                .filter(|token| !token.is_empty())
                .collect::<Vec<_>>();
            let Some(type_pos) = operands.iter().position(|token| *token == ty) else {
                continue;
            };
            let Some(base) = operands
                .get(type_pos + 1)
                .map(|token| token.trim_start_matches('%'))
            else {
                continue;
            };
            let Some(chunk) = operands
                .get(type_pos + 2)
                .map(|token| token.trim_start_matches('%'))
            else {
                continue;
            };
            if base != phi.name {
                continue;
            }
            let Some(chunk_line) = lines
                .iter()
                .copied()
                .find(|line| result_name(line).as_deref() == Some(chunk))
            else {
                continue;
            };
            if !chunk_line.contains("@air.min.") {
                continue;
            }
            let Some(args) = chunk_line
                .split_once(" = ")
                .and_then(|(_, rhs)| call_arg_tokens(rhs))
            else {
                continue;
            };
            if args.len() < 2 {
                continue;
            }
            let lhs = args[0].trim_start_matches('%');
            let rhs = args[1].trim_start_matches('%');
            let bound = if lhs == phi.name {
                rhs
            } else if rhs == phi.name {
                lhs
            } else {
                continue;
            };
            let Some(initial) =
                small_integer_upper_bound_with_facts(&lines, facts, &phi.init, bit_width)
            else {
                continue;
            };
            let Some(chunk) =
                exact_positive_value(&lines, facts, bound, bit_width, &mut Vec::new())
            else {
                continue;
            };
            if initial < 0 || chunk <= 0 {
                continue;
            }
            let trips = (initial.max(1) + chunk - 1) / chunk;
            if (1..=256).contains(&trips) {
                return true;
            }
        }
    }
    false
}

fn bool_toggle_loop(body: &[&str], blocks: &[Block], src: usize, dst: usize) -> bool {
    let header = &blocks[dst];
    let Some(src_label) = blocks[src].label.as_deref() else {
        return false;
    };
    block_lines(body, header).any(|line| {
        line.contains(" = phi i1 ")
            && line.contains("[ true,")
            && line.contains("[ false,")
            && line.contains(&format!("%{src_label}"))
    })
}

fn counted_const_loop(body: &[&str], blocks: &[Block], src: usize, dst: usize) -> bool {
    let header = &blocks[dst];
    let latch = &blocks[src];
    let Some(src_label) = latch.label.as_deref() else {
        return false;
    };
    let has_zero_counter = block_lines(body, header).any(|line| {
        (line.contains(" = phi i32 ") || line.contains(" = phi i64 "))
            && line.contains("[ 0,")
            && line.contains(&format!("%{src_label}"))
    });
    if !has_zero_counter {
        return false;
    }
    block_lines(body, latch).any(|line| {
        (line.contains(" = icmp eq i32 ") || line.contains(" = icmp eq i64 "))
            && trailing_small_const(line).is_some_and(|trip| (1..=256).contains(&trip))
    })
}

fn counted_small_symbolic_loop(
    body: &[&str],
    blocks: &[Block],
    facts: &LoopFacts,
    src: usize,
    dst: usize,
) -> bool {
    let header = &blocks[dst];
    let latch = &blocks[src];
    let Some(src_label) = latch.label.as_deref() else {
        return false;
    };
    let Some(dst_label) = header.label.as_deref() else {
        return false;
    };
    let loop_lines = block_lines(body, header)
        .chain(block_lines(body, latch))
        .collect::<Vec<_>>();
    let def_lines = body.to_vec();
    let latch_lines = block_lines(body, latch).collect::<Vec<_>>();
    let Some(branch) = latch_lines
        .iter()
        .rev()
        .copied()
        .find(|line| line.trim_start().starts_with("br i1 "))
    else {
        return false;
    };
    let Some(cond) = branch_condition(branch) else {
        return false;
    };
    let backedge_on_true = branch_backedge_on_true(branch, dst_label);
    let conds =
        symbolic_loop_condition_icmps(&loop_lines, &cond, backedge_on_true, !facts.is_empty());
    for line in block_lines(body, header) {
        let Some((phi, ty)) = parse_integer_phi_any(line, src_label) else {
            continue;
        };
        if selected_latch_decrement_loop(body, blocks, latch, &phi, ty.bit_width, branch, &cond) {
            return true;
        }
        if conds.is_empty() {
            continue;
        }
        let values = if facts.is_empty() {
            HashMap::new()
        } else {
            consteval_int_values(&def_lines, facts)
        };
        for cond in &conds {
            let Some((pred, lhs, rhs)) = parse_icmp_value_operands(cond.line) else {
                continue;
            };
            if expression_recur_trip_count_from_values(
                &def_lines,
                &values,
                &phi,
                (pred, &lhs, &rhs),
                backedge_on_true,
            )
            .is_some_and(|trips| (1..=256).contains(&trips))
            {
                return true;
            }
        }
        let Some((step_base, step)) = add_step(&def_lines, &phi.recur)
            .or_else(|| add_step_with_values(&def_lines, &values, &phi.recur))
        else {
            continue;
        };
        if step_base != phi.name || step == 0 {
            continue;
        }
        let has_small_step = step.unsigned_abs() <= 32;
        for cond in &conds {
            let (pred, lhs, rhs, cond_line) = (
                cond.pred.as_str(),
                loop_step_icmp_operand(&def_lines, &cond.lhs, &cond.pred),
                loop_step_icmp_operand(&def_lines, &cond.rhs, &cond.pred),
                cond.line,
            );
            let lhs = lhs.as_str();
            let rhs = rhs.as_str();
            if small_guarded_ctz_decrement_loop(
                (body, blocks, src, dst),
                facts,
                &phi,
                ty.bit_width,
                step,
                (lhs, pred, rhs),
                backedge_on_true,
            ) {
                return true;
            }
            if counted_descending_below_step_loop(
                (body, blocks, src, dst),
                facts,
                &phi,
                ty.bit_width,
                step,
                (lhs, pred, rhs),
                backedge_on_true,
            ) {
                return true;
            }
            if !has_small_step {
                if matches!(pred, "ult" | "slt" | "ugt" | "sgt")
                    && step > 0
                    && backedge_on_true == Some(true)
                    && counted_upper_bound_lt_loop(
                        &def_lines,
                        facts,
                        &phi,
                        (lhs, pred, rhs),
                        ty.bit_width,
                        step,
                    )
                {
                    return true;
                }
                continue;
            }
            if pred == "eq" && step > 0 && backedge_on_true == Some(false) {
                if ty.bit_width != 32 {
                    continue;
                }
                if let Some(trip) = const_recur_eq_span(&phi, cond_line, step.unsigned_abs()) {
                    if small_trip_from_span(trip, step.unsigned_abs(), false) {
                        return true;
                    }
                }
                if let Some(trip) = symbolic_span(&def_lines, &phi.init, lhs, rhs, &phi.name, true)
                {
                    if small_trip_from_span(trip, step.unsigned_abs(), true) {
                        return true;
                    }
                }
                if counted_upper_bound_eq_loop(
                    (body, blocks, src, dst),
                    facts,
                    &phi,
                    lhs,
                    rhs,
                    ty.bit_width,
                ) {
                    return true;
                }
            }
            if pred == "ugt" && step < 0 && backedge_on_true == Some(true) {
                if ty.bit_width != 32 {
                    continue;
                }
                if let Some(trip) = symbolic_span(&def_lines, &phi.init, lhs, rhs, &phi.recur, true)
                {
                    if small_trip_from_span(trip, step.unsigned_abs(), false) {
                        return true;
                    }
                }
            }
            if pred == "eq" && step < 0 && backedge_on_true == Some(false) {
                if ty.bit_width != 32 {
                    continue;
                }
                if let Some(trip) = const_descending_eq_span(&phi, cond_line) {
                    if small_trip_from_span(trip, step.unsigned_abs(), true) {
                        return true;
                    }
                }
            }
            if pred == "ult" && step > 0 && backedge_on_true == Some(true) && ty.bit_width == 32 {
                if let Some(trip) = symbolic_span(&def_lines, &phi.init, lhs, rhs, &phi.recur, true)
                {
                    if small_trip_from_span(trip, step.unsigned_abs(), false) {
                        return true;
                    }
                }
            }
            if matches!(pred, "ult" | "slt" | "ugt" | "sgt")
                && step > 0
                && backedge_on_true == Some(true)
                && counted_upper_bound_lt_loop(
                    &def_lines,
                    facts,
                    &phi,
                    (lhs, pred, rhs),
                    ty.bit_width,
                    step,
                )
            {
                return true;
            }
        }
    }
    small_power_of_two_loop(body, header, latch, facts, src_label, dst_label)
}

struct SymbolicCondition<'a> {
    pred: String,
    lhs: String,
    rhs: String,
    line: &'a str,
}

fn symbolic_loop_condition_icmps<'a>(
    loop_lines: &[&'a str],
    cond: &str,
    backedge_on_true: Option<bool>,
    allow_select_conjuncts: bool,
) -> Vec<SymbolicCondition<'a>> {
    let mut out = Vec::new();
    let cond = cond.trim_start_matches('%');
    let Some(cond_line) = loop_lines
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(cond))
    else {
        return out;
    };
    if let Some((pred, lhs, rhs)) = parse_icmp(cond_line) {
        out.push(SymbolicCondition {
            pred: pred.to_string(),
            lhs,
            rhs,
            line: cond_line,
        });
    }
    if allow_select_conjuncts && backedge_on_true == Some(true) {
        for implied in bool_regs_implied_true_by_select_true(cond_line) {
            let Some(line) = loop_lines
                .iter()
                .copied()
                .find(|line| result_name(line).as_deref() == Some(implied))
            else {
                continue;
            };
            let Some((pred, lhs, rhs)) = parse_icmp(line) else {
                continue;
            };
            out.push(SymbolicCondition {
                pred: pred.to_string(),
                lhs,
                rhs,
                line,
            });
        }
    }
    out
}

fn const_descending_eq_span(phi: &PhiInfo, cond_line: &str) -> Option<i32> {
    if !phi.init_is_literal_int {
        return None;
    }
    let limit = literal_icmp_limit_for_operand(cond_line, &phi.name)?;
    let start = phi.init.parse::<i32>().ok()?;
    (start >= limit).then_some(start - limit)
}

fn const_recur_eq_span(phi: &PhiInfo, cond_line: &str, step: u32) -> Option<i32> {
    if !phi.init_is_literal_int {
        return None;
    }
    let limit = literal_icmp_limit_for_operand(cond_line, &phi.recur)?;
    let start = phi.init.parse::<i32>().ok()?;
    let span = limit.checked_sub(start)?;
    if span <= 0 || !(span as u32).is_multiple_of(step) {
        return None;
    }
    Some(span)
}

fn literal_icmp_limit_for_operand(line: &str, operand: &str) -> Option<i32> {
    let (_, rhs) = line.split_once(" = icmp ")?;
    let mut parts = rhs.split_whitespace();
    let _pred = parts.next()?;
    let _ty = parts.next()?;
    let lhs = parts.next()?.trim_end_matches(',');
    let rhs = parts.next()?.trim_end_matches(',');
    let operand = format!("%{}", operand.trim_start_matches('%'));
    if lhs == operand {
        return literal_i32_operand(rhs);
    }
    if rhs == operand {
        return literal_i32_operand(lhs);
    }
    None
}

fn literal_i32_operand(token: &str) -> Option<i32> {
    (!token.starts_with('%'))
        .then_some(token)?
        .parse::<i32>()
        .ok()
}

fn small_guarded_ctz_decrement_loop(
    loop_ctx: (&[&str], &[Block], usize, usize),
    facts: &LoopFacts,
    phi: &PhiInfo,
    bit_width: u32,
    step: i32,
    cond: (&str, &str, &str),
    backedge_on_true: Option<bool>,
) -> bool {
    if step != -1 || backedge_on_true != Some(true) {
        return false;
    }
    let (body, blocks, src, dst) = loop_ctx;
    let (lhs, pred, rhs) = cond;
    if !matches!(pred, "ugt" | "sgt") || (lhs != phi.recur && lhs != phi.name) {
        return false;
    }
    let Ok(limit) = rhs.parse::<i128>() else {
        return false;
    };
    let def_lines = body.to_vec();
    let Some(init_upper) =
        small_integer_upper_bound_with_facts(&def_lines, facts, &phi.init, bit_width)
    else {
        return false;
    };
    if init_upper <= limit {
        return true;
    }
    let trips = init_upper - limit + i128::from(lhs == phi.name);
    if !(1..=256).contains(&trips) {
        return false;
    }
    pred == "sgt" || loop_entry_is_guarded_by_signed_gt(body, blocks, src, dst, &phi.init, limit)
}

fn counted_descending_below_step_loop(
    loop_ctx: (&[&str], &[Block], usize, usize),
    facts: &LoopFacts,
    phi: &PhiInfo,
    bit_width: u32,
    step: i32,
    cond: (&str, &str, &str),
    backedge_on_true: Option<bool>,
) -> bool {
    let (lhs, pred, rhs) = cond;
    if pred != "ult" || lhs != phi.recur || step >= 0 || backedge_on_true != Some(false) {
        return false;
    }
    let decrement = i128::from(-step);
    if decrement <= 0 {
        return false;
    }
    let (body, blocks, src, dst) = loop_ctx;
    let def_lines = body.to_vec();
    let Some(threshold) = exact_positive_value(&def_lines, facts, rhs, bit_width, &mut Vec::new())
    else {
        return false;
    };
    if threshold != decrement {
        return false;
    }
    if !loop_entry_excludes_below((body, blocks, src, dst), (&phi.init, rhs, threshold), facts) {
        return false;
    }
    let Some(init_upper) =
        small_integer_upper_bound_with_facts(&def_lines, facts, &phi.init, bit_width)
    else {
        return false;
    };
    if init_upper < threshold {
        return true;
    }
    let trips = init_upper / decrement + 1;
    (1..=256).contains(&trips)
}

fn selected_latch_decrement_loop(
    body: &[&str],
    blocks: &[Block],
    latch: &Block,
    phi: &PhiInfo,
    bit_width: u32,
    branch: &str,
    cond: &str,
) -> bool {
    if bit_width > 16 || !branch.trim_start().starts_with("br i1 ") {
        return false;
    }
    let Some(select_line) =
        block_lines(body, latch).find(|line| result_name(line).as_deref() == Some(cond))
    else {
        return false;
    };
    let implied = bool_regs_implied_true_by_select_true(select_line);
    if implied.len() < 2 {
        return false;
    }
    let Some(nonnegative_guard) = implied.iter().copied().find(|reg| {
        block_lines(body, latch)
            .find(|line| result_name(line).as_deref() == Some(*reg))
            .is_some_and(|line| signed_nonnegative_guard_for(line, &phi.recur))
    }) else {
        return false;
    };
    let Some(latch_phi_line) =
        block_lines(body, latch).find(|line| result_name(line).as_deref() == Some(&phi.recur))
    else {
        return false;
    };
    let incomings = parse_phi_incomings(latch_phi_line);
    if incomings.is_empty() {
        return false;
    }
    let def_lines = body.to_vec();
    for selector in implied.into_iter().filter(|reg| *reg != nonnegative_guard) {
        for (value, pred) in &incomings {
            let Some((base, step)) = add_step(&def_lines, value) else {
                continue;
            };
            if base != phi.name || step >= 0 {
                continue;
            }
            if !bool_true_targets_label(body, blocks, selector, pred) {
                continue;
            }
            let max_start = (1_i128 << (bit_width - 1)) - 1;
            let trips = ((max_start + 1) + i128::from((-step) - 1)) / i128::from(-step);
            if (1..=i128::from(LOOP_BUDGET_BACKEDGES)).contains(&trips) {
                return true;
            }
        }
    }
    false
}

fn signed_nonnegative_guard_for(line: &str, value: &str) -> bool {
    let Some((pred, lhs, rhs)) = parse_icmp(line) else {
        return false;
    };
    pred == "sgt" && lhs == value && rhs == "-1"
}

fn parse_phi_incomings(line: &str) -> Vec<(String, String)> {
    line.split('[')
        .skip(1)
        .filter_map(|part| {
            let part = part.split(']').next()?;
            let (value, parent) = part.split_once(',')?;
            Some((
                value.trim().trim_start_matches('%').to_string(),
                parent.trim().trim_start_matches('%').to_string(),
            ))
        })
        .collect()
}

fn bool_true_targets_label(body: &[&str], blocks: &[Block], selector: &str, target: &str) -> bool {
    blocks.iter().any(|block| {
        let term = body[block.term_idx].trim_start();
        branch_condition_operand(term).is_some_and(|cond| {
            cond.trim_start_matches('%') == selector
                && branch_target_for_bool(term, true).as_deref() == Some(target)
        })
    })
}

fn counted_upper_bound_eq_loop(
    loop_ctx: (&[&str], &[Block], usize, usize),
    facts: &LoopFacts,
    phi: &PhiInfo,
    lhs: &str,
    rhs: &str,
    bit_width: u32,
) -> bool {
    let (body, blocks, src, dst) = loop_ctx;
    let init = phi_init_const_value(phi, &HashMap::new());
    let init_lower = if let Some(init) = init {
        init
    } else {
        let Some(lower) = small_integer_lower_bound_with_facts(body, facts, &phi.init, bit_width)
        else {
            return false;
        };
        lower
    };
    if init_lower < -256 {
        return false;
    }
    let bound = if lhs == phi.recur {
        rhs
    } else if rhs == phi.recur {
        lhs
    } else {
        return false;
    };
    let Some(upper) = small_integer_upper_bound_with_facts(body, facts, bound, bit_width) else {
        return false;
    };
    let trips = upper.saturating_sub(init_lower);
    if !(1..=256).contains(&trips) {
        return false;
    }
    if let Some(init) = init {
        loop_entry_excludes_at_most(body, blocks, src, dst, bound, init)
    } else {
        true
    }
}

fn counted_upper_bound_lt_loop(
    lines: &[&str],
    facts: &LoopFacts,
    phi: &PhiInfo,
    cond: (&str, &str, &str),
    bit_width: u32,
    step: i32,
) -> bool {
    let (lhs, pred, rhs) = cond;
    let Some(bound) = (match pred {
        "ult" | "slt" if lhs == phi.recur => Some(rhs),
        "ugt" | "sgt" if rhs == phi.recur => Some(lhs),
        _ => None,
    }) else {
        return false;
    };
    let values = if facts.is_empty() {
        HashMap::new()
    } else {
        consteval_int_values(lines, facts)
    };
    let init = if let Some(init) = phi_init_const_value(phi, &values) {
        init
    } else if matches!(pred, "ult" | "ugt") {
        // For unsigned loop predicates, any negative i32/i64 bit pattern is a large unsigned
        // value and exits immediately. The maximum trip count therefore starts at zero when the
        // non-literal seed has a finite launch/input upper bound.
        let Some(init_upper) =
            small_integer_upper_bound_with_facts(lines, facts, &phi.init, bit_width)
        else {
            return false;
        };
        if init_upper < 0 {
            return false;
        }
        0
    } else {
        return false;
    };
    if init < 0 {
        return false;
    }
    let Some(upper) = small_integer_upper_bound_with_facts(lines, facts, bound, bit_width) else {
        return false;
    };
    let step = i128::from(step);
    if step <= 0 {
        return false;
    }
    let span = upper.saturating_sub(init).max(0);
    let trips = ((span + step - 1) / step).max(1);
    if !(1..=256).contains(&trips) {
        return false;
    }
    let max_recur = init.saturating_add(step.saturating_mul(trips));
    let width_max = if matches!(pred, "slt" | "sgt") {
        (1_i128 << (bit_width - 1)) - 1
    } else if bit_width >= 128 {
        i128::MAX
    } else {
        (1_i128 << bit_width) - 1
    };
    max_recur <= width_max
}

fn small_integer_upper_bound_with_facts(
    lines: &[&str],
    facts: &LoopFacts,
    id: &str,
    bit_width: u32,
) -> Option<i128> {
    if !facts.is_empty() {
        if let Some(value) = consteval_int_values(lines, facts).get(id).copied() {
            return Some(value);
        }
    }
    small_integer_upper_bound_inner(lines, facts, id, bit_width, &mut Vec::new())
}

fn small_integer_lower_bound_with_facts(
    lines: &[&str],
    facts: &LoopFacts,
    id: &str,
    bit_width: u32,
) -> Option<i128> {
    if !facts.is_empty() {
        if let Some(value) = consteval_int_values(lines, facts).get(id).copied() {
            return Some(value);
        }
    }
    small_integer_lower_bound_inner(lines, facts, id, bit_width, &mut Vec::new())
}

fn small_integer_upper_bound_inner(
    lines: &[&str],
    facts: &LoopFacts,
    id: &str,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<i128> {
    if let Some(&value) = facts.arg_ints.get(id) {
        return Some(value);
    }
    if let Some(&upper) = facts.arg_upper_bounds.get(id) {
        return Some(upper);
    }
    if stack.iter().any(|seen| seen == id) {
        return None;
    }
    stack.push(id.to_string());
    let Some(line) = lines
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(id))
    else {
        stack.pop();
        return id.parse::<i128>().ok();
    };
    let value = load_global_upper_bound(facts, line)
        .or_else(|| load_pointer_upper_bound(facts, line))
        .or_else(|| extractelement_arg_upper_bound(facts, line))
        .or_else(|| extractelement_vector_upper_bound(lines, facts, line, bit_width))
        .or_else(|| texture_extent_call_upper_bound(facts, line))
        .or_else(|| imageblock_extent_call_bound(facts, line))
        .or_else(|| air_min_upper_bound(lines, facts, line, bit_width, stack))
        .or_else(|| air_max_upper_bound(lines, facts, line, bit_width, stack))
        .or_else(|| ctz_upper_bound(line))
        .or_else(|| small_integer_binary_upper_bound(lines, facts, line, bit_width, stack))
        .or_else(|| small_integer_cast_upper_bound(lines, facts, line, bit_width, stack))
        .or_else(|| float_to_integer_upper_bound(lines, facts, line, bit_width, stack))
        .or_else(|| small_integer_phi_upper_bound(lines, facts, line, bit_width, stack));
    stack.pop();
    value
}

fn small_integer_lower_bound_inner(
    lines: &[&str],
    facts: &LoopFacts,
    id: &str,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<i128> {
    if let Some(&value) = facts.arg_ints.get(id) {
        return Some(value);
    }
    if facts.arg_upper_bounds.contains_key(id) {
        return Some(0);
    }
    if stack.iter().any(|seen| seen == id) {
        return None;
    }
    stack.push(id.to_string());
    let Some(line) = lines
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(id))
    else {
        stack.pop();
        return id.parse::<i128>().ok();
    };
    let value = load_global_upper_bound(facts, line)
        .or_else(|| load_pointer_upper_bound(facts, line))
        .or_else(|| extractelement_arg_lower_bound(facts, line))
        .or_else(|| imageblock_extent_call_bound(facts, line))
        .or_else(|| small_integer_binary_lower_bound(lines, facts, line, bit_width, stack))
        .or_else(|| small_integer_cast_lower_bound(lines, facts, line, bit_width, stack))
        .or_else(|| float_to_integer_lower_bound(lines, facts, line, bit_width, stack));
    stack.pop();
    value
}

fn load_global_upper_bound(facts: &LoopFacts, line: &str) -> Option<i128> {
    let (_, global) = load_result_and_global(line)?;
    facts.global_ints.get(&global).copied()
}

fn load_pointer_upper_bound(facts: &LoopFacts, line: &str) -> Option<i128> {
    let (_, pointer) = load_result_and_pointer_arg(line)?;
    facts
        .arg_ints
        .get(&pointer)
        .or_else(|| facts.arg_upper_bounds.get(&pointer))
        .copied()
}

fn extractelement_arg_upper_bound(facts: &LoopFacts, line: &str) -> Option<i128> {
    let (_, vector, lane) = extractelement_result_vector_lane(line)?;
    facts
        .arg_vector_ints
        .get(&(vector.clone(), lane))
        .or_else(|| facts.arg_vector_upper_bounds.get(&(vector, lane)))
        .copied()
}

fn extractelement_vector_upper_bound(
    lines: &[&str],
    facts: &LoopFacts,
    line: &str,
    bit_width: u32,
) -> Option<i128> {
    let (_, vector, lane) = extractelement_result_vector_lane(line)?;
    vector_lane_upper_bound(lines, facts, &vector, lane, bit_width)
}

fn extractelement_arg_lower_bound(facts: &LoopFacts, line: &str) -> Option<i128> {
    let (_, vector, lane) = extractelement_result_vector_lane(line)?;
    if let Some(&value) = facts.arg_vector_ints.get(&(vector.clone(), lane)) {
        return Some(value);
    }
    facts
        .arg_vector_upper_bounds
        .contains_key(&(vector, lane))
        .then_some(0)
}

fn texture_extent_call_upper_bound(facts: &LoopFacts, line: &str) -> Option<i128> {
    let (_, texture, component) = texture_extent_call_result_arg_component(line)?;
    facts
        .texture_extents
        .get(&texture)
        .and_then(|extent| extent.get(component))
        .copied()
}

fn imageblock_extent_call_bound(facts: &LoopFacts, line: &str) -> Option<i128> {
    let (_, component) = imageblock_extent_call_result_component(line)?;
    facts
        .imageblock_extent
        .and_then(|extent| extent.get(component).copied())
}

fn air_min_upper_bound(
    lines: &[&str],
    facts: &LoopFacts,
    line: &str,
    bit_width: u32,
    _stack: &mut Vec<String>,
) -> Option<i128> {
    if !line.contains("@air.min.") {
        return None;
    }
    let (_, rhs) = line.split_once(" = ")?;
    let args = call_arg_tokens(rhs)?;
    if args.len() < 2 {
        return None;
    }
    let lhs = args[0].trim_start_matches('%');
    let rhs = args[1].trim_start_matches('%');
    let lhs_upper = lhs
        .parse::<i128>()
        .ok()
        .or_else(|| small_integer_upper_bound_with_facts(lines, facts, lhs, bit_width));
    let rhs_upper = rhs
        .parse::<i128>()
        .ok()
        .or_else(|| small_integer_upper_bound_with_facts(lines, facts, rhs, bit_width));
    match (lhs_upper, rhs_upper) {
        (Some(lhs), Some(rhs)) => Some(lhs.min(rhs)),
        (Some(known), None) | (None, Some(known)) => Some(known),
        (None, None) => None,
    }
}

fn air_max_upper_bound(
    lines: &[&str],
    facts: &LoopFacts,
    line: &str,
    bit_width: u32,
    _stack: &mut Vec<String>,
) -> Option<i128> {
    if !line.contains("@air.max.") {
        return None;
    }
    let (_, rhs) = line.split_once(" = ")?;
    let args = call_arg_tokens(rhs)?;
    if args.len() < 2 {
        return None;
    }
    let lhs = args[0].trim_start_matches('%');
    let rhs = args[1].trim_start_matches('%');
    let lhs_upper = lhs
        .parse::<i128>()
        .ok()
        .or_else(|| small_integer_upper_bound_with_facts(lines, facts, lhs, bit_width));
    let rhs_upper = rhs
        .parse::<i128>()
        .ok()
        .or_else(|| small_integer_upper_bound_with_facts(lines, facts, rhs, bit_width));
    Some(lhs_upper?.max(rhs_upper?))
}

fn call_arg_tokens(rhs: &str) -> Option<Vec<String>> {
    let open = rhs.find('(')?;
    let close = rhs.rfind(')')?;
    let mut tokens = Vec::new();
    for arg in rhs[open + 1..close].split(',') {
        let mut parts = arg.split_whitespace();
        let _ty = parts.next()?;
        let value = parts.next()?.trim_end_matches(',');
        tokens.push(value.to_string());
    }
    Some(tokens)
}

fn ctz_upper_bound(line: &str) -> Option<i128> {
    let (_, tail) = line.split_once("@air.ctz.i")?;
    let width = tail
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>()
        .parse::<i128>()
        .ok()?;
    (width > 0).then_some(width)
}

fn small_integer_binary_upper_bound(
    lines: &[&str],
    facts: &LoopFacts,
    line: &str,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<i128> {
    let (_, rhs) = line.split_once(" = ")?;
    let mut parts = rhs.split_whitespace();
    let op = parts.next()?;
    if !matches!(op, "add" | "sub" | "mul" | "and" | "udiv" | "urem" | "lshr") {
        return None;
    }
    let ty = loop {
        let part = parts.next()?;
        if int_type_width(part).is_some() {
            break part;
        }
    };
    let op_width = int_type_width(ty).unwrap_or(bit_width);
    let lhs = parts.next()?.trim_end_matches(',').trim_start_matches('%');
    let rhs = parts.next()?.trim_end_matches(',').trim_start_matches('%');
    match op {
        "add" => small_integer_upper_bound_inner(lines, facts, lhs, op_width, stack).and_then(
            |lhs_upper| {
                small_integer_upper_bound_inner(lines, facts, rhs, op_width, stack)
                    .and_then(|rhs_upper| lhs_upper.checked_add(rhs_upper))
            },
        ),
        "sub" => {
            if let Ok(value) = rhs.parse::<i128>() {
                small_integer_upper_bound_inner(lines, facts, lhs, op_width, stack)
                    .and_then(|upper| upper.checked_sub(value))
            } else if let Ok(value) = lhs.parse::<i128>() {
                small_integer_lower_bound_inner(lines, facts, rhs, op_width, stack)
                    .and_then(|lower| value.checked_sub(lower))
            } else {
                None
            }
        }
        "mul" => small_integer_upper_bound_inner(lines, facts, lhs, op_width, stack).and_then(
            |lhs_upper| {
                small_integer_upper_bound_inner(lines, facts, rhs, op_width, stack).and_then(
                    |rhs_upper| {
                        (lhs_upper >= 0 && rhs_upper >= 0)
                            .then(|| lhs_upper.checked_mul(rhs_upper))
                            .flatten()
                    },
                )
            },
        ),
        "and" => rhs
            .parse::<i128>()
            .ok()
            .or_else(|| lhs.parse::<i128>().ok())
            .map(|mask| mask_to_width(mask, op_width)),
        "udiv" => small_integer_upper_bound_inner(lines, facts, lhs, op_width, stack),
        "urem" => {
            small_integer_upper_bound_inner(lines, facts, lhs, op_width, stack).map(|upper| {
                exact_positive_value(lines, facts, rhs, op_width, stack)
                    .map_or(upper, |modulus| upper.min(modulus - 1))
            })
        }
        "lshr" => rhs.parse::<u32>().ok().and_then(|shift| {
            small_integer_upper_bound_inner(lines, facts, lhs, op_width, stack)
                .map(|upper| upper >> shift.min(127))
        }),
        _ => None,
    }
}

fn small_integer_binary_lower_bound(
    lines: &[&str],
    facts: &LoopFacts,
    line: &str,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<i128> {
    let (_, rhs) = line.split_once(" = ")?;
    let mut parts = rhs.split_whitespace();
    let op = parts.next()?;
    if !matches!(op, "add" | "sub" | "mul" | "and") {
        return None;
    }
    let ty = loop {
        let part = parts.next()?;
        if int_type_width(part).is_some() {
            break part;
        }
    };
    let op_width = int_type_width(ty).unwrap_or(bit_width);
    let lhs = parts.next()?.trim_end_matches(',').trim_start_matches('%');
    let rhs = parts.next()?.trim_end_matches(',').trim_start_matches('%');
    match op {
        "add" => small_integer_lower_bound_inner(lines, facts, lhs, op_width, stack).and_then(
            |lhs_lower| {
                small_integer_lower_bound_inner(lines, facts, rhs, op_width, stack)
                    .and_then(|rhs_lower| lhs_lower.checked_add(rhs_lower))
            },
        ),
        "sub" => small_integer_lower_bound_inner(lines, facts, lhs, op_width, stack).and_then(
            |lhs_lower| {
                small_integer_upper_bound_inner(lines, facts, rhs, op_width, stack)
                    .and_then(|rhs_upper| lhs_lower.checked_sub(rhs_upper))
            },
        ),
        "mul" => small_integer_lower_bound_inner(lines, facts, lhs, op_width, stack).and_then(
            |lhs_lower| {
                small_integer_lower_bound_inner(lines, facts, rhs, op_width, stack).and_then(
                    |rhs_lower| {
                        (lhs_lower >= 0 && rhs_lower >= 0)
                            .then(|| lhs_lower.checked_mul(rhs_lower))
                            .flatten()
                    },
                )
            },
        ),
        "and" => lhs
            .parse::<i128>()
            .ok()
            .or_else(|| rhs.parse::<i128>().ok())
            .filter(|mask| *mask >= 0)
            .map(|_| 0),
        _ => None,
    }
}

fn exact_positive_value(
    lines: &[&str],
    facts: &LoopFacts,
    id: &str,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<i128> {
    let value = id.parse::<i128>().ok().or_else(|| {
        facts
            .arg_ints
            .get(id)
            .copied()
            .or_else(|| small_integer_upper_bound_inner(lines, facts, id, bit_width, stack))
    })?;
    (value > 0).then_some(value)
}

fn small_integer_cast_upper_bound(
    lines: &[&str],
    facts: &LoopFacts,
    line: &str,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<i128> {
    let (_, rhs) = line.split_once(" = ")?;
    let mut parts = rhs.split_whitespace();
    let op = parts.next()?;
    if !matches!(op, "zext" | "sext" | "trunc") {
        return None;
    }
    let src_ty = parts.next()?;
    let src = parts.next()?.trim_end_matches(',').trim_start_matches('%');
    let _to = parts.next()?;
    let dst_ty = parts.next()?;
    let src_width = int_type_width(src_ty).unwrap_or(bit_width);
    let dst_width = int_type_width(dst_ty).unwrap_or(bit_width);
    let upper = small_integer_upper_bound_inner(lines, facts, src, src_width, stack)?;
    Some(match op {
        "zext" => mask_to_width(upper, src_width),
        "sext" => upper,
        "trunc" => mask_to_width(upper, dst_width),
        _ => return None,
    })
}

fn small_integer_cast_lower_bound(
    lines: &[&str],
    facts: &LoopFacts,
    line: &str,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<i128> {
    let (_, rhs) = line.split_once(" = ")?;
    let mut parts = rhs.split_whitespace();
    let op = parts.next()?;
    if !matches!(op, "zext" | "sext" | "trunc") {
        return None;
    }
    let src_ty = parts.next()?;
    let src = parts.next()?.trim_end_matches(',').trim_start_matches('%');
    let _to = parts.next()?;
    let dst_ty = parts.next()?;
    let src_width = int_type_width(src_ty).unwrap_or(bit_width);
    let dst_width = int_type_width(dst_ty).unwrap_or(bit_width);
    let lower = small_integer_lower_bound_inner(lines, facts, src, src_width, stack)?;
    Some(match op {
        "zext" => lower.max(0),
        "sext" => lower,
        "trunc" if lower >= 0 => mask_to_width(lower, dst_width),
        "trunc" => return None,
        _ => return None,
    })
}

#[derive(Clone, Copy)]
struct FloatBounds {
    lower: f64,
    upper: f64,
}

fn float_to_integer_upper_bound(
    lines: &[&str],
    facts: &LoopFacts,
    line: &str,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<i128> {
    let value = float_to_integer_source(line)?;
    let bounds = small_float_bounds(lines, facts, &value, bit_width, stack)?;
    finite_f64_to_i128_ceil(bounds.upper)
}

fn float_to_integer_lower_bound(
    lines: &[&str],
    facts: &LoopFacts,
    line: &str,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<i128> {
    let value = float_to_integer_source(line)?;
    let bounds = small_float_bounds(lines, facts, &value, bit_width, stack)?;
    finite_f64_to_i128_floor(bounds.lower)
}

fn float_to_integer_source(line: &str) -> Option<String> {
    if !((line.contains("@air.convert.s.") || line.contains("@air.convert.u."))
        && line.contains(".f.f32("))
    {
        return None;
    }
    call_arg_tokens(line)?
        .first()
        .map(|arg| arg.trim_start_matches('%').to_string())
}

fn small_float_bounds(
    lines: &[&str],
    facts: &LoopFacts,
    id: &str,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<FloatBounds> {
    if let Some(&value) = facts.arg_floats.get(id) {
        return Some(FloatBounds {
            lower: value,
            upper: value,
        });
    }
    if let Ok(value) = id.parse::<f64>() {
        return Some(FloatBounds {
            lower: value,
            upper: value,
        });
    }
    let key = format!("float:{id}");
    if stack.iter().any(|seen| seen == &key) {
        return None;
    }
    stack.push(key);
    let Some(line) = lines
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(id))
    else {
        stack.pop();
        return None;
    };
    let value = integer_to_float_bounds(lines, facts, line, bit_width, stack)
        .or_else(|| load_pointer_float_bounds(facts, line))
        .or_else(|| floor_float_bounds(lines, facts, line, bit_width, stack))
        .or_else(|| binary_float_bounds(lines, facts, line, bit_width, stack));
    stack.pop();
    value
}

fn load_pointer_float_bounds(facts: &LoopFacts, line: &str) -> Option<FloatBounds> {
    if !line.contains(" = load float") {
        return None;
    }
    let (_, pointer) = load_result_and_pointer_arg(line)?;
    let value = *facts.arg_floats.get(&pointer)?;
    Some(FloatBounds {
        lower: value,
        upper: value,
    })
}

fn integer_to_float_bounds(
    lines: &[&str],
    facts: &LoopFacts,
    line: &str,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<FloatBounds> {
    if !(line.contains("@air.convert.f.") && line.contains(".f32.")) {
        return None;
    }
    let source = call_arg_tokens(line)?
        .first()?
        .trim_start_matches('%')
        .to_string();
    let _ = stack;
    let lower = small_integer_lower_bound_with_facts(lines, facts, &source, bit_width)? as f64;
    let upper = small_integer_upper_bound_with_facts(lines, facts, &source, bit_width)? as f64;
    Some(FloatBounds { lower, upper })
}

fn floor_float_bounds(
    lines: &[&str],
    facts: &LoopFacts,
    line: &str,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<FloatBounds> {
    if !line.contains("@air.fast_floor.f32(") {
        return None;
    }
    let source = call_arg_tokens(line)?
        .first()?
        .trim_start_matches('%')
        .to_string();
    let bounds = small_float_bounds(lines, facts, &source, bit_width, stack)?;
    Some(FloatBounds {
        lower: bounds.lower.floor(),
        upper: bounds.upper.floor(),
    })
}

fn binary_float_bounds(
    lines: &[&str],
    facts: &LoopFacts,
    line: &str,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<FloatBounds> {
    let (_, rhs) = line.split_once(" = ")?;
    let mut parts = rhs.split_whitespace();
    let op = parts.next()?;
    if !matches!(op, "fadd" | "fsub" | "fmul") {
        return None;
    }
    let _flags_or_ty = parts.next()?;
    let ty = loop {
        let part = parts.next()?;
        if part == "float" {
            break part;
        }
    };
    let _ = ty;
    let lhs = parts.next()?.trim_end_matches(',').trim_start_matches('%');
    let rhs = parts.next()?.trim_end_matches(',').trim_start_matches('%');
    let lhs_bounds = small_float_operand_bounds(lines, facts, lhs, bit_width, stack)?;
    let rhs_bounds = small_float_operand_bounds(lines, facts, rhs, bit_width, stack)?;
    match op {
        "fadd" => Some(FloatBounds {
            lower: lhs_bounds.lower + rhs_bounds.lower,
            upper: lhs_bounds.upper + rhs_bounds.upper,
        }),
        "fsub" => Some(FloatBounds {
            lower: lhs_bounds.lower - rhs_bounds.upper,
            upper: lhs_bounds.upper - rhs_bounds.lower,
        }),
        "fmul" => {
            let products = [
                lhs_bounds.lower * rhs_bounds.lower,
                lhs_bounds.lower * rhs_bounds.upper,
                lhs_bounds.upper * rhs_bounds.lower,
                lhs_bounds.upper * rhs_bounds.upper,
            ];
            let lower = products.iter().copied().fold(f64::INFINITY, f64::min);
            let upper = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            Some(FloatBounds { lower, upper })
        }
        _ => None,
    }
}

fn small_float_operand_bounds(
    lines: &[&str],
    facts: &LoopFacts,
    token: &str,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<FloatBounds> {
    if let Ok(value) = token.parse::<f64>() {
        return Some(FloatBounds {
            lower: value,
            upper: value,
        });
    }
    small_float_bounds(lines, facts, token, bit_width, stack)
}

fn finite_f64_to_i128_ceil(value: f64) -> Option<i128> {
    (value.is_finite() && value >= i128::MIN as f64 && value <= i128::MAX as f64)
        .then(|| value.ceil() as i128)
}

fn finite_f64_to_i128_floor(value: f64) -> Option<i128> {
    (value.is_finite() && value >= i128::MIN as f64 && value <= i128::MAX as f64)
        .then(|| value.floor() as i128)
}

fn small_integer_phi_upper_bound(
    lines: &[&str],
    facts: &LoopFacts,
    line: &str,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<i128> {
    let result = result_name(line)?;
    let (_, rhs) = line.split_once(" = phi ")?;
    let mut max_value = None;
    for part in rhs.split('[').skip(1) {
        let part = part.split(']').next()?;
        let (value, _) = part.split_once(',')?;
        let value = value.trim().trim_start_matches('%');
        if value == result || value_is_nonincreasing_self_update(lines, value, &result) {
            continue;
        }
        let upper = small_integer_upper_bound_inner(lines, facts, value, bit_width, stack)?;
        max_value = Some(max_value.map_or(upper, |prev: i128| prev.max(upper)));
    }
    max_value
}

fn value_is_nonincreasing_self_update(lines: &[&str], value: &str, phi: &str) -> bool {
    let Some(line) = lines
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(value))
    else {
        return false;
    };
    let Some((_, rhs)) = line.split_once(" = ") else {
        return false;
    };
    let mut parts = rhs.split_whitespace();
    let Some(op) = parts.next() else {
        return false;
    };
    if op != "lshr" {
        return false;
    }
    while let Some(part) = parts.next() {
        if int_type_width(part).is_some() {
            let Some(lhs) = parts.next() else {
                return false;
            };
            let Some(rhs) = parts.next() else {
                return false;
            };
            return lhs.trim_end_matches(',').trim_start_matches('%') == phi
                && rhs
                    .trim_end_matches(',')
                    .parse::<u32>()
                    .is_ok_and(|shift| shift > 0);
        }
    }
    false
}

fn loop_entry_is_guarded_by_signed_gt(
    body: &[&str],
    blocks: &[Block],
    src: usize,
    dst: usize,
    value: &str,
    min: i128,
) -> bool {
    let preds = block_predecessors(blocks);
    let Some(entries) = preds.get(dst) else {
        return false;
    };
    entries
        .iter()
        .copied()
        .filter(|&pred| pred != src)
        .all(|entry| block_entry_is_guarded_by_signed_gt(body, blocks, &preds, entry, value, min))
}

fn block_entry_is_guarded_by_signed_gt(
    body: &[&str],
    blocks: &[Block],
    preds: &[Vec<usize>],
    entry: usize,
    value: &str,
    min: i128,
) -> bool {
    if block_has_incoming_signed_gt_guard(body, blocks, entry, value, min) {
        return true;
    }
    let Some(incoming) = preds.get(entry) else {
        return false;
    };
    !incoming.is_empty()
        && incoming.iter().copied().all(|pred| {
            block_targets_only(blocks, &blocks[pred], entry)
                && block_has_incoming_signed_gt_guard(body, blocks, pred, value, min)
        })
}

fn block_has_incoming_signed_gt_guard(
    body: &[&str],
    blocks: &[Block],
    target: usize,
    value: &str,
    min: i128,
) -> bool {
    let Some(target_label) = blocks[target].label.as_deref() else {
        return false;
    };
    let label = target_label.to_string();
    blocks.iter().any(|block| {
        branch_target_requires_signed_gt(body, block, &label, value, min).unwrap_or(false)
    })
}

fn branch_target_requires_signed_gt(
    body: &[&str],
    block: &Block,
    target_label: &str,
    value: &str,
    min: i128,
) -> Option<bool> {
    let branch = body[block.term_idx].trim_start();
    let cond = branch_condition(branch)?;
    let cond_line =
        block_lines(body, block).find(|line| result_name(line).as_deref() == Some(&cond))?;
    let Some(("sgt", lhs, rhs)) = parse_icmp(cond_line) else {
        return None;
    };
    let target_on_true = branch_backedge_on_true(branch, target_label)?;
    if !target_on_true {
        return None;
    }
    if lhs == value {
        return rhs.parse::<i128>().ok().map(|guard| guard >= min);
    }
    if rhs == value {
        return lhs.parse::<i128>().ok().map(|guard| guard >= min);
    }
    None
}

fn block_targets_only(blocks: &[Block], block: &Block, target: usize) -> bool {
    let Some(target_label) = blocks[target].label.as_deref() else {
        return false;
    };
    block.succ.as_slice() == [target_label]
}

fn loop_entry_excludes_at_most(
    body: &[&str],
    blocks: &[Block],
    src: usize,
    dst: usize,
    value: &str,
    threshold: i128,
) -> bool {
    let preds = block_predecessors(blocks);
    let Some(entries) = preds.get(dst) else {
        return false;
    };
    entries
        .iter()
        .copied()
        .filter(|&pred| pred != src)
        .all(|entry| {
            block_entry_excludes_at_most(body, blocks, &preds, entry, dst, value, threshold)
        })
}

fn loop_entry_excludes_below(
    loop_ctx: (&[&str], &[Block], usize, usize),
    guard: (&str, &str, i128),
    facts: &LoopFacts,
) -> bool {
    let (body, blocks, src, dst) = loop_ctx;
    let (value, threshold_token, threshold) = guard;
    let values = if facts.is_empty() {
        HashMap::new()
    } else {
        consteval_int_values(body, facts)
    };
    let ctx = BelowEntryGuardCtx {
        body,
        blocks,
        value,
        threshold_token,
        threshold,
        values: &values,
    };
    let preds = block_predecessors(blocks);
    let Some(entries) = preds.get(dst) else {
        return false;
    };
    entries
        .iter()
        .copied()
        .filter(|&pred| pred != src)
        .all(|entry| block_entry_excludes_below(&ctx, &preds, entry, dst))
}

struct BelowEntryGuardCtx<'a> {
    body: &'a [&'a str],
    blocks: &'a [Block],
    value: &'a str,
    threshold_token: &'a str,
    threshold: i128,
    values: &'a HashMap<String, i128>,
}

fn block_entry_excludes_below(
    ctx: &BelowEntryGuardCtx<'_>,
    preds: &[Vec<usize>],
    entry: usize,
    target: usize,
) -> bool {
    if block_edge_excludes_below(ctx, entry, target) {
        return true;
    }
    let Some(incoming) = preds.get(entry) else {
        return false;
    };
    !incoming.is_empty()
        && incoming.iter().copied().all(|pred| {
            block_edge_excludes_below(ctx, pred, entry)
                || (block_targets_only(ctx.blocks, &ctx.blocks[pred], entry)
                    && block_entry_excludes_below(ctx, preds, pred, entry))
        })
}

fn block_edge_excludes_below(ctx: &BelowEntryGuardCtx<'_>, from: usize, to: usize) -> bool {
    let Some(target_label) = ctx.blocks[to].label.as_deref() else {
        return false;
    };
    let label = target_label.to_string();
    let block = &ctx.blocks[from];
    branch_target_excludes_below(ctx, block, &label).unwrap_or(false)
}

fn branch_target_excludes_below(
    ctx: &BelowEntryGuardCtx<'_>,
    block: &Block,
    target_label: &str,
) -> Option<bool> {
    if block.kind != TermKind::Br {
        return None;
    }
    let branch = ctx.body[block.term_idx].trim_start();
    let cond = branch_condition(branch)?;
    let cond_line =
        block_lines(ctx.body, block).find(|line| result_name(line).as_deref() == Some(&cond))?;
    let (pred, lhs, rhs) = parse_icmp(cond_line)?;
    let target_on_true = branch_backedge_on_true(branch, target_label)?;
    let threshold_matches = |token: &str| {
        token == ctx.threshold_token
            || ctx
                .values
                .get(token.trim_start_matches('%'))
                .is_some_and(|value| *value == ctx.threshold)
            || operand_const_value(token, ctx.values).is_some_and(|value| value == ctx.threshold)
    };
    match pred {
        "ult" if lhs == ctx.value && threshold_matches(&rhs) => Some(!target_on_true),
        "uge" if lhs == ctx.value && threshold_matches(&rhs) => Some(target_on_true),
        _ => None,
    }
}

fn block_entry_excludes_at_most(
    body: &[&str],
    blocks: &[Block],
    preds: &[Vec<usize>],
    entry: usize,
    target: usize,
    value: &str,
    threshold: i128,
) -> bool {
    if block_edge_excludes_at_most(body, blocks, entry, target, value, threshold) {
        return true;
    }
    let Some(incoming) = preds.get(entry) else {
        return false;
    };
    !incoming.is_empty()
        && incoming.iter().copied().all(|pred| {
            block_edge_excludes_at_most(body, blocks, pred, entry, value, threshold)
                || (block_targets_only(blocks, &blocks[pred], entry)
                    && block_entry_excludes_at_most(
                        body, blocks, preds, pred, entry, value, threshold,
                    ))
        })
}

fn block_edge_excludes_at_most(
    body: &[&str],
    blocks: &[Block],
    from: usize,
    to: usize,
    value: &str,
    threshold: i128,
) -> bool {
    let Some(target_label) = blocks[to].label.as_deref() else {
        return false;
    };
    let label = target_label.to_string();
    let block = &blocks[from];
    branch_target_excludes_at_most(body, block, &label, value, threshold).unwrap_or(false)
        || (threshold == 0
            && switch_target_excludes_zero(body, block, &label, value).unwrap_or(false))
}

fn branch_target_excludes_at_most(
    body: &[&str],
    block: &Block,
    target_label: &str,
    value: &str,
    threshold: i128,
) -> Option<bool> {
    if block.kind != TermKind::Br {
        return None;
    }
    let branch = body[block.term_idx].trim_start();
    let cond = branch_condition(branch)?;
    let cond_line =
        block_lines(body, block).find(|line| result_name(line).as_deref() == Some(&cond))?;
    let (pred, lhs, rhs) = parse_icmp(cond_line)?;
    let compares_zero = (lhs == value && rhs == "0") || (rhs == value && lhs == "0");
    if !compares_zero && threshold == 0 {
        return None;
    }
    let target_on_true = branch_backedge_on_true(branch, target_label)?;
    match pred {
        "eq" if compares_zero && threshold == 0 => Some(!target_on_true),
        "ne" if compares_zero && threshold == 0 => Some(target_on_true),
        "ugt" | "sgt" if lhs == value && rhs.parse::<i128>().is_ok_and(|rhs| rhs >= threshold) => {
            Some(target_on_true)
        }
        _ => None,
    }
}

fn switch_target_excludes_zero(
    body: &[&str],
    block: &Block,
    target_label: &str,
    value: &str,
) -> Option<bool> {
    if block.kind != TermKind::Switch {
        return None;
    }
    let switch = body[block.term_idx].trim_start();
    if switch_value(switch)? != value {
        return None;
    }
    let mut zero_target = None;
    for line in &body[block.term_idx..=block.end] {
        let Some((case, target)) = switch_case_target(line) else {
            continue;
        };
        if case == 0 {
            zero_target = Some(target);
            break;
        }
    }
    zero_target.map(|target| target != target_label)
}

fn switch_value(line: &str) -> Option<String> {
    let rest = line.strip_prefix("switch ")?;
    let mut parts = rest.split_whitespace();
    let _ty = parts.next()?;
    let value = parts.next()?.trim_end_matches(',').trim_start_matches('%');
    Some(value.to_string())
}

fn switch_case_target(line: &str) -> Option<(i128, String)> {
    let trimmed = line.trim_start();
    let mut parts = trimmed.split_whitespace();
    let ty = parts.next()?;
    int_type_width(ty)?;
    let value = parts.next()?.trim_end_matches(',').parse::<i128>().ok()?;
    let _label = parts.next()?;
    let target = parts.next()?.trim_start_matches('%');
    Some((value, target.to_string()))
}

fn block_predecessors(blocks: &[Block]) -> Vec<Vec<usize>> {
    let label_to_idx = blocks
        .iter()
        .enumerate()
        .filter_map(|(idx, block)| block.label.as_ref().map(|label| (label.as_str(), idx)))
        .collect::<HashMap<_, _>>();
    let mut preds = vec![Vec::new(); blocks.len()];
    for (from, block) in blocks.iter().enumerate() {
        for succ in &block.succ {
            if let Some(&to) = label_to_idx.get(succ.as_str()) {
                preds[to].push(from);
            }
        }
    }
    preds
}

fn counted_consteval_loop(
    body: &[&str],
    blocks: &[Block],
    facts: &LoopFacts,
    src: usize,
    dst: usize,
) -> bool {
    if facts.is_empty() {
        return false;
    }
    let header = &blocks[dst];
    let latch = &blocks[src];
    let Some(src_label) = latch.label.as_deref() else {
        return false;
    };
    let Some(dst_label) = header.label.as_deref() else {
        return false;
    };
    let values = consteval_int_values(body, facts);
    let def_lines = body.to_vec();
    let latch_lines = block_lines(body, latch).collect::<Vec<_>>();
    let Some(branch) = latch_lines
        .iter()
        .rev()
        .copied()
        .find(|line| line.trim_start().starts_with("br i1 "))
    else {
        return false;
    };
    let Some(cond) = branch_condition(branch) else {
        return false;
    };
    let Some(backedge_on_true) = branch_backedge_on_true(branch, dst_label) else {
        return false;
    };
    let mut conds = vec![(cond.as_str(), backedge_on_true)];
    if backedge_on_true {
        if let Some(select_line) = latch_lines
            .iter()
            .copied()
            .find(|line| result_name(line).as_deref() == Some(cond.as_str()))
        {
            for implied in bool_regs_implied_true_by_select_true(select_line) {
                conds.push((implied, true));
            }
        }
    }
    for ty in ["i16", "i32", "i64"] {
        for line in block_lines(body, header).filter(|line| line.contains(&format!(" = phi {ty} ")))
        {
            let Some(phi) = parse_integer_phi(line, src_label, ty) else {
                continue;
            };
            if phi_init_const_value(&phi, &values).is_none() {
                continue;
            }
            if let Some((base, divisor)) = udiv_step_with_values(&def_lines, &values, &phi.recur) {
                if base == phi.name && divisor > 1 {
                    for (cond, continue_on_true) in &conds {
                        let Some(cond_line) = latch_lines
                            .iter()
                            .copied()
                            .find(|line| result_name(line).as_deref() == Some(*cond))
                        else {
                            continue;
                        };
                        let Some((pred, lhs, rhs)) = parse_icmp_value_operands(cond_line) else {
                            continue;
                        };
                        if udiv_trip_count_from_values(
                            &phi,
                            divisor,
                            pred,
                            &lhs,
                            &rhs,
                            *continue_on_true,
                            &values,
                        )
                        .is_some_and(|trip| (1..=256).contains(&trip))
                        {
                            return true;
                        }
                    }
                }
            }
            let Some((step_base, step)) = add_step_with_values(&def_lines, &values, &phi.recur)
            else {
                continue;
            };
            if step_base != phi.name || step == 0 {
                continue;
            }
            for (cond, continue_on_true) in &conds {
                let Some(cond_line) = latch_lines
                    .iter()
                    .copied()
                    .find(|line| result_name(line).as_deref() == Some(*cond))
                else {
                    continue;
                };
                let Some((pred, lhs, rhs)) = parse_icmp_value_operands(cond_line) else {
                    continue;
                };
                if counted_trip_count_from_values(
                    &phi,
                    step,
                    pred,
                    &lhs,
                    &rhs,
                    *continue_on_true,
                    &values,
                )
                .is_some_and(|trip| (1..=256).contains(&trip))
                {
                    return true;
                }
            }
        }
    }
    false
}

fn bool_regs_implied_true_by_select_true(line: &str) -> Vec<&str> {
    let Some((_, rhs)) = line.split_once(" = select i1 ") else {
        return Vec::new();
    };
    let Some((selector, rest)) = rhs.split_once(',') else {
        return Vec::new();
    };
    let mut arms = rest.split(',');
    let Some(true_arm) = arms.next().and_then(|arm| arm.trim().strip_prefix("i1 ")) else {
        return Vec::new();
    };
    let Some(false_arm) = arms.next().and_then(|arm| arm.trim().strip_prefix("i1 ")) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if false_arm.trim() == "false" {
        if let Some(reg) = bool_reg_token(selector.trim()) {
            out.push(reg);
        }
        if let Some(reg) = bool_reg_token(true_arm.trim()) {
            out.push(reg);
        }
    } else if true_arm.trim() == "false" {
        if let Some(reg) = bool_reg_token(false_arm.trim()) {
            out.push(reg);
        }
    }
    out
}

fn bool_reg_token(token: &str) -> Option<&str> {
    token.strip_prefix('%')
}

fn counted_trip_count_from_values(
    phi: &PhiInfo,
    step: i32,
    pred: &str,
    lhs: &str,
    rhs: &str,
    continue_on_true: bool,
    values: &HashMap<String, i128>,
) -> Option<u32> {
    let mut value = phi_init_const_value(phi, values)?;
    for trips in 1..=256 {
        let recur = value.checked_add(i128::from(step))?;
        let lhs = loop_operand_value(lhs, &phi.name, value, &phi.recur, recur, values)?;
        let rhs = loop_operand_value(rhs, &phi.name, value, &phi.recur, recur, values)?;
        let cond = eval_icmp(pred, lhs, rhs)?;
        if cond != continue_on_true {
            return Some(trips);
        }
        value = recur;
    }
    None
}

fn udiv_trip_count_from_values(
    phi: &PhiInfo,
    divisor: i128,
    pred: &str,
    lhs: &str,
    rhs: &str,
    continue_on_true: bool,
    values: &HashMap<String, i128>,
) -> Option<u32> {
    let mut value = phi_init_const_value(phi, values)?;
    if value < 0 || divisor <= 1 {
        return None;
    }
    for trips in 1..=256 {
        let recur = value.checked_div(divisor)?;
        let lhs = loop_operand_value(lhs, &phi.name, value, &phi.recur, recur, values)?;
        let rhs = loop_operand_value(rhs, &phi.name, value, &phi.recur, recur, values)?;
        let cond = eval_icmp(pred, lhs, rhs)?;
        if cond != continue_on_true {
            return Some(trips);
        }
        if recur == value {
            return None;
        }
        value = recur;
    }
    None
}

fn phi_init_const_value(phi: &PhiInfo, values: &HashMap<String, i128>) -> Option<i128> {
    if phi.init_is_literal_int {
        phi.init.parse().ok()
    } else {
        values.get(&phi.init).copied()
    }
}

fn loop_operand_value(
    token: &str,
    phi_name: &str,
    phi_value: i128,
    recur_name: &str,
    recur_value: i128,
    values: &HashMap<String, i128>,
) -> Option<i128> {
    if token == phi_name {
        Some(phi_value)
    } else if token == recur_name {
        Some(recur_value)
    } else {
        let token_name = token.trim_start_matches('%');
        if token_name == phi_name {
            Some(phi_value)
        } else if token_name == recur_name {
            Some(recur_value)
        } else {
            operand_const_value(token, values)
        }
    }
}

fn consteval_int_values(body: &[&str], facts: &LoopFacts) -> HashMap<String, i128> {
    let mut values = facts.arg_ints.clone();
    let mut pointer_values = HashMap::new();
    let mut pointer_field_paths = HashMap::new();
    let mut pointer_byte_offsets = HashMap::new();
    let mut pointer_arg_aliases = HashMap::new();
    let mut vector_values = HashMap::new();
    for line in body {
        if let Some(pointer) = store_dest_pointer(line) {
            pointer_values.remove(&pointer);
            pointer_field_paths.remove(&pointer);
            pointer_byte_offsets.remove(&pointer);
            pointer_arg_aliases.remove(&pointer);
            continue;
        }
        if let Some((result, arg, offset)) = byte_gep_result_arg_offset(line, &values) {
            pointer_byte_offsets.insert(
                result.trim_start_matches('%').to_string(),
                (arg.clone(), offset),
            );
            if let Some(&value) = facts.arg_byte_ints.get(&(arg, offset)) {
                pointer_values.insert(result.trim_start_matches('%').to_string(), value);
            }
            continue;
        }
        if let Some((result, arg, path)) = gep_result_arg_field_path(line, &values) {
            pointer_field_paths.insert(
                result.trim_start_matches('%').to_string(),
                (arg.clone(), path.clone()),
            );
            if let Some(value) = facts.field_value(&arg, &path) {
                pointer_values.insert(result.trim_start_matches('%').to_string(), value);
            }
            continue;
        }
        if line.contains(" = bitcast ") {
            if let Some(result) = result_name(line) {
                let rhs = line.split_once('=').map(|(_, rhs)| rhs).unwrap_or("");
                if let Some(source) = first_percent_name(rhs) {
                    if let Some(value) = pointer_values.get(&source).copied() {
                        pointer_values.insert(result.clone(), value);
                    }
                    if let Some(field) = pointer_field_paths.get(&source).cloned() {
                        pointer_field_paths.insert(result.clone(), field);
                    }
                    if let Some(byte) = pointer_byte_offsets.get(&source).cloned() {
                        pointer_byte_offsets.insert(result.clone(), byte);
                    }
                    if let Some(alias) = pointer_arg_aliases.get(&source).cloned() {
                        pointer_arg_aliases.insert(result, alias);
                    }
                }
            }
            continue;
        }
        if let Some((result, arg)) = gep_result_arg_root_alias(line, &values) {
            pointer_arg_aliases.insert(result.trim_start_matches('%').to_string(), arg);
            continue;
        }
        if let Some((result, arg)) = load_result_and_pointer_arg(line) {
            if let Some(lanes) = vector_int_load_lanes(line) {
                if let Some((field_arg, path)) = pointer_field_paths.get(&arg) {
                    let elements = (0..lanes)
                        .map(|element| {
                            let mut element_path = path.clone();
                            element_path.push(element as i32);
                            facts.field_value(field_arg, &element_path)
                        })
                        .collect::<Option<Vec<_>>>();
                    if let Some(elements) = elements {
                        vector_values.insert(result.trim_start_matches('%').to_string(), elements);
                    }
                } else if let Some(alias_arg) = pointer_arg_aliases.get(&arg) {
                    let elements = (0..lanes)
                        .map(|lane| {
                            facts
                                .arg_vector_ints
                                .get(&(alias_arg.clone(), lane))
                                .copied()
                        })
                        .collect::<Option<Vec<_>>>();
                    if let Some(elements) = elements {
                        vector_values.insert(result.trim_start_matches('%').to_string(), elements);
                    }
                }
                continue;
            }
            if let Some(&value) = pointer_values.get(&arg) {
                values.insert(result.trim_start_matches('%').to_string(), value);
            } else if let Some(alias_arg) = pointer_arg_aliases.get(&arg) {
                if let Some(&value) = facts.arg_ints.get(alias_arg) {
                    values.insert(result.trim_start_matches('%').to_string(), value);
                }
            } else if let Some(&value) = facts.arg_ints.get(&arg) {
                values.insert(result.trim_start_matches('%').to_string(), value);
            }
            continue;
        }
        if let Some((result, global)) = load_result_and_global(line) {
            if let Some(&value) = facts.global_ints.get(&global) {
                values.insert(result.trim_start_matches('%').to_string(), value);
            }
            continue;
        }
        if let Some((result, vector, lane)) = extractelement_result_vector_lane(line) {
            if let Some(&value) = facts.arg_vector_ints.get(&(vector, lane)) {
                values.insert(result, value);
                continue;
            }
        }
        if let Some((result, texture, component)) = texture_extent_call_result_arg_component(line) {
            if let Some(value) = facts
                .texture_extents
                .get(&texture)
                .and_then(|extent| extent.get(component))
                .copied()
            {
                values.insert(result, value);
                continue;
            }
        }
        if let Some((result, component)) = imageblock_extent_call_result_component(line) {
            if let Some(value) = facts
                .imageblock_extent
                .and_then(|extent| extent.get(component).copied())
            {
                values.insert(result, value);
                continue;
            }
        }
        let Some(result) = result_name(line) else {
            continue;
        };
        if let Some(vector) = consteval_vector_value(line, &values, &vector_values) {
            vector_values.insert(result, vector);
            continue;
        }
        if let Some(value) = consteval_icmp_int(line, &values)
            .or_else(|| consteval_select_int(line, &values))
            .or_else(|| consteval_cast(line, &values))
            .or_else(|| consteval_binary(line, &values))
            .or_else(|| consteval_same_value_phi(line, &values))
            .or_else(|| consteval_extractelement(line, &vector_values))
        {
            values.insert(result, value);
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for line in body {
            let Some(result) = result_name(line) else {
                continue;
            };
            if values.contains_key(&result) {
                continue;
            }
            if !vector_values.contains_key(&result) {
                if let Some(vector) = consteval_vector_value(line, &values, &vector_values) {
                    vector_values.insert(result, vector);
                    changed = true;
                    continue;
                }
            }
            if let Some(value) = consteval_icmp_int(line, &values)
                .or_else(|| consteval_select_int(line, &values))
                .or_else(|| consteval_cast(line, &values))
                .or_else(|| consteval_binary(line, &values))
                .or_else(|| consteval_same_value_phi(line, &values))
                .or_else(|| consteval_extractelement(line, &vector_values))
            {
                values.insert(result, value);
                changed = true;
            }
        }
    }
    values
}

fn byte_gep_result_arg_offset(
    line: &str,
    values: &HashMap<String, i128>,
) -> Option<(String, String, usize)> {
    if !line.contains(" = getelementptr ") || !line.contains(" i8,") {
        return None;
    }
    let result = result_name(line)?;
    let (arg, after_arg) = pointer_operand_and_tail_after(line, "ptr addrspace(")
        .or_else(|| pointer_operand_and_tail_after(line, "addrspace("))?;
    for raw in after_arg.split(',') {
        let mut parts = raw.split_whitespace();
        let Some(ty) = parts.next() else {
            continue;
        };
        let Some(value) = parts.next() else {
            continue;
        };
        if !ty.starts_with('i') || !ty[1..].chars().all(|ch| ch.is_ascii_digit()) {
            continue;
        }
        let offset = operand_const_value(value.trim_end_matches(','), values)?;
        return Some((result, arg, usize::try_from(offset).ok()?));
    }
    None
}

fn first_percent_name(text: &str) -> Option<String> {
    let (_, tail) = text.split_once('%')?;
    let name = tail
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '.' && ch != '_')
        .next()?;
    (!name.is_empty()).then(|| name.to_string())
}

fn extractelement_result_vector_lane(line: &str) -> Option<(String, String, usize)> {
    let result = result_name(line)?;
    let (_, rhs) = line.split_once(" = extractelement <")?;
    let (_, rest) = rhs.split_once('>')?;
    let mut parts = rest.split(',');
    let vector = parts.next()?.trim().trim_start_matches('%').to_string();
    let lane = parts
        .next()?
        .split_whitespace()
        .last()?
        .parse::<usize>()
        .ok()?;
    Some((result, vector, lane))
}

fn consteval_vector_value(
    line: &str,
    values: &HashMap<String, i128>,
    vector_values: &HashMap<String, Vec<i128>>,
) -> Option<Vec<i128>> {
    consteval_air_convert_vector(line, vector_values)
        .or_else(|| consteval_vector_binary(line, values, vector_values))
}

fn consteval_air_convert_vector(
    line: &str,
    vector_values: &HashMap<String, Vec<i128>>,
) -> Option<Vec<i128>> {
    if !line.contains("@air.convert.") {
        return None;
    }
    let (_, after_call) = line.split_once(" call <")?;
    let (result_type, _) = after_call.split_once('>')?;
    let (lanes, dst_width) = vector_type_lanes_and_width(result_type)?;
    let intrinsic = line.split_once("@air.convert.")?.1.split_once('(')?.0;
    let mut parts = intrinsic.split('.');
    let _dst_signedness = parts.next()?;
    let dst_vector_type = parts.next()?;
    let src_signedness = parts.next()?;
    let src_vector_type = parts.next()?;
    if parts.next().is_some() || !matches!(src_signedness, "s" | "u") {
        return None;
    }
    let (intrinsic_lanes, intrinsic_dst_width) = vector_name_lanes_and_width(dst_vector_type)?;
    let (src_lanes, src_width) = vector_name_lanes_and_width(src_vector_type)?;
    if intrinsic_lanes != lanes
        || src_lanes != lanes
        || intrinsic_dst_width != dst_width
        || dst_width > 127
    {
        return None;
    }
    let arg = call_vector_argument(line)?;
    let source = vector_values.get(&arg)?;
    if source.len() != lanes {
        return None;
    }
    source
        .iter()
        .map(|&value| {
            let value = if src_signedness == "s" {
                sign_extend(value, src_width)
            } else {
                mask_to_width(value, src_width)
            };
            Some(sign_extend(value, dst_width))
        })
        .collect()
}

fn consteval_vector_binary(
    line: &str,
    values: &HashMap<String, i128>,
    vector_values: &HashMap<String, Vec<i128>>,
) -> Option<Vec<i128>> {
    let (_, rhs) = line.split_once(" = ")?;
    let mut parts = rhs.split_whitespace();
    let op = parts.next()?;
    if !matches!(
        op,
        "add" | "sub" | "mul" | "shl" | "lshr" | "ashr" | "and" | "or"
    ) {
        return None;
    }
    let vector_type_start = rhs.find('<')?;
    let vector_type_end = rhs[vector_type_start + 1..].find('>')? + vector_type_start + 1;
    let (lanes, width) = vector_type_lanes_and_width(&rhs[vector_type_start + 1..vector_type_end])?;
    let operands = rhs[vector_type_end + 1..].trim();
    let (lhs, rhs) = operands.split_once(',')?;
    let lhs = vector_operand_values(lhs.trim(), lanes, values, vector_values)?;
    let rhs = vector_operand_values(rhs.trim(), lanes, values, vector_values)?;
    lhs.into_iter()
        .zip(rhs)
        .map(|(lhs, rhs)| consteval_binary_values(op, width, lhs, rhs))
        .collect()
}

fn vector_operand_values(
    operand: &str,
    lanes: usize,
    values: &HashMap<String, i128>,
    vector_values: &HashMap<String, Vec<i128>>,
) -> Option<Vec<i128>> {
    if let Some(name) = operand.trim().strip_prefix('%') {
        return vector_values
            .get(name)
            .filter(|v| v.len() == lanes)
            .cloned();
    }
    if let Some(inner) = operand
        .trim()
        .strip_prefix("splat (")
        .and_then(|s| s.strip_suffix(')'))
    {
        let value = inner.split_whitespace().last()?;
        return operand_const_value(value, values).map(|value| vec![value; lanes]);
    }
    if let Some(inner) = operand
        .trim()
        .strip_prefix('<')
        .and_then(|s| s.strip_suffix('>'))
    {
        let elements = inner
            .split(',')
            .map(|element| {
                let value = element.split_whitespace().last()?;
                operand_const_value(value, values)
            })
            .collect::<Option<Vec<_>>>()?;
        return (elements.len() == lanes).then_some(elements);
    }
    operand_const_value(operand.trim(), values).map(|value| vec![value; lanes])
}

fn call_vector_argument(line: &str) -> Option<String> {
    let args = line.split_once('(')?.1.rsplit_once(')')?.0;
    args.split(',')
        .find_map(|arg| {
            arg.split_whitespace()
                .find_map(|part| part.strip_prefix('%'))
        })
        .map(|name| {
            name.trim_end_matches(|ch: char| {
                !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '.')
            })
            .to_string()
        })
}

fn vector_type_lanes_and_width(type_body: &str) -> Option<(usize, u32)> {
    let (lanes, ty) = type_body.split_once(" x ")?;
    let lanes = lanes.trim().parse().ok()?;
    let width = int_type_width(ty.trim())?;
    Some((lanes, width))
}

fn vector_name_lanes_and_width(name: &str) -> Option<(usize, u32)> {
    let name = name.trim();
    let after_v = name.strip_prefix('v')?;
    let lane_end = after_v.find('i')?;
    let lanes = after_v[..lane_end].parse().ok()?;
    let width = int_type_width(&after_v[lane_end..])?;
    Some((lanes, width))
}

fn vector_lane_upper_bound(
    lines: &[&str],
    facts: &LoopFacts,
    vector: &str,
    lane: usize,
    bit_width: u32,
) -> Option<i128> {
    vector_lane_upper_bound_inner(lines, facts, vector, lane, bit_width, &mut Vec::new())
}

fn vector_lane_upper_bound_inner(
    lines: &[&str],
    facts: &LoopFacts,
    vector: &str,
    lane: usize,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<i128> {
    if let Some(&value) = facts.arg_vector_ints.get(&(vector.to_string(), lane)) {
        return Some(value);
    }
    if let Some(&upper) = facts
        .arg_vector_upper_bounds
        .get(&(vector.to_string(), lane))
    {
        return Some(upper);
    }
    if stack.iter().any(|seen| seen == vector) {
        return None;
    }
    stack.push(vector.to_string());
    let Some(line) = lines
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(vector))
    else {
        stack.pop();
        return None;
    };
    let value = insertelement_lane_value(line)
        .and_then(|insert| {
            if insert.lane == lane {
                small_integer_upper_bound_with_facts(lines, facts, &insert.value, bit_width)
            } else if matches!(insert.base.as_str(), "undef" | "poison" | "zeroinitializer") {
                None
            } else {
                vector_lane_upper_bound_inner(lines, facts, &insert.base, lane, bit_width, stack)
            }
        })
        .or_else(|| vector_binary_lane_upper_bound(lines, facts, line, lane, bit_width, stack));
    stack.pop();
    value
}

fn vector_binary_lane_upper_bound(
    lines: &[&str],
    facts: &LoopFacts,
    line: &str,
    lane: usize,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<i128> {
    let (_, rhs) = line.split_once(" = ")?;
    let mut parts = rhs.split_whitespace();
    let op = parts.next()?;
    if op != "and" {
        return None;
    }
    let vector_type_start = rhs.find('<')?;
    let vector_type_end = rhs[vector_type_start + 1..].find('>')? + vector_type_start + 1;
    let (lanes, width) = vector_type_lanes_and_width(&rhs[vector_type_start + 1..vector_type_end])?;
    if lane >= lanes {
        return None;
    }
    let operands = split_top_level_commas(rhs[vector_type_end + 1..].trim());
    if operands.len() != 2 {
        return None;
    }
    let lhs_mask = vector_operand_lane_literal(operands[0], lane);
    let rhs_mask = vector_operand_lane_literal(operands[1], lane);
    let mask = lhs_mask.or(rhs_mask)?;
    let masked_upper = mask_to_width(mask, width);
    let other = if lhs_mask.is_some() {
        operands[1]
    } else {
        operands[0]
    };
    vector_operand_lane_upper_bound(lines, facts, other, lane, bit_width, stack)
        .map(|upper| upper.min(masked_upper))
        .or(Some(masked_upper))
}

fn vector_operand_lane_upper_bound(
    lines: &[&str],
    facts: &LoopFacts,
    operand: &str,
    lane: usize,
    bit_width: u32,
    stack: &mut Vec<String>,
) -> Option<i128> {
    let operand = operand.trim();
    if let Some(name) = operand.strip_prefix('%') {
        return vector_lane_upper_bound_inner(lines, facts, name, lane, bit_width, stack);
    }
    vector_operand_lane_literal(operand, lane).map(|value| mask_to_width(value, bit_width))
}

fn vector_operand_lane_literal(operand: &str, lane: usize) -> Option<i128> {
    if let Some(inner) = operand
        .trim()
        .strip_prefix("splat (")
        .and_then(|s| s.strip_suffix(')'))
    {
        return inner.split_whitespace().last()?.parse::<i128>().ok();
    }
    if let Some(inner) = operand
        .trim()
        .strip_prefix('<')
        .and_then(|s| s.strip_suffix('>'))
    {
        let element = inner.split(',').nth(lane)?;
        return element.split_whitespace().last()?.parse::<i128>().ok();
    }
    operand.trim().parse::<i128>().ok()
}

struct InsertElementLane {
    base: String,
    value: String,
    lane: usize,
}

fn insertelement_lane_value(line: &str) -> Option<InsertElementLane> {
    let (_, rhs) = line.split_once(" = insertelement <")?;
    let (_, rest) = rhs.split_once('>')?;
    let mut parts = split_top_level_commas(rest).into_iter();
    let base = parts
        .next()?
        .split_whitespace()
        .last()?
        .trim_start_matches('%')
        .to_string();
    let value = parts
        .next()?
        .split_whitespace()
        .last()?
        .trim_start_matches('%')
        .to_string();
    let lane = parts
        .next()?
        .split_whitespace()
        .last()?
        .parse::<usize>()
        .ok()?;
    Some(InsertElementLane { base, value, lane })
}

fn texture_extent_call_result_arg_component(line: &str) -> Option<(String, String, usize)> {
    let result = result_name(line)?;
    let component = if line.contains("@air.get_width_texture_") {
        0
    } else if line.contains("@air.get_height_texture_") {
        1
    } else if line.contains("@air.get_depth_texture_") {
        2
    } else {
        return None;
    };
    let texture = first_ssa_token_after(line, "ptr addrspace(")
        .or_else(|| first_ssa_token_after(line, "addrspace("))?;
    Some((result, texture, component))
}

fn imageblock_extent_call_result_component(line: &str) -> Option<(String, usize)> {
    let result = result_name(line)?;
    let component = if line.contains("@air.get_imageblock_width(") {
        0
    } else if line.contains("@air.get_imageblock_height(") {
        1
    } else {
        return None;
    };
    Some((result, component))
}

fn first_ssa_token_after(line: &str, marker: &str) -> Option<String> {
    let (_, after_marker) = line.split_once(marker)?;
    let start = after_marker.find('%')? + 1;
    let rest = &after_marker[start..];
    let end = rest
        .find(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '.'))
        .unwrap_or(rest.len());
    (end > 0).then(|| rest[..end].to_string())
}

fn vector_int_load_lanes(line: &str) -> Option<usize> {
    let (_, rhs) = line.split_once(" = load <")?;
    let (lanes, rest) = rhs.split_once(" x ")?;
    let lanes = lanes.trim().parse().ok()?;
    let (ty, _) = rest.split_once('>')?;
    matches!(ty.trim(), "i8" | "i16" | "i32" | "i64").then_some(lanes)
}

fn consteval_extractelement(
    line: &str,
    vector_values: &HashMap<String, Vec<i128>>,
) -> Option<i128> {
    let (_, rhs) = line.split_once(" = extractelement <")?;
    let (_, rest) = rhs.split_once('>')?;
    let mut parts = rest.split(',');
    let vector = parts.next()?.trim().trim_start_matches('%');
    let index = parts
        .next()?
        .split_whitespace()
        .last()?
        .parse::<usize>()
        .ok()?;
    vector_values.get(vector)?.get(index).copied()
}

fn consteval_same_value_phi(line: &str, values: &HashMap<String, i128>) -> Option<i128> {
    let (_, rhs) = line.split_once(" = phi ")?;
    let mut parts = rhs.split_whitespace();
    let ty = parts.next()?;
    if !matches!(ty, "i16" | "i32" | "i64") {
        return None;
    }
    let mut value = None;
    for incoming in rhs.split('[').skip(1) {
        let incoming = incoming.split(']').next()?;
        let (raw_value, _) = incoming.split_once(',')?;
        let incoming_value = operand_const_value(raw_value.trim(), values)?;
        if value.is_some_and(|value| value != incoming_value) {
            return None;
        }
        value = Some(incoming_value);
    }
    value
}

fn gep_result_arg_field_path(
    line: &str,
    values: &HashMap<String, i128>,
) -> Option<(String, String, Vec<i32>)> {
    if !line.contains(" = getelementptr ") {
        return None;
    }
    let result = result_name(line)?;
    let (arg, after_arg) = pointer_operand_and_tail_after(line, "ptr addrspace(")
        .or_else(|| pointer_operand_and_tail_after(line, "addrspace("))?;
    let mut path = Vec::new();
    let mut saw_root = false;
    for raw in after_arg.split(',') {
        let mut parts = raw.split_whitespace();
        let Some(ty) = parts.next() else {
            continue;
        };
        let Some(value) = parts.next() else {
            continue;
        };
        if !ty.starts_with('i') || !ty[1..].chars().all(|ch| ch.is_ascii_digit()) {
            continue;
        }
        let value = value.trim_end_matches(',');
        let Some(index) = operand_const_value(value, values).and_then(|v| i32::try_from(v).ok())
        else {
            continue;
        };
        if !saw_root {
            if index != 0 {
                return None;
            }
            saw_root = true;
        } else {
            path.push(index);
        }
    }
    (saw_root && !path.is_empty()).then_some((result, arg, path))
}

fn gep_result_arg_root_alias(
    line: &str,
    values: &HashMap<String, i128>,
) -> Option<(String, String)> {
    if !line.contains(" = getelementptr ") {
        return None;
    }
    let result = result_name(line)?;
    let (arg, after_arg) = pointer_operand_and_tail_after(line, "ptr addrspace(")
        .or_else(|| pointer_operand_and_tail_after(line, "addrspace("))?;
    let mut saw_root = false;
    for raw in after_arg.split(',') {
        let mut parts = raw.split_whitespace();
        let Some(ty) = parts.next() else {
            continue;
        };
        let Some(value) = parts.next() else {
            continue;
        };
        if !ty.starts_with('i') || !ty[1..].chars().all(|ch| ch.is_ascii_digit()) {
            continue;
        }
        let value = value.trim_end_matches(',');
        let index = operand_const_value(value, values).and_then(|v| i32::try_from(v).ok())?;
        if !saw_root {
            if index != 0 {
                return None;
            }
            saw_root = true;
        } else {
            return None;
        }
    }
    saw_root.then_some((result, arg))
}

fn reachable_blocks_under_facts(body: &[&str], blocks: &[Block], facts: &LoopFacts) -> Vec<bool> {
    if facts.is_empty() {
        return vec![true; blocks.len()];
    }
    if blocks.is_empty() {
        return Vec::new();
    }
    let base_values = consteval_int_values(body, facts);
    let label_to_idx = blocks
        .iter()
        .enumerate()
        .filter_map(|(idx, block)| block.label.as_ref().map(|label| (label.as_str(), idx)))
        .collect::<HashMap<_, _>>();
    let block_labels = blocks
        .iter()
        .map(|block| block.label.clone())
        .collect::<Vec<_>>();
    let mut seen = vec![false; blocks.len()];
    let mut states = vec![None::<HashMap<String, i128>>; blocks.len()];
    let mut stack = vec![(0usize, None::<String>, base_values)];
    while let Some((idx, pred, mut values)) = stack.pop() {
        apply_entry_phi_values(body, &blocks[idx], pred.as_deref(), &mut values);
        if !merge_reachable_state(&mut states[idx], values) {
            continue;
        }
        seen[idx] = true;
        let mut values = states[idx].clone().unwrap_or_default();
        consteval_block_values(body, &blocks[idx], &mut values);
        let pred_label = block_labels[idx].clone();
        for target in reachable_successors(body, &blocks[idx], &values) {
            if let Some(&next) = label_to_idx.get(target.as_str()) {
                stack.push((next, pred_label.clone(), values.clone()));
            }
        }
    }
    seen
}

fn merge_reachable_state(
    current: &mut Option<HashMap<String, i128>>,
    incoming: HashMap<String, i128>,
) -> bool {
    let Some(current) = current else {
        *current = Some(incoming);
        return true;
    };
    let before = current.len();
    current.retain(|key, value| incoming.get(key).is_some_and(|incoming| incoming == value));
    current.len() != before
}

fn apply_entry_phi_values(
    body: &[&str],
    block: &Block,
    pred: Option<&str>,
    values: &mut HashMap<String, i128>,
) {
    let Some(pred) = pred else {
        return;
    };
    for line in block_lines(body, block) {
        if !line.contains(" = phi ") {
            continue;
        }
        let Some(result) = result_name(line) else {
            continue;
        };
        let Some((value, _)) = parse_phi_incomings(line)
            .into_iter()
            .find(|(_, incoming)| incoming == pred)
        else {
            values.remove(&result);
            continue;
        };
        if let Some(value) = value_as_const(&value, values) {
            values.insert(result, value);
        } else {
            values.remove(&result);
        }
    }
}

fn consteval_block_values(body: &[&str], block: &Block, values: &mut HashMap<String, i128>) {
    for line in block_lines(body, block) {
        if line.contains(" = phi ") {
            continue;
        }
        let Some(result) = result_name(line) else {
            continue;
        };
        if let Some(value) = consteval_icmp_int(line, values)
            .or_else(|| consteval_select_int(line, values))
            .or_else(|| consteval_cast(line, values))
            .or_else(|| consteval_binary(line, values))
            .or_else(|| consteval_same_value_phi(line, values))
        {
            values.insert(result, value);
        }
    }
}

fn reachable_successors(
    body: &[&str],
    block: &Block,
    values: &HashMap<String, i128>,
) -> Vec<String> {
    let term = body[block.term_idx].trim_start();
    if let Some(rest) = term.strip_prefix("br label %") {
        return vec![rest.trim().to_string()];
    }
    if let Some(cond) = branch_condition_operand(term) {
        if let Some(value) = consteval_bool_operand(&cond, body, values) {
            return branch_target_for_bool(term, value).into_iter().collect();
        }
    }
    block.succ.clone()
}

fn consteval_bool_operand(
    cond: &str,
    body: &[&str],
    values: &HashMap<String, i128>,
) -> Option<bool> {
    match cond {
        "true" => return Some(true),
        "false" => return Some(false),
        _ => {}
    }
    let cond = cond.trim_start_matches('%');
    let line = body
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(cond))?;
    if let Some((pred, _ty, lhs, rhs)) = parse_icmp_value_parts(line) {
        return eval_icmp_partial(
            pred,
            &lhs,
            &rhs,
            value_as_const(&lhs, values),
            value_as_const(&rhs, values),
        );
    }
    consteval_select_bool(line, values)
}

fn eval_icmp_partial(
    pred: &str,
    lhs_token: &str,
    rhs_token: &str,
    lhs: Option<i128>,
    rhs: Option<i128>,
) -> Option<bool> {
    if lhs_token == rhs_token {
        return Some(matches!(pred, "eq" | "ule" | "uge" | "sle" | "sge"));
    }
    if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
        return eval_icmp(pred, lhs, rhs);
    }
    match (pred, rhs) {
        ("ult", Some(0)) => Some(false),
        ("uge", Some(0)) => Some(true),
        _ => None,
    }
}

fn consteval_select_bool(line: &str, values: &HashMap<String, i128>) -> Option<bool> {
    let (_, rhs) = line.split_once(" = select i1 ")?;
    let (cond, rest) = rhs.split_once(',')?;
    let cond = consteval_bool_token(cond.trim(), values)?;
    let mut arms = rest.split(',');
    let true_arm = arms.next()?.trim().strip_prefix("i1 ")?.trim();
    let false_arm = arms.next()?.trim().strip_prefix("i1 ")?.trim();
    if cond {
        consteval_bool_token(true_arm, values)
    } else {
        consteval_bool_token(false_arm, values)
    }
}

fn consteval_bool_token(token: &str, values: &HashMap<String, i128>) -> Option<bool> {
    match token {
        "true" => Some(true),
        "false" => Some(false),
        _ => value_as_const(token, values).map(|value| value != 0),
    }
}

fn consteval_icmp_int(line: &str, values: &HashMap<String, i128>) -> Option<i128> {
    let (pred, _ty, lhs, rhs) = parse_icmp_value_parts(line)?;
    eval_icmp_partial(
        pred,
        &lhs,
        &rhs,
        value_as_const(&lhs, values),
        value_as_const(&rhs, values),
    )
    .map(i128::from)
}

fn consteval_select_int(line: &str, values: &HashMap<String, i128>) -> Option<i128> {
    let (_, rhs) = line.split_once(" = select i1 ")?;
    let (cond, rest) = rhs.split_once(',')?;
    let cond = consteval_bool_token(cond.trim(), values)?;
    let mut arms = rest.split(',');
    let (true_ty, true_value) = select_value_arm(arms.next()?.trim())?;
    let (false_ty, false_value) = select_value_arm(arms.next()?.trim())?;
    if true_ty != false_ty {
        return None;
    }
    let width = int_type_width(true_ty)?;
    let value = if cond {
        operand_const_value(true_value, values)?
    } else {
        operand_const_value(false_value, values)?
    };
    Some(sign_extend(value, width))
}

fn expression_recur_trip_count_from_values(
    lines: &[&str],
    values: &HashMap<String, i128>,
    phi: &PhiInfo,
    cond: (&str, &str, &str),
    backedge_on_true: Option<bool>,
) -> Option<u32> {
    let backedge_on_true = backedge_on_true?;
    let mut current = phi_init_const_value(phi, values)?;
    let mut seen = HashSet::new();
    let (pred, lhs, rhs) = cond;
    for trips in 1..=256 {
        let step_values = consteval_int_values_with_phi(lines, values, &phi.name, current);
        let next = step_values.get(&phi.recur).copied()?;
        let lhs = value_as_const(lhs, &step_values)?;
        let rhs = value_as_const(rhs, &step_values)?;
        let cond_value = eval_icmp(pred, lhs, rhs)?;
        let continue_loop = cond_value == backedge_on_true;
        if !continue_loop {
            return Some(trips);
        }
        if !seen.insert(next) {
            return None;
        }
        current = next;
    }
    None
}

fn consteval_int_values_with_phi(
    lines: &[&str],
    values: &HashMap<String, i128>,
    phi: &str,
    current: i128,
) -> HashMap<String, i128> {
    let mut out = values.clone();
    out.insert(phi.to_string(), current);
    for _ in 0..lines.len() {
        let mut changed = false;
        for line in lines {
            let Some(name) = result_name(line) else {
                continue;
            };
            if out.contains_key(&name) {
                continue;
            }
            let value = consteval_binary(line, &out)
                .or_else(|| consteval_cast(line, &out))
                .or_else(|| consteval_select_int(line, &out))
                .or_else(|| consteval_icmp_int(line, &out));
            if let Some(value) = value {
                out.insert(name, value);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    out
}

fn select_value_arm(arm: &str) -> Option<(&str, &str)> {
    let mut parts = arm.split_whitespace();
    let ty = parts.next()?;
    int_type_width(ty)?;
    let value = parts.next()?.trim_end_matches(',');
    Some((ty, value))
}

fn eval_icmp(pred: &str, lhs: i128, rhs: i128) -> Option<bool> {
    Some(match pred {
        "eq" => lhs == rhs,
        "ne" => lhs != rhs,
        "slt" => lhs < rhs,
        "sle" => lhs <= rhs,
        "sgt" => lhs > rhs,
        "sge" => lhs >= rhs,
        "ult" => lhs >= 0 && rhs >= 0 && lhs < rhs,
        "ule" => lhs >= 0 && rhs >= 0 && lhs <= rhs,
        "ugt" => lhs >= 0 && rhs >= 0 && lhs > rhs,
        "uge" => lhs >= 0 && rhs >= 0 && lhs >= rhs,
        _ => return None,
    })
}

fn branch_target_for_bool(line: &str, value: bool) -> Option<String> {
    let rest = line.trim_start().strip_prefix("br i1 ")?;
    let mut parts = rest.split(',');
    let _cond = parts.next()?;
    let true_target = parts.next()?.trim().strip_prefix("label %")?;
    let false_target = parts.next()?.trim().strip_prefix("label %")?;
    Some(if value {
        true_target.to_string()
    } else {
        false_target.to_string()
    })
}

fn consteval_cast(line: &str, values: &HashMap<String, i128>) -> Option<i128> {
    let (_, rhs) = line.split_once(" = ")?;
    let mut parts = rhs.split_whitespace();
    let op = parts.next()?;
    if !matches!(op, "zext" | "sext" | "trunc") {
        return None;
    }
    let src_ty = parts.next()?;
    let src = parts.next()?;
    let to = parts.next()?;
    if to != "to" {
        return None;
    }
    let dst_ty = parts.next()?;
    let value = operand_const_value(src.trim_end_matches(','), values)?;
    match op {
        "zext" => Some(mask_to_width(value, int_type_width(src_ty)?)),
        "sext" => Some(sign_extend(value, int_type_width(src_ty)?)),
        "trunc" => Some(sign_extend(value, int_type_width(dst_ty)?)),
        _ => None,
    }
}

fn consteval_binary(line: &str, values: &HashMap<String, i128>) -> Option<i128> {
    let (_, rhs) = line.split_once(" = ")?;
    let mut parts = rhs.split_whitespace();
    let op = parts.next()?;
    if !matches!(
        op,
        "add"
            | "sub"
            | "mul"
            | "sdiv"
            | "udiv"
            | "srem"
            | "urem"
            | "shl"
            | "lshr"
            | "ashr"
            | "and"
            | "or"
    ) {
        return None;
    }
    let ty = loop {
        let part = parts.next()?;
        if part.starts_with('i') && part[1..].chars().all(|ch| ch.is_ascii_digit()) {
            break part;
        }
    };
    let width = int_type_width(ty)?;
    let lhs = operand_const_value(parts.next()?.trim_end_matches(','), values)?;
    let rhs = operand_const_value(parts.next()?.trim_end_matches(','), values)?;
    let value = consteval_binary_values(op, width, lhs, rhs)?;
    Some(sign_extend(value, width))
}

fn consteval_binary_values(op: &str, width: u32, lhs: i128, rhs: i128) -> Option<i128> {
    let value = match op {
        "add" => lhs.checked_add(rhs)?,
        "sub" => lhs.checked_sub(rhs)?,
        "mul" => lhs.checked_mul(rhs)?,
        "sdiv" => {
            if rhs == 0 {
                return None;
            }
            lhs.checked_div(rhs)?
        }
        "udiv" => {
            if rhs == 0 {
                return None;
            }
            let lhs = mask_to_width(lhs, width);
            let rhs = mask_to_width(rhs, width);
            lhs.checked_div(rhs)?
        }
        "srem" => {
            if rhs == 0 {
                return None;
            }
            lhs.checked_rem(rhs)?
        }
        "urem" => {
            if rhs == 0 {
                return None;
            }
            let lhs = mask_to_width(lhs, width);
            let rhs = mask_to_width(rhs, width);
            lhs.checked_rem(rhs)?
        }
        "shl" => {
            let shift = u32::try_from(rhs).ok()?;
            lhs.checked_shl(shift)?
        }
        "lshr" => {
            let shift = u32::try_from(rhs).ok()?;
            mask_to_width(lhs, width).checked_shr(shift)?
        }
        "ashr" => {
            let shift = u32::try_from(rhs).ok()?;
            sign_extend(lhs, width).checked_shr(shift)?
        }
        "and" => mask_to_width(lhs, width) & mask_to_width(rhs, width),
        "or" => mask_to_width(lhs, width) | mask_to_width(rhs, width),
        _ => return None,
    };
    Some(value)
}

fn value_as_const(token: &str, values: &HashMap<String, i128>) -> Option<i128> {
    let token = token.trim().trim_end_matches(',');
    match token {
        "true" => return Some(1),
        "false" => return Some(0),
        _ => {}
    }
    if let Some(name) = token.strip_prefix('%') {
        return values.get(name).copied();
    }
    token
        .parse::<i128>()
        .ok()
        .or_else(|| values.get(token).copied())
}

fn int_type_width(ty: &str) -> Option<u32> {
    ty.strip_prefix('i')?.parse().ok()
}

fn mask_to_width(value: i128, width: u32) -> i128 {
    if width >= 128 {
        value
    } else {
        let mask = (1_i128 << width) - 1;
        value & mask
    }
}

fn sign_extend(value: i128, width: u32) -> i128 {
    if width == 0 || width >= 128 {
        return value;
    }
    let masked = mask_to_width(value, width);
    let sign = 1_i128 << (width - 1);
    if masked & sign == 0 {
        masked
    } else {
        masked - (1_i128 << width)
    }
}

#[derive(Clone)]
struct PhiInfo {
    name: String,
    init: String,
    recur: String,
    init_is_literal_int: bool,
}

fn branch_condition(line: &str) -> Option<String> {
    let rest = line.trim_start().strip_prefix("br i1 ")?;
    Some(
        rest.split(',')
            .next()?
            .trim()
            .trim_start_matches('%')
            .to_string(),
    )
}

fn branch_condition_operand(line: &str) -> Option<String> {
    let cond = line
        .trim_start()
        .strip_prefix("br i1 ")?
        .split(',')
        .next()?
        .trim();
    if cond.starts_with('%') || matches!(cond, "true" | "false") {
        Some(cond.to_string())
    } else {
        Some(format!("%{cond}"))
    }
}

fn parse_icmp(line: &str) -> Option<(&str, String, String)> {
    let (pred, ty, lhs, rhs) = parse_icmp_parts(line)?;
    if !matches!(ty, "i16" | "i32" | "i64") {
        return None;
    }
    Some((pred, lhs, rhs))
}

fn parse_icmp_value_operands(line: &str) -> Option<(&str, String, String)> {
    let (pred, ty, lhs, rhs) = parse_icmp_value_parts(line)?;
    if !matches!(ty, "i16" | "i32" | "i64") {
        return None;
    }
    Some((pred, lhs, rhs))
}

fn parse_icmp_value_parts(line: &str) -> Option<(&str, &str, String, String)> {
    let (_, rhs) = line.split_once(" = icmp ")?;
    let mut parts = rhs.split_whitespace();
    let pred = parts.next()?;
    let ty = parts.next()?;
    let lhs = parts.next()?.trim_end_matches(',');
    let rhs = parts.next()?.trim_end_matches(',');
    Some((pred, ty, lhs.to_string(), rhs.to_string()))
}

fn parse_icmp_parts(line: &str) -> Option<(&str, &str, String, String)> {
    let (_, rhs) = line.split_once(" = icmp ")?;
    let mut parts = rhs.split_whitespace();
    let pred = parts.next()?;
    let ty = parts.next()?;
    let lhs = parts.next()?.trim_end_matches(',').trim_start_matches('%');
    let rhs = parts.next()?.trim_end_matches(',').trim_start_matches('%');
    Some((pred, ty, lhs.to_string(), rhs.to_string()))
}

fn parse_integer_phi(line: &str, src_label: &str, ty: &str) -> Option<PhiInfo> {
    let name = result_name(line)?;
    let (_, rhs) = line.split_once(&format!(" = phi {ty} "))?;
    let mut init = None;
    let mut recur = None;
    for part in rhs.split('[').skip(1) {
        let part = part.split(']').next()?;
        let (value, parent) = part.split_once(',')?;
        let raw_value = value.trim();
        let value = raw_value.trim_start_matches('%').to_string();
        let parent = parent.trim().trim_start_matches('%');
        if parent == src_label {
            recur = Some(value);
        } else {
            let is_literal_int = !raw_value.starts_with('%') && raw_value.parse::<i128>().is_ok();
            init = Some((value, is_literal_int));
        }
    }
    let (init, init_is_literal_int) = init?;
    Some(PhiInfo {
        name,
        init,
        recur: recur?,
        init_is_literal_int,
    })
}

fn add_step(lines: &[&str], id: &str) -> Option<(String, i32)> {
    add_step_inner(lines, id, &mut Vec::new())
}

fn add_step_inner(lines: &[&str], id: &str, stack: &mut Vec<String>) -> Option<(String, i32)> {
    if stack.iter().any(|seen| seen == id) {
        return None;
    }
    stack.push(id.to_string());
    let line = lines
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(id))?;
    let Some((_, rhs)) = line.split_once(" = add ") else {
        let source = sign_extend_or_identity_source(lines, line)?;
        let out = add_step_inner(lines, &source, stack);
        stack.pop();
        return out;
    };
    let mut parts = rhs.split_whitespace();
    let ty = loop {
        let part = parts.next()?;
        if int_type_width(part).is_some() {
            break part;
        }
    };
    let lhs_raw = parts.next()?.trim_end_matches(',');
    let rhs_raw = parts.next()?.trim_end_matches(',');
    if !rhs_raw.starts_with('%') {
        if let Ok(step) = rhs_raw.parse::<i32>() {
            stack.pop();
            return Some((lhs_raw.trim_start_matches('%').to_string(), step));
        }
    }
    if !lhs_raw.starts_with('%') {
        if let Ok(step) = lhs_raw.parse::<i32>() {
            stack.pop();
            return Some((rhs_raw.trim_start_matches('%').to_string(), step));
        }
    }
    let _ = ty;
    stack.pop();
    None
}

fn sign_extend_or_identity_source(lines: &[&str], line: &str) -> Option<String> {
    let (_, rhs) = line.split_once(" = ashr ")?;
    let mut parts = rhs.split_whitespace();
    let first = parts.next()?;
    let ty = if first == "exact" {
        // AIR often spells a 16-bit signed value in an i32 register as:
        // `%wide = shl i32 %value, 16`; `%narrow = ashr exact i32 %wide, 16`.
        loop {
            let part = parts.next()?;
            if int_type_width(part).is_some() {
                break part;
            }
        }
    } else if int_type_width(first).is_some() {
        first
    } else {
        return None;
    };
    let source = parts.next()?.trim_end_matches(',').trim_start_matches('%');
    let shift = parts.next()?.trim_end_matches(',').parse::<i32>().ok()?;
    if shift == 0 {
        return Some(source.to_string());
    }
    let source_line = lines
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(source))?;
    let (_, shl_rhs) = source_line.split_once(" = shl ")?;
    let mut shl_parts = shl_rhs.split_whitespace();
    let shl_ty = loop {
        let part = shl_parts.next()?;
        if int_type_width(part).is_some() {
            break part;
        }
    };
    if shl_ty != ty {
        return None;
    }
    let unshifted = shl_parts
        .next()?
        .trim_end_matches(',')
        .trim_start_matches('%');
    let shl = shl_parts
        .next()?
        .trim_end_matches(',')
        .parse::<i32>()
        .ok()?;
    (shl == shift).then(|| unshifted.to_string())
}

fn add_step_with_values(
    lines: &[&str],
    values: &HashMap<String, i128>,
    id: &str,
) -> Option<(String, i32)> {
    let line = lines
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(id))?;
    let (op, rhs) = if let Some((_, rhs)) = line.split_once(" = add ") {
        ("add", rhs)
    } else {
        let (_, rhs) = line.split_once(" = sub ")?;
        ("sub", rhs)
    };
    let mut parts = rhs.split_whitespace();
    let _ty = loop {
        let part = parts.next()?;
        if int_type_width(part).is_some() {
            break part;
        }
    };
    let lhs_raw = parts.next()?.trim_end_matches(',');
    let rhs_raw = parts.next()?.trim_end_matches(',');
    if op == "sub" {
        let step =
            operand_const_value(rhs_raw, values).and_then(|value| i32::try_from(value).ok())?;
        return Some((lhs_raw.trim_start_matches('%').to_string(), -step));
    }
    if let Some(step) =
        operand_const_value(lhs_raw, values).and_then(|value| i32::try_from(value).ok())
    {
        return Some((rhs_raw.trim_start_matches('%').to_string(), step));
    }
    if let Some(step) =
        operand_const_value(rhs_raw, values).and_then(|value| i32::try_from(value).ok())
    {
        return Some((lhs_raw.trim_start_matches('%').to_string(), step));
    }
    None
}

fn udiv_step_with_values(
    lines: &[&str],
    values: &HashMap<String, i128>,
    id: &str,
) -> Option<(String, i128)> {
    let line = lines
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(id))?;
    let (_, rhs) = line.split_once(" = udiv ")?;
    let mut parts = rhs.split_whitespace();
    let _ty = loop {
        let part = parts.next()?;
        if int_type_width(part).is_some() {
            break part;
        }
    };
    let lhs = parts.next()?.trim_end_matches(',').trim_start_matches('%');
    let divisor = operand_const_value(parts.next()?.trim_end_matches(','), values)?;
    Some((lhs.to_string(), divisor))
}

fn operand_const_value(token: &str, values: &HashMap<String, i128>) -> Option<i128> {
    let token = token.trim().trim_end_matches(',');
    match token {
        "true" => return Some(1),
        "false" => return Some(0),
        _ => {}
    }
    if let Some(name) = token.strip_prefix('%') {
        values.get(name).copied()
    } else {
        token.parse::<i128>().ok()
    }
}

fn branch_backedge_on_true(branch: &str, dst_label: &str) -> Option<bool> {
    let targets = extract_label_targets(branch);
    match targets.as_slice() {
        [true_target, false_target] if true_target == dst_label => {
            let _ = false_target;
            Some(true)
        }
        [true_target, false_target] if false_target == dst_label => {
            let _ = true_target;
            Some(false)
        }
        _ => None,
    }
}

fn small_trip_from_span(span: i32, step: u32, inclusive: bool) -> bool {
    if span < 0 || step == 0 {
        return false;
    }
    let span = span as u32;
    let trips = span.div_ceil(step) + u32::from(inclusive);
    (1..=256).contains(&trips)
}

fn symbolic_span(
    lines: &[&str],
    init: &str,
    lhs: &str,
    rhs: &str,
    induction: &str,
    exclusive_upper: bool,
) -> Option<i32> {
    let limit = if lhs == induction {
        rhs
    } else if rhs == induction {
        lhs
    } else {
        return None;
    };
    let (base_a, start) = affine_small_const(lines, init)?;
    let (base_b, end) = affine_small_const(lines, limit)?;
    if base_a != base_b {
        return None;
    }
    let span = if end >= start {
        end - start
    } else {
        start - end
    };
    Some(if exclusive_upper { span } else { span + 1 })
}

fn small_power_of_two_loop(
    body: &[&str],
    header: &Block,
    latch: &Block,
    facts: &LoopFacts,
    src_label: &str,
    dst_label: &str,
) -> bool {
    let def_lines = body.to_vec();
    let values = consteval_int_values(body, facts);
    let latch_lines = block_lines(body, latch).collect::<Vec<_>>();
    let Some(branch) = latch_lines
        .iter()
        .rev()
        .copied()
        .find(|line| line.trim_start().starts_with("br i1 "))
    else {
        return false;
    };
    let Some(cond) = branch_condition(branch) else {
        return false;
    };
    let Some(backedge_on_true) = branch_backedge_on_true(branch, dst_label) else {
        return false;
    };
    let Some(cond_line) =
        block_lines(body, latch).find(|line| result_name(line).as_deref() == Some(cond.as_str()))
    else {
        return false;
    };
    let Some((pred, lhs, rhs)) = parse_icmp(cond_line) else {
        return false;
    };
    let lhs = loop_step_icmp_operand(&def_lines, &lhs, pred);
    let rhs = loop_step_icmp_operand(&def_lines, &rhs, pred);
    for line in block_lines(body, header) {
        let Some((phi, ty)) = parse_integer_phi_any(line, src_label) else {
            continue;
        };
        if unsigned_right_shift_const_trip_count(
            &def_lines,
            &phi,
            (&lhs, pred, &rhs),
            ty.bit_width,
            backedge_on_true,
            &values,
        )
        .is_some_and(|trips| (1..=256).contains(&trips))
        {
            return true;
        }
        let Some((base, step)) = halve_or_double_by_two(&def_lines, &phi.recur) else {
            continue;
        };
        if base != phi.name {
            continue;
        }
        if halving_gt_limit_trips_are_small(
            &phi.name,
            &phi.recur,
            (&lhs, pred, &rhs),
            step,
            ty.bit_width,
            backedge_on_true,
        ) {
            return true;
        }
        if let Some((limit, recur_on_lhs)) = recur_icmp_limit(&phi.recur, &lhs, &rhs) {
            if power_of_two_trip_count_from_values_after_step_icmp(
                &phi.init,
                limit,
                step,
                pred,
                recur_on_lhs,
                backedge_on_true,
                &values,
            )
            .is_some_and(|trips| (1..=256).contains(&trips))
            {
                return true;
            }
        }
        if let Some(induction) = next_step_doubling_limit(&phi, pred, &lhs, &rhs) {
            if power_of_two_trip_count_from_upper_bound(
                &def_lines,
                facts,
                &phi.init,
                induction,
                step,
                ty.bit_width,
                backedge_on_true,
            )
            .is_some_and(|trips| (1..=256).contains(&trips))
            {
                return true;
            }
            if power_of_two_trip_count_from_values_after_step(
                &phi.init,
                induction,
                step,
                backedge_on_true,
                &values,
            )
            .is_some_and(|trips| (1..=256).contains(&trips))
            {
                return true;
            }
        }
        if pred != "ult" {
            continue;
        }
        if let Some(induction) = if lhs == phi.name {
            Some(rhs.as_str())
        } else if rhs == phi.name {
            Some(lhs.as_str())
        } else {
            None
        } {
            if symbolic_right_shift_trips_are_small(
                &def_lines,
                induction,
                step,
                ty.bit_width,
                backedge_on_true,
            ) {
                return true;
            }
            let Some((start_base, start)) = affine_small_const(&def_lines, &phi.init) else {
                continue;
            };
            let Some((limit_base, limit)) = affine_small_const(&def_lines, induction) else {
                continue;
            };
            if start_base.is_empty()
                && limit_base.is_empty()
                && power_of_two_trip_count(start, limit, step, backedge_on_true)
                    .is_some_and(|trips| (1..=256).contains(&trips))
            {
                return true;
            }
            if power_of_two_trip_count_from_values(
                &phi.init,
                induction,
                step,
                backedge_on_true,
                &values,
            )
            .is_some_and(|trips| (1..=256).contains(&trips))
            {
                return true;
            }
        }
    }
    for line in block_lines(body, header) {
        let Some((phi, ty)) = parse_integer_phi_any(line, src_label) else {
            continue;
        };
        let Some((base, mask)) = masked_double_by_two(&def_lines, &phi.recur) else {
            continue;
        };
        if base != phi.name {
            continue;
        }
        if masked_doubling_until_exceeds_const_trips_are_small(
            &phi.init,
            &phi.recur,
            (&lhs, pred, &rhs),
            backedge_on_true,
            mask,
            ty.bit_width,
            &values,
        ) {
            return true;
        }
    }
    false
}

fn recur_icmp_limit<'a>(recur: &str, lhs: &'a str, rhs: &'a str) -> Option<(&'a str, bool)> {
    if lhs == recur {
        Some((rhs, true))
    } else if rhs == recur {
        Some((lhs, false))
    } else {
        None
    }
}

fn loop_step_icmp_operand(lines: &[&str], operand: &str, pred: &str) -> String {
    preserving_cast_source_for_pred(lines, operand, pred).unwrap_or_else(|| operand.to_string())
}

fn preserving_cast_source_for_pred(lines: &[&str], operand: &str, pred: &str) -> Option<String> {
    let line = lines
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(operand))?;
    let (_, rhs) = line.split_once(" = ")?;
    let mut parts = rhs.split_whitespace();
    let op = parts.next()?;
    let src_ty = parts.next()?;
    let src = parts.next()?.trim_end_matches(',').trim_start_matches('%');
    let _to = parts.next()?;
    let dst_ty = parts.next()?;
    let src_width = int_type_width(src_ty)?;
    let dst_width = int_type_width(dst_ty)?;
    let preserves_order = match (pred.strip_prefix('s'), pred.strip_prefix('u'), op) {
        (Some(_), _, "sext") => dst_width >= src_width,
        (_, Some(_), "zext") => dst_width >= src_width,
        _ => false,
    };
    preserves_order.then(|| src.to_string())
}

fn next_step_doubling_limit<'a>(
    phi: &PhiInfo,
    pred: &str,
    lhs: &'a str,
    rhs: &'a str,
) -> Option<&'a str> {
    match pred {
        "ult" | "slt" if lhs == phi.recur => Some(rhs),
        "ugt" | "sgt" if rhs == phi.recur => Some(lhs),
        _ => None,
    }
}

fn power_of_two_trip_count_from_upper_bound(
    lines: &[&str],
    facts: &LoopFacts,
    init: &str,
    limit: &str,
    step: HalveOrDoubleStep,
    bit_width: u32,
    backedge_on_true: bool,
) -> Option<u32> {
    if step != HalveOrDoubleStep::Shl1 || !backedge_on_true {
        return None;
    }
    let start = init.parse::<i128>().ok()?;
    if start <= 0 {
        return None;
    }
    let limit = small_integer_upper_bound_with_facts(lines, facts, limit, bit_width)?;
    if limit <= start {
        return Some(1);
    }
    let mut value = start;
    for trips in 1..=256 {
        let next = value.checked_mul(2)?;
        if next >= limit {
            return Some(trips);
        }
        value = next;
    }
    None
}

fn halving_gt_limit_trips_are_small(
    phi: &str,
    recur: &str,
    cond: (&str, &str, &str),
    step: HalveOrDoubleStep,
    bit_width: u32,
    backedge_on_true: bool,
) -> bool {
    if !backedge_on_true {
        return false;
    }
    let (lhs, pred, rhs) = cond;
    let continues_while_recur_gt_limit = match step {
        HalveOrDoubleStep::LShr1 => {
            (lhs == recur && pred == "ugt")
                || (rhs == recur && pred == "ult")
                || (lhs == phi && pred == "ugt")
                || (rhs == phi && pred == "ult")
        }
        HalveOrDoubleStep::SDiv2 => (lhs == phi && pred == "sgt") || (rhs == phi && pred == "slt"),
        HalveOrDoubleStep::AShr1 => false,
        HalveOrDoubleStep::Shl1 => false,
    };
    continues_while_recur_gt_limit && (1..=256).contains(&bit_width)
}

fn unsigned_right_shift_const_trip_count(
    lines: &[&str],
    phi: &PhiInfo,
    cond: (&str, &str, &str),
    bit_width: u32,
    backedge_on_true: bool,
    values: &HashMap<String, i128>,
) -> Option<u32> {
    let (base, shift, op_width) = unsigned_right_shift_const(lines, &phi.recur)?;
    if base != phi.name || shift == 0 || shift >= bit_width || op_width != bit_width {
        return None;
    }
    let mut current = value_as_const(&phi.init, values)?;
    if current < 0 {
        return None;
    }
    let mut seen = HashSet::new();
    let (lhs, pred, rhs) = cond;
    for trips in 1..=256 {
        let next = mask_to_width(current, bit_width).checked_shr(shift)?;
        let lhs = loop_reduction_operand_value(lhs, phi, current, next, values)?;
        let rhs = loop_reduction_operand_value(rhs, phi, current, next, values)?;
        let continue_loop = eval_icmp(pred, lhs, rhs)? == backedge_on_true;
        if !continue_loop {
            return Some(trips);
        }
        if !seen.insert(next) {
            return None;
        }
        current = next;
    }
    None
}

fn loop_reduction_operand_value(
    operand: &str,
    phi: &PhiInfo,
    current: i128,
    next: i128,
    values: &HashMap<String, i128>,
) -> Option<i128> {
    if operand == phi.name {
        Some(current)
    } else if operand == phi.recur {
        Some(next)
    } else {
        value_as_const(operand, values)
    }
}

#[derive(Clone, Copy)]
struct IntegerPhiTy {
    bit_width: u32,
}

fn parse_integer_phi_any(line: &str, src_label: &str) -> Option<(PhiInfo, IntegerPhiTy)> {
    for (ty, bit_width) in [("i16", 16), ("i32", 32), ("i64", 64)] {
        if let Some(phi) = parse_integer_phi(line, src_label, ty) {
            return Some((phi, IntegerPhiTy { bit_width }));
        }
    }
    None
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HalveOrDoubleStep {
    Shl1,
    LShr1,
    AShr1,
    SDiv2,
}

fn halve_or_double_by_two(lines: &[&str], id: &str) -> Option<(String, HalveOrDoubleStep)> {
    let line = lines
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(id))?;
    for (op, step, amount) in [
        ("shl", HalveOrDoubleStep::Shl1, "1"),
        ("lshr", HalveOrDoubleStep::LShr1, "1"),
        ("ashr", HalveOrDoubleStep::AShr1, "1"),
        ("sdiv", HalveOrDoubleStep::SDiv2, "2"),
    ] {
        let needle = format!(" = {op} ");
        if let Some((_, rhs)) = line.split_once(&needle) {
            let mut parts = rhs.split_whitespace();
            let _ty = loop {
                let part = parts.next()?;
                if matches!(part, "i16" | "i32" | "i64") {
                    break part;
                }
            };
            let lhs = parts.next()?.trim_end_matches(',').trim_start_matches('%');
            let actual_amount = parts.next()?.trim_end_matches(',');
            if actual_amount == amount {
                return Some((lhs.to_string(), step));
            }
        }
    }
    None
}

fn unsigned_right_shift_const(lines: &[&str], id: &str) -> Option<(String, u32, u32)> {
    let line = lines
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(id))?;
    let (_, rhs) = line.split_once(" = lshr ")?;
    let mut parts = rhs.split_whitespace();
    let ty = loop {
        let part = parts.next()?;
        if int_type_width(part).is_some() {
            break part;
        }
    };
    let bit_width = int_type_width(ty)?;
    let lhs = parts.next()?.trim_end_matches(',').trim_start_matches('%');
    let shift = parts.next()?.trim_end_matches(',').parse::<u32>().ok()?;
    Some((lhs.to_string(), shift, bit_width))
}

fn symbolic_right_shift_trips_are_small(
    lines: &[&str],
    induction: &str,
    step: HalveOrDoubleStep,
    bit_width: u32,
    backedge_on_true: bool,
) -> bool {
    if step != HalveOrDoubleStep::LShr1 || backedge_on_true {
        return false;
    }
    let Some((limit_base, limit)) = affine_small_const(lines, induction) else {
        return false;
    };
    if !limit_base.is_empty() || limit <= 0 {
        return false;
    }
    (1..=256).contains(&bit_width)
}

fn power_of_two_trip_count(
    start: i32,
    limit: i32,
    step: HalveOrDoubleStep,
    backedge_on_true: bool,
) -> Option<u32> {
    let mut value = u32::try_from(start).ok()?;
    let limit = u32::try_from(limit).ok()?;
    let mut trips = 0u32;
    loop {
        trips = trips.checked_add(1)?;
        if trips > 256 {
            return None;
        }
        let cond = value < limit;
        if cond != backedge_on_true {
            return Some(trips);
        }
        value = match step {
            HalveOrDoubleStep::Shl1 => value.checked_mul(2)?,
            HalveOrDoubleStep::LShr1 => {
                if value == 0 {
                    return None;
                }
                value / 2
            }
            HalveOrDoubleStep::AShr1 => return None,
            HalveOrDoubleStep::SDiv2 => return None,
        };
    }
}

fn power_of_two_trip_count_from_values(
    start: &str,
    limit: &str,
    step: HalveOrDoubleStep,
    backedge_on_true: bool,
    values: &HashMap<String, i128>,
) -> Option<u32> {
    let start = i32::try_from(value_as_const(start, values)?).ok()?;
    let limit = i32::try_from(value_as_const(limit, values)?).ok()?;
    power_of_two_trip_count(start, limit, step, backedge_on_true)
}

fn power_of_two_trip_count_from_values_after_step(
    start: &str,
    limit: &str,
    step: HalveOrDoubleStep,
    backedge_on_true: bool,
    values: &HashMap<String, i128>,
) -> Option<u32> {
    let mut value = u32::try_from(value_as_const(start, values)?).ok()?;
    let limit = u32::try_from(value_as_const(limit, values)?).ok()?;
    for trips in 1..=256 {
        value = match step {
            HalveOrDoubleStep::Shl1 => value.checked_mul(2)?,
            HalveOrDoubleStep::LShr1 => {
                if value == 0 {
                    return None;
                }
                value / 2
            }
            HalveOrDoubleStep::AShr1 => return None,
            HalveOrDoubleStep::SDiv2 => return None,
        };
        let cond = value < limit;
        if cond != backedge_on_true {
            return Some(trips);
        }
    }
    None
}

fn power_of_two_trip_count_from_values_after_step_icmp(
    start: &str,
    limit: &str,
    step: HalveOrDoubleStep,
    pred: &str,
    recur_on_lhs: bool,
    backedge_on_true: bool,
    values: &HashMap<String, i128>,
) -> Option<u32> {
    let mut value = value_as_const(start, values)?;
    let limit = value_as_const(limit, values)?;
    for trips in 1..=256 {
        value = match step {
            HalveOrDoubleStep::Shl1 => value.checked_mul(2)?,
            HalveOrDoubleStep::LShr1 => {
                if value == 0 {
                    return None;
                }
                value / 2
            }
            HalveOrDoubleStep::AShr1 => value >> 1,
            HalveOrDoubleStep::SDiv2 => return None,
        };
        let (lhs, rhs) = if recur_on_lhs {
            (value, limit)
        } else {
            (limit, value)
        };
        let cond = eval_icmp(pred, lhs, rhs)?;
        if cond != backedge_on_true {
            return Some(trips);
        }
    }
    None
}

fn masked_double_by_two(lines: &[&str], id: &str) -> Option<(String, u128)> {
    let line = lines
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(id))?;
    let (_, rhs) = line.split_once(" = and ")?;
    let mut parts = rhs.split_whitespace();
    let ty = parts.next()?;
    int_type_width(ty)?;
    let lhs = parts.next()?.trim_end_matches(',').trim_start_matches('%');
    let mask = parts.next()?.trim_end_matches(',').parse::<u128>().ok()?;
    let (base, step) = halve_or_double_by_two(lines, lhs)?;
    (step == HalveOrDoubleStep::Shl1 && mask != 0).then_some((base, mask))
}

fn masked_doubling_until_exceeds_const_trips_are_small(
    start: &str,
    recur: &str,
    cond: (&str, &str, &str),
    backedge_on_true: bool,
    mask: u128,
    bit_width: u32,
    values: &HashMap<String, i128>,
) -> bool {
    if backedge_on_true {
        return false;
    }
    let (lhs, pred, rhs) = cond;
    let limit = if lhs == recur && pred == "ugt" {
        rhs
    } else if rhs == recur && pred == "ult" {
        lhs
    } else {
        return false;
    };
    let Some(start) = value_as_const(start, values) else {
        return false;
    };
    let Some(limit) = value_as_const(limit, values) else {
        return false;
    };
    if start <= 0 || limit < 0 {
        return false;
    }
    let width_mask = unsigned_width_mask(bit_width);
    let limit = (limit as u128) & width_mask;
    let mut value = (start as u128) & width_mask;
    let mut seen = HashSet::new();
    for _ in 1..=256 {
        let Some(doubled) = value.checked_mul(2) else {
            return false;
        };
        let next = doubled & mask & width_mask;
        if next > limit {
            return true;
        }
        if !seen.insert(next) {
            return false;
        }
        value = next;
    }
    false
}

fn unsigned_width_mask(bit_width: u32) -> u128 {
    if bit_width >= 128 {
        u128::MAX
    } else {
        (1_u128 << bit_width) - 1
    }
}

fn affine_small_const(lines: &[&str], id: &str) -> Option<(String, i32)> {
    if let Ok(value) = id.parse::<i32>() {
        return Some(("".into(), value));
    }
    let Some(line) = lines
        .iter()
        .copied()
        .find(|line| result_name(line).as_deref() == Some(id))
    else {
        return Some((id.to_string(), 0));
    };
    for op in ["or", "add"] {
        let needle = format!(" = {op} i32 ");
        if let Some((_, rhs)) = line.split_once(&needle) {
            let (a, b) = rhs.split_once(',')?;
            let a = a.trim().trim_start_matches('%');
            let b = b.trim().trim_start_matches('%');
            if let Ok(value) = b.parse::<i32>() {
                return Some((a.to_string(), value));
            }
            if let Ok(value) = a.parse::<i32>() {
                return Some((b.to_string(), value));
            }
        }
    }
    Some((id.to_string(), 0))
}

fn result_name(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix('%')?;
    let (name, _) = rest.split_once(" = ")?;
    Some(name.to_string())
}

fn block_lines<'a>(body: &'a [&'a str], block: &'a Block) -> impl Iterator<Item = &'a str> {
    body[block.start..=block.end].iter().copied()
}

fn trailing_small_const(line: &str) -> Option<i32> {
    let value = line.rsplit_once(',')?.1.trim();
    value.parse().ok()
}

/// Replace a single `label %<from>` branch target with `label %<to>`. Returns `None` if `<from>`
/// is not present exactly once as a target (ambiguous / degenerate → caller quarantines).
fn replace_branch_target(line: &str, from: &str, to: &str) -> Option<String> {
    let needle = format!("label %{from}");
    let mut matches = Vec::new();
    let mut search_from = 0;
    while let Some(pos) = line[search_from..].find(&needle) {
        let abs = search_from + pos;
        let after = abs + needle.len();
        // Word boundary: the char after the label name must not continue an identifier.
        let boundary = line[after..]
            .chars()
            .next()
            .map(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '$' || c == '"'))
            .unwrap_or(true);
        if boundary {
            matches.push(abs);
        }
        search_from = after;
    }
    if matches.len() != 1 {
        return None;
    }
    let abs = matches[0];
    let mut out = String::with_capacity(line.len() + to.len());
    out.push_str(&line[..abs]);
    out.push_str(&format!("label %{to}"));
    out.push_str(&line[abs + needle.len()..]);
    Some(out)
}

fn split_loop_metadata(line: &str) -> (String, String) {
    if let Some(pos) = line.find(", !llvm.loop") {
        (line[..pos].to_string(), line[pos..].to_string())
    } else {
        (line.to_string(), String::new())
    }
}

/// Rename a phi's predecessor block `%<from>` → `%<to>`. The predecessor is the token immediately
/// before `]` in each `[ value, %pred ]` incoming, so only predecessor slots are touched.
fn rename_phi_pred(line: &str, from: &str, to: &str) -> String {
    let needle = format!("%{from}");
    let mut out = String::with_capacity(line.len() + to.len());
    let mut rest = line;
    while let Some(pos) = rest.find(&needle) {
        let after = pos + needle.len();
        // Boundary after the name, then (spaces) `]` → this is a predecessor slot.
        let tail = &rest[after..];
        let next = tail.chars().next();
        let is_boundary = next
            .map(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '$' || c == '"'))
            .unwrap_or(true);
        let is_pred = is_boundary && tail.trim_start().starts_with(']');
        out.push_str(&rest[..pos]);
        if is_pred {
            out.push('%');
            out.push_str(to);
        } else {
            out.push_str(&needle);
        }
        rest = &rest[after..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instrumented(text: &str, entry: &str) -> String {
        match classify_and_instrument(text, entry) {
            GuardPlan::Instrumented(s) => s,
            other => panic!("expected Instrumented, got {other:?}"),
        }
    }

    #[test]
    fn loop_free_is_unchanged() {
        let ll = "\
define void @k(ptr addrspace(1) %0) {
  %2 = load i32, ptr addrspace(1) %0, align 4
  ret void
}
";
        assert_eq!(classify_and_instrument(ll, "k"), GuardPlan::LoopFree);
        assert_eq!(
            reachable_module_has_cfg_cycle_with_loop_input_facts(
                ll,
                "k",
                LoopInputFacts::default(),
            ),
            Ok(false)
        );
    }

    #[test]
    fn self_loop_is_instrumented_and_bounded() {
        let ll = "\
define void @spin(ptr addrspace(1) %0) {
  br label %1
1:
  br label %1
}
";
        let out = instrumented(ll, "spin");
        assert!(
            out.contains("%m2v.bd = alloca i32, align 4"),
            "budget alloca missing:\n{out}"
        );
        assert!(
            out.contains(&format!("store i32 {LOOP_BUDGET_BACKEDGES}, ptr %m2v.bd")),
            "budget init missing:\n{out}"
        );
        assert!(out.contains("m2v.g.0:"), "guard block missing:\n{out}");
        assert!(
            out.contains("icmp sle i32 %m2v.0.b, 0"),
            "budget check missing:\n{out}"
        );
        assert!(out.contains("m2v.exit:"), "exit block missing:\n{out}");
        assert!(out.contains("ret void"), "void exit missing:\n{out}");
        // The self-branch now targets the guard, and the guard returns to the loop head.
        assert!(
            out.contains("br label %m2v.g.0"),
            "back-edge not redirected:\n{out}"
        );
        assert!(
            out.contains("br i1 %m2v.0.c, label %m2v.exit, label %1"),
            "guard exit branch missing:\n{out}"
        );
    }

    #[test]
    fn explicit_budget_is_embedded_and_must_be_positive() {
        let ll = "\
define void @spin(ptr addrspace(1) %0) {
  br label %1
1:
  br label %1
}
";
        let out = match classify_and_instrument_with_loop_input_facts_and_budget(
            ll,
            "spin",
            LoopInputFacts::default(),
            7,
        ) {
            GuardPlan::Instrumented(text) => text,
            other => panic!("expected Instrumented, got {other:?}"),
        };
        assert!(out.contains("store i32 7, ptr %m2v.bd"), "{out}");
        assert_eq!(
            classify_and_instrument_with_loop_input_facts_and_budget(
                ll,
                "spin",
                LoopInputFacts::default(),
                0,
            ),
            GuardPlan::Quarantine("loop back-edge budget must be positive".into())
        );
    }

    #[test]
    fn exact_raw_byte_field_proves_one_iteration_loop_bounded() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %source) {
entry:
  %count.ptr = getelementptr inbounds i8, ptr addrspace(1) %source, i64 24
  %count.cast = bitcast ptr addrspace(1) %count.ptr to ptr addrspace(1)
  %count = load i32, ptr addrspace(1) %count.cast, align 4
  br label %loop
loop:
  %remaining = phi i32 [ %count, %entry ], [ %next, %loop ]
  %next = sub i32 %remaining, 1
  %done = icmp eq i32 %next, 0
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
"#;
        let byte_values = [("source".to_string(), 24, 1)];
        let facts_input = LoopInputFacts {
            arg_byte_values: &byte_values,
            ..LoopInputFacts::default()
        };
        let lines = ll.lines().collect::<Vec<_>>();
        let facts = LoopFacts::from_module(&lines, &facts_input);
        let function = find_functions(&lines).remove(0);
        let values =
            consteval_int_values(&lines[function.define_idx + 1..function.close_idx], &facts);
        assert_eq!(values.get("count"), Some(&1), "{values:?}");
        assert_eq!(
            classify_and_instrument_with_loop_input_facts(ll, "kernel", facts_input,),
            GuardPlan::LoopFree
        );
    }

    #[test]
    fn exact_raw_header_proves_min_chunk_descending_loop_bounded() {
        let ll = r#"
define void @kernel(ptr addrspace(1) %source) {
entry:
  %count.ptr = getelementptr inbounds i8, ptr addrspace(1) %source, i64 24
  %count = load i32, ptr addrspace(1) %count.ptr, align 4
  %wide = zext i32 %count to i64
  %bytes = shl i64 %wide, 4
  br label %loop
loop:
  %remaining = phi i64 [ %bytes, %entry ], [ %next, %loop ]
  %chunk = tail call i64 @air.min.u.i64(i64 %remaining, i64 1024)
  %next = sub i64 %remaining, %chunk
  %done = icmp eq i64 %next, 0
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
declare i64 @air.min.u.i64(i64, i64)
"#;
        let byte_values = [("source".to_string(), 24, 1)];
        let facts_input = LoopInputFacts {
            arg_byte_values: &byte_values,
            ..LoopInputFacts::default()
        };
        let lines = ll.lines().collect::<Vec<_>>();
        let facts = LoopFacts::from_module(&lines, &facts_input);
        let function = find_functions(&lines).remove(0);
        let body = &lines[function.define_idx + 1..function.close_idx];
        let values = consteval_int_values(body, &facts);
        assert_eq!(values.get("bytes"), Some(&16), "{values:?}");
        assert_eq!(
            classify_and_instrument_with_loop_input_facts(ll, "kernel", facts_input,),
            GuardPlan::LoopFree
        );
    }

    #[test]
    fn phi_predecessor_is_renamed_on_split_edge() {
        // Loop header `%loop` has a phi over {entry, latch}; the latch back-edge must have its phi
        // predecessor rewritten to the guard, while the entry incoming is untouched.
        let ll = "\
define void @counted(ptr addrspace(1) %0) {
  br label %loop
loop:
  %i = phi i32 [ 0, %1 ], [ %next, %latch ]
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, 100
  br i1 %done, label %exit, label %latch
latch:
  br label %loop
exit:
  ret void
}
";
        let out = instrumented(ll, "counted");
        // Entry incoming preserved…
        assert!(
            out.contains("[ 0, %1 ]"),
            "entry phi incoming changed:\n{out}"
        );
        // …latch incoming redirected to the guard block.
        assert!(
            out.contains("[ %next, %m2v.g.0 ]"),
            "latch phi pred not renamed:\n{out}"
        );
        assert!(
            !out.contains("[ %next, %latch ]"),
            "stale latch phi incoming:\n{out}"
        );
        assert!(
            out.contains("br label %m2v.g.0"),
            "latch back-edge not redirected:\n{out}"
        );
    }

    #[test]
    fn loop_metadata_backedge_is_guarded_in_place() {
        let ll = "\
define void @counted(ptr addrspace(1) %0) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, 1000
  br i1 %done, label %exit, label %loop, !llvm.loop !0
exit:
  ret void
}

!0 = distinct !{!0}
";
        let out = instrumented(ll, "counted");
        assert!(
            !out.contains("m2v.g.0:"),
            "metadata-bearing loop should not gain a synthetic back-edge predecessor:\n{out}"
        );
        assert!(
            out.contains("  %m2v.0.ex = icmp sle i32 %m2v.0.b, 0"),
            "budget exhaustion check missing:\n{out}"
        );
        assert!(
            out.contains("  %m2v.0.leave = or i1 %done, %m2v.0.ex"),
            "budget condition not composed with original exit condition:\n{out}"
        );
        assert!(
            out.contains("br i1 %m2v.0.leave, label %exit, label %loop, !llvm.loop !0"),
            "loop metadata back-edge should remain on the real header branch:\n{out}"
        );
    }

    #[test]
    fn nested_loops_guard_every_back_edge() {
        let ll = "\
define void @nested(ptr addrspace(1) %0) {
  br label %outer
outer:
  br label %inner
inner:
  %c = icmp eq i32 0, 0
  br i1 %c, label %inner, label %outer
}
";
        let out = instrumented(ll, "nested");
        assert!(out.contains("m2v.g.0:"), "first guard missing:\n{out}");
        assert!(out.contains("m2v.g.1:"), "second guard missing:\n{out}");
    }

    #[test]
    fn switch_back_edge_is_quarantined() {
        let ll = "\
define void @sw(ptr addrspace(1) %0) {
  br label %head
head:
  switch i32 0, label %exit [
    i32 0, label %head
  ]
exit:
  ret void
}
";
        match classify_and_instrument(ll, "sw") {
            GuardPlan::Quarantine(_) => {}
            other => panic!("expected Quarantine for switch back-edge, got {other:?}"),
        }
    }

    #[test]
    fn loop_with_workgroup_barrier_is_quarantined() {
        let ll = "\
define void @barrier_loop(ptr addrspace(1) %0) {
  br label %loop
loop:
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %loop
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "barrier_loop") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("air.wg.barrier"), "{msg}"),
            other => panic!("expected Quarantine for barrier loop, got {other:?}"),
        }
    }

    #[test]
    fn small_counted_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @barrier_counted(ptr addrspace(1) %0) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %n, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %n = add i32 %i, 1
  %done = icmp eq i32 %n, 4
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "barrier_counted") {
            GuardPlan::LoopFree => {}
            other => panic!("expected small barrier loop to be treated as bounded, got {other:?}"),
        }
    }

    #[test]
    fn counted_barrier_loop_accepts_gt_one_entry_guard() {
        let ll = "\
define void @barrier_counted_gt_guard(i32 %limit) {
entry:
  %run = icmp ugt i32 %limit, 1
  br i1 %run, label %loop, label %exit
loop:
  %i = phi i32 [ 1, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        let arg_upper_bounds = [("limit".to_string(), 16)];
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "barrier_counted_gt_guard",
            LoopInputFacts {
                arg_upper_bounds: &arg_upper_bounds,
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => panic!("expected gt-guarded small barrier loop to be bounded, got {other:?}"),
        }
    }

    #[test]
    fn positive_step_slt_barrier_loop_uses_input_upper_bound() {
        let ll = "\
define void @distance_like(i32 %height) {
entry:
  %run = icmp sgt i32 %height, 0
  br i1 %run, label %loop, label %exit
loop:
  %y = phi i32 [ 0, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %advanced = add nsw i32 %y, 32
  %wide = shl i32 %advanced, 16
  %next = ashr exact i32 %wide, 16
  %more = icmp slt i32 %next, %height
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        let arg_upper_bounds = [("height".to_string(), 8)];
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "distance_like",
            LoopInputFacts {
                arg_upper_bounds: &arg_upper_bounds,
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!(
                    "expected input-bound positive-step barrier loop to be bounded, got {other:?}"
                )
            }
        }
    }

    #[test]
    fn imageblock_extent_facts_prove_masked_barrier_tile_loop() {
        let ll = "\
define void @imageblock_tile(<2 x i16> %gid) {
entry:
  %masked = and <2 x i16> %gid, splat (i16 15)
  %x = extractelement <2 x i16> %masked, i64 0
  %width = tail call i16 @air.get_imageblock_width()
  br label %loop
loop:
  %i = phi i16 [ %x, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = add i16 %i, 16
  %more = icmp ult i16 %next, %width
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare i16 @air.get_imageblock_width()
declare void @air.wg.barrier(i32, i32)
";
        let arg_vector_upper_bounds = [("gid".to_string(), 0, 7)];
        let lines = ll.lines().collect::<Vec<_>>();
        let facts = LoopFacts::from_module(
            &lines,
            &LoopInputFacts {
                arg_vector_upper_bounds: &arg_vector_upper_bounds,
                imageblock_extent: Some([8, 8]),
                ..LoopInputFacts::default()
            },
        );
        assert_eq!(
            small_integer_upper_bound_with_facts(&lines, &facts, "x", 16),
            Some(7)
        );
        assert_eq!(
            small_integer_upper_bound_with_facts(&lines, &facts, "width", 16),
            Some(8)
        );
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "imageblock_tile",
            LoopInputFacts {
                arg_vector_upper_bounds: &arg_vector_upper_bounds,
                imageblock_extent: Some([8, 8]),
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => panic!("expected imageblock-bound barrier loop to be bounded, got {other:?}"),
        }
    }

    #[test]
    fn launch_threads_prove_mtgp32_descending_barrier_loop() {
        let ll = "\
define void @mtgp32_like(i16 %threads, i32 %n, i32 %state) {
entry:
  %step = zext i16 %threads to i32
  %masked = and i32 %state, 65535
  %span = sub nsw i32 351, %masked
  %limit = tail call i32 @air.min.u.i32(i32 %n, i32 %span)
  %small = icmp ult i32 %limit, %step
  br i1 %small, label %exit, label %preheader
preheader:
  br label %loop
loop:
  %remaining = phi i32 [ %limit, %preheader ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = sub i32 %remaining, %step
  %done = icmp ult i32 %next, %step
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare i32 @air.min.u.i32(i32, i32)
declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "mtgp32_like",
            LoopInputFacts {
                arg_values: &[("threads".into(), 64)],
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => panic!("expected mtgp32-style barrier loop to be bounded, got {other:?}"),
        }
    }

    #[test]
    fn exact_input_prunes_phi_selected_barrier_loop_path() {
        let ll = "\
define void @phi_pruned(i32 %state) {
entry:
  %inactive = icmp eq i32 %state, 0
  br i1 %inactive, label %inactive_path, label %active_path
inactive_path:
  br label %join
active_path:
  br label %join
join:
  %count = phi i32 [ 0, %inactive_path ], [ 1000, %active_path ]
  %done = icmp eq i32 %count, 0
  br i1 %done, label %exit, label %loop
loop:
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %loop
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "phi_pruned",
            LoopInputFacts {
                arg_values: &[("state".into(), 0)],
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected exact input to prune phi-selected barrier loop, got {other:?}")
            }
        }
    }

    #[test]
    fn launch_lane_with_fc_stride_ult_barrier_loop_is_left_unguarded() {
        let ll = "\
@_Z21threadsPerThreadgroup.MTL_FC_INIT_11_j = internal unnamed_addr addrspace(2) externally_initialized constant i32 undef, section \"air.fc_initializer\", align 4
@_ZL21threadsPerThreadgroup = internal unnamed_addr addrspace(2) global i32 undef, align 4

define internal void @_GLOBAL__sub_I_test() section \"air.static_init\" {
  %t = load i32, ptr addrspace(2) @_Z21threadsPerThreadgroup.MTL_FC_INIT_11_j, align 4
  store i32 %t, ptr addrspace(2) @_ZL21threadsPerThreadgroup, align 4
  ret void
}

define void @wide_bins_like(i32 %lid) {
entry:
  %active = icmp ult i32 %lid, 48
  br i1 %active, label %preheader, label %exit
preheader:
  %0 = load i32, ptr addrspace(2) @_ZL21threadsPerThreadgroup, align 4
  br label %loop
loop:
  %i = phi i32 [ %lid, %preheader ], [ %1, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %1 = add i32 %i, %0
  %2 = icmp ult i32 %1, 48
  br i1 %2, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "wide_bins_like",
            LoopInputFacts {
                fc_values: &[(11, 2)],
                arg_upper_bounds: &[("lid".into(), 63)],
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected launch lane and FC stride to prove barrier loop, got {other:?}")
            }
        }
    }

    #[test]
    fn same_value_stride_phi_proves_launch_lane_barrier_loop() {
        let ll = "\
@_Z21threadsPerThreadgroup.MTL_FC_INIT_11_j = internal unnamed_addr addrspace(2) externally_initialized constant i32 undef, section \"air.fc_initializer\", align 4
@_ZL21threadsPerThreadgroup = internal unnamed_addr addrspace(2) global i32 undef, align 4

define internal void @_GLOBAL__sub_I_test() section \"air.static_init\" {
  %t = load i32, ptr addrspace(2) @_Z21threadsPerThreadgroup.MTL_FC_INIT_11_j, align 4
  store i32 %t, ptr addrspace(2) @_ZL21threadsPerThreadgroup, align 4
  ret void
}

define void @same_value_stride_phi(i32 %lid) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %latch ]
  %s0 = load i32, ptr addrspace(2) @_ZL21threadsPerThreadgroup, align 4
  %idx = add i32 %i, %lid
  %active = icmp ult i32 %idx, 48
  br i1 %active, label %body, label %latch
body:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %s1 = load i32, ptr addrspace(2) @_ZL21threadsPerThreadgroup, align 4
  br label %latch
latch:
  %stride = phi i32 [ %s0, %loop ], [ %s1, %body ]
  %next = add i32 %i, %stride
  %more = icmp ult i32 %next, 48
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "same_value_stride_phi",
            LoopInputFacts {
                fc_values: &[(11, 2)],
                arg_upper_bounds: &[("lid".into(), 63)],
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected same-value stride phi to prove barrier loop, got {other:?}")
            }
        }
    }

    #[test]
    fn numeric_ssa_icmp_uses_facts_before_literal_parse_for_reachability() {
        let ll = "\
%struct.Params = type { i32 }

define void @fact_pruned_loop(ptr addrspace(2) %params, ptr addrspace(3) %tg, <2 x i16> %lid) {
  %\"m2v.zi.arg1\" = bitcast i32 addrspace(3)* %tg to [512 x i32] addrspace(3)*
  store [512 x i32] zeroinitializer, [512 x i32] addrspace(3)* %\"m2v.zi.arg1\", align 4
  %9 = extractelement <2 x i16> %lid, i64 1
  %10 = zext i16 %9 to i32
  %11 = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 0
  %12 = load i32, ptr addrspace(2) %11, align 4
  %13 = icmp sgt i32 %12, %10
  br i1 %13, label %enter, label %exit
enter:
  br label %loop
loop:
  %14 = phi i32 [ 0, %enter ], [ %15, %loop ]
  tail call fastcc void @spin()
  %15 = add i32 %14, 1
  %16 = icmp slt i32 %15, %12
  br i1 %16, label %loop, label %exit
exit:
  ret void
}

define internal fastcc void @spin() {
  br label %loop
loop:
  br label %loop
}
";
        let arg_field_values = [("params".to_string(), vec![0], 0)];
        let arg_vector_values = [("lid".to_string(), 1, 0)];
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "fact_pruned_loop",
            LoopInputFacts {
                arg_field_values: &arg_field_values,
                arg_vector_values: &arg_vector_values,
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected facts to prune entry loop before compose gate, got {other:?}")
            }
        }
    }

    #[test]
    fn struct_field_call_bound_proves_loopy_callee_small() {
        let ll = "\
%struct.Params = type { i32 }

define void @caller(ptr addrspace(2) %params) {
entry:
  br label %loop
loop:
  %limit_ptr = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 0
  %limit = load i32, ptr addrspace(2) %limit_ptr, align 4
  tail call fastcc void @bounded_callee(i32 %limit)
  br label %loop
}

define internal fastcc void @bounded_callee(i32 %limit) {
entry:
  %nonzero = icmp ne i32 %limit, 0
  br i1 %nonzero, label %loop, label %exit
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
";
        let arg_field_values = [("params".to_string(), vec![0], 16)];
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "caller",
            LoopInputFacts {
                arg_field_values: &arg_field_values,
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::Instrumented(_) => {}
            other => {
                panic!("expected params-field call bound to prove callee small, got {other:?}")
            }
        }
    }

    #[test]
    fn vector_actual_and_float_bounds_prove_loopy_callee_small() {
        let ll = "\
%struct.Params = type { i16 }

define void @caller(ptr addrspace(2) %params, i32 %x, i32 %y) {
entry:
  br label %loop
loop:
  %coord0 = insertelement <2 x i32> undef, i32 %x, i64 0
  %coord = insertelement <2 x i32> %coord0, i32 %y, i64 1
  tail call fastcc void @bounded_float_callee(ptr addrspace(2) %params, <2 x i32> %coord)
  br label %loop
}

define internal fastcc void @bounded_float_callee(ptr addrspace(2) %params, <2 x i32> %coord) {
entry:
  %yval = extractelement <2 x i32> %coord, i64 1
  %yf = tail call fast float @air.convert.f.f32.s.i32(i32 %yval)
  %kptr = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 0
  %k = load i16, ptr addrspace(2) %kptr, align 2
  %kf = tail call fast float @air.convert.f.f32.u.i16(i16 %k)
  %half = fmul fast float %kf, 5.000000e-01
  %lo_half = tail call fast float @air.fast_floor.f32(float %half)
  %lo_f = fsub fast float %yf, %lo_half
  %lo = tail call i32 @air.convert.s.i32.f.f32(float %lo_f)
  %km1 = fadd fast float %kf, -1.000000e+00
  %hi_half_raw = fmul fast float %km1, 5.000000e-01
  %hi_half = tail call fast float @air.fast_floor.f32(float %hi_half_raw)
  %yp1 = fadd fast float %yf, 1.000000e+00
  %hi_f = fadd fast float %yp1, %hi_half
  %hi = tail call i32 @air.convert.s.i32.f.f32(float %hi_f)
  %run = icmp slt i32 %lo, %hi
  br i1 %run, label %loop, label %exit
loop:
  %i = phi i32 [ %lo, %entry ], [ %next, %loop ]
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %hi
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare float @air.convert.f.f32.s.i32(i32)
declare float @air.convert.f.f32.u.i16(i16)
declare float @air.fast_floor.f32(float)
declare i32 @air.convert.s.i32.f.f32(float)
";
        let arg_field_values = [("params".to_string(), vec![0], 8)];
        let arg_upper_bounds = [("x".to_string(), 7), ("y".to_string(), 7)];
        let lines = ll.lines().collect::<Vec<_>>();
        let facts = LoopFacts::from_module(
            &lines,
            &LoopInputFacts {
                arg_field_values: &arg_field_values,
                arg_upper_bounds: &arg_upper_bounds,
                ..LoopInputFacts::default()
            },
        );
        assert_eq!(
            facts.arg_vector_upper_bounds.get(&("coord".into(), 1)),
            Some(&7)
        );
        let callee = find_functions(&lines)
            .into_iter()
            .find(|func| func.name == "bounded_float_callee")
            .unwrap();
        let body = &lines[callee.define_idx + 1..callee.close_idx];
        assert_eq!(
            small_integer_lower_bound_with_facts(body, &facts, "lo", 32),
            Some(-4)
        );
        assert_eq!(
            small_integer_upper_bound_with_facts(body, &facts, "hi", 32),
            Some(11)
        );
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "caller",
            LoopInputFacts {
                arg_field_values: &arg_field_values,
                arg_upper_bounds: &arg_upper_bounds,
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::Instrumented(_) => {}
            other => {
                panic!(
                    "expected vector actual and float bounds to prove callee small, got {other:?}"
                )
            }
        }
    }

    #[test]
    fn scalar_float_arg_and_unsigned_float_convert_prove_loopy_callee_small() {
        let ll = "\
define void @caller(ptr addrspace(2) %rank_mode) {
entry:
  br label %loop
loop:
  tail call fastcc void @bounded_float_callee(ptr addrspace(2) %rank_mode)
  br label %loop
}

define internal fastcc void @bounded_float_callee(ptr addrspace(2) %rank_mode) {
entry:
  %rank = load float, ptr addrspace(2) %rank_mode, align 4
  %scaled = fmul fast float %rank, 1.285000e+03
  %raw = tail call i32 @air.convert.u.i32.f.f32(float %scaled)
  %at_least_one = tail call i32 @air.max.u.i32(i32 %raw, i32 1)
  %limit = tail call i32 @air.min.u.i32(i32 %at_least_one, i32 1285)
  %nonzero = icmp ne i32 %limit, 0
  br i1 %nonzero, label %loop, label %exit
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add nuw i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare i32 @air.convert.u.i32.f.f32(float)
declare i32 @air.max.u.i32(i32, i32)
declare i32 @air.min.u.i32(i32, i32)
";
        let arg_float_values = [("rank_mode".to_string(), 0.0)];
        let lines = ll.lines().collect::<Vec<_>>();
        let facts = LoopFacts::from_module(
            &lines,
            &LoopInputFacts {
                arg_float_values: &arg_float_values,
                ..LoopInputFacts::default()
            },
        );
        let callee = find_functions(&lines)
            .into_iter()
            .find(|func| func.name == "bounded_float_callee")
            .unwrap();
        let body = &lines[callee.define_idx + 1..callee.close_idx];
        assert_eq!(
            small_integer_upper_bound_with_facts(body, &facts, "limit", 32),
            Some(1)
        );
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "caller",
            LoopInputFacts {
                arg_float_values: &arg_float_values,
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::Instrumented(_) => {}
            other => {
                panic!("expected scalar float arg bound to prove callee small, got {other:?}")
            }
        }
    }

    #[test]
    fn fc_consteval_barrier_loop_is_left_unguarded_when_requested_value_is_small() {
        let ll = r#"
@_Z2fc.MTL_FC_INIT_3_t = internal unnamed_addr addrspace(2) externally_initialized constant i16 undef, section "air.fc_initializer", align 2
@_ZL2fc.1 = internal unnamed_addr addrspace(2) global i16 undef, align 2

define internal void @init() section "air.static_init" {
  %v = load i16, ptr addrspace(2) @_Z2fc.MTL_FC_INIT_3_t, align 2
  store i16 %v, ptr addrspace(2) @_ZL2fc.1, align 2
  ret void
}

define void @barrier_fc_bound() {
entry:
  %raw = load i16, ptr addrspace(2) @_ZL2fc.1, align 2
  %wide = zext i16 %raw to i32
  %limit = add i32 %wide, 3
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
"#;
        match classify_and_instrument(ll, "barrier_fc_bound") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("air.wg.barrier"), "{msg}"),
            other => {
                panic!("expected default FC-unknown barrier loop to quarantine, got {other:?}")
            }
        }
        match classify_and_instrument_with_function_constants(ll, "barrier_fc_bound", &[(3, 1)]) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected requested FC value to prove small barrier loop, got {other:?}")
            }
        }
    }

    #[test]
    fn fc_consteval_select_conjoined_barrier_loop_is_left_unguarded() {
        let ll = r#"
@_Z5limit.MTL_FC_INIT_11_j = internal unnamed_addr addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@_ZL5limit = internal unnamed_addr addrspace(2) global i32 undef, align 4

define internal void @init() section "air.static_init" {
  %v = load i32, ptr addrspace(2) @_Z5limit.MTL_FC_INIT_11_j, align 4
  store i32 %v, ptr addrspace(2) @_ZL5limit, align 4
  ret void
}

define void @barrier_fc_select_bound() {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = add nuw nsw i32 %i, 16
  %limit = load i32, ptr addrspace(2) @_ZL5limit, align 4
  %under_fc = icmp ult i32 %next, %limit
  %under_static = icmp ult i32 %next, 48
  %more = select i1 %under_fc, i1 %under_static, i1 false
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
"#;
        match classify_and_instrument(ll, "barrier_fc_select_bound") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("air.wg.barrier"), "{msg}"),
            other => {
                panic!("expected default FC-unknown barrier loop to quarantine, got {other:?}")
            }
        }
        match classify_and_instrument_with_function_constants(
            ll,
            "barrier_fc_select_bound",
            &[(11, 1)],
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!(
                    "expected requested FC value to prove select-bound barrier loop, got {other:?}"
                )
            }
        }
    }

    #[test]
    fn symbolic_select_conjoined_stride_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @barrier_select_stride(ptr addrspace(2) %limit_ptr) {
entry:
  %limit = load i32, ptr addrspace(2) %limit_ptr, align 4
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = add nuw nsw i32 %i, 16
  %under_runtime = icmp ult i32 %next, %limit
  %under_static = icmp ult i32 %next, 48
  %more = select i1 %under_runtime, i1 %under_static, i1 false
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "barrier_select_stride") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("air.wg.barrier"), "{msg}"),
            other => {
                panic!("expected input-unknown select barrier loop to quarantine, got {other:?}")
            }
        }
        match classify_and_instrument_with_input_facts(
            ll,
            "barrier_select_stride",
            &[],
            &[("limit_ptr".into(), 64)],
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected select-conjoined stride barrier loop to be bounded, got {other:?}")
            }
        }
    }

    #[test]
    fn fc_consteval_integer_select_limit_barrier_loop_is_left_unguarded() {
        let ll = r#"
@_Z8batching.MTL_FC_INIT_1_b = internal unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@_Z8fastwide.MTL_FC_INIT_2_b = internal unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@_Z9branching.MTL_FC_INIT_3_j = internal unnamed_addr addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@_ZL8batching = internal unnamed_addr addrspace(2) global i8 undef, align 1
@_ZL8fastwide = internal unnamed_addr addrspace(2) global i8 undef, align 1
@_ZL9branching = internal unnamed_addr addrspace(2) global i32 undef, align 4

define internal void @init() section "air.static_init" {
  %b = load i8, ptr addrspace(2) @_Z8batching.MTL_FC_INIT_1_b, align 1
  store i8 %b, ptr addrspace(2) @_ZL8batching, align 1
  %f = load i8, ptr addrspace(2) @_Z8fastwide.MTL_FC_INIT_2_b, align 1
  store i8 %f, ptr addrspace(2) @_ZL8fastwide, align 1
  %n = load i32, ptr addrspace(2) @_Z9branching.MTL_FC_INIT_3_j, align 4
  store i32 %n, ptr addrspace(2) @_ZL9branching, align 4
  ret void
}

define void @barrier_select_limit() {
entry:
  %batching = load i8, ptr addrspace(2) @_ZL8batching, align 1
  %not_batching = icmp eq i8 %batching, 0
  %fastwide = load i8, ptr addrspace(2) @_ZL8fastwide, align 1
  %not_fastwide = icmp eq i8 %fastwide, 0
  %wide_disabled = select i1 %not_batching, i1 %not_fastwide, i1 false
  %branching = load i32, ptr addrspace(2) @_ZL9branching, align 4
  %branching_minus_two = add i32 %branching, -2
  %limit = select i1 %wide_disabled, i32 %branching_minus_two, i32 1
  br label %loop
loop:
  %i = phi i32 [ 1, %entry ], [ %next, %latch ]
  tail call void @air.wg.barrier(i32 1, i32 1)
  br label %latch
latch:
  %next = add i32 %i, 1
  %done = icmp ugt i32 %next, %limit
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
"#;
        match classify_and_instrument(ll, "barrier_select_limit") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("air.wg.barrier"), "{msg}"),
            other => {
                panic!("expected default FC-unknown barrier loop to quarantine, got {other:?}")
            }
        }
        match classify_and_instrument_with_function_constants(
            ll,
            "barrier_select_limit",
            &[(1, 1), (2, 1), (3, 1)],
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!(
                    "expected FC integer select limit to prove barrier loop bounded, got {other:?}"
                )
            }
        }
    }

    #[test]
    fn zero_fc_unsigned_upper_bound_prunes_unknown_thread_expr_barrier_loop() {
        let ll = r#"
@_Z5limit.MTL_FC_INIT_2_j = internal unnamed_addr addrspace(2) externally_initialized constant i32 undef, section "air.fc_initializer", align 4
@_ZL5limit = internal unnamed_addr addrspace(2) global i32 undef, align 4

define internal void @init() section "air.static_init" {
  %v = load i32, ptr addrspace(2) @_Z5limit.MTL_FC_INIT_2_j, align 4
  store i32 %v, ptr addrspace(2) @_ZL5limit, align 4
  ret void
}

define void @barrier_zero_bound(i32 %tid) {
entry:
  %limit = load i32, ptr addrspace(2) @_ZL5limit, align 4
  %lane = add i32 %tid, 1
  %enter = icmp ult i32 %lane, %limit
  br i1 %enter, label %loop, label %exit
loop:
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %loop
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
"#;
        match classify_and_instrument(ll, "barrier_zero_bound") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("air.wg.barrier"), "{msg}"),
            other => {
                panic!("expected default FC-unknown barrier loop to quarantine, got {other:?}")
            }
        }
        match classify_and_instrument_with_function_constants(ll, "barrier_zero_bound", &[(2, 0)]) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected zero FC bound to make unsigned branch unreachable, got {other:?}")
            }
        }
    }

    #[test]
    fn input_consteval_unreachable_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @barrier_input_bound(ptr addrspace(2) %n) {
entry:
  %v = load i32, ptr addrspace(2) %n, align 4
  %large = icmp sgt i32 %v, 31
  br i1 %large, label %loop, label %tail
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 0, i32 1)
  %limit = sdiv i32 %v, 32
  %next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %next, %limit
  br i1 %done, label %tail, label %loop
tail:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "barrier_input_bound") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("air.wg.barrier"), "{msg}"),
            other => panic!("expected input-unknown barrier loop to quarantine, got {other:?}"),
        }
        match classify_and_instrument_with_input_facts(
            ll,
            "barrier_input_bound",
            &[],
            &[("n".into(), 16)],
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected exact input fact to prove barrier loop unreachable, got {other:?}")
            }
        }
    }

    #[test]
    fn input_struct_field_consteval_unreachable_barrier_loop_is_left_unguarded() {
        let ll = "\
%Inner = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
%Params = type { %Inner, i32 }

define void @barrier_input_field_bound(ptr addrspace(1) %params) {
entry:
  %p = getelementptr inbounds %Params, ptr addrspace(1) %params, i64 0, i32 0, i32 11
  %n = load i32, ptr addrspace(1) %p, align 4
  %run = icmp sgt i32 %n, 31
  br i1 %run, label %loop, label %exit
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %i, 1
  tail call void @air.wg.barrier(i32 0, i32 1)
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "barrier_input_field_bound") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("air.wg.barrier"), "{msg}"),
            other => panic!("expected input-unknown barrier loop to quarantine, got {other:?}"),
        }
        match classify_and_instrument_with_input_facts_and_fields(
            ll,
            "barrier_input_field_bound",
            &[],
            &[],
            &[("params".into(), vec![0, 11], 16)],
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected exact input field fact to prove barrier loop unreachable, got {other:?}")
            }
        }
    }

    #[test]
    fn small_symbolic_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @barrier_prefix(ptr addrspace(1) %0, i32 %base) {
entry:
  br label %setup
setup:
  %start = or i32 %base, 1
  %limit = or i32 %base, 31
  br label %loop
loop:
  %i = phi i32 [ %start, %setup ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = add nuw i32 %i, 1
  %done = icmp eq i32 %i, %limit
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "barrier_prefix") {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected symbolic barrier loop to be treated as bounded, got {other:?}")
            }
        }
    }

    #[test]
    fn barrier_before_loop_is_still_instrumented() {
        let ll = "\
define void @barrier_before_loop(ptr addrspace(1) %0) {
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %loop
loop:
  br label %loop
}

declare void @air.wg.barrier(i32, i32)
";
        let out = instrumented(ll, "barrier_before_loop");
        assert!(out.contains("m2v.g.0:"), "loop guard missing:\n{out}");
    }

    #[test]
    fn small_fixed_trip_loops_are_left_unguarded() {
        let ll = "\
define void @fixed(ptr addrspace(1) %0) {
entry:
  br label %two
two:
  %b = phi i1 [ true, %entry ], [ false, %two ]
  br i1 %b, label %two, label %count
count:
  %i = phi i32 [ 0, %two ], [ %n, %count ]
  %p = inttoptr i64 0 to ptr addrspace(3)
  %old = call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) %p, i32 1, i32 0, i32 1, i1 true)
  %n = add i32 %i, 1
  %done = icmp eq i32 %n, 4
  br i1 %done, label %exit, label %count
exit:
  ret void
}

declare i32 @air.atomic.local.add.u.i32(ptr addrspace(3), i32, i32, i32, i1)
";
        match classify_and_instrument(ll, "fixed") {
            GuardPlan::LoopFree => {}
            other => panic!("expected fixed loops to be treated as bounded, got {other:?}"),
        }
        assert_eq!(
            reachable_module_has_cfg_cycle_with_loop_input_facts(
                ll,
                "fixed",
                LoopInputFacts::default(),
            ),
            Ok(true),
            "semantic safety gate must retain even proven finite CFG cycles"
        );
    }

    #[test]
    fn small_workgroup_symbolic_loops_are_left_unguarded() {
        let ll = "\
define void @prefix(ptr addrspace(3) %shared, i32 %base) {
entry:
  br label %up
up:
  %start = or i32 %base, 1
  %limit = or i32 %base, 31
  br label %loop
loop:
  %i = phi i32 [ %start, %up ], [ %next, %loop ]
  %p = getelementptr inbounds i32, ptr addrspace(3) %shared, i32 %i
  %v = load i32, ptr addrspace(3) %p, align 4
  store i32 %v, ptr addrspace(3) %p, align 4
  %next = add nuw i32 %i, 1
  %done = icmp eq i32 %i, %limit
  br i1 %done, label %down, label %loop
down:
  br label %down_loop
down_loop:
  %j = phi i32 [ %limit, %down ], [ %prev, %down_loop ]
  %prev = add nsw i32 %j, -1
  %q = getelementptr inbounds i32, ptr addrspace(3) %shared, i32 %prev
  %w = load i32, ptr addrspace(3) %q, align 4
  store i32 %w, ptr addrspace(3) %q, align 4
  %keep = icmp ugt i32 %prev, %base
  br i1 %keep, label %down_loop, label %exit
exit:
  ret void
}
";
        match classify_and_instrument(ll, "prefix") {
            GuardPlan::LoopFree => {}
            other => panic!("expected symbolic workgroup loops to be bounded, got {other:?}"),
        }
    }

    #[test]
    fn small_stride_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @radix(ptr addrspace(3) %shared, i32 %base) {
entry:
  %limit = add i32 %base, 8
  br label %loop
loop:
  %i = phi i32 [ %base, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %p = getelementptr inbounds i32, ptr addrspace(3) %shared, i32 %i
  store i32 %i, ptr addrspace(3) %p, align 4
  %next = add nuw i32 %i, 2
  %keep = icmp ult i32 %next, %limit
  br i1 %keep, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "radix") {
            GuardPlan::LoopFree => {}
            other => panic!("expected stride barrier loop to be bounded, got {other:?}"),
        }
    }

    #[test]
    fn exact_doubling_exit_on_next_gt_bound_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @reduce(ptr addrspace(3) %shared, i32 %limit) {
entry:
  br label %loop
loop:
  %width = phi i32 [ 8, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = shl i32 %width, 1
  %done = icmp ugt i32 %next, %limit
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_input_facts(ll, "reduce", &[], &[("limit".into(), 8)]) {
            GuardPlan::LoopFree => {}
            other => {
                panic!(
                    "expected exact next-step doubling barrier loop to be bounded, got {other:?}"
                )
            }
        }
    }

    #[test]
    fn exact_descending_sub_gt_zero_barrier_loop_is_left_unguarded() {
        let ll = "\
%Params = type { i32, i32 }

define void @reduce(ptr addrspace(2) %params) {
entry:
  %count_ptr = getelementptr inbounds %Params, ptr addrspace(2) %params, i64 0, i32 0
  %count = load i32, ptr addrspace(2) %count_ptr, align 4
  %stride_ptr = getelementptr inbounds %Params, ptr addrspace(2) %params, i64 0, i32 1
  %stride = load i32, ptr addrspace(2) %stride_ptr, align 4
  br label %loop
loop:
  %remaining = phi i32 [ %count, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = sub i32 %remaining, %stride
  %more = icmp sgt i32 %next, 0
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_input_facts_and_fields(
            ll,
            "reduce",
            &[],
            &[],
            &[
                ("params".into(), vec![0], 16),
                ("params".into(), vec![1], 16),
            ],
        ) {
            GuardPlan::LoopFree => {}
            other => panic!("expected exact descending barrier loop to be bounded, got {other:?}"),
        }
    }

    #[test]
    fn air_min_bounded_current_phi_descending_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @triangular_like(i32 %n) {
entry:
  %capped = tail call i32 @air.min.u.i32(i32 32, i32 %n)
  %start = add i32 %capped, -2
  %run = icmp sgt i32 %start, -1
  br i1 %run, label %loop, label %exit
loop:
  %i = phi i32 [ %start, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 1, i32 1)
  %next = add nsw i32 %i, -1
  %more = icmp sgt i32 %i, 0
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare i32 @air.min.u.i32(i32, i32)
declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "triangular_like") {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected min-bounded descending barrier loop to be bounded, got {other:?}")
            }
        }
    }

    #[test]
    fn air_min_bounded_eq_loop_through_guarded_preheader_is_left_unguarded() {
        let ll = "\
define void @triangular_preheader_like(i32 %n) {
entry:
  %capped = tail call i32 @air.min.u.i32(i32 32, i32 %n)
  %run = icmp ugt i32 %capped, 1
  br i1 %run, label %preheader, label %exit
preheader:
  br label %loop
loop:
  %i = phi i32 [ 1, %preheader ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 1, i32 1)
  %next = add nuw i32 %i, 1
  %done = icmp eq i32 %next, %capped
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare i32 @air.min.u.i32(i32, i32)
declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "triangular_preheader_like") {
            GuardPlan::LoopFree => {}
            other => {
                panic!(
                    "expected guarded-preheader min-bounded eq loop to be bounded, got {other:?}"
                )
            }
        }
    }

    #[test]
    fn exact_simd_divisor_proves_udiv_barrier_reduction_loop() {
        let ll = "\
define void @simd_reduce_like(<2 x i16> %lsize, i16 %simd_size) {
entry:
  %x16 = extractelement <2 x i16> %lsize, i64 0
  %x = zext i16 %x16 to i32
  %y16 = extractelement <2 x i16> %lsize, i64 1
  %y = zext i16 %y16 to i32
  %lanes = mul nuw nsw i32 %x, %y
  %d = zext i16 %simd_size to i32
  %d_minus_one = add nsw i32 %d, -1
  %rounded = add i32 %d_minus_one, %lanes
  %groups = sdiv i32 %rounded, %d
  %empty = icmp eq i32 %groups, 0
  br i1 %empty, label %exit, label %loop
loop:
  %i = phi i32 [ %groups, %entry ], [ %next, %latch ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %latch
latch:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = udiv i32 %i, %d
  %done = icmp ult i32 %i, %d
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "simd_reduce_like",
            LoopInputFacts {
                arg_values: &[("simd_size".into(), 32)],
                arg_vector_values: &[("lsize".into(), 0, 8), ("lsize".into(), 1, 8)],
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => panic!("expected exact simd divisor to prove udiv reduction, got {other:?}"),
        }
    }

    #[test]
    fn exact_i16_simdgroup_count_prunes_unreachable_barrier_reduction_loop() {
        let ll = "\
define void @stats_like(i16 %simd_size, i16 %simdgroups) {
entry:
  %simd_size_wide = zext i16 %simd_size to i32
  %simdgroups_wide = zext i16 %simdgroups to i32
  %simd_size_minus_one = add nsw i32 %simd_size_wide, -1
  %rounded = add nsw i32 %simd_size_minus_one, %simdgroups_wide
  %groups_wide = sdiv i32 %rounded, %simd_size_wide
  %groups = trunc i32 %groups_wide to i16
  %run = icmp ugt i16 %groups, 1
  br i1 %run, label %loop, label %exit
loop:
  %i = phi i16 [ %groups, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %i_wide = zext i16 %i to i32
  %rounded_next = add nsw i32 %simd_size_minus_one, %i_wide
  %next_wide = sdiv i32 %rounded_next, %simd_size_wide
  %next = trunc i32 %next_wide to i16
  %more = icmp ugt i16 %next, 1
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "stats_like",
            LoopInputFacts {
                arg_values: &[("simd_size".into(), 32), ("simdgroups".into(), 2)],
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => panic!("expected exact launch facts to prune unreachable loop, got {other:?}"),
        }
    }

    #[test]
    fn small_shift_barrier_loops_are_left_unguarded() {
        let ll = "\
define void @scan(ptr addrspace(3) %shared) {
entry:
  br label %up
up:
  %i = phi i32 [ 2, %entry ], [ %next, %up ]
  %next = shl nuw nsw i32 %i, 1
  tail call void @air.wg.barrier(i32 2, i32 1)
  %more = icmp ult i32 %i, 512
  br i1 %more, label %up, label %down
down:
  br label %down_loop
down_loop:
  %j = phi i32 [ 512, %down ], [ %prev, %down_loop ]
  %prev = lshr i32 %j, 1
  tail call void @air.wg.barrier(i32 2, i32 1)
  %done = icmp ult i32 %j, 2
  br i1 %done, label %exit, label %down_loop
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "scan") {
            GuardPlan::LoopFree => {}
            other => panic!("expected shift barrier loops to be bounded, got {other:?}"),
        }
    }

    #[test]
    fn symbolic_i16_right_shift_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @scan16(ptr addrspace(3) %shared, i16 %width, i16 %tid) {
entry:
  br label %loop
loop:
  %i = phi i16 [ %width, %entry ], [ %next, %latch ]
  %next = lshr i16 %i, 1
  tail call void @air.wg.barrier(i32 2, i32 1)
  %active = icmp ult i16 %tid, %next
  br i1 %active, label %body, label %latch
body:
  %slot = zext i16 %tid to i64
  %p = getelementptr inbounds float, ptr addrspace(3) %shared, i64 %slot
  store float 0.0, ptr addrspace(3) %p, align 4
  br label %latch
latch:
  %done = icmp ult i16 %i, 4
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "scan16") {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected symbolic i16 shift barrier loop to be bounded, got {other:?}")
            }
        }
    }

    #[test]
    fn unsigned_halving_phi_condition_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @threadgroup_count(ptr addrspace(3) %shared, i32 %width) {
entry:
  br label %loop
loop:
  %i = phi i32 [ %width, %entry ], [ %next, %latch ]
  %next = lshr i32 %i, 1
  %slot = zext i32 %next to i64
  %p = getelementptr inbounds i32, ptr addrspace(3) %shared, i64 %slot
  %v = load i32, ptr addrspace(3) %p, align 4
  store i32 %v, ptr addrspace(3) %p, align 4
  br label %latch
latch:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %more = icmp ugt i32 %i, 3
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "threadgroup_count") {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected phi-tested unsigned halving loop to be bounded, got {other:?}")
            }
        }
    }

    #[test]
    fn signed_halving_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @threadgroup_sum(ptr addrspace(3) %shared, i32 %lane, i32 %width) {
entry:
  br label %loop
loop:
  %i = phi i32 [ %width, %entry ], [ %next, %latch ]
  %next = sdiv i32 %i, 2
  %active = icmp sgt i32 %next, %lane
  br i1 %active, label %body, label %latch
body:
  %slot = add nsw i32 %next, %lane
  %idx = sext i32 %slot to i64
  %p = getelementptr inbounds float, ptr addrspace(3) %shared, i64 %idx
  %v = load float, ptr addrspace(3) %p, align 4
  store float %v, ptr addrspace(3) %p, align 4
  br label %latch
latch:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %more = icmp sgt i32 %i, 7
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "threadgroup_sum") {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected signed halving barrier loop to be bounded, got {other:?}")
            }
        }
    }

    #[test]
    fn signed_ashr_halving_with_extended_recur_compare_is_left_unguarded() {
        let ll = "\
define void @threadgroup_sum_ashr(ptr addrspace(3) %shared) {
entry:
  br label %loop
loop:
  %i = phi i16 [ 64, %entry ], [ %next, %latch ]
  %next = ashr i16 %i, 1
  %next32 = sext i16 %next to i32
  br label %body
body:
  %slot = sext i16 %next to i64
  %p = getelementptr inbounds float, ptr addrspace(3) %shared, i64 %slot
  %v = load float, ptr addrspace(3) %p, align 4
  store float %v, ptr addrspace(3) %p, align 4
  br label %latch
latch:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %more = icmp slt i32 7, %next32
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "threadgroup_sum_ashr") {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected signed ashr barrier loop to be bounded, got {other:?}")
            }
        }
    }

    #[test]
    fn large_signed_step_with_extended_recur_upper_bound_is_left_unguarded() {
        let ll = "\
define void @large_stride_prefetch(ptr addrspace(3) %shared) {
entry:
  br label %loop
loop:
  %i = phi i16 [ 0, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %slot = sext i16 %i to i64
  %p = getelementptr inbounds i16, ptr addrspace(3) %shared, i64 %slot
  %v = load i16, ptr addrspace(3) %p, align 2
  store i16 %v, ptr addrspace(3) %p, align 2
  %next = add i16 %i, 512
  %next32 = sext i16 %next to i32
  %more = icmp sgt i32 8, %next32
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "large_stride_prefetch") {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected large signed-step barrier loop to be bounded, got {other:?}")
            }
        }
    }

    #[test]
    fn shifted_descending_counter_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @shifted_descending(ptr addrspace(3) %shared) {
entry:
  br label %loop
loop:
  %encoded = phi i32 [ 458752, %entry ], [ %next_encoded, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %decoded = lshr exact i32 %encoded, 16
  %slot = zext i32 %decoded to i64
  %p = getelementptr inbounds i16, ptr addrspace(3) %shared, i64 %slot
  %v = load i16, ptr addrspace(3) %p, align 2
  store i16 %v, ptr addrspace(3) %p, align 2
  %next_decoded = add nsw i32 %decoded, -8
  %next_encoded = shl i32 %next_decoded, 16
  %more = icmp sgt i32 %next_encoded, -65536
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "shifted_descending") {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected shifted descending barrier loop to be bounded, got {other:?}")
            }
        }
    }

    #[test]
    fn shifted_descending_counter_uses_texture_extent_facts() {
        let ll = "\
define void @shifted_descending_texture(ptr addrspace(1) %tex, ptr addrspace(3) %shared) {
entry:
  %13 = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %tex, i32 0)
  %494 = add i32 %13, -1
  %495 = shl i32 %494, 16
  br label %loop
loop:
  %770 = phi i32 [ %495, %entry ], [ %858, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %773 = lshr exact i32 %770, 16
  %774 = zext i32 %773 to i64
  %775 = getelementptr inbounds i16, ptr addrspace(3) %shared, i64 %774
  %776 = load i16, ptr addrspace(3) %775, align 2
  store i16 %776, ptr addrspace(3) %775, align 2
  %857 = add nsw i32 %773, -8
  %858 = shl i32 %857, 16
  %859 = icmp sgt i32 %858, -65536
  br i1 %859, label %loop, label %exit
exit:
  ret void
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "shifted_descending_texture",
            LoopInputFacts {
                texture_extents: &[("tex".to_string(), [8, 8, 1])],
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected texture extent facts to prove shifted barrier loop, got {other:?}")
            }
        }
    }

    #[test]
    fn exact_launch_bound_masked_doubling_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @block_merge(i32 %group_size) {
entry:
  br label %loop
loop:
  %width = phi i32 [ 2, %entry ], [ %masked, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %double = shl nuw nsw i32 %width, 1
  %masked = and i32 %double, 65532
  %done = icmp ugt i32 %masked, %group_size
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "block_merge") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("air.wg.barrier"), "{msg}"),
            other => panic!("expected launch-unknown barrier loop to quarantine, got {other:?}"),
        }
        match classify_and_instrument_with_input_facts(
            ll,
            "block_merge",
            &[],
            &[("group_size".into(), 64)],
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected exact launch fact to prove masked doubling loop, got {other:?}")
            }
        }
    }

    #[test]
    fn exact_launch_bound_next_doubling_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @prefix_sum(i32 %simdgroups) {
entry:
  br label %loop
loop:
  %width = phi i32 [ 1, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = shl i32 %width, 1
  %more = icmp ult i32 %next, %simdgroups
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "prefix_sum") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("air.wg.barrier"), "{msg}"),
            other => panic!("expected launch-unknown barrier loop to quarantine, got {other:?}"),
        }
        match classify_and_instrument_with_input_facts(
            ll,
            "prefix_sum",
            &[],
            &[("simdgroups".into(), 8)],
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected exact launch fact to prove next-doubling loop, got {other:?}")
            }
        }
    }

    #[test]
    fn exact_launch_bound_signed_next_doubling_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @prefix_sum(i32 %group_size) {
entry:
  br label %loop
loop:
  %width = phi i32 [ 1, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = shl nsw i32 %width, 1
  %more = icmp slt i32 %next, %group_size
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "prefix_sum") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("air.wg.barrier"), "{msg}"),
            other => panic!("expected launch-unknown barrier loop to quarantine, got {other:?}"),
        }
        match classify_and_instrument_with_input_facts(
            ll,
            "prefix_sum",
            &[],
            &[("group_size".into(), 64)],
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!(
                    "expected exact launch fact to prove signed next-doubling loop, got {other:?}"
                )
            }
        }
    }

    #[test]
    fn fc_global_next_doubling_gt_limit_barrier_loop_is_left_unguarded() {
        let ll = "\
@_Z7numBins.MTL_FC_INIT_0_j = internal unnamed_addr addrspace(2) externally_initialized constant i32 undef, section \"air.fc_initializer\", align 4
@_ZL7numBins = internal unnamed_addr addrspace(2) global i32 undef, align 4

define internal void @_GLOBAL__sub_I_test() section \"air.static_init\" {
  %v = load i32, ptr addrspace(2) @_Z7numBins.MTL_FC_INIT_0_j, align 4
  store i32 %v, ptr addrspace(2) @_ZL7numBins, align 4
  ret void
}

define void @reduce_bins() {
entry:
  br label %loop
loop:
  %width = phi i32 [ 1, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = shl i32 %width, 1
  %limit = load i32, ptr addrspace(2) @_ZL7numBins, align 4
  %more = icmp ugt i32 %limit, %next
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_function_constants(ll, "reduce_bins", &[(0, 1)]) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected exact FC global to prove next-doubling gt loop, got {other:?}")
            }
        }
    }

    #[test]
    fn derived_fc_global_and_input_facts_prove_barrier_stride_loop() {
        let ll = "\
%struct.Params = type { <4 x i32> }

@_Z16kMPSUserConstant.MTL_FC_INIT_124_j = internal unnamed_addr addrspace(2) externally_initialized constant i32 undef, section \"air.fc_initializer\", align 4
@_ZL9strideXFC = internal unnamed_addr addrspace(2) global i32 0, align 4

define internal void @_GLOBAL__sub_I_test() section \"air.static_init\" {
  %v = load i32, ptr addrspace(2) @_Z16kMPSUserConstant.MTL_FC_INIT_124_j, align 4
  %shifted = lshr i32 %v, 16
  %masked = and i32 %shifted, 3
  %stride = add nuw nsw i32 %masked, 1
  store i32 %stride, ptr addrspace(2) @_ZL9strideXFC, align 4
  ret void
}

define void @pool_like(ptr addrspace(2) %params, <3 x i32> %tg_size) {
entry:
  %field = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 0
  %vec = load <4 x i32>, ptr addrspace(2) %field, align 16
  %extent = extractelement <4 x i32> %vec, i64 0
  %step = extractelement <3 x i32> %tg_size, i64 1
  %stride = load i32, ptr addrspace(2) @_ZL9strideXFC, align 4
  %limit = mul i32 %stride, %extent
  br label %loop
loop:
  %i = phi i32 [ %step, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = add i32 %i, %step
  %more = icmp ult i32 %next, %limit
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "pool_like",
            LoopInputFacts {
                fc_values: &[(124, 0)],
                arg_field_values: &[
                    ("params".into(), vec![0, 0], 0),
                    ("params".into(), vec![0, 1], 0),
                    ("params".into(), vec![0, 2], 0),
                    ("params".into(), vec![0, 3], 0),
                ],
                arg_vector_values: &[("tg_size".into(), 1, 1)],
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!(
                    "expected derived FC global and input facts to prove barrier loop, got {other:?}"
                )
            }
        }
    }

    #[test]
    fn launch_upper_bound_and_fc_global_prove_halved_barrier_reduction() {
        let ll = "\
@_Z21threadsPerThreadgroup.MTL_FC_INIT_11_j = internal unnamed_addr addrspace(2) externally_initialized constant i32 undef, section \"air.fc_initializer\", align 4
@_ZL21threadsPerThreadgroup = internal unnamed_addr addrspace(2) global i32 undef, align 4

define internal void @_GLOBAL__sub_I_test() section \"air.static_init\" {
  %t = load i32, ptr addrspace(2) @_Z21threadsPerThreadgroup.MTL_FC_INIT_11_j, align 4
  store i32 %t, ptr addrspace(2) @_ZL21threadsPerThreadgroup, align 4
  ret void
}

define void @split_like(i32 %lid) {
entry:
  %tg = load i32, ptr addrspace(2) @_ZL21threadsPerThreadgroup, align 4
  %seed = add i32 %lid, %tg
  br label %pre
pre:
  %bound = phi i32 [ %seed, %entry ], [ %shr, %pre ]
  %shr = lshr i32 %bound, 1
  %done_pre = icmp eq i32 %shr, 0
  br i1 %done_pre, label %loop, label %pre
loop:
  %width = phi i32 [ 1, %pre ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = shl i32 %width, 1
  %more = icmp ult i32 %next, %bound
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_function_constants(ll, "split_like", &[(11, 64)]) {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("air.wg.barrier"), "{msg}"),
            other => panic!("expected launch-unknown barrier loop to quarantine, got {other:?}"),
        }
        match classify_and_instrument_with_input_facts_bounds_and_fields(
            ll,
            "split_like",
            &[(11, 64)],
            &[],
            &[("lid".into(), 63)],
            &[],
        ) {
            GuardPlan::Instrumented(_) | GuardPlan::LoopFree => {}
            other => {
                panic!("expected launch upper bound and FC global to prove barrier loop, got {other:?}")
            }
        }
    }

    #[test]
    fn lshr_by_two_barrier_reduction_loop_is_left_unguarded() {
        let ll = "\
define void @quarter_reduce() {
entry:
  br label %loop
loop:
  %width = phi i32 [ 16, %entry ], [ %next, %latch ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %work
work:
  br label %latch
latch:
  %next = lshr i32 %width, 2
  %done = icmp ult i32 %width, 4
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "quarter_reduce") {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected lshr-by-two barrier reduction loop to be bounded, got {other:?}")
            }
        }
    }

    #[test]
    fn input_vector_struct_field_extractelement_proves_barrier_loop() {
        let ll = "\
%Params = type { <2 x i16> }

define void @barrier_vector_bound(ptr addrspace(2) %params) {
entry:
  %p = getelementptr inbounds %Params, ptr addrspace(2) %params, i64 0, i32 0
  %v = load <2 x i16>, ptr addrspace(2) %p, align 4
  %n16 = extractelement <2 x i16> %v, i64 0
  %n = zext i16 %n16 to i32
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 0, i32 1)
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_input_facts_and_fields(
            ll,
            "barrier_vector_bound",
            &[],
            &[],
            &[
                ("params".into(), vec![0, 0], 4),
                ("params".into(), vec![0, 1], 0),
            ],
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected exact vector field lane to prove barrier loop, got {other:?}")
            }
        }
    }

    #[test]
    fn texture_extent_and_vector_arg_prove_barrier_loop() {
        let ll = "\
define void @barrier_texture_extent_bound(<2 x i16> %tg_size, ptr addrspace(1) %tex) {
entry:
  %sx16 = extractelement <2 x i16> %tg_size, i64 0
  %sx = zext i16 %sx16 to i32
  %w = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) %tex, i32 0)
  %has = icmp sgt i32 %w, 0
  br i1 %has, label %loop, label %exit
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = add i32 %i, %sx
  %keep = icmp slt i32 %next, %w
  br i1 %keep, label %loop, label %exit
exit:
  ret void
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "barrier_texture_extent_bound") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("air.wg.barrier"), "{msg}"),
            other => panic!("expected texture-unknown barrier loop to quarantine, got {other:?}"),
        }
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "barrier_texture_extent_bound",
            LoopInputFacts {
                arg_vector_values: &[("tg_size".to_string(), 0, 64)],
                texture_extents: &[("tex".to_string(), [8, 8, 1])],
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => panic!(
                "expected vector lane and texture extent facts to prove barrier loop, got {other:?}"
            ),
        }
    }

    #[test]
    fn typed_texture_extent_proves_positive_step_barrier_loop() {
        let ll = "\
%struct._texture_2d_t = type opaque

define void @typed_texture_extent_bound(%struct._texture_2d_t addrspace(1)* %tex) {
entry:
  %h = tail call i32 @air.get_height_texture_2d(%struct._texture_2d_t addrspace(1)* nocapture readonly %tex, i32 0)
  %has = icmp sgt i32 %h, 0
  br i1 %has, label %loop, label %exit
loop:
  %y = phi i32 [ 0, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %advanced = add nsw i32 %y, 32
  %wide = shl i32 %advanced, 16
  %next = ashr exact i32 %wide, 16
  %keep = icmp slt i32 %next, %h
  br i1 %keep, label %loop, label %exit
exit:
  ret void
}

declare i32 @air.get_height_texture_2d(%struct._texture_2d_t addrspace(1)* nocapture readonly, i32)
declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "typed_texture_extent_bound") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("air.wg.barrier"), "{msg}"),
            other => panic!("expected texture-unknown barrier loop to quarantine, got {other:?}"),
        }
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "typed_texture_extent_bound",
            LoopInputFacts {
                texture_extents: &[("tex".to_string(), [8, 8, 1])],
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected typed texture extent facts to prove barrier loop, got {other:?}")
            }
        }
    }

    #[test]
    fn exact_i16_input_facts_prove_barrier_stride_loop() {
        let ll = "\
define void @i16_stride(ptr addrspace(2) %params) {
entry:
  %step_ptr = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 0
  %limit_ptr = getelementptr inbounds %struct.Params, ptr addrspace(2) %params, i64 0, i32 1
  %step = load i16, ptr addrspace(2) %step_ptr, align 2
  %limit = load i16, ptr addrspace(2) %limit_ptr, align 2
  br label %loop
loop:
  %i = phi i16 [ 0, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = add i16 %i, %step
  %more = icmp slt i16 %next, %limit
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_input_facts_bounds_and_fields(
            ll,
            "i16_stride",
            &[],
            &[],
            &[],
            &[
                ("params".into(), vec![0], 8),
                ("params".into(), vec![1], 16),
            ],
        ) {
            GuardPlan::LoopFree => {}
            other => panic!("expected exact i16 input facts to prove barrier loop, got {other:?}"),
        }
    }

    #[test]
    fn guarded_descending_recur_below_step_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @mtgp_like(i32 %loaded_bound, i32 %offset, i16 %threads) {
entry:
  %masked = and i32 %offset, 65535
  %remaining = sub nsw i32 351, %masked
  %limit = tail call i32 @air.min.u.i32(i32 %loaded_bound, i32 %remaining)
  %step = zext i16 %threads to i32
  %too_small = icmp ult i32 %limit, %step
  br i1 %too_small, label %exit, label %loop
loop:
  %left = phi i32 [ %limit, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = sub i32 %left, %step
  %done = icmp ult i32 %next, %step
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare i32 @air.min.u.i32(i32, i32)
declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "mtgp_like",
            LoopInputFacts {
                arg_values: &[("threads".into(), 64)],
                arg_upper_bounds: &[("offset".into(), 65535)],
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected guarded descending barrier loop to be bounded, got {other:?}")
            }
        }
    }

    #[test]
    fn dynamic_zero_struct_field_and_fc_step_prove_barrier_stride_loop() {
        let ll = "\
@_Z21threadsPerThreadgroup.MTL_FC_INIT_11_j = internal unnamed_addr addrspace(2) externally_initialized constant i32 undef, section \"air.fc_initializer\", align 4
@_ZL21threadsPerThreadgroup = internal unnamed_addr addrspace(2) global i32 undef, align 4

define internal void @_GLOBAL__sub_I_test() section \"air.static_init\" {
  %t = load i32, ptr addrspace(2) @_Z21threadsPerThreadgroup.MTL_FC_INIT_11_j, align 4
  store i32 %t, ptr addrspace(2) @_ZL21threadsPerThreadgroup, align 4
  ret void
}

define void @grid_stride(ptr addrspace(1) %queue, i32 %tgid) {
entry:
  %idx = zext i32 %tgid to i64
  %count_ptr = getelementptr inbounds %struct.Queue, ptr addrspace(1) %queue, i64 %idx, i32 1
  %count = load i32, ptr addrspace(1) %count_ptr, align 4
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %step = load i32, ptr addrspace(2) @_ZL21threadsPerThreadgroup, align 4
  %next = add i32 %step, %i
  %more = icmp ult i32 %next, %count
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_input_facts_bounds_and_fields(
            ll,
            "grid_stride",
            &[(11, 4)],
            &[("tgid".into(), 0)],
            &[],
            &[("queue".into(), vec![1], 16)],
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!(
                    "expected exact dynamic field and FC step to prove barrier loop, got {other:?}"
                )
            }
        }
    }

    #[test]
    fn exact_numeric_arg_field_and_large_stride_prove_prefix_sum_loop() {
        let ll = "\
define void @prefix(ptr addrspace(1) %20, i32 %2) {
entry:
  %34 = zext i32 %2 to i64
  %40 = getelementptr inbounds %struct.Entry, ptr addrspace(1) %20, i64 %34, i32 3
  %41 = load i32, ptr addrspace(1) %40, align 4
  br label %loop
loop:
  %79 = phi i32 [ %41, %entry ], [ %184, %loop ]
  %80 = phi i32 [ 0, %entry ], [ %183, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %183 = add i32 %80, 256
  %184 = load i32, ptr addrspace(1) %40, align 4
  %190 = icmp ult i32 %183, %184
  br i1 %190, label %loop, label %done
done:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_input_facts_bounds_and_fields(
            ll,
            "prefix",
            &[],
            &[("2".into(), 0)],
            &[],
            &[("20".into(), vec![3], 16)],
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!(
                    "expected exact numeric arg and field facts to prove large-stride loop, got {other:?}"
                )
            }
        }
    }

    #[test]
    fn exact_reloaded_field_phi_and_large_stride_prove_prefix_sum_loop() {
        let ll = "\
define void @prefix(ptr addrspace(1) %20, i32 %2) {
entry:
  %34 = zext i32 %2 to i64
  %40 = getelementptr inbounds %struct.Entry, ptr addrspace(1) %20, i64 %34, i32 3
  %41 = load i32, ptr addrspace(1) %40, align 4
  br label %loop
loop:
  %79 = phi i32 [ %41, %entry ], [ %189, %latch ]
  %80 = phi i32 [ 0, %entry ], [ %183, %latch ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %183 = add i32 %80, 256
  %184 = load i32, ptr addrspace(1) %40, align 4
  %185 = icmp ult i32 %183, %184
  br i1 %185, label %reload, label %latch
reload:
  %187 = load i32, ptr addrspace(1) %40, align 4
  br label %latch
latch:
  %189 = phi i32 [ %187, %reload ], [ %184, %loop ]
  %190 = icmp ult i32 %183, %189
  br i1 %190, label %loop, label %done
done:
  ret void
}

declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument_with_input_facts_bounds_and_fields(
            ll,
            "prefix",
            &[],
            &[("2".into(), 0)],
            &[],
            &[("20".into(), vec![3], 16)],
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!(
                    "expected exact reloaded field phi to prove large-stride loop, got {other:?}"
                )
            }
        }
    }

    #[test]
    fn guarded_ctz_decrement_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @linear_reduce(ptr addrspace(3) %shared, i16 %lane, i16 %width) {
entry:
  %ctz = tail call i16 @air.ctz.i16(i16 %width, i1 false)
  %start = add i16 %ctz, -1
  %run = icmp sgt i16 %start, 4
  br i1 %run, label %pre, label %exit
pre:
  %base = zext i16 %lane to i64
  %p = getelementptr inbounds float, ptr addrspace(3) %shared, i64 %base
  br label %loop
loop:
  %i = phi i16 [ %start, %pre ], [ %next, %latch ]
  %shift = and i16 %i, 31
  %wide = zext i16 %shift to i32
  %offset = shl nuw i32 1, %wide
  %active = icmp ugt i32 %offset, 0
  br i1 %active, label %body, label %latch
body:
  %idx = zext i32 %offset to i64
  %q = getelementptr inbounds float, ptr addrspace(3) %shared, i64 %idx
  %v = load float, ptr addrspace(3) %q, align 4
  store float %v, ptr addrspace(3) %p, align 4
  br label %latch
latch:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = add nsw i16 %i, -1
  %more = icmp ugt i16 %next, 4
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare i16 @air.ctz.i16(i16, i1)
declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "linear_reduce") {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected guarded ctz decrement barrier loop to be bounded, got {other:?}")
            }
        }
    }

    #[test]
    fn selected_latch_decrement_barrier_loop_is_left_unguarded() {
        let ll = "\
define void @selected_dec(ptr addrspace(3) %shared, i16 %start, i32 %mask) {
entry:
  br label %loop
loop:
  %i = phi i16 [ %start, %entry ], [ %next, %latch ]
  %slot = sext i16 %i to i64
  %p = getelementptr inbounds i16, ptr addrspace(3) %shared, i64 %slot
  %v = load i16, ptr addrspace(3) %p, align 2
  store i16 %v, ptr addrspace(3) %p, align 2
  tail call void @air.wg.barrier(i32 2, i32 1)
  %zero = icmp eq i32 %mask, 0
  br i1 %zero, label %dec8, label %decmask
decmask:
  %ctz = tail call i32 @air.ctz.i32(i32 %mask, i1 false)
  %ctz16 = trunc i32 %ctz to i16
  %var = sub i16 %i, %ctz16
  br label %latch
dec8:
  %dec = add nsw i16 %i, -8
  br label %latch
latch:
  %next = phi i16 [ %var, %decmask ], [ %dec, %dec8 ]
  %nonnegative = icmp sgt i16 %next, -1
  %more = select i1 %nonnegative, i1 %zero, i1 false
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare i32 @air.ctz.i32(i32, i1)
declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "selected_dec") {
            GuardPlan::LoopFree => {}
            other => {
                panic!(
                    "expected selected latch decrement barrier loop to be bounded, got {other:?}"
                )
            }
        }
    }

    #[test]
    fn unguarded_ctz_decrement_unsigned_barrier_loop_is_quarantined() {
        let ll = "\
define void @linear_reduce(ptr addrspace(3) %shared, i16 %width) {
entry:
  %ctz = tail call i16 @air.ctz.i16(i16 %width, i1 false)
  %start = add i16 %ctz, -1
  br label %loop
loop:
  %i = phi i16 [ %start, %entry ], [ %next, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %next = add nsw i16 %i, -1
  %more = icmp ugt i16 %next, 4
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

declare i16 @air.ctz.i16(i16, i1)
declare void @air.wg.barrier(i32, i32)
";
        match classify_and_instrument(ll, "linear_reduce") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("air.wg.barrier"), "{msg}"),
            other => {
                panic!(
                    "expected unguarded unsigned ctz decrement loop to quarantine, got {other:?}"
                )
            }
        }
    }

    #[test]
    fn loop_calling_loopy_callee_is_quarantined() {
        let ll = "\
define void @helper(ptr addrspace(1) %0) {
  br label %1
1:
  br label %1
}

define void @entry(ptr addrspace(1) %0) {
  br label %loop
loop:
  call void @helper(ptr addrspace(1) %0)
  br label %loop
}
";
        match classify_and_instrument(ll, "entry") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("loopy callee"), "{msg}"),
            other => panic!("expected Quarantine for loop→loopy callee, got {other:?}"),
        }
    }

    #[test]
    fn loop_after_loopy_callee_call_is_instrumented() {
        let ll = "\
define void @helper(ptr addrspace(1) %0) {
  br label %1
1:
  br label %1
}

define void @entry(ptr addrspace(1) %0) {
entry:
  call void @helper(ptr addrspace(1) %0)
  br label %loop
loop:
  br label %loop
}
";
        let out = instrumented(ll, "entry");
        assert!(out.contains("m2v.exit:"), "entry not instrumented:\n{out}");
        assert!(
            !out.contains("loop in \"entry\" calls loopy callee"),
            "call outside caller loop was treated as loop-contained:\n{out}"
        );
    }

    #[test]
    fn entry_input_facts_prune_unreachable_loopy_call_graph() {
        let ll = "\
define internal void @leaf(ptr addrspace(1) %0) {
  br label %spin
spin:
  br label %spin
}

define internal void @helper(ptr addrspace(1) %0) {
entry:
  br label %loop
loop:
  call void @leaf(ptr addrspace(1) %0)
  br label %loop
}

define void @entry(ptr addrspace(1) %out, i32 %count) {
entry:
  %run = icmp ugt i32 %count, 0
  br i1 %run, label %call, label %exit
call:
  call void @helper(ptr addrspace(1) %out)
  br label %exit
exit:
  ret void
}
";
        match classify_and_instrument(ll, "entry") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("loopy callee"), "{msg}"),
            other => panic!("expected facts-free call graph to quarantine, got {other:?}"),
        }

        match classify_and_instrument_with_input_facts(ll, "entry", &[], &[("count".into(), 0)]) {
            GuardPlan::LoopFree => {}
            other => panic!("expected zero count to prune loopy call graph, got {other:?}"),
        }
    }

    #[test]
    fn entry_numbered_pointer_input_facts_prune_unreachable_loopy_call_graph() {
        let ll = r#"
define internal fastcc void @leaf(ptr addrspace(1) %0) {
  br label %spin
spin:
  br label %spin
}

define internal fastcc void @helper(ptr addrspace(1) %0) {
entry:
  br label %loop
loop:
  call fastcc void @leaf(ptr addrspace(1) %0)
  br label %loop
}

define void @"quoted::entry"(i32 addrspace(2)* %4, ptr addrspace(1) %7, i32 %12) {
entry:
  %19 = load i32, i32 addrspace(2)* %4, align 4
  %20 = icmp ugt i32 %19, %12
  br i1 %20, label %call, label %exit
call:
  call fastcc void @helper(ptr addrspace(1) %7)
  br label %exit
exit:
  ret void
}
"#;
        match classify_and_instrument_with_input_facts(
            ll,
            "quoted::entry",
            &[],
            &[("4".into(), 0), ("12".into(), 0)],
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected numbered pointer count to prune loopy call graph, got {other:?}")
            }
        }
    }

    #[test]
    fn quoted_entry_typed_pointer_struct_field_prunes_loopy_call_graph() {
        let ll = r#"
%Params = type { i32, i32 }

define internal fastcc void @leaf(%Params addrspace(2)* %0) {
  br label %spin
spin:
  br label %spin
}

define internal fastcc void @helper(%Params addrspace(2)* %0) {
entry:
  br label %loop
loop:
  call fastcc void @leaf(%Params addrspace(2)* %0)
  br label %loop
}

define void @"quoted::entry"(<2 x i32> %0, %Params addrspace(2)* %1) {
entry:
  %x = extractelement <2 x i32> %0, i64 0
  %p = getelementptr inbounds %Params, %Params addrspace(2)* %1, i64 0, i32 0
  %width = load i32, i32 addrspace(2)* %p, align 4
  %run = icmp ult i32 %x, %width
  br i1 %run, label %call, label %exit
call:
  call fastcc void @helper(%Params addrspace(2)* %1)
  br label %exit
exit:
  ret void
}
"#;
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "quoted::entry",
            LoopInputFacts {
                arg_field_values: &[("1".into(), vec![0], 0)],
                arg_vector_values: &[("0".into(), 0, 0)],
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => panic!("expected zero width to prune loopy call graph, got {other:?}"),
        }
    }

    #[test]
    fn derived_field_pointer_beats_colliding_arg_fact() {
        let ll = r#"
%Params = type { i32 }

define internal fastcc void @leaf(ptr addrspace(1) %0) {
  br label %spin
spin:
  br label %spin
}

define internal fastcc void @helper(ptr addrspace(1) %0) {
entry:
  br label %loop
loop:
  call fastcc void @leaf(ptr addrspace(1) %0)
  br label %loop
}

define void @entry(<2 x i32> %0, %Params addrspace(2)* %1, ptr addrspace(1) %out) {
entry:
  %7 = extractelement <2 x i32> %0, i64 0
  %8 = getelementptr inbounds %Params, %Params addrspace(2)* %1, i64 0, i32 0
  %9 = load i32, i32 addrspace(2)* %8, align 4
  %run = icmp ult i32 %7, %9
  br i1 %run, label %call, label %exit
call:
  call fastcc void @helper(ptr addrspace(1) %out)
  br label %exit
exit:
  ret void
}
"#;
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "entry",
            LoopInputFacts {
                arg_values: &[("8".into(), 11)],
                arg_field_values: &[("1".into(), vec![0], 0)],
                arg_vector_values: &[("0".into(), 0, 0)],
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected field-derived zero to beat colliding arg fact, got {other:?}")
            }
        }
    }

    #[test]
    fn root_gep_vector_buffer_facts_prune_loopy_call_graph() {
        let ll = r#"
%Group = type { i32, i32, i32, i32, i64 }

define internal fastcc void @leaf(ptr addrspace(1) %0) {
  br label %spin
spin:
  br label %spin
}

define internal fastcc void @helper(ptr addrspace(1) %0) {
entry:
  br label %loop
loop:
  call fastcc void @leaf(ptr addrspace(1) %0)
  br label %loop
}

define void @entry(i32 %tgid, <2 x i16> addrspace(2)* %tile_locations, %Group addrspace(2)* %groups, ptr addrspace(1) %out) {
entry:
  %tile_ptr = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(2)* %tile_locations, i64 %tgid
  %tile = load <2 x i16>, <2 x i16> addrspace(2)* %tile_ptr, align 4
  %tile32 = tail call <2 x i32> @air.convert.u.v2i32.u.v2i16(<2 x i16> %tile)
  %coarse = lshr <2 x i32> %tile32, <i32 3, i32 3>
  %fine = and <2 x i32> %tile32, <i32 7, i32 7>
  %x = extractelement <2 x i32> %coarse, i64 0
  %y = extractelement <2 x i32> %coarse, i64 1
  %fx = extractelement <2 x i32> %fine, i64 0
  %fy = extractelement <2 x i32> %fine, i64 1
  %shift = shl i32 %fy, 3
  %lane = or i32 %shift, %fx
  %idx = add i32 %x, %y
  %mask_ptr = getelementptr inbounds %Group, %Group addrspace(2)* %groups, i64 %idx, i32 4
  %mask = load i64, i64 addrspace(2)* %mask_ptr, align 8
  %lane64 = zext i32 %lane to i64
  %bit = shl i64 1, %lane64
  %hit = and i64 %bit, %mask
  %occupied = icmp ne i64 %hit, 0
  br i1 %occupied, label %call, label %exit
call:
  call fastcc void @helper(ptr addrspace(1) %out)
  br label %exit
exit:
  ret void
}

declare <2 x i32> @air.convert.u.v2i32.u.v2i16(<2 x i16>)
"#;
        match classify_and_instrument_with_loop_input_facts(
            ll,
            "entry",
            LoopInputFacts {
                arg_values: &[("tgid".into(), 0)],
                arg_field_values: &[("groups".into(), vec![4], 0)],
                arg_vector_values: &[
                    ("tile_locations".into(), 0, 0),
                    ("tile_locations".into(), 1, 0),
                ],
                ..LoopInputFacts::default()
            },
        ) {
            GuardPlan::LoopFree => {}
            other => {
                panic!("expected root GEP vector facts to prune loopy call graph, got {other:?}")
            }
        }
    }

    #[test]
    fn loop_calling_small_fixed_callee_is_instrumented() {
        let ll = "\
define void @helper(ptr addrspace(1) %0) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, 4
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @entry(ptr addrspace(1) %0) {
  br label %loop
loop:
  call void @helper(ptr addrspace(1) %0)
  br label %loop
}
";
        let out = instrumented(ll, "entry");
        assert!(out.contains("m2v.exit:"), "entry not instrumented:\n{out}");
        assert!(
            !out.contains("loop in \"entry\" calls loopy callee"),
            "small fixed callee treated as unbounded:\n{out}"
        );
    }

    #[test]
    fn loop_calling_small_descending_callee_is_instrumented() {
        let ll = "\
define void @helper(ptr addrspace(1) %0) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 4, %entry ], [ %next, %latch ]
  br label %latch
latch:
  %next = add nsw i32 %i, -1
  %done = icmp eq i32 %i, 0
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @entry(ptr addrspace(1) %0) {
  br label %loop
loop:
  call void @helper(ptr addrspace(1) %0)
  br label %loop
}
";
        let out = instrumented(ll, "entry");
        assert!(out.contains("m2v.exit:"), "entry not instrumented:\n{out}");
        assert!(
            !out.contains("loop in \"entry\" calls loopy callee"),
            "small descending callee treated as unbounded:\n{out}"
        );
    }

    #[test]
    fn loop_calling_recur_compared_small_callee_is_instrumented() {
        let ll = "\
define void @helper(ptr addrspace(1) %0) {
entry:
  br label %loop
loop:
  %write_index = phi i32 [ 1, %entry ], [ %next_write_index, %latch ]
  %index = phi i32 [ 0, %entry ], [ %next_index, %latch ]
  br label %latch
latch:
  %next_write_index = add nuw nsw i32 %write_index, 2
  %next_index = add nuw nsw i32 %index, 1
  %done = icmp eq i32 %next_index, 13
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @entry(ptr addrspace(1) %0) {
  br label %loop
loop:
  call void @helper(ptr addrspace(1) %0)
  br label %loop
}
";
        let out = instrumented(ll, "entry");
        assert!(out.contains("m2v.exit:"), "entry not instrumented:\n{out}");
        assert!(
            !out.contains("loop in \"entry\" calls loopy callee"),
            "small recur-compared callee treated as unbounded:\n{out}"
        );
    }

    #[test]
    fn loop_calling_mask_bounded_callee_is_instrumented_when_zero_is_excluded() {
        let ll = "\
define void @helper(i32 %n) {
entry:
  %zero = icmp eq i32 %n, 0
  br i1 %zero, label %exit, label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @entry(i32 %x) {
entry:
  br label %loop
loop:
  %n = and i32 %x, 15
  call void @helper(i32 %n)
  br label %loop
}
";
        let out = instrumented(ll, "entry");
        assert!(out.contains("m2v.exit:"), "entry not instrumented:\n{out}");
        assert!(
            !out.contains("loop in \"entry\" calls loopy callee"),
            "mask-bounded callee treated as unbounded:\n{out}"
        );
    }

    #[test]
    fn loop_calling_mask_bounded_callee_without_zero_guard_is_quarantined() {
        let ll = "\
define void @helper(i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @entry(i32 %x) {
entry:
  br label %loop
loop:
  %n = and i32 %x, 15
  call void @helper(i32 %n)
  br label %loop
}
";
        match classify_and_instrument(ll, "entry") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("loopy callee"), "{msg}"),
            other => panic!("expected Quarantine for zero-unguarded callee, got {other:?}"),
        }
    }

    #[test]
    fn loop_calling_delegating_mask_bounded_callee_is_instrumented() {
        let ll = "\
define void @leaf(i32 %n) {
entry:
  %zero = icmp eq i32 %n, 0
  br i1 %zero, label %exit, label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @helper(i32 %m) {
entry:
  call void @leaf(i32 %m)
  ret void
}

define void @entry(i32 %x) {
entry:
  br label %loop
loop:
  %n = and i32 %x, 15
  call void @helper(i32 %n)
  br label %loop
}
";
        let out = instrumented(ll, "entry");
        assert!(out.contains("m2v.exit:"), "entry not instrumented:\n{out}");
        assert!(
            !out.contains("loop in \"entry\" calls loopy callee"),
            "delegating small callee treated as unbounded:\n{out}"
        );
    }

    #[test]
    fn loop_calling_delegating_unbounded_callee_is_quarantined() {
        let ll = "\
define void @leaf(i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @helper(i32 %m) {
entry:
  call void @leaf(i32 %m)
  ret void
}

define void @entry(i32 %x) {
entry:
  br label %loop
loop:
  call void @helper(i32 %x)
  br label %loop
}
";
        match classify_and_instrument(ll, "entry") {
            GuardPlan::Quarantine(msg) => assert!(msg.contains("loopy callee"), "{msg}"),
            other => panic!("expected Quarantine for delegated unbounded callee, got {other:?}"),
        }
    }

    #[test]
    fn masked_product_loop_bound_is_small() {
        let ll = "\
define void @helper(i32 %word) {
entry:
  %w = lshr i32 %word, 4
  %a = and i32 %word, 15
  %b = and i32 %w, 15
  %n = mul nuw nsw i32 %a, %b
  %zero = icmp eq i32 %n, 0
  br i1 %zero, label %exit, label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

define void @entry(i32 %x) {
entry:
  br label %loop
loop:
  call void @helper(i32 %x)
  br label %loop
}
";
        let out = instrumented(ll, "entry");
        assert!(out.contains("m2v.exit:"), "entry not instrumented:\n{out}");
        assert!(
            !out.contains("loop in \"entry\" calls loopy callee"),
            "masked product loop bound treated as unbounded:\n{out}"
        );
    }

    #[test]
    fn unknown_loaded_postincrement_eq_loop_is_instrumented_even_with_zero_guard() {
        let ll = "\
define void @k(i32 addrspace(1)* %buf, i32 %tid) {
entry:
  %idx = zext i32 %tid to i64
  %ptr = getelementptr inbounds i32, i32 addrspace(1)* %buf, i64 %idx
  %n = load i32, i32 addrspace(1)* %ptr, align 4
  %zero = icmp eq i32 %n, 0
  br i1 %zero, label %exit, label %loop
loop:
  %i = phi i32 [ %next, %loop ], [ 0, %entry ]
  %acc = phi i32 [ %acc.next, %loop ], [ %tid, %entry ]
  %acc.mul = mul i32 %acc, 1664525
  %acc.next = add i32 %acc.mul, 1013904223
  %next = add nuw i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
";
        let out = instrumented(ll, "k");
        assert!(
            out.contains("m2v.exit:"),
            "unknown loaded loop not instrumented:\n{out}"
        );
    }

    #[test]
    fn loop_calling_small_shift_barrier_callee_is_instrumented() {
        let ll = "\
define void @helper(ptr addrspace(3) %shared, i16 %start, i16 %limit) {
entry:
  %run = icmp ugt i16 %start, %limit
  br i1 %run, label %loop, label %exit
loop:
  %i = phi i16 [ %start, %entry ], [ %next, %latch ]
  %next = lshr i16 %i, 1
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %latch
latch:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %more = icmp ugt i16 %next, %limit
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

define void @entry(ptr addrspace(3) %shared, i16 %start, i16 %limit) {
entry:
  br label %loop
loop:
  call void @helper(ptr addrspace(3) %shared, i16 %start, i16 %limit)
  br label %loop
}

declare void @air.wg.barrier(i32, i32)
";
        let out = instrumented(ll, "entry");
        assert!(out.contains("m2v.exit:"), "entry not instrumented:\n{out}");
        assert!(
            !out.contains("loop in \"entry\" calls loopy callee"),
            "small shift barrier callee treated as unbounded:\n{out}"
        );
    }

    #[test]
    fn loop_calling_loop_free_callee_is_instrumented() {
        let ll = "\
define void @leaf(ptr addrspace(1) %0) {
  ret void
}

define void @entry(ptr addrspace(1) %0) {
  br label %loop
loop:
  call void @leaf(ptr addrspace(1) %0)
  br label %loop
}
";
        let out = instrumented(ll, "entry");
        assert!(out.contains("m2v.exit:"), "entry not instrumented:\n{out}");
        // The loop-free leaf is copied verbatim (no budget added).
        assert!(!out.contains("define void @leaf(ptr addrspace(1) %0) {\n  %m2v.bd"));
    }

    #[test]
    fn non_void_exit_returns_undef() {
        let ll = "\
define <4 x float> @frag(ptr addrspace(1) %0) {
  br label %1
1:
  br label %1
}
";
        let out = instrumented(ll, "frag");
        assert!(
            out.contains("ret <4 x float> undef"),
            "typed undef exit missing:\n{out}"
        );
    }

    #[test]
    fn typed_pointer_dialect_uses_typed_loads() {
        // No opaque `ptr` anywhere → injected loads/stores must use `i32*`.
        let ll = "\
define void @spin(i32 addrspace(1)* %0) {
  br label %1
1:
  br label %1
}
";
        let out = instrumented(ll, "spin");
        assert!(
            out.contains("load i32, i32* %m2v.bd"),
            "expected typed load:\n{out}"
        );
        assert!(
            !out.contains("load i32, ptr %m2v.bd"),
            "unexpected opaque load:\n{out}"
        );
    }

    #[test]
    fn air64_opaque_module_without_pointer_operands_uses_opaque_budget_pointer() {
        let ll = "\
target triple = \"air64_v28-apple-macosx26.5.0\"

define <4 x i32> @frag(<4 x float> %0) {
  br label %1
1:
  br label %1
}
";
        let out = instrumented(ll, "frag");
        assert!(
            out.contains(&format!("store i32 {LOOP_BUDGET_BACKEDGES}, ptr %m2v.bd")),
            "expected opaque budget store:\n{out}"
        );
        assert!(
            out.contains("load i32, ptr %m2v.bd"),
            "expected opaque budget load:\n{out}"
        );
        assert!(
            !out.contains("i32* %m2v.bd"),
            "unexpected typed budget pointer:\n{out}"
        );
    }

    #[test]
    fn mps_like_k_loop_is_instrumented() {
        // The MPS-style GEMM shape whose K-loop (~1.8e9) wedged the GPU: a phi-carried counter loop
        // that must now be budget-bounded.
        let ll = "\
define void @gemm(ptr addrspace(1) %0, ptr addrspace(2) %1) {
  %3 = load i32, ptr addrspace(2) %1, align 4
  br label %loop
loop:
  %i = phi i32 [ 0, %2 ], [ %n, %loop ]
  %n = add nuw i32 %i, 1
  %c = icmp eq i32 %n, %3
  br i1 %c, label %done, label %loop
done:
  ret void
}
";
        let out = instrumented(ll, "gemm");
        assert!(out.contains("m2v.g.0:"), "K-loop guard missing:\n{out}");
        assert!(
            out.contains("[ %n, %m2v.g.0 ]"),
            "K-loop phi pred not renamed:\n{out}"
        );
    }
}
