use super::parse::{
    is_ignored_intrinsic, parse_declaration, parse_function_header, parse_global, parse_type,
    strip_comment,
};
use crate::meta::{self, AirScalar, AirType, KernRole};
use std::collections::{HashMap, HashSet};

mod alloca;
mod metadata_pointees;
mod ordinary_inline;
mod parse;
mod pointer_pointees;
mod raw_buffer;
mod static_init;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum LlType {
    Void,
    Bool,
    Float,
    Half,
    BFloat,
    Int(u32),
    Ptr(u32),
    Vector(Box<LlType>, u32),
    Array(Box<LlType>, u32),
    Struct(Vec<LlType>),
    Named(String),
}

/// SPIR-V capabilities requested as a side effect of materializing an LLVM type. Early helper
/// pruning must retain these declarations: the residual SPIR-V inliner used to materialize the
/// helper first, then remove its dead types without removing module capabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum LlTypeCapability {
    Float16,
    Int8,
    Int16,
    Int64,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct TypedValue {
    pub(super) ty: LlType,
    pub(super) value: LlValue,
}

#[derive(Clone, Debug)]
pub(super) enum LlValue {
    Local(String),
    Global(String),
    Bool(bool),
    Int(u64),
    SignedInt(i64),
    Hex(u64),
    Float(f64),
    Float32Bits(u32),
    HalfBits(u16),
    BFloatBits(u16),
    Vector(Vec<TypedValue>),
    Array(Vec<TypedValue>),
    Struct(Vec<TypedValue>),
    Splat(Box<TypedValue>),
    Gep(Box<LlGep>),
    IntToPtr {
        source: Box<TypedValue>,
        destination: LlType,
    },
    Zero,
    Undef,
}

impl PartialEq for LlValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Local(a), Self::Local(b)) | (Self::Global(a), Self::Global(b)) => a == b,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Int(a), Self::Int(b)) | (Self::Hex(a), Self::Hex(b)) => a == b,
            (Self::SignedInt(a), Self::SignedInt(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a.to_bits() == b.to_bits(),
            (Self::HalfBits(a), Self::HalfBits(b)) | (Self::BFloatBits(a), Self::BFloatBits(b)) => {
                a == b
            }
            (Self::Vector(a), Self::Vector(b))
            | (Self::Array(a), Self::Array(b))
            | (Self::Struct(a), Self::Struct(b)) => a == b,
            (Self::Splat(a), Self::Splat(b)) => a == b,
            (Self::Gep(a), Self::Gep(b)) => a == b,
            (
                Self::IntToPtr {
                    source: a_source,
                    destination: a_destination,
                },
                Self::IntToPtr {
                    source: b_source,
                    destination: b_destination,
                },
            ) => a_source == b_source && a_destination == b_destination,
            (Self::Zero, Self::Zero) | (Self::Undef, Self::Undef) => true,
            _ => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct LlGep {
    pub(super) inbounds: bool,
    pub(super) source_ty: LlType,
    pub(super) base: TypedValue,
    pub(super) indices: Vec<TypedValue>,
}

#[derive(Clone, Debug)]
pub(super) struct LlFunction {
    pub(super) name: String,
    pub(super) ret: LlType,
    pub(super) params: Vec<(String, LlType)>,
    /// Explicit aggregate pointees carried by LLVM's `byval(T)` parameter attribute. Opaque
    /// pointer syntax otherwise erases `T`, and a parameter used only as a memcpy source has no
    /// dereference from which the ordinary body-use inference could recover it.
    pub(super) byval_param_pointees: Vec<Option<LlType>>,
    /// The function's basic blocks lowered to typed carriers — the parse-once typed IR and the SOLE
    /// body substrate. `parse_function` lexes the body lines transiently and hands them to
    /// `parse_inner`, which lowers them once (via `split_body_blocks`, where the module type table is
    /// complete) into these carriers and drops the text. There is no `Vec<String>` body field: after
    /// parse, LLVM-IR text is never re-read. Empty only on the unit-test `LlFunction` constructors
    /// (which build a params-only function for `implicit_entry_block_name`).
    pub(super) blocks: Vec<crate::native::cfg::BodyBlock>,
}

impl LlFunction {
    /// Iterate every instruction of the function's typed carriers, in body order (entry block first,
    /// then each labelled block; within a block, source order). The parse-once dual of walking the flat
    /// body line-by-line: terminators and labels are not instructions, so a reader that matched only
    /// ` = <opcode>` / `store ` / call lines (never a terminator or label) sees the identical
    /// instruction SEQUENCE. Every parse-time block lowers to `Some` for a well-formed function (a block
    /// with no terminator does not lower and is a fail-visible emit error), so on the emitting workload set the
    /// walk is complete; BC is the referee.
    pub(in crate::native) fn carrier_insts(
        &self,
    ) -> impl Iterator<Item = &crate::native::tir::TirInst> {
        self.blocks
            .iter()
            .filter_map(|b| b.typed.as_ref())
            .flat_map(|t| t.insts.iter())
    }
}

#[derive(Clone, Debug)]
pub(super) struct LlDeclaration {
    pub(super) name: String,
    pub(super) ret: LlType,
    pub(super) params: Vec<LlType>,
}

#[derive(Clone, Debug)]
pub(super) struct LlGlobal {
    pub(super) name: String,
    pub(super) addrspace: u32,
    pub(super) ty: LlType,
    pub(super) initializer: Option<TypedValue>,
}

#[derive(Clone, Debug)]
pub(super) struct LlModule {
    pub(super) air_data_layout: Option<crate::layout::AirDataLayout>,
    pub(super) types: HashMap<String, LlType>,
    pub(super) functions: Vec<LlFunction>,
    pub(super) declarations: Vec<LlDeclaration>,
    pub(super) globals: Vec<LlGlobal>,
    /// Proven immutable integer values materialized by AIR static initializers.
    static_init_globals: HashMap<String, meta::StaticIntValue>,
    /// Stage entry selected by the caller's parsed AIR metadata. `None` preserves the downstream
    /// transform's historical fallback to the first bodied function.
    pub(super) entry_name: Option<String>,
    /// Static initializers whose simple single-block bodies were spliced into the typed entry. The
    /// post-emit injector skips these; the emitted-graph closure owns every other shape.
    pub(super) preinlined_static_initializers: HashSet<String>,
    /// Pointer-valued loads cloned across an ordinary-helper boundary. The old SPIR-V inliner kept
    /// cross-function local-pointer-field loads opaque until its sidecar recovery ran before
    /// interface binding. Early typed inlining preserves that boundary fact so emission takes the
    /// same marker/recovery path instead of resolving the load eagerly inside the combined function.
    pub(super) preinlined_helper_pointer_loads: HashSet<String>,
    /// Type capabilities requested while emitting functions removed by the typed helper inliner.
    /// Post-emit helper cleanup prunes those functions and their dead types but retains the capability
    /// declarations, so typed pruning replays the same declaration-only side effect.
    pub(super) preinlined_helper_type_capabilities: HashSet<LlTypeCapability>,
    pub(super) entry_functions: HashSet<String>,
    pub(super) ptr_pointees: HashMap<(String, String), LlType>,
    pub(super) local_alloca_pointees: HashMap<(String, String), LlType>,
    pub(super) imageblock_data_pointee: Option<LlType>,
    pub(super) imageblock_dimensions: Option<[u32; 2]>,
    /// True when one kernel addresses more than one structural imageblock coordinate. Those calls
    /// communicate through the tile and therefore require shared Workgroup cells rather than the
    /// per-invocation scratch used by single-coordinate slice staging.
    pub(super) imageblock_shared_cells: bool,
    /// AIR entry parameter carrying `[[threads_per_threadgroup]]`, used as the row stride for a
    /// shared imageblock whose dimensions are supplied by the dispatch rather than APV metadata.
    pub(super) imageblock_threads_per_threadgroup_param: Option<String>,
    metadata_pointee_params: HashSet<(String, String)>,
    /// For each metadata-seeded pointee param, the buffer element's declared `air.arg_type_size`
    /// (the authoritative byte extent, offset-aware over any union/bitfield tail). Used to let a
    /// same-size concrete LLVM aggregate GEP `source_ty` override the metadata struct, since the
    /// re-derived `LlType::Struct` size can exceed the declared size when overlapping members are
    /// flattened.
    metadata_pointee_sizes: HashMap<(String, String), u64>,
    metadata_byte_buffer_params: HashSet<(String, String)>,
    /// Entry params declared `air.buffer` in the kernel metadata (any element type): DATA pointers,
    /// as opposed to `air.texture`/sampler args that also arrive as `ptr addrspace(1)` params.
    pub(super) metadata_data_buffer_params: HashSet<(String, String)>,
    /// Primitive pointee declared by AIR entry-buffer metadata, retained independently from the
    /// use-inferred logical pointer type. Representation planning needs this source contract when a
    /// helper reinterprets a float buffer through an integer atomic view.
    pub(super) metadata_primitive_buffer_pointees: HashMap<(String, String), LlType>,
    /// Entry function-constant buffer parameters keyed to their shared Metal buffer location.
    pub(super) metadata_fc_buffer_locations: HashMap<(String, String), u32>,
    pub(super) raw_buffer_params: HashSet<(String, String)>,
    /// Buffer parameters whose raw representation was selected because one call-connected object
    /// has storage-incompatible views. Interface construction must retain that representation
    /// instead of reconstructing one endpoint's typed view.
    pub(super) call_connected_raw_params: HashSet<(String, String)>,
    /// Raw parameters connected through an address-preserving call edge to another function
    /// parameter. Workgroup memory in this set can share one raw entry allocation; a raw helper fed
    /// from a typed global cannot and retains its concrete vector-backed lowering.
    pub(super) param_connected_raw_params: HashSet<(String, String)>,
}

fn infer_metadata_fc_buffer_locations(
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    functions: &[LlFunction],
) -> HashMap<(String, String), u32> {
    let Some(kern) = kern else {
        return HashMap::new();
    };
    let Some(entry_name) = entry_name else {
        return HashMap::new();
    };
    let Some(entry) = functions
        .iter()
        .find(|function| function.name == entry_name)
    else {
        return HashMap::new();
    };
    kern.function_constant_buffer_locations
        .iter()
        .filter_map(|(index, location)| {
            let (name, ty) = entry.params.get(*index as usize)?;
            matches!(ty, LlType::Ptr(1 | 2))
                .then(|| ((entry.name.clone(), name.clone()), *location))
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(in crate::native) struct ParamCallEdge {
    caller_func: String,
    caller_param: String,
    callee_func: String,
    callee_param: String,
}

/// Trace a pointer-valued local back to one of `f`'s pointer parameters through aliases that preserve
/// the pointed-to address.  This deliberately accepts only a single, unambiguous parameter root: a
/// cross-buffer select/phi is not one buffer alias and must not make raw-buffer marking contagious.
fn pointer_param_alias_roots(f: &LlFunction) -> HashMap<String, String> {
    let mut roots: HashMap<String, String> = f
        .params
        .iter()
        .filter(|&(_name, ty)| matches!(ty, LlType::Ptr(_)))
        .map(|(name, _ty)| (name.clone(), name.clone()))
        .collect();
    let mut changed = true;
    while changed {
        changed = false;
        for inst in f.carrier_insts() {
            if let Some((res, base)) = inst.identity_ptr_bitcast() {
                if let Some(root) = roots.get(base).cloned() {
                    if roots.insert(res.to_string(), root).is_none() {
                        changed = true;
                    }
                }
                continue;
            }

            let Some(res) = &inst.result else {
                continue;
            };
            if let Some(gep) = &inst.gep() {
                let LlValue::Local(base) = &gep.base.value else {
                    continue;
                };
                if let Some(root) = roots.get(base).cloned() {
                    if roots.insert(res.clone(), root).is_none() {
                        changed = true;
                    }
                }
                continue;
            }

            let common_root = |values: Vec<&LlValue>| -> Option<String> {
                let mut values = values.into_iter();
                let LlValue::Local(first) = values.next()? else {
                    return None;
                };
                let root = roots.get(first)?.clone();
                for value in values {
                    let LlValue::Local(name) = value else {
                        return None;
                    };
                    if roots.get(name) != Some(&root) {
                        return None;
                    }
                }
                Some(root)
            };
            if let Some(incoming) = inst.phi_values() {
                if let Some(root) = common_root(incoming.collect()) {
                    if roots.insert(res.clone(), root).is_none() {
                        changed = true;
                    }
                }
                continue;
            }
            if let Some((true_value, false_value)) = inst.select_arms().as_deref() {
                if !matches!(true_value.ty, LlType::Ptr(_))
                    || !matches!(false_value.ty, LlType::Ptr(_))
                {
                    continue;
                }
                if let Some(root) = common_root(vec![&true_value.value, &false_value.value]) {
                    if roots.insert(res.clone(), root).is_none() {
                        changed = true;
                    }
                }
            }
        }
    }
    roots
}

/// Record the primitive GEP source types reached through a cross-buffer pointer phi, keyed by each
/// entry root. Normal single-root GEP inference — including its existing select-arm path — remains
/// authoritative for every other buffer. The fallback therefore needs BOTH provenance facts: a phi
/// joining distinct roots and a downstream GEP whose concrete source type agrees with the primitive
/// metadata contract.
fn cross_buffer_pointer_phi_gep_sources(f: &LlFunction) -> HashMap<String, HashSet<LlType>> {
    let mut roots: HashMap<String, HashSet<String>> = f
        .params
        .iter()
        .filter(|&(_name, ty)| matches!(ty, LlType::Ptr(_)))
        .map(|(name, _ty)| (name.clone(), HashSet::from([name.clone()])))
        .collect();
    let mut contains_phi: HashMap<String, bool> =
        roots.keys().cloned().map(|name| (name, false)).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for inst in f.carrier_insts() {
            if let Some((res, base)) = inst.identity_ptr_bitcast() {
                let Some(base_roots) = roots.get(base).cloned() else {
                    continue;
                };
                let base_has_phi = contains_phi.get(base).copied().unwrap_or(false);
                let result_roots = roots.entry(res.to_string()).or_default();
                let old_len = result_roots.len();
                result_roots.extend(base_roots);
                changed |= result_roots.len() != old_len;
                if base_has_phi && !contains_phi.get(res).copied().unwrap_or(false) {
                    contains_phi.insert(res.to_string(), true);
                    changed = true;
                }
                continue;
            }

            let Some(res) = &inst.result else {
                continue;
            };
            if let Some(gep) = &inst.gep() {
                let LlValue::Local(base) = &gep.base.value else {
                    continue;
                };
                let Some(base_roots) = roots.get(base).cloned() else {
                    continue;
                };
                let base_has_phi = contains_phi.get(base).copied().unwrap_or(false);
                let result = res.clone();
                let result_roots = roots.entry(result.clone()).or_default();
                let old_len = result_roots.len();
                result_roots.extend(base_roots);
                changed |= result_roots.len() != old_len;
                if base_has_phi && !contains_phi.get(&result).copied().unwrap_or(false) {
                    contains_phi.insert(result, true);
                    changed = true;
                }
                continue;
            }

            let (merge_roots, result_has_phi) = if let Some(incoming) = inst.phi_values() {
                let mut merged = HashSet::new();
                let mut complete = true;
                for value in incoming {
                    let LlValue::Local(name) = value else {
                        complete = false;
                        break;
                    };
                    let Some(value_roots) = roots.get(name) else {
                        complete = false;
                        break;
                    };
                    merged.extend(value_roots.iter().cloned());
                }
                (complete.then_some(merged), true)
            } else if let Some((true_value, false_value)) = inst.select_arms().as_deref() {
                if !matches!(true_value.ty, LlType::Ptr(_))
                    || !matches!(false_value.ty, LlType::Ptr(_))
                {
                    continue;
                }
                let (LlValue::Local(true_name), LlValue::Local(false_name)) =
                    (&true_value.value, &false_value.value)
                else {
                    continue;
                };
                let (Some(true_roots), Some(false_roots)) =
                    (roots.get(true_name), roots.get(false_name))
                else {
                    continue;
                };
                let mut merged = true_roots.clone();
                merged.extend(false_roots.iter().cloned());
                (
                    Some(merged),
                    contains_phi.get(true_name).copied().unwrap_or(false)
                        || contains_phi.get(false_name).copied().unwrap_or(false),
                )
            } else {
                continue;
            };

            let Some(merge_roots) = merge_roots else {
                continue;
            };
            let result = res.clone();
            let result_roots = roots.entry(result.clone()).or_default();
            let old_len = result_roots.len();
            result_roots.extend(merge_roots);
            changed |= result_roots.len() != old_len;
            if result_has_phi && !contains_phi.get(&result).copied().unwrap_or(false) {
                contains_phi.insert(result, true);
                changed = true;
            }
        }
    }

    let mut sources: HashMap<String, HashSet<LlType>> = HashMap::new();
    for inst in f.carrier_insts() {
        let Some(gep) = &inst.gep() else {
            continue;
        };
        let LlValue::Local(base) = &gep.base.value else {
            continue;
        };
        if !contains_phi.get(base).copied().unwrap_or(false) {
            continue;
        }
        let Some(base_roots) = roots.get(base) else {
            continue;
        };
        if base_roots.len() < 2 {
            continue;
        }
        for root in base_roots {
            sources
                .entry(root.clone())
                .or_default()
                .insert(gep.source_ty.clone());
        }
    }
    sources
}

fn infer_entry_functions(ll: &str) -> HashSet<String> {
    ["kernel", "vertex", "fragment"]
        .into_iter()
        .filter_map(|stage| meta::entry_name(ll, stage))
        .collect()
}

fn infer_metadata_byte_buffer_params(
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    functions: &[LlFunction],
) -> HashSet<(String, String)> {
    let mut out = HashSet::new();
    let Some(kern) = kern else {
        return out;
    };
    let Some(entry_name) = entry_name else {
        return out;
    };
    let Some(entry) = functions.iter().find(|f| f.name == entry_name) else {
        return out;
    };
    for (idx, (name, ty)) in entry.params.iter().enumerate() {
        let Some(arg_type) = kern.buffer_type_name(idx as u32) else {
            continue;
        };
        if arg_type != "char" && arg_type != "void" {
            continue;
        }
        if matches!(ty, LlType::Ptr(1 | 2)) {
            out.insert((entry.name.clone(), name.clone()));
        }
    }
    out
}

fn infer_metadata_data_buffer_params(
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    functions: &[LlFunction],
) -> HashSet<(String, String)> {
    let mut out = HashSet::new();
    let Some(kern) = kern else {
        return out;
    };
    let Some(entry_name) = entry_name else {
        return out;
    };
    let Some(entry) = functions.iter().find(|f| f.name == entry_name) else {
        return out;
    };
    for (idx, (name, ty)) in entry.params.iter().enumerate() {
        if !matches!(
            kern.role_of(idx as u32),
            Some(
                KernRole::Buffer(_)
                    | KernRole::AccelerationStructureShadow(_)
                    | KernRole::PrimitiveAccelerationStructureShadow(_)
            )
        ) {
            continue;
        }
        let device_buffer_array = matches!(ty, LlType::Ptr(0))
            && kern
                .buffer_type_name(idx as u32)
                .is_some_and(meta::is_device_buffer_array_type_name);
        if matches!(ty, LlType::Ptr(1 | 2)) || device_buffer_array {
            out.insert((entry.name.clone(), name.clone()));
        }
    }
    out
}

fn infer_metadata_primitive_buffer_pointees(
    kern: Option<&meta::KernMeta>,
    entry_name: Option<&str>,
    functions: &[LlFunction],
) -> HashMap<(String, String), LlType> {
    let mut out = HashMap::new();
    let (Some(kern), Some(entry_name)) = (kern, entry_name) else {
        return out;
    };
    let Some(entry) = functions
        .iter()
        .find(|function| function.name == entry_name)
    else {
        return out;
    };
    for (index, (name, ty)) in entry.params.iter().enumerate() {
        if !matches!(ty, LlType::Ptr(1..=3))
            || !matches!(kern.role_of(index as u32), Some(KernRole::Buffer(_)))
        {
            continue;
        }
        let Some(layout) = kern
            .buffer_type_name(index as u32)
            .and_then(meta::primitive_air_type_from_name)
        else {
            continue;
        };
        out.insert(
            (entry.name.clone(), name.clone()),
            ll_type_from_air_type(&layout),
        );
    }
    out
}

fn infer_cross_coordinate_imageblock(
    functions: &[LlFunction],
    entry_functions: &HashSet<String>,
) -> bool {
    for function in functions
        .iter()
        .filter(|function| entry_functions.contains(&function.name))
    {
        let mut coordinates = HashSet::new();
        for inst in function.carrier_insts() {
            let Some(call) = inst.alias_call() else {
                continue;
            };
            if call.callee != "air.imageblock_data" {
                continue;
            }
            let Some(coordinate) = call.args.first() else {
                continue;
            };
            let key = match &coordinate.value {
                LlValue::Local(name) => format!("local:{name}"),
                LlValue::Zero => "zero".to_string(),
                LlValue::Undef => "undef".to_string(),
                other => format!("{other:?}"),
            };
            coordinates.insert(key);
            if coordinates.len() > 1 {
                return true;
            }
        }
    }
    false
}

/// Whether an entry function byte-addresses a nonzero offset from an `air.imageblock_data` result.
///
/// AIR pointers are opaque, so the parser tracks only SSA pointer provenance here: seed every
/// imageblock-data result, propagate it through identity pointer bitcasts and GEPs, then inspect a
/// byte GEP whose base is in that provenance set. This is deliberately about the decoded pointer
/// graph and constant byte offset, never an AIR identifier or workload-specific shape.
fn infer_imageblock_nonzero_byte_field(
    functions: &[LlFunction],
    entry_functions: &HashSet<String>,
) -> bool {
    let function_by_name = functions
        .iter()
        .map(|function| (function.name.as_str(), function))
        .collect::<HashMap<_, _>>();
    let mut reachable = entry_functions.clone();
    let mut pending = entry_functions.iter().cloned().collect::<Vec<_>>();
    while let Some(name) = pending.pop() {
        let Some(function) = function_by_name.get(name.as_str()) else {
            continue;
        };
        for call in function
            .carrier_insts()
            .filter_map(|inst| inst.call().as_deref())
        {
            if function_by_name.contains_key(call.callee.as_str())
                && reachable.insert(call.callee.clone())
            {
                pending.push(call.callee.clone());
            }
        }
    }
    for function in functions
        .iter()
        .filter(|function| reachable.contains(&function.name))
    {
        let mut roots = HashSet::new();
        let mut changed = true;
        while changed {
            changed = false;
            for inst in function.carrier_insts() {
                let Some(result) = &inst.result else {
                    continue;
                };
                if !result.starts_with('%') {
                    continue;
                }

                // A value call whose rhs `strip_call_prefix` chain parses to `air.imageblock_data`.
                // `alias_call` on a result-bearing line comes only from the value-call rhs path (the
                // void fallback needs a `= `-less line), so this is the reader's `strip_call_prefix(rhs)`.
                if inst
                    .alias_call()
                    .is_some_and(|call| call.callee == "air.imageblock_data")
                {
                    changed |= roots.insert(result.clone());
                    continue;
                }

                if let Some((alias, base)) = inst.identity_ptr_bitcast() {
                    if roots.contains(base) {
                        changed |= roots.insert(alias.to_string());
                    }
                    continue;
                }

                let Some(gep) = &inst.gep() else {
                    continue;
                };
                let LlValue::Local(base) = &gep.base.value else {
                    continue;
                };
                if !roots.contains(base) {
                    continue;
                }
                changed |= roots.insert(result.clone());
                if gep.source_ty == LlType::Int(8)
                    && gep
                        .indices
                        .iter()
                        .filter_map(typed_value_u64)
                        .any(|offset| offset != 0)
                {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod imageblock_reachability_tests {
    use super::*;

    fn module(entry_calls_helper: bool) -> LlModule {
        let call = if entry_calls_helper {
            "  call void @helper()\n"
        } else {
            ""
        };
        LlModule::parse(&format!(
            r#"define void @entry() {{
entry:
{call}  ret void
}}
define internal void @helper() {{
entry:
  %data = call ptr addrspace(4) @air.imageblock_data(<2 x i16> zeroinitializer, i32 0, i16 0)
  %field = getelementptr i8, ptr addrspace(4) %data, i64 16
  ret void
}}
declare ptr addrspace(4) @air.imageblock_data(<2 x i16>, i32, i16)
"#
        ))
        .expect("module parses")
    }

    #[test]
    fn reachable_helper_nonzero_byte_field_requires_complete_imageblock_cell() {
        let reachable = module(true);
        assert!(infer_imageblock_nonzero_byte_field(
            &reachable.functions,
            &HashSet::from(["entry".to_string()])
        ));

        let unreachable = module(false);
        assert!(!infer_imageblock_nonzero_byte_field(
            &unreachable.functions,
            &HashSet::from(["entry".to_string()])
        ));
    }
}

fn infer_apv_imageblock_dimensions(ll: &str) -> Option<[u32; 2]> {
    let root = ll.lines().find_map(|line| {
        let rest = line
            .trim()
            .strip_prefix("!apv.imageblock_dimensions = !{!")?;
        rest.strip_suffix('}')?.parse::<u32>().ok()
    })?;
    let prefix = format!("!{root} = !{{");
    let body = ll.lines().find_map(|line| {
        line.trim()
            .strip_prefix(&prefix)
            .and_then(|rest| rest.strip_suffix('}'))
    })?;
    let mut fields = body.split(',').map(str::trim);
    let width = fields.next()?.strip_prefix("i32 ")?.parse().ok()?;
    let height = fields.next()?.strip_prefix("i32 ")?.parse().ok()?;
    (fields.next().is_none() && width != 0 && height != 0).then_some([width, height])
}

fn infer_imageblock_data_pointee(
    kern: Option<&meta::KernMeta>,
    complete_cell_layout: bool,
) -> Option<LlType> {
    let kern = kern?;
    let mut pointee = None;
    for layout in kern.imageblock_layouts.values() {
        let candidate = imageblock_data_pointee_from_air_type(layout, complete_cell_layout)?;
        match &pointee {
            Some(existing) if existing != &candidate => return None,
            Some(_) => {}
            None => pointee = Some(candidate),
        }
    }
    pointee
}

/// The imageblock-data pointee depends on its proven addressing model. Complete cells are necessary
/// for APV/shared tiles and whenever AIR follows an imageblock pointer with a nonzero byte field;
/// only a single-coordinate, first-member-only scratch may collapse a metadata struct to its first
/// member.
fn imageblock_data_pointee_from_air_type(
    ty: &AirType,
    complete_cell_layout: bool,
) -> Option<LlType> {
    match ty {
        AirType::Struct(members) if complete_cell_layout && members.len() != 1 => {
            Some(ll_type_from_air_type(ty))
        }
        AirType::Struct(members) => members
            .first()
            .map(|member| ll_type_from_air_type(&member.ty)),
        _ => Some(ll_type_from_air_type(ty)),
    }
}

pub(crate) fn ll_type_from_air_type(ty: &AirType) -> LlType {
    match ty {
        AirType::Scalar(scalar) => ll_type_from_air_scalar(*scalar),
        AirType::Vec { scalar, lanes } => {
            LlType::Vector(Box::new(ll_type_from_air_scalar(*scalar)), *lanes)
        }
        AirType::PackedVec { scalar, lanes } => {
            LlType::Array(Box::new(ll_type_from_air_scalar(*scalar)), *lanes)
        }
        AirType::Array { elem, len } => LlType::Array(Box::new(ll_type_from_air_type(elem)), *len),
        AirType::Matrix { scalar, cols, rows } => LlType::Struct(vec![LlType::Array(
            Box::new(LlType::Vector(
                Box::new(ll_type_from_air_scalar(*scalar)),
                *rows,
            )),
            *cols,
        )]),
        AirType::Struct(members) => LlType::Struct(
            members
                .iter()
                .map(|member| ll_type_from_air_type(&member.ty))
                .collect(),
        ),
    }
}

fn ll_type_from_air_scalar(scalar: AirScalar) -> LlType {
    match scalar {
        AirScalar::Float => LlType::Float,
        AirScalar::Half => LlType::Half,
        AirScalar::UInt | AirScalar::SInt => LlType::Int(32),
        AirScalar::ULong | AirScalar::SLong => LlType::Int(64),
        AirScalar::UShort | AirScalar::SShort => LlType::Int(16),
        AirScalar::UChar | AirScalar::Bool => LlType::Int(8),
    }
}

fn is_ignored_global(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("@llvm.global_ctors")
        || t.starts_with("@llvm.global_dtors")
        || t.starts_with("@llvm.used")
        || t.starts_with("@llvm.compiler.used")
}

fn typed_value_u64(value: &TypedValue) -> Option<u64> {
    match value.value {
        LlValue::Int(value) | LlValue::Hex(value) => Some(value),
        LlValue::SignedInt(value) if value >= 0 => Some(value as u64),
        _ => None,
    }
}

pub(crate) fn round_up_u64(value: u64, align: u64) -> u64 {
    if align <= 1 {
        value
    } else {
        value.div_ceil(align) * align
    }
}

#[cfg(test)]
mod layout_abi_tests {
    //! Layout/ABI table + cross-calculator differential (refactor T3). The crate has several
    //! independent size/align calculators that agree on scalars but DIVERGE by design on vector
    //! padding; Phase 2/S4 will fold them into one `layout(ty, LayoutRule)` oracle, and these tests
    //! are the contract that fold must preserve. Each calculator here is one `LayoutRule` variant:
    //!
    //! - `type_storage_size_align`         → "Native"     (packed-style: vec3 = 12/4)
    //! - `native_memcpy_type_size_align`   → "Memcpy"     (source vector ABI allocation layout)
    //! - `air_metadata_type_size_align`    → "AirMetadata" (packed/unpacked distinguished by type)
    //!
    //! The passes layer applies the same source ABI alignment to emitted SPIR-V types through its
    //! sidecar-aware layout calculator.
    use super::*;
    use crate::meta::{AirMember, AirScalar, AirType};

    fn module() -> LlModule {
        LlModule::parse("define void @k() {\nentry:\n  ret void\n}\n")
            .expect("minimal module parses")
    }

    fn vec(elem: LlType, lanes: u32) -> LlType {
        LlType::Vector(Box::new(elem), lanes)
    }

    #[test]
    fn scalar_sizes_match_metal_abi() {
        let m = module();
        for (ty, sz) in [
            (LlType::Bool, 1),
            (LlType::Int(8), 1),
            (LlType::Int(16), 2),
            (LlType::Half, 2),
            (LlType::BFloat, 2),
            (LlType::Int(32), 4),
            (LlType::Float, 4),
            (LlType::Int(64), 8),
            (LlType::Ptr(1), 8),
        ] {
            assert_eq!(
                m.scalar_storage_size(&ty),
                Some(sz),
                "scalar size of {ty:?}"
            );
        }
        // non-scalars have no scalar-storage size
        assert_eq!(m.scalar_storage_size(&vec(LlType::Float, 4)), None);
    }

    #[test]
    fn native_rule_is_packed_style() {
        // "Native" (`type_storage_size_align`): tight packing, align = element align (no vec growth).
        let m = module();
        assert_eq!(m.type_storage_size_align(&LlType::Float), Some((4, 4)));
        assert_eq!(m.type_storage_size_align(&LlType::Half), Some((2, 2)));
        assert_eq!(
            m.type_storage_size_align(&vec(LlType::Float, 2)),
            Some((8, 4))
        );
        assert_eq!(
            m.type_storage_size_align(&vec(LlType::Float, 3)),
            Some((12, 4))
        ); // packed_float3
        assert_eq!(
            m.type_storage_size_align(&vec(LlType::Float, 4)),
            Some((16, 4))
        );
        // odd-width int rounds up to whole bytes, align = next_pow2 capped at 8
        assert_eq!(m.type_storage_size_align(&LlType::Int(24)), Some((3, 4)));
        assert_eq!(m.type_storage_size_align(&LlType::Int(1)), Some((1, 1)));
        // array of scalars: tight
        assert_eq!(
            m.type_storage_size_align(&LlType::Array(Box::new(LlType::Float), 3)),
            Some((12, 4))
        );
        // struct { i8, float }: i8@0, float@4 (align 4), total 8
        assert_eq!(
            m.type_storage_size_align(&LlType::Struct(vec![LlType::Int(8), LlType::Float])),
            Some((8, 4))
        );
    }

    #[test]
    fn memcpy_rule_pads_vec3_to_four_lanes() {
        // "Memcpy" (`native_memcpy_type_size_align`): a vec3 occupies 4 lanes and self-aligns.
        let m = module();
        assert_eq!(
            m.native_memcpy_type_size_align(&LlType::Float),
            Some((4, 4))
        );
        assert_eq!(
            m.native_memcpy_type_size_align(&vec(LlType::Float, 2)),
            Some((8, 8))
        );
        assert_eq!(
            m.native_memcpy_type_size_align(&vec(LlType::Float, 3)),
            Some((16, 16))
        ); // float3
        assert_eq!(
            m.native_memcpy_type_size_align(&vec(LlType::Float, 4)),
            Some((16, 16))
        );
        // array of vec3 strides at the padded 16 and floors align at the element's 16
        assert_eq!(
            m.native_memcpy_type_size_align(&LlType::Array(Box::new(vec(LlType::Float, 3)), 2)),
            Some((32, 16))
        );
        // array of scalars floors align at 4
        assert_eq!(
            m.native_memcpy_type_size_align(&LlType::Array(Box::new(LlType::Float), 3)),
            Some((12, 4))
        );
    }

    #[test]
    fn memcpy_rule_uses_parsed_source_vector_alignment() {
        let m = LlModule::parse(concat!(
            "target datalayout = \"e-v24:64:64\"\n",
            "define void @k() {\nentry:\n  ret void\n}\n",
        ))
        .expect("module with custom datalayout parses");

        assert_eq!(
            m.native_memcpy_type_size_align(&vec(LlType::Int(8), 3)),
            Some((8, 8))
        );
    }

    #[test]
    fn air_metadata_rule_distinguishes_packed_from_unpacked() {
        // "AirMetadata" (`air_metadata_type_size_align`): the metadata type system carries the
        // packed-vs-unpacked distinction the LLVM `LlType` can't express.
        let m = module();
        assert_eq!(
            m.air_metadata_type_size_align(&AirType::Scalar(AirScalar::Float)),
            Some((4, 4))
        );
        assert_eq!(
            m.air_metadata_type_size_align(&AirType::Vec {
                scalar: AirScalar::Float,
                lanes: 3
            }),
            Some((16, 16)) // float3
        );
        assert_eq!(
            m.air_metadata_type_size_align(&AirType::PackedVec {
                scalar: AirScalar::Float,
                lanes: 3
            }),
            Some((12, 4)) // packed_float3
        );
        assert_eq!(
            m.air_metadata_type_size_align(&AirType::Struct(vec![
                AirMember {
                    offset: 0,
                    ty: AirType::Scalar(AirScalar::UChar)
                },
                AirMember {
                    offset: 4,
                    ty: AirType::Scalar(AirScalar::Float)
                },
            ])),
            Some((8, 4))
        );
    }

    #[test]
    fn air_metadata_incompatible_with_vulkan_block_layout_requires_a_byte_view() {
        let m = module();
        let overlapping = AirType::Struct(vec![
            AirMember {
                offset: 0,
                ty: AirType::Array {
                    elem: Box::new(AirType::Scalar(AirScalar::UInt)),
                    len: 8,
                },
            },
            AirMember {
                offset: 16,
                ty: AirType::Array {
                    elem: Box::new(AirType::Scalar(AirScalar::UInt)),
                    len: 8,
                },
            },
        ]);
        assert!(m.air_metadata_requires_byte_view(&overlapping));

        let stride_overlap = AirType::Struct(vec![
            AirMember {
                offset: 0,
                ty: AirType::Array {
                    elem: Box::new(AirType::Struct(vec![
                        AirMember {
                            offset: 0,
                            ty: AirType::Scalar(AirScalar::UShort),
                        },
                        AirMember {
                            offset: 12,
                            ty: AirType::Scalar(AirScalar::UChar),
                        },
                    ])),
                    len: 2,
                },
            },
            AirMember {
                offset: 28,
                ty: AirType::Scalar(AirScalar::UInt),
            },
        ]);
        assert!(m.air_metadata_requires_byte_view(&stride_overlap));

        let adjacent = AirType::Struct(vec![
            AirMember {
                offset: 0,
                ty: AirType::Scalar(AirScalar::UInt),
            },
            AirMember {
                offset: 4,
                ty: AirType::Scalar(AirScalar::UInt),
            },
        ]);
        assert!(!m.air_metadata_requires_byte_view(&adjacent));
    }

    #[test]
    fn differential_calculators_agree_on_scalars_and_scalar_aggregates() {
        // Where the layout rules MUST agree, S4's oracle unification must keep them equal.
        let m = module();
        let agree = [
            LlType::Bool,
            LlType::Int(8),
            LlType::Int(16),
            LlType::Int(32),
            LlType::Int(64),
            LlType::Half,
            LlType::Float,
            LlType::Ptr(1),
            LlType::Array(Box::new(LlType::Float), 3),
            LlType::Struct(vec![LlType::Int(8), LlType::Float]),
            LlType::Struct(vec![LlType::Float, LlType::Float, LlType::Int(32)]),
        ];
        for ty in agree {
            assert_eq!(
                m.type_storage_size_align(&ty),
                m.native_memcpy_type_size_align(&ty),
                "Native and Memcpy rules must agree on {ty:?}"
            );
        }
    }

    #[test]
    fn differential_calculators_diverge_on_vectors_by_design() {
        // The known, intentional divergence — captured so a future "why do these disagree?" reads
        // as documented behaviour, not a regression. vec3 is the canonical packed/unpacked split.
        let m = module();
        for lanes in [2u32, 3, 4] {
            let ty = vec(LlType::Float, lanes);
            let native = m.type_storage_size_align(&ty).unwrap();
            let memcpy = m.native_memcpy_type_size_align(&ty).unwrap();
            assert_ne!(
                native, memcpy,
                "Native vs Memcpy are expected to differ for float{lanes} (packed vs padded)"
            );
        }
        // spelled out for the canonical case
        assert_eq!(
            m.type_storage_size_align(&vec(LlType::Float, 3)),
            Some((12, 4))
        );
        assert_eq!(
            m.native_memcpy_type_size_align(&vec(LlType::Float, 3)),
            Some((16, 16))
        );
    }
}

#[cfg(test)]
mod resolve_known_type_tests {
    //! `resolve_known_type` canonicalizes an `LlType` against the module's named-type table — the
    //! primitive the M2 pointee-carrier census (`native::tir_pointee_check`) applies to BOTH sides
    //! before `operand_type_compatible`, so a carrier `Named("%struct._half8")` and an emitter
    //! `Struct([Array(Half,8)])` are recognized as the SAME type instead of being over-counted as a
    //! divergence (the reconciliation set an M2 consumer must settle). These pin that canonicalization.
    use super::*;

    fn module_with_types(air: &str) -> LlModule {
        LlModule::parse(air).expect("fixture parses")
    }

    #[test]
    fn named_struct_alias_resolves_to_structural_definition() {
        let m = module_with_types(concat!(
            "%struct._half8 = type { [8 x half] }\n",
            "define void @k() {\nentry:\n  ret void\n}\n",
        ));
        let named = LlType::Named("%struct._half8".to_string());
        let structural = LlType::Struct(vec![LlType::Array(Box::new(LlType::Half), 8)]);
        // The carrier holds the Named alias, the emitter the structural form: post-resolve they are
        // the same, so the census must classify this as agreement, not divergence.
        assert_eq!(m.resolve_known_type(&named), structural);
        assert_eq!(m.resolve_known_type(&structural), structural);
    }

    #[test]
    fn i1_and_single_lane_vector_canonicalize() {
        let m = module_with_types("define void @k() {\nentry:\n  ret void\n}\n");
        assert_eq!(m.resolve_known_type(&LlType::Int(1)), LlType::Bool);
        assert_eq!(
            m.resolve_known_type(&LlType::Vector(Box::new(LlType::Float), 1)),
            LlType::Float
        );
    }

    #[test]
    fn unknown_named_type_is_left_as_is() {
        let m = module_with_types("define void @k() {\nentry:\n  ret void\n}\n");
        let named = LlType::Named("%struct.absent".to_string());
        assert_eq!(m.resolve_known_type(&named), named);
    }
}

#[cfg(test)]
mod raw_buffer_inference_tests {
    //! Direct unit coverage for the `infer_raw_buffer_params` fixpoint (refactor T6 gap-list: one
    //! of the "highest-complexity zero-test" inference surfaces). `LlModule::parse` runs the whole
    //! inference chain, so each fixture parses AIR that exercises one raw-buffer signal and asserts
    //! the resulting `raw_buffer_params` set (keyed by `(function-name, param-name)`). A param is a
    //! "raw byte buffer" when the body reinterprets it away from its declared element type — the
    //! signal the passes layer needs to wrap it as `RuntimeArray<uchar>` rather than a typed buffer.
    use super::*;

    fn params(air: &str) -> HashSet<(String, String)> {
        LlModule::parse(air)
            .expect("fixture parses")
            .raw_buffer_params
    }

    #[test]
    fn byte_gep_then_wide_load_marks_param_raw() {
        // `%p = gep i8, %buf` makes %p a byte-alias of the param; a non-i8 load through %p is a
        // byte→wider reinterpret, so %buf is a raw byte buffer.
        let air = concat!(
            "define void @k(ptr addrspace(1) %buf) {\n",
            "entry:\n",
            "  %p = getelementptr i8, ptr addrspace(1) %buf, i64 0\n",
            "  %v = load i32, ptr addrspace(1) %p\n",
            "  ret void\n",
            "}\n",
        );
        assert!(params(air).contains(&("k".to_string(), "%buf".to_string())));
    }

    #[test]
    fn two_distinct_typed_loads_mark_param_raw() {
        // Loading the same param as two different (non-wrapper) types means no single typed view
        // fits it — it is accessed as raw bytes reinterpreted per-load.
        let air = concat!(
            "define void @k(ptr addrspace(1) %buf) {\n",
            "entry:\n",
            "  %a = load float, ptr addrspace(1) %buf\n",
            "  %b = load i32, ptr addrspace(1) %buf\n",
            "  ret void\n",
            "}\n",
        );
        assert!(params(air).contains(&("k".to_string(), "%buf".to_string())));
    }

    #[test]
    fn pointer_value_store_marks_destination_param_raw() {
        let air = concat!(
            "define void @k(ptr addrspace(1) %out, ptr addrspace(1) %source) {\n",
            "entry:\n",
            "  store ptr addrspace(1) %source, ptr addrspace(1) %out, align 8\n",
            "  ret void\n",
            "}\n",
        );
        assert!(params(air).contains(&("k".to_string(), "%out".to_string())));
    }

    #[test]
    fn nested_select_gep_infers_every_pointer_param_pointee() {
        let air = concat!(
            "define void @k(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, i1 %x, i1 %y) {\n",
            "entry:\n",
            "  %inner = select i1 %x, ptr addrspace(1) %a, ptr addrspace(1) %b\n",
            "  %outer = select i1 %y, ptr addrspace(1) %inner, ptr addrspace(1) %c\n",
            "  %element = getelementptr half, ptr addrspace(1) %outer, i64 0\n",
            "  %value = load half, ptr addrspace(1) %element\n",
            "  ret void\n",
            "}\n",
        );
        let module = LlModule::parse(air).expect("fixture parses");
        for param in ["%a", "%b", "%c"] {
            assert_eq!(
                module
                    .ptr_pointees
                    .get(&("k".to_string(), param.to_string())),
                Some(&LlType::Half)
            );
        }
    }

    #[test]
    fn single_typed_load_leaves_param_typed() {
        // A param read only ever as one consistent scalar type has a valid typed view — NOT raw.
        let air = concat!(
            "define void @k(ptr addrspace(1) %buf) {\n",
            "entry:\n",
            "  %v = load float, ptr addrspace(1) %buf\n",
            "  ret void\n",
            "}\n",
        );
        assert!(!params(air).contains(&("k".to_string(), "%buf".to_string())));
    }

    #[test]
    fn opaque_memcpy_source_into_local_aggregate_is_raw() {
        let air = concat!(
            "%S = type { <4 x float> }\n",
            "define void @k(ptr addrspace(2) %src) {\n",
            "entry:\n",
            "  %dst = alloca %S\n",
            "  %field = getelementptr %S, ptr %dst, i64 0, i32 0\n",
            "  %source = bitcast ptr addrspace(2) %src to ptr addrspace(2)\n",
            "  call void @llvm.memcpy.p0.p2.i64(ptr %field, ptr addrspace(2) %source, i64 16, i1 false)\n",
            "  ret void\n",
            "}\n",
            "declare void @llvm.memcpy.p0.p2.i64(ptr, ptr addrspace(2), i64, i1)\n",
        );
        assert!(params(air).contains(&("k".to_string(), "%src".to_string())));
    }

    #[test]
    fn typed_memcpy_source_into_local_aggregate_stays_typed() {
        let air = concat!(
            "%S = type { <4 x float> }\n",
            "define void @k(ptr addrspace(2) %src) {\n",
            "entry:\n",
            "  %dst = alloca %S\n",
            "  %field = getelementptr %S, ptr %dst, i64 0, i32 0\n",
            "  %source = getelementptr <4 x float>, ptr addrspace(2) %src, i64 0\n",
            "  call void @llvm.memcpy.p0.p2.i64(ptr %field, ptr addrspace(2) %source, i64 16, i1 false)\n",
            "  ret void\n",
            "}\n",
            "declare void @llvm.memcpy.p0.p2.i64(ptr, ptr addrspace(2), i64, i1)\n",
        );
        assert!(!params(air).contains(&("k".to_string(), "%src".to_string())));
    }

    #[test]
    fn opaque_memcpy_destination_does_not_imply_local_aggregate_storage() {
        let air = concat!(
            "define void @k(ptr addrspace(2) %src) {\n",
            "entry:\n",
            "  %dst = call ptr @destination()\n",
            "  call void @llvm.memcpy.p0.p2.i64(ptr %dst, ptr addrspace(2) %src, i64 16, i1 false)\n",
            "  ret void\n",
            "}\n",
            "declare ptr @destination()\n",
            "declare void @llvm.memcpy.p0.p2.i64(ptr, ptr addrspace(2), i64, i1)\n",
        );
        assert!(!params(air).contains(&("k".to_string(), "%src".to_string())));
    }

    #[test]
    fn non_pointer_param_is_never_raw() {
        // Only addrspace pointer params are candidates; a by-value scalar param cannot be a buffer.
        let air = concat!(
            "define void @k(i32 %n) {\n",
            "entry:\n",
            "  ret void\n",
            "}\n",
        );
        assert!(params(air).is_empty());
    }

    #[test]
    fn raw_buffer_mark_reaches_helper_through_byte_gep_alias() {
        // `%alias` is not a direct entry parameter: it is a byte-offset GEP followed by an identity
        // pointer bitcast.  The entry's byte→i32 read proves `%buf` is raw, so the same buffer object
        // must make the helper's `%p` raw before helper inlining exposes its float accesses.
        let air = concat!(
            "define void @entry(ptr addrspace(1) %buf) {\n",
            "entry:\n",
            "  %byte = getelementptr i8, ptr addrspace(1) %buf, i64 4\n",
            "  %word = load i32, ptr addrspace(1) %byte\n",
            "  %alias = bitcast ptr addrspace(1) %byte to ptr addrspace(1)\n",
            "  call void @helper(ptr addrspace(1) %alias)\n",
            "  ret void\n",
            "}\n",
            "define void @helper(ptr addrspace(1) %p) {\n",
            "entry:\n",
            "  %f = getelementptr float, ptr addrspace(1) %p, i64 0\n",
            "  %v = load float, ptr addrspace(1) %f\n",
            "  ret void\n",
            "}\n",
        );
        let module = LlModule::parse(air).expect("fixture parses");
        let entry = ("entry".to_string(), "%buf".to_string());
        let helper = ("helper".to_string(), "%p".to_string());
        assert!(module.raw_buffer_params.contains(&entry));
        assert!(module.raw_buffer_params.contains(&helper));
        assert!(module.call_connected_raw_params.contains(&entry));
        assert!(module.call_connected_raw_params.contains(&helper));
    }

    #[test]
    fn incompatible_call_connected_aggregate_views_select_raw_buffers() {
        // One buffer object is a packed-byte view in the entry and an array-of-float view in its
        // helper. Each function has only one local source type, but the call edge proves that a
        // single typed SPIR-V pointee cannot represent both views.
        let air = concat!(
            "define void @entry(ptr addrspace(1) %buf) {\n",
            "entry:\n",
            "  %alias = getelementptr [3 x i8], ptr addrspace(1) %buf, i64 0, i64 0\n",
            "  call void @helper(ptr addrspace(1) %alias)\n",
            "  ret void\n",
            "}\n",
            "define void @helper(ptr addrspace(1) %p) {\n",
            "entry:\n",
            "  %field = getelementptr [3 x float], ptr addrspace(1) %p, i64 0, i64 1\n",
            "  %value = load float, ptr addrspace(1) %field\n",
            "  ret void\n",
            "}\n",
        );
        let raw = params(air);
        assert!(raw.contains(&("entry".to_string(), "%buf".to_string())));
        assert!(raw.contains(&("helper".to_string(), "%p".to_string())));
    }

    #[test]
    fn call_connected_workgroup_reinterpretation_selects_raw_words() {
        let air = concat!(
            "define void @entry(ptr addrspace(3) %scratch) {\n",
            "entry:\n",
            "  %local = getelementptr float, ptr addrspace(3) %scratch, i64 0\n",
            "  store float 0.000000e+00, ptr addrspace(3) %local\n",
            "  call void @write_uint(ptr addrspace(3) %scratch)\n",
            "  call void @write_float(ptr addrspace(3) %scratch)\n",
            "  ret void\n",
            "}\n",
            "define void @write_uint(ptr addrspace(3) %p) {\n",
            "entry:\n",
            "  %slot = getelementptr i32, ptr addrspace(3) %p, i64 0\n",
            "  store i32 1, ptr addrspace(3) %slot\n",
            "  ret void\n",
            "}\n",
            "define void @write_float(ptr addrspace(3) %p) {\n",
            "entry:\n",
            "  %slot = getelementptr float, ptr addrspace(3) %p, i64 0\n",
            "  store float 1.000000e+00, ptr addrspace(3) %slot\n",
            "  ret void\n",
            "}\n",
            "define void @unrelated(ptr addrspace(3) %typed) {\n",
            "entry:\n",
            "  store float 2.000000e+00, ptr addrspace(3) %typed\n",
            "  ret void\n",
            "}\n",
        );
        let module = LlModule::parse(air).expect("fixture parses");
        for key in [
            ("entry".to_string(), "%scratch".to_string()),
            ("write_uint".to_string(), "%p".to_string()),
            ("write_float".to_string(), "%p".to_string()),
        ] {
            assert!(module.raw_buffer_params.contains(&key));
            assert!(module.call_connected_raw_params.contains(&key));
        }
        assert!(!module
            .raw_buffer_params
            .contains(&("unrelated".to_string(), "%typed".to_string())));
    }

    #[test]
    fn helper_raw_mark_reaches_entry_through_byte_gep_alias() {
        // The reverse direction matters too: helper-local byte reinterpretation must make the entry
        // buffer raw when the call receives a GEP alias rather than `%buf` directly. That preserves
        // the dynamic byte offset after helper inlining.
        let air = concat!(
            "define void @entry(ptr addrspace(1) %buf) {\n",
            "entry:\n",
            "  %alias = getelementptr i8, ptr addrspace(1) %buf, i64 4\n",
            "  call void @helper(ptr addrspace(1) %alias)\n",
            "  ret void\n",
            "}\n",
            "define void @helper(ptr addrspace(1) %p) {\n",
            "entry:\n",
            "  %byte = getelementptr i8, ptr addrspace(1) %p, i64 0\n",
            "  %word = load i32, ptr addrspace(1) %byte\n",
            "  ret void\n",
            "}\n",
        );
        let raw = params(air);
        assert!(raw.contains(&("entry".to_string(), "%buf".to_string())));
        assert!(raw.contains(&("helper".to_string(), "%p".to_string())));
    }

    #[test]
    fn raw_buffer_mark_does_not_cross_select_between_parameters() {
        // A select across two entry buffers is not an address-preserving alias of either one.  Even
        // though `%a` is raw, the helper parameter must remain unmarked: otherwise a byte view of one
        // resource would make an unrelated resource's typed accesses use the raw model.
        let air = concat!(
            "define void @entry(ptr addrspace(1) %a, ptr addrspace(1) %b, i1 %cond) {\n",
            "entry:\n",
            "  %byte = getelementptr i8, ptr addrspace(1) %a, i64 4\n",
            "  %word = load i32, ptr addrspace(1) %byte\n",
            "  %merged = select i1 %cond, ptr addrspace(1) %a, ptr addrspace(1) %b\n",
            "  call void @helper(ptr addrspace(1) %merged)\n",
            "  ret void\n",
            "}\n",
            "define void @helper(ptr addrspace(1) %p) {\n",
            "entry:\n",
            "  %f = getelementptr float, ptr addrspace(1) %p, i64 0\n",
            "  %v = load float, ptr addrspace(1) %f\n",
            "  ret void\n",
            "}\n",
        );
        let raw = params(air);
        assert!(raw.contains(&("entry".to_string(), "%a".to_string())));
        assert!(!raw.contains(&("entry".to_string(), "%b".to_string())));
        assert!(!raw.contains(&("helper".to_string(), "%p".to_string())));
    }
}

#[cfg(test)]
mod pointee_inference_tests {
    //! Direct unit coverage for the two remaining "highest-complexity zero-test" pointee-inference
    //! fixpoints (§6 gap-list): `infer_pointer_pointees` (what element type a pointer PARAM is
    //! indexed as) and `infer_local_alloca_pointees` (what same-size type a local ALLOCA is
    //! reinterpreted as). `LlModule::parse` runs both; each fixture parses AIR and asserts the
    //! resulting map keyed by `(function-name, value-name)`.
    use super::*;

    fn parsed(air: &str) -> LlModule {
        LlModule::parse(air).expect("fixture parses")
    }

    #[test]
    fn param_gep_source_becomes_pointer_pointee() {
        // A pointer param indexed as `float` records `float` as its pointee element type.
        let air = concat!(
            "define void @k(ptr addrspace(1) %buf) {\n",
            "entry:\n",
            "  %p = getelementptr float, ptr addrspace(1) %buf, i64 0\n",
            "  ret void\n",
            "}\n",
        );
        assert_eq!(
            parsed(air)
                .ptr_pointees
                .get(&("k".to_string(), "%buf".to_string())),
            Some(&LlType::Float)
        );
    }

    #[test]
    fn first_gep_source_wins_for_pointer_pointee() {
        // Distinct GEP element types off the same param: the first seen is retained (or_insert).
        let air = concat!(
            "define void @k(ptr addrspace(1) %buf) {\n",
            "entry:\n",
            "  %a = getelementptr float, ptr addrspace(1) %buf, i64 0\n",
            "  %b = getelementptr i32, ptr addrspace(1) %buf, i64 4\n",
            "  ret void\n",
            "}\n",
        );
        assert_eq!(
            parsed(air)
                .ptr_pointees
                .get(&("k".to_string(), "%buf".to_string())),
            Some(&LlType::Float)
        );
    }

    #[test]
    fn unindexed_param_has_no_pointee() {
        // A pointer param never used as a GEP base gets no inferred pointee.
        let air = concat!(
            "define void @k(ptr addrspace(1) %buf) {\n",
            "entry:\n",
            "  ret void\n",
            "}\n",
        );
        assert!(parsed(air).ptr_pointees.is_empty());
    }

    #[test]
    fn direct_store_value_becomes_pointer_param_pointee() {
        let air = concat!(
            "define void @k(ptr addrspace(1) %out, float %value) {\n",
            "entry:\n",
            "  store float %value, ptr addrspace(1) %out, align 4\n",
            "  ret void\n",
            "}\n",
        );
        assert_eq!(
            parsed(air)
                .ptr_pointees
                .get(&("k".to_string(), "%out".to_string())),
            Some(&LlType::Float)
        );
    }

    #[test]
    fn primitive_metadata_seeds_cross_buffer_phi_roots_without_direct_geps() {
        // The only float GEP is below a phi joining distinct buffers, so the single-root GEP pass
        // cannot attribute it to either input. `air.arg_type_name` + the declared four-byte extent
        // remain authoritative and seed both otherwise-untyped buffer roots.
        let air = r#"
define void @k(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %select_a, ptr addrspace(1) %select_b, i1 %cond) {
entry:
  %select = select i1 %cond, ptr addrspace(1) %select_a, ptr addrspace(1) %select_b
  br i1 %cond, label %left, label %right
left:
  br label %join
right:
  br label %join
join:
  %merged = phi ptr addrspace(1) [ %a, %left ], [ %b, %right ]
  %p = getelementptr float, ptr addrspace(1) %merged, i64 0
  ret void
}
!air.kernel = !{!0}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5, !6}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"a"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"b"}
!5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"select_a"}
!6 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"select_b"}
"#;
        let unseeded = parsed(air);
        assert!(!unseeded
            .ptr_pointees
            .contains_key(&("k".to_string(), "%a".to_string())));
        assert!(!unseeded
            .ptr_pointees
            .contains_key(&("k".to_string(), "%b".to_string())));
        let module = LlModule::parse_with_primitive_phi_metadata(air).expect("fixture parses");
        assert_eq!(
            module
                .ptr_pointees
                .get(&("k".to_string(), "%a".to_string())),
            Some(&LlType::Float)
        );
        assert_eq!(
            module
                .ptr_pointees
                .get(&("k".to_string(), "%b".to_string())),
            Some(&LlType::Float)
        );
        assert!(!module
            .ptr_pointees
            .contains_key(&("k".to_string(), "%select_a".to_string())));
        assert!(!module
            .ptr_pointees
            .contains_key(&("k".to_string(), "%select_b".to_string())));
    }

    #[test]
    fn same_size_gep_reinterpret_records_alloca_pointee() {
        // An i32 alloca indexed as `float` (same 4-byte size, no pointers) is a reinterpret: the
        // alloca's pointee becomes `float`.
        let air = concat!(
            "define void @k() {\n",
            "entry:\n",
            "  %a = alloca i32\n",
            "  %p = getelementptr float, ptr %a, i64 0\n",
            "  ret void\n",
            "}\n",
        );
        assert_eq!(
            parsed(air)
                .local_alloca_pointees
                .get(&("k".to_string(), "%a".to_string())),
            Some(&LlType::Float)
        );
    }

    #[test]
    fn same_type_gep_leaves_alloca_unreinterpreted() {
        // Indexing an i32 alloca as i32 is not a reinterpret (candidate == original is filtered).
        let air = concat!(
            "define void @k() {\n",
            "entry:\n",
            "  %a = alloca i32\n",
            "  %p = getelementptr i32, ptr %a, i64 0\n",
            "  ret void\n",
            "}\n",
        );
        assert!(parsed(air).local_alloca_pointees.is_empty());
    }

    #[test]
    fn scalar_alloca_with_byte_view_uses_bounded_byte_storage() {
        let air = concat!(
            "define void @k() {\n",
            "entry:\n",
            "  %slot = alloca float, align 4\n",
            "  %alias = bitcast ptr %slot to ptr\n",
            "  %high = getelementptr i8, ptr %alias, i64 2\n",
            "  store half 0xH3C00, ptr %high, align 2\n",
            "  ret void\n",
            "}\n",
        );
        assert_eq!(
            parsed(air)
                .local_alloca_pointees
                .get(&("k".to_string(), "%slot".to_string())),
            Some(&LlType::Array(Box::new(LlType::Int(8)), 4))
        );
    }
}
