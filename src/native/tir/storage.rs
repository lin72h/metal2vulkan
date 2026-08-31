//! Byte-neutral responsibility split of the former monolith; see the parent module.

use super::*;
use crate::native::ir::{LlGep, LlType, LlValue};
use spirv::StorageClass;
use std::collections::HashMap;

/// The SPIR-V storage class of an LLVM address space, mirroring the emitter's `llvm_pointer_storage`
/// (`emitter/helpers.rs`). This is the DEFAULT storage every pointer of that address space starts in —
/// it is NOT the whole story: the emitter overrides it statefully (alloca → `Function`, a buffer-modeled
/// device pointer → `StorageBuffer`, a raw byte-loaded pointer → `Private`), which is exactly why a
/// storage carrier cannot be `f(addrspace)` (proven 2026-06-28: addrspace alone diverges from the
/// emitter on 158k banked pointers). Returns `None` for an address space with no fixed default (the
/// caller leaves such a value unmapped rather than guessing). See [`derive_pointer_storage`].
pub(in crate::native) fn addrspace_default_storage(addrspace: u32) -> Option<StorageClass> {
    match addrspace {
        0 | 4 => Some(StorageClass::Private),
        1 | 2 => Some(StorageClass::UniformConstant),
        3 => Some(StorageClass::Workgroup),
        _ => None,
    }
}

/// M1 storage-carrier measurement (the pointer-typing rewrite): derive, purely from the typed graph, a
/// best-effort `(SSA pointer value → StorageClass)` map and compare it against the emitter's actual
/// `pointer_storage` (`tir_storage_check`). This is the storage half of the `(StorageClass, pointee)`
/// carrier — the pointee half already lives on the value (`use_pointees`). It is NOT yet stored on
/// `TirFunction` or consumed by emission: the byte-conformance gate guards consumption, and this
/// derivation is still an approximation (the residual vs the emitter is what the measurement reports
/// and the next increment closes).
///
/// The rules replicate the emitter's *structural* derivation in a single forward pass over the
/// structurized blocks (the order the emitter itself walks):
///   * a function pointer param seeds [`addrspace_default_storage`] of its address space;
///   * `alloca` → `Function` (the emitter's alloca override, `body.rs`);
///   * `getelementptr`/`bitcast`/`addrspacecast`/`freeze`/`select`/`phi` copy storage from their
///     source pointer(s) when known (the emitter's copy-from-source propagation), agreeing merges keep
///     the common class;
///   * any other pointer result, or a derivation whose source storage is unknown, falls back to
///     [`addrspace_default_storage`].
///     What it does NOT yet model (the known residual, all stateful in the emitter): the buffer-modeling
///     path that promotes a device pointer to `StorageBuffer`, the raw byte-pointer path that demotes a
///     loaded pointer to `Private`, and merge-meta-resolved storage. Those show up as divergences in the
///     measurement.
pub(in crate::native) fn derive_pointer_storage(
    tir: &TirFunction,
    params: &[(String, LlType)],
    named_types: &HashMap<String, LlType>,
) -> HashMap<String, StorageClass> {
    derive_pointer_storage_from(tir, params, named_types, &HashMap::new())
}

/// Derive pointer storage while preserving storage facts already established by the emitter for
/// parameters and globals. Pointer identity operations are solved to a fixpoint: structurization can
/// introduce a cyclic pair of state phis whose only concrete arm is an alloca-derived pointer later in
/// block order, so a forward-only walk would incorrectly assign the LLVM address-space default before
/// reaching that arm.
pub(in crate::native) fn derive_pointer_storage_from(
    tir: &TirFunction,
    params: &[(String, LlType)],
    named_types: &HashMap<String, LlType>,
    seeds: &HashMap<String, StorageClass>,
) -> HashMap<String, StorageClass> {
    let mut storage = seeds.clone();
    for (name, ty) in params {
        if let LlType::Ptr(addrspace) = ty {
            if let Some(s) = addrspace_default_storage(*addrspace) {
                storage.entry(name.clone()).or_insert(s);
            }
        }
    }
    loop {
        let mut changed = false;
        for block in &tir.blocks {
            for inst in &block.insts {
                let Some(result) = &inst.result else {
                    continue;
                };
                if storage.contains_key(result) {
                    continue;
                }
                let Some(LlType::Ptr(addrspace)) = tir.value_types.get(result) else {
                    continue;
                };
                // Identity/merge operations must wait for a concrete source instead of taking the
                // address-space default. In particular, an addrspace(0) phi carrying an alloca remains
                // Function storage even when the phi cycle precedes that alloca in emission order.
                let opcode = inst.opcode.as_str();
                let resolved = match opcode {
                    "alloca" => Some(StorageClass::Function),
                    "getelementptr" => inst
                        .gep()
                        .as_ref()
                        .and_then(|gep| gep_base_storage(gep, &storage, named_types)),
                    "bitcast" | "addrspacecast" => cast_source_storage(&inst.operands, &storage),
                    "freeze" => freeze_source_storage(&inst.operands, &storage),
                    "select" => select_arm_storage(&inst.operands, &storage),
                    "phi" => inst
                        .phi_values()
                        .and_then(|values| phi_incoming_storage(values, &storage)),
                    _ => addrspace_default_storage(*addrspace),
                };
                if let Some(s) = resolved {
                    storage.insert(result.clone(), s);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    storage
}

/// Storage of a `getelementptr` result: copy from its base pointer when the base is a known local —
/// EXCEPT a gep into a `Private` aggregate that CONTAINS a pointer, which the emitter promotes to
/// `Function` storage (the pointers-in-locals model: a function-scoped struct holding pointer fields
/// must live in `Function` storage to be addressable — body.rs, the `base_storage == Private &&
/// source_points_to_aggregate && type_contains_pointer` rule). Mirroring that promotion closes the
/// addrspace-0 by-reference-struct helper gap (e.g. `ptr %0` into `%struct.*_attachment`).
pub(in crate::native) fn gep_base_storage(
    gep: &LlGep,
    storage: &HashMap<String, StorageClass>,
    named_types: &HashMap<String, LlType>,
) -> Option<StorageClass> {
    let base = local_storage(&gep.base.value, storage);
    let source_ty = resolve_named(&gep.source_ty, named_types);
    if base == Some(StorageClass::Private)
        && matches!(source_ty, LlType::Array(_, _) | LlType::Struct(_))
        && type_contains_pointer(&source_ty, named_types)
    {
        return Some(StorageClass::Function);
    }
    base
}

/// Expand a top-level `LlType::Named` against the module's named-type table (one level — nested
/// `Named`s are resolved on demand by [`type_contains_pointer`]). Returns the input unchanged when it is
/// not a known named type.
pub(in crate::native) fn resolve_named(
    ty: &LlType,
    named_types: &HashMap<String, LlType>,
) -> LlType {
    match ty {
        LlType::Named(name) => named_types.get(name).cloned().unwrap_or_else(|| ty.clone()),
        _ => ty.clone(),
    }
}

/// Whether a type holds a pointer anywhere in its tree (mirrors the emitter's `type_contains_pointer`),
/// resolving `Named` structs against the module table so a pointer buried in a nested named struct is
/// still seen (the emitter checks a fully-resolved type).
pub(in crate::native) fn type_contains_pointer(
    ty: &LlType,
    named_types: &HashMap<String, LlType>,
) -> bool {
    match ty {
        LlType::Ptr(_) => true,
        LlType::Vector(elem, _) | LlType::Array(elem, _) => {
            type_contains_pointer(elem, named_types)
        }
        LlType::Struct(fields) => fields.iter().any(|f| type_contains_pointer(f, named_types)),
        LlType::Named(name) => named_types
            .get(name)
            .is_some_and(|t| type_contains_pointer(t, named_types)),
        _ => false,
    }
}

/// Storage of a `bitcast`/`addrspacecast` result: copy from the source operand when it is a known
/// local pointer. The single source value is `operands[0]` (`resolve_operands` lowers a conversion's
/// `<srcty> <val>` to one typed operand).
pub(in crate::native) fn cast_source_storage(
    operands: &[TirOperand],
    storage: &HashMap<String, StorageClass>,
) -> Option<StorageClass> {
    operand_storage(operands.first(), storage)
}

/// Storage of a `freeze` result: copy from the frozen value — `operands[0]` (the one typed operand of
/// `freeze <ty> <val>`).
pub(in crate::native) fn freeze_source_storage(
    operands: &[TirOperand],
    storage: &HashMap<String, StorageClass>,
) -> Option<StorageClass> {
    operand_storage(operands.first(), storage)
}

/// Storage of a `select` result over two pointer arms: the common class when both known arms agree (or
/// the single known arm). `resolve_operands` lowers `select <condty> <cond>, <ty> <v1>, <ty> <v2>` to
/// `[cond, v1, v2]`, so the arms are `operands[1]` / `operands[2]`.
pub(in crate::native) fn select_arm_storage(
    operands: &[TirOperand],
    storage: &HashMap<String, StorageClass>,
) -> Option<StorageClass> {
    if operands.len() < 3 {
        return None;
    }
    merge_storage(
        operand_storage(operands.get(1), storage),
        operand_storage(operands.get(2), storage),
    )
}

/// Storage of a `phi` result over pointer arms: the common class across known incoming values. The
/// predecessor labels remain control-flow edges in the canonical phi carrier and are not visited.
pub(in crate::native) fn phi_incoming_storage<'a>(
    values: impl Iterator<Item = &'a LlValue>,
    storage: &HashMap<String, StorageClass>,
) -> Option<StorageClass> {
    let mut acc: Option<StorageClass> = None;
    let mut any = false;
    for value in values {
        if let Some(s) = local_storage(value, storage) {
            acc = if any {
                merge_storage(acc, Some(s))
            } else {
                Some(s)
            };
            any = true;
        }
    }
    acc
}

/// The recorded storage of a typed operand, when it resolves to a known local pointer. The bridge from
/// a `TirOperand` (via its `TypedValue`) to the [`local_storage`] lookup the arms above share.
pub(in crate::native) fn operand_storage(
    operand: Option<&TirOperand>,
    storage: &HashMap<String, StorageClass>,
) -> Option<StorageClass> {
    operand
        .and_then(TirOperand::as_typed_value)
        .and_then(|tv| local_storage(&tv.value, storage))
}

/// The recorded storage of a value, when it is a known pointer root or SSA value. Globals share the
/// emitter's name-keyed storage map with locals, so a GEP rooted directly at a global must consult the
/// supplied seed instead of falling back independently from its LLVM address space.
pub(in crate::native) fn local_storage(
    value: &LlValue,
    storage: &HashMap<String, StorageClass>,
) -> Option<StorageClass> {
    match value {
        LlValue::Local(name) | LlValue::Global(name) => storage.get(name).copied(),
        _ => None,
    }
}

/// Combine two arm storages: the common class when both are present and agree, otherwise the single
/// present one (an arm with unknown storage does not override a known one).
pub(in crate::native) fn merge_storage(
    a: Option<StorageClass>,
    b: Option<StorageClass>,
) -> Option<StorageClass> {
    match (a, b) {
        (Some(x), Some(y)) if x == y => Some(x),
        (Some(_), Some(_)) => None,
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

/// The `(pointer SSA name, implied pointee)` a single instruction's dereference pins down, or `None`
/// if the instruction is not a pointer dereference whose operand is a plain `%name`.
pub(in crate::native) fn deref_implied_pointee(inst: &TirInst) -> Option<(&str, LlType)> {
    // M-A5 reader reduction: dispatch on the graph's `inst.opcode` (the same
    // `rhs.split_whitespace().next()` token computed at build) and read the GEP source element type from
    // the carrier's `inst.gep_source_ty()` (set by `resolve_gep_source_ty` = `parse_gep(...).source_ty`, the
    // same `parse_type` on the same first comma field this branch used to re-lex). No `inst.text` read.
    match inst.opcode.as_str() {
        // `%r = load <ty>, ptr %p` — the loaded type IS %p's pointee; %p is the sole value operand.
        "load" => {
            let pointee = inst.result_ty.clone()?;
            let ptr = operand_name(inst.operands.first()?)?;
            Some((ptr, pointee))
        }
        // `store <ty> <v>, ptr %p` — operands are [value, pointer]; the value's type is %p's pointee.
        "store" => {
            let pointee = operand_type(inst.operands.first()?)?.clone();
            let ptr = operand_name(inst.operands.get(1)?)?;
            Some((ptr, pointee))
        }
        // `%r = getelementptr [inbounds] <srcty>, ptr %p, ...` — <srcty> is the base pointer's pointee
        // (carried on `gep_source_ty`); the base is the first GEP value operand. Skip a constant base.
        "getelementptr" => {
            let srcty = inst.gep_source_ty()?.clone();
            let ptr = operand_name(inst.operands.first()?)?;
            Some((ptr, srcty))
        }
        _ => None,
    }
}

/// The `%name` of a `Value` operand, or `None` for a constant/unresolved operand (a constant pointer
/// root — a global or `null` — needs no use-based pointee, its type is already known).
pub(in crate::native) fn operand_name(operand: &TirOperand) -> Option<&str> {
    match operand {
        TirOperand::Value { name, .. } => Some(name.as_str()),
        _ => None,
    }
}

/// The use-site type of a `Value`/`Const` operand, or `None` for `Unresolved`.
pub(in crate::native) fn operand_type(operand: &TirOperand) -> Option<&LlType> {
    match operand {
        TirOperand::Value { ty, .. } | TirOperand::Const { ty, .. } => Some(ty),
        TirOperand::Unresolved => None,
    }
}

/// Record a use-implied pointee, keeping the richer view and tallying genuine disagreements.
pub(in crate::native) fn record_use_pointee(
    map: &mut HashMap<String, LlType>,
    conflicts: &mut usize,
    ptr: &str,
    pointee: LlType,
) {
    match map.get(ptr) {
        None => {
            map.insert(ptr.to_string(), pointee);
        }
        Some(existing) if *existing == pointee => {}
        Some(existing) => {
            *conflicts += 1;
            if pointee_richness(&pointee) > pointee_richness(existing) {
                map.insert(ptr.to_string(), pointee);
            }
        }
    }
}

/// A coarse "informativeness" rank for resolving disagreeing use-pointees: an aggregate/vector view of
/// a pointer's storage subsumes a scalar view, which subsumes a byte (`i8`) view. Higher wins.
pub(in crate::native) fn pointee_richness(ty: &LlType) -> u8 {
    match ty {
        LlType::Struct(_) | LlType::Array(_, _) | LlType::Vector(_, _) => 3,
        LlType::Int(8) => 1,
        _ => 2,
    }
}
