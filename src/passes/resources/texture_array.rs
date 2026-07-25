//! Runtime-indexed argument-buffer texture arrays → Vulkan descriptor arrays.
//!
//! An `array_ref<texture2d>` or fixed `array<texture2d, N>` argument is a device buffer of texture
//! handles indexed at runtime:
//! ```text
//!   %e = getelementptr %"struct.metal::texture2d", ptr %argbuf, i64 %idx, i32 0
//!   %h = load ptr addrspace(1), ptr %e         ; the handle for element %idx
//!   ... air.sample_texture_2d(%h, ...)         ; sample it
//! ```
//! The stage-input pass binds such a param as `ParamBinding::ImageArray`:
//! a `UniformConstant OpTypeArray %image N` variable, recorded in `ctx.image_array_vars`, with the
//! param spliced to the array VARIABLE (not a loaded image). This pass then rewrites every per-element
//! handle load `%h = OpLoad <ptr> <chain-rooted-at-arrayvar>` in place into
//! `%p = OpAccessChain %ptr_image %arrayvar %idx ; %h = OpLoad %image %p`, and records `%h` in
//! `image_dims`/`image_comp` (and `image_storage` for storage-image element types) so the existing
//! sample/read/write/size-query lowering (`resolve_image_value`) treats `%h` as an ordinary loaded
//! image. The now-dead pointer derivations (the original access chain, any bitcast alias) are swept.
//!
//! Byte-safe by construction: the standard Vulkan bindless model — the descriptor at index `%idx`
//! carries the same texture the Metal handle referenced. Floor-safe: fires ONLY on a param the
//! stage-input pass bound as `ImageArray` (a texture-handle array shape whose real handle loads the
//! single-image path mis-emits as an illegal `OpLoad` of a pointer from an image); no single-texture
//! case is touched. The index is `air.is_uniform`-marked in the source, so plain uniform indexing is
//! legal (no `ShaderNonUniform`).

use std::collections::HashMap;

use crate::spirv_module::Instruction;
use crate::spirv_module::Operand;
use spirv::{Op, StorageClass, Word};

use super::super::{air_names, type_def_of, Ctx};

/// Materialize per-element image loads for every `ImageArray`-bound texture argument. No-op unless the
/// stage-input pass recorded at least one array-texture variable.
pub(crate) fn materialize_texture_array_loads(ctx: &mut Ctx, entry_idx: usize) {
    if ctx.image_array_vars.is_empty() {
        return;
    }
    let names = air_names(&ctx.module);

    // Clone the entry function's instruction defs (result_id -> instruction) so tracing is independent
    // of the mutation below.
    let mut defs: HashMap<Word, Instruction> = HashMap::new();
    for blk in &ctx.module.functions[entry_idx].blocks {
        for inst in &blk.instructions {
            if let Some(rid) = inst.result_id {
                defs.insert(rid, inst.clone());
            }
        }
    }
    let fixed_elements = ctx
        .emit_sidecar
        .local_pointer_field_loads
        .iter()
        .filter_map(|fact| fixed_array_element(ctx, fact).map(|resolved| (fact.id, resolved)))
        .collect::<Vec<_>>();
    let fixed_handles: HashMap<Word, (Word, Word)> = fixed_elements
        .into_iter()
        .map(|(id, (arrayvar, element))| (id, (arrayvar, ctx.const_uint(element))))
        .collect();
    let dynamic_handles: HashMap<Word, (Word, Word)> = ctx
        .emit_sidecar
        .local_pointer_dynamic_field_loads
        .iter()
        .filter_map(|fact| dynamic_array_element(ctx, fact).map(|resolved| (fact.id, resolved)))
        .collect();

    // 1. Find each texture-intrinsic call's handle operand (arg 0) whose defining load addresses an
    //    array element. Collect `(handle, load-pointer)` first (immutable module borrow), then resolve
    //    to `(arrayvar, idx)` — resolution mints a constant, which mutates `ctx`.
    let mut candidates: Vec<(Word, Word)> = Vec::new();
    let mut handles: HashMap<Word, (Word, Word)> = HashMap::new();
    let mut seen: std::collections::HashSet<Word> = std::collections::HashSet::new();
    for blk in &ctx.module.functions[entry_idx].blocks {
        for inst in &blk.instructions {
            if inst.class.opcode != Op::FunctionCall {
                continue;
            }
            let Some(Operand::IdRef(callee)) = inst.operands.first() else {
                continue;
            };
            let Some(name) = names.get(callee) else {
                continue;
            };
            if !is_texture_intrinsic(name) {
                continue;
            }
            let Some(Operand::IdRef(handle)) = inst.operands.get(1) else {
                continue;
            };
            if !seen.insert(*handle) {
                continue;
            }
            if let Some(&(arrayvar, idx)) = dynamic_handles.get(handle) {
                handles.insert(*handle, (arrayvar, idx));
                continue;
            }
            if let Some(&(arrayvar, idx)) = fixed_handles.get(handle) {
                handles.insert(*handle, (arrayvar, idx));
                continue;
            }
            // The handle must be an OpLoad of a pointer that roots at an array variable.
            let Some(load) = defs.get(handle) else {
                continue;
            };
            if load.class.opcode != Op::Load {
                continue;
            }
            let Some(Operand::IdRef(ptr)) = load.operands.first() else {
                continue;
            };
            candidates.push((*handle, *ptr));
        }
    }
    for (handle, ptr) in candidates {
        if let Some((arrayvar, idx)) = resolve_array_element(ctx, &defs, ptr) {
            handles.insert(handle, (arrayvar, idx));
        }
    }
    if handles.is_empty() {
        return;
    }

    // 2. For each handle load, build `%p = OpAccessChain %ptr_image %arrayvar %idx` and retype the load
    //    in place to `%handle = OpLoad %image %p`. Register the handle as a loaded image.
    let mut new_access_chains: HashMap<Word, Instruction> = HashMap::new(); // handle -> access-chain inst
    let mut retyped_load_ptr: HashMap<Word, (Word, Word)> = HashMap::new(); // handle -> (image_ty, p)
    let mut dead_roots: Vec<Word> = Vec::new();
    for (&handle, &(arrayvar, idx)) in &handles {
        let &(elem_image_ty, dim, comp) = ctx.image_array_vars.get(&arrayvar).unwrap();
        let ptr_image = ctx.ty_ptr(StorageClass::UniformConstant, elem_image_ty);
        let p = ctx.module.fresh_id();
        new_access_chains.insert(
            handle,
            Instruction::new(
                Op::AccessChain,
                Some(ptr_image),
                Some(p),
                vec![Operand::IdRef(arrayvar), Operand::IdRef(idx)],
            ),
        );
        retyped_load_ptr.insert(handle, (elem_image_ty, p));
        ctx.image_dims.insert(handle, dim);
        ctx.image_comp.insert(handle, comp);
        if image_type_is_storage(ctx, elem_image_ty) {
            ctx.image_storage.insert(handle);
        }
        // The original load's pointer operand becomes dead; record its id for the sweep.
        if let Some(load) = defs.get(&handle) {
            if let Some(Operand::IdRef(old_ptr)) = load.operands.first() {
                if *old_ptr != arrayvar {
                    dead_roots.push(*old_ptr);
                }
            }
        }
    }

    // 3. Apply: rewrite each block's instruction list, inserting the access chain before each retyped
    //    load and retyping the load's pointer operand + result type.
    let func = &mut ctx.module.functions[entry_idx];
    for blk in &mut func.blocks {
        let mut rebuilt: Vec<Instruction> =
            Vec::with_capacity(blk.instructions.len() + handles.len());
        for inst in blk.instructions.drain(..) {
            if let Some(handle) = inst.result_id {
                if let (Some(chain), Some(&(image_ty, p))) = (
                    new_access_chains.get(&handle),
                    retyped_load_ptr.get(&handle),
                ) {
                    rebuilt.push(chain.clone());
                    let mut load = inst;
                    load.class.opcode = Op::Load;
                    load.result_type = Some(image_ty);
                    load.operands = vec![Operand::IdRef(p)];
                    rebuilt.push(load);
                    continue;
                }
            }
            rebuilt.push(inst);
        }
        blk.instructions = rebuilt;
    }

    // 4. Sweep the now-dead pointer derivations (the old access chains + any bitcast/copyobject alias
    //    of the array pointer). A pure pointer-derivation with no remaining use is safe to delete; run
    //    to a fixpoint so freeing one frees its sources.
    sweep_dead_pointer_derivations(ctx, entry_idx, &dead_roots);
}

/// The `air.*` texture intrinsics whose arg 0 is a texture handle (sampled/read/write + size queries).
fn is_texture_intrinsic(name: &str) -> bool {
    name.starts_with("air.sample_texture")
        || name.starts_with("air.sample_depth")
        || name.starts_with("air.read_texture")
        || name.starts_with("air.read_depth")
        || name.starts_with("air.write_texture")
        || name.starts_with("air.gather_texture")
        || name.starts_with("air.gather_depth")
        || name.starts_with("air.get_width_texture")
        || name.starts_with("air.get_height_texture")
        || name.starts_with("air.get_depth_texture")
        || name.starts_with("air.get_array_size_texture")
        || name.starts_with("air.get_width_depth")
        || name.starts_with("air.get_height_depth")
        || name.starts_with("air.get_depth_depth")
        || name.starts_with("air.get_num_mip_levels_texture")
        || name.starts_with("air.get_num_mip_levels_depth")
        || name.starts_with("air.get_num_samples_texture")
        || name.starts_with("air.is_null_texture")
}

fn image_type_is_storage(ctx: &Ctx, image_ty: Word) -> bool {
    type_def_of(ctx, image_ty)
        .filter(|def| def.class.opcode == Op::TypeImage)
        .and_then(|def| match def.operands.get(5) {
            Some(Operand::LiteralBit32(sampled)) => Some(*sampled == 2),
            _ => None,
        })
        .unwrap_or(false)
}

fn dynamic_array_element(
    ctx: &Ctx,
    fact: &crate::emit_sidecar::LocalPointerDynamicFieldLoad,
) -> Option<(Word, Word)> {
    if !ctx.image_array_vars.contains_key(&fact.root) || !fact.prefix.is_empty() {
        return None;
    }
    if !(fact.suffix.is_empty() || fact.suffix.as_slice() == [0]) {
        return None;
    }
    Some((fact.root, fact.index))
}

fn fixed_array_element(
    ctx: &Ctx,
    fact: &crate::emit_sidecar::LocalPointerFieldLoad,
) -> Option<(Word, u32)> {
    if !ctx.image_array_vars.contains_key(&fact.root) {
        return None;
    }
    match fact.indices.as_slice() {
        [element] | [element, 0] => Some((fact.root, *element)),
        _ => None,
    }
}

/// Given the pointer operand of a handle load, return `(arrayvar, element_index)` if it addresses an
/// element of an `ImageArray`-bound texture array. A direct load of the array variable is element 0;
/// an access chain whose first index is the element index roots at the array (through bitcast /
/// copyobject aliases of the whole-array pointer).
fn resolve_array_element(
    ctx: &mut Ctx,
    defs: &HashMap<Word, Instruction>,
    ptr: Word,
) -> Option<(Word, Word)> {
    if ctx.image_array_vars.contains_key(&ptr) {
        let zero = ctx.const_uint(0);
        return Some((ptr, zero));
    }
    let inst = defs.get(&ptr)?;
    match inst.class.opcode {
        Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain => {
            let Operand::IdRef(base) = inst.operands.first()? else {
                return None;
            };
            let arrayvar = resolve_ptr_root(ctx, defs, *base)?;
            // The first index selects the array element (a `PtrAccessChain` element index or the
            // leading array index of the struct-array gep).
            let Operand::IdRef(idx) = inst.operands.get(1)? else {
                return None;
            };
            Some((arrayvar, *idx))
        }
        Op::Bitcast | Op::CopyObject => {
            let Operand::IdRef(src) = inst.operands.first()? else {
                return None;
            };
            resolve_array_element(ctx, defs, *src)
        }
        _ => None,
    }
}

/// Follow bitcast / copyobject aliases of a whole-array pointer to the underlying `image_array_vars`
/// variable, if any.
fn resolve_ptr_root(ctx: &Ctx, defs: &HashMap<Word, Instruction>, mut ptr: Word) -> Option<Word> {
    for _ in 0..8 {
        if ctx.image_array_vars.contains_key(&ptr) {
            return Some(ptr);
        }
        let inst = defs.get(&ptr)?;
        match inst.class.opcode {
            Op::Bitcast | Op::CopyObject => {
                let Operand::IdRef(src) = inst.operands.first()? else {
                    return None;
                };
                ptr = *src;
            }
            _ => return None,
        }
    }
    None
}

/// Delete dead pure pointer-derivations (AccessChain/PtrAccessChain/Bitcast/CopyObject) reachable from
/// `roots`, iterating to a fixpoint. An instruction is removed only when no surviving instruction (nor
/// a global/decoration/name) references its result — so this can only drop provably-dead code.
fn sweep_dead_pointer_derivations(ctx: &mut Ctx, entry_idx: usize, roots: &[Word]) {
    let mut candidates: Vec<Word> = roots.to_vec();
    loop {
        // Compute the set of ids currently referenced anywhere.
        let mut used: std::collections::HashSet<Word> = std::collections::HashSet::new();
        for f in &ctx.module.functions {
            for blk in &f.blocks {
                for inst in &blk.instructions {
                    for op in &inst.operands {
                        if let Operand::IdRef(id) = op {
                            used.insert(*id);
                        }
                    }
                }
            }
        }
        for inst in ctx
            .module
            .types_global_values
            .iter()
            .chain(ctx.new_globals.iter())
            .chain(ctx.module.debug_names.iter())
            .chain(ctx.module.annotations.iter())
            .chain(ctx.module.entry_points.iter())
        {
            for op in &inst.operands {
                if let Operand::IdRef(id) = op {
                    used.insert(*id);
                }
            }
        }

        let func = &mut ctx.module.functions[entry_idx];
        let mut removed_any = false;
        let mut freed_sources: Vec<Word> = Vec::new();
        for blk in &mut func.blocks {
            blk.instructions.retain(|inst| {
                let Some(rid) = inst.result_id else {
                    return true;
                };
                let is_pointer_deriv = matches!(
                    inst.class.opcode,
                    Op::AccessChain
                        | Op::InBoundsAccessChain
                        | Op::PtrAccessChain
                        | Op::Bitcast
                        | Op::CopyObject
                );
                if is_pointer_deriv && candidates.contains(&rid) && !used.contains(&rid) {
                    for op in &inst.operands {
                        if let Operand::IdRef(id) = op {
                            freed_sources.push(*id);
                        }
                    }
                    removed_any = true;
                    return false;
                }
                true
            });
        }
        if !removed_any {
            break;
        }
        candidates.extend(freed_sources);
    }
}
