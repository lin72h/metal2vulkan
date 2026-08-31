//! W1 (frontier-walls plan): PhysicalStorageBuffer64 lowering for the cross-binding pointer-merge
//! sub-graph.
//!
//! Under Logical addressing an `OpSelect`/`OpPhi` over pointers from DISTINCT descriptor bindings is
//! illegal — spirv-val: *"Variable pointers must point into the same structure (or OpConstantNull)"*.
//! It is the dominant frontier class (a kernel selecting among N device buffers by a runtime index).
//! Under **PhysicalStorageBuffer64** such pointers are plain 64-bit addresses and the merge is
//! unconditionally legal.
//!
//! This pass rewrites the merge sub-graph (the connected component of StorageBuffer-pointer values
//! containing a cross-binding merge) in place on the already-emitted module:
//! - memory model → `PhysicalStorageBuffer64`, caps += `PhysicalStorageBufferAddresses`/`Int64`,
//!   ext `SPV_KHR_physical_storage_buffer`;
//! - each leaf access chain (zero-offset into a buffer variable) → `OpConvertUToPtr` of the buffer's
//!   64-bit device address, loaded from a synthesized address-table descriptor (`{ runtimearray u64 }`
//!   in the selected translator-owned binding range and descriptor set) indexed by the buffer's OWN descriptor Binding — so the executor
//!   fills `table[binding] = vkGetBufferDeviceAddress(buffer_at_binding)` directly from the bound
//!   resources, no side-channel slot→binding map needed (the consumer's
//!   `buffer_device_address` ABI fills the same table);
//! - the merges / `OpPtrAccessChain`s are retyped to PhysicalStorageBuffer pointers (ArrayStride-
//!   decorated, required for `OpPtrAccessChain` on a physical pointer);
//! - loads/stores through the sub-graph get an `Aligned` operand.
//!
//! Pointer-phi lowering is applied at interface construction, immediately after Metal parameters
//! acquire their final descriptor roots. The lowering is general over the element
//! type/stride and decides purely from IR structure (storage class, access-chain roots, the
//! cross-binding property) — never a shader name.

use crate::spirv_module::Instruction;
use crate::spirv_module::Module;
use crate::spirv_module::Operand;
use spirv::{Capability, Decoration, MemoryModel, Op, StorageClass, Word};
use std::collections::{HashMap, HashSet, VecDeque};

/// One leaf of a cross-binding merge closure: a load-reachable pointer that resolves to a single
/// buffer variable, plus the array-element index its access chain carries (if any).
struct Leaf {
    id: Word,
    var: Word,
    pointee: Word,
    // The array-element index of the leaf access chain, when it carries one beyond the
    // zero-offset member path (`%buf %uint_0 %j` → `Some(j)`); `None` for a pure base-address
    // leaf (all indices const-zero). The element pointer is then
    // `OpPtrAccessChain(ConvertUToPtr(base_address), element)` — byte-identical to the descriptor
    // access because the data array sits at struct byte 0 with the pointee's allocation stride.
    element: Option<Word>,
}

/// The PhysicalStorageBuffer64 lowering plan produced by discovery: the type/SSA lookup tables plus
/// the validated cross-binding merge closure, its leaves, and the per-buffer slot assignment.
struct PsbDiscovery {
    type_defs: HashMap<Word, Instruction>,
    var_storage: HashMap<Word, StorageClass>,
    var_pointee: HashMap<Word, Word>,
    value_type: HashMap<Word, Word>,
    cross_binding_merges: Vec<Word>,
    has_cross_binding_phi: bool,
    closure: HashSet<Word>,
    closure_values: Vec<Word>,
    leaves: Vec<Leaf>,
    buffer_slot: HashMap<Word, u32>,
    const_ids: HashSet<Word>,
}

/// Allocation stride for a scalar/vector element converted to a physical pointer.
///
/// Prefer an existing `ArrayStride` on either a pointer-to-element or an array-of-element: those
/// decorations were produced by the sidecar-aware layout pass and retain non-default AIR vector
/// alignment. Conflicting declarations are rejected. With no explicit evidence, use scalar size or
/// LLVM's ordinary power-of-two vector allocation rule.
fn element_allocation_stride(
    module: &Module,
    type_defs: &HashMap<Word, Instruction>,
    ty: Word,
) -> Option<u32> {
    let mut declared = None;
    for annotation in &module.annotations {
        if annotation.class.opcode != Op::Decorate {
            continue;
        }
        let [Operand::IdRef(target), Operand::Decoration(Decoration::ArrayStride), Operand::LiteralBit32(stride)] =
            annotation.operands.as_slice()
        else {
            continue;
        };
        let Some(definition) = type_defs.get(target) else {
            continue;
        };
        let carries_ty = match definition.class.opcode {
            Op::TypePointer => definition.operands.get(1) == Some(&Operand::IdRef(ty)),
            Op::TypeArray | Op::TypeRuntimeArray => {
                definition.operands.first() == Some(&Operand::IdRef(ty))
            }
            _ => false,
        };
        if !carries_ty {
            continue;
        }
        match declared {
            Some(existing) if existing != *stride => return None,
            _ => declared = Some(*stride),
        }
    }
    if declared.is_some() {
        return declared;
    }

    let definition = type_defs.get(&ty)?;
    let store_size = match definition.class.opcode {
        Op::TypeInt | Op::TypeFloat => match definition.operands.first()? {
            Operand::LiteralBit32(bits) => bits.div_ceil(8),
            _ => return None,
        },
        Op::TypeVector => {
            let (Operand::IdRef(component), Operand::LiteralBit32(count)) =
                (definition.operands.first()?, definition.operands.get(1)?)
            else {
                return None;
            };
            let component = type_defs.get(component)?;
            let bits = match component.class.opcode {
                Op::TypeInt | Op::TypeFloat => match component.operands.first()? {
                    Operand::LiteralBit32(bits) => *bits,
                    _ => return None,
                },
                _ => return None,
            };
            bits.div_ceil(8).checked_mul(*count)?
        }
        _ => return None,
    };
    store_size.max(1).checked_next_power_of_two()
}

/// Physical-storage memory operations require an explicit alignment at least as large as the
/// accessed scalar component. Preserve any stronger declaration and insert the Aligned operand in
/// its grammar-defined position when another memory-access flag is already present.
fn ensure_physical_memory_alignment(instruction: &mut Instruction, required: u32) {
    let Some(memory_access_index) = instruction
        .operands
        .iter()
        .position(|operand| matches!(operand, Operand::MemoryAccess(_)))
    else {
        instruction
            .operands
            .push(Operand::MemoryAccess(spirv::MemoryAccess::ALIGNED));
        instruction.operands.push(Operand::LiteralBit32(required));
        return;
    };
    let Operand::MemoryAccess(memory_access) = instruction.operands[memory_access_index] else {
        unreachable!();
    };
    if memory_access.contains(spirv::MemoryAccess::ALIGNED) {
        if let Some(Operand::LiteralBit32(alignment)) =
            instruction.operands.get_mut(memory_access_index + 1)
        {
            *alignment = (*alignment).max(required);
        }
    } else {
        instruction.operands[memory_access_index] =
            Operand::MemoryAccess(memory_access | spirv::MemoryAccess::ALIGNED);
        instruction
            .operands
            .insert(memory_access_index + 1, Operand::LiteralBit32(required));
    }
}

/// Discovery + lowerability gate (pure): find the cross-binding pointer-merge closure, prove it is
/// PSB-lowerable, and collect its leaves + per-buffer address-table slots. Returns `None` if there is
/// nothing to lower or the closure is not lowerable.
fn discover_cross_binding_psb(module: &Module) -> Option<PsbDiscovery> {
    let type_defs: HashMap<Word, Instruction> = module
        .types_global_values
        .iter()
        .filter_map(|i| i.result_id.map(|id| (id, i.clone())))
        .collect();
    // module-scope OpVariable id -> its storage class (the descriptor-bound buffers are the roots).
    let var_storage: HashMap<Word, StorageClass> = module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::Variable)
        .filter_map(|i| {
            let id = i.result_id?;
            match i.operands.first()? {
                Operand::StorageClass(s) => Some((id, *s)),
                _ => None,
            }
        })
        .collect();
    // module-scope variable id -> its descriptor Binding number. The address-table slot for each
    // merged buffer is its OWN descriptor binding, so the executor can fill the table deterministically
    // from the bound resources (`table[binding] = vkGetBufferDeviceAddress(buffer_at_binding)`) without
    // a side-channel slot->binding map.
    let var_binding: HashMap<Word, u32> = module
        .annotations
        .iter()
        .filter(|i| {
            i.class.opcode == Op::Decorate
                && matches!(
                    i.operands.get(1),
                    Some(Operand::Decoration(Decoration::Binding))
                )
        })
        .filter_map(|i| match (i.operands.first()?, i.operands.get(2)?) {
            (Operand::IdRef(id), Operand::LiteralBit32(b)) => Some((*id, *b)),
            _ => None,
        })
        .collect();

    // (storage class, pointee type) of a pointer type id.
    let ptr_info = |ty: Word| -> Option<(StorageClass, Word)> {
        let inst = type_defs.get(&ty)?;
        if inst.class.opcode != Op::TypePointer {
            return None;
        }
        match (inst.operands.first()?, inst.operands.get(1)?) {
            (Operand::StorageClass(s), Operand::IdRef(pointee)) => Some((*s, *pointee)),
            _ => None,
        }
    };

    // module-scope buffer variable id -> its pointee (struct) type, for the whole-buffer leaf path
    // (a buffer variable appearing DIRECTLY as a select/phi arm — a cross-binding merge at the whole
    // -buffer level rather than the element level).
    let var_pointee: HashMap<Word, Word> = module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::Variable)
        .filter_map(|i| {
            let id = i.result_id?;
            let ty = i.result_type?;
            let (_, pointee) = ptr_info(ty)?;
            Some((id, pointee))
        })
        .collect();
    let is_buffer_var =
        |id: Word| -> bool { var_storage.get(&id) == Some(&StorageClass::StorageBuffer) };

    // Whole-function value def map (every SSA result -> defining instruction + result type).
    let mut value_def: HashMap<Word, Instruction> = HashMap::new();
    let mut value_type: HashMap<Word, Word> = HashMap::new();
    for function in &module.functions {
        for block in &function.blocks {
            for inst in &block.instructions {
                if let Some(rid) = inst.result_id {
                    if let Some(rty) = inst.result_type {
                        if ptr_info(rty)
                            .is_some_and(|(storage, _)| storage == StorageClass::StorageBuffer)
                        {
                            value_type.insert(rid, rty);
                            value_def.insert(rid, inst.clone());
                        }
                    }
                }
            }
        }
    }

    // The pointer SSA values we track: any result whose type is a StorageBuffer pointer and whose op
    // is one we lower (access chain / select / phi / ptr access chain). Edges connect a value to the
    // pointer values it directly derives from (base for an access chain, arms for select/phi).
    let is_sb_pointer = |id: Word| -> bool {
        value_type
            .get(&id)
            .and_then(|t| ptr_info(*t))
            .map(|(sc, _)| sc == StorageClass::StorageBuffer)
            .unwrap_or(false)
    };
    let pointer_operands = |inst: &Instruction| -> Vec<Word> {
        match inst.class.opcode {
            Op::AccessChain | Op::InBoundsAccessChain | Op::PtrAccessChain => inst
                .operands
                .first()
                .and_then(|o| match o {
                    Operand::IdRef(b) => Some(*b),
                    _ => None,
                })
                .into_iter()
                .collect(),
            // A select/phi arm may be a function-level StorageBuffer pointer (element merge) OR a
            // module-scope buffer variable directly (whole-buffer merge — the buffer base itself is
            // selected, with the indexing access chain APPLIED to the merged pointer downstream).
            Op::Select => inst.operands[1..]
                .iter()
                .filter_map(|o| match o {
                    Operand::IdRef(b) => Some(*b),
                    _ => None,
                })
                .filter(|id| is_sb_pointer(*id) || is_buffer_var(*id))
                .collect(),
            Op::Phi => inst
                .operands
                .chunks(2)
                .filter_map(|c| match c.first() {
                    Some(Operand::IdRef(v)) => Some(*v),
                    _ => None,
                })
                .filter(|id| is_sb_pointer(*id) || is_buffer_var(*id))
                .collect(),
            // A redundant whole-pointer alias (`%r = OpCopyObject %ptrT %src`) is transparent: its
            // source is the same StorageBuffer pointer / buffer variable. Treat it like a single-arm
            // merge so a copy of a closure buffer var is pulled into the closure and lowered rather than
            // left dangling (which would fail the self-containment check).
            Op::CopyObject => inst
                .operands
                .first()
                .and_then(|o| match o {
                    Operand::IdRef(b) => Some(*b),
                    _ => None,
                })
                .filter(|id| is_sb_pointer(*id) || is_buffer_var(*id))
                .into_iter()
                .collect(),
            _ => Vec::new(),
        }
    };

    // Propagate descriptor roots through the complete pointer graph. Construct-tree emission can
    // express one source switch as a ladder of nested pointer phis, so looking only through a
    // single access-chain spine misses the cross-binding property once an arm is itself a merge.
    // The three-state lattice is sufficient here: construction only needs to distinguish no root,
    // one exact root, and multiple roots. A worklist reaches a fixed point even for loop-carried
    // pointer phis, while each value can change state at most twice.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum RootState {
        None,
        One(Word),
        Multiple,
    }
    impl RootState {
        fn union(self, other: Self) -> Self {
            match (self, other) {
                (Self::Multiple, _) | (_, Self::Multiple) => Self::Multiple,
                (Self::None, state) | (state, Self::None) => state,
                (Self::One(left), Self::One(right)) if left == right => Self::One(left),
                (Self::One(_), Self::One(_)) => Self::Multiple,
            }
        }
    }
    let dependencies = value_def
        .iter()
        .filter(|(id, _)| is_sb_pointer(**id))
        .map(|(id, definition)| (*id, pointer_operands(definition)))
        .collect::<HashMap<_, _>>();
    let mut dependents = HashMap::<Word, Vec<Word>>::new();
    for (&value, operands) in &dependencies {
        for &operand in operands {
            dependents.entry(operand).or_default().push(value);
        }
    }
    let mut root_states = var_storage
        .iter()
        .filter_map(|(&id, &storage)| {
            (storage == StorageClass::StorageBuffer).then_some((id, RootState::One(id)))
        })
        .collect::<HashMap<_, _>>();
    root_states.extend(dependencies.keys().map(|&id| (id, RootState::None)));
    let mut root_worklist = dependencies.keys().copied().collect::<VecDeque<_>>();
    while let Some(value) = root_worklist.pop_front() {
        let next = dependencies.get(&value).into_iter().flatten().fold(
            RootState::None,
            |state, operand| {
                state.union(root_states.get(operand).copied().unwrap_or(RootState::None))
            },
        );
        if root_states.get(&value).copied() != Some(next) {
            root_states.insert(value, next);
            root_worklist.extend(dependents.get(&value).into_iter().flatten().copied());
        }
    }

    // Find every select/phi whose transitive pointer arms reach distinct descriptor roots.
    let mut cross_binding_merges: Vec<Word> = Vec::new();
    for (id, def) in &value_def {
        if !matches!(def.class.opcode, Op::Select | Op::Phi) {
            continue;
        }
        if !is_sb_pointer(*id) {
            continue;
        }
        if root_states.get(id) == Some(&RootState::Multiple) {
            cross_binding_merges.push(*id);
        }
    }
    cross_binding_merges.sort_unstable();
    if cross_binding_merges.is_empty() {
        return None;
    }

    // Closure: connected component (over derive-edges, both directions) of every cross-binding merge.
    // Build child edges (value -> its pointer parents) and parent->child (reverse) for downstream
    // ptr-access-chains/merges that consume a closure pointer.
    let mut children: HashMap<Word, Vec<Word>> = HashMap::new(); // value -> pointer values it derives from
    let mut parents: HashMap<Word, Vec<Word>> = HashMap::new(); // pointer value -> values that derive from it
    for (id, def) in &value_def {
        if !is_sb_pointer(*id) {
            continue;
        }
        // A buffer variable becomes a closure node ONLY as a direct merge arm (whole-buffer select/phi)
        // or a whole-pointer alias (`OpCopyObject`) — never as an access-chain base (that is an element
        // leaf, handled in-place by `buffer_slot`).
        let is_merge = matches!(def.class.opcode, Op::Select | Op::Phi | Op::CopyObject);
        let ops = pointer_operands(def);
        for p in &ops {
            if is_sb_pointer(*p) || (is_merge && is_buffer_var(*p)) {
                children.entry(*id).or_default().push(*p);
                parents.entry(*p).or_default().push(*id);
            }
        }
    }
    let mut closure: HashSet<Word> = HashSet::new();
    let mut stack: Vec<Word> = cross_binding_merges.clone();
    while let Some(v) = stack.pop() {
        if !closure.insert(v) {
            continue;
        }
        for n in children.get(&v).into_iter().flatten() {
            stack.push(*n);
        }
        for n in parents.get(&v).into_iter().flatten() {
            stack.push(*n);
        }
    }
    let mut closure_values: Vec<Word> = closure.iter().copied().collect();
    closure_values.sort_unstable();

    // Classify the closure and verify it is lowerable. Leaves = access chains, zero-offset, into a
    // StorageBuffer buffer variable, with a scalar/vector pointee. Internal = select/phi/ptr-access.
    // Constant-zero ids (for the zero-offset check) and the uint type for the address-table index.
    let const_zero_ids: HashSet<Word> = module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::Constant)
        .filter(|i| matches!(i.operands.first(), Some(Operand::LiteralBit32(0))))
        .filter_map(|i| i.result_id)
        .collect();
    // All scalar-constant ids — used to tell a constant struct-member selector (legal directly) from a
    // DYNAMIC index applied straight to a whole-buffer struct base (illegal — the flat-element case the
    // post-merge rewrite below must lower via physical address arithmetic).
    let const_ids: HashSet<Word> = module
        .types_global_values
        .iter()
        .filter(|i| matches!(i.class.opcode, Op::Constant | Op::ConstantNull))
        .filter_map(|i| i.result_id)
        .collect();

    let mut leaves: Vec<Leaf> = Vec::new();
    let mut buffer_slot: HashMap<Word, u32> = HashMap::new();
    for &v in &closure_values {
        // Whole-buffer leaf: a buffer variable appearing directly as a merge arm. Its pointee is the
        // buffer's STRUCT type; the merged pointer is the buffer base address (zero element offset),
        // with the indexing access chain applied downstream as a physical access chain. (Module-scope
        // variables are not in `value_def`, so handle them before the SSA lookup below.)
        if is_buffer_var(v) {
            let &pointee = var_pointee.get(&v)?;
            if let std::collections::hash_map::Entry::Vacant(e) = buffer_slot.entry(v) {
                let &binding = var_binding.get(&v)?;
                e.insert(binding);
            }
            leaves.push(Leaf {
                id: v,
                var: v,
                pointee,
                element: None,
            });
            continue;
        }
        let def = value_def.get(&v).unwrap();
        match def.class.opcode {
            Op::AccessChain | Op::InBoundsAccessChain if matches!(def.operands.first(), Some(Operand::IdRef(b)) if is_buffer_var(*b)) =>
            {
                // Leaf into a buffer variable; scalar/vector pointee. The member path must land at
                // struct byte 0 (every index but the last const-zero) so the buffer's device address
                // is the element-0 address; the LAST index may be a runtime array-element index,
                // re-applied below as an `OpPtrAccessChain` over the physical base pointer.
                let Some(Operand::IdRef(base)) = def.operands.first() else {
                    return None;
                };
                if var_storage.get(base) != Some(&StorageClass::StorageBuffer) {
                    return None;
                }
                let indices = &def.operands[1..];
                if indices.is_empty() {
                    return None;
                }
                // every index but the last must be constant zero (member path to byte 0).
                let prefix_zero = indices[..indices.len() - 1].iter().all(|o| match o {
                    Operand::IdRef(idx) => const_zero_ids.contains(idx),
                    _ => false,
                });
                if !prefix_zero {
                    return None;
                }
                let Operand::IdRef(last) = indices[indices.len() - 1] else {
                    return None;
                };
                // The element index is the last operand; a const-zero last index is a pure base
                // address (no `OpPtrAccessChain` needed), matching the prior zero-offset behavior.
                let element = (!const_zero_ids.contains(&last)).then_some(last);
                let pointee = match value_type.get(&v).and_then(|t| ptr_info(*t)) {
                    Some((_, p)) => p,
                    None => return None,
                };
                element_allocation_stride(module, &type_defs, pointee)?;
                if !buffer_slot.contains_key(base) {
                    // The slot IS the buffer's descriptor binding (executor-fillable ABI). A merged
                    // buffer that carries no Binding decoration cannot be addressed by the executor —
                    // bail rather than emit an unfillable table.
                    let &binding = var_binding.get(base)?;
                    buffer_slot.insert(*base, binding);
                }
                leaves.push(Leaf {
                    id: v,
                    var: *base,
                    pointee,
                    element,
                });
            }
            // A post-merge access chain (its base is a merged pointer, not a buffer variable) indexes
            // INTO the selected whole buffer; it is retyped to a physical access chain below (its
            // indices are unchanged — byte-identical, since the struct's member Offsets / array
            // ArrayStride carry over to the physical pointee). No leaf, no value rewrite.
            Op::AccessChain | Op::InBoundsAccessChain => {}
            // Function-local pointer undef is a permitted merge arm, not a descriptor leaf. The
            // synthesis phase below already replaces nullish/undef arms with a dominating zero
            // address converted to the exact physical pointer type.
            Op::Select | Op::Phi | Op::PtrAccessChain | Op::CopyObject | Op::Undef => {}
            _ => {
                return None;
            }
        }
    }
    if leaves.is_empty() {
        return None;
    }

    // Atomic ops whose first operand is the accessed pointer. They are legal consumers of a closure
    // pointer: under PhysicalStorageBuffer64 the atomic stays an atomic over the SAME device memory
    // (byte-correct — the merged buffer's real address is read/modified atomically), and the op carries
    // no `MemoryAccess` operand so it needs no `Aligned` rewrite.
    let is_atomic_ptr_op = |op: Op| -> bool {
        matches!(
            op,
            Op::AtomicLoad
                | Op::AtomicStore
                | Op::AtomicExchange
                | Op::AtomicCompareExchange
                | Op::AtomicCompareExchangeWeak
                | Op::AtomicIIncrement
                | Op::AtomicIDecrement
                | Op::AtomicIAdd
                | Op::AtomicISub
                | Op::AtomicSMin
                | Op::AtomicUMin
                | Op::AtomicSMax
                | Op::AtomicUMax
                | Op::AtomicAnd
                | Op::AtomicOr
                | Op::AtomicXor
                | Op::AtomicFAddEXT
                | Op::AtomicFMinEXT
                | Op::AtomicFMaxEXT
        )
    };

    // Self-containment: every USE of a closure pointer must be another closure pointer op, an
    // OpLoad/OpStore/atomic, or (harmlessly) appear inside the closure ops themselves. A use elsewhere
    // (a function call, a typed access we don't retype) would be left dangling -> bail.
    let mut load_store_ptr_uses: HashSet<Word> = HashSet::new();
    for function in &module.functions {
        for block in &function.blocks {
            for inst in &block.instructions {
                let in_closure_def = inst
                    .result_id
                    .map(|r| closure.contains(&r))
                    .unwrap_or(false);
                match inst.class.opcode {
                    Op::Load => {
                        if let Some(Operand::IdRef(p)) = inst.operands.first() {
                            if closure.contains(p) {
                                load_store_ptr_uses.insert(inst.result_id.unwrap_or(0));
                                continue;
                            }
                        }
                    }
                    Op::Store => {
                        if let Some(Operand::IdRef(p)) = inst.operands.first() {
                            if closure.contains(p) {
                                continue;
                            }
                        }
                    }
                    op if is_atomic_ptr_op(op) => {
                        if let Some(Operand::IdRef(p)) = inst.operands.first() {
                            if closure.contains(p) {
                                continue;
                            }
                        }
                    }
                    _ => {}
                }
                // For any non-closure, non-load/store instruction, no operand may reference a closure
                // pointer (except the closure ops themselves, handled by in_closure_def).
                if !in_closure_def {
                    for op in &inst.operands {
                        if let Operand::IdRef(id) = op {
                            // Whole-buffer variables are not retyped or replaced in place. Their
                            // ordinary descriptor accesses remain valid alongside the synthesized
                            // physical base address used by the merge closure.
                            if closure.contains(id) && !is_buffer_var(*id) {
                                return None;
                            }
                        }
                    }
                }
            }
        }
    }
    let _ = load_store_ptr_uses;

    let has_cross_binding_phi = cross_binding_merges.iter().any(|id| {
        value_def
            .get(id)
            .is_some_and(|inst| inst.class.opcode == Op::Phi)
    });

    Some(PsbDiscovery {
        type_defs,
        var_storage,
        var_pointee,
        value_type,
        cross_binding_merges,
        has_cross_binding_phi,
        closure,
        closure_values,
        leaves,
        buffer_slot,
        const_ids,
    })
}

/// Rewrite every cross-binding StorageBuffer pointer-merge in `module` into a
/// PhysicalStorageBuffer64 device-address lowering. Returns true if any rewrite was applied. Two
/// stages: discover the lowerable closure (pure), then synthesize the address table + physical
/// element pointers and rewrite the merges (mutating).
#[cfg(test)]
pub(super) fn rewrite_cross_binding_pointer_merges(module: &mut Module) -> bool {
    construct_cross_binding_pointer_merges_with_layout(
        module,
        crate::reflect::DescriptorLayout::default(),
    )
    .is_some()
}

pub(super) fn construct_cross_binding_pointer_merges_with_layout(
    module: &mut Module,
    layout: crate::reflect::DescriptorLayout,
) -> Option<Word> {
    rewrite_cross_binding_pointer_merges_inner(module, false, layout)
}

/// Like [`rewrite_cross_binding_pointer_merges`], but declines a closure that has no cross-binding
/// pointer phi. This lets the primary path retain its portable value-domain lowering for ordinary
/// `OpSelect`s while using the address-table form only for the phi family whose post-merge dynamic
/// accesses cannot be replayed at predecessor edges.
#[cfg(test)]
pub(super) fn rewrite_cross_binding_pointer_phis(module: &mut Module) -> bool {
    construct_cross_binding_pointer_phis_with_layout(
        module,
        crate::reflect::DescriptorLayout::default(),
    )
    .is_some()
}

pub(super) fn construct_cross_binding_pointer_phis_with_layout(
    module: &mut Module,
    layout: crate::reflect::DescriptorLayout,
) -> Option<Word> {
    rewrite_cross_binding_pointer_merges_inner(module, true, layout)
}

/// True when the module contains a lowerable cross-binding pointer closure with an `OpPhi`.
/// This is intentionally the same discovery gate as construction, but leaves the module untouched
/// so the structural tests can verify select-only and phi-containing closures independently.
#[cfg(test)]
pub(super) fn has_cross_binding_pointer_phi(module: &Module) -> bool {
    discover_cross_binding_psb(module).is_some_and(|discovery| discovery.has_cross_binding_phi)
}

fn rewrite_cross_binding_pointer_merges_inner(
    module: &mut Module,
    require_cross_binding_phi: bool,
    layout: crate::reflect::DescriptorLayout,
) -> Option<Word> {
    let storage_buffer_pointer_types = module
        .types_global_values
        .iter()
        .filter_map(|instruction| {
            (instruction.class.opcode == Op::TypePointer
                && instruction.operands.first()
                    == Some(&Operand::StorageClass(StorageClass::StorageBuffer)))
            .then_some(instruction.result_id?)
        })
        .collect::<HashSet<_>>();
    let has_pointer_merge = module
        .functions
        .iter()
        .flat_map(|function| &function.blocks)
        .flat_map(|block| &block.instructions)
        .any(|instruction| {
            matches!(instruction.class.opcode, Op::Phi | Op::Select)
                && instruction
                    .result_type
                    .is_some_and(|ty| storage_buffer_pointer_types.contains(&ty))
        });
    if !has_pointer_merge {
        return None;
    }
    let discovery = discover_cross_binding_psb(module)?;
    if require_cross_binding_phi && !discovery.has_cross_binding_phi {
        return None;
    }
    // Reserve the translator-owned table binding before mutating the module. Descriptor exhaustion
    // is a construction failure, not permission to leave a partially synthesized physical-pointer
    // graph behind.
    let occupied = crate::spirv_module::descriptor_bindings_in_set(module, layout.set);
    let address_table_binding = (layout.synthetic.start..layout.synthetic.end)
        .find(|binding| !occupied.contains(binding))?;
    let PsbDiscovery {
        type_defs,
        var_storage,
        var_pointee,
        value_type,
        cross_binding_merges,
        closure,
        closure_values,
        leaves,
        buffer_slot,
        const_ids,
        ..
    } = discovery;
    // Type-introspection closures re-declared over the discovered tables for the synthesis stage.
    let ptr_info = |ty: Word| -> Option<(StorageClass, Word)> {
        let inst = type_defs.get(&ty)?;
        if inst.class.opcode != Op::TypePointer {
            return None;
        }
        match (inst.operands.first()?, inst.operands.get(1)?) {
            (Operand::StorageClass(s), Operand::IdRef(pointee)) => Some((*s, *pointee)),
            _ => None,
        }
    };
    let is_buffer_var =
        |id: Word| -> bool { var_storage.get(&id) == Some(&StorageClass::StorageBuffer) };
    let element_strides = type_defs
        .keys()
        .filter_map(|ty| {
            element_allocation_stride(module, &type_defs, *ty).map(|stride| (*ty, stride))
        })
        .collect::<HashMap<_, _>>();
    // The natural alignment (component scalar size in bytes) of a scalar/vector pointee. This is the
    // value for the `Aligned` operand of a PhysicalStorageBuffer load/store: it is a power of two and
    // divides every element offset `j*element_stride` (the buffer base address is highly aligned). A
    // hardcoded `Aligned 1` is rejected (VUID-StandaloneSpirv-PhysicalStorageBuffer64-06314: the
    // value must be at least the largest scalar). A vector's element size (e.g. 12 for v3float) is
    // NOT necessarily a power of two, so the component size — not the allocation stride — is correct.
    let scalar_align = |ty: Word| -> Option<u32> {
        let inst = type_defs.get(&ty)?;
        let scalar = match inst.class.opcode {
            Op::TypeInt | Op::TypeFloat => ty,
            Op::TypeVector => match inst.operands.first()? {
                Operand::IdRef(elem) => *elem,
                _ => return None,
            },
            _ => return None,
        };
        match type_defs.get(&scalar)?.operands.first()? {
            Operand::LiteralBit32(bits) => Some(bits / 8),
            _ => None,
        }
    };

    // ---- All checks passed; synthesize the PSB lowering. ----
    let mut next_id = module.header.as_ref().map(|h| h.bound).unwrap_or(0);
    let mut fresh = || {
        let id = next_id;
        next_id += 1;
        id
    };

    // Locate / create the uint (32-bit) and ulong (64-bit) integer types and a uint_0 constant.
    fn find_int_ty(module: &Module, bits: u32) -> Option<Word> {
        module.types_global_values.iter().find_map(|i| {
            if i.class.opcode == Op::TypeInt
                && matches!(i.operands.first(), Some(Operand::LiteralBit32(b)) if *b == bits)
                && matches!(i.operands.get(1), Some(Operand::LiteralBit32(0)))
            {
                i.result_id
            } else {
                None
            }
        })
    }
    let uint_ty = match find_int_ty(module, 32) {
        Some(id) => id,
        None => {
            let id = fresh();
            module.types_global_values.push(Instruction::new(
                Op::TypeInt,
                None,
                Some(id),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ));
            id
        }
    };
    let ulong_ty = match find_int_ty(module, 64) {
        Some(id) => id,
        None => {
            let id = fresh();
            module.types_global_values.push(Instruction::new(
                Op::TypeInt,
                None,
                Some(id),
                vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
            ));
            id
        }
    };
    let uint_const = |module: &mut Module, fresh: &mut dyn FnMut() -> Word, v: u32| -> Word {
        if let Some(existing) = module.types_global_values.iter().find_map(|i| {
            (i.class.opcode == Op::Constant
                && i.result_type == Some(uint_ty)
                && matches!(i.operands.first(), Some(Operand::LiteralBit32(b)) if *b == v))
            .then_some(i.result_id)
            .flatten()
        }) {
            return existing;
        }
        let id = fresh();
        module.types_global_values.push(Instruction::new(
            Op::Constant,
            Some(uint_ty),
            Some(id),
            vec![Operand::LiteralBit32(v)],
        ));
        id
    };

    // Physical element-pointer type per pointee, ArrayStride-decorated.
    let mut psb_ptr: HashMap<Word, Word> = HashMap::new();
    for leaf in &leaves {
        if psb_ptr.contains_key(&leaf.pointee) {
            continue;
        }
        let id = fresh();
        module.types_global_values.push(Instruction::new(
            Op::TypePointer,
            None,
            Some(id),
            vec![
                Operand::StorageClass(StorageClass::PhysicalStorageBuffer),
                Operand::IdRef(leaf.pointee),
            ],
        ));
        // ArrayStride is required only when the physical pointer is used in `OpPtrAccessChain` (a
        // scalar/vector element pointer). A whole-buffer leaf's pointee is the buffer STRUCT, indexed
        // by `OpAccessChain` (which derives offsets from the struct's own member decorations), so it
        // gets no ArrayStride.
        if let Some(stride) = element_strides.get(&leaf.pointee).copied() {
            module.annotations.push(Instruction::new(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(id),
                    Operand::Decoration(Decoration::ArrayStride),
                    Operand::LiteralBit32(stride),
                ],
            ));
        }
        psb_ptr.insert(leaf.pointee, id);
    }
    // The closure's merges/ptr-access-chains keep their own (StorageBuffer) pointee; map each such
    // result's pointee to its physical pointer type (creating it if a merge introduces a new pointee).
    let mut retype: HashMap<Word, Word> = HashMap::new(); // value id -> new physical pointer type id
    for &v in &closure_values {
        // Whole-buffer-variable leaves are module-scope globals (not function SSA values); they are not
        // retyped in place — their physical base pointer is synthesized in the entry block below.
        if is_buffer_var(v) {
            continue;
        }
        let pointee = match value_type.get(&v).and_then(|t| ptr_info(*t)) {
            Some((_, p)) => p,
            None => return None,
        };
        let psb = match psb_ptr.get(&pointee) {
            Some(p) => *p,
            None => {
                let id = fresh();
                module.types_global_values.push(Instruction::new(
                    Op::TypePointer,
                    None,
                    Some(id),
                    vec![
                        Operand::StorageClass(StorageClass::PhysicalStorageBuffer),
                        Operand::IdRef(pointee),
                    ],
                ));
                // Scalar/vector pointee (element pointer used in OpPtrAccessChain) gets an ArrayStride;
                // an aggregate pointee (the merged whole-buffer struct, indexed by OpAccessChain) does
                // not — see the leaf pointer creation above.
                if let Some(stride) = element_strides.get(&pointee).copied() {
                    module.annotations.push(Instruction::new(
                        Op::Decorate,
                        None,
                        None,
                        vec![
                            Operand::IdRef(id),
                            Operand::Decoration(Decoration::ArrayStride),
                            Operand::LiteralBit32(stride),
                        ],
                    ));
                }
                psb_ptr.insert(pointee, id);
                id
            }
        };
        retype.insert(v, psb);
    }

    // LLVM opaque pointers permit a closure pointer's recovered element spelling to differ from the
    // type of a direct memory operation through it. Once the closure becomes a physical-address
    // graph, SPIR-V makes that relationship explicit: the pointer consumed by OpLoad/OpStore must
    // point to the accessed value type. Collect those typed views before rewriting instructions and
    // construct one PhysicalStorageBuffer pointer type per accessed value type. The instruction
    // rewrite below preserves the selected byte address and changes only its typed view.
    let all_value_types = module
        .all_inst_iter()
        .filter_map(|instruction| Some((instruction.result_id?, instruction.result_type?)))
        .collect::<HashMap<_, _>>();
    let all_value_defs = module
        .all_inst_iter()
        .filter_map(|instruction| Some((instruction.result_id?, instruction.clone())))
        .collect::<HashMap<_, _>>();
    let mut memory_view_pointees = HashSet::new();
    for function in &module.functions {
        for block in &function.blocks {
            for instruction in &block.instructions {
                let Some(Operand::IdRef(pointer)) = instruction.operands.first() else {
                    continue;
                };
                if !closure.contains(pointer) {
                    continue;
                }
                let accessed_type = match instruction.class.opcode {
                    Op::Load => instruction.result_type,
                    Op::Store => instruction
                        .operands
                        .get(1)
                        .and_then(|operand| match operand {
                            Operand::IdRef(object) => all_value_types.get(object).copied(),
                            _ => None,
                        }),
                    _ => None,
                };
                let pointer_pointee = value_type
                    .get(pointer)
                    .and_then(|ty| ptr_info(*ty))
                    .map(|(_, pointee)| pointee);
                if accessed_type != pointer_pointee {
                    memory_view_pointees.extend(accessed_type);
                }
            }
        }
    }
    for pointee in memory_view_pointees {
        if psb_ptr.contains_key(&pointee) {
            continue;
        }
        let id = fresh();
        module.types_global_values.push(Instruction::new(
            Op::TypePointer,
            None,
            Some(id),
            vec![
                Operand::StorageClass(StorageClass::PhysicalStorageBuffer),
                Operand::IdRef(pointee),
            ],
        ));
        psb_ptr.insert(pointee, id);
    }

    // Retype null/undef arms at the same construction boundary as their pointer merge. Physical
    // pointers cannot be OpConstantNull, so refine either source form to address zero and convert it
    // in the function entry. That value dominates every reachable phi parent/select use and keeps
    // null semantics while choosing one permitted value for LLVM `undef`.
    let mut nullish_retype_requests = Vec::new();
    for (function_index, function) in module.functions.iter().enumerate() {
        for block in &function.blocks {
            for instruction in &block.instructions {
                let Some(result) = instruction.result_id else {
                    continue;
                };
                let Some(&new_type) = retype.get(&result) else {
                    continue;
                };
                let pointer_arm_indices = match instruction.class.opcode {
                    Op::Select => (1..instruction.operands.len()).collect::<Vec<_>>(),
                    Op::Phi => (0..instruction.operands.len()).step_by(2).collect(),
                    Op::CopyObject => vec![0],
                    _ => continue,
                };
                for index in pointer_arm_indices {
                    let Some(Operand::IdRef(source)) = instruction.operands.get(index) else {
                        continue;
                    };
                    let Some(definition) = all_value_defs.get(source) else {
                        continue;
                    };
                    if matches!(definition.class.opcode, Op::Undef | Op::ConstantNull)
                        && definition.result_type != Some(new_type)
                    {
                        nullish_retype_requests.push((function_index, *source, new_type));
                    }
                }
            }
        }
    }
    nullish_retype_requests
        .sort_unstable_by_key(|(function, source, ty)| (*function, *source, *ty));
    nullish_retype_requests.dedup();
    let uint_zero = uint_const(module, &mut fresh, 0);
    let mut nullish_retypes = HashMap::new();
    let mut nullish_conversions = vec![Vec::new(); module.functions.len()];
    let mut nullish_zero_addresses = vec![None; module.functions.len()];
    for (function_index, source, new_type) in nullish_retype_requests {
        let zero_address = match nullish_zero_addresses[function_index] {
            Some(id) => id,
            None => {
                let id = fresh();
                nullish_conversions[function_index].push(Instruction::new(
                    Op::UConvert,
                    Some(ulong_ty),
                    Some(id),
                    vec![Operand::IdRef(uint_zero)],
                ));
                nullish_zero_addresses[function_index] = Some(id);
                id
            }
        };
        let replacement = fresh();
        nullish_conversions[function_index].push(Instruction::new(
            Op::ConvertUToPtr,
            Some(new_type),
            Some(replacement),
            vec![Operand::IdRef(zero_address)],
        ));
        nullish_retypes.insert((function_index, source, new_type), replacement);
    }
    for (function, conversions) in module.functions.iter_mut().zip(nullish_conversions) {
        if conversions.is_empty() {
            continue;
        }
        let entry = function.blocks.first_mut()?;
        let insertion = entry
            .instructions
            .iter()
            .position(|instruction| {
                !matches!(
                    instruction.class.opcode,
                    Op::Variable | Op::Line | Op::NoLine
                )
            })
            .unwrap_or(entry.instructions.len());
        entry.instructions.splice(insertion..insertion, conversions);
    }

    // Address-table descriptor: struct { runtimearray u64 } in the translator-owned binding range.
    let addr_rt = fresh();
    module.types_global_values.push(Instruction::new(
        Op::TypeRuntimeArray,
        None,
        Some(addr_rt),
        vec![Operand::IdRef(ulong_ty)],
    ));
    module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(addr_rt),
            Operand::Decoration(Decoration::ArrayStride),
            Operand::LiteralBit32(8),
        ],
    ));
    let addr_struct = fresh();
    module.types_global_values.push(Instruction::new(
        Op::TypeStruct,
        None,
        Some(addr_struct),
        vec![Operand::IdRef(addr_rt)],
    ));
    module.annotations.push(Instruction::new(
        Op::MemberDecorate,
        None,
        None,
        vec![
            Operand::IdRef(addr_struct),
            Operand::LiteralBit32(0),
            Operand::Decoration(Decoration::Offset),
            Operand::LiteralBit32(0),
        ],
    ));
    module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(addr_struct),
            Operand::Decoration(Decoration::Block),
        ],
    ));
    let ptr_sb_struct = fresh();
    module.types_global_values.push(Instruction::new(
        Op::TypePointer,
        None,
        Some(ptr_sb_struct),
        vec![
            Operand::StorageClass(StorageClass::StorageBuffer),
            Operand::IdRef(addr_struct),
        ],
    ));
    let ptr_sb_u64 = fresh();
    module.types_global_values.push(Instruction::new(
        Op::TypePointer,
        None,
        Some(ptr_sb_u64),
        vec![
            Operand::StorageClass(StorageClass::StorageBuffer),
            Operand::IdRef(ulong_ty),
        ],
    ));
    let addr_var = fresh();
    module.types_global_values.push(Instruction::new(
        Op::Variable,
        Some(ptr_sb_struct),
        Some(addr_var),
        vec![Operand::StorageClass(StorageClass::StorageBuffer)],
    ));
    // Allocate the translator-owned address table within the selected synthetic band.
    module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(addr_var),
            Operand::Decoration(Decoration::DescriptorSet),
            Operand::LiteralBit32(layout.set),
        ],
    ));
    module.annotations.push(Instruction::new(
        Op::Decorate,
        None,
        None,
        vec![
            Operand::IdRef(addr_var),
            Operand::Decoration(Decoration::Binding),
            Operand::LiteralBit32(address_table_binding),
        ],
    ));
    // SPIR-V >=1.4 lists every global variable in the entry-point interface.
    let uses_full_interface = module
        .header
        .as_ref()
        .map(|h| h.version() >= (1, 4))
        .unwrap_or(false);
    if uses_full_interface {
        for ep in &mut module.entry_points {
            ep.operands.push(Operand::IdRef(addr_var));
        }
    }

    // Precompute, per leaf id, the spliced address-load + convert instructions and the replacement id.
    // We rewrite each leaf access-chain instruction into a sequence; collect by (block,pos).
    let uint_zero = uint_const(module, &mut fresh, 0);
    let mut slots: Vec<u32> = buffer_slot.values().copied().collect();
    slots.sort_unstable();
    slots.dedup();
    let mut slot_consts: HashMap<u32, Word> = HashMap::new();
    for slot in slots {
        slot_consts
            .entry(slot)
            .or_insert_with(|| uint_const(module, &mut fresh, slot));
    }
    let leaf_map: HashMap<Word, &Leaf> = leaves.iter().map(|l| (l.id, l)).collect();

    // Whole-buffer leaves (a buffer variable selected/phi'd directly) are not function instructions, so
    // they cannot be rewritten in place like element-access-chain leaves. Synthesize each buffer's
    // physical base pointer (address-table load + ConvertUToPtr to the physical STRUCT pointer) once at
    // the top of the entry block — which dominates every merge — and record `var -> base id`. The merge
    // arms that referenced the variable are repointed to this base in the rewrite loop below.
    let mut var_base: HashMap<Word, Word> = HashMap::new();
    let whole_buffer_leaves: Vec<Word> = leaves
        .iter()
        .filter(|l| is_buffer_var(l.id))
        .map(|l| l.id)
        .collect();
    if !whole_buffer_leaves.is_empty() {
        let mut prelude: Vec<Instruction> = Vec::new();
        for &var in &whole_buffer_leaves {
            let slot = *buffer_slot.get(&var).unwrap();
            let slot_c = *slot_consts.get(&slot).unwrap();
            let pointee = *var_pointee.get(&var).unwrap();
            let psb_struct = *psb_ptr.get(&pointee).unwrap();
            let ac = fresh();
            prelude.push(Instruction::new(
                Op::AccessChain,
                Some(ptr_sb_u64),
                Some(ac),
                vec![
                    Operand::IdRef(addr_var),
                    Operand::IdRef(uint_zero),
                    Operand::IdRef(slot_c),
                ],
            ));
            let addr = fresh();
            prelude.push(Instruction::new(
                Op::Load,
                Some(ulong_ty),
                Some(addr),
                vec![Operand::IdRef(ac)],
            ));
            let base = fresh();
            prelude.push(Instruction::new(
                Op::ConvertUToPtr,
                Some(psb_struct),
                Some(base),
                vec![Operand::IdRef(addr)],
            ));
            var_base.insert(var, base);
        }
        // Insert into the entry block of the function that holds the merges, after its leading
        // function-local `OpVariable` declarations (which must stay first in the block).
        let merge_fn = module.functions.iter_mut().find(|f| {
            f.blocks.iter().any(|b| {
                b.instructions.iter().any(|i| {
                    i.result_id
                        .is_some_and(|r| cross_binding_merges.contains(&r))
                })
            })
        });
        if let Some(func) = merge_fn {
            if let Some(block) = func.blocks.first_mut() {
                let at = block
                    .instructions
                    .iter()
                    .position(|i| i.class.opcode != Op::Variable)
                    .unwrap_or(0);
                for (k, inst) in prelude.into_iter().enumerate() {
                    block.instructions.insert(at + k, inst);
                }
            }
        }
    }

    // Per closure-pointer `Aligned` operand value: the scalar alignment of its pointee. A load/store
    // through a closure pointer becomes a PhysicalStorageBuffer access and needs a valid `Aligned`
    // operand (see `scalar_align`). An untyped pointee is omitted rather than assigned a guessed
    // alignment.
    let closure_align: HashMap<Word, u32> = closure_values
        .iter()
        .filter_map(|p| {
            let pointee = value_type
                .get(p)
                .and_then(|t| ptr_info(*t))
                .map(|(_, pe)| pe)?;
            scalar_align(pointee).map(|a| (*p, a))
        })
        .collect();

    // Rewrite instructions.
    for (function_index, function) in module.functions.iter_mut().enumerate() {
        for block in &mut function.blocks {
            let mut new_insts: Vec<Instruction> = Vec::with_capacity(block.instructions.len());
            for inst in block.instructions.clone() {
                let rid = inst.result_id;
                // Leaf access chain -> address load + ConvertUToPtr.
                if let Some(r) = rid {
                    if let Some(leaf) = leaf_map.get(&r) {
                        let slot = *buffer_slot.get(&leaf.var).unwrap();
                        let slot_c = *slot_consts.get(&slot).unwrap();
                        let ac = fresh();
                        new_insts.push(Instruction::new(
                            Op::AccessChain,
                            Some(ptr_sb_u64),
                            Some(ac),
                            vec![
                                Operand::IdRef(addr_var),
                                Operand::IdRef(uint_zero),
                                Operand::IdRef(slot_c),
                            ],
                        ));
                        let addr = fresh();
                        new_insts.push(Instruction::new(
                            Op::Load,
                            Some(ulong_ty),
                            Some(addr),
                            vec![Operand::IdRef(ac)],
                        ));
                        let psb_pointee_ptr = *psb_ptr.get(&leaf.pointee).unwrap();
                        match leaf.element {
                            // Pure base-address leaf: ConvertUToPtr reuses the leaf's own result id so
                            // all consumers stay wired.
                            None => {
                                new_insts.push(Instruction::new(
                                    Op::ConvertUToPtr,
                                    Some(psb_pointee_ptr),
                                    Some(r),
                                    vec![Operand::IdRef(addr)],
                                ));
                            }
                            // Array-element leaf (`%buf %uint_0 %j`): ConvertUToPtr the base address,
                            // then OpPtrAccessChain by the SAME element index `j` (the physical pointee
                            // pointer carries the source allocation stride, so `base + j*stride`
                            // is byte-identical to the descriptor access). The PtrAccessChain reuses
                            // the leaf id so consumers stay wired.
                            Some(element) => {
                                let base = fresh();
                                new_insts.push(Instruction::new(
                                    Op::ConvertUToPtr,
                                    Some(psb_pointee_ptr),
                                    Some(base),
                                    vec![Operand::IdRef(addr)],
                                ));
                                new_insts.push(Instruction::new(
                                    Op::PtrAccessChain,
                                    Some(psb_pointee_ptr),
                                    Some(r),
                                    vec![Operand::IdRef(base), Operand::IdRef(element)],
                                ));
                            }
                        }
                        continue;
                    }
                }
                let mut inst = inst;
                // Repoint whole-buffer merge arms from the buffer variable to its synthesized physical
                // base pointer (only for closure merges/aliases; a non-closure select never references a
                // merged buffer var as a pointer arm).
                if !var_base.is_empty()
                    && matches!(inst.class.opcode, Op::Select | Op::Phi | Op::CopyObject)
                    && rid.is_some_and(|r| closure.contains(&r))
                {
                    for op in inst.operands.iter_mut() {
                        if let Operand::IdRef(id) = op {
                            if let Some(&base) = var_base.get(id) {
                                *op = Operand::IdRef(base);
                            }
                        }
                    }
                }
                // Flat-element post-merge chain: an OpAccessChain/OpInBoundsAccessChain that applies a
                // SINGLE DYNAMIC index straight to a merged whole-buffer pointer (`%merged %idx`, no
                // constant member-0 selector) — the shape a scalar element walk over a cross-binding
                // select cascade produces (e.g. b00a8a8d's 28-way output-buffer scatter). Indexing a
                // struct member dynamically is illegal, so the simple retype below cannot fix it.
                // Lower it like the array-element leaf: take the merged pointer's device address
                // (ConvertPtrToU), reinterpret as the element pointer (ConvertUToPtr to the
                // ArrayStride-decorated physical element type), and OpPtrAccessChain by the same index
                // (`base + idx*stride`, byte-identical). The valid `%merged %uint_0 %i` post-merge chains
                // the cleared whole-buffer cases carry have a CONSTANT first index, so they never match
                // and fall through to the plain retype.
                if let Some(r) = rid {
                    if matches!(inst.class.opcode, Op::AccessChain | Op::InBoundsAccessChain) {
                        let base = match inst.operands.first() {
                            Some(Operand::IdRef(b)) => Some(*b),
                            _ => None,
                        };
                        let single_dyn_index = (inst.operands.len() == 2)
                            .then(|| match inst.operands[1] {
                                Operand::IdRef(idx) if !const_ids.contains(&idx) => Some(idx),
                                _ => None,
                            })
                            .flatten();
                        if let (Some(base), Some(idx), Some(&elem_ptr)) =
                            (base, single_dyn_index, retype.get(&r))
                        {
                            if closure.contains(&base) {
                                let addr = fresh();
                                new_insts.push(Instruction::new(
                                    Op::ConvertPtrToU,
                                    Some(ulong_ty),
                                    Some(addr),
                                    vec![Operand::IdRef(base)],
                                ));
                                let elem_base = fresh();
                                new_insts.push(Instruction::new(
                                    Op::ConvertUToPtr,
                                    Some(elem_ptr),
                                    Some(elem_base),
                                    vec![Operand::IdRef(addr)],
                                ));
                                new_insts.push(Instruction::new(
                                    Op::PtrAccessChain,
                                    Some(elem_ptr),
                                    Some(r),
                                    vec![Operand::IdRef(elem_base), Operand::IdRef(idx)],
                                ));
                                continue;
                            }
                        }
                    }
                }
                // Retype closure merges / access chains. A post-merge access chain (an
                // OpAccessChain/OpInBoundsAccessChain indexing INTO a merged whole-buffer pointer)
                // becomes a physical access chain — same indices, physical result pointer.
                if let Some(r) = rid {
                    if let Some(new_ty) = retype.get(&r) {
                        if matches!(
                            inst.class.opcode,
                            Op::Select
                                | Op::Phi
                                | Op::PtrAccessChain
                                | Op::AccessChain
                                | Op::InBoundsAccessChain
                                | Op::CopyObject
                        ) {
                            inst.result_type = Some(*new_ty);
                            let pointer_arm_indices = match inst.class.opcode {
                                Op::Select => (1..inst.operands.len()).collect::<Vec<_>>(),
                                Op::Phi => (0..inst.operands.len()).step_by(2).collect(),
                                Op::CopyObject => vec![0],
                                _ => Vec::new(),
                            };
                            for index in pointer_arm_indices {
                                let Some(Operand::IdRef(source)) = inst.operands.get_mut(index)
                                else {
                                    continue;
                                };
                                if let Some(&replacement) =
                                    nullish_retypes.get(&(function_index, *source, *new_ty))
                                {
                                    *source = replacement;
                                }
                            }
                        }
                    }
                }
                // A direct memory access can carry an opaque-pointer spelling different from its
                // accessed value type. Materialize the exact same address with the accessed pointee
                // type before the memory operation; this makes the physical pointer contract explicit
                // without changing the selected buffer or byte offset.
                let closure_memory_pointer = match inst.class.opcode {
                    Op::Load | Op::Store => {
                        inst.operands.first().and_then(|operand| match operand {
                            Operand::IdRef(pointer) if closure.contains(pointer) => Some(*pointer),
                            _ => None,
                        })
                    }
                    _ => None,
                };
                let accessed_type = match inst.class.opcode {
                    Op::Load => inst.result_type,
                    Op::Store => inst.operands.get(1).and_then(|operand| match operand {
                        Operand::IdRef(object) => all_value_types.get(object).copied(),
                        _ => None,
                    }),
                    _ => None,
                };
                if let (Some(pointer), Some(accessed_type)) =
                    (closure_memory_pointer, accessed_type)
                {
                    let pointer_pointee = value_type
                        .get(&pointer)
                        .and_then(|ty| ptr_info(*ty))
                        .map(|(_, pointee)| pointee);
                    if closure.contains(&pointer) && pointer_pointee != Some(accessed_type) {
                        let address = fresh();
                        new_insts.push(Instruction::new(
                            Op::ConvertPtrToU,
                            Some(ulong_ty),
                            Some(address),
                            vec![Operand::IdRef(pointer)],
                        ));
                        let typed_pointer = fresh();
                        new_insts.push(Instruction::new(
                            Op::ConvertUToPtr,
                            Some(*psb_ptr.get(&accessed_type).unwrap()),
                            Some(typed_pointer),
                            vec![Operand::IdRef(address)],
                        ));
                        inst.operands[0] = Operand::IdRef(typed_pointer);
                    }
                }
                // Aligned operand on loads/stores through a closure pointer.
                match inst.class.opcode {
                    Op::Load => {
                        if let Some(pointer) = closure_memory_pointer {
                            let align = inst
                                .result_type
                                .and_then(scalar_align)
                                .or_else(|| closure_align.get(&pointer).copied())
                                .unwrap_or(4);
                            ensure_physical_memory_alignment(&mut inst, align);
                        }
                    }
                    Op::Store => {
                        if let Some(pointer) = closure_memory_pointer {
                            let align = accessed_type
                                .and_then(scalar_align)
                                .or_else(|| closure_align.get(&pointer).copied())
                                .unwrap_or(4);
                            ensure_physical_memory_alignment(&mut inst, align);
                        }
                    }
                    _ => {}
                }
                new_insts.push(inst);
            }
            block.instructions = new_insts;
        }
    }

    // Memory model -> PhysicalStorageBuffer64; capabilities + extension.
    if let Some(mm) = module.memory_model.as_mut() {
        if let Some(op) = mm.operands.first_mut() {
            *op = Operand::AddressingModel(spirv::AddressingModel::PhysicalStorageBuffer64);
        }
    }
    let has_cap = |module: &Module, c: Capability| {
        module
            .capabilities
            .iter()
            .any(|i| matches!(i.operands.first(), Some(Operand::Capability(x)) if *x == c))
    };
    for cap in [
        Capability::PhysicalStorageBufferAddresses,
        Capability::Int64,
    ] {
        if !has_cap(module, cap) {
            module.capabilities.push(Instruction::new(
                Op::Capability,
                None,
                None,
                vec![Operand::Capability(cap)],
            ));
        }
    }
    let has_ext = module.extensions.iter().any(|i| {
        matches!(i.operands.first(), Some(Operand::LiteralString(s)) if s == "SPV_KHR_physical_storage_buffer")
    });
    if !has_ext {
        module.extensions.push(Instruction::new(
            Op::Extension,
            None,
            None,
            vec![Operand::LiteralString(
                "SPV_KHR_physical_storage_buffer".to_string(),
            )],
        ));
    }
    let _ = MemoryModel::GLSL450;

    if let Some(header) = module.header.as_mut() {
        header.bound = next_id;
    }
    Some(addr_var)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::{Block, Function, ModuleHeader};

    fn inst(op: Op, ty: Option<Word>, res: Option<Word>, ops: Vec<Operand>) -> Instruction {
        Instruction::new(op, ty, res, ops)
    }

    #[test]
    fn physical_vector_stride_prefers_explicit_layout_and_rejects_conflicts() {
        let mut module = Module::new();
        module.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
            ),
            inst(
                Op::TypeVector,
                None,
                Some(2),
                vec![Operand::IdRef(1), Operand::LiteralBit32(3)],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(3),
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(2),
                ],
            ),
            inst(Op::TypeRuntimeArray, None, Some(4), vec![Operand::IdRef(2)]),
        ];
        let type_defs = module
            .types_global_values
            .iter()
            .filter_map(|instruction| {
                instruction
                    .result_id
                    .map(|result_id| (result_id, instruction.clone()))
            })
            .collect::<HashMap<_, _>>();

        assert_eq!(element_allocation_stride(&module, &type_defs, 2), Some(4));

        module.annotations.push(inst(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(3),
                Operand::Decoration(Decoration::ArrayStride),
                Operand::LiteralBit32(8),
            ],
        ));
        assert_eq!(element_allocation_stride(&module, &type_defs, 2), Some(8));

        module.annotations.push(inst(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(4),
                Operand::Decoration(Decoration::ArrayStride),
                Operand::LiteralBit32(4),
            ],
        ));
        assert_eq!(element_allocation_stride(&module, &type_defs, 2), None);
    }

    #[test]
    fn rewrite_cross_binding_vector_elements_preserves_source_array_stride() {
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(40));
        module.memory_model = Some(inst(
            Op::MemoryModel,
            None,
            None,
            vec![
                Operand::AddressingModel(spirv::AddressingModel::Logical),
                Operand::MemoryModel(MemoryModel::GLSL450),
            ],
        ));
        module.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(8), Operand::LiteralBit32(0)],
            ),
            inst(
                Op::TypeVector,
                None,
                Some(2),
                vec![Operand::IdRef(1), Operand::LiteralBit32(3)],
            ),
            inst(Op::TypeRuntimeArray, None, Some(3), vec![Operand::IdRef(2)]),
            inst(Op::TypeStruct, None, Some(4), vec![Operand::IdRef(3)]),
            inst(
                Op::TypePointer,
                None,
                Some(5),
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(4),
                ],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(6),
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(2),
                ],
            ),
            inst(Op::TypeBool, None, Some(7), vec![]),
            inst(
                Op::TypeInt,
                None,
                Some(8),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            inst(
                Op::Constant,
                Some(8),
                Some(10),
                vec![Operand::LiteralBit32(0)],
            ),
            inst(
                Op::Constant,
                Some(8),
                Some(11),
                vec![Operand::LiteralBit32(1)],
            ),
            inst(Op::ConstantTrue, Some(7), Some(12), vec![]),
            inst(
                Op::Variable,
                Some(5),
                Some(20),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
            inst(
                Op::Variable,
                Some(5),
                Some(21),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
        ];
        module.annotations = vec![
            inst(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(3),
                    Operand::Decoration(Decoration::ArrayStride),
                    Operand::LiteralBit32(8),
                ],
            ),
            inst(
                Op::Decorate,
                None,
                None,
                vec![Operand::IdRef(4), Operand::Decoration(Decoration::Block)],
            ),
            inst(
                Op::MemberDecorate,
                None,
                None,
                vec![
                    Operand::IdRef(4),
                    Operand::LiteralBit32(0),
                    Operand::Decoration(Decoration::Offset),
                    Operand::LiteralBit32(0),
                ],
            ),
            inst(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(20),
                    Operand::Decoration(Decoration::Binding),
                    Operand::LiteralBit32(0),
                ],
            ),
            inst(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(21),
                    Operand::Decoration(Decoration::Binding),
                    Operand::LiteralBit32(1),
                ],
            ),
        ];
        let mut block = Block::new();
        block.label = Some(inst(Op::Label, None, Some(30), vec![]));
        block.instructions = vec![
            inst(
                Op::AccessChain,
                Some(6),
                Some(31),
                vec![Operand::IdRef(20), Operand::IdRef(10), Operand::IdRef(11)],
            ),
            inst(
                Op::AccessChain,
                Some(6),
                Some(32),
                vec![Operand::IdRef(21), Operand::IdRef(10), Operand::IdRef(11)],
            ),
            inst(
                Op::Select,
                Some(6),
                Some(33),
                vec![Operand::IdRef(12), Operand::IdRef(31), Operand::IdRef(32)],
            ),
            inst(Op::Load, Some(2), Some(34), vec![Operand::IdRef(33)]),
            inst(Op::Return, None, None, vec![]),
        ];
        let mut function = Function::new();
        function.blocks = vec![block];
        module.functions = vec![function];

        assert!(rewrite_cross_binding_pointer_merges(&mut module));
        let physical_vector_pointer = module
            .types_global_values
            .iter()
            .find_map(|instruction| {
                (instruction.class.opcode == Op::TypePointer
                    && instruction.operands
                        == [
                            Operand::StorageClass(StorageClass::PhysicalStorageBuffer),
                            Operand::IdRef(2),
                        ])
                .then_some(instruction.result_id)
                .flatten()
            })
            .expect("physical uchar3 pointer");
        assert!(module.annotations.iter().any(|annotation| {
            annotation.class.opcode == Op::Decorate
                && annotation.operands
                    == [
                        Operand::IdRef(physical_vector_pointer),
                        Operand::Decoration(Decoration::ArrayStride),
                        Operand::LiteralBit32(8),
                    ]
        }));
    }

    // A WHOLE-BUFFER cross-binding select — `OpSelect %ptr_struct %cond %bufA %bufB` over two distinct
    // descriptor-bound buffer VARIABLES directly, with the indexing access chain applied to the merged
    // pointer downstream — must lower to PhysicalStorageBuffer64: each buffer's device address is loaded
    // from the synthesized address table and `OpConvertUToPtr`-ed to a physical struct pointer, the
    // select chooses between those base addresses, and the post-merge access chain becomes a physical
    // access chain. This is the raw-emission shape of the `binFragmentsKernel` BVH cases; without the
    // whole-buffer leaf path the select's variable arms are filtered out and no merge is detected.
    #[test]
    fn rewrite_whole_buffer_cross_binding_select_lowers_to_physical() {
        // ids: uint=1 rtarr=2 struct=3 ptrSbStruct=4 ptrSbUint=5 bool=6 float=7 vec4=8 |
        //      uint_0=10 true=11 undefPtr=12
        //      bufA=20 bufB=21 | entry=30 select=31 chain=32 load=33
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(40));
        m.memory_model = Some(inst(
            Op::MemoryModel,
            None,
            None,
            vec![
                Operand::AddressingModel(spirv::AddressingModel::Logical),
                Operand::MemoryModel(MemoryModel::GLSL450),
            ],
        ));
        m.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            inst(Op::TypeRuntimeArray, None, Some(2), vec![Operand::IdRef(1)]),
            inst(Op::TypeStruct, None, Some(3), vec![Operand::IdRef(2)]),
            inst(
                Op::TypePointer,
                None,
                Some(4),
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(3),
                ],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(5),
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(1),
                ],
            ),
            inst(Op::TypeBool, None, Some(6), vec![]),
            inst(
                Op::TypeFloat,
                None,
                Some(7),
                vec![Operand::LiteralBit32(32)],
            ),
            inst(
                Op::TypeVector,
                None,
                Some(8),
                vec![Operand::IdRef(7), Operand::LiteralBit32(4)],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(10),
                vec![Operand::LiteralBit32(0)],
            ),
            inst(Op::ConstantTrue, Some(6), Some(11), vec![]),
            inst(Op::Undef, Some(4), Some(12), vec![]),
            inst(
                Op::Variable,
                Some(4),
                Some(20),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
            inst(
                Op::Variable,
                Some(4),
                Some(21),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
        ];
        m.annotations = vec![
            inst(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(2),
                    Operand::Decoration(Decoration::ArrayStride),
                    Operand::LiteralBit32(4),
                ],
            ),
            inst(
                Op::Decorate,
                None,
                None,
                vec![Operand::IdRef(3), Operand::Decoration(Decoration::Block)],
            ),
            inst(
                Op::MemberDecorate,
                None,
                None,
                vec![
                    Operand::IdRef(3),
                    Operand::LiteralBit32(0),
                    Operand::Decoration(Decoration::Offset),
                    Operand::LiteralBit32(0),
                ],
            ),
            inst(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(20),
                    Operand::Decoration(Decoration::Binding),
                    Operand::LiteralBit32(0),
                ],
            ),
            inst(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(21),
                    Operand::Decoration(Decoration::Binding),
                    Operand::LiteralBit32(1),
                ],
            ),
        ];
        let mut block = Block::new();
        block.label = Some(inst(Op::Label, None, Some(30), vec![]));
        block.instructions = vec![
            inst(
                Op::Select,
                Some(4),
                Some(31),
                vec![Operand::IdRef(11), Operand::IdRef(20), Operand::IdRef(21)],
            ),
            inst(
                Op::InBoundsAccessChain,
                Some(5),
                Some(32),
                vec![Operand::IdRef(31), Operand::IdRef(10), Operand::IdRef(10)],
            ),
            // The recovered pointer spelling is uint*, while the opaque-pointer memory operation
            // accesses a float4. Physical construction must preserve the address and create the
            // exact float4 pointer view consumed by this load.
            inst(Op::Load, Some(8), Some(33), vec![Operand::IdRef(32)]),
            inst(Op::Return, None, None, vec![]),
        ];
        let mut func = Function::new();
        func.blocks = vec![block];
        m.functions = vec![func];

        let mut select_only = m.clone();
        assert!(!has_cross_binding_pointer_phi(&select_only));
        let mut pointer_invalid = select_only.clone();
        pointer_invalid.functions[0].blocks[0].instructions.insert(
            1,
            inst(Op::Bitcast, Some(4), Some(39), vec![Operand::IdRef(20)]),
        );
        let pointer_invalid_before = pointer_invalid.assemble();
        assert_eq!(
            super::super::rewrites::construct_interface_cross_binding_pointer_merges_module(
                &mut pointer_invalid,
                crate::reflect::DescriptorLayout::default(),
            ),
            None
        );
        assert_eq!(pointer_invalid.assemble(), pointer_invalid_before);
        let mut memory_invalid = select_only.clone();
        memory_invalid.functions[0].blocks[0].instructions.insert(
            1,
            inst(Op::Load, Some(1), Some(38), vec![Operand::IdRef(7)]),
        );
        let memory_invalid_before = memory_invalid.assemble();
        assert_eq!(
            super::super::rewrites::construct_interface_cross_binding_pointer_merges_module(
                &mut memory_invalid,
                crate::reflect::DescriptorLayout::default(),
            ),
            None
        );
        assert_eq!(memory_invalid.assemble(), memory_invalid_before);
        let mut phi = m.clone();
        phi.functions[0].blocks[0].instructions[0] = inst(
            Op::Phi,
            Some(4),
            Some(31),
            vec![
                Operand::IdRef(20),
                Operand::IdRef(30),
                Operand::IdRef(21),
                Operand::IdRef(30),
                Operand::IdRef(12),
                Operand::IdRef(30),
            ],
        );
        assert!(has_cross_binding_pointer_phi(&phi));
        let mut nested_phi = m.clone();
        nested_phi.functions[0].blocks[0].instructions[0] = inst(
            Op::Phi,
            Some(4),
            Some(31),
            vec![
                Operand::IdRef(20),
                Operand::IdRef(30),
                Operand::IdRef(34),
                Operand::IdRef(30),
            ],
        );
        nested_phi.functions[0].blocks[0].instructions.insert(
            0,
            inst(
                Op::Phi,
                Some(4),
                Some(34),
                vec![
                    Operand::IdRef(21),
                    Operand::IdRef(30),
                    Operand::IdRef(12),
                    Operand::IdRef(30),
                ],
            ),
        );
        assert!(has_cross_binding_pointer_phi(&nested_phi));
        assert!(rewrite_cross_binding_pointer_phis(&mut nested_phi));
        assert!(!rewrite_cross_binding_pointer_phis(&mut select_only));
        let before_exhaustion = phi.assemble();
        let exhausted_layout = crate::reflect::DescriptorLayout {
            synthetic: crate::reflect::DescriptorBindingRange {
                start: crate::reflect::SYNTHETIC_BINDING_BASE,
                end: crate::reflect::SYNTHETIC_BINDING_BASE,
            },
            ..crate::reflect::DescriptorLayout::default()
        };
        assert_eq!(
            construct_cross_binding_pointer_phis_with_layout(&mut phi, exhausted_layout),
            None
        );
        assert_eq!(phi.assemble(), before_exhaustion);
        assert!(rewrite_cross_binding_pointer_phis(&mut phi));
        assert!(rewrite_cross_binding_pointer_merges(&mut m));

        // Memory model is now PhysicalStorageBuffer64.
        assert!(matches!(
            m.memory_model.as_ref().unwrap().operands.first(),
            Some(Operand::AddressingModel(
                spirv::AddressingModel::PhysicalStorageBuffer64
            ))
        ));
        // The select no longer references the buffer variables directly — its arms are the synthesized
        // physical base pointers (OpConvertUToPtr results).
        let select = m.functions[0].blocks[0]
            .instructions
            .iter()
            .find(|i| i.result_id == Some(31))
            .unwrap();
        assert!(!select
            .operands
            .iter()
            .any(|o| matches!(o, Operand::IdRef(20) | Operand::IdRef(21))));
        // Two base pointers (one per merged buffer) and the load's typed float4 view were
        // synthesized.
        let n_convert = m.functions[0].blocks[0]
            .instructions
            .iter()
            .filter(|i| i.class.opcode == Op::ConvertUToPtr)
            .count();
        assert_eq!(n_convert, 3);
        // The post-merge access chain's result type is now a PhysicalStorageBuffer pointer.
        let chain = m.functions[0].blocks[0]
            .instructions
            .iter()
            .find(|i| i.result_id == Some(32))
            .unwrap();
        let chain_ty = chain.result_type.unwrap();
        let sc = m
            .types_global_values
            .iter()
            .find(|i| i.result_id == Some(chain_ty))
            .and_then(|i| i.operands.first());
        assert!(matches!(
            sc,
            Some(Operand::StorageClass(StorageClass::PhysicalStorageBuffer))
        ));
        let load = m.functions[0].blocks[0]
            .instructions
            .iter()
            .find(|i| i.result_id == Some(33))
            .unwrap();
        let Operand::IdRef(load_pointer) = load.operands[0] else {
            panic!("load pointer is not an id");
        };
        let load_pointer_type = m.functions[0].blocks[0]
            .instructions
            .iter()
            .find(|i| i.result_id == Some(load_pointer))
            .and_then(|i| i.result_type)
            .unwrap();
        let load_pointer_definition = m
            .types_global_values
            .iter()
            .find(|i| i.result_id == Some(load_pointer_type))
            .unwrap();
        assert_eq!(
            load_pointer_definition.operands,
            [
                Operand::StorageClass(StorageClass::PhysicalStorageBuffer),
                Operand::IdRef(8)
            ]
        );
        let rewritten_phi = phi.functions[0].blocks[0]
            .instructions
            .iter()
            .find(|i| i.result_id == Some(31))
            .unwrap();
        let Operand::IdRef(retyped_undef) = rewritten_phi.operands[4] else {
            panic!("phi undef arm is not an id");
        };
        assert_ne!(retyped_undef, 12);
        let retyped_undef_definition = phi.functions[0].blocks[0]
            .instructions
            .iter()
            .find(|i| i.result_id == Some(retyped_undef))
            .unwrap();
        assert_eq!(retyped_undef_definition.class.opcode, Op::ConvertUToPtr);
        assert_eq!(
            retyped_undef_definition.result_type,
            rewritten_phi.result_type
        );
        let Operand::IdRef(zero_address) = retyped_undef_definition.operands[0] else {
            panic!("physical null address is not an id");
        };
        let zero_address_definition = phi.functions[0].blocks[0]
            .instructions
            .iter()
            .find(|instruction| instruction.result_id == Some(zero_address))
            .expect("zero address conversion");
        assert_eq!(zero_address_definition.class.opcode, Op::UConvert);
    }

    // The same whole-buffer cross-binding select, but the post-merge access chain feeds an ATOMIC op
    // (OpAtomicIAdd) instead of a plain load. The pass must still lower it — under PhysicalStorageBuffer64
    // the atomic stays an atomic over the merged buffer's real device memory (byte-correct). Without the
    // atomic-consumer whitelist the self-containment check bails (the closure pointer is "used by a
    // non-load/store"). This is the `01/b511a833` shape (a select-among-N-buffers cascade + OpAtomicAnd).
    #[test]
    fn rewrite_whole_buffer_cross_binding_select_allows_atomic_consumer() {
        // ids: uint=1 rtarr=2 struct=3 ptrSbStruct=4 ptrSbUint=5 bool=6 | uint_0=10 true=11 scope=12 val=13
        //      bufA=20 bufB=21 | entry=30 select=31 chain=32 atomic=33
        let mut m = Module::new();
        m.header = Some(ModuleHeader::new(40));
        m.memory_model = Some(inst(
            Op::MemoryModel,
            None,
            None,
            vec![
                Operand::AddressingModel(spirv::AddressingModel::Logical),
                Operand::MemoryModel(MemoryModel::GLSL450),
            ],
        ));
        m.types_global_values = vec![
            inst(
                Op::TypeInt,
                None,
                Some(1),
                vec![Operand::LiteralBit32(32), Operand::LiteralBit32(0)],
            ),
            inst(Op::TypeRuntimeArray, None, Some(2), vec![Operand::IdRef(1)]),
            inst(Op::TypeStruct, None, Some(3), vec![Operand::IdRef(2)]),
            inst(
                Op::TypePointer,
                None,
                Some(4),
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(3),
                ],
            ),
            inst(
                Op::TypePointer,
                None,
                Some(5),
                vec![
                    Operand::StorageClass(StorageClass::StorageBuffer),
                    Operand::IdRef(1),
                ],
            ),
            inst(Op::TypeBool, None, Some(6), vec![]),
            inst(
                Op::Constant,
                Some(1),
                Some(10),
                vec![Operand::LiteralBit32(0)],
            ),
            inst(Op::ConstantTrue, Some(6), Some(11), vec![]),
            inst(
                Op::Constant,
                Some(1),
                Some(12),
                vec![Operand::LiteralBit32(1)],
            ),
            inst(
                Op::Constant,
                Some(1),
                Some(13),
                vec![Operand::LiteralBit32(7)],
            ),
            inst(
                Op::Variable,
                Some(4),
                Some(20),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
            inst(
                Op::Variable,
                Some(4),
                Some(21),
                vec![Operand::StorageClass(StorageClass::StorageBuffer)],
            ),
        ];
        m.annotations = vec![
            inst(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(2),
                    Operand::Decoration(Decoration::ArrayStride),
                    Operand::LiteralBit32(4),
                ],
            ),
            inst(
                Op::Decorate,
                None,
                None,
                vec![Operand::IdRef(3), Operand::Decoration(Decoration::Block)],
            ),
            inst(
                Op::MemberDecorate,
                None,
                None,
                vec![
                    Operand::IdRef(3),
                    Operand::LiteralBit32(0),
                    Operand::Decoration(Decoration::Offset),
                    Operand::LiteralBit32(0),
                ],
            ),
            inst(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(20),
                    Operand::Decoration(Decoration::Binding),
                    Operand::LiteralBit32(0),
                ],
            ),
            inst(
                Op::Decorate,
                None,
                None,
                vec![
                    Operand::IdRef(21),
                    Operand::Decoration(Decoration::Binding),
                    Operand::LiteralBit32(1),
                ],
            ),
        ];
        let mut block = Block::new();
        block.label = Some(inst(Op::Label, None, Some(30), vec![]));
        block.instructions = vec![
            inst(
                Op::Select,
                Some(4),
                Some(31),
                vec![Operand::IdRef(11), Operand::IdRef(20), Operand::IdRef(21)],
            ),
            inst(
                Op::InBoundsAccessChain,
                Some(5),
                Some(32),
                vec![Operand::IdRef(31), Operand::IdRef(10), Operand::IdRef(10)],
            ),
            // OpAtomicIAdd %uint %ptr %scope %semantics %value — pointer is the post-merge access chain.
            inst(
                Op::AtomicIAdd,
                Some(1),
                Some(33),
                vec![
                    Operand::IdRef(32),
                    Operand::IdRef(12),
                    Operand::IdRef(10),
                    Operand::IdRef(13),
                ],
            ),
            inst(Op::Return, None, None, vec![]),
        ];
        let mut func = Function::new();
        func.blocks = vec![block];
        m.functions = vec![func];

        assert!(rewrite_cross_binding_pointer_merges(&mut m));
        // Lowered to PhysicalStorageBuffer64 with the select's variable arms repointed to base pointers.
        assert!(matches!(
            m.memory_model.as_ref().unwrap().operands.first(),
            Some(Operand::AddressingModel(
                spirv::AddressingModel::PhysicalStorageBuffer64
            ))
        ));
        let select = m.functions[0].blocks[0]
            .instructions
            .iter()
            .find(|i| i.result_id == Some(31))
            .unwrap();
        assert!(!select
            .operands
            .iter()
            .any(|o| matches!(o, Operand::IdRef(20) | Operand::IdRef(21))));
        // The atomic op is preserved (not de-atomicized) and still points at the post-merge chain.
        let atomic = m.functions[0].blocks[0]
            .instructions
            .iter()
            .find(|i| i.result_id == Some(33))
            .unwrap();
        assert_eq!(atomic.class.opcode, Op::AtomicIAdd);
        assert!(matches!(atomic.operands.first(), Some(Operand::IdRef(32))));
    }
}
