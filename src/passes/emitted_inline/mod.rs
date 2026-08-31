// Producer-side emitted-graph function inliner. The LLVM Vulkan backend does NOT inline helper
// functions, so a real shader module is a graph of functions: the entry calls helpers
// taking texture/sampler/buffer POINTERS + calling air.* intrinsics. Logical-addressing SPIR-V cannot
// pass most pointers as function arguments, and our interface/lowering passes assume a single
// function — so we inline every non-air.* call into the entry, to a fixpoint. air.* calls are left
// intact for the lowering pass; air.* declarations are bodyless and never inlined.
//
// Strategy per call site (callee may be multi-block):
//   * clone the callee, allocating a FRESH id for every result id it defines (params + block labels +
//     instruction results), and remap operand references through that map;
//   * substitute each parameter id with the corresponding call argument id;
//   * if the callee is single-block (the common case for these helpers after the backend's own
//     structurization): splice its body instructions (minus the terminator) directly in place of the
//     call, and replace the call's result id with the callee's returned value;
//   * if multi-block: split the caller block at the call, wire the callee's blocks in between, and
//     route the callee's OpReturnValue through an OpPhi-free single return by rewriting returns to a
//     branch to the continuation with the value captured.
// We only implement the single-block fast path + a conservative multi-block path; if a callee shape is
// unsupported we return an error (non-PASS exit) so callers can skip or flag that shader.

use super::spirv_cfg::{block_successors, block_successors_by_label};
use super::*;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct InlineStats {
    pub(super) splices: usize,
    pub(super) helper_instances: usize,
}

#[cfg(test)]
pub(super) fn inline_helpers(ctx: &mut Ctx, entry_idx: usize) -> Result<InlineStats, String> {
    let entry_id = ctx.module.functions[entry_idx]
        .def
        .as_ref()
        .and_then(|definition| definition.result_id);
    let selected = ctx
        .module
        .functions
        .iter()
        .filter(|function| !function.blocks.is_empty())
        .filter_map(|function| {
            let id = function
                .def
                .as_ref()
                .and_then(|definition| definition.result_id)?;
            (Some(id) != entry_id).then_some(id)
        })
        .collect::<HashSet<_>>();
    let stats = inline_selected_helpers(ctx, entry_idx, &selected)?;
    complete_inlined_access_chain_descent(ctx, entry_idx);
    compose_chained_access_chains(ctx, entry_idx);
    prune_unreferenced_functions(ctx, entry_idx);
    Ok(stats)
}

/// Finish the self-contained entry's member-access chains after every helper splice has completed.
/// A substituted pointer may add address-equivalent array/vector wrapper layers around the helper's
/// declared pointee. Completing those zero-offset descents once at the closure boundary prevents a
/// cloned helper body from being completed again by its caller.
pub(super) fn complete_inlined_access_chain_descent(ctx: &mut Ctx, entry_idx: usize) {
    let defs = type_defs_with_new_globals(ctx);
    let result_types = collect_result_types(ctx);
    for block_idx in 0..ctx.module.functions[entry_idx].blocks.len() {
        for instruction_idx in 0..ctx.module.functions[entry_idx].blocks[block_idx]
            .instructions
            .len()
        {
            let instruction = ctx.module.functions[entry_idx].blocks[block_idx].instructions
                [instruction_idx]
                .clone();
            if !is_member_access_chain(instruction.class.opcode) || instruction.operands.len() < 2 {
                continue;
            }
            let Some(result_pointee) = instruction
                .result_type
                .and_then(|result_type| ptr_pointee(&defs, result_type))
            else {
                continue;
            };
            let Some(Operand::IdRef(base)) = instruction.operands.first() else {
                continue;
            };
            let Some(base_pointee) = result_types
                .get(base)
                .and_then(|base_type| ptr_pointee(&defs, *base_type))
            else {
                continue;
            };
            let Some(mut reached) =
                walk_access_chain_type(&defs, base_pointee, &instruction.operands[1..])
            else {
                continue;
            };
            let mut zero_tail = 0usize;
            while reached != result_pointee {
                let Some(definition) = defs.get(&reached) else {
                    zero_tail = 0;
                    break;
                };
                reached = match definition.class.opcode {
                    Op::TypeArray | Op::TypeRuntimeArray | Op::TypeVector | Op::TypeMatrix => {
                        let Some(Operand::IdRef(element)) = definition.operands.first() else {
                            zero_tail = 0;
                            break;
                        };
                        *element
                    }
                    _ => {
                        zero_tail = 0;
                        break;
                    }
                };
                zero_tail += 1;
            }
            if zero_tail == 0 || reached != result_pointee {
                continue;
            }
            let Some(index_type) = instruction.operands[1..].iter().rev().find_map(|operand| {
                let Operand::IdRef(index) = operand else {
                    return None;
                };
                result_types.get(index).copied()
            }) else {
                continue;
            };
            let zero = get_or_create_int_const(ctx, index_type, 0);
            ctx.module.functions[entry_idx].blocks[block_idx].instructions[instruction_idx]
                .operands
                .extend(std::iter::repeat_n(Operand::IdRef(zero), zero_tail));
        }
    }
}

/// Inline only calls whose callee id is in `selected`.
///
/// This is the producer-side seam before serialization. It deliberately leaves dead helper
/// functions and chained-access composition for their downstream cleanup phase.
pub(super) fn inline_selected_helpers(
    ctx: &mut Ctx,
    entry_idx: usize,
    selected: &HashSet<Word>,
) -> Result<InlineStats, String> {
    let air_ids: HashSet<Word> = air_names(&ctx.module).keys().copied().collect();
    // Map function-def id -> function index, for bodied non-air.* functions (inline candidates).
    let mut stats = InlineStats::default();
    let mut processed_helpers = HashSet::new();
    let mut pointer_types = collect_pointer_types(ctx);
    let mut guard = 0;
    loop {
        guard += 1;
        if guard > 2000 {
            return Err("inliner exceeded iteration budget".into());
        }
        // Find a call in the entry to a bodied, non-air.* function.
        let entry_id = ctx.module.functions[entry_idx]
            .def
            .as_ref()
            .and_then(|d| d.result_id);
        let mut target: Option<(usize, usize, Word)> = None; // (block, inst, callee_id)
        'scan: for (bi, blk) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
            for (ii, inst) in blk.instructions.iter().enumerate() {
                if inst.class.opcode == Op::FunctionCall {
                    if let Some(Operand::IdRef(callee)) = inst.operands.first() {
                        if !selected.contains(callee) {
                            continue;
                        }
                        if air_ids.contains(callee) {
                            continue;
                        }
                        if Some(*callee) == entry_id {
                            continue; // never inline self-recursion
                        }
                        // bodied function?
                        if ctx.module.functions.iter().any(|f| {
                            f.def.as_ref().and_then(|d| d.result_id) == Some(*callee)
                                && !f.blocks.is_empty()
                        }) {
                            target = Some((bi, ii, *callee));
                            break 'scan;
                        }
                    }
                }
            }
        }
        let Some((bi, ii, callee_id)) = target else {
            break;
        };
        inline_one_call(ctx, entry_idx, bi, ii, callee_id, &mut pointer_types)?;
        stats.splices += 1;
        processed_helpers.insert(callee_id);
    }

    stats.helper_instances = processed_helpers.len();
    Ok(stats)
}

/// Drop unreferenced helper functions after the producer has made the entry self-contained.
pub(super) fn prune_unreferenced_functions(ctx: &mut Ctx, entry_idx: usize) {
    let entry_id = ctx.module.functions[entry_idx]
        .def
        .as_ref()
        .and_then(|d| d.result_id);
    let mut called: HashSet<Word> = HashSet::new();
    for blk in &ctx.module.functions[entry_idx].blocks {
        for inst in &blk.instructions {
            if inst.class.opcode == Op::FunctionCall {
                if let Some(Operand::IdRef(c)) = inst.operands.first() {
                    called.insert(*c);
                }
            }
        }
    }
    ctx.module.functions.retain(|f| {
        let id = f.def.as_ref().and_then(|d| d.result_id);
        id == entry_id || id.map(|i| called.contains(&i)).unwrap_or(false)
    });
}

/// How to materialize the merged array index for a re-rooted access chain.
enum MergePatch {
    /// Reuse an existing index operand verbatim (one side of the merge was the identity 0).
    None,
    /// A folded constant `OpConstant ty val` (both indices were constant).
    Const { slot: usize, ty: Word, val: u32 },
    /// A runtime `OpIAdd ty lhs rhs` inserted before the chain (an index was dynamic).
    Add {
        slot: usize,
        ty: Word,
        lhs: Word,
        rhs: Word,
    },
}

/// Member/element access chains ONLY (NOT `OpPtrAccessChain`, whose first index is a pointer STRIDE
/// rather than a member select). Restricting the re-root pass to these makes `walk_access_chain_type`
/// a faithful SPIR-V validity oracle — every index is a member/element select that the walk models
/// exactly — so the pass never misjudges a valid chain as invalid.
fn is_member_access_chain(op: Op) -> bool {
    matches!(op, Op::AccessChain | Op::InBoundsAccessChain)
}

/// The constant `u64` value of an id if it names an `OpConstant` of an integer type, else `None`.
fn const_int_value(defs: &HashMap<Word, Instruction>, id: Word) -> Option<u64> {
    let inst = defs.get(&id)?;
    if inst.class.opcode != Op::Constant {
        return None;
    }
    match inst.operands.first()? {
        Operand::LiteralBit32(v) => Some(*v as u64),
        _ => None,
    }
}

/// Walk a composite TYPE id through a sequence of access-chain index operands, returning the final
/// (innermost) type id, or `None` if any step indexes a non-composite (i.e. the chain is invalid).
/// Arrays/runtime-arrays/vectors/matrices deref to their element regardless of the index VALUE; a
/// struct needs a constant member index.
fn walk_access_chain_type(
    defs: &HashMap<Word, Instruction>,
    mut cur: Word,
    indices: &[Operand],
) -> Option<Word> {
    for op in indices {
        let def = defs.get(&cur)?;
        cur = match def.class.opcode {
            Op::TypeStruct => {
                let Operand::IdRef(idx_id) = op else {
                    return None;
                };
                let member = const_int_value(defs, *idx_id)? as usize;
                match def.operands.get(member)? {
                    Operand::IdRef(m) => *m,
                    _ => return None,
                }
            }
            Op::TypeArray | Op::TypeRuntimeArray | Op::TypeVector | Op::TypeMatrix => {
                match def.operands.first()? {
                    Operand::IdRef(elem) => *elem,
                    _ => return None,
                }
            }
            _ => return None,
        };
    }
    Some(cur)
}

/// Get or create an `OpConstant` of `type_id` holding `value` (32-bit literal), returning its id.
fn get_or_create_int_const(ctx: &mut Ctx, type_id: Word, value: u32) -> Word {
    for inst in ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.new_globals.iter())
    {
        if inst.class.opcode == Op::Constant
            && inst.result_type == Some(type_id)
            && inst.operands.first() == Some(&Operand::LiteralBit32(value))
        {
            if let Some(id) = inst.result_id {
                return id;
            }
        }
    }
    let id = ctx.module.fresh_id();
    ctx.new_globals.push(Instruction::new(
        Op::Constant,
        Some(type_id),
        Some(id),
        vec![Operand::LiteralBit32(value)],
    ));
    id
}

// --- M-B2 lever (a): inline splice-time byte reassembly for union byte-buffer reinterprets ----------
//
// A `union`/byte-buffer helper takes a pointer to a struct VIEW (`%struct.anon`) but the caller
// allocates the storage as a `[N x uchar]` byte array or a same-size scalar. When the inliner
// substitutes that differently shaped argument for the struct-pointer parameter, the callee's
// struct member-index chain (`i,0,0`) is applied to it, which spirv-val rejects
// (`OpInBoundsAccessChain reached non-composite`). The struct byte layout is known ONLY at splice
// time (post-inline the struct type is gone), so we compute the constant byte offset here and rewrite
// loads into little-endian extraction; explicit byte-array stores are reassembled too. See dead-end
// #23.

/// A recognized byte-view reinterpret chain reconstructed from its caller-side backing storage.
struct ByteViewPlan {
    /// Caller-side `[N x uchar]` array VARIABLE (the substituted argument).
    arg: Word,
    storage: StorageClass,
    /// The `uchar` (8-bit int) element type id when the caller stores an explicit byte array.
    elem_uchar_ty: Option<Word>,
    /// Caller scalar storage type when a same-size scalar allocation is viewed as a byte struct.
    /// Such plans are read-only and extract the requested little-endian field from one scalar load.
    scalar_backing_ty: Option<Word>,
    /// Byte offset of the reached scalar within the backing storage.
    offset: u32,
    /// Byte width of the reached scalar. Explicit byte arrays use this path only for width >= 2;
    /// scalar backing also admits one-byte views.
    width: u32,
    /// The reached scalar type id (== the original chain result pointee).
    scalar_ty: Word,
}

/// `[N x uchar]` info `(uchar_elem_ty, len)` if `ty` is an array of 8-bit ints, else `None`.
fn uchar_array_info(defs: &HashMap<Word, Instruction>, ty: Word) -> Option<(Word, u32)> {
    let def = defs.get(&ty)?;
    if def.class.opcode != Op::TypeArray {
        return None;
    }
    let Operand::IdRef(elem) = def.operands.first()? else {
        return None;
    };
    let ed = defs.get(elem)?;
    let is_uchar = ed.class.opcode == Op::TypeInt
        && matches!(ed.operands.first(), Some(Operand::LiteralBit32(8)));
    if !is_uchar {
        return None;
    }
    let Operand::IdRef(len_id) = def.operands.get(1)? else {
        return None;
    };
    let len = const_int_value(defs, *len_id)? as u32;
    Some((*elem, len))
}

/// Byte width of an OpTypeInt/OpTypeFloat scalar, else `None`.
fn scalar_byte_width(defs: &HashMap<Word, Instruction>, ty: Word) -> Option<u32> {
    let def = defs.get(&ty)?;
    match def.class.opcode {
        Op::TypeInt | Op::TypeFloat => match def.operands.first()? {
            Operand::LiteralBit32(bits) => Some(bits / 8),
            _ => None,
        },
        _ => None,
    }
}

/// Byte offset of `member` within `struct_ty`, mirroring `layout_ty_size_align`'s struct rule
/// (explicit AIR offsets when present, else natural `round_up` packing).
fn struct_member_byte_offset(
    ctx: &Ctx,
    defs: &HashMap<Word, Instruction>,
    struct_ty: Word,
    member: usize,
) -> Option<u32> {
    crate::layout::spirv_struct_member(
        struct_ty,
        member,
        defs,
        crate::layout::SpirvLayout::air_offsets(
            &ctx.air_struct_offsets,
            ctx.air_data_layout.as_ref(),
        ),
    )
    .map(|(offset, _)| offset)
}

/// Walk a composite aggregate through constant access-chain indices, returning the byte offset of the
/// reached member and its type id. Structs use `struct_member_byte_offset`; array/vector/matrix use the
/// element size. Any dynamic index or non-composite step aborts (`None`) — the reinterpret is only
/// recognized when the whole path is a constant byte offset.
fn const_member_byte_offset(
    ctx: &Ctx,
    defs: &HashMap<Word, Instruction>,
    mut cur: Word,
    indices: &[Operand],
) -> Option<(u32, Word)> {
    let mut offset: u32 = 0;
    for op in indices {
        let Operand::IdRef(idx_id) = op else {
            return None;
        };
        let def = defs.get(&cur)?;
        match def.class.opcode {
            Op::TypeStruct => {
                let member = const_int_value(defs, *idx_id)? as usize;
                let Operand::IdRef(member_ty) = def.operands.get(member)? else {
                    return None;
                };
                offset = offset.checked_add(struct_member_byte_offset(ctx, defs, cur, member)?)?;
                cur = *member_ty;
            }
            Op::TypeArray | Op::TypeRuntimeArray | Op::TypeVector | Op::TypeMatrix => {
                let Operand::IdRef(elem) = def.operands.first()? else {
                    return None;
                };
                let idx = const_int_value(defs, *idx_id)? as u32;
                let (esz, _) = crate::passes::stage_input::layout_ty_size_align(ctx, *elem, defs);
                offset = offset.checked_add(idx.checked_mul(esz)?)?;
                cur = *elem;
            }
            _ => return None,
        }
    }
    Some((offset, cur))
}

/// True iff every use of `chain_id` in the callee is an `OpLoad`/`OpStore` THROUGH it (as the pointer,
/// never the stored value). Only then can the chain be dropped and its accesses replaced with byte
/// reassembly without leaving a dangling reference.
fn callee_chain_uses_all_mem(callee: &Function, chain_id: Word) -> bool {
    let mut used = false;
    for blk in &callee.blocks {
        for inst in &blk.instructions {
            let refs = inst
                .operands
                .iter()
                .any(|o| matches!(o, Operand::IdRef(r) if *r == chain_id));
            if !refs {
                continue;
            }
            used = true;
            match inst.class.opcode {
                Op::Load => {
                    if inst.operands.first() != Some(&Operand::IdRef(chain_id)) {
                        return false;
                    }
                }
                Op::Store => {
                    // Must be the pointer (operand 0), never the stored value (operand 1).
                    if inst.operands.first() != Some(&Operand::IdRef(chain_id))
                        || inst.operands.get(1) == Some(&Operand::IdRef(chain_id))
                    {
                        return false;
                    }
                }
                _ => return false,
            }
        }
    }
    used
}

fn callee_chain_uses_all_loads(callee: &Function, chain_id: Word) -> bool {
    let mut used = false;
    for inst in callee
        .blocks
        .iter()
        .flat_map(|block| &block.instructions)
        .filter(|inst| {
            inst.operands
                .iter()
                .any(|operand| matches!(operand, Operand::IdRef(id) if *id == chain_id))
        })
    {
        used = true;
        if inst.class.opcode != Op::Load || inst.operands.first() != Some(&Operand::IdRef(chain_id))
        {
            return false;
        }
    }
    used
}

/// Scan the callee for byte-view reinterpret chains (struct-pointee PARAMETER substituted by a caller
/// `[N x uchar]` or same-size scalar argument). Returns a map keyed by the callee chain result id.
/// Immutable over `ctx`.
fn plan_byte_view_reinterprets(
    ctx: &Ctx,
    callee: &Function,
    remap: &HashMap<Word, Word>,
    result_types: &HashMap<Word, Word>,
) -> HashMap<Word, ByteViewPlan> {
    let mut plan = HashMap::new();
    let defs = type_defs_with_new_globals(ctx);
    for blk in &callee.blocks {
        for inst in &blk.instructions {
            if !is_member_access_chain(inst.class.opcode) {
                continue;
            }
            let Some(chain_id) = inst.result_id else {
                continue;
            };
            let indices = &inst.operands[1..];
            if indices.is_empty() {
                continue;
            }
            let Some(Operand::IdRef(base)) = inst.operands.first() else {
                continue;
            };
            // The callee base must point at a STRUCT (the union's struct view).
            let Some(base_ty) = result_types.get(base).copied() else {
                continue;
            };
            let Some(callee_pointee) = ptr_pointee(&defs, base_ty) else {
                continue;
            };
            if defs.get(&callee_pointee).map(|d| d.class.opcode) != Some(Op::TypeStruct) {
                continue;
            }
            // The substituted argument must be Function/Private storage backed by `[N x uchar]`, or
            // by a same-size scalar when every use of this chain is a load.
            let Some(&arg) = remap.get(base) else {
                continue;
            };
            let Some(arg_ty) = result_types.get(&arg).copied() else {
                continue;
            };
            let Some(storage) = ptr_storage(&defs, arg_ty) else {
                continue;
            };
            if !matches!(storage, StorageClass::Function | StorageClass::Private) {
                continue;
            }
            let Some(arg_pointee) = ptr_pointee(&defs, arg_ty) else {
                continue;
            };
            let (elem_uchar_ty, scalar_backing_ty, backing_size) =
                if let Some((elem_uchar_ty, array_len)) = uchar_array_info(&defs, arg_pointee) {
                    (Some(elem_uchar_ty), None, array_len)
                } else if let Some(width) = scalar_byte_width(&defs, arg_pointee) {
                    let (callee_size, _) = crate::passes::stage_input::layout_ty_size_align(
                        ctx,
                        callee_pointee,
                        &defs,
                    );
                    if width != callee_size || !callee_chain_uses_all_loads(callee, chain_id) {
                        continue;
                    }
                    (None, Some(arg_pointee), width)
                } else {
                    continue;
                };
            // Constant byte offset + reached scalar from the CALLEE struct layout.
            let Some((offset, reached)) =
                const_member_byte_offset(ctx, &defs, callee_pointee, indices)
            else {
                continue;
            };
            let Some(width) = scalar_byte_width(&defs, reached) else {
                continue;
            };
            if width == 0 || (elem_uchar_ty.is_some() && width < 2) || offset + width > backing_size
            {
                continue;
            }
            // The chain's declared result pointee must equal the reached scalar (sanity).
            if inst.result_type.and_then(|rt| ptr_pointee(&defs, rt)) != Some(reached) {
                continue;
            }
            if !callee_chain_uses_all_mem(callee, chain_id) {
                continue;
            }
            plan.insert(
                chain_id,
                ByteViewPlan {
                    arg,
                    storage,
                    elem_uchar_ty,
                    scalar_backing_ty,
                    offset,
                    width,
                    scalar_ty: reached,
                },
            );
        }
    }
    plan
}

/// Emit a little-endian reassembly of a `plan.width`-byte scalar from `plan.arg[offset..]`, binding the
/// result to `out_id : out_ty` (the original load's result).
fn emit_byte_view_load(
    ctx: &mut Ctx,
    plan: &ByteViewPlan,
    out_id: Word,
    out_ty: Word,
) -> Vec<Instruction> {
    if let Some(source_ty) = plan.scalar_backing_ty {
        let mut insts = Vec::new();
        let source_width = scalar_byte_width(&type_defs_with_new_globals(ctx), source_ty)
            .expect("scalar byte-view plans carry a scalar backing type");
        let source_word_ty = ctx.get_or_create(
            Op::TypeInt,
            None,
            vec![
                Operand::LiteralBit32(source_width * 8),
                Operand::LiteralBit32(0),
            ],
        );
        let loaded = ctx.module.fresh_id();
        insts.push(Instruction::new(
            Op::Load,
            Some(source_ty),
            Some(loaded),
            vec![Operand::IdRef(plan.arg)],
        ));
        let source_word = if source_ty == source_word_ty {
            loaded
        } else {
            let word = ctx.module.fresh_id();
            insts.push(Instruction::new(
                Op::Bitcast,
                Some(source_word_ty),
                Some(word),
                vec![Operand::IdRef(loaded)],
            ));
            word
        };
        let shifted = if plan.offset == 0 {
            source_word
        } else {
            let shifted = ctx.module.fresh_id();
            let shift = ctx.const_int_of(source_word_ty, (plan.offset * 8) as i64);
            insts.push(Instruction::new(
                Op::ShiftRightLogical,
                Some(source_word_ty),
                Some(shifted),
                vec![Operand::IdRef(source_word), Operand::IdRef(shift)],
            ));
            shifted
        };
        let field_word_ty = ctx.get_or_create(
            Op::TypeInt,
            None,
            vec![
                Operand::LiteralBit32(plan.width * 8),
                Operand::LiteralBit32(0),
            ],
        );
        let field_word = if source_word_ty == field_word_ty {
            shifted
        } else {
            let narrowed = ctx.module.fresh_id();
            insts.push(Instruction::new(
                Op::UConvert,
                Some(field_word_ty),
                Some(narrowed),
                vec![Operand::IdRef(shifted)],
            ));
            narrowed
        };
        insts.push(Instruction::new(
            if out_ty == field_word_ty {
                Op::CopyObject
            } else {
                Op::Bitcast
            },
            Some(out_ty),
            Some(out_id),
            vec![Operand::IdRef(field_word)],
        ));
        return insts;
    }
    let mut insts = vec![];
    let elem_uchar_ty = plan
        .elem_uchar_ty
        .expect("array byte-view plans carry an uchar element type");
    let uchar_ptr = ctx.ty_ptr(plan.storage, elem_uchar_ty);
    let idx_ty = ctx.ty_uint();
    let acc_ty = ctx.get_or_create(
        Op::TypeInt,
        None,
        vec![
            Operand::LiteralBit32(plan.width * 8),
            Operand::LiteralBit32(0),
        ],
    );
    let mut acc: Option<Word> = None;
    for k in 0..plan.width {
        let off_const = ctx.const_int_of(idx_ty, (plan.offset + k) as i64);
        let bp = ctx.module.fresh_id();
        insts.push(Instruction::new(
            Op::InBoundsAccessChain,
            Some(uchar_ptr),
            Some(bp),
            vec![Operand::IdRef(plan.arg), Operand::IdRef(off_const)],
        ));
        let byte = ctx.module.fresh_id();
        insts.push(Instruction::new(
            Op::Load,
            Some(elem_uchar_ty),
            Some(byte),
            vec![Operand::IdRef(bp)],
        ));
        let widened = ctx.module.fresh_id();
        insts.push(Instruction::new(
            Op::UConvert,
            Some(acc_ty),
            Some(widened),
            vec![Operand::IdRef(byte)],
        ));
        let contrib = if k == 0 {
            widened
        } else {
            let shamt = ctx.const_int_of(idx_ty, (k * 8) as i64);
            let sh = ctx.module.fresh_id();
            insts.push(Instruction::new(
                Op::ShiftLeftLogical,
                Some(acc_ty),
                Some(sh),
                vec![Operand::IdRef(widened), Operand::IdRef(shamt)],
            ));
            sh
        };
        acc = Some(match acc {
            None => contrib,
            Some(a) => {
                let o = ctx.module.fresh_id();
                insts.push(Instruction::new(
                    Op::BitwiseOr,
                    Some(acc_ty),
                    Some(o),
                    vec![Operand::IdRef(a), Operand::IdRef(contrib)],
                ));
                o
            }
        });
    }
    let word = acc.expect("width >= 2 guarantees at least one byte");
    let final_op = if out_ty == acc_ty {
        Op::CopyObject
    } else {
        Op::Bitcast
    };
    insts.push(Instruction::new(
        final_op,
        Some(out_ty),
        Some(out_id),
        vec![Operand::IdRef(word)],
    ));
    insts
}

/// Emit a little-endian decomposition of `value_id : plan.scalar_ty` into `plan.width` byte stores at
/// `plan.arg[offset..]`.
fn emit_byte_view_store(ctx: &mut Ctx, plan: &ByteViewPlan, value_id: Word) -> Vec<Instruction> {
    debug_assert!(plan.scalar_backing_ty.is_none());
    let mut insts = vec![];
    let elem_uchar_ty = plan
        .elem_uchar_ty
        .expect("writable byte-view plans carry an uchar element type");
    let uchar_ptr = ctx.ty_ptr(plan.storage, elem_uchar_ty);
    let idx_ty = ctx.ty_uint();
    let acc_ty = ctx.get_or_create(
        Op::TypeInt,
        None,
        vec![
            Operand::LiteralBit32(plan.width * 8),
            Operand::LiteralBit32(0),
        ],
    );
    let word = if plan.scalar_ty == acc_ty {
        value_id
    } else {
        let w = ctx.module.fresh_id();
        insts.push(Instruction::new(
            Op::Bitcast,
            Some(acc_ty),
            Some(w),
            vec![Operand::IdRef(value_id)],
        ));
        w
    };
    for k in 0..plan.width {
        let src = if k == 0 {
            word
        } else {
            let shamt = ctx.const_int_of(idx_ty, (k * 8) as i64);
            let sh = ctx.module.fresh_id();
            insts.push(Instruction::new(
                Op::ShiftRightLogical,
                Some(acc_ty),
                Some(sh),
                vec![Operand::IdRef(word), Operand::IdRef(shamt)],
            ));
            sh
        };
        let byte = ctx.module.fresh_id();
        insts.push(Instruction::new(
            Op::UConvert,
            Some(elem_uchar_ty),
            Some(byte),
            vec![Operand::IdRef(src)],
        ));
        let off_const = ctx.const_int_of(idx_ty, (plan.offset + k) as i64);
        let bp = ctx.module.fresh_id();
        insts.push(Instruction::new(
            Op::InBoundsAccessChain,
            Some(uchar_ptr),
            Some(bp),
            vec![Operand::IdRef(plan.arg), Operand::IdRef(off_const)],
        ));
        insts.push(Instruction::new(
            Op::Store,
            None,
            None,
            vec![Operand::IdRef(bp), Operand::IdRef(byte)],
        ));
    }
    insts
}

/// If `inst` (an ORIGINAL callee instruction) participates in a recognized byte-view reinterpret,
/// return the replacement instructions to splice in its place (already id-mapped); else `None` to fall
/// through to the normal per-instruction splice. A byte-view chain itself is dropped (empty vec) — its
/// loads/stores rebuild their own byte pointers off the array.
fn byte_view_splice(
    ctx: &mut Ctx,
    inst: &Instruction,
    plan: &HashMap<Word, ByteViewPlan>,
    remap: &HashMap<Word, Word>,
) -> Option<Vec<Instruction>> {
    if plan.is_empty() {
        return None;
    }
    if is_member_access_chain(inst.class.opcode)
        && inst
            .result_id
            .map(|r| plan.contains_key(&r))
            .unwrap_or(false)
    {
        return Some(vec![]);
    }
    if inst.class.opcode == Op::Load {
        if let Some(Operand::IdRef(ptr)) = inst.operands.first() {
            if let Some(p) = plan.get(ptr) {
                let out_id = inst.result_id.and_then(|r| remap.get(&r).copied())?;
                let out_ty = inst.result_type?;
                return Some(emit_byte_view_load(ctx, p, out_id, out_ty));
            }
        }
    }
    if inst.class.opcode == Op::Store {
        if let Some(Operand::IdRef(ptr)) = inst.operands.first() {
            if let Some(p) = plan.get(ptr) {
                let Some(Operand::IdRef(val)) = inst.operands.get(1) else {
                    return None;
                };
                let val_mapped = remap.get(val).copied().unwrap_or(*val);
                return Some(emit_byte_view_store(ctx, p, val_mapped));
            }
        }
    }
    None
}

/// Re-root an inlined helper's illegal strided access chain onto its caller's array root.
///
/// An inlined helper that did pointer arithmetic on a pointer PARAMETER `%p` emits a chain whose first
/// index is an LLVM pointer-stride (`getelementptr %T, ptr %p, i64 N, …`). After the inline pass
/// substitutes `%p` by the caller argument — itself an access chain `%root i0 … im` selecting one
/// element of an array of `%T` — that stride becomes an illegal index into the element type `%T`
/// (e.g. "index 1 into a 1-member struct" / "reached non-composite"). Compose the two chains: merge the
/// stride `N` into the array index `im` and re-root onto `%root`, then append the helper's remaining
/// in-element indices. Re-rooting an element pointer to its true array root is BYTE-IDENTICAL (same
/// address); the merge is verified by type-walking it to the SAME result pointee, and only chains that
/// are currently INVALID are rewritten — so a banked-passing module (all chains already valid) is never
/// touched, and the pass can only convert invalid→valid, never the reverse.
pub(super) fn compose_chained_access_chains(ctx: &mut Ctx, entry_idx: usize) {
    let mut guard = 0;
    loop {
        guard += 1;
        if guard > 256 {
            break;
        }
        let defs = type_defs_with_new_globals(ctx);
        let result_types = collect_result_types(ctx);
        let mut chains: HashMap<Word, Instruction> = HashMap::new();
        for blk in &ctx.module.functions[entry_idx].blocks {
            for inst in &blk.instructions {
                if is_member_access_chain(inst.class.opcode) {
                    if let Some(rid) = inst.result_id {
                        chains.insert(rid, inst.clone());
                    }
                }
            }
        }

        // (block, inst, root, merged-indices, how to materialize merged[patch_slot]).
        let mut found: Option<(usize, usize, Word, Vec<Operand>, MergePatch)> = None;
        'scan: for (bi, blk) in ctx.module.functions[entry_idx].blocks.iter().enumerate() {
            for (ii, o) in blk.instructions.iter().enumerate() {
                if !is_member_access_chain(o.class.opcode) {
                    continue;
                }
                let Some(o_res_ty) = o.result_type else {
                    continue;
                };
                let o_indices = &o.operands[1..];
                if o_indices.is_empty() {
                    continue;
                }
                let Some(Operand::IdRef(base)) = o.operands.first() else {
                    continue;
                };
                let Some(d) = chains.get(base) else {
                    continue;
                };
                let d_indices = &d.operands[1..];
                if d_indices.is_empty() {
                    continue;
                }
                // `P` = the pointee `%p` was substituted to point at (the array element type).
                let Some(base_ptr_ty) = result_types.get(base).copied() else {
                    continue;
                };
                let Some(p) = ptr_pointee(&defs, base_ptr_ty) else {
                    continue;
                };
                // Only re-root chains that are CURRENTLY INVALID (walking O over `P` fails). A valid
                // chain is a normal in-element access and must be left exactly as-is.
                if walk_access_chain_type(&defs, p, o_indices).is_some() {
                    continue;
                }
                // The caller chain D's root pointer + its pointee (the containing aggregate).
                let Some(Operand::IdRef(d_root)) = d.operands.first() else {
                    continue;
                };
                let Some(root_ptr_ty) = result_types.get(d_root).copied() else {
                    continue;
                };
                let Some(root_pointee) = ptr_pointee(&defs, root_ptr_ty) else {
                    continue;
                };
                // D's LAST index must select an ARRAY/vector/matrix element (not a struct member): only
                // then is `(D.last + helper.stride)` a valid same-address array-stride merge. If D's last
                // index were a struct member-select, adding the stride would silently land on a DIFFERENT
                // member (valid SPIR-V but byte-WRONG = fake conformance). The type reached by all of D's
                // indices EXCEPT the last is the container being indexed.
                let container_ty = walk_access_chain_type(
                    &defs,
                    root_pointee,
                    &d.operands[1..d.operands.len() - 1],
                );
                let container_is_array = container_ty.and_then(|t| defs.get(&t)).is_some_and(|d| {
                    matches!(
                        d.class.opcode,
                        Op::TypeArray | Op::TypeRuntimeArray | Op::TypeVector | Op::TypeMatrix
                    )
                });
                if !container_is_array {
                    continue;
                }
                // Merge the helper's stride (O's first index) into the array index (D's last index).
                // The dominant case is the caller passing `&array[..][0]` (D's last index == const 0),
                // so `0 + stride == stride` and the helper's (possibly dynamic) stride becomes the array
                // index directly — no new instruction needed. Also handle stride==0 and const+const
                // folding; a dynamic+dynamic / dynamic+nonzero-const merge would need an OpIAdd in the
                // caller block (not emitted here) → bail.
                let stride_op = o_indices[0].clone();
                let last_op = d_indices[d_indices.len() - 1].clone();
                let d_last_const = match &last_op {
                    Operand::IdRef(id) => const_int_value(&defs, *id),
                    _ => None,
                };
                let stride_const = match &stride_op {
                    Operand::IdRef(id) => const_int_value(&defs, *id),
                    _ => None,
                };
                let patch_slot = d_indices.len() - 1;
                let idx_ty = match (&last_op, &stride_op) {
                    (Operand::IdRef(id), _) | (_, Operand::IdRef(id)) => {
                        result_types.get(id).copied()
                    }
                    _ => None,
                };
                let (merged_last, patch): (Operand, MergePatch) = if d_last_const == Some(0) {
                    (stride_op.clone(), MergePatch::None)
                } else if stride_const == Some(0) {
                    (last_op.clone(), MergePatch::None)
                } else if let (Some(d_val), Some(s_val)) = (d_last_const, stride_const) {
                    let Some(idx_ty) = idx_ty else { continue };
                    // placeholder; the real OpConstant id is patched in after the scan borrow ends.
                    (
                        Operand::IdRef(0),
                        MergePatch::Const {
                            slot: patch_slot,
                            ty: idx_ty,
                            val: (d_val + s_val) as u32,
                        },
                    )
                } else {
                    // Both dynamic (or const≠0 + dynamic): synthesize `OpIAdd %idx d_last stride` in the
                    // caller block. Both operands are defined before O (D is an earlier chain; the stride
                    // is O's own index operand), so the inserted add dominates the use. OpIAdd requires
                    // the result + both operands share one int type, so only merge when the two index
                    // operands already have the SAME type (else bail — a width-converting merge is not
                    // worth the risk here).
                    let (Operand::IdRef(last_id), Operand::IdRef(stride_id2)) =
                        (&last_op, &stride_op)
                    else {
                        continue;
                    };
                    let (Some(lt), Some(st)) = (
                        result_types.get(last_id).copied(),
                        result_types.get(stride_id2).copied(),
                    ) else {
                        continue;
                    };
                    if lt != st {
                        continue;
                    }
                    (
                        Operand::IdRef(0),
                        MergePatch::Add {
                            slot: patch_slot,
                            ty: lt,
                            lhs: *last_id,
                            rhs: *stride_id2,
                        },
                    )
                };

                let mut merged: Vec<Operand> = Vec::new();
                merged.extend(d_indices[..patch_slot].iter().cloned());
                merged.push(merged_last);
                merged.extend(o_indices[1..].iter().cloned());

                // Verify the merged chain type-walks to the SAME result pointee (byte-safe gate). The
                // placeholder index (when present) only affects an Array/Vector deref, whose element
                // type is index-independent, so the walk is valid despite the not-yet-resolved const.
                let Some(want_pointee) = ptr_pointee(&defs, o_res_ty) else {
                    continue;
                };
                let reached = walk_access_chain_type(&defs, root_pointee, &merged);
                match reached {
                    Some(reached) if reached == want_pointee => {}
                    _ => continue,
                }

                found = Some((bi, ii, *d_root, merged, patch));
                break 'scan;
            }
        }

        let Some((bi, ii, d_root, mut merged, patch)) = found else {
            break;
        };
        // Materialize the merged index; `insert_at` is where the rewritten chain ends up (an inserted
        // OpIAdd pushes it down by one).
        let mut insert_at = ii;
        match patch {
            MergePatch::None => {}
            MergePatch::Const { slot, ty, val } => {
                let cid = get_or_create_int_const(ctx, ty, val);
                merged[slot] = Operand::IdRef(cid);
            }
            MergePatch::Add { slot, ty, lhs, rhs } => {
                let add_id = ctx.module.fresh_id();
                let add = Instruction::new(
                    Op::IAdd,
                    Some(ty),
                    Some(add_id),
                    vec![Operand::IdRef(lhs), Operand::IdRef(rhs)],
                );
                ctx.module.functions[entry_idx].blocks[bi]
                    .instructions
                    .insert(ii, add);
                insert_at = ii + 1;
                merged[slot] = Operand::IdRef(add_id);
            }
        }
        let mut new_ops = vec![Operand::IdRef(d_root)];
        new_ops.extend(merged);
        let inst = &mut ctx.module.functions[entry_idx].blocks[bi].instructions[insert_at];
        inst.operands = new_ops;
    }
}

/// Inline a single call instruction at (block bi, inst ii) calling `callee_id`.
fn inline_one_call(
    ctx: &mut Ctx,
    entry_idx: usize,
    bi: usize,
    ii: usize,
    callee_id: Word,
    pointer_types: &mut HashMap<Word, (StorageClass, Word)>,
) -> Result<(), String> {
    // Clone the callee function out of the module.
    let callee = ctx
        .module
        .functions
        .iter()
        .find(|f| {
            f.def.as_ref().and_then(|d| d.result_id) == Some(callee_id) && !f.blocks.is_empty()
        })
        .cloned()
        .ok_or_else(|| format!("bodied callee {callee_id} not found"))?;

    // Gather the call's result id + args.
    let (call_res, args): (Option<Word>, Vec<Word>) = {
        let inst = &ctx.module.functions[entry_idx].blocks[bi].instructions[ii];
        let args = inst.operands[1..]
            .iter()
            .filter_map(|o| match o {
                Operand::IdRef(r) => Some(*r),
                _ => None,
            })
            .collect();
        (inst.result_id, args)
    };

    // Build the id remap: params -> args; every other defined id -> a fresh id.
    let mut remap: HashMap<Word, Word> = HashMap::new();
    for (p, a) in callee.parameters.iter().zip(args.iter()) {
        if let Some(pid) = p.result_id {
            remap.insert(pid, *a);
        }
    }
    let definitions = ctx
        .module
        .types_global_values
        .iter()
        .chain(ctx.module.functions.iter().flat_map(|function| {
            function
                .parameters
                .iter()
                .chain(function.blocks.iter().flat_map(|block| &block.instructions))
        }))
        .filter_map(|instruction| instruction.result_id.map(|id| (id, instruction)))
        .collect::<HashMap<_, _>>();
    let copy_aliases = definitions
        .iter()
        .filter_map(|(&result, instruction)| {
            if instruction.class.opcode != Op::CopyObject {
                return None;
            }
            let Some(Operand::IdRef(source)) = instruction.operands.first() else {
                return None;
            };
            Some((result, resolve_copy_object_source(*source, &definitions)))
        })
        .collect::<HashMap<_, _>>();
    let mut elided_aggregate_extracts = HashSet::new();
    for instruction in callee.blocks.iter().flat_map(|block| &block.instructions) {
        if instruction.class.opcode != Op::CompositeExtract {
            continue;
        }
        let (Some(result), Some(Operand::IdRef(composite))) =
            (instruction.result_id, instruction.operands.first())
        else {
            continue;
        };
        let Some(root) = remap.get(composite).copied() else {
            continue;
        };
        let Some(path) = instruction
            .operands
            .iter()
            .skip(1)
            .map(|operand| match operand {
                Operand::LiteralBit32(index) => Some(*index),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        if let Some(source) = ctx
            .emit_sidecar
            .aggregate_pointer_values
            .iter()
            .find(|fact| fact.aggregate == root && fact.indices == path)
            .map(|fact| fact.source)
        {
            remap.insert(result, source);
            elided_aggregate_extracts.insert(result);
            continue;
        }
        let result_is_pointer = instruction
            .result_type
            .and_then(|ty| definitions.get(&ty))
            .is_some_and(|definition| definition.class.opcode == Op::TypePointer);
        if result_is_pointer
            || aggregate_path_type(root, &path, &definitions) == instruction.result_type
        {
            if let Some(object) = resolve_inserted_object(root, &path, &definitions) {
                remap.insert(result, object);
                elided_aggregate_extracts.insert(result);
            }
        }
    }

    // A raw pointer-handle load is represented by a module-scope Private placeholder, not by an
    // instruction result owned by the helper. Clone that placeholder per call instance when its
    // sidecar root is one of this callee's parameters. This gives the ordinary operand remap a
    // call-local id and lets the sidecar carry the matching parameter substitution to the entry.
    // Without the clone, two calls through different argument-buffer fields would share one global
    // placeholder and could not be routed independently.
    let callee_params = callee
        .parameters
        .iter()
        .filter_map(|parameter| parameter.result_id)
        .collect::<HashSet<_>>();
    let pointer_placeholders = ctx
        .emit_sidecar
        .buffer_pointer_field_loads
        .iter()
        .filter(|fact| callee_params.contains(&fact.root))
        .map(|fact| fact.id)
        .chain(
            ctx.emit_sidecar
                .buffer_pointer_dynamic_field_loads
                .iter()
                .filter(|fact| callee_params.contains(&fact.root))
                .map(|fact| fact.id),
        )
        .collect::<HashSet<_>>();
    for placeholder in pointer_placeholders {
        let Some(mut cloned) = ctx
            .module
            .types_global_values
            .iter()
            .find(|instruction| instruction.result_id == Some(placeholder))
            .cloned()
        else {
            continue;
        };
        if cloned.class.opcode != Op::Variable
            || !matches!(
                cloned.operands.first(),
                Some(Operand::StorageClass(StorageClass::Private))
            )
        {
            continue;
        }
        let fresh = ctx.module.fresh_id();
        cloned.result_id = Some(fresh);
        ctx.new_globals.push(cloned);
        remap.insert(placeholder, fresh);
    }
    let fresh_for = |old: Word, ctx: &mut Ctx, remap: &mut HashMap<Word, Word>| -> Word {
        *remap.entry(old).or_insert_with(|| ctx.module.fresh_id())
    };
    // Pre-allocate fresh ids for all block labels + instruction results.
    for blk in &callee.blocks {
        if let Some(lbl) = &blk.label {
            if let Some(rid) = lbl.result_id {
                fresh_for(rid, ctx, &mut remap);
            }
        }
        for inst in &blk.instructions {
            if let Some(rid) = inst.result_id {
                if !remap.contains_key(&rid) {
                    fresh_for(rid, ctx, &mut remap);
                }
            }
        }
    }
    ctx.emit_sidecar.clone_inlined_facts(&remap);
    ctx.emit_sidecar
        .remap_local_pointer_field_store_sources(&remap);
    ctx.emit_sidecar
        .remap_local_pointer_field_store_sources(&copy_aliases);

    let map_op = |op: &Operand, remap: &HashMap<Word, Word>| -> Operand {
        if let Operand::IdRef(r) = op {
            if let Some(n) = remap.get(r) {
                return Operand::IdRef(*n);
            }
        }
        op.clone()
    };
    let mut result_types = collect_result_types(ctx);
    // M-B2 lever (a): recognize union byte-buffer reinterprets (a struct-pointee param substituted by
    // a caller byte-array or same-size scalar) so their memory operations can be reconstructed at
    // splice time. An empty plan (no such chain) makes `byte_view_splice` a no-op.
    let byte_view_plan = plan_byte_view_reinterprets(ctx, &callee, &remap, &result_types);

    // Single-block callee: splice instructions (minus terminator), capture return value.
    if callee.blocks.len() == 1 {
        let blk = &callee.blocks[0];
        let mut spliced: Vec<Instruction> = vec![];
        let mut ret_val: Option<Word> = None;
        for inst in &blk.instructions {
            if inst
                .result_id
                .is_some_and(|result| elided_aggregate_extracts.contains(&result))
            {
                continue;
            }
            match inst.class.opcode {
                Op::ReturnValue => {
                    ret_val = inst.operands.first().map(|o| match map_op(o, &remap) {
                        Operand::IdRef(r) => r,
                        _ => 0,
                    });
                }
                Op::Return | Op::FunctionEnd | Op::Label => {}
                _ => {
                    if let Some(repl) = byte_view_splice(ctx, inst, &byte_view_plan, &remap) {
                        for r in &repl {
                            record_result_type(&mut result_types, r);
                        }
                        spliced.extend(repl);
                        continue;
                    }
                    let mut ni = inst.clone();
                    ni.result_id = inst.result_id.and_then(|r| remap.get(&r).copied());
                    ni.result_type = inst.result_type;
                    ni.operands = inst.operands.iter().map(|o| map_op(o, &remap)).collect();
                    retarget_inlined_pointer_result(ctx, &mut ni, &result_types, pointer_types);
                    record_result_type(&mut result_types, &ni);
                    spliced.push(ni);
                }
            }
        }
        // Replace the call instruction with the spliced body; rewire the call's result to ret_val.
        let block = &mut ctx.module.functions[entry_idx].blocks[bi];
        block.instructions.remove(ii);
        for (at, s) in (ii..).zip(spliced) {
            block.instructions.insert(at, s);
        }
        if let (Some(cr), Some(rv)) = (call_res, ret_val) {
            let func = &mut ctx.module.functions[entry_idx];
            replace_id_in_function(func, cr, rv);
        }
        return Ok(());
    }

    // Multi-block callee: split the caller block at the call and stitch the callee's blocks in.
    inline_multiblock(
        ctx,
        entry_idx,
        bi,
        ii,
        &callee,
        call_res,
        &remap,
        &map_op,
        &mut result_types,
        pointer_types,
        &byte_view_plan,
        &elided_aggregate_extracts,
    )
}

fn resolve_inserted_object(
    mut aggregate: Word,
    mut path: &[u32],
    definitions: &HashMap<Word, &Instruction>,
) -> Option<Word> {
    let mut visited = HashSet::new();
    while visited.insert(aggregate) {
        let definition = definitions.get(&aggregate)?;
        match definition.class.opcode {
            Op::CompositeInsert => {
                let [Operand::IdRef(object), Operand::IdRef(composite), indices @ ..] =
                    definition.operands.as_slice()
                else {
                    return None;
                };
                let insert_path = indices
                    .iter()
                    .map(|operand| match operand {
                        Operand::LiteralBit32(index) => Some(*index),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()?;
                if path == insert_path {
                    return Some(*object);
                }
                if let Some(suffix) = path.strip_prefix(insert_path.as_slice()) {
                    aggregate = *object;
                    path = suffix;
                } else {
                    aggregate = *composite;
                }
            }
            Op::CopyObject => {
                let Some(Operand::IdRef(source)) = definition.operands.first() else {
                    return None;
                };
                aggregate = *source;
            }
            _ => return None,
        }
    }
    None
}

fn resolve_copy_object_source(mut value: Word, definitions: &HashMap<Word, &Instruction>) -> Word {
    let mut visited = HashSet::new();
    while visited.insert(value) {
        let Some(definition) = definitions.get(&value) else {
            break;
        };
        if definition.class.opcode != Op::CopyObject {
            break;
        }
        let Some(Operand::IdRef(source)) = definition.operands.first() else {
            break;
        };
        value = *source;
    }
    value
}

fn aggregate_path_type(
    aggregate: Word,
    path: &[u32],
    definitions: &HashMap<Word, &Instruction>,
) -> Option<Word> {
    let mut ty = definitions.get(&aggregate)?.result_type?;
    for index in path {
        let definition = definitions.get(&ty)?;
        ty = match definition.class.opcode {
            Op::TypeStruct => match definition.operands.get(*index as usize)? {
                Operand::IdRef(member) => *member,
                _ => return None,
            },
            Op::TypeArray | Op::TypeRuntimeArray | Op::TypeVector | Op::TypeMatrix => {
                match definition.operands.first()? {
                    Operand::IdRef(element) => *element,
                    _ => return None,
                }
            }
            _ => return None,
        };
    }
    Some(ty)
}

/// Multi-block inline: the caller block is split into [pre | call | post]. The callee's entry block's
/// instructions are appended to `pre` (which then branches into the callee body), each callee
/// OpReturnValue is rewritten to store-into a result + branch to a continuation block holding `post`,
/// and the call result is replaced by the captured value via OpPhi at the continuation.
#[allow(clippy::too_many_arguments)]
fn inline_multiblock(
    ctx: &mut Ctx,
    entry_idx: usize,
    bi: usize,
    ii: usize,
    callee: &Function,
    call_res: Option<Word>,
    remap: &HashMap<Word, Word>,
    map_op: &dyn Fn(&Operand, &HashMap<Word, Word>) -> Operand,
    result_types: &mut HashMap<Word, Word>,
    pointer_types: &mut HashMap<Word, (StorageClass, Word)>,
    byte_view_plan: &HashMap<Word, ByteViewPlan>,
    elided_aggregate_extracts: &HashSet<Word>,
) -> Result<(), String> {
    // Continuation block id + the post-call instructions.
    let cont_id = ctx.module.fresh_id();
    // The caller block's label. Splitting moves this block's terminator into the continuation, so any
    // OpPhi in a successor that named this block as a predecessor must be redirected to `cont_id`.
    let caller_label = ctx.module.functions[entry_idx].blocks[bi]
        .label
        .as_ref()
        .and_then(|l| l.result_id);
    let mut post: Vec<Instruction> = ctx.module.functions[entry_idx].blocks[bi]
        .instructions
        .split_off(ii + 1);
    let (header_loop_merge, split_selection_merge) = lift_loop_merge_from_split_tail(&mut post);
    // remove the call instruction itself (last of pre now).
    ctx.module.functions[entry_idx].blocks[bi]
        .instructions
        .pop();

    let ret_ty = callee.def.as_ref().and_then(|d| d.result_type);

    // Build the callee blocks with remapped ids; collect (pred_label, value) for the return phi.
    let mut new_blocks: Vec<Block> = vec![];
    let mut phi_sources: Vec<(Word, Word)> = vec![];
    let old_callee_entry_label = callee.blocks[0].label.as_ref().and_then(|l| l.result_id);
    let callee_entry_label = old_callee_entry_label.and_then(|r| remap.get(&r).copied());
    let append_entry_to_caller = match (caller_label, old_callee_entry_label, callee_entry_label) {
        (Some(_), Some(old_entry), Some(_)) => {
            header_loop_merge.is_none()
                && split_selection_merge.is_none()
                && !callee_entry_has_internal_predecessor(callee, old_entry)
        }
        _ => false,
    };
    let mapped_operand = |op: &Operand, remap: &HashMap<Word, Word>| -> Operand {
        let mapped = map_op(op, remap);
        if append_entry_to_caller {
            if let (Some(caller), Some(entry)) = (caller_label, callee_entry_label) {
                if mapped == Operand::IdRef(entry) {
                    return Operand::IdRef(caller);
                }
            }
        }
        mapped
    };
    let mut caller_entry_insts: Vec<Instruction> = Vec::new();

    for (k, blk) in callee.blocks.iter().enumerate() {
        let new_label = blk
            .label
            .as_ref()
            .and_then(|l| l.result_id)
            .and_then(|r| remap.get(&r).copied());
        let label_inst = new_label.map(|id| Instruction::new(Op::Label, None, Some(id), vec![]));
        let mut insts: Vec<Instruction> = vec![];
        for inst in &blk.instructions {
            if inst
                .result_id
                .is_some_and(|result| elided_aggregate_extracts.contains(&result))
            {
                continue;
            }
            match inst.class.opcode {
                Op::ReturnValue => {
                    let val = inst
                        .operands
                        .first()
                        .map(|o| match mapped_operand(o, remap) {
                            Operand::IdRef(r) => r,
                            _ => 0,
                        });
                    let pred_label = if append_entry_to_caller && k == 0 {
                        caller_label
                    } else {
                        new_label
                    };
                    if let (Some(v), Some(lbl)) = (val, pred_label) {
                        phi_sources.push((lbl, v));
                    }
                    insts.push(Instruction::new(
                        Op::Branch,
                        None,
                        None,
                        vec![Operand::IdRef(cont_id)],
                    ));
                }
                Op::Return => {
                    insts.push(Instruction::new(
                        Op::Branch,
                        None,
                        None,
                        vec![Operand::IdRef(cont_id)],
                    ));
                }
                Op::FunctionEnd | Op::Label => {}
                _ => {
                    if let Some(repl) = byte_view_splice(ctx, inst, byte_view_plan, remap) {
                        for r in &repl {
                            record_result_type(result_types, r);
                        }
                        insts.extend(repl);
                        continue;
                    }
                    let mut ni = inst.clone();
                    ni.result_id = inst.result_id.and_then(|r| remap.get(&r).copied());
                    ni.result_type = inst.result_type;
                    ni.operands = inst
                        .operands
                        .iter()
                        .map(|o| mapped_operand(o, remap))
                        .collect();
                    retarget_inlined_pointer_result(ctx, &mut ni, result_types, pointer_types);
                    record_result_type(result_types, &ni);
                    insts.push(ni);
                }
            }
        }
        if append_entry_to_caller && k == 0 {
            caller_entry_insts = insts;
        } else {
            new_blocks.push(Block {
                label: label_inst,
                instructions: insts,
            });
        }
    }

    // If the callee entry has no internal predecessors, append it to the caller's split pre-block.
    // This keeps the common helper-call shape as straight-line code and avoids introducing an
    // unnecessary cross-block live range for resources and builtins. Otherwise, branch into the cloned
    // entry block so loops that target the callee entry remain self-contained.
    if append_entry_to_caller {
        let block = &mut ctx.module.functions[entry_idx].blocks[bi];
        block.instructions.extend(caller_entry_insts);
    } else if let Some(entry_lbl) = callee_entry_label {
        if let Some(loop_merge) = header_loop_merge {
            ctx.module.functions[entry_idx].blocks[bi]
                .instructions
                .push(loop_merge);
        }
        ctx.module.functions[entry_idx].blocks[bi]
            .instructions
            .push(Instruction::new(
                Op::Branch,
                None,
                None,
                vec![Operand::IdRef(entry_lbl)],
            ));
    } else {
        return Err("callee entry block has no label".into());
    }

    // Continuation block: OpPhi for the result (if any), then the post instructions.
    let mut cont_insts: Vec<Instruction> = vec![];
    if let (Some(cr), Some(rty)) = (call_res, ret_ty) {
        if phi_sources.len() == 1 {
            // single return -> OpCopyObject is enough (no real phi needed), but keep ids valid by phi.
            let (lbl, v) = phi_sources[0];
            cont_insts.push(Instruction::new(
                Op::Phi,
                Some(rty),
                Some(cr),
                vec![Operand::IdRef(v), Operand::IdRef(lbl)],
            ));
        } else if !phi_sources.is_empty() {
            let mut ops = vec![];
            for (lbl, v) in &phi_sources {
                ops.push(Operand::IdRef(*v));
                ops.push(Operand::IdRef(*lbl));
            }
            cont_insts.push(Instruction::new(Op::Phi, Some(rty), Some(cr), ops));
        }
    }
    cont_insts.extend(post);
    let cont_block = Block {
        label: Some(Instruction::new(Op::Label, None, Some(cont_id), vec![])),
        instructions: cont_insts,
    };

    // Insert the callee blocks + continuation right after the (split) caller block.
    let insert_at = bi + 1;
    {
        let func = &mut ctx.module.functions[entry_idx];
        let mut tail: Vec<Block> = func.blocks.split_off(insert_at);
        let tail_labels = tail
            .iter()
            .filter_map(|block| block.label.as_ref()?.result_id)
            .collect::<HashSet<_>>();
        func.blocks.extend(new_blocks);
        func.blocks.push(cont_block);
        func.blocks.append(&mut tail);

        // Redirect OpPhi predecessors from the (now-split) caller block to the continuation: the edge
        // into the caller's original successors now originates from `cont_id`, not the caller block
        // (which now branches only into the callee entry). Phi operands are (value, label) pairs —
        // labels at odd idx.
        if let Some(l) = caller_label {
            for blk in &mut func.blocks {
                let Some(label) = blk.label.as_ref().and_then(|label| label.result_id) else {
                    continue;
                };
                if !tail_labels.contains(&label) {
                    continue;
                }
                for inst in &mut blk.instructions {
                    if inst.class.opcode == Op::Phi {
                        let mut k = 1;
                        while k < inst.operands.len() {
                            if inst.operands[k] == Operand::IdRef(l) {
                                inst.operands[k] = Operand::IdRef(cont_id);
                            }
                            k += 2;
                        }
                    }
                }
            }
        }
    }
    if let Some(merge_id) = split_selection_merge {
        split_inlined_selection_merge(ctx, entry_idx, cont_id, merge_id);
    }
    Ok(())
}

fn lift_loop_merge_from_split_tail(
    post: &mut Vec<Instruction>,
) -> (Option<Instruction>, Option<Word>) {
    let loop_merge_idx = post
        .iter()
        .position(|inst| inst.class.opcode == Op::LoopMerge);
    let Some(loop_merge_idx) = loop_merge_idx else {
        return (None, None);
    };
    let loop_merge = post.remove(loop_merge_idx);
    if post.iter().any(|inst| inst.class.opcode == Op::LoopMerge) {
        return (Some(loop_merge), None);
    }
    let loop_merge_label = loop_merge.operands.first().and_then(as_id_ref);
    let continue_label = loop_merge.operands.get(1).and_then(as_id_ref);
    if let Some(selection_merge) = post
        .iter()
        .find(|inst| inst.class.opcode == Op::SelectionMerge)
        .and_then(|inst| inst.operands.first())
        .and_then(as_id_ref)
    {
        let shared_loop_boundary = (Some(selection_merge) == continue_label
            || Some(selection_merge) == loop_merge_label)
            .then_some(selection_merge);
        return (Some(loop_merge), shared_loop_boundary);
    }
    let Some(term_idx) = post.len().checked_sub(1) else {
        return (Some(loop_merge), None);
    };
    if post[term_idx].class.opcode != Op::BranchConditional {
        return (Some(loop_merge), None);
    }
    let true_label = post[term_idx].operands.get(1).and_then(as_id_ref);
    let false_label = post[term_idx].operands.get(2).and_then(as_id_ref);
    let selection_merge = if true_label == continue_label || false_label == continue_label {
        continue_label
    } else if true_label == loop_merge_label || false_label == loop_merge_label {
        loop_merge_label
    } else {
        continue_label.or(loop_merge_label)
    };
    if let Some(selection_merge) = selection_merge {
        post.insert(
            term_idx,
            Instruction::new(
                Op::SelectionMerge,
                None,
                None,
                vec![
                    Operand::IdRef(selection_merge),
                    Operand::SelectionControl(spirv::SelectionControl::NONE),
                ],
            ),
        );
        return (Some(loop_merge), Some(selection_merge));
    }
    (Some(loop_merge), None)
}

fn as_id_ref(operand: &Operand) -> Option<Word> {
    match operand {
        Operand::IdRef(id) => Some(*id),
        _ => None,
    }
}

fn split_inlined_selection_merge(ctx: &mut Ctx, entry_idx: usize, header_id: Word, merge_id: Word) {
    let synthetic_label = ctx.module.fresh_id();
    let Some((header_idx, merge_idx, insert_idx, construct_labels)) = (|| {
        let func = &ctx.module.functions[entry_idx];
        let header_idx = block_index_by_label(&func.blocks, header_id)?;
        let merge_idx = block_index_by_label(&func.blocks, merge_id)?;
        let construct_labels = construct_label_ids(&func.blocks, header_id, merge_id);
        let insert_idx = if merge_idx > header_idx {
            merge_idx
        } else {
            construct_insert_index(&func.blocks, &construct_labels).unwrap_or(header_idx + 1)
        };
        Some((header_idx, merge_idx, insert_idx, construct_labels))
    })() else {
        return;
    };

    let mut redirected_preds = HashSet::new();
    {
        let func = &mut ctx.module.functions[entry_idx];
        for block in &mut func.blocks {
            let Some(pred) = block.label.as_ref().and_then(|label| label.result_id) else {
                continue;
            };
            if !construct_labels.contains(&pred) {
                continue;
            }
            if redirect_terminator_target(block, merge_id, synthetic_label) {
                rewrite_structured_merge(block, merge_id, synthetic_label);
                redirected_preds.insert(pred);
            }
        }
        if !redirected_preds.is_empty() {
            rewrite_structured_merge(&mut func.blocks[header_idx], merge_id, synthetic_label);
        }
    }
    if redirected_preds.is_empty() {
        return;
    }

    let phi_splits = {
        let func = &ctx.module.functions[entry_idx];
        let Some(merge_block) = func.blocks.get(merge_idx) else {
            return;
        };
        let mut splits = Vec::new();
        for (inst_idx, inst) in merge_block.instructions.iter().enumerate() {
            if inst.class.opcode != Op::Phi {
                break;
            }
            let mut kept = Vec::new();
            let mut redirected = Vec::new();
            for pair in inst.operands.chunks(2) {
                if pair.len() != 2 {
                    kept.extend_from_slice(pair);
                    continue;
                }
                let is_redirected =
                    matches!(pair[1], Operand::IdRef(pred) if redirected_preds.contains(&pred));
                if is_redirected {
                    redirected.extend_from_slice(pair);
                } else {
                    kept.extend_from_slice(pair);
                }
            }
            if redirected.is_empty() {
                continue;
            }
            splits.push((inst_idx, inst.result_type, kept, redirected));
        }
        splits
    };

    let mut synthetic_instructions = Vec::new();
    let mut merge_phi_updates = Vec::new();
    for (inst_idx, result_type, mut kept, redirected) in phi_splits {
        let phi_id = ctx.module.fresh_id();
        synthetic_instructions.push(Instruction::new(
            Op::Phi,
            result_type,
            Some(phi_id),
            redirected,
        ));
        kept.push(Operand::IdRef(phi_id));
        kept.push(Operand::IdRef(synthetic_label));
        merge_phi_updates.push((inst_idx, kept));
    }
    synthetic_instructions.push(Instruction::new(
        Op::Branch,
        None,
        None,
        vec![Operand::IdRef(merge_id)],
    ));

    let func = &mut ctx.module.functions[entry_idx];
    if let Some(merge_block) = func.blocks.get_mut(merge_idx) {
        for (inst_idx, operands) in merge_phi_updates {
            if let Some(inst) = merge_block.instructions.get_mut(inst_idx) {
                inst.operands = operands;
            }
        }
    }
    func.blocks.insert(
        insert_idx,
        Block {
            label: Some(Instruction::new(
                Op::Label,
                None,
                Some(synthetic_label),
                vec![],
            )),
            instructions: synthetic_instructions,
        },
    );
}

fn construct_insert_index(blocks: &[Block], construct_labels: &HashSet<Word>) -> Option<usize> {
    blocks
        .iter()
        .enumerate()
        .filter_map(|(idx, block)| {
            let label = block.label.as_ref()?.result_id?;
            construct_labels.contains(&label).then_some(idx + 1)
        })
        .max()
}

fn block_index_by_label(blocks: &[Block], label_id: Word) -> Option<usize> {
    blocks
        .iter()
        .position(|block| block.label.as_ref().and_then(|label| label.result_id) == Some(label_id))
}

fn construct_label_ids(blocks: &[Block], header_id: Word, merge_id: Word) -> HashSet<Word> {
    let successors = block_successors_by_label(blocks);
    let mut seen = HashSet::new();
    let mut stack = vec![header_id];
    while let Some(label) = stack.pop() {
        if label == merge_id || !seen.insert(label) {
            continue;
        }
        if let Some(next) = successors.get(&label) {
            stack.extend(next.iter().copied().filter(|target| *target != merge_id));
        }
    }
    seen
}

fn callee_entry_has_internal_predecessor(callee: &Function, entry_label: Word) -> bool {
    callee
        .blocks
        .iter()
        .skip(1)
        .any(|block| block_successors(block).contains(&entry_label))
}

fn rewrite_structured_merge(block: &mut Block, from: Word, to: Word) {
    for inst in &mut block.instructions {
        if !matches!(inst.class.opcode, Op::SelectionMerge | Op::LoopMerge) {
            continue;
        }
        if let Some(Operand::IdRef(label)) = inst.operands.first_mut() {
            if *label == from {
                *label = to;
            }
        }
    }
}

fn redirect_terminator_target(block: &mut Block, from: Word, to: Word) -> bool {
    let Some(term) = block.instructions.last_mut() else {
        return false;
    };
    let mut changed = false;
    match term.class.opcode {
        Op::Branch => {
            if let Some(Operand::IdRef(label)) = term.operands.first_mut() {
                if *label == from {
                    *label = to;
                    changed = true;
                }
            }
        }
        Op::BranchConditional => {
            for operand in term.operands.iter_mut().skip(1).take(2) {
                if let Operand::IdRef(label) = operand {
                    if *label == from {
                        *label = to;
                        changed = true;
                    }
                }
            }
        }
        Op::Switch => {
            let mut idx = 1;
            while idx < term.operands.len() {
                if let Some(Operand::IdRef(label)) = term.operands.get_mut(idx) {
                    if *label == from {
                        *label = to;
                        changed = true;
                    }
                }
                idx += 2;
            }
        }
        _ => {}
    }
    changed
}

fn collect_result_types(ctx: &Ctx) -> HashMap<Word, Word> {
    let mut result_types = HashMap::new();
    for inst in ctx
        .module
        .ext_inst_imports
        .iter()
        .chain(ctx.module.types_global_values.iter())
        .chain(ctx.new_globals.iter())
    {
        record_result_type(&mut result_types, inst);
    }
    for func in &ctx.module.functions {
        if let Some(def) = &func.def {
            record_result_type(&mut result_types, def);
        }
        for param in &func.parameters {
            record_result_type(&mut result_types, param);
        }
        for block in &func.blocks {
            if let Some(label) = &block.label {
                record_result_type(&mut result_types, label);
            }
            for inst in &block.instructions {
                record_result_type(&mut result_types, inst);
            }
        }
    }
    result_types
}

fn record_result_type(result_types: &mut HashMap<Word, Word>, inst: &Instruction) {
    if let (Some(result), Some(result_type)) = (inst.result_id, inst.result_type) {
        result_types.insert(result, result_type);
    }
}

fn type_defs_with_new_globals(ctx: &Ctx) -> HashMap<Word, Instruction> {
    let mut defs = type_defs(&ctx.module);
    for inst in &ctx.new_globals {
        if let Some(result) = inst.result_id {
            defs.insert(result, inst.clone());
        }
    }
    defs
}

fn collect_pointer_types(ctx: &Ctx) -> HashMap<Word, (StorageClass, Word)> {
    ctx.module
        .types_global_values
        .iter()
        .chain(&ctx.new_globals)
        .filter_map(|inst| {
            match (
                inst.class.opcode,
                inst.result_id,
                inst.operands.first(),
                inst.operands.get(1),
            ) {
                (
                    Op::TypePointer,
                    Some(id),
                    Some(Operand::StorageClass(storage)),
                    Some(Operand::IdRef(pointee)),
                ) => Some((id, (*storage, *pointee))),
                _ => None,
            }
        })
        .collect()
}

fn pointer_type_info(
    ctx: &Ctx,
    pointer_types: &mut HashMap<Word, (StorageClass, Word)>,
    ty: Word,
) -> Option<(StorageClass, Word)> {
    if let Some(info) = pointer_types.get(&ty).copied() {
        return Some(info);
    }
    let info = ctx
        .module
        .types_global_values
        .iter()
        .chain(&ctx.new_globals)
        .find_map(|inst| {
            if inst.class.opcode != Op::TypePointer || inst.result_id != Some(ty) {
                return None;
            }
            match (inst.operands.first()?, inst.operands.get(1)?) {
                (Operand::StorageClass(storage), Operand::IdRef(pointee)) => {
                    Some((*storage, *pointee))
                }
                _ => None,
            }
        })?;
    pointer_types.insert(ty, info);
    Some(info)
}

fn retarget_inlined_pointer_result(
    ctx: &mut Ctx,
    inst: &mut Instruction,
    result_types: &HashMap<Word, Word>,
    pointer_types: &mut HashMap<Word, (StorageClass, Word)>,
) {
    let Some(result_type) = inst.result_type else {
        return;
    };
    let base = match inst.class.opcode {
        Op::AccessChain
        | Op::InBoundsAccessChain
        | Op::PtrAccessChain
        | Op::Bitcast
        | Op::CopyObject => match inst.operands.first() {
            Some(Operand::IdRef(id)) => *id,
            _ => return,
        },
        _ => return,
    };
    let Some(base_type) = result_types.get(&base).copied() else {
        return;
    };
    let Some((base_storage, _)) = pointer_type_info(ctx, pointer_types, base_type) else {
        return;
    };
    let Some((result_storage, pointee)) = pointer_type_info(ctx, pointer_types, result_type) else {
        return;
    };
    if base_storage != result_storage {
        let pointer_type = ctx.ty_ptr(base_storage, pointee);
        pointer_types.insert(pointer_type, (base_storage, pointee));
        inst.result_type = Some(pointer_type);
    }
}

#[cfg(test)]
mod tests;
