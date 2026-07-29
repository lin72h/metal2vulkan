//! Dead-global/debug cleanup and capability/extension closure.

use super::*;

fn needed_variable_pointer_capabilities(ctx: &Ctx) -> (bool, bool) {
    crate::spirv_variable_ptr::variable_pointer_requirement(&ctx.module)
}

/// Collect every id DEFINED in the module (globals + function defs/params/labels/instruction
/// results). Used to spot dangling debug-name / decoration references.
fn defined_ids(module: &Module) -> HashSet<Word> {
    let mut s = HashSet::new();
    for inst in &module.types_global_values {
        if let Some(id) = inst.result_id {
            s.insert(id);
        }
    }
    for inst in &module.ext_inst_imports {
        if let Some(id) = inst.result_id {
            s.insert(id);
        }
    }
    for f in &module.functions {
        if let Some(id) = f.def.as_ref().and_then(|d| d.result_id) {
            s.insert(id);
        }
        for p in &f.parameters {
            if let Some(id) = p.result_id {
                s.insert(id);
            }
        }
        for b in &f.blocks {
            if let Some(id) = b.label.as_ref().and_then(|l| l.result_id) {
                s.insert(id);
            }
            for inst in &b.instructions {
                if let Some(id) = inst.result_id {
                    s.insert(id);
                }
            }
        }
    }
    s
}

/// Collect ids referenced from function bodies. Entry-point interface variables should be live
/// because code loads/stores/access-chains them, not because an earlier pass registered them before a
/// later raw-alias rewrite replaced every use.
pub(super) fn function_referenced_ids(module: &Module) -> HashSet<Word> {
    let mut s = HashSet::new();
    for f in &module.functions {
        for b in &f.blocks {
            for inst in &b.instructions {
                for op in &inst.operands {
                    if let Operand::IdRef(r) | Operand::IdScope(r) | Operand::IdMemorySemantics(r) =
                        op
                    {
                        s.insert(*r);
                    }
                }
            }
        }
    }
    s
}

/// Remove global variables that are no longer referenced by code and no longer listed on the filtered
/// entry interface. Decorations/debug names for the removed ids are swept by drop_dangling_debug.
pub(super) fn drop_dead_unreferenced_variables(
    ctx: &mut Ctx,
    function_refs: &HashSet<Word>,
    interface_ids: &HashSet<Word>,
) {
    ctx.module.types_global_values.retain(|inst| {
        if inst.class.opcode != Op::Variable {
            return true;
        }
        let Some(id) = inst.result_id else {
            return true;
        };
        function_refs.contains(&id) || interface_ids.contains(&id)
    });
}

/// Remove OpName/OpDecorate/OpMemberDecorate whose target id is not defined anywhere.
pub(super) fn drop_dangling_debug(ctx: &mut Ctx) {
    let defined = defined_ids(&ctx.module);
    let keep = |inst: &Instruction| -> bool {
        match inst.operands.first() {
            Some(Operand::IdRef(id)) => defined.contains(id),
            _ => true,
        }
    };
    ctx.module.debug_names.retain(&keep);
    ctx.module.annotations.retain(&keep);
}

/// Iteratively drop types/global-values whose result id is referenced nowhere else in the module
/// (instruction operands, result-types, decorations, entry-point interface, function bodies). Keeps
/// variables referenced by the entry interface. Repeats to a fixpoint so chains die cleanly.
pub(super) fn gc_dead_globals(ctx: &mut Ctx) {
    loop {
        // Collect all referenced ids EXCEPT the self-definition in types_global_values.
        let mut used: HashSet<Word> = HashSet::new();
        // Local-pointer store sentinels are semantic roots even after their uses have been recovered
        // and rewritten. The typed sidecar replaces the former OpName reference that kept these
        // constants serialized through final GC; retaining them makes marker removal debug-only.
        used.extend(
            ctx.emit_sidecar
                .local_pointer_field_stores
                .iter()
                .map(|fact| fact.id),
        );
        let mut add = |id: &Word, set: &mut HashSet<Word>| {
            set.insert(*id);
        };
        let collect = |inst: &Instruction, set: &mut HashSet<Word>| {
            if let Some(t) = inst.result_type {
                set.insert(t);
            }
            for op in &inst.operands {
                if let Operand::IdRef(r) | Operand::IdScope(r) | Operand::IdMemorySemantics(r) = op
                {
                    set.insert(*r);
                }
            }
        };
        for sect in [
            &ctx.module.entry_points,
            &ctx.module.execution_modes,
            &ctx.module.annotations,
            &ctx.module.debug_names,
        ] {
            for inst in sect {
                collect(inst, &mut used);
            }
        }
        for f in &ctx.module.functions {
            if let Some(d) = &f.def {
                collect(d, &mut used);
            }
            for p in &f.parameters {
                collect(p, &mut used);
            }
            for b in &f.blocks {
                if let Some(l) = &b.label {
                    collect(l, &mut used);
                }
                for inst in &b.instructions {
                    collect(inst, &mut used);
                }
            }
        }
        // A types_global_values entry is live if its result id is used by anything above OR by
        // another (kept) global. Compute references among globals too.
        for inst in &ctx.module.types_global_values {
            // Don't let a dead global keep its own operands alive: only count references FROM a global
            // that is itself used. Approximate by two-phase: first pass treat all globals' operands as
            // potential refs, then the fixpoint loop removes the rest.
            collect(inst, &mut used);
            let _ = &mut add;
        }
        let before = ctx.module.types_global_values.len();
        ctx.module
            .types_global_values
            .retain(|inst| inst.result_id.map(|id| used.contains(&id)).unwrap_or(true));
        if ctx.module.types_global_values.len() == before {
            break;
        }
    }
}

/// Remove `OpCapability Int64` when no `OpTypeInt 64` survives (i.e. no genuine 64-bit-int type is
/// used anywhere after access-chain narrowing + dead-global gc). Leaving a capability whose feature is
/// unused is legal for spirv-val, but the Int64 declaration is what cues NVIDIA's compiler down the
/// 64-bit path that crashes; dropping it keeps the module strictly 32-bit. If genuine i64 math remains
/// the type is still present and we leave the capability (and `add_needed_capabilities` would re-add it).
pub(super) fn drop_unused_int64_capability(ctx: &mut Ctx) {
    let has_int64_type = ctx.module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::TypeInt && i.operands.first() == Some(&Operand::LiteralBit32(64))
    });
    if has_int64_type {
        return;
    }
    ctx.module.capabilities.retain(|c| {
        !matches!(
            c.operands.first(),
            Some(Operand::Capability(spirv::Capability::Int64))
        )
    });
}

pub(super) fn drop_unused_variable_pointer_capabilities(ctx: &mut Ctx) {
    crate::spirv_variable_ptr::lower_zero_base_storage_buffer_ptr_access_chains(&mut ctx.module);
    crate::spirv_variable_ptr::rewrite_storage_buffer_atomic_scopes(&mut ctx.module);
    let (needs_storage_buffer, needs_other) = needed_variable_pointer_capabilities(ctx);
    ctx.module
        .capabilities
        .retain(|c| match c.operands.first() {
            Some(Operand::Capability(spirv::Capability::VariablePointersStorageBuffer)) => {
                needs_storage_buffer
            }
            Some(Operand::Capability(spirv::Capability::VariablePointers)) => needs_other,
            _ => true,
        });
}

pub(super) fn add_needed_capabilities(ctx: &mut Ctx) {
    use spirv::Capability;
    let mut want: Vec<Capability> = vec![];
    let has_demote = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| i.class.opcode == Op::DemoteToHelperInvocation);
    if has_demote {
        want.push(Capability::DemoteToHelperInvocation);
    }
    let has_sampled_1d = ctx.module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::TypeImage && i.operands.get(1) == Some(&Operand::Dim(Dim::Dim1D))
    });
    let has_storage_1d = ctx.module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::TypeImage
            && i.operands.get(1) == Some(&Operand::Dim(Dim::Dim1D))
            && i.operands.get(5) == Some(&Operand::LiteralBit32(2))
    });
    if has_sampled_1d {
        want.push(Capability::Sampled1D);
    }
    if has_storage_1d {
        want.push(Capability::Image1D);
    }
    let has_buffer_image = ctx.module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::TypeImage && i.operands.get(1) == Some(&Operand::Dim(Dim::DimBuffer))
    });
    let has_storage_buffer_image = ctx.module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::TypeImage
            && i.operands.get(1) == Some(&Operand::Dim(Dim::DimBuffer))
            && i.operands.get(5) == Some(&Operand::LiteralBit32(2))
    });
    if has_buffer_image {
        want.push(Capability::SampledBuffer);
    }
    if has_storage_buffer_image {
        want.push(Capability::ImageBuffer);
    }
    let has_input_attachment_type = ctx.module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::TypeImage
            && i.operands.get(1) == Some(&Operand::Dim(Dim::DimSubpassData))
    });
    let has_input_attachment_decor = ctx.module.annotations.iter().any(|i| {
        i.class.opcode == Op::Decorate
            && i.operands.get(1) == Some(&Operand::Decoration(Decoration::InputAttachmentIndex))
    });
    if has_input_attachment_type || has_input_attachment_decor {
        want.push(Capability::InputAttachment);
    }
    let has_viewport_index = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::ViewportIndex))
    });
    if has_viewport_index {
        want.push(Capability::ShaderViewportIndex);
        if let Some(header) = ctx.module.header.as_mut() {
            let version = header.version();
            if version < (1, 5) {
                header.set_version(1, 5);
            }
        }
    }
    let has_layer = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::Layer))
    });
    if has_layer {
        want.push(Capability::ShaderLayer);
        if let Some(header) = ctx.module.header.as_mut() {
            let version = header.version();
            if version < (1, 5) {
                header.set_version(1, 5);
            }
        }
    }
    let has_clip_distance = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::ClipDistance))
    });
    if has_clip_distance {
        want.push(Capability::ClipDistance);
    }
    let has_primitive_id = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::PrimitiveId))
    });
    if has_primitive_id {
        want.push(Capability::Geometry);
    }
    let has_sample_id = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::SampleId))
    });
    if has_sample_id {
        want.push(Capability::SampleRateShading);
    }
    let has_stencil_export = ctx.module.annotations.iter().any(|instruction| {
        instruction.class.opcode == Op::Decorate
            && instruction.operands.get(1) == Some(&Operand::Decoration(Decoration::BuiltIn))
            && instruction.operands.get(2) == Some(&Operand::BuiltIn(BuiltIn::FragStencilRefEXT))
    });
    if has_stencil_export {
        want.push(Capability::StencilExportEXT);
    }
    // Texture query opcodes need the ImageQuery capability.
    let has_query = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::ImageQuerySize
                    | Op::ImageQuerySizeLod
                    | Op::ImageQueryLod
                    | Op::ImageQueryLevels
            )
        });
    if has_query {
        want.push(Capability::ImageQuery);
    }
    let has_group_shuffle = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformShuffle | Op::GroupNonUniformShuffleXor
            )
        });
    if has_group_shuffle {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformShuffle);
    }
    let has_group_shuffle_relative = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformShuffleUp | Op::GroupNonUniformShuffleDown
            )
        });
    if has_group_shuffle_relative {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformShuffleRelative);
    }
    let has_group_elect = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| i.class.opcode == Op::GroupNonUniformElect);
    if has_group_elect {
        want.push(Capability::GroupNonUniform);
    }
    let has_group_vote = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformAll | Op::GroupNonUniformAny | Op::GroupNonUniformAllEqual
            )
        });
    if has_group_vote {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformVote);
    }
    let has_group_ballot = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformBallot
                    | Op::GroupNonUniformBallotBitExtract
                    | Op::GroupNonUniformBallotBitCount
                    | Op::GroupNonUniformBallotFindLSB
                    | Op::GroupNonUniformBallotFindMSB
            )
        });
    if has_group_ballot {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformBallot);
    }
    let has_group_broadcast = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformBroadcast | Op::GroupNonUniformBroadcastFirst
            )
        });
    if has_group_broadcast {
        // BroadcastFirst (and Broadcast) are gated by the Ballot capability in core SPIR-V.
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformBallot);
    }
    let has_group_arithmetic = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            matches!(
                i.class.opcode,
                Op::GroupNonUniformIAdd
                    | Op::GroupNonUniformFAdd
                    | Op::GroupNonUniformSMin
                    | Op::GroupNonUniformUMin
                    | Op::GroupNonUniformFMin
                    | Op::GroupNonUniformSMax
                    | Op::GroupNonUniformUMax
                    | Op::GroupNonUniformFMax
                    | Op::GroupNonUniformBitwiseAnd
                    | Op::GroupNonUniformBitwiseOr
                    | Op::GroupNonUniformBitwiseXor
            )
        });
    if has_group_arithmetic {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformArithmetic);
    }
    // A `ClusteredReduce` group operation (emitted by the M-D2 `TransformOptions::simd_cluster32`
    // simd lowering) additionally needs the `GroupNonUniformClustered` capability.
    let has_clustered_reduce = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| {
            i.operands.iter().any(|o| {
                matches!(
                    o,
                    Operand::GroupOperation(spirv::GroupOperation::ClusteredReduce)
                )
            })
        });
    if has_clustered_reduce {
        want.push(Capability::GroupNonUniform);
        want.push(Capability::GroupNonUniformClustered);
    }
    let has_atomic_fadd = ctx
        .module
        .functions
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.instructions.iter())
        .any(|i| i.class.opcode == Op::AtomicFAddEXT);
    if has_atomic_fadd {
        want.push(Capability::AtomicFloat32AddEXT);
    }
    // Width-based scalar capabilities: a `half` (OpTypeFloat 16) needs Float16; an 8-/16-bit int needs
    // Int8/Int16. llc usually emits these for half shaders, but we synthesize half consts + types in the
    // lowering (saturate edges, FConvert results, the flat-buffer `<2 x half>`/`<2 x i16>` extracts), so
    // assert them here so the module is always self-consistent (a missing one is a spirv-val error).
    let int_width = |w: u32| {
        ctx.module.types_global_values.iter().any(|i| {
            i.class.opcode == Op::TypeInt && i.operands.first() == Some(&Operand::LiteralBit32(w))
        })
    };
    let has_half = ctx.module.types_global_values.iter().any(|i| {
        i.class.opcode == Op::TypeFloat && i.operands.first() == Some(&Operand::LiteralBit32(16))
    });
    if has_half {
        want.push(Capability::Float16);
    }
    if int_width(8) {
        want.push(Capability::Int8);
    }
    if int_width(16) {
        want.push(Capability::Int16);
    }
    if int_width(64) {
        want.push(Capability::Int64);
    }
    let (has_storage_buffer_pointer_merge, has_other_pointer_merge) =
        needed_variable_pointer_capabilities(ctx);
    if has_storage_buffer_pointer_merge {
        want.push(Capability::VariablePointersStorageBuffer);
    }
    if has_other_pointer_merge {
        want.push(Capability::VariablePointers);
    }
    for cap in &want {
        let already = ctx.module.capabilities.iter().any(|c| {
            matches!(c.operands.first(), Some(Operand::Capability(existing)) if existing == cap)
        });
        if !already {
            ctx.module.capabilities.push(Instruction::new(
                Op::Capability,
                None,
                None,
                vec![Operand::Capability(*cap)],
            ));
        }
    }
    // DemoteToHelperInvocation needs its SPIR-V extension declared (core only from SPIR-V 1.6).
    if want.contains(&Capability::DemoteToHelperInvocation) {
        let ext = "SPV_EXT_demote_to_helper_invocation";
        let have = ctx
            .module
            .extensions
            .iter()
            .any(|e| matches!(e.operands.first(), Some(Operand::LiteralString(s)) if s == ext));
        if !have {
            ctx.module.extensions.push(Instruction::new(
                Op::Extension,
                None,
                None,
                vec![Operand::LiteralString(ext.into())],
            ));
        }
    }
    if want.contains(&Capability::AtomicFloat32AddEXT) {
        let ext = "SPV_EXT_shader_atomic_float_add";
        let have = ctx
            .module
            .extensions
            .iter()
            .any(|e| matches!(e.operands.first(), Some(Operand::LiteralString(s)) if s == ext));
        if !have {
            ctx.module.extensions.push(Instruction::new(
                Op::Extension,
                None,
                None,
                vec![Operand::LiteralString(ext.into())],
            ));
        }
    }
    if want.contains(&Capability::StencilExportEXT) {
        let ext = "SPV_EXT_shader_stencil_export";
        let have = ctx
            .module
            .extensions
            .iter()
            .any(|e| matches!(e.operands.first(), Some(Operand::LiteralString(s)) if s == ext));
        if !have {
            ctx.module.extensions.push(Instruction::new(
                Op::Extension,
                None,
                None,
                vec![Operand::LiteralString(ext.into())],
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::Module;
    use crate::spirv_module::ModuleHeader;

    #[test]
    fn gc_keeps_local_pointer_store_sentinel_rooted_by_typed_sidecar() {
        let ulong = 1;
        let sentinel = 2;
        let dead = 3;
        let mut module = Module::new();
        module.header = Some(ModuleHeader::new(4));
        module.types_global_values = vec![
            Instruction::new(
                Op::TypeInt,
                None,
                Some(ulong),
                vec![Operand::LiteralBit32(64), Operand::LiteralBit32(0)],
            ),
            Instruction::new(Op::ConstantNull, Some(ulong), Some(sentinel), vec![]),
            Instruction::new(Op::ConstantNull, Some(ulong), Some(dead), vec![]),
        ];
        let mut ctx = Ctx::new(module);
        ctx.emit_sidecar.local_pointer_field_stores.push(
            crate::emit_sidecar::LocalPointerFieldStore {
                id: sentinel,
                source: 99,
            },
        );

        gc_dead_globals(&mut ctx);

        let ids = ctx
            .module
            .types_global_values
            .iter()
            .filter_map(|inst| inst.result_id)
            .collect::<HashSet<_>>();
        assert!(ids.contains(&ulong));
        assert!(ids.contains(&sentinel));
        assert!(!ids.contains(&dead));
    }
}
