//! Structural access-chain, pointer-index, and typed memory lowering.

use super::*;

mod access_provenance;
pub(in crate::passes) use access_provenance::*;
mod private_memory;
pub(in crate::passes) use private_memory::*;
mod access_chain;
pub(in crate::passes) use access_chain::*;
mod index_remap;
pub(in crate::passes) use index_remap::*;
mod vector_subword;
pub(in crate::passes) use vector_subword::*;
mod scalar_store;
pub(in crate::passes) use scalar_store::*;
mod workgroup;
pub(in crate::passes) use workgroup::*;
mod byte_aggregate;
pub(in crate::passes) use byte_aggregate::*;
mod dynamic_reinterpret;
pub(in crate::passes) use dynamic_reinterpret::*;
mod raw_byte;
pub(in crate::passes) use raw_byte::*;

#[cfg(test)]
mod byte_reinterpret_tests {
    //! Fixture + idempotence coverage for the byte-reinterpret / strided access-chain rewrites
    //! (plan milestone M-D6). These are the highest silent-wrong-bytes risk in the passes layer and
    //! carried zero unit tests. Every fixture is hand-built from crate-owned module/function/block
    //! carriers plus grammar-derived instruction/operand nodes, and every assertion inspects the
    //! parsed module (opcodes/operands/result types), never stringified SPIR-V.
    use super::*;
    use crate::spirv_module::{Block, Function};

    /// Install a single-block entry function (index 0) whose instructions are `body` followed by a
    /// terminator, so a pass keyed on `functions[entry_idx]` operates on exactly this block.
    fn install_entry(ctx: &mut Ctx, mut body: Vec<Instruction>) {
        body.push(Instruction::new(Op::Return, None, None, vec![]));
        let label = ctx.module.fresh_id();
        let func_id = ctx.module.fresh_id();
        ctx.module.functions.push(Function {
            def: Some(Instruction::new(Op::Function, None, Some(func_id), vec![])),
            end: Some(Instruction::new(Op::FunctionEnd, None, None, vec![])),
            parameters: vec![],
            blocks: vec![Block {
                label: Some(Instruction::new(Op::Label, None, Some(label), vec![])),
                instructions: body,
            }],
        });
    }

    /// A StorageBuffer `OpVariable` of pointer type `ptr_ty`, discoverable by `value_result_type`.
    fn storage_buffer_var(ctx: &mut Ctx, ptr_ty: Word) -> Word {
        let id = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::Variable,
            Some(ptr_ty),
            Some(id),
            vec![Operand::StorageClass(StorageClass::StorageBuffer)],
        ));
        id
    }

    fn only_inst(ctx: &Ctx) -> &Instruction {
        &ctx.module.functions[0].blocks[0].instructions[0]
    }

    // ---- rewrite_strided_descent_access_chains ------------------------------------------------

    /// A leading-stride GEP over a `RuntimeArray<float>` base — indices `[i, j]` over-index (the
    /// second index lands on the scalar `float` and fails the walk) while the descent reading `[j]`
    /// reaches the `float` result pointee. The pass must flip the opcode to `OpPtrAccessChain` and
    /// leave the operands untouched.
    #[test]
    fn strided_descent_flips_overindexing_chain_to_ptr_access_chain() {
        let mut ctx = Ctx::new(Module::new());
        let float = ctx.ty_float();
        let rt = ctx.ty_runtime_array(float);
        let ptr_rt = ctx.ty_ptr(StorageClass::StorageBuffer, rt);
        let ptr_float = ctx.ty_ptr(StorageClass::StorageBuffer, float);
        let base = storage_buffer_var(&mut ctx, ptr_rt);
        let i = ctx.const_uint(0);
        let j = ctx.const_uint(1);
        let chain_id = ctx.module.fresh_id();
        let ops = vec![Operand::IdRef(base), Operand::IdRef(i), Operand::IdRef(j)];
        install_entry(
            &mut ctx,
            vec![Instruction::new(
                Op::InBoundsAccessChain,
                Some(ptr_float),
                Some(chain_id),
                ops.clone(),
            )],
        );

        rewrite_strided_descent_access_chains(&mut ctx, 0);

        let inst = only_inst(&ctx);
        assert_eq!(inst.class.opcode, Op::PtrAccessChain, "opcode flipped");
        assert_eq!(inst.result_type, Some(ptr_float), "result type preserved");
        assert_eq!(inst.result_id, Some(chain_id), "result id preserved");
        assert_eq!(
            inst.operands, ops,
            "operands (base + both indices) unchanged"
        );
    }

    /// Running the pass on its own output is a no-op: the flipped `OpPtrAccessChain` no longer
    /// matches the `InBoundsAccessChain | AccessChain` gate, so nothing else in the block moves.
    #[test]
    fn strided_descent_is_idempotent() {
        let mut ctx = Ctx::new(Module::new());
        let float = ctx.ty_float();
        let rt = ctx.ty_runtime_array(float);
        let ptr_rt = ctx.ty_ptr(StorageClass::StorageBuffer, rt);
        let ptr_float = ctx.ty_ptr(StorageClass::StorageBuffer, float);
        let base = storage_buffer_var(&mut ctx, ptr_rt);
        let i = ctx.const_uint(0);
        let j = ctx.const_uint(1);
        let chain_id = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![Instruction::new(
                Op::InBoundsAccessChain,
                Some(ptr_float),
                Some(chain_id),
                vec![Operand::IdRef(base), Operand::IdRef(i), Operand::IdRef(j)],
            )],
        );

        rewrite_strided_descent_access_chains(&mut ctx, 0);
        let after_first = ctx.module.functions[0].blocks[0].instructions.clone();
        rewrite_strided_descent_access_chains(&mut ctx, 0);
        let after_second = &ctx.module.functions[0].blocks[0].instructions;
        assert_eq!(
            &after_first, after_second,
            "second application leaves the block byte-identical"
        );
    }

    /// A fully in-bounds two-index chain (`RuntimeArray<RuntimeArray<float>>`) walks cleanly, so the
    /// pass must decline it — a valid/banked chain is never disturbed (the floor guard).
    #[test]
    fn strided_descent_leaves_valid_chain_untouched() {
        let mut ctx = Ctx::new(Module::new());
        let float = ctx.ty_float();
        let inner = ctx.ty_runtime_array(float);
        let outer = ctx.ty_runtime_array(inner);
        let ptr_outer = ctx.ty_ptr(StorageClass::StorageBuffer, outer);
        let ptr_float = ctx.ty_ptr(StorageClass::StorageBuffer, float);
        let base = storage_buffer_var(&mut ctx, ptr_outer);
        let i = ctx.const_uint(0);
        let j = ctx.const_uint(1);
        let chain_id = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![Instruction::new(
                Op::InBoundsAccessChain,
                Some(ptr_float),
                Some(chain_id),
                vec![Operand::IdRef(base), Operand::IdRef(i), Operand::IdRef(j)],
            )],
        );

        rewrite_strided_descent_access_chains(&mut ctx, 0);

        assert_eq!(
            only_inst(&ctx).class.opcode,
            Op::InBoundsAccessChain,
            "a cleanly-walking chain is not rewritten"
        );
    }

    // ---- rewrite_dynamic_struct_index_reinterpret ---------------------------------------------

    /// A single-member Block `{ RuntimeArray<uint> }` accessed by ONE dynamic index that yields a
    /// `float*` view (`OpInBoundsAccessChain %ptr_float %buf %dyn`, exactly 2 operands, `%dyn`
    /// non-constant, `uint`/`float` same-width). The pass must descend member-0 (insert `%uint_0`),
    /// retype the chain to the `uint` element pointer, and split the `float` load into a `uint` load
    /// + `OpBitcast` to `float`.
    fn find_inst(ctx: &Ctx, id: Word) -> &Instruction {
        ctx.module.functions[0].blocks[0]
            .instructions
            .iter()
            .find(|i| i.result_id == Some(id))
            .expect("instruction with the given result id")
    }

    #[test]
    fn dynamic_struct_index_reinterpret_descends_member0_and_bitcasts_load() {
        let mut ctx = Ctx::new(Module::new());
        let uint = ctx.ty_uint();
        let float = ctx.ty_float();
        let rt = ctx.ty_runtime_array(uint);
        let struct_id = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_id),
            vec![Operand::IdRef(rt)],
        ));
        let ptr_struct = ctx.ty_ptr(StorageClass::StorageBuffer, struct_id);
        let ptr_float = ctx.ty_ptr(StorageClass::StorageBuffer, float);
        let base = storage_buffer_var(&mut ctx, ptr_struct);
        let dyn_idx = ctx.module.fresh_id();
        let chain_id = ctx.module.fresh_id();
        let val_id = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![
                // A dynamic (non-constant) index value.
                Instruction::new(Op::Undef, Some(uint), Some(dyn_idx), vec![]),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_float),
                    Some(chain_id),
                    vec![Operand::IdRef(base), Operand::IdRef(dyn_idx)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(float),
                    Some(val_id),
                    vec![Operand::IdRef(chain_id)],
                ),
            ],
        );

        rewrite_dynamic_struct_index_reinterpret(&mut ctx, 0).unwrap();

        // Chain now descends member-0 into the uint element and is retyped to the uint pointer.
        let chain = find_inst(&ctx, chain_id);
        assert_eq!(chain.class.opcode, Op::InBoundsAccessChain);
        assert_eq!(chain.operands.len(), 3, "member-0 index inserted");
        assert_eq!(chain.operands[0], Operand::IdRef(base));
        let Operand::IdRef(member0) = chain.operands[1] else {
            panic!("member index is not an id");
        };
        assert_eq!(
            const_u32(&ctx, member0),
            Some(0),
            "inserted index is constant 0"
        );
        let Some(Operand::IdRef(chain_pointee)) =
            type_def_of(&ctx, chain.result_type.unwrap()).and_then(|d| d.operands.get(1).cloned())
        else {
            panic!("chain result type is not a pointer");
        };
        assert_eq!(
            chain_pointee, uint,
            "chain retyped to the uint element pointer"
        );

        // The original float load id is now an OpBitcast fed by a fresh uint load.
        let cast = find_inst(&ctx, val_id);
        assert_eq!(cast.class.opcode, Op::Bitcast, "reinterpret load → bitcast");
        assert_eq!(cast.result_type, Some(float));
        let Operand::IdRef(load_id) = cast.operands[0] else {
            panic!("bitcast source is not an id");
        };
        assert_eq!(
            find_inst(&ctx, load_id).result_type,
            Some(uint),
            "the split load reads the uint element"
        );
    }

    /// The rewritten chain carries two indices (member-0 + dyn), so re-running the pass — which only
    /// matches exactly-one-index chains — changes nothing.
    #[test]
    fn dynamic_struct_index_reinterpret_is_idempotent() {
        let mut ctx = Ctx::new(Module::new());
        let uint = ctx.ty_uint();
        let float = ctx.ty_float();
        let rt = ctx.ty_runtime_array(uint);
        let struct_id = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_id),
            vec![Operand::IdRef(rt)],
        ));
        let ptr_struct = ctx.ty_ptr(StorageClass::StorageBuffer, struct_id);
        let ptr_float = ctx.ty_ptr(StorageClass::StorageBuffer, float);
        let base = storage_buffer_var(&mut ctx, ptr_struct);
        let dyn_idx = ctx.module.fresh_id();
        let chain_id = ctx.module.fresh_id();
        let val_id = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![
                Instruction::new(Op::Undef, Some(uint), Some(dyn_idx), vec![]),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_float),
                    Some(chain_id),
                    vec![Operand::IdRef(base), Operand::IdRef(dyn_idx)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(float),
                    Some(val_id),
                    vec![Operand::IdRef(chain_id)],
                ),
            ],
        );

        rewrite_dynamic_struct_index_reinterpret(&mut ctx, 0).unwrap();
        let after_first = ctx.module.functions[0].blocks[0].instructions.clone();
        rewrite_dynamic_struct_index_reinterpret(&mut ctx, 0).unwrap();
        assert_eq!(
            after_first, ctx.module.functions[0].blocks[0].instructions,
            "second application is a no-op"
        );
    }

    /// A CONSTANT struct index is legal SPIR-V, so the pass must decline it (the dynamic-index guard
    /// is what keeps the floor untouched).
    #[test]
    fn dynamic_struct_index_reinterpret_skips_constant_index() {
        let mut ctx = Ctx::new(Module::new());
        let uint = ctx.ty_uint();
        let float = ctx.ty_float();
        let rt = ctx.ty_runtime_array(uint);
        let struct_id = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_id),
            vec![Operand::IdRef(rt)],
        ));
        let ptr_struct = ctx.ty_ptr(StorageClass::StorageBuffer, struct_id);
        let ptr_float = ctx.ty_ptr(StorageClass::StorageBuffer, float);
        let base = storage_buffer_var(&mut ctx, ptr_struct);
        let const_idx = ctx.const_uint(3);
        let chain_id = ctx.module.fresh_id();
        let val_id = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_float),
                    Some(chain_id),
                    vec![Operand::IdRef(base), Operand::IdRef(const_idx)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(float),
                    Some(val_id),
                    vec![Operand::IdRef(chain_id)],
                ),
            ],
        );

        rewrite_dynamic_struct_index_reinterpret(&mut ctx, 0).unwrap();

        let chain = find_inst(&ctx, chain_id);
        assert_eq!(
            chain.operands.len(),
            2,
            "constant-index chain is left untouched"
        );
    }

    #[test]
    fn dynamic_struct_index_subword_reinterpret_extracts_half_from_word_lane() {
        let mut ctx = Ctx::new(Module::new());
        let uint = ctx.ty_uint();
        let half = ctx.ty_half();
        let rt = ctx.ty_runtime_array(uint);
        let struct_id = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_id),
            vec![Operand::IdRef(rt)],
        ));
        let ptr_struct = ctx.ty_ptr(StorageClass::StorageBuffer, struct_id);
        let ptr_half = ctx.ty_ptr(StorageClass::StorageBuffer, half);
        let base = storage_buffer_var(&mut ctx, ptr_struct);
        let dyn_idx = ctx.module.fresh_id();
        let chain_id = ctx.module.fresh_id();
        let val_id = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![
                Instruction::new(Op::Undef, Some(uint), Some(dyn_idx), vec![]),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_half),
                    Some(chain_id),
                    vec![Operand::IdRef(base), Operand::IdRef(dyn_idx)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(half),
                    Some(val_id),
                    vec![Operand::IdRef(chain_id)],
                ),
            ],
        );

        rewrite_dynamic_struct_index_subword_reinterpret(&mut ctx, 0).unwrap();

        let insts = &ctx.module.functions[0].blocks[0].instructions;
        assert!(
            !insts.iter().any(|inst| inst.result_id == Some(chain_id)),
            "the invalid half pointer should be eliminated"
        );
        let word_chain = insts
            .iter()
            .find(|inst| {
                matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain)
                    && inst.operands.first() == Some(&Operand::IdRef(base))
            })
            .expect("replacement chain into the backing word array");
        assert_eq!(word_chain.operands.len(), 3);
        let Some(Operand::IdRef(word_pointee)) = type_def_of(&ctx, word_chain.result_type.unwrap())
            .and_then(|d| d.operands.get(1).cloned())
        else {
            panic!("replacement chain result is not a pointer");
        };
        assert_eq!(word_pointee, uint, "replacement chain reads uint words");
        let cast = find_inst(&ctx, val_id);
        assert_eq!(cast.class.opcode, Op::Bitcast);
        assert_eq!(cast.result_type, Some(half));
        assert!(
            insts.iter().any(|inst| inst.class.opcode == Op::UDiv),
            "half element index is divided by two to address the backing word"
        );
        assert!(
            insts
                .iter()
                .any(|inst| inst.class.opcode == Op::ShiftRightLogical),
            "selected half lane is shifted down before truncation"
        );
    }

    #[test]
    fn dynamic_struct_index_subword_reinterpret_extracts_uchar_from_word_lane() {
        let mut ctx = Ctx::new(Module::new());
        let uint = ctx.ty_uint();
        let uchar = ctx.ty_int8();
        let rt = ctx.ty_runtime_array(uint);
        let struct_id = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_id),
            vec![Operand::IdRef(rt)],
        ));
        let ptr_struct = ctx.ty_ptr(StorageClass::StorageBuffer, struct_id);
        let ptr_uchar = ctx.ty_ptr(StorageClass::StorageBuffer, uchar);
        let base = storage_buffer_var(&mut ctx, ptr_struct);
        let dyn_idx = ctx.module.fresh_id();
        let chain_id = ctx.module.fresh_id();
        let val_id = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![
                Instruction::new(Op::Undef, Some(uint), Some(dyn_idx), vec![]),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_uchar),
                    Some(chain_id),
                    vec![Operand::IdRef(base), Operand::IdRef(dyn_idx)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(uchar),
                    Some(val_id),
                    vec![Operand::IdRef(chain_id)],
                ),
            ],
        );

        rewrite_dynamic_struct_index_subword_reinterpret(&mut ctx, 0).unwrap();

        let insts = &ctx.module.functions[0].blocks[0].instructions;
        assert!(
            !insts.iter().any(|inst| inst.result_id == Some(chain_id)),
            "the invalid uchar pointer should be eliminated"
        );
        let div = insts
            .iter()
            .find(|inst| inst.class.opcode == Op::UDiv)
            .expect("byte element index is divided by four to address the backing word");
        let Operand::IdRef(divisor) = div.operands[1] else {
            panic!("divisor is not an id")
        };
        assert_eq!(const_u32(&ctx, divisor), Some(4));
        let lane = insts
            .iter()
            .find(|inst| inst.class.opcode == Op::BitwiseAnd && inst.result_type == Some(uint))
            .expect("byte lane is masked out of the dynamic index");
        let Operand::IdRef(mask) = lane.operands[1] else {
            panic!("lane mask is not an id")
        };
        assert_eq!(const_u32(&ctx, mask), Some(3));
        assert_eq!(find_inst(&ctx, val_id).result_type, Some(uchar));
    }

    #[test]
    fn dynamic_struct_index_subword_reinterpret_packs_ushort_store_into_word_lane() {
        let mut ctx = Ctx::new(Module::new());
        let uint = ctx.ty_uint();
        let ushort = ctx.ty_int16();
        let rt = ctx.ty_runtime_array(uint);
        let struct_id = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_id),
            vec![Operand::IdRef(rt)],
        ));
        let ptr_struct = ctx.ty_ptr(StorageClass::StorageBuffer, struct_id);
        let ptr_ushort = ctx.ty_ptr(StorageClass::StorageBuffer, ushort);
        let base = storage_buffer_var(&mut ctx, ptr_struct);
        let dyn_idx = ctx.module.fresh_id();
        let object = ctx.module.fresh_id();
        let chain_id = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![
                Instruction::new(Op::Undef, Some(uint), Some(dyn_idx), vec![]),
                Instruction::new(Op::Undef, Some(ushort), Some(object), vec![]),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_ushort),
                    Some(chain_id),
                    vec![Operand::IdRef(base), Operand::IdRef(dyn_idx)],
                ),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(chain_id), Operand::IdRef(object)],
                ),
            ],
        );

        rewrite_dynamic_struct_index_subword_reinterpret(&mut ctx, 0).unwrap();

        let insts = &ctx.module.functions[0].blocks[0].instructions;
        assert!(
            !insts.iter().any(|inst| inst.result_id == Some(chain_id)),
            "the invalid ushort pointer should be eliminated"
        );
        assert!(
            insts.iter().any(|inst| inst.class.opcode == Op::Not),
            "store clears the selected 16-bit lane before OR-ing in new bits"
        );
        assert!(
            insts.iter().any(|inst| inst.class.opcode == Op::BitwiseOr),
            "store merges preserved word bits with shifted object bits"
        );
        let stores: Vec<&Instruction> = insts
            .iter()
            .filter(|inst| inst.class.opcode == Op::Store)
            .collect();
        assert_eq!(
            stores.len(),
            1,
            "the original halfword store becomes one word store"
        );
        let Operand::IdRef(stored) = stores[0].operands[1] else {
            panic!("store object is not an id");
        };
        assert_eq!(value_result_type(&ctx, stored), Some(uint));
    }

    #[test]
    fn dynamic_struct_index_wide_word_reinterpret_assembles_ulong_from_two_words() {
        let mut ctx = Ctx::new(Module::new());
        let uint = ctx.ty_uint();
        let ulong = ctx.ty_ulong();
        let rt = ctx.ty_runtime_array(uint);
        let struct_id = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_id),
            vec![Operand::IdRef(rt)],
        ));
        let ptr_struct = ctx.ty_ptr(StorageClass::StorageBuffer, struct_id);
        let ptr_ulong = ctx.ty_ptr(StorageClass::StorageBuffer, ulong);
        let base = storage_buffer_var(&mut ctx, ptr_struct);
        let dyn_idx = ctx.module.fresh_id();
        let chain_id = ctx.module.fresh_id();
        let val_id = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![
                Instruction::new(Op::Undef, Some(uint), Some(dyn_idx), vec![]),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_ulong),
                    Some(chain_id),
                    vec![Operand::IdRef(base), Operand::IdRef(dyn_idx)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(ulong),
                    Some(val_id),
                    vec![Operand::IdRef(chain_id)],
                ),
            ],
        );

        rewrite_dynamic_struct_index_wide_word_reinterpret(&mut ctx, 0).unwrap();

        let insts = &ctx.module.functions[0].blocks[0].instructions;
        assert!(
            !insts.iter().any(|inst| inst.result_id == Some(chain_id)),
            "the invalid ulong pointer should be eliminated"
        );
        let chains: Vec<&Instruction> = insts
            .iter()
            .filter(|inst| {
                matches!(inst.class.opcode, Op::InBoundsAccessChain | Op::AccessChain)
                    && inst.operands.first() == Some(&Operand::IdRef(base))
            })
            .collect();
        assert_eq!(chains.len(), 2, "one word pointer per half of the u64");
        assert!(
            insts
                .iter()
                .filter(|inst| inst.class.opcode == Op::Load && inst.result_type == Some(uint))
                .count()
                == 2,
            "wide load reads two uint words"
        );
        assert_eq!(find_inst(&ctx, val_id).result_type, Some(ulong));
        assert!(
            insts.iter().any(|inst| inst.class.opcode == Op::BitwiseOr),
            "low and high words are OR-assembled"
        );
    }

    #[test]
    fn dynamic_struct_index_wide_word_reinterpret_splits_ulong_store_into_two_words() {
        let mut ctx = Ctx::new(Module::new());
        let uint = ctx.ty_uint();
        let ulong = ctx.ty_ulong();
        let rt = ctx.ty_runtime_array(uint);
        let struct_id = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_id),
            vec![Operand::IdRef(rt)],
        ));
        let ptr_struct = ctx.ty_ptr(StorageClass::StorageBuffer, struct_id);
        let ptr_ulong = ctx.ty_ptr(StorageClass::StorageBuffer, ulong);
        let base = storage_buffer_var(&mut ctx, ptr_struct);
        let dyn_idx = ctx.module.fresh_id();
        let object = ctx.module.fresh_id();
        let chain_id = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![
                Instruction::new(Op::Undef, Some(uint), Some(dyn_idx), vec![]),
                Instruction::new(Op::Undef, Some(ulong), Some(object), vec![]),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_ulong),
                    Some(chain_id),
                    vec![Operand::IdRef(base), Operand::IdRef(dyn_idx)],
                ),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(chain_id), Operand::IdRef(object)],
                ),
            ],
        );

        rewrite_dynamic_struct_index_wide_word_reinterpret(&mut ctx, 0).unwrap();

        let insts = &ctx.module.functions[0].blocks[0].instructions;
        assert!(
            !insts.iter().any(|inst| inst.result_id == Some(chain_id)),
            "the invalid ulong pointer should be eliminated"
        );
        let stores: Vec<&Instruction> = insts
            .iter()
            .filter(|inst| inst.class.opcode == Op::Store)
            .collect();
        assert_eq!(stores.len(), 2, "wide store writes two uint words");
        for store in stores {
            let Operand::IdRef(stored) = store.operands[1] else {
                panic!("store object is not an id");
            };
            assert_eq!(value_result_type(&ctx, stored), Some(uint));
        }
        assert!(
            insts
                .iter()
                .any(|inst| inst.class.opcode == Op::ShiftRightLogical),
            "high word is extracted by shifting the 64-bit object down"
        );
    }

    #[test]
    fn dynamic_struct_index_vector_reinterpret_replays_scalar_lanes() {
        let mut ctx = Ctx::new(Module::new());
        let uint = ctx.ty_uint();
        let v2uint = ctx.ty_vec_uint(2);
        let runtime = ctx.ty_runtime_array(uint);
        let block = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::TypeStruct,
            None,
            Some(block),
            vec![Operand::IdRef(runtime)],
        ));
        let ptr_block = ctx.ty_ptr(StorageClass::StorageBuffer, block);
        let ptr_v2uint = ctx.ty_ptr(StorageClass::StorageBuffer, v2uint);
        let base = storage_buffer_var(&mut ctx, ptr_block);
        let dyn_idx = ctx.module.fresh_id();
        let chain = ctx.module.fresh_id();
        let value = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![
                Instruction::new(Op::Undef, Some(uint), Some(dyn_idx), vec![]),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_v2uint),
                    Some(chain),
                    vec![Operand::IdRef(base), Operand::IdRef(dyn_idx)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(v2uint),
                    Some(value),
                    vec![Operand::IdRef(chain)],
                ),
            ],
        );

        rewrite_dynamic_struct_index_vector_reinterpret(&mut ctx, 0).unwrap();

        let insts = &ctx.module.functions[0].blocks[0].instructions;
        assert!(
            !insts.iter().any(|inst| inst.result_id == Some(chain)),
            "the illegal vector pointer must be eliminated"
        );
        let mul = insts
            .iter()
            .find(|inst| inst.class.opcode == Op::IMul)
            .expect("vector index is scaled by lane count");
        assert_eq!(mul.operands.first(), Some(&Operand::IdRef(dyn_idx)));
        let Operand::IdRef(two) = mul.operands.get(1).expect("lane multiplier") else {
            panic!("lane multiplier is not an id")
        };
        assert_eq!(const_u32(&ctx, *two), Some(2));

        let chains: Vec<&Instruction> = insts
            .iter()
            .filter(|inst| inst.class.opcode == Op::InBoundsAccessChain)
            .collect();
        assert_eq!(chains.len(), 2, "one scalar pointer per vector lane");
        for chain in &chains {
            assert_eq!(chain.operands.first(), Some(&Operand::IdRef(base)));
            let Operand::IdRef(member0) = chain.operands.get(1).expect("member-0 index") else {
                panic!("member index is not an id")
            };
            assert_eq!(const_u32(&ctx, *member0), Some(0));
        }

        let rebuilt = find_inst(&ctx, value);
        assert_eq!(rebuilt.class.opcode, Op::CompositeConstruct);
        assert_eq!(rebuilt.result_type, Some(v2uint));
        assert_eq!(rebuilt.operands.len(), 2);
    }

    #[test]
    fn dynamic_homogeneous_function_struct_index_load_becomes_select() {
        let mut ctx = Ctx::new(Module::new());
        let v3u16 = ctx.ty_vec_u16(3);
        let inner_struct = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::TypeStruct,
            None,
            Some(inner_struct),
            vec![
                Operand::IdRef(v3u16),
                Operand::IdRef(v3u16),
                Operand::IdRef(v3u16),
            ],
        ));
        let struct_id = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_id),
            vec![Operand::IdRef(inner_struct)],
        ));
        let ptr_struct = ctx.ty_ptr(StorageClass::Function, struct_id);
        let ptr_v3u16 = ctx.ty_ptr(StorageClass::Function, v3u16);
        let uint = ctx.ty_uint();
        let zero = ctx.const_uint(0);
        let dyn_idx = ctx.module.fresh_id();
        let base = ctx.module.fresh_id();
        let chain = ctx.module.fresh_id();
        let loaded = ctx.module.fresh_id();

        install_entry(
            &mut ctx,
            vec![
                Instruction::new(
                    Op::Variable,
                    Some(ptr_struct),
                    Some(base),
                    vec![Operand::StorageClass(StorageClass::Function)],
                ),
                Instruction::new(
                    Op::Bitcast,
                    Some(uint),
                    Some(dyn_idx),
                    vec![Operand::IdRef(zero)],
                ),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_v3u16),
                    Some(chain),
                    vec![Operand::IdRef(base), Operand::IdRef(dyn_idx)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(v3u16),
                    Some(loaded),
                    vec![Operand::IdRef(chain)],
                ),
            ],
        );

        rewrite_dynamic_homogeneous_struct_index_load(&mut ctx, 0).unwrap();

        let insts = &ctx.module.functions[0].blocks[0].instructions;
        assert!(
            !insts.iter().any(|inst| inst.result_id == Some(chain)),
            "invalid dynamic pointer should be deleted"
        );
        assert!(
            insts
                .iter()
                .any(|inst| inst.class.opcode == Op::Select && inst.result_id == Some(loaded)),
            "load result should be produced by the select cascade"
        );
        assert_eq!(
            insts
                .iter()
                .filter(|inst| {
                    matches!(inst.class.opcode, Op::AccessChain | Op::InBoundsAccessChain)
                        && inst.result_type == Some(ptr_v3u16)
                })
                .count(),
            3,
            "one constant-member chain per homogeneous field"
        );
        for chain in insts.iter().filter(|inst| {
            matches!(inst.class.opcode, Op::AccessChain | Op::InBoundsAccessChain)
                && inst.result_type == Some(ptr_v3u16)
        }) {
            let Operand::IdRef(wrapper_member) = chain.operands.get(1).unwrap() else {
                panic!("wrapper member is not an id")
            };
            assert_eq!(const_u32(&ctx, *wrapper_member), Some(0));
        }
        assert!(
            insts.iter().any(|inst| {
                inst.class.opcode == Op::CompositeConstruct
                    && inst.result_type.and_then(|ty| {
                        type_def_of(&ctx, ty).and_then(|def| match def.operands.get(1) {
                            Some(Operand::LiteralBit32(lanes)) => Some(*lanes),
                            _ => None,
                        })
                    }) == Some(3)
            }),
            "vector selects need a v3bool condition"
        );
    }

    // ---- rewrite_chained_element_reinterpret --------------------------------------------------

    struct ChainedFixture {
        buf: Word,
        dyn_f: Word,
        dyn_u: Word,
        out_id: Word,
        val_id: Word,
        uint: Word,
        float: Word,
    }

    /// Build the nested-chain reinterpret shape: an inner element chain
    /// `%inner = AC %ptr_uint %buf [%uint_0, %dynF]` into `{ RuntimeArray<uint> }`, an outer chain
    /// `%out = AC %ptr_float %inner [%uint_0, %dynU]` (invalid — indexes the scalar element) that
    /// reinterprets `uint`→`float`, and a `float` load of `%out`.
    fn build_chained_reinterpret(ctx: &mut Ctx) -> ChainedFixture {
        let uint = ctx.ty_uint();
        let float = ctx.ty_float();
        let rt = ctx.ty_runtime_array(uint);
        let struct_id = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_id),
            vec![Operand::IdRef(rt)],
        ));
        let ptr_struct = ctx.ty_ptr(StorageClass::StorageBuffer, struct_id);
        let ptr_uint = ctx.ty_ptr(StorageClass::StorageBuffer, uint);
        let ptr_float = ctx.ty_ptr(StorageClass::StorageBuffer, float);
        let buf = storage_buffer_var(ctx, ptr_struct);
        let zero = ctx.const_uint(0);
        let dyn_f = ctx.module.fresh_id();
        let dyn_u = ctx.module.fresh_id();
        let inner_id = ctx.module.fresh_id();
        let out_id = ctx.module.fresh_id();
        let val_id = ctx.module.fresh_id();
        install_entry(
            ctx,
            vec![
                Instruction::new(Op::Undef, Some(uint), Some(dyn_f), vec![]),
                Instruction::new(Op::Undef, Some(uint), Some(dyn_u), vec![]),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_uint),
                    Some(inner_id),
                    vec![
                        Operand::IdRef(buf),
                        Operand::IdRef(zero),
                        Operand::IdRef(dyn_f),
                    ],
                ),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_float),
                    Some(out_id),
                    vec![
                        Operand::IdRef(inner_id),
                        Operand::IdRef(zero),
                        Operand::IdRef(dyn_u),
                    ],
                ),
                Instruction::new(
                    Op::Load,
                    Some(float),
                    Some(val_id),
                    vec![Operand::IdRef(out_id)],
                ),
            ],
        );
        ChainedFixture {
            buf,
            dyn_f,
            dyn_u,
            out_id,
            val_id,
            uint,
            float,
        }
    }

    /// The pass re-roots `%out` onto `%buf` with a summed index (`%sum = %dynF + %dynU`), retypes it
    /// to the `uint` element pointer, and splits the `float` load into a `uint` load + `OpBitcast`.
    #[test]
    fn chained_element_reinterpret_reroots_with_summed_index() {
        let mut ctx = Ctx::new(Module::new());
        let fx = build_chained_reinterpret(&mut ctx);

        rewrite_chained_element_reinterpret(&mut ctx, 0).unwrap();

        let out = find_inst(&ctx, fx.out_id);
        assert_eq!(out.class.opcode, Op::InBoundsAccessChain);
        assert_eq!(
            out.operands.len(),
            3,
            "outer chain keeps member-0 + summed index"
        );
        assert_eq!(
            out.operands[0],
            Operand::IdRef(fx.buf),
            "re-rooted onto the buffer"
        );
        let Operand::IdRef(member0) = out.operands[1] else {
            panic!("member index is not an id");
        };
        assert_eq!(const_u32(&ctx, member0), Some(0));
        let Operand::IdRef(sum) = out.operands[2] else {
            panic!("summed index is not an id");
        };
        let sum_inst = find_inst(&ctx, sum);
        assert_eq!(
            sum_inst.class.opcode,
            Op::IAdd,
            "index is the sum of the two dynamic indices"
        );
        assert_eq!(sum_inst.result_type, Some(fx.uint));
        assert!(
            sum_inst.operands.contains(&Operand::IdRef(fx.dyn_f))
                && sum_inst.operands.contains(&Operand::IdRef(fx.dyn_u)),
            "sum adds dynF and dynU"
        );

        let cast = find_inst(&ctx, fx.val_id);
        assert_eq!(cast.class.opcode, Op::Bitcast, "reinterpret load → bitcast");
        assert_eq!(cast.result_type, Some(fx.float));
        let Operand::IdRef(load_id) = cast.operands[0] else {
            panic!("bitcast source is not an id");
        };
        assert_eq!(
            find_inst(&ctx, load_id).result_type,
            Some(fx.uint),
            "split load reads uint"
        );
    }

    /// After re-rooting, `%out`'s base is the buffer variable (not a chain), so the nested-chain
    /// matcher no longer fires — the pass is a no-op on its own output.
    #[test]
    fn chained_element_reinterpret_is_idempotent() {
        let mut ctx = Ctx::new(Module::new());
        build_chained_reinterpret(&mut ctx);

        rewrite_chained_element_reinterpret(&mut ctx, 0).unwrap();
        let after_first = ctx.module.functions[0].blocks[0].instructions.clone();
        rewrite_chained_element_reinterpret(&mut ctx, 0).unwrap();
        assert_eq!(
            after_first, ctx.module.functions[0].blocks[0].instructions,
            "second application is a no-op"
        );
    }

    // ---- rewrite_byte_buffer_chained_reinterpret ----------------------------------------------

    /// A fresh `OpTypeInt 16 unsigned` in `new_globals` (Ctx exposes no u16 builder).
    fn ty_int16(ctx: &mut Ctx) -> Word {
        let id = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::TypeInt,
            None,
            Some(id),
            vec![Operand::LiteralBit32(16), Operand::LiteralBit32(0)],
        ));
        id
    }

    struct ByteBufFixture {
        out_id: Word,
        val_id: Word,
        u16: Word,
        u32: Word,
    }

    /// Widen shape: an inner element chain `%inner = AC %ptr_u16 %buf [%uint_0, %byteIdx]` into
    /// `{ RuntimeArray<u16> }`, an outer chain `%out = AC %ptr_u32 %inner %k` (2 operands — invalid,
    /// indexes the u16 scalar) that reinterprets to a WIDER `u32` (ratio 2), and a `u32` load of it.
    fn build_byte_buffer_widen(ctx: &mut Ctx) -> ByteBufFixture {
        let u16 = ty_int16(ctx);
        let u32 = ctx.ty_uint();
        let rt = ctx.ty_runtime_array(u16);
        let struct_id = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::TypeStruct,
            None,
            Some(struct_id),
            vec![Operand::IdRef(rt)],
        ));
        let ptr_struct = ctx.ty_ptr(StorageClass::StorageBuffer, struct_id);
        let ptr_u16 = ctx.ty_ptr(StorageClass::StorageBuffer, u16);
        let ptr_u32 = ctx.ty_ptr(StorageClass::StorageBuffer, u32);
        let buf = storage_buffer_var(ctx, ptr_struct);
        let zero = ctx.const_uint(0);
        let byte_idx = ctx.module.fresh_id();
        let k = ctx.module.fresh_id();
        let inner_id = ctx.module.fresh_id();
        let out_id = ctx.module.fresh_id();
        let val_id = ctx.module.fresh_id();
        install_entry(
            ctx,
            vec![
                Instruction::new(Op::Undef, Some(u32), Some(byte_idx), vec![]),
                Instruction::new(Op::Undef, Some(u32), Some(k), vec![]),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_u16),
                    Some(inner_id),
                    vec![
                        Operand::IdRef(buf),
                        Operand::IdRef(zero),
                        Operand::IdRef(byte_idx),
                    ],
                ),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_u32),
                    Some(out_id),
                    vec![Operand::IdRef(inner_id), Operand::IdRef(k)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(u32),
                    Some(val_id),
                    vec![Operand::IdRef(out_id)],
                ),
            ],
        );
        ByteBufFixture {
            out_id,
            val_id,
            u16,
            u32,
        }
    }

    /// The invalid widening chain is replaced by byte-offset arithmetic (`out_idx*ratio + byteIdx`)
    /// and the `u32` load expands into `ratio` little-endian `u16` slot loads that are zero-extended,
    /// shifted, and OR-assembled into the original result id.
    #[test]
    fn byte_buffer_chained_reinterpret_expands_widening_load_into_slots() {
        let mut ctx = Ctx::new(Module::new());
        let fx = build_byte_buffer_widen(&mut ctx);

        rewrite_byte_buffer_chained_reinterpret(&mut ctx, 0).unwrap();

        let block = &ctx.module.functions[0].blocks[0].instructions;
        // The widened value is now an OR-assembly of the narrow slots.
        let val = find_inst(&ctx, fx.val_id);
        assert_eq!(
            val.class.opcode,
            Op::BitwiseOr,
            "little-endian OR-assembled result"
        );
        assert_eq!(val.result_type, Some(fx.u32));
        // ratio=2 → exactly two u16 slot loads.
        let slot_loads = block
            .iter()
            .filter(|i| i.class.opcode == Op::Load && i.result_type == Some(fx.u16))
            .count();
        assert_eq!(slot_loads, 2, "two little-endian narrow slot loads");
        // The byte offset is computed as out_idx*ratio then + byteIdx.
        assert!(
            block.iter().any(|i| i.class.opcode == Op::IMul),
            "out_idx * ratio"
        );
        assert!(
            block.iter().any(|i| i.class.opcode == Op::IAdd),
            "byteIdx + out_idx*ratio"
        );
        // The original outer (invalid) chain id no longer exists.
        assert!(
            block.iter().all(|i| i.result_id != Some(fx.out_id)),
            "outer widening chain replaced"
        );
    }

    /// After expansion no 2-operand widening chain remains, so re-running the pass is a no-op.
    #[test]
    fn byte_buffer_chained_reinterpret_is_idempotent() {
        let mut ctx = Ctx::new(Module::new());
        build_byte_buffer_widen(&mut ctx);

        rewrite_byte_buffer_chained_reinterpret(&mut ctx, 0).unwrap();
        let after_first = ctx.module.functions[0].blocks[0].instructions.clone();
        rewrite_byte_buffer_chained_reinterpret(&mut ctx, 0).unwrap();
        assert_eq!(
            after_first, ctx.module.functions[0].blocks[0].instructions,
            "second application is a no-op"
        );
    }

    // ---- rewrite_raw_byte_pointer_wide_loads --------------------------------------------------

    /// A fresh unsigned-byte type in `new_globals` (the main Ctx builders intentionally expose only
    /// the common 16/32/64-bit scalar types).
    fn ty_uint8(ctx: &mut Ctx) -> Word {
        let id = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::TypeInt,
            None,
            Some(id),
            vec![
                Operand::LiteralBit32(RAW_BYTE_POINTER_ELEMENT_BITS),
                Operand::LiteralBit32(0),
            ],
        ));
        id
    }

    struct RawByteWideFixture {
        chain_id: Word,
        load_id: Word,
        byte: Word,
        v4float: Word,
        base_phi: Word,
        index: Word,
        ulong: Word,
    }

    /// Build the raw-byte wide-load form emitted for `getelementptr <4 x float>, ptr %raw, i64 i`.
    /// `%raw` is an `OpPhi` so the fixture covers the key case where the byte view is selected across
    /// descriptor arms rather than rooted directly at a buffer variable.  The predecessor labels are
    /// intentionally opaque here: this unit tests the local access rewrite, whose contract is the
    /// pointer type and use closure rather than CFG construction.
    fn build_raw_byte_phi_wide_load(ctx: &mut Ctx) -> RawByteWideFixture {
        let byte = ty_uint8(ctx);
        let ulong = ctx.ty_ulong();
        let v4float = ctx.ty_vecf(4);
        let ptr_byte = ctx.ty_ptr(StorageClass::StorageBuffer, byte);
        let ptr_v4float = ctx.ty_ptr(StorageClass::StorageBuffer, v4float);
        let left = ctx.module.fresh_id();
        let right = ctx.module.fresh_id();
        let base_phi = ctx.module.fresh_id();
        let pred_left = ctx.module.fresh_id();
        let pred_right = ctx.module.fresh_id();
        let index = ctx.module.fresh_id();
        let chain_id = ctx.module.fresh_id();
        let load_id = ctx.module.fresh_id();
        install_entry(
            ctx,
            vec![
                Instruction::new(Op::Undef, Some(ptr_byte), Some(left), vec![]),
                Instruction::new(Op::Undef, Some(ptr_byte), Some(right), vec![]),
                Instruction::new(
                    Op::Phi,
                    Some(ptr_byte),
                    Some(base_phi),
                    vec![
                        Operand::IdRef(left),
                        Operand::IdRef(pred_left),
                        Operand::IdRef(right),
                        Operand::IdRef(pred_right),
                    ],
                ),
                Instruction::new(Op::Undef, Some(ulong), Some(index), vec![]),
                Instruction::new(
                    Op::InBoundsAccessChain,
                    Some(ptr_v4float),
                    Some(chain_id),
                    vec![Operand::IdRef(base_phi), Operand::IdRef(index)],
                ),
                Instruction::new(
                    Op::Load,
                    Some(v4float),
                    Some(load_id),
                    vec![Operand::IdRef(chain_id)],
                ),
            ],
        );
        RawByteWideFixture {
            chain_id,
            load_id,
            byte,
            v4float,
            base_phi,
            index,
            ulong,
        }
    }

    /// The invalid vector access disappears; its original result id becomes a vector construct of
    /// four little-endian byte-assembled float components, and all generated byte accesses remain
    /// rooted at the raw pointer phi with a 64-bit index arithmetic path.
    #[test]
    fn raw_byte_pointer_wide_load_replays_vector_through_phi() {
        let mut ctx = Ctx::new(Module::new());
        let fx = build_raw_byte_phi_wide_load(&mut ctx);

        rewrite_raw_byte_pointer_wide_loads(&mut ctx, 0);

        let block = &ctx.module.functions[0].blocks[0].instructions;
        assert!(
            block.iter().all(|inst| inst.result_id != Some(fx.chain_id)),
            "the invalid wide pointer is removed"
        );
        let result = find_inst(&ctx, fx.load_id);
        assert_eq!(result.class.opcode, Op::CompositeConstruct);
        assert_eq!(result.result_type, Some(fx.v4float));
        assert_eq!(
            block
                .iter()
                .filter(|inst| inst.class.opcode == Op::PtrAccessChain)
                .count(),
            16,
            "four 32-bit lanes replay through sixteen byte pointers"
        );
        assert_eq!(
            block
                .iter()
                .filter(|inst| inst.class.opcode == Op::Load && inst.result_type == Some(fx.byte))
                .count(),
            16,
            "one byte load per byte of the float4"
        );
        assert!(
            block.iter().any(|inst| {
                inst.class.opcode == Op::IMul
                    && inst.result_type == Some(fx.ulong)
                    && inst.operands.contains(&Operand::IdRef(fx.index))
            }),
            "the original 64-bit GEP index scales the whole float4"
        );
        assert!(
            block
                .iter()
                .filter(|inst| inst.class.opcode == Op::PtrAccessChain)
                .all(|inst| { inst.operands.first() == Some(&Operand::IdRef(fx.base_phi)) }),
            "all byte accesses preserve the selected/phi raw base"
        );
    }

    /// Once the invalid pointer and its load have been replaced, no `AccessChain` candidate remains,
    /// so a second application is structurally a no-op.
    #[test]
    fn raw_byte_pointer_wide_load_is_idempotent() {
        let mut ctx = Ctx::new(Module::new());
        build_raw_byte_phi_wide_load(&mut ctx);

        rewrite_raw_byte_pointer_wide_loads(&mut ctx, 0);
        let after_first = ctx.module.functions[0].blocks[0].instructions.clone();
        rewrite_raw_byte_pointer_wide_loads(&mut ctx, 0);
        assert_eq!(after_first, ctx.module.functions[0].blocks[0].instructions);
    }

    /// A pointer escape means the pass cannot replace the pointer with a value replay.  Keep the
    /// original invalid chain intact rather than guessing how a later call/store/alias would use it.
    #[test]
    fn raw_byte_pointer_wide_load_declines_pointer_escape() {
        let mut ctx = Ctx::new(Module::new());
        let fx = build_raw_byte_phi_wide_load(&mut ctx);
        let pointer_ty = find_inst(&ctx, fx.chain_id)
            .result_type
            .expect("fixture chain has a pointer result type");
        let escape = ctx.module.fresh_id();
        let block = &mut ctx.module.functions[0].blocks[0].instructions;
        let ret = block.pop().expect("fixture ends in return");
        block.push(Instruction::new(
            Op::CopyObject,
            Some(pointer_ty),
            Some(escape),
            vec![Operand::IdRef(fx.chain_id)],
        ));
        block.push(ret);

        rewrite_raw_byte_pointer_wide_loads(&mut ctx, 0);

        assert_eq!(
            find_inst(&ctx, fx.chain_id).class.opcode,
            Op::InBoundsAccessChain,
            "a pointer escape disqualifies the entire byte replay"
        );
        assert_eq!(
            find_inst(&ctx, fx.load_id).class.opcode,
            Op::Load,
            "the original load remains paired with the untouched pointer"
        );
    }

    // ---- rewrite_reinterpret_scalar_loads -----------------------------------------------------

    /// Build a same-width reinterpret load: a StorageBuffer `float*` pointer read as `uint`
    /// (`%val = OpLoad %uint %ptr`, declared pointee `float`, same 32-bit width). Returns
    /// `(ptr, val_id, uint, float)`.
    fn build_samewidth_reinterpret_load(ctx: &mut Ctx) -> (Word, Word, Word, Word) {
        let float = ctx.ty_float();
        let uint = ctx.ty_uint();
        let ptr_float = ctx.ty_ptr(StorageClass::StorageBuffer, float);
        let ptr = storage_buffer_var(ctx, ptr_float);
        let val_id = ctx.module.fresh_id();
        install_entry(
            ctx,
            vec![Instruction::new(
                Op::Load,
                Some(uint),
                Some(val_id),
                vec![Operand::IdRef(ptr)],
            )],
        );
        (ptr, val_id, uint, float)
    }

    /// A same-width mismatched load (`uint` from a `float*`) is rewritten to a declared-type load
    /// (`%lo = OpLoad %float %ptr`) plus an `OpBitcast` that rebinds the original result id to `uint`.
    #[test]
    fn reinterpret_scalar_load_samewidth_becomes_typed_load_plus_bitcast() {
        let mut ctx = Ctx::new(Module::new());
        let (_ptr, val_id, uint, float) = build_samewidth_reinterpret_load(&mut ctx);

        rewrite_reinterpret_scalar_loads(&mut ctx, 0);

        let cast = find_inst(&ctx, val_id);
        assert_eq!(
            cast.class.opcode,
            Op::Bitcast,
            "same-width reinterpret → bitcast"
        );
        assert_eq!(
            cast.result_type,
            Some(uint),
            "result keeps the loaded (uint) type"
        );
        let Operand::IdRef(lo) = cast.operands[0] else {
            panic!("bitcast source is not an id");
        };
        let lo_inst = find_inst(&ctx, lo);
        assert_eq!(
            lo_inst.class.opcode,
            Op::Load,
            "slot is loaded in its declared type"
        );
        assert_eq!(
            lo_inst.result_type,
            Some(float),
            "the declared-type load reads float"
        );
    }

    /// After the rewrite the slot load reads its declared pointee type (a valid load), so the pass —
    /// which only matches loads whose result type differs from the pointee — is a no-op on re-run.
    #[test]
    fn reinterpret_scalar_load_is_idempotent() {
        let mut ctx = Ctx::new(Module::new());
        build_samewidth_reinterpret_load(&mut ctx);

        rewrite_reinterpret_scalar_loads(&mut ctx, 0);
        let after_first = ctx.module.functions[0].blocks[0].instructions.clone();
        rewrite_reinterpret_scalar_loads(&mut ctx, 0);
        assert_eq!(
            after_first, ctx.module.functions[0].blocks[0].instructions,
            "second application is a no-op"
        );
    }

    // ---- rewrite_scalar_pointer_arithmetic_access_chains --------------------------------------

    /// `OpInBoundsAccessChain %ptr_float %base %idx` where `%base` is ITSELF a `ptr_float` (scalar
    /// pointer arithmetic — the base already points at the scalar element and the index strides past
    /// it) in a storage class Vulkan allows as an `OpPtrAccessChain` base (StorageBuffer). The pass
    /// must flip the opcode to `OpPtrAccessChain` and leave result type / id / operands intact.
    #[test]
    fn scalar_pointer_arithmetic_flips_samewidth_chain_to_ptr_access_chain() {
        let mut ctx = Ctx::new(Module::new());
        let float = ctx.ty_float();
        let ptr_float = ctx.ty_ptr(StorageClass::StorageBuffer, float);
        let base = storage_buffer_var(&mut ctx, ptr_float);
        let idx = ctx.const_uint(3);
        let chain_id = ctx.module.fresh_id();
        let ops = vec![Operand::IdRef(base), Operand::IdRef(idx)];
        install_entry(
            &mut ctx,
            vec![Instruction::new(
                Op::InBoundsAccessChain,
                Some(ptr_float),
                Some(chain_id),
                ops.clone(),
            )],
        );

        rewrite_scalar_pointer_arithmetic_access_chains(&mut ctx, 0);

        let inst = only_inst(&ctx);
        assert_eq!(inst.class.opcode, Op::PtrAccessChain, "opcode flipped");
        assert_eq!(inst.result_type, Some(ptr_float), "result type preserved");
        assert_eq!(inst.result_id, Some(chain_id), "result id preserved");
        assert_eq!(inst.operands, ops, "base + index operands unchanged");
    }

    /// Running the pass on its own output is a no-op: the flipped `OpPtrAccessChain` no longer matches
    /// the `InBoundsAccessChain` gate.
    #[test]
    fn scalar_pointer_arithmetic_is_idempotent() {
        let mut ctx = Ctx::new(Module::new());
        let float = ctx.ty_float();
        let ptr_float = ctx.ty_ptr(StorageClass::StorageBuffer, float);
        let base = storage_buffer_var(&mut ctx, ptr_float);
        let idx = ctx.const_uint(3);
        let chain_id = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![Instruction::new(
                Op::InBoundsAccessChain,
                Some(ptr_float),
                Some(chain_id),
                vec![Operand::IdRef(base), Operand::IdRef(idx)],
            )],
        );

        rewrite_scalar_pointer_arithmetic_access_chains(&mut ctx, 0);
        let after_first = ctx.module.functions[0].blocks[0].instructions.clone();
        rewrite_scalar_pointer_arithmetic_access_chains(&mut ctx, 0);
        assert_eq!(
            after_first, ctx.module.functions[0].blocks[0].instructions,
            "second application leaves the block byte-identical"
        );
    }

    /// A genuine composite descent (`%base` is a `ptr` to `RuntimeArray<float>`, result is `ptr_float`)
    /// — the base type differs from the chain result type, so it is NOT scalar pointer arithmetic. The
    /// pass declines it (the floor guard: a legal aggregate-indexing chain is never disturbed).
    #[test]
    fn scalar_pointer_arithmetic_leaves_composite_descent_untouched() {
        let mut ctx = Ctx::new(Module::new());
        let float = ctx.ty_float();
        let rt = ctx.ty_runtime_array(float);
        let ptr_rt = ctx.ty_ptr(StorageClass::StorageBuffer, rt);
        let ptr_float = ctx.ty_ptr(StorageClass::StorageBuffer, float);
        let base = storage_buffer_var(&mut ctx, ptr_rt);
        let idx = ctx.const_uint(0);
        let chain_id = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![Instruction::new(
                Op::InBoundsAccessChain,
                Some(ptr_float),
                Some(chain_id),
                vec![Operand::IdRef(base), Operand::IdRef(idx)],
            )],
        );

        rewrite_scalar_pointer_arithmetic_access_chains(&mut ctx, 0);

        assert_eq!(
            only_inst(&ctx).class.opcode,
            Op::InBoundsAccessChain,
            "an aggregate-indexing chain (base type != result type) is not rewritten"
        );
    }

    /// Same scalar-pointer-arithmetic shape but in a storage class `OpPtrAccessChain` does NOT allow as
    /// a base (Private). The storage gate declines it — a Private scalar chain must be composed back to
    /// an aggregate root, not turned into pointer arithmetic (the floor guard for the storage check).
    #[test]
    fn scalar_pointer_arithmetic_declines_private_storage() {
        let mut ctx = Ctx::new(Module::new());
        let float = ctx.ty_float();
        let ptr_float_priv = ctx.ty_ptr(StorageClass::Private, float);
        let base = ctx.module.fresh_id();
        ctx.new_globals.push(Instruction::new(
            Op::Variable,
            Some(ptr_float_priv),
            Some(base),
            vec![Operand::StorageClass(StorageClass::Private)],
        ));
        let idx = ctx.const_uint(3);
        let chain_id = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![Instruction::new(
                Op::InBoundsAccessChain,
                Some(ptr_float_priv),
                Some(chain_id),
                vec![Operand::IdRef(base), Operand::IdRef(idx)],
            )],
        );

        rewrite_scalar_pointer_arithmetic_access_chains(&mut ctx, 0);

        assert_eq!(
            only_inst(&ctx).class.opcode,
            Op::InBoundsAccessChain,
            "a Private-storage scalar chain is not turned into OpPtrAccessChain"
        );
    }
}
