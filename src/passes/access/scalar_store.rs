//! Scalar `OpStore` type repair for direct scalar pointee/object mismatches.

use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum DirectScalar {
    Int { width: u32 },
    Float { width: u32 },
}

impl DirectScalar {
    fn width(self) -> u32 {
        match self {
            DirectScalar::Int { width } | DirectScalar::Float { width } => width,
        }
    }
}

fn direct_scalar(ctx: &Ctx, ty: Word) -> Option<DirectScalar> {
    let def = type_def_of(ctx, ty)?;
    let width = match def.operands.first() {
        Some(Operand::LiteralBit32(width)) => *width,
        _ => return None,
    };
    match def.class.opcode {
        Op::TypeInt => Some(DirectScalar::Int { width }),
        Op::TypeFloat => Some(DirectScalar::Float { width }),
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum StoreCoercion {
    Bitcast,
    FConvert,
}

/// Normalize an invalid scalar store by inserting a value conversion to the pointer's declared
/// pointee type immediately before the store.
///
/// This targets the FC-promoted Logical fallback shape where the native emitter extracts a 32-bit
/// float lane from a vector and stores it through a `half*` StorageBuffer pointer, plus the sibling
/// same-width bit reinterpret shape where a `ushort` lane is stored through `half*`. SPIR-V requires
/// the stored object's type to match the pointer pointee exactly.
///
/// Floor-safe gates:
/// - the store must already be invalid (`object_ty != pointee_ty`);
/// - both sides must be direct scalar numeric types;
/// - same-width mismatches use `OpBitcast` (bit-preserving reinterpret);
/// - differing-width float/float mismatches use `OpFConvert` (semantic float width conversion);
/// - integer and byte/subword stores are deliberately ignored so `lower_subword_scalar_store` keeps
///   owning packed little-endian reinterpret semantics.
pub(in crate::passes) fn normalize_scalar_store_types(ctx: &mut Ctx, entry_idx: usize) {
    let mut ptr_pointees: HashMap<Word, Word> = HashMap::new();
    for inst in ctx
        .new_globals
        .iter()
        .chain(ctx.module.types_global_values.iter())
    {
        if inst.class.opcode != Op::TypePointer {
            continue;
        }
        let (Some(ptr_ty), Some(Operand::IdRef(pointee))) = (inst.result_id, inst.operands.get(1))
        else {
            continue;
        };
        ptr_pointees.insert(ptr_ty, *pointee);
    }

    let func = &ctx.module.functions[entry_idx];
    let mut plans: HashMap<(usize, usize), (Word, StoreCoercion)> = HashMap::new();
    for (bi, block) in func.blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if inst.class.opcode != Op::Store {
                continue;
            }
            let (Some(ptr), Some(obj)) = (operand_id(inst, 0), operand_id(inst, 1)) else {
                continue;
            };
            let Some(ptr_ty) = value_result_type(ctx, ptr) else {
                continue;
            };
            let Some(&pointee_ty) = ptr_pointees.get(&ptr_ty) else {
                continue;
            };
            let Some(obj_ty) = value_result_type(ctx, obj) else {
                continue;
            };
            if obj_ty == pointee_ty {
                continue;
            }
            let (Some(obj_scalar), Some(pointee_scalar)) =
                (direct_scalar(ctx, obj_ty), direct_scalar(ctx, pointee_ty))
            else {
                continue;
            };
            let coercion = if obj_scalar.width() == pointee_scalar.width() {
                StoreCoercion::Bitcast
            } else if matches!(
                (obj_scalar, pointee_scalar),
                (DirectScalar::Float { .. }, DirectScalar::Float { .. })
            ) {
                StoreCoercion::FConvert
            } else {
                continue;
            };
            plans.insert((bi, ii), (pointee_ty, coercion));
        }
    }

    if plans.is_empty() {
        return;
    }
    let plans = plans
        .into_iter()
        .map(|(site, (pointee_ty, coercion))| (site, (pointee_ty, coercion, ctx.module.fresh_id())))
        .collect::<HashMap<_, _>>();

    for (bi, block) in ctx.module.functions[entry_idx]
        .blocks
        .iter_mut()
        .enumerate()
    {
        let insts = std::mem::take(&mut block.instructions);
        let mut out = Vec::with_capacity(insts.len() + plans.len());
        for (ii, mut inst) in insts.into_iter().enumerate() {
            if let Some(&(pointee_ty, coercion, converted)) = plans.get(&(bi, ii)) {
                let obj = operand_id(&inst, 1).expect("planned OpStore lost object operand");
                out.push(Instruction::new(
                    match coercion {
                        StoreCoercion::Bitcast => Op::Bitcast,
                        StoreCoercion::FConvert => Op::FConvert,
                    },
                    Some(pointee_ty),
                    Some(converted),
                    vec![Operand::IdRef(obj)],
                ));
                inst.operands[1] = Operand::IdRef(converted);
            }
            out.push(inst);
        }
        block.instructions = out;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spirv_module::{Block, Function};

    fn install_entry(ctx: &mut Ctx, body: Vec<Instruction>) {
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

    #[test]
    fn normalizes_float_store_width_to_pointee_type() {
        let mut ctx = Ctx::new(Module::new());
        let half = ctx.ty_half();
        let float = ctx.ty_float();
        let ptr_half = ctx.ty_ptr(StorageClass::StorageBuffer, half);
        let ptr = storage_buffer_var(&mut ctx, ptr_half);
        let value = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![
                Instruction::new(Op::Undef, Some(float), Some(value), vec![]),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(ptr), Operand::IdRef(value)],
                ),
                Instruction::new(Op::Return, None, None, vec![]),
            ],
        );

        normalize_scalar_store_types(&mut ctx, 0);

        let insts = &ctx.module.functions[0].blocks[0].instructions;
        assert_eq!(insts[1].class.opcode, Op::FConvert);
        assert_eq!(insts[1].result_type, Some(half));
        assert_eq!(insts[1].operands, vec![Operand::IdRef(value)]);
        let converted = insts[1].result_id.expect("convert result id");
        assert_eq!(insts[2].class.opcode, Op::Store);
        assert_eq!(insts[2].operands[0], Operand::IdRef(ptr));
        assert_eq!(insts[2].operands[1], Operand::IdRef(converted));
    }

    #[test]
    fn bitcasts_same_width_scalar_store_to_pointee_type() {
        let mut ctx = Ctx::new(Module::new());
        let half = ctx.ty_half();
        let ushort = ctx.ty_int16();
        let ptr_half = ctx.ty_ptr(StorageClass::StorageBuffer, half);
        let ptr = storage_buffer_var(&mut ctx, ptr_half);
        let value = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![
                Instruction::new(Op::Undef, Some(ushort), Some(value), vec![]),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(ptr), Operand::IdRef(value)],
                ),
                Instruction::new(Op::Return, None, None, vec![]),
            ],
        );

        normalize_scalar_store_types(&mut ctx, 0);

        let insts = &ctx.module.functions[0].blocks[0].instructions;
        assert_eq!(insts[1].class.opcode, Op::Bitcast);
        assert_eq!(insts[1].result_type, Some(half));
        let converted = insts[1].result_id.expect("bitcast result id");
        assert_eq!(insts[2].class.opcode, Op::Store);
        assert_eq!(insts[2].operands[1], Operand::IdRef(converted));
    }

    #[test]
    fn leaves_matching_and_wide_integer_stores_untouched() {
        let mut ctx = Ctx::new(Module::new());
        let half = ctx.ty_half();
        let uint = ctx.ty_uint();
        let ptr_half = ctx.ty_ptr(StorageClass::StorageBuffer, half);
        let ptr_uint = ctx.ty_ptr(StorageClass::StorageBuffer, uint);
        let half_ptr = storage_buffer_var(&mut ctx, ptr_half);
        let uint_ptr = storage_buffer_var(&mut ctx, ptr_uint);
        let half_value = ctx.module.fresh_id();
        let uint_value = ctx.module.fresh_id();
        install_entry(
            &mut ctx,
            vec![
                Instruction::new(Op::Undef, Some(half), Some(half_value), vec![]),
                Instruction::new(Op::Undef, Some(uint), Some(uint_value), vec![]),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(half_ptr), Operand::IdRef(half_value)],
                ),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(half_ptr), Operand::IdRef(uint_value)],
                ),
                Instruction::new(
                    Op::Store,
                    None,
                    None,
                    vec![Operand::IdRef(uint_ptr), Operand::IdRef(half_value)],
                ),
                Instruction::new(Op::Return, None, None, vec![]),
            ],
        );
        let before = ctx.module.functions[0].blocks[0].instructions.clone();

        normalize_scalar_store_types(&mut ctx, 0);

        assert_eq!(
            ctx.module.functions[0].blocks[0].instructions, before,
            "matched stores and non-float mismatches stay on existing repair paths"
        );
    }
}
